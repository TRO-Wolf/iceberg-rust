// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use parquet::arrow::arrow_reader::RowSelection;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::basic::Compression;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use super::reader::{
    ArrowReader, ArrowReaderBuilder, build_fallback_field_id_map, build_field_id_map,
};
use crate::Result;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::io::{
    FileIO, FileIOBuilder, FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorage,
    OutputFile, Storage, StorageConfig, StorageFactory,
};
use crate::puffin::PuffinReader;
use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
use crate::spec::{
    DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema, SchemaRef, Type,
};
use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

pub(crate) const PAGE_ROWS: usize = 64;
pub(crate) const NUM_PAGES: usize = 8;
pub(crate) const ROWS: usize = PAGE_ROWS * NUM_PAGES;

pub(crate) fn field(name: &str, data_type: DataType, nullable: bool, field_id: i32) -> Field {
    Field::new(name, data_type, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        field_id.to_string(),
    )]))
}

pub(crate) fn iceberg_schema(fields: Vec<NestedField>) -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(fields.into_iter().map(Into::into))
            .build()
            .expect("schema"),
    )
}

pub(crate) fn id_schema() -> SchemaRef {
    iceberg_schema(vec![NestedField::required(
        1,
        "id",
        Type::Primitive(PrimitiveType::Int),
    )])
}

pub(crate) fn id_s_schema() -> SchemaRef {
    iceberg_schema(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "s", Type::Primitive(PrimitiveType::String)),
    ])
}

pub(crate) fn page_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .build()
}

pub(crate) fn write_parquet(
    path: &str,
    arrow_schema: Arc<ArrowSchema>,
    batches: &[RecordBatch],
    props: WriterProperties,
) {
    let file = File::create(path).expect("create");
    let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).expect("writer");
    for batch in batches {
        writer.write(batch).expect("write");
    }
    writer.close().expect("close");
}

pub(crate) fn write_id_pages(path: &str, ids: &[i32]) {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "id",
        DataType::Int32,
        false,
        1,
    )]));
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
        ])
        .expect("batch");
    write_parquet(path, arrow_schema, &[batch], page_props());
}

pub(crate) fn write_id_s_pages(path: &str, ids: &[i32], strings: &[Option<String>]) {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, false, 1),
        field("s", DataType::Utf8, true, 2),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(strings.to_vec())) as ArrayRef,
    ])
    .expect("batch");
    write_parquet(path, arrow_schema, &[batch], page_props());
}

pub(crate) fn file_metadata(path: &str) -> Arc<ParquetMetaData> {
    let file = File::open(path).expect("open");
    let metadata = ParquetMetaDataReader::new()
        .with_page_index_policy(PageIndexPolicy::Optional)
        .parse_and_finish(&file)
        .expect("metadata");
    Arc::new(metadata)
}

pub(crate) fn assert_page_count(
    metadata: &ParquetMetaData,
    row_group: usize,
    column: usize,
    min: usize,
) {
    let pages = metadata.offset_index().expect("offset index")[row_group][column]
        .page_locations()
        .len();
    assert!(
        pages >= min,
        "fixture must produce >= {min} pages, got {pages}"
    );
}

pub(crate) fn task(
    path: &str,
    schema: SchemaRef,
    project_ids: &[i32],
    predicate: Option<BoundPredicate>,
) -> FileScanTask {
    FileScanTask {
        file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
        start: 0,
        length: 0,
        record_count: None,
        file_record_count: None,
        data_file_path: Arc::from(path.to_string()),
        data_file_format: DataFileFormat::Parquet,
        schema,
        project_field_ids: Arc::from(project_ids.to_vec()),
        predicate: predicate.map(Arc::new),
        deletes: Arc::from(vec![]),
        partition: None,
        partition_spec: None,
        name_mapping: None,
        case_sensitive: false,
        split_offsets: None,
        first_row_id: None,
        file_sequence_number: None,
    }
}

pub(crate) fn bound(schema: &SchemaRef, predicate: Predicate) -> BoundPredicate {
    predicate.bind(schema.clone(), false).expect("bind")
}

pub(crate) async fn collect(task: FileScanTask, row_selection: bool) -> Vec<RecordBatch> {
    let reader = ArrowReaderBuilder::new(FileIO::new_with_fs())
        .with_batch_size(37)
        .with_row_group_filtering_enabled(true)
        .with_row_selection_enabled(row_selection)
        .build();
    reader
        .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
        .expect("read")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect")
}

pub(crate) async fn collect_with_io(
    task: FileScanTask,
    row_selection: bool,
    file_io: FileIO,
    metadata_size_hint: usize,
) -> Vec<RecordBatch> {
    let reader = ArrowReaderBuilder::new(file_io)
        .with_batch_size(37)
        .with_row_group_filtering_enabled(true)
        .with_row_selection_enabled(row_selection)
        .with_metadata_size_hint(metadata_size_hint)
        .build();
    reader
        .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
        .expect("read")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect")
}

pub(crate) fn dump(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            rows.push(
                (0..batch.num_columns())
                    .map(|col| {
                        arrow_cast::display::array_value_to_string(batch.column(col).as_ref(), row)
                            .expect("value")
                    })
                    .collect(),
            );
        }
    }
    rows
}

pub(crate) async fn on_off(task: FileScanTask) -> Vec<Vec<String>> {
    let on = collect(task.clone(), true).await;
    let off = collect(task, false).await;
    let on_rows = dump(&on);
    assert_eq!(
        on_rows,
        dump(&off),
        "row-selection ON and OFF must return identical rows"
    );
    if let (Some(first_on), Some(first_off)) = (on.first(), off.first()) {
        assert_eq!(first_on.schema(), first_off.schema());
    }
    on_rows
}

pub(crate) fn field_id_map(metadata: &ParquetMetaData) -> HashMap<i32, usize> {
    let schema_descr = metadata.file_metadata().schema_descr();
    build_field_id_map(schema_descr)
        .expect("field id map")
        .unwrap_or_else(|| build_fallback_field_id_map(schema_descr))
}

pub(crate) fn page_selection(
    metadata: &Arc<ParquetMetaData>,
    schema: &Schema,
    predicate: &BoundPredicate,
    selected_row_groups: &Option<Vec<usize>>,
) -> Option<RowSelection> {
    let map = field_id_map(metadata);
    ArrowReader::get_row_selection_for_filter_predicate(
        predicate,
        metadata,
        selected_row_groups,
        &map,
        schema,
    )
    .ok()
    .flatten()
}

pub(crate) fn assert_prunes(selection: &RowSelection) {
    let selectors: Vec<_> = selection.iter().collect();
    assert!(
        selectors.iter().any(|s| s.skip),
        "selection must skip at least one page"
    );
    assert!(
        selectors.iter().any(|s| !s.skip),
        "selection must keep at least one page"
    );
}

pub(crate) fn selected_rows(selection: &RowSelection) -> usize {
    selection
        .iter()
        .filter(|s| !s.skip)
        .map(|s| s.row_count)
        .sum()
}

pub(crate) fn write_pos_delete_file(
    path: &str,
    data_path: &str,
    positions: &[i64],
) -> FileScanTaskDeleteFile {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("file_path", DataType::Utf8, false, i32::MAX - 101),
        field("pos", DataType::Int64, false, i32::MAX - 102),
    ]));
    let paths: Vec<&str> = positions.iter().map(|_| data_path).collect();
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions.to_vec())) as ArrayRef,
    ])
    .expect("batch");
    write_parquet(
        path,
        arrow_schema,
        &[batch],
        WriterProperties::builder().build(),
    );
    FileScanTaskDeleteFile {
        file_path: path.to_string(),
        file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        file_format: DataFileFormat::Parquet,
        referenced_data_file: None,
        content_offset: None,
        content_size_in_bytes: None,
        record_count: None,
    }
}

pub(crate) fn write_eq_delete_file(
    path: &str,
    keys: &[Option<i32>],
    equality_ids: Vec<i32>,
) -> FileScanTaskDeleteFile {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "id",
        DataType::Int32,
        true,
        *equality_ids.first().expect("equality id"),
    )]));
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(keys.to_vec())) as ArrayRef,
        ])
        .expect("batch");
    write_parquet(
        path,
        arrow_schema,
        &[batch],
        WriterProperties::builder().build(),
    );
    FileScanTaskDeleteFile {
        file_path: path.to_string(),
        file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
        file_type: DataContentType::EqualityDeletes,
        partition_spec_id: 0,
        equality_ids: Some(equality_ids),
        file_format: DataFileFormat::Parquet,
        referenced_data_file: None,
        content_offset: None,
        content_size_in_bytes: None,
        record_count: None,
    }
}

pub(crate) async fn write_dv_delete_file(
    file_io: &FileIO,
    path: &str,
    data_path: &str,
    positions: &[u64],
) -> FileScanTaskDeleteFile {
    let output = file_io.new_output(path).expect("dv output");
    let mut writer = DVFileWriter::new(output).unpartitioned();
    for position in positions {
        writer
            .delete(data_path, *position, None)
            .expect("dv delete");
    }
    writer.close().await.expect("dv close");
    let input = file_io.new_input(path).expect("dv input");
    let puffin_reader = PuffinReader::new(input);
    let footer = puffin_reader.file_metadata().await.expect("dv footer");
    let blob = footer.blobs().first().expect("dv blob");
    FileScanTaskDeleteFile {
        file_path: path.to_string(),
        file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        file_format: DataFileFormat::Puffin,
        referenced_data_file: Some(data_path.to_string()),
        content_offset: Some(i64::try_from(blob.offset()).expect("offset")),
        content_size_in_bytes: Some(i64::try_from(blob.length()).expect("length")),
        record_count: Some(positions.len() as u64),
    }
}

pub(crate) fn with_deletes(
    mut task: FileScanTask,
    deletes: Vec<FileScanTaskDeleteFile>,
) -> FileScanTask {
    task.deletes = Arc::from(deletes);
    task
}

pub(crate) fn i64_column(batches: &[RecordBatch], column: usize) -> Vec<Option<i64>> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch.column(column);
        for row in 0..batch.num_rows() {
            values.push(
                arrow_cast::display::array_value_to_string(array.as_ref(), row)
                    .ok()
                    .and_then(|v| v.parse::<i64>().ok()),
            );
        }
    }
    values
}

pub(crate) fn tmpdir() -> TempDir {
    TempDir::new().expect("tempdir")
}

pub(crate) fn path(tmp: &TempDir, name: &str) -> String {
    tmp.path().join(name).to_string_lossy().to_string()
}

type ReadRanges = Arc<Mutex<Vec<(String, Range<u64>)>>>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RecordingStorage {
    #[serde(skip)]
    new_input_calls: Arc<AtomicUsize>,
    #[serde(skip)]
    read_ranges: ReadRanges,
}

struct RecordingFileRead {
    inner: Box<dyn FileRead>,
    path: String,
    read_ranges: ReadRanges,
}

#[async_trait]
impl FileRead for RecordingFileRead {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        self.read_ranges
            .lock()
            .expect("read ranges")
            .push((self.path.clone(), range.clone()));
        self.inner.read(range).await
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for RecordingStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        LocalFsStorage::new().exists(path).await
    }
    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        LocalFsStorage::new().metadata(path).await
    }
    async fn read(&self, path: &str) -> Result<Bytes> {
        LocalFsStorage::new().read(path).await
    }
    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        let inner = LocalFsStorage::new().reader(path).await?;
        Ok(Box::new(RecordingFileRead {
            inner,
            path: path.to_string(),
            read_ranges: self.read_ranges.clone(),
        }))
    }
    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        LocalFsStorage::new().write(path, bs).await
    }
    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        LocalFsStorage::new().writer(path).await
    }
    async fn delete(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete(path).await
    }
    async fn delete_prefix(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete_prefix(path).await
    }
    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        LocalFsStorage::new().list(prefix).await
    }
    fn new_input(&self, path: &str) -> Result<InputFile> {
        self.new_input_calls.fetch_add(1, Ordering::Relaxed);
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }
    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RecordingStorageFactory {
    #[serde(skip)]
    new_input_calls: Arc<AtomicUsize>,
    #[serde(skip)]
    read_ranges: ReadRanges,
}

#[typetag::serde]
impl StorageFactory for RecordingStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(RecordingStorage {
            new_input_calls: self.new_input_calls.clone(),
            read_ranges: self.read_ranges.clone(),
        }))
    }
}

pub(crate) fn recording_io() -> (FileIO, Arc<AtomicUsize>, ReadRanges) {
    let new_input_calls = Arc::new(AtomicUsize::new(0));
    let read_ranges = Arc::new(Mutex::new(Vec::new()));
    let io = FileIOBuilder::new(Arc::new(RecordingStorageFactory {
        new_input_calls: new_input_calls.clone(),
        read_ranges: read_ranges.clone(),
    }))
    .build();
    (io, new_input_calls, read_ranges)
}

fn byte_range(offset: Option<i64>, length: Option<i32>) -> Option<Range<u64>> {
    let start = u64::try_from(offset?).ok()?;
    let length = u64::try_from(length?).ok()?;
    Some(start..(start + length))
}

pub(crate) fn index_byte_ranges(metadata: &ParquetMetaData) -> Vec<Range<u64>> {
    let mut ranges = Vec::new();
    for group in 0..metadata.num_row_groups() {
        for column in metadata.row_group(group).columns() {
            if let Some(range) =
                byte_range(column.column_index_offset(), column.column_index_length())
            {
                ranges.push(range);
            }
            if let Some(range) =
                byte_range(column.offset_index_offset(), column.offset_index_length())
            {
                ranges.push(range);
            }
        }
    }
    ranges
}

pub(crate) fn any_read_intersects(
    read_ranges: &Mutex<Vec<(String, Range<u64>)>>,
    path: &str,
    ranges: &[Range<u64>],
) -> bool {
    read_ranges
        .lock()
        .expect("read ranges")
        .iter()
        .any(|(p, read)| {
            p == path
                && ranges
                    .iter()
                    .any(|r| read.start < r.end && r.start < read.end)
        })
}
