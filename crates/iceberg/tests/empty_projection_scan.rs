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
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::arrow::ArrowReaderBuilder;
use iceberg::expr::{Bind, Reference};
use iceberg::io::FileIO;
use iceberg::puffin::PuffinReader;
use iceberg::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
use iceberg::spec::{
    DataContentType, DataFileFormat, Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type,
};
use iceberg::writer::base_writer::deletion_vector_writer::DVFileWriter;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

fn id_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .expect("schema"),
    )
}

fn write_id_parquet(path: &str, ids: &[i32]) {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
    ]));
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
        ])
        .expect("batch");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(30))
        .build();
    let file = File::create(path).expect("create data");
    let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
}

fn empty_task(
    data_path: &str,
    schema: SchemaRef,
    deletes: Vec<FileScanTaskDeleteFile>,
    predicate: Option<iceberg::expr::BoundPredicate>,
) -> FileScanTask {
    FileScanTask {
        file_size_in_bytes: std::fs::metadata(data_path).expect("stat").len(),
        start: 0,
        length: 0,
        record_count: None,
        data_file_path: Arc::from(data_path.to_string()),
        data_file_format: DataFileFormat::Parquet,
        schema,
        project_field_ids: Arc::from(vec![]),
        predicate: predicate.map(Arc::new),
        deletes: Arc::from(deletes),
        partition: None,
        partition_spec: None,
        name_mapping: None,
        case_sensitive: false,
        split_offsets: None,
        first_row_id: None,
        file_sequence_number: None,
    }
}

async fn run_empty_scan(
    task: FileScanTask,
    batch_size: Option<usize>,
    row_selection_enabled: bool,
) -> Vec<RecordBatch> {
    let file_io = FileIO::new_with_fs();
    let mut builder = ArrowReaderBuilder::new(file_io);
    if let Some(size) = batch_size {
        builder = builder.with_batch_size(size);
    }
    let reader = builder
        .with_row_group_filtering_enabled(true)
        .with_row_selection_enabled(row_selection_enabled)
        .build();
    reader
        .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
        .expect("read")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect")
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

fn write_pos_delete_file(
    path: &str,
    referenced_data_path: &str,
    positions: &[i64],
) -> FileScanTaskDeleteFile {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            (i32::MAX - 101).to_string(),
        )])),
        Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            (i32::MAX - 102).to_string(),
        )])),
    ]));
    let paths: Vec<&str> = positions.iter().map(|_| referenced_data_path).collect();
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions.to_vec())) as ArrayRef,
    ])
    .expect("delete batch");
    let file = File::create(path).expect("create delete file");
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
        .expect("delete writer");
    writer.write(&batch).expect("write deletes");
    writer.close().expect("close delete writer");
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

fn write_eq_delete_file(path: &str, keys: &[i32]) -> FileScanTaskDeleteFile {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(Int32Array::from(keys.to_vec())) as ArrayRef,
    ])
    .expect("eq batch");
    let file = File::create(path).expect("create eq file");
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
        .expect("eq writer");
    writer.write(&batch).expect("write eq");
    writer.close().expect("close eq writer");
    FileScanTaskDeleteFile {
        file_path: path.to_string(),
        file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
        file_type: DataContentType::EqualityDeletes,
        partition_spec_id: 0,
        equality_ids: Some(vec![1]),
        file_format: DataFileFormat::Parquet,
        referenced_data_file: None,
        content_offset: None,
        content_size_in_bytes: None,
        record_count: None,
    }
}

async fn write_dv_delete_file(
    file_io: &FileIO,
    path: &str,
    referenced_data_path: &str,
    positions: &[u64],
) -> FileScanTaskDeleteFile {
    let output = file_io.new_output(path).expect("dv output");
    let mut writer = DVFileWriter::new(output).unpartitioned();
    for position in positions {
        writer
            .delete(referenced_data_path, *position, None)
            .expect("dv delete");
    }
    writer.close().await.expect("dv close");
    let input = file_io.new_input(path).expect("dv input");
    let puffin_reader = PuffinReader::new(input);
    let footer = puffin_reader.file_metadata().await.expect("dv footer");
    let blob = footer.blobs().first().expect("one dv blob");
    FileScanTaskDeleteFile {
        file_path: path.to_string(),
        file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        file_format: DataFileFormat::Puffin,
        referenced_data_file: Some(referenced_data_path.to_string()),
        content_offset: Some(i64::try_from(blob.offset()).expect("offset fits")),
        content_size_in_bytes: Some(i64::try_from(blob.length()).expect("length fits")),
        record_count: Some(positions.len() as u64),
    }
}

#[tokio::test]
async fn empty_projection_reads_row_count_with_zero_columns() {
    let tmp = TempDir::new().expect("temp dir");
    let data_path = tmp.path().join("data.parquet");
    let data_path = data_path.to_str().expect("utf-8").to_string();
    let ids: Vec<i32> = (0..100).collect();
    write_id_parquet(&data_path, &ids);
    let task = empty_task(&data_path, id_schema(), vec![], None);
    let batches = run_empty_scan(task, Some(7), false).await;
    assert!(batches.len() > 1, "batching must stream, not collapse");
    for batch in &batches {
        assert_eq!(batch.num_columns(), 0);
    }
    assert_eq!(total_rows(&batches), 100);
}

#[tokio::test]
async fn empty_projection_applies_positional_deletes() {
    let tmp = TempDir::new().expect("temp dir");
    let data_path = tmp.path().join("data.parquet");
    let data_path = data_path.to_str().expect("utf-8").to_string();
    let ids: Vec<i32> = (0..100).collect();
    write_id_parquet(&data_path, &ids);
    let delete_path = tmp.path().join("pos-deletes.parquet");
    let delete_path = delete_path.to_str().expect("utf-8").to_string();
    let delete = write_pos_delete_file(&delete_path, &data_path, &[3, 7, 50]);
    let task = empty_task(&data_path, id_schema(), vec![delete], None);
    let batches = run_empty_scan(task, None, false).await;
    for batch in &batches {
        assert_eq!(batch.num_columns(), 0);
    }
    assert_eq!(total_rows(&batches), 97);
}

#[tokio::test]
async fn empty_projection_applies_deletion_vector() {
    let tmp = TempDir::new().expect("temp dir");
    let data_path = tmp.path().join("data.parquet");
    let data_path = data_path.to_str().expect("utf-8").to_string();
    let ids: Vec<i32> = (0..100).collect();
    write_id_parquet(&data_path, &ids);
    let file_io = FileIO::new_with_fs();
    let dv_path = tmp.path().join("deletes.puffin");
    let dv_path = dv_path.to_str().expect("utf-8").to_string();
    let delete = write_dv_delete_file(&file_io, &dv_path, &data_path, &[10, 20]).await;
    let task = empty_task(&data_path, id_schema(), vec![delete], None);
    let batches = run_empty_scan(task, None, false).await;
    for batch in &batches {
        assert_eq!(batch.num_columns(), 0);
    }
    assert_eq!(total_rows(&batches), 98);
}

#[tokio::test]
async fn empty_projection_applies_residual_predicate() {
    let tmp = TempDir::new().expect("temp dir");
    let data_path = tmp.path().join("data.parquet");
    let data_path = data_path.to_str().expect("utf-8").to_string();
    let ids: Vec<i32> = (0..100).collect();
    write_id_parquet(&data_path, &ids);
    let schema = id_schema();
    let predicate = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(50))
        .bind(schema.clone(), false)
        .expect("bind");
    let task = empty_task(&data_path, schema, vec![], Some(predicate));
    let batches = run_empty_scan(task, None, false).await;
    for batch in &batches {
        assert_eq!(batch.num_columns(), 0);
    }
    assert_eq!(total_rows(&batches), 50);
}

#[tokio::test]
async fn empty_projection_applies_residual_predicate_with_row_selection() {
    let tmp = TempDir::new().expect("temp dir");
    let data_path = tmp.path().join("data.parquet");
    let data_path = data_path.to_str().expect("utf-8").to_string();
    let ids: Vec<i32> = (0..100).collect();
    write_id_parquet(&data_path, &ids);
    let schema = id_schema();
    let predicate = Reference::new("id")
        .greater_than_or_equal_to(Datum::int(50))
        .bind(schema.clone(), false)
        .expect("bind");
    let task = empty_task(&data_path, schema, vec![], Some(predicate));
    let batches = run_empty_scan(task, None, true).await;
    for batch in &batches {
        assert_eq!(batch.num_columns(), 0);
    }
    assert_eq!(total_rows(&batches), 50);
}

#[tokio::test]
async fn empty_projection_ignores_corrupt_column_bytes() {
    use std::io::{Read, Seek, SeekFrom, Write};

    use parquet::file::reader::{FileReader, SerializedFileReader};
    let tmp = TempDir::new().expect("temp dir");
    let data_path = tmp.path().join("data.parquet");
    let data_path = data_path.to_str().expect("utf-8").to_string();
    let ids: Vec<i32> = (0..100).collect();
    write_id_parquet(&data_path, &ids);
    let data_page_offset = {
        let file = File::open(&data_path).expect("open");
        let reader = SerializedFileReader::new(file).expect("footer");
        reader.metadata().row_group(0).column(0).data_page_offset() as u64
    };
    let mut file = File::options()
        .read(true)
        .write(true)
        .open(&data_path)
        .expect("open rw");
    file.seek(SeekFrom::Start(data_page_offset)).expect("seek");
    let mut page_head = [0u8; 16];
    file.read_exact(&mut page_head).expect("read page head");
    for byte in page_head.iter_mut() {
        *byte ^= 0xFF;
    }
    file.seek(SeekFrom::Start(data_page_offset))
        .expect("seek back");
    file.write_all(&page_head).expect("corrupt page head");
    drop(file);
    let task = empty_task(&data_path, id_schema(), vec![], None);
    let batches = run_empty_scan(task, None, false).await;
    assert_eq!(total_rows(&batches), 100);
}

#[tokio::test]
async fn empty_projection_applies_equality_deletes() {
    let tmp = TempDir::new().expect("temp dir");
    let data_path = tmp.path().join("data.parquet");
    let data_path = data_path.to_str().expect("utf-8").to_string();
    let ids: Vec<i32> = (0..100).collect();
    write_id_parquet(&data_path, &ids);
    let eq_path = tmp.path().join("eq-deletes.parquet");
    let eq_path = eq_path.to_str().expect("utf-8").to_string();
    let delete = write_eq_delete_file(&eq_path, &[20, 40, 60]);
    let task = empty_task(&data_path, id_schema(), vec![delete], None);
    let batches = run_empty_scan(task, None, false).await;
    for batch in &batches {
        assert_eq!(batch.num_columns(), 0);
    }
    assert_eq!(total_rows(&batches), 97);
}
