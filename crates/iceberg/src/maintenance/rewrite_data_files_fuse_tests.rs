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
use std::ops::Range;
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;

use crate::io::{
    FileIO, FileIOBuilder, FileInfo, FileMetadata, FileRead, FileWrite, InputFile,
    MemoryStorageFactory, OutputFile, Storage, StorageConfig, StorageFactory,
};
use crate::maintenance::rewrite_data_files_write::{
    dictionary_fallback_columns, input_parquet_metadata,
};
use crate::scan::FileScanTask;
use crate::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, SchemaRef, Type};

type ReadLog = Arc<Mutex<HashMap<String, Vec<Range<u64>>>>>;

struct CountingFileRead {
    inner: Box<dyn FileRead>,
    path: String,
    reads: ReadLog,
}

#[async_trait::async_trait]
impl FileRead for CountingFileRead {
    async fn read(&self, range: Range<u64>) -> crate::Result<Bytes> {
        self.reads
            .lock()
            .expect("read log")
            .entry(self.path.clone())
            .or_default()
            .push(range.clone());
        self.inner.read(range).await
    }
}

#[derive(Debug)]
struct CountingStorage {
    inner: Arc<dyn Storage>,
    reads: ReadLog,
}

impl serde::Serialize for CountingStorage {
    fn serialize<S>(&self, _serializer: S) -> std::result::Result<S::Ok, S::Error>
    where S: serde::Serializer {
        Err(serde::ser::Error::custom("counting storage is test-only"))
    }
}

impl<'de> serde::Deserialize<'de> for CountingStorage {
    fn deserialize<D>(_deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        Err(serde::de::Error::custom("counting storage is test-only"))
    }
}

#[async_trait::async_trait]
#[typetag::serde(name = "counting-storage")]
impl Storage for CountingStorage {
    async fn exists(&self, path: &str) -> crate::Result<bool> {
        self.inner.exists(path).await
    }

    async fn metadata(&self, path: &str) -> crate::Result<FileMetadata> {
        self.inner.metadata(path).await
    }

    async fn read(&self, path: &str) -> crate::Result<Bytes> {
        self.inner.read(path).await
    }

    async fn reader(&self, path: &str) -> crate::Result<Box<dyn FileRead>> {
        Ok(Box::new(CountingFileRead {
            inner: self.inner.reader(path).await?,
            path: path.to_string(),
            reads: Arc::clone(&self.reads),
        }))
    }

    async fn write(&self, path: &str, bs: Bytes) -> crate::Result<()> {
        self.inner.write(path, bs).await
    }

    async fn writer(&self, path: &str) -> crate::Result<Box<dyn FileWrite>> {
        self.inner.writer(path).await
    }

    async fn delete(&self, path: &str) -> crate::Result<()> {
        self.inner.delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> crate::Result<()> {
        self.inner.delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> crate::Result<Vec<FileInfo>> {
        self.inner.list(prefix).await
    }

    fn new_input(&self, path: &str) -> crate::Result<InputFile> {
        Ok(InputFile::new(
            Arc::new(CountingStorage {
                inner: Arc::clone(&self.inner),
                reads: Arc::clone(&self.reads),
            }),
            path.to_string(),
        ))
    }

    fn new_output(&self, path: &str) -> crate::Result<OutputFile> {
        self.inner.new_output(path)
    }
}

#[derive(Debug)]
struct CountingStorageFactory {
    reads: ReadLog,
}

impl serde::Serialize for CountingStorageFactory {
    fn serialize<S>(&self, _serializer: S) -> std::result::Result<S::Ok, S::Error>
    where S: serde::Serializer {
        Err(serde::ser::Error::custom(
            "counting storage factory is test-only",
        ))
    }
}

impl<'de> serde::Deserialize<'de> for CountingStorageFactory {
    fn deserialize<D>(_deserializer: D) -> std::result::Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        Err(serde::de::Error::custom(
            "counting storage factory is test-only",
        ))
    }
}

#[typetag::serde(name = "counting-storage-factory")]
impl StorageFactory for CountingStorageFactory {
    fn build(&self, config: &StorageConfig) -> crate::Result<Arc<dyn Storage>> {
        Ok(Arc::new(CountingStorage {
            inner: MemoryStorageFactory.build(config)?,
            reads: Arc::clone(&self.reads),
        }))
    }
}

fn counting_file_io() -> (FileIO, ReadLog) {
    let reads: ReadLog = Arc::new(Mutex::new(HashMap::new()));
    (
        FileIOBuilder::new(Arc::new(CountingStorageFactory {
            reads: Arc::clone(&reads),
        }))
        .build(),
        reads,
    )
}

fn parquet_task(path: &str, size: u64) -> FileScanTask {
    FileScanTask {
        file_size_in_bytes: size,
        start: 0,
        length: size,
        record_count: None,
        data_file_path: Arc::from(path),
        data_file_format: DataFileFormat::Parquet,
        schema: test_schema(),
        project_field_ids: Arc::from(vec![1]),
        predicate: None,
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

fn test_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("test schema"),
    )
}

fn parquet_bytes(batch: &RecordBatch) -> Bytes {
    let mut buffer = Vec::new();
    let mut writer =
        ArrowWriter::try_new(&mut buffer, batch.schema(), None).expect("parquet writer");
    writer.write(batch).expect("write batch");
    writer.close().expect("close parquet writer");
    Bytes::from(buffer)
}

fn footer_metadata_length(bytes: &[u8]) -> u64 {
    let tail = &bytes[bytes.len() - 8..bytes.len() - 4];
    u32::from_le_bytes(tail.try_into().expect("4-byte footer length")) as u64
}

fn narrow_batch() -> RecordBatch {
    let values: Vec<i64> = (0..200).map(|row| row % 7).collect();
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(values)) as ArrayRef],
    )
    .expect("narrow batch")
}

fn wide_batch() -> RecordBatch {
    const COLUMNS: usize = 190;
    const ROWS: usize = 4000;
    let fields: Vec<Field> = (0..COLUMNS)
        .map(|column| Field::new(format!("c{column}"), DataType::Int64, false))
        .collect();
    let columns: Vec<ArrayRef> = (0..COLUMNS)
        .map(|column| {
            let values: Vec<i64> = (0..ROWS)
                .map(|row| {
                    ((row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ column as u64) as i64
                })
                .collect();
            Arc::new(Int64Array::from(values)) as ArrayRef
        })
        .collect();
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).expect("wide batch")
}

#[tokio::test]
async fn decision_footers_are_bounded_to_the_group_and_carry_no_indexes() {
    const FILE_COUNT: usize = 200;
    let (file_io, _reads) = counting_file_io();
    let batch = narrow_batch();
    let mut tasks = Vec::with_capacity(FILE_COUNT);
    for index in 0..FILE_COUNT {
        let path = format!("memory://t/data/fuse-{index}.parquet");
        let bytes = parquet_bytes(&batch);
        let size = bytes.len() as u64;
        file_io
            .new_output(&path)
            .expect("output")
            .write(bytes)
            .await
            .expect("write file");
        tasks.push(parquet_task(&path, size));
    }

    let (_fallback, footers) = dictionary_fallback_columns(&file_io, &tasks)
        .await
        .expect("decision pass");
    assert_eq!(
        footers.len(),
        FILE_COUNT,
        "retained footers must be bounded to the parquet files of the group, got {}",
        footers.len(),
    );
    for (path, metadata) in &footers {
        assert!(
            metadata.column_index().is_none(),
            "{path}: the retained footer must not carry a column index",
        );
        assert!(
            metadata.offset_index().is_none(),
            "{path}: the retained footer must not carry an offset index",
        );
    }
}

#[tokio::test]
async fn decision_footer_fetch_is_tail_plus_exact_metadata() {
    let (file_io, reads) = counting_file_io();
    let bytes = parquet_bytes(&wide_batch());
    let size = bytes.len() as u64;
    assert!(
        size > 512 * 1024,
        "the wide bed file must exceed the old 512 KiB prefetch, got {size}",
    );
    let footer_len = footer_metadata_length(&bytes);
    let path = "memory://t/data/wide.parquet";
    file_io
        .new_output(path)
        .expect("output")
        .write(bytes)
        .await
        .expect("write file");
    reads.lock().expect("read log").clear();

    let task = parquet_task(path, size);
    let _ = input_parquet_metadata(&file_io, &task)
        .await
        .expect("decision metadata");

    let ranges = reads
        .lock()
        .expect("read log")
        .remove(path)
        .expect("recorded reads");
    let fetched: u64 = ranges.iter().map(|range| range.end - range.start).sum();
    assert_eq!(
        fetched,
        footer_len + 8,
        "the decision pass must fetch only the 8-byte footer tail plus the exact footer (fetched={fetched}, footer_len={footer_len}, ranges: {ranges:?})",
    );
}
