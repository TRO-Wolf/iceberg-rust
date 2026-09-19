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
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use crate::io::{
    FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorage, OutputFile, Storage,
    StorageConfig, StorageFactory,
};
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, config_for, create_partitioned_table, scan_rows,
    synthetic_spec_and_schema, synthetic_task, write_data_file, write_equality_delete_file,
};
use crate::maintenance::rewrite_data_files_plan::{input_split_size, plan_read_tasks};
use crate::memory::MemoryCatalogBuilder;
use crate::{Catalog, CatalogBuilder, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CountingStorage {
    #[serde(skip)]
    inner: LocalFsStorage,
    #[serde(skip)]
    reader_calls: Arc<Mutex<HashMap<String, u64>>>,
}

#[async_trait]
#[typetag::serde]
impl Storage for CountingStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        self.inner.metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        self.inner.read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        *self
            .reader_calls
            .lock()
            .expect("reader call counts")
            .entry(path.to_string())
            .or_insert(0) += 1;
        self.inner.reader(path).await
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        self.inner.write(path, bs).await
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        self.inner.writer(path).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner.delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        self.inner.delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        self.inner.list(prefix).await
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        self.inner.new_output(path)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CountingStorageFactory {
    #[serde(skip)]
    reader_calls: Arc<Mutex<HashMap<String, u64>>>,
}

#[typetag::serde]
impl StorageFactory for CountingStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(CountingStorage {
            inner: LocalFsStorage::new(),
            reader_calls: self.reader_calls.clone(),
        }))
    }
}

async fn counting_catalog() -> (impl Catalog, TempDir, Arc<Mutex<HashMap<String, u64>>>) {
    let reader_calls = Arc::new(Mutex::new(HashMap::new()));
    let temp_dir = TempDir::new().expect("temp dir");
    let warehouse = temp_dir
        .path()
        .to_str()
        .expect("utf8 temp path")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(CountingStorageFactory {
            reader_calls: reader_calls.clone(),
        }))
        .load(
            "memory",
            HashMap::from([("warehouse".to_string(), warehouse)]),
        )
        .await
        .expect("load counting local-fs catalog");
    (catalog, temp_dir, reader_calls)
}

#[tokio::test]
async fn test_partition_scoped_delete_loads_once_across_read_tasks() {
    let (catalog, _temp, reader_calls) = counting_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

    let mut sizes = Vec::new();
    let mut files = Vec::new();
    for index in 0..4i64 {
        let rows: Vec<(i64, i64, i64)> = (0..50).map(|row| (0, index * 50 + row, row)).collect();
        let file = write_data_file(&table, &format!("d-{index}.parquet"), 0, &rows).await;
        sizes.push(file.file_size_in_bytes());
        files.push(file);
    }
    let table = append_files(&catalog, &table, files).await;

    let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
    let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 199, "the eq delete drops y=20");
    assert!(!rows_before.iter().any(|(_, y, _)| *y == 20));

    let (spec, schema) = synthetic_spec_and_schema();
    let config = config_for(2_000, 1_500, 3_600, 1);
    let tasks: Vec<crate::scan::FileScanTask> = sizes
        .iter()
        .enumerate()
        .map(|(i, size)| synthetic_task(&format!("s-{i}"), *size, 0, 0, &spec, &schema))
        .collect();
    let input_size: u64 = tasks.iter().map(|task| task.length).sum();
    let expected_read_tasks = plan_read_tasks(tasks, input_split_size(input_size, &config))
        .unwrap()
        .len();
    println!("sizes {sizes:?} input {input_size} read tasks {expected_read_tasks}");
    assert!(
        expected_read_tasks >= 3,
        "the fixture must split the group into >= 3 read tasks, got {expected_read_tasks}"
    );

    reader_calls.lock().expect("reader call counts").clear();
    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .target_file_size_bytes(2_000)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");
    assert_eq!(result.rewritten_data_files_count, 4);
    assert_eq!(
        result.added_data_files_count, expected_read_tasks,
        "one output file per read task"
    );

    let loads: u64 = reader_calls
        .lock()
        .expect("reader call counts")
        .iter()
        .filter(|(path, _)| path.contains("eq-del"))
        .map(|(_, count)| *count)
        .sum();
    assert_eq!(
        loads, 1,
        "the partition-scoped delete file must be read exactly once across all read tasks"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    let rows_after = scan_rows(&table).await;
    assert_eq!(rows_after, rows_before, "row conservation");
}
