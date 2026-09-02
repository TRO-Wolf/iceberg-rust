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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use super::{close_touched_dv_containers_at, close_touched_dv_containers_with_partitions};
use crate::io::{
    FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorageFactory, OutputFile,
    Storage, StorageConfig, StorageFactory,
};
use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use crate::spec::{DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Struct};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, CatalogBuilder, Result};

fn is_manifest(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.ends_with(".avro") && !name.starts_with("snap-")
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CountingStorage {
    #[serde(skip)]
    inner: Option<Arc<dyn Storage>>,
    #[serde(skip)]
    manifest_reads: Arc<AtomicU64>,
}

impl CountingStorage {
    fn inner(&self) -> &Arc<dyn Storage> {
        self.inner.as_ref().expect("counting storage was built")
    }

    fn count(&self, path: &str) {
        if is_manifest(path) {
            self.manifest_reads.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for CountingStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner().exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        self.inner().metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        self.count(path);
        self.inner().read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        self.count(path);
        self.inner().reader(path).await
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        self.inner().write(path, bs).await
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        self.inner().writer(path).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner().delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        self.inner().delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        self.inner().list(prefix).await
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        self.inner().new_output(path)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CountingStorageFactory {
    #[serde(skip)]
    manifest_reads: Arc<AtomicU64>,
}

#[typetag::serde]
impl StorageFactory for CountingStorageFactory {
    fn build(&self, config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(CountingStorage {
            inner: Some(LocalFsStorageFactory.build(config)?),
            manifest_reads: self.manifest_reads.clone(),
        }))
    }
}

fn synthetic_data_file(path: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(0))]))
        .build()
        .expect("build synthetic data file")
}

async fn append(catalog: &impl Catalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

struct Fixture {
    table: Table,
    paths: Vec<String>,
    manifest_reads: Arc<AtomicU64>,
    _warehouse: TempDir,
}

async fn three_data_manifests() -> Fixture {
    let warehouse = TempDir::new().expect("warehouse");
    let manifest_reads = Arc::new(AtomicU64::new(0));
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(CountingStorageFactory {
            manifest_reads: manifest_reads.clone(),
        }))
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().expect("utf8").to_string(),
            )]),
        )
        .await
        .expect("catalog");
    let mut table = crate::transaction::tests::make_v3_minimal_table_in_catalog(&catalog).await;
    let mut paths = Vec::new();
    for index in 0..3 {
        let path = format!("{}/data/f{index}.parquet", table.metadata().location());
        table = append(&catalog, &table, synthetic_data_file(&path)).await;
        paths.push(path);
    }
    manifest_reads.store(0, Ordering::Relaxed);
    Fixture {
        table,
        paths,
        manifest_reads,
        _warehouse: warehouse,
    }
}

#[tokio::test]
async fn a_supplied_partition_map_reads_no_data_manifest() {
    let fixture = three_data_manifests().await;
    let new_positions = HashMap::from([(fixture.paths[0].clone(), vec![0u64])]);
    let known = HashMap::from([(
        fixture.paths[0].clone(),
        (0i32, Struct::from_iter([Some(Literal::long(0))])),
    )]);
    let close =
        close_touched_dv_containers_with_partitions(&fixture.table, &new_positions, None, &known)
            .await
            .expect("close with a supplied partition map");
    assert_eq!(close.added.len(), 1);
    assert_eq!(
        fixture.manifest_reads.load(Ordering::Relaxed),
        0,
        "a supplied partition map must not walk any manifest"
    );
}

#[tokio::test]
async fn without_the_map_every_data_manifest_is_read_once() {
    let fixture = three_data_manifests().await;
    let new_positions = HashMap::from([(fixture.paths[0].clone(), vec![0u64])]);
    let close = close_touched_dv_containers_at(&fixture.table, &new_positions, None)
        .await
        .expect("close without a partition map");
    assert_eq!(close.added.len(), 1);
    assert_eq!(
        fixture.manifest_reads.load(Ordering::Relaxed),
        3,
        "each of the three data manifests is read exactly once"
    );
}
