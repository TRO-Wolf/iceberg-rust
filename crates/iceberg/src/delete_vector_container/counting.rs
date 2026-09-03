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

use std::collections::HashSet;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::io::{
    FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorageFactory, OutputFile,
    Storage, StorageConfig, StorageFactory,
};
use crate::spec::{DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Struct};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, Result};

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn is_snapshot_list(path: &str) -> bool {
    let name = file_name(path);
    name.starts_with("snap-") && name.ends_with(".avro")
}

fn is_manifest(path: &str) -> bool {
    let name = file_name(path);
    name.ends_with(".avro") && !name.starts_with("snap-")
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CountingStorage {
    #[serde(skip)]
    inner: Option<Arc<dyn Storage>>,
    #[serde(skip)]
    manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    bytes_read: Arc<AtomicU64>,
    #[serde(skip)]
    delete_manifest_paths: Arc<Mutex<HashSet<String>>>,
    #[serde(skip)]
    delete_manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    data_manifest_paths: Arc<Mutex<HashSet<String>>>,
    #[serde(skip)]
    data_manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    snapshot_list_reads: Arc<AtomicU64>,
    #[serde(skip)]
    opens: Arc<AtomicU64>,
}

impl CountingStorage {
    fn inner(&self) -> &Arc<dyn Storage> {
        self.inner.as_ref().expect("counting storage was built")
    }

    fn count(&self, path: &str) {
        if path.ends_with(".parquet") {
            self.opens.fetch_add(1, Ordering::Relaxed);
        }
        if is_snapshot_list(path) {
            self.snapshot_list_reads.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if is_manifest(path) {
            self.manifest_reads.fetch_add(1, Ordering::Relaxed);
            if self
                .delete_manifest_paths
                .lock()
                .expect("delete-manifest path set")
                .contains(path)
            {
                self.delete_manifest_reads.fetch_add(1, Ordering::Relaxed);
            }
            if self
                .data_manifest_paths
                .lock()
                .expect("data-manifest path set")
                .contains(path)
            {
                self.data_manifest_reads.fetch_add(1, Ordering::Relaxed);
            }
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
        let bytes = self.inner().read(path).await?;
        self.bytes_read
            .fetch_add(u64::try_from(bytes.len()).unwrap_or(0), Ordering::Relaxed);
        Ok(bytes)
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        self.count(path);
        let inner = self.inner().reader(path).await?;
        Ok(Box::new(CountingFileRead {
            inner,
            bytes_read: self.bytes_read.clone(),
        }))
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

struct CountingFileRead {
    inner: Box<dyn FileRead>,
    bytes_read: Arc<AtomicU64>,
}

#[async_trait]
impl FileRead for CountingFileRead {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        let bytes = self.inner.read(range).await?;
        self.bytes_read
            .fetch_add(u64::try_from(bytes.len()).unwrap_or(0), Ordering::Relaxed);
        Ok(bytes)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct CountingStorageFactory {
    #[serde(skip)]
    pub(crate) manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    pub(crate) bytes_read: Arc<AtomicU64>,
    #[serde(skip)]
    pub(crate) delete_manifest_paths: Arc<Mutex<HashSet<String>>>,
    #[serde(skip)]
    pub(crate) delete_manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    pub(crate) data_manifest_paths: Arc<Mutex<HashSet<String>>>,
    #[serde(skip)]
    pub(crate) data_manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    pub(crate) snapshot_list_reads: Arc<AtomicU64>,
    #[serde(skip)]
    pub(crate) opens: Arc<AtomicU64>,
}

#[typetag::serde]
impl StorageFactory for CountingStorageFactory {
    fn build(&self, config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(CountingStorage {
            inner: Some(LocalFsStorageFactory.build(config)?),
            manifest_reads: self.manifest_reads.clone(),
            bytes_read: self.bytes_read.clone(),
            delete_manifest_paths: self.delete_manifest_paths.clone(),
            delete_manifest_reads: self.delete_manifest_reads.clone(),
            data_manifest_paths: self.data_manifest_paths.clone(),
            data_manifest_reads: self.data_manifest_reads.clone(),
            snapshot_list_reads: self.snapshot_list_reads.clone(),
            opens: self.opens.clone(),
        }))
    }
}

pub(crate) fn synthetic_data_file(path: &str) -> DataFile {
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

pub(crate) async fn append(catalog: &impl Catalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}
