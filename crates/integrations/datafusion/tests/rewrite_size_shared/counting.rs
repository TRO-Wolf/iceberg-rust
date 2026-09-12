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

use bytes::Bytes;
use iceberg::io::{
    FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorageFactory, OutputFile,
    Storage, StorageConfig, StorageFactory,
};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec};

use crate::rewrite_size_shared::{ProbeFixture, create_fixture_inner};

pub type ReadLog = Arc<Mutex<HashMap<String, Vec<Range<u64>>>>>;

struct CountingFileRead {
    inner: Box<dyn FileRead>,
    path: String,
    reads: ReadLog,
}

#[async_trait::async_trait]
impl FileRead for CountingFileRead {
    async fn read(&self, range: Range<u64>) -> iceberg::Result<Bytes> {
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
    async fn exists(&self, path: &str) -> iceberg::Result<bool> {
        self.inner.exists(path).await
    }

    async fn metadata(&self, path: &str) -> iceberg::Result<FileMetadata> {
        self.inner.metadata(path).await
    }

    async fn read(&self, path: &str) -> iceberg::Result<Bytes> {
        self.inner.read(path).await
    }

    async fn reader(&self, path: &str) -> iceberg::Result<Box<dyn FileRead>> {
        Ok(Box::new(CountingFileRead {
            inner: self.inner.reader(path).await?,
            path: path.to_string(),
            reads: Arc::clone(&self.reads),
        }))
    }

    async fn write(&self, path: &str, bs: Bytes) -> iceberg::Result<()> {
        self.inner.write(path, bs).await
    }

    async fn writer(&self, path: &str) -> iceberg::Result<Box<dyn FileWrite>> {
        self.inner.writer(path).await
    }

    async fn delete(&self, path: &str) -> iceberg::Result<()> {
        self.inner.delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> iceberg::Result<()> {
        self.inner.delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> iceberg::Result<Vec<FileInfo>> {
        self.inner.list(prefix).await
    }

    fn new_input(&self, path: &str) -> iceberg::Result<InputFile> {
        Ok(InputFile::new(
            Arc::new(CountingStorage {
                inner: Arc::clone(&self.inner),
                reads: Arc::clone(&self.reads),
            }),
            path.to_string(),
        ))
    }

    fn new_output(&self, path: &str) -> iceberg::Result<OutputFile> {
        self.inner.new_output(path)
    }
}

#[derive(Debug)]
pub struct CountingStorageFactory {
    pub reads: ReadLog,
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
    fn build(&self, config: &StorageConfig) -> iceberg::Result<Arc<dyn Storage>> {
        Ok(Arc::new(CountingStorage {
            inner: LocalFsStorageFactory.build(config)?,
            reads: Arc::clone(&self.reads),
        }))
    }
}

pub async fn create_counting_fixture(compression_level: Option<&str>) -> (ProbeFixture, ReadLog) {
    let reads: ReadLog = Arc::new(Mutex::new(HashMap::new()));
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp)).into(),
            NestedField::optional(2, "grp", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("build schema");
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "grp", Transform::Identity)
        .expect("partition field")
        .build();
    let fixture = create_fixture_inner(
        schema,
        partition_spec,
        "probe",
        "bed",
        compression_level,
        Arc::new(CountingStorageFactory {
            reads: Arc::clone(&reads),
        }),
    )
    .await;
    (fixture, reads)
}
