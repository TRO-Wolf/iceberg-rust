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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use super::*;
use crate::io::{
    FileInfo, FileMetadata, FileRead, FileWrite, InputFile, MemoryStorage,
    MemoryStorageFactory, OutputFile, Storage, StorageConfig, StorageFactory,
};
use crate::spec::{
    NestedField, PrimitiveType, Schema, TableMetadata, TableMetadataBuilder, Type,
};
use crate::{Error, ErrorKind, TableCreation};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CountingStorage {
    #[serde(skip)]
    inner: MemoryStorage,
    #[serde(skip)]
    body_reads: Arc<AtomicU64>,
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
        self.body_reads.fetch_add(1, Ordering::Relaxed);
        self.inner.read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        self.body_reads.fetch_add(1, Ordering::Relaxed);
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
    storage: MemoryStorage,
    #[serde(skip)]
    body_reads: Arc<AtomicU64>,
}

impl CountingStorageFactory {
    fn new() -> (Self, Arc<AtomicU64>) {
        let body_reads = Arc::new(AtomicU64::new(0));
        (
            Self {
                storage: MemoryStorage::new(),
                body_reads: body_reads.clone(),
            },
            body_reads,
        )
    }
}

#[typetag::serde]
impl StorageFactory for CountingStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(CountingStorage {
            inner: self.storage.clone(),
            body_reads: self.body_reads.clone(),
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GatedStorage {
    #[serde(skip)]
    inner: MemoryStorage,
    #[serde(skip)]
    body_reads: Arc<AtomicU64>,
    #[serde(skip)]
    fail_reads: Arc<AtomicBool>,
    #[serde(skip)]
    read_yields: u32,
}

#[async_trait]
#[typetag::serde]
impl Storage for GatedStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        self.inner.metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        for _ in 0..self.read_yields {
            tokio::task::yield_now().await;
        }
        self.body_reads.fetch_add(1, Ordering::Relaxed);
        if self.fail_reads.load(Ordering::Relaxed) {
            return Err(Error::new(ErrorKind::Unexpected, "injected read failure"));
        }
        self.inner.read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
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
struct GatedStorageFactory {
    #[serde(skip)]
    storage: MemoryStorage,
    #[serde(skip)]
    body_reads: Arc<AtomicU64>,
    #[serde(skip)]
    fail_reads: Arc<AtomicBool>,
    #[serde(skip)]
    read_yields: u32,
}

impl GatedStorageFactory {
    fn new(read_yields: u32) -> (Self, Arc<AtomicU64>, Arc<AtomicBool>) {
        let body_reads = Arc::new(AtomicU64::new(0));
        let fail_reads = Arc::new(AtomicBool::new(false));
        (
            Self {
                storage: MemoryStorage::new(),
                body_reads: body_reads.clone(),
                fail_reads: fail_reads.clone(),
                read_yields,
            },
            body_reads,
            fail_reads,
        )
    }
}

#[typetag::serde]
impl StorageFactory for GatedStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(GatedStorage {
            inner: self.storage.clone(),
            body_reads: self.body_reads.clone(),
            fail_reads: self.fail_reads.clone(),
            read_yields: self.read_yields,
        }))
    }
}

fn sample_metadata(location: &str) -> TableMetadata {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("schema");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(location.to_string())
        .schema(schema)
        .build();
    TableMetadataBuilder::from_table_creation(creation)
        .expect("builder")
        .build()
        .expect("metadata")
        .metadata
}

fn test_scope() -> CacheScope {
    CacheScope::new("test:catalog", "test:creds")
}

#[tokio::test]
async fn two_loads_unchanged_pointer_zero_body_get_on_second() {
    let (factory, body_reads) = CountingStorageFactory::new();
    let file_io = crate::io::FileIOBuilder::new(Arc::new(factory)).build();
    let location = "memory://warehouse/t/metadata/v1.metadata.json";
    let metadata = sample_metadata("memory://warehouse/t");
    metadata
        .write_to(&file_io, location)
        .await
        .expect("write metadata");

    let cache = TableMetadataCache::new();
    let scope = test_scope();
    body_reads.store(0, Ordering::Relaxed);

    let first = load_or_fetch_table_metadata(&file_io, &scope, location, Some(&cache), None)
        .await
        .expect("first load");
    let reads_after_first = body_reads.load(Ordering::Relaxed);
    assert_eq!(reads_after_first, 1, "first load must body-GET once");
    assert_eq!(cache.stats().body_fetches, 1);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 0);

    let second = load_or_fetch_table_metadata(&file_io, &scope, location, Some(&cache), None)
        .await
        .expect("second load");
    let reads_after_second = body_reads.load(Ordering::Relaxed);
    assert_eq!(
        reads_after_second, reads_after_first,
        "second load unchanged pointer must add ZERO body GETs"
    );
    assert_eq!(cache.stats().body_fetches, 1, "still one body fetch");
    assert_eq!(cache.stats().hits, 1);
    assert!(
        Arc::ptr_eq(&first, &second),
        "cache hit must return the same Arc"
    );
    assert_eq!(first.location(), second.location());
}

#[tokio::test]
async fn default_off_always_body_gets() {
    let (factory, body_reads) = CountingStorageFactory::new();
    let file_io = crate::io::FileIOBuilder::new(Arc::new(factory)).build();
    let location = "memory://warehouse/t/metadata/v1.metadata.json";
    sample_metadata("memory://warehouse/t")
        .write_to(&file_io, location)
        .await
        .expect("write");

    let scope = test_scope();
    body_reads.store(0, Ordering::Relaxed);
    let _a = load_or_fetch_table_metadata(&file_io, &scope, location, None, None)
        .await
        .expect("load a");
    let _b = load_or_fetch_table_metadata(&file_io, &scope, location, None, None)
        .await
        .expect("load b");
    assert_eq!(
        body_reads.load(Ordering::Relaxed),
        2,
        "without cache every load is a body GET"
    );
}

#[tokio::test]
async fn object_version_mismatch_fail_closed_refetches() {
    let (factory, body_reads) = CountingStorageFactory::new();
    let file_io = crate::io::FileIOBuilder::new(Arc::new(factory)).build();
    let location = "memory://warehouse/t/metadata/v1.metadata.json";
    sample_metadata("memory://warehouse/t")
        .write_to(&file_io, location)
        .await
        .expect("write");

    let cache = TableMetadataCache::new();
    let scope = test_scope();
    body_reads.store(0, Ordering::Relaxed);

    let _ = load_or_fetch_table_metadata(
        &file_io,
        &scope,
        location,
        Some(&cache),
        Some("v1"),
    )
    .await
    .expect("seed");
    assert_eq!(body_reads.load(Ordering::Relaxed), 1);

    let _ = load_or_fetch_table_metadata(
        &file_io,
        &scope,
        location,
        Some(&cache),
        Some("v2"),
    )
    .await
    .expect("guard mismatch must re-fetch");
    assert_eq!(
        body_reads.load(Ordering::Relaxed),
        2,
        "version mismatch must not soft-reuse"
    );
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.stats().hits, 0);

    let _ = load_or_fetch_table_metadata(
        &file_io,
        &scope,
        location,
        Some(&cache),
        Some("v2"),
    )
    .await
    .expect("hit");
    assert_eq!(body_reads.load(Ordering::Relaxed), 2);
    assert_eq!(cache.stats().hits, 1);
}

#[tokio::test]
async fn version_mismatch_failed_refetch_evicts_stale_entry() {
    let (factory, _body_reads) = CountingStorageFactory::new();
    let file_io = crate::io::FileIOBuilder::new(Arc::new(factory)).build();
    let location = "memory://warehouse/t/metadata/gone.metadata.json";
    sample_metadata("memory://warehouse/t")
        .write_to(&file_io, location)
        .await
        .expect("write");

    let cache = TableMetadataCache::new();
    let scope = test_scope();
    let _ = load_or_fetch_table_metadata(
        &file_io,
        &scope,
        location,
        Some(&cache),
        Some("v1"),
    )
    .await
    .expect("seed");
    file_io.delete(location).await.expect("delete");

    let err = load_or_fetch_table_metadata(
        &file_io,
        &scope,
        location,
        Some(&cache),
        Some("v2"),
    )
    .await
    .expect_err("re-fetch must fail after delete");
    let _ = err;

    assert!(
        cache.lookup(&scope, location, Some("v1")).await.is_none(),
        "after guard-mismatch miss + failed fetch, old version must not hit"
    );
    assert!(
        cache.lookup(&scope, location, Some("v2")).await.is_none(),
        "failed fetch must not install a new entry"
    );
}

#[tokio::test]
async fn location_change_is_miss() {
    let factory = MemoryStorageFactory;
    let file_io = crate::io::FileIOBuilder::new(Arc::new(factory)).build();
    let loc1 = "memory://warehouse/t/metadata/v1.metadata.json";
    let loc2 = "memory://warehouse/t/metadata/v2.metadata.json";
    let meta = sample_metadata("memory://warehouse/t");
    meta.write_to(&file_io, loc1).await.expect("w1");
    meta.write_to(&file_io, loc2).await.expect("w2");

    let cache = TableMetadataCache::new();
    let scope = test_scope();
    let _ = load_or_fetch_table_metadata(&file_io, &scope, loc1, Some(&cache), None)
        .await
        .expect("l1");
    let _ = load_or_fetch_table_metadata(&file_io, &scope, loc2, Some(&cache), None)
        .await
        .expect("l2");
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.stats().hits, 0);
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 2);
}

#[tokio::test]
async fn invalidate_forces_refetch() {
    let (factory, body_reads) = CountingStorageFactory::new();
    let file_io = crate::io::FileIOBuilder::new(Arc::new(factory)).build();
    let location = "memory://warehouse/t/metadata/v1.metadata.json";
    sample_metadata("memory://warehouse/t")
        .write_to(&file_io, location)
        .await
        .expect("write");

    let cache = TableMetadataCache::new();
    let scope = test_scope();
    let _ = load_or_fetch_table_metadata(&file_io, &scope, location, Some(&cache), None)
        .await
        .expect("seed");
    cache.invalidate(&scope, location).await;
    body_reads.store(0, Ordering::Relaxed);
    let _ = load_or_fetch_table_metadata(&file_io, &scope, location, Some(&cache), None)
        .await
        .expect("after invalidate");
    assert_eq!(body_reads.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn version_never_sole_check_location_required() {
    let cache = TableMetadataCache::new();
    let scope = test_scope();
    let meta = Arc::new(sample_metadata("memory://warehouse/t"));
    cache
        .put(
            &scope,
            "memory://a".to_string(),
            meta.clone(),
            Some("etag-1".to_string()),
        )
        .await;
    assert!(
        cache
            .lookup(&scope, "memory://b", Some("etag-1"))
            .await
            .is_none()
    );
    assert!(
        cache
            .lookup(&scope, "memory://a", Some("etag-1"))
            .await
            .is_some()
    );
    assert!(cache.lookup(&scope, "memory://a", None).await.is_some());
}

#[tokio::test]
async fn learn_version_guard_then_mismatch_fail_closed() {
    let cache = TableMetadataCache::new();
    let scope = test_scope();
    let meta = Arc::new(sample_metadata("memory://warehouse/t"));
    cache.put(&scope, "memory://a".to_string(), meta, None).await;

    assert!(
        cache.lookup(&scope, "memory://a", Some("v1")).await.is_some(),
        "first versioned lookup on unguarded entry must hit and learn"
    );
    assert!(
        cache
            .lookup(&scope, "memory://a", Some("v2"))
            .await
            .is_none(),
        "after learn, disagreeing version must fail closed"
    );
    assert!(
        cache.lookup(&scope, "memory://a", Some("v1")).await.is_some(),
        "matching learned version still hits"
    );
}

#[tokio::test]
async fn eviction_under_pressure_bounds_and_counts_and_never_stale() {
    let (factory, _body_reads) = CountingStorageFactory::new();
    let file_io = crate::io::FileIOBuilder::new(Arc::new(factory)).build();
    let scope = test_scope();
    let cache = TableMetadataCache::with_max_entries(2);

    for i in 0..5 {
        let location = format!("memory://warehouse/t/metadata/v{i}.metadata.json");
        sample_metadata(&format!("memory://warehouse/t{i}"))
            .write_to(&file_io, &location)
            .await
            .expect("write");
        let metadata = load_or_fetch_table_metadata(
            &file_io,
            &scope,
            &location,
            Some(&cache),
            None,
        )
        .await
        .expect("load");
        assert_eq!(
            metadata.location(),
            format!("memory://warehouse/t{i}").as_str(),
            "every returned entry must be the metadata for ITS location"
        );
    }
    cache.run_pending_tasks().await;

    assert!(
        cache.len() <= 2,
        "cache must hold at most the configured bound: {}",
        cache.len()
    );
    assert!(
        cache.stats().evictions >= 3,
        "5 inserts into a bound of 2 must evict at least 3: {}",
        cache.stats().evictions
    );

    for i in 0..5 {
        let location = format!("memory://warehouse/t/metadata/v{i}.metadata.json");
        let metadata = load_or_fetch_table_metadata(
            &file_io,
            &scope,
            &location,
            Some(&cache),
            None,
        )
        .await
        .expect("reload");
        assert_eq!(
            metadata.location(),
            format!("memory://warehouse/t{i}").as_str(),
            "evicted or retained, a load must never return another location's metadata"
        );
    }
}

#[tokio::test]
async fn different_scopes_same_location_never_share() {
    let (factory, body_reads) = CountingStorageFactory::new();
    let file_io = crate::io::FileIOBuilder::new(Arc::new(factory)).build();
    let location = "memory://warehouse/t/metadata/v1.metadata.json";
    sample_metadata("memory://warehouse/t")
        .write_to(&file_io, location)
        .await
        .expect("write");

    let cache = TableMetadataCache::new();
    let scope_a = CacheScope::new("s3tables:arn:a", "creds-a");
    let scope_b = CacheScope::new("s3tables:arn:b", "creds-b");
    body_reads.store(0, Ordering::Relaxed);

    let first = load_or_fetch_table_metadata(&file_io, &scope_a, location, Some(&cache), None)
        .await
        .expect("load a");
    let second = load_or_fetch_table_metadata(&file_io, &scope_b, location, Some(&cache), None)
        .await
        .expect("load b");

    assert_eq!(
        body_reads.load(Ordering::Relaxed),
        2,
        "two scopes must body-GET the same location string twice"
    );
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.stats().hits, 0);
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 2, "one entry per scope, same location");
    assert!(
        !Arc::ptr_eq(&first, &second),
        "different scopes must hold distinct Arcs"
    );
    assert_eq!(first.location(), second.location());
}

#[tokio::test]
async fn concurrent_cold_loads_dedup_single_fetch_and_errors_reach_all() {
    let (factory, body_reads, fail_reads) = GatedStorageFactory::new(64);
    let file_io = crate::io::FileIOBuilder::new(Arc::new(factory)).build();
    let location = "memory://warehouse/t/metadata/v1.metadata.json";
    sample_metadata("memory://warehouse/t")
        .write_to(&file_io, location)
        .await
        .expect("write");

    let cache = Arc::new(TableMetadataCache::new());
    let scope = test_scope();
    body_reads.store(0, Ordering::Relaxed);

    let results = futures::future::join_all((0..16).map(|_| {
        load_or_fetch_table_metadata(&file_io, &scope, location, Some(&cache), None)
    }))
    .await;
    for result in &results {
        assert!(result.is_ok(), "every waiter must get the shared result");
    }
    assert_eq!(
        body_reads.load(Ordering::Relaxed),
        1,
        "16 concurrent cold loads must dedup to exactly one body GET"
    );
    assert_eq!(cache.stats().body_fetches, 1);
    let first = &results[0];
    let first = first.as_ref().expect("first result");
    for result in &results[1..] {
        let result = result.as_ref().expect("waiter result");
        assert!(
            Arc::ptr_eq(first, result),
            "all waiters must receive the same Arc"
        );
    }

    fail_reads.store(true, Ordering::Relaxed);
    let other = "memory://warehouse/t/metadata/v2.metadata.json";
    sample_metadata("memory://warehouse/t")
        .write_to(&file_io, other)
        .await
        .expect("write v2");
    let failures = futures::future::join_all((0..16).map(|_| {
        load_or_fetch_table_metadata(&file_io, &scope, other, Some(&cache), None)
    }))
    .await;
    for result in &failures {
        assert!(
            result.is_err(),
            "injected read error must reach every waiter"
        );
    }

    fail_reads.store(false, Ordering::Relaxed);
    let retry = load_or_fetch_table_metadata(&file_io, &scope, other, Some(&cache), None)
        .await
        .expect("error was not cached; retry must succeed");
    assert_eq!(retry.location(), "memory://warehouse/t");
}

#[test]
fn _counting_storage_is_dyn_storage() {
    let _f: Arc<dyn StorageFactory> = Arc::new(CountingStorageFactory::new().0);
    let _ = Error::new(ErrorKind::Unexpected, "compile-only");
}
