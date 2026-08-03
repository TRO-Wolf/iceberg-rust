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

//! Session-scoped metadata-pointer cache (FK4.1 / scout #7).
//!
//! # Contract (v1)
//!
//! * **Key** = metadata-file location string equality only (Iceberg catalog pointer).
//! * **Opt-in, default OFF** — catalogs hold `Option<Arc<TableMetadataCache>>`; no global
//!   or thread-local state.
//! * **Fail closed** — on any mismatch (unknown location, or optional object-version guard
//!   disagreeing when *both* sides present) → full body GET + re-parse; never soft-reuse.
//! * **Object version / ETag** is a free extra guard when a store or catalog service hands
//!   it (Glue `version_id`, S3 ETag, …). It is **never** the sole check.
//!
//! Inject via catalog builders (e.g. [`crate::memory::MemoryCatalogBuilder::with_table_metadata_cache`])
//! and consult on the `load_table` path through [`load_or_fetch_table_metadata`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::Result;
use crate::io::FileIO;
use crate::spec::{TableMetadata, TableMetadataRef};

/// Snapshot of cache traffic counters (for tests / op-count pins).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TableMetadataCacheStats {
    /// Lookups that returned a cached `Arc` without a body GET.
    pub hits: u64,
    /// Lookups that fell through to a body GET (miss or fail-closed guard mismatch).
    pub misses: u64,
    /// Times the helper performed `TableMetadata::read_from` (body GET + parse).
    pub body_fetches: u64,
}

#[derive(Debug, Clone)]
struct CachedEntry {
    metadata: TableMetadataRef,
    /// Optional object-store ETag / catalog service version token.
    object_version: Option<String>,
}

/// Session-scoped cache: `metadata_location` → parsed [`TableMetadata`].
///
/// Share one `Arc<TableMetadataCache>` across catalog instances in a session if desired.
/// Construct with [`TableMetadataCache::new`] and inject at catalog construction; catalogs
/// without an injection never consult a cache (default OFF).
#[derive(Debug, Default)]
pub struct TableMetadataCache {
    entries: RwLock<HashMap<String, CachedEntry>>,
    hits: AtomicU64,
    misses: AtomicU64,
    body_fetches: AtomicU64,
}

impl TableMetadataCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Counters for hits / misses / body fetches since construction (or last reset).
    pub fn stats(&self) -> TableMetadataCacheStats {
        TableMetadataCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            body_fetches: self.body_fetches.load(Ordering::Relaxed),
        }
    }

    /// Reset traffic counters (entries are left intact). Intended for tests.
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.body_fetches.store(0, Ordering::Relaxed);
    }

    /// Number of retained location keys.
    pub fn len(&self) -> usize {
        self.entries.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop the entry for `metadata_location` if present.
    pub fn invalidate(&self, metadata_location: &str) {
        let mut guard = self.entries.write().unwrap_or_else(|e| e.into_inner());
        guard.remove(metadata_location);
    }

    /// Drop every entry (fail-closed invalidation when a table identity cannot be resolved
    /// to a single location, or for `Catalog::invalidate_table` without a reverse index).
    pub fn clear(&self) {
        let mut guard = self.entries.write().unwrap_or_else(|e| e.into_inner());
        guard.clear();
    }

    /// Insert or replace the entry for `metadata_location`.
    ///
    /// `object_version` is retained as an optional extra guard for later lookups; it is never
    /// required for a hit when the caller omits a guard.
    pub fn put(
        &self,
        metadata_location: impl Into<String>,
        metadata: TableMetadataRef,
        object_version: Option<String>,
    ) {
        let mut guard = self.entries.write().unwrap_or_else(|e| e.into_inner());
        guard.insert(metadata_location.into(), CachedEntry {
            metadata,
            object_version,
        });
    }

    /// Look up by location string equality, with optional object-version fail-closed check.
    ///
    /// Returns `Some` only when:
    /// 1. an entry exists for exact `metadata_location`, AND
    /// 2. the optional object-version guard does not disagree (see below).
    ///
    /// Guard rules (version is **never** the sole check — location must match first):
    /// * both cached and caller `Some` and equal → hit
    /// * both `Some` and **not** equal → miss (fail closed)
    /// * caller `None` → hit on location alone (guard not required)
    /// * cached `None`, caller `Some` → hit on location, and **learn** the caller's guard
    ///   into the entry so a later different version fail-closes (create-seed often puts
    ///   `None`; a subsequent service version must still be able to arm the guard)
    ///
    /// Otherwise returns `None` (caller must full-fetch).
    pub fn lookup(
        &self,
        metadata_location: &str,
        object_version: Option<&str>,
    ) -> Option<TableMetadataRef> {
        // Fast path under read lock when no learn-upgrade is needed.
        {
            let guard = self.entries.read().unwrap_or_else(|e| e.into_inner());
            let entry = guard.get(metadata_location)?;
            match (entry.object_version.as_deref(), object_version) {
                (Some(cached_v), Some(given_v)) if cached_v != given_v => return None,
                (Some(_), Some(_)) | (Some(_), None) | (None, None) => {
                    return Some(entry.metadata.clone());
                }
                (None, Some(_)) => {
                    // Need write lock to learn the guard — drop read lock first.
                }
            }
        }

        // Learn path: cached guard absent, caller supplied one. Re-check under write lock.
        let mut guard = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let entry = guard.get_mut(metadata_location)?;
        match (entry.object_version.as_deref(), object_version) {
            (Some(cached_v), Some(given_v)) if cached_v != given_v => None,
            (None, Some(given_v)) => {
                entry.object_version = Some(given_v.to_string());
                Some(entry.metadata.clone())
            }
            _ => Some(entry.metadata.clone()),
        }
    }

    fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_body_fetch(&self) {
        self.body_fetches.fetch_add(1, Ordering::Relaxed);
    }
}

/// Load table metadata for `metadata_location`, consulting `cache` when present.
///
/// * `cache == None` — always body GET + parse (default OFF path); result is **not** stored.
/// * `cache == Some` — location-key lookup; on hit return shared `Arc` with **zero** body GET;
///   on miss or fail-closed guard mismatch, body GET, parse, `put`, return.
///
/// `object_version` is the free extra guard from a catalog service / object store when available.
pub async fn load_or_fetch_table_metadata(
    file_io: &FileIO,
    metadata_location: &str,
    cache: Option<&TableMetadataCache>,
    object_version: Option<&str>,
) -> Result<TableMetadataRef> {
    if let Some(cache) = cache {
        if let Some(hit) = cache.lookup(metadata_location, object_version) {
            cache.record_hit();
            return Ok(hit);
        }
        cache.record_miss();
        let metadata = fetch_table_metadata(file_io, metadata_location).await?;
        cache.record_body_fetch();
        let metadata = Arc::new(metadata);
        cache.put(
            metadata_location.to_string(),
            metadata.clone(),
            object_version.map(str::to_string),
        );
        return Ok(metadata);
    }

    let metadata = fetch_table_metadata(file_io, metadata_location).await?;
    Ok(Arc::new(metadata))
}

async fn fetch_table_metadata(file_io: &FileIO, metadata_location: &str) -> Result<TableMetadata> {
    TableMetadata::read_from(file_io, metadata_location).await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

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

    /// Storage wrapper that counts body `read` calls (op-count injector for FK4.1 pins).
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
            // Streaming reads also count as body GETs for the pin.
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
            // Write path uses the inner storage directly so close() lands in the shared map;
            // FileIO still goes through this factory's build() for the catalog FileIO.
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
        body_reads.store(0, Ordering::Relaxed);

        let first = load_or_fetch_table_metadata(&file_io, location, Some(&cache), None)
            .await
            .expect("first load");
        let reads_after_first = body_reads.load(Ordering::Relaxed);
        assert_eq!(reads_after_first, 1, "first load must body-GET once");
        assert_eq!(cache.stats().body_fetches, 1);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        let second = load_or_fetch_table_metadata(&file_io, location, Some(&cache), None)
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

        body_reads.store(0, Ordering::Relaxed);
        let _a = load_or_fetch_table_metadata(&file_io, location, None, None)
            .await
            .expect("load a");
        let _b = load_or_fetch_table_metadata(&file_io, location, None, None)
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
        body_reads.store(0, Ordering::Relaxed);

        let _ = load_or_fetch_table_metadata(&file_io, location, Some(&cache), Some("v1"))
            .await
            .expect("seed");
        assert_eq!(body_reads.load(Ordering::Relaxed), 1);

        // Same location, different service version → fail closed, full re-fetch.
        let _ = load_or_fetch_table_metadata(&file_io, location, Some(&cache), Some("v2"))
            .await
            .expect("guard mismatch must re-fetch");
        assert_eq!(
            body_reads.load(Ordering::Relaxed),
            2,
            "version mismatch must not soft-reuse"
        );
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 0);

        // Matching guard → hit.
        let _ = load_or_fetch_table_metadata(&file_io, location, Some(&cache), Some("v2"))
            .await
            .expect("hit");
        assert_eq!(body_reads.load(Ordering::Relaxed), 2);
        assert_eq!(cache.stats().hits, 1);
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
        let _ = load_or_fetch_table_metadata(&file_io, loc1, Some(&cache), None)
            .await
            .expect("l1");
        let _ = load_or_fetch_table_metadata(&file_io, loc2, Some(&cache), None)
            .await
            .expect("l2");
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 0);
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
        let _ = load_or_fetch_table_metadata(&file_io, location, Some(&cache), None)
            .await
            .expect("seed");
        cache.invalidate(location);
        body_reads.store(0, Ordering::Relaxed);
        let _ = load_or_fetch_table_metadata(&file_io, location, Some(&cache), None)
            .await
            .expect("after invalidate");
        assert_eq!(body_reads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn version_never_sole_check_location_required() {
        let cache = TableMetadataCache::new();
        let meta = Arc::new(sample_metadata("memory://warehouse/t"));
        cache.put(
            "memory://a".to_string(),
            meta.clone(),
            Some("etag-1".to_string()),
        );
        // Wrong location even with matching version → miss.
        assert!(cache.lookup("memory://b", Some("etag-1")).is_none());
        // Right location, matching version → hit.
        assert!(cache.lookup("memory://a", Some("etag-1")).is_some());
        // Right location, no caller version → hit (version not required).
        assert!(cache.lookup("memory://a", None).is_some());
    }

    /// Create-seed often puts `object_version=None`. The first caller-supplied version must
    /// **learn** into the entry; a later different version must fail-close (miss).
    #[test]
    fn learn_version_guard_then_mismatch_fail_closed() {
        let cache = TableMetadataCache::new();
        let meta = Arc::new(sample_metadata("memory://warehouse/t"));
        cache.put("memory://a".to_string(), meta, None);

        assert!(
            cache.lookup("memory://a", Some("v1")).is_some(),
            "first versioned lookup on unguarded entry must hit and learn"
        );
        // Learned: mismatch must miss.
        assert!(
            cache.lookup("memory://a", Some("v2")).is_none(),
            "after learn, disagreeing version must fail closed"
        );
        assert!(
            cache.lookup("memory://a", Some("v1")).is_some(),
            "matching learned version still hits"
        );
    }

    #[test]
    fn _counting_storage_is_dyn_storage() {
        let _f: Arc<dyn StorageFactory> = Arc::new(CountingStorageFactory::new().0);
        let _ = Error::new(ErrorKind::Unexpected, "compile-only");
    }
}
