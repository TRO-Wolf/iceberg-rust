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

use uuid::Uuid;

use crate::io::{
    FileIO, S3_ACCESS_KEY_ID, S3_ASSUME_ROLE_ARN, S3_ASSUME_ROLE_SESSION_NAME, S3_REGION,
};
use crate::spec::{TableMetadata, TableMetadataRef};
use crate::{Error, Result};

const DEFAULT_MAX_ENTRIES: u64 = 1024;

const CREDENTIAL_CONTEXT_PROP_KEYS: &[&str] = &[
    "aws_access_key_id",
    "profile_name",
    "region_name",
    S3_ACCESS_KEY_ID,
    S3_REGION,
    S3_ASSUME_ROLE_ARN,
    S3_ASSUME_ROLE_SESSION_NAME,
];

/// Snapshot of cache traffic counters (for tests / op-count pins).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TableMetadataCacheStats {
    /// Lookups that returned a cached `Arc` without a body GET.
    pub hits: u64,
    /// Lookups that fell through to a body GET (miss or fail-closed guard mismatch).
    pub misses: u64,
    /// Times the helper performed `TableMetadata::read_from` (body GET + parse).
    pub body_fetches: u64,
    #[allow(missing_docs)]
    pub evictions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct CacheScope {
    catalog_identity: String,
    credential_context: String,
}

#[allow(missing_docs)]
impl CacheScope {
    pub fn new(catalog_identity: impl Into<String>, credential_context: impl Into<String>) -> Self {
        Self {
            catalog_identity: catalog_identity.into(),
            credential_context: credential_context.into(),
        }
    }

    pub fn isolated(catalog_identity: impl Into<String>) -> Self {
        Self::new(catalog_identity, Self::unique_instance_context())
    }

    pub fn unique_instance_context() -> String {
        format!("instance:{}", Uuid::new_v4().simple())
    }

    pub fn credential_context_from_props(props: &HashMap<String, String>) -> Option<String> {
        let mut parts: Vec<String> = CREDENTIAL_CONTEXT_PROP_KEYS
            .iter()
            .filter_map(|key| props.get(*key).map(|value| format!("{key}={value}")))
            .collect();
        parts.sort();
        (!parts.is_empty()).then(|| parts.join(";"))
    }

    pub fn for_catalog(
        catalog_identity: impl Into<String>,
        credential_context: Option<String>,
        props: &HashMap<String, String>,
    ) -> Self {
        match credential_context.or_else(|| Self::credential_context_from_props(props)) {
            Some(context) => Self::new(catalog_identity, context),
            None => Self::isolated(catalog_identity),
        }
    }

    pub fn catalog_identity(&self) -> &str {
        &self.catalog_identity
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    scope: CacheScope,
    location: String,
}

#[derive(Debug, Clone)]
struct CachedEntry {
    metadata: TableMetadataRef,
    object_version: Option<String>,
}

impl CachedEntry {
    fn version_conflicts(&self, object_version: Option<&str>) -> bool {
        match (self.object_version.as_deref(), object_version) {
            (Some(cached), Some(given)) => cached != given,
            _ => false,
        }
    }
}

fn key(scope: &CacheScope, metadata_location: &str) -> CacheKey {
    CacheKey {
        scope: scope.clone(),
        location: metadata_location.to_string(),
    }
}

#[derive(Debug)]
#[allow(missing_docs)]
pub struct TableMetadataCache {
    entries: moka::future::Cache<CacheKey, CachedEntry>,
    hits: AtomicU64,
    misses: AtomicU64,
    body_fetches: AtomicU64,
    evictions: Arc<AtomicU64>,
}

impl Default for TableMetadataCache {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl TableMetadataCache {
    pub fn new() -> Self {
        Self::with_max_entries(DEFAULT_MAX_ENTRIES)
    }

    pub fn with_max_entries(max_entries: u64) -> Self {
        let evictions = Arc::new(AtomicU64::new(0));
        let entries = {
            let evictions = Arc::clone(&evictions);
            moka::future::Cache::builder()
                .max_capacity(max_entries)
                .eviction_listener(move |_key, _value, cause| {
                    if cause == moka::notification::RemovalCause::Size {
                        evictions.fetch_add(1, Ordering::Relaxed);
                    }
                })
                .build()
        };
        Self {
            entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            body_fetches: AtomicU64::new(0),
            evictions,
        }
    }

    /// Counters for hits / misses / body fetches since construction (or last reset).
    pub fn stats(&self) -> TableMetadataCacheStats {
        TableMetadataCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            body_fetches: self.body_fetches.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    /// Reset traffic counters (entries are left intact). Intended for tests.
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.body_fetches.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }

    /// Number of retained location keys.
    pub fn len(&self) -> usize {
        usize::try_from(self.entries.entry_count()).unwrap_or(usize::MAX)
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every entry (fail-closed invalidation when a table identity cannot be resolved
    /// to a single location, or for `Catalog::invalidate_table` without a reverse index).
    pub fn clear(&self) {
        self.entries.invalidate_all();
    }

    #[allow(missing_docs)]
    pub async fn invalidate(&self, scope: &CacheScope, metadata_location: &str) {
        self.entries
            .invalidate(&key(scope, metadata_location))
            .await;
    }

    #[allow(missing_docs)]
    pub async fn run_pending_tasks(&self) {
        self.entries.run_pending_tasks().await;
    }

    #[allow(missing_docs)]
    pub async fn put(
        &self,
        scope: &CacheScope,
        metadata_location: impl Into<String>,
        metadata: TableMetadataRef,
        object_version: Option<String>,
    ) {
        let metadata_location = metadata_location.into();
        self.entries
            .insert(key(scope, &metadata_location), CachedEntry {
                metadata,
                object_version,
            })
            .await;
    }

    #[allow(missing_docs)]
    pub async fn lookup(
        &self,
        scope: &CacheScope,
        metadata_location: &str,
        object_version: Option<&str>,
    ) -> Option<TableMetadataRef> {
        let key = key(scope, metadata_location);
        let entry = self.entries.get(&key).await?;
        if entry.version_conflicts(object_version) {
            return None;
        }
        self.arm_version(&key, &entry, object_version).await;
        Some(entry.metadata)
    }

    async fn arm_version(&self, key: &CacheKey, entry: &CachedEntry, object_version: Option<&str>) {
        if entry.object_version.is_none()
            && let Some(given) = object_version
        {
            let mut armed = entry.clone();
            armed.object_version = Some(given.to_string());
            self.entries.insert(key.clone(), armed).await;
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

#[allow(missing_docs)]
pub async fn load_or_fetch_table_metadata(
    file_io: &FileIO,
    scope: &CacheScope,
    metadata_location: &str,
    cache: Option<&TableMetadataCache>,
    object_version: Option<&str>,
) -> Result<TableMetadataRef> {
    let Some(cache) = cache else {
        let metadata = TableMetadata::read_from(file_io, metadata_location).await?;
        return Ok(Arc::new(metadata));
    };

    if let Some(hit) = cache.lookup(scope, metadata_location, object_version).await {
        cache.record_hit();
        return Ok(hit);
    }
    cache.record_miss();

    for _ in 0..3 {
        cache.invalidate(scope, metadata_location).await;
        let init = async {
            let metadata = TableMetadata::read_from(file_io, metadata_location).await?;
            cache.record_body_fetch();
            Ok::<CachedEntry, Error>(CachedEntry {
                metadata: Arc::new(metadata),
                object_version: object_version.map(str::to_string),
            })
        };
        match cache
            .entries
            .try_get_with(key(scope, metadata_location), init)
            .await
        {
            Ok(entry) if !entry.version_conflicts(object_version) => {
                let key = key(scope, metadata_location);
                cache.arm_version(&key, &entry, object_version).await;
                return Ok(entry.metadata);
            }
            Ok(_) => continue,
            Err(err) => {
                return Err(Error::new(err.kind(), err.message()).with_retryable(err.retryable()));
            }
        }
    }

    let metadata = TableMetadata::read_from(file_io, metadata_location).await?;
    cache.record_body_fetch();
    let metadata = Arc::new(metadata);
    cache
        .put(
            scope,
            metadata_location.to_string(),
            metadata.clone(),
            object_version.map(str::to_string),
        )
        .await;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    include!("table_metadata_cache_tests.rs");
}
