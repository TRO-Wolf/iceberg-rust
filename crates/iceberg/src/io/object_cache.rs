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

use crate::io::FileIO;
use crate::spec::{
    FormatVersion, Manifest, ManifestEntry, ManifestFile, ManifestList, SchemaId, SchemaRef,
    SnapshotRef, TableMetadataRef, apply_manifest_list_context,
};
use crate::{Error, ErrorKind, Result};

const DEFAULT_CACHE_SIZE_BYTES: u64 = 32 * 1024 * 1024; // 32MB

/// Rough per-entry memory estimate for a parsed [`Manifest`].
///
/// `size_of_val` only measures the shallow `Manifest` shell (metadata + `Vec` header), not
/// the heap-backed entry list. Entry count × this constant is a stable capacity-accounting
/// proxy so large manifests weigh more than tiny ones under moka's weighted eviction.
///
/// Note: 768 is intentionally a coarse under-account for large nested partition stats;
/// correcting it is deferred (C1-SEC-003) — prefer re-tuning after real production
/// eviction metrics rather than over-weighting every list entry.
const ROUGH_MANIFEST_ENTRY_BYTES: u64 = 768;

/// Per-entry resident estimate for a parsed [`ManifestList`].
///
/// A manifest list holds only [`ManifestFile`] metadata rows (path, counts, partition
/// summaries) — not the child manifests themselves. Do **not** sum child
/// `manifest_length` values: those are on-disk sizes of separate objects and would
/// thrash the 32 MiB budget when one list points at many large manifests (C1-Q-001).
const ROUGH_MANIFEST_LIST_ENTRY_BYTES: u64 = 256;

/// Floor at 1 and clamp to `u32::MAX` for moka's weigher signature.
/// Accumulation paths must use saturating arithmetic before calling this (C1-Q-002).
fn clamp_cache_weight(bytes: u64) -> u32 {
    let clamped = bytes.clamp(1, u32::MAX as u64);
    // Domain is bounded to `[1, u32::MAX]` by the clamp above.
    clamped as u32
}

/// Estimated resident weight of a parsed manifest for the object cache.
fn estimate_manifest_weight(manifest: &Manifest) -> u32 {
    let n = (manifest.entries().len() as u64).max(1);
    clamp_cache_weight(n.saturating_mul(ROUGH_MANIFEST_ENTRY_BYTES))
}

/// Estimated resident weight of a parsed manifest list for the object cache.
///
/// Weight = `entry_count.max(1) × ROUGH_MANIFEST_LIST_ENTRY_BYTES`, then clamped to
/// `[1, u32::MAX]`. Uses saturating multiply so huge entry counts never panic.
fn estimate_manifest_list_weight(list: &ManifestList) -> u32 {
    let n = (list.entries().len() as u64).max(1);
    clamp_cache_weight(n.saturating_mul(ROUGH_MANIFEST_LIST_ENTRY_BYTES))
}

#[derive(Clone, Debug)]
pub(crate) enum CachedItem {
    ManifestList(Arc<ManifestList>),
    RawManifest(Arc<Manifest>),
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub(crate) enum CachedObjectKey {
    ManifestList((String, FormatVersion, Option<SchemaId>)),
    /// Manifest path plus optional fallback schema id used when the embedded
    /// `"schema"` key fails strict parse (QD). Path-only keys are wrong once
    /// parse depends on caller-supplied fallback (C1-SEC-002).
    Manifest((String, Option<SchemaId>)),
}

/// Caches metadata objects deserialized from immutable files
#[derive(Clone, Debug)]
pub struct ObjectCache {
    cache: moka::future::Cache<CachedObjectKey, CachedItem>,
    file_io: FileIO,
    cache_disabled: bool,
}

impl ObjectCache {
    /// Creates a new [`ObjectCache`]
    /// with the default cache size
    pub(crate) fn new(file_io: FileIO) -> Self {
        Self::new_with_capacity(file_io, DEFAULT_CACHE_SIZE_BYTES)
    }

    /// Creates a new [`ObjectCache`] with a specific cache size, shareable across tables.
    pub fn new_with_capacity(file_io: FileIO, cache_size_bytes: u64) -> Self {
        if cache_size_bytes == 0 {
            Self::with_disabled_cache(file_io)
        } else {
            Self {
                cache: moka::future::Cache::builder()
                    .weigher(|_, val: &CachedItem| match val {
                        CachedItem::ManifestList(item) => estimate_manifest_list_weight(item),
                        CachedItem::RawManifest(item) => estimate_manifest_weight(item),
                    })
                    .max_capacity(cache_size_bytes)
                    .build(),
                file_io,
                cache_disabled: false,
            }
        }
    }

    /// Creates a new [`ObjectCache`]
    /// with caching disabled
    pub(crate) fn with_disabled_cache(file_io: FileIO) -> Self {
        Self {
            cache: moka::future::Cache::new(0),
            file_io,
            cache_disabled: true,
        }
    }

    /// Retrieves an Arc [`Manifest`] from the cache
    /// or retrieves one from FileIO and parses it if not present.
    ///
    /// `schema_fallback` is the table/snapshot schema used when the manifest's embedded
    /// `"schema"` key fails strict parse (DuckDB malformation tolerance).
    pub(crate) async fn get_manifest(
        &self,
        manifest_file: &ManifestFile,
        schema_fallback: Option<SchemaRef>,
    ) -> Result<Arc<Manifest>> {
        if self.cache_disabled {
            return manifest_file
                .load_manifest_with_schema_fallback(&self.file_io, schema_fallback)
                .await
                .map(Arc::new);
        }

        let fallback_schema_id = schema_fallback.as_ref().map(|s| s.schema_id());
        let key =
            CachedObjectKey::Manifest((manifest_file.manifest_path.clone(), fallback_schema_id));

        let cache_entry = self
            .cache
            .entry_by_ref(&key)
            .or_try_insert_with(self.fetch_and_parse_manifest(manifest_file, schema_fallback))
            .await
            .map_err(|err| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to load manifest {}", manifest_file.manifest_path),
                )
                .with_source(err)
            })?
            .into_value();

        let raw_manifest = match cache_entry {
            CachedItem::RawManifest(arc_manifest) => arc_manifest,
            _ => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("cached object for key '{key:?}' is not a RawManifest"),
                ));
            }
        };

        let mut entries: Vec<ManifestEntry> = raw_manifest
            .entries()
            .iter()
            .map(|entry| entry.as_ref().clone())
            .collect();
        apply_manifest_list_context(&mut entries, manifest_file)?;

        Ok(Arc::new(Manifest::new(
            raw_manifest.metadata().clone(),
            entries,
        )))
    }

    /// Retrieves an Arc [`ManifestList`] from the cache
    /// or retrieves one from FileIO and parses it if not present
    pub(crate) async fn get_manifest_list(
        &self,
        snapshot: &SnapshotRef,
        table_metadata: &TableMetadataRef,
    ) -> Result<Arc<ManifestList>> {
        if self.cache_disabled {
            return snapshot
                .load_manifest_list(&self.file_io, table_metadata)
                .await
                .map(Arc::new);
        }

        // `Snapshot::schema_id` is `Option`: V1/legacy snapshots may omit it. The manifest-list
        // path already uniquely identifies the cache entry, so key on the `Option` directly
        // rather than unwrapping (which panicked on a schema-id-less snapshot).
        let key = CachedObjectKey::ManifestList((
            snapshot.manifest_list().to_string(),
            table_metadata.format_version,
            snapshot.schema_id(),
        ));
        let cache_entry = self
            .cache
            .entry_by_ref(&key)
            .or_try_insert_with(self.fetch_and_parse_manifest_list(snapshot, table_metadata))
            .await
            .map_err(|err| {
                Arc::try_unwrap(err).unwrap_or_else(|err| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Failed to load manifest list in cache",
                    )
                    .with_source(err)
                })
            })?
            .into_value();

        match cache_entry {
            CachedItem::ManifestList(arc_manifest_list) => Ok(arc_manifest_list),
            _ => Err(Error::new(
                ErrorKind::Unexpected,
                format!("cached object for path '{key:?}' is not a manifest list"),
            )),
        }
    }

    async fn fetch_and_parse_manifest(
        &self,
        manifest_file: &ManifestFile,
        schema_fallback: Option<SchemaRef>,
    ) -> Result<CachedItem> {
        let (metadata, entries) = manifest_file
            .load_manifest_parts_with_schema_fallback(&self.file_io, schema_fallback)
            .await?;

        Ok(CachedItem::RawManifest(Arc::new(Manifest::new(
            metadata, entries,
        ))))
    }

    async fn fetch_and_parse_manifest_list(
        &self,
        snapshot: &SnapshotRef,
        table_metadata: &TableMetadataRef,
    ) -> Result<CachedItem> {
        let manifest_list = snapshot
            .load_manifest_list(&self.file_io, table_metadata)
            .await?;

        Ok(CachedItem::ManifestList(Arc::new(manifest_list)))
    }
}

#[cfg(test)]
#[path = "object_cache_tests.rs"]
mod tests;
