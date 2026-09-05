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
    FormatVersion, Manifest, ManifestFile, ManifestList, SchemaId, SchemaRef, SnapshotRef,
    TableMetadataRef,
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
    Manifest(Arc<Manifest>),
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
                        CachedItem::Manifest(item) => estimate_manifest_weight(item),
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

        match cache_entry {
            CachedItem::Manifest(arc_manifest) => Ok(arc_manifest),
            _ => Err(Error::new(
                ErrorKind::Unexpected,
                format!("cached object for key '{key:?}' is not a Manifest"),
            )),
        }
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
        let manifest = manifest_file
            .load_manifest_with_schema_fallback(&self.file_io, schema_fallback)
            .await?;

        Ok(CachedItem::Manifest(Arc::new(manifest)))
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
mod tests {
    use std::fs;
    use std::sync::Arc;

    use minijinja::value::Value;
    use minijinja::{AutoEscape, Environment, context};
    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;
    use crate::TableIdent;
    use crate::io::{FileIO, OutputFile};
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestEntry,
        ManifestListWriter, ManifestStatus, ManifestWriterBuilder, Snapshot, Struct, TableMetadata,
    };
    use crate::table::Table;

    fn render_template(template: &str, ctx: Value) -> String {
        let mut env = Environment::new();
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.render_str(template, ctx).unwrap()
    }

    struct TableTestFixture {
        table_location: String,
        table: Table,
    }

    impl TableTestFixture {
        fn new() -> Self {
            let tmp_dir = TempDir::new().unwrap();
            let table_location = tmp_dir.path().join("table1");
            let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
            let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
            let table_metadata1_location = table_location.join("metadata/v1.json");

            let file_io = FileIO::new_with_fs();

            let table_metadata = {
                let template_json_str = fs::read_to_string(format!(
                    "{}/testdata/example_table_metadata_v2.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let metadata_json = render_template(&template_json_str, context! {
                    table_location => &table_location,
                    manifest_list_1_location => &manifest_list1_location,
                    manifest_list_2_location => &manifest_list2_location,
                    table_metadata_1_location => &table_metadata1_location,
                });
                serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
            };

            let table = Table::builder()
                .metadata(table_metadata)
                .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
                .file_io(file_io.clone())
                .metadata_location(table_metadata1_location.as_os_str().to_str().unwrap())
                .build()
                .unwrap();

            Self {
                table_location: table_location.to_str().unwrap().to_string(),
                table,
            }
        }

        fn next_manifest_file(&self) -> OutputFile {
            self.table
                .file_io()
                .new_output(format!(
                    "{}/metadata/manifest_{}.avro",
                    self.table_location,
                    Uuid::new_v4()
                ))
                .unwrap()
        }

        async fn setup_manifest_files(&mut self) {
            let current_snapshot = self.table.metadata().current_snapshot().unwrap();
            let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
            let current_partition_spec = self.table.metadata().default_partition_spec();

            // Write data files
            let mut writer = ManifestWriterBuilder::new(
                self.next_manifest_file(),
                Some(current_snapshot.snapshot_id()),
                None,
                current_schema.clone(),
                current_partition_spec.as_ref().clone(),
            )
            .build_v2_data();
            writer
                .add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(
                            DataFileBuilder::default()
                                .partition_spec_id(0)
                                .content(DataContentType::Data)
                                .file_path(format!("{}/1.parquet", &self.table_location))
                                .file_format(DataFileFormat::Parquet)
                                .file_size_in_bytes(100)
                                .record_count(1)
                                .partition(Struct::from_iter([Some(Literal::long(100))]))
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .unwrap();
            let data_file_manifest = writer.write_manifest_file().await.unwrap();

            // Write to manifest list
            let mut manifest_list_write = ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(current_snapshot.manifest_list())
                    .unwrap(),
                current_snapshot.snapshot_id(),
                current_snapshot.parent_snapshot_id(),
                current_snapshot.sequence_number(),
            );
            manifest_list_write
                .add_manifests(vec![data_file_manifest].into_iter())
                .unwrap();
            manifest_list_write.close().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_get_manifest_list_and_manifest_from_disabled_cache() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let object_cache = ObjectCache::with_disabled_cache(fixture.table.file_io().clone());

        let result_manifest_list = object_cache
            .get_manifest_list(
                fixture.table.metadata().current_snapshot().unwrap(),
                &fixture.table.metadata_ref(),
            )
            .await
            .unwrap();

        assert_eq!(result_manifest_list.entries().len(), 1);

        let manifest_file = result_manifest_list.entries().first().unwrap();
        let result_manifest = object_cache
            .get_manifest(manifest_file, None)
            .await
            .unwrap();

        assert_eq!(
            result_manifest
                .entries()
                .first()
                .unwrap()
                .file_path()
                .split("/")
                .last()
                .unwrap(),
            "1.parquet"
        );
    }

    /// SAF-001: a V1/legacy snapshot may omit `schema_id` (`Snapshot::schema_id` is `Option`).
    /// The default (enabled) `ObjectCache` must build its manifest-list key and load the list
    /// without panicking — matching the cache-disabled path, which already handles `None`.
    ///
    /// MUTATION (restore the `SchemaId` key element and `snapshot.schema_id().unwrap()` in
    /// `get_manifest_list`): key construction panics on this schema-id-less snapshot, before any
    /// I/O, so this test aborts instead of returning the manifest list.
    #[tokio::test]
    async fn test_get_manifest_list_from_default_cache_with_schemaless_snapshot() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let current = fixture.table.metadata().current_snapshot().unwrap();
        // A legacy snapshot that omits `schema_id`, reusing the real manifest-list path.
        let schemaless = Arc::new(
            Snapshot::builder()
                .with_snapshot_id(current.snapshot_id())
                .with_sequence_number(current.sequence_number())
                .with_timestamp_ms(current.timestamp_ms())
                .with_manifest_list(current.manifest_list())
                .with_summary(current.summary().clone())
                .build(),
        );
        assert_eq!(schemaless.schema_id(), None);

        let object_cache = ObjectCache::new(fixture.table.file_io().clone());
        let manifest_list = object_cache
            .get_manifest_list(&schemaless, &fixture.table.metadata_ref())
            .await
            .unwrap();

        assert_eq!(manifest_list.entries().len(), 1);
    }

    /// Wave A: weigher clamp floors at 1 and caps at `u32::MAX`.
    #[test]
    fn test_clamp_cache_weight_floor_and_cap() {
        assert_eq!(clamp_cache_weight(0), 1);
        assert_eq!(clamp_cache_weight(1), 1);
        assert_eq!(clamp_cache_weight(u32::MAX as u64), u32::MAX);
        assert_eq!(clamp_cache_weight(u32::MAX as u64 + 1), u32::MAX);
        assert_eq!(clamp_cache_weight(100), 100);

        // Relative scale of the entry-count estimate used for manifests.
        let one = clamp_cache_weight(1u64.saturating_mul(ROUGH_MANIFEST_ENTRY_BYTES));
        let ten = clamp_cache_weight(10u64.saturating_mul(ROUGH_MANIFEST_ENTRY_BYTES));
        assert!(
            ten > one,
            "more entries must weigh more: one={one} ten={ten}"
        );
        assert_eq!(one, ROUGH_MANIFEST_ENTRY_BYTES as u32);

        // Saturating multiply before clamp must not panic and must cap at u32::MAX.
        let overflow_weight =
            clamp_cache_weight(u64::MAX.saturating_mul(ROUGH_MANIFEST_LIST_ENTRY_BYTES));
        assert_eq!(overflow_weight, u32::MAX);
    }

    /// Wave A: loaded manifest / manifest-list weights use real estimates (≥ 1, and
    /// manifest entry weight scales with the entry count for a 1-entry fixture).
    #[tokio::test]
    async fn test_estimate_weights_on_loaded_manifests() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let object_cache = ObjectCache::new(fixture.table.file_io().clone());
        let manifest_list = object_cache
            .get_manifest_list(
                fixture
                    .table
                    .metadata()
                    .current_snapshot()
                    .expect("fixture must have a current snapshot"),
                &fixture.table.metadata_ref(),
            )
            .await
            .expect("manifest list must load");

        let list_weight = estimate_manifest_list_weight(&manifest_list);
        assert!(
            list_weight >= 1,
            "manifest list weight must floor at 1, got {list_weight}"
        );
        // List weight is entry_count × list-entry estimate — never the sum of child
        // manifest_length values (those are separate cached objects).
        let entry = manifest_list
            .entries()
            .first()
            .expect("fixture list has one entry");
        let expected_list = clamp_cache_weight(
            (manifest_list.entries().len() as u64)
                .max(1)
                .saturating_mul(ROUGH_MANIFEST_LIST_ENTRY_BYTES),
        );
        assert_eq!(
            list_weight, expected_list,
            "list weight must be entry_count × list-entry estimate, not child manifest_length"
        );
        // Sanity: when the child has a declared length, it must NOT equal the list weight
        // (unless by coincidence the length equals the list-entry constant).
        if entry.manifest_length > 0
            && entry.manifest_length as u64 != ROUGH_MANIFEST_LIST_ENTRY_BYTES
        {
            assert_ne!(
                list_weight,
                clamp_cache_weight(entry.manifest_length as u64),
                "list weight must not use child manifest_length"
            );
        }

        let manifest = object_cache
            .get_manifest(entry, None)
            .await
            .expect("manifest must load");
        let manifest_weight = estimate_manifest_weight(&manifest);
        let expected = clamp_cache_weight(
            (manifest.entries().len() as u64)
                .max(1)
                .saturating_mul(ROUGH_MANIFEST_ENTRY_BYTES),
        );
        assert_eq!(
            manifest_weight, expected,
            "manifest weight must be entry_count × rough bytes (floored)"
        );
    }

    #[tokio::test]
    async fn test_get_manifest_list_and_manifest_from_default_cache() {
        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let object_cache = ObjectCache::new(fixture.table.file_io().clone());

        // not in cache
        let result_manifest_list = object_cache
            .get_manifest_list(
                fixture.table.metadata().current_snapshot().unwrap(),
                &fixture.table.metadata_ref(),
            )
            .await
            .unwrap();

        assert_eq!(result_manifest_list.entries().len(), 1);

        // retrieve cached version
        let result_manifest_list = object_cache
            .get_manifest_list(
                fixture.table.metadata().current_snapshot().unwrap(),
                &fixture.table.metadata_ref(),
            )
            .await
            .unwrap();

        assert_eq!(result_manifest_list.entries().len(), 1);

        let manifest_file = result_manifest_list.entries().first().unwrap();

        // not in cache
        let result_manifest = object_cache
            .get_manifest(manifest_file, None)
            .await
            .unwrap();

        assert_eq!(
            result_manifest
                .entries()
                .first()
                .unwrap()
                .file_path()
                .split("/")
                .last()
                .unwrap(),
            "1.parquet"
        );

        // retrieve cached version
        let result_manifest = object_cache
            .get_manifest(manifest_file, None)
            .await
            .unwrap();

        assert_eq!(
            result_manifest
                .entries()
                .first()
                .unwrap()
                .file_path()
                .split("/")
                .last()
                .unwrap(),
            "1.parquet"
        );
    }
}
