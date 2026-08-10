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

//! A [moka](https://github.com/moka-rs/moka)-backed [`ObjectCacheProvide`] implementation.
//!
//! # Capacity is a byte budget, not an entry count
//!
//! `moka::sync::Cache::new(n)` sizes a cache at `n` **entries**: with no weigher configured,
//! moka weighs every entry as `1` (`moka` 0.12.15, `src/sync/base_cache.rs`:
//! `fn weigh(..) { self.weigher.as_ref().map_or(1, |w| w(key, value)) }`). The caches built here
//! therefore install an explicit `weigher` and treat [`MokaObjectCacheProvider::new_with_capacity`]
//! as a **byte** budget, matching the core crate's object cache.

use std::hash::Hash;
use std::sync::Arc;

use iceberg::cache::{ObjectCache, ObjectCacheProvide};
use iceberg::spec::{Manifest, ManifestList};

/// Default byte budget applied to **each** of the two caches this provider owns.
///
/// The provider keeps a manifest cache and a manifest-list cache as separate moka caches
/// (they hold different value types), so the aggregate resident ceiling of a default
/// provider is `2 × DEFAULT_CACHE_SIZE_BYTES` — 64 MiB, not 32 MiB. Callers that need a
/// specific aggregate ceiling should pass their own per-cache budget to
/// [`MokaObjectCacheProvider::new_with_capacity`].
const DEFAULT_CACHE_SIZE_BYTES: u64 = 32 * 1024 * 1024; // 32MiB

/// Rough per-entry memory estimate for a parsed [`Manifest`].
///
/// **`crates/iceberg/src/io/object_cache.rs` is the source of truth for this constant and for
/// [`ROUGH_MANIFEST_LIST_ENTRY_BYTES`].** The duplication here is deliberate: those helpers are
/// crate-private to `iceberg` and this is a separate crate, so the two copies must be kept in
/// step by hand — change one, change the other, or the pluggable provider and the built-in
/// object cache will account for the same table differently.
const ROUGH_MANIFEST_ENTRY_BYTES: u64 = 768;

/// Per-entry resident estimate for a parsed [`ManifestList`].
///
/// A manifest list holds only `ManifestFile` metadata rows (path, counts, partition summaries) —
/// not the child manifests themselves. Do **not** sum child `manifest_length` values: those are
/// the on-disk sizes of separate objects (each cached under its own key in the manifest cache)
/// and folding them in here would thrash the budget whenever one list points at many large
/// manifests. See the C1-Q-001 note in `crates/iceberg/src/io/object_cache.rs`, which owns this
/// constant.
const ROUGH_MANIFEST_LIST_ENTRY_BYTES: u64 = 256;

/// Floor at 1 and clamp to `u32::MAX`, the width of moka's weigher return type.
///
/// Accumulation paths must use saturating arithmetic *before* calling this.
fn clamp_cache_weight(bytes: u64) -> u32 {
    let clamped = bytes.clamp(1, u32::MAX as u64);
    // The clamp above bounds the domain to `[1, u32::MAX]`, so this cast cannot truncate.
    clamped as u32
}

/// `entry_count × per_entry_bytes`, floored at one entry and clamped into the weigher's `u32`.
///
/// The `max(1)` keeps an empty manifest/list from weighing the same as a one-row one purely
/// because of the clamp floor: an empty object still costs its shell plus its schema.
fn weight_from_entry_count(entry_count: usize, per_entry_bytes: u64) -> u32 {
    let n = (entry_count as u64).max(1);
    clamp_cache_weight(n.saturating_mul(per_entry_bytes))
}

/// Estimated resident weight of a parsed manifest.
fn estimate_manifest_weight(manifest: &Manifest) -> u32 {
    weight_from_entry_count(manifest.entries().len(), ROUGH_MANIFEST_ENTRY_BYTES)
}

/// Estimated resident weight of a parsed manifest list.
fn estimate_manifest_list_weight(list: &ManifestList) -> u32 {
    weight_from_entry_count(list.entries().len(), ROUGH_MANIFEST_LIST_ENTRY_BYTES)
}

/// Builds a moka cache whose `max_capacity` is denominated in the units the `weigher` returns.
///
/// A `cache_size_bytes` of `0` yields a disabled cache, mirroring
/// `ObjectCache::with_disabled_cache` in the core crate: moka short-circuits both `get` and
/// `insert` when `max_capacity == Some(0)`.
fn build_weighted_cache<K, V>(
    cache_size_bytes: u64,
    weigher: fn(&V) -> u32,
) -> moka::sync::Cache<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    if cache_size_bytes == 0 {
        return moka::sync::Cache::new(0);
    }

    moka::sync::Cache::builder()
        .weigher(move |_key, value: &V| weigher(value))
        .max_capacity(cache_size_bytes)
        .build()
}

struct MokaObjectCache<K, V>(moka::sync::Cache<K, V>);

impl<K, V> ObjectCache<K, V> for MokaObjectCache<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn get(&self, key: &K) -> Option<V> {
        self.0.get(key)
    }

    fn set(&self, key: K, value: V) {
        self.0.insert(key, value);
    }
}

/// A cache provider that uses Moka for caching objects.
///
/// # Capacity semantics
///
/// Both caches are weighted by an estimate of the parsed object's resident size, so the
/// capacity passed to [`MokaObjectCacheProvider::new_with_capacity`] is a **byte budget per
/// cache**. The estimate is `entry_count × a per-entry constant`; the constants are owned by
/// `crates/iceberg/src/io/object_cache.rs` and duplicated here across the crate boundary.
///
/// # A miss is always safe
///
/// An object whose estimated weight exceeds the budget is rejected by moka outright rather
/// than evicting the rest of the cache (`moka` 0.12.15, `src/sync/base_cache.rs`: "The candidate
/// is too big to fit in the cache. Reject it."), so it never becomes resident. That is benign:
/// [`ObjectCache::get`] returns `Option<V>` and [`ObjectCache::set`] returns `()`, so every
/// caller must already handle a miss, and manifests and manifest lists are immutable once
/// written — re-reading one yields an equal value. This cache is a pure memoization port;
/// correctness never depends on a hit. It does mean a single manifest larger than the whole
/// budget is permanently uncached, which is a throughput cost, not a correctness one — raise
/// the budget if you see it.
pub struct MokaObjectCacheProvider {
    manifest_cache: MokaObjectCache<String, Arc<Manifest>>,
    manifest_list_cache: MokaObjectCache<String, Arc<ManifestList>>,
}

impl Default for MokaObjectCacheProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MokaObjectCacheProvider {
    /// Creates a new `MokaObjectCacheProvider` with the default per-cache byte budget of
    /// 32 MiB each (so a 64 MiB aggregate ceiling).
    pub fn new() -> Self {
        Self::new_with_capacity(DEFAULT_CACHE_SIZE_BYTES)
    }

    /// Creates a new `MokaObjectCacheProvider` with `cache_size_bytes` as the byte budget of
    /// **each** of the two caches (so the aggregate ceiling is twice this value).
    ///
    /// Passing `0` disables caching entirely, as `ObjectCache::with_disabled_cache` does in the
    /// core crate.
    pub fn new_with_capacity(cache_size_bytes: u64) -> Self {
        let manifest_cache = MokaObjectCache(build_weighted_cache::<String, Arc<Manifest>>(
            cache_size_bytes,
            |manifest| estimate_manifest_weight(manifest),
        ));
        let manifest_list_cache = MokaObjectCache(
            build_weighted_cache::<String, Arc<ManifestList>>(cache_size_bytes, |list| {
                estimate_manifest_list_weight(list)
            }),
        );

        Self {
            manifest_cache,
            manifest_list_cache,
        }
    }

    /// Set the cache for manifests.
    ///
    /// **The caller's builder wins.** The supplied cache is used exactly as configured: this
    /// provider does not attach a weigher to it and does not re-interpret its `max_capacity`.
    /// A cache built with `moka::sync::Cache::new(n)` — or with any builder that omits
    /// `.weigher(..)` — is therefore bounded by **entry count**, not bytes, because moka weighs
    /// every entry as `1` without a weigher. To keep the byte semantics of
    /// [`Self::new_with_capacity`], supply a cache whose builder sets both `.weigher(..)` and
    /// `.max_capacity(..)`.
    pub fn with_manifest_cache(mut self, cache: moka::sync::Cache<String, Arc<Manifest>>) -> Self {
        self.manifest_cache = MokaObjectCache(cache);
        self
    }

    /// Set the cache for manifest lists.
    ///
    /// **The caller's builder wins** — see [`Self::with_manifest_cache`] for what that means for
    /// byte-vs-entry-count capacity.
    pub fn with_manifest_list_cache(
        mut self,
        cache: moka::sync::Cache<String, Arc<ManifestList>>,
    ) -> Self {
        self.manifest_list_cache = MokaObjectCache(cache);
        self
    }
}

impl ObjectCacheProvide for MokaObjectCacheProvider {
    fn manifest_cache(&self) -> &dyn ObjectCache<String, Arc<Manifest>> {
        &self.manifest_cache
    }

    fn manifest_list_cache(&self) -> &dyn ObjectCache<String, Arc<ManifestList>> {
        &self.manifest_list_cache
    }
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, ManifestContentType,
        ManifestEntry, ManifestMetadata, ManifestStatus, NestedField, PartitionSpec, PrimitiveType,
        Schema, Struct, Type,
    };

    use super::*;

    /// A manifest carrying exactly `entry_count` metadata-only entries.
    fn manifest_with_entries(entry_count: usize) -> Arc<Manifest> {
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("build a one-column test schema");

        let metadata = ManifestMetadata {
            schema: Arc::new(schema),
            schema_id: 0,
            partition_spec: PartitionSpec::unpartition_spec(),
            format_version: FormatVersion::V2,
            content: ManifestContentType::Data,
        };

        let entries = (0..entry_count)
            .map(|i| {
                let data_file = DataFileBuilder::default()
                    .content(DataContentType::Data)
                    .file_path(format!("memory:/t/data/{i}.parquet"))
                    .file_format(DataFileFormat::Parquet)
                    .partition(Struct::empty())
                    .partition_spec_id(0)
                    .record_count(1)
                    .file_size_in_bytes(100)
                    .build()
                    .expect("build a metadata-only data file");

                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(data_file)
                    .build()
            })
            .collect();

        Arc::new(Manifest::new(metadata, entries))
    }

    /// Writer schema of the fixture below: the V2 `manifest_file` record.
    ///
    /// Field names and order match `_serde::ManifestFileV2` in
    /// `crates/iceberg/src/spec/manifest_list.rs`; `partitions` / `key_metadata` are present as
    /// nullable unions because that struct's `Option` fields carry no serde default and would
    /// otherwise fail to deserialize.
    const MANIFEST_FILE_V2_SCHEMA: &str = concat!(
        r#"{"type":"record","name":"manifest_file","fields":["#,
        r#"{"name":"manifest_path","type":"string"},"#,
        r#"{"name":"manifest_length","type":"long"},"#,
        r#"{"name":"partition_spec_id","type":"int"},"#,
        r#"{"name":"content","type":"int"},"#,
        r#"{"name":"sequence_number","type":"long"},"#,
        r#"{"name":"min_sequence_number","type":"long"},"#,
        r#"{"name":"added_snapshot_id","type":"long"},"#,
        r#"{"name":"added_files_count","type":"int"},"#,
        r#"{"name":"existing_files_count","type":"int"},"#,
        r#"{"name":"deleted_files_count","type":"int"},"#,
        r#"{"name":"added_rows_count","type":"long"},"#,
        r#"{"name":"existing_rows_count","type":"long"},"#,
        r#"{"name":"deleted_rows_count","type":"long"},"#,
        r#"{"name":"partitions","type":["null",{"type":"array","items":"#,
        r#"{"type":"record","name":"field_summary","fields":["#,
        r#"{"name":"contains_null","type":"boolean"}]}}],"default":null},"#,
        r#"{"name":"key_metadata","type":["null","bytes"],"default":null}]}"#,
    );

    /// Sync marker of the fixture container file. Any 16 bytes will do, as long as the header
    /// and every block trailer agree.
    const FIXTURE_SYNC_MARKER: [u8; 16] = [0x69; 16];

    /// Appends an Avro `int`/`long`: zig-zag, then variable-length base-128.
    fn push_avro_long(out: &mut Vec<u8>, value: i64) {
        // Zig-zag is defined on the two's-complement bit pattern, so the reinterpreting cast
        // to `u64` is the encoding, not a lossy conversion.
        let mut n = ((value << 1) ^ (value >> 63)) as u64;
        loop {
            let low = u8::try_from(n & 0x7f).expect("masked to seven bits");
            n >>= 7;
            if n == 0 {
                out.push(low);
                break;
            }
            out.push(low | 0x80);
        }
    }

    /// Appends an Avro `bytes`/`string`: a long length prefix, then the payload.
    fn push_avro_bytes(out: &mut Vec<u8>, payload: &[u8]) {
        push_avro_long(
            out,
            i64::try_from(payload.len()).expect("fixture payload length fits in an i64"),
        );
        out.extend_from_slice(payload);
    }

    /// One `manifest_file` datum, encoded in the field order of [`MANIFEST_FILE_V2_SCHEMA`].
    fn manifest_file_datum(index: usize) -> Vec<u8> {
        let mut datum = Vec::new();
        push_avro_bytes(
            &mut datum,
            format!("memory:/t/metadata/m{index}.avro").as_bytes(),
        );
        push_avro_long(&mut datum, 1024); // manifest_length
        push_avro_long(&mut datum, 0); // partition_spec_id
        push_avro_long(&mut datum, 0); // content: 0 == data
        push_avro_long(&mut datum, 1); // sequence_number
        push_avro_long(&mut datum, 1); // min_sequence_number
        push_avro_long(&mut datum, 42); // added_snapshot_id
        push_avro_long(&mut datum, 1); // added_files_count
        push_avro_long(&mut datum, 0); // existing_files_count
        push_avro_long(&mut datum, 0); // deleted_files_count
        push_avro_long(&mut datum, 1); // added_rows_count
        push_avro_long(&mut datum, 0); // existing_rows_count
        push_avro_long(&mut datum, 0); // deleted_rows_count
        push_avro_long(&mut datum, 0); // partitions: union branch 0 == null
        push_avro_long(&mut datum, 0); // key_metadata: union branch 0 == null
        datum
    }

    /// A real [`ManifestList`] carrying exactly `entry_count` rows.
    ///
    /// Built as an Avro object-container file and parsed back through the public
    /// [`ManifestList::parse_with_version`], so the value under test is one the production read
    /// path would produce. Hand-encoding the container is what keeps this crate free of a
    /// dev-dependency on an Avro writer.
    fn manifest_list_with_entries(entry_count: usize) -> Arc<ManifestList> {
        let mut file = Vec::new();
        file.extend_from_slice(b"Obj\x01"); // magic
        push_avro_long(&mut file, 1); // metadata map: one block of one entry
        push_avro_bytes(&mut file, b"avro.schema");
        push_avro_bytes(&mut file, MANIFEST_FILE_V2_SCHEMA.as_bytes());
        push_avro_long(&mut file, 0); // end of the metadata map
        file.extend_from_slice(&FIXTURE_SYNC_MARKER);

        if entry_count > 0 {
            let mut objects = Vec::new();
            for i in 0..entry_count {
                objects.extend_from_slice(&manifest_file_datum(i));
            }
            push_avro_long(
                &mut file,
                i64::try_from(entry_count).expect("fixture entry count fits in an i64"),
            );
            push_avro_long(
                &mut file,
                i64::try_from(objects.len()).expect("fixture block length fits in an i64"),
            );
            file.extend_from_slice(&objects);
            file.extend_from_slice(&FIXTURE_SYNC_MARKER);
        }

        let list = ManifestList::parse_with_version(&file, FormatVersion::V2)
            .expect("parse the hand-encoded V2 manifest-list fixture");
        assert_eq!(
            list.entries().len(),
            entry_count,
            "fixture must round-trip the requested row count"
        );
        Arc::new(list)
    }

    /// The clamp floors at one weight unit and saturates at `u32::MAX`.
    ///
    /// Mutation caught: widening the floor to `0` (`bytes.clamp(0, ..)`) — a zero-weight entry
    /// would be free and the budget would stop binding; or dropping the upper clamp, which
    /// would make the `as u32` truncate.
    #[test]
    fn clamp_cache_weight_floors_at_one_and_saturates() {
        assert_eq!(
            clamp_cache_weight(0),
            1,
            "a zero estimate must still cost 1"
        );
        assert_eq!(clamp_cache_weight(1), 1);
        assert_eq!(clamp_cache_weight(4096), 4096);
        assert_eq!(clamp_cache_weight(u32::MAX as u64), u32::MAX);
        assert_eq!(
            clamp_cache_weight(u32::MAX as u64 + 1),
            u32::MAX,
            "an over-wide estimate must saturate, not truncate"
        );
        assert_eq!(clamp_cache_weight(u64::MAX), u32::MAX);
    }

    /// Weight scales with entry count, floors an empty object at one entry, and saturates.
    ///
    /// Mutation caught: dropping the `.max(1)` in `weight_from_entry_count` (an empty manifest
    /// would weigh `1` instead of `ROUGH_MANIFEST_ENTRY_BYTES`), or swapping the saturating
    /// multiply for `*` (which would panic in debug on the overflow case below).
    #[test]
    fn weight_from_entry_count_scales_floors_and_saturates() {
        assert_eq!(
            weight_from_entry_count(0, ROUGH_MANIFEST_ENTRY_BYTES),
            ROUGH_MANIFEST_ENTRY_BYTES as u32,
            "an empty object is floored at one entry, not at one byte"
        );
        assert_eq!(
            weight_from_entry_count(1, ROUGH_MANIFEST_ENTRY_BYTES),
            ROUGH_MANIFEST_ENTRY_BYTES as u32
        );
        assert_eq!(
            weight_from_entry_count(10, ROUGH_MANIFEST_ENTRY_BYTES),
            10 * ROUGH_MANIFEST_ENTRY_BYTES as u32
        );
        assert_eq!(
            weight_from_entry_count(10, ROUGH_MANIFEST_LIST_ENTRY_BYTES),
            10 * ROUGH_MANIFEST_LIST_ENTRY_BYTES as u32
        );
        assert!(
            weight_from_entry_count(10, ROUGH_MANIFEST_ENTRY_BYTES)
                > weight_from_entry_count(10, ROUGH_MANIFEST_LIST_ENTRY_BYTES),
            "a manifest entry must outweigh a manifest-list row"
        );
        assert_eq!(
            weight_from_entry_count(usize::MAX, ROUGH_MANIFEST_LIST_ENTRY_BYTES),
            u32::MAX,
            "saturating multiply, then clamp — never a panic or a wrap"
        );
    }

    /// The manifest weigher reads the real entry count off a parsed `Manifest`.
    ///
    /// Mutation caught: `estimate_manifest_weight` returning a constant, or using the
    /// manifest-list constant.
    #[test]
    fn estimate_manifest_weight_tracks_entry_count() {
        let empty = manifest_with_entries(0);
        let ten = manifest_with_entries(10);

        assert_eq!(empty.entries().len(), 0);
        assert_eq!(ten.entries().len(), 10);
        assert_eq!(
            estimate_manifest_weight(&empty),
            ROUGH_MANIFEST_ENTRY_BYTES as u32
        );
        assert_eq!(
            estimate_manifest_weight(&ten),
            10 * ROUGH_MANIFEST_ENTRY_BYTES as u32,
            "weight must be entry_count x the per-entry constant"
        );
    }

    /// The manifest-list weigher reads the real entry count off a parsed `ManifestList`.
    ///
    /// Mutation caught: `estimate_manifest_list_weight` returning a constant (the two counts
    /// below differ), or reaching for `ROUGH_MANIFEST_ENTRY_BYTES` instead of
    /// `ROUGH_MANIFEST_LIST_ENTRY_BYTES` (768 vs 256 on the empty case).
    #[test]
    fn estimate_manifest_list_weight_tracks_entry_count() {
        let empty = manifest_list_with_entries(0);
        let three = manifest_list_with_entries(3);

        assert_eq!(
            estimate_manifest_list_weight(&empty),
            ROUGH_MANIFEST_LIST_ENTRY_BYTES as u32,
            "an empty list is floored at one row, at the manifest-LIST constant"
        );
        assert_eq!(
            estimate_manifest_list_weight(&three),
            3 * ROUGH_MANIFEST_LIST_ENTRY_BYTES as u32,
            "weight must be entry_count x the per-entry constant"
        );
    }

    /// The byte budget actually binds: a budget that fits five empty manifests evicts when
    /// forty are inserted, even though forty is nowhere near an entry-count budget of 4096.
    ///
    /// Mutation caught: deleting `.weigher(..)` from `build_weighted_cache`. Without it moka
    /// weighs each entry as `1`, `max_capacity(4096)` becomes "4096 entries", and all forty
    /// manifests stay resident — which is exactly the pre-fix behaviour this unit closes.
    #[test]
    fn byte_budget_evicts_where_an_entry_count_budget_would_not() {
        // Five empty manifests (768 each) fit in 4096; the sixth does not.
        const BUDGET_BYTES: u64 = 4096;
        const INSERTS: usize = 40;
        let fits = (BUDGET_BYTES / ROUGH_MANIFEST_ENTRY_BYTES) as usize;
        assert_eq!(fits, 5, "test arithmetic: 4096 / 768 == 5");
        assert!(
            (INSERTS as u64) < BUDGET_BYTES,
            "the insert count must be far below the budget read as an entry count, \
             otherwise the test cannot tell bytes from entries"
        );

        let provider = MokaObjectCacheProvider::new_with_capacity(BUDGET_BYTES);
        for i in 0..INSERTS {
            provider
                .manifest_cache()
                .set(format!("manifest-{i}"), manifest_with_entries(0));
        }
        // moka applies admission/eviction on its maintenance path; drain it instead of sleeping.
        provider.manifest_cache.0.run_pending_tasks();

        let entry_count = provider.manifest_cache.0.entry_count();
        let weighted_size = provider.manifest_cache.0.weighted_size();
        assert!(
            entry_count <= fits as u64,
            "byte budget must cap the cache at {fits} empty manifests, got {entry_count}"
        );
        assert!(
            weighted_size <= BUDGET_BYTES,
            "weighted size {weighted_size} must stay within the {BUDGET_BYTES}-byte budget"
        );
        assert!(
            entry_count > 0,
            "eviction must not empty the cache outright"
        );
    }

    /// The same pin on the *manifest-list* cache: `new_with_capacity` must route it through
    /// `build_weighted_cache` too, not just its manifest twin.
    ///
    /// Mutation caught: building `manifest_list_cache` as `moka::sync::Cache::new(cache_size_bytes)`
    /// — the pre-fix line. Unweighted, each of the forty inserts weighs `1`, `max_capacity(1024)`
    /// reads as "1024 entries", and all forty stay resident instead of four. Also caught: using
    /// `ROUGH_MANIFEST_ENTRY_BYTES` in `estimate_manifest_list_weight`, which would settle the
    /// cache at one resident row rather than four.
    #[test]
    fn manifest_list_byte_budget_evicts_where_an_entry_count_budget_would_not() {
        // Four empty manifest lists (256 each) exactly fill 1024; the fifth does not fit.
        const BUDGET_BYTES: u64 = 1024;
        const INSERTS: usize = 40;
        let fits = BUDGET_BYTES / ROUGH_MANIFEST_LIST_ENTRY_BYTES;
        assert_eq!(fits, 4, "test arithmetic: 1024 / 256 == 4");
        assert!(
            (INSERTS as u64) < BUDGET_BYTES,
            "the insert count must be far below the budget read as an entry count, \
             otherwise the test cannot tell bytes from entries"
        );

        let provider = MokaObjectCacheProvider::new_with_capacity(BUDGET_BYTES);
        let list = manifest_list_with_entries(0);
        for i in 0..INSERTS {
            provider
                .manifest_list_cache()
                .set(format!("manifest-list-{i}"), Arc::clone(&list));
        }
        provider.manifest_list_cache.0.run_pending_tasks();

        assert_eq!(
            provider.manifest_list_cache.0.entry_count(),
            fits,
            "the byte budget must cap the manifest-list cache at {fits} empty lists"
        );
        assert_eq!(
            provider.manifest_list_cache.0.weighted_size(),
            BUDGET_BYTES,
            "the resident weight must be the budget, in bytes"
        );
    }

    /// An object heavier than the whole budget is never resident, and it does not poison the
    /// cache for subsequent entries. A miss is safe: `get` is `Option`-typed and the underlying
    /// metadata files are immutable, so the caller simply re-reads.
    ///
    /// Mutation caught: sizing the manifest cache by entry count again (the oversized manifest
    /// would then be resident and `get` would return `Some`).
    #[test]
    fn oversized_object_misses_and_leaves_the_cache_usable() {
        const BUDGET_BYTES: u64 = 4096;
        let oversized = manifest_with_entries(10); // 7680 bytes > 4096
        assert!(estimate_manifest_weight(&oversized) as u64 > BUDGET_BYTES);

        let provider = MokaObjectCacheProvider::new_with_capacity(BUDGET_BYTES);
        provider
            .manifest_cache()
            .set("oversized".to_string(), oversized);
        provider.manifest_cache.0.run_pending_tasks();

        assert!(
            provider
                .manifest_cache()
                .get(&"oversized".to_string())
                .is_none(),
            "an object heavier than the budget must be rejected, not admitted"
        );

        // The cache still works for objects that do fit.
        provider
            .manifest_cache()
            .set("small".to_string(), manifest_with_entries(0));
        provider.manifest_cache.0.run_pending_tasks();
        let cached = provider
            .manifest_cache()
            .get(&"small".to_string())
            .expect("a within-budget object must still be cacheable after an oversized reject");
        assert_eq!(cached.entries().len(), 0);
    }

    /// A zero byte budget disables caching, matching `ObjectCache::with_disabled_cache`.
    ///
    /// Mutation caught: treating `0` as "unbounded" — e.g. replacing the zero arm of
    /// `build_weighted_cache` with a builder that omits `.max_capacity(..)`.
    #[test]
    fn zero_capacity_disables_both_caches() {
        let provider = MokaObjectCacheProvider::new_with_capacity(0);

        provider
            .manifest_cache()
            .set("k".to_string(), manifest_with_entries(0));
        provider.manifest_cache.0.run_pending_tasks();
        assert!(
            provider.manifest_cache().get(&"k".to_string()).is_none(),
            "a zero byte budget must cache nothing"
        );
        assert_eq!(provider.manifest_cache.0.entry_count(), 0);
        assert_eq!(provider.manifest_list_cache.0.entry_count(), 0);
    }

    /// The default provider is byte-bounded, not entry-bounded — on **both** caches.
    ///
    /// `max_capacity()` alone cannot discriminate the two: it reports the same 33,554,432 whether
    /// that number counts bytes or entries. What separates them is the weigher, so this test
    /// observes `weighted_size()` after a real insert — 1 per entry with no weigher installed,
    /// `entry_count × the per-entry constant` with one.
    ///
    /// Mutation caught: reverting `new()` to construct the caches directly as
    /// `moka::sync::Cache::new(DEFAULT_CACHE_SIZE_BYTES)` — the pre-fix path. The resident weight
    /// then collapses to 1 per entry and 33,554,432 becomes an entry count.
    #[test]
    fn default_provider_budget_is_bytes() {
        let provider = MokaObjectCacheProvider::new();
        assert_eq!(
            provider.manifest_cache.0.policy().max_capacity(),
            Some(DEFAULT_CACHE_SIZE_BYTES)
        );
        assert_eq!(
            provider.manifest_list_cache.0.policy().max_capacity(),
            Some(DEFAULT_CACHE_SIZE_BYTES)
        );

        provider
            .manifest_cache()
            .set("m".to_string(), manifest_with_entries(10));
        provider.manifest_cache.0.run_pending_tasks();
        assert_eq!(
            provider.manifest_cache.0.weighted_size(),
            10 * ROUGH_MANIFEST_ENTRY_BYTES,
            "the default manifest cache must charge bytes, not one unit per entry"
        );

        provider
            .manifest_list_cache()
            .set("l".to_string(), manifest_list_with_entries(3));
        provider.manifest_list_cache.0.run_pending_tasks();
        assert_eq!(
            provider.manifest_list_cache.0.weighted_size(),
            3 * ROUGH_MANIFEST_LIST_ENTRY_BYTES,
            "the default manifest-list cache must charge bytes, not one unit per entry"
        );

        let entries_to_fill = DEFAULT_CACHE_SIZE_BYTES / ROUGH_MANIFEST_ENTRY_BYTES;
        assert!(
            weight_from_entry_count(entries_to_fill as usize + 1, ROUGH_MANIFEST_ENTRY_BYTES)
                as u64
                > DEFAULT_CACHE_SIZE_BYTES,
            "one manifest of {} entries must already exceed the default budget",
            entries_to_fill + 1
        );
    }

    /// A caller-supplied cache is used verbatim: the provider does not attach a weigher or
    /// rewrite the capacity, which is what the `with_*` docs promise.
    ///
    /// Mutation caught: `with_manifest_cache` silently rebuilding the cache with this crate's
    /// weigher and default budget.
    #[test]
    fn caller_supplied_cache_keeps_its_own_policy() {
        let caller_cache = moka::sync::Cache::new(7);
        let provider = MokaObjectCacheProvider::new().with_manifest_cache(caller_cache);

        assert_eq!(
            provider.manifest_cache.0.policy().max_capacity(),
            Some(7),
            "the caller's builder must win"
        );

        // Entry-count semantics, exactly as the caller configured: a 10-entry manifest weighs 1.
        provider
            .manifest_cache()
            .set("big".to_string(), manifest_with_entries(10));
        provider.manifest_cache.0.run_pending_tasks();
        assert!(
            provider.manifest_cache().get(&"big".to_string()).is_some(),
            "under the caller's unweighted cache the entry weighs 1 and is admitted"
        );
        assert_eq!(provider.manifest_cache.0.weighted_size(), 1);
    }
}
