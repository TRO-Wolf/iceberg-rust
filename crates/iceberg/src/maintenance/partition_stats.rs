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

//! `ComputePartitionStats` — the compute core (the Rust port of Java 1.10.0 core
//! `org.apache.iceberg.PartitionStatsHandler`'s full-compute aggregation).
//!
//! This module is the **X1** scope: it computes per-partition statistics by aggregating every
//! manifest entry of a snapshot into the Java-exact partition-stats schema. The stats-file WRITE +
//! metadata registration (`set_partition_statistics`) is **X2** — a separate increment — and lives
//! elsewhere. `ComputeTableStats` (NDV / theta sketches) is explicitly out of scope (it needs a
//! sketch dependency the workspace does not carry).
//!
//! # Why this matters
//!
//! Partition stats feed planning / cost decisions downstream. A wrong aggregation silently misleads
//! every consumer; a wrong unified-partition-tuple mapping across evolved specs corrupts the stats
//! file's keying. The aggregation therefore mirrors Java's traversal exactly (verified against the
//! `iceberg-core-1.10.0.jar` bytecode — see the per-item provenance below).
//!
//! # Java provenance (1.10.0 JAR BYTECODE, not the /tmp MAIN-source checkout)
//!
//! The 1.10.0 jar's `PartitionStatsHandler` holds the stats schema as field-id constants directly
//! and uses a concrete `PartitionStats` class. The MAIN-source checkout under `/tmp/iceberg-java-ref`
//! is POST-1.10.0 — it was refactored to a `PartitionStatistics` interface + `BasePartitionStatistics`
//! with a `DV_COUNT` backward-compat guard in `appendStats` the jar does not have. Per the repo
//! lesson "bytecode outranks MAIN source for version-sensitive claims," every fact below is
//! disassembled from the 1.10.0 jar.
//!
//! ## Schema field ids (these become the ON-DISK parquet field ids in X2 — they MUST match Java)
//!
//! The partition struct is `required(1, "partition", <unified partition type>)`. Then:
//!
//! | id | name | type | v2 | v3 |
//! |----|------|------|----|----|
//! | 2 | `spec_id` | int | required | required |
//! | 3 | `data_record_count` | long | required | required |
//! | 4 | `data_file_count` | int | required | required |
//! | 5 | `total_data_file_size_in_bytes` | long | required | required |
//! | 6 | `position_delete_record_count` | long | **optional** | **required** |
//! | 7 | `position_delete_file_count` | int | **optional** | **required** |
//! | 8 | `equality_delete_record_count` | long | **optional** | **required** |
//! | 9 | `equality_delete_file_count` | int | **optional** | **required** |
//! | 10 | `total_record_count` | long | optional | optional |
//! | 11 | `last_updated_at` | long | optional | optional |
//! | 12 | `last_updated_snapshot_id` | long | optional | optional |
//! | 13 | `dv_count` | int | (absent) | required (default 0) |
//!
//! `schema(StructType)` = v2 (12 fields); `schema(StructType, formatVersion)` validates
//! `0 < formatVersion <= 4`, then picks v2 for `formatVersion <= 2` else v3 (13 fields).
//!
//! ## Traversal (full compute)
//!
//! [`compute_partition_stats`] mirrors `computeAndWriteStatsFile(table, snapshotId)`'s *full-compute*
//! branch (no prior stats file): read `snapshot.allManifests(io)` — the given snapshot's ENTIRE
//! manifest list (all data + delete manifests reachable from it) — and run `computeStats(..,
//! incremental=false)`. `collectStatsForManifest` opens each manifest and iterates `reader.entries()`
//! = **ALL entries, LIVE and DELETED**. A LIVE entry calls `liveEntry`; a DELETED tombstone calls
//! `deletedEntry` (which only bumps last-updated, no counter). The incremental path (diff between two
//! stats files) is X2's concern and is NOT ported here.

use std::collections::HashMap;

use crate::Result;
use crate::error::Error;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, Manifest, ManifestEntry,
    NestedField, PartitionSpec, PrimitiveLiteral, PrimitiveType, Schema, SchemaRef, Snapshot,
    Struct, StructType, TableMetadata, Type,
};
use crate::table::Table;

/// The partition struct's field id in the stats schema (Java `PartitionStatsHandler.PARTITION_FIELD_ID`).
const PARTITION_FIELD_ID: i32 = 1;
/// The partition struct's field name (Java `PartitionStatsHandler.PARTITION_FIELD_NAME`).
const PARTITION_FIELD_NAME: &str = "partition";

/// `spec_id` field id (Java `PartitionStatsHandler.SPEC_ID`).
const SPEC_ID_FIELD_ID: i32 = 2;
/// `data_record_count` field id (Java `DATA_RECORD_COUNT`).
const DATA_RECORD_COUNT_FIELD_ID: i32 = 3;
/// `data_file_count` field id (Java `DATA_FILE_COUNT`).
const DATA_FILE_COUNT_FIELD_ID: i32 = 4;
/// `total_data_file_size_in_bytes` field id (Java `TOTAL_DATA_FILE_SIZE_IN_BYTES`).
const TOTAL_DATA_FILE_SIZE_IN_BYTES_FIELD_ID: i32 = 5;
/// `position_delete_record_count` field id (Java `POSITION_DELETE_RECORD_COUNT`).
const POSITION_DELETE_RECORD_COUNT_FIELD_ID: i32 = 6;
/// `position_delete_file_count` field id (Java `POSITION_DELETE_FILE_COUNT`).
const POSITION_DELETE_FILE_COUNT_FIELD_ID: i32 = 7;
/// `equality_delete_record_count` field id (Java `EQUALITY_DELETE_RECORD_COUNT`).
const EQUALITY_DELETE_RECORD_COUNT_FIELD_ID: i32 = 8;
/// `equality_delete_file_count` field id (Java `EQUALITY_DELETE_FILE_COUNT`).
const EQUALITY_DELETE_FILE_COUNT_FIELD_ID: i32 = 9;
/// `total_record_count` field id (Java `TOTAL_RECORD_COUNT`).
const TOTAL_RECORD_COUNT_FIELD_ID: i32 = 10;
/// `last_updated_at` field id (Java `LAST_UPDATED_AT`).
const LAST_UPDATED_AT_FIELD_ID: i32 = 11;
/// `last_updated_snapshot_id` field id (Java `LAST_UPDATED_SNAPSHOT_ID`).
const LAST_UPDATED_SNAPSHOT_ID_FIELD_ID: i32 = 12;
/// `dv_count` field id (Java `DV_COUNT`; v3+ only).
const DV_COUNT_FIELD_ID: i32 = 13;

/// One partition's rolled-up statistics, mirroring Java 1.10.0 `org.apache.iceberg.PartitionStats`
/// field-for-field (and its in-memory nullability).
///
/// Java's `PartitionStats(StructLike partition, int specId)` keeps the count members as primitives
/// (default 0) and only `total_record_count` / `last_updated_at` / `last_updated_snapshot_id` as
/// boxed `Long` (genuinely nullable). This struct preserves that distinction: the counters are plain
/// integers initialized to 0, and the three boxed members are `Option<i64>`.
///
/// Note the count widths follow Java's column types in the stats schema, NOT the `u64`/`i32` of the
/// in-memory `inspect::partitions` aggregation: record counts are `i64` (Java `long`), file counts
/// are `i32` (Java `int`). `data_file_count` etc. can never legitimately be negative in a full
/// compute (the deleted-entry path does not decrement — only the X2 incremental path does), but the
/// signed widths match the on-disk schema exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionStats {
    /// The partition tuple coerced into the table's unified partition type (the aggregation key).
    partition: Struct,
    /// The spec id of the files contributing to this partition (Java `specId`). All files keyed
    /// here share one spec id (the per-`(specId, partition)` map key).
    spec_id: i32,
    /// `data_record_count` (Java `dataRecordCount`, `long`): sum of data files' record counts.
    data_record_count: i64,
    /// `data_file_count` (Java `dataFileCount`, `int`): number of live data files.
    data_file_count: i32,
    /// `total_data_file_size_in_bytes` (Java `totalDataFileSizeInBytes`, `long`).
    total_data_file_size_in_bytes: i64,
    /// `position_delete_record_count` (Java `positionDeleteRecordCount`, `long`).
    position_delete_record_count: i64,
    /// `position_delete_file_count` (Java `positionDeleteFileCount`, `int`): NON-DV position deletes.
    position_delete_file_count: i32,
    /// `equality_delete_record_count` (Java `equalityDeleteRecordCount`, `long`).
    equality_delete_record_count: i64,
    /// `equality_delete_file_count` (Java `equalityDeleteFileCount`, `int`).
    equality_delete_file_count: i32,
    /// `total_record_count` (Java `totalRecordCount`, boxed `Long`): NEVER computed in the compute
    /// path (Java: "Not computing the TOTAL_RECORD_COUNT for now as it needs scanning the data") —
    /// stays `None`. Merged through if a prior stats file carried it (X2 incremental), but the
    /// full-compute path never sets it.
    total_record_count: Option<i64>,
    /// `last_updated_at` (Java `lastUpdatedAt`, boxed `Long`): the MAX `snapshot.timestampMillis()`
    /// across the snapshots that touched this partition; `None` until first set.
    last_updated_at: Option<i64>,
    /// `last_updated_snapshot_id` (Java `lastUpdatedSnapshotId`, boxed `Long`): the snapshot id whose
    /// timestamp won the `last_updated_at` max.
    last_updated_snapshot_id: Option<i64>,
    /// `dv_count` (Java `dvCount`, `int`): number of PUFFIN (deletion-vector) position-delete files.
    dv_count: i32,
}

impl PartitionStats {
    /// Constructs an empty `PartitionStats` for a partition key + spec id, all counters zero and the
    /// nullable members `None` (Java `new PartitionStats(partition, specId)`).
    pub fn new(partition: Struct, spec_id: i32) -> Self {
        Self {
            partition,
            spec_id,
            data_record_count: 0,
            data_file_count: 0,
            total_data_file_size_in_bytes: 0,
            position_delete_record_count: 0,
            position_delete_file_count: 0,
            equality_delete_record_count: 0,
            equality_delete_file_count: 0,
            total_record_count: None,
            last_updated_at: None,
            last_updated_snapshot_id: None,
            dv_count: 0,
        }
    }

    /// The coerced partition tuple (Java `partition()`).
    pub fn partition(&self) -> &Struct {
        &self.partition
    }

    /// The spec id (Java `specId()`).
    pub fn spec_id(&self) -> i32 {
        self.spec_id
    }

    /// `data_record_count` (Java `dataRecordCount()`).
    pub fn data_record_count(&self) -> i64 {
        self.data_record_count
    }

    /// `data_file_count` (Java `dataFileCount()`).
    pub fn data_file_count(&self) -> i32 {
        self.data_file_count
    }

    /// `total_data_file_size_in_bytes` (Java `totalDataFileSizeInBytes()`).
    pub fn total_data_file_size_in_bytes(&self) -> i64 {
        self.total_data_file_size_in_bytes
    }

    /// `position_delete_record_count` (Java `positionDeleteRecordCount()`).
    pub fn position_delete_record_count(&self) -> i64 {
        self.position_delete_record_count
    }

    /// `position_delete_file_count` (Java `positionDeleteFileCount()`).
    pub fn position_delete_file_count(&self) -> i32 {
        self.position_delete_file_count
    }

    /// `equality_delete_record_count` (Java `equalityDeleteRecordCount()`).
    pub fn equality_delete_record_count(&self) -> i64 {
        self.equality_delete_record_count
    }

    /// `equality_delete_file_count` (Java `equalityDeleteFileCount()`).
    pub fn equality_delete_file_count(&self) -> i32 {
        self.equality_delete_file_count
    }

    /// `total_record_count` (Java `totalRecords()`): `None` after a full compute.
    pub fn total_record_count(&self) -> Option<i64> {
        self.total_record_count
    }

    /// `last_updated_at` (Java `lastUpdatedAt()`), in milliseconds.
    pub fn last_updated_at(&self) -> Option<i64> {
        self.last_updated_at
    }

    /// `last_updated_snapshot_id` (Java `lastUpdatedSnapshotId()`).
    pub fn last_updated_snapshot_id(&self) -> Option<i64> {
        self.last_updated_snapshot_id
    }

    /// `dv_count` (Java `dvCount()`).
    pub fn dv_count(&self) -> i32 {
        self.dv_count
    }

    /// Updates the per-content-type counters for a LIVE manifest entry's file, then the last-updated
    /// snapshot info — the Rust port of Java `PartitionStats.liveEntry(ContentFile, Snapshot)`.
    ///
    /// `record_count` / `file_size_in_bytes` come from the file; `file_format` distinguishes a PUFFIN
    /// position delete (a deletion vector → `dv_count`) from a parquet position delete
    /// (→ `position_delete_file_count`). `snapshot_timestamp_ms` / `snapshot_id` are the entry's OWN
    /// snapshot (`table.snapshot(entry.snapshotId())`), or `None` when that snapshot is absent.
    ///
    /// The spec-id-must-match precondition Java enforces is guaranteed by construction here: the
    /// caller only ever invokes this on the row keyed by the file's own `(specId, partition)`.
    fn live_entry(
        &mut self,
        content_type: DataContentType,
        file_format: DataFileFormat,
        record_count: i64,
        file_size_in_bytes: i64,
        snapshot_timestamp_ms: Option<i64>,
        snapshot_id: Option<i64>,
    ) {
        match content_type {
            DataContentType::Data => {
                self.data_record_count += record_count;
                self.data_file_count += 1;
                self.total_data_file_size_in_bytes += file_size_in_bytes;
            }
            DataContentType::PositionDeletes => {
                self.position_delete_record_count += record_count;
                // Java: a PUFFIN position delete is a deletion vector (bumps dv_count); a non-PUFFIN
                // (parquet) position delete bumps position_delete_file_count.
                if file_format == DataFileFormat::Puffin {
                    self.dv_count += 1;
                } else {
                    self.position_delete_file_count += 1;
                }
            }
            DataContentType::EqualityDeletes => {
                self.equality_delete_record_count += record_count;
                self.equality_delete_file_count += 1;
            }
        }

        if let (Some(timestamp_ms), Some(id)) = (snapshot_timestamp_ms, snapshot_id) {
            self.update_snapshot_info(id, timestamp_ms);
        }
    }

    /// Updates only the last-updated snapshot info for a DELETED manifest entry (Java
    /// `PartitionStats.deletedEntry(Snapshot)`) — no counter is touched. A DELETED tombstone still
    /// creates / keeps the partition row (via `computeIfAbsent`), which is why a fully-deleted
    /// partition keeps a zero-count row (`testCopyOnWriteDelete`).
    fn deleted_entry(&mut self, snapshot_timestamp_ms: Option<i64>, snapshot_id: Option<i64>) {
        if let (Some(timestamp_ms), Some(id)) = (snapshot_timestamp_ms, snapshot_id) {
            self.update_snapshot_info(id, timestamp_ms);
        }
    }

    /// Sets `last_updated_at` / `last_updated_snapshot_id` iff this entry's timestamp is strictly
    /// newer than the current `last_updated_at` (Java `updateSnapshotInfo`: `lastUpdatedAt == null ||
    /// lastUpdatedAt < updatedAt`). Ties keep the first-seen snapshot.
    fn update_snapshot_info(&mut self, snapshot_id: i64, updated_at_ms: i64) {
        if self
            .last_updated_at
            .is_none_or(|current| current < updated_at_ms)
        {
            self.last_updated_at = Some(updated_at_ms);
            self.last_updated_snapshot_id = Some(snapshot_id);
        }
    }

    /// Merges `input`'s statistics into `self` — the Rust port of Java `PartitionStats.appendStats`.
    ///
    /// All primitive counters add directly (including `dv_count`, which the 1.10.0 jar adds
    /// unconditionally — the V2-backward-compat guard is a POST-1.10.0 addition). `total_record_count`
    /// (nullable) is set-if-null-else-added when `input` has it. Finally, if `input.last_updated_at`
    /// is set, last-updated info is re-evaluated against it via `update_snapshot_info`.
    ///
    /// # Errors
    ///
    /// Returns a `DataInvalid` error if the spec ids differ (Java `Preconditions.checkArgument(...,
    /// "Spec IDs must match")`). Two rows merged together must share a spec id by construction (they
    /// are keyed by `(specId, partition)`), so this is defense-in-depth.
    fn append_stats(&mut self, input: &PartitionStats) -> Result<()> {
        if self.spec_id != input.spec_id {
            return Err(Error::new(
                crate::ErrorKind::DataInvalid,
                format!("Spec IDs must match: {} != {}", self.spec_id, input.spec_id),
            ));
        }

        self.data_record_count += input.data_record_count;
        self.data_file_count += input.data_file_count;
        self.total_data_file_size_in_bytes += input.total_data_file_size_in_bytes;
        self.position_delete_record_count += input.position_delete_record_count;
        self.position_delete_file_count += input.position_delete_file_count;
        self.equality_delete_record_count += input.equality_delete_record_count;
        self.equality_delete_file_count += input.equality_delete_file_count;
        self.dv_count += input.dv_count;

        if let Some(input_total) = input.total_record_count {
            self.total_record_count = Some(match self.total_record_count {
                Some(current) => current + input_total,
                None => input_total,
            });
        }

        if let Some(input_last_updated_at) = input.last_updated_at {
            // Java passes input.lastUpdatedSnapshotId() unconditionally here; it is non-null whenever
            // last_updated_at is non-null (they are set together in update_snapshot_info).
            if let Some(input_snapshot_id) = input.last_updated_snapshot_id {
                self.update_snapshot_info(input_snapshot_id, input_last_updated_at);
            }
        }

        Ok(())
    }
}

/// Builds the partition-stats [`Schema`] for a given unified partition type and format version — the
/// Rust port of Java 1.10.0 `PartitionStatsHandler.schema(StructType, int)`.
///
/// `format_version <= 2` → the **v2** schema (12 fields; the position/equality delete fields and
/// `total_record_count` / `last_updated_*` are OPTIONAL; no `dv_count`). `format_version >= 3` → the
/// **v3** schema (13 fields; the position/equality delete fields become REQUIRED and `dv_count` is
/// added as a required field with default 0).
///
/// The field ids are the exact Java constants (1..=13) — in X2 these become the on-disk parquet field
/// ids, so they must match Java byte-for-byte.
///
/// # Errors
///
/// Returns a `DataInvalid` error if `unified_partition_type` is empty (Java
/// `Preconditions.checkState(!partitionType.fields().isEmpty(), "Table must be partitioned")`).
pub fn partition_stats_schema(
    unified_partition_type: &StructType,
    format_version: FormatVersion,
) -> Result<Schema> {
    if unified_partition_type.fields().is_empty() {
        return Err(Error::new(
            crate::ErrorKind::DataInvalid,
            "Table must be partitioned",
        ));
    }

    let is_v3 = format_version >= FormatVersion::V3;

    // Java rebuilds the position/equality delete fields as REQUIRED for v3 (same id + name + type);
    // they stay OPTIONAL for v2. total_record_count / last_updated_* are OPTIONAL in both.
    let delete_field = |id: i32, name: &str, field_type: Type| {
        if is_v3 {
            NestedField::required(id, name, field_type)
        } else {
            NestedField::optional(id, name, field_type)
        }
    };

    let mut fields = vec![
        NestedField::required(
            PARTITION_FIELD_ID,
            PARTITION_FIELD_NAME,
            Type::Struct(unified_partition_type.clone()),
        )
        .into(),
        NestedField::required(
            SPEC_ID_FIELD_ID,
            "spec_id",
            Type::Primitive(PrimitiveType::Int),
        )
        .into(),
        NestedField::required(
            DATA_RECORD_COUNT_FIELD_ID,
            "data_record_count",
            Type::Primitive(PrimitiveType::Long),
        )
        .into(),
        NestedField::required(
            DATA_FILE_COUNT_FIELD_ID,
            "data_file_count",
            Type::Primitive(PrimitiveType::Int),
        )
        .into(),
        NestedField::required(
            TOTAL_DATA_FILE_SIZE_IN_BYTES_FIELD_ID,
            "total_data_file_size_in_bytes",
            Type::Primitive(PrimitiveType::Long),
        )
        .into(),
        delete_field(
            POSITION_DELETE_RECORD_COUNT_FIELD_ID,
            "position_delete_record_count",
            Type::Primitive(PrimitiveType::Long),
        )
        .into(),
        delete_field(
            POSITION_DELETE_FILE_COUNT_FIELD_ID,
            "position_delete_file_count",
            Type::Primitive(PrimitiveType::Int),
        )
        .into(),
        delete_field(
            EQUALITY_DELETE_RECORD_COUNT_FIELD_ID,
            "equality_delete_record_count",
            Type::Primitive(PrimitiveType::Long),
        )
        .into(),
        delete_field(
            EQUALITY_DELETE_FILE_COUNT_FIELD_ID,
            "equality_delete_file_count",
            Type::Primitive(PrimitiveType::Int),
        )
        .into(),
        NestedField::optional(
            TOTAL_RECORD_COUNT_FIELD_ID,
            "total_record_count",
            Type::Primitive(PrimitiveType::Long),
        )
        .into(),
        NestedField::optional(
            LAST_UPDATED_AT_FIELD_ID,
            "last_updated_at",
            Type::Primitive(PrimitiveType::Long),
        )
        .into(),
        NestedField::optional(
            LAST_UPDATED_SNAPSHOT_ID_FIELD_ID,
            "last_updated_snapshot_id",
            Type::Primitive(PrimitiveType::Long),
        )
        .into(),
    ];

    if is_v3 {
        // dv_count: required Int with initial + write default 0 (Java NestedField.Builder
        // .withInitialDefault(Literal.of(0)).withWriteDefault(Literal.of(0))). The on-disk default is
        // an X2 concern; for the in-memory schema the required-Int field id 13 is what matters here.
        fields.push(
            NestedField::required(
                DV_COUNT_FIELD_ID,
                "dv_count",
                Type::Primitive(PrimitiveType::Int),
            )
            .into(),
        );
    }

    Schema::builder().with_fields(fields).build()
}

/// Builds the table's unified partition type — the Rust port of Java
/// `Partitioning.partitionType(table)` → `buildPartitionProjectionType`.
///
/// One struct field per UNIQUE partition field id across all of the table's specs (deduped by id),
/// keeping only field ids whose source column still exists in the current schema. Specs are visited
/// spec-id DESCENDING so the newest spec's name/type wins on a shared id; the output fields are sorted
/// by field id ASCENDING and each made `optional`. A data partition under two specs that share a
/// partition field id therefore lands in ONE unified field, not two.
///
/// This re-derives the unifier locally rather than reusing `inspect::partitions` — that table keys
/// rows by the file's OWN partition `Struct` against `default_partition_type()` and documents that the
/// Rust core has no cross-spec unifier (see `inspect/partitions.rs` module doc). It also re-derives
/// rather than touching `spec/` (READ-ONLY) — it is built entirely from the public `PartitionSpec`
/// accessors.
///
/// # Errors
///
/// Returns an error if a spec's `partition_type(schema)` cannot be resolved.
pub fn unified_partition_type(metadata: &TableMetadata) -> Result<StructType> {
    let schema = metadata.current_schema();

    // Collect specs sorted by spec id DESCENDING (newest first → most recent field names win).
    let mut specs: Vec<&PartitionSpec> = metadata
        .partition_specs_iter()
        .map(|spec_ref| spec_ref.as_ref())
        .collect();
    specs.sort_by_key(|spec| std::cmp::Reverse(spec.spec_id()));

    // fieldId -> (name, type), keeping the first-seen (newest-spec) entry per field id.
    let mut fields_by_id: HashMap<i32, (String, Type)> = HashMap::new();
    // Track insertion shape so the output can be field-id ASCENDING (Java's natural-order sort).
    let mut field_ids: Vec<i32> = Vec::new();

    for spec in &specs {
        let spec_type = spec.partition_type(schema)?;
        for (field, struct_field) in spec.fields().iter().zip(spec_type.fields().iter()) {
            // Java keeps only field ids whose source column is still present in the current schema
            // (`allActiveFieldIds`: schema.findField(field.sourceId()) != null).
            if schema.field_by_id(field.source_id).is_none() {
                continue;
            }
            let field_id = field.field_id;
            if fields_by_id.contains_key(&field_id) {
                // Newest spec already supplied the name/type for this id; Java additionally swaps in
                // a non-void field's type over a void one for dropped v1 partitions. The Rust core
                // does not model void-transform partition types distinctly here, and the conflict
                // validation Java runs is on identical-id fields that are equivalent-ignoring-names;
                // keeping the newest-spec entry matches the common (non-v1-void) case.
                continue;
            }
            fields_by_id.insert(
                field_id,
                (
                    struct_field.name.clone(),
                    struct_field.field_type.as_ref().clone(),
                ),
            );
            field_ids.push(field_id);
        }
    }

    field_ids.sort_unstable();
    let unified_fields = field_ids
        .into_iter()
        .map(|field_id| {
            let (name, field_type) = fields_by_id
                .get(&field_id)
                .expect("field id was just inserted into fields_by_id");
            NestedField::optional(field_id, name.clone(), field_type.clone()).into()
        })
        .collect::<Vec<_>>();

    Ok(StructType::new(unified_fields))
}

/// Coerces a file's per-spec partition tuple into the table's unified partition type — the Rust port
/// of Java `PartitionUtil.coercePartition` (`StructProjection.createAllowMissing(spec.partitionType,
/// unifiedType)`).
///
/// For each unified field (by id), the value is taken from the position in the file's spec whose
/// partition field has the matching id; a unified field absent from this spec is null-filled. The
/// file's partition `Struct` is positional against `spec.partition_type(schema).fields()`, which is in
/// `spec.fields()` order — so the index of a unified field id within the spec's partition type is the
/// index into the file's partition tuple.
fn coerce_partition(
    unified_partition_type: &StructType,
    spec_type: &StructType,
    file_partition: &Struct,
) -> Struct {
    let spec_fields = spec_type.fields();
    let file_values = file_partition.fields();

    unified_partition_type
        .fields()
        .iter()
        .map(|unified_field| {
            // Find the position of this unified field id within the file's spec.
            let position = spec_fields
                .iter()
                .position(|spec_field| spec_field.id == unified_field.id);
            match position {
                // Null-fill when the field is absent from this spec, or (defensively) when the file's
                // tuple is shorter than its declared spec type.
                Some(index) if index < file_values.len() => file_values[index].clone(),
                _ => None,
            }
        })
        .collect()
}

/// Computes per-partition statistics for a snapshot — the Rust port of Java 1.10.0
/// `PartitionStatsHandler.computeAndWriteStatsFile(table, snapshotId)`'s *full-compute* branch (the
/// X1 compute core; the file write + registration is X2).
///
/// Mirrors Java's structure exactly: `computeStats` collects a per-manifest `PartitionMap`
/// ([`collect_stats_for_manifest`]) for every manifest in `snapshot.allManifests` (all data + delete
/// manifests), then folds each per-manifest map into the final map with `appendStats`
/// ([`merge_partition_map`] → [`PartitionStats::append_stats`]). Within one manifest, every entry —
/// LIVE and DELETED — is processed:
///
/// - a LIVE entry updates the content-type counters + last-updated info ([`PartitionStats::live_entry`]);
/// - a DELETED tombstone updates only last-updated info ([`PartitionStats::deleted_entry`]).
///
/// The result is sorted by partition tuple (Java `sortStatsByPartition` with
/// `Comparators.forType(partitionType)`). `total_record_count` is never set (Java does not compute it).
///
/// # Errors
///
/// - `DataInvalid` "Table must be partitioned" if no spec has a non-void partition field (Java
///   `Preconditions.checkArgument(Partitioning.isPartitioned(table))`).
/// - Propagates manifest-list / manifest read errors and a spec-id-mismatch on merge.
///
/// # Edge cases (pinned from the 1.10.0 bytecode + `PartitionStatsHandlerTestBase`)
///
/// - An UNPARTITIONED table is an error, NOT an empty result.
/// - A snapshot with no manifests, or an empty table branch, yields an empty `Vec`.
/// - A partition present only via DELETE files yields a row with zero data counters + the delete
///   counters set.
/// - A fully-deleted partition keeps a zero-count row (the DELETED tombstone still creates the row).
pub async fn compute_partition_stats(
    table: &Table,
    snapshot: &Snapshot,
) -> Result<Vec<PartitionStats>> {
    let metadata = table.metadata();

    // Java: Preconditions.checkArgument(Partitioning.isPartitioned(table), "Table must be partitioned").
    // isPartitioned == any spec has at least one non-void field == NOT all specs unpartitioned.
    if metadata
        .partition_specs_iter()
        .all(|spec| spec.is_unpartitioned())
    {
        return Err(Error::new(
            crate::ErrorKind::DataInvalid,
            "Table must be partitioned",
        ));
    }

    let unified_type = unified_partition_type(metadata)?;
    let schema = metadata.current_schema();
    let file_io = table.file_io();

    // The final per-(spec_id, coerced-partition) map (Java's `statsMap`). Each manifest produces its
    // own map, which is merged into this one via append_stats (Java `mergePartitionMap`).
    let mut stats_by_key: HashMap<(i32, Struct), PartitionStats> = HashMap::new();

    let manifest_list = snapshot.load_manifest_list(file_io, metadata).await?;
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(file_io).await?;
        // All files in one manifest share the manifest's partition spec id; resolve the spec's
        // partition type ONCE per manifest (the residual/constants-map per-manifest idiom).
        let spec_id = manifest_file.partition_spec_id;
        let spec_type = resolve_spec_partition_type(metadata, schema, spec_id)?;

        let per_manifest =
            collect_stats_for_manifest(metadata, &manifest, spec_id, &spec_type, &unified_type);
        merge_partition_map(per_manifest, &mut stats_by_key)?;
    }

    let mut stats: Vec<PartitionStats> = stats_by_key.into_values().collect();
    stats.sort_by(|left, right| compare_partition_values(&left.partition, &right.partition));
    Ok(stats)
}

/// Collects one manifest's per-partition statistics into a fresh map — the Rust port of Java
/// `PartitionStatsHandler.collectStatsForManifest` (full-compute, `incremental=false`).
///
/// Iterates every entry (LIVE + DELETED), coerces the file's partition into the unified type, and
/// keys the row by `(spec_id, coerced-partition)`. A LIVE entry rolls up the content-type counters;
/// a DELETED tombstone updates only last-updated info — but both create/keep the partition row.
fn collect_stats_for_manifest(
    metadata: &TableMetadata,
    manifest: &Manifest,
    spec_id: i32,
    spec_type: &StructType,
    unified_type: &StructType,
) -> HashMap<(i32, Struct), PartitionStats> {
    let mut stats_map: HashMap<(i32, Struct), PartitionStats> = HashMap::new();

    for entry in manifest.entries() {
        let data_file = entry.data_file();
        let coerced = coerce_partition(unified_type, spec_type, data_file.partition());

        // The entry's OWN snapshot (Java table.snapshot(entry.snapshotId())) supplies the
        // last-updated timestamp/id; absent if that snapshot is not in the table's metadata.
        let (snapshot_timestamp_ms, snapshot_id) = entry_snapshot_info(metadata, entry);

        let row = stats_map
            .entry((spec_id, coerced.clone()))
            .or_insert_with(|| PartitionStats::new(coerced, spec_id));

        if entry.is_alive() {
            accumulate_live_entry(row, data_file, snapshot_timestamp_ms, snapshot_id);
        } else {
            row.deleted_entry(snapshot_timestamp_ms, snapshot_id);
        }
    }

    stats_map
}

/// Folds one manifest's stats map into the running total — the Rust port of Java `mergePartitionMap`.
/// A key present in both maps is merged via [`PartitionStats::append_stats`]; a key new to the target
/// is inserted as-is.
///
/// # Errors
///
/// Propagates the spec-id-mismatch error from `append_stats` (never reachable in practice — the same
/// `(spec_id, partition)` key always carries the same spec id).
fn merge_partition_map(
    from_map: HashMap<(i32, Struct), PartitionStats>,
    to_map: &mut HashMap<(i32, Struct), PartitionStats>,
) -> Result<()> {
    for (key, value) in from_map {
        match to_map.get_mut(&key) {
            Some(existing) => existing.append_stats(&value)?,
            None => {
                to_map.insert(key, value);
            }
        }
    }
    Ok(())
}

/// Resolves a manifest's spec's partition type, erroring loudly if the spec id is unknown (a
/// corrupt manifest list referencing a spec the metadata does not carry).
fn resolve_spec_partition_type(
    metadata: &TableMetadata,
    schema: &SchemaRef,
    spec_id: i32,
) -> Result<StructType> {
    let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
        Error::new(
            crate::ErrorKind::DataInvalid,
            format!("Cannot find partition spec for manifest: {spec_id}"),
        )
    })?;
    spec.partition_type(schema)
}

/// Pulls the `(timestamp_ms, snapshot_id)` for an entry's own snapshot, or `(None, None)` if the
/// entry has no snapshot id or the snapshot is absent from the metadata.
fn entry_snapshot_info(
    metadata: &TableMetadata,
    entry: &ManifestEntry,
) -> (Option<i64>, Option<i64>) {
    match entry.snapshot_id() {
        Some(id) => match metadata.snapshot_by_id(id) {
            Some(snapshot) => (Some(snapshot.timestamp_ms()), Some(snapshot.snapshot_id())),
            None => (None, None),
        },
        None => (None, None),
    }
}

/// Applies a live entry's file to the row, narrowing the file's `u64` counts to the `i64`/`i32`
/// schema widths (saturating, so a hostile on-disk count cannot wrap or panic in a debug build).
fn accumulate_live_entry(
    row: &mut PartitionStats,
    data_file: &DataFile,
    snapshot_timestamp_ms: Option<i64>,
    snapshot_id: Option<i64>,
) {
    let record_count = i64::try_from(data_file.record_count()).unwrap_or(i64::MAX);
    let file_size = i64::try_from(data_file.file_size_in_bytes()).unwrap_or(i64::MAX);
    row.live_entry(
        data_file.content_type(),
        data_file.file_format(),
        record_count,
        file_size,
        snapshot_timestamp_ms,
        snapshot_id,
    );
}

/// Compares two unified-partition tuples field-by-field for the deterministic output sort — the local
/// analogue of Java `Comparators.forType(partitionType)`. Mirrors `inspect::partitions`'s comparator
/// (which is in a READ-ONLY module): null sorts before any value, primitive values compare via
/// [`PrimitiveLiteral`]'s `PartialOrd`, and any incomparable pair falls back to `Equal` so the order
/// stays total + deterministic under a stable sort.
fn compare_partition_values(left: &Struct, right: &Struct) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let left_fields = left.fields();
    let right_fields = right.fields();
    let len = left_fields.len().min(right_fields.len());
    for index in 0..len {
        let ordering = compare_partition_field(&left_fields[index], &right_fields[index]);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left_fields.len().cmp(&right_fields.len())
}

/// Compares one optional partition field value; `None` (null) sorts before any value.
fn compare_partition_field(left: &Option<Literal>, right: &Option<Literal>) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(Literal::Primitive(left)), Some(Literal::Primitive(right))) => {
            compare_primitive(left, right)
        }
        // Non-primitive partition literals are not valid partition values; keep order stable.
        _ => Ordering::Equal,
    }
}

/// Compares two [`PrimitiveLiteral`]s, falling back to `Equal` for an incomparable pair.
fn compare_primitive(left: &PrimitiveLiteral, right: &PrimitiveLiteral) -> std::cmp::Ordering {
    left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::io::{FileIO, FileIOBuilder, LocalFsStorageFactory};
    use crate::memory::MemoryCatalogBuilder;
    use crate::spec::{
        DataFileBuilder, ManifestListWriter, NestedFieldRef, Transform, UnboundPartitionField,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

    /// A fixed data-file size for the fixtures (so total-size assertions are exact multiples).
    const DATA_FILE_SIZE: u64 = 100;
    /// A fixed delete-file size for the fixtures.
    const DELETE_FILE_SIZE: u64 = 7;

    // =========================================================================================
    // Pure-fn tests — schema field ids, the row type, the unifier, coercion
    //
    // These pin the exact-byte / exact-id surface deterministically without a catalog (the schema
    // field ids become on-disk parquet field ids in X2 and MUST match Java).
    // =========================================================================================

    /// Builds a `StructType` of `identity(x)` (field id 1000) for spec 0's partition type.
    fn x_partition_type() -> StructType {
        StructType::new(vec![Arc::new(NestedField::optional(
            1000,
            "x",
            Type::Primitive(PrimitiveType::Long),
        ))])
    }

    /// Builds a `StructType` of `identity(x), identity(y)` (field ids 1000, 1001) for the evolved
    /// spec 1's partition type.
    fn xy_partition_type() -> StructType {
        StructType::new(vec![
            Arc::new(NestedField::optional(
                1000,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                1001,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
    }

    /// RISK: the on-disk field ids drift from Java's constants → an X2 stats file Java cannot read,
    /// or a reader maps the wrong column. Pins every field id / name / type / nullability for the v2
    /// schema (12 fields, delete fields OPTIONAL, no dv_count).
    #[test]
    fn test_v2_schema_has_java_exact_field_ids_names_types_and_nullability() {
        let schema = partition_stats_schema(&x_partition_type(), FormatVersion::V2).unwrap();
        let fields = schema.as_struct().fields();
        assert_eq!(fields.len(), 12, "v2 schema has 12 fields");

        // (id, name, is_required, is_long): the Java v2Schema, field-for-field.
        let expected: [(i32, &str, bool, Option<bool>); 12] = [
            (1, "partition", true, None),
            (2, "spec_id", true, Some(false)),
            (3, "data_record_count", true, Some(true)),
            (4, "data_file_count", true, Some(false)),
            (5, "total_data_file_size_in_bytes", true, Some(true)),
            (6, "position_delete_record_count", false, Some(true)),
            (7, "position_delete_file_count", false, Some(false)),
            (8, "equality_delete_record_count", false, Some(true)),
            (9, "equality_delete_file_count", false, Some(false)),
            (10, "total_record_count", false, Some(true)),
            (11, "last_updated_at", false, Some(true)),
            (12, "last_updated_snapshot_id", false, Some(true)),
        ];
        for (field, (id, name, required, is_long)) in fields.iter().zip(expected) {
            assert_eq!(field.id, id, "field id for {name}");
            assert_eq!(field.name, name, "field name for id {id}");
            assert_eq!(field.required, required, "nullability for {name}");
            match is_long {
                Some(true) => assert_eq!(
                    *field.field_type,
                    Type::Primitive(PrimitiveType::Long),
                    "{name} is long"
                ),
                Some(false) => assert_eq!(
                    *field.field_type,
                    Type::Primitive(PrimitiveType::Int),
                    "{name} is int"
                ),
                None => assert!(
                    field.field_type.is_struct(),
                    "{name} is the partition struct"
                ),
            }
        }

        // The partition struct field carries the unified partition type verbatim.
        if let Type::Struct(partition_struct) = fields[0].field_type.as_ref() {
            assert_eq!(partition_struct, &x_partition_type());
        } else {
            panic!("field 1 must be a struct");
        }
    }

    /// RISK: the v3 schema must MAKE the delete fields REQUIRED and ADD dv_count(13) — a v3 table's
    /// stats file otherwise has the wrong nullability/columns vs Java. Pins the v2→v3 difference.
    #[test]
    fn test_v3_schema_makes_delete_fields_required_and_adds_dv_count() {
        let schema = partition_stats_schema(&x_partition_type(), FormatVersion::V3).unwrap();
        let fields = schema.as_struct().fields();
        assert_eq!(fields.len(), 13, "v3 schema has 13 fields");

        // Fields 6-9 (the position/equality delete fields) are REQUIRED in v3 (OPTIONAL in v2).
        for id in [6, 7, 8, 9] {
            let field = schema.field_by_id(id).unwrap();
            assert!(field.required, "v3 field id {id} must be required");
        }
        // total_record_count / last_updated_* stay OPTIONAL in v3.
        for id in [10, 11, 12] {
            assert!(
                !schema.field_by_id(id).unwrap().required,
                "v3 field id {id} optional"
            );
        }
        // dv_count(13) is added, required, Int.
        let dv = schema.field_by_id(13).unwrap();
        assert_eq!(dv.name, "dv_count");
        assert!(dv.required);
        assert_eq!(*dv.field_type, Type::Primitive(PrimitiveType::Int));
    }

    /// RISK: `schema(StructType)` defaults to v2 and an EMPTY partition type is rejected ("Table must
    /// be partitioned"). A schema built over an empty struct would have a degenerate partition column.
    #[test]
    fn test_schema_for_format_version_picks_v2_for_v1_and_v2_v3_for_v3() {
        assert_eq!(
            partition_stats_schema(&x_partition_type(), FormatVersion::V1)
                .unwrap()
                .as_struct()
                .fields()
                .len(),
            12,
            "v1 uses the v2 schema (12 fields)"
        );
        assert_eq!(
            partition_stats_schema(&x_partition_type(), FormatVersion::V2)
                .unwrap()
                .as_struct()
                .fields()
                .len(),
            12
        );
        assert_eq!(
            partition_stats_schema(&x_partition_type(), FormatVersion::V3)
                .unwrap()
                .as_struct()
                .fields()
                .len(),
            13
        );

        let empty = StructType::new(vec![]);
        let error = partition_stats_schema(&empty, FormatVersion::V2).unwrap_err();
        assert!(error.message().contains("Table must be partitioned"));
    }

    /// A partition tuple `(x)` of a single long value.
    fn x_struct(x: i64) -> Struct {
        Struct::from_iter([Some(Literal::long(x))])
    }

    /// RISK: per-content-type counter ROUTING — a swap (data↔delete, pos↔eq, parquet-pos↔DV) silently
    /// reports the wrong category. Pins each `live_entry` arm in isolation onto a single row.
    #[test]
    fn test_live_entry_routes_each_content_type_to_its_own_counters() {
        // DATA → data_record_count += records, data_file_count += 1, total_size += size.
        let mut data_row = PartitionStats::new(x_struct(1), 0);
        data_row.live_entry(
            DataContentType::Data,
            DataFileFormat::Parquet,
            5,
            100,
            None,
            None,
        );
        assert_eq!(data_row.data_record_count(), 5);
        assert_eq!(data_row.data_file_count(), 1);
        assert_eq!(data_row.total_data_file_size_in_bytes(), 100);
        assert_eq!(data_row.position_delete_record_count(), 0);
        assert_eq!(data_row.equality_delete_record_count(), 0);
        assert_eq!(data_row.dv_count(), 0);

        // POSITION_DELETES, parquet → position_delete_record_count += records, position_delete_file_count += 1.
        let mut pos_row = PartitionStats::new(x_struct(1), 0);
        pos_row.live_entry(
            DataContentType::PositionDeletes,
            DataFileFormat::Parquet,
            3,
            7,
            None,
            None,
        );
        assert_eq!(pos_row.position_delete_record_count(), 3);
        assert_eq!(pos_row.position_delete_file_count(), 1);
        assert_eq!(pos_row.dv_count(), 0, "a parquet pos delete is NOT a DV");
        assert_eq!(pos_row.data_record_count(), 0);
        assert_eq!(
            pos_row.total_data_file_size_in_bytes(),
            0,
            "delete size is not summed"
        );

        // POSITION_DELETES, PUFFIN → position_delete_record_count += records, dv_count += 1 (NOT file count).
        let mut dv_row = PartitionStats::new(x_struct(1), 0);
        dv_row.live_entry(
            DataContentType::PositionDeletes,
            DataFileFormat::Puffin,
            4,
            9,
            None,
            None,
        );
        assert_eq!(dv_row.position_delete_record_count(), 4);
        assert_eq!(dv_row.dv_count(), 1, "a PUFFIN pos delete IS a DV");
        assert_eq!(
            dv_row.position_delete_file_count(),
            0,
            "DV does not bump the file count"
        );

        // EQUALITY_DELETES → equality_delete_record_count += records, equality_delete_file_count += 1.
        let mut eq_row = PartitionStats::new(x_struct(1), 0);
        eq_row.live_entry(
            DataContentType::EqualityDeletes,
            DataFileFormat::Parquet,
            6,
            11,
            None,
            None,
        );
        assert_eq!(eq_row.equality_delete_record_count(), 6);
        assert_eq!(eq_row.equality_delete_file_count(), 1);
        assert_eq!(eq_row.position_delete_record_count(), 0);
        assert_eq!(eq_row.dv_count(), 0);
    }

    /// RISK: last-updated semantics — the MAX timestamp must win (strict `<`), and a TIE must keep the
    /// FIRST-seen snapshot (a `<=` regression would let a later equal-time snapshot misattribute the
    /// partition). Also: an OLDER timestamp seen AFTER a newer one must NOT overwrite. This is the
    /// last-updated routing bug the brief calls out.
    #[test]
    fn test_update_snapshot_info_keeps_max_timestamp_strict_and_first_on_tie() {
        let mut row = PartitionStats::new(x_struct(1), 0);

        // First update sets it.
        row.update_snapshot_info(/* snapshot_id */ 10, /* updated_at_ms */ 1000);
        assert_eq!(row.last_updated_at(), Some(1000));
        assert_eq!(row.last_updated_snapshot_id(), Some(10));

        // A strictly-newer timestamp wins.
        row.update_snapshot_info(20, 2000);
        assert_eq!(row.last_updated_at(), Some(2000));
        assert_eq!(row.last_updated_snapshot_id(), Some(20));

        // An OLDER timestamp does NOT overwrite.
        row.update_snapshot_info(30, 1500);
        assert_eq!(row.last_updated_at(), Some(2000));
        assert_eq!(
            row.last_updated_snapshot_id(),
            Some(20),
            "older must not win"
        );

        // A TIE (exact same timestamp) keeps the first-seen snapshot (strict `<`).
        row.update_snapshot_info(40, 2000);
        assert_eq!(
            row.last_updated_snapshot_id(),
            Some(20),
            "a tie must keep the first-seen snapshot, not snapshot 40"
        );
    }

    /// RISK: a DELETED tombstone must NOT touch any counter, only last-updated — and it must still
    /// leave the row present (zero counts). A delete-entry that decremented (the incremental path's
    /// behavior, which is X2) would corrupt a full compute.
    #[test]
    fn test_deleted_entry_updates_only_last_updated_no_counter() {
        let mut row = PartitionStats::new(x_struct(1), 0);
        row.live_entry(
            DataContentType::Data,
            DataFileFormat::Parquet,
            5,
            100,
            Some(1000),
            Some(10),
        );
        assert_eq!(row.data_record_count(), 5);

        // A deleted tombstone in a NEWER snapshot bumps last-updated but leaves counters untouched.
        row.deleted_entry(Some(2000), Some(20));
        assert_eq!(
            row.data_record_count(),
            5,
            "delete must not decrement in full compute"
        );
        assert_eq!(row.data_file_count(), 1);
        assert_eq!(
            row.last_updated_at(),
            Some(2000),
            "delete bumps last-updated"
        );
        assert_eq!(row.last_updated_snapshot_id(), Some(20));
    }

    /// RISK: the `appendStats` merge — every primitive counter must ADD (including dv_count, which the
    /// 1.10.0 jar adds UNCONDITIONALLY), the nullable `total_record_count` must set-if-null-else-add,
    /// and last-updated must re-evaluate against the input. Pins the merge primitive used to fold
    /// per-manifest maps together.
    #[test]
    fn test_append_stats_adds_all_counters_and_merges_nullables() {
        let mut target = PartitionStats::new(x_struct(1), 0);
        target.live_entry(
            DataContentType::Data,
            DataFileFormat::Parquet,
            5,
            100,
            Some(1000),
            Some(10),
        );
        target.live_entry(
            DataContentType::PositionDeletes,
            DataFileFormat::Puffin,
            1,
            9,
            Some(1000),
            Some(10),
        );
        // target has total_record_count None (full compute never sets it); set it for the merge test.
        target.total_record_count = Some(2);

        let mut input = PartitionStats::new(x_struct(1), 0);
        input.live_entry(
            DataContentType::Data,
            DataFileFormat::Parquet,
            7,
            200,
            Some(3000),
            Some(30),
        );
        input.live_entry(
            DataContentType::EqualityDeletes,
            DataFileFormat::Parquet,
            4,
            11,
            Some(3000),
            Some(30),
        );
        input.live_entry(
            DataContentType::PositionDeletes,
            DataFileFormat::Puffin,
            2,
            9,
            Some(3000),
            Some(30),
        );
        input.total_record_count = Some(3);

        target.append_stats(&input).unwrap();

        assert_eq!(target.data_record_count(), 5 + 7);
        assert_eq!(target.data_file_count(), 1 + 1);
        assert_eq!(target.total_data_file_size_in_bytes(), 100 + 200);
        assert_eq!(target.equality_delete_record_count(), 4);
        assert_eq!(target.equality_delete_file_count(), 1);
        assert_eq!(target.dv_count(), 1 + 1, "dv_count adds unconditionally");
        assert_eq!(
            target.total_record_count(),
            Some(2 + 3),
            "nullable adds when both present"
        );
        // input's last-updated (3000 @ 30) is newer than target's (1000 @ 10) → it wins.
        assert_eq!(target.last_updated_at(), Some(3000));
        assert_eq!(target.last_updated_snapshot_id(), Some(30));
    }

    /// RISK: `append_stats` set-if-null on `total_record_count` — when the target is None and the input
    /// has a value, the value must transfer (not stay None, not panic on add).
    #[test]
    fn test_append_stats_sets_total_record_count_when_target_is_null() {
        let mut target = PartitionStats::new(x_struct(1), 0);
        let mut input = PartitionStats::new(x_struct(1), 0);
        input.total_record_count = Some(42);
        target.append_stats(&input).unwrap();
        assert_eq!(target.total_record_count(), Some(42));
    }

    /// RISK: the spec-id-must-match precondition on merge — two rows with different spec ids must NOT
    /// be merged silently (that would conflate distinct evolved-spec partitions). Pins the guard fires.
    #[test]
    fn test_append_stats_rejects_mismatched_spec_ids() {
        let mut target = PartitionStats::new(x_struct(1), 0);
        let input = PartitionStats::new(x_struct(1), 1);
        let error = target.append_stats(&input).unwrap_err();
        assert!(error.message().contains("Spec IDs must match"));
    }

    /// RISK: the unified partition type — across two specs `(x)` (spec 0) and `(x, y)` (spec 1) sharing
    /// field id 1000 for `x`, the result must be ONE struct with TWO fields {1000:x, 1001:y} sorted by
    /// id ASCENDING, NOT a duplicated `x` nor a per-spec struct. A wrong unifier corrupts the stats
    /// file's keying across evolved specs.
    #[test]
    fn test_unified_partition_type_dedups_shared_field_id_and_sorts_ascending() {
        let metadata = two_spec_metadata();
        let unified = unified_partition_type(&metadata).unwrap();
        let fields = unified.fields();
        assert_eq!(fields.len(), 2, "x and y dedup to two unified fields");
        assert_eq!(fields[0].id, 1000, "sorted ascending: x first");
        assert_eq!(fields[0].name, "x");
        assert_eq!(fields[1].id, 1001, "then y");
        assert_eq!(fields[1].name, "y");
        // Both unified fields are OPTIONAL (Java `NestedField.optional`).
        assert!(!fields[0].required && !fields[1].required);
    }

    /// RISK: coercion DIRECTION 1 — a spec-1 file `(x, y)` projects identically into the unified type
    /// {x, y} (both values carried through, in unified order).
    #[test]
    fn test_coerce_partition_carries_full_tuple_for_the_newer_spec() {
        let unified = xy_partition_type();
        let spec1_type = xy_partition_type();
        let file_partition = Struct::from_iter([Some(Literal::long(7)), Some(Literal::long(9))]);
        let coerced = coerce_partition(&unified, &spec1_type, &file_partition);
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(7)), Some(Literal::long(9))])
        );
    }

    /// RISK: coercion DIRECTION 2 + the NULL-FILL — a spec-0 file `(x)` projects into the unified type
    /// {x, y} as `(x, NULL)`: `x` carries through, `y` (absent from spec 0) is null-filled. Dropping
    /// the null-fill (e.g. carrying the wrong index) would key the partition incorrectly.
    #[test]
    fn test_coerce_partition_null_fills_field_absent_from_the_older_spec() {
        let unified = xy_partition_type();
        let spec0_type = x_partition_type();
        let file_partition = Struct::from_iter([Some(Literal::long(7))]);
        let coerced = coerce_partition(&unified, &spec0_type, &file_partition);
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(7)), None]),
            "y must be null-filled for a spec-0 file"
        );
    }

    /// RISK (the load-bearing coercion pin): when the file's spec lists its partition fields in a
    /// DIFFERENT positional order than the unified type's ascending-by-id order, coercion MUST remap by
    /// FIELD ID, not by position. A "drop the spec-coercion" mutation that indexes by the unified
    /// position would read the WRONG value here (the same-order fixtures above mask it because spec
    /// order coincides with ascending-id order). The spec's tuple is `[y=9 @ id 1001, x=7 @ id 1000]`
    /// (y FIRST), and the unified type is ascending `{x @ 1000, y @ 1001}` — so the coerced tuple must
    /// be `(x=7, y=9)`, which requires pulling spec index 1 for x and index 0 for y.
    #[test]
    fn test_coerce_partition_remaps_by_field_id_not_position() {
        let unified = xy_partition_type(); // {x @ 1000, y @ 1001} (ascending)
        // The spec lists y (id 1001) at index 0 and x (id 1000) at index 1 — REVERSED order.
        let spec_type = StructType::new(vec![
            Arc::new(NestedField::optional(
                1001,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                1000,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
        ]);
        // The file's tuple matches the spec order: [y=9, x=7].
        let file_partition = Struct::from_iter([Some(Literal::long(9)), Some(Literal::long(7))]);
        let coerced = coerce_partition(&unified, &spec_type, &file_partition);
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(7)), Some(Literal::long(9))]),
            "coercion must remap by field id: unified (x,y) = (7,9), not the file's (9,7)"
        );
    }

    /// Builds a `TableMetadata` with two specs — spec 0 = `identity(x)`, spec 1 = `identity(x),
    /// identity(y)` — over a `{x: long, y: long, z: long}` schema, with NO snapshots. Used by the
    /// unifier pure-fn tests.
    fn two_spec_metadata() -> TableMetadata {
        use crate::spec::{PartitionSpec, TableMetadataBuilder};

        let schema = Schema::builder()
            .with_fields(vec![
                NestedFieldRef::from(NestedField::required(
                    1,
                    "x",
                    Type::Primitive(PrimitiveType::Long),
                )),
                NestedFieldRef::from(NestedField::required(
                    2,
                    "y",
                    Type::Primitive(PrimitiveType::Long),
                )),
                NestedFieldRef::from(NestedField::required(
                    3,
                    "z",
                    Type::Primitive(PrimitiveType::Long),
                )),
            ])
            .build()
            .unwrap();

        // Spec 0: identity(x), field id 1000.
        let spec0 = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: Some(1000),
                name: "x".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap();

        let creation = TableCreation::builder()
            .name("t".to_string())
            .location("memory://t".to_string())
            .schema(schema.clone())
            .partition_spec(spec0.into_unbound())
            .build();
        let builder = TableMetadataBuilder::from_table_creation(creation).unwrap();
        let metadata = builder.build().unwrap().metadata;

        // Evolve to spec 1: identity(x), identity(y) — field ids 1000, 1001.
        let unbound_spec1 = crate::spec::UnboundPartitionSpec::builder()
            .with_spec_id(1)
            .add_partition_fields(vec![
                UnboundPartitionField {
                    source_id: 1,
                    field_id: Some(1000),
                    name: "x".to_string(),
                    transform: Transform::Identity,
                },
                UnboundPartitionField {
                    source_id: 2,
                    field_id: Some(1001),
                    name: "y".to_string(),
                    transform: Transform::Identity,
                },
            ])
            .unwrap()
            .build();
        TableMetadataBuilder::new_from_metadata(metadata, None)
            .add_default_partition_spec(unbound_spec1)
            .unwrap()
            .build()
            .unwrap()
            .metadata
    }

    // =========================================================================================
    // End-to-end tests — the full traversal over a real MemoryCatalog table's committed manifests
    //
    // The catalog assigns wall-clock snapshot timestamps, so last-updated assertions derive their
    // expected ids/timestamps from the ACTUAL committed snapshots (the LATER commit wins) — the
    // hand-derivation is of the SEMANTICS (max-timestamp snapshot), not a hard-coded clock value.
    // The strict-`<` tie and the exact last-updated arithmetic are pinned by the pure-fn tests above.
    // =========================================================================================

    /// A local-filesystem-backed MemoryCatalog (so the producer writes real avro manifests on disk
    /// that [`compute_partition_stats`] reads back).
    async fn e2e_catalog() -> (impl Catalog, FileIO, TempDir) {
        let temp_dir = TempDir::new().expect("temp dir");
        let warehouse = temp_dir
            .path()
            .to_str()
            .expect("utf8 temp path")
            .to_string();
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "memory",
                HashMap::from([("warehouse".to_string(), warehouse)]),
            )
            .await
            .expect("load local-fs memory catalog");
        let file_io = FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build();
        (catalog, file_io, temp_dir)
    }

    /// Schema `{x: long, y: long, z: long}`.
    fn three_long_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedFieldRef::from(NestedField::required(
                    1,
                    "x",
                    Type::Primitive(PrimitiveType::Long),
                )),
                NestedFieldRef::from(NestedField::required(
                    2,
                    "y",
                    Type::Primitive(PrimitiveType::Long),
                )),
                NestedFieldRef::from(NestedField::required(
                    3,
                    "z",
                    Type::Primitive(PrimitiveType::Long),
                )),
            ])
            .build()
            .unwrap()
    }

    /// Creates a table partitioned by `identity(x)` (spec 0, field id 1000) under a fresh namespace.
    async fn create_x_partitioned_table(catalog: &impl Catalog) -> Table {
        let spec = crate::spec::PartitionSpec::builder(three_long_schema())
            .with_spec_id(0)
            .add_partition_field("x", "x", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(three_long_schema())
            .partition_spec(spec.into_unbound())
            .build();
        catalog.create_table(&namespace, creation).await.unwrap()
    }

    /// Creates an UNPARTITIONED table (empty spec).
    async fn create_unpartitioned_table(catalog: &impl Catalog) -> Table {
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(three_long_schema())
            .partition_spec(crate::spec::PartitionSpec::unpartition_spec().into_unbound())
            .build();
        catalog.create_table(&namespace, creation).await.unwrap()
    }

    /// Evolves the table's spec to `identity(x), identity(y)` (a new spec id) via the real
    /// `update_partition_spec` transaction.
    async fn evolve_to_xy_spec(catalog: &impl Catalog, table: &Table) -> Table {
        let tx = Transaction::new(table);
        let tx = tx.update_partition_spec().add_field("y").apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Writes `content` to `path` on disk.
    async fn write_file(file_io: &FileIO, path: &str, content: &[u8]) {
        file_io
            .new_output(path)
            .unwrap()
            .write(bytes::Bytes::copy_from_slice(content))
            .await
            .unwrap();
    }

    /// A real data file stamped with `spec_id` + the given partition tuple, holding `records` rows.
    async fn data_file(
        file_io: &FileIO,
        path: &str,
        spec_id: i32,
        partition: Struct,
        records: u64,
    ) -> DataFile {
        write_file(file_io, path, &vec![0u8; DATA_FILE_SIZE as usize]).await;
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(DATA_FILE_SIZE)
            .record_count(records)
            .partition_spec_id(spec_id)
            .partition(partition)
            .build()
            .unwrap()
    }

    /// A real parquet position-delete file stamped with `spec_id` + partition, holding `records` rows.
    async fn position_delete_file(
        file_io: &FileIO,
        path: &str,
        referenced: &str,
        spec_id: i32,
        partition: Struct,
        records: u64,
    ) -> DataFile {
        write_file(file_io, path, &vec![1u8; DELETE_FILE_SIZE as usize]).await;
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(DELETE_FILE_SIZE)
            .record_count(records)
            .partition_spec_id(spec_id)
            .partition(partition)
            .referenced_data_file(Some(referenced.to_string()))
            .build()
            .unwrap()
    }

    /// A real equality-delete file stamped with `spec_id` + partition, holding `records` rows.
    async fn equality_delete_file(
        file_io: &FileIO,
        path: &str,
        spec_id: i32,
        partition: Struct,
        records: u64,
    ) -> DataFile {
        write_file(file_io, path, &vec![2u8; DELETE_FILE_SIZE as usize]).await;
        DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(DELETE_FILE_SIZE)
            .record_count(records)
            .partition_spec_id(spec_id)
            .partition(partition)
            .equality_ids(Some(vec![1]))
            .build()
            .unwrap()
    }

    /// Fast-appends `files` to `table`, committed through `catalog`.
    async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Adds `delete_files` via a row delta, committed through `catalog`.
    async fn add_deletes(
        catalog: &impl Catalog,
        table: &Table,
        delete_files: Vec<DataFile>,
    ) -> Table {
        let tx = Transaction::new(table);
        let tx = tx.row_delta().add_deletes(delete_files).apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// The coerced unified tuple `(x, y)` (`y` may be `None`) — every row's partition key after the
    /// table has a 2-field unified type.
    fn xy_key(x: i64, y: Option<i64>) -> Struct {
        Struct::from_iter([Some(Literal::long(x)), y.map(Literal::long)])
    }

    /// Looks up the computed row for a coerced `(x, y)` tuple (`y == None` is the null-filled spec-0 row).
    fn row_for_xy(stats: &[PartitionStats], x: i64, y: Option<i64>) -> &PartitionStats {
        let key = xy_key(x, y);
        stats
            .iter()
            .find(|row| row.partition() == &key)
            .unwrap_or_else(|| panic!("no row for (x={x},y={y:?}) in {stats:#?}"))
    }

    /// CROWN JEWEL. A 2-spec table evolved mid-history, with data + position deletes + equality
    /// deletes across 3 partitions and 2 snapshots, computed and matched against a HAND-DERIVED
    /// expected table (every field, every partition, derived in comments from the fixture below — not
    /// from running the code).
    ///
    /// RISK: this is the whole point — a wrong aggregation (counter routing, spec coercion, key
    /// collapse, last-updated) silently misleads every downstream consumer.
    ///
    /// FIXTURE CONSTRUCTION (and the resulting HAND-DERIVED expected rows):
    ///
    /// Schema `{x,y,z: long}`. Spec 0 = `identity(x)` (field id 1000). Spec 1 = `identity(x),
    /// identity(y)` (field ids 1000, 1001). Unified type = {1000:x, 1001:y}.
    ///
    /// - Snapshot S1 (spec 0): append two DATA files —
    ///   * `d_x1` in partition (x=1): 3 records, size 100.
    ///   * `d_x2` in partition (x=2): 5 records, size 100.
    /// - Evolve spec 0 → spec 1.
    /// - Snapshot S2 (spec 1): append one DATA file —
    ///   * `d_x1y10` in partition (x=1, y=10): 7 records, size 100.
    /// - Snapshot S3 (spec 1): add deletes (row delta) —
    ///   * `pos_x1y10` a parquet POSITION delete in partition (x=1, y=10): 2 records, size 7.
    ///   * `eq_x1y10` an EQUALITY delete in partition (x=1, y=10): 4 records, size 7.
    ///
    /// Because spec 0's `(x=1)` file coerces to unified `(x=1, y=NULL)` and spec 1's `(x=1, y=10)`
    /// files coerce to `(x=1, y=10)`, these are TWO DISTINCT unified rows (different `y`), NOT merged.
    /// The expected rows (coerced partition → stats), keyed by `(spec_id, coerced-partition)`:
    ///
    /// | partition (x,y) | spec | data_rec | data_files | size | pos_rec | pos_files | eq_rec | eq_files | last_updated |
    /// |-----------------|------|----------|------------|------|---------|-----------|--------|----------|--------------|
    /// | (1, NULL)       | 0    | 3        | 1          | 100  | 0       | 0         | 0      | 0        | S1           |
    /// | (2, NULL)       | 0    | 5        | 1          | 100  | 0       | 0         | 0      | 0        | S1           |
    /// | (1, 10)         | 1    | 7        | 1          | 100  | 2       | 1         | 4      | 1        | S3           |
    ///
    /// total_record_count is None everywhere (never computed); dv_count is 0 (no PUFFIN).
    #[tokio::test]
    async fn test_crown_jewel_two_specs_data_and_deletes_match_hand_derived_table() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // Snapshot S1 (spec 0): two data files in partitions x=1 and x=2.
        let d_x1 = data_file(
            &file_io,
            &format!("{location}/data/x=1/d1.parquet"),
            0,
            x_struct(1),
            3,
        )
        .await;
        let d_x2 = data_file(
            &file_io,
            &format!("{location}/data/x=2/d2.parquet"),
            0,
            x_struct(2),
            5,
        )
        .await;
        let table = append(&catalog, &table, vec![d_x1, d_x2]).await;
        let s1 = table.metadata().current_snapshot().unwrap().snapshot_id();

        // Evolve to spec 1: identity(x), identity(y).
        let table = evolve_to_xy_spec(&catalog, &table).await;
        let spec1_id = table.metadata().default_partition_spec_id();
        assert_ne!(spec1_id, 0, "fixture sanity: spec evolved away from 0");

        // Snapshot S2 (spec 1): one data file in partition (x=1, y=10).
        let xy_partition = Struct::from_iter([Some(Literal::long(1)), Some(Literal::long(10))]);
        let d_x1y10 = data_file(
            &file_io,
            &format!("{location}/data/x=1/y=10/d3.parquet"),
            spec1_id,
            xy_partition.clone(),
            7,
        )
        .await;
        let table = append(&catalog, &table, vec![d_x1y10]).await;

        // Snapshot S3 (spec 1): a parquet position delete + an equality delete in (x=1, y=10).
        let pos = position_delete_file(
            &file_io,
            &format!("{location}/data/x=1/y=10/pos.parquet"),
            &format!("{location}/data/x=1/y=10/d3.parquet"),
            spec1_id,
            xy_partition.clone(),
            2,
        )
        .await;
        let eq = equality_delete_file(
            &file_io,
            &format!("{location}/data/x=1/y=10/eq.parquet"),
            spec1_id,
            xy_partition.clone(),
            4,
        )
        .await;
        let table = add_deletes(&catalog, &table, vec![pos, eq]).await;
        let s3 = table.metadata().current_snapshot().unwrap().snapshot_id();

        let snapshot = table.metadata().current_snapshot().unwrap();
        let stats = compute_partition_stats(&table, snapshot).await.unwrap();

        // Three distinct unified rows (every key is the 2-field unified tuple).
        assert_eq!(stats.len(), 3, "(1,NULL), (2,NULL), (1,10): {stats:#?}");

        // Output sorted by partition tuple: (1,NULL) < (1,10) < (2,NULL) (null y sorts before 10).
        assert_eq!(
            stats[0].partition(),
            &xy_key(1, None),
            "(1,NULL) sorts first"
        );
        assert_eq!(
            stats[1].partition(),
            &xy_key(1, Some(10)),
            "(1,10) sorts second (null y < 10)"
        );
        assert_eq!(
            stats[2].partition(),
            &xy_key(2, None),
            "(2,NULL) sorts last"
        );

        // Row (1, NULL) — spec 0, the x=1 data file only (y null-filled).
        let r1 = row_for_xy(&stats, 1, None);
        assert_eq!(r1.spec_id(), 0);
        assert_eq!(r1.data_record_count(), 3);
        assert_eq!(r1.data_file_count(), 1);
        assert_eq!(r1.total_data_file_size_in_bytes(), 100);
        assert_eq!(r1.position_delete_record_count(), 0);
        assert_eq!(r1.position_delete_file_count(), 0);
        assert_eq!(r1.equality_delete_record_count(), 0);
        assert_eq!(r1.equality_delete_file_count(), 0);
        assert_eq!(r1.dv_count(), 0);
        assert_eq!(r1.total_record_count(), None);
        assert_eq!(
            r1.last_updated_snapshot_id(),
            Some(s1),
            "(1,NULL) last touched by S1"
        );

        // Row (2, NULL) — spec 0, the x=2 data file only (y null-filled).
        let r2 = row_for_xy(&stats, 2, None);
        assert_eq!(r2.spec_id(), 0);
        assert_eq!(r2.data_record_count(), 5);
        assert_eq!(r2.data_file_count(), 1);
        assert_eq!(r2.total_data_file_size_in_bytes(), 100);
        assert_eq!(r2.position_delete_record_count(), 0);
        assert_eq!(r2.equality_delete_record_count(), 0);
        assert_eq!(r2.dv_count(), 0);
        assert_eq!(
            r2.last_updated_snapshot_id(),
            Some(s1),
            "(2,NULL) last touched by S1"
        );

        // Row (1, 10) — spec 1, the data file + both deletes.
        let r3 = row_for_xy(&stats, 1, Some(10));
        assert_eq!(r3.spec_id(), spec1_id);
        assert_eq!(r3.data_record_count(), 7);
        assert_eq!(r3.data_file_count(), 1);
        assert_eq!(
            r3.total_data_file_size_in_bytes(),
            100,
            "delete sizes are NOT in total size"
        );
        assert_eq!(r3.position_delete_record_count(), 2);
        assert_eq!(r3.position_delete_file_count(), 1);
        assert_eq!(r3.equality_delete_record_count(), 4);
        assert_eq!(r3.equality_delete_file_count(), 1);
        assert_eq!(r3.dv_count(), 0, "parquet deletes are not DVs");
        assert_eq!(r3.total_record_count(), None);
        assert_eq!(
            r3.last_updated_snapshot_id(),
            Some(s3),
            "(1,10) last touched by the delete S3"
        );
    }

    /// RISK: per-content-type ISOLATION across PARTITIONS — a partition with ONLY equality deletes,
    /// one with ONLY position deletes, and one with mixed must each report exactly their own
    /// categories (no cross-bleed). A counter-routing swap shows up here per partition.
    #[tokio::test]
    async fn test_per_content_type_counters_isolated_across_partitions() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // Three data partitions x=1, x=2, x=3 (one data file each, so a min-data row exists).
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                10,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/x=2/d.parquet"),
                0,
                x_struct(2),
                10,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/x=3/d.parquet"),
                0,
                x_struct(3),
                10,
            )
            .await,
        ])
        .await;

        // x=1: ONLY a position delete; x=2: ONLY an equality delete; x=3: BOTH.
        let table = add_deletes(&catalog, &table, vec![
            position_delete_file(
                &file_io,
                &format!("{location}/data/x=1/pos.parquet"),
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                2,
            )
            .await,
            equality_delete_file(
                &file_io,
                &format!("{location}/data/x=2/eq.parquet"),
                0,
                x_struct(2),
                3,
            )
            .await,
            position_delete_file(
                &file_io,
                &format!("{location}/data/x=3/pos.parquet"),
                &format!("{location}/data/x=3/d.parquet"),
                0,
                x_struct(3),
                4,
            )
            .await,
            equality_delete_file(
                &file_io,
                &format!("{location}/data/x=3/eq.parquet"),
                0,
                x_struct(3),
                5,
            )
            .await,
        ])
        .await;

        let snapshot = table.metadata().current_snapshot().unwrap();
        let stats = compute_partition_stats(&table, snapshot).await.unwrap();
        assert_eq!(stats.len(), 3);

        // x=1: ONLY position delete.
        let r1 = row_for_x_single(&stats, 1);
        assert_eq!(r1.position_delete_record_count(), 2);
        assert_eq!(r1.position_delete_file_count(), 1);
        assert_eq!(r1.equality_delete_record_count(), 0, "x=1 has no eq delete");
        assert_eq!(r1.equality_delete_file_count(), 0);

        // x=2: ONLY equality delete.
        let r2 = row_for_x_single(&stats, 2);
        assert_eq!(r2.equality_delete_record_count(), 3);
        assert_eq!(r2.equality_delete_file_count(), 1);
        assert_eq!(
            r2.position_delete_record_count(),
            0,
            "x=2 has no pos delete"
        );
        assert_eq!(r2.position_delete_file_count(), 0);

        // x=3: BOTH.
        let r3 = row_for_x_single(&stats, 3);
        assert_eq!(r3.position_delete_record_count(), 4);
        assert_eq!(r3.position_delete_file_count(), 1);
        assert_eq!(r3.equality_delete_record_count(), 5);
        assert_eq!(r3.equality_delete_file_count(), 1);
    }

    /// RISK: last-updated MAX across MULTIPLE snapshots touching ONE partition — three appends to the
    /// same partition x=1 must leave the partition's last-updated pointing at the NEWEST (third)
    /// snapshot, never the first. A misrouted last-updated (first-wins / wrong snapshot) fails here.
    #[tokio::test]
    async fn test_last_updated_tracks_newest_snapshot_for_a_multiply_updated_partition() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/a.parquet"),
                0,
                x_struct(1),
                1,
            )
            .await,
        ])
        .await;
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/b.parquet"),
                0,
                x_struct(1),
                1,
            )
            .await,
        ])
        .await;
        let s2 = table.metadata().current_snapshot().unwrap().snapshot_id();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/c.parquet"),
                0,
                x_struct(1),
                1,
            )
            .await,
        ])
        .await;
        let s3 = table.metadata().current_snapshot().unwrap().snapshot_id();

        let snapshot = table.metadata().current_snapshot().unwrap();
        let stats = compute_partition_stats(&table, snapshot).await.unwrap();
        assert_eq!(stats.len(), 1, "all three files are in partition x=1");
        let row = row_for_x_single(&stats, 1);
        assert_eq!(
            row.data_record_count(),
            3,
            "all three live data files counted"
        );
        assert_eq!(row.data_file_count(), 3);
        assert_eq!(
            row.last_updated_snapshot_id(),
            Some(s3),
            "last-updated points at the NEWEST snapshot, not the first"
        );
        assert_ne!(
            row.last_updated_snapshot_id(),
            Some(s2),
            "not the middle snapshot either"
        );
        // The last-updated timestamp equals the newest snapshot's commit time.
        let s3_ts = table.metadata().snapshot_by_id(s3).unwrap().timestamp_ms();
        assert_eq!(row.last_updated_at(), Some(s3_ts));
    }

    /// RISK: a partition present ONLY via a global/zero-data position delete must still produce a row
    /// with zero DATA counters and the delete counters set (a partition that has deletes but whose
    /// data lives elsewhere / was never added). Confirms the row is keyed by the delete file too.
    #[tokio::test]
    async fn test_partition_present_only_via_delete_files_has_zero_data_counters() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // Append a data file in x=1 (so the table has a snapshot to delta from), then add an equality
        // delete in a DIFFERENT partition x=2 (no data file there).
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                4,
            )
            .await,
        ])
        .await;
        let table = add_deletes(&catalog, &table, vec![
            equality_delete_file(
                &file_io,
                &format!("{location}/data/x=2/eq.parquet"),
                0,
                x_struct(2),
                9,
            )
            .await,
        ])
        .await;

        let snapshot = table.metadata().current_snapshot().unwrap();
        let stats = compute_partition_stats(&table, snapshot).await.unwrap();
        assert_eq!(stats.len(), 2, "x=1 (data) and x=2 (delete-only)");

        let delete_only = row_for_x_single(&stats, 2);
        assert_eq!(delete_only.data_record_count(), 0, "no data in x=2");
        assert_eq!(delete_only.data_file_count(), 0);
        assert_eq!(delete_only.total_data_file_size_in_bytes(), 0);
        assert_eq!(delete_only.equality_delete_record_count(), 9);
        assert_eq!(delete_only.equality_delete_file_count(), 1);
    }

    /// RISK: a fully-DELETED partition must keep a zero-count row (the DELETED tombstone still creates
    /// the row, Java `testCopyOnWriteDelete`). A traversal that skips DELETED entries (like the
    /// `inspect::partitions` table does) would DROP the partition entirely.
    #[tokio::test]
    async fn test_fully_deleted_partition_keeps_a_zero_count_row() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let d1 = data_file(
            &file_io,
            &format!("{location}/data/x=1/d.parquet"),
            0,
            x_struct(1),
            3,
        )
        .await;
        let table = append(&catalog, &table, vec![d1.clone()]).await;

        // Delete the only data file — the next snapshot's manifest carries a DELETED tombstone.
        let tx = Transaction::new(&table);
        let tx = tx.rewrite_files(vec![d1], vec![]).apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let delete_snapshot = table.metadata().current_snapshot().unwrap().snapshot_id();

        let snapshot = table.metadata().current_snapshot().unwrap();
        let stats = compute_partition_stats(&table, snapshot).await.unwrap();

        // The partition row survives with zero data counters (the tombstone created/kept it).
        assert_eq!(
            stats.len(),
            1,
            "the fully-deleted partition's row persists: {stats:#?}"
        );
        let row = row_for_x_single(&stats, 1);
        assert_eq!(
            row.data_record_count(),
            0,
            "the deleted file's records are gone"
        );
        assert_eq!(row.data_file_count(), 0);
        assert_eq!(
            row.last_updated_snapshot_id(),
            Some(delete_snapshot),
            "the DELETED tombstone bumps last-updated to the deleting snapshot"
        );
    }

    /// RISK: an UNPARTITIONED table is an ERROR ("Table must be partitioned"), NOT an empty result
    /// (Java `Preconditions.checkArgument(Partitioning.isPartitioned(table))`). A no-op-on-unpartitioned
    /// regression would silently produce nonsense rows keyed by the empty struct.
    #[tokio::test]
    async fn test_unpartitioned_table_is_an_error_not_empty() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let unpartitioned = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("{location}/data/d.parquet"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(DATA_FILE_SIZE)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .unwrap();
        write_file(
            &file_io,
            &format!("{location}/data/d.parquet"),
            &[0u8; DATA_FILE_SIZE as usize],
        )
        .await;
        let table = append(&catalog, &table, vec![unpartitioned]).await;

        let snapshot = table.metadata().current_snapshot().unwrap();
        let error = compute_partition_stats(&table, snapshot).await.unwrap_err();
        assert!(
            error.message().contains("Table must be partitioned"),
            "got: {error}"
        );
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    }

    /// RISK: a snapshot with NO manifests yields an empty `Vec`, never a spurious row. Pins the
    /// empty-result edge for a partitioned table (Java's empty-branch / no-files case).
    ///
    /// Built by committing a data file (so a real snapshot exists), then OVERWRITING that snapshot's
    /// manifest-list file on disk with an EMPTY manifest list — the traversal then reads zero
    /// manifests and must return an empty result.
    #[tokio::test]
    async fn test_snapshot_with_no_manifests_yields_no_rows() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                1,
            )
            .await,
        ])
        .await;

        // Overwrite the current snapshot's manifest list with an EMPTY one (zero manifests).
        let metadata = table.metadata();
        let current = metadata.current_snapshot().unwrap();
        let mut writer = ManifestListWriter::v2(
            file_io.new_output(current.manifest_list()).unwrap(),
            current.snapshot_id(),
            current.parent_snapshot_id(),
            current.sequence_number(),
        );
        writer.add_manifests(std::iter::empty()).unwrap();
        writer.close().await.unwrap();

        let stats = compute_partition_stats(&table, current).await.unwrap();
        assert!(
            stats.is_empty(),
            "a snapshot with no manifests has no rows: {stats:#?}"
        );
    }

    /// Looks up the computed row for a SINGLE-FIELD `(x)` tuple (the table has not evolved, so the
    /// unified type is 1-field and rows key by `(x)`).
    fn row_for_x_single(stats: &[PartitionStats], x: i64) -> &PartitionStats {
        stats
            .iter()
            .find(|row| row.partition() == &x_struct(x))
            .unwrap_or_else(|| panic!("no row for x={x} in {stats:#?}"))
    }

    // =========================================================================================
    // Reviewer-added pins (X1 review, 2026-06-12) — DV routing end-to-end, carried-EXISTING
    // attribution, explicit spec_id-across-evolution, and a second independent coercion-drop pin.
    // =========================================================================================

    /// A real PUFFIN deletion-vector file (content=PositionDeletes, format=Puffin, carries a
    /// referenced_data_file + content offset/size) stamped with `spec_id` + partition, holding
    /// `records` rows.
    async fn dv_file(
        file_io: &FileIO,
        path: &str,
        referenced: &str,
        spec_id: i32,
        partition: Struct,
        records: u64,
    ) -> DataFile {
        write_file(file_io, path, &vec![3u8; DELETE_FILE_SIZE as usize]).await;
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(DELETE_FILE_SIZE)
            .record_count(records)
            .partition_spec_id(spec_id)
            .partition(partition)
            .referenced_data_file(Some(referenced.to_string()))
            .content_offset(Some(0))
            .content_size_in_bytes(Some(DELETE_FILE_SIZE as i64))
            .build()
            .unwrap()
    }

    /// RISK (DV routing, end-to-end — the unmentioned semantic): a PUFFIN deletion vector must route
    /// into `dv_count` (+ `position_delete_record_count`), NOT `position_delete_file_count` (Java 1.10.0
    /// `PartitionStats.liveEntry` POSITION_DELETES branch: `if format == PUFFIN → dvCount++ else
    /// positionDeleteFileCount++`). Pinned with a REAL V3 table carrying a real Puffin DV — asserts the
    /// exact counter cells move (dv_count=1, pos_rec=N, pos_file=0).
    #[tokio::test]
    async fn test_puffin_dv_routes_to_dv_count_not_position_delete_file_count() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        // V3 table partitioned by identity(x) (DVs are only legal in V3).
        let spec = crate::spec::PartitionSpec::builder(three_long_schema())
            .with_spec_id(0)
            .add_partition_field("x", "x", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(three_long_schema())
            .partition_spec(spec.into_unbound())
            .format_version(FormatVersion::V3)
            .build();
        let table = catalog.create_table(&namespace, creation).await.unwrap();
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                10,
            )
            .await,
        ])
        .await;

        // Add a real Puffin DV (referenced_data_file set) in partition x=1.
        let table = add_deletes(&catalog, &table, vec![
            dv_file(
                &file_io,
                &format!("{location}/data/x=1/dv.puffin"),
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                6,
            )
            .await,
        ])
        .await;

        let snapshot = table.metadata().current_snapshot().unwrap();
        let stats = compute_partition_stats(&table, snapshot).await.unwrap();
        let row = row_for_x_single(&stats, 1);
        assert_eq!(row.dv_count(), 1, "the Puffin DV bumps dv_count");
        assert_eq!(
            row.position_delete_record_count(),
            6,
            "DV records go into position_delete_record_count"
        );
        assert_eq!(
            row.position_delete_file_count(),
            0,
            "a DV must NOT bump position_delete_file_count"
        );
        assert_eq!(row.equality_delete_file_count(), 0);
    }

    /// RISK (carried-EXISTING attribution): a manifest carried forward re-lists its files as EXISTING
    /// entries whose `snapshot_id` is the ORIGINAL committer (Rust `ManifestEntry::inherit_data` keeps
    /// the entry's own id; Java `collectStatsForManifest` keys last-updated off
    /// `table.snapshot(entry.snapshotId())`). A wrong attribution (keying off the compute-target
    /// snapshot) would inflate freshness for every carried partition. Append x=2 (S1), then touch a
    /// DIFFERENT partition x=1 (S2): x=2's last_updated MUST stay S1, x=1's MUST be S2.
    #[tokio::test]
    async fn test_carried_existing_entry_attributes_to_original_committer_not_target() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // S1: data in x=2.
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=2/a.parquet"),
                0,
                x_struct(2),
                4,
            )
            .await,
        ])
        .await;
        let s1 = table.metadata().current_snapshot().unwrap().snapshot_id();

        // S2: data in a DIFFERENT partition x=1 (x=2's files are carried forward as EXISTING).
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/b.parquet"),
                0,
                x_struct(1),
                9,
            )
            .await,
        ])
        .await;
        let s2 = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(s1, s2, "two distinct snapshots");

        let snapshot = table.metadata().current_snapshot().unwrap();
        let stats = compute_partition_stats(&table, snapshot).await.unwrap();

        let x2 = row_for_x_single(&stats, 2);
        assert_eq!(
            x2.data_record_count(),
            4,
            "x=2's carried data still counted"
        );
        assert_eq!(
            x2.last_updated_snapshot_id(),
            Some(s1),
            "x=2 (untouched in S2) keeps S1 attribution — carried EXISTING entries attribute to the ORIGINAL committer"
        );
        let x1 = row_for_x_single(&stats, 1);
        assert_eq!(
            x1.last_updated_snapshot_id(),
            Some(s2),
            "x=1 (added in S2) attributes to S2"
        );
    }

    /// RISK (spec_id row semantics, EXPLICIT): each output row's `spec_id` is the spec the file was
    /// written under = the MANIFEST's `partition_spec_id` (Java `collectStatsForManifest` builds the
    /// row via `new PartitionStats(coercedPartition, manifestFile.partitionSpecId())`, NOT the
    /// newest-seen spec). A partition written under spec 0 then spec 1 lands in two rows reporting
    /// spec 0 and spec 1 respectively. Makes explicit what the crown jewel pins implicitly.
    #[tokio::test]
    async fn test_spec_id_is_the_files_own_manifest_spec_across_evolution() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // Spec 0: data in (x=5).
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=5/d0.parquet"),
                0,
                x_struct(5),
                2,
            )
            .await,
        ])
        .await;

        // Evolve to spec 1 (x, y), then write (x=5, y=NULL... actually y must be set under spec 1).
        let table = evolve_to_xy_spec(&catalog, &table).await;
        let spec1_id = table.metadata().default_partition_spec_id();

        // Spec 1: data in (x=5, y=99).
        let xy = Struct::from_iter([Some(Literal::long(5)), Some(Literal::long(99))]);
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=5/y=99/d1.parquet"),
                spec1_id,
                xy,
                3,
            )
            .await,
        ])
        .await;

        let snapshot = table.metadata().current_snapshot().unwrap();
        let stats = compute_partition_stats(&table, snapshot).await.unwrap();

        // (5, NULL) is the spec-0 row; (5, 99) is the spec-1 row.
        let spec0_row = row_for_xy(&stats, 5, None);
        assert_eq!(spec0_row.spec_id(), 0, "spec-0 file → spec_id 0");
        let spec1_row = row_for_xy(&stats, 5, Some(99));
        assert_eq!(
            spec1_row.spec_id(),
            spec1_id,
            "spec-1 file → spec_id 1 (the file's OWN/manifest spec, not the newest-seen)"
        );
    }

    /// RISK (the load-bearing coercion pin, a SECOND independent disorder): the field-id remap must
    /// hold for more than the 2-field reversed case. A spec tuple `[z@1002, x@1000, y@1001]` (a THREE-
    /// field scramble whose positional order differs from the unified ascending order in a *different*
    /// way than the reversed-order pin) coerced into unified `{x@1000, y@1001, z@1002}` must produce
    /// `(x,y,z)`. Gives the coercion two unrelated pins so a "drop the spec-coercion" mutation
    /// (index-by-unified-position) fails on BOTH, not just one fragile fixture.
    #[test]
    fn test_coerce_partition_three_field_scramble_remaps_by_id() {
        let unified = StructType::new(vec![
            Arc::new(NestedField::optional(
                1000,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                1001,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                1002,
                "z",
                Type::Primitive(PrimitiveType::Long),
            )),
        ]);
        // spec tuple order: z (1002), x (1000), y (1001).
        let spec_type = StructType::new(vec![
            Arc::new(NestedField::optional(
                1002,
                "z",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                1000,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                1001,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
        ]);
        // file values in spec order: z=30, x=10, y=20.
        let file_partition = Struct::from_iter([
            Some(Literal::long(30)),
            Some(Literal::long(10)),
            Some(Literal::long(20)),
        ]);
        let coerced = coerce_partition(&unified, &spec_type, &file_partition);
        assert_eq!(
            coerced,
            Struct::from_iter([
                Some(Literal::long(10)),
                Some(Literal::long(20)),
                Some(Literal::long(30)),
            ]),
            "unified (x,y,z) must be (10,20,30) — remapped by id from the scrambled spec tuple"
        );
    }
}
