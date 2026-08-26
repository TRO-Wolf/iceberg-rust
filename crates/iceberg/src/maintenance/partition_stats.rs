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

//! `ComputePartitionStats`: the Rust port of Java 1.10.0 `PartitionStatsHandler`.
//!
//! It aggregates every manifest entry of a snapshot into the Java-exact partition-stats schema, then
//! writes and registers the stats file. `ComputeTableStats` stays out of scope, because the
//! workspace carries no sketch dependency.
//!
//! Partition stats feed planning downstream. A wrong aggregation misleads every consumer, and a
//! wrong unified-partition-tuple mapping corrupts the file's keying. So the traversal mirrors Java
//! exactly, from the 1.10.0 jar rather than the later MAIN source.
//!
//! # Schema field ids
//!
//! These become the on-disk parquet field ids, so they must match Java. Field 1 is
//! `required(1, "partition", <unified partition type>)`. Then:
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
//! Java `schema(StructType)` returns v2. `schema(StructType, formatVersion)` returns v2 for a
//! version of 2 or less, and v3 otherwise.
//!
//! A full compute reads every data and delete manifest the snapshot reaches, and iterates ALL
//! entries. A live entry rolls up the counters. A DELETED tombstone only bumps the last-updated
//! info, and still creates the row, so a fully-deleted partition keeps a zero-count row.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, FixedSizeBinaryArray, Int32Array, Int64Array, LargeBinaryArray,
    RecordBatch, StringArray, StructArray, Time64MicrosecondArray,
};
use arrow_schema::{Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use uuid::Uuid;

use crate::arrow::{
    arrow_struct_to_literal, create_primitive_array_single_element, schema_to_arrow_schema,
};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, Manifest, ManifestEntry,
    ManifestFile, ManifestStatus, NestedField, NestedFieldRef, PartitionSpec,
    PartitionStatisticsFile, PrimitiveLiteral, PrimitiveType, Schema, Snapshot, Struct, StructType,
    TableMetadata, Type, coerce_partition,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, Error, ErrorKind, Result};

// The stats-schema field ids and names, from the Java `PartitionStatsHandler` constants. They land
// on disk as the parquet field ids, so they must match Java exactly.
const PARTITION_FIELD_ID: i32 = 1;
const PARTITION_FIELD_NAME: &str = "partition";

const SPEC_ID_FIELD_ID: i32 = 2;
const DATA_RECORD_COUNT_FIELD_ID: i32 = 3;
const DATA_FILE_COUNT_FIELD_ID: i32 = 4;
const TOTAL_DATA_FILE_SIZE_IN_BYTES_FIELD_ID: i32 = 5;
const POSITION_DELETE_RECORD_COUNT_FIELD_ID: i32 = 6;
const POSITION_DELETE_FILE_COUNT_FIELD_ID: i32 = 7;
const EQUALITY_DELETE_RECORD_COUNT_FIELD_ID: i32 = 8;
const EQUALITY_DELETE_FILE_COUNT_FIELD_ID: i32 = 9;
const TOTAL_RECORD_COUNT_FIELD_ID: i32 = 10;
const LAST_UPDATED_AT_FIELD_ID: i32 = 11;
const LAST_UPDATED_SNAPSHOT_ID_FIELD_ID: i32 = 12;
/// `dv_count`, v3 and later only.
const DV_COUNT_FIELD_ID: i32 = 13;

/// One partition's rolled-up statistics. It mirrors Java 1.10.0 `PartitionStats` field for field.
///
/// Java boxes only `total_record_count` and the two last-updated members, so only those are
/// `Option` here. The widths follow the stats schema: a record count is `i64` and a file count is
/// `i32`. The incremental subtract can make one negative, so the signed widths matter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionStats {
    partition: Struct,
    /// The spec id of the files here. The map key `(specId, partition)` keeps it to one.
    spec_id: i32,
    data_record_count: i64,
    data_file_count: i32,
    total_data_file_size_in_bytes: i64,
    position_delete_record_count: i64,
    /// Java `positionDeleteFileCount`: the position deletes that are not deletion vectors.
    position_delete_file_count: i32,
    equality_delete_record_count: i64,
    equality_delete_file_count: i32,
    /// Java `totalRecordCount`. The compute path never sets it, because Java does not scan for it.
    total_record_count: Option<i64>,
    last_updated_at: Option<i64>,
    last_updated_snapshot_id: Option<i64>,
    dv_count: i32,
}

impl PartitionStats {
    /// Constructs an empty row: every counter is zero and every nullable member is `None`.
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

    /// Rolls a LIVE manifest entry into the counters, then updates the last-updated info. It ports
    /// Java `PartitionStats.liveEntry`.
    ///
    /// `file_format` separates a Puffin position delete, a deletion vector, from a parquet one. The
    /// snapshot arguments come from the entry's own snapshot, and are `None` once it is gone. Java's
    /// spec-id precondition holds by construction: the caller passes the row keyed by this file.
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
                // A Puffin position delete is a deletion vector. A parquet one is a file.
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

    /// Updates only the last-updated info for a DELETED entry, and touches no counter. The row
    /// survives, which is why a fully-deleted partition keeps a zero-count row.
    fn deleted_entry(&mut self, snapshot_timestamp_ms: Option<i64>, snapshot_id: Option<i64>) {
        if let (Some(timestamp_ms), Some(id)) = (snapshot_timestamp_ms, snapshot_id) {
            self.update_snapshot_info(id, timestamp_ms);
        }
    }

    /// Subtracts a DELETED entry's file from the counters, then updates the last-updated info (Java
    /// `deletedEntryForIncrementalCompute`).
    ///
    /// It mirrors [`PartitionStats::live_entry`] with every `+=` replaced by `-=`. Only the
    /// incremental branch calls it. A counter can go negative when the diff removes a file the base
    /// never added, and Java has the same signed arithmetic with no clamp.
    fn deleted_entry_for_incremental_compute(
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
                self.data_record_count -= record_count;
                self.data_file_count -= 1;
                self.total_data_file_size_in_bytes -= file_size_in_bytes;
            }
            DataContentType::PositionDeletes => {
                self.position_delete_record_count -= record_count;
                if file_format == DataFileFormat::Puffin {
                    self.dv_count -= 1;
                } else {
                    self.position_delete_file_count -= 1;
                }
            }
            DataContentType::EqualityDeletes => {
                self.equality_delete_record_count -= record_count;
                self.equality_delete_file_count -= 1;
            }
        }

        if let (Some(timestamp_ms), Some(id)) = (snapshot_timestamp_ms, snapshot_id) {
            self.update_snapshot_info(id, timestamp_ms);
        }
    }

    /// Sets the last-updated pair only if this timestamp is strictly newer (Java
    /// `updateSnapshotInfo`). A tie keeps the snapshot seen first.
    fn update_snapshot_info(&mut self, snapshot_id: i64, updated_at_ms: i64) {
        if self
            .last_updated_at
            .is_none_or(|current| current < updated_at_ms)
        {
            self.last_updated_at = Some(updated_at_ms);
            self.last_updated_snapshot_id = Some(snapshot_id);
        }
    }

    /// Merges `input` into `self`. It ports Java `PartitionStats.appendStats`.
    ///
    /// Every primitive counter adds, `dv_count` included: the 1.10.0 jar adds it unconditionally.
    /// `total_record_count` sets if null and adds otherwise. A set `input.last_updated_at` then
    /// re-evaluates the last-updated pair.
    ///
    /// # Errors
    ///
    /// Returns `DataInvalid` if the spec ids differ. The map key makes them equal by construction, so
    /// this is defense in depth.
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
            // The id is non-null whenever the timestamp is: `update_snapshot_info` sets both.
            if let Some(input_snapshot_id) = input.last_updated_snapshot_id {
                self.update_snapshot_info(input_snapshot_id, input_last_updated_at);
            }
        }

        Ok(())
    }
}

/// Builds the partition-stats [`Schema`]. It ports Java `PartitionStatsHandler.schema`.
///
/// A `format_version` of 2 or less gives the v2 schema of 12 fields. A higher one gives the v3
/// schema, which makes the delete fields required and adds `dv_count`.
///
/// # Errors
///
/// Returns `DataInvalid` if `unified_partition_type` is empty, because the table is unpartitioned.
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

    // v3 makes the delete fields required. The last-updated fields stay optional in both.
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
        // `dv_count` is a required Int whose initial and write defaults are 0.
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

/// Deprecated alias for [`TableMetadata::unified_partition_type`], which lives in
/// `spec/partitioning.rs`. It keeps downstream code and the interop harness compiling.
///
/// # Errors
///
/// Propagates [`TableMetadata::unified_partition_type`].
pub fn unified_partition_type(metadata: &TableMetadata) -> Result<StructType> {
    metadata.unified_partition_type()
}

/// Computes per-partition statistics for a snapshot. It ports the full-compute branch of Java
/// `computeAndWriteStatsFile`. It folds a per-manifest map for every manifest in
/// `snapshot.allManifests` into one map, sorted by partition tuple.
///
/// # Errors
///
/// Returns `DataInvalid` if no spec has a non-void partition field. An unpartitioned table is an
/// error, never an empty result. Propagates manifest read errors and a merge mismatch.
///
/// # Notes
///
/// A snapshot with no manifests gives an empty `Vec`. A partition that only delete files reach
/// still gets a row. `total_record_count` stays unset.
pub async fn compute_partition_stats(
    table: &Table,
    snapshot: &Snapshot,
) -> Result<Vec<PartitionStats>> {
    let metadata = table.metadata();

    // Java `isPartitioned` means any spec has at least one non-void field.
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
    let file_io = table.file_io();

    // A full compute passes every manifest of the snapshot, with `incremental` false.
    let manifest_list = snapshot.load_manifest_list(file_io, metadata).await?;
    let manifest_files: Vec<_> = manifest_list.entries().to_vec();
    let stats_by_key =
        compute_stats_over_manifests(table, &unified_type, &manifest_files, false).await?;

    let mut stats: Vec<PartitionStats> = stats_by_key.into_values().collect();
    stats.sort_by(|left, right| compare_partition_values(&left.partition, &right.partition));
    Ok(stats)
}

/// Aggregates manifest files into a `(spec_id, coerced-partition) -> PartitionStats` map (Java
/// `computeStats`). Each manifest builds its own map, which folds into the running total. The result
/// is unsorted, so the caller sorts it or merges it into a seeded map.
async fn compute_stats_over_manifests(
    table: &Table,
    unified_type: &StructType,
    manifest_files: &[ManifestFile],
    incremental: bool,
) -> Result<HashMap<(i32, Struct), PartitionStats>> {
    let metadata = table.metadata();
    let schema = metadata.current_schema();
    let file_io = table.file_io();

    let mut stats_by_key: HashMap<(i32, Struct), PartitionStats> = HashMap::new();
    for manifest_file in manifest_files {
        let manifest = manifest_file.load_manifest(file_io).await?;
        // Every file in one manifest shares its spec id, so resolve the type once per manifest.
        let spec_id = manifest_file.partition_spec_id;
        let spec = resolve_spec(metadata, spec_id)?;

        let per_manifest = collect_stats_for_manifest(
            metadata,
            &manifest,
            spec,
            schema,
            unified_type,
            incremental,
        )?;
        merge_partition_map(per_manifest, &mut stats_by_key)?;
    }
    Ok(stats_by_key)
}

/// Finds the most recent partition-stats file in `snapshot_id`'s lineage (Java `latestStatsFile`).
/// It walks back through `parentId`, starting at `snapshot_id` itself, and returns the first
/// ancestor that carries a stats file. `None` triggers a full compute.
fn latest_stats_file(
    metadata: &TableMetadata,
    snapshot_id: i64,
) -> Option<&PartitionStatisticsFile> {
    let stats_by_snapshot: HashMap<i64, &PartitionStatisticsFile> = metadata
        .partition_statistics_iter()
        .map(|file| (file.snapshot_id, file))
        .collect();
    if stats_by_snapshot.is_empty() {
        return None;
    }

    // The snapshot history bounds this walk.
    let mut current = metadata.snapshot_by_id(snapshot_id);
    while let Some(snapshot) = current {
        if let Some(file) = stats_by_snapshot.get(&snapshot.snapshot_id()) {
            return Some(file);
        }
        current = match snapshot.parent_snapshot_id() {
            Some(parent_id) => metadata.snapshot_by_id(parent_id),
            None => None,
        };
    }
    None
}

/// Aggregates the incremental diff over the lineage range `(from_snapshot, to_snapshot]`. It ports
/// Java `computeStatsDiff`. For each snapshot in the range it takes only the manifests that snapshot
/// added, which is `added_snapshot_id == snapshot_id`.
///
/// # Errors
///
/// Propagates manifest read errors.
async fn compute_stats_diff(
    table: &Table,
    unified_type: &StructType,
    from_snapshot: &Snapshot,
    to_snapshot: &Snapshot,
) -> Result<HashMap<(i32, Struct), PartitionStats>> {
    let metadata = table.metadata();
    let file_io = table.file_io();
    let from_snapshot_id = from_snapshot.snapshot_id();

    // Walk back from `to` through the parent ids, and stop BEFORE `from`.
    let mut range_snapshots: Vec<&Snapshot> = Vec::new();
    let mut current = Some(to_snapshot);
    while let Some(snapshot) = current {
        if snapshot.snapshot_id() == from_snapshot_id {
            break;
        }
        range_snapshots.push(snapshot);
        current = match snapshot.parent_snapshot_id() {
            Some(parent_id) => metadata.snapshot_by_id(parent_id).map(|s| s.as_ref()),
            None => None,
        };
    }

    // Take only the manifests each snapshot added itself.
    let mut diff_manifests: Vec<ManifestFile> = Vec::new();
    for snapshot in range_snapshots {
        let manifest_list = snapshot.load_manifest_list(file_io, metadata).await?;
        for manifest_file in manifest_list.entries() {
            if manifest_file.added_snapshot_id == snapshot.snapshot_id() {
                diff_manifests.push(manifest_file.clone());
            }
        }
    }

    compute_stats_over_manifests(table, unified_type, &diff_manifests, true).await
}

/// Seeds from a base stats file, then merges the diff into it (Java
/// `computeAndMergeStatsIncremental`).
///
/// It reads `base_stats_file` into the seed map, then merges the `(base.snapshot, target]` diff. The
/// counts add and the last-updated pair takes the maximum. `Ok(None)` means the base read failed, so
/// the caller must compute in full. A diff read failure after the seed is a hard `Err`, as in Java.
async fn compute_and_merge_stats_incremental(
    table: &Table,
    unified_type: &StructType,
    target_snapshot: &Snapshot,
    base_stats_file: &PartitionStatisticsFile,
) -> Result<Option<Vec<PartitionStats>>> {
    let metadata = table.metadata();
    let format_version = metadata.format_version();
    let base_schema = partition_stats_schema(unified_type, format_version)?;

    // Any failure here is Java's `InvalidStatsFileException`, so signal the full-compute fallback.
    let seed_rows = match read_partition_stats_file(
        table,
        &base_schema,
        &base_stats_file.statistics_path,
    )
    .await
    {
        Ok(rows) => rows,
        Err(_corrupt_base) => return Ok(None),
    };

    let mut stats_by_key: HashMap<(i32, Struct), PartitionStats> = seed_rows
        .into_iter()
        .map(|row| ((row.spec_id, row.partition.clone()), row))
        .collect();

    // The base file's snapshot starts the range, exclusive. The target ends it, inclusive.
    let base_snapshot = metadata
        .snapshot_by_id(base_stats_file.snapshot_id)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Base partition-stats file references snapshot {} which is absent from metadata",
                    base_stats_file.snapshot_id
                ),
            )
        })?;

    let diff =
        compute_stats_diff(table, unified_type, base_snapshot.as_ref(), target_snapshot).await?;
    merge_partition_map(diff, &mut stats_by_key)?;

    Ok(Some(stats_by_key.into_values().collect()))
}

/// Collects one manifest's per-partition statistics into a fresh map (Java
/// `collectStatsForManifest`). It coerces each file's partition into the unified type, and keys the
/// row by `(spec_id, coerced-partition)`. The per-entry dispatch follows the `incremental` flag:
///
/// | entry | full compute | incremental |
/// |---|---|---|
/// | live | roll up the counters | roll up only an `Added` entry, skip a carried-forward one |
/// | DELETED | update the last-updated info | subtract the file from the base stats |
///
/// An `Existing` entry is already in the base stats, so counting it again would double it. Every
/// branch keeps the partition row, so a fully-deleted partition retains one.
fn collect_stats_for_manifest(
    metadata: &TableMetadata,
    manifest: &Manifest,
    spec: &PartitionSpec,
    schema: &Schema,
    unified_type: &StructType,
    incremental: bool,
) -> Result<HashMap<(i32, Struct), PartitionStats>> {
    let spec_id = spec.spec_id();
    let mut stats_map: HashMap<(i32, Struct), PartitionStats> = HashMap::new();

    for entry in manifest.entries() {
        let data_file = entry.data_file();
        let coerced = coerce_partition(unified_type, spec, schema, data_file.partition())?;

        // The entry's own snapshot carries the last-updated pair, and may be gone from metadata.
        let (snapshot_timestamp_ms, snapshot_id) = entry_snapshot_info(metadata, entry);

        let row = stats_map
            .entry((spec_id, coerced.clone()))
            .or_insert_with(|| PartitionStats::new(coerced, spec_id));

        if entry.is_alive() {
            // Only a newly added live file contributes to the diff. The row is already kept.
            if !incremental || entry.status() == ManifestStatus::Added {
                accumulate_live_entry(row, data_file, snapshot_timestamp_ms, snapshot_id);
            }
        } else if incremental {
            accumulate_deleted_entry_incremental(
                row,
                data_file,
                snapshot_timestamp_ms,
                snapshot_id,
            );
        } else {
            row.deleted_entry(snapshot_timestamp_ms, snapshot_id);
        }
    }

    Ok(stats_map)
}

/// Folds one manifest's stats map into the running total (Java `mergePartitionMap`). A shared key
/// merges, and a new key inserts as-is.
///
/// # Errors
///
/// Propagates the spec-id mismatch from `append_stats`, which a shared key makes unreachable.
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

/// Resolves a manifest's partition spec. An unknown spec id means a corrupt manifest list.
fn resolve_spec(metadata: &TableMetadata, spec_id: i32) -> Result<&PartitionSpec> {
    let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
        Error::new(
            crate::ErrorKind::DataInvalid,
            format!("Cannot find partition spec for manifest: {spec_id}"),
        )
    })?;
    Ok(spec.as_ref())
}

/// The `(timestamp_ms, snapshot_id)` of an entry's own snapshot, or `(None, None)` if it is gone.
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

/// Applies a live entry's file to the row. The narrowing saturates, so a hostile on-disk count
/// cannot wrap or panic.
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

/// Subtracts a DELETED entry's file from the row during an incremental diff. It mirrors
/// [`accumulate_live_entry`], with the same width narrowing.
fn accumulate_deleted_entry_incremental(
    row: &mut PartitionStats,
    data_file: &DataFile,
    snapshot_timestamp_ms: Option<i64>,
    snapshot_id: Option<i64>,
) {
    let record_count = i64::try_from(data_file.record_count()).unwrap_or(i64::MAX);
    let file_size = i64::try_from(data_file.file_size_in_bytes()).unwrap_or(i64::MAX);
    row.deleted_entry_for_incremental_compute(
        data_file.content_type(),
        data_file.file_format(),
        record_count,
        file_size,
        snapshot_timestamp_ms,
        snapshot_id,
    );
}

/// Compares two unified-partition tuples field by field, for the output sort. It is the local
/// analogue of Java `Comparators.forType(partitionType)`, and mirrors the comparator in the
/// read-only `inspect::partitions`. A null sorts before any value. An incomparable pair falls back
/// to `Equal`, which keeps the order total under a stable sort.
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
        // A non-primitive literal is not a valid partition value. Keep the order stable.
        _ => Ordering::Equal,
    }
}

/// Compares two [`PrimitiveLiteral`]s, falling back to `Equal` for an incomparable pair.
fn compare_primitive(left: &PrimitiveLiteral, right: &PrimitiveLiteral) -> std::cmp::Ordering {
    left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
}

// ---- The on-disk stats file: write it, register it, read it back. --------------------------------
// The format is the table's `write.format.default`. The location is
// `<table.location()>/metadata/partition-stats-<snapshotId>-<uuid>.<ext>`. The field ids 1 to 13 are
// stamped on the parquet columns, which is the on-disk contract.

/// The file-name prefix. Java formats `partition-stats-<snapshot id>-<random uuid>`.
const PARTITION_STATS_FILE_NAME_PREFIX: &str = "partition-stats";
/// The table property selecting the stats-file format (Java `TableProperties.DEFAULT_FILE_FORMAT`).
const WRITE_FORMAT_DEFAULT_PROPERTY: &str = "write.format.default";
/// The table property that overrides the metadata directory.
const WRITE_METADATA_PATH_PROPERTY: &str = "write.metadata.path";

/// Computes the statistics for `snapshot` and writes them to one on-disk stats file (Java
/// `computeAndWriteStatsFile`), the incremental-versus-full selection included. It first locates the
/// most recent stats file in the snapshot's lineage, then branches:
///
/// | base file | action |
/// |---|---|
/// | none | full compute over `snapshot.allManifests` |
/// | for `snapshot` itself | return it unchanged, with no recompute and no rewrite |
/// | for an older snapshot | seed from it and merge the diff, falling back to full if it is corrupt |
///
/// A computed result then sorts by partition tuple and writes one file. An empty result returns
/// `Ok(None)` and writes nothing. The file is not yet registered: pass it to
/// [`register_partition_stats_file`].
///
/// # Errors
///
/// Returns `DataInvalid` for an unpartitioned table, and `FeatureUnsupported` if
/// `write.format.default` is not parquet. Only a base-file read failure falls back to a full
/// compute. A diff read failure is hard, as in Java.
pub async fn compute_and_write_stats_file(
    table: &Table,
    snapshot: &Snapshot,
) -> Result<Option<PartitionStatisticsFile>> {
    let metadata = table.metadata();
    let snapshot_id = snapshot.snapshot_id();

    // The compute paths raise the unpartitioned error themselves.
    let unified_type = unified_partition_type(metadata)?;

    // Branch on the base stats file in this snapshot's lineage.
    let stats = match latest_stats_file(metadata, snapshot_id) {
        // A stats file for THIS snapshot is already up to date.
        Some(base) if base.snapshot_id == snapshot_id => {
            return Ok(Some(base.clone()));
        }
        // An older base file allows an incremental compute.
        Some(base) => {
            match compute_and_merge_stats_incremental(table, &unified_type, snapshot, base).await? {
                Some(mut rows) => {
                    rows.sort_by(|left, right| {
                        compare_partition_values(&left.partition, &right.partition)
                    });
                    rows
                }
                // An unreadable base file falls back to a full compute.
                None => compute_partition_stats(table, snapshot).await?,
            }
        }
        // No base file in the lineage, so compute in full.
        None => compute_partition_stats(table, snapshot).await?,
    };

    // Write no file for an empty result. An empty file with a degenerate schema would mislead a
    // later incremental compute.
    if stats.is_empty() {
        return Ok(None);
    }

    let format_version = metadata.format_version();
    let stats_schema = partition_stats_schema(&unified_type, format_version)?;

    let file_format = stats_file_format(metadata)?;
    let path = new_partition_stats_file_path(metadata, snapshot_id, file_format);

    let batch = partition_stats_to_record_batch(&stats, &stats_schema, &unified_type)?;
    let file_size_in_bytes =
        write_partition_stats_parquet(table, &path, &stats_schema, batch).await?;

    Ok(Some(PartitionStatisticsFile {
        snapshot_id,
        statistics_path: path,
        file_size_in_bytes,
    }))
}

/// Registers a [`PartitionStatisticsFile`] in the table metadata, and returns the refreshed
/// [`Table`]. It ports Java `updatePartitionStatistics().setPartitionStatistics(file).commit()`.
///
/// The commit runs through the
/// [`UpdatePartitionStatisticsAction`](crate::transaction::Transaction::update_partition_statistics)
/// seam, which emits a `SetPartitionStatistics` update and one `AssertTableUUID` requirement. That
/// update replaces any prior entry for the same snapshot id. The [`Transaction`] supplies the
/// commit-retry loop. [`ComputePartitionStats`](crate::maintenance::ComputePartitionStats) uses the
/// same seam, so no commit logic is duplicated.
///
/// # Errors
///
/// Propagates the catalog commit error and any metadata-build error.
pub async fn register_partition_stats_file(
    catalog: &dyn Catalog,
    table: &Table,
    partition_statistics_file: PartitionStatisticsFile,
) -> Result<Table> {
    let transaction = Transaction::new(table);
    let transaction = transaction
        .update_partition_statistics()
        .set_partition_statistics(partition_statistics_file)
        .apply(transaction)?;
    transaction.commit(catalog).await
}

/// Reads a partition-stats file back into [`PartitionStats`] rows (Java `readPartitionStatsFile`).
/// It decodes each row's columns positionally against `stats_schema`. [`arrow_struct_to_literal`]
/// decodes the partition struct by field id, so a v2 and a v3 file each decode against their own.
///
/// # Errors
///
/// Returns `DataInvalid` for an unexpected Arrow type or a row shape the schema does not match,
/// which means a corrupt or foreign file. Propagates IO and decode errors.
pub async fn read_partition_stats_file(
    table: &Table,
    stats_schema: &Schema,
    path: &str,
) -> Result<Vec<PartitionStats>> {
    let bytes = table.file_io().new_input(path)?.read().await?;
    read_partition_stats_from_bytes(stats_schema, bytes)
}

/// Decodes the rows from already-read parquet `bytes`. It is split out of
/// [`read_partition_stats_file`], so a test can decode without a `Table` or `FileIO`.
fn read_partition_stats_from_bytes(
    stats_schema: &Schema,
    bytes: Bytes,
) -> Result<Vec<PartitionStats>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                "Failed to open partition-stats parquet file",
            )
            .with_source(error)
        })?
        .build()
        .map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                "Failed to build partition-stats parquet reader",
            )
            .with_source(error)
        })?;

    let struct_type = stats_schema.as_struct();
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                "Failed to read a partition-stats record batch",
            )
            .with_source(error)
        })?;
        decode_record_batch(&batch, struct_type, &mut rows)?;
    }
    Ok(rows)
}

/// Decodes one [`RecordBatch`] of rows into `rows`, through a `StructArray` over the schema's
/// struct type.
///
/// `struct_type` projects down to the field ids the batch holds first, which ports Java's
/// `project(schema)`. Without it [`arrow_struct_to_literal`] errors on the field the file lacks. The
/// projection keeps schema order and drops only trailing fields, so the positional decode holds.
fn decode_record_batch(
    batch: &RecordBatch,
    struct_type: &StructType,
    rows: &mut Vec<PartitionStats>,
) -> Result<()> {
    let projected_type = project_struct_type_to_batch(struct_type, batch);
    let struct_array: ArrayRef = Arc::new(StructArray::from(batch.clone()));
    let literals = arrow_struct_to_literal(&struct_array, &projected_type)?;
    for literal in literals {
        let Some(Literal::Struct(record)) = literal else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Partition-stats row decoded to a null / non-struct record",
            ));
        };
        rows.push(partition_stats_from_record(&record)?);
    }
    Ok(())
}

/// Projects `struct_type` down to the field ids in `batch`, and keeps schema order. It is the
/// equivalent of Java `project(schema)` reading an older file with a newer schema. Only the trailing
/// `dv_count` can go missing, and the shorter-record tolerance handles that.
fn project_struct_type_to_batch(struct_type: &StructType, batch: &RecordBatch) -> StructType {
    use std::collections::HashSet;

    let present_field_ids: HashSet<i32> = batch
        .schema()
        .fields()
        .iter()
        .filter_map(|field| {
            field
                .metadata()
                .get(PARQUET_FIELD_ID_META_KEY)
                .and_then(|value| value.parse::<i32>().ok())
        })
        .collect();

    let projected_fields: Vec<NestedFieldRef> = struct_type
        .fields()
        .iter()
        .filter(|field| present_field_ids.contains(&field.id))
        .cloned()
        .collect();

    StructType::new(projected_fields)
}

/// Reconstructs a [`PartitionStats`] from one decoded record's positional fields (Java
/// `recordToPartitionStats`). The positions follow the stats-schema order in the module docs, offset
/// by one: position 0 is the partition struct and position 12 is the v3-only `dv_count`.
fn partition_stats_from_record(record: &Struct) -> Result<PartitionStats> {
    let fields = record.fields();
    if fields.len() < 12 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Partition-stats record has {} fields, expected at least 12",
                fields.len()
            ),
        ));
    }

    let partition = match &fields[0] {
        Some(Literal::Struct(partition)) => partition.clone(),
        _ => {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Partition-stats record field 0 (partition) is not a struct",
            ));
        }
    };

    let spec_id = require_i32(&fields[1], "spec_id")?;
    let mut stats = PartitionStats::new(partition, spec_id);
    stats.data_record_count = require_i64(&fields[2], "data_record_count")?;
    stats.data_file_count = require_i32(&fields[3], "data_file_count")?;
    stats.total_data_file_size_in_bytes = require_i64(&fields[4], "total_data_file_size_in_bytes")?;
    stats.position_delete_record_count =
        optional_i64(&fields[5], "position_delete_record_count")?.unwrap_or(0);
    stats.position_delete_file_count =
        optional_i32(&fields[6], "position_delete_file_count")?.unwrap_or(0);
    stats.equality_delete_record_count =
        optional_i64(&fields[7], "equality_delete_record_count")?.unwrap_or(0);
    stats.equality_delete_file_count =
        optional_i32(&fields[8], "equality_delete_file_count")?.unwrap_or(0);
    stats.total_record_count = optional_i64(&fields[9], "total_record_count")?;
    stats.last_updated_at = optional_i64(&fields[10], "last_updated_at")?;
    stats.last_updated_snapshot_id = optional_i64(&fields[11], "last_updated_snapshot_id")?;
    if fields.len() >= 13 {
        stats.dv_count = optional_i32(&fields[12], "dv_count")?.unwrap_or(0);
    }

    Ok(stats)
}

/// Reads a required `int` field. A null or a non-`int` literal is an error.
fn require_i32(value: &Option<Literal>, name: &str) -> Result<i32> {
    optional_i32(value, name)?.ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Partition-stats field `{name}` is required but null"),
        )
    })
}

/// Reads a required `long` field. A null or a non-`long` literal is an error.
fn require_i64(value: &Option<Literal>, name: &str) -> Result<i64> {
    optional_i64(value, name)?.ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Partition-stats field `{name}` is required but null"),
        )
    })
}

/// Reads an optional `int` field. A non-`int` literal is an error.
fn optional_i32(value: &Option<Literal>, name: &str) -> Result<Option<i32>> {
    match value {
        None => Ok(None),
        Some(Literal::Primitive(PrimitiveLiteral::Int(value))) => Ok(Some(*value)),
        Some(_) => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Partition-stats field `{name}` is not an int"),
        )),
    }
}

/// Reads an optional `long` field. A non-`long` literal is an error.
fn optional_i64(value: &Option<Literal>, name: &str) -> Result<Option<i64>> {
    match value {
        None => Ok(None),
        Some(Literal::Primitive(PrimitiveLiteral::Long(value))) => Ok(Some(*value)),
        Some(_) => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Partition-stats field `{name}` is not a long"),
        )),
    }
}

/// Resolves the stats-file format from `write.format.default`, which defaults to parquet. This fork
/// writes parquet only, so another default fails loudly.
fn stats_file_format(metadata: &TableMetadata) -> Result<DataFileFormat> {
    let format = metadata
        .properties()
        .get(WRITE_FORMAT_DEFAULT_PROPERTY)
        .map(String::as_str)
        .unwrap_or("parquet");
    match format.to_ascii_lowercase().as_str() {
        "parquet" => Ok(DataFileFormat::Parquet),
        other => Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!("Partition-stats file format `{other}` is not supported (only parquet)"),
        )),
    }
}

/// Builds the stats-file path. It ports Java `newPartitionStatsFile` and `metadataFileLocation`.
///
/// The name is `partition-stats-<snapshotId>-<uuid>.<ext>`. The directory is `write.metadata.path`
/// with any trailing slash stripped, or `<location()>/metadata` when that property is unset.
fn new_partition_stats_file_path(
    metadata: &TableMetadata,
    snapshot_id: i64,
    file_format: DataFileFormat,
) -> String {
    let uuid = Uuid::new_v4();
    let extension = match file_format {
        DataFileFormat::Parquet => "parquet",
        DataFileFormat::Avro => "avro",
        DataFileFormat::Orc => "orc",
        DataFileFormat::Puffin => "puffin",
    };
    let name = format!("{PARTITION_STATS_FILE_NAME_PREFIX}-{snapshot_id}-{uuid}.{extension}");

    match metadata.properties().get(WRITE_METADATA_PATH_PROPERTY) {
        Some(write_metadata_path) => {
            let base = write_metadata_path.trim_end_matches('/');
            format!("{base}/{name}")
        }
        None => format!("{}/metadata/{name}", metadata.location()),
    }
}

/// Writes `batch` to `path` as parquet with the schema's field ids stamped, and returns the on-disk
/// size in bytes.
///
/// Java streams through a `FileAppender`. This buffers in memory and writes once, for the same
/// output, and it avoids the crate-private async file-writer wrapper in `writer/`.
async fn write_partition_stats_parquet(
    table: &Table,
    path: &str,
    stats_schema: &Schema,
    batch: RecordBatch,
) -> Result<i64> {
    let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(stats_schema)?);

    let mut buffer: Vec<u8> = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, arrow_schema, None).map_err(|error| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to create the partition-stats parquet writer",
        )
        .with_source(error)
    })?;
    writer.write(&batch).map_err(|error| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to encode the partition-stats record batch",
        )
        .with_source(error)
    })?;
    writer.close().map_err(|error| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to finalize the partition-stats parquet file",
        )
        .with_source(error)
    })?;

    let file_size_in_bytes = i64::try_from(buffer.len()).unwrap_or(i64::MAX);
    table
        .file_io()
        .new_output(path)?
        .write(Bytes::from(buffer))
        .await?;

    Ok(file_size_in_bytes)
}

/// Builds the Arrow [`RecordBatch`] for the stats file: one row per [`PartitionStats`], with the
/// columns in stats-schema order. [`schema_to_arrow_schema`] stamps the iceberg field id on every
/// column, nested fields included. That stamping is the on-disk contract, so each column must match
/// the derived schema exactly.
fn partition_stats_to_record_batch(
    stats: &[PartitionStats],
    stats_schema: &Schema,
    unified_partition_type: &StructType,
) -> Result<RecordBatch> {
    let arrow_schema = schema_to_arrow_schema(stats_schema)?;

    let partition_column =
        build_partition_struct_array(stats, unified_partition_type, &arrow_schema)?;

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    columns.push(partition_column);
    columns.push(Arc::new(Int32Array::from_iter_values(
        stats.iter().map(|row| row.spec_id),
    )));
    columns.push(Arc::new(Int64Array::from_iter_values(
        stats.iter().map(|row| row.data_record_count),
    )));
    columns.push(Arc::new(Int32Array::from_iter_values(
        stats.iter().map(|row| row.data_file_count),
    )));
    columns.push(Arc::new(Int64Array::from_iter_values(
        stats.iter().map(|row| row.total_data_file_size_in_bytes),
    )));
    columns.push(Arc::new(Int64Array::from_iter_values(
        stats.iter().map(|row| row.position_delete_record_count),
    )));
    columns.push(Arc::new(Int32Array::from_iter_values(
        stats.iter().map(|row| row.position_delete_file_count),
    )));
    columns.push(Arc::new(Int64Array::from_iter_values(
        stats.iter().map(|row| row.equality_delete_record_count),
    )));
    columns.push(Arc::new(Int32Array::from_iter_values(
        stats.iter().map(|row| row.equality_delete_file_count),
    )));
    columns.push(Arc::new(Int64Array::from(
        stats
            .iter()
            .map(|row| row.total_record_count)
            .collect::<Vec<_>>(),
    )));
    columns.push(Arc::new(Int64Array::from(
        stats
            .iter()
            .map(|row| row.last_updated_at)
            .collect::<Vec<_>>(),
    )));
    columns.push(Arc::new(Int64Array::from(
        stats
            .iter()
            .map(|row| row.last_updated_snapshot_id)
            .collect::<Vec<_>>(),
    )));
    // `dv_count` is a required Int column, in v3 only.
    if arrow_schema.fields().len() >= 13 {
        columns.push(Arc::new(Int32Array::from_iter_values(
            stats.iter().map(|row| row.dv_count),
        )));
    }

    RecordBatch::try_new(Arc::new(arrow_schema), columns).map_err(|error| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to assemble the partition-stats record batch",
        )
        .with_source(error)
    })
}

/// Builds the partition-struct column from each row's coerced partition tuple. The Arrow fields
/// come from the stats schema's first field, so the column matches the on-disk schema exactly.
fn build_partition_struct_array(
    stats: &[PartitionStats],
    unified_partition_type: &StructType,
    arrow_schema: &ArrowSchema,
) -> Result<ArrayRef> {
    let partition_arrow_field = arrow_schema.field(0);
    let arrow_struct_fields: Fields = match partition_arrow_field.data_type() {
        arrow_schema::DataType::Struct(fields) => fields.clone(),
        other => {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("Expected the partition column to be a struct, got {other:?}"),
            ));
        }
    };

    let unified_fields = unified_partition_type.fields();
    let mut child_arrays: Vec<(arrow_schema::FieldRef, ArrayRef)> =
        Vec::with_capacity(unified_fields.len());

    for (field_index, unified_field) in unified_fields.iter().enumerate() {
        let arrow_field = arrow_struct_fields[field_index].clone();
        let column = build_partition_field_column(
            stats,
            field_index,
            &unified_field.field_type,
            arrow_field.data_type(),
        )?;
        child_arrays.push((arrow_field, column));
    }

    Ok(Arc::new(StructArray::from(child_arrays)))
}

/// Builds one partition-field child array from the per-row partition tuples.
///
/// `arrow_data_type` comes from [`schema_to_arrow_schema`]. It drives both the fast paths and the
/// logical-type path, so the built array matches the on-disk type, timezone included. A partition
/// value is never nested, so a non-primitive type fails loudly. A missing position is a null entry.
fn build_partition_field_column(
    stats: &[PartitionStats],
    field_index: usize,
    field_type: &Type,
    arrow_data_type: &arrow_schema::DataType,
) -> Result<ArrayRef> {
    let Type::Primitive(primitive_type) = field_type else {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!("Partition field type {field_type:?} is not a supported partition value type"),
        ));
    };

    fn value_at(row: &PartitionStats, field_index: usize) -> Option<&Literal> {
        row.partition
            .fields()
            .get(field_index)
            .and_then(|value| value.as_ref())
    }

    macro_rules! collect_primitive {
        ($variant:path) => {
            stats
                .iter()
                .map(|row| match value_at(row, field_index) {
                    None => Ok(None),
                    Some(Literal::Primitive($variant(value))) => Ok(Some(value.clone())),
                    Some(other) => Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Partition value {other:?} does not match field type {primitive_type:?}"
                        ),
                    )),
                })
                .collect::<Result<Vec<_>>>()?
        };
    }

    // Boolean, Int, Long, String, Time, Uuid, Fixed and Binary build with the plain Arrow
    // constructors, into the types `schema_to_arrow_schema` emits for them: a Time64 of micros since
    // midnight, a FixedSizeBinary(16) of the uuid's big-endian bytes, and a FixedSizeBinary(len) or
    // LargeBinary of the raw bytes.
    //
    // A logical type carries a timezone, or a precision and scale, so it builds through
    // `create_primitive_array_single_element` driven by the field's exact `arrow_data_type`.
    let array: ArrayRef = match primitive_type {
        PrimitiveType::Boolean => {
            let values = collect_primitive!(PrimitiveLiteral::Boolean);
            Arc::new(BooleanArray::from(values))
        }
        PrimitiveType::Int => {
            let values = collect_primitive!(PrimitiveLiteral::Int);
            Arc::new(Int32Array::from(values))
        }
        PrimitiveType::Long => {
            let values = collect_primitive!(PrimitiveLiteral::Long);
            Arc::new(Int64Array::from(values))
        }
        PrimitiveType::String => {
            let values = collect_primitive!(PrimitiveLiteral::String);
            Arc::new(StringArray::from(values))
        }
        PrimitiveType::Time => {
            // An iceberg time is micros since midnight, held as a `Long`. The read-back path
            // decodes the Time64 array to `Literal::time(micros)`, so the round trip is exact.
            let values = collect_primitive!(PrimitiveLiteral::Long);
            Arc::new(Time64MicrosecondArray::from(values))
        }
        PrimitiveType::Uuid => build_uuid_partition_field_column(stats, field_index)?,
        PrimitiveType::Fixed(length) => {
            build_fixed_partition_field_column(stats, field_index, *length, primitive_type)?
        }
        PrimitiveType::Binary => {
            let values: Vec<Option<Vec<u8>>> = stats
                .iter()
                .map(|row| match value_at(row, field_index) {
                    None => Ok(None),
                    Some(Literal::Primitive(PrimitiveLiteral::Binary(value))) => {
                        Ok(Some(value.clone()))
                    }
                    Some(other) => Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Partition value {other:?} does not match field type Binary"),
                    )),
                })
                .collect::<Result<Vec<_>>>()?;
            Arc::new(LargeBinaryArray::from_iter(values))
        }
        PrimitiveType::Date
        | PrimitiveType::Timestamp
        | PrimitiveType::Timestamptz
        | PrimitiveType::TimestampNs
        | PrimitiveType::TimestamptzNs
        | PrimitiveType::Decimal { .. }
        | PrimitiveType::Float
        | PrimitiveType::Double => build_logical_partition_field_column(
            stats,
            field_index,
            primitive_type,
            arrow_data_type,
        )?,
        // `Transform::result_type` rejects `unknown` as a partition source, so it cannot be a
        // partition-field type. Fail loudly here rather than fabricate a column.
        PrimitiveType::Unknown => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Partition field type unknown is not a supported partition value type: unknown is always null and cannot be a partition source",
            ));
        }
    };

    Ok(array)
}

/// Builds a partition-field child array for a logical Arrow type, such as Date32 or Decimal128.
/// Each row becomes a single-element array through [`create_primitive_array_single_element`], which
/// honors the timezone, precision and scale. One array per row then concatenated is acceptable,
/// because a stats file holds one row per partition.
fn build_logical_partition_field_column(
    stats: &[PartitionStats],
    field_index: usize,
    primitive_type: &PrimitiveType,
    arrow_data_type: &arrow_schema::DataType,
) -> Result<ArrayRef> {
    let per_row_arrays: Vec<ArrayRef> = stats
        .iter()
        .map(|row| {
            let literal = row
                .partition
                .fields()
                .get(field_index)
                .and_then(|value| value.as_ref());
            let primitive = match literal {
                None => None,
                Some(Literal::Primitive(primitive)) => Some(primitive.clone()),
                Some(other) => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Partition value {other:?} does not match field type {primitive_type:?}"
                        ),
                    ));
                }
            };
            create_primitive_array_single_element(arrow_data_type, &primitive)
        })
        .collect::<Result<Vec<_>>>()?;

    let array_refs: Vec<&dyn arrow_array::Array> =
        per_row_arrays.iter().map(|array| array.as_ref()).collect();
    arrow_select::concat::concat(&array_refs).map_err(|error| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to concatenate partition-field values for type {primitive_type:?}"),
        )
        .with_source(error)
    })
}

/// Builds a `uuid` child array as a `FixedSizeBinary(16)` of 16 big-endian bytes per row. That is the
/// on-disk form Java emits, and [`arrow_struct_to_literal`] decodes it back to `Literal::uuid`.
fn build_uuid_partition_field_column(
    stats: &[PartitionStats],
    field_index: usize,
) -> Result<ArrayRef> {
    let rows: Vec<Option<[u8; 16]>> = stats
        .iter()
        .map(|row| {
            match row
                .partition
                .fields()
                .get(field_index)
                .and_then(|value| value.as_ref())
            {
                None => Ok(None),
                Some(Literal::Primitive(PrimitiveLiteral::UInt128(value))) => {
                    Ok(Some(Uuid::from_u128(*value).into_bytes()))
                }
                Some(other) => Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("Partition value {other:?} does not match field type Uuid"),
                )),
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let array = FixedSizeBinaryArray::try_from_sparse_iter_with_size(rows.into_iter(), 16)
        .map_err(|error| {
            Error::new(
                ErrorKind::Unexpected,
                "Failed to assemble the uuid partition-field FixedSizeBinary(16) array",
            )
            .with_source(error)
        })?;
    Ok(Arc::new(array))
}

/// Builds a `fixed[L]` child array as a `FixedSizeBinary(L)` per row, which is the on-disk form Java
/// emits. Every non-null value must be exactly `length` bytes, and the constructor rejects any other
/// width loudly.
fn build_fixed_partition_field_column(
    stats: &[PartitionStats],
    field_index: usize,
    length: u64,
    primitive_type: &PrimitiveType,
) -> Result<ArrayRef> {
    let width = i32::try_from(length).map_err(|_| {
        Error::new(
            ErrorKind::FeatureUnsupported,
            format!("Fixed partition value width {length} exceeds the Arrow FixedSizeBinary limit"),
        )
    })?;

    let rows: Vec<Option<Vec<u8>>> = stats
        .iter()
        .map(|row| {
            match row
                .partition
                .fields()
                .get(field_index)
                .and_then(|value| value.as_ref())
            {
                None => Ok(None),
                Some(Literal::Primitive(PrimitiveLiteral::Binary(value))) => {
                    Ok(Some(value.clone()))
                }
                Some(other) => Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Partition value {other:?} does not match field type {primitive_type:?}"
                    ),
                )),
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let array = FixedSizeBinaryArray::try_from_sparse_iter_with_size(rows.into_iter(), width)
        .map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to assemble the fixed[{length}] partition-field array (a value's byte length must equal {length})"),
            )
            .with_source(error)
        })?;
    Ok(Arc::new(array))
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

    /// A fixed data-file size, so a total-size assertion is an exact multiple.
    const DATA_FILE_SIZE: u64 = 100;
    const DELETE_FILE_SIZE: u64 = 7;

    // ---- Pure-fn tests. The ids they pin land on disk, so they must match Java. ----------------

    fn x_partition_type() -> StructType {
        StructType::new(vec![Arc::new(NestedField::optional(
            1000,
            "x",
            Type::Primitive(PrimitiveType::Long),
        ))])
    }

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

    fn xyz_schema() -> Schema {
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
            .expect("x/y/z schema")
    }

    fn identity_unbound(source_id: i32, field_id: i32, name: &str) -> UnboundPartitionField {
        UnboundPartitionField {
            source_id,
            field_id: Some(field_id),
            name: name.to_string(),
            transform: Transform::Identity,
        }
    }

    fn bind_spec(
        schema: &Schema,
        spec_id: i32,
        fields: Vec<UnboundPartitionField>,
    ) -> PartitionSpec {
        PartitionSpec::builder(schema.clone())
            .with_spec_id(spec_id)
            .add_unbound_fields(fields)
            .expect("add unbound fields")
            .build()
            .expect("bind spec")
    }

    /// Pins every field id, name, type and nullability of the v2 schema. A drift from the Java
    /// constants makes a file Java cannot read.
    #[test]
    fn test_v2_schema_has_java_exact_field_ids_names_types_and_nullability() {
        let schema = partition_stats_schema(&x_partition_type(), FormatVersion::V2).unwrap();
        let fields = schema.as_struct().fields();
        assert_eq!(fields.len(), 12, "v2 schema has 12 fields");

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

        if let Type::Struct(partition_struct) = fields[0].field_type.as_ref() {
            assert_eq!(partition_struct, &x_partition_type());
        } else {
            panic!("field 1 must be a struct");
        }
    }

    /// The v3 schema must make the delete fields required and add `dv_count`.
    #[test]
    fn test_v3_schema_makes_delete_fields_required_and_adds_dv_count() {
        let schema = partition_stats_schema(&x_partition_type(), FormatVersion::V3).unwrap();
        let fields = schema.as_struct().fields();
        assert_eq!(fields.len(), 13, "v3 schema has 13 fields");

        for id in [6, 7, 8, 9] {
            let field = schema.field_by_id(id).unwrap();
            assert!(field.required, "v3 field id {id} must be required");
        }
        for id in [10, 11, 12] {
            assert!(
                !schema.field_by_id(id).unwrap().required,
                "v3 field id {id} optional"
            );
        }
        let dv = schema.field_by_id(13).unwrap();
        assert_eq!(dv.name, "dv_count");
        assert!(dv.required);
        assert_eq!(*dv.field_type, Type::Primitive(PrimitiveType::Int));
    }

    /// The schema defaults to v2, and rejects an empty partition type as degenerate.
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

    fn x_struct(x: i64) -> Struct {
        Struct::from_iter([Some(Literal::long(x))])
    }

    /// Pins each `live_entry` arm alone on one row. A swap of data for delete, position for equality,
    /// or parquet-position for DV reports the wrong category silently.
    #[test]
    fn test_live_entry_routes_each_content_type_to_its_own_counters() {
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

        // A Puffin position delete bumps `dv_count`, never the file count.
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

    /// `deleted_entry_for_incremental_compute` must subtract exactly the cells `live_entry` adds. A
    /// row seeded with a file, then subtracted, must return to zero. This runs every content type and
    /// the DV split, so a mutation that subtracts the wrong cell leaves a residue.
    #[test]
    fn test_deleted_entry_for_incremental_compute_subtracts_each_cell_back_to_zero() {
        let mut data_row = PartitionStats::new(x_struct(1), 0);
        data_row.live_entry(
            DataContentType::Data,
            DataFileFormat::Parquet,
            5,
            100,
            None,
            None,
        );
        data_row.deleted_entry_for_incremental_compute(
            DataContentType::Data,
            DataFileFormat::Parquet,
            5,
            100,
            None,
            None,
        );
        assert_eq!(data_row.data_record_count(), 0, "data records subtracted");
        assert_eq!(data_row.data_file_count(), 0, "data file count subtracted");
        assert_eq!(
            data_row.total_data_file_size_in_bytes(),
            0,
            "data size subtracted"
        );

        // A parquet subtract lowers the file count, never `dv_count`.
        let mut pos_row = PartitionStats::new(x_struct(1), 0);
        pos_row.live_entry(
            DataContentType::PositionDeletes,
            DataFileFormat::Parquet,
            3,
            7,
            None,
            None,
        );
        pos_row.deleted_entry_for_incremental_compute(
            DataContentType::PositionDeletes,
            DataFileFormat::Parquet,
            3,
            7,
            None,
            None,
        );
        assert_eq!(pos_row.position_delete_record_count(), 0);
        assert_eq!(pos_row.position_delete_file_count(), 0);
        assert_eq!(
            pos_row.dv_count(),
            0,
            "a parquet pos delete never touches dv_count"
        );

        // A Puffin subtract lowers `dv_count`, never the file count.
        let mut dv_row = PartitionStats::new(x_struct(1), 0);
        dv_row.live_entry(
            DataContentType::PositionDeletes,
            DataFileFormat::Puffin,
            4,
            9,
            None,
            None,
        );
        dv_row.deleted_entry_for_incremental_compute(
            DataContentType::PositionDeletes,
            DataFileFormat::Puffin,
            4,
            9,
            None,
            None,
        );
        assert_eq!(dv_row.position_delete_record_count(), 0);
        assert_eq!(dv_row.dv_count(), 0, "the DV is subtracted from dv_count");
        assert_eq!(dv_row.position_delete_file_count(), 0);

        let mut eq_row = PartitionStats::new(x_struct(1), 0);
        eq_row.live_entry(
            DataContentType::EqualityDeletes,
            DataFileFormat::Parquet,
            6,
            11,
            None,
            None,
        );
        eq_row.deleted_entry_for_incremental_compute(
            DataContentType::EqualityDeletes,
            DataFileFormat::Parquet,
            6,
            11,
            None,
            None,
        );
        assert_eq!(eq_row.equality_delete_record_count(), 0);
        assert_eq!(eq_row.equality_delete_file_count(), 0);
    }

    /// The highest timestamp must win, under a strict `<`. A tie must keep the snapshot seen first,
    /// which a `<=` regression breaks. An older timestamp seen after a newer one must not overwrite.
    #[test]
    fn test_update_snapshot_info_keeps_max_timestamp_strict_and_first_on_tie() {
        let mut row = PartitionStats::new(x_struct(1), 0);

        row.update_snapshot_info(/* snapshot_id */ 10, /* updated_at_ms */ 1000);
        assert_eq!(row.last_updated_at(), Some(1000));
        assert_eq!(row.last_updated_snapshot_id(), Some(10));

        row.update_snapshot_info(20, 2000);
        assert_eq!(row.last_updated_at(), Some(2000));
        assert_eq!(row.last_updated_snapshot_id(), Some(20));

        row.update_snapshot_info(30, 1500);
        assert_eq!(row.last_updated_at(), Some(2000));
        assert_eq!(
            row.last_updated_snapshot_id(),
            Some(20),
            "older must not win"
        );

        // A tie keeps the snapshot seen first, under the strict `<`.
        row.update_snapshot_info(40, 2000);
        assert_eq!(
            row.last_updated_snapshot_id(),
            Some(20),
            "a tie must keep the first-seen snapshot, not snapshot 40"
        );
    }

    /// A DELETED tombstone must touch no counter, only the last-updated pair, and must leave the row
    /// present at zero. A decrement here, which is the incremental behavior, corrupts a full compute.
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

    /// Pins the merge that folds the per-manifest maps. Every primitive counter must add, `dv_count`
    /// included. `total_record_count` must set if null and add otherwise. The last-updated pair must
    /// re-evaluate against the input.
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
        // A full compute leaves `total_record_count` unset, so the merge test sets it here.
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
        assert_eq!(target.last_updated_at(), Some(3000));
        assert_eq!(target.last_updated_snapshot_id(), Some(30));
    }

    /// `append_stats` must transfer `total_record_count` onto a `None` target, and never panic.
    #[test]
    fn test_append_stats_sets_total_record_count_when_target_is_null() {
        let mut target = PartitionStats::new(x_struct(1), 0);
        let mut input = PartitionStats::new(x_struct(1), 0);
        input.total_record_count = Some(42);
        target.append_stats(&input).unwrap();
        assert_eq!(target.total_record_count(), Some(42));
    }

    /// The merge guard must fire on two spec ids, because merging them conflates two partitions.
    #[test]
    fn test_append_stats_rejects_mismatched_spec_ids() {
        let mut target = PartitionStats::new(x_struct(1), 0);
        let input = PartitionStats::new(x_struct(1), 1);
        let error = target.append_stats(&input).unwrap_err();
        assert!(error.message().contains("Spec IDs must match"));
    }

    /// Two specs that share field id 1000 must unify into one struct of {1000:x, 1001:y}, sorted by
    /// id, with no duplicated `x`. A wrong unifier corrupts the file's keying.
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
        assert!(!fields[0].required && !fields[1].required);
    }

    /// The alias, `TableMetadata::unified_partition_type` and `spec::partition_type` must agree.
    #[test]
    fn test_unified_partition_type_delegates_to_spec_partitioning() {
        let metadata = two_spec_metadata();
        let via_alias = unified_partition_type(&metadata).expect("alias");
        let via_method = metadata
            .unified_partition_type()
            .expect("TableMetadata method");
        let specs: Vec<_> = metadata.partition_specs_iter().cloned().collect();
        let via_fn = crate::spec::partition_type(metadata.current_schema(), &specs)
            .expect("spec::partition_type");
        assert_eq!(via_alias, via_method);
        assert_eq!(via_alias, via_fn);
    }

    /// A spec-1 file `(x, y)` must project into the unified `{x, y}` unchanged.
    #[test]
    fn test_coerce_partition_carries_full_tuple_for_the_newer_spec() {
        let schema = xyz_schema();
        let spec = bind_spec(&schema, 1, vec![
            identity_unbound(1, 1000, "x"),
            identity_unbound(2, 1001, "y"),
        ]);
        let unified = xy_partition_type();
        let file_partition = Struct::from_iter([Some(Literal::long(7)), Some(Literal::long(9))]);
        let coerced =
            coerce_partition(&unified, &spec, &schema, &file_partition).expect("coerce full tuple");
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(7)), Some(Literal::long(9))])
        );
    }

    /// A spec-0 file `(x)` must project into the unified `{x, y}` as `(x, NULL)`.
    #[test]
    fn test_coerce_partition_null_fills_field_absent_from_the_older_spec() {
        let schema = xyz_schema();
        let spec = bind_spec(&schema, 0, vec![identity_unbound(1, 1000, "x")]);
        let unified = xy_partition_type();
        let file_partition = Struct::from_iter([Some(Literal::long(7))]);
        let coerced =
            coerce_partition(&unified, &spec, &schema, &file_partition).expect("coerce null-fill");
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(7)), None]),
            "y must be null-filled for a spec-0 file"
        );
    }

    /// The load-bearing coercion pin. The spec tuple is `[y=9 @ 1001, x=7 @ 1000]` and the unified
    /// type is ascending, so the coerced tuple must be `(x=7, y=9)`. Coercion must remap by field id,
    /// not by position. A mutation that indexes by unified position reads the wrong value here. The
    /// same-order fixtures above cannot catch it.
    #[test]
    fn test_coerce_partition_remaps_by_field_id_not_position() {
        let schema = xyz_schema();
        let spec = bind_spec(&schema, 0, vec![
            identity_unbound(2, 1001, "y"),
            identity_unbound(1, 1000, "x"),
        ]);
        let unified = xy_partition_type();
        let file_partition = Struct::from_iter([Some(Literal::long(9)), Some(Literal::long(7))]);
        let coerced =
            coerce_partition(&unified, &spec, &schema, &file_partition).expect("coerce remap");
        assert_eq!(
            coerced,
            Struct::from_iter([Some(Literal::long(7)), Some(Literal::long(9))]),
            "coercion must remap by field id: unified (x,y) = (7,9), not the file's (9,7)"
        );
    }

    /// A `TableMetadata` with spec 0 `identity(x)` and spec 1 `identity(x), identity(y)`.
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

    // ---- End-to-end tests over a real `MemoryCatalog` table's committed manifests. -------------
    // The catalog assigns wall-clock timestamps, so a last-updated assertion derives its expected id
    // from the committed snapshots. It hand-derives the semantics, never a clock value.

    /// A `MemoryCatalog` on a local filesystem, so the manifests it writes are real files on disk.
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

    /// Creates a table partitioned by `identity(x)` under a fresh namespace.
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

    /// Evolves the spec to `identity(x), identity(y)` through a real transaction.
    async fn evolve_to_xy_spec(catalog: &impl Catalog, table: &Table) -> Table {
        let tx = Transaction::new(table);
        let tx = tx.update_partition_spec().add_field("y").apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    async fn write_file(file_io: &FileIO, path: &str, content: &[u8]) {
        file_io
            .new_output(path)
            .unwrap()
            .write(bytes::Bytes::copy_from_slice(content))
            .await
            .unwrap();
    }

    /// A real data file stamped with `spec_id` and `partition`, holding `records` rows.
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

    async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    async fn add_deletes(
        catalog: &impl Catalog,
        table: &Table,
        delete_files: Vec<DataFile>,
    ) -> Table {
        let tx = Transaction::new(table);
        let tx = tx.row_delta().add_deletes(delete_files).apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// The coerced unified tuple `(x, y)`, which every row keys by once the type has two fields.
    fn xy_key(x: i64, y: Option<i64>) -> Struct {
        Struct::from_iter([Some(Literal::long(x)), y.map(Literal::long)])
    }

    /// The computed row for a coerced `(x, y)` tuple. A `None` y is the null-filled spec-0 row.
    fn row_for_xy(stats: &[PartitionStats], x: i64, y: Option<i64>) -> &PartitionStats {
        let key = xy_key(x, y);
        stats
            .iter()
            .find(|row| row.partition() == &key)
            .unwrap_or_else(|| panic!("no row for (x={x},y={y:?}) in {stats:#?}"))
    }

    /// The headline aggregation test. A 2-spec table evolved mid-history, with data, position
    /// deletes and equality deletes across 3 partitions, matched against a hand-derived table.
    ///
    /// Spec 0 is `identity(x)` and spec 1 is `identity(x), identity(y)`, so the unified type is
    /// {1000:x, 1001:y}. S1 appends `d_x1` (x=1, 3 records) and `d_x2` (x=2, 5 records). The spec
    /// evolves. S2 appends `d_x1y10` (x=1, y=10, 7 records). S3 adds a position delete of 2 records
    /// and an equality delete of 4, both in (x=1, y=10). A data file is 100 bytes, a delete file 7.
    ///
    /// Spec 0's `(x=1)` coerces to `(x=1, y=NULL)` and spec 1's to `(x=1, y=10)`, so they are two
    /// distinct rows, never merged. The hand-derived expected rows:
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

        let table = evolve_to_xy_spec(&catalog, &table).await;
        let spec1_id = table.metadata().default_partition_spec_id();
        assert_ne!(spec1_id, 0, "fixture sanity: spec evolved away from 0");

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

        assert_eq!(stats.len(), 3, "(1,NULL), (2,NULL), (1,10): {stats:#?}");

        // The output sorts (1,NULL) < (1,10) < (2,NULL), because a null y sorts first.
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

    /// One partition takes only position deletes, one only equality deletes, and one both.
    #[tokio::test]
    async fn test_per_content_type_counters_isolated_across_partitions() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
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

        let r1 = row_for_x_single(&stats, 1);
        assert_eq!(r1.position_delete_record_count(), 2);
        assert_eq!(r1.position_delete_file_count(), 1);
        assert_eq!(r1.equality_delete_record_count(), 0, "x=1 has no eq delete");
        assert_eq!(r1.equality_delete_file_count(), 0);

        let r2 = row_for_x_single(&stats, 2);
        assert_eq!(r2.equality_delete_record_count(), 3);
        assert_eq!(r2.equality_delete_file_count(), 1);
        assert_eq!(
            r2.position_delete_record_count(),
            0,
            "x=2 has no pos delete"
        );
        assert_eq!(r2.position_delete_file_count(), 0);

        let r3 = row_for_x_single(&stats, 3);
        assert_eq!(r3.position_delete_record_count(), 4);
        assert_eq!(r3.position_delete_file_count(), 1);
        assert_eq!(r3.equality_delete_record_count(), 5);
        assert_eq!(r3.equality_delete_file_count(), 1);
    }

    /// Three appends to x=1 must leave its last-updated on the third snapshot, never the first.
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
        let s3_ts = table.metadata().snapshot_by_id(s3).unwrap().timestamp_ms();
        assert_eq!(row.last_updated_at(), Some(s3_ts));
    }

    /// A partition that only a delete file reaches must still produce a row, with zero data.
    #[tokio::test]
    async fn test_partition_present_only_via_delete_files_has_zero_data_counters() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // x=1 gets a data file, so a snapshot exists to delta from. x=2 gets only the delete.
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

    /// A fully-deleted partition must keep a zero-count row, because the tombstone creates it. A
    /// traversal that skips DELETED entries drops the partition.
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

        let tx = Transaction::new(&table);
        let tx = tx.rewrite_files(vec![d1], vec![]).apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let delete_snapshot = table.metadata().current_snapshot().unwrap().snapshot_id();

        let snapshot = table.metadata().current_snapshot().unwrap();
        let stats = compute_partition_stats(&table, snapshot).await.unwrap();

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

    /// An unpartitioned table is an error. A no-op regression gives rows keyed by an empty struct.
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

    /// A snapshot with no manifests must yield an empty `Vec`, never a spurious row.
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

    /// The computed row for a single-field `(x)` tuple, on a table whose spec never evolved.
    fn row_for_x_single(stats: &[PartitionStats], x: i64) -> &PartitionStats {
        stats
            .iter()
            .find(|row| row.partition() == &x_struct(x))
            .unwrap_or_else(|| panic!("no row for x={x} in {stats:#?}"))
    }

    // ---- DV routing, carried-EXISTING attribution, spec_id across evolution, coercion. ---------

    /// A real Puffin deletion-vector file, with a referenced data file, offset and size.
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

    /// A Puffin deletion vector must route into `dv_count` and `position_delete_record_count`, never
    /// into `position_delete_file_count`. A real V3 table with a real DV pins the cells.
    #[tokio::test]
    async fn test_puffin_dv_routes_to_dv_count_not_position_delete_file_count() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
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

    /// A carried-forward manifest re-lists its files as EXISTING entries whose `snapshot_id` is the
    /// original committer. Last-updated must key off that id. Keying off the compute-target snapshot
    /// instead inflates freshness for every carried partition. S1 appends x=2 and S2 touches x=1, so
    /// x=2 must stay on S1.
    #[tokio::test]
    async fn test_carried_existing_entry_attributes_to_original_committer_not_target() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

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

        // S2 touches x=1, so x=2's files carry forward as EXISTING.
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

    /// A row's `spec_id` is the manifest's `partition_spec_id`, never the newest spec seen. A
    /// partition written under spec 0, then spec 1, lands in two rows.
    #[tokio::test]
    async fn test_spec_id_is_the_files_own_manifest_spec_across_evolution() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

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

        let table = evolve_to_xy_spec(&catalog, &table).await;
        let spec1_id = table.metadata().default_partition_spec_id();

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

        let spec0_row = row_for_xy(&stats, 5, None);
        assert_eq!(spec0_row.spec_id(), 0, "spec-0 file → spec_id 0");
        let spec1_row = row_for_xy(&stats, 5, Some(99));
        assert_eq!(
            spec1_row.spec_id(),
            spec1_id,
            "spec-1 file → spec_id 1 (the file's OWN/manifest spec, not the newest-seen)"
        );
    }

    /// A second, independent disorder for the field-id remap. The spec tuple `[z@1002, x@1000,
    /// y@1001]` scrambles differently from the reversed 2-field pin, and must still coerce to
    /// `(x,y,z)`. Two unrelated pins mean an index-by-unified-position mutation fails on both.
    #[test]
    fn test_coerce_partition_three_field_scramble_remaps_by_id() {
        let schema = xyz_schema();
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
        let spec = bind_spec(&schema, 0, vec![
            identity_unbound(3, 1002, "z"),
            identity_unbound(1, 1000, "x"),
            identity_unbound(2, 1001, "y"),
        ]);
        let file_partition = Struct::from_iter([
            Some(Literal::long(30)),
            Some(Literal::long(10)),
            Some(Literal::long(20)),
        ]);
        let coerced =
            coerce_partition(&unified, &spec, &schema, &file_partition).expect("coerce scramble");
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

    // ---- The on-disk stats file: write, register, read back. -----------------------------------
    // They pin the on-disk contract: the parquet schema carries the iceberg field ids, the file
    // round-trips field for field, and registration keeps one entry per snapshot id.

    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    /// Reads the TOP-LEVEL iceberg field ids stamped in a parquet file's arrow schema, in order.
    async fn raw_top_level_field_ids(file_io: &FileIO, path: &str) -> Vec<i32> {
        let bytes = file_io.new_input(path).unwrap().read().await.unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
        builder
            .schema()
            .fields()
            .iter()
            .filter_map(|field| {
                field
                    .metadata()
                    .get(PARQUET_FIELD_ID_META_KEY)
                    .and_then(|value| value.parse::<i32>().ok())
            })
            .collect()
    }

    /// The nested field ids of the partition struct column, which are the on-disk keying contract.
    async fn raw_partition_struct_field_ids(file_io: &FileIO, path: &str) -> Vec<i32> {
        let bytes = file_io.new_input(path).unwrap().read().await.unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
        let partition_field = builder.schema().field(0).clone();
        match partition_field.data_type() {
            arrow_schema::DataType::Struct(fields) => fields
                .iter()
                .filter_map(|field| {
                    field
                        .metadata()
                        .get(PARQUET_FIELD_ID_META_KEY)
                        .and_then(|value| value.parse::<i32>().ok())
                })
                .collect(),
            other => panic!("partition column must be a struct, got {other:?}"),
        }
    }

    async fn raw_row_count(file_io: &FileIO, path: &str) -> i64 {
        let bytes = file_io.new_input(path).unwrap().read().await.unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
        builder.metadata().file_metadata().num_rows()
    }

    /// Sorts rows by spec id, then partition tuple, so two row sets compare free of map order.
    fn sort_rows(mut rows: Vec<PartitionStats>) -> Vec<PartitionStats> {
        rows.sort_by(|left, right| {
            compare_partition_values(&left.partition, &right.partition)
                .then(left.spec_id.cmp(&right.spec_id))
        });
        rows
    }

    /// The headline file test: write, reopen raw, register, re-parse and read back, over the 2-spec
    /// fixture. A wrong file shape breaks every other engine that reads the file. A wrong registration
    /// loses the entry. A wrong read-back corrupts a later incremental compute.
    #[tokio::test]
    async fn test_crown_jewel_write_register_and_read_back_round_trip() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d1.parquet"),
                0,
                x_struct(1),
                3,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/x=2/d2.parquet"),
                0,
                x_struct(2),
                5,
            )
            .await,
        ])
        .await;
        let table = evolve_to_xy_spec(&catalog, &table).await;
        let spec1_id = table.metadata().default_partition_spec_id();
        let xy = Struct::from_iter([Some(Literal::long(1)), Some(Literal::long(10))]);
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/y=10/d3.parquet"),
                spec1_id,
                xy.clone(),
                7,
            )
            .await,
        ])
        .await;
        let table = add_deletes(&catalog, &table, vec![
            position_delete_file(
                &file_io,
                &format!("{location}/data/x=1/y=10/pos.parquet"),
                &format!("{location}/data/x=1/y=10/d3.parquet"),
                spec1_id,
                xy.clone(),
                2,
            )
            .await,
            equality_delete_file(
                &file_io,
                &format!("{location}/data/x=1/y=10/eq.parquet"),
                spec1_id,
                xy,
                4,
            )
            .await,
        ])
        .await;

        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let computed = compute_partition_stats(&table, &snapshot).await.unwrap();

        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .expect("a partitioned table with data writes a stats file");
        assert_eq!(stats_file.snapshot_id, snapshot.snapshot_id());
        assert!(stats_file.file_size_in_bytes > 0, "real on-disk size");
        let expected_prefix = format!(
            "{location}/metadata/partition-stats-{}-",
            snapshot.snapshot_id()
        );
        assert!(
            stats_file.statistics_path.starts_with(&expected_prefix),
            "path `{}` must start with `{expected_prefix}`",
            stats_file.statistics_path
        );
        assert!(stats_file.statistics_path.ends_with(".parquet"));

        // A raw reopen must show the field ids 1 to 12 in order, for a v2 table.
        let top_ids = raw_top_level_field_ids(&file_io, &stats_file.statistics_path).await;
        assert_eq!(
            top_ids,
            (1..=12).collect::<Vec<i32>>(),
            "v2 stats file must carry field ids 1..=12 in order"
        );
        let partition_ids =
            raw_partition_struct_field_ids(&file_io, &stats_file.statistics_path).await;
        assert_eq!(
            partition_ids,
            vec![1000, 1001],
            "unified partition field ids"
        );
        assert_eq!(
            raw_row_count(&file_io, &stats_file.statistics_path).await,
            3
        );

        let stats_schema = partition_stats_schema(
            &unified_partition_type(table.metadata()).unwrap(),
            table.metadata().format_version(),
        )
        .unwrap();
        let read_back =
            read_partition_stats_file(&table, &stats_schema, &stats_file.statistics_path)
                .await
                .unwrap();
        assert_eq!(
            sort_rows(read_back),
            sort_rows(computed),
            "read-back rows must equal the computed rows field-for-field (incl. NULL total_record_count)"
        );

        let registered = register_partition_stats_file(&catalog, &table, stats_file.clone())
            .await
            .unwrap();
        let entries: Vec<_> = registered.metadata().partition_statistics_iter().collect();
        assert_eq!(entries.len(), 1, "exactly one partition-statistics entry");
        assert_eq!(entries[0], &stats_file, "the registered entry matches");
        assert_eq!(
            registered
                .metadata()
                .partition_statistics_for_snapshot(snapshot.snapshot_id())
                .unwrap(),
            &stats_file
        );
    }

    /// A V3 file must carry the required `dv_count` at field id 13, and a real DV's count must
    /// survive it. A V2 file must not hold field 13, or a reader resolves the wrong column.
    #[tokio::test]
    async fn test_v3_stats_file_has_dv_count_and_round_trips_the_dv() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
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

        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let computed = compute_partition_stats(&table, &snapshot).await.unwrap();
        assert_eq!(
            computed[0].dv_count(),
            1,
            "fixture sanity: a DV was computed"
        );

        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();

        let top_ids = raw_top_level_field_ids(&file_io, &stats_file.statistics_path).await;
        assert_eq!(
            top_ids,
            (1..=13).collect::<Vec<i32>>(),
            "v3 stats file must carry field ids 1..=13 (dv_count present)"
        );

        let stats_schema = partition_stats_schema(
            &unified_partition_type(table.metadata()).unwrap(),
            FormatVersion::V3,
        )
        .unwrap();
        let read_back =
            read_partition_stats_file(&table, &stats_schema, &stats_file.statistics_path)
                .await
                .unwrap();
        assert_eq!(sort_rows(read_back), sort_rows(computed));
    }

    /// A V2 file must not carry field id 13. A 13th column breaks a Java v2 reader.
    #[tokio::test]
    async fn test_v2_stats_file_lacks_dv_count_field() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await; // V2 by default.
        let location = table.metadata().location().to_string();
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
        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();
        let top_ids = raw_top_level_field_ids(&file_io, &stats_file.statistics_path).await;
        assert_eq!(top_ids, (1..=12).collect::<Vec<i32>>());
        assert!(!top_ids.contains(&13), "v2 must NOT have dv_count(13)");
    }

    /// A second stats file for the same snapshot must replace the entry, never accumulate two,
    /// because the metadata keys them by snapshot id. A recompute for a snapshot that already has one
    /// must return it unchanged. Two snapshots' entries must coexist, which a wrong key clobbers.
    #[tokio::test]
    async fn test_replace_on_rewrite_same_snapshot_and_multi_snapshot_coexistence() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d1.parquet"),
                0,
                x_struct(1),
                3,
            )
            .await,
        ])
        .await;
        let s1 = table.metadata().current_snapshot().unwrap().clone();

        let file_s1_first = compute_and_write_stats_file(&table, &s1)
            .await
            .unwrap()
            .unwrap();
        let table = register_partition_stats_file(&catalog, &table, file_s1_first.clone())
            .await
            .unwrap();

        let file_s1_again = compute_and_write_stats_file(&table, &s1)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            file_s1_again, file_s1_first,
            "re-computing for an already-stats'd snapshot returns the existing file unchanged (Java case 3)"
        );

        // Re-registering for the same snapshot id replaces the entry. The path stays readable.
        let file_s1_replacement = PartitionStatisticsFile {
            file_size_in_bytes: file_s1_first.file_size_in_bytes + 7,
            ..file_s1_first.clone()
        };
        let table = register_partition_stats_file(&catalog, &table, file_s1_replacement.clone())
            .await
            .unwrap();
        let after_replace: Vec<_> = table.metadata().partition_statistics_iter().collect();
        assert_eq!(
            after_replace.len(),
            1,
            "same-snapshot rewrite REPLACES, not accumulates"
        );
        assert_eq!(
            table
                .metadata()
                .partition_statistics_for_snapshot(s1.snapshot_id())
                .unwrap(),
            &file_s1_replacement,
            "the second registration wins"
        );

        let table = register_partition_stats_file(&catalog, &table, file_s1_first.clone())
            .await
            .unwrap();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=2/d2.parquet"),
                0,
                x_struct(2),
                5,
            )
            .await,
        ])
        .await;
        let s2 = table.metadata().current_snapshot().unwrap().clone();
        assert_ne!(s1.snapshot_id(), s2.snapshot_id());
        let file_s2 = compute_and_write_stats_file(&table, &s2)
            .await
            .unwrap()
            .unwrap();
        let table = register_partition_stats_file(&catalog, &table, file_s2.clone())
            .await
            .unwrap();

        let both: Vec<_> = table.metadata().partition_statistics_iter().collect();
        assert_eq!(both.len(), 2, "S1 and S2 stats both registered");
        assert_eq!(
            table
                .metadata()
                .partition_statistics_for_snapshot(s1.snapshot_id())
                .unwrap(),
            &file_s1_first,
            "S1's (restored real) entry is still present after S2 registers"
        );
        assert_eq!(
            table
                .metadata()
                .partition_statistics_for_snapshot(s2.snapshot_id())
                .unwrap(),
            &file_s2
        );
    }

    /// Java writes the rows pre-sorted by partition tuple, so the on-disk order must be sorted too.
    /// This decodes in file order, with no re-sort.
    #[tokio::test]
    async fn test_on_disk_row_order_is_sorted_by_partition_tuple() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=3/d.parquet"),
                0,
                x_struct(3),
                1,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                1,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/x=2/d.parquet"),
                0,
                x_struct(2),
                1,
            )
            .await,
        ])
        .await;
        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();

        let stats_schema = partition_stats_schema(
            &unified_partition_type(table.metadata()).unwrap(),
            table.metadata().format_version(),
        )
        .unwrap();
        let read_back =
            read_partition_stats_file(&table, &stats_schema, &stats_file.statistics_path)
                .await
                .unwrap();
        let xs: Vec<Option<i64>> = read_back
            .iter()
            .map(|row| match row.partition().fields().first() {
                Some(Some(Literal::Primitive(PrimitiveLiteral::Long(value)))) => Some(*value),
                _ => None,
            })
            .collect();
        assert_eq!(
            xs,
            vec![Some(1), Some(2), Some(3)],
            "on-disk rows must be sorted by partition tuple, not append order"
        );
    }

    /// An empty result must return `Ok(None)` and write no file, as Java returns null. An empty file
    /// with a degenerate schema would mislead a later incremental compute. The fixture overwrites the
    /// manifest list with an empty one.
    #[tokio::test]
    async fn test_empty_stats_writes_no_file() {
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

        let current = table.metadata().current_snapshot().unwrap().clone();
        let mut writer = ManifestListWriter::v2(
            file_io.new_output(current.manifest_list()).unwrap(),
            current.snapshot_id(),
            current.parent_snapshot_id(),
            current.sequence_number(),
        );
        writer.add_manifests(std::iter::empty()).unwrap();
        writer.close().await.unwrap();

        let result = compute_and_write_stats_file(&table, &current)
            .await
            .unwrap();
        assert!(
            result.is_none(),
            "an empty-stats snapshot writes no file (Java returns null): {result:#?}"
        );
    }

    /// The field-id stamping is the on-disk contract: without it a reader maps the wrong column.
    /// This asserts every top-level and nested id raw, so dropping the stamping fails here.
    #[tokio::test]
    async fn test_written_file_stamps_every_iceberg_field_id() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=7/d.parquet"),
                0,
                x_struct(7),
                2,
            )
            .await,
        ])
        .await;
        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();

        let top_ids = raw_top_level_field_ids(&file_io, &stats_file.statistics_path).await;
        assert_eq!(
            top_ids,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "the on-disk parquet must stamp iceberg field ids 1..=12 on every column"
        );
        let nested = raw_partition_struct_field_ids(&file_io, &stats_file.statistics_path).await;
        assert_eq!(nested, vec![1000], "the single unified partition field id");
    }

    /// The decoder must be faithful: a row written with a distinct value in every counter reads back
    /// with those exact values. That is what makes a write-side counter swap surface as a mismatch in
    /// the headline test above.
    #[tokio::test]
    async fn test_record_batch_round_trips_every_counter_distinctly() {
        let unified = x_partition_type();
        let stats_schema = partition_stats_schema(&unified, FormatVersion::V3).unwrap();

        let mut row = PartitionStats::new(x_struct(42), 3);
        row.data_record_count = 11;
        row.data_file_count = 12;
        row.total_data_file_size_in_bytes = 13;
        row.position_delete_record_count = 14;
        row.position_delete_file_count = 15;
        row.equality_delete_record_count = 16;
        row.equality_delete_file_count = 17;
        row.total_record_count = Some(18);
        row.last_updated_at = Some(19);
        row.last_updated_snapshot_id = Some(20);
        row.dv_count = 21;

        let batch =
            partition_stats_to_record_batch(std::slice::from_ref(&row), &stats_schema, &unified)
                .unwrap();

        let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&stats_schema).unwrap());
        let mut buffer: Vec<u8> = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buffer, arrow_schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let decoded = read_partition_stats_from_bytes(&stats_schema, Bytes::from(buffer)).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(
            decoded[0], row,
            "every counter must round-trip to its exact written value (a write-side swap fails here)"
        );
    }

    /// The registered entry must key off the file's snapshot id, and a lookup by it must find it.
    #[tokio::test]
    async fn test_register_keys_entry_by_file_snapshot_id() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                3,
            )
            .await,
        ])
        .await;
        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();
        let registered = register_partition_stats_file(&catalog, &table, stats_file.clone())
            .await
            .unwrap();

        let entry = registered
            .metadata()
            .partition_statistics_for_snapshot(snapshot.snapshot_id())
            .expect("entry keyed by the file's snapshot id");
        assert_eq!(entry.snapshot_id, snapshot.snapshot_id());
        assert!(
            registered
                .metadata()
                .partition_statistics_for_snapshot(snapshot.snapshot_id() + 1)
                .is_none()
        );
    }

    // ---- Date partitions, cross-version projection, and `write.metadata.path`. -----------------

    fn date_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedFieldRef::from(NestedField::required(
                    1,
                    "d",
                    Type::Primitive(PrimitiveType::Date),
                )),
                NestedFieldRef::from(NestedField::required(
                    2,
                    "n",
                    Type::Primitive(PrimitiveType::Long),
                )),
            ])
            .build()
            .unwrap()
    }

    /// Creates a table partitioned by `identity(d)` over a `date` column (spec 0, field id 1000).
    async fn create_date_partitioned_table(catalog: &impl Catalog) -> Table {
        let spec = crate::spec::PartitionSpec::builder(date_schema())
            .with_spec_id(0)
            .add_partition_field("d", "d", Transform::Identity)
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
            .schema(date_schema())
            .partition_spec(spec.into_unbound())
            .build();
        catalog.create_table(&namespace, creation).await.unwrap()
    }

    fn date_struct(days: i32) -> Struct {
        Struct::from_iter([Some(Literal::date(days))])
    }

    /// Reads the Arrow `DataType` of each nested partition-struct field (field 0) from a stats file.
    async fn raw_partition_struct_field_types(
        file_io: &FileIO,
        path: &str,
    ) -> Vec<arrow_schema::DataType> {
        let bytes = file_io.new_input(path).unwrap().read().await.unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).unwrap();
        match builder.schema().field(0).data_type() {
            arrow_schema::DataType::Struct(fields) => fields
                .iter()
                .map(|field| field.data_type().clone())
                .collect(),
            other => panic!("partition column must be a struct, got {other:?}"),
        }
    }

    /// A `date`-partitioned table is the most common production shape, so it must write its stats
    /// file rather than error. This pins three things: the file writes, the partition child column is
    /// a real Arrow `Date32` carrying field id 1000, and the date value round-trips.
    #[tokio::test]
    async fn test_date_partitioned_table_writes_and_round_trips_date32_partition_column() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_date_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/d=19000/d1.parquet"),
                0,
                date_struct(19000),
                3,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/d=18000/d2.parquet"),
                0,
                date_struct(18000),
                5,
            )
            .await,
        ])
        .await;

        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let computed = compute_partition_stats(&table, &snapshot).await.unwrap();

        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .expect("a date-partitioned table writes a stats file");

        // The child column must be a real Date32, not an Int32, and carry field id 1000.
        let child_types =
            raw_partition_struct_field_types(&file_io, &stats_file.statistics_path).await;
        assert_eq!(
            child_types,
            vec![arrow_schema::DataType::Date32],
            "the date partition column must be a logical Date32 on disk"
        );
        let partition_ids =
            raw_partition_struct_field_ids(&file_io, &stats_file.statistics_path).await;
        assert_eq!(partition_ids, vec![1000], "the date partition field id");

        let stats_schema = partition_stats_schema(
            &unified_partition_type(table.metadata()).unwrap(),
            table.metadata().format_version(),
        )
        .unwrap();
        let read_back =
            read_partition_stats_file(&table, &stats_schema, &stats_file.statistics_path)
                .await
                .unwrap();
        let dates: Vec<Option<i32>> = read_back
            .iter()
            .map(|row| match row.partition().fields().first() {
                Some(Some(Literal::Primitive(PrimitiveLiteral::Int(value)))) => Some(*value),
                _ => None,
            })
            .collect();
        assert_eq!(
            dates,
            vec![Some(18000), Some(19000)],
            "date partition values must round-trip, sorted ascending"
        );
        assert_eq!(
            sort_rows(read_back),
            sort_rows(computed),
            "date-partitioned read-back equals the computed rows field-for-field"
        );
    }

    /// Round-trips a single-row `PartitionStats` holding `partition_value` through the production
    /// write and read paths. It returns the decoded row and the Arrow child type of the only partition
    /// field, so a test can pin the on-disk type.
    fn round_trip_single_partition_value(
        partition_type: PrimitiveType,
        partition_value: Literal,
    ) -> (PartitionStats, arrow_schema::DataType) {
        let unified = StructType::new(vec![Arc::new(NestedField::optional(
            1000,
            "p",
            Type::Primitive(partition_type),
        ))]);
        let stats_schema = partition_stats_schema(&unified, FormatVersion::V2).unwrap();

        let mut row = PartitionStats::new(Struct::from_iter([Some(partition_value)]), 0);
        row.data_record_count = 7;
        row.data_file_count = 1;
        row.total_data_file_size_in_bytes = 123;

        let batch =
            partition_stats_to_record_batch(std::slice::from_ref(&row), &stats_schema, &unified)
                .expect("the exotic partition value must write, not error");

        let child_type = match batch.schema().field(0).data_type() {
            arrow_schema::DataType::Struct(fields) => fields[0].data_type().clone(),
            other => panic!("partition column must be a struct, got {other:?}"),
        };

        let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&stats_schema).unwrap());
        let mut buffer: Vec<u8> = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buffer, arrow_schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let decoded = read_partition_stats_from_bytes(&stats_schema, Bytes::from(buffer)).unwrap();
        assert_eq!(decoded.len(), 1, "exactly one row round-trips");
        (decoded.into_iter().next().unwrap(), child_type)
    }

    /// A `time` partition must write a `Time64(Microsecond)` column and round-trip the micros.
    #[test]
    fn test_time_partition_value_round_trips_as_time64_micros() {
        let micros = 13 * 3_600_000_000 + 45 * 60_000_000 + 30_000_001; // 13:45:30.000001
        let (decoded, child_type) =
            round_trip_single_partition_value(PrimitiveType::Time, Literal::time(micros));
        assert_eq!(
            child_type,
            arrow_schema::DataType::Time64(arrow_schema::TimeUnit::Microsecond),
            "time partition column must be a logical Time64(Microsecond) on disk"
        );
        assert_eq!(
            decoded.partition().fields().first(),
            Some(&Some(Literal::Primitive(PrimitiveLiteral::Long(micros)))),
            "the time value (micros since midnight) must round-trip exactly"
        );
    }

    /// A `uuid` partition must write 16 big-endian bytes, so a little-endian write fails here.
    #[test]
    fn test_uuid_partition_value_round_trips_as_fixed_size_binary_16_big_endian() {
        let uuid = uuid::Uuid::parse_str("a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8").unwrap();
        let (decoded, child_type) =
            round_trip_single_partition_value(PrimitiveType::Uuid, Literal::uuid(uuid));
        assert_eq!(
            child_type,
            arrow_schema::DataType::FixedSizeBinary(16),
            "uuid partition column must be a FixedSizeBinary(16) on disk"
        );
        assert_eq!(
            decoded.partition().fields().first(),
            Some(&Some(Literal::Primitive(PrimitiveLiteral::UInt128(
                uuid.as_u128()
            )))),
            "the uuid value must round-trip exactly (16 BE bytes, Java byte form)"
        );
    }

    #[test]
    fn test_fixed_partition_value_round_trips_as_fixed_size_binary_len() {
        let bytes = vec![0xde, 0xad, 0xbe, 0xef];
        let (decoded, child_type) = round_trip_single_partition_value(
            PrimitiveType::Fixed(4),
            Literal::fixed(bytes.clone()),
        );
        assert_eq!(
            child_type,
            arrow_schema::DataType::FixedSizeBinary(4),
            "fixed[4] partition column must be a FixedSizeBinary(4) on disk"
        );
        assert_eq!(
            decoded.partition().fields().first(),
            Some(&Some(Literal::Primitive(PrimitiveLiteral::Binary(bytes)))),
            "the fixed bytes must round-trip exactly"
        );
    }

    #[test]
    fn test_binary_partition_value_round_trips_as_large_binary() {
        let bytes = vec![1u8, 2, 3, 255, 0, 128];
        let (decoded, child_type) = round_trip_single_partition_value(
            PrimitiveType::Binary,
            Literal::binary(bytes.clone()),
        );
        assert_eq!(
            child_type,
            arrow_schema::DataType::LargeBinary,
            "binary partition column must be a LargeBinary on disk"
        );
        assert_eq!(
            decoded.partition().fields().first(),
            Some(&Some(Literal::Primitive(PrimitiveLiteral::Binary(bytes)))),
            "the binary bytes must round-trip exactly"
        );
    }

    /// A `fixed[L]` value of the wrong byte length must fail loudly, never write a corrupt column.
    #[test]
    fn test_fixed_partition_value_wrong_width_errors_loudly() {
        let unified = StructType::new(vec![Arc::new(NestedField::optional(
            1000,
            "p",
            Type::Primitive(PrimitiveType::Fixed(4)),
        ))]);
        let stats_schema = partition_stats_schema(&unified, FormatVersion::V2).unwrap();
        let mut row =
            PartitionStats::new(Struct::from_iter([Some(Literal::fixed(vec![1, 2, 3]))]), 0);
        row.data_record_count = 1;
        row.data_file_count = 1;
        let error =
            partition_stats_to_record_batch(std::slice::from_ref(&row), &stats_schema, &unified)
                .expect_err("a fixed value of the wrong byte length must error, not write");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    /// The upgrade direction of the cross-version projection. A V2 file read against the V3 schema
    /// must null-fill the missing `dv_count` to 0, as Java's `project(v3Schema)` does. The
    /// shorter-record guard must handle the 12-field record, and never decode it wrongly.
    #[tokio::test]
    async fn test_v2_file_read_against_v3_schema_null_fills_dv_count() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await; // V2 by default.
        let location = table.metadata().location().to_string();
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
        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();

        let top_ids = raw_top_level_field_ids(&file_io, &stats_file.statistics_path).await;
        assert_eq!(top_ids.len(), 12, "the written file is v2 (12 columns)");

        let unified = unified_partition_type(table.metadata()).unwrap();
        let v3_schema = partition_stats_schema(&unified, FormatVersion::V3).unwrap();
        let read_back = read_partition_stats_file(&table, &v3_schema, &stats_file.statistics_path)
            .await
            .unwrap();
        assert_eq!(
            read_back.len(),
            1,
            "the single partition row decodes cleanly"
        );
        assert_eq!(
            read_back[0].dv_count(),
            0,
            "a v2 file projected onto the v3 schema null-fills dv_count to 0 (Java parity)"
        );
        assert_eq!(read_back[0].data_record_count(), 4);
        assert_eq!(read_back[0].spec_id(), 0);
    }

    /// The downgrade direction. A V3 file read against the V2 schema must decode its first 12 columns
    /// and ignore `dv_count`, with no error and no wrong decode. The projection keeps the v2 field ids,
    /// which the v3 file all holds.
    #[tokio::test]
    async fn test_v3_file_read_against_v2_schema_ignores_dv_count_column() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
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
                9,
            )
            .await,
        ])
        .await;
        let table = add_deletes(&catalog, &table, vec![
            dv_file(
                &file_io,
                &format!("{location}/data/x=1/dv.puffin"),
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                4,
            )
            .await,
        ])
        .await;

        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();
        let top_ids = raw_top_level_field_ids(&file_io, &stats_file.statistics_path).await;
        assert_eq!(top_ids.len(), 13, "the written file is v3 (13 columns)");

        let unified = unified_partition_type(table.metadata()).unwrap();
        let v2_schema = partition_stats_schema(&unified, FormatVersion::V2).unwrap();
        let read_back = read_partition_stats_file(&table, &v2_schema, &stats_file.statistics_path)
            .await
            .unwrap();
        assert_eq!(
            read_back.len(),
            1,
            "the single partition row decodes cleanly"
        );
        assert_eq!(read_back[0].data_record_count(), 9);
        assert_eq!(read_back[0].spec_id(), 0);
        assert_eq!(
            read_back[0].dv_count(),
            0,
            "v2 schema does not surface dv_count"
        );
    }

    /// A table that sets `write.metadata.path` must land its stats file under that directory. A
    /// hardcoded `<location>/metadata` ignores the override silently.
    #[tokio::test]
    async fn test_write_metadata_path_property_redirects_the_stats_file_location() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d.parquet"),
                0,
                x_struct(1),
                2,
            )
            .await,
        ])
        .await;

        let custom_metadata_dir = format!("{location}/custom-meta");
        let table = {
            let tx = Transaction::new(&table);
            let tx = tx
                .update_table_properties()
                .set(
                    WRITE_METADATA_PATH_PROPERTY.to_string(),
                    format!("{custom_metadata_dir}/"),
                )
                .apply(tx)
                .unwrap();
            tx.commit(&catalog).await.unwrap()
        };

        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let stats_file = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();

        let expected_prefix = format!(
            "{custom_metadata_dir}/partition-stats-{}-",
            snapshot.snapshot_id()
        );
        assert!(
            stats_file.statistics_path.starts_with(&expected_prefix),
            "stats file `{}` must land under write.metadata.path (`{expected_prefix}`), not <location>/metadata",
            stats_file.statistics_path
        );
        assert!(
            !stats_file
                .statistics_path
                .contains("/metadata/partition-stats-"),
            "must NOT fall back to <location>/metadata when write.metadata.path is set"
        );
        let stats_schema = partition_stats_schema(
            &unified_partition_type(table.metadata()).unwrap(),
            table.metadata().format_version(),
        )
        .unwrap();
        let read_back =
            read_partition_stats_file(&table, &stats_schema, &stats_file.statistics_path)
                .await
                .unwrap();
        assert_eq!(read_back.len(), 1);
    }

    // ---- The incremental compute: the diff-from-base path against the full recompute. ----------

    /// Registers a stats file for the current snapshot, to seed a base for the incremental path.
    async fn compute_register_current(catalog: &impl Catalog, table: &Table) -> Table {
        let snapshot = table.metadata().current_snapshot().unwrap().clone();
        let file = compute_and_write_stats_file(table, &snapshot)
            .await
            .unwrap()
            .expect("a partitioned snapshot with data writes a stats file");
        register_partition_stats_file(catalog, table, file)
            .await
            .unwrap()
    }

    /// Reads back the registered stats file for `snapshot`, returning the rows sorted for comparison.
    async fn read_back_registered(table: &Table, snapshot: &Snapshot) -> Vec<PartitionStats> {
        let file = table
            .metadata()
            .partition_statistics_for_snapshot(snapshot.snapshot_id())
            .expect("a stats file is registered for this snapshot");
        let stats_schema = partition_stats_schema(
            &unified_partition_type(table.metadata()).unwrap(),
            table.metadata().format_version(),
        )
        .unwrap();
        sort_rows(
            read_partition_stats_file(table, &stats_schema, &file.statistics_path)
                .await
                .unwrap(),
        )
    }

    /// The incremental compute must equal a full recompute of S2 on an append-only history. That is
    /// the contract: incremental is an optimization, so its stats must be identical.
    ///
    /// This shape partitions on `identity(x)`, and the diff adds files to an existing partition and to
    /// a new one. A bug in the added-only live filter, or in the diff manifest selection, breaks the
    /// equality here.
    #[tokio::test]
    async fn test_incremental_equals_full_recompute_append_only_single_field() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/a.parquet"),
                0,
                x_struct(1),
                3,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/x=2/b.parquet"),
                0,
                x_struct(2),
                5,
            )
            .await,
        ])
        .await;
        let table = compute_register_current(&catalog, &table).await;

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/c.parquet"),
                0,
                x_struct(1),
                4,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/x=3/d.parquet"),
                0,
                x_struct(3),
                2,
            )
            .await,
        ])
        .await;
        let s2 = table.metadata().current_snapshot().unwrap().clone();

        let table = compute_register_current(&catalog, &table).await;
        let incremental = read_back_registered(&table, &s2).await;

        let full = sort_rows(compute_partition_stats(&table, &s2).await.unwrap());

        assert_eq!(
            incremental, full,
            "incremental == full recompute on an append-only history (single-field partition)"
        );
        let x1 = full
            .iter()
            .find(|row| row.partition() == &x_struct(1))
            .unwrap();
        assert_eq!(
            x1.data_record_count(),
            7,
            "x=1 = S1's 3 + S2's 4 (not double-counted)"
        );
        assert_eq!(x1.data_file_count(), 2);
    }

    /// The two paths must still agree when the S2 diff adds an equality delete on top of an append.
    #[tokio::test]
    async fn test_incremental_equals_full_recompute_with_delete_in_diff() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/a.parquet"),
                0,
                x_struct(1),
                6,
            )
            .await,
        ])
        .await;
        let table = compute_register_current(&catalog, &table).await;

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=2/b.parquet"),
                0,
                x_struct(2),
                4,
            )
            .await,
        ])
        .await;
        let table = add_deletes(&catalog, &table, vec![
            equality_delete_file(
                &file_io,
                &format!("{location}/data/x=1/eq.parquet"),
                0,
                x_struct(1),
                2,
            )
            .await,
        ])
        .await;
        let s2 = table.metadata().current_snapshot().unwrap().clone();

        let table = compute_register_current(&catalog, &table).await;
        let incremental = read_back_registered(&table, &s2).await;
        let full = sort_rows(compute_partition_stats(&table, &s2).await.unwrap());

        assert_eq!(
            incremental, full,
            "incremental == full recompute when the diff carries a newly-added delete file"
        );
        let x1 = full
            .iter()
            .find(|row| row.partition() == &x_struct(1))
            .unwrap();
        assert_eq!(x1.data_record_count(), 6);
        assert_eq!(x1.equality_delete_record_count(), 2);
        assert_eq!(x1.equality_delete_file_count(), 1);
    }

    /// The added-only live filter, against a silent double-count. A merge-append re-stamps S1's
    /// carried files as EXISTING into a new manifest that S2 adds. The diff selects that manifest, so
    /// it meets those EXISTING entries, which the filter must skip. Without the filter x=1's records
    /// count twice, once from the seed and once from the diff. A fast-append fixture cannot see this,
    /// because its S2 manifest holds only ADDED entries.
    #[tokio::test]
    async fn test_incremental_added_only_filter_skips_existing_entries_in_merged_manifest() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // Merging at 2 manifests makes S2's merge-append re-stamp S1's entry as EXISTING.
        let table = {
            let tx = Transaction::new(&table);
            let tx = tx
                .update_table_properties()
                .set(
                    "commit.manifest.min-count-to-merge".to_string(),
                    "2".to_string(),
                )
                .apply(tx)
                .unwrap();
            tx.commit(&catalog).await.unwrap()
        };

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/a.parquet"),
                0,
                x_struct(1),
                3,
            )
            .await,
        ])
        .await;
        let table = compute_register_current(&catalog, &table).await;

        let table = {
            let tx = Transaction::new(&table);
            let tx = tx
                .merge_append()
                .add_data_files(vec![
                    data_file(
                        &file_io,
                        &format!("{location}/data/x=1/b.parquet"),
                        0,
                        x_struct(1),
                        5,
                    )
                    .await,
                ])
                .apply(tx)
                .unwrap();
            tx.commit(&catalog).await.unwrap()
        };
        let s2 = table.metadata().current_snapshot().unwrap().clone();

        let table = compute_register_current(&catalog, &table).await;
        let incremental = read_back_registered(&table, &s2).await;
        let full = sort_rows(compute_partition_stats(&table, &s2).await.unwrap());

        assert_eq!(
            incremental, full,
            "the ADDED-only filter must skip the EXISTING (carried) entries in S2's merged manifest"
        );
        // x=1 holds 8 records over 2 files, counted once. A double-count reads 11.
        let x1 = full
            .iter()
            .find(|row| row.partition() == &x_struct(1))
            .unwrap();
        assert_eq!(
            x1.data_record_count(),
            8,
            "x=1 = 3 (seed) + 5 (diff's ADDED file); the EXISTING carried file is NOT re-counted"
        );
        assert_eq!(x1.data_file_count(), 2);
    }

    /// A registered but unreadable base file must fall back to a full compute, and must not error.
    #[tokio::test]
    async fn test_incremental_falls_back_to_full_when_base_stats_file_is_corrupt() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/a.parquet"),
                0,
                x_struct(1),
                3,
            )
            .await,
        ])
        .await;
        let s1 = table.metadata().current_snapshot().unwrap().clone();

        let bogus = PartitionStatisticsFile {
            snapshot_id: s1.snapshot_id(),
            statistics_path: format!(
                "{location}/metadata/partition-stats-{}-bogus.parquet",
                s1.snapshot_id()
            ),
            file_size_in_bytes: 999,
        };
        let table = register_partition_stats_file(&catalog, &table, bogus)
            .await
            .unwrap();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=2/b.parquet"),
                0,
                x_struct(2),
                5,
            )
            .await,
        ])
        .await;
        let s2 = table.metadata().current_snapshot().unwrap().clone();

        let table = compute_register_current(&catalog, &table).await;
        let written = read_back_registered(&table, &s2).await;
        let full = sort_rows(compute_partition_stats(&table, &s2).await.unwrap());

        assert_eq!(
            written, full,
            "a corrupt base stats file makes the incremental path fall back to a full compute"
        );
        assert_eq!(written.len(), 2, "both x=1 and x=2 present after fallback");
    }

    /// A snapshot that already has a registered stats file must get it back with no recompute.
    #[tokio::test]
    async fn test_compute_returns_existing_stats_file_for_already_computed_snapshot() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/a.parquet"),
                0,
                x_struct(1),
                3,
            )
            .await,
        ])
        .await;
        let snapshot = table.metadata().current_snapshot().unwrap().clone();

        let first = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();
        let table = register_partition_stats_file(&catalog, &table, first.clone())
            .await
            .unwrap();

        let again = compute_and_write_stats_file(&table, &snapshot)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            again, first,
            "an already-stats'd snapshot returns its existing file unchanged (no rewrite)"
        );
    }

    /// The strongest end-to-end pin for the SUBTRACT arm. The diff range holds a DELETE that leaves a
    /// DELETED tombstone for a file the base counted. The seed carries that file's contribution, so
    /// the diff must subtract it back out to match a full recompute, which never counts it at all. The
    /// append-only shapes above cannot reach the subtract arm, because their diff manifests hold no
    /// tombstone.
    #[tokio::test]
    async fn test_incremental_equals_full_recompute_with_delete_subtracting_base_file() {
        let (catalog, file_io, _temp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/a.parquet"),
                0,
                x_struct(1),
                3,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/x=1/b.parquet"),
                0,
                x_struct(1),
                5,
            )
            .await,
        ])
        .await;
        let table = compute_register_current(&catalog, &table).await;

        // Deleting a.parquet leaves a tombstone in an S2 manifest, so the diff subtracts it.
        let table = {
            let tx = Transaction::new(&table);
            let tx = tx
                .delete_files()
                .delete_file(format!("{location}/data/x=1/a.parquet"))
                .apply(tx)
                .unwrap();
            tx.commit(&catalog).await.unwrap()
        };
        let s2 = table.metadata().current_snapshot().unwrap().clone();

        let table = compute_register_current(&catalog, &table).await;
        let incremental = read_back_registered(&table, &s2).await;
        let full = sort_rows(compute_partition_stats(&table, &s2).await.unwrap());

        assert_eq!(
            incremental, full,
            "incremental == full recompute when the diff's DELETED tombstone subtracts a base file"
        );
        // The seed of 8 records over 2 files, less a.parquet, leaves b alone: 5 records, 1 file.
        let x1 = full
            .iter()
            .find(|row| row.partition() == &x_struct(1))
            .unwrap();
        assert_eq!(
            x1.data_record_count(),
            5,
            "the deleted a.parquet (3 rec) is subtracted back out: 8 - 3 = 5"
        );
        assert_eq!(x1.data_file_count(), 1, "2 base files - 1 deleted = 1");
    }
}
