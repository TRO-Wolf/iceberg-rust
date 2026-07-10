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

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize, Serializer};

use crate::expr::BoundPredicate;
use crate::spec::{
    DataContentType, DataFileFormat, ManifestEntryRef, NameMapping, PartitionSpec, Schema,
    SchemaRef, Struct,
};
use crate::{Error, ErrorKind, Result};

/// Whether a data file in this format can be split into byte ranges, porting the
/// `splittable` flag on Java `org.apache.iceberg.FileFormat` (1.10.0): `PARQUET` / `AVRO` /
/// `ORC` are splittable; `PUFFIN` (and `METADATA`, which has no Rust analogue) are NOT. A
/// data file never carries `METADATA`, so the only non-splittable case reachable here is a
/// `PUFFIN` file.
fn is_splittable(format: DataFileFormat) -> bool {
    match format {
        DataFileFormat::Parquet | DataFileFormat::Avro | DataFileFormat::Orc => true,
        DataFileFormat::Puffin => false,
    }
}

/// Whether `values` is strictly ascending (each element strictly greater than its predecessor),
/// porting Java `ArrayUtil.isStrictlyAscending(long[])` — the gate Java's offsets-aware split
/// uses before trusting the split offsets. An empty or single-element array is vacuously
/// ascending in Java; the split caller separately requires a non-empty offsets array.
fn is_strictly_ascending(values: &[i64]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

/// A stream of [`FileScanTask`].
pub type FileScanTaskStream = BoxStream<'static, Result<FileScanTask>>;

/// A stream of [`ChangelogScanTask`].
pub type ChangelogScanTaskStream = BoxStream<'static, Result<ChangelogScanTask>>;

/// The kind of row-level change a [`ChangelogScanTask`] produces.
///
/// Ports Java `org.apache.iceberg.ChangelogOperation`
/// (`api/src/main/java/org/apache/iceberg/ChangelogOperation.java`: `INSERT, DELETE,
/// UPDATE_BEFORE, UPDATE_AFTER`). The core changelog planner only ever produces
/// [`Insert`](Self::Insert) / [`Delete`](Self::Delete) — Java 1.10.0's
/// `BaseIncrementalChangelogScan` constructs only `BaseAddedRowsScanTask` /
/// `BaseDeletedDataFileScanTask`, whose `operation()` defaults are INSERT / DELETE.
/// `UpdateBefore` / `UpdateAfter` are declared for API parity and for downstream
/// engines: collapsing a delete+insert at the same row key into an update pair is an
/// ENGINE-side step in Java (Spark's `ChangelogIterator`; no such class exists in the
/// `iceberg-core`/`iceberg-api` 1.10.0 jars), so this library declares the variants but
/// never emits them from the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangelogOperation {
    /// Rows were INSERTED — the task's data file was ADDED by its commit snapshot.
    Insert,
    /// Rows were DELETED — removed with the whole data file, or marked deleted by
    /// row-level delete files added by the commit snapshot.
    Delete,
    /// The BEFORE image of an update pair (Java `UPDATE_BEFORE`). Never emitted by the
    /// planner — produced only by an engine-side net-change pairing step.
    UpdateBefore,
    /// The AFTER image of an update pair (Java `UPDATE_AFTER`). Never emitted by the
    /// planner — produced only by an engine-side net-change pairing step.
    UpdateAfter,
}

/// Which Java changelog task type a [`ChangelogScanTask`] corresponds to.
///
/// Ports the Java `ChangelogScanTask` sub-interface split
/// (`api/AddedRowsScanTask.java` / `api/DeletedDataFileScanTask.java` /
/// `api/DeletedRowsScanTask.java`). The [`ChangelogScanTask::operation`] is DERIVED
/// from this kind exactly as Java's per-interface `default ChangelogOperation
/// operation()` methods do: `AddedRows → INSERT`, `DeletedDataFile → DELETE`,
/// `DeletedRows → DELETE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangelogTaskKind {
    /// The task's data file was ADDED by the commit snapshot (Java
    /// `AddedRowsScanTask`). Reading the file with
    /// [`added_deletes`](ChangelogScanTask::added_deletes) applied yields the NET
    /// inserted rows (matching deletes committed in the same snapshot fold in here —
    /// see the `AddedRowsScanTask` javadoc example).
    AddedRows,
    /// The task's data file was REMOVED (whole file) by the commit snapshot (Java
    /// `DeletedDataFileScanTask`). Reading the file with
    /// [`existing_deletes`](ChangelogScanTask::existing_deletes) applied yields the
    /// rows that were still live when the file was removed — the rows this change
    /// deletes.
    DeletedDataFile,
    /// The commit snapshot added row-level delete files that apply to this EXISTING
    /// data file (Java `DeletedRowsScanTask`). The rows removed by
    /// [`added_deletes`](ChangelogScanTask::added_deletes) — among the rows that
    /// survive [`existing_deletes`](ChangelogScanTask::existing_deletes) — are the rows
    /// this change deletes. **ENGINE-FIRST:** Java 1.10.0 core never constructs its
    /// `BaseDeletedRowsScanTask`; this variant is produced only by the opt-in
    /// [`with_row_level_deletes`](crate::scan::IncrementalChangelogScanBuilder::with_row_level_deletes)
    /// mode.
    DeletedRows,
}

/// A changelog scan task: the row-level changes (all inserts or all deletes) carried by
/// a single data file for one snapshot in the changelog range.
///
/// Ports Java `ChangelogScanTask` + its `BaseAddedRowsScanTask` /
/// `BaseDeletedDataFileScanTask` / `BaseDeletedRowsScanTask` implementations, collapsed
/// into one struct discriminated by [`kind`](Self::kind). A task embeds the
/// [`FileScanTask`] that reads the underlying data file (path, schema, projection,
/// residual predicate) and tags it with the change metadata: the
/// [`kind`](Self::kind) (which Java task type it is), the derived
/// [`operation`](Self::operation) (insert vs delete), the
/// [`change_ordinal`](Self::change_ordinal) (0 for the oldest snapshot in the range,
/// incrementing — changes with a lower ordinal must be applied first), the
/// [`commit_snapshot_id`](Self::commit_snapshot_id) (the snapshot that committed the
/// change), and the delete-file attachments
/// ([`added_deletes`](Self::added_deletes) / [`existing_deletes`](Self::existing_deletes))
/// mirroring `AddedRowsScanTask.deletes()` / `DeletedRowsScanTask.addedDeletes()` +
/// `existingDeletes()` / `DeletedDataFileScanTask.existingDeletes()`.
///
/// In the DEFAULT (Java-1.10.0-parity) mode both delete lists are always empty (Java
/// passes `NO_DELETES`) and the scan rejects a range that contains row-level delete
/// manifests (see [`IncrementalChangelogScan`](super::IncrementalChangelogScan)). The
/// opt-in row-level mode populates them.
#[derive(Debug, Clone, PartialEq)]
pub struct ChangelogScanTask {
    /// The change ordinal: `0` for the oldest snapshot in the range, incrementing for
    /// each newer snapshot. Changes with a lower ordinal must be applied first (Java
    /// `ChangelogScanTask.changeOrdinal()`).
    pub change_ordinal: i32,
    /// The id of the snapshot that committed this change (Java
    /// `ChangelogScanTask.commitSnapshotId()`).
    pub commit_snapshot_id: i64,
    /// Which Java changelog task type this is; [`operation`](Self::operation) derives
    /// from it (making a kind/operation mismatch unrepresentable).
    pub kind: ChangelogTaskKind,
    /// Delete files ADDED by the commit snapshot that apply to the task's data file:
    /// Java `AddedRowsScanTask.deletes()` for [`ChangelogTaskKind::AddedRows`],
    /// `DeletedRowsScanTask.addedDeletes()` for [`ChangelogTaskKind::DeletedRows`].
    /// Always empty for [`ChangelogTaskKind::DeletedDataFile`] and in the default
    /// data-file changelog mode.
    pub added_deletes: Vec<FileScanTaskDeleteFile>,
    /// Delete files that existed BEFORE the commit snapshot and apply to the task's
    /// data file: Java `DeletedDataFileScanTask.existingDeletes()` /
    /// `DeletedRowsScanTask.existingDeletes()` ("must be applied prior to determining
    /// which records are deleted"). Always empty for
    /// [`ChangelogTaskKind::AddedRows`] (a file added by the commit snapshot postdates
    /// every pre-existing delete) and in the default data-file changelog mode.
    pub existing_deletes: Vec<FileScanTaskDeleteFile>,
    /// The underlying file scan task that reads the data file whose rows changed. Its
    /// [`deletes`](FileScanTask::deletes) carry the delete files a plain MoR read of
    /// this task should apply: `added_deletes` for an `AddedRows` task (⇒ the net
    /// inserted rows) and `existing_deletes` for a `DeletedDataFile` or `DeletedRows`
    /// task (⇒ the rows live before this change; for `DeletedRows` the engine then uses
    /// `added_deletes` as the SELECTOR of which of those rows became deleted).
    pub file_scan_task: FileScanTask,
}

impl ChangelogScanTask {
    /// Returns the kind of change (insert / delete) this task produces, derived from
    /// [`kind`](Self::kind) exactly as Java's per-task-interface `default operation()`
    /// implementations: `AddedRowsScanTask → INSERT`, `DeletedDataFileScanTask` /
    /// `DeletedRowsScanTask → DELETE`.
    pub fn operation(&self) -> ChangelogOperation {
        match self.kind {
            ChangelogTaskKind::AddedRows => ChangelogOperation::Insert,
            ChangelogTaskKind::DeletedDataFile | ChangelogTaskKind::DeletedRows => {
                ChangelogOperation::Delete
            }
        }
    }

    /// Returns which Java changelog task type this task corresponds to.
    pub fn kind(&self) -> ChangelogTaskKind {
        self.kind
    }

    /// Delete files ADDED by the commit snapshot that apply to this task's data file
    /// (Java `AddedRowsScanTask.deletes()` / `DeletedRowsScanTask.addedDeletes()`).
    pub fn added_deletes(&self) -> &[FileScanTaskDeleteFile] {
        &self.added_deletes
    }

    /// Delete files that existed before the commit snapshot and apply to this task's
    /// data file (Java `DeletedDataFileScanTask.existingDeletes()` /
    /// `DeletedRowsScanTask.existingDeletes()`).
    pub fn existing_deletes(&self) -> &[FileScanTaskDeleteFile] {
        &self.existing_deletes
    }

    /// Returns the change ordinal — changes with a lower ordinal must be applied first.
    pub fn change_ordinal(&self) -> i32 {
        self.change_ordinal
    }

    /// Returns the id of the snapshot that committed this change.
    pub fn commit_snapshot_id(&self) -> i64 {
        self.commit_snapshot_id
    }

    /// Returns the underlying [`FileScanTask`] that reads the changed data file.
    pub fn file_scan_task(&self) -> &FileScanTask {
        &self.file_scan_task
    }

    /// Returns the data file path of the changed file.
    pub fn data_file_path(&self) -> &str {
        &self.file_scan_task.data_file_path
    }
}

/// Serialization helper that always returns NotImplementedError.
/// Used for fields that should not be serialized but we want to be explicit about it.
fn serialize_not_implemented<S, T>(_: &T, _: S) -> std::result::Result<S::Ok, S::Error>
where S: Serializer {
    Err(serde::ser::Error::custom(
        "Serialization not implemented for this field",
    ))
}

/// Deserialization helper that always returns NotImplementedError.
/// Used for fields that should not be deserialized but we want to be explicit about it.
fn deserialize_not_implemented<'de, D, T>(_: D) -> std::result::Result<T, D::Error>
where D: serde::Deserializer<'de> {
    Err(serde::de::Error::custom(
        "Deserialization not implemented for this field",
    ))
}

/// A task to scan part of file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileScanTask {
    /// The total size of the data file in bytes, from the manifest entry.
    /// Used to skip a stat/HEAD request when reading Parquet footers.
    pub file_size_in_bytes: u64,
    /// The start offset of the file to scan.
    pub start: u64,
    /// The length of the file to scan.
    pub length: u64,
    /// The number of records in the file to scan.
    ///
    /// This is an optional field, and only available if we are
    /// reading the entire data file.
    pub record_count: Option<u64>,

    /// The data file path corresponding to the task.
    pub data_file_path: String,

    /// The format of the file to scan.
    pub data_file_format: DataFileFormat,

    /// The schema of the file to scan.
    pub schema: SchemaRef,
    /// The field ids to project.
    pub project_field_ids: Vec<i32>,
    /// The predicate to filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<BoundPredicate>,

    /// The list of delete files that may need to be applied to this data file
    pub deletes: Vec<FileScanTaskDeleteFile>,

    /// Partition data from the manifest entry, used to identify which columns can use
    /// constant values from partition metadata vs. reading from the data file.
    /// Per the Iceberg spec, only identity-transformed partition fields should use constants.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_not_implemented")]
    #[serde(deserialize_with = "deserialize_not_implemented")]
    pub partition: Option<Struct>,

    /// The partition spec for this file, used to distinguish identity transforms
    /// (which use partition metadata constants) from non-identity transforms like
    /// bucket/truncate (which must read source columns from the data file).
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_not_implemented")]
    #[serde(deserialize_with = "deserialize_not_implemented")]
    pub partition_spec: Option<Arc<PartitionSpec>>,

    /// Name mapping from table metadata (property: schema.name-mapping.default),
    /// used to resolve field IDs from column names when Parquet files lack field IDs
    /// or have field ID conflicts.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_not_implemented")]
    #[serde(deserialize_with = "deserialize_not_implemented")]
    pub name_mapping: Option<Arc<NameMapping>>,

    /// Whether this scan task should treat column names as case-sensitive when binding predicates.
    pub case_sensitive: bool,

    /// The data file's split offsets (Java `ContentFile.splitOffsets()` / field id 132 —
    /// "all row group offsets in a Parquet file"), threaded from the manifest entry's
    /// [`DataFile::split_offsets`](crate::spec::DataFile::split_offsets).
    ///
    /// **Flagged additive field.** Whole-file tasks produced by [`plan_files`] carry this so the
    /// split layer ([`plan_tasks`]) can split each file at its row-group boundaries (Java
    /// `BaseContentScanTask.split` offsets-aware branch). It is `None` when the manifest entry has
    /// no split offsets, and it is reset to `None` on every SUB-task a split produces (a sub-task
    /// covers an arbitrary `[start, start+length)` window, NOT the file's row-group grid, so it
    /// must not re-split). It serializes as an absent key when `None` so a serialized whole-file
    /// task that predates this field round-trips unchanged.
    ///
    /// [`plan_files`]: crate::scan::TableScan::plan_files
    /// [`plan_tasks`]: crate::scan::TableScan::plan_tasks
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_offsets: Option<Vec<i64>>,
}

impl FileScanTask {
    /// Returns the data file path of this file scan task.
    pub fn data_file_path(&self) -> &str {
        &self.data_file_path
    }

    /// Returns the byte length of the data file region this task reads.
    pub fn length(&self) -> u64 {
        self.length
    }

    /// Returns the start byte offset of the data file region this task reads.
    pub fn start(&self) -> u64 {
        self.start
    }

    /// Splits this task into target-sized sub-tasks, porting Java
    /// `BaseContentScanTask.split(long targetSplitSize)` (1.10.0) exactly.
    ///
    /// The three branches, verified against the live 1.10.0 `iceberg-core` bytecode:
    ///
    /// 1. **Not splittable** (`!file.format().isSplittable()`): return `[self]` unchanged. Per the
    ///    Java `FileFormat` table, `PUFFIN` is the only non-splittable format a data file could
    ///    carry; `PARQUET` / `AVRO` / `ORC` are splittable.
    /// 2. **Offsets-aware** (split offsets present AND strictly ascending —
    ///    `ArrayUtil.isStrictlyAscending`): emit ONE sub-task per offset. For `i in 0..n-1`,
    ///    `length[i] = offsets[i+1] - offsets[i]`; the LAST is `length[n-1] = fileLength - offsets[n-1]`,
    ///    with `start[i] = offsets[i]`. **`target` is IGNORED in this branch** (Java's
    ///    `OffsetsAwareSplitScanTaskIterator` never reads it).
    /// 3. **Fixed-size** (else): walk `0..length` emitting windows of
    ///    `min(target, remaining)` (Java `FixedSizeSplitScanTaskIterator`).
    ///
    /// Every sub-task carries the SAME `deletes` / `predicate` (residual) / `partition` / spec /
    /// schema / projection as `self` — only `start` + `length` change (Java's split task delegates
    /// every field but `start`/`length` to its parent), and `record_count` is dropped (`None`,
    /// matching the "only available when reading the entire data file" contract) and `split_offsets`
    /// is cleared (a sub-task window is not the file's row-group grid).
    ///
    /// `target` must be `> 0` (Java `TableScanUtil.splitFiles` precondition); a non-positive target
    /// is a [`DataInvalid`](crate::ErrorKind::DataInvalid) error.
    pub fn split(&self, target: u64) -> Result<Vec<FileScanTask>> {
        if target == 0 {
            return Err(Error::new(ErrorKind::DataInvalid, "Split size must be > 0"));
        }

        // (1) Non-splittable format ⇒ the whole file is one task.
        if !is_splittable(self.data_file_format) {
            return Ok(vec![self.clone()]);
        }

        // (2) Offsets-aware: split offsets present AND strictly ascending.
        if let Some(offsets) = self.split_offsets.as_ref()
            && !offsets.is_empty()
            && is_strictly_ascending(offsets)
        {
            return self.split_at_offsets(offsets);
        }

        // (3) Fixed-size: walk the file in `min(target, remaining)` windows.
        Ok(self.split_fixed_size(target))
    }

    /// The offsets-aware split (branch 2). Each sub-task starts at `offsets[i]` with length
    /// `offsets[i+1] - offsets[i]`, the last running to `self.length` (the file length). Offsets
    /// are stored as `i64` in the manifest (Iceberg spec); they are guaranteed `>= 0` and strictly
    /// ascending here, so the `u64` conversion is on a bounded non-negative domain.
    fn split_at_offsets(&self, offsets: &[i64]) -> Result<Vec<FileScanTask>> {
        let mut sub_tasks = Vec::with_capacity(offsets.len());
        for (i, &offset) in offsets.iter().enumerate() {
            let start: u64 = offset.try_into().map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("split offset must be non-negative, got {offset}"),
                )
            })?;
            // length = next_offset - this_offset, or file_length - this_offset for the last.
            let end: u64 = if i + 1 < offsets.len() {
                offsets[i + 1].try_into().map_err(|_| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("split offset must be non-negative, got {}", offsets[i + 1]),
                    )
                })?
            } else {
                self.length
            };
            let length = end.saturating_sub(start);
            sub_tasks.push(self.sub_task(start, length));
        }
        Ok(sub_tasks)
    }

    /// The fixed-size split (branch 3). Walks `0..self.length` emitting windows of
    /// `min(target, remaining)`, advancing `offset += len` and `remaining -= len` each step
    /// (Java `FixedSizeSplitScanTaskIterator`).
    fn split_fixed_size(&self, target: u64) -> Vec<FileScanTask> {
        let mut sub_tasks = Vec::new();
        let mut offset = 0u64;
        let mut remaining = self.length;
        while remaining > 0 {
            let len = target.min(remaining);
            sub_tasks.push(self.sub_task(offset, len));
            offset += len;
            remaining -= len;
        }
        sub_tasks
    }

    /// Builds one sub-task covering `[start, start + length)`, cloning every parent field but
    /// `start`/`length` (Java's split task delegates all but the byte window to its parent). The
    /// `record_count` is dropped (only meaningful for the whole file) and `split_offsets` is
    /// cleared (a sub-task window is not the file's row-group grid).
    fn sub_task(&self, start: u64, length: u64) -> FileScanTask {
        FileScanTask {
            file_size_in_bytes: self.file_size_in_bytes,
            start,
            length,
            record_count: None,
            data_file_path: self.data_file_path.clone(),
            data_file_format: self.data_file_format,
            schema: self.schema.clone(),
            project_field_ids: self.project_field_ids.clone(),
            predicate: self.predicate.clone(),
            deletes: self.deletes.clone(),
            partition: self.partition.clone(),
            partition_spec: self.partition_spec.clone(),
            name_mapping: self.name_mapping.clone(),
            case_sensitive: self.case_sensitive,
            split_offsets: None,
        }
    }

    /// The bin-packing WEIGHT of this task, porting Java `TableScanUtil.lambda$planTasks$3`:
    /// `max( length + contentSizeInBytes(deletes), (1 + deletes.len()) * openFileCost )`.
    ///
    /// `contentSizeInBytes` (Java `ScanTaskUtil`) sums, per attached delete file, the deletion-vector
    /// blob size (`content_size_in_bytes`) when the delete is a Puffin DV
    /// (`ContentFileUtil.isDV`: `format == PUFFIN`), else the delete file's `file_size_in_bytes`. The
    /// floor term `(1 + #deletes) * openFileCost` charges one open per data file plus one per delete.
    ///
    /// Arithmetic is saturating to avoid an overflow panic on adversarial sizes (Java uses `long`
    /// and would silently wrap; saturation is the safer faithful-on-the-real-domain choice).
    pub(crate) fn weight(&self, open_file_cost: u64) -> u64 {
        let delete_bytes: u64 = self
            .deletes
            .iter()
            .map(FileScanTaskDeleteFile::content_size_in_bytes)
            .fold(0u64, u64::saturating_add);
        let size_term = self.length.saturating_add(delete_bytes);

        // `deletes.len()` is a `usize`; `u64::try_from` is exact on all 64-bit targets and a
        // clamp-free widening here — fall back to `u64::MAX` only on the (impossible on a real
        // plan) >u64 case, which keeps the floor saturating rather than panicking.
        let num_deletes = u64::try_from(self.deletes.len()).unwrap_or(u64::MAX);
        let opens: u64 = 1u64.saturating_add(num_deletes);
        let cost_term = opens.saturating_mul(open_file_cost);

        size_term.max(cost_term)
    }

    /// Returns the project field id of this file scan task.
    pub fn project_field_ids(&self) -> &[i32] {
        &self.project_field_ids
    }

    /// Returns the predicate of this file scan task.
    pub fn predicate(&self) -> Option<&BoundPredicate> {
        self.predicate.as_ref()
    }

    /// Returns the schema of this file scan task as a reference
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Returns the schema of this file scan task as a SchemaRef
    pub fn schema_ref(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[derive(Debug)]
pub(crate) struct DeleteFileContext {
    pub(crate) manifest_entry: ManifestEntryRef,
    pub(crate) partition_spec_id: i32,
}

impl From<&DeleteFileContext> for FileScanTaskDeleteFile {
    fn from(ctx: &DeleteFileContext) -> Self {
        FileScanTaskDeleteFile {
            file_path: ctx.manifest_entry.file_path().to_string(),
            file_size_in_bytes: ctx.manifest_entry.file_size_in_bytes(),
            file_type: ctx.manifest_entry.content_type(),
            partition_spec_id: ctx.partition_spec_id,
            equality_ids: ctx.manifest_entry.data_file.equality_ids.clone(),
            file_format: ctx.manifest_entry.data_file.file_format,
            referenced_data_file: ctx.manifest_entry.data_file.referenced_data_file.clone(),
            content_offset: ctx.manifest_entry.data_file.content_offset,
            content_size_in_bytes: ctx.manifest_entry.data_file.content_size_in_bytes,
            record_count: Some(ctx.manifest_entry.data_file.record_count),
        }
    }
}

/// The format a [`FileScanTaskDeleteFile`] deserialized from a pre-deletion-vector
/// serialization defaults to: every delete file was a parquet file before Puffin deletion
/// vectors existed, so absent means parquet.
fn default_delete_file_format() -> DataFileFormat {
    DataFileFormat::Parquet
}

/// A task to scan part of file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileScanTaskDeleteFile {
    /// The delete file path
    pub file_path: String,

    /// The total size of the delete file in bytes, from the manifest entry.
    pub file_size_in_bytes: u64,

    /// delete file type
    pub file_type: DataContentType,

    /// partition id
    pub partition_spec_id: i32,

    /// equality ids for equality deletes (null for anything other than equality-deletes)
    pub equality_ids: Option<Vec<i32>>,

    /// The on-disk format of the delete file. This is the deletion-vector discriminator Java
    /// uses (`ContentFileUtil.isDV`: `deleteFile.format() == FileFormat.PUFFIN`): a
    /// position-delete entry whose format is [`DataFileFormat::Puffin`] is a deletion vector and
    /// must be loaded from its Puffin blob, never the parquet reader.
    #[serde(default = "default_delete_file_format")]
    pub file_format: DataFileFormat,

    /// The data file path a deletion vector (or file-scoped position delete) applies to, from
    /// the manifest entry's `referenced_data_file`. A loaded deletion vector is keyed by THIS
    /// path — required for deletion vectors.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub referenced_data_file: Option<String>,

    /// Offset of the `deletion-vector-v1` blob within the Puffin file, from the manifest
    /// entry's `content_offset`; required for deletion vectors.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_offset: Option<i64>,

    /// Length of the `deletion-vector-v1` blob in bytes, from the manifest entry's
    /// `content_size_in_bytes`; required for deletion vectors.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_size_in_bytes: Option<i64>,

    /// The record count from the manifest entry. For a deletion vector this is its cardinality
    /// (the number of deleted positions) and is validated against the decoded bitmap, mirroring
    /// Java `BitmapPositionDeleteIndex.deserializeBitmap`'s "Invalid cardinality" check.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_count: Option<u64>,
}

impl FileScanTaskDeleteFile {
    /// The "content size" of this delete file for bin-packing weight, porting Java
    /// `ScanTaskUtil.contentSizeInBytes(ContentFile)` for the delete-file case
    /// (`FileContent != DATA`):
    ///
    /// * a deletion vector (`ContentFileUtil.isDV`: `format == PUFFIN`) contributes its DV-blob
    ///   size — the manifest entry's `content_size_in_bytes` (the Puffin blob length), NOT the
    ///   whole Puffin file size;
    /// * any other delete file contributes its `file_size_in_bytes`.
    ///
    /// A DV row missing `content_size_in_bytes` is malformed (the field is required for a DV);
    /// rather than panic on adversarial input, fall back to `file_size_in_bytes` so the weight
    /// stays finite. A negative `content_size_in_bytes` (impossible from a well-formed manifest)
    /// is treated as the same fall-back.
    pub(crate) fn content_size_in_bytes(&self) -> u64 {
        if self.file_format == DataFileFormat::Puffin
            && let Some(size) = self.content_size_in_bytes
            && let Ok(size) = u64::try_from(size)
        {
            return size;
        }
        self.file_size_in_bytes
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::spec::{DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema, Type};

    /// A bare whole-file [`FileScanTask`] for the split/weight unit tests: `length` byte file in
    /// `format`, no deletes, no split offsets. The schema/partition fields are inert here (split +
    /// weight only read `length` / `data_file_format` / `deletes` / `split_offsets`).
    fn task(length: u64, format: DataFileFormat, split_offsets: Option<Vec<i64>>) -> FileScanTask {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("schema builds"),
        );
        FileScanTask {
            file_size_in_bytes: length,
            start: 0,
            length,
            record_count: Some(1000),
            data_file_path: "memory://t/data/1.parquet".to_string(),
            data_file_format: format,
            schema,
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets,
        }
    }

    /// A position-delete attachment of `size` bytes for the weight tests.
    fn pos_delete(size: u64) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile {
            file_path: "memory://t/data/del.parquet".to_string(),
            file_size_in_bytes: size,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }
    }

    /// A deletion-vector (Puffin) attachment: whole Puffin file is `file_size`, the DV blob is
    /// `blob_size` — the weight must charge the BLOB size, not the file size.
    fn dv_delete(file_size: u64, blob_size: i64) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile {
            file_path: "memory://t/data/del.puffin".to_string(),
            file_size_in_bytes: file_size,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Puffin,
            referenced_data_file: Some("memory://t/data/1.parquet".to_string()),
            content_offset: Some(4),
            content_size_in_bytes: Some(blob_size),
            record_count: Some(10),
        }
    }

    // ---- split: non-splittable passthrough ----

    #[test]
    fn split_non_splittable_returns_self() {
        // Puffin is not splittable ⇒ the whole file is one task even with offsets + a tiny target.
        let t = task(1000, DataFileFormat::Puffin, Some(vec![0, 500]));
        let parts = t.split(100).expect("split ok");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].start, 0);
        assert_eq!(parts[0].length, 1000);
        // Passthrough returns self verbatim (record_count + offsets retained).
        assert_eq!(parts[0].record_count, Some(1000));
    }

    // ---- split: offsets-aware ----

    #[test]
    fn split_offsets_aware_uses_offsets_not_target() {
        // offsets [0, 300, 700] over a 1000-byte file ⇒ windows (0,300) (300,400) (700,300).
        // target=100 is IGNORED in this branch.
        let t = task(1000, DataFileFormat::Parquet, Some(vec![0, 300, 700]));
        let parts = t.split(100).expect("split ok");
        let windows: Vec<(u64, u64)> = parts.iter().map(|p| (p.start, p.length)).collect();
        assert_eq!(windows, vec![(0, 300), (300, 400), (700, 300)]);
        // length conservation: the windows tile the whole file with no gap/overlap.
        let total: u64 = parts.iter().map(|p| p.length).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn split_non_ascending_offsets_fall_back_to_fixed_size() {
        // Offsets present but NOT strictly ascending ⇒ Java falls through to fixed-size on `target`.
        let t = task(1000, DataFileFormat::Parquet, Some(vec![0, 300, 300]));
        let parts = t.split(400).expect("split ok");
        // Fixed-size on target=400 ⇒ (0,400) (400,400) (800,200), NOT the offset windows.
        let windows: Vec<(u64, u64)> = parts.iter().map(|p| (p.start, p.length)).collect();
        assert_eq!(windows, vec![(0, 400), (400, 400), (800, 200)]);
    }

    // ---- split: fixed-size ----

    #[test]
    fn split_fixed_size_walks_the_file() {
        let t = task(1000, DataFileFormat::Parquet, None);
        let parts = t.split(400).expect("split ok");
        let windows: Vec<(u64, u64)> = parts.iter().map(|p| (p.start, p.length)).collect();
        // min(400, remaining): (0,400) (400,400) (800,200).
        assert_eq!(windows, vec![(0, 400), (400, 400), (800, 200)]);
    }

    #[test]
    fn split_fixed_size_target_larger_than_file_is_one_task() {
        let t = task(300, DataFileFormat::Parquet, None);
        let parts = t.split(1000).expect("split ok");
        assert_eq!(parts.len(), 1);
        assert_eq!((parts[0].start, parts[0].length), (0, 300));
    }

    #[test]
    fn split_sub_tasks_inherit_parent_fields_and_clear_record_count_and_offsets() {
        let mut t = task(1000, DataFileFormat::Parquet, None);
        t.deletes = vec![pos_delete(50)];
        let parts = t.split(400).expect("split ok");
        assert_eq!(parts.len(), 3);
        for p in &parts {
            // Deletes / schema / projection carried; record_count + offsets cleared on a sub-task.
            assert_eq!(p.deletes.len(), 1);
            assert_eq!(p.project_field_ids, vec![1]);
            assert_eq!(p.record_count, None);
            assert_eq!(p.split_offsets, None);
            assert_eq!(p.file_size_in_bytes, 1000);
        }
    }

    #[test]
    fn split_zero_target_is_an_error() {
        let t = task(1000, DataFileFormat::Parquet, None);
        assert!(t.split(0).is_err(), "a zero split target must be rejected");
    }

    // ---- weight ----

    #[test]
    fn weight_no_deletes_is_max_of_length_and_open_cost() {
        // length=1000, no deletes ⇒ max(1000 + 0, 1 * open_cost).
        let t = task(1000, DataFileFormat::Parquet, None);
        assert_eq!(t.weight(500), 1000, "length term dominates");
        assert_eq!(t.weight(5000), 5000, "open-file-cost floor dominates");
    }

    #[test]
    fn weight_adds_position_delete_bytes() {
        let mut t = task(1000, DataFileFormat::Parquet, None);
        t.deletes = vec![pos_delete(200), pos_delete(300)];
        // size term = 1000 + 200 + 300 = 1500; floor = (1 + 2) * 100 = 300 ⇒ max = 1500.
        assert_eq!(t.weight(100), 1500);
        // With a big open cost the floor dominates: (1 + 2) * 1000 = 3000.
        assert_eq!(t.weight(1000), 3000);
    }

    #[test]
    fn weight_dv_charges_blob_size_not_file_size() {
        let mut t = task(1000, DataFileFormat::Parquet, None);
        // DV: whole puffin file is 9_000_000 bytes but the DV blob is only 64 bytes.
        t.deletes = vec![dv_delete(9_000_000, 64)];
        // size term must use the BLOB size: 1000 + 64 = 1064 (NOT 1000 + 9_000_000).
        assert_eq!(
            t.weight(0),
            1064,
            "DV weight must use content_size_in_bytes (blob), not file size"
        );
    }

    #[test]
    fn delete_content_size_dv_vs_parquet() {
        // Parquet delete contributes its file size; a Puffin DV contributes its blob size.
        assert_eq!(pos_delete(777).content_size_in_bytes(), 777);
        assert_eq!(dv_delete(9_000_000, 64).content_size_in_bytes(), 64);
    }

    // ---- serde: the flagged-additive split_offsets field ----

    #[test]
    fn split_offsets_round_trips_and_is_absent_when_none() {
        // Present: serializes as a "split_offsets" key and round-trips.
        let mut with_offsets = task(1000, DataFileFormat::Parquet, Some(vec![0, 400, 800]));
        with_offsets.start = 0;
        let json = serde_json::to_string(&with_offsets).expect("serialize");
        assert!(
            json.contains("split_offsets"),
            "split_offsets must serialize when present"
        );
        let back: FileScanTask = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.split_offsets, Some(vec![0, 400, 800]));

        // None: the key is ABSENT (skip_serializing_if), so a pre-field serialization round-trips.
        let without = task(1000, DataFileFormat::Parquet, None);
        let json_none = serde_json::to_string(&without).expect("serialize none");
        assert!(
            !json_none.contains("split_offsets"),
            "split_offsets must be absent from the JSON when None"
        );
        let back_none: FileScanTask = serde_json::from_str(&json_none).expect("deserialize none");
        assert_eq!(back_none.split_offsets, None);
    }
}
