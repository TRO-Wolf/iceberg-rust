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

/// Whether a data file in this format can be split into byte ranges. Ports the `splittable` flag
/// on Java `FileFormat`: Parquet, Avro and ORC are splittable, Puffin is not.
fn is_splittable(format: DataFileFormat) -> bool {
    match format {
        DataFileFormat::Parquet | DataFileFormat::Avro | DataFileFormat::Orc => true,
        DataFileFormat::Puffin => false,
    }
}

/// Whether this crate's reader honours a task's byte window. A property of the read path, so
/// narrower than [`is_splittable`].
///
/// Only the Parquet reader reads `start` and `length`. The Avro and ORC readers materialize the
/// whole file, so an N-way split of one returns N copies of every row. Java splits all three,
/// because each Java reader seeks to its own block boundaries. Declining to split costs
/// parallelism. Splitting would cost rows.
fn reader_honors_byte_range(format: DataFileFormat) -> bool {
    match format {
        DataFileFormat::Parquet => true,
        DataFileFormat::Avro | DataFileFormat::Orc | DataFileFormat::Puffin => false,
    }
}

/// Whether `values` is strictly ascending. Ports Java `ArrayUtil.isStrictlyAscending`, the gate
/// before Java trusts the split offsets. An empty array is vacuously ascending, so the split
/// caller separately requires a non-empty one.
fn is_strictly_ascending(values: &[i64]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

/// A stream of [`FileScanTask`].
pub type FileScanTaskStream = BoxStream<'static, Result<FileScanTask>>;

/// A stream of [`ChangelogScanTask`].
pub type ChangelogScanTaskStream = BoxStream<'static, Result<ChangelogScanTask>>;

/// The kind of row-level change a [`ChangelogScanTask`] produces.
///
/// Ports Java `ChangelogOperation`. The planner emits only [`Insert`](Self::Insert) and
/// [`Delete`](Self::Delete), as Java `BaseIncrementalChangelogScan` does.
///
/// `UpdateBefore` and `UpdateAfter` exist for API parity. Pairing a delete and an insert into
/// an update is an engine-side step in Java, so this library never emits them.
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
/// Ports the Java `ChangelogScanTask` sub-interface split. [`ChangelogScanTask::operation`] is
/// derived from this kind, as Java derives it per interface: `AddedRows` gives insert, and both
/// deleted kinds give delete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangelogTaskKind {
    /// The commit snapshot added the task's data file (Java `AddedRowsScanTask`). Reading it
    /// with [`added_deletes`](ChangelogScanTask::added_deletes) applied gives the net inserts.
    AddedRows,
    /// The commit snapshot removed the whole data file (Java `DeletedDataFileScanTask`).
    /// Reading it with [`existing_deletes`](ChangelogScanTask::existing_deletes) applied gives
    /// the rows this change deletes.
    DeletedDataFile,
    /// The commit snapshot added row-level delete files for this existing data file (Java
    /// `DeletedRowsScanTask`). Java core never builds this task. Only the opt-in
    /// [`with_row_level_deletes`](crate::scan::IncrementalChangelogScanBuilder::with_row_level_deletes)
    /// mode produces it.
    DeletedRows,
}

/// The row-level changes one data file carries for one snapshot in the changelog range.
///
/// Ports Java `ChangelogScanTask` and its three implementations, collapsed into one struct that
/// [`kind`](Self::kind) discriminates.
///
/// In the default Java-parity mode both delete lists stay empty, and the scan rejects a range
/// that holds row-level delete manifests. The opt-in row-level mode fills them.
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
        self.file_scan_task.data_file_path()
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

/// Serde for [`Arc<str>`] as a JSON string (byte-identical to plain [`String`]).
mod serde_arc_str {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        serializer.serialize_str(value)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<str>, D::Error>
    where D: Deserializer<'de> {
        String::deserialize(deserializer).map(Arc::from)
    }
}

/// Serde for [`Arc<[T]>`] as a JSON array (byte-identical to plain [`Vec<T>`]).
mod serde_arc_slice {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S, T>(value: &Arc<[T]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        value.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<Arc<[T]>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        Vec::<T>::deserialize(deserializer).map(Arc::from)
    }
}

/// A task to scan part of file.
///
/// Shared innards (`data_file_path`, `project_field_ids`, `predicate`, `deletes`) are
/// [`Arc`]-backed so [`FileScanTask::split`] / [`FileScanTask::sub_task`] clone cheaply
/// (pointer share). JSON serde still emits plain string/array shapes — see the
/// `serde_*` helpers — so wire format stays byte-compatible with pre-Arc tasks.
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
    ///
    /// Arc-shared across split sub-tasks; serializes as a JSON string.
    #[serde(with = "serde_arc_str")]
    pub data_file_path: Arc<str>,

    /// The format of the file to scan.
    pub data_file_format: DataFileFormat,

    /// The schema of the file to scan.
    pub schema: SchemaRef,
    /// The field ids to project.
    ///
    /// Arc-shared across split sub-tasks; serializes as a JSON array.
    #[serde(with = "serde_arc_slice")]
    pub project_field_ids: Arc<[i32]>,
    /// The residual predicate to filter rows while reading this task.
    ///
    /// Arc-shared across co-partition files (via residual memo) and split sub-tasks;
    /// serializes as the bare [`BoundPredicate`] (or absent when `None`).
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<Arc<BoundPredicate>>,

    /// The list of delete files that may need to be applied to this data file.
    ///
    /// Arc-shared across split sub-tasks; serializes as a JSON array.
    #[serde(with = "serde_arc_slice")]
    pub deletes: Arc<[FileScanTaskDeleteFile]>,

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

    /// The data file's split offsets, from
    /// [`DataFile::split_offsets`](crate::spec::DataFile::split_offsets).
    ///
    /// A whole-file task carries this so the split layer can cut the file at its row-group
    /// boundaries. A sub-task resets it to `None`, because a sub-task window is not the row-group
    /// grid, so it must not re-split. `None` serializes as an absent key.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_offsets: Option<Vec<i64>>,

    /// V3 row lineage: start of the row-id range this data file owns, inherited at manifest read
    /// (Java `ManifestReader.idAssigner`). `None` for V1/V2 and for a V3 file with no assigned
    /// range, which projects `_row_id` as all-NULL. Survives a split.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_row_id: Option<i64>,

    /// V3 row lineage: the data file's sequence number, the fallback for
    /// `_last_updated_sequence_number`. `None` when unknown. Survives a split.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_sequence_number: Option<i64>,
}

impl FileScanTask {
    /// Returns the data file path of this file scan task.
    pub fn data_file_path(&self) -> &str {
        self.data_file_path.as_ref()
    }

    /// Returns the byte length of the data file region this task reads.
    pub fn length(&self) -> u64 {
        self.length
    }

    /// Returns the start byte offset of the data file region this task reads.
    pub fn start(&self) -> u64 {
        self.start
    }

    /// Splits this task into target-sized sub-tasks. Ports Java
    /// `BaseContentScanTask.split(long targetSplitSize)`.
    ///
    /// | Branch | Condition | Result |
    /// |---|---|---|
    /// | 1 | the format is not splittable, or the reader ignores the byte window | `[self]` |
    /// | 1a | already ranged: `start != 0` or `length != file_size_in_bytes` | `[self]` |
    /// | 1b | the legacy whole-file sentinel `length == 0` | `[self]` |
    /// | 1c | the projection includes `_pos` or `_row_id` | `[self]` |
    /// | 2 | split offsets present and strictly ascending | one sub-task per offset |
    /// | 3 | otherwise | windows of `min(target, remaining)` |
    ///
    /// Branches 1a to 1c are fork-local. Java reaches none of them, because its split product is
    /// a different type that cannot be re-split. Each one guards a reader that cannot honour a
    /// window. See [`reader_honors_byte_range`] and the branch comments in the body.
    ///
    /// A sub-task keeps every parent field but `start` and `length`. It clears `record_count`
    /// and `split_offsets`.
    ///
    /// # Errors
    ///
    /// `target` must be `> 0`, as Java `TableScanUtil.splitFiles` requires.
    pub fn split(&self, target: u64) -> Result<Vec<FileScanTask>> {
        if target == 0 {
            return Err(Error::new(ErrorKind::DataInvalid, "Split size must be > 0"));
        }

        // (1) Non-splittable format. Avro and ORC take this branch too: this crate's readers
        // ignore the byte window, so a split would duplicate every row once per sub-task.
        if !is_splittable(self.data_file_format) || !reader_honors_byte_range(self.data_file_format)
        {
            return Ok(vec![self.clone()]);
        }

        // (1a) An already-ranged parent stays whole: `start != 0`, or `length` is not the file
        // size. Branches (2) and (3) measure the byte space from zero and size it from the file,
        // not from the parent's window. So a re-split reads bytes the parent never owned, and
        // drops the tail it did own.
        //
        // Java cannot reach this shape, because its split product is not re-splittable. This
        // crate uses one public type for both, so a caller can build it.
        if self.start != 0 || self.length != self.file_size_in_bytes {
            return Ok(vec![self.clone()]);
        }

        // (1b) The legacy whole-file sentinel `length == 0` stays whole. Every read path here
        // spells "whole file" as `start == 0` with that sentinel or the file size, so `split`
        // must agree. The fixed-size walk loops `while remaining > 0`, so a zero length emits no
        // sub-tasks and `plan_tasks` drops the file with no error.
        //
        // Branch (1a) already returned for a ranged task, so only a zero-size file reaches here.
        // The branch stays to state the sentinel rule where a reader looks for it.
        if self.length == 0 {
            return Ok(vec![self.clone()]);
        }

        // (1c) A task projecting `_pos` or `_row_id` stays whole. Both read paths decode the file
        // in physical order and number rows from zero, so `arrow::reader` admits only a whole-file
        // task and fails every other window closed. A split would manufacture exactly the shape
        // the reader refuses, turning rows into errors.
        //
        // `TableScan::plan_tasks` carries the same rule at its call site. That guard looks
        // redundant but is not: `split` is `pub` and reachable without either caller.
        if self.project_field_ids.iter().any(|&id| {
            id == crate::metadata_columns::RESERVED_FIELD_ID_POS
                || id == crate::metadata_columns::RESERVED_FIELD_ID_ROW_ID
        }) {
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

    /// The offsets-aware split (branch 2). Each sub-task starts at `offsets[i]`, the last one
    /// running to `self.length`.
    ///
    /// # Notes
    ///
    /// The last window is correct only because branch (1a) guarantees the parent is the whole
    /// file. On a partial parent the manifest offsets could run past the parent's end.
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
    ///
    /// The walk anchors at `self.start`, not at a literal `0`. Branch (1a) makes the two equal
    /// here, so the anchor keeps the window correct if that guard ever moves.
    fn split_fixed_size(&self, target: u64) -> Vec<FileScanTask> {
        let mut sub_tasks = Vec::new();
        let mut offset = self.start;
        let mut remaining = self.length;
        while remaining > 0 {
            let len = target.min(remaining);
            sub_tasks.push(self.sub_task(offset, len));
            offset += len;
            remaining -= len;
        }
        sub_tasks
    }

    /// Builds one sub-task covering `[start, start + length)`.
    ///
    /// The shared fields are Arc-cloned, so a sub-task shares them without a deep copy. Only
    /// `start` and `length` change. `record_count` and `split_offsets` are cleared, because
    /// neither describes a sub-window.
    fn sub_task(&self, start: u64, length: u64) -> FileScanTask {
        FileScanTask {
            file_size_in_bytes: self.file_size_in_bytes,
            start,
            length,
            record_count: None,
            data_file_path: Arc::clone(&self.data_file_path),
            data_file_format: self.data_file_format,
            schema: Arc::clone(&self.schema),
            project_field_ids: Arc::clone(&self.project_field_ids),
            predicate: self.predicate.as_ref().map(Arc::clone),
            deletes: Arc::clone(&self.deletes),
            partition: self.partition.clone(),
            partition_spec: self.partition_spec.as_ref().map(Arc::clone),
            name_mapping: self.name_mapping.as_ref().map(Arc::clone),
            case_sensitive: self.case_sensitive,
            split_offsets: None,
            // Row lineage survives the split: a sub-task is a window of the same data file, so
            // it keeps that file's row-id range and sequence number.
            first_row_id: self.first_row_id,
            file_sequence_number: self.file_sequence_number,
        }
    }

    /// Whether this task can merge with `other` into one contiguous span. Ports Java
    /// `SplitScanTask.canMerge`: same data file, and exactly contiguous.
    ///
    /// Java compares no delete, residual or schema field, because splits of one file share those
    /// by construction. A gap or an overlap never merges.
    fn can_merge(&self, other: &FileScanTask) -> bool {
        self.data_file_path == other.data_file_path
            && self.start.checked_add(self.length) == Some(other.start)
    }

    /// Merge this task with a contiguous same-file `other`. Ports Java `SplitScanTask.merge`:
    /// keep this task's `start` and every non-window field, and sum the lengths.
    ///
    /// Only [`can_merge`](Self::can_merge) callers reach here, so the span is exactly
    /// `[start, other.start + other.length)`. The add saturates, so an adversarial pair cannot
    /// panic. Java would wrap.
    fn merge_with(&self, other: &FileScanTask) -> FileScanTask {
        let mut merged = self.clone();
        merged.length = self.length.saturating_add(other.length);
        merged
    }

    /// The bin-packing weight of this task. Ports Java `TableScanUtil`:
    /// `max(length + contentSizeInBytes(deletes), (1 + deletes.len()) * openFileCost)`.
    ///
    /// The floor term charges one open per data file, plus one per delete. The arithmetic
    /// saturates, so an adversarial size cannot panic. Java would wrap.
    pub(crate) fn weight(&self, open_file_cost: u64) -> u64 {
        let delete_bytes: u64 = self
            .deletes
            .iter()
            .map(FileScanTaskDeleteFile::content_size_in_bytes)
            .fold(0u64, u64::saturating_add);
        let size_term = self.length.saturating_add(delete_bytes);

        // `u64::try_from` is exact on a 64-bit target. The `u64::MAX` fallback keeps the floor
        // saturating rather than panicking.
        let num_deletes = u64::try_from(self.deletes.len()).unwrap_or(u64::MAX);
        let opens: u64 = 1u64.saturating_add(num_deletes);
        let cost_term = opens.saturating_mul(open_file_cost);

        size_term.max(cost_term)
    }

    /// Returns the project field id of this file scan task.
    pub fn project_field_ids(&self) -> &[i32] {
        self.project_field_ids.as_ref()
    }

    /// Returns the predicate of this file scan task.
    pub fn predicate(&self) -> Option<&BoundPredicate> {
        self.predicate.as_deref()
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

/// Merge adjacent contiguous same-file split tasks. Ports Java `TableScanUtil.mergeTasks`.
///
/// Java walks the list once with an accumulator. Order is preserved, and only adjacent tasks are
/// compared. A merged task stays mergeable, so a run of three or more collapses into one. Java
/// runs this over each bin of `planTasks`, from the `BaseCombinedScanTask` list constructor.
///
/// [`FileScanTask::can_merge`] subsumes Java's `MergeableScanTask` type guard, because only a
/// split of one file can produce a same-file contiguous pair here.
pub(crate) fn merge_tasks(tasks: Vec<FileScanTask>) -> Vec<FileScanTask> {
    let mut merged: Vec<FileScanTask> = Vec::with_capacity(tasks.len());
    let mut prev: Option<FileScanTask> = None;
    for current in tasks {
        match prev {
            // Contiguous same-file split ⇒ fold into the accumulator (the merged task stays
            // mergeable, so a longer run keeps collapsing).
            Some(p) if p.can_merge(&current) => {
                prev = Some(p.merge_with(&current));
            }
            // A non-mergeable neighbour ⇒ flush the run and restart at `current`.
            Some(p) => {
                merged.push(p);
                prev = Some(current);
            }
            None => {
                prev = Some(current);
            }
        }
    }
    if let Some(p) = prev {
        merged.push(p);
    }
    merged
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
    /// The content size of this delete file for bin-packing weight. Ports Java
    /// `ScanTaskUtil.contentSizeInBytes`.
    ///
    /// A Puffin deletion vector contributes `content_size_in_bytes`, the blob length, not the
    /// whole Puffin file size. Any other delete file contributes `file_size_in_bytes`.
    ///
    /// A malformed or negative `content_size_in_bytes` falls back to `file_size_in_bytes`, so
    /// adversarial input keeps the weight finite instead of panicking.
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
    use crate::metadata_columns::{RESERVED_FIELD_ID_FILE, RESERVED_FIELD_ID_POS};
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
            data_file_path: Arc::from("memory://t/data/1.parquet"),
            data_file_format: format,
            schema,
            project_field_ids: Arc::from(vec![1]),
            predicate: None,
            deletes: Arc::from(vec![]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets,
            first_row_id: None,
            file_sequence_number: None,
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

    /// A split sub-task over `path` covering `[start, start + length)`, for the merge predicate
    /// tests. Only `data_file_path` / `start` / `length` / `deletes` are load-bearing here.
    fn split_like(path: &str, start: u64, length: u64) -> FileScanTask {
        let mut t = task(length, DataFileFormat::Parquet, None);
        t.data_file_path = Arc::from(path);
        t.start = start;
        t.record_count = None;
        t
    }

    // ---- merge: canMerge / merge (Java SplitScanTask) ----

    #[test]
    fn can_merge_requires_same_file_and_exact_contiguity() {
        let a = split_like("f.parquet", 0, 100);
        assert!(
            a.can_merge(&split_like("f.parquet", 100, 50)),
            "same file, offset+len == next start ⇒ mergeable"
        );
        assert!(
            !a.can_merge(&split_like("f.parquet", 200, 50)),
            "same file but a GAP (100 != 200) ⇒ not mergeable"
        );
        assert!(
            !a.can_merge(&split_like("f.parquet", 50, 50)),
            "same file but OVERLAP (100 != 50) ⇒ not mergeable (contiguity is exact ==)"
        );
        assert!(
            !a.can_merge(&split_like("g.parquet", 100, 50)),
            "different file ⇒ never mergeable even when arithmetically contiguous"
        );
    }

    #[test]
    fn merge_with_sums_length_keeps_start_and_carries_parent_fields() {
        let mut a = split_like("f.parquet", 0, 100);
        a.deletes = Arc::from(vec![pos_delete(50)]);
        let merged = a.merge_with(&split_like("f.parquet", 100, 50));
        assert_eq!(
            (merged.start, merged.length),
            (0, 150),
            "start stays at self's; length is self+other"
        );
        assert_eq!(merged.data_file_path.as_ref(), "f.parquet");
        assert_eq!(
            merged.deletes.len(),
            1,
            "the merged task carries self's deletes (same file ⇒ same delete set)"
        );
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

    /// AVRO and ORC are splittable per the Java `FileFormat` table, but THIS crate's readers
    /// materialize whole files and never read `start`/`length`
    /// ([`reader_honors_byte_range`]), so a split would hand every sub-task the whole file: an
    /// N-way split silently returns N copies, with no error. The planner must therefore decline
    /// to split them — through BOTH the fixed-size branch and the offsets-aware branch, which is
    /// the one a real Avro/ORC manifest entry with `split_offsets` would take.
    #[test]
    fn split_declines_formats_whose_reader_ignores_byte_ranges() {
        for format in [DataFileFormat::Avro, DataFileFormat::Orc] {
            // Non-vacuity: the same geometry MUST split for a format whose reader honours the
            // window, otherwise this test would pass on a `split` that never splits anything.
            for offsets in [None, Some(vec![0i64, 300, 700])] {
                let parquet = task(1000, DataFileFormat::Parquet, offsets.clone());
                assert!(
                    parquet.split(100).expect("parquet split ok").len() > 1,
                    "fixture is non-discriminating: parquet with offsets {offsets:?} must split"
                );

                let t = task(1000, format, offsets.clone());
                let parts = t.split(100).expect("split ok");
                assert_eq!(
                    parts.len(),
                    1,
                    "{format:?} with offsets {offsets:?} must not be split into byte windows its \
                     reader cannot honour"
                );
                assert_eq!(
                    (parts[0].start, parts[0].length),
                    (0, 1000),
                    "{format:?} passthrough must cover the whole file"
                );
            }
        }
    }

    /// A whole-file task spelled with the legacy `length == 0` sentinel must survive `split` as
    /// one task, never as an empty Vec.
    ///
    /// Mutation this catches: dropping the `length == 0` guard. The fixed-size branch then
    /// returns zero sub-tasks, `plan_tasks` drops the file, and nothing errors. The readers
    /// accept `start == 0, length == 0` as whole-file, so an empty Vec is silent row loss.
    #[test]
    fn split_whole_file_length_sentinel_is_one_task_not_zero() {
        // The sentinel must win in BOTH the fixed-size branch and the offsets-aware branch (a
        // manifest entry can carry split offsets alongside the sentinel length).
        for offsets in [None, Some(vec![0i64, 300, 700])] {
            let mut t = task(1000, DataFileFormat::Parquet, offsets.clone());
            t.length = 0; // legacy whole-file sentinel; file_size_in_bytes stays 1000

            // Non-vacuity: the SAME geometry with a real length does split into many windows, so
            // this assertion cannot pass on a `split` that never splits anything.
            let sized = task(1000, DataFileFormat::Parquet, offsets.clone());
            assert!(
                sized.split(100).expect("sized split ok").len() > 1,
                "fixture is non-discriminating: offsets {offsets:?} must split when length > 0"
            );

            let parts = t.split(100).expect("split ok");
            assert_eq!(
                parts.len(),
                1,
                "the `length == 0` whole-file sentinel (offsets {offsets:?}) must split to ONE \
                 task; an empty Vec loses every row of the file with no error"
            );
            assert_eq!(
                (parts[0].start, parts[0].length),
                (0, 0),
                "the sentinel task must pass through verbatim, keeping the spelling the readers \
                 accept as whole-file"
            );
            assert_eq!(
                parts[0].file_size_in_bytes, 1000,
                "passthrough must not disturb the file size"
            );
        }
    }

    /// An already-ranged task (`start != 0`) must pass through `split` verbatim, never relocate
    /// to offset 0.
    ///
    /// Mutation this catches: dropping branch (1a). Both real branches measure the byte space
    /// from zero, so a parent covering `[600, 1000)` comes back as windows over `[0, 400)`. The
    /// products read bytes the parent never owned and drop the tail it did.
    #[test]
    fn split_of_an_already_ranged_task_is_a_passthrough_not_a_relocation() {
        for offsets in [None, Some(vec![0i64, 300, 700])] {
            // Non-vacuity: the SAME geometry with `start == 0` really does split into several
            // windows, so a `split` that never splits anything cannot pass this test.
            let whole = task(1000, DataFileFormat::Parquet, offsets.clone());
            assert!(
                whole.split(200).expect("whole split ok").len() > 1,
                "fixture is non-discriminating: offsets {offsets:?} must split from start 0"
            );

            let mut ranged = task(1000, DataFileFormat::Parquet, offsets.clone());
            ranged.start = 600;
            ranged.length = 400;

            let parts = ranged.split(200).expect("split ok");
            assert_eq!(
                parts.len(),
                1,
                "an already-ranged task (offsets {offsets:?}) must pass through split as ONE task"
            );
            assert_eq!(
                (parts[0].start, parts[0].length),
                (600, 400),
                "the sub-task window must stay EXACTLY the parent's; anything anchored at 0 reads \
                 bytes the parent never owned and drops the tail it did"
            );
        }
    }

    /// A partial parent (`start == 0`, `length < file_size_in_bytes`) must pass through `split`
    /// verbatim. It is ranged just as much as one with a moved left edge.
    ///
    /// Mutation this catches: dropping the `length != file_size_in_bytes` disjunct of branch
    /// (1a). The offsets-aware branch takes every window but the last from the manifest offsets,
    /// which describe the whole file, so it runs past the parent's end.
    #[test]
    fn split_of_a_partial_parent_is_a_passthrough_not_an_over_read() {
        for offsets in [None, Some(vec![0i64, 300, 700])] {
            // Non-vacuity: the SAME geometry spanning the WHOLE file really does split into
            // several windows, so a `split` that never splits anything cannot pass this test.
            let whole = task(1000, DataFileFormat::Parquet, offsets.clone());
            assert!(
                whole.split(200).expect("whole split ok").len() > 1,
                "fixture is non-discriminating: offsets {offsets:?} must split from a whole-file \
                 parent"
            );

            let mut partial = task(1000, DataFileFormat::Parquet, offsets.clone());
            partial.length = 500; // file_size_in_bytes stays 1000 ⇒ the parent owns [0, 500)

            let parts = partial.split(200).expect("split ok");
            assert_eq!(
                parts.len(),
                1,
                "a partial parent (offsets {offsets:?}) must pass through split as ONE task"
            );
            assert_eq!(
                (parts[0].start, parts[0].length),
                (0, 500),
                "the sub-task window must stay EXACTLY the parent's; the manifest offsets describe \
                 the whole FILE and would run past the parent's end"
            );
        }
    }

    /// A task projecting `_pos` must pass through `split` verbatim.
    ///
    /// Mutation this catches: dropping branch (1c). The `_pos` reader rejects every ranged task,
    /// including a `start == 0` sub-task whose length is no longer the file size. A split then
    /// turns a scan that reads every row into one that reads none and errors per sub-task.
    #[test]
    fn split_of_a_pos_projecting_task_is_a_passthrough() {
        for offsets in [None, Some(vec![0i64, 300, 700])] {
            // Non-vacuity: the SAME geometry WITHOUT `_pos` in the projection really does split.
            let plain = task(1000, DataFileFormat::Parquet, offsets.clone());
            assert!(
                plain.split(200).expect("plain split ok").len() > 1,
                "fixture is non-discriminating: offsets {offsets:?} must split without `_pos`"
            );

            let mut pos = task(1000, DataFileFormat::Parquet, offsets.clone());
            pos.project_field_ids = Arc::from(vec![1, RESERVED_FIELD_ID_POS]);

            let parts = pos.split(200).expect("split ok");
            assert_eq!(
                parts.len(),
                1,
                "a `_pos`-projecting task (offsets {offsets:?}) must pass through split as ONE task"
            );
            assert_eq!(
                (parts[0].start, parts[0].length),
                (0, 1000),
                "the passthrough must keep the whole-file spelling the `_pos` reader accepts"
            );
            assert_eq!(
                parts[0].project_field_ids.as_ref(),
                &[1, RESERVED_FIELD_ID_POS],
                "the passthrough must not disturb the projection"
            );
        }
    }

    /// Branch (1c) must fire on a projection of `_pos` alone. Every other (1c) fixture pairs the
    /// metadata id with a data column, so all of them miss this shape.
    ///
    /// `scan().select(["_pos"])` reaches it from the public builder.
    ///
    /// Mutation this catches: narrowing the guard to `project_field_ids.len() > 1 && ...`. Such
    /// a task then splits and the reader rejects every sub-task.
    #[test]
    fn split_of_a_pos_only_projection_is_a_passthrough() {
        // Non-vacuity: the same geometry with a lone DATA column really does split.
        let plain = task(1000, DataFileFormat::Parquet, None);
        assert!(
            plain.split(200).expect("plain split ok").len() > 1,
            "fixture is non-discriminating: a 1000-byte file must split at target 200"
        );

        let mut pos_only = task(1000, DataFileFormat::Parquet, None);
        pos_only.project_field_ids = Arc::from(vec![RESERVED_FIELD_ID_POS]);

        let parts = pos_only.split(200).expect("split ok");
        assert_eq!(
            parts.len(),
            1,
            "a projection of `_pos` ALONE must pass through split as ONE task; a guard that also \
             demanded a second projected column would split it and the reader would then reject \
             every sub-task"
        );
        assert_eq!(
            (parts[0].start, parts[0].length),
            (0, 1000),
            "the passthrough must keep the whole-file spelling the `_pos` reader accepts"
        );
    }

    /// Branch (1a)'s `self.start != 0` disjunct must hold on its own, at the one shape the other
    /// disjunct cannot see: a relocated left edge whose length still spans the file.
    ///
    /// Every other ranged-task fixture trips both disjuncts, so they cannot discriminate this.
    ///
    /// Mutation this catches: dropping `self.start != 0`. The mutant splits a parent owning
    /// `[600, 1600)` into three sub-tasks over `[0, 1000)`.
    #[test]
    fn split_of_a_relocated_parent_is_a_passthrough_even_when_length_spans_the_file() {
        for offsets in [None, Some(vec![0i64, 300, 700])] {
            // Non-vacuity: the SAME geometry at `start == 0` really does split.
            let whole = task(1000, DataFileFormat::Parquet, offsets.clone());
            assert!(
                whole.split(200).expect("whole split ok").len() > 1,
                "fixture is non-discriminating: offsets {offsets:?} must split from start 0"
            );

            let mut relocated = task(1000, DataFileFormat::Parquet, offsets.clone());
            relocated.start = 600; // length stays 1000 == file_size_in_bytes

            let parts = relocated.split(200).expect("split ok");
            assert_eq!(
                parts.len(),
                1,
                "a relocated parent (offsets {offsets:?}) must pass through split as ONE task even \
                 when its length still equals the file size"
            );
            assert_eq!(
                (parts[0].start, parts[0].length),
                (600, 1000),
                "the passthrough must keep the parent's own window; re-splitting would RELOCATE it \
                 to offset 0 and read bytes the parent never owned"
            );
        }
    }

    /// Branch (1b), the `length == 0` sentinel, must stay pinned at the only shape that reaches
    /// it: a `file_size_in_bytes == 0` file.
    ///
    /// Branch (1a) returns first for every other sentinel task, so no other fixture reaches (1b).
    ///
    /// Mutation this catches: corrupting the sentinel condition. The fixed-size walk then emits
    /// zero sub-tasks and `plan_tasks` reads no rows, with no error.
    #[test]
    fn split_whole_file_sentinel_on_an_empty_file_is_one_task_not_zero() {
        // Non-vacuity: `split` really does split when there are bytes to split.
        let sized = task(1000, DataFileFormat::Parquet, None);
        assert!(
            sized.split(100).expect("sized split ok").len() > 1,
            "fixture is non-discriminating: a 1000-byte file must split at target 100"
        );

        // start == 0 and length == file_size_in_bytes == 0, so branch (1a)'s inequality is FALSE
        // and only the sentinel branch can answer.
        let empty = task(0, DataFileFormat::Parquet, None);
        let parts = empty.split(100).expect("split ok");
        assert_eq!(
            parts.len(),
            1,
            "the whole-file sentinel on an empty file must split to ONE task; an empty Vec drops \
             the file from `plan_tasks` with no error"
        );
        assert_eq!((parts[0].start, parts[0].length), (0, 0));
    }

    /// A parent whose length OVERRUNS the file (`length > file_size_in_bytes`) is ranged in the
    /// same sense as a truncated one, and (1a)'s `!=` — not a `<` — is what covers it. Without this
    /// pin the inequality could be narrowed to `<` and the fixed-size walk would happily emit
    /// windows past EOF.
    #[test]
    fn split_of_an_overlong_parent_is_a_passthrough() {
        // Non-vacuity: without it, a `split` that declined EVERYTHING would satisfy the assertions
        // below. (Measured: under a `target.min(remaining)` → `remaining` mutant this test alone of
        // the four passthrough tests stayed green.)
        let sized = task(1000, DataFileFormat::Parquet, None);
        assert!(
            sized.split(200).expect("sized split ok").len() > 1,
            "fixture is non-discriminating: a 1000-byte file must split at target 200"
        );

        let mut overlong = task(1000, DataFileFormat::Parquet, None);
        overlong.length = 1500; // file_size_in_bytes stays 1000

        let parts = overlong.split(200).expect("split ok");
        assert_eq!(
            parts.len(),
            1,
            "a parent whose window overruns the file must pass through as ONE task"
        );
        assert_eq!((parts[0].start, parts[0].length), (0, 1500));
    }

    /// Branch (1c) declines `_pos` SPECIFICALLY, not metadata columns in general.
    ///
    /// The Parquet path re-supplies `_file` as a per-file constant, so a byte window serves it
    /// exactly as the whole file does. `_pos` is the one metadata column whose value depends on
    /// the window. This test claims nothing about `_spec_id`, `_partition` or `_deleted`, which
    /// this path never serves.
    ///
    /// Mutation this catches: widening the guard to any metadata field id.
    #[test]
    fn split_declines_pos_specifically_not_every_metadata_column() {
        let mut file_col = task(1000, DataFileFormat::Parquet, None);
        file_col.project_field_ids = Arc::from(vec![1, RESERVED_FIELD_ID_FILE]);

        assert!(
            file_col.split(200).expect("split ok").len() > 1,
            "`_file` is served correctly over a ranged window — only `_pos` needs whole-file \
             ordinals, so only `_pos` may suppress the split"
        );
    }

    /// The Java `FileFormat` splittable table and this crate's READ-path predicate are separate
    /// facts, and the divergence is deliberate: `is_splittable` ports Java verbatim, while
    /// `reader_honors_byte_range` is what actually gates `split`. Neither is observable on its own
    /// through `split` (each format is masked by the other predicate), so assert the tables
    /// directly — GAP_MATRIX row R148 claims parity for the Java table specifically.
    #[test]
    fn format_predicate_tables_match_java_and_the_read_path() {
        assert!(is_splittable(DataFileFormat::Parquet));
        assert!(is_splittable(DataFileFormat::Avro));
        assert!(is_splittable(DataFileFormat::Orc));
        assert!(
            !is_splittable(DataFileFormat::Puffin),
            "Java `FileFormat.PUFFIN` is not splittable"
        );

        assert!(reader_honors_byte_range(DataFileFormat::Parquet));
        for format in [
            DataFileFormat::Avro,
            DataFileFormat::Orc,
            DataFileFormat::Puffin,
        ] {
            assert!(
                !reader_honors_byte_range(format),
                "{format:?} is materialized whole-file by this crate's reader"
            );
        }
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

    /// A single split offset takes the offsets-aware branch, because `split`'s gate is
    /// `!offsets.is_empty()`. Java answers the same way, and loses the same leading bytes.
    ///
    /// `TableScan::expand_within_file_parallel_tasks` deliberately uses a different gate,
    /// `offsets.len() > 1`. It is a fork-local optimisation with no Java counterpart, and it must
    /// never make `to_arrow()` disagree with a whole-file read.
    ///
    /// Do not "fix" either gate to match the other.
    #[test]
    fn split_single_offset_takes_the_offsets_aware_branch() {
        let t = task(1000, DataFileFormat::Parquet, Some(vec![300]));
        let parts = t.split(400).expect("split ok");
        let windows: Vec<(u64, u64)> = parts.iter().map(|p| (p.start, p.length)).collect();
        assert_eq!(
            windows,
            vec![(300, 700)],
            "one offset ⇒ ONE offsets-aware window running to the end of the file; the fixed-size \
             fallback would have emitted (0,400) (400,400) (800,200) instead"
        );
    }

    /// A corrupt manifest can carry strictly-ascending but NEGATIVE split offsets (the field is
    /// `i64` in the spec). Those pass the ascending gate and reach the `u64` conversion, which must
    /// fail with a typed error rather than clamping — a clamped `0` would silently merge two
    /// windows and misreport where the row group begins.
    #[test]
    fn split_negative_offsets_are_a_typed_error_not_a_clamp() {
        let t = task(1000, DataFileFormat::Parquet, Some(vec![-10, 0, 100]));
        let err = t
            .split(100)
            .expect_err("negative split offsets must not be accepted");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string()
                .contains("split offset must be non-negative"),
            "the error must name the offending offset, got: {err}"
        );
    }

    /// A hostile manifest whose last offset runs past the end of the file. The last window's
    /// length underflows `u64`, so the `saturating_sub` yields an empty trailing window.
    ///
    /// Strict ascent guarantees `end > start` for every earlier window, so the last offset is the
    /// only reachable shape.
    ///
    /// Mutation this catches: a `wrapping_sub`, which hands the reader a ~2^64 length.
    #[test]
    fn split_offsets_running_past_eof_yield_an_empty_trailing_window_not_an_underflow() {
        let t = task(1000, DataFileFormat::Parquet, Some(vec![0, 300, 2000]));
        let parts = t.split(100).expect("split ok");
        let windows: Vec<(u64, u64)> = parts.iter().map(|p| (p.start, p.length)).collect();
        assert_eq!(
            windows,
            vec![(0, 300), (300, 1700), (2000, 0)],
            "the past-EOF trailing offset must saturate to a ZERO-length window; wrapping would \
             give it a ~2^64 length"
        );
    }

    /// An empty offsets vector must fall through to the fixed-size walk. An empty Avro array
    /// decodes to `Some(vec![])`, so a manifest can reach this shape.
    ///
    /// Mutation this catches: dropping the `!offsets.is_empty()` conjunct. Empty offsets are
    /// vacuously ascending, so `split_at_offsets` returns an empty Vec and `plan_tasks` drops the
    /// file with no error.
    #[test]
    fn split_empty_offsets_fall_back_to_fixed_size_and_never_drop_the_file() {
        let t = task(1000, DataFileFormat::Parquet, Some(vec![]));
        let parts = t.split(200).expect("split ok");
        assert_eq!(
            parts.len(),
            5,
            "empty offsets must take the FIXED-SIZE branch — the offsets-aware branch returns an \
             empty Vec, which drops the file out of `plan_tasks` with no error at all"
        );
        let windows: Vec<(u64, u64)> = parts.iter().map(|p| (p.start, p.length)).collect();
        assert_eq!(
            windows,
            vec![(0, 200), (200, 200), (400, 200), (600, 200), (800, 200)],
            "the fallback must tile the whole file at the target size"
        );
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

    /// The fixed-size walk's loop bound is load-bearing at `remaining == 1`. This pins both
    /// halves: a one-byte parent, and a walk whose last window is one byte.
    ///
    /// The other fixed-size fixtures cannot see it. One has a 200-byte last window, the other
    /// never loops twice.
    ///
    /// Mutation this catches: `while remaining > 1`. The walk then drops a one-byte file.
    #[test]
    fn split_fixed_size_emits_the_final_one_byte_window_and_never_an_empty_vec() {
        // Half 1 — a walk whose LAST window is exactly 1 byte: 1000 at target 333.
        // This assertion is also the NON-VACUITY guard for half 2: a `split` that declined
        // everything would return one task here, not four.
        let t = task(1000, DataFileFormat::Parquet, None);
        let windows: Vec<(u64, u64)> = t
            .split(333)
            .expect("split ok")
            .iter()
            .map(|p| (p.start, p.length))
            .collect();
        assert_eq!(
            windows,
            vec![(0, 333), (333, 333), (666, 333), (999, 1)],
            "the walk must emit its final ONE-byte window; dropping it loses the last byte of the \
             file — and with it any row whose data lives there — with no error"
        );

        // Half 2 — a parent that IS one byte: exactly one `(0, 1)` window, never an empty Vec.
        let one = task(1, DataFileFormat::Parquet, None);
        let parts = one.split(4).expect("split ok");
        assert_eq!(
            parts.len(),
            1,
            "a one-byte parent must split to ONE task; an empty Vec drops the file out of \
             `plan_tasks` entirely, with no error anywhere"
        );
        assert_eq!(
            (parts[0].start, parts[0].length),
            (0, 1),
            "the single window must cover the parent exactly"
        );
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
        t.deletes = Arc::from(vec![pos_delete(50)]);
        let parts = t.split(400).expect("split ok");
        assert_eq!(parts.len(), 3);
        for p in &parts {
            // Deletes / schema / projection carried; record_count + offsets cleared on a sub-task.
            assert_eq!(p.deletes.len(), 1);
            assert_eq!(p.project_field_ids.as_ref(), &[1][..]);
            assert_eq!(p.record_count, None);
            assert_eq!(p.split_offsets, None);
            assert_eq!(p.file_size_in_bytes, 1000);
        }
    }

    /// FK2.1 pin: split sub-tasks Arc-share path / projection / deletes (and residual when set).
    #[test]
    fn split_sub_tasks_arc_share_path_projection_deletes_and_predicate() {
        let mut t = task(1000, DataFileFormat::Parquet, None);
        t.deletes = Arc::from(vec![pos_delete(50), pos_delete(75)]);
        t.predicate = Some(Arc::new(BoundPredicate::AlwaysTrue));
        let parts = t.split(400).expect("split ok");
        assert_eq!(
            parts.len(),
            3,
            "fixed-size split of 1000/400 yields 3 sub-tasks"
        );
        for p in &parts {
            assert!(
                Arc::ptr_eq(&p.data_file_path, &t.data_file_path),
                "sub-task path must Arc-share parent"
            );
            assert!(
                Arc::ptr_eq(&p.project_field_ids, &t.project_field_ids),
                "sub-task project_field_ids must Arc-share parent"
            );
            assert!(
                Arc::ptr_eq(&p.deletes, &t.deletes),
                "sub-task deletes must Arc-share parent"
            );
            assert!(
                Arc::ptr_eq(
                    p.predicate.as_ref().expect("predicate set"),
                    t.predicate.as_ref().expect("predicate set")
                ),
                "sub-task residual must Arc-share parent"
            );
            // Window fields are the only ones that change.
            assert_eq!(p.record_count, None);
            assert_eq!(p.split_offsets, None);
        }
        // Sibling sub-tasks share with each other too (same parent Arc).
        assert!(Arc::ptr_eq(&parts[0].deletes, &parts[1].deletes));
        assert!(Arc::ptr_eq(
            &parts[0].project_field_ids,
            &parts[2].project_field_ids
        ));
        assert!(Arc::ptr_eq(
            &parts[0].data_file_path,
            &parts[2].data_file_path
        ));
    }

    /// Offsets-aware split must Arc-share the same way as fixed-size.
    #[test]
    fn split_offsets_aware_sub_tasks_arc_share_innards() {
        let mut t = task(1000, DataFileFormat::Parquet, Some(vec![0, 300, 700]));
        t.deletes = Arc::from(vec![pos_delete(50)]);
        t.predicate = Some(Arc::new(BoundPredicate::AlwaysFalse));
        let parts = t.split(1).expect("offsets-aware ignores target");
        assert_eq!(parts.len(), 3);
        for p in &parts {
            assert!(Arc::ptr_eq(&p.deletes, &t.deletes));
            assert!(Arc::ptr_eq(&p.project_field_ids, &t.project_field_ids));
            assert!(Arc::ptr_eq(&p.data_file_path, &t.data_file_path));
            assert!(Arc::ptr_eq(
                p.predicate.as_ref().expect("set"),
                t.predicate.as_ref().expect("set")
            ));
            assert_eq!(p.split_offsets, None);
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
        t.deletes = Arc::from(vec![pos_delete(200), pos_delete(300)]);
        // size term = 1000 + 200 + 300 = 1500; floor = (1 + 2) * 100 = 300 ⇒ max = 1500.
        assert_eq!(t.weight(100), 1500);
        // With a big open cost the floor dominates: (1 + 2) * 1000 = 3000.
        assert_eq!(t.weight(1000), 3000);
    }

    #[test]
    fn weight_dv_charges_blob_size_not_file_size() {
        let mut t = task(1000, DataFileFormat::Parquet, None);
        // DV: whole puffin file is 9_000_000 bytes but the DV blob is only 64 bytes.
        t.deletes = Arc::from(vec![dv_delete(9_000_000, 64)]);
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

    /// Row lineage survives a split, unlike `split_offsets`, which is cleared.
    #[test]
    fn row_lineage_survives_a_split_while_split_offsets_is_cleared() {
        let mut whole = task(1000, DataFileFormat::Parquet, Some(vec![0, 400, 800]));
        whole.start = 0;
        whole.first_row_id = Some(4_242);
        whole.file_sequence_number = Some(9);

        let parts = whole.split(400).expect("split");
        assert!(parts.len() > 1, "fixture precondition: the task must split");
        for part in &parts {
            assert_eq!(
                part.first_row_id,
                Some(4_242),
                "every sub-task keeps the file's row-id range"
            );
            assert_eq!(
                part.file_sequence_number,
                Some(9),
                "every sub-task keeps the file's sequence number"
            );
            assert_eq!(
                part.split_offsets, None,
                "control: split_offsets IS cleared — the two fields behave differently on purpose"
            );
        }
    }

    /// Both fields are public on an engine-serialized struct: present when set, absent when
    /// `None` so a serialization predating them still round-trips.
    #[test]
    fn row_lineage_fields_round_trip_and_are_absent_when_none() {
        let mut with_lineage = task(1000, DataFileFormat::Parquet, None);
        with_lineage.start = 0;
        with_lineage.first_row_id = Some(4_242);
        with_lineage.file_sequence_number = Some(9);
        let json = serde_json::to_string(&with_lineage).expect("serialize");
        assert!(json.contains("first_row_id"), "must serialize when present");
        assert!(json.contains("file_sequence_number"));
        let back: FileScanTask = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.first_row_id, Some(4_242));
        assert_eq!(back.file_sequence_number, Some(9));

        let without = task(1000, DataFileFormat::Parquet, None);
        let json_none = serde_json::to_string(&without).expect("serialize none");
        assert!(
            !json_none.contains("first_row_id"),
            "absent from the JSON when None, so a task predating the field round-trips"
        );
        assert!(!json_none.contains("file_sequence_number"));
        let back_none: FileScanTask = serde_json::from_str(&json_none).expect("deserialize none");
        assert_eq!(back_none.first_row_id, None);
        assert_eq!(back_none.file_sequence_number, None);
    }

    /// `_row_id` suppresses splitting exactly as `_pos` does (branch 1c); without it the planner
    /// hands the reader ranged sub-tasks it then refuses one by one.
    #[test]
    fn projecting_row_id_suppresses_splitting() {
        let mut whole = task(1000, DataFileFormat::Parquet, Some(vec![0, 400, 800]));
        whole.start = 0;
        whole.project_field_ids =
            Arc::from(vec![1, crate::metadata_columns::RESERVED_FIELD_ID_ROW_ID]);

        let parts = whole.split(400).expect("split");
        assert_eq!(
            parts.len(),
            1,
            "a `_row_id` projection must yield ONE whole-file task, not ranged sub-tasks"
        );

        // Control: the same task WITHOUT `_row_id` really does split, so the assertion above is
        // discriminating rather than vacuous.
        let mut splittable = task(1000, DataFileFormat::Parquet, Some(vec![0, 400, 800]));
        splittable.start = 0;
        assert!(splittable.split(400).expect("split").len() > 1);
    }

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

    /// FK2.1 STOP bar: Arc wrappers must not change the JSON shape of FileScanTask fields
    /// that engines serialize (path string, projection array, deletes array, residual).
    #[test]
    fn arc_fields_serialize_as_plain_string_and_arrays() {
        let mut t = task(1000, DataFileFormat::Parquet, None);
        t.deletes = Arc::from(vec![pos_delete(50)]);
        t.predicate = Some(Arc::new(BoundPredicate::AlwaysTrue));
        t.project_field_ids = Arc::from(vec![1, 2, 3]);

        let json = serde_json::to_string(&t).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse json");

        // Path is a JSON string, not an object.
        assert!(
            v["data_file_path"].is_string(),
            "data_file_path must serialize as a JSON string, got {json}"
        );
        assert_eq!(
            v["data_file_path"].as_str(),
            Some("memory://t/data/1.parquet")
        );

        // Projection / deletes are JSON arrays (not Arc-tagged objects).
        assert!(
            v["project_field_ids"].is_array(),
            "project_field_ids must serialize as a JSON array"
        );
        assert_eq!(v["project_field_ids"], serde_json::json!([1, 2, 3]));
        assert!(
            v["deletes"].is_array(),
            "deletes must serialize as a JSON array"
        );
        assert_eq!(v["deletes"].as_array().expect("arr").len(), 1);

        // Residual serializes as the bare BoundPredicate variant name, not Arc-wrapped.
        assert_eq!(v["predicate"], serde_json::json!("AlwaysTrue"));

        // Round-trip restores values (new Arcs, same content).
        let back: FileScanTask = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.data_file_path.as_ref(), t.data_file_path.as_ref());
        assert_eq!(
            back.project_field_ids.as_ref(),
            t.project_field_ids.as_ref()
        );
        assert_eq!(back.deletes.as_ref(), t.deletes.as_ref());
        assert_eq!(back.predicate.as_deref(), t.predicate.as_deref());
    }

    /// STOP bar: frozen golden JSON (pre-Arc field shapes) must match
    /// exactly for a representative task — not just Value-level type checks.
    #[test]
    fn arc_fields_json_matches_pre_arc_golden_bytes() {
        // A task with no residual / no deletes / no split_offsets — the common engine
        // wire shape. Field order follows the struct declaration (serde_json preserves it).
        let t = task(1000, DataFileFormat::Parquet, None);
        let json = serde_json::to_string(&t).expect("serialize");
        // Golden: plain string path, plain array projection, plain array deletes, no
        // predicate key (None), no split_offsets key (None). Arc must not wrap any of these.
        let golden = concat!(
            r#"{"file_size_in_bytes":1000,"start":0,"length":1000,"record_count":1000,"#,
            r#""data_file_path":"memory://t/data/1.parquet","data_file_format":"parquet","#,
            r#""schema":{"#,
        );
        assert!(
            json.starts_with(golden),
            "JSON prefix must match pre-Arc shape; got {json}"
        );
        assert!(
            json.contains(r#""project_field_ids":[1]"#),
            "project_field_ids must be a bare JSON array [1], got {json}"
        );
        assert!(
            json.contains(r#""deletes":[]"#),
            "deletes must be a bare JSON array [], got {json}"
        );
        assert!(
            !json.contains("predicate"),
            "None residual must omit the predicate key, got {json}"
        );
        assert!(
            !json.contains("split_offsets"),
            "None split_offsets must omit the key, got {json}"
        );
        // No Arc-tagged / newtype object wrappers around shared fields.
        assert!(
            !json.contains("Arc") && !json.contains("\"ptr\""),
            "JSON must not expose Arc internals, got {json}"
        );
    }
}
