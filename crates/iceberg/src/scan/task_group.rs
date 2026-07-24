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

//! Scan-task GROUPS — the output of split + bin-pack planning.
//!
//! Ports Java `org.apache.iceberg.ScanTaskGroup<T>` (the generic "group of tasks read together")
//! and its `CombinedScanTask` realization for [`FileScanTask`]s (Java `BaseCombinedScanTask` /
//! `CombinedScanTask`). `TableScan::plan_tasks` produces a stream of [`CombinedScanTask`]s; each is
//! one bin emitted by the bin-packer, holding the file-scan tasks an executor should read together.

use serde::{Deserialize, Serialize};

use super::task::{FileScanTask, merge_tasks};

/// A group of scan tasks to be read together, porting Java `ScanTaskGroup<T>`.
///
/// The Java interface exposes `tasks()`, `sizeBytes()` (the combined weight), and `filesCount()`.
/// The combined [`CombinedScanTask`] implements it over [`FileScanTask`]; the trait is kept so the
/// future typed `BatchScan` surface (GAP_MATRIX row R124) and any other group realization can share
/// one shape.
pub trait ScanTaskGroup<T> {
    /// The tasks that make up this group, in packing order.
    fn tasks(&self) -> &[T];

    /// The combined size of the group in bytes — the sum of the member tasks' lengths. Mirrors
    /// Java `ScanTaskGroup.sizeBytes()`.
    fn size_bytes(&self) -> u64;

    /// The number of files (tasks) in the group. Mirrors Java `ScanTaskGroup.filesCount()`.
    fn files_count(&self) -> usize {
        self.tasks().len()
    }
}

/// A combined scan task: a bin of [`FileScanTask`]s an executor reads as one unit.
///
/// Ports Java `CombinedScanTask` / `BaseCombinedScanTask` — the element type
/// `TableScanUtil.planTasks` produces. It carries NO grouping key (Java's partition-aware
/// `planTaskGroups(groupingKeyType)` overload is out of scope), but it DOES merge adjacent
/// contiguous same-file splits: [`new`](Self::new) mirrors Java's `BaseCombinedScanTask(List)`
/// constructor, whose bytecode calls `TableScanUtil.mergeTasks` (`REF_newInvokeSpecial
/// BaseCombinedScanTask.<init>:(List)` is exactly the mapper `TableScanUtil.planTasks` runs each bin
/// through). So when the bin-packer places two contiguous byte-range splits of ONE file in the same
/// bin, they collapse into a single spanning member — matching Java's emitted `planTasks` groups.
/// See [`merge_tasks`](super::task::merge_tasks) for the semantics.
///
/// `size_bytes` is the sum of the member tasks' [`FileScanTask::length`]s (Java
/// `BaseCombinedScanTask.sizeBytes` sums `task.length()`), NOT the bin-packing weight (which adds
/// delete bytes and the open-file-cost floor); the two differ when deletes or the floor dominate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CombinedScanTask {
    /// The file-scan tasks in this combined task, in packing order.
    pub tasks: Vec<FileScanTask>,
}

impl CombinedScanTask {
    /// Builds a combined task from the bin's member tasks (the order the bin-packer emitted them),
    /// MERGING adjacent contiguous same-file splits via [`merge_tasks`](super::task::merge_tasks) —
    /// a faithful port of Java's `BaseCombinedScanTask(List)` constructor, which calls
    /// `TableScanUtil.mergeTasks` on the bin. See the type docs for why merging lives here.
    pub fn new(tasks: Vec<FileScanTask>) -> Self {
        Self {
            tasks: merge_tasks(tasks),
        }
    }

    /// The member tasks, in packing order.
    pub fn tasks(&self) -> &[FileScanTask] {
        &self.tasks
    }

    /// The combined size in bytes — the sum of the member tasks' lengths (Java
    /// `BaseCombinedScanTask.sizeBytes`). Saturating to avoid an overflow panic on adversarial
    /// inputs.
    pub fn size_bytes(&self) -> u64 {
        self.tasks
            .iter()
            .map(FileScanTask::length)
            .fold(0u64, u64::saturating_add)
    }

    /// The number of files in the group (Java `BaseCombinedScanTask.filesCount`).
    pub fn files_count(&self) -> usize {
        self.tasks.len()
    }
}

impl ScanTaskGroup<FileScanTask> for CombinedScanTask {
    fn tasks(&self) -> &[FileScanTask] {
        &self.tasks
    }

    fn size_bytes(&self) -> u64 {
        CombinedScanTask::size_bytes(self)
    }

    fn files_count(&self) -> usize {
        self.tasks.len()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, Type};

    /// A whole-file [`FileScanTask`] of `length` bytes for the group-arithmetic tests.
    fn task(length: u64) -> FileScanTask {
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
            record_count: Some(1),
            data_file_path: "memory://t/1.parquet".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema,
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets: None,
        }
    }

    /// A split sub-task over `path` covering `[start, start + length)` — the exact shape
    /// [`FileScanTask::split`](super::super::task::FileScanTask) produces (`record_count` None,
    /// `split_offsets` None). Used to pin the adjacent-split merge in [`CombinedScanTask::new`].
    fn split_task(path: &str, start: u64, length: u64) -> FileScanTask {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("schema builds"),
        );
        FileScanTask {
            file_size_in_bytes: 10_000,
            start,
            length,
            record_count: None,
            data_file_path: path.to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema,
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets: None,
        }
    }

    /// The `(path, start, length)` key of each member, in order.
    fn member_keys(group: &CombinedScanTask) -> Vec<(String, u64, u64)> {
        group
            .tasks()
            .iter()
            .map(|t| (t.data_file_path().to_string(), t.start(), t.length()))
            .collect()
    }

    #[test]
    fn new_merges_adjacent_contiguous_same_file_splits() {
        // The runner-observed bin: two CONTIGUOUS big.parquet splits (4,469) + (473,470) — 4+469==473
        // — which Java's `BaseCombinedScanTask(List)` ctor (`TableScanUtil.mergeTasks`) coalesces into
        // one span (4,939). Before this fix Rust emitted BOTH members and diverged from Java's plan.
        let group = CombinedScanTask::new(vec![
            split_task("s3://b/big.parquet", 4, 469),
            split_task("s3://b/big.parquet", 473, 470),
        ]);
        assert_eq!(
            group.tasks().len(),
            1,
            "adjacent same-file splits must merge into ONE member"
        );
        assert_eq!(
            (group.tasks()[0].start(), group.tasks()[0].length()),
            (4, 939),
            "the merged span keeps the first start (4) and sums the lengths (469+470=939)"
        );
        assert_eq!(group.size_bytes(), 939);
        assert_eq!(group.files_count(), 1);
    }

    #[test]
    fn new_merges_a_run_of_three_contiguous_splits() {
        // A run of 3 contiguous splits collapses into ONE: the merged task is itself still a split
        // (same path, extended length), so `mergeTasks` keeps folding the run.
        let group = CombinedScanTask::new(vec![
            split_task("s3://b/f.parquet", 0, 100),
            split_task("s3://b/f.parquet", 100, 50),
            split_task("s3://b/f.parquet", 150, 25),
        ]);
        assert_eq!(member_keys(&group), vec![(
            "s3://b/f.parquet".to_string(),
            0,
            175
        )]);
    }

    #[test]
    fn new_does_not_merge_non_contiguous_same_file_splits() {
        // A GAP (100..200 unread) between two same-file splits ⇒ NO merge: 0+100 != 200, so
        // `canMerge`'s `offset + len == other.start` is false. Both members survive, unchanged.
        let group = CombinedScanTask::new(vec![
            split_task("s3://b/f.parquet", 0, 100),
            split_task("s3://b/f.parquet", 200, 50),
        ]);
        assert_eq!(member_keys(&group), vec![
            ("s3://b/f.parquet".to_string(), 0, 100),
            ("s3://b/f.parquet".to_string(), 200, 50),
        ]);
    }

    #[test]
    fn new_does_not_merge_different_file_members() {
        // Numerically "contiguous" (0+100==100) but DIFFERENT files ⇒ never merge — Java compares
        // `file()` (identity), so a same-offset arithmetic coincidence across files is irrelevant.
        let group = CombinedScanTask::new(vec![
            split_task("s3://b/a.parquet", 0, 100),
            split_task("s3://b/b.parquet", 100, 50),
        ]);
        assert_eq!(member_keys(&group), vec![
            ("s3://b/a.parquet".to_string(), 0, 100),
            ("s3://b/b.parquet".to_string(), 100, 50),
        ]);
    }

    #[test]
    fn new_merges_only_adjacent_runs_and_preserves_order() {
        // [a(0,100), a(100,50), b(0,30), b(30,20), a(0,10)] ⇒ [a(0,150), b(0,50), a(0,10)]:
        // `mergeTasks` compares ONLY list-adjacent tasks, so the trailing a-split — not adjacent to
        // the first a-run — stays a separate member. Proves adjacency-only + order preservation.
        let group = CombinedScanTask::new(vec![
            split_task("s3://b/a.parquet", 0, 100),
            split_task("s3://b/a.parquet", 100, 50),
            split_task("s3://b/b.parquet", 0, 30),
            split_task("s3://b/b.parquet", 30, 20),
            split_task("s3://b/a.parquet", 0, 10),
        ]);
        assert_eq!(member_keys(&group), vec![
            ("s3://b/a.parquet".to_string(), 0, 150),
            ("s3://b/b.parquet".to_string(), 0, 50),
            ("s3://b/a.parquet".to_string(), 0, 10),
        ]);
    }

    #[test]
    fn size_bytes_sums_member_lengths_and_files_count_is_member_count() {
        let group = CombinedScanTask::new(vec![task(100), task(250), task(7)]);
        assert_eq!(group.size_bytes(), 357);
        assert_eq!(group.files_count(), 3);
        assert_eq!(group.tasks().len(), 3);
    }

    #[test]
    fn scan_task_group_trait_object_matches_inherent_methods() {
        let group = CombinedScanTask::new(vec![task(40), task(60)]);
        let as_trait: &dyn ScanTaskGroup<FileScanTask> = &group;
        // The trait surface (tasks/size_bytes/files_count) agrees with the inherent surface.
        assert_eq!(as_trait.size_bytes(), 100);
        assert_eq!(as_trait.files_count(), 2);
        assert_eq!(as_trait.tasks().len(), 2);
    }

    #[test]
    fn empty_group_has_zero_size_and_count() {
        let group = CombinedScanTask::new(vec![]);
        assert_eq!(group.size_bytes(), 0);
        assert_eq!(group.files_count(), 0);
    }
}
