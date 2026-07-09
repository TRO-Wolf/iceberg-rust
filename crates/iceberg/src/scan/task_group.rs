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

use super::task::FileScanTask;

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
/// `TableScanUtil.planTasks` produces. It is the lowest-level data group: it carries NO grouping
/// key and does NOT merge tasks (Java's plain `CombinedScanTask` path sets no grouping key and
/// performs no `mergeTasks` — that is the partition-aware `planTaskGroups(groupingKeyType)`
/// overload, which is out of scope for this unit).
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
    /// Builds a combined task from the bin's member tasks (the order the bin-packer emitted them).
    pub fn new(tasks: Vec<FileScanTask>) -> Self {
        Self { tasks }
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
