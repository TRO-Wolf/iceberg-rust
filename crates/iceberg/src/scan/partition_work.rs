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

//! Multi-partition scan work assignment (DataFusion `UnknownPartitioning(N)` seam).
//!
//! Pipeline (total order — do not reorder):
//!
//! ```text
//! plan_tasks groups → strip empties → fixed-T round-robin → PartitionWork[N]
//! ```
//!
//! Cover + disjoint are at **task-range grain** `(data_file_path, start, length)`, not file path
//! alone. Streaming a [`PartitionWork`] uses the legacy single-stream reader path (no within-file
//! parallel expand) so multi-partition SELECT does not inherit WG2 composition risk.

use std::collections::HashSet;

use futures::{TryStreamExt, stream};

use super::task::FileScanTask;
use super::task_group::CombinedScanTask;
use super::{ArrowRecordBatchStream, TableScan};
use crate::arrow::ArrowReaderBuilder;
use crate::io::FileIO;
use crate::{Error, ErrorKind, Result};

/// Identity of a planned [`FileScanTask`] at byte-range grain.
///
/// File-path-only equality is **wrong**: splits of one file may land in different partitions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileScanTaskId {
    /// Data file path.
    pub data_file_path: String,
    /// Start offset within the file.
    pub start: u64,
    /// Byte length of the range.
    pub length: u64,
}

impl FileScanTaskId {
    /// Build an identity from a planned task.
    pub fn from_task(task: &FileScanTask) -> Self {
        Self {
            data_file_path: task.data_file_path.clone(),
            start: task.start,
            length: task.length,
        }
    }
}

/// How residual row filtering was applied when the work was planned.
///
/// Scan mode is a **required** input to planning: there is no mode-free residual default that
/// serves both SELECT (residual on) and COW MERGE (file-prune-only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanFilterMode {
    /// Plan-time prune + residual row filter on surviving tasks (normal SELECT).
    Residual,
    /// Plan-time partition + metrics prune only; tasks carry no residual (COW MERGE target).
    FilePruneOnly,
}

/// Ordered scan work assigned to one DataFusion output partition.
///
/// Owns the constituent [`CombinedScanTask`] groups (planner order preserved within the partition).
/// MoR delete **apply** scope remains per-task (`task.deletes()` / R117); this type does not fuse
/// a super-group apply map.
#[derive(Debug, Clone)]
pub struct PartitionWork {
    snapshot_id: i64,
    filter_mode: ScanFilterMode,
    /// Non-empty groups assigned to this partition (empty only for the empty-table N=1 case).
    groups: Vec<CombinedScanTask>,
}

impl PartitionWork {
    /// Snapshot id frozen at plan time for this work.
    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    /// Filter mode used when the work was planned.
    pub fn filter_mode(&self) -> ScanFilterMode {
        self.filter_mode
    }

    /// Combined-scan-task groups in assignment order.
    pub fn groups(&self) -> &[CombinedScanTask] {
        &self.groups
    }

    /// Flat iterator over all member tasks.
    pub fn tasks(&self) -> impl Iterator<Item = &FileScanTask> {
        self.groups.iter().flat_map(|g| g.tasks().iter())
    }

    /// Task-range identities for structural cover / isolation pins.
    pub fn task_ids(&self) -> HashSet<FileScanTaskId> {
        self.tasks().map(FileScanTaskId::from_task).collect()
    }

    /// True when this partition has no tasks (empty-table degenerate stream).
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty() || self.tasks().next().is_none()
    }
}

/// Strip empty groups, then assign into `t` (or fewer) partitions.
///
/// - `|G| == 0` after strip → `N = 1`, single empty work (empty table).
/// - `|G| ≤ t` → one partition per group, `N = |G|`.
/// - `|G| > t` → fixed-`t` round-robin of whole groups (`group j → bin j % t`), `N = t`.
///
/// Never drops a group. Capacity re-bin-packing is **out of scope** for v1.
///
/// `t` is clamped to at least 1.
pub fn assign_partition_work(
    snapshot_id: i64,
    filter_mode: ScanFilterMode,
    groups: Vec<CombinedScanTask>,
    t: usize,
) -> Vec<PartitionWork> {
    let t = t.max(1);
    let non_empty: Vec<CombinedScanTask> = groups
        .into_iter()
        .filter(|g| !g.tasks().is_empty())
        .collect();

    if non_empty.is_empty() {
        return vec![PartitionWork {
            snapshot_id,
            filter_mode,
            groups: Vec::new(),
        }];
    }

    if non_empty.len() <= t {
        return non_empty
            .into_iter()
            .map(|g| PartitionWork {
                snapshot_id,
                filter_mode,
                groups: vec![g],
            })
            .collect();
    }

    // Fixed-T round-robin of whole non-empty groups.
    let mut bins: Vec<Vec<CombinedScanTask>> = (0..t).map(|_| Vec::new()).collect();
    for (j, group) in non_empty.into_iter().enumerate() {
        bins[j % t].push(group);
    }

    bins.into_iter()
        .map(|groups| PartitionWork {
            snapshot_id,
            filter_mode,
            groups,
        })
        .collect()
}

/// Structural cover + disjoint check at task-range grain.
///
/// Returns `Ok(())` when `⋃ Work(i) = planned` and pairwise intersections are empty.
pub fn assert_partition_cover(
    planned: &HashSet<FileScanTaskId>,
    partitions: &[PartitionWork],
) -> Result<()> {
    let mut seen: HashSet<FileScanTaskId> = HashSet::new();
    for (i, work) in partitions.iter().enumerate() {
        for id in work.task_ids() {
            if !seen.insert(id.clone()) {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "partition work not disjoint: task {:?} appears in more than one partition (seen at partition {i})",
                        id
                    ),
                ));
            }
        }
    }
    if seen != *planned {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "partition work does not cover planned tasks: planned={} assigned={}",
                planned.len(),
                seen.len()
            ),
        ));
    }
    Ok(())
}

/// Stream Arrow batches for one [`PartitionWork`] using the **legacy single-stream** path.
///
/// Does **not** expand within-file `split_offsets` (WG2 composition deferred; disclosed in G1 PR).
/// MoR deletes are applied by [`crate::arrow::ArrowReader`] with per-task `task.deletes()` scope
/// (R117). Shared delete **load** is internal to a single reader instance.
///
/// # Snapshot freeze
///
/// Callers must ensure `work.snapshot_id()` matches the snapshot used to plan the embedded tasks.
/// This helper does not re-plan; a mismatched catalog snapshot is a caller / provider error.
pub fn stream_partition_work(
    file_io: FileIO,
    work: &PartitionWork,
    data_file_concurrency: usize,
    batch_size: Option<usize>,
    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
) -> Result<ArrowRecordBatchStream> {
    let tasks: Vec<FileScanTask> = work.tasks().cloned().collect();
    let task_stream = Box::pin(stream::iter(tasks.into_iter().map(Ok)));

    let mut builder = ArrowReaderBuilder::new(file_io)
        .with_data_file_concurrency_limit(data_file_concurrency.max(1))
        .with_row_group_filtering_enabled(row_group_filtering_enabled)
        .with_row_selection_enabled(row_selection_enabled);
    if let Some(bs) = batch_size {
        builder = builder.with_batch_size(bs.max(1));
    }
    builder.build().read(task_stream)
}

/// Plan [`PartitionWork`] for a built [`TableScan`] under target partition budget `t`.
///
/// Resolves the concrete snapshot id from the scan (empty table → synthetic id `0` with one
/// empty partition). Filter mode is inferred from whether residual application is enabled on
/// the plan context (`FilePruneOnly` when residuals are off).
pub async fn plan_partition_work_from_scan(
    scan: &TableScan,
    t: usize,
) -> Result<Vec<PartitionWork>> {
    let (snapshot_id, filter_mode) = match scan.plan_context_snapshot_and_mode() {
        Some(v) => v,
        None => {
            // No snapshot → empty table: N=1 empty stream.
            return Ok(assign_partition_work(
                0,
                ScanFilterMode::Residual,
                Vec::new(),
                t,
            ));
        }
    };

    let groups: Vec<CombinedScanTask> = scan.plan_tasks().await?.try_collect().await?;
    Ok(assign_partition_work(snapshot_id, filter_mode, groups, t))
}

impl TableScan {
    /// Snapshot id + filter mode used for multi-partition assignment, when a snapshot exists.
    pub(crate) fn plan_context_snapshot_and_mode(&self) -> Option<(i64, ScanFilterMode)> {
        let ctx = self.plan_context.as_ref()?;
        let mode = if ctx.apply_residual_filter {
            ScanFilterMode::Residual
        } else {
            ScanFilterMode::FilePruneOnly
        };
        Some((ctx.snapshot.snapshot_id(), mode))
    }

    /// Public helper: plan multi-partition work under budget `t` (see
    /// [`plan_partition_work_from_scan`]).
    pub async fn plan_partition_work(&self, t: usize) -> Result<Vec<PartitionWork>> {
        plan_partition_work_from_scan(self, t).await
    }

    /// Stream one [`PartitionWork`] with this scan's FileIO and reader knobs (legacy single-stream).
    pub fn stream_partition_work(&self, work: &PartitionWork) -> Result<ArrowRecordBatchStream> {
        stream_partition_work(
            self.file_io.clone(),
            work,
            self.concurrency_limit_data_files,
            self.batch_size,
            self.row_group_filtering_enabled,
            self.row_selection_enabled,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, Type};

    fn dummy_task(path: &str, start: u64, length: u64) -> FileScanTask {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                ))])
                .build()
                .expect("schema"),
        );
        FileScanTask {
            file_size_in_bytes: start + length,
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

    fn group(tasks: Vec<FileScanTask>) -> CombinedScanTask {
        CombinedScanTask::new(tasks)
    }

    #[test]
    fn assign_empty_yields_single_empty_partition() {
        let parts = assign_partition_work(42, ScanFilterMode::Residual, vec![], 4);
        assert_eq!(parts.len(), 1);
        assert!(parts[0].is_empty());
        assert_eq!(parts[0].snapshot_id(), 42);
    }

    #[test]
    fn assign_le_t_one_group_per_partition() {
        let g0 = group(vec![dummy_task("a.parquet", 0, 100)]);
        let g1 = group(vec![dummy_task("b.parquet", 0, 100)]);
        let parts = assign_partition_work(1, ScanFilterMode::Residual, vec![g0, g1], 4);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].tasks().count(), 1);
        assert_eq!(parts[1].tasks().count(), 1);
    }

    #[test]
    fn assign_gt_t_round_robin_never_drops() {
        let groups: Vec<_> = (0..5)
            .map(|i| group(vec![dummy_task(&format!("{i}.parquet"), 0, 50)]))
            .collect();
        let parts = assign_partition_work(7, ScanFilterMode::FilePruneOnly, groups, 2);
        assert_eq!(parts.len(), 2);
        // 5 groups → bins of 3 and 2
        let n0 = parts[0].groups().len();
        let n1 = parts[1].groups().len();
        assert_eq!(n0 + n1, 5);
        assert_eq!(n0, 3);
        assert_eq!(n1, 2);

        let planned: HashSet<_> = parts.iter().flat_map(|p| p.task_ids()).collect();
        assert_eq!(planned.len(), 5);
        assert_partition_cover(&planned, &parts).expect("cover");
    }

    #[test]
    fn cover_detects_overlap() {
        let t = dummy_task("x.parquet", 0, 10);
        let g = group(vec![t.clone()]);
        let parts = vec![
            PartitionWork {
                snapshot_id: 1,
                filter_mode: ScanFilterMode::Residual,
                groups: vec![g.clone()],
            },
            PartitionWork {
                snapshot_id: 1,
                filter_mode: ScanFilterMode::Residual,
                groups: vec![g],
            },
        ];
        let planned = parts[0].task_ids();
        assert!(assert_partition_cover(&planned, &parts).is_err());
    }

    #[test]
    fn strip_empty_groups() {
        let empty = CombinedScanTask::new(vec![]);
        let full = group(vec![dummy_task("a.parquet", 0, 1)]);
        let parts = assign_partition_work(1, ScanFilterMode::Residual, vec![empty, full], 2);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].tasks().count(), 1);
    }

    #[test]
    fn t_clamped_to_at_least_one() {
        let g = group(vec![dummy_task("a.parquet", 0, 1)]);
        let parts = assign_partition_work(1, ScanFilterMode::Residual, vec![g], 0);
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn task_id_is_range_grain_not_path_only() {
        let a = FileScanTaskId::from_task(&dummy_task("f.parquet", 0, 50));
        let b = FileScanTaskId::from_task(&dummy_task("f.parquet", 50, 50));
        assert_ne!(a, b, "splits of same path are distinct identities");
    }

    /// Pin 7 (DV multi-group) is DOCUMENT-ONLY for night 1: multi-group concurrent
    /// deletion-vector apply under multi-partition execute is an untested surface.
    /// Pos + eq deletes remain mandatory pins. Follow-up: multi-group DV pin.
    #[test]
    fn pin7_dv_multi_group_documented_untested() {
        let surface = "multi-group DV apply under multi-partition";
        assert!(
            surface.contains("DV"),
            "document-only: {surface} is untested in v1 — follow-up queue item required"
        );
    }
}
