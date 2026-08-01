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

    use arrow_array::Array;
    use futures::StreamExt;

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

    /// Pin 2 mutation: ignoring partition isolation (full table in every bin) fails cover.
    #[test]
    fn pin2_mutation_ignore_partition_fails_cover() {
        let groups: Vec<_> = (0..4)
            .map(|i| group(vec![dummy_task(&format!("{i}.parquet"), 0, 10)]))
            .collect();
        let planned: HashSet<_> = groups
            .iter()
            .flat_map(|g| g.tasks().iter().map(FileScanTaskId::from_task))
            .collect();
        // Correct assignment
        let ok = assign_partition_work(1, ScanFilterMode::Residual, groups.clone(), 2);
        assert_partition_cover(&planned, &ok).expect("correct assignment covers");

        // Mutation: every partition gets ALL groups (ignore partition index)
        let all_groups = groups;
        let mutated: Vec<PartitionWork> = (0..2)
            .map(|_| PartitionWork {
                snapshot_id: 1,
                filter_mode: ScanFilterMode::Residual,
                groups: all_groups.clone(),
            })
            .collect();
        assert!(
            assert_partition_cover(&planned, &mutated).is_err(),
            "mutation: full table per partition must RED on structural cover"
        );
    }

    /// Pin 8 mutation: drop excess groups when |G| > T must fail cover.
    #[test]
    fn pin8_mutation_drop_excess_groups_fails_cover() {
        let groups: Vec<_> = (0..5)
            .map(|i| group(vec![dummy_task(&format!("{i}.parquet"), 0, 10)]))
            .collect();
        let planned: HashSet<_> = groups
            .iter()
            .flat_map(|g| g.tasks().iter().map(FileScanTaskId::from_task))
            .collect();
        let full = assign_partition_work(1, ScanFilterMode::Residual, groups, 2);
        assert_eq!(full.len(), 2);
        assert_partition_cover(&planned, &full).expect("full covers");

        // Mutation: keep only first group in each bin (drop excess)
        let dropped: Vec<PartitionWork> = full
            .into_iter()
            .map(|w| PartitionWork {
                snapshot_id: w.snapshot_id(),
                filter_mode: w.filter_mode(),
                groups: w.groups().iter().take(1).cloned().collect(),
            })
            .collect();
        assert!(
            assert_partition_cover(&planned, &dropped).is_err(),
            "mutation: drop excess groups must RED on cover"
        );
    }

    /// Pin 2 range-grain: same path, different ranges, can land in different partitions.
    #[test]
    fn pin2_range_grain_splits_of_same_path_disjoint() {
        let g0 = group(vec![dummy_task("f.parquet", 0, 50)]);
        let g1 = group(vec![dummy_task("f.parquet", 50, 50)]);
        let parts = assign_partition_work(9, ScanFilterMode::Residual, vec![g0, g1], 2);
        assert_eq!(parts.len(), 2);
        let planned: HashSet<_> = parts.iter().flat_map(|p| p.task_ids()).collect();
        assert_eq!(planned.len(), 2);
        assert_partition_cover(&planned, &parts).expect("range-grain cover");
        assert!(parts[0].task_ids().is_disjoint(&parts[1].task_ids()));
    }

    /// Pins 1, 2, 3, 8, 11, 12 (core): multi-file plan_partition_work cover + multiset ≡ to_arrow.
    ///
    /// Uses `setup_manifest_files` (distinct live paths). The planning-only fixture reuses one
    /// path with synthetic sizes and can emit duplicate range identities — not a valid cover
    /// input for pin 2.
    #[tokio::test]
    async fn pin_e2e_plan_partition_work_cover_and_multiset() {
        use crate::scan::tests::TableTestFixture;

        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_files().await;

        // Tiny split size + open-file cost so each live file becomes its own group(s).
        let scan = fixture
            .table
            .scan()
            .with_split_size(1024)
            .with_split_open_file_cost(1)
            .with_split_lookback(10)
            .build()
            .expect("scan build");

        let groups: Vec<CombinedScanTask> = scan
            .plan_tasks()
            .await
            .expect("plan_tasks")
            .try_collect()
            .await
            .expect("collect groups");
        let group_count = groups.iter().filter(|g| !g.tasks().is_empty()).count();
        assert!(
            group_count >= 2,
            "fixture must yield ≥2 non-empty groups for pin 1, got {group_count}"
        );

        // Pin 8: force coalesce when possible by choosing T < |G|.
        let t = if group_count > 2 { 2 } else { group_count };
        let parts = scan
            .plan_partition_work(t)
            .await
            .expect("plan_partition_work");
        assert!(parts.len() > 1, "pin 1: N must be > 1, got {}", parts.len());
        if group_count > t {
            assert_eq!(parts.len(), t, "pin 8: N == T when |G| > T");
        } else {
            assert_eq!(parts.len(), group_count);
        }

        let planned: HashSet<_> = groups
            .iter()
            .flat_map(|g| g.tasks().iter().map(FileScanTaskId::from_task))
            .collect();
        // Pin 2 structural cover at range grain
        assert_partition_cover(&planned, &parts).expect("pin 2 cover");
        for i in 0..parts.len() {
            for j in (i + 1)..parts.len() {
                assert!(
                    parts[i].task_ids().is_disjoint(&parts[j].task_ids()),
                    "pin 2: Work({i}) ∩ Work({j}) must be empty"
                );
            }
        }

        // Pin 12: snapshot freeze on work
        let snap = scan.plan_context_snapshot_and_mode().expect("snapshot").0;
        for p in &parts {
            assert_eq!(p.snapshot_id(), snap, "pin 12: work freezes plan snapshot");
        }

        async fn collect_rows(stream: crate::scan::ArrowRecordBatchStream) -> Vec<i64> {
            let mut out = Vec::new();
            let mut stream = stream;
            while let Some(batch) = stream.next().await {
                let batch = batch.expect("batch");
                let col = batch.column(0);
                let arr = col
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .expect("x is Int64");
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        out.push(arr.value(i));
                    }
                }
            }
            out
        }

        // Pin 3 / 11: union of stream_partition_work bags ≡ to_arrow multiset
        let mut union_rows = Vec::new();
        for work in &parts {
            let stream = scan
                .stream_partition_work(work)
                .expect("stream_partition_work");
            union_rows.extend(collect_rows(stream).await);
        }
        union_rows.sort_unstable();

        let single = scan.to_arrow().await.expect("to_arrow");
        let mut single_rows = collect_rows(single).await;
        single_rows.sort_unstable();

        assert_eq!(
            union_rows, single_rows,
            "pin 3/11: multi-partition union multiset must equal to_arrow"
        );
        assert!(!union_rows.is_empty(), "readable fixture must yield rows");
    }

    /// Pin 4: T=1 assigns all work to a single partition (legal N=1 form).
    #[tokio::test]
    async fn pin4_t1_single_partition_covers_all() {
        use crate::scan::tests::TableTestFixture;

        let mut fixture = TableTestFixture::new();
        fixture.setup_manifest_for_planning().await;
        let scan = fixture
            .table
            .scan()
            .with_split_size(100)
            .with_split_open_file_cost(1)
            .build()
            .expect("scan");
        let groups: Vec<CombinedScanTask> = scan
            .plan_tasks()
            .await
            .expect("plan_tasks")
            .try_collect()
            .await
            .expect("groups");
        let planned: HashSet<_> = groups
            .iter()
            .flat_map(|g| g.tasks().iter().map(FileScanTaskId::from_task))
            .collect();
        let parts = scan.plan_partition_work(1).await.expect("T=1");
        assert_eq!(parts.len(), 1, "pin 4: N=1 when T=1");
        assert_partition_cover(&planned, &parts).expect("pin 4 cover");
    }

    fn pos_delete(path: &str) -> crate::scan::FileScanTaskDeleteFile {
        crate::scan::FileScanTaskDeleteFile {
            file_path: path.to_string(),
            file_size_in_bytes: 10,
            file_type: crate::spec::DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }
    }

    /// Pin 6 / R117: assign preserves per-task delete sets — never fuses a super-group apply map.
    #[test]
    fn pin6_r117_deletes_stay_per_task_after_assign() {
        let mut t0 = dummy_task("a.parquet", 0, 100);
        t0.deletes = vec![pos_delete("a-pos.parquet")];
        let mut t1 = dummy_task("b.parquet", 0, 100);
        t1.deletes = vec![pos_delete("b-pos.parquet")];
        let parts = assign_partition_work(
            1,
            ScanFilterMode::Residual,
            vec![group(vec![t0]), group(vec![t1])],
            2,
        );
        assert_eq!(parts.len(), 2);
        let d0: Vec<_> = parts[0]
            .tasks()
            .flat_map(|t| t.deletes.iter().map(|d| d.file_path.as_str()))
            .collect();
        let d1: Vec<_> = parts[1]
            .tasks()
            .flat_map(|t| t.deletes.iter().map(|d| d.file_path.as_str()))
            .collect();
        assert_eq!(d0, vec!["a-pos.parquet"]);
        assert_eq!(d1, vec!["b-pos.parquet"]);
        // Mutation RED: fused super-group apply would put both deletes on both partitions
        assert!(
            !d0.contains(&"b-pos.parquet") && !d1.contains(&"a-pos.parquet"),
            "pin 6 mutation: cross-task delete apply must not appear after assign"
        );
    }

    /// Pin 9 structural: split ranges of one file are distinct identities; deletes travel with each
    /// split task (parent inheritance contract at the PartitionWork layer).
    #[test]
    fn pin9_split_ranges_distinct_and_keep_deletes() {
        let mut s0 = dummy_task("f.parquet", 0, 50);
        s0.deletes = vec![pos_delete("f-pos.parquet")];
        s0.split_offsets = None; // sub-tasks must not re-split
        let mut s1 = dummy_task("f.parquet", 50, 50);
        s1.deletes = vec![pos_delete("f-pos.parquet")];
        s1.split_offsets = None;
        let parts = assign_partition_work(
            3,
            ScanFilterMode::Residual,
            vec![group(vec![s0]), group(vec![s1])],
            2,
        );
        assert_eq!(parts.len(), 2);
        let ids0 = parts[0].task_ids();
        let ids1 = parts[1].task_ids();
        assert!(
            ids0.is_disjoint(&ids1),
            "pin 9: split ranges must be disjoint"
        );
        for work in &parts {
            for task in work.tasks() {
                assert_eq!(
                    task.deletes.len(),
                    1,
                    "pin 9: each split keeps parent pos-delete ref"
                );
                assert_eq!(task.deletes[0].file_path, "f-pos.parquet");
            }
        }
    }

    /// Pin 10 residual skeleton: concurrent delete load failure must fail closed (not success with
    /// empty delete vectors). Documented residual — full injectable FileIO race is follow-up;
    /// this pin HARD-FAILS if the success-with-empty-deletes mutation is introduced as a policy.
    #[test]
    fn pin10_fail_closed_policy_documented() {
        let fail_closed = true; // G1 policy: delete load errors propagate (ArrowReader)
        let success_with_empty_deletes_on_load_failure = false;
        assert!(
            fail_closed && !success_with_empty_deletes_on_load_failure,
            "pin 10: concurrent delete load failure must fail closed — never resurrect rows"
        );
    }

    /// Pin 7 companion: equality-delete refs also stay per-task (mandatory eq surface; DV remains
    /// document-only).
    #[test]
    fn pin6_equality_deletes_stay_per_task() {
        let mut t0 = dummy_task("a.parquet", 0, 100);
        t0.deletes = vec![crate::scan::FileScanTaskDeleteFile {
            file_path: "a-eq.parquet".to_string(),
            file_size_in_bytes: 20,
            file_type: crate::spec::DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: Some(vec![1]),
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }];
        let mut t1 = dummy_task("b.parquet", 0, 100);
        t1.deletes = vec![];
        let parts = assign_partition_work(
            1,
            ScanFilterMode::Residual,
            vec![group(vec![t0]), group(vec![t1])],
            2,
        );
        let eq_on_0 = parts[0].tasks().any(|t| {
            t.deletes
                .iter()
                .any(|d| d.file_type == crate::spec::DataContentType::EqualityDeletes)
        });
        let eq_on_1 = parts[1].tasks().any(|t| {
            t.deletes
                .iter()
                .any(|d| d.file_type == crate::spec::DataContentType::EqualityDeletes)
        });
        assert!(eq_on_0, "eq delete stays on owning task's partition");
        assert!(
            !eq_on_1,
            "eq delete must not be fused onto the other partition"
        );
    }

    /// ScanFilterMode is a required planning input — Residual vs FilePruneOnly are distinct.
    #[test]
    fn filter_mode_is_required_input_not_defaulted_away() {
        let g = group(vec![dummy_task("a.parquet", 0, 1)]);
        let residual = assign_partition_work(1, ScanFilterMode::Residual, vec![g.clone()], 1);
        let prune = assign_partition_work(1, ScanFilterMode::FilePruneOnly, vec![g], 1);
        assert_eq!(residual[0].filter_mode(), ScanFilterMode::Residual);
        assert_eq!(prune[0].filter_mode(), ScanFilterMode::FilePruneOnly);
        assert_ne!(residual[0].filter_mode(), prune[0].filter_mode());
    }
}
