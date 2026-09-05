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

//! A faithful port of Java `org.apache.iceberg.util.BinPacking.PackingIterable` (1.10.0).
//!
//! Re-verified against the live 1.10.0 `iceberg-core` jar bytecode (`BinPacking$PackingIterator`,
//! `BinPacking$Bin`). This is the bin-packing core that `TableScanUtil.planTasks` drives with
//! `largestBinFirst = true` to group split [`FileScanTask`](super::FileScanTask)s into
//! [`CombinedScanTask`](super::CombinedScanTask) bins of roughly the split target size.
//!
//! The algorithm (Java `PackingIterator.next`, decoded):
//!
//! * Pull input items in order. For each item compute `w = weight(item)`.
//! * `findBin(w)`: iterate the OPEN bins in INSERTION ORDER (a FIFO `Deque`) and return the FIRST
//!   bin that `canAdd(w)` — `binWeight + w <= target` (`<=`, so a single oversized item still gets
//!   its own bin). If one is found, add the item to it and continue (no group emitted yet).
//! * Otherwise open a `newBin()`, add the item, append it to the bin deque (`addLast`). If the
//!   number of open bins now EXCEEDS `lookback`, evict EXACTLY ONE and emit it:
//!     * `largestBinFirst = true` (our path): evict the LARGEST-weight bin
//!       (`Collections.max(bins, by weight)`, removed by identity — ties resolve to the
//!       FIRST-inserted max in FIFO iteration order);
//!     * `largestBinFirst = false`: evict the first bin (`removeFirst`).
//! * When the input is exhausted, DRAIN the remaining open bins in FIFO order (`removeFirst`),
//!   emitting each as a group.
//!
//! The iterator is lazy (it pulls from the input only as groups are demanded) and self-contained so
//! it is unit-testable independent of the scan plumbing.

use std::collections::VecDeque;
use std::sync::Arc;

use super::task::FileScanTask;
use super::task_group::CombinedScanTask;
use crate::Result;
use crate::metadata_columns::{RESERVED_FIELD_ID_POS, RESERVED_FIELD_ID_ROW_ID};
use crate::spec::DataFileFormat;

/// One open bin: the items packed into it plus their running weight total. Mirrors Java
/// `BinPacking$Bin` (`binWeight` + `items`), minus the `targetWeight` field — the target is held
/// once on the [`PackingIterator`] and passed to [`Bin::can_add`].
#[derive(Debug)]
struct Bin<T> {
    items: Vec<T>,
    bin_weight: u64,
}

impl<T> Bin<T> {
    fn new() -> Self {
        Self {
            items: Vec::new(),
            bin_weight: 0,
        }
    }

    /// Java `Bin.canAdd`: `binWeight + weight <= target`. Saturating add so an adversarial weight
    /// near `u64::MAX` cannot panic (Java's `long` would wrap; saturation only ever makes `canAdd`
    /// MORE conservative — it can never falsely admit).
    fn can_add(&self, weight: u64, target: u64) -> bool {
        self.bin_weight.saturating_add(weight) <= target
    }

    /// Java `Bin.add`: append the item and accumulate its weight.
    fn add(&mut self, item: T, weight: u64) {
        self.bin_weight = self.bin_weight.saturating_add(weight);
        self.items.push(item);
    }

    fn weight(&self) -> u64 {
        self.bin_weight
    }
}

/// A lazy bin-packing iterator over `items`, porting Java `BinPacking$PackingIterator`.
///
/// `weight_fn` maps an item to its weight (Java's `Function<T, Long>`); `target` is the per-bin
/// weight budget; `lookback` is the maximum number of simultaneously-open bins; `largest_bin_first`
/// selects the eviction policy (`true` for the `planTasks` path). Each `next()` yields the items of
/// one emitted bin, in the order they were packed.
pub(crate) struct PackingIterator<T, I, F>
where
    I: Iterator<Item = T>,
    F: Fn(&T) -> u64,
{
    items: I,
    target: u64,
    lookback: usize,
    largest_bin_first: bool,
    weight_fn: F,
    /// Open bins, in INSERTION ORDER (FIFO) — the order Java's `Deque` iterates and `removeFirst`
    /// drains.
    bins: VecDeque<Bin<T>>,
}

impl<T, I, F> PackingIterator<T, I, F>
where
    I: Iterator<Item = T>,
    F: Fn(&T) -> u64,
{
    /// Builds the iterator. `lookback` is taken as a `usize` (Java's positive `int`); the
    /// `> 0` precondition is enforced by the caller (`TableScanUtil.validatePlanningArguments`).
    pub(crate) fn new(
        items: I,
        target: u64,
        lookback: usize,
        largest_bin_first: bool,
        weight_fn: F,
    ) -> Self {
        Self {
            items,
            target,
            lookback,
            largest_bin_first,
            weight_fn,
            bins: VecDeque::new(),
        }
    }

    /// Java `PackingIterator.findBin`: the index of the FIRST open bin (insertion order) that can
    /// admit `weight`, or `None`.
    fn find_bin(&self, weight: u64) -> Option<usize> {
        self.bins
            .iter()
            .position(|bin| bin.can_add(weight, self.target))
    }

    /// Java `removeLargestBin`: the index of the largest-weight open bin, ties resolving to the
    /// FIRST such bin in FIFO order. `Collections.max` keeps the first maximum it encounters (it
    /// only replaces the candidate on a STRICT `>`), and the deque iterates in insertion order, so
    /// the earliest-inserted max wins — which `.position(... == max)` reproduces.
    fn largest_bin_index(&self) -> Option<usize> {
        let max_weight = self.bins.iter().map(Bin::weight).max()?;
        self.bins.iter().position(|bin| bin.weight() == max_weight)
    }
}

impl<T, I, F> Iterator for PackingIterator<T, I, F>
where
    I: Iterator<Item = T>,
    F: Fn(&T) -> u64,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        // Pull input items until one triggers an eviction (which emits a group) or the input ends.
        loop {
            match self.items.next() {
                Some(item) => {
                    let weight = (self.weight_fn)(&item);
                    match self.find_bin(weight) {
                        // Found an open bin with room: add (no emit), keep pulling.
                        Some(idx) => {
                            self.bins[idx].add(item, weight);
                        }
                        // No room anywhere: open a new bin, add, append to the deque.
                        None => {
                            let mut bin = Bin::new();
                            bin.add(item, weight);
                            self.bins.push_back(bin);

                            // Over the lookback budget ⇒ evict exactly one bin and emit it.
                            if self.bins.len() > self.lookback {
                                let evict_idx = if self.largest_bin_first {
                                    self.largest_bin_index()
                                } else {
                                    Some(0)
                                };
                                if let Some(idx) = evict_idx {
                                    let evicted = self
                                        .bins
                                        .remove(idx)
                                        .expect("evicted bin index is in range");
                                    return Some(evicted.items);
                                }
                            }
                        }
                    }
                }
                // Input exhausted: drain remaining open bins FIFO.
                None => {
                    return self.bins.pop_front().map(|bin| bin.items);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity weight: each item weighs its own value. Lets the tests reason about packing in
    /// plain integers without the FileScanTask weight machinery.
    fn pack(items: Vec<u64>, target: u64, lookback: usize, largest_first: bool) -> Vec<Vec<u64>> {
        PackingIterator::new(items.into_iter(), target, lookback, largest_first, |w| *w).collect()
    }

    #[test]
    fn single_oversized_item_gets_its_own_bin() {
        // `canAdd` uses `<=`, so an item EQUAL to the target fits one bin; a STRICTLY larger one
        // still gets its own bin (a fresh bin's `0 + w <= target` is false, but it is added anyway).
        let groups = pack(vec![200], 100, 10, true);
        assert_eq!(
            groups,
            vec![vec![200]],
            "an oversized lone item is its own group"
        );
    }

    #[test]
    fn items_below_target_pack_into_one_bin() {
        // 30 + 30 + 30 = 90 <= 100 ⇒ one bin; the 4th 30 would overflow (120 > 100).
        let groups = pack(vec![30, 30, 30], 100, 10, true);
        assert_eq!(groups, vec![vec![30, 30, 30]]);
    }

    #[test]
    fn canadd_is_inclusive_at_target() {
        // 100 exactly fills the bin; a second item must open a new bin.
        let groups = pack(vec![60, 40, 1], 100, 10, true);
        // 60 -> bin0(60); 40 -> bin0(100, canAdd 60+40<=100 true); 1 -> bin0 canAdd 100+1<=100 FALSE
        // -> new bin1(1). Drain FIFO: [60,40] then [1].
        assert_eq!(groups, vec![vec![60, 40], vec![1]]);
    }

    #[test]
    fn largest_bin_first_eviction_order() {
        // target=100, lookback=1 ⇒ at most ONE open bin; opening a 2nd evicts the LARGEST by weight.
        // Sequence: 90 -> bin0(90). 80 -> no room (90+80>100) -> new bin1(80); now 2>1, evict
        // largest = bin0(90) -> emit [90]. 70 -> bin1 canAdd 80+70>100 false -> new bin2(70); 2>1,
        // evict largest = bin1(80) -> emit [80]. Input ends -> drain [70].
        let groups = pack(vec![90, 80, 70], 100, 1, true);
        assert_eq!(groups, vec![vec![90], vec![80], vec![70]]);
    }

    #[test]
    fn largest_bin_first_picks_the_heaviest_open_bin() {
        // target=100, lookback=2. Pack so two bins are open with different weights, then force an
        // eviction and assert the HEAVIER bin is evicted (not the oldest — that is the FIFO policy).
        // 50 -> bin0(50). 90 -> bin0 canAdd 50+90>100 false -> new bin1(90). 60 -> bin0 canAdd
        // 50+60>100 false, bin1 90+60>100 false -> new bin2(60); now 3>2, evict largest = bin1(90)
        // -> emit [90]. Input ends -> drain FIFO: bin0[50], bin2[60].
        let groups = pack(vec![50, 90, 60], 100, 2, true);
        assert_eq!(groups, vec![vec![90], vec![50], vec![60]]);
    }

    #[test]
    fn fifo_eviction_differs_from_largest_first() {
        // SAME input as `largest_bin_first_picks_the_heaviest_open_bin` but FIFO (largest_first=false):
        // the eviction removes the OLDEST bin (bin0[50]) instead of the heaviest (bin1[90]).
        let groups = pack(vec![50, 90, 60], 100, 2, false);
        // 50->bin0; 90->bin1; 60-> new bin2; 3>2 evict FIRST = bin0[50]; drain bin1[90], bin2[60].
        assert_eq!(groups, vec![vec![50], vec![90], vec![60]]);
    }

    #[test]
    fn largest_first_tie_breaks_to_first_inserted() {
        // Two equal-weight open bins; the evicted one must be the FIRST inserted (Collections.max
        // keeps the first max). target=100, lookback=2.
        // 70 -> bin0(70). 70 -> bin0 canAdd 70+70>100 false -> new bin1(70). 70 -> both full ->
        // new bin2(70); 3>2, evict largest; bin0 and bin1 tie at 70 -> evict FIRST = bin0 -> [70](bin0).
        // drain bin1[70], bin2[70]. All groups equal [70] but the ORDER pins first-inserted eviction:
        // we tag items by index to make the tie-break observable.
        let groups = PackingIterator::new(
            vec![(0u32, 70u64), (1, 70), (2, 70)].into_iter(),
            100,
            2,
            true,
            |x| x.1,
        )
        .collect::<Vec<_>>();
        // First emitted group is bin0 = item 0 (the first-inserted max), proving the tie-break.
        assert_eq!(groups[0], vec![(0, 70)]);
        assert_eq!(groups[1], vec![(1, 70)]);
        assert_eq!(groups[2], vec![(2, 70)]);
    }

    #[test]
    fn empty_input_yields_no_groups() {
        assert!(pack(vec![], 100, 10, true).is_empty());
    }

    #[test]
    fn weight_total_per_group_respects_target_except_oversized() {
        let target = 100;
        let groups = pack(vec![40, 40, 40, 200, 10, 10], target, 10, true);
        for group in &groups {
            let sum: u64 = group.iter().sum();
            // Each group is within target UNLESS it is a single oversized item.
            assert!(
                sum <= target || group.len() == 1,
                "group {group:?} sum {sum} exceeds target {target} and is not a lone oversized item"
            );
        }
        // Conservation: every input item appears exactly once across the groups.
        let mut all: Vec<u64> = groups.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, vec![10, 10, 40, 40, 40, 200]);
    }
}

pub(crate) const MIN_SPLIT_TARGET_BYTES: u64 = 65536;

pub(crate) fn expand_groups_for_target(
    groups: Vec<CombinedScanTask>,
    target_partitions: usize,
    lookback: usize,
    open_file_cost: u64,
    split_size: u64,
    project_field_ids: &[i32],
    apply_residual_filter: bool,
) -> Result<Vec<CombinedScanTask>> {
    let target = target_partitions.max(1);
    if groups.len() >= target
        || !apply_residual_filter
        || project_field_ids.is_empty()
        || project_field_ids
            .iter()
            .any(|id| *id == RESERVED_FIELD_ID_POS || *id == RESERVED_FIELD_ID_ROW_ID)
    {
        return Ok(groups);
    }
    let total: u64 = groups
        .iter()
        .flat_map(|group| group.tasks().iter())
        .map(task_window_bytes)
        .fold(0u64, u64::saturating_add);
    if total == 0 {
        return Ok(groups);
    }
    let target_bytes = u64::try_from(target).unwrap_or(u64::MAX);
    let pack_target = split_size.min(total.div_ceil(target_bytes).max(MIN_SPLIT_TARGET_BYTES));
    let split_target = pack_target.max(MIN_SPLIT_TARGET_BYTES);
    let mut pieces: Vec<FileScanTask> = Vec::new();
    for group in &groups {
        for task in group.tasks() {
            pieces.extend(split_task_for_target(task, split_target)?);
        }
    }
    Ok(PackingIterator::new(
        pieces.into_iter(),
        pack_target.max(1),
        lookback.max(1),
        true,
        |task: &FileScanTask| task.weight(open_file_cost),
    )
    .map(CombinedScanTask::new)
    .collect())
}

fn task_window_bytes(task: &FileScanTask) -> u64 {
    if task.length == 0 {
        task.file_size_in_bytes
    } else {
        task.length
    }
}

fn split_task_for_target(task: &FileScanTask, split_target: u64) -> Result<Vec<FileScanTask>> {
    if task_window_bytes(task) <= split_target {
        return Ok(vec![task.clone()]);
    }
    if task.start == 0 && (task.length == 0 || task.length == task.file_size_in_bytes) {
        return task.split(split_target);
    }
    Ok(subdivide_ranged_task(task, split_target).unwrap_or_else(|| vec![task.clone()]))
}

fn subdivide_ranged_task(task: &FileScanTask, split_target: u64) -> Option<Vec<FileScanTask>> {
    if task.data_file_format != DataFileFormat::Parquet
        || task.length == 0
        || task
            .project_field_ids
            .iter()
            .any(|id| *id == RESERVED_FIELD_ID_POS || *id == RESERVED_FIELD_ID_ROW_ID)
    {
        return None;
    }
    let mut pieces = Vec::new();
    let mut offset = task.start;
    let mut remaining = task.length;
    while remaining > 0 {
        let length = split_target.min(remaining);
        pieces.push(split_window(task, offset, length));
        offset = offset.checked_add(length)?;
        remaining -= length;
    }
    Some(pieces)
}

#[cfg(test)]
mod split_tests {
    use std::sync::Arc;

    use super::*;
    use crate::spec::{NestedField, PrimitiveType, Schema, Type};

    const OPEN_FILE_COST: u64 = 4_194_304;

    fn task_schema() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                ))])
                .build()
                .expect("schema"),
        )
    }

    fn task(
        path: &str,
        file_size: u64,
        start: u64,
        length: u64,
        project: Vec<i32>,
    ) -> FileScanTask {
        FileScanTask {
            file_size_in_bytes: file_size,
            start,
            length,
            record_count: Some(1000),
            data_file_path: Arc::from(path),
            data_file_format: DataFileFormat::Parquet,
            schema: task_schema(),
            project_field_ids: Arc::from(project),
            predicate: None,
            deletes: Arc::from(vec![]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        }
    }

    fn whole_file(path: &str, size: u64) -> FileScanTask {
        task(path, size, 0, size, vec![1])
    }

    fn windows(groups: &[CombinedScanTask], path: &str) -> Vec<(u64, u64)> {
        let mut found: Vec<(u64, u64)> = groups
            .iter()
            .flat_map(|group| group.tasks().iter())
            .filter(|task| task.data_file_path.as_ref() == path)
            .map(|task| (task.start, task.length))
            .collect();
        found.sort_unstable();
        found
    }

    fn assert_tiles(mut found: Vec<(u64, u64)>, end: u64) {
        found.sort_unstable();
        let mut cursor = 0u64;
        for (start, length) in &found {
            assert_eq!(*start, cursor, "gap or overlap in {found:?}");
            cursor += length;
        }
        assert_eq!(cursor, end, "short cover in {found:?}");
    }

    fn expand(
        groups: Vec<CombinedScanTask>,
        target: usize,
        fields: &[i32],
        residual: bool,
    ) -> Vec<CombinedScanTask> {
        expand_groups_for_target(
            groups,
            target,
            10,
            OPEN_FILE_COST,
            134_217_728,
            fields,
            residual,
        )
        .expect("expand")
    }

    #[test]
    fn expand_leaves_enough_groups_untouched() {
        let groups: Vec<CombinedScanTask> = (0..8)
            .map(|index| {
                CombinedScanTask::new(vec![whole_file(&format!("{index}.parquet"), 5_000_000)])
            })
            .collect();
        let out = expand(groups, 8, &[1], true);
        assert_eq!(out.len(), 8);
        assert_eq!(windows(&out, "0.parquet"), vec![(0, 5_000_000)]);
    }

    #[test]
    fn expand_splits_small_whole_files_to_target() {
        let groups = vec![CombinedScanTask::new(vec![
            whole_file("a.parquet", 5_000_000),
            whole_file("b.parquet", 5_000_000),
            whole_file("c.parquet", 5_000_000),
            whole_file("d.parquet", 5_000_000),
        ])];
        let out = expand(groups, 8, &[1], true);
        assert_eq!(out.len(), 8);
        for path in ["a.parquet", "b.parquet", "c.parquet", "d.parquet"] {
            let found = windows(&out, path);
            assert_eq!(found.len(), 2);
            assert_tiles(found, 5_000_000);
        }
        for group in &out {
            for piece in group.tasks() {
                assert_eq!(piece.record_count, None);
                assert_eq!(piece.split_offsets, None);
            }
        }
        let total: u64 = out
            .iter()
            .flat_map(|group| group.tasks().iter())
            .map(task_window_bytes)
            .sum();
        assert_eq!(total, 20_000_000);
    }

    #[test]
    fn expand_takes_offsets_branch_for_offset_files() {
        let mut file = whole_file("a.parquet", 6_000_000);
        file.split_offsets = Some(vec![0, 2_000_000, 4_000_000]);
        let groups = vec![CombinedScanTask::new(vec![file])];
        let out = expand(groups, 8, &[1], true);
        assert_eq!(out.len(), 3);
        assert_eq!(windows(&out, "a.parquet"), vec![
            (0, 2_000_000),
            (2_000_000, 2_000_000),
            (4_000_000, 2_000_000)
        ]);
    }

    #[test]
    fn expand_subdivides_ranged_tasks() {
        let groups = vec![CombinedScanTask::new(vec![task(
            "a.parquet",
            200_000_000,
            128_000_000,
            72_000_000,
            vec![1],
        )])];
        let out = expand(groups, 8, &[1], true);
        let found = windows(&out, "a.parquet");
        assert_eq!(found.len(), 8);
        let mut cursor = 128_000_000u64;
        for (start, length) in &found {
            assert_eq!(*start, cursor);
            cursor += length;
        }
        assert_eq!(cursor, 200_000_000);
    }

    #[test]
    fn expand_repacks_without_subdividing_within_target_tasks() {
        let groups = vec![CombinedScanTask::new(vec![
            task("a.parquet", 1_000_000, 0, 1_000_000, vec![1]),
            task("b.parquet", 1_000_000, 0, 1_000_000, vec![1]),
        ])];
        let out = expand(groups, 2, &[1], true);
        assert_eq!(out.len(), 2);
        assert_eq!(windows(&out, "a.parquet"), vec![(0, 1_000_000)]);
        assert_eq!(windows(&out, "b.parquet"), vec![(0, 1_000_000)]);
    }

    #[test]
    fn expand_skips_empty_projection() {
        let groups = vec![CombinedScanTask::new(vec![whole_file(
            "a.parquet",
            5_000_000,
        )])];
        let out = expand(groups, 8, &[], true);
        assert_eq!(out.len(), 1);
        assert_eq!(windows(&out, "a.parquet"), vec![(0, 5_000_000)]);
    }

    #[test]
    fn expand_skips_pos_projection() {
        let groups = vec![CombinedScanTask::new(vec![whole_file(
            "a.parquet",
            5_000_000,
        )])];
        let out = expand(groups, 8, &[RESERVED_FIELD_ID_POS], true);
        assert_eq!(out.len(), 1);
        assert_eq!(windows(&out, "a.parquet"), vec![(0, 5_000_000)]);
    }

    #[test]
    fn expand_skips_row_id_projection() {
        let groups = vec![CombinedScanTask::new(vec![whole_file(
            "a.parquet",
            5_000_000,
        )])];
        let out = expand(groups, 8, &[RESERVED_FIELD_ID_ROW_ID], true);
        assert_eq!(out.len(), 1);
        assert_eq!(windows(&out, "a.parquet"), vec![(0, 5_000_000)]);
    }

    #[test]
    fn expand_skips_file_prune_only() {
        let groups = vec![CombinedScanTask::new(vec![whole_file(
            "a.parquet",
            5_000_000,
        )])];
        let out = expand(groups, 8, &[1], false);
        assert_eq!(out.len(), 1);
        assert_eq!(windows(&out, "a.parquet"), vec![(0, 5_000_000)]);
    }

    #[test]
    fn expand_skips_tiny_tables() {
        let groups = vec![CombinedScanTask::new(vec![whole_file("a.parquet", 4096)])];
        let out = expand(groups, 8, &[1], true);
        assert_eq!(out.len(), 1);
        assert_eq!(windows(&out, "a.parquet"), vec![(0, 4096)]);
    }

    #[test]
    fn expand_declines_avro_tasks() {
        let mut file = whole_file("a.avro", 5_000_000);
        file.data_file_format = DataFileFormat::Avro;
        let groups = vec![CombinedScanTask::new(vec![file])];
        let out = expand(groups, 8, &[1], true);
        assert_eq!(out.len(), 1);
        assert_eq!(windows(&out, "a.avro"), vec![(0, 5_000_000)]);
    }

    #[test]
    fn expand_zero_target_means_one_partition() {
        let groups = vec![CombinedScanTask::new(vec![whole_file(
            "a.parquet",
            5_000_000,
        )])];
        let out = expand(groups, 0, &[1], true);
        assert_eq!(out.len(), 1);
        assert_eq!(windows(&out, "a.parquet"), vec![(0, 5_000_000)]);
    }

    #[test]
    fn expand_zero_total_bytes_is_noop() {
        let groups = vec![CombinedScanTask::new(vec![task(
            "a.parquet",
            0,
            0,
            0,
            vec![1],
        )])];
        let out = expand(groups, 8, &[1], true);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn subdivide_overflow_stays_whole() {
        let groups = vec![CombinedScanTask::new(vec![task(
            "a.parquet",
            u64::MAX,
            u64::MAX - 70_000,
            100_000,
            vec![1],
        )])];
        let out = expand(groups, 8, &[1], true);
        assert_eq!(out.len(), 1);
        assert_eq!(windows(&out, "a.parquet"), vec![(
            u64::MAX - 70_000,
            100_000
        )]);
    }

    #[test]
    fn expand_zero_lookback_is_clamped() {
        let groups = vec![CombinedScanTask::new(vec![whole_file(
            "a.parquet",
            5_000_000,
        )])];
        let out = expand_groups_for_target(groups, 2, 0, OPEN_FILE_COST, 134_217_728, &[1], true)
            .expect("ok");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn expand_honors_tiny_configured_split_size() {
        let groups = vec![CombinedScanTask::new(vec![
            task("a.parquet", 3000, 4, 2996, vec![1]),
            task("b.parquet", 3000, 4, 2996, vec![1]),
            task("c.parquet", 3000, 4, 2996, vec![1]),
        ])];
        let out = expand_groups_for_target(groups, 4, 100, 1, 1, &[1], true).expect("ok");
        assert_eq!(out.len(), 3);
        assert_eq!(windows(&out, "a.parquet"), vec![(4, 2996)]);
    }

    #[test]
    fn expand_never_emits_windows_below_floor() {
        let groups = vec![CombinedScanTask::new(vec![task(
            "a.parquet",
            200_000,
            0,
            200_000,
            vec![1],
        )])];
        let out =
            expand_groups_for_target(groups, 8, 10, OPEN_FILE_COST, 1, &[1], true).expect("ok");
        let found = windows(&out, "a.parquet");
        assert_eq!(found.len(), 4);
        assert_tiles(found, 200_000);
    }

    #[test]
    fn expand_explicit_small_split_size_wins_over_derived() {
        let groups = vec![CombinedScanTask::new(vec![whole_file(
            "a.parquet",
            20_000_000,
        )])];
        let out = expand_groups_for_target(groups, 8, 10, OPEN_FILE_COST, 1_000_000, &[1], true)
            .expect("ok");
        let found = windows(&out, "a.parquet");
        assert_eq!(found.len(), 20);
        assert_tiles(found, 20_000_000);
    }

    #[test]
    fn split_pieces_inherit_deletes() {
        use crate::scan::FileScanTaskDeleteFile;
        use crate::spec::DataContentType;
        let mut file = whole_file("a.parquet", 5_000_000);
        file.deletes = Arc::from(vec![FileScanTaskDeleteFile {
            file_path: "d.parquet".to_string(),
            file_size_in_bytes: 100,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }]);
        let groups = vec![CombinedScanTask::new(vec![file])];
        let out = expand(groups, 2, &[1], true);
        assert_eq!(out.len(), 2);
        for group in &out {
            for piece in group.tasks() {
                assert_eq!(piece.deletes.len(), 1);
                assert_eq!(piece.record_count, None);
            }
        }
    }
}

fn split_window(task: &FileScanTask, start: u64, length: u64) -> FileScanTask {
    FileScanTask {
        file_size_in_bytes: task.file_size_in_bytes,
        start,
        length,
        record_count: None,
        data_file_path: Arc::clone(&task.data_file_path),
        data_file_format: task.data_file_format,
        schema: Arc::clone(&task.schema),
        project_field_ids: Arc::clone(&task.project_field_ids),
        predicate: task.predicate.as_ref().map(Arc::clone),
        deletes: Arc::clone(&task.deletes),
        partition: task.partition.clone(),
        partition_spec: task.partition_spec.as_ref().map(Arc::clone),
        name_mapping: task.name_mapping.as_ref().map(Arc::clone),
        case_sensitive: task.case_sensitive,
        split_offsets: None,
        first_row_id: task.first_row_id,
        file_sequence_number: task.file_sequence_number,
    }
}
