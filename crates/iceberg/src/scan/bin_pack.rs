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
