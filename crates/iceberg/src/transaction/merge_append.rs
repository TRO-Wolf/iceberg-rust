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

//! This module contains the merge-append action (Java `MergeAppend` / `ManifestMergeManager`).
//!
//! [`MergeAppendAction`] is the minimal-manifest-count append. It appends data files in one
//! `Operation::Append` snapshot exactly like [`crate::transaction::append::FastAppendAction`], then
//! BIN-PACKS the manifest list and MERGES the small manifests into fewer, larger ones. Java's
//! `Table.newAppend()` returns this merging producer, and `newFastAppend()` the non-merging one.
//!
//! ## The merge contract (Java `ManifestMergeManager`)
//!
//! After the producer writes the added-data manifest and carries the existing ones forward,
//! [`MergeManifestProcess`] runs the manager:
//!
//! | step | rule |
//! |---|---|
//! | 1 | Short-circuit when merging is off or the list is empty. |
//! | 2 | Group by spec id, reverse-sorted. Specs never merge across. |
//! | 3 | Bin-pack from the end. A one-file bin, or this commit's new manifest below `minCountToMerge`, stays. |
//! | 4 | `DELETED` only for this snapshot. `ADDED` from this snapshot stays Added. Else Existing, provenance intact. |
//!
//! The apply order mirrors Java: the new added-data manifest first, then the existing manifests in
//! order, then the merge.
//!
//! ## Physics: the new added manifest is read back before commit
//!
//! The new manifest is read back while the snapshot is still UNCOMMITTED, so its list entry carries
//! `UNASSIGNED_SEQUENCE_NUMBER`. Each `Added` entry inherits `Some(-1)`, and the merged writer
//! strips that negative seq back to `None` on disk. The entry then re-inherits the real sequence
//! number at commit, exactly as in a fast append. Carried-forward entries from COMMITTED manifests
//! hold real values and are written explicitly. The merged manifest's `added_snapshot_id` is the new
//! snapshot id, so the manifest-list writer legally stamps the new sequence number onto it.
//!
//! ## Named deviations from Java
//!
//! | not ported | consequence |
//! |---|---|
//! | delete-manifest merge | DELETE manifests carry forward unchanged |
//! | retry cache / `cleanUncommitted` | recompute from the refreshed base |
//! | `appendManifest` | same gap as `fast_append` |
//! | extra summary keys | same shape as fast_append |
//! | `scanManifestsWith` | sequential async |

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::TableUpdate;
use crate::error::Result;
use crate::spec::{
    DataFile, MAIN_BRANCH, ManifestContentType, ManifestEntry, ManifestFile, ManifestStatus,
    Operation, TableProperties,
};
use crate::table::Table;
use crate::transaction::snapshot::{
    FirstRowIdPolicy, ManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// A transaction action that appends data files in one `Operation::Append` snapshot and then MERGES the
/// resulting manifest list into a minimal number of manifests (Java `MergeAppend` / `Table.newAppend()`).
///
/// Create one with [`crate::transaction::Transaction::merge_append`]. It mirrors
/// [`crate::transaction::append::FastAppendAction`]'s public surface
/// ([`MergeAppendAction::add_data_files`], [`MergeAppendAction::set_commit_uuid`],
/// [`MergeAppendAction::set_key_metadata`], [`MergeAppendAction::set_snapshot_properties`],
/// [`MergeAppendAction::with_check_duplicate`]); the only difference is that on commit it bin-packs and
/// merges manifests per the three `commit.manifest*` table properties (see the module doc). The live
/// file set (the paths a scan reads) is identical to what the equivalent fast append would produce.
pub struct MergeAppendAction {
    check_duplicate: bool,
    trust_partition_metrics: bool,
    // below are properties used to create SnapshotProducer when commit
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    pub(crate) stage_only: bool,
    pub(crate) target_branch: String,
}

impl MergeAppendAction {
    pub(crate) fn new() -> Self {
        Self {
            check_duplicate: true,
            trust_partition_metrics: true,
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            stage_only: false,
            target_branch: MAIN_BRANCH.to_string(),
        }
    }

    /// Set whether to check duplicate files (mirrors `FastAppendAction::with_check_duplicate`).
    pub fn with_check_duplicate(mut self, v: bool) -> Self {
        self.check_duplicate = v;
        self
    }

    pub(crate) fn with_trust_partition_metrics(mut self, trust: bool) -> Self {
        self.trust_partition_metrics = trust;
        self
    }

    /// Add data files to the snapshot.
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Set commit UUID for the snapshot.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set key metadata for manifest files.
    pub fn set_key_metadata(mut self, key_metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(key_metadata);
        self
    }

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }
}

#[async_trait]
impl TransactionAction for MergeAppendAction {
    fn target_ref(&self) -> &str {
        self.target_branch.as_str()
    }

    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
            FirstRowIdPolicy::Suppress,
        )?
        .with_stage_only(self.stage_only)
        .with_target_branch(self.target_branch.clone())?;

        // Validate added files (identical to fast append — only DATA content, matching spec, valid
        // partition values).
        snapshot_producer.validate_added_data_files()?;

        // Check duplicate files (identical to fast append).
        if self.check_duplicate {
            snapshot_producer.validate_duplicate_files().await?;
        }

        // Read the three merge properties from the table at commit time (Java
        // `ManifestMergeManager` ctor args, from `TableProperties`).
        let merge_settings = MergeSettings::from_table(table);
        let merge_process =
            MergeManifestProcess::new(snapshot_producer.snapshot_id(), merge_settings);

        let mut commit = snapshot_producer
            .commit(MergeAppendOperation, merge_process)
            .await?;
        if !self.trust_partition_metrics {
            let updates: Vec<TableUpdate> = commit
                .take_updates()
                .into_iter()
                .map(|update| {
                    if let TableUpdate::AddSnapshot { mut snapshot } = update {
                        snapshot
                            .summary
                            .additional_properties
                            .remove("changed-partition-count");
                        snapshot
                            .summary
                            .additional_properties
                            .remove("partition-summaries-included");
                        snapshot
                            .summary
                            .additional_properties
                            .retain(|key, _| !key.starts_with("partitions."));
                        TableUpdate::AddSnapshot { snapshot }
                    } else {
                        update
                    }
                })
                .collect();
            commit = ActionCommit::new(updates, commit.take_requirements());
        }
        Ok(commit)
    }
}

/// The three `commit.manifest*` settings that drive the merge, read from the table properties at commit
/// time (Java `ManifestMergeManager` constructor: `targetSizeBytes`, `minCountToMerge`, `mergeEnabled`).
#[derive(Debug, Clone, Copy)]
struct MergeSettings {
    /// `commit.manifest.target-size-bytes` (default 8 MB) — the bin-packing target weight.
    target_size_bytes: u64,
    /// `commit.manifest.min-count-to-merge` (default 100) — the new-manifest bin is merged only at/above
    /// this many manifests.
    min_count_to_merge: u32,
    /// `commit.manifest-merge.enabled` (default true) — when false the manifest list is returned as-is.
    merge_enabled: bool,
}

impl MergeSettings {
    fn from_table(table: &Table) -> Self {
        let properties = table.metadata().properties();

        let target_size_bytes = properties
            .get(TableProperties::PROPERTY_COMMIT_MANIFEST_TARGET_SIZE_BYTES)
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(TableProperties::PROPERTY_COMMIT_MANIFEST_TARGET_SIZE_BYTES_DEFAULT);

        let min_count_to_merge = properties
            .get(TableProperties::PROPERTY_COMMIT_MANIFEST_MIN_COUNT_TO_MERGE)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(TableProperties::PROPERTY_COMMIT_MANIFEST_MIN_COUNT_TO_MERGE_DEFAULT);

        let merge_enabled = properties
            .get(TableProperties::PROPERTY_COMMIT_MANIFEST_MERGE_ENABLED)
            .and_then(|value| value.parse::<bool>().ok())
            .unwrap_or(TableProperties::PROPERTY_COMMIT_MANIFEST_MERGE_ENABLED_DEFAULT);

        Self {
            target_size_bytes,
            min_count_to_merge,
            merge_enabled,
        }
    }
}

/// The [`SnapshotProduceOperation`] for [`MergeAppendAction`] — identical to the fast-append operation
/// (records `Operation::Append`, removes no files, carries forward every existing manifest with live
/// files). The MERGE happens in [`MergeManifestProcess`], not here.
struct MergeAppendOperation;

impl SnapshotProduceOperation for MergeAppendOperation {
    fn operation(&self) -> Operation {
        Operation::Append
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn delete_files(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<DataFile>> {
        Ok(vec![])
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        // Carry forward every existing manifest that still has live files (Java
        // `MergingSnapshotProducer.apply`'s `shouldKeep = hasAddedFiles || hasExistingFiles ||
        // snapshotId() == snapshotId()`; the third clause is unreachable here — carried manifests
        // keep their old snapshot id). DELIBERATELY DIFFERENT from `FastAppendOperation`, which
        // carries ALL manifests unfiltered (Java `FastAppend.apply` -> `allManifests`): Java itself
        // is asymmetric between the two append flavors (O1, 2026-06-11, bytecode-pinned both ways).
        // The merge step then decides which of these to combine.
        let Some(snapshot) = snapshot_produce.parent_snapshot() else {
            return Ok(vec![]);
        };

        let manifest_list = snapshot
            .load_manifest_list(
                snapshot_produce.table.file_io(),
                &snapshot_produce.table.metadata_ref(),
            )
            .await?;

        Ok(manifest_list
            .entries()
            .iter()
            .filter(|entry| entry.has_added_files() || entry.has_existing_files())
            .cloned()
            .collect())
    }
}

/// The [`ManifestProcess`] that bin-packs and merges the snapshot's manifest list (Java
/// `ManifestMergeManager`, run from `MergingSnapshotProducer.apply`).
pub(crate) struct MergeManifestProcess {
    /// The new snapshot's id — the producer's `snapshot_id()`. Used to identify this commit's new added
    /// manifest (the Java `first` manifest) and to route entries (this-snapshot Added/Deleted vs carried).
    snapshot_id: i64,
    settings: MergeSettings,
}

impl MergeManifestProcess {
    fn new(snapshot_id: i64, settings: MergeSettings) -> Self {
        Self {
            snapshot_id,
            settings,
        }
    }

    fn split_and_reorder(
        &self,
        manifests: Vec<ManifestFile>,
    ) -> (Vec<ManifestFile>, Vec<ManifestFile>) {
        let mut new_added_data: Vec<ManifestFile> = Vec::new();
        let mut existing_data: Vec<ManifestFile> = Vec::new();
        let mut delete_manifests: Vec<ManifestFile> = Vec::new();

        for manifest in manifests {
            match manifest.content {
                ManifestContentType::Deletes => delete_manifests.push(manifest),
                ManifestContentType::Data => {
                    if manifest.added_snapshot_id == self.snapshot_id {
                        new_added_data.push(manifest);
                    } else {
                        existing_data.push(manifest);
                    }
                }
            }
        }

        new_added_data.sort_by_key(|manifest| manifest.partition_spec_id);
        let mut data_manifests = new_added_data;
        data_manifests.extend(existing_data);

        (data_manifests, delete_manifests)
    }

    /// Run the merge over the DATA manifests (Java `ManifestMergeManager.mergeManifests` over the data
    /// manager). Returns the merged data manifest list in Java's order (higher spec id first; within a
    /// spec id, the bin order).
    async fn merge_data_manifests(
        &self,
        snapshot_producer: &mut SnapshotProducer<'_>,
        data_manifests: Vec<ManifestFile>,
    ) -> Result<(Vec<ManifestFile>, usize)> {
        // Disabled / empty short-circuit (Java `mergeManifests` L80-83).
        if !self.settings.merge_enabled || data_manifests.is_empty() {
            return Ok((data_manifests, 0));
        }

        // Java's `first` is the unconditional STREAM HEAD (`ManifestFile first = manifestIter.next()`,
        // ManifestMergeManager L85) — NOT "the new manifest". After `split_and_reorder` the head is the
        // new added manifest when one exists; for an empty-data merging append (a properties-only
        // commit) the head is the first EXISTING manifest, and Java still gives ITS bin the min-count
        // protection. Gating this on `added_snapshot_id == self.snapshot_id` would drop the protection
        // for every bin on the empty-data path and merge manifests Java keeps (audit fix 2026-06-10).
        let first_manifest_path = data_manifests
            .first()
            .map(|manifest| manifest.manifest_path.clone());

        // Group by partition spec id, REVERSE-sorted (Java `groupBySpec` L129-137: a TreeMap with
        // `Comparator.reverseOrder()` ⇒ higher spec ids first). Preserve the within-group order.
        let mut groups: HashMap<i32, Vec<ManifestFile>> = HashMap::new();
        let mut spec_order: Vec<i32> = Vec::new();
        for manifest in data_manifests {
            let spec_id = manifest.partition_spec_id;
            groups.entry(spec_id).or_insert_with(|| {
                spec_order.push(spec_id);
                Vec::new()
            });
            groups
                .get_mut(&spec_id)
                .expect("group was just inserted")
                .push(manifest);
        }
        spec_order.sort_unstable_by(|a, b| b.cmp(a)); // reverse order

        let mut merged: Vec<ManifestFile> = Vec::new();
        let mut replaced_count = 0usize;
        for spec_id in spec_order {
            let group = groups.remove(&spec_id).expect("spec id came from the keys");
            let (group_result, group_replaced) = self
                .merge_group(
                    snapshot_producer,
                    spec_id,
                    group,
                    first_manifest_path.as_deref(),
                )
                .await?;
            merged.extend(group_result);
            replaced_count += group_replaced;
        }

        Ok((merged, replaced_count))
    }

    /// Bin-pack and merge one spec-id group (Java `mergeGroup` L140-185).
    async fn merge_group(
        &self,
        snapshot_producer: &mut SnapshotProducer<'_>,
        spec_id: i32,
        group: Vec<ManifestFile>,
        first_manifest_path: Option<&str>,
    ) -> Result<(Vec<ManifestFile>, usize)> {
        // ListPacker(targetSizeBytes, lookback=1, largestBinFirst=false).packEnd(group,
        // ManifestFile::length) (Java L146-148). The weight is the manifest's on-disk length.
        let bins = bin_packing::pack_end(group, self.settings.target_size_bytes, |manifest| {
            manifest.manifest_length.max(0) as u64
        });

        let mut output: Vec<ManifestFile> = Vec::new();
        let mut replaced_count = 0usize;
        for bin in bins {
            let bin_contains_first = first_manifest_path.is_some_and(|first_path| {
                bin.iter()
                    .any(|manifest| manifest.manifest_path == first_path)
            });
            match bin_disposition(
                bin.len(),
                bin_contains_first,
                self.settings.min_count_to_merge,
            ) {
                BinDisposition::Keep => output.extend(bin),
                BinDisposition::Merge => {
                    replaced_count += bin
                        .iter()
                        .filter(|manifest| manifest.added_snapshot_id != self.snapshot_id)
                        .count();
                    let merged = self
                        .create_manifest(snapshot_producer, spec_id, &bin)
                        .await?;
                    output.push(merged);
                }
            }
        }

        Ok((output, replaced_count))
    }

    /// Merge `bin` into a single manifest (Java `createManifest` L187-239). Per entry of each source
    /// manifest, route via the three-way rule (this-snapshot DELETED → delete; this-snapshot ADDED →
    /// add; else → existing). Older tombstones (DELETED by a previous snapshot) are SUPPRESSED.
    async fn create_manifest(
        &self,
        snapshot_producer: &mut SnapshotProducer<'_>,
        spec_id: i32,
        bin: &[ManifestFile],
    ) -> Result<ManifestFile> {
        let mut writer =
            snapshot_producer.new_cluster_manifest_writer(spec_id, ManifestContentType::Data)?;

        for manifest_file in bin {
            let manifest = manifest_file
                .load_manifest(snapshot_producer.table.file_io())
                .await?;
            for entry in manifest.entries() {
                let entry = entry.as_ref().clone();
                match entry.status() {
                    ManifestStatus::Deleted => {
                        // Suppress deletes from previous snapshots: only files DELETED by THIS snapshot
                        // are carried into the merged manifest (Java L203-208). For merge_append this is
                        // unreachable (an append never deletes a file), but the routing is kept faithful.
                        if entry.snapshot_id() == Some(self.snapshot_id) {
                            writer.add_delete_entry(entry)?;
                        }
                    }
                    ManifestStatus::Added if entry.snapshot_id() == Some(self.snapshot_id) => {
                        // Adds from THIS snapshot stay adds (Java L209-211). `add_entry` re-stamps the
                        // entry to Added + this snapshot and strips the inherited `Some(-1)` seq back to
                        // `None` so it re-inherits the new snapshot's real seq at commit (see the Physics
                        // section in the module doc).
                        writer.add_entry(entry)?;
                    }
                    _ => {
                        // Everything else (older Added, or Existing) becomes an Existing entry with its
                        // ORIGINAL provenance preserved (Java L212-214, `writer.existing(entry)`). This is
                        // the load-bearing invariant — re-stamping here is the silent-corruption class.
                        writer.add_existing_entry(entry)?;
                    }
                }
            }
        }

        writer.write_manifest_file().await
    }
}

impl ManifestProcess for MergeManifestProcess {
    async fn process_manifests(
        &self,
        snapshot_produce: &mut SnapshotProducer<'_>,
        manifests: Vec<ManifestFile>,
    ) -> Result<(Vec<ManifestFile>, usize)> {
        // Split DATA from DELETE manifests and reorder DATA so the new added manifest is first.
        let (data_manifests, delete_manifests) = self.split_and_reorder(manifests);

        // Merge the data manifests (Java `mergeManager.mergeManifests(unmergedManifests)`).
        let (merged_data, replaced_count) = self
            .merge_data_manifests(snapshot_produce, data_manifests)
            .await?;

        // Output order: merged data manifests (bin order), then the delete manifests carried unchanged
        // (Java appends `deleteMergeManager.mergeManifests(unmergedDeleteManifests)` after the data
        // manifests; this port leaves the delete manifests un-merged — see the module doc).
        let mut result = merged_data;
        result.extend(delete_manifests);
        Ok((result, replaced_count))
    }
}

/// What `merge_group` does with a single bin (Java `mergeGroup` L162-181, factored out so the rule is
/// unit-testable without driving real manifest lengths).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum BinDisposition {
    /// Keep every manifest in the bin as-is (no rewrite).
    Keep,
    /// Merge the whole bin into one manifest.
    Merge,
}

/// Decide a bin's disposition (Java `mergeGroup` L162-181):
/// - `bin_len == 1` ⇒ keep the single manifest as-is.
/// - the bin CONTAINS the FIRST (new added) manifest AND `bin_len < min_count_to_merge` ⇒ keep ALL (the
///   min-count gate applies ONLY to the new-manifest bin, so a big old manifest can't block merging
///   older groups).
/// - otherwise ⇒ merge.
fn bin_disposition(
    bin_len: usize,
    bin_contains_first: bool,
    min_count_to_merge: u32,
) -> BinDisposition {
    if bin_len == 1 {
        return BinDisposition::Keep;
    }
    if bin_contains_first && (bin_len as u32) < min_count_to_merge {
        return BinDisposition::Keep;
    }
    BinDisposition::Merge
}

/// A faithful port of Java `BinPacking.ListPacker.packEnd` (`core/util/BinPacking.java`) for the
/// lookback-1, non-largest-bin-first case the manifest merge uses.
///
/// `pack_end` packs `items` into bins of total weight `<= target_weight` BY PACKING FROM THE END: it
/// reverses the input, runs the greedy first-fit packer (lookback 1 ⇒ a single open bin), then reverses
/// each bin's items AND the list of bins so the original order is restored. The "from the end" property
/// is what makes the UNDER-FILLED bin the FIRST one in the output (Java `mergeGroup`'s comment), so the
/// small leftover bin is the one merged on the next append.
///
/// This module ports the general `PackingIterable`/`PackingIterator` algorithm (configurable lookback,
/// `largest_bin_first`, `max_items_per_bin`), not just the lookback-1 special case, so the bin-packing
/// can be unit-tested against hand-computed Java outcomes independent of the merge manager.
mod bin_packing {
    /// Pack `items` into weight-bounded bins from the END, restoring the original order (Java
    /// `ListPacker.packEnd(items, weightFunc)` with `lookback = 1`, `largest_bin_first = false`,
    /// `max_items_per_bin = u64::MAX`).
    pub(super) fn pack_end<T>(
        items: Vec<T>,
        target_weight: u64,
        weight_func: impl Fn(&T) -> u64,
    ) -> Vec<Vec<T>> {
        pack_end_with(items, target_weight, 1, false, u64::MAX, weight_func)
    }

    /// The general `packEnd` (Java `ListPacker.packEnd` with all `PackingIterable` knobs exposed): pack
    /// `reverse(items)` with the greedy packer, then `reverse` each bin and the bin list.
    pub(super) fn pack_end_with<T>(
        mut items: Vec<T>,
        target_weight: u64,
        lookback: usize,
        largest_bin_first: bool,
        max_items_per_bin: u64,
        weight_func: impl Fn(&T) -> u64,
    ) -> Vec<Vec<T>> {
        items.reverse();
        let mut bins = pack(
            items,
            target_weight,
            lookback,
            largest_bin_first,
            max_items_per_bin,
            weight_func,
        );
        for bin in &mut bins {
            bin.reverse();
        }
        bins.reverse();
        bins
    }

    /// The forward greedy packer — Java `BinPacking.PackingIterator.next` materialized eagerly. With a
    /// `lookback` window of open bins, each item is placed in the FIRST open bin that can still hold it
    /// (`bin_weight + weight <= target_weight && bin_size < max_items_per_bin`); when a new bin is opened
    /// and the open-bin count exceeds `lookback`, the oldest (or largest, if `largest_bin_first`) open bin
    /// is emitted. At the end the remaining open bins are emitted in order.
    ///
    /// `lookback` must be `>= 1` (Java `Preconditions.checkArgument(lookback > 0)`); the callers here
    /// always pass a positive value, so an invalid lookback is a programming error (asserted).
    pub(super) fn pack<T>(
        items: Vec<T>,
        target_weight: u64,
        lookback: usize,
        largest_bin_first: bool,
        max_items_per_bin: u64,
        weight_func: impl Fn(&T) -> u64,
    ) -> Vec<Vec<T>> {
        assert!(lookback >= 1, "bin look-back size must be greater than 0");

        let mut closed_bins: Vec<Vec<T>> = Vec::new();
        // Open bins, oldest first. Each is (items, accumulated_weight).
        let mut open_bins: Vec<(Vec<T>, u64)> = Vec::new();

        for item in items {
            let weight = weight_func(&item);

            // findBin: the first open bin that can still hold this item (Java `findBin`,
            // `bin.weight + weight <= targetWeight`). Saturating add: weights come from the
            // UNTRUSTED `manifest_length` field of a manifest list read from storage, and a hostile
            // value near u64::MAX must not panic (debug) or wrap into "fits" (release) — saturation
            // makes an absurd sum simply never fit, which opens a fresh bin (audit hardening
            // 2026-06-10; identical to Java for every realistic weight).
            let target_bin = open_bins.iter().position(|(bin_items, bin_weight)| {
                bin_weight.saturating_add(weight) <= target_weight
                    && (bin_items.len() as u64) < max_items_per_bin
            });

            match target_bin {
                Some(index) => {
                    let (bin_items, bin_weight) = &mut open_bins[index];
                    bin_items.push(item);
                    *bin_weight = bin_weight.saturating_add(weight);
                }
                None => {
                    // Open a new bin for this item.
                    open_bins.push((vec![item], weight));

                    // If we now exceed the lookback window, emit one open bin (Java
                    // `bins.removeFirst()` / `removeLargestBin`).
                    if open_bins.len() > lookback {
                        let remove_index = if largest_bin_first {
                            largest_bin_index(&open_bins)
                        } else {
                            0
                        };
                        let (bin_items, _) = open_bins.remove(remove_index);
                        closed_bins.push(bin_items);
                    }
                }
            }
        }

        // Emit the remaining open bins in order (Java drains `bins.removeFirst()` until empty).
        for (bin_items, _) in open_bins {
            closed_bins.push(bin_items);
        }

        closed_bins
    }

    /// The index of the open bin with the greatest accumulated weight (Java `removeLargestBin` ⇒
    /// `Collections.max(bins, comparingLong(Bin::weight))`). Ties take the FIRST max, matching Java's
    /// `Collections.max` (which returns the first element that is not less than every other).
    fn largest_bin_index<T>(open_bins: &[(Vec<T>, u64)]) -> usize {
        let mut max_index = 0;
        let mut max_weight = open_bins[0].1;
        for (index, (_, weight)) in open_bins.iter().enumerate().skip(1) {
            if *weight > max_weight {
                max_weight = *weight;
                max_index = index;
            }
        }
        max_index
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // Risk: the lookback-1 packEnd order semantics drift from Java — items must pack from the END so
        // the under-filled bin is FIRST. Weights [3,3,3,3] target 6 ⇒ reversed [3,3,3,3] packs into
        // [[3,3],[3,3]]; reversing restores [[3,3],[3,3]]. Exact-boundary: a bin holding two 3s == target
        // 6 (canAdd uses `<=`), so each bin holds exactly two.
        #[test]
        fn test_pack_end_exact_boundary_pairs() {
            let bins = pack_end(vec![3u64, 3, 3, 3], 6, |w| *w);
            assert_eq!(bins, vec![vec![3, 3], vec![3, 3]]);
        }

        // Risk: an item exactly at the target on its own opens its own bin, and the order is preserved.
        // Weights [4,3,3] target 6: reversed [3,3,4]; first 3 → bin A(3); second 3 fits A → A(6); 4 does
        // not fit A (6+4>6) → new bin B(4). Open bins after loop: [A=[3,3], B=[4]] (lookback 1 ⇒ when B
        // opened, A was emitted first). Reversing each bin + the list ⇒ [[4],[3,3]].
        #[test]
        fn test_pack_end_under_filled_bin_is_first() {
            let bins = pack_end(vec![4u64, 3, 3], 6, |w| *w);
            assert_eq!(bins, vec![vec![4], vec![3, 3]]);
        }

        // Risk: a single item is its own bin (the bin.size()==1 keep-as-is path upstream depends on this).
        #[test]
        fn test_pack_end_single_item() {
            let bins = pack_end(vec![5u64], 6, |w| *w);
            assert_eq!(bins, vec![vec![5]]);
        }

        // Risk: everything fits in one bin when the target is large — the merge-everything case. Weights
        // [1,1,1] target 100 ⇒ one bin [1,1,1], order preserved.
        #[test]
        fn test_pack_end_all_in_one_bin() {
            let bins = pack_end(vec![1u64, 1, 1], 100, |w| *w);
            assert_eq!(bins, vec![vec![1, 1, 1]]);
        }

        // Risk: an item heavier than the target still gets its own bin (Java never drops an item — canAdd
        // is false for an empty bin too, so a new bin is opened and the over-weight item sits alone).
        #[test]
        fn test_pack_end_over_weight_item_alone() {
            let bins = pack_end(vec![1u64, 10, 1], 6, |w| *w);
            // reversed [1,10,1]: 1→A(1); 10 doesn't fit A(1+10>6) → emit A, B(10); 1 doesn't fit
            // B(10+1>6) → emit B, C(1). closed=[[1],[10]], open=[[1]] ⇒ [[1],[10],[1]]. reverse each
            // (no-op) + list ⇒ [[1],[10],[1]].
            assert_eq!(bins, vec![vec![1], vec![10], vec![1]]);
        }

        // Risk: reverse-order preservation — the forward `pack` (not packEnd) packs from the FRONT.
        // Weights [3,3,3,3] target 6 ⇒ [[3,3],[3,3]] (front packing, lookback 1).
        #[test]
        fn test_pack_forward_pairs() {
            let bins = pack(vec![3u64, 3, 3, 3], 6, 1, false, u64::MAX, |w| *w);
            assert_eq!(bins, vec![vec![3, 3], vec![3, 3]]);
        }

        // Risk: max_items_per_bin caps a bin even when weight would allow more (Java `binSize < maxSize`).
        // Weights [1,1,1,1] target 100, max 2 ⇒ [[1,1],[1,1]] despite the huge target.
        #[test]
        fn test_pack_respects_max_items_per_bin() {
            let bins = pack(vec![1u64, 1, 1, 1], 100, 1, false, 2, |w| *w);
            assert_eq!(bins, vec![vec![1, 1], vec![1, 1]]);
        }
    }
}

#[cfg(test)]
#[path = "merge_append_tests.rs"]
mod tests;
