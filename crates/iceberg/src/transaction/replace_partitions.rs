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

//! This module contains the replace-partitions action (dynamic partition overwrite).
//!
//! [`ReplacePartitionsAction`] mirrors Java `BaseReplacePartitions`: when committed, for every
//! partition that an ADDED file belongs to it DELETES all existing live data files in that same
//! `(spec_id, partition)` tuple, then adds the new files — in ONE `Overwrite` snapshot. The replace is
//! BY PARTITION VALUE (not by file path or row filter), so no metrics evaluators are needed.
//!
//! How it reuses the existing machinery: the added files reach the producer exactly as fast-append does
//! (written to a new added manifest), and the partition-scoped deletes are resolved + filtered out of the
//! current snapshot's manifests via the shared [`SnapshotProducer::resolve_partition_deletes`] (the
//! by-partition sibling of `resolve_delete_paths`) feeding the SAME `process_deletes` rewrite path that
//! `DeleteFiles` / `OverwriteFiles` use. Both happen in one snapshot.
//!
//! **Java semantics mirrored (cited against `core/.../BaseReplacePartitions.java`):**
//! - `operation()` returns `DataOperations.OVERWRITE` → always [`Operation::Overwrite`].
//! - the constructor sets `SnapshotSummary.REPLACE_PARTITIONS_PROP = "replace-partitions" = "true"`.
//! - `addFile(file)` drops `file.partition()` (in `file.specId()`) then adds the file.
//! - **Unpartitioned table = FULL replace:** every file is in the single empty partition, so adding any
//!   file drops that one partition → ALL existing files are replaced (Java `apply()` reaches the same end
//!   via `deleteByRowFilter(alwaysTrue)` when `dataSpec().isUnpartitioned()`).
//! - **A replaced partition with no existing files is a pure add** (no spurious delete, no error): Java's
//!   `failMissingDeletePaths` guards only path/file deletes, never partition drops.
//!
//! **Summary note (Java-faithful, NOT a Rust truncate):** Java `SnapshotProducer.summary()` has NO
//! full-table-truncate branch — it computes `total = previous + added - removed` unconditionally via
//! `updateTotal`. So the producer's `truncate_full_table` flag stays `false` here: the by-partition
//! resolution already reports EVERY removed file, so `deleted-data-files` / `deleted-records` are correct
//! and `update_totals` yields the right post-replace totals (e.g. an unpartitioned full replace of N files
//! adding M: `N + M - N = M`, no underflow). `replace-partitions=true` is just a summary property.
//!
//! **Out of scope (deferred):**
//! - static `replaceByRowFilter` / explicit-partition overwrite APIs — need inclusive/strict metrics
//!   evaluators to select files by row predicate.
//! - concurrent-commit conflict validation (`validateNoConflictingData` / `validateNoConflictingDeletes` /
//!   `validateFromSnapshot`) — serializable isolation, needs ancestor-chain validation-history replay.
//!
//! This increment is the dynamic-by-added-files core.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{DataFile, ManifestEntry, ManifestFile, Operation, Struct};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// The snapshot-summary property Java `BaseReplacePartitions` sets to mark a dynamic partition overwrite
/// (`SnapshotSummary.REPLACE_PARTITIONS_PROP`).
const REPLACE_PARTITIONS_PROP: &str = "replace-partitions";

/// A transaction action that performs a dynamic partition overwrite: for every partition an added file
/// belongs to, it replaces (deletes) all existing live data files in that same partition, then adds the
/// new files — in a single `Overwrite` snapshot.
///
/// Use [`crate::transaction::Transaction::replace_partitions`] to create one. Accumulate the files to add
/// with [`ReplacePartitionsAction::add_file`] / [`ReplacePartitionsAction::add_files`], then apply and
/// commit the transaction. The set of partitions to replace is derived from the added files' partition
/// values — replace is BY PARTITION, not by path.
///
/// On an UNPARTITIONED table every file is in the single empty partition, so adding any file replaces ALL
/// existing files (a full-table replace). A replace touching a partition that has no existing files is a
/// pure add. A replace with no added files (and therefore no resolved deletes) is rejected as empty.
pub struct ReplacePartitionsAction {
    /// Data files to add to the table (validated like fast append). Their partition values also determine
    /// which partitions are replaced.
    added_data_files: Vec<DataFile>,
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
}

impl ReplacePartitionsAction {
    pub(crate) fn new() -> Self {
        Self {
            added_data_files: vec![],
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
        }
    }

    /// Add a single [`DataFile`] to the table (Java `ReplacePartitions.addFile`). Its partition is
    /// scheduled for replacement: all existing live data files in the same `(spec_id, partition)` are
    /// removed when the action commits.
    pub fn add_file(mut self, data_file: DataFile) -> Self {
        self.added_data_files.push(data_file);
        self
    }

    /// Add multiple [`DataFile`]s to the table. Every partition they belong to is replaced.
    pub fn add_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Set the commit UUID for the snapshot (otherwise a fresh v7 UUID is generated).
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set key metadata for manifest files.
    pub fn set_key_metadata(mut self, key_metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(key_metadata);
        self
    }

    /// Set snapshot summary properties. The `replace-partitions` marker is added on top of these (Java
    /// sets it in the action constructor), so an explicit value here does not clear it.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }

    /// Collect the set of `(partition_spec_id, partition)` tuples the added files belong to — the
    /// partitions to replace (Java `replacedPartitions` / the per-file `dropPartition`).
    fn drop_partitions(&self) -> HashSet<(i32, Struct)> {
        self.added_data_files
            .iter()
            .map(|data_file| (data_file.partition_spec_id, data_file.partition().clone()))
            .collect()
    }
}

#[async_trait]
impl TransactionAction for ReplacePartitionsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // Mark the snapshot as a dynamic partition overwrite (Java `BaseReplacePartitions` constructor
        // sets `replace-partitions=true`). Layer it on top of any caller-provided properties.
        let mut snapshot_properties = self.snapshot_properties.clone();
        snapshot_properties.insert(REPLACE_PARTITIONS_PROP.to_string(), "true".to_string());

        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            snapshot_properties,
            self.added_data_files.clone(),
        );

        // Validate the added files like fast append: data content type, partition-spec match, and
        // partition-value compatibility (Java `MergingSnapshotProducer.add`). The partition-scoped deletes
        // are resolved inside the producer's commit via the operation's `delete_files` seam.
        snapshot_producer.validate_added_data_files()?;

        snapshot_producer
            .commit(
                ReplacePartitionsOperation {
                    drop_partitions: self.drop_partitions(),
                },
                DefaultManifestProcess,
            )
            .await
    }
}

/// The [`SnapshotProduceOperation`] for [`ReplacePartitionsAction`].
///
/// Records `Operation::Overwrite` (Java `BaseReplacePartitions.operation()` = `OVERWRITE`), exposes every
/// current data manifest as the set to filter, and resolves the drop-partition set against the current
/// snapshot's live data entries (the resolved [`DataFile`]s drive the producer's by-path manifest rewrite).
/// The added files reach the producer separately (passed to `SnapshotProducer::new`), so a single snapshot
/// carries both the added manifest and the rewritten (filtered) manifests.
struct ReplacePartitionsOperation {
    /// The `(partition_spec_id, partition)` tuples to replace, derived from the added files.
    drop_partitions: HashSet<(i32, Struct)>,
}

impl SnapshotProduceOperation for ReplacePartitionsOperation {
    fn operation(&self) -> Operation {
        // Java `BaseReplacePartitions.operation()` returns `DataOperations.OVERWRITE`.
        Operation::Overwrite
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn delete_files(&self, snapshot_produce: &SnapshotProducer<'_>) -> Result<Vec<DataFile>> {
        // Resolve the drop-partition set against the current snapshot's live data entries: every live
        // data file whose `(spec_id, partition)` is in the set is removed (Java
        // `ManifestFilterManager`'s `dropPartitions.contains(...)`). No missing-target validation —
        // a replaced partition with no existing files is a pure add.
        snapshot_produce
            .resolve_partition_deletes(&self.drop_partitions)
            .await
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        // Expose every current data manifest; the producer's `process_deletes` decides per manifest
        // whether to rewrite (to drop replaced-partition files), carry forward unchanged, or drop it.
        snapshot_produce.current_data_manifests().await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::io::BufReader;

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestStatus,
        Operation, Struct, TableMetadata,
    };
    use crate::table::Table;
    use crate::transaction::tests::make_v3_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, TableCreation, TableIdent};

    /// Build a data file routed to partition `x = part_value` (the V3 minimal table is partitioned by
    /// identity(x), spec id 0) with a unique path.
    fn data_file(path: &str, part_value: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .build()
            .unwrap()
    }

    /// Build a data file for an UNPARTITIONED table (spec id 0, empty partition struct).
    fn unpartitioned_data_file(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .unwrap()
    }

    /// Collect the set of live (Added or Existing) data file paths across the table's current snapshot —
    /// the real correctness signal (what a scan would read).
    async fn live_file_paths(table: &Table) -> HashSet<String> {
        let snapshot = table
            .metadata()
            .current_snapshot()
            .expect("table should have a current snapshot");
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .expect("manifest list should load");

        let mut live = HashSet::new();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file
                .load_manifest(table.file_io())
                .await
                .expect("manifest should load");
            for entry in manifest.entries() {
                if entry.is_alive() {
                    live.insert(entry.file_path().to_string());
                }
            }
        }
        live
    }

    /// Append the given files in a single fast-append commit and return the updated table.
    async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let action = tx.fast_append().add_data_files(files);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Return (snapshot_id, sequence_number, file_sequence_number) of the live entry for `path`.
    async fn entry_provenance(
        table: &Table,
        path: &str,
    ) -> (Option<i64>, Option<i64>, Option<i64>) {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.is_alive() && entry.file_path() == path {
                    return (
                        entry.snapshot_id(),
                        entry.sequence_number(),
                        entry.file_sequence_number,
                    );
                }
            }
        }
        panic!("no live entry for {path}");
    }

    /// Create an UNPARTITIONED V3 table in the catalog (same schema as the minimal fixture, but with no
    /// partition spec). Used to pin the full-table-replace behavior of `ReplacePartitions`.
    async fn make_v3_unpartitioned_table_in_catalog(catalog: &impl Catalog) -> Table {
        let table_ident =
            TableIdent::from_strs([format!("ns1-{}", uuid::Uuid::new_v4()), "test1".to_string()])
                .unwrap();
        catalog
            .create_namespace(table_ident.namespace(), HashMap::new())
            .await
            .unwrap();

        let file = File::open(format!(
            "{}/testdata/table_metadata/{}",
            env!("CARGO_MANIFEST_DIR"),
            "TableMetadataV3ValidMinimal.json"
        ))
        .unwrap();
        let reader = BufReader::new(file);
        let base_metadata = serde_json::from_reader::<_, TableMetadata>(reader).unwrap();

        // No partition_spec → unpartitioned table.
        let table_creation = TableCreation::builder()
            .schema((**base_metadata.current_schema()).clone())
            .sort_order((**base_metadata.default_sort_order()).clone())
            .name(table_ident.name().to_string())
            .format_version(crate::spec::FormatVersion::V3)
            .build();

        catalog
            .create_table(table_ident.namespace(), table_creation)
            .await
            .unwrap()
    }

    /// Read a u64 total from a snapshot summary property, defaulting to 0 when absent.
    fn total(table: &Table, prop: &str) -> u64 {
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties
            .get(prop)
            .map(|value| value.parse::<u64>().unwrap())
            .unwrap_or(0)
    }

    /// THE KEY CROSS-PARTITION-ISOLATION TEST. Append fileA in x=0 and fileB in x=1; then
    /// `replace_partitions().add_file(fileA2 in x=0)` → the post-commit SCAN live set is exactly
    /// {fileA2 (x=0), fileB (x=1)}: fileA is replaced, fileB in the UNTOUCHED partition survives.
    ///
    /// Risk pinned: cross-partition data loss / wrong live set. A bug that loses fileB (over-deletes the
    /// untouched partition) or keeps fileA (under-deletes the replaced partition) is silent data
    /// corruption — the single most dangerous failure for a partition overwrite.
    #[tokio::test]
    async fn test_replace_partitions_replaces_only_the_added_partition_keeps_others() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Append fileA @ x=0 and fileB @ x=1 (one manifest holding both partitions).
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 1),
        ])
        .await;

        // Replace partition x=0 with fileA2 — fileB in x=1 must be untouched.
        let tx = Transaction::new(&table);
        let action = tx
            .replace_partitions()
            .add_file(data_file("test/a2.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Overwrite,
            "ReplacePartitions records an Overwrite operation (Java BaseReplacePartitions.operation())"
        );
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/a2.parquet".to_string(), "test/b.parquet".to_string(),]),
            "x=0 replaced (a2 replaces a), x=1 untouched (b survives)"
        );
    }

    /// Pins: replacing MULTIPLE partitions in one action replaces each of them and leaves every other
    /// partition untouched. Risk: a replace that only handles the first added file's partition (wrong live
    /// set across multiple replaced partitions).
    #[tokio::test]
    async fn test_replace_partitions_replaces_multiple_partitions_at_once() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Three partitions populated: x=0 (a), x=1 (b), x=2 (c).
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 1),
            data_file("test/c.parquet", 2),
        ])
        .await;

        // Replace partitions x=0 and x=1; leave x=2 alone.
        let tx = Transaction::new(&table);
        let action = tx
            .replace_partitions()
            .add_file(data_file("test/a2.parquet", 0))
            .add_file(data_file("test/b2.parquet", 1));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from([
                "test/a2.parquet".to_string(),
                "test/b2.parquet".to_string(),
                "test/c.parquet".to_string(),
            ]),
            "x=0 and x=1 replaced, x=2 (c) untouched"
        );
    }

    /// Pins: replacing a single partition with MULTIPLE new files removes the old file and adds all the
    /// new ones in that partition. Risk: only the first added file landing (lost new files).
    #[tokio::test]
    async fn test_replace_partition_with_multiple_new_files() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/old.parquet", 0),
            data_file("test/keep.parquet", 1),
        ])
        .await;

        // Replace x=0 with TWO files; keep.parquet in x=1 must survive.
        let tx = Transaction::new(&table);
        let action = tx.replace_partitions().add_files(vec![
            data_file("test/new1.parquet", 0),
            data_file("test/new2.parquet", 0),
        ]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from([
                "test/new1.parquet".to_string(),
                "test/new2.parquet".to_string(),
                "test/keep.parquet".to_string(),
            ]),
            "x=0 now holds new1+new2 (old replaced), x=1 keep survives"
        );
    }

    /// Pins PROVENANCE preservation (the Increment-1 lesson, the #1 corruption risk). When a replace
    /// rewrites a manifest, every SURVIVING entry in an untouched partition must be copied forward as
    /// `Existing` carrying its ORIGINAL `snapshot_id` + both sequence numbers — NOT re-stamped with the
    /// new replace snapshot/seq. A carried-forward (untouched) manifest's entries also keep provenance.
    ///
    /// Risk pinned: a rewrite that re-stamps surviving entries is silent table corruption (a wrong
    /// data-sequence number breaks merge-on-read delete application and incremental scans). The live-set
    /// tests above assert only the live PATH set, so they pass under a snapshot-id re-stamp — only this
    /// test catches it.
    #[tokio::test]
    async fn test_replace_partitions_preserves_surviving_entry_provenance() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Append A@x=0 and B@x=1 in ONE commit (snapshot S1, data seq 1; one manifest holding both).
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 1),
        ])
        .await;
        let s1 = table.metadata().current_snapshot().unwrap().snapshot_id();
        let (b_snap, b_seq, b_fseq) = entry_provenance(&table, "test/b.parquet").await;
        assert_eq!(b_snap, Some(s1), "B added by S1");

        // Replace partition x=0 → rewrites S1's manifest (B in x=1 survives as Existing).
        let tx = Transaction::new(&table);
        let action = tx
            .replace_partitions()
            .add_file(data_file("test/a2.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s2 = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(s1, s2);

        // B survived in the untouched partition: MUST keep S1's snapshot id + seq numbers (NOT S2).
        let (b2_snap, b2_seq, b2_fseq) = entry_provenance(&table, "test/b.parquet").await;
        assert_eq!(
            b2_snap,
            Some(s1),
            "surviving B (untouched partition) must keep its ORIGINAL snapshot id S1, not the replace S2"
        );
        assert_eq!(
            b2_seq, b_seq,
            "surviving B must keep its ORIGINAL data seq, not the replace seq"
        );
        assert_eq!(
            b2_fseq, b_fseq,
            "surviving B must keep its ORIGINAL file seq"
        );

        // The newly added A2 gets the NEW replace snapshot's provenance.
        let (a2_snap, _a2_seq, _) = entry_provenance(&table, "test/a2.parquet").await;
        assert_eq!(
            a2_snap,
            Some(s2),
            "added A2 gets the new replace snapshot id S2"
        );

        // The replaced A is a Deleted tombstone (not live).
        assert!(
            !live_file_paths(&table).await.contains("test/a.parquet"),
            "the replaced A must no longer be live"
        );
    }

    /// THE FULL-REPLACE TEST. On an UNPARTITIONED table every file is in the single empty partition, so
    /// adding any file replaces ALL existing files. Append A, B; `replace_partitions().add_file(C)` →
    /// the live set is {C} only, the `replace-partitions` summary prop is set, and the cumulative totals
    /// are correct (2 prev - 2 removed + 1 added = 1, no underflow).
    ///
    /// Risk pinned: a full replace that fails to delete the old files (live set {A,B,C}) or underflows the
    /// running totals (the Rust `update_totals` is u64 — a wrong removed-count would panic).
    #[tokio::test]
    async fn test_replace_partitions_full_replace_on_unpartitioned_table() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_unpartitioned_table_in_catalog(&catalog).await;

        // Append A, B (2 files, 2 records). Running total-data-files = 2.
        let table = append_files(&catalog, &table, vec![
            unpartitioned_data_file("test/a.parquet"),
            unpartitioned_data_file("test/b.parquet"),
        ])
        .await;
        assert_eq!(total(&table, "total-data-files"), 2);

        // Replace: add C. The single (empty) partition is replaced → A and B removed.
        let tx = Transaction::new(&table);
        let action = tx
            .replace_partitions()
            .add_file(unpartitioned_data_file("test/c.parquet"));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/c.parquet".to_string()]),
            "unpartitioned replace is a FULL replace: only C survives"
        );
        // The replace-partitions marker is present.
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .additional_properties
                .get("replace-partitions")
                .map(String::as_str),
            Some("true"),
            "the replace-partitions summary property must be set (Java REPLACE_PARTITIONS_PROP)"
        );
        // Totals are correct: 2 - 2 + 1 = 1 (cumulative, no underflow).
        assert_eq!(
            total(&table, "total-data-files"),
            1,
            "full replace of 2 files adding 1 → total-data-files = 2 - 2 + 1 = 1"
        );
        assert_eq!(
            total(&table, "total-records"),
            1,
            "full replace → total-records = 2 - 2 + 1 = 1 (no underflow)"
        );
        // The deleted counts reflect the full replace.
        let summary = table.metadata().current_snapshot().unwrap().summary();
        assert_eq!(
            summary
                .additional_properties
                .get("deleted-data-files")
                .map(String::as_str),
            Some("2"),
            "the full replace reports both prior files as deleted"
        );
    }

    /// Pins: replacing a partition that has NO existing files is a PURE ADD (no spurious deletes, no
    /// error) — Java's `failMissingDeletePaths` guards only path deletes, never partition drops. Append
    /// A@x=0; replace add B@x=1 (an empty partition) → live set {A, B}: A in the untouched x=0 survives and
    /// B is added, with no error.
    ///
    /// Risk pinned: a by-partition resolution that wrongly errors on (or over-deletes for) a replaced
    /// partition with no existing files.
    #[tokio::test]
    async fn test_replace_empty_partition_is_pure_add() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // Replace x=1, which currently has no files → pure add of B; A in x=0 untouched.
        let tx = Transaction::new(&table);
        let action = tx
            .replace_partitions()
            .add_file(data_file("test/b.parquet", 1));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/a.parquet".to_string(), "test/b.parquet".to_string()]),
            "replacing an empty partition adds B and leaves A untouched"
        );
    }

    /// Pins: the replaced partition's old file appears as a Deleted tombstone in the rewritten manifest
    /// (so downstream tooling / expiry sees the removal), and the `replace-partitions` marker is present on
    /// a PARTITIONED replace too (not only the unpartitioned full-replace case).
    #[tokio::test]
    async fn test_replace_partition_marks_old_file_deleted_and_sets_marker() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/old.parquet", 0),
            data_file("test/keep.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .replace_partitions()
            .add_file(data_file("test/new.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // old.parquet is a Deleted tombstone.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut old_deleted = false;
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.file_path() == "test/old.parquet" {
                    assert_eq!(entry.status(), ManifestStatus::Deleted);
                    old_deleted = true;
                }
            }
        }
        assert!(
            old_deleted,
            "the replaced old.parquet must be a Deleted tombstone"
        );

        assert_eq!(
            snapshot
                .summary()
                .additional_properties
                .get("replace-partitions")
                .map(String::as_str),
            Some("true"),
            "the replace-partitions marker is present on a partitioned replace too"
        );
    }

    /// Pins: a `replace_partitions` with NO added files (and therefore no resolved deletes and no snapshot
    /// properties beyond the marker) does not silently produce an empty no-op snapshot. Java requires the
    /// commit to produce content; the producer's relaxed precondition still rejects a truly-empty commit.
    ///
    /// Note: the action always sets the `replace-partitions` property, so the "no added data files,
    /// deleted data files, or added snapshot properties" precondition is bypassed by that property. This
    /// test pins the resulting behavior (an empty replace produces a no-content Overwrite snapshot with
    /// only the marker) so a future change to the precondition does not silently alter it.
    #[tokio::test]
    async fn test_replace_partitions_with_no_added_files_adds_nothing() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx.replace_partitions();
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Nothing was added and no partition was named for replacement, so the existing file is untouched.
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/a.parquet".to_string()]),
            "a no-added-files replace must not delete or add anything"
        );
    }
}
