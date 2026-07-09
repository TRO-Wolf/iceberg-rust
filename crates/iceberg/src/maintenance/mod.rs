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

//! Table maintenance actions — engine-agnostic ports of Java's `org.apache.iceberg.actions`
//! that operate on an already-committed [`Table`](crate::table::Table) rather than producing a
//! new snapshot through the [`transaction`](crate::transaction) seam.
//!
//! Unlike a [`TransactionAction`](crate::transaction::TransactionAction) (which appends/rewrites
//! metadata and commits through the catalog), a maintenance action reads the table's current
//! metadata + physical storage and performs out-of-band work — here, physically deleting files
//! that no live snapshot references.
//!
//! # Contents
//!
//! - [`DeleteOrphanFiles`] — list a location and delete files unreachable from any valid
//!   snapshot (the Rust port of Java's `DeleteOrphanFiles` Spark action, minus the Spark
//!   distribution layer). **This action deletes files.**
//! - [`DeleteReachableFiles`] — given a table metadata LOCATION, delete EVERY file reachable from it
//!   (all snapshots' manifest lists + manifests + data/position-delete/equality-delete files + DVs,
//!   current + all previous `metadata.json`, version-hint, and statistics + partition-statistics).
//!   The Rust port of Java's `DeleteReachableFiles` (the engine behind `DROP TABLE PURGE`), with the
//!   six-count [`DeleteReachableFilesResult`] mirroring Java's `Result`. The destructive COMPLEMENT
//!   of [`DeleteOrphanFiles`] (it deletes the reachable set ITSELF). **This action deletes the whole
//!   table.**
//! - [`RewriteDataFiles`] — bin-pack compaction: plan small-file groups per partition, read each
//!   group's live rows (merge-on-read deletes applied), and rewrite them into target-sized files
//!   committed through the seq-preserving [`RewriteFilesAction`](crate::transaction::rewrite_files).
//!   The Rust port of Java's `RewriteDataFiles` bin-pack strategy. **This action rewrites data.**
//! - [`RemoveDanglingDeleteFiles`] — remove delete files that can no longer apply to any live data
//!   file (per-partition/spec min-data-seq comparison + DV reference check), committed through the
//!   [`RewriteFilesAction`](crate::transaction::rewrite_files) delete-file-removal surface. The Rust
//!   port of Java's `RemoveDanglingDeletesSparkAction`. **This action removes delete files.**
//! - [`ComputePartitionStats`] — the engine-agnostic action wrapper (the Rust port of Java 1.10.0
//!   `ComputePartitionStats` / `BaseComputePartitionStats`): `new(table).snapshot_id(id).execute(catalog)`
//!   resolves the snapshot, runs the compute/write core, and registers the file through the
//!   [`UpdatePartitionStatisticsAction`](crate::transaction::Transaction::update_partition_statistics)
//!   seam, returning a [`ComputePartitionStatsResult`].
//! - [`compute_partition_stats`] / [`PartitionStats`] — the `ComputePartitionStats` compute core (the
//!   Rust port of Java 1.10.0 `PartitionStatsHandler`'s full-compute aggregation): per-partition
//!   statistics rolled up over a snapshot's manifests into the Java-exact partition-stats schema.
//!   [`compute_and_write_stats_file`] writes those rows to an on-disk partition-stats parquet file at
//!   Java's location/naming (`<location>/metadata/partition-stats-<snapshotId>-<uuid>.parquet`) with the
//!   field ids 1..=13 stamped; [`register_partition_stats_file`] commits it into the table metadata via
//!   the `UpdatePartitionStatisticsAction` seam (`SetPartitionStatistics`); [`read_partition_stats_file`]
//!   decodes a written file back into rows. See [`partition_stats`] for the schema + traversal + on-disk
//!   format.
//! - [`ConvertEqualityDeleteFiles`] — materialize every live EQUALITY-delete file in the current
//!   snapshot into an equivalent POSITION-delete file and commit the swap in one `Replace` snapshot (the
//!   Rust port of Java 1.10.0 `ConvertEqualityDeleteFiles`). For each eq-delete: build its predicate from
//!   its on-disk tuples, find the applicable lower-data-seq same-partition (global if unpartitioned) data
//!   files, read each with an ABSOLUTE `_pos`, collect the MATCHING rows, and write a position-delete file
//!   stamped (via `add_delete_file_with_sequence_number`) with the eq-delete's data seq so the converted
//!   delete masks exactly the same rows. A FREE-STANDING action (not an `ActionsProvider` method, per
//!   Java's javap-confirmed 12-method surface). Returns a [`ConvertEqualityDeleteFilesResult`] mirroring
//!   Java `Result`'s two `int` counts. **This action rewrites delete files.**
//! - [`RewritePositionDeleteFiles`] — COMPACT the live PARQUET position-delete files in the current
//!   snapshot, per `(spec, partition)` group, into FEWER position-delete files and commit the swap in one
//!   `Replace` snapshot per group (the Rust port of Java 1.10.0 `RewritePositionDeleteFiles`). For each
//!   group of 2+ live parquet pos-deletes: read every member's `(file_path, pos)` pairs by RESERVED FIELD
//!   ID, concat + sort, write one compacted file, and commit ONE `RewriteFiles` STAMPING the compacted
//!   file with the group MAX rewritten data seq (via `add_delete_file_with_sequence_number`, NOT inherit)
//!   so it masks exactly the same data generation. Puffin V3 DELETION VECTORS are SKIPPED (file-scoped,
//!   never bin-packed) — V2 PARQUET only (documented divergence, GAP_MATRIX row R136). A STRICT SUBSET of
//!   `ConvertEqualityDeleteFiles` (no row matching / predicate inversion / tuple parsing). One of the
//!   twelve `ActionsProvider` methods (`rewrite_position_deletes(Table)`). Returns a
//!   [`RewritePositionDeleteFilesResult`] mirroring Java `Result`'s four counts (rewritten/added file
//!   counts + rewritten/added byte counts). **This action rewrites delete files.**
//! - [`ComputeTableStats`] — per-column NDV (number of distinct values) via Apache DataSketches theta
//!   sketches (the [`iceberg_sketches`] crate), written as one `apache-datasketches-theta-v1` Puffin
//!   blob per column into a single statistics file and registered through the existing
//!   [`UpdateStatisticsAction`](crate::transaction::Transaction::update_statistics). The Rust port of
//!   Java's `ComputeTableStats` action; each value is fed to the sketch in Iceberg single-value
//!   serialization form (`Conversions.toByteBuffer`). See [`compute_table_stats`].
//! - [`RewriteTablePath`] — FULL-rewrite mode: rewrite every absolute path prefix in a table's metadata
//!   graph (metadata.json + manifest-lists + manifests + position-delete CONTENT) from `source` to
//!   `target`, STAGING the rewritten metadata at a caller-chosen location and emitting a `(source,
//!   target)` COPY-PLAN ([`RewriteTablePathResult`]). The Rust port of Java 1.10.0's engine-agnostic
//!   core `RewriteTablePathUtil` (`replacePaths` / `rewriteManifestList` / `rewriteDataManifest` /
//!   `rewriteDeleteManifest` / `rewritePositionDeleteFile`). Mirrors the 1.10.0 divergences EXACTLY:
//!   `location` via regex `replaceFirst`; `partition_statistics` PASSED THROUGH un-rewritten;
//!   `referenced_data_file` rewritten; position-delete content is the ONLY rewritten payload (+
//!   `replacePathBounds`); copy-plan direction differs by class (STAGED files copy FROM staging, VERBATIM
//!   files copy FROM source). One of the twelve `ActionsProvider` methods (`rewrite_table_path(Table)`).
//!   Incremental mode + version-hint are DEFERRED. **This action STAGES rewritten metadata + emits a
//!   copy-plan; it does NOT physically copy data files.**
//! - [`Actions`](crate::maintenance::Actions) / [`ActionsProvider`](crate::maintenance::ActionsProvider)
//!   — the factory surface mirroring Java's `org.apache.iceberg.actions.ActionsProvider` (1.10.0): one
//!   entry point that hands out the maintenance actions above (constructed `X::new(table)`) AND the
//!   transaction-seam actions ([`ExpireSnapshotsAction`](crate::transaction::ExpireSnapshotsAction) /
//!   [`RewriteManifestsAction`](crate::transaction::RewriteManifestsAction)), bridging both
//!   construction idioms. Java methods with no Rust action surface as a typed
//!   [`NoAction`](crate::maintenance::NoAction)-returning `FeatureUnsupported` (never faked).
//!
//! # Relationship to `transaction::expire_cleanup`
//!
//! [`ExpireSnapshotsCleanup`](crate::transaction::ExpireSnapshotsCleanup) deletes files made
//! unreachable by a *specific* expire-snapshots commit (a `before − after` reachability delta).
//! `DeleteOrphanFiles` is the complementary safety net: it deletes files present in storage that
//! *no* valid snapshot references at all — files leaked by failed writes, interrupted compactions,
//! or non-Iceberg processes. The two derive their valid-file universe differently (see
//! [`DeleteOrphanFiles`] for why this module re-derives the *full* reachable set instead of
//! reusing `expire_cleanup`'s delta machinery).

mod actions_provider;
mod compute_partition_stats;
mod compute_table_stats;
mod convert_equality_delete_files;
mod delete_orphan_files;
mod delete_reachable_files;
pub mod partition_stats;
mod remove_dangling_delete_files;
mod rewrite_data_files;
mod rewrite_position_delete_files;
mod rewrite_table_path;

#[cfg(test)]
mod tests;

pub use actions_provider::{Actions, ActionsProvider, NoAction};
pub use compute_partition_stats::{ComputePartitionStats, ComputePartitionStatsResult};
pub use compute_table_stats::{ComputeTableStats, ComputeTableStatsResult};
pub use convert_equality_delete_files::{
    ConvertEqualityDeleteFiles, ConvertEqualityDeleteFilesResult,
};
pub use delete_orphan_files::{DeleteOrphanFiles, DeleteOrphanFilesResult, PrefixMismatchMode};
pub use delete_reachable_files::{
    DeleteReachableFiles, DeleteReachableFilesResult, ReachableDeleteFailure,
    ReachableDeleteFunction,
};
pub use partition_stats::{
    PartitionStats, compute_and_write_stats_file, compute_partition_stats, partition_stats_schema,
    read_partition_stats_file, register_partition_stats_file, unified_partition_type,
};
pub use remove_dangling_delete_files::{
    RemoveDanglingDeleteFiles, RemoveDanglingDeleteFilesResult,
};
pub use rewrite_data_files::{FileGroupRewriteResult, RewriteDataFiles, RewriteDataFilesResult};
pub use rewrite_position_delete_files::{
    RewritePositionDeleteFiles, RewritePositionDeleteFilesResult,
};
pub use rewrite_table_path::{RewriteTablePath, RewriteTablePathResult};
