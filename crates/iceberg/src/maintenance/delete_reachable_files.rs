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

//! `DeleteReachableFiles` — given a table METADATA LOCATION, delete EVERY file reachable from it.
//! That set is all snapshots' manifest lists, all manifests, all data plus position-delete plus
//! equality-delete files plus deletion vectors, the current AND all previous `metadata.json`, the
//! version-hint, and all statistics plus partition-statistics files. The Rust port of Java's
//! `org.apache.iceberg.actions.DeleteReachableFiles` (api 1.10.0) /
//! `DeleteReachableFilesSparkAction` — the action behind `DROP TABLE PURGE` — minus the Spark
//! distribution layer.
//!
//! **THIS ACTION DELETES THE WHOLE TABLE.** It is the destructive COMPLEMENT of
//! [`DeleteOrphanFiles`](crate::maintenance::DeleteOrphanFiles): orphan-deletion lists storage and
//! removes files that *no* snapshot references (biasing toward UNDER-deletion to never touch live
//! data); reachable-deletion removes the reachable set ITSELF — the entire on-disk footprint of the
//! table. The two corruption classes are reversed: an OMISSION here leaks a file (an orphan left
//! behind after the table is gone), and any path computed here that does NOT belong to the table is
//! catastrophic over-deletion. Because the inputs are the table's own metadata (not a storage
//! listing), the reachable set is exact: it is precisely the set the table's metadata points at.
//!
//! # Java provenance and the 1.10.0 pin
//!
//! The action interface + its result shape are `iceberg-api` 1.10.0 (javap-verified):
//!
//! - `DeleteReachableFiles` (api 1.10.0) extends `Action<DeleteReachableFiles, Result>` with
//!   `deleteWith(Consumer<String>)`, `executeDeleteWith(ExecutorService)`, and `io(FileIO)`. The
//!   entry point (`SparkActions.deleteReachableFiles(String metadataLocation)`) takes the metadata
//!   LOCATION as a `String` — mirrored here as [`DeleteReachableFiles::new`]'s `&str`.
//! - `DeleteReachableFiles$Result` (api 1.10.0, javap-verified) is the six `long` counts mirrored
//!   1:1 by [`DeleteReachableFilesResult`]: `deletedDataFilesCount`,
//!   `deletedEqualityDeleteFilesCount`, `deletedPositionDeleteFilesCount`, `deletedManifestsCount`,
//!   `deletedManifestListsCount`, `deletedOtherFilesCount`.
//! - The reachable-file universe is `ReachableFileUtil` (core 1.10.0, javap-verified) —
//!   `metadataFileLocations(table, recursive)` (current + previous `metadata.json`),
//!   `manifestListLocations(table)`, `statisticsFilesLocations(table)`, `versionHintLocation(table)`
//!   — PLUS a scan of every snapshot's `allManifests` for the data/delete file paths (Java
//!   `DeleteReachableFilesSparkAction` composes `contentFileDS ∪ manifestDS ∪ manifestListDS ∪
//!   allReachableOtherMetadataFileDS`). The content-file walk reads EVERY manifest entry (incl.
//!   `DELETED` tombstones) of EVERY snapshot — a tombstoned file is still a physical file the table
//!   wrote, so it is reachable and must be deleted.
//!
//! The action CLASS itself lives in the Spark module (no 1.10.0 Spark bytecode is available
//! locally), so the categorization-into-counts and the metadata-location entry shape are pinned to
//! the tagless `DeleteReachableFilesSparkAction.java` MAIN source; every load-bearing helper it
//! delegates to (above) is the bytecode-verified `iceberg-core` / `iceberg-api` 1.10.0 surface. This
//! mirrors [`DeleteOrphanFiles`](crate::maintenance::DeleteOrphanFiles)'s provenance split exactly.
//!
//! # The reachable set (Java `ReachableFileUtil` + the `allManifests` content scan)
//!
//! Collected across **ALL** snapshots (Java purge covers every snapshot, not just the current one):
//!
//! 1. **Manifest lists** — one per snapshot (`snapshot.manifest_list()` /
//!    `ReachableFileUtil.manifestListLocations`).
//! 2. **Manifests** — every manifest of every snapshot (the manifest-list entries).
//! 3. **Data files** — every `Data`-content manifest entry of every snapshot, incl. DELETED
//!    tombstones.
//! 4. **Position-delete files + deletion vectors** — every `PositionDeletes`-content manifest entry
//!    (a Puffin DV is `PositionDeletes` content; Java's count folds DVs into the position-delete
//!    bucket — the count is content-type-keyed, not format-keyed).
//! 5. **Equality-delete files** — every `EqualityDeletes`-content manifest entry.
//! 6. **"Other" metadata** — the current `metadata.json` (the entry-point location), all PREVIOUS
//!    `metadata.json` files (the metadata-log entries — Java `metadataFileLocations(table, true)`),
//!    the version-hint location, and every statistics + partition-statistics file. Java's
//!    `deletedOtherFilesCount` is this bucket.
//!
//! `metadataFileLocations(table, recursive=true)` walks the previous-metadata chain transitively; in
//! this fork `TableMetadata::metadata_log()` already holds the full previous-files list (the
//! recursion is pre-materialized by the metadata writer), so iterating it is the recursive walk.
//!
//! # Categorization → the six Java counts
//!
//! Each reachable file is bucketed exactly once (a `HashSet` per bucket dedups; a path appearing in
//! two snapshots' manifests is one file, one deletion, one count). The buckets map 1:1 to the Java
//! `Result` counts. A delete failure does NOT decrement a count — Java counts the files it
//! IDENTIFIED for deletion (the count is the size of the reachable set), and per-file delete
//! failures are collected separately (the [`DeleteReachableFiles`] failure posture below), matching
//! the [`DeleteOrphanFiles`] posture.
//!
//! # Failure posture (Java parity)
//!
//! Java's `DeleteReachableFilesSparkAction` deletes via
//! `Tasks.foreach(...).suppressFailureWhenFinished()` — a per-file delete failure is logged and the
//! sweep continues; the returned counts come from the planned reachable set, not the delete outcome.
//! This port mirrors that: per-file delete failures are **collected** in
//! [`DeleteReachableFilesResult::delete_failures`] (the `iceberg` crate has no logging facade, and
//! silent swallowing is unacceptable for a deletion sweep) and the sweep continues; the six counts
//! always reflect the full reachable set.
//!
//! A **planning-stage** failure (an unreadable `metadata.json`, manifest list, or manifest) returns
//! `Err` BEFORE any deletion — the reachable set is computed in full first, so a read error can
//! never strand a half-deleted table.
//!
//! # Deferred (loudly)
//!
//! - **Executor parallelism** (Java `executeDeleteWith(ExecutorService)`): the sweep is SEQUENTIAL.
//! - **Bulk deletes** (`SupportsBulkOperations.deleteFiles`): the fork's [`FileIO`] has no bulk
//!   surface; deletes go one-by-one.
//! - **A `gc.enabled` gate:** Java's `DeleteReachableFiles` does NOT gate on `gc.enabled` (unlike
//!   `DeleteOrphanFiles`) — `DROP TABLE PURGE` removes a table the operator is explicitly dropping —
//!   so this port has no GC gate either.

use std::collections::HashSet;

use futures::future::BoxFuture;

use crate::error::Result;
use crate::io::FileIO;
use crate::spec::{DataContentType, TableMetadata};
use crate::table::Table;
use crate::{Error, TableIdent};

/// The injected delete function (Java `deleteWith(Consumer<String>)`): receives a file location,
/// resolves to a deletion outcome. The default deletes through [`FileIO::delete`].
pub type ReachableDeleteFunction = dyn Fn(String) -> BoxFuture<'static, Result<()>> + Send + Sync;

/// The six removed-file counts of a [`DeleteReachableFiles::execute`] sweep — a 1:1 mirror of Java's
/// `DeleteReachableFiles$Result` (api 1.10.0, javap-verified): six `long` accessors.
///
/// Each count is the SIZE of its reachable-file bucket (the files identified for deletion), NOT the
/// number successfully deleted — Java derives the counts from the planned set and logs-and-continues
/// on per-file delete failure (collected here in [`Self::delete_failures`]).
#[derive(Debug, Default)]
pub struct DeleteReachableFilesResult {
    /// Java `deletedDataFilesCount()` — every `Data`-content file of every snapshot.
    pub deleted_data_files_count: u64,
    /// Java `deletedEqualityDeleteFilesCount()` — every `EqualityDeletes`-content file.
    pub deleted_equality_delete_files_count: u64,
    /// Java `deletedPositionDeleteFilesCount()` — every `PositionDeletes`-content file (incl.
    /// deletion vectors, which are `PositionDeletes` content regardless of Puffin format).
    pub deleted_position_delete_files_count: u64,
    /// Java `deletedManifestsCount()` — every manifest of every snapshot.
    pub deleted_manifests_count: u64,
    /// Java `deletedManifestListsCount()` — every snapshot's manifest list.
    pub deleted_manifest_lists_count: u64,
    /// Java `deletedOtherFilesCount()` — current + previous `metadata.json`, version-hint, and all
    /// statistics + partition-statistics files.
    pub deleted_other_files_count: u64,
    /// Per-file delete failures collected during the sweep (Java logs-and-continues; this port
    /// collects). Empty means every reachable file deleted cleanly. NOT part of the Java `Result`
    /// (Java's logging facade has no equivalent), but the counts above DO match Java regardless of
    /// these (the counts are the planned-set sizes).
    pub delete_failures: Vec<ReachableDeleteFailure>,
}

impl DeleteReachableFilesResult {
    /// The total reachable-file count (the sum of the six buckets) — the number of files the action
    /// identified for deletion. Convenience for delete-completeness assertions.
    pub fn total_deleted_files_count(&self) -> u64 {
        self.deleted_data_files_count
            + self.deleted_equality_delete_files_count
            + self.deleted_position_delete_files_count
            + self.deleted_manifests_count
            + self.deleted_manifest_lists_count
            + self.deleted_other_files_count
    }
}

/// One collected, non-aborting delete failure (the Rust replacement for Java's log-and-continue).
#[derive(Debug)]
pub struct ReachableDeleteFailure {
    /// The reachable file whose deletion failed.
    pub path: String,
    /// The underlying error.
    pub error: Error,
}

/// The categorized reachable-file set: one bucket per Java `Result` count. Each bucket is a
/// `HashSet` so a file referenced by multiple snapshots is one entry (one deletion, one count).
#[derive(Debug, Default)]
struct ReachableFiles {
    data_files: HashSet<String>,
    equality_delete_files: HashSet<String>,
    position_delete_files: HashSet<String>,
    manifests: HashSet<String>,
    manifest_lists: HashSet<String>,
    other_files: HashSet<String>,
}

impl ReachableFiles {
    /// Every reachable path across all buckets, deterministically sorted (for the sweep + for tests).
    fn all_sorted(&self) -> Vec<String> {
        let mut all: Vec<String> = self
            .data_files
            .iter()
            .chain(self.equality_delete_files.iter())
            .chain(self.position_delete_files.iter())
            .chain(self.manifests.iter())
            .chain(self.manifest_lists.iter())
            .chain(self.other_files.iter())
            .cloned()
            .collect();
        all.sort();
        all.dedup();
        all
    }

    /// The result counts derived from the bucket sizes (Java's `Result` counts).
    fn counts(&self) -> DeleteReachableFilesResult {
        DeleteReachableFilesResult {
            deleted_data_files_count: self.data_files.len() as u64,
            deleted_equality_delete_files_count: self.equality_delete_files.len() as u64,
            deleted_position_delete_files_count: self.position_delete_files.len() as u64,
            deleted_manifests_count: self.manifests.len() as u64,
            deleted_manifest_lists_count: self.manifest_lists.len() as u64,
            deleted_other_files_count: self.other_files.len() as u64,
            delete_failures: Vec::new(),
        }
    }
}

/// An action that deletes EVERY file reachable from a table metadata location (the Rust port of
/// Java's `DeleteReachableFiles` — the engine behind `DROP TABLE PURGE`). See the module docs for
/// the reachable set, the Java provenance, and the failure posture.
///
/// **This action deletes the whole table.** Build it with [`DeleteReachableFiles::new`] (passing the
/// `metadata.json` LOCATION, matching Java's `String` arg), optionally override the
/// [`FileIO`](Self::io) or the [`delete function`](Self::delete_with), and run it with
/// [`Self::execute`].
pub struct DeleteReachableFiles {
    metadata_location: String,
    file_io: Option<FileIO>,
    delete_function: Option<Box<ReachableDeleteFunction>>,
}

impl DeleteReachableFiles {
    /// Create a `DeleteReachableFiles` action for the table whose current metadata is at
    /// `metadata_location` (Java `SparkActions.deleteReachableFiles(String metadataLocation)`).
    ///
    /// The `FileIO` defaults to a local-filesystem `FileIO` ([`FileIO::new_with_fs`]); set a
    /// different one with [`Self::io`] (Java's required `io(FileIO)` — for non-local storage the
    /// caller MUST supply the table's `FileIO`). The delete function defaults to
    /// [`FileIO::delete`]; override it with [`Self::delete_with`] (Java `deleteWith`).
    pub fn new(metadata_location: impl Into<String>) -> Self {
        DeleteReachableFiles {
            metadata_location: metadata_location.into(),
            file_io: None,
            delete_function: None,
        }
    }

    /// Set the [`FileIO`] used to read the metadata + delete files (Java `io(FileIO)`). REQUIRED for
    /// any non-local-filesystem storage; the default is a local-FS `FileIO`.
    pub fn io(mut self, file_io: FileIO) -> Self {
        self.file_io = Some(file_io);
        self
    }

    /// Replace the delete function (Java `deleteWith(Consumer<String>)`). The default deletes
    /// through [`FileIO::delete`]. A custom function receives exactly the reachable set (e.g. to
    /// collect the set without deleting, or route deletions through an external queue).
    pub fn delete_with(
        mut self,
        delete_function: impl Fn(String) -> BoxFuture<'static, Result<()>> + Send + Sync + 'static,
    ) -> Self {
        self.delete_function = Some(Box::new(delete_function));
        self
    }

    /// Run the action: compute the full reachable set across all snapshots, then delete every file.
    /// See the module docs for the reachable set and the failure posture.
    ///
    /// Returns `Err` WITHOUT deleting anything when the metadata, a manifest list, or a manifest
    /// cannot be read during planning. Per-file delete failures are collected in the returned
    /// [`DeleteReachableFilesResult`]; the six counts always reflect the full reachable set.
    pub async fn execute(self) -> Result<DeleteReachableFilesResult> {
        let file_io = self.resolve_file_io();

        // Load the table from its metadata location (Java's String arg shape) as a read-only static
        // table — the reachable walk reads metadata only, no catalog binding.
        let metadata = TableMetadata::read_from(&file_io, &self.metadata_location).await?;
        let table = Table::builder()
            .metadata(metadata)
            .metadata_location(self.metadata_location.clone())
            // A synthetic read-only identity (the action never binds a catalog; the walk reads
            // metadata only). `TableIdent` requires a non-empty namespace, hence the two components.
            .identifier(
                TableIdent::from_strs(["delete_reachable_files", "table"]).map_err(|error| {
                    error.with_context(
                        "reason",
                        "building the internal reachable-files table identity",
                    )
                })?,
            )
            .file_io(file_io.clone())
            .readonly(true)
            .build()?;

        // 1. The full reachable set across ALL snapshots + metadata (read-only; aborts before any
        //    deletion on a read error).
        let reachable = collect_reachable_files(&table).await?;

        // 2. Delete every reachable file (collecting per-file failures), then report the counts.
        let mut result = reachable.counts();
        for path in reachable.all_sorted() {
            let outcome = match &self.delete_function {
                Some(delete) => delete(path.clone()).await,
                None => file_io.delete(&path).await,
            };
            if let Err(error) = outcome {
                result
                    .delete_failures
                    .push(ReachableDeleteFailure { path, error });
            }
        }
        Ok(result)
    }

    /// The configured [`FileIO`], or a local-filesystem default (Java requires `io(FileIO)`; the
    /// default here mirrors `DeleteOrphanFiles`/`StaticTable`'s local-FS convenience).
    fn resolve_file_io(&self) -> FileIO {
        self.file_io.clone().unwrap_or_else(FileIO::new_with_fs)
    }
}

/// Build the categorized reachable-file set for `table` across ALL snapshots (Java's
/// `ReachableFileUtil` set ∪ the `allManifests` content scan). Shares the SAME walk shape as
/// [`DeleteOrphanFiles`](crate::maintenance::DeleteOrphanFiles)'s `collect_valid_files` (manifest
/// lists → manifests → entries; + current/previous metadata.json, version-hint, statistics), but
/// keeps the files CATEGORIZED so the six Java `Result` counts can be derived. The orphan collector
/// returns a flat set (it only needs membership); this one needs per-bucket sizes — hence a separate
/// categorizing collector rather than reusing the flat one (the orphan walk is intentionally left
/// untouched).
///
/// A manifest-list or manifest read failure aborts with `Err` BEFORE any deletion (this runs inside
/// [`DeleteReachableFiles::execute`] strictly before the sweep).
async fn collect_reachable_files(table: &Table) -> Result<ReachableFiles> {
    let metadata = table.metadata();
    let file_io = table.file_io();
    let mut reachable = ReachableFiles::default();

    for snapshot in metadata.snapshots() {
        // The manifest list of this snapshot (Java `manifestListLocations`).
        reachable
            .manifest_lists
            .insert(snapshot.manifest_list().to_string());

        let manifest_list = snapshot
            .load_manifest_list(file_io, metadata)
            .await
            .map_err(|error| {
                error.with_context(
                    "snapshot_id",
                    format!(
                        "failed to read manifest list of snapshot {} while planning \
                         delete-reachable-files (no files were deleted)",
                        snapshot.snapshot_id()
                    ),
                )
            })?;

        for manifest_file in manifest_list.entries() {
            // The manifest file itself (Java `manifestDS` over ALL_MANIFESTS).
            reachable
                .manifests
                .insert(manifest_file.manifest_path.clone());

            // Every content file of every entry — INCLUDING DELETED tombstones (Java reads via
            // ManifestFiles.read, not liveEntries; a tombstoned file is still a physical file the
            // table wrote and must be deleted). Bucket by content type for the Java counts.
            let manifest = manifest_file
                .load_manifest(file_io)
                .await
                .map_err(|error| {
                    error.with_context(
                        "manifest_path",
                        format!(
                            "failed to read manifest {} while planning delete-reachable-files \
                             (no files were deleted)",
                            manifest_file.manifest_path
                        ),
                    )
                })?;
            for entry in manifest.entries() {
                let path = entry.file_path().to_string();
                match entry.content_type() {
                    DataContentType::Data => {
                        reachable.data_files.insert(path);
                    }
                    DataContentType::PositionDeletes => {
                        // Position-delete files AND deletion vectors (a DV is PositionDeletes
                        // content with Puffin format; Java folds DVs into this bucket).
                        reachable.position_delete_files.insert(path);
                    }
                    DataContentType::EqualityDeletes => {
                        reachable.equality_delete_files.insert(path);
                    }
                }
            }
        }
    }

    // "Other" metadata files (Java `deletedOtherFilesCount`): the current metadata.json, all
    // PREVIOUS metadata.json (the metadata-log — Java metadataFileLocations(table, recursive=true)),
    // the version-hint, and all statistics + partition-statistics.
    reachable
        .other_files
        .insert(table.metadata_location_result()?.to_string());
    for log_entry in metadata.metadata_log() {
        reachable
            .other_files
            .insert(log_entry.metadata_file.clone());
    }
    reachable
        .other_files
        .insert(version_hint_location(metadata.location()));
    for statistics in metadata.statistics_iter() {
        reachable
            .other_files
            .insert(statistics.statistics_path.clone());
    }
    for statistics in metadata.partition_statistics_iter() {
        reachable
            .other_files
            .insert(statistics.statistics_path.clone());
    }

    Ok(reachable)
}

/// The version-hint file location (Java `ReachableFileUtil.versionHintLocation`:
/// `<location>/metadata/version-hint.text`). Java always adds it to the reachable set even for
/// non-Hadoop tables, so a stray hint file is always cleaned by a purge.
fn version_hint_location(table_location: &str) -> String {
    let trimmed = table_location.strip_suffix('/').unwrap_or(table_location);
    format!("{trimmed}/metadata/version-hint.text")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use futures::FutureExt;

    use super::*;
    use crate::io::{FileIOBuilder, LocalFsStorageFactory};
    use crate::memory::MemoryCatalogBuilder;
    use crate::spec::{
        DataFile, DataFileBuilder, DataFileFormat, NestedField, PartitionSpec, PrimitiveType,
        Schema, StatisticsFile, Struct, Type,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};

    /// A `MemoryCatalog` backed by a real local filesystem under a `TempDir`, plus a matching
    /// `FileIO` for planting/inspecting files. Returns the temp-dir guard (kept alive by the caller).
    async fn local_fs_catalog() -> (impl Catalog, FileIO, tempfile::TempDir) {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let warehouse = temp_dir
            .path()
            .to_str()
            .expect("utf8 temp path")
            .to_string();
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "memory",
                HashMap::from([("warehouse".to_string(), warehouse)]),
            )
            .await
            .expect("load local-fs memory catalog");
        let file_io = FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build();
        (catalog, file_io, temp_dir)
    }

    /// A minimal two-long-column schema `{1 x long, 2 y long}`.
    fn two_long_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "x",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::required(
                    2,
                    "y",
                    Type::Primitive(PrimitiveType::Long),
                )),
            ])
            .build()
            .expect("build schema")
    }

    /// Create an unpartitioned table under a fresh namespace.
    async fn create_table(catalog: &impl Catalog) -> Table {
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace");
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(two_long_schema())
            .partition_spec(PartitionSpec::unpartition_spec())
            .build();
        catalog
            .create_table(&namespace, creation)
            .await
            .expect("create table")
    }

    /// Write `content` to `path` through `file_io` (creates parent dirs on the local fs).
    async fn write_real_file(file_io: &FileIO, path: &str, content: &[u8]) {
        file_io
            .new_output(path)
            .expect("new output")
            .write(Bytes::copy_from_slice(content))
            .await
            .expect("write file");
    }

    /// A real `DataFile` of the given content type: write `content` to `path` on disk, then build a
    /// metadata-consistent unpartitioned descriptor.
    async fn real_file(
        file_io: &FileIO,
        path: &str,
        content: &[u8],
        content_type: DataContentType,
    ) -> DataFile {
        write_real_file(file_io, path, content).await;
        let mut builder = DataFileBuilder::default();
        builder
            .content(content_type)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(content.len() as u64)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty());
        if content_type != DataContentType::Data {
            // Delete files carry a referenced data file / equality ids in real tables; for the
            // reachable walk only the PATH + content type matter, so the minimal descriptor is fine.
        }
        builder.build().expect("build data file")
    }

    /// Append `files` to `table` via a fast append, committed through `catalog`.
    async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let tx = tx
            .fast_append()
            .add_data_files(files)
            .apply(tx)
            .expect("apply fast append");
        tx.commit(catalog).await.expect("commit fast append")
    }

    /// True iff `path` exists on disk through `file_io`.
    async fn exists(file_io: &FileIO, path: &str) -> bool {
        file_io.exists(path).await.expect("exists check")
    }

    // ---- the reachable-set categorization (mutation-mindset) ----------------------------------

    /// A multi-snapshot table with a data file, a previous metadata.json (every commit advances the
    /// log), and a statistics file: the reachable categories must be exactly right.
    #[tokio::test]
    async fn reachable_set_categorizes_every_file_category() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // s1: a data file. (Committing advances the metadata log, creating a previous metadata.json.)
        let d1 = real_file(
            &file_io,
            &format!("{location}/data/d1.parquet"),
            b"d1",
            DataContentType::Data,
        )
        .await;
        let table = append(&catalog, &table, vec![d1]).await;
        // s2: a second data file (a second snapshot ⇒ a second manifest list + a second metadata.json).
        let d2 = real_file(
            &file_io,
            &format!("{location}/data/d2.parquet"),
            b"d2",
            DataContentType::Data,
        )
        .await;
        let table = append(&catalog, &table, vec![d2]).await;

        // Plant + register a statistics file (the "other" bucket).
        let stats_path = format!("{location}/metadata/stats.puffin");
        write_real_file(&file_io, &stats_path, b"stats").await;
        let snapshot_id = table
            .metadata()
            .current_snapshot()
            .expect("snapshot")
            .snapshot_id();
        let stats = StatisticsFile {
            snapshot_id,
            statistics_path: stats_path.clone(),
            file_size_in_bytes: 5,
            file_footer_size_in_bytes: 1,
            key_metadata: None,
            blob_metadata: Vec::new(),
        };
        let tx = Transaction::new(&table);
        let tx = tx
            .update_statistics()
            .set_statistics(stats)
            .apply(tx)
            .expect("apply set statistics");
        let table = tx.commit(&catalog).await.expect("commit set statistics");

        let reachable = collect_reachable_files(&table)
            .await
            .expect("collect reachable");

        // TWO data files (one per snapshot).
        assert_eq!(reachable.data_files.len(), 2, "two data files reachable");
        // TWO snapshots ⇒ two manifest lists.
        assert_eq!(
            reachable.manifest_lists.len(),
            2,
            "two manifest lists (one per snapshot)"
        );
        // At least two manifests (each fast append writes a manifest).
        assert!(
            reachable.manifests.len() >= 2,
            "at least two manifests reachable, got {}",
            reachable.manifests.len()
        );
        // The "other" bucket: current metadata.json + previous metadata.json(s) + version-hint +
        // the statistics file. At minimum: current + version-hint + stats + ≥1 previous = ≥4.
        assert!(
            reachable.other_files.contains(&stats_path),
            "the statistics file is in the other bucket"
        );
        assert!(
            reachable
                .other_files
                .contains(&version_hint_location(&location)),
            "the version-hint is in the other bucket"
        );
        assert!(
            reachable
                .other_files
                .contains(table.metadata_location_result().expect("metadata location")),
            "the current metadata.json is in the other bucket"
        );
        assert!(
            reachable.other_files.len() >= 4,
            "current + previous metadata.json + version-hint + stats ≥ 4, got {}",
            reachable.other_files.len()
        );
        // No deletes in this fixture.
        assert_eq!(reachable.position_delete_files.len(), 0);
        assert_eq!(reachable.equality_delete_files.len(), 0);
    }

    /// MUTATION-MINDSET: position-delete and equality-delete entries land in their OWN buckets (a
    /// walk that dropped the delete categories — or wrongly bucketed them as data — fails here).
    #[tokio::test]
    async fn reachable_set_buckets_position_and_equality_deletes() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let data = real_file(
            &file_io,
            &format!("{location}/data/d.parquet"),
            b"d",
            DataContentType::Data,
        )
        .await;
        let pos = real_file(
            &file_io,
            &format!("{location}/data/pos-deletes.parquet"),
            b"pos",
            DataContentType::PositionDeletes,
        )
        .await;
        let eq = real_file(
            &file_io,
            &format!("{location}/data/eq-deletes.parquet"),
            b"eq",
            DataContentType::EqualityDeletes,
        )
        .await;

        // Commit the data, then a row-delta carrying both delete files.
        let table = append(&catalog, &table, vec![data]).await;
        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_deletes(vec![pos, eq])
            .apply(tx)
            .expect("apply row delta");
        let table = tx.commit(&catalog).await.expect("commit row delta");

        let reachable = collect_reachable_files(&table)
            .await
            .expect("collect reachable");
        assert_eq!(reachable.data_files.len(), 1, "one data file");
        assert_eq!(
            reachable.position_delete_files.len(),
            1,
            "one position-delete file in its own bucket"
        );
        assert_eq!(
            reachable.equality_delete_files.len(),
            1,
            "one equality-delete file in its own bucket"
        );
        // The counts mirror the buckets 1:1.
        let counts = reachable.counts();
        assert_eq!(counts.deleted_data_files_count, 1);
        assert_eq!(counts.deleted_position_delete_files_count, 1);
        assert_eq!(counts.deleted_equality_delete_files_count, 1);
    }

    // ---- the delete sweep: completeness + nothing-extra (the two corruption classes) ----------

    /// After `execute`, EVERY reachable file is gone from FileIO; the six counts equal the reachable
    /// bucket sizes; and a file OUTSIDE the reachable set is untouched (over-delete = data loss).
    #[tokio::test]
    async fn execute_deletes_every_reachable_file_and_nothing_outside() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let d1 = real_file(
            &file_io,
            &format!("{location}/data/d1.parquet"),
            b"d1",
            DataContentType::Data,
        )
        .await;
        let table = append(&catalog, &table, vec![d1]).await;
        let d2 = real_file(
            &file_io,
            &format!("{location}/data/d2.parquet"),
            b"d2",
            DataContentType::Data,
        )
        .await;
        let table = append(&catalog, &table, vec![d2]).await;

        // A file OUTSIDE the table footprint — must survive (it is not reachable).
        let outsider = format!("{location}/data/not-ours.txt");
        write_real_file(&file_io, &outsider, b"keep me").await;

        // Snapshot the reachable set first (for the completeness assertion), then execute.
        let metadata_location = table
            .metadata_location_result()
            .expect("metadata location")
            .to_string();
        let reachable = collect_reachable_files(&table)
            .await
            .expect("collect reachable");
        let all_reachable = reachable.all_sorted();
        assert!(
            all_reachable.iter().all(|p| !p.ends_with("not-ours.txt")),
            "the outsider must NOT be in the reachable set"
        );

        let result = DeleteReachableFiles::new(&metadata_location)
            .io(file_io.clone())
            .execute()
            .await
            .expect("execute delete reachable files");

        // Every reachable file is physically gone.
        for path in &all_reachable {
            assert!(
                !exists(&file_io, path).await,
                "reachable file must be deleted: {path}"
            );
        }
        // The outsider survives (over-delete = data loss).
        assert!(
            exists(&file_io, &outsider).await,
            "a file outside the reachable set must NOT be deleted"
        );
        // The counts equal the reachable bucket sizes (no delete failures on a clean local FS).
        assert!(
            result.delete_failures.is_empty(),
            "no delete failures expected"
        );
        assert_eq!(result.deleted_data_files_count, 2, "two data files deleted");
        assert_eq!(
            result.deleted_manifest_lists_count, 2,
            "two manifest lists deleted"
        );
        assert_eq!(
            result.total_deleted_files_count(),
            all_reachable.len() as u64,
            "total count == reachable-set size"
        );
    }

    /// The `delete_with` consumer receives EXACTLY the reachable set (Java `deleteWith`): a
    /// collect-only run deletes nothing but reports the full set + counts.
    #[tokio::test]
    async fn delete_with_collects_exactly_the_reachable_set() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let d1 = real_file(
            &file_io,
            &format!("{location}/data/d1.parquet"),
            b"d1",
            DataContentType::Data,
        )
        .await;
        let table = append(&catalog, &table, vec![d1]).await;
        let metadata_location = table
            .metadata_location_result()
            .expect("metadata location")
            .to_string();
        let reachable_expected = collect_reachable_files(&table)
            .await
            .expect("collect reachable")
            .all_sorted();

        let collected: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = collected.clone();
        let result = DeleteReachableFiles::new(&metadata_location)
            .io(file_io.clone())
            .delete_with(move |path| {
                let sink = sink.clone();
                async move {
                    sink.lock().expect("lock").push(path);
                    Ok(())
                }
                .boxed()
            })
            .execute()
            .await
            .expect("execute collect-only");

        let mut got = collected.lock().expect("lock").clone();
        got.sort();
        assert_eq!(
            got, reachable_expected,
            "the consumer receives exactly the reachable set"
        );
        // Nothing was physically deleted (collect-only) — the data file still exists.
        assert!(
            exists(&file_io, &format!("{location}/data/d1.parquet")).await,
            "collect-only deletes nothing"
        );
        // The counts still reflect the full set.
        assert_eq!(
            result.total_deleted_files_count(),
            reachable_expected.len() as u64
        );
    }

    /// EDGE: a freshly-created table with NO snapshots is still purgeable — its reachable set is the
    /// metadata.json + version-hint (no manifests/data), and execute removes them cleanly.
    #[tokio::test]
    async fn empty_table_purges_metadata_only() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_table(&catalog).await;
        let metadata_location = table
            .metadata_location_result()
            .expect("metadata location")
            .to_string();

        let reachable = collect_reachable_files(&table)
            .await
            .expect("collect reachable");
        assert_eq!(
            reachable.data_files.len(),
            0,
            "no data files on a fresh table"
        );
        assert_eq!(
            reachable.manifest_lists.len(),
            0,
            "no snapshots ⇒ no manifest lists"
        );
        assert!(
            reachable.other_files.contains(&metadata_location),
            "the current metadata.json is reachable on a fresh table"
        );

        let result = DeleteReachableFiles::new(&metadata_location)
            .io(file_io.clone())
            .execute()
            .await
            .expect("execute on empty table");
        assert!(result.delete_failures.is_empty());
        assert_eq!(result.deleted_data_files_count, 0);
        assert_eq!(result.deleted_manifests_count, 0);
        assert!(
            result.deleted_other_files_count >= 1,
            "at least the current metadata.json is removed"
        );
        // The metadata.json is physically gone.
        assert!(
            !exists(&file_io, &metadata_location).await,
            "the metadata.json must be deleted"
        );
    }

    /// A delete failure on one file does NOT abort the sweep and does NOT change the counts (Java
    /// logs-and-continues; the counts are the planned-set sizes).
    #[tokio::test]
    async fn delete_failures_are_collected_not_fatal() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let d1 = real_file(
            &file_io,
            &format!("{location}/data/d1.parquet"),
            b"d1",
            DataContentType::Data,
        )
        .await;
        let table = append(&catalog, &table, vec![d1]).await;
        let metadata_location = table
            .metadata_location_result()
            .expect("metadata location")
            .to_string();
        let reachable_size = collect_reachable_files(&table)
            .await
            .expect("collect reachable")
            .all_sorted()
            .len();

        // A delete function that fails for exactly one path but succeeds for the rest.
        let failing = format!("{location}/data/d1.parquet");
        let failing_for_closure = failing.clone();
        let result = DeleteReachableFiles::new(&metadata_location)
            .io(file_io.clone())
            .delete_with(move |path| {
                let failing = failing_for_closure.clone();
                async move {
                    if path == failing {
                        Err(Error::new(
                            ErrorKind::Unexpected,
                            "simulated delete failure",
                        ))
                    } else {
                        Ok(())
                    }
                }
                .boxed()
            })
            .execute()
            .await
            .expect("execute with one failing delete");

        assert_eq!(
            result.delete_failures.len(),
            1,
            "exactly one collected delete failure"
        );
        assert_eq!(result.delete_failures[0].path, failing);
        assert_eq!(
            result.total_deleted_files_count(),
            reachable_size as u64,
            "the counts are the planned-set size regardless of the failure"
        );
    }
}
