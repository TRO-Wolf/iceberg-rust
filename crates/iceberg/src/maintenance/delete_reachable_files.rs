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

//! `DeleteReachableFiles` deletes every file that a table metadata location reaches. It ports Java
//! `DeleteReachableFiles` (api 1.10.0) and `DeleteReachableFilesSparkAction`, the action behind
//! `DROP TABLE PURGE`.
//!
//! **THIS ACTION DELETES THE WHOLE TABLE.** It is the destructive complement of
//! [`DeleteOrphanFiles`](crate::maintenance::DeleteOrphanFiles), and the corruption classes
//! reverse. An omission here only leaks a file. Any path in the set that does not belong to the
//! table is catastrophic over-deletion. The input is the table's own metadata, never a storage
//! listing, so the reachable set is exact.
//!
//! The set covers ALL snapshots, because a Java purge covers every snapshot. Each bucket maps 1:1
//! to a Java `DeleteReachableFiles$Result` count:
//!
//! | Bucket | Contents |
//! |---|---|
//! | `deletedManifestListsCount` | one manifest list per snapshot |
//! | `deletedManifestsCount` | every manifest of every snapshot |
//! | `deletedDataFilesCount` | every `Data` entry, DELETED tombstones included |
//! | `deletedPositionDeleteFilesCount` | every `PositionDeletes` entry, deletion vectors included |
//! | `deletedEqualityDeleteFilesCount` | every `EqualityDeletes` entry |
//! | `deletedOtherFilesCount` | current + previous `metadata.json`, version-hint, statistics files |
//!
//! A tombstoned file is still a physical file the table wrote, so the walk reads every manifest
//! entry, not the live entries alone. A count is the size of the planned set, not of the delete
//! outcome, as in Java. `TableMetadata::metadata_log()` already holds the whole previous-metadata
//! chain, so iterating it is Java's recursive `metadataFileLocations(table, true)` walk.
//!
//! Java deletes under `suppressFailureWhenFinished()` and logs each per-file failure. This port
//! collects them in [`DeleteReachableFilesResult::delete_failures`], because the crate has no
//! logging facade and a silent swallow is unacceptable for a deletion sweep. A planning-stage
//! failure returns `Err` before any deletion, so a read error cannot strand a half-deleted table.
//!
//! The sweep is sequential: [`FileIO`] has no bulk-delete surface, and there is no port of
//! `executeDeleteWith(ExecutorService)`. Java does not gate this action on `gc.enabled`, unlike
//! `DeleteOrphanFiles`, so this port has no GC gate either.

use std::collections::HashSet;

use futures::future::BoxFuture;

use crate::error::Result;
use crate::io::FileIO;
use crate::spec::{DataContentType, TableMetadata};
use crate::table::Table;
use crate::{Error, TableIdent};

/// The injected delete function (Java `deleteWith`). The default deletes through [`FileIO::delete`].
pub type ReachableDeleteFunction = dyn Fn(String) -> BoxFuture<'static, Result<()>> + Send + Sync;

/// The six removed-file counts of a sweep. It mirrors Java `DeleteReachableFiles$Result` (1.10.0).
///
/// Each count is the size of its bucket: the files identified for deletion, not the files deleted.
/// [`Self::delete_failures`] carries the per-file failures.
#[derive(Debug, Default)]
pub struct DeleteReachableFilesResult {
    /// Java `deletedDataFilesCount()`. Every `Data`-content file of every snapshot.
    pub deleted_data_files_count: u64,
    /// Java `deletedEqualityDeleteFilesCount()`. Every `EqualityDeletes`-content file.
    pub deleted_equality_delete_files_count: u64,
    /// Java `deletedPositionDeleteFilesCount()`. Every `PositionDeletes` file, deletion vectors too.
    pub deleted_position_delete_files_count: u64,
    /// Java `deletedManifestsCount()`. Every manifest of every snapshot.
    pub deleted_manifests_count: u64,
    /// Java `deletedManifestListsCount()`. Every snapshot's manifest list.
    pub deleted_manifest_lists_count: u64,
    /// Java `deletedOtherFilesCount()`. Metadata files, the version-hint, and statistics files.
    pub deleted_other_files_count: u64,
    /// Per-file delete failures. Empty means every reachable file was deleted. Java has no such
    /// field, and the counts above still match Java, because a count is a planned-set size.
    pub delete_failures: Vec<ReachableDeleteFailure>,
}

impl DeleteReachableFilesResult {
    /// The sum of the six buckets: the files the action identified for deletion.
    pub fn total_deleted_files_count(&self) -> u64 {
        self.deleted_data_files_count
            + self.deleted_equality_delete_files_count
            + self.deleted_position_delete_files_count
            + self.deleted_manifests_count
            + self.deleted_manifest_lists_count
            + self.deleted_other_files_count
    }
}

/// One collected delete failure. It does not abort the sweep.
#[derive(Debug)]
pub struct ReachableDeleteFailure {
    /// The reachable file whose deletion failed.
    pub path: String,
    /// The underlying error.
    pub error: Error,
}

/// The reachable-file set, one bucket per Java `Result` count. The `HashSet` keeps a file that
/// several snapshots share to one entry, one deletion, and one count.
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
    /// Every reachable path across all buckets, sorted for a deterministic sweep.
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

    /// The result counts, derived from the bucket sizes.
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

/// An action that deletes every file reachable from a table metadata location.
///
/// **This action deletes the whole table.** Build it with [`DeleteReachableFiles::new`] and run it
/// with [`Self::execute`]. The module docs hold the reachable set and the failure posture.
pub struct DeleteReachableFiles {
    metadata_location: String,
    file_io: Option<FileIO>,
    delete_function: Option<Box<ReachableDeleteFunction>>,
}

impl DeleteReachableFiles {
    /// Creates the action for the table whose current metadata is at `metadata_location`. The
    /// `FileIO` defaults to [`FileIO::new_with_fs`], so non-local storage needs [`Self::io`].
    pub fn new(metadata_location: impl Into<String>) -> Self {
        DeleteReachableFiles {
            metadata_location: metadata_location.into(),
            file_io: None,
            delete_function: None,
        }
    }

    /// Sets the [`FileIO`] that reads the metadata and deletes the files (Java `io(FileIO)`).
    pub fn io(mut self, file_io: FileIO) -> Self {
        self.file_io = Some(file_io);
        self
    }

    /// Replaces the delete function. It receives exactly the reachable set, so a caller can collect
    /// that set without deleting anything.
    pub fn delete_with(
        mut self,
        delete_function: impl Fn(String) -> BoxFuture<'static, Result<()>> + Send + Sync + 'static,
    ) -> Self {
        self.delete_function = Some(Box::new(delete_function));
        self
    }

    /// Computes the reachable set across all snapshots, then deletes every file in it.
    ///
    /// # Errors
    ///
    /// Returns `Err` without deleting anything if planning cannot read the metadata, a manifest
    /// list, or a manifest. Per-file delete failures instead land in the returned result.
    pub async fn execute(self) -> Result<DeleteReachableFilesResult> {
        let file_io = self.resolve_file_io();

        // The walk reads metadata only, so the table is read-only and binds no catalog.
        let metadata = TableMetadata::read_from(&file_io, &self.metadata_location).await?;
        let table = Table::builder()
            .metadata(metadata)
            .metadata_location(self.metadata_location.clone())
            // A synthetic identity: `TableIdent` rejects an empty namespace, so it needs two parts.
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

        // A read error here aborts before any file is deleted.
        let reachable = collect_reachable_files(&table).await?;

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

    /// The configured [`FileIO`], or the local-filesystem default that `StaticTable` also uses.
    fn resolve_file_io(&self) -> FileIO {
        self.file_io.clone().unwrap_or_else(FileIO::new_with_fs)
    }
}

/// Builds the categorized reachable-file set for `table` across all snapshots.
///
/// The walk shape matches `DeleteOrphanFiles::collect_valid_files`. That collector returns a flat
/// set, because it only needs membership. This one keeps the buckets, because the six Java counts
/// are bucket sizes. A read failure returns `Err` before the caller deletes anything.
async fn collect_reachable_files(table: &Table) -> Result<ReachableFiles> {
    let metadata = table.metadata();
    let file_io = table.file_io();
    let mut reachable = ReachableFiles::default();

    for snapshot in metadata.snapshots() {
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
            reachable
                .manifests
                .insert(manifest_file.manifest_path.clone());

            // Read every entry, DELETED tombstones included. Java reads the same way, because a
            // tombstoned file is still a physical file the table wrote.
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
                        // A deletion vector is `PositionDeletes` content, so it lands here too.
                        reachable.position_delete_files.insert(path);
                    }
                    DataContentType::EqualityDeletes => {
                        reachable.equality_delete_files.insert(path);
                    }
                }
            }
        }
    }

    // The "other" bucket: current and previous metadata.json, version-hint, and statistics.
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

/// The version-hint location. Java adds it even for a non-Hadoop table, so a purge cleans a stray.
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

    /// A `MemoryCatalog` on a real local filesystem. The caller must hold the temp-dir guard alive.
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

    /// A two-long-column schema `{1 x long, 2 y long}`.
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

    /// Writes `content` to `path` through `file_io`.
    async fn write_real_file(file_io: &FileIO, path: &str, content: &[u8]) {
        file_io
            .new_output(path)
            .expect("new output")
            .write(Bytes::copy_from_slice(content))
            .await
            .expect("write file");
    }

    /// Writes `content` to `path`, then builds a matching descriptor.
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
            // A real delete file also carries a referenced path or equality ids. This walk reads
            // only the path and the content type.
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

    /// True if `path` exists on disk.
    async fn exists(file_io: &FileIO, path: &str) -> bool {
        file_io.exists(path).await.expect("exists check")
    }

    // ---- the reachable-set categorization ------------------------------------------------------

    /// Every reachable category must be exact on a multi-snapshot table with a statistics file.
    #[tokio::test]
    async fn reachable_set_categorizes_every_file_category() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // s1: a data file. The commit advances the metadata log.
        let d1 = real_file(
            &file_io,
            &format!("{location}/data/d1.parquet"),
            b"d1",
            DataContentType::Data,
        )
        .await;
        let table = append(&catalog, &table, vec![d1]).await;
        // s2: a second snapshot, so a second manifest list and metadata.json.
        let d2 = real_file(
            &file_io,
            &format!("{location}/data/d2.parquet"),
            b"d2",
            DataContentType::Data,
        )
        .await;
        let table = append(&catalog, &table, vec![d2]).await;

        // A statistics file, for the "other" bucket.
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

        assert_eq!(reachable.data_files.len(), 2, "two data files reachable");
        assert_eq!(
            reachable.manifest_lists.len(),
            2,
            "two manifest lists (one per snapshot)"
        );
        assert!(
            reachable.manifests.len() >= 2,
            "at least two manifests reachable, got {}",
            reachable.manifests.len()
        );
        // The "other" bucket holds the current and one previous metadata.json, the version-hint,
        // and the statistics file.
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
        assert_eq!(reachable.position_delete_files.len(), 0);
        assert_eq!(reachable.equality_delete_files.len(), 0);
    }

    /// A walk that drops a delete category, or buckets it as data, fails here.
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

        // Commit the data, then a row delta that carries both delete files.
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
        let counts = reachable.counts();
        assert_eq!(counts.deleted_data_files_count, 1);
        assert_eq!(counts.deleted_position_delete_files_count, 1);
        assert_eq!(counts.deleted_equality_delete_files_count, 1);
    }

    // ---- the delete sweep: both corruption classes ---------------------------------------------

    /// After `execute` every reachable file is gone, and a file outside the set survives.
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

        // A file outside the table footprint. It is not reachable, so it must survive.
        let outsider = format!("{location}/data/not-ours.txt");
        write_real_file(&file_io, &outsider, b"keep me").await;

        // Snapshot the reachable set first, for the completeness assertion.
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

        for path in &all_reachable {
            assert!(
                !exists(&file_io, path).await,
                "reachable file must be deleted: {path}"
            );
        }
        // The outsider survives. Over-deletion is data loss.
        assert!(
            exists(&file_io, &outsider).await,
            "a file outside the reachable set must NOT be deleted"
        );
        // The counts equal the bucket sizes. A clean local FS fails no delete.
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

    /// The `delete_with` consumer receives exactly the reachable set, and may delete nothing.
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
        assert!(
            exists(&file_io, &format!("{location}/data/d1.parquet")).await,
            "collect-only deletes nothing"
        );
        assert_eq!(
            result.total_deleted_files_count(),
            reachable_expected.len() as u64
        );
    }

    /// A table with no snapshots is still purgeable: metadata.json plus version-hint.
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
        assert!(
            !exists(&file_io, &metadata_location).await,
            "the metadata.json must be deleted"
        );
    }

    /// One failed delete neither aborts the sweep nor changes the counts.
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
