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

//! A factory surface that mirrors Java `org.apache.iceberg.actions.ActionsProvider` (1.10.0).
//! It hands out the table-maintenance action builders.
//!
//! In Java every method is a `default` that throws `UnsupportedOperationException`. A concrete
//! engine factory overrides the ones it supports. The twelve methods map as follows:
//!
//! | Java method | arg | this crate |
//! |---|---|---|
//! | `snapshotTable(String)` | name | unsupported (no Rust action) |
//! | `migrateTable(String)` | name | unsupported (no Rust action) |
//! | `deleteOrphanFiles(Table)` | table | [`Actions::delete_orphan_files`] |
//! | `rewriteManifests(Table)` | table | [`Actions::rewrite_manifests`] |
//! | `rewriteDataFiles(Table)` | table | [`Actions::rewrite_data_files`] |
//! | `expireSnapshots(Table)` | table | [`Actions::expire_snapshots`] |
//! | `deleteReachableFiles(String)` | location | [`Actions::delete_reachable_files`] |
//! | `rewritePositionDeletes(Table)` | table | [`Actions::rewrite_position_deletes`] |
//! | `computeTableStats(Table)` | table | [`Actions::compute_table_stats`] |
//! | `computePartitionStats(Table)` | table | [`Actions::compute_partition_stats`] |
//! | `rewriteTablePath(Table)` | table | [`Actions::rewrite_table_path`] |
//! | `removeDanglingDeleteFiles(Table)` | table | [`Actions::remove_dangling_delete_files`] |
//!
//! A maintenance action binds the table at `X::new(table)`, then runs with `.execute(..)`. A
//! transaction-seam action binds no table at construction. It binds at commit time. The factory
//! still takes a `Table` to keep the Java shape, then discards it. The caller applies that action
//! onto a `Transaction::new(&table)`.
//!
//! `snapshotTable` and `migrateTable` need an external source table, which this library cannot
//! supply. Both keep the trait default, which returns `FeatureUnsupported`.

use crate::Result;
use crate::error::Error;
use crate::maintenance::{
    ComputePartitionStats, ComputeTableStats, DeleteOrphanFiles, DeleteReachableFiles,
    RemoveDanglingDeleteFiles, RewriteDataFiles, RewritePositionDeleteFiles, RewriteTablePath,
};
use crate::table::Table;
use crate::transaction::{ExpireSnapshotsAction, RewriteManifestsAction};

/// The Rust analog of Java `org.apache.iceberg.actions.ActionsProvider` (1.10.0).
///
/// Every method defaults to a typed
/// [`ErrorKind::FeatureUnsupported`](crate::ErrorKind::FeatureUnsupported) error, and [`Actions`]
/// overrides the supported ones. The `Result` return carries both cases in one trait, in place of
/// Java's throwing path.
pub trait ActionsProvider {
    /// Mirrors Java `snapshotTable(String)`. Unsupported: this crate has no `SnapshotTable` action.
    fn snapshot_table(&self, source_table_name: &str) -> Result<NoAction> {
        let _ = source_table_name;
        Err(unsupported("snapshot_table", "SnapshotTable"))
    }

    /// Mirrors Java `migrateTable(String)`. Unsupported: this crate has no `MigrateTable` action.
    fn migrate_table(&self, table_name: &str) -> Result<NoAction> {
        let _ = table_name;
        Err(unsupported("migrate_table", "MigrateTable"))
    }

    /// Mirrors Java `deleteOrphanFiles(Table)`. [`Actions`] returns [`DeleteOrphanFiles::new`].
    fn delete_orphan_files(&self, table: Table) -> Result<DeleteOrphanFiles> {
        let _ = table;
        Err(unsupported("delete_orphan_files", "DeleteOrphanFiles"))
    }

    /// Mirrors Java `rewriteManifests(Table)`. [`Actions`] returns a `RewriteManifestsAction`.
    fn rewrite_manifests(&self, table: Table) -> Result<RewriteManifestsAction> {
        let _ = table;
        Err(unsupported("rewrite_manifests", "RewriteManifests"))
    }

    /// Mirrors Java `rewriteDataFiles(Table)`. [`Actions`] returns [`RewriteDataFiles::new`].
    fn rewrite_data_files(&self, table: Table) -> Result<RewriteDataFiles> {
        let _ = table;
        Err(unsupported("rewrite_data_files", "RewriteDataFiles"))
    }

    /// Mirrors Java `expireSnapshots(Table)`. [`Actions`] returns an `ExpireSnapshotsAction`.
    fn expire_snapshots(&self, table: Table) -> Result<ExpireSnapshotsAction> {
        let _ = table;
        Err(unsupported("expire_snapshots", "ExpireSnapshots"))
    }

    /// Mirrors Java `deleteReachableFiles(String)`. The argument is the `metadata.json` location.
    fn delete_reachable_files(&self, metadata_location: &str) -> Result<DeleteReachableFiles> {
        let _ = metadata_location;
        Err(unsupported(
            "delete_reachable_files",
            "DeleteReachableFiles",
        ))
    }

    /// Mirrors Java `rewritePositionDeletes(Table)`. [`Actions`] returns a built action.
    fn rewrite_position_deletes(&self, table: Table) -> Result<RewritePositionDeleteFiles> {
        let _ = table;
        Err(unsupported(
            "rewrite_position_deletes",
            "RewritePositionDeleteFiles",
        ))
    }

    /// Mirrors Java `computeTableStats(Table)`. [`Actions`] returns [`ComputeTableStats::new`].
    fn compute_table_stats(&self, table: Table) -> Result<ComputeTableStats> {
        let _ = table;
        Err(unsupported("compute_table_stats", "ComputeTableStats"))
    }

    /// Mirrors Java `computePartitionStats(Table)`. [`Actions`] returns a built action.
    fn compute_partition_stats(&self, table: Table) -> Result<ComputePartitionStats> {
        let _ = table;
        Err(unsupported(
            "compute_partition_stats",
            "ComputePartitionStats",
        ))
    }

    /// Mirrors Java `rewriteTablePath(Table)`. [`Actions`] returns [`RewriteTablePath::new`].
    fn rewrite_table_path(&self, table: Table) -> Result<RewriteTablePath> {
        let _ = table;
        Err(unsupported("rewrite_table_path", "RewriteTablePath"))
    }

    /// Mirrors Java `removeDanglingDeleteFiles(Table)`. [`Actions`] returns a built action.
    fn remove_dangling_delete_files(&self, table: Table) -> Result<RemoveDanglingDeleteFiles> {
        let _ = table;
        Err(unsupported(
            "remove_dangling_delete_files",
            "RemoveDanglingDeleteFiles",
        ))
    }
}

/// An uninhabited placeholder for the return type of an unsupported factory method. The `Ok` arm
/// is unreachable, so nothing constructs a `NoAction`.
#[derive(Debug)]
pub enum NoAction {}

/// Builds the `FeatureUnsupported` error for an unsupported method, named after the Java action.
fn unsupported(method: &str, java_action: &str) -> Error {
    Error::new(
        crate::ErrorKind::FeatureUnsupported,
        format!(
            "ActionsProvider::{method} is not supported: this crate has no {java_action} action \
             (see docs/parity/GAP_MATRIX.md row R153)"
        ),
    )
}

/// The concrete [`ActionsProvider`] for this crate, the analog of Java `SparkActions.get()`. These
/// actions are engine-agnostic, so the factory carries no state.
///
/// ```
/// use iceberg::maintenance::{Actions, ActionsProvider};
///
/// // The factory is a zero-state value; supported methods return a built action, unsupported
/// // methods return a typed `FeatureUnsupported` error.
/// let actions = Actions::default();
/// // `snapshot_table` has no Rust action — it is honestly unsupported.
/// assert!(actions.snapshot_table("db.src").is_err());
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Actions;

impl Actions {
    /// Returns the actions factory. Mirrors a Java engine's `Actions.get()` entry point.
    pub fn get() -> Self {
        Actions
    }
}

impl ActionsProvider for Actions {
    /// Returns a [`DeleteOrphanFiles`] action. Its `execute` needs no catalog. **It deletes files.**
    fn delete_orphan_files(&self, table: Table) -> Result<DeleteOrphanFiles> {
        Ok(DeleteOrphanFiles::new(table))
    }

    /// Returns a [`DeleteReachableFiles`] action for the table at `metadata_location`. This is the
    /// engine behind `DROP TABLE PURGE`. **This action deletes the whole table.**
    fn delete_reachable_files(&self, metadata_location: &str) -> Result<DeleteReachableFiles> {
        Ok(DeleteReachableFiles::new(metadata_location))
    }

    /// Returns a [`RewriteManifestsAction`](crate::transaction::RewriteManifestsAction). Run it
    /// with `action.apply(Transaction::new(&table))?.commit(catalog).await`.
    fn rewrite_manifests(&self, table: Table) -> Result<RewriteManifestsAction> {
        // The arg keeps Java's shape. The seam binds the table at commit time, so do not store it.
        let _ = table;
        Ok(RewriteManifestsAction::new())
    }

    /// Returns a [`RewriteDataFiles`] bin-pack action. **This action rewrites data.**
    fn rewrite_data_files(&self, table: Table) -> Result<RewriteDataFiles> {
        Ok(RewriteDataFiles::new(table))
    }

    /// Returns an [`ExpireSnapshotsAction`](crate::transaction::ExpireSnapshotsAction). Run it with
    /// `action.apply(Transaction::new(&table))?.commit(catalog).await`. **This action never deletes
    /// files.** [`ExpireSnapshotsCleanup`](crate::transaction::ExpireSnapshotsCleanup) does that.
    fn expire_snapshots(&self, table: Table) -> Result<ExpireSnapshotsAction> {
        // The arg keeps Java's shape. The seam binds the table at commit time, so do not store it.
        let _ = table;
        Ok(ExpireSnapshotsAction::new())
    }

    /// Returns a [`ComputeTableStats`] action for `table`.
    fn compute_table_stats(&self, table: Table) -> Result<ComputeTableStats> {
        Ok(ComputeTableStats::new(table))
    }

    /// Returns a [`ComputePartitionStats`] action. Its `execute` registers the stats file.
    fn compute_partition_stats(&self, table: Table) -> Result<ComputePartitionStats> {
        Ok(ComputePartitionStats::new(table))
    }

    /// Returns a [`RemoveDanglingDeleteFiles`] action. **This action removes delete files.**
    fn remove_dangling_delete_files(&self, table: Table) -> Result<RemoveDanglingDeleteFiles> {
        Ok(RemoveDanglingDeleteFiles::new(table))
    }

    /// Returns a [`RewritePositionDeleteFiles`] action. It bin-packs the candidate Parquet
    /// position-delete files of each `(spec, partition)` group. It rewrites every bin that Java's
    /// three-clause admission gate admits, and preserves the masked row set. The file-count floor
    /// is `min_input_files`, default five. **This action rewrites delete files.**
    fn rewrite_position_deletes(&self, table: Table) -> Result<RewritePositionDeleteFiles> {
        Ok(RewritePositionDeleteFiles::new(table))
    }

    /// Returns a [`RewriteTablePath`] action. Set [`RewriteTablePath::rewrite_location_prefix`] and
    /// [`RewriteTablePath::staging_location`] first. **It stages the rewritten metadata and emits a
    /// `(source, target)` copy-plan. It does NOT copy data files.**
    fn rewrite_table_path(&self, table: Table) -> Result<RewriteTablePath> {
        Ok(RewriteTablePath::new(table))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use bytes::Bytes;

    use super::*;
    use crate::io::{FileIO, FileIOBuilder, LocalFsStorageFactory};
    use crate::memory::MemoryCatalogBuilder;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, NestedField,
        PartitionSpec, PrimitiveType, Schema, Struct, Transform, Type,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};

    /// The methods [`Actions`] overrides. A wiring that drops or adds one fails the test below.
    const SUPPORTED_METHODS: [&str; 10] = [
        "delete_orphan_files",
        "delete_reachable_files",
        "rewrite_manifests",
        "rewrite_data_files",
        "expire_snapshots",
        "compute_table_stats",
        "compute_partition_stats",
        "remove_dangling_delete_files",
        "rewrite_position_deletes",
        "rewrite_table_path",
    ];

    /// The Java `ActionsProvider` methods with no Rust action behind them.
    const UNSUPPORTED_METHODS: [&str; 2] = ["snapshot_table", "migrate_table"];

    // ---- fixtures ------------------------------------------------------------------------------

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

    /// A minimal three-long-column schema.
    fn three_long_schema() -> Schema {
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
                Arc::new(NestedField::required(
                    3,
                    "z",
                    Type::Primitive(PrimitiveType::Long),
                )),
            ])
            .build()
            .expect("build schema")
    }

    /// Create an unpartitioned table under a fresh namespace.
    async fn create_unpartitioned_table(catalog: &impl Catalog) -> Table {
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace");
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(three_long_schema())
            .partition_spec(PartitionSpec::unpartition_spec())
            .build();
        catalog
            .create_table(&namespace, creation)
            .await
            .expect("create table")
    }

    /// Creates a table partitioned by `identity(x)`. `compute_partition_stats` needs one.
    async fn create_x_partitioned_table(catalog: &impl Catalog) -> Table {
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace");
        let spec = PartitionSpec::builder(three_long_schema())
            .with_spec_id(0)
            .add_partition_field("x", "x", Transform::Identity)
            .expect("add partition field")
            .build()
            .expect("build spec");
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(three_long_schema())
            .partition_spec(spec.into_unbound())
            .build();
        catalog
            .create_table(&namespace, creation)
            .await
            .expect("create table")
    }

    /// A real data file stamped with spec 0 and the given partition tuple, holding `records` rows.
    async fn partitioned_data_file(
        file_io: &FileIO,
        path: &str,
        partition: Struct,
        records: u64,
    ) -> DataFile {
        write_real_file(file_io, path, b"data").await;
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(4)
            .record_count(records)
            .partition_spec_id(0)
            .partition(partition)
            .build()
            .expect("build data file")
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

    /// Writes `content` to `path`, then builds a matching unpartitioned [`DataFile`].
    async fn real_data_file(file_io: &FileIO, path: &str, content: &[u8]) -> DataFile {
        write_real_file(file_io, path, content).await;
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(content.len() as u64)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .expect("build data file")
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

    // ---- tests --------------------------------------------------------------------------------

    #[test]
    fn supported_and_unsupported_partition_the_twelve_java_methods() {
        // The two sets are disjoint and cover all twelve Java methods.
        let supported: HashSet<&str> = SUPPORTED_METHODS.into_iter().collect();
        let unsupported: HashSet<&str> = UNSUPPORTED_METHODS.into_iter().collect();
        assert!(
            supported.is_disjoint(&unsupported),
            "a method cannot be both supported and unsupported"
        );
        assert_eq!(
            supported.len() + unsupported.len(),
            12,
            "the factory must account for all twelve Java ActionsProvider methods"
        );
    }

    #[tokio::test]
    async fn supported_methods_return_built_actions() {
        let (catalog, _file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let actions = Actions::get();

        // Every supported method must hand out an action, which proves the override is wired.
        assert!(actions.delete_orphan_files(table.clone()).is_ok());
        assert!(
            actions
                .delete_reachable_files(
                    table.metadata_location_result().expect("metadata location")
                )
                .is_ok()
        );
        assert!(actions.rewrite_manifests(table.clone()).is_ok());
        assert!(actions.rewrite_data_files(table.clone()).is_ok());
        assert!(actions.expire_snapshots(table.clone()).is_ok());
        assert!(actions.compute_table_stats(table.clone()).is_ok());
        assert!(actions.compute_partition_stats(table.clone()).is_ok());
        assert!(actions.remove_dangling_delete_files(table.clone()).is_ok());
        assert!(actions.rewrite_position_deletes(table.clone()).is_ok());
        assert!(actions.rewrite_table_path(table.clone()).is_ok());
    }

    #[tokio::test]
    async fn unsupported_methods_return_typed_feature_unsupported() {
        let (catalog, _file_io, _tmp) = local_fs_catalog().await;
        let _table = create_unpartitioned_table(&catalog).await;
        let actions = Actions::get();

        // Only the String-arg methods stay unsupported. Both need an external source table.
        for err in [
            actions.snapshot_table("db.src").unwrap_err(),
            actions.migrate_table("db.src").unwrap_err(),
        ] {
            assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
        }
    }

    /// The factory-built action deletes a planted orphan and spares the live file.
    #[tokio::test]
    async fn delete_orphan_files_from_factory_executes_live() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // One committed data file, and one orphan that nothing references.
        let live =
            real_data_file(&file_io, &format!("{location}/data/live.parquet"), b"live").await;
        let table = append(&catalog, &table, vec![live]).await;
        let orphan_path = format!("{location}/data/orphan.parquet");
        write_real_file(&file_io, &orphan_path, b"orphan").await;

        let result = Actions::get()
            .delete_orphan_files(table)
            .expect("factory returns delete-orphan-files action")
            // The grace bound must be in the future, or the new orphan is not eligible.
            .older_than(i64::MAX)
            .execute()
            .await
            .expect("execute delete orphan files");

        assert!(
            result
                .orphan_file_locations
                .iter()
                .any(|p| p.ends_with("orphan.parquet")),
            "the planted orphan must be deleted by the factory-built action"
        );
        assert!(
            !exists(&file_io, &orphan_path).await,
            "orphan file must be physically gone"
        );
    }

    /// The factory-built action purges the metadata.json, manifest list, manifest and data file.
    /// **The table is a throwaway local-fs `MemoryCatalog` table, never a live catalog.**
    #[tokio::test]
    async fn delete_reachable_files_from_factory_executes_live() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // A committed data file, reachable from the snapshot.
        let data = real_data_file(&file_io, &format!("{location}/data/d.parquet"), b"data").await;
        let table = append(&catalog, &table, vec![data]).await;
        let metadata_location = table
            .metadata_location_result()
            .expect("metadata location")
            .to_string();

        let result = Actions::get()
            .delete_reachable_files(&metadata_location)
            .expect("factory returns delete-reachable-files action")
            .io(file_io.clone())
            .execute()
            .await
            .expect("execute delete reachable files");

        assert_eq!(
            result.deleted_data_files_count, 1,
            "the one committed data file is purged"
        );
        assert!(
            result.deleted_manifest_lists_count >= 1,
            "the snapshot's manifest list is purged"
        );
        assert!(
            result.deleted_other_files_count >= 1,
            "the metadata.json is purged"
        );
        // The metadata.json is gone, so the table no longer loads from disk.
        assert!(
            !exists(&file_io, &metadata_location).await,
            "the table metadata.json must be physically gone after a reachable-files purge"
        );
        assert!(
            !exists(&file_io, &format!("{location}/data/d.parquet")).await,
            "the committed data file must be physically gone"
        );
    }

    /// The factory-built `RewriteDataFiles` runs end to end. The rewrite itself is a no-op here.
    #[tokio::test]
    async fn rewrite_data_files_from_factory_executes_live() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let f = real_data_file(&file_io, &format!("{location}/data/a.parquet"), b"a").await;
        let table = append(&catalog, &table, vec![f]).await;

        // One small file is below `min_input_files`, so the rewrite is a no-op. It must not error.
        let result = Actions::get()
            .rewrite_data_files(table)
            .expect("factory returns rewrite-data-files action")
            .execute(&catalog)
            .await
            .expect("execute rewrite data files");
        assert_eq!(
            result.rewritten_data_files_count, 0,
            "a single below-threshold file is not rewritten"
        );
    }

    /// The factory-built action runs. An unpartitioned single-spec table hits Java's early return.
    #[tokio::test]
    async fn remove_dangling_delete_files_from_factory_executes_live() {
        let (catalog, _file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;

        let result = Actions::get()
            .remove_dangling_delete_files(table)
            .expect("factory returns remove-dangling action")
            .execute(&catalog)
            .await
            .expect("execute remove dangling delete files");
        assert!(
            result.removed_delete_files.is_empty(),
            "an unpartitioned single-spec table has nothing to remove (Java early return)"
        );
    }

    /// The factory-built action runs with nothing to compact.
    #[tokio::test]
    async fn rewrite_position_deletes_from_factory_executes_live() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let f = real_data_file(&file_io, &format!("{location}/data/a.parquet"), b"a").await;
        let table = append(&catalog, &table, vec![f]).await;

        let result = Actions::get()
            .rewrite_position_deletes(table)
            .expect("factory returns rewrite-position-deletes action")
            .execute(&catalog)
            .await
            .expect("execute rewrite position deletes");
        assert_eq!(
            result.rewritten_delete_files_count, 0,
            "a table with no position-delete files has nothing to compact"
        );
        assert_eq!(result.added_delete_files_count, 0);
    }

    /// The factory-built `ExpireSnapshotsAction` applies onto a transaction and commits.
    #[tokio::test]
    async fn expire_snapshots_from_factory_applies_through_transaction() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let f = real_data_file(&file_io, &format!("{location}/data/a.parquet"), b"a").await;
        let table = append(&catalog, &table, vec![f]).await;

        let action = Actions::get()
            .expire_snapshots(table.clone())
            .expect("factory returns expire-snapshots action");
        // The retain-all default expires nothing here. The wiring is under test, not the retention.
        let tx = action
            .apply(Transaction::new(&table))
            .expect("apply expire snapshots onto transaction");
        let committed = tx.commit(&catalog).await.expect("commit expire snapshots");
        assert!(
            committed.metadata().current_snapshot().is_some(),
            "the live snapshot survives a retain-all expire"
        );
    }

    /// The factory-built `RewriteManifestsAction` applies onto a transaction and commits.
    #[tokio::test]
    async fn rewrite_manifests_from_factory_applies_through_transaction() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let f = real_data_file(&file_io, &format!("{location}/data/a.parquet"), b"a").await;
        let table = append(&catalog, &table, vec![f]).await;

        let action = Actions::get()
            .rewrite_manifests(table.clone())
            .expect("factory returns rewrite-manifests action");
        let tx = action
            .apply(Transaction::new(&table))
            .expect("apply rewrite manifests onto transaction");
        let committed = tx.commit(&catalog).await.expect("commit rewrite manifests");
        assert!(
            committed.metadata().current_snapshot().is_some(),
            "the live snapshot survives a no-op manifest rewrite"
        );
    }

    /// The factory-built action writes and registers a partition-stats file.
    #[tokio::test]
    async fn compute_partition_stats_from_factory_executes_live() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // Two partitions keep the computed collection non-empty. The action errors on an empty one.
        let table = append(&catalog, &table, vec![
            partitioned_data_file(
                &file_io,
                &format!("{location}/data/x=1/d1.parquet"),
                Struct::from_iter([Some(Literal::long(1))]),
                3,
            )
            .await,
            partitioned_data_file(
                &file_io,
                &format!("{location}/data/x=2/d2.parquet"),
                Struct::from_iter([Some(Literal::long(2))]),
                5,
            )
            .await,
        ])
        .await;
        let snapshot_id = table.metadata().current_snapshot_id().unwrap();

        let result = Actions::get()
            .compute_partition_stats(table)
            .expect("factory returns compute-partition-stats action")
            .execute(&catalog)
            .await
            .expect("execute compute partition stats");

        assert_eq!(result.statistics_file.snapshot_id, snapshot_id);
        assert!(result.statistics_file.file_size_in_bytes > 0);
        // The registered file lands in the refreshed metadata, so the seam commit fired.
        let registered = result
            .table
            .metadata()
            .partition_statistics_for_snapshot(snapshot_id)
            .expect("registered partition statistics");
        assert_eq!(
            registered.statistics_path,
            result.statistics_file.statistics_path
        );
        // The file exists on disk.
        assert!(
            exists(&file_io, &result.statistics_file.statistics_path).await,
            "the partition-stats file must be written to disk"
        );
    }
}
