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

//! A factory surface mirroring Java's `org.apache.iceberg.actions.ActionsProvider` (1.10.0) — the
//! per-engine entry point that hands out the table-maintenance action builders.
//!
//! # The Java contract this mirrors
//!
//! Java `ActionsProvider` is an interface whose every method is a `default` that throws
//! `UnsupportedOperationException` unless a concrete engine factory (`SparkActions`,
//! `FlinkActions`, …) overrides it. The twelve methods are:
//!
//! | Java method | arg | this crate |
//! |---|---|---|
//! | `snapshotTable(String)` | name | unsupported (no Rust action) |
//! | `migrateTable(String)` | name | unsupported (no Rust action) |
//! | `deleteOrphanFiles(Table)` | table | [`Actions::delete_orphan_files`] |
//! | `rewriteManifests(Table)` | table | [`Actions::rewrite_manifests`] |
//! | `rewriteDataFiles(Table)` | table | [`Actions::rewrite_data_files`] |
//! | `expireSnapshots(Table)` | table | [`Actions::expire_snapshots`] |
//! | `deleteReachableFiles(String)` | location | unsupported (no Rust action) |
//! | `rewritePositionDeletes(Table)` | table | unsupported (no Rust action) |
//! | `computeTableStats(Table)` | table | [`Actions::compute_table_stats`] |
//! | `computePartitionStats(Table)` | table | unsupported (compute core only — see below) |
//! | `rewriteTablePath(Table)` | table | unsupported (no Rust action) |
//! | `removeDanglingDeleteFiles(Table)` | table | [`Actions::remove_dangling_delete_files`] |
//!
//! # Two construction idioms, one factory
//!
//! The Rust actions this factory hands out are built two different ways, and the factory bridges
//! both honestly:
//!
//! - **Maintenance idiom** — [`DeleteOrphanFiles`], [`RewriteDataFiles`], [`ComputeTableStats`],
//!   [`RemoveDanglingDeleteFiles`] are constructed `X::new(table)` and run with
//!   `.execute(..)` directly against a [`Catalog`](crate::Catalog) (or, for
//!   `DeleteOrphanFiles`, with no catalog). The factory returns the fully-built action; the caller
//!   configures it with its builder methods and calls `.execute`.
//! - **Transaction-seam idiom** — [`ExpireSnapshotsAction`](crate::transaction::ExpireSnapshotsAction)
//!   and [`RewriteManifestsAction`](crate::transaction::RewriteManifestsAction) are *stateless*
//!   builders that do not bind a table at construction; they bind at commit time through the
//!   [`transaction`](crate::transaction) seam. Java's `expireSnapshots(Table)` /
//!   `rewriteManifests(Table)` take a `Table`, so the factory methods do too — but because the Rust
//!   action runs through a [`Transaction`](crate::transaction::Transaction), the factory returns the
//!   bare action and the caller applies it via
//!   [`ApplyTransactionAction::apply`](crate::transaction::ApplyTransactionAction) onto a
//!   `Transaction::new(&table)`. See each method's doc for the exact run recipe.
//!
//! # Unsupported actions — surfaced honestly, never faked
//!
//! Five Java methods have no Rust action behind them yet (`snapshotTable`, `migrateTable`,
//! `deleteReachableFiles`, `rewritePositionDeletes`, `rewriteTablePath`), and `computePartitionStats`
//! has only the *compute core* ([`compute_partition_stats`](crate::maintenance::compute_partition_stats))
//! with no action wrapper. Rather than fabricate them, the [`ActionsProvider`] trait mirrors Java's
//! *throw-by-default* shape: each unsupported method has a default that returns a typed
//! [`ErrorKind::FeatureUnsupported`](crate::ErrorKind::FeatureUnsupported) error naming the gap (the
//! Rust analog of Java's `UnsupportedOperationException`). The concrete [`Actions`] factory overrides
//! exactly the six methods Rust can actually run, and leaves the rest at the unsupported default. The
//! gap is tracked in `docs/parity/GAP_MATRIX.md` row 151.

use crate::Result;
use crate::error::Error;
use crate::maintenance::{
    ComputeTableStats, DeleteOrphanFiles, RemoveDanglingDeleteFiles, RewriteDataFiles,
};
use crate::table::Table;
use crate::transaction::{ExpireSnapshotsAction, RewriteManifestsAction};

/// The Rust analog of Java's `org.apache.iceberg.actions.ActionsProvider` (1.10.0): a factory that
/// hands out table-maintenance action builders.
///
/// Like the Java interface, every method has a default that signals *unsupported* — here a typed
/// [`ErrorKind::FeatureUnsupported`](crate::ErrorKind::FeatureUnsupported) error rather than Java's
/// thrown `UnsupportedOperationException`. The concrete [`Actions`] implementation overrides exactly
/// the subset of methods this crate has actions for. Method names mirror Java in `snake_case`; the
/// argument shape (`Table` vs `&str`) mirrors Java's `Table` vs `String`.
///
/// **Returning a `Result`** (rather than the bare action) is the deliberate adaptation that lets one
/// trait surface both the supported actions *and* the unsupported defaults without a separate
/// throwing path: a supported method returns `Ok(action)`, an unsupported one returns
/// `Err(FeatureUnsupported)`.
pub trait ActionsProvider {
    /// Mirrors Java `snapshotTable(String)`. **Unsupported in this crate** (no Rust `SnapshotTable`
    /// action): returns [`ErrorKind::FeatureUnsupported`](crate::ErrorKind::FeatureUnsupported).
    fn snapshot_table(&self, source_table_name: &str) -> Result<NoAction> {
        let _ = source_table_name;
        Err(unsupported("snapshot_table", "SnapshotTable"))
    }

    /// Mirrors Java `migrateTable(String)`. **Unsupported in this crate** (no Rust `MigrateTable`
    /// action): returns [`ErrorKind::FeatureUnsupported`](crate::ErrorKind::FeatureUnsupported).
    fn migrate_table(&self, table_name: &str) -> Result<NoAction> {
        let _ = table_name;
        Err(unsupported("migrate_table", "MigrateTable"))
    }

    /// Mirrors Java `deleteOrphanFiles(Table)`. **Unsupported by default**; the concrete [`Actions`]
    /// factory overrides it to return [`DeleteOrphanFiles::new`].
    fn delete_orphan_files(&self, table: Table) -> Result<DeleteOrphanFiles> {
        let _ = table;
        Err(unsupported("delete_orphan_files", "DeleteOrphanFiles"))
    }

    /// Mirrors Java `rewriteManifests(Table)`. **Unsupported by default**; the concrete [`Actions`]
    /// factory overrides it to return a [`RewriteManifestsAction`](crate::transaction::RewriteManifestsAction).
    fn rewrite_manifests(&self, table: Table) -> Result<RewriteManifestsAction> {
        let _ = table;
        Err(unsupported("rewrite_manifests", "RewriteManifests"))
    }

    /// Mirrors Java `rewriteDataFiles(Table)`. **Unsupported by default**; the concrete [`Actions`]
    /// factory overrides it to return [`RewriteDataFiles::new`].
    fn rewrite_data_files(&self, table: Table) -> Result<RewriteDataFiles> {
        let _ = table;
        Err(unsupported("rewrite_data_files", "RewriteDataFiles"))
    }

    /// Mirrors Java `expireSnapshots(Table)`. **Unsupported by default**; the concrete [`Actions`]
    /// factory overrides it to return an [`ExpireSnapshotsAction`](crate::transaction::ExpireSnapshotsAction).
    fn expire_snapshots(&self, table: Table) -> Result<ExpireSnapshotsAction> {
        let _ = table;
        Err(unsupported("expire_snapshots", "ExpireSnapshots"))
    }

    /// Mirrors Java `deleteReachableFiles(String)`. **Unsupported in this crate** (no Rust
    /// `DeleteReachableFiles` action): returns
    /// [`ErrorKind::FeatureUnsupported`](crate::ErrorKind::FeatureUnsupported).
    fn delete_reachable_files(&self, metadata_location: &str) -> Result<NoAction> {
        let _ = metadata_location;
        Err(unsupported(
            "delete_reachable_files",
            "DeleteReachableFiles",
        ))
    }

    /// Mirrors Java `rewritePositionDeletes(Table)`. **Unsupported in this crate** (no Rust
    /// `RewritePositionDeleteFiles` action): returns
    /// [`ErrorKind::FeatureUnsupported`](crate::ErrorKind::FeatureUnsupported).
    fn rewrite_position_deletes(&self, table: Table) -> Result<NoAction> {
        let _ = table;
        Err(unsupported(
            "rewrite_position_deletes",
            "RewritePositionDeleteFiles",
        ))
    }

    /// Mirrors Java `computeTableStats(Table)`. **Unsupported by default**; the concrete [`Actions`]
    /// factory overrides it to return [`ComputeTableStats::new`].
    fn compute_table_stats(&self, table: Table) -> Result<ComputeTableStats> {
        let _ = table;
        Err(unsupported("compute_table_stats", "ComputeTableStats"))
    }

    /// Mirrors Java `computePartitionStats(Table)`. **Unsupported in this crate**: only the compute
    /// core ([`compute_partition_stats`](crate::maintenance::compute_partition_stats)) is ported, with
    /// no action wrapper, so the factory cannot hand out a configurable action. Returns
    /// [`ErrorKind::FeatureUnsupported`](crate::ErrorKind::FeatureUnsupported).
    fn compute_partition_stats(&self, table: Table) -> Result<NoAction> {
        let _ = table;
        Err(unsupported(
            "compute_partition_stats",
            "ComputePartitionStats",
        ))
    }

    /// Mirrors Java `rewriteTablePath(Table)`. **Unsupported in this crate** (no Rust
    /// `RewriteTablePath` action): returns
    /// [`ErrorKind::FeatureUnsupported`](crate::ErrorKind::FeatureUnsupported).
    fn rewrite_table_path(&self, table: Table) -> Result<NoAction> {
        let _ = table;
        Err(unsupported("rewrite_table_path", "RewriteTablePath"))
    }

    /// Mirrors Java `removeDanglingDeleteFiles(Table)`. **Unsupported by default**; the concrete
    /// [`Actions`] factory overrides it to return [`RemoveDanglingDeleteFiles::new`].
    fn remove_dangling_delete_files(&self, table: Table) -> Result<RemoveDanglingDeleteFiles> {
        let _ = table;
        Err(unsupported(
            "remove_dangling_delete_files",
            "RemoveDanglingDeleteFiles",
        ))
    }
}

/// An uninhabited placeholder for the return type of an *unsupported* factory method.
///
/// A method that cannot hand out an action declares `Result<NoAction>`; because it always returns
/// `Err`, the `Ok` arm is unreachable and `NoAction` is never constructed. This keeps the trait's
/// unsupported methods typed (no `()` masquerading as an action) and makes "you cannot get an action
/// here" legible at the call site.
#[derive(Debug)]
pub enum NoAction {}

/// Build the typed `FeatureUnsupported` error for an unsupported factory method, naming both the Rust
/// method and the Java action it would mirror. Centralized so every unsupported arm reports the same
/// shape (the Rust analog of Java's `UnsupportedOperationException`).
fn unsupported(method: &str, java_action: &str) -> Error {
    Error::new(
        crate::ErrorKind::FeatureUnsupported,
        format!(
            "ActionsProvider::{method} is not supported: this crate has no {java_action} action \
             (see docs/parity/GAP_MATRIX.md row 151)"
        ),
    )
}

/// The engine-agnostic concrete [`ActionsProvider`] for this crate: a zero-state factory that hands
/// out the six table-maintenance actions Rust has built, leaving the other six Java methods at their
/// unsupported default.
///
/// This is the Rust analog of a Java engine's `ActionsProvider` implementation
/// (`SparkActions.get()` / `FlinkActions.get()`), minus any engine binding — these actions are
/// engine-agnostic, so the factory carries no state.
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
    /// Returns the engine-agnostic actions factory. Mirrors a Java engine's `Actions.get()` entry
    /// point (e.g. `SparkActions.get()`), minus the engine binding.
    pub fn get() -> Self {
        Actions
    }
}

impl ActionsProvider for Actions {
    /// Returns a [`DeleteOrphanFiles`] action for `table` (Java `deleteOrphanFiles(Table)`). Configure
    /// it with the builder methods and run it with [`DeleteOrphanFiles::execute`] (no catalog — it
    /// lists storage and deletes orphans directly). **This action deletes files.**
    fn delete_orphan_files(&self, table: Table) -> Result<DeleteOrphanFiles> {
        Ok(DeleteOrphanFiles::new(table))
    }

    /// Returns a [`RewriteManifestsAction`](crate::transaction::RewriteManifestsAction) (Java
    /// `rewriteManifests(Table)`). Because this is a transaction-seam action it does not bind `table`
    /// at construction; run it by applying it onto a transaction:
    /// `action.apply(Transaction::new(&table))?.commit(catalog).await`.
    fn rewrite_manifests(&self, table: Table) -> Result<RewriteManifestsAction> {
        // `table` is accepted to mirror Java's `rewriteManifests(Table)` shape; the seam action binds
        // the table at `apply`/`commit` time, so it is intentionally not stored here.
        let _ = table;
        Ok(RewriteManifestsAction::new())
    }

    /// Returns a [`RewriteDataFiles`] bin-pack action for `table` (Java `rewriteDataFiles(Table)`).
    /// Configure it with the builder methods and run it with [`RewriteDataFiles::execute`]. **This
    /// action rewrites data.**
    fn rewrite_data_files(&self, table: Table) -> Result<RewriteDataFiles> {
        Ok(RewriteDataFiles::new(table))
    }

    /// Returns an [`ExpireSnapshotsAction`](crate::transaction::ExpireSnapshotsAction) (Java
    /// `expireSnapshots(Table)`). Because this is a transaction-seam action it does not bind `table` at
    /// construction; run it by applying it onto a transaction:
    /// `action.apply(Transaction::new(&table))?.commit(catalog).await`. **This action never deletes
    /// files** (see [`ExpireSnapshotsCleanup`](crate::transaction::ExpireSnapshotsCleanup) for the
    /// explicit cleanup step).
    fn expire_snapshots(&self, table: Table) -> Result<ExpireSnapshotsAction> {
        // `table` is accepted to mirror Java's `expireSnapshots(Table)` shape; the seam action binds
        // the table at `apply`/`commit` time, so it is intentionally not stored here.
        let _ = table;
        Ok(ExpireSnapshotsAction::new())
    }

    /// Returns a [`ComputeTableStats`] action for `table` (Java `computeTableStats(Table)`). Configure
    /// it with the builder methods and run it with [`ComputeTableStats::execute`].
    fn compute_table_stats(&self, table: Table) -> Result<ComputeTableStats> {
        Ok(ComputeTableStats::new(table))
    }

    /// Returns a [`RemoveDanglingDeleteFiles`] action for `table` (Java
    /// `removeDanglingDeleteFiles(Table)`). Run it with [`RemoveDanglingDeleteFiles::execute`].
    /// **This action removes delete files.**
    fn remove_dangling_delete_files(&self, table: Table) -> Result<RemoveDanglingDeleteFiles> {
        Ok(RemoveDanglingDeleteFiles::new(table))
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
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, NestedField, PartitionSpec,
        PrimitiveType, Schema, Struct, Type,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};

    /// The exact set of factory methods the concrete [`Actions`] supports (overrides off the
    /// unsupported default). Pins the supported surface so a wiring that silently drops or adds an
    /// override fails this test. The six map 1:1 to Java `ActionsProvider` methods with a built Rust
    /// action.
    const SUPPORTED_METHODS: [&str; 6] = [
        "delete_orphan_files",
        "rewrite_manifests",
        "rewrite_data_files",
        "expire_snapshots",
        "compute_table_stats",
        "remove_dangling_delete_files",
    ];

    /// The Java `ActionsProvider` methods with NO Rust action behind them — the factory honestly
    /// reports these as unsupported.
    const UNSUPPORTED_METHODS: [&str; 6] = [
        "snapshot_table",
        "migrate_table",
        "delete_reachable_files",
        "rewrite_position_deletes",
        "compute_partition_stats",
        "rewrite_table_path",
    ];

    // ---- self-contained test fixtures (a local-fs MemoryCatalog) ------------------------------

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

    /// Write `content` to `path` through `file_io` (creates parent dirs on the local fs).
    async fn write_real_file(file_io: &FileIO, path: &str, content: &[u8]) {
        file_io
            .new_output(path)
            .expect("new output")
            .write(Bytes::copy_from_slice(content))
            .await
            .expect("write file");
    }

    /// A real data file: write `content` to `path` on disk, then build a metadata-consistent
    /// unpartitioned [`DataFile`].
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
        // The two sets are disjoint and together cover all twelve Java `ActionsProvider` methods.
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

        // Every supported method must hand out an action (Ok), proving the override is wired.
        assert!(actions.delete_orphan_files(table.clone()).is_ok());
        assert!(actions.rewrite_manifests(table.clone()).is_ok());
        assert!(actions.rewrite_data_files(table.clone()).is_ok());
        assert!(actions.expire_snapshots(table.clone()).is_ok());
        assert!(actions.compute_table_stats(table.clone()).is_ok());
        assert!(actions.remove_dangling_delete_files(table.clone()).is_ok());
    }

    #[tokio::test]
    async fn unsupported_methods_return_typed_feature_unsupported() {
        let (catalog, _file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let actions = Actions::get();

        // String-arg unsupported methods.
        for err in [
            actions.snapshot_table("db.src").unwrap_err(),
            actions.migrate_table("db.src").unwrap_err(),
            actions
                .delete_reachable_files("s3://b/t/metadata.json")
                .unwrap_err(),
        ] {
            assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
        }
        // Table-arg unsupported methods.
        for err in [
            actions.rewrite_position_deletes(table.clone()).unwrap_err(),
            actions.compute_partition_stats(table.clone()).unwrap_err(),
            actions.rewrite_table_path(table.clone()).unwrap_err(),
        ] {
            assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
        }
    }

    /// Smoke test: a `DeleteOrphanFiles` handed out by the factory actually RUNS (deletes a planted
    /// orphan, spares the live file) — proving the wiring is live, not a stub. A broken
    /// `delete_orphan_files` override (e.g. one that ignored the table) fails here.
    #[tokio::test]
    async fn delete_orphan_files_from_factory_executes_live() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        // A live data file (committed) and an orphan file (never referenced).
        let live =
            real_data_file(&file_io, &format!("{location}/data/live.parquet"), b"live").await;
        let table = append(&catalog, &table, vec![live]).await;
        let orphan_path = format!("{location}/data/orphan.parquet");
        write_real_file(&file_io, &orphan_path, b"orphan").await;

        let result = Actions::get()
            .delete_orphan_files(table)
            .expect("factory returns delete-orphan-files action")
            // grace must be in the future so the just-written orphan is eligible.
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

    /// Smoke test: a `RewriteDataFiles` handed out by the factory actually RUNS (no-op on a tiny
    /// table, but the execute path is exercised end-to-end), proving the override is live.
    #[tokio::test]
    async fn rewrite_data_files_from_factory_executes_live() {
        let (catalog, file_io, _tmp) = local_fs_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let f = real_data_file(&file_io, &format!("{location}/data/a.parquet"), b"a").await;
        let table = append(&catalog, &table, vec![f]).await;

        // A single small file is below `min_input_files`, so this is a no-op rewrite — but it
        // exercises the factory → action → execute path and must not error.
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

    /// Smoke test: a `RemoveDanglingDeleteFiles` handed out by the factory actually RUNS (no-op on an
    /// unpartitioned single-spec table — Java's early return — but the override is exercised live).
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

    /// Smoke test: an `ExpireSnapshotsAction` handed out by the factory is a real transaction-seam
    /// action that applies onto a transaction and commits, proving the seam-idiom bridge is live.
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
        // The seam action applies onto a transaction and commits without error (retain-all default
        // expires nothing on a single-snapshot table — the wiring, not the retention, is under test).
        let tx = action
            .apply(Transaction::new(&table))
            .expect("apply expire snapshots onto transaction");
        let committed = tx.commit(&catalog).await.expect("commit expire snapshots");
        assert!(
            committed.metadata().current_snapshot().is_some(),
            "the live snapshot survives a retain-all expire"
        );
    }

    /// Smoke test: a `RewriteManifestsAction` handed out by the factory is a real transaction-seam
    /// action that applies onto a transaction and commits a no-op rewrite, proving the seam bridge.
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
}
