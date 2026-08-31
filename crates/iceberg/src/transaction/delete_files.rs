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

//! The delete-files action.
//!
//! [`DeleteFilesAction`] removes data files from a table by file path or [`DataFile`] reference. It
//! mirrors Java `StreamingDelete` / `DeleteFiles` and reuses the manifest-rewrite machinery in
//! [`SnapshotProducer`]. At commit the requested paths are resolved against the current snapshot's
//! manifests. Each manifest with a matching live entry is rewritten. Matching entries become `Deleted`,
//! and the rest are copied forward as `Existing`.
//!
//! Deleting a path that no live entry carries is an error (Java `failMissingDeletePaths`).
//!
//! **Delete by row filter** (Java `StreamingDelete.deleteFromRowFilter`).
//! [`DeleteFilesAction::delete_from_row_filter`] stores a row predicate. At apply time
//! [`SnapshotProducer::resolve_filter_deletes`] resolves every live data file the predicate strictly
//! matches. This is the same helper `OverwriteFiles.overwriteByRowFilter` drives, reused unchanged.
//!
//! Per live data file the predicate is reduced to its per-partition residual under that file's OWN
//! partition spec. The strict and inclusive metrics evaluators then classify the file as KEEP, DELETE,
//! or PARTIAL. A PARTIAL match is a non-retryable error, because `StreamingDelete` deletes whole files
//! and cannot split one. An unpartitioned `Predicate::AlwaysTrue` filter deletes every data file.
//!
//! The recorded operation stays [`Operation::Delete`] unconditionally. Java
//! `StreamingDelete.operation()` is a constant, unlike the dynamic `BaseOverwriteFiles.operation()`.
//! A by-path delete set at the same time is unioned with the by-filter matches, de-duped by path, in the
//! one `Delete` snapshot.
//!
//! **Inherited divergence:** a by-path delete plus a PARTIAL filter match. Java deletes the
//! file (path mark skips metrics). This port raises "some, but not all, rows match". Fail-safe:
//! it never wrongly deletes. Fix belongs in `snapshot.rs`.
//!
//! [`DeleteFilesAction::case_sensitive`] defaults true, matching Java.
//!
//! Out of scope: `DeleteFiles.dropPartition`, which is `ReplacePartitions` territory.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::expr::Predicate;
use crate::spec::{DataFile, MAIN_BRANCH, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, FirstRowIdPolicy, SnapshotProduceOperation, SnapshotProducer,
    deleted_data_files_after_on,
};
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind};

/// A transaction action that deletes data files from a table by file path.
///
/// Use [`crate::transaction::Transaction::delete_files`] to create one. Accumulate the files to
/// remove with [`DeleteFilesAction::delete_file`] / [`DeleteFilesAction::delete_files`] /
/// [`DeleteFilesAction::delete_data_files`], then apply and commit the transaction. Committing
/// produces a new `Delete` snapshot whose live file set excludes the removed files.
pub struct DeleteFilesAction {
    /// Fully-qualified file paths to remove from the table.
    delete_paths: HashSet<String>,
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    /// Whether to reject the commit when a concurrent commit already DELETED a data file this action
    /// deletes (Java `StreamingDelete.validateFilesExist`). Off by default, which is snapshot isolation.
    validate_files_exist: bool,
    /// An explicit starting snapshot for the files-exist check (Java `validateFromSnapshot`). `None`
    /// uses the transaction's starting snapshot.
    validate_from_snapshot: Option<i64>,
    /// Stage the produced delete snapshot for write-audit-publish instead of moving `main` (Java
    /// `SnapshotProducer.stageOnly()`). See [`DeleteFilesAction::stage_only`].
    stage_only: bool,
    /// The delete-by-row-filter predicate (Java `deleteExpression`). `Some` removes every live data file
    /// the predicate strictly matches at apply time. A PARTIAL match is a non-retryable error. `None`
    /// performs no by-filter delete. See [`DeleteFilesAction::delete_from_row_filter`].
    row_filter: Option<Predicate>,
    /// Column-name case sensitivity for binding [`Self::row_filter`] (Java
    /// `MergingSnapshotProducer.caseSensitive`). It defaults to `true`, the Java default.
    case_sensitive: bool,
    pub(crate) target_branch: String,
}

impl DeleteFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            delete_paths: HashSet::default(),
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            validate_files_exist: false,
            validate_from_snapshot: None,
            stage_only: false,
            row_filter: None,
            // Java `MergingSnapshotProducer` defaults `caseSensitive` to true.
            case_sensitive: true,
            target_branch: MAIN_BRANCH.to_string(),
        }
    }

    /// Delete a single file by its fully-qualified path.
    ///
    /// The path must equal a path in the table's metadata. A different but equivalent path, such as
    /// `file:/p/f.parquet` against `file:///p/f.parquet`, does not match (Java
    /// `DeleteFiles.deleteFile(CharSequence)`).
    pub fn delete_file(mut self, path: impl Into<String>) -> Self {
        self.delete_paths.insert(path.into());
        self
    }

    /// Delete multiple files by their fully-qualified paths.
    pub fn delete_files(mut self, paths: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.delete_paths.extend(paths.into_iter().map(Into::into));
        self
    }

    /// Delete multiple files referenced by [`DataFile`]s (their paths are used).
    pub fn delete_data_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        self.delete_paths
            .extend(files.into_iter().map(|file| file.file_path));
        self
    }

    /// DELETE every current data file the `predicate` strictly matches (Java
    /// `DeleteFiles.deleteFromRowFilter(Expression)`). The matches are removed in the same `Delete`
    /// snapshot as any explicit by-path removal. A PARTIAL match fails the commit with a non-retryable
    /// error, because `StreamingDelete` deletes whole files and cannot split one. An unpartitioned
    /// [`Predicate::AlwaysTrue`] filter deletes every data file. The module doc states the full
    /// classification rule.
    pub fn delete_from_row_filter(mut self, predicate: Predicate) -> Self {
        self.row_filter = Some(predicate);
        self
    }

    /// Set whether the [`Self::delete_from_row_filter`] predicate resolves column names case-sensitively
    /// (Java `DeleteFiles.caseSensitive(boolean)`).
    ///
    /// The default is `true`, the Java default. A filter on `X` then binds only to a schema column named
    /// exactly `X`, and a wrong-cased reference fails to bind, so the commit errors.
    /// `case_sensitive(false)` binds `X` to the schema column `x` instead. This affects only the
    /// by-row-filter mode. By-path deletes resolve no column names.
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
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

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }

    /// Enable the files-exist conflict check (Java `StreamingDelete.validateFilesExist`). The commit is
    /// then rejected, non-retryably, when a snapshot committed since the starting snapshot already
    /// deleted a data file this action deletes. Without the check that concurrent removal is absorbed
    /// silently, because the path no longer resolves to a live entry on the re-based commit.
    ///
    /// The default is snapshot isolation, with no check.
    pub fn validate_files_exist(mut self) -> Self {
        self.validate_files_exist = true;
        self
    }

    /// Override the snapshot where the files-exist check starts (Java
    /// `DeleteFiles.validateFromSnapshot(long)`). By default the check uses the transaction's starting
    /// snapshot. This pins an earlier snapshot id instead.
    ///
    /// On its own this does not enable the check. Call [`Self::validate_files_exist`] for that.
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.validate_from_snapshot = Some(snapshot_id);
        self
    }

    /// STAGE this delete for write-audit-publish instead of publishing it to `main` (Java
    /// `SnapshotProducer.stageOnly()`). The commit adds the new `Delete` snapshot to table metadata but
    /// moves no ref. `current-snapshot-id`, the `main` ref, and the snapshot log stay unchanged, so
    /// readers still see the deleted rows until [`crate::transaction::Transaction::cherry_pick`]
    /// publishes the staged snapshot. The staged snapshot still consumes a sequence number.
    pub fn stage_only(mut self) -> Self {
        self.stage_only = true;
        self
    }
}

#[async_trait]
impl TransactionAction for DeleteFilesAction {
    fn target_ref(&self) -> &str {
        self.target_branch.as_str()
    }

    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            // A delete-only commit adds no data files.
            vec![],
            FirstRowIdPolicy::Suppress,
        )
        .with_stage_only(self.stage_only)
        .with_target_branch(self.target_branch.clone())?;

        snapshot_producer
            .commit(
                DeleteFilesOperation {
                    delete_paths: self.delete_paths.clone(),
                    row_filter: self.row_filter.clone(),
                    case_sensitive: self.case_sensitive,
                },
                DefaultManifestProcess,
            )
            .await
    }

    /// Files-exist conflict validation (Java `MergingSnapshotProducer.validateDataFilesExist`). It runs
    /// only when [`Self::validate_files_exist`] is enabled, and is otherwise a no-op.
    ///
    /// It takes the effective starting snapshot, enumerates every DATA file deleted from the refreshed
    /// base since then, and rejects the commit when one of those files is also in `self.delete_paths`.
    /// The rejection is a non-retryable [`ErrorKind::DataInvalid`] naming the missing file, so the retry
    /// loop stops and the message reaches the caller.
    async fn validate(
        self: Arc<Self>,
        starting_snapshot_id: Option<i64>,
        current: &Table,
    ) -> Result<()> {
        if !self.validate_files_exist {
            // The default is snapshot isolation, with no files-exist check.
            return Ok(());
        }

        // With nothing to delete, nothing can be missing. Skip the manifest walk.
        if self.delete_paths.is_empty() {
            return Ok(());
        }

        let effective_start = self.validate_from_snapshot.or(starting_snapshot_id);

        // `skip_deletes == false`: a `DeleteFiles` commit validates against ALL data-removing operations,
        // including concurrent DELETE-op snapshots. The files it deletes must still exist whichever
        // operation removed them. `RowDelta` is the variant that skips DELETE-op snapshots by default.
        let deleted = deleted_data_files_after_on(
            current,
            effective_start,
            false,
            self.target_branch.as_str(),
        )
        .await?;

        // Reject on the FIRST concurrently-deleted file this action also requires, as Java does.
        if let Some(missing) = deleted
            .iter()
            .find(|file| self.delete_paths.contains(file.file_path()))
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot commit, missing data files: {}", missing.file_path()),
            ));
        }

        Ok(())
    }
}

/// The [`SnapshotProduceOperation`] for [`DeleteFilesAction`].
///
/// It records `Operation::Delete`, exposes every current manifest as the set to filter, and resolves
/// the files to remove. The resolved set is the requested paths against the current snapshot's live
/// entries, unioned with the live data files the optional row filter strictly matches.
struct DeleteFilesOperation {
    delete_paths: HashSet<String>,
    /// The delete-by-row-filter predicate (Java `deleteExpression`). `Some` unions every live data file
    /// this predicate strictly matches with the path-resolved deletes.
    row_filter: Option<Predicate>,
    /// Column-name case sensitivity for binding `row_filter`. It defaults to `true`.
    case_sensitive: bool,
}

impl SnapshotProduceOperation for DeleteFilesOperation {
    fn operation(&self) -> Operation {
        // Java `StreamingDelete.operation()` is a constant. A by-row-filter delete is still a `Delete`.
        Operation::Delete
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn delete_files(&self, snapshot_produce: &SnapshotProducer<'_>) -> Result<Vec<DataFile>> {
        // Every requested path must match a live entry (Java `failMissingDeletePaths`).
        let mut resolved = snapshot_produce
            .resolve_delete_paths(&self.delete_paths)
            .await?;

        // Union the delete-by-row-filter matches. De-dupe by path so a file removed by BOTH a path and
        // the row filter is not counted twice. `process_deletes` matches by path, so a duplicate is
        // harmless there, but the summary counts must stay accurate.
        if let Some(row_filter) = &self.row_filter {
            let filter_deletes = snapshot_produce
                .resolve_filter_deletes(row_filter, self.case_sensitive)
                .await?;
            let mut seen: HashSet<String> = resolved
                .iter()
                .map(|data_file| data_file.file_path().to_string())
                .collect();
            for data_file in filter_deletes {
                if seen.insert(data_file.file_path().to_string()) {
                    resolved.push(data_file);
                }
            }
        }

        Ok(resolved)
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        // Expose EVERY current manifest, DATA and DELETE. `process_deletes` decides per DATA manifest
        // whether to rewrite, carry forward, or drop it. Every DELETE manifest carries forward unchanged,
        // so a delete on a merge-on-read table keeps all outstanding position and equality deletes
        // instead of dropping them and resurrecting deleted rows.
        snapshot_produce.current_manifests().await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use futures::TryStreamExt;

    use crate::expr::{Predicate, Reference};
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal,
        ManifestContentType, ManifestStatus, Operation, Struct,
    };
    use crate::table::Table;
    use crate::transaction::tests::{
        make_v2_minimal_table_in_catalog, make_v3_minimal_table_in_catalog,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::position_delete_writer::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
    };
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    use crate::{Catalog, ErrorKind};

    /// Build a data file with a unique path, routed to partition `x = part_value`.
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

    /// Build a data file in partition `x = part_value` whose column `y` carries `[y_lower, y_upper]`
    /// bounds and no null or nan counts. The bounds drive the inclusive evaluator, which discriminates
    /// KEEP from PARTIAL. Without the zero counts the strict evaluator returns "might not match", so a
    /// file from this helper never classifies as DELETE on a `y` predicate.
    fn data_file_with_y_bounds(
        path: &str,
        part_value: i64,
        y_lower: i64,
        y_upper: i64,
    ) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .lower_bounds(HashMap::from([(2, Datum::long(y_lower))]))
            .upper_bounds(HashMap::from([(2, Datum::long(y_upper))]))
            .build()
            .unwrap()
    }

    /// Like [`data_file_with_y_bounds`] but with complete `y` stats: bounds plus `value_counts` and zero
    /// null and nan counts. The zero counts let the strict evaluator classify the file as
    /// strictly-all-match, which the DELETE branch needs. The strict evaluator returns "might not match"
    /// whenever a column might hold a null or a nan.
    fn data_file_with_y_stats(path: &str, part_value: i64, y_lower: i64, y_upper: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .value_counts(HashMap::from([(2, 1)]))
            .null_value_counts(HashMap::from([(2, 0)]))
            .nan_value_counts(HashMap::from([(2, 0)]))
            .lower_bounds(HashMap::from([(2, Datum::long(y_lower))]))
            .upper_bounds(HashMap::from([(2, Datum::long(y_upper))]))
            .build()
            .unwrap()
    }

    /// Build a data file in a two-field partition `(x = x_value, y = y_value)` under partition spec
    /// `spec_id`. It carries NO `y` bounds, so only the partition residual can decide a `y` predicate.
    /// Under the 2-field spec the residual of `y == y_value` is `alwaysTrue`, which is DELETE. Under the
    /// base 1-field spec it stays `y == y_value` with no bounds, which is PARTIAL and errors. That
    /// difference separates "uses this file's OWN spec" from "hard-codes spec 0".
    fn data_file_spec1_xy(path: &str, spec_id: i32, x_value: i64, y_value: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(spec_id)
            .partition(Struct::from_iter([
                Some(Literal::long(x_value)),
                Some(Literal::long(y_value)),
            ]))
            .build()
            .unwrap()
    }

    /// Collect the live data file paths in the current snapshot. This is what a scan reads.
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

    /// Commit a concurrent `delete_files` through the catalog, in its own `Delete` snapshot. This is the
    /// removal a files-exist check must detect.
    async fn commit_concurrent_delete(
        catalog: &impl Catalog,
        table: &Table,
        paths: impl IntoIterator<Item = &str>,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx
            .delete_files()
            .delete_files(paths.into_iter().map(str::to_string));
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// The deleted file's entry must be present as `Deleted` in the rewritten manifest, with correct
    /// existing and deleted counts. Wrong counts are manifest corruption.
    #[tokio::test]
    async fn test_delete_files_marks_entry_deleted_and_counts_are_correct() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx.delete_files().delete_file("test/b.parquet");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert_eq!(manifest_list.entries().len(), 1);
        let manifest_file = &manifest_list.entries()[0];
        assert_eq!(manifest_file.existing_files_count, Some(1));
        assert_eq!(manifest_file.deleted_files_count, Some(1));
        assert_eq!(manifest_file.added_files_count, Some(0));

        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        let mut statuses: Vec<(String, ManifestStatus)> = manifest
            .entries()
            .iter()
            .map(|entry| (entry.file_path().to_string(), entry.status()))
            .collect();
        statuses.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(statuses, vec![
            ("test/a.parquet".to_string(), ManifestStatus::Existing),
            ("test/b.parquet".to_string(), ManifestStatus::Deleted),
        ]);
    }

    /// A delete that targets a file in only ONE of two manifests must carry the other forward unchanged,
    /// at the same manifest path. Rewriting an untouched manifest risks corrupting it.
    #[tokio::test]
    async fn test_delete_files_carries_untouched_manifest_forward_unchanged() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
        let table = append_files(&catalog, &table, vec![data_file("test/b.parquet", 0)]).await;

        // Find the manifest that holds A. The delete must not touch it.
        let before = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut a_manifest_path = None;
        for manifest_file in before.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            if manifest
                .entries()
                .iter()
                .any(|e| e.file_path() == "test/a.parquet")
            {
                a_manifest_path = Some(manifest_file.manifest_path.clone());
            }
        }
        let a_manifest_path = a_manifest_path.expect("A's manifest should exist");

        // Delete B.
        let tx = Transaction::new(&table);
        let action = tx.delete_files().delete_file("test/b.parquet");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let after = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            after
                .entries()
                .iter()
                .any(|m| m.manifest_path == a_manifest_path),
            "the manifest untouched by the delete must be carried forward unchanged"
        );
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/a.parquet".to_string()])
        );
    }

    /// A delete that spans multiple manifests must remove the targeted file from each, not only from the
    /// first matching manifest.
    #[tokio::test]
    async fn test_delete_files_across_multiple_manifests() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
        let table = append_files(&catalog, &table, vec![data_file("test/b.parquet", 0)]).await;
        let table = append_files(&catalog, &table, vec![data_file("test/c.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_files(["test/a.parquet", "test/c.parquet"]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/b.parquet".to_string()])
        );
    }

    /// Deleting EVERY live file in a manifest must leave an empty live set and must not error.
    #[tokio::test]
    async fn test_delete_all_files_in_a_manifest_leaves_empty_live_set() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_files(["test/a.parquet", "test/b.parquet"]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert!(live_file_paths(&table).await.is_empty());
    }

    /// A delete-only commit, with no added files, is allowed.
    #[tokio::test]
    async fn test_delete_only_commit_is_allowed() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx.delete_files().delete_file("test/a.parquet");
        let tx = action.apply(tx).unwrap();
        let result = tx.commit(&catalog).await;

        assert!(result.is_ok(), "delete-only commit should be allowed");
        assert!(live_file_paths(&result.unwrap()).await.is_empty());
    }

    /// A truly empty delete commit is rejected, as Java `SnapshotProducer` rejects one. The relaxation
    /// that allows delete-only commits must not also allow a no-op snapshot.
    #[tokio::test]
    async fn test_empty_delete_commit_is_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx.delete_files();
        let tx = action.apply(tx).unwrap();
        let result = tx.commit(&catalog).await;

        assert!(
            result.is_err(),
            "a truly-empty delete commit must be rejected"
        );
    }

    /// Deleting a file that is NOT in the table is an error (Java `failMissingDeletePaths`). A silent
    /// drop of the unmatched path produces a no-op delete snapshot.
    #[tokio::test]
    async fn test_delete_absent_file_errors() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx.delete_files().delete_file("test/does-not-exist.parquet");
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("absent file must error");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("Missing required files to delete"),
            "unexpected error message: {}",
            error.message()
        );
    }

    /// Every SURVIVING entry must be copied forward as `Existing` with its original `snapshot_id`,
    /// `sequence_number`, and `file_sequence_number`. The `Deleted` tombstone takes the new snapshot id
    /// but keeps the removed file's original seqs. Re-stamping a survivor is silent table corruption: it
    /// breaks merge-on-read delete application and incremental scans. Every other delete_files test
    /// asserts only paths, statuses, and counts, so all of them pass under a snapshot-id re-stamp.
    #[tokio::test]
    async fn test_delete_preserves_surviving_entry_provenance_across_snapshots() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
        let s1 = table.metadata().current_snapshot().unwrap().snapshot_id();

        let table = append_files(&catalog, &table, vec![
            data_file("test/b.parquet", 0),
            data_file("test/c.parquet", 0),
        ])
        .await;
        let s2 = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(s1, s2);

        let (a_snap, a_seq, a_fseq) = entry_provenance(&table, "test/a.parquet").await;
        let (b_snap, b_seq, b_fseq) = entry_provenance(&table, "test/b.parquet").await;
        assert_eq!(a_snap, Some(s1), "A added by S1");
        assert_eq!(b_snap, Some(s2), "B added by S2");
        assert_ne!(a_seq, b_seq, "A and B must have different data seq numbers");

        // Deleting B rewrites S2's manifest. The surviving C must keep S2 and seq 2.
        let tx = Transaction::new(&table);
        let action = tx.delete_files().delete_file("test/b.parquet");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s3 = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(s3, s2);

        let (c_snap, c_seq, c_fseq) = entry_provenance(&table, "test/c.parquet").await;
        assert_eq!(
            c_snap,
            Some(s2),
            "surviving C must keep its ORIGINAL snapshot id S2, not S3"
        );
        assert_eq!(
            c_seq, b_seq,
            "surviving C must keep its ORIGINAL data seq, not the delete seq"
        );
        assert_eq!(
            c_fseq, b_fseq,
            "surviving C must keep its ORIGINAL file seq"
        );

        let (a2_snap, a2_seq, a2_fseq) = entry_provenance(&table, "test/a.parquet").await;
        assert_eq!(a2_snap, Some(s1), "carried-forward A keeps S1");
        assert_eq!(a2_seq, a_seq, "carried-forward A keeps its data seq");
        assert_eq!(a2_fseq, a_fseq, "carried-forward A keeps its file seq");

        // B's tombstone carries the new snapshot id S3 but keeps B's original data and file seqs.
        let del = deleted_entry_provenance(&table, "test/b.parquet").await;
        assert_eq!(
            del.0,
            Some(s3),
            "the Deleted tombstone for B gets the new snapshot id S3"
        );
        assert_eq!(
            del.1, b_seq,
            "the Deleted tombstone keeps B's original data seq"
        );
        assert_eq!(
            del.2, b_fseq,
            "the Deleted tombstone keeps B's original file seq"
        );
    }

    /// The all-deleted-manifest lifecycle (Java `MergingSnapshotProducer.apply`). The commit that
    /// creates an all-deleted manifest KEEPS it, because its `added_snapshot_id` equals the new snapshot
    /// id, so its tombstones survive one snapshot. The next delete commit then DROPS it. Dropping the
    /// tombstones early loses records that expiry needs. Keeping the manifest forever is clutter.
    #[tokio::test]
    async fn test_all_deleted_manifest_kept_by_creating_commit_then_dropped_by_next() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;
        let table = append_files(&catalog, &table, vec![data_file("test/c.parquet", 0)]).await;

        // Deleting A and B makes M1 all-deleted. It must be kept, with both tombstones.
        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_files(["test/a.parquet", "test/b.parquet"]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let manifest_list = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut total_deleted = 0;
        let mut total_existing = 0;
        let mut total_added = 0;
        for mf in manifest_list.entries() {
            let m = mf.load_manifest(table.file_io()).await.unwrap();
            for e in m.entries() {
                match e.status() {
                    ManifestStatus::Deleted => total_deleted += 1,
                    ManifestStatus::Existing => total_existing += 1,
                    ManifestStatus::Added => total_added += 1,
                }
            }
        }
        assert_eq!(
            total_deleted, 2,
            "all-deleted manifest must be KEPT with its tombstones"
        );
        assert_eq!(total_added + total_existing, 1, "only C is live");

        // A new commit that deletes C must now drop the all-deleted M1'.
        let tx = Transaction::new(&table);
        let action = tx.delete_files().delete_file("test/c.parquet");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let manifest_list = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut a_or_b_tombstone = false;
        for mf in manifest_list.entries() {
            let m = mf.load_manifest(table.file_io()).await.unwrap();
            for e in m.entries() {
                if e.file_path() == "test/a.parquet" || e.file_path() == "test/b.parquet" {
                    a_or_b_tombstone = true;
                }
            }
        }
        assert!(
            !a_or_b_tombstone,
            "the all-deleted M1' must be DROPPED by the next commit (no live files)"
        );
        assert!(live_file_paths(&table).await.is_empty());
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

    /// Return (snapshot_id, sequence_number, file_sequence_number) of the DELETED entry for `path`.
    async fn deleted_entry_provenance(
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
                if entry.status() == ManifestStatus::Deleted && entry.file_path() == path {
                    return (
                        entry.snapshot_id(),
                        entry.sequence_number(),
                        entry.file_sequence_number,
                    );
                }
            }
        }
        panic!("no deleted entry for {path}");
    }

    /// A delete that targets a present file AND an absent file must still error. The present file must
    /// not be removed while the absent one is ignored.
    #[tokio::test]
    async fn test_delete_mixed_present_and_absent_errors() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_files(["test/a.parquet", "test/absent.parquet"]);
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("mixed delete must error");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(error.message().contains("test/absent.parquet"));
    }

    // Files-exist conflict validation (Java `MergingSnapshotProducer.validateDataFilesExist`).
    //
    // The race: a `delete_files` is built against head S0, then a separate commit deletes a live data file
    // and advances the head to S1. `do_commit` refreshes to S1 and runs `validate` against it. With the
    // check enabled, a concurrent removal of a file THIS action deletes must fail the commit, because
    // committing over a vanished required file violates serializable isolation. A concurrent removal of a
    // different file must not fail. With the check off, neither fails.

    /// With the check enabled and nothing landing concurrently, the delete commits normally. The check
    /// must not block a race-free commit.
    #[tokio::test]
    async fn test_delete_files_exist_validation_no_concurrent_deletion_succeeds() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;
        let s0 = table.metadata().current_snapshot().unwrap().snapshot_id();

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_file("test/a.parquet")
            .validate_from_snapshot(s0)
            .validate_files_exist();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a race-free delete must commit even with the files-exist check enabled");

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/b.parquet".to_string()]),
            "the delete applied: only b survives"
        );
    }

    /// A concurrent `delete_files(a)` removes the same file this action deletes. The commit must fail
    /// with a non-retryable `DataInvalid` naming `a`. Without the check the re-based delete no-ops, or
    /// fails with the generic "missing required files to delete" instead.
    #[tokio::test]
    async fn test_delete_files_exist_rejects_concurrent_deletion_of_same_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;
        let s0 = table.metadata().current_snapshot().unwrap().snapshot_id();

        // Build delete(a) with the check enabled, pinned to start at S0.
        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_file("test/a.parquet")
            .validate_from_snapshot(s0)
            .validate_files_exist();
        let tx = action.apply(tx).unwrap();

        // The concurrent commit S1 removes the same file a.
        let _table_after_concurrent =
            commit_concurrent_delete(&catalog, &table, ["test/a.parquet"]).await;

        let err = tx.commit(&catalog).await.expect_err(
            "delete must fail: a concurrent commit removed the file this delete also requires",
        );

        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "a files-exist conflict is a non-retryable validation failure (DataInvalid)"
        );
        assert!(
            !err.retryable(),
            "the validation failure must be NON-retryable so the retry loop stops and it propagates \
             (it is NOT a retry-exhausted CatalogCommitConflicts)"
        );
        assert!(
            err.message().contains("Cannot commit, missing data files"),
            "the error must use the validateDataFilesExist wording, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/a.parquet"),
            "the error must NAME the missing file, got: {}",
            err.message()
        );

        // The catalog head is still S1. The conflicting delete did not commit.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from(["test/b.parquet".to_string()]),
            "only the concurrent delete applied: b survives, a is gone"
        );
    }

    /// Commit a concurrent compaction that rewrites `delete_path` into `add_path`. Its snapshot records
    /// `Operation::Replace`, and the rewritten file gets a `Deleted` tombstone.
    async fn commit_concurrent_replace_compaction(
        catalog: &impl Catalog,
        table: &Table,
        delete_path: &str,
        add_path: &str,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx.rewrite_files(vec![data_file(delete_path, 0)], vec![data_file(
            add_path, 0,
        )]);
        let tx = action.apply(tx).expect("rewrite_files action applies");
        tx.commit(catalog)
            .await
            .expect("the concurrent compaction commit must succeed")
    }

    /// The `skip_deletes == false` arm. `DeleteFiles` always validates against
    /// `operation_removes_data_files`, which is Java's `{overwrite, replace, delete}` set, not the
    /// predicate `RowDelta` uses by default. A concurrent compaction rewrites `a`, so the commit must
    /// fail with the non-retryable error naming `a`.
    ///
    /// The discriminating mutation: drop `Operation::Replace` from `operation_removes_data_files`. The
    /// compaction's tombstone is then never inspected. The test asserts the validation message is present
    /// AND the generic path-resolution message is absent, so the mutant cannot pass by failing elsewhere.
    #[tokio::test]
    async fn test_delete_files_exist_rejects_concurrent_replace_compaction_of_same_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;
        let s0 = table
            .metadata()
            .current_snapshot()
            .expect("S0 exists")
            .snapshot_id();

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_file("test/a.parquet")
            .validate_from_snapshot(s0)
            .validate_files_exist();
        let tx = action.apply(tx).expect("delete_files action applies");

        // The concurrent commit S1 is a compaction that replaces the file this delete requires.
        let compacted = commit_concurrent_replace_compaction(
            &catalog,
            &table,
            "test/a.parquet",
            "test/a-compacted.parquet",
        )
        .await;

        // The concurrent snapshot really is a Replace, and `a` really is gone.
        assert_eq!(
            compacted
                .metadata()
                .current_snapshot()
                .expect("the compaction produced a snapshot")
                .summary()
                .operation,
            Operation::Replace,
            "the concurrent compaction must record Operation::Replace — otherwise this test would \
             exercise a different op-set member and prove nothing about REPLACE"
        );
        let live = live_file_paths(&compacted).await;
        assert!(
            !live.contains("test/a.parquet"),
            "the compaction removed a, live = {live:?}"
        );

        let err = tx.commit(&catalog).await.expect_err(
            "delete must fail: a concurrent REPLACE (compaction) removed the file it requires",
        );

        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "a files-exist conflict is a non-retryable validation failure (DataInvalid)"
        );
        assert!(
            !err.retryable(),
            "the validation failure must be NON-retryable so the retry loop stops and it propagates"
        );
        assert!(
            err.message().contains("Cannot commit, missing data files"),
            "the error must be the validateDataFilesExist rejection, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/a.parquet"),
            "the error must NAME the missing file, got: {}",
            err.message()
        );
        assert!(
            !err.message().contains("Missing required files to delete"),
            "the rejection must come from validateDataFilesExist, NOT from the later generic \
             path-resolution check, got: {}",
            err.message()
        );
    }

    /// The concurrent deletion removes a different file than this action deletes, so the check passes and
    /// the commit succeeds. An over-eager check that rejects any concurrent deletion would break
    /// legitimate concurrent deletes of disjoint files.
    #[tokio::test]
    async fn test_delete_files_exist_allows_concurrent_deletion_of_different_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
            data_file("test/c.parquet", 0),
        ])
        .await;
        let s0 = table.metadata().current_snapshot().unwrap().snapshot_id();

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_file("test/a.parquet")
            .validate_from_snapshot(s0)
            .validate_files_exist();
        let tx = action.apply(tx).unwrap();

        // The concurrent commit S1 removes b, which does not race a's deletion.
        let _ = commit_concurrent_delete(&catalog, &table, ["test/b.parquet"]).await;

        let table = tx.commit(&catalog).await.expect(
            "delete must succeed: the concurrent deletion removed a different file (b), not a",
        );

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/c.parquet".to_string()]),
            "a and b both deleted (this + concurrent), c survives"
        );
    }

    /// Without `validate_files_exist()`, a concurrent deletion of the same file does not raise the
    /// validation error. That is snapshot isolation, the default. The re-based delete instead reports the
    /// generic missing-file error from path resolution, which proves the validation is opt-in.
    #[tokio::test]
    async fn test_delete_files_exist_off_does_not_run_validation() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;

        // Build delete(a) without the check.
        let tx = Transaction::new(&table);
        let action = tx.delete_files().delete_file("test/a.parquet");
        let tx = action.apply(tx).unwrap();

        // The concurrent commit S1 removes the same file a.
        let _ = commit_concurrent_delete(&catalog, &table, ["test/a.parquet"]).await;

        // With the check off, the re-based delete fails the generic path-resolution check instead.
        let err = tx.commit(&catalog).await.expect_err(
            "with the check OFF, the re-based delete still cannot resolve the vanished file",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Missing required files to delete"),
            "with validation OFF, the failure is the generic path-resolution error, not \
             validateDataFilesExist — got: {}",
            err.message()
        );
        assert!(
            !err.message().contains("Cannot commit, missing data files"),
            "the validateDataFilesExist message must NOT appear when the check is OFF: {}",
            err.message()
        );
    }

    /// `validate_from_snapshot(id)` changes which commits count as concurrent. With S0, the earlier
    /// snapshot, S1's removal is in the window and the check fails. With S1, the current head, the window
    /// is empty and the commit instead fails later on path resolution. Ignoring the override would change
    /// which concurrent removals are detected.
    #[tokio::test]
    async fn test_delete_files_exist_validate_from_snapshot_override_changes_window() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;
        let s0 = table.metadata().current_snapshot().unwrap().snapshot_id();

        // S1 is a concurrent delete of a, which advances the head.
        let table_s1 = commit_concurrent_delete(&catalog, &table, ["test/a.parquet"]).await;
        let s1 = table_s1
            .metadata()
            .current_snapshot()
            .unwrap()
            .snapshot_id();
        assert_ne!(s0, s1);

        // With S0 the window includes S1's removal, so the check fails naming a.
        let tx = Transaction::new(&table_s1);
        let action = tx
            .delete_files()
            .delete_file("test/a.parquet")
            .validate_from_snapshot(s0)
            .validate_files_exist();
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "validate_from_snapshot(S0) includes S1's removal of a ⇒ files-exist conflict",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message().contains("Cannot commit, missing data files")
                && err.message().contains("test/a.parquet"),
            "the override-widened window must surface the validateDataFilesExist error naming a: {}",
            err.message()
        );

        // With S1 the window is empty, so the commit fails on the generic path-resolution error instead.
        let tx = Transaction::new(&table_s1);
        let action = tx
            .delete_files()
            .delete_file("test/a.parquet")
            .validate_from_snapshot(s1)
            .validate_files_exist();
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "a is already gone, so even with an empty validation window the path cannot resolve",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Missing required files to delete")
                && !err.message().contains("Cannot commit, missing data files"),
            "validate_from_snapshot(S1) empties the window ⇒ the failure is path resolution, not \
             validateDataFilesExist — got: {}",
            err.message()
        );
    }

    /// The check must work without `validate_from_snapshot`, on the transaction-captured starting
    /// snapshot alone. `do_commit` overwrites `self.table` with the refreshed base, but
    /// `starting_snapshot_id` must survive, so S1's removal of `a` is still enumerated and rejected.
    ///
    /// The discriminating mutation: read `effective_start` from the refreshed head instead. The window is
    /// then empty, the check always passes, and the commit fails on the generic path-resolution error.
    /// Every other files-exist test pins `validate_from_snapshot`, which short-circuits
    /// `validate_from_snapshot.or(starting_snapshot_id)`, so only this test covers the capture.
    #[tokio::test]
    async fn test_delete_files_exist_rejects_concurrent_using_tx_captured_starting_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;

        // Build delete(a) with the check enabled and no validate_from_snapshot. The start is S0.
        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_file("test/a.parquet")
            .validate_files_exist();
        let tx = action.apply(tx).unwrap();

        // The concurrent commit S1 removes the same file a and advances the head.
        let _concurrent = commit_concurrent_delete(&catalog, &table, ["test/a.parquet"]).await;

        // The tx-captured start survives the re-base, so S1's removal is in the window and the check fires.
        let err = tx.commit(&catalog).await.expect_err(
            "conflict must be detected via the tx-captured starting snapshot (no validate_from_snapshot)",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message().contains("Cannot commit, missing data files")
                && err.message().contains("test/a.parquet"),
            "the tx-captured window must surface the validateDataFilesExist error naming a (NOT the generic \
             path-resolution error) — got: {}",
            err.message()
        );
    }

    // Merge-on-read delete-manifest carry.
    //
    // `existing_manifest` returns the full manifest list, DATA and DELETE, so a `delete_files` commit on a
    // table with outstanding position or equality deletes keeps those delete manifests. These tests use
    // real parquet, a real position-delete file from the production writer, and a real scan, so the
    // resurrection physics is proven end-to-end.

    /// Write a real parquet data file with rows `(x, y, z)`, routed to partition `x = part_value`.
    async fn write_data_file(
        table: &Table,
        file_name: &str,
        part_value: i64,
        rows: &[(i64, i64, i64)],
    ) -> DataFile {
        use crate::arrow::schema_to_arrow_schema;

        let schema = table.metadata().current_schema();
        let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());

        let xs: Vec<i64> = rows.iter().map(|(x, _, _)| *x).collect();
        let ys: Vec<i64> = rows.iter().map(|(_, y, _)| *y).collect();
        let zs: Vec<i64> = rows.iter().map(|(_, _, z)| *z).collect();
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from(xs)) as ArrayRef,
            Arc::new(Int64Array::from(ys)) as ArrayRef,
            Arc::new(Int64Array::from(zs)) as ArrayRef,
        ])
        .unwrap();

        let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
        let output = table.file_io().new_output(file_path).unwrap();
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            schema.clone(),
        );
        let mut writer = parquet_builder.build(output).await.unwrap();
        writer.write(&batch).await.unwrap();
        let data_file_builders = writer.close().await.unwrap();

        let mut builder = data_file_builders.into_iter().next().unwrap();
        builder
            .content(DataContentType::Data)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .build()
            .unwrap()
    }

    /// Write a real position-delete parquet file for the given `(data_file_path, pos)` pairs, in
    /// partition `x = part_value`.
    async fn write_position_delete_file(
        table: &Table,
        part_value: i64,
        deletes: &[(String, i64)],
    ) -> DataFile {
        use arrow_array::StringArray;

        let config = PositionDeleteWriterConfig::new().unwrap();
        let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
        let file_name_gen = DefaultFileNameGenerator::new(
            "pos-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            config.schema().clone(),
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );
        let partition_key = crate::spec::PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
            .build(Some(partition_key))
            .await
            .unwrap();

        let paths: Vec<&str> = deletes.iter().map(|(p, _)| p.as_str()).collect();
        let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
        let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions)) as ArrayRef,
        ])
        .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap().into_iter().next().unwrap()
    }

    /// Scan the table and collect the `y` values. This is the read-side signal, with merge-on-read
    /// deletes applied.
    async fn scan_y_values(table: &Table) -> HashSet<i64> {
        let stream = table
            .scan()
            .select(["y"])
            .build()
            .unwrap()
            .to_arrow()
            .await
            .unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let mut values = HashSet::new();
        for batch in batches {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for index in 0..col.len() {
                values.insert(col.value(index));
            }
        }
        values
    }

    /// Count the DELETE-content manifests in the current snapshot. This is a structural signal,
    /// independent of the read path.
    async fn count_delete_manifests(table: &Table) -> usize {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        manifest_list
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Deletes)
            .count()
    }

    /// A `delete_files` on a merge-on-read table must not drop the outstanding delete manifests, which
    /// would resurrect deleted rows table-wide. `delete_files(Y)` must remove Y and keep X's position
    /// delete applying, so the scan is exactly {10}.
    ///
    /// The discriminating mutation, run manually: in `DeleteFilesOperation::existing_manifest`, filter
    /// `current_manifests()` to DATA manifests only. y=20 then resurrects and the delete-manifest count
    /// drops to 0.
    #[tokio::test]
    async fn test_delete_files_preserves_outstanding_delete_manifests_no_resurrection() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        // X holds y = [10, 20] in partition 0. Y holds y = [60, 70] in partition 1.
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let x_path = x.file_path().to_string();
        let y = write_data_file(&table, "y.parquet", 1, &[(1, 60, 600), (1, 70, 700)]).await;
        let y_path = y.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x, y]).await;

        // Row-delta a real position delete that masks X's row at position 1, which is y=20.
        let pos_delete = write_position_delete_file(&table, 0, &[(x_path.clone(), 1)]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![pos_delete]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            count_delete_manifests(&table).await,
            1,
            "the row_delta must leave one delete manifest in the snapshot"
        );

        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 60, 70]),
            "the position delete masks y=20 from X; Y's rows are present"
        );

        // Deleting Y must not drop X's outstanding delete manifest.
        let tx = Transaction::new(&table);
        let action = tx.delete_files().delete_file(&y_path);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Delete
        );

        // X's masked y=20 stays absent and Y's rows are gone, so the scan is exactly {10}.
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10]),
            "Y's rows are deleted AND X's masked y=20 stays absent — no resurrection"
        );

        // The delete manifest survived the commit.
        assert_eq!(
            count_delete_manifests(&table).await,
            1,
            "the delete_files commit must carry the outstanding delete manifest forward (not drop it)"
        );
    }

    // stage_only() on a delete-bearing action. A staged delete adds its `Delete` snapshot to metadata but
    // moves no ref, so the deleted rows stay visible to readers until the snapshot is published.

    /// A staged delete adds its `Delete` snapshot to metadata but leaves `main`, current-snapshot-id, and
    /// the snapshot log unchanged. A scan still sees the row it would remove. A staged delete that
    /// published early would drop rows from readers before the audit.
    #[tokio::test]
    async fn test_delete_files_stage_only_stages_without_moving_main() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;
        let base_id = table.metadata().current_snapshot_id();
        let base_log = table.metadata().history().to_vec();
        let base_snapshot_count = table.metadata().snapshots().count();

        let tx = Transaction::new(&table);
        let action = tx.delete_files().stage_only().delete_file("test/a.parquet");
        let tx = action.apply(tx).unwrap();
        let staged_table = tx.commit(&catalog).await.unwrap();
        let reloaded = catalog.load_table(staged_table.identifier()).await.unwrap();
        let metadata = reloaded.metadata();

        // A new snapshot was added, but main did not move.
        assert_eq!(
            metadata.snapshots().count(),
            base_snapshot_count + 1,
            "the staged delete snapshot must be added to metadata"
        );
        assert_eq!(
            metadata.current_snapshot_id(),
            base_id,
            "stage_only on a delete must NOT advance current-snapshot-id"
        );
        assert_eq!(
            metadata.snapshot_for_ref("main").unwrap().snapshot_id(),
            base_id.unwrap(),
            "stage_only on a delete must NOT move the main ref"
        );
        assert_eq!(
            metadata.history().to_vec(),
            base_log,
            "stage_only on a delete must NOT add a snapshot-log entry"
        );
        let staged_snapshot = metadata
            .snapshots()
            .find(|s| Some(s.snapshot_id()) != base_id)
            .unwrap();
        assert_eq!(staged_snapshot.summary().operation, Operation::Delete);

        // The readable live set still includes a, so the staged delete is hidden.
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from(["test/a.parquet".to_string(), "test/b.parquet".to_string()]),
            "the staged delete does not remove a from the readable table; both files stay live"
        );
    }

    // Delete by row filter (Java `StreamingDelete.deleteFromRowFilter`). Per live data file,
    // `resolve_filter_deletes` reduces the predicate to its per-partition residual under the file's own
    // spec. It then deletes when strict metrics say all rows match, keeps when inclusive says none
    // match, and errors on a partial match. `StreamingDelete` deletes whole files only, so a partial match is a hard
    // error, never a silent partial delete. The recorded operation stays the constant `Delete`.

    /// All three classifications under one filter `y >= 50`. A is strictly covered and would be deleted.
    /// B provably cannot match and is KEPT. C straddles 50 and is PARTIAL. Because C is partial, the
    /// whole commit fails and nothing is committed. A partially-matching file is never silently dropped,
    /// and the failure is atomic, so A is not deleted either.
    #[tokio::test]
    async fn test_delete_from_row_filter_partial_file_is_never_silently_dropped() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            // A: complete stats, so strict metrics prove all rows >= 50. DELETE.
            data_file_with_y_stats("test/a.parquet", 0, 60, 70),
            // B: upper bound 10 is below 50, so inclusive says no rows match. KEEP.
            data_file_with_y_bounds("test/b.parquet", 1, 0, 10),
            // C: straddles 50, so inclusive says yes and strict says no. PARTIAL.
            data_file_with_y_bounds("test/c.parquet", 0, 40, 60),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)));
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a partial (some-but-not-all) row match must fail the whole delete");

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            !err.retryable(),
            "a partial-match delete is a non-retryable validation error"
        );
        assert!(
            err.message()
                .contains("Cannot delete file where some, but not all, rows match filter"),
            "must match Java's ManifestFilterManager message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/c.parquet"),
            "the error must name the offending partial file C, got: {}",
            err.message()
        );

        // The failed commit changed nothing. All three files remain live.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from([
                "test/a.parquet".to_string(),
                "test/b.parquet".to_string(),
                "test/c.parquet".to_string(),
            ]),
            "the partial-match failure is atomic: A is not deleted, B and C survive"
        );
    }

    /// The DELETE branch. With no partial file present, `delete_from_row_filter(y >= 50)` deletes the
    /// strictly-covered file A, and the recorded operation is the constant `Delete`, not Overwrite.
    #[tokio::test]
    async fn test_delete_from_row_filter_deletes_strictly_matching_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file_with_y_stats("test/a.parquet", 0, 60, 70),
            // A keep-anchor with y entirely below 50, so the commit is non-empty and it survives.
            data_file_with_y_bounds("test/keep.parquet", 1, 0, 10),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)));
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a strictly-covered file must be deleted by the row filter");

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/keep.parquet".to_string()]),
            "A (all rows y>=50) is deleted; the y<50 anchor survives"
        );
        // A by-row-filter delete is still a plain Delete.
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Delete,
            "a delete-by-row-filter records Delete (Java StreamingDelete.operation() == \"delete\")"
        );
    }

    /// The KEEP branch. `delete_from_row_filter` must not delete a file whose per-partition residual
    /// cannot match. Under `x == 0`, file B in partition x=1 has residual `alwaysFalse`, so inclusive says
    /// no rows match and B survives.
    #[tokio::test]
    async fn test_delete_from_row_filter_keeps_residual_non_matching_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            // x=0: the residual of `x==0` is alwaysTrue, so strict metrics match. DELETE.
            data_file("test/del.parquet", 0),
            // x=1: the residual is alwaysFalse, so inclusive says no rows match. KEEP.
            data_file("test/keep.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Reference::new("x").equal_to(Datum::long(0)));
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("x==0 deletes the x=0 file and keeps the x=1 file");

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/keep.parquet".to_string()]),
            "the x=1 file (residual alwaysFalse) is KEPT; only the x=0 file is deleted"
        );
    }

    /// A file in partition x=0 carries NO `x` bounds, yet `delete_from_row_filter(x == 0)` must delete
    /// it. The residual for partition x=0 is `alwaysTrue`, which strict metrics satisfy trivially.
    ///
    /// The discriminating mutation: run the metrics on the FULL `x == 0` predicate. With no `x` bounds the
    /// file then classifies as PARTIAL and the commit errors.
    #[tokio::test]
    async fn test_delete_from_row_filter_uses_partition_residual_not_full_predicate() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // This file has no column bounds. Only the partition value identifies it as x=0.
        let table = append_files(&catalog, &table, vec![
            data_file("test/no-bounds.parquet", 0),
            data_file("test/other.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Reference::new("x").equal_to(Datum::long(0)));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.expect(
            "the no-`x`-bounds file in partition x=0 must DELETE via the alwaysTrue partition residual, \
             not error as a partial match",
        );

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/other.parquet".to_string()]),
            "the partition-residual makes the no-bounds x=0 file deletable; x=1 survives"
        );
    }

    /// `delete_from_row_filter(AlwaysTrue)` deletes every live data file and leaves an empty live set.
    #[tokio::test]
    async fn test_delete_from_row_filter_always_true_deletes_all() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Predicate::AlwaysTrue);
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("AlwaysTrue must delete every live data file");

        assert!(
            live_file_paths(&table).await.is_empty(),
            "AlwaysTrue deletes every file; the live set is empty"
        );
    }

    /// A by-filter delete that matches nothing resolves to an empty delete set, and the producer rejects
    /// the empty commit. It must not produce a no-op Delete snapshot.
    #[tokio::test]
    async fn test_delete_from_row_filter_matching_nothing_is_rejected_and_deletes_nothing() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Reference::new("x").equal_to(Datum::long(99)));
        let tx = action.apply(tx).unwrap();
        let result = tx.commit(&catalog).await;

        assert!(
            result.is_err(),
            "a by-filter delete that matches no file is an empty no-op commit and must be rejected"
        );
        // Nothing was deleted. Both files stay live.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from(["test/a.parquet".to_string(), "test/b.parquet".to_string()]),
            "the non-matching by-filter delete removed nothing"
        );
    }

    /// A by-path delete and a by-filter delete in one action remove both files, de-duped, in one Delete
    /// snapshot. The unmatched third file survives. Java carries both sets on one
    /// `ManifestFilterManager`.
    #[tokio::test]
    async fn test_delete_from_row_filter_combines_with_by_path_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 1),
            data_file("test/c.parquet", 0),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_file("test/a.parquet")
            .delete_from_row_filter(Reference::new("x").equal_to(Datum::long(1)));
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("by-path delete of a + by-filter delete of x=1 must both apply");

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/c.parquet".to_string()]),
            "a removed by path, b removed by filter (x=1); c (x=0, not named) survives"
        );
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Delete
        );
    }

    /// Provenance across snapshots for a by-filter delete. The filter `y >= 100` deletes B and keeps C in
    /// the same rewritten manifest. The surviving C must carry its original S2 snapshot id and sequence
    /// numbers, not the delete snapshot's.
    #[tokio::test]
    async fn test_delete_from_row_filter_preserves_surviving_entry_provenance() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file_with_y_stats(
            "test/a.parquet",
            0,
            0,
            5,
        )])
        .await;

        // S2 holds B, which the filter deletes, and C, which survives, in one manifest.
        let table = append_files(&catalog, &table, vec![
            data_file_with_y_stats("test/b.parquet", 0, 100, 110),
            data_file_with_y_bounds("test/c.parquet", 0, 0, 10),
        ])
        .await;
        let s2 = table.metadata().current_snapshot().unwrap().snapshot_id();
        let (c_snap_before, c_seq_before, c_fseq_before) =
            entry_provenance(&table, "test/c.parquet").await;
        assert_eq!(c_snap_before, Some(s2), "C added by S2");

        // The filter rewrites S2's manifest. C survives.
        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(100)));
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("y>=100 deletes B and keeps A and C");
        let s3 = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(s3, s2);

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/a.parquet".to_string(), "test/c.parquet".to_string()]),
            "B (y>=100) deleted; A and C survive"
        );

        // The surviving C keeps its original S2 provenance, not the delete snapshot S3.
        let (c_snap_after, c_seq_after, c_fseq_after) =
            entry_provenance(&table, "test/c.parquet").await;
        assert_eq!(
            c_snap_after,
            Some(s2),
            "surviving C must keep its ORIGINAL snapshot id S2, not the delete snapshot S3"
        );
        assert_eq!(
            c_seq_after, c_seq_before,
            "surviving C must keep its ORIGINAL data sequence number"
        );
        assert_eq!(
            c_fseq_after, c_fseq_before,
            "surviving C must keep its ORIGINAL file sequence number"
        );
    }

    /// The inherited divergence: a by-path delete plus a PARTIAL row-filter match on the same file.
    /// Java's `ManifestFilterManager.manifestHasDeletedFiles` marks a file for deletion by path before the
    /// metrics check. A marked file skips the classification and Java deletes it outright.
    ///
    /// The fork errors instead. `SnapshotProducer::resolve_filter_deletes` classifies every live data file
    /// by metrics with no knowledge of the by-path set, so file C raises the non-retryable partial-match
    /// error where Java would delete it.
    ///
    /// When the deferred `snapshot.rs` fix lands, this assertion flips from an error to a deletion.
    /// Update this test in the same change, so the divergence cannot close silently.
    #[tokio::test]
    async fn test_delete_from_row_filter_bypath_and_partial_match_diverges_failsafe() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // C straddles 50, so it is PARTIAL for `y >= 50`. It is also named in `delete_file` below, which
        // in Java would mark it for delete by path and skip the metrics check.
        let table = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/c.parquet",
            0,
            40,
            60,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            // C is requested by path.
            .delete_file("test/c.parquet")
            // C is also a PARTIAL match for this filter. The shared helper has no short-circuit, so it
            // errors.
            .delete_from_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)));
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "INHERITED DIVERGENCE: Rust errors on a by-path file that is ALSO a partial filter match, \
             where Java's markedForDelete short-circuit would have deleted it (fail-safe: refuses the \
             commit, never wrongly deletes). The fix lives in snapshot.rs::resolve_filter_deletes — when \
             it lands, flip this assertion to expect the file deleted, in the SAME change.",
        );

        // The divergence is fail-safe: a non-retryable error naming the file, never a partial delete.
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "the inherited divergence surfaces as a non-retryable validation error, not data loss"
        );
        assert!(
            !err.retryable(),
            "the partial-match error is non-retryable (the retry loop must not spin on it)"
        );
        assert!(
            err.message()
                .contains("Cannot delete file where some, but not all, rows match filter"),
            "the divergence uses the shared ManifestFilterManager partial-match wording, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/c.parquet"),
            "the fail-safe error must NAME the by-path file Java would have deleted, got: {}",
            err.message()
        );

        // The refused commit changed nothing. C is still live.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from(["test/c.parquet".to_string()]),
            "the fail-safe rejection is atomic: C is not removed (no silent data loss in either direction)"
        );
    }

    /// The per-file partition spec is what reduces the predicate. `resolve_filter_deletes` builds one
    /// `ResidualEvaluator` per spec id and picks it by each file's own `partition_spec_id`. Every other
    /// row-filter test here and in `overwrite_files.rs` uses a single-spec table, so a mutation that
    /// hard-codes spec 0 would survive the whole suite.
    ///
    /// Spec 0 is `identity(x)` and spec 1 is `identity(x), identity(y)`. Under the filter `y == 99`,
    /// `s0-keep.parquet` keeps the residual `y == 99`, which its `y` bounds `[0,10]` exclude, so it is
    /// KEPT. `s1-del.parquet` has residual `alwaysTrue` under spec 1, so it is deleted. The commit
    /// succeeds.
    ///
    /// The discriminating mutation: hard-code the residual evaluator to spec 0. `s1-del.parquet` then
    /// keeps the full `y == 99` with no `y` bounds, classifies as PARTIAL, and the whole commit errors.
    #[tokio::test]
    async fn test_delete_from_row_filter_uses_each_files_own_partition_spec() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // A keep-anchor under spec 0, in partition x=5 with y bounds [0,10].
        let table = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/s0-keep.parquet",
            5,
            0,
            10,
        )])
        .await;

        // Evolve to a second, distinct spec id.
        let tx = Transaction::new(&table);
        let action = tx.update_partition_spec().add_field("y");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let spec1 = table.metadata().default_partition_spec_id();
        assert_ne!(spec1, 0, "fixture sanity: the spec evolved away from 0");
        assert_eq!(
            table.metadata().default_partition_spec().fields().len(),
            2,
            "fixture sanity: the evolved spec partitions by both x and y"
        );

        // A file under spec 1 in partition (x=0, y=99) with no y bounds. Only the spec-1 partition
        // residual can delete it.
        let table = append_files(&catalog, &table, vec![data_file_spec1_xy(
            "test/s1-del.parquet",
            spec1,
            0,
            99,
        )])
        .await;

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from([
                "test/s0-keep.parquet".to_string(),
                "test/s1-del.parquet".to_string(),
            ]),
        );

        // The spec-1 file deletes via its own spec's residual. The spec-0 file is kept.
        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Reference::new("y").equal_to(Datum::long(99)));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.expect(
            "the spec-1 file must DELETE via its OWN partition residual (y==99 ⇒ alwaysTrue); a hard-coded \
             spec-0 residual would make it a PARTIAL match and error",
        );

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/s0-keep.parquet".to_string()]),
            "the spec-1 file (partition y=99) is deleted via its own spec; the spec-0 keep-anchor survives"
        );
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Delete,
        );
    }

    /// A file removed by BOTH a path and the row filter must be counted once. `process_deletes` keys on a
    /// `HashSet`, so the live set is correct either way. The summary counters instead iterate the resolved
    /// `removed_data_files` one entry at a time, so a double push reports two deleted files and two
    /// deleted records for a single file. That is downstream-readable metadata.
    ///
    /// The discriminating mutation: drop the `seen.insert` guard and push every filter match. A is then
    /// pushed twice and the summary reports 2 instead of 1.
    #[tokio::test]
    async fn test_delete_from_row_filter_bypath_and_filter_same_file_counted_once() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            // A strictly matches `x == 0` and is also named by path below, so both removal paths resolve
            // it.
            data_file("test/a.parquet", 0),
            // A keep-anchor in another partition, so the commit is non-empty.
            data_file("test/keep.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_file("test/a.parquet")
            .delete_from_row_filter(Reference::new("x").equal_to(Datum::long(0)));
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("A removed by both path and filter; the keep-anchor survives");

        // The live set is correct whether or not the dedup runs.
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/keep.parquet".to_string()]),
            "A is removed (by path and filter); the x=1 anchor survives"
        );

        // The summary must count A exactly once.
        let summary = table.metadata().current_snapshot().unwrap().summary();
        let props = &summary.additional_properties;
        assert_eq!(
            props.get("deleted-data-files").map(String::as_str),
            Some("1"),
            "a file removed by BOTH path and filter is one deleted data file, not two"
        );
        assert_eq!(
            props.get("deleted-records").map(String::as_str),
            Some("1"),
            "a file removed by BOTH path and filter contributes its records once, not twice"
        );
    }

    // caseSensitive(boolean) on the delete-by-row-filter binding. The default is true.
    //
    // The schema column is `x`. A row filter on the wrong-cased `X` binds ONLY when case sensitivity is
    // off. Three tests pin the flag in both directions on the boundary:
    //   1. Default plus correctly-cased `x` binds and deletes.
    //   2. `case_sensitive(false)` plus wrong-cased `X` binds case-insensitively and deletes.
    //   3. Default plus wrong-cased `X` rejects at bind.
    // Discriminating mutations: ignoring the flag breaks 2; hard-coding false breaks 1 and 3.

    /// With the flag unset, a correctly-cased filter `x == 0` binds and deletes the x=0 file. A default
    /// flip to case-insensitive would still bind here, so this alone is not the boundary. It catches a
    /// regression that breaks the case-sensitive happy path.
    #[tokio::test]
    async fn test_delete_from_row_filter_default_case_sensitive_correct_case_deletes() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/del.parquet", 0),
            data_file("test/keep.parquet", 1),
        ])
        .await;

        // No `case_sensitive` call, so the default is true. The filter is correctly cased.
        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Reference::new("x").equal_to(Datum::long(0)));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.expect(
            "a correctly-cased x==0 filter binds and deletes under the default (case-sensitive)",
        );

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/keep.parquet".to_string()]),
            "x==0 deletes the x=0 file; the x=1 anchor survives"
        );
    }

    /// With `case_sensitive(false)` a wrong-cased filter `X == 0` binds case-insensitively and deletes the
    /// x=0 file. The discriminating mutation: ignore the flag and always bind case-sensitively. `X` then
    /// fails to bind and this test errors.
    #[tokio::test]
    async fn test_delete_from_row_filter_case_insensitive_wrong_case_deletes() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/del.parquet", 0),
            data_file("test/keep.parquet", 1),
        ])
        .await;

        // The wrong-cased `X` binds case-insensitively under `case_sensitive(false)`.
        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .case_sensitive(false)
            .delete_from_row_filter(Reference::new("X").equal_to(Datum::long(0)));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.expect(
            "case_sensitive(false) binds the wrong-cased X to schema column x and deletes the x=0 file",
        );

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/keep.parquet".to_string()]),
            "the wrong-cased X==0 deletes the x=0 file under case-insensitive resolution"
        );
    }

    /// With the default, a wrong-cased filter `X == 0` must reject at bind, so nothing is deleted. A
    /// wrongly-matched bind deletes under a name the user did not spell. The discriminating mutation:
    /// hard-code the flag to false. `X` then binds and this test fails.
    #[tokio::test]
    async fn test_delete_from_row_filter_default_case_sensitive_wrong_case_rejects() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/del.parquet", 0),
            data_file("test/keep.parquet", 1),
        ])
        .await;

        // No `case_sensitive` call, so the default is true. The wrong-cased `X` must fail to bind.
        let tx = Transaction::new(&table);
        let action = tx
            .delete_files()
            .delete_from_row_filter(Reference::new("X").equal_to(Datum::long(0)));
        let tx = action.apply(tx).unwrap();
        let error = tx.commit(&catalog).await.expect_err(
            "a wrong-cased X must NOT bind under the default (case-sensitive) and must reject",
        );

        // The rejected delete removed nothing.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from([
                "test/del.parquet".to_string(),
                "test/keep.parquet".to_string()
            ]),
            "the rejected case-sensitive bind deleted nothing: both files survive"
        );
        // The same wrong-cased column succeeds under `case_sensitive(false)`. See
        // `test_delete_from_row_filter_case_insensitive_wrong_case_deletes`.
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    mod delete_files_extracted;
}
