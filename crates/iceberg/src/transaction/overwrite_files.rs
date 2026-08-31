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

//! This module contains the overwrite-files action.
//!
//! [`OverwriteFilesAction`] adds and removes data files in one snapshot (Java `BaseOverwriteFiles`).
//! Adds reach the producer like fast-append. Deletes filter the current manifests like `DeleteFiles`.
//!
//! **Recorded operation.** The action classifies the snapshot on the REQUESTED sets, before it
//! resolves any delete path (Java `BaseOverwriteFiles.operation()`): delete-only records
//! [`Operation::Delete`], add-only records [`Operation::Append`], both record [`Operation::Overwrite`].
//!
//! **`validateNoConflictingData` (opt-in).** The commit fails if a concurrent snapshot added a DATA
//! file that can hold records matching the conflict-detection filter. It shares
//! [`validate_no_conflicting_added_data_files`] with `RowDelta` so the two checks cannot drift.
//!
//! **`validateNoConflictingDeletes` (opt-in).** One flag, two Java sub-branches. Branch A applies when
//! the row filter is set. It fails on a concurrently added delete file that can apply to matching
//! records, and on a concurrently deleted data file that can hold them. Branch B applies to the data
//! files removed through [`OverwriteFilesAction::delete_data_files`]. It fails when a concurrent
//! commit added a delete file that applies to one of them. You must not drop a data file out from
//! under a concurrent row-level delete. The delete-file sub-checks are a no-op on a V1 table.
//!
//! The two `validateNoConflicting*` flags are INDEPENDENT. One does not enable the other.
//!
//! **Delete-by-row-filter.** [`OverwriteFilesAction::overwrite_by_row_filter`] removes every live data
//! file the predicate STRICTLY matches (Java `deleteByRowFilter`). A file that matches only partially
//! is a non-retryable error. An unpartitioned `alwaysTrue` filter deletes every file.
//!
//! **`validateAddedFilesMatchOverwriteFilter` (opt-in).** Each added file must keep all of its rows
//! inside the row filter: `inclusive_partition && (strict_partition || strict_metrics)`. Failure is a
//! non-retryable `DataInvalid`. It only means something with a row filter set.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::expr::visitors::expression_evaluator::ExpressionEvaluator;
use crate::expr::visitors::inclusive_projection::InclusiveProjection;
use crate::expr::visitors::strict_metrics_evaluator::StrictMetricsEvaluator;
use crate::expr::visitors::strict_projection::StrictProjection;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::spec::{DataFile, MAIN_BRANCH, ManifestEntry, ManifestFile, Operation, Schema};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, FirstRowIdPolicy, SnapshotProduceOperation, SnapshotProducer,
    validate_deleted_data_files_on, validate_no_conflicting_added_data_files_on,
    validate_no_conflicting_added_delete_files_on, validate_no_new_deletes_for_data_files_on,
};
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind};

/// A transaction action that overwrites files: it adds data files AND removes data files in a single
/// `Overwrite` snapshot.
///
/// Use [`crate::transaction::Transaction::overwrite_files`] to create one. Accumulate the files to
/// add with [`OverwriteFilesAction::add_file`] / [`OverwriteFilesAction::add_files`] and the files to
/// remove with [`OverwriteFilesAction::delete_file`] / [`OverwriteFilesAction::delete_files`] /
/// [`OverwriteFilesAction::delete_data_files`], then apply and commit the transaction.
///
/// An overwrite with adds-only (no deletes) or deletes-only (no adds) is allowed; a truly-empty
/// overwrite (no adds, no deletes, no snapshot properties) is rejected.
pub struct OverwriteFilesAction {
    /// Data files to add to the table (validated like fast append).
    added_data_files: Vec<DataFile>,
    /// Fully-qualified file paths to remove from the table.
    delete_paths: HashSet<String>,
    /// The removed data files that carry full metadata (Java `BaseOverwriteFiles.deletedDataFiles`).
    /// `validateNoConflictingDeletes` validates ONLY these. It needs each file's partition and sequence
    /// number, which a bare path does not carry. Their paths are also in `delete_paths`, so the manifest
    /// rewrite removes them like any path-only delete.
    deleted_data_files: Vec<DataFile>,
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    /// Java `validateNoConflictingData`. OFF by default, which gives snapshot isolation. When ON, the
    /// commit fails if a concurrent snapshot added a DATA file that can hold records matching the filter.
    validate_no_conflicting_data: bool,
    /// Java `OverwriteFiles.validateNoConflictingDeletes`. OFF by default, which gives snapshot isolation.
    /// When ON, the commit fails if a concurrent snapshot added a delete file that applies to a data file
    /// this overwrite removes through [`OverwriteFilesAction::delete_data_files`]. INDEPENDENT of
    /// [`Self::validate_no_conflicting_data`]. Java sets the two flags from two separate methods.
    validate_no_conflicting_deletes: bool,
    /// Java `conflictDetectionFilter`. `Some` makes only a concurrently added file whose metrics can match
    /// the predicate a conflict. `None` defaults to `AlwaysTrue`, so any concurrently added data file
    /// conflicts (Java `BaseOverwriteFiles.dataConflictDetectionFilter()` with no filter and no row filter).
    conflict_detection_filter: Option<Predicate>,
    /// An explicit starting snapshot for conflict validation (Java `validateFromSnapshot`). `None` uses the
    /// transaction's starting snapshot, the table head when the transaction was created.
    validate_from_snapshot: Option<i64>,
    /// The delete-by-row-filter predicate (Java `deleteExpression`). `Some` removes every live data file the
    /// predicate STRICTLY matches, through [`SnapshotProducer::resolve_filter_deletes`]. `None` means Java
    /// `alwaysFalse()`, which the conflict-filter default and the added-file validation both read.
    row_filter: Option<Predicate>,
    /// Java `OverwriteFiles.validateAddedFilesMatchOverwriteFilter`. OFF by default. It asserts that every
    /// added data file lies inside `row_filter`, so it only means something with [`Self::row_filter`] set.
    validate_added_files_match_overwrite_filter: bool,
    /// Case sensitivity for binding this action's predicates (Java `MergingSnapshotProducer.caseSensitive`).
    /// Defaults to `true`, the Java default. `false` switches EVERY filter binding this action performs to
    /// case-insensitive column resolution. See [`OverwriteFilesAction::case_sensitive`].
    case_sensitive: bool,
    pub(crate) target_branch: String,
}

impl OverwriteFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            added_data_files: vec![],
            delete_paths: HashSet::default(),
            deleted_data_files: vec![],
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            validate_no_conflicting_data: false,
            validate_no_conflicting_deletes: false,
            conflict_detection_filter: None,
            validate_from_snapshot: None,
            row_filter: None,
            validate_added_files_match_overwrite_filter: false,
            // Java `MergingSnapshotProducer` defaults `caseSensitive` to true.
            case_sensitive: true,
            target_branch: MAIN_BRANCH.to_string(),
        }
    }

    /// Add a single [`DataFile`] to the table (Java `OverwriteFiles.addFile`).
    pub fn add_file(mut self, data_file: DataFile) -> Self {
        self.added_data_files.push(data_file);
        self
    }

    /// Add multiple [`DataFile`]s to the table.
    pub fn add_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Delete a single file by its fully-qualified path.
    ///
    /// To remove a file from the table, this path must equal a path in the table's metadata (mirrors
    /// Java `OverwriteFiles.deleteFile` / `MergingSnapshotProducer.delete`).
    pub fn delete_file(mut self, path: impl Into<String>) -> Self {
        self.delete_paths.insert(path.into());
        self
    }

    /// Delete multiple files by their fully-qualified paths.
    pub fn delete_files(mut self, paths: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.delete_paths.extend(paths.into_iter().map(Into::into));
        self
    }

    /// Delete multiple files supplied as full [`DataFile`]s (Java `OverwriteFiles.deleteFile(DataFile)`).
    ///
    /// Each path joins the delete set that drives the manifest rewrite, like [`Self::delete_file`]. The full
    /// [`DataFile`] also stays in [`Self::deleted_data_files`], because `validateNoConflictingDeletes` needs
    /// the file's partition and metrics to test it against a concurrently added delete.
    pub fn delete_data_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        for file in files {
            self.delete_paths.insert(file.file_path().to_string());
            self.deleted_data_files.push(file);
        }
        self
    }

    /// Delete every live data file the `predicate` STRICTLY matches (Java
    /// `OverwriteFiles.overwriteByRowFilter`). The files drop in the SAME snapshot as any explicit add or
    /// path delete. A file the predicate matches only partially fails the commit, and the error is not
    /// retryable. An unpartitioned `Predicate::AlwaysTrue` filter deletes every data file. A set row filter
    /// requests a delete, so an add plus a row filter records `Overwrite`.
    pub fn overwrite_by_row_filter(mut self, predicate: Predicate) -> Self {
        self.row_filter = Some(predicate);
        self
    }

    /// Assert that every added data file keeps all of its rows inside the
    /// [`Self::overwrite_by_row_filter`] predicate (Java
    /// `OverwriteFiles.validateAddedFilesMatchOverwriteFilter`). Failure rejects the commit and does not
    /// retry. The check stops a replace-by-predicate from re-adding rows outside the predicate it deleted.
    /// It only means something with a row filter set. Default is no assertion.
    pub fn validate_added_files_match_overwrite_filter(mut self) -> Self {
        self.validate_added_files_match_overwrite_filter = true;
        self
    }

    /// Set whether this action's filters resolve column names case-sensitively (Java
    /// `OverwriteFiles.caseSensitive`). The default is `true`, the Java default. A wrong-cased column
    /// reference then fails to bind and the commit errors. `false` switches EVERY filter this action binds
    /// to case-insensitive resolution: the row-filter delete, the added-file check, and the conflict
    /// validations. Java threads one `caseSensitive` field into all of them. By-path deletes read no column.
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

    /// Enable concurrent-commit conflict validation (Java `OverwriteFiles.validateNoConflictingData`). The
    /// commit fails, and does not retry, if a data file added since the starting snapshot can hold records
    /// matching [`Self::conflict_detection_filter`]. The check stops a silent overwrite of concurrently
    /// appended data. Default is no validation, which gives snapshot isolation.
    pub fn validate_no_conflicting_data(mut self) -> Self {
        self.validate_no_conflicting_data = true;
        self
    }

    /// Enable conflicting-delete validation (Java `OverwriteFiles.validateNoConflictingDeletes`). The commit
    /// fails, and does not retry, if a concurrent snapshot added a delete file that applies to a data file
    /// this overwrite removes. You must not drop a data file out from under a concurrent row-level delete.
    /// It is INDEPENDENT of [`Self::validate_no_conflicting_data`]. Only [`Self::delete_data_files`] entries
    /// are checked, because a bare path lacks the partition and metrics the check needs. It is a no-op on a
    /// V1 table. Default is no validation, which gives snapshot isolation.
    pub fn validate_no_conflicting_deletes(mut self) -> Self {
        self.validate_no_conflicting_deletes = true;
        self
    }

    /// Set the conflict-detection filter (Java `OverwriteFiles.conflictDetectionFilter`). Only a
    /// concurrently added data file whose metrics can hold matching records is a conflict. With no filter
    /// set, the filter is `AlwaysTrue` and any concurrently added data file conflicts. This does not enable
    /// validation on its own. Call [`Self::validate_no_conflicting_data`] for that.
    pub fn conflict_detection_filter(mut self, filter: Predicate) -> Self {
        self.conflict_detection_filter = Some(filter);
        self
    }

    /// Pin the snapshot the conflict validation starts from (Java `OverwriteFiles.validateFromSnapshot`).
    /// The default is the transaction's starting snapshot. This does not enable validation on its own.
    /// Call [`Self::validate_no_conflicting_data`] for that.
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.validate_from_snapshot = Some(snapshot_id);
        self
    }

    /// The row filter the validations read (Java `MergingSnapshotProducer.rowFilter()`). It is the
    /// [`Self::overwrite_by_row_filter`] predicate, or `AlwaysFalse` when unset.
    fn row_filter(&self) -> Predicate {
        self.row_filter.clone().unwrap_or(Predicate::AlwaysFalse)
    }

    /// The conflict-detection filter for `validateNoConflictingData` (Java
    /// `BaseOverwriteFiles.dataConflictDetectionFilter()`):
    /// - the explicit [`Self::conflict_detection_filter`] when set; else
    /// - the row filter when it is set AND [`Self::deleted_data_files`] is empty; else
    /// - `None`, which the shared helper binds as `AlwaysTrue` (Java `alwaysTrue()`).
    fn data_conflict_detection_filter(&self) -> Option<&Predicate> {
        if self.conflict_detection_filter.is_some() {
            return self.conflict_detection_filter.as_ref();
        }
        match &self.row_filter {
            Some(row_filter) if self.deleted_data_files.is_empty() => Some(row_filter),
            // No row filter, or explicit deletes present, means Java `alwaysTrue()`. The shared helper
            // binds `None` as `AlwaysTrue`.
            _ => None,
        }
    }

    /// Assert every added data file lies inside the row filter (Java `BaseOverwriteFiles.validate`). Per
    /// file: `inclusive_partition && (strict_partition || StrictMetricsEvaluator::eval(rowFilter, file))`.
    /// The partition evaluators come from the [`InclusiveProjection`] and [`StrictProjection`] of the row
    /// filter. Failure returns a `DataInvalid` error that does not retry.
    fn check_added_files_match_overwrite_filter(&self, current: &Table) -> Result<()> {
        // With an `alwaysFalse` row filter every non-empty added file fails. That matches Java, which
        // also evaluates the block whenever the flag is on.
        if self.added_data_files.is_empty() {
            return Ok(());
        }

        let row_filter = self.row_filter();
        let schema = current.metadata().current_schema().clone();
        // `rewrite_not` keeps the visitor from seeing a `Not`. This is the ONLY binding of the user's
        // column names, so it uses this action's case sensitivity. The partition projection below binds the
        // PROJECTED predicate to spec-derived field names, which the user does not influence, so it stays
        // case-sensitive like Java's default `Projections` and `Evaluator` overloads.
        let bound_row_filter: BoundPredicate = row_filter
            .clone()
            .rewrite_not()
            .bind(schema.clone(), self.case_sensitive)?;

        // `commit` validates that every added file uses the table default spec.
        let spec_id = current.metadata().default_partition_spec_id();
        let inclusive_partition =
            self.build_partition_evaluator(current, &bound_row_filter, spec_id, true)?;
        let strict_partition =
            self.build_partition_evaluator(current, &bound_row_filter, spec_id, false)?;

        for file in &self.added_data_files {
            // Strict partition or strict metrics proves all rows match. The inclusive partition test
            // skips the metrics work for a file the partition already excludes.
            let all_rows_match = inclusive_partition.eval(file)?
                && (strict_partition.eval(file)?
                    || StrictMetricsEvaluator::eval(&bound_row_filter, file)?);
            if !all_rows_match {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot append file with rows that do not match filter {row_filter}: {}",
                        file.file_path()
                    ),
                ));
            }
        }

        Ok(())
    }

    /// Build the partition [`ExpressionEvaluator`] for `spec_id` from the row filter. `inclusive` projects
    /// through [`InclusiveProjection`], otherwise through [`StrictProjection`] (Java `Projections`). The
    /// result evaluates a [`DataFile`] partition tuple.
    fn build_partition_evaluator(
        &self,
        current: &Table,
        bound_row_filter: &BoundPredicate,
        spec_id: i32,
        inclusive: bool,
    ) -> Result<ExpressionEvaluator> {
        let schema = current.metadata().current_schema();
        let partition_spec = current
            .metadata()
            .partition_spec_by_id(spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Cannot validate added files: unknown partition spec id {spec_id}"),
                )
            })?;

        let partition_type = partition_spec.partition_type(schema)?;
        let partition_schema = Arc::new(
            Schema::builder()
                .with_schema_id(partition_spec.spec_id())
                .with_fields(partition_type.fields().to_owned())
                .build()?,
        );

        let projected = if inclusive {
            InclusiveProjection::new(partition_spec.clone()).project(bound_row_filter)?
        } else {
            StrictProjection::new(partition_spec.clone()).strict_project(bound_row_filter)?
        };

        let partition_filter = projected.rewrite_not().bind(partition_schema, true)?;
        Ok(ExpressionEvaluator::new(partition_filter))
    }
}

#[async_trait]
impl TransactionAction for OverwriteFilesAction {
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
        )
        .with_target_branch(self.target_branch.clone())?;

        // Validate the added files like fast append: content type, spec match, partition values. The
        // producer's commit resolves the delete paths and fails on an absent one (Java
        // `failMissingDeletePaths`).
        snapshot_producer.validate_added_data_files()?;

        snapshot_producer
            .commit(
                OverwriteFilesOperation {
                    delete_paths: self.delete_paths.clone(),
                    // `Some` also resolves the live data files this predicate strictly matches.
                    row_filter: self.row_filter.clone(),
                    // The operation is classified on the REQUESTED sets, before the deletes resolve.
                    adds_data_files: !self.added_data_files.is_empty(),
                    // Case sensitivity for binding the row filter (Java default `true`).
                    case_sensitive: self.case_sensitive,
                },
                DefaultManifestProcess,
            )
            .await
    }

    /// Serializable-isolation conflict validation (Java `BaseOverwriteFiles.validate`). It runs three
    /// opt-in checks against the refreshed base. With none enabled it is a no-op and the transaction keeps
    /// snapshot isolation. All of them share one starting snapshot: [`Self::validate_from_snapshot`] when
    /// set, else the transaction's `starting_snapshot_id`. Any failure rejects the commit with a
    /// `DataInvalid` error, so the retry loop stops and the error reaches the caller.
    ///
    /// # Notes
    ///
    /// | flag | check |
    /// |---|---|
    /// | added-files match filter | [`Self::check_added_files_match_overwrite_filter`] |
    /// | no conflicting data | concurrently added DATA matching the conflict filter |
    /// | no conflicting deletes A | row-filter set: concurrent added deletes / deleted data |
    /// | no conflicting deletes B | [`Self::delete_data_files`] removals; V1 no-op |
    ///
    /// Every check binds with [`Self::case_sensitive`].
    async fn validate(
        self: Arc<Self>,
        starting_snapshot_id: Option<i64>,
        current: &Table,
    ) -> Result<()> {
        // CRITICAL: `starting_snapshot_id` is the base the transaction captured, not the refreshed head.
        // A refreshed head would empty the concurrent set and pass every check silently.
        let effective_start = self.validate_from_snapshot.or(starting_snapshot_id);

        // Runs first, matching Java's block order.
        if self.validate_added_files_match_overwrite_filter {
            self.check_added_files_match_overwrite_filter(current)?;
        }

        // Concurrent-added DATA-file conflict (Java `validateNewDataFiles`).
        if self.validate_no_conflicting_data {
            let conflict_filter = self.data_conflict_detection_filter();
            validate_no_conflicting_added_data_files_on(
                current,
                effective_start,
                conflict_filter,
                self.case_sensitive,
                self.target_branch.as_str(),
            )
            .await?;
        }

        // Java `validateNewDeletes`: two delete-conflict sub-branches under one flag.
        if self.validate_no_conflicting_deletes {
            // Branch A guards an overwrite-by-row-filter against a concurrent merge-on-read delete that it
            // would otherwise invalidate silently. Java's filter is the explicit `conflictDetectionFilter`
            // when set, else the row filter.
            let row_filter = self.row_filter();
            if row_filter != Predicate::AlwaysFalse {
                let filter = self.conflict_detection_filter.clone().unwrap_or(row_filter);
                validate_no_conflicting_added_delete_files_on(
                    current,
                    effective_start,
                    Some(&filter),
                    self.case_sensitive,
                    self.target_branch.as_str(),
                )
                .await?;
                validate_deleted_data_files_on(
                    current,
                    effective_start,
                    Some(&filter),
                    self.case_sensitive,
                    self.target_branch.as_str(),
                )
                .await?;
            }

            // Branch B rejects a concurrently added delete file that applies to a removed data file. Only
            // the full-metadata removals are checked, like Java's `deletedDataFiles`.
            // `ignore_equality_deletes = false`, because Java counts equality deletes here.
            if !self.deleted_data_files.is_empty() {
                // Bind here, not in the shared helper, so the helper signature stays stable across actions
                // (`RewriteFiles` passes `None`). A `None` filter narrows nothing, so every concurrently
                // added delete stays a candidate.
                let bound_conflict_filter = match self.conflict_detection_filter.as_ref() {
                    Some(filter) => Some(filter.clone().bind(
                        current.metadata().current_schema().clone(),
                        self.case_sensitive,
                    )?),
                    None => None,
                };
                validate_no_new_deletes_for_data_files_on(
                    current,
                    effective_start,
                    bound_conflict_filter.as_ref(),
                    &self.deleted_data_files,
                    false,
                    self.target_branch.as_str(),
                )
                .await?;
            }
        }

        Ok(())
    }
}

/// The [`SnapshotProduceOperation`] for [`OverwriteFilesAction`].
///
/// It classifies the operation (Java `BaseOverwriteFiles.operation()`), exposes the current manifests, and
/// resolves the delete paths against the live data entries. The added files reach the producer separately,
/// so one snapshot carries both the added manifest and the rewritten manifests.
struct OverwriteFilesOperation {
    delete_paths: HashSet<String>,
    /// The delete-by-row-filter predicate (Java `deleteExpression`). `Some` unions every strictly matched
    /// live data file with the path-resolved deletes. `None` means Java `alwaysFalse`.
    row_filter: Option<Predicate>,
    /// Whether this overwrite requested any added data files. With the requested delete state it classifies
    /// the operation like Java `BaseOverwriteFiles.operation()`.
    adds_data_files: bool,
    /// Case sensitivity for binding `row_filter` (Java default `true`).
    case_sensitive: bool,
}

impl SnapshotProduceOperation for OverwriteFilesOperation {
    /// Classify the operation on the REQUESTED sets, like Java `BaseOverwriteFiles.operation()`. Delete-only
    /// gives [`Operation::Delete`], add-only gives [`Operation::Append`], both give [`Operation::Overwrite`].
    /// An empty overwrite is rejected earlier, so the both-empty arm never commits.
    fn operation(&self) -> Operation {
        // Java `containsDeletes()`: a set row filter counts as a delete before any file resolves.
        let deletes_data_files = !self.delete_paths.is_empty() || self.row_filter.is_some();
        match (self.adds_data_files, deletes_data_files) {
            (false, true) => Operation::Delete,
            (true, false) => Operation::Append,
            _ => Operation::Overwrite,
        }
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

        // Union the row-filter matches (Java `deleteByRowFilter`). De-dupe by path so a file removed by
        // both a path and the filter counts once. `process_deletes` matches by path and tolerates a
        // duplicate, but the summary counts must stay accurate, and Java's `DataFileSet` dedupes too.
        if let Some(row_filter) = &self.row_filter {
            let filter_deletes = snapshot_produce
                .resolve_filter_deletes(row_filter, self.case_sensitive)
                .await?;
            let mut seen: HashSet<String> = resolved
                .iter()
                .map(|df| df.file_path().to_string())
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
        // Expose every current manifest, DATA and DELETE. `process_deletes` rewrites, carries, or drops
        // each DATA manifest. Every DELETE manifest carries forward unchanged, because its entries are
        // delete-file paths and never appear in `delete_paths`. Dropping one would resurrect deleted rows on
        // a merge-on-read table. The helper documents the conservative dangling-delete posture.
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

    /// Build a data file routed to partition `x = part_value` whose column `y` (schema field id 2, a `long`)
    /// carries `[y_lower, y_upper]` value bounds. The bounds let [`InclusiveMetricsEvaluator`] include or
    /// exclude this file against a conflict-detection filter on `y` — the discriminating input for the
    /// metrics-MATCH vs metrics-EXCLUDE conflict tests. The minimal V3 schema is `x,y,z: long` (ids 1,2,3).
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

    /// Like [`data_file_with_y_bounds`] but with complete `y` stats, including zero null and nan counts.
    /// The strict evaluator returns "might not match" when a column can hold a null or a nan. The zero
    /// counts therefore let the delete-by-row-filter path decide DELETE or KEEP without a partial-match
    /// error. A real Parquet writer produces this shape.
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

    /// Collect the live data file paths in the current snapshot. This is what a scan would read.
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

    /// Append A, B, C, then delete B and add D in one overwrite. The live set must be {A, C, D}, the
    /// operation `Overwrite`, and B's entry `Deleted`. A wrong live set is silent data corruption.
    #[tokio::test]
    async fn test_overwrite_delete_one_add_one_yields_correct_live_scan_set() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Fast-append A, B, C in one commit (one manifest containing all three).
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
            data_file("test/c.parquet", 0),
        ])
        .await;

        // Overwrite: delete B, add D — in one snapshot.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/b.parquet")
            .add_file(data_file("test/d.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // The new snapshot is an Overwrite, and the live scan set is exactly {A, C, D}.
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Overwrite
        );
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from([
                "test/a.parquet".to_string(),
                "test/c.parquet".to_string(),
                "test/d.parquet".to_string(),
            ])
        );

        // B's entry is present as Deleted (the rewritten manifest tombstone).
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut b_deleted = false;
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.file_path() == "test/b.parquet" {
                    assert_eq!(entry.status(), ManifestStatus::Deleted);
                    b_deleted = true;
                }
            }
        }
        assert!(b_deleted, "B must appear as a Deleted tombstone");
    }

    /// A delete-only overwrite succeeds and records `Delete` (Java `BaseOverwriteFiles.operation()`). A
    /// pure delete mislabeled `Overwrite` corrupts the history for consumers that distinguish the two.
    #[tokio::test]
    async fn test_overwrite_delete_only_records_delete_operation() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx.overwrite_files().delete_file("test/b.parquet");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Delete,
            "a delete-only overwrite records Delete (Java BaseOverwriteFiles.operation())"
        );
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/a.parquet".to_string()])
        );
    }

    /// Replace a file with a new one in the SAME partition. A partition-keyed rewrite must not drop the
    /// new file or keep the old one.
    #[tokio::test]
    async fn test_overwrite_replaces_file_in_same_partition() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // old.parquet lives in partition x=0.
        let table = append_files(&catalog, &table, vec![data_file("test/old.parquet", 0)]).await;

        // Replace old.parquet with new.parquet, both in partition x=0.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/old.parquet")
            .add_file(data_file("test/new.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/new.parquet".to_string()])
        );
    }

    /// An overwrite that deletes an absent file errors (Java `failMissingDeletePaths`) and adds nothing.
    /// A dropped delete path would commit a partial overwrite.
    #[tokio::test]
    async fn test_overwrite_delete_absent_file_errors() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/does-not-exist.parquet")
            .add_file(data_file("test/b.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("absent delete file must error");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("Missing required files to delete"),
            "unexpected error message: {}",
            error.message()
        );

        // The table is unchanged — the failed overwrite did not add b.parquet.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from(["test/a.parquet".to_string()])
        );
    }

    /// A mixed present and absent delete set still errors. It must not remove the matched file and skip
    /// the unmatched one.
    #[tokio::test]
    async fn test_overwrite_mixed_present_and_absent_delete_errors() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_files(["test/a.parquet", "test/absent.parquet"]);
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("mixed present+absent delete must error");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(error.message().contains("test/absent.parquet"));
    }

    /// An empty overwrite is rejected. A permissive precondition would produce a no-op snapshot.
    #[tokio::test]
    async fn test_empty_overwrite_is_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx.overwrite_files();
        let tx = action.apply(tx).unwrap();
        let result = tx.commit(&catalog).await;

        assert!(result.is_err(), "a truly-empty overwrite must be rejected");
    }

    /// A rewritten manifest must copy every surviving entry forward as `Existing` with its ORIGINAL
    /// snapshot id and both sequence numbers. The added file takes the new snapshot's provenance. The
    /// `Deleted` tombstone keeps the removed file's sequence numbers but takes the new snapshot id.
    /// A re-stamp corrupts the table, because a wrong data-sequence number breaks merge-on-read delete
    /// application and incremental scans. The other tests assert only the live path set and the
    /// operation, so they survive a re-stamp. Only this test catches it.
    #[tokio::test]
    async fn test_overwrite_preserves_surviving_entry_provenance_across_snapshots() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Append A in its OWN commit (snapshot S1, data seq 1).
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
        let s1 = table.metadata().current_snapshot().unwrap().snapshot_id();

        // Append B and C in ONE commit (snapshot S2, data seq 2; one manifest with both).
        let table = append_files(&catalog, &table, vec![
            data_file("test/b.parquet", 0),
            data_file("test/c.parquet", 0),
        ])
        .await;
        let s2 = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(s1, s2);

        // Capture original provenance before the overwrite.
        let (a_snap, a_seq, a_fseq) = entry_provenance(&table, "test/a.parquet").await;
        let (b_snap, b_seq, b_fseq) = entry_provenance(&table, "test/b.parquet").await;
        assert_eq!(a_snap, Some(s1), "A added by S1");
        assert_eq!(b_snap, Some(s2), "B added by S2");
        assert_ne!(a_seq, b_seq, "A and B must have different data seq numbers");

        // Overwrite: delete B + add D → rewrites S2's manifest (C survives) and adds a new manifest (D).
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/b.parquet")
            .add_file(data_file("test/d.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s3 = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(s3, s2);

        // C survived: rewritten as Existing, MUST keep S2's snapshot id + seq numbers (NOT S3).
        let (c_snap, c_seq, c_fseq) = entry_provenance(&table, "test/c.parquet").await;
        assert_eq!(
            c_snap,
            Some(s2),
            "surviving C must keep its ORIGINAL snapshot id S2, not the overwrite snapshot S3"
        );
        assert_eq!(
            c_seq, b_seq,
            "surviving C must keep its ORIGINAL data seq, not the overwrite seq"
        );
        assert_eq!(
            c_fseq, b_fseq,
            "surviving C must keep its ORIGINAL file seq"
        );

        // A survived in its own (untouched, carried-forward) manifest with S1 provenance intact.
        let (a2_snap, a2_seq, a2_fseq) = entry_provenance(&table, "test/a.parquet").await;
        assert_eq!(a2_snap, Some(s1), "carried-forward A keeps S1");
        assert_eq!(a2_seq, a_seq, "carried-forward A keeps its data seq");
        assert_eq!(a2_fseq, a_fseq, "carried-forward A keeps its file seq");

        // The added file D gets the NEW overwrite snapshot's provenance (S3 + the new seq).
        let (d_snap, d_seq, _d_fseq) = entry_provenance(&table, "test/d.parquet").await;
        assert_eq!(
            d_snap,
            Some(s3),
            "added D gets the new overwrite snapshot id"
        );
        assert_ne!(
            d_seq, b_seq,
            "added D gets the new (higher) data seq, not the deleted file's seq"
        );

        // The DELETED tombstone for B carries the NEW snapshot id S3 but keeps B's original data/file seq.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut b_tombstone = None;
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.status() == ManifestStatus::Deleted
                    && entry.file_path() == "test/b.parquet"
                {
                    b_tombstone = Some((
                        entry.snapshot_id(),
                        entry.sequence_number(),
                        entry.file_sequence_number,
                    ));
                }
            }
        }
        let b_tombstone = b_tombstone.expect("B must have a Deleted tombstone");
        assert_eq!(
            b_tombstone.0,
            Some(s3),
            "the Deleted tombstone for B gets the new snapshot id S3"
        );
        assert_eq!(
            b_tombstone.1, b_seq,
            "the Deleted tombstone keeps B's original data seq"
        );
        assert_eq!(
            b_tombstone.2, b_fseq,
            "the Deleted tombstone keeps B's original file seq"
        );
    }

    /// The summary must carry both the added and the deleted file and record counts (Java
    /// `MergingSnapshotProducer.apply`). An added-only summary under-reports the overwrite and breaks
    /// tooling that reads `deleted-data-files` or `deleted-records`.
    #[tokio::test]
    async fn test_overwrite_summary_reflects_added_and_deleted_counts() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // A and B each carry one record.
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;

        // Overwrite: delete B (1 file, 1 record) + add D (1 file, 1 record).
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/b.parquet")
            .add_file(data_file("test/d.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let summary = table.metadata().current_snapshot().unwrap().summary();
        let props = &summary.additional_properties;
        assert_eq!(
            props.get("added-data-files").map(String::as_str),
            Some("1"),
            "summary must report one added data file"
        );
        assert_eq!(
            props.get("added-records").map(String::as_str),
            Some("1"),
            "summary must report one added record"
        );
        assert_eq!(
            props.get("deleted-data-files").map(String::as_str),
            Some("1"),
            "summary must report one deleted data file (Java overwrite summary)"
        );
        assert_eq!(
            props.get("deleted-records").map(String::as_str),
            Some("1"),
            "summary must report one deleted record (Java overwrite summary)"
        );
    }

    /// Read a usize total from a snapshot summary property, defaulting to 0 when absent.
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

    /// Running totals must accumulate across snapshots (Java `SnapshotProducer.summary(previous)` seeds
    /// each snapshot from the previous branch head). Append two, append two more, then overwrite-delete
    /// one: the totals must run 2, 4, 3, not the per-commit delta.
    ///
    /// A producer that seeds from the not-yet-committed snapshot id sees seed 0. The final snapshot then
    /// computes `0 + 0 - 1` and underflows instead of `4 - 1`. This test discriminates that seed bug,
    /// which affects every snapshot action.
    #[tokio::test]
    async fn test_running_totals_accumulate_across_snapshots() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Snapshot 1: append A, B (2 files, 2 records). Running totals = 2.
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
        ])
        .await;
        assert_eq!(
            total(&table, "total-data-files"),
            2,
            "after appending 2 files, total-data-files = 2"
        );
        assert_eq!(
            total(&table, "total-records"),
            2,
            "after appending 2 files (1 record each), total-records = 2"
        );

        // Snapshot 2: append C, D (2 more). Running totals must ACCUMULATE to 4 (not reset to 2).
        let table = append_files(&catalog, &table, vec![
            data_file("test/c.parquet", 0),
            data_file("test/d.parquet", 0),
        ])
        .await;
        assert_eq!(
            total(&table, "total-data-files"),
            4,
            "totals accumulate from the previous branch head: 2 + 2 = 4, not just this commit's 2"
        );
        assert_eq!(
            total(&table, "total-records"),
            4,
            "records accumulate: 2 + 2 = 4"
        );

        // Snapshot 3: overwrite-delete A (net removal). Running totals must be 4 - 1 = 3. Under the old
        // seed-0 logic this computes 0 - 1 and underflows; under the fix it is the correct cumulative 3.
        let tx = Transaction::new(&table);
        let action = tx.overwrite_files().delete_file("test/a.parquet");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            total(&table, "total-data-files"),
            3,
            "after deleting 1 of 4 files, total-data-files = 4 - 1 = 3 (cumulative, no underflow)"
        );
        assert_eq!(
            total(&table, "total-records"),
            3,
            "after deleting 1 record, total-records = 4 - 1 = 3 (cumulative, no underflow)"
        );
        // The delete-only overwrite is recorded as a Delete (dynamic operation).
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

    // Filter-based conflict validation (Java `validateNoConflictingData`). It enumerates the DATA files
    // added since the starting snapshot and rejects any that can hold records matching the filter.
    //
    // The race: the overwrite is built against head S0. A separate `fast_append` then lands as S1.
    // `do_commit` refreshes to S1 and runs `validate` against it. With the flag on, a matching concurrent
    // file must fail the commit. With the flag off, it must not.

    /// Append the given files in a fast-append commit and return the snapshot id that commit produced, plus
    /// the updated table. Used to capture the starting snapshot id S0 before a concurrent commit.
    async fn append_and_snapshot_id(
        catalog: &impl Catalog,
        table: &Table,
        files: Vec<DataFile>,
    ) -> (Table, i64) {
        let table = append_files(catalog, table, files).await;
        let id = table.metadata().current_snapshot().unwrap().snapshot_id();
        (table, id)
    }

    /// With validation on and no concurrent commit, the overwrite commits. Validation must not block a
    /// race-free commit.
    #[tokio::test]
    async fn test_overwrite_validation_no_concurrent_commit_succeeds() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) =
            append_and_snapshot_id(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // Overwrite delete A + add B with validation enabled — but NO concurrent commit lands.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a race-free overwrite must commit even with validation enabled");

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/b.parquet".to_string()])
        );
    }

    /// A concurrent append whose `y` bounds `[60,70]` overlap the filter `y >= 50` must fail the commit with
    /// a `DataInvalid` error that names the file. Without the check the overwrite drops S1's file, which is
    /// a lost write.
    #[tokio::test]
    async fn test_overwrite_rejects_concurrent_added_file_matching_filter() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) =
            append_and_snapshot_id(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // Overwrite delete A + add B, conflict filter `y >= 50`, validation enabled, pinned to S0.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a file whose y bounds [60,70] overlap `y >= 50` (could match).
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            60,
            70,
        )])
        .await;

        let err = tx
            .commit(&catalog)
            .await
            .expect_err("overwrite must fail: a concurrent file could match the conflict filter");

        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "a conflict is a non-retryable validation failure (DataInvalid), not a commit conflict"
        );
        assert!(
            !err.retryable(),
            "the validation failure must be NON-retryable so the retry loop stops and it propagates"
        );
        assert!(
            err.message().contains("conflicting files"),
            "the error must name the conflict, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/concurrent.parquet"),
            "the error must name the conflicting FILE, got: {}",
            err.message()
        );

        // The catalog head is still S1 (the concurrent append) — the overwrite did NOT commit over it.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        let live = live_file_paths(&reloaded).await;
        assert!(
            live.contains("test/concurrent.parquet"),
            "the concurrently-added file must survive (the conflicting overwrite was rejected)"
        );
        assert!(
            !live.contains("test/b.parquet"),
            "the rejected overwrite's added file must NOT be in the table"
        );
    }

    /// The concurrent file's `y` bounds `[10,20]` lie below the filter `y >= 50`, so the overwrite must
    /// commit. A check that ignores the metrics would reject every concurrent append.
    #[tokio::test]
    async fn test_overwrite_allows_concurrent_added_file_excluded_by_filter() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) =
            append_and_snapshot_id(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a file whose y bounds [10,20] are entirely BELOW `y >= 50` (cannot match).
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            10,
            20,
        )])
        .await;

        // The overwrite must SUCCEED — the concurrent file's metrics exclude the filter.
        let table = tx
            .commit(&catalog)
            .await
            .expect("overwrite must commit: the concurrent file cannot match the conflict filter");

        let live = live_file_paths(&table).await;
        assert!(
            live.contains("test/b.parquet"),
            "the overwrite's added file must be in the table (commit succeeded)"
        );
        // The overwrite re-bases onto S1, so the non-conflicting concurrent file also survives.
        assert!(
            live.contains("test/concurrent.parquet"),
            "the non-conflicting concurrent file survives the re-based overwrite"
        );
        assert!(
            !live.contains("test/a.parquet"),
            "A was deleted by the overwrite"
        );
    }

    /// With validation off, a concurrent append that would match the filter still commits. The check must
    /// stay opt-in, because callers rely on snapshot isolation by default.
    #[tokio::test]
    async fn test_overwrite_without_validation_allows_conflicting_concurrent_append() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // Build an overwrite WITHOUT enabling validation (default = snapshot isolation). A conflict filter is
        // even provided, to prove it is inert without the flag.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            );
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a file whose y bounds [60,70] WOULD match `y >= 50` if validation were on.
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            60,
            70,
        )])
        .await;

        // With validation OFF, the overwrite COMMITS (default behavior unchanged).
        let table = tx.commit(&catalog).await.expect(
            "with validation OFF, a conflicting concurrent append must not block the commit",
        );

        let live = live_file_paths(&table).await;
        assert!(
            live.contains("test/b.parquet"),
            "the overwrite committed (snapshot isolation, no conflict check)"
        );
    }

    /// With no conflict filter set, the filter defaults to `AlwaysTrue` (Java
    /// `dataConflictDetectionFilter()`), so any concurrently added data file conflicts, even one with no
    /// bounds. A `None` filter that meant "no conflict" would let every concurrent append through.
    #[tokio::test]
    async fn test_overwrite_none_filter_treats_any_concurrent_add_as_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) =
            append_and_snapshot_id(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // Overwrite with validation enabled but NO conflict_detection_filter ⇒ AlwaysTrue default.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a plain file with NO bounds — still a conflict under AlwaysTrue.
        let _concurrent = append_files(&catalog, &table, vec![data_file(
            "test/concurrent.parquet",
            0,
        )])
        .await;

        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a None filter defaults to AlwaysTrue: any concurrent add is a conflict");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(err.message().contains("test/concurrent.parquet"));
    }

    /// The `validate_from_snapshot` override changes which commits count as concurrent. Build the overwrite
    /// when the head is already S1 and pin the start to the earlier S0. S1's file then counts as concurrent
    /// and the commit is rejected. An ignored override would miss a conflict the caller asked to guard.
    #[tokio::test]
    async fn test_overwrite_validate_from_snapshot_override_changes_concurrent_window() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // S0: a. Capture S0.
        let (table, s0) =
            append_and_snapshot_id(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
        // S1: a file added BEFORE the transaction is built (so it is part of the base, not "concurrent" by
        // the default tx-captured start).
        let (table, _s1) =
            append_and_snapshot_id(&catalog, &table, vec![data_file("test/s1.parquet", 0)]).await;

        // Build the overwrite when the head is S1. Override the start to the EARLIER S0 so S1 counts as
        // concurrent. None filter ⇒ AlwaysTrue ⇒ S1's added file is a conflict.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        let err = tx.commit(&catalog).await.expect_err(
            "validate_from_snapshot(S0) widens the window to include S1's add ⇒ conflict",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(err.message().contains("test/s1.parquet"));
    }

    /// With the start pinned to S1, the current head, S1's file sits at the boundary and is not concurrent,
    /// so the same overwrite commits. The S0 half above rejects the same file, so the override shifts the
    /// boundary.
    #[tokio::test]
    async fn test_overwrite_validate_from_snapshot_at_head_finds_no_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
        let (table, s1) =
            append_and_snapshot_id(&catalog, &table, vec![data_file("test/s1.parquet", 0)]).await;

        // Override the start to S1 (the current head) — nothing is concurrent.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .validate_from_snapshot(s1)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("with start = current head, nothing is concurrent ⇒ commit succeeds");

        assert!(live_file_paths(&table).await.contains("test/b.parquet"));
    }

    /// The check must work without `validate_from_snapshot`, so the start captured in `Transaction::new`
    /// must survive `do_commit`'s re-base. A start re-read from the refreshed head would empty the
    /// concurrent set and pass silently. Every other enabled test pins `validate_from_snapshot`, so this
    /// test alone discriminates that mutation.
    #[tokio::test]
    async fn test_overwrite_rejects_concurrent_using_tx_captured_starting_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // Build the overwrite with validation enabled but WITHOUT validate_from_snapshot — the start is the
        // tx-captured head (S0).
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1).
        let _concurrent = append_files(&catalog, &table, vec![data_file(
            "test/concurrent.parquet",
            0,
        )])
        .await;

        let err = tx
            .commit(&catalog)
            .await
            .expect_err("conflict must be detected via the tx-captured starting snapshot");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(err.message().contains("test/concurrent.parquet"));
    }

    // Conflicting-DELETE validation (Java `validateNoConflictingDeletes` →
    // `MergingSnapshotProducer.validateNoNewDeletesForDataFiles`). For the data files this overwrite
    // removes through `delete_data_files`, a concurrent delete file that applies to one of them fails the
    // commit. You must not drop a data file out from under a concurrent row-level delete.
    //
    // The race: the overwrite is built against head S0 and removes A. A concurrent `row_delta` then lands a
    // position delete in A's partition. With the flag on the overwrite must fail. A delete in another
    // partition, a delete at or before the start, the flag off, or A removed by path only must all commit.

    /// A synthetic position-delete file in partition `x = part_value`. Manifest-only, not a real file.
    fn position_delete_file(path: &str, part_value: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .build()
            .unwrap()
    }

    /// A synthetic equality-delete file in partition `x = part_value`, on field id 1. Manifest-only.
    fn equality_delete_file(path: &str, part_value: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .equality_ids(Some(vec![1]))
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .build()
            .unwrap()
    }

    /// Add the given delete files in one `row_delta` commit, which records `Operation::Delete`.
    async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let action = tx.row_delta().add_deletes(deletes);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Remove A through `delete_data_files` with the deletes validation on. A concurrent position delete in
    /// A's partition must fail the commit with a `DataInvalid` error naming A (Java "Cannot commit, found
    /// new delete for replaced data file"). Without the check the overwrite discards the concurrent delete.
    #[tokio::test]
    async fn test_overwrite_rejects_concurrent_delete_for_removed_data_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        // Build the overwrite removing A via delete_data_files (full metadata ⇒ validated), validation on,
        // pinned to S0.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_data_files(vec![a])
            .add_file(data_file("test/b.parquet", 0))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a position delete in A's partition (x=0), seq > start.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "overwrite must fail: a concurrent delete applies to the removed data file A",
        );
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "a conflict is a non-retryable validation failure (DataInvalid)"
        );
        assert!(
            !err.retryable(),
            "the validation failure must be NON-retryable so the retry loop stops"
        );
        assert!(
            err.message()
                .contains("found new delete for replaced data file"),
            "the error must match Java's message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/a.parquet"),
            "the error must name the replaced data file, got: {}",
            err.message()
        );

        // The catalog head is still S1 (the concurrent delete) — the overwrite did NOT commit over it.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        let live = live_file_paths(&reloaded).await;
        assert!(
            !live.contains("test/b.parquet"),
            "the rejected overwrite's added file must NOT be in the table"
        );
    }

    /// The concurrent position delete sits in partition x=1, not A's x=0, so it does not apply and the
    /// overwrite commits. A partition-blind check would reject every concurrent delete.
    #[tokio::test]
    async fn test_overwrite_allows_concurrent_delete_in_other_partition() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_data_files(vec![a])
            .add_file(data_file("test/b.parquet", 0))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a position delete in a DIFFERENT partition (x=1) — cannot apply to A (x=0).
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file(
            "test/pos-del-other.parquet",
            1,
        )])
        .await;

        let table = tx
            .commit(&catalog)
            .await
            .expect("overwrite must commit: the concurrent delete is in a different partition");
        let live = live_file_paths(&table).await;
        assert!(
            live.contains("test/b.parquet"),
            "the overwrite's added file must be in the table (commit succeeded)"
        );
        assert!(
            !live.contains("test/a.parquet"),
            "A was removed by the overwrite"
        );
    }

    /// The delete lands before the start, so it belongs to the base. With the start pinned to the current
    /// head the overwrite must commit. A check that ignores the boundary would flag a pre-start delete.
    #[tokio::test]
    async fn test_overwrite_allows_delete_at_or_before_start() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone()]).await;

        // A delete lands in A's partition BEFORE the transaction's validation window (it becomes the head S1).
        let table = add_deletes(&catalog, &table, vec![position_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;
        let s1 = table.metadata().current_snapshot().unwrap().snapshot_id();

        // Build the overwrite pinned to S1 (the current head) — the pre-existing delete is NOT concurrent.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_data_files(vec![a])
            .add_file(data_file("test/b.parquet", 0))
            .validate_from_snapshot(s1)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("with start = current head, the pre-existing delete is not concurrent ⇒ commit succeeds");
        assert!(live_file_paths(&table).await.contains("test/b.parquet"));
    }

    /// OverwriteFiles passes `ignore_equality_deletes = false`, so a concurrent equality delete in A's
    /// partition is a conflict. Java counts any applicable delete here.
    #[tokio::test]
    async fn test_overwrite_rejects_concurrent_equality_delete_for_removed_data_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_data_files(vec![a])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): an EQUALITY delete in A's partition (x=0).
        let _concurrent = add_deletes(&catalog, &table, vec![equality_delete_file(
            "test/eq-del.parquet",
            0,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "overwrite must fail: an equality delete applies to the removed data file A",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message()
                .contains("found new delete for replaced data file"),
            "got: {}",
            err.message()
        );
    }

    /// With the deletes validation off, a concurrent delete that applies to the removed file still commits.
    /// The check must stay opt-in.
    #[tokio::test]
    async fn test_overwrite_without_deletes_validation_allows_conflicting_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone()]).await;

        // Build the overwrite WITHOUT enabling the deletes validation (default = snapshot isolation).
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_data_files(vec![a])
            .add_file(data_file("test/b.parquet", 0));
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a position delete applying to A.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;

        let table = tx.commit(&catalog).await.expect(
            "with the deletes-validation OFF, a conflicting concurrent delete must not block the commit",
        );
        assert!(
            live_file_paths(&table).await.contains("test/b.parquet"),
            "the overwrite committed (snapshot isolation, no conflicting-delete check)"
        );
    }

    /// A path-only removal is not in the validated set, so the overwrite commits even with the deletes
    /// validation on and a concurrent delete applying to A. Java validates only `deletedDataFiles`.
    /// Validating a path-only removal would over-reject beyond Java's contract.
    #[tokio::test]
    async fn test_overwrite_path_only_removal_is_not_validated_for_deletes() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // Remove A by PATH only (not delete_data_files) — so it is not in the validated set.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a position delete in A's partition — would conflict IF A were validated.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;

        let table = tx.commit(&catalog).await.expect(
            "a path-only removal is not in the validated deletedDataFiles set ⇒ no conflict ⇒ commit succeeds",
        );
        assert!(live_file_paths(&table).await.contains("test/b.parquet"));
    }

    /// The deletes check must work without `validate_from_snapshot`, so the start captured in
    /// `Transaction::new` must survive the re-base. A start re-read from the refreshed head would empty the
    /// concurrent set and pass silently. Every other deletes test pins `validate_from_snapshot`, so this
    /// test alone discriminates that mutation.
    #[tokio::test]
    async fn test_overwrite_rejects_concurrent_delete_using_tx_captured_starting_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone()]).await;

        // Build the overwrite with the deletes validation enabled but WITHOUT validate_from_snapshot — the
        // start is the tx-captured head (S0).
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_data_files(vec![a])
            .add_file(data_file("test/b.parquet", 0))
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a position delete applying to A.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;

        let err = tx
            .commit(&catalog)
            .await
            .expect_err("conflict must be detected via the tx-captured starting snapshot");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message()
                .contains("found new delete for replaced data file"),
            "got: {}",
            err.message()
        );
        assert!(err.message().contains("test/a.parquet"));
    }

    /// Only the deletes check is on, so a concurrent data-file append does not fail the commit even though
    /// the data check would flag it under `AlwaysTrue`. The two flags are independent.
    #[tokio::test]
    async fn test_overwrite_deletes_validation_does_not_enable_data_validation() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        // ONLY the deletes check is enabled (no validate_no_conflicting_data).
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_data_files(vec![a])
            .add_file(data_file("test/b.parquet", 0))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a plain data APPEND — a DATA-file conflict under AlwaysTrue, NOT a delete.
        let _concurrent = append_files(&catalog, &table, vec![data_file(
            "test/concurrent.parquet",
            0,
        )])
        .await;

        // The deletes check ignores added data files ⇒ the commit succeeds (the data check is OFF).
        let table = tx.commit(&catalog).await.expect(
            "only the deletes check is on; a concurrent data APPEND is not a conflicting-delete ⇒ commit",
        );
        assert!(live_file_paths(&table).await.contains("test/b.parquet"));
    }

    // Delete-by-row-filter mode (Java `BaseOverwriteFiles.overwriteByRowFilter`). `resolve_filter_deletes`
    // reduces the predicate to its per-partition residual per live data file. It then deletes when strict
    // metrics say all rows match, keeps when inclusive metrics say none match, and errors on a partial
    // match.

    /// `overwrite_by_row_filter(x == 0)` strictly matches the whole x=0 partition, so a and b are deleted
    /// and c in x=1 is kept. The added d lands, leaving {c, d}. A file with NO `x` column bounds must still
    /// be deleted, because the partition value satisfies the predicate.
    #[tokio::test]
    async fn test_overwrite_by_row_filter_deletes_strictly_matching_partition() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 0),
            data_file("test/c.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("x").equal_to(Datum::long(0)))
            .add_file(data_file("test/d.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("overwrite_by_row_filter(x == 0) must delete the x=0 files and add d");

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/c.parquet".to_string(), "test/d.parquet".to_string(),]),
            "x=0 files (a, b) deleted by the row filter; c (x=1) kept; d added"
        );
        // Both an add and a (row-filter) delete ⇒ the operation is Overwrite.
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Overwrite
        );
    }

    /// `overwrite_by_row_filter(AlwaysTrue)` deletes every live data file, so only the added file remains.
    /// The residual stays `alwaysTrue`, which every file strict-matches.
    #[tokio::test]
    async fn test_overwrite_by_row_filter_always_true_replaces_all() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/a.parquet", 0),
            data_file("test/b.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Predicate::AlwaysTrue)
            .add_file(data_file("test/new.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("overwrite_by_row_filter(AlwaysTrue) must replace every file");

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/new.parquet".to_string()]),
            "AlwaysTrue deletes every live file; only the added file remains"
        );
    }

    /// A file whose `y` bounds `[0,10]` straddle `y == 5` matches some rows only. The commit must error with
    /// Java's message, "Cannot delete file where some, but not all, rows match filter". A partial match is
    /// never a silent partial delete.
    #[tokio::test]
    async fn test_overwrite_by_row_filter_partial_match_errors() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // y bounds [0,10] straddle `y == 5` ⇒ some-but-not-all rows match.
        let table = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/straddle.parquet",
            0,
            0,
            10,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").equal_to(Datum::long(5)));
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a partial (some-but-not-all) row match must error");

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable(), "a partial-match delete is non-retryable");
        assert!(
            err.message()
                .contains("Cannot delete file where some, but not all, rows match filter"),
            "must match Java's message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/straddle.parquet"),
            "the error must name the offending file, got: {}",
            err.message()
        );

        // The table is unchanged (the failed overwrite committed nothing).
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from(["test/straddle.parquet".to_string()])
        );
    }

    /// A file whose `y` bounds `[60,70]` lie outside `y == 5` survives the row filter. A non-matching file
    /// is neither deleted nor an error.
    #[tokio::test]
    async fn test_overwrite_by_row_filter_keeps_non_matching_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // y bounds [60,70] are entirely above `y == 5` ⇒ no rows match ⇒ KEEP.
        let table = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/high.parquet",
            0,
            60,
            70,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").equal_to(Datum::long(5)))
            .add_file(data_file("test/added.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a non-matching file is kept; the row filter deletes nothing");

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from([
                "test/high.parquet".to_string(),
                "test/added.parquet".to_string(),
            ]),
            "the non-matching file survives; the added file lands"
        );
    }

    // validateAddedFilesMatchOverwriteFilter (Java `BaseOverwriteFiles.validate`). Every added data file
    // must lie inside the row filter:
    //   inclusive_partition.eval(partition) && (strict_partition.eval(partition) || strictMetrics(rowFilter))

    /// With the added-file validation on and row filter `x == 0`, a file added to partition x=0 has all rows
    /// matching, which strict-partition on identity(x) proves, so the commit succeeds.
    #[tokio::test]
    async fn test_validate_added_files_match_filter_accepts_in_filter_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("x").equal_to(Datum::long(0)))
            .add_file(data_file("test/in-filter.parquet", 0))
            .validate_added_files_match_overwrite_filter();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("an added file whose rows all match the filter must be accepted");

        assert!(
            live_file_paths(&table)
                .await
                .contains("test/in-filter.parquet"),
            "the in-filter added file is present (commit succeeded)"
        );
    }

    /// A file added to partition x=1 under row filter `x == 0` has rows outside the filter. Both
    /// strict-partition and strict-metrics fail, so the commit is rejected with Java's "Cannot append file
    /// with rows that do not match filter" message.
    #[tokio::test]
    async fn test_validate_added_files_match_filter_rejects_out_of_filter_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("x").equal_to(Datum::long(0)))
            // Added file in partition x=1 — its rows are OUTSIDE `x == 0`.
            .add_file(data_file("test/out-of-filter.parquet", 1))
            .validate_added_files_match_overwrite_filter();
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("an added file with rows outside the filter must be rejected");

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable(), "the validation failure is non-retryable");
        assert!(
            err.message()
                .contains("Cannot append file with rows that do not match filter"),
            "must match Java's message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/out-of-filter.parquet"),
            "the error must name the offending file, got: {}",
            err.message()
        );
    }

    // Conflict-filter default with a row filter (Java `dataConflictDetectionFilter()`): with no explicit
    // conflict filter, no explicitly removed data files, and a set row filter, the row filter becomes the
    // conflict filter for `validateNoConflictingData`.

    /// With a row filter `y >= 50`, the data validation on, and no explicit conflict filter or deletes, a
    /// concurrent append with `y` bounds `[60,70]` must conflict. The row filter became the conflict filter.
    #[tokio::test]
    async fn test_row_filter_is_default_conflict_filter_matching_add_conflicts() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // The seed lies outside the row filter, so the filter keeps it and no partial match can appear.
        // The conflict is about the concurrent and added files.
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![data_file_with_y_bounds(
            "test/seed.parquet",
            0,
            0,
            10,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/added.parquet", 0, 80, 90))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT add: y bounds [60,70] MATCH `y >= 50` ⇒ a conflict under the row-filter default.
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            60,
            70,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "a concurrent add matching the row filter conflicts (row filter is the default conflict filter)",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message().contains("test/concurrent.parquet"),
            "the conflict must name the concurrent file, got: {}",
            err.message()
        );
    }

    /// The concurrent file's `y` bounds `[10,20]` lie below the row filter `y >= 50`, so the overwrite
    /// commits. Under an `AlwaysTrue` default this add would conflict, so the successful commit
    /// discriminates the row-filter default.
    #[tokio::test]
    async fn test_row_filter_is_default_conflict_filter_outside_add_does_not_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // The seed lies outside the row filter, so the filter keeps it and no partial match can appear.
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![data_file_with_y_bounds(
            "test/seed.parquet",
            0,
            0,
            10,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/added.parquet", 0, 80, 90))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT add: y bounds [10,20] are entirely BELOW `y >= 50` ⇒ NOT a conflict.
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            10,
            20,
        )])
        .await;

        let table = tx.commit(&catalog).await.expect(
            "a concurrent add OUTSIDE the row filter is not a conflict (proves row filter is the default)",
        );
        assert!(
            live_file_paths(&table).await.contains("test/added.parquet"),
            "the overwrite committed (no conflict under the row-filter default)"
        );
        assert!(
            live_file_paths(&table)
                .await
                .contains("test/concurrent.parquet"),
            "the non-conflicting concurrent add survives the re-based overwrite"
        );
    }

    /// An explicitly removed data file disables the row-filter default (Java's `deletedDataFiles.isEmpty()`
    /// guard). The conflict filter falls back to `AlwaysTrue`, so a concurrent add outside the row filter
    /// still conflicts.
    #[tokio::test]
    async fn test_row_filter_default_disabled_when_explicit_deletes_present() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // The seed lies outside the row filter, so the filter keeps it. The explicit `delete_data_files`
        // below removes it, and that is what disables the row-filter default.
        let seed = data_file_with_y_bounds("test/seed.parquet", 0, 0, 10);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![seed.clone()]).await;

        // The explicit `delete_data_files` disables the row-filter default, so the filter is `AlwaysTrue`.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .delete_data_files(vec![seed])
            .add_file(data_file_with_y_bounds("test/added.parquet", 0, 80, 90))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT add OUTSIDE the row filter (y bounds [10,20] < 50). Under AlwaysTrue it STILL conflicts.
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            10,
            20,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "with explicit deletes present the conflict filter is AlwaysTrue ⇒ even an outside add conflicts",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(err.message().contains("test/concurrent.parquet"));
    }

    /// An explicit `conflict_detection_filter` wins over the row filter. The explicit `y >= 100` excludes a
    /// concurrent add at `[60,70]` that the row filter `y >= 50` would catch, so the commit succeeds.
    #[tokio::test]
    async fn test_explicit_conflict_filter_takes_precedence_over_row_filter() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // The seed lies outside the row filter, so the filter keeps it and no partial match can appear.
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![data_file_with_y_bounds(
            "test/seed.parquet",
            0,
            0,
            10,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(100)),
            )
            .add_file(data_file_with_y_bounds("test/added.parquet", 0, 120, 130))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // The concurrent add matches the row filter but not the explicit filter. Complete stats let the
        // re-based row-filter delete classify it as a full match, so no partial-match noise reaches the
        // conflict-filter precedence this test isolates.
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_stats(
            "test/concurrent.parquet",
            0,
            60,
            70,
        )])
        .await;

        let table = tx.commit(&catalog).await.expect(
            "the explicit conflict filter (y >= 100) excludes the [60,70] add ⇒ commit (explicit wins)",
        );
        assert!(live_file_paths(&table).await.contains("test/added.parquet"));
    }

    // Branch A: row-filter-keyed delete conflicts. Needs the deletes flag and a set row filter.
    // | check | concurrent event |
    // |---|---|
    // | validateNoNewDeleteFiles | added delete file |
    // | validateDeletedDataFiles | deleted data file |
    //
    // The race: the overwrite is built against head S0. A concurrent commit S1 then either adds a delete
    // file matching the predicate or removes a data file whose metrics match it. On re-base `validate` runs
    // against S1 and must reject.

    /// A synthetic position-delete file in partition `x = part_value` whose column `y` carries value bounds.
    /// The bounds let the metrics evaluator inside `validateNoNewDeleteFiles` include or exclude it against
    /// a row filter on `y`. A real position-delete file carries no data-column bounds.
    fn position_delete_file_with_y_bounds(
        path: &str,
        part_value: i64,
        y_lower: i64,
        y_upper: i64,
    ) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
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

    /// Branch A, `validateNoNewDeleteFiles`. A concurrent `row_delta` adds a delete file whose `y` bounds
    /// `[60,70]` match the row filter `y >= 50`. The commit must be rejected with Java's "Found new
    /// conflicting delete files that can apply to records matching" message, naming the file. Without branch
    /// A the concurrent delete is invalidated silently. It discriminates a missing branch, a wrong gate, and
    /// an unrun added-delete walk.
    #[tokio::test]
    async fn test_overwrite_row_filter_rejects_concurrent_added_delete_file_matching_filter() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        // Seed `a` lies outside the row filter, so the filter keeps it. The conflict is about the
        // concurrent delete file, not the base.
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![data_file_with_y_bounds(
            "test/a.parquet",
            0,
            0,
            10,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): ADD a position-delete file whose y bounds [60,70] match `y >= 50`.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file_with_y_bounds(
            "test/concurrent-del.parquet",
            0,
            60,
            70,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "overwrite must fail: a concurrent ADDED delete file matches the row filter (branch A)",
        );
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "a conflict is a non-retryable validation failure (DataInvalid)"
        );
        assert!(
            !err.retryable(),
            "the validation failure must be NON-retryable so the retry loop stops"
        );
        assert!(
            err.message()
                .contains("Found new conflicting delete files that can apply to records matching"),
            "must match Java validateNoNewDeleteFiles message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/concurrent-del.parquet"),
            "the error must name the conflicting delete file, got: {}",
            err.message()
        );

        // The overwrite did NOT commit over the concurrent delete.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            !live_file_paths(&reloaded).await.contains("test/b.parquet"),
            "the rejected overwrite's added file must NOT be in the table"
        );
    }

    /// Branch A, `validateDeletedDataFiles`. A concurrent commit removes a data file whose `y` bounds
    /// `[60,70]` match the row filter. The commit must be rejected with Java's "Found conflicting deleted
    /// files that can contain records matching" message. The concurrent commit adds no delete file, so the
    /// added-delete walk finds nothing and this rejection comes only from `validate_deleted_data_files`.
    /// Without it, a concurrent commit that already deleted these rows is re-overwritten silently.
    #[tokio::test]
    async fn test_overwrite_row_filter_rejects_concurrent_deleted_data_file_matching_filter() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // Seed `a` carries `y` bounds inside the row filter. A concurrent delete leaves a tombstone
        // with those bounds, and that is what `validate_deleted_data_files` flags.
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![
            data_file_with_y_bounds("test/a.parquet", 0, 60, 70),
            // A second file the overwrite can keep, so the base is non-trivial.
            data_file_with_y_bounds("test/keep.parquet", 1, 0, 10),
        ])
        .await;

        // No explicit `delete_data_files`, so branch B stays inert and only branch A runs.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90))
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // The concurrent commit deletes `a`, leaving a tombstone that carries its `y` bounds.
        let concurrent_tx = Transaction::new(&table);
        let concurrent_action = concurrent_tx
            .overwrite_files()
            .delete_file("test/a.parquet");
        let concurrent_tx = concurrent_action.apply(concurrent_tx).unwrap();
        let _concurrent = concurrent_tx.commit(&catalog).await.unwrap();

        let err = tx.commit(&catalog).await.expect_err(
            "overwrite must fail: a concurrent DELETED data file matches the row filter (branch A)",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message()
                .contains("Found conflicting deleted files that can contain records matching"),
            "must match Java validateDeletedDataFiles message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/a.parquet"),
            "the error must name the concurrently-deleted data file, got: {}",
            err.message()
        );

        // The overwrite did NOT commit over the concurrent delete.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            !live_file_paths(&reloaded).await.contains("test/b.parquet"),
            "the rejected overwrite's added file must NOT be in the table"
        );
    }

    /// Branch A must reject without `validate_from_snapshot`, so the start captured in `Transaction::new`
    /// must survive the re-base. Setting `effective_start` to
    /// `current.metadata().current_snapshot_id()` makes exactly this test fail. Every other branch-A test
    /// pins `validate_from_snapshot(s0)` and survives that change.
    #[tokio::test]
    async fn test_overwrite_row_filter_rejects_concurrent_delete_using_tx_captured_starting_snapshot()
     {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        // Seed `a` sits outside the row filter, so the base does not self-conflict.
        let table = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/a.parquet",
            0,
            0,
            10,
        )])
        .await;

        // Build the overwrite WITHOUT validate_from_snapshot — the start is the tx-captured head (S0).
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90))
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): ADD a delete file whose y bounds [60,70] match `y >= 50`.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file_with_y_bounds(
            "test/concurrent-del.parquet",
            0,
            60,
            70,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "branch A conflict must be detected via the tx-captured starting snapshot (no override)",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message()
                .contains("Found new conflicting delete files that can apply to records matching"),
            "got: {}",
            err.message()
        );
        assert!(err.message().contains("test/concurrent-del.parquet"));
    }

    /// The same concurrent conflict as the positive test, but with the deletes validation off, commits
    /// cleanly. Branch A must stay opt-in, because callers rely on snapshot isolation by default.
    #[tokio::test]
    async fn test_overwrite_row_filter_without_validation_allows_concurrent_added_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/a.parquet",
            0,
            0,
            10,
        )])
        .await;

        // Build the overwrite WITHOUT enabling the deletes validation (default = snapshot isolation).
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90));
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): ADD a delete file whose y bounds [60,70] WOULD match if branch A ran.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file_with_y_bounds(
            "test/concurrent-del.parquet",
            0,
            60,
            70,
        )])
        .await;

        // With the deletes validation OFF, the overwrite COMMITS (branch A did not run).
        let table = tx.commit(&catalog).await.expect(
            "with validation OFF, a conflicting concurrent delete must not block the commit",
        );
        assert!(
            live_file_paths(&table).await.contains("test/b.parquet"),
            "the overwrite committed (snapshot isolation, branch A not run)"
        );
    }

    /// Branch A is gated on `rowFilter() != alwaysFalse()`. With the deletes validation on but no row filter
    /// and no `delete_data_files`, branch A does not run and a concurrent added delete file does not reject.
    /// A dropped gate would run the row-filter checks on every deletes-validated overwrite. An `AlwaysFalse`
    /// filter matches nothing, so it would not over-reject here, but branch A is keyed on a set row filter.
    #[tokio::test]
    async fn test_overwrite_no_row_filter_skips_branch_a() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/a.parquet",
            0,
            0,
            10,
        )])
        .await;

        // No row filter gates branch A off. No `delete_data_files` leaves branch B inert.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90))
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // The concurrent delete file would match a `y >= 50` row filter if branch A ran.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file_with_y_bounds(
            "test/concurrent-del.parquet",
            0,
            60,
            70,
        )])
        .await;

        // Branch A does not run (no row filter) ⇒ the overwrite COMMITS despite the concurrent delete file.
        let table = tx.commit(&catalog).await.expect(
            "with no row filter, branch A is gated off ⇒ a concurrent added delete does not reject",
        );
        assert!(
            live_file_paths(&table).await.contains("test/b.parquet"),
            "the add-only overwrite committed (branch A skipped, gate held)"
        );
    }

    /// The branch-A gate reads `rowFilter()`, not the conflict filter. Here a conflict filter is set but no
    /// row filter and no `delete_data_files`. Java skips branch A, so a concurrent added delete file that
    /// matches the conflict filter does not reject. A gate keyed on the conflict filter would over-reject
    /// and diverge from Java. Widening the gate to
    /// `row_filter != Predicate::AlwaysFalse || self.conflict_detection_filter.is_some()` makes exactly this
    /// test fail. The sibling gate test has no conflict filter and survives that change.
    #[tokio::test]
    async fn test_overwrite_conflict_filter_without_row_filter_skips_branch_a() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/a.parquet",
            0,
            0,
            10,
        )])
        .await;

        // A conflict filter is set but no row filter, so the gate keeps branch A off. Branch B is inert too.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90))
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // The concurrent delete file matches the conflict filter. Branch A would reject it if it ran.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file_with_y_bounds(
            "test/concurrent-del.parquet",
            0,
            60,
            70,
        )])
        .await;

        // Branch A does not run (no row filter, despite the conflict filter) ⇒ the overwrite COMMITS.
        let table = tx.commit(&catalog).await.expect(
            "gate is on rowFilter(); with no row filter branch A is skipped even when a conflict filter \
             is set, so a concurrent added delete does not reject",
        );
        assert!(
            live_file_paths(&table).await.contains("test/b.parquet"),
            "the add-only overwrite committed (branch A skipped — gate keyed on the absent row filter)"
        );
    }

    // Merge-on-read delete-manifest carry. `existing_manifest` returns DATA and DELETE manifests, so an
    // overwrite preserves outstanding deletes instead of dropping them table-wide. The fixture writes real
    // parquet and a real position-delete file, then scans, so it proves the resurrection end to end.

    /// Write a real parquet data file with rows `(x, y, z)` into partition `x = part_value`.
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

    /// Write a real position-delete file with the production writer, deleting the given path and position
    /// pairs in partition `x = part_value`.
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

    /// Scan the table and collect the `y` values. This is what a query sees with the deletes applied.
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

    /// Count the DELETE-content manifests in the current snapshot. This signal does not use the read path.
    /// An overwrite carries outstanding delete manifests forward, so the count must not drop to zero.
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

    /// An overwrite on a merge-on-read table must not drop the outstanding delete manifests, which would
    /// resurrect deleted rows table-wide. File X in partition 0 carries a real position delete masking its
    /// row y=20. File Y lives in partition 1. An overwrite that adds G and deletes Y must keep X's delete
    /// applying, so the scan returns exactly {10, 80}.
    ///
    /// Filtering `current_manifests()` to DATA manifests in `existing_manifest` makes this test fail: the
    /// scan returns {10, 20, 80} and the delete-manifest count drops to zero.
    #[tokio::test]
    async fn test_overwrite_files_preserves_outstanding_delete_manifests_no_resurrection() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        // X holds y = [10, 20] in partition 0. Y holds y = [60, 70] in partition 1.
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let x_path = x.file_path().to_string();
        let y = write_data_file(&table, "y.parquet", 1, &[(1, 60, 600), (1, 70, 700)]).await;
        let y_path = y.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x, y]).await;

        // Add a real position delete masking X's row at position 1, which holds y=20.
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

        // Before the overwrite the scan drops y=20 and shows Y's rows.
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 60, 70]),
            "the position delete masks y=20 from X; Y's rows are present"
        );

        // Add G and delete Y. This must not drop X's outstanding delete.
        let g = write_data_file(&table, "g.parquet", 1, &[(1, 80, 800)]).await;
        let tx = Transaction::new(&table);
        let action = tx.overwrite_files().add_file(g).delete_file(&y_path);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Overwrite
        );

        // X's masked y=20 stays absent, Y is gone, and G's y=80 is present.
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 80]),
            "Y replaced by G AND X's masked y=20 stays absent — no resurrection"
        );

        // The delete manifest survived the commit.
        assert_eq!(
            count_delete_manifests(&table).await,
            1,
            "the overwrite_files commit must carry the outstanding delete manifest forward (not drop it)"
        );
    }

    // `case_sensitive` on the overwrite-by-row-filter bind.
    // | flag | filter | result |
    // |---|---|---|
    // | default | `x` | binds and deletes |
    // | false | `X` | binds case-insensitively |
    // | default | `X` | bind error |

    /// With the flag unset, a correctly-cased `overwrite_by_row_filter(x == 0)` binds and deletes the x=0
    /// file. It asserts the post-commit live set.
    #[tokio::test]
    async fn test_overwrite_row_filter_default_case_sensitive_correct_case_deletes() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/del.parquet", 0),
            data_file("test/keep.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("x").equal_to(Datum::long(0)));
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a correctly-cased x==0 overwrite filter binds and deletes under the default");

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/keep.parquet".to_string()]),
            "x==0 overwrite-by-row-filter deletes the x=0 file; the x=1 anchor survives"
        );
    }

    /// With `case_sensitive(false)`, a wrong-cased `overwrite_by_row_filter(X == 0)` binds
    /// case-insensitively and deletes the x=0 file. A missed overwrite leaves stale data. Ignoring the flag
    /// and always binding case-sensitively makes `X` fail to bind and this test error.
    #[tokio::test]
    async fn test_overwrite_row_filter_case_insensitive_wrong_case_deletes() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/del.parquet", 0),
            data_file("test/keep.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .case_sensitive(false)
            .overwrite_by_row_filter(Reference::new("X").equal_to(Datum::long(0)));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.expect(
            "case_sensitive(false) binds the wrong-cased X to schema column x and deletes the x=0 file",
        );

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/keep.parquet".to_string()]),
            "the wrong-cased X==0 overwrite deletes the x=0 file under case-insensitive resolution"
        );
    }

    /// With the default, a wrong-cased `overwrite_by_row_filter(X == 0)` must reject at bind and leave the
    /// table untouched. This is the other direction of the pin above. Hard-coding the bind to
    /// case-insensitive makes `X` bind and this test fail.
    #[tokio::test]
    async fn test_overwrite_row_filter_default_case_sensitive_wrong_case_rejects() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            data_file("test/del.parquet", 0),
            data_file("test/keep.parquet", 1),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("X").equal_to(Datum::long(0)));
        let tx = action.apply(tx).unwrap();
        let error = tx.commit(&catalog).await.expect_err(
            "a wrong-cased X must NOT bind under the default (case-sensitive) and must reject",
        );

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from([
                "test/del.parquet".to_string(),
                "test/keep.parquet".to_string()
            ]),
            "the rejected case-sensitive bind deleted nothing: both files survive"
        );
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    // Branch B `case_sensitive` bind. The row-filter tests do not reach this helper.
    // | flag | `Y` | message |
    // |---|---|---|
    // | false | binds | found new delete for replaced data file |
    // | default | bind error | Field Y not found |

    /// `case_sensitive(false)` with a wrong-cased `conflict_detection_filter(Y >= 50)`, removing A through
    /// `delete_data_files`, while a concurrent position delete applies to A. The filter binds
    /// case-insensitively and the removed-data-file conflict rejects. A dropped flag would bind
    /// case-sensitively and surface a bind error instead of the conflict.
    #[tokio::test]
    async fn test_overwrite_branch_b_case_insensitive_wrong_case_detects_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        // Remove A with a wrong-cased conflict filter. `case_sensitive(false)` binds it.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_data_files(vec![a])
            .add_file(data_file("test/b.parquet", 0))
            .case_sensitive(false)
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // The concurrent position delete lands in A's partition after the start, so it applies to A.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "case_sensitive(false) binds the wrong-cased Y; the removed-data-file conflict must fire",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("found new delete for replaced data file"),
            "the wrong-cased Y must bind case-insensitively and the removed-data-file conflict (not a bind \
             error) must fire, got: {}",
            err.message()
        );
    }

    /// With the default and the same wrong-cased `conflict_detection_filter(Y >= 50)`, the branch-B bind
    /// fails and the commit errors with "Field Y not found", not the conflict. Hard-coding the bind to
    /// case-insensitive makes `Y` bind and this test see the conflict message instead.
    #[tokio::test]
    async fn test_overwrite_branch_b_default_case_sensitive_wrong_case_fails_to_bind() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        // The default is case-sensitive, so the wrong-cased `Y` must fail the branch-B bind.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_data_files(vec![a])
            .add_file(data_file("test/b.parquet", 0))
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "a wrong-cased Y must NOT bind under the default in Branch B; the validate must error",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Field Y not found"),
            "the rejection must be a BIND failure on the wrong-cased Y, not the conflict, got: {}",
            err.message()
        );
    }

    // Added-file strict-metrics `case_sensitive` bind. Reached when the partition is not enough.
    // | flag | `X` | message |
    // |---|---|---|
    // | false | binds | Cannot append file with rows that do not match filter |
    // | default | bind error | Field X not found |

    /// `case_sensitive(false)` with a wrong-cased `overwrite_by_row_filter(X == 0)`, the added-file
    /// validation on, and a file added to partition x=1. The row filter binds case-insensitively, the
    /// strict-metrics fallback fires, and the commit rejects with "Cannot append file with rows that do not
    /// match filter". A dropped flag would bind case-sensitively and surface a bind error instead.
    #[tokio::test]
    async fn test_overwrite_added_files_match_filter_case_insensitive_wrong_case_rejects_out_of_filter()
     {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .case_sensitive(false)
            .overwrite_by_row_filter(Reference::new("X").equal_to(Datum::long(0)))
            // Added file in partition x=1 — its rows are OUTSIDE `X == 0`.
            .add_file(data_file("test/out-of-filter.parquet", 1))
            .validate_added_files_match_overwrite_filter();
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "case_sensitive(false) binds the wrong-cased X; the out-of-filter added file must be rejected",
        );

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Cannot append file with rows that do not match filter"),
            "the wrong-cased X must bind case-insensitively and the strict-match check (not a bind error) \
             must fire, got: {}",
            err.message()
        );
    }

    /// With the default and the same wrong-cased row filter, the strict-metrics bind fails and the commit
    /// errors with "Field X not found", not the strict-match rejection. Hard-coding the bind to
    /// case-insensitive makes `X` bind and this test see the other message.
    #[tokio::test]
    async fn test_overwrite_added_files_match_filter_default_case_sensitive_wrong_case_fails_to_bind()
     {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // The default is case-sensitive, so the wrong-cased `X` must fail the strict-metrics bind.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("X").equal_to(Datum::long(0)))
            .add_file(data_file("test/out-of-filter.parquet", 1))
            .validate_added_files_match_overwrite_filter();
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "a wrong-cased X must NOT bind under the default; the strict-match validate must error",
        );

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Field X not found"),
            "the rejection must be a BIND failure on the wrong-cased X, not the strict-match check, got: {}",
            err.message()
        );
    }

    // Data-conflict `case_sensitive` bind. Existing tests use a correctly-cased `y`.
    // | flag | `Y` | message |
    // |---|---|---|
    // | false | binds | Found conflicting files that can contain records matching |
    // | default | bind error | Field Y not found |

    /// `case_sensitive(false)` with a wrong-cased `conflict_detection_filter(Y >= 50)` and the data
    /// validation on, while a concurrent append lands a file whose `y` bounds overlap the filter. The filter
    /// binds case-insensitively and the conflict rejects. A dropped flag would bind case-sensitively and
    /// surface a bind error instead.
    #[tokio::test]
    async fn test_overwrite_data_conflict_case_insensitive_wrong_case_detects_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) =
            append_and_snapshot_id(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // Delete A and add B with a wrong-cased conflict filter. `case_sensitive(false)` binds it.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .case_sensitive(false)
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // CONCURRENT commit (S1): a file whose y bounds [60,70] overlap `y >= 50` (could match).
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            60,
            70,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "case_sensitive(false) binds the wrong-cased Y; the data-file conflict must fire",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("conflicting files"),
            "the wrong-cased Y must bind case-insensitively and the data-file conflict (not a bind error) \
             must fire, got: {}",
            err.message()
        );
    }

    /// With the default and the same wrong-cased conflict filter, the bind fails and the commit errors with
    /// "Field Y not found", not a conflict. The concurrent file's `y` bounds `[10,20]` do not match
    /// `y >= 50`, so a case-insensitive bind would find no conflict and commit. The failed bind is the only
    /// thing that errors here, so hard-coding the bind to case-insensitive makes this test fail.
    #[tokio::test]
    async fn test_overwrite_data_conflict_default_case_sensitive_wrong_case_fails_to_bind() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) =
            append_and_snapshot_id(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        // The default is case-sensitive, so the wrong-cased `Y` must fail the conflict-filter bind.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("test/a.parquet")
            .add_file(data_file("test/b.parquet", 0))
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data();
        let tx = action.apply(tx).unwrap();

        // The concurrent file's `y` bounds do not match `y >= 50`, so a case-insensitive bind would commit.
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            10,
            20,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "a wrong-cased Y must NOT bind under the default (case-sensitive); the validate must error",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Field Y not found"),
            "the rejection must be a BIND failure on the wrong-cased Y (not a conflict — the concurrent file \
             does not match), got: {}",
            err.message()
        );
    }

    // `case_sensitive` on the branch-A delete-conflict binds. With a row filter set, both
    // `validate_no_conflicting_added_delete_files` and `validate_deleted_data_files` bind the conflict filter
    // with the action's case-sensitivity. The other branch-A tests use a correctly-cased `y`, so hard-coding
    // either bind to case-sensitive fails none of them.
    //
    // Each helper binds only when its walk finds a candidate, so each bind needs its own test.
    // | helper | concurrent event |
    // |---|---|
    // | `validate_no_conflicting_added_delete_files` | added delete |
    // | `validate_deleted_data_files` | deleted data |

    /// `case_sensitive(false)` with a row filter, a wrong-cased `conflict_detection_filter(Y >= 50)`, and the
    /// deletes validation on, while a concurrent `row_delta` adds a matching delete file. The filter binds
    /// case-insensitively and the added-delete conflict rejects. A dropped flag would bind case-sensitively
    /// and surface a bind error instead.
    #[tokio::test]
    async fn test_overwrite_branch_a_added_delete_file_case_insensitive_wrong_case_detects_conflict()
     {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        // Seed `a` lies outside the row filter, so the filter keeps it. The conflict is about the
        // concurrent delete file, not the base.
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![data_file_with_y_bounds(
            "test/a.parquet",
            0,
            0,
            10,
        )])
        .await;

        // The row filter gates branch A on. `case_sensitive(false)` binds the wrong-cased conflict filter.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90))
            .case_sensitive(false)
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // The concurrent delete file matches the filter, so the added-delete walk binds the wrong-cased `Y`.
        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file_with_y_bounds(
            "test/concurrent-del.parquet",
            0,
            60,
            70,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "case_sensitive(false) binds the wrong-cased Y in the line-628 check; the added-delete conflict \
             must fire",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Found new conflicting delete files that can apply to records matching"),
            "the wrong-cased Y must bind case-insensitively in the added-delete check and the conflict (not a \
             bind error) must fire, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/concurrent-del.parquet"),
            "the error must name the conflicting delete file, got: {}",
            err.message()
        );
    }

    /// With the default and the same wrong-cased conflict filter, the added-delete bind fails and the commit
    /// errors with "Field Y not found", not the conflict. Hard-coding that bind to case-insensitive makes
    /// `Y` bind and this test see the conflict message.
    #[tokio::test]
    async fn test_overwrite_branch_a_added_delete_file_default_case_sensitive_wrong_case_fails_to_bind()
     {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![data_file_with_y_bounds(
            "test/a.parquet",
            0,
            0,
            10,
        )])
        .await;

        // The default is case-sensitive, so the wrong-cased `Y` must fail the added-delete bind.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90))
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        let _concurrent = add_deletes(&catalog, &table, vec![position_delete_file_with_y_bounds(
            "test/concurrent-del.parquet",
            0,
            60,
            70,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "a wrong-cased Y must NOT bind under the default in the line-628 check; the validate must error",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Field Y not found"),
            "the rejection must be a BIND failure on the wrong-cased Y, not the conflict, got: {}",
            err.message()
        );
    }

    /// `case_sensitive(false)` with a row filter, a wrong-cased `conflict_detection_filter(Y >= 50)`, and the
    /// deletes validation on, while a concurrent commit deletes a matching data file. The added-delete walk
    /// finds nothing and short-circuits, so `validate_deleted_data_files` binds the filter
    /// case-insensitively and rejects. A dropped flag would surface a bind error instead.
    #[tokio::test]
    async fn test_overwrite_branch_a_deleted_data_file_case_insensitive_wrong_case_detects_conflict()
     {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        // Seed `a` carries `y` bounds inside the row filter. A concurrent delete leaves a tombstone
        // with those bounds, and that is what `validate_deleted_data_files` flags.
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![
            data_file_with_y_bounds("test/a.parquet", 0, 60, 70),
            // A second file the overwrite can keep, so the base is non-trivial.
            data_file_with_y_bounds("test/keep.parquet", 1, 0, 10),
        ])
        .await;

        // The row filter gates branch A on. No explicit `delete_data_files` leaves branch B inert.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90))
            .case_sensitive(false)
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        // The concurrent commit deletes `a` and adds no delete file. The added-delete walk therefore
        // short-circuits without binding, and `validate_deleted_data_files` resolves the wrong-cased `Y`.
        let concurrent_tx = Transaction::new(&table);
        let concurrent_action = concurrent_tx
            .overwrite_files()
            .delete_file("test/a.parquet");
        let concurrent_tx = concurrent_action.apply(concurrent_tx).unwrap();
        let _concurrent = concurrent_tx.commit(&catalog).await.unwrap();

        let err = tx.commit(&catalog).await.expect_err(
            "case_sensitive(false) binds the wrong-cased Y in the line-635 check; the deleted-data-file \
             conflict must fire",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Found conflicting deleted files that can contain records matching"),
            "the wrong-cased Y must bind case-insensitively in the deleted-data-file check and the conflict \
             (not a bind error) must fire, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/a.parquet"),
            "the error must name the concurrently-deleted data file, got: {}",
            err.message()
        );
    }

    /// With the default and only a concurrently deleted data file, the `validate_deleted_data_files` bind
    /// fails and the commit errors with "Field Y not found", not the conflict. Hard-coding that bind to
    /// case-insensitive makes `Y` bind and this test see the conflict message.
    #[tokio::test]
    async fn test_overwrite_branch_a_deleted_data_file_default_case_sensitive_wrong_case_fails_to_bind()
     {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![
            data_file_with_y_bounds("test/a.parquet", 0, 60, 70),
            data_file_with_y_bounds("test/keep.parquet", 1, 0, 10),
        ])
        .await;

        // The default is case-sensitive, so the wrong-cased `Y` must fail the branch-A bind.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .overwrite_by_row_filter(Reference::new("y").greater_than_or_equal_to(Datum::long(50)))
            .add_file(data_file_with_y_bounds("test/b.parquet", 0, 80, 90))
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_deletes();
        let tx = action.apply(tx).unwrap();

        let concurrent_tx = Transaction::new(&table);
        let concurrent_action = concurrent_tx
            .overwrite_files()
            .delete_file("test/a.parquet");
        let concurrent_tx = concurrent_action.apply(concurrent_tx).unwrap();
        let _concurrent = concurrent_tx.commit(&catalog).await.unwrap();

        let err = tx.commit(&catalog).await.expect_err(
            "a wrong-cased Y must NOT bind under the default in Branch A; the validate must error",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Field Y not found"),
            "the rejection must be a BIND failure on the wrong-cased Y, not the conflict, got: {}",
            err.message()
        );
    }

    mod overwrite_extracted;
}
