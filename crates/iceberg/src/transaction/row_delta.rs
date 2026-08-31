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

//! Row delta: the merge-on-read write commit (Java `BaseRowDelta`).
//!
//! [`RowDeltaAction`] adds data files and row-level DELETE files in one snapshot. The delete files
//! go into a DELETE manifest beside the DATA manifest, both in the same manifest list. Added delete
//! entries inherit the new snapshot's sequence number, so a delete applies to data written by
//! earlier snapshots (`data_seq <= delete_seq`).
//!
//! The read side ([`crate::arrow::delete_filter`]) applies these deletes during a scan.
//!
//! **Operation** (Java `BaseRowDelta.operation()`): data only gives [`Operation::Append`], deletes
//! only gives [`Operation::Delete`], both give [`Operation::Overwrite`]. The summary always carries
//! the added data-file, delete-file, and position/equality-delete counts.
//!
//! # Conflict validation
//!
//! `RowDeltaAction::validate` runs the checks and documents them one by one. Two flags arm the
//! opt-in ones, and `validateAddedDVs` always runs. With no flag set the action gives snapshot
//! isolation.
//!
//! **Format-version gating** (Java `validateDeleteFileForVersion`, applied in
//! `SnapshotProducer::validate_added_delete_files` against the refreshed base): V1 rejects all
//! deletes, V2 rejects Puffin DVs for position deletes, V3 requires position deletes to be DVs.
//! Equality deletes are exempt at every version.
//!
//! # Removals
//!
//! [`RowDeltaAction::remove_rows`] drops the data file from the table as well as recording it for
//! validation, so it leaves the scan in the same snapshot that adds the new deletes.
//! [`RowDeltaAction::remove_deletes`] drops a superseded delete file from the DELETE manifests.
//! Operation classification ignores both removal sets, so a remove-only row delta is `Overwrite`.
//!
//! Divergence, and Rust is stricter than Java here: a removed path that matches no live data entry
//! fails loud. Java's `BaseRowDelta` does not set `failMissingDeletePaths` for `removeRows`, so a
//! missing path is silently ignored there. One consequence is that a retry whose removal target was
//! concurrently removed fails non-retryably where Java converges.
//!
//! [`RowDeltaAction::validate_fresh_dvs_only`] is a Rust-conservative door with no Java
//! counterpart. It rejects a DV for a data file that already carries a live position-scoped delete,
//! unless this same commit removes that delete.
//!
//! # Out of scope
//!
//! - The equality-delete writer exists, but the RowDelta-with-equality-deletes scan path may have
//!   gaps. The end-to-end test covers position deletes.
//! - Merging a legacy parquet position delete into a new DV. Java `loadPreviousDeletes` unions
//!   whatever covers the data file. This port reads DVs only, so a data file that a parquet
//!   position delete still covers is refused here.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::delete_file_index::is_deletion_vector;
use crate::error::Result;
use crate::expr::visitors::inclusive_metrics_evaluator::InclusiveMetricsEvaluator;
use crate::expr::{Bind, Predicate};
use crate::spec::{DataContentType, DataFile, MAIN_BRANCH, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, FirstRowIdPolicy, SnapshotProduceOperation, SnapshotProducer,
    added_dv_candidate_delete_files_after_on, deleted_data_files_after_on, dv_desc,
    validate_no_conflicting_added_data_files_on, validate_no_conflicting_added_delete_files_on,
    validate_no_new_deletes_for_data_files_on,
};
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind};

/// Adds data files and row-level DELETE files in one snapshot (Java `BaseRowDelta`).
///
/// Create one with [`crate::transaction::Transaction::row_delta`]. Add files with
/// [`RowDeltaAction::add_data_files`] and [`RowDeltaAction::add_deletes`], then commit.
///
/// A deletes-only or a data-only row delta is allowed. A row delta with no data, no deletes, and no
/// snapshot properties is rejected.
pub struct RowDeltaAction {
    /// Validated like fast append, and must be `Data` content.
    added_data_files: Vec<DataFile>,
    /// Position or equality deletes. They go into a DELETE manifest.
    /// `Some(seq)` is an explicit data sequence (Java `addFile(DeleteFile, long)`).
    added_delete_files: Vec<(DataFile, Option<i64>)>,
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    /// Java `RowDelta.validateNoConflictingDataFiles`. Off by default, which gives snapshot isolation.
    validate_no_conflicting_data_files: bool,
    /// Java `RowDelta.validateNoConflictingDeleteFiles`. Off by default. It is independent of
    /// [`Self::validate_no_conflicting_data_files`], because Java sets two separate flags.
    validate_no_conflicting_delete_files: bool,
    /// Java `RowDelta.conflictDetectionFilter`. `None` means `AlwaysTrue`, so any concurrently added
    /// data file conflicts. That matches Java's default and is the most conservative check.
    conflict_detection_filter: Option<Predicate>,
    /// Java `validateFromSnapshot`. `None` uses the transaction's starting snapshot.
    validate_from_snapshot: Option<i64>,
    /// The data-file paths the added position deletes reference (Java `referencedDataFiles`). A
    /// non-empty set arms the files-exist check: the commit is rejected when a concurrent commit
    /// deleted one of these files, because a position delete cannot apply to a file that is gone.
    /// The caller supplies the set. It is not derived from the added delete files.
    referenced_data_files: HashSet<String>,
    /// Java `BaseRowDelta.validateDeletes`, set by `validateDeletedFiles()`. `false` by default, which
    /// makes the files-exist check use the `{Overwrite, Replace}` operation set. `true` widens it to
    /// `{Overwrite, Replace, Delete}`, so a concurrent merge-on-read delete also trips the check.
    validate_deleted_files: bool,
    /// The data files this row delta removes (Java `removedDataFiles`, from `removeRows`). Each path
    /// resolves against the current snapshot's live data entries and is tombstoned in apply. A path
    /// that matches no live entry fails loud.
    ///
    /// The full [`DataFile`] is stored, not a bare path, because the
    /// `validateNoNewDeletesForDataFiles` check needs the partition and the metrics.
    removed_data_files: Vec<DataFile>,
    /// The delete files this row delta removes (Java `BaseRowDelta.removeDeletes`). They reach the
    /// producer through [`SnapshotProducer::with_removed_delete_files`], resolve against the current
    /// snapshot's DELETE manifests by path, and are tombstoned in the rewritten DELETE manifest.
    ///
    /// The use case is merge-on-read superseding, so the table never holds two live DVs for one data
    /// file. Each file must be `PositionDeletes` or `EqualityDeletes` content.
    removed_delete_files: Vec<DataFile>,
    /// Case sensitivity for binding [`Self::conflict_detection_filter`] (Java
    /// `MergingSnapshotProducer.caseSensitive`). `true` by default, as in Java. `false` switches every
    /// filter binding this action's `validate` performs. See [`RowDeltaAction::case_sensitive`].
    case_sensitive: bool,
    pub(crate) target_branch: String,
}

impl RowDeltaAction {
    pub(crate) fn new() -> Self {
        Self {
            added_data_files: vec![],
            added_delete_files: vec![],
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            validate_no_conflicting_data_files: false,
            validate_no_conflicting_delete_files: false,
            conflict_detection_filter: None,
            validate_from_snapshot: None,
            referenced_data_files: HashSet::default(),
            validate_deleted_files: false,
            removed_data_files: vec![],
            removed_delete_files: vec![],
            // Java `MergingSnapshotProducer` defaults this to true.
            case_sensitive: true,
            target_branch: MAIN_BRANCH.to_string(),
        }
    }

    /// Add data files (rows) to the table (Java `RowDelta.addRows`). Each file must be `Data` content.
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Add row-level DELETE files (Java `RowDelta.addDeletes`).
    ///
    /// Java has a separate `DeleteFile` type. Here both kinds are [`DataFile`] and the content type
    /// separates them. Each file must be `PositionDeletes` or `EqualityDeletes` content.
    pub fn add_deletes(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_delete_files
            .extend(delete_files.into_iter().map(|file| (file, None)));
        self
    }

    /// Add a DELETE file with an explicit data sequence number (Java `addFile(DeleteFile, long)`).
    ///
    /// A rewritten sibling DV keeps the original data seq so its applicability window is unchanged.
    pub fn add_delete_file_with_sequence_number(
        mut self,
        delete_file: DataFile,
        sequence_number: i64,
    ) -> Self {
        self.added_delete_files
            .push((delete_file, Some(sequence_number)));
        self
    }

    /// Record DATA files this row delta removes (Java `RowDelta.removeRows(DataFile)`). The files drop
    /// from the table in this same snapshot, and they drive the `validateNoNewDeletesForDataFiles`
    /// check. Repeated calls accumulate.
    ///
    /// Apply side: each path resolves against the current snapshot's live data entries and is
    /// tombstoned by the producer, the same machinery `delete_files` and `overwrite_files` use. The
    /// summary counters follow. A path that matches no live entry fails loud at commit. Pass the full
    /// [`DataFile`], because the conflict check needs the partition and the metrics.
    ///
    /// Validation side, opt-in through [`Self::validate_no_conflicting_delete_files`]: the commit is
    /// rejected when a concurrent commit added a delete that applies to one of these files. A removed
    /// file that this row delta's added deletes also reference (see [`Self::validate_data_files_exist`])
    /// is a self-contradiction and is always rejected.
    pub fn remove_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.removed_data_files.extend(data_files);
        self
    }

    /// Record one DATA file this row delta removes (Java `RowDelta.removeRows`). See
    /// [`Self::remove_data_files`] for the contract.
    pub fn remove_rows(self, data_file: DataFile) -> Self {
        self.remove_data_files([data_file])
    }

    /// Remove one DELETE file from the table (Java `RowDelta.removeDeletes(DeleteFile)`).
    ///
    /// The path resolves against the current snapshot's DELETE manifests and is tombstoned in the
    /// rewritten DELETE manifest. Surviving delete entries keep their provenance.
    ///
    /// The main use is merge-on-read superseding. A merged super-set DV replaces an older DV in the
    /// same commit, so the table never holds two live DVs for one data file. That removal is also the
    /// escape hatch for the fresh-DV door, see [`Self::validate_fresh_dvs_only`]. The file must be
    /// `PositionDeletes` or `EqualityDeletes` content. Repeated calls accumulate.
    pub fn remove_deletes(self, delete_file: DataFile) -> Self {
        self.remove_deletes_many([delete_file])
    }

    /// Remove several DELETE files, the plural of [`Self::remove_deletes`]. Each file must be
    /// `PositionDeletes` or `EqualityDeletes` content. Repeated calls accumulate.
    pub fn remove_deletes_many(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.removed_delete_files.extend(delete_files);
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

    /// Enable DATA-file conflict validation (Java `RowDelta.validateNoConflictingDataFiles`). The commit
    /// is rejected when a concurrently added data file could contain records that match
    /// [`Self::conflict_detection_filter`]. This guards against committing a row delta against data that
    /// another writer appended.
    ///
    /// Default, when you do not call this, is snapshot isolation with no validation.
    pub fn validate_no_conflicting_data_files(mut self) -> Self {
        self.validate_no_conflicting_data_files = true;
        self
    }

    /// Enable DELETE-file conflict validation (Java `RowDelta.validateNoConflictingDeleteFiles`). The
    /// commit is rejected when a concurrently added delete file could apply to records that match
    /// [`Self::conflict_detection_filter`].
    ///
    /// This flag is independent of [`Self::validate_no_conflicting_data_files`]. When both are on, both
    /// checks run and either failure rejects the commit.
    ///
    /// It is a no-op on a V1 table, because delete files start at format version 2.
    ///
    /// Default, when you do not call this, is snapshot isolation with no validation.
    pub fn validate_no_conflicting_delete_files(mut self) -> Self {
        self.validate_no_conflicting_delete_files = true;
        self
    }

    /// Set the conflict-detection filter (Java `RowDelta.conflictDetectionFilter`). Only a
    /// concurrently added file whose metrics could match this predicate is a conflict. The default is
    /// `AlwaysTrue`, as in Java, so any concurrently added data file conflicts.
    ///
    /// This alone does not enable validation. Call [`Self::validate_no_conflicting_data_files`] too.
    pub fn conflict_detection_filter(mut self, filter: Predicate) -> Self {
        self.conflict_detection_filter = Some(filter);
        self
    }

    /// Set how [`Self::conflict_detection_filter`] resolves column names (Java
    /// `RowDelta.caseSensitive(boolean)`).
    ///
    /// The default is `true`, as in Java. A wrong-cased column reference then fails to bind and the
    /// commit errors. `false` switches every filter this action's `validate` binds to case-insensitive
    /// resolution: both `validateNoConflicting*` checks and the `validateAddedDVs` metrics narrowing.
    /// It has no effect without a filter, because the `AlwaysTrue` default binds no column names.
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Override the snapshot from which concurrent-commit conflict validation starts (Java
    /// `RowDelta.validateFromSnapshot(long)`). The default is the transaction's starting snapshot. Use
    /// this to pin the earlier snapshot the caller read when it built the row delta.
    ///
    /// This alone does not enable validation. Call [`Self::validate_no_conflicting_data_files`] too.
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.validate_from_snapshot = Some(snapshot_id);
        self
    }

    /// Give the data files the added position deletes reference, which arms the files-exist check (Java
    /// `RowDelta.validateDataFilesExist`). The commit is rejected when a concurrent commit deleted one of
    /// these files. A position delete cannot apply to a file that is gone, so the commit would lose it.
    ///
    /// The caller supplies the set. It is not derived from the added delete files. A non-empty call is
    /// what arms the check. Repeated calls accumulate.
    ///
    /// The check ignores concurrent DELETE-op snapshots by default. Call
    /// [`Self::validate_deleted_files`] to count those removals as conflicts too.
    pub fn validate_data_files_exist(
        mut self,
        referenced_files: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.referenced_data_files
            .extend(referenced_files.into_iter().map(Into::into));
        self
    }

    /// Include concurrent DELETE-op snapshots in the files-exist check (Java
    /// `RowDelta.validateDeletedFiles()`). The check uses the `{Overwrite, Replace}` operation set by
    /// default, so a concurrent merge-on-read delete of a referenced file is not a conflict. After this
    /// call it uses `{Overwrite, Replace, Delete}`, so it is. `Replace` sits in both sets, so a
    /// concurrent compaction is inspected either way. This flag only toggles `Delete`.
    ///
    /// This alone does not arm the check. Call [`Self::validate_data_files_exist`] too.
    pub fn validate_deleted_files(mut self) -> Self {
        self.validate_deleted_files = true;
        self
    }

    /// Map each data-file path to the added deletion vector that covers it (Java
    /// `MergingSnapshotProducer.newDVRefs`).
    ///
    /// A DV is an added delete file in Puffin format. The spec requires its `referenced_data_file`, so a
    /// Puffin delete file without one is malformed and this errors. Position and equality deletes are
    /// skipped, so a row delta with no DVs gets an empty map. The empty map is what makes
    /// `validate_added_dvs` and the fresh-DV door self-skip.
    ///
    /// # Errors
    ///
    /// Returns `DataInvalid` when an added DV has no referenced data file.
    fn added_dvs_by_referenced_file(&self) -> Result<HashMap<String, &DataFile>> {
        let mut referenced = HashMap::new();
        for (delete_file, _) in &self.added_delete_files {
            if !is_deletion_vector(delete_file) {
                continue;
            }
            match delete_file.referenced_data_file() {
                Some(path) => {
                    referenced.insert(path, delete_file);
                }
                None => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Deletion vector {} is missing its referenced data file",
                            delete_file.file_path()
                        ),
                    ));
                }
            }
        }
        Ok(referenced)
    }

    /// Reject the commit when a concurrent commit since `effective_start` added a deletion vector for a
    /// data file this row delta also adds a DV for (Java `MergingSnapshotProducer.validateAddedDVs`).
    ///
    /// This step always runs, unlike the opt-in checks before it. It self-skips when this row delta adds
    /// no DV. The common merge-on-read row delta adds position or equality deletes, so it is a no-op.
    ///
    /// The concurrent walk is [`added_dv_candidate_delete_files_after`], gated to Java's `{Overwrite,
    /// Delete, Replace}`. `Replace` is in the set because a compaction can rewrite DVs, so this walk is
    /// wider than the one the added-delete-file check uses.
    ///
    /// Of the concurrent deletes, only DVs count. `conflict_filter` narrows them, because a DV whose
    /// metrics cannot match cannot conflict. The first collision on `referenced_data_file` returns
    /// [`ErrorKind::DataInvalid`], which is non-retryable, so the retry loop stops.
    ///
    /// # Errors
    ///
    /// Returns `DataInvalid` on a concurrently added DV for the same data file.
    async fn validate_added_dvs(
        &self,
        current: &Table,
        effective_start: Option<i64>,
        conflict_filter: Option<&Predicate>,
        case_sensitive: bool,
    ) -> Result<()> {
        // Skip when this operation adds no DV.
        let added_dv_referenced = self.added_dvs_by_referenced_file()?;
        if added_dv_referenced.is_empty() {
            return Ok(());
        }

        // The DELETE-manifest walk is gated to Java's `{Overwrite, Delete, Replace}`.
        let added_deletes = added_dv_candidate_delete_files_after_on(
            current,
            effective_start,
            self.target_branch.as_str(),
        )
        .await?;
        if added_deletes.is_empty() {
            return Ok(());
        }

        // A concurrently added DV whose metrics cannot match the filter cannot conflict. Bind once.
        // `None` narrows nothing, which mirrors Java's conservative `alwaysTrue()` default.
        let bound_filter = match conflict_filter {
            Some(filter) => Some(
                filter
                    .clone()
                    .bind(current.metadata().current_schema().clone(), case_sensitive)?,
            ),
            None => None,
        };

        for concurrent in &added_deletes {
            if !is_deletion_vector(concurrent) {
                continue;
            }

            // Metrics narrowing, as Java's `filterRows(conflictDetectionFilter)` does.
            if let Some(bound_filter) = &bound_filter
                && !InclusiveMetricsEvaluator::eval(bound_filter, concurrent, true)?
            {
                continue;
            }

            // A concurrent DV without `referenced_data_file` is malformed. It can collide with nothing,
            // so it is skipped.
            if let Some(referenced) = concurrent.referenced_data_file()
                && added_dv_referenced.contains_key(&referenced)
            {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Found concurrently added DV for {}: {}",
                        referenced,
                        dv_desc(concurrent)
                    ),
                ));
            }
        }

        Ok(())
    }

    /// The fresh-DV door. Rust-conservative, not Java.
    ///
    /// | live delete | why it blocks |
    /// |---|---|
    /// | another DV for the same data file | two live DVs, which the spec forbids |
    /// | a still-applying parquet position delete | the new DV would shadow it and resurrect rows |
    ///
    /// Same-commit `remove_deletes` is the escape hatch.
    ///
    /// "Applies" is the read-path test, evaluated against the referenced data file's LIVE manifest
    /// entry, never against the added DV's own metadata. The DV always carries the current default spec,
    /// so after a partition evolution a referenced file written under an older spec would never match
    /// it. A path-scoped delete applies when it names the same path. A partition-scoped delete applies
    /// when its spec id and partition equal the data entry's. Both need `delete_seq >= data_seq`. A
    /// referenced file with no live entry is added in this commit, so nothing applies to it.
    ///
    /// The escape hatch: Java never commits a second DV, because `BaseDVFileWriter.loadPreviousDeletes`
    /// merges the previous deletes and removes the superseded files through `RowDelta.removeDeletes`.
    /// This port supports that removal ([`RowDeltaAction::remove_deletes`]), so a row delta may add a DV
    /// for a file with a live position-scoped delete when it removes that delete in the same commit. The
    /// table then holds exactly one live DV per file. The writer-side automatic merge stays deferred.
    /// Equality deletes are not superseded by a DV and do not trip the door.
    ///
    /// This runs in `commit()` against the refreshed base. It self-skips when this row delta adds no DV.
    ///
    /// # Errors
    ///
    /// Returns `DataInvalid` when a live delete would be shadowed without being removed.
    async fn validate_fresh_dvs_only(&self, table: &Table) -> Result<()> {
        let added_dvs = self.added_dvs_by_referenced_file()?;
        super::row_delta_fresh_dv::validate_fresh_dvs_only(
            table,
            &added_dvs,
            &self.removed_delete_files,
        )
        .await
    }

    /// Reject a `Data`-content file in the removed-delete set. Java `RowDelta.removeDeletes` takes
    /// delete files only, and a data file must go through `removeRows`. The format-version gate does not
    /// apply to a removal, because the removed file already exists on a versioned table.
    fn validate_removed_delete_files(&self) -> Result<()> {
        for delete_file in &self.removed_delete_files {
            if delete_file.content_type() == DataContentType::Data {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Only position-delete or equality-delete content is allowed for removed delete files (use remove_rows to remove data files)",
                ));
            }
        }
        Ok(())
    }

    /// Reject a row delta that removes a data file its added delete files reference (Java
    /// `BaseRowDelta.validateNoConflictingFileAndPositionDeletes`). The intersection of the removed paths
    /// and the referenced paths must be empty. Such a commit contradicts itself, because the delete would
    /// silently apply to nothing. The message matches Java's.
    fn validate_no_conflicting_file_and_position_deletes(&self) -> Result<()> {
        let deleted_files_with_new_deletes: Vec<&str> = self
            .removed_data_files
            .iter()
            .map(|file| file.file_path())
            .filter(|path| self.referenced_data_files.contains(*path))
            .collect();

        if !deleted_files_with_new_deletes.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot delete data files [{}] that are referenced by new delete files",
                    deleted_files_with_new_deletes.join(", ")
                ),
            ));
        }

        Ok(())
    }
}

#[async_trait]
impl TransactionAction for RowDeltaAction {
    fn target_ref(&self) -> &str {
        self.target_branch.as_str()
    }

    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // Java's `removeDeletes` takes a delete file, so a `Data` file here means the caller wanted
        // `removeRows`. Reject before the producer exists, because the producer guards only added files.
        self.validate_removed_delete_files()?;
        for (_, sequence_number) in &self.added_delete_files {
            if let Some(sequence_number) = sequence_number
                && *sequence_number < 0
            {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot add delete file with negative data sequence number {sequence_number}; \
                         a negative value would be stripped into sequence-number inheritance"
                    ),
                ));
            }
        }

        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
            FirstRowIdPolicy::Suppress,
        )
        .with_removed_delete_files(self.removed_delete_files.clone())
        .with_target_branch(self.target_branch.clone())?;
        let snapshot_producer = if self
            .added_delete_files
            .iter()
            .any(|(_, sequence)| sequence.is_some())
        {
            snapshot_producer.with_added_delete_files_with_seq(self.added_delete_files.clone())
        } else {
            snapshot_producer.with_added_delete_files(
                self.added_delete_files
                    .iter()
                    .map(|(file, _)| file.clone())
                    .collect(),
            )
        };

        // These run against the REFRESHED base, because `do_commit` re-bases first. A concurrent format
        // upgrade therefore re-gates the buffered files, which is where Java applies the version gate.
        snapshot_producer.validate_added_data_files()?;
        snapshot_producer.validate_added_delete_files()?;

        // See `validate_fresh_dvs_only`. A DV for a data file that already has a live position-scoped
        // delete commits only when this same commit removes that delete.
        self.validate_fresh_dvs_only(table).await?;

        snapshot_producer
            .commit(
                RowDeltaOperation {
                    // Classify on the REQUESTED sets, before the files resolve against the table.
                    // Java 1.10.0 `BaseRowDelta.operation()` does the same.
                    adds_data_files: !self.added_data_files.is_empty(),
                    adds_delete_files: !self.added_delete_files.is_empty(),
                    removes_delete_files: !self.removed_delete_files.is_empty(),
                    // Keyed by path, because `delete_files` resolves them against the current snapshot.
                    removed_data_file_paths: self
                        .removed_data_files
                        .iter()
                        .map(|file| file.file_path().to_string())
                        .collect(),
                },
                DefaultManifestProcess,
            )
            .await
    }

    /// Serializable-isolation conflict validation (Java `BaseRowDelta.validate`). With no opt-in check
    /// enabled it is a no-op, which gives snapshot isolation.
    ///
    /// Every check shares one effective starting snapshot ([`Self::validate_from_snapshot`] if set, else
    /// the transaction's) and one conflict filter ([`Self::conflict_detection_filter`] if set, else
    /// `AlwaysTrue`). Any failure rejects the commit with `DataInvalid`, which is non-retryable, so the
    /// retry loop stops. The top-level checks are independent, as they are in Java.
    ///
    /// | # | check |
    /// |---|---|
    /// | 1 | conflicting added DATA matching the filter |
    /// | 2a | new deletes for removed data files (position and equality) |
    /// | 2b | conflicting added DELETE files; V1 no-op |
    /// | 3 | referenced data files still exist |
    /// | 4 | a removed data file is not also referenced by added deletes |
    /// | 5 | no concurrent DV for the same data file |
    ///
    /// Divergences, both conservative over-scans that can only over-reject. Step 2b omits Java's
    /// `startingSequenceNumber` refinement. Step 3 does not thread the conflict filter into
    /// [`deleted_data_files_after`]; the referenced-set intersection is the load-bearing gate there.
    ///
    /// Case sensitivity: every check threads [`Self::case_sensitive`] into the shared helpers, as Java
    /// threads `isCaseSensitive()` into its filter binding.
    ///
    /// # Errors
    ///
    /// Returns `DataInvalid` when any enabled check finds a conflict.
    async fn validate(
        self: Arc<Self>,
        starting_snapshot_id: Option<i64>,
        current: &Table,
    ) -> Result<()> {
        // The `validateFromSnapshot` override wins over the operation's starting snapshot, as in Java.
        let effective_start = self.validate_from_snapshot.or(starting_snapshot_id);
        let conflict_filter = self.conflict_detection_filter.as_ref();

        // 1. Concurrently added DATA-file conflict (Java `validateNewDataFiles`).
        if self.validate_no_conflicting_data_files {
            validate_no_conflicting_added_data_files_on(
                current,
                effective_start,
                conflict_filter,
                self.case_sensitive,
                self.target_branch.as_str(),
            )
            .await?;
        }

        // 2. Concurrently added DELETE-file conflict (Java `validateNewDeleteFiles`). One flag gates both
        //    sub-checks, as it does in Java.
        if self.validate_no_conflicting_delete_files {
            // 2a. `validateNoNewDeletesForDataFiles` on the removed data files. `ignore_equality_deletes`
            //     is false, because Java's non-rewrite path counts position AND equality deletes.
            if !self.removed_data_files.is_empty() {
                // Bind here and pass the bound predicate, which keeps the shared helper's signature
                // stable across actions. `RewriteFiles` passes `None`, which narrows nothing.
                let bound_conflict_filter = match conflict_filter {
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
                    &self.removed_data_files,
                    false,
                    self.target_branch.as_str(),
                )
                .await?;
            }

            // 2b. `validateNoNewDeleteFiles`, the filter-based check. The helper owns the DELETE-manifest
            //     walk and the V2 guard.
            validate_no_conflicting_added_delete_files_on(
                current,
                effective_start,
                conflict_filter,
                self.case_sensitive,
                self.target_branch.as_str(),
            )
            .await?;
        }

        // 3. Referenced-data-files-exist check (Java `validateDataFilesExist`). It runs only when the
        //    caller supplied referenced files. `skip_deletes` mirrors Java's `!validateDeletes`, so the
        //    default excludes concurrent DELETE-op snapshots. `Replace` is in both operation sets.
        if !self.referenced_data_files.is_empty() {
            let skip_deletes = !self.validate_deleted_files;
            let deleted = deleted_data_files_after_on(
                current,
                effective_start,
                skip_deletes,
                self.target_branch.as_str(),
            )
            .await?;
            if let Some(missing) = deleted
                .iter()
                .find(|file| self.referenced_data_files.contains(file.file_path()))
            {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("Cannot commit, missing data files: {}", missing.file_path()),
                ));
            }
        }

        // 4. Removed data files against referenced data files (Java
        //    `validateNoConflictingFileAndPositionDeletes`). Always runs, and self-skips on an empty set.
        self.validate_no_conflicting_file_and_position_deletes()?;

        // 5. Concurrently added deletion-vector conflict (Java `validateAddedDVs`). Always runs, unlike
        //    steps 1 to 3, and self-skips when this row delta adds no DV.
        self.validate_added_dvs(
            current,
            effective_start,
            conflict_filter,
            self.case_sensitive,
        )
        .await?;

        Ok(())
    }
}

/// The [`SnapshotProduceOperation`] for [`RowDeltaAction`].
///
/// One snapshot carries the new DATA manifest, the new DELETE manifest, the rewritten manifests
/// that tombstone the removed data files, and the carried-forward manifests. A row delta that
/// removes no data files takes the empty path, so every manifest carries forward unchanged.
struct RowDeltaOperation {
    /// Whether this row delta requested any added data files (Java `addsDataFiles()`).
    adds_data_files: bool,
    /// Whether this row delta requested any added delete files (Java `addsDeleteFiles()`).
    adds_delete_files: bool,
    /// Whether this row delta requested any removed delete files (Java `deletesDeleteFiles()`).
    /// 1.10.0 `operation()` does not consult this. It stays here so this seam documents the full
    /// request shape, including that a remove-deletes-only commit is not empty.
    removes_delete_files: bool,
    /// The paths of the DATA files this row delta removes (Java `removeRows`). [`Self::delete_files`]
    /// resolves them against the current snapshot's live data entries. An empty set makes
    /// `delete_files` return `[]`.
    removed_data_file_paths: HashSet<String>,
}

impl SnapshotProduceOperation for RowDeltaOperation {
    /// Classify the operation as Java 1.10.0 `BaseRowDelta.operation()` does:
    ///
    /// ```text
    /// if (addsDeleteFiles() && !addsDataFiles()) return DELETE;
    /// return OVERWRITE;
    /// ```
    ///
    /// Deletes only gives [`Operation::Delete`]. Everything else gives [`Operation::Overwrite`].
    ///
    /// Do not add Java MAIN's leading `APPEND` branch. That branch is a post-1.10.0 addition, and the
    /// interop oracle pins 1.10.0, whose bytecode has neither the branch nor a `deletes*` term. An
    /// add-data-only row delta is therefore `Overwrite` here and `Append` on MAIN. Interop cannot see
    /// the difference, because the oracle appends data with `newFastAppend`.
    fn operation(&self) -> Operation {
        // 1.10.0 `operation()` ignores `deletesDeleteFiles()`, so this field is deliberately unused.
        let _ = self.removes_delete_files;
        if self.adds_delete_files && !self.adds_data_files {
            Operation::Delete
        } else {
            Operation::Overwrite
        }
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn delete_files(&self, snapshot_produce: &SnapshotProducer<'_>) -> Result<Vec<DataFile>> {
        // Every requested path must match a live entry, or this fails loud (Java
        // `failMissingDeletePaths`). An empty set returns `[]`, so every manifest carries forward.
        snapshot_produce
            .resolve_delete_paths(&self.removed_data_file_paths)
            .await
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        // A row delta adds manifests without rewriting the existing ones, so carry every data and
        // delete manifest forward unchanged.
        let Some(snapshot) = snapshot_produce.parent_snapshot() else {
            return Ok(vec![]);
        };

        let manifest_list = snapshot
            .load_manifest_list(
                snapshot_produce.table.file_io(),
                &snapshot_produce.table.metadata_ref(),
            )
            .await?;

        Ok(manifest_list.entries().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use futures::TryStreamExt;

    use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
    use crate::delete_file_index::is_deletion_vector;
    use crate::delete_vector::DeleteVector;
    use crate::expr::Reference;
    use crate::memory::tests::new_memory_catalog;
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;
    use crate::scan::FileScanTaskDeleteFile;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal,
        ManifestContentType, ManifestStatus, Operation, Struct,
    };
    use crate::table::Table;
    use crate::transaction::snapshot::{
        DefaultManifestProcess, FirstRowIdPolicy, SnapshotProduceOperation, SnapshotProducer,
    };
    use crate::transaction::tests::{
        make_v2_minimal_table_in_catalog, make_v3_minimal_table_in_catalog,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::position_delete_writer::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    use crate::{Catalog, ErrorKind};

    /// A position-delete file in partition `x = part_value`. Not a real parquet file, so it suits
    /// manifest-only tests.
    fn synthetic_delete_file(path: &str, part_value: i64) -> DataFile {
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

    /// A synthetic deletion vector: a Puffin delete file with the required `referenced_data_file`,
    /// `content_offset`, and `content_size_in_bytes`. Not a real puffin file, so it suits validation
    /// tests only.
    fn synthetic_dv_file(path: &str, part_value: i64, referenced_data_file: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .referenced_data_file(Some(referenced_data_file.to_string()))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .build()
            .unwrap()
    }

    /// A data file in partition `x = part_value`. Not a real parquet file.
    fn synthetic_data_file(path: &str, part_value: i64) -> DataFile {
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

    /// Append the given data files in a single fast-append commit and return the updated table.
    async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let action = tx.fast_append().add_data_files(files);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Read a u64 total from a snapshot summary property, defaulting to 0 when absent.
    fn summary_prop(table: &Table, prop: &str) -> Option<String> {
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties
            .get(prop)
            .cloned()
    }

    /// The end-to-end merge-on-read chain: append a real parquet file, write a real position delete for
    /// positions 1 and 3, commit it with `row_delta`, then scan.
    ///
    /// It discriminates two mutants. A delete the read side never applies leaves all five rows. Mangled
    /// positions drop the wrong rows. No other test proves the write path produces delete files the scan
    /// applies.
    #[tokio::test]
    async fn test_row_delta_position_deletes_drop_deleted_rows_from_scan() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let data_file = write_data_file(&table, "rows.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
            (0, 40, 400),
            (0, 50, 500),
        ])
        .await;
        let data_file_path = data_file.file_path().to_string();
        let table = append_files(&catalog, &table, vec![data_file]).await;

        // Sanity: before any delete, the scan returns all five y values.
        let before: HashSet<i64> = scan_y_values(&table).await;
        assert_eq!(
            before,
            HashSet::from([10, 20, 30, 40, 50]),
            "before the row delta, the scan returns all five rows"
        );

        // Delete positions 1 and 3, which are y=20 and y=40.
        let delete_file = write_position_delete_file(&table, 0, &[
            (data_file_path.clone(), 1),
            (data_file_path.clone(), 3),
        ])
        .await;
        assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![delete_file]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let after: HashSet<i64> = scan_y_values(&table).await;
        assert_eq!(
            after,
            HashSet::from([10, 30, 50]),
            "after the row delta, the scan drops the deleted rows (y=20 and y=40)"
        );
    }

    /// A position delete applies only when `data_seq <= delete_seq`, so it must not reach forward to
    /// data written by a later snapshot.
    ///
    /// D1 (seq 1) gets a delete at positions 1 and 3 (seq 2). D2 (seq 3) then lands in the SAME
    /// partition with live rows at the SAME positions. Only the sequence guard can save D2, so a mutant
    /// that drops the guard wrongly deletes y=70 and y=90.
    #[tokio::test]
    async fn test_row_delta_position_delete_does_not_apply_to_later_data() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let d1 = write_data_file(&table, "d1.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
            (0, 40, 400),
            (0, 50, 500),
        ])
        .await;
        let d1_path = d1.file_path().to_string();
        let table = append_files(&catalog, &table, vec![d1]).await;
        let d1_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        let delete_file =
            write_position_delete_file(&table, 0, &[(d1_path.clone(), 1), (d1_path.clone(), 3)])
                .await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![delete_file]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let delete_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        // D2 has live rows at positions 1 and 3 too.
        let d2 = write_data_file(&table, "d2.parquet", 0, &[
            (0, 60, 600),
            (0, 70, 700),
            (0, 80, 800),
            (0, 90, 900),
            (0, 100, 1000),
        ])
        .await;
        let table = append_files(&catalog, &table, vec![d2]).await;
        let d2_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        assert!(
            d1_seq < delete_seq && delete_seq < d2_seq,
            "expected d1_seq({d1_seq}) < delete_seq({delete_seq}) < d2_seq({d2_seq})"
        );

        let after: HashSet<i64> = scan_y_values(&table).await;
        assert_eq!(
            after,
            HashSet::from([10, 30, 50, 60, 70, 80, 90, 100]),
            "the delete (seq 2) drops only D1's pos 1,3 (y=20,40); D2 (seq 3) is fully intact"
        );
        // Belt-and-suspenders: the rows at the SAME positions in D2 (y=70 at pos 1, y=90 at pos 3) must
        // survive — proving it is the sequence number, not the position, that spares D2.
        assert!(
            after.contains(&70) && after.contains(&90),
            "D2's rows at the deleted positions must survive (the delete's seq does not reach forward)"
        );
    }

    // Manifest, summary, and sequence-number tests. They use synthetic files and do not scan.

    /// A row delta that adds a delete file writes a DELETE manifest into the snapshot's manifest list.
    /// The mutants: the delete file lands in a DATA manifest, which Java cannot read and the read side
    /// never indexes, or no delete manifest is written at all.
    #[tokio::test]
    async fn test_row_delta_writes_delete_manifest_with_deletes_content() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();

        let delete_manifests: Vec<_> = manifest_list
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Deletes)
            .collect();
        assert_eq!(
            delete_manifests.len(),
            1,
            "exactly one DELETE manifest must be written and referenced in the manifest list"
        );

        let delete_manifest = delete_manifests[0]
            .load_manifest(table.file_io())
            .await
            .unwrap();
        let entries: Vec<_> = delete_manifest.entries().iter().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content_type(), DataContentType::PositionDeletes);
        assert_eq!(entries[0].file_path(), "test/a-pos-del.parquet");
        assert_eq!(entries[0].status(), ManifestStatus::Added);
    }

    /// One snapshot carries both a DATA manifest and a DELETE manifest, and records `Overwrite`. The
    /// mutants: only one manifest is written, or the operation is wrong.
    #[tokio::test]
    async fn test_row_delta_add_data_and_deletes_in_one_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_data_files(vec![synthetic_data_file("test/b.parquet", 0)])
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snapshot = table.metadata().current_snapshot().unwrap();
        assert_eq!(
            snapshot.summary().operation,
            Operation::Overwrite,
            "adds-data + adds-deletes records Overwrite (Java BaseRowDelta.operation())"
        );

        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();

        // Key the live paths by manifest content type, so each added file is proven to land in the
        // right manifest and the earlier fast-appended file is proven to survive.
        let mut data_paths = HashSet::new();
        let mut delete_paths = HashSet::new();
        let mut delete_manifest_count = 0;
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            if manifest_file.content == ManifestContentType::Deletes {
                delete_manifest_count += 1;
            }
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                match manifest_file.content {
                    ManifestContentType::Data => {
                        data_paths.insert(entry.file_path().to_string());
                    }
                    ManifestContentType::Deletes => {
                        delete_paths.insert(entry.file_path().to_string());
                    }
                }
            }
        }

        assert!(
            data_paths.contains("test/b.parquet"),
            "the added data file b.parquet lands in a DATA manifest; data paths = {data_paths:?}"
        );
        assert!(
            data_paths.contains("test/a.parquet"),
            "the prior fast-appended data file a.parquet survives"
        );
        assert_eq!(
            delete_manifest_count, 1,
            "exactly one DELETE manifest is written in the row-delta snapshot"
        );
        assert!(
            delete_paths.contains("test/a-pos-del.parquet"),
            "the added delete file lands in the DELETE manifest; delete paths = {delete_paths:?}"
        );
    }

    /// A data-only row delta records `Overwrite`, per Java 1.10.0 `BaseRowDelta.operation()`, which has
    /// no `APPEND` branch. The interop oracle pins 1.10.0, so `Overwrite` is the faithful answer.
    ///
    /// The mutant: re-add Java MAIN's leading `APPEND` branch and this test reads `Append`. With the
    /// deletes-only and remove-only pins it covers both branches of the 1.10.0 form.
    #[tokio::test]
    async fn test_row_delta_add_data_only_records_overwrite_per_1_10_0() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_data_files(vec![synthetic_data_file("test/b.parquet", 0)]);
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
            "an add-data-only row delta records Overwrite per Java 1.10.0 BaseRowDelta.operation() \
             (no APPEND branch in 1.10.0 — MAIN's append arm is post-1.10.0)"
        );
    }

    /// The summary counts one added data file, one added delete file, one added position delete, and
    /// the matching record counts. The mutant: the summary omits the added delete files, so tooling that
    /// reads `added-delete-files` under-reports.
    #[tokio::test]
    async fn test_row_delta_summary_reflects_added_data_and_delete_counts() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/a-pos-del.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(3)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_data_files(vec![synthetic_data_file("test/b.parquet", 0)])
            .add_deletes(vec![delete_file]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            summary_prop(&table, "added-data-files").as_deref(),
            Some("1"),
            "one added data file"
        );
        assert_eq!(
            summary_prop(&table, "added-delete-files").as_deref(),
            Some("1"),
            "one added delete file"
        );
        assert_eq!(
            summary_prop(&table, "added-position-delete-files").as_deref(),
            Some("1"),
            "one added position-delete file"
        );
        assert_eq!(
            summary_prop(&table, "added-position-deletes").as_deref(),
            Some("3"),
            "three added position deletes (the delete file's record count)"
        );
    }

    /// The added delete entry inherits the new snapshot's sequence number, which is strictly greater
    /// than the earlier data file's. That is what makes the delete apply to that data. The mutants: a
    /// stale seq, a zero seq, or the data file's own seq, none of which apply.
    #[tokio::test]
    async fn test_row_delta_added_delete_entry_inherits_new_snapshot_sequence_number() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;
        let data_snapshot = table.metadata().current_snapshot().unwrap();
        let data_seq = data_snapshot.sequence_number();

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let delete_snapshot = table.metadata().current_snapshot().unwrap();
        let delete_seq = delete_snapshot.sequence_number();
        assert!(
            delete_seq > data_seq,
            "the row-delta snapshot's sequence number ({delete_seq}) must exceed the data snapshot's ({data_seq})"
        );

        let manifest_list = delete_snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut found = false;
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != ManifestContentType::Deletes {
                continue;
            }
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.file_path() == "test/a-pos-del.parquet" {
                    assert_eq!(
                        entry.sequence_number(),
                        Some(delete_seq),
                        "the added delete entry inherits the new snapshot's sequence number"
                    );
                    assert_eq!(
                        entry.snapshot_id(),
                        Some(delete_snapshot.snapshot_id()),
                        "the added delete entry carries the new snapshot id"
                    );
                    found = true;
                }
            }
        }
        assert!(
            found,
            "the added delete entry must be present in a DELETE manifest"
        );
    }

    /// `add_deletes` rejects a `Data`-content file. The mutant commits a data file as a delete, so the
    /// table indexes it as a delete and never reads it as data.
    #[tokio::test]
    async fn test_row_delta_rejects_data_content_in_add_deletes() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_data_file("test/not-a-delete.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a Data-content file in add_deletes must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("position-delete or equality-delete"),
            "unexpected error: {}",
            err.message()
        );
    }

    /// A delete file whose partition spec id matches no table spec is rejected with Java's exact
    /// message. The read side could associate such a file with no spec. A delete under a KNOWN
    /// non-default spec is accepted; `snapshot::multispec_tests` covers that.
    #[tokio::test]
    async fn test_row_delta_rejects_unknown_partition_spec() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let bad_delete = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/bad-spec.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            // Unknown partition spec id (the table has only spec 0).
            .partition_spec_id(999)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .unwrap();

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![bad_delete]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("an unknown-spec delete file must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Cannot find partition spec 999 for delete file: test/bad-spec.parquet"),
            "unexpected error: {}",
            err.message()
        );
    }

    /// A row delta with no data, no deletes, and no snapshot properties is rejected. The mutant makes
    /// the precondition permissive, which produces an empty snapshot.
    #[tokio::test]
    async fn test_empty_row_delta_is_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx.row_delta();
        let tx = action.apply(tx).unwrap();
        let result = tx.commit(&catalog).await;
        assert!(result.is_err(), "a truly-empty row delta must be rejected");
    }

    // Helpers that write real parquet data and position-delete files into the table's FileIO.

    /// Write a real parquet data file with the given `(x, y, z)` rows and return the [`DataFile`] for
    /// it. It goes through the table's own `FileIO`, so the scan can read it back.
    async fn write_data_file(
        table: &Table,
        file_name: &str,
        part_value: i64,
        rows: &[(i64, i64, i64)],
    ) -> DataFile {
        use crate::arrow::schema_to_arrow_schema;
        use crate::writer::file_writer::{FileWriter, FileWriterBuilder};

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
        let output = table.file_io().new_output(file_path.clone()).unwrap();
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            schema.clone(),
        );
        let mut writer = parquet_builder.build(output).await.unwrap();
        writer.write(&batch).await.unwrap();
        let data_file_builders = writer.close().await.unwrap();

        // The parquet writer leaves content and partition unstamped, so finish them here.
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

        // The delete-file index keys position deletes by partition and spec id, so the partitions must
        // match.
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

    /// Scan the table and collect the `y` column values across all returned batches.
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
            for i in 0..col.len() {
                values.insert(col.value(i));
            }
        }
        values
    }

    // Filter-based conflict validation (Java `validateNoConflictingDataFiles`).
    //
    // The race these tests simulate: a row delta is built against head S0, then a separate fast append
    // advances the head to S1 before it commits. `do_commit` refreshes to S1 and runs `validate` there.
    // With the check enabled, a concurrent append that could match the filter fails the commit. With the
    // check off, the default, it does not.

    /// A data file in partition `x = part_value` whose column `y`, field id 2, carries the value bounds
    /// `[y_lower, y_upper]`. The bounds are what let the metrics evaluator include or exclude the file
    /// against a filter on `y`.
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

    /// Collect the live DATA file paths in the current snapshot, which is what a scan would read.
    async fn live_data_file_paths(table: &Table) -> HashSet<String> {
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
            if manifest_file.content != ManifestContentType::Data {
                continue;
            }
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

    /// Fast-append the files and return the new snapshot id with the updated table. Use it to capture
    /// the starting snapshot S0 before a concurrent commit.
    async fn append_and_snapshot_id(
        catalog: &impl Catalog,
        table: &Table,
        files: Vec<DataFile>,
    ) -> (Table, i64) {
        let table = append_files(catalog, table, files).await;
        let id = table.metadata().current_snapshot().unwrap().snapshot_id();
        (table, id)
    }

    /// With validation enabled and nothing landing concurrently, the row delta commits. The mutant is a
    /// check that fails a race-free commit.
    #[tokio::test]
    async fn test_row_delta_validation_no_concurrent_commit_succeeds() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a race-free row delta must commit even with validation enabled");

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// A concurrent append lands a file whose `y` bounds `[60,70]` overlap the filter `y >= 50`. The
    /// commit must fail with a non-retryable `DataInvalid` that names the conflicting file.
    ///
    /// The mutant commits the row delta blind to S1's new rows, which loses the merge-on-read result
    /// under serializable isolation.
    #[tokio::test]
    async fn test_row_delta_rejects_concurrent_added_file_matching_filter() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();

        // S1 lands a file whose y bounds [60,70] overlap `y >= 50`.
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
            .expect_err("row delta must fail: a concurrent file could match the conflict filter");

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

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        let live = live_data_file_paths(&reloaded).await;
        assert!(
            live.contains("test/concurrent.parquet"),
            "the concurrently-added file must survive (the conflicting row delta was rejected)"
        );
    }

    /// The concurrent file's `y` bounds `[10,20]` sit entirely below the filter `y >= 50`, so the
    /// evaluator excludes it and the row delta commits. This test fails when the helper's metrics
    /// decision is inverted, or when the check rejects any concurrent append at all.
    #[tokio::test]
    async fn test_row_delta_allows_concurrent_added_file_excluded_by_filter() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();

        // S1 lands a file whose y bounds [10,20] are entirely below `y >= 50`.
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            10,
            20,
        )])
        .await;

        let table = tx
            .commit(&catalog)
            .await
            .expect("row delta must commit: the concurrent file cannot match the conflict filter");

        // The row delta re-bases onto S1, so the concurrent file survives too.
        let live = live_data_file_paths(&table).await;
        assert!(
            live.contains("test/concurrent.parquet"),
            "the non-conflicting concurrent file survives the re-based row delta"
        );
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// Without the flag, a concurrent append that would match the filter does not fail the commit. The
    /// mutant makes the check unconditional, which breaks callers that rely on snapshot isolation.
    #[tokio::test]
    async fn test_row_delta_without_validation_allows_conflicting_concurrent_append() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // The conflict filter is supplied to prove it stays inert without the flag.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            );
        let tx = action.apply(tx).unwrap();

        // S1's y bounds [60,70] would match `y >= 50` if validation were on.
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            60,
            70,
        )])
        .await;

        let table = tx.commit(&catalog).await.expect(
            "with validation OFF, a conflicting concurrent append must not block the commit",
        );

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed (snapshot isolation, no conflict check)"
        );
    }

    /// With no filter set, the default is `AlwaysTrue`, so any concurrently added data file conflicts,
    /// even one with no bounds. The mutant treats a `None` filter as "no conflict", which lets every
    /// concurrent append through.
    #[tokio::test]
    async fn test_row_delta_none_filter_treats_any_concurrent_add_as_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();

        // A file with no bounds still conflicts under `AlwaysTrue`.
        let _concurrent = append_files(&catalog, &table, vec![synthetic_data_file(
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

    /// `validate_from_snapshot` widens the concurrent window. S1 lands before the transaction is built,
    /// so it is part of the base by default. Pinning the start to the earlier S0 makes S1 concurrent, and
    /// the commit is rejected. The mutant ignores the override and misses that conflict.
    #[tokio::test]
    async fn test_row_delta_validate_from_snapshot_override_changes_concurrent_window() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;
        // S1 lands before the transaction, so the default start treats it as base.
        let (table, _s1) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/s1.parquet",
            0,
        )])
        .await;

        // Override the start to S0, so S1 counts as concurrent under the `AlwaysTrue` default.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();

        let err = tx.commit(&catalog).await.expect_err(
            "validate_from_snapshot(S0) widens the window to include S1's add ⇒ conflict",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(err.message().contains("test/s1.parquet"));
    }

    /// The negative half of the override test. Pinning the start to S1, the current head, puts S1's file
    /// on the boundary, so the same row delta commits. The S0 half rejects the very same file, which is
    /// what proves the override moves the boundary.
    #[tokio::test]
    async fn test_row_delta_validate_from_snapshot_at_head_finds_no_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;
        let (table, s1) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/s1.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .validate_from_snapshot(s1)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("with start = current head, nothing is concurrent ⇒ commit succeeds");

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// The starting snapshot captured in `Transaction::new` must survive `do_commit`'s re-base. This test
    /// sets no explicit `validate_from_snapshot`, so only that captured S0 can make S1 concurrent.
    ///
    /// The mutant re-reads the start from the refreshed head, which makes start equal current head, so the
    /// concurrent set is always empty and the check always passes. Every other test here pins the start
    /// explicitly, so this is the only one that discriminates that mutant.
    #[tokio::test]
    async fn test_row_delta_rejects_concurrent_using_tx_captured_starting_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // No `validate_from_snapshot`, so the start is the transaction-captured head S0.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();

        let _concurrent = append_files(&catalog, &table, vec![synthetic_data_file(
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

    // Filter-based DELETE-file conflict validation (Java `validateNoConflictingDeleteFiles`). The walk is
    // V2-only and gated to the `{Overwrite, Delete}` operation set.
    //
    // The race: a row delta is built against head S0, then a separate `row_delta().add_deletes` advances
    // the head to S1 before it commits. With the check enabled, a concurrent delete whose metrics could
    // match the filter fails the commit. With the check off, the default, it does not.

    /// A position-delete file in partition `x = part_value` with the value bounds `[y_lower, y_upper]` on
    /// column `y`, field id 2. The evaluator is content-agnostic, so it reads these bounds on a delete
    /// file just as it does on a data file.
    fn delete_file_with_y_bounds(
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

    /// Commit a concurrent deletes-only row delta through the catalog. Its operation is `Delete`, which
    /// is in the delete walk's `{Overwrite, Delete}` set, so the walk enumerates it.
    async fn commit_concurrent_deletes(
        catalog: &impl Catalog,
        table: &Table,
        delete_files: Vec<DataFile>,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx.row_delta().add_deletes(delete_files);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// A V1 minimal table, the shape of `make_v3_minimal_table_in_catalog` at format version 1, for the
    /// V2-guard tests. The schema is hand-built with no column defaults, because the V3 fixture's
    /// `initial-default` on `x` is V3-only and the schema guard rejects it on V1.
    async fn make_v1_minimal_table_in_catalog(catalog: &impl Catalog) -> Table {
        use crate::spec::{
            NestedField, PartitionSpec, PrimitiveType, Schema, Transform, Type,
            UnboundPartitionField,
        };
        use crate::{TableCreation, TableIdent};

        let table_ident =
            TableIdent::from_strs([format!("ns1-{}", uuid::Uuid::new_v4()), "test1".to_string()])
                .unwrap();
        catalog
            .create_namespace(table_ident.namespace(), HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(3, "z", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_unbound_field(
                UnboundPartitionField::builder()
                    .source_id(1)
                    .name("x".to_string())
                    .transform(Transform::Identity)
                    .build(),
            )
            .unwrap()
            .build()
            .unwrap();

        let table_creation = TableCreation::builder()
            .schema(schema)
            .partition_spec(partition_spec)
            .name(table_ident.name().to_string())
            .format_version(crate::spec::FormatVersion::V1)
            .build();

        catalog
            .create_table(table_ident.namespace(), table_creation)
            .await
            .unwrap()
    }

    /// With the delete check enabled and nothing landing concurrently, the row delta commits. The mutant
    /// is a check that fails a race-free commit.
    #[tokio::test]
    async fn test_row_delta_delete_validation_no_concurrent_commit_succeeds() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a race-free row delta must commit even with the delete check enabled");

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// A concurrent row delta lands a DELETE file whose `y` bounds `[60,70]` overlap the filter
    /// `y >= 50`. The commit must fail with a non-retryable `DataInvalid` that names the file.
    ///
    /// The message assertion demands the delete-specific wording, "conflicting delete files", not the
    /// data-file wording. That is what proves the delete branch fired and not the data branch.
    #[tokio::test]
    async fn test_row_delta_rejects_concurrent_added_delete_file_matching_filter() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/my-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // S1 adds a DELETE file whose y bounds [60,70] overlap `y >= 50`.
        let _concurrent =
            commit_concurrent_deletes(&catalog, &table, vec![delete_file_with_y_bounds(
                "test/concurrent-del.parquet",
                0,
                60,
                70,
            )])
            .await;

        let err = tx.commit(&catalog).await.expect_err(
            "row delta must fail: a concurrent delete file could apply to the conflict filter",
        );

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
            err.message().contains("conflicting delete files"),
            "the error must use the DELETE-specific message, got: {}",
            err.message()
        );
        assert!(
            !err.message().contains("can contain records"),
            "the DELETE message must NOT be the data-file message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/concurrent-del.parquet"),
            "the error must name the conflicting DELETE file, got: {}",
            err.message()
        );
    }

    /// The concurrent DELETE file's `y` bounds `[10,20]` sit entirely below the filter `y >= 50`, so the
    /// row delta commits. This test fails when the shared `first_conflicting_file` metrics decision is
    /// inverted, and so does its data-file twin.
    #[tokio::test]
    async fn test_row_delta_allows_concurrent_added_delete_file_excluded_by_filter() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/my-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // S1's delete file has y bounds [10,20], entirely below `y >= 50`.
        let _concurrent =
            commit_concurrent_deletes(&catalog, &table, vec![delete_file_with_y_bounds(
                "test/concurrent-del.parquet",
                0,
                10,
                20,
            )])
            .await;

        let table = tx.commit(&catalog).await.expect(
            "row delta must commit: the concurrent delete file cannot apply to the conflict filter",
        );

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// Without the delete flag, a concurrent delete that would match the filter does not fail the commit.
    /// The mutant makes the check unconditional.
    #[tokio::test]
    async fn test_row_delta_without_delete_validation_allows_conflicting_concurrent_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // The conflict filter is supplied to prove it stays inert without the flag.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/my-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            );
        let tx = action.apply(tx).unwrap();

        // S1's delete file would match if the check were on.
        let _concurrent =
            commit_concurrent_deletes(&catalog, &table, vec![delete_file_with_y_bounds(
                "test/concurrent-del.parquet",
                0,
                60,
                70,
            )])
            .await;

        let table = tx.commit(&catalog).await.expect(
            "with the delete check OFF, a conflicting concurrent delete must not block the commit",
        );
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed (snapshot isolation, no delete-conflict check)"
        );
    }

    /// With the delete check on and no filter, the default is `AlwaysTrue`, so any concurrently added
    /// delete file conflicts, even one with no bounds. The mutant treats `None` as "no conflict".
    #[tokio::test]
    async fn test_row_delta_delete_none_filter_treats_any_concurrent_delete_as_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/my-del.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // A delete file with no bounds still conflicts under `AlwaysTrue`.
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![synthetic_delete_file(
            "test/concurrent-del.parquet",
            0,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "a None filter defaults to AlwaysTrue: any concurrent delete is a conflict",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(err.message().contains("conflicting delete files"));
        assert!(err.message().contains("test/concurrent-del.parquet"));
    }

    /// Delete files do not exist on a V1 table, so the delete check is a guarded no-op there and the row
    /// delta commits. The mutant drops the guard and walks a V1 table, which at best wastes work and at
    /// worst panics or rejects the commit.
    #[tokio::test]
    async fn test_row_delta_delete_check_is_noop_on_v1_table() {
        let catalog = new_memory_catalog().await;
        let table = make_v1_minimal_table_in_catalog(&catalog).await;
        assert_eq!(
            table.metadata().format_version(),
            crate::spec::FormatVersion::V1,
            "the table must be V1 for the guard to be under test"
        );
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_data_files(vec![synthetic_data_file("test/b.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // V1 cannot add delete files, so the concurrent commit is a DATA append.
        let _concurrent = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/concurrent.parquet",
            0,
        )])
        .await;

        let table = tx
            .commit(&catalog)
            .await
            .expect("the delete check is a no-op on a V1 table (V2 guard) — the row delta commits");

        let live = live_data_file_paths(&table).await;
        assert!(
            live.contains("test/b.parquet"),
            "the row delta's added data file landed on V1 (delete check no-op)"
        );

        // Assert the guard directly: on V1 the walk returns empty for any starting snapshot, without
        // walking at all.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        let added_deletes = crate::transaction::snapshot::added_delete_files_after(&reloaded, None)
            .await
            .unwrap();
        assert!(
            added_deletes.is_empty(),
            "the V2 guard returns an empty added-delete set on a V1 table"
        );
    }

    /// The DELETE flag alone must not run the DATA check. A concurrent data append that would match the
    /// filter passes. The mutant couples the two flags, which over-rejects here.
    #[tokio::test]
    async fn test_row_delta_delete_check_does_not_run_data_check() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/my-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // S1's y bounds [60,70] would match the filter if the data check ran.
        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent-data.parquet",
            0,
            60,
            70,
        )])
        .await;

        let table = tx.commit(&catalog).await.expect(
            "enabling only the DELETE check must not run the DATA check: a matching concurrent DATA append is allowed",
        );
        let live = live_data_file_paths(&table).await;
        assert!(
            live.contains("test/concurrent-data.parquet"),
            "the concurrent DATA file survives — the delete flag did not run the data check"
        );
    }

    /// The mirror: the DATA flag alone must not run the DELETE check. The pair pins that neither flag
    /// implies the other.
    #[tokio::test]
    async fn test_row_delta_data_check_does_not_run_delete_check() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/my-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();

        // S1's delete file would match the filter if the delete check ran.
        let _concurrent =
            commit_concurrent_deletes(&catalog, &table, vec![delete_file_with_y_bounds(
                "test/concurrent-del.parquet",
                0,
                60,
                70,
            )])
            .await;

        let table = tx.commit(&catalog).await.expect(
            "enabling only the DATA check must not run the DELETE check: a matching concurrent delete is allowed",
        );
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        // The commit re-based onto S1, so the concurrent delete file survives.
        let mut concurrent_delete_present = false;
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != ManifestContentType::Deletes {
                continue;
            }
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.is_alive() && entry.file_path() == "test/concurrent-del.parquet" {
                    concurrent_delete_present = true;
                }
            }
        }
        assert!(
            concurrent_delete_present,
            "the concurrent DELETE file survives — the data flag did not run the delete check"
        );
    }

    // The referenced-data-files-exist check (Java `validateDataFilesExist`).
    //
    // A position delete references the data file whose rows it removes. When a concurrent commit deletes
    // that file, the delete can no longer apply, so committing it would lose the delete silently.
    //
    // The skip-deletes axis: by default the walk uses `{Overwrite, Replace}`, so a concurrent
    // merge-on-read DELETE-op removal is excluded but a concurrent compaction is not. After
    // `validate_deleted_files()` the walk uses `{Overwrite, Replace, Delete}`.
    //
    // The race: a row delta is built against head S0, then a separate commit deletes a referenced file
    // and advances the head to S1.

    /// Commit a concurrent overwrite that deletes `delete_path` and adds `add_path`. It records
    /// `Operation::Overwrite`, which is in both op sets, so the skip-deletes default still sees it.
    async fn commit_concurrent_overwrite_deletion(
        catalog: &impl Catalog,
        table: &Table,
        delete_path: &str,
        add_path: &str,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx
            .overwrite_files()
            .add_file(synthetic_data_file(add_path, 0))
            .delete_file(delete_path.to_string());
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Commit a concurrent delete that removes `delete_path`. It records `Operation::Delete`, which is in
    /// the non-skip op set only, so the skip-deletes default excludes it.
    async fn commit_concurrent_delete_op_deletion(
        catalog: &impl Catalog,
        table: &Table,
        delete_path: &str,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx.delete_files().delete_file(delete_path.to_string());
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// With the files-exist check enabled and nothing removing `f`, the row delta commits. The mutant is
    /// a check that fails while the referenced file is still present.
    #[tokio::test]
    async fn test_row_delta_files_exist_no_concurrent_deletion_succeeds() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/f.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/f-pos-del.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_data_files_exist(["test/f.parquet"]);
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a row delta whose referenced file still exists must commit");

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// A concurrent overwrite deletes the referenced `f`, so the commit fails with a non-retryable
    /// `DataInvalid` that names `f`. The mutant commits a position delete over a file that is already
    /// gone, which loses the delete. An overwrite is in the default op set, so this needs no
    /// `validate_deleted_files()`.
    #[tokio::test]
    async fn test_row_delta_files_exist_rejects_concurrent_deletion_of_referenced_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/f.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/f-pos-del.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_data_files_exist(["test/f.parquet"]);
        let tx = action.apply(tx).unwrap();

        // S1 overwrites: it deletes the referenced f and adds a sibling.
        let _concurrent = commit_concurrent_overwrite_deletion(
            &catalog,
            &table,
            "test/f.parquet",
            "test/g.parquet",
        )
        .await;

        let err = tx
            .commit(&catalog)
            .await
            .expect_err("row delta must fail: a referenced data file was concurrently deleted");

        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "a missing referenced data file is a non-retryable validation failure (DataInvalid)"
        );
        assert!(
            !err.retryable(),
            "the validation failure must be NON-retryable so the retry loop stops and it propagates"
        );
        assert!(
            err.message().contains("Cannot commit, missing data files"),
            "the error must use the missing-data-files message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/f.parquet"),
            "the error must name the missing referenced FILE, got: {}",
            err.message()
        );
    }

    /// Commit a concurrent compaction: rewrite `delete_path` into `add_path`, which records
    /// `Operation::Replace`. It produces the same tombstone shape as an overwrite under a different
    /// operation. A maintained table produces this concurrency constantly, because `RewriteDataFiles`,
    /// `RemoveDanglingDeleteFiles`, and `RewritePositionDeleteFiles` all commit this way.
    async fn commit_concurrent_replace_compaction(
        catalog: &impl Catalog,
        table: &Table,
        delete_path: &str,
        add_path: &str,
    ) -> Table {
        let replaced = synthetic_data_file(delete_path, 0);
        let compacted = synthetic_data_file(add_path, 0);
        let tx = Transaction::new(table);
        let action = tx.rewrite_files(vec![replaced], vec![compacted]);
        let tx = action.apply(tx).expect("rewrite_files action applies");
        tx.commit(catalog)
            .await
            .expect("the concurrent compaction commit must succeed")
    }

    /// The `skip_deletes == true` arm, which is RowDelta's default. A concurrent compaction rewrites the
    /// referenced `f`, and the commit must fail with a non-retryable `DataInvalid` naming `f`.
    ///
    /// The mutant drops `Operation::Replace` from `operation_removes_data_files_skip_deletes`. The
    /// compaction's tombstone for `f` is then never inspected, the row delta commits, and the rows its
    /// position delete removed are live again in the compacted output. Silent, with no error and no retry.
    ///
    /// The test also pins the fixture: the concurrent snapshot must really record `Replace`, or it would
    /// pass vacuously through another member of the op set.
    #[tokio::test]
    async fn test_row_delta_files_exist_rejects_concurrent_replace_compaction_of_referenced_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/f.parquet",
            0,
        )])
        .await;

        // `validate_deleted_files()` is deliberately not called, so `skip_deletes` stays true.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/f-pos-del.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_data_files_exist(["test/f.parquet"]);
        let tx = action.apply(tx).expect("row delta action applies");

        let compacted_table = commit_concurrent_replace_compaction(
            &catalog,
            &table,
            "test/f.parquet",
            "test/f-compacted.parquet",
        )
        .await;

        // Fixture pin: the concurrent snapshot really records `Replace`, and `f` really is gone.
        let concurrent_snapshot = compacted_table
            .metadata()
            .current_snapshot()
            .expect("the compaction produced a snapshot");
        assert_eq!(
            concurrent_snapshot.summary().operation,
            Operation::Replace,
            "the concurrent compaction must record Operation::Replace — otherwise this test would \
             exercise a different op-set member and prove nothing about REPLACE"
        );
        let live = live_data_file_paths(&compacted_table).await;
        assert!(
            !live.contains("test/f.parquet"),
            "the compaction removed the referenced data file f, live = {live:?}"
        );
        assert!(
            live.contains("test/f-compacted.parquet"),
            "the compaction added the rewritten file, live = {live:?}"
        );

        let err = tx.commit(&catalog).await.expect_err(
            "row delta must fail: a concurrent REPLACE (compaction) removed the referenced data file",
        );

        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "a missing referenced data file is a non-retryable validation failure (DataInvalid)"
        );
        assert!(
            !err.retryable(),
            "the validation failure must be NON-retryable so the retry loop stops and it propagates"
        );
        assert!(
            err.message().contains("Cannot commit, missing data files"),
            "the error must use the missing-data-files message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/f.parquet"),
            "the error must name the missing referenced FILE, got: {}",
            err.message()
        );
    }

    /// The concurrent overwrite deletes a file the row delta does not reference, so the commit succeeds.
    /// The mutant rejects any concurrent deletion. This test is what makes the referenced set, and not the
    /// bare fact of a deletion, the load-bearing gate.
    #[tokio::test]
    async fn test_row_delta_files_exist_allows_concurrent_deletion_of_different_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![
            synthetic_data_file("test/f.parquet", 0),
            synthetic_data_file("test/other.parquet", 0),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/f-pos-del.parquet", 0)])
            .validate_from_snapshot(s0)
            .validate_data_files_exist(["test/f.parquet"]);
        let tx = action.apply(tx).unwrap();

        // S1 deletes the non-referenced `other`.
        let _concurrent = commit_concurrent_overwrite_deletion(
            &catalog,
            &table,
            "test/other.parquet",
            "test/g.parquet",
        )
        .await;

        let table = tx
            .commit(&catalog)
            .await
            .expect("the row delta must commit: the concurrently-deleted file is not referenced");

        let live = live_data_file_paths(&table).await;
        assert!(
            live.contains("test/f.parquet"),
            "the referenced file f survives the concurrent deletion of a DIFFERENT file"
        );
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// With an empty referenced set, a concurrent deletion of the referenced file does not fail the
    /// commit. Only a non-empty set arms the check. The mutant runs it for every row delta.
    #[tokio::test]
    async fn test_row_delta_files_exist_without_referenced_set_does_not_check() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/f.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/f-pos-del.parquet", 0)]);
        let tx = action.apply(tx).unwrap();

        // S1 deletes f, which would conflict if the check were armed.
        let _concurrent = commit_concurrent_overwrite_deletion(
            &catalog,
            &table,
            "test/f.parquet",
            "test/g.parquet",
        )
        .await;

        let table = tx.commit(&catalog).await.expect(
            "with an empty referenced set, a concurrent deletion must not block the commit",
        );
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed (no files-exist check ran)"
        );
    }

    /// Both halves of the skip-deletes axis on one concurrent DELETE-op removal of the referenced `f`. By
    /// default the row delta commits, because the op set excludes `Delete`. After
    /// `validate_deleted_files()` the same removal is rejected. `Replace` is in both sets, so this
    /// isolates the `Delete` member.
    ///
    /// The mutant flips the default to always include DELETE-op snapshots, which rejects the concurrent
    /// merge-on-read delete the default is meant to tolerate. No other test tells the two op sets apart.
    #[tokio::test]
    async fn test_row_delta_files_exist_skip_deletes_default_excludes_delete_op_snapshot() {
        // Half A: the default excludes a DELETE-op deletion, so the commit succeeds.
        {
            let catalog = new_memory_catalog().await;
            let table = make_v2_minimal_table_in_catalog(&catalog).await;
            let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
                "test/f.parquet",
                0,
            )])
            .await;

            let tx = Transaction::new(&table);
            let action = tx
                .row_delta()
                .add_deletes(vec![synthetic_delete_file("test/f-pos-del.parquet", 0)])
                .validate_from_snapshot(s0)
                .validate_data_files_exist(["test/f.parquet"]);
            let tx = action.apply(tx).unwrap();

            let _concurrent =
                commit_concurrent_delete_op_deletion(&catalog, &table, "test/f.parquet").await;

            let table = tx.commit(&catalog).await.expect(
                "by default a concurrent DELETE-op deletion is excluded (skip_deletes) ⇒ commit succeeds",
            );
            let snapshot = table.metadata().current_snapshot().unwrap();
            let manifest_list = snapshot
                .load_manifest_list(table.file_io(), table.metadata())
                .await
                .unwrap();
            assert!(
                manifest_list
                    .entries()
                    .iter()
                    .any(|m| m.content == ManifestContentType::Deletes),
                "the row delta committed under the skip-deletes default"
            );
        }

        // Half B: `validate_deleted_files` makes the same deletion a conflict.
        {
            let catalog = new_memory_catalog().await;
            let table = make_v2_minimal_table_in_catalog(&catalog).await;
            let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
                "test/f.parquet",
                0,
            )])
            .await;

            let tx = Transaction::new(&table);
            let action = tx
                .row_delta()
                .add_deletes(vec![synthetic_delete_file("test/f-pos-del.parquet", 0)])
                .validate_from_snapshot(s0)
                .validate_data_files_exist(["test/f.parquet"])
                .validate_deleted_files();
            let tx = action.apply(tx).unwrap();

            let _concurrent =
                commit_concurrent_delete_op_deletion(&catalog, &table, "test/f.parquet").await;

            let err = tx.commit(&catalog).await.expect_err(
                "validate_deleted_files() includes DELETE-op snapshots ⇒ the deletion of f is a conflict",
            );
            assert_eq!(err.kind(), ErrorKind::DataInvalid);
            assert!(!err.retryable());
            assert!(
                err.message().contains("Cannot commit, missing data files")
                    && err.message().contains("test/f.parquet"),
                "the error must name the missing referenced file, got: {}",
                err.message()
            );
        }
    }

    /// The files-exist check with no explicit `validate_from_snapshot`, so only the start captured in
    /// `Transaction::new` can make S1's deletion of `f` concurrent.
    ///
    /// The mutant re-reads the start from the refreshed head, so the concurrently-deleted set is always
    /// empty and the check always passes. Every other files-exist test pins the start explicitly, which
    /// short-circuits the captured field, so this is the only one that discriminates that mutant.
    #[tokio::test]
    async fn test_row_delta_files_exist_rejects_concurrent_using_tx_captured_starting_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/f.parquet",
            0,
        )])
        .await;

        // No `validate_from_snapshot`, so the start is the transaction-captured head S0.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/f-pos-del.parquet", 0)])
            .validate_data_files_exist(["test/f.parquet"]);
        let tx = action.apply(tx).unwrap();

        let _concurrent = commit_concurrent_overwrite_deletion(
            &catalog,
            &table,
            "test/f.parquet",
            "test/g.parquet",
        )
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "the missing referenced file must be detected via the tx-captured starting snapshot",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message().contains("Cannot commit, missing data files")
                && err.message().contains("test/f.parquet"),
            "got: {}",
            err.message()
        );
    }

    // `validateNoNewDeletesForDataFiles` on removed data files. You must not drop a data file out from
    // under a concurrent row-level delete.
    //
    // It rides the same `validate_no_conflicting_delete_files()` flag as the filter-based delete check and
    // uses the same shared helper, with `ignore_equality_deletes = false`.
    //
    // The race: a row delta built against head S0 removes data file A, then a concurrent
    // `row_delta().add_deletes` lands a position delete in A's partition. With the flag on, the commit
    // must fail. A delete in another partition, a delete at or before the start, the flag off, no removed
    // files, and a V1 table must all commit.

    /// An equality-delete file in partition `x = part_value`, equality on field id 1. It proves the
    /// removed-data-files check counts equality deletes.
    fn synthetic_equality_delete_file(path: &str, part_value: i64) -> DataFile {
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

    /// With the delete check enabled, a removed file A, and nothing landing concurrently, the row delta
    /// commits. The mutant fails a race-free commit.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_no_concurrent_delete_succeeds() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .remove_data_files(vec![a])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a race-free row delta removing a data file must commit");

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// A row delta removes A while a concurrent commit lands a position delete in A's partition. The
    /// commit must fail with a non-retryable `DataInvalid` that names A. The mutant drops A out from under
    /// that delete, which loses it under serializable isolation.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_rejects_concurrent_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .remove_data_files(vec![a])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // S1 lands a position delete in A's partition, at a sequence number after the start.
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![synthetic_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "row delta must fail: a concurrent delete applies to the removed data file A",
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
            "the error must name the removed data file, got: {}",
            err.message()
        );
    }

    /// The concurrent position delete is in partition x=1, and the removed file A is in x=0, so sub-check
    /// 2a does not apply it to A. The mutant makes 2a partition-blind and rejects any concurrent delete.
    ///
    /// The delete's `y` bounds `[10,20]` sit below the filter `y >= 50` only to keep sub-check 2b, which
    /// runs on the same flag, quiet. The partition logic is what this test proves.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_allows_concurrent_delete_in_other_partition() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .remove_data_files(vec![a])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // S1's delete is in partition x=1 with y bounds [10,20], so neither sub-check applies it to A.
        let _concurrent =
            commit_concurrent_deletes(&catalog, &table, vec![delete_file_with_y_bounds(
                "test/pos-del-other.parquet",
                1,
                10,
                20,
            )])
            .await;

        let table = tx
            .commit(&catalog)
            .await
            .expect("row delta must commit: the concurrent delete is in a different partition");
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// The delete lands before the validation window, because the start is pinned to the current head, so
    /// the row delta commits. The mutant ignores the starting-snapshot boundary and flags a pre-start
    /// delete.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_allows_delete_at_or_before_start() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone()]).await;

        // This delete lands in A's partition before the validation window and becomes head S1.
        let table = commit_concurrent_deletes(&catalog, &table, vec![synthetic_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;
        let s1 = table.metadata().current_snapshot().unwrap().snapshot_id();

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .remove_data_files(vec![a])
            .validate_from_snapshot(s1)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("with start = current head, the pre-existing delete is not concurrent ⇒ commit succeeds");
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// A concurrent equality delete in A's partition conflicts with removing A, because RowDelta passes
    /// `ignore_equality_deletes = false`. The mutant ignores equality deletes, as the rewrite path does.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_rejects_concurrent_equality_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .remove_data_files(vec![a])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // S1 lands an equality delete in A's partition.
        let _concurrent =
            commit_concurrent_deletes(&catalog, &table, vec![synthetic_equality_delete_file(
                "test/eq-del.parquet",
                0,
            )])
            .await;

        let err = tx.commit(&catalog).await.expect_err(
            "row delta must fail: an equality delete applies to the removed data file A",
        );
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

    /// Without the flag, a concurrent delete that applies to the removed file does not fail the commit.
    /// It also proves `remove_data_files` alone arms nothing.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_without_validation_allows_conflicting_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .remove_data_files(vec![a]);
        let tx = action.apply(tx).unwrap();

        // S1 lands a position delete that applies to A.
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![synthetic_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;

        let table = tx.commit(&catalog).await.expect(
            "with the delete check OFF, a conflicting concurrent delete must not block the commit",
        );
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed (snapshot isolation, no conflicting-delete check)"
        );
    }

    /// With the delete check on but no removed files, sub-check 2a skips outright. The concurrent delete's
    /// `y` bounds `[10,20]` sit below the filter `y >= 50`, so sub-check 2b excludes it too, and the row
    /// delta commits. The mutant runs 2a on an empty removed set with `AlwaysTrue` semantics.
    #[tokio::test]
    async fn test_row_delta_no_removed_data_files_skips_removed_check() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // S1's delete sits below the filter, so 2b excludes it, and 2a never runs.
        let _concurrent =
            commit_concurrent_deletes(&catalog, &table, vec![delete_file_with_y_bounds(
                "test/pos-del-other.parquet",
                0,
                10,
                20,
            )])
            .await;

        let table = tx.commit(&catalog).await.expect(
            "with no removed data files, the removed-data-files sub-check is skipped ⇒ commit succeeds",
        );
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// On a V1 table the removed-data-files check is a guarded no-op, so the row delta commits. The mutant
    /// walks a V1 table, where delete manifests cannot exist.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_check_is_noop_on_v1_table() {
        let catalog = new_memory_catalog().await;
        let table = make_v1_minimal_table_in_catalog(&catalog).await;
        assert_eq!(
            table.metadata().format_version(),
            crate::spec::FormatVersion::V1,
            "the table must be V1 for the guard to be under test"
        );
        let a = synthetic_data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_data_files(vec![synthetic_data_file("test/b.parquet", 0)])
            .remove_data_files(vec![a])
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // V1 cannot add delete files, so the concurrent commit is a DATA append.
        let _concurrent = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/concurrent.parquet",
            0,
        )])
        .await;

        let table = tx.commit(&catalog).await.expect(
            "the removed-data-files check is a no-op on a V1 table (V2 guard) — the row delta commits",
        );
        assert!(
            live_data_file_paths(&table)
                .await
                .contains("test/b.parquet"),
            "the row delta's added data file landed on V1 (removed-data-files check no-op)"
        );
    }

    /// The removed-data-files check with no explicit `validate_from_snapshot`, so only the start captured
    /// in `Transaction::new` can make the concurrent delete visible.
    ///
    /// The mutant re-reads the start from the refreshed head, so the concurrent set is always empty and the
    /// check always passes. Every other test of this sub-check pins the start explicitly.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_rejects_using_tx_captured_starting_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone()]).await;

        // No `validate_from_snapshot`, so the start is the transaction-captured head S0.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .remove_data_files(vec![a])
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // S1 lands a position delete that applies to A.
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![synthetic_delete_file(
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

    // `removeRows` apply-side removal. These tests pin that a removed data file drops from the scan, that
    // the operation classification does not change, that a missing path fails loud, that the
    // removed-against-referenced rejection still fires first in `validate()`, and that the summary
    // counters move.

    /// One row delta removes A and adds a position delete for B. The live set afterwards is exactly {B},
    /// and the snapshot carries a rewritten DATA manifest with A tombstoned plus a DELETE manifest.
    ///
    /// The mutant severs the producer routing, so `RowDeltaOperation::delete_files` returns an empty vec
    /// and A stays visible.
    #[tokio::test]
    async fn test_row_delta_remove_data_file_drops_it_from_scan_with_added_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let b = synthetic_data_file("test/b.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone(), b]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/b-pos-del.parquet", 0)])
            .remove_rows(a);
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a row delta removing A and adding a delete must commit");

        assert_eq!(
            live_data_file_paths(&table).await,
            HashSet::from(["test/b.parquet".to_string()]),
            "removeRows must drop A from the scan (apply-side removal); B remains"
        );

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut a_tombstoned = false;
        let mut has_delete_manifest = false;
        for manifest_file in manifest_list.entries() {
            if manifest_file.content == ManifestContentType::Deletes {
                has_delete_manifest = true;
            }
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.file_path() == "test/a.parquet" {
                    assert_eq!(entry.status(), ManifestStatus::Deleted);
                    a_tombstoned = true;
                }
            }
        }
        assert!(a_tombstoned, "A must be a Deleted tombstone in the rewrite");
        assert!(
            has_delete_manifest,
            "the added delete's DELETE manifest must be present"
        );
    }

    /// A remove-only row delta drops A from the scan and records `Operation::Overwrite`. `removeRows` is
    /// not an `addsDeleteFiles`, so the deletes-only branch does not fire and the classification falls
    /// through. It also pins that a remove-only commit is not empty.
    #[tokio::test]
    async fn test_row_delta_remove_only_drops_file_and_records_overwrite() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let b = synthetic_data_file("test/b.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone(), b]).await;

        let tx = Transaction::new(&table);
        let action = tx.row_delta().remove_rows(a);
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a remove-only row delta is non-empty and must commit");

        assert_eq!(
            live_data_file_paths(&table).await,
            HashSet::from(["test/b.parquet".to_string()]),
            "remove-only row delta drops A from the scan"
        );
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Overwrite,
            "1.10.0 operation() is two-branch: a remove-only row delta (no added deletes) is Overwrite"
        );
    }

    /// Removing a data file that the table does not hold fails loud at commit and adds nothing. The
    /// apply-side removal validates its target exists, as `OverwriteFiles` does.
    #[tokio::test]
    async fn test_row_delta_remove_absent_data_file_errors() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_data_files(vec![synthetic_data_file("test/c.parquet", 0)])
            .remove_rows(synthetic_data_file("test/does-not-exist.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("removing an absent data file must fail loud");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Missing required files to delete"),
            "unexpected message: {}",
            err.message()
        );

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_data_file_paths(&reloaded).await,
            HashSet::from(["test/a.parquet".to_string()]),
            "a failed removal leaves the table untouched (no partial add)"
        );
    }

    /// The removed-against-referenced rejection fires in `validate()`, which runs before `commit()`. A row
    /// delta that removes a file its added deletes reference is rejected first, so the apply-side removal
    /// never runs and the table is untouched. This pins the rejection and its precedence.
    #[tokio::test]
    async fn test_row_delta_removed_referenced_rejection_fires_before_apply_side_removal() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .validate_data_files_exist(["test/a.parquet"])
            .remove_rows(synthetic_data_file("test/a.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "removing a referenced data file is a self-contradiction rejected before apply-side removal",
        );
        assert_eq!(
            err.message(),
            "Cannot delete data files [test/a.parquet] that are referenced by new delete files",
            "the removed∩referenced check (in validate) fires first, before the apply-side removal"
        );

        // A is still live, because the rejection happened in `validate()`.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            live_data_file_paths(&reloaded)
                .await
                .contains("test/a.parquet"),
            "A survives — the rejection precedes any apply-side removal"
        );
    }

    /// A removal flows through the summary's `remove_file`. With A and B appended and A removed,
    /// `total-data-files` is 1, `deleted-data-files` is 1, and `deleted-records` is 1.
    #[tokio::test]
    async fn test_row_delta_remove_data_file_summary_counters() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let b = synthetic_data_file("test/b.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone(), b]).await;
        assert_eq!(
            summary_prop(&table, "total-data-files").as_deref(),
            Some("2"),
            "after appending A, B: two data files"
        );

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/b-pos-del.parquet", 0)])
            .remove_rows(a);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            summary_prop(&table, "deleted-data-files").as_deref(),
            Some("1"),
            "removeRows must surface deleted-data-files = 1 via remove_file"
        );
        assert_eq!(
            summary_prop(&table, "deleted-records").as_deref(),
            Some("1"),
            "removeRows must surface deleted-records = 1 (A's record_count)"
        );
        assert_eq!(
            summary_prop(&table, "total-data-files").as_deref(),
            Some("1"),
            "cumulative total-data-files = 2 + 0 − 1 = 1 (seed-from-previous − removed)"
        );
        assert_eq!(
            summary_prop(&table, "total-records").as_deref(),
            Some("1"),
            "cumulative total-records = 2 + 0 − 1 = 1"
        );
    }

    /// A row delta that removes path X and adds a fresh file at the SAME path X. The manifest filter
    /// tombstones the old entry and the producer writes the new one, so X survives as the added file.
    ///
    /// This pins three non-outcomes: X does not vanish, it is not live twice, and the removal is not a
    /// silent no-op. The summary shows one removal and one add, and the total stays at one file.
    #[tokio::test]
    async fn test_row_delta_remove_and_add_same_path_replaces_in_place() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a_old = synthetic_data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a_old.clone()]).await;

        // The record_count differs, so the add is observable in the summary.
        let a_new = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/a.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(7)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_data_files(vec![a_new])
            .remove_rows(a_old);
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("remove + add of the same path commits (replace-in-place)");

        assert_eq!(
            live_data_file_paths(&table).await,
            HashSet::from(["test/a.parquet".to_string()]),
            "X survives as the freshly-added file — replace-in-place, not a vanish"
        );

        // The snapshot carries an Added AND a Deleted entry for the path.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let (mut added, mut deleted) = (false, false);
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.file_path() == "test/a.parquet" {
                    match entry.status() {
                        ManifestStatus::Added => added = true,
                        ManifestStatus::Deleted => deleted = true,
                        ManifestStatus::Existing => {}
                    }
                }
            }
        }
        assert!(added, "the fresh file is Added");
        assert!(deleted, "the old file is Deleted (the removal is honored)");

        assert_eq!(
            summary_prop(&table, "deleted-data-files").as_deref(),
            Some("1"),
            "the remove is counted"
        );
        assert_eq!(
            summary_prop(&table, "added-data-files").as_deref(),
            Some("1"),
            "the add is counted"
        );
        assert_eq!(
            summary_prop(&table, "total-data-files").as_deref(),
            Some("1"),
            "cumulative total-data-files = 1 + 1 − 1 = 1"
        );
    }

    // `validateAddedDVs`, the V3 deletion-vector conflict check. It always runs, and it self-skips when
    // this row delta adds no DV.
    //
    // Two DVs for one data file is a write-write conflict. The concurrent walk is
    // `added_dv_candidate_delete_files_after`, gated to Java's `{Overwrite, Delete, Replace}`. `Replace` is
    // in the set because a compaction can rewrite DVs.
    //
    // The race: a row delta adding a DV for A is built against head S0, then a concurrent row delta lands
    // a DV for the same A.

    /// Commit a concurrent DV-only row delta through the catalog. Its operation is `Delete`, which is in
    /// the DV walk's op set. The `Replace` member has its own test through `ReplaceOpAddDvAction`.
    async fn commit_concurrent_dvs(
        catalog: &impl Catalog,
        table: &Table,
        dv_files: Vec<DataFile>,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx.row_delta().add_deletes(dv_files);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// A concurrent row delta lands a DV for the same data file A. The commit must fail with a
    /// non-retryable `DataInvalid` that names A. No flag arms this check.
    ///
    /// The mutant commits a second DV for A blind to the concurrent one, and that DV silently shadows or
    /// loses the first.
    #[tokio::test]
    async fn test_row_delta_rejects_concurrent_dv_for_same_referenced_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // No conflict flag is set, because the DV check always runs.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv.puffin",
                0,
                "test/a.parquet",
            )])
            .validate_from_snapshot(s0);
        let tx = action.apply(tx).unwrap();

        let _concurrent = commit_concurrent_dvs(&catalog, &table, vec![synthetic_dv_file(
            "test/a-dv-concurrent.puffin",
            0,
            "test/a.parquet",
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "row delta must fail: a concurrent DV was added for the same referenced data file",
        );

        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "a DV conflict is a non-retryable validation failure (DataInvalid)"
        );
        assert!(
            !err.retryable(),
            "the validation failure must be NON-retryable so the retry loop stops and it propagates"
        );
        assert!(
            err.message().contains("Found concurrently added DV for"),
            "the error must use the DV-specific message, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/a.parquet"),
            "the error must name the referenced data file, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/a-dv-concurrent.puffin"),
            "the error must describe the concurrently-added DV, got: {}",
            err.message()
        );
    }

    /// The concurrent DV references B, and this row delta's DV references A, so nothing collides and the
    /// commit succeeds. The mutant rejects any concurrently added DV, which breaks legitimate concurrent DV
    /// writes on unrelated data files.
    #[tokio::test]
    async fn test_row_delta_allows_concurrent_dv_for_different_referenced_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![
            synthetic_data_file("test/a.parquet", 0),
            synthetic_data_file("test/b.parquet", 0),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv.puffin",
                0,
                "test/a.parquet",
            )])
            .validate_from_snapshot(s0);
        let tx = action.apply(tx).unwrap();

        // S1's DV references B, so it cannot collide with A.
        let _concurrent = commit_concurrent_dvs(&catalog, &table, vec![synthetic_dv_file(
            "test/b-dv-concurrent.puffin",
            0,
            "test/b.parquet",
        )])
        .await;

        let table = tx
            .commit(&catalog)
            .await
            .expect("row delta must commit: the concurrent DV references a different data file");

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// A row delta adds a DV for A with nothing landing concurrently, so it commits. The mutant is an
    /// always-on check that blocks a race-free DV commit.
    #[tokio::test]
    async fn test_row_delta_dv_no_concurrent_commit_succeeds() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv.puffin",
                0,
                "test/a.parquet",
            )])
            .validate_from_snapshot(s0);
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("a race-free DV row delta must commit (no concurrent DV ⇒ no conflict)");

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the row delta committed: a DELETE manifest is present"
        );
    }

    /// A row delta that adds only non-DV deletes commits even while a concurrent DV is present, because
    /// the always-on check self-skips. Every pre-DV test here adds non-Puffin deletes, so this pin carries
    /// their behavior preservation.
    ///
    /// The mutant fires the DV check on a non-DV row delta, which over-rejects the common merge-on-read
    /// case.
    #[tokio::test]
    async fn test_row_delta_non_dv_delete_is_noop_even_with_concurrent_dv() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // The non-DV delete must be an EQUALITY delete on this V3 table. Equality deletes are exempt
        // from the format gate, but a parquet position delete would fail the gate before the DV check.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_equality_delete_file(
                "test/a-eq-del.parquet",
                0,
            )])
            .validate_from_snapshot(s0);
        let tx = action.apply(tx).unwrap();

        // S1's DV for the same A would collide if this row delta added a DV.
        let _concurrent = commit_concurrent_dvs(&catalog, &table, vec![synthetic_dv_file(
            "test/a-dv-concurrent.puffin",
            0,
            "test/a.parquet",
        )])
        .await;

        let table = tx.commit(&catalog).await.expect(
            "a non-DV row delta adds no DV ⇒ the always-on DV check self-skips ⇒ commit succeeds even with a concurrent DV",
        );

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the non-DV row delta committed: a DELETE manifest is present"
        );
    }

    /// The DV check with no explicit `validate_from_snapshot`, so only the start captured in
    /// `Transaction::new` can make the concurrent DV visible.
    ///
    /// The mutant re-reads the start from the refreshed head, so the concurrent DV set is always empty and
    /// the check always passes. Every other DV test pins the start explicitly.
    #[tokio::test]
    async fn test_row_delta_dv_rejects_using_tx_captured_starting_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // No `validate_from_snapshot`, so the start is the transaction-captured head S0.
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();

        let _concurrent = commit_concurrent_dvs(&catalog, &table, vec![synthetic_dv_file(
            "test/a-dv-concurrent.puffin",
            0,
            "test/a.parquet",
        )])
        .await;

        let err = tx
            .commit(&catalog)
            .await
            .expect_err("the DV conflict must be detected via the tx-captured starting snapshot");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message().contains("Found concurrently added DV for")
                && err.message().contains("test/a.parquet"),
            "got: {}",
            err.message()
        );
    }

    /// A Puffin delete file with no `referenced_data_file` is malformed, and validating it must return a
    /// clear `DataInvalid`, not panic and not skip. The mutant lets it through, which puts a DV that covers
    /// no data file into the DV index.
    #[tokio::test]
    async fn test_row_delta_dv_missing_referenced_data_file_is_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let mut malformed_dv = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/bad-dv.puffin".to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .referenced_data_file(Some("placeholder.parquet".to_string()))
            .build()
            .unwrap();
        // The builder refuses this shape, so only a decoded manifest entry can carry it.
        malformed_dv.referenced_data_file = None;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![malformed_dv])
            .validate_from_snapshot(s0);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a Puffin DV missing its referenced data file must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("missing its referenced data file"),
            "got: {}",
            err.message()
        );
    }

    // The format-version gate (Java `validateDeleteFileForVersion`). V1 rejects all deletes, V2
    // forbids DVs for position deletes, and V3 requires them. Equality deletes are exempt at every
    // version. The gate runs in `validate_added_delete_files` against the refreshed base.
    //
    // The accept directions are pinned elsewhere: V2 with a parquet position delete by the whole V2
    // suite, and V3 with a DV by `test_row_delta_dv_no_concurrent_commit_succeeds`.

    /// V2 rejects a deletion vector with Java's exact message. A V2 table that carries a Puffin DV
    /// is unreadable by every V2 reader, so the gate must fail the commit.
    #[tokio::test]
    async fn test_row_delta_v2_rejects_deletion_vector_with_exact_java_message() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a deletion vector must be rejected on a V2 table");

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable(), "the format gate is non-retryable");
        assert_eq!(
            err.message(),
            "Must not use DVs for position deletes in V2: DV{location=test/a-dv.puffin, \
             offset=4, length=40, referencedDataFile=test/a.parquet}",
            "the V2 gate message must match Java byte-for-byte (incl. dvDesc)"
        );
    }

    /// V3 rejects a parquet position delete with Java's exact message. A fresh parquet position
    /// delete on a V3 table breaks the read precedence that lets a DV supersede position deletes.
    #[tokio::test]
    async fn test_row_delta_v3_rejects_parquet_position_delete_with_exact_java_message() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a parquet position delete must be rejected on a V3 table");

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable(), "the format gate is non-retryable");
        assert_eq!(
            err.message(),
            "Must use DVs for position deletes in V3: test/a-pos-del.parquet",
            "the V3 gate message must match Java byte-for-byte"
        );
    }

    /// V1 rejects every added delete file, position and equality, with Java's exact message. A V1
    /// manifest cannot encode delete content at all. The test runs at producer level, because no
    /// in-catalog V1 fixture exists, and the gate is the same one.
    #[tokio::test]
    async fn test_v1_producer_rejects_all_added_delete_files() {
        use crate::transaction::tests::make_v1_table;

        let table = make_v1_table();
        for delete_file in [
            synthetic_delete_file("test/a-pos-del.parquet", 0),
            synthetic_equality_delete_file("test/a-eq-del.parquet", 0),
        ] {
            let producer = SnapshotProducer::new(
                &table,
                uuid::Uuid::now_v7(),
                None,
                HashMap::new(),
                vec![],
                FirstRowIdPolicy::Suppress,
            )
            .with_added_delete_files(vec![delete_file]);
            let err = producer
                .validate_added_delete_files()
                .expect_err("a V1 table must reject every added delete file");
            assert_eq!(err.kind(), ErrorKind::DataInvalid);
            assert_eq!(
                err.message(),
                "Deletes are supported in V2 and above",
                "the V1 gate message must match Java byte-for-byte"
            );
        }
    }

    /// Equality deletes are exempt from the gate at V2 and V3. The mutant demands DVs for equality
    /// deletes too, which breaks every V3 equality-delete commit, because Puffin cannot carry them.
    #[tokio::test]
    async fn test_equality_deletes_exempt_from_version_gate_on_v2_and_v3() {
        let catalog = new_memory_catalog().await;
        for v3 in [false, true] {
            let table = if v3 {
                make_v3_minimal_table_in_catalog(&catalog).await
            } else {
                make_v2_minimal_table_in_catalog(&catalog).await
            };
            let table = append_files(&catalog, &table, vec![synthetic_data_file(
                "test/a.parquet",
                0,
            )])
            .await;

            let tx = Transaction::new(&table);
            let action = tx
                .row_delta()
                .add_deletes(vec![synthetic_equality_delete_file(
                    "test/a-eq-del.parquet",
                    0,
                )]);
            let tx = action.apply(tx).unwrap();
            tx.commit(&catalog).await.unwrap_or_else(|err| {
                panic!(
                    "an equality delete must pass the format gate on a {} table, got: {err}",
                    if v3 { "V3" } else { "V2" }
                )
            });
        }
    }

    // The fresh-DV-only door, see `validate_fresh_dvs_only`. A DV added for a data file that already
    // has a live position-scoped delete is rejected loud at commit, instead of corrupting the table
    // late at scan. `test_row_delta_dv_no_concurrent_commit_succeeds` pins the accept direction.

    /// A bounds-scoped position delete must not escape the door. Java's `PositionDeleteWriter`
    /// never sets `referenced_data_file`, and leaves equal `file_path` bounds instead, so that is how
    /// nearly every Java-written file-granularity delete names its data file.
    ///
    /// The mutant reads the field alone, which makes the whole class look partition-scoped. A delete
    /// stamped under another spec then passes the door, the DV supersedes it at read time, and its
    /// rows come back. Spark writes at FILE granularity by default, so this shape is routine.
    #[tokio::test]
    async fn test_row_delta_dv_over_bounds_scoped_position_delete_is_rejected() {
        use crate::spec::FormatVersion;

        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // This delete names test/a.parquet only through equal file_path bounds, and is stamped
        // under a partition the data file does not share, which is legal on V2.
        let bounds_scoped = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/a-pos.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(999))]))
            .lower_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("test/a.parquet"),
            )]))
            .upper_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("test/a.parquet"),
            )]))
            .build()
            .unwrap();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![bounds_scoped]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // The parquet position delete stays live across the upgrade to V3.
        let tx = Transaction::new(&table);
        let action = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a live bounds-scoped position delete still applies to that data file");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("test/a-pos.parquet"),
            "the door must name the delete file it protected, got: {}",
            err.message()
        );
    }

    /// DV1 for data file A commits, then a LATER transaction adds DV2 for the same A. That
    /// transaction starts after DV1 lands, so there is no concurrent window and `validateAddedDVs`
    /// passes. Only the door can reject it.
    ///
    /// The mutant leaves two live DVs for A, which the scan's duplicate-DV door rejects later and the
    /// table reads as unusable. The message must name the referenced file and the way out.
    #[tokio::test]
    async fn test_row_delta_second_dv_for_same_file_rejected_unless_it_supersedes() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv1.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // This transaction starts after DV1 lands, so the door is the only guard here.
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv2.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a second DV for a data file with a live DV must be rejected");

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message()
                .contains("Cannot commit deletion vector for test/a.parquet"),
            "the door must name the referenced data file, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/a-dv1.puffin"),
            "the door must describe the existing live DV, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("remove_deletes_many"),
            "the door must tell the caller how to merge instead, got: {}",
            err.message()
        );
    }

    /// A DV for a different data file commits while another file already has a live DV. The mutant
    /// keys the door on "any live DV exists", which freezes every DV write after the first.
    #[tokio::test]
    async fn test_row_delta_dv_for_different_file_commits_despite_existing_dv() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            synthetic_data_file("test/a.parquet", 0),
            synthetic_data_file("test/b.parquet", 0),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/b-dv.puffin",
            0,
            "test/b.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(&catalog)
            .await
            .expect("a DV for a different data file must commit — the door is per-file");
    }

    /// The V2 to V3 upgrade case. A V2 table commits a partition-scoped parquet position delete in
    /// x=0, upgrades to V3, then adds a DV for a data file in x=0.
    ///
    /// A DV supersedes every parquet position delete for its data file at read time, so the mutant
    /// commits the DV unmerged and resurrects the parquet delete's positions. A DV for a data file in
    /// x=1, where the parquet delete does not apply, must still commit.
    #[tokio::test]
    async fn test_row_delta_dv_rejected_when_legacy_parquet_position_delete_still_applies() {
        use crate::spec::FormatVersion;

        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            synthetic_data_file("test/a.parquet", 0),
            synthetic_data_file("test/b.parquet", 1),
        ])
        .await;

        // A partition-scoped parquet position delete in x=0, which is legal on V2.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/x0-pos-del.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // The parquet position delete stays live across the upgrade to V3.
        let tx = Transaction::new(&table);
        let action = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // The live parquet delete still applies to A, so this DV would supersede it.
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "a DV for a data file a live parquet position delete still applies to must be rejected",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Cannot commit deletion vector for test/a.parquet")
                && err.message().contains("test/x0-pos-del.parquet")
                && err.message().contains("superseded"),
            "the door must name the referenced file, the shadowed parquet delete, and the \
             supersede hazard, got: {}",
            err.message()
        );

        // Negative control: the x=0 parquet delete does not apply to B, so its DV commits.
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/b-dv.puffin",
            1,
            "test/b.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(&catalog)
            .await
            .expect("a DV in a partition the legacy parquet delete does not cover must commit");
    }

    // The `validateAddedDVs` op set, Java's `{Overwrite, Delete, Replace}`. `Replace` is the member
    // the generic added-delete-file op set lacks, because a compaction can rewrite DVs.

    /// A test-only action that commits a DV under `Operation::Replace`. No public Rust action adds
    /// delete files under `Replace` yet, so this hand-builds that window through the real producer.
    struct ReplaceOpAddDvAction {
        dv: DataFile,
    }

    struct ReplaceOpAddDvOperation;

    impl SnapshotProduceOperation for ReplaceOpAddDvOperation {
        fn operation(&self) -> Operation {
            Operation::Replace
        }

        async fn delete_entries(
            &self,
            _snapshot_produce: &SnapshotProducer<'_>,
        ) -> crate::Result<Vec<crate::spec::ManifestEntry>> {
            Ok(vec![])
        }

        async fn delete_files(
            &self,
            _snapshot_produce: &SnapshotProducer<'_>,
        ) -> crate::Result<Vec<DataFile>> {
            Ok(vec![])
        }

        async fn existing_manifest(
            &self,
            snapshot_produce: &SnapshotProducer<'_>,
        ) -> crate::Result<Vec<crate::spec::ManifestFile>> {
            snapshot_produce.current_manifests().await
        }
    }

    #[async_trait::async_trait]
    impl crate::transaction::TransactionAction for ReplaceOpAddDvAction {
        async fn commit(
            self: Arc<Self>,
            table: &Table,
        ) -> crate::Result<crate::transaction::ActionCommit> {
            SnapshotProducer::new(
                table,
                uuid::Uuid::now_v7(),
                None,
                HashMap::new(),
                vec![],
                FirstRowIdPolicy::Suppress,
            )
            .with_added_delete_files(vec![self.dv.clone()])
            .commit(ReplaceOpAddDvOperation, DefaultManifestProcess)
            .await
        }
    }

    /// A concurrent `Replace` snapshot adds a DV for the same data file, and `validateAddedDVs` must
    /// detect it. The mutant reuses the generic `{Overwrite, Delete}` op set and skips the snapshot.
    ///
    /// The assertion reads the walk's own message. The fresh-DV door would also reject this state,
    /// but with a different message, so pinning the message isolates the op set.
    #[tokio::test]
    async fn test_row_delta_dv_conflict_detected_from_concurrent_replace_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv.puffin",
                0,
                "test/a.parquet",
            )])
            .validate_from_snapshot(s0);
        let tx = action.apply(tx).unwrap();

        // The concurrent snapshot records `Replace` and adds a DV for the same A.
        let concurrent_tx = Transaction::new(&table);
        let concurrent_tx = ReplaceOpAddDvAction {
            dv: synthetic_dv_file("test/a-dv-replace.puffin", 0, "test/a.parquet"),
        }
        .apply(concurrent_tx)
        .unwrap();
        let concurrent = concurrent_tx.commit(&catalog).await.unwrap();
        assert_eq!(
            concurrent
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace,
            "fixture sanity: the concurrent snapshot records REPLACE"
        );

        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a concurrent REPLACE-op DV for the same referenced file must conflict");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message()
                .contains("Found concurrently added DV for test/a.parquet"),
            "the WALK (not the door) must catch the REPLACE-op DV — its message pins the op set, \
             got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/a-dv-replace.puffin"),
            "the message must carry Java's dvDesc of the concurrent DV, got: {}",
            err.message()
        );
    }

    // Removed against referenced, Java `validateNoConflictingFileAndPositionDeletes`. Always on.

    /// A row delta that removes a data file its added deletes reference is rejected with Java's
    /// message. The mutant commits both, which leaves a delete that applies to nothing.
    #[tokio::test]
    async fn test_row_delta_rejects_removing_data_file_referenced_by_added_deletes() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .validate_data_files_exist(["test/a.parquet"])
            .remove_rows(synthetic_data_file("test/a.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("removing a data file the added deletes reference must be rejected");

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert_eq!(
            err.message(),
            "Cannot delete data files [test/a.parquet] that are referenced by new delete files",
            "the message must match Java's (List rendering of the offending paths)"
        );
    }

    /// Disjoint removed and referenced sets commit. The row delta removes A and its deletes
    /// reference only B. The mutant rejects any combination of a removal and a reference.
    #[tokio::test]
    async fn test_row_delta_disjoint_removed_and_referenced_files_commit() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            synthetic_data_file("test/a.parquet", 0),
            synthetic_data_file("test/b.parquet", 0),
        ])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/b-pos-del.parquet", 0)])
            .validate_data_files_exist(["test/b.parquet"])
            .remove_rows(synthetic_data_file("test/a.parquet", 0));
        let tx = action.apply(tx).unwrap();
        tx.commit(&catalog)
            .await
            .expect("disjoint removed/referenced sets are not a conflict");
    }

    // The write-then-read round trip for a committed DV's metadata, and the DV summary counters.

    /// A committed DV's `referenced_data_file`, `content_offset`, `content_size_in_bytes`, and
    /// `record_count` survive the delete-manifest avro write, read back from the on-disk bytes. Other
    /// tests prove Rust READS Java-written manifests; this one pins the Rust writer's schema.
    ///
    /// The mutant drops the optional DV fields on write. The scan can then neither locate the DV blob
    /// nor key it to its data file, which corrupts the table for every engine.
    #[tokio::test]
    async fn test_committed_dv_metadata_survives_manifest_write_read_round_trip() {
        use crate::spec::Manifest;

        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Read the committed DELETE manifest back from its raw avro bytes.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let delete_manifest = manifest_list
            .entries()
            .iter()
            .find(|m| m.content == ManifestContentType::Deletes)
            .expect("the row delta committed a DELETE manifest");
        let bytes = table
            .file_io()
            .new_input(&delete_manifest.manifest_path)
            .unwrap()
            .read()
            .await
            .unwrap();
        let (_, entries) = Manifest::try_from_avro_bytes(&bytes).unwrap();

        assert_eq!(entries.len(), 1, "exactly the one added DV entry");
        let dv = entries[0].data_file();
        assert_eq!(dv.content_type(), DataContentType::PositionDeletes);
        assert_eq!(dv.file_format(), DataFileFormat::Puffin);
        assert_eq!(
            dv.referenced_data_file(),
            Some("test/a.parquet".to_string()),
            "referenced_data_file must survive the Rust manifest write→read round-trip"
        );
        assert_eq!(
            dv.content_offset(),
            Some(4),
            "content_offset must survive the round-trip (the scan's ranged blob read needs it)"
        );
        assert_eq!(
            dv.content_size_in_bytes(),
            Some(40),
            "content_size_in_bytes must survive the round-trip"
        );
        assert_eq!(
            dv.record_count(),
            1,
            "record_count (cardinality) must survive"
        );
    }

    // The all-Rust deletion-vector end-to-end: writer, commit, read path.

    /// The all-Rust DV chain on a real warehouse: append a real parquet file, write a real Puffin DV
    /// for positions 1 and 3, commit it, then scan and get exactly the survivors {10, 30, 50}.
    ///
    /// The mutant strips the DV from the commit, or breaks the gate, and y=20 and y=40 come back.
    ///
    /// It also pins the summary keys: a DV increments `added-dvs` instead of
    /// `added-position-delete-files`, and still counts in `added-delete-files` and
    /// `added-position-deletes`.
    #[tokio::test]
    async fn test_row_delta_deletion_vector_end_to_end_write_commit_scan() {
        use crate::spec::PartitionKey;
        use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let data_file = write_data_file(&table, "rows.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
            (0, 40, 400),
            (0, 50, 500),
        ])
        .await;
        let data_file_path = data_file.file_path().to_string();
        let table = append_files(&catalog, &table, vec![data_file]).await;

        let before: HashSet<i64> = scan_y_values(&table).await;
        assert_eq!(before, HashSet::from([10, 20, 30, 40, 50]));

        // Write the DV in the data file's partition context, so it carries the matching partition
        // and spec id.
        let partition_key = PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::long(0))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let dv_path = format!("{}/data/deletes-dv.puffin", table.metadata().location());
        let output_file = table.file_io().new_output(&dv_path).unwrap();
        let mut dv_writer = DVFileWriter::new(output_file).unpartitioned();
        dv_writer
            .delete(&data_file_path, 1, Some(&partition_key))
            .unwrap();
        dv_writer
            .delete(&data_file_path, 3, Some(&partition_key))
            .unwrap();
        let dv_files = dv_writer.close().await.unwrap();
        assert_eq!(dv_files.len(), 1, "one DV (one referenced data file)");
        assert_eq!(dv_files[0].file_format(), DataFileFormat::Puffin);
        assert_eq!(
            dv_files[0].referenced_data_file(),
            Some(data_file_path.clone())
        );
        assert_eq!(
            dv_files[0].record_count(),
            2,
            "cardinality = 2 deleted positions"
        );

        // The V3 gate passes for a Puffin position delete, and the fresh-DV door passes because
        // the file has no live delete.
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(dv_files);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // The summary must use `added-dvs` and not `added-position-delete-files`.
        assert_eq!(
            summary_prop(&table, "added-dvs").as_deref(),
            Some("1"),
            "a committed DV must emit added-dvs (the D4 interop canonical view compares it)"
        );
        assert_eq!(
            summary_prop(&table, "added-position-delete-files"),
            None,
            "a DV must NOT count as a position-delete FILE (Java's instead-of branch)"
        );
        assert_eq!(
            summary_prop(&table, "added-delete-files").as_deref(),
            Some("1"),
            "a DV still counts as an added delete file"
        );
        assert_eq!(
            summary_prop(&table, "added-position-deletes").as_deref(),
            Some("2"),
            "a DV's record count still counts as added position deletes"
        );

        let after: HashSet<i64> = scan_y_values(&table).await;
        assert_eq!(
            after,
            HashSet::from([10, 30, 50]),
            "the scan must return exactly the DV's survivors — resurrection of y=20/y=40 means \
             the commit path broke the write→read chain"
        );
    }

    // The fresh-DV door's applicability rule. It must fire exactly when a live parquet position
    // delete would apply to the DV's referenced data file at read time. Equality deletes never trip
    // it. A path-scoped delete trips it only for the same path. A partition-scoped delete trips it
    // only on the referenced data file's spec id and partition, read from its live manifest entry so
    // a partition evolution does not break it, and only when `delete_seq >= data_seq`.

    /// A live equality delete in the DV's partition must not trip the door. A DV supersedes position
    /// deletes only, so equality deletes coexist with it at read time.
    #[tokio::test]
    async fn test_row_delta_dv_commits_despite_live_equality_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_equality_delete_file("test/a-eq.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(&catalog)
            .await
            .expect("a live equality delete must not trip the fresh-DV door");
    }

    /// A live path-scoped position delete for a different file in the same partition must not trip
    /// the door. It holds only that file's positions, so the DV supersedes nothing.
    #[tokio::test]
    async fn test_row_delta_dv_commits_despite_path_scoped_delete_for_other_file() {
        use crate::spec::FormatVersion;

        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            synthetic_data_file("test/a.parquet", 0),
            synthetic_data_file("test/b.parquet", 0),
        ])
        .await;

        let path_scoped = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/b-pos-del.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .referenced_data_file(Some("test/b.parquet".to_string()))
            .build()
            .unwrap();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![path_scoped]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(&catalog)
            .await
            .expect("a path-scoped delete for a different file must not trip the door");
    }

    /// A legacy partition-scoped delete under the old spec still applies to its old-spec data file at
    /// read time, because the index matches on the DATA file's spec id, which a partition evolution
    /// does not change.
    ///
    /// The added DV must carry the NEW default spec. The mutant tests the shadow against the DV's own
    /// spec and partition, which can never match, so the DV commits and supersedes the legacy delete.
    /// The door must resolve the referenced file's live entry instead.
    #[tokio::test]
    async fn test_row_delta_dv_rejected_when_cross_spec_legacy_partition_delete_applies() {
        use crate::spec::FormatVersion;

        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // A partition-scoped parquet position delete under spec 0 that applies to A at read time.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/x0-pos-del.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Evolving the spec with identity(y) gives a new default spec id.
        let tx = Transaction::new(&table);
        let action = tx.update_partition_spec().add_field("y");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let new_spec_id = table.metadata().default_partition_spec_id();
        assert_ne!(new_spec_id, 0, "fixture sanity: the spec evolved");

        let tx = Transaction::new(&table);
        let action = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // The DV carries the new spec, because the producer requires the default spec id.
        let dv = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/a-dv.puffin".to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(new_spec_id)
            .partition(Struct::from_iter([
                Some(Literal::long(0)),
                Some(Literal::long(0)),
            ]))
            .referenced_data_file(Some("test/a.parquet".to_string()))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .build()
            .unwrap();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![dv]);
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "the legacy spec-0 partition delete still applies to A — the DV must be rejected \
             (silent supersede = resurrection)",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    /// A row delta adding a parquet position delete is built against a V2 table, then a concurrent
    /// upgrade to V3 lands before it commits. `do_commit` re-loads the table, so the gate must see V3
    /// and reject the now-illegal delete.
    ///
    /// This pins the gate's placement. Java 1.10.0 gates at add time and would accept this race. The
    /// commit-time placement here matches Java MAIN's apply-time re-validation.
    #[tokio::test]
    async fn test_row_delta_parquet_delete_rejected_after_concurrent_format_upgrade() {
        use crate::spec::FormatVersion;

        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)]);
        let tx = action.apply(tx).unwrap();

        // A concurrent commit from the same base upgrades the table to V3.
        let concurrent_tx = Transaction::new(&table);
        let concurrent_action = concurrent_tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3);
        let concurrent_tx = concurrent_action.apply(concurrent_tx).unwrap();
        concurrent_tx.commit(&catalog).await.unwrap();

        let err = tx
            .commit(&catalog)
            .await
            .expect_err("the gate must re-run against the refreshed V3 base");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            err.message(),
            "Must use DVs for position deletes in V3: test/a-pos-del.parquet",
            "the refreshed-base gate must reject with the V3 message"
        );
    }

    /// A legacy partition-scoped delete does not apply to a data file appended after it, because a
    /// position delete applies only when `delete_seq >= data_seq`. A DV for that newer file shadows
    /// nothing and must commit. The mutant fires on bare partition equality, which freezes every DV
    /// write into a partition that carries a legacy delete.
    #[tokio::test]
    async fn test_row_delta_dv_commits_when_legacy_delete_predates_data_file() {
        use crate::spec::FormatVersion;

        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // This partition-scoped delete applies to A only, because seq 2 is at or after seq 1.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/x0-pos-del.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Y lands in the same partition at a sequence number after the delete's.
        let tx = Transaction::new(&table);
        let action = tx
            .upgrade_table_version()
            .set_format_version(FormatVersion::V3);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/y.parquet",
            0,
        )])
        .await;

        // The legacy delete does not apply to Y, so its DV shadows nothing.
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/y-dv.puffin",
            0,
            "test/y.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(&catalog)
            .await
            .expect("the legacy delete does not apply to the newer file — the DV must commit");
    }

    // Apply-side delete-file removal (`RowDelta.removeDeletes`) and the fresh-DV door's escape hatch.

    /// Write a real Puffin deletion vector for `data_file_path` at `positions`, in partition
    /// `x = part_value`. It returns the single produced DV.
    async fn write_real_dv_file(
        table: &Table,
        file_name: &str,
        data_file_path: &str,
        part_value: i64,
        positions: &[u64],
    ) -> DataFile {
        use crate::spec::PartitionKey;
        use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

        let partition_key = PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let dv_path = format!("{}/data/{}", table.metadata().location(), file_name);
        let output_file = table.file_io().new_output(&dv_path).unwrap();
        let mut dv_writer = DVFileWriter::new(output_file).unpartitioned();
        for pos in positions {
            dv_writer
                .delete(data_file_path, *pos, Some(&partition_key))
                .unwrap();
        }
        dv_writer.close().await.unwrap().into_iter().next().unwrap()
    }

    /// Collect the live delete entries across the current snapshot's DELETE manifests as
    /// `(path, is_dv, referenced_data_file)`. Use it to assert the one-live-DV-per-file invariant.
    async fn live_delete_entries(table: &Table) -> Vec<(String, bool, Option<String>)> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut out = Vec::new();
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != ManifestContentType::Deletes {
                continue;
            }
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let file = entry.data_file();
                out.push((
                    file.file_path().to_string(),
                    is_deletion_vector(file),
                    file.referenced_data_file(),
                ));
            }
        }
        out
    }

    /// DV replaces DV, all-Rust end-to-end. DV#1 deletes position 1, then a hand-merged DV#2 for
    /// positions 1 and 3 is added while DV#1 is removed in the same commit. The scan then returns
    /// {10, 30, 50}.
    ///
    /// It also asserts the old DV is tombstoned with the new snapshot id, that the surviving DV keeps
    /// its own provenance, that the summary carries `removed-dvs: 1` and `added-dvs: 1`, and that
    /// exactly one live DV covers the data file.
    ///
    /// The mutants: a broken removal leaves two live DVs, which the scan's duplicate-DV door rejects
    /// later, or the wrong positions drop and rows resurrect. No other test proves the apply-side
    /// removal closes the merge-and-replace loop.
    #[tokio::test]
    async fn test_row_delta_dv_replaces_dv_end_to_end_remove_deletes() {
        use crate::spec::Manifest;

        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let data_file = write_data_file(&table, "rows.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
            (0, 40, 400),
            (0, 50, 500),
        ])
        .await;
        let data_file_path = data_file.file_path().to_string();
        let table = append_files(&catalog, &table, vec![data_file]).await;

        let dv1 = write_real_dv_file(&table, "dv1.puffin", &data_file_path, 0, &[1]).await;
        let dv1_path = dv1.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![dv1.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30, 40, 50]),
            "after DV#1 the scan drops y=20"
        );

        // DV#2 is the hand-merged super-set {1,3}.
        let dv2 = write_real_dv_file(&table, "dv2.puffin", &data_file_path, 0, &[1, 3]).await;
        let dv2_path = dv2.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![dv2.clone()])
            .remove_deletes(dv1.clone());
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let new_snapshot_id = table.metadata().current_snapshot().unwrap().snapshot_id();

        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30, 50]),
            "after the DV replace, the scan drops the super-set {{1,3}} (y=20, y=40)"
        );

        // Exactly one live DV covers the data file.
        let live = live_delete_entries(&table).await;
        let live_dvs_for_file: Vec<&(String, bool, Option<String>)> = live
            .iter()
            .filter(|(_, is_dv, referenced)| {
                *is_dv && referenced.as_deref() == Some(data_file_path.as_str())
            })
            .collect();
        assert_eq!(
            live_dvs_for_file.len(),
            1,
            "exactly ONE live DV for the data file post-commit — got {:?}",
            live
        );
        assert_eq!(
            live_dvs_for_file[0].0, dv2_path,
            "the surviving live DV is DV#2 (the merged super-set), not the removed DV#1"
        );

        assert_eq!(
            summary_prop(&table, "added-dvs").as_deref(),
            Some("1"),
            "the added super-set DV bumps added-dvs"
        );
        assert_eq!(
            summary_prop(&table, "removed-dvs").as_deref(),
            Some("1"),
            "the removed old DV bumps removed-dvs (the D3 branch, now reachable end-to-end)"
        );

        // Provenance, read from raw avro: DV#1's tombstone carries the NEW snapshot id, and a
        // surviving existing entry keeps its original one.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut found_dv1_tombstone = false;
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != ManifestContentType::Deletes {
                continue;
            }
            let bytes = table
                .file_io()
                .new_input(&manifest_file.manifest_path)
                .unwrap()
                .read()
                .await
                .unwrap();
            let (_, entries) = Manifest::try_from_avro_bytes(&bytes).unwrap();
            for entry in &entries {
                if entry.file_path() == dv1_path {
                    assert_eq!(
                        entry.status(),
                        ManifestStatus::Deleted,
                        "DV#1 must be a DELETED tombstone in the rewritten DELETE manifest"
                    );
                    assert_eq!(
                        entry.snapshot_id(),
                        Some(new_snapshot_id),
                        "the tombstone carries the NEW snapshot id (the rewrite's provenance stamp)"
                    );
                    found_dv1_tombstone = true;
                }
            }
        }
        assert!(
            found_dv1_tombstone,
            "DV#1 must appear as a tombstone in a rewritten DELETE manifest (it was not dropped \
             silently)"
        );
    }

    /// Load a committed DV's positions back through the production read path, the
    /// `CachingDeleteFileLoader`, and not a hand-built vector. This is what an engine's
    /// `loadPreviousDeletes` does.
    async fn load_dv_positions_via_production_loader(
        table: &Table,
        dv_file: &DataFile,
        referenced_data_file: &str,
    ) -> DeleteVector {
        let task = FileScanTaskDeleteFile {
            file_path: dv_file.file_path().to_string(),
            file_size_in_bytes: dv_file.file_size_in_bytes(),
            file_type: dv_file.content_type(),
            partition_spec_id: dv_file.partition_spec_id,
            equality_ids: None,
            file_format: dv_file.file_format(),
            referenced_data_file: dv_file.referenced_data_file(),
            content_offset: dv_file.content_offset(),
            content_size_in_bytes: dv_file.content_size_in_bytes(),
            record_count: Some(dv_file.record_count()),
        };
        let loader = CachingDeleteFileLoader::new(table.file_io().clone(), 4);
        let delete_filter = loader
            .load_deletes(
                std::slice::from_ref(&task),
                Arc::new(
                    crate::spec::Schema::builder()
                        .build()
                        .expect("empty schema"),
                ),
            )
            .await
            .expect("loader future")
            .expect("the production loader must load the committed DV");
        let vector = delete_filter
            .resolve_delete_vector(std::slice::from_ref(&task), referenced_data_file)
            .expect("a delete vector for the referenced data file");
        DeleteVector::new(vector.iter().collect())
    }

    /// Write a DV through the writer-side merge hook. The writer unions `previous_positions` into its
    /// new positions and returns the merged DVs with the superseded delete files.
    async fn write_merged_dv_file(
        table: &Table,
        file_name: &str,
        data_file_path: &str,
        part_value: i64,
        new_positions: &[u64],
        previous_positions: DeleteVector,
        previous_dv: DataFile,
    ) -> crate::writer::base_writer::deletion_vector_writer::DVWriteResult {
        use crate::spec::PartitionKey;
        use crate::writer::base_writer::deletion_vector_writer::{DVFileWriter, PreviousDeletes};

        let partition_key = PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let dv_path = format!("{}/data/{}", table.metadata().location(), file_name);
        let output_file = table.file_io().new_output(&dv_path).unwrap();
        let previous = PreviousDeletes::new(previous_positions, vec![previous_dv]);
        let mut dv_writer = DVFileWriter::new(output_file)
            .with_previous_deletes(HashMap::from([(data_file_path.to_string(), previous)]));
        for pos in new_positions {
            dv_writer
                .delete(data_file_path, *pos, Some(&partition_key))
                .unwrap();
        }
        dv_writer.close_with_result().await.unwrap()
    }

    /// The writer-side previous-deletes merge, all-Rust end-to-end, mirroring the engine flow that
    /// Spark's `SparkPositionDeltaWrite` runs.
    ///
    /// DV#1 deletes position 1. Its positions load back through the production loader and feed a new
    /// writer that adds only position 3. The WRITER, not the test, must produce the union {1, 3} and
    /// return DV#1 as rewritten. Committing the merged DV with that removal leaves one live DV and a
    /// scan of {10, 30, 50}.
    ///
    /// The mutants: a broken merge writes only {3} and y=20 resurrects; broken rewritten-file
    /// plumbing fails to remove DV#1, so the door rejects the commit or two DVs survive.
    #[tokio::test]
    async fn test_row_delta_dv_writer_merges_previous_deletes_end_to_end() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let data_file = write_data_file(&table, "rows.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
            (0, 40, 400),
            (0, 50, 500),
        ])
        .await;
        let data_file_path = data_file.file_path().to_string();
        let table = append_files(&catalog, &table, vec![data_file]).await;

        let dv1 = write_real_dv_file(&table, "dv1.puffin", &data_file_path, 0, &[1]).await;
        let dv1_path = dv1.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![dv1.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30, 40, 50]),
            "after DV#1 the scan drops y=20"
        );

        // Load DV#1's positions through the production loader, then let the writer merge them with
        // the new position 3.
        let previous_positions =
            load_dv_positions_via_production_loader(&table, &dv1, &data_file_path).await;
        assert_eq!(
            previous_positions.iter().collect::<Vec<_>>(),
            vec![1],
            "the production loader must read back DV#1's position {{1}}"
        );

        let merge_result = write_merged_dv_file(
            &table,
            "dv2.puffin",
            &data_file_path,
            0,
            &[3],
            previous_positions,
            dv1.clone(),
        )
        .await;

        // The writer produced one merged DV with the union cardinality, plus rewritten DV#1.
        assert_eq!(merge_result.delete_files.len(), 1);
        let dv2 = merge_result.delete_files[0].clone();
        let dv2_path = dv2.file_path().to_string();
        assert_eq!(
            dv2.record_count(),
            2,
            "the merged DV must carry the UNION {{1,3}} (cardinality 2), proving the WRITER merged"
        );
        assert_eq!(
            merge_result.rewritten_delete_files.len(),
            1,
            "DV#1 (file-scoped) must be returned as a rewritten/superseded delete file"
        );
        assert_eq!(
            merge_result.rewritten_delete_files[0].file_path(),
            dv1_path,
            "the rewritten file is DV#1"
        );

        // The escape hatch unlocks, because this same commit removes the live DV#1.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(merge_result.delete_files)
            .remove_deletes_many(merge_result.rewritten_delete_files);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // A broken merge that wrote only {3} would resurrect y=20 here.
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30, 50]),
            "after the writer-merged DV replace, the scan drops {{1,3}} (y=20 + y=40)"
        );

        // Exactly one live DV covers the data file, and it is the merged DV#2.
        let live = live_delete_entries(&table).await;
        let live_dvs_for_file: Vec<&(String, bool, Option<String>)> = live
            .iter()
            .filter(|(_, is_dv, referenced)| {
                *is_dv && referenced.as_deref() == Some(data_file_path.as_str())
            })
            .collect();
        assert_eq!(
            live_dvs_for_file.len(),
            1,
            "exactly ONE live DV for the data file post-commit — got {live:?}"
        );
        assert_eq!(
            live_dvs_for_file[0].0, dv2_path,
            "the surviving live DV is the writer-merged DV#2, not the removed DV#1"
        );

        assert_eq!(summary_prop(&table, "added-dvs").as_deref(), Some("1"));
        assert_eq!(summary_prop(&table, "removed-dvs").as_deref(), Some("1"));
    }

    /// Skip the removal and add DV#2 alone, and the fresh-DV door rejects the commit. That proves the
    /// door, and not the removal, is what stops the two-DV state. The read side enforces the same
    /// invariant at its load door; this is the fail-loud half, testable without mutating production
    /// code.
    #[tokio::test]
    async fn test_row_delta_second_dv_without_removal_still_rejected_by_door() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv1.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // DV#2 for A without removing DV#1, so no escape hatch engages.
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![synthetic_dv_file(
            "test/a-dv2.puffin",
            0,
            "test/a.parquet",
        )]);
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "without removing the live DV, a second DV for the same file must still be rejected",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Cannot commit deletion vector for test/a.parquet"),
            "got: {}",
            err.message()
        );
    }

    /// A DV#2 for A that removes the live DV#1 in the same commit succeeds, because the escape hatch
    /// matches on the removed path. `test_row_delta_second_dv_for_same_file_rejected_until_merge_lands`
    /// is the negative half.
    #[tokio::test]
    async fn test_row_delta_dv_with_removal_of_live_dv_commits() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let dv1 = synthetic_dv_file("test/a-dv1.puffin", 0, "test/a.parquet");
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![dv1.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv2.puffin",
                0,
                "test/a.parquet",
            )])
            .remove_deletes(dv1);
        let tx = action.apply(tx).unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("removing the live DV in the same commit must let the new DV through the door");

        let live = live_delete_entries(&table).await;
        let live_dv_paths: Vec<&String> = live
            .iter()
            .filter(|(_, is_dv, _)| *is_dv)
            .map(|(path, _, _)| path)
            .collect();
        assert_eq!(
            live_dv_paths,
            vec![&"test/a-dv2.puffin".to_string()],
            "exactly DV#2 is live; DV#1 is tombstoned — got {:?}",
            live
        );
        assert_eq!(summary_prop(&table, "removed-dvs").as_deref(), Some("1"));
        assert_eq!(summary_prop(&table, "added-dvs").as_deref(), Some("1"));
    }

    /// The escape hatch is per referenced file, not global. A and B each carry a live DV, and a row
    /// delta adds a new DV for A while removing B's DV. The door must still reject A's new DV, because
    /// the removed-set match is keyed on the existing delete's own path.
    ///
    /// The mutant keys the skip on `!removed_delete_paths.is_empty()` instead of
    /// `removed_delete_paths.contains(existing.file_path())`, which lets A's DV through and leaves two
    /// live DVs for A.
    #[tokio::test]
    async fn test_row_delta_remove_different_files_dv_does_not_unlock_door() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            synthetic_data_file("test/a.parquet", 0),
            synthetic_data_file("test/b.parquet", 0),
        ])
        .await;

        // One row delta commits both DVs. They reference different files, so neither trips the door.
        let a_dv1 = synthetic_dv_file("test/a-dv1.puffin", 0, "test/a.parquet");
        let b_dv1 = synthetic_dv_file("test/b-dv1.puffin", 0, "test/b.parquet");
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![a_dv1.clone(), b_dv1.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // A's live DV is not in the removed set, so the escape hatch does not engage for A.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv2.puffin",
                0,
                "test/a.parquet",
            )])
            .remove_deletes(b_dv1);
        let tx = action.apply(tx).unwrap();
        let err = tx.commit(&catalog).await.expect_err(
            "removing a DIFFERENT file's DV must NOT unlock the door for A's new DV (per-file, not global)",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Cannot commit deletion vector for test/a.parquet"),
            "got: {}",
            err.message()
        );
    }

    /// Remove a parquet position delete end-to-end on V2. A real position delete drops y=20 and y=40,
    /// then a removal-only row delta takes it away and the scan returns all five rows. The summary
    /// carries `removed-position-delete-files: 1`, and the operation is `Overwrite`, because the row
    /// delta adds nothing.
    ///
    /// The mutants: the removal does nothing and the rows stay missing, or the rewritten DELETE
    /// manifest is misclassified as DATA and the read path stops applying its survivors.
    #[tokio::test]
    async fn test_row_delta_remove_parquet_position_delete_restores_rows() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let data_file = write_data_file(&table, "rows.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
            (0, 40, 400),
            (0, 50, 500),
        ])
        .await;
        let data_file_path = data_file.file_path().to_string();
        let table = append_files(&catalog, &table, vec![data_file]).await;

        let delete_file = write_position_delete_file(&table, 0, &[
            (data_file_path.clone(), 1),
            (data_file_path.clone(), 3),
        ])
        .await;
        let delete_file_path = delete_file.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![delete_file.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30, 50]),
            "the position delete drops y=20, y=40"
        );

        let tx = Transaction::new(&table);
        let action = tx.row_delta().remove_deletes(delete_file);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 20, 30, 40, 50]),
            "removing the position delete restores all five rows"
        );
        assert_eq!(
            summary_prop(&table, "removed-position-delete-files").as_deref(),
            Some("1"),
            "a removed parquet position delete bumps removed-position-delete-files (NOT removed-dvs)"
        );
        assert_eq!(
            summary_prop(&table, "removed-dvs"),
            None,
            "a parquet position delete is not a DV"
        );
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Overwrite,
            "a remove-only row delta records Overwrite per 1.10.0 (adds no delete files)"
        );

        let live = live_delete_entries(&table).await;
        assert!(
            !live.iter().any(|(path, _, _)| *path == delete_file_path),
            "the removed position delete must not be live anymore — got {:?}",
            live
        );
    }

    /// Remove an equality delete at manifest level. The summary carries
    /// `removed-equality-delete-files: 1` and the entry is tombstoned. It exercises the equality branch
    /// of `remove_file` and the content-keyed DELETE-manifest rewrite.
    #[tokio::test]
    async fn test_row_delta_remove_equality_delete_bumps_removed_equality_counter() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let eq_delete = synthetic_equality_delete_file("test/a-eq.parquet", 0);
        let eq_path = eq_delete.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![eq_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx.row_delta().remove_deletes(eq_delete);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            summary_prop(&table, "removed-equality-delete-files").as_deref(),
            Some("1"),
            "a removed equality delete bumps removed-equality-delete-files"
        );
        let live = live_delete_entries(&table).await;
        assert!(
            !live.iter().any(|(path, _, _)| *path == eq_path),
            "the removed equality delete must be tombstoned — got {:?}",
            live
        );
    }

    /// Removing a delete file that is not live in the current snapshot fails loud with Java's
    /// `failMissingDeletePaths` message. The mutant makes the removal a silent no-op.
    #[tokio::test]
    async fn test_row_delta_remove_nonexistent_delete_file_errors_with_missing_message() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .remove_deletes(synthetic_delete_file("test/ghost-pos-del.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("removing a delete file that is not live must fail loud");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            err.message(),
            "Missing required files to delete: test/ghost-pos-del.parquet",
            "the missing-removal-path error must match Java failMissingDeletePaths shape"
        );
    }

    /// A row delta that only removes a delete file records `Operation::Overwrite`, because
    /// `addsDeleteFiles()` is false and the `Delete` branch does not fire. It also confirms the
    /// empty-commit guard counts a removal as content.
    #[tokio::test]
    async fn test_row_delta_remove_deletes_only_records_overwrite() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let pos_delete = synthetic_delete_file("test/a-pos-del.parquet", 0);
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![pos_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx.row_delta().remove_deletes(pos_delete);
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
            "a remove-deletes-only row delta records Overwrite per 1.10.0"
        );
    }

    /// `remove_deletes` rejects a `Data`-content file, because it is the delete-side surface and a data
    /// file belongs in `remove_rows`. The mutant routes a data file into the delete-removal path.
    #[tokio::test]
    async fn test_row_delta_remove_deletes_rejects_data_content() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .remove_deletes(synthetic_data_file("test/a.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("remove_deletes must reject a Data-content file");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Only position-delete or equality-delete content is allowed for removed delete files"),
            "got: {}",
            err.message()
        );
    }

    /// Provenance on the rewritten DELETE manifest. A manifest holds two live deletes, D1 and D2, and
    /// D1 is removed. D1 must become a tombstone with the new snapshot id, and D2 must survive as an
    /// existing entry with its original snapshot id and sequence number.
    ///
    /// The mutant routes the survivor through `add_entry`, which re-stamps it.
    #[tokio::test]
    async fn test_row_delta_remove_delete_preserves_surviving_entry_provenance() {
        use crate::spec::Manifest;

        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            synthetic_data_file("test/a.parquet", 0),
            synthetic_data_file("test/b.parquet", 0),
        ])
        .await;

        // Both position deletes go into one delete manifest.
        let d1 = synthetic_delete_file("test/d1-pos-del.parquet", 0);
        let d2 = synthetic_delete_file("test/d2-pos-del.parquet", 0);
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![d1.clone(), d2.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let add_snapshot_id = table.metadata().current_snapshot().unwrap().snapshot_id();
        let add_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        let tx = Transaction::new(&table);
        let action = tx.row_delta().remove_deletes(d1);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let remove_snapshot_id = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(add_snapshot_id, remove_snapshot_id);

        // Read the DELETE manifest this snapshot wrote, raw.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let rewritten = manifest_list
            .entries()
            .iter()
            .find(|m| {
                m.content == ManifestContentType::Deletes
                    && m.added_snapshot_id == remove_snapshot_id
            })
            .expect("the remove commit wrote a rewritten DELETE manifest");
        let bytes = table
            .file_io()
            .new_input(&rewritten.manifest_path)
            .unwrap()
            .read()
            .await
            .unwrap();
        let (_, entries) = Manifest::try_from_avro_bytes(&bytes).unwrap();

        let d1_entry = entries
            .iter()
            .find(|e| e.file_path() == "test/d1-pos-del.parquet")
            .expect("D1 is in the rewritten manifest as a tombstone");
        assert_eq!(d1_entry.status(), ManifestStatus::Deleted);
        assert_eq!(
            d1_entry.snapshot_id(),
            Some(remove_snapshot_id),
            "the D1 tombstone carries the NEW (remove) snapshot id"
        );

        let d2_entry = entries
            .iter()
            .find(|e| e.file_path() == "test/d2-pos-del.parquet")
            .expect("D2 survives in the rewritten manifest");
        assert_eq!(
            d2_entry.status(),
            ManifestStatus::Existing,
            "the surviving D2 is copied forward as Existing (provenance preserved)"
        );
        assert_eq!(
            d2_entry.snapshot_id(),
            Some(add_snapshot_id),
            "the surviving D2 keeps its ORIGINAL (add) snapshot id — re-stamping is the corruption \
             class this pins"
        );
        // `add_existing_entry` copies the post-inheritance sequence number from the source manifest,
        // so D2 never re-inherits the new snapshot's.
        assert_eq!(
            d2_entry.sequence_number(),
            Some(add_seq),
            "the surviving D2 keeps its original sequence number (no re-inheritance)"
        );
    }

    /// Cumulative totals across append, then a DV, then a DV replace. Each `total-*` key must be the
    /// previous value plus added minus removed. After the replace, `total-delete-files` stays 1.
    ///
    /// The mutant reseeds the totals per commit instead of carrying the previous summary forward. Only
    /// a multi-commit chain catches it.
    #[tokio::test]
    async fn test_row_delta_cumulative_totals_across_dv_replace() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let data_file = write_data_file(&table, "rows.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
            (0, 40, 400),
            (0, 50, 500),
        ])
        .await;
        let data_file_path = data_file.file_path().to_string();
        let table = append_files(&catalog, &table, vec![data_file]).await;

        let dv1 = write_real_dv_file(&table, "dv1.puffin", &data_file_path, 0, &[1]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![dv1.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            summary_prop(&table, "total-delete-files").as_deref(),
            Some("1"),
            "after DV#1: one delete file"
        );
        assert_eq!(
            summary_prop(&table, "total-position-deletes").as_deref(),
            Some("1"),
            "after DV#1: one position delete"
        );

        // DV#2 replaces DV#1, so total-delete-files stays 1 and total-position-deletes becomes 2.
        let dv2 = write_real_dv_file(&table, "dv2.puffin", &data_file_path, 0, &[1, 3]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![dv2]).remove_deletes(dv1);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            summary_prop(&table, "total-delete-files").as_deref(),
            Some("1"),
            "after the DV replace: total-delete-files = 1 + 1 − 1 = 1 (the seed-from-previous rule)"
        );
        assert_eq!(
            summary_prop(&table, "total-position-deletes").as_deref(),
            Some("2"),
            "after the DV replace: total-position-deletes = 1 + 2 − 1 = 2"
        );
    }

    /// The data-side filtering writer is unchanged by the content-keyed extension. An
    /// `overwrite_files().delete_files` commit still rewrites the DATA manifest with the removed file
    /// as a tombstone and the survivors existing.
    ///
    /// The mutant builds a DELETE writer for a DATA source, which misclassifies the rewritten manifest.
    #[tokio::test]
    async fn test_overwrite_delete_data_file_still_rewrites_data_manifest() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![
            synthetic_data_file("test/a.parquet", 0),
            synthetic_data_file("test/b.parquet", 0),
        ])
        .await;

        // `overwrite_files` is the data-side filter path.
        let tx = Transaction::new(&table);
        let action = tx.overwrite_files().delete_files(["test/a.parquet"]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut a_tombstoned = false;
        let mut b_live = false;
        for manifest_file in manifest_list.entries() {
            assert_eq!(
                manifest_file.content,
                ManifestContentType::Data,
                "no DELETE manifest should exist on this data-only table"
            );
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                match entry.file_path() {
                    "test/a.parquet" if entry.status() == ManifestStatus::Deleted => {
                        a_tombstoned = true
                    }
                    "test/b.parquet" if entry.is_alive() => b_live = true,
                    _ => {}
                }
            }
        }
        assert!(a_tombstoned, "A must be a DATA-manifest tombstone");
        assert!(b_live, "B must remain live in the DATA manifest");
    }

    // `caseSensitive(boolean)` on the conflict-detection-filter bind in
    // `validateNoConflictingDataFiles`. The schema column is `y`, so a filter on `Y` binds only when
    // case sensitivity is off. The observable outcome tells the flag values apart:
    //
    // | case | filter | concurrent file | outcome |
    // |---|---|---|---|
    // | default | `y >= 50` | matches | "conflicting files" |
    // | `false` | `Y >= 50` | matches | "conflicting files" |
    // | default | `Y >= 50` | no match | bind error, "Field Y not found" |
    //
    // A mutant that ignores the flag turns row 2 into a bind error. A mutant that hard-codes `false`
    // turns row 3 into a successful commit.

    /// The default with a correctly-cased filter `y >= 50` and a matching concurrent file rejects with
    /// the "conflicting files" message. This is the unchanged-behavior row of the table above.
    #[tokio::test]
    async fn test_row_delta_conflict_filter_default_case_sensitive_correct_case_rejects() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();

        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            60,
            70,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "row delta must fail: a concurrent file matches the correctly-cased conflict filter",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("conflicting files"),
            "the correctly-cased conflict filter must bind and detect the conflict, got: {}",
            err.message()
        );
    }

    /// `case_sensitive(false)` with the wrong-cased filter `Y >= 50` and a matching concurrent file
    /// binds case-insensitively and rejects with "conflicting files".
    ///
    /// The mutant ignores the flag, so `Y` fails to bind and the error reads "Field Y not found"
    /// instead. This test asserts the conflict message, so it discriminates that mutant.
    #[tokio::test]
    async fn test_row_delta_conflict_filter_case_insensitive_wrong_case_detects_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .case_sensitive(false)
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();

        let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
            "test/concurrent.parquet",
            0,
            60,
            70,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "case_sensitive(false) binds the wrong-cased Y and the conflict check must still fire",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("conflicting files"),
            "the wrong-cased Y must bind case-insensitively and detect the CONFLICT (not a bind error), \
             got: {}",
            err.message()
        );
    }

    /// The default with the wrong-cased filter `Y >= 50` and a non-matching concurrent file must error
    /// with "Field Y not found", because `Y` cannot bind case-sensitively.
    ///
    /// The mutant hard-codes `false`, so `Y` binds, the non-matching file is no conflict, and the commit
    /// succeeds. This test requires an error, so it discriminates that mutant.
    #[tokio::test]
    async fn test_row_delta_conflict_filter_default_case_sensitive_wrong_case_fails_to_bind() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).unwrap();

        // These y bounds do not match `y >= 50`, so a case-insensitive bind would find no conflict.
        // Only the failed case-sensitive bind can make this error.
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

        // The catalog head is still S0's append, so no DELETE manifest landed.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        let manifest_list = reloaded
            .metadata()
            .current_snapshot()
            .unwrap()
            .load_manifest_list(reloaded.file_io(), reloaded.metadata())
            .await
            .unwrap();
        assert!(
            !manifest_list
                .entries()
                .iter()
                .any(|m| m.content == ManifestContentType::Deletes),
            "the rejected row delta must not have committed a DELETE manifest"
        );
    }

    // `caseSensitive(boolean)` on the always-on `validateAddedDVs` filter bind. This is a SECOND
    // binding site, reached only when this row delta adds a DV, a concurrent commit added a DV
    // candidate, and a conflict filter is set. The tests above never reach it, so hard-coding this bind
    // to case-sensitive `true` fails none of them.
    //
    // The discriminator is the message: the default gives "Field Y not found", and
    // `case_sensitive(false)` gives "Found concurrently added DV for ...".

    /// `case_sensitive(false)` with the wrong-cased filter `Y >= 50` binds case-insensitively inside
    /// `validate_added_dvs`, so the DV check reaches the collision and gives the DV-conflict message.
    ///
    /// The mutant drops the flag on the DV path, binds case-sensitively, and surfaces a bind error.
    #[tokio::test]
    async fn test_row_delta_validate_added_dvs_case_insensitive_wrong_case_detects_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // The filter is wrong-cased, and `case_sensitive(false)` is what makes it bind.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv.puffin",
                0,
                "test/a.parquet",
            )])
            .case_sensitive(false)
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0);
        let tx = action.apply(tx).unwrap();

        // The concurrent DV references the same A, which is the collision.
        let _concurrent = commit_concurrent_dvs(&catalog, &table, vec![synthetic_dv_file(
            "test/a-dv-concurrent.puffin",
            0,
            "test/a.parquet",
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "case_sensitive(false) binds the wrong-cased Y in validate_added_dvs; the DV conflict must fire",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Found concurrently added DV for"),
            "the wrong-cased Y must bind case-insensitively and the DV-conflict (not a bind error) must \
             fire, got: {}",
            err.message()
        );
    }

    /// The default with the same wrong-cased filter fails to bind inside `validate_added_dvs`, so the
    /// commit errors with "Field Y not found" and not the DV-conflict message. The mutant hard-codes the
    /// bind to `false`, which surfaces the conflict instead.
    #[tokio::test]
    async fn test_row_delta_validate_added_dvs_default_case_sensitive_wrong_case_fails_to_bind() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // The default is case-sensitive, so the wrong-cased `Y` must fail to bind.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_dv_file(
                "test/a-dv.puffin",
                0,
                "test/a.parquet",
            )])
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0);
        let tx = action.apply(tx).unwrap();

        let _concurrent = commit_concurrent_dvs(&catalog, &table, vec![synthetic_dv_file(
            "test/a-dv-concurrent.puffin",
            0,
            "test/a.parquet",
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "a wrong-cased Y must NOT bind under the default in validate_added_dvs; the validate must error",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Field Y not found"),
            "the rejection must be a BIND failure on the wrong-cased Y, not the DV conflict, got: {}",
            err.message()
        );
    }

    // `caseSensitive(boolean)` on the shared `validate_no_new_deletes_for_data_files` bind. This is a
    // THIRD binding site, reached through the removed-data-files sub-check and also used by
    // `OverwriteFiles` and `RewriteFiles`. No test above reaches it, because their fixtures set no
    // removed data files, so hard-coding this bind to case-sensitive `true` fails none of them.
    //
    // The discriminator is the message: `case_sensitive(false)` gives "found new delete for replaced
    // data file", and the default gives "Field Y not found".

    /// `case_sensitive(false)` with the wrong-cased filter `Y >= 50`, removing A while a concurrent
    /// position delete applies to A, binds case-insensitively and rejects on the removed data file.
    ///
    /// The mutant drops the flag on this shared helper, binds case-sensitively, and surfaces a bind
    /// error.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_case_insensitive_wrong_case_detects_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        // The filter is wrong-cased, and `case_sensitive(false)` is what makes it bind.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .remove_data_files(vec![a])
            .case_sensitive(false)
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // This delete lands in A's partition after the start, so it applies to the removed A.
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![synthetic_delete_file(
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

    /// The default with the same wrong-cased filter fails to bind, so the commit errors with "Field Y
    /// not found" and not the removed-data-file conflict. The mutant hard-codes the bind to `false`,
    /// which surfaces the conflict instead.
    #[tokio::test]
    async fn test_row_delta_removed_data_files_default_case_sensitive_wrong_case_fails_to_bind() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let a = synthetic_data_file("test/a.parquet", 0);
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![a.clone()]).await;

        // The default is case-sensitive, so the wrong-cased `Y` must fail to bind.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)])
            .remove_data_files(vec![a])
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![synthetic_delete_file(
            "test/pos-del.parquet",
            0,
        )])
        .await;

        let err = tx.commit(&catalog).await.expect_err(
            "a wrong-cased Y must NOT bind under the default in the shared helper; the validate must error",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Field Y not found"),
            "the rejection must be a BIND failure on the wrong-cased Y, not the conflict, got: {}",
            err.message()
        );
    }

    // `caseSensitive(boolean)` on the 2b filter-based delete-conflict bind. With no removed data files
    // the 2a sub-check skips, and with no data-file check step 1 skips, so the 2b call is the FIRST and
    // only bind of the conflict filter. Every other 2b test uses a correctly-cased `y`, so hard-coding
    // this caller-side bind to case-sensitive `true` fails none of them.
    //
    // The discriminator is the message: `case_sensitive(false)` gives "conflicting delete files", and
    // the default gives "Field Y not found".

    /// `case_sensitive(false)` with the wrong-cased filter `Y >= 50` and no removed data files, so 2b is
    /// the first bind. A concurrent delete file whose `y` bounds match the filter then rejects with the
    /// delete-file conflict.
    ///
    /// The mutant drops the flag on this consumer, binds case-sensitively, and surfaces a bind error.
    #[tokio::test]
    async fn test_row_delta_delete_conflict_case_insensitive_wrong_case_detects_conflict() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // No `remove_data_files`, so the 2b check is the first and only bind of the conflict filter.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/my-del.parquet", 0)])
            .case_sensitive(false)
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        // S1 adds a DELETE file whose y bounds [60,70] overlap `y >= 50`.
        let _concurrent =
            commit_concurrent_deletes(&catalog, &table, vec![delete_file_with_y_bounds(
                "test/concurrent-del.parquet",
                0,
                60,
                70,
            )])
            .await;

        let err = tx.commit(&catalog).await.expect_err(
            "case_sensitive(false) binds the wrong-cased Y in the 2b check; the delete-file conflict must fire",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("conflicting delete files"),
            "the wrong-cased Y must bind case-insensitively and the DELETE-file conflict (not a bind error) \
             must fire, got: {}",
            err.message()
        );
        assert!(
            err.message().contains("test/concurrent-del.parquet"),
            "the error must name the conflicting DELETE file, got: {}",
            err.message()
        );
    }

    /// The default with the same wrong-cased filter fails the 2b bind, so the commit errors with "Field
    /// Y not found" and not the delete-file conflict. The mutant hard-codes the bind to `false`, which
    /// surfaces the conflict instead.
    #[tokio::test]
    async fn test_row_delta_delete_conflict_default_case_sensitive_wrong_case_fails_to_bind() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, s0) = append_and_snapshot_id(&catalog, &table, vec![synthetic_data_file(
            "test/a.parquet",
            0,
        )])
        .await;

        // The default is case-sensitive, so the wrong-cased `Y` must fail to bind.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_delete_file("test/my-del.parquet", 0)])
            .conflict_detection_filter(
                Reference::new("Y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_from_snapshot(s0)
            .validate_no_conflicting_delete_files();
        let tx = action.apply(tx).unwrap();

        let _concurrent =
            commit_concurrent_deletes(&catalog, &table, vec![delete_file_with_y_bounds(
                "test/concurrent-del.parquet",
                0,
                60,
                70,
            )])
            .await;

        let err = tx.commit(&catalog).await.expect_err(
            "a wrong-cased Y must NOT bind under the default in the 2b check; the validate must error",
        );
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Field Y not found"),
            "the rejection must be a BIND failure on the wrong-cased Y, not the conflict, got: {}",
            err.message()
        );
    }

    mod row_delta_extracted;
}
