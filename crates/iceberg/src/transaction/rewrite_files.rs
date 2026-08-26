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

//! The rewrite-files action: the compaction-commit primitive.
//!
//! [`RewriteFilesAction`] replaces a set of data and DELETE files with a new set.
//! It does this in one `Replace` snapshot (Java `BaseRewriteFiles`).
//! The added files reach the producer as fast-append does.
//! The files to delete are resolved BY PATH against the current snapshot's manifests.
//!
//! The operation is always [`Operation::Replace`]. A rewrite does not change the row set.
//!
//! Every file to delete must be live in the current snapshot.
//! An absent path errors (Java `failMissingDeletePaths`).
//!
//! Three preconditions (Java `BaseRewriteFiles.validateReplacedAndAddedFiles()`):
//!   1. The to-delete set (data OR delete files) must be non-empty ("Files to delete cannot be
//!      empty"). A delete-only rewrite is legal. An add-only rewrite is rejected.
//!   2. Data files may be added only if data files are deleted.
//!   3. Delete files may be added only if delete files are deleted.
//!
//! **Data sequence number** (Java `RewriteFiles.dataSequenceNumber`). By default the added files take a
//! fresh, higher data seq. An equality delete applies only to data with a strictly lower data seq, so a
//! fresh seq makes outstanding equality deletes stop applying and resurrects deleted rows.
//! [`RewriteFilesAction::data_sequence_number`] stamps the replaced files' seq instead. Java has no guard
//! here either. Preserving the seq is the caller's job when the table carries deletes.
//!
//! A negative explicit seq is rejected at commit with [`ErrorKind::DataInvalid`]. This is a Rust-only
//! fail-loud addition. The manifest writer strips a negative seq back into re-inheritance, which is the
//! exact resurrection corruption this path prevents.
//!
//! **Concurrent-commit conflict validation** (Java `BaseRewriteFiles.validate`).
//! [`RewriteFilesAction::validate`] runs against the refreshed base. It rejects the commit if a
//! concurrent commit added a row-level DELETE file that applies to a replaced data file. It passes
//! `ignore_equality_deletes = self.data_sequence_number.is_some()`: a preserved seq keeps a concurrent
//! equality delete applying, so that is no conflict. A new POSITION delete is always fatal, because its
//! path target dies with the replaced file. A conflict is a non-retryable [`ErrorKind::DataInvalid`].
//!
//! **DELETE-file surfaces.** [`RewriteFilesAction::delete_delete_file`] removes position-delete,
//! equality-delete, and deletion-vector files (Java `RewriteFiles.deleteFile(DeleteFile)`). This is the
//! commit vehicle `RemoveDanglingDeleteFiles` drives. [`RewriteFilesAction::add_delete_file`] and
//! [`RewriteFilesAction::add_delete_file_with_sequence_number`] add them (Java `addFile(DeleteFile)` and
//! `addFile(DeleteFile, long)`). Both route through the producer machinery `RowDelta` uses.
//!
//! A delete file rewritten into a new file must keep its original data seq. An inherited, higher seq
//! makes it stop applying, so deleted rows resurrect. A negative explicit seq is rejected at commit for
//! the same reason as the data-file guard above.
//!
//! Surviving entries are copied forward as `Existing`. They keep their original snapshot id and both
//! sequence numbers.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{DataFile, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, FirstRowIdPolicy, PendingDeleteFile, SnapshotProduceOperation,
    SnapshotProducer, validate_no_new_deletes_for_data_files,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// A transaction action that rewrites files. It removes a set of data and DELETE files and adds a new
/// set, in one `Replace` snapshot.
///
/// Use [`crate::transaction::Transaction::rewrite_files`] to create one. The primary entry point is
/// [`RewriteFilesAction::rewrite_files`]. The `*_file` / `*_files` builders are the incremental
/// equivalents. [`RewriteFilesAction::rewrite_files_with_deletes`] is the 4-set form. The files to delete
/// are passed as [`DataFile`]s and resolved against the current snapshot BY PATH.
///
/// The set of files to delete must be non-empty (Java `BaseRewriteFiles`). A delete-only rewrite is
/// legal. An add-only rewrite is rejected. Deleting a file that is not live in the current snapshot
/// errors (Java `failMissingDeletePaths`).
///
/// To keep outstanding merge-on-read EQUALITY deletes applying, call
/// [`RewriteFilesAction::data_sequence_number`] with the max data seq of the replaced files. Without it
/// the added files take a fresh, higher seq and the old equality deletes stop applying. Java has no guard
/// against this either. [`RewriteFilesAction::validate`] rejects conflicting concurrent row-level
/// deletes (Java `validateNoNewDeletesForDataFiles`).
pub struct RewriteFilesAction {
    /// Data files to add to the table (validated like fast append).
    added_data_files: Vec<DataFile>,
    /// Data files to remove from the table (their paths are resolved against the current snapshot).
    deleted_data_files: Vec<DataFile>,
    /// DELETE files to remove from the table (Java `RewriteFiles.deleteFile(DeleteFile)`). Their paths are
    /// resolved against the current snapshot's DELETE manifests and tombstoned in one snapshot. This is
    /// the commit vehicle `RemoveDanglingDeleteFiles` drives.
    deleted_delete_files: Vec<DataFile>,
    /// DELETE files to ADD to the table (Java `RewriteFiles.addFile(DeleteFile)` and
    /// `addFile(DeleteFile, long)`). Each carries an optional explicit DATA sequence number: `None`
    /// inherits the new snapshot's seq, `Some(seq)` stamps that seq. The right seq is the
    /// silent-corruption line. A rewritten delete file with the wrong seq stops applying, so rows
    /// resurrect, or it over-applies.
    added_delete_files: Vec<PendingDeleteFile>,
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    /// An explicit DATA sequence number for the added files (Java `RewriteFiles.dataSequenceNumber`).
    /// `Some(seq)` keeps the replaced files' seq, so outstanding equality deletes still apply. `None`
    /// gives the added files a fresh, higher seq. A negative value is rejected at commit.
    data_sequence_number: Option<i64>,
    /// An explicit starting snapshot for concurrent-commit conflict validation (Java
    /// `RewriteFiles.validateFromSnapshot`). `None` uses the transaction's starting snapshot.
    validate_from_snapshot: Option<i64>,
}

impl RewriteFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            added_data_files: vec![],
            deleted_data_files: vec![],
            deleted_delete_files: vec![],
            added_delete_files: vec![],
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            data_sequence_number: None,
            validate_from_snapshot: None,
        }
    }

    /// Rewrite `files_to_delete` into `files_to_add` (Java `RewriteFiles.rewriteFiles`). This is the
    /// primary entry point.
    pub fn rewrite_files(
        mut self,
        files_to_delete: impl IntoIterator<Item = DataFile>,
        files_to_add: impl IntoIterator<Item = DataFile>,
    ) -> Self {
        self.deleted_data_files.extend(files_to_delete);
        self.added_data_files.extend(files_to_add);
        self
    }

    /// Add a single rewritten [`DataFile`] to the table (Java `RewriteFiles.addFile`).
    pub fn add_file(mut self, data_file: DataFile) -> Self {
        self.added_data_files.push(data_file);
        self
    }

    /// Add multiple rewritten [`DataFile`]s to the table.
    pub fn add_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Remove a single rewritten [`DataFile`] from the table (Java `RewriteFiles.deleteFile`). Its path
    /// must equal a live file path in the current snapshot, or the commit errors.
    pub fn delete_file(mut self, data_file: DataFile) -> Self {
        self.deleted_data_files.push(data_file);
        self
    }

    /// Remove multiple rewritten [`DataFile`]s from the table.
    pub fn delete_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.deleted_data_files.extend(data_files);
        self
    }

    /// Remove a single DELETE file from the table (Java `RewriteFiles.deleteFile(DeleteFile)`). Its path
    /// must equal a live DELETE-file path in the current snapshot, or the commit errors
    /// (`failMissingDeletePaths`). The file must be `PositionDeletes` or `EqualityDeletes` content. A
    /// `Data` file is rejected at commit; use [`Self::delete_file`] for one.
    pub fn delete_delete_file(mut self, delete_file: DataFile) -> Self {
        self.deleted_delete_files.push(delete_file);
        self
    }

    /// Remove multiple DELETE files from the table. Each file must be `PositionDeletes` or
    /// `EqualityDeletes` content.
    pub fn delete_delete_files(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.deleted_delete_files.extend(delete_files);
        self
    }

    /// Add a rewritten DELETE file with the DEFAULT inherited sequence number (Java
    /// `RewriteFiles.addFile(DeleteFile)`). The added file inherits the new snapshot's seq, so it applies
    /// to data at a strictly lower data seq. The file must be `PositionDeletes` or `EqualityDeletes`
    /// content. A `Data` file is rejected at commit; use [`Self::add_file`] for one.
    ///
    /// To rewrite an EXISTING delete file that must keep applying to the same data, use
    /// [`Self::add_delete_file_with_sequence_number`] with the original file's data seq. An inherited,
    /// higher seq makes the rewritten delete stop applying and resurrects deleted rows.
    pub fn add_delete_file(mut self, delete_file: DataFile) -> Self {
        self.added_delete_files.push((delete_file, None));
        self
    }

    /// Add multiple rewritten DELETE files with the DEFAULT inherited sequence number. Each must be
    /// `PositionDeletes` or `EqualityDeletes` content.
    pub fn add_delete_files(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_delete_files
            .extend(delete_files.into_iter().map(|file| (file, None)));
        self
    }

    /// Add a rewritten DELETE file with an EXPLICIT data sequence number (Java
    /// `RewriteFiles.addFile(DeleteFile, long)`). The file is stamped with `sequence_number` instead of
    /// inheriting the new snapshot's seq, so a delete rewritten into a new file keeps applying to exactly
    /// the data it applied to before. A higher seq resurrects deleted rows. A lower seq over-applies.
    ///
    /// `sequence_number` must be non-negative. A negative value is rejected at commit with
    /// [`ErrorKind::DataInvalid`]. The manifest writer strips a negative seq back into re-inheritance,
    /// which is the corruption this overload prevents.
    pub fn add_delete_file_with_sequence_number(
        mut self,
        delete_file: DataFile,
        sequence_number: i64,
    ) -> Self {
        self.added_delete_files
            .push((delete_file, Some(sequence_number)));
        self
    }

    /// The 4-set rewrite (Java's 4-arg `RewriteFiles.rewriteFiles`). It replaces data AND delete files and
    /// adds data AND delete files, all in ONE `Replace` snapshot. The added DELETE files take the DEFAULT
    /// inherited seq, because Java's 4-arg form routes through `addFile(DeleteFile)`.
    ///
    /// All three `validateReplacedAndAddedFiles()` preconditions apply at commit. The third one means
    /// adding delete files requires deleting delete files.
    pub fn rewrite_files_with_deletes(
        mut self,
        data_to_replace: impl IntoIterator<Item = DataFile>,
        delete_to_replace: impl IntoIterator<Item = DataFile>,
        data_to_add: impl IntoIterator<Item = DataFile>,
        delete_to_add: impl IntoIterator<Item = DataFile>,
    ) -> Self {
        self.deleted_data_files.extend(data_to_replace);
        self.deleted_delete_files.extend(delete_to_replace);
        self.added_data_files.extend(data_to_add);
        self.added_delete_files
            .extend(delete_to_add.into_iter().map(|file| (file, None)));
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

    /// Keep a DATA sequence number on the added files (Java `RewriteFiles.dataSequenceNumber(long)`).
    /// Pass the max data seq of the files being replaced, so any outstanding merge-on-read EQUALITY
    /// delete still applies to the rewritten data. Without this the added files take a fresh, higher seq,
    /// the old equality deletes stop applying, and deleted rows resurrect.
    ///
    /// `sequence_number` must be non-negative. A negative value is rejected at commit with
    /// [`ErrorKind::DataInvalid`]. The manifest writer strips a negative seq back into re-inheritance,
    /// which is the corruption this prevents.
    pub fn data_sequence_number(mut self, sequence_number: i64) -> Self {
        self.data_sequence_number = Some(sequence_number);
        self
    }

    /// Override the snapshot where concurrent-commit conflict validation starts (Java
    /// `RewriteFiles.validateFromSnapshot(long)`). By default the validation uses the transaction's
    /// starting snapshot. This pins an earlier snapshot id instead.
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.validate_from_snapshot = Some(snapshot_id);
        self
    }

    /// The set of paths to delete, derived from the to-delete [`DataFile`]s.
    fn delete_paths(&self) -> HashSet<String> {
        self.deleted_data_files
            .iter()
            .map(|file| file.file_path.clone())
            .collect()
    }
}

#[async_trait]
impl TransactionAction for RewriteFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // Java `BaseRewriteFiles.validateReplacedAndAddedFiles()`. It runs before the producer's own
        // machinery, so the exact Java message reaches the caller.
        let deletes_data_files = !self.deleted_data_files.is_empty();
        let deletes_delete_files = !self.deleted_delete_files.is_empty();
        let adds_data_files = !self.added_data_files.is_empty();
        let adds_delete_files = !self.added_delete_files.is_empty();

        // Precondition (1): the to-delete set must be non-empty. A delete-file-only rewrite is legal.
        // An add-only or empty rewrite is rejected.
        if !deletes_data_files && !deletes_delete_files {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Files to delete cannot be empty",
            ));
        }

        // Precondition (2): data files may be added only when data files are deleted. This rejects a
        // delete-file-only rewrite that also adds data files.
        if !deletes_data_files && adds_data_files {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Data files to add must be empty because there's no data file to be rewritten",
            ));
        }

        // Precondition (3): delete files may be ADDED only when delete files are deleted.
        if !deletes_delete_files && adds_delete_files {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Delete files to add must be empty because there's no delete file to be rewritten",
            ));
        }

        // Content-type guard on the removed DELETE set. Java `RewriteFiles.deleteFile(DeleteFile)` takes a
        // DeleteFile, so a Data file must go through `delete_file`.
        for delete_file in &self.deleted_delete_files {
            if delete_file.content_type() == crate::spec::DataContentType::Data {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Only position-delete or equality-delete content is allowed for removed delete files (use delete_file to remove data files)",
                ));
            }
        }

        // Content-type and negative-seq guards on the ADDED DELETE set. The format-version gate runs
        // later, in the producer's `validate_added_delete_files`, against the refreshed base.
        for (delete_file, explicit_seq) in &self.added_delete_files {
            if delete_file.content_type() == crate::spec::DataContentType::Data {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Only position-delete or equality-delete content is allowed for added delete files (use add_file to add data files)",
                ));
            }
            // The manifest writer strips a negative explicit seq back into re-inheritance. The added
            // delete then takes the new, higher snapshot seq, stops applying to its older data, and
            // resurrects rows. Fail loudly instead.
            if let Some(sequence_number) = explicit_seq
                && *sequence_number < 0
            {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Invalid data sequence number for added delete file: {sequence_number} (must be \
                         non-negative; a negative value would be stripped into sequence-number \
                         re-inheritance and resurrect deleted rows)"
                    ),
                ));
            }
        }

        // Fail loudly on a negative `dataSequenceNumber`. This is a Rust-only addition, not a Java mirror.
        // The manifest writer's `add_entry` strips a negative seq back to `None`, which re-inherits the
        // new, higher snapshot seq at read time and resurrects deleted rows.
        if let Some(sequence_number) = self.data_sequence_number
            && sequence_number < 0
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid data sequence number for rewrite: {sequence_number} (must be non-negative; \
                     a negative value would be stripped into sequence-number re-inheritance and resurrect \
                     deleted rows)"
                ),
            ));
        }

        let mut snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
            FirstRowIdPolicy::Suppress,
        );

        // Keep the replaced files' data seq on the added files when the caller asks for it. `None` leaves
        // the added files inheriting the new snapshot's seq.
        if let Some(sequence_number) = self.data_sequence_number {
            snapshot_producer =
                snapshot_producer.with_new_data_files_data_sequence_number(sequence_number);
        }

        // Route the removed DELETE files through the producer path `RowDelta.removeDeletes` uses. At
        // commit the producer resolves each path against the current DELETE manifests, fails loudly on a
        // missing path, tombstones the file, and bumps the matching summary counter.
        if deletes_delete_files {
            snapshot_producer =
                snapshot_producer.with_removed_delete_files(self.deleted_delete_files.clone());
        }

        // Route the ADDED DELETE files through the producer machinery `RowDelta` uses. Each carries its
        // optional explicit data seq. The producer writes them into a per-spec DELETE manifest beside the
        // rewritten manifests, in the same snapshot.
        if adds_delete_files {
            snapshot_producer =
                snapshot_producer.with_added_delete_files_with_seq(self.added_delete_files.clone());
        }

        // Validate the added files like fast append: content type, partition-spec match, and
        // partition-value compatibility. The producer's commit resolves the delete paths and errors on an
        // absent one (Java `failMissingDeletePaths`).
        snapshot_producer.validate_added_data_files()?;
        if adds_delete_files {
            snapshot_producer.validate_added_delete_files()?;
        }

        snapshot_producer
            .commit(
                RewriteFilesOperation {
                    delete_paths: self.delete_paths(),
                },
                DefaultManifestProcess,
            )
            .await
    }

    /// Concurrent-commit conflict validation (Java `BaseRewriteFiles.validate`). It runs against the
    /// refreshed base before this action's updates are re-applied. When the rewrite replaces data files,
    /// it rejects the commit if a concurrent commit added a row-level DELETE file that applies to one of
    /// them. This is unconditional. Java has no opt-in flag here.
    ///
    /// It delegates to the shared [`validate_no_new_deletes_for_data_files`] helper with
    /// `ignore_equality_deletes = self.data_sequence_number.is_some()`. A preserved seq keeps a concurrent
    /// equality delete applying, so only a new POSITION delete is fatal. Without a preserved seq, any
    /// applicable delete is fatal. A conflict is a non-retryable [`ErrorKind::DataInvalid`], so the commit
    /// retry loop stops.
    ///
    /// The starting snapshot is [`Self::validate_from_snapshot`] when set, else the transaction-captured
    /// `starting_snapshot_id`. The transaction-captured base is the head when `Transaction::new` ran, NOT
    /// the refreshed head. Re-reading the refreshed head makes the concurrent set empty and passes
    /// silently. The walk is a no-op on a V1 table and when there is no current snapshot.
    async fn validate(
        self: Arc<Self>,
        starting_snapshot_id: Option<i64>,
        current: &Table,
    ) -> Result<()> {
        // Validate only when this rewrite replaces data files, as Java does.
        if self.deleted_data_files.is_empty() {
            return Ok(());
        }

        // `starting_snapshot_id` is the transaction-captured base, not the refreshed head.
        let effective_start = self.validate_from_snapshot.or(starting_snapshot_id);

        // Java `validateNoNewDeletesForDataFiles`. A preserved seq ignores equality deletes, because they
        // still apply. Java passes no data filter here.
        validate_no_new_deletes_for_data_files(
            current,
            effective_start,
            None,
            &self.deleted_data_files,
            self.data_sequence_number.is_some(),
        )
        .await
    }
}

/// The [`SnapshotProduceOperation`] for [`RewriteFilesAction`].
///
/// It records [`Operation::Replace`], exposes every current manifest as the set to filter, and resolves
/// the requested delete paths against the current snapshot's live entries. The added files reach the
/// producer separately, so one snapshot carries both the added manifest and the rewritten manifests.
struct RewriteFilesOperation {
    delete_paths: HashSet<String>,
}

impl SnapshotProduceOperation for RewriteFilesOperation {
    fn operation(&self) -> Operation {
        // Java `BaseRewriteFiles.operation()` returns `DataOperations.REPLACE`.
        Operation::Replace
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn delete_files(&self, snapshot_produce: &SnapshotProducer<'_>) -> Result<Vec<DataFile>> {
        // Every requested path must match a live entry (Java `failMissingDeletePaths`).
        snapshot_produce
            .resolve_delete_paths(&self.delete_paths)
            .await
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        // Expose EVERY current manifest, DATA and DELETE. DELETE manifests carry forward unchanged. A
        // rewrite that dropped them would lose every outstanding merge-on-read delete and resurrect
        // deleted rows.
        snapshot_produce.current_manifests().await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use futures::TryStreamExt;

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Manifest,
        ManifestStatus, Operation, Struct,
    };
    use crate::table::Table;
    use crate::transaction::tests::{
        make_v2_minimal_table_in_catalog, make_v3_minimal_table_in_catalog,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::equality_delete_writer::{
        EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
    };
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

    /// Assert that `path` appears as a `Deleted` tombstone in the table's current snapshot.
    async fn assert_deleted_tombstone(table: &Table, path: &str) {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.file_path() == path && entry.status() == ManifestStatus::Deleted {
                    return;
                }
            }
        }
        panic!("{path} must appear as a Deleted tombstone");
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

    /// Append A, B, C, then rewrite `delete=[A, B] add=[D]`. The post-commit scan live set must be
    /// exactly {C, D}. A wrong live set is silent data corruption.
    #[tokio::test]
    async fn test_rewrite_delete_two_add_one_yields_correct_live_scan_set() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let a = data_file("test/a.parquet", 0);
        let b = data_file("test/b.parquet", 0);
        let c = data_file("test/c.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone(), b.clone(), c.clone()]).await;
        let s_append = table.metadata().current_snapshot().unwrap().snapshot_id();
        let (_, c_seq, c_fseq) = entry_provenance(&table, "test/c.parquet").await;

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![a, b], vec![data_file("test/d.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace
        );
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/c.parquet".to_string(), "test/d.parquet".to_string()])
        );

        assert_deleted_tombstone(&table, "test/a.parquet").await;
        assert_deleted_tombstone(&table, "test/b.parquet").await;

        // C keeps its ORIGINAL provenance. The rewrite must not re-stamp it.
        let (c_snap2, c_seq2, c_fseq2) = entry_provenance(&table, "test/c.parquet").await;
        assert_eq!(
            c_snap2,
            Some(s_append),
            "surviving C keeps its original snapshot id, not the rewrite snapshot"
        );
        assert_eq!(c_seq2, c_seq, "surviving C keeps its original data seq");
        assert_eq!(c_fseq2, c_fseq, "surviving C keeps its original file seq");
    }

    /// A rewrite must drop files that live in DIFFERENT source manifests and keep the rest. A is in its
    /// own manifest, B and C in another. The resolver must reach both.
    #[tokio::test]
    async fn test_rewrite_across_multiple_manifests() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let a = data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone()]).await;
        let b = data_file("test/b.parquet", 0);
        let c = data_file("test/c.parquet", 0);
        let table = append_files(&catalog, &table, vec![b.clone(), c.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![a, b], vec![data_file("test/d.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace
        );
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/c.parquet".to_string(), "test/d.parquet".to_string()])
        );
        assert_deleted_tombstone(&table, "test/a.parquet").await;
        assert_deleted_tombstone(&table, "test/b.parquet").await;
    }

    /// The canonical compaction shape: 3 files into 1. The live set must be exactly {big}. Keeping a
    /// compacted-away file, or dropping a file it should keep, is data corruption.
    #[tokio::test]
    async fn test_rewrite_compaction_to_fewer_files() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let a = data_file("test/a.parquet", 0);
        let b = data_file("test/b.parquet", 0);
        let c = data_file("test/c.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone(), b.clone(), c.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![a, b, c], vec![data_file("test/big.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/big.parquet".to_string()]),
            "compaction must replace the 3 small files with the single big file"
        );
        assert_deleted_tombstone(&table, "test/a.parquet").await;
        assert_deleted_tombstone(&table, "test/b.parquet").await;
        assert_deleted_tombstone(&table, "test/c.parquet").await;
    }

    /// Deleting a file that is NOT in the current snapshot must error (Java `failMissingDeletePaths`) and
    /// must not add the added file. A silent drop keeps the add and loses the removal.
    #[tokio::test]
    async fn test_rewrite_delete_absent_file_errors() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![data_file("test/does-not-exist.parquet", 0)], vec![
            data_file("test/b.parquet", 0),
        ]);
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

        // The failed rewrite did not add b.parquet.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from(["test/a.parquet".to_string()])
        );
    }

    /// An empty rewrite must be rejected, not committed as a no-op Replace snapshot.
    #[tokio::test]
    async fn test_rewrite_empty_delete_set_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![], vec![]);
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("an empty-delete rewrite must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("Files to delete cannot be empty"),
            "unexpected error message: {}",
            error.message()
        );
    }

    /// An add-only rewrite must be rejected. Otherwise it behaves like an append and corrupts the live
    /// set. Precondition (1) fires first.
    #[tokio::test]
    async fn test_rewrite_add_without_delete_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![], vec![data_file("test/b.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("an add-only rewrite must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("Files to delete cannot be empty"),
            "unexpected error message: {}",
            error.message()
        );

        // The rejected rewrite did not add b.parquet.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            live_file_paths(&reloaded).await,
            HashSet::from(["test/a.parquet".to_string()])
        );
    }

    /// A delete-only rewrite is LEGAL, because only the DELETE set must be non-empty. The guard must not
    /// reject it.
    #[tokio::test]
    async fn test_rewrite_delete_only_is_allowed() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let b = data_file("test/b.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone(), b]).await;

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![a], vec![]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace,
            "a delete-only rewrite still records Replace"
        );
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/b.parquet".to_string()])
        );
        assert_deleted_tombstone(&table, "test/a.parquet").await;
    }

    /// The summary must report BOTH `added-*` and `deleted-*`, because the producer merges the two
    /// summaries.
    #[tokio::test]
    async fn test_rewrite_summary_reflects_added_and_deleted_counts() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let b = data_file("test/b.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone(), b.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![a, b], vec![data_file("test/d.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            summary_prop(&table, "added-data-files").as_deref(),
            Some("1"),
            "summary must report one added data file"
        );
        assert_eq!(
            summary_prop(&table, "added-records").as_deref(),
            Some("1"),
            "summary must report one added record"
        );
        assert_eq!(
            summary_prop(&table, "deleted-data-files").as_deref(),
            Some("2"),
            "summary must report two deleted data files"
        );
        assert_eq!(
            summary_prop(&table, "deleted-records").as_deref(),
            Some("2"),
            "summary must report two deleted records"
        );
    }

    /// A SURVIVING entry must keep its original snapshot id and seqs. Re-stamping it with the rewrite
    /// snapshot breaks merge-on-read delete application and incremental scans.
    #[tokio::test]
    async fn test_rewrite_preserves_surviving_entry_provenance_across_snapshots() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let a = data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone()]).await;
        let s1 = table.metadata().current_snapshot().unwrap().snapshot_id();

        let b = data_file("test/b.parquet", 0);
        let c = data_file("test/c.parquet", 0);
        let table = append_files(&catalog, &table, vec![b.clone(), c]).await;
        let s2 = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(s1, s2);

        let (a_snap, a_seq, a_fseq) = entry_provenance(&table, "test/a.parquet").await;
        let (b_snap, b_seq, b_fseq) = entry_provenance(&table, "test/b.parquet").await;
        assert_eq!(a_snap, Some(s1), "A added by S1");
        assert_eq!(b_snap, Some(s2), "B added by S2");
        assert_ne!(a_seq, b_seq, "A and B must have different data seq numbers");

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![b], vec![data_file("test/d.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s3 = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_ne!(s3, s2);

        // C survived. It must keep S2's snapshot id and seq numbers, not S3.
        let (c_snap, c_seq, c_fseq) = entry_provenance(&table, "test/c.parquet").await;
        assert_eq!(
            c_snap,
            Some(s2),
            "surviving C must keep its ORIGINAL snapshot id S2, not the rewrite snapshot S3"
        );
        assert_eq!(
            c_seq, b_seq,
            "surviving C must keep its ORIGINAL data seq, not the rewrite seq"
        );
        assert_eq!(
            c_fseq, b_fseq,
            "surviving C must keep its ORIGINAL file seq"
        );

        let (a2_snap, a2_seq, a2_fseq) = entry_provenance(&table, "test/a.parquet").await;
        assert_eq!(a2_snap, Some(s1), "carried-forward A keeps S1");
        assert_eq!(a2_seq, a_seq, "carried-forward A keeps its data seq");
        assert_eq!(a2_fseq, a_fseq, "carried-forward A keeps its file seq");

        let (d_snap, d_seq, _d_fseq) = entry_provenance(&table, "test/d.parquet").await;
        assert_eq!(d_snap, Some(s3), "added D gets the new rewrite snapshot id");
        assert_ne!(
            d_seq, b_seq,
            "added D gets the new (higher) data seq, not the rewritten file's seq"
        );

        // B's tombstone carries S3 but keeps B's original data and file seq.
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

    /// The incremental `delete_file` / `add_file` builders must produce the same live set as
    /// `rewrite_files`.
    #[tokio::test]
    async fn test_rewrite_via_incremental_builders() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let b = data_file("test/b.parquet", 0);
        let c = data_file("test/c.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone(), b.clone(), c]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_file(a)
            .delete_file(b)
            .add_file(data_file("test/d.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace
        );
        assert_eq!(
            live_file_paths(&table).await,
            HashSet::from(["test/c.parquet".to_string(), "test/d.parquet".to_string()])
        );
    }

    // dataSequenceNumber preservation, guard lift, and validateNoNewDeletes.
    //
    // These tests use real parquet data, real delete files from the production writers, and a real scan.
    // The resurrection physics is therefore proven end-to-end, not only at the manifest-metadata level.

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

    /// Write a real equality-delete parquet file that deletes rows whose `y` (field id 2) is in
    /// `delete_ys`, in partition `x = part_value`.
    async fn write_equality_delete_file(
        table: &Table,
        part_value: i64,
        delete_ys: &[i64],
    ) -> DataFile {
        use crate::arrow::arrow_schema_to_schema;

        let schema = table.metadata().current_schema().clone();
        // The `y` column is the equality key.
        let config = EqualityDeleteWriterConfig::new(vec![2], schema.clone()).unwrap();
        // The file's parquet schema is the projected, equality_ids-only schema. The writer projects the
        // full-schema input batch down to it.
        let delete_schema =
            Arc::new(arrow_schema_to_schema(config.projected_arrow_schema_ref()).unwrap());

        let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
        let file_name_gen = DefaultFileNameGenerator::new(
            "eq-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            delete_schema,
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );

        let partition_key = crate::spec::PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            schema.clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
            .build(Some(partition_key))
            .await
            .unwrap();

        // One row per deleted `y` value. Only the equality_ids column matters for the match.
        use crate::arrow::schema_to_arrow_schema;
        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let xs: Vec<i64> = delete_ys.iter().map(|_| part_value).collect();
        let ys: Vec<i64> = delete_ys.to_vec();
        let zs: Vec<i64> = delete_ys.iter().map(|_| 0).collect();
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from(xs)) as ArrayRef,
            Arc::new(Int64Array::from(ys)) as ArrayRef,
            Arc::new(Int64Array::from(zs)) as ArrayRef,
        ])
        .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap().into_iter().next().unwrap()
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

    /// Read the explicit, pre-inheritance data seq stored on disk for `path`. It reads the raw avro
    /// bytes, which do not run inheritance. `None` means the entry re-inherits.
    async fn on_disk_data_seq(table: &Table, path: &str) -> Option<i64> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        for manifest_file in manifest_list.entries() {
            let bytes = table
                .file_io()
                .new_input(&manifest_file.manifest_path)
                .unwrap()
                .read()
                .await
                .unwrap();
            let (_, raw_entries) = Manifest::try_from_avro_bytes(&bytes).unwrap();
            for entry in raw_entries {
                if entry.is_alive() && entry.file_path() == path {
                    return entry.sequence_number();
                }
            }
        }
        panic!("no live entry for {path}");
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
            .filter(|m| m.content == crate::spec::ManifestContentType::Deletes)
            .count()
    }

    /// Structural carry-forward pin: the rewrite must not drop the outstanding delete manifest. It
    /// asserts at the manifest-list level, and it is insensitive to the seq-strip mutation. It therefore
    /// separates the carry-forward fix from the seq-preservation fix.
    ///
    /// The discriminating mutation, run manually: in `RewriteFilesOperation::existing_manifest`, filter
    /// `current_manifests()` to `content == ManifestContentType::Data` only. The delete manifest count
    /// then drops to 0 and this test fails.
    #[tokio::test]
    async fn test_rewrite_carries_delete_manifest_forward_structurally() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        // The row delta gives the table exactly one DELETE manifest.
        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![eq_delete]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            count_delete_manifests(&table).await,
            1,
            "the row_delta must leave one delete manifest in the snapshot"
        );

        // Rewrite X into X' with the seq preserved. The delete manifest must survive.
        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![x], vec![x_prime])
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            count_delete_manifests(&table).await,
            1,
            "the rewrite must carry the outstanding delete manifest forward (not drop it)"
        );
    }

    /// Equality-delete row resurrection is silent data corruption. An equality delete at seq 2 removes
    /// y=20 from data file X at seq 1. A rewrite of X into X' that keeps the data seq must keep the
    /// delete applying, so the scan still drops y=20.
    ///
    /// The discriminating mutation, run manually: force `new_data_files_data_sequence_number` to `None`
    /// in `write_added_manifest`. X' then re-inherits the fresh seq and y=20 resurrects.
    #[tokio::test]
    async fn test_rewrite_with_preserved_seq_keeps_equality_delete_applying_no_resurrection() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Append X at seq 1 with rows y = [10, 20, 30].
        let x = write_data_file(&table, "x.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        // Row-delta an equality delete on y that removes y=20, at seq 2.
        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![eq_delete]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let delete_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();
        assert!(x_seq < delete_seq, "delete must be at a higher seq than X");

        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30]),
            "the equality delete drops y=20 from X"
        );

        // Rewrite X into X' with the same rows, keeping the data seq.
        let x_prime = write_data_file(&table, "x-prime.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let x_prime_path = x_prime.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![x], vec![x_prime])
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace
        );

        // The scan stays {10, 30}. The equality delete at seq 2 still applies to X' at seq 1.
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30]),
            "with the seq preserved, the equality delete still drops y=20 — no resurrection"
        );

        // X' carries the explicit data seq 1 on disk, pre-inheritance. The seq-strip mutation breaks this.
        assert_eq!(
            on_disk_data_seq(&table, &x_prime_path).await,
            Some(x_seq),
            "X' must store the preserved data seq EXPLICITLY on disk (never null ⇒ never re-inherits)"
        );
    }

    /// The no-preservation path is the Java-identical hazard, not a safe path. Without
    /// `data_sequence_number` the rewrite still commits, X' takes a fresh higher seq, the old equality
    /// delete stops applying, and y=20 resurrects. Java has no guard here either.
    #[tokio::test]
    async fn test_rewrite_without_preserved_seq_lets_old_equality_deletes_expire_java_parity() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let x = write_data_file(&table, "x.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let table = append_files(&catalog, &table, vec![x.clone()]).await;

        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![eq_delete]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30]),
            "before the rewrite the delete drops y=20"
        );

        // Rewrite X into X' with no seq preservation. X' gets a fresh higher seq, so the equality delete
        // stops applying.
        let x_prime = write_data_file(&table, "x-prime.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![x], vec![x_prime]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 20, 30]),
            "without seq preservation the fresh-seq rewrite resurrects y=20 (Java-identical hazard)"
        );
    }

    /// Commit a concurrent row delta that adds the given delete files. It lands a delete between
    /// tx-build and tx-commit, so the rewrite's `validate` sees it on the refreshed base.
    async fn commit_concurrent_deletes(
        catalog: &impl Catalog,
        table: &Table,
        deletes: Vec<DataFile>,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx.row_delta().add_deletes(deletes);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// A concurrent EQUALITY delete must NOT block a seq-preserving rewrite. The delete still applies to
    /// X', so there is no conflict.
    #[tokio::test]
    async fn test_rewrite_with_preserved_seq_ignores_concurrent_equality_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        // Build the rewrite transaction now. This captures the starting snapshot.
        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![x], vec![x_prime])
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();

        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![eq_delete]).await;

        let committed = tx.commit(&catalog).await;
        assert!(
            committed.is_ok(),
            "a seq-preserving rewrite must ignore a concurrent equality delete (Java L475-479): {:?}",
            committed.err()
        );
    }

    /// A concurrent EQUALITY delete MUST block a non-preserving rewrite. X' takes a fresh seq, so the
    /// delete stops applying and rows resurrect.
    #[tokio::test]
    async fn test_rewrite_without_preserved_seq_rejects_concurrent_equality_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let x_path = x.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x.clone()]).await;

        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let tx = Transaction::new(&table);
        // With no data_sequence_number, any applicable delete is a conflict.
        let action = tx.rewrite_files(vec![x], vec![x_prime]);
        let tx = action.apply(tx).unwrap();

        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![eq_delete]).await;

        let error = tx
            .commit(&catalog)
            .await
            .expect_err("a non-preserving rewrite must reject a concurrent equality delete");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(!error.retryable(), "the conflict must be non-retryable");
        assert_eq!(
            error.message(),
            format!("Cannot commit, found new delete for replaced data file: {x_path}"),
            "exact Java message (any applicable delete is a conflict when seq is not preserved)"
        );
    }

    /// A concurrent POSITION delete blocks the rewrite even with seq preservation. A position delete is
    /// path-scoped, so its target dies with the replaced file.
    #[tokio::test]
    async fn test_rewrite_with_preserved_seq_rejects_concurrent_position_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let x_path = x.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 30, 300)]).await;
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![x], vec![x_prime])
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();

        let pos_delete = write_position_delete_file(&table, 0, &[(x_path.clone(), 1)]).await;
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![pos_delete]).await;

        let error = tx.commit(&catalog).await.expect_err(
            "a concurrent position delete must block the rewrite even with seq preservation",
        );
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(!error.retryable(), "the conflict must be non-retryable");
        assert_eq!(
            error.message(),
            format!("Cannot commit, found new position delete for replaced data file: {x_path}"),
            "exact Java message (a new position delete is always fatal)"
        );
    }

    /// With no `validate_from_snapshot`, a concurrent position delete must be rejected on the
    /// transaction-captured starting snapshot alone.
    ///
    /// The discriminating mutation, run manually: in `RewriteFilesAction::validate`, set
    /// `effective_start` to the refreshed head. The concurrent set is then empty and the commit wrongly
    /// succeeds.
    #[tokio::test]
    async fn test_rewrite_conflict_uses_tx_captured_start_without_override() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let x_path = x.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        // Build the rewrite tx now, with no validate_from_snapshot.
        let x_prime = write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100)]).await;
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![x], vec![x_prime])
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();

        let pos_delete = write_position_delete_file(&table, 0, &[(x_path.clone(), 1)]).await;
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![pos_delete]).await;

        let error = tx
            .commit(&catalog)
            .await
            .expect_err("the conflict must be caught via the tx-captured start (no override)");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains("found new position delete for replaced data file"),
            "unexpected message: {}",
            error.message()
        );
    }

    /// A concurrent position delete on a different file in a different partition must NOT block a
    /// rewrite of X. The walk must not treat an unrelated delete as a conflict.
    #[tokio::test]
    async fn test_rewrite_commits_when_concurrent_delete_targets_disjoint_file() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let y = write_data_file(&table, "y.parquet", 1, &[(1, 60, 600), (1, 70, 700)]).await;
        let y_path = y.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x.clone(), y]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![x], vec![x_prime])
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();

        // The concurrent position delete targets Y, not X.
        let pos_delete = write_position_delete_file(&table, 1, &[(y_path.clone(), 0)]).await;
        let _concurrent = commit_concurrent_deletes(&catalog, &table, vec![pos_delete]).await;

        let committed = tx.commit(&catalog).await;
        assert!(
            committed.is_ok(),
            "a concurrent delete on a disjoint file must not block the rewrite of X: {:?}",
            committed.err()
        );
    }

    /// A delete committed BEFORE the rewrite tx is built belongs to the base, not to a concurrent commit.
    /// The walk covers only the window after the starting snapshot, exclusive.
    #[tokio::test]
    async fn test_rewrite_with_preserved_seq_allows_preexisting_delete() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let table = commit_concurrent_deletes(&catalog, &table, vec![eq_delete]).await;

        // The tx-captured start is the post-delete head.
        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![x], vec![x_prime])
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();

        let committed = tx.commit(&catalog).await;
        assert!(
            committed.is_ok(),
            "a pre-existing delete (before the tx start) must not be treated as a conflict: {:?}",
            committed.err()
        );
    }

    /// The `validate` walk must not fire when there is no concurrent window. Here the starting snapshot
    /// is the current head, so the walk short-circuits.
    #[tokio::test]
    async fn test_rewrite_validate_no_concurrent_commit_is_clean() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![x], vec![x_prime])
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();

        let committed = tx.commit(&catalog).await;
        assert!(
            committed.is_ok(),
            "a rewrite with no concurrent commit must validate cleanly: {:?}",
            committed.err()
        );
    }

    /// The manifest writer strips a negative `data_sequence_number` back into re-inheritance, which
    /// resurrects rows. The action must reject it loudly, not pass it to the writer.
    #[tokio::test]
    async fn test_rewrite_negative_data_sequence_number_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let a = data_file("test/a.parquet", 0);
        let table = append_files(&catalog, &table, vec![a.clone()]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![a], vec![data_file("test/b.parquet", 0)])
            .data_sequence_number(-1);
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("a negative data sequence number must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("Invalid data sequence number")
                && error.message().contains("non-negative"),
            "unexpected error message: {}",
            error.message()
        );

        // The rejected rewrite did not add b.parquet.
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            !live_file_paths(&reloaded).await.contains("test/b.parquet"),
            "the rejected rewrite must not have added b.parquet"
        );
    }

    // DELETE-file REMOVAL surface (Java `RewriteFiles.deleteFile(DeleteFile)`), the commit vehicle
    // `RemoveDanglingDeleteFiles` drives. These pin precondition (1), the Replace operation, the
    // `removed-*` summary counters, the content guard, and the producer routing.

    /// The set of live DELETE-file paths in the table's current snapshot.
    async fn live_delete_file_paths(table: &Table) -> HashSet<String> {
        live_paths_of_content(table, crate::spec::ManifestContentType::Deletes).await
    }

    /// The set of live DATA-file paths in the current snapshot. [`live_file_paths`] instead counts every
    /// live entry across DATA and DELETE manifests, which mixes the two sets.
    async fn live_data_file_paths(table: &Table) -> HashSet<String> {
        live_paths_of_content(table, crate::spec::ManifestContentType::Data).await
    }

    /// The set of live file paths in manifests of the given content type (Data or Deletes).
    async fn live_paths_of_content(
        table: &Table,
        content: crate::spec::ManifestContentType,
    ) -> HashSet<String> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut paths = HashSet::new();
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != content {
                continue;
            }
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.is_alive() {
                    paths.insert(entry.file_path().to_string());
                }
            }
        }
        paths
    }

    /// A delete-file-only rewrite records Replace and bumps `removed-equality-delete-files`. Java's
    /// operation is "replace" even when only delete files are removed.
    #[tokio::test]
    async fn test_rewrite_delete_file_only_records_replace_and_removed_counter() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;

        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let eq_path = eq_delete.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![eq_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert!(
            live_delete_file_paths(&table).await.contains(&eq_path),
            "the equality delete is live before removal"
        );

        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![eq_delete]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace,
            "a delete-file-only RewriteFiles records Replace (Java operation() = replace always)"
        );
        assert_eq!(
            summary_prop(&table, "removed-equality-delete-files").as_deref(),
            Some("1"),
            "the removed equality delete bumps removed-equality-delete-files"
        );
        assert!(
            !live_delete_file_paths(&table).await.contains(&eq_path),
            "the removed equality delete must be tombstoned"
        );
    }

    /// A removed parquet POSITION delete bumps `removed-position-delete-files` and restores the masked
    /// rows. It pins the summary branch and the end-to-end read effect.
    #[tokio::test]
    async fn test_rewrite_delete_file_only_removes_position_delete_restores_rows() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let x = write_data_file(&table, "x.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
            (0, 30, 300),
        ])
        .await;
        let x_path = x.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x]).await;

        let pos_delete = write_position_delete_file(&table, 0, &[(x_path, 1)]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![pos_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 30]),
            "the position delete drops y=20"
        );

        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![pos_delete]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            summary_prop(&table, "removed-position-delete-files").as_deref(),
            Some("1"),
            "the removed position delete bumps removed-position-delete-files"
        );
        assert_eq!(
            summary_prop(&table, "removed-dvs"),
            None,
            "a parquet position delete is not a DV"
        );
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 20, 30]),
            "removing the position delete restores y=20"
        );
    }

    /// A `Data`-content file passed to the DELETE-removal surface is rejected. Otherwise a data file
    /// routes silently into the delete-removal path.
    #[tokio::test]
    async fn test_rewrite_delete_delete_files_rejects_data_content() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![data_file("test/a.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("delete_delete_files must reject a Data-content file");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains("Only position-delete or equality-delete content is allowed for removed delete files"),
            "unexpected message: {}",
            error.message()
        );
    }

    /// Removing a delete file that is NOT live fails loudly with Java's `failMissingDeletePaths` message
    /// shape. Otherwise the removal no-ops and the rewrite is partial.
    #[tokio::test]
    async fn test_rewrite_delete_delete_file_missing_path_errors() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

        let ghost = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/ghost-pos-del.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .referenced_data_file(Some("test/a.parquet".to_string()))
            .build()
            .unwrap();
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![ghost]);
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("removing a delete file that is not live must fail loud");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("Missing required files to delete"),
            "unexpected message: {}",
            error.message()
        );
    }

    /// A rewrite that removes a delete file must route it through `with_removed_delete_files`, so the
    /// producer tombstones it in the rewritten DELETE manifest.
    ///
    /// The discriminating mutation, run manually: in `RewriteFilesAction::commit`, gate the
    /// `with_removed_delete_files` call on `false`. The delete then stays live with no tombstone.
    #[tokio::test]
    async fn test_rewrite_delete_file_routes_through_producer_tombstone() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;
        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let eq_path = eq_delete.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![eq_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![eq_delete]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut found_tombstone = false;
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != crate::spec::ManifestContentType::Deletes {
                continue;
            }
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.file_path() == eq_path && entry.status() == ManifestStatus::Deleted {
                    found_tombstone = true;
                }
            }
        }
        assert!(
            found_tombstone,
            "the removed delete must be tombstoned (producer routing fired)"
        );
    }

    // DELETE-file ADD surface (Java `addFile(DeleteFile)`, `addFile(DeleteFile, long)`, and the 4-set
    // `rewriteFiles`). These pin the rewrite of a delete file into a new one with the explicit seq, the
    // on-disk pre-inheritance seq, the 4-arg atomic rewrite, precondition (3) in both directions, and the
    // content and negative-seq guards.

    /// A rewritten DELETE file stamped with an inherited, higher seq over-deletes a re-inserted row. An
    /// equality delete E at seq 2 removes y=20 from X at seq 1. A later append W at seq 3 re-inserts
    /// y=20, which E must not touch. Rewriting E into E' with E's original seq keeps E' applying to X
    /// alone, so the scan stays {10, 20, 30}.
    ///
    /// The discriminating mutation, run manually: in `write_added_delete_manifests`, drop the
    /// explicit-seq branch so the added delete inherits the fresh rewrite seq. E' then applies to W too
    /// and the scan becomes {10, 30}.
    #[tokio::test]
    async fn test_rewrite_delete_into_new_delete_with_explicit_seq_no_over_deletion() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Append X at seq 1 with y = [10, 20].
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;

        // Row-delta an equality delete E at seq 2 that removes y=20.
        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let eq_delete_seq = {
            let tx = Transaction::new(&table);
            let action = tx.row_delta().add_deletes(vec![eq_delete.clone()]);
            let tx = action.apply(tx).unwrap();
            let table = tx.commit(&catalog).await.unwrap();
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .sequence_number()
        };
        let table = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10]),
            "before the re-insert E drops both X's rows except y=10"
        );

        // Append W at seq 3, which re-inserts y=20. E at seq 2 must not apply to W.
        let w = write_data_file(&table, "w.parquet", 0, &[(0, 20, 201), (0, 30, 300)]).await;
        let table = append_files(&catalog, &table, vec![w]).await;
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 20, 30]),
            "W re-inserts y=20 (E does not apply to the later W)"
        );

        // Rewrite E into E' with E's original seq. E' must drop the same rows E did.
        let eq_delete_prime = write_equality_delete_file(&table, 0, &[20]).await;
        let eq_prime_path = eq_delete_prime.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![], vec![]).rewrite_files_with_deletes(
            vec![],
            vec![eq_delete],
            vec![],
            vec![],
        );
        // The 4-set form's added-delete default inherits, so use the explicit overload here.
        let action = action.add_delete_file_with_sequence_number(eq_delete_prime, eq_delete_seq);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace
        );

        // The scan stays {10, 20, 30}. E' drops X's y=20 and leaves W's re-inserted y=20.
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10, 20, 30]),
            "E' (explicit seq 2) drops X's y=20 but leaves W's re-inserted y=20 — no over-deletion"
        );

        // E' carries the explicit seq on disk, pre-inheritance.
        assert_eq!(
            on_disk_data_seq(&table, &eq_prime_path).await,
            Some(eq_delete_seq),
            "E' must store the explicit data seq EXPLICITLY on disk (never null ⇒ never re-inherits a higher seq)"
        );
    }

    /// `add_delete_file_with_sequence_number(file, seq)` must write `seq` as the entry's data seq on
    /// disk, pre-inheritance. It reads the raw avro entry, so a re-inheriting entry reads back `None` and
    /// fails. This is the direct metadata pin, independent of the scan.
    #[tokio::test]
    async fn test_rewrite_add_delete_file_with_sequence_number_stamps_on_disk_seq() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;

        // An existing delete to satisfy precondition (3) (addsDeleteFiles ⇒ deletesDeleteFiles).
        let old_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![old_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // The explicit seq differs from the new snapshot's seq, so an inherited entry would differ.
        let explicit_seq = 1i64;
        let new_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let new_delete_path = new_delete.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![old_delete])
            .add_delete_file_with_sequence_number(new_delete, explicit_seq);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snapshot_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();
        assert_ne!(
            explicit_seq, snapshot_seq,
            "the test is only meaningful when the explicit seq differs from the snapshot seq"
        );
        assert_eq!(
            on_disk_data_seq(&table, &new_delete_path).await,
            Some(explicit_seq),
            "the explicit-seq overload must stamp the given seq on disk (pre-inheritance), not inherit"
        );
    }

    /// The default `add_delete_file` must leave the entry with NO explicit seq on disk, so it inherits
    /// the new snapshot's seq at read time. A `None` on-disk seq is the inherit signal. This covers the
    /// other direction of the `Option<i64>` stamping.
    #[tokio::test]
    async fn test_rewrite_add_delete_file_default_inherits_seq_on_disk() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;

        let old_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![old_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let new_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let new_delete_path = new_delete.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![old_delete])
            .add_delete_file(new_delete);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            on_disk_data_seq(&table, &new_delete_path).await,
            None,
            "the default add_delete_file must leave NO explicit seq on disk (it inherits the snapshot seq)"
        );
    }

    /// `rewrite_files_with_deletes` must apply all four sets in ONE Replace snapshot. A partial rewrite
    /// leaves the live DATA set or the live DELETE set wrong.
    #[tokio::test]
    async fn test_rewrite_four_arg_replaces_data_and_delete_atomically() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let x_path = x.file_path().to_string();
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        // Row-delta an equality delete D_old that removes y=20.
        let d_old = write_equality_delete_file(&table, 0, &[20]).await;
        let d_old_path = d_old.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![d_old.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Replace X with X' and D_old with D_new, in one snapshot. X keeps its data seq, so D_new still
        // applies at its inherited, higher seq.
        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let x_prime_path = x_prime.file_path().to_string();
        let d_new = write_equality_delete_file(&table, 0, &[20]).await;
        let d_new_path = d_new.file_path().to_string();

        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .rewrite_files_with_deletes(vec![x], vec![d_old], vec![x_prime], vec![d_new])
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Replace
        );

        assert_eq!(
            live_data_file_paths(&table).await,
            HashSet::from([x_prime_path]),
            "the 4-set rewrite must replace X with X' in the same snapshot"
        );
        assert_deleted_tombstone(&table, &x_path).await;

        let live_deletes = live_delete_file_paths(&table).await;
        assert!(
            live_deletes.contains(&d_new_path),
            "D_new must be live after the 4-set rewrite"
        );
        assert!(
            !live_deletes.contains(&d_old_path),
            "D_old must be replaced (not live) after the 4-set rewrite"
        );

        // The scan still drops y=20, so data and delete were rewritten consistently.
        assert_eq!(
            scan_y_values(&table).await,
            HashSet::from([10]),
            "after the atomic data+delete rewrite the equality delete still drops y=20"
        );
    }

    /// Precondition (3), rejection direction. A rewrite that adds a delete file but deletes only a DATA
    /// file must be rejected.
    ///
    /// The discriminating mutation, run manually: in `RewriteFilesAction::commit`, change the
    /// precondition (3) guard to `if false`. The illegal rewrite then commits.
    #[tokio::test]
    async fn test_rewrite_add_delete_without_deleting_delete_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x.clone()]).await;
        let x_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        // Deleting a DATA file and adding a delete file fires precondition (3).
        let x_prime =
            write_data_file(&table, "x-prime.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let new_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![x], vec![x_prime])
            .add_delete_file(new_delete)
            .data_sequence_number(x_seq);
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("adding a delete file without deleting one must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains(
                "Delete files to add must be empty because there's no delete file to be rewritten"
            ),
            "unexpected message: {}",
            error.message()
        );
    }

    /// Precondition (3), legal direction. When delete files are deleted, adding delete files is legal.
    /// The guard must not over-fire.
    ///
    /// The discriminating mutation, run manually: over-broaden precondition (3) to `if
    /// adds_delete_files`, dropping the `!deletes_delete_files` conjunct. The legal rewrite is then
    /// rejected.
    #[tokio::test]
    async fn test_rewrite_replace_delete_with_delete_is_allowed() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;

        let d_old = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![d_old.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let d_old_seq = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .sequence_number();

        let d_new = write_equality_delete_file(&table, 0, &[20]).await;
        let d_new_path = d_new.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![d_old])
            .add_delete_file_with_sequence_number(d_new, d_old_seq);
        let tx = action.apply(tx).unwrap();
        let committed = tx.commit(&catalog).await;
        assert!(
            committed.is_ok(),
            "replacing a delete file with a delete file is legal under precondition (3): {:?}",
            committed.err()
        );
        let table = committed.unwrap();
        assert!(
            live_delete_file_paths(&table).await.contains(&d_new_path),
            "D_new must be live after the legal delete-rewrite"
        );
    }

    /// Content guard on the ADDED delete set. A Data-content file must not route into the delete-add
    /// path.
    #[tokio::test]
    async fn test_rewrite_add_delete_file_rejects_data_content() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;

        // A deleted delete file passes precondition (3), so the content guard is what fires.
        let old_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![old_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![old_delete])
            .add_delete_file(data_file("test/not-a-delete.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("add_delete_file must reject a Data-content file");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains(
                "Only position-delete or equality-delete content is allowed for added delete files"
            ),
            "unexpected message: {}",
            error.message()
        );
    }

    /// The action-level content guard pinned at its own door. Without this, disabling the action guard
    /// stays green, because the producer guard in `SnapshotProducer::validate_added_delete_files` fires
    /// with the same generic substring. Only the action-level message carries the "(use add_file to add
    /// data files)" suffix, so asserting that suffix goes red when the action guard is bypassed.
    #[tokio::test]
    async fn test_rewrite_add_delete_file_data_content_hits_action_guard_first() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;

        // A deleted delete file satisfies precondition (3), so the commit reaches the content guard.
        let old_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![old_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![old_delete])
            .add_delete_file(data_file("test/not-a-delete.parquet", 0));
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("add_delete_file must reject a Data-content file at the ACTION guard");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        // Only the action-level guard's message carries this suffix. The producer guard stops before it.
        assert!(
            error.message().contains("(use add_file to add data files)"),
            "the ACTION-level guard message must carry the 'use add_file' suffix (a bypass that lets the \
             producer guard fire instead loses this suffix); got: {}",
            error.message()
        );
    }

    /// A negative explicit delete seq is stripped into re-inheritance, so the rewritten delete takes a
    /// higher seq and over-deletes. The action must reject it loudly.
    #[tokio::test]
    async fn test_rewrite_add_delete_file_negative_sequence_number_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;

        let old_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![old_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let new_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![old_delete])
            .add_delete_file_with_sequence_number(new_delete, -1);
        let tx = action.apply(tx).unwrap();
        let error = tx
            .commit(&catalog)
            .await
            .expect_err("a negative added-delete data sequence number must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains("Invalid data sequence number for added delete file")
                && error.message().contains("non-negative"),
            "unexpected message: {}",
            error.message()
        );
    }

    /// The negative-seq guard at its lower boundary. Seq 0 is the initial-sequence-number sentinel and
    /// is legal, so the guard must reject only strictly-negative seqs. An explicit seq of 0 must commit
    /// and be stamped on disk as `Some(0)`. Only a test at exactly 0 separates `< 0` from `<= 0`.
    #[tokio::test]
    async fn test_rewrite_add_delete_file_zero_sequence_number_allowed() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let table = append_files(&catalog, &table, vec![x]).await;

        // An existing delete to satisfy precondition (3) (addsDeleteFiles ⇒ deletesDeleteFiles).
        let old_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![old_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let new_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let new_delete_path = new_delete.file_path().to_string();
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files(vec![], vec![])
            .delete_delete_files(vec![old_delete])
            .add_delete_file_with_sequence_number(new_delete, 0);
        let tx = action.apply(tx).unwrap();
        let committed = tx.commit(&catalog).await;
        assert!(
            committed.is_ok(),
            "an explicit data sequence number of 0 is legal and must commit (guard is `< 0`, not `<= 0`): {:?}",
            committed.err()
        );
        let table = committed.unwrap();

        // The explicit seq 0 is stamped on disk, never re-inherited into the higher snapshot seq.
        assert_eq!(
            on_disk_data_seq(&table, &new_delete_path).await,
            Some(0),
            "the explicit seq 0 must be stamped on disk, not re-inherited"
        );
    }
}
