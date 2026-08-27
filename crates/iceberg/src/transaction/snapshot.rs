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

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::ops::RangeFrom;

use uuid::Uuid;

use crate::error::Result;
use crate::expr::visitors::inclusive_metrics_evaluator::InclusiveMetricsEvaluator;
use crate::expr::visitors::residual_evaluator::ResidualEvaluator;
use crate::expr::visitors::strict_metrics_evaluator::StrictMetricsEvaluator;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::spec::{
    DataFile, DataFileFormat, FormatVersion, MAIN_BRANCH, Manifest, ManifestContentType,
    ManifestEntry, ManifestFile, ManifestListWriter, ManifestStatus, ManifestWriter,
    ManifestWriterBuilder, Operation, Schema, Snapshot, SnapshotReference, SnapshotRetention,
    SnapshotSummaryCollector, Struct, StructType, Summary, TableProperties,
    update_snapshot_summaries,
};
use crate::table::Table;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";

/// A trait that defines how different table operations produce new snapshots.
///
/// [`SnapshotProducer`] uses it to customize snapshot creation per operation type. Each
/// implementation states three things: the operation to record in the summary, which existing
/// manifests carry forward, and which entries are marked deleted.
pub(crate) trait SnapshotProduceOperation: Send + Sync {
    /// Returns the operation type that will be recorded in the snapshot summary.
    ///
    /// This determines what kind of operation is being performed (e.g., `Append`, `Overwrite`),
    /// which is stored in the snapshot metadata for tracking and auditing purposes.
    fn operation(&self) -> Operation;

    /// Returns manifest entries that should be marked as deleted in the new snapshot.
    #[allow(unused)]
    fn delete_entries(
        &self,
        snapshot_produce: &SnapshotProducer,
    ) -> impl Future<Output = Result<Vec<ManifestEntry>>> + Send;

    /// Returns the data files this operation wants to remove from the table.
    ///
    /// The producer resolves these against the current snapshot's manifests at commit time: every
    /// existing manifest that contains a live entry for one of these files is rewritten with the
    /// matching entries marked `Deleted` (mirroring Java `ManifestFilterManager.filterManifest`).
    /// Operations that only add files (e.g. fast append) return an empty vector.
    fn delete_files(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> impl Future<Output = Result<Vec<DataFile>>> + Send;

    /// Returns existing manifest files that should be included in the new snapshot.
    ///
    /// This method determines which manifest files from the current snapshot should be
    /// carried forward to the new snapshot. The selection depends on the operation type:
    ///
    /// - **Append operations**: Typically include all existing manifests
    /// - **Overwrite operations**: May exclude manifests for partitions being overwritten
    /// - **Delete operations**: May exclude manifests for partitions being deleted
    fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> impl Future<Output = Result<Vec<ManifestFile>>> + Send;
}

pub(crate) struct DefaultManifestProcess;

impl ManifestProcess for DefaultManifestProcess {
    async fn process_manifests(
        &self,
        _snapshot_produce: &mut SnapshotProducer<'_>,
        manifests: Vec<ManifestFile>,
    ) -> Result<Vec<ManifestFile>> {
        // Pass the manifest list through unchanged — the fast-append / single-manifest path. This MUST
        // stay a no-op so `FastAppend` behavior is byte-identical to the pre-seam-change producer.
        Ok(manifests)
    }
}

/// Post-process the manifest list a snapshot is about to commit, after the producer has written the
/// added DATA/DELETE manifests and rewritten any delete-bearing manifests (Java
/// `MergingSnapshotProducer.apply`'s `mergeManager.mergeManifests(...)` step). The default
/// ([`DefaultManifestProcess`]) returns the list untouched (fast append); the merge-append manager
/// ([`crate::transaction::merge_append::MergeManifestProcess`]) bin-packs and merges them.
///
/// Takes `&mut SnapshotProducer` because a manager that MERGES manifests needs the producer's writer
/// factory ([`SnapshotProducer::new_cluster_manifest_writer`]) — which advances the manifest-name
/// counter — to write the merged manifests. It is async + `Result` because merging reads the input
/// manifests back from object storage and writes new ones.
pub(crate) trait ManifestProcess: Send + Sync {
    fn process_manifests(
        &self,
        snapshot_produce: &mut SnapshotProducer<'_>,
        manifests: Vec<ManifestFile>,
    ) -> impl Future<Output = Result<Vec<ManifestFile>>> + Send;
}

/// What a producer does with the `first_row_id` an ADDED data file already carries.
///
/// Java splits this by base class, so the fork makes it a REQUIRED constructor argument: a new
/// producer cannot inherit the wrong half by omission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FirstRowIdPolicy {
    /// Write the value the caller supplied. Java `FastAppend` and `BaseRewriteManifests`, which
    /// extend `SnapshotProducer` and never call `Delegates.suppressFirstRowId`.
    Preserve,
    /// Force the field absent. Java `MergingSnapshotProducer.add(DataFile)`. A stale id survives
    /// read-side inheritance, so the file keeps a row-id range that describes other rows.
    Suppress,
}

/// An ADDED delete file paired with its OPTIONAL explicit DATA sequence number — the Rust analogue of
/// Java's `Delegates.PendingDeleteFile` (a delete file wrapped with a nullable `dataSequenceNumber()`).
/// `None` ⇒ the entry inherits the new snapshot's sequence number at read time (Java
/// `addFile(DeleteFile)`); `Some(seq)` ⇒ the entry is written with that explicit data seq (Java
/// `addFile(DeleteFile, long)` → `writeDeleteFileGroup`'s `writer.add(file, dataSeq)`).
pub(crate) type PendingDeleteFile = (DataFile, Option<i64>);

pub(crate) struct SnapshotProducer<'a> {
    pub(crate) table: &'a Table,
    snapshot_id: i64,
    commit_uuid: Uuid,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    // DELETE files this snapshot adds, each paired with an OPTIONAL explicit data sequence number.
    // Java `Delegates.PendingDeleteFile`. `None` inherits the new snapshot's sequence number at
    // read time, so a delete added now applies to earlier data. `Some(seq)` writes that explicit
    // seq, the `RewriteFiles.addFile(DeleteFile, long)` path that re-stamps a rewritten delete file
    // with the seq it must keep to still apply to its data.
    added_delete_files: Vec<PendingDeleteFile>,
    // An explicit DATA sequence number stamped on every ADDED data file. Java
    // `MergingSnapshotProducer.newDataFilesDataSequenceNumber`. It is the `RewriteFiles`
    // preservation path that keeps outstanding equality deletes applying to rewritten data. `None`
    // makes the added files inherit the new snapshot's sequence number.
    new_data_files_data_sequence_number: Option<i64>,
    // Data files removed by this snapshot, resolved against the current snapshot at commit time. Held
    // so the snapshot summary can reflect the deleted file/record counts (Java overwrite/delete summary).
    // Empty for add-only operations such as fast append.
    removed_data_files: Vec<DataFile>,
    // DELETE files removed by this snapshot: the apply-side removal of superseded merge-on-read
    // delete files. Java `MergingSnapshotProducer.delete(DeleteFile)`. Resolved by path against the
    // current DELETE manifests in `commit()`, then fed to the same `process_deletes` rewrite path
    // and to the summary's `remove_file`.
    removed_delete_files: Vec<DataFile>,
    // The write-audit-publish staging flag. Java `SnapshotProducer.stageOnly`. When `true` the
    // commit emits `AddSnapshot` ALONE, with no `SetSnapshotRef`, so `current-snapshot-id`, the
    // `main` ref and the snapshot log stay unchanged. A cherry-pick publishes it later. The staged
    // snapshot still CONSUMES a sequence number, exactly like a normal commit.
    stage_only: bool,
    // A counter used to generate unique manifest file names.
    // It starts from 0 and increments for each new manifest file.
    // Note: This counter is limited to the range of (0..u64::MAX).
    manifest_counter: RangeFrom<u64>,
}

impl<'a> SnapshotProducer<'a> {
    /// Build a producer for one snapshot.
    ///
    /// `first_row_id_policy` is the add-seam rule of [`FirstRowIdPolicy`]. It is applied here, so
    /// validation, the summary and the manifest writer all see one value.
    pub(crate) fn new(
        table: &'a Table,
        commit_uuid: Uuid,
        key_metadata: Option<Vec<u8>>,
        snapshot_properties: HashMap<String, String>,
        added_data_files: Vec<DataFile>,
        first_row_id_policy: FirstRowIdPolicy,
    ) -> Self {
        let added_data_files = match first_row_id_policy {
            FirstRowIdPolicy::Preserve => added_data_files,
            FirstRowIdPolicy::Suppress => added_data_files
                .into_iter()
                .map(|mut data_file| {
                    data_file.first_row_id = None;
                    data_file
                })
                .collect(),
        };
        Self {
            table,
            snapshot_id: Self::generate_unique_snapshot_id(table),
            commit_uuid,
            key_metadata,
            snapshot_properties,
            added_data_files,
            added_delete_files: vec![],
            new_data_files_data_sequence_number: None,
            removed_data_files: vec![],
            removed_delete_files: vec![],
            stage_only: false,
            manifest_counter: (0..),
        }
    }

    /// STAGE the produced snapshot for write-audit-publish instead of publishing it to `main` (Java
    /// `SnapshotProducer.stageOnly()`). When enabled, [`SnapshotProducer::commit`] emits ONLY the
    /// `AddSnapshot` update — no `SetSnapshotRef` — so the new snapshot is added to table metadata but the
    /// `main` ref, `current-snapshot-id`, and the snapshot-log are left UNCHANGED on disk. The snapshot is
    /// staged for a later cherry-pick/publish (the WAP "write" half; [`crate::transaction::cherry_pick`] is
    /// the "publish" half). The staged snapshot still consumes a sequence number exactly like a normal
    /// commit (Java's `apply()` assigns `base.nextSequenceNumber()` regardless of the flag). Idempotent.
    pub(crate) fn with_stage_only(mut self, stage_only: bool) -> Self {
        self.stage_only = stage_only;
        self
    }

    /// Attach the DELETE files (position / equality) this snapshot adds with the DEFAULT (inherited)
    /// sequence number. They are written into a DELETE manifest alongside the DATA manifest in the same
    /// snapshot (Java `MergingSnapshotProducer.add(DeleteFile)` → `pendingDeleteFile(file, null)`). Each
    /// entry inherits the new snapshot's sequence number at read time. Used by the merge-on-read write
    /// path (`RowDelta`).
    pub(crate) fn with_added_delete_files(mut self, added_delete_files: Vec<DataFile>) -> Self {
        self.added_delete_files = added_delete_files
            .into_iter()
            .map(|file| (file, None))
            .collect();
        self
    }

    /// Attach the DELETE files this snapshot adds, each paired with an OPTIONAL explicit DATA sequence
    /// number — the Rust analogue of Java's per-file `Delegates.PendingDeleteFile.dataSequenceNumber()`.
    /// `None` ⇒ the entry inherits the new snapshot's seq (Java `addFile(DeleteFile)`); `Some(seq)` ⇒ the
    /// entry is written with that explicit data seq (Java `addFile(DeleteFile, long)` →
    /// `writeDeleteFileGroup`'s `writer.add(file, dataSeq)`). Used by the compaction write path
    /// (`RewriteFiles`) to re-stamp a rewritten delete file with the data seq it must keep so it still
    /// applies to its data. A `Some(seq)` value MUST be non-negative (the caller validates this; the
    /// manifest writer silently strips a negative explicit seq back into re-inheritance).
    pub(crate) fn with_added_delete_files_with_seq(
        mut self,
        added_delete_files: Vec<PendingDeleteFile>,
    ) -> Self {
        self.added_delete_files = added_delete_files;
        self
    }

    /// Attach the DELETE files this snapshot REMOVES. Java
    /// `MergingSnapshotProducer.delete(DeleteFile)`. `RowDelta.removeDeletes` uses it to drop a
    /// delete file the new delete supersedes, such as the old DV a merged super-set DV replaces.
    ///
    /// [`SnapshotProducer::commit`] resolves the paths against the current DELETE manifests, and a
    /// missing path fails loud. `process_deletes` matches by path across EVERY manifest, so the
    /// tombstone lands in the rewritten DELETE manifest and DATA manifests are untouched.
    pub(crate) fn with_removed_delete_files(mut self, removed_delete_files: Vec<DataFile>) -> Self {
        self.removed_delete_files = removed_delete_files;
        self
    }

    /// Stamp every ADDED data file with an explicit DATA sequence number instead of inheriting the
    /// new snapshot's. Java `MergingSnapshotProducer.setNewDataFilesDataSequenceNumber`. Compaction
    /// preserves the replaced files' seq so outstanding equality deletes still apply. Without it the
    /// added files take a higher seq, the old deletes stop applying, and deleted rows return.
    ///
    /// # Notes
    ///
    /// `seq` must be non-negative. The manifest writer silently strips a negative one back into
    /// re-inheritance, so the caller validates first.
    pub(crate) fn with_new_data_files_data_sequence_number(mut self, sequence_number: i64) -> Self {
        self.new_data_files_data_sequence_number = Some(sequence_number);
        self
    }

    /// The id of the snapshot this producer is creating. Exposed so an action that pre-computes its
    /// own manifest list (e.g. `RewriteManifests`) can stamp externally-added manifests with the new
    /// snapshot id before they reach the manifest-list writer (Java `withSnapshotId`,
    /// `BaseRewriteManifests.apply` L184-187 — required by
    /// [`ManifestListWriter::add_manifests`]'s `assign_sequence_numbers` precondition).
    pub(crate) fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    /// Merge additional snapshot summary properties computed AFTER construction (Java
    /// `RewriteManifests.summary()` sets `manifests-created` / `-kept` / `-replaced` /
    /// `entries-processed` only once the rewrite has run). [`SnapshotProducer::new`] takes the
    /// user-supplied properties up front; this additive setter lets the rewrite inject the counts it
    /// can only know post-rewrite. These non-empty properties also satisfy the empty-commit
    /// precondition in [`SnapshotProducer::manifest_file`] for an action that adds no data files.
    pub(crate) fn extend_snapshot_properties(
        &mut self,
        properties: impl IntoIterator<Item = (String, String)>,
    ) {
        self.snapshot_properties.extend(properties);
    }

    /// Build a manifest writer for a brand-new manifest of `content` under `partition_spec_id`.
    /// Java `BaseRewriteManifests.getWriter`.
    ///
    /// It keys on the partition-spec id directly, which is the difference from
    /// [`SnapshotProducer::new_filtering_manifest_writer`], which keys off a source manifest. The
    /// `content` axis mirrors Java's `writeDataManifests` against `writeDeleteManifests`.
    pub(crate) fn new_cluster_manifest_writer(
        &mut self,
        partition_spec_id: i32,
        content: ManifestContentType,
    ) -> Result<ManifestWriter> {
        let partition_spec = self
            .table
            .metadata()
            .partition_spec_by_id(partition_spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot rewrite manifests: unknown partition spec id {partition_spec_id}"
                    ),
                )
            })?
            .as_ref()
            .clone();

        let new_manifest_path = format!(
            "{}/{}/{}-m{}.{}",
            self.table.metadata().location(),
            META_ROOT_PATH,
            self.commit_uuid,
            self.manifest_counter.next().ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Exhausted manifest file name counter",
                )
            })?,
            DataFileFormat::Avro
        );
        let output_file = self.table.file_io().new_output(new_manifest_path)?;
        let builder = ManifestWriterBuilder::new(
            output_file,
            Some(self.snapshot_id),
            self.key_metadata.clone(),
            self.table.metadata().current_schema().clone(),
            partition_spec,
        );
        match self.table.metadata().format_version() {
            FormatVersion::V1 => Ok(builder.build_v1()),
            FormatVersion::V2 => match content {
                ManifestContentType::Data => Ok(builder.build_v2_data()),
                ManifestContentType::Deletes => Ok(builder.build_v2_deletes()),
            },
            FormatVersion::V3 => match content {
                ManifestContentType::Data => Ok(builder.build_v3_data()),
                ManifestContentType::Deletes => Ok(builder.build_v3_deletes()),
            },
        }
    }

    /// Validate the added DELETE files (Java `RowDelta.addDeletes` / `MergingSnapshotProducer.add`):
    /// each must be a `PositionDeletes` or `EqualityDeletes` content file (a `Data` file is rejected —
    /// it must be added as a row, not a delete), must pass the FORMAT-VERSION gate (see
    /// [`validate_delete_file_for_version`]), and its partition spec must EXIST in the table's specs.
    ///
    /// **Per-spec, not default-spec-only.** Java resolves `spec(file.specId())` for EACH added
    /// delete file and rejects only when no such spec exists. A delete file under an OLDER spec is
    /// accepted, and the producer writes per-spec DELETE manifest groups. The partition-value
    /// compatibility check runs against that file's own spec.
    ///
    /// **Placement of the format-version gate.** Java 1.10.0 gates at add time. This model has no
    /// table access then, so the gate runs in the action's `commit()` against the REFRESHED base.
    /// That is Java main's stronger apply-time placement, and it subsumes the add-time check: a row
    /// delta built before a concurrent format upgrade is re-gated on every retry.
    pub(crate) fn validate_added_delete_files(&self) -> Result<()> {
        let format_version = self.table.metadata().format_version();
        for (delete_file, _explicit_seq) in &self.added_delete_files {
            match delete_file.content_type() {
                crate::spec::DataContentType::PositionDeletes
                | crate::spec::DataContentType::EqualityDeletes => {}
                crate::spec::DataContentType::Data => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        "Only position-delete or equality-delete content is allowed for added delete files",
                    ));
                }
            }
            validate_delete_file_for_version(delete_file, format_version)?;
            // Java `addInternal`: `spec(file.specId())` must exist (any table spec, not just the default).
            let partition_type = self.partition_type_for_added_file(delete_file, true)?;
            Self::validate_partition_value(delete_file.partition(), &partition_type)?;
        }

        Ok(())
    }

    pub(crate) fn validate_added_data_files(&self) -> Result<()> {
        for data_file in &self.added_data_files {
            if data_file.content_type() != crate::spec::DataContentType::Data {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Only data content type is allowed for fast append",
                ));
            }
            // Java `MergingSnapshotProducer.add(DataFile)` / `FastAppend.appendFile`: `spec(file.specId())`
            // must exist (ANY table spec, not just the default). A file under an older spec is accepted and
            // routed into its own per-spec manifest group; only an UNKNOWN spec id is rejected.
            let partition_type = self.partition_type_for_added_file(data_file, false)?;
            Self::validate_partition_value(data_file.partition(), &partition_type)?;
        }

        Ok(())
    }

    /// Resolve the partition TYPE of the spec a freshly-added file claims. Java
    /// `MergingSnapshotProducer.add` / `FastAppend.appendFile`.
    ///
    /// `is_delete_file` selects the message noun, matching Java's two distinct precondition
    /// messages. The partition-value compatibility check binds against the returned type.
    ///
    /// # Errors
    ///
    /// No such spec exists. The message is Java's verbatim.
    fn partition_type_for_added_file(
        &self,
        file: &DataFile,
        is_delete_file: bool,
    ) -> Result<StructType> {
        let metadata = self.table.metadata();
        let spec = metadata
            .partition_spec_by_id(file.partition_spec_id)
            .ok_or_else(|| {
                let noun = if is_delete_file {
                    "delete file"
                } else {
                    "data file"
                };
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot find partition spec {} for {}: {}",
                        file.partition_spec_id,
                        noun,
                        file.file_path()
                    ),
                )
            })?;
        spec.partition_type(metadata.current_schema())
    }

    pub(crate) async fn validate_duplicate_files(&self) -> Result<()> {
        let new_files: HashSet<&str> = self
            .added_data_files
            .iter()
            .map(|df| df.file_path.as_str())
            .collect();

        let mut referenced_files = Vec::new();
        if let Some(current_snapshot) = self.table.metadata().current_snapshot() {
            let manifest_list = current_snapshot
                .load_manifest_list(self.table.file_io(), &self.table.metadata_ref())
                .await?;
            for manifest_list_entry in manifest_list.entries() {
                let manifest = manifest_list_entry
                    .load_manifest(self.table.file_io())
                    .await?;
                for entry in manifest.entries() {
                    let file_path = entry.file_path();
                    if new_files.contains(file_path) && entry.is_alive() {
                        referenced_files.push(file_path.to_string());
                    }
                }
            }
        }

        if !referenced_files.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot add files that are already referenced by table, files: {}",
                    referenced_files.join(", ")
                ),
            ));
        }

        Ok(())
    }

    /// Return EVERY current manifest — DATA **and** DELETE — from the current snapshot's manifest list,
    /// the complete candidate set a delete-bearing operation's `existing_manifest` hands to the producer.
    ///
    /// Shared by every delete-bearing operation. Each exposes the FULL manifest list so
    /// `process_deletes` can rewrite, carry forward, or drop each DATA manifest. Every DELETE
    /// manifest carries forward UNCHANGED, because its entries are delete-file paths, which never
    /// appear in a DATA `delete_paths` set.
    ///
    /// Carrying delete manifests forward is REQUIRED FOR CORRECTNESS, not an optimization. Java
    /// `MergingSnapshotProducer.apply` composes both filtered lists into the new manifest list. An
    /// action returning DATA manifests only would omit every delete manifest the current snapshot
    /// carried, which silently drops all outstanding deletes table-wide and resurrects every
    /// deleted row. This helper makes that bug class unrepresentable.
    ///
    /// **Conservative dangling-delete posture, a documented divergence.** Java's `apply` also drops
    /// delete files older than the surviving data's minimum sequence number, and removes DVs
    /// orphaned by the data files it deleted. This port carries every delete manifest forward
    /// UNCHANGED. Keeping a delete that no longer applies is harmless, while dropping one that still
    /// applies resurrects rows. Cleanup belongs to `RemoveDanglingDeleteFiles`.
    pub(crate) async fn current_manifests(&self) -> Result<Vec<ManifestFile>> {
        let Some(snapshot) = self.table.metadata().current_snapshot() else {
            return Ok(vec![]);
        };

        let manifest_list = snapshot
            .load_manifest_list(self.table.file_io(), &self.table.metadata_ref())
            .await?;

        Ok(manifest_list.entries().to_vec())
    }

    /// Resolve `delete_paths` against the current snapshot's live data entries, returning the matching
    /// [`DataFile`]s, and fail if any requested path matched no live entry.
    ///
    /// Shared by `DeleteFiles`, `OverwriteFiles`, and `RowDelta`. Only the calling operation knows
    /// the requested path set, so the missing-path check must happen here. A present-and-absent mix
    /// errors rather than silently dropping the present file.
    ///
    /// **Per-caller faithfulness.** `StreamingDelete` and `BaseOverwriteFiles` both call
    /// `failMissingDeletePaths()`, so the loud failure is Java-faithful for them. Java's
    /// `BaseRowDelta` does NOT set the flag for `removeRows`, so this path is STRICTER than Java for
    /// that caller. It is the same conservative posture as
    /// [`Self::resolve_delete_file_paths`].
    pub(crate) async fn resolve_delete_paths(
        &self,
        delete_paths: &HashSet<String>,
    ) -> Result<Vec<DataFile>> {
        if delete_paths.is_empty() {
            return Ok(vec![]);
        }

        let mut resolved = Vec::new();
        let mut found_paths: HashSet<String> = HashSet::new();
        if let Some(snapshot) = self.table.metadata().current_snapshot() {
            let manifest_list = snapshot
                .load_manifest_list(self.table.file_io(), &self.table.metadata_ref())
                .await?;

            for manifest_file in manifest_list.entries() {
                if manifest_file.content != ManifestContentType::Data {
                    continue;
                }
                let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
                for entry in manifest.entries() {
                    if entry.is_alive() && delete_paths.contains(entry.file_path()) {
                        found_paths.insert(entry.file_path().to_string());
                        resolved.push(entry.data_file().clone());
                    }
                }
            }
        }

        let missing: Vec<&str> = delete_paths
            .iter()
            .map(String::as_str)
            .filter(|path| !found_paths.contains(*path))
            .collect();
        if !missing.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Missing required files to delete: {}", missing.join(", ")),
            ));
        }

        Ok(resolved)
    }

    /// Resolve `delete_paths` against the current snapshot's live DELETE entries, returning the matching
    /// [`DataFile`]s, and fail if any requested path matched no live delete entry — the DELETE-manifest
    /// sibling of [`SnapshotProducer::resolve_delete_paths`] (Java
    /// `MergingSnapshotProducer.delete(DeleteFile)` → `deleteFilterManager.delete(file)` resolved at
    /// `filterManifests` time).
    ///
    /// Scans every current DELETE manifest, never data manifests, and collects each live entry whose
    /// path is in `delete_paths`. The missing-path check mirrors `resolve_delete_paths`.
    ///
    /// **Posture: stricter than Java's `RowDelta.removeDeletes` default.** Java fails only when
    /// `failMissingDeletePaths` is set, and `RowDelta` does not set it. This port always fails loud,
    /// as `process_deletes` already does for removed DATA files. One consequence: a retry whose
    /// target delete file was concurrently removed fails loud where Java would converge. That is the
    /// safe direction, and it is accepted.
    pub(crate) async fn resolve_delete_file_paths(
        &self,
        delete_paths: &HashSet<String>,
    ) -> Result<Vec<DataFile>> {
        if delete_paths.is_empty() {
            return Ok(vec![]);
        }

        let mut resolved = Vec::new();
        let mut found_paths: HashSet<String> = HashSet::new();
        if let Some(snapshot) = self.table.metadata().current_snapshot() {
            let manifest_list = snapshot
                .load_manifest_list(self.table.file_io(), &self.table.metadata_ref())
                .await?;

            for manifest_file in manifest_list.entries() {
                if manifest_file.content != ManifestContentType::Deletes {
                    continue;
                }
                let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
                for entry in manifest.entries() {
                    if entry.is_alive() && delete_paths.contains(entry.file_path()) {
                        found_paths.insert(entry.file_path().to_string());
                        resolved.push(entry.data_file().clone());
                    }
                }
            }
        }

        let missing: Vec<&str> = delete_paths
            .iter()
            .map(String::as_str)
            .filter(|path| !found_paths.contains(*path))
            .collect();
        if !missing.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Missing required files to delete: {}", missing.join(", ")),
            ));
        }

        Ok(resolved)
    }

    /// Resolve a set of `(partition_spec_id, partition)` tuples against the current snapshot's live data
    /// entries, returning every matching [`DataFile`] (the ones a partition-scoped replace removes).
    ///
    /// The by-PARTITION sibling of the by-PATH [`SnapshotProducer::resolve_delete_paths`], used by
    /// `ReplacePartitions`. Java `ManifestFilterManager.filterManifestWithDeletedFiles`. The
    /// resolved files feed the same `process_deletes` machinery, so the rewrite and
    /// provenance-preservation logic is reused unchanged.
    ///
    /// There is deliberately NO missing-target validation. Java's `failMissingDeletePaths` guards
    /// only path deletes, never partition drops, so replacing an empty partition is a pure add.
    pub(crate) async fn resolve_partition_deletes(
        &self,
        drop_partitions: &HashSet<(i32, Struct)>,
    ) -> Result<Vec<DataFile>> {
        if drop_partitions.is_empty() {
            return Ok(vec![]);
        }

        let mut resolved = Vec::new();
        if let Some(snapshot) = self.table.metadata().current_snapshot() {
            let manifest_list = snapshot
                .load_manifest_list(self.table.file_io(), &self.table.metadata_ref())
                .await?;

            for manifest_file in manifest_list.entries() {
                if manifest_file.content != ManifestContentType::Data {
                    continue;
                }
                let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
                for entry in manifest.entries() {
                    if !entry.is_alive() {
                        continue;
                    }
                    let data_file = entry.data_file();
                    let key = (data_file.partition_spec_id, data_file.partition().clone());
                    if drop_partitions.contains(&key) {
                        resolved.push(data_file.clone());
                    }
                }
            }
        }

        Ok(resolved)
    }

    /// Resolve live data files the row predicate strictly matches. Metrics run on the
    /// per-partition residual, not the full predicate: a partition-column filter with no
    /// bounds residualizes to `alwaysTrue` and deletes the file.
    ///
    /// | eval | action |
    /// |---|---|
    /// | inclusive says no rows match | keep |
    /// | strict says all rows match | delete |
    /// | partial | non-retryable error |
    pub(crate) async fn resolve_filter_deletes(
        &self,
        predicate: &Predicate,
        case_sensitive: bool,
    ) -> Result<Vec<DataFile>> {
        let Some(snapshot) = self.table.metadata().current_snapshot() else {
            return Ok(vec![]);
        };

        let schema = self.table.metadata().current_schema().clone();
        // Bind the row predicate to the table schema once (Java `deleteExpression`). `rewrite_not` first so
        // the projection / residual visitors never see a `Not` (they reject it). `case_sensitive` mirrors
        // Java `ManifestFilterManager.caseSensitive` (column-name resolution case sensitivity).
        let bound_predicate = predicate
            .clone()
            .rewrite_not()
            .bind(schema.clone(), case_sensitive)?;

        // Per-partition-spec cache of the residual evaluator (Java's per-spec `PartitionAndMetricsEvaluator`).
        let mut residual_evaluators: HashMap<i32, ResidualEvaluator> = HashMap::new();

        let manifest_list = snapshot
            .load_manifest_list(self.table.file_io(), &self.table.metadata_ref())
            .await?;

        let mut resolved = Vec::new();
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != ManifestContentType::Data {
                continue;
            }
            let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let data_file = entry.data_file();

                // Reduce the predicate to its residual for this file's partition (Java
                // `residualEvaluator.residualFor(partition)`), then bind the residual to the table schema for
                // the metrics evaluators. A new spec id builds (and caches) its residual evaluator.
                let spec_id = data_file.partition_spec_id;
                let residual_evaluator = match residual_evaluators.entry(spec_id) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        let evaluator = Self::build_residual_evaluator(
                            self.table,
                            &bound_predicate,
                            &schema,
                            spec_id,
                            case_sensitive,
                        )?;
                        e.insert(evaluator)
                    }
                };
                let residual = residual_evaluator
                    .residual_for(data_file.partition())?
                    .rewrite_not()
                    .bind(schema.clone(), case_sensitive)?;

                // 1. `rowsMightMatch` (Java L470, L592-596): no rows can match ⇒ KEEP.
                if !InclusiveMetricsEvaluator::eval(&residual, data_file, true)? {
                    continue;
                }

                // 2. `rowsMustMatch` (Java L471, L598-602): all rows match ⇒ DELETE.
                if StrictMetricsEvaluator::eval(&residual, data_file)? {
                    resolved.push(data_file.clone());
                    continue;
                }

                // 3. PARTIAL match: might-match but NOT strictly all ⇒ non-retryable error (Java L472-477).
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot delete file where some, but not all, rows match filter {predicate}: {}",
                        data_file.file_path()
                    ),
                ));
            }
        }

        Ok(resolved)
    }

    /// Build the [`ResidualEvaluator`] for `spec_id` from the bound row predicate (Java
    /// `ResidualEvaluator.of(spec, deleteExpression, caseSensitive)` inside `PartitionAndMetricsEvaluator`).
    /// An unpartitioned spec degrades to `ResidualEvaluator::unpartitioned` (every residual is the whole
    /// filter). `case_sensitive` is threaded from the action (Java `ManifestFilterManager.caseSensitive` →
    /// the `PartitionAndMetricsEvaluator`'s `ResidualEvaluator.of(..., caseSensitive)`).
    fn build_residual_evaluator(
        table: &Table,
        bound_predicate: &BoundPredicate,
        schema: &Schema,
        spec_id: i32,
        case_sensitive: bool,
    ) -> Result<ResidualEvaluator> {
        let partition_spec = table
            .metadata()
            .partition_spec_by_id(spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Cannot resolve filter deletes: unknown partition spec id {spec_id}"),
                )
            })?;

        ResidualEvaluator::of(
            partition_spec.clone(),
            schema,
            bound_predicate.clone(),
            case_sensitive,
        )
    }

    fn generate_unique_snapshot_id(table: &Table) -> i64 {
        let generate_random_id = || -> i64 {
            let (lhs, rhs) = Uuid::new_v4().as_u64_pair();
            let snapshot_id = (lhs ^ rhs) as i64;
            if snapshot_id < 0 {
                -snapshot_id
            } else {
                snapshot_id
            }
        };
        let mut snapshot_id = generate_random_id();

        while table
            .metadata()
            .snapshots()
            .any(|s| s.snapshot_id() == snapshot_id)
        {
            snapshot_id = generate_random_id();
        }
        snapshot_id
    }

    // Check if the partition value is compatible with the partition type.
    fn validate_partition_value(
        partition_value: &Struct,
        partition_type: &StructType,
    ) -> Result<()> {
        if partition_value.fields().len() != partition_type.fields().len() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Partition value is not compatible with partition type",
            ));
        }

        for (value, field) in partition_value.fields().iter().zip(partition_type.fields()) {
            let field = field.field_type.as_primitive_type().ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Partition field should only be primitive type.",
                )
            })?;
            if let Some(value) = value {
                // A non-primitive literal in a primitive-typed partition slot is caller-supplied
                // invalid data: surface it as a typed error (Java's typed `PartitionData` accessors
                // throw `IllegalArgumentException` for a wrong-kind value — never an abort).
                let primitive_literal = value.as_primitive_literal().ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Partition value must be a primitive literal, got `{value:?}`"),
                    )
                })?;
                if !field.compatible(&primitive_literal) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        "Partition value is not compatible partition type",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Group the added files by their own `partition_spec_id`, in spec-id DESCENDING order. Java
    /// `groupBySpec` uses a reverse-ordered `TreeMap`, and the sibling actions already reverse-sort.
    ///
    /// Java's own added-file grouping is a `HashMap`, whose iteration order is undefined. The group
    /// order never reaches the spec-canonical metadata view, because a manifest list compares as a
    /// set. A stable order makes the on-disk manifest list reproducible across runs.
    fn group_files_by_spec(files: Vec<DataFile>) -> Vec<(i32, Vec<DataFile>)> {
        let mut groups: HashMap<i32, Vec<DataFile>> = HashMap::new();
        for file in files {
            groups.entry(file.partition_spec_id).or_default().push(file);
        }
        let mut groups: Vec<(i32, Vec<DataFile>)> = groups.into_iter().collect();
        // Reverse spec-id order (Java `groupBySpec` TreeMap with `Comparator.reverseOrder()`).
        groups.sort_unstable_by(|(left, _), (right, _)| right.cmp(left));
        groups
    }

    // Write one DATA manifest per partition-spec group of the added data files, returning the
    // ManifestFiles for the ManifestList (Java `MergingSnapshotProducer.newDataFilesAsManifests`:
    // `newDataFilesBySpec.forEach((specId, files) -> writeDataManifests(files, ..., spec(specId)))`).
    // Each group's entries are written by a writer built under THAT group's spec, so a file added under an
    // older spec keeps its own spec id / partition type instead of being stamped under the default.
    async fn write_added_manifests(&mut self) -> Result<Vec<ManifestFile>> {
        let added_data_files = std::mem::take(&mut self.added_data_files);
        if added_data_files.is_empty() {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "No added data files found when write an added manifest file",
            ));
        }

        let snapshot_id = self.snapshot_id;
        let format_version = self.table.metadata().format_version();
        // When set (the `RewriteFiles.dataSequenceNumber` preservation path, Java
        // `newDataFilesDataSequenceNumber`), every added data entry carries this EXPLICIT data sequence
        // number so the manifest writer keeps it (mirrors Java `writeDataFileGroup` calling
        // `writer.add(file, dataSeq)` instead of `writer.add(file)`). V2/V3 only — V1 manifests carry no
        // sequence numbers, so on V1 this is ignored and the added entry just stamps the snapshot id.
        let new_data_seq = self.new_data_files_data_sequence_number;

        let mut manifest_files = Vec::new();
        for (partition_spec_id, group) in Self::group_files_by_spec(added_data_files) {
            let mut writer =
                self.new_cluster_manifest_writer(partition_spec_id, ManifestContentType::Data)?;
            for data_file in group {
                let builder = ManifestEntry::builder()
                    .status(crate::spec::ManifestStatus::Added)
                    .data_file(data_file);
                let entry = if format_version == FormatVersion::V1 {
                    builder.snapshot_id(snapshot_id).build()
                } else if let Some(sequence_number) = new_data_seq {
                    // Preserve the explicit data sequence number on the added entry (Java
                    // `writeDataFileGroup` with a non-null `dataSeq`). The writer keeps a non-negative
                    // explicit data seq and lets the FILE sequence number inherit at read time — matching
                    // Java `wrapAppend(snapshotId, dataSeq, file)` with a null file seq.
                    builder.sequence_number(sequence_number).build()
                } else {
                    // For format version > 1, set the snapshot id at inherited time to avoid rewriting
                    // the manifest file when a commit fails.
                    builder.build()
                };
                writer.add_entry(entry)?;
            }
            manifest_files.push(writer.write_manifest_file().await?);
        }
        Ok(manifest_files)
    }

    /// Write one DELETE manifest per partition-spec group of the added delete files, returning the
    /// [`ManifestFile`]s for the manifest list. Mirrors [`write_added_manifests`](Self::write_added_manifests)
    /// but uses the `Deletes` cluster writer (Java `MergingSnapshotProducer.newDeleteFilesAsManifests`:
    /// `newDeleteFilesBySpec.forEach((specId, files) -> writeDeleteManifests(files, spec(specId)))`).
    ///
    /// Each added delete file carries an OPTIONAL explicit DATA sequence number, and Java
    /// `writeDeleteFileGroup` branches per file:
    ///
    /// - `None` writes an `Added` entry with no sequence number, so it inherits the new snapshot's
    ///   at read time. This is the merge-on-read default.
    /// - `Some(seq)` writes that explicit data seq, so a rewritten delete file keeps the seq it
    ///   needs to still apply to its data. The FILE sequence number still inherits.
    ///
    /// A V1 table has no delete manifests, so the explicit seq is ignored there. Each group's writer
    /// is built under THAT group's spec, so a delete file under an older spec keeps its own spec.
    async fn write_added_delete_manifests(&mut self) -> Result<Vec<ManifestFile>> {
        let added_delete_files = std::mem::take(&mut self.added_delete_files);
        if added_delete_files.is_empty() {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "No added delete files found when writing an added delete manifest file",
            ));
        }

        let snapshot_id = self.snapshot_id;
        let format_version = self.table.metadata().format_version();

        let mut manifest_files = Vec::new();
        for (partition_spec_id, group) in Self::group_delete_files_by_spec(added_delete_files) {
            let mut writer =
                self.new_cluster_manifest_writer(partition_spec_id, ManifestContentType::Deletes)?;
            for (delete_file, explicit_seq) in group {
                let builder = ManifestEntry::builder()
                    .status(crate::spec::ManifestStatus::Added)
                    .data_file(delete_file);
                let entry = if format_version == FormatVersion::V1 {
                    // Position/equality deletes are V2+ concepts; a V1 table has no delete manifests, and
                    // V1 manifests carry no sequence numbers — so the explicit seq is irrelevant here.
                    builder.snapshot_id(snapshot_id).build()
                } else if let Some(sequence_number) = explicit_seq {
                    // Explicit per-file data seq (Java `writeDeleteFileGroup`'s `writer.add(file, dataSeq)`
                    // when the pending delete file's `dataSequenceNumber()` is non-null). The writer keeps a
                    // non-negative explicit data seq and lets the FILE sequence number inherit at read time.
                    builder.sequence_number(sequence_number).build()
                } else {
                    // For format version > 1, set the snapshot id + sequence number at inherited time so
                    // the manifest does not need rewriting on a commit retry (same as added data files).
                    builder.build()
                };
                writer.add_entry(entry)?;
            }
            manifest_files.push(writer.write_manifest_file().await?);
        }
        Ok(manifest_files)
    }

    /// Group the added delete files (each paired with its optional explicit data seq) by their own
    /// `partition_spec_id`, in the same DETERMINISTIC spec-id-DESCENDING order as
    /// [`group_files_by_spec`](Self::group_files_by_spec) (the data-file sibling). Carries each file's
    /// explicit seq through the grouping so [`write_added_delete_manifests`](Self::write_added_delete_manifests)
    /// can stamp it per Java `writeDeleteFileGroup`.
    fn group_delete_files_by_spec(
        files: Vec<PendingDeleteFile>,
    ) -> Vec<(i32, Vec<PendingDeleteFile>)> {
        let mut groups: HashMap<i32, Vec<PendingDeleteFile>> = HashMap::new();
        for (file, explicit_seq) in files {
            groups
                .entry(file.partition_spec_id)
                .or_default()
                .push((file, explicit_seq));
        }
        let mut groups: Vec<(i32, Vec<PendingDeleteFile>)> = groups.into_iter().collect();
        // Reverse spec-id order (Java `groupBySpec` TreeMap with `Comparator.reverseOrder()`).
        groups.sort_unstable_by(|(left, _), (right, _)| right.cmp(left));
        groups
    }

    async fn manifest_file<OP: SnapshotProduceOperation, MP: ManifestProcess>(
        &mut self,
        snapshot_produce_operation: &OP,
        manifest_process: &MP,
    ) -> Result<Vec<ManifestFile>> {
        // The files to remove were resolved in `commit()` (before `summary()`, so the summary can reflect
        // the deletes) and stored in `self.removed_data_files` / `self.removed_delete_files`. Take both
        // here and pass them as ONE set to `process_deletes`, which matches by path across the full
        // manifest list (DATA and DELETE manifests): a removed DATA file's tombstone lands in the
        // rewritten DATA manifest, a removed DELETE file's in the rewritten DELETE manifest — the Rust
        // analogue of Java composing `filterManager.filterManifests(dataManifests)` AND
        // `deleteFilterManager.filterManifests(deleteManifests)` (`MergingSnapshotProducer.apply` L977-1000).
        let mut delete_files = std::mem::take(&mut self.removed_data_files);
        let removed_delete_files = std::mem::take(&mut self.removed_delete_files);
        delete_files.extend(removed_delete_files);

        // Assert the new snapshot contributes content: added data files, added DELETE files, removed
        // (deleted) data or delete files, or added snapshot properties. An add-deletes-only commit (delete
        // files, no data files) is allowed (the merge-on-read `RowDelta` path); a delete-only data commit
        // (rewrite data manifests, no adds) is allowed; a remove-deletes-only commit (drop a superseded
        // delete file, no adds) is allowed; a truly-empty commit is not.
        //
        // TODO: Allowing snapshot property setup with no added data files is a workaround.
        // We should clean it up after all necessary actions are supported.
        // For details, please refer to https://github.com/apache/iceberg-rust/issues/1548
        if self.added_data_files.is_empty()
            && self.added_delete_files.is_empty()
            && delete_files.is_empty()
            && self.snapshot_properties.is_empty()
        {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "No added data files, added delete files, deleted data files, or added snapshot properties found when write a manifest file",
            ));
        }

        let existing_manifests = snapshot_produce_operation.existing_manifest(self).await?;

        // Rewrite existing manifests to remove the requested deletes (Java
        // `ManifestFilterManager.filterManifests`). Manifests that contain none of the target files are
        // carried forward unchanged.
        let processed = self
            .process_deletes(existing_manifests, &delete_files)
            .await?;
        let (existing_data_manifests, existing_delete_manifests): (Vec<_>, Vec<_>) = processed
            .into_iter()
            .partition(|manifest| manifest.content == ManifestContentType::Data);

        let mut manifest_files =
            Vec::with_capacity(existing_data_manifests.len() + existing_delete_manifests.len() + 2);

        // Process added data entries — ONE DATA manifest per partition-spec group (Java
        // `newDataFilesBySpec.forEach(writeDataManifests)`), so a file under an older spec keeps its own
        // spec id rather than being stamped under the table default.
        if !self.added_data_files.is_empty() {
            let added_manifests = self.write_added_manifests().await?;
            manifest_files.extend(added_manifests);
        }
        // DATA ORDER IS LOAD-BEARING, in both halves of this line (Java
        // `MergingSnapshotProducer.apply`: `concat(prepareNewDataManifests(), filteredExistingData)`).
        // The V3 manifest-list writer gives each unassigned DATA manifest its `first_row_id` range in
        // list order. Put the added group second and a new file takes a row id Java does not give it;
        // reorder the existing group and two carried-forward manifests that still need ranges swap
        // theirs.
        manifest_files.extend(existing_data_manifests);

        // Process added DELETE entries — ONE DELETE manifest per partition-spec group (Java
        // `newDeleteFilesBySpec.forEach(writeDeleteManifests)`) for merge-on-read deletes added now.
        if !self.added_delete_files.is_empty() {
            let added_delete_manifests = self.write_added_delete_manifests().await?;
            manifest_files.extend(added_delete_manifests);
        }
        // The data-before-deletes split is `MergingSnapshotProducer.apply`'s shape ALONE. Java
        // `FastAppend.apply` and `BaseRewriteManifests.apply` append the prior list un-split. The
        // difference is unreachable: only a merging producer can introduce a delete manifest, and it
        // appends its delete group last, so every list either engine writes is already data-then-
        // deletes and the split is a no-op there.
        manifest_files.extend(existing_delete_manifests);

        let manifest_files = manifest_process
            .process_manifests(self, manifest_files)
            .await?;
        Ok(manifest_files)
    }

    /// Rewrite the existing manifests to remove `delete_files`, mirroring Java
    /// `ManifestFilterManager.filterManifests` + `MergingSnapshotProducer.apply`'s keep rule.
    ///
    /// A manifest holding a live entry whose path is in `delete_files` is rewritten. Matching live
    /// entries become `Deleted`, keeping their data file and both sequence numbers. Every other live
    /// entry is copied forward as `Existing`, keeping its snapshot id and both sequence numbers.
    /// Every other manifest carries forward unchanged.
    ///
    /// A rewritten manifest is kept even when every live entry became `Deleted`. An unrewritten
    /// manifest with no live files is dropped.
    ///
    /// # Errors
    ///
    /// A requested delete path matched no live entry. Java `failMissingDeletePaths`.
    async fn process_deletes(
        &mut self,
        existing_manifests: Vec<ManifestFile>,
        delete_files: &[DataFile],
    ) -> Result<Vec<ManifestFile>> {
        if delete_files.is_empty() {
            return Ok(existing_manifests);
        }

        let delete_paths: HashSet<&str> = delete_files
            .iter()
            .map(|df| df.file_path.as_str())
            .collect();

        // Track which requested paths were actually removed, to validate that none was missing.
        let mut deleted_paths: HashSet<String> = HashSet::new();
        let mut result_manifests = Vec::with_capacity(existing_manifests.len());

        for manifest_file in existing_manifests {
            let manifest = manifest_file.load_manifest(self.table.file_io()).await?;

            // Does any live entry in this manifest target one of the files to delete?
            let has_matching_delete = manifest
                .entries()
                .iter()
                .any(|entry| entry.is_alive() && delete_paths.contains(entry.file_path()));

            if !has_matching_delete {
                // Carry the manifest forward unchanged unless it has no live files at all.
                if manifest_file.has_added_files() || manifest_file.has_existing_files() {
                    result_manifests.push(manifest_file);
                }
                continue;
            }

            let rewritten = self
                .rewrite_manifest_with_deletes(
                    &manifest_file,
                    &manifest,
                    &delete_paths,
                    &mut deleted_paths,
                )
                .await?;
            result_manifests.push(rewritten);
        }

        // Validate that every requested delete path was found in a live entry (Java
        // `failMissingDeletePaths`).
        let missing: Vec<&str> = delete_paths
            .iter()
            .filter(|path| !deleted_paths.contains(**path))
            .copied()
            .collect();
        if !missing.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Missing required files to delete: {}", missing.join(", ")),
            ));
        }

        Ok(result_manifests)
    }

    /// Write a rewritten copy of `manifest` with the entries in `delete_paths` marked `Deleted` and the
    /// rest copied forward as `Existing`. Records each removed path in `deleted_paths`.
    async fn rewrite_manifest_with_deletes(
        &mut self,
        manifest_file: &ManifestFile,
        manifest: &Manifest,
        delete_paths: &HashSet<&str>,
        deleted_paths: &mut HashSet<String>,
    ) -> Result<ManifestFile> {
        // Rewrite with the source manifest's own partition spec so the spec id / partition type of the
        // copied-forward entries is preserved (Java writes with `reader.spec()`).
        let mut writer = self.new_filtering_manifest_writer(manifest_file)?;

        for entry in manifest.entries() {
            // Already-deleted entries are informational only and are not carried forward.
            if !entry.is_alive() {
                continue;
            }

            let entry = entry.as_ref().clone();
            if delete_paths.contains(entry.file_path()) {
                deleted_paths.insert(entry.file_path().to_string());
                writer.add_delete_entry(entry)?;
            } else {
                writer.add_existing_entry(entry)?;
            }
        }

        writer.write_manifest_file().await
    }

    /// Build a manifest writer for a rewritten (filtered) manifest, using the partition spec of the
    /// source manifest so existing entries keep their spec id and partition type.
    ///
    /// **Content-keyed. Read this if you touch it.** The writer's CONTENT matches the SOURCE
    /// manifest's. A rewritten DELETE manifest MUST stay a DELETE manifest, or the manifest list
    /// misclassifies it and the read path stops applying its surviving deletes, which resurrects
    /// rows. Java keys the same choice on its filter manager. Mirroring it off
    /// `source_manifest.content` keeps one `process_deletes` path serving both.
    fn new_filtering_manifest_writer(
        &mut self,
        source_manifest: &ManifestFile,
    ) -> Result<ManifestWriter> {
        let partition_spec = self
            .table
            .metadata()
            .partition_spec_by_id(source_manifest.partition_spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot rewrite manifest: unknown partition spec id {}",
                        source_manifest.partition_spec_id
                    ),
                )
            })?
            .as_ref()
            .clone();

        let new_manifest_path = format!(
            "{}/{}/{}-m{}.{}",
            self.table.metadata().location(),
            META_ROOT_PATH,
            self.commit_uuid,
            self.manifest_counter.next().ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Exhausted manifest file name counter",
                )
            })?,
            DataFileFormat::Avro
        );
        let output_file = self.table.file_io().new_output(new_manifest_path)?;
        let builder = ManifestWriterBuilder::new(
            output_file,
            Some(self.snapshot_id),
            self.key_metadata.clone(),
            self.table.metadata().current_schema().clone(),
            partition_spec,
        );
        match self.table.metadata().format_version() {
            FormatVersion::V1 => Ok(builder.build_v1()),
            FormatVersion::V2 => match source_manifest.content {
                ManifestContentType::Data => Ok(builder.build_v2_data()),
                ManifestContentType::Deletes => Ok(builder.build_v2_deletes()),
            },
            FormatVersion::V3 => match source_manifest.content {
                ManifestContentType::Data => Ok(builder.build_v3_data()),
                ManifestContentType::Deletes => Ok(builder.build_v3_deletes()),
            },
        }
    }

    /// Resolve the partition spec a summarized file belongs to, for the per-file partition-summary
    /// path. Java `SnapshotSummary.Builder` uses `spec(file.specId())`. It falls back to the table
    /// default only when the file's spec is absent. That is unreachable on the ADD path, and
    /// REACHABLE for the removed-file loops, whose files can name a spec `remove_partition_specs`
    /// dropped.
    ///
    /// The substituted spec can therefore have a different arity than the file's tuple. That is safe
    /// ONLY because [`PartitionSpec::partition_to_path`] is total: it renders unmatched fields as
    /// `null` and warns. Before that it aborted mid-commit.
    fn file_partition_spec(&self, file: &DataFile) -> crate::spec::PartitionSpecRef {
        let metadata = self.table.metadata();
        metadata
            .partition_spec_by_id(file.partition_spec_id)
            .cloned()
            .unwrap_or_else(|| metadata.default_partition_spec().clone())
    }

    // Returns a `Summary` of the current snapshot
    fn summary<OP: SnapshotProduceOperation>(
        &self,
        snapshot_produce_operation: &OP,
    ) -> Result<Summary> {
        let mut summary_collector = SnapshotSummaryCollector::default();
        let table_metadata = self.table.metadata_ref();

        let partition_summary_limit = if let Some(limit) = table_metadata
            .properties()
            .get(TableProperties::PROPERTY_WRITE_PARTITION_SUMMARY_LIMIT)
        {
            if let Ok(limit) = limit.parse::<u64>() {
                limit
            } else {
                TableProperties::PROPERTY_WRITE_PARTITION_SUMMARY_LIMIT_DEFAULT
            }
        } else {
            TableProperties::PROPERTY_WRITE_PARTITION_SUMMARY_LIMIT_DEFAULT
        };

        summary_collector.set_partition_summary_limit(partition_summary_limit);

        // Each file is summarized under its OWN partition spec, not the table default (Java
        // `SnapshotSummary.Builder.addedFile(spec(file.specId()), file)` →
        // `updatePartitions(spec, file)` computes the partition path with the file's own spec). On a
        // multi-spec commit (a file added under an older spec) using the default spec would compute the
        // WRONG partition path and corrupt the changed-partition summaries. `file_partition_spec` falls
        // back to the default for the "spec vanished" case — unreachable HERE (validation already proved
        // every added file's spec exists) but reachable in the `removed_*_files` loops below, which
        // bypass `validate_partition_value`. This whole path is infallible, so it stays total only
        // because `partition_to_path` is (see `file_partition_spec`).
        for data_file in &self.added_data_files {
            summary_collector.add_file(
                data_file,
                table_metadata.current_schema().clone(),
                self.file_partition_spec(data_file),
            );
        }

        // Reflect added DELETE files (position / equality) in the summary. `add_file` branches on the
        // file's content type and increments the added-delete-file + added-position/equality-delete
        // counters (Java `MergingSnapshotProducer.add(DeleteFile)` → the delete-file summary). Empty for
        // operations that add no delete files. Summarized under each delete file's OWN spec.
        for (delete_file, _explicit_seq) in &self.added_delete_files {
            summary_collector.add_file(
                delete_file,
                table_metadata.current_schema().clone(),
                self.file_partition_spec(delete_file),
            );
        }

        // Reflect deleted files/records in the summary (Java overwrite/delete summary). `removed_data_files`
        // is populated in `commit()` (the resolved delete set) before `summary()` is called; it is empty
        // for add-only operations such as fast append, so this loop is a no-op there. Summarized under each
        // removed file's OWN spec (a removed file may belong to an older spec than the table default).
        // NOTE: unlike the add loops, nothing validated these tuples against their spec — this loop is
        // the reachable half of `file_partition_spec`'s substitution (WG3-L2 test
        // `test_summary_survives_a_removed_file_under_a_substituted_spec`).
        for data_file in &self.removed_data_files {
            summary_collector.remove_file(
                data_file,
                table_metadata.current_schema().clone(),
                self.file_partition_spec(data_file),
            );
        }

        // Reflect removed DELETE files (position / equality / DV) in the summary. `remove_file` branches
        // on content type: a removed DV increments `removed-dvs` (D3's reachable-end-to-end branch — this
        // is the path that makes it live), a removed parquet position delete increments
        // `removed-position-delete-files`, an equality delete `removed-equality-delete-files` (Java
        // `SnapshotSummary.UpdateMetrics.removedFile`). `removed_delete_files` is populated in `commit()`
        // (the resolved removal set) before `summary()`; empty for every operation that removes no delete
        // files, so this loop is a no-op there. Summarized under each removed delete file's OWN spec.
        for delete_file in &self.removed_delete_files {
            summary_collector.remove_file(
                delete_file,
                table_metadata.current_schema().clone(),
                self.file_partition_spec(delete_file),
            );
        }

        // The previous snapshot is the current branch head (the parent of the snapshot being produced):
        // at summary time the new snapshot is not yet in `table_metadata`, so its totals are seeded from
        // the current snapshot's summary. Mirrors Java `SnapshotProducer.summary(previous)` which reads
        // `previous.snapshot(previousBranchHead.snapshotId()).summary()`. (Looking up `self.snapshot_id`
        // here would always miss — the new snapshot does not exist yet — leaving totals seeded from zero,
        // which underflows the moment an operation removes more files than it adds.)
        let previous_snapshot = table_metadata.current_snapshot();

        let mut additional_properties = summary_collector.build();
        additional_properties.extend(self.snapshot_properties.clone());

        let summary = Summary {
            operation: snapshot_produce_operation.operation(),
            additional_properties,
        };

        // Compute totals as previous + added - removed for ALL operations, mirroring Java
        // `SnapshotProducer.summary(previous)` (which calls `updateTotal` unconditionally and has NO
        // full-table-truncate branch). `OverwriteFilesAction` is a PARTIAL overwrite (delete some, add
        // some), so its totals must NOT be reset to zero. The Rust-specific `truncate_full_table` path
        // (which zeroes totals + reports every prior file as deleted) is for a future full-table
        // replace/truncate action, not for a partial overwrite — so pass `false` here.
        update_snapshot_summaries(summary, previous_snapshot.map(|s| s.summary()), false)
    }

    fn generate_manifest_list_file_path(&self, attempt: i64) -> String {
        format!(
            "{}/{}/snap-{}-{}-{}.{}",
            self.table.metadata().location(),
            META_ROOT_PATH,
            self.snapshot_id,
            attempt,
            self.commit_uuid,
            DataFileFormat::Avro
        )
    }

    /// Finished building the action and return the [`ActionCommit`] to the transaction.
    pub(crate) async fn commit<OP: SnapshotProduceOperation, MP: ManifestProcess>(
        mut self,
        snapshot_produce_operation: OP,
        process: MP,
    ) -> Result<ActionCommit> {
        // Resolve the data files this operation removes up front (before `summary()`), so the snapshot
        // summary can reflect the deleted file/record counts and `manifest_file()` can reuse the result
        // without re-resolving. Empty for add-only operations (e.g. fast append).
        self.removed_data_files = snapshot_produce_operation.delete_files(&self).await?;

        // Resolve the DELETE files this operation removes against the current snapshot's DELETE manifests by
        // path (the apply-side `RowDelta.removeDeletes` path). Re-binding `self.removed_delete_files` to the
        // RESOLVED set (a) validates every requested removal is a live delete file (missing path fails loud)
        // and (b) replaces the caller-supplied (possibly-stale) `DataFile`s with the ON-DISK entries, so the
        // summary's `remove_file` reads the committed metadata. Empty when no delete files are removed.
        if !self.removed_delete_files.is_empty() {
            let requested_paths: HashSet<String> = self
                .removed_delete_files
                .iter()
                .map(|file| file.file_path().to_string())
                .collect();
            self.removed_delete_files = self.resolve_delete_file_paths(&requested_paths).await?;
        }

        let manifest_list_path = self.generate_manifest_list_file_path(0);
        let next_seq_num = self.table.metadata().next_sequence_number();
        let first_row_id = self.table.metadata().next_row_id();
        let mut manifest_list_writer = match self.table.metadata().format_version() {
            FormatVersion::V1 => ManifestListWriter::v1(
                self.table
                    .file_io()
                    .new_output(manifest_list_path.clone())?,
                self.snapshot_id,
                self.table.metadata().current_snapshot_id(),
            ),
            FormatVersion::V2 => ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(manifest_list_path.clone())?,
                self.snapshot_id,
                self.table.metadata().current_snapshot_id(),
                next_seq_num,
            ),
            FormatVersion::V3 => ManifestListWriter::v3(
                self.table
                    .file_io()
                    .new_output(manifest_list_path.clone())?,
                self.snapshot_id,
                self.table.metadata().current_snapshot_id(),
                next_seq_num,
                Some(first_row_id),
            ),
        };

        // Calling self.summary() before self.manifest_file() is important because self.added_data_files
        // will be set to an empty vec after self.manifest_file() returns, resulting in an empty summary
        // being generated.
        let summary = self.summary(&snapshot_produce_operation).map_err(|err| {
            Error::new(ErrorKind::Unexpected, "Failed to create snapshot summary.").with_source(err)
        })?;

        let new_manifests = self
            .manifest_file(&snapshot_produce_operation, &process)
            .await?;

        manifest_list_writer.add_manifests(new_manifests.into_iter())?;
        let writer_next_row_id = manifest_list_writer.next_row_id();
        manifest_list_writer.close().await?;

        let commit_ts = chrono::Utc::now().timestamp_millis();
        let new_snapshot = Snapshot::builder()
            .with_manifest_list(manifest_list_path)
            .with_snapshot_id(self.snapshot_id)
            .with_parent_snapshot_id(self.table.metadata().current_snapshot_id())
            .with_sequence_number(next_seq_num)
            .with_summary(summary)
            .with_schema_id(self.table.metadata().current_schema_id())
            .with_timestamp_ms(commit_ts);

        let new_snapshot = if let Some(writer_next_row_id) = writer_next_row_id {
            let assigned_rows = writer_next_row_id - self.table.metadata().next_row_id();
            new_snapshot
                .with_row_range(first_row_id, assigned_rows)
                .build()
        } else {
            new_snapshot.build()
        };

        // The staged (WAP) commit emits ONLY `AddSnapshot` — no `SetSnapshotRef` — mirroring Java
        // `lambda$commit$2`: `stageOnly ? builder.addSnapshot(snapshot) : builder.setBranchSnapshot(...)`
        // (1.10.0 bytecode). Adding the snapshot WITHOUT a ref move leaves `current-snapshot-id`, the `main`
        // ref, and the snapshot-log unchanged (Java's `addSnapshot` touches none of them; the Rust
        // `TableMetadataBuilder::add_snapshot` matches — and `update_snapshot_log` adds no entry without a
        // `SetSnapshotRef(main)`). The non-staged commit adds the snapshot AND moves `main` to it.
        let mut updates = vec![TableUpdate::AddSnapshot {
            snapshot: new_snapshot,
        }];
        if !self.stage_only {
            updates.push(TableUpdate::SetSnapshotRef {
                ref_name: MAIN_BRANCH.to_string(),
                reference: SnapshotReference::new(
                    self.snapshot_id,
                    SnapshotRetention::branch(None, None, None),
                ),
            });
        }

        // The `main`-ref optimistic-concurrency guard is only meaningful when this commit moves `main`. A
        // staged commit moves no ref, so it emits only the table-uuid guard (it still requires the same
        // table — a staged snapshot added to a different table's metadata would be nonsense). This mirrors
        // Java deriving an `AssertRefSnapshotID` requirement only for the `SetSnapshotRef` update it emits
        // (`UpdateRequirements`); with no ref update there is no ref requirement.
        let mut requirements = vec![TableRequirement::UuidMatch {
            uuid: self.table.metadata().uuid(),
        }];
        if !self.stage_only {
            requirements.push(TableRequirement::RefSnapshotIdMatch {
                r#ref: MAIN_BRANCH.to_string(),
                snapshot_id: self.table.metadata().current_snapshot_id(),
            });
        }

        Ok(ActionCommit::new(updates, requirements))
    }
}

/// Render a delete file in Java's deletion-vector description format — the Rust port of
/// `ContentFileUtil.dvDesc` (`core/.../util/ContentFileUtil.java` L150-157, 1.10.0-bytecode-verified):
/// `DV{location=%s, offset=%s, length=%s, referencedDataFile=%s}`. Java formats the nullable
/// `contentOffset` / `contentSizeInBytes` / `referencedDataFile` with `%s`, which renders a missing
/// value as `null` — mirrored here (NOT Rust's `Some(..)`/`None` debug rendering), so the
/// gate/validation messages are byte-identical to Java's.
pub(crate) fn dv_desc(delete_file: &DataFile) -> String {
    fn opt_to_java<T: std::fmt::Display>(value: Option<T>) -> String {
        value.map_or_else(|| "null".to_string(), |v| v.to_string())
    }
    format!(
        "DV{{location={}, offset={}, length={}, referencedDataFile={}}}",
        delete_file.file_path(),
        opt_to_java(delete_file.content_offset()),
        opt_to_java(delete_file.content_size_in_bytes()),
        opt_to_java(delete_file.referenced_data_file()),
    )
}

/// The format-version gate for an added DELETE file. Java
/// `MergingSnapshotProducer.validateDeleteFileForVersion`.
///
/// - **V1:** delete files do not exist — `"Deletes are supported in V2 and above"`.
/// - **V2:** equality deletes OK; a position delete must NOT be a deletion vector
///   (`!ContentFileUtil.isDV`, i.e. not Puffin format) — `"Must not use DVs for position deletes
///   in V2: %s"` with [`dv_desc`].
/// - **V3 (and Java's V4):** equality deletes OK; a position delete MUST be a deletion vector —
///   `"Must use DVs for position deletes in V%s: %s"` with the format version + the file location.
///
/// Equality deletes are exempt at EVERY version >= 2. A wrongly-gated DV commit corrupts
/// merge-on-read tables for every engine. A V2 reader cannot load a Puffin DV, and a V3 table
/// mixing fresh parquet position deletes with DVs breaks the read precedence.
///
/// # Errors
///
/// A NON-retryable [`ErrorKind::DataInvalid`], so the commit retry loop stops.
fn validate_delete_file_for_version(
    delete_file: &DataFile,
    format_version: FormatVersion,
) -> Result<()> {
    use crate::delete_file_index::is_deletion_vector;
    use crate::spec::DataContentType;

    let is_equality_delete = delete_file.content_type() == DataContentType::EqualityDeletes;
    match format_version {
        FormatVersion::V1 => Err(Error::new(
            ErrorKind::DataInvalid,
            "Deletes are supported in V2 and above",
        )),
        FormatVersion::V2 => {
            if is_equality_delete || !is_deletion_vector(delete_file) {
                Ok(())
            } else {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Must not use DVs for position deletes in V2: {}",
                        dv_desc(delete_file)
                    ),
                ))
            }
        }
        FormatVersion::V3 => {
            if is_equality_delete || is_deletion_vector(delete_file) {
                Ok(())
            } else {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Must use DVs for position deletes in V{}: {}",
                        format_version as u8,
                        delete_file.file_path()
                    ),
                ))
            }
        }
    }
}

/// Operations whose snapshots can ADD data files — the only ones a "no conflicting data" validation needs
/// to inspect (Java `MergingSnapshotProducer.VALIDATE_ADDED_FILES_OPERATIONS = {APPEND, OVERWRITE}`). A
/// `Delete` / `Replace` snapshot never introduces brand-new conflicting rows.
fn operation_adds_data_files(operation: &Operation) -> bool {
    matches!(operation, Operation::Append | Operation::Overwrite)
}

/// Operations whose snapshots can ADD delete files — the only ones a "no conflicting delete" validation
/// needs to inspect (Java `MergingSnapshotProducer.VALIDATE_ADDED_DELETE_FILES_OPERATIONS = {OVERWRITE,
/// DELETE}`). Note this differs from [`operation_adds_data_files`] (`{APPEND, OVERWRITE}`): an `Append`
/// snapshot never adds delete files, while a `Delete` snapshot (a pure merge-on-read delete commit) does.
fn operation_adds_delete_files(operation: &Operation) -> bool {
    matches!(operation, Operation::Overwrite | Operation::Delete)
}

/// Operations whose snapshots can ADD deletion vectors — the op set the `validateAddedDVs` walk
/// inspects (Java `MergingSnapshotProducer.VALIDATE_ADDED_DVS_OPERATIONS = {OVERWRITE, DELETE,
/// REPLACE}`, L84-85; 1.10.0-bytecode-verified: `ImmutableSet.of("overwrite", "delete",
/// "replace")`).
///
/// This is STRICTLY WIDER than [`operation_adds_delete_files`]. A REPLACE compaction snapshot can
/// rewrite deletion vectors, so the DV conflict check must inspect REPLACE snapshots too. The
/// fork's own `rewrite_files` records `Operation::Replace`, so dropping it here would silently miss
/// a concurrent REPLACE snapshot that added a DV for the same referenced data file.
fn operation_adds_dvs(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Overwrite | Operation::Delete | Operation::Replace
    )
}

/// Enumerate the DELETE files ADDED after `starting_snapshot_id` by operations that can add
/// DELETION VECTORS. This is the walk behind `RowDelta`'s `validateAddedDVs`.
///
/// The [`files_after`] walk semantics match [`added_delete_files_after`]. The ONLY difference is
/// the op set: [`operation_adds_dvs`] adds `Replace`. The caller filters the result to DVs and
/// applies the conflict test, so the non-DV entries a REPLACE snapshot carries never collide.
///
/// There is no format-version guard, because Java's `validateAddedDVs` has none. The caller's
/// self-skip means the walk runs only when this operation adds DVs.
pub(crate) async fn added_dv_candidate_delete_files_after(
    table: &Table,
    starting_snapshot_id: Option<i64>,
) -> Result<Vec<DataFile>> {
    files_after(
        table,
        starting_snapshot_id,
        ManifestContentType::Deletes,
        operation_adds_dvs,
        ManifestStatus::Added,
    )
    .await
}

/// Operations whose snapshots can REMOVE data files — the only ones a "data files still exist" validation
/// needs to inspect (Java `MergingSnapshotProducer.VALIDATE_DATA_FILES_EXIST_OPERATIONS = {OVERWRITE,
/// REPLACE, DELETE}`). An `Append` snapshot never removes a live data file, so it is not inspected.
///
/// All three Java members are matched here. `Operation::Replace` is what
/// [`crate::transaction::rewrite_files`] commits, so omitting it would let a concurrent COMPACTION
/// snapshot's `Deleted` tombstones escape this walk. [`operation_adds_dvs`] includes `Replace` for
/// the same reason.
///
/// This is the `skipDeletes == false` variant. See
/// [`operation_removes_data_files_skip_deletes`] for the other.
fn operation_removes_data_files(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Overwrite | Operation::Replace | Operation::Delete
    )
}

/// The `skipDeletes == true` variant of [`operation_removes_data_files`] — the operations whose snapshots can
/// remove data files when DELETE-op snapshots are EXCLUDED (Java
/// `MergingSnapshotProducer.VALIDATE_DATA_FILES_EXIST_SKIP_DELETE_OPERATIONS = {OVERWRITE, REPLACE}`).
///
/// Java drops `DELETE` so a concurrent merge-on-read DELETE snapshot does not trip the files-exist
/// check. `BaseRowDelta` uses this set by DEFAULT. `REPLACE` is NOT dropped, and dropping it here
/// would be the corruption line: a concurrent compaction removes the data file a position delete
/// references, the row delta commits anyway, and the deleted rows are live again in the output.
fn operation_removes_data_files_skip_deletes(operation: &Operation) -> bool {
    matches!(operation, Operation::Overwrite | Operation::Replace)
}

/// Enumerate the files of manifest `content` that snapshots committed after `starting_snapshot_id`
/// recorded with status `status_to_keep`. It is the shared walk behind [`added_data_files_after`],
/// [`added_delete_files_after`] and [`deleted_data_files_after`].
///
/// The Rust port of Java `MergingSnapshotProducer.validationHistory`. It walks the parent chain of
/// `table`'s current snapshot, INCLUSIVE of that snapshot and EXCLUSIVE of `starting_snapshot_id`.
/// For each visited snapshot whose operation passes `operation_filter`, it keeps the manifests of
/// `content` that snapshot WROTE and collects every entry matching `status_to_keep`.
///
/// The `status_to_keep` axis selects the per-check entry filter:
///
/// - `ManifestStatus::Added`: files ADDED by concurrent snapshots, for the conflict checks.
/// - `ManifestStatus::Deleted`: files DELETED by them, for the `validateDataFilesExist` check.
///
/// A concurrent delete or overwrite records its removals as `Deleted` tombstones in a manifest it
/// wrote itself, so the `added_snapshot_id == snapshot_id` filter finds them.
///
/// # Notes
///
/// `starting_snapshot_id == None` validates from the beginning of history. The walk yields nothing
/// when the current snapshot already IS the starting one, or when the table has no snapshot.
async fn files_after(
    table: &Table,
    starting_snapshot_id: Option<i64>,
    content: ManifestContentType,
    operation_filter: fn(&Operation) -> bool,
    status_to_keep: ManifestStatus,
) -> Result<Vec<DataFile>> {
    let metadata = table.metadata();

    // The "parent" of the operation in Java terms: the current head of the refreshed base. If there is no
    // current snapshot, nothing has been added.
    let Some(mut current) = metadata.current_snapshot().cloned() else {
        return Ok(vec![]);
    };

    let mut collected = Vec::new();

    loop {
        // Java `ancestorsBetween` is EXCLUSIVE of the starting snapshot: stop before re-visiting it (and
        // never inspect the snapshot the operation started from — its files are part of the base, not a
        // concurrent commit).
        if Some(current.snapshot_id()) == starting_snapshot_id {
            break;
        }

        if operation_filter(&current.summary().operation) {
            let manifest_list = current
                .load_manifest_list(table.file_io(), metadata)
                .await?;
            for manifest_file in manifest_list.entries() {
                // Only manifests of the requested `content` that THIS snapshot wrote (Java
                // `manifest.snapshotId() == currentSnapshot.snapshotId()`) — carried-forward manifests
                // belong to older snapshots and their files were not added/removed since the starting
                // snapshot. A delete/overwrite's rewritten manifest (carrying its `Deleted` tombstones)
                // also has `added_snapshot_id == snapshot.snapshot_id()`, so it is included here.
                if manifest_file.content != content
                    || manifest_file.added_snapshot_id != current.snapshot_id()
                {
                    continue;
                }
                let manifest = manifest_file.load_manifest(table.file_io()).await?;
                for entry in manifest.entries() {
                    // Keep only entries of the requested status (the per-check axis): `Added` for the
                    // conflict checks (Java `ignoreDeleted().ignoreExisting()` keeping `Status.ADDED`) or
                    // `Deleted` for the files-exist check (Java `deletedDataFiles` keeping `Status.DELETED`,
                    // with `ignoreExisting()`). An `Existing` entry was added by an earlier snapshot and
                    // copied forward, so it is never the relevant status here.
                    if entry.status() == status_to_keep {
                        collected.push(entry.data_file().clone());
                    }
                }
            }
        }

        // Walk to the parent; stop at the root. A missing parent (dangling id) also terminates the walk,
        // mirroring Java `ancestorsOf` returning when `lookup.apply(parentId)` is null.
        match current.parent_snapshot_id() {
            Some(parent_id) => match metadata.snapshot_by_id(parent_id) {
                Some(parent) => current = parent.clone(),
                None => break,
            },
            None => break,
        }
    }

    Ok(collected)
}

/// Enumerate the DATA files ADDED to `table` by snapshots committed AFTER `starting_snapshot_id` — the
/// concurrent commits a serializable-isolation conflict check must inspect.
///
/// This is the Rust port of Java `MergingSnapshotProducer.addedDataFiles` + `validationHistory`
/// (`core/MergingSnapshotProducer.java`): the shared [`files_after`] walk over DATA manifests, gated
/// to the operations that can add data ([`operation_adds_data_files`] = Java
/// `VALIDATE_ADDED_FILES_OPERATIONS = {APPEND, OVERWRITE}`), keeping `ManifestStatus::Added` entries. See
/// [`files_after`] for the walk semantics (inclusive of the current snapshot, exclusive of the starting
/// snapshot, only manifests the snapshot itself wrote, only entries of the requested status).
///
/// This is the shared foundation the per-action data-file conflict validations (`ReplacePartitions`
/// `validateNoConflictingData`, `OverwriteFiles` / `RowDelta` `validateNoConflictingDataFiles`) build on.
pub(crate) async fn added_data_files_after(
    table: &Table,
    starting_snapshot_id: Option<i64>,
) -> Result<Vec<DataFile>> {
    files_after(
        table,
        starting_snapshot_id,
        ManifestContentType::Data,
        operation_adds_data_files,
        ManifestStatus::Added,
    )
    .await
}

/// Enumerate the DELETE files (position / equality deletes) ADDED to `table` by snapshots committed AFTER
/// `starting_snapshot_id` — the concurrent commits a `validateNoConflictingDeleteFiles` check must inspect.
///
/// The Rust port of Java `MergingSnapshotProducer.addedDeleteFiles`: the shared [`files_after`] walk
/// over DELETE manifests, gated by [`operation_adds_delete_files`], keeping `Added` entries.
///
/// Delete files do not exist before format version 2, so a V1 table returns an empty set without
/// walking the history.
///
/// **Documented over-scan.** Java also filters by the operation's `startingSequenceNumber`. This
/// port walks the snapshots alone and applies the metrics filter later in
/// [`validate_no_conflicting_added_delete_files`]. Omitting the refinement is CONSERVATIVE: it can
/// only consider more delete files, never fewer.
pub(crate) async fn added_delete_files_after(
    table: &Table,
    starting_snapshot_id: Option<i64>,
) -> Result<Vec<DataFile>> {
    // V2 guard (Java `addedDeleteFiles`: `base.formatVersion() < 2` ⇒ empty). Delete files don't exist in
    // V1, so there is nothing to enumerate and no history to walk.
    if table.metadata().format_version() < FormatVersion::V2 {
        return Ok(vec![]);
    }

    files_after(
        table,
        starting_snapshot_id,
        ManifestContentType::Deletes,
        operation_adds_delete_files,
        ManifestStatus::Added,
    )
    .await
}

/// Enumerate the DELETE files (position / equality deletes) ADDED to `table` by snapshots committed AFTER
/// `starting_snapshot_id`, PAIRED with each entry's data sequence number — the sequence-preserving sibling
/// of [`added_delete_files_after`].
///
/// [`added_delete_files_after`] strips the entry's sequence number, which is all the metrics-only
/// conflict checks need. Java `validateNoNewDeletesForDataFiles` compares each delete's data
/// sequence number against the operation's starting sequence number, so this variant preserves it.
///
/// The walk semantics match [`added_delete_files_after`]. The per-entry `Option<i64>` is the data
/// sequence number a V2 or V3 added delete inherits from its committing snapshot. It is always
/// above any pre-start data file's, so the partition match is the load-bearing test. The comparison
/// is preserved for faithfulness to Java.
async fn added_delete_files_with_seq_after(
    table: &Table,
    starting_snapshot_id: Option<i64>,
) -> Result<Vec<(DataFile, Option<i64>)>> {
    let metadata = table.metadata();

    // V2 guard (Java `addedDeleteFiles`: `base.formatVersion() < 2` ⇒ empty `DeleteFileIndex`).
    if metadata.format_version() < FormatVersion::V2 {
        return Ok(vec![]);
    }

    // The "parent" of the operation in Java terms: the current head of the refreshed base.
    let Some(mut current) = metadata.current_snapshot().cloned() else {
        return Ok(vec![]);
    };

    let mut collected = Vec::new();

    loop {
        // Java `ancestorsBetween` is EXCLUSIVE of the starting snapshot (mirrors [`files_after`]).
        if Some(current.snapshot_id()) == starting_snapshot_id {
            break;
        }

        if operation_adds_delete_files(&current.summary().operation) {
            let manifest_list = current
                .load_manifest_list(table.file_io(), metadata)
                .await?;
            for manifest_file in manifest_list.entries() {
                // Only DELETE manifests THIS snapshot wrote (Java `manifest.snapshotId() ==
                // currentSnapshot.snapshotId()`) — mirrors the manifest filter in [`files_after`].
                if manifest_file.content != ManifestContentType::Deletes
                    || manifest_file.added_snapshot_id != current.snapshot_id()
                {
                    continue;
                }
                let manifest = manifest_file.load_manifest(table.file_io()).await?;
                for entry in manifest.entries() {
                    if entry.status() == ManifestStatus::Added {
                        collected.push((entry.data_file().clone(), entry.sequence_number()));
                    }
                }
            }
        }

        // Walk to the parent; stop at the root or a dangling parent id (mirrors [`files_after`]).
        match current.parent_snapshot_id() {
            Some(parent_id) => match metadata.snapshot_by_id(parent_id) {
                Some(parent) => current = parent.clone(),
                None => break,
            },
            None => break,
        }
    }

    Ok(collected)
}

/// The sequence number of the snapshot the operation started from, or `0` if there is none. Java
/// `MergingSnapshotProducer.startingSequenceNumber`. The `0` literal is `INITIAL_SEQUENCE_NUMBER`,
/// inlined so the spec module's export surface stays narrow.
fn starting_sequence_number(table: &Table, starting_snapshot_id: Option<i64>) -> i64 {
    match starting_snapshot_id {
        Some(id) => table
            .metadata()
            .snapshot_by_id(id)
            .map_or(0, |snapshot| snapshot.sequence_number()),
        None => 0,
    }
}

/// Reject the commit if a DELETE file added since `starting_snapshot_id` applies to a DATA file this
/// operation REMOVES. Java `MergingSnapshotProducer.validateNoNewDeletesForDataFiles`. It is the
/// serializable-isolation guard: you cannot drop a data file out from under a concurrent row-level
/// delete.
///
/// No current snapshot, or a table below format version 2, means no delete file can exist, so the
/// check is a no-op. `bound_conflict_filter` narrows the concurrently-added deletes by metrics; a
/// `None` filter keeps every one, the conservative default.
///
/// A concurrently-added delete applies to a removed data file when both hold:
///
/// 1. Its data sequence number is `>= starting_sequence_number`. An added delete inherits its
///    snapshot's sequence number, so this is effectively always true. The test below is the
///    load-bearing one. The comparison is kept for faithfulness.
/// 2. It matches the data file: partition-scoped position deletes and equality deletes match on
///    `(spec_id, partition)`; a path-scoped position delete matches only that exact path; a global
///    equality delete matches ANY data file.
///
/// The applicability test is implemented DIRECTLY here rather than through
/// [`crate::delete_file_index`]. That index keys on SCAN-time semantics and compares against the
/// data file's OWN sequence number, while this validation compares against the operation's starting
/// sequence number.
///
/// `ignore_equality_deletes` makes only POSITION deletes count as a conflict.
/// `bound_conflict_filter` is bound by the CALLER, so the case sensitivity is the caller's and this
/// signature stays stable across the three actions.
///
/// # Errors
///
/// A NON-retryable [`ErrorKind::DataInvalid`] on the FIRST conflicting data file, carrying Java's
/// verbatim message, so the commit retry loop stops.
pub(crate) async fn validate_no_new_deletes_for_data_files(
    table: &Table,
    starting_snapshot_id: Option<i64>,
    bound_conflict_filter: Option<&BoundPredicate>,
    data_files: &[DataFile],
    ignore_equality_deletes: bool,
) -> Result<()> {
    // Java L526-528: no current table state (`parent == null`) or a pre-V2 table ⇒ no delete files exist.
    if table.metadata().current_snapshot().is_none()
        || table.metadata().format_version() < FormatVersion::V2
    {
        return Ok(());
    }

    // Java L530: the DELETE files concurrently added since the start (with their data sequence numbers).
    let added_deletes = added_delete_files_with_seq_after(table, starting_snapshot_id).await?;
    if added_deletes.is_empty() {
        return Ok(());
    }

    // A delete whose metrics cannot match the conflict filter cannot conflict. The CALLER binds the
    // filter, so this shared signature stays stable across the three actions.
    let bound_filter = bound_conflict_filter;

    // Java L533: the sequence number of the starting snapshot (or 0 if none).
    let starting_sequence_number = starting_sequence_number(table, starting_snapshot_id);

    for data_file in data_files {
        // Java L536: `deletes.forDataFile(startingSequenceNumber, dataFile)` — the applicable concurrently
        // -added deletes. We compute applicability inline (see the doc comment) and branch on
        // `ignore_equality_deletes` per Java L538-548 on the first applicable delete.
        for (delete_file, delete_seq) in &added_deletes {
            // Metrics narrowing (Java `addedDeleteFiles(dataFilter)`): skip a delete whose metrics cannot
            // match the conflict filter.
            if let Some(bound_filter) = &bound_filter
                && !InclusiveMetricsEvaluator::eval(bound_filter, delete_file, true)?
            {
                continue;
            }

            if !delete_applies_to_data_file(
                delete_file,
                *delete_seq,
                data_file,
                starting_sequence_number,
            ) {
                continue;
            }

            let is_position_delete =
                delete_file.content_type() == crate::spec::DataContentType::PositionDeletes;

            if ignore_equality_deletes {
                // Java L538-543: only POSITION deletes are a conflict when equality deletes are ignored.
                if is_position_delete {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot commit, found new position delete for replaced data file: {}",
                            data_file.file_path()
                        ),
                    ));
                }
            } else {
                // Java L544-548: ANY applicable delete is a conflict.
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot commit, found new delete for replaced data file: {}",
                        data_file.file_path()
                    ),
                ));
            }
        }
    }

    Ok(())
}

/// Whether a single concurrently-added delete file APPLIES to `data_file`. Java
/// `DeleteFileIndex.forDataFile`. The rules:
///
/// | test | rule |
/// |---|---|
/// | sequence | delete seq `>= starting_sequence_number`; absent seq applies |
/// | global equality | empty partition applies to any data file |
/// | partition | same spec id and tuple |
/// | path-scoped position | `referenced_data_file` equals the data file path |
fn delete_applies_to_data_file(
    delete_file: &DataFile,
    delete_sequence_number: Option<i64>,
    data_file: &DataFile,
    starting_sequence_number: i64,
) -> bool {
    use crate::spec::DataContentType;

    // Java `*.filter`: keep only deletes whose data sequence number is `>= starting_sequence_number`. An
    // absent sequence number is treated as applicable (conservative — not yet narrowed out).
    if let Some(delete_seq) = delete_sequence_number
        && delete_seq < starting_sequence_number
    {
        return false;
    }

    let is_unpartitioned = delete_file.partition().fields().is_empty();

    match delete_file.content_type() {
        DataContentType::EqualityDeletes => {
            // Java `findGlobalDeletes`: an unpartitioned equality delete is a GLOBAL delete (any data file).
            if is_unpartitioned {
                return true;
            }
            // Java `findEqPartitionDeletes`: same spec id + equal partition tuple.
            delete_file.partition_spec_id == data_file.partition_spec_id
                && delete_file.partition() == data_file.partition()
        }
        DataContentType::PositionDeletes => {
            // Java `findPathDeletes`: a path-scoped position delete matches only the referenced data file.
            if let Some(referenced) = &delete_file.referenced_data_file {
                return referenced == data_file.file_path();
            }
            // Java `findPosPartitionDeletes`: same spec id + equal partition tuple.
            delete_file.partition_spec_id == data_file.partition_spec_id
                && delete_file.partition() == data_file.partition()
        }
        // A `Data` file is never a delete; it cannot apply as one.
        DataContentType::Data => false,
    }
}

/// Enumerate the DATA files DELETED after `starting_snapshot_id`. A `validateDataFilesExist` check
/// inspects these to detect that a concurrent commit already removed a file this operation needs.
///
/// The Rust port of Java `MergingSnapshotProducer.validateDataFilesExist`: the shared
/// [`files_after`] walk over DATA manifests, keeping `Deleted` tombstone entries.
///
/// `skip_deletes` selects the operation set, mirroring Java's two:
///
/// - `false` uses [`operation_removes_data_files`]. `DeleteFiles` uses this arm.
/// - `true` uses [`operation_removes_data_files_skip_deletes`], which `RowDelta` uses by DEFAULT so a
///   concurrent merge-on-read DELETE snapshot does not trip the referenced-files check.
///
/// BOTH sets include `REPLACE`, so a concurrent COMPACTION snapshot's tombstones are inspected on
/// both arms.
///
/// The caller intersects these paths with the set it requires, to decide whether to reject the
/// commit.
pub(crate) async fn deleted_data_files_after(
    table: &Table,
    starting_snapshot_id: Option<i64>,
    skip_deletes: bool,
) -> Result<Vec<DataFile>> {
    let operation_filter = if skip_deletes {
        operation_removes_data_files_skip_deletes
    } else {
        operation_removes_data_files
    };

    files_after(
        table,
        starting_snapshot_id,
        ManifestContentType::Data,
        operation_filter,
        ManifestStatus::Deleted,
    )
    .await
}

/// Reject the commit if any DATA file ADDED by a concurrent commit since `effective_start` COULD contain
/// records matching `conflict_filter` — the filter-based serializable-isolation conflict check shared by
/// the write actions that mirror Java `MergingSnapshotProducer.validateAddedDataFiles`
/// (`OverwriteFiles.validateNoConflictingData`, `RowDelta.validateNoConflictingDataFiles`).
///
/// Java `MergingSnapshotProducer.validateAddedDataFiles`. Rejects the first concurrently
/// added DATA file whose metrics match. `current` is the refreshed base. A `None` filter is
/// `AlwaysTrue`. Default `case_sensitive` is `true`.
///
/// One home keeps `OverwriteFiles` and `RowDelta` from drifting on the walk, the bind, the per-file
/// evaluation, and the error contract.
///
/// # Errors
///
/// A NON-retryable [`ErrorKind::DataInvalid`] naming the filter and the conflicting path, so the
/// commit retry loop stops.
pub(crate) async fn validate_no_conflicting_added_data_files(
    current: &Table,
    effective_start: Option<i64>,
    conflict_filter: Option<&Predicate>,
    case_sensitive: bool,
) -> Result<()> {
    let added = added_data_files_after(current, effective_start).await?;
    if let Some(file) = first_conflicting_file(&added, current, conflict_filter, case_sensitive)? {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Found conflicting files that can contain records matching {}: {}",
                conflict_filter.map_or_else(|| "true".to_string(), |filter| format!("{filter}")),
                file.file_path()
            ),
        ));
    }

    Ok(())
}

/// Reject the commit if any DELETE file ADDED by a concurrent commit since `effective_start` COULD apply to
/// records matching `conflict_filter` — the filter-based serializable-isolation conflict check for the
/// merge-on-read delete path, mirroring Java `MergingSnapshotProducer.validateNoNewDeleteFiles`.
///
/// This is the Rust port of Java `MergingSnapshotProducer.validateNoNewDeleteFiles`
/// (`core/MergingSnapshotProducer.java` L562-570): it enumerates the concurrently-added DELETE files via the
/// shared [`added_delete_files_after`] walk (which applies the V2 guard) and throws a non-retryable
/// `ValidationException` ("Found new conflicting delete files that can apply to records matching %s: %s") on
/// the FIRST file whose metrics permit a match. The per-file "could this added delete file apply to records
/// matching the filter?" test is the SAME [`first_conflicting_file`] (the existing
/// [`InclusiveMetricsEvaluator`]) the data-file check uses.
///
/// Arguments mirror [`validate_no_conflicting_added_data_files`]. The only differences from the data-file
/// check are (1) the DELETE-manifest walk + V2 guard (in [`added_delete_files_after`]) and (2) the
/// DELETE-specific error message — the per-file conflict test is shared.
///
/// **Over-scan vs Java (documented):** see [`added_delete_files_after`] — this port omits Java's
/// `DeleteFileIndex` `startingSequenceNumber` refinement, a conservative over-scan (can only over-reject).
pub(crate) async fn validate_no_conflicting_added_delete_files(
    current: &Table,
    effective_start: Option<i64>,
    conflict_filter: Option<&Predicate>,
    case_sensitive: bool,
) -> Result<()> {
    let added = added_delete_files_after(current, effective_start).await?;
    if let Some(file) = first_conflicting_file(&added, current, conflict_filter, case_sensitive)? {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Found new conflicting delete files that can apply to records matching {}: {}",
                conflict_filter.map_or_else(|| "true".to_string(), |filter| format!("{filter}")),
                file.file_path()
            ),
        ));
    }

    Ok(())
}

/// Reject the commit if any DATA file DELETED by a concurrent commit since `effective_start` COULD contain
/// records matching `conflict_filter` — the filter-based serializable-isolation check that a concurrent
/// commit did not remove data this operation's row filter also targets, mirroring Java
/// `MergingSnapshotProducer.validateDeletedDataFiles` (the `Expression` variant).
///
/// The Rust port of Java `MergingSnapshotProducer.validateDeletedDataFiles`, the `dataFilter`
/// overload. It enumerates the concurrently-DELETED DATA files through
/// [`deleted_data_files_after`] with `skip_deletes = false`, and rejects the FIRST removed file
/// whose metrics permit a match. The per-file test is the SAME [`first_conflicting_file`] the
/// added-file check uses, so the two cannot drift.
///
/// Arguments mirror [`validate_no_conflicting_added_delete_files`]. Only the walk and the message
/// differ.
///
/// **Conservative posture.** [`InclusiveMetricsEvaluator`] over-approximates, so it can only
/// over-reject, never under-reject. That is safe under serializable isolation. The op set matches
/// Java member for member, `REPLACE` included, so a concurrent compaction's removals are scanned
/// here too.
///
/// # Errors
///
/// A NON-retryable [`ErrorKind::DataInvalid`] naming the filter and the conflicting path, so the
/// commit retry loop stops.
pub(crate) async fn validate_deleted_data_files(
    current: &Table,
    effective_start: Option<i64>,
    conflict_filter: Option<&Predicate>,
    case_sensitive: bool,
) -> Result<()> {
    let deleted = deleted_data_files_after(current, effective_start, false).await?;
    if let Some(file) = first_conflicting_file(&deleted, current, conflict_filter, case_sensitive)?
    {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Found conflicting deleted files that can contain records matching {}: {}",
                conflict_filter.map_or_else(|| "true".to_string(), |filter| format!("{filter}")),
                file.file_path()
            ),
        ));
    }

    Ok(())
}

/// Return the first file in `files` that COULD contain records matching `conflict_filter` — the shared
/// per-file conflict test behind both [`validate_no_conflicting_added_data_files`] and
/// [`validate_no_conflicting_added_delete_files`].
///
/// Binds `conflict_filter` to `current`'s current schema ONCE (the caller's filter when `Some`, else
/// `AlwaysTrue` = any file conflicts — the most conservative serializable check, Java
/// `dataConflictDetectionFilter()` returning `alwaysTrue()` when no filter is set), then tests each file
/// with the existing [`InclusiveMetricsEvaluator`] (Java `ManifestGroup.filterData` = inclusive-metrics
/// evaluation over the file's bounds / null / nan stats). Returns the FIRST matching file (Java throws on
/// the first conflict entry), or `None` when nothing can match (including an empty `files`).
///
/// `include_empty_files = true` keeps a zero-record file's evaluation conservative (it never excludes on
/// emptiness alone). The bind happens once for the whole set, not per file.
fn first_conflicting_file(
    files: &[DataFile],
    current: &Table,
    conflict_filter: Option<&Predicate>,
    case_sensitive: bool,
) -> Result<Option<DataFile>> {
    if files.is_empty() {
        // No concurrently-added file of the relevant content — nothing can conflict.
        return Ok(None);
    }

    let schema = current.metadata().current_schema().clone();
    let bound_filter: BoundPredicate = conflict_filter
        .cloned()
        .unwrap_or(Predicate::AlwaysTrue)
        .bind(schema, case_sensitive)?;

    for file in files {
        if InclusiveMetricsEvaluator::eval(&bound_filter, file, true)? {
            return Ok(Some(file.clone()));
        }
    }

    Ok(None)
}

#[cfg(test)]
mod multispec_tests {
    //! Multi-spec producer tests, driven end to end through the `fast_append` and `row_delta`
    //! actions so they exercise the real per-spec manifest grouping and validation.
    //!
    //! The fixture is a V2 table on `identity(x)` as spec 0, evolved to `identity(x) + identity(y)`
    //! as spec 1. Both specs stay resolvable, so one commit can add files under both.

    use std::collections::HashMap;

    use super::FirstRowIdPolicy;
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestContentType,
        ManifestStatus, Struct,
    };
    use crate::table::Table;
    use crate::transaction::tests::make_v2_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, ErrorKind};

    // ============================================================================================
    // Fixtures.
    // ============================================================================================

    /// A data file under spec 0 (`identity(x)`), partition `(x = part_value)`.
    fn data_file_spec0(path: &str, part_value: i64) -> DataFile {
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

    /// A data file with a 1-field partition `(value)` claiming an arbitrary `spec_id` — for a
    /// same-arity rename fixture where spec 1 is also 1-field (`identity(y)`). Shape-identical to
    /// [`data_file_spec0`] but lets the caller stamp the new spec id.
    fn data_file_spec0_under(path: &str, value: i64, spec_id: i32) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(spec_id)
            .partition(Struct::from_iter([Some(Literal::long(value))]))
            .build()
            .unwrap()
    }

    /// A data file under spec 1 (`identity(x) + identity(y)`), partition `(x, y)`.
    fn data_file_spec1(path: &str, x: i64, y: i64, spec_id: i32) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(spec_id)
            .partition(Struct::from_iter([
                Some(Literal::long(x)),
                Some(Literal::long(y)),
            ]))
            .build()
            .unwrap()
    }

    /// A parquet position-delete file under spec 0, partition `(x = part_value)`.
    fn delete_file_spec0(path: &str, part_value: i64) -> DataFile {
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

    /// A parquet position-delete file under spec 1, partition `(x, y)`.
    fn delete_file_spec1(path: &str, x: i64, y: i64, spec_id: i32) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(spec_id)
            .partition(Struct::from_iter([
                Some(Literal::long(x)),
                Some(Literal::long(y)),
            ]))
            .build()
            .unwrap()
    }

    /// Evolve the table's partition spec by adding `identity(y)` and return `(table, new_spec_id)`.
    async fn evolve_spec(catalog: &impl Catalog, table: &Table) -> (Table, i32) {
        let tx = Transaction::new(table);
        let action = tx.update_partition_spec().add_field("y");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(catalog).await.unwrap();
        let new_spec_id = table.metadata().default_partition_spec_id();
        assert_ne!(
            new_spec_id, 0,
            "fixture sanity: the spec evolved away from 0"
        );
        (table, new_spec_id)
    }

    /// Evolve the table's partition spec to a SAME-ARITY but DIFFERENT-NAME 1-field spec: drop
    /// `identity(x)` and add `identity(y)` (on V2 the removed field is OMITTED, not void-replaced —
    /// `update_partition_spec` "removing the only base field on V2 omits it entirely"), so the new
    /// spec is `(identity(y))`, a 1-field spec like spec 0 but rendering the partition path under field
    /// `y` instead of `x`. Returns `(table, new_spec_id)`. This is the panic-FREE multi-spec shape: a
    /// spec-0 file rendered under the (default) new spec would NOT trip the arity `zip_eq` guard — it
    /// would silently render the WRONG field name, which is exactly the summary-path corruption the
    /// per-file-spec fix prevents.
    async fn evolve_spec_same_arity_rename(catalog: &impl Catalog, table: &Table) -> (Table, i32) {
        let tx = Transaction::new(table);
        let action = tx.update_partition_spec().remove_field("x").add_field("y");
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(catalog).await.unwrap();
        let new_spec_id = table.metadata().default_partition_spec_id();
        let new_spec = table.metadata().default_partition_spec();
        assert_eq!(
            new_spec.fields().len(),
            1,
            "fixture sanity: the renamed spec must stay 1-field (no void placeholder on V2): {new_spec:?}"
        );
        assert_eq!(
            new_spec.fields()[0].name,
            "y",
            "fixture sanity: the renamed spec's only field is identity(y): {new_spec:?}"
        );
        assert_ne!(
            new_spec_id, 0,
            "fixture sanity: the spec evolved away from 0"
        );
        (table, new_spec_id)
    }

    /// Set a table property in its own commit (e.g. `write.summary.partition-limit`).
    async fn set_property(catalog: &impl Catalog, table: &Table, key: &str, value: &str) -> Table {
        let tx = Transaction::new(table);
        let action = tx
            .update_table_properties()
            .set(key.to_string(), value.to_string());
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// A fresh V1 table (schema x/y/z longs, spec 0 = `identity(x)`) in `catalog` — for the V1
    /// multi-spec append probe. V1 spec evolution VOID-replaces removed fields (field-id stability),
    /// so an evolved V1 spec stays multi-spec-resolvable just like V2.
    async fn make_v1_minimal_table_in_catalog(catalog: &impl Catalog) -> Table {
        use crate::spec::{
            FormatVersion, NestedField, PartitionSpec, PrimitiveType, Schema, Transform, Type,
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
            .format_version(FormatVersion::V1)
            .build();
        catalog
            .create_table(table_ident.namespace(), table_creation)
            .await
            .unwrap()
    }

    /// The `(spec_id, content)` of every NEW manifest the current snapshot wrote, paired with the set of
    /// live (Added) file paths it carries. Filters to manifests this snapshot itself wrote.
    async fn new_manifests_by_spec(table: &Table) -> Vec<(i32, ManifestContentType, Vec<String>)> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut result = Vec::new();
        for manifest_file in manifest_list.entries() {
            if manifest_file.added_snapshot_id != snapshot.snapshot_id() {
                continue;
            }
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            let added: Vec<String> = manifest
                .entries()
                .iter()
                .filter(|entry| entry.status() == ManifestStatus::Added)
                .map(|entry| entry.file_path().to_string())
                .collect();
            result.push((
                manifest_file.partition_spec_id,
                manifest_file.content,
                added,
            ));
        }
        result
    }

    /// The path → manifest-spec-id map of every live (Added/Existing) entry in the current snapshot —
    /// what a scan would read, with each file's containing-manifest spec id.
    async fn live_paths_with_manifest_spec(table: &Table) -> HashMap<String, i32> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut live = HashMap::new();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.is_alive() {
                    live.insert(
                        entry.file_path().to_string(),
                        manifest_file.partition_spec_id,
                    );
                }
            }
        }
        live
    }

    // ============================================================================================
    // Added DATA files under two specs ⇒ one DATA manifest per spec.
    // ============================================================================================

    /// MULTI-SPEC FAST APPEND. A single fast-append commit adding one file under spec 0 and one under
    /// spec 1 produces TWO data manifests — one per spec — each stamped with its own spec id, with the
    /// partition tuples intact, and both files live for a scan. Risk pinned: the default-spec-only
    /// producer would write BOTH files into ONE default-spec manifest, wrongly stamping the spec-0 file's
    /// partition under the 2-field default spec (partition-tuple corruption / wrong manifest spec id).
    /// Java `FastAppend.appendFile` groups into `newDataFilesBySpec` and writes per-spec manifests.
    #[tokio::test]
    async fn test_fast_append_two_specs_produces_per_spec_data_manifests() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, new_spec_id) = evolve_spec(&catalog, &table).await;

        // One commit adding a spec-0 file AND a spec-1 file.
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![
            data_file_spec0("test/old.parquet", 5),
            data_file_spec1("test/new.parquet", 7, 9, new_spec_id),
        ]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Exactly two NEW data manifests, one per spec id.
        let manifests = new_manifests_by_spec(&table).await;
        let data_manifests: Vec<_> = manifests
            .iter()
            .filter(|(_, content, _)| *content == ManifestContentType::Data)
            .collect();
        assert_eq!(
            data_manifests.len(),
            2,
            "a two-spec append must write ONE data manifest PER spec, got: {manifests:?}"
        );

        // Each manifest carries exactly its own spec's file.
        let spec0_manifest = data_manifests
            .iter()
            .find(|(spec_id, _, _)| *spec_id == 0)
            .expect("a spec-0 data manifest must exist");
        let spec1_manifest = data_manifests
            .iter()
            .find(|(spec_id, _, _)| *spec_id == new_spec_id)
            .expect("a spec-1 data manifest must exist");
        assert_eq!(spec0_manifest.2, vec!["test/old.parquet".to_string()]);
        assert_eq!(spec1_manifest.2, vec!["test/new.parquet".to_string()]);

        // Both files live for a scan, each under its own manifest spec id (partition tuples intact: the
        // spec-0 file is NOT re-stamped under the 2-field default spec).
        let live = live_paths_with_manifest_spec(&table).await;
        assert_eq!(live.get("test/old.parquet"), Some(&0));
        assert_eq!(live.get("test/new.parquet"), Some(&new_spec_id));
    }

    /// MULTI-SPEC CUMULATIVE TOTALS. After appending a spec-0 file, then a commit adding one file under
    /// each spec, the snapshot summary `total-data-files` reflects ALL three files (previous + added).
    /// Risk pinned: a per-spec grouping bug that dropped one group's files would under-count the totals;
    /// per-commit `added-data-files` assertions cannot catch a previous-total seed bug.
    #[tokio::test]
    async fn test_fast_append_multispec_cumulative_totals() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, new_spec_id) = evolve_spec(&catalog, &table).await;

        // First commit: one spec-0 file.
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .add_data_files(vec![data_file_spec0("test/a.parquet", 1)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Second commit: one spec-0 + one spec-1 file in ONE multi-spec commit.
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![
            data_file_spec0("test/b.parquet", 2),
            data_file_spec1("test/c.parquet", 3, 4, new_spec_id),
        ]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let summary = &table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties;
        assert_eq!(
            summary.get("added-data-files"),
            Some(&"2".to_string()),
            "this commit added two data files across two specs"
        );
        assert_eq!(
            summary.get("total-data-files"),
            Some(&"3".to_string()),
            "cumulative total = previous (1) + added (2); a dropped spec group would under-count"
        );
    }

    // ============================================================================================
    // THE SUMMARY-COLLECTOR PIN: per-partition summary keys render each file under ITS OWN spec.
    // ============================================================================================

    /// MULTI-SPEC PARTITION SUMMARY KEYS (the summary-collector per-file-spec fix).
    ///
    /// A multi-spec commit's `partitions.{path}` summary keys MUST render each file's partition under
    /// THAT file's own spec's field names — Java `SnapshotSummary.Builder.addedFile(spec(file.specId()),
    /// file)` → `updatePartitions(spec, file)` → `spec.partitionToPath(file.partition())`. The producer's
    /// `summary()` previously rendered EVERY file under the table default spec; on a multi-spec commit a
    /// spec-0 file's path would be computed under the (default) new spec, producing the WRONG key and
    /// miscounting `changed-partition-count`.
    ///
    /// The fixture deliberately uses a SAME-ARITY rename (spec 0 = `identity(x)`, spec 1 = `identity(y)`)
    /// and the SAME partition VALUE 5 on both files so the bug does NOT trip the arity `zip_eq` panic the
    /// other multi-spec tests catch incidentally — it silently collapses both files onto the SAME `y=5`
    /// path. Risk pinned: (a) the spec-0 file's key is `partitions.x=5` (rendered under spec 0's field
    /// `x`), NOT `partitions.y=5`; (b) `changed-partition-count` is 2 (two distinct per-spec tuples), not
    /// 1 (the collapse the default-spec bug produces — both render `y=5`). Under the revert-to-default
    /// mutation this test fails on BOTH the missing `partitions.x=5` key AND `changed-partition-count`.
    #[tokio::test]
    async fn test_fast_append_multispec_partition_summary_keys_use_file_spec() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        // High partition-summary limit so the per-partition `partitions.{path}` keys are emitted.
        let table = set_property(
            &catalog,
            &table,
            crate::spec::TableProperties::PROPERTY_WRITE_PARTITION_SUMMARY_LIMIT,
            "100",
        )
        .await;
        // Same-arity rename: spec 0 = identity(x), spec 1 = identity(y). Default spec is now spec 1 (y).
        let (table, new_spec_id) = evolve_spec_same_arity_rename(&catalog, &table).await;

        // One commit adding a spec-0 file (partition x=5) AND a spec-1 file (partition y=5) — the SAME
        // numeric value 5, distinguished only by which spec's field names the path renders.
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![
            data_file_spec0("test/old.parquet", 5),
            data_file_spec0_under("test/new.parquet", 5, new_spec_id),
        ]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let summary = &table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties;

        // (a) The spec-0 file renders under spec 0's field `x` (`partitions.x=5`), NOT the default
        // spec's field `y`. The default-spec bug renders it as `partitions.y=5` (wrong) and the
        // spec-0 key vanishes.
        assert!(
            summary.contains_key("partitions.x=5"),
            "the spec-0 file's partition path must render under spec 0's field `x` \
             (`partitions.x=5`); summary keys: {:?}",
            summary.keys().collect::<Vec<_>>()
        );
        // The spec-1 file renders under spec 1's field `y`.
        assert!(
            summary.contains_key("partitions.y=5"),
            "the spec-1 file's partition path must render under spec 1's field `y` \
             (`partitions.y=5`); summary keys: {:?}",
            summary.keys().collect::<Vec<_>>()
        );
        // (b) Two DISTINCT per-spec partition tuples ⇒ count 2. The default-spec bug collapses both
        // onto `y=5` ⇒ count 1.
        assert_eq!(
            summary.get("changed-partition-count"),
            Some(&"2".to_string()),
            "two distinct per-spec partition tuples (x=5 under spec 0, y=5 under spec 1) ⇒ \
             changed-partition-count 2; the default-spec bug collapses both onto y=5 ⇒ 1. summary: {summary:?}"
        );
        assert_eq!(
            summary.get("partition-summaries-included"),
            Some(&"true".to_string()),
            "per-partition summaries must be included under a generous partition-limit"
        );
    }

    // ============================================================================================
    // V1 multi-spec DATA append (spec evolution exists on V1 via void replacement).
    // ============================================================================================

    /// V1 MULTI-SPEC DATA APPEND. Spec evolution is legal on V1 (removed fields are VOID-replaced to
    /// keep field ids stable), so a V1 table can carry multiple resolvable specs. This probes that a
    /// single V1 fast-append adding one spec-0 file and one spec-1 file WORKS (Java parity — the V1
    /// arm of `new_cluster_manifest_writer` builds a V1 manifest per spec group, content-agnostic) and
    /// produces ONE V1 data manifest PER spec, each stamped with its own spec id. Documents the V1
    /// behavior the brief asked to probe: it WORKS, not fail-loud.
    #[tokio::test]
    async fn test_v1_fast_append_two_specs_produces_per_spec_data_manifests() {
        let catalog = new_memory_catalog().await;
        let table = make_v1_minimal_table_in_catalog(&catalog).await;
        assert_eq!(
            table.metadata().format_version(),
            crate::spec::FormatVersion::V1,
            "fixture sanity: V1 table"
        );
        // Evolve: spec 1 = identity(x) + identity(y) (2-field), spec 0 stays identity(x) (1-field).
        let (table, new_spec_id) = evolve_spec(&catalog, &table).await;

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![
            data_file_spec0("test/v1-old.parquet", 5),
            data_file_spec1("test/v1-new.parquet", 7, 9, new_spec_id),
        ]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let manifests = new_manifests_by_spec(&table).await;
        let data_manifests: Vec<_> = manifests
            .iter()
            .filter(|(_, content, _)| *content == ManifestContentType::Data)
            .collect();
        assert_eq!(
            data_manifests.len(),
            2,
            "a V1 two-spec append must write ONE data manifest PER spec, got: {manifests:?}"
        );
        let spec0_manifest = data_manifests
            .iter()
            .find(|(spec_id, _, _)| *spec_id == 0)
            .expect("a V1 spec-0 data manifest must exist");
        let spec1_manifest = data_manifests
            .iter()
            .find(|(spec_id, _, _)| *spec_id == new_spec_id)
            .expect("a V1 spec-1 data manifest must exist");
        assert_eq!(spec0_manifest.2, vec!["test/v1-old.parquet".to_string()]);
        assert_eq!(spec1_manifest.2, vec!["test/v1-new.parquet".to_string()]);

        let live = live_paths_with_manifest_spec(&table).await;
        assert_eq!(live.get("test/v1-old.parquet"), Some(&0));
        assert_eq!(live.get("test/v1-new.parquet"), Some(&new_spec_id));
    }

    // ============================================================================================
    // Added DELETE files under two specs ⇒ one DELETE manifest per spec (V2 position deletes).
    // ============================================================================================

    /// MULTI-SPEC ROW DELTA. A single `row_delta` commit adding one position-delete file under spec 0 and
    /// one under spec 1 produces TWO delete manifests — one per spec — each stamped with its own spec id,
    /// with the partition tuples intact. Risk pinned: the default-spec-only producer would write BOTH
    /// deletes into ONE default-spec delete manifest, wrongly stamping the spec-0 delete's partition (a
    /// delete then stops matching its target rows = silent resurrection). Java `MergingSnapshotProducer`
    /// groups into `newDeleteFilesBySpec` and writes per-spec delete manifests.
    #[tokio::test]
    async fn test_row_delta_two_specs_produces_per_spec_delete_manifests() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        // Seed live data so the deletes have targets (and the table is non-empty).
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .add_data_files(vec![data_file_spec0("test/seed.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let (table, new_spec_id) = evolve_spec(&catalog, &table).await;

        // One row_delta commit adding a spec-0 delete AND a spec-1 delete.
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![
            delete_file_spec0("test/old-del.parquet", 5),
            delete_file_spec1("test/new-del.parquet", 7, 9, new_spec_id),
        ]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let manifests = new_manifests_by_spec(&table).await;
        let delete_manifests: Vec<_> = manifests
            .iter()
            .filter(|(_, content, _)| *content == ManifestContentType::Deletes)
            .collect();
        assert_eq!(
            delete_manifests.len(),
            2,
            "a two-spec row_delta must write ONE delete manifest PER spec, got: {manifests:?}"
        );

        let spec0_manifest = delete_manifests
            .iter()
            .find(|(spec_id, _, _)| *spec_id == 0)
            .expect("a spec-0 delete manifest must exist");
        let spec1_manifest = delete_manifests
            .iter()
            .find(|(spec_id, _, _)| *spec_id == new_spec_id)
            .expect("a spec-1 delete manifest must exist");
        assert_eq!(spec0_manifest.2, vec!["test/old-del.parquet".to_string()]);
        assert_eq!(spec1_manifest.2, vec!["test/new-del.parquet".to_string()]);
    }

    // ============================================================================================
    // Unknown-spec rejection (the lifted validation's exact Java message), data + delete.
    // ============================================================================================

    /// UNKNOWN-SPEC DATA REJECTION. Adding a data file whose `partition_spec_id` matches no table spec
    /// fails with Java's EXACT message ("Cannot find partition spec %s for data file: %s"). Risk pinned:
    /// the lift from "spec == default" to "spec EXISTS" must still reject a genuinely-unknown spec id
    /// (the validation guard must fire on the bad case, not just accept the good ones).
    #[tokio::test]
    async fn test_fast_append_unknown_spec_id_data_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        // Spec id 99 does not exist on the table.
        let bogus = data_file_spec1("test/bogus.parquet", 1, 2, 99);
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![bogus]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("an unknown spec id must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Cannot find partition spec 99 for data file: test/bogus.parquet"),
            "unexpected message: {}",
            err.message()
        );
    }

    /// UNKNOWN-SPEC DELETE REJECTION. Adding a delete file whose `partition_spec_id` matches no table
    /// spec fails with Java's EXACT message ("Cannot find partition spec %s for delete file: %s") — the
    /// delete-file noun, distinct from the data-file message. Risk pinned: the lifted delete validation
    /// must reject an unknown spec id AND use the delete-file noun (Java has two distinct messages).
    #[tokio::test]
    async fn test_row_delta_unknown_spec_id_delete_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .add_data_files(vec![data_file_spec0("test/seed.parquet", 0)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Spec id 99 does not exist on the table.
        let bogus = delete_file_spec1("test/bogus-del.parquet", 1, 2, 99);
        let tx = Transaction::new(&table);
        let action = tx.row_delta().add_deletes(vec![bogus]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("an unknown spec id must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Cannot find partition spec 99 for delete file: test/bogus-del.parquet"),
            "unexpected message: {}",
            err.message()
        );
    }

    // ============================================================================================
    // Partition-value-vs-wrong-spec-type rejection.
    // ============================================================================================

    /// PARTITION-VALUE-VS-WRONG-SPEC-TYPE REJECTION. A file claiming spec 0 (a 1-field `identity(x)`
    /// partition) but carrying a 2-field partition tuple is rejected — the partition value must be
    /// compatible with THAT spec's partition type (not the default's). Risk pinned: the per-spec lift
    /// must validate the partition value against the FILE's claimed spec, so a tuple that matches the
    /// default spec's arity but not the claimed spec's is still caught.
    #[tokio::test]
    async fn test_fast_append_partition_value_against_wrong_spec_type_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let (table, _new_spec_id) = evolve_spec(&catalog, &table).await;

        // Claims spec 0 (1-field partition type) but carries a 2-field tuple — incompatible with spec 0.
        let mismatched = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/mismatch.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([
                Some(Literal::long(1)),
                Some(Literal::long(2)),
            ]))
            .build()
            .unwrap();
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![mismatched]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a partition tuple incompatible with the claimed spec must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Partition value is not compatible with partition type"),
            "unexpected message: {}",
            err.message()
        );
    }

    // ============================================================================================
    // WG3-L2 COMMIT-PATH REACHABILITY: the summary path must not ABORT on a mismatched
    // (spec, tuple) pair it cannot validate.
    // ============================================================================================

    /// A minimal operation for calling `summary()` directly: the producer's OWN state (the
    /// `removed_*_files` vectors `commit()` assigns before `summary()`) is what the summary walks.
    struct SummaryOnlyOperation;

    impl super::SnapshotProduceOperation for SummaryOnlyOperation {
        fn operation(&self) -> crate::spec::Operation {
            crate::spec::Operation::Delete
        }

        async fn delete_entries(
            &self,
            _snapshot_produce: &super::SnapshotProducer<'_>,
        ) -> crate::Result<Vec<crate::spec::ManifestEntry>> {
            Ok(vec![])
        }

        async fn delete_files(
            &self,
            _snapshot_produce: &super::SnapshotProducer<'_>,
        ) -> crate::Result<Vec<DataFile>> {
            Ok(vec![])
        }

        async fn existing_manifest(
            &self,
            _snapshot_produce: &super::SnapshotProducer<'_>,
        ) -> crate::Result<Vec<crate::spec::ManifestFile>> {
            Ok(vec![])
        }
    }

    /// SUMMARY-PATH TOTALITY (the `removed_data_files` loop × `file_partition_spec`'s default-spec
    /// substitution). `commit()` assigns `removed_data_files` from the resolved delete set and calls
    /// `summary()` BEFORE `manifest_file()`; that loop never runs `validate_partition_value`, and
    /// `file_partition_spec` substitutes the table DEFAULT spec when a file's spec id is unknown. A
    /// removed file claiming an absent spec therefore reaches `partition_to_path` paired with a spec
    /// of a DIFFERENT arity.
    ///
    /// Before WG3-L2 this ABORTED (`partition.rs` index out of bounds) — mid-commit, in the
    /// infallible summary path, with no error to catch. It now renders the missing field as `null`
    /// and the summary is produced.
    ///
    /// MUTATION (restore the positional `data[index]` lookup in `PartitionSpec::partition_to_path`):
    /// this test panics.
    #[tokio::test]
    async fn test_summary_survives_a_removed_file_under_a_substituted_spec() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let table = set_property(
            &catalog,
            &table,
            crate::spec::TableProperties::PROPERTY_WRITE_PARTITION_SUMMARY_LIMIT,
            "100",
        )
        .await;
        // Default spec is now spec 1 = identity(x) + identity(y) — TWO fields.
        let (table, new_spec_id) = evolve_spec(&catalog, &table).await;
        assert_eq!(
            table
                .metadata()
                .partition_spec_by_id(new_spec_id)
                .expect("the evolved spec must exist")
                .fields()
                .len(),
            2,
            "fixture sanity: the substituted default spec has two fields"
        );

        // A REMOVED file claiming spec 99 (absent) and carrying a ONE-value tuple.
        let removed = data_file_spec0_under("test/gone.parquet", 5, 99);
        let mut producer = super::SnapshotProducer::new(
            &table,
            uuid::Uuid::now_v7(),
            None,
            HashMap::new(),
            vec![],
            FirstRowIdPolicy::Suppress,
        );
        producer.removed_data_files = vec![removed];

        let summary = producer
            .summary(&SummaryOnlyOperation)
            .expect("the summary path must stay total for a tuple it cannot validate");
        // The per-partition summary VALUE is a `HashMap`-ordered join, so assert its components.
        let partition_summary = summary
            .additional_properties
            .get("partitions.x=5/y=null")
            .cloned()
            .unwrap_or_default();
        assert!(
            partition_summary.contains("deleted-data-files=1")
                && partition_summary.contains("deleted-records=1"),
            "the missing tuple slot must render `null` under the substituted spec: {:?}",
            summary.additional_properties
        );
        assert_eq!(
            summary.additional_properties.get("changed-partition-count"),
            Some(&"1".to_string())
        );
    }

    /// END-TO-END COMMIT-PATH REACHABILITY. `remove_partition_specs` only refuses to drop the
    /// DEFAULT spec — it does not check whether live files still reference the spec being dropped.
    /// Deleting such a file drives `summary()` (which runs BEFORE `manifest_file()`) into
    /// `file_partition_spec`'s default-spec substitution with a one-value tuple and a two-field spec.
    ///
    /// Before WG3-L2 the commit ABORTED there. It now reaches the first consumer that CAN report the
    /// broken metadata — the manifest rewriter — and fails with a typed, catchable error. The
    /// engine-trust property under test is "no abort", not "the commit succeeds": the metadata is
    /// genuinely inconsistent, so failing loudly is correct.
    ///
    /// MUTATION (restore the positional `data[index]` lookup in `PartitionSpec::partition_to_path`):
    /// this test panics instead of returning the typed error.
    #[tokio::test]
    async fn test_delete_of_a_file_whose_spec_was_dropped_errors_instead_of_aborting() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .add_data_files(vec![data_file_spec0("test/seed.parquet", 5)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let (table, _new_spec_id) = evolve_spec(&catalog, &table).await;

        // Drop spec 0 while `test/seed.parquet` still references it.
        let commit = crate::TableCommit::builder()
            .ident(table.identifier().to_owned())
            .updates(vec![crate::TableUpdate::RemovePartitionSpecs {
                spec_ids: vec![0],
            }])
            .requirements(vec![])
            .base_metadata_location(table.metadata_location().map(str::to_string))
            .build();
        let table = catalog
            .update_table(commit)
            .await
            .expect("dropping a non-default spec is permitted");
        assert!(
            table.metadata().partition_spec_by_id(0).is_none(),
            "fixture sanity: spec 0 is gone while a live file still claims it"
        );

        let tx = Transaction::new(&table);
        let action = tx.delete_files().delete_file("test/seed.parquet");
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("the dropped spec must surface as an error, not an abort");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message()
                .contains("Cannot rewrite manifest: unknown partition spec id 0"),
            "unexpected message: {}",
            err.message()
        );
    }
}

#[cfg(test)]
mod validate_partition_value_tests {
    //! SAF-005 panic-hardening pins: `validate_partition_value` must surface a NON-primitive
    //! partition literal as a typed `DataInvalid` error — never `unwrap`-panic mid-commit.
    //! (Java posture: the typed `PartitionData.get(int, Class<T>)` accessor throws
    //! `IllegalArgumentException` for a wrong-kind value — an error, never an abort;
    //! `core/src/main/java/org/apache/iceberg/PartitionData.java` L119-129 @ 1.10.0.)

    use std::sync::Arc;

    use super::SnapshotProducer;
    use crate::ErrorKind;
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, NestedField, PrimitiveType,
        Struct, StructType, Type,
    };
    use crate::transaction::tests::make_v2_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction};

    fn long_partition_type() -> StructType {
        StructType::new(vec![Arc::new(NestedField::optional(
            1000,
            "x",
            Type::Primitive(PrimitiveType::Long),
        ))])
    }

    /// P2 (direct): a nested-struct literal in a primitive-typed partition slot → `Err`, not panic.
    #[test]
    fn test_validate_partition_value_rejects_non_primitive_literal_as_error() {
        let nested = Struct::from_iter([Some(Literal::long(1))]);
        let partition_value = Struct::from_iter([Some(Literal::Struct(nested))]);

        let err =
            SnapshotProducer::validate_partition_value(&partition_value, &long_partition_type())
                .expect_err("a non-primitive partition literal must be rejected, not panic");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("primitive literal"),
            "the error should name the primitive-literal requirement: {}",
            err.message()
        );
    }

    /// P2 regression halves: a compatible primitive passes; an incompatible primitive still errs
    /// through the pre-existing compatibility arm.
    #[test]
    fn test_validate_partition_value_primitive_compatibility_unchanged() {
        let ok_value = Struct::from_iter([Some(Literal::long(7))]);
        SnapshotProducer::validate_partition_value(&ok_value, &long_partition_type())
            .expect("a compatible primitive partition value must pass");

        let incompatible = Struct::from_iter([Some(Literal::string("not-a-long"))]);
        let err = SnapshotProducer::validate_partition_value(&incompatible, &long_partition_type())
            .expect_err("an incompatible primitive partition value must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    /// P2 (real path): the same rejection reaches the caller as a commit ERROR through
    /// `fast_append` → `validate_added_data_files` (previously: a process-aborting panic).
    #[tokio::test]
    async fn test_fast_append_non_primitive_partition_literal_errors_not_panics() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let bad = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/non-primitive-part.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::Struct(
                Struct::from_iter([Some(Literal::long(1))]),
            ))]))
            .build()
            .unwrap();

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![bad]);
        let tx = action.apply(tx).unwrap();
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("a non-primitive partition literal must fail the commit, not abort");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("primitive literal"),
            "unexpected message: {}",
            err.message()
        );
    }
}

#[cfg(test)]
mod first_row_id_suppression_tests {
    //! The add-seam rule of [`FirstRowIdPolicy`], driven end to end through every producer that can
    //! add a data file (Java `MergingSnapshotProducer.add(DataFile)` →
    //! `Delegates.suppressFirstRowId`, and the `FastAppend` that does not call it).
    //!
    //! The probe file carries a `first_row_id` no reader would compute for it. The assertions read
    //! the manifest bytes back with [`Manifest::parse_avro`], which skips read-side inheritance, so
    //! they see the value the producer STORED rather than the one a reader derives.

    use std::collections::HashMap;

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Manifest,
        ManifestContentType, ManifestStatus, Struct,
    };
    use crate::table::Table;
    use crate::transaction::tests::make_v3_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction};

    /// The value the probe file arrives with. No manifest range in these fixtures reaches it, so a
    /// stored `Some(PROBE_FIRST_ROW_ID)` can only have come from the caller.
    const PROBE_FIRST_ROW_ID: i64 = 90_000;

    /// A data file under the fixture's spec 0 (`identity(x)`), partition `(x = 0)`.
    fn data_file(path: &str, first_row_id: Option<i64>) -> DataFile {
        let mut file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(3)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .expect("build the fixture data file");
        file.first_row_id = first_row_id;
        file
    }

    /// Every live DATA entry's STORED `first_row_id`, keyed by file path.
    ///
    /// Reads the Avro directly: `ManifestFile::load_manifest` runs `assign_first_row_ids`, which
    /// overwrites an absent value and would hide the difference this module measures.
    async fn stored_first_row_ids(table: &Table) -> HashMap<String, Option<i64>> {
        let metadata = table.metadata();
        let snapshot = metadata
            .current_snapshot()
            .expect("the committed table has a current snapshot");
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), metadata)
            .await
            .expect("load the manifest list");

        let mut stored = HashMap::new();
        for manifest_file in manifest_list.entries() {
            if manifest_file.content != ManifestContentType::Data {
                continue;
            }
            let bytes = table
                .file_io()
                .new_input(&manifest_file.manifest_path)
                .expect("open the manifest")
                .read()
                .await
                .expect("read the manifest bytes");
            let manifest = Manifest::parse_avro(&bytes).expect("parse the manifest avro");
            for entry in manifest.entries() {
                if entry.status() == ManifestStatus::Deleted {
                    continue;
                }
                stored.insert(
                    entry.file_path().to_string(),
                    entry.data_file().first_row_id(),
                );
            }
        }
        stored
    }

    /// The producers of the charter's partition that can add a data file.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Producer {
        FastAppend,
        MergeAppend,
        OverwriteFiles,
        ReplacePartitions,
        RewriteFiles,
        RowDelta,
    }

    impl Producer {
        /// Java `FastAppend` extends `SnapshotProducer`; every other arm extends
        /// `MergingSnapshotProducer`, whose `add(DataFile)` suppresses.
        fn suppresses(self) -> bool {
            self != Producer::FastAppend
        }
    }

    /// Seed the table with `seed`, then add `probe` through `producer`, and return what each file's
    /// `first_row_id` was STORED as.
    async fn commit_probe(producer: Producer) -> HashMap<String, Option<i64>> {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let seed = data_file("test/seed.parquet", None);
        let transaction = Transaction::new(&table);
        let transaction = transaction
            .fast_append()
            .add_data_files(vec![seed.clone()])
            .apply(transaction)
            .expect("apply the seed append");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the seed append");

        let probe = data_file("test/probe.parquet", Some(PROBE_FIRST_ROW_ID));
        let transaction = Transaction::new(&table);
        let transaction = match producer {
            Producer::FastAppend => transaction
                .fast_append()
                .add_data_files(vec![probe])
                .apply(transaction),
            Producer::MergeAppend => transaction
                .merge_append()
                .add_data_files(vec![probe])
                .apply(transaction),
            Producer::OverwriteFiles => transaction
                .overwrite_files()
                .add_file(probe)
                .delete_file(seed.file_path().to_string())
                .apply(transaction),
            Producer::ReplacePartitions => transaction
                .replace_partitions()
                .add_file(probe)
                .apply(transaction),
            Producer::RewriteFiles => transaction
                .rewrite_files(vec![seed.clone()], vec![probe])
                .apply(transaction),
            Producer::RowDelta => transaction
                .row_delta()
                .add_data_files(vec![probe])
                .apply(transaction),
        }
        .expect("apply the probe action");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the probe action");

        stored_first_row_ids(&table).await
    }

    /// The domain table: one row per producer that can add a data file.
    ///
    /// Risk pinned: a stale `first_row_id` survives read-side inheritance, so the added file claims
    /// a row-id range that describes other rows. `FastAppend` is the deliberate exception — Java
    /// does not suppress there, and matching that asymmetry is the point of the seam.
    #[tokio::test]
    async fn every_merging_producer_suppresses_first_row_id_and_fast_append_does_not() {
        for producer in [
            Producer::FastAppend,
            Producer::MergeAppend,
            Producer::OverwriteFiles,
            Producer::ReplacePartitions,
            Producer::RewriteFiles,
            Producer::RowDelta,
        ] {
            let stored = commit_probe(producer).await;
            let probe = stored
                .get("test/probe.parquet")
                .copied()
                .unwrap_or_else(|| panic!("{producer:?} committed no probe entry"));
            let expected = if producer.suppresses() {
                None
            } else {
                Some(PROBE_FIRST_ROW_ID)
            };
            assert_eq!(
                probe, expected,
                "{producer:?} stored the wrong first_row_id for the added file"
            );
        }
    }

    /// The seed file's stored `first_row_id` must stay absent whatever the producer does to it, so
    /// the domain table above cannot pass by suppressing every entry in the manifest.
    #[tokio::test]
    async fn suppression_reaches_only_the_added_file() {
        let stored = commit_probe(Producer::MergeAppend).await;
        assert_eq!(
            stored.get("test/seed.parquet").copied(),
            Some(None),
            "the carried-forward seed entry must still be present and unassigned"
        );
    }

    /// `DeleteFiles` is the seventh producer of the partition. It passes `Suppress` like every other
    /// merging producer, but it hands the producer no data file at all, so the rule is vacuous
    /// there. Asserted through the commit rather than by reading the call site.
    ///
    /// The survivor's stored id is its INHERITED one, not an absent value: the rewrite that
    /// tombstones the deleted file reads the source manifest through the assigning reader.
    #[tokio::test]
    async fn delete_files_adds_no_data_file_to_suppress() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let seed = data_file("test/seed.parquet", None);
        let other = data_file("test/other.parquet", None);
        let transaction = Transaction::new(&table);
        let transaction = transaction
            .fast_append()
            .add_data_files(vec![seed.clone(), other])
            .apply(transaction)
            .expect("apply the seed append");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the seed append");

        let transaction = Transaction::new(&table);
        let transaction = transaction
            .delete_files()
            .delete_file(seed.file_path().to_string())
            .apply(transaction)
            .expect("apply the delete");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the delete");

        let stored = stored_first_row_ids(&table).await;
        assert!(
            !stored.contains_key("test/seed.parquet"),
            "the deleted file must not survive as a live entry"
        );
        assert_eq!(
            stored.len(),
            1,
            "a delete-only commit adds no data file, so it has none to suppress: {stored:?}"
        );
        assert_eq!(
            stored.get("test/other.parquet").copied(),
            Some(Some(3)),
            "the survivor keeps the id it inherited behind the 3-row seed"
        );
    }
}

#[cfg(test)]
mod manifest_list_order_tests {
    //! The three conjuncts of the manifest-list order [`SnapshotProducer::manifest_file`] emits.
    //!
    //! The added-data-first conjunct is pinned cross-engine by the row-lineage interop suite. These
    //! two cover the other two, which that suite cannot see: the existing DATA manifests keep their
    //! source-list order, and every DATA manifest precedes every DELETE manifest.

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestContentType,
        ManifestStatus, Struct,
    };
    use crate::table::Table;
    use crate::transaction::tests::make_v2_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction};

    /// A data file under the fixture's spec 0 (`identity(x)`), partition `(x = 0)`.
    fn data_file(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .expect("build the fixture data file")
    }

    /// A parquet position-delete file under the same spec, so the commit writes a DELETE manifest.
    fn position_delete_file(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .expect("build the fixture delete file")
    }

    /// The committed manifest list's `content` sequence.
    async fn manifest_contents(table: &Table) -> Vec<ManifestContentType> {
        manifest_list(table)
            .await
            .into_iter()
            .map(|(content, _)| content)
            .collect()
    }

    /// Each manifest in list order, as its content plus the paths of its live entries.
    async fn manifest_list(table: &Table) -> Vec<(ManifestContentType, Vec<String>)> {
        let metadata = table.metadata();
        let snapshot = metadata
            .current_snapshot()
            .expect("the committed table has a current snapshot");
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), metadata)
            .await
            .expect("load the manifest list");

        let mut described = Vec::new();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file
                .load_manifest(table.file_io())
                .await
                .expect("load the manifest");
            let mut live: Vec<String> = manifest
                .entries()
                .iter()
                .filter(|entry| entry.status() != ManifestStatus::Deleted)
                .map(|entry| entry.file_path().to_string())
                .collect();
            live.sort();
            described.push((manifest_file.content, live));
        }
        described
    }

    /// Conjunct (b): the carried-forward DATA manifests keep the order they had in the source list.
    ///
    /// Risk pinned: two carried-forward manifests that both still need a `first_row_id` range take
    /// each other's range when the order moves, which is the same cross-engine row-identity
    /// divergence the added-data conjunct removes. Reached here without V3, by identifying each
    /// manifest by the files it holds.
    #[tokio::test]
    async fn carried_forward_data_manifests_keep_their_source_order() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let transaction = Transaction::new(&table);
        let transaction = transaction
            .fast_append()
            .add_data_files(vec![
                data_file("test/a1.parquet"),
                data_file("test/a2.parquet"),
            ])
            .apply(transaction)
            .expect("apply the first append");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the first append");

        let transaction = Transaction::new(&table);
        let transaction = transaction
            .fast_append()
            .add_data_files(vec![data_file("test/b.parquet")])
            .apply(transaction)
            .expect("apply the second append");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the second append");

        let before: Vec<Vec<String>> = manifest_list(&table)
            .await
            .into_iter()
            .map(|(_, live)| live)
            .collect();
        assert_eq!(
            before,
            vec![vec!["test/b.parquet".to_string()], vec![
                "test/a1.parquet".to_string(),
                "test/a2.parquet".to_string()
            ]],
            "fixture precondition: two data manifests, newest first"
        );

        // Delete a1. Its manifest is rewritten; the other is carried forward untouched. Neither is
        // emptied, so both still hold live rows.
        let transaction = Transaction::new(&table);
        let transaction = transaction
            .delete_files()
            .delete_file("test/a1.parquet".to_string())
            .apply(transaction)
            .expect("apply the delete");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the delete");

        let after: Vec<Vec<String>> = manifest_list(&table)
            .await
            .into_iter()
            .map(|(_, live)| live)
            .collect();
        assert_eq!(
            after,
            vec![vec!["test/b.parquet".to_string()], vec![
                "test/a2.parquet".to_string()
            ]],
            "the carried-forward manifest and the rewritten one kept their source-list order"
        );
    }

    /// Conjunct (c): every DATA manifest precedes every DELETE manifest.
    ///
    /// Risk pinned: an interleaved list diverges from Java `MergingSnapshotProducer.apply`, which
    /// builds the data group and the delete group separately and concatenates them in that order.
    #[tokio::test]
    async fn every_data_manifest_precedes_every_delete_manifest() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let transaction = Transaction::new(&table);
        let transaction = transaction
            .fast_append()
            .add_data_files(vec![data_file("test/d1.parquet")])
            .apply(transaction)
            .expect("apply the append");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the append");

        let transaction = Transaction::new(&table);
        let transaction = transaction
            .row_delta()
            .add_deletes(vec![position_delete_file("test/d1-pos-del.parquet")])
            .apply(transaction)
            .expect("apply the row delta");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the row delta");
        assert_eq!(
            manifest_contents(&table).await,
            vec![ManifestContentType::Data, ManifestContentType::Deletes],
            "fixture precondition: the table now carries a DELETE manifest"
        );

        // A commit that adds a data manifest while a delete manifest is carried forward is the only
        // shape in which the two groups can interleave.
        let transaction = Transaction::new(&table);
        let transaction = transaction
            .fast_append()
            .add_data_files(vec![data_file("test/d2.parquet")])
            .apply(transaction)
            .expect("apply the second append");
        let table = transaction
            .commit(&catalog)
            .await
            .expect("commit the second append");

        let contents = manifest_contents(&table).await;
        assert_eq!(
            contents,
            vec![
                ManifestContentType::Data,
                ManifestContentType::Data,
                ManifestContentType::Deletes
            ],
            "the manifest list is all DATA then all DELETES"
        );
    }
}
