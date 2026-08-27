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

//! ExpireSnapshots file cleanup: the physical half of Java's `ExpireSnapshots`
//! (`cleanExpiredFiles(true)`), on top of the metadata-only
//! [`ExpireSnapshotsAction`](crate::transaction::ExpireSnapshotsAction).
//!
//! **This module deletes files.** Deleting a file that a retained snapshot still reaches destroys
//! data unrecoverably. Every choice here biases toward under-deletion, and the deletion set comes
//! only from the set algebra below.
//!
//! # The set algebra (Java 1.10.0 `ReachableFileCleanup.cleanFiles(before, after)`)
//!
//! | set | rule |
//! |---|---|
//! | expired snapshots | `before.snapshots() − after.snapshots()`, by id |
//! | manifest lists | each expired snapshot's `manifest_list`, unless a retained snapshot names it too |
//! | candidate manifests | every manifest the expired lists name, minus every manifest a retained snapshot names, by path |
//! | content files | live entry paths of the candidate manifests, minus live entry paths of every retained manifest |
//! | statistics files | present in `before`, absent from `after` |
//!
//! Sparing a shared manifest list is a Rust divergence. Java deletes unconditionally, which is safe
//! only because every Java-written snapshot owns its list file.
//!
//! The candidates-minus-retained subtraction protects a carried-forward manifest: a fast append
//! re-lists every prior manifest, so an expired snapshot's manifests usually survive in a retained
//! descendant. Live means status `ADDED` or `EXISTING`, on **both** sides. A deletion vector's
//! `file_path` is its Puffin location, so one retained vector protects the whole Puffin file.
//!
//! Deletion order mirrors Java: content files, manifests, manifest lists, statistics files.
//!
//! # The post-commit seam
//!
//! Java runs the metadata commit in the retry loop and cleans files only after it succeeds.
//! Deletion never runs inside the retry loop, and never on a failed commit.
//!
//! The Rust [`TransactionAction`](crate::transaction::TransactionAction) seam has no post-commit
//! hook, so cleanup lives outside the action:
//!
//! - [`ExpireSnapshotsCleanup::clean_expired_files`] is the two-state core. **`after` must be the
//!   table's current committed metadata.** A staler `after` deletes files the live table reaches.
//! - [`ExpireSnapshotsCleanup::commit_and_clean`] wraps commit and cleanup in Java's order. The
//!   `?` on the commit makes the deletion path unreachable after a failed commit.
//!
//! The captured `before` may be staler than the final retry base. A snapshot a concurrent commit
//! added is absent from `before`, so it never becomes a candidate. A snapshot a concurrent expire
//! removed does enter the expired set, but it is absent from the committed `after` too, so the
//! sweep still deletes only what the current metadata cannot reach. Java's `cleanFiles(base,
//! current)` has the same property. The overlap is a benign double-delete race with the other
//! expirer's own sweep.
//!
//! **The inherited ref-resurrection window.** A ref created concurrently at a to-be-expired
//! snapshot id is not guardable by the expiry's `RefSnapshotIdMatch` guards. If it lands first, the
//! expiry drops the now-dangling ref and the committed metadata stops reaching that snapshot. This
//! cleanup derives from that metadata, so it makes the loss physical. The window belongs to the
//! metadata commit: a full-CAS catalog rejects the racing ref commit, and a ref commit after the
//! expiry fails validation.
//!
//! **Default posture divergence.** Java's `cleanExpiredFiles` defaults to true. The Rust action
//! commits metadata only, and cleanup runs only when the caller invokes this module. An
//! irreversible file-deletion path in a library must be opt-in, not ambient.
//!
//! # Failure posture
//!
//! Java logs and continues. Silent swallowing is unacceptable for a deletion sweep, so every
//! failure is collected in the [`CleanupReport`], which is the authoritative contract. Each
//! failure also emits a `tracing::warn!`, so an operator sees it without waiting for the report.
//!
//! | failure | effect |
//! |---|---|
//! | manifest-list read, either side | `Err` before any deletion |
//! | candidate-manifest read | skip that manifest's content files; still delete the manifest |
//! | retained-manifest read | clear the whole content-file set: liveness cannot be proven |
//! | per-file delete | record the path and funnel kind, continue the sweep |
//!
//! Rust is marginally stricter than Java on the first row. Java never reads the retained lists
//! when no candidate manifest exists, so a corrupt retained list can escape its walk.
//!
//! # Deferred
//!
//! `IncrementalFileCleanup` is Java's other strategy, chosen only for a pure linear main-ancestry
//! trim. It walks manifest provenance to avoid reading retained manifests on a huge table. That is
//! throughput, not correctness: `ReachableFileCleanup` is what Java itself falls back to for every
//! non-linear history, and its deletion set is correct for every incremental case too.
//! `cleanExpiredMetadata`, the unreachable spec and schema pruning, is also deferred.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use futures::future::BoxFuture;

use crate::error::Result;
use crate::io::FileIO;
use crate::spec::{DataContentType, ManifestFile, SnapshotRef, TableMetadata, TableProperties};
use crate::table::Table;
use crate::transaction::Transaction;
use crate::transaction::expire_snapshots::parse_property;
use crate::{Catalog, Error, ErrorKind};

/// The injectable delete function. The default deletes through [`FileIO::delete`].
pub type DeleteFunction = dyn Fn(String) -> BoxFuture<'static, Result<()>> + Send + Sync;

/// Which cleanup step a [`CleanupFailure`] belongs to: the four delete funnels, plus the two
/// manifest-read planning steps whose failures Java suppresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupFailureKind {
    /// Deleting a data file, a delete file, or a deletion-vector Puffin failed. Java's `"data"`
    /// funnel.
    DeleteContentFile,
    /// Deleting a manifest file failed.
    DeleteManifest,
    /// Deleting a manifest-list file failed.
    DeleteManifestList,
    /// Deleting a statistics or partition-statistics file failed.
    DeleteStatisticsFile,
    /// A candidate manifest could not be read. Its content files were skipped, which
    /// under-deletes. No retained snapshot references the manifest, so it is still deleted.
    ReadCandidateManifest,
    /// A retained manifest could not be read, so the whole content-file set was cleared. When the
    /// live set cannot be proven, no content file may die.
    ReadRetainedManifest,
}

/// One collected, non-aborting cleanup failure. Java logs and continues instead.
#[derive(Debug)]
pub struct CleanupFailure {
    /// The file the failed step was operating on.
    pub path: String,
    /// Which cleanup step failed.
    pub kind: CleanupFailureKind,
    /// The underlying error.
    pub error: Error,
}

/// The outcome of one cleanup sweep: every deleted path per funnel, plus every collected failure.
/// Paths sort within a funnel, and funnels follow Java's deletion order. The report is
/// `#[non_exhaustive]` because the cleanup produces it and callers only read it.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct CleanupReport {
    /// Deleted content files: data files, delete files, and deletion-vector Puffins.
    ///
    /// **This union is the authority on membership.** Every per-content-type view filters this
    /// vector, so a view can never name a file the union does not, including when the fail-closed
    /// posture clears the content set.
    pub deleted_content_files: Vec<String>,
    /// The content type of each path in [`Self::deleted_content_files`]. This is a type lookup,
    /// not a second membership list. The same walk fills both, so the key set equals the union.
    ///
    /// Read it through the typed accessors. It is a [`BTreeMap`] so the report's derived [`Debug`]
    /// output stays deterministic.
    pub deleted_content_file_types: BTreeMap<String, DataContentType>,
    /// Deleted manifest files (data and delete manifests).
    pub deleted_manifests: Vec<String>,
    /// Deleted manifest-list files.
    pub deleted_manifest_lists: Vec<String>,
    /// Deleted statistics / partition-statistics files.
    pub deleted_statistics_files: Vec<String>,
    /// Collected failures, in plan-then-sweep order. Empty means a fully clean sweep.
    pub failures: Vec<CleanupFailure>,
}

impl CleanupReport {
    /// True when the sweep deleted nothing and collected no failures.
    pub fn is_empty(&self) -> bool {
        self.deleted_content_files.is_empty()
            && self.deleted_manifests.is_empty()
            && self.deleted_manifest_lists.is_empty()
            && self.deleted_statistics_files.is_empty()
            && self.failures.is_empty()
    }

    /// The deleted content files of `content_type`, in the union's path order.
    ///
    /// This filters [`Self::deleted_content_files`] and stores nothing, so the union stays the
    /// single membership authority and an empty union empties every view.
    pub fn deleted_content_files_of_type(&self, content_type: DataContentType) -> Vec<&str> {
        self.deleted_content_files
            .iter()
            .filter(|path| {
                self.deleted_content_file_types.get(path.as_str()) == Some(&content_type)
            })
            .map(String::as_str)
            .collect()
    }

    /// The deleted data files. Spark's `deleted_data_files_count` column.
    pub fn deleted_data_files(&self) -> Vec<&str> {
        self.deleted_content_files_of_type(DataContentType::Data)
    }

    /// The deleted position-delete files. Spark's `deleted_position_delete_files_count` column.
    ///
    /// **A deletion-vector Puffin lands here.** Java tags a content file by `ContentFile.content()`
    /// alone, never by file format, and a vector is a delete file whose content is
    /// `POSITION_DELETES`. Java has no fourth bucket, so neither does this.
    pub fn deleted_position_delete_files(&self) -> Vec<&str> {
        self.deleted_content_files_of_type(DataContentType::PositionDeletes)
    }

    /// The deleted equality-delete files. Spark's `deleted_equality_delete_files_count` column.
    pub fn deleted_equality_delete_files(&self) -> Vec<&str> {
        self.deleted_content_files_of_type(DataContentType::EqualityDeletes)
    }
}

/// Post-commit file cleanup for snapshot expiry (Java `ReachableFileCleanup`). The module docs
/// carry the set algebra, the seam, and the failure posture.
///
/// **This deletes files.** Run it only against a successfully committed expiry, preferably through
/// [`Self::commit_and_clean`].
pub struct ExpireSnapshotsCleanup {
    file_io: FileIO,
    delete_function: Box<DeleteFunction>,
}

impl ExpireSnapshotsCleanup {
    /// Creates a cleanup with the default delete function ([`FileIO::delete`]).
    pub fn new(file_io: FileIO) -> Self {
        let delete_io = file_io.clone();
        ExpireSnapshotsCleanup {
            file_io,
            delete_function: Box::new(move |path: String| {
                let delete_io = delete_io.clone();
                Box::pin(async move { delete_io.delete(&path).await })
            }),
        }
    }

    /// Replaces the delete function (Java `ExpireSnapshots.deleteWith`). A recorder that returns
    /// `Ok` without deleting computes the would-be deletion set. Planning reads still use the
    /// construction-time [`FileIO`].
    pub fn delete_with(
        mut self,
        delete_function: impl Fn(String) -> BoxFuture<'static, Result<()>> + Send + Sync + 'static,
    ) -> Self {
        self.delete_function = Box::new(delete_function);
        self
    }

    /// Commits `transaction`, then cleans the files it expired only if the commit succeeded.
    /// This is Java's `RemoveSnapshots.commit()` ordering. A failed commit propagates before
    /// anything is planned or deleted.
    ///
    /// A pre-commit table with no snapshots commits but skips cleanup, matching Java's gate.
    pub async fn commit_and_clean(
        &self,
        transaction: Transaction,
        catalog: &dyn Catalog,
    ) -> Result<(Table, CleanupReport)> {
        let before = transaction.table.metadata_ref();
        // The `?` is the safety gate. A failed or uncertain commit returns here, so the deletion
        // path below is structurally unreachable.
        let committed = transaction.commit(catalog).await?;
        if before.snapshots().len() == 0 {
            return Ok((committed, CleanupReport::default()));
        }
        let report = self
            .clean_expired_files(&before, committed.metadata())
            .await?;
        Ok((committed, report))
    }

    /// Deletes every file that `before` reaches and `after` does not (Java `cleanFiles`). **`after`
    /// must be the table's current committed metadata.** Anything staler deletes files the live
    /// table still reaches. Prefer [`Self::commit_and_clean`].
    pub async fn clean_expired_files(
        &self,
        before: &TableMetadata,
        after: &TableMetadata,
    ) -> Result<CleanupReport> {
        // Java's gate sits in the RemoveSnapshots constructor, which covers cleanup because
        // cleanup runs on the same object. This standalone entry point must re-check, or a direct
        // call bypasses the gate the action enforced at commit.
        let gc_enabled = parse_property(
            before.properties(),
            TableProperties::PROPERTY_GC_ENABLED,
            TableProperties::PROPERTY_GC_ENABLED_DEFAULT,
        )?;
        if !gc_enabled {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Cannot expire snapshots: GC is disabled (deleting files may corrupt other tables)"
                    .to_string(),
            ));
        }

        // Step 1 and 2. The retained-shared filter on the lists is the Rust-only under-deletion
        // guard; Java deletes an expired manifest-list location unconditionally.
        let retained_ids: HashSet<i64> = after.snapshots().map(|s| s.snapshot_id()).collect();
        let expired_snapshots: Vec<&SnapshotRef> = before
            .snapshots()
            .filter(|snapshot| !retained_ids.contains(&snapshot.snapshot_id()))
            .collect();
        let retained_list_locations: HashSet<&str> =
            after.snapshots().map(|s| s.manifest_list()).collect();
        let manifest_lists_to_delete: BTreeSet<String> = expired_snapshots
            .iter()
            .map(|snapshot| snapshot.manifest_list())
            .filter(|location| !retained_list_locations.contains(location))
            .map(str::to_string)
            .collect();

        // Step 3. A manifest-list read failure on either side aborts here, before any deletion.
        let candidate_manifests = self
            .manifests_by_path(expired_snapshots.iter().copied(), before)
            .await?;
        let retained_manifests = self.manifests_by_path(after.snapshots(), after).await?;
        let manifests_to_delete: BTreeMap<String, ManifestFile> = candidate_manifests
            .into_iter()
            .filter(|(path, _)| !retained_manifests.contains_key(path))
            .collect();

        let mut report = CleanupReport::default();

        // Step 4, skipped entirely when no manifest dies. The map keys on path, so the set
        // algebra is unchanged. Two entries that share a path share a content type in every legal
        // shape, so the last-write-wins insert cannot misclassify.
        let mut content_files_to_delete: BTreeMap<String, DataContentType> = BTreeMap::new();
        if !manifests_to_delete.is_empty() {
            for (path, manifest_file) in &manifests_to_delete {
                match manifest_file.load_manifest(&self.file_io).await {
                    Ok(manifest) => {
                        for entry in manifest.entries() {
                            if entry.is_alive() {
                                content_files_to_delete
                                    .insert(entry.file_path().to_string(), entry.content_type());
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            path = %path,
                            kind = ?CleanupFailureKind::ReadCandidateManifest,
                            ?error,
                            "expire-snapshots cleanup: failed to read a candidate manifest; its \
                             content files are skipped (under-deletion) — the manifest itself is \
                             still deleted"
                        );
                        report.failures.push(CleanupFailure {
                            path: path.clone(),
                            kind: CleanupFailureKind::ReadCandidateManifest,
                            error,
                        });
                    }
                }
            }
            if !content_files_to_delete.is_empty() {
                for (path, manifest_file) in &retained_manifests {
                    match manifest_file.load_manifest(&self.file_io).await {
                        Ok(manifest) => {
                            for entry in manifest.entries() {
                                if entry.is_alive() {
                                    content_files_to_delete.remove(entry.file_path());
                                }
                            }
                        }
                        Err(error) => {
                            // Java catches any Throwable here and returns the empty set. When the
                            // live file set cannot be proven, no content file may die.
                            tracing::warn!(
                                path = %path,
                                kind = ?CleanupFailureKind::ReadRetainedManifest,
                                ?error,
                                "expire-snapshots cleanup: failed to read a retained manifest; \
                                 clearing the entire content-file deletion set (liveness cannot \
                                 be proven, so no content file may die)"
                            );
                            report.failures.push(CleanupFailure {
                                path: path.clone(),
                                kind: CleanupFailureKind::ReadRetainedManifest,
                                error,
                            });
                            content_files_to_delete.clear();
                            break;
                        }
                    }
                }
            }
        }

        let statistics_to_delete: BTreeSet<String> = statistics_locations(before)
            .difference(&statistics_locations(after))
            .cloned()
            .collect();

        // The sweep follows Java's deletion order and never aborts. The content funnel sweeps the
        // map's keys, then records the type of each path it deleted. Both come from
        // `content_files_to_delete`, so a path that failed to delete is in neither.
        let content_paths: BTreeSet<String> = content_files_to_delete.keys().cloned().collect();
        report.deleted_content_files = self
            .delete_all(
                content_paths,
                CleanupFailureKind::DeleteContentFile,
                &mut report.failures,
            )
            .await;
        let deleted_content: HashSet<&String> = report.deleted_content_files.iter().collect();
        report.deleted_content_file_types = content_files_to_delete
            .iter()
            .filter(|(path, _)| deleted_content.contains(path))
            .map(|(path, content_type)| (path.clone(), *content_type))
            .collect();
        report.deleted_manifests = self
            .delete_all(
                manifests_to_delete.into_keys().collect(),
                CleanupFailureKind::DeleteManifest,
                &mut report.failures,
            )
            .await;
        report.deleted_manifest_lists = self
            .delete_all(
                manifest_lists_to_delete,
                CleanupFailureKind::DeleteManifestList,
                &mut report.failures,
            )
            .await;
        report.deleted_statistics_files = self
            .delete_all(
                statistics_to_delete,
                CleanupFailureKind::DeleteStatisticsFile,
                &mut report.failures,
            )
            .await;

        Ok(report)
    }

    /// Reads the manifest lists of `snapshots` and collects every listed [`ManifestFile`] by
    /// path, data and delete alike. A manifest-list read failure is a hard error, and the caller
    /// runs this before any deletion.
    async fn manifests_by_path<'a>(
        &self,
        snapshots: impl Iterator<Item = &'a SnapshotRef>,
        metadata: &TableMetadata,
    ) -> Result<BTreeMap<String, ManifestFile>> {
        let mut manifests = BTreeMap::new();
        for snapshot in snapshots {
            let manifest_list = snapshot
                .load_manifest_list(&self.file_io, metadata)
                .await
                .map_err(|error| {
                    error.with_context(
                        "manifest_list",
                        format!(
                            "failed to read manifest list of snapshot {} during expire-snapshots \
                             cleanup planning (no files were deleted)",
                            snapshot.snapshot_id()
                        ),
                    )
                })?;
            for manifest_file in manifest_list.entries() {
                manifests.insert(manifest_file.manifest_path.clone(), manifest_file.clone());
            }
        }
        Ok(manifests)
    }

    /// Deletes every path through the injected delete function and returns the ones that
    /// succeeded, in sorted order. A failure is recorded under `kind`. This never aborts.
    async fn delete_all(
        &self,
        paths: BTreeSet<String>,
        kind: CleanupFailureKind,
        failures: &mut Vec<CleanupFailure>,
    ) -> Vec<String> {
        let mut deleted = Vec::with_capacity(paths.len());
        for path in paths {
            match (self.delete_function)(path.clone()).await {
                Ok(()) => deleted.push(path),
                Err(error) => {
                    tracing::warn!(
                        path = %path,
                        kind = ?kind,
                        ?error,
                        "expire-snapshots cleanup: failed to delete a file; collecting the \
                         failure and continuing the sweep"
                    );
                    failures.push(CleanupFailure { path, kind, error });
                }
            }
        }
        deleted
    }
}

/// Every statistics and partition-statistics location of `metadata`.
fn statistics_locations(metadata: &TableMetadata) -> BTreeSet<String> {
    metadata
        .statistics_iter()
        .map(|statistics| statistics.statistics_path.clone())
        .chain(
            metadata
                .partition_statistics_iter()
                .map(|statistics| statistics.statistics_path.clone()),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fmt::Write as _;
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use futures::future::BoxFuture;
    use tracing::field::{Field, Visit};
    use tracing::{Event, Subscriber};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;

    use super::{CleanupFailureKind, ExpireSnapshotsCleanup};
    use crate::error::Result;
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestContentType,
        ManifestStatus, Operation, Snapshot, StatisticsFile, Struct, Summary, TableProperties,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, Error, ErrorKind};

    /// Formats each `tracing` event onto a shared sink, so a test can assert on log events.
    struct CapturingLayer {
        sink: Arc<Mutex<Vec<String>>>,
    }

    /// Formats an event's message and structured fields into a single string.
    #[derive(Default)]
    struct StringVisitor(String);

    impl Visit for StringVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            let _ = write!(self.0, " {}={:?}", field.name(), value);
        }
    }

    impl<S> Layer<S> for CapturingLayer
    where S: Subscriber + for<'a> LookupSpan<'a>
    {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut visitor = StringVisitor::default();
            event.record(&mut visitor);
            self.sink
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(visitor.0);
        }
    }

    /// Fixture helpers — real tables in a memory catalog, real manifest /
    /// manifest-list files in the table's FileIO
    /// A synthetic data file routed to partition `x = 0` (metadata-only; the parquet bytes never
    /// exist — content-file deletions are asserted on the REPORT and the injected delete fn).
    fn synthetic_data_file(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .expect("build synthetic data file")
    }

    /// A synthetic deletion vector: PUFFIN-format position delete with the DV-required
    /// `referenced_data_file` / `content_offset` / `content_size_in_bytes`. Two DVs sharing one
    /// puffin differ only in referenced file + offset — the real multi-blob shape.
    fn synthetic_dv_file(puffin_path: &str, referenced_data_file: &str, offset: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(puffin_path.to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(200)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .referenced_data_file(Some(referenced_data_file.to_string()))
            .content_offset(Some(offset))
            .content_size_in_bytes(Some(40))
            .build()
            .expect("build synthetic dv file")
    }

    async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let tx = tx
            .fast_append()
            .add_data_files(files)
            .apply(tx)
            .expect("apply fast append");
        tx.commit(catalog).await.expect("commit fast append")
    }

    /// The manifest-list location of `snapshot_id`.
    fn list_path(table: &Table, snapshot_id: i64) -> String {
        table
            .metadata()
            .snapshot_by_id(snapshot_id)
            .expect("snapshot present")
            .manifest_list()
            .to_string()
    }

    /// `(manifest_path, content)` pairs listed by `snapshot_id`'s manifest list.
    async fn manifests_of(table: &Table, snapshot_id: i64) -> Vec<(String, ManifestContentType)> {
        let snapshot = table
            .metadata()
            .snapshot_by_id(snapshot_id)
            .expect("snapshot present");
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .expect("load manifest list");
        manifest_list
            .entries()
            .iter()
            .map(|manifest| (manifest.manifest_path.clone(), manifest.content))
            .collect()
    }

    async fn exists(table: &Table, path: &str) -> bool {
        table.file_io().exists(path).await.expect("exists check")
    }

    /// A recording delete function that succeeds without touching storage.
    #[allow(clippy::type_complexity)] // the tuple IS the seam: (recorded paths, injectable fn)
    fn recording_delete_fn() -> (
        Arc<Mutex<Vec<String>>>,
        impl Fn(String) -> BoxFuture<'static, Result<()>> + Send + Sync + 'static,
    ) {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&recorded);
        let delete_fn = move |path: String| -> BoxFuture<'static, Result<()>> {
            sink.lock().expect("recorder lock").push(path);
            Box::pin(async { Ok(()) })
        };
        (recorded, delete_fn)
    }

    /// Expires everything age-eligible, keeping each branch head, through `commit_and_clean`.
    async fn expire_and_clean(
        catalog: &impl Catalog,
        table: &Table,
        cleanup: &ExpireSnapshotsCleanup,
    ) -> (Table, super::CleanupReport) {
        let tx = Transaction::new(table);
        let tx = tx
            .expire_snapshots()
            .expire_older_than(i64::MAX)
            .retain_last(1)
            .apply(tx)
            .expect("apply expire");
        cleanup
            .commit_and_clean(tx, catalog)
            .await
            .expect("commit and clean")
    }

    /// Commits the same expiry without cleanup, for a test that intervenes in between.
    async fn expire_metadata_only(catalog: &impl Catalog, table: &Table) -> (Table, Table) {
        let tx = Transaction::new(table);
        let tx = tx
            .expire_snapshots()
            .expire_older_than(i64::MAX)
            .retain_last(1)
            .apply(tx)
            .expect("apply expire");
        let committed = tx.commit(catalog).await.expect("commit expire");
        (table.clone(), committed)
    }

    /// The deletion-set pins — every class, both directions
    /// A fast-append chain carries manifests forward, so an expired snapshot's manifest usually
    /// survives in a retained descendant. Dropping the candidates-minus-retained subtraction
    /// deletes that shared manifest and destroys the retained snapshot's data. The expired
    /// snapshot's own manifest list must still die.
    #[tokio::test]
    async fn test_carried_forward_shared_manifest_survives_expired_list_dies() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/a.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let table2 = append(&catalog, &table1, vec![synthetic_data_file(
            "test/wtB2/b.parquet",
        )])
        .await;
        let s2 = table2.metadata().current_snapshot_id().expect("s2");

        let s1_list = list_path(&table2, s1);
        let s2_list = list_path(&table2, s2);
        let m1 = manifests_of(&table2, s1).await[0].0.clone();
        // Pre-flight: the fixture really shares — S2's list carries M1 forward.
        let s2_manifest_paths: Vec<String> = manifests_of(&table2, s2)
            .await
            .into_iter()
            .map(|(path, _)| path)
            .collect();
        assert!(
            s2_manifest_paths.contains(&m1),
            "fixture must carry M1 forward: {s2_manifest_paths:?}"
        );

        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone());
        let (expired_table, report) = expire_and_clean(&catalog, &table2, &cleanup).await;
        assert!(expired_table.metadata().snapshot_by_id(s1).is_none());

        assert_eq!(report.deleted_manifest_lists, vec![s1_list.clone()]);
        assert!(
            report.deleted_manifests.is_empty(),
            "the SHARED manifest must survive: {:?}",
            report.deleted_manifests
        );
        assert!(report.deleted_content_files.is_empty());
        assert!(report.failures.is_empty());
        assert!(!exists(&table2, &s1_list).await, "expired list must die");
        assert!(exists(&table2, &m1).await, "shared manifest must survive");
        assert!(exists(&table2, &s2_list).await);
    }

    /// A data file rewritten into a new manifest stays live while its original manifest dies.
    /// Dropping the retained-side live-file subtraction deletes live data.
    #[tokio::test]
    async fn test_rewritten_but_live_data_file_survives_its_old_manifest_dies() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let data_path = "test/wtB2/rewritten-live.parquet";
        let table1 = append(&catalog, &table, vec![synthetic_data_file(data_path)]).await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let m1 = manifests_of(&table1, s1).await[0].0.clone();

        // S2: cluster every data manifest into a fresh one — the entry for `data_path` is
        // carried as EXISTING in a NEW manifest file.
        let tx = Transaction::new(&table1);
        let tx = tx
            .rewrite_manifests()
            .cluster_by(|_file| "all".to_string())
            .apply(tx)
            .expect("apply rewrite manifests");
        let table2 = tx.commit(&catalog).await.expect("commit rewrite manifests");
        let s2 = table2.metadata().current_snapshot_id().expect("s2");
        let s2_manifests = manifests_of(&table2, s2).await;
        assert!(
            !s2_manifests.iter().any(|(path, _)| path == &m1),
            "pre-flight: the rewrite must have replaced M1: {s2_manifests:?}"
        );

        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone());
        let (_, report) = expire_and_clean(&catalog, &table2, &cleanup).await;

        assert_eq!(report.deleted_manifests, vec![m1.clone()]);
        assert!(
            report.deleted_content_files.is_empty(),
            "the still-live data file must NOT die: {:?}",
            report.deleted_content_files
        );
        assert!(report.failures.is_empty());
        assert!(!exists(&table2, &m1).await);
        for (path, _) in &s2_manifests {
            assert!(
                exists(&table2, path).await,
                "retained manifest {path} must survive"
            );
        }
    }

    /// A data file live only in expired snapshots must die. The retained manifest's DELETED
    /// tombstone must not protect it: liveness excludes DELETED on the retained side too.
    #[tokio::test]
    async fn test_data_file_only_in_expired_snapshots_dies_tombstone_does_not_protect() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let data_path = "test/wtB2/expired-only.parquet";
        let table1 = append(&catalog, &table, vec![synthetic_data_file(data_path)]).await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let m1 = manifests_of(&table1, s1).await[0].0.clone();

        // S2: delete the data file (copy-on-write tombstone in a rewritten manifest).
        let tx = Transaction::new(&table1);
        let tx = tx
            .delete_files()
            .delete_file(data_path)
            .apply(tx)
            .expect("apply delete files");
        let table2 = tx.commit(&catalog).await.expect("commit delete files");
        let s2 = table2.metadata().current_snapshot_id().expect("s2");
        // Prove the fixture reaches the path under test.
        let s2_manifests = manifests_of(&table2, s2).await;
        let mut tombstone_seen = false;
        for (path, _) in &s2_manifests {
            assert_ne!(path, &m1, "pre-flight: M1 must have been rewritten");
            let manifest_file = table2
                .metadata()
                .snapshot_by_id(s2)
                .expect("s2 snapshot")
                .load_manifest_list(table2.file_io(), table2.metadata())
                .await
                .expect("s2 list")
                .entries()
                .iter()
                .find(|m| &m.manifest_path == path)
                .expect("listed manifest")
                .clone();
            let manifest = manifest_file
                .load_manifest(table2.file_io())
                .await
                .expect("load retained manifest");
            tombstone_seen |= manifest.entries().iter().any(|entry| {
                entry.status() == ManifestStatus::Deleted && entry.file_path() == data_path
            });
        }
        assert!(
            tombstone_seen,
            "pre-flight: tombstone for {data_path} must exist"
        );

        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone());
        let (_, report) = expire_and_clean(&catalog, &table2, &cleanup).await;

        assert_eq!(
            report.deleted_content_files,
            vec![data_path.to_string()],
            "the expired-only data file must die, exactly once"
        );
        assert_eq!(report.deleted_manifests, vec![m1.clone()]);
        assert!(report.failures.is_empty());
        for (path, _) in &s2_manifests {
            assert!(
                exists(&table2, path).await,
                "retained manifest {path} must survive"
            );
        }
    }

    /// A Puffin that a dying delete manifest and a retained one both reference must survive. A
    /// Puffin only the dying manifest references must die. Both directions in one fixture.
    ///
    /// The shape shares across manifests, not within one Puffin. Java removes a delete file by
    /// path, so removing one vector of a shared Puffin tombstones every same-path entry and the
    /// Puffin does become unreachable. The cross-manifest share is the real hazard.
    #[tokio::test]
    async fn test_shared_puffin_with_one_retained_dv_survives() {
        let catalog = new_memory_catalog().await;
        let table = crate::transaction::tests::make_v3_minimal_table_in_catalog(&catalog).await;
        let (a_path, c_path) = ("test/wtB2/dv-a.parquet", "test/wtB2/dv-c.parquet");
        let table1 = append(&catalog, &table, vec![
            synthetic_data_file(a_path),
            synthetic_data_file(c_path),
        ])
        .await;

        // S2: one delete manifest holding DV-A in the shared puffin and DV-C in its own.
        let shared_puffin = "test/wtB2/shared.puffin";
        let replaced_puffin = "test/wtB2/replaced.puffin";
        let dv_a = synthetic_dv_file(shared_puffin, a_path, 4);
        let dv_c = synthetic_dv_file(replaced_puffin, c_path, 4);
        let tx = Transaction::new(&table1);
        let tx = tx
            .row_delta()
            .add_deletes(vec![dv_a, dv_c.clone()])
            .apply(tx)
            .expect("apply row delta dvs");
        let table2 = tx.commit(&catalog).await.expect("commit row delta dvs");
        let s2 = table2.metadata().current_snapshot_id().expect("s2");
        let dm1 = manifests_of(&table2, s2)
            .await
            .into_iter()
            .find(|(_, content)| *content == ManifestContentType::Deletes)
            .expect("S2 delete manifest")
            .0;

        // S3: replace DV-C (remove + successor puffin). The rewrite carries DV-A as EXISTING —
        // still at the shared puffin path — into the retained rewritten delete manifest.
        let dv_c2 = synthetic_dv_file("test/wtB2/successor.puffin", c_path, 4);
        let tx = Transaction::new(&table2);
        let tx = tx
            .row_delta()
            .remove_deletes(dv_c)
            .add_deletes(vec![dv_c2])
            .apply(tx)
            .expect("apply dv replacement");
        let table3 = tx.commit(&catalog).await.expect("commit dv replacement");
        let s3 = table3.metadata().current_snapshot_id().expect("s3");
        let s3_manifests = manifests_of(&table3, s3).await;
        assert!(
            !s3_manifests.iter().any(|(path, _)| path == &dm1),
            "pre-flight: DM1 must have been rewritten by the removal: {s3_manifests:?}"
        );

        let cleanup = ExpireSnapshotsCleanup::new(table3.file_io().clone());
        let (_, report) = expire_and_clean(&catalog, &table3, &cleanup).await;

        assert_eq!(report.deleted_manifests, vec![dm1]);
        assert_eq!(
            report.deleted_content_files,
            vec![replaced_puffin.to_string()],
            "the SHARED puffin must survive (DV-A lives EXISTING in the retained rewrite); \
             only the fully-replaced puffin dies"
        );
        assert!(report.failures.is_empty());
    }

    /// A Puffin whose every vector expired is referenced by no retained manifest and must die.
    /// The successor Puffin and the data files stay.
    #[tokio::test]
    async fn test_expired_only_dv_puffin_dies() {
        let catalog = new_memory_catalog().await;
        let table = crate::transaction::tests::make_v3_minimal_table_in_catalog(&catalog).await;
        let a_path = "test/wtB2/dv-only.parquet";
        let table1 = append(&catalog, &table, vec![synthetic_data_file(a_path)]).await;

        let old_puffin = "test/wtB2/old.puffin";
        let dv_a = synthetic_dv_file(old_puffin, a_path, 4);
        let tx = Transaction::new(&table1);
        let tx = tx
            .row_delta()
            .add_deletes(vec![dv_a.clone()])
            .apply(tx)
            .expect("apply row delta dv");
        let table2 = tx.commit(&catalog).await.expect("commit row delta dv");

        let dv_a2 = synthetic_dv_file("test/wtB2/new.puffin", a_path, 4);
        let tx = Transaction::new(&table2);
        let tx = tx
            .row_delta()
            .remove_deletes(dv_a)
            .add_deletes(vec![dv_a2])
            .apply(tx)
            .expect("apply dv replacement");
        let table3 = tx.commit(&catalog).await.expect("commit dv replacement");

        let cleanup = ExpireSnapshotsCleanup::new(table3.file_io().clone());
        let (_, report) = expire_and_clean(&catalog, &table3, &cleanup).await;

        assert_eq!(
            report.deleted_content_files,
            vec![old_puffin.to_string()],
            "the fully-expired puffin must die — and ONLY it (data file + successor survive)"
        );
        assert!(report.failures.is_empty());
    }

    /// The Rust-only safety divergence: this port spares a manifest-list location a retained
    /// snapshot also references, where Java deletes unconditionally. No Java writer produces the
    /// shape, but unconditional deletion would destroy the retained snapshot.
    #[tokio::test]
    async fn test_manifest_list_shared_with_retained_snapshot_survives() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/shared-list.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let s1_list = list_path(&table1, s1);

        // "before" = the real table plus a grafted dangling snapshot X whose manifest_list
        // points at S1's list file; "after" = the real table (X expired, S1 retained).
        let grafted = Snapshot::builder()
            .with_snapshot_id(999_001)
            .with_parent_snapshot_id(Some(s1))
            .with_sequence_number(table1.metadata().last_sequence_number() + 1)
            .with_timestamp_ms(table1.metadata().last_updated_ms())
            .with_manifest_list(s1_list.clone())
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties: HashMap::new(),
            })
            .with_schema_id(0)
            .build();
        let before = table1
            .metadata()
            .clone()
            .into_builder(None)
            .add_snapshot(grafted)
            .expect("graft snapshot")
            .build()
            .expect("build grafted metadata")
            .metadata;

        let cleanup = ExpireSnapshotsCleanup::new(table1.file_io().clone());
        let report = cleanup
            .clean_expired_files(&before, table1.metadata())
            .await
            .expect("clean expired files");

        assert!(
            report.is_empty(),
            "the shared manifest list (and everything under it) must survive: {report:?}"
        );
        assert!(exists(&table1, &s1_list).await);
    }

    /// An expired snapshot's statistics die and a retained snapshot's survive.
    #[tokio::test]
    async fn test_expired_snapshot_statistics_file_dies_retained_one_survives() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/stats-a.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let table2 = append(&catalog, &table1, vec![synthetic_data_file(
            "test/wtB2/stats-b.parquet",
        )])
        .await;
        let s2 = table2.metadata().current_snapshot_id().expect("s2");

        let stats_path = |id: i64| {
            format!(
                "{}/metadata/wtB2-stats-{id}.puffin",
                table2.metadata().location()
            )
        };
        let mut table_with_stats = table2.clone();
        for id in [s1, s2] {
            let path = stats_path(id);
            table_with_stats
                .file_io()
                .new_output(&path)
                .expect("stats output")
                .write(Bytes::from_static(b"wtB2 stats fixture"))
                .await
                .expect("write stats fixture");
            let tx = Transaction::new(&table_with_stats);
            let tx = tx
                .update_statistics()
                .set_statistics(StatisticsFile {
                    snapshot_id: id,
                    statistics_path: path,
                    file_size_in_bytes: 18,
                    file_footer_size_in_bytes: 4,
                    key_metadata: None,
                    blob_metadata: vec![],
                })
                .apply(tx)
                .expect("apply set statistics");
            table_with_stats = tx.commit(&catalog).await.expect("commit statistics");
        }

        let cleanup = ExpireSnapshotsCleanup::new(table_with_stats.file_io().clone());
        let (_, report) = expire_and_clean(&catalog, &table_with_stats, &cleanup).await;

        assert_eq!(report.deleted_statistics_files, vec![stats_path(s1)]);
        assert!(report.failures.is_empty());
        assert!(!exists(&table_with_stats, &stats_path(s1)).await);
        assert!(
            exists(&table_with_stats, &stats_path(s2)).await,
            "the retained snapshot's statistics must survive"
        );
    }

    /// The seam pins — commit ordering, dry-run, failure posture, gates
    /// A failed commit must make deletion impossible. A refusing catalog must propagate the
    /// error with zero delete calls and storage untouched.
    #[tokio::test]
    async fn test_failed_commit_performs_zero_deletions() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/failed-commit.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let table2 = append(&catalog, &table1, vec![synthetic_data_file(
            "test/wtB2/failed-commit-2.parquet",
        )])
        .await;
        let s1_list = list_path(&table2, s1);

        // A catalog that loads the real table but refuses the commit, non-retryably.
        let mut failing_catalog = crate::catalog::MockCatalog::new();
        let loaded = table2.clone();
        failing_catalog.expect_load_table().returning_st(move |_| {
            let table = loaded.clone();
            Box::pin(async move { Ok(table) })
        });
        failing_catalog
            .expect_update_table()
            .times(1)
            .returning_st(|_| {
                Box::pin(async {
                    Err(Error::new(ErrorKind::Unexpected, "injected commit failure")
                        .with_retryable(false))
                })
            });

        let (recorded, delete_fn) = recording_delete_fn();
        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone()).delete_with(delete_fn);
        let tx = Transaction::new(&table2);
        let tx = tx
            .expire_snapshots()
            .expire_older_than(i64::MAX)
            .retain_last(1)
            .apply(tx)
            .expect("apply expire");

        let result = cleanup.commit_and_clean(tx, &failing_catalog).await;
        assert!(result.is_err(), "the failed commit must propagate");
        assert!(
            recorded.lock().expect("recorder lock").is_empty(),
            "NO deletion may run on a failed commit"
        );
        assert!(exists(&table2, &s1_list).await, "storage must be untouched");
    }

    /// An injected recorder computes the full would-be deletion set while every file survives.
    #[tokio::test]
    async fn test_dry_run_by_injection_leaves_storage_untouched() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/dry-run.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let table2 = append(&catalog, &table1, vec![synthetic_data_file(
            "test/wtB2/dry-run-2.parquet",
        )])
        .await;
        let s1_list = list_path(&table2, s1);

        let (recorded, delete_fn) = recording_delete_fn();
        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone()).delete_with(delete_fn);
        let (_, report) = expire_and_clean(&catalog, &table2, &cleanup).await;

        assert_eq!(report.deleted_manifest_lists, vec![s1_list.clone()]);
        assert_eq!(
            *recorded.lock().expect("recorder lock"),
            vec![s1_list.clone()],
            "the recorder must see exactly the would-be deletion set"
        );
        assert!(
            exists(&table2, &s1_list).await,
            "dry-run must leave storage untouched"
        );
    }

    /// A delete failure is collected and the sweep continues. It never aborts and leaves the rest
    /// unreported, and it never swallows the error.
    #[tokio::test]
    async fn test_injected_delete_failure_is_collected_and_sweep_continues() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/fail-collect-1.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let table2 = append(&catalog, &table1, vec![synthetic_data_file(
            "test/wtB2/fail-collect-2.parquet",
        )])
        .await;
        let s2 = table2.metadata().current_snapshot_id().expect("s2");
        let table3 = append(&catalog, &table2, vec![synthetic_data_file(
            "test/wtB2/fail-collect-3.parquet",
        )])
        .await;
        let s1_list = list_path(&table3, s1);
        let s2_list = list_path(&table3, s2);

        // Fail exactly S1's list; delete everything else for real.
        let io = table3.file_io().clone();
        let fail_path = s1_list.clone();
        let cleanup = ExpireSnapshotsCleanup::new(table3.file_io().clone()).delete_with(
            move |path: String| -> BoxFuture<'static, Result<()>> {
                let io = io.clone();
                let fail_path = fail_path.clone();
                Box::pin(async move {
                    if path == fail_path {
                        Err(Error::new(ErrorKind::Unexpected, "injected delete failure"))
                    } else {
                        io.delete(&path).await
                    }
                })
            },
        );
        // The failure must be both returned in the report and warn-logged.
        let logs = Arc::new(Mutex::new(Vec::<String>::new()));
        let subscriber = tracing_subscriber::registry().with(CapturingLayer { sink: logs.clone() });
        let (report, captured) = {
            let _log_guard = tracing::subscriber::set_default(subscriber);
            let (_, report) = expire_and_clean(&catalog, &table3, &cleanup).await;
            let captured = logs.lock().unwrap_or_else(|p| p.into_inner()).clone();
            (report, captured)
        };

        assert_eq!(
            report.deleted_manifest_lists,
            vec![s2_list.clone()],
            "the sweep must continue past the failure"
        );
        assert_eq!(report.failures.len(), 1, "failures: {:?}", report.failures);
        assert_eq!(report.failures[0].path, s1_list);
        assert_eq!(
            report.failures[0].kind,
            CleanupFailureKind::DeleteManifestList
        );
        assert!(exists(&table3, &s1_list).await);
        assert!(!exists(&table3, &s2_list).await);

        // Removing the `warn!` in `delete_all` reddens this assertion.
        assert!(
            captured.iter().any(|line| {
                line.contains("failed to delete a file") && line.contains(&s1_list)
            }),
            "the collected delete failure must ALSO be warn-logged with the path: {captured:?}"
        );
    }

    /// An unreadable retained manifest clears the whole content-file set: liveness cannot be
    /// proven, so no content file may die. Manifests and lists still sweep, and the failure is
    /// reported.
    #[tokio::test]
    async fn test_unreadable_retained_manifest_spares_all_content_files() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/spared.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let m1 = manifests_of(&table1, s1).await[0].0.clone();
        let tx = Transaction::new(&table1);
        let tx = tx
            .rewrite_manifests()
            .cluster_by(|_file| "all".to_string())
            .apply(tx)
            .expect("apply rewrite manifests");
        let table2 = tx.commit(&catalog).await.expect("commit rewrite manifests");
        let s2 = table2.metadata().current_snapshot_id().expect("s2");
        let m2 = manifests_of(&table2, s2).await[0].0.clone();

        let (before, after) = expire_metadata_only(&catalog, &table2).await;
        // Corrupt the RETAINED manifest after the commit, before the cleanup.
        table2
            .file_io()
            .new_output(&m2)
            .expect("corrupt output")
            .write(Bytes::from_static(b"wtB2 corrupted avro"))
            .await
            .expect("corrupt retained manifest");

        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone());
        let report = cleanup
            .clean_expired_files(before.metadata(), after.metadata())
            .await
            .expect("clean expired files");

        assert!(
            report.deleted_content_files.is_empty(),
            "no content file may die when a retained manifest is unreadable: {:?}",
            report.deleted_content_files
        );
        assert_eq!(report.deleted_manifests, vec![m1]);
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].path, m2);
        assert_eq!(
            report.failures[0].kind,
            CleanupFailureKind::ReadRetainedManifest
        );
    }

    /// An unreadable candidate manifest skips only its own content files, is itself still
    /// deleted, and the failure is reported.
    #[tokio::test]
    async fn test_unreadable_candidate_manifest_skips_its_files_but_still_dies() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/skipped.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let m1 = manifests_of(&table1, s1).await[0].0.clone();
        let tx = Transaction::new(&table1);
        let tx = tx
            .rewrite_manifests()
            .cluster_by(|_file| "all".to_string())
            .apply(tx)
            .expect("apply rewrite manifests");
        let table2 = tx.commit(&catalog).await.expect("commit rewrite manifests");

        let (before, after) = expire_metadata_only(&catalog, &table2).await;
        table2
            .file_io()
            .new_output(&m1)
            .expect("corrupt output")
            .write(Bytes::from_static(b"wtB2 corrupted avro"))
            .await
            .expect("corrupt candidate manifest");

        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone());
        let report = cleanup
            .clean_expired_files(before.metadata(), after.metadata())
            .await
            .expect("clean expired files");

        assert!(report.deleted_content_files.is_empty());
        assert_eq!(
            report.deleted_manifests,
            vec![m1.clone()],
            "the unreadable expired-only manifest itself must still die"
        );
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].path, m1);
        assert_eq!(
            report.failures[0].kind,
            CleanupFailureKind::ReadCandidateManifest
        );
    }

    /// An unreadable manifest list aborts with `Err` before any deletion, because planning runs
    /// strictly before the sweep.
    #[tokio::test]
    async fn test_unreadable_manifest_list_aborts_before_any_deletion() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/abort.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let table2 = append(&catalog, &table1, vec![synthetic_data_file(
            "test/wtB2/abort-2.parquet",
        )])
        .await;
        let s1_list = list_path(&table2, s1);
        let m1 = manifests_of(&table2, s1).await[0].0.clone();

        let (before, after) = expire_metadata_only(&catalog, &table2).await;
        table2
            .file_io()
            .new_output(&s1_list)
            .expect("corrupt output")
            .write(Bytes::from_static(b"wtB2 corrupted avro"))
            .await
            .expect("corrupt manifest list");

        let (recorded, delete_fn) = recording_delete_fn();
        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone()).delete_with(delete_fn);
        let error = cleanup
            .clean_expired_files(before.metadata(), after.metadata())
            .await
            .expect_err("unreadable manifest list must abort");
        assert!(
            error.to_string().contains("cleanup planning"),
            "the planning context must name the abort: {error}"
        );
        assert!(
            recorded.lock().expect("recorder lock").is_empty(),
            "nothing may be deleted on a planning abort"
        );
        assert!(exists(&table2, &m1).await);
    }

    /// An empty expiry deletes nothing, so scheduled maintenance can run the cleanup
    /// unconditionally.
    #[tokio::test]
    async fn test_empty_expiry_is_a_noop() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/noop.parquet",
        )])
        .await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        let s1_list = list_path(&table1, s1);

        let cleanup = ExpireSnapshotsCleanup::new(table1.file_io().clone());
        let report = cleanup
            .clean_expired_files(table1.metadata(), table1.metadata())
            .await
            .expect("clean expired files");
        assert!(
            report.is_empty(),
            "no-op expiry must report nothing: {report:?}"
        );
        assert!(exists(&table1, &s1_list).await);

        // And through the wrapper: an expiry that expires nothing cleans nothing.
        let tx = Transaction::new(&table1);
        let tx = tx
            .expire_snapshots()
            .expire_older_than(0)
            .apply(tx)
            .expect("apply no-op expire");
        let (_, report) = cleanup
            .commit_and_clean(tx, &catalog)
            .await
            .expect("commit and clean no-op");
        assert!(report.is_empty());
        assert!(exists(&table1, &s1_list).await);
    }

    /// The sweep follows Java's funnel order, so a leaf dies before the structure that indexes
    /// it. A crash mid-sweep leaves the expired manifest lists readable, and a re-run can still
    /// plan the remainder. A lists-first sweep would orphan everything beneath them.
    ///
    /// No per-funnel report assertion can see cross-funnel order, so this pins the recorder's raw
    /// invocation sequence.
    #[tokio::test]
    async fn test_sweep_order_content_manifests_lists_statistics() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let data_path = "test/wtB2/order.parquet";
        let table1 = append(&catalog, &table, vec![synthetic_data_file(data_path)]).await;
        let s1 = table1.metadata().current_snapshot_id().expect("s1");
        // Stats on S1 (die with it); the delete_files rewrite then makes S1's manifest and the
        // data file expired-only — all four funnels non-empty in one sweep.
        let stats_path = format!(
            "{}/metadata/wtB2-order-stats.puffin",
            table1.metadata().location()
        );
        table1
            .file_io()
            .new_output(&stats_path)
            .expect("stats output")
            .write(Bytes::from_static(b"wtB2 stats fixture"))
            .await
            .expect("write stats fixture");
        let tx = Transaction::new(&table1);
        let tx = tx
            .update_statistics()
            .set_statistics(StatisticsFile {
                snapshot_id: s1,
                statistics_path: stats_path.clone(),
                file_size_in_bytes: 18,
                file_footer_size_in_bytes: 4,
                key_metadata: None,
                blob_metadata: vec![],
            })
            .apply(tx)
            .expect("apply set statistics");
        let table1 = tx.commit(&catalog).await.expect("commit statistics");
        let tx = Transaction::new(&table1);
        let tx = tx
            .delete_files()
            .delete_file(data_path)
            .apply(tx)
            .expect("apply delete files");
        let table2 = tx.commit(&catalog).await.expect("commit delete files");

        let (recorded, delete_fn) = recording_delete_fn();
        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone()).delete_with(delete_fn);
        let (_, report) = expire_and_clean(&catalog, &table2, &cleanup).await;

        // An unexercised funnel would make the order pin vacuous.
        assert_eq!(report.deleted_content_files, vec![data_path.to_string()]);
        assert!(!report.deleted_manifests.is_empty(), "fixture: no manifest");
        assert!(
            !report.deleted_manifest_lists.is_empty(),
            "fixture: no list"
        );
        assert_eq!(report.deleted_statistics_files, vec![stats_path]);
        assert!(report.failures.is_empty());
        let expected: Vec<String> = report
            .deleted_content_files
            .iter()
            .chain(&report.deleted_manifests)
            .chain(&report.deleted_manifest_lists)
            .chain(&report.deleted_statistics_files)
            .cloned()
            .collect();
        assert_eq!(
            *recorded.lock().expect("recorder lock"),
            expected,
            "the sweep must run content → manifests → manifest lists → statistics"
        );
    }

    /// Running the same cleanup twice must never over-delete and never panic. After a complete
    /// first sweep the expired manifest lists are gone, so the second run aborts at planning with
    /// zero delete calls. A sweep interrupted earlier re-plans from the intact lists.
    #[tokio::test]
    async fn test_rerun_after_complete_sweep_aborts_at_planning_with_zero_deletions() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let data_path = "test/wtB2/rerun.parquet";
        let table1 = append(&catalog, &table, vec![synthetic_data_file(data_path)]).await;
        let tx = Transaction::new(&table1);
        let tx = tx
            .delete_files()
            .delete_file(data_path)
            .apply(tx)
            .expect("apply delete files");
        let table2 = tx.commit(&catalog).await.expect("commit delete files");

        let cleanup = ExpireSnapshotsCleanup::new(table2.file_io().clone());
        let (before, after) = expire_metadata_only(&catalog, &table2).await;
        let first = cleanup
            .clean_expired_files(before.metadata(), after.metadata())
            .await
            .expect("first sweep");
        assert!(!first.deleted_manifest_lists.is_empty());
        assert!(first.failures.is_empty());

        let (recorded, delete_fn) = recording_delete_fn();
        let rerun = ExpireSnapshotsCleanup::new(table2.file_io().clone()).delete_with(delete_fn);
        let error = rerun
            .clean_expired_files(before.metadata(), after.metadata())
            .await
            .expect_err("re-run must abort at planning: the expired manifest lists are gone");
        assert!(
            error.to_string().contains("cleanup planning"),
            "the planning context must name the abort: {error}"
        );
        assert!(
            recorded.lock().expect("recorder lock").is_empty(),
            "the re-run may delete NOTHING"
        );
    }

    /// The `gc.enabled` gate is re-honored at the cleanup door, so a direct call cannot bypass
    /// the gate the action enforced at commit.
    #[tokio::test]
    async fn test_gc_disabled_cleanup_refused() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let table1 = append(&catalog, &table, vec![synthetic_data_file(
            "test/wtB2/gc-gate.parquet",
        )])
        .await;
        let disabled = table1
            .metadata()
            .clone()
            .into_builder(None)
            .set_properties(HashMap::from([(
                TableProperties::PROPERTY_GC_ENABLED.to_string(),
                "false".to_string(),
            )]))
            .expect("set gc.enabled")
            .build()
            .expect("build gc-disabled metadata")
            .metadata;

        let cleanup = ExpireSnapshotsCleanup::new(table1.file_io().clone());
        let error = cleanup
            .clean_expired_files(&disabled, &disabled)
            .await
            .expect_err("gc.enabled=false must refuse cleanup");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            error.message(),
            "Cannot expire snapshots: GC is disabled (deleting files may corrupt other tables)"
        );

        // The gate reads the pre-expiry state, as Java's constructor does. A gate that read only
        // `after` would run a cleanup the expiry's own commit-time gate refused.
        let error = cleanup
            .clean_expired_files(&disabled, table1.metadata())
            .await
            .expect_err("before=disabled / after=enabled must still refuse");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            error.message(),
            "Cannot expire snapshots: GC is disabled (deleting files may corrupt other tables)"
        );
    }

    /// Content-type classification of the deleted content funnel (F-2)
    /// A parquet position-delete file, not a deletion vector, so the position-delete bucket is
    /// pinned by a non-Puffin file too.
    fn synthetic_position_delete_file(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .expect("build synthetic position delete file")
    }

    /// A synthetic EQUALITY-delete file (equality on field id 1, `x`), partition `x = 0`.
    fn synthetic_equality_delete_file(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .equality_ids(Some(vec![1]))
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .expect("build synthetic equality delete file")
    }

    /// The union must equal the three typed views concatenated. Java's `FileContent` has exactly
    /// three members, so the views partition the funnel with nothing left over.
    fn assert_union_is_concatenation_of_parts(report: &super::CleanupReport) {
        let mut concatenated: Vec<&str> = report.deleted_data_files();
        concatenated.extend(report.deleted_position_delete_files());
        concatenated.extend(report.deleted_equality_delete_files());
        concatenated.sort_unstable();
        assert_eq!(
            concatenated, report.deleted_content_files,
            "the three typed views must partition the union exactly — no file may be dropped \
             by, or duplicated across, the classification"
        );
    }

    /// Builds and expires a table that kills one data file, one parquet position delete, and one
    /// equality delete. Two tests share it so the partition invariant can be asserted alone: after
    /// exhaustive per-bucket equality it would be dominated and could not be mutation-proven.
    async fn expire_one_file_of_each_content_type() -> (
        &'static str,
        &'static str,
        &'static str,
        super::CleanupReport,
    ) {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let data_path = "test/wtB2/f2-a-data.parquet";
        let pos_path = "test/wtB2/f2-b-pos-delete.parquet";
        let eq_path = "test/wtB2/f2-c-eq-delete.parquet";

        let data_file = synthetic_data_file(data_path);
        let table1 = append(&catalog, &table, vec![data_file.clone()]).await;

        let pos_delete = synthetic_position_delete_file(pos_path);
        let eq_delete = synthetic_equality_delete_file(eq_path);
        let tx = Transaction::new(&table1);
        let tx = tx
            .row_delta()
            .add_deletes(vec![pos_delete.clone(), eq_delete.clone()])
            .apply(tx)
            .expect("apply row delta deletes");
        let table2 = tx.commit(&catalog).await.expect("commit row delta deletes");

        // One commit, so the retained head holds only tombstones and all three become
        // expired-only live entries.
        let tx = Transaction::new(&table2);
        let tx = tx
            .row_delta()
            .remove_data_files(vec![data_file])
            .remove_deletes_many(vec![pos_delete, eq_delete])
            .apply(tx)
            .expect("apply removals");
        let table3 = tx.commit(&catalog).await.expect("commit removals");

        let cleanup = ExpireSnapshotsCleanup::new(table3.file_io().clone());
        let (_, report) = expire_and_clean(&catalog, &table3, &cleanup).await;

        assert!(report.failures.is_empty(), "{:?}", report.failures);
        // An empty funnel would make every caller's assertion vacuous.
        assert_eq!(
            report.deleted_content_files,
            vec![
                data_path.to_string(),
                pos_path.to_string(),
                eq_path.to_string()
            ],
            "fixture must delete exactly the three content files, sorted"
        );
        (data_path, pos_path, eq_path, report)
    }

    /// The content funnel splits by entry content type, so a consumer can fill Spark's three
    /// per-type columns separately. Each of the three killed files lands in its own view.
    #[tokio::test]
    async fn test_deleted_content_files_split_by_content_type() {
        let (data_path, pos_path, eq_path, report) = expire_one_file_of_each_content_type().await;

        assert_eq!(report.deleted_data_files(), vec![data_path]);
        assert_eq!(report.deleted_position_delete_files(), vec![pos_path]);
        assert_eq!(report.deleted_equality_delete_files(), vec![eq_path]);
    }

    /// The three typed views partition the union: the classification drops no file and counts
    /// none twice. Asserted alone, so a bucket that drops a file reddens this assertion.
    #[tokio::test]
    async fn test_typed_views_partition_the_deleted_content_union() {
        let (_, _, _, report) = expire_one_file_of_each_content_type().await;
        assert_union_is_concatenation_of_parts(&report);
    }

    /// A deletion-vector Puffin counts as a position delete, not as a fourth class. Java tags a
    /// file by `content()` alone and never consults the format, so a vector is not separable from
    /// a parquet position delete.
    #[tokio::test]
    async fn test_deletion_vector_puffin_is_counted_as_a_position_delete_not_a_fourth_bucket() {
        let catalog = new_memory_catalog().await;
        let table = crate::transaction::tests::make_v3_minimal_table_in_catalog(&catalog).await;
        let data_path = "test/wtB2/f2-dv-data.parquet";
        let table1 = append(&catalog, &table, vec![synthetic_data_file(data_path)]).await;

        let old_puffin = "test/wtB2/f2-old.puffin";
        let dv = synthetic_dv_file(old_puffin, data_path, 4);
        let tx = Transaction::new(&table1);
        let tx = tx
            .row_delta()
            .add_deletes(vec![dv.clone()])
            .apply(tx)
            .expect("apply row delta dv");
        let table2 = tx.commit(&catalog).await.expect("commit row delta dv");

        // Replace the DV: the old puffin becomes referenced by no retained live entry and dies.
        let successor = synthetic_dv_file("test/wtB2/f2-new.puffin", data_path, 4);
        let tx = Transaction::new(&table2);
        let tx = tx
            .row_delta()
            .remove_deletes(dv)
            .add_deletes(vec![successor])
            .apply(tx)
            .expect("apply dv replacement");
        let table3 = tx.commit(&catalog).await.expect("commit dv replacement");

        let cleanup = ExpireSnapshotsCleanup::new(table3.file_io().clone());
        let (_, report) = expire_and_clean(&catalog, &table3, &cleanup).await;

        assert!(report.failures.is_empty(), "{:?}", report.failures);
        assert_eq!(
            report.deleted_content_files,
            vec![old_puffin.to_string()],
            "fixture must delete exactly the replaced puffin"
        );
        assert_eq!(
            report.deleted_position_delete_files(),
            vec![old_puffin],
            "a deletion vector's PUFFIN must be counted as a POSITION delete — Java tags by \
             content() alone, never by format"
        );
        assert!(
            report.deleted_data_files().is_empty(),
            "a DV is not a data file: {:?}",
            report.deleted_data_files()
        );
        assert!(
            report.deleted_equality_delete_files().is_empty(),
            "a DV is not an equality delete: {:?}",
            report.deleted_equality_delete_files()
        );
        assert_union_is_concatenation_of_parts(&report);
    }

    /// The fail-closed posture covers the typed views too. Every view must empty with the union,
    /// which holds because each view filters it. A stored classification could report deletions
    /// the union denies.
    #[tokio::test]
    async fn test_unreadable_retained_manifest_spares_every_typed_content_view() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog).await;
        let data_path = "test/wtB2/f2-failclosed-data.parquet";
        let pos_path = "test/wtB2/f2-failclosed-pos.parquet";
        let data_file = synthetic_data_file(data_path);
        let table1 = append(&catalog, &table, vec![data_file.clone()]).await;

        let pos_delete = synthetic_position_delete_file(pos_path);
        let tx = Transaction::new(&table1);
        let tx = tx
            .row_delta()
            .add_deletes(vec![pos_delete.clone()])
            .apply(tx)
            .expect("apply row delta deletes");
        let table2 = tx.commit(&catalog).await.expect("commit row delta deletes");

        let tx = Transaction::new(&table2);
        let tx = tx
            .row_delta()
            .remove_data_files(vec![data_file])
            .remove_deletes(pos_delete)
            .apply(tx)
            .expect("apply removals");
        let table3 = tx.commit(&catalog).await.expect("commit removals");
        let s3 = table3.metadata().current_snapshot_id().expect("s3");
        let retained_manifests: Vec<String> = manifests_of(&table3, s3)
            .await
            .into_iter()
            .map(|(path, _)| path)
            .collect();
        assert!(
            !retained_manifests.is_empty(),
            "fixture must retain at least one manifest to corrupt"
        );

        let (before, after) = expire_metadata_only(&catalog, &table3).await;
        // Corrupt every retained manifest, so the liveness walk fails whichever it reads first.
        for path in &retained_manifests {
            table3
                .file_io()
                .new_output(path)
                .expect("corrupt output")
                .write(Bytes::from_static(b"wtB2 corrupted avro"))
                .await
                .expect("corrupt retained manifest");
        }

        let cleanup = ExpireSnapshotsCleanup::new(table3.file_io().clone());
        let report = cleanup
            .clean_expired_files(before.metadata(), after.metadata())
            .await
            .expect("clean expired files");

        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.kind == CleanupFailureKind::ReadRetainedManifest),
            "fixture must actually trip the retained-manifest read failure: {:?}",
            report.failures
        );
        assert!(
            report.deleted_content_files.is_empty(),
            "union: {:?}",
            report.deleted_content_files
        );
        assert!(
            report.deleted_data_files().is_empty(),
            "the DATA view must be empty when liveness cannot be proven: {:?}",
            report.deleted_data_files()
        );
        assert!(
            report.deleted_position_delete_files().is_empty(),
            "the POSITION-delete view must be empty when liveness cannot be proven: {:?}",
            report.deleted_position_delete_files()
        );
        assert!(
            report.deleted_equality_delete_files().is_empty(),
            "the EQUALITY-delete view must be empty when liveness cannot be proven: {:?}",
            report.deleted_equality_delete_files()
        );
        assert!(
            report.deleted_content_file_types.is_empty(),
            "the classification lookup must be cleared with the union: {:?}",
            report.deleted_content_file_types
        );
    }

    /// Shared fixture root: a fresh V2 table in the catalog.
    async fn make_table(catalog: &impl Catalog) -> Table {
        crate::transaction::tests::make_v2_minimal_table_in_catalog(catalog).await
    }
}
