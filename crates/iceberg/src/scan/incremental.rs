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

//! Incremental scans.
//!
//! Two incremental / change-data read primitives over the range
//! `(from_snapshot_id exclusive, to_snapshot_id inclusive]`, both SEPARATE planners from
//! the single-snapshot [`TableScan`](super::TableScan) (they do not touch its
//! [`plan_files`](super::TableScan::plan_files)) that REUSE the existing [`PlanContext`]
//! / [`ManifestEntryContext`] machinery (partition-filter pruning, the residual
//! evaluator, and `into_file_scan_task`):
//!
//! - [`IncrementalAppendScan`] returns the data files APPENDED in the range, considering
//!   ONLY `APPEND`-operation snapshots. Overwrites and deletes in the range are excluded
//!   and no delete files are applied. Mirrors Java `BaseIncrementalAppendScan`
//!   (`core/src/main/java/org/apache/iceberg/BaseIncrementalAppendScan.java`):
//!   `doPlanFiles` → `appendsBetween` (the APPEND snapshots in the range, via
//!   `SnapshotUtil.ancestorsBetween` filtered to `operation == APPEND`) →
//!   `appendFilesFromSnapshots` (plan tasks for the data files those snapshots ADDED:
//!   only the `Added` entries of the manifests each snapshot itself added, where
//!   `manifest.snapshotId() == snapshot.snapshotId()`).
//!
//! - [`IncrementalChangelogScan`] returns row-level CHANGE tasks ([`ChangelogScanTask`]):
//!   an INSERT task per data file ADDED and a DELETE task per data file REMOVED by the
//!   snapshots in the range, each tagged with a change ordinal (oldest snapshot → 0) and
//!   its commit snapshot id. `Operation::Replace` snapshots (e.g. compaction) are
//!   excluded — they rewrite files without changing rows. Mirrors Java
//!   `BaseIncrementalChangelogScan`
//!   (`core/src/main/java/org/apache/iceberg/BaseIncrementalChangelogScan.java`).
//!   Like Java's current data-file changelog, a range whose snapshots carry row-level
//!   DELETE manifests is rejected by default (`FeatureUnsupported`; Java 1.10.0 throws
//!   `UnsupportedOperationException` — bytecode-verified). The opt-in **ENGINE-FIRST**
//!   row-level mode
//!   ([`with_row_level_deletes`](IncrementalChangelogScanBuilder::with_row_level_deletes))
//!   goes beyond Java 1.10.0 core: it accepts such ranges and emits the Java-api task
//!   taxonomy (`AddedRowsScanTask` with same-snapshot deletes folded in,
//!   `DeletedDataFileScanTask` with pre-existing deletes, `DeletedRowsScanTask` for
//!   existing files hit by the snapshot's ADDED delete files) — the task shapes Java's
//!   api defines but whose emission 1.10.0 core does not yet implement. Net-change
//!   UPDATE_BEFORE/UPDATE_AFTER pairing stays engine-side (Spark `ChangelogIterator`)
//!   and is NOT performed here.

use std::sync::Arc;

use futures::channel::mpsc::{Sender, channel};
use futures::{SinkExt, StreamExt, TryStreamExt};

use super::context::{ManifestEntryContext, PlanContext, parse_name_mapping};
use crate::delete_file_index::DeleteFileIndex;
use crate::events::{self, IncrementalScanEvent};
use crate::expr::{Bind, Predicate};
use crate::io::FileIO;
use crate::metadata_columns::{get_metadata_field_id, is_metadata_column_name};
use crate::runtime::spawn;
use crate::scan::{
    BoundPredicates, ChangelogScanTask, ChangelogScanTaskStream, ChangelogTaskKind,
    DeleteFileContext, ExpressionEvaluatorCache, FileScanTask, FileScanTaskStream,
    ManifestEvaluatorCache, PartitionFilterCache,
};
use crate::spec::{
    DataContentType, ManifestContentType, ManifestFile, ManifestStatus, Operation, SchemaRef,
    SnapshotRef,
};
use crate::table::Table;
use crate::utils::available_parallelism;
use crate::{Error, ErrorKind, Result};

/// Builder to create an [`IncrementalAppendScan`].
///
/// Mirrors the Java `IncrementalAppendScan` API (`api/IncrementalAppendScan.java` +
/// `BaseIncrementalScan`): the `from` snapshot can be set as exclusive
/// ([`Self::from_snapshot_id_exclusive`], Java `fromSnapshotExclusive`) or inclusive
/// ([`Self::from_snapshot_id_inclusive`], Java `fromSnapshotInclusive`); the `to`
/// snapshot ([`Self::to_snapshot_id`], Java `toSnapshot`) defaults to the table's
/// current snapshot when unset. The builder ergonomics (`with_filter`, `select`,
/// concurrency limits) mirror [`TableScanBuilder`](super::TableScanBuilder).
pub struct IncrementalAppendScanBuilder<'a> {
    table: &'a Table,
    column_names: Option<Vec<String>>,
    /// The exclusive lower bound of the range — the parent below which appends are
    /// counted. `None` means "from the beginning of history" (every ancestor of `to`).
    from_snapshot_id_exclusive: Option<i64>,
    /// When the caller set the `from` bound as INCLUSIVE, this holds that snapshot id.
    /// Resolution at `build()` time converts it to the exclusive bound (the snapshot's
    /// parent), mirroring Java `fromSnapshotIdExclusive` for the inclusive case.
    from_snapshot_id_inclusive: Option<i64>,
    to_snapshot_id: Option<i64>,
    batch_size: Option<usize>,
    case_sensitive: bool,
    filter: Option<Predicate>,
    concurrency_limit_manifest_entries: usize,
    concurrency_limit_manifest_files: usize,
}

impl<'a> IncrementalAppendScanBuilder<'a> {
    pub(crate) fn new(table: &'a Table) -> Self {
        let num_cpus = available_parallelism().get();

        Self {
            table,
            column_names: None,
            from_snapshot_id_exclusive: None,
            from_snapshot_id_inclusive: None,
            to_snapshot_id: None,
            batch_size: None,
            case_sensitive: true,
            filter: None,
            concurrency_limit_manifest_entries: num_cpus,
            concurrency_limit_manifest_files: num_cpus,
        }
    }

    /// Sets the EXCLUSIVE `from` snapshot id (Java `fromSnapshotExclusive`): appends in
    /// `(from, to]` are returned — `from`'s own files are NOT included. Supersedes any
    /// previously-set inclusive bound.
    pub fn from_snapshot_id_exclusive(mut self, from_snapshot_id: i64) -> Self {
        self.from_snapshot_id_exclusive = Some(from_snapshot_id);
        self.from_snapshot_id_inclusive = None;
        self
    }

    /// Sets the INCLUSIVE `from` snapshot id (Java `fromSnapshotInclusive`): appends in
    /// `[from, to]` are returned — `from`'s own files ARE included (provided `from` is an
    /// APPEND snapshot). Resolved to the exclusive bound (`from`'s parent) at `build()`.
    /// Supersedes any previously-set exclusive bound.
    pub fn from_snapshot_id_inclusive(mut self, from_snapshot_id: i64) -> Self {
        self.from_snapshot_id_inclusive = Some(from_snapshot_id);
        self.from_snapshot_id_exclusive = None;
        self
    }

    /// Sets the INCLUSIVE `to` snapshot id (Java `toSnapshot`). When unset, defaults to
    /// the table's current snapshot.
    pub fn to_snapshot_id(mut self, to_snapshot_id: i64) -> Self {
        self.to_snapshot_id = Some(to_snapshot_id);
        self
    }

    /// Sets the desired size of batches in the response to something other than the default.
    pub fn with_batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the scan's case sensitivity.
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Specifies a predicate to use as a filter.
    pub fn with_filter(mut self, predicate: Predicate) -> Self {
        // calls rewrite_not to remove Not nodes, which must be absent
        // when applying the manifest evaluator
        self.filter = Some(predicate.rewrite_not());
        self
    }

    /// Select all columns.
    pub fn select_all(mut self) -> Self {
        self.column_names = None;
        self
    }

    /// Select empty columns.
    pub fn select_empty(mut self) -> Self {
        self.column_names = Some(vec![]);
        self
    }

    /// Select some columns of the table.
    pub fn select(mut self, column_names: impl IntoIterator<Item = impl ToString>) -> Self {
        self.column_names = Some(
            column_names
                .into_iter()
                .map(|item| item.to_string())
                .collect(),
        );
        self
    }

    /// Sets the concurrency limit for both manifest files and manifest entries.
    pub fn with_concurrency_limit(mut self, limit: usize) -> Self {
        self.concurrency_limit_manifest_files = limit;
        self.concurrency_limit_manifest_entries = limit;
        self
    }

    /// Build the incremental append scan.
    ///
    /// Validates that the `to` snapshot exists (defaulting to the current snapshot),
    /// that an explicit `from` snapshot exists, and that `from` is an ancestor of `to`
    /// (Java requires `from` be an ancestor of `to`). Resolves the schema, projected
    /// field ids, and bound filter exactly as the normal scan does.
    pub fn build(self) -> Result<IncrementalAppendScan> {
        let metadata = self.table.metadata();

        // Captured once for the `IncrementalScanEvent` fired at plan time (Java's shared
        // `BaseIncrementalScan` reads `table().name()`). Both the empty and the normal scan
        // carry it.
        let table_name = self.table.identifier().to_string();

        // Resolve the inclusive `to` snapshot (Java `toSnapshotIdInclusive()`): explicit
        // id if set, else the current snapshot. With no current snapshot and no explicit
        // `to`, the scan is empty.
        let to_snapshot = match self.to_snapshot_id {
            Some(to_snapshot_id) => metadata.snapshot_by_id(to_snapshot_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Cannot find the end snapshot: {to_snapshot_id}"),
                )
            })?,
            None => {
                let Some(current_snapshot) = metadata.current_snapshot() else {
                    return Ok(IncrementalAppendScan {
                        from_snapshot_id_exclusive: None,
                        to_snapshot_id: None,
                        plan_context: None,
                        column_names: self.column_names,
                        batch_size: self.batch_size,
                        file_io: self.table.file_io().clone(),
                        concurrency_limit_manifest_entries: self.concurrency_limit_manifest_entries,
                        concurrency_limit_manifest_files: self.concurrency_limit_manifest_files,
                        table_name,
                    });
                };
                current_snapshot
            }
        }
        .clone();

        let to_snapshot_id = to_snapshot.snapshot_id();

        // Resolve the EXCLUSIVE `from` bound (Java `fromSnapshotIdExclusive(toInclusive)`).
        //
        // - inclusive `from`: validate `from` is an ancestor of `to`, then the exclusive
        //   bound is `from`'s parent (which may be `None` = the whole history up to and
        //   including `from`).
        // - exclusive `from`: validate `from` is a *parent ancestor* of `to` (some
        //   ancestor of `to` has `parent_id == from`), then the exclusive bound is `from`.
        // - neither set: `None` = scan the whole current lineage.
        let from_snapshot_id_exclusive = if let Some(from_inclusive) =
            self.from_snapshot_id_inclusive
        {
            let from_snapshot = metadata.snapshot_by_id(from_inclusive).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Cannot find the starting snapshot: {from_inclusive}"),
                )
            })?;
            if !is_ancestor_of(self.table, to_snapshot_id, from_inclusive) {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Starting snapshot (inclusive) {from_inclusive} is not an ancestor of end snapshot {to_snapshot_id}"
                    ),
                ));
            }
            from_snapshot.parent_snapshot_id()
        } else if let Some(from_exclusive) = self.from_snapshot_id_exclusive {
            if !is_parent_ancestor_of(self.table, to_snapshot_id, from_exclusive) {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Starting snapshot (exclusive) {from_exclusive} is not a parent ancestor of end snapshot {to_snapshot_id}"
                    ),
                ));
            }
            Some(from_exclusive)
        } else {
            None
        };

        // Resolve schema, projected field ids, and the bound filter from the `to`
        // snapshot — mirroring `TableScanBuilder::build`.
        let schema = to_snapshot.schema(metadata)?;

        if let Some(column_names) = self.column_names.as_ref() {
            for column_name in column_names {
                if is_metadata_column_name(column_name) {
                    continue;
                }
                if schema.field_by_name(column_name).is_none() {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Column {column_name} not found in table. Schema: {schema}"),
                    ));
                }
            }
        }

        let column_names = self.column_names.clone().unwrap_or_else(|| {
            schema
                .as_struct()
                .fields()
                .iter()
                .map(|f| f.name.clone())
                .collect()
        });

        let mut field_ids = vec![];
        for column_name in column_names.iter() {
            if is_metadata_column_name(column_name) {
                field_ids.push(get_metadata_field_id(column_name)?);
                continue;
            }

            let field_id = schema.field_id_by_name(column_name).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Column {column_name} not found in table. Schema: {schema}"),
                )
            })?;

            schema
                .as_struct()
                .field_by_id(field_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!(
                            "Column {column_name} is not a direct child of schema but a nested field, which is not supported now. Schema: {schema}"
                        ),
                    )
                })?;

            field_ids.push(field_id);
        }

        let snapshot_bound_predicate = if let Some(ref predicate) = self.filter {
            Some(predicate.bind(schema.clone(), true)?)
        } else {
            None
        };

        let plan_context = PlanContext {
            // The PlanContext's `snapshot` is the `to` snapshot, used only for its
            // schema and as the metadata anchor. The incremental planner does NOT call
            // `PlanContext::get_manifest_list` (which would read only this one snapshot);
            // it drives the manifest selection itself across the range's snapshots.
            snapshot: to_snapshot,
            table_metadata: self.table.metadata_ref(),
            snapshot_schema: schema,
            case_sensitive: self.case_sensitive,
            predicate: self.filter.map(Arc::new),
            snapshot_bound_predicate: snapshot_bound_predicate.map(Arc::new),
            object_cache: self.table.object_cache(),
            field_ids: Arc::new(field_ids),
            // Parse the table's default name mapping once per plan (Java
            // `NameMappingParser.fromJson`) so an id-less data file added within the incremental
            // range resolves field ids by column name, exactly as the snapshot scan does.
            name_mapping: parse_name_mapping(self.table.metadata())?,
            partition_filter_cache: Arc::new(PartitionFilterCache::new()),
            manifest_evaluator_cache: Arc::new(ManifestEvaluatorCache::new()),
            expression_evaluator_cache: Arc::new(ExpressionEvaluatorCache::new()),
            // The incremental-append scan does not emit a `ScanReport` (Java's metrics
            // reporting lives on the snapshot scan, not `IncrementalDataTableScan`); leave
            // it uninstrumented.
            metrics_collector: None,
        };

        Ok(IncrementalAppendScan {
            from_snapshot_id_exclusive,
            to_snapshot_id: Some(to_snapshot_id),
            plan_context: Some(plan_context),
            table_name,
            column_names: self.column_names,
            batch_size: self.batch_size,
            file_io: self.table.file_io().clone(),
            concurrency_limit_manifest_entries: self.concurrency_limit_manifest_entries,
            concurrency_limit_manifest_files: self.concurrency_limit_manifest_files,
        })
    }
}

/// An incremental append scan over the range `(from_snapshot_id exclusive, to_snapshot_id inclusive]`.
///
/// Built via [`Table::incremental_append_scan`](crate::table::Table::incremental_append_scan).
/// Its [`plan_files`](Self::plan_files) streams the [`FileScanTask`]s for the data files
/// the APPEND snapshots in the range added — see the module docs.
#[derive(Debug)]
pub struct IncrementalAppendScan {
    /// The exclusive lower bound — `None` means the whole current lineage of `to`.
    from_snapshot_id_exclusive: Option<i64>,
    /// The inclusive upper bound — `None` only when the table has no snapshots
    /// (the scan is then empty).
    to_snapshot_id: Option<i64>,
    /// `None` when the table has no snapshots and no explicit `to` was set — the scan
    /// produces no rows.
    plan_context: Option<PlanContext>,
    column_names: Option<Vec<String>>,
    batch_size: Option<usize>,
    file_io: FileIO,
    concurrency_limit_manifest_entries: usize,
    concurrency_limit_manifest_files: usize,
    /// The fully-qualified table name, captured at [`build`](IncrementalAppendScanBuilder::build)
    /// time for the [`IncrementalScanEvent`] fired from [`plan_files`](Self::plan_files) (Java's
    /// shared `BaseIncrementalScan` reads `table().name()`).
    table_name: String,
}

impl IncrementalAppendScan {
    /// Returns a stream of [`FileScanTask`]s for the appended data files in the range.
    ///
    /// Mirrors Java `BaseIncrementalAppendScan.doPlanFiles` →
    /// `appendFilesFromSnapshots`: compute the APPEND snapshots in
    /// `(from_snapshot_id exclusive, to_snapshot_id inclusive]`; for each, load its
    /// manifest list, keep the DATA manifests it ADDED, read each manifest's `Added`
    /// entries, and stream a `FileScanTask` per entry — applying the same partition
    /// filtering and residual evaluation as the normal scan, with an EMPTY delete index
    /// (an append scan applies no deletes).
    pub async fn plan_files(&self) -> Result<FileScanTaskStream> {
        let Some(plan_context) = self.plan_context.as_ref() else {
            return Ok(Box::pin(futures::stream::empty()));
        };
        let Some(to_snapshot_id) = self.to_snapshot_id else {
            return Ok(Box::pin(futures::stream::empty()));
        };

        // Fire the `IncrementalScanEvent` (Java's shared `BaseIncrementalScan.planFiles` fires it
        // before `doPlanFiles`). Placed AFTER the snapshotless / no-`to` guards, so a scan with
        // no resolvable range fires nothing; fired BEFORE the empty-range check below, matching
        // Java (which fires for a valid `to` regardless of whether the range turns out empty).
        self.notify_incremental_scan_event(plan_context, to_snapshot_id);

        // Compute the APPEND snapshots in the range (Java `appendsBetween`). If there are
        // none, the scan is empty.
        let append_snapshots = self.appends_between(
            plan_context,
            self.from_snapshot_id_exclusive,
            to_snapshot_id,
        )?;
        if append_snapshots.is_empty() {
            return Ok(Box::pin(futures::stream::empty()));
        }

        let concurrency_limit_manifest_files = self.concurrency_limit_manifest_files;
        let concurrency_limit_manifest_entries = self.concurrency_limit_manifest_entries;

        let (manifest_entry_data_ctx_tx, manifest_entry_data_ctx_rx) =
            channel(concurrency_limit_manifest_files);
        // A second, never-fed channel for the delete branch — an append scan applies no
        // deletes, so it stays empty. `build_manifest_file_contexts_from_files` needs a
        // delete sender to satisfy its signature; we only pass DATA manifests, so nothing
        // is ever routed to it.
        let (delete_ctx_tx, _delete_ctx_rx) = channel::<ManifestEntryContext>(1);
        let (file_scan_task_tx, file_scan_task_rx) = channel(concurrency_limit_manifest_entries);

        // An EMPTY delete index: an append scan applies no delete files. We build it but
        // never send anything, then drop the sender so the index resolves to "no deletes".
        let (delete_file_idx, delete_file_tx) = DeleteFileIndex::new();
        drop(delete_file_tx);

        // Collect, across every append snapshot, ONLY the DATA manifests that snapshot
        // itself added (Java: `snapshot.dataManifests(io).filter(m -> snapshotIds.contains(
        // m.snapshotId()))`). A manifest carried forward from an older snapshot belongs to
        // that older snapshot and its files were not appended in this range.
        let mut selected_manifests = Vec::new();
        for snapshot in &append_snapshots {
            let manifest_list = snapshot
                .load_manifest_list(&self.file_io, &plan_context.table_metadata)
                .await?;
            for manifest_file in manifest_list.consume_entries() {
                if manifest_file.content == ManifestContentType::Data
                    && manifest_file.added_snapshot_id == snapshot.snapshot_id()
                {
                    selected_manifests.push(manifest_file);
                }
            }
        }

        let manifest_file_contexts = plan_context.build_manifest_file_contexts_from_files(
            selected_manifests,
            manifest_entry_data_ctx_tx,
            delete_file_idx,
            delete_ctx_tx,
        )?;

        let mut channel_for_manifest_error = file_scan_task_tx.clone();

        // Concurrently load all selected manifests and stream their entries.
        spawn(async move {
            let result = futures::stream::iter(manifest_file_contexts)
                .try_for_each_concurrent(concurrency_limit_manifest_files, |ctx| async move {
                    ctx.fetch_manifest_and_stream_manifest_entries().await
                })
                .await;

            if let Err(error) = result {
                let _ = channel_for_manifest_error.send(Err(error)).await;
            }
        });

        let mut channel_for_data_manifest_entry_error = file_scan_task_tx.clone();

        // Process the data entries in parallel, keeping only `Added` entries.
        spawn(async move {
            let result = manifest_entry_data_ctx_rx
                .map(|me_ctx| Ok((me_ctx, file_scan_task_tx.clone())))
                .try_for_each_concurrent(
                    concurrency_limit_manifest_entries,
                    |(manifest_entry_context, tx)| async move {
                        spawn(async move {
                            Self::process_append_manifest_entry(manifest_entry_context, tx).await
                        })
                        .await
                    },
                )
                .await;

            if let Err(error) = result {
                let _ = channel_for_data_manifest_entry_error.send(Err(error)).await;
            }
        });

        Ok(file_scan_task_rx.boxed())
    }

    /// Fires the [`IncrementalScanEvent`] for this scan over `(from, to]`.
    ///
    /// Mirrors the shared `BaseIncrementalScan.planFiles` event, which resolves the `from`
    /// bound into the event two ways:
    /// - an explicit exclusive `from` → `(from, inclusive = false)`;
    /// - an absent `from` → `(oldestAncestorOf(to), inclusive = true)` — the whole lineage of
    ///   `to`, whose lower edge is the history root.
    ///
    /// The Rust builder collapsed Java's two `from` selectors into a single
    /// `Option<i64>` exclusive bound, so it is RE-RESOLVED here. The filter is the UNBOUND row
    /// filter (`plan_context.predicate`, defaulting to `AlwaysTrue`); the projection is the
    /// scan's schema. Like the snapshot-scan event, the call is unguarded.
    fn notify_incremental_scan_event(&self, plan_context: &PlanContext, to_snapshot_id: i64) {
        let (from_snapshot_id, from_inclusive) = match self.from_snapshot_id_exclusive {
            Some(from_exclusive) => (from_exclusive, false),
            None => (
                Self::oldest_ancestor_id_of(plan_context, to_snapshot_id),
                true,
            ),
        };

        let filter = plan_context
            .predicate
            .as_ref()
            .map(|p| p.as_ref().clone())
            .unwrap_or(Predicate::AlwaysTrue);

        events::notify_all(&IncrementalScanEvent::new(
            self.table_name.clone(),
            from_snapshot_id,
            to_snapshot_id,
            filter,
            plan_context.snapshot_schema.clone(),
            from_inclusive,
        ));
    }

    /// Returns the id of the OLDEST ancestor of `to_snapshot_id` (the history root reachable by
    /// walking `parent_snapshot_id`), Java `SnapshotUtil.oldestAncestorOf(to).snapshotId()`.
    /// Falls back to `to_snapshot_id` itself if `to` is not found (it always is at this point).
    fn oldest_ancestor_id_of(plan_context: &PlanContext, to_snapshot_id: i64) -> i64 {
        let metadata = &plan_context.table_metadata;
        let mut oldest = to_snapshot_id;
        let mut current = metadata.snapshot_by_id(to_snapshot_id).cloned();
        while let Some(snapshot) = current {
            oldest = snapshot.snapshot_id();
            current = match snapshot.parent_snapshot_id() {
                Some(parent_id) => metadata.snapshot_by_id(parent_id).cloned(),
                None => None,
            };
        }
        oldest
    }

    /// Returns the APPEND snapshots in `(from_snapshot_id_exclusive, to_snapshot_id]`,
    /// ordered newest-first (Java `appendsBetween`).
    ///
    /// Walks the parent chain from `to_snapshot_id` back via `parent_snapshot_id`,
    /// stopping BEFORE `from_snapshot_id_exclusive` (the start is excluded), and keeps
    /// only snapshots whose `operation == Append`. `from_snapshot_id_exclusive == None`
    /// walks to the history root (the whole lineage). Mirrors
    /// `SnapshotUtil.ancestorsBetween(to, from, lookup)` filtered to `APPEND`.
    fn appends_between(
        &self,
        plan_context: &PlanContext,
        from_snapshot_id_exclusive: Option<i64>,
        to_snapshot_id: i64,
    ) -> Result<Vec<SnapshotRef>> {
        let metadata = &plan_context.table_metadata;

        // Java `ancestorsBetween`: an equal from/to yields an empty range.
        if from_snapshot_id_exclusive == Some(to_snapshot_id) {
            return Ok(vec![]);
        }

        let mut snapshots = Vec::new();
        let mut current = metadata.snapshot_by_id(to_snapshot_id).cloned();

        while let Some(snapshot) = current {
            // Stop BEFORE the exclusive start (Java's lookup returns null for the start id).
            if Some(snapshot.snapshot_id()) == from_snapshot_id_exclusive {
                break;
            }

            if snapshot.summary().operation == Operation::Append {
                snapshots.push(snapshot.clone());
            }

            current = match snapshot.parent_snapshot_id() {
                Some(parent_id) => metadata.snapshot_by_id(parent_id).cloned(),
                None => None,
            };
        }

        Ok(snapshots)
    }

    /// Processes a single data-manifest entry for the incremental append scan: keeps only
    /// `Added`-status entries (Java `filterManifestEntries(status == ADDED)`), applies the
    /// scan's partition filter, and emits a [`FileScanTask`].
    async fn process_append_manifest_entry(
        manifest_entry_context: ManifestEntryContext,
        mut file_scan_task_tx: Sender<Result<FileScanTask>>,
    ) -> Result<()> {
        // Only ADDED entries: an `Existing` entry was added by an earlier snapshot and
        // copied forward (its files were NOT appended in this range), and a `Deleted`
        // tombstone is a removal. Java keeps `manifestEntry.status() == ADDED`.
        if manifest_entry_context.manifest_entry.status() != ManifestStatus::Added {
            return Ok(());
        }

        // An append scan never reads a delete-file manifest (we only select DATA
        // manifests), but guard the invariant the same way the normal scan does.
        if manifest_entry_context.manifest_entry.content_type() != DataContentType::Data {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Encountered an entry for a delete file in an incremental append scan",
            ));
        }

        if let Some(ref bound_predicates) = manifest_entry_context.bound_predicates {
            let BoundPredicates {
                snapshot_bound_predicate,
                partition_bound_predicate,
            } = bound_predicates.as_ref();

            let expression_evaluator_cache =
                manifest_entry_context.expression_evaluator_cache.as_ref();

            let expression_evaluator = expression_evaluator_cache.get(
                manifest_entry_context.partition_spec_id,
                partition_bound_predicate,
            )?;

            // skip any data file whose partition data indicates it can't match the filter
            if !expression_evaluator.eval(manifest_entry_context.manifest_entry.data_file())? {
                return Ok(());
            }

            // skip any data file whose metrics don't match the filter
            if !crate::expr::visitors::inclusive_metrics_evaluator::InclusiveMetricsEvaluator::eval(
                snapshot_bound_predicate,
                manifest_entry_context.manifest_entry.data_file(),
                false,
            )? {
                return Ok(());
            }
        }

        file_scan_task_tx
            .send(Ok(manifest_entry_context.into_file_scan_task().await?))
            .await?;

        Ok(())
    }

    /// Returns the projected column names of this scan, if a projection was set.
    pub fn column_names(&self) -> Option<&[String]> {
        self.column_names.as_deref()
    }

    /// Returns the inclusive `to` snapshot id of this scan, if the table has snapshots.
    pub fn to_snapshot_id(&self) -> Option<i64> {
        self.to_snapshot_id
    }

    /// Returns the exclusive `from` snapshot id of this scan (`None` = the whole lineage).
    pub fn from_snapshot_id_exclusive(&self) -> Option<i64> {
        self.from_snapshot_id_exclusive
    }

    /// Returns the scan's batch size, if set.
    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    /// The schema the scan projects (the `to` snapshot's schema), if the table has snapshots.
    pub fn snapshot_schema(&self) -> Option<&SchemaRef> {
        self.plan_context.as_ref().map(|ctx| &ctx.snapshot_schema)
    }

    /// The resolved [`PlanContext`] (schema + metadata anchor + caches), if the table has
    /// snapshots. Shared with [`IncrementalChangelogScan`], which reuses the append scan's
    /// range resolution but drives its own snapshot selection.
    pub(crate) fn plan_context(&self) -> Option<&PlanContext> {
        self.plan_context.as_ref()
    }
}

/// Builder to create an [`IncrementalChangelogScan`].
///
/// Mirrors the Java `IncrementalChangelogScan` API (`api/IncrementalChangelogScan.java` +
/// `BaseIncrementalScan`): the range bounds (`from` exclusive/inclusive, `to` defaulting
/// to the current snapshot), `with_filter`, and `select*` mirror
/// [`IncrementalAppendScanBuilder`]. The builder shares that scan's range-resolution
/// logic; the difference is entirely in [`IncrementalChangelogScan::plan_files`].
pub struct IncrementalChangelogScanBuilder<'a> {
    table: &'a Table,
    column_names: Option<Vec<String>>,
    from_snapshot_id_exclusive: Option<i64>,
    from_snapshot_id_inclusive: Option<i64>,
    to_snapshot_id: Option<i64>,
    batch_size: Option<usize>,
    case_sensitive: bool,
    filter: Option<Predicate>,
    /// The per-manifest concurrency limit, forwarded to the underlying append scan
    /// builder's `with_concurrency_limit` (which sets both the manifest-file and
    /// manifest-entry limits together — the only public way to set them).
    concurrency_limit: usize,
    /// Whether the ENGINE-FIRST row-level mode is enabled — see
    /// [`Self::with_row_level_deletes`]. Defaults to `false` (exact Java 1.10.0 core
    /// behavior).
    include_row_level_deletes: bool,
}

impl<'a> IncrementalChangelogScanBuilder<'a> {
    pub(crate) fn new(table: &'a Table) -> Self {
        let num_cpus = available_parallelism().get();

        Self {
            table,
            column_names: None,
            from_snapshot_id_exclusive: None,
            from_snapshot_id_inclusive: None,
            to_snapshot_id: None,
            batch_size: None,
            case_sensitive: true,
            filter: None,
            concurrency_limit: num_cpus,
            include_row_level_deletes: false,
        }
    }

    /// Sets the EXCLUSIVE `from` snapshot id (Java `fromSnapshotExclusive`): changes in
    /// `(from, to]` are returned — `from`'s own changes are NOT included. Supersedes any
    /// previously-set inclusive bound.
    pub fn from_snapshot_id_exclusive(mut self, from_snapshot_id: i64) -> Self {
        self.from_snapshot_id_exclusive = Some(from_snapshot_id);
        self.from_snapshot_id_inclusive = None;
        self
    }

    /// Sets the INCLUSIVE `from` snapshot id (Java `fromSnapshotInclusive`): changes in
    /// `[from, to]` are returned — `from`'s own changes ARE included. Resolved to the
    /// exclusive bound (`from`'s parent) at `build()`. Supersedes any previously-set
    /// exclusive bound.
    pub fn from_snapshot_id_inclusive(mut self, from_snapshot_id: i64) -> Self {
        self.from_snapshot_id_inclusive = Some(from_snapshot_id);
        self.from_snapshot_id_exclusive = None;
        self
    }

    /// Sets the INCLUSIVE `to` snapshot id (Java `toSnapshot`). When unset, defaults to
    /// the table's current snapshot.
    pub fn to_snapshot_id(mut self, to_snapshot_id: i64) -> Self {
        self.to_snapshot_id = Some(to_snapshot_id);
        self
    }

    /// Sets the desired size of batches in the response to something other than the default.
    pub fn with_batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the scan's case sensitivity.
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Specifies a predicate to use as a filter.
    pub fn with_filter(mut self, predicate: Predicate) -> Self {
        // calls rewrite_not to remove Not nodes, which must be absent
        // when applying the manifest evaluator
        self.filter = Some(predicate.rewrite_not());
        self
    }

    /// Select all columns.
    pub fn select_all(mut self) -> Self {
        self.column_names = None;
        self
    }

    /// Select empty columns.
    pub fn select_empty(mut self) -> Self {
        self.column_names = Some(vec![]);
        self
    }

    /// Select some columns of the table.
    pub fn select(mut self, column_names: impl IntoIterator<Item = impl ToString>) -> Self {
        self.column_names = Some(
            column_names
                .into_iter()
                .map(|item| item.to_string())
                .collect(),
        );
        self
    }

    /// Sets the concurrency limit for both manifest files and manifest entries.
    pub fn with_concurrency_limit(mut self, limit: usize) -> Self {
        self.concurrency_limit = limit;
        self
    }

    /// **ENGINE-FIRST (beyond Java 1.10.0 core):** enables row-level changelog planning.
    ///
    /// Java 1.10.0's `BaseIncrementalChangelogScan.orderedChangelogSnapshots` throws
    /// `UnsupportedOperationException` ("Delete files are currently not supported in
    /// changelog scans") for ANY non-`replace` range snapshot whose manifest list carries
    /// a delete manifest — the default (`false`) mirrors that rejection surface exactly.
    ///
    /// With `true`, such ranges are accepted and the scan emits the row-level task
    /// taxonomy Java's *api* defines (`AddedRowsScanTask` / `DeletedDataFileScanTask` /
    /// `DeletedRowsScanTask` — see [`ChangelogTaskKind`]) but whose emission 1.10.0
    /// *core* does not implement: an existing data file that the commit snapshot's ADDED
    /// delete files apply to yields a [`ChangelogTaskKind::DeletedRows`] task carrying
    /// exactly those added deletes (plus the pre-existing deletes to apply first), and a
    /// data file added in the same snapshot as deletes that match it folds them into its
    /// [`ChangelogTaskKind::AddedRows`] task instead (the `AddedRowsScanTask` javadoc
    /// contract). This is a Rust-side extension serving the downstream CDC engine (the
    /// DML-foundation direction); it is NOT claimed as core parity.
    pub fn with_row_level_deletes(mut self, include_row_level_deletes: bool) -> Self {
        self.include_row_level_deletes = include_row_level_deletes;
        self
    }

    /// Build the incremental changelog scan.
    ///
    /// Resolves the range bounds, schema, projected field ids, and bound filter exactly
    /// as [`IncrementalAppendScanBuilder::build`] does (the two scans share range
    /// resolution). The changelog-specific snapshot selection — exclude `Replace`,
    /// guard delete manifests, assign ordinals — happens lazily in
    /// [`IncrementalChangelogScan::plan_files`].
    pub fn build(self) -> Result<IncrementalChangelogScan> {
        // Reuse the append scan's builder to resolve the range + plan context. The two
        // builders share every field; only `plan_files` differs. This avoids duplicating
        // the (non-trivial) range-resolution + projection logic. The public API sets both
        // concurrency limits together (`with_concurrency_limit`), so passing the files
        // limit covers the entries limit too.
        let mut append_builder = IncrementalAppendScanBuilder::new(self.table)
            .with_case_sensitive(self.case_sensitive)
            .with_batch_size(self.batch_size)
            .with_concurrency_limit(self.concurrency_limit);

        if let Some(from_exclusive) = self.from_snapshot_id_exclusive {
            append_builder = append_builder.from_snapshot_id_exclusive(from_exclusive);
        } else if let Some(from_inclusive) = self.from_snapshot_id_inclusive {
            append_builder = append_builder.from_snapshot_id_inclusive(from_inclusive);
        }
        if let Some(to_snapshot_id) = self.to_snapshot_id {
            append_builder = append_builder.to_snapshot_id(to_snapshot_id);
        }
        if let Some(ref column_names) = self.column_names {
            append_builder = append_builder.select(column_names);
        }
        if let Some(ref filter) = self.filter {
            // `filter` is already `rewrite_not`-normalized; `with_filter` re-normalizes,
            // which is idempotent.
            append_builder = append_builder.with_filter(filter.clone());
        }

        Ok(IncrementalChangelogScan {
            append_scan: append_builder.build()?,
            file_io: self.table.file_io().clone(),
            include_row_level_deletes: self.include_row_level_deletes,
        })
    }
}

/// An incremental changelog scan over `(from_snapshot_id exclusive, to_snapshot_id inclusive]`.
///
/// Built via
/// [`Table::incremental_changelog_scan`](crate::table::Table::incremental_changelog_scan).
/// Its [`plan_files`](Self::plan_files) streams a [`ChangelogScanTask`] per data file
/// added (INSERT) or removed (DELETE) by the snapshots in the range — see the module docs.
#[derive(Debug)]
pub struct IncrementalChangelogScan {
    /// The underlying append scan carries the resolved range bounds + plan context. The
    /// changelog scan reuses its range resolution and `PlanContext` but drives its own
    /// snapshot selection + per-entry task construction in `plan_files`.
    append_scan: IncrementalAppendScan,
    file_io: FileIO,
    /// Whether the ENGINE-FIRST row-level mode is enabled — see
    /// [`IncrementalChangelogScanBuilder::with_row_level_deletes`].
    include_row_level_deletes: bool,
}

/// The per-snapshot delete-file indexes the ENGINE-FIRST row-level changelog mode plans
/// against (see [`IncrementalChangelogScan::build_snapshot_delete_indexes`]).
struct SnapshotDeleteIndexes {
    /// Delete files this snapshot ADDED (Java `DeletedRowsScanTask.addedDeletes()` /
    /// `AddedRowsScanTask.deletes()`).
    added: DeleteFileIndex,
    /// Live delete files that pre-existed this snapshot (Java `existingDeletes()`).
    existing: DeleteFileIndex,
    /// Whether the snapshot added ANY delete file — `false` skips the DeletedRows
    /// candidate pass entirely, keeping delete-free snapshots on the default plan shape.
    has_added_deletes: bool,
}

impl IncrementalChangelogScan {
    /// Returns a stream of [`ChangelogScanTask`]s for the row-level changes in the range.
    ///
    /// Mirrors Java `BaseIncrementalChangelogScan.doPlanFiles`:
    /// 1. compute the changelog snapshots in `(from, to]` oldest-first, EXCLUDING
    ///    `Operation::Replace`; if any range snapshot carries a row-level DELETE manifest,
    ///    return `FeatureUnsupported` (Java throws `UnsupportedOperationException`);
    /// 2. assign each snapshot a change ordinal (oldest → 0, incrementing);
    /// 3. for each changelog snapshot, read the DATA manifests IT added
    ///    (`added_snapshot_id == snapshot_id`) and emit a task per ADDED entry (INSERT) or
    ///    DELETED entry (DELETE) — skipping `Existing` entries — tagged with that
    ///    snapshot's ordinal and `commit_snapshot_id = entry.snapshot_id()`, reusing the
    ///    same partition-filter pruning + residual as the append scan.
    pub async fn plan_files(&self) -> Result<ChangelogScanTaskStream> {
        let Some(plan_context) = self.append_scan.plan_context() else {
            return Ok(Box::pin(futures::stream::empty()));
        };
        let Some(to_snapshot_id) = self.append_scan.to_snapshot_id() else {
            return Ok(Box::pin(futures::stream::empty()));
        };

        // Fire the `IncrementalScanEvent` here too: Java fires it from the SHARED
        // `BaseIncrementalScan.planFiles`, which both the append scan and the changelog scan
        // inherit. Reuse the append scan's emit (same range bounds / plan context / table name),
        // placed after the same guards so an unresolvable range fires nothing.
        self.append_scan
            .notify_incremental_scan_event(plan_context, to_snapshot_id);

        // Step 1: the changelog snapshots in the range, oldest-first, excluding Replace,
        // guarding delete manifests.
        let changelog_snapshots = self
            .ordered_changelog_snapshots(
                plan_context,
                self.append_scan.from_snapshot_id_exclusive(),
                to_snapshot_id,
            )
            .await?;
        if changelog_snapshots.is_empty() {
            return Ok(Box::pin(futures::stream::empty()));
        }

        // Step 2 + 3: walk the snapshots oldest-first, assigning ordinals, and build a
        // task per added/deleted entry of each snapshot's own added DATA manifests. The
        // changelog range is bounded, so collecting the tasks eagerly (rather than the
        // append scan's concurrent channel pipeline) keeps the per-snapshot ordinal
        // attachment simple and correct — each snapshot's tasks share one ordinal.
        let mut tasks: Vec<ChangelogScanTask> = Vec::new();
        for (ordinal, snapshot) in changelog_snapshots.iter().enumerate() {
            let change_ordinal = i32::try_from(ordinal).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Too many changelog snapshots in range to assign a change ordinal",
                )
            })?;
            let snapshot_id = snapshot.snapshot_id();

            // Load the snapshot's manifest list ONCE and split it three ways: the DATA
            // manifests the snapshot itself added (`added_snapshot_id == snapshot_id` —
            // its whole-file changes; a manifest carried forward from an older snapshot
            // belongs to that older snapshot and is read for ITS ordinal, while a
            // manifest this snapshot rewrote to delete a file carries this snapshot's
            // id, so its `Deleted` tombstones are read here), ALL data manifests (the
            // live-file candidates for row-level DeletedRows tasks), and the DELETE
            // manifests (only reachable in row-level mode — the default mode already
            // rejected any range carrying them).
            let manifest_list = snapshot
                .load_manifest_list(&self.file_io, &plan_context.table_metadata)
                .await?;
            let mut own_added_data_manifests = Vec::new();
            let mut all_data_manifests = Vec::new();
            let mut delete_manifests = Vec::new();
            for manifest_file in manifest_list.consume_entries() {
                match manifest_file.content {
                    ManifestContentType::Data => {
                        if manifest_file.added_snapshot_id == snapshot_id {
                            own_added_data_manifests.push(manifest_file.clone());
                        }
                        all_data_manifests.push(manifest_file);
                    }
                    ManifestContentType::Deletes => delete_manifests.push(manifest_file),
                }
            }

            // ENGINE-FIRST row-level mode: index this snapshot's delete files, split into
            // the deletes it ADDED vs the live ones that pre-existed it (Java
            // `DeletedRowsScanTask.addedDeletes()` vs `existingDeletes()`).
            let row_level_indexes = if self.include_row_level_deletes {
                Some(
                    Self::build_snapshot_delete_indexes(
                        plan_context,
                        &delete_manifests,
                        snapshot_id,
                    )
                    .await?,
                )
            } else {
                None
            };

            if !own_added_data_manifests.is_empty() {
                let snapshot_tasks = Self::plan_snapshot_change_tasks(
                    plan_context,
                    own_added_data_manifests,
                    change_ordinal,
                    snapshot_id,
                    row_level_indexes
                        .as_ref()
                        .map(|indexes| (indexes.added.clone(), indexes.existing.clone())),
                )
                .await?;
                tasks.extend(snapshot_tasks);
            }

            // ENGINE-FIRST row-level mode: every LIVE data file NOT added by this snapshot
            // that its ADDED delete files apply to yields a DeletedRows task. Skipped
            // entirely when the snapshot added no delete files, so an append/overwrite
            // snapshot plans identically with the flag on or off.
            if let Some(indexes) = &row_level_indexes
                && indexes.has_added_deletes
            {
                let deleted_rows_tasks = Self::plan_deleted_rows_tasks(
                    plan_context,
                    all_data_manifests,
                    change_ordinal,
                    snapshot_id,
                    indexes.added.clone(),
                    indexes.existing.clone(),
                )
                .await?;
                tasks.extend(deleted_rows_tasks);
            }
        }

        Ok(Box::pin(futures::stream::iter(tasks.into_iter().map(Ok))))
    }

    /// Returns the changelog snapshots in `(from_snapshot_id_exclusive, to_snapshot_id]`
    /// ordered OLDEST-FIRST (Java `orderedChangelogSnapshots`).
    ///
    /// Walks the parent chain from `to_snapshot_id` back to (but excluding)
    /// `from_snapshot_id_exclusive` — the same range walk the append scan uses — but:
    /// - EXCLUDES `Operation::Replace` snapshots (compaction rewrites files without
    ///   changing rows, so they produce no changelog), and
    /// - REJECTS the range with `FeatureUnsupported` if any kept snapshot references a
    ///   row-level DELETE manifest (Java throws `UnsupportedOperationException`: "Delete
    ///   files are currently not supported in changelog scans").
    ///
    /// The walk visits newest-first; the result is reversed to oldest-first so the caller
    /// can assign change ordinals (oldest → 0).
    async fn ordered_changelog_snapshots(
        &self,
        plan_context: &PlanContext,
        from_snapshot_id_exclusive: Option<i64>,
        to_snapshot_id: i64,
    ) -> Result<Vec<SnapshotRef>> {
        let metadata = &plan_context.table_metadata;

        // An equal from/to yields an empty range (Java `ancestorsBetween`).
        if from_snapshot_id_exclusive == Some(to_snapshot_id) {
            return Ok(vec![]);
        }

        let mut newest_first = Vec::new();
        let mut current = metadata.snapshot_by_id(to_snapshot_id).cloned();

        while let Some(snapshot) = current {
            // Stop BEFORE the exclusive start.
            if Some(snapshot.snapshot_id()) == from_snapshot_id_exclusive {
                break;
            }

            // Exclude Replace snapshots (compaction): they rewrite files without changing
            // rows, so they contribute no row-level changes.
            if snapshot.summary().operation != Operation::Replace {
                // Guard (default mode only): a snapshot referencing a row-level DELETE
                // manifest is out of scope for the data-file changelog — the exact Java
                // 1.10.0 rejection surface (`BaseIncrementalChangelogScan.
                // orderedChangelogSnapshots` throws `UnsupportedOperationException`).
                // The ENGINE-FIRST row-level mode lifts the guard and plans the
                // row-level task taxonomy instead.
                if !self.include_row_level_deletes
                    && self
                        .snapshot_has_delete_manifest(plan_context, &snapshot)
                        .await?
                {
                    return Err(Error::new(
                        ErrorKind::FeatureUnsupported,
                        "Delete files are currently not supported in changelog scans",
                    ));
                }
                newest_first.push(snapshot.clone());
            }

            current = match snapshot.parent_snapshot_id() {
                Some(parent_id) => metadata.snapshot_by_id(parent_id).cloned(),
                None => None,
            };
        }

        // Reverse to oldest-first (Java builds the deque with `addFirst`).
        newest_first.reverse();
        Ok(newest_first)
    }

    /// Returns whether `snapshot` references any row-level DELETE manifest — Java
    /// `!snapshot.deleteManifests(io).isEmpty()`. Loads the snapshot's manifest list and
    /// checks for any `ManifestContentType::Deletes` entry.
    async fn snapshot_has_delete_manifest(
        &self,
        plan_context: &PlanContext,
        snapshot: &SnapshotRef,
    ) -> Result<bool> {
        let manifest_list = snapshot
            .load_manifest_list(&self.file_io, &plan_context.table_metadata)
            .await?;
        Ok(manifest_list
            .entries()
            .iter()
            .any(|manifest_file| manifest_file.content == ManifestContentType::Deletes))
    }

    /// Reads the DELETE manifests of one changelog snapshot's manifest list and builds
    /// the two [`DeleteFileIndex`]es the ENGINE-FIRST row-level mode plans against: the
    /// delete files the snapshot itself ADDED (status `Added`, committed by this
    /// snapshot — Java `DeletedRowsScanTask.addedDeletes()`) and the LIVE delete files
    /// that pre-existed it (an `Added` entry from an earlier snapshot carried forward,
    /// or an `Existing` entry — Java `existingDeletes()`, "must be applied prior to
    /// determining which records are deleted"). A `Deleted` tombstone (a delete file
    /// removed by this snapshot) belongs to neither.
    async fn build_snapshot_delete_indexes(
        plan_context: &PlanContext,
        delete_manifests: &[ManifestFile],
        snapshot_id: i64,
    ) -> Result<SnapshotDeleteIndexes> {
        let (added, mut added_tx) = DeleteFileIndex::new();
        let (existing, mut existing_tx) = DeleteFileIndex::new();
        let mut has_added_deletes = false;

        for manifest_file in delete_manifests {
            let manifest = plan_context
                .object_cache
                .get_manifest(manifest_file)
                .await?;
            for entry in manifest.entries() {
                if entry.status() == ManifestStatus::Deleted {
                    continue;
                }
                // Manifest loading applies V2/V3 metadata inheritance
                // (`ManifestEntry::inherit_data`), so `snapshot_id()` is populated for
                // entries the manifest's own snapshot added; fall back to the manifest's
                // adding snapshot defensively.
                let entry_snapshot_id = entry
                    .snapshot_id()
                    .unwrap_or(manifest_file.added_snapshot_id);
                let context = DeleteFileContext {
                    manifest_entry: entry.clone(),
                    partition_spec_id: manifest_file.partition_spec_id,
                };
                let added_by_this_snapshot =
                    entry.status() == ManifestStatus::Added && entry_snapshot_id == snapshot_id;
                let sender = if added_by_this_snapshot {
                    has_added_deletes = true;
                    &mut added_tx
                } else {
                    &mut existing_tx
                };
                sender
                    .send(context)
                    .await
                    .map_err(|_| Error::new(ErrorKind::Unexpected, "mpsc channel SendError"))?;
            }
        }
        drop(added_tx);
        drop(existing_tx);

        Ok(SnapshotDeleteIndexes {
            added,
            existing,
            has_added_deletes,
        })
    }

    /// Plans the changelog tasks for ONE snapshot's own added DATA manifests, tagging each
    /// with the snapshot's `change_ordinal` and `commit_snapshot_id`. ADDED entries become
    /// `AddedRows` (INSERT) tasks, DELETED entries become `DeletedDataFile` (DELETE)
    /// tasks; `Existing` entries are skipped.
    ///
    /// Reuses the shared `PlanContext::build_manifest_file_contexts_from_files` (the same
    /// partition-filter pruning + residual evaluator the append + normal scans use) over
    /// an EMPTY delete index, then converts the surviving entries into
    /// `ChangelogScanTask`s. In the default data-file mode (`row_level_indexes = None`)
    /// every task carries empty delete lists — Java 1.10.0's `NO_DELETES`; in row-level
    /// mode the `(added, existing)` indexes attach the applicable delete files per the
    /// Java task taxonomy.
    async fn plan_snapshot_change_tasks(
        plan_context: &PlanContext,
        selected_manifests: Vec<ManifestFile>,
        change_ordinal: i32,
        snapshot_id: i64,
        row_level_indexes: Option<(DeleteFileIndex, DeleteFileIndex)>,
    ) -> Result<Vec<ChangelogScanTask>> {
        let manifest_count = selected_manifests.len().max(1);
        let (manifest_entry_data_ctx_tx, manifest_entry_data_ctx_rx) = channel(manifest_count);
        // A never-fed delete-manifest channel (we only pass DATA manifests).
        let (delete_ctx_tx, _delete_ctx_rx) = channel::<ManifestEntryContext>(1);
        // The output channel carries `Result` so a producer's manifest-fetch error reaches
        // the consumer instead of being silently dropped when the producer task ends.
        let (task_tx, task_rx) = channel::<Result<ChangelogScanTask>>(manifest_count);

        // An EMPTY delete index: a changelog task applies no delete files.
        let (delete_file_idx, delete_file_tx) = DeleteFileIndex::new();
        drop(delete_file_tx);

        let manifest_file_contexts = plan_context.build_manifest_file_contexts_from_files(
            selected_manifests,
            manifest_entry_data_ctx_tx,
            delete_file_idx,
            delete_ctx_tx,
        )?;

        // Spawn the producers (fetch each manifest, stream its entries into the entry
        // channel) so the consumer below can drain CONCURRENTLY — a manifest may hold more
        // entries than the channel's capacity, so a producer's `send` can block until the
        // consumer reads. Draining in the same task after a blocking send would deadlock. A
        // producer error is forwarded into the output channel so it is not lost.
        let mut producer_error_tx = task_tx.clone();
        spawn(async move {
            let result = futures::stream::iter(manifest_file_contexts)
                .try_for_each_concurrent(manifest_count, |ctx| async move {
                    ctx.fetch_manifest_and_stream_manifest_entries().await
                })
                .await;
            if let Err(error) = result {
                let _ = producer_error_tx.send(Err(error)).await;
            }
        });

        // Convert each kept entry into a changelog task, sending the result to the output
        // channel. Run on a task so it interleaves with the producers (avoiding the
        // blocking-send deadlock).
        spawn(async move {
            let result = manifest_entry_data_ctx_rx
                .map(Ok)
                .try_for_each(|manifest_entry_context| {
                    let mut task_tx = task_tx.clone();
                    // Clones of the two index handles (cheap `Arc` clones), moved into the
                    // per-entry future.
                    let row_level_indexes = row_level_indexes.clone();
                    async move {
                        let task = Self::changelog_task_from_entry(
                            manifest_entry_context,
                            change_ordinal,
                            snapshot_id,
                            row_level_indexes,
                        )
                        .await;
                        match task {
                            Ok(Some(task)) => {
                                let _ = task_tx.send(Ok(task)).await;
                            }
                            Ok(None) => {}
                            Err(error) => {
                                let _ = task_tx.send(Err(error)).await;
                            }
                        }
                        Ok(())
                    }
                })
                .await;
            // `try_for_each` over `map(Ok)` never yields an Err; bind to satisfy the type.
            let _: Result<()> = result;
        });

        task_rx.try_collect().await
    }

    /// Plans the ENGINE-FIRST [`ChangelogTaskKind::DeletedRows`] tasks for ONE changelog
    /// snapshot: every LIVE data file that was NOT added by this snapshot but is matched
    /// by delete files the snapshot ADDED yields a task carrying exactly those added
    /// deletes, plus the pre-existing deletes to apply first. Files the snapshot itself
    /// added are excluded — their matching deletes fold into their `AddedRows` task
    /// instead (the Java `AddedRowsScanTask` javadoc contract), and `Deleted` tombstones
    /// are excluded — a file removed outright is a `DeletedDataFile` change.
    ///
    /// `all_data_manifests` is the snapshot's ENTIRE data-manifest list (own + carried
    /// forward): the live files a delete can hit live anywhere in it. Uses the same
    /// channel pipeline + partition-filter pruning as `plan_snapshot_change_tasks`.
    async fn plan_deleted_rows_tasks(
        plan_context: &PlanContext,
        all_data_manifests: Vec<ManifestFile>,
        change_ordinal: i32,
        snapshot_id: i64,
        added_index: DeleteFileIndex,
        existing_index: DeleteFileIndex,
    ) -> Result<Vec<ChangelogScanTask>> {
        let manifest_count = all_data_manifests.len().max(1);
        let (manifest_entry_data_ctx_tx, manifest_entry_data_ctx_rx) = channel(manifest_count);
        // A never-fed delete-manifest channel (we only pass DATA manifests).
        let (delete_ctx_tx, _delete_ctx_rx) = channel::<ManifestEntryContext>(1);
        let (task_tx, task_rx) = channel::<Result<ChangelogScanTask>>(manifest_count);

        // The pipeline's task-attachment index stays EMPTY: one pipeline index cannot
        // represent the added/existing split, so both attachments are made explicitly in
        // `deleted_rows_task_from_entry`.
        let (empty_delete_index, empty_delete_tx) = DeleteFileIndex::new();
        drop(empty_delete_tx);

        let manifest_file_contexts = plan_context.build_manifest_file_contexts_from_files(
            all_data_manifests,
            manifest_entry_data_ctx_tx,
            empty_delete_index,
            delete_ctx_tx,
        )?;

        // Same producer/consumer split as `plan_snapshot_change_tasks` — see the deadlock
        // note there.
        let mut producer_error_tx = task_tx.clone();
        spawn(async move {
            let result = futures::stream::iter(manifest_file_contexts)
                .try_for_each_concurrent(manifest_count, |ctx| async move {
                    ctx.fetch_manifest_and_stream_manifest_entries().await
                })
                .await;
            if let Err(error) = result {
                let _ = producer_error_tx.send(Err(error)).await;
            }
        });

        spawn(async move {
            let result = manifest_entry_data_ctx_rx
                .map(Ok)
                .try_for_each(|manifest_entry_context| {
                    let mut task_tx = task_tx.clone();
                    let added_index = added_index.clone();
                    let existing_index = existing_index.clone();
                    async move {
                        let task = Self::deleted_rows_task_from_entry(
                            manifest_entry_context,
                            change_ordinal,
                            snapshot_id,
                            added_index,
                            existing_index,
                        )
                        .await;
                        match task {
                            Ok(Some(task)) => {
                                let _ = task_tx.send(Ok(task)).await;
                            }
                            Ok(None) => {}
                            Err(error) => {
                                let _ = task_tx.send(Err(error)).await;
                            }
                        }
                        Ok(())
                    }
                })
                .await;
            // `try_for_each` over `map(Ok)` never yields an Err; bind to satisfy the type.
            let _: Result<()> = result;
        });

        task_rx.try_collect().await
    }

    /// Converts a live, not-added-here data-manifest entry into a
    /// [`ChangelogTaskKind::DeletedRows`] task, or `None` when the entry is out of scope
    /// (a `Deleted` tombstone, a file added by this snapshot, pruned by the scan filter,
    /// or matched by none of the snapshot's ADDED delete files).
    async fn deleted_rows_task_from_entry(
        manifest_entry_context: ManifestEntryContext,
        change_ordinal: i32,
        snapshot_id: i64,
        added_index: DeleteFileIndex,
        existing_index: DeleteFileIndex,
    ) -> Result<Option<ChangelogScanTask>> {
        // Only LIVE files can lose rows to added deletes: a `Deleted` tombstone's file
        // was removed outright by this snapshot — that change is the `DeletedDataFile`
        // task planned from the snapshot's own manifests, and it must not ALSO surface
        // as row-level deletes.
        if manifest_entry_context.manifest_entry.status() == ManifestStatus::Deleted {
            return Ok(None);
        }

        // Only DATA manifests are fed to this planner, but guard the invariant the same
        // way the other planners do.
        if manifest_entry_context.manifest_entry.content_type() != DataContentType::Data {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Encountered an entry for a delete file in an incremental changelog scan",
            ));
        }

        // A file ADDED by this snapshot folds its matching deletes into its `AddedRows`
        // task (Java `AddedRowsScanTask` javadoc: "added data files may have matching
        // delete files ... committed in the same snapshot") — never a separate
        // DeletedRows task.
        if manifest_entry_context.manifest_entry.snapshot_id() == Some(snapshot_id) {
            return Ok(None);
        }

        if !Self::entry_matches_scan_filter(&manifest_entry_context)? {
            return Ok(None);
        }

        let added_deletes = added_index
            .get_deletes_for_data_file(
                manifest_entry_context.manifest_entry.data_file(),
                manifest_entry_context.manifest_entry.sequence_number(),
            )
            .await?;
        if added_deletes.is_empty() {
            return Ok(None);
        }
        let existing_deletes = existing_index
            .get_deletes_for_data_file(
                manifest_entry_context.manifest_entry.data_file(),
                manifest_entry_context.manifest_entry.sequence_number(),
            )
            .await?;

        let mut file_scan_task = manifest_entry_context.into_file_scan_task().await?;
        // A plain MoR read of the embedded task yields the rows live BEFORE this change
        // (Java `DeletedRowsScanTask.existingDeletes()` "must be applied prior to
        // determining which records are deleted"); the engine then uses `added_deletes`
        // as the SELECTOR of which of those rows became deleted.
        file_scan_task.deletes = existing_deletes.clone();

        Ok(Some(ChangelogScanTask {
            change_ordinal,
            // The change was committed by the snapshot that ADDED the deletes — not by
            // the (older) snapshot that added the data file.
            commit_snapshot_id: snapshot_id,
            kind: ChangelogTaskKind::DeletedRows,
            added_deletes,
            existing_deletes,
            file_scan_task,
        }))
    }

    /// Converts a single manifest entry into a [`ChangelogScanTask`], or `None` when the
    /// entry is `Existing` (no change) or pruned by the partition filter.
    ///
    /// Mirrors Java `CreateDataFileChangeTasks.apply`: ADDED → `BaseAddedRowsScanTask`
    /// (INSERT), DELETED → `BaseDeletedDataFileScanTask` (DELETE), each carrying the
    /// change ordinal + `commit_snapshot_id = entry.snapshotId()` (the snapshot stamped
    /// on the entry, falling back to the snapshot id when an inherited entry's snapshot
    /// id is not yet materialized). Applies the same partition-filter +
    /// inclusive-metrics pruning as the append scan before emitting.
    ///
    /// In the default data-file mode (`row_level_indexes = None`) both delete lists are
    /// empty (Java 1.10.0 passes `NO_DELETES` to both task constructors). In row-level
    /// mode the `(added, existing)` indexes attach: the deletes ADDED with the file to
    /// an `AddedRows` task (Java `AddedRowsScanTask.deletes()` — the same-snapshot fold),
    /// and the PRE-EXISTING deletes to a `DeletedDataFile` task (Java
    /// `DeletedDataFileScanTask.existingDeletes()` — so only rows live at removal are
    /// output as deleted).
    async fn changelog_task_from_entry(
        manifest_entry_context: ManifestEntryContext,
        change_ordinal: i32,
        snapshot_id: i64,
        row_level_indexes: Option<(DeleteFileIndex, DeleteFileIndex)>,
    ) -> Result<Option<ChangelogScanTask>> {
        let kind = match manifest_entry_context.manifest_entry.status() {
            ManifestStatus::Added => ChangelogTaskKind::AddedRows,
            ManifestStatus::Deleted => ChangelogTaskKind::DeletedDataFile,
            // `Existing` entries were added by an earlier snapshot and copied forward —
            // they are not a change committed by THIS snapshot. Java `ignoreExisting()`.
            ManifestStatus::Existing => return Ok(None),
        };

        // A changelog scan never reads a delete-file manifest (we only select DATA
        // manifests), but guard the invariant the same way the append scan does.
        if manifest_entry_context.manifest_entry.content_type() != DataContentType::Data {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Encountered an entry for a delete file in an incremental changelog scan",
            ));
        }

        // The snapshot that committed the change. An `Added`/`Deleted` entry written by
        // this snapshot carries its id; a V2/V3 added entry whose id is inherited at read
        // time may be absent, so fall back to the snapshot id (Java reads the entry's
        // snapshotId, which inheritance has populated by plan time).
        let commit_snapshot_id = manifest_entry_context
            .manifest_entry
            .snapshot_id()
            .unwrap_or(snapshot_id);

        // Apply the same partition-filter + inclusive-metrics pruning as the append scan.
        if !Self::entry_matches_scan_filter(&manifest_entry_context)? {
            return Ok(None);
        }

        // Row-level mode: attach the applicable delete files per the Java task taxonomy.
        let (added_deletes, existing_deletes) = match &row_level_indexes {
            Some((added_index, existing_index)) => {
                let data_file = manifest_entry_context.manifest_entry.data_file();
                let sequence_number = manifest_entry_context.manifest_entry.sequence_number();
                match kind {
                    // Deletes committed with (or squashed onto) the added file. A
                    // pre-existing delete can never apply to a file whose data sequence
                    // number postdates it, so `existing` is structurally empty.
                    ChangelogTaskKind::AddedRows => (
                        added_index
                            .get_deletes_for_data_file(data_file, sequence_number)
                            .await?,
                        Vec::new(),
                    ),
                    // The historical deletes to apply so only rows still live at removal
                    // are output as deleted.
                    ChangelogTaskKind::DeletedDataFile => (
                        Vec::new(),
                        existing_index
                            .get_deletes_for_data_file(data_file, sequence_number)
                            .await?,
                    ),
                    // Never produced by this converter — DeletedRows tasks come from
                    // `deleted_rows_task_from_entry`.
                    ChangelogTaskKind::DeletedRows => (Vec::new(), Vec::new()),
                }
            }
            None => (Vec::new(), Vec::new()),
        };

        let mut file_scan_task = manifest_entry_context.into_file_scan_task().await?;
        if row_level_indexes.is_some() {
            // A plain MoR read of the task applies: the added deletes for an `AddedRows`
            // task (yielding the NET inserted rows) and the existing deletes for a
            // `DeletedDataFile` task (yielding the rows live at removal). In default mode
            // the pipeline's empty index already left `deletes` empty — untouched.
            file_scan_task.deletes = match kind {
                ChangelogTaskKind::AddedRows => added_deletes.clone(),
                ChangelogTaskKind::DeletedDataFile | ChangelogTaskKind::DeletedRows => {
                    existing_deletes.clone()
                }
            };
        }

        Ok(Some(ChangelogScanTask {
            change_ordinal,
            commit_snapshot_id,
            kind,
            added_deletes,
            existing_deletes,
            file_scan_task,
        }))
    }

    /// Applies the scan's partition-filter + inclusive-metrics pruning to one manifest
    /// entry — the same two evaluator steps the append scan and the normal scan run —
    /// returning whether the entry's data file can match the scan filter. Always `true`
    /// for an unfiltered scan.
    fn entry_matches_scan_filter(manifest_entry_context: &ManifestEntryContext) -> Result<bool> {
        if let Some(ref bound_predicates) = manifest_entry_context.bound_predicates {
            let BoundPredicates {
                snapshot_bound_predicate,
                partition_bound_predicate,
            } = bound_predicates.as_ref();

            let expression_evaluator_cache =
                manifest_entry_context.expression_evaluator_cache.as_ref();

            let expression_evaluator = expression_evaluator_cache.get(
                manifest_entry_context.partition_spec_id,
                partition_bound_predicate,
            )?;

            // skip any data file whose partition data indicates it can't match the filter
            if !expression_evaluator.eval(manifest_entry_context.manifest_entry.data_file())? {
                return Ok(false);
            }

            // skip any data file whose metrics don't match the filter
            if !crate::expr::visitors::inclusive_metrics_evaluator::InclusiveMetricsEvaluator::eval(
                snapshot_bound_predicate,
                manifest_entry_context.manifest_entry.data_file(),
                false,
            )? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Returns the inclusive `to` snapshot id of this scan, if the table has snapshots.
    pub fn to_snapshot_id(&self) -> Option<i64> {
        self.append_scan.to_snapshot_id()
    }

    /// Returns the exclusive `from` snapshot id of this scan (`None` = the whole lineage).
    pub fn from_snapshot_id_exclusive(&self) -> Option<i64> {
        self.append_scan.from_snapshot_id_exclusive()
    }

    /// The schema the scan projects (the `to` snapshot's schema), if the table has snapshots.
    pub fn snapshot_schema(&self) -> Option<&SchemaRef> {
        self.append_scan.snapshot_schema()
    }
}

/// Returns whether `ancestor_id` is an ancestor of `snapshot_id` (inclusive of
/// `snapshot_id` itself) — Java `SnapshotUtil.isAncestorOf`. Walks the parent chain of
/// `snapshot_id`.
fn is_ancestor_of(table: &Table, snapshot_id: i64, ancestor_id: i64) -> bool {
    let metadata = table.metadata();
    let mut current = metadata.snapshot_by_id(snapshot_id).cloned();
    while let Some(snapshot) = current {
        if snapshot.snapshot_id() == ancestor_id {
            return true;
        }
        current = match snapshot.parent_snapshot_id() {
            Some(parent_id) => metadata.snapshot_by_id(parent_id).cloned(),
            None => None,
        };
    }
    false
}

/// Returns whether some ancestor of `snapshot_id` has `parent_id == parent_ancestor_id`
/// — Java `SnapshotUtil.isParentAncestorOf`. This is the exclusive-`from` validity check:
/// `from` is a parent of some ancestor of `to`, so the range `(from, to]` is well-defined
/// even when `from` itself has been expired.
fn is_parent_ancestor_of(table: &Table, snapshot_id: i64, parent_ancestor_id: i64) -> bool {
    let metadata = table.metadata();
    let mut current = metadata.snapshot_by_id(snapshot_id).cloned();
    while let Some(snapshot) = current {
        if snapshot.parent_snapshot_id() == Some(parent_ancestor_id) {
            return true;
        }
        current = match snapshot.parent_snapshot_id() {
            Some(parent_id) => metadata.snapshot_by_id(parent_id).cloned(),
            None => None,
        };
    }
    false
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::io::BufReader;

    use arrow_array::RecordBatch;
    use futures::TryStreamExt;

    use crate::arrow::ArrowReaderBuilder;
    use crate::expr::Reference;
    use crate::memory::tests::new_memory_catalog;
    use crate::scan::FileScanTask;
    use crate::scan::tests::{NAME_MAPPING_X1_Y2, TableTestFixture, decode_int64_column};
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Literal,
        Operation, Struct, TableMetadata,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, ErrorKind, TableCreation, TableIdent};

    /// Create a V3 table partitioned by identity(x) in the catalog from the shared
    /// `TableMetadataV3ValidMinimal` fixture (schema `x, y, z` longs; spec id 0 =
    /// identity(x)). Inlined here rather than reusing `transaction::tests::
    /// make_v3_minimal_table_in_catalog` because that module is private to `transaction/`;
    /// copying the 15-line helper keeps the visibility narrow (lessons 2026-06-08).
    async fn make_minimal_table(catalog: &impl Catalog) -> Table {
        let table_ident =
            TableIdent::from_strs([format!("ns-{}", uuid::Uuid::new_v4()), "t".to_string()])
                .unwrap();
        catalog
            .create_namespace(table_ident.namespace(), HashMap::new())
            .await
            .unwrap();

        let file = File::open(format!(
            "{}/testdata/table_metadata/TableMetadataV3ValidMinimal.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let base_metadata =
            serde_json::from_reader::<_, TableMetadata>(BufReader::new(file)).unwrap();

        let table_creation = TableCreation::builder()
            .schema((**base_metadata.current_schema()).clone())
            .partition_spec((**base_metadata.default_partition_spec()).clone())
            .sort_order((**base_metadata.default_sort_order()).clone())
            .name(table_ident.name().to_string())
            .format_version(FormatVersion::V3)
            .build();

        catalog
            .create_table(table_ident.namespace(), table_creation)
            .await
            .unwrap()
    }

    /// Build a data file routed to partition `x = part_value` (the V3 minimal table is
    /// partitioned by identity(x), spec id 0) with a unique path.
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

    /// Append the given files in a single fast-append commit and return the updated table.
    async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let action = tx.fast_append().add_data_files(files);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Collect the data-file paths an incremental append scan plans.
    async fn planned_paths(scan: &super::IncrementalAppendScan) -> HashSet<String> {
        let tasks: Vec<FileScanTask> = scan
            .plan_files()
            .await
            .expect("plan_files should succeed")
            .try_collect()
            .await
            .expect("collecting file scan tasks should succeed");
        tasks.into_iter().map(|t| t.data_file_path).collect()
    }

    /// CORE BEHAVIOR: from=S0(exclusive) to=S2(inclusive) returns ONLY the files appended
    /// in S1 + S2, never S0's file. Risk pinned: the range planner must include both later
    /// append snapshots and exclude the starting snapshot's own files.
    #[tokio::test]
    async fn test_incremental_append_returns_appends_in_range_excluding_from() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s2.parquet", 1)]).await;
        let s2 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s2)
            .build()
            .unwrap();

        let paths = planned_paths(&scan).await;
        assert_eq!(
            paths,
            HashSet::from(["s1.parquet".to_string(), "s2.parquet".to_string()]),
            "should return only S1 + S2 appended files, not S0's"
        );
    }

    /// EXCLUSIVE-FROM BOUNDARY: from=S1(exclusive) to=S2 returns ONLY S2's file — S1's own
    /// file is NOT included even though S1 is the immediate `from`. Mutation-pins the
    /// exclusive boundary (an inclusive walk would wrongly add S1's file).
    #[tokio::test]
    async fn test_incremental_append_from_is_exclusive() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("s2.parquet", 1)]).await;
        let s2 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s1)
            .to_snapshot_id(s2)
            .build()
            .unwrap();

        let paths = planned_paths(&scan).await;
        assert_eq!(
            paths,
            HashSet::from(["s2.parquet".to_string()]),
            "exclusive from=S1 must exclude S1's own file"
        );
    }

    /// INCLUSIVE-FROM: from=S1(inclusive) to=S2 returns BOTH S1 and S2 — the inclusive
    /// bound resolves to S1's parent (S0) as the exclusive boundary, so S1's APPEND files
    /// are included. Pins the inclusive→parent resolution.
    #[tokio::test]
    async fn test_incremental_append_from_inclusive_includes_from_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("s2.parquet", 1)]).await;
        let s2 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_append_scan()
            .from_snapshot_id_inclusive(s1)
            .to_snapshot_id(s2)
            .build()
            .unwrap();

        let paths = planned_paths(&scan).await;
        assert_eq!(
            paths,
            HashSet::from(["s1.parquet".to_string(), "s2.parquet".to_string()]),
            "inclusive from=S1 must include S1's own file"
        );
    }

    /// FROM == TO (exclusive): rejected by the `isParentAncestorOf` precondition, exactly
    /// like Java (a snapshot is never a parent ancestor of itself). Java's `ancestorsBetween`
    /// empty short-circuit is unreachable here because the precondition fails first.
    #[tokio::test]
    async fn test_incremental_append_from_equals_to_exclusive_is_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let result = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s1)
            .to_snapshot_id(s1)
            .build();
        assert!(
            result.is_err(),
            "from == to (exclusive) must be rejected: a snapshot is not its own parent ancestor"
        );
    }

    /// EMPTY RANGE (no APPEND snapshots in range): the only snapshot in `(from, to]` is an OVERWRITE,
    /// so the append-only filter drops it and zero tasks are planned. This is the
    /// Java-reachable "empty range → zero tasks" case (the range is non-empty but contains
    /// no APPEND snapshot).
    #[tokio::test]
    async fn test_incremental_append_range_with_no_append_snapshots_is_empty() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        // S0 (append a, b), then S1 = OVERWRITE (delete a, add c). The range (S0, S1] holds
        // only the overwrite.
        let table = append_files(&catalog, &table, vec![
            data_file("a.parquet", 1),
            data_file("b.parquet", 1),
        ])
        .await;
        let s0 = table.metadata().current_snapshot_id().unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("a.parquet")
            .add_file(data_file("c.parquet", 1));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .build()
            .unwrap();

        let paths = planned_paths(&scan).await;
        assert!(
            paths.is_empty(),
            "a range whose only snapshot is an overwrite plans zero tasks"
        );
    }

    /// APPEND-ONLY: an OVERWRITE snapshot in the range is EXCLUDED — only files added by
    /// the APPEND snapshot are returned, NOT the file the overwrite added. Mutation-pins the
    /// `Operation::Append` op filter (dropping it would include the overwrite's added file).
    #[tokio::test]
    async fn test_incremental_append_excludes_overwrite_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        // S0 (append a), S1 (append b), S2 (overwrite: delete b, add c → Operation::Overwrite).
        let table = append_files(&catalog, &table, vec![data_file("a.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("b.parquet", 1)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("b.parquet")
            .add_file(data_file("c.parquet", 1));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s2 = table.metadata().current_snapshot_id().unwrap();

        // The S2 snapshot must be an OVERWRITE (delete + add).
        assert_eq!(
            table
                .metadata()
                .snapshot_by_id(s2)
                .unwrap()
                .summary()
                .operation,
            crate::spec::Operation::Overwrite,
            "S2 must be an overwrite for this test to be meaningful"
        );

        let scan = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s2)
            .build()
            .unwrap();

        let paths = planned_paths(&scan).await;
        assert_eq!(
            paths,
            HashSet::from(["b.parquet".to_string()]),
            "only the APPEND (S1) file b is returned; the OVERWRITE (S2) file c is excluded"
        );
    }

    /// FILTER PRUNES BY PARTITION: a `with_filter(x == 10)` over an identity(x)-partitioned
    /// table prunes the appended file in partition x = 20. Reuses the same partition-filter
    /// machinery as the normal scan.
    #[tokio::test]
    async fn test_incremental_append_with_filter_prunes_by_partition() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("base.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        // S1 appends one file in x = 10 and one in x = 20.
        let table = append_files(&catalog, &table, vec![
            data_file("x10.parquet", 10),
            data_file("x20.parquet", 20),
        ])
        .await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .with_filter(Reference::new("x").equal_to(Datum::long(10)))
            .build()
            .unwrap();

        let paths = planned_paths(&scan).await;
        assert_eq!(
            paths,
            HashSet::from(["x10.parquet".to_string()]),
            "filter x == 10 must prune the x = 20 appended file"
        );
    }

    /// DEFAULT to=current: when `to_snapshot_id` is unset the scan ends at the current
    /// snapshot. from=S0(excl), no `to` ⇒ returns S1 + S2.
    #[tokio::test]
    async fn test_incremental_append_to_defaults_to_current_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s2.parquet", 1)]).await;

        let scan = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s0)
            .build()
            .unwrap();

        assert_eq!(
            scan.to_snapshot_id(),
            table.metadata().current_snapshot_id(),
            "unset to_snapshot_id must default to the current snapshot"
        );

        let paths = planned_paths(&scan).await;
        assert_eq!(
            paths,
            HashSet::from(["s1.parquet".to_string(), "s2.parquet".to_string()]),
            "default to=current returns the appends after S0"
        );
    }

    /// WHOLE LINEAGE: with no `from` set, the scan returns every appended file in the
    /// current lineage (Java's null `fromSnapshotId` → walk to the root).
    #[tokio::test]
    async fn test_incremental_append_no_from_scans_whole_lineage() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s2.parquet", 1)]).await;

        let scan = table.incremental_append_scan().build().unwrap();

        let paths = planned_paths(&scan).await;
        assert_eq!(
            paths,
            HashSet::from([
                "s0.parquet".to_string(),
                "s1.parquet".to_string(),
                "s2.parquet".to_string(),
            ]),
            "no from bound returns every appended file in the lineage"
        );
    }

    /// EMPTY TABLE: an incremental scan on a table with no snapshots produces zero tasks
    /// (no current snapshot, no explicit to).
    #[tokio::test]
    async fn test_incremental_append_empty_table_is_empty() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let scan = table.incremental_append_scan().build().unwrap();
        let paths = planned_paths(&scan).await;
        assert!(paths.is_empty(), "an empty table plans zero tasks");
    }

    /// VALIDATION: a non-ancestor exclusive `from` is rejected (Java
    /// `isParentAncestorOf` precondition).
    #[tokio::test]
    async fn test_incremental_append_rejects_non_ancestor_from() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        // S1 is a descendant of S0, so from=S1 to=S0 is invalid (S1 is not a parent
        // ancestor of S0).
        let result = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s1)
            .to_snapshot_id(s0)
            .build();
        assert!(
            result.is_err(),
            "a from that is not a parent ancestor of to must be rejected"
        );
    }

    /// ADDED-MANIFEST FILTER: a manifest carried forward into a later snapshot (not added
    /// by that snapshot) must NOT re-surface its files. After S0 appends a, S1 appends b;
    /// S1's manifest list carries S0's manifest forward, but only S1's OWN added manifest
    /// (holding b) counts for the (S0, S1] range. Mutation-pins the
    /// `added_snapshot_id == snapshot_id` manifest filter and the `Added`-entry filter.
    #[tokio::test]
    async fn test_incremental_append_only_counts_snapshots_own_added_manifests() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("a.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("b.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .build()
            .unwrap();

        let paths = planned_paths(&scan).await;
        assert_eq!(
            paths,
            HashSet::from(["b.parquet".to_string()]),
            "only S1's own added manifest (b) counts; S0's carried-forward manifest (a) does not"
        );
    }

    /// ADDED-ENTRY FILTER (in isolation): a snapshot's OWN added manifest can carry a
    /// non-`Added` entry (an `Existing` file copied forward, or a `Deleted` tombstone) when
    /// that snapshot rewrote a manifest — Java's `MergeAppend` produces exactly this. The
    /// incremental append scan must keep ONLY the `Added` entry (Java
    /// `filterManifestEntries(status == ADDED)`), never the `Existing`/`Deleted` ones.
    ///
    /// The fast-append fixtures elsewhere in this module can't pin this: a fast-append's own
    /// manifest holds only `Added` entries, so the status filter is never exercised. This
    /// test reuses `TableTestFixture::setup_manifest_files`, which writes a single manifest —
    /// added by the CURRENT (APPEND) snapshot — holding `1.parquet` (Added), `2.parquet`
    /// (Deleted), `3.parquet` (Existing). The current snapshot's parent is the exclusive
    /// `from`, so the range `(parent, current]` selects exactly that manifest. A normal scan
    /// over the same fixture returns BOTH `1.parquet` and `3.parquet`; the incremental scan
    /// must return ONLY `1.parquet`. Mutation-pins the `status == Added` entry filter:
    /// dropping it re-surfaces `3.parquet`.
    #[tokio::test]
    async fn test_incremental_append_keeps_only_added_entries_of_own_manifest() {
        let mut fixture = crate::scan::tests::TableTestFixture::new();
        fixture.setup_manifest_files().await;

        let metadata = fixture.table.metadata();
        let current_snapshot_id = metadata.current_snapshot_id().unwrap();
        let parent_snapshot_id = metadata
            .current_snapshot()
            .unwrap()
            .parent_snapshot_id()
            .unwrap();

        let scan = fixture
            .table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(parent_snapshot_id)
            .to_snapshot_id(current_snapshot_id)
            .build()
            .unwrap();

        let paths = planned_paths(&scan).await;
        assert_eq!(
            paths,
            HashSet::from([format!("{}/1.parquet", &fixture.table_location)]),
            "only the Added entry (1.parquet) is returned; the Existing (3.parquet) and \
             Deleted (2.parquet) entries of the snapshot's own manifest are excluded"
        );
    }

    // ---- incremental name-mapping wiring pins ----
    //
    // The `name_mapping: parse_name_mapping(...)` line in `IncrementalAppendScanBuilder::build`
    // is a SEPARATE wiring site from the snapshot scan's (`TableScanBuilder::build`): the
    // snapshot-scan pins in `scan/mod.rs` stay green if the incremental parse is dropped. The
    // pins below uniquely guard the incremental site (mutation-proven: hardcoding
    // `name_mapping: None` there reds exactly these while the snapshot pins stay green).

    /// Resolves the incremental range `(parent, current]` over a [`TableTestFixture`]: the
    /// fixture's single written manifest is added by the CURRENT snapshot, so this range
    /// selects exactly that manifest (the same technique as
    /// [`test_incremental_append_keeps_only_added_entries_of_own_manifest`]).
    fn parent_to_current_range(table: &Table) -> (i64, i64) {
        let metadata = table.metadata();
        let current_snapshot = metadata
            .current_snapshot()
            .expect("fixture has a current snapshot");
        let parent = current_snapshot
            .parent_snapshot_id()
            .expect("fixture's current snapshot has a parent");
        (parent, current_snapshot.snapshot_id())
    }

    /// PLAN-LEVEL NAME MAPPING: an incremental append scan over a table whose metadata
    /// carries `schema.name-mapping.default` yields tasks that ALL carry the parsed mapping
    /// CONTENT (field ids + names, not merely `is_some`) — mirroring the snapshot-scan pin
    /// `test_plan_threads_name_mapping_onto_every_task` in `scan/mod.rs`, but driving the
    /// `IncrementalAppendScanBuilder::build` wiring site the snapshot pins never touch.
    #[tokio::test]
    async fn test_incremental_append_threads_name_mapping_onto_every_task() {
        let mut fixture = TableTestFixture::new_with_name_mapping_property(NAME_MAPPING_X1_Y2);
        fixture.setup_name_mapping_manifest_files().await;
        let (parent, current) = parent_to_current_range(&fixture.table);

        let scan = fixture
            .table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(parent)
            .to_snapshot_id(current)
            .build()
            .expect("building the incremental scan over a name-mapped table should succeed");

        let tasks: Vec<FileScanTask> = scan
            .plan_files()
            .await
            .expect("plan_files should succeed")
            .try_collect()
            .await
            .expect("collecting file scan tasks should succeed");

        assert!(
            !tasks.is_empty(),
            "the name-mapping fixture must produce at least one incremental task"
        );
        for task in &tasks {
            let mapping = task
                .name_mapping
                .as_ref()
                .expect("every incremental task carries the parsed name mapping");
            let fields = mapping.fields();
            assert_eq!(fields.len(), 2, "mapping must have both mapped fields");
            assert_eq!(fields[0].field_id(), Some(1));
            assert_eq!(fields[0].names().to_vec(), vec!["x".to_string()]);
            assert_eq!(fields[1].field_id(), Some(2));
            assert_eq!(fields[1].names().to_vec(), vec!["y".to_string()]);
        }
    }

    /// END-TO-END NAME MAPPING: an ID-less parquet whose physical column order is REVERSED
    /// relative to the table schema, appended WITHIN the incremental range, reads to the
    /// CORRECT columns through the incremental scan's stream. `IncrementalAppendScan` has no
    /// `to_arrow`, so the planned stream feeds the same [`ArrowReaderBuilder`] path
    /// `TableScan::to_arrow` uses — exactly as an engine would. The mapping is NON-trivial:
    /// a positional fallback would read physical column 0 (`y` = 20,30,40,50) into `x`, so
    /// the exact-value asserts below go RED if `IncrementalAppendScanBuilder::build`
    /// re-hardcodes `name_mapping: None`.
    #[tokio::test]
    async fn test_incremental_append_applies_name_mapping_to_id_less_parquet() {
        let mut fixture = TableTestFixture::new_with_name_mapping_property(NAME_MAPPING_X1_Y2);
        fixture.setup_name_mapping_manifest_files().await;
        let (parent, current) = parent_to_current_range(&fixture.table);

        let scan = fixture
            .table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(parent)
            .to_snapshot_id(current)
            .select(["x", "y"])
            .build()
            .expect("building the incremental scan over a name-mapped table should succeed");

        let batches: Vec<RecordBatch> = ArrowReaderBuilder::new(fixture.table.file_io().clone())
            .build()
            .read(scan.plan_files().await.expect("plan_files should succeed"))
            .expect("reading the incremental task stream should succeed")
            .try_collect()
            .await
            .expect("collecting record batches should succeed");
        assert!(
            !batches.is_empty(),
            "the incremental read must return at least one batch"
        );

        let x = decode_int64_column(
            batches[0]
                .column_by_name("x")
                .expect("batch must have column x"),
        );
        let y = decode_int64_column(
            batches[0]
                .column_by_name("y")
                .expect("batch must have column y"),
        );
        assert_eq!(
            x.values(),
            &[1, 1, 1, 1],
            "x must be the mapped physical x column (all 1s), not the positional y column"
        );
        assert_eq!(
            y.values(),
            &[20, 30, 40, 50],
            "y must be the mapped physical y column"
        );
    }

    /// ABSENT PROPERTY (positional fallback): the same ID-less reversed-parquet shape
    /// WITHOUT `schema.name-mapping.default` plans every task with `name_mapping: None` and
    /// still reads via the POSITIONAL fallback — physical column 0 is assigned field id 1
    /// (= `x`), so `x` carries the physical first column's values (20,30,40,50). Regression
    /// guard for the `None` path: the absent property must neither error nor accidentally
    /// name-map.
    #[tokio::test]
    async fn test_incremental_append_absent_name_mapping_uses_positional_fallback() {
        let mut fixture = TableTestFixture::new_unpartitioned();
        fixture.setup_name_mapping_manifest_files().await;
        let (parent, current) = parent_to_current_range(&fixture.table);

        let scan = fixture
            .table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(parent)
            .to_snapshot_id(current)
            .select(["x", "y"])
            .build()
            .expect("building the incremental scan without a name mapping should succeed");

        let tasks: Vec<FileScanTask> = scan
            .plan_files()
            .await
            .expect("plan_files should succeed")
            .try_collect()
            .await
            .expect("collecting file scan tasks should succeed");
        assert!(
            !tasks.is_empty(),
            "the fixture must produce at least one incremental task"
        );
        for task in &tasks {
            assert!(
                task.name_mapping.is_none(),
                "an absent property must leave name_mapping None on every incremental task"
            );
        }

        // Re-plan and read end-to-end: with no mapping the reader falls back to positional
        // field ids, so the schema's `x` (field id 1) resolves to PHYSICAL column 0 — the
        // reversed file's `y` data. This is the documented fallback semantic (Java
        // `ParquetSchemaUtil.addFallbackIds`), and the value-level CONTRAST with the
        // name-mapped test above proves the property alone flips the outcome.
        let batches: Vec<RecordBatch> = ArrowReaderBuilder::new(fixture.table.file_io().clone())
            .build()
            .read(scan.plan_files().await.expect("plan_files should succeed"))
            .expect("reading the incremental task stream should succeed")
            .try_collect()
            .await
            .expect("collecting record batches should succeed");
        assert!(
            !batches.is_empty(),
            "the incremental fallback read must return at least one batch"
        );

        let x = decode_int64_column(
            batches[0]
                .column_by_name("x")
                .expect("batch must have column x"),
        );
        let y = decode_int64_column(
            batches[0]
                .column_by_name("y")
                .expect("batch must have column y"),
        );
        assert_eq!(
            x.values(),
            &[20, 30, 40, 50],
            "without a mapping, x must be read positionally (physical column 0)"
        );
        assert_eq!(
            y.values(),
            &[1, 1, 1, 1],
            "without a mapping, y must be read positionally (physical column 1)"
        );
    }

    // ===================================================================================
    // IncrementalChangelogScan tests
    // ===================================================================================

    use super::super::{ChangelogOperation, ChangelogScanTask, ChangelogTaskKind};

    /// A position-delete file routed to partition `x = part_value`, shaped as a DELETION
    /// VECTOR (Puffin format + `referenced_data_file` pointing at `referenced_path` + blob
    /// coordinates), used to create a DELETE-content manifest in the range (NOT a real
    /// puffin file — manifest-only). A DV rather than a parquet position delete because the
    /// fixture table is V3 and the D3 format-version gate rejects parquet position deletes
    /// on V3 ("Must use DVs for position deletes in V3") — the tests' subject (DELETE
    /// manifests in the changelog range) is content-format-agnostic.
    fn synthetic_position_delete_file(
        path: &str,
        referenced_path: &str,
        part_value: i64,
    ) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .referenced_data_file(Some(referenced_path.to_string()))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .build()
            .unwrap()
    }

    /// An equality-delete file routed to partition `x = part_value` (equality field: `x`,
    /// field id 1 in the V3 minimal fixture schema), used to pin the added-vs-preexisting
    /// delete split — an eq delete is PARTITION-scoped (unlike the path-scoped DV), so two
    /// of them on different snapshots both apply to the same data file.
    fn synthetic_equality_delete_file(path: &str, part_value: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .equality_ids(Some(vec![1]))
            .build()
            .unwrap()
    }

    /// Collect the changelog tasks a scan plans (path, operation, ordinal, commit id).
    async fn changelog_tasks(scan: &super::IncrementalChangelogScan) -> Vec<ChangelogScanTask> {
        scan.plan_files()
            .await
            .expect("plan_files should succeed")
            .try_collect()
            .await
            .expect("collecting changelog tasks should succeed")
    }

    /// Index the changelog tasks by data-file path for per-file assertions.
    fn by_path(tasks: &[ChangelogScanTask]) -> HashMap<String, &ChangelogScanTask> {
        tasks
            .iter()
            .map(|task| (task.data_file_path().to_string(), task))
            .collect()
    }

    /// CORE: a range with 2 APPEND snapshots yields INSERT tasks with the OLDEST snapshot's
    /// files at ordinal 0 and the next at ordinal 1, each `commit_snapshot_id` = the adding
    /// snapshot. Mutation-pins the ordinal scheme (oldest → 0) and the commit-id stamping:
    /// reversing the ordinal order (newest = 0) flips both files' ordinals and fails this.
    #[tokio::test]
    async fn test_changelog_two_appends_assigns_ordinals_oldest_first() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("base.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("a.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("b.parquet", 1)]).await;
        let s2 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s2)
            .build()
            .unwrap();

        let tasks = changelog_tasks(&scan).await;
        let by_path = by_path(&tasks);
        assert_eq!(
            by_path.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from(["a.parquet".to_string(), "b.parquet".to_string()]),
            "only the two appends after S0 are in the changelog"
        );

        let task_a = by_path["a.parquet"];
        assert_eq!(task_a.operation(), ChangelogOperation::Insert);
        assert_eq!(
            task_a.change_ordinal(),
            0,
            "S1 is the oldest in range → ordinal 0"
        );
        assert_eq!(task_a.commit_snapshot_id(), s1);

        let task_b = by_path["b.parquet"];
        assert_eq!(task_b.operation(), ChangelogOperation::Insert);
        assert_eq!(task_b.change_ordinal(), 1, "S2 follows S1 → ordinal 1");
        assert_eq!(task_b.commit_snapshot_id(), s2);
    }

    /// DELETE OPERATION: a snapshot that REMOVES a live data file (an overwrite writing a
    /// `Deleted` manifest ENTRY into its own rewritten manifest, NOT a delete-FILE manifest)
    /// produces a DELETE changelog task for that file, with the added file as an INSERT in
    /// the same snapshot. Mutation-pins the Deleted→Delete mapping: mapping Deleted→Insert
    /// makes the removed file's task assert-fail on operation.
    #[tokio::test]
    async fn test_changelog_overwrite_emits_delete_for_removed_file() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        // S0 appends a + b; S1 overwrites: delete a, add c.
        let table = append_files(&catalog, &table, vec![
            data_file("a.parquet", 1),
            data_file("b.parquet", 1),
        ])
        .await;
        let s0 = table.metadata().current_snapshot_id().unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("a.parquet")
            .add_file(data_file("c.parquet", 1));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .build()
            .unwrap();

        let tasks = changelog_tasks(&scan).await;
        let by_path = by_path(&tasks);

        // The overwrite removed `a` (DELETE) and added `c` (INSERT); `b` is untouched and
        // not part of this range's changelog.
        assert_eq!(
            by_path.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from(["a.parquet".to_string(), "c.parquet".to_string()]),
            "the overwrite's removed (a) + added (c) files are the changelog; b is untouched"
        );

        let deleted = by_path["a.parquet"];
        assert_eq!(
            deleted.operation(),
            ChangelogOperation::Delete,
            "the removed file a is a DELETE change"
        );
        assert_eq!(
            deleted.kind(),
            ChangelogTaskKind::DeletedDataFile,
            "a whole-file removal is Java's DeletedDataFileScanTask"
        );
        assert_eq!(deleted.commit_snapshot_id(), s1);
        assert_eq!(deleted.change_ordinal(), 0);

        let added = by_path["c.parquet"];
        assert_eq!(
            added.operation(),
            ChangelogOperation::Insert,
            "the added file c is an INSERT change"
        );
        assert_eq!(
            added.kind(),
            ChangelogTaskKind::AddedRows,
            "a whole-file addition is Java's AddedRowsScanTask"
        );
        assert_eq!(added.commit_snapshot_id(), s1);
    }

    /// REPLACE EXCLUSION: a `RewriteFiles` (compaction) snapshot in the range produces an
    /// `Operation::Replace` snapshot, which the changelog scan EXCLUDES (it rewrites files
    /// without changing rows). After S0 appends a + b, S1 rewrites {a,b}→{c}. The (S0, S1]
    /// changelog is EMPTY. Mutation-pins the Replace exclusion: including Replace snapshots
    /// would surface c (and a/b as deletes), making the changelog non-empty.
    #[tokio::test]
    async fn test_changelog_excludes_replace_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let file_a = data_file("a.parquet", 1);
        let file_b = data_file("b.parquet", 1);
        let table = append_files(&catalog, &table, vec![file_a.clone(), file_b.clone()]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();

        // Compaction: replace {a, b} with {c} → Operation::Replace.
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![file_a, file_b], vec![data_file("c.parquet", 1)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s1 = table.metadata().current_snapshot_id().unwrap();

        assert_eq!(
            table
                .metadata()
                .snapshot_by_id(s1)
                .unwrap()
                .summary()
                .operation,
            Operation::Replace,
            "the rewrite must be a Replace for this test to be meaningful"
        );

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .build()
            .unwrap();

        let tasks = changelog_tasks(&scan).await;
        assert!(
            tasks.is_empty(),
            "a Replace (compaction) snapshot contributes no changelog tasks, got: {:?}",
            tasks.iter().map(|t| t.data_file_path()).collect::<Vec<_>>()
        );
    }

    /// DELETE-MANIFEST GUARD: a range whose snapshots reference a row-level DELETE manifest
    /// (here a `RowDelta` adding a position-delete file) is rejected with
    /// `FeatureUnsupported` — matching Java's current data-file-changelog limitation.
    /// Mutation-pins the guard: dropping the delete-manifest check lets `plan_files` proceed
    /// (returning Ok), failing this assertion.
    #[tokio::test]
    async fn test_changelog_rejects_range_with_delete_manifest() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("a.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();

        // S1 adds a position-delete file → its manifest list carries a DELETE manifest.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_position_delete_file(
                "a-pos-del.puffin",
                "a.parquet",
                1,
            )]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .build()
            .unwrap();

        let result = scan.plan_files().await;
        let error = result
            .err()
            .expect("a range with a delete manifest must error");
        assert_eq!(
            error.kind(),
            ErrorKind::FeatureUnsupported,
            "a delete-manifest range is FeatureUnsupported, got: {error}"
        );
    }

    /// FILTER PRUNES BY PARTITION: a `with_filter(x == 10)` over the identity(x)-partitioned
    /// table prunes the appended file in partition x = 20 from the changelog, keeping x = 10.
    #[tokio::test]
    async fn test_changelog_with_filter_prunes_by_partition() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("base.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![
            data_file("x10.parquet", 10),
            data_file("x20.parquet", 20),
        ])
        .await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .with_filter(Reference::new("x").equal_to(Datum::long(10)))
            .build()
            .unwrap();

        let tasks = changelog_tasks(&scan).await;
        let paths: HashSet<String> = tasks
            .iter()
            .map(|task| task.data_file_path().to_string())
            .collect();
        assert_eq!(
            paths,
            HashSet::from(["x10.parquet".to_string()]),
            "filter x == 10 must prune the x = 20 appended file from the changelog"
        );
    }

    /// FROM == TO (inclusive): an inclusive `from` equal to `to` resolves to the range
    /// `(to's parent, to]`, which is just `to` itself — its own changes. But `from == to`
    /// EXCLUSIVE is the natural empty case; here we assert the explicit empty range via a
    /// `to` that has no changes after an identical exclusive `from`. We use the
    /// Java-reachable empty case: `from == to` exclusive resolves to an empty changelog.
    #[tokio::test]
    async fn test_changelog_from_equals_to_inclusive_is_only_to_change() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        // Inclusive from == to == S1: the range is just S1's own change (s1.parquet).
        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_inclusive(s1)
            .to_snapshot_id(s1)
            .build()
            .unwrap();

        let tasks = changelog_tasks(&scan).await;
        let paths: HashSet<String> = tasks
            .iter()
            .map(|task| task.data_file_path().to_string())
            .collect();
        assert_eq!(
            paths,
            HashSet::from(["s1.parquet".to_string()]),
            "inclusive from == to returns only that snapshot's own change"
        );
        assert_eq!(
            tasks[0].change_ordinal(),
            0,
            "the single snapshot is ordinal 0"
        );
    }

    /// EMPTY RANGE: `from == to` EXCLUSIVE is rejected by the underlying append scan's
    /// `isParentAncestorOf` precondition (a snapshot is not its own parent ancestor), so the
    /// builder errors — Java-faithful. (The reachable runtime-empty case is the Replace-only
    /// range above.)
    #[tokio::test]
    async fn test_changelog_from_equals_to_exclusive_is_rejected() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let result = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s1)
            .to_snapshot_id(s1)
            .build();
        assert!(
            result.is_err(),
            "from == to (exclusive) must be rejected: a snapshot is not its own parent ancestor"
        );
    }

    /// CARRIED-FORWARD ENTRY (ordinal correctness across snapshots): a file appended in an
    /// OLD snapshot must NOT re-appear in a LATER snapshot's changelog. After S0 appends a,
    /// S1 appends b: the (S0, S1] changelog must contain ONLY b at ordinal 0 — a was added
    /// before the range. Pins that only a snapshot's OWN added manifests are read (a's
    /// manifest is carried forward into S1's list but belongs to S0).
    #[tokio::test]
    async fn test_changelog_only_reads_snapshots_own_added_manifests() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("a.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("b.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .build()
            .unwrap();

        let tasks = changelog_tasks(&scan).await;
        assert_eq!(
            tasks.len(),
            1,
            "only S1's own added file b is in the (S0, S1] changelog"
        );
        assert_eq!(tasks[0].data_file_path(), "b.parquet");
        assert_eq!(tasks[0].operation(), ChangelogOperation::Insert);
        assert_eq!(tasks[0].change_ordinal(), 0);
        assert_eq!(tasks[0].commit_snapshot_id(), s1);
    }

    // ===================================================================================
    // ENGINE-FIRST row-level changelog tests (`with_row_level_deletes(true)`)
    // ===================================================================================

    /// One comparable row per changelog task: `(data-file path, kind token, operation
    /// token, ordinal, commit snapshot, added-delete paths, existing-delete paths,
    /// embedded FileScanTask delete paths)` — sortable so whole plans compare as sets.
    type TaskTuple = (
        String,
        &'static str,
        &'static str,
        i32,
        i64,
        Vec<String>,
        Vec<String>,
        Vec<String>,
    );

    fn task_tuple(task: &ChangelogScanTask) -> TaskTuple {
        let kind_token = match task.kind() {
            ChangelogTaskKind::AddedRows => "ADDED_ROWS",
            ChangelogTaskKind::DeletedDataFile => "DELETED_DATA_FILE",
            ChangelogTaskKind::DeletedRows => "DELETED_ROWS",
        };
        let op_token = match task.operation() {
            ChangelogOperation::Insert => "INSERT",
            ChangelogOperation::Delete => "DELETE",
            ChangelogOperation::UpdateBefore => "UPDATE_BEFORE",
            ChangelogOperation::UpdateAfter => "UPDATE_AFTER",
        };
        (
            task.data_file_path().to_string(),
            kind_token,
            op_token,
            task.change_ordinal(),
            task.commit_snapshot_id(),
            task.added_deletes()
                .iter()
                .map(|d| d.file_path.clone())
                .collect(),
            task.existing_deletes()
                .iter()
                .map(|d| d.file_path.clone())
                .collect(),
            task.file_scan_task()
                .deletes
                .iter()
                .map(|d| d.file_path.clone())
                .collect(),
        )
    }

    fn sorted_task_tuples(tasks: &[ChangelogScanTask]) -> Vec<TaskTuple> {
        let mut tuples: Vec<TaskTuple> = tasks.iter().map(task_tuple).collect();
        tuples.sort();
        tuples
    }

    /// CROWN JEWEL — the Java `DeletedDataFileScanTask` javadoc chain, end to end.
    /// S0 appends a+b, S1 appends c, S2 is a merge-on-read DELETE (`RowDelta` adding a DV
    /// that references the EXISTING file a), S3 overwrites (removes a, adds d). The
    /// row-level changelog over (S0, S3] must emit the Java task taxonomy:
    ///
    /// - c → AddedRows/INSERT @ ordinal 0, commit S1, no deletes;
    /// - a → DeletedRows/DELETE @ ordinal 1, commit S2 (the snapshot that ADDED the
    ///   delete, NOT the one that added a), carrying EXACTLY the DV added in S2 as
    ///   `added_deletes` and nothing pre-existing;
    /// - a → DeletedDataFile/DELETE @ ordinal 2, commit S3, carrying the DV as
    ///   `existing_deletes` (rows already deleted must not re-surface as deleted);
    /// - d → AddedRows/INSERT @ ordinal 2, commit S3;
    /// - b → NO task (the DV is path-scoped to a; b was never touched).
    ///
    /// Risks pinned: MoR-delete snapshots plan instead of rejecting (only with the
    /// opt-in flag), the deleted-rows task carries ONLY the range snapshot's ADDED
    /// deletes, commit ordinals stay oldest→0 across mixed snapshot types (a delete-only
    /// snapshot consumes its own ordinal), and the DEFAULT mode still REJECTS this exact
    /// range (the Java 1.10.0 rejection surface is unchanged by the feature).
    #[tokio::test]
    async fn test_changelog_row_level_merge_on_read_chain_emits_java_taxonomy_tasks() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![
            data_file("a.parquet", 1),
            data_file("b.parquet", 1),
        ])
        .await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("c.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        // S2: merge-on-read DELETE — a DV referencing the EXISTING file a.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_position_delete_file(
                "a-dv.puffin",
                "a.parquet",
                1,
            )]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s2 = table.metadata().current_snapshot_id().unwrap();

        // S3: overwrite — remove a outright, add d.
        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("a.parquet")
            .add_file(data_file("d.parquet", 1));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s3 = table.metadata().current_snapshot_id().unwrap();

        // The DEFAULT mode must still REJECT this range — the Java 1.10.0 rejection
        // surface does not vanish because the opt-in exists.
        let default_scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s3)
            .build()
            .unwrap();
        let default_error = default_scan
            .plan_files()
            .await
            .err()
            .expect("the default data-file changelog must reject a MoR-delete range");
        assert_eq!(
            default_error.kind(),
            ErrorKind::FeatureUnsupported,
            "default-mode rejection classification must stay FeatureUnsupported"
        );

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s3)
            .with_row_level_deletes(true)
            .build()
            .unwrap();
        let tasks = changelog_tasks(&scan).await;

        let dv = vec!["a-dv.puffin".to_string()];
        let expected: Vec<TaskTuple> = {
            let mut expected = vec![
                (
                    "c.parquet".to_string(),
                    "ADDED_ROWS",
                    "INSERT",
                    0,
                    s1,
                    vec![],
                    vec![],
                    vec![],
                ),
                (
                    "a.parquet".to_string(),
                    "DELETED_ROWS",
                    "DELETE",
                    1,
                    s2,
                    dv.clone(),
                    vec![],
                    vec![],
                ),
                (
                    "a.parquet".to_string(),
                    "DELETED_DATA_FILE",
                    "DELETE",
                    2,
                    s3,
                    vec![],
                    dv.clone(),
                    dv.clone(),
                ),
                (
                    "d.parquet".to_string(),
                    "ADDED_ROWS",
                    "INSERT",
                    2,
                    s3,
                    vec![],
                    vec![],
                    vec![],
                ),
            ];
            expected.sort();
            expected
        };

        assert_eq!(
            sorted_task_tuples(&tasks),
            expected,
            "the row-level changelog must emit exactly the Java taxonomy task split"
        );
    }

    /// ADDED-vs-PREEXISTING SPLIT: a DeletedRows task carries ONLY the deletes ADDED by
    /// its commit snapshot in `added_deletes`; a delete from an EARLIER snapshot lands in
    /// `existing_deletes` (Java `DeletedRowsScanTask.addedDeletes()` vs
    /// `existingDeletes()` — "records removed by [existing deletes] should not appear in
    /// the changelog"). S0 appends a; S1 adds eq-delete E1; S2 adds eq-delete E2 (both
    /// partition-scoped, so both apply to a). The (S1, S2] changelog holds ONE
    /// DeletedRows task for a with added = [E2] and existing = [E1]. Mutation-pins the
    /// split: routing pre-existing deletes into `added_deletes` (or dropping the
    /// existing-index query) fails the exact-list assertions.
    #[tokio::test]
    async fn test_changelog_row_level_deleted_rows_splits_added_vs_preexisting_deletes() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("a.parquet", 1)]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_equality_delete_file("e1-eq-del.parquet", 1)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![synthetic_equality_delete_file("e2-eq-del.parquet", 1)]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s2 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s1)
            .to_snapshot_id(s2)
            .with_row_level_deletes(true)
            .build()
            .unwrap();
        let tasks = changelog_tasks(&scan).await;

        assert_eq!(
            sorted_task_tuples(&tasks),
            vec![(
                "a.parquet".to_string(),
                "DELETED_ROWS",
                "DELETE",
                0,
                s2,
                vec!["e2-eq-del.parquet".to_string()],
                vec!["e1-eq-del.parquet".to_string()],
                // The embedded FileScanTask applies the EXISTING deletes (rows already
                // deleted before S2 must not re-surface); the added deletes are the
                // engine's selector, carried separately.
                vec!["e1-eq-del.parquet".to_string()],
            )],
            "added_deletes must carry ONLY the S2-added delete; the S1 delete is existing"
        );
    }

    /// SAME-SNAPSHOT FOLD: deletes committed in the SAME snapshot as the data file they
    /// match fold into that file's AddedRows task (Java `AddedRowsScanTask` javadoc:
    /// `AddedRowsScanTask(file=F1, deletes=[D1], snapshot=S1)`) — they must NOT also
    /// produce a spurious DeletedRows task for the file. Mutation-pins the
    /// added-by-this-snapshot exclusion in the DeletedRows candidate walk: dropping it
    /// emits a second task for f and fails the exact-plan assertion.
    #[tokio::test]
    async fn test_changelog_row_level_same_snapshot_deletes_fold_into_added_rows_task() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("base.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();

        // S1: one RowDelta committing BOTH the data file f AND a DV that references it.
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_data_files(vec![data_file("f.parquet", 1)])
            .add_deletes(vec![synthetic_position_delete_file(
                "f-dv.puffin",
                "f.parquet",
                1,
            )]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .with_row_level_deletes(true)
            .build()
            .unwrap();
        let tasks = changelog_tasks(&scan).await;

        assert_eq!(
            sorted_task_tuples(&tasks),
            vec![(
                "f.parquet".to_string(),
                "ADDED_ROWS",
                "INSERT",
                0,
                s1,
                vec!["f-dv.puffin".to_string()],
                vec![],
                // Reading the AddedRows task applies the folded deletes → NET added rows.
                vec!["f-dv.puffin".to_string()],
            )],
            "the same-snapshot DV folds into f's AddedRows task; no DeletedRows task, \
             and base.parquet (untouched by the path-scoped DV) contributes nothing"
        );
    }

    /// CONTROL — pure-append output is IDENTICAL with the row-level flag on or off: the
    /// engine-first mode must not perturb the interop-pinned data-file changelog for
    /// ranges without delete files. Mutation-pins the flag's blast radius: any change to
    /// the append planning path (ordinals, commit ids, task kinds, delete attachments)
    /// breaks the tuple-for-tuple equality.
    #[tokio::test]
    async fn test_changelog_row_level_flag_on_pure_append_range_matches_default_output() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("base.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("a.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("b.parquet", 1)]).await;
        let s2 = table.metadata().current_snapshot_id().unwrap();

        let default_scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s2)
            .build()
            .unwrap();
        let row_level_scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s2)
            .with_row_level_deletes(true)
            .build()
            .unwrap();

        let default_tuples = sorted_task_tuples(&changelog_tasks(&default_scan).await);
        let row_level_tuples = sorted_task_tuples(&changelog_tasks(&row_level_scan).await);

        assert_eq!(
            default_tuples.len(),
            2,
            "the control range must actually plan the two appended files"
        );
        assert_eq!(
            default_tuples, row_level_tuples,
            "a pure-append range must plan identically with the row-level flag on"
        );
        assert!(
            row_level_tuples
                .iter()
                .all(|(_, kind, op, _, _, added, existing, task_deletes)| {
                    *kind == "ADDED_ROWS"
                        && *op == "INSERT"
                        && added.is_empty()
                        && existing.is_empty()
                        && task_deletes.is_empty()
                }),
            "pure appends are AddedRows/INSERT tasks with no delete attachments"
        );
    }

    /// ORDINALS SKIP EXCLUDED SNAPSHOTS: Java assigns change ordinals over the FILTERED
    /// snapshot deque (`computeSnapshotOrdinals` runs AFTER `orderedChangelogSnapshots`
    /// dropped REPLACE snapshots), so a compaction between two changes does NOT consume
    /// an ordinal. S0 appends a; S1 compacts (Replace, excluded); S2 appends e. The
    /// (S0, S2] changelog holds only e at ordinal 0 — NOT 1. Mutation-pins assigning
    /// ordinals before the Replace exclusion.
    #[tokio::test]
    async fn test_changelog_replace_snapshot_consumes_no_ordinal() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let file_a = data_file("a.parquet", 1);
        let table = append_files(&catalog, &table, vec![file_a.clone()]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();

        // S1: compaction — Operation::Replace, excluded from the changelog.
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files(vec![file_a], vec![data_file("c.parquet", 1)]);
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
            "the rewrite must be a Replace for this test to be meaningful"
        );

        let table = append_files(&catalog, &table, vec![data_file("e.parquet", 1)]).await;
        let s2 = table.metadata().current_snapshot_id().unwrap();

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s2)
            .build()
            .unwrap();
        let tasks = changelog_tasks(&scan).await;

        assert_eq!(tasks.len(), 1, "only the S2 append is in the changelog");
        assert_eq!(tasks[0].data_file_path(), "e.parquet");
        assert_eq!(
            tasks[0].change_ordinal(),
            0,
            "ordinals are assigned over the FILTERED snapshots — the excluded Replace \
             consumes no ordinal"
        );
        assert_eq!(tasks[0].commit_snapshot_id(), s2);
    }

    // ===== Event listeners: a REAL incremental scan genuinely fires an `IncrementalScanEvent` =====

    struct IncEventRecorder {
        sink: std::sync::Arc<std::sync::Mutex<Vec<crate::events::IncrementalScanEvent>>>,
    }
    impl crate::events::Listener<crate::events::IncrementalScanEvent> for IncEventRecorder {
        fn notify(&self, event: &crate::events::IncrementalScanEvent) {
            self.sink.lock().unwrap().push(event.clone());
        }
    }

    /// Risk: the incremental emit is wired but never fires, or resolves the `from` bound wrong.
    /// Pins that a REAL incremental APPEND scan with an EXPLICIT exclusive `from` fires one
    /// `IncrementalScanEvent` with `from = that id`, `inclusive = false`, the inclusive `to`, and
    /// the table name.
    #[tokio::test]
    async fn test_real_incremental_append_from_present_fires_exclusive_event() {
        let _guard = crate::events::test_support::lock();

        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        crate::events::register::<crate::events::IncrementalScanEvent>(std::sync::Arc::new(
            IncEventRecorder { sink: sink.clone() },
        ));

        let scan = table
            .incremental_append_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .build()
            .unwrap();
        let _paths = planned_paths(&scan).await;

        let events = sink.lock().unwrap();
        assert_eq!(events.len(), 1, "one IncrementalScanEvent per plan");
        let event = &events[0];
        assert_eq!(event.from_snapshot_id(), s0);
        assert_eq!(event.to_snapshot_id(), s1);
        assert!(
            !event.is_from_snapshot_inclusive(),
            "an explicit exclusive `from` is NOT inclusive"
        );
        assert_eq!(event.table_name(), &table.identifier().to_string());
    }

    /// Risk: an ABSENT `from` resolves to the wrong lower bound or the wrong inclusive flag.
    /// Java: absent `from` → `(oldestAncestorOf(to), inclusive = true)`. Pins `from` is the
    /// OLDEST ancestor (the history root S0) and `inclusive = true`.
    #[tokio::test]
    async fn test_real_incremental_append_from_absent_fires_oldest_ancestor_inclusive() {
        let _guard = crate::events::test_support::lock();

        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        crate::events::register::<crate::events::IncrementalScanEvent>(std::sync::Arc::new(
            IncEventRecorder { sink: sink.clone() },
        ));

        // No `from`: the whole lineage of `to`.
        let scan = table
            .incremental_append_scan()
            .to_snapshot_id(s1)
            .build()
            .unwrap();
        let _paths = planned_paths(&scan).await;

        let events = sink.lock().unwrap();
        assert_eq!(events.len(), 1);
        let event = &events[0];
        assert_eq!(
            event.from_snapshot_id(),
            s0,
            "absent `from` resolves to the oldest ancestor (history root)"
        );
        assert_eq!(event.to_snapshot_id(), s1);
        assert!(
            event.is_from_snapshot_inclusive(),
            "absent `from` is inclusive of the oldest ancestor"
        );
    }

    /// Risk: the CHANGELOG scan does NOT fire the `IncrementalScanEvent` even though Java fires
    /// it from the SHARED `BaseIncrementalScan.planFiles` (so both planners fire). Pins that a
    /// real changelog scan fires the event too.
    #[tokio::test]
    async fn test_real_changelog_scan_fires_incremental_event() {
        let _guard = crate::events::test_support::lock();

        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;
        let table = append_files(&catalog, &table, vec![data_file("a.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("b.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

        let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        crate::events::register::<crate::events::IncrementalScanEvent>(std::sync::Arc::new(
            IncEventRecorder { sink: sink.clone() },
        ));

        let scan = table
            .incremental_changelog_scan()
            .from_snapshot_id_exclusive(s0)
            .to_snapshot_id(s1)
            .build()
            .unwrap();
        let _tasks = changelog_tasks(&scan).await;

        let events = sink.lock().unwrap();
        assert_eq!(
            events.len(),
            1,
            "the changelog scan fires the shared IncrementalScanEvent too"
        );
        assert_eq!(events[0].from_snapshot_id(), s0);
        assert_eq!(events[0].to_snapshot_id(), s1);
        assert!(!events[0].is_from_snapshot_inclusive());
    }
}
