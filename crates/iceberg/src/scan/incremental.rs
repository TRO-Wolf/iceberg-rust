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
//! Two planners over the range `(from_snapshot_id exclusive, to_snapshot_id inclusive]`.
//! Both are separate from [`TableScan`](super::TableScan). Both reuse [`PlanContext`] and
//! [`ManifestEntryContext`].
//!
//! | Scan | Returns | Java mirror |
//! |---|---|---|
//! | [`IncrementalAppendScan`] | the data files `APPEND` snapshots added; no deletes applied | `BaseIncrementalAppendScan` |
//! | [`IncrementalChangelogScan`] | row-level [`ChangelogScanTask`]s, one per file added or removed, tagged with a change ordinal (oldest snapshot = 0); excludes `Operation::Replace` | `BaseIncrementalChangelogScan` |
//!
//! The changelog scan rejects a range whose snapshots carry a DELETE manifest, as Java
//! 1.10.0 does. [`with_row_level_deletes`](IncrementalChangelogScanBuilder::with_row_level_deletes)
//! is an ENGINE-FIRST opt-in that accepts such a range. Net-change
//! UPDATE_BEFORE/UPDATE_AFTER pairing stays engine-side.

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
/// Mirrors Java `IncrementalAppendScan` and `BaseIncrementalScan`. The `to` snapshot
/// defaults to the table's current snapshot.
pub struct IncrementalAppendScanBuilder<'a> {
    table: &'a Table,
    column_names: Option<Vec<String>>,
    /// `None` means every ancestor of `to`.
    from_snapshot_id_exclusive: Option<i64>,
    /// `build()` converts this inclusive bound into the exclusive bound: the parent.
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

    /// Sets the EXCLUSIVE `from` snapshot id, Java `fromSnapshotExclusive`. It supersedes
    /// an inclusive bound set before.
    pub fn from_snapshot_id_exclusive(mut self, from_snapshot_id: i64) -> Self {
        self.from_snapshot_id_exclusive = Some(from_snapshot_id);
        self.from_snapshot_id_inclusive = None;
        self
    }

    /// Sets the INCLUSIVE `from` snapshot id, Java `fromSnapshotInclusive`. It supersedes
    /// an exclusive bound set before.
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
        // The manifest evaluator rejects a Not node, so normalize it away here.
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
    /// # Errors
    ///
    /// Fails when `to` or an explicit `from` is absent, or `from` is not an ancestor of `to`.
    pub fn build(self) -> Result<IncrementalAppendScan> {
        let metadata = self.table.metadata();

        // The plan-time `IncrementalScanEvent` needs the name, Java `table().name()`.
        let table_name = self.table.identifier().to_string();

        // Java `toSnapshotIdInclusive()`. No snapshot and no explicit `to` gives an empty scan.
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

        // Java `fromSnapshotIdExclusive(toInclusive)`. An exclusive `from` must be the parent
        // of some ancestor of `to`, so an expired `from` still bounds the range.
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

        // The `to` snapshot supplies the schema, as `TableScanBuilder::build` does.
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
            Some(predicate.bind(schema.clone(), self.case_sensitive)?)
        } else {
            None
        };

        let plan_context = PlanContext {
            // The incremental planner selects manifests across the range itself. It never
            // calls `PlanContext::get_manifest_list`, which reads only this one snapshot.
            snapshot: to_snapshot,
            table_metadata: self.table.metadata_ref(),
            snapshot_schema: schema,
            case_sensitive: self.case_sensitive,
            predicate: self.filter.map(Arc::new),
            snapshot_bound_predicate: snapshot_bound_predicate.map(Arc::new),
            // This surface has no file-prune-only mode.
            apply_residual_filter: true,
            object_cache: self.table.object_cache(),
            field_ids: Arc::new(field_ids),
            // An id-less data file added in the range resolves field ids by column name.
            name_mapping: parse_name_mapping(self.table.metadata())?,
            partition_filter_cache: Arc::new(PartitionFilterCache::new()),
            manifest_evaluator_cache: Arc::new(ManifestEvaluatorCache::new()),
            expression_evaluator_cache: Arc::new(ExpressionEvaluatorCache::new()),
            // Java reports scan metrics on the snapshot scan only.
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

/// An incremental append scan over `(from_snapshot_id exclusive, to_snapshot_id inclusive]`.
///
/// Built via [`Table::incremental_append_scan`](crate::table::Table::incremental_append_scan).
/// [`plan_files`](Self::plan_files) streams one [`FileScanTask`] per data file the APPEND
/// snapshots in the range added.
#[derive(Debug)]
pub struct IncrementalAppendScan {
    /// `None` means the whole current lineage of `to`.
    from_snapshot_id_exclusive: Option<i64>,
    /// `None` only when the table has no snapshots. The scan is then empty.
    to_snapshot_id: Option<i64>,
    /// `None` when the table has no snapshots and no explicit `to` is set.
    plan_context: Option<PlanContext>,
    column_names: Option<Vec<String>>,
    batch_size: Option<usize>,
    file_io: FileIO,
    concurrency_limit_manifest_entries: usize,
    concurrency_limit_manifest_files: usize,
    /// Captured at build time for the [`IncrementalScanEvent`].
    table_name: String,
}

impl IncrementalAppendScan {
    /// Returns a stream of [`FileScanTask`]s for the appended data files in the range.
    ///
    /// Mirrors Java `BaseIncrementalAppendScan.doPlanFiles`. The delete index stays EMPTY:
    /// an append scan applies no deletes. Pruning matches the normal scan.
    pub async fn plan_files(&self) -> Result<FileScanTaskStream> {
        let Some(plan_context) = self.plan_context.as_ref() else {
            return Ok(Box::pin(futures::stream::empty()));
        };
        let Some(to_snapshot_id) = self.to_snapshot_id else {
            return Ok(Box::pin(futures::stream::empty()));
        };

        // It must stay after the guards, so an unresolvable range fires nothing, and before
        // the empty-range check, because Java fires for a valid `to` on an empty range.
        self.notify_incremental_scan_event(plan_context, to_snapshot_id);

        // Java `appendsBetween`.
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
        // Never fed: `build_manifest_file_contexts_from_files` requires a delete sender.
        let (delete_ctx_tx, _delete_ctx_rx) = channel::<ManifestEntryContext>(1);
        let (file_scan_task_tx, file_scan_task_rx) = channel(concurrency_limit_manifest_entries);

        // Dropping the sender resolves the index to "no deletes".
        let (delete_file_idx, delete_file_tx) = DeleteFileIndex::new();
        drop(delete_file_tx);

        // Only the DATA manifests each snapshot itself added. A manifest carried forward
        // from an older snapshot belongs to that snapshot. Its files were not appended here.
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

        // A nested per-entry `spawn` added overhead but no parallelism past the limit.
        spawn(async move {
            let result = manifest_entry_data_ctx_rx
                .map(|me_ctx| Ok((me_ctx, file_scan_task_tx.clone())))
                .try_for_each_concurrent(
                    concurrency_limit_manifest_entries,
                    |(manifest_entry_context, tx)| async move {
                        Self::process_append_manifest_entry(manifest_entry_context, tx).await
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
    /// Java `BaseIncrementalScan.planFiles` resolves the `from` bound two ways. An explicit
    /// exclusive `from` gives `(from, inclusive = false)`. An absent `from` gives
    /// `(oldestAncestorOf(to), inclusive = true)`. The Rust builder keeps one exclusive
    /// `Option<i64>`, so this method RE-RESOLVES the pair.
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

    /// Returns the id of the OLDEST ancestor of `to_snapshot_id`, Java
    /// `SnapshotUtil.oldestAncestorOf`. Falls back to `to_snapshot_id` when `to` is absent.
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
    /// newest-first (Java `appendsBetween`). `None` walks to the history root.
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

    /// Processes one data-manifest entry. Keeps `Added` entries only (Java
    /// `filterManifestEntries(status == ADDED)`), applies the filter, emits a task.
    async fn process_append_manifest_entry(
        manifest_entry_context: ManifestEntryContext,
        mut file_scan_task_tx: Sender<Result<FileScanTask>>,
    ) -> Result<()> {
        // An `Existing` entry was appended before this range. A `Deleted` entry is a removal.
        if manifest_entry_context.manifest_entry.status() != ManifestStatus::Added {
            return Ok(());
        }

        // Unreachable: only DATA manifests are selected. A silent delete-file entry corrupts
        // the result, so fail loudly, as the normal scan does.
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
            );

            if !expression_evaluator.eval(manifest_entry_context.manifest_entry.data_file())? {
                return Ok(());
            }

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

    /// The resolved [`PlanContext`], if the table has snapshots.
    pub(crate) fn plan_context(&self) -> Option<&PlanContext> {
        self.plan_context.as_ref()
    }
}

/// Builder to create an [`IncrementalChangelogScan`].
///
/// Mirrors Java `IncrementalChangelogScan` and `BaseIncrementalScan`. It shares the
/// range resolution of [`IncrementalAppendScanBuilder`]; only `plan_files` differs.
pub struct IncrementalChangelogScanBuilder<'a> {
    table: &'a Table,
    column_names: Option<Vec<String>>,
    from_snapshot_id_exclusive: Option<i64>,
    from_snapshot_id_inclusive: Option<i64>,
    to_snapshot_id: Option<i64>,
    batch_size: Option<usize>,
    case_sensitive: bool,
    filter: Option<Predicate>,
    /// The append scan builder sets the file and entry limits together.
    concurrency_limit: usize,
    /// See [`Self::with_row_level_deletes`]. `false` matches Java 1.10.0.
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

    /// Sets the EXCLUSIVE `from` snapshot id, Java `fromSnapshotExclusive`. It supersedes
    /// an inclusive bound set before.
    pub fn from_snapshot_id_exclusive(mut self, from_snapshot_id: i64) -> Self {
        self.from_snapshot_id_exclusive = Some(from_snapshot_id);
        self.from_snapshot_id_inclusive = None;
        self
    }

    /// Sets the INCLUSIVE `from` snapshot id, Java `fromSnapshotInclusive`. It supersedes
    /// an exclusive bound set before.
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
        // The manifest evaluator rejects a Not node, so normalize it away here.
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

    /// **ENGINE-FIRST (beyond Java 1.10.0 core):** enables row-level changelog planning. The
    /// default `false` mirrors the Java rejection surface: 1.10.0
    /// `BaseIncrementalChangelogScan.orderedChangelogSnapshots` throws for any non-`replace` range
    /// snapshot that carries a delete manifest. `true` accepts such a range and emits the task
    /// taxonomy the Java api defines but 1.10.0 core does not emit.
    pub fn with_row_level_deletes(mut self, include_row_level_deletes: bool) -> Self {
        self.include_row_level_deletes = include_row_level_deletes;
        self
    }

    /// Build the incremental changelog scan.
    ///
    /// Resolves the range as [`IncrementalAppendScanBuilder::build`] does. The changelog
    /// snapshot selection happens later, in [`IncrementalChangelogScan::plan_files`].
    pub fn build(self) -> Result<IncrementalChangelogScan> {
        // The append scan builder resolves the range and the plan context. The two builders
        // share every field, so duplicating the range resolution here would drift.
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
            // `rewrite_not` is idempotent, so the second normalization is safe.
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
/// It streams one [`ChangelogScanTask`] per data file the range adds or removes.
#[derive(Debug)]
pub struct IncrementalChangelogScan {
    /// Carries the range bounds and the `PlanContext`. `plan_files` selects its own snapshots.
    append_scan: IncrementalAppendScan,
    file_io: FileIO,
    /// See [`IncrementalChangelogScanBuilder::with_row_level_deletes`].
    include_row_level_deletes: bool,
}

/// The per-snapshot delete-file indexes the ENGINE-FIRST row-level mode plans against.
struct SnapshotDeleteIndexes {
    /// Delete files this snapshot ADDED (Java `DeletedRowsScanTask.addedDeletes()`).
    added: DeleteFileIndex,
    /// Live delete files that pre-existed this snapshot (Java `existingDeletes()`).
    existing: DeleteFileIndex,
    /// `false` skips the DeletedRows pass, keeping the default plan shape.
    has_added_deletes: bool,
}

impl IncrementalChangelogScan {
    /// Returns a stream of [`ChangelogScanTask`]s for the row-level changes in the range.
    ///
    /// Mirrors Java `BaseIncrementalChangelogScan.doPlanFiles`. It walks the changelog
    /// snapshots oldest-first, gives each a change ordinal, then emits one task per ADDED or
    /// DELETED entry of the DATA manifests that snapshot added.
    pub async fn plan_files(&self) -> Result<ChangelogScanTaskStream> {
        let Some(plan_context) = self.append_scan.plan_context() else {
            return Ok(Box::pin(futures::stream::empty()));
        };
        let Some(to_snapshot_id) = self.append_scan.to_snapshot_id() else {
            return Ok(Box::pin(futures::stream::empty()));
        };

        // Both scans inherit Java's shared `BaseIncrementalScan.planFiles`, so this one fires
        // the event too. The guards above keep an unresolvable range silent.
        self.append_scan
            .notify_incremental_scan_event(plan_context, to_snapshot_id);

        // Oldest-first, excluding Replace, guarding delete manifests.
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

        // The tasks are collected eagerly: a concurrent pipeline makes the one-ordinal-per-
        // snapshot attachment hard to hold, and the range is bounded.
        let mut tasks: Vec<ChangelogScanTask> = Vec::new();
        for (ordinal, snapshot) in changelog_snapshots.iter().enumerate() {
            let change_ordinal = i32::try_from(ordinal).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Too many changelog snapshots in range to assign a change ordinal",
                )
            })?;
            let snapshot_id = snapshot.snapshot_id();

            // One load, split three ways. A manifest this snapshot added carries its
            // whole-file changes; one carried forward is read for ITS snapshot's ordinal. All
            // data manifests are the live-file candidates for DeletedRows tasks.
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

            // Java `DeletedRowsScanTask.addedDeletes()` vs `existingDeletes()`.
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

            // A snapshot that added no delete file plans identically with the flag on or off.
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
    /// OLDEST-FIRST (Java `orderedChangelogSnapshots`). It excludes `Operation::Replace` snapshots.
    /// The caller assigns change ordinals over this order, oldest = 0.
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

            // A Replace snapshot rewrites files without changing rows.
            if snapshot.summary().operation != Operation::Replace {
                // The Java 1.10.0 rejection surface. The row-level mode lifts this guard
                // and plans the row-level task taxonomy instead.
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

    /// Returns whether `snapshot` references any row-level DELETE manifest.
    /// Java `!snapshot.deleteManifests(io).isEmpty()`.
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

    /// Builds the two [`DeleteFileIndex`]es the ENGINE-FIRST row-level mode plans against.
    ///
    /// `added` holds the delete files this snapshot committed, Java `addedDeletes()`.
    /// `existing` holds the live deletes that pre-existed it, Java `existingDeletes()`, which
    /// the engine applies first. A `Deleted` tombstone belongs to neither.
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
                .get_manifest(manifest_file, Some(plan_context.snapshot_schema.clone()))
                .await?;
            for entry in manifest.entries() {
                if entry.status() == ManifestStatus::Deleted {
                    continue;
                }
                // `ManifestEntry::inherit_data` populates `snapshot_id()`; the fallback is
                // defensive.
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

    /// Plans the changelog tasks for ONE snapshot's own added DATA manifests. ADDED entries
    /// become `AddedRows` tasks, DELETED entries become `DeletedDataFile` tasks, and
    /// `Existing` entries are skipped. In the default mode every task carries empty delete
    /// lists, Java 1.10.0 `NO_DELETES`.
    async fn plan_snapshot_change_tasks(
        plan_context: &PlanContext,
        selected_manifests: Vec<ManifestFile>,
        change_ordinal: i32,
        snapshot_id: i64,
        row_level_indexes: Option<(DeleteFileIndex, DeleteFileIndex)>,
    ) -> Result<Vec<ChangelogScanTask>> {
        let manifest_count = selected_manifests.len().max(1);
        let (manifest_entry_data_ctx_tx, manifest_entry_data_ctx_rx) = channel(manifest_count);
        // Never fed: only DATA manifests are passed.
        let (delete_ctx_tx, _delete_ctx_rx) = channel::<ManifestEntryContext>(1);
        // The channel carries `Result` so a producer's fetch error reaches the consumer
        // instead of vanishing when the producer task ends.
        let (task_tx, task_rx) = channel::<Result<ChangelogScanTask>>(manifest_count);

        let (delete_file_idx, delete_file_tx) = DeleteFileIndex::new();
        drop(delete_file_tx);

        let manifest_file_contexts = plan_context.build_manifest_file_contexts_from_files(
            selected_manifests,
            manifest_entry_data_ctx_tx,
            delete_file_idx,
            delete_ctx_tx,
        )?;

        // The producers must run apart from the consumer. A manifest can hold more entries
        // than the channel capacity, so draining after a blocking `send` deadlocks.
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

        // A task, so the conversion interleaves with the producers.
        spawn(async move {
            let result = manifest_entry_data_ctx_rx
                .map(Ok)
                .try_for_each(|manifest_entry_context| {
                    let mut task_tx = task_tx.clone();
                    // Cheap `Arc` clones, moved into the per-entry future.
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

    /// Plans the ENGINE-FIRST [`ChangelogTaskKind::DeletedRows`] tasks for ONE snapshot.
    ///
    /// Every live data file the snapshot's ADDED deletes match yields a task. A file the
    /// snapshot added itself is excluded: its deletes fold into its `AddedRows` task, per the
    /// Java `AddedRowsScanTask` contract. A `Deleted` tombstone is a `DeletedDataFile` change.
    /// `all_data_manifests` must be the whole list; a matching file sits in any manifest.
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

        // One pipeline index cannot hold the added/existing split, so
        // `deleted_rows_task_from_entry` attaches both explicitly.
        let (empty_delete_index, empty_delete_tx) = DeleteFileIndex::new();
        drop(empty_delete_tx);

        let manifest_file_contexts = plan_context.build_manifest_file_contexts_from_files(
            all_data_manifests,
            manifest_entry_data_ctx_tx,
            empty_delete_index,
            delete_ctx_tx,
        )?;

        // The producer/consumer split of `plan_snapshot_change_tasks`. Same deadlock reason.
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

    /// Converts a live data-manifest entry into a [`ChangelogTaskKind::DeletedRows`] task.
    /// Returns `None` when the entry is out of scope.
    async fn deleted_rows_task_from_entry(
        manifest_entry_context: ManifestEntryContext,
        change_ordinal: i32,
        snapshot_id: i64,
        added_index: DeleteFileIndex,
        existing_index: DeleteFileIndex,
    ) -> Result<Option<ChangelogScanTask>> {
        // The tombstoned file is already a `DeletedDataFile` task. It must not surface twice.
        if manifest_entry_context.manifest_entry.status() == ManifestStatus::Deleted {
            return Ok(None);
        }

        // Unreachable: only DATA manifests reach this planner. Fail loudly all the same.
        if manifest_entry_context.manifest_entry.content_type() != DataContentType::Data {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Encountered an entry for a delete file in an incremental changelog scan",
            ));
        }

        // A file this snapshot added folds its matching deletes into its `AddedRows` task,
        // per the Java `AddedRowsScanTask` contract. It never gets a DeletedRows task.
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
        // A plain MoR read of the embedded task gives the rows live BEFORE this change. The
        // engine then uses `added_deletes` to select which of them became deleted.
        file_scan_task.deletes = Arc::from(existing_deletes.clone());

        Ok(Some(ChangelogScanTask {
            change_ordinal,
            // The snapshot that ADDED the deletes committed the change, not the older
            // snapshot that added the data file.
            commit_snapshot_id: snapshot_id,
            kind: ChangelogTaskKind::DeletedRows,
            added_deletes,
            existing_deletes,
            file_scan_task,
        }))
    }

    /// Converts one manifest entry into a [`ChangelogScanTask`]. Returns `None` for an `Existing`
    /// entry or an entry the filter prunes. Mirrors Java `CreateDataFileChangeTasks.apply`.
    async fn changelog_task_from_entry(
        manifest_entry_context: ManifestEntryContext,
        change_ordinal: i32,
        snapshot_id: i64,
        row_level_indexes: Option<(DeleteFileIndex, DeleteFileIndex)>,
    ) -> Result<Option<ChangelogScanTask>> {
        let kind = match manifest_entry_context.manifest_entry.status() {
            ManifestStatus::Added => ChangelogTaskKind::AddedRows,
            ManifestStatus::Deleted => ChangelogTaskKind::DeletedDataFile,
            // An earlier snapshot added it. Java `ignoreExisting()`.
            ManifestStatus::Existing => return Ok(None),
        };

        // Unreachable: only DATA manifests are selected. The append scan guards it too.
        if manifest_entry_context.manifest_entry.content_type() != DataContentType::Data {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Encountered an entry for a delete file in an incremental changelog scan",
            ));
        }

        // A V2/V3 inherited entry id can be absent, so fall back to the snapshot id.
        let commit_snapshot_id = manifest_entry_context
            .manifest_entry
            .snapshot_id()
            .unwrap_or(snapshot_id);

        if !Self::entry_matches_scan_filter(&manifest_entry_context)? {
            return Ok(None);
        }

        let (added_deletes, existing_deletes) = match &row_level_indexes {
            Some((added_index, existing_index)) => {
                let data_file = manifest_entry_context.manifest_entry.data_file();
                let sequence_number = manifest_entry_context.manifest_entry.sequence_number();
                match kind {
                    // A pre-existing delete never applies to a newer file, so this is empty.
                    ChangelogTaskKind::AddedRows => (
                        added_index
                            .get_deletes_for_data_file(data_file, sequence_number)
                            .await?,
                        Vec::new(),
                    ),
                    // Applying these keeps rows already deleted out of the changelog.
                    ChangelogTaskKind::DeletedDataFile => (
                        Vec::new(),
                        existing_index
                            .get_deletes_for_data_file(data_file, sequence_number)
                            .await?,
                    ),
                    // `deleted_rows_task_from_entry` produces every DeletedRows task.
                    ChangelogTaskKind::DeletedRows => (Vec::new(), Vec::new()),
                }
            }
            None => (Vec::new(), Vec::new()),
        };

        let mut file_scan_task = manifest_entry_context.into_file_scan_task().await?;
        if row_level_indexes.is_some() {
            // A plain MoR read then gives the net inserted rows, or the rows live at removal.
            file_scan_task.deletes = Arc::from(match kind {
                ChangelogTaskKind::AddedRows => added_deletes.clone(),
                ChangelogTaskKind::DeletedDataFile | ChangelogTaskKind::DeletedRows => {
                    existing_deletes.clone()
                }
            });
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

    /// Returns whether the entry's data file can match the scan filter, running the same
    /// evaluators as the normal scan. An unfiltered scan gets `true`.
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
            );

            if !expression_evaluator.eval(manifest_entry_context.manifest_entry.data_file())? {
                return Ok(false);
            }

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

/// Returns whether `ancestor_id` is an ancestor of `snapshot_id`, or is `snapshot_id`
/// itself. Java `SnapshotUtil.isAncestorOf`.
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

/// Returns whether some ancestor of `snapshot_id` has `parent_id == parent_ancestor_id`.
/// Java `SnapshotUtil.isParentAncestorOf`. It keeps `(from, to]` defined after `from` expires.
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

    /// Create a V3 table partitioned by identity(x) from `TableMetadataV3ValidMinimal`
    /// (schema `x, y, z` longs, spec id 0). Copied because the `transaction/` helper is private.
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

    /// Build a data file routed to partition `x = part_value`, spec id 0.
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
        tasks
            .into_iter()
            .map(|t| t.data_file_path.to_string())
            .collect()
    }

    /// CORE: from=S0(exclusive) to=S2 returns the S1 and S2 files only, never S0's.
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

    /// EXCLUSIVE-FROM BOUNDARY: from=S1(exclusive) to=S2 returns S2's file only.
    /// Mutation: an inclusive walk adds S1's file.
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

    /// INCLUSIVE-FROM: from=S1(inclusive) to=S2 returns both, so the bound resolved to S0.
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

    /// FROM == TO (exclusive): `isParentAncestorOf` rejects it, as Java does.
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

    /// EMPTY RANGE: the only snapshot in `(from, to]` is an OVERWRITE, so zero tasks plan.
    #[tokio::test]
    async fn test_incremental_append_range_with_no_append_snapshots_is_empty() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

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

    /// APPEND-ONLY: the scan excludes an OVERWRITE snapshot in the range.
    /// Mutation: dropping the `Operation::Append` filter returns the overwrite's added file.
    #[tokio::test]
    async fn test_incremental_append_excludes_overwrite_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

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

    /// FILTER PRUNES BY PARTITION: `with_filter(x == 10)` prunes the x = 20 file.
    #[tokio::test]
    async fn test_incremental_append_with_filter_prunes_by_partition() {
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

    /// DEFAULT to=current: with `to_snapshot_id` unset, from=S0(excl) returns S1 and S2.
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

    /// WHOLE LINEAGE: with no `from`, the scan walks to the root of the current lineage.
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

    /// EMPTY TABLE: a table with no snapshots plans zero tasks.
    #[tokio::test]
    async fn test_incremental_append_empty_table_is_empty() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let scan = table.incremental_append_scan().build().unwrap();
        let paths = planned_paths(&scan).await;
        assert!(paths.is_empty(), "an empty table plans zero tasks");
    }

    /// VALIDATION: a non-ancestor exclusive `from` is rejected.
    #[tokio::test]
    async fn test_incremental_append_rejects_non_ancestor_from() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

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

    /// ADDED-MANIFEST FILTER: S0 appends a, S1 appends b, and (S0, S1] holds b only, though
    /// S1 carries S0's manifest forward.
    /// Mutation: dropping the `added_snapshot_id == snapshot_id` filter returns a too.
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

    /// ADDED-ENTRY FILTER: a snapshot's own added manifest can carry an `Existing` or
    /// `Deleted` entry when the snapshot rewrote a manifest, as Java `MergeAppend` does. The
    /// fast-append fixtures cannot pin this; theirs hold `Added` entries only. This fixture
    /// writes `1.parquet` (Added), `2.parquet` (Deleted) and `3.parquet` (Existing).
    /// Mutation: dropping the `status == Added` filter re-surfaces `3.parquet`.
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
    // `IncrementalAppendScanBuilder::build` is a SEPARATE `parse_name_mapping` site from
    // `TableScanBuilder::build`, whose pins stay green when the incremental parse is dropped.

    /// Resolves `(parent, current]` over a [`TableTestFixture`], selecting the one manifest
    /// its current snapshot added.
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

    /// PLAN-LEVEL NAME MAPPING: with `schema.name-mapping.default` set, every planned task
    /// carries the parsed mapping CONTENT, not merely `is_some`.
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

    /// END-TO-END NAME MAPPING: an ID-less parquet with REVERSED physical column order,
    /// appended inside the range, reads to the correct columns. The stream feeds the
    /// [`ArrowReaderBuilder`] path an engine uses, because there is no `to_arrow` here.
    /// Mutation: `name_mapping: None` reads physical column 0 into `x`, so the asserts go RED.
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

    /// ABSENT PROPERTY: the same reversed parquet WITHOUT `schema.name-mapping.default` reads
    /// by the POSITIONAL fallback. Physical column 0 takes field id 1, so `x` carries
    /// 20,30,40,50. The absent property must neither error nor name-map.
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

        // Java `ParquetSchemaUtil.addFallbackIds`. The value-level contrast with the test
        // above proves the property alone flips the outcome.
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

    // ===== IncrementalChangelogScan tests =====

    use super::super::{ChangelogOperation, ChangelogScanTask, ChangelogTaskKind};

    /// A position-delete file for partition `x = part_value`, shaped as a deletion vector to
    /// pass the V3 format gate. It is manifest-only: no puffin file exists.
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

    /// An equality-delete file for partition `x = part_value`, equality field `x`. An eq
    /// delete is PARTITION-scoped, so two on different snapshots both apply to one data file.
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

    /// CORE: two APPEND snapshots give INSERT tasks at ordinal 0 and 1, oldest first, each
    /// stamped with the adding snapshot. Mutation: newest = 0 flips both ordinals.
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

    /// DELETE OPERATION: a snapshot that removes a live file, by a `Deleted` manifest entry
    /// rather than a delete-FILE manifest, gives a DELETE task and an INSERT for its new file.
    /// Mutation: mapping Deleted to Insert fails the operation assert.
    #[tokio::test]
    async fn test_changelog_overwrite_emits_delete_for_removed_file() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

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

    /// REPLACE EXCLUSION: S0 appends a and b, S1 rewrites them to c, and (S0, S1] is EMPTY.
    /// Mutation: including Replace snapshots surfaces c, plus a and b as deletes.
    #[tokio::test]
    async fn test_changelog_excludes_replace_snapshot() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let file_a = data_file("a.parquet", 1);
        let file_b = data_file("b.parquet", 1);
        let table = append_files(&catalog, &table, vec![file_a.clone(), file_b.clone()]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();

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

    /// DELETE-MANIFEST GUARD: a range referencing a row-level DELETE manifest is rejected
    /// with `FeatureUnsupported`, as Java rejects it.
    /// Mutation: dropping the check lets `plan_files` return Ok.
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

    /// FILTER PRUNES BY PARTITION: `with_filter(x == 10)` drops the x = 20 file from the
    /// changelog and keeps x = 10.
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

    /// FROM == TO (inclusive): the range resolves to `(to's parent, to]`, so the changelog
    /// holds `to`'s own changes.
    #[tokio::test]
    async fn test_changelog_from_equals_to_inclusive_is_only_to_change() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("s0.parquet", 1)]).await;
        let table = append_files(&catalog, &table, vec![data_file("s1.parquet", 1)]).await;
        let s1 = table.metadata().current_snapshot_id().unwrap();

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

    /// EMPTY RANGE: `from == to` EXCLUSIVE fails `isParentAncestorOf`, so the builder errors,
    /// as Java does. The Replace-only range above is the runtime-empty case.
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

    /// CARRIED-FORWARD ENTRY: S0 appends a, S1 appends b, and (S0, S1] holds b at ordinal 0
    /// only. It pins that the scan reads a snapshot's OWN added manifests.
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

    // ===== ENGINE-FIRST row-level changelog tests (`with_row_level_deletes(true)`) =====

    /// One sortable row per task, so whole plans compare as sets: `(path, kind, operation,
    /// ordinal, commit snapshot, added deletes, existing deletes, embedded task deletes)`.
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

    /// The Java `DeletedDataFileScanTask` javadoc chain, end to end. S0 appends a+b, S1
    /// appends c, S2 adds a DV referencing a, S3 removes a and adds d, so (S0, S3] emits:
    ///
    /// | File | Task | Ordinal | Commit | Deletes |
    /// |---|---|---|---|---|
    /// | c | AddedRows | 0 | S1 | none |
    /// | a | DeletedRows | 1 | S2 | added = the S2 DV |
    /// | a | DeletedDataFile | 2 | S3 | existing = the S2 DV |
    /// | d | AddedRows | 2 | S3 | none |
    /// | b | none | | | the DV is path-scoped to a |
    ///
    /// Pins that a MoR-delete snapshot plans under the flag, that a DeletedRows task carries
    /// the ADDED deletes only, and that the DEFAULT mode still rejects this range.
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

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file("a.parquet")
            .add_file(data_file("d.parquet", 1));
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let s3 = table.metadata().current_snapshot_id().unwrap();

        // The opt-in must not remove the Java 1.10.0 rejection surface.
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

    /// ADDED-vs-PREEXISTING SPLIT: a DeletedRows task carries the deletes its commit snapshot
    /// ADDED; an earlier snapshot's delete lands in `existing_deletes`, whose records must not
    /// appear in the changelog. S0 appends a, S1 adds eq-delete E1, S2 adds E2, and (S1, S2]
    /// holds one task with added = [E2] and existing = [E1].
    /// Mutation: routing pre-existing deletes into `added_deletes` fails the exact lists.
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
                // The embedded task applies the EXISTING deletes, so rows deleted before S2
                // do not re-surface. The added deletes are the engine's selector.
                vec!["e1-eq-del.parquet".to_string()],
            )],
            "added_deletes must carry ONLY the S2-added delete; the S1 delete is existing"
        );
    }

    /// SAME-SNAPSHOT FOLD: deletes committed in the same snapshot as the file they match fold
    /// into that file's AddedRows task, and must not also produce a DeletedRows task.
    /// Mutation: dropping the added-by-this-snapshot exclusion emits a second task for f.
    #[tokio::test]
    async fn test_changelog_row_level_same_snapshot_deletes_fold_into_added_rows_task() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let table = append_files(&catalog, &table, vec![data_file("base.parquet", 1)]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();

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
                // Reading the AddedRows task applies the folded deletes, giving net rows.
                vec!["f-dv.puffin".to_string()],
            )],
            "the same-snapshot DV folds into f's AddedRows task; no DeletedRows task, \
             and base.parquet (untouched by the path-scoped DV) contributes nothing"
        );
    }

    /// CONTROL: pure-append output is IDENTICAL with the row-level flag on or off.
    /// Mutation: any change to ordinals, commit ids, task kinds or delete attachments in the
    /// append path breaks the tuple-for-tuple equality.
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

    /// ORDINALS SKIP EXCLUDED SNAPSHOTS: Java assigns ordinals over the FILTERED snapshots,
    /// so the compaction in S1 consumes none and e lands at ordinal 0.
    /// Mutation: assigning ordinals before the Replace exclusion gives e ordinal 1.
    #[tokio::test]
    async fn test_changelog_replace_snapshot_consumes_no_ordinal() {
        let catalog = new_memory_catalog().await;
        let table = make_minimal_table(&catalog).await;

        let file_a = data_file("a.parquet", 1);
        let table = append_files(&catalog, &table, vec![file_a.clone()]).await;
        let s0 = table.metadata().current_snapshot_id().unwrap();

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

    /// Risk: the emit is wired but never fires, or resolves `from` wrong. Pins one
    /// `IncrementalScanEvent` with that id, `inclusive = false`, the `to`, and the name.
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

    /// Risk: an absent `from` resolves to the wrong lower bound or inclusive flag. Java
    /// gives `(oldestAncestorOf(to), inclusive = true)`. Pins the history root S0 and `true`.
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

    /// Risk: the changelog scan does not fire the `IncrementalScanEvent`, though Java fires
    /// it from the shared `BaseIncrementalScan.planFiles`. Pins that it fires.
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
