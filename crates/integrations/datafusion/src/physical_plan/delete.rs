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

//! `DELETE FROM` and `UPDATE` physical plans. One `UInt64` `count` row, per DataFusion's DML contract.
//!
//! | Mode | Writes | Commit |
//! |---|---|---|
//! | `merge-on-read` | position deletes or deletion vectors, plus new rows for UPDATE | `RowDelta` |
//! | `copy-on-write` (default) | rewrite of files that hold a matched row | `OverwriteFiles` |
//!
//! The original DataFusion `WHERE` is the contract. Iceberg pushdown is inexact and would over-delete.
//! Copy-on-write is two-pass: the affected set must be complete before the first survivor is written.
//! Both passes read one frozen snapshot. Conflict filter is `AlwaysTrue`. A zero-match DML commits nothing.
//! | Path | Always validates | Serializable adds |
//! |---|---|---|
//! | copy-on-write DELETE and UPDATE | no conflicting deletes | no conflicting data |
//! | merge-on-read DELETE (V2) | referenced data files exist; skip `Operation::Delete` | no conflicting data files |
//! | merge-on-read DELETE (V3) | every replacement DV reference exists; `validate_deleted_files` (F-17, named Java skip-delete divergence) | no conflicting data files |
//! | merge-on-read UPDATE | files exist, deleted files, no conflicting delete files | no conflicting data files |

use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray, UInt64Array,
};
use datafusion::arrow::compute::filter_record_batch;
use datafusion::arrow::compute::kernels::zip::zip;
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::TryStreamExt;
use iceberg::Catalog;
use iceberg::delete_vector_container::DvContainerClose;
use iceberg::expr::Predicate;
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::scan::ArrowRecordBatchStream;
use iceberg::spec::{DataFile, DataFileFormat, FormatVersion, MetricsConfig, PartitionKey, Struct};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig, position_delete_writer_properties,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};

use super::cow_affected::{live_data_file_partitions, resolve_affected_data_files};
use super::row_lineage::{
    StreamingDataFileWriter, attach_lineage, attach_update_lineage, cow_scan_stream,
    filter_lineage_columns, push_lineage_scan_columns,
};
use super::snapshot_target::{
    maybe_to_branch, maybe_validate_from_snapshot, resolve_scan_snapshot_id,
};
use crate::physical_plan::delete_legacy_merge::write_deletion_vectors;
use crate::to_datafusion_error;

pub(crate) const WRITE_DELETE_MODE: &str = "write.delete.mode";
pub(crate) const WRITE_UPDATE_MODE: &str = "write.update.mode";
const MODE_MERGE_ON_READ: &str = "merge-on-read";

/// The resolved row-level write strategy for a `DELETE` or `UPDATE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteMode {
    /// Write position-delete files (+ for UPDATE, new data files) + commit a `RowDelta`.
    MergeOnRead,
    /// Rewrite affected data files + commit an `OverwriteFiles`.
    CopyOnWrite,
}

impl WriteMode {
    /// Resolves the mode from a table property. Iceberg defaults to copy-on-write.
    pub(crate) fn from_property(table: &Table, property: &str) -> Self {
        match table
            .metadata()
            .properties()
            .get(property)
            .map(String::as_str)
        {
            Some(MODE_MERGE_ON_READ) => WriteMode::MergeOnRead,
            _ => WriteMode::CopyOnWrite,
        }
    }
}

/// The Iceberg row-level isolation-level table properties. Both default to `"serializable"`.
pub(crate) const WRITE_DELETE_ISOLATION_LEVEL: &str = "write.delete.isolation-level";
pub(crate) const WRITE_UPDATE_ISOLATION_LEVEL: &str = "write.update.isolation-level";

/// The isolation level of a row-level write. It picks the §5 validations the commit arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IsolationLevel {
    /// Reject concurrent conflicting DATA and concurrent conflicting DELETES.
    Serializable,
    /// Reject only concurrent conflicting DELETES; concurrent inserts are tolerated.
    Snapshot,
}

impl IsolationLevel {
    /// Parses a level name case-insensitively. An unknown name fails LOUD, never defaulted.
    pub(crate) fn parse(name: &str) -> DFResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "serializable" => Ok(IsolationLevel::Serializable),
            "snapshot" => Ok(IsolationLevel::Snapshot),
            _ => Err(DataFusionError::Plan(format!(
                "Invalid isolation level: {name}"
            ))),
        }
    }

    /// Resolves the isolation level from the table property, defaulting to serializable as Java
    /// does. Resolution happens at PLAN time, so an invalid value fails before any scan or write.
    pub(crate) fn for_row_level_op(table: &Table, property: &str) -> DFResult<Self> {
        match table.metadata().properties().get(property) {
            Some(name) => Self::parse(name),
            None => Ok(IsolationLevel::Serializable),
        }
    }
}

/// `DELETE FROM` plan. It finds the matching rows, writes the deletes, commits, and counts.
pub(crate) struct IcebergDeleteExec {
    table: Table,
    catalog: Arc<dyn Catalog>,
    /// The EXACT row filter, or `None` to delete every row (`DELETE FROM t`).
    predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Iceberg file prune only. Never replaces [`predicate`].
    prune: Option<Predicate>,
    mode: WriteMode,
    /// The §5 isolation level, resolved at plan time from `write.delete.isolation-level`.
    isolation: IsolationLevel,
    /// The scan's projection base, and the schema the `predicate` is bound to.
    table_schema: SchemaRef,
    count_schema: SchemaRef,
    plan_properties: Arc<PlanProperties>,
    commit_branch: Option<String>,
}

impl IcebergDeleteExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        table: Table,
        catalog: Arc<dyn Catalog>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        prune: Option<Predicate>,
        mode: WriteMode,
        isolation: IsolationLevel,
        table_schema: SchemaRef,
        commit_branch: Option<String>,
    ) -> Self {
        let count_schema = Self::make_count_schema();
        let plan_properties = Self::compute_properties(Arc::clone(&count_schema));
        Self {
            table,
            catalog,
            predicate,
            prune,
            mode,
            isolation,
            table_schema,
            count_schema,
            plan_properties,
            commit_branch,
        }
    }

    pub(crate) fn compute_properties(schema: SchemaRef) -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ))
    }

    pub(crate) fn make_count_schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![Field::new(
            "count",
            DataType::UInt64,
            false,
        )]))
    }

    pub(crate) fn make_count_batch(schema: SchemaRef, count: u64) -> DFResult<RecordBatch> {
        let count_array = Arc::new(UInt64Array::from(vec![count])) as ArrayRef;
        RecordBatch::try_new(schema, vec![count_array]).map_err(|e| {
            DataFusionError::ArrowError(
                Box::new(e),
                Some("Failed to make delete count batch".into()),
            )
        })
    }
}

impl Debug for IcebergDeleteExec {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "IcebergDeleteExec(table={}, mode={:?})",
            self.table.identifier(),
            self.mode
        )
    }
}

impl DisplayAs for IcebergDeleteExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "IcebergDeleteExec: table={}, mode={:?}",
            self.table.identifier(),
            self.mode
        )
    }
}

impl ExecutionPlan for IcebergDeleteExec {
    fn name(&self) -> &str {
        "IcebergDeleteExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "IcebergDeleteExec only has one partition, but got partition {partition}"
            )));
        }

        let table = self.table.clone();
        let catalog = Arc::clone(&self.catalog);
        let predicate = self.predicate.clone();
        let prune = self.prune.clone();
        let mode = self.mode;
        let isolation = self.isolation;
        let table_schema = Arc::clone(&self.table_schema);
        let count_schema = Arc::clone(&self.count_schema);
        let commit_branch = self.commit_branch.clone();

        let stream = futures::stream::once(async move {
            let deleted = match mode {
                WriteMode::MergeOnRead => {
                    merge_on_read_delete(
                        &table,
                        catalog.as_ref(),
                        predicate,
                        prune,
                        &table_schema,
                        isolation,
                        commit_branch.as_deref(),
                    )
                    .await?
                }
                WriteMode::CopyOnWrite => {
                    copy_on_write_delete(
                        &table,
                        catalog.as_ref(),
                        predicate,
                        prune,
                        &table_schema,
                        isolation,
                        commit_branch.as_deref(),
                    )
                    .await?
                }
            };
            Self::make_count_batch(count_schema, deleted)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.count_schema),
            stream,
        )))
    }
}

/// The delete-file kind a table's format version requires for merge-on-read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeOnReadDeleteKind {
    /// V2: Parquet position-delete files.
    PositionDeletes,
    /// V3: Puffin deletion vectors. The spec forbids NEW position-delete files at this version.
    DeletionVectors,
}

/// Resolves which delete-file kind this table takes. Call it BEFORE any I/O: a format rejection at
/// commit time would orphan an already written delete or data file.
///
/// # Errors
///
/// `NotImplemented` for a V1 table, which has no delete files of any kind.
fn merge_on_read_delete_kind(table: &Table) -> DFResult<MergeOnReadDeleteKind> {
    match table.metadata().format_version() {
        FormatVersion::V2 => Ok(MergeOnReadDeleteKind::PositionDeletes),
        FormatVersion::V3 => Ok(MergeOnReadDeleteKind::DeletionVectors),
        version => Err(DataFusionError::NotImplemented(format!(
            "merge-on-read DELETE/UPDATE needs delete files, which a {version:?} table does not \
             have — use copy-on-write instead"
        ))),
    }
}

async fn mor_scan_stream(
    table: &Table,
    projection: Vec<String>,
    prune: Option<Predicate>,
    scan_snapshot_id: Option<i64>,
) -> DFResult<(
    ArrowRecordBatchStream,
    HashMap<String, (i32, Struct)>,
)> {
    let mut builder = table.scan().select(projection);
    if let Some(snapshot_id) = scan_snapshot_id {
        builder = builder.snapshot_id(snapshot_id);
    }
    if let Some(prune) = prune {
        builder = builder.with_file_prune_only(prune);
    }
    let scan = builder.build().map_err(to_datafusion_error)?;
    scan
        .to_arrow_with_file_partitions()
        .await
        .map_err(to_datafusion_error)
}

/// Writes the merge-on-read delete files for `pairs`, which the caller already sorted by
/// `(path, pos)` for the V2 writer. Returns `(files to add, files the commit must remove)`. The
/// removal half is non-empty on the V3 path: a merged DV supersedes the DV or file-scoped parquet it absorbed.
async fn write_merge_on_read_deletes(
    table: &Table,
    kind: MergeOnReadDeleteKind,
    pairs: &[(String, i64)],
    scan_snapshot_id: Option<i64>,
) -> DFResult<DvContainerClose> {
    match kind {
        MergeOnReadDeleteKind::PositionDeletes => Ok(DvContainerClose {
            added: write_position_deletes(table, pairs, scan_snapshot_id).await?,
            ..DvContainerClose::default()
        }),
        MergeOnReadDeleteKind::DeletionVectors => {
            write_deletion_vectors(table, pairs, scan_snapshot_id).await
        }
    }
}

fn apply_dv_container_close(
    mut action: iceberg::transaction::RowDeltaAction,
    close: DvContainerClose,
) -> iceberg::transaction::RowDeltaAction {
    if !close.added.is_empty() {
        action = action.add_deletes(close.added);
    }
    if !close.removed.is_empty() {
        action = action.remove_deletes_many(close.removed);
    }
    action
}

/// Sorts position-delete pairs into the ascending order the Iceberg spec requires. The concurrent
/// scan interleaves files, so the pairs arrive unordered. A named seam, so a test can pin it.
fn sort_position_delete_pairs(pairs: &mut [(String, i64)]) {
    pairs.sort();
}

fn files_exist_set(close: &DvContainerClose, pairs: &[(String, i64)]) -> HashSet<String> {
    if close
        .added
        .iter()
        .any(|file| file.file_format() == DataFileFormat::Puffin)
    {
        return close.referenced_data_files();
    }
    let mut paths = HashSet::new();
    let mut previous: Option<&str> = None;
    for (path, _) in pairs {
        if previous == Some(path.as_str()) {
            continue;
        }
        paths.insert(path.clone());
        previous = Some(path.as_str());
    }
    paths
}

/// Merge-on-read DELETE: finds the matching rows' `_file`/`_pos`, writes the delete files, and
/// commits a `RowDelta`. Returns the number of rows deleted.
///
/// The scan is consumed batch by batch. Only the matched `(path, pos)` pairs accumulate, so the
/// floor is O(matched rows), not O(1): `write_position_deletes` must group and sort the whole pair
/// set before it writes.
async fn merge_on_read_delete(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    prune: Option<Predicate>,
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
    commit_branch: Option<&str>,
) -> DFResult<u64> {
    let delete_kind = merge_on_read_delete_kind(table)?;
    let scan_snapshot_id =
        resolve_scan_snapshot_id(table, commit_branch).map_err(to_datafusion_error)?;
    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());
    projection.push(RESERVED_COL_NAME_POS.to_string());

    let (mut stream, _) = mor_scan_stream(table, projection, prune, scan_snapshot_id).await?;

    let mut pairs: Vec<(String, i64)> = Vec::new();
    while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
        // Resolve the predicate's columns BY NAME. The scan's output column order is not fixed.
        let keep_mask = match &predicate {
            None => None, // `DELETE FROM t` — every row matches.
            Some(physical_expr) => {
                let columns: Vec<ArrayRef> = table_schema
                    .fields()
                    .iter()
                    .map(|field| {
                        batch.column_by_name(field.name()).cloned().ok_or_else(|| {
                            DataFusionError::Internal(format!(
                                "delete scan is missing table column '{}'",
                                field.name()
                            ))
                        })
                    })
                    .collect::<DFResult<_>>()?;
                let table_batch = RecordBatch::try_new(Arc::clone(table_schema), columns)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
                let evaluated = physical_expr.evaluate(&table_batch)?;
                let array = evaluated.into_array(table_batch.num_rows())?;
                let mask = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "DELETE filter did not evaluate to a boolean".to_string(),
                        )
                    })?
                    .clone();
                Some(mask)
            }
        };

        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("delete scan missing _file column".to_string())
            })?;
        let pos_col = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .ok_or_else(|| {
                DataFusionError::Internal("delete scan missing _pos column".to_string())
            })?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| DataFusionError::Internal("_pos column is not Int64".to_string()))?;

        for row in 0..batch.num_rows() {
            // A NULL predicate result does NOT match, per SQL three-valued logic.
            let delete_row = match &keep_mask {
                None => true,
                Some(mask) => mask.is_valid(row) && mask.value(row),
            };
            if delete_row {
                pairs.push((
                    decode_file_path(file_col, row)?,
                    decode_position(pos_col, row)?,
                ));
            }
        }
    }

    // No matching rows → no-op (an empty RowDelta would be a pointless snapshot).
    if pairs.is_empty() {
        return Ok(0);
    }

    // Position deletes MUST be sorted by (path, pos) per the spec.
    sort_position_delete_pairs(&mut pairs);
    let deleted = pairs.len() as u64;

    // The §5 `validate_data_files_exist` set. Java arms this for every command, DELETE included: a
    // referenced file rewritten away by a concurrent commit would silently lose these deletes.
    let close = write_merge_on_read_deletes(table, delete_kind, &pairs, scan_snapshot_id).await?;
    let referenced_files = files_exist_set(&close, &pairs);

    // §5 row-delta recipe, MoR DELETE. `AlwaysTrue` is Java-exact because this path pushes no filter
    // into the scan. V3 shared-Puffin closure arms deleted-files checks (F-17 C-013); V2 keeps
    // Java's skip-delete default.
    let tx = Transaction::new(table);
    let mut action = tx
        .row_delta()
        .conflict_detection_filter(Predicate::AlwaysTrue)
        .validate_data_files_exist(referenced_files);
    if delete_kind == MergeOnReadDeleteKind::DeletionVectors {
        action = action.validate_deleted_files();
    }
    action = apply_dv_container_close(action, close);
    action = maybe_validate_from_snapshot(action, scan_snapshot_id, |action, snapshot_id| {
        action.validate_from_snapshot(snapshot_id)
    });
    if isolation == IsolationLevel::Serializable {
        action = action.validate_no_conflicting_data_files();
    }
    let action = maybe_to_branch(action, commit_branch, |action, branch| {
        action.to_branch(branch)
    });
    action
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(deleted)
}

/// Copy-on-write DELETE: a file-level rewrite. It finds the data files holding at least one deleted
/// row, rewrites only those files' survivors, and commits an `OverwriteFiles`. Unaffected files stay
/// untouched. The survivors span many partitions and one batch may interleave files, so the
/// [`TaskWriter`] runs with `fanout_enabled = true` and routes each row without pre-sorting.
async fn copy_on_write_delete(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    prune: Option<Predicate>,
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
    commit_branch: Option<&str>,
) -> DFResult<u64> {
    let scan_snapshot_id =
        resolve_scan_snapshot_id(table, commit_branch).map_err(to_datafusion_error)?;

    let mut stream = cow_scan_stream(table, table_schema, scan_snapshot_id, prune.clone()).await?;
    let mut deleted: u64 = 0;
    let mut affected: HashSet<String> = HashSet::new();

    while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("delete scan missing _file column".to_string())
            })?;
        let table_batch = table_column_batch(&batch, table_schema)?;
        // `match_mask` collapses NULL → false (3VL) and is all-true when there is no predicate.
        let mask = match_mask(&predicate, &table_batch)?;

        let paths = decode_file_paths_batch(file_col)?;
        for (row, path) in paths.iter().enumerate() {
            if mask.value(row) {
                deleted += 1;
                if !affected.contains(*path) {
                    affected.insert((*path).to_string());
                }
            }
        }
    }

    // Rust drops a shadowed value at scope end, not at the shadowing point. Release pass 1's scan
    // state explicitly, so the peak really is one scan plus one batch.
    drop(stream);

    // No deleted rows → no-op. This also skips the second scan: a zero-match DELETE reads once.
    if deleted == 0 {
        return Ok(0);
    }

    // Pass 2 — re-scan the same snapshot and stream the survivors of affected files into the writer.
    // A row is kept when it is not deleted AND comes from an affected file. Nothing accumulates.
    //
    // `DELETE FROM t` (no predicate) is short-circuited. `match_mask` is then all-true by
    // construction, so no row can be kept: pass 2 is provably empty, not approximated.
    let new_files = if predicate.is_none() {
        Vec::new()
    } else {
        let mut stream =
            cow_scan_stream(table, table_schema, scan_snapshot_id, prune.clone()).await?;
        // Build the writer on the first batch that HAS survivors. A DELETE that empties every
        // affected file must not fail inside a constructor it never needed.
        let mut data_writer: Option<StreamingDataFileWriter> = None;

        while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
            let num_rows = batch.num_rows();
            let file_col = batch
                .column_by_name(RESERVED_COL_NAME_FILE)
                .ok_or_else(|| {
                    DataFusionError::Internal("delete scan missing _file column".to_string())
                })?;
            let table_batch = table_column_batch(&batch, table_schema)?;
            // Pass 2 is an independent scan, so no per-batch state from pass 1 aligns with it.
            let delete_mask = match_mask(&predicate, &table_batch)?;

            let paths = decode_file_paths_batch(file_col)?;
            let keep: BooleanArray = (0..num_rows)
                .map(|row| !delete_mask.value(row) && affected.contains(paths[row]))
                .collect();
            if keep.true_count() == 0 {
                continue;
            }

            let mut surviving = filter_record_batch(&table_batch, &keep)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
            if let Some((row_id, last_updated)) = filter_lineage_columns(&batch, &keep)? {
                surviving = attach_lineage(surviving, row_id, last_updated)?;
            }
            if data_writer.is_none() {
                data_writer = Some(StreamingDataFileWriter::try_new(table)?);
            }
            let Some(writer) = data_writer.as_mut() else {
                return Err(DataFusionError::Internal(
                    "copy-on-write DELETE writer was not initialized".to_string(),
                ));
            };
            writer.write_batch(surviving).await?;
        }

        // No survivors ⇒ no writer was built ⇒ an empty Vec, so no empty data file is committed.
        match data_writer {
            Some(writer) => writer.finish().await?,
            None => Vec::new(),
        }
    };

    // Commit: remove the affected source files, add the rewritten ones. The removals carry FULL
    // `DataFile` metadata, so the §5 conflicting-deletes check is LIVE. It tests concurrent delete
    // files against partition and metrics, which a bare path cannot carry.
    let removed_data_files =
        resolve_affected_data_files(table, &affected, scan_snapshot_id).await?;
    let tx = Transaction::new(table);
    let mut action = tx
        .overwrite_files()
        .delete_data_files(removed_data_files)
        .add_files(new_files)
        .conflict_detection_filter(Predicate::AlwaysTrue)
        .validate_no_conflicting_deletes();
    action = maybe_validate_from_snapshot(action, scan_snapshot_id, |action, snapshot_id| {
        action.validate_from_snapshot(snapshot_id)
    });
    if isolation == IsolationLevel::Serializable {
        action = action.validate_no_conflicting_data();
    }
    let action = maybe_to_branch(action, commit_branch, |action, branch| {
        action.to_branch(branch)
    });
    action
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(deleted)
}

/// Writes Parquet position-delete files from sorted `(path, pos)` pairs and returns EVERY file the
/// rolling writer produced; dropping one silently resurrects its rows. Each file is stamped with the
/// `(spec_id, partition)` of the DATA file it deletes from, which the partitioned path reads from
/// the snapshot's manifests. The commit validates that stamp against the spec.
///
/// This predicate decides which table shape may skip that walk (BUG-001, C1-L-002):
///
/// | Table shape | Path |
/// |---|---|
/// | one spec, zero fields | fast path: one file, stamped through `with_partition_spec` so the real spec id survives |
/// | multi-spec, empty default (after `DROP PARTITION FIELD`) | walk: old data files keep their own partition, and a fabricated `None`/spec-0 stamp misses on read and resurrects rows |
/// | one all-Void spec (unpartitioned, non-empty fields) | walk: it needs a null tuple of matching arity |
/// | partitioned | walk |
pub(crate) use super::cow_affected::position_delete_unpartitioned_fast_path;

async fn write_position_deletes(
    table: &Table,
    pairs: &[(String, i64)],
    scan_snapshot_id: Option<i64>,
) -> DFResult<Vec<DataFile>> {
    let config = PositionDeleteWriterConfig::new().map_err(to_datafusion_error)?;
    let metadata = table.metadata();
    let default_spec = metadata.default_partition_spec();
    let schema = metadata.current_schema();

    // Only a never-evolved empty spec skips the manifest walk — see the fast-path table above.
    if position_delete_unpartitioned_fast_path(
        metadata.partition_specs_iter().len(),
        default_spec.fields().len(),
    ) {
        // `with_partition_spec` keeps the sole spec's real id; `None` would fabricate spec id 0.
        return write_position_deletes_for_partition(
            table,
            &config,
            pairs,
            None,
            Some(default_spec.as_ref().clone()),
        )
        .await;
    }

    let path_to_partition = live_data_file_partitions(table, scan_snapshot_id, None).await?;

    let path_to_partition: HashMap<String, (i32, Struct)> = path_to_partition
        .into_iter()
        .map(|(path, (spec_id, partition, _))| (path, (spec_id, partition)))
        .collect();
    let groups = group_pairs_by_partition(pairs, &path_to_partition)?;

    let mut all_delete_files: Vec<DataFile> = Vec::new();
    for ((spec_id, partition), mut group_pairs) in groups {
        // Maintain the per-file (path, pos) sort order within each group.
        sort_position_delete_pairs(&mut group_pairs);

        let spec = metadata
            .partition_spec_by_id(spec_id)
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "position-delete: data file references unknown partition spec {spec_id}"
                ))
            })?
            .as_ref()
            .clone();
        // Carry the data file's own (spec, partition), including empty and all-Void null tuples. A
        // `None` key would fabricate spec id 0 and under-attach after DROP PARTITION FIELD.
        let partition_key =
            PartitionKey::new(spec, schema.clone(), partition).map_err(to_datafusion_error)?;

        let files = write_position_deletes_for_partition(
            table,
            &config,
            &group_pairs,
            Some(partition_key),
            None,
        )
        .await?;
        all_delete_files.extend(files);
    }

    Ok(all_delete_files)
}

/// The `(path, pos)` pairs of one position-delete output file, keyed by the `(spec_id, partition)`
/// of the data files they delete from.
type PositionDeleteGroups = HashMap<(i32, Struct), Vec<(String, i64)>>;

/// Groups `(path, pos)` pairs by the `(spec_id, partition)` of the data file each deletes from, so
/// every output file is stamped like its target. Only the partitioned path reaches this. A pair
/// whose data file is absent from `path_to_partition` is a hard error: the pairs come from a scan of
/// the same snapshot that built the map. The old fallback fabricated an EMPTY tuple under a
/// PARTITIONED spec, writing a delete file under a `field=null` path that no reader can match.
fn group_pairs_by_partition(
    pairs: &[(String, i64)],
    path_to_partition: &HashMap<String, (i32, Struct)>,
) -> DFResult<PositionDeleteGroups> {
    let mut groups = PositionDeleteGroups::new();
    for pair in pairs {
        let key = path_to_partition.get(&pair.0).cloned().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "position-delete: data file `{}` is not a live file of the current snapshot, so \
                 its partition cannot be resolved",
                pair.0
            ))
        })?;
        groups.entry(key).or_default().push(pair.clone());
    }
    Ok(groups)
}

/// Writes one position-delete file for a SINGLE `(spec_id, partition)` group. `pairs` must already
/// be sorted by `(path, pos)`. With `partition_key = None`, `configured_spec` MUST be `Some`, or the
/// writer fabricates `DEFAULT_PARTITION_SPEC_ID` (0) instead of the real spec id.
async fn write_position_deletes_for_partition(
    table: &Table,
    config: &PositionDeleteWriterConfig,
    pairs: &[(String, i64)],
    partition_key: Option<PartitionKey>,
    configured_spec: Option<iceberg::spec::PartitionSpec>,
) -> DFResult<Vec<DataFile>> {
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).map_err(to_datafusion_error)?;
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    // Keep the `file_path` and `pos` bounds FULL and EXACT: no parquet stats truncation, so
    // min_is_exact/max_is_exact stay true and equal-bounds path routing works for long S3 URIs.
    let parquet_builder =
        ParquetWriterBuilder::new(position_delete_writer_properties(), config.schema().clone())
            .with_metrics_config(MetricsConfig::for_position_delete());
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    if partition_key.is_none() && configured_spec.is_none() {
        return Err(DataFusionError::Internal(
            "position-delete: write_position_deletes_for_partition requires either a PartitionKey \
             or a configured_spec; both None would fabricate partition_spec_id 0"
                .to_string(),
        ));
    }
    let mut builder = PositionDeleteFileWriterBuilder::new(rolling, config.clone());
    if let Some(spec) = configured_spec {
        builder = builder.with_partition_spec(spec);
    }
    let mut writer = builder
        .build(partition_key)
        .await
        .map_err(to_datafusion_error)?;

    let paths: Vec<&str> = pairs.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .map_err(|e| {
        DataFusionError::ArrowError(
            Box::new(e),
            Some("Failed to build position-delete batch".into()),
        )
    })?;
    writer.write(batch).await.map_err(to_datafusion_error)?;
    let files = writer.close().await.map_err(to_datafusion_error)?;
    // A non-empty group MUST produce a file, or the deletes vanish and the rows come back.
    if files.is_empty() {
        return Err(DataFusionError::Internal(
            "position-delete writer produced no file for a non-empty pair group".to_string(),
        ));
    }
    Ok(files)
}

/// Decodes the reserved `_file` column at `row`, tolerating run-end-encoded and plain `Utf8`. A NULL
/// slot is an error, not a value: arrow's `value()` returns `""` there, and an empty path becomes a
/// position delete against a file that does not exist.
fn decode_file_path(col: &ArrayRef, row: usize) -> DFResult<String> {
    use datafusion::arrow::array::RunArray;
    use datafusion::arrow::datatypes::Int32Type;

    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        if plain.is_null(row) {
            return Err(null_file_path_error(row));
        }
        return Ok(plain.value(row).to_string());
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let physical = run.get_physical_index(row);
        let values = run
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                DataFusionError::Internal("_file REE values are not Utf8".to_string())
            })?;
        if values.is_null(physical) {
            return Err(null_file_path_error(row));
        }
        return Ok(values.value(physical).to_string());
    }
    Err(DataFusionError::Internal(format!(
        "unexpected _file column type: {:?}",
        col.data_type()
    )))
}

/// The one error raised for a NULL reserved `_file` slot, in both decode paths.
fn null_file_path_error(row: usize) -> DataFusionError {
    DataFusionError::Internal(format!(
        "reserved _file column is NULL at row {row}; a position delete cannot be keyed by an \
         unknown data file"
    ))
}

/// Decodes the reserved `_pos` column at `row`. A NULL slot is an error for the same reason as
/// [`decode_file_path`]: arrow returns `0`, which would position-delete row 0 of a real data file.
fn decode_position(col: &Int64Array, row: usize) -> DFResult<i64> {
    if col.is_null(row) {
        return Err(DataFusionError::Internal(format!(
            "reserved _pos column is NULL at row {row}; a position delete cannot be keyed by an \
             unknown row position"
        )));
    }
    Ok(col.value(row))
}

/// Decodes the `_file` column for a whole batch in one pass (row `i` → `out[i]`). Equivalent to
/// [`decode_file_path`] per row, but it allocates no `String`: each run's value of a run-end-encoded
/// column is resolved once and reused. The strings are byte-identical, and in the same order.
fn decode_file_paths_batch(col: &ArrayRef) -> DFResult<Vec<&str>> {
    use datafusion::arrow::array::RunArray;
    use datafusion::arrow::datatypes::Int32Type;

    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        return (0..plain.len())
            .map(|row| {
                if plain.is_null(row) {
                    return Err(null_file_path_error(row));
                }
                Ok(plain.value(row))
            })
            .collect();
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let values = run
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                DataFusionError::Internal("_file REE values are not Utf8".to_string())
            })?;
        let mut out = Vec::with_capacity(run.len());
        if run.offset() == 0 {
            // Unsliced REE, the only shape the COW scan produces: the logical index equals the
            // physical run-end offset, so one walk gives the same `&str` per row as the form below.
            let run_ends = run.run_ends().values();
            let mut start = 0usize;
            for (physical, &end) in run_ends.iter().enumerate() {
                let end = usize::try_from(end).map_err(|_| {
                    DataFusionError::Internal("_file REE run-end is negative".to_string())
                })?;
                if start < end && values.is_null(physical) {
                    return Err(null_file_path_error(start));
                }
                let value = values.value(physical);
                for _ in start..end {
                    out.push(value);
                }
                start = end;
            }
        } else {
            // Sliced REE: the logical-to-physical map is offset-relative, so defer per row. A
            // sliced run-ends walk is easy to get subtly wrong.
            for row in 0..run.len() {
                let physical = run.get_physical_index(row);
                if values.is_null(physical) {
                    return Err(null_file_path_error(row));
                }
                out.push(values.value(physical));
            }
        }
        return Ok(out);
    }
    Err(DataFusionError::Internal(format!(
        "unexpected _file column type: {:?}",
        col.data_type()
    )))
}

/// Evaluate the `WHERE` predicate (or all-true when `None`) over `table_batch` to a NULL-free keep mask
/// (`true` ⇒ the row matches — a NULL result is NOT a match, per SQL three-valued logic).
fn match_mask(
    predicate: &Option<Arc<dyn PhysicalExpr>>,
    table_batch: &RecordBatch,
) -> DFResult<BooleanArray> {
    let num_rows = table_batch.num_rows();
    match predicate {
        None => Ok(BooleanArray::from(vec![true; num_rows])),
        Some(physical_expr) => {
            let array = physical_expr.evaluate(table_batch)?.into_array(num_rows)?;
            let raw = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal("filter did not evaluate to a boolean".to_string())
                })?;
            Ok((0..num_rows)
                .map(|row| raw.is_valid(row) && raw.value(row))
                .collect())
        }
    }
}

/// Rebuilds a batch of exactly the table columns, resolved BY NAME, in table-schema order. That is
/// the schema the `PhysicalExpr`s are bound to and the writer matches against.
fn table_column_batch(batch: &RecordBatch, table_schema: &SchemaRef) -> DFResult<RecordBatch> {
    let columns: Vec<ArrayRef> = table_schema
        .fields()
        .iter()
        .map(|field| {
            batch.column_by_name(field.name()).cloned().ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "scan is missing table column '{}'",
                    field.name()
                ))
            })
        })
        .collect::<DFResult<_>>()?;
    RecordBatch::try_new(Arc::clone(table_schema), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

/// Applies the `SET` assignments to `table_batch`. With `Some(mask)` only masked-true rows take the
/// new value: copy-on-write, whose batch holds matching and non-matching rows. With `None` every row
/// is updated, because merge-on-read already filtered the batch.
fn apply_assignments(
    table_batch: &RecordBatch,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    table_schema: &SchemaRef,
    mask: Option<&BooleanArray>,
) -> DFResult<RecordBatch> {
    let num_rows = table_batch.num_rows();
    let mut columns: Vec<ArrayRef> = table_batch.columns().to_vec();
    for (col_idx, value_expr) in assignments {
        let new_values = value_expr.evaluate(table_batch)?.into_array(num_rows)?;
        let assigned = match mask {
            None => new_values,
            Some(mask) => zip(mask, &new_values, &columns[*col_idx])
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
        };
        // An assignment must not put a NULL into a REQUIRED column. Use `logical_null_count`: the
        // physical count is 0 for a dictionary- or REE-encoded array whose VALUES carry the NULL,
        // and `RecordBatch::try_new`'s own check is physical too, so the NULL clears both gates.
        let field = table_schema.field(*col_idx);
        if !field.is_nullable() && assigned.logical_null_count() > 0 {
            return Err(DataFusionError::Plan(format!(
                "UPDATE cannot assign NULL to required column '{}'",
                field.name()
            )));
        }
        columns[*col_idx] = assigned;
    }
    RecordBatch::try_new(Arc::clone(table_schema), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

/// Merge-on-read UPDATE: position-delete the OLD matching rows and insert NEW rows with the updated
/// values, in one `RowDelta`. Returns the number of rows updated. The new rows go through
/// [`StreamingDataFileWriter`], which reads partition values from the POST-assignment columns.
/// Position deletes are keyed by `(path, pos)` and are partition-agnostic.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn merge_on_read_update(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    prune: Option<Predicate>,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
    commit_branch: Option<&str>,
) -> DFResult<u64> {
    let delete_kind = merge_on_read_delete_kind(table)?;
    let scan_snapshot_id =
        resolve_scan_snapshot_id(table, commit_branch).map_err(to_datafusion_error)?;

    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());
    projection.push(RESERVED_COL_NAME_POS.to_string());
    push_lineage_scan_columns(&mut projection, table.metadata().format_version());

    let (mut stream, _) = mor_scan_stream(table, projection, prune, scan_snapshot_id).await?;

    // The delete side buffers the matched pairs, because `write_position_deletes` must group and
    // sort them. The new-row side streams into the writer per batch.
    let mut pairs: Vec<(String, i64)> = Vec::new();
    let mut data_writer = StreamingDataFileWriter::try_new(table)?;
    while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
        let table_batch = table_column_batch(&batch, table_schema)?;
        let mask = match_mask(&predicate, &table_batch)?;
        if mask.true_count() == 0 {
            continue;
        }

        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("update scan missing _file column".to_string())
            })?;
        let pos_col = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .ok_or_else(|| {
                DataFusionError::Internal("update scan missing _pos column".to_string())
            })?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| DataFusionError::Internal("_pos column is not Int64".to_string()))?;
        for row in 0..mask.len() {
            if mask.value(row) {
                pairs.push((
                    decode_file_path(file_col, row)?,
                    decode_position(pos_col, row)?,
                ));
            }
        }

        // All rows here match, so the assignments need no per-row mask.
        let matching = filter_record_batch(&table_batch, &mask)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        let new_rows_batch = apply_assignments(&matching, assignments, table_schema, None)?;
        let new_rows_batch = attach_update_lineage(&batch, &mask, new_rows_batch, None)?;
        data_writer.write_batch(new_rows_batch).await?;
    }

    let updated = pairs.len() as u64;
    if updated == 0 {
        // No batch ever reached the writer, so `finish` produces no file. Nothing to commit.
        let empty = data_writer.finish().await?;
        debug_assert!(empty.is_empty());
        return Ok(0);
    }

    // Grouping and sorting need the whole pair set up front. Both sides complete BEFORE the single
    // commit below.
    sort_position_delete_pairs(&mut pairs);
    let close = write_merge_on_read_deletes(table, delete_kind, &pairs, scan_snapshot_id).await?;
    let referenced_files = files_exist_set(&close, &pairs);
    let data_files = data_writer.finish().await?;

    // §5 row-delta recipe, MoR UPDATE. UPDATE arms the deleted-files checks at BOTH levels, because
    // the op READ the rows it rewrote: a concurrent delete of them conflicts. Java arms these for
    // UPDATE and MERGE only, never DELETE.
    let tx = Transaction::new(table);
    let mut action = tx
        .row_delta()
        .add_data_files(data_files)
        .conflict_detection_filter(Predicate::AlwaysTrue)
        .validate_data_files_exist(referenced_files)
        .validate_deleted_files()
        .validate_no_conflicting_delete_files();
    action = apply_dv_container_close(action, close);
    action = maybe_validate_from_snapshot(action, scan_snapshot_id, |action, snapshot_id| {
        action.validate_from_snapshot(snapshot_id)
    });
    if isolation == IsolationLevel::Serializable {
        action = action.validate_no_conflicting_data_files();
    }
    let action = maybe_to_branch(action, commit_branch, |action, branch| {
        action.to_branch(branch)
    });
    action
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(updated)
}

/// Copy-on-write UPDATE: a file-level rewrite. It finds the data files holding at least one updated
/// row and rewrites those files in full: matched rows take the new values, the rest are carried
/// unchanged. It then commits an `OverwriteFiles`. A SET on a partition-key column moves the row to
/// its new partition, because [`StreamingDataFileWriter`] reads the post-assignment columns.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn copy_on_write_update(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    prune: Option<Predicate>,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
    commit_branch: Option<&str>,
) -> DFResult<u64> {
    let scan_snapshot_id =
        resolve_scan_snapshot_id(table, commit_branch).map_err(to_datafusion_error)?;

    let mut stream = cow_scan_stream(table, table_schema, scan_snapshot_id, prune.clone()).await?;
    let mut updated: u64 = 0;
    let mut affected: HashSet<String> = HashSet::new();

    while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("update scan missing _file column".to_string())
            })?;
        let table_batch = table_column_batch(&batch, table_schema)?;
        let mask = match_mask(&predicate, &table_batch)?;

        let paths = decode_file_paths_batch(file_col)?;
        for (row, path) in paths.iter().enumerate() {
            // A row is updated iff the predicate is TRUE (`match_mask` already coerced NULL → false).
            if mask.value(row) {
                updated += 1;
                if !affected.contains(*path) {
                    affected.insert((*path).to_string());
                }
            }
        }
    }

    // Release pass 1's exhausted scan: a shadowed binding keeps its state alive for all of pass 2.
    drop(stream);

    // No updated rows → no-op, and the second scan is skipped.
    if updated == 0 {
        return Ok(0);
    }

    // Pass 2 — re-scan and rewrite affected files only. Matched rows take the new SET values; other
    // rows of the SAME affected file keep their original values. Nothing accumulates.
    let mut stream = cow_scan_stream(table, table_schema, scan_snapshot_id, prune.clone()).await?;
    // Eager construction is safe here, unlike the DELETE path: `updated > 0` means at least one file
    // is affected, and every row of an affected file is rewritten.
    let mut data_writer = StreamingDataFileWriter::try_new(table)?;

    while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
        let num_rows = batch.num_rows();
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("update scan missing _file column".to_string())
            })?;

        let table_batch = table_column_batch(&batch, table_schema)?;

        let paths = decode_file_paths_batch(file_col)?;
        let keep_affected: BooleanArray = (0..num_rows)
            .map(|row| affected.contains(paths[row]))
            .collect();
        if keep_affected.true_count() == 0 {
            continue;
        }

        let affected_batch = filter_record_batch(&table_batch, &keep_affected)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

        // Evaluate the mask over `affected_batch` instead of caching pass 1's. The old cache was
        // indexed by batch POSITION, and pass 2 is an independent scan whose batch boundaries need
        // not match, so it could apply one batch's mask to another batch's rows.
        let affected_match_mask = match_mask(&predicate, &affected_batch)?;

        let rewritten = apply_assignments(
            &affected_batch,
            assignments,
            table_schema,
            Some(&affected_match_mask),
        )?;
        let rewritten = attach_update_lineage(
            &batch,
            &keep_affected,
            rewritten,
            Some(&affected_match_mask),
        )?;
        data_writer.write_batch(rewritten).await?;
    }

    // The writer routes each row by its POST-assignment values, so a partition-key-changing UPDATE
    // moves the row to the new partition.
    let new_files = data_writer.finish().await?;

    // Commit: remove the affected source files, add the rewritten ones. The removals carry FULL
    // metadata, not bare paths, because the §5 conflicting-deletes check needs partition and metrics.
    // Java's isolation switch does not branch on the command, so this matches the DELETE recipe.
    let removed_data_files =
        resolve_affected_data_files(table, &affected, scan_snapshot_id).await?;
    let tx = Transaction::new(table);
    let mut action = tx
        .overwrite_files()
        .delete_data_files(removed_data_files)
        .add_files(new_files)
        .conflict_detection_filter(Predicate::AlwaysTrue)
        .validate_no_conflicting_deletes();
    action = maybe_validate_from_snapshot(action, scan_snapshot_id, |action, snapshot_id| {
        action.validate_from_snapshot(snapshot_id)
    });
    if isolation == IsolationLevel::Serializable {
        action = action.validate_no_conflicting_data();
    }
    let action = maybe_to_branch(action, commit_branch, |action, branch| {
        action.to_branch(branch)
    });
    action
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(updated)
}
#[cfg(test)]
#[path = "delete_tests.rs"]
mod tests;
