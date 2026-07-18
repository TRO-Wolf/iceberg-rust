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

//! Physical plans for `DELETE FROM` and `UPDATE` — the [`TableProvider::delete_from`] and
//! [`TableProvider::update`] hooks.
//!
//! The mode is chosen by the table's `write.delete.mode` / `write.update.mode` property (Iceberg
//! standard). For DELETE:
//!   * **`merge-on-read`** — find the matching rows' reserved `_file`/`_pos` identity, write a
//!     position-delete file, and commit a `RowDelta`. Data files are untouched; the next scan applies
//!     the deletes. This is the engine-facing seam the core was built for.
//!   * **`copy-on-write`** (the Iceberg default when unset) — file-level rewrite: only the data files
//!     that contain at least one deleted row are rewritten; unaffected files are left in place. Survivors
//!     from affected files are routed through the partition-aware `TaskWriter`, so both partitioned and
//!     unpartitioned tables are supported. The commit is a `OverwriteFiles` that deletes the affected
//!     paths and adds the rewritten files.
//!
//! **Correctness — why we evaluate the filter ourselves.** The matching rows are identified by
//! evaluating the *original* DataFusion `WHERE` filters (as a [`PhysicalExpr`]) against the scanned
//! rows. We deliberately do **not** delete by Iceberg predicate pushdown: `convert_filters_to_predicate`
//! is *inexact* — it loosens an `AND` whose branch it cannot convert (returning the convertible side
//! alone) — which is harmless for a SELECT (DataFusion re-filters) but would **over-delete** here. The
//! exact filter is the contract; pushdown is only ever a (future) pruning optimization layered under it.
//!
//! **Memory.** The **merge-on-read** DELETE/UPDATE paths STREAM the live scan batch-by-batch (H7-S1):
//! they never hold the whole live row set. MoR DELETE buffers only the matched `(path, pos)` pairs
//! (two small fields per deleted row); MoR UPDATE additionally streams the new data rows straight into
//! the writer. The floor is O(matched rows), not O(1) — `write_position_deletes` must group + sort the
//! whole pair set before writing (the default scan interleaves files unordered). The **copy-on-write**
//! paths still buffer the full live row set (their two-pass affected-file rewrite) — streaming COW is a
//! follow-up (H7-S2).
//!
//! **Concurrency — the ENGINE_CONTRACT §5 recipes are ARMED (2026-07-18).** Every DELETE/UPDATE commit
//! enables the per-operation isolation validations with **Java's per-operation defaults as the oracle**
//! (`SparkRowLevelOperationBuilder.isolationLevel`, 1.10.0 L96-115: table property
//! `write.delete.isolation-level` / `write.update.isolation-level`, default **serializable**;
//! `IsolationLevel.fromName` parse semantics). Copy-on-write commits validate from the scanned snapshot
//! with an `AlwaysTrue` conflict-detection filter (this path pushes NO filters into the scan, so the
//! AND-of-pushed-filters Java computes — `SparkWrite.conflictDetectionFilter()` L417-428 — is exactly
//! `alwaysTrue`), reject concurrent conflicting deletes at BOTH levels, and reject concurrent
//! conflicting data (inserts) under serializable (`SparkWrite.java` L448-456, L467-509). The removed
//! files are supplied with FULL metadata (`delete_data_files`) so the conflicting-deletes check is live
//! — a bare path carries no partition/metrics and would make it inert. Merge-on-read commits always
//! validate that the data files their position deletes reference still exist
//! (`SparkPositionDeltaWrite.commit` L243), UPDATE additionally arms `validate_deleted_files` +
//! `validate_no_conflicting_delete_files` (L251-254 — UPDATE/MERGE only, NOT DELETE), and serializable
//! adds `validate_no_conflicting_data_files` (L256-258). A zero-match DML commits NOTHING (stronger
//! than Java's scan==null no-validation arm, L446-447). A validation failure is a NON-retryable
//! `DataInvalid` surfaced to the caller — see `docs/ENGINE_CONTRACT.md` §5.
//!
//! **Scope / limitations (out of scope here, named honestly):**
//!   * **Partition evolution / multi-spec tables** — copy-on-write rewrites survivors under the table's
//!     *current* partition spec (as Java does) and merge-on-read stamps each position-delete file with
//!     its target data file's *own* `(spec_id, partition)`; both are exercised on single-spec tables but
//!     a table whose specs have evolved is not yet covered by a test.
//!   * **Streaming** — merge-on-read streams (see *Memory* above); copy-on-write does not yet.
//!
//! The plan emits a single `UInt64` `count` row (rows affected), per DataFusion's DML contract.

use std::any::Any;
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
use iceberg::arrow::{FieldMatchMode, PROJECTED_PARTITION_VALUE_COLUMN, PartitionValueCalculator};
use iceberg::expr::Predicate;
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::spec::{DataFile, DataFileFormat, FormatVersion, MetricsConfig, PartitionKey, Struct};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};

use crate::task_writer::TaskWriter;
use crate::to_datafusion_error;

/// The Iceberg row-level write-mode properties and the `merge-on-read` value.
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
    /// Resolve from a table property (`write.delete.mode` / `write.update.mode`); Iceberg's default is
    /// copy-on-write when the property is absent or unrecognized.
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

/// The Iceberg row-level isolation-level table properties (Java `TableProperties.DELETE_ISOLATION_LEVEL`
/// / `UPDATE_ISOLATION_LEVEL`, 1.10.0 `TableProperties.java` L361/L369; shared default `"serializable"`,
/// L362/L370).
pub(crate) const WRITE_DELETE_ISOLATION_LEVEL: &str = "write.delete.isolation-level";
pub(crate) const WRITE_UPDATE_ISOLATION_LEVEL: &str = "write.update.isolation-level";

/// The isolation level of a row-level write (Java `org.apache.iceberg.IsolationLevel`) — the
/// engine-owned policy that selects which ENGINE_CONTRACT §5 conflict validations the DML commit
/// enables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IsolationLevel {
    /// Reject concurrent conflicting DATA (inserts matching the condition) AND concurrent conflicting
    /// DELETES.
    Serializable,
    /// Reject only concurrent conflicting DELETES; concurrent inserts are tolerated.
    Snapshot,
}

impl IsolationLevel {
    /// Parse an isolation-level name CASE-INSENSITIVELY (Java `IsolationLevel.fromName` =
    /// `valueOf(levelName.toUpperCase(Locale.ENGLISH))`). An unknown name fails LOUD with Java's
    /// message shape (`"Invalid isolation level: %s"`) — never silently defaulted.
    pub(crate) fn parse(name: &str) -> DFResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "serializable" => Ok(IsolationLevel::Serializable),
            "snapshot" => Ok(IsolationLevel::Snapshot),
            _ => Err(DataFusionError::Plan(format!(
                "Invalid isolation level: {name}"
            ))),
        }
    }

    /// Resolve the isolation level for a row-level DELETE/UPDATE from its table property, defaulting
    /// to SERIALIZABLE — Java's per-operation default (`SparkRowLevelOperationBuilder.isolationLevel`,
    /// 1.10.0 L96-115: `properties.getOrDefault(<op>_ISOLATION_LEVEL, <op>_ISOLATION_LEVEL_DEFAULT)`
    /// with both defaults `"serializable"`, then `IsolationLevel.fromName`). Like Java, this resolves
    /// at PLAN time (Java: the row-level-operation-builder constructor), so an invalid property value
    /// fails the statement before any scan or write happens.
    pub(crate) fn for_row_level_op(table: &Table, property: &str) -> DFResult<Self> {
        match table.metadata().properties().get(property) {
            Some(name) => Self::parse(name),
            None => Ok(IsolationLevel::Serializable),
        }
    }
}

/// `DELETE FROM` execution plan. Finds the matching rows, writes the delete artifacts, commits, and
/// emits the deleted-row count.
pub(crate) struct IcebergDeleteExec {
    table: Table,
    catalog: Arc<dyn Catalog>,
    /// The EXACT row filter (the `WHERE` clause as a `PhysicalExpr` over the table schema), or `None`
    /// to delete every row (`DELETE FROM t`).
    predicate: Option<Arc<dyn PhysicalExpr>>,
    mode: WriteMode,
    /// The §5 isolation level (resolved at plan time from `write.delete.isolation-level`, default
    /// serializable — Java's per-operation default).
    isolation: IsolationLevel,
    /// The table's Arrow schema — the projection base for the scan and the schema the `predicate` is
    /// bound to.
    table_schema: SchemaRef,
    count_schema: SchemaRef,
    plan_properties: PlanProperties,
}

impl IcebergDeleteExec {
    pub(crate) fn new(
        table: Table,
        catalog: Arc<dyn Catalog>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        mode: WriteMode,
        isolation: IsolationLevel,
        table_schema: SchemaRef,
    ) -> Self {
        let count_schema = Self::make_count_schema();
        let plan_properties = Self::compute_properties(Arc::clone(&count_schema));
        Self {
            table,
            catalog,
            predicate,
            mode,
            isolation,
            table_schema,
            count_schema,
            plan_properties,
        }
    }

    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        )
    }

    fn make_count_schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![Field::new(
            "count",
            DataType::UInt64,
            false,
        )]))
    }

    fn make_count_batch(schema: SchemaRef, count: u64) -> DFResult<RecordBatch> {
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
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
        let mode = self.mode;
        let isolation = self.isolation;
        let table_schema = Arc::clone(&self.table_schema);
        let count_schema = Arc::clone(&self.count_schema);

        let stream = futures::stream::once(async move {
            let deleted = match mode {
                WriteMode::MergeOnRead => {
                    merge_on_read_delete(
                        &table,
                        catalog.as_ref(),
                        predicate,
                        &table_schema,
                        isolation,
                    )
                    .await?
                }
                WriteMode::CopyOnWrite => {
                    copy_on_write_delete(
                        &table,
                        catalog.as_ref(),
                        predicate,
                        &table_schema,
                        isolation,
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

/// Merge-on-read writes Parquet position-delete files, which only the **V2** format supports — V1 has no
/// delete files, and V3 mandates Puffin deletion vectors (which this writer does not produce). Guard
/// BEFORE any I/O so a commit-time format rejection cannot orphan an already-written delete/data file.
fn require_v2_for_merge_on_read(table: &Table) -> DFResult<()> {
    let version = table.metadata().format_version();
    if version != FormatVersion::V2 {
        return Err(DataFusionError::NotImplemented(format!(
            "merge-on-read DELETE/UPDATE writes Parquet position deletes, which require a V2 table \
             (this table is {version:?}; V3 needs deletion vectors, not yet supported) — use \
             copy-on-write instead"
        )));
    }
    Ok(())
}

/// Sort position-delete `(file_path, pos)` pairs into the ascending `(file_path, pos)` order the
/// Iceberg spec requires for every position-delete file (Java `PositionDeleteWriter`). The default
/// concurrent scan interleaves files unordered, so the collected pairs are NOT sorted at scan time —
/// this restores the spec order before the pairs are written. Extracted as a named seam so the
/// ordering guarantee can be pinned by a deterministic unit test independent of scan interleaving.
fn sort_position_delete_pairs(pairs: &mut [(String, i64)]) {
    pairs.sort();
}

/// Merge-on-read DELETE: identify the matching rows' `_file`/`_pos`, write a position-delete file, and
/// commit a `RowDelta`. Returns the number of rows deleted.
///
/// **Streaming.** The live-row scan is consumed batch-by-batch (never the whole live row set is held in
/// RAM). For each batch we evaluate the exact `PhysicalExpr` and accumulate ONLY the matched
/// `(path, pos)` pairs — two small fields per deleted row — into `pairs`. This drops the previous
/// full-column `Vec<RecordBatch>` buffer. The memory floor is O(matched rows), NOT O(1): the position
/// deletes must be grouped by `(spec_id, partition)` and sorted `(path, pos)` before writing (the
/// default scan interleaves files unordered), so `write_position_deletes` still consumes the whole
/// `pairs` vector — see `task/h7-dml-streaming-scope.md` MEDIUM-1. For a whole-table DELETE this
/// degenerates to O(table rows × 2 fields), still far below the full-column buffer.
async fn merge_on_read_delete(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
) -> DFResult<u64> {
    require_v2_for_merge_on_read(table)?;
    // The snapshot this DML's scan reads — the §5 `validate_from_snapshot` anchor. Java sets it only
    // when the scan captured a snapshot (`SparkPositionDeltaWrite.java` L245-249; a table that was
    // empty at read time has none). The commit below is only reached when rows matched, which implies
    // a snapshot existed, but the guard keeps the Java shape.
    let scan_snapshot_id = table.metadata().current_snapshot_id();
    // 1. Scan EVERY live row, projecting the table columns (so the exact filter can be evaluated) plus
    //    the reserved `_file`/`_pos` row identity. We do not push the filter into the scan — see the
    //    module-level note on why Iceberg pushdown is inexact and unsafe for DELETE.
    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());
    projection.push(RESERVED_COL_NAME_POS.to_string());

    // Stream the scan batch-by-batch. Awaiting `stream.try_next()` polls the scan only as we consume
    // batches, so the scan is naturally back-pressured — no unbounded producer.
    let mut stream = table
        .scan()
        .select(projection)
        .build()
        .map_err(to_datafusion_error)?
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?;

    let mut pairs: Vec<(String, i64)> = Vec::new();
    while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
        // Build the table-column-only sub-batch (matching the schema the predicate is bound to) by
        // resolving each table field BY NAME — robust to the scan's output column ordering.
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
            // A row is deleted iff the WHERE predicate is TRUE for it (a NULL result, under SQL
            // three-valued logic, does NOT match), or there is no predicate (`DELETE FROM t`).
            let delete_row = match &keep_mask {
                None => true,
                Some(mask) => mask.is_valid(row) && mask.value(row),
            };
            if delete_row {
                pairs.push((decode_file_path(file_col, row)?, pos_col.value(row)));
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

    // The DATA files the position deletes reference — the §5 `validate_data_files_exist` set. Java
    // enables this check UNCONDITIONALLY for every command, DELETE included
    // (`SparkPositionDeltaWrite.commit` L243): a referenced file compacted or rewritten away by a
    // concurrent commit would silently lose these deletes.
    let referenced_files: HashSet<String> = pairs.iter().map(|(path, _)| path.clone()).collect();

    // Write ALL position-delete files the (rolling) writer produced and commit EVERY one of them.
    let delete_files = write_position_deletes(table, &pairs).await?;

    // ENGINE_CONTRACT §5 row-delta recipe, MoR DELETE row. The conflict-detection filter is the AND of
    // the scan's PUSHED filters (`SparkPositionDeltaWrite.conflictDetectionFilter` L284-292); this path
    // pushes NOTHING into the scan (exact-filter design, module docs), so `AlwaysTrue` is the
    // Java-exact value. DELETE does NOT arm `validate_deleted_files`/`validate_no_conflicting_delete_files`
    // (UPDATE/MERGE only — Java L251-254); serializable adds the conflicting-data check (L256-258).
    let tx = Transaction::new(table);
    let mut action = tx
        .row_delta()
        .add_deletes(delete_files)
        .conflict_detection_filter(Predicate::AlwaysTrue)
        .validate_data_files_exist(referenced_files);
    if let Some(snapshot_id) = scan_snapshot_id {
        action = action.validate_from_snapshot(snapshot_id);
    }
    if isolation == IsolationLevel::Serializable {
        action = action.validate_no_conflicting_data_files();
    }
    action
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(deleted)
}

/// Copy-on-write DELETE: **file-level** rewrite — scan every live row projecting the table columns
/// PLUS the reserved `_file` path, identify which source data files contain at least one deleted row
/// (the "affected" set), rewrite only those files' surviving rows through the partition-aware
/// [`TaskWriter`], and commit a `OverwriteFiles` that deletes the affected source paths and adds the
/// rewritten files. Unaffected data files are left completely untouched.
///
/// Works for BOTH partitioned and unpartitioned tables. A single Iceberg data file is always
/// single-partition, but the survivor set spans every affected file and therefore many partitions,
/// and a scan batch may interleave rows from several files — so the rewrite routes through a
/// `TaskWriter` with `fanout_enabled = true`, which sends each row to its correct partition writer
/// without requiring the survivors to be pre-sorted by partition.
async fn copy_on_write_delete(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
) -> DFResult<u64> {
    // The snapshot this DML's scan reads — the §5 `validate_from_snapshot` anchor (Java sets it only
    // when the scan captured one: `SparkWrite.java` L470-472 / L493-495).
    let scan_snapshot_id = table.metadata().current_snapshot_id();
    // 1. Scan EVERY live row, projecting the table columns PLUS `_file` (not `_pos` — COW doesn't need
    //    positions). We do NOT push the filter into the scan — Iceberg pushdown is inexact (see the
    //    module note); the exact `PhysicalExpr` evaluation here is the correctness contract.
    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());

    let batches: Vec<RecordBatch> = table
        .scan()
        .select(projection)
        .build()
        .map_err(to_datafusion_error)?
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .try_collect()
        .await
        .map_err(to_datafusion_error)?;

    // 2. Collect all batches into memory (documented: full-table buffer, fits in executor memory).

    // 3. Pass 1 — affected-file detection. A source file is AFFECTED iff at least one of its rows
    //    matches the predicate (or the predicate is None → all rows deleted → all files affected).
    //    Also counts total deleted rows for the return value.
    let mut deleted: u64 = 0;
    let mut affected: HashSet<String> = HashSet::new();

    for batch in &batches {
        let num_rows = batch.num_rows();
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("delete scan missing _file column".to_string())
            })?;

        // Build a table-column-only sub-batch for predicate evaluation (by name, robust to ordering).
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

        let paths = decode_file_paths_batch(file_col)?;
        match &predicate {
            None => {
                // DELETE FROM t — every row is deleted; every source file is affected.
                deleted += num_rows as u64;
                for path in &paths {
                    if !affected.contains(*path) {
                        affected.insert((*path).to_string());
                    }
                }
            }
            Some(physical_expr) => {
                let evaluated = physical_expr.evaluate(&table_batch)?;
                let array = evaluated.into_array(num_rows)?;
                let mask = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "DELETE filter did not evaluate to a boolean".to_string(),
                        )
                    })?
                    .clone();

                for (row, path) in paths.iter().enumerate() {
                    // A row is deleted iff the predicate is TRUE (NULL → SQL three-valued logic → NOT deleted).
                    let is_deleted = mask.is_valid(row) && mask.value(row);
                    if is_deleted {
                        deleted += 1;
                        if !affected.contains(*path) {
                            affected.insert((*path).to_string());
                        }
                    }
                }
            }
        }
    }

    // 4. No deleted rows → no-op (avoid a pointless snapshot).
    if deleted == 0 {
        return Ok(0);
    }

    // 5. Pass 2 — collect survivor rows from AFFECTED files only. Rows from unaffected files are left
    //    in place (their source files are unchanged). For each batch, build a keep-mask for rows that
    //    are (a) NOT deleted AND (b) from an affected file (those are the rows that need a new home).
    let mut survivors_to_rewrite: Vec<RecordBatch> = Vec::new();

    for batch in &batches {
        let num_rows = batch.num_rows();
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("delete scan missing _file column".to_string())
            })?;

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

        // Determine the delete-mask for this batch (same logic as pass 1, recomputed).
        let delete_mask: Vec<bool> = match &predicate {
            None => vec![true; num_rows],
            Some(physical_expr) => {
                let evaluated = physical_expr.evaluate(&table_batch)?;
                let array = evaluated.into_array(num_rows)?;
                let mask = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(
                            "DELETE filter did not evaluate to a boolean".to_string(),
                        )
                    })?
                    .clone();
                (0..num_rows)
                    .map(|row| mask.is_valid(row) && mask.value(row))
                    .collect()
            }
        };

        // Keep a row iff: it is NOT deleted AND its source file is in the affected set.
        let paths = decode_file_paths_batch(file_col)?;
        let keep: BooleanArray = (0..num_rows)
            .map(|row| !delete_mask[row] && affected.contains(paths[row]))
            .collect();

        let surviving = filter_record_batch(&table_batch, &keep)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        if surviving.num_rows() > 0 {
            survivors_to_rewrite.push(surviving);
        }
    }

    // 6. Write survivors through the partition-aware TaskWriter. When there are no survivors-to-rewrite
    //    (e.g. every affected file was fully deleted), this produces an empty Vec — correct.
    let new_files = write_partitioned_data_files(table, &survivors_to_rewrite).await?;

    // 7. Commit: delete the affected source files, add the rewritten files. The removals carry FULL
    //    `DataFile` metadata (`delete_data_files`, resolved from the scanned snapshot's manifests) so
    //    the §5 conflicting-deletes validation is LIVE — it tests concurrently-added delete files
    //    against the removed files' partition + metrics, which a bare path cannot carry (Java validates
    //    the scan tasks' `DataFile` objects, `SparkWrite.commit` L434-437). §5 CoW recipe per
    //    `SparkWrite.java`: deletes-conflict at BOTH levels (L477/L499), data-conflict under
    //    serializable only (L476), `AlwaysTrue` conflict filter (= Java's AND of pushed filters when
    //    nothing is pushed, L417-428).
    let removed_data_files = resolve_affected_data_files(table, &affected).await?;
    let tx = Transaction::new(table);
    let mut action = tx
        .overwrite_files()
        .delete_data_files(removed_data_files)
        .add_files(new_files)
        .conflict_detection_filter(Predicate::AlwaysTrue)
        .validate_no_conflicting_deletes();
    if let Some(snapshot_id) = scan_snapshot_id {
        action = action.validate_from_snapshot(snapshot_id);
    }
    if isolation == IsolationLevel::Serializable {
        action = action.validate_no_conflicting_data();
    }
    action
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(deleted)
}

/// Resolve affected file PATHS (collected from the scan's reserved `_file` column) to their full live
/// [`DataFile`] entries in the scanned snapshot's DATA manifests. The full metadata (partition +
/// metrics) is what makes the §5 `validate_no_conflicting_deletes` check live on the copy-on-write
/// commit — the fork validates only `delete_data_files` entries, never bare paths.
///
/// Every affected path MUST resolve: the scan just read these files from this same immutable table
/// handle, so a missing path is an internal invariant breach, not a user error.
async fn resolve_affected_data_files(
    table: &Table,
    affected: &HashSet<String>,
) -> DFResult<Vec<DataFile>> {
    let metadata = table.metadata();
    let mut resolved: Vec<DataFile> = Vec::with_capacity(affected.len());
    let mut found: HashSet<String> = HashSet::with_capacity(affected.len());

    if let Some(snapshot) = metadata.current_snapshot() {
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), metadata)
            .await
            .map_err(to_datafusion_error)?;
        for manifest_entry in manifest_list.entries() {
            if manifest_entry.content != iceberg::spec::ManifestContentType::Data {
                continue;
            }
            let manifest = manifest_entry
                .load_manifest(table.file_io())
                .await
                .map_err(to_datafusion_error)?;
            for entry in manifest.entries() {
                if entry.is_alive()
                    && entry.data_file().content_type() == iceberg::spec::DataContentType::Data
                    && affected.contains(entry.file_path())
                    && !found.contains(entry.file_path())
                {
                    found.insert(entry.file_path().to_string());
                    resolved.push(entry.data_file().clone());
                }
            }
        }
    }

    if found.len() != affected.len() {
        let missing: Vec<&str> = affected
            .iter()
            .map(String::as_str)
            .filter(|path| !found.contains(*path))
            .collect();
        return Err(DataFusionError::Internal(format!(
            "copy-on-write: scanned data file(s) not live in the current snapshot: {}",
            missing.join(", ")
        )));
    }

    Ok(resolved)
}

/// A streaming, partition-correct data-file writer. Each call to [`Self::write_batch`] feeds one
/// table-column batch through the production `TaskWriter` without buffering it — so a caller can drain
/// a scan stream into it batch-by-batch and never hold the whole row set in memory.
///
/// Works for BOTH partitioned and unpartitioned tables. Each batch must contain only table-schema
/// columns (no `_file` or other reserved columns). For partitioned tables the internal
/// `PartitionValueCalculator` computes and injects the `_partition` struct column that `TaskWriter`'s
/// splitter reads. `fanout_enabled = true` because successive batches may carry rows from different
/// partitions (and a single scan batch may interleave them); the `FanoutWriter` routes each row to its
/// correct partition writer without requiring pre-sorting.
///
/// The underlying `TaskWriter` is created lazily on the FIRST batch, so a writer that is finished
/// without ever receiving a batch produces zero files (matching the previous "empty input → empty Vec"
/// contract) — no empty data file is committed.
/// The concrete data-file writer builder the DML paths use: a `DataFileWriter` over a rolling Parquet
/// writer with the default location / file-name generators. Aliased so the `StreamingDataFileWriter`
/// field types stay readable.
type DmlDataFileWriterBuilder =
    DataFileWriterBuilder<ParquetWriterBuilder, DefaultLocationGenerator, DefaultFileNameGenerator>;

struct StreamingDataFileWriter {
    writer: Option<TaskWriter<DmlDataFileWriterBuilder>>,
    schema: iceberg::spec::SchemaRef,
    partition_spec: iceberg::spec::PartitionSpecRef,
    /// Present only for partitioned tables; computes the `_partition` struct column per batch.
    calculator: Option<PartitionValueCalculator>,
    /// The builder used to lazily create the `TaskWriter` on the first batch.
    builder: Option<DmlDataFileWriterBuilder>,
}

impl StreamingDataFileWriter {
    /// Prepare a streaming writer for `table`. No `TaskWriter` (and therefore no output file) is
    /// created until the first [`Self::write_batch`] call.
    fn try_new(table: &Table) -> DFResult<Self> {
        let schema = table.metadata().current_schema().clone();
        let partition_spec = table.metadata().default_partition_spec().clone();

        let parquet_builder = ParquetWriterBuilder::new_with_match_mode(
            parquet::file::properties::WriterProperties::default(),
            schema.clone(),
            FieldMatchMode::Name,
        );
        let location_gen =
            DefaultLocationGenerator::new(table.metadata().clone()).map_err(to_datafusion_error)?;
        let file_name_gen = DefaultFileNameGenerator::new(
            uuid::Uuid::now_v7().to_string(),
            None,
            DataFileFormat::Parquet,
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );
        let builder = DataFileWriterBuilder::new(rolling);

        let calculator = if partition_spec.is_unpartitioned() {
            None
        } else {
            Some(
                PartitionValueCalculator::try_new(&partition_spec, &schema)
                    .map_err(to_datafusion_error)?,
            )
        };

        Ok(Self {
            writer: None,
            schema,
            partition_spec,
            calculator,
            builder: Some(builder),
        })
    }

    /// Lazily construct the underlying `TaskWriter` on first use.
    fn ensure_writer(&mut self) -> DFResult<&mut TaskWriter<DmlDataFileWriterBuilder>> {
        if self.writer.is_none() {
            let builder = self.builder.take().ok_or_else(|| {
                DataFusionError::Internal(
                    "StreamingDataFileWriter builder already consumed".to_string(),
                )
            })?;
            // fanout_enabled = true: successive batches may be unsorted across partitions.
            let writer = TaskWriter::try_new(
                builder,
                true,
                self.schema.clone(),
                self.partition_spec.clone(),
            )
            .map_err(to_datafusion_error)?;
            self.writer = Some(writer);
        }
        // Just-initialized above, so the writer is present.
        self.writer.as_mut().ok_or_else(|| {
            DataFusionError::Internal("StreamingDataFileWriter not initialized".into())
        })
    }

    /// Feed ONE table-column batch to the writer, injecting the `_partition` struct column for
    /// partitioned tables. Awaiting the inner `write` naturally back-pressures the upstream scan.
    async fn write_batch(&mut self, batch: RecordBatch) -> DFResult<()> {
        if self.partition_spec.is_unpartitioned() {
            // Unpartitioned: TaskWriter writes directly; no partition column needed.
            self.ensure_writer()?
                .write(batch)
                .await
                .map_err(to_datafusion_error)
        } else {
            // Partitioned: compute the `_partition` struct column and append it so the TaskWriter's
            // partition splitter can route rows to the correct partition writer.
            let calculator = self.calculator.as_ref().ok_or_else(|| {
                DataFusionError::Internal(
                    "StreamingDataFileWriter partition calculator missing".to_string(),
                )
            })?;
            let partition_array = calculator.calculate(&batch).map_err(to_datafusion_error)?;

            // Extend the batch's schema with the `_partition` struct field.
            let partition_field = datafusion::arrow::datatypes::Field::new(
                PROJECTED_PARTITION_VALUE_COLUMN,
                partition_array.data_type().clone(),
                false,
            );
            let extended_schema = Arc::new(ArrowSchema::new(
                batch
                    .schema()
                    .fields()
                    .iter()
                    .cloned()
                    .chain(std::iter::once(Arc::new(partition_field)))
                    .collect::<Vec<_>>(),
            ));
            let mut extended_columns: Vec<ArrayRef> = batch.columns().to_vec();
            extended_columns.push(partition_array);
            let extended_batch = RecordBatch::try_new(extended_schema, extended_columns)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

            self.ensure_writer()?
                .write(extended_batch)
                .await
                .map_err(to_datafusion_error)
        }
    }

    /// Close the writer and return every `DataFile` produced. If no batch was ever written, the
    /// `TaskWriter` was never created and this returns an empty `Vec` — no empty file is committed.
    async fn finish(self) -> DFResult<Vec<DataFile>> {
        match self.writer {
            None => Ok(Vec::new()),
            Some(writer) => writer.close().await.map_err(to_datafusion_error),
        }
    }
}

/// Write data file(s) partition-correctly through the production `TaskWriter`, buffering-slice form.
/// Retained for the copy-on-write paths (which pre-buffer their survivor/rewrite batches); the
/// merge-on-read UPDATE path streams via [`StreamingDataFileWriter`] instead.
///
/// Returns every `DataFile` the (possibly rolling) writer produced — correctly partitioned.
async fn write_partitioned_data_files(
    table: &Table,
    batches: &[RecordBatch],
) -> DFResult<Vec<DataFile>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }
    let mut writer = StreamingDataFileWriter::try_new(table)?;
    for batch in batches {
        writer.write_batch(batch.clone()).await?;
    }
    writer.finish().await
}

/// Write REAL parquet position-delete file(s) from sorted `(data_file_path, position)` pairs via the
/// production `PositionDeleteFileWriter`. Returns EVERY file the (rolling) writer produced — a large
/// DELETE may roll into more than one file, and ALL of them must be committed or the deletes in the
/// dropped files would be silently lost (rows resurrected on the next scan).
///
/// **Partition-aware.** Position-delete files are associated with the `(spec_id, partition)` of the
/// DATA file they delete from — the Iceberg commit validates that the delete file's partition matches the
/// registered spec for `partition_spec_id`. For UNPARTITIONED tables the partition key is `None`
/// (existing behavior). For PARTITIONED tables:
///
/// 1. The current snapshot manifests are scanned once to build a `path → (spec_id, Struct)` map.
/// 2. The `(path, pos)` pairs are grouped by their data file's `(spec_id, Struct)`.
/// 3. One position-delete file is written per group, stamped with that group's `PartitionKey`.
///
/// This mirrors Java `PositionDeleteWriter` which always carries a per-data-file `PartitionKey` and
/// `RewritePositionDeleteFiles` which groups delete files by `(spec_id, partition)`.
async fn write_position_deletes(table: &Table, pairs: &[(String, i64)]) -> DFResult<Vec<DataFile>> {
    let config = PositionDeleteWriterConfig::new().map_err(to_datafusion_error)?;
    let metadata = table.metadata();
    let default_spec = metadata.default_partition_spec();
    let schema = metadata.current_schema();

    // For unpartitioned tables, write a single delete file with no partition key — the fast path.
    if default_spec.is_unpartitioned() {
        return write_position_deletes_for_partition(table, &config, pairs, None).await;
    }

    // Partitioned: build path → (spec_id, partition Struct) from the current snapshot manifests.
    // This lets us stamp each delete file with the SAME spec + partition as the data file it deletes.
    let mut path_to_partition: HashMap<String, (i32, Struct)> = HashMap::new();

    if let Some(snapshot) = metadata.current_snapshot() {
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), metadata)
            .await
            .map_err(to_datafusion_error)?;

        for manifest_entry in manifest_list.entries() {
            // Skip delete-file manifests — we only need data file partitions.
            if manifest_entry.content != iceberg::spec::ManifestContentType::Data {
                continue;
            }
            let manifest = manifest_entry
                .load_manifest(table.file_io())
                .await
                .map_err(to_datafusion_error)?;
            for entry in manifest.entries() {
                if entry.is_alive()
                    && entry.data_file().content_type() == iceberg::spec::DataContentType::Data
                {
                    let df = entry.data_file();
                    path_to_partition
                        .entry(df.file_path().to_string())
                        .or_insert_with(|| (df.partition_spec_id(), df.partition().clone()));
                }
            }
        }
    }

    // Group pairs by (spec_id, partition). Pairs whose data file path is not found in the manifest
    // (shouldn't happen in practice) are grouped under the default spec with an empty partition as a
    // safe fallback — the validation will reject them if the spec is partitioned, exposing the bug.
    let mut groups: HashMap<(i32, Struct), Vec<(String, i64)>> = HashMap::new();
    for pair in pairs {
        let key = path_to_partition
            .get(&pair.0)
            .cloned()
            .unwrap_or_else(|| (default_spec.spec_id(), Struct::empty()));
        groups.entry(key).or_default().push(pair.clone());
    }

    // Write one position-delete file per (spec_id, partition) group.
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
        let partition_key = if spec.is_unpartitioned() {
            None
        } else {
            Some(PartitionKey::new(spec, schema.clone(), partition))
        };

        let files =
            write_position_deletes_for_partition(table, &config, &group_pairs, partition_key)
                .await?;
        all_delete_files.extend(files);
    }

    // Each group above is non-empty and `write_position_deletes_for_partition` guarantees it
    // produced at least one file, so `all_delete_files` is non-empty whenever `pairs` was.
    Ok(all_delete_files)
}

/// Write one position-delete file for a SINGLE `(spec_id, partition)` group. When `partition_key`
/// is `None` the file is unpartitioned (spec_id 0, empty struct). The caller must have pre-sorted
/// `pairs` by `(path, pos)`.
async fn write_position_deletes_for_partition(
    table: &Table,
    config: &PositionDeleteWriterConfig,
    pairs: &[(String, i64)],
    partition_key: Option<PartitionKey>,
) -> DFResult<Vec<DataFile>> {
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).map_err(to_datafusion_error)?;
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    // Keep the position-delete `file_path` / `pos` bounds FULL (Java `MetricsConfig.forPositionDelete`)
    // so delete-file path pruning stays precise — the default `truncate(16)` would widen the path range.
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    )
    .with_metrics_config(MetricsConfig::for_position_delete());
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
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
    // A non-empty group of pairs MUST produce at least one delete file — otherwise the deletes
    // would be silently lost (rows resurrected on re-scan). Guard both the unpartitioned fast-path
    // and every partitioned group here so the check can never be skipped.
    if files.is_empty() {
        return Err(DataFusionError::Internal(
            "position-delete writer produced no file for a non-empty pair group".to_string(),
        ));
    }
    Ok(files)
}

/// Decode the reserved `_file` column at `row`. The scan emits `_file` as a per-file constant, which the
/// transformer materializes as a Run-End-Encoded `Utf8` column; tolerate both REE and plain `Utf8`.
fn decode_file_path(col: &ArrayRef, row: usize) -> DFResult<String> {
    use datafusion::arrow::array::RunArray;
    use datafusion::arrow::datatypes::Int32Type;

    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
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
        return Ok(values.value(physical).to_string());
    }
    Err(DataFusionError::Internal(format!(
        "unexpected _file column type: {:?}",
        col.data_type()
    )))
}

/// Decode the `_file` column for an ENTIRE batch in one pass, returning one borrowed path per row
/// (row `i` → `out[i]`).
///
/// Equivalent to calling [`decode_file_path`] for every row, but it allocates NO per-row `String`:
/// for a run-end-encoded column (`_file` is REE with only F ≪ R distinct values) each run's value is
/// resolved once and reused across the run; for a plain `StringArray` each row's `&str` is returned
/// directly. The returned strings are byte-identical, in the same order, to what `decode_file_path`
/// would produce per row — callers that need owned paths intern via the affected/path set instead of
/// allocating one `String` per row.
fn decode_file_paths_batch(col: &ArrayRef) -> DFResult<Vec<&str>> {
    use datafusion::arrow::array::RunArray;
    use datafusion::arrow::datatypes::Int32Type;

    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        return Ok((0..plain.len()).map(|row| plain.value(row)).collect());
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
            // Fast path (the only shape the COW scan produces — whole, unsliced REE batches): walk
            // the run-ends ONCE, emitting each run's value across its whole logical span. For an
            // unsliced array the logical index equals the physical run-end offset, so this yields
            // exactly the same `&str` per row as `run.get_physical_index(row)` — the row-wise form
            // below — without the per-row binary search.
            let run_ends = run.run_ends().values();
            let mut start = 0usize;
            for (physical, &end) in run_ends.iter().enumerate() {
                let end = usize::try_from(end).map_err(|_| {
                    DataFusionError::Internal("_file REE run-end is negative".to_string())
                })?;
                let value = values.value(physical);
                for _ in start..end {
                    out.push(value);
                }
                start = end;
            }
        } else {
            // Sliced REE array: the logical→physical mapping is offset-relative, so defer to
            // `get_physical_index` per row (still allocation-free). Behaviorally identical to the
            // fast path; kept separate because a sliced run-ends walk is easy to get subtly wrong.
            for row in 0..run.len() {
                out.push(values.value(run.get_physical_index(row)));
            }
        }
        return Ok(out);
    }
    Err(DataFusionError::Internal(format!(
        "unexpected _file column type: {:?}",
        col.data_type()
    )))
}

// =================================================================================================
// UPDATE
// =================================================================================================

/// `UPDATE … SET … WHERE` execution plan. Applies the `SET` assignments to the rows matching `WHERE`,
/// commits, and emits the updated-row count.
pub(crate) struct IcebergUpdateExec {
    table: Table,
    catalog: Arc<dyn Catalog>,
    /// The WHERE clause as a `PhysicalExpr`, or `None` to update every row.
    predicate: Option<Arc<dyn PhysicalExpr>>,
    /// The `SET` assignments: `(table-schema column index, new-value PhysicalExpr)`.
    assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
    mode: WriteMode,
    /// The §5 isolation level (resolved at plan time from `write.update.isolation-level`, default
    /// serializable — Java's per-operation default).
    isolation: IsolationLevel,
    table_schema: SchemaRef,
    count_schema: SchemaRef,
    plan_properties: PlanProperties,
}

impl IcebergUpdateExec {
    pub(crate) fn new(
        table: Table,
        catalog: Arc<dyn Catalog>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
        mode: WriteMode,
        isolation: IsolationLevel,
        table_schema: SchemaRef,
    ) -> Self {
        let count_schema = IcebergDeleteExec::make_count_schema();
        let plan_properties = IcebergDeleteExec::compute_properties(Arc::clone(&count_schema));
        Self {
            table,
            catalog,
            predicate,
            assignments,
            mode,
            isolation,
            table_schema,
            count_schema,
            plan_properties,
        }
    }
}

impl Debug for IcebergUpdateExec {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "IcebergUpdateExec(table={}, mode={:?})",
            self.table.identifier(),
            self.mode
        )
    }
}

impl DisplayAs for IcebergUpdateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "IcebergUpdateExec: table={}, mode={:?}",
            self.table.identifier(),
            self.mode
        )
    }
}

impl ExecutionPlan for IcebergUpdateExec {
    fn name(&self) -> &str {
        "IcebergUpdateExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
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
                "IcebergUpdateExec only has one partition, but got partition {partition}"
            )));
        }

        let table = self.table.clone();
        let catalog = Arc::clone(&self.catalog);
        let predicate = self.predicate.clone();
        let assignments = self.assignments.clone();
        let mode = self.mode;
        let isolation = self.isolation;
        let table_schema = Arc::clone(&self.table_schema);
        let count_schema = Arc::clone(&self.count_schema);

        let stream = futures::stream::once(async move {
            let updated = match mode {
                WriteMode::MergeOnRead => {
                    merge_on_read_update(
                        &table,
                        catalog.as_ref(),
                        predicate,
                        &assignments,
                        &table_schema,
                        isolation,
                    )
                    .await?
                }
                WriteMode::CopyOnWrite => {
                    copy_on_write_update(
                        &table,
                        catalog.as_ref(),
                        predicate,
                        &assignments,
                        &table_schema,
                        isolation,
                    )
                    .await?
                }
            };
            IcebergDeleteExec::make_count_batch(count_schema, updated)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.count_schema),
            stream,
        )))
    }
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

/// Rebuild a batch holding exactly the table columns (resolved BY NAME, in table-schema order) — the
/// schema the predicate/assignment `PhysicalExpr`s are bound to and the writer matches against.
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

/// Apply the `SET` assignments to `table_batch`, replacing each assigned column. When `mask` is `Some`,
/// only the masked-`true` rows take the new value (the rest keep the old) — used by copy-on-write where
/// the batch holds matching AND non-matching rows. When `None`, every row is updated (merge-on-read,
/// where the batch is already filtered to matching rows).
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
        // An assignment must not introduce NULLs into a REQUIRED (non-nullable) column — Parquet would
        // write the null and silently violate the Iceberg schema contract.
        let field = table_schema.field(*col_idx);
        if !field.is_nullable() && assigned.null_count() > 0 {
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

/// Merge-on-read UPDATE: position-delete the OLD matching rows and insert NEW rows carrying the updated
/// values, in one `RowDelta`. Returns the number of rows updated. Works for both partitioned and
/// unpartitioned tables: the NEW rows are routed through the partition-aware [`write_partitioned_data_files`]
/// helper, which computes partition values from the POST-assignment column values. Position deletes are
/// keyed by (data-file path, position) and are partition-agnostic, so the delete side is unchanged.
async fn merge_on_read_update(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
) -> DFResult<u64> {
    require_v2_for_merge_on_read(table)?;

    // The snapshot this DML's scan reads — the §5 `validate_from_snapshot` anchor
    // (`SparkPositionDeltaWrite.java` L245-249).
    let scan_snapshot_id = table.metadata().current_snapshot_id();

    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());
    projection.push(RESERVED_COL_NAME_POS.to_string());

    // Stream the scan batch-by-batch. Awaiting `try_next` / the data writer's `write` back-pressures
    // the scan (single-threaded poll) — no unbounded producer.
    let mut stream = table
        .scan()
        .select(projection)
        .build()
        .map_err(to_datafusion_error)?
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?;

    // The delete side still buffers the matched `(path, pos)` pairs (two small fields per updated row),
    // because `write_position_deletes` must group by `(spec_id, partition)` and sort `(path, pos)` and
    // the default scan interleaves files unordered — see MEDIUM-1. The NEW-row (data-file) side, by
    // contrast, streams straight into the writer per batch — its rows are never buffered.
    let mut pairs: Vec<(String, i64)> = Vec::new();
    let mut data_writer = StreamingDataFileWriter::try_new(table)?;
    while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
        let table_batch = table_column_batch(&batch, table_schema)?;
        let mask = match_mask(&predicate, &table_batch)?;
        if mask.true_count() == 0 {
            continue;
        }

        // Record the (path, pos) of every OLD matching row to position-delete.
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
                pairs.push((decode_file_path(file_col, row)?, pos_col.value(row)));
            }
        }

        // The matching rows, with the assignments applied (all of them match → no per-row mask).
        // Stream them straight into the data-file writer rather than buffering `new_rows`.
        let matching = filter_record_batch(&table_batch, &mask)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        let new_rows_batch = apply_assignments(&matching, assignments, table_schema, None)?;
        data_writer.write_batch(new_rows_batch).await?;
    }

    let updated = pairs.len() as u64;
    if updated == 0 {
        // No rows matched: no position deletes and no new data. The data writer was never fed a batch
        // (every batch had `true_count() == 0`), so `finish` produces no file — nothing to commit.
        let empty = data_writer.finish().await?;
        debug_assert!(empty.is_empty());
        return Ok(0);
    }

    // The DATA files the position deletes reference — the §5 `validate_data_files_exist` set
    // (`SparkPositionDeltaWrite.commit` L243, unconditional).
    let referenced_files: HashSet<String> = pairs.iter().map(|(path, _)| path.clone()).collect();

    // Position deletes MUST be grouped + sorted (path, pos) before writing — the whole `pairs` set is
    // required up front (MEDIUM-1). The data files, in contrast, were already streamed above; `finish`
    // just closes the writer. Both complete BEFORE the single commit below (commit-once atomicity).
    sort_position_delete_pairs(&mut pairs);
    let delete_files = write_position_deletes(table, &pairs).await?;
    let data_files = data_writer.finish().await?;

    // ENGINE_CONTRACT §5 row-delta recipe, MoR UPDATE row. Beyond the base (conflict filter +
    // files-exist + from-snapshot), UPDATE arms `validate_deleted_files` +
    // `validate_no_conflicting_delete_files` at BOTH isolation levels — the op READ rows to produce
    // its output, so a concurrent delete of those rows conflicts (Java `command == UPDATE || MERGE`,
    // `SparkPositionDeltaWrite.commit` L251-254 — deliberately NOT armed for DELETE). Serializable
    // adds the conflicting-data check (L256-258). `AlwaysTrue` = Java's AND of pushed filters when
    // nothing is pushed (L284-292).
    let tx = Transaction::new(table);
    let mut action = tx
        .row_delta()
        .add_data_files(data_files)
        .add_deletes(delete_files)
        .conflict_detection_filter(Predicate::AlwaysTrue)
        .validate_data_files_exist(referenced_files)
        .validate_deleted_files()
        .validate_no_conflicting_delete_files();
    if let Some(snapshot_id) = scan_snapshot_id {
        action = action.validate_from_snapshot(snapshot_id);
    }
    if isolation == IsolationLevel::Serializable {
        action = action.validate_no_conflicting_data_files();
    }
    action
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(updated)
}

/// Copy-on-write UPDATE: **file-level** rewrite — scan every live row projecting the table columns
/// PLUS the reserved `_file` path, identify which source data files contain at least one updated row
/// (the "affected" set), rewrite only those files in full (matched rows take the new values; rows of
/// the same file that did NOT match are carried unchanged), and commit a `OverwriteFiles` that deletes
/// the affected source paths and adds the rewritten files. Unaffected data files are left completely
/// untouched.
///
/// Works for BOTH partitioned and unpartitioned tables. When the SET expression changes a
/// partition-key column, the rewritten row is routed to its NEW partition automatically because
/// `write_partitioned_data_files` computes partition values from the post-assignment column values.
async fn copy_on_write_update(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
) -> DFResult<u64> {
    // The snapshot this DML's scan reads — the §5 `validate_from_snapshot` anchor (`SparkWrite.java`
    // L470-472 / L493-495).
    let scan_snapshot_id = table.metadata().current_snapshot_id();
    // 1. Scan EVERY live row projecting the table columns PLUS `_file` (not `_pos` — COW does not
    //    need positions). We do NOT push the filter into the scan — Iceberg pushdown is inexact (see
    //    the module note); the exact `PhysicalExpr` evaluation here is the correctness contract.
    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());

    let batches: Vec<RecordBatch> = table
        .scan()
        .select(projection)
        .build()
        .map_err(to_datafusion_error)?
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .try_collect()
        .await
        .map_err(to_datafusion_error)?;

    // 2. Pass 1 — affected-file detection. A source file is AFFECTED iff at least one of its rows
    //    matches the predicate (or the predicate is None → all rows match → all files affected).
    //    Also counts total updated rows for the return value.
    let mut updated: u64 = 0;
    let mut affected: HashSet<String> = HashSet::new();
    // M7: cache the per-batch WHERE match mask computed here so pass 2 can REUSE it (filtered to the
    // affected rows) instead of re-evaluating the predicate a second time. `match_mask` already
    // collapses NULL→false (three-valued logic), so the cached mask is the final 2-valued mask.
    let mut batch_masks: Vec<BooleanArray> = Vec::with_capacity(batches.len());

    for batch in &batches {
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("update scan missing _file column".to_string())
            })?;
        let table_batch = table_column_batch(batch, table_schema)?;
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
        batch_masks.push(mask);
    }

    // 3. No updated rows → no-op (avoid a pointless rewrite of unchanged data).
    if updated == 0 {
        return Ok(0);
    }

    // 4. Pass 2 — build rewrite content for affected files only.
    //    For each batch: filter down to rows whose source file is in the affected set, then apply
    //    the assignments with the per-row match mask so that:
    //      * matched rows (WHERE = TRUE) take the new SET values
    //      * other rows of the SAME affected file keep their original values (carried unchanged)
    //    Rows from unaffected files are NOT included — their source files are untouched.
    let mut rewritten_batches: Vec<RecordBatch> = Vec::new();

    for (batch_idx, batch) in batches.iter().enumerate() {
        let num_rows = batch.num_rows();
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("update scan missing _file column".to_string())
            })?;

        // Build table-column sub-batch (rows from the FULL batch including unaffected-file rows).
        let table_batch = table_column_batch(batch, table_schema)?;

        // Keep-mask: only rows whose source file is in the affected set.
        let paths = decode_file_paths_batch(file_col)?;
        let keep_affected: BooleanArray = (0..num_rows)
            .map(|row| affected.contains(paths[row]))
            .collect();

        // Filter down to affected-file rows (table columns only, no _file).
        let affected_batch = filter_record_batch(&table_batch, &keep_affected)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        if affected_batch.num_rows() == 0 {
            continue;
        }

        // M7: the per-row WHERE match mask within the affected rows is exactly the pass-1 mask for
        // this batch FILTERED by the same `keep_affected` predicate — `match_mask` is row-wise and
        // arrow `filter` preserves row order, so this equals re-evaluating the predicate over
        // `affected_batch` (proven by `test_m7_filtered_mask_equals_reeval`), without a second
        // predicate evaluation.
        let affected_match_mask =
            datafusion::arrow::compute::filter(&batch_masks[batch_idx], &keep_affected)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        let affected_match_mask = affected_match_mask
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::Internal("filtered match mask is not boolean".to_string())
            })?
            .clone();

        // Apply assignments: matched rows take new values; non-matched rows keep old values.
        let rewritten = apply_assignments(
            &affected_batch,
            assignments,
            table_schema,
            Some(&affected_match_mask),
        )?;
        rewritten_batches.push(rewritten);
    }

    // 5. Write rewritten content via the partition-aware TaskWriter. Routes each row to its correct
    //    partition by the POST-assignment column values — a partition-key-changing UPDATE automatically
    //    moves the row to the new partition file.
    let new_files = write_partitioned_data_files(table, &rewritten_batches).await?;

    // 6. Commit: delete the affected source files, add the rewritten files. Full-metadata removals
    //    (`delete_data_files`, NOT `overwrite_by_row_filter` — unaffected files stay in place, and NOT
    //    bare paths — the §5 conflicting-deletes check needs partition + metrics). Same §5 CoW recipe
    //    as DELETE: Java's isolation `switch` does not branch on the command (`SparkWrite.java`
    //    L448-456) — deletes-conflict at BOTH levels, data-conflict under serializable.
    let removed_data_files = resolve_affected_data_files(table, &affected).await?;
    let tx = Transaction::new(table);
    let mut action = tx
        .overwrite_files()
        .delete_data_files(removed_data_files)
        .add_files(new_files)
        .conflict_detection_filter(Predicate::AlwaysTrue)
        .validate_no_conflicting_deletes();
    if let Some(snapshot_id) = scan_snapshot_id {
        action = action.validate_from_snapshot(snapshot_id);
    }
    if isolation == IsolationLevel::Serializable {
        action = action.validate_no_conflicting_data();
    }
    action
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(updated)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{ArrayRef, Int32Array, RunArray, StringArray};
    use datafusion::arrow::datatypes::Int32Type;

    use super::{
        IsolationLevel, decode_file_path, decode_file_paths_batch, sort_position_delete_pairs,
    };

    /// `decode_file_paths_batch` must produce, for every row, EXACTLY the string
    /// `decode_file_path` would — for a plain `StringArray`, for a run-end-encoded `_file` column
    /// (the shape the COW scan produces, with F ≪ R distinct values and duplicate runs), and for a
    /// SLICED REE array (the offset≠0 fallback path). This pins the H8 per-run decode optimization
    /// to byte-identical per-row results — the correctness contract for COW DELETE/UPDATE
    /// affected-file detection and keep-masks.
    fn assert_batch_matches_per_row(col: &ArrayRef) {
        let batch = decode_file_paths_batch(col).expect("batch decode");
        assert_eq!(batch.len(), col.len(), "one decoded path per row");
        for (row, decoded) in batch.iter().enumerate() {
            let per_row = decode_file_path(col, row).expect("per-row decode");
            assert_eq!(
                *decoded, per_row,
                "row {row}: batch decode must equal per-row decode"
            );
        }
    }

    #[test]
    fn test_decode_file_paths_batch_plain_string_array() {
        let col: ArrayRef = Arc::new(StringArray::from(vec![
            "s3://b/a.parquet",
            "s3://b/a.parquet",
            "s3://b/c.parquet",
        ]));
        assert_batch_matches_per_row(&col);
    }

    #[test]
    fn test_decode_file_paths_batch_ree_with_runs() {
        // run-end-encoded: values [a, b, a] over logical rows with runs of length 3, 1, 2.
        let run_ends = Int32Array::from(vec![3, 4, 6]);
        let values = StringArray::from(vec!["f/a.parquet", "f/b.parquet", "f/a.parquet"]);
        let ree = RunArray::<Int32Type>::try_new(&run_ends, &values).expect("build REE");
        let col: ArrayRef = Arc::new(ree);
        // Sanity: distinct runs, duplicate value across non-adjacent runs.
        assert_eq!(col.len(), 6);
        assert_batch_matches_per_row(&col);
    }

    #[test]
    fn test_decode_file_paths_batch_ree_single_run() {
        let run_ends = Int32Array::from(vec![5]);
        let values = StringArray::from(vec!["only/file.parquet"]);
        let ree = RunArray::<Int32Type>::try_new(&run_ends, &values).expect("build REE");
        let col: ArrayRef = Arc::new(ree);
        assert_batch_matches_per_row(&col);
    }

    /// M7 correctness property: the per-row WHERE match mask over the AFFECTED sub-batch equals the
    /// full-batch match mask FILTERED by the same affected-keep mask. The COW UPDATE pass-2 reuse
    /// relies on exactly this identity to avoid re-evaluating the predicate; `arrow::compute::filter`
    /// preserving row order is what makes it hold. (`match_mask` is a row-wise pure function of the
    /// table batch, so filtering the input batch then evaluating == evaluating then filtering.)
    #[test]
    fn test_m7_filtered_mask_equals_reeval() {
        use datafusion::arrow::array::BooleanArray;
        use datafusion::arrow::compute::filter;

        // Full-batch match mask (what pass 1 cached) and the affected-keep mask.
        let full_match = BooleanArray::from(vec![true, false, true, true, false, true]);
        let keep_affected = BooleanArray::from(vec![true, true, false, true, true, false]);

        // What pass 2 now computes: filter the cached mask by keep.
        let reused = filter(&full_match, &keep_affected).expect("filter mask");
        let reused = reused.as_any().downcast_ref::<BooleanArray>().unwrap();

        // The reference (the pre-M7 form): the match values at the KEPT rows, in order.
        let reference: Vec<bool> = (0..full_match.len())
            .filter(|&i| keep_affected.value(i))
            .map(|i| full_match.value(i))
            .collect();
        let reference = BooleanArray::from(reference);

        assert_eq!(
            reused, &reference,
            "filtered cached mask must equal the affected-rows match mask, in order"
        );
        // Rows kept: 0,1,3,4 → their match values: true,false,true,false.
        assert_eq!(
            reference,
            BooleanArray::from(vec![true, false, true, false])
        );
    }

    #[test]
    fn test_decode_file_paths_batch_sliced_ree_offset_fallback() {
        // Slice a REE array so offset != 0 exercises the get_physical_index fallback branch.
        let run_ends = Int32Array::from(vec![3, 4, 7]);
        let values = StringArray::from(vec!["f/a.parquet", "f/b.parquet", "f/c.parquet"]);
        let ree = RunArray::<Int32Type>::try_new(&run_ends, &values).expect("build REE");
        // Logical rows: a a a b c c c — take rows [2,5) → a b c c.
        let sliced = ree.slice(2, 3);
        let col: ArrayRef = Arc::new(sliced);
        assert_eq!(col.len(), 3);
        assert_batch_matches_per_row(&col);
    }

    /// MEDIUM-1 (H-ORDER), deterministic seam test: `sort_position_delete_pairs` — the sort applied at
    /// every MoR position-delete write site (`merge_on_read_delete`, `merge_on_read_update`, and the
    /// per-partition-group path in `write_position_deletes`) — MUST produce ascending `(file_path,
    /// pos)` order for ANY input. The default concurrent scan interleaves files unordered, so the
    /// collected pairs arrive out of order; this pins the spec-required order independent of scan
    /// interleaving (which an integration test cannot pin deterministically).
    ///
    /// MUTATION PROOF: turn `sort_position_delete_pairs` into a no-op (delete the `pairs.sort()`) → this
    /// test goes RED (the deliberately-unsorted input stays unsorted).
    #[test]
    fn test_sort_position_delete_pairs_orders_by_path_then_pos() {
        // Deliberately unsorted: files interleaved (b before a), positions descending within a file —
        // exactly the shape a concurrent, cross-file scan produces before the sort restores order.
        let mut pairs: Vec<(String, i64)> = vec![
            ("s3://b/file_b.parquet".to_string(), 5),
            ("s3://b/file_a.parquet".to_string(), 2),
            ("s3://b/file_b.parquet".to_string(), 1),
            ("s3://b/file_a.parquet".to_string(), 0),
            ("s3://b/file_a.parquet".to_string(), 10),
        ];
        sort_position_delete_pairs(&mut pairs);
        let expected: Vec<(String, i64)> = vec![
            ("s3://b/file_a.parquet".to_string(), 0),
            ("s3://b/file_a.parquet".to_string(), 2),
            ("s3://b/file_a.parquet".to_string(), 10),
            ("s3://b/file_b.parquet".to_string(), 1),
            ("s3://b/file_b.parquet".to_string(), 5),
        ];
        assert_eq!(
            pairs, expected,
            "position-delete pairs must be sorted ascending by (file_path, pos) — spec order"
        );
        // Independent, form-agnostic check that it is globally non-decreasing (catches any sort that
        // is not a true (path, pos) ascending order).
        for window in pairs.windows(2) {
            assert!(
                window[0] <= window[1],
                "pairs must be non-decreasing by (file_path, pos): {:?} then {:?}",
                window[0],
                window[1]
            );
        }
    }

    /// §5 isolation-level parse parity with Java `IsolationLevel.fromName` (1.10.0
    /// `core/IsolationLevel.java`): case-INSENSITIVE accept (`valueOf(levelName.toUpperCase(ENGLISH))`)
    /// and a LOUD `"Invalid isolation level: <name>"` error on an unknown name — never a silent
    /// default. (Ledger P14a; MUTATION M7: make the parse default instead of erroring → RED.)
    #[test]
    fn test_isolation_level_parse_java_parity() {
        // Case-insensitive accepts, both levels (Java upper-cases before valueOf).
        for accepted in ["serializable", "SERIALIZABLE", "Serializable"] {
            assert_eq!(
                IsolationLevel::parse(accepted).expect("parse serializable spelling"),
                IsolationLevel::Serializable,
                "'{accepted}' must parse as serializable"
            );
        }
        for accepted in ["snapshot", "SNAPSHOT", "Snapshot"] {
            assert_eq!(
                IsolationLevel::parse(accepted).expect("parse snapshot spelling"),
                IsolationLevel::Snapshot,
                "'{accepted}' must parse as snapshot"
            );
        }

        // Unknown name → loud error carrying Java's message shape and the offending name.
        let err = IsolationLevel::parse("read-committed")
            .expect_err("an unknown isolation level must fail loud, not default");
        assert!(
            err.to_string()
                .contains("Invalid isolation level: read-committed"),
            "error must carry Java's message + the offending name, got: {err}"
        );
        // 'none' is NOT a row-level isolation level (Java has no way to disable row-level
        // validation; absence-of-option exists only on the INSERT OVERWRITE write path).
        assert!(
            IsolationLevel::parse("none").is_err(),
            "'none' must be rejected for row-level operations"
        );
    }
}
