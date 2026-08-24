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
//! whole pair set before writing (the default scan interleaves files unordered).
//!
//! The **copy-on-write** paths STREAM both of their passes (H7-S2). COW is inherently two-pass — a
//! source file is "affected" the moment *any* of its rows matches, possibly the last row of the last
//! batch, so the affected set must be COMPLETE before the first survivor may be emitted — but neither
//! pass buffers rows. Pass 1 streams the scan and retains only `affected: HashSet<String>` (one entry
//! per affected FILE) plus a row counter; pass 2 RE-SCANS the same snapshot and feeds each batch's
//! rewrite rows straight into [`StreamingDataFileWriter`]. The DML path's own contribution to peak is
//! O(#affected files) + one batch + the writer's own buffers — but that is NOT the total: the scan
//! underneath holds up to `concurrency_limit_data_files` (default `num_cpus`) in-flight Parquet row
//! groups plus per-file task state, and on a many-core host that term DOMINATES the absolute peak. It
//! is bounded by concurrency, not by row count, which is why it cancels out of the marginal assertion
//! in `tests/cow_memory_bound.rs`. **The named cost is a second full read of the live data** — the accepted
//! price of bounded memory, and one extra `ScanReport` per statement for catalogs with a metrics
//! reporter installed. (Two shapes skip pass 2 outright: a zero-match DML, and a predicate-less
//! `DELETE FROM t`, whose pass 2 is provably empty because every row is deleted.)
//! Restricting pass 2 to the affected files is not possible through the
//! public scan API today (`_file` is a reserved metadata column, not a pushdown-able predicate term);
//! that is a follow-up. Both passes are snapshot-consistent by construction: `Table` is a frozen
//! handle (`metadata` is a plain `TableMetadataRef` with no interior mutability, and the only mutator
//! takes `mut self` by value), and an unpinned `TableScanBuilder::build()` resolves from that frozen
//! metadata, never a fresh catalog read — so two `table.scan()` calls on one handle resolve the
//! IDENTICAL snapshot regardless of concurrent commits. Both scans additionally pin the snapshot id
//! explicitly; that is documentation of the invariant, NOT a fix for a live bug.
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
//!   * **Streaming** — both merge-on-read and copy-on-write stream their scans (see *Memory* above).
//!     Neither is O(1): MoR holds one `(path, pos)` pair per matched row, COW holds one path per
//!     affected file. Writer-side buffering (a fanout `TaskWriter` holds one open writer per
//!     partition) is a separate, still-unbounded axis — see the QB writer-bounds unit.
//!
//! The plan emits a single `UInt64` `count` row (rows affected), per DataFusion's DML contract.

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
use iceberg::delete_vector::load_delete_vector;
use iceberg::expr::Predicate;
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, MetricsConfig, PartitionKey, Struct,
    referenced_data_file_location,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::base_writer::deletion_vector_writer::{
    DVFileWriter, DVWriteResult, PreviousDeletes,
};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig, position_delete_writer_properties,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
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
    plan_properties: Arc<PlanProperties>,
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

    fn compute_properties(schema: SchemaRef) -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ))
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

/// The delete-file kind a table's format version requires for merge-on-read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeOnReadDeleteKind {
    /// V2: Parquet position-delete files.
    PositionDeletes,
    /// V3: Puffin deletion vectors. The spec forbids NEW position-delete files at this version.
    DeletionVectors,
}

/// Resolves which delete-file kind this table takes.
///
/// # Errors
///
/// `NotImplemented` for a V1 table, which has no delete files of any kind.
///
/// # Notes
///
/// Call this BEFORE any I/O. A format rejection raised at commit time would orphan an already
/// written delete or data file.
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

/// Writes the merge-on-read delete files for `pairs` in the kind this table takes.
///
/// Returns `(files to add, files the commit must remove)`. The removal half is only ever non-empty
/// on the V3 path, where a merged DV supersedes the file-scoped one it absorbed.
///
/// # Notes
///
/// `pairs` must already be sorted by `(path, pos)`; the V2 writer needs that order, and the caller
/// sorts once for both paths.
async fn write_merge_on_read_deletes(
    table: &Table,
    kind: MergeOnReadDeleteKind,
    pairs: &[(String, i64)],
) -> DFResult<(Vec<DataFile>, Vec<DataFile>)> {
    match kind {
        MergeOnReadDeleteKind::PositionDeletes => {
            Ok((write_position_deletes(table, pairs).await?, Vec::new()))
        }
        MergeOnReadDeleteKind::DeletionVectors => {
            let result = write_deletion_vectors(table, pairs).await?;
            Ok((result.delete_files, result.rewritten_delete_files))
        }
    }
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
    let delete_kind = merge_on_read_delete_kind(table)?;
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

    // The DATA files the position deletes reference — the §5 `validate_data_files_exist` set. Java
    // enables this check UNCONDITIONALLY for every command, DELETE included
    // (`SparkPositionDeltaWrite.commit` L243): a referenced file compacted or rewritten away by a
    // concurrent commit would silently lose these deletes.
    let referenced_files: HashSet<String> = pairs.iter().map(|(path, _)| path.clone()).collect();

    // Write ALL delete files the writer produced and commit EVERY one of them.
    let (delete_files, superseded_delete_files) =
        write_merge_on_read_deletes(table, delete_kind, &pairs).await?;

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
    // A merged DV supersedes the one it absorbed. Leaving that one live would double-count its
    // positions, and V3 allows only one DV per data file.
    if !superseded_delete_files.is_empty() {
        action = action.remove_deletes_many(superseded_delete_files);
    }
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

/// Open ONE copy-on-write scan stream: every live row, projecting the table columns PLUS the reserved
/// `_file` path (not `_pos` — COW does not need positions).
///
/// We do NOT push the filter into the scan — Iceberg pushdown is inexact (see the module note); the
/// exact `PhysicalExpr` evaluation in the caller is the correctness contract.
///
/// **On the explicit snapshot pin.** `scan_snapshot_id` is the caller's `current_snapshot_id()`, which
/// is exactly what an UNPINNED `build()` would resolve from the same frozen `Table` handle
/// (`scan/mod.rs`: `metadata().current_snapshot()`, never a fresh catalog read). Passing it is
/// therefore a **no-op today**, not a bug fix: it documents at the call site that the two COW passes
/// read one snapshot, and keeps that true if `Table` ever gains a refresh path. `None` (a snapshotless
/// table) is left unpinned, which yields the same empty scan.
async fn cow_scan_stream(
    table: &Table,
    table_schema: &SchemaRef,
    scan_snapshot_id: Option<i64>,
) -> DFResult<iceberg::scan::ArrowRecordBatchStream> {
    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());

    let mut builder = table.scan().select(projection);
    if let Some(snapshot_id) = scan_snapshot_id {
        builder = builder.snapshot_id(snapshot_id);
    }
    // Awaiting `try_next()` on the returned stream polls the scan only as batches are consumed, so
    // the scan is naturally back-pressured — no unbounded producer.
    builder
        .build()
        .map_err(to_datafusion_error)?
        .to_arrow()
        .await
        .map_err(to_datafusion_error)
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
///
/// **Streaming (H7-S2).** Neither pass buffers rows. Pass 1 streams the scan and retains only the
/// affected-file path set and the deleted-row count; pass 2 RE-SCANS the same snapshot and streams
/// each batch's survivors straight into [`StreamingDataFileWriter`]. The cost is a second full read
/// of the live data — see the module-level *Memory* note.
async fn copy_on_write_delete(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
) -> DFResult<u64> {
    // The snapshot this DML's scan reads — the §5 `validate_from_snapshot` anchor (Java sets it only
    // when the scan captured one: `SparkWrite.java` L470-472 / L493-495). Both passes below pin this
    // same snapshot, so they read the identical row set.
    let scan_snapshot_id = table.metadata().current_snapshot_id();

    // 1. Pass 1 — affected-file detection, STREAMED. A source file is AFFECTED iff at least one of its
    //    rows matches the predicate (or the predicate is None → all rows deleted → all files affected).
    //    Also counts total deleted rows for the return value. The ONLY state retained across the pass
    //    is `affected` (one path per affected FILE) and the counter — no rows are buffered.
    let mut stream = cow_scan_stream(table, table_schema, scan_snapshot_id).await?;
    let mut deleted: u64 = 0;
    let mut affected: HashSet<String> = HashSet::new();

    while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("delete scan missing _file column".to_string())
            })?;
        // Table-column-only sub-batch for predicate evaluation (by name, robust to scan ordering).
        let table_batch = table_column_batch(&batch, table_schema)?;
        // `match_mask` already collapses NULL → false (SQL three-valued logic: a NULL predicate
        // result does NOT delete the row) and returns all-true for `DELETE FROM t` (no predicate).
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

    // Pass 1's stream is EXHAUSTED but its scan state (plan context, task state, channels) would live
    // to the end of the function body if it were merely shadowed by pass 2's binding — Rust drops a
    // shadowed value at scope end, not at the shadowing point. Release it explicitly so the peak really
    // is "one scan + one batch", as the module note claims.
    drop(stream);

    // 2. No deleted rows → no-op (avoid a pointless snapshot). This now also skips the SECOND scan
    //    entirely — a zero-match DELETE reads the table exactly once.
    if deleted == 0 {
        return Ok(0);
    }

    // 3. Pass 2 — RE-SCAN the same snapshot and stream the survivors of AFFECTED files straight into
    //    the writer. Rows from unaffected files are left in place (their source files are unchanged).
    //    Per batch, a row is kept iff it is (a) NOT deleted AND (b) from an affected file — those are
    //    exactly the rows that need a new home. Nothing is accumulated: each filtered batch is handed
    //    to the writer and dropped.
    //
    //    The affected set is complete before any survivor is emitted, which is why COW needs two
    //    passes at all (a file becomes affected on its LAST row just as easily as its first).
    //
    //    `DELETE FROM t` (no predicate) is short-circuited: every row is deleted, so pass 2's keep-mask
    //    is all-false for every batch and the whole table would be re-read to produce nothing. The
    //    result is EXACT, not an approximation — with `predicate == None`, `match_mask` is all-true by
    //    construction, so `!deleted && affected.contains(..)` cannot be true for any row.
    let new_files = if predicate.is_none() {
        Vec::new()
    } else {
        let mut stream = cow_scan_stream(table, table_schema, scan_snapshot_id).await?;
        // The writer is built on the FIRST batch that actually has survivors, not up front. That keeps
        // this path byte-identical to the pre-H7-S2 form, where `write_partitioned_data_files` returned
        // early on an empty survivor slice and never ran `DefaultLocationGenerator::new` /
        // `PartitionValueCalculator::try_new` — so a DELETE that fully empties every affected file
        // still cannot fail in a constructor it never needed. (COW UPDATE has no such case: an affected
        // file always yields rewrite rows, so its writer was always constructed.)
        let mut data_writer: Option<StreamingDataFileWriter> = None;

        while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
            let num_rows = batch.num_rows();
            let file_col = batch
                .column_by_name(RESERVED_COL_NAME_FILE)
                .ok_or_else(|| {
                    DataFusionError::Internal("delete scan missing _file column".to_string())
                })?;
            let table_batch = table_column_batch(&batch, table_schema)?;
            // Re-evaluated (not cached from pass 1): pass 2 is a fresh scan, so no per-batch state from
            // pass 1 could be aligned to it. Same row-wise function, same rows ⇒ same mask.
            let delete_mask = match_mask(&predicate, &table_batch)?;

            let paths = decode_file_paths_batch(file_col)?;
            let keep: BooleanArray = (0..num_rows)
                .map(|row| !delete_mask.value(row) && affected.contains(paths[row]))
                .collect();
            if keep.true_count() == 0 {
                // Nothing to rewrite from this batch — leave the writer uncreated.
                continue;
            }

            let surviving = filter_record_batch(&table_batch, &keep)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
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

        // 4. Close the writer. When there were no survivors to rewrite (e.g. every affected file was
        //    fully deleted) no writer was ever built and this yields an empty Vec — no empty data file
        //    is committed, matching the previous buffering form's `batches.is_empty()` contract.
        match data_writer {
            Some(writer) => writer.finish().await?,
            None => Vec::new(),
        }
    };

    // 5. Commit: delete the affected source files, add the rewritten files. The removals carry FULL
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
        // Always configure the default partition spec so an unpartitioned build(None) stamps the
        // real default_spec_id (post–DROP PARTITION FIELD may be non-zero empty), never fabricated 0
        // (C5-L-001 / C6-L-001 DATA dual of BUG-001). PartitionKey still wins when present.
        let builder = DataFileWriterBuilder::new(rolling)
            .with_partition_spec(partition_spec.as_ref().clone());

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

/// Write REAL parquet position-delete file(s) from sorted `(data_file_path, position)` pairs via the
/// production `PositionDeleteFileWriter`. Returns EVERY file the (rolling) writer produced — a large
/// DELETE may roll into more than one file, and ALL of them must be committed or the deletes in the
/// dropped files would be silently lost (rows resurrected on the next scan).
///
/// **Partition-aware.** Position-delete files are associated with the `(spec_id, partition)` of the
/// DATA file they delete from — the Iceberg commit validates that the delete file's partition matches the
/// registered spec for `partition_spec_id`.
///
/// **Fast path (never-evolved empty-spec tables only).** When the table has exactly one partition
/// spec AND that spec has **zero fields**, every data file carries an empty partition tuple and we
/// write a single delete file stamped via `with_partition_spec(default)` (so the real spec id is
/// kept — never a hard-coded 0). This is **not** the same as "the default spec is unpartitioned":
/// after `DROP PARTITION FIELD` / `updateSpec().removeField(...)` the default becomes unpartitioned
/// while older data files still carry their original `(spec_id, partition)`. Stamping those deletes
/// with fabricated `None`/spec-0 makes the read-side attach miss and resurrects rows (BUG-001).
/// All-Void single-spec tables also skip the fast path (need null-tuple arity). Multi-spec and
/// partitioned shapes always walk manifests and stamp each group with
/// `PartitionKey::new(data_file_spec, schema, partition)`.
///
/// For (still-)partitioned default specs:
///
/// 1. The current snapshot manifests are scanned once to build a `path → (spec_id, Struct)` map.
/// 2. The `(path, pos)` pairs are grouped by their data file's `(spec_id, Struct)`.
/// 3. One position-delete file is written per group, stamped with that group's `PartitionKey`.
///
/// This mirrors Java `PositionDeleteWriter` which always carries a per-data-file `PartitionKey` and
/// `RewritePositionDeleteFiles` which groups delete files by `(spec_id, partition)`.
/// Whether position deletes may take the empty-partition fast path.
///
/// Option A (BUG-001), refined for all-Void arity (C1-L-002): only when the table has **exactly
/// one** partition spec AND that spec has **zero fields** (truly unpartitioned).  
/// - Multi-spec tables whose *default* is unpartitioned after evolution still have partitioned
///   data under older specs → manifest walk.  
/// - Single-spec all-Void (`is_unpartitioned()` true but non-empty fields) needs a null-tuple
///   `PartitionKey` matching the void arity, not `None`/empty → also takes the walk.
pub(crate) fn position_delete_unpartitioned_fast_path(
    spec_count: usize,
    default_field_count: usize,
) -> bool {
    spec_count == 1 && default_field_count == 0
}

async fn write_position_deletes(table: &Table, pairs: &[(String, i64)]) -> DFResult<Vec<DataFile>> {
    let config = PositionDeleteWriterConfig::new().map_err(to_datafusion_error)?;
    let metadata = table.metadata();
    let default_spec = metadata.default_partition_spec();
    let schema = metadata.current_schema();

    // Never-evolved empty-spec tables only: one delete file under that sole empty spec.
    // Multi-spec / all-Void / partitioned → manifest walk + per-group PartitionKey (C1-L-001).
    if position_delete_unpartitioned_fast_path(
        metadata.partition_specs_iter().len(),
        default_spec.fields().len(),
    ) {
        // Stamp the real default spec id (not a hard-coded 0): build with_partition_spec so
        // resolve_partition_spec_id does not fabricate DEFAULT_PARTITION_SPEC_ID when the sole
        // empty spec happens to carry a non-zero id.
        return write_position_deletes_for_partition(
            table,
            &config,
            pairs,
            None,
            Some(default_spec.as_ref().clone()),
        )
        .await;
    }

    // Partitioned: stamp each delete file with the SAME spec + partition as the data file it
    // deletes.
    let path_to_partition = live_data_file_partitions(table).await?;

    // Group pairs by (spec_id, partition) — every pair's data file must be live in the snapshot the
    // map was built from.
    let path_to_partition: HashMap<String, (i32, Struct)> = path_to_partition
        .into_iter()
        .map(|(path, (spec_id, partition, _))| (path, (spec_id, partition)))
        .collect();
    let groups = group_pairs_by_partition(pairs, &path_to_partition)?;

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
        // Always carry the data file's own (spec, partition) — including empty/unpartitioned and
        // all-Void null tuples. `partition_key = None` without with_partition_spec would fabricate
        // spec_id 0 and under-attach or fail commit after DROP PARTITION FIELD (C1-L-001).
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

    // Each group above is non-empty and `write_position_deletes_for_partition` guarantees it
    // produced at least one file, so `all_delete_files` is non-empty whenever `pairs` was.
    Ok(all_delete_files)
}

/// Maps every live data file of the current snapshot to its `(spec_id, partition)`.
///
/// # Notes
///
/// Both delete write paths stamp from this, so a position-delete file and a deletion vector
/// covering the same data file cannot disagree about its partition.
async fn live_data_file_partitions(
    table: &Table,
) -> DFResult<HashMap<String, (i32, Struct, Option<i64>)>> {
    let metadata = table.metadata();
    let mut path_to_partition: HashMap<String, (i32, Struct, Option<i64>)> = HashMap::new();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(path_to_partition);
    };
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
                    .or_insert_with(|| {
                        (
                            df.partition_spec_id(),
                            df.partition().clone(),
                            entry.sequence_number(),
                        )
                    });
            }
        }
    }
    Ok(path_to_partition)
}

/// The live delete files of the current snapshot, split by whether they are deletion vectors.
struct LiveDeletes {
    /// Puffin DVs, keyed by the data file each covers.
    dv_by_data_file: HashMap<String, DataFile>,
    /// Non-Puffin position deletes, as `(referenced_data_file, spec_id, partition, sequence)`.
    /// The reference comes from [`referenced_data_file_location`], the same derivation the scan
    /// uses — a delete with equal `file_path` bounds names its data file even with the field
    /// unset, which is how virtually every Java-written file-granularity delete is recognised.
    legacy_position_deletes: Vec<(Option<String>, i32, Struct, Option<i64>)>,
}

/// Reads the current snapshot's delete manifests once.
///
/// # Notes
///
/// V3 allows at most one DV per data file, so `dv_by_data_file` is unambiguous. A second delete on
/// a data file must merge that DV and supersede it; leaving it live would double-count the
/// positions.
async fn live_delete_vectors_by_data_file(table: &Table) -> DFResult<LiveDeletes> {
    let metadata = table.metadata();
    let mut live = LiveDeletes {
        dv_by_data_file: HashMap::new(),
        legacy_position_deletes: Vec::new(),
    };
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(live);
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await
        .map_err(to_datafusion_error)?;

    for manifest_entry in manifest_list.entries() {
        if manifest_entry.content != iceberg::spec::ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_entry
            .load_manifest(table.file_io())
            .await
            .map_err(to_datafusion_error)?;
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            if df.content_type() != iceberg::spec::DataContentType::PositionDeletes {
                continue;
            }
            if df.file_format() == DataFileFormat::Puffin {
                if let Some(referenced) = df.referenced_data_file() {
                    live.dv_by_data_file.insert(referenced, df.clone());
                }
            } else {
                live.legacy_position_deletes
                    .push(legacy_position_delete_entry(df, entry.sequence_number()));
            }
        }
    }
    Ok(live)
}

/// Reduces a live non-Puffin position delete to what the applicability test needs.
///
/// # Notes
///
/// The reference is [`referenced_data_file_location`], not the raw `referenced_data_file` field.
/// Java's `PositionDeleteWriter.close()` never sets that field — it leaves equal `file_path`
/// bounds — so reading the field alone treats virtually every Java-written file-granularity delete
/// as partition-scoped.
fn legacy_position_delete_entry(
    delete_file: &DataFile,
    sequence_number: Option<i64>,
) -> (Option<String>, i32, Struct, Option<i64>) {
    (
        referenced_data_file_location(delete_file),
        delete_file.partition_spec_id(),
        delete_file.partition().clone(),
        sequence_number,
    )
}

/// Whether a live non-Puffin position delete still applies to a data file.
///
/// Args mirror the commit door's own test (`RowDeltaAction::validate_fresh_dvs_only`), so the
/// pre-IO refusal and the commit-time rejection cannot disagree about what "covers" means.
///
/// # Notes
///
/// `delete.0` is [`referenced_data_file_location`], not the raw field: a delete with equal
/// `file_path` bounds names its data file even with the field unset, and that is how virtually
/// every Java-written file-granularity delete is recognised. A named delete is matched on PATH
/// alone — the partition it happens to be stamped with is irrelevant, exactly as the scan does.
/// An unknown sequence number errs toward "applies", so the caller refuses rather than corrupts.
fn legacy_position_delete_applies(
    delete: &(Option<String>, i32, Struct, Option<i64>),
    data_file_path: &str,
    data_spec_id: i32,
    data_partition: &Struct,
    data_seq: Option<i64>,
) -> bool {
    let (referenced, delete_spec_id, delete_partition, delete_seq) = delete;
    let scope_matches = match referenced {
        Some(referenced) => referenced == data_file_path,
        // Partition-scoped: covers every data file sharing its (spec_id, partition).
        None => *delete_spec_id == data_spec_id && delete_partition == data_partition,
    };
    scope_matches
        && match (delete_seq, data_seq) {
            (Some(delete_seq), Some(data_seq)) => *delete_seq >= data_seq,
            _ => true,
        }
}

/// Writes the deletion vectors for `pairs` — the V3 merge-on-read delete output.
///
/// One Puffin file carries one `deletion-vector-v1` blob per data file touched, so unlike the V2
/// path there is no per-partition grouping: the writer splits by referenced data file itself. Each
/// position still carries its data file's own `PartitionKey`, so a DV is stamped with the spec and
/// partition of the file it covers.
///
/// # Errors
///
/// Fails when a pair's data file is not live in the current snapshot, when its spec is unknown, or
/// when an existing DV cannot be read back.
///
/// # Notes
///
/// A data file that already has a DV has it loaded and merged, and the superseded DV comes back in
/// `rewritten_delete_files` for the commit to remove. V3 allows only one DV per data file, so
/// skipping that merge would leave two live DVs covering one file.
async fn write_deletion_vectors(table: &Table, pairs: &[(String, i64)]) -> DFResult<DVWriteResult> {
    let metadata = table.metadata();
    let schema = metadata.current_schema();
    let path_to_partition = live_data_file_partitions(table).await?;

    // Resolve every PartitionKey BEFORE opening the Puffin file, so an unresolvable one cannot
    // leave a fully written, unreferenced Puffin on storage.
    let mut partition_key_by_path: HashMap<&str, PartitionKey> = HashMap::new();
    let mut resolved: Vec<(&str, i32, Struct, Option<i64>)> = Vec::new();
    for (path, _) in pairs {
        if partition_key_by_path.contains_key(path.as_str()) {
            continue;
        }
        let (spec_id, partition, data_seq) = path_to_partition.get(path).cloned().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "deletion-vector: data file `{path}` is not a live file of the current snapshot, so its partition cannot be resolved"
            ))
        })?;
        let spec = metadata
            .partition_spec_by_id(spec_id)
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "deletion-vector: data file references unknown partition spec {spec_id}"
                ))
            })?
            .as_ref()
            .clone();
        let partition_key = PartitionKey::new(spec, schema.clone(), partition.clone())
            .map_err(to_datafusion_error)?;
        partition_key_by_path.insert(path.as_str(), partition_key);
        resolved.push((path.as_str(), spec_id, partition, data_seq));
    }

    // Load the DV each touched data file already has, if any. Java's `loadPreviousDeletes` is
    // called per touched path, so a data file with only previous deletes and no new position is
    // never visited and keeps its DV.
    let live = live_delete_vectors_by_data_file(table).await?;

    // Refuse a data file still covered by a Parquet position delete, BEFORE the Puffin is opened.
    // Java's `loadPreviousDeletes` unions those positions into the new DV and rewrites the
    // file-scoped sources; this port reads DVs only, so merging them is not yet possible. The
    // commit door rejects it too, but only after a fully written, unreferenced Puffin reached
    // storage. Reachable on a V2 table with position deletes upgraded to V3. GAP_MATRIX row R114
    // carries the residue.
    for (path, spec_id, partition, data_seq) in &resolved {
        let covered = live.legacy_position_deletes.iter().any(|delete| {
            legacy_position_delete_applies(delete, path, *spec_id, partition, *data_seq)
        });
        if covered {
            return Err(DataFusionError::NotImplemented(format!(
                "deletion-vector: data file `{path}` is still covered by a Parquet position-delete \
                 file. Merging those positions into a new DV is not yet ported (Java \
                 BaseDVFileWriter.loadPreviousDeletes does it), and V3 forbids writing another \
                 position-delete file — rewrite the table's position deletes as DVs first"
            )));
        }
    }

    let mut previous_deletes_by_path: HashMap<String, PreviousDeletes> = HashMap::new();
    for path in partition_key_by_path.keys() {
        let Some(existing) = live.dv_by_data_file.get(*path) else {
            continue;
        };
        let positions = load_delete_vector(table.file_io(), existing)
            .await
            .map_err(to_datafusion_error)?;
        previous_deletes_by_path.insert(
            (*path).to_string(),
            PreviousDeletes::new(positions, vec![existing.clone()]),
        );
    }

    let location_gen =
        DefaultLocationGenerator::new(metadata.clone()).map_err(to_datafusion_error)?;
    let file_name_gen = DefaultFileNameGenerator::new(
        "dv".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Puffin,
    );
    // The Puffin file spans every partition this delete touches, so it is not partition-scoped.
    let location = location_gen.generate_location(None, &file_name_gen.generate_file_name());
    let output_file = table
        .file_io()
        .new_output(&location)
        .map_err(to_datafusion_error)?;

    let mut writer = DVFileWriter::new(output_file).with_previous_deletes(previous_deletes_by_path);
    for (path, position) in pairs {
        let position = u64::try_from(*position).map_err(|_| {
            DataFusionError::Internal(format!(
                "deletion-vector: negative row position {position} for data file `{path}`"
            ))
        })?;
        writer
            .delete(path, position, partition_key_by_path.get(path.as_str()))
            .map_err(to_datafusion_error)?;
    }
    writer
        .close_with_result()
        .await
        .map_err(to_datafusion_error)
}

/// The `(path, pos)` pairs of one position-delete output file, keyed by the `(spec_id, partition)`
/// of the data files they delete from.
type PositionDeleteGroups = HashMap<(i32, Struct), Vec<(String, i64)>>;

/// Group `(path, pos)` pairs by the `(spec_id, partition)` of the data file each one deletes from,
/// so every position-delete file can be stamped with the SAME spec and partition as its target
/// (Java `PositionDeleteWriter` always carries a per-data-file `PartitionKey`).
///
/// A pair whose data file is absent from `path_to_partition` is a hard error. The map is built from
/// the current snapshot's DATA manifests, so a miss means the pair references a file that is not
/// live — the pairs come from a scan of that same snapshot, so it cannot happen without a bug.
/// Fabricating `(default_spec, Struct::empty())` for it (the previous fallback) pairs a PARTITIONED
/// spec with an EMPTY tuple: that used to ABORT in `PartitionKey::to_path` before any validation
/// could see it, and with the path walk totalised it would instead write a delete file under a
/// `field=null` path carrying a tuple no reader can match.
///
/// Only reached on the PARTITIONED path — the unpartitioned table returns before this.
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

/// Write one position-delete file for a SINGLE `(spec_id, partition)` group.
///
/// Prefer `Some(partition_key)` carrying the data file's own spec (always, on the multi-path).
/// When `partition_key` is `None`, `configured_spec` MUST be `Some` so the writer stamps that
/// spec's id via `with_partition_spec` instead of fabricating `DEFAULT_PARTITION_SPEC_ID` (0).
/// The caller must have pre-sorted `pairs` by `(path, pos)`.
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
    // Keep the position-delete `file_path` / `pos` bounds FULL and EXACT:
    // - MetricsConfig::for_position_delete → Full mode (Java MetricsConfig.forPositionDelete)
    // - position_delete_writer_properties → no 64-byte parquet stats truncate (so min_is_exact /
    //   max_is_exact stay true and equal-bounds path routing works for long S3 URIs)
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
///
/// A NULL slot is a hard error rather than a decoded value: arrow's `value()` returns `""` for a
/// null string, and an empty path silently becomes a position delete against a file that does not
/// exist. `_file` is a reserved metadata column the scan always materializes, so a NULL means the
/// batch did not come from where this code believes it did.
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

/// Decode the reserved `_pos` column at `row`.
///
/// A NULL slot is a hard error for the same reason as [`decode_file_path`]: arrow's `value()`
/// returns `0` for a null `i64`, which would silently position-delete row 0 of a real data file.
fn decode_position(col: &Int64Array, row: usize) -> DFResult<i64> {
    if col.is_null(row) {
        return Err(DataFusionError::Internal(format!(
            "reserved _pos column is NULL at row {row}; a position delete cannot be keyed by an \
             unknown row position"
        )));
    }
    Ok(col.value(row))
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
            // Sliced REE array: the logical→physical mapping is offset-relative, so defer to
            // `get_physical_index` per row (still allocation-free). Behaviorally identical to the
            // fast path; kept separate because a sliced run-ends walk is easy to get subtly wrong.
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
    plan_properties: Arc<PlanProperties>,
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
        //
        // `logical_null_count`, not `null_count`: the latter is the PHYSICAL count, which is 0 for a
        // dictionary- or run-end-encoded array whose *values* carry the NULL. `RecordBatch::try_new`'s
        // own nullability check is physical too, so such a NULL would clear both gates and be written.
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

/// Merge-on-read UPDATE: position-delete the OLD matching rows and insert NEW rows carrying the updated
/// values, in one `RowDelta`. Returns the number of rows updated. Works for both partitioned and
/// unpartitioned tables: the NEW rows are routed through the partition-aware
/// [`StreamingDataFileWriter`], which computes partition values from the POST-assignment column values.
/// Position deletes are keyed by (data-file path, position) and are partition-agnostic, so the delete
/// side is unchanged.
async fn merge_on_read_update(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
) -> DFResult<u64> {
    let delete_kind = merge_on_read_delete_kind(table)?;

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
                pairs.push((
                    decode_file_path(file_col, row)?,
                    decode_position(pos_col, row)?,
                ));
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
    let (delete_files, superseded_delete_files) =
        write_merge_on_read_deletes(table, delete_kind, &pairs).await?;
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
    // See the MoR DELETE path: a merged DV supersedes the one it absorbed.
    if !superseded_delete_files.is_empty() {
        action = action.remove_deletes_many(superseded_delete_files);
    }
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
/// [`StreamingDataFileWriter`] computes partition values from the post-assignment column values.
///
/// **Streaming (H7-S2).** Same two-pass streaming shape as [`copy_on_write_delete`]: pass 1 retains
/// only the affected-file set and the counter, pass 2 re-scans and streams each rewritten batch into
/// the writer. The cost is a second full read — see the module-level *Memory* note.
async fn copy_on_write_update(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    table_schema: &SchemaRef,
    isolation: IsolationLevel,
) -> DFResult<u64> {
    // The snapshot this DML's scan reads — the §5 `validate_from_snapshot` anchor (`SparkWrite.java`
    // L470-472 / L493-495). Both passes below pin this same snapshot.
    let scan_snapshot_id = table.metadata().current_snapshot_id();

    // 1. Pass 1 — affected-file detection, STREAMED. A source file is AFFECTED iff at least one of its
    //    rows matches the predicate (or the predicate is None → all rows match → all files affected).
    //    Also counts total updated rows for the return value. Only `affected` (one path per affected
    //    FILE) and the counter survive the pass — no rows and no per-batch masks are retained.
    let mut stream = cow_scan_stream(table, table_schema, scan_snapshot_id).await?;
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

    // Release pass 1's exhausted scan explicitly — a shadowed binding would otherwise keep its state
    // alive for all of pass 2 (see the same note in `copy_on_write_delete`).
    drop(stream);

    // 2. No updated rows → no-op (avoid a pointless rewrite of unchanged data). Also skips the second
    //    scan entirely.
    if updated == 0 {
        return Ok(0);
    }

    // 3. Pass 2 — RE-SCAN the same snapshot and stream the rewrite content for affected files only.
    //    For each batch: filter down to rows whose source file is in the affected set, then apply
    //    the assignments with the per-row match mask so that:
    //      * matched rows (WHERE = TRUE) take the new SET values
    //      * other rows of the SAME affected file keep their original values (carried unchanged)
    //    Rows from unaffected files are NOT included — their source files are untouched. Each rewritten
    //    batch goes straight to the writer and is dropped; nothing accumulates.
    let mut stream = cow_scan_stream(table, table_schema, scan_snapshot_id).await?;
    // Eager construction is behaviour-preserving here (unlike the DELETE path): `updated > 0` means at
    // least one file is affected, and every row of an affected file is rewritten, so pass 2 always
    // feeds the writer at least one batch — the pre-H7-S2 form always constructed it too.
    let mut data_writer = StreamingDataFileWriter::try_new(table)?;

    while let Some(batch) = stream.try_next().await.map_err(to_datafusion_error)? {
        let num_rows = batch.num_rows();
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .ok_or_else(|| {
                DataFusionError::Internal("update scan missing _file column".to_string())
            })?;

        // Build table-column sub-batch (rows from the FULL batch including unaffected-file rows).
        let table_batch = table_column_batch(&batch, table_schema)?;

        // Keep-mask: only rows whose source file is in the affected set.
        let paths = decode_file_paths_batch(file_col)?;
        let keep_affected: BooleanArray = (0..num_rows)
            .map(|row| affected.contains(paths[row]))
            .collect();
        if keep_affected.true_count() == 0 {
            continue;
        }

        // Filter down to affected-file rows (table columns only, no _file).
        let affected_batch = filter_record_batch(&table_batch, &keep_affected)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

        // The per-row WHERE match mask WITHIN the affected rows, evaluated directly over
        // `affected_batch`. The previous form (M7) instead cached pass 1's per-batch mask and filtered
        // it by `keep_affected`; the two are equal because `match_mask` is a row-wise pure function and
        // arrow `filter` preserves row order. The cache is GONE because it was indexed by batch
        // POSITION, and pass 2 is now an independent scan whose batch boundaries and arrival order are
        // not guaranteed to match pass 1's — indexing across them would silently apply one batch's mask
        // to another batch's rows. The traded cost is one extra predicate evaluation per batch.
        let affected_match_mask = match_mask(&predicate, &affected_batch)?;

        // Apply assignments: matched rows take new values; non-matched rows keep old values.
        let rewritten = apply_assignments(
            &affected_batch,
            assignments,
            table_schema,
            Some(&affected_match_mask),
        )?;
        data_writer.write_batch(rewritten).await?;
    }

    // 4. Close the writer. Routes each row to its correct partition by the POST-assignment column
    //    values — a partition-key-changing UPDATE automatically moves the row to the new partition
    //    file. A writer never fed a batch produces no file.
    let new_files = data_writer.finish().await?;

    // 5. Commit: delete the affected source files, add the rewritten files. Full-metadata removals
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

    use datafusion::arrow::array::{
        ArrayRef, DictionaryArray, Int32Array, Int64Array, RecordBatch, RunArray, StringArray,
    };
    use datafusion::arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::Column;
    use iceberg::spec::Literal;

    use super::{
        IsolationLevel, apply_assignments, decode_file_path, decode_file_paths_batch,
        decode_position, group_pairs_by_partition, legacy_position_delete_applies,
        legacy_position_delete_entry, position_delete_unpartitioned_fast_path,
        sort_position_delete_pairs,
    };

    // =============================================================================================
    // WG5 (c) — an assignment must never smuggle a NULL into a REQUIRED column. `null_count()` is
    // the PHYSICAL count: a dictionary (or run-end) array whose *values* hold a NULL reports 0
    // while `logical_null_count()` reports the real answer, and `RecordBatch::try_new`'s own
    // nullability check is physical too — so the NULL passes both gates and is written.
    // =============================================================================================

    /// A single-column table schema for `d`, `nullable` as given, dictionary-encoded Utf8.
    fn dict_column_schema(nullable: bool) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            "d",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            nullable,
        )]))
    }

    /// Dictionary array with a NULL hiding in the VALUES: physically null-free keys, logically a
    /// NULL at row 1.
    fn dict_with_null_value() -> ArrayRef {
        let values = StringArray::from(vec![Some("x"), None]);
        let keys = Int32Array::from(vec![0, 1]);
        Arc::new(
            DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values))
                .expect("dictionary array"),
        )
    }

    #[test]
    fn test_dictionary_encoded_null_cannot_be_assigned_to_a_required_column() {
        let column = dict_with_null_value();
        // The premise of the whole test: physically clean, logically NULL.
        assert_eq!(column.null_count(), 0, "physical null count must be 0");
        assert_eq!(column.logical_null_count(), 1, "row 1 is logically NULL");

        let schema = dict_column_schema(false);
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&column)]).expect("batch");
        let assignment: Arc<dyn PhysicalExpr> = Arc::new(Column::new("d", 0));

        let err = apply_assignments(&batch, &[(0, assignment)], &schema, None)
            .expect_err("a dictionary-encoded NULL must not reach a required column");
        assert!(
            err.to_string()
                .contains("UPDATE cannot assign NULL to required column 'd'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_dictionary_encoded_null_is_fine_for_an_optional_column() {
        // The negative pin: the guard must reject only REQUIRED columns.
        let column = dict_with_null_value();
        let schema = dict_column_schema(true);
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&column)]).expect("batch");
        let assignment: Arc<dyn PhysicalExpr> = Arc::new(Column::new("d", 0));

        let out = apply_assignments(&batch, &[(0, assignment)], &schema, None)
            .expect("an optional column may take a NULL");
        assert_eq!(out.column(0).logical_null_count(), 1);
    }

    #[test]
    fn test_null_free_assignment_to_a_required_column_still_succeeds() {
        // The other negative pin: `logical_null_count` must not reject clean data.
        let values = StringArray::from(vec![Some("x"), Some("y")]);
        let keys = Int32Array::from(vec![0, 1]);
        let column: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values))
                .expect("dictionary array"),
        );
        let schema = dict_column_schema(false);
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&column)]).expect("batch");
        let assignment: Arc<dyn PhysicalExpr> = Arc::new(Column::new("d", 0));

        let out = apply_assignments(&batch, &[(0, assignment)], &schema, None)
            .expect("a NULL-free assignment to a required column must succeed");
        assert_eq!(out.column(0).logical_null_count(), 0);
    }

    // =============================================================================================
    // WG5 (d) — the reserved `_file` / `_pos` decode read `.value(i)` with no validity check.
    // Arrow's `value()` on a NULL slot returns a well-formed lie: `""` for a string, `0` for an
    // i64. Both feed straight into a position-delete tuple, so a NULL `_file` deletes against an
    // empty path and a NULL `_pos` deletes ROW 0 of a real data file.
    // =============================================================================================

    #[test]
    fn test_decode_file_path_rejects_a_null_path() {
        let col: ArrayRef = Arc::new(StringArray::from(vec![Some("s3://b/a.parquet"), None]));
        assert!(
            decode_file_path(&col, 0).is_ok(),
            "the live row must still decode"
        );
        let err = decode_file_path(&col, 1).expect_err("a NULL _file must not decode to \"\"");
        assert!(err.to_string().contains("_file"), "unexpected error: {err}");
    }

    #[test]
    fn test_decode_file_paths_batch_rejects_a_null_path() {
        let col: ArrayRef = Arc::new(StringArray::from(vec![Some("s3://b/a.parquet"), None]));
        let err = decode_file_paths_batch(&col).expect_err("a NULL _file must not decode to \"\"");
        assert!(err.to_string().contains("_file"), "unexpected error: {err}");
    }

    #[test]
    fn test_decode_file_path_rejects_a_null_ree_value() {
        // The REE shape the COW scan actually produces, with a NULL in the run VALUES.
        let run_ends = Int32Array::from(vec![2, 4]);
        let values = StringArray::from(vec![Some("f/a.parquet"), None]);
        let ree = RunArray::<Int32Type>::try_new(&run_ends, &values).expect("build REE");
        let col: ArrayRef = Arc::new(ree);
        assert!(decode_file_path(&col, 0).is_ok(), "run 0 is live");
        let err = decode_file_path(&col, 3).expect_err("a NULL REE _file value must not decode");
        assert!(err.to_string().contains("_file"), "unexpected error: {err}");
        let err = decode_file_paths_batch(&col).expect_err("batch decode must reject it too");
        assert!(err.to_string().contains("_file"), "unexpected error: {err}");
    }

    #[test]
    fn test_decode_position_rejects_a_null_position() {
        let col = Int64Array::from(vec![Some(7), None]);
        assert_eq!(
            decode_position(&col, 0).expect("the live row must decode"),
            7
        );
        let err = decode_position(&col, 1).expect_err("a NULL _pos must not decode to 0");
        assert!(err.to_string().contains("_pos"), "unexpected error: {err}");
    }

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

    /// The applicability domain of [`legacy_position_delete_applies`], one test per cell. Java's
    /// own writer never sets `referenced_data_file`, so the BOUNDS leg and the PARTITION leg are
    /// the two that fire in practice, and they disagree: a named delete ignores the partition.
    /// Risk pinned: reading `referenced_data_file` instead of the shared derivation. Java's writer
    /// leaves the field unset and only equal `file_path` bounds, so the field-only read would treat
    /// this delete as partition-scoped and miss the file it actually names.
    #[test]
    fn test_legacy_delete_entry_derives_the_name_from_equal_file_path_bounds() {
        use iceberg::spec::{
            DataContentType, DataFileBuilder, DataFileFormat, Datum, Struct as IcebergStruct,
        };

        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("s3://b/pos-del.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(7)
            .partition(IcebergStruct::from_iter([Some(Literal::long(999))]))
            .lower_bounds(std::collections::HashMap::from([(
                iceberg::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("s3://b/a.parquet"),
            )]))
            .upper_bounds(std::collections::HashMap::from([(
                iceberg::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("s3://b/a.parquet"),
            )]))
            .build()
            .expect("build a bounds-scoped position delete");

        let entry = legacy_position_delete_entry(&delete_file, Some(3));
        assert_eq!(
            entry.0.as_deref(),
            Some("s3://b/a.parquet"),
            "equal file_path bounds name the data file even with referenced_data_file unset"
        );
        assert_eq!(
            entry.1, 7,
            "the stamp is carried for the partition-scoped leg"
        );
        assert_eq!(
            entry.2,
            IcebergStruct::from_iter([Some(Literal::long(999))]),
            "the partition TUPLE is carried too — an empty one matches nothing on a partitioned \
             table, so the refusal would silently stop firing"
        );
        assert_eq!(entry.3, Some(3), "the sequence number is carried");
    }

    #[test]
    fn test_legacy_delete_named_by_path_applies_across_partitions() {
        let delete = (
            Some("s3://b/a.parquet".to_string()),
            7,
            iceberg::spec::Struct::from_iter([Some(Literal::long(999))]),
            Some(1),
        );
        let data_partition = iceberg::spec::Struct::from_iter([Some(Literal::long(0))]);
        assert!(
            legacy_position_delete_applies(
                &delete,
                "s3://b/a.parquet",
                0,
                &data_partition,
                Some(1)
            ),
            "a delete that NAMES the file applies whatever partition it is stamped with — Spark's default write granularity is FILE, so a mismatched stamp is routine"
        );
        assert!(
            !legacy_position_delete_applies(&delete, "s3://b/other.parquet", 7, &delete.2, Some(1)),
            "and it applies to no other file, even one sharing its stamp"
        );
    }

    #[test]
    fn test_legacy_delete_without_a_name_applies_by_partition() {
        let partition = iceberg::spec::Struct::from_iter([Some(Literal::long(0))]);
        let delete = (None, 0, partition.clone(), Some(1));
        assert!(
            legacy_position_delete_applies(&delete, "s3://b/a.parquet", 0, &partition, Some(1)),
            "a partition-scoped delete covers every data file in its partition"
        );
        let other = iceberg::spec::Struct::from_iter([Some(Literal::long(1))]);
        assert!(
            !legacy_position_delete_applies(&delete, "s3://b/a.parquet", 0, &other, Some(1)),
            "but not one in another partition"
        );
        assert!(
            !legacy_position_delete_applies(&delete, "s3://b/a.parquet", 1, &partition, Some(1)),
            "nor one under another spec"
        );
    }

    /// Risk pinned: refusing a V3 delete because of a position delete that CANNOT apply. A data
    /// file written after the delete is not covered by it, and Java writes that DV happily.
    #[test]
    fn test_legacy_delete_older_than_the_data_file_does_not_apply() {
        let partition = iceberg::spec::Struct::from_iter([Some(Literal::long(0))]);
        let delete = (None, 0, partition.clone(), Some(1));
        assert!(
            !legacy_position_delete_applies(&delete, "s3://b/new.parquet", 0, &partition, Some(2)),
            "delete_seq 1 < data_seq 2 — the delete predates the file and cannot cover it"
        );
        assert!(
            legacy_position_delete_applies(&delete, "s3://b/old.parquet", 0, &partition, Some(1)),
            "an equal sequence number DOES apply (delete_seq >= data_seq)"
        );
        assert!(
            legacy_position_delete_applies(&delete, "s3://b/x.parquet", 0, &partition, None),
            "an unknown sequence errs toward applying, so the caller refuses rather than corrupts"
        );

        // The sequence rule is the same `>=` on BOTH legs — only the key changes. Asserting it on
        // the partition leg alone leaves the named leg free to skip it.
        let named = (
            Some("s3://b/new.parquet".to_string()),
            0,
            partition.clone(),
            Some(1),
        );
        assert!(
            !legacy_position_delete_applies(&named, "s3://b/new.parquet", 0, &partition, Some(2)),
            "a NAMED delete that predates its data file cannot cover it either"
        );
        assert!(
            legacy_position_delete_applies(&named, "s3://b/new.parquet", 0, &partition, Some(1)),
            "and it does apply at an equal sequence number"
        );
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

    // ============================================================================================
    // BUG-001 Option A — unpartitioned fast-path predicate (mutation-proven).
    // ============================================================================================

    #[test]
    fn test_pos_delete_fast_path_only_for_single_empty_spec() {
        // Never-evolved empty partition type (field_count == 0).
        assert!(position_delete_unpartitioned_fast_path(1, 0));
        // Partitioned or all-Void (non-zero fields): always walk manifests.
        assert!(!position_delete_unpartitioned_fast_path(1, 1));
        // Evolved: multi-spec + empty default (DROP PARTITION FIELD) — MUST NOT fast-path.
        assert!(
            !position_delete_unpartitioned_fast_path(2, 0),
            "BUG-001: multi-spec with empty default must take the manifest walk"
        );
        assert!(!position_delete_unpartitioned_fast_path(2, 1));
        // Zero specs is not a real table shape; refuse the fast path.
        assert!(!position_delete_unpartitioned_fast_path(0, 0));
    }

    /// Mutation twin: weakening to "default is empty" alone (forgetting multi-spec) would make
    /// this assert fail — keeps Option A load-bearing.
    #[test]
    fn test_pos_delete_fast_path_mutation_field_count_only_is_wrong() {
        let evolved_empty_default = position_delete_unpartitioned_fast_path(2, 0);
        assert!(
            !evolved_empty_default,
            "mutation RED: field_count-only condition would take the fast path here"
        );
    }

    /// C1-L-002: all-Void is_unpartitioned but has fields — must NOT fast-path.
    #[test]
    fn test_pos_delete_fast_path_rejects_all_void_single_spec() {
        // One void field ⇒ field_count 1.
        assert!(
            !position_delete_unpartitioned_fast_path(1, 1),
            "all-Void needs a null-tuple PartitionKey, not the empty fast path"
        );
    }

    // ============================================================================================
    // WG3-L3: the position-delete grouping resolves every pair's real partition instead of
    // fabricating `(default_spec, Struct::empty())` for an unmatched data file.
    // ============================================================================================

    /// `path → (spec_id, partition)` for two files of a one-field partitioned spec.
    fn partition_map() -> std::collections::HashMap<String, (i32, iceberg::spec::Struct)> {
        use iceberg::spec::{Literal, Struct};

        let mut map = std::collections::HashMap::new();
        map.insert(
            "s3://b/x0.parquet".to_string(),
            (1, Struct::from_iter([Some(Literal::long(0))])),
        );
        map.insert(
            "s3://b/x1.parquet".to_string(),
            (1, Struct::from_iter([Some(Literal::long(1))])),
        );
        map
    }

    /// The normal path: pairs are grouped by their data file's own `(spec_id, partition)`, so each
    /// delete file is stamped with the spec + partition of the file it deletes from.
    #[test]
    fn test_group_pairs_by_partition_groups_by_the_target_files_partition() {
        let map = partition_map();
        let pairs = vec![
            ("s3://b/x0.parquet".to_string(), 3),
            ("s3://b/x1.parquet".to_string(), 7),
            ("s3://b/x0.parquet".to_string(), 1),
        ];

        let groups = group_pairs_by_partition(&pairs, &map).expect("every pair resolves");
        assert_eq!(
            groups.len(),
            2,
            "one group per distinct partition: {groups:?}"
        );
        let x0 = groups
            .get(&map["s3://b/x0.parquet"])
            .expect("the x=0 group must exist");
        assert_eq!(x0.len(), 2, "both x=0 pairs land in the same group");
        assert_eq!(
            groups
                .get(&map["s3://b/x1.parquet"])
                .expect("the x=1 group must exist")
                .len(),
            1
        );
    }

    /// A pair whose data file is not live in the snapshot the map was built from must FAIL — the
    /// previous fallback fabricated `(default_spec, Struct::empty())`, pairing a partitioned spec
    /// with an empty tuple. That aborted in `PartitionKey::to_path` before any validation ran, and
    /// with the path walk totalised it would write a delete file under a `field=null` path that no
    /// reader can ever match (a silent under-delete: the rows come back).
    ///
    /// MUTATION (restore the `unwrap_or_else(|| (default_spec.spec_id(), Struct::empty()))`
    /// fallback): this test goes RED.
    #[test]
    fn test_group_pairs_by_partition_rejects_an_unmatched_data_file() {
        let map = partition_map();
        let pairs = vec![
            ("s3://b/x0.parquet".to_string(), 3),
            ("s3://b/ghost.parquet".to_string(), 0),
        ];

        let err = group_pairs_by_partition(&pairs, &map)
            .expect_err("an unresolvable data file must fail loudly");
        assert!(
            err.to_string().contains("s3://b/ghost.parquet")
                && err.to_string().contains("is not a live file"),
            "the error must name the offending file: {err}"
        );
    }
}
