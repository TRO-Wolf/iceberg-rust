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
//!   * **`copy-on-write`** (the Iceberg default when unset) — rewrite the surviving (non-matching) rows
//!     into new data files and replace ALL data via `OverwriteFiles`. Unpartitioned tables only for now;
//!     a partitioned table errors (partition-aware rewrite, routing survivors to their partitions, is a
//!     follow-up).
//!
//! **Correctness — why we evaluate the filter ourselves.** The matching rows are identified by
//! evaluating the *original* DataFusion `WHERE` filters (as a [`PhysicalExpr`]) against the scanned
//! rows. We deliberately do **not** delete by Iceberg predicate pushdown: `convert_filters_to_predicate`
//! is *inexact* — it loosens an `AND` whose branch it cannot convert (returning the convertible side
//! alone) — which is harmless for a SELECT (DataFusion re-filters) but would **over-delete** here. The
//! exact filter is the contract; pushdown is only ever a (future) pruning optimization layered under it.
//!
//! **Memory.** Every path buffers the full live row set (the table scan is collected before any write
//! begins) — intended for tables that fit in executor memory; streaming + partition-aware rewrites are a
//! follow-up.
//!
//! The plan emits a single `UInt64` `count` row (rows affected), per DataFusion's DML contract.

use std::any::Any;
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
use iceberg::arrow::FieldMatchMode;
use iceberg::expr::Predicate;
use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};
use iceberg::spec::{DataFile, DataFileFormat, FormatVersion};
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

/// `DELETE FROM` execution plan. Finds the matching rows, writes the delete artifacts, commits, and
/// emits the deleted-row count.
pub(crate) struct IcebergDeleteExec {
    table: Table,
    catalog: Arc<dyn Catalog>,
    /// The EXACT row filter (the `WHERE` clause as a `PhysicalExpr` over the table schema), or `None`
    /// to delete every row (`DELETE FROM t`).
    predicate: Option<Arc<dyn PhysicalExpr>>,
    mode: WriteMode,
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
        table_schema: SchemaRef,
    ) -> Self {
        let count_schema = Self::make_count_schema();
        let plan_properties = Self::compute_properties(Arc::clone(&count_schema));
        Self {
            table,
            catalog,
            predicate,
            mode,
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
        let table_schema = Arc::clone(&self.table_schema);
        let count_schema = Arc::clone(&self.count_schema);

        let stream = futures::stream::once(async move {
            let deleted = match mode {
                WriteMode::MergeOnRead => {
                    merge_on_read_delete(&table, catalog.as_ref(), predicate, &table_schema).await?
                }
                WriteMode::CopyOnWrite => {
                    copy_on_write_delete(&table, catalog.as_ref(), predicate, &table_schema).await?
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

/// Merge-on-read DELETE: identify the matching rows' `_file`/`_pos`, write a position-delete file, and
/// commit a `RowDelta`. Returns the number of rows deleted. **Buffers the full live row set in memory**
/// (the scan is collected before writing) — intended for tables that fit in executor memory.
async fn merge_on_read_delete(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    table_schema: &SchemaRef,
) -> DFResult<u64> {
    require_v2_for_merge_on_read(table)?;
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

    let mut pairs: Vec<(String, i64)> = Vec::new();
    for batch in &batches {
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
    pairs.sort();
    let deleted = pairs.len() as u64;

    // Write ALL position-delete files the (rolling) writer produced and commit EVERY one of them.
    let delete_files = write_position_deletes(table, &pairs).await?;

    let tx = Transaction::new(table);
    tx.row_delta()
        .add_deletes(delete_files)
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(deleted)
}

/// Copy-on-write DELETE: rewrite the surviving (non-matching) rows into new data files and replace ALL
/// existing data in one atomic `OverwriteFiles` snapshot. Returns the number of rows deleted.
///
/// Partition-aware rewrite (routing survivors back to their partitions) is not yet supported, so a
/// partitioned table errors with guidance to use merge-on-read.
async fn copy_on_write_delete(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    table_schema: &SchemaRef,
) -> DFResult<u64> {
    if !table.metadata().default_partition_spec().is_unpartitioned() {
        return Err(DataFusionError::NotImplemented(
            "copy-on-write DELETE on a partitioned table is not yet supported; set the table property \
             'write.delete.mode' = 'merge-on-read' to enable DELETE on this table"
                .to_string(),
        ));
    }

    // Scan EVERY live row (the scan applies any existing merge-on-read deletes), projecting the table
    // columns. We rewrite the survivors; we do not push the filter into the scan (Iceberg pushdown is
    // inexact — see the module note).
    let projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
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

    let mut survivors: Vec<RecordBatch> = Vec::new();
    let mut deleted: u64 = 0;
    for batch in &batches {
        let num_rows = batch.num_rows();
        // Rebuild the table-column batch BY NAME — the schema the predicate is bound to and the writer
        // matches against (robust to the scan's output column ordering).
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

        match &predicate {
            // `DELETE FROM t` — every row deleted; no survivors.
            None => deleted += num_rows as u64,
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
                    })?;
                // Keep a row iff the WHERE predicate is NOT TRUE for it (a NULL result keeps the row,
                // per SQL three-valued logic).
                let keep: BooleanArray = (0..num_rows)
                    .map(|row| !(mask.is_valid(row) && mask.value(row)))
                    .collect();
                let surviving = filter_record_batch(&table_batch, &keep)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
                deleted += (num_rows - surviving.num_rows()) as u64;
                if surviving.num_rows() > 0 {
                    survivors.push(surviving);
                }
            }
        }
    }

    // No matching rows → no-op.
    if deleted == 0 {
        return Ok(0);
    }

    // Rewrite the survivors and replace ALL data (delete every existing data file + add the rewrites) in
    // one atomic OverwriteFiles snapshot. When every row was deleted, `survivors` is empty → no files are
    // added → the table is left empty.
    let new_files = write_survivor_data_files(table, &survivors).await?;
    let tx = Transaction::new(table);
    tx.overwrite_files()
        .overwrite_by_row_filter(Predicate::AlwaysTrue)
        .add_files(new_files)
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(deleted)
}

/// Write surviving rows to new parquet data file(s) via the production `DataFileWriter` (unpartitioned),
/// matching columns to the table schema BY NAME. Returns every file produced (the writer may roll).
async fn write_survivor_data_files(
    table: &Table,
    batches: &[RecordBatch],
) -> DFResult<Vec<DataFile>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }
    let schema = table.metadata().current_schema().clone();
    let parquet_builder = ParquetWriterBuilder::new_with_match_mode(
        parquet::file::properties::WriterProperties::default(),
        schema,
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
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .map_err(to_datafusion_error)?;
    for batch in batches {
        writer
            .write(batch.clone())
            .await
            .map_err(to_datafusion_error)?;
    }
    writer.close().await.map_err(to_datafusion_error)
}

/// Write REAL parquet position-delete file(s) from sorted `(data_file_path, position)` pairs via the
/// production `PositionDeleteFileWriter`. Returns EVERY file the (rolling) writer produced — a large
/// DELETE may roll into more than one file, and ALL of them must be committed or the deletes in the
/// dropped files would be silently lost (rows resurrected on the next scan).
async fn write_position_deletes(table: &Table, pairs: &[(String, i64)]) -> DFResult<Vec<DataFile>> {
    let config = PositionDeleteWriterConfig::new().map_err(to_datafusion_error)?;
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).map_err(to_datafusion_error)?;
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
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(None)
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
    let delete_files = writer.close().await.map_err(to_datafusion_error)?;
    if delete_files.is_empty() {
        return Err(DataFusionError::Internal(
            "position-delete writer produced no file".to_string(),
        ));
    }
    Ok(delete_files)
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
/// values, in one `RowDelta`. Returns the number of rows updated. Unpartitioned tables only for now.
async fn merge_on_read_update(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    table_schema: &SchemaRef,
) -> DFResult<u64> {
    if !table.metadata().default_partition_spec().is_unpartitioned() {
        return Err(DataFusionError::NotImplemented(
            "UPDATE on a partitioned table is not yet supported".to_string(),
        ));
    }
    require_v2_for_merge_on_read(table)?;

    let mut projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    projection.push(RESERVED_COL_NAME_FILE.to_string());
    projection.push(RESERVED_COL_NAME_POS.to_string());

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

    let mut pairs: Vec<(String, i64)> = Vec::new();
    let mut new_rows: Vec<RecordBatch> = Vec::new();
    for batch in &batches {
        let table_batch = table_column_batch(batch, table_schema)?;
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
        let matching = filter_record_batch(&table_batch, &mask)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        new_rows.push(apply_assignments(
            &matching,
            assignments,
            table_schema,
            None,
        )?);
    }

    let updated = pairs.len() as u64;
    if updated == 0 {
        return Ok(0);
    }

    pairs.sort();
    let delete_files = write_position_deletes(table, &pairs).await?;
    let data_files = write_survivor_data_files(table, &new_rows).await?;

    let tx = Transaction::new(table);
    tx.row_delta()
        .add_data_files(data_files)
        .add_deletes(delete_files)
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(updated)
}

/// Copy-on-write UPDATE: rewrite ALL rows (matching rows take the new values, the rest unchanged) into
/// new data files and replace all data via `OverwriteFiles`. Returns the number of rows updated.
/// Unpartitioned tables only for now.
async fn copy_on_write_update(
    table: &Table,
    catalog: &dyn Catalog,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
    table_schema: &SchemaRef,
) -> DFResult<u64> {
    if !table.metadata().default_partition_spec().is_unpartitioned() {
        return Err(DataFusionError::NotImplemented(
            "UPDATE on a partitioned table is not yet supported".to_string(),
        ));
    }

    let projection: Vec<String> = table_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
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

    let mut output_batches: Vec<RecordBatch> = Vec::new();
    let mut updated: u64 = 0;
    for batch in &batches {
        let table_batch = table_column_batch(batch, table_schema)?;
        let mask = match_mask(&predicate, &table_batch)?;
        updated += mask.true_count() as u64;
        // Per-row select: matching rows take the new value, the rest keep the old.
        output_batches.push(apply_assignments(
            &table_batch,
            assignments,
            table_schema,
            Some(&mask),
        )?);
    }

    // No matching rows → no-op (avoid a pointless rewrite of unchanged data).
    if updated == 0 {
        return Ok(0);
    }

    let new_files = write_survivor_data_files(table, &output_batches).await?;
    let tx = Transaction::new(table);
    tx.overwrite_files()
        .overwrite_by_row_filter(Predicate::AlwaysTrue)
        .add_files(new_files)
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(updated)
}
