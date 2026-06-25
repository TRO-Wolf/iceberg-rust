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
//! **Memory.** Every path buffers the full live row set (the table scan is collected before any write
//! begins) — intended for tables that fit in executor memory; streaming + partition-aware rewrites are a
//! follow-up.
//!
//! The plan emits a single `UInt64` `count` row (rows affected), per DataFusion's DML contract.

use std::any::Any;
use std::collections::HashSet;
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
use iceberg::spec::{DataFile, DataFileFormat, FormatVersion, MetricsConfig};
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
) -> DFResult<u64> {
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

        match &predicate {
            None => {
                // DELETE FROM t — every row is deleted; every source file is affected.
                deleted += num_rows as u64;
                for row in 0..num_rows {
                    affected.insert(decode_file_path(file_col, row)?);
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

                for row in 0..num_rows {
                    // A row is deleted iff the predicate is TRUE (NULL → SQL three-valued logic → NOT deleted).
                    let is_deleted = mask.is_valid(row) && mask.value(row);
                    if is_deleted {
                        deleted += 1;
                        affected.insert(decode_file_path(file_col, row)?);
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
        let keep: BooleanArray = (0..num_rows)
            .map(|row| -> DFResult<bool> {
                let file_path = decode_file_path(file_col, row)?;
                Ok(!delete_mask[row] && affected.contains(&file_path))
            })
            .collect::<DFResult<Vec<bool>>>()?
            .into_iter()
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

    // 7. Commit: delete the affected source files by path, add the rewritten files.
    let tx = Transaction::new(table);
    tx.overwrite_files()
        .delete_files(affected.iter().cloned())
        .add_files(new_files)
        .apply(tx)
        .map_err(to_datafusion_error)?
        .commit(catalog)
        .await
        .map_err(to_datafusion_error)?;

    Ok(deleted)
}

/// Write data file(s) partition-correctly through the production `TaskWriter`. Works for BOTH
/// partitioned and unpartitioned tables.
///
/// For partitioned tables, each batch must contain only table-schema columns (no `_file` or other
/// reserved columns). The `PartitionValueCalculator` is used internally to compute and inject the
/// `_partition` struct column that `TaskWriter`'s splitter reads. `fanout_enabled = true` because the
/// `batches` vector may contain rows from multiple affected files belonging to different partitions
/// (and a single scan batch may interleave them); the `FanoutWriter` routes each row to its correct
/// partition writer without requiring pre-sorting.
///
/// Returns every `DataFile` the (possibly rolling) writer produced — correctly partitioned.
async fn write_partitioned_data_files(
    table: &Table,
    batches: &[RecordBatch],
) -> DFResult<Vec<DataFile>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

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

    // fanout_enabled = true: survivors may be unsorted across partitions.
    let mut writer = TaskWriter::try_new(builder, true, schema.clone(), partition_spec.clone())
        .map_err(to_datafusion_error)?;

    if partition_spec.is_unpartitioned() {
        // Unpartitioned: TaskWriter writes directly; no partition column needed.
        for batch in batches {
            writer
                .write(batch.clone())
                .await
                .map_err(to_datafusion_error)?;
        }
    } else {
        // Partitioned: compute the `_partition` struct column for each batch and append it so
        // the TaskWriter's partition splitter can route rows to the correct partition writer.
        let calculator = PartitionValueCalculator::try_new(&partition_spec, &schema)
            .map_err(to_datafusion_error)?;

        for batch in batches {
            let partition_array = calculator.calculate(batch).map_err(to_datafusion_error)?;

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

            writer
                .write(extended_batch)
                .await
                .map_err(to_datafusion_error)?;
        }
    }

    writer.close().await.map_err(to_datafusion_error)
}

/// Write surviving rows to new parquet data file(s) via the production `DataFileWriter` (unpartitioned),
/// matching columns to the table schema BY NAME. Returns every file produced (the writer may roll).
///
/// Used by the UPDATE paths (merge-on-read and copy-on-write), which still operate on unpartitioned
/// tables only. For DELETE, the partition-aware [`write_partitioned_data_files`] is used instead.
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
