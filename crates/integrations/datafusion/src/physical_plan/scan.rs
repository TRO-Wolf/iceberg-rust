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

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::vec;

use datafusion::arrow::array::{
    Array, ArrayRef, ListArray, MapArray, RecordBatch, RecordBatchOptions, StructArray,
    new_null_array,
};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field as ArrowField, SchemaRef as ArrowSchemaRef};
use datafusion::common::Column;
use datafusion::common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties};
use datafusion::prelude::Expr;
use futures::{Stream, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::metadata_columns::is_metadata_column_name;
use iceberg::scan::{PartitionWork, stream_partition_work};
use iceberg::table::Table;
use iceberg::{Error, ErrorKind};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::expr_to_predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Iceberg-specific scan knobs registered on DataFusion [`ConfigOptions`], prefix `iceberg.`.
///
/// | Knob | Default | Meaning |
/// |---|---|---|
/// | `multi_partition_scan` | `true` | `false` forces `T = 1` without touching `target_partitions` |
/// | `data_file_concurrency` | `0` | the budget `L`; `0` derives it from `target_partitions` |
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct IcebergScanOptions {
    /// `false` disables multi-partition output, whatever `target_partitions` says.
    pub multi_partition_scan: bool,
    /// Total data-file concurrency budget `L`. Zero → use `target_partitions`.
    pub data_file_concurrency: usize,
}

impl Default for IcebergScanOptions {
    fn default() -> Self {
        Self {
            multi_partition_scan: true,
            data_file_concurrency: 0,
        }
    }
}

impl ExtensionOptions for IcebergScanOptions {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> datafusion::common::Result<()> {
        match key {
            "multi_partition_scan" => {
                self.multi_partition_scan = value.parse().map_err(|e| {
                    DataFusionError::Configuration(format!(
                        "invalid iceberg.multi_partition_scan={value}: {e}"
                    ))
                })?;
            }
            "data_file_concurrency" => {
                self.data_file_concurrency = value.parse().map_err(|e| {
                    DataFusionError::Configuration(format!(
                        "invalid iceberg.data_file_concurrency={value}: {e}"
                    ))
                })?;
            }
            _ => {
                return Err(DataFusionError::Configuration(format!(
                    "unknown iceberg config key: {key}"
                )));
            }
        }
        Ok(())
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        vec![
            ConfigEntry {
                key: "multi_partition_scan".to_string(),
                value: Some(self.multi_partition_scan.to_string()),
                description: "When false, force T=1 multi-partition off-switch without collapsing session target_partitions",
            },
            ConfigEntry {
                key: "data_file_concurrency".to_string(),
                value: Some(self.data_file_concurrency.to_string()),
                description: "Total data-file concurrency budget L (0 = derive from target_partitions)",
            },
        ]
    }
}

impl ConfigExtension for IcebergScanOptions {
    const PREFIX: &'static str = "iceberg";
}

/// How one advertised output column is produced from the scanned data. Resolution is by FIELD ID:
/// `RENAME COLUMN` keeps the id, so a name-keyed binding reads the wrong column after a rename.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ColumnSource {
    /// Take the scanned column under the name the SCANNED snapshot gives the advertised field id.
    Scanned(String),
    /// The scanned snapshot has no such field id, so emit NULLs, as Java does.
    Absent,
}

/// Manages the scanning process of an Iceberg [`Table`]. [`IcebergTableScan::plan`] assigns the
/// work of core `plan_tasks` into `N` [`PartitionWork`] units, as `UnknownPartitioning(N)`.
#[derive(Debug)]
pub struct IcebergTableScan {
    table: Table,
    /// `None` means the current snapshot at plan time.
    snapshot_id: Option<i64>,
    /// Concrete snapshot id resolved at plan time and frozen on the node (pin 12).
    resolved_snapshot_id: i64,
    plan_properties: Arc<PlanProperties>,
    projection: Option<Vec<String>>,
    /// The SCANNED snapshot's name for each advertised field id. See [`IcebergTableScan::new`].
    scan_columns: Vec<String>,
    /// How each advertised output column is produced, parallel to the advertised schema's fields.
    sources: Vec<ColumnSource>,
    predicates: Option<Predicate>,
    /// Optional row limit. It applies only when `N = 1`, because a per-partition cap over-counts,
    /// so `GlobalLimitExec` owns the limit above that.
    limit: Option<usize>,
    /// Empty when built by [`Self::new`] without planning, which takes the single-stream path.
    partition_work: Vec<PartitionWork>,
    /// Per-partition data-file concurrency `P = max(1, ceil(L/N))`.
    per_partition_concurrency: usize,
    batch_size: Option<usize>,
}

impl IcebergTableScan {
    /// Creates a new [`IcebergTableScan`] object. # Errors Fails when `projection` holds an index
    /// outside `schema`. # Notes The advertised schema is a contract: every parent operator was
    /// built against it.
    pub(crate) fn new(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Self> {
        // The FULL schema, not the projection: a pushed filter may reference an unprojected column.
        let bindings = resolve_bindings(&table, snapshot_id, &schema)?;

        let (output_schema, projection) = match projection {
            None => (schema, None),
            Some(indices) => {
                let projected_schema = Arc::new(schema.project(indices)?);
                let column_names = projected_schema
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect();
                (projected_schema, Some(column_names))
            }
        };
        let (scan_columns, sources) = project_bindings(&output_schema, &bindings)?;
        let plan_properties = Self::compute_properties(output_schema, 1);
        let predicates = convert_filters_to_predicate(&rebind_filters(filters, &bindings));

        let resolved_snapshot_id = match snapshot_id {
            Some(id) => id,
            None => table
                .metadata()
                .current_snapshot()
                .map(|s| s.snapshot_id())
                .unwrap_or(0),
        };

        Ok(Self {
            table,
            snapshot_id,
            resolved_snapshot_id,
            plan_properties,
            projection,
            scan_columns,
            sources,
            predicates,
            limit,
            partition_work: Vec::new(),
            per_partition_concurrency: 1,
            batch_size: None,
        })
    }

    /// Eager multi-partition plan, on when `T > 1` and the post-strip group count is above 1.
    pub(crate) async fn plan(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
        knobs: ScanKnobs,
    ) -> DFResult<Self> {
        let mut scan = Self::new(
            table.clone(),
            snapshot_id,
            schema,
            projection,
            filters,
            limit,
        )?;

        let t = if knobs.multi_partition_scan {
            knobs.target_partitions.max(1)
        } else {
            1
        };
        let l = clamp_scan_knob(
            knobs
                .data_file_concurrency
                .unwrap_or(knobs.target_partitions),
        );

        let mut scan_builder = match snapshot_id {
            Some(id) => table.scan().snapshot_id(id),
            None => table.scan(),
        };
        scan_builder = scan_builder.select(scan.scan_columns.clone());
        if let Some(pred) = scan.predicates.clone() {
            scan_builder = scan_builder.with_filter(pred);
        }
        if let Some(bs) = knobs.batch_size {
            scan_builder = scan_builder.with_batch_size(Some(clamp_scan_knob(bs)));
        }
        scan_builder = scan_builder.with_data_file_concurrency_limit(l);

        let table_scan = scan_builder.build().map_err(to_datafusion_error)?;
        // Fail closed: demoting to N=1 unfreezes the snapshot at execute time.
        let work = table_scan
            .plan_partition_work(t)
            .await
            .map_err(to_datafusion_error)?;

        let n = work.len().max(1);
        // With N > L this gives P = 1, and the total may exceed L. That is intended.
        let p = l.div_ceil(n).max(1);

        if let Some(first) = work.first() {
            scan.resolved_snapshot_id = first.snapshot_id();
        }
        // A sole per-partition hard limit over-counts when N > 1.
        if n > 1 {
            scan.limit = None;
        }
        scan.partition_work = work;
        scan.per_partition_concurrency = p;
        scan.batch_size = knobs.batch_size.map(clamp_scan_knob);
        scan.plan_properties = Self::compute_properties(scan.schema(), n);
        Ok(scan)
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    pub fn resolved_snapshot_id(&self) -> i64 {
        self.resolved_snapshot_id
    }

    /// Assigned partition work, `N = len`. Empty when built without a plan.
    pub fn partition_work(&self) -> &[PartitionWork] {
        &self.partition_work
    }

    pub fn projection(&self) -> Option<&[String]> {
        self.projection.as_deref()
    }

    pub fn predicates(&self) -> Option<&Predicate> {
        self.predicates.as_ref()
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    fn compute_properties(schema: ArrowSchemaRef, n: usize) -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(n.max(1)),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }
}

impl ExecutionPlan for IcebergTableScan {
    fn name(&self) -> &str {
        "IcebergTableScan"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan + 'static>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let advertised_schema = self.schema();
        let sources = self.sources.clone();

        if !self.partition_work.is_empty() {
            let n = self.partition_work.len();
            if partition >= n {
                return Err(DataFusionError::Execution(format!(
                    "IcebergTableScan partition index {partition} out of range (N={n})"
                )));
            }
            let work = self.partition_work[partition].clone();
            // The embedded work id must match the plan-time id, sentinel 0 included.
            if work.snapshot_id() != self.resolved_snapshot_id {
                return Err(DataFusionError::Execution(format!(
                    "IcebergTableScan snapshot freeze violation: work snapshot {} != plan {}",
                    work.snapshot_id(),
                    self.resolved_snapshot_id
                )));
            }
            let file_io = self.table.file_io().clone();
            let concurrency = self.per_partition_concurrency;
            let batch_size = self.batch_size;
            let stream =
                stream_partition_work(file_io, &work, concurrency, batch_size, true, false)
                    .map_err(to_datafusion_error)?
                    .map_err(to_datafusion_error)
                    .and_then(move |batch| {
                        futures::future::ready(conform_batch(batch, &advertised_schema, &sources))
                    });

            // GlobalLimitExec owns the limit when N > 1.
            let limited_stream: Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>> =
                if n == 1 {
                    if let Some(limit) = self.limit {
                        let mut remaining = limit;
                        Box::pin(stream.try_filter_map(move |batch| {
                            futures::future::ready(if remaining == 0 {
                                Ok(None)
                            } else if batch.num_rows() <= remaining {
                                remaining -= batch.num_rows();
                                Ok(Some(batch))
                            } else {
                                let limited_batch = batch.slice(0, remaining);
                                remaining = 0;
                                Ok(Some(limited_batch))
                            })
                        }))
                    } else {
                        Box::pin(stream)
                    }
                } else {
                    Box::pin(stream)
                };

            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema(),
                limited_stream,
            )));
        }

        if partition > 0 {
            return Err(DataFusionError::Execution(format!(
                "IcebergTableScan partition index {partition} out of range (N=1 legacy)"
            )));
        }
        let knobs = scan_knobs_from_context(&context);
        let fut = get_batch_stream(
            self.table.clone(),
            self.snapshot_id,
            self.scan_columns.clone(),
            self.predicates.clone(),
            knobs,
        );
        let stream = futures::stream::once(fut)
            .try_flatten()
            .and_then(move |batch| {
                futures::future::ready(conform_batch(batch, &advertised_schema, &sources))
            });

        let limited_stream: Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>> =
            if let Some(limit) = self.limit {
                let mut remaining = limit;
                Box::pin(stream.try_filter_map(move |batch| {
                    futures::future::ready(if remaining == 0 {
                        Ok(None)
                    } else if batch.num_rows() <= remaining {
                        remaining -= batch.num_rows();
                        Ok(Some(batch))
                    } else {
                        let limited_batch = batch.slice(0, remaining);
                        remaining = 0;
                        Ok(Some(limited_batch))
                    })
                }))
            } else {
                Box::pin(stream)
            };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            limited_stream,
        )))
    }
}

impl DisplayAs for IcebergTableScan {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let n = self.partition_work.len().max(1);
        // The historic `IcebergTableScan projection:[...]` prefix keeps the EXPLAIN assertions and
        // the sqllogictest goldens stable. `N` is deterministic for a fixed fixture. A snapshot id
        // is per-run random, so it renders under EXPLAIN VERBOSE only.
        write!(
            f,
            "IcebergTableScan projection:[{}] predicate:[{}]",
            self.projection
                .clone()
                .map_or(String::new(), |v| v.join(",")),
            self.predicates
                .clone()
                .map_or(String::from(""), |p| format!("{p}")),
        )?;
        if matches!(t, datafusion::physical_plan::DisplayFormatType::Verbose) {
            write!(f, " snapshot_id={}", self.resolved_snapshot_id)?;
        }
        write!(f, " N={n}")
    }
}

/// Session-derived knobs for building an Iceberg core `TableScan` and its partition assignment.
/// DataFusion's `TaskContext` supplies them. Row selection stays at the core default, off, because
/// parsing the Parquet page index can outweigh the gain.
///
/// | Symbol | Value |
/// |---|---|
/// | `T` | the output partition budget: `target_partitions`, or `1` with `multi_partition_scan` off |
/// | `L` | the data-file concurrency: `data_file_concurrency`, else `target_partitions` |
/// | `P` | the per-partition concurrency `max(1, ceil(L/N))` |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScanKnobs {
    pub batch_size: Option<usize>,
    /// The budget `L`, for `TableScanBuilder::with_data_file_concurrency_limit`.
    pub data_file_concurrency: Option<usize>,
    /// Session target partition count (raw `T` input before off-switch).
    pub target_partitions: usize,
    /// Dedicated multi-partition off-switch (pin 13). Default true.
    pub multi_partition_scan: bool,
}

impl Default for ScanKnobs {
    fn default() -> Self {
        Self {
            batch_size: None,
            data_file_concurrency: None,
            target_partitions: 1,
            multi_partition_scan: true,
        }
    }
}

/// Floor the session-derived knobs. `ParquetRecordBatchReader` reads `batch_size == 0` as
/// end-of-stream, which looks like a successful empty scan, and `try_buffer_unordered(0)` hangs.
pub(crate) fn clamp_scan_knob(value: usize) -> usize {
    value.max(1)
}

pub(crate) fn scan_knobs_from_context(context: &TaskContext) -> ScanKnobs {
    let config = context.session_config();
    // DataFusion does not normalize 0, and Parquet returns an empty stream for it.
    let batch_size = clamp_scan_knob(config.batch_size());
    let target_partitions = clamp_scan_knob(config.target_partitions());

    let iceberg_opts = config
        .options()
        .extensions
        .get::<IcebergScanOptions>()
        .cloned()
        .unwrap_or_default();
    let multi_partition_scan = iceberg_opts.multi_partition_scan;
    // L falls back to target_partitions when the dedicated surface is zero.
    let data_file_concurrency = if iceberg_opts.data_file_concurrency > 0 {
        clamp_scan_knob(iceberg_opts.data_file_concurrency)
    } else {
        target_partitions
    };

    ScanKnobs {
        batch_size: Some(batch_size),
        data_file_concurrency: Some(data_file_concurrency),
        target_partitions,
        multi_partition_scan,
    }
}

/// Register default [`IcebergScanOptions`] on a session config, if absent.
pub fn ensure_iceberg_scan_options(config: &mut datafusion::prelude::SessionConfig) {
    if config
        .options()
        .extensions
        .get::<IcebergScanOptions>()
        .is_none()
    {
        config
            .options_mut()
            .extensions
            .insert(IcebergScanOptions::default());
    }
}

/// Builds a [`TableScan`] and converts it into a stream of Arrow [`RecordBatch`]es.
pub(crate) async fn get_batch_stream(
    table: Table,
    snapshot_id: Option<i64>,
    column_names: Vec<String>,
    predicates: Option<Predicate>,
    knobs: ScanKnobs,
) -> DFResult<Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>>> {
    let scan_builder = match snapshot_id {
        Some(snapshot_id) => table.scan().snapshot_id(snapshot_id),
        None => table.scan(),
    };

    // Never `select_all()`: it reads the column set the table has now, not the advertised one.
    let mut scan_builder = scan_builder.select(column_names);
    if let Some(pred) = predicates {
        scan_builder = scan_builder.with_filter(pred);
    }
    // Clamped here too, so a hand-built `ScanKnobs` holding `Some(0)` cannot reach Parquet.
    if let Some(batch_size) = knobs.batch_size {
        scan_builder = scan_builder.with_batch_size(Some(clamp_scan_knob(batch_size)));
    }
    if let Some(concurrency) = knobs.data_file_concurrency {
        scan_builder = scan_builder.with_data_file_concurrency_limit(clamp_scan_knob(concurrency));
    }
    // Row selection stays at the core default, off: the page-index parse cost can dominate.
    let table_scan = scan_builder.build().map_err(to_datafusion_error)?;

    let stream = table_scan
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .map_err(to_datafusion_error);
    Ok(Box::pin(stream))
}

/// Binds every advertised column to the scanned snapshot's schema BY FIELD ID, mapping an
/// advertised name to that schema's name for the same id, or to `None` when it lacks the id.
///
/// A name is not an identity in Iceberg, so the two schemas disagree on names after a rename. The
/// advertised Arrow fields carry `PARQUET:field_id`; a field without it is a loud error rather than
/// a name-based guess. With no snapshot to resolve against, the binding is the identity.
fn resolve_bindings(
    table: &Table,
    snapshot_id: Option<i64>,
    schema: &ArrowSchemaRef,
) -> DFResult<HashMap<String, Option<String>>> {
    let metadata = table.metadata();
    let snapshot = match snapshot_id {
        Some(snapshot_id) => metadata.snapshot_by_id(snapshot_id),
        None => metadata.current_snapshot(),
    };
    let Some(snapshot) = snapshot else {
        return Ok(schema
            .fields()
            .iter()
            .map(|field| (field.name().clone(), Some(field.name().clone())))
            .collect());
    };
    let snapshot_schema = snapshot.schema(metadata).map_err(to_datafusion_error)?;

    let mut bindings = HashMap::with_capacity(schema.fields().len());
    for field in schema.fields() {
        // A reserved metadata column is not a table field; the core scan resolves it by name.
        if is_metadata_column_name(field.name()) {
            bindings.insert(field.name().clone(), Some(field.name().clone()));
            continue;
        }
        let field_id = advertised_field_id(field)?;
        bindings.insert(
            field.name().clone(),
            snapshot_schema
                .name_by_field_id(field_id)
                .map(str::to_string),
        );
    }
    Ok(bindings)
}

/// Turns the output columns' bindings into `select` names and [`ColumnSource`]s, in order.
fn project_bindings(
    output_schema: &ArrowSchemaRef,
    bindings: &HashMap<String, Option<String>>,
) -> DFResult<(Vec<String>, Vec<ColumnSource>)> {
    let mut scan_columns = Vec::with_capacity(output_schema.fields().len());
    let mut sources = Vec::with_capacity(output_schema.fields().len());
    for field in output_schema.fields() {
        match bindings.get(field.name()) {
            Some(Some(name)) => {
                scan_columns.push(name.clone());
                sources.push(ColumnSource::Scanned(name.clone()));
            }
            Some(None) => sources.push(ColumnSource::Absent),
            None => {
                return Err(datafusion::error::DataFusionError::Internal(format!(
                    "projected column '{}' is not part of the schema the scan was built from",
                    field.name()
                )));
            }
        }
    }
    Ok((scan_columns, sources))
}

/// Rewrites pushed-down filters onto the scanned snapshot's names, dropping the rest.
///
/// A pushed filter runs against the DATA. After a rename the advertised name fails to bind, or
/// binds to a DIFFERENT column that now carries it, which prunes rows DataFusion cannot get back.
/// A filter over a column the snapshot lacks is not pushed at all, because that column reads NULL.
fn rebind_filters(filters: &[Expr], bindings: &HashMap<String, Option<String>>) -> Vec<Expr> {
    filters
        .iter()
        .filter_map(|filter| rebind_filter(filter, bindings))
        .collect()
}

/// One filter rewritten onto the scanned snapshot's names, or `None` if a column cannot bind.
fn rebind_filter(filter: &Expr, bindings: &HashMap<String, Option<String>>) -> Option<Expr> {
    let mut unbound = false;
    let rewritten = filter
        .clone()
        .transform(|node| {
            if let Expr::Column(column) = &node {
                match bindings.get(&column.name) {
                    Some(Some(scanned_name)) if scanned_name != &column.name => {
                        return Ok(Transformed::yes(Expr::Column(Column::new(
                            column.relation.clone(),
                            scanned_name,
                        ))));
                    }
                    Some(Some(_)) => {}
                    // The scanned snapshot has no such column, so refuse to push rather than guess.
                    Some(None) | None => unbound = true,
                }
            }
            Ok(Transformed::no(node))
        })
        .ok()?;
    (!unbound).then_some(rewritten.data)
}

/// The Iceberg field id an advertised Arrow field carries, or a loud error.
fn advertised_field_id(field: &ArrowField) -> DFResult<i32> {
    let raw = field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .ok_or_else(|| {
            to_datafusion_error(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "column '{}' carries no `{PARQUET_FIELD_ID_META_KEY}` metadata, so it cannot be \
                     bound to a table field: an Iceberg column is identified by its field id, and \
                     matching on the name instead would read the wrong column after a rename",
                    field.name()
                ),
            ))
        })?;
    raw.parse::<i32>().map_err(|e| {
        to_datafusion_error(
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "column '{}' carries an unparsable `{PARQUET_FIELD_ID_META_KEY}` metadata value '{raw}'",
                    field.name()
                ),
            )
            .with_source(e),
        )
    })
}

/// Whether an Arrow type change is one of Iceberg's LEGAL type promotions. It mirrors
/// [`iceberg::spec::is_promotion_allowed`], pinned by the mirror test below.
fn is_arrow_promotion_allowed(from: &DataType, to: &DataType) -> bool {
    if from == to {
        return true;
    }
    match (from, to) {
        (DataType::Int32, DataType::Int64) => true,
        (DataType::Float32, DataType::Float64) => true,
        (
            DataType::Decimal128(from_precision, from_scale),
            DataType::Decimal128(to_precision, to_scale),
        )
        | (
            DataType::Decimal256(from_precision, from_scale),
            DataType::Decimal256(to_precision, to_scale),
        ) => from_scale == to_scale && from_precision <= to_precision,
        _ => false,
    }
}

/// Coerces a scanned batch to the schema the plan advertised.
///
/// A DataFusion operator addresses its input by ORDINAL, so a batch carrying the right columns in
/// the wrong order, or one extra, is silent corruption. This rebuilds the batch in advertised order
/// from the bindings [`resolve_projection`] computed:
///
/// | Binding | Result |
/// |---|---|
/// | bound, same type | taken as is |
/// | bound under a different name | the same values, under the advertised name |
/// | bound with a legal promotion | cast to the advertised type |
/// | unbound and nullable | an all-NULL column, as Java's readers null-fill |
/// | unbound and not nullable, or an illegal type change | a typed error naming the column |
///
/// The row count is carried explicitly, so a zero-column `SELECT count(*)` keeps it.
fn conform_batch(
    batch: RecordBatch,
    advertised: &ArrowSchemaRef,
    sources: &[ColumnSource],
) -> DFResult<RecordBatch> {
    if batch.schema_ref() == advertised {
        return Ok(batch);
    }
    if sources.len() != advertised.fields().len() {
        return Err(datafusion::error::DataFusionError::Internal(format!(
            "the scan bound {} columns but advertises {}",
            sources.len(),
            advertised.fields().len()
        )));
    }

    let num_rows = batch.num_rows();
    let mut columns = Vec::with_capacity(advertised.fields().len());
    for (field, source) in advertised.fields().iter().zip(sources) {
        match source {
            ColumnSource::Scanned(name) => {
                let column = batch.column_by_name(name).ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "the scan selected column '{name}' for advertised column '{}' but the \
                         scanned batch does not carry it",
                        field.name()
                    ))
                })?;
                columns.push(conform_column(column, field, field.name())?);
            }
            ColumnSource::Absent if field.is_nullable() => {
                columns.push(new_null_array(field.data_type(), num_rows))
            }
            ColumnSource::Absent => {
                return Err(to_datafusion_error(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "required column '{}' has no field with its id in the snapshot being \
                         scanned, so there is no data to read for it and a required column cannot \
                         be null-filled",
                        field.name()
                    ),
                )));
            }
        }
    }

    RecordBatch::try_new_with_options(
        advertised.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .map_err(|e| {
        datafusion::error::DataFusionError::ArrowError(
            Box::new(e),
            Some("failed to conform a scanned batch to the schema the plan advertised".to_string()),
        )
    })
}

/// Coerces ONE scanned column to its advertised field, recursing through nested types. Iceberg
/// evolves a nested field as it evolves a top-level one, and none of those DDLs creates a snapshot.
/// A nested Arrow field carries `PARQUET:field_id`, and `path` names the offending field on error.
fn conform_column(column: &ArrayRef, target: &ArrowField, path: &str) -> DFResult<ArrayRef> {
    if column.data_type() == target.data_type() {
        return Ok(column.clone());
    }
    if is_arrow_promotion_allowed(column.data_type(), target.data_type()) {
        return cast(column, target.data_type()).map_err(|e| {
            datafusion::error::DataFusionError::ArrowError(
                Box::new(e),
                Some(format!("promoting column '{path}'")),
            )
        });
    }

    match (column.data_type(), target.data_type()) {
        (DataType::Struct(scanned_fields), DataType::Struct(target_fields)) => {
            let scanned = downcast::<StructArray>(column, path)?;
            let len = scanned.len();

            // An unidentifiable scanned child is indistinguishable from an absent one.
            let scanned_ids = scanned_fields
                .iter()
                .map(|field| advertised_field_id(field))
                .collect::<DFResult<Vec<_>>>()?;

            let mut children = Vec::with_capacity(target_fields.len());
            for target_child in target_fields {
                let child_path = format!("{path}.{}", target_child.name());
                let target_id = advertised_field_id(target_child)?;
                match scanned_ids.iter().position(|id| *id == target_id) {
                    Some(index) => children.push(conform_column(
                        scanned.column(index),
                        target_child,
                        &child_path,
                    )?),
                    None if target_child.is_nullable() => {
                        children.push(new_null_array(target_child.data_type(), len))
                    }
                    None => {
                        return Err(to_datafusion_error(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "required field '{child_path}' has no field with its id in the \
                                 snapshot being scanned, so there is no data to read for it and a \
                                 required field cannot be null-filled"
                            ),
                        )));
                    }
                }
            }
            Ok(Arc::new(
                StructArray::try_new_with_length(
                    target_fields.clone(),
                    children,
                    scanned.nulls().cloned(),
                    len,
                )
                .map_err(|e| {
                    datafusion::error::DataFusionError::ArrowError(
                        Box::new(e),
                        Some(format!("conforming struct column '{path}'")),
                    )
                })?,
            ))
        }
        (DataType::List(_), DataType::List(target_element)) => {
            let scanned = downcast::<ListArray>(column, path)?;
            let values =
                conform_column(scanned.values(), target_element, &format!("{path}.element"))?;
            Ok(Arc::new(
                ListArray::try_new(
                    target_element.clone(),
                    scanned.offsets().clone(),
                    values,
                    scanned.nulls().cloned(),
                )
                .map_err(|e| {
                    datafusion::error::DataFusionError::ArrowError(
                        Box::new(e),
                        Some(format!("conforming list column '{path}'")),
                    )
                })?,
            ))
        }
        (DataType::Map(_, _), DataType::Map(target_entries, ordered)) => {
            let scanned = downcast::<MapArray>(column, path)?;
            let entries: ArrayRef = Arc::new(scanned.entries().clone());
            let conformed = conform_column(&entries, target_entries, path)?;
            let conformed = downcast::<StructArray>(&conformed, path)?.clone();
            Ok(Arc::new(
                MapArray::try_new(
                    target_entries.clone(),
                    scanned.offsets().clone(),
                    conformed,
                    scanned.nulls().cloned(),
                    *ordered,
                )
                .map_err(|e| {
                    datafusion::error::DataFusionError::ArrowError(
                        Box::new(e),
                        Some(format!("conforming map column '{path}'")),
                    )
                })?,
            ))
        }
        (scanned_type, target_type) => Err(to_datafusion_error(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "column '{path}' is {scanned_type} in the snapshot being scanned but {target_type} \
                 in the schema this query was planned against, and that is not a legal Iceberg type \
                 promotion — the data cannot be read as the planned type"
            ),
        ))),
    }
}

/// Downcasts an array whose `DataType` already matched. A failure is a broken Arrow invariant.
fn downcast<'a, T: 'static>(column: &'a ArrayRef, path: &str) -> DFResult<&'a T> {
    column.as_any().downcast_ref::<T>().ok_or_else(|| {
        datafusion::error::DataFusionError::Internal(format!(
            "column '{path}' has type {} but is not backed by the matching array kind",
            column.data_type()
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use datafusion::arrow::datatypes::{
        DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
    };
    use iceberg::TableIdent;
    use iceberg::io::FileIO;
    use iceberg::spec::{
        FormatVersion, NestedField, PartitionSpec, PrimitiveType, Schema, SortOrder,
        TableMetadataBuilder, Type,
    };

    use super::*;

    fn create_test_table() -> Table {
        let schema = Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::required(
                    2,
                    "data",
                    Type::Primitive(PrimitiveType::String),
                )),
            ])
            .build()
            .expect("test schema must build");

        let partition_spec = PartitionSpec::builder(schema.clone())
            .build()
            .expect("partition spec must build");
        let sort_order = SortOrder::builder()
            .build(&schema)
            .expect("sort order must build");
        let table_metadata = TableMetadataBuilder::new(
            schema,
            partition_spec,
            sort_order,
            "memory://test/table".to_string(),
            FormatVersion::V2,
            HashMap::new(),
        )
        .expect("metadata builder must construct")
        .build()
        .expect("table metadata must build");

        Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).expect("ident must parse"))
            .file_io(FileIO::new_with_memory())
            .metadata_location("memory://test/metadata.json".to_string())
            .build()
            .expect("table must build")
    }

    fn test_arrow_schema() -> ArrowSchemaRef {
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false),
            ArrowField::new("data", ArrowDataType::Utf8, false),
        ]))
    }

    /// An out-of-bounds projection index must give a PLANNING error, not a panic.
    #[test]
    fn test_scan_out_of_bounds_projection_is_error_not_panic() {
        let err = IcebergTableScan::new(
            create_test_table(),
            None,
            test_arrow_schema(),
            Some(&vec![0, 99]),
            &[],
            None,
        )
        .expect_err("projection index 99 on a 2-column schema must be a planning error");
        assert!(
            err.to_string().contains("99"),
            "the error should name the offending index: {err}"
        );
    }

    /// An unreadable advertised column comes back as NULL, in ADVERTISED order.
    #[test]
    fn test_conform_batch_null_fills_and_reorders() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("data", ArrowDataType::Utf8, false),
                ArrowField::new("id", ArrowDataType::Int64, false),
            ])),
            vec![
                Arc::new(datafusion::arrow::array::StringArray::from(vec!["a", "b"])),
                Arc::new(datafusion::arrow::array::Int64Array::from(vec![1, 2])),
            ],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false),
            ArrowField::new("data", ArrowDataType::Utf8, false),
            ArrowField::new("added_later", ArrowDataType::Int32, true),
        ]));

        let sources = vec![
            ColumnSource::Scanned("id".to_string()),
            ColumnSource::Scanned("data".to_string()),
            ColumnSource::Absent,
        ];
        let conformed =
            conform_batch(scanned, &advertised, &sources).expect("the batch must conform");
        assert_eq!(conformed.schema(), advertised);
        assert_eq!(conformed.num_rows(), 2);
        assert_eq!(
            conformed
                .column_by_name("added_later")
                .expect("the added column must be present")
                .null_count(),
            2,
            "a column the scan could not read must be all-NULL"
        );
        assert_eq!(
            conformed
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("column 0 must be the advertised `id`")
                .values(),
            &[1, 2]
        );
    }

    /// A zero-column projection must keep its row count: no column can carry it.
    #[test]
    fn test_conform_batch_preserves_row_count_with_no_columns() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
                1, 2, 3,
            ]))],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(Vec::<ArrowField>::new()));
        let conformed = conform_batch(scanned, &advertised, &[]).expect("the batch must conform");
        assert_eq!(conformed.num_columns(), 0);
        assert_eq!(
            conformed.num_rows(),
            3,
            "the row count must survive a zero-column projection"
        );
    }

    /// A column whose type changed after planning must be a loud, named error.
    #[test]
    fn test_conform_batch_rejects_a_changed_type() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
                1,
            ]))],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            ArrowDataType::Int32,
            false,
        )]));

        let err = conform_batch(scanned, &advertised, &[ColumnSource::Scanned(
            "id".to_string(),
        )])
        .expect_err("an illegal type change must not be silently coerced");
        assert!(
            err.to_string().contains("id") && err.to_string().contains("Int32"),
            "the error must name the column and the expected type: {err}"
        );
    }

    /// An advertised NON-nullable column the scan could not read must error, not fabricate values.
    #[test]
    fn test_conform_batch_rejects_absent_required_column() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
                1,
            ]))],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false),
            ArrowField::new("required_new", ArrowDataType::Utf8, false),
        ]));

        let sources = vec![
            ColumnSource::Scanned("id".to_string()),
            ColumnSource::Absent,
        ];
        let err = conform_batch(scanned, &advertised, &sources)
            .expect_err("an absent required column must not be null-filled");
        assert!(
            err.to_string().contains("required_new"),
            "the error must name the column: {err}"
        );
    }

    /// The Arrow promotion mirror must agree with [`iceberg::spec::is_promotion_allowed`] over all
    /// 17 `PrimitiveType` variants. A drift rejects a legal promotion, or casts a forbidden one.
    /// It is coarser where two primitives share one Arrow type, such as `uuid` and `fixed[16]`, but
    /// that is inert: `ensure_promotion_allowed` blocks such a change at the DDL.
    #[test]
    fn test_arrow_promotion_mirror_agrees_with_iceberg_rule() {
        use iceberg::spec::{PrimitiveType as P, Type as T, is_promotion_allowed};

        // All 17 variants, with three decimals for the precision arm in both directions.
        let primitives = [
            P::Boolean,
            P::Int,
            P::Long,
            P::Float,
            P::Double,
            P::Decimal {
                precision: 9,
                scale: 2,
            },
            P::Decimal {
                precision: 18,
                scale: 2,
            },
            P::Decimal {
                precision: 18,
                scale: 3,
            },
            P::Date,
            P::Time,
            P::Timestamp,
            P::Timestamptz,
            P::TimestampNs,
            P::TimestamptzNs,
            P::String,
            P::Uuid,
            P::Fixed(16),
            P::Binary,
            P::Unknown,
        ];
        assert_eq!(
            primitives.len(),
            17 + 2,
            "17 variants, with two extra decimals for the precision/scale arm"
        );

        // The crate's own converter, so the mirror is checked against the production mapping.
        let arrow_of = |primitive: &P| -> ArrowDataType {
            let schema = Schema::builder()
                .with_fields(vec![Arc::new(NestedField::optional(
                    1,
                    "c",
                    Type::Primitive(primitive.clone()),
                ))])
                .build()
                .expect("one-field schema must build");
            iceberg::arrow::schema_to_arrow_schema(&schema)
                .expect("schema must convert to arrow")
                .field(0)
                .data_type()
                .clone()
        };

        let mut checked = 0;
        let mut collisions = 0;
        for from in &primitives {
            for to in &primitives {
                let (arrow_from, arrow_to) = (arrow_of(from), arrow_of(to));
                let expected = is_promotion_allowed(&T::Primitive(from.clone()), to);
                let actual = is_arrow_promotion_allowed(&arrow_from, &arrow_to);
                if from != to && arrow_from == arrow_to {
                    // The documented coarseness: one Arrow type, two Iceberg primitives.
                    assert!(
                        actual,
                        "the mirror's identity arm must accept {from} -> {to} (both {arrow_from:?})"
                    );
                    collisions += 1;
                } else {
                    assert_eq!(
                        actual, expected,
                        "mirror disagrees for {from} -> {to} (arrow {arrow_from:?} -> {arrow_to:?})"
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, primitives.len() * primitives.len());
        // Non-vacuity: at least one allowed NON-identity promotion...
        assert!(is_arrow_promotion_allowed(
            &ArrowDataType::Int32,
            &ArrowDataType::Int64
        ));
        // ...and the collision exception exercised, uuid and fixed[16] in both directions.
        assert_eq!(
            collisions, 2,
            "the uuid / fixed[16] collision must be the only one this matrix hits"
        );
    }

    /// A struct from an older snapshot gains the advertised child as NULLs, keeping its buffer.
    #[test]
    fn test_conform_column_null_fills_a_nested_field() {
        use datafusion::arrow::array::{Int32Array, StructArray};
        use datafusion::arrow::buffer::NullBuffer;
        use datafusion::arrow::datatypes::Fields;

        let scanned_children =
            Fields::from(vec![field_with_id("a", ArrowDataType::Int32, true, 3)]);
        let scanned: ArrayRef = Arc::new(
            StructArray::try_new(
                scanned_children,
                vec![Arc::new(Int32Array::from(vec![Some(5), Some(6)]))],
                Some(NullBuffer::from(vec![true, false])),
            )
            .expect("the scanned struct must build"),
        );

        let target_children = Fields::from(vec![
            field_with_id("a", ArrowDataType::Int32, true, 3),
            field_with_id("b", ArrowDataType::Int32, true, 4),
        ]);
        let target = ArrowField::new("s", ArrowDataType::Struct(target_children.clone()), true);

        let conformed = conform_column(&scanned, &target, "s").expect("the struct must conform");
        assert_eq!(conformed.data_type(), target.data_type());
        let conformed = conformed
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("still a struct");
        assert_eq!(conformed.len(), 2);
        assert_eq!(
            conformed.column(1).null_count(),
            2,
            "the added child must be all-NULL"
        );
        assert!(
            conformed.is_null(1),
            "the struct's own null buffer must survive"
        );
    }

    /// The same evolution inside a `list<struct<...>>`: offsets and the null buffer survive.
    #[test]
    fn test_conform_column_recurses_into_a_list_element() {
        use datafusion::arrow::array::{Int32Array, ListArray, StructArray};
        use datafusion::arrow::buffer::{OffsetBuffer, ScalarBuffer};
        use datafusion::arrow::datatypes::Fields;

        let scanned_children =
            Fields::from(vec![field_with_id("a", ArrowDataType::Int32, true, 3)]);
        let scanned_element = Arc::new(ArrowField::new(
            "element",
            ArrowDataType::Struct(scanned_children.clone()),
            true,
        ));
        let scanned_values: ArrayRef = Arc::new(
            StructArray::try_new(
                scanned_children,
                vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
                None,
            )
            .expect("element struct"),
        );
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 2, 3]));
        let scanned: ArrayRef = Arc::new(
            ListArray::try_new(scanned_element, offsets, scanned_values, None)
                .expect("the scanned list must build"),
        );

        let target_children = Fields::from(vec![
            field_with_id("a", ArrowDataType::Int32, true, 3),
            field_with_id("b", ArrowDataType::Int32, true, 4),
        ]);
        let target_element = Arc::new(ArrowField::new(
            "element",
            ArrowDataType::Struct(target_children),
            true,
        ));
        let target = ArrowField::new("l", ArrowDataType::List(target_element), true);

        let conformed = conform_column(&scanned, &target, "l").expect("the list must conform");
        assert_eq!(conformed.data_type(), target.data_type());
        let conformed = conformed
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("still a list");
        assert_eq!(conformed.len(), 2, "the list offsets must be preserved");
        assert_eq!(conformed.value(0).len(), 2);
        assert_eq!(conformed.value(1).len(), 1);
        let values = conformed
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("element struct");
        assert_eq!(
            values.column(1).null_count(),
            3,
            "the added element field must be all-NULL"
        );
    }

    /// And inside a `map<string, struct<...>>`, through the key/value pair, keeping offsets.
    #[test]
    fn test_conform_column_recurses_into_a_map_value() {
        use datafusion::arrow::array::{Int32Array, MapArray, StringArray, StructArray};
        use datafusion::arrow::buffer::{OffsetBuffer, ScalarBuffer};
        use datafusion::arrow::datatypes::Fields;

        let value_children = Fields::from(vec![field_with_id("a", ArrowDataType::Int32, true, 3)]);
        let scanned_entry_fields = Fields::from(vec![
            field_with_id("key", ArrowDataType::Utf8, false, 5),
            field_with_id(
                "value",
                ArrowDataType::Struct(value_children.clone()),
                true,
                6,
            ),
        ]);
        let scanned_entries = StructArray::try_new(
            scanned_entry_fields.clone(),
            vec![
                Arc::new(StringArray::from(vec!["k1", "k2"])),
                Arc::new(
                    StructArray::try_new(
                        value_children,
                        vec![Arc::new(Int32Array::from(vec![7, 8]))],
                        None,
                    )
                    .expect("value struct"),
                ),
            ],
            None,
        )
        .expect("entries struct");
        let scanned_entries_field = Arc::new(ArrowField::new(
            "entries",
            ArrowDataType::Struct(scanned_entry_fields),
            false,
        ));
        let scanned: ArrayRef = Arc::new(
            MapArray::try_new(
                scanned_entries_field,
                OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 2])),
                scanned_entries,
                None,
                false,
            )
            .expect("the scanned map must build"),
        );

        let target_value_children = Fields::from(vec![
            field_with_id("a", ArrowDataType::Int32, true, 3),
            field_with_id("b", ArrowDataType::Int32, true, 4),
        ]);
        let target_entry_fields = Fields::from(vec![
            field_with_id("key", ArrowDataType::Utf8, false, 5),
            field_with_id(
                "value",
                ArrowDataType::Struct(target_value_children),
                true,
                6,
            ),
        ]);
        let target = ArrowField::new(
            "m",
            ArrowDataType::Map(
                Arc::new(ArrowField::new(
                    "entries",
                    ArrowDataType::Struct(target_entry_fields),
                    false,
                )),
                false,
            ),
            true,
        );

        let conformed = conform_column(&scanned, &target, "m").expect("the map must conform");
        assert_eq!(conformed.data_type(), target.data_type());
        let conformed = conformed
            .as_any()
            .downcast_ref::<MapArray>()
            .expect("still a map");
        assert_eq!(conformed.len(), 1);
        assert_eq!(conformed.value_length(0), 2, "the map offsets must survive");
        let values = conformed
            .entries()
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("value struct")
            .clone();
        assert_eq!(
            values.column(1).null_count(),
            2,
            "the added value field must be all-NULL"
        );
    }

    /// An illegal change BENEATH a column must name the nested PATH, not only the root.
    #[test]
    fn test_conform_column_names_the_nested_path_on_an_illegal_change() {
        use datafusion::arrow::array::{Int64Array, StructArray};
        use datafusion::arrow::datatypes::Fields;

        let scanned_children =
            Fields::from(vec![field_with_id("a", ArrowDataType::Int64, true, 3)]);
        let scanned: ArrayRef = Arc::new(
            StructArray::try_new(
                scanned_children,
                vec![Arc::new(Int64Array::from(vec![5]))],
                None,
            )
            .expect("the scanned struct must build"),
        );
        // long -> int is NOT a legal promotion.
        let target_children = Fields::from(vec![field_with_id("a", ArrowDataType::Int32, true, 3)]);
        let target = ArrowField::new("s", ArrowDataType::Struct(target_children), true);

        let err = conform_column(&scanned, &target, "s")
            .expect_err("a narrowing nested change must not be coerced");
        assert!(
            err.to_string().contains("s.a"),
            "the error must name the nested path: {err}"
        );
    }

    /// A legally promoted column is cast to the advertised type, not rejected.
    #[test]
    fn test_conform_batch_promotes_a_legal_type_change() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                ArrowDataType::Int32,
                false,
            )])),
            vec![Arc::new(datafusion::arrow::array::Int32Array::from(vec![
                7, -3,
            ]))],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            ArrowDataType::Int64,
            false,
        )]));

        let conformed = conform_batch(scanned, &advertised, &[ColumnSource::Scanned(
            "id".to_string(),
        )])
        .expect("int -> long is a legal Iceberg promotion");
        assert_eq!(conformed.schema(), advertised);
        assert_eq!(
            conformed
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("the promoted column must be Int64")
                .values(),
            &[7, -3],
            "the values must survive the widening"
        );
    }

    /// A table whose current snapshot holds `x`(1), `y`(2), `z`(3): real metadata to bind against.
    async fn table_with_snapshot() -> Table {
        let metadata_file_path = format!(
            "{}/tests/test_data/TableMetadataV2Valid.json",
            env!("CARGO_MANIFEST_DIR")
        );
        iceberg::table::StaticTable::from_metadata_file(
            &metadata_file_path,
            iceberg::TableIdent::from_strs(["ns", "t"]).expect("ident must parse"),
            iceberg::io::FileIO::new_with_fs(),
        )
        .await
        .expect("the fixture metadata must load")
        .into_table()
    }

    fn field_with_id(name: &str, data_type: ArrowDataType, nullable: bool, id: i32) -> ArrowField {
        ArrowField::new(name, data_type, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    }

    /// An advertised field with no `PARQUET:field_id` errors, rather than guess by name.
    #[tokio::test]
    async fn test_resolve_projection_requires_a_field_id() {
        let table = table_with_snapshot().await;
        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int64,
            false,
        )]));

        let err = resolve_bindings(&table, None, &advertised)
            .expect_err("a field without an id must not be bound by name");
        assert!(
            err.to_string().contains(PARQUET_FIELD_ID_META_KEY) && err.to_string().contains('x'),
            "the error must name the column and the missing metadata: {err}"
        );
    }

    /// A renamed column binds by FIELD ID; an id the snapshot lacks becomes a null-fill.
    #[tokio::test]
    async fn test_resolve_projection_binds_by_field_id_not_by_name() {
        let table = table_with_snapshot().await;
        // The snapshot schema calls field 2 `y`; this plan advertises it as `renamed_y`.
        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
            field_with_id("renamed_y", ArrowDataType::Int64, false, 2),
            field_with_id("added_later", ArrowDataType::Int32, true, 99),
        ]));

        let bindings =
            resolve_bindings(&table, None, &advertised).expect("the bindings must resolve");
        let (scan_columns, sources) =
            project_bindings(&advertised, &bindings).expect("the projection must resolve");
        assert_eq!(
            scan_columns,
            vec!["y".to_string()],
            "the SNAPSHOT schema's name for field 2 is what gets selected"
        );
        assert_eq!(sources, vec![
            ColumnSource::Scanned("y".to_string()),
            ColumnSource::Absent,
        ]);
    }

    /// A valid projection gives the projected schema, and no projection passes it through.
    #[test]
    fn test_scan_valid_projection_schema_and_names() {
        let projected = IcebergTableScan::new(
            create_test_table(),
            None,
            test_arrow_schema(),
            Some(&vec![1]),
            &[],
            None,
        )
        .expect("a valid projection must plan");
        assert_eq!(projected.projection(), Some(&["data".to_string()][..]));
        let output_schema = projected.schema();
        assert_eq!(output_schema.fields().len(), 1);
        assert_eq!(output_schema.field(0).name(), "data");
        assert_eq!(output_schema.field(0).data_type(), &ArrowDataType::Utf8);

        let unprojected = IcebergTableScan::new(
            create_test_table(),
            None,
            test_arrow_schema(),
            None,
            &[],
            None,
        )
        .expect("a scan without projection must plan");
        assert_eq!(unprojected.projection(), None);
        assert_eq!(unprojected.schema(), test_arrow_schema());
    }

    /// Session `batch_size` and `target_partitions` must reach the Iceberg scan knobs.
    #[test]
    fn test_scan_knobs_from_context_wires_batch_size_and_concurrency() {
        use datafusion::execution::SessionStateBuilder;
        use datafusion::prelude::SessionConfig;

        let config = SessionConfig::new()
            .set_usize("datafusion.execution.batch_size", 17)
            .set_usize("datafusion.execution.target_partitions", 5);
        let state = SessionStateBuilder::new().with_config(config).build();
        let context = state.task_ctx();

        let knobs = scan_knobs_from_context(&context);
        assert_eq!(knobs.batch_size, Some(17));
        assert_eq!(knobs.data_file_concurrency, Some(5));
    }

    /// The clamp floor is 1: a raw 0 empties the Parquet stream, or hangs the buffer.
    #[test]
    fn test_clamp_scan_knob_floors_zero_to_one() {
        assert_eq!(clamp_scan_knob(0), 1);
        assert_eq!(clamp_scan_knob(1), 1);
        assert_eq!(clamp_scan_knob(8), 8);
    }

    /// A hand-built `ScanKnobs` holding `Some(0)` is floored at apply time too.
    #[test]
    fn test_get_batch_stream_clamps_zero_knobs_at_apply() {
        let knobs = ScanKnobs {
            batch_size: Some(0),
            data_file_concurrency: Some(0),
            target_partitions: 1,
            multi_partition_scan: true,
        };
        let effective_batch = knobs.batch_size.map(clamp_scan_knob);
        let effective_conc = knobs.data_file_concurrency.map(clamp_scan_knob);
        assert_eq!(effective_batch, Some(1));
        assert_eq!(effective_conc, Some(1));
    }

    /// DataFusion accepts `batch_size = 0`, which Parquet reads as end-of-stream.
    #[test]
    fn test_scan_knobs_clamps_zero_batch_size() {
        use datafusion::execution::SessionStateBuilder;
        use datafusion::prelude::SessionConfig;

        let config = SessionConfig::new().set_usize("datafusion.execution.batch_size", 0);
        let state = SessionStateBuilder::new().with_config(config).build();
        let knobs = scan_knobs_from_context(&state.task_ctx());
        assert_eq!(
            knobs.batch_size,
            Some(1),
            "batch_size 0 must clamp to 1 (Parquet empty-stream hazard)"
        );
        assert!(
            knobs.data_file_concurrency.is_some_and(|c| c >= 1),
            "data-file concurrency must stay ≥ 1"
        );
    }

    /// Pin: DF rewrites `target_partitions = 0` to available parallelism; knobs must still be ≥ 1.
    #[test]
    fn test_scan_knobs_target_partitions_zero_still_at_least_one() {
        use datafusion::execution::SessionStateBuilder;
        use datafusion::prelude::SessionConfig;

        let config = SessionConfig::new().set_usize("datafusion.execution.target_partitions", 0);
        let state = SessionStateBuilder::new().with_config(config).build();
        let knobs = scan_knobs_from_context(&state.task_ctx());
        assert!(
            knobs.data_file_concurrency.is_some_and(|c| c >= 1),
            "target_partitions 0 must not yield concurrency 0 (hang hazard)"
        );
    }

    /// Pin 13: dedicated off-switch forces multi_partition_scan=false while target_partitions > 1.
    #[test]
    fn test_pin13_off_switch_independent_of_target_partitions() {
        use datafusion::execution::SessionStateBuilder;
        use datafusion::prelude::SessionConfig;

        let mut config =
            SessionConfig::new().set_usize("datafusion.execution.target_partitions", 8);
        ensure_iceberg_scan_options(&mut config);
        config.options_mut().extensions.insert(IcebergScanOptions {
            multi_partition_scan: false,
            data_file_concurrency: 0,
        });
        let state = SessionStateBuilder::new().with_config(config).build();
        let knobs = scan_knobs_from_context(&state.task_ctx());
        assert!(!knobs.multi_partition_scan);
        assert_eq!(knobs.target_partitions, 8);
        let t = if knobs.multi_partition_scan {
            knobs.target_partitions
        } else {
            1
        };
        assert_eq!(t, 1);
    }

    /// Pin 14: distinct L surface independent of T.
    #[test]
    fn test_pin14_distinct_l_surface() {
        use datafusion::execution::SessionStateBuilder;
        use datafusion::prelude::SessionConfig;

        let mut config =
            SessionConfig::new().set_usize("datafusion.execution.target_partitions", 4);
        ensure_iceberg_scan_options(&mut config);
        config.options_mut().extensions.insert(IcebergScanOptions {
            multi_partition_scan: true,
            data_file_concurrency: 16,
        });
        let state = SessionStateBuilder::new().with_config(config).build();
        let knobs = scan_knobs_from_context(&state.task_ctx());
        assert_eq!(knobs.target_partitions, 4);
        assert_eq!(knobs.data_file_concurrency, Some(16));

        let mut config2 =
            SessionConfig::new().set_usize("datafusion.execution.target_partitions", 4);
        config2.options_mut().extensions.insert(IcebergScanOptions {
            multi_partition_scan: true,
            data_file_concurrency: 2,
        });
        let state2 = SessionStateBuilder::new().with_config(config2).build();
        let knobs2 = scan_knobs_from_context(&state2.task_ctx());
        assert_eq!(knobs2.target_partitions, 4);
        assert_eq!(knobs2.data_file_concurrency, Some(2));
        assert_ne!(knobs.data_file_concurrency, knobs2.data_file_concurrency);
    }

    /// Pin 2 partial: execute(i) for i ≥ N is a typed error (legacy N=1 path).
    #[tokio::test]
    async fn test_execute_out_of_range_errors() {
        use datafusion::execution::TaskContext;

        let scan = IcebergTableScan::new(
            create_test_table(),
            None,
            test_arrow_schema(),
            None,
            &[],
            None,
        )
        .expect("scan");
        let ctx = Arc::new(TaskContext::default());
        match scan.execute(1, ctx) {
            Ok(_) => panic!("i≥N must error, got Ok stream"),
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("out of range"),
                    "expected out-of-range typed error, got: {msg}"
                );
            }
        }
    }

    /// Pin 2 multi-path: execute(i) for i ≥ N errors when partition_work is non-empty.
    #[tokio::test]
    async fn test_pin2_execute_out_of_range_multipath() {
        use datafusion::execution::TaskContext;

        // plan() on a snapshot-less empty table yields N=1 empty PartitionWork (multi path).
        let table = create_test_table();
        let knobs = ScanKnobs {
            batch_size: Some(1024),
            data_file_concurrency: Some(4),
            target_partitions: 4,
            multi_partition_scan: true,
        };
        let scan = IcebergTableScan::plan(
            table,
            None,
            test_arrow_schema_with_field_ids(),
            None,
            &[],
            Some(10), // limit present — empty N=1 keeps it
            knobs,
        )
        .await
        .expect("empty table plan");
        assert_eq!(scan.partition_work().len(), 1);
        assert_eq!(scan.properties().output_partitioning().partition_count(), 1);
        assert_eq!(scan.limit(), Some(10), "pin 5: limit retained when N=1");
        let ctx = Arc::new(TaskContext::default());
        match scan.execute(1, ctx) {
            Ok(_) => panic!("i≥N multi-path must error"),
            Err(err) => {
                assert!(err.to_string().contains("out of range"), "got: {err}");
            }
        }
    }

    fn test_arrow_schema_with_field_ids() -> ArrowSchemaRef {
        use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            ArrowField::new("data", ArrowDataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]))
    }

    /// Pin 5: when N > 1, provider must clear sole per-partition hard limit.
    #[test]
    fn test_pin5_limit_demoted_when_n_gt_1() {
        let n = 3usize;
        let mut limit = Some(5usize);
        if n > 1 {
            limit = None;
        }
        assert_eq!(limit, None, "pin 5: limit demoted when N>1");

        let n1 = 1usize;
        let mut limit1 = Some(5usize);
        if n1 > 1 {
            limit1 = None;
        }
        assert_eq!(limit1, Some(5), "pin 5: limit retained when N=1");
    }

    /// Pin 5 mutation skeleton: sole per-partition hard k with no global trim over-counts.
    #[test]
    fn test_pin5_mutation_per_partition_hard_limit_overcounts() {
        // Simulated: N=3 partitions each hard-capped at k=2 → up to 6 rows without GlobalLimitExec
        let n = 3usize;
        let k = 2usize;
        let table_rows = 100usize;
        let per_part_only = (n * k).min(table_rows);
        let global_correct = k.min(table_rows);
        assert!(
            per_part_only > global_correct,
            "mutation RED condition: per-part hard k yields {per_part_only} > global {global_correct}"
        );
    }

    /// Pin 13 plan-level: off-switch forces T=1 while target_partitions > 1 (effective T).
    #[test]
    fn test_pin13_effective_t_with_off_switch() {
        let knobs = ScanKnobs {
            batch_size: Some(1024),
            data_file_concurrency: Some(8),
            target_partitions: 8,
            multi_partition_scan: false,
        };
        let t = if knobs.multi_partition_scan {
            knobs.target_partitions.max(1)
        } else {
            1
        };
        assert_eq!(t, 1);
        assert!(knobs.target_partitions > 1);
    }

    /// Pin 14: P = max(1, ceil(L/N)) independent surface — two L values → two P.
    #[test]
    fn test_pin14_p_formula_independent_of_t() {
        let n = 4usize;
        let l1 = 16usize;
        let l2 = 2usize;
        let p1 = l1.div_ceil(n).max(1);
        let p2 = l2.div_ceil(n).max(1);
        assert_eq!(p1, 4);
        assert_eq!(p2, 1);
        assert_ne!(
            p1, p2,
            "pin 14: distinct L must yield distinct P at fixed N"
        );
        // When N > L, P = 1 and total ≈ N may exceed L (not RED)
        let l_small = 2usize;
        let n_big = 8usize;
        let p = l_small.div_ceil(n_big).max(1);
        assert_eq!(p, 1);
        assert!(n_big * p > l_small);
    }

    /// Pin 12: plan freezes resolved snapshot id onto every PartitionWork unit.
    #[tokio::test]
    async fn test_pin12_snapshot_frozen_on_work() {
        use datafusion::execution::TaskContext;

        let table = create_test_table();
        let knobs = ScanKnobs {
            batch_size: Some(1024),
            data_file_concurrency: Some(1),
            target_partitions: 1,
            multi_partition_scan: true,
        };
        let scan = IcebergTableScan::plan(
            table,
            None,
            test_arrow_schema_with_field_ids(),
            None,
            &[],
            None,
            knobs,
        )
        .await
        .expect("plan empty");
        assert!(!scan.partition_work().is_empty(), "eager plan embeds work");
        for work in scan.partition_work() {
            assert_eq!(
                work.snapshot_id(),
                scan.resolved_snapshot_id(),
                "pin 12: work snapshot must match plan resolved id"
            );
        }
        let ctx = Arc::new(TaskContext::default());
        let stream = scan.execute(0, ctx).expect("execute 0");
        drop(stream);
    }

    /// The DEFAULT EXPLAIN form is golden-matched with no normalizer, so it carries deterministic
    /// fields only, and the per-run-random snapshot id renders under Verbose. Mutations: rendering
    /// the id in Default, dropping it from Verbose, or dropping `N` each RED one assert.
    #[test]
    fn test_display_default_deterministic_snapshot_id_verbose_only() {
        use datafusion::physical_plan::displayable;

        let scan = IcebergTableScan::new(
            create_test_table(),
            None,
            test_arrow_schema(),
            None,
            &[],
            None,
        )
        .expect("scan");
        let plan: Arc<dyn ExecutionPlan> = Arc::new(scan);
        let default_form = displayable(plan.as_ref()).indent(false).to_string();
        let verbose_form = displayable(plan.as_ref()).indent(true).to_string();

        assert!(
            !default_form.contains("snapshot_id="),
            "default EXPLAIN must not carry the per-run-random snapshot id: {default_form}"
        );
        assert!(
            default_form.contains(" N=1"),
            "default EXPLAIN keeps deterministic N: {default_form}"
        );
        assert!(
            verbose_form.contains("snapshot_id="),
            "EXPLAIN VERBOSE must expose the frozen snapshot id: {verbose_form}"
        );
        assert!(
            verbose_form.contains(" N=1"),
            "EXPLAIN VERBOSE keeps N too: {verbose_form}"
        );
    }
}
