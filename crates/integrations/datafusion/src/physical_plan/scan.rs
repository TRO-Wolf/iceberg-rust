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

use std::pin::Pin;
use std::sync::Arc;
use std::vec;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::common::stats::{Precision, Statistics};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties};
use datafusion::prelude::Expr;
use futures::{Stream, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::scan::{PartitionWork, stream_partition_work};
use iceberg::table::Table;

use super::conform::{
    ColumnSource, conform_batch, strip_nested_metadata_from_record_batch,
    strip_nested_metadata_from_schema,
};
#[cfg(test)]
use super::conform::{conform_column, is_arrow_promotion_allowed};
use super::expr_to_predicate::scan_predicates;
use super::scan_helpers::{exact_table_row_count, project_bindings, resolve_bindings};
pub use super::scan_knobs::{IcebergScanOptions, ensure_iceberg_scan_options};
pub(crate) use super::scan_knobs::{
    ScanKnobs, clamp_scan_knob, get_batch_stream, scan_knobs_from_context,
};
use crate::table::uuid_text;
use crate::to_datafusion_error;

/// Manages the scanning process of an Iceberg [`Table`]. [`IcebergTableScan::plan`] assigns the
/// work of core `plan_tasks` into `N` [`PartitionWork`] units, as `UnknownPartitioning(N)`.
#[derive(Debug)]
pub struct IcebergTableScan {
    table: Table,
    /// `None` means the current snapshot at plan time.
    snapshot_id: Option<i64>,
    project_current_schema: bool,
    /// Concrete snapshot id resolved at plan time and frozen on the node (pin 12).
    resolved_snapshot_id: i64,
    plan_properties: Arc<PlanProperties>,
    projection: Option<Vec<String>>,
    /// The SCANNED snapshot's name for each advertised field id. See [`IcebergTableScan::new`].
    scan_columns: Vec<String>,
    /// How each advertised output column is produced, parallel to the advertised schema's fields.
    sources: Vec<ColumnSource>,
    conform_schema: ArrowSchemaRef,
    predicates: Option<Predicate>,
    /// Optional row limit. It applies only when `N = 1`, because a per-partition cap over-counts,
    /// so `GlobalLimitExec` owns the limit above that.
    limit: Option<usize>,
    /// Empty when built by [`Self::new`] without planning, which takes the single-stream path.
    partition_work: Vec<PartitionWork>,
    exact_row_count: Option<usize>,
    /// Per-partition data-file concurrency `P = max(1, ceil(L/N))`.
    per_partition_concurrency: usize,
    batch_size: Option<usize>,
    pub(crate) row_selection_enabled: bool,
    uuid_as_string: bool,
    text_schema: Option<ArrowSchemaRef>,
}

impl IcebergTableScan {
    /// Creates a new [`IcebergTableScan`] object. # Errors Fails when `projection` holds an index
    /// outside `schema`. # Notes The advertised schema is a contract: every parent operator was
    /// built against it.
    pub(crate) fn new(
        table: Table,
        snapshot_id: Option<i64>,
        project_current_schema: bool,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Self> {
        // The FULL schema, not the projection: a pushed filter may reference an unprojected column.
        let bindings = resolve_bindings(&table, snapshot_id, &schema, project_current_schema)?;

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
        let emit_schema = Arc::new(strip_nested_metadata_from_schema(&output_schema));
        let plan_properties = Self::compute_properties(emit_schema, 1);
        let predicates = scan_predicates(
            &table,
            snapshot_id,
            filters,
            &bindings,
            project_current_schema,
        )?;

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
            project_current_schema,
            resolved_snapshot_id,
            plan_properties,
            projection,
            scan_columns,
            sources,
            conform_schema: output_schema,
            predicates,
            limit,
            partition_work: Vec::new(),
            exact_row_count: None,
            per_partition_concurrency: 1,
            batch_size: None,
            row_selection_enabled: true,
            uuid_as_string: false,
            text_schema: None,
        })
    }

    /// Eager multi-partition plan, on when `T > 1` and the post-strip group count is above 1.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn plan(
        table: Table,
        snapshot_id: Option<i64>,
        project_current_schema: bool,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
        knobs: ScanKnobs,
    ) -> DFResult<Self> {
        let mut scan = Self::new(
            table.clone(),
            snapshot_id,
            project_current_schema,
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
        if project_current_schema {
            scan_builder = scan_builder.project_current_schema();
        }
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
        scan.exact_row_count = exact_table_row_count(&scan.partition_work);
        scan.per_partition_concurrency = p;
        scan.batch_size = knobs.batch_size.map(clamp_scan_knob);
        scan.row_selection_enabled = knobs.row_selection_enabled;
        scan.plan_properties = Self::compute_properties(scan.schema(), n);
        if knobs.uuid_as_string {
            let uuid_ids = uuid_text::collect_uuid_field_ids(table.metadata().current_schema());
            let text_schema = Arc::new(uuid_text::arrow_schema_with_uuid_as_text(
                &scan.conform_schema,
                &uuid_ids,
            ));
            scan.plan_properties = Self::compute_properties(
                Arc::new(strip_nested_metadata_from_schema(&text_schema)),
                n,
            );
            scan.uuid_as_string = true;
            scan.text_schema = Some(text_schema);
        }
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

    fn partition_statistics(&self, partition: Option<usize>) -> DFResult<Arc<Statistics>> {
        if let Some(index) = partition {
            let count = self.properties().partitioning.partition_count();
            if index >= count {
                return Err(DataFusionError::Internal(format!(
                    "Invalid partition index: {index}, the partition count is {count}"
                )));
            }
        }
        if self.partition_work.is_empty() || self.limit.is_some() || partition.is_some() {
            return Ok(Arc::new(Statistics::new_unknown(&self.schema())));
        }
        let mut statistics = Statistics::new_unknown(&self.schema());
        if let Some(rows) = self.exact_row_count {
            statistics.num_rows = Precision::Exact(rows);
        }
        Ok(Arc::new(statistics))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let conform_schema = self.conform_schema.clone();
        let sources = self.sources.clone();
        let text_schema = self.text_schema.clone();
        let uuid_as_string = self.uuid_as_string;

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
            let row_selection = self.row_selection_enabled;
            let footer_cache = self.table.footer_cache();
            let stream = stream_partition_work(
                file_io,
                &work,
                concurrency,
                batch_size,
                true,
                row_selection,
                footer_cache,
            )
            .map_err(to_datafusion_error)?
            .map_err(to_datafusion_error)
            .and_then(move |batch| {
                futures::future::ready(conform_render_strip(
                    batch,
                    &conform_schema,
                    &sources,
                    &text_schema,
                    uuid_as_string,
                ))
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
            self.project_current_schema,
            self.scan_columns.clone(),
            self.predicates.clone(),
            knobs,
        );
        let stream = futures::stream::once(fut)
            .try_flatten()
            .and_then(move |batch| {
                futures::future::ready(conform_render_strip(
                    batch,
                    &conform_schema,
                    &sources,
                    &text_schema,
                    uuid_as_string,
                ))
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

fn conform_render_strip(
    batch: RecordBatch,
    conform_schema: &ArrowSchemaRef,
    sources: &[ColumnSource],
    text_schema: &Option<ArrowSchemaRef>,
    uuid_as_string: bool,
) -> DFResult<RecordBatch> {
    let batch = conform_batch(batch, conform_schema, sources)?;
    let batch = match (uuid_as_string, text_schema) {
        (true, Some(text)) => uuid_text::render_batch_uuid_as_text(batch, text)?,
        _ => batch,
    };
    strip_nested_metadata_from_record_batch(batch)
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

#[cfg(test)]
#[path = "scan_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "scan_pin_tests.rs"]
mod pin_tests;
