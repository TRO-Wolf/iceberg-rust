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

//! `UPDATE` physical plan. Exact `WHERE` stays a DataFusion `PhysicalExpr`. Iceberg scan
//! filters are prune-only.

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use iceberg::Catalog;
use iceberg::expr::Predicate;
use iceberg::table::Table;

use super::delete::{
    IcebergDeleteExec, IsolationLevel, WriteMode, copy_on_write_update, merge_on_read_update,
};

/// `UPDATE … SET … WHERE` plan. It applies the assignments, commits, and counts the rows.
pub(crate) struct IcebergUpdateExec {
    table: Table,
    catalog: Arc<dyn Catalog>,
    /// The WHERE clause as a `PhysicalExpr`, or `None` to update every row.
    predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Iceberg file prune only. Never replaces [`predicate`].
    prune: Option<Predicate>,
    /// The `SET` assignments: `(table-schema column index, new-value PhysicalExpr)`.
    assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
    mode: WriteMode,
    /// The §5 isolation level, resolved at plan time from `write.update.isolation-level`.
    isolation: IsolationLevel,
    table_schema: SchemaRef,
    count_schema: SchemaRef,
    plan_properties: Arc<PlanProperties>,
    commit_branch: Option<String>,
}

impl IcebergUpdateExec {
    #[allow(clippy::too_many_arguments)] // SET assignments; DeleteExec is already at the 7-arg cap
    pub(crate) fn new(
        table: Table,
        catalog: Arc<dyn Catalog>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        prune: Option<Predicate>,
        assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
        mode: WriteMode,
        isolation: IsolationLevel,
        table_schema: SchemaRef,
        commit_branch: Option<String>,
    ) -> Self {
        let count_schema = IcebergDeleteExec::make_count_schema();
        let plan_properties = IcebergDeleteExec::compute_properties(Arc::clone(&count_schema));
        Self {
            table,
            catalog,
            predicate,
            prune,
            assignments,
            mode,
            isolation,
            table_schema,
            count_schema,
            plan_properties,
            commit_branch,
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
        let prune = self.prune.clone();
        let assignments = self.assignments.clone();
        let mode = self.mode;
        let isolation = self.isolation;
        let table_schema = Arc::clone(&self.table_schema);
        let count_schema = Arc::clone(&self.count_schema);
        let commit_branch = self.commit_branch.clone();

        let stream = futures::stream::once(async move {
            let updated = match mode {
                WriteMode::MergeOnRead => {
                    merge_on_read_update(
                        &table,
                        catalog.as_ref(),
                        predicate,
                        prune,
                        &assignments,
                        &table_schema,
                        isolation,
                        commit_branch.as_deref(),
                    )
                    .await?
                }
                WriteMode::CopyOnWrite => {
                    copy_on_write_update(
                        &table,
                        catalog.as_ref(),
                        predicate,
                        prune,
                        &assignments,
                        &table_schema,
                        isolation,
                        commit_branch.as_deref(),
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
