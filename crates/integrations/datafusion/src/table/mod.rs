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

//! Iceberg table providers for DataFusion.
//!
//! Two table provider implementations:
//!
//! - [`IcebergTableProvider`], catalog-backed with metadata refresh. Use it to write.
//! - [`IcebergStaticTableProvider`], read-only over one snapshot. Use it for time travel.

mod loaded;
pub mod metadata_table;
mod static_provider;
pub mod table_provider_factory;

use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::catalog::Session;
use datafusion::common::{DFSchema, DataFusionError};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::inspect::MetadataTableType;
use iceberg::table::Table;
use iceberg::{Catalog, NamespaceIdent, Result, TableIdent};
use metadata_table::IcebergMetadataTableProvider;
pub use static_provider::IcebergStaticTableProvider;

use crate::error::to_datafusion_error;
use crate::physical_plan::commit::IcebergCommitExec;
use crate::physical_plan::conform::strip_nested_metadata_from_schema;
use crate::physical_plan::delete::{
    IcebergDeleteExec, IsolationLevel, WRITE_DELETE_ISOLATION_LEVEL, WRITE_DELETE_MODE,
    WRITE_UPDATE_ISOLATION_LEVEL, WRITE_UPDATE_MODE, WriteMode,
};
use crate::physical_plan::expr_to_predicate::convert_filters_to_predicate;
use crate::physical_plan::project::project_with_partition;
use crate::physical_plan::repartition::repartition;
use crate::physical_plan::scan::IcebergTableScan;
use crate::physical_plan::sort::sort_for_write;
use crate::physical_plan::update::IcebergUpdateExec;
use crate::physical_plan::write::IcebergWriteExec;

/// Catalog-backed table provider. It loads fresh table metadata on every scan and write. For
/// read-only access to one snapshot, use [`IcebergStaticTableProvider`].
#[derive(Debug, Clone)]
pub struct IcebergTableProvider {
    pub(crate) catalog: Arc<dyn Catalog>,
    pub(crate) table_ident: TableIdent,
    /// FIXED for the life of the instance: DataFusion stores ordinals against it.
    pub(crate) schema: ArrowSchemaRef,
    pub(crate) commit_branch: Option<String>,
    pub(crate) planning_table: Option<Table>,
}

impl IcebergTableProvider {
    /// Creates a catalog-backed provider. Writes land on `main` until [`Self::with_commit_branch`].
    pub async fn try_new(
        catalog: Arc<dyn Catalog>,
        namespace: NamespaceIdent,
        name: impl Into<String>,
    ) -> Result<Self> {
        let table_ident = TableIdent::new(namespace, name.into());
        let table = catalog.load_table(&table_ident).await?;
        let schema = Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?);

        Ok(IcebergTableProvider {
            catalog,
            table_ident,
            schema,
            commit_branch: None,
            planning_table: None,
        })
    }

    /// Returns a NEW provider for the same table, advertising its current schema. This one is left
    /// untouched, so plans already built against it stay valid. A caller going through
    /// [`crate::IcebergCatalogProvider`] never needs it: each query resolves a fresh provider.
    pub async fn refreshed(&self) -> Result<Self> {
        let table = self.catalog.load_table(&self.table_ident).await?;
        Ok(IcebergTableProvider {
            catalog: self.catalog.clone(),
            table_ident: self.table_ident.clone(),
            schema: Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?),
            commit_branch: self.commit_branch.clone(),
            planning_table: None,
        })
    }

    /// Scan and commit snapshot-producing DML against `branch` instead of `main`. Java `SnapshotUpdate.toBranch`.
    pub fn with_commit_branch(mut self, branch: impl Into<String>) -> Self {
        self.commit_branch = Some(branch.into());
        self
    }

    pub(crate) async fn metadata_table(
        &self,
        r#type: MetadataTableType,
    ) -> Result<IcebergMetadataTableProvider> {
        let table = self.catalog.load_table(&self.table_ident).await?;
        IcebergMetadataTableProvider::try_new(table, r#type)
    }
}

#[async_trait]
impl TableProvider for IcebergTableProvider {
    fn schema(&self) -> ArrowSchemaRef {
        Arc::new(strip_nested_metadata_from_schema(&self.schema))
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let table = self.planning_table_or_load().await?;
        let snapshot_id = crate::physical_plan::snapshot_target::resolve_scan_snapshot_id(
            &table,
            self.commit_branch.as_deref(),
        )
        .map_err(to_datafusion_error)?;
        let knobs = crate::physical_plan::scan::scan_knobs_from_context(&state.task_ctx());
        Ok(Arc::new(
            IcebergTableScan::plan(
                table,
                snapshot_id,
                self.schema.clone(),
                projection,
                filters,
                limit,
                knobs,
            )
            .await?,
        ))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // One source of truth: the scanner drops the filters it cannot push down.
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // The write plans against the CURRENT schema: the state it commits against.
        let (table, current_schema) = self
            .load_table_with_current_schema()
            .await
            .map_err(to_datafusion_error)?;

        let partition_spec = table.metadata().default_partition_spec();

        let plan_with_partition = if !partition_spec.is_unpartitioned() {
            project_with_partition(input, &table)?
        } else {
            input
        };

        let target_partitions =
            NonZeroUsize::new(state.config().target_partitions()).ok_or_else(|| {
                DataFusionError::Configuration(
                    "target_partitions must be greater than 0".to_string(),
                )
            })?;

        let repartitioned_plan =
            repartition(plan_with_partition, table.metadata_ref(), target_partitions)?;

        let (write_input, sort_order_id) = sort_for_write(repartitioned_plan, &table)?;

        let write_plan = Arc::new(IcebergWriteExec::new(
            table.clone(),
            write_input,
            sort_order_id,
        ));

        // Merge the outputs of write_plan into one so we can commit all files together
        let coalesce_partitions = Arc::new(CoalescePartitionsExec::new(write_plan));

        Ok(Arc::new(
            IcebergCommitExec::new(
                table,
                self.catalog.clone(),
                coalesce_partitions,
                current_schema,
                insert_op,
            )
            .with_commit_branch(self.commit_branch.clone()),
        ))
    }

    async fn delete_from(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let (table, current_schema) = self
            .load_table_with_current_schema()
            .await
            .map_err(to_datafusion_error)?;
        let mode = WriteMode::from_property(&table, WRITE_DELETE_MODE);
        let isolation = IsolationLevel::for_row_level_op(&table, WRITE_DELETE_ISOLATION_LEVEL)?;

        // Exact PhysicalExpr is the row contract. Iceberg gets prune-only.
        let prune = convert_filters_to_predicate(&filters, table.metadata().current_schema());
        let predicate = match filters.into_iter().reduce(Expr::and) {
            None => None,
            Some(combined) => {
                let df_schema = DFSchema::try_from(current_schema.as_ref().clone())?;
                Some(state.create_physical_expr(combined, &df_schema)?)
            }
        };

        Ok(Arc::new(IcebergDeleteExec::new(
            table,
            self.catalog.clone(),
            predicate,
            prune,
            mode,
            isolation,
            current_schema,
            self.commit_branch.clone(),
        )))
    }

    async fn update(
        &self,
        state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let (table, current_schema) = self
            .load_table_with_current_schema()
            .await
            .map_err(to_datafusion_error)?;
        let mode = WriteMode::from_property(&table, WRITE_UPDATE_MODE);
        let isolation = IsolationLevel::for_row_level_op(&table, WRITE_UPDATE_ISOLATION_LEVEL)?;

        let df_schema = DFSchema::try_from(current_schema.as_ref().clone())?;

        let prune = convert_filters_to_predicate(&filters, table.metadata().current_schema());
        let predicate = match filters.into_iter().reduce(Expr::and) {
            None => None,
            Some(combined) => Some(state.create_physical_expr(combined, &df_schema)?),
        };

        let mut physical_assignments = Vec::with_capacity(assignments.len());
        for (column, expr) in assignments {
            let col_idx = current_schema.index_of(&column).map_err(|e| {
                DataFusionError::Plan(format!(
                    "UPDATE assignment to unknown column '{column}': {e}"
                ))
            })?;
            let value = state.create_physical_expr(expr, &df_schema)?;
            physical_assignments.push((col_idx, value));
        }

        Ok(Arc::new(IcebergUpdateExec::new(
            table,
            self.catalog.clone(),
            predicate,
            prune,
            physical_assignments,
            mode,
            isolation,
            current_schema,
            self.commit_branch.clone(),
        )))
    }
}
#[cfg(test)]
mod schema_evo_tests;
#[cfg(test)]
mod tests;
