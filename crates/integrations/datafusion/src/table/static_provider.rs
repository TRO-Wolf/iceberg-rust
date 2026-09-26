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

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::spec::Schema as IcebergSchema;
use iceberg::table::Table;
use iceberg::{Error, ErrorKind, Result};

use super::{uuid_text, uuid_text_filters};
use crate::error::to_datafusion_error;
use crate::physical_plan::conform::strip_nested_metadata_from_schema;
use crate::physical_plan::scan::IcebergTableScan;

/// Static table provider for read-only snapshot access. It holds a cached table instance and
/// refreshes no metadata. To write, use [`super::IcebergTableProvider`].
#[derive(Debug, Clone)]
pub struct IcebergStaticTableProvider {
    /// Never refreshed.
    table: Table,
    snapshot_id: Option<i64>,
    project_current_schema: bool,
    schema: ArrowSchemaRef,
    uuid_text_schema: ArrowSchemaRef,
    ice_schema: IcebergSchema,
    uuid_as_string: bool,
}

impl IcebergStaticTableProvider {
    fn from_parts(
        table: Table,
        snapshot_id: Option<i64>,
        project_current_schema: bool,
        ice_schema: &IcebergSchema,
    ) -> Result<Self> {
        let schema = Arc::new(schema_to_arrow_schema(ice_schema)?);
        let uuid_text_schema = Arc::new(uuid_text::arrow_schema_with_uuid_as_text(
            &schema,
            &uuid_text::collect_uuid_field_ids(ice_schema),
        ));
        Ok(IcebergStaticTableProvider {
            table,
            snapshot_id,
            project_current_schema,
            schema,
            uuid_text_schema,
            ice_schema: ice_schema.clone(),
            uuid_as_string: false,
        })
    }

    pub async fn try_new_from_table(table: Table) -> Result<Self> {
        let ice_schema = table.metadata().current_schema().clone();
        Self::from_parts(table, None, true, &ice_schema)
    }

    /// Creates a read-only provider over one snapshot, for a time-travel query.
    pub async fn try_new_from_table_snapshot(table: Table, snapshot_id: i64) -> Result<Self> {
        let snapshot = table
            .metadata()
            .snapshot_by_id(snapshot_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "snapshot id {snapshot_id} not found in table {}",
                        table.identifier().name()
                    ),
                )
            })?;
        let table_schema = snapshot.schema(table.metadata())?;
        Self::from_parts(table, Some(snapshot_id), false, &table_schema)
    }

    pub async fn try_new_from_table_ref(table: Table, ref_name: &str) -> Result<Self> {
        let (snapshot_id, is_branch) = table
            .snapshot_ref(ref_name)
            .map(|reference| (reference.snapshot_id, reference.is_branch()))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "reference {ref_name} not found in table {}",
                        table.identifier().name()
                    ),
                )
            })?;
        let table_schema = if is_branch {
            table.metadata().current_schema().clone()
        } else {
            table
                .metadata()
                .snapshot_by_id(snapshot_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "snapshot id {snapshot_id} for reference {ref_name} not found in table {}",
                            table.identifier().name()
                        ),
                    )
                })?
                .schema(table.metadata())?
        };
        Self::from_parts(table, Some(snapshot_id), is_branch, &table_schema)
    }

    pub fn with_uuid_as_string(mut self, enabled: bool) -> Self {
        self.uuid_as_string = enabled;
        self
    }
}

#[async_trait]
impl TableProvider for IcebergStaticTableProvider {
    fn schema(&self) -> ArrowSchemaRef {
        if self.uuid_as_string {
            Arc::new(strip_nested_metadata_from_schema(&self.uuid_text_schema))
        } else {
            Arc::new(strip_nested_metadata_from_schema(&self.schema))
        }
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
        let mut knobs = crate::physical_plan::scan::scan_knobs_from_context(&state.task_ctx());
        knobs.uuid_as_string = self.uuid_as_string;
        let rewritten;
        let filters: &[Expr] = if self.uuid_as_string {
            rewritten = uuid_text_filters::rewrite_uuid_text_filters(filters, &self.ice_schema);
            &rewritten
        } else {
            filters
        };
        Ok(Arc::new(
            IcebergTableScan::plan(
                self.table.clone(),
                self.snapshot_id,
                self.project_current_schema,
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
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        _input: Arc<dyn ExecutionPlan>,
        _insert_op: InsertOp,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Err(to_datafusion_error(Error::new(
            ErrorKind::FeatureUnsupported,
            "Write operations are not supported on IcebergStaticTableProvider. \
             Use IcebergTableProvider with a catalog for write support."
                .to_string(),
        )))
    }
}
