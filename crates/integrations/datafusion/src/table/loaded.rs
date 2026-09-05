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

use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::error::Result as DFResult;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::table::Table;
use iceberg::{Catalog, Result};

use super::IcebergTableProvider;
use crate::error::to_datafusion_error;

impl IcebergTableProvider {
    pub(crate) fn from_planning_load(catalog: Arc<dyn Catalog>, table: Table) -> Result<Self> {
        let schema = Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?);
        Ok(IcebergTableProvider {
            catalog,
            table_ident: table.identifier().clone(),
            schema,
            commit_branch: None,
            planning_table: Some(table),
        })
    }

    pub(crate) async fn planning_table_or_load(&self) -> DFResult<Table> {
        match self.planning_table.as_ref() {
            Some(table) => Ok(table.clone()),
            None => self
                .catalog
                .load_table(&self.table_ident)
                .await
                .map_err(to_datafusion_error),
        }
    }

    /// Loads the table's current state and Arrow schema. Write paths plan against this, not [`Self::schema`].
    pub(crate) async fn load_table_with_current_schema(&self) -> Result<(Table, ArrowSchemaRef)> {
        let table = self.catalog.load_table(&self.table_ident).await?;
        let schema: ArrowSchemaRef =
            Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?);
        Ok((table, schema))
    }
}
