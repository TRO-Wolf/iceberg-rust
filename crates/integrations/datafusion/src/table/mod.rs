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
use iceberg::spec::TableProperties;
use iceberg::table::Table;
use iceberg::{Catalog, Error, ErrorKind, NamespaceIdent, Result, TableIdent};
use metadata_table::IcebergMetadataTableProvider;
pub use static_provider::IcebergStaticTableProvider;

use crate::error::to_datafusion_error;
use crate::physical_plan::commit::IcebergCommitExec;
use crate::physical_plan::delete::{
    IcebergDeleteExec, IsolationLevel, WRITE_DELETE_ISOLATION_LEVEL, WRITE_DELETE_MODE,
    WRITE_UPDATE_ISOLATION_LEVEL, WRITE_UPDATE_MODE, WriteMode,
};
use crate::physical_plan::expr_to_predicate::convert_filters_to_predicate;
use crate::physical_plan::project::project_with_partition;
use crate::physical_plan::repartition::repartition;
use crate::physical_plan::scan::IcebergTableScan;
use crate::physical_plan::sort::sort_by_partition;
use crate::physical_plan::update::IcebergUpdateExec;
use crate::physical_plan::write::IcebergWriteExec;

/// Catalog-backed table provider. It loads fresh table metadata on every scan and write. For
/// read-only access to one snapshot, use [`IcebergStaticTableProvider`].
#[derive(Debug, Clone)]
pub struct IcebergTableProvider {
    catalog: Arc<dyn Catalog>,
    table_ident: TableIdent,
    /// FIXED for the life of the instance: DataFusion stores ordinals against it.
    schema: ArrowSchemaRef,
    commit_branch: Option<String>,
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
        })
    }

    /// Commit snapshot-producing DML onto `branch` instead of `main`. Java `SnapshotUpdate.toBranch`.
    pub fn with_commit_branch(mut self, branch: impl Into<String>) -> Self {
        self.commit_branch = Some(branch.into());
        self
    }

    /// Loads the table's current state, with its CURRENT schema in Arrow form. The write paths plan
    /// against this, not [`Self::schema`], because they re-scan and commit against what they load.
    async fn load_table_with_current_schema(&self) -> Result<(Table, ArrowSchemaRef)> {
        let table = self.catalog.load_table(&self.table_ident).await?;
        let schema: ArrowSchemaRef =
            Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?);
        Ok((table, schema))
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
        self.schema.clone()
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
        // The DATA is always current.
        let table = self
            .catalog
            .load_table(&self.table_ident)
            .await
            .map_err(to_datafusion_error)?;

        // `self.schema` IS the schema DataFusion planned against, and it cannot have moved since.
        // `IcebergTableScan` binds it to the reloaded table by FIELD ID.
        let knobs = crate::physical_plan::scan::scan_knobs_from_context(&state.task_ctx());
        Ok(Arc::new(
            IcebergTableScan::plan(
                table,
                None, // Always use current snapshot for catalog-backed provider
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

        let fanout_enabled = table
            .metadata()
            .properties()
            .get(TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED)
            .map(|value| {
                value
                    .parse::<bool>()
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Invalid value for {}, expected 'true' or 'false'",
                                TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED
                            ),
                        )
                        .with_source(e)
                    })
                    .map_err(to_datafusion_error)
            })
            .transpose()?
            .unwrap_or(TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED_DEFAULT);

        let write_input = if fanout_enabled {
            repartitioned_plan
        } else {
            sort_by_partition(repartitioned_plan)?
        };

        let write_plan = Arc::new(IcebergWriteExec::new(table.clone(), write_input));

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
        let prune = convert_filters_to_predicate(&filters);
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

        let prune = convert_filters_to_predicate(&filters);
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
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::common::Column;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::SessionContext;
    use iceberg::io::FileIO;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use iceberg::table::{StaticTable, Table};
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
    use tempfile::TempDir;

    use super::*;

    async fn get_test_table_from_metadata_file() -> Table {
        let metadata_file_name = "TableMetadataV2Valid.json";
        let metadata_file_path = format!(
            "{}/tests/test_data/{}",
            env!("CARGO_MANIFEST_DIR"),
            metadata_file_name
        );
        let file_io = FileIO::new_with_fs();
        let static_identifier = TableIdent::from_strs(["static_ns", "static_table"]).unwrap();
        let static_table =
            StaticTable::from_metadata_file(&metadata_file_path, static_identifier, file_io)
                .await
                .unwrap();
        static_table.into_table()
    }

    async fn get_test_catalog_and_table() -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let warehouse_path = temp_dir.path().to_str().unwrap().to_string();

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
            )
            .await
            .unwrap();

        let namespace = NamespaceIdent::new("test_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let table_creation = TableCreation::builder()
            .name("test_table".to_string())
            .location(format!("{warehouse_path}/test_table"))
            .schema(schema)
            .properties(HashMap::new())
            .build();

        catalog
            .create_table(&namespace, table_creation)
            .await
            .unwrap();

        (
            Arc::new(catalog),
            namespace,
            "test_table".to_string(),
            temp_dir,
        )
    }

    // Tests for IcebergStaticTableProvider

    #[tokio::test]
    async fn test_static_provider_from_table() {
        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();
        let df = ctx.sql("SELECT * FROM mytable").await.unwrap();
        let df_schema = df.schema();
        let df_columns = df_schema.fields();
        assert_eq!(df_columns.len(), 3);
        let x_column = df_columns.first().unwrap();
        let column_data = format!(
            "{:?}:{:?}",
            x_column.name(),
            x_column.data_type().to_string()
        );
        assert_eq!(column_data, "\"x\":\"Int64\"");
        let has_column = df_schema.has_column(&Column::from_name("z"));
        assert!(has_column);
    }

    #[tokio::test]
    async fn test_static_provider_from_snapshot() {
        let table = get_test_table_from_metadata_file().await;
        let snapshot_id = table.metadata().snapshots().next().unwrap().snapshot_id();
        let table_provider =
            IcebergStaticTableProvider::try_new_from_table_snapshot(table.clone(), snapshot_id)
                .await
                .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();
        let df = ctx.sql("SELECT * FROM mytable").await.unwrap();
        let df_schema = df.schema();
        let df_columns = df_schema.fields();
        assert_eq!(df_columns.len(), 3);
        let x_column = df_columns.first().unwrap();
        let column_data = format!(
            "{:?}:{:?}",
            x_column.name(),
            x_column.data_type().to_string()
        );
        assert_eq!(column_data, "\"x\":\"Int64\"");
        let has_column = df_schema.has_column(&Column::from_name("z"));
        assert!(has_column);
    }

    #[tokio::test]
    async fn test_static_provider_rejects_writes() {
        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();

        let result = ctx.sql("INSERT INTO mytable VALUES (1, 2, 3)").await;

        assert!(
            result.is_err() || {
                let df = result.unwrap();
                df.collect().await.is_err()
            }
        );
    }

    #[tokio::test]
    async fn test_static_provider_scan() {
        // A real empty table: an incomplete fixture must fail closed at plan time, not demote.
        let (_catalog, _ns, _name, table, _tmp) = get_static_test_table().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table)
            .await
            .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();

        let df = ctx.sql("SELECT count(*) FROM mytable").await.unwrap();
        let physical_plan = df.create_physical_plan().await;
        assert!(physical_plan.is_ok());
    }

    // Tests for IcebergTableProvider

    #[tokio::test]
    async fn test_catalog_backed_provider_creation() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let schema = provider.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "name");
    }

    #[tokio::test]
    async fn test_catalog_backed_provider_scan() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        ctx.register_table("test_table", Arc::new(provider))
            .unwrap();

        let df = ctx.sql("SELECT * FROM test_table").await.unwrap();

        let df_schema = df.schema();
        assert_eq!(df_schema.fields().len(), 2);
        assert_eq!(df_schema.field(0).name(), "id");
        assert_eq!(df_schema.field(1).name(), "name");

        let physical_plan = df.create_physical_plan().await;
        assert!(physical_plan.is_ok());
    }

    #[tokio::test]
    async fn test_catalog_backed_provider_insert() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        ctx.register_table("test_table", Arc::new(provider))
            .unwrap();

        let result = ctx.sql("INSERT INTO test_table VALUES (1, 'test')").await;

        assert!(result.is_ok());

        let df = result.unwrap();
        let execution_result = df.collect().await;

        assert!(execution_result.is_ok());
    }

    /// Pin 13 DF: multi_partition_scan=false forces T=1 (N=1) while target_partitions > 1.
    #[tokio::test]
    async fn test_pin13_off_switch_forces_n1_with_target_partitions_gt1() {
        use datafusion::prelude::SessionConfig;

        use crate::physical_plan::scan::{IcebergScanOptions, IcebergTableScan};

        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("provider");

        let mut config = SessionConfig::new().with_target_partitions(8);
        config.options_mut().extensions.insert(IcebergScanOptions {
            multi_partition_scan: false,
            data_file_concurrency: 8,
        });
        let ctx = SessionContext::new_with_config(config);
        ctx.register_table("test_table", Arc::new(provider))
            .expect("register");

        // Multiple files so multi-partition would otherwise engage when ON.
        for sql in [
            "INSERT INTO test_table VALUES (1, 'a')",
            "INSERT INTO test_table VALUES (2, 'b')",
            "INSERT INTO test_table VALUES (3, 'c')",
        ] {
            ctx.sql(sql)
                .await
                .expect("insert plan")
                .collect()
                .await
                .expect("insert");
        }

        let plan = ctx
            .sql("SELECT id FROM test_table")
            .await
            .expect("select")
            .create_physical_plan()
            .await
            .expect("physical");
        fn find_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<&IcebergTableScan> {
            if let Some(s) = plan.downcast_ref::<IcebergTableScan>() {
                return Some(s);
            }
            for c in plan.children() {
                if let Some(s) = find_scan(c) {
                    return Some(s);
                }
            }
            None
        }
        let scan = find_scan(&plan).expect("IcebergTableScan present");
        assert_eq!(
            scan.partition_work().len(),
            1,
            "pin 13: off-switch must force N=1 even with multi-file + target_partitions=8"
        );
        assert_eq!(scan.properties().output_partitioning().partition_count(), 1);
        // Multiset still complete (pin 4 under off-switch)
        let rows: usize = ctx
            .sql("SELECT id FROM test_table")
            .await
            .expect("sel")
            .collect()
            .await
            .expect("collect")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows, 3, "pin 13/4: off-switch must not drop rows");
    }

    /// Pins 1 + 5 (DF): multi-file + tiny split props force N>1; LIMIT k card + sub-multiset.
    #[tokio::test]
    async fn test_pin1_pin5_multi_file_partitioning_and_limit() {
        use datafusion::prelude::SessionConfig;

        use crate::physical_plan::scan::{IcebergScanOptions, IcebergTableScan};

        let temp_dir = TempDir::new().unwrap();
        let warehouse_path = temp_dir.path().to_str().unwrap().to_string();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
            )
            .await
            .unwrap();
        let namespace = NamespaceIdent::new("pin15_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();
        let table_creation = TableCreation::builder()
            .name("pin15".to_string())
            .location(format!("{warehouse_path}/pin15"))
            .schema(schema)
            .properties(HashMap::from([
                ("read.split.target-size".to_string(), "1".to_string()),
                ("read.split.open-file-cost".to_string(), "1".to_string()),
                (
                    "read.split.planning-lookback".to_string(),
                    "100".to_string(),
                ),
            ]))
            .build();
        catalog
            .create_table(&namespace, table_creation)
            .await
            .unwrap();
        let catalog = Arc::new(catalog);
        let provider = IcebergTableProvider::try_new(catalog, namespace, "pin15".to_string())
            .await
            .expect("provider");

        let mut config = SessionConfig::new().with_target_partitions(4);
        config.options_mut().extensions.insert(IcebergScanOptions {
            multi_partition_scan: true,
            data_file_concurrency: 4,
        });
        let ctx = SessionContext::new_with_config(config);
        ctx.register_table("test_table", Arc::new(provider))
            .expect("register");

        for sql in [
            "INSERT INTO test_table VALUES (1, 'a'), (2, 'b')",
            "INSERT INTO test_table VALUES (3, 'c'), (4, 'd')",
            "INSERT INTO test_table VALUES (5, 'e')",
        ] {
            ctx.sql(sql)
                .await
                .expect("insert plan")
                .collect()
                .await
                .expect("insert");
        }

        let unlimited = ctx
            .sql("SELECT id FROM test_table")
            .await
            .expect("select")
            .collect()
            .await
            .expect("collect unlimited");
        let unlimited_rows: usize = unlimited.iter().map(|b| b.num_rows()).sum();
        assert_eq!(unlimited_rows, 5, "seeded 5 rows");

        let df = ctx
            .sql("SELECT id FROM test_table LIMIT 2")
            .await
            .expect("limit sql");
        let plan = df.create_physical_plan().await.expect("physical plan");
        fn find_iceberg_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<&IcebergTableScan> {
            if let Some(s) = plan.downcast_ref::<IcebergTableScan>() {
                return Some(s);
            }
            for c in plan.children() {
                if let Some(s) = find_iceberg_scan(c) {
                    return Some(s);
                }
            }
            None
        }
        let scan = find_iceberg_scan(&plan).expect("IcebergTableScan in plan");
        let n = scan.partition_work().len();
        assert!(
            n > 1,
            "pin 1: multi-file + tiny split props must yield N>1, got N={n}"
        );
        assert_eq!(scan.limit(), None, "pin 5: provider limit demoted when N>1");
        assert!(
            scan.properties().output_partitioning().partition_count() > 1,
            "pin 1: UnknownPartitioning(N>1)"
        );

        let limited = ctx
            .sql("SELECT id FROM test_table LIMIT 2")
            .await
            .expect("limit2")
            .collect()
            .await
            .expect("collect limit");
        let limited_rows: usize = limited.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            limited_rows, 2,
            "pin 5: LIMIT 2 must return exactly min(2, 5)=2 rows, got {limited_rows}"
        );

        let mut unlimited_ids = std::collections::HashSet::new();
        for b in &unlimited {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .expect("id int");
            for i in 0..col.len() {
                unlimited_ids.insert(col.value(i));
            }
        }
        for b in &limited {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .expect("id int");
            for i in 0..col.len() {
                assert!(
                    unlimited_ids.contains(&col.value(i)),
                    "pin 5: limited row must be sub-multiset of unlimited"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_physical_input_schema_consistent_with_logical_input_schema() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        ctx.register_table("test_table", Arc::new(provider))
            .unwrap();

        let df = ctx.sql("SELECT id, name FROM test_table").await.unwrap();

        let logical_schema = df.schema().clone();

        let physical_plan = df.create_physical_plan().await.unwrap();
        let physical_schema = physical_plan.schema();

        assert_eq!(
            logical_schema.fields().len(),
            physical_schema.fields().len()
        );

        for (logical_field, physical_field) in logical_schema
            .fields()
            .iter()
            .zip(physical_schema.fields().iter())
        {
            assert_eq!(logical_field.name(), physical_field.name());
            assert_eq!(logical_field.data_type(), physical_field.data_type());
        }
    }

    async fn get_partitioned_test_catalog_and_table(
        fanout_enabled: Option<bool>,
    ) -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
        use iceberg::spec::{Transform, UnboundPartitionSpec};

        let temp_dir = TempDir::new().unwrap();
        let warehouse_path = temp_dir.path().to_str().unwrap().to_string();

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
            )
            .await
            .unwrap();

        let namespace = NamespaceIdent::new("test_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "category", Transform::Identity)
            .unwrap()
            .build();

        let mut properties = HashMap::new();
        if let Some(enabled) = fanout_enabled {
            properties.insert(
                iceberg::spec::TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED
                    .to_string(),
                enabled.to_string(),
            );
        }

        let table_creation = TableCreation::builder()
            .name("partitioned_table".to_string())
            .location(format!("{warehouse_path}/partitioned_table"))
            .schema(schema)
            .partition_spec(partition_spec)
            .properties(properties)
            .build();

        catalog
            .create_table(&namespace, table_creation)
            .await
            .unwrap();

        (
            Arc::new(catalog),
            namespace,
            "partitioned_table".to_string(),
            temp_dir,
        )
    }

    fn plan_contains_sort(plan: &Arc<dyn ExecutionPlan>) -> bool {
        if plan.name() == "SortExec" {
            return true;
        }
        for child in plan.children() {
            if plan_contains_sort(child) {
                return true;
            }
        }
        false
    }

    #[tokio::test]
    async fn test_insert_plan_fanout_enabled_no_sort() {
        use datafusion::datasource::TableProvider;
        use datafusion::logical_expr::dml::InsertOp;
        use datafusion::physical_plan::empty::EmptyExec;

        let (catalog, namespace, table_name, _temp_dir) =
            get_partitioned_test_catalog_and_table(Some(true)).await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        let input_schema = provider.schema();
        let input = Arc::new(EmptyExec::new(input_schema)) as Arc<dyn ExecutionPlan>;

        let state = ctx.state();
        let insert_plan = provider
            .insert_into(&state, input, InsertOp::Append)
            .await
            .unwrap();

        assert!(
            !plan_contains_sort(&insert_plan),
            "Plan should NOT contain SortExec when fanout is enabled"
        );
    }

    #[tokio::test]
    async fn test_insert_plan_fanout_disabled_has_sort() {
        use datafusion::datasource::TableProvider;
        use datafusion::logical_expr::dml::InsertOp;
        use datafusion::physical_plan::empty::EmptyExec;

        let (catalog, namespace, table_name, _temp_dir) =
            get_partitioned_test_catalog_and_table(Some(false)).await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        let input_schema = provider.schema();
        let input = Arc::new(EmptyExec::new(input_schema)) as Arc<dyn ExecutionPlan>;

        let state = ctx.state();
        let insert_plan = provider
            .insert_into(&state, input, InsertOp::Append)
            .await
            .unwrap();

        assert!(
            plan_contains_sort(&insert_plan),
            "Plan should contain SortExec when fanout is disabled"
        );
    }

    /// Empty table with a local warehouse path — safe for eager `plan_tasks` (G1 fail-closed).
    async fn get_static_test_table() -> (Arc<dyn Catalog>, NamespaceIdent, String, Table, TempDir) {
        let (catalog, namespace, table_name, temp_dir) = get_test_catalog_and_table().await;
        let table = catalog
            .load_table(&TableIdent::new(namespace.clone(), table_name.clone()))
            .await
            .expect("load empty test table");
        (catalog, namespace, table_name, table, temp_dir)
    }

    #[tokio::test]
    async fn test_limit_pushdown_static_provider() {
        use datafusion::datasource::TableProvider;

        let (_catalog, _ns, _name, table, _tmp) = get_static_test_table().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table)
            .await
            .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        let scan_plan = table_provider
            .scan(&state, None, &[], Some(10))
            .await
            .unwrap();

        let iceberg_scan = scan_plan
            .downcast_ref::<IcebergTableScan>()
            .expect("Expected IcebergTableScan");

        assert_eq!(
            iceberg_scan.limit(),
            Some(10),
            "Limit should be set to 10 in the scan plan"
        );
    }

    #[tokio::test]
    async fn test_limit_pushdown_catalog_backed_provider() {
        use datafusion::datasource::TableProvider;

        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        let scan_plan = provider.scan(&state, None, &[], Some(5)).await.unwrap();

        let iceberg_scan = scan_plan
            .downcast_ref::<IcebergTableScan>()
            .expect("Expected IcebergTableScan");

        assert_eq!(
            iceberg_scan.limit(),
            Some(5),
            "Limit should be set to 5 in the scan plan"
        );
    }

    // ===== Live-schema regressions =====
    //
    // Two halves of one defect class. A provider that caches the Arrow schema forever plans every
    // later query against a schema that no longer describes the table. And a scan that reloads the
    // table while the adapter advertises the construction-time schema emits mismatched batches.

    /// An out-of-band schema evolution. None of these creates a snapshot, so the CURRENT schema
    /// and the schema the data is read with disagree.
    enum SchemaOp<'a> {
        /// `ALTER TABLE ADD COLUMN <name> int` (optional).
        AddOptionalInt(&'a str),
        /// `ALTER TABLE RENAME COLUMN <from> TO <to>`, which keeps the field id.
        Rename(&'a str, &'a str),
        /// `ALTER TABLE ALTER COLUMN <name> TYPE bigint`, a legal int to long promotion.
        PromoteToLong(&'a str),
        /// `ALTER TABLE DROP COLUMN <name>`.
        Drop(&'a str),
    }

    /// Applies an evolution through a SECOND catalog handle. The provider under test never sees it.
    async fn evolve_schema(catalog: &Arc<dyn Catalog>, ident: &TableIdent, op: SchemaOp<'_>) {
        use iceberg::transaction::{ApplyTransactionAction, Transaction};

        let table = catalog
            .load_table(ident)
            .await
            .expect("load table for out-of-band evolution");
        let tx = Transaction::new(&table);
        let action = tx.update_schema();
        let action = match op {
            SchemaOp::AddOptionalInt(name) => {
                action.add_column(name, Type::Primitive(PrimitiveType::Int))
            }
            SchemaOp::Rename(from, to) => action.rename_column(from, to),
            SchemaOp::PromoteToLong(name) => action.update_column(name, PrimitiveType::Long),
            SchemaOp::Drop(name) => action.delete_column(name),
        };
        let tx = action.apply(tx).expect("queue the schema update");
        tx.commit(catalog.as_ref())
            .await
            .expect("commit the out-of-band schema evolution");
    }

    async fn query_through(
        provider: Arc<dyn TableProvider>,
        sql: &str,
    ) -> Vec<datafusion::arrow::array::RecordBatch> {
        let ctx = SessionContext::new();
        ctx.register_table("t", provider)
            .expect("register the provider under test");
        ctx.sql(sql)
            .await
            .unwrap_or_else(|e| panic!("plan `{sql}`: {e}"))
            .collect()
            .await
            .unwrap_or_else(|e| panic!("execute `{sql}`: {e}"))
    }

    /// Seeds `rows` through a provider resolved fresh against the table's current schema.
    async fn seed(
        catalog: &Arc<dyn Catalog>,
        namespace: &NamespaceIdent,
        table_name: &str,
        sql: &str,
    ) {
        let provider = IcebergTableProvider::try_new(
            catalog.clone(),
            namespace.clone(),
            table_name.to_string(),
        )
        .await
        .expect("construct a provider for the seed write");
        let batches: Vec<_> = query_through(Arc::new(provider), sql).await;
        assert!(!batches.is_empty(), "a write must report its row count");
    }

    /// An advertised schema is STABLE, and freshness comes from `refreshed()`.
    #[tokio::test]
    async fn test_provider_schema_is_stable_and_refreshed_serves_the_current_schema() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct the catalog-backed provider");
        assert_eq!(provider.schema().fields().len(), 2);

        evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("extra")).await;

        // An ordinary operation must NOT move the advertised schema.
        let ctx = SessionContext::new();
        let state = ctx.state();
        provider
            .scan(&state, None, &[], None)
            .await
            .expect("scan against the evolved table");
        assert_eq!(
            provider.schema().fields().len(),
            2,
            "an instance's advertised schema must not move under the plans built on it"
        );

        // A NEW instance carries the current schema.
        let refreshed = provider
            .refreshed()
            .await
            .expect("refresh into a new provider");
        assert_eq!(
            refreshed.schema().fields().len(),
            3,
            "refreshed() must serve the CURRENT schema, got {:?}",
            refreshed.schema()
        );
        assert_eq!(refreshed.schema().field(2).name(), "extra");
        assert_eq!(
            provider.schema().fields().len(),
            2,
            "refreshed() must leave the original instance alone"
        );
    }

    /// A catalog query resolves a provider per planning round, as `SparkCatalog.loadTable` does,
    /// so the next query sees an evolution with no refresh call.
    #[tokio::test]
    async fn test_catalog_resolves_a_fresh_provider_per_query() {
        use datafusion::catalog::SchemaProvider;

        use crate::schema::IcebergSchemaProvider;

        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        let schema_provider = IcebergSchemaProvider::try_new(catalog.clone(), namespace.clone())
            .await
            .expect("construct the namespace schema provider");

        let before = schema_provider
            .table(&table_name)
            .await
            .expect("resolve the table")
            .expect("the table is listed");
        assert_eq!(before.schema().fields().len(), 2);

        evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("extra")).await;

        let after = schema_provider
            .table(&table_name)
            .await
            .expect("re-resolve the table")
            .expect("the table is still listed");
        assert_eq!(
            after.schema().fields().len(),
            3,
            "the next resolution must carry the evolved schema, got {:?}",
            after.schema()
        );
        assert_eq!(
            before.schema().fields().len(),
            2,
            "the previously resolved provider must be untouched — plans hold ordinals into it"
        );
    }

    /// `ADD COLUMN` creates no snapshot, so the advertised schema has a column the scanned one
    /// lacks. The batches must still match it, with that column read as NULL, as Java null-fills.
    #[tokio::test]
    async fn test_scan_batches_match_advertised_schema_after_add_column() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        // One committed row, so the snapshot schema is the 2-column original.
        {
            let provider = IcebergTableProvider::try_new(
                catalog.clone(),
                namespace.clone(),
                table_name.clone(),
            )
            .await
            .expect("construct provider for the seed insert");
            let ctx = SessionContext::new();
            ctx.register_table("t", Arc::new(provider))
                .expect("register table for the seed insert");
            ctx.sql("INSERT INTO t VALUES (1, 'a')")
                .await
                .expect("plan the seed insert")
                .collect()
                .await
                .expect("execute the seed insert");
        }

        evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("extra")).await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a provider on the evolved table");
        assert_eq!(
            provider.schema().fields().len(),
            3,
            "the provider advertises the CURRENT (post-ADD COLUMN) schema"
        );

        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register the evolved table");
        let batches = ctx
            .sql("SELECT * FROM t")
            .await
            .expect("plan SELECT * on the evolved table")
            .collect()
            .await
            .expect("execute SELECT * on the evolved table");

        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 1, "the seeded row must still be readable");
        for batch in &batches {
            assert_eq!(
                batch.num_columns(),
                3,
                "the emitted batch must carry the advertised column set, got {:?}",
                batch.schema()
            );
            let extra = batch
                .column_by_name("extra")
                .expect("the added column must be present in the emitted batch");
            assert_eq!(
                extra.null_count(),
                batch.num_rows(),
                "a column added after the scanned snapshot must read as NULL"
            );
        }
    }

    /// A provider built before an evolution advertises the OLD schema, while the scan reloads the
    /// table and would otherwise `select_all()` the NEW column set.
    #[tokio::test]
    async fn test_stale_provider_scan_is_self_consistent() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        let stale_provider = Arc::new(
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct the provider BEFORE the evolution"),
        );

        // The current snapshot carries 3 columns while `stale_provider` advertises 2.
        evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("extra")).await;
        {
            let fresh = IcebergTableProvider::try_new(
                catalog.clone(),
                namespace.clone(),
                table_name.clone(),
            )
            .await
            .expect("construct a fresh provider for the out-of-band write");
            let ctx = SessionContext::new();
            ctx.register_table("t", Arc::new(fresh))
                .expect("register for the out-of-band write");
            ctx.sql("INSERT INTO t VALUES (1, 'a', 7)")
                .await
                .expect("plan the out-of-band insert")
                .collect()
                .await
                .expect("execute the out-of-band insert");
        }

        let ctx = SessionContext::new();
        ctx.register_table("t", stale_provider.clone() as Arc<dyn TableProvider>)
            .expect("register the stale provider");

        let batches = ctx
            .sql("SELECT * FROM t")
            .await
            .expect("plan SELECT * through the stale provider")
            .collect()
            .await
            .expect("execute SELECT * through the stale provider");
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 1, "the row written out of band must be visible");
        for batch in &batches {
            assert_eq!(
                batch.num_columns(),
                2,
                "the batch must match the 2-column schema the plan advertised, got {:?}",
                batch.schema()
            );
            assert!(
                batch.column_by_name("extra").is_none(),
                "a column the plan never advertised must not appear in the batch"
            );
        }

        // The advertised schema does not move, so a second query through the SAME provider is
        // planned and answered identically.
        assert_eq!(
            stale_provider.schema().fields().len(),
            2,
            "the advertised schema must not move under the plans built on it"
        );
        let batches = ctx
            .sql("SELECT * FROM t")
            .await
            .expect("plan the follow-up SELECT *")
            .collect()
            .await
            .expect("execute the follow-up SELECT *");
        for batch in &batches {
            assert_eq!(batch.num_columns(), 2, "got {:?}", batch.schema());
        }
    }

    /// The silent path: with NO projection the scan once asked the reloaded table for
    /// `select_all()`. DataFusion addresses batch columns by ordinal, so an extra one corrupts.
    #[tokio::test]
    async fn test_unprojected_scan_advertises_the_schema_it_emits() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        let stale_provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct the provider BEFORE the evolution");

        evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("extra")).await;
        {
            let fresh = IcebergTableProvider::try_new(
                catalog.clone(),
                namespace.clone(),
                table_name.clone(),
            )
            .await
            .expect("construct a fresh provider for the out-of-band write");
            let ctx = SessionContext::new();
            ctx.register_table("t", Arc::new(fresh))
                .expect("register for the out-of-band write");
            ctx.sql("INSERT INTO t VALUES (1, 'a', 7)")
                .await
                .expect("plan the out-of-band insert")
                .collect()
                .await
                .expect("execute the out-of-band insert");
        }

        let ctx = SessionContext::new();
        let state = ctx.state();
        // `projection: None` is the path that once became `select_all()`.
        let plan = stale_provider
            .scan(&state, None, &[], None)
            .await
            .expect("plan an unprojected scan through the stale provider");
        let advertised = plan.schema();
        assert_eq!(
            advertised.fields().len(),
            2,
            "the plan advertises the schema it was planned against"
        );

        let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .expect("execute the unprojected scan");
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 1, "the out-of-band row must be visible");
        for batch in &batches {
            assert_eq!(
                batch.schema(),
                advertised,
                "an emitted batch must match the schema the plan advertised"
            );
        }
    }

    /// Builds a table whose snapshot carries an out-of-band `extra` column, plus a provider
    /// constructed BEFORE that evolution.
    async fn stale_provider_over_evolved_table(
        catalog: &Arc<dyn Catalog>,
        namespace: &NamespaceIdent,
        table_name: &str,
    ) -> IcebergTableProvider {
        let ident = TableIdent::new(namespace.clone(), table_name.to_string());
        let stale_provider = IcebergTableProvider::try_new(
            catalog.clone(),
            namespace.clone(),
            table_name.to_string(),
        )
        .await
        .expect("construct the provider BEFORE the evolution");

        evolve_schema(catalog, &ident, SchemaOp::AddOptionalInt("extra")).await;

        let fresh = IcebergTableProvider::try_new(
            catalog.clone(),
            namespace.clone(),
            table_name.to_string(),
        )
        .await
        .expect("construct a fresh provider for the out-of-band write");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(fresh))
            .expect("register for the out-of-band write");
        ctx.sql("INSERT INTO t VALUES (1, 'a', 7), (2, 'b', 8)")
            .await
            .expect("plan the out-of-band insert")
            .collect()
            .await
            .expect("execute the out-of-band insert");

        stale_provider
    }

    fn dml_row_count(batches: &[datafusion::arrow::array::RecordBatch]) -> u64 {
        batches
            .iter()
            .map(|batch| {
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::UInt64Array>()
                    .expect("the DML result column must be UInt64");
                (0..array.len()).map(|i| array.value(i)).sum::<u64>()
            })
            .sum()
    }

    /// `delete_from` binds its row filter and projection base to the CURRENT schema. A filter over
    /// a column added out of band cannot bind against the cached one.
    #[tokio::test]
    async fn test_delete_binds_to_current_schema_not_the_cached_one() {
        use datafusion::prelude::{col, lit};

        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let stale_provider =
            stale_provider_over_evolved_table(&catalog, &namespace, &table_name).await;

        let ctx = SessionContext::new();
        let state = ctx.state();
        let plan = stale_provider
            .delete_from(&state, vec![col("extra").eq(lit(7))])
            .await
            .expect("plan a DELETE filtered on the out-of-band column");
        let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .expect("execute the DELETE");
        assert_eq!(
            dml_row_count(&batches),
            1,
            "exactly the row matching `extra = 7` must be deleted"
        );
    }

    /// `update` resolves each `SET` target against the CURRENT schema, so the column index and the
    /// projection base describe one state. The cached schema calls a new column unknown.
    #[tokio::test]
    async fn test_update_binds_to_current_schema_not_the_cached_one() {
        use datafusion::prelude::{col, lit};

        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let stale_provider =
            stale_provider_over_evolved_table(&catalog, &namespace, &table_name).await;

        let ctx = SessionContext::new();
        let state = ctx.state();
        let plan = stale_provider
            .update(&state, vec![("extra".to_string(), lit(99))], vec![
                col("id").eq(lit(1)),
            ])
            .await
            .expect("plan an UPDATE assigning to the out-of-band column");
        let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .expect("execute the UPDATE");
        assert_eq!(
            dml_row_count(&batches),
            1,
            "exactly the row matching `id = 1` must be updated"
        );
    }

    /// `RENAME COLUMN` keeps the field id and creates NO snapshot, so the advertised name and the
    /// snapshot's name differ. Name binding null-fills over live data; field-id binding does not.
    #[tokio::test]
    async fn test_rename_preserves_values_under_the_new_name() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("opt")).await;
        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, 'a', 7)",
        )
        .await;

        // The rename lands AFTER the write, so the data sits under the old name.
        evolve_schema(&catalog, &ident, SchemaOp::Rename("opt", "opt2")).await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a provider on the renamed table");
        let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

        let mut seen = 0;
        for batch in &batches {
            let renamed = batch
                .column_by_name("opt2")
                .expect("the renamed column must be present under its NEW name");
            assert_eq!(
                renamed.null_count(),
                0,
                "the renamed column must carry its data, not NULLs: {:?}",
                batch.schema()
            );
            let values = renamed
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .expect("opt2 must be Int32");
            for i in 0..values.len() {
                assert_eq!(
                    values.value(i),
                    7,
                    "the stored value must survive the rename"
                );
                seen += 1;
            }
        }
        assert_eq!(seen, 1, "the seeded row must be readable");
    }

    /// A renamed REQUIRED column has no null-fill escape, so name binding fails the query outright.
    #[tokio::test]
    async fn test_required_column_rename_preserves_values() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, 'a')",
        )
        .await;
        // `name` is REQUIRED in the fixture schema.
        evolve_schema(&catalog, &ident, SchemaOp::Rename("name", "full_name")).await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a provider on the renamed table");
        let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

        let mut seen = 0;
        for batch in &batches {
            use datafusion::arrow::array::Array;
            let renamed = batch
                .column_by_name("full_name")
                .expect("the renamed required column must be present")
                .as_any()
                .downcast_ref::<datafusion::arrow::array::StringArray>()
                .expect("full_name must be Utf8");
            for i in 0..renamed.len() {
                assert_eq!(renamed.value(i), "a");
                seen += 1;
            }
        }
        assert_eq!(seen, 1, "the seeded row must be readable");
    }

    /// A VIEW captures the provider and its projection ORDINALS. A schema that shrank under it
    /// makes `TableScan::try_new` index it with a stale ordinal and PANIC.
    #[tokio::test]
    async fn test_view_survives_an_out_of_band_column_drop() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, 'a')",
        )
        .await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct the provider the view will capture");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register the table");
        ctx.sql("CREATE VIEW v AS SELECT name FROM t")
            .await
            .expect("plan CREATE VIEW")
            .collect()
            .await
            .expect("create the view");

        // The drop lands BEFORE round 1, so a provider that republished its schema there leaves
        // round 2 indexing a SHORTER schema with round-1 ordinals, which panics.
        evolve_schema(&catalog, &ident, SchemaOp::Drop("id")).await;

        let round1 = ctx
            .sql("SELECT * FROM v")
            .await
            .expect("plan round 1")
            .collect()
            .await
            .expect("execute round 1");
        assert_eq!(round1.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

        let round2 = ctx
            .sql("SELECT * FROM v")
            .await
            .expect("plan round 2 (must not panic)")
            .collect()
            .await
            .expect("execute round 2 (must not panic)");
        assert_eq!(
            round2.iter().map(|b| b.num_rows()).sum::<usize>(),
            1,
            "the view must keep answering against the schema it was created with"
        );
        for batch in &round2 {
            assert_eq!(batch.num_columns(), 1);
            assert!(batch.column_by_name("name").is_some());
        }
    }

    /// A DataFrame collected after an evolution holds planning-time ordinals, and must not panic.
    #[tokio::test]
    async fn test_deferred_dataframe_survives_an_out_of_band_evolution() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, 'a')",
        )
        .await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct the provider");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register the table");

        let df = ctx.sql("SELECT name FROM t").await.expect("plan the query");

        evolve_schema(&catalog, &ident, SchemaOp::Drop("id")).await;

        // Execute twice: the second physical-planning round re-indexes the provider schema with
        // the plan's stored ordinals.
        let first = df
            .clone()
            .collect()
            .await
            .expect("the deferred plan must execute, not panic");
        assert_eq!(first.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        let second = df
            .collect()
            .await
            .expect("re-executing the deferred plan must not panic either");
        assert_eq!(second.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        for batch in &second {
            assert_eq!(batch.num_columns(), 1);
        }
    }

    /// `int` to `long` is a legal promotion and creates no snapshot, so the scanned data is still
    /// `int` while the plan advertises `long`. The values must be read, widened.
    #[tokio::test]
    async fn test_legal_int_to_long_promotion_reads_widened_values() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (42, 'a')",
        )
        .await;
        evolve_schema(&catalog, &ident, SchemaOp::PromoteToLong("id")).await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a provider on the promoted table");
        assert_eq!(
            provider.schema().field(0).data_type(),
            &datafusion::arrow::datatypes::DataType::Int64,
            "the provider advertises the promoted type"
        );
        let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

        let mut seen = 0;
        for batch in &batches {
            let ids = batch
                .column_by_name("id")
                .expect("id must be present")
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("id must be read as the PROMOTED Int64");
            for i in 0..ids.len() {
                assert_eq!(
                    ids.value(i),
                    42,
                    "the stored value must survive the promotion"
                );
                seen += 1;
            }
        }
        assert_eq!(seen, 1, "the seeded row must be readable");
    }

    /// In the steady state the scanned batch's schema is IDENTICAL to the advertised one, metadata
    /// included, so `conform_batch` rebuilds nothing. A reader change would move every scan onto
    /// the rebuild path.
    #[tokio::test]
    async fn test_steady_state_batch_schema_is_identical_to_the_advertised_schema() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, 'a')",
        )
        .await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct the provider");
        let ctx = SessionContext::new();
        let state = ctx.state();
        let plan = provider
            .scan(&state, None, &[], None)
            .await
            .expect("plan the scan");
        let advertised = plan.schema();
        let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .expect("execute the scan");
        assert!(!batches.is_empty(), "the seeded row must produce a batch");
        for batch in &batches {
            assert_eq!(
                batch.schema(),
                advertised,
                "the reader's schema must be identical to the advertised one in the steady state"
            );
        }
    }

    /// The nested-evolution fixture `(id int, s struct<a int>)`. `s.a` is field 3, so an added
    /// `s.b` takes field 4, as Iceberg assigns.
    async fn get_test_catalog_and_struct_table()
    -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
        use iceberg::spec::StructType;

        let temp_dir = TempDir::new().expect("temp dir");
        let warehouse_path = temp_dir.path().to_str().expect("utf-8 path").to_string();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
            )
            .await
            .expect("memory catalog");
        let namespace = NamespaceIdent::new("test_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace");

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .expect("nested schema");

        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("nested_table".to_string())
                    .location(format!("{warehouse_path}/nested_table"))
                    .schema(schema)
                    .properties(HashMap::new())
                    .build(),
            )
            .await
            .expect("create nested table");

        (
            Arc::new(catalog),
            namespace,
            "nested_table".to_string(),
            temp_dir,
        )
    }

    /// Adds a NESTED column out of band: `ALTER TABLE ADD COLUMN <parent>.<name> int`.
    async fn evolve_add_nested_column(
        catalog: &Arc<dyn Catalog>,
        ident: &TableIdent,
        parent: &str,
        name: &str,
    ) {
        use iceberg::transaction::{ApplyTransactionAction, Transaction};

        let table = catalog
            .load_table(ident)
            .await
            .expect("load table for out-of-band evolution");
        let tx = Transaction::new(&table);
        let tx = tx
            .update_schema()
            .add_column_to(
                Some(parent),
                name,
                Type::Primitive(PrimitiveType::Int),
                None,
            )
            .apply(tx)
            .expect("queue the nested add-column");
        tx.commit(catalog.as_ref())
            .await
            .expect("commit the nested add-column");
    }

    /// `ADD COLUMN s.b` creates no snapshot, so the scanned struct holds only `a` while the plan
    /// advertises `{a, b}`. Conforming must recurse into the struct, as Spark does.
    #[tokio::test]
    async fn test_nested_add_column_reads_null_for_the_new_field() {
        use datafusion::arrow::array::{Array, Int32Array, StructArray};

        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_struct_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, named_struct('a', 5))",
        )
        .await;

        evolve_add_nested_column(&catalog, &ident, "s", "b").await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a provider on the nested-evolved table");
        let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

        let mut seen = 0;
        for batch in &batches {
            let structs = batch
                .column_by_name("s")
                .expect("the struct column must be present")
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("s must be a struct");
            assert_eq!(
                structs.num_columns(),
                2,
                "the struct must carry the advertised child set, got {:?}",
                structs.data_type()
            );
            let a = structs
                .column_by_name("a")
                .expect("s.a must be present")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("s.a must be Int32");
            let b = structs.column_by_name("b").expect("s.b must be present");
            for i in 0..a.len() {
                assert_eq!(a.value(i), 5, "the stored nested value must survive");
                assert!(b.is_null(i), "a nested column added later reads as NULL");
                seen += 1;
            }
        }
        assert_eq!(seen, 1, "the seeded row must be readable");
    }

    /// Renames a column out of band; `name` may be a dotted path for a nested field.
    async fn evolve_rename(catalog: &Arc<dyn Catalog>, ident: &TableIdent, name: &str, to: &str) {
        use iceberg::transaction::{ApplyTransactionAction, Transaction};

        let table = catalog
            .load_table(ident)
            .await
            .expect("load table for out-of-band evolution");
        let tx = Transaction::new(&table);
        let tx = tx
            .update_schema()
            .rename_column(name, to)
            .apply(tx)
            .expect("queue the rename");
        tx.commit(catalog.as_ref())
            .await
            .expect("commit the rename");
    }

    /// The nested rename: the child keeps its field id, so its value comes back under the new name.
    #[tokio::test]
    async fn test_nested_rename_preserves_values() {
        use datafusion::arrow::array::{Array, Int32Array, StructArray};

        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_struct_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, named_struct('a', 5))",
        )
        .await;

        evolve_rename(&catalog, &ident, "s.a", "renamed_a").await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a provider on the nested-renamed table");
        let batches = query_through(Arc::new(provider), "SELECT * FROM t").await;

        let mut seen = 0;
        for batch in &batches {
            let structs = batch
                .column_by_name("s")
                .expect("the struct column must be present")
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("s must be a struct");
            let renamed = structs
                .column_by_name("renamed_a")
                .expect("the renamed child must be present under its NEW name")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("s.renamed_a must be Int32");
            for i in 0..renamed.len() {
                assert!(
                    renamed.is_valid(i),
                    "the renamed nested column must carry its data, not NULLs"
                );
                assert_eq!(renamed.value(i), 5);
                seen += 1;
            }
        }
        assert_eq!(seen, 1, "the seeded row must be readable");
    }

    // ===== Pushed-down-filter rebinding =====
    //
    // A pushed filter PRUNES rows before DataFusion sees them, and `Inexact` pushdown only discards
    // false positives. These three pin how a name-keyed filter fails once names and ids disagree.

    /// A table `(a int, b int)` for the name-swap case: one shared type, so a filter bound to the
    /// wrong column type-checks and returns the wrong rows.
    async fn get_test_catalog_and_two_int_table()
    -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
        let temp_dir = TempDir::new().expect("temp dir");
        let warehouse_path = temp_dir.path().to_str().expect("utf-8 path").to_string();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
            )
            .await
            .expect("memory catalog");
        let namespace = NamespaceIdent::new("test_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace");

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "b", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .expect("two-int schema");

        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("swap_table".to_string())
                    .location(format!("{warehouse_path}/swap_table"))
                    .schema(schema)
                    .properties(HashMap::new())
                    .build(),
            )
            .await
            .expect("create two-int table");

        (
            Arc::new(catalog),
            namespace,
            "swap_table".to_string(),
            temp_dir,
        )
    }

    /// After a rename, a filter over the NEW name must go down under the snapshot's name. The
    /// advertised name fails to bind, so the query dies instead of returning the row.
    #[tokio::test]
    async fn test_pushdown_after_a_rename_binds_the_snapshot_name() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("opt")).await;
        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, 'a', 7), (2, 'b', 8)",
        )
        .await;
        evolve_rename(&catalog, &ident, "opt", "opt2").await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a provider on the renamed table");
        let batches = query_through(Arc::new(provider), "SELECT id FROM t WHERE opt2 = 7").await;
        assert_eq!(
            batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            1,
            "exactly the row with opt2 = 7 must come back"
        );
    }

    /// After DROP and re-ADD the advertised column has a FRESH field id the snapshot lacks, so
    /// every row reads NULL. Pushing under the old name prunes rows DataFusion cannot get back.
    #[tokio::test]
    async fn test_pushdown_after_drop_and_readd_keeps_the_rows() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("opt")).await;
        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, 'a', 7), (2, 'b', 8)",
        )
        .await;
        evolve_schema(&catalog, &ident, SchemaOp::Drop("opt")).await;
        evolve_schema(&catalog, &ident, SchemaOp::AddOptionalInt("opt")).await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a provider on the re-added column");
        let batches = query_through(Arc::new(provider), "SELECT id FROM t WHERE opt IS NULL").await;
        assert_eq!(
            batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            2,
            "the re-added column reads NULL for every row, so both must match"
        );
    }

    /// The silent one: swap two names, and a filter pushed under the advertised name binds to the
    /// OTHER column's data. Same type, no error, wrong rows.
    #[tokio::test]
    async fn test_pushdown_after_a_name_swap_filters_the_right_column() {
        let (catalog, namespace, table_name, _temp_dir) =
            get_test_catalog_and_two_int_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, 2)",
        )
        .await;

        // a -> tmp, b -> a, tmp -> b: field 1 is now called `b` and field 2 is now called `a`.
        evolve_rename(&catalog, &ident, "a", "tmp").await;
        evolve_rename(&catalog, &ident, "b", "a").await;
        evolve_rename(&catalog, &ident, "tmp", "b").await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct a provider on the swapped table");
        // Only the NAMES moved. The column advertised as `a` is field id 2, which the snapshot
        // still calls `b`, holding the value 2.
        assert_eq!(provider.schema().field(0).name(), "b");
        assert_eq!(provider.schema().field(1).name(), "a");

        let batches = query_through(Arc::new(provider), "SELECT * FROM t WHERE a = 2").await;
        assert_eq!(
            batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            1,
            "`a` now holds the value 2, so the row must match"
        );
    }

    /// The scan must read ONLY the projected column. It drives
    /// [`crate::physical_plan::scan::get_batch_stream`] with the resolved column set, so a revert
    /// to `select_all()` widens the pre-conform batch and REDs here. `conform_batch` would
    /// otherwise hide it, because it narrows the batch again.
    #[tokio::test]
    async fn test_scan_reads_only_the_projected_column() {
        use futures::TryStreamExt;

        use crate::physical_plan::scan::get_batch_stream;

        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        seed(
            &catalog,
            &namespace,
            &table_name,
            "INSERT INTO t VALUES (1, 'a')",
        )
        .await;

        let table = catalog
            .load_table(&TableIdent::new(namespace.clone(), table_name.clone()))
            .await
            .expect("load the seeded table");
        let batches: Vec<_> = get_batch_stream(
            table,
            None,
            vec!["name".to_string()],
            None,
            crate::physical_plan::scan::ScanKnobs::default(),
        )
        .await
        .expect("open the scan stream")
        .try_collect()
        .await
        .expect("read the scan stream");

        assert!(!batches.is_empty(), "the seeded row must produce a batch");
        for batch in &batches {
            assert_eq!(
                batch.num_columns(),
                1,
                "only the projected column may be read, got {:?}",
                batch.schema()
            );
            assert_eq!(batch.schema().field(0).name(), "name");
        }
    }

    #[tokio::test]
    async fn test_no_limit_pushdown() {
        use datafusion::datasource::TableProvider;

        let (_catalog, _ns, _name, table, _tmp) = get_static_test_table().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table)
            .await
            .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        let scan_plan = table_provider.scan(&state, None, &[], None).await.unwrap();

        let iceberg_scan = scan_plan
            .downcast_ref::<IcebergTableScan>()
            .expect("Expected IcebergTableScan");

        assert_eq!(
            iceberg_scan.limit(),
            None,
            "Limit should be None when not specified"
        );
    }

    /// Incomplete metadata must surface a planning error, not demote to `UnknownPartitioning(1)`.
    #[tokio::test]
    async fn test_plan_tasks_failure_fail_closed_not_n1_demote() {
        use datafusion::datasource::TableProvider;

        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table)
            .await
            .unwrap();
        let ctx = SessionContext::new();
        let state = ctx.state();
        let err = table_provider
            .scan(&state, None, &[], None)
            .await
            .expect_err("incomplete fixture must fail plan, not demote to N=1");
        let msg = err.to_string();
        assert!(
            msg.contains("manifest")
                || msg.contains("file")
                || msg.contains("Failed")
                || msg.contains("not found")
                || msg.contains("No such"),
            "expected planning/IO root cause, got: {msg}"
        );
    }
}
