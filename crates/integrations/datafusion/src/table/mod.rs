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
//! This module provides two table provider implementations:
//!
//! - [`IcebergTableProvider`]: Catalog-backed provider with automatic metadata refresh.
//!   Use for write operations and when you need to see the latest table state.
//!
//! - [`IcebergStaticTableProvider`]: Static provider for read-only access to a specific
//!   table snapshot. Use for consistent analytical queries or time-travel scenarios.

pub mod metadata_table;
pub mod table_provider_factory;

use std::any::Any;
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

use crate::error::to_datafusion_error;
use crate::physical_plan::commit::IcebergCommitExec;
use crate::physical_plan::delete::{
    IcebergDeleteExec, IcebergUpdateExec, IsolationLevel, WRITE_DELETE_ISOLATION_LEVEL,
    WRITE_DELETE_MODE, WRITE_UPDATE_ISOLATION_LEVEL, WRITE_UPDATE_MODE, WriteMode,
};
use crate::physical_plan::project::project_with_partition;
use crate::physical_plan::repartition::repartition;
use crate::physical_plan::scan::IcebergTableScan;
use crate::physical_plan::sort::sort_by_partition;
use crate::physical_plan::write::IcebergWriteExec;

/// Catalog-backed table provider with automatic metadata refresh.
///
/// This provider loads fresh table metadata from the catalog on every scan and write
/// operation, ensuring you always see the latest table state. Use this when you need
/// write operations or want to see the most up-to-date data.
///
/// For read-only access to a specific snapshot without catalog overhead, use
/// [`IcebergStaticTableProvider`] instead.
///
/// # Schema freshness (BUG-005) — a new INSTANCE, never a mutating one
///
/// [`TableProvider::schema`] is synchronous, so it can only ever return a snapshot of the table's
/// Arrow schema. This provider's snapshot is fixed for the life of the INSTANCE, and that
/// immutability is load-bearing rather than an oversight: DataFusion resolves a projection to
/// ORDINALS against `schema()` at planning time and stores them in the logical plan (a view, a
/// deferred `DataFrame`). A provider whose schema mutates under such a plan makes those ordinals
/// lie — DataFusion's own `TableScan::try_new` indexes the schema with them and PANICS with
/// "index out of bounds" when the schema has since shrunk, and silently maps the wrong columns when a
/// drop-plus-add keeps the length. So the schema an instance advertises never changes.
///
/// Freshness therefore comes from a NEW instance, exactly as it does in Java: Spark's `SparkTable`
/// is built per query by `SparkCatalog.loadTable`, so a plan never sees its table's schema move
/// underneath it. Two ways to get one:
///
/// * the catalog path (the normal one) — `IcebergSchemaProvider::table` builds a provider bound to
///   freshly loaded metadata on EVERY resolution, so each query plans against the current schema
///   with no action from the caller;
/// * [`IcebergTableProvider::refreshed`] — for a provider registered directly with a
///   `SessionContext`, returns a NEW provider carrying the table's current schema, leaving this one
///   (and every plan built against it) untouched. Re-register it to pick up the evolution.
///
/// DATA is always current regardless: every operation reloads the table from the catalog, so a scan
/// reads the latest snapshot even when it is advertising an older schema. [`IcebergTableScan`]
/// reconciles the two by field id — see there.
#[derive(Debug, Clone)]
pub struct IcebergTableProvider {
    /// The catalog that manages this table
    catalog: Arc<dyn Catalog>,
    /// The table identifier (namespace + name)
    table_ident: TableIdent,
    /// The Arrow schema this instance advertises. FIXED for the life of the instance — DataFusion
    /// stores projection ordinals resolved against it (see the type docs).
    schema: ArrowSchemaRef,
}

impl IcebergTableProvider {
    /// Creates a new catalog-backed table provider.
    ///
    /// Loads the table once to capture the schema this instance will advertise, then keeps the
    /// catalog reference so every operation can reload the table's data.
    pub(crate) async fn try_new(
        catalog: Arc<dyn Catalog>,
        namespace: NamespaceIdent,
        name: impl Into<String>,
    ) -> Result<Self> {
        let table_ident = TableIdent::new(namespace, name.into());

        // Load table once to get initial schema
        let table = catalog.load_table(&table_ident).await?;
        let schema = Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?);

        Ok(IcebergTableProvider {
            catalog,
            table_ident,
            schema,
        })
    }

    /// Returns a NEW provider for the same table, advertising the table's current schema.
    ///
    /// This provider is left untouched, so logical plans already built against it stay valid — that
    /// is the whole point of returning a new instance instead of mutating this one (see the type
    /// docs). Register the result to plan subsequent queries against the evolved schema; callers
    /// going through [`crate::IcebergCatalogProvider`] never need this, since each query resolves a
    /// fresh provider already.
    pub async fn refreshed(&self) -> Result<Self> {
        let table = self.catalog.load_table(&self.table_ident).await?;
        Ok(IcebergTableProvider {
            catalog: self.catalog.clone(),
            table_ident: self.table_ident.clone(),
            schema: Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?),
        })
    }

    /// Loads the table's current state from the catalog, with the Arrow form of its CURRENT schema.
    ///
    /// The write paths plan against this rather than against [`Self::schema`]: they re-scan and
    /// commit against the table state they load, so their row filters, assignment indices and
    /// projections must describe that same state.
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
        // Load fresh table metadata for metadata table access
        let table = self.catalog.load_table(&self.table_ident).await?;
        IcebergMetadataTableProvider::try_new(table, r#type)
    }
}

#[async_trait]
impl TableProvider for IcebergTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Load fresh table metadata from catalog — the DATA is always current.
        let table = self
            .catalog
            .load_table(&self.table_ident)
            .await
            .map_err(to_datafusion_error)?;

        // `self.schema` IS the schema DataFusion planned against: `projection` holds ordinals it
        // resolved against `schema()`, and this instance's schema cannot have moved since (see the
        // type docs). `IcebergTableScan` binds that schema to the freshly loaded table by FIELD ID,
        // so the emitted batches match what this plan advertises even when the table has evolved.
        Ok(Arc::new(IcebergTableScan::new(
            table,
            None, // Always use current snapshot for catalog-backed provider
            self.schema.clone(),
            projection,
            filters,
            limit,
        )?))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // Push down all filters, as a single source of truth, the scanner will drop the filters which couldn't be push down
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Load fresh table metadata from catalog. The write is planned against the CURRENT schema —
        // the table state the data files will be written and committed against.
        let (table, current_schema) = self
            .load_table_with_current_schema()
            .await
            .map_err(to_datafusion_error)?;

        let partition_spec = table.metadata().default_partition_spec();

        // Step 1: Project partition values for partitioned tables
        let plan_with_partition = if !partition_spec.is_unpartitioned() {
            project_with_partition(input, &table)?
        } else {
            input
        };

        // Step 2: Repartition for parallel processing
        let target_partitions =
            NonZeroUsize::new(state.config().target_partitions()).ok_or_else(|| {
                DataFusionError::Configuration(
                    "target_partitions must be greater than 0".to_string(),
                )
            })?;

        let repartitioned_plan =
            repartition(plan_with_partition, table.metadata_ref(), target_partitions)?;

        // Apply sort node when it's not fanout mode
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

        Ok(Arc::new(IcebergCommitExec::new(
            table,
            self.catalog.clone(),
            coalesce_partitions,
            current_schema,
            insert_op,
        )))
    }

    async fn delete_from(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Load fresh table metadata from the catalog. Everything below — the row filter's binding
        // schema and the exec's projection base — is derived from THIS state's schema, never from
        // the cached one: the exec re-scans the table itself, and a projection naming a column the
        // table no longer has fails the whole DELETE (BUG-005).
        let (table, current_schema) = self
            .load_table_with_current_schema()
            .await
            .map_err(to_datafusion_error)?;
        let mode = WriteMode::from_property(&table, WRITE_DELETE_MODE);
        // §5 isolation level, resolved at PLAN time like Java's row-level-operation builder
        // (`SparkRowLevelOperationBuilder` ctor); default serializable (Java's per-op default).
        let isolation = IsolationLevel::for_row_level_op(&table, WRITE_DELETE_ISOLATION_LEVEL)?;

        // Build the EXACT row filter as a `PhysicalExpr` (the `WHERE` clause, AND-combined). We
        // evaluate this ourselves against the scanned rows rather than relying on Iceberg predicate
        // pushdown, which is INEXACT and would over-delete (see `physical_plan::delete`). An empty
        // filter set means `DELETE FROM t` — delete every row.
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
            mode,
            isolation,
            current_schema,
        )))
    }

    async fn update(
        &self,
        state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // As in `delete_from`: the CURRENT schema is the one the row filter, the `SET` assignment
        // indices and the exec's projection base are all derived from (BUG-005).
        let (table, current_schema) = self
            .load_table_with_current_schema()
            .await
            .map_err(to_datafusion_error)?;
        let mode = WriteMode::from_property(&table, WRITE_UPDATE_MODE);
        // §5 isolation level, resolved at PLAN time (see `delete_from`); default serializable.
        let isolation = IsolationLevel::for_row_level_op(&table, WRITE_UPDATE_ISOLATION_LEVEL)?;

        let df_schema = DFSchema::try_from(current_schema.as_ref().clone())?;

        // The WHERE clause as an EXACT `PhysicalExpr` (see `delete_from` on why Iceberg pushdown is
        // unsafe for a row-level mutation). `None` means update every row.
        let predicate = match filters.into_iter().reduce(Expr::and) {
            None => None,
            Some(combined) => Some(state.create_physical_expr(combined, &df_schema)?),
        };

        // Resolve each `SET col = expr` to `(table-column index, value PhysicalExpr)`.
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
            physical_assignments,
            mode,
            isolation,
            current_schema,
        )))
    }
}

/// Static table provider for read-only snapshot access.
///
/// This provider holds a cached table instance and does not refresh metadata or support
/// write operations. Use this for consistent analytical queries, time-travel scenarios,
/// or when you want to avoid catalog overhead.
///
/// For catalog-backed tables with write support and automatic refresh, use
/// [`IcebergTableProvider`] instead.
#[derive(Debug, Clone)]
pub struct IcebergStaticTableProvider {
    /// The static table instance (never refreshed)
    table: Table,
    /// Optional snapshot ID for this static view
    snapshot_id: Option<i64>,
    /// A reference-counted arrow `Schema`
    schema: ArrowSchemaRef,
}

impl IcebergStaticTableProvider {
    /// Creates a static provider from a table instance.
    ///
    /// Uses the table's current snapshot for all queries. Does not support write operations.
    pub async fn try_new_from_table(table: Table) -> Result<Self> {
        let schema = Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?);
        Ok(IcebergStaticTableProvider {
            table,
            snapshot_id: None,
            schema,
        })
    }

    /// Creates a static provider for a specific table snapshot.
    ///
    /// Queries the specified snapshot for all operations. Useful for time-travel queries.
    /// Does not support write operations.
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
        let schema = Arc::new(schema_to_arrow_schema(&table_schema)?);
        Ok(IcebergStaticTableProvider {
            table,
            snapshot_id: Some(snapshot_id),
            schema,
        })
    }
}

#[async_trait]
impl TableProvider for IcebergStaticTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Use cached table (no refresh)
        Ok(Arc::new(IcebergTableScan::new(
            self.table.clone(),
            self.snapshot_id,
            self.schema.clone(),
            projection,
            filters,
            limit,
        )?))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // Push down all filters, as a single source of truth, the scanner will drop the filters which couldn't be push down
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

        // Attempt to insert into the static provider should fail
        let result = ctx.sql("INSERT INTO mytable VALUES (1, 2, 3)").await;

        // The error should occur during planning or execution
        // We expect an error indicating write operations are not supported
        assert!(
            result.is_err() || {
                let df = result.unwrap();
                df.collect().await.is_err()
            }
        );
    }

    #[tokio::test]
    async fn test_static_provider_scan() {
        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();

        // Test that scan operations work correctly
        let df = ctx.sql("SELECT count(*) FROM mytable").await.unwrap();
        let physical_plan = df.create_physical_plan().await;
        assert!(physical_plan.is_ok());
    }

    // Tests for IcebergTableProvider

    #[tokio::test]
    async fn test_catalog_backed_provider_creation() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        // Test creating a catalog-backed provider
        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        // Verify the schema is loaded correctly
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

        // Test that scan operations work correctly
        let df = ctx.sql("SELECT * FROM test_table").await.unwrap();

        // Verify the schema in the query result
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

        // Test that insert operations work correctly
        let result = ctx.sql("INSERT INTO test_table VALUES (1, 'test')").await;

        // Insert should succeed (or at least not fail during planning)
        assert!(result.is_ok());

        // Try to execute the insert plan
        let df = result.unwrap();
        let execution_result = df.collect().await;

        // The execution should succeed
        assert!(execution_result.is_ok());
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

        // Create a query plan
        let df = ctx.sql("SELECT id, name FROM test_table").await.unwrap();

        // Get logical schema before consuming df
        let logical_schema = df.schema().clone();

        // Get physical plan (this consumes df)
        let physical_plan = df.create_physical_plan().await.unwrap();
        let physical_schema = physical_plan.schema();

        // Verify that logical and physical schemas are consistent
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

    /// Helper to check if a plan contains a SortExec node
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

        // When fanout is enabled (default), no sort node should be added
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

        // With fanout enabled, there should be no SortExec in the plan
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

        // When fanout is disabled, a sort node should be added
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

        // With fanout disabled, there should be a SortExec in the plan
        assert!(
            plan_contains_sort(&insert_plan),
            "Plan should contain SortExec when fanout is disabled"
        );
    }

    #[tokio::test]
    async fn test_limit_pushdown_static_provider() {
        use datafusion::datasource::TableProvider;

        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        // Test scan with limit
        let scan_plan = table_provider
            .scan(&state, None, &[], Some(10))
            .await
            .unwrap();

        // Verify that the scan plan is an IcebergTableScan
        let iceberg_scan = scan_plan
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .expect("Expected IcebergTableScan");

        // Verify the limit is set
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

        // Test scan with limit
        let scan_plan = provider.scan(&state, None, &[], Some(5)).await.unwrap();

        // Verify that the scan plan is an IcebergTableScan
        let iceberg_scan = scan_plan
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .expect("Expected IcebergTableScan");

        // Verify the limit is set
        assert_eq!(
            iceberg_scan.limit(),
            Some(5),
            "Limit should be set to 5 in the scan plan"
        );
    }

    // Live-schema regressions (BUG-005 / BUG-011)
    //
    // These pin the two halves of the "frozen schema" defect class:
    //   * BUG-005 — the provider cached the Arrow schema at construction, so a long-lived provider
    //     (the `IcebergSchemaProvider` caches one FOREVER) planned every later query against a
    //     schema that no longer described the table.
    //   * BUG-011 — the scan reloaded the table but the stream adapter still advertised the
    //     construction-time schema, so the emitted batches could not match the advertised schema.

    /// A schema evolution another engine applies out of band. NONE of these create a snapshot, so
    /// after any of them the table's CURRENT schema and the schema its data is read with (the
    /// current snapshot's) disagree — which is the whole point.
    enum SchemaOp<'a> {
        /// `ALTER TABLE ADD COLUMN <name> int` (optional).
        AddOptionalInt(&'a str),
        /// `ALTER TABLE RENAME COLUMN <from> TO <to>` — keeps the field id, changes only the name.
        Rename(&'a str, &'a str),
        /// `ALTER TABLE ALTER COLUMN <name> TYPE bigint` — a legal Iceberg int → long promotion.
        PromoteToLong(&'a str),
        /// `ALTER TABLE DROP COLUMN <name>`.
        Drop(&'a str),
    }

    /// Applies an out-of-band schema evolution through a SECOND catalog handle — exactly what
    /// another engine or writer does. The provider under test never sees this transaction.
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

    /// Registers `provider` in a fresh session and runs `sql`, returning the batches.
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

    /// BUG-005 + S1-2: an instance's advertised schema is STABLE — DataFusion stores projection
    /// ordinals resolved against it — and freshness comes from a NEW instance via `refreshed()`.
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

        // An ordinary operation must NOT move the advertised schema: plans already hold ordinals
        // resolved against it.
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

        // A NEW instance carries the current schema; the original is untouched.
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

    /// BUG-005 on the SHIPPING path: queries that go through the catalog resolve a provider per
    /// planning round (Java: `SparkCatalog.loadTable` per query), so a session sees an out-of-band
    /// evolution on its very next query without any refresh call.
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

    /// BUG-011, the same-instant case: `ADD COLUMN` does not create a snapshot, so the table's
    /// CURRENT schema (what the provider advertises) has a column the scanned snapshot's schema
    /// does not. The scan must still emit batches that match what it advertised — the added column
    /// read as NULL, per Java (readers project the table schema and null-fill absent fields).
    #[tokio::test]
    async fn test_scan_batches_match_advertised_schema_after_add_column() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        // One committed row, so the table has a snapshot whose schema is the 2-column original.
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

    /// BUG-011, the stale-provider case: a provider built before an evolution advertises the OLD
    /// schema at planning time, while the scan reloads the table and would otherwise `select_all()`
    /// the NEW column set. The emitted batches must match the schema the plan advertised — and the
    /// provider must then self-heal, so the next query sees the evolved schema.
    #[tokio::test]
    async fn test_stale_provider_scan_is_self_consistent() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
        let ident = TableIdent::new(namespace.clone(), table_name.clone());

        let stale_provider = Arc::new(
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .expect("construct the provider BEFORE the evolution"),
        );

        // Out-of-band: add a column AND write a row through it, so the table's current snapshot
        // carries the 3-column schema while `stale_provider` still advertises 2 columns.
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

        // And it stays that way: the instance's advertised schema does not move, so a second query
        // through the SAME provider is planned and answered identically (freshness comes from a new
        // instance — see `test_catalog_resolves_a_fresh_provider_per_query`).
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

    /// BUG-011's silent path: with NO projection the scan used to ask the (reloaded) table for
    /// `select_all()` — whatever column set it has NOW — while the stream adapter still advertised
    /// the planning-time schema. DataFusion addresses batch columns by ordinal against the schema
    /// its child advertised, so an extra or reordered column there is silent corruption. Whatever
    /// the plan advertises, the batches must match it exactly.
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
        // `projection: None` — the path that used to become `select_all()`.
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

    /// Builds a table with an out-of-band-added `extra` column AND a row written through it, so the
    /// table's current snapshot carries the 3-column schema, and returns a provider that was
    /// constructed BEFORE the evolution (its cache still says 2 columns).
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

    /// Sums the `count` column of a DML result.
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

    /// The DML half (BUG-005): `delete_from` must bind its row filter — and the projection base the
    /// exec re-scans with — to the table's CURRENT schema, not the provider's cached one. A filter
    /// over a column added out of band cannot even be bound against the cached schema.
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

    /// The DML half (BUG-005), assignment side: `update` resolves each `SET` target against the
    /// CURRENT schema, so the assignment's column index and the exec's projection base describe the
    /// same table state. Against the cached schema, a column added out of band is "unknown".
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

    /// S1-1: `RENAME COLUMN` keeps the field id, rewrites the name and creates NO snapshot, so the
    /// advertised (current) name and the scanned snapshot's name for the SAME column differ. Binding
    /// by name would null-fill over live data — losing the value silently. Binding by field id
    /// returns the value under its new name.
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

        // The rename lands AFTER the write, so the data is stored under the old name/id.
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

    /// S1-1, the REQUIRED-column case: a renamed required column has no null-fill escape hatch, so
    /// name-binding would fail the query outright. Field-id binding returns the values.
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

    /// S1-2 (probe F): a VIEW captures the provider and its projection ORDINALS. If the provider's
    /// advertised schema shrank under it, DataFusion's own `TableScan::try_new` would index the
    /// shorter schema with a stale ordinal and PANIC. Two rounds, with a column dropped out of band
    /// in between, must both answer normally.
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

        // The out-of-band drop that used to make the stored ordinals lie. It lands BEFORE the first
        // round, so a provider that republished its schema during round 1 would leave round 2's
        // re-plan indexing a SHORTER schema with round-1's ordinals — DataFusion's own
        // `TableScan::try_new` panics with "index out of bounds" when that happens.
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

    /// S1-2 (probe C): a DataFrame built before an evolution and collected after it must not panic
    /// either — its plan holds ordinals against the schema it was planned with.
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

        // Plan now...
        let df = ctx.sql("SELECT name FROM t").await.expect("plan the query");

        // ...evolve out of band...
        evolve_schema(&catalog, &ident, SchemaOp::Drop("id")).await;

        // ...and only then execute — twice, because the second physical-planning round is what
        // re-indexes the (possibly moved) provider schema with the plan's stored ordinals.
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

    /// S2-3: `int` → `long` is a LEGAL Iceberg promotion and creates no snapshot, so the scanned
    /// data is still `int` while the plan advertises `long`. The values must be read, widened.
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

    /// LOW-5 (measured, then pinned): in the steady state the scanned batch's schema is IDENTICAL to
    /// the advertised one — same fields, same `PARQUET:field_id` metadata — so `conform_batch`'s
    /// equality fast path hits and no batch is rebuilt. A reader change that broke that identity
    /// would silently move every scan onto the rebuild path; this catches it.
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

    #[tokio::test]
    async fn test_no_limit_pushdown() {
        use datafusion::datasource::TableProvider;

        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        // Test scan without limit
        let scan_plan = table_provider.scan(&state, None, &[], None).await.unwrap();

        // Verify that the scan plan is an IcebergTableScan
        let iceberg_scan = scan_plan
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .expect("Expected IcebergTableScan");

        // Verify the limit is None
        assert_eq!(
            iceberg_scan.limit(),
            None,
            "Limit should be None when not specified"
        );
    }
}
