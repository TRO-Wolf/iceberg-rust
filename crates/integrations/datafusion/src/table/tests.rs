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

pub(super) async fn get_test_table_from_metadata_file() -> Table {
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

pub(super) async fn get_test_catalog_and_table()
-> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
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

pub(super) async fn get_partitioned_test_catalog_and_table(
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
            iceberg::spec::TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED.to_string(),
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

pub(super) fn plan_contains_sort(plan: &Arc<dyn ExecutionPlan>) -> bool {
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

pub(super) async fn get_static_test_table()
-> (Arc<dyn Catalog>, NamespaceIdent, String, Table, TempDir) {
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

pub(super) enum SchemaOp<'a> {
    AddOptionalInt(&'a str),
    Rename(&'a str, &'a str),
    PromoteToLong(&'a str),
    Drop(&'a str),
}

pub(super) async fn evolve_schema(
    catalog: &Arc<dyn Catalog>,
    ident: &TableIdent,
    op: SchemaOp<'_>,
) {
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

pub(super) async fn query_through(
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

pub(super) async fn seed(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    table_name: &str,
    sql: &str,
) {
    let provider =
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.to_string())
            .await
            .expect("construct a provider for the seed write");
    let batches: Vec<_> = query_through(Arc::new(provider), sql).await;
    assert!(!batches.is_empty(), "a write must report its row count");
}

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

#[tokio::test]
async fn test_scan_batches_match_advertised_schema_after_add_column() {
    let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;
    let ident = TableIdent::new(namespace.clone(), table_name.clone());

    {
        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
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
