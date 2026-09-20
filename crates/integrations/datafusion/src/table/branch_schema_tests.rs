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

use datafusion::arrow::array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
use datafusion::datasource::TableProvider;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{FormatVersion, NestedField, PrimitiveType, Schema, Type};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

use super::tests::{SchemaOp, evolve_schema};
use super::{IcebergStaticTableProvider, IcebergTableProvider};
use crate::physical_plan::scan::IcebergTableScan;

fn cell_schema(id_type: PrimitiveType) -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(id_type)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "cat", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("cell schema must build")
}

async fn setup_table(
    version: FormatVersion,
    schema: Schema,
    properties: HashMap<String, String>,
) -> (Arc<dyn Catalog>, NamespaceIdent, TableIdent, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let warehouse_path = temp_dir.path().to_str().unwrap().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .unwrap();
    let namespace = NamespaceIdent::new("bs_ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let creation = TableCreation::builder()
        .name("bs_table".to_string())
        .location(format!("{warehouse_path}/bs_table"))
        .schema(schema)
        .properties(properties)
        .format_version(version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create the bs table");
    let ident = TableIdent::new(namespace.clone(), "bs_table".to_string());
    (Arc::new(catalog), namespace, ident, temp_dir)
}

async fn load(catalog: &Arc<dyn Catalog>, ident: &TableIdent) -> Table {
    catalog.load_table(ident).await.expect("reload the table")
}

async fn sql_exec(provider: Arc<dyn TableProvider>, sql: &str) -> Vec<RecordBatch> {
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

async fn seed_two(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    name: &str,
    branch: Option<&str>,
) {
    let mut provider = IcebergTableProvider::try_new(
        catalog.clone(),
        namespace.clone(),
        name.to_string(),
    )
    .await
    .expect("provider for the seed write");
    if let Some(branch) = branch {
        provider = provider.with_commit_branch(branch);
    }
    let batches = sql_exec(
        Arc::new(provider),
        "INSERT INTO t VALUES (1, 'a', 'x'), (2, 'b', 'y')",
    )
    .await;
    assert!(!batches.is_empty(), "a write must report its row count");
}

fn current_snapshot_id(table: &Table) -> i64 {
    table
        .metadata()
        .current_snapshot()
        .expect("a seeded table has a current snapshot")
        .snapshot_id()
}

async fn create_ref(
    catalog: &Arc<dyn Catalog>,
    ident: &TableIdent,
    name: &str,
    snapshot_id: i64,
    branch: bool,
) {
    let table = load(catalog, ident).await;
    let tx = Transaction::new(&table);
    let action = tx.manage_snapshots();
    let action = if branch {
        action.create_branch(name, snapshot_id)
    } else {
        action.create_tag(name, snapshot_id)
    };
    let tx = action.apply(tx).expect("queue the ref creation");
    tx.commit(catalog.as_ref())
        .await
        .expect("commit the ref creation");
}

async fn add_z(catalog: &Arc<dyn Catalog>, ident: &TableIdent) {
    evolve_schema(catalog, ident, SchemaOp::AddOptionalInt("z")).await;
}

async fn provider_for_ref(table: &Table, ref_name: &str) -> IcebergStaticTableProvider {
    let snapshot_id = table
        .metadata()
        .snapshot_for_ref(ref_name)
        .unwrap_or_else(|| panic!("the test ref '{ref_name}' must exist"))
        .snapshot_id();
    IcebergStaticTableProvider::try_new_from_table_snapshot(table.clone(), snapshot_id)
        .await
        .expect("snapshot-schema provider")
}

async fn provider_for_snapshot(table: &Table, snapshot_id: i64) -> IcebergStaticTableProvider {
    IcebergStaticTableProvider::try_new_from_table_snapshot(table.clone(), snapshot_id)
        .await
        .expect("snapshot provider")
}

fn rows_as_strings(batches: &[RecordBatch]) -> Vec<String> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let cells: Vec<String> = (0..batch.num_columns())
                .map(|col| cell_to_string(batch.column(col).as_ref(), row))
                .collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    rows
}

fn cell_to_string(array: &dyn Array, row: usize) -> String {
    if array.is_null(row) {
        return "NULL".to_string();
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        return array.value(row).to_string();
    }
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        return array.value(row).to_string();
    }
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return array.value(row).to_string();
    }
    panic!("unsupported column type in cell_to_string: {}", array.data_type())
}

async fn run_query(provider: Arc<dyn TableProvider>, sql: &str) -> (Vec<String>, Vec<String>) {
    let ctx = SessionContext::new();
    ctx.register_table("t", provider)
        .expect("register the provider under test");
    let df = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|e| panic!("plan `{sql}`: {e}"));
    let cols: Vec<String> = df
        .schema()
        .fields()
        .iter()
        .map(|field| format!("{}:{:?}", field.name(), field.data_type()))
        .collect();
    let batches = df
        .collect()
        .await
        .unwrap_or_else(|e| panic!("execute `{sql}`: {e}"));
    (cols, rows_as_strings(&batches))
}

fn assert_shape(cols: &[String], expected: &[&str]) {
    let expected: Vec<String> = expected.iter().map(ToString::to_string).collect();
    assert_eq!(cols, &expected, "columns must match the cell");
}

fn assert_rows(rows: Vec<String>, expected: &[&str]) {
    let mut expected: Vec<String> = expected.iter().map(ToString::to_string).collect();
    expected.sort();
    assert_eq!(rows, expected, "rows must match the cell");
}

fn find_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<&IcebergTableScan> {
    if let Some(scan) = plan.downcast_ref::<IcebergTableScan>() {
        return Some(scan);
    }
    plan.children().iter().find_map(|child| find_scan(child))
}

async fn branch_add_setup(version: FormatVersion) -> (Arc<dyn Catalog>, TableIdent, TempDir, i64) {
    let (catalog, namespace, ident, temp_dir) = setup_table(
        version,
        cell_schema(PrimitiveType::Long),
        HashMap::new(),
    )
    .await;
    seed_two(&catalog, &namespace, ident.name(), None).await;
    let table = load(&catalog, &ident).await;
    let snapshot_id = current_snapshot_id(&table);
    create_ref(&catalog, &ident, "b0", snapshot_id, true).await;
    create_ref(&catalog, &ident, "t0", snapshot_id, false).await;
    add_z(&catalog, &ident).await;
    (catalog, ident, temp_dir, snapshot_id)
}

#[tokio::test]
async fn bs_add_ident_v2() {
    bs_add_ident(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_add_ident_v3() {
    bs_add_ident(FormatVersion::V3).await;
}

async fn bs_add_ident(version: FormatVersion) {
    let (catalog, ident, _tmp, _seed_snap) = branch_add_setup(version).await;
    let table = load(&catalog, &ident).await;
    let provider = provider_for_ref(&table, "b0").await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8", "z:Int32"]);
    assert_rows(rows, &["1|a|x|NULL", "2|b|y|NULL"]);
}

#[tokio::test]
async fn bs_add_version_v2() {
    bs_add_version(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_add_version_v3() {
    bs_add_version(FormatVersion::V3).await;
}

async fn bs_add_version(version: FormatVersion) {
    let (catalog, ident, _tmp, _seed_snap) = branch_add_setup(version).await;
    let table = load(&catalog, &ident).await;
    let provider = provider_for_ref(&table, "b0").await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8", "z:Int32"]);
    assert_rows(rows, &["1|a|x|NULL", "2|b|y|NULL"]);
}

#[tokio::test]
async fn bs_add_tag_v2() {
    bs_add_tag(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_add_tag_v3() {
    bs_add_tag(FormatVersion::V3).await;
}

async fn bs_add_tag(version: FormatVersion) {
    let (catalog, ident, _tmp, _seed_snap) = branch_add_setup(version).await;
    let table = load(&catalog, &ident).await;
    let provider = provider_for_ref(&table, "t0").await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8"]);
    assert_rows(rows, &["1|a|x", "2|b|y"]);
}

#[tokio::test]
async fn bs_add_snapid_v2() {
    bs_add_snapid(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_add_snapid_v3() {
    bs_add_snapid(FormatVersion::V3).await;
}

async fn bs_add_snapid(version: FormatVersion) {
    let (catalog, ident, _tmp, seed_snap) = branch_add_setup(version).await;
    let table = load(&catalog, &ident).await;
    let provider = provider_for_snapshot(&table, seed_snap).await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8"]);
    assert_rows(rows, &["1|a|x", "2|b|y"]);
}

#[tokio::test]
async fn bs_drop_v2() {
    bs_drop(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_drop_v3() {
    bs_drop(FormatVersion::V3).await;
}

async fn bs_drop(version: FormatVersion) {
    let (catalog, namespace, ident, _tmp) = setup_table(
        version,
        cell_schema(PrimitiveType::Long),
        HashMap::new(),
    )
    .await;
    seed_two(&catalog, &namespace, ident.name(), None).await;
    let table = load(&catalog, &ident).await;
    create_ref(&catalog, &ident, "b0", current_snapshot_id(&table), true).await;
    evolve_schema(&catalog, &ident, SchemaOp::Drop("data")).await;
    let table = load(&catalog, &ident).await;
    let provider = provider_for_ref(&table, "b0").await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "cat:Utf8"]);
    assert_rows(rows, &["1|x", "2|y"]);
}

#[tokio::test]
async fn bs_rename_v2() {
    bs_rename(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_rename_v3() {
    bs_rename(FormatVersion::V3).await;
}

async fn bs_rename(version: FormatVersion) {
    let (catalog, namespace, ident, _tmp) = setup_table(
        version,
        cell_schema(PrimitiveType::Long),
        HashMap::new(),
    )
    .await;
    seed_two(&catalog, &namespace, ident.name(), None).await;
    let table = load(&catalog, &ident).await;
    create_ref(&catalog, &ident, "b0", current_snapshot_id(&table), true).await;
    evolve_schema(&catalog, &ident, SchemaOp::Rename("data", "payload")).await;
    let table = load(&catalog, &ident).await;
    let provider = provider_for_ref(&table, "b0").await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "payload:Utf8", "cat:Utf8"]);
    assert_rows(rows, &["1|a|x", "2|b|y"]);
}

#[tokio::test]
async fn bs_widen_v2() {
    bs_widen(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_widen_v3() {
    bs_widen(FormatVersion::V3).await;
}

async fn bs_widen(version: FormatVersion) {
    let (catalog, namespace, ident, _tmp) = setup_table(
        version,
        cell_schema(PrimitiveType::Int),
        HashMap::new(),
    )
    .await;
    seed_two(&catalog, &namespace, ident.name(), None).await;
    let table = load(&catalog, &ident).await;
    create_ref(&catalog, &ident, "b0", current_snapshot_id(&table), true).await;
    evolve_schema(&catalog, &ident, SchemaOp::PromoteToLong("id")).await;
    let table = load(&catalog, &ident).await;
    let provider = provider_for_ref(&table, "b0").await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8"]);
    assert_rows(rows, &["1|a|x", "2|b|y"]);
}

#[tokio::test]
async fn bs_write_then_add_v2() {
    bs_write_then_add(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_write_then_add_v3() {
    bs_write_then_add(FormatVersion::V3).await;
}

async fn bs_write_then_add(version: FormatVersion) {
    let (catalog, namespace, ident, _tmp) = setup_table(
        version,
        cell_schema(PrimitiveType::Long),
        HashMap::new(),
    )
    .await;
    seed_two(&catalog, &namespace, ident.name(), None).await;
    let table = load(&catalog, &ident).await;
    create_ref(&catalog, &ident, "b0", current_snapshot_id(&table), true).await;
    let provider = IcebergTableProvider::try_new(
        catalog.clone(),
        namespace.clone(),
        ident.name().to_string(),
    )
    .await
    .expect("provider for the branch write")
    .with_commit_branch("b0");
    let batches = sql_exec(Arc::new(provider), "INSERT INTO t VALUES (5, 'e', 'z')").await;
    assert!(!batches.is_empty(), "the branch write must report a row count");
    add_z(&catalog, &ident).await;

    let table = load(&catalog, &ident).await;
    let provider = provider_for_ref(&table, "b0").await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8", "z:Int32"]);
    assert_rows(rows, &["1|a|x|NULL", "2|b|y|NULL", "5|e|z|NULL"]);

    let provider = IcebergStaticTableProvider::try_new_from_table(table)
        .await
        .expect("main provider");
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8", "z:Int32"]);
    assert_rows(rows, &["1|a|x|NULL", "2|b|y|NULL"]);
}

#[tokio::test]
async fn bs_add_then_write_branch_v2() {
    bs_add_then_write_branch(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_add_then_write_branch_v3() {
    bs_add_then_write_branch(FormatVersion::V3).await;
}

async fn bs_add_then_write_branch(version: FormatVersion) {
    let (catalog, namespace, ident, _tmp) = setup_table(
        version,
        cell_schema(PrimitiveType::Long),
        HashMap::new(),
    )
    .await;
    seed_two(&catalog, &namespace, ident.name(), None).await;
    let table = load(&catalog, &ident).await;
    create_ref(&catalog, &ident, "b0", current_snapshot_id(&table), true).await;
    add_z(&catalog, &ident).await;
    let provider = IcebergTableProvider::try_new(
        catalog.clone(),
        namespace.clone(),
        ident.name().to_string(),
    )
    .await
    .expect("provider for the branch write")
    .with_commit_branch("b0");
    let batches = sql_exec(Arc::new(provider), "INSERT INTO t VALUES (5, 'e', 'z', 9)").await;
    assert!(!batches.is_empty(), "the branch write must report a row count");

    let table = load(&catalog, &ident).await;
    let provider = provider_for_ref(&table, "b0").await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8", "z:Int32"]);
    assert_rows(rows, &["1|a|x|NULL", "2|b|y|NULL", "5|e|z|9"]);

    let provider = IcebergStaticTableProvider::try_new_from_table(table)
        .await
        .expect("main provider");
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8", "z:Int32"]);
    assert_rows(rows, &["1|a|x|NULL", "2|b|y|NULL"]);
}

#[tokio::test]
async fn bs_main_ident_v2() {
    bs_main_ident(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_main_ident_v3() {
    bs_main_ident(FormatVersion::V3).await;
}

async fn bs_main_ident(version: FormatVersion) {
    let (catalog, ident, _tmp, _seed_snap) = branch_add_setup(version).await;
    let table = load(&catalog, &ident).await;
    let provider = provider_for_ref(&table, "main").await;
    let (cols, rows) = run_query(Arc::new(provider), "SELECT * FROM t").await;
    assert_shape(&cols, &["id:Int64", "data:Utf8", "cat:Utf8", "z:Int32"]);
    assert_rows(rows, &["1|a|x|NULL", "2|b|y|NULL"]);
}

#[tokio::test]
async fn bs_where_newcol_v2() {
    bs_where_newcol(FormatVersion::V2).await;
}

#[tokio::test]
async fn bs_where_newcol_v3() {
    bs_where_newcol(FormatVersion::V3).await;
}

async fn bs_where_newcol(version: FormatVersion) {
    let (catalog, ident, _tmp, _seed_snap) = branch_add_setup(version).await;
    let table = load(&catalog, &ident).await;
    let provider = Arc::new(provider_for_ref(&table, "b0").await);

    let ctx = SessionContext::new();
    ctx.register_table("t", provider.clone() as Arc<dyn TableProvider>)
        .expect("register the branch provider");
    let plan = ctx
        .sql("SELECT id FROM t WHERE z IS NULL")
        .await
        .expect("plan the filtered read")
        .create_physical_plan()
        .await
        .expect("physical plan");
    let scan = find_scan(&plan).expect("an IcebergTableScan must be in the plan");
    let predicate = scan
        .predicates()
        .expect("the z IS NULL filter must be pushed into the scan");
    let shown = format!("{predicate}");
    assert!(
        shown.contains('z') && shown.contains("IS NULL"),
        "the pushed predicate must be the z null test: {shown}"
    );

    let (cols, rows) = run_query(provider, "SELECT id FROM t WHERE z IS NULL").await;
    assert_shape(&cols, &["id:Int64"]);
    assert_rows(rows, &["1", "2"]);
}
