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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, FixedSizeBinaryArray, Int32Array, RecordBatch, StringArray, StructArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::catalog::{CatalogProvider, SchemaProvider, TableProvider};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use futures::StreamExt;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, StructType, Type};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

use super::uuid_text::{
    collect_uuid_field_ids, is_canonical_uuid_text, parse_uuid_text, render_uuid_text,
};
use super::*;

const U1: &str = "123e4567-e89b-12d3-a456-426614174000";
const U1_UPPER: &str = "123E4567-E89B-12D3-A456-426614174000";
const U2: &str = "123e4567-e89b-12d3-a456-4266141740ff";

fn uuid_bytes(text: &str) -> [u8; 16] {
    parse_uuid_text(text).expect("test uuid literal parses")
}

async fn uuid_catalog_and_table() -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
    let temp_dir = TempDir::new().expect("temp warehouse");
    let warehouse_path = temp_dir.path().to_str().unwrap().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .expect("memory catalog loads");
    let namespace = NamespaceIdent::new("uuid_ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace creates");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Uuid)).into(),
            NestedField::optional(
                3,
                "s",
                Type::Struct(StructType::new(vec![Arc::new(NestedField::optional(
                    4,
                    "inner",
                    Type::Primitive(PrimitiveType::Uuid),
                ))])),
            )
            .into(),
        ])
        .build()
        .expect("uuid schema builds");
    let creation = TableCreation::builder()
        .name("uuid_table".to_string())
        .location(format!("{warehouse_path}/uuid_table"))
        .schema(schema)
        .properties(HashMap::new())
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("uuid table creates");
    (
        Arc::new(catalog),
        namespace,
        "uuid_table".to_string(),
        temp_dir,
    )
}

async fn uuid_provider(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    name: &str,
    as_string: bool,
) -> Arc<IcebergTableProvider> {
    Arc::new(
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), name.to_string())
            .await
            .expect("provider builds")
            .with_uuid_as_string(as_string),
    )
}

async fn run_sql(provider: Arc<dyn TableProvider>, sql: &str) -> Vec<RecordBatch> {
    let ctx = SessionContext::new();
    ctx.register_table("t", provider)
        .expect("register provider");
    ctx.sql(sql)
        .await
        .unwrap_or_else(|e| panic!("plan `{sql}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("execute `{sql}`: {e}"))
}

fn text_column(batches: &[RecordBatch], name: &str) -> Vec<Option<String>> {
    let mut out = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("batch carries `{name}`"));
        let strings = column
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("`{name}` renders as text"));
        for row in 0..strings.len() {
            out.push(if strings.is_null(row) {
                None
            } else {
                Some(strings.value(row).to_string())
            });
        }
    }
    out
}

fn int_column(batches: &[RecordBatch], name: &str) -> Vec<i32> {
    let mut out = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("batch carries `{name}`"));
        let ints = column
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("`{name}` stays Int32"));
        for row in 0..ints.len() {
            assert!(ints.is_valid(row), "id column holds no nulls");
            out.push(ints.value(row));
        }
    }
    out
}

#[test]
fn uuid_text_helpers_reject_non_hyphenated_and_bad_hex() {
    assert_eq!(render_uuid_text(&uuid_bytes(U1)), U1);
    assert_eq!(
        parse_uuid_text(U1_UPPER).expect("upper parses"),
        uuid_bytes(U1)
    );
    assert!(parse_uuid_text("123e4567e89b12d3a456426614174000").is_err());
    assert!(parse_uuid_text("not-a-uuid").is_err());
    assert!(parse_uuid_text("123e4567-e89b-12d3-a456-42661417400z").is_err());
    assert_eq!(
        parse_uuid_text("123e4567-e89b-12d3-a456-42661417400")
            .map(|bytes| render_uuid_text(&bytes)),
        Ok("123e4567-e89b-12d3-a456-042661417400".to_string())
    );
    assert!(is_canonical_uuid_text(U1));
    assert!(!is_canonical_uuid_text(U1_UPPER));
    assert!(!is_canonical_uuid_text("not-a-uuid"));
}

#[tokio::test]
async fn uuid_schema_advertises_text_only_with_option_on() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let off = uuid_provider(&catalog, &namespace, &name, false).await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    let off_u = off
        .schema()
        .field_with_name("u")
        .expect("uuid field")
        .data_type()
        .clone();
    assert_eq!(off_u, DataType::FixedSizeBinary(16));
    let on_u = on
        .schema()
        .field_with_name("u")
        .expect("uuid field")
        .data_type()
        .clone();
    assert_eq!(on_u, DataType::Utf8);
    let inner_type = |provider: &IcebergTableProvider| {
        let field = provider
            .schema()
            .field_with_name("s")
            .expect("struct field")
            .clone();
        let DataType::Struct(children) = field.data_type() else {
            panic!("s stays a struct");
        };
        children
            .iter()
            .find(|child| child.name() == "inner")
            .expect("inner field")
            .data_type()
            .clone()
    };
    assert_eq!(inner_type(&off), DataType::FixedSizeBinary(16));
    assert_eq!(inner_type(&on), DataType::Utf8);
    let table = catalog
        .load_table(&iceberg::TableIdent::new(namespace.clone(), name.clone()))
        .await
        .expect("table loads");
    let ids = collect_uuid_field_ids(table.metadata().current_schema());
    assert_eq!(ids, HashSet::from([2, 4]));
}

#[tokio::test]
async fn uuid_scan_renders_canonical_lowercase_text() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1_UPPER}', NULL), (2, '{U2}', NULL)"),
    )
    .await;
    let batches = run_sql(on.clone(), "SELECT id, u FROM t ORDER BY id").await;
    assert_eq!(int_column(&batches, "id"), vec![1, 2]);
    assert_eq!(text_column(&batches, "u"), vec![
        Some(U1.to_string()),
        Some(U2.to_string())
    ]);
}

#[tokio::test]
async fn uuid_bytes_written_directly_render_lowercase() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let off = uuid_provider(&catalog, &namespace, &name, false).await;
    let inner_fields = datafusion::arrow::datatypes::Fields::from(vec![Arc::new(Field::new(
        "inner",
        DataType::FixedSizeBinary(16),
        true,
    ))]);
    let byte_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("u", DataType::FixedSizeBinary(16), true),
        Field::new("s", DataType::Struct(inner_fields.clone()), true),
    ]));
    let id_array: ArrayRef = Arc::new(Int32Array::from(vec![7]));
    let u_array: ArrayRef = Arc::new(
        FixedSizeBinaryArray::try_from_iter(vec![uuid_bytes(U1_UPPER).as_slice()].into_iter())
            .expect("byte uuid builds"),
    );
    let inner_array: ArrayRef = Arc::new(
        FixedSizeBinaryArray::try_from_iter(vec![uuid_bytes(U2).as_slice()].into_iter())
            .expect("byte inner builds"),
    );
    let struct_array: ArrayRef = Arc::new(
        StructArray::try_new(inner_fields, vec![inner_array], None).expect("byte struct builds"),
    );
    let batch = RecordBatch::try_new(byte_schema.clone(), vec![id_array, u_array, struct_array])
        .expect("byte batch builds");
    let ctx = SessionContext::new();
    let input = MemorySourceConfig::try_new_exec(&[vec![batch]], byte_schema, None)
        .expect("memory input builds");
    let plan = off
        .insert_into(&ctx.state(), input, InsertOp::Append)
        .await
        .expect("byte insert plans");
    let written = plan
        .execute(0, ctx.task_ctx())
        .expect("byte insert executes")
        .collect::<Vec<_>>()
        .await;
    assert!(!written.is_empty(), "the write reports its files");
    for batch in written {
        batch.expect("byte insert runs");
    }
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    let batches = run_sql(on.clone(), "SELECT id, u, s FROM t").await;
    assert_eq!(int_column(&batches, "id"), vec![7]);
    assert_eq!(text_column(&batches, "u"), vec![Some(U1.to_string())]);
    let structs = batches[0]
        .column_by_name("s")
        .expect("struct scans")
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("s stays a struct");
    let inner = structs
        .column_by_name("inner")
        .expect("inner scans")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("inner renders as text");
    assert_eq!(inner.value(0), U2);
}

#[tokio::test]
async fn uuid_insert_uppercases_and_refuses_invalid() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1_UPPER}', NULL)"),
    )
    .await;
    let off = uuid_provider(&catalog, &namespace, &name, false).await;
    let batches = run_sql(off.clone(), "SELECT id, u FROM t").await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
    let column = batches[0].column_by_name("u").expect("byte scan carries u");
    let bytes = column
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("option off still scans bytes");
    assert!(!bytes.is_null(0));
    assert_eq!(bytes.value(0), uuid_bytes(U1));

    let ctx = SessionContext::new();
    ctx.register_table("t", on.clone())
        .expect("register provider");
    let err = ctx
        .sql("INSERT INTO t VALUES (2, 'not-a-uuid', NULL)")
        .await
        .expect("invalid insert plans")
        .collect()
        .await
        .expect_err("invalid uuid string refuses the write");
    assert!(
        format!("{err}").contains("Invalid UUID string: not-a-uuid"),
        "refusal names the value, got {err}"
    );
    let batches = run_sql(on.clone(), "SELECT id FROM t ORDER BY id").await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
}

#[tokio::test]
async fn uuid_string_predicate_pushes_down_to_same_files() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', NULL)"),
    )
    .await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (2, '{U2}', NULL)"),
    )
    .await;

    let ctx = SessionContext::new();
    let filter = Expr::BinaryExpr(BinaryExpr::new(
        Box::new(Expr::Column(datafusion::common::Column::from_name("u"))),
        Operator::Eq,
        Box::new(Expr::Literal(ScalarValue::Utf8(Some(U1.to_string())), None)),
    ));
    let plan = on
        .scan(&ctx.state(), None, std::slice::from_ref(&filter), None)
        .await
        .expect("filtered scan plans");
    let scan = plan
        .downcast_ref::<crate::physical_plan::IcebergTableScan>()
        .expect("provider scan is an IcebergTableScan");
    assert!(
        scan.predicates().is_some(),
        "the text predicate reaches the file prune"
    );
    let filtered_files: HashSet<String> = scan
        .partition_work()
        .iter()
        .flat_map(|work| work.tasks())
        .map(|task| task.data_file_path().to_string())
        .collect();
    let unfiltered = on
        .scan(&ctx.state(), None, &[], None)
        .await
        .expect("unfiltered scan plans");
    let unfiltered_scan = unfiltered
        .downcast_ref::<crate::physical_plan::IcebergTableScan>()
        .expect("provider scan is an IcebergTableScan");
    let all_files: HashSet<String> = unfiltered_scan
        .partition_work()
        .iter()
        .flat_map(|work| work.tasks())
        .map(|task| task.data_file_path().to_string())
        .collect();
    assert_eq!(all_files.len(), 2);
    assert_eq!(filtered_files.len(), 1);
    assert!(filtered_files.iter().all(|path| all_files.contains(path)));

    let batches = run_sql(on.clone(), &format!("SELECT id FROM t WHERE u = '{U1}'")).await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
    let batches = run_sql(
        on.clone(),
        &format!("SELECT id FROM t WHERE u <> '{U1}' ORDER BY id"),
    )
    .await;
    assert_eq!(int_column(&batches, "id"), vec![2]);
    let batches = run_sql(
        on.clone(),
        &format!("SELECT id FROM t WHERE u IN ('{U1}', '{U2}') ORDER BY id"),
    )
    .await;
    assert_eq!(int_column(&batches, "id"), vec![1, 2]);
    let batches = run_sql(
        on.clone(),
        &format!("SELECT id FROM t WHERE u < '{U2}' ORDER BY id"),
    )
    .await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
    let batches = run_sql(
        on.clone(),
        &format!("SELECT id FROM t WHERE u = '{U1_UPPER}'"),
    )
    .await;
    assert!(
        int_column(&batches, "id").is_empty(),
        "text comparison stays case-sensitive like Spark"
    );
    let batches = run_sql(on.clone(), "SELECT id FROM t WHERE u IS NULL").await;
    assert!(int_column(&batches, "id").is_empty());
}

#[tokio::test]
async fn uuid_nested_struct_round_trips_as_text() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', named_struct('inner', '{U1_UPPER}'))"),
    )
    .await;
    let batches = run_sql(on.clone(), "SELECT id, u, s FROM t").await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
    assert_eq!(text_column(&batches, "u"), vec![Some(U1.to_string())]);
    let column = batches[0].column_by_name("s").expect("struct scans");
    let structs = column
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("s stays a struct");
    let inner = structs
        .column_by_name("inner")
        .expect("inner scans")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("inner renders as text");
    assert!(!inner.is_null(0));
    assert_eq!(inner.value(0), U1);
}

#[tokio::test]
async fn uuid_row_level_delete_and_update_use_text() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', NULL), (2, '{U2}', NULL)"),
    )
    .await;
    run_sql(on.clone(), &format!("DELETE FROM t WHERE u = '{U1}'")).await;
    let batches = run_sql(on.clone(), "SELECT id, u FROM t").await;
    assert_eq!(int_column(&batches, "id"), vec![2]);
    assert_eq!(text_column(&batches, "u"), vec![Some(U2.to_string())]);
    run_sql(
        on.clone(),
        &format!("UPDATE t SET u = '{U1_UPPER}' WHERE id = 2"),
    )
    .await;
    let batches = run_sql(on.clone(), "SELECT id, u FROM t").await;
    assert_eq!(int_column(&batches, "id"), vec![2]);
    assert_eq!(text_column(&batches, "u"), vec![Some(U1.to_string())]);
}

#[tokio::test]
async fn uuid_option_off_keeps_byte_path() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', NULL)"),
    )
    .await;
    let off = uuid_provider(&catalog, &namespace, &name, false).await;
    let batches = run_sql(off.clone(), "SELECT id, u FROM t").await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
    let column = batches[0].column_by_name("u").expect("u scans");
    assert_eq!(*column.data_type(), DataType::FixedSizeBinary(16));
    let bytes = column
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("bytes scan");
    assert_eq!(bytes.value(0), uuid_bytes(U1));
    let refreshed = off.refreshed().await.expect("refresh keeps working");
    assert!(!refreshed.uuid_as_string);
    let refreshed_on = on.refreshed().await.expect("refresh keeps working");
    assert!(refreshed_on.uuid_as_string);
}

async fn current_snapshot_id(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    name: &str,
) -> i64 {
    catalog
        .load_table(&TableIdent::new(namespace.clone(), name.to_string()))
        .await
        .expect("table loads")
        .metadata()
        .current_snapshot()
        .expect("a snapshot exists")
        .snapshot_id()
}

async fn uuid_static_provider(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    name: &str,
    snapshot_id: i64,
    as_string: bool,
) -> Arc<IcebergStaticTableProvider> {
    let table = catalog
        .load_table(&TableIdent::new(namespace.clone(), name.to_string()))
        .await
        .expect("table loads");
    Arc::new(
        IcebergStaticTableProvider::try_new_from_table_snapshot(table, snapshot_id)
            .await
            .expect("static provider builds")
            .with_uuid_as_string(as_string),
    )
}

async fn pruned_files(provider: Arc<dyn TableProvider>, filter: Option<Expr>) -> HashSet<String> {
    let ctx = SessionContext::new();
    let filters: Vec<Expr> = filter.into_iter().collect();
    let plan = provider
        .scan(&ctx.state(), None, &filters, None)
        .await
        .expect("scan plans");
    let scan = plan
        .downcast_ref::<crate::physical_plan::IcebergTableScan>()
        .expect("provider scan is an IcebergTableScan");
    scan.partition_work()
        .iter()
        .flat_map(|work| work.tasks())
        .map(|task| task.data_file_path().to_string())
        .collect()
}

fn uuid_eq_filter(literal: ScalarValue) -> Expr {
    Expr::BinaryExpr(BinaryExpr::new(
        Box::new(Expr::Column(datafusion::common::Column::from_name("u"))),
        Operator::Eq,
        Box::new(Expr::Literal(literal, None)),
    ))
}

#[tokio::test]
async fn uuid_static_provider_advertises_text_and_renders_pinned_rows() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', NULL)"),
    )
    .await;
    let first = current_snapshot_id(&catalog, &namespace, &name).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (2, '{U2}', NULL)"),
    )
    .await;
    let pinned = uuid_static_provider(&catalog, &namespace, &name, first, true).await;
    let text_type = pinned
        .schema()
        .field_with_name("u")
        .expect("uuid field")
        .data_type()
        .clone();
    assert_eq!(text_type, DataType::Utf8);
    let batches = run_sql(pinned.clone(), "SELECT id, u FROM t ORDER BY id").await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
    assert_eq!(text_column(&batches, "u"), vec![Some(U1.to_string())]);
    let off = uuid_static_provider(&catalog, &namespace, &name, first, false).await;
    let byte_type = off
        .schema()
        .field_with_name("u")
        .expect("uuid field")
        .data_type()
        .clone();
    assert_eq!(byte_type, DataType::FixedSizeBinary(16));
}

#[tokio::test]
async fn uuid_static_provider_pushes_text_predicate_to_same_files() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', NULL)"),
    )
    .await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (2, '{U2}', NULL)"),
    )
    .await;
    let latest = current_snapshot_id(&catalog, &namespace, &name).await;
    let pinned = uuid_static_provider(&catalog, &namespace, &name, latest, true).await;
    let text_files = pruned_files(
        pinned.clone(),
        Some(uuid_eq_filter(ScalarValue::Utf8(Some(U1.to_string())))),
    )
    .await;
    let off = uuid_static_provider(&catalog, &namespace, &name, latest, false).await;
    let byte_files = pruned_files(
        off.clone(),
        Some(uuid_eq_filter(ScalarValue::FixedSizeBinary(
            16,
            Some(uuid_bytes(U1).to_vec()),
        ))),
    )
    .await;
    let all_files = pruned_files(pinned.clone(), None).await;
    assert_eq!(all_files.len(), 2);
    assert_eq!(text_files.len(), 1);
    assert_eq!(text_files, byte_files);
    let batches = run_sql(
        pinned.clone(),
        &format!("SELECT id FROM t WHERE u = '{U1}'"),
    )
    .await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
}

#[tokio::test]
async fn uuid_static_option_off_keeps_byte_path() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', NULL)"),
    )
    .await;
    let latest = current_snapshot_id(&catalog, &namespace, &name).await;
    let off = uuid_static_provider(&catalog, &namespace, &name, latest, false).await;
    let batches = run_sql(off.clone(), "SELECT id, u FROM t").await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
    let column = batches[0].column_by_name("u").expect("u scans");
    assert_eq!(*column.data_type(), DataType::FixedSizeBinary(16));
    let bytes = column
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("bytes scan");
    assert_eq!(bytes.value(0), uuid_bytes(U1));
}

#[tokio::test]
async fn uuid_catalog_provider_propagates_option_to_resolved_tables() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let on = uuid_provider(&catalog, &namespace, &name, true).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', NULL)"),
    )
    .await;
    let catalog_provider = crate::IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("catalog provider builds")
        .with_uuid_as_string(true);
    let schema_provider =
        CatalogProvider::schema(&catalog_provider, "uuid_ns").expect("namespace resolves");
    let resolved = schema_provider
        .table(&name)
        .await
        .expect("table resolves")
        .expect("table listed");
    let text_type = resolved
        .schema()
        .field_with_name("u")
        .expect("uuid field")
        .data_type()
        .clone();
    assert_eq!(text_type, DataType::Utf8);
    let metadata = schema_provider
        .table("uuid_table$files")
        .await
        .expect("metadata lookup runs");
    assert!(metadata.is_some(), "metadata tables still resolve");
    let off_catalog = crate::IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("catalog provider builds");
    let off_schema = CatalogProvider::schema(&off_catalog, "uuid_ns").expect("namespace resolves");
    let off_table = off_schema
        .table(&name)
        .await
        .expect("table resolves")
        .expect("table listed");
    let byte_type = off_table
        .schema()
        .field_with_name("u")
        .expect("uuid field")
        .data_type()
        .clone();
    assert_eq!(byte_type, DataType::FixedSizeBinary(16));
}

#[tokio::test]
async fn uuid_schema_provider_flags_resolved_tables() {
    let (catalog, namespace, name, _temp) = uuid_catalog_and_table().await;
    let schema_provider =
        crate::schema::IcebergSchemaProvider::try_new(catalog.clone(), namespace.clone())
            .await
            .expect("schema provider builds");
    schema_provider.with_uuid_as_string(true);
    let resolved = schema_provider
        .table(&name)
        .await
        .expect("table resolves")
        .expect("table listed");
    let text_type = resolved
        .schema()
        .field_with_name("u")
        .expect("uuid field")
        .data_type()
        .clone();
    assert_eq!(text_type, DataType::Utf8);
}
