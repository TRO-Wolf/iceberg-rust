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
    Array, FixedSizeBinaryArray, Int32Array, ListArray, MapArray, RecordBatch, StringArray,
    StructArray,
};
use datafusion::arrow::datatypes::{DataType, Schema as ArrowSchema};
use datafusion::catalog::{CatalogProvider, TableProvider};
use datafusion::logical_expr::{Expr, col, lit};
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{ListType, MapType, NestedField, PrimitiveType, Schema, StructType, Type};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

use super::uuid_text::parse_uuid_text;
use super::{IcebergStaticTableProvider, IcebergTableProvider};

const U1: &str = "123e4567-e89b-12d3-a456-426614174000";
const U1_UPPER: &str = "123E4567-E89B-12D3-A456-426614174000";
const U2: &str = "123e4567-e89b-12d3-a456-4266141740ff";
const U2_UPPER: &str = "123E4567-E89B-12D3-A456-4266141740FF";

fn uuid_bytes(text: &str) -> [u8; 16] {
    parse_uuid_text(text).expect("test uuid literal parses")
}

fn byte_literal(text: &str) -> ScalarValue {
    ScalarValue::FixedSizeBinary(16, Some(uuid_bytes(text).to_vec()))
}

async fn nested_catalog_and_table() -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
    let temp_dir = TempDir::new().expect("temp warehouse");
    let warehouse_path = temp_dir.path().to_str().unwrap().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
        )
        .await
        .expect("memory catalog loads");
    let namespace = NamespaceIdent::new("uuid_nested_ns".to_string());
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
            NestedField::optional(
                5,
                "l",
                Type::List(ListType::new(Arc::new(NestedField::list_element(
                    6,
                    Type::Primitive(PrimitiveType::Uuid),
                    false,
                )))),
            )
            .into(),
            NestedField::optional(
                7,
                "m",
                Type::Map(MapType::new(
                    Arc::new(NestedField::map_key_element(
                        8,
                        Type::Primitive(PrimitiveType::String),
                    )),
                    Arc::new(NestedField::map_value_element(
                        9,
                        Type::Primitive(PrimitiveType::Uuid),
                        false,
                    )),
                )),
            )
            .into(),
        ])
        .build()
        .expect("nested uuid schema builds");
    let creation = TableCreation::builder()
        .name("uuid_nested".to_string())
        .location(format!("{warehouse_path}/uuid_nested"))
        .schema(schema)
        .properties(HashMap::new())
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("nested uuid table creates");
    (
        Arc::new(catalog),
        namespace,
        "uuid_nested".to_string(),
        temp_dir,
    )
}

async fn text_table_provider(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    name: &str,
) -> Arc<IcebergTableProvider> {
    Arc::new(
        IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), name.to_string())
            .await
            .expect("provider builds")
            .with_uuid_as_string(true),
    )
}

async fn static_provider(
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

fn struct_inner_type(schema: &ArrowSchema, name: &str, child: &str) -> DataType {
    let field = schema.field_with_name(name).expect("struct field");
    let DataType::Struct(children) = field.data_type() else {
        panic!("{name} stays a struct");
    };
    children
        .iter()
        .find(|entry| entry.name() == child)
        .expect("nested uuid field")
        .data_type()
        .clone()
}

fn list_element_type(schema: &ArrowSchema, name: &str) -> DataType {
    let field = schema.field_with_name(name).expect("list field");
    let DataType::List(element) = field.data_type() else {
        panic!("{name} stays a list");
    };
    element.data_type().clone()
}

fn map_entry_types(schema: &ArrowSchema, name: &str) -> (DataType, DataType) {
    let field = schema.field_with_name(name).expect("map field");
    let DataType::Map(entries, _) = field.data_type() else {
        panic!("{name} stays a map");
    };
    let DataType::Struct(pair) = entries.data_type() else {
        panic!("map entries stay a key-value struct");
    };
    let mut entry_types = pair.iter().map(|entry| entry.data_type().clone());
    let key = entry_types.next().expect("map key");
    let value = entry_types.next().expect("map value");
    assert!(
        entry_types.next().is_none(),
        "map entries hold key and value only"
    );
    (key, value)
}

#[tokio::test]
async fn uuid_static_nested_uuid_advertises_and_renders_text() {
    let (catalog, namespace, name, _temp) = nested_catalog_and_table().await;
    let on = text_table_provider(&catalog, &namespace, &name).await;
    run_sql(
        on.clone(),
        &format!(
            "INSERT INTO t VALUES (1, '{U1_UPPER}', named_struct('inner', '{U2_UPPER}'), make_array('{U1_UPPER}', '{U2}'), MAP {{'k': '{U2_UPPER}'}})"
        ),
    )
    .await;
    let snapshot = current_snapshot_id(&catalog, &namespace, &name).await;
    let pinned = static_provider(&catalog, &namespace, &name, snapshot, true).await;
    let schema = pinned.schema();
    assert_eq!(
        schema.field_with_name("u").expect("uuid field").data_type(),
        &DataType::Utf8
    );
    assert_eq!(struct_inner_type(&schema, "s", "inner"), DataType::Utf8);
    assert_eq!(list_element_type(&schema, "l"), DataType::Utf8);
    let (key_type, value_type) = map_entry_types(&schema, "m");
    assert_eq!(key_type, DataType::Utf8);
    assert_eq!(value_type, DataType::Utf8);
    let batches = run_sql(pinned.clone(), "SELECT id, u, s, l, m FROM t").await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
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
    assert!(!inner.is_null(0));
    assert_eq!(inner.value(0), U2);
    let list = batches[0]
        .column_by_name("l")
        .expect("list scans")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("l stays a list");
    let elements = list.value(0);
    let elements = elements
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("list elements render as text");
    assert_eq!(elements.len(), 2);
    assert_eq!(elements.value(0), U1);
    assert_eq!(elements.value(1), U2);
    let map = batches[0]
        .column_by_name("m")
        .expect("map scans")
        .as_any()
        .downcast_ref::<MapArray>()
        .expect("m stays a map");
    let keys = map
        .keys()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("map keys scan as text");
    let values = map
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("map values render as text");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys.value(0), "k");
    assert_eq!(values.len(), 1);
    assert_eq!(values.value(0), U2);
}

#[tokio::test]
async fn uuid_static_text_predicates_scan_same_files_as_bytes() {
    let (catalog, namespace, name, _temp) = nested_catalog_and_table().await;
    let on = text_table_provider(&catalog, &namespace, &name).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', NULL, NULL, NULL)"),
    )
    .await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (2, '{U2}', NULL, NULL, NULL)"),
    )
    .await;
    run_sql(
        on.clone(),
        "INSERT INTO t VALUES (3, NULL, NULL, NULL, NULL)",
    )
    .await;
    let snapshot = current_snapshot_id(&catalog, &namespace, &name).await;
    let text = static_provider(&catalog, &namespace, &name, snapshot, true).await;
    let bytes = static_provider(&catalog, &namespace, &name, snapshot, false).await;
    let all = pruned_files(text.clone(), None).await;
    assert_eq!(all.len(), 3);
    let text_eq = pruned_files(text.clone(), Some(col("u").eq(lit(U1)))).await;
    let byte_eq = pruned_files(bytes.clone(), Some(col("u").eq(lit(byte_literal(U1))))).await;
    assert_eq!(text_eq.len(), 1);
    assert_eq!(text_eq, byte_eq);
    let text_in = pruned_files(
        text.clone(),
        Some(col("u").in_list(vec![lit(U1), lit(U2)], false)),
    )
    .await;
    let byte_in = pruned_files(
        bytes.clone(),
        Some(col("u").in_list(vec![lit(byte_literal(U1)), lit(byte_literal(U2))], false)),
    )
    .await;
    assert_eq!(text_in.len(), 2);
    assert_eq!(text_in, byte_in);
    let text_null = pruned_files(text.clone(), Some(col("u").is_null())).await;
    let byte_null = pruned_files(bytes.clone(), Some(col("u").is_null())).await;
    assert_eq!(text_null.len(), 1);
    assert_eq!(text_null, byte_null);
    let batches = run_sql(
        text.clone(),
        &format!("SELECT id FROM t WHERE u IN ('{U1}', '{U2}') ORDER BY id"),
    )
    .await;
    assert_eq!(int_column(&batches, "id"), vec![1, 2]);
    let batches = run_sql(text.clone(), "SELECT id FROM t WHERE u IS NULL").await;
    assert_eq!(int_column(&batches, "id"), vec![3]);
}

#[tokio::test]
async fn uuid_static_option_off_advertises_and_returns_bytes() {
    let (catalog, namespace, name, _temp) = nested_catalog_and_table().await;
    let on = text_table_provider(&catalog, &namespace, &name).await;
    run_sql(
        on.clone(),
        &format!(
            "INSERT INTO t VALUES (1, '{U1_UPPER}', named_struct('inner', '{U2_UPPER}'), make_array('{U1_UPPER}', '{U2}'), MAP {{'k': '{U2_UPPER}'}})"
        ),
    )
    .await;
    let snapshot = current_snapshot_id(&catalog, &namespace, &name).await;
    let off = static_provider(&catalog, &namespace, &name, snapshot, false).await;
    let schema = off.schema();
    assert_eq!(
        schema.field_with_name("u").expect("uuid field").data_type(),
        &DataType::FixedSizeBinary(16)
    );
    assert_eq!(
        struct_inner_type(&schema, "s", "inner"),
        DataType::FixedSizeBinary(16)
    );
    assert_eq!(
        list_element_type(&schema, "l"),
        DataType::FixedSizeBinary(16)
    );
    let (key_type, value_type) = map_entry_types(&schema, "m");
    assert_eq!(key_type, DataType::Utf8);
    assert_eq!(value_type, DataType::FixedSizeBinary(16));
    let batches = run_sql(off.clone(), "SELECT id, u, s, l, m FROM t").await;
    assert_eq!(int_column(&batches, "id"), vec![1]);
    let column = batches[0].column_by_name("u").expect("u scans");
    assert_eq!(*column.data_type(), DataType::FixedSizeBinary(16));
    let scanned = column
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("bytes scan");
    assert_eq!(scanned.value(0), uuid_bytes(U1));
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
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("inner scans as bytes");
    assert_eq!(inner.value(0), uuid_bytes(U2));
    let list = batches[0]
        .column_by_name("l")
        .expect("list scans")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("l stays a list");
    let elements = list.value(0);
    let elements = elements
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("list elements scan as bytes");
    assert_eq!(elements.len(), 2);
    assert_eq!(elements.value(0), uuid_bytes(U1));
    assert_eq!(elements.value(1), uuid_bytes(U2));
    let map = batches[0]
        .column_by_name("m")
        .expect("map scans")
        .as_any()
        .downcast_ref::<MapArray>()
        .expect("m stays a map");
    let keys = map
        .keys()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("map keys scan as text");
    let values = map
        .values()
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("map values scan as bytes");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys.value(0), "k");
    assert_eq!(values.len(), 1);
    assert_eq!(values.value(0), uuid_bytes(U2));
}

#[tokio::test]
async fn uuid_catalog_schema_table_and_deregister_keep_text_flag() {
    let (catalog, namespace, name, _temp) = nested_catalog_and_table().await;
    let on = text_table_provider(&catalog, &namespace, &name).await;
    run_sql(
        on.clone(),
        &format!("INSERT INTO t VALUES (1, '{U1}', NULL, NULL, NULL)"),
    )
    .await;
    let catalog_provider = crate::IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("catalog provider builds")
        .with_uuid_as_string(true);
    let schema_provider =
        CatalogProvider::schema(&catalog_provider, "uuid_nested_ns").expect("namespace resolves");
    let resolved = schema_provider
        .table(&name)
        .await
        .expect("table resolves")
        .expect("table listed");
    assert_eq!(
        resolved
            .schema()
            .field_with_name("u")
            .expect("uuid field")
            .data_type(),
        &DataType::Utf8
    );
    let metadata_name = format!("{name}$files");
    let metadata = schema_provider
        .table(&metadata_name)
        .await
        .expect("metadata lookup runs")
        .expect("metadata tables still resolve");
    let off_catalog = crate::IcebergCatalogProvider::try_new(catalog.clone())
        .await
        .expect("catalog provider builds");
    let off_schema =
        CatalogProvider::schema(&off_catalog, "uuid_nested_ns").expect("namespace resolves");
    let off_metadata = off_schema
        .table(&metadata_name)
        .await
        .expect("metadata lookup runs")
        .expect("metadata tables still resolve");
    assert_eq!(metadata.schema(), off_metadata.schema());
    let dropped = schema_provider
        .deregister_table(&name)
        .expect("deregister runs")
        .expect("dropped provider returns");
    assert_eq!(
        dropped
            .schema()
            .field_with_name("u")
            .expect("uuid field")
            .data_type(),
        &DataType::Utf8
    );
    assert!(!schema_provider.table_exist(&name));
}
