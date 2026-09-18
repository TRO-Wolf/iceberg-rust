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
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use datafusion::arrow::util::display::array_value_to_string;
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{ListType, MapType, NestedField, PrimitiveType, Schema, StructType, Type};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::{IcebergCatalogProvider, IcebergStaticTableProvider};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use tempfile::TempDir;

async fn run(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan {sql}: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("run {sql}: {error}"));
}

fn string_field(id: i32, name: &str) -> Arc<NestedField> {
    NestedField::optional(id, name, Type::Primitive(PrimitiveType::String)).into()
}

fn long_id_field() -> Arc<NestedField> {
    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into()
}

async fn catalog_ctx(
    mode: &str,
    fields: Vec<Arc<NestedField>>,
) -> (SessionContext, TempDir, Arc<dyn Catalog>, NamespaceIdent) {
    let warehouse = TempDir::new().expect("warehouse");
    let path = warehouse.path().to_str().expect("utf-8 path").to_string();
    let iceberg_catalog: Arc<dyn Catalog> = Arc::new(
        MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), path)]),
            )
            .await
            .expect("memory catalog"),
    );
    let namespace = NamespaceIdent::new("ns".to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(fields)
        .build()
        .expect("evo schema");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), mode.to_string()),
            ("write.update.mode".to_string(), mode.to_string()),
        ]))
        .schema(schema)
        .build();
    iceberg_catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");
    let provider =
        IcebergCatalogProvider::try_new(Arc::clone(&iceberg_catalog) as Arc<dyn Catalog>)
            .await
            .expect("catalog provider");
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", Arc::new(provider));
    (ctx, warehouse, iceberg_catalog, namespace)
}

async fn base_table(
    mode: &str,
    fields: Vec<Arc<NestedField>>,
    first: &str,
    second: &str,
) -> (SessionContext, TempDir, Arc<dyn Catalog>, NamespaceIdent) {
    let (ctx, warehouse, catalog, namespace) = catalog_ctx(mode, fields).await;
    run(&ctx, &format!("INSERT INTO catalog.ns.t VALUES {first}")).await;
    run(&ctx, &format!("INSERT INTO catalog.ns.t VALUES {second}")).await;
    (ctx, warehouse, catalog, namespace)
}

async fn load(namespace: &NamespaceIdent, catalog: &Arc<dyn Catalog>) -> iceberg::table::Table {
    catalog
        .load_table(&TableIdent::new(namespace.clone(), "t".to_string()))
        .await
        .expect("load table")
}

async fn added_column_table(
    mode: &str,
) -> (SessionContext, TempDir, Arc<dyn Catalog>, NamespaceIdent) {
    let (ctx, warehouse, catalog, namespace) = base_table(
        mode,
        vec![long_id_field(), string_field(2, "v")],
        "(1, 'a')",
        "(2, 'b')",
    )
    .await;
    let table = load(&namespace, &catalog).await;
    let tx = Transaction::new(&table);
    tx.update_schema()
        .add_column("extra", Type::Primitive(PrimitiveType::String))
        .apply(tx)
        .expect("apply add column")
        .commit(catalog.as_ref())
        .await
        .expect("commit add column");
    (ctx, warehouse, catalog, namespace)
}

async fn renamed_column_table(
    mode: &str,
) -> (SessionContext, TempDir, Arc<dyn Catalog>, NamespaceIdent) {
    let (ctx, warehouse, catalog, namespace) = base_table(
        mode,
        vec![long_id_field(), string_field(2, "w")],
        "(1, 'a')",
        "(2, 'b')",
    )
    .await;
    let table = load(&namespace, &catalog).await;
    let tx = Transaction::new(&table);
    tx.update_schema()
        .rename_column("w", "v")
        .apply(tx)
        .expect("apply rename")
        .commit(catalog.as_ref())
        .await
        .expect("commit rename");
    (ctx, warehouse, catalog, namespace)
}

async fn swapped_names_table(
    mode: &str,
) -> (SessionContext, TempDir, Arc<dyn Catalog>, NamespaceIdent) {
    let (ctx, warehouse, catalog, namespace) = base_table(
        mode,
        vec![
            long_id_field(),
            string_field(2, "v"),
            string_field(3, "extra"),
        ],
        "(1, 'a', 'e1')",
        "(2, 'b', 'e2')",
    )
    .await;
    let table = load(&namespace, &catalog).await;
    let tx = Transaction::new(&table);
    tx.update_schema()
        .rename_column("v", "tmp")
        .apply(tx)
        .expect("apply first rename")
        .commit(catalog.as_ref())
        .await
        .expect("commit first rename");
    let table = load(&namespace, &catalog).await;
    let tx = Transaction::new(&table);
    tx.update_schema()
        .rename_column("extra", "v")
        .apply(tx)
        .expect("apply second rename")
        .commit(catalog.as_ref())
        .await
        .expect("commit second rename");
    let table = load(&namespace, &catalog).await;
    let tx = Transaction::new(&table);
    tx.update_schema()
        .rename_column("tmp", "extra")
        .apply(tx)
        .expect("apply swap")
        .commit(catalog.as_ref())
        .await
        .expect("commit swap");
    (ctx, warehouse, catalog, namespace)
}

async fn select_all(ctx: &SessionContext, sql: &str, columns: usize) -> Vec<Vec<String>> {
    let batches = ctx
        .sql(sql)
        .await
        .expect("plan select")
        .collect()
        .await
        .expect("run select");
    let mut out = vec![];
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut rendered = vec![];
            for column in 0..columns {
                if batch.column(column).is_null(row) {
                    rendered.push("NULL".to_string());
                } else {
                    rendered
                        .push(array_value_to_string(batch.column(column), row).expect("renders"));
                }
            }
            out.push(rendered);
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn copy_on_write_update_after_add_column_sets_the_added_column() {
    let (ctx, _warehouse, _, _) = added_column_table("copy-on-write").await;
    run(&ctx, "UPDATE catalog.ns.t SET extra = 'x' WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![
            vec!["1".to_string(), "a".to_string(), "x".to_string()],
            vec!["2".to_string(), "b".to_string(), "NULL".to_string()],
        ]
    );
}

#[tokio::test]
async fn merge_on_read_update_after_add_column_sets_the_added_column() {
    let (ctx, _warehouse, _, _) = added_column_table("merge-on-read").await;
    run(&ctx, "UPDATE catalog.ns.t SET extra = 'x' WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![
            vec!["1".to_string(), "a".to_string(), "x".to_string()],
            vec!["2".to_string(), "b".to_string(), "NULL".to_string()],
        ]
    );
}

#[tokio::test]
async fn copy_on_write_delete_after_add_column_removes_the_matching_row() {
    let (ctx, _warehouse, _, _) = added_column_table("copy-on-write").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![vec!["2".to_string(), "b".to_string(), "NULL".to_string()],]
    );
}

#[tokio::test]
async fn merge_on_read_delete_after_add_column_removes_the_matching_row() {
    let (ctx, _warehouse, _, _) = added_column_table("merge-on-read").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![vec!["2".to_string(), "b".to_string(), "NULL".to_string()],]
    );
}

#[tokio::test]
async fn copy_on_write_update_after_rename_sets_the_renamed_column() {
    let (ctx, _warehouse, _, _) = renamed_column_table("copy-on-write").await;
    run(&ctx, "UPDATE catalog.ns.t SET v = 'x' WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v FROM catalog.ns.t", 2).await,
        vec![vec!["1".to_string(), "x".to_string()], vec![
            "2".to_string(),
            "b".to_string()
        ],]
    );
}

#[tokio::test]
async fn merge_on_read_update_after_rename_sets_the_renamed_column() {
    let (ctx, _warehouse, _, _) = renamed_column_table("merge-on-read").await;
    run(&ctx, "UPDATE catalog.ns.t SET v = 'x' WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v FROM catalog.ns.t", 2).await,
        vec![vec!["1".to_string(), "x".to_string()], vec![
            "2".to_string(),
            "b".to_string()
        ],]
    );
}

#[tokio::test]
async fn copy_on_write_delete_after_rename_removes_the_matching_row() {
    let (ctx, _warehouse, _, _) = renamed_column_table("copy-on-write").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v FROM catalog.ns.t", 2).await,
        vec![vec!["2".to_string(), "b".to_string()]]
    );
}

#[tokio::test]
async fn merge_on_read_delete_after_rename_removes_the_matching_row() {
    let (ctx, _warehouse, _, _) = renamed_column_table("merge-on-read").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v FROM catalog.ns.t", 2).await,
        vec![vec!["2".to_string(), "b".to_string()]]
    );
}

#[tokio::test]
async fn copy_on_write_update_after_a_name_swap_sets_the_own_field() {
    let (ctx, _warehouse, _, _) = swapped_names_table("copy-on-write").await;
    run(&ctx, "UPDATE catalog.ns.t SET v = 'x' WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![
            vec!["1".to_string(), "x".to_string(), "a".to_string()],
            vec!["2".to_string(), "e2".to_string(), "b".to_string()],
        ]
    );
}

#[tokio::test]
async fn merge_on_read_update_after_a_name_swap_sets_the_own_field() {
    let (ctx, _warehouse, _, _) = swapped_names_table("merge-on-read").await;
    run(&ctx, "UPDATE catalog.ns.t SET v = 'x' WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![
            vec!["1".to_string(), "x".to_string(), "a".to_string()],
            vec!["2".to_string(), "e2".to_string(), "b".to_string()],
        ]
    );
}

#[tokio::test]
async fn copy_on_write_delete_after_a_name_swap_keeps_each_value_under_its_field() {
    let (ctx, _warehouse, _, _) = swapped_names_table("copy-on-write").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![vec!["2".to_string(), "e2".to_string(), "b".to_string()],]
    );
}

#[tokio::test]
async fn merge_on_read_delete_after_a_name_swap_keeps_each_value_under_its_field() {
    let (ctx, _warehouse, _, _) = swapped_names_table("merge-on-read").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![vec!["2".to_string(), "e2".to_string(), "b".to_string()],]
    );
}

fn int_list(element_id: i32) -> Type {
    Type::List(ListType::new(
        NestedField::list_element(element_id, Type::Primitive(PrimitiveType::Int), false).into(),
    ))
}

fn nested_fields() -> Vec<Arc<NestedField>> {
    vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
        NestedField::optional(2, "xs", int_list(3)).into(),
        NestedField::optional(
            4,
            "pairs",
            Type::List(ListType::new(
                NestedField::list_element(
                    5,
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(6, "a", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(7, "b", Type::Primitive(PrimitiveType::String))
                            .into(),
                    ])),
                    false,
                )
                .into(),
            )),
        )
        .into(),
        NestedField::optional(
            8,
            "props",
            Type::Map(MapType::new(
                NestedField::map_key_element(9, Type::Primitive(PrimitiveType::String)).into(),
                NestedField::map_value_element(10, int_list(11), false).into(),
            )),
        )
        .into(),
    ]
}

async fn nested_table() -> (SessionContext, TempDir) {
    let (ctx, warehouse, _, _) = catalog_ctx("copy-on-write", nested_fields()).await;
    (ctx, warehouse)
}

fn parquet_leaf_ids(dir: &Path) -> Vec<Vec<i32>> {
    let mut out = vec![];
    for entry in std::fs::read_dir(dir).expect("list warehouse dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            out.extend(parquet_leaf_ids(&path));
        } else if path.extension().is_some_and(|ext| ext == "parquet") {
            let reader = ArrowReaderMetadata::load(
                &File::open(&path).expect("open data file"),
                ArrowReaderOptions::default(),
            )
            .expect("read parquet footer");
            out.push(
                reader
                    .parquet_schema()
                    .columns()
                    .iter()
                    .map(|column| column.self_type().get_basic_info().id())
                    .collect(),
            );
        }
    }
    out
}

const NESTED_LEAF_IDS: [i32; 6] = [1, 5, 7, 8, 9, 11];

#[tokio::test]
async fn insert_values_into_a_list_column_writes_and_stamps_the_element_id() {
    let (ctx, warehouse) = nested_table().await;
    run(
        &ctx,
        "INSERT INTO catalog.ns.t VALUES (1, make_array(1, 2), NULL, NULL), (2, make_array(3), NULL, NULL)",
    )
    .await;
    run(
        &ctx,
        "INSERT INTO catalog.ns.t VALUES (5, make_array(6), NULL, NULL), (6, NULL, NULL, NULL)",
    )
    .await;
    run(
        &ctx,
        "INSERT INTO catalog.ns.t VALUES (7, make_array(5, NULL, 7), NULL, NULL)",
    )
    .await;
    for leaf_ids in parquet_leaf_ids(warehouse.path()) {
        assert_eq!(leaf_ids, NESTED_LEAF_IDS);
    }
    assert_eq!(
        select_all(&ctx, "SELECT id, xs FROM catalog.ns.t", 2).await,
        vec![
            vec!["1".to_string(), "[1, 2]".to_string()],
            vec!["2".to_string(), "[3]".to_string()],
            vec!["5".to_string(), "[6]".to_string()],
            vec!["6".to_string(), "NULL".to_string()],
            vec!["7".to_string(), "[5, , 7]".to_string()],
        ]
    );
}

#[tokio::test]
async fn insert_values_into_list_struct_and_map_columns_writes() {
    let (ctx, warehouse) = nested_table().await;
    run(
        &ctx,
        "INSERT INTO catalog.ns.t VALUES (8, NULL, make_array(named_struct('a', CAST(10 AS INT), 'b', 'x'), named_struct('a', CAST(20 AS INT), 'b', NULL)), map('k1', make_array(CAST(5 AS INT), CAST(6 AS INT)))), (9, NULL, NULL, NULL)",
    )
    .await;
    for leaf_ids in parquet_leaf_ids(warehouse.path()) {
        assert_eq!(leaf_ids, NESTED_LEAF_IDS);
    }
    assert_eq!(
        select_all(&ctx, "SELECT id, pairs, props FROM catalog.ns.t", 3).await,
        vec![
            vec![
                "8".to_string(),
                "[{a: 10, b: x}, {a: 20, b: }]".to_string(),
                "{k1: [5, 6]}".to_string(),
            ],
            vec!["9".to_string(), "NULL".to_string(), "NULL".to_string()],
        ]
    );
}

#[tokio::test]
async fn insert_select_from_a_static_provider_into_nested_columns_writes() {
    let (ctx, warehouse, catalog, namespace) = catalog_ctx("copy-on-write", nested_fields()).await;
    run(
        &ctx,
        "INSERT INTO catalog.ns.t VALUES (1, make_array(1, 2), NULL, NULL)",
    )
    .await;
    let table = load(&namespace, &catalog).await;
    let source = IcebergStaticTableProvider::try_new_from_table(table)
        .await
        .expect("static provider");
    ctx.register_table("src", Arc::new(source))
        .expect("register src");
    run(
        &ctx,
        "INSERT INTO catalog.ns.t SELECT id, xs, pairs, props FROM src",
    )
    .await;
    for leaf_ids in parquet_leaf_ids(warehouse.path()) {
        assert_eq!(leaf_ids, NESTED_LEAF_IDS);
    }
    assert_eq!(
        select_all(&ctx, "SELECT id, xs FROM catalog.ns.t", 2).await,
        vec![vec!["1".to_string(), "[1, 2]".to_string()], vec![
            "1".to_string(),
            "[1, 2]".to_string()
        ],]
    );
}

#[tokio::test]
async fn insert_values_into_a_static_provider_fails_on_write_not_planning() {
    let (ctx, _warehouse, catalog, namespace) = catalog_ctx("copy-on-write", nested_fields()).await;
    run(
        &ctx,
        "INSERT INTO catalog.ns.t VALUES (1, make_array(1, 2), NULL, NULL)",
    )
    .await;
    let table = load(&namespace, &catalog).await;
    let source = IcebergStaticTableProvider::try_new_from_table(table)
        .await
        .expect("static provider");
    ctx.register_table("src", Arc::new(source))
        .expect("register src");
    let df = ctx
        .sql("INSERT INTO src VALUES (5, make_array(6), NULL, NULL), (6, NULL, NULL, NULL)")
        .await
        .expect("plan");
    let msg = df
        .collect()
        .await
        .expect_err("static provider refuses writes")
        .to_string();
    assert!(
        msg.contains("Write operations are not supported"),
        "expected the write refusal, got: {msg}"
    );
}
