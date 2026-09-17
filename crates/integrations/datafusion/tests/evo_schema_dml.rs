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

use datafusion::arrow::util::display::array_value_to_string;
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_datafusion::IcebergCatalogProvider;
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

async fn base_table(
    mode: &str,
    fields: Vec<Arc<NestedField>>,
    first: &str,
    second: &str,
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
    run(
        &ctx,
        &format!("INSERT INTO catalog.ns.t VALUES {first}"),
    )
    .await;
    run(
        &ctx,
        &format!("INSERT INTO catalog.ns.t VALUES {second}"),
    )
    .await;
    (ctx, warehouse, iceberg_catalog, namespace)
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
                rendered.push(array_value_to_string(batch.column(column), row).expect("renders"));
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
        vec![vec![
            "2".to_string(),
            "b".to_string(),
            "NULL".to_string()
        ],]
    );
}

#[tokio::test]
async fn merge_on_read_delete_after_add_column_removes_the_matching_row() {
    let (ctx, _warehouse, _, _) = added_column_table("merge-on-read").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![vec![
            "2".to_string(),
            "b".to_string(),
            "NULL".to_string()
        ],]
    );
}

#[tokio::test]
async fn copy_on_write_update_after_rename_sets_the_renamed_column() {
    let (ctx, _warehouse, _, _) = renamed_column_table("copy-on-write").await;
    run(&ctx, "UPDATE catalog.ns.t SET v = 'x' WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v FROM catalog.ns.t", 2).await,
        vec![
            vec!["1".to_string(), "x".to_string()],
            vec!["2".to_string(), "b".to_string()],
        ]
    );
}

#[tokio::test]
async fn merge_on_read_update_after_rename_sets_the_renamed_column() {
    let (ctx, _warehouse, _, _) = renamed_column_table("merge-on-read").await;
    run(&ctx, "UPDATE catalog.ns.t SET v = 'x' WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v FROM catalog.ns.t", 2).await,
        vec![
            vec!["1".to_string(), "x".to_string()],
            vec!["2".to_string(), "b".to_string()],
        ]
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
        vec![vec![
            "2".to_string(),
            "e2".to_string(),
            "b".to_string()
        ],]
    );
}

#[tokio::test]
async fn merge_on_read_delete_after_a_name_swap_keeps_each_value_under_its_field() {
    let (ctx, _warehouse, _, _) = swapped_names_table("merge-on-read").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 1").await;
    assert_eq!(
        select_all(&ctx, "SELECT id, v, extra FROM catalog.ns.t", 3).await,
        vec![vec![
            "2".to_string(),
            "e2".to_string(),
            "b".to_string()
        ],]
    );
}
