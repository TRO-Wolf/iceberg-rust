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

//! H7-P1: DML Iceberg filters are prune-only. The DataFusion `WHERE` stays exact.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::UInt64Array;
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec};
use iceberg::{Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, TableCreation};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

fn temp_path() -> String {
    TempDir::new()
        .expect("temp dir")
        .path()
        .to_str()
        .expect("utf-8")
        .to_string()
}

async fn catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), temp_path())]),
        )
        .await
        .expect("memory catalog")
}

fn cow_table(location: &str, name: &str) -> TableCreation {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    TableCreation::builder()
        .location(location.to_string())
        .name(name.to_string())
        .schema(schema)
        .build()
}

fn mor_table(location: &str, name: &str) -> TableCreation {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "foo1", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "foo2", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let partition_spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "foo2", Transform::Identity)
        .expect("partition field")
        .build();
    TableCreation::builder()
        .location(location.to_string())
        .name(name.to_string())
        .properties(HashMap::from([
            ("write.delete.mode".to_string(), "merge-on-read".to_string()),
            ("write.update.mode".to_string(), "merge-on-read".to_string()),
        ]))
        .schema(schema)
        .partition_spec(partition_spec)
        .build()
}

async fn ctx_for(ns: &str, table: &str) -> SessionContext {
    let iceberg_catalog = catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    iceberg_catalog
        .create_table(&namespace, mor_table(&temp_path(), table))
        .await
        .expect("create table");
    let catalog = Arc::new(
        IcebergCatalogProvider::try_new(Arc::new(iceberg_catalog))
            .await
            .expect("provider"),
    );
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", catalog);
    ctx
}

async fn sql_count(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|e| panic!("sql {sql}: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("collect {sql}: {e}"));
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("count column")
        .value(0)
}

async fn live_foo1(ctx: &SessionContext, table: &str) -> Vec<i32> {
    let batches = ctx
        .sql(&format!("SELECT foo1 FROM {table} ORDER BY foo1"))
        .await
        .expect("select")
        .collect()
        .await
        .expect("collect");
    let mut ids = Vec::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int32Array>()
            .expect("foo1");
        for i in 0..col.len() {
            ids.push(col.value(i));
        }
    }
    ids
}

/// Two files so a `foo2` equality prune can skip one. File A is west, file B is us.
async fn seed_west_and_us(ctx: &SessionContext, table: &str) {
    ctx.sql(&format!(
        "INSERT INTO {table} VALUES (1, 'west'), (20, 'west')"
    ))
    .await
    .expect("insert west")
    .collect()
    .await
    .expect("collect west");
    ctx.sql(&format!("INSERT INTO {table} VALUES (2, 'us')"))
        .await
        .expect("insert us")
        .collect()
        .await
        .expect("collect us");
}

#[tokio::test]
async fn delete_does_not_treat_a_pushable_conjunct_as_the_row_filter() {
    let ctx = ctx_for("h7_p1_exact", "t").await;
    let table = "catalog.h7_p1_exact.t";
    seed_west_and_us(&ctx, table).await;

    // `lower` does not convert. If Iceberg prune became the exact filter, `foo2 = 'us'`
    // would delete id 2. The exact WHERE matches no row.
    let deleted = sql_count(
        &ctx,
        &format!("DELETE FROM {table} WHERE foo2 = 'us' AND lower(foo2) = 'xx'"),
    )
    .await;
    assert_eq!(deleted, 0, "exact WHERE matches no row");
    assert_eq!(live_foo1(&ctx, table).await, vec![1, 2, 20]);
}

#[tokio::test]
async fn delete_not_over_partial_and_matches_no_pushdown_row_set() {
    let ctx = ctx_for("h7_p1_not_and", "t").await;
    let table = "catalog.h7_p1_not_and.t";
    seed_west_and_us(&ctx, table).await;

    // NOT (foo2 = 'west' AND length(foo2) > 10): length('west') is 4, so every row matches
    // and the table is empty. AND-drop under NOT prunes the west file and leaves ids 1 and 20.
    let deleted = sql_count(
        &ctx,
        &format!("DELETE FROM {table} WHERE NOT (foo2 = 'west' AND length(foo2) > 10)"),
    )
    .await;
    assert_eq!(deleted, 3, "every row matches the exact WHERE");
    assert_eq!(live_foo1(&ctx, table).await, Vec::<i32>::new());
}

#[tokio::test]
async fn update_not_over_partial_and_matches_no_pushdown_row_set() {
    let ctx = ctx_for("h7_p1_upd", "t").await;
    let table = "catalog.h7_p1_upd.t";
    seed_west_and_us(&ctx, table).await;

    let updated = sql_count(
        &ctx,
        &format!("UPDATE {table} SET foo2 = 'z' WHERE NOT (foo2 = 'west' AND length(foo2) > 10)"),
    )
    .await;
    assert_eq!(updated, 3, "every row matches the exact WHERE");
    let batches = ctx
        .sql(&format!("SELECT foo1, foo2 FROM {table} ORDER BY foo1"))
        .await
        .expect("select")
        .collect()
        .await
        .expect("collect");
    let foo2 = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::StringArray>()
        .expect("foo2");
    assert_eq!(foo2.value(0), "z");
    assert_eq!(foo2.value(1), "z");
    assert_eq!(foo2.value(2), "z");
}

#[tokio::test]
async fn cow_delete_keeps_co_located_rows_that_miss_a_convertible_where() {
    let iceberg_catalog = catalog().await;
    let namespace = NamespaceIdent::new("h7_p1_cow".to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    iceberg_catalog
        .create_table(&namespace, cow_table(&temp_path(), "t"))
        .await
        .expect("create table");
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(
            IcebergCatalogProvider::try_new(Arc::new(iceberg_catalog))
                .await
                .expect("provider"),
        ),
    );
    let table = "catalog.h7_p1_cow.t";
    // One INSERT = one file. `foo1 = 1` converts fully and matches only one row in that file.
    ctx.sql(&format!("INSERT INTO {table} VALUES (1, 'a'), (20, 'b')"))
        .await
        .expect("insert")
        .collect()
        .await
        .expect("collect insert");
    let deleted = sql_count(&ctx, &format!("DELETE FROM {table} WHERE foo1 = 1")).await;
    assert_eq!(deleted, 1);
    assert_eq!(
        live_foo1(&ctx, table).await,
        vec![20],
        "the non-matching row in the same file must survive; with_filter residual would drop it"
    );
}

#[tokio::test]
async fn cow_update_keeps_co_located_rows_that_miss_a_convertible_where() {
    let iceberg_catalog = catalog().await;
    let namespace = NamespaceIdent::new("h7_p1_cow_upd".to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    iceberg_catalog
        .create_table(&namespace, cow_table(&temp_path(), "t"))
        .await
        .expect("create table");
    let ctx = SessionContext::new();
    ctx.register_catalog(
        "catalog",
        Arc::new(
            IcebergCatalogProvider::try_new(Arc::new(iceberg_catalog))
                .await
                .expect("provider"),
        ),
    );
    let table = "catalog.h7_p1_cow_upd.t";
    ctx.sql(&format!("INSERT INTO {table} VALUES (1, 'a'), (20, 'b')"))
        .await
        .expect("insert")
        .collect()
        .await
        .expect("collect insert");
    let updated = sql_count(
        &ctx,
        &format!("UPDATE {table} SET foo2 = 'z' WHERE foo1 = 1"),
    )
    .await;
    assert_eq!(updated, 1);
    let batches = ctx
        .sql(&format!("SELECT foo1, foo2 FROM {table} ORDER BY foo1"))
        .await
        .expect("select")
        .collect()
        .await
        .expect("collect");
    let foo1 = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int32Array>()
        .expect("foo1");
    let foo2 = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::StringArray>()
        .expect("foo2");
    assert_eq!(foo1.values(), &[1, 20]);
    assert_eq!(foo2.value(0), "z");
    assert_eq!(foo2.value(1), "b");
}
