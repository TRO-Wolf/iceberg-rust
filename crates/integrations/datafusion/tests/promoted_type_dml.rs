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

use datafusion::arrow::array::{Array, Int64Array};
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

async fn single_era_promoted_table(mode: &str) -> (SessionContext, TempDir) {
    let warehouse = TempDir::new().expect("warehouse");
    let path = warehouse.path().to_str().expect("utf-8 path").to_string();
    let iceberg_catalog = Arc::new(
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
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "s", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("pre-promotion schema");
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
    run(&ctx, "INSERT INTO catalog.ns.t VALUES (5, 'e5'), (6, 'e6')").await;
    let table = iceberg_catalog
        .load_table(&TableIdent::new(namespace, "t".to_string()))
        .await
        .expect("load table");
    let tx = Transaction::new(&table);
    let action = tx.update_schema().update_column("id", PrimitiveType::Long);
    action
        .apply(tx)
        .expect("apply the promotion")
        .commit(iceberg_catalog.as_ref())
        .await
        .expect("commit the promotion");
    (ctx, warehouse)
}

async fn rows(ctx: &SessionContext) -> Vec<(i64, String)> {
    let batches = ctx
        .sql("SELECT id, s FROM catalog.ns.t")
        .await
        .expect("plan select")
        .collect()
        .await
        .expect("run select");
    let mut out = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id reads as long after the promotion");
        for row in 0..batch.num_rows() {
            let name = array_value_to_string(batch.column(1), row).expect("s renders");
            out.push((ids.value(row), name));
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn copy_on_write_update_after_a_promotion_updates_pre_promotion_rows() {
    let (ctx, _warehouse) = single_era_promoted_table("copy-on-write").await;
    run(&ctx, "UPDATE catalog.ns.t SET s = 'u5' WHERE id = 5").await;
    assert_eq!(rows(&ctx).await, vec![
        (5, "u5".to_string()),
        (6, "e6".to_string())
    ]);
}

#[tokio::test]
async fn merge_on_read_update_after_a_promotion_updates_pre_promotion_rows() {
    let (ctx, _warehouse) = single_era_promoted_table("merge-on-read").await;
    run(&ctx, "UPDATE catalog.ns.t SET s = 'u5' WHERE id = 5").await;
    assert_eq!(rows(&ctx).await, vec![
        (5, "u5".to_string()),
        (6, "e6".to_string())
    ]);
}

#[tokio::test]
async fn copy_on_write_delete_after_a_promotion_removes_the_matching_row() {
    let (ctx, _warehouse) = single_era_promoted_table("copy-on-write").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 6").await;
    assert_eq!(rows(&ctx).await, vec![(5, "e5".to_string())]);
}

#[tokio::test]
async fn merge_on_read_delete_after_a_promotion_removes_the_matching_row() {
    let (ctx, _warehouse) = single_era_promoted_table("merge-on-read").await;
    run(&ctx, "DELETE FROM catalog.ns.t WHERE id = 6").await;
    assert_eq!(rows(&ctx).await, vec![(5, "e5".to_string())]);
}
