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

use datafusion::arrow::array::Int32Array;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{MAIN_BRANCH, NestedField, Operation, PrimitiveType, Schema, Type};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{
    Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, Result, TableCreation, TableIdent,
};
use iceberg_datafusion::IcebergTableProvider;
use tempfile::TempDir;

fn temp_path() -> String {
    TempDir::new()
        .expect("temp dir")
        .path()
        .to_str()
        .expect("utf-8")
        .to_string()
}

async fn memory_catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), temp_path())]),
        )
        .await
        .expect("memory catalog")
}

fn schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema")
}

async fn create_table(
    catalog: &MemoryCatalog,
    ns: &str,
    name: &str,
    properties: HashMap<String, String>,
) -> NamespaceIdent {
    let namespace = NamespaceIdent::new(ns.to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name(name.to_string())
                .location(format!("{}/{}", temp_path(), name))
                .schema(schema())
                .properties(properties)
                .build(),
        )
        .await
        .expect("create table");
    namespace
}

async fn provider(
    catalog: Arc<dyn Catalog>,
    namespace: NamespaceIdent,
    name: &str,
    branch: Option<&str>,
    stage_only: bool,
) -> IcebergTableProvider {
    let base = IcebergTableProvider::try_new(catalog, namespace, name.to_string())
        .await
        .expect("provider");
    let branched = match branch {
        Some(name) => base.with_commit_branch(name),
        None => base,
    };
    branched.with_stage_only(stage_only)
}

async fn register(provider: IcebergTableProvider) -> SessionContext {
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register");
    ctx
}

async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|e| panic!("plan `{sql}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("exec `{sql}`: {e}"));
}

async fn load(catalog: &dyn Catalog, namespace: &NamespaceIdent, name: &str) -> Table {
    catalog
        .load_table(&TableIdent::new(namespace.clone(), name.to_string()))
        .await
        .expect("load table")
}

fn ref_id(table: &Table, name: &str) -> Option<i64> {
    table
        .metadata()
        .snapshot_for_ref(name)
        .map(|snapshot| snapshot.snapshot_id())
}

async fn create_named_branch(catalog: &dyn Catalog, table: &Table, branch: &str) -> Table {
    let main_id = table
        .metadata()
        .current_snapshot_id()
        .expect("main head for branch");
    let tx = Transaction::new(table);
    let tx = tx
        .manage_snapshots()
        .create_branch(branch, main_id)
        .apply(tx)
        .expect("apply create_branch");
    tx.commit(catalog).await.expect("commit create_branch")
}

fn sorted_ids(batches: &[RecordBatch]) -> Vec<i32> {
    let mut ids = Vec::new();
    for batch in batches {
        let column = batch
            .column_by_name("id")
            .expect("id")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id int");
        for row in 0..column.len() {
            ids.push(column.value(row));
        }
    }
    ids.sort_unstable();
    ids
}

async fn query_ids(ctx: &SessionContext, sql: &str) -> Vec<i32> {
    let batches = ctx
        .sql(sql)
        .await
        .unwrap_or_else(|e| panic!("plan `{sql}`: {e}"))
        .collect()
        .await
        .unwrap_or_else(|e| panic!("exec `{sql}`: {e}"));
    sorted_ids(&batches)
}

async fn seed(catalog: &Arc<dyn Catalog>, namespace: &NamespaceIdent) -> i64 {
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None, false).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a')").await;
    load(catalog.as_ref(), namespace, "t")
        .await
        .metadata()
        .current_snapshot_id()
        .expect("seed head")
}

#[tokio::test]
async fn insert_stage_only_adds_snapshot_without_moving_main() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_stage_append", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let main_id = seed(&catalog, &namespace).await;

    let staged_ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", None, true).await).await;
    run_sql(&staged_ctx, "INSERT INTO t VALUES (2, 'b')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    assert_eq!(ref_id(&table, MAIN_BRANCH), Some(main_id));
    assert_eq!(table.metadata().snapshots().len(), 2);
    let staged = table
        .metadata()
        .snapshots()
        .find(|s| s.snapshot_id() != main_id)
        .expect("staged snapshot");
    assert_eq!(staged.parent_snapshot_id(), Some(main_id));
    assert_eq!(staged.summary().operation, Operation::Append);

    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None, false).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1]);
    Ok(())
}

#[tokio::test]
async fn insert_overwrite_stage_only_adds_snapshot_without_moving_main() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_stage_overwrite", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let main_id = seed(&catalog, &namespace).await;

    let staged_ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", None, true).await).await;
    run_sql(&staged_ctx, "INSERT OVERWRITE t VALUES (9, 'z')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    assert_eq!(ref_id(&table, MAIN_BRANCH), Some(main_id));
    assert_eq!(table.metadata().snapshots().len(), 2);
    let staged = table
        .metadata()
        .snapshots()
        .find(|s| s.snapshot_id() != main_id)
        .expect("staged snapshot");
    assert_eq!(staged.parent_snapshot_id(), Some(main_id));
    assert_eq!(staged.summary().operation, Operation::Overwrite);

    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None, false).await).await;
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1]);
    Ok(())
}

#[tokio::test]
async fn stage_only_with_commit_branch_leaves_branch_unmoved() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_stage_branch", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let main_id = seed(&catalog, &namespace).await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let table = create_named_branch(catalog.as_ref(), &table, "audit").await;
    assert_eq!(ref_id(&table, "audit"), Some(main_id));

    let staged_ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("audit"), true).await)
            .await;
    run_sql(&staged_ctx, "INSERT INTO t VALUES (2, 'b')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    assert_eq!(ref_id(&table, "audit"), Some(main_id));
    assert_eq!(table.metadata().snapshots().len(), 2);
    Ok(())
}

#[tokio::test]
async fn insert_without_stage_only_advances_main() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_plain_append", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None, false).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    let main_id = table.metadata().current_snapshot_id();
    assert!(main_id.is_some());
    assert_eq!(ref_id(&table, MAIN_BRANCH), main_id);
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![1]);
    Ok(())
}

#[tokio::test]
async fn insert_overwrite_without_stage_only_advances_main() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_plain_overwrite", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let main_id = seed(&catalog, &namespace).await;

    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None, false).await).await;
    run_sql(&ctx, "INSERT OVERWRITE t VALUES (9, 'z')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    let moved = table.metadata().current_snapshot_id().expect("moved");
    assert_ne!(moved, main_id);
    assert_eq!(ref_id(&table, MAIN_BRANCH), Some(moved));
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .expect("head")
            .summary()
            .operation,
        Operation::Overwrite
    );
    assert_eq!(query_ids(&ctx, "SELECT id FROM t").await, vec![9]);
    Ok(())
}

#[tokio::test]
async fn commit_branch_without_stage_only_still_targets_branch() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_plain_branch", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let main_id = seed(&catalog, &namespace).await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let table = create_named_branch(catalog.as_ref(), &table, "audit").await;
    assert_eq!(ref_id(&table, "audit"), Some(main_id));

    let ctx = register(
        provider(
            catalog.clone(),
            namespace.clone(),
            "t",
            Some("audit"),
            false,
        )
        .await,
    )
    .await;
    run_sql(&ctx, "INSERT INTO t VALUES (2, 'b')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let branch_id = ref_id(&table, "audit").expect("audit");
    assert_ne!(branch_id, main_id);
    Ok(())
}
