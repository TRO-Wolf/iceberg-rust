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

//! F-6b: `IcebergTableProvider::with_commit_branch` hands `to_branch` at every DML commit site.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::execution::context::SessionContext;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{MAIN_BRANCH, NestedField, PrimitiveType, Schema, Type};
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
) -> IcebergTableProvider {
    let provider = IcebergTableProvider::try_new(catalog, namespace, name.to_string())
        .await
        .expect("provider");
    match branch {
        Some(name) => provider.with_commit_branch(name),
        None => provider,
    }
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

#[tokio::test]
async fn insert_without_target_advances_main() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_default", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    let main_id = table.metadata().current_snapshot_id();
    assert!(main_id.is_some(), "default INSERT must stamp main");
    assert_eq!(ref_id(&table, MAIN_BRANCH), main_id);
    assert!(
        table.metadata().snapshot_for_ref("audit").is_none(),
        "default INSERT must not invent a branch"
    );
    Ok(())
}

#[tokio::test]
async fn insert_with_commit_branch_does_not_move_main() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_named", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let table = create_named_branch(catalog.as_ref(), &table, "audit").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");
    assert_eq!(ref_id(&table, "audit"), Some(main_id));

    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("audit")).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (2, 'b')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(
        table.metadata().current_snapshot_id(),
        Some(main_id),
        "main must be byte-unmoved"
    );
    let branch_id = ref_id(&table, "audit").expect("audit");
    assert_ne!(branch_id, main_id, "named branch must advance");
    Ok(())
}

#[tokio::test]
async fn insert_with_commit_branch_creates_missing_branch() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_missing", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");
    assert!(table.metadata().snapshot_for_ref("audit").is_none());

    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("audit")).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (2, 'b')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let branch_id = ref_id(&table, "audit").expect("created audit");
    assert_ne!(branch_id, main_id);
    let parent = table
        .metadata()
        .snapshot_by_id(branch_id)
        .expect("audit snapshot")
        .parent_snapshot_id();
    assert_eq!(parent, Some(main_id));
    Ok(())
}

#[tokio::test]
async fn insert_with_commit_branch_rejects_tag() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_tag", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");
    let tx = Transaction::new(&table);
    let tx = tx
        .manage_snapshots()
        .create_tag("some-tag", main_id)
        .apply(tx)
        .expect("apply create_tag");
    tx.commit(catalog.as_ref())
        .await
        .expect("commit create_tag");

    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("some-tag")).await).await;
    let err = ctx
        .sql("INSERT INTO t VALUES (2, 'b')")
        .await
        .expect("plan")
        .collect()
        .await
        .expect_err("a tag must not be a commit target");
    let message = err.to_string();
    assert!(
        message.contains(
            "some-tag is a tag, not a branch. Tags cannot be targets for producing snapshots"
        ),
        "unexpected message: {message}"
    );
    Ok(())
}

#[tokio::test]
async fn insert_overwrite_with_commit_branch_does_not_move_main() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_ow", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let table = create_named_branch(catalog.as_ref(), &table, "audit").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");

    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("audit")).await).await;
    run_sql(&ctx, "INSERT OVERWRITE t VALUES (9, 'z')").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let branch_id = ref_id(&table, "audit").expect("audit");
    assert_ne!(branch_id, main_id);
    Ok(())
}

#[tokio::test]
async fn insert_overwrite_on_diverged_branch_does_not_treat_branch_files_as_concurrent()
-> Result<()> {
    let properties = HashMap::from([(
        "write.overwrite.isolation-level".to_string(),
        "serializable".to_string(),
    )]);
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_ow_div", "t", properties).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let ctx = register(provider(catalog.clone(), namespace.clone(), "t", None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    let table = create_named_branch(catalog.as_ref(), &table, "audit").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");

    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("audit")).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (2, 'b')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let diverged = ref_id(&table, "audit").expect("diverged audit");
    assert_ne!(diverged, main_id);

    run_sql(&ctx, "INSERT OVERWRITE t VALUES (9, 'z')").await;
    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(
        table.metadata().current_snapshot_id(),
        Some(main_id),
        "main must stay at the pre-diverge snapshot"
    );
    let after = ref_id(&table, "audit").expect("audit after overwrite");
    assert_ne!(after, diverged);
    Ok(())
}

async fn seed_then_branch(
    catalog: &Arc<dyn Catalog>,
    namespace: NamespaceIdent,
) -> (NamespaceIdent, i64) {
    let name = "t";
    let ctx = register(provider(catalog.clone(), namespace.clone(), name, None).await).await;
    run_sql(&ctx, "INSERT INTO t VALUES (1, 'a'), (2, 'b')").await;
    let table = load(catalog.as_ref(), &namespace, name).await;
    let table = create_named_branch(catalog.as_ref(), &table, "audit").await;
    let main_id = table.metadata().current_snapshot_id().expect("main");
    (namespace, main_id)
}

#[tokio::test]
async fn copy_on_write_delete_with_commit_branch_does_not_move_main() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_cow_del", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let (namespace, main_id) = seed_then_branch(&catalog, namespace).await;

    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("audit")).await).await;
    run_sql(&ctx, "DELETE FROM t WHERE id = 1").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let branch_id = ref_id(&table, "audit").expect("audit");
    assert_ne!(branch_id, main_id);
    Ok(())
}

#[tokio::test]
async fn merge_on_read_delete_with_commit_branch_does_not_move_main() -> Result<()> {
    let properties = HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        ("write.update.mode".to_string(), "merge-on-read".to_string()),
    ]);
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_mor_del", "t", properties).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let (namespace, main_id) = seed_then_branch(&catalog, namespace).await;

    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("audit")).await).await;
    run_sql(&ctx, "DELETE FROM t WHERE id = 1").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let branch_id = ref_id(&table, "audit").expect("audit");
    assert_ne!(branch_id, main_id);
    Ok(())
}

#[tokio::test]
async fn copy_on_write_update_with_commit_branch_does_not_move_main() -> Result<()> {
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_cow_upd", "t", HashMap::new()).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let (namespace, main_id) = seed_then_branch(&catalog, namespace).await;

    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("audit")).await).await;
    run_sql(&ctx, "UPDATE t SET name = 'z' WHERE id = 1").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let branch_id = ref_id(&table, "audit").expect("audit");
    assert_ne!(branch_id, main_id);
    Ok(())
}

#[tokio::test]
async fn merge_on_read_update_with_commit_branch_does_not_move_main() -> Result<()> {
    let properties = HashMap::from([
        ("write.delete.mode".to_string(), "merge-on-read".to_string()),
        ("write.update.mode".to_string(), "merge-on-read".to_string()),
    ]);
    let catalog = memory_catalog().await;
    let namespace = create_table(&catalog, "ns_mor_upd", "t", properties).await;
    let catalog: Arc<dyn Catalog> = Arc::new(catalog);
    let (namespace, main_id) = seed_then_branch(&catalog, namespace).await;

    let ctx =
        register(provider(catalog.clone(), namespace.clone(), "t", Some("audit")).await).await;
    run_sql(&ctx, "UPDATE t SET name = 'z' WHERE id = 1").await;

    let table = load(catalog.as_ref(), &namespace, "t").await;
    assert_eq!(table.metadata().current_snapshot_id(), Some(main_id));
    let branch_id = ref_id(&table, "audit").expect("audit");
    assert_ne!(branch_id, main_id);
    Ok(())
}
