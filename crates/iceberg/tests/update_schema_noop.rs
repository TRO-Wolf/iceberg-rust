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

use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{Literal, NestedField, PrimitiveType, Schema, Type};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

fn schema_abc() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "b", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "c", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("schema builds")
}

async fn new_local_catalog(name: &str, warehouse: &str) -> iceberg::memory::MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name.to_string(),
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("catalog loads")
}

async fn create_table(
    catalog: &impl Catalog,
    ns: &NamespaceIdent,
    name: &str,
) -> (iceberg::table::Table, TableIdent) {
    let table = catalog
        .create_table(
            ns,
            TableCreation::builder()
                .name(name.to_string())
                .schema(schema_abc())
                .build(),
        )
        .await
        .expect("table creates");
    let ident = table.identifier().clone();
    (table, ident)
}

fn field_names(table: &iceberg::table::Table) -> Vec<String> {
    table
        .metadata()
        .current_schema()
        .as_struct()
        .fields()
        .iter()
        .map(|field| field.name.clone())
        .collect()
}

#[tokio::test]
async fn noop_move_commits_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let catalog = new_local_catalog("cat", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("namespace creates");
    let (table, ident) = create_table(&catalog, &ns, "t").await;

    let loc_before = table.metadata_location().unwrap().to_string();
    let schema_id_before = table.metadata().current_schema_id();
    let schemas_before = table.metadata().schemas_iter().count();

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .move_before("a", "b")
        .apply(tx)
        .expect("apply stages");
    let committed = tx.commit(&catalog).await.expect("no-op commit succeeds");

    assert_eq!(
        committed.metadata_location().map(str::to_string).as_deref(),
        Some(loc_before.as_str()),
        "a no-op move must not write a new metadata version"
    );
    assert_eq!(committed.metadata().current_schema_id(), schema_id_before);
    assert_eq!(
        committed.metadata().schemas_iter().count(),
        schemas_before,
        "a no-op move must not add a schema"
    );
    assert_eq!(
        field_names(&committed),
        vec!["a", "b", "c"],
        "field order unchanged"
    );

    let reloaded = catalog.load_table(&ident).await.expect("reload");
    assert_eq!(
        reloaded.metadata_location().map(str::to_string).as_deref(),
        Some(loc_before.as_str()),
        "the catalog pointer must not move on a no-op commit"
    );

    let tx = Transaction::new(&committed);
    let tx = tx
        .update_schema()
        .move_after("a", "b")
        .move_before("a", "b")
        .apply(tx)
        .expect("apply stages");
    let committed = tx
        .commit(&catalog)
        .await
        .expect("round-trip commit succeeds");
    assert_eq!(
        committed.metadata_location().map(str::to_string).as_deref(),
        Some(loc_before.as_str()),
        "a move that round-trips must not write a new metadata version"
    );
    assert_eq!(committed.metadata().current_schema_id(), schema_id_before);
}

#[tokio::test]
async fn real_move_still_commits() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let catalog = new_local_catalog("cat", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("namespace creates");
    let (table, _ident) = create_table(&catalog, &ns, "t").await;

    let loc_before = table.metadata_location().unwrap().to_string();
    let schema_id_before = table.metadata().current_schema_id();
    let schemas_before = table.metadata().schemas_iter().count();

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .move_after("a", "b")
        .apply(tx)
        .expect("apply stages");
    let committed = tx.commit(&catalog).await.expect("real move commits");

    assert_ne!(
        committed.metadata_location().map(str::to_string).as_deref(),
        Some(loc_before.as_str()),
        "a real move must write a new metadata version"
    );
    assert_ne!(committed.metadata().current_schema_id(), schema_id_before);
    assert_eq!(
        committed.metadata().schemas_iter().count(),
        schemas_before + 1,
        "a real move mints a new schema id"
    );
    assert_eq!(field_names(&committed), vec!["b", "a", "c"]);
}

#[tokio::test]
async fn add_plus_noop_move_keeps_the_add() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let catalog = new_local_catalog("cat", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("namespace creates");
    let (table, _ident) = create_table(&catalog, &ns, "t").await;

    let schema_id_before = table.metadata().current_schema_id();

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .add_column("d", Type::Primitive(PrimitiveType::Int))
        .move_before("a", "b")
        .apply(tx)
        .expect("apply stages");
    let committed = tx.commit(&catalog).await.expect("batch commits");

    assert_ne!(committed.metadata().current_schema_id(), schema_id_before);
    assert_eq!(field_names(&committed), vec!["a", "b", "c", "d"]);
}

#[tokio::test]
async fn stale_base_noop_move_revalidates_against_refreshed_schema() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let catalog = new_local_catalog("cat", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("namespace creates");
    let (table, ident) = create_table(&catalog, &ns, "t").await;

    let tx = Transaction::new(&table);
    let stale_tx = tx
        .update_schema()
        .move_before("a", "b")
        .apply(tx)
        .expect("apply stages");

    let fresh = catalog.load_table(&ident).await.expect("reload");
    let tx2 = Transaction::new(&fresh);
    let tx2 = tx2
        .update_schema()
        .rename_column("b", "bb")
        .apply(tx2)
        .expect("apply stages");
    tx2.commit(&catalog)
        .await
        .expect("concurrent rename commits");

    let err = match stale_tx.commit(&catalog).await {
        Ok(_) => panic!("a stale-base move must re-validate against the refreshed schema"),
        Err(e) => e,
    };
    assert!(
        err.message()
            .contains("Cannot move a before missing column: b"),
        "unexpected message: {}",
        err.message()
    );
}

#[tokio::test]
async fn same_doc_reapply_commits_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let catalog = new_local_catalog("cat", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("namespace creates");
    let (table, _ident) = create_table(&catalog, &ns, "t").await;

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .update_column_doc("a", Some("doc-a"))
        .apply(tx)
        .expect("apply stages");
    let committed = tx.commit(&catalog).await.expect("doc set commits");
    let loc_after_doc = committed.metadata_location().unwrap().to_string();
    let schema_id_after_doc = committed.metadata().current_schema_id();

    let tx = Transaction::new(&committed);
    let tx = tx
        .update_schema()
        .update_column_doc("a", Some("doc-a"))
        .apply(tx)
        .expect("apply stages");
    let committed = tx.commit(&catalog).await.expect("same-doc commit succeeds");

    assert_eq!(
        committed.metadata_location().map(str::to_string).as_deref(),
        Some(loc_after_doc.as_str()),
        "re-applying the same doc must not write a new metadata version"
    );
    assert_eq!(
        committed.metadata().current_schema_id(),
        schema_id_after_doc
    );
}

#[tokio::test]
async fn same_default_reapply_commits_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let catalog = new_local_catalog("cat", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("namespace creates");
    let (table, _ident) = create_table(&catalog, &ns, "t").await;

    let tx = Transaction::new(&table);
    let tx = tx
        .update_schema()
        .update_column_default("a", Literal::int(7))
        .apply(tx)
        .expect("apply stages");
    let committed = tx.commit(&catalog).await.expect("default set commits");
    let loc_after = committed.metadata_location().unwrap().to_string();
    let schema_id_after = committed.metadata().current_schema_id();

    let tx = Transaction::new(&committed);
    let tx = tx
        .update_schema()
        .update_column_default("a", Literal::int(7))
        .apply(tx)
        .expect("apply stages");
    let committed = tx
        .commit(&catalog)
        .await
        .expect("same-default commit succeeds");

    assert_eq!(
        committed.metadata_location().map(str::to_string).as_deref(),
        Some(loc_after.as_str()),
        "re-applying the same default must not write a new metadata version"
    );
    assert_eq!(committed.metadata().current_schema_id(), schema_id_after);
}
