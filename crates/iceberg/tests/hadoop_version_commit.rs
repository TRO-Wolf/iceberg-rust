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

use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema, Type,
};
use iceberg::transaction::{ApplyTransactionAction, StagedTableTransaction, Transaction};
use iceberg::{
    Catalog, CatalogBuilder, ErrorKind, MetadataLocation, NamespaceIdent, TableCreation, TableIdent,
};

fn test_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
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

async fn create_ns_and_source(catalog: &impl Catalog, ns: &NamespaceIdent) -> String {
    catalog
        .create_namespace(ns, HashMap::new())
        .await
        .expect("namespace creates");
    let source = catalog
        .create_table(
            ns,
            TableCreation::builder()
                .name("src".to_string())
                .schema(test_schema())
                .build(),
        )
        .await
        .expect("source table creates");
    source.metadata().location().to_string()
}

async fn commit_property(
    catalog: &impl Catalog,
    ident: &TableIdent,
    key: &str,
    value: &str,
) -> Result<String, iceberg::Error> {
    let table = catalog.load_table(ident).await.expect("load base");
    let tx = Transaction::new(&table);
    let committed = tx
        .update_table_properties()
        .set(key.to_string(), value.to_string())
        .apply(tx)
        .expect("apply stages")
        .commit(catalog)
        .await?;
    Ok(committed.metadata_location().expect("location").to_string())
}

fn table_property(catalog_table: &iceberg::table::Table, key: &str) -> Option<String> {
    catalog_table.metadata().properties().get(key).cloned()
}

async fn race_once(
    catalog: &iceberg::memory::MemoryCatalog,
    ident: &TableIdent,
    value: &'static str,
    barrier: Arc<tokio::sync::Barrier>,
) -> Result<&'static str, iceberg::Error> {
    barrier.wait().await;
    commit_property(catalog, ident, "winner", value)
        .await
        .map(|_| value)
}

#[tokio::test]
async fn hadoop_second_commit_to_same_vn_fails_and_preserves_winner() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let file_io = FileIO::new_with_fs();

    let cat1 = new_local_catalog("one", &warehouse).await;
    let cat2 = new_local_catalog("two", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    let table_location = create_ns_and_source(&cat1, &ns).await;
    create_ns_and_source(&cat2, &ns).await;

    let v2 = format!("{table_location}/metadata/v2.metadata.json");
    let base = cat1
        .load_table(&TableIdent::new(ns.clone(), "src".to_string()))
        .await
        .expect("load src");
    base.metadata()
        .write_to(&file_io, &v2)
        .await
        .expect("seed v2");

    let ident = TableIdent::new(ns, "hadoop".to_string());
    cat1.register_table(&ident, v2.clone())
        .await
        .expect("cat1 registers v2");
    cat2.register_table(&ident, v2.clone())
        .await
        .expect("cat2 registers v2");

    let v3 = commit_property(&cat1, &ident, "winner", "one")
        .await
        .expect("first commit lands");
    assert!(v3.ends_with("/metadata/v3.metadata.json"));
    let winner_bytes = std::fs::read(&v3).expect("read v3");

    let err = commit_property(&cat2, &ident, "winner", "two")
        .await
        .expect_err("stale second commit to v3 must fail");
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(err.retryable());

    assert_eq!(
        std::fs::read(&v3).expect("re-read v3"),
        winner_bytes,
        "loser must not touch the winner file"
    );

    let loser = cat2.load_table(&ident).await.expect("loser loads");
    assert_eq!(
        loser.metadata_location().expect("loser pointer"),
        v2.as_str(),
        "losing catalog pointer stays at v2"
    );

    let winner = cat1.load_table(&ident).await.expect("winner loads");
    assert_eq!(table_property(&winner, "winner").as_deref(), Some("one"));
}

#[tokio::test]
async fn uuid_names_keep_overwrite_behaviour() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let file_io = FileIO::new_with_fs();

    let cat1 = new_local_catalog("one", &warehouse).await;
    let cat2 = new_local_catalog("two", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    let table_location = create_ns_and_source(&cat1, &ns).await;
    create_ns_and_source(&cat2, &ns).await;

    let uuid_base = MetadataLocation::new_with_table_location(&table_location).to_string();
    let base = cat1
        .load_table(&TableIdent::new(ns.clone(), "src".to_string()))
        .await
        .expect("load src");
    base.metadata()
        .write_to(&file_io, &uuid_base)
        .await
        .expect("seed uuid base");

    let ident = TableIdent::new(ns, "hive".to_string());
    cat1.register_table(&ident, uuid_base.clone())
        .await
        .expect("cat1 registers uuid base");
    cat2.register_table(&ident, uuid_base)
        .await
        .expect("cat2 registers uuid base");

    let first = commit_property(&cat1, &ident, "writer", "one")
        .await
        .expect("first uuid commit lands");
    let second = commit_property(&cat2, &ident, "writer", "two")
        .await
        .expect("second uuid commit keeps current behaviour");
    assert_ne!(first, second, "uuid commits mint distinct files");

    assert_eq!(
        table_property(
            &cat1.load_table(&ident).await.expect("cat1 loads"),
            "writer"
        )
        .as_deref(),
        Some("one")
    );
    assert_eq!(
        table_property(
            &cat2.load_table(&ident).await.expect("cat2 loads"),
            "writer"
        )
        .as_deref(),
        Some("two")
    );
}

#[tokio::test]
async fn hadoop_concurrent_commits_yield_exactly_one_winner() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let file_io = FileIO::new_with_fs();

    let cat1 = new_local_catalog("one", &warehouse).await;
    let cat2 = new_local_catalog("two", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    let table_location = create_ns_and_source(&cat1, &ns).await;
    create_ns_and_source(&cat2, &ns).await;

    let v2 = format!("{table_location}/metadata/v2.metadata.json");
    let base = cat1
        .load_table(&TableIdent::new(ns.clone(), "src".to_string()))
        .await
        .expect("load src");
    base.metadata()
        .write_to(&file_io, &v2)
        .await
        .expect("seed v2");

    let ident = TableIdent::new(ns, "hadoop".to_string());
    cat1.register_table(&ident, v2.clone())
        .await
        .expect("cat1 registers v2");
    cat2.register_table(&ident, v2)
        .await
        .expect("cat2 registers v2");

    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let (first, second) = tokio::join!(
        race_once(&cat1, &ident, "one", Arc::clone(&barrier)),
        race_once(&cat2, &ident, "two", Arc::clone(&barrier))
    );

    let winners = [&first, &second]
        .into_iter()
        .filter_map(|result| result.as_ref().ok())
        .collect::<Vec<_>>();
    assert_eq!(winners.len(), 1, "exactly one racer wins v3");

    let v3 = format!("{table_location}/metadata/v3.metadata.json");
    let raw = std::fs::read(&v3).expect("v3 exists");
    let parsed: serde_json::Value = serde_json::from_slice(&raw).expect("v3 parses");
    let props = parsed
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .expect("v3 has properties");
    assert_eq!(
        props.get("winner").and_then(serde_json::Value::as_str),
        Some(*winners[0]),
        "v3 carries the winner rows"
    );

    let loser = if first.is_err() { &first } else { &second };
    let loser_err = loser.as_ref().expect_err("loser fails");
    assert_eq!(loser_err.kind(), ErrorKind::CatalogCommitConflicts);
}

async fn apply_locally_property(
    catalog: &impl Catalog,
    ident: &TableIdent,
    key: &str,
    value: &str,
) -> Result<String, iceberg::Error> {
    let table = catalog.load_table(ident).await.expect("load base");
    let tx = Transaction::new(&table);
    let staged = tx
        .update_table_properties()
        .set(key.to_string(), value.to_string())
        .apply(tx)
        .expect("apply stages");
    let out = staged.apply_locally().await?;
    Ok(out.metadata_location().expect("location").to_string())
}

async fn seed_and_register(
    cat1: &impl Catalog,
    cat2: &impl Catalog,
    file_io: &FileIO,
    table_location: &str,
    ns: NamespaceIdent,
    name: &str,
) -> TableIdent {
    let v2 = format!("{table_location}/metadata/v2.metadata.json");
    let base = cat1
        .load_table(&TableIdent::new(ns.clone(), "src".to_string()))
        .await
        .expect("load src");
    base.metadata()
        .write_to(file_io, &v2)
        .await
        .expect("seed v2");
    let ident = TableIdent::new(ns, name.to_string());
    cat1.register_table(&ident, v2.clone())
        .await
        .expect("cat1 registers v2");
    cat2.register_table(&ident, v2)
        .await
        .expect("cat2 registers v2");
    ident
}

#[tokio::test]
async fn hadoop_gzip_sibling_v3_blocks_uncompressed_v3_commit() {
    for sibling in ["v3.gz.metadata.json", "v3.metadata.json.gz"] {
        let dir = tempfile::tempdir().expect("tempdir");
        let warehouse = dir.path().to_str().expect("utf8 path").to_string();
        let file_io = FileIO::new_with_fs();

        let cat1 = new_local_catalog("one", &warehouse).await;
        let ns = NamespaceIdent::new("ns".to_string());
        let table_location = create_ns_and_source(&cat1, &ns).await;

        let v2 = format!("{table_location}/metadata/v2.metadata.json");
        let base = cat1
            .load_table(&TableIdent::new(ns.clone(), "src".to_string()))
            .await
            .expect("load src");
        base.metadata()
            .write_to(&file_io, &v2)
            .await
            .expect("seed v2");
        let sibling_path = format!("{table_location}/metadata/{sibling}");
        std::fs::copy(&v2, &sibling_path).expect("seed sibling");
        let sibling_bytes = std::fs::read(&sibling_path).expect("read sibling");

        let ident = TableIdent::new(ns, "hadoop".to_string());
        cat1.register_table(&ident, v2.clone())
            .await
            .expect("register v2");

        let err = commit_property(&cat1, &ident, "writer", "one")
            .await
            .expect_err("sibling of same version must conflict");
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
        assert!(err.retryable());
        assert_eq!(
            std::fs::read(&sibling_path).expect("sibling intact"),
            sibling_bytes
        );
        assert!(
            !std::path::Path::new(&format!("{table_location}/metadata/v3.metadata.json")).exists(),
            "no uncompressed v3 may appear next to {sibling}"
        );
        let table = cat1.load_table(&ident).await.expect("loads");
        assert_eq!(table.metadata_location().expect("pointer"), v2.as_str());
    }
}

#[tokio::test]
async fn hadoop_apply_locally_collision_fails() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let file_io = FileIO::new_with_fs();

    let cat1 = new_local_catalog("one", &warehouse).await;
    let cat2 = new_local_catalog("two", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    let table_location = create_ns_and_source(&cat1, &ns).await;
    create_ns_and_source(&cat2, &ns).await;
    let ident = seed_and_register(&cat1, &cat2, &file_io, &table_location, ns, "hadoop").await;

    let v3 = apply_locally_property(&cat1, &ident, "writer", "one")
        .await
        .expect("first apply_locally lands");
    assert!(v3.ends_with("/metadata/v3.metadata.json"));
    let winner_bytes = std::fs::read(&v3).expect("read v3");

    let err = apply_locally_property(&cat2, &ident, "writer", "two")
        .await
        .expect_err("stale apply_locally to v3 must fail");
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(err.retryable());
    assert_eq!(
        std::fs::read(&v3).expect("re-read v3"),
        winner_bytes,
        "loser must not touch the winner file"
    );
}

#[tokio::test]
async fn orphan_v3_wedges_stale_pointer_loud_then_reregister_recovers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let file_io = FileIO::new_with_fs();

    let cat1 = new_local_catalog("one", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    let table_location = create_ns_and_source(&cat1, &ns).await;

    let v2 = format!("{table_location}/metadata/v2.metadata.json");
    let base = cat1
        .load_table(&TableIdent::new(ns.clone(), "src".to_string()))
        .await
        .expect("load src");
    base.metadata()
        .write_to(&file_io, &v2)
        .await
        .expect("seed v2");
    let ident = TableIdent::new(ns.clone(), "hadoop".to_string());
    cat1.register_table(&ident, v2.clone())
        .await
        .expect("register v2");

    let v3 = format!("{table_location}/metadata/v3.metadata.json");
    std::fs::copy(&v2, &v3).expect("plant orphan v3");
    let orphan_bytes = std::fs::read(&v3).expect("read orphan");

    let err = commit_property(&cat1, &ident, "writer", "one")
        .await
        .expect_err("orphan v3 wedges the stale pointer loud");
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert_eq!(std::fs::read(&v3).expect("orphan intact"), orphan_bytes);
    let stuck = cat1.load_table(&ident).await.expect("loads");
    assert_eq!(stuck.metadata_location().expect("pointer"), v2.as_str());

    let cat2 = new_local_catalog("two", &warehouse).await;
    create_ns_and_source(&cat2, &ns).await;
    cat2.register_table(&ident, v3)
        .await
        .expect("re-register at newest version");
    let v4 = commit_property(&cat2, &ident, "writer", "two")
        .await
        .expect("commit resumes after re-register");
    assert!(v4.ends_with("/metadata/v4.metadata.json"));
    let resumed = cat2.load_table(&ident).await.expect("loads");
    assert_eq!(table_property(&resumed, "writer").as_deref(), Some("two"));
}

#[tokio::test]
async fn hadoop_staged_replace_second_stager_fails_and_preserves_winner() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let file_io = FileIO::new_with_fs();

    let cat1 = new_local_catalog("one", &warehouse).await;
    let cat2 = new_local_catalog("two", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    let table_location = create_ns_and_source(&cat1, &ns).await;
    create_ns_and_source(&cat2, &ns).await;

    let v2 = format!("{table_location}/metadata/v2.metadata.json");
    let base = cat1
        .load_table(&TableIdent::new(ns.clone(), "src".to_string()))
        .await
        .expect("load src");
    base.metadata()
        .write_to(&file_io, &v2)
        .await
        .expect("seed v2");

    let ident = TableIdent::new(ns, "hadoop".to_string());
    let table1 = cat1
        .register_table(&ident, v2.clone())
        .await
        .expect("cat1 registers v2");
    let table2 = cat2
        .register_table(&ident, v2.clone())
        .await
        .expect("cat2 registers v2");

    let creation = || {
        TableCreation::builder()
            .name("hadoop".to_string())
            .schema(test_schema())
            .build()
    };
    let staged = StagedTableTransaction::begin_replace(&table1, creation())
        .await
        .expect("first replace stages");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert_eq!(
        staged_location,
        format!("{table_location}/metadata/v3.metadata.json"),
        "a Hadoop base must stage vN+1, not a uuid name"
    );
    staged.commit(&cat1).await.expect("publish replace");
    let winner_bytes = std::fs::read(&staged_location).expect("read staged v3");

    let err = match StagedTableTransaction::begin_replace(&table2, creation()).await {
        Ok(_) => panic!("a second stager onto the existing v3 must fail"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(err.retryable());
    assert_eq!(
        std::fs::read(&staged_location).expect("re-read v3"),
        winner_bytes,
        "loser must not overwrite the winner's staged file"
    );
    let loser = cat2.load_table(&ident).await.expect("loser loads");
    assert_eq!(
        loser.metadata_location().expect("loser pointer"),
        v2.as_str(),
        "losing catalog pointer stays at v2"
    );
}

#[tokio::test]
async fn hadoop_staged_replace_never_overwrites_a_live_next_version() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let file_io = FileIO::new_with_fs();

    let cat1 = new_local_catalog("one", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    let table_location = create_ns_and_source(&cat1, &ns).await;

    let v2 = format!("{table_location}/metadata/v2.metadata.json");
    let base = cat1
        .load_table(&TableIdent::new(ns.clone(), "src".to_string()))
        .await
        .expect("load src");
    base.metadata()
        .write_to(&file_io, &v2)
        .await
        .expect("seed v2");

    let ident = TableIdent::new(ns, "hadoop".to_string());
    let registered = cat1
        .register_table(&ident, v2.clone())
        .await
        .expect("register v2");

    let creation = TableCreation::builder()
        .name("hadoop".to_string())
        .schema(test_schema())
        .build();
    let staged = StagedTableTransaction::begin_replace(&registered, creation)
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert_eq!(
        staged_location,
        format!("{table_location}/metadata/v3.metadata.json")
    );

    let winner = iceberg::table::Table::builder()
        .identifier(ident.clone())
        .metadata(registered.metadata().clone())
        .metadata_location(staged_location.clone())
        .file_io(file_io.clone())
        .build()
        .expect("winner table");
    winner
        .metadata()
        .write_to(&file_io, &staged_location)
        .await
        .expect("winner writes v3");
    let winner_bytes = std::fs::read(&staged_location).expect("read winner v3");
    cat1.publish_replace_table(winner, Some(v2))
        .await
        .expect("winner CAS lands v3");

    let file = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{table_location}/data/f.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(4)
        .partition_spec_id(0)
        .partition(iceberg::spec::Struct::empty())
        .build()
        .expect("build data file");
    let err = staged
        .add_data_files(vec![file])
        .commit(&cat1)
        .await
        .expect_err("a replace losing the v3 slot must fail");
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(err.retryable());
    assert_eq!(
        std::fs::read(&staged_location).expect("re-read v3"),
        winner_bytes,
        "a losing replace must never overwrite a live Hadoop version file"
    );
    let current = cat1.load_table(&ident).await.expect("loads");
    assert_eq!(
        current.metadata_location().expect("pointer"),
        staged_location.as_str()
    );
    assert!(
        !std::path::Path::new(&format!("{table_location}/metadata/v4.metadata.json")).exists(),
        "a losing replace must not leave a v4 behind"
    );
}

#[tokio::test]
async fn uuid_staged_replace_bases_keep_distinct_names() {
    let dir = tempfile::tempdir().expect("tempdir");
    let warehouse = dir.path().to_str().expect("utf8 path").to_string();
    let file_io = FileIO::new_with_fs();

    let cat1 = new_local_catalog("one", &warehouse).await;
    let cat2 = new_local_catalog("two", &warehouse).await;
    let ns = NamespaceIdent::new("ns".to_string());
    let table_location = create_ns_and_source(&cat1, &ns).await;
    create_ns_and_source(&cat2, &ns).await;

    let uuid_base = MetadataLocation::new_with_table_location(&table_location).to_string();
    let base = cat1
        .load_table(&TableIdent::new(ns.clone(), "src".to_string()))
        .await
        .expect("load src");
    base.metadata()
        .write_to(&file_io, &uuid_base)
        .await
        .expect("seed uuid base");

    let ident = TableIdent::new(ns, "hive".to_string());
    let table1 = cat1
        .register_table(&ident, uuid_base.clone())
        .await
        .expect("cat1 registers uuid base");
    let table2 = cat2
        .register_table(&ident, uuid_base)
        .await
        .expect("cat2 registers uuid base");

    let creation = || {
        TableCreation::builder()
            .name("hive".to_string())
            .schema(test_schema())
            .build()
    };
    let first = StagedTableTransaction::begin_replace(&table1, creation())
        .await
        .expect("first replace stages");
    let second = StagedTableTransaction::begin_replace(&table2, creation())
        .await
        .expect("second replace stages");
    let first_loc = first
        .table()
        .metadata_location_result()
        .expect("first location")
        .to_string();
    let second_loc = second
        .table()
        .metadata_location_result()
        .expect("second location")
        .to_string();
    assert_ne!(first_loc, second_loc, "uuid staged names cannot collide");
    first.commit(&cat1).await.expect("first publishes");
    second.commit(&cat2).await.expect("second publishes");
}

mod update_schema_noop {
    use iceberg::spec::Literal;

    use super::*;

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
}
