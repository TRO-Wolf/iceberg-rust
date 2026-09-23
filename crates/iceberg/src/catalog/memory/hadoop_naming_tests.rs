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
use std::path::Path;
use std::sync::Arc;

use regex::Regex;
use tempfile::TempDir;

use super::{
    MEMORY_CATALOG_METADATA_NAMING, MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder,
};
use crate::io::LocalFsStorageFactory;
use crate::spec::{
    NestedField, PrimitiveType, Schema, TableMetadataBuilder, TableProperties, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{
    Catalog, CatalogBuilder, ErrorKind, MetadataLocation, NamespaceIdent, Result, TableCreation,
    TableIdent,
};

const UUID_REGEX_STR: &str = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";

async fn load_catalog(warehouse: &TempDir, naming: Option<&str>) -> Result<MemoryCatalog> {
    let mut props = HashMap::from([(
        MEMORY_CATALOG_WAREHOUSE.to_string(),
        warehouse.path().to_str().expect("utf8 path").to_string(),
    )]);
    if let Some(naming) = naming {
        props.insert(
            MEMORY_CATALOG_METADATA_NAMING.to_string(),
            naming.to_string(),
        );
    }
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load("memory", props)
        .await
}

fn schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("schema")
}

fn ident() -> TableIdent {
    TableIdent::new(NamespaceIdent::new("ns".to_string()), "t".to_string())
}

async fn create(catalog: &MemoryCatalog, properties: HashMap<String, String>) -> Result<Table> {
    let ident = ident();
    if !catalog.namespace_exists(&ident.namespace).await? {
        catalog
            .create_namespace(&ident.namespace, HashMap::new())
            .await?;
    }
    catalog
        .create_table(
            &ident.namespace,
            TableCreation::builder()
                .name(ident.name().to_string())
                .schema(schema())
                .properties(properties)
                .build(),
        )
        .await
}

async fn commit_property(catalog: &MemoryCatalog, value: &str) -> Table {
    let table = catalog.load_table(&ident()).await.expect("load");
    let tx = Transaction::new(&table);
    tx.update_table_properties()
        .set("k".to_string(), value.to_string())
        .apply(tx)
        .expect("apply")
        .commit(catalog)
        .await
        .expect("commit")
}

fn location(table: &Table) -> String {
    table.metadata_location().expect("location").to_string()
}

fn metadata_dir(table: &Table) -> String {
    format!("{}/metadata", table.metadata().location())
}

fn hint(table: &Table) -> Option<String> {
    std::fs::read_to_string(format!("{}/version-hint.text", metadata_dir(table))).ok()
}

fn assert_uuid_named(location: &str, version: &str) {
    let regex = Regex::new(&format!(
        "/metadata/{version}-{UUID_REGEX_STR}\\.metadata\\.json$"
    ))
    .expect("regex");
    assert!(regex.is_match(location), "{location}");
}

#[tokio::test]
async fn test_hadoop_create_writes_v1_and_hint() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let table = create(&catalog, HashMap::new()).await.expect("create");

    let location = location(&table);
    assert!(
        location.ends_with("/metadata/v1.metadata.json"),
        "{location}"
    );
    assert!(Path::new(&location).is_file());
    assert_eq!(hint(&table).as_deref(), Some("1"));
}

#[tokio::test]
async fn test_hadoop_three_commits_reach_v4_and_hint() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let table = create(&catalog, HashMap::new()).await.expect("create");
    let mut last = table;
    for value in ["a", "b", "c"] {
        last = commit_property(&catalog, value).await;
    }

    let location = location(&last);
    assert!(
        location.ends_with("/metadata/v4.metadata.json"),
        "{location}"
    );
    for version in 1..=4 {
        let file = format!("{}/v{version}.metadata.json", metadata_dir(&last));
        assert!(Path::new(&file).is_file(), "{file}");
    }
    assert_eq!(hint(&last).as_deref(), Some("4"));
}

#[tokio::test]
async fn test_hadoop_load_round_trips_vn_location() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let created = create(&catalog, HashMap::new()).await.expect("create");
    let loaded = catalog.load_table(&ident()).await.expect("load v1");
    assert_eq!(location(&loaded), location(&created));

    let committed = commit_property(&catalog, "a").await;
    commit_property(&catalog, "b").await;
    let loaded = catalog.load_table(&ident()).await.expect("load v3");
    assert!(location(&committed).ends_with("/metadata/v2.metadata.json"));
    assert!(location(&loaded).ends_with("/metadata/v3.metadata.json"));
    assert_eq!(
        loaded.metadata().properties().get("k").map(String::as_str),
        Some("b")
    );
}

async fn assert_uuid_mode(naming: Option<&str>) {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, naming).await.expect("load");
    let table = create(&catalog, HashMap::new()).await.expect("create");
    assert_uuid_named(&location(&table), "00000");

    let committed = commit_property(&catalog, "a").await;
    assert_uuid_named(&location(&committed), "00001");
    assert_eq!(hint(&committed), None);
}

#[tokio::test]
async fn test_default_naming_is_uuid_without_hint() {
    assert_uuid_mode(None).await;
}

#[tokio::test]
async fn test_explicit_uuid_naming_matches_default() {
    assert_uuid_mode(Some("uuid")).await;
}

#[tokio::test]
async fn test_near_miss_naming_values_refused() {
    for value in ["Hadoop", "HADOOP", "v", ""] {
        let warehouse = TempDir::new().expect("tempdir");
        let err = load_catalog(&warehouse, Some(value))
            .await
            .expect_err(value);
        assert_eq!(err.kind(), ErrorKind::DataInvalid, "{value:?}");
        assert!(
            err.message().contains(MEMORY_CATALOG_METADATA_NAMING),
            "{}",
            err.message()
        );
        assert!(
            err.message().contains(&format!("{value:?}")),
            "{}",
            err.message()
        );
    }
}

#[tokio::test]
async fn test_hadoop_refuses_write_metadata_path_before_writing() {
    let warehouse = TempDir::new().expect("tempdir");
    let relocated = warehouse.path().join("relocated");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let properties = HashMap::from([(
        TableProperties::PROPERTY_WRITE_METADATA_LOCATION.to_string(),
        relocated.to_str().expect("utf8").to_string(),
    )]);

    let err = create(&catalog, properties).await.expect_err("refused");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        err.message(),
        "Hadoop path-based tables cannot relocate metadata"
    );
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    assert!(!warehouse.path().join("ns/t/metadata").exists());
    assert!(!relocated.exists());
}

#[tokio::test]
async fn test_hadoop_register_uuid_location_stays_uuid() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    let table_location = warehouse.path().join("registered");
    let metadata = TableMetadataBuilder::from_table_creation(
        TableCreation::builder()
            .name("t".to_string())
            .location(table_location.to_str().expect("utf8").to_string())
            .schema(schema())
            .build(),
    )
    .expect("builder")
    .build()
    .expect("metadata")
    .metadata;
    let uuid_location = MetadataLocation::for_metadata(&metadata)
        .expect("location")
        .to_string();
    metadata
        .write_to(&catalog.file_io, &uuid_location)
        .await
        .expect("write");

    let registered = catalog
        .register_table(&ident(), uuid_location.clone())
        .await
        .expect("register");
    assert_eq!(location(&registered), uuid_location);

    let committed = commit_property(&catalog, "a").await;
    assert_uuid_named(&location(&committed), "00001");
    assert_eq!(hint(&committed), None);
}
