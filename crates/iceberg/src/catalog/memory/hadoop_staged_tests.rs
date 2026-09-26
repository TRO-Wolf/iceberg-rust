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
use crate::io::{FileIOBuilder, LocalFsStorageFactory};
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
    Struct, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, StagedTableTransaction, Transaction};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, Result, TableCreation, TableIdent};

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

fn staged_creation(table_location: &str) -> TableCreation {
    TableCreation::builder()
        .name(ident().name().to_string())
        .location(table_location.to_string())
        .schema(schema())
        .build()
}

fn data_file(path: &str, records: u64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(records)
        .partition(Struct::empty())
        .partition_spec_id(0)
        .build()
        .expect("build data file")
}

fn location(table: &Table) -> String {
    table.metadata_location().expect("location").to_string()
}

fn metadata_dir(table: &Table) -> String {
    format!("{}/metadata", table.metadata().location())
}

fn hint_path(table: &Table) -> String {
    format!("{}/version-hint.text", metadata_dir(table))
}

fn hint(table: &Table) -> String {
    std::fs::read_to_string(hint_path(table)).expect("read version-hint.text")
}

#[track_caller]
fn assert_absent(path: &Path) {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        other => panic!("{} must not exist: {other:?}", path.display()),
    }
}

#[track_caller]
fn assert_no_hint(table: &Table) {
    assert_absent(Path::new(&hint_path(table)));
}

#[track_caller]
fn assert_no_uuid_staged_files(metadata_dir: &str) {
    let entries = std::fs::read_dir(metadata_dir).expect("read metadata dir");
    for entry in entries {
        let name = entry
            .expect("dir entry")
            .file_name()
            .into_string()
            .expect("utf8 name");
        assert!(
            !name.starts_with("00000-"),
            "staged uuid file left behind: {metadata_dir}/{name}"
        );
    }
}

fn assert_uuid_named(location: &str, version: &str) {
    let regex = Regex::new(&format!(
        "/metadata/{version}-{UUID_REGEX_STR}\\.metadata\\.json$"
    ))
    .expect("regex");
    assert!(regex.is_match(location), "{location}");
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

#[tokio::test]
async fn test_hadoop_staged_create_publishes_v1_and_hint() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    let table_location = warehouse.path().join("ns/t");
    let staged_file_io = FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build();
    let staged = StagedTableTransaction::begin_create(
        staged_file_io,
        ident(),
        staged_creation(table_location.to_str().expect("utf8")),
    )
    .await
    .expect("begin create");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert_uuid_named(&staged_location, "00000");

    let published = staged.commit(&catalog).await.expect("commit");
    let published_location = location(&published);
    assert!(
        published_location.ends_with("/metadata/v1.metadata.json"),
        "{published_location}"
    );
    assert!(Path::new(&published_location).is_file());
    assert_eq!(hint(&published), "1");
    assert_absent(Path::new(&staged_location));
    assert_no_uuid_staged_files(&metadata_dir(&published));

    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(location(&loaded), published_location);

    let committed = commit_property(&catalog, "a").await;
    assert!(
        location(&committed).ends_with("/metadata/v2.metadata.json"),
        "{}",
        location(&committed)
    );
    assert_eq!(hint(&committed), "2");
}

#[tokio::test]
async fn test_hadoop_staged_create_with_files_publishes_v1_and_hint() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    let table_location = warehouse.path().join("ns/t");
    let staged_file_io = FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build();
    let staged = StagedTableTransaction::begin_create(
        staged_file_io,
        ident(),
        staged_creation(table_location.to_str().expect("utf8")),
    )
    .await
    .expect("begin create");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    let data_path = format!("{}/data/f1.parquet", table_location.display());

    let published = staged
        .add_data_files(vec![data_file(&data_path, 2)])
        .commit(&catalog)
        .await
        .expect("commit");
    let published_location = location(&published);
    assert!(
        published_location.ends_with("/metadata/v1.metadata.json"),
        "{published_location}"
    );
    assert_eq!(hint(&published), "1");
    assert_absent(Path::new(&staged_location));
    assert_no_uuid_staged_files(&metadata_dir(&published));

    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(location(&loaded), published_location);
    assert!(
        loaded.metadata().current_snapshot().is_some(),
        "the staged files must be appended at publish"
    );
}

#[tokio::test]
async fn test_hadoop_staged_replace_from_v2_publishes_v3_and_hint() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    catalog
        .create_table(
            &ident().namespace,
            TableCreation::builder()
                .name(ident().name().to_string())
                .schema(schema())
                .build(),
        )
        .await
        .expect("create");
    let base = commit_property(&catalog, "a").await;
    assert!(
        location(&base).ends_with("/metadata/v2.metadata.json"),
        "{}",
        location(&base)
    );

    let staged = StagedTableTransaction::begin_replace(
        &base,
        TableCreation::builder()
            .name(ident().name().to_string())
            .schema(schema())
            .build(),
    )
    .await
    .expect("begin replace");
    let published = staged.commit(&catalog).await.expect("commit");
    let published_location = location(&published);
    assert!(
        published_location.ends_with("/metadata/v3.metadata.json"),
        "{published_location}"
    );
    assert_eq!(hint(&published), "3");
    for version in 1..=3 {
        let file = format!("{}/v{version}.metadata.json", metadata_dir(&published));
        assert!(Path::new(&file).is_file(), "{file}");
    }

    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(location(&loaded), published_location);
}

#[tokio::test]
async fn test_hadoop_staged_create_then_two_commits_reaches_v3() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    let table_location = warehouse.path().join("ns/t");
    let staged_file_io = FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build();
    let staged = StagedTableTransaction::begin_create(
        staged_file_io,
        ident(),
        staged_creation(table_location.to_str().expect("utf8")),
    )
    .await
    .expect("begin create");

    staged.commit(&catalog).await.expect("commit");
    commit_property(&catalog, "a").await;
    let last = commit_property(&catalog, "b").await;
    assert!(
        location(&last).ends_with("/metadata/v3.metadata.json"),
        "{}",
        location(&last)
    );
    assert_eq!(hint(&last), "3");
}

async fn assert_uuid_staged_create(naming: Option<&str>) {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, naming).await.expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    let table_location = warehouse.path().join("ns/t");
    let staged_file_io = FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build();
    let staged = StagedTableTransaction::begin_create(
        staged_file_io,
        ident(),
        staged_creation(table_location.to_str().expect("utf8")),
    )
    .await
    .expect("begin create");

    let published = staged.commit(&catalog).await.expect("commit");
    assert_uuid_named(&location(&published), "00000");
    assert_no_hint(&published);
    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(location(&loaded), location(&published));
}

#[tokio::test]
async fn test_uuid_staged_create_stays_uuid_without_hint() {
    assert_uuid_staged_create(None).await;
    assert_uuid_staged_create(Some("uuid")).await;
}
