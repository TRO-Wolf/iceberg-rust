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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use tokio::sync::Notify;

use super::{
    MEMORY_CATALOG_METADATA_NAMING, MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder,
};
use crate::io::{
    FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorageFactory, MemoryStorage,
    OutputFile, Storage, StorageConfig, StorageFactory,
};
use crate::spec::{
    NestedField, PrimitiveType, Schema, TableMetadata, TableMetadataBuilder, TableProperties, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{
    Catalog, CatalogBuilder, Error, ErrorKind, MetadataLocation, NamespaceIdent, Result,
    TableCreation, TableIdent,
};

const UUID_REGEX_STR: &str = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";

async fn load_catalog(warehouse: &TempDir, naming: Option<&str>) -> Result<MemoryCatalog> {
    load_catalog_with(
        Arc::new(LocalFsStorageFactory),
        warehouse.path().to_str().expect("utf8 path"),
        naming,
    )
    .await
}

async fn load_catalog_with(
    factory: Arc<dyn StorageFactory>,
    warehouse: &str,
    naming: Option<&str>,
) -> Result<MemoryCatalog> {
    let mut props = HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]);
    if let Some(naming) = naming {
        props.insert(
            MEMORY_CATALOG_METADATA_NAMING.to_string(),
            naming.to_string(),
        );
    }
    MemoryCatalogBuilder::default()
        .with_storage_factory(factory)
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

const INJECTED_HINT_FAILURE: &str = "injected failure after writing version-hint.text";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SteppingStorage {
    #[serde(skip, default = "memory_storage")]
    inner: Arc<dyn Storage>,
    #[serde(skip)]
    delayed_hint: Option<Arc<Notify>>,
    #[serde(skip)]
    failing_hint: Option<Arc<AtomicBool>>,
}

fn memory_storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::default())
}

impl Default for SteppingStorage {
    fn default() -> Self {
        Self {
            inner: memory_storage(),
            delayed_hint: None,
            failing_hint: None,
        }
    }
}

impl SteppingStorage {
    async fn step(&self) {
        tokio::task::yield_now().await;
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for SteppingStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        self.step().await;
        self.inner.exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        self.step().await;
        self.inner.metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        self.step().await;
        self.inner.read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        self.step().await;
        self.inner.reader(path).await
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        self.step().await;
        if let Some(entered) = &self.delayed_hint
            && path.ends_with("/version-hint.text")
            && bs.as_ref() == b"2"
        {
            entered.notify_one();
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        if let Some(failing) = &self.failing_hint
            && failing.load(Ordering::SeqCst)
            && path.ends_with("/metadata/version-hint.text")
        {
            self.inner.write(path, bs).await?;
            return Err(Error::new(ErrorKind::Unexpected, INJECTED_HINT_FAILURE));
        }
        self.inner.write(path, bs).await
    }

    async fn write_new(&self, path: &str, bs: Bytes) -> Result<()> {
        self.step().await;
        self.inner.write_new(path, bs).await
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        self.step().await;
        self.inner.writer(path).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.step().await;
        self.inner.delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        self.step().await;
        self.inner.delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        self.step().await;
        self.inner.list(prefix).await
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SteppingStorageFactory {
    #[serde(skip)]
    storage: SteppingStorage,
}

#[typetag::serde]
impl StorageFactory for SteppingStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(self.storage.clone()))
    }
}

async fn stepping_catalog(delayed_hint: Option<Arc<Notify>>) -> Arc<MemoryCatalog> {
    let factory = SteppingStorageFactory {
        storage: SteppingStorage {
            inner: memory_storage(),
            delayed_hint,
            failing_hint: None,
        },
    };
    let catalog = load_catalog_with(Arc::new(factory), "memory:///warehouse", Some("hadoop"))
        .await
        .expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    Arc::new(catalog)
}

fn wide_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema")
}

async fn create_with_schema(catalog: &MemoryCatalog, schema: Schema) -> Result<Table> {
    catalog
        .create_table(
            &ident().namespace,
            TableCreation::builder()
                .name(ident().name().to_string())
                .schema(schema)
                .build(),
        )
        .await
}

async fn read_bytes(catalog: &MemoryCatalog, path: &str) -> Bytes {
    catalog
        .file_io
        .new_input(path)
        .expect("input")
        .read()
        .await
        .expect("read")
}

fn pointer_version(location: &str) -> String {
    Regex::new(r"/metadata/v(\d+)\.metadata\.json$")
        .expect("regex")
        .captures(location)
        .unwrap_or_else(|| panic!("{location}"))[1]
        .to_string()
}

async fn assert_hint_matches_pointer(catalog: &MemoryCatalog, expected_version: &str) -> Table {
    let loaded = catalog.load_table(&ident()).await.expect("load");
    let pointer = location(&loaded);
    assert_eq!(pointer_version(&pointer), expected_version, "{pointer}");
    let hint = read_bytes(
        catalog,
        &format!("{}/version-hint.text", metadata_dir(&loaded)),
    )
    .await;
    assert_eq!(
        String::from_utf8_lossy(&hint),
        pointer_version(&pointer),
        "{pointer}"
    );
    loaded
}

#[tokio::test]
async fn test_hadoop_duplicate_create_keeps_registered_v1_bytes() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let created = create(&catalog, HashMap::new()).await.expect("create");
    let v1 = location(&created);
    let original = std::fs::read(&v1).expect("read v1");

    let err = create_with_schema(&catalog, wide_schema())
        .await
        .expect_err("duplicate");
    assert_eq!(err.kind(), ErrorKind::TableAlreadyExists);
    assert_eq!(std::fs::read(&v1).expect("reread v1"), original);
    assert_eq!(hint(&created).as_deref(), Some("1"));

    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(location(&loaded), v1);
    assert_eq!(
        loaded.metadata().current_schema(),
        created.metadata().current_schema()
    );
    assert_eq!(loaded.metadata().uuid(), created.metadata().uuid());
}

#[tokio::test]
async fn test_hadoop_concurrent_create_registers_winner_bytes() {
    for offset in 0..8 {
        assert_concurrent_create_registers_winner_bytes(offset).await;
    }
}

async fn assert_concurrent_create_registers_winner_bytes(offset: usize) {
    let catalog = stepping_catalog(None).await;
    let first = tokio::spawn({
        let catalog = catalog.clone();
        async move { create_with_schema(&catalog, schema()).await }
    });
    let second = tokio::spawn({
        let catalog = catalog.clone();
        async move {
            for _ in 0..offset {
                tokio::task::yield_now().await;
            }
            create_with_schema(&catalog, wide_schema()).await
        }
    });
    let results = [
        first.await.expect("join first"),
        second.await.expect("join second"),
    ];

    let winners: Vec<&Table> = results.iter().filter_map(|r| r.as_ref().ok()).collect();
    assert_eq!(winners.len(), 1, "offset {offset}: {results:?}");
    let winner = winners[0];
    let loser = results
        .iter()
        .find_map(|r| r.as_ref().err())
        .expect("loser");
    assert!(
        matches!(
            loser.kind(),
            ErrorKind::TableAlreadyExists | ErrorKind::CatalogCommitConflicts
        ),
        "offset {offset}: {loser}"
    );

    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(location(&loaded), location(winner), "offset {offset}");
    let bytes = read_bytes(&catalog, &location(&loaded)).await;
    let stored: TableMetadata = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(stored.uuid(), winner.metadata().uuid(), "offset {offset}");
    assert_eq!(
        stored.current_schema(),
        winner.metadata().current_schema(),
        "offset {offset}"
    );
    assert_hint_matches_pointer(&catalog, "1").await;
}

#[tokio::test]
async fn test_hadoop_hint_follows_pointer_when_older_hint_write_finishes_last() {
    let entered = Arc::new(Notify::new());
    let catalog = stepping_catalog(Some(entered.clone())).await;
    create_with_schema(&catalog, schema())
        .await
        .expect("create");

    let slow = tokio::spawn({
        let catalog = catalog.clone();
        async move { location(&commit_property(&catalog, "a").await) }
    });
    entered.notified().await;
    let fast = commit_property(&catalog, "b").await;
    let slow = slow.await.expect("join");

    assert!(slow.ends_with("/metadata/v2.metadata.json"), "{slow}");
    assert!(
        location(&fast).ends_with("/metadata/v3.metadata.json"),
        "{}",
        location(&fast)
    );
    assert_hint_matches_pointer(&catalog, "3").await;
}

#[tokio::test]
async fn test_hadoop_racing_commits_from_one_base_end_with_hint_at_pointer() {
    let catalog = stepping_catalog(None).await;
    let retry = HashMap::from([
        ("commit.retry.num-retries".to_string(), "100".to_string()),
        ("commit.retry.min-wait-ms".to_string(), "1".to_string()),
        ("commit.retry.max-wait-ms".to_string(), "5".to_string()),
    ]);
    catalog
        .create_table(
            &ident().namespace,
            TableCreation::builder()
                .name(ident().name().to_string())
                .schema(schema())
                .properties(retry)
                .build(),
        )
        .await
        .expect("create");
    let base = catalog.load_table(&ident()).await.expect("base");

    let writers = 6;
    let handles: Vec<_> = (0..writers)
        .map(|writer| {
            let catalog = catalog.clone();
            let base = base.clone();
            tokio::spawn(async move {
                let tx = Transaction::new(&base);
                tx.update_table_properties()
                    .set(format!("writer-{writer}"), writer.to_string())
                    .apply(tx)
                    .expect("apply")
                    .commit(catalog.as_ref())
                    .await
                    .expect("commit")
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("join");
    }

    let loaded = assert_hint_matches_pointer(&catalog, &(writers + 1).to_string()).await;
    for writer in 0..writers {
        assert_eq!(
            loaded
                .metadata()
                .properties()
                .get(&format!("writer-{writer}"))
                .map(String::as_str),
            Some(writer.to_string().as_str())
        );
    }
}

#[tokio::test]
async fn test_default_naming_register_vn_pointer_writes_no_hint() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, None).await.expect("load");
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
    let v3 = format!("{}/metadata/v3.metadata.json", metadata.location());
    metadata
        .write_to(&catalog.file_io, &v3)
        .await
        .expect("write");

    let registered = catalog
        .register_table(&ident(), v3.clone())
        .await
        .expect("register");
    assert_eq!(location(&registered), v3);

    let committed = commit_property(&catalog, "a").await;
    assert!(
        location(&committed).ends_with("/metadata/v4.metadata.json"),
        "{}",
        location(&committed)
    );
    assert!(!table_location.join("metadata/version-hint.text").exists());
    assert_eq!(hint(&committed), None);
}

#[tokio::test]
async fn test_hadoop_create_with_failed_hint_registers_v1_and_commits_on() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let hint_dir = warehouse.path().join("ns/t/metadata/version-hint.text");
    std::fs::create_dir_all(&hint_dir).expect("hint dir");

    let created = create(&catalog, HashMap::new()).await.expect("create");
    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(location(&loaded), location(&created));
    assert!(location(&loaded).ends_with("/metadata/v1.metadata.json"));
    assert!(Path::new(&location(&loaded)).is_file());
    let err = create(&catalog, HashMap::new())
        .await
        .expect_err("duplicate");
    assert_eq!(err.kind(), ErrorKind::TableAlreadyExists);

    std::fs::remove_dir(&hint_dir).expect("remove hint dir");
    let committed = commit_property(&catalog, "a").await;
    assert!(location(&committed).ends_with("/metadata/v2.metadata.json"));
    assert_eq!(hint(&committed).as_deref(), Some("2"));
}

#[tokio::test]
async fn test_hadoop_create_racing_register_keeps_hint_at_registered_pointer() {
    let mut create_wins = 0;
    for round in 0..32 {
        if assert_create_racing_register(round % 2 == 0, round / 2).await {
            create_wins += 1;
        }
    }
    assert!(
        (1..32).contains(&create_wins),
        "create won {create_wins} of 32"
    );
}

async fn assert_create_racing_register(delay_register: bool, yields: usize) -> bool {
    let warehouse = TempDir::new().expect("tempdir");
    let factory = SteppingStorageFactory {
        storage: SteppingStorage {
            inner: LocalFsStorageFactory
                .build(&StorageConfig::default())
                .expect("local fs"),
            delayed_hint: None,
            failing_hint: None,
        },
    };
    let catalog = load_catalog_with(
        Arc::new(factory),
        warehouse.path().to_str().expect("utf8"),
        Some("hadoop"),
    )
    .await
    .expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    let table_dir = warehouse.path().join("ns/t");
    let metadata = TableMetadataBuilder::from_table_creation(
        TableCreation::builder()
            .name("t".to_string())
            .location(table_dir.to_str().expect("utf8").to_string())
            .schema(wide_schema())
            .build(),
    )
    .expect("builder")
    .build()
    .expect("metadata")
    .metadata;
    let v3 = format!("{}/metadata/v3.metadata.json", metadata.location());
    metadata
        .write_to(&catalog.file_io, &v3)
        .await
        .expect("write v3");

    let catalog = Arc::new(catalog);
    let pause = move |delayed: bool| async move {
        if delayed {
            for _ in 0..yields {
                tokio::task::yield_now().await;
            }
        }
    };
    let created = tokio::spawn({
        let catalog = catalog.clone();
        async move {
            pause(!delay_register).await;
            create_with_schema(&catalog, schema()).await
        }
    });
    let registered = tokio::spawn({
        let catalog = catalog.clone();
        let v3 = v3.clone();
        async move {
            pause(delay_register).await;
            catalog.register_table(&ident(), v3).await
        }
    });
    let created = created.await.expect("join create");
    let registered = registered.await.expect("join register");
    let context = format!("delay_register {delay_register} yields {yields}");

    assert_ne!(created.is_ok(), registered.is_ok(), "{context}");
    let pointer = location(&catalog.load_table(&ident()).await.expect("load"));
    let metadata_dir = table_dir.join("metadata");
    if let Ok(hint) = std::fs::read_to_string(metadata_dir.join("version-hint.text")) {
        assert_eq!(hint, pointer_version(&pointer), "{context}: {pointer}");
    }
    if created.is_ok() {
        assert!(pointer.ends_with("/metadata/v1.metadata.json"), "{context}");
    } else {
        assert_eq!(pointer, v3, "{context}");
        assert!(!metadata_dir.join("v1.metadata.json").exists(), "{context}");
    }
    created.is_ok()
}

async fn failing_hint_catalog(warehouse: &TempDir, failing: Arc<AtomicBool>) -> MemoryCatalog {
    let factory = SteppingStorageFactory {
        storage: SteppingStorage {
            inner: LocalFsStorageFactory
                .build(&StorageConfig::default())
                .expect("local fs"),
            delayed_hint: None,
            failing_hint: Some(failing),
        },
    };
    load_catalog_with(
        Arc::new(factory),
        warehouse.path().to_str().expect("utf8"),
        Some("hadoop"),
    )
    .await
    .expect("load")
}

async fn assert_create_survives_partial_hint(pre_existing_hint: Option<&str>) {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = failing_hint_catalog(&warehouse, Arc::new(AtomicBool::new(true))).await;
    let metadata_dir = warehouse.path().join("ns/t/metadata");
    if let Some(bytes) = pre_existing_hint {
        std::fs::create_dir_all(&metadata_dir).expect("metadata dir");
        std::fs::write(metadata_dir.join("version-hint.text"), bytes).expect("hint");
    }

    let created = create(&catalog, HashMap::new()).await.expect("create");
    assert!(catalog.table_exists(&ident()).await.expect("exists"));
    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(location(&loaded), location(&created));
    assert!(location(&loaded).ends_with("/metadata/v1.metadata.json"));
    assert!(metadata_dir.join("v1.metadata.json").is_file());
    assert_eq!(hint(&loaded).as_deref(), Some("1"));
}

#[tokio::test]
async fn test_hadoop_create_succeeds_when_hint_write_fails_after_bytes() {
    assert_create_survives_partial_hint(None).await;
}

#[tokio::test]
async fn test_hadoop_create_succeeds_over_pre_existing_hint_when_hint_write_fails() {
    assert_create_survives_partial_hint(Some("7")).await;
}

fn renamed() -> TableIdent {
    TableIdent::new(ident().namespace, "u".to_string())
}

async fn create_named(catalog: &MemoryCatalog, table: &TableIdent) -> Result<Table> {
    catalog
        .create_table(
            &table.namespace,
            TableCreation::builder()
                .name(table.name().to_string())
                .schema(schema())
                .build(),
        )
        .await
}

#[tokio::test]
async fn test_hadoop_rename_refused_without_state_change() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    create(&catalog, HashMap::new()).await.expect("create");

    let err = catalog
        .rename_table(&ident(), &renamed())
        .await
        .expect_err("refused");
    assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    assert_eq!(err.message(), "Cannot rename Hadoop tables");
    assert!(catalog.table_exists(&ident()).await.expect("exists"));
    assert!(!catalog.table_exists(&renamed()).await.expect("exists"));
    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert!(
        location(&loaded).ends_with("/metadata/v1.metadata.json"),
        "{}",
        location(&loaded)
    );
}

#[tokio::test]
async fn test_hadoop_create_at_target_name_after_refused_rename() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let original = create(&catalog, HashMap::new()).await.expect("create");
    catalog
        .rename_table(&ident(), &renamed())
        .await
        .expect_err("refused");

    let created = create_named(&catalog, &renamed()).await.expect("create u");
    assert!(
        location(&created).ends_with("/ns/u/metadata/v1.metadata.json"),
        "{}",
        location(&created)
    );
    assert_eq!(hint(&created).as_deref(), Some("1"));

    let loaded = catalog.load_table(&ident()).await.expect("load t");
    assert_eq!(location(&loaded), location(&original));
    assert_eq!(loaded.metadata().uuid(), original.metadata().uuid());
    assert_eq!(hint(&loaded).as_deref(), Some("1"));
}

#[tokio::test]
async fn test_uuid_rename_then_create_at_old_name_succeeds() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, None).await.expect("load");
    let original = create(&catalog, HashMap::new()).await.expect("create");

    catalog
        .rename_table(&ident(), &renamed())
        .await
        .expect("rename");
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    let moved = catalog.load_table(&renamed()).await.expect("load u");
    assert_eq!(location(&moved), location(&original));

    let recreated = create_named(&catalog, &ident()).await.expect("create t");
    assert_uuid_named(&location(&recreated), "00000");
    assert_ne!(location(&recreated), location(&original));
}

#[tokio::test]
async fn test_hadoop_rename_of_missing_source_is_refused_first() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");

    let err = catalog
        .rename_table(&ident(), &renamed())
        .await
        .expect_err("refused");
    assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    assert_eq!(err.message(), "Cannot rename Hadoop tables");
    assert!(!catalog.table_exists(&renamed()).await.expect("exists"));
}
