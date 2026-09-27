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
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::io::{
    FileIO, FileInfo, FileMetadata, FileRead, FileWrite, InputFile, MemoryStorage, OutputFile,
    Storage, StorageConfig, StorageFactory,
};
use crate::memory::{
    MEMORY_CATALOG_METADATA_NAMING, MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder,
};
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, NestedField, Operation,
    PrimitiveType, Schema, Struct, TableMetadataBuilder, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, StagedTableTransaction, Transaction};
use crate::{
    Catalog, CatalogBuilder, Error, ErrorKind, NamespaceIdent, Result, TableCreation, TableIdent,
};

fn schema() -> Schema {
    Schema::builder()
        .with_fields(vec![Arc::new(NestedField::required(
            1,
            "id",
            Type::Primitive(PrimitiveType::Long),
        ))])
        .build()
        .expect("schema")
}

async fn table_at(
    file_io: &FileIO,
    ident: &TableIdent,
    table_location: &str,
    metadata_location: &str,
) -> Table {
    let creation = TableCreation::builder()
        .name(ident.name().to_string())
        .location(table_location.to_string())
        .schema(schema())
        .build();
    let metadata = TableMetadataBuilder::from_table_creation(creation)
        .expect("metadata builder")
        .build()
        .expect("metadata")
        .metadata;
    metadata
        .write_to(file_io, metadata_location)
        .await
        .expect("write metadata");
    Table::builder()
        .identifier(ident.clone())
        .metadata(metadata)
        .metadata_location(metadata_location.to_string())
        .file_io(file_io.clone())
        .build()
        .expect("table")
}

fn replace_creation(ident: &TableIdent) -> TableCreation {
    TableCreation::builder()
        .name(ident.name().to_string())
        .schema(schema())
        .build()
}

#[tokio::test]
async fn replace_stages_next_version_after_a_hive_named_pointer() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let base_location = format!(
        "{table_location}/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json"
    );
    let table = table_at(&file_io, &ident, table_location, &base_location).await;

    let staged = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert!(
        staged_location.starts_with(&format!("{table_location}/metadata/00004-")),
        "the staged file must continue the base pointer's version, got {staged_location}"
    );
    assert!(staged_location.ends_with(".metadata.json"));
    assert!(
        !file_io.exists(&staged_location).await.expect("exists"),
        "the staged metadata file must be written once at commit, not at begin"
    );
}

#[tokio::test]
async fn replace_stages_next_version_after_a_hadoop_named_pointer() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!("{table_location}/metadata/v7.metadata.json"),
    )
    .await;

    let staged = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert_eq!(
        staged_location,
        format!("{table_location}/metadata/v8.metadata.json"),
        "a Hadoop-named pointer must continue vN naming, got {staged_location}"
    );
}

#[tokio::test]
async fn concurrent_replace_from_a_hadoop_pointer_fails_on_exclusive_create() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!("{table_location}/metadata/v3.metadata.json"),
    )
    .await;

    let staged = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("first begin replace");
    let first = staged
        .table()
        .metadata_location_result()
        .expect("first staged location")
        .to_string();
    assert_eq!(first, format!("{table_location}/metadata/v4.metadata.json"));
    table
        .metadata()
        .write_commit_metadata(&file_io, &first)
        .await
        .expect("winner lands v4");
    let winner = file_io
        .new_input(&first)
        .expect("open winner")
        .read()
        .await
        .expect("read winner bytes");

    let err = match StagedTableTransaction::begin_replace(&table, replace_creation(&ident)).await {
        Ok(_) => panic!("a second replace onto the existing v4 must fail"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(err.retryable(), "the collision must be retryable: {err}");
    assert_eq!(
        file_io
            .new_input(&first)
            .expect("reopen winner")
            .read()
            .await
            .expect("reread winner"),
        winner,
        "the losing replace must not overwrite the winner's file"
    );
}

#[tokio::test]
async fn concurrent_replaces_from_a_uuid_pointer_stage_distinct_files() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!(
            "{table_location}/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json"
        ),
    )
    .await;

    let first = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("first begin replace")
        .table()
        .metadata_location_result()
        .expect("first staged location")
        .to_string();
    let second = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("second begin replace")
        .table()
        .metadata_location_result()
        .expect("second staged location")
        .to_string();

    assert!(
        first.starts_with(&format!("{table_location}/metadata/00004-")),
        "uuid-named base keeps version continuation, got {first}"
    );
    assert!(
        second.starts_with(&format!("{table_location}/metadata/00004-")),
        "uuid-named base keeps version continuation, got {second}"
    );
    assert_ne!(first, second, "uuid names cannot collide");
    assert!(!file_io.exists(&first).await.expect("first exists"));
    assert!(!file_io.exists(&second).await.expect("second exists"));
}

#[tokio::test]
async fn replace_restarts_versioning_when_base_pointer_does_not_parse() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!("{table_location}/metadata/final.metadata.json"),
    )
    .await;

    let staged = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location");
    assert!(
        staged_location.starts_with(&format!("{table_location}/metadata/00000-")),
        "an unparsable base pointer keeps the v0 restart, got {staged_location}"
    );
}

#[tokio::test]
async fn replace_from_a_hadoop_pointer_refuses_write_metadata_path() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!("{table_location}/metadata/v7.metadata.json"),
    )
    .await;

    let creation = TableCreation::builder()
        .name(ident.name().to_string())
        .schema(schema())
        .properties(HashMap::from([(
            "write.metadata.path".to_string(),
            "memory://alt-meta".to_string(),
        )]))
        .build();
    let err = match StagedTableTransaction::begin_replace(&table, creation).await {
        Ok(_) => panic!("a hadoop-pointer replace carrying write.metadata.path must fail"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.message()
            .contains("Hadoop path-based tables cannot relocate metadata"),
        "the refusal must carry Java's message, got: {err}"
    );
    assert!(
        !file_io
            .exists(format!("{table_location}/metadata/v8.metadata.json"))
            .await
            .expect("v8 exists check"),
        "no staged metadata file may be written"
    );
    assert!(
        file_io
            .list("memory://alt-meta")
            .await
            .expect("list alt-meta")
            .is_empty(),
        "nothing may be written under write.metadata.path"
    );
}

#[tokio::test]
async fn apply_locally_on_a_hadoop_pointer_refuses_write_metadata_path() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!("{table_location}/metadata/v7.metadata.json"),
    )
    .await;

    let tx = Transaction::new(&table);
    let tx = tx
        .update_table_properties()
        .set(
            "write.metadata.path".to_string(),
            "memory://alt-meta".to_string(),
        )
        .apply(tx)
        .expect("apply");
    let err = match tx.apply_locally().await {
        Ok(_) => panic!("a hadoop-pointer local apply carrying write.metadata.path must fail"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.message()
            .contains("Hadoop path-based tables cannot relocate metadata"),
        "the refusal must carry Java's message, got: {err}"
    );
    assert!(
        !file_io
            .exists(format!("{table_location}/metadata/v8.metadata.json"))
            .await
            .expect("v8 exists check"),
        "no metadata file may be written"
    );
    assert!(
        file_io
            .list("memory://alt-meta")
            .await
            .expect("list alt-meta")
            .is_empty(),
        "nothing may be written under write.metadata.path"
    );
}

#[tokio::test]
async fn replace_stages_uncompressed_next_version_after_a_gzip_hadoop_pointer() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let table_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        table_location,
        &format!("{table_location}/metadata/v7.gz.metadata.json"),
    )
    .await;

    let staged = StagedTableTransaction::begin_replace(&table, replace_creation(&ident))
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert_eq!(
        staged_location,
        format!("{table_location}/metadata/v8.metadata.json"),
        "a gzip Hadoop pointer must stage the next uncompressed version, got {staged_location}"
    );
    assert!(
        !file_io.exists(&staged_location).await.expect("exists"),
        "a Hadoop staged target is written once at commit, not at begin"
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NoOverwriteStorage {
    #[serde(skip, default = "shared_memory_storage")]
    inner: Arc<dyn Storage>,
    #[serde(skip, default = "default_write_counts")]
    writes: Arc<Mutex<HashMap<String, usize>>>,
}

fn shared_memory_storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::default())
}

fn default_write_counts() -> Arc<Mutex<HashMap<String, usize>>> {
    Arc::new(Mutex::new(HashMap::new()))
}

#[async_trait]
#[typetag::serde]
impl Storage for NoOverwriteStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        self.inner.metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        self.inner.read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        self.inner.reader(path).await
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        if self.inner.exists(path).await? {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("second write of an existing path refused: {path}"),
            ));
        }
        self.inner.write(path, bs).await?;
        *self
            .writes
            .lock()
            .expect("write counts")
            .entry(path.to_string())
            .or_insert(0) += 1;
        Ok(())
    }

    async fn write_new(&self, path: &str, bs: Bytes) -> Result<()> {
        self.inner.write_new(path, bs).await?;
        *self
            .writes
            .lock()
            .expect("write counts")
            .entry(path.to_string())
            .or_insert(0) += 1;
        Ok(())
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        self.inner.writer(path).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner.delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        self.inner.delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        self.inner.list(prefix).await
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NoOverwriteStorageFactory {
    #[serde(skip, default = "shared_memory_storage")]
    inner: Arc<dyn Storage>,
    #[serde(skip, default = "default_write_counts")]
    writes: Arc<Mutex<HashMap<String, usize>>>,
}

#[typetag::serde]
impl StorageFactory for NoOverwriteStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(NoOverwriteStorage {
            inner: self.inner.clone(),
            writes: self.writes.clone(),
        }))
    }
}

fn single_write_storage() -> (
    NoOverwriteStorageFactory,
    Arc<Mutex<HashMap<String, usize>>>,
) {
    let writes = default_write_counts();
    let factory = NoOverwriteStorageFactory {
        inner: shared_memory_storage(),
        writes: writes.clone(),
    };
    (factory, writes)
}

async fn single_write_catalog(factory: NoOverwriteStorageFactory) -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(factory))
        .load(
            "mem",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                "memory://warehouse".to_string(),
            )]),
        )
        .await
        .expect("load memory catalog")
}

async fn single_write_hadoop_catalog(factory: NoOverwriteStorageFactory) -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(factory))
        .load(
            "mem",
            HashMap::from([
                (
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    "memory://warehouse".to_string(),
                ),
                (
                    MEMORY_CATALOG_METADATA_NAMING.to_string(),
                    "hadoop".to_string(),
                ),
            ]),
        )
        .await
        .expect("load hadoop memory catalog")
}

async fn seed_single_write_table(catalog: &MemoryCatalog) -> Table {
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    catalog
        .create_namespace(ident.namespace(), HashMap::new())
        .await
        .expect("create namespace");
    catalog
        .create_table(
            ident.namespace(),
            TableCreation::builder()
                .name(ident.name().to_string())
                .schema(schema())
                .build(),
        )
        .await
        .expect("create table")
}

fn version_data_file(path: &str, records: u64) -> DataFile {
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

fn replace_operation(table: &Table) -> Operation {
    table
        .metadata()
        .current_snapshot()
        .expect("replace commit must leave a current snapshot")
        .summary()
        .operation
        .clone()
}

fn assert_metadata_keys_written_once(
    writes: &Arc<Mutex<HashMap<String, usize>>>,
    expected: &[String],
) {
    let counts = writes.lock().expect("write counts");
    for path in expected {
        assert_eq!(
            counts.get(path),
            Some(&1),
            "metadata key must be written exactly once: {path}"
        );
    }
    for (path, count) in counts.iter() {
        if path.ends_with(".metadata.json") {
            assert_eq!(*count, 1, "metadata key written {count} times: {path}");
        }
    }
}

async fn assert_two_metadata_versions(table: &Table, metadata_dir: &str) {
    let listed = table
        .file_io()
        .list(metadata_dir)
        .await
        .expect("list metadata dir");
    let versions: Vec<_> = listed
        .iter()
        .filter(|entry| entry.location.ends_with(".metadata.json"))
        .collect();
    assert_eq!(
        versions.len(),
        2,
        "one replace must leave exactly the seed version plus the staged version, got {:?}",
        versions
            .iter()
            .map(|entry| &entry.location)
            .collect::<Vec<_>>()
    );
}

async fn assert_metadata_version_count(table: &Table, metadata_dir: &str, expected: usize) {
    let listed = table
        .file_io()
        .list(metadata_dir)
        .await
        .expect("list metadata dir");
    let versions: Vec<_> = listed
        .iter()
        .filter(|entry| entry.location.ends_with(".metadata.json"))
        .collect();
    assert_eq!(
        versions.len(),
        expected,
        "expected {expected} metadata versions, got {:?}",
        versions
            .iter()
            .map(|entry| &entry.location)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn replace_with_files_writes_each_metadata_key_once() {
    let (factory, writes) = single_write_storage();
    let catalog = single_write_catalog(factory).await;
    let table = seed_single_write_table(&catalog).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();
    let metadata_dir = format!("{}/metadata", table.metadata().location());

    let staged =
        StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
            .await
            .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert!(
        staged_location.starts_with(&format!("{metadata_dir}/00001-")),
        "the staged target must be version 00001, got {staged_location}"
    );
    assert!(
        !staged
            .table()
            .file_io()
            .exists(&staged_location)
            .await
            .expect("probe staged target"),
        "begin_replace must not write the staged target; commit writes it once"
    );

    let committed = staged
        .with_replace_write(true)
        .add_data_files(vec![version_data_file(
            "memory://warehouse/ns/t/data/r.parquet",
            7,
        )])
        .commit(&catalog)
        .await
        .expect("replace commit under a no-overwrite store");
    assert_eq!(
        committed.metadata_location_result().expect("location"),
        staged_location.as_str()
    );
    assert_eq!(replace_operation(&committed), Operation::Overwrite);
    assert_metadata_keys_written_once(&writes, &[base_location, staged_location]);
    assert_two_metadata_versions(&committed, &metadata_dir).await;
}

#[tokio::test]
async fn replace_write_without_files_writes_each_metadata_key_once() {
    let (factory, writes) = single_write_storage();
    let catalog = single_write_catalog(factory).await;
    let table = seed_single_write_table(&catalog).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();
    let metadata_dir = format!("{}/metadata", table.metadata().location());

    let staged =
        StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
            .await
            .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();

    let committed = staged
        .with_replace_write(true)
        .commit(&catalog)
        .await
        .expect("empty replace commit under a no-overwrite store");
    assert_eq!(
        committed.metadata_location_result().expect("location"),
        staged_location.as_str()
    );
    assert_eq!(replace_operation(&committed), Operation::Delete);
    assert_metadata_keys_written_once(&writes, &[base_location, staged_location]);
    assert_two_metadata_versions(&committed, &metadata_dir).await;
}

#[tokio::test]
async fn empty_plain_replace_writes_staged_target_once_at_commit() {
    let (factory, writes) = single_write_storage();
    let catalog = single_write_catalog(factory).await;
    let table = seed_single_write_table(&catalog).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();
    let metadata_dir = format!("{}/metadata", table.metadata().location());

    let staged =
        StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
            .await
            .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location")
        .to_string();
    assert!(
        !staged
            .table()
            .file_io()
            .exists(&staged_location)
            .await
            .expect("probe staged target"),
        "begin_replace must not write the staged target; commit writes it once"
    );

    let committed = staged
        .commit(&catalog)
        .await
        .expect("empty plain replace commit under a no-overwrite store");
    assert_eq!(
        committed.metadata_location_result().expect("location"),
        staged_location.as_str()
    );
    assert!(
        committed.metadata().current_snapshot().is_none(),
        "an empty plain replace must leave no current snapshot"
    );
    assert_metadata_keys_written_once(&writes, &[base_location, staged_location]);
    assert_two_metadata_versions(&committed, &metadata_dir).await;
}

#[tokio::test]
async fn replace_restarts_versioning_under_a_different_caller_location() {
    let file_io = FileIO::new_with_memory();
    let ident = TableIdent::new(NamespaceIdent::new("ns".into()), "t".into());
    let base_location = "memory://wh/ns/t";
    let table = table_at(
        &file_io,
        &ident,
        base_location,
        &format!(
            "{base_location}/metadata/00003-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json"
        ),
    )
    .await;
    let moved_location = "memory://wh/ns/relocated";
    let creation = TableCreation::builder()
        .name(ident.name().to_string())
        .location(moved_location.to_string())
        .schema(schema())
        .build();

    let staged = StagedTableTransaction::begin_replace(&table, creation)
        .await
        .expect("begin replace");
    let staged_location = staged
        .table()
        .metadata_location_result()
        .expect("staged location");
    assert!(
        staged_location.starts_with(&format!("{moved_location}/metadata/00000-")),
        "a relocated replace must stage v0 under the new location, got {staged_location}"
    );
}

#[tokio::test]
async fn retry_after_a_lost_uuid_pointer_cas_stages_a_fresh_key() {
    let (factory, writes) = single_write_storage();
    let catalog = single_write_catalog(factory).await;
    let table = seed_single_write_table(&catalog).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();
    let metadata_dir = format!("{}/metadata", table.metadata().location());

    let loser = StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
        .await
        .expect("begin loser");
    let loser_location = loser
        .table()
        .metadata_location_result()
        .expect("loser staged location")
        .to_string();
    let winner =
        StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
            .await
            .expect("begin winner");
    let winner_location = winner
        .table()
        .metadata_location_result()
        .expect("winner staged location")
        .to_string();
    assert_ne!(
        loser_location, winner_location,
        "concurrent replaces must stage distinct keys"
    );

    let committed_winner = winner.commit(&catalog).await.expect("winner commits");
    assert_eq!(
        committed_winner
            .metadata_location_result()
            .expect("winner pointer"),
        winner_location.as_str()
    );

    let err = match loser.commit(&catalog).await {
        Ok(_) => panic!("a replace on a moved pointer must fail the CAS"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(
        err.retryable(),
        "a lost pointer CAS must be retryable: {err}"
    );

    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after conflict");
    assert_eq!(
        reloaded
            .metadata_location_result()
            .expect("reloaded pointer"),
        winner_location.as_str()
    );
    let retry =
        StagedTableTransaction::begin_replace(&reloaded, replace_creation(table.identifier()))
            .await
            .expect("re-begin after conflict");
    let retry_location = retry
        .table()
        .metadata_location_result()
        .expect("retry staged location")
        .to_string();
    assert_ne!(
        retry_location, loser_location,
        "a retry must stage a fresh key, not reuse the loser's written key"
    );
    assert_ne!(
        retry_location, winner_location,
        "a retry must stage a fresh key, not reuse the winner's key"
    );
    assert!(
        retry_location.starts_with(&format!("{metadata_dir}/00002-")),
        "the retry must continue the winner's version, got {retry_location}"
    );

    let committed_retry = retry.commit(&catalog).await.expect("retry commits");
    assert_eq!(
        committed_retry
            .metadata_location_result()
            .expect("retry pointer"),
        retry_location.as_str()
    );
    assert_metadata_keys_written_once(&writes, &[
        base_location,
        winner_location,
        loser_location,
        retry_location,
    ]);
    assert_metadata_version_count(&committed_retry, &metadata_dir, 4).await;
}

#[tokio::test]
async fn concurrent_uuid_replaces_keep_the_winners_file_byte_identical() {
    let (factory, writes) = single_write_storage();
    let catalog = single_write_catalog(factory).await;
    let table = seed_single_write_table(&catalog).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();

    let loser = StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
        .await
        .expect("begin loser")
        .add_data_files(vec![version_data_file(
            "memory://warehouse/ns/t/data/loser.parquet",
            3,
        )]);
    let loser_location = loser
        .table()
        .metadata_location_result()
        .expect("loser staged location")
        .to_string();
    let winner =
        StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
            .await
            .expect("begin winner")
            .add_data_files(vec![version_data_file(
                "memory://warehouse/ns/t/data/winner.parquet",
                5,
            )]);
    let winner_location = winner
        .table()
        .metadata_location_result()
        .expect("winner staged location")
        .to_string();
    assert_ne!(
        loser_location, winner_location,
        "concurrent replaces must stage distinct keys"
    );

    let committed = winner.commit(&catalog).await.expect("winner commits");
    let winner_bytes = committed
        .file_io()
        .new_input(&winner_location)
        .expect("open winner")
        .read()
        .await
        .expect("read winner bytes");

    let err = match loser.commit(&catalog).await {
        Ok(_) => panic!("a replace on a moved pointer must fail the CAS"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(
        err.retryable(),
        "a lost pointer CAS must be retryable: {err}"
    );
    assert_eq!(
        committed
            .file_io()
            .new_input(&winner_location)
            .expect("reopen winner")
            .read()
            .await
            .expect("reread winner"),
        winner_bytes,
        "the loser must not touch the winner's file"
    );
    let current = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after race");
    assert_eq!(
        current.metadata_location_result().expect("pointer"),
        winner_location.as_str()
    );
    assert_metadata_keys_written_once(&writes, &[base_location, winner_location, loser_location]);
}

#[tokio::test]
async fn hadoop_retry_after_a_lost_next_version_stages_a_fresh_version() {
    let (factory, writes) = single_write_storage();
    let catalog = single_write_hadoop_catalog(factory).await;
    let table = seed_single_write_table(&catalog).await;
    let base_location = table
        .metadata_location_result()
        .expect("base location")
        .to_string();
    let metadata_dir = format!("{}/metadata", table.metadata().location());
    assert_eq!(
        base_location,
        format!("{metadata_dir}/v1.metadata.json"),
        "a hadoop seed must publish v1, got {base_location}"
    );

    let loser = StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
        .await
        .expect("begin loser");
    let loser_location = loser
        .table()
        .metadata_location_result()
        .expect("loser staged location")
        .to_string();
    assert_eq!(
        loser_location,
        format!("{metadata_dir}/v2.metadata.json"),
        "a hadoop replace must stage v2, got {loser_location}"
    );
    let winner =
        StagedTableTransaction::begin_replace(&table, replace_creation(table.identifier()))
            .await
            .expect("begin winner");
    let winner_location = winner
        .table()
        .metadata_location_result()
        .expect("winner staged location")
        .to_string();
    assert_eq!(
        winner_location, loser_location,
        "hadoop replaces from one base share one deterministic target"
    );

    let committed = winner.commit(&catalog).await.expect("winner lands v2");
    let winner_bytes = committed
        .file_io()
        .new_input(&winner_location)
        .expect("open winner")
        .read()
        .await
        .expect("read winner bytes");

    let err = match loser.commit(&catalog).await {
        Ok(_) => panic!("a replace onto a landed v2 must fail"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert!(
        err.retryable(),
        "a lost next version must be retryable: {err}"
    );
    assert_eq!(
        committed
            .file_io()
            .new_input(&winner_location)
            .expect("reopen winner")
            .read()
            .await
            .expect("reread winner"),
        winner_bytes,
        "the loser must not touch the winner's file"
    );

    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after conflict");
    assert_eq!(
        reloaded
            .metadata_location_result()
            .expect("reloaded pointer"),
        winner_location.as_str()
    );
    let retry =
        StagedTableTransaction::begin_replace(&reloaded, replace_creation(table.identifier()))
            .await
            .expect("re-begin after conflict");
    let retry_location = retry
        .table()
        .metadata_location_result()
        .expect("retry staged location")
        .to_string();
    assert_eq!(
        retry_location,
        format!("{metadata_dir}/v3.metadata.json"),
        "the retry must stage v3, got {retry_location}"
    );
    let committed_retry = retry.commit(&catalog).await.expect("retry commits");
    assert_eq!(
        committed_retry
            .metadata_location_result()
            .expect("retry pointer"),
        retry_location.as_str()
    );
    assert_metadata_keys_written_once(&writes, &[base_location, winner_location, retry_location]);
    assert_metadata_version_count(&committed_retry, &metadata_dir, 3).await;
}
