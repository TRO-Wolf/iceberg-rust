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

use tempfile::TempDir;

use super::{
    MEMORY_CATALOG_METADATA_NAMING, MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder,
};
use crate::io::LocalFsStorageFactory;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
    Struct, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, StagedTableTransaction, Transaction};
use crate::{
    Catalog, CatalogBuilder, NamespaceIdent, Result, TableCreation, TableIdent, TableMetadataCache,
};

pub(super) async fn catalog(
    warehouse: &TempDir,
    naming: &str,
    cache: Option<Arc<TableMetadataCache>>,
) -> MemoryCatalog {
    let props = HashMap::from([
        (
            MEMORY_CATALOG_WAREHOUSE.to_string(),
            warehouse_path(warehouse),
        ),
        (
            MEMORY_CATALOG_METADATA_NAMING.to_string(),
            naming.to_string(),
        ),
    ]);
    let builder =
        MemoryCatalogBuilder::default().with_storage_factory(Arc::new(LocalFsStorageFactory));
    let builder = match cache {
        Some(cache) => builder.with_table_metadata_cache(cache),
        None => builder,
    };
    let catalog = builder
        .load("memory", props)
        .await
        .expect("load memory catalog");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("create namespace");
    catalog
}

fn warehouse_path(warehouse: &TempDir) -> String {
    warehouse
        .path()
        .to_str()
        .expect("utf8 warehouse")
        .to_string()
}

pub(super) fn table_location(warehouse: &TempDir) -> String {
    format!("{}/ns/t", warehouse_path(warehouse))
}

pub(super) fn ident() -> TableIdent {
    TableIdent::new(NamespaceIdent::new("ns".to_string()), "t".to_string())
}

pub(super) fn id_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("id schema")
}

pub(super) fn wide_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "n", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("wide schema")
}

pub(super) fn creation(schema: Schema) -> TableCreation {
    TableCreation::builder()
        .name(ident().name().to_string())
        .schema(schema)
        .build()
}

pub(super) fn location(table: &Table) -> String {
    table.metadata_location().expect("location").to_string()
}

fn data_file(table: &Table, name: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!(
            "{}/data/{name}.parquet",
            table.metadata().location()
        ))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition(Struct::empty())
        .partition_spec_id(table.metadata().default_partition_spec_id())
        .build()
        .expect("build data file")
}

pub(super) async fn insert(catalog: &MemoryCatalog, name: &str) -> Result<Table> {
    let table = catalog.load_table(&ident()).await?;
    let file = data_file(&table, name);
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files(vec![file])
        .apply(tx)?
        .commit(catalog)
        .await
}

pub(super) async fn begin_replace(
    catalog: &MemoryCatalog,
    schema: Schema,
    name: Option<&str>,
) -> Result<StagedTableTransaction> {
    let table = catalog.load_table(&ident()).await?;
    let staged = StagedTableTransaction::begin_replace(&table, creation(schema)).await?;
    Ok(match name {
        Some(name) => {
            let file = data_file(staged.table(), name);
            staged.add_data_files(vec![file]).with_replace_write(true)
        }
        None => staged,
    })
}

pub(super) async fn replace(
    catalog: &MemoryCatalog,
    schema: Schema,
    name: Option<&str>,
) -> Result<Table> {
    begin_replace(catalog, schema, name)
        .await?
        .commit(catalog)
        .await
}

pub(super) async fn create_as(
    catalog: &MemoryCatalog,
    warehouse: &TempDir,
    name: &str,
) -> Result<Table> {
    let staged = StagedTableTransaction::begin_create(
        catalog.file_io.clone(),
        ident(),
        TableCreation::builder()
            .name(ident().name().to_string())
            .location(table_location(warehouse))
            .schema(id_schema())
            .build(),
    )
    .await?;
    let file = data_file(staged.table(), name);
    staged.add_data_files(vec![file]).commit(catalog).await
}

async fn cached(catalog: &MemoryCatalog, cache: &TableMetadataCache, location: &str) -> bool {
    cache
        .lookup(&catalog.cache_scope, location, None)
        .await
        .is_some()
}

async fn assert_replace_moves_the_cache_entry(naming: &str) {
    let warehouse = TempDir::new().expect("tempdir");
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = catalog(&warehouse, naming, Some(cache.clone())).await;
    catalog
        .create_table(&ident().namespace, creation(id_schema()))
        .await
        .expect("create");
    let base = insert(&catalog, "a").await.expect("insert");
    let base_location = location(&base);
    assert!(
        cached(&catalog, &cache, &base_location).await,
        "{naming}: the insert must seed {base_location}"
    );

    let published = replace(&catalog, wide_schema(), Some("b"))
        .await
        .expect("replace");
    let published_location = location(&published);
    assert_ne!(published_location, base_location, "{naming}");
    assert!(
        !cached(&catalog, &cache, &base_location).await,
        "{naming}: the replace left the base entry {base_location}"
    );
    let entry = cache
        .lookup(&catalog.cache_scope, &published_location, None)
        .await
        .unwrap_or_else(|| panic!("{naming}: the replace must put {published_location}"));
    assert!(
        Arc::ptr_eq(&entry, &published.metadata_ref()),
        "{naming}: the cached entry must be the published metadata"
    );

    cache.reset_stats();
    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(cache.stats().body_fetches, 0, "{naming}");
    assert_eq!(
        loaded.metadata().uuid(),
        published.metadata().uuid(),
        "{naming}"
    );
}

#[tokio::test]
async fn test_replace_evicts_the_base_and_puts_the_published_location() {
    assert_replace_moves_the_cache_entry("hadoop").await;
    assert_replace_moves_the_cache_entry("uuid").await;
}

#[tokio::test]
async fn test_hadoop_drop_evicts_every_version_of_the_chain() {
    let warehouse = TempDir::new().expect("tempdir");
    let cache = Arc::new(TableMetadataCache::new());
    let catalog = catalog(&warehouse, "hadoop", Some(cache.clone())).await;
    catalog
        .create_table(&ident().namespace, creation(id_schema()))
        .await
        .expect("create");
    insert(&catalog, "a").await.expect("insert a");
    replace(&catalog, wide_schema(), Some("b"))
        .await
        .expect("replace");
    let table = insert(&catalog, "c").await.expect("insert c");
    let current = location(&table);
    assert!(current.ends_with("/metadata/v4.metadata.json"), "{current}");
    let (dir, _) = current.rsplit_once('/').expect("metadata dir");
    let chain: Vec<String> = (1..=4)
        .map(|version| format!("{dir}/v{version}.metadata.json"))
        .collect();
    let other = format!(
        "{}/ns/other/metadata/v1.metadata.json",
        warehouse_path(&warehouse)
    );
    for path in chain.iter().chain([&other]) {
        cache
            .put(&catalog.cache_scope, path, table.metadata_ref(), None, None)
            .await;
    }

    catalog.drop_table(&ident()).await.expect("drop");

    for path in &chain {
        assert!(
            !cached(&catalog, &cache, path).await,
            "the drop left {path}"
        );
    }
    assert!(
        cached(&catalog, &cache, &other).await,
        "a drop must leave another table's entry"
    );
}

#[path = "hadoop_cache_differential_tests.rs"]
mod differential_tests;
