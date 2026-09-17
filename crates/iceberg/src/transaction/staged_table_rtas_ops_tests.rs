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

use super::*;
use crate::io::{FileIOBuilder, LocalFsStorageFactory};
use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, NestedField, Operation, PrimitiveType,
    Schema, Struct, Type,
};
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, CatalogBuilder};

fn rtas_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "name",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .unwrap()
}

fn rtas_data_file(path: &str, records: u64) -> DataFile {
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

async fn rtas_catalog(warehouse: &str) -> (impl Catalog, FileIO) {
    let factory = Arc::new(LocalFsStorageFactory);
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(factory.clone())
        .load(
            "mem",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .unwrap();
    let file_io = FileIOBuilder::new(factory).build();
    (catalog, file_io)
}

async fn rtas_seed_table(
    catalog: &impl Catalog,
    ns: &NamespaceIdent,
    name: &str,
    files: Vec<DataFile>,
) -> Table {
    let table = catalog
        .create_table(
            ns,
            TableCreation::builder()
                .name(name.to_string())
                .schema(rtas_schema())
                .build(),
        )
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

fn rtas_summary(table: &Table) -> HashMap<String, String> {
    table
        .metadata()
        .current_snapshot()
        .expect("replace commit must leave a current snapshot")
        .summary()
        .additional_properties
        .clone()
}

fn rtas_operation(table: &Table) -> Operation {
    table
        .metadata()
        .current_snapshot()
        .expect("replace commit must leave a current snapshot")
        .summary()
        .operation
        .clone()
}

#[tokio::test]
async fn rtas_replace_existing_with_files_records_overwrite() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = rtas_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let seeded = rtas_seed_table(
        &catalog,
        &ns,
        "orders",
        vec![
            rtas_data_file(&format!("{warehouse}/old/a.parquet"), 700),
            rtas_data_file(&format!("{warehouse}/old/b.parquet"), 700),
            rtas_data_file(&format!("{warehouse}/old/c.parquet"), 600),
        ],
    )
    .await;
    let loaded = catalog.load_table(seeded.identifier()).await.unwrap();
    assert_eq!(rtas_operation(&loaded), Operation::Append);

    let creation = TableCreation::builder()
        .name("orders".into())
        .schema(rtas_schema())
        .build();
    let staged = StagedTableTransaction::begin_replace(&loaded, creation)
        .await
        .unwrap()
        .with_replace_write(true)
        .add_data_files(vec![
            rtas_data_file(&format!("{warehouse}/new/a.parquet"), 40),
            rtas_data_file(&format!("{warehouse}/new/b.parquet"), 60),
        ]);
    let committed = staged.commit(&catalog).await.unwrap();

    assert_eq!(rtas_operation(&committed), Operation::Overwrite);
    let summary = rtas_summary(&committed);
    assert_eq!(summary.get("added-data-files").map(String::as_str), Some("2"));
    assert_eq!(summary.get("added-records").map(String::as_str), Some("100"));
    assert_eq!(summary.get("total-data-files").map(String::as_str), Some("2"));
    assert_eq!(summary.get("total-records").map(String::as_str), Some("100"));
    assert!(!summary.contains_key("deleted-data-files"), "{summary:?}");
    assert!(!summary.contains_key("deleted-records"), "{summary:?}");
    assert!(summary.contains_key("changed-partition-count"), "{summary:?}");
}

#[tokio::test]
async fn rtas_create_new_with_files_records_overwrite() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, file_io) = rtas_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let ident = TableIdent::new(ns, "orders".into());
    let location = format!("{warehouse}/sales/orders");
    let creation = TableCreation::builder()
        .name("orders".into())
        .location(location.clone())
        .schema(rtas_schema())
        .build();
    let staged = StagedTableTransaction::begin_create(file_io, ident, creation)
        .await
        .unwrap()
        .with_replace_write(true)
        .add_data_files(vec![rtas_data_file(&format!("{location}/data/f.parquet"), 10)]);
    let committed = staged.commit(&catalog).await.unwrap();

    assert_eq!(rtas_operation(&committed), Operation::Overwrite);
    let summary = rtas_summary(&committed);
    assert_eq!(summary.get("added-data-files").map(String::as_str), Some("1"));
    assert_eq!(summary.get("added-records").map(String::as_str), Some("10"));
    assert_eq!(summary.get("total-data-files").map(String::as_str), Some("1"));
    assert_eq!(summary.get("total-records").map(String::as_str), Some("10"));
    assert!(!summary.contains_key("deleted-data-files"), "{summary:?}");
    assert!(!summary.contains_key("deleted-records"), "{summary:?}");
}

#[tokio::test]
async fn rtas_replace_without_files_records_delete() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, _) = rtas_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let seeded = rtas_seed_table(
        &catalog,
        &ns,
        "orders",
        vec![rtas_data_file(&format!("{warehouse}/old/a.parquet"), 5)],
    )
    .await;
    let loaded = catalog.load_table(seeded.identifier()).await.unwrap();

    let creation = TableCreation::builder()
        .name("orders".into())
        .schema(rtas_schema())
        .build();
    let staged = StagedTableTransaction::begin_replace(&loaded, creation)
        .await
        .unwrap()
        .with_replace_write(true);
    let committed = staged.commit(&catalog).await.unwrap();

    assert_eq!(rtas_operation(&committed), Operation::Delete);
    let summary = rtas_summary(&committed);
    assert!(!summary.contains_key("added-data-files"), "{summary:?}");
    assert!(!summary.contains_key("added-records"), "{summary:?}");
    assert_eq!(summary.get("total-data-files").map(String::as_str), Some("0"));
    assert_eq!(summary.get("total-records").map(String::as_str), Some("0"));
    assert_eq!(
        summary.get("changed-partition-count").map(String::as_str),
        Some("0")
    );
}

#[tokio::test]
async fn rtas_create_new_without_files_records_delete() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, file_io) = rtas_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let ident = TableIdent::new(ns, "orders".into());
    let location = format!("{warehouse}/sales/orders");
    let creation = TableCreation::builder()
        .name("orders".into())
        .location(location)
        .schema(rtas_schema())
        .build();
    let staged = StagedTableTransaction::begin_create(file_io, ident, creation)
        .await
        .unwrap()
        .with_replace_write(true);
    let committed = staged.commit(&catalog).await.unwrap();

    assert_eq!(rtas_operation(&committed), Operation::Delete);
    let summary = rtas_summary(&committed);
    assert!(!summary.contains_key("added-data-files"), "{summary:?}");
    assert!(!summary.contains_key("added-records"), "{summary:?}");
    assert_eq!(summary.get("total-data-files").map(String::as_str), Some("0"));
    assert_eq!(summary.get("total-records").map(String::as_str), Some("0"));
}

#[tokio::test]
async fn create_without_replace_write_stays_append() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, file_io) = rtas_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let ident = TableIdent::new(ns, "orders".into());
    let location = format!("{warehouse}/sales/orders");
    let creation = TableCreation::builder()
        .name("orders".into())
        .location(location.clone())
        .schema(rtas_schema())
        .build();
    let staged = StagedTableTransaction::begin_create(file_io, ident, creation)
        .await
        .unwrap()
        .add_data_files(vec![rtas_data_file(&format!("{location}/data/f.parquet"), 4)]);
    let committed = staged.commit(&catalog).await.unwrap();

    assert_eq!(rtas_operation(&committed), Operation::Append);
}

#[tokio::test]
async fn create_empty_without_flag_commits_no_snapshot() {
    let tmp = TempDir::new().unwrap();
    let warehouse = tmp.path().to_string_lossy().to_string();
    let (catalog, file_io) = rtas_catalog(&warehouse).await;
    let ns = NamespaceIdent::new("sales".into());
    catalog.create_namespace(&ns, HashMap::new()).await.unwrap();
    let ident = TableIdent::new(ns, "orders".into());
    let location = format!("{warehouse}/sales/orders");
    let creation = TableCreation::builder()
        .name("orders".into())
        .location(location)
        .schema(rtas_schema())
        .build();
    let staged = StagedTableTransaction::begin_create(file_io, ident.clone(), creation)
        .await
        .unwrap();
    let committed = staged.commit(&catalog).await.unwrap();

    assert!(committed.metadata().current_snapshot().is_none());
}
