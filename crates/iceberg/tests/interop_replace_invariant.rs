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
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, NestedField, PrimitiveType, Schema, SortOrder, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation, TableIdent};
use serde::Deserialize;

const EXPECTED_IDS: [i64; 3] = [10, 20, 30];
const EXPECTED_DATA: [&str; 3] = ["a", "b", "c"];

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ScanRow {
    id: i64,
    data: Option<String>,
}

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REPLACE_INVARIANT_GEN_DIR").map(PathBuf::from)
}

fn d1_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REPLACE_INVARIANT_DIR").map(PathBuf::from)
}

fn schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build {id, data} schema")
}

async fn create_table(catalog: &impl Catalog, table_location: &str, name: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    if !catalog
        .namespace_exists(&namespace)
        .await
        .expect("namespace_exists")
    {
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace");
    }
    let creation = TableCreation::builder()
        .name(name.to_string())
        .location(table_location.to_string())
        .schema(schema())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create replace-invariant table")
}

async fn write_rows(table: &Table, n: usize) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let iceberg_schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(iceberg_schema).expect("schema to arrow"));
    let ids: Vec<i64> = (1..=n).map(|i| 10 * i as i64).collect();
    let all_data = ["a", "b", "c", "d", "e"];
    let data: Vec<&str> = all_data[..n].to_vec();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(data)) as ArrayRef,
    ])
    .expect("build data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "rinvar".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        iceberg_schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .unpartitioned()
        .build(None)
        .await
        .expect("build unpartitioned writer");
    writer.write(batch).await.expect("write batch");
    writer
        .close()
        .await
        .expect("close writer")
        .into_iter()
        .next()
        .expect("one data file")
}

async fn live_rows(table: &Table) -> Vec<ScanRow> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect batches");
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id column");
        let data = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("data column");
        for i in 0..batch.num_rows() {
            rows.push(ScanRow {
                id: ids.value(i),
                data: if data.is_null(i) {
                    None
                } else {
                    Some(data.value(i).to_string())
                },
            });
        }
    }
    rows.sort_by_key(|row| row.id);
    rows
}

fn expected_rows() -> Vec<ScanRow> {
    EXPECTED_IDS
        .iter()
        .zip(EXPECTED_DATA.iter())
        .map(|(id, data)| ScanRow {
            id: *id,
            data: Some((*data).to_string()),
        })
        .collect()
}

async fn write_final_metadata(table: &Table) {
    let location = table.metadata().location();
    let final_metadata_path = format!("{location}/metadata/final.metadata.json");
    table
        .metadata()
        .clone()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");
}

#[tokio::test]
async fn test_replace_invariant_gen_rust_writes_valid_and_refuses_invalid() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping replace-invariant GEN — set ICEBERG_INTEROP_REPLACE_INVARIANT_GEN_DIR \
             (run dev/java-interop/run-interop-replace-invariant.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_replace_invariant_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");

    let invalid_location = format!("{warehouse}/invalid_rust");
    let invalid_table = create_table(&catalog, &invalid_location, "invalid_rust").await;
    let original = write_rows(&invalid_table, 3).await;
    let tx = Transaction::new(&invalid_table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![original.clone()])
        .apply(tx)
        .expect("apply append of 3-row file");
    let invalid_table = tx.commit(&catalog).await.expect("commit 3-row append");
    let grown = write_rows(&invalid_table, 5).await;
    let tx = Transaction::new(&invalid_table);
    let tx = tx
        .rewrite_files(vec![original], vec![grown])
        .apply(tx)
        .expect("apply invalid rewrite");
    let error = tx
        .commit(&catalog)
        .await
        .expect_err("Rust must refuse 3-to-5 REPLACE");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("Invalid REPLACE operation: 5 added records > 3 replaced records"),
        "unexpected message: {}",
        error.message()
    );

    let valid_location = format!("{warehouse}/valid_rust/rust_table");
    let valid_table = create_table(&catalog, &valid_location, "valid_rust").await;
    let original = write_rows(&valid_table, 3).await;
    let tx = Transaction::new(&valid_table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![original.clone()])
        .apply(tx)
        .expect("apply valid append");
    let valid_table = tx.commit(&catalog).await.expect("commit valid append");
    let replacement = write_rows(&valid_table, 3).await;
    let tx = Transaction::new(&valid_table);
    let tx = tx
        .rewrite_files(vec![original], vec![replacement])
        .apply(tx)
        .expect("apply equal rewrite");
    let valid_table = tx.commit(&catalog).await.expect("commit 3-to-3 rewrite");
    assert_eq!(live_rows(&valid_table).await, expected_rows());
    write_final_metadata(&valid_table).await;

    println!(
        "interop_replace_invariant GEN OK — Rust refused 3-to-5 REPLACE and wrote a 3-to-3 table"
    );
}

#[tokio::test]
async fn test_replace_invariant_rust_reads_java_valid_table() {
    let Some(dir) = d1_dir() else {
        println!(
            "skipping replace-invariant D1 — set ICEBERG_INTEROP_REPLACE_INVARIANT_DIR \
             (run dev/java-interop/run-interop-replace-invariant.sh)"
        );
        return;
    };

    let java_rows_path = dir.join("valid_java").join("java_rows.json");
    let java_rows: Vec<ScanRow> = serde_json::from_str(
        &std::fs::read_to_string(&java_rows_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", java_rows_path.display())),
    )
    .unwrap_or_else(|e| panic!("parse {}: {e}", java_rows_path.display()));

    let metadata_path = dir.join("valid_java/table/metadata/final.metadata.json");
    let json = std::fs::read_to_string(&metadata_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", metadata_path.display()));
    let metadata: iceberg::spec::TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("parse {}: {e}", metadata_path.display()));
    let table = Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", "valid_java"]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from Java-written final.metadata.json");

    let rust_rows = live_rows(&table).await;
    assert_eq!(
        rust_rows, java_rows,
        "Rust scan of Java 3-to-3 rewrite must equal Java IcebergGenerics"
    );
    assert_eq!(rust_rows, expected_rows());
    println!("interop_replace_invariant D1 OK — Rust read Java 3-to-3 rewrite as {{10,20,30}}");
}

#[tokio::test]
async fn test_replace_invariant_java_invalid_fixture_is_present() {
    let Some(dir) = d1_dir() else {
        println!(
            "skipping replace-invariant invalid-fixture check — set \
             ICEBERG_INTEROP_REPLACE_INVARIANT_DIR"
        );
        return;
    };
    let threw = dir.join("invalid/threw.json");
    let body =
        std::fs::read_to_string(&threw).unwrap_or_else(|e| panic!("read {}: {e}", threw.display()));
    assert!(
        body.contains("Invalid REPLACE operation"),
        "Java threw.json must carry the REPLACE invariant message, got {body}"
    );
}
