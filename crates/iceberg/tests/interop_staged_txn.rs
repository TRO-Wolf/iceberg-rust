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

//! Java interop for the R158 STAGED CREATE/REPLACE TABLE TRANSACTION
//! (`StagedTableTransaction::{begin_create,begin_replace,add_data_files,commit}` ⇄ Java
//! `Catalog.newCreateTableTransaction` / `newReplaceTableTransaction`). Driven by
//! `dev/java-interop/run-interop-staged-txn.sh`.
//!
//! # The two directions (one fixture tree, `$dir`)
//!
//! **Direction 1 — Java acts, Rust verifies.** Java's `StagedTxnOracle.generate` builds the
//! create / replace / fmtv scenarios under `$dir/d1` via the engine-agnostic iceberg-core surface
//! (`Transactions.createTableTransaction` / `replaceTableTransaction` over
//! `ops.current().buildReplacement(...)`) that the catalog methods wrap.
//! [`test_staged_txn_rust_verifies_java`] reads each Java-produced table and asserts the create
//! single-publish + row content, the replace invariant set `E-INV` (1)-(7) per cycle, the
//! format-version directive contract, and — for `create` — that Rust's canonical snapshot-metadata
//! view reproduces Java's `java_meta.json`.
//!
//! **Direction 2 — Rust acts, Java verifies.** [`test_staged_txn_gen_rust_produces_fixtures`]
//! performs the SAME scenarios under `$dir/d2` through the production `StagedTableTransaction`, so
//! Java's `verify-interop-staged-txn` (and the run script's cross-check + sabotage battery) can
//! judge the Rust-produced metadata. Java's view of the Rust `create`/`replace-r2` tables must equal
//! Java's own view (C-5 structural equivalence — paths are erased by the canonical view).
//!
//! # E-INV — the replace invariant set (asserted per cycle, both directions)
//!
//! (1) `table_uuid` identical across cycles; (2) pre-replace snapshots retained; (3) `metadata_log`
//! grows (never truncated); (4) `main` ref reset — current snapshot is the replace's own new state
//! (never a pre-replace snapshot; NONE before the first append, proven by the fmtv no-append
//! replace) and reads through `main` expose ONLY the latest replace's rows; (5) `location()` stable;
//! (6) format version preserved absent a directive; (7) `last_column_id` monotonic (never reduced
//! below the base = 3, the dropped `note`).
//!
//! # The env gates
//!
//! Both tests are clean NO-OPS unless their env var is set non-empty (runtime early-return, not
//! `#[ignore]`), so the offline `cargo test` gate needs no Java/Maven.
//! `ICEBERG_INTEROP_STAGED_TXN_GEN_DIR` drives the Rust GEN (Direction 2, writes `$dir/d2`);
//! `ICEBERG_INTEROP_STAGED_TXN_DIR` drives the verify (Direction 1, reads `$dir/d1`).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::io::{FileIO, FileIOBuilder, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, NestedField,
    PrimitiveType, Schema, SortOrder, Struct, TableMetadata, TableProperties, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, StagedTableTransaction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use serde_json::Value as JsonValue;

mod common;
use common::snapshot_meta_view::snapshot_meta_view;

// ===========================================================================================
// LOCKED constants — identical to the Java `StagedTxnOracle` (anti-circular).
// ===========================================================================================

const CREATE_IDS: &[i64] = &[10, 20, 30];
const CREATE_DATA: &[&str] = &["a", "b", "c"];
const R1_IDS: &[i64] = &[30, 31];
const R1_DATA: &[&str] = &["x", "y"];
const R2_IDS: &[i64] = &[40, 41, 42];
const R2_DATA: &[&str] = &["p", "q", "r"];
/// Base `last_column_id`: `{id,data,note}` => 3. The dropped `note` id must survive as `last_col`.
const BASE_LAST_COLUMN_ID: i32 = 3;

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_STAGED_TXN_GEN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn compare_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_STAGED_TXN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// `{1 id long req, 2 data string req}` — the create + replacement schema.
fn schema_c() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema {id,data}")
}

/// `{1 id long req, 2 data string req, 3 note string opt}` — the replace-BASE schema (last_col=3).
fn schema_b() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "note", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build schema {id,data,note}")
}

fn unpartitioned() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder().with_spec_id(0).build()
}

/// A metadata-only `DataFile` (no parquet on disk — off-main base history, never scanned).
fn meta_only(table_location: &str, name: &str, record_count: u64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{table_location}/data/{name}.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(record_count * 100)
        .record_count(record_count)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build metadata-only data file")
}

/// Write a REAL unpartitioned parquet `{id,data}` file against `table` (its current schema, FileIO,
/// and location generator), returning the produced `DataFile` with real metrics.
async fn write_real(table: &Table, ids: &[i64], data: &[&str]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(data.to_vec())) as ArrayRef,
    ])
    .expect("build the {id,data} data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "staged".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .expect("build unpartitioned data file writer");
    writer.write(batch).await.expect("write data batch");
    writer
        .close()
        .await
        .expect("close data file writer")
        .into_iter()
        .next()
        .expect("one data file")
}

/// A MemoryCatalog over a shared local-FS storage factory + a FileIO on the SAME store (so the
/// catalog can read the staged metadata a `begin_create` writes). Mirrors `staged_table.rs` tests.
async fn shared_fs_catalog(warehouse: &str, name: &str) -> (impl Catalog, FileIO) {
    let factory = Arc::new(LocalFsStorageFactory);
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(factory.clone())
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let file_io = FileIOBuilder::new(factory).build();
    (catalog, file_io)
}

/// Write `table.metadata()` to `<table_location>/metadata/<name>` as a stable, known file.
async fn write_meta(table: &Table, table_location: &str, name: &str) {
    let path = format!("{table_location}/metadata/{name}");
    table
        .metadata_ref()
        .write_to(table.file_io(), &path)
        .await
        .unwrap_or_else(|error| panic!("write {name}: {error}"));
}

// ===========================================================================================
// DIRECTION 2 — the Rust GEN: build each scenario under $dir/d2 via StagedTableTransaction.
// ===========================================================================================

#[tokio::test]
async fn test_staged_txn_gen_rust_produces_fixtures() {
    let Some(dir) = gen_dir() else {
        println!(
            "skipping interop_staged_txn GEN — set ICEBERG_INTEROP_STAGED_TXN_GEN_DIR \
             (run dev/java-interop/run-interop-staged-txn.sh)"
        );
        return;
    };
    let d2 = dir.join("d2");
    gen_create(&d2.join("create")).await;
    gen_replace(&d2.join("replace")).await;
    gen_fmtv(&d2.join("fmtv_preserve"), false).await;
    gen_fmtv(&d2.join("fmtv_upgrade"), true).await;
    println!(
        "interop_staged_txn GEN: wrote d2 (create/replace/fmtv) under {}",
        d2.display()
    );
}

/// C-3: `begin_create` + REAL parquet append + one publish.
async fn gen_create(dir: &Path) {
    let table_location = dir.join("table").to_string_lossy().to_string();
    fs::create_dir_all(&table_location).expect("create table dir");
    let warehouse = dir.to_string_lossy().to_string();
    let (catalog, file_io) = shared_fs_catalog(&warehouse, "interop_staged_txn_create").await;
    let ns = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("create namespace");
    let ident = TableIdent::new(ns, "table".to_string());

    let creation = TableCreation::builder()
        .name("table".to_string())
        .location(table_location.clone())
        .schema(schema_c())
        .partition_spec(unpartitioned())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let staged = StagedTableTransaction::begin_create(file_io, ident.clone(), creation)
        .await
        .expect("begin_create");
    let df = write_real(staged.table(), CREATE_IDS, CREATE_DATA).await;
    let published = staged
        .add_data_files(vec![df])
        .commit(&catalog)
        .await
        .expect("commit staged create");

    write_meta(&published, &table_location, "final.metadata.json").await;
    println!("interop_staged_txn GEN/create: single-publish table written to {table_location}");
}

/// C-4: base (2 snapshots) + two `begin_replace` cycles with REAL parquet. Emits base/r1/r2 metadata.
async fn gen_replace(dir: &Path) {
    let table_location = dir.join("table").to_string_lossy().to_string();
    fs::create_dir_all(&table_location).expect("create table dir");
    let warehouse = dir.to_string_lossy().to_string();
    let (catalog, file_io) = shared_fs_catalog(&warehouse, "interop_staged_txn_replace").await;
    let ns = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("create namespace");
    let ident = TableIdent::new(ns, "table".to_string());

    // base: begin_create(schema B) + metadata-only S1 (rc2) => 1 snapshot, then a normal fast_append
    // S2 (rc1) => 2 snapshots. last_column_id = 3 (the `note` field).
    let creation_base = TableCreation::builder()
        .name("table".to_string())
        .location(table_location.clone())
        .schema(schema_b())
        .partition_spec(unpartitioned())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let staged = StagedTableTransaction::begin_create(file_io, ident.clone(), creation_base)
        .await
        .expect("begin_create base");
    let s1 = meta_only(&table_location, "base-s1", 2);
    let published = staged
        .add_data_files(vec![s1])
        .commit(&catalog)
        .await
        .expect("commit base S1");
    // S2 — a second append so the base carries >= 2 snapshots (the C-4 precondition).
    let tx = Transaction::new(&published);
    let tx = tx
        .fast_append()
        .add_data_files(vec![meta_only(&table_location, "base-s2", 1)])
        .apply(tx)
        .expect("apply base S2");
    tx.commit(&catalog).await.expect("commit base S2");
    let base_table = catalog.load_table(&ident).await.expect("reload base");
    write_meta(&base_table, &table_location, "base.final.metadata.json").await;

    // r1: begin_replace(schema C) + REAL [(30,x),(31,y)].
    let r1_table = replace_cycle(&catalog, &base_table, R1_IDS, R1_DATA).await;
    write_meta(&r1_table, &table_location, "r1.final.metadata.json").await;

    // r2: begin_replace(schema C) + REAL [(40,p),(41,q),(42,r)].
    let r2_table = replace_cycle(&catalog, &r1_table, R2_IDS, R2_DATA).await;
    write_meta(&r2_table, &table_location, "r2.final.metadata.json").await;
    println!("interop_staged_txn GEN/replace: base + r1 + r2 metadata written to {table_location}");
}

/// One replace cycle over `existing`: `begin_replace(schema C)` + REAL append + commit; reload.
async fn replace_cycle(
    catalog: &impl Catalog,
    existing: &Table,
    ids: &[i64],
    data: &[&str],
) -> Table {
    let creation = TableCreation::builder()
        .name("table".to_string())
        .schema(schema_c())
        .partition_spec(unpartitioned())
        .sort_order(SortOrder::unsorted_order())
        .build();
    let staged = StagedTableTransaction::begin_replace(existing, creation)
        .await
        .expect("begin_replace");
    let df = write_real(staged.table(), ids, data).await;
    staged
        .add_data_files(vec![df])
        .commit(catalog)
        .await
        .expect("commit staged replace");
    catalog
        .load_table(existing.identifier())
        .await
        .expect("reload after replace")
}

/// C-6: V1 `begin_create` + a NO-append `begin_replace`. `upgrade` toggles the `format-version=2`
/// directive (the default-built TableCreation's V2 `format_version` is IGNORED on the replace path).
async fn gen_fmtv(dir: &Path, upgrade: bool) {
    let table_location = dir.join("table").to_string_lossy().to_string();
    fs::create_dir_all(&table_location).expect("create table dir");
    let warehouse = dir.to_string_lossy().to_string();
    let name = if upgrade {
        "interop_staged_txn_fmtv_up"
    } else {
        "interop_staged_txn_fmtv_pr"
    };
    let (catalog, file_io) = shared_fs_catalog(&warehouse, name).await;
    let ns = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&ns, HashMap::new())
        .await
        .expect("create namespace");
    let ident = TableIdent::new(ns, "table".to_string());

    // base: V1 create + one metadata-only append (1 snapshot).
    let creation_base = TableCreation::builder()
        .name("table".to_string())
        .location(table_location.clone())
        .schema(schema_c())
        .partition_spec(unpartitioned())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V1)
        .build();
    let staged = StagedTableTransaction::begin_create(file_io, ident.clone(), creation_base)
        .await
        .expect("begin_create fmtv base");
    let published = staged
        .add_data_files(vec![meta_only(&table_location, "fmtv-s1", 1)])
        .commit(&catalog)
        .await
        .expect("commit fmtv base");
    write_meta(&published, &table_location, "base.final.metadata.json").await;
    let base_table = catalog.load_table(&ident).await.expect("reload fmtv base");

    // replace with NO append (proves E-INV#4 "none before first append": main ref reset). The
    // default-built TableCreation carries the builder's V2 `format_version` default, which the
    // replace path IGNORES — only a `format-version` PROPERTY directs an upgrade.
    let creation = if upgrade {
        TableCreation::builder()
            .name("table".to_string())
            .schema(schema_c())
            .partition_spec(unpartitioned())
            .sort_order(SortOrder::unsorted_order())
            .properties(HashMap::from([(
                TableProperties::PROPERTY_FORMAT_VERSION.to_string(),
                "2".to_string(),
            )]))
            .build()
    } else {
        TableCreation::builder()
            .name("table".to_string())
            .schema(schema_c())
            .partition_spec(unpartitioned())
            .sort_order(SortOrder::unsorted_order())
            .build()
    };
    let staged = StagedTableTransaction::begin_replace(&base_table, creation)
        .await
        .expect("begin_replace fmtv");
    let replaced = staged.commit(&catalog).await.expect("commit fmtv replace");
    write_meta(&replaced, &table_location, "replaced.final.metadata.json").await;
    println!(
        "interop_staged_txn GEN/fmtv_{}: base(V1) + replaced written to {table_location}",
        if upgrade { "upgrade" } else { "preserve" }
    );
}

// ===========================================================================================
// DIRECTION 1 — Java acts, Rust verifies: read $dir/d1 and assert every clause.
// ===========================================================================================

#[tokio::test]
async fn test_staged_txn_rust_verifies_java() {
    let Some(dir) = compare_dir() else {
        println!(
            "skipping interop_staged_txn verify — set ICEBERG_INTEROP_STAGED_TXN_DIR \
             (run dev/java-interop/run-interop-staged-txn.sh)"
        );
        return;
    };
    let d1 = dir.join("d1");
    verify_create(&d1.join("create")).await;
    verify_replace(&d1.join("replace")).await;
    verify_fmtv(&d1.join("fmtv_preserve"), false).await;
    verify_fmtv(&d1.join("fmtv_upgrade"), true).await;
    println!("interop_staged_txn verify: Java's d1 create/replace/fmtv all pass Rust's assertions");
}

fn load_metadata(path: &Path) -> TableMetadata {
    let json =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&json).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn load_json(path: &Path) -> JsonValue {
    let raw =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Load an on-disk table (from its final metadata json) via a local-FS FileIO, for scanning.
fn load_table(metadata_path: &Path, name: &str) -> Table {
    let metadata = load_metadata(metadata_path);
    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", name]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .unwrap_or_else(|error| panic!("build table from {}: {error}", metadata_path.display()))
}

/// Scan `table` through `main` → sorted `(id, data)` live rows.
async fn scan_live_rows(table: &Table) -> Vec<(i64, Option<String>)> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect scan batches");
    let mut rows = Vec::new();
    for batch in &batches {
        let id = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let data = batch.column_by_name("data").expect("data column");
        for i in 0..batch.num_rows() {
            let value = if data.is_null(i) {
                None
            } else {
                Some(data.as_string::<i32>().value(i).to_string())
            };
            rows.push((id.value(i), value));
        }
    }
    rows.sort_by_key(|(id, _)| *id);
    rows
}

async fn scan_live_ids(table: &Table) -> Vec<i64> {
    scan_live_rows(table)
        .await
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

/// C-1: Java-created table — one publish, schema {id,data}, exact row content, canonical view.
async fn verify_create(dir: &Path) {
    let meta_path = dir.join("table/metadata/final.metadata.json");
    let metadata = load_metadata(&meta_path);

    // Single publish => exactly ONE snapshot, current present.
    assert_eq!(
        metadata.snapshots().count(),
        1,
        "create: expected exactly 1 snapshot (single publish), got {}",
        metadata.snapshots().count()
    );
    assert!(
        metadata.current_snapshot().is_some(),
        "create: no current snapshot after the staged create publish"
    );
    // Schema {id,data}.
    let schema = metadata.current_schema();
    assert!(
        schema.field_by_id(1).is_some_and(|f| f.name == "id"),
        "create: field 1 not id"
    );
    assert!(
        schema.field_by_id(2).is_some_and(|f| f.name == "data"),
        "create: field 2 not data"
    );
    assert_eq!(
        schema.as_struct().fields().len(),
        2,
        "create: schema must be exactly {{id,data}}"
    );

    // Row content: exactly [(10,a),(20,b),(30,c)].
    let table = load_table(&meta_path, "staged_txn_create");
    let rows = scan_live_rows(&table).await;
    let expected: Vec<(i64, Option<String>)> = CREATE_IDS
        .iter()
        .zip(CREATE_DATA)
        .map(|(id, d)| (*id, Some((*d).to_string())))
        .collect();
    assert_eq!(
        rows, expected,
        "create: scanned rows diverge from the committed content"
    );

    // Rust's canonical view reproduces Java's own view (java_meta.json) — structural equivalence.
    let rust_view = snapshot_meta_view(&meta_path).await;
    let java_view = load_json(&dir.join("java_meta.json"));
    assert_eq!(
        rust_view, java_view,
        "create: Rust's canonical view of the Java table diverges from Java's own view"
    );

    // Cross-check against the hand-declared java_rows.json (anti-circular: same constants both sides).
    let java_rows = load_json(&dir.join("java_rows.json"));
    let java_ids: Vec<i64> = java_rows
        .as_array()
        .expect("java_rows is an array")
        .iter()
        .map(|row| row["id"].as_i64().expect("id"))
        .collect();
    assert_eq!(
        java_ids,
        CREATE_IDS.to_vec(),
        "create: java_rows.json ids diverge from constants"
    );
    println!("interop_staged_txn verify/create: 1 publish, schema {{id,data}}, rows {rows:?} OK");
}

/// C-2 + E-INV: Java replace base/r1/r2 — assert the replace invariant set per cycle + row content.
async fn verify_replace(dir: &Path) {
    let base = load_metadata(&dir.join("table/metadata/base.final.metadata.json"));
    let r1 = load_metadata(&dir.join("table/metadata/r1.final.metadata.json"));
    let r2 = load_metadata(&dir.join("table/metadata/r2.final.metadata.json"));
    let r1_table = load_table(
        &dir.join("table/metadata/r1.final.metadata.json"),
        "staged_txn_r1",
    );
    let r2_table = load_table(
        &dir.join("table/metadata/r2.final.metadata.json"),
        "staged_txn_r2",
    );
    let r1_live = scan_live_ids(&r1_table).await;
    let r2_live = scan_live_ids(&r2_table).await;

    assert_replace_invariants("java", &base, &r1, &r2, &r1_live, &r2_live);
    println!("interop_staged_txn verify/replace: E-INV(1-7) hold over Java base->r1->r2 OK");
}

/// The E-INV replace invariant set (1)-(7) — the Rust mirror of Java `assertReplaceInvariants`.
/// Panics with a specific message on the first violated invariant.
fn assert_replace_invariants(
    tag: &str,
    base: &TableMetadata,
    r1: &TableMetadata,
    r2: &TableMetadata,
    r1_live: &[i64],
    r2_live: &[i64],
) {
    // (1) table_uuid identical across every cycle.
    assert_eq!(
        base.uuid(),
        r1.uuid(),
        "{tag} E-INV#1: uuid changed base->r1"
    );
    assert_eq!(
        base.uuid(),
        r2.uuid(),
        "{tag} E-INV#1: uuid changed base->r2"
    );

    // (2) pre-replace snapshots retained: base ⊆ r1 ⊆ r2.
    let base_ids: Vec<i64> = base.snapshots().map(|s| s.snapshot_id()).collect();
    let r1_ids: Vec<i64> = r1.snapshots().map(|s| s.snapshot_id()).collect();
    let r2_ids: Vec<i64> = r2.snapshots().map(|s| s.snapshot_id()).collect();
    assert!(
        base_ids.len() >= 2,
        "{tag} E-INV#2: base must have >=2 snapshots, got {}",
        base_ids.len()
    );
    assert!(
        base_ids.iter().all(|id| r1_ids.contains(id)),
        "{tag} E-INV#2: r1 dropped a base snapshot (base={base_ids:?} r1={r1_ids:?})"
    );
    assert!(
        r1_ids.iter().all(|id| r2_ids.contains(id)),
        "{tag} E-INV#2: r2 dropped an r1 snapshot (r1={r1_ids:?} r2={r2_ids:?})"
    );

    // (3) metadata_log grows (appended, never truncated): r1 > base, r2 > r1.
    let base_log = base.metadata_log().len();
    let r1_log = r1.metadata_log().len();
    let r2_log = r2.metadata_log().len();
    assert!(
        r1_log > base_log && r2_log > r1_log,
        "{tag} E-INV#3: metadata_log did not grow (base={base_log} r1={r1_log} r2={r2_log})"
    );

    // (4) main ref reset + reads expose ONLY the latest replace's rows.
    assert_main_reset_and_rows(tag, "r1", &base_ids, r1, r1_live, R1_IDS);
    assert_main_reset_and_rows(tag, "r2", &base_ids, r2, r2_live, R2_IDS);

    // (5) location() stable.
    assert_eq!(
        base.location(),
        r1.location(),
        "{tag} E-INV#5: location drifted base->r1"
    );
    assert_eq!(
        base.location(),
        r2.location(),
        "{tag} E-INV#5: location drifted base->r2"
    );

    // (6) format version preserved absent a directive.
    assert_eq!(
        base.format_version(),
        r1.format_version(),
        "{tag} E-INV#6: format version changed r1"
    );
    assert_eq!(
        base.format_version(),
        r2.format_version(),
        "{tag} E-INV#6: format version changed r2"
    );

    // (7) last_column_id monotonic — never reduced below the base (=3, the dropped `note`).
    assert_eq!(
        base.last_column_id(),
        BASE_LAST_COLUMN_ID,
        "{tag} E-INV#7: base last_column_id must be {BASE_LAST_COLUMN_ID}"
    );
    assert!(
        r1.last_column_id() >= base.last_column_id(),
        "{tag} E-INV#7: last_column_id reduced base->r1 ({} < {})",
        r1.last_column_id(),
        base.last_column_id()
    );
    assert!(
        r2.last_column_id() >= r1.last_column_id(),
        "{tag} E-INV#7: last_column_id reduced r1->r2 ({} < {})",
        r2.last_column_id(),
        r1.last_column_id()
    );
}

fn assert_main_reset_and_rows(
    tag: &str,
    cycle: &str,
    base_ids: &[i64],
    replaced: &TableMetadata,
    live: &[i64],
    expected: &[i64],
) {
    let current = replaced
        .current_snapshot()
        .unwrap_or_else(|| panic!("{tag} E-INV#4 {cycle}: no current snapshot after replace"));
    assert!(
        !base_ids.contains(&current.snapshot_id()),
        "{tag} E-INV#4 {cycle}: main still points at a pre-replace snapshot {}",
        current.snapshot_id()
    );
    assert_eq!(
        live, expected,
        "{tag} E-INV#4 {cycle}: main exposes {live:?}, expected ONLY the latest replace's rows {expected:?}"
    );
}

/// C-6: Java V1 create + replace — assert the format-version directive contract + the "none before
/// first append" main-ref-reset branch + uuid/location/history retention.
async fn verify_fmtv(dir: &Path, upgrade: bool) {
    let base = load_metadata(&dir.join("table/metadata/base.final.metadata.json"));
    let replaced = load_metadata(&dir.join("table/metadata/replaced.final.metadata.json"));
    let tag = if upgrade {
        "fmtv_upgrade"
    } else {
        "fmtv_preserve"
    };

    assert_eq!(
        base.format_version(),
        FormatVersion::V1,
        "{tag}: base not created V1"
    );
    let want = if upgrade {
        FormatVersion::V2
    } else {
        FormatVersion::V1
    };
    assert_eq!(
        replaced.format_version(),
        want,
        "{tag}: replaced format version"
    );
    assert!(
        !replaced
            .properties()
            .contains_key(TableProperties::PROPERTY_FORMAT_VERSION),
        "{tag}: the `format-version` directive leaked into the persisted property map"
    );
    // No-append replace => main ref reset, NO current snapshot (E-INV#4 "none" branch).
    assert!(
        replaced.current_snapshot().is_none(),
        "{tag}: a no-append replace left a current snapshot (main ref not reset)"
    );
    // History + uuid + location retained.
    let base_ids: Vec<i64> = base.snapshots().map(|s| s.snapshot_id()).collect();
    assert!(
        base_ids
            .iter()
            .all(|id| replaced.snapshot_by_id(*id).is_some()),
        "{tag}: base snapshot(s) lost across the replace"
    );
    assert_eq!(
        base.uuid(),
        replaced.uuid(),
        "{tag}: uuid not retained across replace"
    );
    assert_eq!(
        base.location(),
        replaced.location(),
        "{tag}: location not retained across replace"
    );
    println!("interop_staged_txn verify/{tag}: V1 -> {want:?}, directive filtered, main reset OK");
}
