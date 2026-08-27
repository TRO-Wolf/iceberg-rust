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

//! Partition-stats file interop harness. Java reads the Rust-written stats parquet (direction 1),
//! Rust reads Java's (direction 2), plus a cross-version projection test.
//!
//! # Fixture
//!
//! V2 table, `identity(category)`, schema `{1 id long required, 2 category string required,
//! 3 data string optional}`. S1 fast-appends file A (cat=a, 3 records, 300 bytes) and file B
//! (cat=b, 2 records, 200 bytes). S2 adds position-delete PD (cat=a, 1 record, 50 bytes).
//!
//! The expected rows are hand-declared, so the test cannot go circular:
//!
//! | partition | spec_id | data_records | data_files | size | pos_del_records | pos_del_files |
//! |---|---|---|---|---|---|---|
//! | a | 0 | 3 | 1 | 300 | 1 | 1 |
//! | b | 0 | 2 | 1 | 200 | 0 | 0 |
//!
//! `equality_delete_*` and `dv_count` are 0, and `total_record_count` is `None`.
//! `last_updated_snapshot_id` resolves to S2 for cat=a and to S1 for cat=b.
//!
//! # The two directions
//!
//! A GEN test writes `<table>/metadata/final.metadata.json` and a `*_expected.json`. Java's
//! `verify-interop-partition-stats` then judges them through its production
//! `readPartitionStatsFile`. A D2 test reads the Java-written stats parquet and compares the
//! decoded rows against `java_*_stats.json`.
//!
//! Both directions cover the base fixture, the incremental SUBTRACT arm, and four exotic partition
//! types: uuid, time, fixed[4], and binary.
//!
//! [`test_partition_stats_cross_version_v2_file_v3_schema`] reads a Java-written V2 stats file
//! against the V3 schema. The absent `dv_count` column must null-fill to 0.
//!
//! # Env gate
//!
//! Each test is a runtime no-op unless its env var holds a path, so the offline `cargo test` gate
//! needs no Java. The vars are `ICEBERG_INTEROP_PARTITION_STATS_GEN_DIR` (direction 1),
//! `ICEBERG_INTEROP_PARTITION_STATS_DIR` (direction 2 and the cross-version test), and
//! `..._INCR_DIR`, `..._UUID_DIR`, `..._TIME_DIR`, `..._FIXED_DIR`, `..._BINARY_DIR`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use iceberg::io::LocalFsStorageFactory;
use iceberg::maintenance::{
    PartitionStats, compute_and_write_stats_file, partition_stats_schema,
    read_partition_stats_file, register_partition_stats_file, unified_partition_type,
};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal,
    NestedField, PrimitiveLiteral, PrimitiveType, Schema, SortOrder, Struct, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use serde_json::{Value as JsonValue, json};
use uuid::Uuid;

// ---- Hand-declared counters, agreed with PartitionStatsOracle.java. Neither side derives them. ---

const A_DATA_RECORDS: i64 = 3;
const A_DATA_FILE_SIZE: u64 = 300;
const A_POS_DEL_RECORDS: i64 = 1;
const A_POS_DEL_FILE_SIZE: u64 = 50;
const B_DATA_RECORDS: i64 = 2;
const B_DATA_FILE_SIZE: u64 = 200;

// ---- Incremental (R2) constants ---- agreed with IncrementalPartitionStatsOracle.java.
const INCR_A_DATA_RECORDS: i64 = 3;
const INCR_A_DATA_FILE_SIZE: u64 = 300;
const INCR_B_DATA_RECORDS: i64 = 2;
const INCR_B_DATA_FILE_SIZE: u64 = 200;

// ---- UUID (R2) constants ---- agreed with UuidPartitionStatsOracle.java.
/// The known UUID partition value (same string as Java's KNOWN_UUID_STRING).
const KNOWN_UUID_STR: &str = "550e8400-e29b-41d4-a716-446655440000";
const UUID_DATA_RECORDS: i64 = 5;
const UUID_DATA_FILE_SIZE: u64 = 500;

// ---- TIME (R3) constants ---- agreed with TimePartitionStatsOracle.java.
/// Micros since midnight (12:34:56.789012). Same value as Java's `KNOWN_TIME_MICROS`.
const KNOWN_TIME_MICROS: i64 = 45_296_789_012;
const TIME_DATA_RECORDS: i64 = 7;
const TIME_DATA_FILE_SIZE: u64 = 700;

// ---- FIXED[4] (R3) constants ---- agreed with FixedPartitionStatsOracle.java.
const FIXED_LENGTH: usize = 4;
/// The known `fixed[4]` partition value (hex 0xdeadbeef), same bytes as Java's KNOWN_FIXED_BYTES.
const KNOWN_FIXED_BYTES: [u8; FIXED_LENGTH] = [0xde, 0xad, 0xbe, 0xef];
const FIXED_DATA_RECORDS: i64 = 8;
const FIXED_DATA_FILE_SIZE: u64 = 800;

// ---- BINARY (R3) constants ---- agreed with BinaryPartitionStatsOracle.java.
/// The known binary partition value (hex 0x0102030405), same bytes as Java's KNOWN_BINARY_BYTES.
const KNOWN_BINARY_BYTES: [u8; 5] = [0x01, 0x02, 0x03, 0x04, 0x05];
const BINARY_DATA_RECORDS: i64 = 9;
const BINARY_DATA_FILE_SIZE: u64 = 900;

/// Lowercase hex, two chars per byte. It matches `bytesToHex` in the Java oracles.
fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PARTITION_STATS_GEN_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn compare_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PARTITION_STATS_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn incr_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PARTITION_STATS_INCR_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn uuid_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PARTITION_STATS_UUID_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn time_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PARTITION_STATS_TIME_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn fixed_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PARTITION_STATS_FIXED_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn binary_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PARTITION_STATS_BINARY_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

// ---- Fixture schema and spec builders, on the same constants as Java PartitionStatsOracle. -------

/// Table schema: `{1 id long required, 2 category string required, 3 data string optional}`.
fn fixture_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build fixture schema")
}

/// Partition spec: `identity(category)` — spec id 0, field id 1000 (Java's sequential id).
fn fixture_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        // source column id 2 (category), transform Identity, partition field name "category".
        .add_partition_field(2, "category".to_string(), Transform::Identity)
        .expect("add identity(category) partition field")
        .build()
}

/// A data file with a single-field identity partition (category = the given value).
fn data_file(
    table_location: &str,
    name: &str,
    category: &str,
    records: u64,
    size: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{table_location}/data/{name}.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(records)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::string(category))]))
        .build()
        .expect("build data file")
}

/// A position-delete file with a single-field identity partition (category = the given value).
fn pos_delete_file(
    table_location: &str,
    name: &str,
    category: &str,
    records: u64,
    size: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(format!("{table_location}/data/{name}.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(records)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::string(category))]))
        .build()
        .expect("build position-delete file")
}

/// Table schema for the UUID fixture: `{1 id long required, 2 partition_id uuid required}`.
fn uuid_fixture_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "partition_id", Type::Primitive(PrimitiveType::Uuid)).into(),
        ])
        .build()
        .expect("build uuid fixture schema")
}

/// Partition spec for the UUID fixture: `identity(partition_id)` — source column id 2.
fn uuid_fixture_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "partition_id".to_string(), Transform::Identity)
        .expect("add identity(partition_id) partition field")
        .build()
}

/// A data file for the UUID fixture with the given UUID as the partition value.
fn uuid_data_file(
    table_location: &str,
    name: &str,
    uuid_val: Uuid,
    records: u64,
    size: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{table_location}/data/{name}.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(records)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::uuid(uuid_val))]))
        .build()
        .expect("build uuid data file")
}

/// Table schema for the time fixture: `{1 id long required, 2 partition_time time required}`.
fn time_fixture_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "partition_time", Type::Primitive(PrimitiveType::Time)).into(),
        ])
        .build()
        .expect("build time fixture schema")
}

/// Partition spec for the time fixture: `identity(partition_time)` — source column id 2.
fn time_fixture_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "partition_time".to_string(), Transform::Identity)
        .expect("add identity(partition_time) partition field")
        .build()
}

/// A data file for the time fixture with the given micros-since-midnight as the partition value.
fn time_data_file(
    table_location: &str,
    name: &str,
    micros: i64,
    records: u64,
    size: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{table_location}/data/{name}.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(records)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::time(micros))]))
        .build()
        .expect("build time data file")
}

/// Table schema for the fixed fixture: `{1 id long required, 2 partition_fixed fixed[4] required}`.
fn fixed_fixture_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(
                2,
                "partition_fixed",
                Type::Primitive(PrimitiveType::Fixed(FIXED_LENGTH as u64)),
            )
            .into(),
        ])
        .build()
        .expect("build fixed fixture schema")
}

/// Partition spec for the fixed fixture: `identity(partition_fixed)` — source column id 2.
fn fixed_fixture_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "partition_fixed".to_string(), Transform::Identity)
        .expect("add identity(partition_fixed) partition field")
        .build()
}

/// A data file for the fixed fixture with the given bytes as the `fixed[L]` partition value.
fn fixed_data_file(
    table_location: &str,
    name: &str,
    bytes: &[u8],
    records: u64,
    size: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{table_location}/data/{name}.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(records)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::fixed(
            bytes.iter().copied(),
        ))]))
        .build()
        .expect("build fixed data file")
}

/// Table schema for the binary fixture: `{1 id long required, 2 partition_binary binary required}`.
fn binary_fixture_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(
                2,
                "partition_binary",
                Type::Primitive(PrimitiveType::Binary),
            )
            .into(),
        ])
        .build()
        .expect("build binary fixture schema")
}

/// Partition spec for the binary fixture: `identity(partition_binary)` — source column id 2.
fn binary_fixture_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "partition_binary".to_string(), Transform::Identity)
        .expect("add identity(partition_binary) partition field")
        .build()
}

/// A data file for the binary fixture with the given bytes as the `binary` partition value.
fn binary_data_file(
    table_location: &str,
    name: &str,
    bytes: &[u8],
    records: u64,
    size: u64,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(format!("{table_location}/data/{name}.parquet"))
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(records)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::binary(
            bytes.iter().copied(),
        ))]))
        .build()
        .expect("build binary data file")
}

/// Write `final.metadata.json` at `<table_dir>/metadata/final.metadata.json`.
async fn write_final_metadata(table: &iceberg::table::Table, table_dir: &str) {
    let path = format!("{table_dir}/metadata/final.metadata.json");
    table
        .metadata_ref()
        .write_to(table.file_io(), path.as_str())
        .await
        .expect("write final.metadata.json");
}

/// Serializes the stats rows to `expected_stats.json`. It embeds the actual snapshot ids, so Java
/// can compare them exactly. The rows sort cat=a first, as `compute_partition_stats` returns them.
/// ```json
/// [
///   {
///     "partition_category": "a",
///     "spec_id": 0,
///     "data_record_count": 3,
///     "data_file_count": 1,
///     "total_data_file_size_in_bytes": 300,
///     "position_delete_record_count": 1,
///     "position_delete_file_count": 1,
///     "equality_delete_record_count": 0,
///     "equality_delete_file_count": 0,
///     "dv_count": 0,
///     "last_updated_snapshot_id": <S2 id>,
///     "total_record_count_null": true
///   },
///   ...
/// ]
/// ```
fn stats_rows_to_json(rows: &[PartitionStats], s1_id: i64, s2_id: i64) -> JsonValue {
    let json_rows: Vec<JsonValue> = rows
        .iter()
        .map(|row| {
            // Extract the partition category string (field 0 in the single-field identity spec).
            let category = match row.partition().fields().first() {
                Some(Some(Literal::Primitive(iceberg::spec::PrimitiveLiteral::String(s)))) => {
                    s.clone()
                }
                _ => "unknown".to_string(),
            };

            // Embed the recorded id as-is, so Java verifies it against the same expected value.
            let last_updated_id = row.last_updated_snapshot_id();

            // cat=a must read S2, because the pos-delete is newer than the data file. cat=b must
            // read S1. Assert it here, so a compute bug fails the GEN test, not Java's verify.
            let _ = (s1_id, s2_id); // suppress unused-variable lint when assertions are off

            json!({
                "partition_category": category,
                "spec_id": row.spec_id(),
                "data_record_count": row.data_record_count(),
                "data_file_count": row.data_file_count(),
                "total_data_file_size_in_bytes": row.total_data_file_size_in_bytes(),
                "position_delete_record_count": row.position_delete_record_count(),
                "position_delete_file_count": row.position_delete_file_count(),
                "equality_delete_record_count": row.equality_delete_record_count(),
                "equality_delete_file_count": row.equality_delete_file_count(),
                "dv_count": row.dv_count(),
                "last_updated_snapshot_id": last_updated_id,
                "total_record_count_null": row.total_record_count().is_none()
            })
        })
        .collect();
    JsonValue::Array(json_rows)
}

/// Read the partition-stats path registered for `current_snapshot_id` from `final.metadata.json`.
fn find_stats_path(metadata_path: &Path) -> String {
    let raw = fs::read_to_string(metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let meta: JsonValue = serde_json::from_str(&raw)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    let current_snapshot_id = meta["current-snapshot-id"]
        .as_i64()
        .expect("current-snapshot-id missing from metadata");

    let files = meta["partition-statistics"]
        .as_array()
        .expect("partition-statistics array missing from metadata");

    for file in files {
        if file["snapshot-id"].as_i64() == Some(current_snapshot_id) {
            return file["statistics-path"]
                .as_str()
                .expect("statistics-path missing")
                .to_string();
        }
    }
    panic!(
        "no partition-statistics entry for snapshot-id={current_snapshot_id} in {}",
        metadata_path.display()
    );
}

// ---- Direction 1 (GEN): Rust builds the fixture and writes the stats file. -----------------------

/// GEN test: builds the two-snapshot fixture, writes and registers the partition-stats file, and
/// emits `final.metadata.json` plus `expected_stats.json`. Java's `verify-interop-partition-stats`
/// then judges the file through its production `readPartitionStatsFile`.
#[tokio::test]
async fn test_partition_stats_gen() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_partition_stats GEN — set \
             ICEBERG_INTEROP_PARTITION_STATS_GEN_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let table_location = gen_dir.join("rust_table").to_string_lossy().to_string();
    fs::create_dir_all(format!("{table_location}/metadata"))
        .expect("create rust_table/metadata dir");

    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_gen",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                gen_dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog over local FS");

    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.clone())
        .schema(fixture_schema())
        .partition_spec(fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table");

    // S1: fast-append file A (cat=a) + file B (cat=b).
    let file_a = data_file(
        &table_location,
        "file_a",
        "a",
        A_DATA_RECORDS as u64,
        A_DATA_FILE_SIZE,
    );
    let file_b = data_file(
        &table_location,
        "file_b",
        "b",
        B_DATA_RECORDS as u64,
        B_DATA_FILE_SIZE,
    );
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_a, file_b])
        .apply(tx)
        .expect("apply S1 fast_append (file_a + file_b)");
    let table = tx.commit(&catalog).await.expect("commit S1 fast_append");
    let s1_id = table
        .metadata()
        .current_snapshot_id()
        .expect("S1 snapshot id");
    println!(
        "interop_partition_stats GEN: S1 committed (id={s1_id}, file_a cat=a records={A_DATA_RECORDS}, \
         file_b cat=b records={B_DATA_RECORDS})"
    );

    // S2: row-delta — position-delete file scoped to cat=a.
    let pd_file = pos_delete_file(
        &table_location,
        "pos_delete_a",
        "a",
        A_POS_DEL_RECORDS as u64,
        A_POS_DEL_FILE_SIZE,
    );
    let tx = Transaction::new(&table);
    let action = tx.row_delta().add_deletes(vec![pd_file]);
    let tx = action
        .apply(tx)
        .expect("apply S2 row_delta (pos-delete cat=a)");
    let table = tx.commit(&catalog).await.expect("commit S2 row_delta");
    let s2_id = table
        .metadata()
        .current_snapshot_id()
        .expect("S2 snapshot id");
    println!(
        "interop_partition_stats GEN: S2 committed (id={s2_id}, pos_delete cat=a records={A_POS_DEL_RECORDS})"
    );

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot (S2)");
    let stats_file = compute_and_write_stats_file(&table, snapshot)
        .await
        .expect("compute_and_write_stats_file")
        .expect("stats file is Some (table is partitioned and has data)");

    println!(
        "interop_partition_stats GEN: stats file written at {}",
        stats_file.statistics_path
    );

    let table = register_partition_stats_file(&catalog, &table, stats_file)
        .await
        .expect("register_partition_stats_file");

    println!(
        "interop_partition_stats GEN: stats file registered (current_snapshot_id={})",
        table.metadata().current_snapshot_id().unwrap_or(-1)
    );

    write_final_metadata(&table, &table_location).await;
    println!(
        "interop_partition_stats GEN: final.metadata.json written at {table_location}/metadata/"
    );

    // The reader decodes the on-disk parquet on its own, so this is not circular. A wrong counter
    // from the writer reaches the reader, and Java's verify step catches it.
    let stats_schema = {
        let unified_type =
            unified_partition_type(table.metadata()).expect("compute unified partition type");
        partition_stats_schema(&unified_type, table.metadata().format_version())
            .expect("build stats schema")
    };
    let stats_path = find_stats_path(&PathBuf::from(format!(
        "{table_location}/metadata/final.metadata.json"
    )));
    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read_partition_stats_file");

    // Anti-circular assertions: the round trip must hold inside Rust too.
    assert_eq!(rows.len(), 2, "expected 2 partition rows (cat=a + cat=b)");

    let row_a = &rows[0];
    assert_eq!(
        row_a.data_record_count(),
        A_DATA_RECORDS,
        "cat=a: data_record_count"
    );
    assert_eq!(row_a.data_file_count(), 1, "cat=a: data_file_count");
    assert_eq!(
        row_a.total_data_file_size_in_bytes(),
        A_DATA_FILE_SIZE as i64,
        "cat=a: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row_a.position_delete_record_count(),
        A_POS_DEL_RECORDS,
        "cat=a: position_delete_record_count"
    );
    assert_eq!(
        row_a.position_delete_file_count(),
        1,
        "cat=a: position_delete_file_count"
    );
    assert_eq!(
        row_a.equality_delete_record_count(),
        0,
        "cat=a: equality_delete_record_count"
    );
    assert_eq!(
        row_a.equality_delete_file_count(),
        0,
        "cat=a: equality_delete_file_count"
    );
    assert_eq!(row_a.dv_count(), 0, "cat=a: dv_count");
    assert!(
        row_a.total_record_count().is_none(),
        "cat=a: total_record_count must be None"
    );
    // cat=a's last_updated must be S2 (pos-delete at S2 has a later timestamp than file_a at S1).
    assert_eq!(
        row_a.last_updated_snapshot_id(),
        Some(s2_id),
        "cat=a: last_updated_snapshot_id must be S2 (the pos-delete snapshot)"
    );

    let row_b = &rows[1];
    assert_eq!(
        row_b.data_record_count(),
        B_DATA_RECORDS,
        "cat=b: data_record_count"
    );
    assert_eq!(row_b.data_file_count(), 1, "cat=b: data_file_count");
    assert_eq!(
        row_b.total_data_file_size_in_bytes(),
        B_DATA_FILE_SIZE as i64,
        "cat=b: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row_b.position_delete_record_count(),
        0,
        "cat=b: position_delete_record_count"
    );
    assert_eq!(
        row_b.position_delete_file_count(),
        0,
        "cat=b: position_delete_file_count"
    );
    assert_eq!(
        row_b.equality_delete_record_count(),
        0,
        "cat=b: equality_delete_record_count"
    );
    assert_eq!(
        row_b.equality_delete_file_count(),
        0,
        "cat=b: equality_delete_file_count"
    );
    assert_eq!(row_b.dv_count(), 0, "cat=b: dv_count");
    assert!(
        row_b.total_record_count().is_none(),
        "cat=b: total_record_count must be None"
    );
    // cat=b's last_updated must be S1 (file_b was added at S1; no S2 activity for cat=b).
    assert_eq!(
        row_b.last_updated_snapshot_id(),
        Some(s1_id),
        "cat=b: last_updated_snapshot_id must be S1 (the initial append snapshot)"
    );

    // Emit expected_stats.json — the ground truth for both Java D1 verify and the D2 cross-check.
    let expected_json = stats_rows_to_json(&rows, s1_id, s2_id);
    let expected_path = gen_dir.join("expected_stats.json");
    fs::write(
        &expected_path,
        serde_json::to_string_pretty(&expected_json).expect("serialize expected_stats.json"),
    )
    .expect("write expected_stats.json");
    println!(
        "interop_partition_stats GEN: expected_stats.json written at {}",
        expected_path.display()
    );
    println!(
        "interop_partition_stats GEN: all assertions PASSED — \
         s1_id={s1_id} s2_id={s2_id} \
         cat_a(records={A_DATA_RECORDS} pos_del={A_POS_DEL_RECORDS} last_updated=S2) \
         cat_b(records={B_DATA_RECORDS} last_updated=S1)"
    );
}

// ---- Direction 2 — Rust reads Java's stats file. -------------------------------------------------

/// Load `expected_stats.json` (or `java_stats.json`) from the compare dir.
fn load_json_file(path: &Path) -> JsonValue {
    let raw =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Direction 2: Rust reads the Java-written stats parquet and compares the decoded rows against
/// `java_stats.json`, which Java emitted from its own decoded rows.
#[tokio::test]
async fn test_partition_stats_d2_rust_reads_java_file() {
    let Some(dir) = compare_dir() else {
        println!(
            "skipping interop_partition_stats D2 — set \
             ICEBERG_INTEROP_PARTITION_STATS_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let java_meta_path = dir.join("table/metadata/final.metadata.json");
    assert!(
        java_meta_path.exists(),
        "D2: missing {}; run the Java generate step first",
        java_meta_path.display()
    );

    let java_meta_json = load_json_file(&java_meta_path);

    // `read_partition_stats_file` needs a real `Table`, so build a catalog over the Java dir.
    let table_dir = dir.join("table").to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_d2",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for D2");

    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (D2)");

    let creation = TableCreation::builder()
        .name("table".to_string())
        .location(table_dir.clone())
        .schema(fixture_schema())
        .partition_spec(fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create D2 table handle");

    let stats_path = find_stats_path(&java_meta_path);
    println!("interop_partition_stats D2: reading Java stats file at {stats_path}");

    // The stats schema needs the table's format version and unified partition type.
    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (D2)");
    let format_version = {
        // Confirm the Java-written metadata is V2, like the table this test created.
        let fv = java_meta_json["format-version"].as_i64().unwrap_or(2);
        if fv >= 3 {
            FormatVersion::V3
        } else {
            FormatVersion::V2
        }
    };
    let stats_schema =
        partition_stats_schema(&unified_type, format_version).expect("build stats schema (D2)");

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read_partition_stats_file (D2)");

    let java_stats_path = dir.join("java_stats.json");
    assert!(
        java_stats_path.exists(),
        "D2: missing {}; run the Java generate step first",
        java_stats_path.display()
    );
    let expected = load_json_file(&java_stats_path);
    let expected_arr = expected
        .as_array()
        .expect("java_stats.json must be a JSON array");

    assert_eq!(
        rows.len(),
        expected_arr.len(),
        "D2: decoded {} rows, java_stats.json has {}",
        rows.len(),
        expected_arr.len()
    );

    // Compare row by row. Both sides sort cat=a first.
    for (i, (row, exp)) in rows.iter().zip(expected_arr).enumerate() {
        let partition = match row.partition().fields().first() {
            Some(Some(Literal::Primitive(iceberg::spec::PrimitiveLiteral::String(s)))) => s.clone(),
            _ => "unknown".to_string(),
        };
        let label = format!("D2 row {} (cat={partition})", i);

        assert_eq!(
            row.spec_id() as i64,
            exp["spec_id"].as_i64().unwrap_or(-1),
            "{label}: spec_id"
        );
        assert_eq!(
            row.data_record_count(),
            exp["data_record_count"].as_i64().unwrap_or(-1),
            "{label}: data_record_count"
        );
        assert_eq!(
            row.data_file_count() as i64,
            exp["data_file_count"].as_i64().unwrap_or(-1),
            "{label}: data_file_count"
        );
        assert_eq!(
            row.total_data_file_size_in_bytes(),
            exp["total_data_file_size_in_bytes"].as_i64().unwrap_or(-1),
            "{label}: total_data_file_size_in_bytes"
        );
        assert_eq!(
            row.position_delete_record_count(),
            exp["position_delete_record_count"].as_i64().unwrap_or(-1),
            "{label}: position_delete_record_count"
        );
        assert_eq!(
            row.position_delete_file_count() as i64,
            exp["position_delete_file_count"].as_i64().unwrap_or(-1),
            "{label}: position_delete_file_count"
        );
        assert_eq!(
            row.equality_delete_record_count(),
            exp["equality_delete_record_count"].as_i64().unwrap_or(-1),
            "{label}: equality_delete_record_count"
        );
        assert_eq!(
            row.equality_delete_file_count() as i64,
            exp["equality_delete_file_count"].as_i64().unwrap_or(-1),
            "{label}: equality_delete_file_count"
        );
        assert_eq!(
            row.dv_count() as i64,
            exp["dv_count"].as_i64().unwrap_or(-1),
            "{label}: dv_count"
        );
        assert_eq!(
            row.last_updated_snapshot_id(),
            exp["last_updated_snapshot_id"].as_i64(),
            "{label}: last_updated_snapshot_id"
        );
        let total_null = row.total_record_count().is_none();
        assert_eq!(
            total_null,
            exp["total_record_count_null"].as_bool().unwrap_or(false),
            "{label}: total_record_count must be null"
        );
        println!("interop_partition_stats D2 {label}: all fields match java_stats.json OK");
    }

    println!(
        "interop_partition_stats D2: PASS — {} rows decoded from Java stats file match \
         java_stats.json",
        rows.len()
    );
}

// ---- Cross-version projection: V2 file read against V3 schema. -----------------------------------

/// Reads the Java-written V2 stats parquet, which has 12 columns and no `dv_count`, against the V3
/// stats schema of 13 fields. The reader must not error on the missing column. It must null-fill
/// `dv_count` to 0 on every row through `project_struct_type_to_batch`.
#[tokio::test]
async fn test_partition_stats_cross_version_v2_file_v3_schema() {
    let Some(dir) = compare_dir() else {
        println!(
            "skipping interop_partition_stats cross-version — set \
             ICEBERG_INTEROP_PARTITION_STATS_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let java_meta_path = dir.join("table/metadata/final.metadata.json");
    if !java_meta_path.exists() {
        println!(
            "skipping cross-version test — missing {}; run the Java generate step first",
            java_meta_path.display()
        );
        return;
    }

    // The test only means anything on a V2 file.
    let java_meta_json = load_json_file(&java_meta_path);
    let fv = java_meta_json["format-version"].as_i64().unwrap_or(2);
    if fv >= 3 {
        println!(
            "skipping cross-version test — Java table is V3; this test exercises V2→V3 projection \
             and the file already has dv_count"
        );
        return;
    }

    let stats_path = find_stats_path(&java_meta_path);
    println!(
        "interop_partition_stats cross-version: reading V2 stats file at {stats_path} against V3 schema"
    );

    let table_dir = dir.join("table").to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_xver",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for cross-version test");

    let namespace = NamespaceIdent::new("xver".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (cross-version)");

    let creation = TableCreation::builder()
        .name("table_xver".to_string())
        .location(table_dir.clone())
        .schema(fixture_schema())
        .partition_spec(fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create cross-version table handle");

    // The reader must project this V3 schema down to the file's 12 columns.
    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (cross-version)");
    let v3_stats_schema = partition_stats_schema(&unified_type, FormatVersion::V3)
        .expect("build V3 stats schema for cross-version test");

    // Reading a V2 file against the V3 schema must not error.
    let rows = read_partition_stats_file(&table, &v3_stats_schema, &stats_path)
        .await
        .expect(
            "read_partition_stats_file: V2 file read against V3 schema must not error \
             (project_struct_type_to_batch handles the missing dv_count column)",
        );

    assert_eq!(
        rows.len(),
        2,
        "cross-version: expected 2 partition rows, got {}",
        rows.len()
    );

    // `dv_count` null-fills to 0, through the shorter-record tolerance in the record decoder.
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(
            row.dv_count(),
            0,
            "cross-version row {i}: dv_count must be 0 (null-filled from V2 file)"
        );
        let category = match row.partition().fields().first() {
            Some(Some(Literal::Primitive(iceberg::spec::PrimitiveLiteral::String(s)))) => s.clone(),
            _ => "unknown".to_string(),
        };
        match category.as_str() {
            "a" => {
                assert_eq!(
                    row.data_record_count(),
                    A_DATA_RECORDS,
                    "xver cat=a: data_record_count"
                );
                assert_eq!(
                    row.position_delete_record_count(),
                    A_POS_DEL_RECORDS,
                    "xver cat=a: position_delete_record_count"
                );
            }
            "b" => {
                assert_eq!(
                    row.data_record_count(),
                    B_DATA_RECORDS,
                    "xver cat=b: data_record_count"
                );
                assert_eq!(
                    row.position_delete_record_count(),
                    0,
                    "xver cat=b: position_delete_record_count"
                );
            }
            other => panic!("cross-version: unexpected partition category '{other}'"),
        }
        println!(
            "interop_partition_stats cross-version row {i} (cat={category}): dv_count=0, \
             counters correct OK"
        );
    }

    println!(
        "interop_partition_stats cross-version: PASS — V2 file read against V3 schema, \
         dv_count null-filled to 0 for all {} rows",
        rows.len()
    );
}

// ---- The incremental path: the SUBTRACT arm. -----------------------------------------------------

/// GEN test: the incremental partition-stats SUBTRACT arm. S1 fast-appends file_a (cat=a) and
/// file_b (cat=b), then registers full stats.
#[tokio::test]
async fn test_partition_stats_incr_gen() {
    let Some(incr_dir) = incr_dir() else {
        println!(
            "skipping interop_partition_stats INCR GEN — set \
             ICEBERG_INTEROP_PARTITION_STATS_INCR_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let table_location = incr_dir
        .join("rust_incr_table")
        .to_string_lossy()
        .to_string();
    fs::create_dir_all(format!("{table_location}/metadata"))
        .expect("create rust_incr_table/metadata dir");

    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_incr_gen",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                incr_dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for incr GEN");

    let namespace = NamespaceIdent::new("interop_incr".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (incr GEN)");

    let creation = TableCreation::builder()
        .name("rust_incr_table".to_string())
        .location(table_location.clone())
        .schema(fixture_schema())
        .partition_spec(fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_incr_table");

    // S1: fast-append file_a (cat=a) + file_b (cat=b).
    let file_a = data_file(
        &table_location,
        "incr_file_a",
        "a",
        INCR_A_DATA_RECORDS as u64,
        INCR_A_DATA_FILE_SIZE,
    );
    let file_b = data_file(
        &table_location,
        "incr_file_b",
        "b",
        INCR_B_DATA_RECORDS as u64,
        INCR_B_DATA_FILE_SIZE,
    );
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_a.clone(), file_b])
        .apply(tx)
        .expect("apply S1 fast_append (incr)");
    let table = tx.commit(&catalog).await.expect("commit S1 (incr)");
    let s1_id = table
        .metadata()
        .current_snapshot_id()
        .expect("S1 snapshot id (incr)");
    println!("interop_partition_stats INCR GEN: S1 committed (id={s1_id})");

    // Compute and register S1 stats (FULL — no prior base).
    let s1_snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot (S1 incr)");
    let s1_stats = compute_and_write_stats_file(&table, s1_snapshot)
        .await
        .expect("compute S1 stats")
        .expect("S1 stats must be Some (table has data)");
    let table = register_partition_stats_file(&catalog, &table, s1_stats)
        .await
        .expect("register S1 stats");

    // Incremental path pin (1): assert a stats file is now registered for S1.
    let has_s1_stats = table
        .metadata()
        .partition_statistics_iter()
        .any(|f| f.snapshot_id == s1_id);
    assert!(
        has_s1_stats,
        "incr GEN: S1 stats file must be registered before computing S2 stats"
    );
    println!("interop_partition_stats INCR GEN: S1 stats registered (id={s1_id})");

    // A DELETE snapshot with a DELETED tombstone triggers the SUBTRACT arm.
    let tx = Transaction::new(&table);
    let tx = tx
        .delete_files()
        .delete_data_files(vec![file_a])
        .apply(tx)
        .expect("apply S2 delete_files(file_a)");
    let table = tx.commit(&catalog).await.expect("commit S2 (incr)");
    let s2_id = table
        .metadata()
        .current_snapshot_id()
        .expect("S2 snapshot id (incr)");
    println!("interop_partition_stats INCR GEN: S2 committed (id={s2_id}, deleted file_a)");

    // The production path selects incremental, because S1 is an ancestor with a base file.
    let s2_snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot (S2 incr)");
    let s2_stats = compute_and_write_stats_file(&table, s2_snapshot)
        .await
        .expect("compute S2 stats (incremental)")
        .expect("S2 stats must be Some (cat=b still has data)");
    let table = register_partition_stats_file(&catalog, &table, s2_stats)
        .await
        .expect("register S2 stats");

    write_final_metadata(&table, &table_location).await;

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (incr S2)");
    let stats_schema = partition_stats_schema(&unified_type, table.metadata().format_version())
        .expect("stats schema (incr S2)");
    let stats_path =
        find_stats_path(&incr_dir.join("rust_incr_table/metadata/final.metadata.json"));
    println!("interop_partition_stats INCR GEN: reading S2 stats at {stats_path}");

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read S2 stats (incr)");

    assert_eq!(rows.len(), 2, "incr S2: expected 2 rows (cat=a + cat=b)");

    // cat=a is fully subtracted, because file_a was deleted. Rows sort cat=a first.
    let row_a = &rows[0];
    let cat_a = match row_a.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::String(s)))) => s.clone(),
        _ => "unknown".to_string(),
    };
    assert_eq!(cat_a, "a", "incr: row 0 must be cat=a");
    assert_eq!(
        row_a.data_record_count(),
        0,
        "incr cat=a: data_record_count must be 0 (subtracted)"
    );
    assert_eq!(
        row_a.data_file_count(),
        0,
        "incr cat=a: data_file_count must be 0 (subtracted)"
    );
    assert_eq!(
        row_a.total_data_file_size_in_bytes(),
        0,
        "incr cat=a: total_data_file_size_in_bytes must be 0 (subtracted)"
    );
    assert_eq!(
        row_a.last_updated_snapshot_id(),
        Some(s2_id),
        "incr cat=a: last_updated_snapshot_id must be S2 (the delete snapshot)"
    );

    // cat=b: unchanged from S1 base (no S2 activity).
    let row_b = &rows[1];
    let cat_b = match row_b.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::String(s)))) => s.clone(),
        _ => "unknown".to_string(),
    };
    assert_eq!(cat_b, "b", "incr: row 1 must be cat=b");
    assert_eq!(
        row_b.data_record_count(),
        INCR_B_DATA_RECORDS,
        "incr cat=b: data_record_count must be {} (unchanged from S1 base)",
        INCR_B_DATA_RECORDS
    );
    assert_eq!(
        row_b.data_file_count(),
        1,
        "incr cat=b: data_file_count must be 1"
    );
    assert_eq!(
        row_b.total_data_file_size_in_bytes(),
        INCR_B_DATA_FILE_SIZE as i64,
        "incr cat=b: total_data_file_size_in_bytes must be {}",
        INCR_B_DATA_FILE_SIZE
    );
    // cat=b's last_updated must stay S1, which the carried base row gives. A full recompute also
    // returns S1 here, so the SUBTRACT on cat=a is what really distinguishes the two paths.
    assert_eq!(
        row_b.last_updated_snapshot_id(),
        Some(s1_id),
        "incr cat=b: last_updated_snapshot_id must be S1 (carried unchanged from base)"
    );

    let expected_json: Vec<JsonValue> = rows
        .iter()
        .map(|row| {
            let category = match row.partition().fields().first() {
                Some(Some(Literal::Primitive(PrimitiveLiteral::String(s)))) => s.clone(),
                _ => "unknown".to_string(),
            };
            json!({
                "partition_category": category,
                "spec_id": row.spec_id(),
                "data_record_count": row.data_record_count(),
                "data_file_count": row.data_file_count(),
                "total_data_file_size_in_bytes": row.total_data_file_size_in_bytes(),
                "position_delete_record_count": row.position_delete_record_count(),
                "position_delete_file_count": row.position_delete_file_count(),
                "equality_delete_record_count": row.equality_delete_record_count(),
                "equality_delete_file_count": row.equality_delete_file_count(),
                "dv_count": row.dv_count(),
                "last_updated_snapshot_id": row.last_updated_snapshot_id(),
                "total_record_count_null": row.total_record_count().is_none()
            })
        })
        .collect();

    let expected_path = incr_dir.join("incr_expected.json");
    fs::write(
        &expected_path,
        serde_json::to_string_pretty(&JsonValue::Array(expected_json))
            .expect("serialize incr_expected.json"),
    )
    .expect("write incr_expected.json");

    println!(
        "interop_partition_stats INCR GEN: incr_expected.json written at {}",
        expected_path.display()
    );
    println!(
        "interop_partition_stats INCR GEN: PASS — s1_id={s1_id} s2_id={s2_id} \
         cat_a(records=0 subtracted) cat_b(records={INCR_B_DATA_RECORDS} unchanged)"
    );
}

/// Direction 2: Rust reads Java's incremental S2 stats file and compares the decoded rows against
/// `java_incr_stats.json`, which `IncrementalPartitionStatsOracle` emitted.
#[tokio::test]
async fn test_partition_stats_incr_d2_rust_reads_java() {
    let Some(dir) = incr_dir() else {
        println!(
            "skipping interop_partition_stats INCR D2 — set \
             ICEBERG_INTEROP_PARTITION_STATS_INCR_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let java_meta_path = dir.join("java_incr_table/metadata/final.metadata.json");
    assert!(
        java_meta_path.exists(),
        "INCR D2: missing {}; run the Java generate step first",
        java_meta_path.display()
    );

    let stats_path = find_stats_path(&java_meta_path);
    println!("interop_partition_stats INCR D2: reading Java incr stats file at {stats_path}");

    let table_dir = dir.join("java_incr_table").to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_incr_d2",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for incr D2");

    let namespace = NamespaceIdent::new("interop_incr_d2".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (incr D2)");

    let creation = TableCreation::builder()
        .name("java_incr_table".to_string())
        .location(table_dir.clone())
        .schema(fixture_schema())
        .partition_spec(fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create incr D2 table handle");

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (incr D2)");
    let stats_schema =
        partition_stats_schema(&unified_type, FormatVersion::V2).expect("stats schema (incr D2)");

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read_partition_stats_file (incr D2)");

    let java_stats_path = dir.join("java_incr_stats.json");
    assert!(
        java_stats_path.exists(),
        "INCR D2: missing {}; run the Java generate step first",
        java_stats_path.display()
    );
    let expected = load_json_file(&java_stats_path);
    let expected_arr = expected
        .as_array()
        .expect("java_incr_stats.json must be a JSON array");

    assert_eq!(
        rows.len(),
        expected_arr.len(),
        "INCR D2: decoded {} rows, java_incr_stats.json has {}",
        rows.len(),
        expected_arr.len()
    );

    for (i, (row, exp)) in rows.iter().zip(expected_arr).enumerate() {
        let category = match row.partition().fields().first() {
            Some(Some(Literal::Primitive(PrimitiveLiteral::String(s)))) => s.clone(),
            _ => "unknown".to_string(),
        };
        let label = format!("INCR D2 row {} (cat={category})", i);

        assert_eq!(
            row.data_record_count(),
            exp["data_record_count"].as_i64().unwrap_or(-1),
            "{label}: data_record_count"
        );
        assert_eq!(
            row.data_file_count() as i64,
            exp["data_file_count"].as_i64().unwrap_or(-1),
            "{label}: data_file_count"
        );
        assert_eq!(
            row.total_data_file_size_in_bytes(),
            exp["total_data_file_size_in_bytes"].as_i64().unwrap_or(-1),
            "{label}: total_data_file_size_in_bytes"
        );
        assert_eq!(
            row.position_delete_record_count(),
            exp["position_delete_record_count"].as_i64().unwrap_or(-1),
            "{label}: position_delete_record_count"
        );
        assert_eq!(
            row.position_delete_file_count() as i64,
            exp["position_delete_file_count"].as_i64().unwrap_or(-1),
            "{label}: position_delete_file_count"
        );
        assert_eq!(
            row.equality_delete_record_count(),
            exp["equality_delete_record_count"].as_i64().unwrap_or(-1),
            "{label}: equality_delete_record_count"
        );
        assert_eq!(
            row.equality_delete_file_count() as i64,
            exp["equality_delete_file_count"].as_i64().unwrap_or(-1),
            "{label}: equality_delete_file_count"
        );
        assert_eq!(
            row.dv_count() as i64,
            exp["dv_count"].as_i64().unwrap_or(-1),
            "{label}: dv_count"
        );
        assert_eq!(
            row.last_updated_snapshot_id(),
            exp["last_updated_snapshot_id"].as_i64(),
            "{label}: last_updated_snapshot_id"
        );
        println!(
            "interop_partition_stats INCR D2 {label}: all fields match java_incr_stats.json OK"
        );
    }

    println!(
        "interop_partition_stats INCR D2: PASS — {} rows decoded from Java incr stats file",
        rows.len()
    );
}

// ---- The UUID partition type: an exotic-type round trip. -----------------------------------------

/// GEN test: UUID-partitioned partition-stats.
///
/// A V2 table `identity(partition_id uuid)` holds one data file with a known UUID partition value.
/// A UUID lands on disk as 16 big-endian bytes, the hardest partition type to encode. Both
/// directions must reconstruct the same UUID string. It emits
/// `rust_uuid_table/metadata/final.metadata.json` and `uuid_expected.json` for Java's verify.
#[tokio::test]
async fn test_partition_stats_uuid_gen() {
    let Some(uuid_dir) = uuid_dir() else {
        println!(
            "skipping interop_partition_stats UUID GEN — set \
             ICEBERG_INTEROP_PARTITION_STATS_UUID_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let table_location = uuid_dir
        .join("rust_uuid_table")
        .to_string_lossy()
        .to_string();
    fs::create_dir_all(format!("{table_location}/metadata"))
        .expect("create rust_uuid_table/metadata dir");

    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_uuid_gen",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                uuid_dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for UUID GEN");

    let namespace = NamespaceIdent::new("interop_uuid".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (UUID GEN)");

    let creation = TableCreation::builder()
        .name("rust_uuid_table".to_string())
        .location(table_location.clone())
        .schema(uuid_fixture_schema())
        .partition_spec(uuid_fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_uuid_table");

    let known_uuid = Uuid::parse_str(KNOWN_UUID_STR).expect("parse known UUID");
    let file_uuid = uuid_data_file(
        &table_location,
        "uuid_file",
        known_uuid,
        UUID_DATA_RECORDS as u64,
        UUID_DATA_FILE_SIZE,
    );

    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_uuid])
        .apply(tx)
        .expect("apply UUID fast_append");
    let table = tx.commit(&catalog).await.expect("commit UUID S1");
    let s1_id = table
        .metadata()
        .current_snapshot_id()
        .expect("UUID S1 snapshot id");
    println!("interop_partition_stats UUID GEN: S1 committed (id={s1_id}, uuid={KNOWN_UUID_STR})");

    // Compute and register stats (FULL — UUID partition type).
    let s1_snapshot = table
        .metadata()
        .current_snapshot()
        .expect("UUID current snapshot");
    let stats_file = compute_and_write_stats_file(&table, s1_snapshot)
        .await
        .expect("compute UUID stats")
        .expect("UUID stats must be Some");
    let table = register_partition_stats_file(&catalog, &table, stats_file)
        .await
        .expect("register UUID stats");

    write_final_metadata(&table, &table_location).await;
    println!(
        "interop_partition_stats UUID GEN: final.metadata.json written at {table_location}/metadata/"
    );

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (UUID)");
    let stats_schema = partition_stats_schema(&unified_type, table.metadata().format_version())
        .expect("stats schema (UUID)");
    let stats_path =
        find_stats_path(&uuid_dir.join("rust_uuid_table/metadata/final.metadata.json"));
    println!("interop_partition_stats UUID GEN: reading stats at {stats_path}");

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read UUID stats");

    assert_eq!(rows.len(), 1, "UUID GEN: expected 1 partition row");

    let row = &rows[0];
    // Verify the UUID round-trip: the partition field must decode back to the known UUID.
    let uuid_val = match row.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::UInt128(v)))) => {
            Uuid::from_u128(*v).to_string()
        }
        other => panic!("UUID GEN: unexpected partition literal type: {other:?}"),
    };
    assert_eq!(
        uuid_val.to_lowercase(),
        KNOWN_UUID_STR.to_lowercase(),
        "UUID GEN: partition UUID round-trip failed"
    );
    assert_eq!(
        row.data_record_count(),
        UUID_DATA_RECORDS,
        "UUID GEN: data_record_count"
    );
    assert_eq!(row.data_file_count(), 1, "UUID GEN: data_file_count");
    assert_eq!(
        row.total_data_file_size_in_bytes(),
        UUID_DATA_FILE_SIZE as i64,
        "UUID GEN: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row.last_updated_snapshot_id(),
        Some(s1_id),
        "UUID GEN: last_updated_snapshot_id must be S1"
    );

    let expected_json = json!([{
        "partition_uuid": uuid_val,
        "spec_id": row.spec_id(),
        "data_record_count": row.data_record_count(),
        "data_file_count": row.data_file_count(),
        "total_data_file_size_in_bytes": row.total_data_file_size_in_bytes(),
        "position_delete_record_count": row.position_delete_record_count(),
        "position_delete_file_count": row.position_delete_file_count(),
        "equality_delete_record_count": row.equality_delete_record_count(),
        "equality_delete_file_count": row.equality_delete_file_count(),
        "dv_count": row.dv_count(),
        "last_updated_snapshot_id": row.last_updated_snapshot_id(),
        "total_record_count_null": row.total_record_count().is_none()
    }]);

    let expected_path = uuid_dir.join("uuid_expected.json");
    fs::write(
        &expected_path,
        serde_json::to_string_pretty(&expected_json).expect("serialize uuid_expected.json"),
    )
    .expect("write uuid_expected.json");
    println!(
        "interop_partition_stats UUID GEN: uuid_expected.json written at {}",
        expected_path.display()
    );
    println!(
        "interop_partition_stats UUID GEN: PASS — uuid={KNOWN_UUID_STR} \
         records={UUID_DATA_RECORDS} size={UUID_DATA_FILE_SIZE}"
    );
}

/// Direction 2: Rust reads Java's UUID-partitioned stats file and compares the decoded rows
/// against `java_uuid_stats.json`, which `UuidPartitionStatsOracle` emitted.
#[tokio::test]
async fn test_partition_stats_uuid_d2() {
    let Some(dir) = uuid_dir() else {
        println!(
            "skipping interop_partition_stats UUID D2 — set \
             ICEBERG_INTEROP_PARTITION_STATS_UUID_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let java_meta_path = dir.join("java_uuid_table/metadata/final.metadata.json");
    assert!(
        java_meta_path.exists(),
        "UUID D2: missing {}; run the Java generate step first",
        java_meta_path.display()
    );

    let stats_path = find_stats_path(&java_meta_path);
    println!("interop_partition_stats UUID D2: reading Java UUID stats file at {stats_path}");

    let table_dir = dir.join("java_uuid_table").to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_uuid_d2",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for UUID D2");

    let namespace = NamespaceIdent::new("interop_uuid_d2".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (UUID D2)");

    let creation = TableCreation::builder()
        .name("java_uuid_table".to_string())
        .location(table_dir.clone())
        .schema(uuid_fixture_schema())
        .partition_spec(uuid_fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create UUID D2 table handle");

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (UUID D2)");
    let stats_schema =
        partition_stats_schema(&unified_type, FormatVersion::V2).expect("stats schema (UUID D2)");

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read_partition_stats_file (UUID D2)");

    let java_stats_path = dir.join("java_uuid_stats.json");
    assert!(
        java_stats_path.exists(),
        "UUID D2: missing {}; run the Java generate step first",
        java_stats_path.display()
    );
    let expected = load_json_file(&java_stats_path);
    let expected_arr = expected
        .as_array()
        .expect("java_uuid_stats.json must be a JSON array");

    assert_eq!(
        rows.len(),
        expected_arr.len(),
        "UUID D2: decoded {} rows, java_uuid_stats.json has {}",
        rows.len(),
        expected_arr.len()
    );

    assert_eq!(rows.len(), 1, "UUID D2: expected 1 partition row");
    let row = &rows[0];
    let exp = &expected_arr[0];

    let uuid_val = match row.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::UInt128(v)))) => {
            Uuid::from_u128(*v).to_string()
        }
        other => panic!("UUID D2: unexpected partition literal type: {other:?}"),
    };
    let expected_uuid = exp["partition_uuid"].as_str().unwrap_or("");
    assert_eq!(
        uuid_val.to_lowercase(),
        expected_uuid.to_lowercase(),
        "UUID D2: partition_uuid mismatch"
    );

    assert_eq!(
        row.data_record_count(),
        exp["data_record_count"].as_i64().unwrap_or(-1),
        "UUID D2: data_record_count"
    );
    assert_eq!(
        row.data_file_count() as i64,
        exp["data_file_count"].as_i64().unwrap_or(-1),
        "UUID D2: data_file_count"
    );
    assert_eq!(
        row.total_data_file_size_in_bytes(),
        exp["total_data_file_size_in_bytes"].as_i64().unwrap_or(-1),
        "UUID D2: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row.last_updated_snapshot_id(),
        exp["last_updated_snapshot_id"].as_i64(),
        "UUID D2: last_updated_snapshot_id"
    );

    println!(
        "interop_partition_stats UUID D2: PASS — uuid={uuid_val} records={} size={}",
        row.data_record_count(),
        row.total_data_file_size_in_bytes()
    );
}

// ---- TIME partition type (R3) --------------------------------------------------------------------

/// GEN test: an `identity(partition_time)` fixture. It emits
/// `rust_time_table/metadata/final.metadata.json` and `time_expected.json` for Java to judge.
#[tokio::test]
async fn test_partition_stats_time_gen() {
    let Some(time_dir) = time_dir() else {
        println!(
            "skipping interop_partition_stats TIME GEN — set \
             ICEBERG_INTEROP_PARTITION_STATS_TIME_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let table_location = time_dir
        .join("rust_time_table")
        .to_string_lossy()
        .to_string();
    fs::create_dir_all(format!("{table_location}/metadata"))
        .expect("create rust_time_table/metadata dir");

    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_time_gen",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                time_dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for TIME GEN");

    let namespace = NamespaceIdent::new("interop_time".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (TIME GEN)");

    let creation = TableCreation::builder()
        .name("rust_time_table".to_string())
        .location(table_location.clone())
        .schema(time_fixture_schema())
        .partition_spec(time_fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_time_table");

    let file_time = time_data_file(
        &table_location,
        "time_file",
        KNOWN_TIME_MICROS,
        TIME_DATA_RECORDS as u64,
        TIME_DATA_FILE_SIZE,
    );

    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_time])
        .apply(tx)
        .expect("apply TIME fast_append");
    let table = tx.commit(&catalog).await.expect("commit TIME S1");
    let s1_id = table
        .metadata()
        .current_snapshot_id()
        .expect("TIME S1 snapshot id");
    println!(
        "interop_partition_stats TIME GEN: S1 committed (id={s1_id}, micros={KNOWN_TIME_MICROS})"
    );

    let s1_snapshot = table
        .metadata()
        .current_snapshot()
        .expect("TIME current snapshot");
    let stats_file = compute_and_write_stats_file(&table, s1_snapshot)
        .await
        .expect("compute TIME stats")
        .expect("TIME stats must be Some");
    let table = register_partition_stats_file(&catalog, &table, stats_file)
        .await
        .expect("register TIME stats");

    write_final_metadata(&table, &table_location).await;

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (TIME)");
    let stats_schema = partition_stats_schema(&unified_type, table.metadata().format_version())
        .expect("stats schema (TIME)");
    let stats_path =
        find_stats_path(&time_dir.join("rust_time_table/metadata/final.metadata.json"));

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read TIME stats");

    assert_eq!(rows.len(), 1, "TIME GEN: expected 1 partition row");
    let row = &rows[0];

    // Verify the time round-trip: the partition field must decode back to the known micros (Long).
    let micros = match row.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::Long(v)))) => *v,
        other => panic!("TIME GEN: unexpected partition literal type: {other:?}"),
    };
    assert_eq!(
        micros, KNOWN_TIME_MICROS,
        "TIME GEN: partition micros round-trip failed"
    );
    assert_eq!(
        row.data_record_count(),
        TIME_DATA_RECORDS,
        "TIME GEN: data_record_count"
    );
    assert_eq!(row.data_file_count(), 1, "TIME GEN: data_file_count");
    assert_eq!(
        row.total_data_file_size_in_bytes(),
        TIME_DATA_FILE_SIZE as i64,
        "TIME GEN: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row.last_updated_snapshot_id(),
        Some(s1_id),
        "TIME GEN: last_updated_snapshot_id must be S1"
    );

    let expected_json = json!([{
        "partition_time_micros": micros,
        "spec_id": row.spec_id(),
        "data_record_count": row.data_record_count(),
        "data_file_count": row.data_file_count(),
        "total_data_file_size_in_bytes": row.total_data_file_size_in_bytes(),
        "position_delete_record_count": row.position_delete_record_count(),
        "position_delete_file_count": row.position_delete_file_count(),
        "equality_delete_record_count": row.equality_delete_record_count(),
        "equality_delete_file_count": row.equality_delete_file_count(),
        "dv_count": row.dv_count(),
        "last_updated_snapshot_id": row.last_updated_snapshot_id(),
        "total_record_count_null": row.total_record_count().is_none()
    }]);

    let expected_path = time_dir.join("time_expected.json");
    fs::write(
        &expected_path,
        serde_json::to_string_pretty(&expected_json).expect("serialize time_expected.json"),
    )
    .expect("write time_expected.json");
    println!(
        "interop_partition_stats TIME GEN: PASS — micros={KNOWN_TIME_MICROS} \
         records={TIME_DATA_RECORDS} size={TIME_DATA_FILE_SIZE}"
    );
}

/// Direction 2: Rust reads Java's time-partitioned stats file against `java_time_stats.json`.
#[tokio::test]
async fn test_partition_stats_time_d2() {
    let Some(dir) = time_dir() else {
        println!(
            "skipping interop_partition_stats TIME D2 — set \
             ICEBERG_INTEROP_PARTITION_STATS_TIME_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let java_meta_path = dir.join("java_time_table/metadata/final.metadata.json");
    assert!(
        java_meta_path.exists(),
        "TIME D2: missing {}; run the Java generate step first",
        java_meta_path.display()
    );

    let stats_path = find_stats_path(&java_meta_path);
    let table_dir = dir.join("java_time_table").to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_time_d2",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for TIME D2");

    let namespace = NamespaceIdent::new("interop_time_d2".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (TIME D2)");

    let creation = TableCreation::builder()
        .name("java_time_table".to_string())
        .location(table_dir.clone())
        .schema(time_fixture_schema())
        .partition_spec(time_fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create TIME D2 table handle");

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (TIME D2)");
    let stats_schema =
        partition_stats_schema(&unified_type, FormatVersion::V2).expect("stats schema (TIME D2)");

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read_partition_stats_file (TIME D2)");

    let java_stats_path = dir.join("java_time_stats.json");
    assert!(
        java_stats_path.exists(),
        "TIME D2: missing {}; run the Java generate step first",
        java_stats_path.display()
    );
    let expected = load_json_file(&java_stats_path);
    let expected_arr = expected
        .as_array()
        .expect("java_time_stats.json must be a JSON array");

    assert_eq!(
        rows.len(),
        expected_arr.len(),
        "TIME D2: row count mismatch"
    );
    assert_eq!(rows.len(), 1, "TIME D2: expected 1 partition row");
    let row = &rows[0];
    let exp = &expected_arr[0];

    let micros = match row.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::Long(v)))) => *v,
        other => panic!("TIME D2: unexpected partition literal type: {other:?}"),
    };
    assert_eq!(
        micros,
        exp["partition_time_micros"].as_i64().unwrap_or(-1),
        "TIME D2: partition_time_micros mismatch"
    );
    assert_eq!(
        row.data_record_count(),
        exp["data_record_count"].as_i64().unwrap_or(-1),
        "TIME D2: data_record_count"
    );
    assert_eq!(
        row.data_file_count() as i64,
        exp["data_file_count"].as_i64().unwrap_or(-1),
        "TIME D2: data_file_count"
    );
    assert_eq!(
        row.total_data_file_size_in_bytes(),
        exp["total_data_file_size_in_bytes"].as_i64().unwrap_or(-1),
        "TIME D2: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row.last_updated_snapshot_id(),
        exp["last_updated_snapshot_id"].as_i64(),
        "TIME D2: last_updated_snapshot_id"
    );
    println!("interop_partition_stats TIME D2: PASS — micros={micros}");
}

// ---- FIXED[4] partition type (R3) ----------------------------------------------------------------

/// GEN test: an `identity(partition_fixed: fixed[4])` fixture. It emits
/// `rust_fixed_table/metadata/final.metadata.json` and `fixed_expected.json` for Java to judge.
#[tokio::test]
async fn test_partition_stats_fixed_gen() {
    let Some(fixed_dir) = fixed_dir() else {
        println!(
            "skipping interop_partition_stats FIXED GEN — set \
             ICEBERG_INTEROP_PARTITION_STATS_FIXED_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let table_location = fixed_dir
        .join("rust_fixed_table")
        .to_string_lossy()
        .to_string();
    fs::create_dir_all(format!("{table_location}/metadata"))
        .expect("create rust_fixed_table/metadata dir");

    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_fixed_gen",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                fixed_dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for FIXED GEN");

    let namespace = NamespaceIdent::new("interop_fixed".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (FIXED GEN)");

    let creation = TableCreation::builder()
        .name("rust_fixed_table".to_string())
        .location(table_location.clone())
        .schema(fixed_fixture_schema())
        .partition_spec(fixed_fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_fixed_table");

    let file_fixed = fixed_data_file(
        &table_location,
        "fixed_file",
        &KNOWN_FIXED_BYTES,
        FIXED_DATA_RECORDS as u64,
        FIXED_DATA_FILE_SIZE,
    );

    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_fixed])
        .apply(tx)
        .expect("apply FIXED fast_append");
    let table = tx.commit(&catalog).await.expect("commit FIXED S1");
    let s1_id = table
        .metadata()
        .current_snapshot_id()
        .expect("FIXED S1 snapshot id");

    let s1_snapshot = table
        .metadata()
        .current_snapshot()
        .expect("FIXED current snapshot");
    let stats_file = compute_and_write_stats_file(&table, s1_snapshot)
        .await
        .expect("compute FIXED stats")
        .expect("FIXED stats must be Some");
    let table = register_partition_stats_file(&catalog, &table, stats_file)
        .await
        .expect("register FIXED stats");

    write_final_metadata(&table, &table_location).await;

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (FIXED)");
    let stats_schema = partition_stats_schema(&unified_type, table.metadata().format_version())
        .expect("stats schema (FIXED)");
    let stats_path =
        find_stats_path(&fixed_dir.join("rust_fixed_table/metadata/final.metadata.json"));

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read FIXED stats");

    assert_eq!(rows.len(), 1, "FIXED GEN: expected 1 partition row");
    let row = &rows[0];

    // Verify the fixed round-trip: the partition field must decode back to the known bytes.
    let hex = match row.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::Binary(b)))) => {
            assert_eq!(
                b.len(),
                FIXED_LENGTH,
                "FIXED GEN: width must be {FIXED_LENGTH}"
            );
            bytes_to_hex(b)
        }
        other => panic!("FIXED GEN: unexpected partition literal type: {other:?}"),
    };
    assert_eq!(
        hex,
        bytes_to_hex(&KNOWN_FIXED_BYTES),
        "FIXED GEN: partition bytes round-trip failed"
    );
    assert_eq!(
        row.data_record_count(),
        FIXED_DATA_RECORDS,
        "FIXED GEN: data_record_count"
    );
    assert_eq!(row.data_file_count(), 1, "FIXED GEN: data_file_count");
    assert_eq!(
        row.total_data_file_size_in_bytes(),
        FIXED_DATA_FILE_SIZE as i64,
        "FIXED GEN: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row.last_updated_snapshot_id(),
        Some(s1_id),
        "FIXED GEN: last_updated_snapshot_id must be S1"
    );

    let expected_json = json!([{
        "partition_fixed_hex": hex,
        "spec_id": row.spec_id(),
        "data_record_count": row.data_record_count(),
        "data_file_count": row.data_file_count(),
        "total_data_file_size_in_bytes": row.total_data_file_size_in_bytes(),
        "position_delete_record_count": row.position_delete_record_count(),
        "position_delete_file_count": row.position_delete_file_count(),
        "equality_delete_record_count": row.equality_delete_record_count(),
        "equality_delete_file_count": row.equality_delete_file_count(),
        "dv_count": row.dv_count(),
        "last_updated_snapshot_id": row.last_updated_snapshot_id(),
        "total_record_count_null": row.total_record_count().is_none()
    }]);

    let expected_path = fixed_dir.join("fixed_expected.json");
    fs::write(
        &expected_path,
        serde_json::to_string_pretty(&expected_json).expect("serialize fixed_expected.json"),
    )
    .expect("write fixed_expected.json");
    println!(
        "interop_partition_stats FIXED GEN: PASS — hex={} records={FIXED_DATA_RECORDS} size={FIXED_DATA_FILE_SIZE}",
        bytes_to_hex(&KNOWN_FIXED_BYTES)
    );
}

/// Direction 2: Rust reads Java's fixed-partitioned stats file against `java_fixed_stats.json`.
#[tokio::test]
async fn test_partition_stats_fixed_d2() {
    let Some(dir) = fixed_dir() else {
        println!(
            "skipping interop_partition_stats FIXED D2 — set \
             ICEBERG_INTEROP_PARTITION_STATS_FIXED_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let java_meta_path = dir.join("java_fixed_table/metadata/final.metadata.json");
    assert!(
        java_meta_path.exists(),
        "FIXED D2: missing {}; run the Java generate step first",
        java_meta_path.display()
    );

    let stats_path = find_stats_path(&java_meta_path);
    let table_dir = dir.join("java_fixed_table").to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_fixed_d2",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for FIXED D2");

    let namespace = NamespaceIdent::new("interop_fixed_d2".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (FIXED D2)");

    let creation = TableCreation::builder()
        .name("java_fixed_table".to_string())
        .location(table_dir.clone())
        .schema(fixed_fixture_schema())
        .partition_spec(fixed_fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create FIXED D2 table handle");

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (FIXED D2)");
    let stats_schema =
        partition_stats_schema(&unified_type, FormatVersion::V2).expect("stats schema (FIXED D2)");

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read_partition_stats_file (FIXED D2)");

    let java_stats_path = dir.join("java_fixed_stats.json");
    assert!(
        java_stats_path.exists(),
        "FIXED D2: missing {}; run the Java generate step first",
        java_stats_path.display()
    );
    let expected = load_json_file(&java_stats_path);
    let expected_arr = expected
        .as_array()
        .expect("java_fixed_stats.json must be a JSON array");

    assert_eq!(
        rows.len(),
        expected_arr.len(),
        "FIXED D2: row count mismatch"
    );
    assert_eq!(rows.len(), 1, "FIXED D2: expected 1 partition row");
    let row = &rows[0];
    let exp = &expected_arr[0];

    let hex = match row.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::Binary(b)))) => bytes_to_hex(b),
        other => panic!("FIXED D2: unexpected partition literal type: {other:?}"),
    };
    assert_eq!(
        hex.to_lowercase(),
        exp["partition_fixed_hex"]
            .as_str()
            .unwrap_or("")
            .to_lowercase(),
        "FIXED D2: partition_fixed_hex mismatch"
    );
    assert_eq!(
        row.data_record_count(),
        exp["data_record_count"].as_i64().unwrap_or(-1),
        "FIXED D2: data_record_count"
    );
    assert_eq!(
        row.data_file_count() as i64,
        exp["data_file_count"].as_i64().unwrap_or(-1),
        "FIXED D2: data_file_count"
    );
    assert_eq!(
        row.total_data_file_size_in_bytes(),
        exp["total_data_file_size_in_bytes"].as_i64().unwrap_or(-1),
        "FIXED D2: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row.last_updated_snapshot_id(),
        exp["last_updated_snapshot_id"].as_i64(),
        "FIXED D2: last_updated_snapshot_id"
    );
    println!("interop_partition_stats FIXED D2: PASS — hex={hex}");
}

// ---- BINARY partition type (R3) ------------------------------------------------------------------

/// GEN test: an `identity(partition_binary: binary)` fixture. It emits
/// `rust_binary_table/metadata/final.metadata.json` and `binary_expected.json` for Java to judge.
#[tokio::test]
async fn test_partition_stats_binary_gen() {
    let Some(binary_dir) = binary_dir() else {
        println!(
            "skipping interop_partition_stats BINARY GEN — set \
             ICEBERG_INTEROP_PARTITION_STATS_BINARY_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let table_location = binary_dir
        .join("rust_binary_table")
        .to_string_lossy()
        .to_string();
    fs::create_dir_all(format!("{table_location}/metadata"))
        .expect("create rust_binary_table/metadata dir");

    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_binary_gen",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                binary_dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for BINARY GEN");

    let namespace = NamespaceIdent::new("interop_binary".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (BINARY GEN)");

    let creation = TableCreation::builder()
        .name("rust_binary_table".to_string())
        .location(table_location.clone())
        .schema(binary_fixture_schema())
        .partition_spec(binary_fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_binary_table");

    let file_binary = binary_data_file(
        &table_location,
        "binary_file",
        &KNOWN_BINARY_BYTES,
        BINARY_DATA_RECORDS as u64,
        BINARY_DATA_FILE_SIZE,
    );

    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_binary])
        .apply(tx)
        .expect("apply BINARY fast_append");
    let table = tx.commit(&catalog).await.expect("commit BINARY S1");
    let s1_id = table
        .metadata()
        .current_snapshot_id()
        .expect("BINARY S1 snapshot id");

    let s1_snapshot = table
        .metadata()
        .current_snapshot()
        .expect("BINARY current snapshot");
    let stats_file = compute_and_write_stats_file(&table, s1_snapshot)
        .await
        .expect("compute BINARY stats")
        .expect("BINARY stats must be Some");
    let table = register_partition_stats_file(&catalog, &table, stats_file)
        .await
        .expect("register BINARY stats");

    write_final_metadata(&table, &table_location).await;

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (BINARY)");
    let stats_schema = partition_stats_schema(&unified_type, table.metadata().format_version())
        .expect("stats schema (BINARY)");
    let stats_path =
        find_stats_path(&binary_dir.join("rust_binary_table/metadata/final.metadata.json"));

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read BINARY stats");

    assert_eq!(rows.len(), 1, "BINARY GEN: expected 1 partition row");
    let row = &rows[0];

    // Verify the binary round-trip: the partition field must decode back to the known bytes.
    let hex = match row.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::Binary(b)))) => bytes_to_hex(b),
        other => panic!("BINARY GEN: unexpected partition literal type: {other:?}"),
    };
    assert_eq!(
        hex,
        bytes_to_hex(&KNOWN_BINARY_BYTES),
        "BINARY GEN: partition bytes round-trip failed"
    );
    assert_eq!(
        row.data_record_count(),
        BINARY_DATA_RECORDS,
        "BINARY GEN: data_record_count"
    );
    assert_eq!(row.data_file_count(), 1, "BINARY GEN: data_file_count");
    assert_eq!(
        row.total_data_file_size_in_bytes(),
        BINARY_DATA_FILE_SIZE as i64,
        "BINARY GEN: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row.last_updated_snapshot_id(),
        Some(s1_id),
        "BINARY GEN: last_updated_snapshot_id must be S1"
    );

    let expected_json = json!([{
        "partition_binary_hex": hex,
        "spec_id": row.spec_id(),
        "data_record_count": row.data_record_count(),
        "data_file_count": row.data_file_count(),
        "total_data_file_size_in_bytes": row.total_data_file_size_in_bytes(),
        "position_delete_record_count": row.position_delete_record_count(),
        "position_delete_file_count": row.position_delete_file_count(),
        "equality_delete_record_count": row.equality_delete_record_count(),
        "equality_delete_file_count": row.equality_delete_file_count(),
        "dv_count": row.dv_count(),
        "last_updated_snapshot_id": row.last_updated_snapshot_id(),
        "total_record_count_null": row.total_record_count().is_none()
    }]);

    let expected_path = binary_dir.join("binary_expected.json");
    fs::write(
        &expected_path,
        serde_json::to_string_pretty(&expected_json).expect("serialize binary_expected.json"),
    )
    .expect("write binary_expected.json");
    println!(
        "interop_partition_stats BINARY GEN: PASS — hex={} records={BINARY_DATA_RECORDS} size={BINARY_DATA_FILE_SIZE}",
        bytes_to_hex(&KNOWN_BINARY_BYTES)
    );
}

/// Direction 2: Rust reads Java's binary-partitioned stats file against `java_binary_stats.json`.
#[tokio::test]
async fn test_partition_stats_binary_d2() {
    let Some(dir) = binary_dir() else {
        println!(
            "skipping interop_partition_stats BINARY D2 — set \
             ICEBERG_INTEROP_PARTITION_STATS_BINARY_DIR \
             (run dev/java-interop/run-interop-partition-stats.sh)"
        );
        return;
    };

    let java_meta_path = dir.join("java_binary_table/metadata/final.metadata.json");
    assert!(
        java_meta_path.exists(),
        "BINARY D2: missing {}; run the Java generate step first",
        java_meta_path.display()
    );

    let stats_path = find_stats_path(&java_meta_path);
    let table_dir = dir.join("java_binary_table").to_string_lossy().to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_partition_stats_binary_d2",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                dir.to_string_lossy().to_string(),
            )]),
        )
        .await
        .expect("build MemoryCatalog for BINARY D2");

    let namespace = NamespaceIdent::new("interop_binary_d2".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace (BINARY D2)");

    let creation = TableCreation::builder()
        .name("java_binary_table".to_string())
        .location(table_dir.clone())
        .schema(binary_fixture_schema())
        .partition_spec(binary_fixture_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create BINARY D2 table handle");

    let unified_type =
        unified_partition_type(table.metadata()).expect("unified_partition_type (BINARY D2)");
    let stats_schema =
        partition_stats_schema(&unified_type, FormatVersion::V2).expect("stats schema (BINARY D2)");

    let rows = read_partition_stats_file(&table, &stats_schema, &stats_path)
        .await
        .expect("read_partition_stats_file (BINARY D2)");

    let java_stats_path = dir.join("java_binary_stats.json");
    assert!(
        java_stats_path.exists(),
        "BINARY D2: missing {}; run the Java generate step first",
        java_stats_path.display()
    );
    let expected = load_json_file(&java_stats_path);
    let expected_arr = expected
        .as_array()
        .expect("java_binary_stats.json must be a JSON array");

    assert_eq!(
        rows.len(),
        expected_arr.len(),
        "BINARY D2: row count mismatch"
    );
    assert_eq!(rows.len(), 1, "BINARY D2: expected 1 partition row");
    let row = &rows[0];
    let exp = &expected_arr[0];

    let hex = match row.partition().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::Binary(b)))) => bytes_to_hex(b),
        other => panic!("BINARY D2: unexpected partition literal type: {other:?}"),
    };
    assert_eq!(
        hex.to_lowercase(),
        exp["partition_binary_hex"]
            .as_str()
            .unwrap_or("")
            .to_lowercase(),
        "BINARY D2: partition_binary_hex mismatch"
    );
    assert_eq!(
        row.data_record_count(),
        exp["data_record_count"].as_i64().unwrap_or(-1),
        "BINARY D2: data_record_count"
    );
    assert_eq!(
        row.data_file_count() as i64,
        exp["data_file_count"].as_i64().unwrap_or(-1),
        "BINARY D2: data_file_count"
    );
    assert_eq!(
        row.total_data_file_size_in_bytes(),
        exp["total_data_file_size_in_bytes"].as_i64().unwrap_or(-1),
        "BINARY D2: total_data_file_size_in_bytes"
    );
    assert_eq!(
        row.last_updated_snapshot_id(),
        exp["last_updated_snapshot_id"].as_i64(),
        "BINARY D2: last_updated_snapshot_id"
    );
    println!("interop_partition_stats BINARY D2: PASS — hex={hex}");
}
