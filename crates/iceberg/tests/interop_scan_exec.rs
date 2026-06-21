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

//! Java interop test for DATA-LEVEL scan execution with MERGE-ON-READ position deletes (the capstone).
//!
//! Every other interop suite in this crate reads METADATA — committed JSON ([`interop_inspection`]) or
//! on-disk AVRO manifests ([`interop_inspection_manifests`]). THIS test reads DATA: it proves Rust's
//! `table.scan().build()?.to_arrow()` — which opens the real parquet AND APPLIES position deletes
//! (merge-on-read) — produces the SAME live rows Java's own read produces.
//!
//! THE FIXTURE. The Java oracle's `generate-interop-scan-exec` mode WRITES A REAL TABLE to a temp dir:
//! an unpartitioned V2 table (schema {1 id long required, 2 data string optional}) with two real files:
//!   * a REAL parquet DATA file (`00000-data.parquet`) of 5 rows — (10,"a") (20,"b") (30,"c") (40,"d")
//!     (50,"e") at positions 0..4 — written via iceberg-data's generic parquet appender;
//!   * a REAL parquet POSITION-DELETE file (`00000-data-deletes.parquet`) deleting positions {1, 3} of
//!     that data file (rows 20 and 40) — written via the generic position-delete writer.
//!
//! They are committed via `newAppend(dataFile)` then `newRowDelta(deleteFile)` (real AVRO manifests +
//! manifest-list on disk), with `final.metadata.json` written to a known path. The LIVE rows after
//! merge-on-read are {10, 30, 50}. Java materializes its OWN merge-on-read read
//! (`IcebergGenerics.read(table).build()`, which applies the position deletes), sorts by id, and emits
//! `java_scan_rows.json` = `[{10,a},{30,c},{50,e}]`. That is the GROUND TRUTH.
//!
//! THIS test loads the SAME `final.metadata.json`, builds a `Table` over a local-filesystem `FileIO` (which
//! resolves the absolute manifest + parquet paths the commits wrote), runs `scan().build()?.to_arrow()`,
//! collects the Arrow `RecordBatch`es, extracts the (id, data) rows, sorts by id, and asserts they EQUAL
//! Java's read. **This is the merge-on-read proof:** the deleted rows (20, 40) must be ABSENT; the live set
//! is exactly {10, 30, 50}.
//!
//! THE ENV GATE. Because the table is regenerated each run (nothing binary is committed), this test is
//! GATED on `ICEBERG_INTEROP_SCAN_DIR`. When the var is UNSET the test is a clean NO-OP (a runtime
//! early-return, NOT `#[ignore]`) so the offline `cargo test` gate stays green with no Java/Maven. The
//! `dev/java-interop/run-interop-scan-exec.sh` script sets the var and runs the REAL comparison.
//!
//! NO PRODUCTION CHANGE is needed: Rust's `to_arrow` already applies position deletes (the row_delta scan
//! tests in `scan/mod.rs` prove it). This test is the byte-level, Java-written-table proof of that path.
//!
//! DIRECTION 2 (the GEN path — "Java reads what WE write"). When `ICEBERG_INTEROP_SCAN_GEN_DIR` is SET,
//! [`test_scan_exec_gen_rust_writes_java_readable_table`] WRITES a real on-disk table there using the
//! PRODUCTION write path (mirroring the `row_delta.rs` crown jewel), and the Java oracle's
//! `verify-interop-scan-exec` mode READS it back with `IcebergGenerics` and asserts the merge-on-read rows.
//! This is the parity flip for the write actions (append / row_delta): we write REAL parquet data + a REAL
//! position-delete via `PositionDeleteFileWriter`, commit through a `MemoryCatalog` over the local FS, and
//! land a `final.metadata.json` at a known path for Java to load. When the GEN var is UNSET this is a clean
//! NO-OP. The two env vars are independent: Direction-1 (`ICEBERG_INTEROP_SCAN_DIR`) is unchanged.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::arrow::DeleteFilter;
use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, FormatVersion, Literal, NestedField, PartitionKey, PartitionSpec,
    PrimitiveType, Schema, SchemaRef, SortOrder, Struct, TableMetadata, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;

// ===========================================================================================
// The Java oracle row model — deserialized from java_scan_rows.json: a JSON array of {id, data}.
// ===========================================================================================

/// One live row of Java's merge-on-read read (`IcebergGenerics`): the `id` (long) + nullable `data` string.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ScanRow {
    id: i64,
    data: Option<String>,
}

/// Sort rows by id for an order-independent comparison (both Java and Rust sort the same way).
fn sorted_by_id(mut rows: Vec<ScanRow>) -> Vec<ScanRow> {
    rows.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| cmp_opt(&a.data, &b.data)));
    rows
}

fn cmp_opt(a: &Option<String>, b: &Option<String>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(x), Some(y)) => x.cmp(y),
    }
}

// ===========================================================================================
// Fixture loading + Table construction.
// ===========================================================================================

/// The temp dir the Java oracle wrote the table + JSON rows into. `None` when the env var is unset.
fn scan_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_SCAN_DIR").map(PathBuf::from)
}

/// The temp dir into which the DIRECTION-2 GEN path writes a Rust-authored table for Java to read.
/// `None` when `ICEBERG_INTEROP_SCAN_GEN_DIR` is unset (the GEN test is then a clean no-op).
fn scan_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_SCAN_GEN_DIR").map(PathBuf::from)
}

/// The temp dir the Java oracle wrote the EQUALITY-delete table + JSON rows into (Direction 1, eq-delete).
/// `None` when `ICEBERG_INTEROP_EQ_SCAN_DIR` is unset (the eq-delete read test is then a clean no-op).
fn eq_scan_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_EQ_SCAN_DIR").map(PathBuf::from)
}

/// The temp dir into which the DIRECTION-2 eq-delete GEN path writes a Rust-authored equality-delete table
/// for Java to read. `None` when `ICEBERG_INTEROP_EQ_SCAN_GEN_DIR` is unset (then a clean no-op).
fn eq_scan_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_EQ_SCAN_GEN_DIR").map(PathBuf::from)
}

/// The temp dir the Java oracle wrote the PARTITIONED table + JSON rows into (Direction 1, partitioned).
/// `None` when `ICEBERG_INTEROP_PART_SCAN_DIR` is unset (the partitioned read test is then a clean no-op).
fn part_scan_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PART_SCAN_DIR").map(PathBuf::from)
}

/// The temp dir into which the DIRECTION-2 partitioned GEN path writes a Rust-authored partitioned table
/// (identity(category) + a partition-scoped position-delete) for Java to read. `None` when
/// `ICEBERG_INTEROP_PART_SCAN_GEN_DIR` is unset (then a clean no-op).
fn part_scan_gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_PART_SCAN_GEN_DIR").map(PathBuf::from)
}

/// Load + parse the Java ground-truth PARTITIONED rows from `<dir>/java_part_scan_rows.json`.
fn read_java_part_rows(dir: &std::path::Path) -> Vec<ScanRow> {
    let path = dir.join("java_part_scan_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<ScanRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Load + parse the Java ground-truth EQUALITY-delete rows from `<dir>/java_eq_scan_rows.json`.
fn read_java_eq_rows(dir: &std::path::Path) -> Vec<ScanRow> {
    let path = dir.join("java_eq_scan_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<ScanRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Load + parse the Java ground-truth rows from `<dir>/java_scan_rows.json`.
fn read_java_rows(dir: &std::path::Path) -> Vec<ScanRow> {
    let path = dir.join("java_scan_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<ScanRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Build a `Table` over the Java-written `final.metadata.json`, using a LOCAL-FILESYSTEM `FileIO` so the
/// absolute on-disk manifest-list + manifest + parquet paths the commits wrote resolve directly.
fn load_table(dir: &std::path::Path) -> Table {
    let metadata_path = dir.join("table/metadata/final.metadata.json");
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", "scan_exec"]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from Java-written final.metadata.json")
}

// ===========================================================================================
// Arrow column extraction — a scan batch into the comparable [`ScanRow`]s (by COLUMN NAME).
// ===========================================================================================

/// Extract the `id` (Int64) + `data` (Utf8 string, nullable) columns from one scan batch. The `data`
/// column reads via either i32- or i64-offset Utf8 to be robust to the offset width `to_arrow` emits.
fn extract_rows(batch: &RecordBatch) -> Vec<ScanRow> {
    let id = batch
        .column_by_name("id")
        .expect("id column present")
        .as_primitive::<Int64Type>();
    let data = batch.column_by_name("data").expect("data column present");

    (0..batch.num_rows())
        .map(|i| ScanRow {
            id: id.value(i),
            data: string_value(data, i),
        })
        .collect()
}

/// Read row `i` of a nullable string column as `Option<String>`, tolerating Utf8 (i32) / LargeUtf8 (i64).
fn string_value(array: &arrow_array::ArrayRef, i: usize) -> Option<String> {
    use arrow_schema::DataType;
    if array.is_null(i) {
        return None;
    }
    match array.data_type() {
        DataType::Utf8 => Some(array.as_string::<i32>().value(i).to_string()),
        DataType::LargeUtf8 => Some(array.as_string::<i64>().value(i).to_string()),
        other => panic!("unexpected data column arrow type: {other:?}"),
    }
}

// ===========================================================================================
// The single env-gated interop test.
// ===========================================================================================

#[tokio::test]
async fn test_scan_exec_merge_on_read_matches_java_read() {
    let Some(dir) = scan_dir() else {
        println!(
            "skipping interop_scan_exec — set ICEBERG_INTEROP_SCAN_DIR \
             (run dev/java-interop/run-interop-scan-exec.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // Rust's scan → Arrow applies the position deletes (merge-on-read): the row_delta scan tests in
    // scan/mod.rs prove the path; here we prove it byte-for-byte against a Java-written table.
    let batch_stream = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow");
    let batches: Vec<RecordBatch> = batch_stream
        .try_collect()
        .await
        .expect("collect scan batches");

    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_rows(&dir));

    // -- The merge-on-read proof. ----------------------------------------------------------------------

    // Exactly 3 live rows survive (5 written - 2 deleted).
    assert_eq!(
        rust_rows.len(),
        3,
        "exactly 3 rows survive merge-on-read (5 written, positions 1 and 3 deleted)"
    );

    // The deleted rows (id 20 at position 1, id 40 at position 3) must be ABSENT.
    assert!(
        !rust_rows.iter().any(|r| r.id == 20),
        "id 20 (deleted at position 1) must be ABSENT after merge-on-read"
    );
    assert!(
        !rust_rows.iter().any(|r| r.id == 40),
        "id 40 (deleted at position 3) must be ABSENT after merge-on-read"
    );

    // The surviving (id, data) values match Java's read exactly: {(10,a),(30,c),(50,e)}.
    assert_eq!(
        rust_rows, java_rows,
        "Rust scan→Arrow (merge-on-read) rows must equal Java's IcebergGenerics read field-for-field"
    );

    // Pin the exact live set so it cannot drift unnoticed.
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "the live id set after merge-on-read is exactly {{10, 30, 50}}"
    );
    let live_data: Vec<Option<&str>> = rust_rows.iter().map(|r| r.data.as_deref()).collect();
    assert_eq!(
        live_data,
        vec![Some("a"), Some("c"), Some("e")],
        "the live data column matches the committed values for ids 10/30/50"
    );

    println!(
        "interop_scan_exec OK — Rust scan→Arrow merge-on-read = Java read: 3 live rows {{10,30,50}}, \
         deleted 20/40 absent"
    );
}

// ===========================================================================================
// EQUALITY-DELETE, DIRECTION 1 — Java writes the equality delete, Rust reads it.
//
// The sibling of the position-delete read test above, but the merge-on-read mechanism is delete-by-VALUE.
// The Java oracle's `generate-interop-eq-delete` mode wrote an unpartitioned V2 table with a REAL parquet
// data file (5 rows, appended at sequence 1) + a REAL parquet EQUALITY-delete file (equality_ids = [1] =
// the `id` field, deleting rows id=20 and id=40, committed at sequence 2). Because the data (seq 1) precedes
// the delete (seq 2), the equality delete applies (1 < 2) and the live set is {10,30,50}. Java emitted its
// OWN read into `java_eq_scan_rows.json`. This test loads the same table, runs `scan().to_arrow()` — which
// applies the equality delete by VALUE — and asserts the rows equal Java's read (ids 20/40 absent).
//
// Gated on `ICEBERG_INTEROP_EQ_SCAN_DIR`: a clean no-op when unset, so the offline gate stays green. If Rust
// did NOT apply the equality delete this assertion would FAIL (a real read gap) — but Rust's delete_filter +
// delete_file_index already support equality deletes, so it applies.
// ===========================================================================================

#[tokio::test]
async fn test_scan_exec_equality_delete_matches_java_read() {
    let Some(dir) = eq_scan_dir() else {
        println!(
            "skipping interop_scan_exec equality-delete — set ICEBERG_INTEROP_EQ_SCAN_DIR \
             (run dev/java-interop/run-interop-eq-delete.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // Rust's scan → Arrow applies the EQUALITY delete (merge-on-read, by VALUE): rows whose `id` equals
    // a delete value (20 or 40) are dropped from the seq-1 data file by the seq-2 equality delete.
    let batch_stream = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow");
    let batches: Vec<RecordBatch> = batch_stream
        .try_collect()
        .await
        .expect("collect scan batches");

    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_eq_rows(&dir));

    // -- The equality-delete merge-on-read proof. ----------------------------------------------------------

    // Exactly 3 live rows survive (5 written - 2 deleted by VALUE).
    assert_eq!(
        rust_rows.len(),
        3,
        "exactly 3 rows survive merge-on-read (5 written, ids 20 and 40 deleted by VALUE)"
    );

    // The deleted rows (id 20, id 40) must be ABSENT — the equality delete keyed on field id 1 dropped them.
    assert!(
        !rust_rows.iter().any(|r| r.id == 20),
        "id 20 (equality-deleted by value) must be ABSENT after merge-on-read"
    );
    assert!(
        !rust_rows.iter().any(|r| r.id == 40),
        "id 40 (equality-deleted by value) must be ABSENT after merge-on-read"
    );

    // The surviving (id, data) values match Java's read exactly: {(10,a),(30,c),(50,e)}.
    assert_eq!(
        rust_rows, java_rows,
        "Rust scan→Arrow (equality merge-on-read) rows must equal Java's IcebergGenerics read field-for-field"
    );

    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "the live id set after equality merge-on-read is exactly {{10, 30, 50}}"
    );
    let live_data: Vec<Option<&str>> = rust_rows.iter().map(|r| r.data.as_deref()).collect();
    assert_eq!(
        live_data,
        vec![Some("a"), Some("c"), Some("e")],
        "the live data column matches the committed values for ids 10/30/50"
    );

    println!(
        "interop_scan_exec equality-delete OK — Rust scan→Arrow equality merge-on-read = Java read: \
         3 live rows {{10,30,50}}, deleted 20/40 absent"
    );
}

// ===========================================================================================
// PARTITIONED merge-on-read, DIRECTION 1 — Java writes the PARTITIONED table + partition-scoped delete,
// Rust reads it. The partition-handling proof.
//
// The sibling of the position-delete read test above, but the table is PARTITIONED by identity(category)
// and the position-delete is PARTITION-SCOPED. The Java oracle's `generate-interop-part-scan` mode wrote a
// V2 table {1 id long required, 2 category string required, 3 data string optional} partitioned by
// identity(category) with one REAL parquet data file PER PARTITION (category=a: (10,a,x),(20,a,y),(30,a,z)
// at positions 0..2; category=b: (40,b,p),(50,b,q) at positions 0..1), each DataFile stamped with its
// partition value (spec id 0). It then wrote a PARTITION-SCOPED position-delete in partition a deleting
// position 1 (id=20), committed via newRowDelta at sequence 2 (the data appended FIRST at sequence 1). The
// live merge-on-read set is {10,30,40,50} (only id=20 deleted; both partitions otherwise intact). Java
// emitted its OWN read into `java_part_scan_rows.json`. This test loads the same table, runs
// `scan().to_arrow()` — which must apply the partition-scoped position delete — and asserts the rows equal
// Java's read (id=20 absent; cat=a survivors 10/30 AND cat=b's 40/50 all present).
//
// Gated on `ICEBERG_INTEROP_PART_SCAN_DIR`: a clean no-op when unset, so the offline gate stays green. If
// Rust MISHANDLED partition-scoped merge-on-read (e.g. applied the cat=a delete to cat=b, or dropped a
// partition) this assertion would FAIL — a real partition-aware read gap. Rust's delete_file_index keys
// deletes by partition + spec id, so the cat=a delete reaches only the cat=a data file.
// ===========================================================================================

#[tokio::test]
async fn test_part_scan_exec_partition_scoped_merge_on_read_matches_java_read() {
    let Some(dir) = part_scan_dir() else {
        println!(
            "skipping interop_scan_exec partitioned — set ICEBERG_INTEROP_PART_SCAN_DIR \
             (run dev/java-interop/run-interop-part.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // Rust's scan → Arrow applies the PARTITION-SCOPED position delete (merge-on-read): position 1 of the
    // category=a data file (id=20) is dropped; category=b is untouched.
    let batch_stream = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow");
    let batches: Vec<RecordBatch> = batch_stream
        .try_collect()
        .await
        .expect("collect scan batches");

    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let java_rows = sorted_by_id(read_java_part_rows(&dir));

    // -- The partition-aware merge-on-read proof. ---------------------------------------------------------

    // Exactly 4 live rows survive (5 written across both partitions, position 1 of cat=a deleted).
    assert_eq!(
        rust_rows.len(),
        4,
        "exactly 4 rows survive partition-aware merge-on-read (5 written, cat=a position 1 deleted)"
    );

    // The deleted row (id 20 at position 1 of the cat=a data file) must be ABSENT.
    assert!(
        !rust_rows.iter().any(|r| r.id == 20),
        "id 20 (partition-scoped delete at cat=a position 1) must be ABSENT after merge-on-read"
    );

    // Both partitions are otherwise intact: cat=a survivors 10/30 AND cat=b's 40/50 must all be present.
    for id in [10_i64, 30, 40, 50] {
        assert!(
            rust_rows.iter().any(|r| r.id == id),
            "id {id} must be present — both partitions intact except cat=a's deleted id=20"
        );
    }

    // The surviving (id, data) values match Java's read exactly: {(10,x),(30,z),(40,p),(50,q)}.
    assert_eq!(
        rust_rows, java_rows,
        "Rust scan→Arrow (partition-aware merge-on-read) rows must equal Java's IcebergGenerics read \
         field-for-field"
    );

    // Pin the exact live set so it cannot drift unnoticed.
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 50],
        "the live id set after partition-aware merge-on-read is exactly {{10, 30, 40, 50}}"
    );
    let live_data: Vec<Option<&str>> = rust_rows.iter().map(|r| r.data.as_deref()).collect();
    assert_eq!(
        live_data,
        vec![Some("x"), Some("z"), Some("p"), Some("q")],
        "the live data column matches the committed values for ids 10/30 (cat=a) and 40/50 (cat=b)"
    );

    println!(
        "interop_scan_exec partitioned OK — Rust scan→Arrow partition-aware merge-on-read = Java read: \
         4 live rows {{10,30,40,50}}, cat=a's id=20 deleted, cat=b intact"
    );
}

// ===========================================================================================
// DIRECTION 2 — the GEN path: Rust WRITES a real on-disk table; Java reads it back.
//
// Mirrors the `row_delta.rs` crown jewel exactly, but commits through a `MemoryCatalog` backed by
// `LocalFsStorageFactory` (so metadata + manifests + parquet land on the REAL local FS) and writes a
// `final.metadata.json` at a known path. The Java oracle's `verify-interop-scan-exec` mode loads that
// metadata, reads with `IcebergGenerics` (applying our position delete), and asserts {10,30,50}.
//
// When `ICEBERG_INTEROP_SCAN_GEN_DIR` is UNSET this is a clean no-op — the offline gate stays green.
// ===========================================================================================

/// The unpartitioned V2 schema Java expects: {1 id long required, 2 data string optional}.
fn gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, data string} schema")
}

/// Create the unpartitioned V2 table at EXACTLY `<gen_dir>/rust_table` in a `MemoryCatalog` over the
/// local FS, so the on-disk layout is the deterministic `rust_table/{metadata,data}/...` Java loads.
async fn create_rust_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(gen_schema())
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table")
}

/// Write a REAL parquet DATA file of 5 rows (10,"a")…(50,"e") at positions 0..4 into the table's
/// location via the production `ParquetWriterBuilder` + `FileWriter`, returning the [`DataFile`]
/// (content `Data`, unpartitioned). Reuses the crown-jewel machinery — no hand-rolled parquet.
async fn write_gen_data_file(table: &Table) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));

    let ids = Int64Array::from(vec![10_i64, 20, 30, 40, 50]);
    let data = StringArray::from(vec!["a", "b", "c", "d", "e"]);
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(ids) as ArrayRef,
        Arc::new(data) as ArrayRef,
    ])
    .expect("build the 5-row data batch");

    // Write the parquet directly under the table location so Java's FileIO resolves it from the
    // manifest entry (same convention as the crown jewel's `write_data_file`).
    let file_path = format!(
        "{}/data/00000-rust-data.parquet",
        table.metadata().location()
    );
    let output = table
        .file_io()
        .new_output(file_path)
        .expect("new parquet output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder
        .build(output)
        .await
        .expect("build parquet writer");
    writer.write(&batch).await.expect("write data batch");
    let data_file_builders = writer.close().await.expect("close parquet writer");

    // The parquet writer returns builders without content/partition stamped — finish as an
    // unpartitioned data file (empty partition struct, default spec id 0).
    let mut builder = data_file_builders
        .into_iter()
        .next()
        .expect("one data file builder");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build unpartitioned data file")
}

#[tokio::test]
async fn test_scan_exec_gen_rust_writes_java_readable_table() {
    let Some(gen_dir) = scan_gen_dir() else {
        println!(
            "skipping interop_scan_exec GEN — set ICEBERG_INTEROP_SCAN_GEN_DIR \
             (run dev/java-interop/run-interop-scan-exec-d2.sh)"
        );
        return;
    };

    // 1. A MemoryCatalog over the LOCAL FS, warehouse = <gen_dir>, table pinned to <gen_dir>/rust_table.
    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_rust_table(&catalog, &table_location).await;

    // 2. fast_append a REAL parquet data file of 5 rows (10,a)..(50,e).
    let data_file = write_gen_data_file(&table).await;
    let data_file_path = data_file.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // 3. ENGINE STEP — scan (_file, _pos) to DISCOVER the (file, pos) of ids 20/40 (so this
    //    Java-verified scenario exercises the `_pos` row-identity column round-tripping into the
    //    position-delete Java reads back), then row_delta a REAL position-delete built from the pairs.
    let mut pairs = discover_row_identities(&table, &[20, 40]).await;
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            (data_file_path.clone(), 1_i64),
            (data_file_path.clone(), 3_i64),
        ],
        "scan(_file,_pos) discovered ids 20/40 at their true positions 1/3 in the committed data file"
    );
    let delete_file = write_pos_delete_from_pairs(&table, &pairs).await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // 4. Sanity: OUR OWN scan→Arrow already applies the delete → {10,30,50}. (Direction-1 proves Rust
    //    reads what Java writes; here we confirm the table is internally consistent before handing to Java.)
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
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "Rust's own scan of the written table must already be {{10,30,50}} (20/40 deleted)"
    );

    // 5. Write the FINAL metadata to a KNOWN path so Java loads it deterministically. The real on-disk
    //    manifest-list + manifests + parquet already live under <gen_dir>/rust_table.
    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_scan_exec GEN OK — Rust wrote {table_location} (parquet data + position-delete + \
         final.metadata.json); Rust scan = {{10,30,50}}. Java verify-interop-scan-exec reads it next."
    );
}

// ===========================================================================================
// ENGINE-BOUNDARY OFFLINE PROOF — play the downstream engine's role using ONLY the public core-crate
// surface: scan `(_file, _pos)` to discover row identity, write a position-delete file from the
// discovered pairs, commit it via `RowDelta`, and confirm merge-on-read omits exactly those rows.
// Unlike the GEN test above this needs NO Java/Maven — it runs in the offline `cargo test` gate, and
// is the executable contract a DataFusion-wrapped engine builds its DELETE/UPDATE/MERGE on.
// ===========================================================================================

/// Decode the reserved `_file` column at row `i`. The scan emits `_file` as a per-file constant, which
/// the transformer materializes as a Run-End-Encoded `Utf8` column; tolerate both the REE and plain
/// `Utf8` forms.
fn decode_file_path(col: &ArrayRef, i: usize) -> String {
    use arrow_array::RunArray;
    use arrow_array::types::Int32Type;

    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        return plain.value(i).to_string();
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let physical = run.get_physical_index(i);
        return run
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("_file REE values are Utf8")
            .value(physical)
            .to_string();
    }
    panic!("unexpected _file column type: {:?}", col.data_type());
}

/// Scan the committed table selecting `[id, _file, _pos]` and return the `(data_file_path, position)`
/// of every row whose id is in `target_ids` — i.e. discover row identity exactly the way a downstream
/// engine would, to write position deletes. Exercises the `_file` and `_pos` reserved metadata columns.
async fn discover_row_identities(table: &Table, target_ids: &[i64]) -> Vec<(String, i64)> {
    use iceberg::metadata_columns::{RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS};

    let batches: Vec<RecordBatch> = table
        .scan()
        .select(["id", RESERVED_COL_NAME_FILE, RESERVED_COL_NAME_POS])
        .build()
        .expect("build identity scan")
        .to_arrow()
        .await
        .expect("identity scan to_arrow")
        .try_collect()
        .await
        .expect("collect identity batches");

    let mut pairs = Vec::new();
    for batch in &batches {
        let id_col = batch
            .column_by_name("id")
            .expect("id column")
            .as_primitive::<Int64Type>();
        let file_col = batch
            .column_by_name(RESERVED_COL_NAME_FILE)
            .expect("_file column");
        let pos_col = batch
            .column_by_name(RESERVED_COL_NAME_POS)
            .expect("_pos column")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            if target_ids.contains(&id_col.value(i)) {
                pairs.push((decode_file_path(file_col, i), pos_col.value(i)));
            }
        }
    }
    pairs
}

/// Write a REAL parquet position-delete file from discovered `(data_file_path, position)` pairs (the
/// pairs a scan discovered) via the production `PositionDeleteFileWriter`. The pairs MUST already be
/// sorted by `(file_path, pos)` per the spec.
async fn write_pos_delete_from_pairs(table: &Table, pairs: &[(String, i64)]) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(None)
        .await
        .expect("build position-delete writer");

    let paths: Vec<&str> = pairs.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("build position-delete batch");
    writer
        .write(batch)
        .await
        .expect("write position-delete batch");
    writer
        .close()
        .await
        .expect("close position-delete writer")
        .into_iter()
        .next()
        .expect("one position-delete file")
}

#[tokio::test]
async fn test_engine_boundary_scan_pos_then_row_delta() {
    use tempfile::TempDir;

    // 1. A MemoryCatalog over the local FS into a throwaway temp dir — fully offline, no Java.
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "engine_boundary",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_rust_table(&catalog, &table_location).await;

    // 2. fast_append a real 5-row parquet data file: (10,a)..(50,e) at positions 0..4.
    let data_file = write_gen_data_file(&table).await;
    let data_file_path = data_file.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // 3. ENGINE STEP — scan `(_file, _pos)` to discover the row identity of the rows to delete
    //    (ids 20 and 40). This is exactly what a downstream engine does to build position deletes,
    //    and it asserts `_pos` reports the TRUE physical ordinal (20 and 40 sit at positions 1 and 3).
    let mut pairs = discover_row_identities(&table, &[20, 40]).await;
    pairs.sort(); // spec: position deletes sorted by (file_path, pos)
    assert_eq!(
        pairs,
        vec![
            (data_file_path.clone(), 1_i64),
            (data_file_path.clone(), 3_i64),
        ],
        "scan(_file,_pos) must report ids 20/40 at their TRUE physical positions 1 and 3"
    );

    // 4. Write a position-delete file from the DISCOVERED pairs and commit it via RowDelta.
    let delete_file = write_pos_delete_from_pairs(&table, &pairs).await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // 5. Re-scan: merge-on-read must omit exactly the targeted rows → {10, 30, 50}.
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
        rows.extend(extract_rows(batch));
    }
    let live_ids: Vec<i64> = sorted_by_id(rows).iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "after committing the scan-discovered position deletes, ids 20/40 must be gone"
    );
}

// ===========================================================================================
// EQUALITY-DELETE, DIRECTION 2 — the GEN path: Rust WRITES a real on-disk table with an EQUALITY delete;
// Java reads it back.
//
// The sibling of the position-delete GEN path above, but the delete is an EQUALITY delete (delete-by-VALUE,
// keyed on field id 1 = `id`, deleting rows id=20 and id=40) written via the production
// `EqualityDeleteFileWriter`. The SEQUENCE ORDERING is the correctness point: the data is `fast_append`ed
// FIRST (data-sequence-number 1), the equality delete `row_delta`ed SECOND (sequence-number 2), so the
// delete (seq 2) applies to the data (seq 1) — 1 < 2. The table lands at `<gen_dir>/rust_table` with a
// `final.metadata.json` at a known path; the Java oracle's `verify-interop-eq-delete` mode reads it with
// `IcebergGenerics` (applying our equality delete) and asserts {10,30,50}.
//
// When `ICEBERG_INTEROP_EQ_SCAN_GEN_DIR` is UNSET this is a clean no-op — the offline gate stays green.
// ===========================================================================================

/// Write a REAL parquet EQUALITY-delete file (via the production `EqualityDeleteFileWriter`) keyed on
/// field id 1 (the `id` column), deleting rows id=20 and id=40, unpartitioned, stamped with content
/// `EqualityDeletes` + `equality_ids = [1]`. The Java-readable fixture's specific case of the general
/// [`write_equality_delete_for_ids`] (the `data` column is projected away, so only `id` lands on disk).
async fn write_gen_equality_delete_file(table: &Table) -> DataFile {
    write_equality_delete_for_ids(table, &[20, 40]).await
}

#[tokio::test]
async fn test_scan_exec_gen_rust_writes_java_readable_equality_delete_table() {
    let Some(gen_dir) = eq_scan_gen_dir() else {
        println!(
            "skipping interop_scan_exec equality-delete GEN — set ICEBERG_INTEROP_EQ_SCAN_GEN_DIR \
             (run dev/java-interop/run-interop-eq-delete-d2.sh)"
        );
        return;
    };

    // 1. A MemoryCatalog over the LOCAL FS, warehouse = <gen_dir>, table pinned to <gen_dir>/rust_table.
    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_eq_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_rust_table(&catalog, &table_location).await;

    // 2. fast_append a REAL parquet data file of 5 rows (10,a)..(50,e) at SEQUENCE 1.
    let data_file = write_gen_data_file(&table).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // 3. row_delta a REAL EQUALITY-delete (equality_ids = [1], ids 20/40) at SEQUENCE 2. Because the data
    //    (seq 1) was committed FIRST, the equality delete (seq 2) applies to it (1 < 2).
    let delete_file = write_gen_equality_delete_file(&table).await;
    assert_eq!(delete_file.content_type(), DataContentType::EqualityDeletes);
    assert_eq!(
        delete_file.equality_ids(),
        Some(vec![1]),
        "the equality delete must carry equality_ids = [1] (field id of `id`)"
    );
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // 4. Sanity: OUR OWN scan→Arrow already applies the equality delete → {10,30,50} before handing to Java.
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
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "Rust's own scan of the written table must already be {{10,30,50}} (20/40 equality-deleted)"
    );

    // 5. Write the FINAL metadata to a KNOWN path so Java loads it deterministically.
    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_scan_exec equality-delete GEN OK — Rust wrote {table_location} (parquet data seq 1 + \
         equality-delete seq 2 + final.metadata.json); Rust scan = {{10,30,50}}. Java verify-interop-eq-delete \
         reads it next."
    );
}

// ===========================================================================================
// PARTITIONED merge-on-read, DIRECTION 2 — the GEN path: Rust WRITES a real on-disk PARTITIONED table
// (identity(category)) with a PARTITION-SCOPED position delete; Java reads it back. The partition-WRITE
// proof.
//
// The sibling of the position-delete GEN path above, but the table is PARTITIONED. We create a MemoryCatalog
// table with an identity(category) spec (spec id 0), write one REAL parquet data file PER PARTITION via the
// production `DataFileWriter` built with a `PartitionKey` (which auto-stamps the partition Struct + spec id
// onto the DataFile AND routes the parquet under the partition path via the location generator), and
// fast_append both at SEQUENCE 1. Then we write a PARTITION-SCOPED position-delete in partition a (deleting
// position 1 = id=20) via `PositionDeleteFileWriter` built with the cat=a `PartitionKey` (so the delete
// carries the partition Struct + spec id), and row_delta it at SEQUENCE 2. The table lands at
// `<gen_dir>/rust_table` with a `final.metadata.json`; the Java oracle's `verify-interop-part-scan` mode
// reads it with `IcebergGenerics` (applying our partition-scoped delete) and asserts {10,30,40,50}.
//
// When `ICEBERG_INTEROP_PART_SCAN_GEN_DIR` is UNSET this is a clean no-op — the offline gate stays green.
// ===========================================================================================

/// The PARTITIONED V2 schema Java expects: {1 id long required, 2 category string required, 3 data string
/// optional}. The partition column (`category`) is a required top-level field; the spec partitions by
/// identity(category).
fn part_gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, category string, data string} schema")
}

/// Build the identity(category) unbound partition spec (spec id 0) the partitioned table is created with.
fn part_gen_unbound_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category".to_string(), Transform::Identity)
        .expect("add identity(category) partition field")
        .build()
}

/// Create the PARTITIONED V2 table at EXACTLY `<gen_dir>/rust_table` in a `MemoryCatalog` over the local FS,
/// partitioned by identity(category) (spec id 0), so the on-disk layout is the deterministic
/// `rust_table/{metadata,data/category=.../...}` Java loads.
async fn create_partitioned_rust_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(part_gen_schema())
        .partition_spec(part_gen_unbound_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create partitioned rust_table")
}

/// Build the `PartitionKey` for a single identity(category) partition value (e.g. `"a"`). The bound spec is
/// the table's default partition spec; the partition `Struct` carries the single string category value.
fn category_partition_key(schema: SchemaRef, spec: PartitionSpec, category: &str) -> PartitionKey {
    PartitionKey::new(
        spec,
        schema,
        Struct::from_iter([Some(Literal::string(category))]),
    )
}

/// Write a REAL parquet DATA file for ONE partition via the production `DataFileWriter` built with the
/// partition's `PartitionKey`. The writer auto-stamps the partition `Struct` + spec id onto the returned
/// `DataFile` and routes the parquet under the partition path (`data/category=.../...`) via the location
/// generator. Each row's `category` matches the partition so the on-disk data is consistent with the stamp.
async fn write_partitioned_gen_data_file(
    table: &Table,
    partition_key: &PartitionKey,
    category: &str,
    ids: Vec<i64>,
    data_values: Vec<&str>,
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));

    let row_count = ids.len();
    let categories: Vec<&str> = std::iter::repeat_n(category, row_count).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(categories)) as ArrayRef,
        Arc::new(StringArray::from(data_values)) as ArrayRef,
    ])
    .expect("build the per-partition data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "rust-data".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
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

    // DataFileWriter built with the PartitionKey: close() stamps `partition` = the key's Struct and
    // `partition_spec_id` = the key's spec id, and the location generator routes the parquet under the
    // partition path — exactly how the production partitioning writers stamp partition values.
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key.clone()))
        .await
        .expect("build partitioned data file writer");
    writer
        .write(batch)
        .await
        .expect("write per-partition batch");
    writer
        .close()
        .await
        .expect("close partitioned data file writer")
        .into_iter()
        .next()
        .expect("one data file per partition")
}

/// Write a REAL parquet PARTITION-SCOPED position-delete file (via the production
/// `PositionDeleteFileWriter` built with the cat=a `PartitionKey`) deleting position 1 of `data_file_path`
/// (id=20). The writer stamps the partition `Struct` + spec id onto the delete file, so it is associated
/// with partition a — the delete-file index keys deletes by partition + spec id, reaching only the cat=a
/// data file.
async fn write_partitioned_gen_position_delete_file(
    table: &Table,
    partition_key: &PartitionKey,
    data_file_path: &str,
) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    // Build WITH the cat=a partition key so the delete is partition-scoped (carries the cat=a Struct + spec
    // id 0). This is the partition-handling proof for the WRITE side.
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key.clone()))
        .await
        .expect("build partition-scoped position-delete writer");

    let paths = StringArray::from(vec![data_file_path]);
    let positions = Int64Array::from(vec![1_i64]);
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(paths) as ArrayRef,
        Arc::new(positions) as ArrayRef,
    ])
    .expect("build the partition-scoped position-delete batch");
    writer
        .write(batch)
        .await
        .expect("write partition-scoped position-delete batch");
    writer
        .close()
        .await
        .expect("close partition-scoped position-delete writer")
        .into_iter()
        .next()
        .expect("one partition-scoped position-delete file")
}

#[tokio::test]
async fn test_part_scan_exec_gen_rust_writes_java_readable_partitioned_table() {
    let Some(gen_dir) = part_scan_gen_dir() else {
        println!(
            "skipping interop_scan_exec partitioned GEN — set ICEBERG_INTEROP_PART_SCAN_GEN_DIR \
             (run dev/java-interop/run-interop-part-d2.sh)"
        );
        return;
    };

    // 1. A MemoryCatalog over the LOCAL FS, warehouse = <gen_dir>, table pinned to <gen_dir>/rust_table,
    //    partitioned by identity(category) (spec id 0).
    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_part_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_partitioned_rust_table(&catalog, &table_location).await;

    // The bound default partition spec + schema the partition keys reference.
    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let partition_key_a = category_partition_key(schema.clone(), spec.clone(), "a");
    let partition_key_b = category_partition_key(schema.clone(), spec.clone(), "b");

    // 2. Write one REAL parquet data file PER PARTITION (each stamped with its partition value), then
    //    fast_append BOTH at SEQUENCE 1. cat=a: (10,a,x),(20,a,y),(30,a,z); cat=b: (40,b,p),(50,b,q).
    let data_file_a =
        write_partitioned_gen_data_file(&table, &partition_key_a, "a", vec![10, 20, 30], vec![
            "x", "y", "z",
        ])
        .await;
    let data_file_b =
        write_partitioned_gen_data_file(&table, &partition_key_b, "b", vec![40, 50], vec![
            "p", "q",
        ])
        .await;

    // Sanity: each data file carries the RIGHT partition value (category Struct) + spec id 0.
    assert_eq!(data_file_a.content_type(), DataContentType::Data);
    assert_eq!(data_file_b.content_type(), DataContentType::Data);
    assert_eq!(
        data_file_a.partition(),
        &Struct::from_iter([Some(Literal::string("a"))]),
        "cat=a data file must carry the category=a partition value"
    );
    assert_eq!(
        data_file_b.partition(),
        &Struct::from_iter([Some(Literal::string("b"))]),
        "cat=b data file must carry the category=b partition value"
    );

    let data_file_a_path = data_file_a.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file_a, data_file_b])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // 3. row_delta a PARTITION-SCOPED position-delete in partition a (position 1 of the cat=a data file =
    //    id=20) at SEQUENCE 2. Because the data (seq 1) was committed FIRST, the delete (seq 2) applies.
    let delete_file =
        write_partitioned_gen_position_delete_file(&table, &partition_key_a, &data_file_a_path)
            .await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    assert_eq!(
        delete_file.partition(),
        &Struct::from_iter([Some(Literal::string("a"))]),
        "the position-delete must be PARTITION-SCOPED to category=a"
    );
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // 4. Sanity: OUR OWN scan→Arrow already applies the partition-scoped delete → {10,30,40,50} (only id=20
    //    deleted from cat=a; cat=b untouched). Confirm the table is internally consistent before Java reads.
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
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_rows(batch));
    }
    let rust_rows = sorted_by_id(rust_rows);
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 50],
        "Rust's own scan of the written partitioned table must already be {{10,30,40,50}} (cat=a id=20 deleted)"
    );

    // 5. Write the FINAL metadata to a KNOWN path so Java loads it deterministically. The real on-disk
    //    manifest-list + manifests + per-partition parquet already live under <gen_dir>/rust_table.
    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_scan_exec partitioned GEN OK — Rust wrote {table_location} (per-partition parquet data + \
         partition-scoped position-delete + final.metadata.json); Rust scan = {{10,30,40,50}}. Java \
         verify-interop-part-scan reads it next."
    );
}

// ===========================================================================================
// ENGINE CUSTOM-SCAN EQUIVALENCE — the public `DeleteFilter` (engine-facing merge-on-read delete
// application, mirroring Java `org.apache.iceberg.data.DeleteFilter`) reproduces the built-in scan
// EXACTLY. A downstream DataFusion-wrapped engine plans the files, does its OWN physical parquet read
// of each data file (the `parquet` crate directly — what a DataFusion `ParquetExec` does, NOT
// iceberg's delete-applying reader), then reuses iceberg's delete resolution via
// `DeleteFilter::{load, equality_delete_predicate, apply}` to drop deleted rows. These OFFLINE tests
// (no Java/Maven) prove that path yields rows IDENTICAL to `table.scan().to_arrow()` (which applies the
// same deletes internally, via the reader's `survival_mask`) across position deletes, equality deletes,
// and the two combined.
//
// This is the compose-equivalence the in-`delete_filter.rs` unit test does not reach: that test proves
// `load` + `apply` in isolation; here the SAME deletes, committed to a REAL table, are resolved two ways
// — the engine `DeleteFilter` over a raw read vs the built-in scan — and asserted equal. That equality IS
// the seam's contract: an engine can BYO physical scan and still get iceberg-correct merge-on-read rows.
//
// Non-vacuity: the equality-delete predicate resolves the `id` column by Iceberg field id from the raw
// parquet's `PARQUET_FIELD_ID_META_KEY` metadata. If that metadata did NOT round-trip, the column would
// read as absent, the predicate would keep every row, and the engine path would diverge from ground
// truth — so these assertions FAIL LOUDLY rather than pass vacuously if the round-trip ever breaks.
// ===========================================================================================

/// Strip a `file://` scheme so a `FileScanTask` data-file path opens as a local filesystem path (the
/// `MemoryCatalog` over `LocalFsStorageFactory` writes absolute local paths; tolerate either form).
fn strip_file_scheme(path: &str) -> &str {
    path.strip_prefix("file://").unwrap_or(path)
}

/// Write a REAL parquet EQUALITY-delete file keyed on field id 1 (the `id` column), deleting the given
/// `ids`, unpartitioned, via the production `EqualityDeleteFileWriter`. The writer projects the table
/// schema down to the single `id` column, so the `data` values are placeholders (projected away — only
/// `id` lands on disk) and the stamped delete carries content `EqualityDeletes` + `equality_ids = [1]`.
async fn write_equality_delete_for_ids(table: &Table, ids: &[i64]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    // equality_ids = [1] (the `id` field). The config builds a projector from the FULL table schema down to
    // just the `id` column, so we feed it a FULL-schema (id, data) batch and it extracts the `id` values.
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete writer config (equality_ids = [1])");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "eq-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    // The parquet writer must use the PROJECTED schema (just `id`), since that is what lands on disk.
    let projected_iceberg_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected arrow schema → iceberg schema"),
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        projected_iceberg_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(None)
        .await
        .expect("build equality-delete writer");

    // A FULL-schema (id, data) batch carrying the delete keys; the writer's projector keeps only `id`. The
    // `data` values are projected away, but the batch must match the full table schema so the column-index
    // projection resolves.
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let data: Vec<&str> = std::iter::repeat_n("x", ids.len()).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(data)) as ArrayRef,
    ])
    .expect("build the equality-delete key batch");
    writer
        .write(batch)
        .await
        .expect("write equality-delete batch");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

/// The DOWNSTREAM-ENGINE custom-scan path, end to end through the PUBLIC surface: plan the files, do the
/// engine's OWN physical parquet read of each data file (the `parquet` crate directly — what a DataFusion
/// `ParquetExec` does, NOT iceberg's delete-applying reader), then reuse iceberg's merge-on-read delete
/// resolution via the public [`DeleteFilter`] to drop deleted rows. Returns the surviving `(id, data)`
/// rows — which MUST equal the built-in `to_arrow()` scan.
async fn engine_custom_scan_rows(table: &Table) -> Vec<ScanRow> {
    let tasks: Vec<_> = table
        .scan()
        .build()
        .expect("build scan")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect()
        .await
        .expect("collect scan tasks");

    let mut rows = Vec::new();
    for task in &tasks {
        // Resolve the task's deletes once (position deletes + DVs eagerly; the equality predicate lazily).
        let deletes = DeleteFilter::load(task, table.file_io().clone())
            .await
            .expect("load DeleteFilter");
        let equality_predicate = deletes
            .equality_delete_predicate(task)
            .await
            .expect("resolve equality-delete predicate");

        // The engine's OWN read of the data file: every physical row, in file order, NO deletes applied.
        let file = fs::File::open(strip_file_scheme(task.data_file_path()))
            .expect("open data-file parquet for the engine's own read");
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("parquet reader builder")
            .build()
            .expect("build parquet record-batch reader");

        // Apply the deletes batch by batch, tracking the absolute file position of each batch's row 0.
        let mut row_base = 0u64;
        for batch in reader {
            let batch = batch.expect("read a data batch");
            let row_count = batch.num_rows() as u64;
            let survivors = deletes
                .apply(task, batch, row_base, equality_predicate.as_ref())
                .expect("apply merge-on-read deletes");
            rows.extend(extract_rows(&survivors));
            row_base += row_count;
        }
    }
    rows
}

/// The BUILT-IN scan (the ground truth): `table.scan().to_arrow()` applies the SAME merge-on-read deletes
/// internally (the reader's `survival_mask`). Returns the surviving `(id, data)` rows.
async fn builtin_scan_rows(table: &Table) -> Vec<ScanRow> {
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
    batches.iter().flat_map(extract_rows).collect()
}

/// Build a fresh unpartitioned 5-row table {(10,a)..(50,e)} in `catalog` at `table_location` and
/// `fast_append` it (sequence 1). Returns the committed table + the data file's path — the shared setup
/// for the equivalence scenarios below.
async fn append_5row_table(catalog: &impl Catalog, table_location: &str) -> (Table, String) {
    let table = create_rust_table(catalog, table_location).await;
    let data_file = write_gen_data_file(&table).await;
    let data_file_path = data_file.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(catalog).await.expect("commit fast append");
    (table, data_file_path)
}

#[tokio::test]
async fn test_engine_deletefilter_equivalence_position_deletes() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_pos",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let (table, data_file_path) = append_5row_table(&catalog, &table_location).await;

    // Commit a POSITION delete of ids 20/40 (positions 1/3), discovered the way an engine would.
    let mut pairs = discover_row_identities(&table, &[20, 40]).await;
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            (data_file_path.clone(), 1_i64),
            (data_file_path.clone(), 3_i64),
        ],
        "ids 20/40 sit at file positions 1/3"
    );
    let delete_file = write_pos_delete_from_pairs(&table, &pairs).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // The public `deleted_row_positions` (≈ Java `deletedRowPositions()`) reports exactly {1, 3}.
    {
        let tasks: Vec<_> = table
            .scan()
            .build()
            .expect("build scan")
            .plan_files()
            .await
            .expect("plan files")
            .try_collect()
            .await
            .expect("collect tasks");
        let deletes = DeleteFilter::load(&tasks[0], table.file_io().clone())
            .await
            .expect("load DeleteFilter");
        let positions = deletes
            .deleted_row_positions(&tasks[0])
            .expect("a positional delete vector is present");
        let guard = positions.lock().expect("lock the delete vector");
        assert_eq!(guard.len(), 2, "exactly two positions are deleted");
        assert!(
            guard.contains(1) && guard.contains(3),
            "positions 1 and 3 (ids 20, 40) are the deleted positions"
        );
    }

    // The equivalence: engine custom-scan DeleteFilter path == built-in MoR scan == {10, 30, 50}.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine DeleteFilter (raw read + apply) must equal the built-in delete-applying scan"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "position-delete survivors are exactly {{10, 30, 50}}"
    );
}

#[tokio::test]
async fn test_engine_deletefilter_equivalence_equality_deletes() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_eq",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let (table, _data_file_path) = append_5row_table(&catalog, &table_location).await;

    // Commit an EQUALITY delete of ids 20/40 (delete-by-VALUE, keyed on field id 1) at sequence 2.
    let delete_file = write_equality_delete_for_ids(&table, &[20, 40]).await;
    assert_eq!(delete_file.content_type(), DataContentType::EqualityDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // The public accessors: an equality-delete predicate is present (≈ Java `eqDeletedRowFilter()`), and
    // there are no positional deletes for an equality-only task.
    {
        let tasks: Vec<_> = table
            .scan()
            .build()
            .expect("build scan")
            .plan_files()
            .await
            .expect("plan files")
            .try_collect()
            .await
            .expect("collect tasks");
        let deletes = DeleteFilter::load(&tasks[0], table.file_io().clone())
            .await
            .expect("load DeleteFilter");
        assert!(
            deletes
                .equality_delete_predicate(&tasks[0])
                .await
                .expect("resolve equality-delete predicate")
                .is_some(),
            "an equality-delete predicate is present"
        );
        let positions = deletes.deleted_row_positions(&tasks[0]);
        assert!(
            positions.is_none_or(|dv| dv.lock().expect("lock the delete vector").is_empty()),
            "an equality-only task has no positional deletes"
        );
    }

    // The equivalence: engine equality-DeleteFilter path == built-in equality MoR scan == {10, 30, 50}.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine equality DeleteFilter (raw read + predicate apply) must equal the built-in equality scan"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 50],
        "equality-delete survivors are exactly {{10, 30, 50}}"
    );
}

#[tokio::test]
async fn test_engine_deletefilter_equivalence_position_and_equality_deletes() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_combined",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let (table, data_file_path) = append_5row_table(&catalog, &table_location).await;

    // Commit BOTH a position delete (position 0 = id 10) AND an equality delete (id 30) in ONE row_delta
    // at sequence 2 — this exercises DeleteFilter::apply's combined positional-AND-equality mask path.
    let pos_delete = write_pos_delete_from_pairs(&table, &[(data_file_path.clone(), 0)]).await;
    let eq_delete = write_equality_delete_for_ids(&table, &[30]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![pos_delete, eq_delete])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // The equivalence: engine combined-DeleteFilter path == built-in combined MoR scan == {20, 40, 50}.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine combined DeleteFilter (positional AND equality mask) must equal the built-in scan"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![20, 40, 50],
        "combined survivors are exactly {{20, 40, 50}} (position 0 = id 10, and id 30, deleted)"
    );
}

// ===========================================================================================
// MULTI-FILE-PER-PARTITION merge-on-read — the delete-apply residue the engine-first closeout flags
// (GAP_MATRIX "Read: merge-on-read apply" 🟡). When a single partition holds MORE THAN ONE data file,
// a partition-scoped position-delete file (routed to EVERY data-file task in that partition by the
// `DeleteFileIndex` partition keying) must apply ONLY to the rows of the data file it REFERENCES BY
// PATH — its siblings in the same partition must be untouched. This is the correctness property the
// existing one-file-per-partition interop never exercised; the path-keyed loader
// (`parse_positional_deletes_record_batch_stream` groups positions into a `HashMap<file_path, _>`)
// is what makes it hold. This OFFLINE test proves it at the scan AND engine-`DeleteFilter` level over
// a REAL committed two-files-in-one-partition table. If the loader ever partition-broadcast positions
// instead of path-keying them, the sibling row would wrongly vanish and this assertion would FAIL.
// ===========================================================================================

#[tokio::test]
async fn test_engine_deletefilter_multifile_partition_position_delete_spares_sibling() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_multifile",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_partitioned_rust_table(&catalog, &table_location).await;

    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let key_a = category_partition_key(schema.clone(), spec.clone(), "a");

    // TWO data files in the SAME partition (category=a): file1 ids {10,20,30} @ positions 0..2,
    // file2 ids {40,50,60} @ positions 0..2. Both stamped category=a (spec id 0).
    let file1 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![10, 20, 30], vec!["x", "y", "z"])
            .await;
    let file2 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![40, 50, 60], vec!["p", "q", "r"])
            .await;
    let file1_path = file1.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file1, file2])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // A partition-scoped position delete referencing FILE1 at position 1 (id 20). The DeleteFileIndex
    // routes it to BOTH data-file tasks (same partition), so correctness hinges on the loader applying
    // it ONLY to file1 by path — file2's position 1 (id 50) must be SPARED.
    let delete_file = write_partitioned_gen_position_delete_file(&table, &key_a, &file1_path).await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // Built-in scan and the engine DeleteFilter path must agree, and both must drop ONLY file1's id 20.
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine DeleteFilter path == built-in scan for a multi-file partition"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 50, 60],
        "only file1's position 1 (id 20) is deleted; file2's same-ordinal position 1 (id 50) is SPARED"
    );
    assert!(
        engine.iter().any(|row| row.id == 50),
        "id 50 (file2 position 1) MUST survive — position deletes are path-keyed, not partition-broadcast"
    );
    assert!(
        !engine.iter().any(|row| row.id == 20),
        "id 20 (file1 position 1) must be deleted"
    );
}

/// Write a REAL parquet PARTITION-SCOPED EQUALITY-delete file (via the production
/// `EqualityDeleteFileWriter` built WITH `partition_key`) keyed on field id 1 (`id`), deleting the given
/// `ids`, for the `{id, category, data}` partitioned schema. The writer stamps the partition `Struct` +
/// spec id onto the delete file (so the `DeleteFileIndex` routes it to that partition's data-file tasks)
/// and projects the batch down to just `id` (category/data are placeholders, projected away).
async fn write_partitioned_equality_delete_for_ids(
    table: &Table,
    partition_key: &PartitionKey,
    category: &str,
    ids: &[i64],
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete writer config (equality_ids = [1])");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "eq-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let projected_iceberg_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected arrow schema → iceberg schema"),
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        projected_iceberg_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    // Build WITH the partition key so the eq delete is partition-scoped (carries the partition Struct + spec id).
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(Some(partition_key.clone()))
        .await
        .expect("build partition-scoped equality-delete writer");

    // A FULL-schema {id, category, data} batch carrying the delete keys; the projector keeps only `id`.
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let categories: Vec<&str> = std::iter::repeat_n(category, ids.len()).collect();
    let data: Vec<&str> = std::iter::repeat_n("x", ids.len()).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(categories)) as ArrayRef,
        Arc::new(StringArray::from(data)) as ArrayRef,
    ])
    .expect("build the partition-scoped equality-delete key batch");
    writer
        .write(batch)
        .await
        .expect("write equality-delete batch");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

#[tokio::test]
async fn test_engine_deletefilter_multifile_partition_equality_delete_applies_across_files() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "equivalence_multifile_eq",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let table = create_partitioned_rust_table(&catalog, &table_location).await;

    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let key_a = category_partition_key(schema.clone(), spec.clone(), "a");

    // TWO data files in the SAME partition (category=a): file1 ids {10,20,30}, file2 ids {40,50,60}.
    let file1 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![10, 20, 30], vec!["x", "y", "z"])
            .await;
    let file2 =
        write_partitioned_gen_data_file(&table, &key_a, "a", vec![40, 50, 60], vec!["p", "q", "r"])
            .await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file1, file2])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // A partition-scoped equality delete keyed on `id` for {20, 50} — id 20 lives in file1, id 50 in
    // file2. It must apply to BOTH data files in partition a (the index returns a partition's eq deletes
    // for every data-file task in it), dropping id 20 from file1 AND id 50 from file2.
    let delete_file =
        write_partitioned_equality_delete_for_ids(&table, &key_a, "a", &[20, 50]).await;
    assert_eq!(delete_file.content_type(), DataContentType::EqualityDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // Built-in scan and the engine DeleteFilter path must agree, dropping id 20 (file1) AND id 50 (file2).
    let ground_truth = sorted_by_id(builtin_scan_rows(&table).await);
    let engine = sorted_by_id(engine_custom_scan_rows(&table).await);
    assert_eq!(
        engine, ground_truth,
        "engine equality DeleteFilter path == built-in scan across a multi-file partition"
    );
    let live_ids: Vec<i64> = engine.iter().map(|row| row.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 30, 40, 60],
        "the partition eq-delete drops id 20 (file1) AND id 50 (file2) — it applies to BOTH files"
    );
}
