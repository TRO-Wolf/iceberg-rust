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

use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::io::FileIO;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;
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
