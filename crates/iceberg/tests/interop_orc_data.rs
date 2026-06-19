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

//! ORC DATA-FILE READ interop (GAP_MATRIX row 116, Direction 1 — "Rust reads what Java writes").
//!
//! Java's `generate-interop-orc-data` mode (`OrcDataOracle`) writes a V2 Iceberg table whose DATA
//! file is in the **ORC** format (`GenericAppenderFactory.newDataWriter(..., FileFormat.ORC, null)`
//! → iceberg-orc's `GenericOrcWriter`, the production ORC data writer — which stamps the `iceberg.id`
//! ORC type attributes the Rust by-field-id reader resolves on) over a fixture that covers every
//! Iceberg primitive + logical type + an optional/null column, across 5 rows; a real parquet
//! POSITION-delete removes one row (position 1, the id=20 row). It materializes its OWN
//! `IcebergGenerics` read — asserted equal to a hand-declared expected on the Java side — into
//! `java_orc_rows.json`.
//!
//! THIS test loads the SAME `final.metadata.json`, builds a `Table` over a local-filesystem `FileIO`,
//! runs `scan().build()?.to_arrow()` (which materializes the ORC data file via the U2 ORC scan path
//! AND applies the position delete), extracts every column into the SAME canonical shape Java emitted,
//! and asserts ROW-IDENTITY against `java_orc_rows.json`. The expected rows are HAND-DECLARED
//! identically on both sides (anti-circular): a bug that reads the wrong value on either side fails.
//!
//! The interop fixture covers every Iceberg PRIMITIVE + LOGICAL type + an OPTIONAL column. Nested
//! struct/list/map are NOT in the interop fixture (the U1 ORC reader rejects nested top-level fields
//! loudly — nested ORC read-by-id is deferred); flat primitive + logical + optional/null is the full
//! cross-engine surface here.
//!
//! Gated on `ICEBERG_INTEROP_ORC_DATA_DIR`: a clean no-op when unset, so the offline gate stays
//! green. If Rust did NOT read the ORC data file (the pre-U2 state, where an ORC file failed as a
//! `FeatureUnsupported` error) this test would FAIL — a real read gap.

use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::io::FileIO;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;
use serde::Deserialize;

/// One canonical row, deserialized from `java_orc_rows.json` and reconstructed from the Rust scan.
/// Every Iceberg type the fixture covers is rendered to a JSON-stable form: logical types as their
/// physical encoding (`dt` = days, `ts` = micros), decimal as a plain string, binary as lowercase hex.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct OrcRow {
    id: i64,
    name: Option<String>,
    i32: i32,
    f64: f64,
    flag: bool,
    dt: i64,
    ts: i64,
    dec: String,
    bin: String,
}

fn sorted_by_id(mut rows: Vec<OrcRow>) -> Vec<OrcRow> {
    rows.sort_by(|a, b| a.id.cmp(&b.id).then(Ordering::Equal));
    rows
}

fn orc_data_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_ORC_DATA_DIR").map(PathBuf::from)
}

fn read_java_rows(dir: &std::path::Path) -> Vec<OrcRow> {
    let path = dir.join("java_orc_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<OrcRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn load_table(dir: &std::path::Path) -> Table {
    let metadata_path = dir.join("table/metadata/final.metadata.json");
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", "orc_data"]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from Java-written final.metadata.json")
}

/// Lowercase hex of a binary value (matching Java's `bytesToHex`).
fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Render a Decimal128 value (scale 2) to a plain decimal string matching Java
/// `BigDecimal.setScale(2).toPlainString()`.
fn decimal_plain_string(unscaled: i128, scale: i8) -> String {
    let scale = scale as usize;
    let negative = unscaled < 0;
    let mut digits = unscaled.unsigned_abs().to_string();
    if scale > 0 {
        while digits.len() <= scale {
            digits.insert(0, '0');
        }
        let point = digits.len() - scale;
        digits.insert(point, '.');
    }
    if negative {
        digits.insert(0, '-');
    }
    digits
}

/// Extract one batch's rows into the canonical [`OrcRow`] shape, by column name.
fn extract_rows(batch: &RecordBatch) -> Vec<OrcRow> {
    let id = batch
        .column_by_name("id")
        .expect("id")
        .as_primitive::<Int64Type>();
    let name = batch.column_by_name("name").expect("name");
    let i32c = batch
        .column_by_name("i32")
        .expect("i32")
        .as_primitive::<Int32Type>();
    let f64c = batch
        .column_by_name("f64")
        .expect("f64")
        .as_primitive::<arrow_array::types::Float64Type>();
    let flag = batch.column_by_name("flag").expect("flag").as_boolean();
    let dt = batch
        .column_by_name("dt")
        .expect("dt")
        .as_primitive::<Date32Type>();
    let ts = batch
        .column_by_name("ts")
        .expect("ts")
        .as_primitive::<TimestampMicrosecondType>();
    let dec = batch
        .column_by_name("dec")
        .expect("dec")
        .as_primitive::<Decimal128Type>();
    let dec_scale = match dec.data_type() {
        arrow_schema::DataType::Decimal128(_, s) => *s,
        other => panic!("dec column is not Decimal128: {other:?}"),
    };
    let bin = batch.column_by_name("bin").expect("bin");

    (0..batch.num_rows())
        .map(|i| {
            let name_val = if name.is_null(i) {
                None
            } else {
                Some(string_at(name, i))
            };
            let bin_val = bytes_to_hex(&binary_at(bin, i));

            OrcRow {
                id: id.value(i),
                name: name_val,
                i32: i32c.value(i),
                f64: f64c.value(i),
                flag: flag.value(i),
                dt: dt.value(i) as i64,
                ts: ts.value(i),
                dec: decimal_plain_string(dec.value(i), dec_scale),
                bin: bin_val,
            }
        })
        .collect()
}

/// Read a non-null string cell (Utf8 or LargeUtf8).
fn string_at(array: &arrow_array::ArrayRef, i: usize) -> String {
    use arrow_schema::DataType;
    match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value(i).to_string(),
        DataType::LargeUtf8 => array.as_string::<i64>().value(i).to_string(),
        other => panic!("unexpected string arrow type: {other:?}"),
    }
}

/// Read a non-null binary cell (Binary or LargeBinary).
fn binary_at(array: &arrow_array::ArrayRef, i: usize) -> Vec<u8> {
    use arrow_schema::DataType;
    match array.data_type() {
        DataType::Binary => array.as_binary::<i32>().value(i).to_vec(),
        DataType::LargeBinary => array.as_binary::<i64>().value(i).to_vec(),
        other => panic!("unexpected binary arrow type: {other:?}"),
    }
}

#[tokio::test]
async fn test_orc_data_scan_matches_java_read() {
    let Some(dir) = orc_data_dir() else {
        println!(
            "skipping interop_orc_data — set ICEBERG_INTEROP_ORC_DATA_DIR \
             (run dev/java-interop/run-interop-orc-data.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    let batch_stream = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow over an ORC data file");
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

    // The deleted row (id=20, position 1) must be ABSENT after merge-on-read over the ORC data file.
    assert!(
        !rust_rows.iter().any(|r| r.id == 20),
        "id=20 (deleted at position 1) must be ABSENT after merge-on-read over the ORC data file"
    );

    // 4 of 5 rows survive (position 1 deleted).
    assert_eq!(
        rust_rows.len(),
        4,
        "exactly 4 rows survive (5 written to the ORC data file, position 1 deleted)"
    );

    // The full row-identity proof: every column (primitive + logical + optional null) of every
    // surviving row equals Java's hand-declared expected. (Nested struct/list/map are excluded from
    // this oracle — nested ORC read-by-id is deferred; the U1 offline tests cover the decode core.)
    assert_eq!(
        rust_rows, java_rows,
        "Rust scan→Arrow of the ORC data file (merge-on-read) must equal Java's expected rows \
         column-for-column"
    );

    // Pin the live id set so it cannot drift unnoticed.
    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(live_ids, vec![10, 30, 40, 50]);

    println!(
        "interop_orc_data OK — Rust scan→Arrow over an ORC data file (merge-on-read) = Java's \
         expected: 4 live rows {{10,30,40,50}}, id=20 deleted, every column identical"
    );
}
