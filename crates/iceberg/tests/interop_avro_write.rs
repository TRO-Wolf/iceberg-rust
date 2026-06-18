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

//! AVRO DATA-FILE WRITE interop (GAP_MATRIX row 117, Direction 2 — "Java reads what Rust writes").
//!
//! The mirror-image (and INVERSION) of [`interop_avro_data`] (Direction 1, where Java writes the Avro
//! data file and Rust reads it). HERE the **W1 production Avro data writer**
//! ([`AvroWriterBuilder`](iceberg::writer::file_writer::AvroWriterBuilder) /
//! [`AvroWriter`](iceberg::writer::file_writer::AvroWriter)) writes `00000-rust-data.avro` over the
//! SAME flat fixture `AvroDataOracle` uses, and Java reads that raw file (via
//! `Avro.read(...).project(schema).createReaderFunc(PlannedDataReader::create).build()`) and asserts
//! row-identity against TWO independent anchors: (a) Java's OWN hand-declared constants, and (b) the
//! Rust-emitted expected JSON (`rust_avro_rows.json`). The `.avro` file is the ONLY artifact that
//! crosses the engine boundary.
//!
//! NO delete is applied — all 5 raw rows are read back (merge-on-read over Avro is already Direction
//! 1's job). The fixture is FLAT (every Iceberg primitive + logical type + one optional/null column):
//! nested struct/list/map WRITE is proven offline by the U1 round-trip tests in `avro_writer.rs`, but
//! Java's `GenericAppenderFactory` AVRO writer + `PlannedDataReader` cannot round-trip a nested record
//! cross-engine, so the oracle fixture stays flat — same boundary the Direction-1 oracle documents.
//!
//! ANTI-CIRCULAR. The expected rows are HAND-DECLARED here from the Rust literal constant arrays
//! (NOT by reading back the `.avro` — that would be circular), in the SAME canonical shape Java emits.
//! Java separately hand-declares the SAME constants and compares the read both ways. A bug that writes
//! the wrong value fails the Java read; a bug that wrongly declares the expected fails one of the two
//! anchors.
//!
//! Gated on `ICEBERG_INTEROP_AVRO_WRITE_DIR`: a clean no-op (prints "skipping") when unset, so the
//! offline `cargo test` gate stays green (this test needs the Java oracle to actually verify).

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array, Int64Array,
    LargeBinaryArray, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::SchemaRef as ArrowSchemaRef;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::io::FileIO;
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::writer::file_writer::{AvroWriterBuilder, FileWriter, FileWriterBuilder};

// ============================================================================================
// THE FIXTURE — hand-declared, byte-identical to the Java `AvroDataOracle` constants.
// Row i (0-based) carries id = 10*(i+1). `name` is optional: row index 2 (id=30) is NULL.
// ============================================================================================

const IDS: [i64; 5] = [10, 20, 30, 40, 50];
/// `name` is optional: index 2 (id=30) is NULL to exercise the optional/null path.
const NAMES: [Option<&str>; 5] = [
    Some("alpha"),
    Some("bravo"),
    None,
    Some("delta"),
    Some("echo"),
];
const I32: [i32; 5] = [1, 2, 3, 4, 5];
const F64: [f64; 5] = [1.5, 2.5, 3.5, 4.5, 5.5];
const FLAG: [bool; 5] = [true, false, true, false, true];
/// dt: days since epoch.
const DT_DAYS: [i32; 5] = [0, 100, 19000, 19001, 19002];
/// ts: micros since epoch (timestamptz, UTC).
const TS_MICROS: [i64; 5] = [
    0,
    1_000_000,
    1_600_000_000_000_000,
    1_600_000_000_000_001,
    1_600_000_000_000_002,
];
/// dec: decimal(9,2) — the plain-string form Java emits via `BigDecimal.setScale(2).toPlainString()`.
const DEC_PLAIN: [&str; 5] = ["1.23", "-4.56", "78.90", "0.01", "-0.99"];
/// dec: the unscaled i128 (scale 2) the Arrow `Decimal128` column carries.
const DEC_UNSCALED: [i128; 5] = [123, -456, 7890, 1, -99];
/// bin: lowercase hex. Row index 3 (id=40) is the EMPTY binary value.
const BIN_HEX: [&str; 5] = ["00", "01ff", "deadbeef", "", "7f"];

/// The Iceberg fixture schema, byte-for-byte the Java `AvroDataOracle.schema()`:
///   id(1) long, name(2) string?, i32(3) int, f64(4) double, flag(5) boolean, dt(6) date,
///   ts(7) timestamptz, dec(8) decimal(9,2), bin(9) binary.
fn fixture_schema() -> Schema {
    Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "i32", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(4, "f64", Type::Primitive(PrimitiveType::Double)).into(),
            NestedField::required(5, "flag", Type::Primitive(PrimitiveType::Boolean)).into(),
            NestedField::required(6, "dt", Type::Primitive(PrimitiveType::Date)).into(),
            NestedField::required(7, "ts", Type::Primitive(PrimitiveType::Timestamptz)).into(),
            NestedField::required(
                8,
                "dec",
                Type::Primitive(PrimitiveType::Decimal {
                    precision: 9,
                    scale: 2,
                }),
            )
            .into(),
            NestedField::required(9, "bin", Type::Primitive(PrimitiveType::Binary)).into(),
        ])
        .build()
        .expect("build the flat fixture schema")
}

/// Build the 5-row Arrow batch. The Arrow schema is derived via [`schema_to_arrow_schema`] so each
/// field carries its id in `PARQUET_FIELD_ID_META_KEY` — the W1 writer matches columns by id.
fn fixture_batch(schema: &Schema) -> RecordBatch {
    let arrow_schema: ArrowSchemaRef =
        Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));

    let id = Arc::new(Int64Array::from(IDS.to_vec())) as ArrayRef;
    let name = Arc::new(StringArray::from(
        NAMES
            .iter()
            .map(|n| n.map(str::to_string))
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    let i32c = Arc::new(Int32Array::from(I32.to_vec())) as ArrayRef;
    let f64c = Arc::new(Float64Array::from(F64.to_vec())) as ArrayRef;
    let flag = Arc::new(BooleanArray::from(FLAG.to_vec())) as ArrayRef;
    let dt = Arc::new(Date32Array::from(DT_DAYS.to_vec())) as ArrayRef;
    let ts = Arc::new(TimestampMicrosecondArray::from(TS_MICROS.to_vec()).with_timezone_utc())
        as ArrayRef;
    let dec = Arc::new(
        Decimal128Array::from(DEC_UNSCALED.to_vec())
            .with_precision_and_scale(9, 2)
            .expect("decimal(9,2)"),
    ) as ArrayRef;
    let bin = Arc::new(LargeBinaryArray::from(
        BIN_HEX
            .iter()
            .map(|h| hex_to_bytes(h))
            .collect::<Vec<_>>()
            .iter()
            .map(|v| Some(v.as_slice()))
            .collect::<Vec<_>>(),
    )) as ArrayRef;

    RecordBatch::try_new(arrow_schema, vec![
        id, name, i32c, f64c, flag, dt, ts, dec, bin,
    ])
    .expect("build the 5-row fixture batch")
}

/// Decode a lowercase-hex string to bytes (the inverse of Java `bytesToHex`). The empty string → `[]`.
fn hex_to_bytes(hex: &str) -> Vec<u8> {
    assert!(
        hex.len().is_multiple_of(2),
        "hex must be even-length: {hex:?}"
    );
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex"))
        .collect()
}

/// Emit `rust_avro_rows.json` = the expected rows HAND-DECLARED from the fixture constants (NOT by
/// reading back the `.avro`), in the SAME canonical shape Java emits: id i64, name Option<String>,
/// i32 i32, f64 f64, flag bool, dt epoch-days, ts micros, dec plain-string, bin lowercase-hex.
fn emit_rust_rows_json() -> String {
    let mut rows = String::from("[\n");
    for i in 0..IDS.len() {
        rows.push_str("  {\n");
        rows.push_str(&format!("    \"id\": {},\n", IDS[i]));
        match NAMES[i] {
            Some(n) => rows.push_str(&format!("    \"name\": {},\n", json_string(n))),
            None => rows.push_str("    \"name\": null,\n"),
        }
        rows.push_str(&format!("    \"i32\": {},\n", I32[i]));
        rows.push_str(&format!("    \"f64\": {},\n", json_f64(F64[i])));
        rows.push_str(&format!("    \"flag\": {},\n", FLAG[i]));
        rows.push_str(&format!("    \"dt\": {},\n", DT_DAYS[i] as i64));
        rows.push_str(&format!("    \"ts\": {},\n", TS_MICROS[i]));
        rows.push_str(&format!("    \"dec\": {},\n", json_string(DEC_PLAIN[i])));
        rows.push_str(&format!("    \"bin\": {}\n", json_string(BIN_HEX[i])));
        rows.push_str(if i + 1 == IDS.len() {
            "  }\n"
        } else {
            "  },\n"
        });
    }
    rows.push(']');
    rows
}

/// JSON-encode a string value (escaping is unnecessary for this fixture's ASCII values, but encode
/// defensively for the quote/backslash chars).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Render an f64 the way the fixture's values need (all are exact halves: 1.5, 2.5, ...).
fn json_f64(v: f64) -> String {
    // Every fixture value has a non-zero fractional part, so the default Rust formatting (e.g.
    // "1.5") is already JSON-valid and round-trips.
    format!("{v}")
}

fn avro_write_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_AVRO_WRITE_DIR").map(PathBuf::from)
}

/// Direction-2 GEN: write `00000-rust-data.avro` through the W1 production writer + emit
/// `rust_avro_rows.json`. Java then reads the raw `.avro` and verifies both anchors.
#[tokio::test]
async fn test_avro_write_gen_java_reads_rust_written_avro() {
    let Some(dir) = avro_write_dir() else {
        println!(
            "skipping interop_avro_write — set ICEBERG_INTEROP_AVRO_WRITE_DIR \
             (run dev/java-interop/run-interop-avro-write.sh)"
        );
        return;
    };

    fs::create_dir_all(&dir).unwrap_or_else(|error| panic!("create {}: {error}", dir.display()));

    let schema = Arc::new(fixture_schema());
    let batch = fixture_batch(&schema);

    // Write 00000-rust-data.avro THROUGH THE W1 PRODUCTION WRITER (load-bearing: the flip proves the
    // production writer — do NOT hand-roll a RawLiteral encode here).
    let avro_path = dir.join("00000-rust-data.avro");
    let avro_path_str = avro_path.to_string_lossy().to_string();
    let file_io = FileIO::new_with_fs();
    let output = file_io
        .new_output(&avro_path_str)
        .expect("new avro output file");

    let mut writer = AvroWriterBuilder::new(schema.clone())
        .build(output)
        .await
        .expect("build the W1 Avro production writer");
    writer.write(&batch).await.expect("write the 5-row batch");
    let data_files = writer.close().await.expect("close the Avro writer");
    assert_eq!(
        data_files.len(),
        1,
        "exactly one Avro data file must be produced"
    );

    // Emit the hand-declared expected rows (anti-circular: from the constants, NOT the .avro).
    let rows_json = emit_rust_rows_json();
    let rows_path = dir.join("rust_avro_rows.json");
    fs::write(&rows_path, &rows_json)
        .unwrap_or_else(|error| panic!("write {}: {error}", rows_path.display()));

    assert!(
        fs::metadata(&avro_path)
            .map(|m| m.len() > 0)
            .unwrap_or(false),
        "the written .avro file must be non-empty"
    );

    println!(
        "interop_avro_write GEN OK — wrote {} (W1 production writer, 5 rows) + {} (hand-declared \
         expected). Java now reads the raw .avro and asserts row-identity vs its own constants AND \
         this JSON.",
        avro_path.display(),
        rows_path.display()
    );
}
