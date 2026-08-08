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

//! Java interop for RANGED READS — which row groups a byte-range split actually decodes (U3 / hazard-1).
//!
//! Iceberg hands a reader a `(start, length)` window and expects it to read the row groups that BELONG to
//! that window — every row of the file exactly once across a full tiling, never twice. Java decides that
//! with parquet-mr's MIDPOINT rule: `Parquet.ReadBuilder.split(start, length)` →
//! `ParquetReadOptions.Builder.withRange(start, start + length)` →
//! `ParquetMetadataConverter.filterFileMetaDataByMidpoint`, which keeps a row group iff
//! `getOffset(columns[0]) + totalCompressedSize / 2` lies in the HALF-OPEN `[start, start + length)`.
//! `org.apache.iceberg.data.GenericReader.openFile` makes exactly that call for every `FileScanTask`.
//!
//! An OVERLAP rule instead hands a row group that straddles a split boundary to BOTH adjacent tasks —
//! silent duplicate rows, never an error. A SYNTHESIZED `4 + Σ compressed_size` offset model drifts on any
//! file whose row groups are not perfectly contiguous (padding, inline bloom filters), duplicating rows
//! even for splits that were aligned to the file's own row-group offsets. This suite pins both.
//!
//! ANTI-CIRCULARITY. The windows are NOT taken from either engine's splitter. Both sides tile
//! `[0, file_len)` at the HAND-DECLARED [`STRIDE`], mirrored in
//! `InteropOracle.RangedReadOracle.STRIDE`. Deriving them from `FileScanTask::split` /
//! `TableScanUtil.splitFiles` would make the comparison circular with respect to the split layer.
//!
//! TWO DIRECTIONS, two env gates (both a clean runtime NO-OP when unset, so the offline `cargo test` gate
//! needs no Java/Maven):
//!
//! * `ICEBERG_INTEROP_RANGED_READ_DIR` — DIRECTION 1. Java wrote `java_ranged.parquet` (many small row
//!   groups, so the tiling straddles them) and emitted `java_ranged_read.json` =
//!   `[{file,start,length,ids}]` read through its REAL midpoint filter. We read the SAME file over the
//!   SAME windows and assert identical id lists, plus the exactly-once tiling property.
//! * `ICEBERG_INTEROP_RANGED_READ_GEN_DIR` — DIRECTION 2. We write TWO fixtures — `rust_contig.parquet`
//!   (row groups back to back) and `rust_padded.parquet` (bloom filters, which parquet-rs places AFTER
//!   each row group, so the real starts run AHEAD of `4 + Σ compressed_size`) — read every window
//!   ourselves, and emit `rust_ranged_read.json` for Java's `verify-interop-ranged-read` to replay. The
//!   padded file is the leg that proves the OFFSET SOURCE, not merely the rule.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::arrow::ArrowReaderBuilder;
use iceberg::io::FileIO;
use iceberg::scan::{FileScanTask, FileScanTaskStream};
use iceberg::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, SchemaRef, Type};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde::Deserialize;

/// The HAND-DECLARED window stride in bytes. Mirrored EXACTLY in
/// `InteropOracle.RangedReadOracle.STRIDE` — changing one without the other breaks the oracle.
const STRIDE: u64 = 800;

/// Rows in each Rust-written fixture.
const ROWS: i64 = 400;

/// One `(file, window)` reading, as exchanged with the Java oracle.
#[derive(Debug, Deserialize)]
struct WindowRead {
    file: String,
    start: u64,
    length: u64,
    ids: Vec<i64>,
}

/// The shared fixture schema: `{1 id long required, 2 data string optional}`.
fn fixture_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("fixture schema"),
    )
}

fn fixture_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("data", DataType::Utf8, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]))
}

/// Tile `[0, file_len)` into half-open windows of [`STRIDE`] bytes; the last window is short.
fn tile(file_len: u64) -> Vec<(u64, u64)> {
    let mut windows = Vec::new();
    let mut start = 0u64;
    while start < file_len {
        let length = STRIDE.min(file_len - start);
        windows.push((start, length));
        start += length;
    }
    windows
}

/// Read one byte-range window through the production `ArrowReader` and return the ids in order.
async fn read_window(path: &str, start: u64, length: u64) -> Vec<i64> {
    let task = FileScanTask {
        file_size_in_bytes: std::fs::metadata(path).expect("stat fixture").len(),
        start,
        length,
        record_count: None,
        data_file_path: Arc::from(path.to_string()),
        data_file_format: DataFileFormat::Parquet,
        schema: fixture_schema(),
        project_field_ids: Arc::from(vec![1]),
        predicate: None,
        deletes: Arc::from(vec![]),
        partition: None,
        partition_spec: None,
        name_mapping: None,
        case_sensitive: false,
        split_offsets: None,
    };
    let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
    let batches: Vec<RecordBatch> = reader
        .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
        .expect("reader stream")
        .try_collect()
        .await
        .expect("ranged read");
    let mut ids = Vec::new();
    for batch in &batches {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id column is Int64");
        for i in 0..column.len() {
            ids.push(column.value(i));
        }
    }
    ids
}

/// Assert that the per-window readings tile the file: every id exactly once, none missing.
fn assert_exactly_once(label: &str, readings: &[(u64, u64, Vec<i64>)], expected_rows: i64) {
    let mut all: Vec<i64> = readings
        .iter()
        .flat_map(|(_, _, ids)| ids.clone())
        .collect();
    all.sort_unstable();
    assert_eq!(
        all,
        (0..expected_rows).collect::<Vec<i64>>(),
        "{label}: the union over a full tiling must be every row EXACTLY once (duplicates mean a \
         straddling row group was handed to two adjacent windows)"
    );
}

/// DIRECTION 1 — Rust must select the same row groups Java's real midpoint filter selects.
#[tokio::test]
async fn test_ranged_read_matches_java() {
    let Ok(dir) = std::env::var("ICEBERG_INTEROP_RANGED_READ_DIR") else {
        eprintln!("ICEBERG_INTEROP_RANGED_READ_DIR unset — skipping (clean no-op)");
        return;
    };

    let json = std::fs::read_to_string(format!("{dir}/java_ranged_read.json"))
        .expect("java_ranged_read.json (run the Java generate mode first)");
    let windows: Vec<WindowRead> =
        serde_json::from_str(&json).expect("parse java_ranged_read.json");
    assert!(
        windows.len() >= 3,
        "the Java oracle must declare several windows, got {}",
        windows.len()
    );

    let mut readings = Vec::new();
    let mut total_rows = 0i64;
    for window in &windows {
        let ids = read_window(&window.file, window.start, window.length).await;
        assert_eq!(
            ids,
            window.ids,
            "window [{}, {}) of {} must select exactly the row groups whose MIDPOINT lands in it \
             (Java parquet-mr filterFileMetaDataByMidpoint)",
            window.start,
            window.start + window.length,
            window.file
        );
        total_rows += ids.len() as i64;
        readings.push((window.start, window.length, ids));
    }
    // Non-vacuity: at least one window must be non-empty AND at least one window must be a strict
    // subset of the file, else the "tiling" was a single whole-file read and proves nothing.
    assert!(total_rows > 0, "the Java fixture read no rows at all");
    assert!(
        readings.iter().any(|(_, _, ids)| !ids.is_empty())
            && readings
                .iter()
                .filter(|(_, _, ids)| !ids.is_empty())
                .count()
                > 1,
        "the tiling must spread rows across MULTIPLE windows, else it cannot discriminate the \
         midpoint rule from an overlap rule"
    );
    assert_exactly_once("java fixture", &readings, total_rows);
}

/// DIRECTION 2 (GEN) — write the fixtures Java replays, including the bloom-PADDED offset-drift leg.
#[tokio::test]
async fn test_ranged_read_gen_rust_fixtures() {
    let Ok(dir) = std::env::var("ICEBERG_INTEROP_RANGED_READ_GEN_DIR") else {
        eprintln!("ICEBERG_INTEROP_RANGED_READ_GEN_DIR unset — skipping (clean no-op)");
        return;
    };
    std::fs::create_dir_all(&dir).expect("create gen dir");

    let mut out: Vec<serde_json::Value> = Vec::new();
    for (name, bloom) in [
        ("rust_contig.parquet", false),
        ("rust_padded.parquet", true),
    ] {
        let path = format!("{dir}/{name}");
        write_fixture(&path, bloom);

        let metadata = SerializedFileReader::new(File::open(&path).expect("open fixture"))
            .expect("read footer");
        let num_row_groups = metadata.metadata().num_row_groups();
        assert!(
            num_row_groups >= 3,
            "{name}: fixture must have several row groups, got {num_row_groups}"
        );
        if bloom {
            // Non-vacuity for the OFFSET-SOURCE leg: the padded file's real row-group starts must
            // differ from the `4 + Σ compressed_size` model, otherwise this fixture is a duplicate of
            // the contiguous one and proves nothing about where the offsets come from.
            let mut synthetic = 4u64;
            let mut drifted = false;
            for rg in metadata.metadata().row_groups() {
                let col = rg.columns().first().expect("row group has columns");
                let data = col.data_page_offset();
                let real = match col.dictionary_page_offset() {
                    Some(dict) if data > dict => dict,
                    _ => data,
                };
                if u64::try_from(real).expect("non-negative offset") != synthetic {
                    drifted = true;
                }
                synthetic += u64::try_from(rg.compressed_size()).expect("non-negative size");
            }
            assert!(
                drifted,
                "{name}: bloom filters must make the row groups NON-contiguous, else the \
                 offset-source leg is vacuous"
            );
        }

        let file_len = std::fs::metadata(&path).expect("stat").len();
        let windows = tile(file_len);
        let mut readings = Vec::new();
        for (start, length) in windows {
            let ids = read_window(&path, start, length).await;
            out.push(serde_json::json!({
                "file": path,
                "start": start,
                "length": length,
                "ids": ids,
            }));
            readings.push((start, length, ids));
        }
        assert_exactly_once(name, &readings, ROWS);
    }

    std::fs::write(
        format!("{dir}/rust_ranged_read.json"),
        serde_json::to_vec_pretty(&out).expect("serialize rust_ranged_read.json"),
    )
    .expect("write rust_ranged_read.json");
    eprintln!(
        "wrote {}/rust_ranged_read.json ({} windows)",
        dir,
        out.len()
    );
}

/// Write a [`ROWS`]-row fixture with small row groups; with `bloom` the writer emits a bloom-filter
/// section after each row group (parquet-rs's default position), making the row groups non-contiguous.
fn write_fixture(path: &str, bloom: bool) {
    let arrow_schema = fixture_arrow_schema();
    let ids: Vec<i64> = (0..ROWS).collect();
    let data: Vec<String> = (0..ROWS).map(|i| format!("row-{i}")).collect();
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int64Array::from(ids)),
        Arc::new(StringArray::from(data)),
    ])
    .expect("fixture batch");

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(50))
        .set_bloom_filter_enabled(bloom)
        .build();
    let file = File::create(path).expect("create fixture");
    let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).expect("arrow writer");
    writer.write(&batch).expect("write fixture");
    writer.close().expect("close fixture");
}
