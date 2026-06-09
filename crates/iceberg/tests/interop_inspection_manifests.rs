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

//! Java interop test for the MANIFEST-READING inspection tables `files` / `data_files` / `delete_files`
//! (the A1 foundation increment).
//!
//! Unlike the pure-metadata inspection interop ([`interop_inspection`], offline, reading committed JSON
//! fixtures), these tables read REAL ON-DISK AVRO MANIFESTS. The byte-level "read a table Java wrote"
//! proof therefore needs a real table: the Java oracle's `generate-inspection-manifests` mode WRITES a
//! partitioned V2 table to a temp dir via real commits (`newAppend` writes a DATA manifest + manifest-list;
//! `newRowDelta` writes a DELETE manifest), writes `final.metadata.json` to a known path, and materializes
//! the rows of Java's REAL `FilesTable` / `DataFilesTable` / `DeleteFilesTable` (via
//! `MetadataTableUtils.createMetadataTableInstance` + `task.asDataTask().rows()`, each `ManifestReadTask`
//! opening the on-disk AVRO) into `java_files.json` / `java_data_files.json` / `java_delete_files.json`.
//!
//! This test reads the SAME on-disk manifests: it loads `final.metadata.json`, builds a `Table` over a
//! local-filesystem `FileIO` (which resolves the absolute manifest paths the commits wrote), runs
//! `inspect().files()/.data_files()/.delete_files().scan()`, extracts EVERY column except the deferred
//! `readable_metrics` virtual struct, and asserts field-for-field equality against the Java rows
//! ORDER-INDEPENDENTLY (sorted by `file_path`).
//!
//! THE ENV GATE. Because the table is regenerated each run (nothing binary is committed), this test is
//! GATED on `ICEBERG_INTEROP_MANIFEST_DIR`. When the var is UNSET the test is a clean NO-OP (a runtime
//! early-return, NOT `#[ignore]`) so the offline `cargo test` gate stays green with no Java/Docker. The
//! `dev/java-interop/run-inspection-manifests.sh` script sets the var and runs the REAL comparison.
//!
//! DEFERRED — `readable_metrics`. The trailing virtual `readable_metrics` STRUCT column is DERIVED (one
//! per-leaf-column struct of human-readable min/max/counts). Its interior field ordering depends on a JVM
//! HashMap iteration order (a documented divergence), so it is OUT OF SCOPE for A1 — the RAW metric MAPS +
//! bound MAPS this test DOES compare are the load-bearing source those readable values derive from.
//!
//! `file_format` NOW MATCHES JAVA EXACTLY. Rust's `inspect` projection upper-cases the rendered
//! `file_format` column (`PARQUET`/`AVRO`/`ORC`) to match Java's `FilesTable`/`ManifestEntriesTable`, which
//! emit the UPPERCASE `FileFormat` enum NAME via `format.toString()`. The on-disk AVRO stores the lowercase
//! string on BOTH (the manifest serde is unchanged; Java/Rust read each other's lowercase via the
//! case-insensitive `from_str`). So the comparison asserts EXACT equality on `file_format` — no canonicalization.
//!
//! ONE KNOWN, NON-CORRUPTING REPRESENTATION DIVERGENCE (content-identical; surfaced, not hidden). It is a
//! presentation-only difference in how each library's metadata table RENDERS a column; the underlying
//! on-disk manifest value is identical, so it is NOT a production bug and NOT in scope to "fix" here (a fix
//! would be a spec-type change, out of bounds for an interop test). It is collapsed to a canonical form by
//! [`FileRow::canonical`] for the bulk equality AND pinned RAW by a focused assertion so it cannot drift
//! unnoticed:
//!   - ABSENT METRIC/BOUND MAP — empty `{}` vs `null`. Rust's `spec::DataFile` stores the metric/bound maps
//!     as NON-optional `HashMap`, so an absent map projects to an EMPTY map; Java stores `null` and emits
//!     JSON `null`. An empty map and a null map carry identical information (no metrics). Canonicalized by
//!     treating `None` and `Some(empty)` as equal.
//!
//! NO PRODUCTION CHANGE is needed: every OTHER column the Rust `files` family projects matches Java's
//! `FilesTable` row byte-for-byte when both read the same on-disk manifest. (If a column genuinely diverged
//! in CONTENT, the contract is to STOP and report it — never to hide a column. These two are content-equal.)

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray, StructArray};
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::io::FileIO;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;
use serde::Deserialize;

// ===========================================================================================
// The Java oracle row model — deserialized from java_{files,data_files,delete_files}.json. Every column is
// present EXCEPT the deferred `readable_metrics`. Bound maps are {field_id-as-string: hex-of-bytes}.
// ===========================================================================================

/// One row of Java's `FilesTable` / `DataFilesTable` / `DeleteFilesTable`, keyed by COLUMN NAME (the Java
/// oracle derives name→position from `mt.schema().columns()`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct JavaFileRow {
    content: i32,
    file_path: String,
    file_format: String,
    spec_id: i32,
    partition: JavaPartition,
    record_count: i64,
    file_size_in_bytes: i64,
    #[serde(default)]
    column_sizes: Option<HashMap<i32, i64>>,
    #[serde(default)]
    value_counts: Option<HashMap<i32, i64>>,
    #[serde(default)]
    null_value_counts: Option<HashMap<i32, i64>>,
    #[serde(default)]
    nan_value_counts: Option<HashMap<i32, i64>>,
    /// {field_id: hex-of-bytes} — decoded to raw bytes before comparison.
    #[serde(default)]
    lower_bounds: Option<HashMap<i32, String>>,
    #[serde(default)]
    upper_bounds: Option<HashMap<i32, String>>,
    #[serde(default)]
    key_metadata: Option<String>,
    #[serde(default)]
    split_offsets: Option<Vec<i64>>,
    #[serde(default)]
    equality_ids: Option<Vec<i32>>,
    #[serde(default)]
    sort_order_id: Option<i32>,
    #[serde(default)]
    first_row_id: Option<i64>,
    #[serde(default)]
    referenced_data_file: Option<String>,
    #[serde(default)]
    content_offset: Option<i64>,
    #[serde(default)]
    content_size_in_bytes: Option<i64>,
}

/// The identity-`category` partition tuple as a name→value object. Only `category` is present in this
/// fixture; kept as a map so the comparison is robust to sub-field ordering.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct JavaPartition {
    category: Option<String>,
}

/// A normalized, fully-comparable row — Java + Rust both materialize one of these so equality is a single
/// `==`. Bound maps are raw bytes (Java hex decoded; Rust binary verbatim).
#[derive(Debug, Clone, PartialEq)]
struct FileRow {
    content: i32,
    file_path: String,
    file_format: String,
    spec_id: i32,
    partition_category: Option<String>,
    record_count: i64,
    file_size_in_bytes: i64,
    column_sizes: Option<HashMap<i32, i64>>,
    value_counts: Option<HashMap<i32, i64>>,
    null_value_counts: Option<HashMap<i32, i64>>,
    nan_value_counts: Option<HashMap<i32, i64>>,
    lower_bounds: Option<HashMap<i32, Vec<u8>>>,
    upper_bounds: Option<HashMap<i32, Vec<u8>>>,
    key_metadata: Option<Vec<u8>>,
    split_offsets: Option<Vec<i64>>,
    equality_ids: Option<Vec<i32>>,
    sort_order_id: Option<i32>,
    first_row_id: Option<i64>,
    referenced_data_file: Option<String>,
    content_offset: Option<i64>,
    content_size_in_bytes: Option<i64>,
}

impl JavaFileRow {
    /// Decode the Java row into the comparable [`FileRow`] (hex bound maps → raw bytes).
    fn into_file_row(self) -> FileRow {
        FileRow {
            content: self.content,
            file_path: self.file_path,
            file_format: self.file_format,
            spec_id: self.spec_id,
            partition_category: self.partition.category,
            record_count: self.record_count,
            file_size_in_bytes: self.file_size_in_bytes,
            column_sizes: self.column_sizes,
            value_counts: self.value_counts,
            null_value_counts: self.null_value_counts,
            nan_value_counts: self.nan_value_counts,
            lower_bounds: self.lower_bounds.map(decode_hex_map),
            upper_bounds: self.upper_bounds.map(decode_hex_map),
            key_metadata: self.key_metadata.map(|hex| decode_hex(&hex)),
            split_offsets: self.split_offsets,
            equality_ids: self.equality_ids,
            sort_order_id: self.sort_order_id,
            first_row_id: self.first_row_id,
            referenced_data_file: self.referenced_data_file,
            content_offset: self.content_offset,
            content_size_in_bytes: self.content_size_in_bytes,
        }
    }
}

impl FileRow {
    /// Collapse the ONE KNOWN, content-identical representation divergence (see the module docs) to a
    /// canonical form so the bulk equality compares CONTENT: an absent metric/bound map (`None` on Java,
    /// `Some(empty)` on Rust) normalized to `None`. `file_format` is NOT canonicalized — Rust now upper-cases
    /// the rendered value to match Java's enum name exactly, so it is compared verbatim.
    fn canonical(mut self) -> FileRow {
        self.column_sizes = none_if_empty_long(self.column_sizes);
        self.value_counts = none_if_empty_long(self.value_counts);
        self.null_value_counts = none_if_empty_long(self.null_value_counts);
        self.nan_value_counts = none_if_empty_long(self.nan_value_counts);
        self.lower_bounds = none_if_empty_bytes(self.lower_bounds);
        self.upper_bounds = none_if_empty_bytes(self.upper_bounds);
        self
    }
}

/// `Some(empty)` → `None` (an absent count map; the empty-vs-null divergence).
fn none_if_empty_long(map: Option<HashMap<i32, i64>>) -> Option<HashMap<i32, i64>> {
    map.filter(|m| !m.is_empty())
}

/// `Some(empty)` → `None` (an absent bound map; the empty-vs-null divergence).
fn none_if_empty_bytes(map: Option<HashMap<i32, Vec<u8>>>) -> Option<HashMap<i32, Vec<u8>>> {
    map.filter(|m| !m.is_empty())
}

/// Decode a {field_id: hex} map into {field_id: raw bytes}.
fn decode_hex_map(map: HashMap<i32, String>) -> HashMap<i32, Vec<u8>> {
    map.into_iter().map(|(k, v)| (k, decode_hex(&v))).collect()
}

/// Decode a lowercase hex string into raw bytes (Java emitted `String.format("%02x", b & 0xff)` per byte).
fn decode_hex(hex: &str) -> Vec<u8> {
    assert!(
        hex.len().is_multiple_of(2),
        "hex string must have even length: {hex}"
    );
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .unwrap_or_else(|error| panic!("invalid hex byte in {hex}: {error}"))
        })
        .collect()
}

// ===========================================================================================
// Fixture loading + Table construction.
// ===========================================================================================

/// The temp dir the Java oracle wrote the table + JSON rows into. `None` when the env var is unset.
fn manifest_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_MANIFEST_DIR").map(PathBuf::from)
}

/// Load + parse one Java JSON fixture from the temp dir.
fn read_java_rows(dir: &std::path::Path, file_name: &str) -> Vec<JavaFileRow> {
    let path = dir.join(file_name);
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<JavaFileRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Build a `Table` over the Java-written `final.metadata.json`, using a LOCAL-FILESYSTEM `FileIO` so the
/// absolute on-disk manifest-list + manifest paths the commits wrote resolve directly.
fn load_table(dir: &std::path::Path) -> Table {
    let metadata_path = dir.join("table/metadata/final.metadata.json");
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(
            TableIdent::from_strs(["interop", "inspection_manifests"]).expect("valid identifier"),
        )
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from Java-written final.metadata.json")
}

// ===========================================================================================
// Arrow column extraction — a files-table batch into the comparable [`FileRow`]s.
// ===========================================================================================

/// A by-name column source: a `files`-family Arrow batch (the A1 `files` scans) OR the nested `data_file`
/// STRUCT inside an `entries` row (A2). Both `RecordBatch` and `StructArray` expose an identically-typed
/// `column_by_name`, so the [`FileRow`] extraction below works against either — A1's `files` rows are read
/// straight from the batch; A2's `entries.data_file` rows are read from the nested struct WITHOUT
/// duplicating the (large) DataFile-projection extraction.
trait ColumnSource {
    fn column(&self, name: &str) -> Option<&ArrayRef>;
    fn rows(&self) -> usize;
}

impl ColumnSource for RecordBatch {
    fn column(&self, name: &str) -> Option<&ArrayRef> {
        self.column_by_name(name)
    }
    fn rows(&self) -> usize {
        self.num_rows()
    }
}

impl ColumnSource for StructArray {
    fn column(&self, name: &str) -> Option<&ArrayRef> {
        self.column_by_name(name)
    }
    fn rows(&self) -> usize {
        self.len()
    }
}

/// Extract a files-family column source (a `files` batch OR an `entries.data_file` struct) into
/// [`FileRow`]s by COLUMN NAME (never by position), covering every column except the deferred
/// `readable_metrics`.
fn extract_rust_rows(batch: &dyn ColumnSource) -> Vec<FileRow> {
    let content = primitive::<Int32Type>(batch, "content");
    let file_path = string_col(batch, "file_path");
    let file_format = string_col(batch, "file_format");
    let spec_id = primitive::<Int32Type>(batch, "spec_id");
    let record_count = primitive::<Int64Type>(batch, "record_count");
    let file_size = primitive::<Int64Type>(batch, "file_size_in_bytes");
    let sort_order_id = primitive::<Int32Type>(batch, "sort_order_id");
    let first_row_id = primitive::<Int64Type>(batch, "first_row_id");
    let content_offset = primitive::<Int64Type>(batch, "content_offset");
    let content_size = primitive::<Int64Type>(batch, "content_size_in_bytes");
    let referenced_data_file = string_col(batch, "referenced_data_file");

    // The `partition` struct's single `category` (Utf8) sub-field.
    let partition = batch.column("partition").expect("partition").as_struct();
    let partition_category = partition
        .column_by_name("category")
        .map(|c| c.as_string::<i32>());

    (0..batch.rows())
        .map(|i| FileRow {
            content: content.value(i),
            file_path: file_path.value(i).to_string(),
            file_format: file_format.value(i).to_string(),
            spec_id: spec_id.value(i),
            partition_category: partition_category.and_then(|c| {
                if c.is_null(i) {
                    None
                } else {
                    Some(c.value(i).to_string())
                }
            }),
            record_count: record_count.value(i),
            file_size_in_bytes: file_size.value(i),
            column_sizes: long_map(batch, "column_sizes", i),
            value_counts: long_map(batch, "value_counts", i),
            null_value_counts: long_map(batch, "null_value_counts", i),
            nan_value_counts: long_map(batch, "nan_value_counts", i),
            lower_bounds: bytes_map(batch, "lower_bounds", i),
            upper_bounds: bytes_map(batch, "upper_bounds", i),
            key_metadata: opt_binary(batch, "key_metadata", i),
            split_offsets: long_list(batch, "split_offsets", i),
            equality_ids: int_list(batch, "equality_ids", i),
            sort_order_id: opt_i32(sort_order_id, i),
            first_row_id: opt_i64(first_row_id, i),
            referenced_data_file: opt_str(referenced_data_file, i),
            content_offset: opt_i64(content_offset, i),
            content_size_in_bytes: opt_i64(content_size, i),
        })
        .collect()
}

fn primitive<'a, T: arrow_array::types::ArrowPrimitiveType>(
    batch: &'a dyn ColumnSource,
    name: &str,
) -> &'a arrow_array::PrimitiveArray<T> {
    batch
        .column(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_primitive::<T>()
}

fn string_col<'a>(batch: &'a dyn ColumnSource, name: &str) -> &'a StringArray {
    batch
        .column(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_string::<i32>()
}

fn opt_i32(arr: &arrow_array::PrimitiveArray<Int32Type>, i: usize) -> Option<i32> {
    if arr.is_null(i) {
        None
    } else {
        Some(arr.value(i))
    }
}

fn opt_i64(arr: &arrow_array::PrimitiveArray<Int64Type>, i: usize) -> Option<i64> {
    if arr.is_null(i) {
        None
    } else {
        Some(arr.value(i))
    }
}

fn opt_str(arr: &StringArray, i: usize) -> Option<String> {
    if arr.is_null(i) {
        None
    } else {
        Some(arr.value(i).to_string())
    }
}

/// A `map<int, long>` metrics column at row `i` → `Some(HashMap)` when present, `None` when the map cell is
/// NULL (matching Java's `null` for an absent metric map).
fn long_map(batch: &dyn ColumnSource, name: &str, i: usize) -> Option<HashMap<i32, i64>> {
    let map = batch
        .column(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_map();
    if map.is_null(i) {
        return None;
    }
    let entries = map.value(i);
    let keys = entries.column(0).as_primitive::<Int32Type>();
    let values = entries.column(1).as_primitive::<Int64Type>();
    let mut out = HashMap::new();
    for e in 0..entries.len() {
        out.insert(keys.value(e), values.value(e));
    }
    Some(out)
}

/// A `map<int, binary>` bound column at row `i` → `Some(HashMap<field_id, raw bytes>)` or `None`.
fn bytes_map(batch: &dyn ColumnSource, name: &str, i: usize) -> Option<HashMap<i32, Vec<u8>>> {
    let map = batch
        .column(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_map();
    if map.is_null(i) {
        return None;
    }
    let entries = map.value(i);
    let keys = entries.column(0).as_primitive::<Int32Type>();
    // Iceberg `binary` maps to Arrow LargeBinary (`schema_to_arrow_schema`), so the bound-map value column
    // is a `LargeBinaryArray` — read it with the i64 offset width.
    let values = entries.column(1).as_binary::<i64>();
    let mut out = HashMap::new();
    for e in 0..entries.len() {
        out.insert(keys.value(e), values.value(e).to_vec());
    }
    Some(out)
}

/// A `list<long>` column at row `i` → `Some(Vec)` or `None`.
fn long_list(batch: &dyn ColumnSource, name: &str, i: usize) -> Option<Vec<i64>> {
    let list = batch
        .column(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_list::<i32>();
    if list.is_null(i) {
        return None;
    }
    let values = list.value(i);
    let values = values.as_primitive::<Int64Type>();
    Some((0..values.len()).map(|e| values.value(e)).collect())
}

/// A `list<int>` column at row `i` → `Some(Vec)` or `None`.
fn int_list(batch: &dyn ColumnSource, name: &str, i: usize) -> Option<Vec<i32>> {
    let list = batch
        .column(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_list::<i32>();
    if list.is_null(i) {
        return None;
    }
    let values = list.value(i);
    let values = values.as_primitive::<Int32Type>();
    Some((0..values.len()).map(|e| values.value(e)).collect())
}

/// An optional binary column (`key_metadata`) at row `i`. The files table stores it as `LargeBinary`.
fn opt_binary(batch: &dyn ColumnSource, name: &str, i: usize) -> Option<Vec<u8>> {
    let arr = batch
        .column(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_binary::<i64>();
    if arr.is_null(i) {
        None
    } else {
        Some(arr.value(i).to_vec())
    }
}

/// Collect a metadata-table scan into the comparable [`FileRow`]s (the files scans emit one batch).
async fn scan_rows(stream: iceberg::scan::ArrowRecordBatchStream) -> Vec<FileRow> {
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect files scan");
    let mut rows = Vec::new();
    for batch in &batches {
        rows.extend(extract_rust_rows(batch));
    }
    rows
}

/// Canonicalize (collapse the one known representation divergence — absent map vs empty) + sort by
/// `file_path` for an order-independent CONTENT comparison.
fn canonical_sorted(rows: Vec<FileRow>) -> Vec<FileRow> {
    let mut rows: Vec<FileRow> = rows.into_iter().map(FileRow::canonical).collect();
    rows.sort_by(|a, b| a.file_path.cmp(&b.file_path));
    rows
}

// ===========================================================================================
// The single env-gated interop test.
// ===========================================================================================

#[tokio::test]
async fn test_files_tables_match_java_rows_from_real_manifests() {
    let Some(dir) = manifest_dir() else {
        println!(
            "skipping interop_inspection_manifests — set ICEBERG_INTEROP_MANIFEST_DIR \
             (run dev/java-interop/run-inspection-manifests.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    // RAW (un-canonicalized) Rust `files` rows — kept so the file_format case (now matching Java) and the
    // one known representation divergence (empty-map-vs-null) can be PINNED below, not silently masked.
    let rust_files_raw = scan_rows(table.inspect().files().scan().await.expect("files scan")).await;

    // -- files: ALL live entries (2 data + 1 delete). -------------------------------------------------
    let rust_files = canonical_sorted(rust_files_raw.clone());
    let java_files = canonical_sorted(
        read_java_rows(&dir, "java_files.json")
            .into_iter()
            .map(JavaFileRow::into_file_row)
            .collect(),
    );

    assert_eq!(
        rust_files.len(),
        3,
        "the `files` table has 2 data files + 1 position-delete file"
    );
    assert_eq!(
        rust_files, java_files,
        "Rust `files` rows must equal Java's FilesTable rows field-for-field (content, paths, partition, \
         file_format, counts, the metric + bound maps, the list + V2 delete columns) — readable_metrics \
         excluded, the one known representation divergence (absent map vs empty) canonicalized"
    );

    // -- data_files: the 2 DATA files, the delete EXCLUDED. -------------------------------------------
    let rust_data = canonical_sorted(
        scan_rows(
            table
                .inspect()
                .data_files()
                .scan()
                .await
                .expect("data_files scan"),
        )
        .await,
    );
    let java_data = canonical_sorted(
        read_java_rows(&dir, "java_data_files.json")
            .into_iter()
            .map(JavaFileRow::into_file_row)
            .collect(),
    );
    assert_eq!(
        rust_data.len(),
        2,
        "the `data_files` table has the 2 data files"
    );
    assert_eq!(
        rust_data, java_data,
        "Rust `data_files` rows must equal Java's DataFilesTable rows field-for-field"
    );

    // -- delete_files: the 1 DELETE file, the data files EXCLUDED. ------------------------------------
    let rust_deletes = canonical_sorted(
        scan_rows(
            table
                .inspect()
                .delete_files()
                .scan()
                .await
                .expect("delete_files scan"),
        )
        .await,
    );
    let java_deletes = canonical_sorted(
        read_java_rows(&dir, "java_delete_files.json")
            .into_iter()
            .map(JavaFileRow::into_file_row)
            .collect(),
    );
    assert_eq!(
        rust_deletes.len(),
        1,
        "the `delete_files` table has the 1 position-delete file"
    );
    assert_eq!(
        rust_deletes, java_deletes,
        "Rust `delete_files` rows must equal Java's DeleteFilesTable rows field-for-field"
    );

    // ============================================================================================
    // Pin the RAW Rust rows so behavior cannot drift unnoticed:
    //   1. file_format renders UPPERCASE in Rust (matching Java's `FileFormat` enum name) — the inspection
    //      projection upper-cases the lowercase on-disk string. Compared verbatim in the bulk equality.
    //   2. an absent metric map projects to an EMPTY map in Rust (Java emits null) — content-identical; the
    //      ONE remaining divergence, surfaced + reported, NOT masked (the bulk equality canonicalizes it).
    // ============================================================================================
    let raw_data_file = rust_files_raw
        .iter()
        .find(|r| r.content == 0)
        .expect("a raw data-file row");
    assert_eq!(
        raw_data_file.file_format, "PARQUET",
        "Rust renders file_format UPPERCASE, matching Java's `FileFormat` enum name"
    );
    let raw_delete_file = rust_files_raw
        .iter()
        .find(|r| r.content == 1)
        .expect("a raw delete-file row");
    assert_eq!(
        raw_delete_file.column_sizes,
        Some(HashMap::new()),
        "KNOWN DIVERGENCE pin: an absent metric map projects to an EMPTY map in Rust (Java emits null)"
    );

    // ============================================================================================
    // Focused, named assertions — a single regressed column / filter is pinpointed here.
    // ============================================================================================

    // The content FILTER: `data_files` excludes the delete file; `delete_files` excludes the data files.
    let delete_path = &rust_deletes[0].file_path;
    assert!(
        !rust_data.iter().any(|r| &r.file_path == delete_path),
        "`data_files` must NOT include the delete file (content filter)"
    );
    let data_paths: Vec<&String> = rust_data.iter().map(|r| &r.file_path).collect();
    assert!(
        !data_paths
            .iter()
            .any(|p| rust_deletes.iter().any(|d| &&d.file_path == p)),
        "`delete_files` must NOT include any data file (content filter)"
    );

    // The `content` column: 0 for every data file, 1 for the position-delete.
    assert!(
        rust_data.iter().all(|r| r.content == 0),
        "every data file must report content == 0"
    );
    assert_eq!(
        rust_deletes[0].content, 1,
        "the position-delete file must report content == 1"
    );

    // record_count / file_size match the committed DataFiles (a=3 rows / 1100 B, b=2 rows / 900 B,
    // delete=1 row / 150 B), and the partition tuple is the committed `category`.
    let by_leaf: HashMap<String, &FileRow> =
        rust_files.iter().map(|r| (leaf(&r.file_path), r)).collect();
    let file_a = by_leaf["00000-a.parquet"];
    let file_b = by_leaf["00000-b.parquet"];
    let delete_a = by_leaf["00000-a-deletes.parquet"];
    assert_eq!((file_a.record_count, file_a.file_size_in_bytes), (3, 1100));
    assert_eq!((file_b.record_count, file_b.file_size_in_bytes), (2, 900));
    assert_eq!(
        (delete_a.record_count, delete_a.file_size_in_bytes),
        (1, 150)
    );
    assert_eq!(file_a.partition_category.as_deref(), Some("a"));
    assert_eq!(file_b.partition_category.as_deref(), Some("b"));
    assert_eq!(delete_a.partition_category.as_deref(), Some("a"));

    // The category=a data file's lower/upper bound bytes for `id` (field id 1, a long) DECODE to the
    // committed values 1 (lower) and 3 (upper) — 8-byte little-endian. This pins that the on-disk bound
    // bytes survive the round-trip byte-for-byte (the same check the Java hex encodes).
    let lower = file_a
        .lower_bounds
        .as_ref()
        .expect("category=a data file carries lower bounds");
    let upper = file_a
        .upper_bounds
        .as_ref()
        .expect("category=a data file carries upper bounds");
    assert_eq!(
        i64::from_le_bytes(lower[&1].clone().try_into().expect("8-byte long bound")),
        1,
        "category=a id lower bound decodes to the committed long 1"
    );
    assert_eq!(
        i64::from_le_bytes(upper[&1].clone().try_into().expect("8-byte long bound")),
        3,
        "category=a id upper bound decodes to the committed long 3"
    );

    // The delete file carries NO metrics maps (built without `.withMetrics`), so they are all NULL/None.
    assert_eq!(delete_a.column_sizes, None);
    assert_eq!(delete_a.lower_bounds, None);

    println!(
        "interop_inspection_manifests OK — files=3, data_files=2, delete_files=1 rows matched Java \
         field-for-field (readable_metrics deferred)"
    );
}

/// The trailing path segment (e.g. `00000-a.parquet`) of a file path.
fn leaf(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

// ===========================================================================================
// A2 — the `entries` / `manifests` / `partitions` manifest-reading inspection tables.
//
// Builds DIRECTLY on the A1 harness above (reusing `manifest_dir()`, the `FileRow` model + its `canonical`
// representation-divergence collapse, the `ColumnSource`-based DataFile extraction, the hex decode, and
// `FileIO::new_with_fs()`). The Java oracle's `generate-inspection-manifests` mode now ALSO writes a
// richer V2 table to `<dir>/table_a2` (A1's `<dir>/table` is untouched) — partition by identity(category)
// with snapshots {append A,B,C,D; row-delta +pos-delete(cat=a); delete B} — and emits
// `java_entries.json` / `java_manifests.json` / `java_partitions.json` (the rows of Java's REAL
// ManifestEntriesTable / ManifestsTable / PartitionsTable). These three tests load
// `<dir>/table_a2/metadata/final.metadata.json`, run `inspect().entries()/.manifests()/.partitions()`,
// and assert field-for-field equality vs the Java rows, order-independent.
//
// Same env gate as A1 (`ICEBERG_INTEROP_MANIFEST_DIR`): a clean NO-OP when unset. `readable_metrics` (the
// entries table's trailing top-level virtual struct) is DEFERRED exactly as A1.
// ===========================================================================================

/// Build a `Table` over the Java-written A2 `final.metadata.json` (under `<dir>/table_a2`), local-fs FileIO.
fn load_table_a2(dir: &std::path::Path) -> Table {
    let metadata_path = dir.join("table_a2/metadata/final.metadata.json");
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));
    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(
            TableIdent::from_strs(["interop", "inspection_manifests_a2"])
                .expect("valid identifier"),
        )
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build A2 table from Java-written final.metadata.json")
}

/// Read + parse one Java JSON fixture into `T` (a typed row vector).
fn read_java<T: serde::de::DeserializeOwned>(dir: &std::path::Path, file_name: &str) -> Vec<T> {
    let path = dir.join(file_name);
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<T>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

// ---------------------------------------------------------------------------------------------
// `entries` — one row per manifest entry of the current snapshot's manifests, INCLUDING DELETED
// tombstones (status 2). Columns: status, snapshot_id, sequence_number, file_sequence_number, and the
// NESTED `data_file` struct (the SAME DataFile projection A1 flattened — reused via `extract_rust_rows`).
// ---------------------------------------------------------------------------------------------

/// One Java `ManifestEntriesTable` row: the 4 scalar columns + the nested `data_file` (a [`JavaFileRow`]).
#[derive(Debug, Clone, Deserialize)]
struct JavaEntryRow {
    status: i32,
    snapshot_id: Option<i64>,
    sequence_number: Option<i64>,
    file_sequence_number: Option<i64>,
    data_file: JavaFileRow,
}

/// A normalized, fully-comparable `entries` row — the 4 scalars + the canonicalized nested [`FileRow`].
#[derive(Debug, Clone, PartialEq)]
struct EntryRow {
    status: i32,
    snapshot_id: Option<i64>,
    sequence_number: Option<i64>,
    file_sequence_number: Option<i64>,
    data_file: FileRow,
}

impl JavaEntryRow {
    fn into_entry_row(self) -> EntryRow {
        EntryRow {
            status: self.status,
            snapshot_id: self.snapshot_id,
            sequence_number: self.sequence_number,
            file_sequence_number: self.file_sequence_number,
            // Same canonicalization A1 applies to the files rows (absent metric/bound map None≡empty).
            data_file: self.data_file.into_file_row().canonical(),
        }
    }
}

/// Extract the Rust `entries` batch into [`EntryRow`]s: the 4 scalar columns by name, plus the nested
/// `data_file` STRUCT fed through the A1 [`extract_rust_rows`] (reused via the [`ColumnSource`] trait) and
/// canonicalized. `readable_metrics` (a trailing TOP-LEVEL struct, not nested in `data_file`) is ignored.
fn extract_entry_rows(batch: &RecordBatch) -> Vec<EntryRow> {
    let status = primitive::<Int32Type>(batch, "status");
    let snapshot_id = primitive::<Int64Type>(batch, "snapshot_id");
    let sequence_number = primitive::<Int64Type>(batch, "sequence_number");
    let file_sequence_number = primitive::<Int64Type>(batch, "file_sequence_number");
    let data_file_struct = batch
        .column_by_name("data_file")
        .expect("data_file struct column")
        .as_struct();
    let data_file_rows: Vec<FileRow> = extract_rust_rows(data_file_struct)
        .into_iter()
        .map(FileRow::canonical)
        .collect();

    (0..batch.num_rows())
        .map(|i| EntryRow {
            status: status.value(i),
            snapshot_id: opt_i64(snapshot_id, i),
            sequence_number: opt_i64(sequence_number, i),
            file_sequence_number: opt_i64(file_sequence_number, i),
            data_file: data_file_rows[i].clone(),
        })
        .collect()
}

/// Sort `entries` rows order-independently by `(data_file.file_path, status)`.
fn sorted_entries(mut rows: Vec<EntryRow>) -> Vec<EntryRow> {
    rows.sort_by(|a, b| {
        a.data_file
            .file_path
            .cmp(&b.data_file.file_path)
            .then(a.status.cmp(&b.status))
    });
    rows
}

#[tokio::test]
async fn test_entries_table_matches_java_rows() {
    let Some(dir) = manifest_dir() else {
        println!(
            "skipping test_entries_table_matches_java_rows — set ICEBERG_INTEROP_MANIFEST_DIR \
             (run dev/java-interop/run-inspection-manifests.sh)"
        );
        return;
    };

    let table = load_table_a2(&dir);
    let batches: Vec<RecordBatch> = table
        .inspect()
        .entries()
        .scan()
        .await
        .expect("entries scan")
        .try_collect()
        .await
        .expect("collect entries scan");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_entry_rows(batch));
    }
    let rust_rows = sorted_entries(rust_rows);

    let java_rows = sorted_entries(
        read_java::<JavaEntryRow>(&dir, "java_entries.json")
            .into_iter()
            .map(JavaEntryRow::into_entry_row)
            .collect(),
    );

    assert_eq!(
        rust_rows, java_rows,
        "Rust `entries` rows must equal Java's ManifestEntriesTable rows field-for-field (status, \
         snapshot_id, sequence_number, file_sequence_number, and the nested data_file struct) — \
         readable_metrics deferred, the absent-map-vs-empty divergence canonicalized as in A1"
    );

    // Focused: the headline difference from `files` — a DELETED tombstone (status 2). The A2 table deletes
    // data file B in the last commit, so B appears as a status-2 row that `files` would have excluded.
    let tombstones: Vec<&EntryRow> = rust_rows.iter().filter(|r| r.status == 2).collect();
    assert!(
        !tombstones.is_empty(),
        "the `entries` table MUST surface ≥1 DELETED tombstone (status 2) — the difference from `files`"
    );
    assert!(
        tombstones.iter().all(|r| r.data_file.content == 0),
        "the deleted tombstone is the data file B (content == 0)"
    );

    // The position-delete file is the ADDED (status 1) row; its nested data_file reports content == 1.
    let added_delete: Vec<&EntryRow> = rust_rows
        .iter()
        .filter(|r| r.data_file.content == 1)
        .collect();
    assert_eq!(
        added_delete.len(),
        1,
        "exactly one position-delete entry in the current snapshot's manifests"
    );
    assert_eq!(
        added_delete[0].status, 1,
        "the position-delete file was ADDED (status 1) and never rewritten, so it stays status 1"
    );

    // file_format renders UPPERCASE in the nested struct, matching Java's `FileFormat` enum name.
    assert!(
        rust_rows
            .iter()
            .all(|r| r.data_file.file_format == "PARQUET"),
        "the nested data_file.file_format is UPPERCASE (matching Java)"
    );

    println!(
        "test_entries_table_matches_java_rows OK — {} entries matched Java (≥1 status-2 tombstone)",
        rust_rows.len()
    );
}

// ---------------------------------------------------------------------------------------------
// `manifests` — one row per manifest in the CURRENT snapshot's manifest list. Content-gated counts + the
// partition_summaries list (contains_null / contains_nan / lower_bound STRING / upper_bound STRING).
// ---------------------------------------------------------------------------------------------

/// One Java `ManifestsTable` partition-summary struct (lower/upper bounds are STRINGS in Java).
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct JavaPartitionSummary {
    contains_null: bool,
    contains_nan: Option<bool>,
    lower_bound: Option<String>,
    upper_bound: Option<String>,
}

/// One Java `ManifestsTable` row — every column incl. the six content-gated counts + the summaries list.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct JavaManifestRow {
    content: i32,
    path: String,
    length: i64,
    partition_spec_id: i32,
    added_snapshot_id: i64,
    added_data_files_count: i32,
    existing_data_files_count: i32,
    deleted_data_files_count: i32,
    added_delete_files_count: i32,
    existing_delete_files_count: i32,
    deleted_delete_files_count: i32,
    partition_summaries: Vec<JavaPartitionSummary>,
}

/// A normalized, comparable `manifests` row (Java + Rust both produce one; equality is a single `==`).
#[derive(Debug, Clone, PartialEq)]
struct ManifestRow {
    content: i32,
    path: String,
    length: i64,
    partition_spec_id: i32,
    added_snapshot_id: i64,
    added_data_files_count: i32,
    existing_data_files_count: i32,
    deleted_data_files_count: i32,
    added_delete_files_count: i32,
    existing_delete_files_count: i32,
    deleted_delete_files_count: i32,
    partition_summaries: Vec<PartitionSummary>,
}

#[derive(Debug, Clone, PartialEq)]
struct PartitionSummary {
    contains_null: bool,
    contains_nan: Option<bool>,
    lower_bound: Option<String>,
    upper_bound: Option<String>,
}

impl JavaManifestRow {
    fn into_manifest_row(self) -> ManifestRow {
        ManifestRow {
            content: self.content,
            path: self.path,
            length: self.length,
            partition_spec_id: self.partition_spec_id,
            added_snapshot_id: self.added_snapshot_id,
            added_data_files_count: self.added_data_files_count,
            existing_data_files_count: self.existing_data_files_count,
            deleted_data_files_count: self.deleted_data_files_count,
            added_delete_files_count: self.added_delete_files_count,
            existing_delete_files_count: self.existing_delete_files_count,
            deleted_delete_files_count: self.deleted_delete_files_count,
            partition_summaries: self
                .partition_summaries
                .into_iter()
                .map(|s| PartitionSummary {
                    contains_null: s.contains_null,
                    contains_nan: s.contains_nan,
                    lower_bound: s.lower_bound,
                    upper_bound: s.upper_bound,
                })
                .collect(),
        }
    }
}

/// Extract the Rust `manifests` batch into [`ManifestRow`]s by COLUMN NAME, including the nested
/// `partition_summaries` list<struct>.
fn extract_manifest_rows(batch: &RecordBatch) -> Vec<ManifestRow> {
    let content = primitive::<Int32Type>(batch, "content");
    let path = string_col(batch, "path");
    let length = primitive::<Int64Type>(batch, "length");
    let partition_spec_id = primitive::<Int32Type>(batch, "partition_spec_id");
    let added_snapshot_id = primitive::<Int64Type>(batch, "added_snapshot_id");
    let added_data = primitive::<Int32Type>(batch, "added_data_files_count");
    let existing_data = primitive::<Int32Type>(batch, "existing_data_files_count");
    let deleted_data = primitive::<Int32Type>(batch, "deleted_data_files_count");
    let added_delete = primitive::<Int32Type>(batch, "added_delete_files_count");
    let existing_delete = primitive::<Int32Type>(batch, "existing_delete_files_count");
    let deleted_delete = primitive::<Int32Type>(batch, "deleted_delete_files_count");
    let summaries = batch
        .column_by_name("partition_summaries")
        .expect("partition_summaries")
        .as_list::<i32>();

    (0..batch.num_rows())
        .map(|i| ManifestRow {
            content: content.value(i),
            path: path.value(i).to_string(),
            length: length.value(i),
            partition_spec_id: partition_spec_id.value(i),
            added_snapshot_id: added_snapshot_id.value(i),
            added_data_files_count: added_data.value(i),
            existing_data_files_count: existing_data.value(i),
            deleted_data_files_count: deleted_data.value(i),
            added_delete_files_count: added_delete.value(i),
            existing_delete_files_count: existing_delete.value(i),
            deleted_delete_files_count: deleted_delete.value(i),
            partition_summaries: extract_partition_summaries(&summaries.value(i)),
        })
        .collect()
}

/// Extract one row's `partition_summaries` list element (a struct array) into [`PartitionSummary`]s.
fn extract_partition_summaries(list_values: &ArrayRef) -> Vec<PartitionSummary> {
    let structs = list_values.as_struct();
    let contains_null = structs
        .column_by_name("contains_null")
        .expect("contains_null")
        .as_boolean();
    let contains_nan = structs
        .column_by_name("contains_nan")
        .expect("contains_nan")
        .as_boolean();
    let lower = structs
        .column_by_name("lower_bound")
        .expect("lower_bound")
        .as_string::<i32>();
    let upper = structs
        .column_by_name("upper_bound")
        .expect("upper_bound")
        .as_string::<i32>();
    (0..structs.len())
        .map(|i| PartitionSummary {
            contains_null: contains_null.value(i),
            contains_nan: if contains_nan.is_null(i) {
                None
            } else {
                Some(contains_nan.value(i))
            },
            lower_bound: if lower.is_null(i) {
                None
            } else {
                Some(lower.value(i).to_string())
            },
            upper_bound: if upper.is_null(i) {
                None
            } else {
                Some(upper.value(i).to_string())
            },
        })
        .collect()
}

#[tokio::test]
async fn test_manifests_table_matches_java_rows() {
    let Some(dir) = manifest_dir() else {
        println!(
            "skipping test_manifests_table_matches_java_rows — set ICEBERG_INTEROP_MANIFEST_DIR \
             (run dev/java-interop/run-inspection-manifests.sh)"
        );
        return;
    };

    let table = load_table_a2(&dir);
    let batches: Vec<RecordBatch> = table
        .inspect()
        .manifests()
        .scan()
        .await
        .expect("manifests scan")
        .try_collect()
        .await
        .expect("collect manifests scan");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_manifest_rows(batch));
    }
    rust_rows.sort_by(|a, b| a.path.cmp(&b.path));

    let mut java_rows: Vec<ManifestRow> = read_java::<JavaManifestRow>(&dir, "java_manifests.json")
        .into_iter()
        .map(JavaManifestRow::into_manifest_row)
        .collect();
    java_rows.sort_by(|a, b| a.path.cmp(&b.path));

    assert_eq!(
        rust_rows, java_rows,
        "Rust `manifests` rows must equal Java's ManifestsTable rows field-for-field (content, path, \
         length, spec id, added_snapshot_id, the six content-gated counts, and partition_summaries)"
    );

    // Focused: content gating. A DATA manifest (content == 0) has ZERO delete-file counts; a DELETE
    // manifest (content == 1) has ZERO data-file counts.
    let data_manifests: Vec<&ManifestRow> = rust_rows.iter().filter(|r| r.content == 0).collect();
    let delete_manifests: Vec<&ManifestRow> = rust_rows.iter().filter(|r| r.content == 1).collect();
    assert!(
        !data_manifests.is_empty(),
        "≥1 DATA manifest in the current snapshot's manifest list"
    );
    assert!(
        !delete_manifests.is_empty(),
        "≥1 DELETE manifest in the current snapshot's manifest list"
    );
    for m in &data_manifests {
        assert_eq!(
            (
                m.added_delete_files_count,
                m.existing_delete_files_count,
                m.deleted_delete_files_count
            ),
            (0, 0, 0),
            "a DATA manifest carries ZERO delete-file counts (content gating)"
        );
    }
    for m in &delete_manifests {
        assert_eq!(
            (
                m.added_data_files_count,
                m.existing_data_files_count,
                m.deleted_data_files_count
            ),
            (0, 0, 0),
            "a DELETE manifest carries ZERO data-file counts (content gating)"
        );
    }

    // partition_summaries are non-empty (the spec is partitioned by identity(category)).
    assert!(
        rust_rows.iter().all(|m| !m.partition_summaries.is_empty()),
        "every manifest's partition_summaries is non-empty (partitioned spec)"
    );

    println!(
        "test_manifests_table_matches_java_rows OK — {} manifests matched Java (content-gated counts + \
         non-empty summaries)",
        rust_rows.len()
    );
}

// ---------------------------------------------------------------------------------------------
// `partitions` — one row per partition value over the current snapshot's LIVE entries. The partition
// struct + spec_id + record/file/size rollups + the four delete-count columns + last_updated_at (µs) +
// last_updated_snapshot_id.
// ---------------------------------------------------------------------------------------------

/// One Java `PartitionsTable` row. The `partition` struct reuses the A1 [`JavaPartition`] (single
/// `category`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct JavaPartitionRow {
    partition: JavaPartition,
    spec_id: i32,
    record_count: i64,
    file_count: i32,
    total_data_file_size_in_bytes: i64,
    position_delete_record_count: i64,
    position_delete_file_count: i32,
    equality_delete_record_count: i64,
    equality_delete_file_count: i32,
    last_updated_at: Option<i64>,
    last_updated_snapshot_id: Option<i64>,
}

/// A normalized, comparable `partitions` row.
#[derive(Debug, Clone, PartialEq)]
struct PartitionRow {
    partition_category: Option<String>,
    spec_id: i32,
    record_count: i64,
    file_count: i32,
    total_data_file_size_in_bytes: i64,
    position_delete_record_count: i64,
    position_delete_file_count: i32,
    equality_delete_record_count: i64,
    equality_delete_file_count: i32,
    last_updated_at: Option<i64>,
    last_updated_snapshot_id: Option<i64>,
}

impl JavaPartitionRow {
    fn into_partition_row(self) -> PartitionRow {
        PartitionRow {
            partition_category: self.partition.category,
            spec_id: self.spec_id,
            record_count: self.record_count,
            file_count: self.file_count,
            total_data_file_size_in_bytes: self.total_data_file_size_in_bytes,
            position_delete_record_count: self.position_delete_record_count,
            position_delete_file_count: self.position_delete_file_count,
            equality_delete_record_count: self.equality_delete_record_count,
            equality_delete_file_count: self.equality_delete_file_count,
            last_updated_at: self.last_updated_at,
            last_updated_snapshot_id: self.last_updated_snapshot_id,
        }
    }
}

/// Extract the Rust `partitions` batch into [`PartitionRow`]s by COLUMN NAME, incl. the `partition` struct
/// (`category` sub-field) and `last_updated_at` (timestamptz µs).
fn extract_partition_rows(batch: &RecordBatch) -> Vec<PartitionRow> {
    let partition = batch
        .column_by_name("partition")
        .expect("partition")
        .as_struct();
    let partition_category = partition
        .column_by_name("category")
        .map(|c| c.as_string::<i32>());
    let spec_id = primitive::<Int32Type>(batch, "spec_id");
    let record_count = primitive::<Int64Type>(batch, "record_count");
    let file_count = primitive::<Int32Type>(batch, "file_count");
    let total_size = primitive::<Int64Type>(batch, "total_data_file_size_in_bytes");
    let pos_del_records = primitive::<Int64Type>(batch, "position_delete_record_count");
    let pos_del_files = primitive::<Int32Type>(batch, "position_delete_file_count");
    let eq_del_records = primitive::<Int64Type>(batch, "equality_delete_record_count");
    let eq_del_files = primitive::<Int32Type>(batch, "equality_delete_file_count");
    let last_updated_at = primitive::<TimestampMicrosecondType>(batch, "last_updated_at");
    let last_updated_snapshot_id = primitive::<Int64Type>(batch, "last_updated_snapshot_id");

    (0..batch.num_rows())
        .map(|i| PartitionRow {
            partition_category: partition_category.and_then(|c| {
                if c.is_null(i) {
                    None
                } else {
                    Some(c.value(i).to_string())
                }
            }),
            spec_id: spec_id.value(i),
            record_count: record_count.value(i),
            file_count: file_count.value(i),
            total_data_file_size_in_bytes: total_size.value(i),
            position_delete_record_count: pos_del_records.value(i),
            position_delete_file_count: pos_del_files.value(i),
            equality_delete_record_count: eq_del_records.value(i),
            equality_delete_file_count: eq_del_files.value(i),
            last_updated_at: if last_updated_at.is_null(i) {
                None
            } else {
                Some(last_updated_at.value(i))
            },
            last_updated_snapshot_id: opt_i64(last_updated_snapshot_id, i),
        })
        .collect()
}

#[tokio::test]
async fn test_partitions_table_matches_java_rows() {
    let Some(dir) = manifest_dir() else {
        println!(
            "skipping test_partitions_table_matches_java_rows — set ICEBERG_INTEROP_MANIFEST_DIR \
             (run dev/java-interop/run-inspection-manifests.sh)"
        );
        return;
    };

    let table = load_table_a2(&dir);
    let batches: Vec<RecordBatch> = table
        .inspect()
        .partitions()
        .scan()
        .await
        .expect("partitions scan")
        .try_collect()
        .await
        .expect("collect partitions scan");
    let mut rust_rows = Vec::new();
    for batch in &batches {
        rust_rows.extend(extract_partition_rows(batch));
    }
    rust_rows.sort_by(|a, b| a.partition_category.cmp(&b.partition_category));

    let mut java_rows: Vec<PartitionRow> =
        read_java::<JavaPartitionRow>(&dir, "java_partitions.json")
            .into_iter()
            .map(JavaPartitionRow::into_partition_row)
            .collect();
    java_rows.sort_by(|a, b| a.partition_category.cmp(&b.partition_category));

    assert_eq!(
        rust_rows, java_rows,
        "Rust `partitions` rows must equal Java's PartitionsTable rows field-for-field (the partition \
         struct, spec_id, record/file/size rollups, the four delete-count columns, last_updated_at µs, \
         last_updated_snapshot_id)"
    );

    // Focused: ≥2 partition rows; the cat=a partition received a position-delete so its delete counts are
    // non-zero, and its total_data_file_size_in_bytes counts ONLY the data files (not the delete file).
    assert!(
        rust_rows.len() >= 2,
        "≥2 partition rows (the A2 table partitions cat=a and cat=b)"
    );
    let cat_a = rust_rows
        .iter()
        .find(|r| r.partition_category.as_deref() == Some("a"))
        .expect("a cat=a partition row");
    assert!(
        cat_a.position_delete_record_count > 0 && cat_a.position_delete_file_count > 0,
        "the cat=a partition received a position-delete: its position_delete_* counts are non-zero"
    );
    // cat=a has data files A (1100) + C (1300) = 2400; the 150-byte delete file is NOT counted here.
    assert_eq!(
        cat_a.total_data_file_size_in_bytes, 2400,
        "total_data_file_size_in_bytes counts ONLY data-file sizes (A 1100 + C 1300), not the delete file"
    );
    assert_eq!(
        cat_a.file_count, 2,
        "cat=a file_count is the 2 DATA files (A, C); deletes are counted in the delete columns"
    );

    let cat_b = rust_rows
        .iter()
        .find(|r| r.partition_category.as_deref() == Some("b"))
        .expect("a cat=b partition row");
    assert_eq!(
        cat_b.position_delete_record_count, 0,
        "the cat=b partition received no delete (only the surviving data file D)"
    );

    println!(
        "test_partitions_table_matches_java_rows OK — {} partitions matched Java (cat=a delete counts \
         non-zero)",
        rust_rows.len()
    );
}
