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
//! TWO KNOWN, NON-CORRUPTING REPRESENTATION DIVERGENCES (content-identical; surfaced, not hidden). Both
//! are presentation-only differences in how each library's metadata table RENDERS a column; the underlying
//! on-disk manifest value is identical, so they are NOT production bugs and NOT in scope to "fix" here (a
//! fix would be a spec-type change, out of bounds for an interop test). They are collapsed to a canonical
//! form by [`FileRow::canonical`] for the bulk equality AND pinned RAW by a focused assertion so neither can
//! drift unnoticed:
//!   1. `file_format` CASE. Java's `FilesTable` emits the UPPERCASE `FileFormat` enum name (`PARQUET`,
//!      since `FileFormat` does not override `toString()`); Rust's `inspect` renders `DataFileFormat`'s
//!      lowercase `Display` (`parquet`). The on-disk AVRO stores `PARQUET` either way (Rust reads it via the
//!      case-insensitive `from_str`). Canonicalized by ASCII-lowercasing.
//!   2. ABSENT METRIC/BOUND MAP — empty `{}` vs `null`. Rust's `spec::DataFile` stores the metric/bound maps
//!      as NON-optional `HashMap`, so an absent map projects to an EMPTY map; Java stores `null` and emits
//!      JSON `null`. An empty map and a null map carry identical information (no metrics). Canonicalized by
//!      treating `None` and `Some(empty)` as equal.
//!
//! NO PRODUCTION CHANGE is needed: every OTHER column the Rust `files` family projects matches Java's
//! `FilesTable` row byte-for-byte when both read the same on-disk manifest. (If a column genuinely diverged
//! in CONTENT, the contract is to STOP and report it — never to hide a column. These two are content-equal.)

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, RecordBatch, StringArray};
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
    /// Collapse the two KNOWN, content-identical representation divergences (see the module docs) to a
    /// canonical form so the bulk equality compares CONTENT: lowercase `file_format`, and an absent
    /// metric/bound map (`None` on Java, `Some(empty)` on Rust) normalized to `None`.
    fn canonical(mut self) -> FileRow {
        self.file_format = self.file_format.to_ascii_lowercase();
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

/// Extract a files-table Arrow batch into [`FileRow`]s by COLUMN NAME (never by position), covering every
/// column except the deferred `readable_metrics`.
fn extract_rust_rows(batch: &RecordBatch) -> Vec<FileRow> {
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
    let partition = batch
        .column_by_name("partition")
        .expect("partition")
        .as_struct();
    let partition_category = partition
        .column_by_name("category")
        .map(|c| c.as_string::<i32>());

    (0..batch.num_rows())
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
    batch: &'a RecordBatch,
    name: &str,
) -> &'a arrow_array::PrimitiveArray<T> {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column {name} present"))
        .as_primitive::<T>()
}

fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> &'a StringArray {
    batch
        .column_by_name(name)
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
fn long_map(batch: &RecordBatch, name: &str, i: usize) -> Option<HashMap<i32, i64>> {
    let map = batch
        .column_by_name(name)
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
fn bytes_map(batch: &RecordBatch, name: &str, i: usize) -> Option<HashMap<i32, Vec<u8>>> {
    let map = batch
        .column_by_name(name)
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
fn long_list(batch: &RecordBatch, name: &str, i: usize) -> Option<Vec<i64>> {
    let list = batch
        .column_by_name(name)
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
fn int_list(batch: &RecordBatch, name: &str, i: usize) -> Option<Vec<i32>> {
    let list = batch
        .column_by_name(name)
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
fn opt_binary(batch: &RecordBatch, name: &str, i: usize) -> Option<Vec<u8>> {
    let arr = batch
        .column_by_name(name)
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

/// Canonicalize (collapse the two known representation divergences) + sort by `file_path` for an
/// order-independent CONTENT comparison.
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

    // RAW (un-canonicalized) Rust `files` rows — kept so the two known representation divergences
    // (file_format case + empty-map-vs-null) can be PINNED below, not silently masked.
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
         counts, the metric + bound maps, the list + V2 delete columns) — readable_metrics excluded, the \
         two known representation divergences canonicalized"
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
    // Pin the TWO known representation divergences on the RAW Rust rows, so neither can drift unnoticed:
    //   1. file_format renders LOWERCASE in Rust (Java emits the uppercase enum name) — content-identical.
    //   2. an absent metric map projects to an EMPTY map in Rust (Java emits null) — content-identical.
    // (These are surfaced + reported, NOT masked; the bulk equality above canonicalizes them.)
    // ============================================================================================
    let raw_data_file = rust_files_raw
        .iter()
        .find(|r| r.content == 0)
        .expect("a raw data-file row");
    assert_eq!(
        raw_data_file.file_format, "parquet",
        "KNOWN DIVERGENCE pin: Rust renders file_format lowercase (Java emits PARQUET)"
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
