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

//! Java interop for `RewriteManifests` DATA-LEVEL (GAP_MATRIX row 100).
//!
//! The Java oracle (`generate-interop-rewrite-manifests`) writes an UNPARTITIONED V2 table under
//! `<dir>/table` with THREE REAL parquet data files (ids 10/20, 30/40, 50/60) each committed in a
//! SEPARATE `newAppend` — 3 data manifests at baseline. It emits `java_rows.json` (all 6 live rows
//! sorted by id) + `final.metadata.json`.
//!
//! This test (gated on `ICEBERG_INTEROP_RWM_DIR`):
//!
//! 1. Builds a fresh table in a `MemoryCatalogBuilder` over a local-FS `FileIO` at the SAME
//!    location as Java's table (`<dir>/table`), so the catalog's commit path writes new manifests
//!    there. Creates metadata-only `DataFile` entries that point to Java's real parquet paths on
//!    disk (same file-path convention as `interop_expire.rs`), and commits THREE separate
//!    `fast_append`s — mirroring Java's 3-commit chain — so the catalog-managed table starts with
//!    exactly 3 data manifests.
//! 2. Runs the production `rewrite_manifests().cluster_by(|_| "all".to_string())` — a constant
//!    cluster key forces all 3 data-manifest entries into ONE output manifest.
//! 3. Commits and verifies:
//!    - data-manifest count == 1 (re-clustering fired; started from 3)
//!    - scans the rewritten table → live rows == `java_rows.json` (data preserved, reads real parquet)
//! 4. Writes the rewritten table metadata to
//!    `<dir>/rust_rewritten/metadata/final.metadata.json` for the Java verify step
//!    (`verify-interop-rewrite-manifests`) to read back via `IcebergGenerics`.
//!
//! Without `ICEBERG_INTEROP_RWM_DIR` the test returns early — a clean NO-OP so the offline
//! `cargo test` gate needs no Java/Maven.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use futures::TryStreamExt;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, ManifestContentType,
    NestedField, PrimitiveType, Schema, SortOrder, Struct, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use serde::Deserialize;

// ===========================================================================================
// Row model — `{id, data}` matching the Java oracle's java_rows.json format.
// ===========================================================================================

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ScanRow {
    id: i64,
    data: Option<String>,
}

fn sorted_by_id(mut rows: Vec<ScanRow>) -> Vec<ScanRow> {
    rows.sort_by_key(|r| r.id);
    rows
}

// ===========================================================================================
// Env gate.
// ===========================================================================================

fn rwm_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_RWM_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// ===========================================================================================
// Helpers.
// ===========================================================================================

/// Schema: `{1 id long required, 2 data string optional}` — matches `RewriteManifestsOracle.SCHEMA`.
fn rwm_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, data string} schema")
}

/// Build a metadata-only `DataFile` pointing to the given `path` on local disk.
/// The file MUST exist (Java wrote it); we read its true size so the Parquet reader can
/// seek the footer correctly (`FileMetadata::size` is used by `ArrowFileReader`).
/// `record_count` is the number of rows Java wrote to this parquet file.
fn metadata_only_data_file(path: &str, record_count: u64) -> DataFile {
    let file_size_in_bytes = std::fs::metadata(path)
        .unwrap_or_else(|e| panic!("stat {path}: {e}"))
        .len();
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(file_size_in_bytes)
        .record_count(record_count)
        .partition_spec_id(0)
        .partition(Struct::empty()) // unpartitioned
        .build()
        .expect("build metadata-only data file")
}

/// Read `java_rows.json` from `dir` into a sorted `Vec<ScanRow>`.
fn read_java_rows(dir: &Path) -> Vec<ScanRow> {
    let path = dir.join("java_rows.json");
    let json = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let rows: Vec<ScanRow> =
        serde_json::from_str(&json).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    sorted_by_id(rows)
}

/// Extract `(id, data)` rows from one Arrow `RecordBatch`.
fn extract_rows(batch: &RecordBatch) -> Vec<ScanRow> {
    let id = batch
        .column_by_name("id")
        .expect("id column")
        .as_primitive::<Int64Type>();
    let data_col = batch.column_by_name("data").expect("data column");

    (0..batch.num_rows())
        .map(|i| ScanRow {
            id: id.value(i),
            data: string_value(data_col, i),
        })
        .collect()
}

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

/// Count the data manifests in the current snapshot of `table`.
async fn count_data_manifests(table: &Table) -> usize {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
    snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list")
        .entries()
        .iter()
        .filter(|m| m.content == ManifestContentType::Data)
        .count()
}

/// Scan `table` → Arrow, collect all `ScanRow`s sorted by id.
async fn scan_rows(table: &Table) -> Vec<ScanRow> {
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
    sorted_by_id(rows)
}

/// Commit one `fast_append` with `files` and return the updated `Table`.
async fn fast_append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply fast_append");
    tx.commit(catalog).await.expect("commit fast_append")
}

// ===========================================================================================
// The env-gated interop GEN test.
// ===========================================================================================

#[tokio::test]
async fn test_rewrite_manifests_interop_gen() {
    let Some(dir) = rwm_dir() else {
        println!(
            "skipping interop_rewrite_manifests — set ICEBERG_INTEROP_RWM_DIR \
             (run dev/java-interop/run-interop-rewrite-manifests.sh)"
        );
        return;
    };

    // ---- 1. Build a fresh table in the catalog at <dir>/table (the SAME location Java used). ----
    //
    // Java wrote its parquet files under <dir>/table/data/. We build a MemoryCatalog table at
    // the exact same location so our metadata-only DataFile entries reference Java's real parquet
    // paths. When we scan later, the local-FS FileIO resolves those paths directly.

    let warehouse = dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/table");

    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_rwm",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");

    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rwm_table".to_string())
        .location(table_location.clone())
        .schema(rwm_schema())
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table in catalog");

    // ---- 2. Mirror Java's 3 separate newAppend commits via metadata-only DataFiles. ----
    //
    // Java wrote three parquet files under <dir>/table/data/ and committed each in its own
    // newAppend. We build metadata-only DataFile entries pointing to those SAME absolute paths
    // and fast_append each in a separate commit — producing 3 data manifests, just like Java.
    //
    // The DataFile record_count is approximate (not read by rewrite_manifests); we use 2 per
    // file (matching Java's FILE_IDS arrays of 2 entries each).
    //
    // NOTE: The catalog's create_table already wrote a v0-style metadata. MemoryCatalog uses
    // the `<version>-<uuid>.metadata.json` naming convention for subsequent commits, so all 3
    // fast_append commits use the catalog's internal versioning (no name-format conflict).

    let file_a = metadata_only_data_file(
        &format!("{table_location}/data/00000-rwm-file-a.parquet"),
        2,
    );
    let table = fast_append(&catalog, &table, vec![file_a]).await;

    let file_b = metadata_only_data_file(
        &format!("{table_location}/data/00000-rwm-file-b.parquet"),
        2,
    );
    let table = fast_append(&catalog, &table, vec![file_b]).await;

    let file_c = metadata_only_data_file(
        &format!("{table_location}/data/00000-rwm-file-c.parquet"),
        2,
    );
    let table = fast_append(&catalog, &table, vec![file_c]).await;

    // Verify the baseline: 3 separate appends → 3 data manifests.
    let pre_count = count_data_manifests(&table).await;
    assert_eq!(
        pre_count, 3,
        "baseline must have exactly 3 data manifests (one per fast_append, mirroring Java's newAppend chain)"
    );

    // ---- 3. Run rewrite_manifests with a constant cluster_by key (3 → 1 manifest). ----

    let tx = Transaction::new(&table);
    let action = tx.rewrite_manifests().cluster_by(|_| "all".to_string());
    let tx = action.apply(tx).expect("apply rewrite_manifests");
    let rewritten_table = tx.commit(&catalog).await.expect("commit rewrite_manifests");

    // ---- 4a. Assert manifest count dropped 3 → 1 (the re-clustering proof). ----

    let post_count = count_data_manifests(&rewritten_table).await;
    assert_eq!(
        post_count, 1,
        "rewrite_manifests (constant cluster_by) must produce exactly 1 data manifest (was 3)"
    );

    // ---- 4b. Scan the rewritten table and assert live rows == java_rows.json. ----
    //
    // The DataFile entries in the rewritten manifest point to Java's real parquet files on disk.
    // Scanning reads those parquet files via the local-FS FileIO and produces the real row data.

    let rust_rows = scan_rows(&rewritten_table).await;
    let java_rows = read_java_rows(&dir);

    assert_eq!(
        rust_rows.len(),
        6,
        "all 6 live rows must survive rewrite_manifests (manifest re-clustering must not lose data)"
    );
    assert_eq!(
        rust_rows, java_rows,
        "Rust scan of the rewritten table must match Java's IcebergGenerics read (java_rows.json)"
    );

    let live_ids: Vec<i64> = rust_rows.iter().map(|r| r.id).collect();
    assert_eq!(
        live_ids,
        vec![10, 20, 30, 40, 50, 60],
        "live ids after rewrite_manifests must be exactly {{10,20,30,40,50,60}}"
    );

    println!(
        "interop_rewrite_manifests GEN: 3→{post_count} data manifests, \
         6 live rows preserved, rows match java_rows.json"
    );

    // ---- 5. Write the rewritten metadata for the Java verify step. ----

    let rewritten_metadata_dir = dir.join("rust_rewritten").join("metadata");
    fs::create_dir_all(&rewritten_metadata_dir).expect("create rust_rewritten/metadata directory");

    let final_metadata_path = rewritten_metadata_dir
        .join("final.metadata.json")
        .to_string_lossy()
        .to_string();
    rewritten_table
        .metadata()
        .write_to(rewritten_table.file_io(), &final_metadata_path)
        .await
        .expect("write rust_rewritten/metadata/final.metadata.json");

    println!("interop_rewrite_manifests GEN OK — rewritten metadata at {final_metadata_path}");
}
