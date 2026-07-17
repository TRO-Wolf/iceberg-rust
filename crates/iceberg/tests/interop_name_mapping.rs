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

//! NAME-MAPPING scan-wiring interop, DIRECTION 1 ("Rust reads what JAVA wrote") — GAP_MATRIX row R143.
//!
//! The Java oracle (`dev/java-interop`, mode `generate-interop-name-mapping`) writes a V2
//! unpartitioned table `{1: id long, 2: val long}` over a single ID-LESS parquet data file (written by
//! `AvroParquetWriter` so NO Iceberg field ids are stamped — the `add_files` migration shape) whose
//! PHYSICAL column order is REVERSED relative to the schema: physical column 0 is `val`, column 1 is
//! `id`. Rows: `id = [10, 20, 30]`, `val = [100, 200, 300]`. It sets `schema.name-mapping.default` and
//! writes `final.metadata.json` + the ground-truth rows (by construction) to
//! `java_name_mapping_rows.json`.
//!
//! THIS test loads that `final.metadata.json`, runs a Rust `TableScan` → Arrow, and asserts the
//! `(id, val)` rows equal the Java-written ground truth. Because the physical column order is reversed,
//! ONLY a correct field-id-by-name resolution (via the name mapping) yields the right columns; a
//! positional fallback reads them transposed. The `dev/java-interop/run-interop-name-mapping.sh` driver
//! runs this against TWO fixtures: `normal/` (correct mapping ⇒ this test PASSES) and `sabotage/` (the
//! two mapped names SWAPPED ⇒ this same test must go RED, proving the mapping is load-bearing).
//!
//! Without `ICEBERG_INTEROP_NAME_MAPPING_DIR` the test is a clean no-op (the offline `cargo test` suite
//! never invokes Java); the shell script sets that var to flip it into the real comparison.

use std::fs;
use std::path::PathBuf;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::io::FileIO;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;
use serde::Deserialize;

/// One row of the Java ground truth (`java_name_mapping_rows.json`): the two `long` columns.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct NameMappingRow {
    id: i64,
    val: i64,
}

/// The fixture dir the Java oracle wrote (contains `table/metadata/final.metadata.json` +
/// `java_name_mapping_rows.json`). `None` when `ICEBERG_INTEROP_NAME_MAPPING_DIR` is unset — the test
/// is then a clean no-op.
fn name_mapping_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_NAME_MAPPING_DIR").map(PathBuf::from)
}

/// Load + parse the Java ground-truth rows from `<dir>/java_name_mapping_rows.json`.
fn read_java_rows(dir: &std::path::Path) -> Vec<NameMappingRow> {
    let path = dir.join("java_name_mapping_rows.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<Vec<NameMappingRow>>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Build a `Table` over the Java-written `final.metadata.json`, using a local-filesystem `FileIO` so the
/// bare absolute manifest-list / manifest / parquet paths the Java commits wrote resolve directly.
fn load_table(dir: &std::path::Path) -> Table {
    let metadata_path = dir.join("table/metadata/final.metadata.json");
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", "name_mapping"]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from Java-written final.metadata.json")
}

/// Extract the `(id, val)` rows from every scan batch, sorted by `id` for an order-independent compare.
fn scan_rows(batches: &[RecordBatch]) -> Vec<NameMappingRow> {
    let mut rows = Vec::new();
    for batch in batches {
        let id = batch
            .column_by_name("id")
            .expect("id column present")
            .as_primitive::<Int64Type>();
        let val = batch
            .column_by_name("val")
            .expect("val column present")
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            rows.push(NameMappingRow {
                id: id.value(i),
                val: val.value(i),
            });
        }
    }
    rows.sort_by_key(|r| r.id);
    rows
}

#[tokio::test]
async fn test_name_mapping_scan_matches_java() {
    let Some(dir) = name_mapping_dir() else {
        println!(
            "skipping interop_name_mapping — set ICEBERG_INTEROP_NAME_MAPPING_DIR \
             (run dev/java-interop/run-interop-name-mapping.sh)"
        );
        return;
    };

    let table = load_table(&dir);

    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build table scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect scan batches");

    let actual = scan_rows(&batches);
    let mut expected = read_java_rows(&dir);
    expected.sort_by_key(|r| r.id);

    // With the correct name mapping the reversed physical columns resolve by field id to the right
    // Iceberg fields (id -> [10,20,30], val -> [100,200,300]). Against the SABOTAGE fixture (swapped
    // mapping) the columns come back transposed, so this equality FAILS — the sabotage RED the driver
    // requires. A positional fallback would also transpose them, so a pass here proves the mapping is
    // genuinely consulted.
    assert_eq!(
        actual, expected,
        "Rust name-mapped scan rows must equal the Java-written ground truth"
    );
}
