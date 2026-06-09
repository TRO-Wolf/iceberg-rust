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

//! Java interop tests for the pure-metadata inspection tables `snapshots` and `refs`.
//!
//! This is the byte-/field-level "read a table Java wrote" evidence that flips the GAP_MATRIX inspection
//! rows (snapshots + refs) toward ✅. It mirrors the proven `interop_manage_snapshots.rs` harness, but
//! because the inspection metadata tables are **READ-ONLY** (they project a base `TableMetadata` into rows;
//! there is nothing to write back), there is only **ONE direction**:
//!
//! - **Direction 1 (Rust reproduces Java's projection).** The Java oracle's `generate-inspection` mode
//!   materializes the ACTUAL rows of Java `iceberg-core`'s own `SnapshotsTable` / `RefsTable` — obtained
//!   through `MetadataTableUtils.createMetadataTableInstance` + `task.asDataTask().rows()` on a RE-PARSED
//!   base (the same on-disk bytes the Rust reader consumes) — into `java_snapshots.json` /
//!   `java_refs.json`. This test loads the same Java-written `base.metadata.json`, runs
//!   `table.inspect().snapshots().scan()` / `.refs().scan()`, extracts every Arrow column into comparable
//!   Rust values, and asserts field-for-field equality against the Java rows for ALL columns. It runs in
//!   the normal offline suite — no Java / Docker needed; it reads the committed fixtures.
//!
//! There is no Direction 2: a metadata table cannot be "written", so Java has nothing of Rust's to read
//! back. The equality of the projected rows IS the round-trip proof.
//!
//! CORRECTNESS NOTE (the summary map). Java `SnapshotsTable.snapshotToRow` puts `snap.summary()` into the
//! summary MAP column. On the on-disk round-trip, `SnapshotParser.fromJson` splits the `operation` key OUT
//! of the summary map (operation becomes the typed `operation` column; the map keeps only the OTHER
//! properties). Rust's `spec::Summary` likewise hoists `operation` out, and `inspect/snapshots.rs` emits
//! only `additional_properties` into the summary column. Because the Java oracle materializes its rows from
//! a RE-PARSED base, the summary maps match Rust's with NO production-code change. (The ROOT snapshot's
//! `operation`-only summary therefore projects to an EMPTY map on both sides — a column this fixture
//! exercises on purpose.)
//!
//! Comparison is ORDER-INDEPENDENT: snapshots are sorted by `snapshot_id` and refs by `name` on BOTH sides
//! before the bulk field-by-field equality, and the summary map is compared as a real `HashMap` (not by
//! key order). A handful of focused named assertions sit alongside the bulk equality so a single regressed
//! column (a dropped retention field, a wrongly split summary, a wrong micros conversion) is pinpointed.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::io::FileIO;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;
use serde::Deserialize;

/// Root of the committed inspection fixtures, relative to the `iceberg` crate manifest.
fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/interop/inspection")
}

/// The base snapshot ids — MUST match `InteropOracle.InspectionOracle.{ROOT,CURRENT,SIBLING}_ID`.
const ROOT: i64 = 3051729675574597004;
const CURRENT: i64 = 3055729675574597004;
const SIBLING: i64 = 3060729675574597004;

// ===========================================================================================
// The Java oracle row models — deserialized from java_snapshots.json / java_refs.json.
// ===========================================================================================

/// One row of Java's `SnapshotsTable`. `committed_at` is the raw micros long Java emits (`ts_ms * 1000`);
/// `summary` is the post-operation-split property map.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct JavaSnapshotRow {
    committed_at: i64,
    snapshot_id: i64,
    parent_id: Option<i64>,
    operation: Option<String>,
    manifest_list: Option<String>,
    summary: HashMap<String, String>,
}

/// One row of Java's `RefsTable`. The branch-only retention fields are `None` for a tag; `main` (a branch
/// with no retention) is `None` everywhere.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct JavaRefRow {
    name: String,
    #[serde(rename = "type")]
    ref_type: String,
    snapshot_id: i64,
    max_reference_age_in_ms: Option<i64>,
    min_snapshots_to_keep: Option<i32>,
    max_snapshot_age_in_ms: Option<i64>,
}

/// The Rust-side equivalent of [`JavaSnapshotRow`], extracted from the `snapshots` Arrow batch. Field types
/// + `PartialEq` are identical to the Java model so the two can be compared directly.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RustSnapshotRow {
    committed_at: i64,
    snapshot_id: i64,
    parent_id: Option<i64>,
    operation: Option<String>,
    manifest_list: Option<String>,
    summary: HashMap<String, String>,
}

/// The Rust-side equivalent of [`JavaRefRow`], extracted from the `refs` Arrow batch.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RustRefRow {
    name: String,
    ref_type: String,
    snapshot_id: i64,
    max_reference_age_in_ms: Option<i64>,
    min_snapshots_to_keep: Option<i32>,
    max_snapshot_age_in_ms: Option<i64>,
}

// ===========================================================================================
// Fixture loading + Table construction.
// ===========================================================================================

/// Load + parse a JSON fixture from the inspection fixtures dir.
fn read_fixture<T: for<'de> Deserialize<'de>>(file_name: &str) -> T {
    let path = fixtures_root().join(file_name);
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str::<T>(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Build a `Table` over the Java-written `base.metadata.json`, exactly like the `inspect/*.rs` unit tests:
/// in-memory FileIO + a dummy metadata-location + identifier. The metadata table scans read only the
/// in-memory `TableMetadata`, so no file is ever opened.
fn base_table() -> Table {
    let metadata: TableMetadata = read_fixture("base.metadata.json");
    Table::builder()
        .metadata(metadata)
        .metadata_location("s3://interop-bucket/inspection/metadata/v1.metadata.json".to_string())
        .identifier(TableIdent::from_strs(["interop", "inspection"]).expect("valid identifier"))
        .file_io(FileIO::new_with_memory())
        .build()
        .expect("build base table from Java-written base.metadata.json")
}

/// Collect a metadata-table scan into a single record batch (the inspect scans emit exactly one batch).
async fn single_batch(stream: iceberg::scan::ArrowRecordBatchStream) -> RecordBatch {
    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .expect("collect inspection scan batches");
    assert_eq!(
        batches.len(),
        1,
        "inspection scans emit exactly one batch (got {})",
        batches.len()
    );
    batches.into_iter().next().expect("one batch")
}

// ===========================================================================================
// Arrow column extraction — snapshots + refs into the comparable Rust row structs.
// ===========================================================================================

/// Extract the `snapshots` Arrow batch into [`RustSnapshotRow`]s. Column positions per
/// `inspect/snapshots.rs`: 0=committed_at (Timestamp µs), 1=snapshot_id, 2=parent_id, 3=operation,
/// 4=manifest_list, 5=summary (Map<Utf8,Utf8>).
fn extract_snapshot_rows(batch: &RecordBatch) -> Vec<RustSnapshotRow> {
    let committed_at = batch.column(0).as_primitive::<TimestampMicrosecondType>();
    let snapshot_id = batch.column(1).as_primitive::<Int64Type>();
    let parent_id = batch.column(2).as_primitive::<Int64Type>();
    let operation = batch.column(3).as_string::<i32>();
    let manifest_list = batch.column(4).as_string::<i32>();
    let summary = batch.column(5).as_map();

    (0..batch.num_rows())
        .map(|i| {
            // The summary MAP value for row i: a struct array with "key" (Utf8) + "value" (Utf8) children.
            let entries = summary.value(i);
            let keys = entries.column(0).as_string::<i32>();
            let values = entries.column(1).as_string::<i32>();
            let mut summary_map = HashMap::new();
            for e in 0..entries.len() {
                summary_map.insert(keys.value(e).to_string(), values.value(e).to_string());
            }

            RustSnapshotRow {
                committed_at: committed_at.value(i),
                snapshot_id: snapshot_id.value(i),
                parent_id: optional_i64(parent_id, i),
                operation: optional_str(operation, i),
                manifest_list: optional_str(manifest_list, i),
                summary: summary_map,
            }
        })
        .collect()
}

/// Extract the `refs` Arrow batch into [`RustRefRow`]s. Column positions per `inspect/refs.rs`: 0=name,
/// 1=type, 2=snapshot_id, 3=max_reference_age_in_ms, 4=min_snapshots_to_keep, 5=max_snapshot_age_in_ms.
fn extract_ref_rows(batch: &RecordBatch) -> Vec<RustRefRow> {
    let name = batch.column(0).as_string::<i32>();
    let ref_type = batch.column(1).as_string::<i32>();
    let snapshot_id = batch.column(2).as_primitive::<Int64Type>();
    let max_reference_age_in_ms = batch.column(3).as_primitive::<Int64Type>();
    let min_snapshots_to_keep = batch.column(4).as_primitive::<Int32Type>();
    let max_snapshot_age_in_ms = batch.column(5).as_primitive::<Int64Type>();

    (0..batch.num_rows())
        .map(|i| RustRefRow {
            name: name.value(i).to_string(),
            ref_type: ref_type.value(i).to_string(),
            snapshot_id: snapshot_id.value(i),
            max_reference_age_in_ms: optional_i64(max_reference_age_in_ms, i),
            min_snapshots_to_keep: {
                if min_snapshots_to_keep.is_null(i) {
                    None
                } else {
                    Some(min_snapshots_to_keep.value(i))
                }
            },
            max_snapshot_age_in_ms: optional_i64(max_snapshot_age_in_ms, i),
        })
        .collect()
}

fn optional_i64(arr: &arrow_array::PrimitiveArray<Int64Type>, i: usize) -> Option<i64> {
    if arr.is_null(i) {
        None
    } else {
        Some(arr.value(i))
    }
}

fn optional_str(arr: &StringArray, i: usize) -> Option<String> {
    if arr.is_null(i) {
        None
    } else {
        Some(arr.value(i).to_string())
    }
}

// ===========================================================================================
// Direction 1 — bulk field-by-field equality (snapshots).
// ===========================================================================================

/// RISK: a divergent column projection — a wrong micros conversion (`ts_ms * 1000`), a wrongly split
/// summary (the `operation` key leaking into / out of the map), a dropped parent-id, or a wrong manifest —
/// would silently break "read a table Java wrote". Asserting the full row set ORDER-INDEPENDENTLY (sorted by
/// snapshot_id, summary compared as a map) pins EVERY snapshots column against Java's real `SnapshotsTable`
/// rows at once.
#[tokio::test]
async fn test_snapshots_table_matches_java_rows() {
    let table = base_table();
    let batch = single_batch(
        table
            .inspect()
            .snapshots()
            .scan()
            .await
            .expect("snapshots scan"),
    )
    .await;

    let mut rust_rows = extract_snapshot_rows(&batch);
    rust_rows.sort_by_key(|r| r.snapshot_id);

    let java_rows: Vec<JavaSnapshotRow> = read_fixture("java_snapshots.json");
    let mut java_as_rust: Vec<RustSnapshotRow> = java_rows
        .into_iter()
        .map(|j| RustSnapshotRow {
            committed_at: j.committed_at,
            snapshot_id: j.snapshot_id,
            parent_id: j.parent_id,
            operation: j.operation,
            manifest_list: j.manifest_list,
            summary: j.summary,
        })
        .collect();
    java_as_rust.sort_by_key(|r| r.snapshot_id);

    assert_eq!(
        rust_rows.len(),
        3,
        "the fixture has exactly three snapshots (ROOT, CURRENT, SIBLING)"
    );
    assert_eq!(
        rust_rows, java_as_rust,
        "Rust `snapshots` rows must equal Java's SnapshotsTable rows field-for-field (committed_at micros, \
         snapshot_id, parent_id, operation, manifest_list, summary map)"
    );

    // Focused, named assertions so a single regressed column is pinpointed.
    let by_id: HashMap<i64, &RustSnapshotRow> =
        rust_rows.iter().map(|r| (r.snapshot_id, r)).collect();

    // committed_at = ts_ms * 1000. ROOT_TS_MS = 1515100955770.
    assert_eq!(
        by_id[&ROOT].committed_at,
        1515100955770 * 1000,
        "committed_at must be timestamp_ms * 1000 (micros)"
    );

    // ROOT's summary was operation-only → projects to an EMPTY map (operation hoisted out on both sides).
    assert!(
        by_id[&ROOT].summary.is_empty(),
        "ROOT's operation-only summary must project to an empty map (operation split out)"
    );
    assert_eq!(by_id[&ROOT].parent_id, None, "ROOT has no parent");

    // CURRENT carries the MULTI-KEY summary: operation split out, three properties survive.
    let current_summary = &by_id[&CURRENT].summary;
    assert_eq!(
        current_summary.get("added-data-files").map(String::as_str),
        Some("3")
    );
    assert_eq!(
        current_summary.get("added-records").map(String::as_str),
        Some("100")
    );
    assert_eq!(
        current_summary.get("total-records").map(String::as_str),
        Some("100")
    );
    assert!(
        !current_summary.contains_key("operation"),
        "the `operation` key must NOT appear in the summary map (it is the typed column)"
    );
    assert_eq!(by_id[&CURRENT].parent_id, Some(ROOT));
    assert_eq!(
        by_id[&CURRENT].operation.as_deref(),
        Some("append"),
        "operation is the typed column"
    );

    // SIBLING: a single surviving summary property.
    assert_eq!(
        by_id[&SIBLING]
            .summary
            .get("added-data-files")
            .map(String::as_str),
        Some("1")
    );
    assert_eq!(
        by_id[&SIBLING].summary.len(),
        1,
        "SIBLING keeps one property"
    );
    assert_eq!(by_id[&SIBLING].parent_id, Some(ROOT));
}

// ===========================================================================================
// Direction 1 — bulk field-by-field equality (refs).
// ===========================================================================================

/// RISK: a flipped branch-vs-tag `type`, a dropped or misrouted retention field, or a wrong snapshot-id
/// would silently break round-trip interop with Java-written tables. Asserting the full row set
/// ORDER-INDEPENDENTLY (sorted by name) pins EVERY refs column against Java's real `RefsTable` rows.
#[tokio::test]
async fn test_refs_table_matches_java_rows() {
    let table = base_table();
    let batch = single_batch(table.inspect().refs().scan().await.expect("refs scan")).await;

    let mut rust_rows = extract_ref_rows(&batch);
    rust_rows.sort_by(|a, b| a.name.cmp(&b.name));

    let java_rows: Vec<JavaRefRow> = read_fixture("java_refs.json");
    let mut java_as_rust: Vec<RustRefRow> = java_rows
        .into_iter()
        .map(|j| RustRefRow {
            name: j.name,
            ref_type: j.ref_type,
            snapshot_id: j.snapshot_id,
            max_reference_age_in_ms: j.max_reference_age_in_ms,
            min_snapshots_to_keep: j.min_snapshots_to_keep,
            max_snapshot_age_in_ms: j.max_snapshot_age_in_ms,
        })
        .collect();
    java_as_rust.sort_by(|a, b| a.name.cmp(&b.name));

    assert_eq!(
        rust_rows.len(),
        3,
        "the fixture has exactly three refs (main, dev, stable)"
    );
    assert_eq!(
        rust_rows, java_as_rust,
        "Rust `refs` rows must equal Java's RefsTable rows field-for-field (name, type, snapshot_id, all \
         three retention fields)"
    );

    // Focused, named assertions pinning the retention NULL pattern per ref kind.
    let by_name: HashMap<String, &RustRefRow> =
        rust_rows.iter().map(|r| (r.name.clone(), r)).collect();

    // main: a branch with NO retention → all three retention fields NULL.
    let main = by_name.get("main").expect("main ref present");
    assert_eq!(main.ref_type, "BRANCH");
    assert_eq!(main.snapshot_id, CURRENT);
    assert_eq!(main.max_reference_age_in_ms, None, "main: max_ref_age NULL");
    assert_eq!(main.min_snapshots_to_keep, None, "main: min_keep NULL");
    assert_eq!(main.max_snapshot_age_in_ms, None, "main: max_snap_age NULL");

    // dev: a branch with FULL retention → all three set.
    let dev = by_name.get("dev").expect("dev ref present");
    assert_eq!(dev.ref_type, "BRANCH");
    assert_eq!(dev.snapshot_id, CURRENT);
    assert_eq!(dev.max_reference_age_in_ms, Some(604_800_000));
    assert_eq!(dev.min_snapshots_to_keep, Some(2));
    assert_eq!(dev.max_snapshot_age_in_ms, Some(86_400_000));

    // stable: a TAG with ONLY max_reference_age set → the two branch-only fields are NULL.
    let stable = by_name.get("stable").expect("stable ref present");
    assert_eq!(stable.ref_type, "TAG");
    assert_eq!(stable.snapshot_id, ROOT);
    assert_eq!(stable.max_reference_age_in_ms, Some(259_200_000));
    assert_eq!(
        stable.min_snapshots_to_keep, None,
        "a tag has no min_snapshots_to_keep"
    );
    assert_eq!(
        stable.max_snapshot_age_in_ms, None,
        "a tag has no max_snapshot_age_in_ms"
    );
}
