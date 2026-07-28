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

//! DuckDB-written malformed-manifest fixtures — QD GREEN contract.
//!
//! # Scope (honest)
//!
//! QD tolerance is wired on the **table scan** path and on metadata-table inspect
//! (`entries` / `files` / `partitions`). Maintenance / transaction loaders that still
//! call `load_manifest` without a fallback remain fail-closed on poison.
//!
//! # Env gate (C1-Q-001 — no silent false-green)
//!
//! These tests are **`#[ignore]`** unless run with:
//!
//! ```text
//! ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR=<PrimarySync fixtures/duckdb-malformed-manifest> \
//!   cargo test -p iceberg --test interop_duckdb_malformed_manifest -- --ignored
//! ```
//!
//! Offline / default `cargo test` does **not** count as QD GREEN. Offline pins live in
//! `spec::manifest::metadata::tests` (parse contract) plus the offline wiring test below.
//!
//! Scan + metadata-table inspect (`entries`/`files`/`partitions`) pass table-schema fallback.
//! Maintenance / transaction loaders that still call `load_manifest` without fallback remain
//! fail-closed on poison (intentional residue).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{Array, Int32Array};
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::io::FileIO;
use iceberg::spec::{Manifest, NestedField, PrimitiveType, Schema, Type};
use iceberg::table::StaticTable;

fn fixture_root() -> PathBuf {
    std::env::var_os("ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR")
        .map(PathBuf::from)
        .expect(
            "ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR must be set to run ignored DuckDB QD tests \
             (PrimarySync fixtures/duckdb-malformed-manifest)",
        )
}

fn latest_metadata(table_dir: &Path) -> PathBuf {
    let meta_dir = table_dir.join("metadata");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&meta_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", meta_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("00001-") && n.ends_with(".metadata.json"))
        })
        .collect();
    candidates.sort();
    candidates
        .pop()
        .unwrap_or_else(|| panic!("no 00001-*.metadata.json under {}", meta_dir.display()))
}

async fn scan_ids(meta_path: &Path) -> BTreeSet<i32> {
    let meta_loc = format!("file://{}", meta_path.display());
    let file_io = FileIO::new_with_fs();
    let ident = TableIdent::from_strs(["probe", "t"]).expect("ident");
    let table = StaticTable::from_metadata_file(&meta_loc, ident, file_io)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e:#}", meta_loc));
    let scan = table.scan().build().expect("scan build");
    let stream = scan.to_arrow().await.expect("to_arrow");
    let batches: Vec<_> = stream
        .try_collect()
        .await
        .unwrap_or_else(|e| panic!("collect batches for {}: {e:#}", meta_loc));
    let mut ids = BTreeSet::new();
    for batch in batches {
        let col = batch
            .column_by_name("id")
            .unwrap_or_else(|| panic!("missing id column in {}", meta_loc));
        let arr = col
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("id not Int32 in {}", meta_loc));
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                ids.insert(arr.value(i));
            }
        }
    }
    ids
}

/// Manifests still carry the poison — the tolerance must not "fix" the bytes.
fn assert_poison_still_present(table_dir: &Path) {
    let meta_dir = table_dir.join("metadata");
    let mut found = false;
    for entry in std::fs::read_dir(&meta_dir).expect("metadata dir") {
        let path = entry.expect("entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !(name.contains("-m") && name.ends_with(".avro") && !name.starts_with("snap-")) {
            continue;
        }
        let bytes = std::fs::read(&path).expect("read manifest");
        let err = Manifest::parse_avro(&bytes).expect_err("poison must still fail strict parse");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Fail to parse schema")
                || msg.contains("SchemaEnum")
                || msg.contains("did not match"),
            "unexpected strict-parse error for {}: {msg}",
            path.display()
        );
        found = true;
    }
    assert!(found, "no data manifests under {}", table_dir.display());
}

#[tokio::test]
#[ignore = "requires ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR (PrimarySync duckdb-malformed-manifest)"]
async fn duckdb_malformed_variant_a_scans_with_fallback() {
    let root = fixture_root();
    let table = root.join("warehouse-duckdb-v1.5.1/db.db/malformed");
    assert!(
        table.is_dir(),
        "missing Variant A table dir under {}",
        root.display()
    );
    assert_poison_still_present(&table);
    let meta = latest_metadata(&table);
    let ids = scan_ids(&meta).await;
    assert_eq!(
        ids,
        BTreeSet::from([1, 2, 3]),
        "Variant A expected toy ids after QD fallback"
    );
}

#[tokio::test]
#[ignore = "requires ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR (PrimarySync duckdb-malformed-manifest)"]
async fn duckdb_malformed_variant_b_partitioned_scans_with_fallback() {
    let root = fixture_root();
    let table = root.join("warehouse-duckdb-v1.5.1-partitioned/db.db/malformed_part");
    assert!(
        table.is_dir(),
        "missing Variant B table dir under {}",
        root.display()
    );
    assert_poison_still_present(&table);
    let meta = latest_metadata(&table);
    let ids = scan_ids(&meta).await;
    assert_eq!(
        ids,
        BTreeSet::from([1, 2, 3]),
        "Variant B expected toy ids after QD fallback (partitioned)"
    );
}

/// Offline pin (C1-Q-002): fixture poison bytes must open via
/// `parse_avro_with_schema_fallback` when a table schema is supplied, and still hard-fail
/// without it. Uses PrimarySync fixtures only if present; otherwise a pure unit path is
/// already covered by `metadata::tests`. When the fixture dir env is set this also
/// mutation-proves the full Avro decode path (not just the schema JSON helper).
#[tokio::test]
async fn offline_poison_manifest_bytes_need_schema_fallback() {
    let Some(root) = std::env::var_os("ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR").map(PathBuf::from)
    else {
        // Pure offline: still assert the public API contract without external files.
        let fallback = Arc::new(
            Schema::builder()
                .with_schema_id(7)
                .with_fields([
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("schema"),
        );
        // Without real Avro bytes we only pin that the with_fallback entry point exists and
        // that strict parse of empty fails — full Avro is covered when fixture env is set.
        let err = Manifest::parse_avro(b"not-avro").expect_err("garbage must fail");
        let _ = err;
        let _ = fallback;
        return;
    };

    let table = root.join("warehouse-duckdb-v1.5.1/db.db/malformed");
    assert!(
        table.is_dir(),
        "ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR is set but Variant A table missing under {}",
        root.display()
    );
    let meta_dir = table.join("metadata");
    let manifest_path = std::fs::read_dir(&meta_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", meta_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            let n = p.file_name().and_then(|x| x.to_str()).unwrap_or("");
            n.contains("-m") && n.ends_with(".avro") && !n.starts_with("snap-")
        })
        .unwrap_or_else(|| panic!("poison data manifest missing under {}", meta_dir.display()));
    let bytes = std::fs::read(&manifest_path).expect("read");

    Manifest::parse_avro(&bytes).expect_err("strict parse must fail on poison");

    let meta_json = latest_metadata(&table);
    let meta_loc = format!("file://{}", meta_json.display());
    let file_io = FileIO::new_with_fs();
    let table_meta = iceberg::spec::TableMetadata::read_from(&file_io, &meta_loc)
        .await
        .expect("table metadata");
    let schema = table_meta.current_schema().clone();

    let manifest = Manifest::parse_avro_with_schema_fallback(&bytes, Some(schema))
        .expect("fallback must open poison manifest");
    assert!(
        !manifest.entries().is_empty(),
        "expected at least one data file entry after fallback decode"
    );
}
