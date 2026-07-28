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
//! Gated on `ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR` = the directory holding
//! `warehouse-duckdb-v1.5.1/` and `warehouse-duckdb-v1.5.1-partitioned/`
//! (PrimarySync `fixtures/duckdb-malformed-manifest/`). When unset, tests no-op.
//!
//! Manifest Avro `"schema"` keys contain the DuckDB poison (manifest-entry record + raw Avro
//! bound types). Pre-QD the scan died with `SchemaEnum`. Post-QD: strict parse fails, table
//! schema fallback succeeds, and the scan returns the toy rows. The on-disk poison is
//! **unchanged** (checker still exits 0).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use arrow_array::{Array, Int32Array};
use futures::TryStreamExt;
use iceberg::TableIdent;
use iceberg::io::FileIO;
use iceberg::spec::Manifest;
use iceberg::table::StaticTable;

fn fixture_root() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_DUCKDB_MALFORMED_FIXTURE_DIR").map(PathBuf::from)
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
        // Data manifests: `…-m0.avro` — not snap-*.avro lists.
        if !(name.contains("-m") && name.ends_with(".avro") && !name.starts_with("snap-")) {
            continue;
        }
        let bytes = std::fs::read(&path).expect("read manifest");
        // Strict parse (no table-schema fallback) must still hard-fail.
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
async fn duckdb_malformed_variant_a_scans_with_fallback() {
    let Some(root) = fixture_root() else {
        return;
    };
    let table = root.join("warehouse-duckdb-v1.5.1/db.db/malformed");
    if !table.is_dir() {
        return;
    }
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
async fn duckdb_malformed_variant_b_partitioned_scans_with_fallback() {
    let Some(root) = fixture_root() else {
        return;
    };
    let table = root.join("warehouse-duckdb-v1.5.1-partitioned/db.db/malformed_part");
    if !table.is_dir() {
        return;
    }
    assert_poison_still_present(&table);
    let meta = latest_metadata(&table);
    let ids = scan_ids(&meta).await;
    assert_eq!(
        ids,
        BTreeSet::from([1, 2, 3]),
        "Variant B expected toy ids after QD fallback (partitioned)"
    );
}
