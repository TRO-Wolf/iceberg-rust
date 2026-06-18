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

//! REWRITE-TABLE-PATH interop (GAP_MATRIX row 137) — proving the Rust
//! [`RewriteTablePath`](iceberg::maintenance::RewriteTablePath) action agrees with Java 1.10.0's
//! engine-agnostic iceberg-CORE `RewriteTablePathUtil` (`replacePaths` / `rewriteManifestList` /
//! `rewriteDataManifest` / `rewriteDeleteManifest` / `rewritePositionDeleteFile`) on a fixture table
//! carrying data, a POSITION-delete, and an EQUALITY-delete (every branch).
//!
//! # The anti-circular, BIDIRECTIONAL comparison
//!
//! `RewriteTablePathUtil` is engine-agnostic CORE (not Spark), so Java can DRIVE the real rewrite. Java's
//! `RewriteTablePathOracle`:
//!
//! 1. builds the fixture at `<dir>/table`, declares the SAME source/target prefixes BOTH sides
//!    hand-declare here (anti-circular — never copies the other engine's output paths),
//! 2. runs the REAL `RewriteTablePathUtil.*` over the fixture and emits two PATH-INDEPENDENT descriptors:
//!    `java_graph.json` (the SET of all rewritten locations, each RELATIVIZED to the target prefix so it
//!    is comparable across temp roots) and `java_copy_plan.json` (the `(from, to)` copy-plan, each side
//!    relativized to its prefix — STAGED entries' `from` relativized to `<STAGING>`, VERBATIM entries'
//!    `from` relativized to `<SOURCE>`, every `to` relativized to `<TARGET>`).
//!
//! This Rust test reads Java's `final.metadata.json`, runs the Rust `RewriteTablePath` with the SAME
//! prefixes + a hand-declared staging dir, builds the SAME two descriptors, and asserts they EQUAL
//! Java's (DIRECTION 2 — "Rust computes, asserts == Java"). It also emits `rust_graph.json` /
//! `rust_copy_plan.json` so the Java `verify` step (DIRECTION 1 — "Java judges Rust") can re-judge.
//!
//! # The two descriptors (the cross-engine tokens — NEVER raw absolute paths)
//!
//! - GRAPH: the set of every rewritten metadata location (metadata.json + manifest-lists + manifests +
//!   data + position-delete + equality-delete + statistics targets), each relativized to the TARGET
//!   prefix. NOTHING in the rewritten graph may retain the source prefix.
//! - COPY-PLAN: the set of `(fromTag, toTag)` pairs, where `toTag` = relativize(target_loc, target) and
//!   `fromTag` = `STAGED:<rel>` if the `from` is under the staging location else `SOURCE:<rel>` (rel =
//!   relativize(from, staging|source)). This makes the per-class direction (STAGED vs SOURCE) part of
//!   the cross-engine token — a flipped direction diverges.
//!
//! # The env gate
//!
//! The test is a clean NO-OP (runtime early-return) unless `ICEBERG_INTEROP_REWRITE_TABLE_PATH_DIR` is
//! set non-empty, so the offline `cargo test` gate needs no Java/Maven. The driver
//! `dev/java-interop/run-interop-rewrite-table-path.sh` sets it.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use iceberg::io::FileIO;
use iceberg::maintenance::RewriteTablePath;
use iceberg::spec::TableMetadata;
use iceberg::table::Table;
use iceberg::{NamespaceIdent, TableIdent};

/// The env var that drives the test; absent/empty ⇒ a clean no-op (offline gate stays green).
fn interop_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REWRITE_TABLE_PATH_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The HAND-DECLARED target prefix — identical on BOTH sides (anti-circular). The source prefix is the
/// table's own on-disk location (read from the Java metadata), declared identically by both engines.
const TARGET_PREFIX: &str = "s3://relocated-bucket/warehouse/db/table";

#[tokio::test]
async fn rewrite_table_path_graph_and_copy_plan_match_java() {
    let Some(dir) = interop_dir() else {
        eprintln!(
            "skipping interop_rewrite_table_path — set ICEBERG_INTEROP_REWRITE_TABLE_PATH_DIR to run \
             it (driven by dev/java-interop/run-interop-rewrite-table-path.sh)"
        );
        return;
    };

    let file_io = FileIO::new_with_fs();
    let java_metadata_path = dir.join("table/metadata/final.metadata.json");
    assert!(
        java_metadata_path.exists(),
        "the Java fixture metadata must exist at {} (run the Java generate step first)",
        java_metadata_path.display()
    );

    let metadata = TableMetadata::read_from(&file_io, java_metadata_path.to_str().expect("utf8"))
        .await
        .expect("read Java metadata");
    let source_prefix = metadata.location().to_string();
    let staging = dir
        .join("rust_staging")
        .to_str()
        .expect("utf8 staging")
        .to_string();

    // Build a read-only static table over Java's metadata.
    let table = Table::builder()
        .metadata(metadata.clone())
        .metadata_location(java_metadata_path.to_str().expect("utf8").to_string())
        .identifier(TableIdent::new(
            NamespaceIdent::new("interop".to_string()),
            "rewrite_table_path".to_string(),
        ))
        .file_io(file_io.clone())
        .readonly(true)
        .build()
        .expect("build static table");

    // Run the REAL Rust action with the SAME prefixes Java declared.
    let result = RewriteTablePath::new(table)
        .rewrite_location_prefix(&source_prefix, TARGET_PREFIX)
        .staging_location(&staging)
        .execute(&file_io)
        .await
        .expect("execute rewrite table path");

    // --- Build the GRAPH descriptor: the SET of every rewritten TARGET location (the copy-plan
    // targets — manifest-lists + manifests + data + position-delete + equality-delete), each
    // relativized to the target prefix. (The metadata.json target is named by the caller with an
    // engine-specific file name and is NOT in Java's RewriteResult.copyPlan(); its rewrite is covered by
    // the offline replace_paths tests + the staged-metadata re-read, so it is excluded from this
    // cross-engine graph token.)
    let mut graph: BTreeSet<String> = BTreeSet::new();
    for (_from, to) in &result.copy_plan {
        graph.insert(relativize_to(to, TARGET_PREFIX));
        // No rewritten target may retain the source prefix.
        assert!(
            !to.starts_with(&format!("{source_prefix}/")),
            "a rewritten target retained the source prefix: {to}"
        );
    }

    // --- Build the COPY-PLAN descriptor: (fromTag, toTag) with the per-class direction encoded.
    let mut copy_plan: BTreeSet<(String, String)> = BTreeSet::new();
    for (from, to) in &result.copy_plan {
        let to_tag = relativize_to(to, TARGET_PREFIX);
        let from_tag = if from.starts_with(&format!("{staging}/")) {
            format!("STAGED:{}", relativize_to(from, &staging))
        } else if from.starts_with(&format!("{source_prefix}/")) {
            format!("SOURCE:{}", relativize_to(from, &source_prefix))
        } else {
            panic!("copy-plan `from` is neither under staging nor source: {from}");
        };
        copy_plan.insert((from_tag, to_tag));
    }

    let rust_graph = serde_json::to_string(&graph).expect("serialize graph");
    let rust_copy_plan_vec: Vec<[String; 2]> = copy_plan
        .iter()
        .map(|(f, t)| [f.clone(), t.clone()])
        .collect();
    let rust_copy_plan = serde_json::to_string(&rust_copy_plan_vec).expect("serialize copy plan");

    // --- DIRECTION 2: assert the Rust descriptors EQUAL Java's ground truth.
    let java_graph = read_string(&dir.join("java_graph.json"));
    let java_copy_plan = read_string(&dir.join("java_copy_plan.json"));
    let java_graph_set: BTreeSet<String> =
        serde_json::from_str(&java_graph).expect("parse java graph");
    let java_copy_plan_set: BTreeSet<(String, String)> = parse_copy_plan(&java_copy_plan);

    assert_eq!(
        graph, java_graph_set,
        "the Rust rewritten-path GRAPH must equal Java's RewriteTablePathUtil rewrite"
    );
    assert_eq!(
        copy_plan, java_copy_plan_set,
        "the Rust copy-plan (with per-class direction) must equal Java's RewriteResult.copyPlan()"
    );

    // Emit the Rust descriptors for Java's verify (DIRECTION 1) to re-judge.
    fs::write(dir.join("rust_graph.json"), &rust_graph).expect("write rust_graph.json");
    fs::write(dir.join("rust_copy_plan.json"), &rust_copy_plan).expect("write rust_copy_plan.json");

    eprintln!(
        "interop_rewrite_table_path: graph={} entries, copy_plan={} pairs (D2 match)",
        graph.len(),
        copy_plan.len()
    );
}

/// Relativize `path` to `prefix` (strip the prefix + the separator), returning the source-relative
/// portion. Panics if `path` is not under `prefix` (the caller guarantees it).
fn relativize_to(path: &str, prefix: &str) -> String {
    let with_sep = if prefix.ends_with('/') {
        prefix.to_string()
    } else {
        format!("{prefix}/")
    };
    path.strip_prefix(&with_sep)
        .unwrap_or_else(|| panic!("path {path} is not under prefix {prefix}"))
        .to_string()
}

/// Parse the Java copy-plan JSON (an array of two-element arrays) into a set of `(from, to)` pairs.
fn parse_copy_plan(raw: &str) -> BTreeSet<(String, String)> {
    let pairs: Vec<[String; 2]> = serde_json::from_str(raw).expect("parse copy plan json");
    pairs
        .into_iter()
        .map(|p| (p[0].clone(), p[1].clone()))
        .collect()
}

/// Read a file to a string (the Java oracle emits raw JSON).
fn read_string(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}
