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

//! Java interop test for the INCREMENTAL SCANS (rows R122/R123) — the sole deferral on those rows.
//!
//! Two incremental read primitives over a snapshot RANGE `(from exclusive | inclusive, to inclusive]`:
//!
//!   * `Table::incremental_append_scan()` — the data files APPENDED in the range (Java
//!     `IncrementalAppendScan.fromSnapshotExclusive/fromSnapshotInclusive(...).toSnapshot(...).planFiles()`),
//!     considering ONLY `APPEND`-operation snapshots.
//!   * `Table::incremental_changelog_scan()` — the DATA-FILE-LEVEL changelog (an INSERT per data file added
//!     and a DELETE per data file removed) over the range (Java `IncrementalChangelogScan` /
//!     `IncrementalDataTableScan`). This is the data-file changelog, NOT row-level CDC — see the residue note.
//!
//! THE FIXTURE (V2, unpartitioned, schema `{1 id long required, 2 data string optional}`), built identically
//! by BOTH the Java oracle (`IncrementalScanOracle.generate`) and the Rust GEN path below, with DETERMINISTIC
//! data-file basenames so the planned-file SETS compare across the two engines (which use different snapshot
//! ids + different temp roots — the comparison key is the BASENAME, never the snapshot id or the full path):
//!
//!   S1: fast_append A (`a.parquet`, id=10)              op=append
//!   S2: fast_append B (`b.parquet`, id=20)              op=append
//!   S3: fast_append C (`c.parquet`, id=30)              op=append
//!   S4: OVERWRITE — delete A, add D (`d.parquet`, id=40) op=overwrite
//!
//! THE EXPECTED SETS (HAND-DECLARED IDENTICALLY ON BOTH SIDES — anti-circular). The Java oracle declares the
//! SAME expected basename sets in `IncrementalScanOracle`; neither side derives its expectation from the
//! other's output. The inclusive/exclusive boundary is the corruption edge (an off-by-one snapshot
//! includes/excludes the WRONG file), pinned explicitly per scenario:
//!
//!   APPEND scan:
//!     append_excl  : from S1 EXCLUSIVE, to S3            ⇒ {b, c}     (S1's own A excluded)
//!     append_incl  : from S1 INCLUSIVE, to S3            ⇒ {a, b, c}  (S1's own A included)
//!     append_to_cur: from S2 EXCLUSIVE, to=current(S4)  ⇒ {c}        (S3's C; S4 overwrite excluded)
//!   CHANGELOG scan:
//!     changelog_full: from S1 EXCLUSIVE, to S4 ⇒ INSERT b, INSERT c, DELETE a, INSERT d
//!
//! THE TWO DIRECTIONS:
//!   * DIRECTION 1 (`ICEBERG_INTEROP_INCREMENTAL_SCANS_DIR`): Java writes the table + emits the planned sets
//!     it computed with the REAL `IncrementalAppendScan` / `IncrementalDataTableScan`; Rust loads the SAME
//!     table, runs ITS incremental scans, and asserts its planned sets == Java's AND == the hand-declared
//!     ground truth.
//!   * DIRECTION 2 (`ICEBERG_INTEROP_INCREMENTAL_SCANS_GEN_DIR`): Rust WRITES the same chain to
//!     `<dir>/rust_table` via the production write path; the Java oracle's `verify-interop-incremental-scans`
//!     mode runs the REAL Java incremental scans over the RUST-written table and asserts the SAME sets.
//!
//! THE ENV GATE. Both tests are clean NO-OPs when their env var is unset (a runtime early-return, NOT
//! `#[ignore]`), so the offline `cargo test` gate stays green with no Java/Maven. The
//! `dev/java-interop/run-interop-incremental-scans.sh` driver sets the vars and runs the REAL comparison
//! (reset → Java gen → Rust gen → D2 → D1 → sabotage).
//!
//! RESIDUE (row R123, NAMED, not over-claimed). The changelog is DATA-FILE-level (whole-file added/deleted),
//! matching Java's CURRENT `IncrementalDataTableScan` data-file changelog. It is NOT row-level CDC:
//! `ChangelogOperation::{UpdateBefore, UpdateAfter}` net-row changes are NOT computed, and `BatchScan` is
//! unimplemented. A range whose snapshots carry row-level DELETE manifests is rejected (`FeatureUnsupported`),
//! exactly as Java's data-file changelog rejects them today.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::scan::ChangelogOperation;
use iceberg::spec::{
    DataContentType, DataFile, FormatVersion, NestedField, PrimitiveType, Schema, SortOrder,
    Struct, TableMetadata, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use serde::Deserialize;

// ===========================================================================================
// Hand-declared GROUND TRUTH (anti-circular — declared IDENTICALLY in IncrementalScanOracle.java).
// The comparison key is the data-file BASENAME (the two engines write at different temp roots / with
// different uuids, so the full path never matches; the basename is stamped deterministically on both sides).
// ===========================================================================================

/// The append-scan scenarios + the EXPECTED appended-file basename set for each. The off-by-one
/// inclusive/exclusive boundary is the corruption edge, pinned by the contrast between `append_excl`
/// (S1's own `a.parquet` EXCLUDED) and `append_incl` (S1's own `a.parquet` INCLUDED).
fn expected_append_scenarios() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        // from S1 EXCLUSIVE → S3: S1's own A is NOT appended in the range.
        ("append_excl", vec!["b.parquet", "c.parquet"]),
        // from S1 INCLUSIVE → S3: the inclusive bound resolves to S1's parent, so S1's own A IS included.
        ("append_incl", vec!["a.parquet", "b.parquet", "c.parquet"]),
        // from S2 EXCLUSIVE, default to=current(S4): only S3's C; the S4 OVERWRITE is excluded (append-only).
        ("append_to_cur", vec!["c.parquet"]),
    ]
}

/// The changelog scenario + the EXPECTED `(basename, operation)` entries, the operation as the wire token
/// ("INSERT" / "DELETE" — the SAME tokens Java emits and `ChangelogOperation` maps to). The S4 overwrite
/// removes A (DELETE) and adds D (INSERT); the S2/S3 appends of B/C are each an INSERT. (`ChangelogOperation`
/// is not `Hash`, so the set key carries the operation as a `&str`.)
fn expected_changelog_entries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("b.parquet", "INSERT"),
        ("c.parquet", "INSERT"),
        ("a.parquet", "DELETE"),
        ("d.parquet", "INSERT"),
    ]
}

/// Map a Rust [`ChangelogOperation`] to its wire token, matching Java's `ChangelogOperation.name()`.
fn op_token(op: ChangelogOperation) -> &'static str {
    match op {
        ChangelogOperation::Insert => "INSERT",
        ChangelogOperation::Delete => "DELETE",
    }
}

// ===========================================================================================
// Env gates + the Java planned-set model.
// ===========================================================================================

/// The temp dir the Java oracle wrote the multi-snapshot table + its planned-set JSON into (Direction 1).
fn d1_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_INCREMENTAL_SCANS_DIR").map(PathBuf::from)
}

/// The temp dir into which the Direction-2 GEN path writes a Rust-authored table for Java to judge.
fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_INCREMENTAL_SCANS_GEN_DIR").map(PathBuf::from)
}

/// Java's emitted APPEND planned sets: `{ scenario_name -> [basename, ...] }`. The basenames are the tails
/// of the paths Java's `IncrementalAppendScan.planFiles()` returned.
#[derive(Debug, Deserialize)]
struct JavaAppendSets {
    scenarios: HashMap<String, Vec<String>>,
}

/// Java's emitted CHANGELOG entries: a list of `{ basename, operation }` from Java's `IncrementalDataTableScan`.
#[derive(Debug, Deserialize)]
struct JavaChangelogEntry {
    basename: String,
    /// "INSERT" or "DELETE" (the data-file changelog operations Java emits).
    operation: String,
}

/// Strip a path down to its file basename (the comparison key).
fn basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

fn read_java_append_sets(dir: &Path) -> JavaAppendSets {
    let path = dir.join("java_append_sets.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&json).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn read_java_changelog(dir: &Path) -> Vec<JavaChangelogEntry> {
    let path = dir.join("java_changelog_entries.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&json).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

// ===========================================================================================
// Table construction + the deterministic multi-snapshot chain (shared by GEN + the load path).
// ===========================================================================================

/// The unpartitioned V2 schema both sides build: `{1 id long required, 2 data string optional}`.
fn gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, data string} schema")
}

/// Build a `Table` over the Java-written `final.metadata.json`, using a local-filesystem `FileIO` so the
/// absolute on-disk manifest-list + manifest + parquet paths resolve directly (same convention as
/// `interop_scan_exec`).
fn load_table(dir: &Path) -> Table {
    let metadata_path = dir.join("table/metadata/final.metadata.json");
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(
            TableIdent::from_strs(["interop", "incremental_scans"]).expect("valid identifier"),
        )
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from Java-written final.metadata.json")
}

/// Write a REAL one-row parquet data file `(id, data)` at `<table>/data/<basename>` via the production
/// `ParquetWriterBuilder`, returning an unpartitioned [`DataFile`] (content `Data`, default spec id 0).
async fn write_one_row_data_file(table: &Table, basename: &str, id: i64, data: &str) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));

    let ids = Int64Array::from(vec![id]);
    let values = StringArray::from(vec![data]);
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(ids) as ArrayRef,
        Arc::new(values) as ArrayRef,
    ])
    .expect("build the one-row data batch");

    let file_path = format!("{}/data/{}", table.metadata().location(), basename);
    let output = table
        .file_io()
        .new_output(file_path)
        .expect("new parquet output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder
        .build(output)
        .await
        .expect("build parquet writer");
    writer.write(&batch).await.expect("write data batch");
    let data_file_builders = writer.close().await.expect("close parquet writer");

    data_file_builders
        .into_iter()
        .next()
        .expect("one data file builder")
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build unpartitioned data file")
}

/// Create the unpartitioned V2 table at exactly `<table_location>` in a `MemoryCatalog` over the local FS.
async fn create_rust_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");

    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(gen_schema())
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();

    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table")
}

/// fast_append a single data file and return the updated table.
async fn fast_append(catalog: &impl Catalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

/// Build the deterministic 4-snapshot chain into `table` (already created), returning the final table:
///   S1 append A, S2 append B, S3 append C, S4 OVERWRITE (delete A, add D).
async fn build_chain(catalog: &impl Catalog, table: Table) -> Table {
    let a = write_one_row_data_file(&table, "a.parquet", 10, "a").await;
    let a_path = a.file_path().to_string();
    let table = fast_append(catalog, &table, a).await;

    let b = write_one_row_data_file(&table, "b.parquet", 20, "b").await;
    let table = fast_append(catalog, &table, b).await;

    let c = write_one_row_data_file(&table, "c.parquet", 30, "c").await;
    let table = fast_append(catalog, &table, c).await;

    // S4: an OVERWRITE that deletes A (by path) and adds D — produces a DELETE changelog for A and an
    // INSERT for D, and (being a non-APPEND snapshot) is EXCLUDED from the append scan.
    let d = write_one_row_data_file(&table, "d.parquet", 40, "d").await;
    let tx = Transaction::new(&table);
    let tx = tx
        .overwrite_files()
        .delete_file(a_path)
        .add_file(d)
        .apply(tx)
        .expect("apply overwrite");
    tx.commit(catalog).await.expect("commit overwrite")
}

/// The snapshot ids of the 4-snapshot chain, in commit order (S1..S4). Derived by walking the parent chain
/// from the current snapshot back to the root, so the scans key off REAL ids (never hardcoded).
fn chain_snapshot_ids(table: &Table) -> [i64; 4] {
    let metadata = table.metadata();
    let mut ids = Vec::new();
    let mut current = metadata.current_snapshot().cloned();
    while let Some(snapshot) = current {
        ids.push(snapshot.snapshot_id());
        current = snapshot
            .parent_snapshot_id()
            .and_then(|p| metadata.snapshot_by_id(p).cloned());
    }
    ids.reverse();
    assert_eq!(
        ids.len(),
        4,
        "the chain must have exactly 4 snapshots, got {ids:?}"
    );
    [ids[0], ids[1], ids[2], ids[3]]
}

// ===========================================================================================
// Rust incremental-scan planning → comparable basename sets.
// ===========================================================================================

/// The appended-file BASENAME set the Rust incremental append scan plans for a `(from_exclusive?,
/// from_inclusive?, to?)` range.
async fn rust_append_basenames(
    table: &Table,
    from_exclusive: Option<i64>,
    from_inclusive: Option<i64>,
    to: Option<i64>,
) -> HashSet<String> {
    let mut builder = table.incremental_append_scan();
    if let Some(from) = from_exclusive {
        builder = builder.from_snapshot_id_exclusive(from);
    }
    if let Some(from) = from_inclusive {
        builder = builder.from_snapshot_id_inclusive(from);
    }
    if let Some(to) = to {
        builder = builder.to_snapshot_id(to);
    }
    let scan = builder.build().expect("build incremental append scan");
    let tasks: Vec<_> = scan
        .plan_files()
        .await
        .expect("plan_files should succeed")
        .try_collect()
        .await
        .expect("collect file scan tasks");
    tasks
        .into_iter()
        .map(|t| basename(&t.data_file_path))
        .collect()
}

/// The `(basename, operation_token)` entries the Rust incremental changelog scan plans for
/// `(from_exclusive, to]`. The operation is the wire token so the set key is `Hash`-able.
async fn rust_changelog_entries(
    table: &Table,
    from_exclusive: i64,
    to: i64,
) -> HashSet<(String, &'static str)> {
    let scan = table
        .incremental_changelog_scan()
        .from_snapshot_id_exclusive(from_exclusive)
        .to_snapshot_id(to)
        .build()
        .expect("build incremental changelog scan");
    let tasks: Vec<_> = scan
        .plan_files()
        .await
        .expect("plan_files should succeed")
        .try_collect()
        .await
        .expect("collect changelog tasks");
    tasks
        .into_iter()
        .map(|t| (basename(t.data_file_path()), op_token(t.operation())))
        .collect()
}

// ===========================================================================================
// DIRECTION 1 — Java writes the multi-snapshot table; Rust plans + asserts == Java AND == ground truth.
// ===========================================================================================

#[tokio::test]
async fn test_incremental_scans_d1_rust_plans_java_table() {
    let Some(dir) = d1_dir() else {
        println!(
            "skipping interop_incremental_scans D1 — set ICEBERG_INTEROP_INCREMENTAL_SCANS_DIR \
             (run dev/java-interop/run-interop-incremental-scans.sh)"
        );
        return;
    };

    let table = load_table(&dir);
    let [s1, s2, s3, _s4] = chain_snapshot_ids(&table);
    let current = table
        .metadata()
        .current_snapshot_id()
        .expect("current snapshot");

    let java_append = read_java_append_sets(&dir);
    let java_changelog = read_java_changelog(&dir);

    // -- APPEND scan: each scenario's Rust planned set == the hand-declared ground truth == Java's planned set.
    for (name, expected_basenames) in expected_append_scenarios() {
        let expected: HashSet<String> = expected_basenames.iter().map(|s| s.to_string()).collect();

        let rust_set = match name {
            "append_excl" => rust_append_basenames(&table, Some(s1), None, Some(s3)).await,
            "append_incl" => rust_append_basenames(&table, None, Some(s1), Some(s3)).await,
            // default to=current; assert the resolved `to` IS the current snapshot (S4).
            "append_to_cur" => rust_append_basenames(&table, Some(s2), None, None).await,
            other => panic!("unknown append scenario {other}"),
        };

        assert_eq!(
            rust_set, expected,
            "[{name}] Rust appended-file set must equal the hand-declared ground truth"
        );

        let java_set: HashSet<String> = java_append
            .scenarios
            .get(name)
            .unwrap_or_else(|| panic!("Java emitted no append scenario {name}"))
            .iter()
            .map(|p| basename(p))
            .collect();
        assert_eq!(
            rust_set, java_set,
            "[{name}] Rust appended-file set must equal Java's IncrementalAppendScan.planFiles() over the \
             SAME range"
        );
    }

    // Pin the inclusive/exclusive boundary explicitly: excl drops A, incl keeps A — the ONLY difference.
    let excl = rust_append_basenames(&table, Some(s1), None, Some(s3)).await;
    let incl = rust_append_basenames(&table, None, Some(s1), Some(s3)).await;
    assert!(
        !excl.contains("a.parquet"),
        "from S1 EXCLUSIVE must NOT include S1's own a.parquet (the off-by-one corruption edge)"
    );
    assert!(
        incl.contains("a.parquet"),
        "from S1 INCLUSIVE MUST include S1's own a.parquet"
    );

    // Pin that the unset `to` resolves to the current snapshot (S4).
    let scan = table
        .incremental_append_scan()
        .from_snapshot_id_exclusive(s2)
        .build()
        .expect("build append scan with default to");
    assert_eq!(
        scan.to_snapshot_id(),
        Some(current),
        "unset to_snapshot_id must default to the current snapshot (S4)"
    );

    // -- CHANGELOG scan: the data-file-level added/deleted entries == ground truth == Java's.
    let rust_changelog = rust_changelog_entries(&table, s1, current).await;
    let expected_changelog: HashSet<(String, &'static str)> = expected_changelog_entries()
        .into_iter()
        .map(|(b, op)| (b.to_string(), op))
        .collect();
    assert_eq!(
        rust_changelog, expected_changelog,
        "Rust changelog (S1 excl, S4] must equal the hand-declared INSERT/DELETE ground truth"
    );

    let java_changelog_set: HashSet<(String, &'static str)> = java_changelog
        .iter()
        .map(|e| {
            let op = match e.operation.as_str() {
                "INSERT" => "INSERT",
                "DELETE" => "DELETE",
                other => panic!("unexpected Java changelog operation {other}"),
            };
            (e.basename.clone(), op)
        })
        .collect();
    assert_eq!(
        rust_changelog, java_changelog_set,
        "Rust changelog must equal Java's IncrementalDataTableScan over the SAME range"
    );

    println!(
        "interop_incremental_scans D1 OK — Rust incremental_append_scan + incremental_changelog_scan over the \
         Java-written multi-snapshot table match Java's IncrementalAppendScan/IncrementalDataTableScan AND the \
         hand-declared ground truth (append: {{b,c}} / {{a,b,c}} / {{c}}; changelog: +b +c -a +d)"
    );
}

// ===========================================================================================
// DIRECTION 2 — the GEN path: Rust WRITES the same chain; Java's verify runs the REAL Java scans over it.
// ===========================================================================================

#[tokio::test]
async fn test_incremental_scans_gen_rust_writes_java_judgeable_table() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_incremental_scans GEN — set ICEBERG_INTEROP_INCREMENTAL_SCANS_GEN_DIR \
             (run dev/java-interop/run-interop-incremental-scans.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_incremental_gen",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");

    let table = create_rust_table(&catalog, &table_location).await;
    let table = build_chain(&catalog, table).await;

    // SANITY: Rust's OWN incremental scans must already match the ground truth before handing to Java.
    // (Direction-1 proves Rust reads what Java writes; here we confirm the Rust-written chain is internally
    // consistent so a Java verify failure is unambiguously a Java-read finding.)
    let [s1, s2, s3, _s4] = chain_snapshot_ids(&table);
    let current = table
        .metadata()
        .current_snapshot_id()
        .expect("current snapshot");

    let excl = rust_append_basenames(&table, Some(s1), None, Some(s3)).await;
    assert_eq!(
        excl,
        HashSet::from(["b.parquet".to_string(), "c.parquet".to_string()]),
        "GEN sanity: from S1 EXCLUSIVE → S3 must be {{b,c}}"
    );
    let incl = rust_append_basenames(&table, None, Some(s1), Some(s3)).await;
    assert_eq!(
        incl,
        HashSet::from([
            "a.parquet".to_string(),
            "b.parquet".to_string(),
            "c.parquet".to_string(),
        ]),
        "GEN sanity: from S1 INCLUSIVE → S3 must be {{a,b,c}}"
    );
    let to_cur = rust_append_basenames(&table, Some(s2), None, None).await;
    assert_eq!(
        to_cur,
        HashSet::from(["c.parquet".to_string()]),
        "GEN sanity: from S2 EXCLUSIVE, default to=current(S4) must be {{c}} (overwrite excluded)"
    );

    let changelog = rust_changelog_entries(&table, s1, current).await;
    assert_eq!(
        changelog,
        expected_changelog_entries()
            .into_iter()
            .map(|(b, op)| (b.to_string(), op))
            .collect::<HashSet<_>>(),
        "GEN sanity: changelog (S1 excl, S4] must be +b +c -a +d"
    );

    // Write the FINAL metadata to a KNOWN path so Java loads it deterministically. The real on-disk
    // manifest-list + manifests + parquet already live under <gen_dir>/rust_table.
    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    println!(
        "interop_incremental_scans GEN OK — Rust wrote {table_location} (4-snapshot chain: append a/b/c + \
         overwrite delete-a/add-d + final.metadata.json); Rust's own incremental scans match the ground truth. \
         Java verify-interop-incremental-scans runs the REAL Java scans over it next."
    );
}
