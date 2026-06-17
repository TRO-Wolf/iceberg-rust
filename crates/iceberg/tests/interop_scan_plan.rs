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

//! Java interop for the SCAN-PLAN layer (`plan_tasks`) — GAP_MATRIX row 146.
//!
//! Proves Rust `TableScan::plan_tasks` produces the SAME bin-packed `CombinedScanTask` GROUPS as Java's
//! REAL `table.newScan().option(SPLIT_SIZE/LOOKBACK/OPEN_FILE_COST, ...).planTasks()`, in BOTH directions,
//! with the target/lookback/open-file-cost HAND-DECLARED IDENTICALLY on both sides (anti-circular — the
//! constants below mirror `InteropOracle.ScanPlanOracle` EXACTLY; neither side derives its knobs from the
//! other).
//!
//! THE FIXTURE (V2, UNPARTITIONED, schema `{1 id long required, 2 data string optional}`), built identically
//! by BOTH the Java oracle and the Rust GEN path: several REAL parquet data files of VARYING size + a MoR
//! position delete, so split + bin-pack are non-trivial:
//!
//! * `big.parquet` — many rows written with a TINY parquet row-group size, so it has MULTIPLE row groups
//!   ⇒ non-null strictly-ascending split offsets ⇒ the OFFSETS-AWARE split fires.
//! * `mid.parquet` — a medium single-row-group file ⇒ FIXED-SIZE split under the small target.
//! * `small1/small2` — two small files that PACK together.
//! * `big-deletes` — a position delete over `big.parquet` so big's sub-tasks carry deletes and the
//!   bin-pack WEIGHT includes the delete bytes.
//!
//! THE COMPARISON. Each emitted group is a SORTED set of member keys `(basename,start,length)` (basename =
//! the file's tail — the cross-engine key, since the two engines write at different roots). The plan is the
//! MULTISET of per-group member-key sets + the group count. Rust and Java BOTH plan the SAME on-disk table
//! within a direction, so split offsets (hence start/length) are byte-identical; the set-of-sets + count
//! must match exactly. Group emission ORDER is NOT compared (an internal bin-packer detail).
//!
//! THE TWO DIRECTIONS (driven by `dev/java-interop/run-interop-scan-plan.sh`):
//!
//! * D1 (`ICEBERG_INTEROP_SCAN_PLAN_DIR`): Java writes the table + emits `java_scan_plan.json`; Rust loads
//!   the SAME table, runs `plan_tasks` with the hand-declared knobs, asserts its plan == Java's.
//! * GEN/D2 (`ICEBERG_INTEROP_SCAN_PLAN_GEN_DIR`): Rust WRITES the same logical table to `<dir>/rust_table`
//!   and emits `rust_scan_plan.json`; the Java oracle's `verify-interop-scan-plan` runs the REAL Java
//!   planTasks over the RUST-written table and asserts the SAME plan.
//!
//! THE ENV GATE. Both tests are clean NO-OPs when their env var is unset (a runtime early-return, NOT
//! `#[ignore]`), so the offline `cargo test` gate stays green with no Java/Maven.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::io::{FileIO, LocalFsStorageFactory};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::scan::CombinedScanTask;
use iceberg::spec::{
    DataContentType, DataFile, FormatVersion, NestedField, PrimitiveType, Schema, SortOrder,
    Struct, TableMetadata, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use serde::Deserialize;

// ===========================================================================================
// HAND-DECLARED knobs (anti-circular — mirror InteropOracle.ScanPlanOracle EXACTLY).
// ===========================================================================================

/// The bin-pack target in bytes (Java `ScanPlanOracle.TARGET`).
const TARGET: u64 = 4096;
/// The planning lookback (Java `ScanPlanOracle.LOOKBACK`).
const LOOKBACK: usize = 5;
/// The per-open file cost in bytes (Java `ScanPlanOracle.OPEN_FILE_COST`).
const OPEN_FILE_COST: u64 = 0;

// ===========================================================================================
// Env gates + the Java plan model.
// ===========================================================================================

/// The dir the Java oracle wrote its table + `java_scan_plan.json` into (Direction 1).
fn d1_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_SCAN_PLAN_DIR").map(PathBuf::from)
}

/// The dir into which the Direction-2 GEN path writes the Rust-authored table for Java to judge.
fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_SCAN_PLAN_GEN_DIR").map(PathBuf::from)
}

/// Java's emitted plan: `{ groupCount, groups: [[memberKey, ...], ...] }`.
#[derive(Debug, Deserialize)]
struct JavaScanPlan {
    #[serde(rename = "groupCount")]
    group_count: usize,
    groups: Vec<Vec<String>>,
}

/// Strip a path to its basename (the cross-engine comparison key).
fn basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

/// The MULTISET of per-group member-key sets, as a sorted `Vec` of sorted member `Vec`s. Using a sorted
/// `Vec`-of-`Vec`s (NOT a `Set`-of-`Set`s) preserves DUPLICATE groups — two distinct bins that happen to
/// hold the same member set must both count — which is the faithful multiset contract.
type PlanMultiset = Vec<Vec<String>>;

/// Normalize a set of groups (each a member-key set) into the canonical comparison form: each group sorted,
/// then the list of groups sorted. Order-insensitive across groups, duplicate-preserving.
fn normalize(groups: Vec<BTreeSet<String>>) -> PlanMultiset {
    let mut out: PlanMultiset = groups
        .into_iter()
        .map(|group| group.into_iter().collect::<Vec<_>>())
        .collect();
    out.sort();
    out
}

/// The member key for one file-scan task: `(basename,start,length)` — identical to Java's `memberKey`.
fn member_key(task: &iceberg::scan::FileScanTask) -> String {
    format!(
        "({},{},{})",
        basename(task.data_file_path()),
        task.start(),
        task.length()
    )
}

/// Run `plan_tasks` and collect the canonical plan multiset for the table scan built with the hand-declared
/// knobs (target / lookback / open-file-cost set via the builder).
async fn rust_plan_multiset(
    table: &Table,
    target: u64,
    lookback: usize,
    cost: u64,
) -> PlanMultiset {
    let scan = table
        .scan()
        .with_split_size(target)
        .with_split_lookback(lookback)
        .with_split_open_file_cost(cost)
        .build()
        .expect("build scan");
    let groups: Vec<CombinedScanTask> = scan
        .plan_tasks()
        .await
        .expect("plan_tasks")
        .try_collect()
        .await
        .expect("collect groups");

    let group_sets: Vec<BTreeSet<String>> = groups
        .iter()
        .map(|group| group.tasks().iter().map(member_key).collect())
        .collect();
    normalize(group_sets)
}

/// Convert Java's emitted plan into the same canonical multiset form for comparison.
fn java_plan_multiset(plan: &JavaScanPlan) -> PlanMultiset {
    let group_sets: Vec<BTreeSet<String>> = plan
        .groups
        .iter()
        .map(|group| group.iter().cloned().collect())
        .collect();
    let normalized = normalize(group_sets);
    assert_eq!(
        normalized.len(),
        plan.group_count,
        "Java groupCount must equal the number of emitted groups"
    );
    normalized
}

fn read_java_plan(dir: &Path) -> JavaScanPlan {
    let path = dir.join("java_scan_plan.json");
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&json).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

// ===========================================================================================
// Table construction (shared by GEN + the load path).
// ===========================================================================================

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

/// Build a `Table` over the Java-written `final.metadata.json` (local-filesystem `FileIO`).
fn load_table(dir: &Path) -> Table {
    let metadata_path = dir.join("table/metadata/final.metadata.json");
    let json = fs::read_to_string(&metadata_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", metadata_path.display()));
    let metadata: TableMetadata = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("parse {}: {error}", metadata_path.display()));

    Table::builder()
        .metadata(metadata)
        .metadata_location(metadata_path.to_string_lossy().to_string())
        .identifier(TableIdent::from_strs(["interop", "scan_plan"]).expect("valid identifier"))
        .file_io(FileIO::new_with_fs())
        .build()
        .expect("build table from Java-written final.metadata.json")
}

/// Write a REAL parquet data file of `row_count` rows at `<table>/data/<basename>`. When `tiny_row_groups`
/// is set, the parquet writer uses a small max-row-group-size so a many-row file gets MULTIPLE row groups
/// (hence non-null strictly-ascending split offsets, exercising the offsets-aware split).
async fn write_data_file(
    table: &Table,
    basename: &str,
    row_count: usize,
    tiny_row_groups: bool,
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));

    let ids: Vec<i64> = (0..row_count as i64).collect();
    let values: Vec<String> = (0..row_count).map(|i| format!("row-{i:06}")).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(values)) as ArrayRef,
    ])
    .expect("build the data batch");

    let mut props = parquet::file::properties::WriterProperties::builder();
    if tiny_row_groups {
        // A tiny max row-group size forces several row groups for a many-row file.
        props = props.set_max_row_group_size(64);
    }
    let file_path = format!("{}/data/{}", table.metadata().location(), basename);
    let output = table
        .file_io()
        .new_output(file_path)
        .expect("new parquet output");
    let parquet_builder = ParquetWriterBuilder::new(props.build(), schema.clone());
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

/// Write a REAL parquet position-delete deleting position 0 of `data_file_path` (unpartitioned).
async fn write_position_delete(table: &Table, data_file_path: &str) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "big-deletes".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        iceberg::spec::DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(None)
        .await
        .expect("build position-delete writer");

    let paths = StringArray::from(vec![data_file_path]);
    let positions = Int64Array::from(vec![0_i64]);
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(paths) as ArrayRef,
        Arc::new(positions) as ArrayRef,
    ])
    .expect("build the position-delete batch");
    writer
        .write(batch)
        .await
        .expect("write position-delete batch");
    writer
        .close()
        .await
        .expect("close position-delete writer")
        .into_iter()
        .next()
        .expect("one position-delete file")
}

/// Create the unpartitioned V2 table at exactly `table_location`.
async fn create_rust_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, std::collections::HashMap::new())
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

// ===========================================================================================
// Direction 1 — Rust plans the JAVA-written table.
// ===========================================================================================

#[tokio::test]
async fn test_scan_plan_d1_rust_plans_java_table() {
    let Some(dir) = d1_dir() else {
        println!(
            "skipping interop_scan_plan D1 — set ICEBERG_INTEROP_SCAN_PLAN_DIR \
             (run dev/java-interop/run-interop-scan-plan.sh)"
        );
        return;
    };

    let table = load_table(&dir);
    let rust = rust_plan_multiset(&table, TARGET, LOOKBACK, OPEN_FILE_COST).await;
    let java = java_plan_multiset(&read_java_plan(&dir));

    assert_eq!(
        rust, java,
        "Rust plan_tasks over the Java table must equal Java's planTasks plan (multiset of per-group \
         member-key sets + group count)"
    );
    println!(
        "interop_scan_plan D1 OK — Rust plan_tasks over the Java table matches Java ({} groups)",
        rust.len()
    );
}

// ===========================================================================================
// Direction 2 GEN — Rust writes a Java-judgeable table + emits its own plan.
// ===========================================================================================

#[tokio::test]
async fn test_scan_plan_gen_rust_writes_java_judgeable_table() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_scan_plan GEN — set ICEBERG_INTEROP_SCAN_PLAN_GEN_DIR \
             (run dev/java-interop/run-interop-scan-plan.sh)"
        );
        return;
    };

    let warehouse = gen_dir.to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "interop_scan_plan_gen",
            std::collections::HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.clone(),
            )]),
        )
        .await
        .expect("build MemoryCatalog over local FS");

    let table = create_rust_table(&catalog, &table_location).await;

    // The same varying-size fixture the Java oracle builds: big (multi-row-group), mid, small1, small2.
    let big = write_data_file(&table, "big.parquet", 800, true).await;
    let big_path = big.file_path().to_string();
    let mid = write_data_file(&table, "mid.parquet", 40, false).await;
    let small1 = write_data_file(&table, "small1.parquet", 5, false).await;
    let small2 = write_data_file(&table, "small2.parquet", 5, false).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![big, mid, small1, small2])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");

    // A MoR position delete over big.parquet (position 0).
    let delete_file = write_position_delete(&table, &big_path).await;
    assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete_file])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");

    // Compute OUR OWN plan and emit it for Java to verify.
    let rust = rust_plan_multiset(&table, TARGET, LOOKBACK, OPEN_FILE_COST).await;

    // SANITY: the big file MUST have split into more than one sub-task (multi-row-group ⇒ offsets-aware
    // split), otherwise the offsets-aware branch is not actually exercised by the GEN fixture.
    let big_sub_tasks: usize = rust
        .iter()
        .flat_map(|group| group.iter())
        .filter(|member| member.starts_with("(big.parquet,"))
        .count();
    assert!(
        big_sub_tasks > 1,
        "GEN sanity: big.parquet must split into >1 sub-task (got {big_sub_tasks}); the tiny row-group \
         size should produce multiple row groups + split offsets"
    );

    // Write the FINAL metadata for Java to load, and the Rust plan for Java to compare.
    let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
    table
        .metadata()
        .write_to(table.file_io(), &final_metadata_path)
        .await
        .expect("write final.metadata.json");

    let plan_json = rust_plan_to_json(&rust);
    let plan_path = format!("{table_location}/rust_scan_plan.json");
    fs::write(&plan_path, plan_json).expect("write rust_scan_plan.json");

    println!(
        "interop_scan_plan GEN OK — Rust wrote {table_location} (big/mid/small1/small2 + a position delete) \
         and emitted rust_scan_plan.json ({} groups, big split into {big_sub_tasks} sub-tasks). Java \
         verify-interop-scan-plan runs the REAL planTasks over it next.",
        rust.len()
    );
}

/// Serialize the Rust plan multiset as `{groupCount, groups:[[memberKey,...],...]}` for the Java verify
/// (the SAME JSON shape Java's `planToJson` emits — each group sorted, the group list sorted).
fn rust_plan_to_json(plan: &PlanMultiset) -> String {
    // `plan` is already normalized (each group sorted, list sorted).
    let groups: Vec<serde_json::Value> = plan
        .iter()
        .map(|group| serde_json::Value::Array(group.iter().map(|m| m.clone().into()).collect()))
        .collect();
    serde_json::json!({
        "groupCount": plan.len(),
        "groups": groups,
    })
    .to_string()
}

// ===========================================================================================
// Offline self-test — the env-gated tests are no-ops without Java; this one runs ALWAYS and exercises
// the comparison plumbing (normalize / multiset equality) so the gate has live coverage of the oracle's
// own model independent of Java.
// ===========================================================================================

#[test]
fn normalize_is_order_insensitive_across_groups_and_duplicate_preserving() {
    // Two groups in either order normalize to the same canonical form.
    let a = normalize(vec![
        BTreeSet::from(["(b.parquet,0,10)".to_string()]),
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
    ]);
    let b = normalize(vec![
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
        BTreeSet::from(["(b.parquet,0,10)".to_string()]),
    ]);
    assert_eq!(a, b, "group order must not matter");

    // A duplicate group (same member set) is PRESERVED — the multiset, not the set, is the contract.
    let with_dup = normalize(vec![
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
        BTreeSet::from(["(a.parquet,0,10)".to_string()]),
    ]);
    assert_eq!(with_dup.len(), 2, "duplicate groups must both count");

    // A genuinely different plan does NOT compare equal (the off-by-one start that the sabotage exploits).
    let shifted = normalize(vec![BTreeSet::from(["(a.parquet,1,10)".to_string()])]);
    let original = normalize(vec![BTreeSet::from(["(a.parquet,0,10)".to_string()])]);
    assert_ne!(shifted, original, "a shifted start must diverge");

    // Round-trip through the JSON the Java verify reads, then parse it back as a Java plan.
    let plan = normalize(vec![
        BTreeSet::from([
            "(a.parquet,0,10)".to_string(),
            "(b.parquet,0,20)".to_string(),
        ]),
        BTreeSet::from(["(c.parquet,0,30)".to_string()]),
    ]);
    let json = rust_plan_to_json(&plan);
    let parsed: JavaScanPlan = serde_json::from_str(&json).expect("round-trip JSON");
    assert_eq!(
        java_plan_multiset(&parsed),
        plan,
        "JSON round-trip must be lossless"
    );
}
