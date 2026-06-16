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

//! CONFLICT-VALIDATION write-action interop (increment C4) — `ReplacePartitions`'
//! `validate_no_conflicting_data()` proven against Java `BaseReplacePartitions`'s conflict
//! validation on the conflict DECISION (ACCEPT vs REJECT) over a concurrent-commit history.
//!
//! This is the SECOND conflict-validation interop slice (after `interop_overwrite_conflict.rs`),
//! covering the residue named identically in every GAP_MATRIX write-action cell
//! ("conflict-validation paths NOT covered"). Like the `OverwriteFiles` slice, it proves the
//! conflict DECISION (not surviving rows) matches Java for the same S0→S1 history.
//!
//! ## How this DIFFERS from the `OverwriteFiles` slice (C1)
//!
//! `ReplacePartitions`' conflict is **PARTITION-SCOPED** — a concurrent commit that added data into
//! a REPLACED `(spec_id, partition)` tuple is a conflict (Java `file_in_replaced_partition`). There
//! is **NO `conflict_detection_filter`** and **NO `case_sensitive`** on `ReplacePartitions`
//! (confirmed: `javap` shows neither). So the table is **PARTITIONED** by `identity(category)`, and
//! the partition — not an inclusive-metrics y-bound — is the conflict axis.
//!
//! ## The insight that makes it tractable (same shape as C1)
//!
//! `validate_no_conflicting_data` depends ONLY on (a) which data files a CONCURRENT commit added
//! after the read snapshot and (b) the partitions the replace replaces — NOT on the replace's own
//! payload. So both engines run a SYMMETRIC replace against the OTHER engine's table — "add one
//! fresh local file in partition cat=a + `validate_from_snapshot(S0)` + `validate_no_conflicting_data()`"
//! — and the ACCEPT/REJECT outcome is a pure function of the table's S0→S1 history. The expected
//! outcome is HAND-DECLARED identically on both sides (anti-circularity), with Java as the reference
//! engine.
//!
//! ## The scenario contract (hard-coded identically in Java + Rust)
//!
//! Schema `{1 id long required, 2 category string required}`, PARTITIONED by `identity(category)`,
//! V2. Per scenario the table is:
//! - S0: `fast_append` base files — cat=a (id=1) + cat=b (id=2).
//! - S1: `fast_append` CONCURRENT file into ONE partition (cat=a or cat=b) — id=100.
//!
//! The validating engine loads the table (current = S1), derives `validate_from_snapshot(S0)` (the
//! root snapshot), and runs the symmetric replace (add a fresh cat=a file — a REPLACED partition).
//!
//! | scenario             | concurrent add | expected |
//! |----------------------|----------------|----------|
//! | `replaced_partition` | cat=a          | REJECT   |
//! | `other_partition`    | cat=b          | ACCEPT   |
//!
//! - `replaced_partition`: the concurrent file landed in cat=a — the SAME partition the replace
//!   replaces ⇒ a lost write under serializable isolation ⇒ REJECT (non-retryable `DataInvalid`).
//! - `other_partition`: the concurrent file landed in cat=b — a DIFFERENT partition ⇒ no conflict ⇒
//!   ACCEPT (the false-positive guard: an over-eager check must not reject disjoint-partition writes).
//!
//! ## Directions
//!
//! - **D1 (Rust validates Java's table):** `test_rust_validates_java_replace_partitions_conflict_tables`.
//!   Rust `register_table`s `<dir>/<scenario>/table` (Java-built) into a `MemoryCatalog` and runs the
//!   replace; the outcome must equal the scenario's hand-declared expected. (The ACCEPT commit writes
//!   new `vN.metadata.json` orphans but never the fixed-name `final.metadata.json`, so this is
//!   re-run-safe.)
//! - **D2 (Java validates Rust's table):**
//!   `test_replace_partitions_conflict_gen_rust_writes_java_validatable_tables` writes
//!   `<dir>/<scenario>/rust_table` (S0+S1); Java's `verify-interop-replace-partitions-conflict` loads
//!   it, runs the replace, and asserts the same outcome.
//!
//! GATED on env vars (both unset ⇒ clean no-ops; offline `cargo test` gate stays green):
//! - `ICEBERG_INTEROP_REPLACE_PARTITIONS_CONFLICT_GEN_DIR` — Rust GEN (Rust writes the per-scenario tables)
//! - `ICEBERG_INTEROP_REPLACE_PARTITIONS_CONFLICT_DIR`     — D1 comparison (Rust validates Java's tables)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PartitionKey, PartitionSpec,
    PrimitiveType, Schema, SchemaRef, SortOrder, Struct, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation, TableIdent};

// ===========================================================================================
// The scenario contract — hand-declared IDENTICALLY in Java (ReplacePartitionsConflictOracle) and here.
// ===========================================================================================

/// The expected conflict-validation outcome of the symmetric replace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The replace committed — no conflicting concurrent data in a replaced partition.
    Accept,
    /// The replace was rejected with a non-retryable `DataInvalid` (Java `ValidationException`).
    Reject,
}

/// One conflict scenario: its on-disk directory name, the partition the CONCURRENT S1 add lands in,
/// and the hand-declared expected outcome (Java is the reference engine).
#[derive(Debug, Clone, Copy)]
struct Scenario {
    name: &'static str,
    concurrent_category: &'static str,
    expected: Outcome,
}

/// The two scenarios. Java's `ReplacePartitionsConflictOracle.SCENARIOS` MUST match this set exactly
/// (same names, same concurrent partition, same expected outcomes).
fn scenarios() -> Vec<Scenario> {
    vec![
        // Concurrent S1 added data into cat=a — the SAME partition the symmetric replace replaces.
        Scenario {
            name: "replaced_partition",
            concurrent_category: "a",
            expected: Outcome::Reject,
        },
        // Concurrent S1 added data into cat=b — a DIFFERENT partition; the replace (cat=a) is unaffected.
        Scenario {
            name: "other_partition",
            concurrent_category: "b",
            expected: Outcome::Accept,
        },
    ]
}

// ===========================================================================================
// Env-var gates.
// ===========================================================================================

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REPLACE_PARTITIONS_CONFLICT_GEN_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn validate_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REPLACE_PARTITIONS_CONFLICT_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// ===========================================================================================
// Schema + spec + real-parquet helpers (partitioned {id, category} by identity(category)).
// ===========================================================================================

/// The fixture schema `{1 id long required, 2 category string required}`.
fn conflict_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, category string} schema")
}

/// `identity(category)` unbound partition spec, spec id 0.
fn conflict_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category".to_string(), Transform::Identity)
        .expect("add identity(category) partition field")
        .build()
}

/// Build the `PartitionKey` for `category = <value>` over the bound spec + schema.
fn partition_key(schema: SchemaRef, spec: PartitionSpec, category: &str) -> PartitionKey {
    PartitionKey::new(
        spec,
        schema,
        Struct::from_iter([Some(Literal::string(category))]),
    )
}

/// Build a `MemoryCatalog` over local-FS storage rooted at `warehouse`.
async fn build_catalog(name: &str, warehouse: &str) -> impl Catalog + use<> {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .expect("build MemoryCatalog over local FS")
}

/// Create a PARTITIONED (identity(category)) V2 `{id, category}` table at `table_location`.
async fn create_conflict_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(conflict_schema())
        .partition_spec(conflict_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create conflict rust_table")
}

/// Write a REAL parquet `{id, category}` data file for ONE identity(category) partition via the
/// production `DataFileWriter`. Every row carries `category`; the partition value drives where the
/// file's manifest entry is routed — the partition-scoped conflict axis.
async fn write_category_file(
    table: &Table,
    partition_key: &PartitionKey,
    category: &str,
    ids: Vec<i64>,
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let row_count = ids.len();
    let categories: Vec<&str> = std::iter::repeat_n(category, row_count).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(categories)) as ArrayRef,
    ])
    .expect("build the {id, category} data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "cpdata".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key.clone()))
        .await
        .expect("build partitioned data file writer");
    writer.write(batch).await.expect("write data batch");
    writer
        .close()
        .await
        .expect("close data file writer")
        .into_iter()
        .next()
        .expect("one data file")
}

/// `fast_append` the given files and return the committed table.
async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply fast_append");
    tx.commit(catalog).await.expect("commit fast_append")
}

/// Build the per-scenario table history: `fast_append` base cat=a (id=1) + cat=b (id=2) at S0, then
/// a `fast_append` of the CONCURRENT file (id=100) into the scenario's partition at S1. Returns the
/// table at S1.
async fn build_scenario_table(catalog: &impl Catalog, table: Table, scenario: &Scenario) -> Table {
    let schema = table.metadata().current_schema().clone();
    let bound_spec = table.metadata().default_partition_spec().as_ref().clone();
    let pk_a = partition_key(schema.clone(), bound_spec.clone(), "a");
    let pk_b = partition_key(schema.clone(), bound_spec.clone(), "b");

    // S0: base files cat=a (id=1) + cat=b (id=2).
    let base_a = write_category_file(&table, &pk_a, "a", vec![1]).await;
    let base_b = write_category_file(&table, &pk_b, "b", vec![2]).await;
    let table = append_files(catalog, &table, vec![base_a, base_b]).await;

    // S1: the CONCURRENT file (id=100) into the scenario's partition.
    let pk_concurrent = partition_key(schema, bound_spec, scenario.concurrent_category);
    let concurrent =
        write_category_file(&table, &pk_concurrent, scenario.concurrent_category, vec![
            100,
        ])
        .await;
    append_files(catalog, &table, vec![concurrent]).await
}

/// Derive `validate_from_snapshot(S0)` — the ROOT snapshot (the one with no parent). After S0+S1
/// the root is S0, so S1 counts as the concurrent commit.
fn root_snapshot_id(table: &Table) -> i64 {
    table
        .metadata()
        .snapshots()
        .find(|s| s.parent_snapshot_id().is_none())
        .expect("a root snapshot (S0) exists")
        .snapshot_id()
}

/// Run the SYMMETRIC replace against `table` and return ACCEPT/REJECT. The replace adds one fresh
/// cat=a file (a REPLACED partition), enables conflict validation, and pins
/// `validate_from_snapshot(S0)`. A non-retryable `DataInvalid` (Java `ValidationException`) is
/// REJECT; a successful commit is ACCEPT. The replace's own added file is irrelevant to the conflict
/// decision — only the concurrent S1 add (in the table history) and the replaced partitions matter.
async fn replace_outcome(catalog: &impl Catalog, table: &Table, _scenario: &Scenario) -> Outcome {
    let from = root_snapshot_id(table);
    let schema = table.metadata().current_schema().clone();
    let bound_spec = table.metadata().default_partition_spec().as_ref().clone();
    let pk_a = partition_key(schema, bound_spec, "a");
    // A fresh cat=a file (id=999) — the replaced partition is consistently cat=a.
    let fresh = write_category_file(table, &pk_a, "a", vec![999]).await;

    let tx = Transaction::new(table);
    let action = tx
        .replace_partitions()
        .add_file(fresh)
        .validate_from_snapshot(from)
        .validate_no_conflicting_data();
    let tx = action.apply(tx).expect("apply replace_partitions");

    match tx.commit(catalog).await {
        Ok(_) => Outcome::Accept,
        Err(e) => {
            assert_eq!(
                e.kind(),
                ErrorKind::DataInvalid,
                "a conflict must be a non-retryable validation failure (DataInvalid), got: {e}"
            );
            assert!(
                !e.retryable(),
                "the conflict validation failure must be NON-retryable, got: {e}"
            );
            assert!(
                e.message().contains("conflicting files"),
                "the error must name the conflict, got: {}",
                e.message()
            );
            Outcome::Reject
        }
    }
}

// ===========================================================================================
// D2 GEN — Rust writes the per-scenario tables for Java's verify to validate.
// ===========================================================================================

/// Rust builds each scenario's `<gen_dir>/<scenario>/rust_table` (S0+S1) and lands
/// `final.metadata.json`. The sanity check confirms Rust's OWN conflict decision matches the
/// scenario's hand-declared expected outcome before handing the table to Java — so a Rust-side
/// regression is caught here, not silently shipped to the Java verify.
#[tokio::test]
async fn test_replace_partitions_conflict_gen_rust_writes_java_validatable_tables() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_replace_partitions_conflict GEN — set \
             ICEBERG_INTEROP_REPLACE_PARTITIONS_CONFLICT_GEN_DIR \
             (run dev/java-interop/run-interop-replace-partitions-conflict.sh)"
        );
        return;
    };

    for scenario in scenarios() {
        let scenario_dir = gen_dir.join(scenario.name);
        let warehouse = scenario_dir.to_string_lossy().to_string();
        let table_location = format!("{warehouse}/rust_table");
        let catalog = build_catalog(
            &format!("interop_replace_partitions_conflict_gen_{}", scenario.name),
            &warehouse,
        )
        .await;

        let table = create_conflict_table(&catalog, &table_location).await;
        let table = build_scenario_table(&catalog, table, &scenario).await;

        // Land final metadata at the known path BEFORE the sanity replace (which would otherwise
        // write further metadata versions). final.metadata.json reflects the clean S0+S1 history.
        let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
        table
            .metadata()
            .clone()
            .write_to(table.file_io(), &final_metadata_path)
            .await
            .expect("write final.metadata.json");

        // Sanity: Rust's OWN conflict decision must equal the hand-declared expected outcome.
        let outcome = replace_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision must match the hand-declared expected outcome",
            scenario.name
        );

        println!(
            "interop_replace_partitions_conflict GEN OK — scenario {} wrote {table_location} \
             (S0 cat=a+cat=b + concurrent S1 cat={}); Rust decision = {:?} (expected {:?})",
            scenario.name, scenario.concurrent_category, outcome, scenario.expected
        );
    }
}

// ===========================================================================================
// D1 — Rust validates the JAVA-written tables (register + replace).
// ===========================================================================================

/// Rust registers each JAVA-written `<dir>/<scenario>/table` into a `MemoryCatalog` and runs the
/// symmetric replace; the conflict decision must equal the scenario's hand-declared expected
/// outcome. This is DIRECTION 1: Rust's validation runs against Java's exact on-disk manifests +
/// partition metadata. The ACCEPT commit writes orphan `vN.metadata.json` files but never the
/// fixed-name `final.metadata.json`, so the fixture stays re-run-safe.
#[tokio::test]
async fn test_rust_validates_java_replace_partitions_conflict_tables() {
    let Some(dir) = validate_dir() else {
        println!(
            "skipping interop_replace_partitions_conflict D1 — set \
             ICEBERG_INTEROP_REPLACE_PARTITIONS_CONFLICT_DIR \
             (run dev/java-interop/run-interop-replace-partitions-conflict.sh)"
        );
        return;
    };

    for scenario in scenarios() {
        let scenario_dir = dir.join(scenario.name);
        let metadata_path = scenario_dir.join("table/metadata/final.metadata.json");
        assert!(
            metadata_path.exists(),
            "scenario {}: missing Java table at {} (run the Java generate step first)",
            scenario.name,
            metadata_path.display()
        );

        let warehouse = scenario_dir.to_string_lossy().to_string();
        let catalog = build_catalog(
            &format!("interop_replace_partitions_conflict_d1_{}", scenario.name),
            &warehouse,
        )
        .await;
        let namespace = NamespaceIdent::new("interop".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace for register");
        // The catalog derives the next metadata version from the registered file NAME, which must
        // match `<version>-<uuid>.metadata.json`. Java writes a fixed-name `final.metadata.json`, so
        // register a conventionally-named COPY (an ACCEPT commit then writes `<version+1>-…`;
        // `final.metadata.json` is left untouched, keeping the fixture re-run-safe).
        let reg_path = scenario_dir.join(format!(
            "table/metadata/99999-{}.metadata.json",
            uuid::Uuid::now_v7()
        ));
        std::fs::copy(&metadata_path, &reg_path)
            .expect("copy final.metadata.json to a registerable <version>-<uuid> name");

        let ident = TableIdent::new(namespace, format!("java_{}", scenario.name));
        let table = catalog
            .register_table(&ident, reg_path.to_string_lossy().to_string())
            .await
            .expect("register the Java-written conflict table");

        let outcome = replace_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision on the JAVA table must match the expected outcome",
            scenario.name
        );

        println!(
            "interop_replace_partitions_conflict D1 OK — scenario {}: Rust validated the Java table \
             (concurrent S1 cat={}) → {:?} (expected {:?})",
            scenario.name, scenario.concurrent_category, outcome, scenario.expected
        );
    }
}
