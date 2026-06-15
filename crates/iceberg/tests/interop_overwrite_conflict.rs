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

//! CONFLICT-VALIDATION write-action interop (increment C1) — `OverwriteFiles`'
//! `validate_no_conflicting_data()` + `conflict_detection_filter` proven against Java
//! `BaseOverwriteFiles.validate` → `validateAddedDataFiles` (`validateNoConflictingData`).
//!
//! This is the FIRST conflict-validation interop slice — the residue named identically in every
//! GAP_MATRIX write-action cell ("conflict-validation paths NOT covered"). Unlike the data-level
//! fixtures (`interop_write_data.rs`), which prove that ROWS survive a commit, this proves the
//! conflict DECISION (accept vs. reject) matches Java for the same concurrent-commit history.
//!
//! ## The insight that makes it tractable
//!
//! `validateNoConflictingData` depends ONLY on (a) the data files a CONCURRENT commit added after
//! the read snapshot and (b) the conflict-detection filter — NOT on the overwrite's own add/delete
//! payload. So both engines can run a SYMMETRIC overwrite against the OTHER engine's table —
//! "add one fresh local file + `conflict_detection_filter(F)` + `validate_from_snapshot(S0)` +
//! `validate_no_conflicting_data()`" — and the ACCEPT/REJECT outcome is a pure function of the
//! table's S0→S1 history plus F. The expected outcome is HAND-DECLARED identically on both sides
//! (anti-circularity, like the data-level oracles' hand-declared row sets), with Java as the
//! reference engine.
//!
//! ## The scenario contract (hard-coded identically in Java + Rust)
//!
//! Schema `{1 id long required, 2 y long required}`, UNPARTITIONED, V2. Per scenario the table is:
//! - S0: `fast_append` base file A0 — rows `(id=1, y=0)`, `(id=2, y=5)` → y bounds `[0,5]`, seq 1.
//! - S1: `fast_append` CONCURRENT file — rows `(id=100, y=lo)`, `(id=101, y=hi)` → y bounds
//!   `[lo,hi]`, seq 2.
//!
//! The validating engine loads the table (current = S1), derives `validate_from_snapshot(S0)` (the
//! root snapshot), and runs the symmetric overwrite. Real parquet gives the concurrent file its
//! `y` column bounds, which drive the inclusive-metrics evaluation on BOTH sides.
//!
//! | scenario          | conflict filter | concurrent `y` bounds | expected |
//! |-------------------|-----------------|-----------------------|----------|
//! | `ge50_overlap`    | `y >= 50`       | `[60,70]` (overlaps)  | REJECT   |
//! | `ge50_excluded`   | `y >= 50`       | `[10,20]` (below 50)  | ACCEPT   |
//! | `nofilter_any`    | none (AlwaysTrue) | `[60,70]`           | REJECT   |
//!
//! - `ge50_overlap`: the concurrent file COULD contain `y >= 50` ⇒ a lost write under serializable
//!   isolation ⇒ the overwrite must be REJECTED (non-retryable `DataInvalid`).
//! - `ge50_excluded`: the concurrent file's bounds are entirely below `50` ⇒ the inclusive evaluator
//!   excludes it ⇒ the overwrite ACCEPTS (a false-positive guard).
//! - `nofilter_any`: no filter ⇒ Java `dataConflictDetectionFilter()` → `alwaysTrue()` ⇒ ANY
//!   concurrent add is a conflict ⇒ REJECT (the conservative serializable default).
//!
//! ## Directions
//!
//! - **D1 (Rust validates Java's table):** `test_rust_validates_java_overwrite_conflict_tables`.
//!   Rust `register_table`s `<dir>/<scenario>/table` (Java-built) into a `MemoryCatalog` and runs
//!   the overwrite; the outcome must equal the scenario's hand-declared expected. (The ACCEPT commit
//!   writes new `vN.metadata.json` orphans but never the fixed-name `final.metadata.json`, so this is
//!   re-run-safe.)
//! - **D2 (Java validates Rust's table):** `test_overwrite_conflict_gen_rust_writes_java_validatable_tables`
//!   writes `<dir>/<scenario>/rust_table` (S0+S1); Java's `verify-interop-overwrite-conflict` loads
//!   it, runs the overwrite, and asserts the same outcome.
//!
//! GATED on env vars (both unset ⇒ clean no-ops; offline `cargo test` gate stays green):
//! - `ICEBERG_INTEROP_OVERWRITE_CONFLICT_GEN_DIR` — Rust GEN (Rust writes the per-scenario tables)
//! - `ICEBERG_INTEROP_OVERWRITE_CONFLICT_DIR`     — D1 comparison (Rust validates Java's tables)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use iceberg::expr::Reference;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, Datum, FormatVersion, NestedField, PrimitiveType, Schema, SortOrder,
    Type,
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
// The scenario contract — hand-declared IDENTICALLY in Java (OverwriteConflictOracle) and here.
// ===========================================================================================

/// Whether the symmetric overwrite sets a `conflict_detection_filter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterKind {
    /// `conflict_detection_filter(y >= 50)`.
    Ge50,
    /// No filter set ⇒ Java `dataConflictDetectionFilter()` → `alwaysTrue()` (any concurrent add is
    /// a conflict).
    None,
}

/// The expected conflict-validation outcome of the symmetric overwrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The overwrite committed — no conflicting concurrent data under the filter.
    Accept,
    /// The overwrite was rejected with a non-retryable `DataInvalid` (Java `ValidationException`).
    Reject,
}

/// One conflict scenario: its on-disk directory name, the overwrite's filter, the concurrent file's
/// `y` bounds, and the hand-declared expected outcome (Java is the reference engine).
#[derive(Debug, Clone, Copy)]
struct Scenario {
    name: &'static str,
    filter: FilterKind,
    concurrent_lo: i64,
    concurrent_hi: i64,
    expected: Outcome,
}

/// The three scenarios. Java's `OverwriteConflictOracle.SCENARIOS` MUST match this set exactly
/// (same names, same bounds, same expected outcomes).
fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "ge50_overlap",
            filter: FilterKind::Ge50,
            concurrent_lo: 60,
            concurrent_hi: 70,
            expected: Outcome::Reject,
        },
        Scenario {
            name: "ge50_excluded",
            filter: FilterKind::Ge50,
            concurrent_lo: 10,
            concurrent_hi: 20,
            expected: Outcome::Accept,
        },
        Scenario {
            name: "nofilter_any",
            filter: FilterKind::None,
            concurrent_lo: 60,
            concurrent_hi: 70,
            expected: Outcome::Reject,
        },
    ]
}

// ===========================================================================================
// Env-var gates.
// ===========================================================================================

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_OVERWRITE_CONFLICT_GEN_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn validate_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_OVERWRITE_CONFLICT_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// ===========================================================================================
// Schema + real-parquet helpers (unpartitioned {id, y}).
// ===========================================================================================

/// The fixture schema `{1 id long required, 2 y long required}`.
fn conflict_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("build the {id long, y long} schema")
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

/// Create an UNPARTITIONED V2 `{id, y}` table at `table_location`.
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
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create conflict rust_table")
}

/// Write a REAL parquet `{id, y}` data file via the production `DataFileWriter`. The `y` values
/// become the file's column bounds in parquet stats — the inputs the inclusive-metrics conflict
/// check reads.
async fn write_yid_file(table: &Table, ids: Vec<i64>, ys: Vec<i64>) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
    ])
    .expect("build the {id, y} data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "cdata".to_string(),
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
        .build(None)
        .await
        .expect("build unpartitioned data file writer");
    writer.write(batch).await.expect("write data batch");
    writer
        .close()
        .await
        .expect("close data file writer")
        .into_iter()
        .next()
        .expect("one data file")
}

/// Build the per-scenario table history: `fast_append` A0 (y bounds `[0,5]`) at S0, then a
/// `fast_append` of the CONCURRENT file (y bounds `[lo,hi]`) at S1. Returns the table at S1.
async fn build_scenario_table(catalog: &impl Catalog, table: Table, scenario: &Scenario) -> Table {
    // S0: base file A0 — rows (1,0),(2,5) → y bounds [0,5].
    let file_a0 = write_yid_file(&table, vec![1, 2], vec![0, 5]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_a0])
        .apply(tx)
        .expect("apply fast_append A0 (S0)");
    let table = tx.commit(catalog).await.expect("commit S0");

    // S1: the CONCURRENT file — rows (100,lo),(101,hi) → y bounds [lo,hi].
    let concurrent = write_yid_file(&table, vec![100, 101], vec![
        scenario.concurrent_lo,
        scenario.concurrent_hi,
    ])
    .await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![concurrent])
        .apply(tx)
        .expect("apply fast_append concurrent (S1)");
    tx.commit(catalog).await.expect("commit S1")
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

/// Run the SYMMETRIC overwrite against `table` and return ACCEPT/REJECT. The overwrite adds one
/// fresh local file, enables conflict validation, pins `validate_from_snapshot(S0)`, and (for
/// `FilterKind::Ge50`) sets `conflict_detection_filter(y >= 50)`. A non-retryable `DataInvalid`
/// (Java `ValidationException`) is REJECT; a successful commit is ACCEPT.
async fn overwrite_outcome(catalog: &impl Catalog, table: &Table, scenario: &Scenario) -> Outcome {
    let from = root_snapshot_id(table);
    // The overwrite's own added file is irrelevant to the conflict decision; a fresh (999,1) file.
    let fresh = write_yid_file(table, vec![999], vec![1]).await;

    let tx = Transaction::new(table);
    let mut action = tx
        .overwrite_files()
        .add_file(fresh)
        .validate_from_snapshot(from)
        .validate_no_conflicting_data();
    if scenario.filter == FilterKind::Ge50 {
        action = action.conflict_detection_filter(
            Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
        );
    }
    let tx = action.apply(tx).expect("apply overwrite");

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
async fn test_overwrite_conflict_gen_rust_writes_java_validatable_tables() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_overwrite_conflict GEN — set \
             ICEBERG_INTEROP_OVERWRITE_CONFLICT_GEN_DIR \
             (run dev/java-interop/run-interop-overwrite-conflict.sh)"
        );
        return;
    };

    for scenario in scenarios() {
        let scenario_dir = gen_dir.join(scenario.name);
        let warehouse = scenario_dir.to_string_lossy().to_string();
        let table_location = format!("{warehouse}/rust_table");
        let catalog = build_catalog(
            &format!("interop_overwrite_conflict_gen_{}", scenario.name),
            &warehouse,
        )
        .await;

        let table = create_conflict_table(&catalog, &table_location).await;
        let table = build_scenario_table(&catalog, table, &scenario).await;

        // Land final metadata at the known path BEFORE the sanity overwrite (which would otherwise
        // write further metadata versions). final.metadata.json reflects the clean S0+S1 history.
        let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
        table
            .metadata()
            .clone()
            .write_to(table.file_io(), &final_metadata_path)
            .await
            .expect("write final.metadata.json");

        // Sanity: Rust's OWN conflict decision must equal the hand-declared expected outcome.
        let outcome = overwrite_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision must match the hand-declared expected outcome",
            scenario.name
        );

        println!(
            "interop_overwrite_conflict GEN OK — scenario {} wrote {table_location} \
             (S0 y[0,5] + concurrent S1 y[{},{}]); Rust decision = {:?} (expected {:?})",
            scenario.name,
            scenario.concurrent_lo,
            scenario.concurrent_hi,
            outcome,
            scenario.expected
        );
    }
}

// ===========================================================================================
// D1 — Rust validates the JAVA-written tables (register + overwrite).
// ===========================================================================================

/// Rust registers each JAVA-written `<dir>/<scenario>/table` into a `MemoryCatalog` and runs the
/// symmetric overwrite; the conflict decision must equal the scenario's hand-declared expected
/// outcome. This is DIRECTION 1: Rust's validation runs against Java's exact on-disk manifests +
/// parquet metrics. The ACCEPT commit writes orphan `vN.metadata.json` files but never the
/// fixed-name `final.metadata.json`, so the fixture stays re-run-safe.
#[tokio::test]
async fn test_rust_validates_java_overwrite_conflict_tables() {
    let Some(dir) = validate_dir() else {
        println!(
            "skipping interop_overwrite_conflict D1 — set ICEBERG_INTEROP_OVERWRITE_CONFLICT_DIR \
             (run dev/java-interop/run-interop-overwrite-conflict.sh)"
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
            &format!("interop_overwrite_conflict_d1_{}", scenario.name),
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

        let outcome = overwrite_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision on the JAVA table must match the expected outcome",
            scenario.name
        );

        println!(
            "interop_overwrite_conflict D1 OK — scenario {}: Rust validated the Java table \
             (concurrent S1 y[{},{}]) → {:?} (expected {:?})",
            scenario.name,
            scenario.concurrent_lo,
            scenario.concurrent_hi,
            outcome,
            scenario.expected
        );
    }
}
