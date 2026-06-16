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

//! CONFLICT-VALIDATION write-action interop (increment C2) — `DeleteFiles`'s SINGLE conflict axis,
//! `validate_files_exist`, proven against Java `StreamingDelete.validateFilesExist` →
//! `MergingSnapshotProducer` / `ManifestFilterManager.failMissingDeletePaths` on the conflict DECISION
//! (ACCEPT vs REJECT) over a concurrent-removal history. `DeleteFiles` is the SIMPLEST write-action
//! conflict unit (GAP_MATRIX row 93): one axis, no conflict filter. This is the SAME concurrent-removal
//! shape as C3's files-exist axis (axis 3 of `interop_rowdelta_conflict.rs`), just driven through
//! `DeleteFiles` instead of `RowDelta`.
//!
//! ## The single conflict axis
//!
//! `validate_files_exist()` — reject the commit if a data file THIS action is deleting was ALREADY
//! DELETED by a snapshot committed since the start snapshot; accept otherwise. Java
//! `StreamingDelete.validateFilesExist()` sets `validateFilesToDeleteExist`; at commit time
//! `validate()` calls `ManifestFilterManager.failMissingDeletePaths()`, so the re-based manifest filter
//! raises a non-retryable `ValidationException` ("Missing required files to delete") when a requested
//! delete path is no longer a live entry. Rust `DeleteFiles::validate_files_exist` →
//! `deleted_data_files_after` (the status-axis walk over `VALIDATE_DATA_FILES_EXIST_OPERATIONS`) raises
//! a non-retryable `DataInvalid` ("Cannot commit, missing data files"). Different enforcement
//! mechanism, IDENTICAL conflict decision: a concurrently-removed required file ⇒ REJECT.
//!
//! ## The insight that makes it tractable (same shape as C1/C3/C4)
//!
//! The check depends ONLY on the table's S0→S1 history (the concurrent removal) plus WHICH file the
//! symmetric delete targets — NOT on anything the symmetric delete adds (it adds nothing). So both
//! engines run the SAME symmetric `delete_files(<derived path>).validate_files_exist()` against the
//! OTHER engine's table and the ACCEPT/REJECT outcome is a pure function of the history + the target.
//! The expected outcome is HAND-DECLARED identically on both sides (anti-circular), with Java as the
//! reference engine.
//!
//! Because file names are random, the targeted path is DERIVED per engine from the loaded table:
//! `removed` = a data file live at S0 but tombstoned by S1; `survivor` = a data file live at BOTH S0
//! and S1. The reject scenario targets `removed`, the accept scenario targets `survivor` — both pure
//! functions of the loaded table, no cross-engine path coupling.
//!
//! ## The scenario contract (hard-coded identically in Java + Rust)
//!
//! Schema `{1 id long required, 2 y long required}`, UNPARTITIONED, V2 (like C1/C3).
//!
//! | scenario                | S1 concurrent commit             | targets   | expected |
//! |-------------------------|----------------------------------|-----------|----------|
//! | `same_file_reject`      | overwrite removes f0, adds f2    | f0 (gone) | REJECT   |
//! | `different_file_accept` | overwrite removes f0, adds f2    | f1 (live) | ACCEPT   |
//!
//! Both scenarios share the SAME generated history (S0 appends `f0`+`f1`; S1 overwrite removes `f0`,
//! adds `f2`). The reject scenario targets the REMOVED `f0`; the accept scenario targets the SURVIVING
//! `f1`. Both derived from the loaded table.
//!
//! ## Directions
//!
//! - **D1 (Rust validates Java's table):** `test_rust_validates_java_deletefiles_conflict_tables`. Rust
//!   `register_table`s `<dir>/<scenario>/table` (Java-built) into a `MemoryCatalog` and runs the
//!   symmetric `delete_files`; the outcome must equal the scenario's hand-declared expected. (An ACCEPT
//!   commit writes new `vN.metadata.json` orphans but never the fixed-name `final.metadata.json`, so
//!   this is re-run-safe.)
//! - **D2 (Java validates Rust's table):** `test_deletefiles_conflict_gen_rust_writes_java_validatable_tables`
//!   writes `<dir>/<scenario>/rust_table` (S0+S1); Java's `verify-interop-deletefiles-conflict` loads
//!   it, runs the symmetric `delete_files`, and asserts the same outcome.
//!
//! GATED on env vars (both unset ⇒ clean no-ops; offline `cargo test` gate stays green):
//! - `ICEBERG_INTEROP_DELETEFILES_CONFLICT_GEN_DIR` — Rust GEN (Rust writes the per-scenario tables)
//! - `ICEBERG_INTEROP_DELETEFILES_CONFLICT_DIR`     — D1 comparison (Rust validates Java's tables)

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, FormatVersion, ManifestContentType, NestedField, PrimitiveType,
    Schema, SortOrder, Type,
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
// The scenario contract — hand-declared IDENTICALLY in Java (DeleteFilesConflictOracle) and here.
// ===========================================================================================

/// The expected conflict-validation outcome of the symmetric `delete_files`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The delete committed — the targeted file was still live (no concurrent removal of it).
    Accept,
    /// The delete was rejected with a non-retryable `DataInvalid` (Java `ValidationException`).
    Reject,
}

/// One conflict scenario: its on-disk directory name and the hand-declared expected outcome (Java is
/// the reference engine). Both scenarios share the SAME S0→S1 history; the targeted path (removed vs
/// survivor) is DERIVED from the loaded table per `expected`.
#[derive(Debug, Clone, Copy)]
struct Scenario {
    name: &'static str,
    expected: Outcome,
}

/// The two scenarios (a reject + an accept false-positive guard). Java's
/// `DeleteFilesConflictOracle.SCENARIOS` MUST match this set exactly (same names, expected outcomes).
fn scenarios() -> Vec<Scenario> {
    vec![
        // Target the REMOVED f0 (live@S0, tombstoned by S1) ⇒ the required file is gone ⇒ REJECT.
        Scenario {
            name: "same_file_reject",
            expected: Outcome::Reject,
        },
        // Target the SURVIVING f1 (live@S0 and @S1) ⇒ the required file still exists ⇒ ACCEPT.
        Scenario {
            name: "different_file_accept",
            expected: Outcome::Accept,
        },
    ]
}

// ===========================================================================================
// Env-var gates.
// ===========================================================================================

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_DELETEFILES_CONFLICT_GEN_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn validate_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_DELETEFILES_CONFLICT_DIR")
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

/// Write a REAL parquet `{id, y}` DATA file via the production `DataFileWriter`. The `y` values become
/// the file's column bounds in parquet stats (inert for the files-exist axis, but keeps the on-disk
/// file shape identical to C1/C3 so the same Java reader path exercises it).
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

// ===========================================================================================
// History builder — S0 appends f0 + f1; S1 overwrite removes f0, adds f2 (shared by both scenarios).
// ===========================================================================================

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

/// Build the shared S0→S1 history and return the table at S1:
/// - S0: append two base files f0 (y[0,5]) + f1 (y[6,9]).
/// - S1: an OVERWRITE that REMOVES f0 (in the `{OVERWRITE}` default op set) and adds f2.
///
/// Identical to C3's files-exist history; the only difference is the validating action (`DeleteFiles`
/// here, `RowDelta` there).
async fn build_scenario_table(catalog: &impl Catalog, table: Table) -> Table {
    // S0: append two base files f0 (y[0,5]) + f1 (y[6,9]).
    let f0 = write_yid_file(&table, vec![1, 2], vec![0, 5]).await;
    let f1 = write_yid_file(&table, vec![3, 4], vec![6, 9]).await;
    let table = append_files(catalog, &table, vec![f0.clone(), f1]).await;
    // S1: an OVERWRITE that REMOVES f0 (in the `{OVERWRITE}` default op set) and adds f2.
    let f2 = write_yid_file(&table, vec![5, 6], vec![10, 12]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .overwrite_files()
        .add_file(f2)
        .delete_file(f0.file_path().to_string())
        .apply(tx)
        .expect("apply concurrent overwrite removal (S1)");
    tx.commit(catalog)
        .await
        .expect("commit concurrent overwrite removal S1")
}

/// Derive `validate_from_snapshot(S0)` — the ROOT snapshot (the one with no parent). After S0+S1 the
/// root is S0, so S1 counts as the concurrent commit.
fn root_snapshot_id(table: &Table) -> i64 {
    table
        .metadata()
        .snapshots()
        .find(|s| s.parent_snapshot_id().is_none())
        .expect("a root snapshot (S0) exists")
        .snapshot_id()
}

/// The set of live DATA file paths AS OF the snapshot `snapshot_id` (alive entries in its DATA
/// manifests). Used to derive the removed vs surviving path without coupling to the other engine's
/// file names.
async fn live_data_paths_at(table: &Table, snapshot_id: i64) -> HashSet<String> {
    let snapshot = table
        .metadata()
        .snapshot_by_id(snapshot_id)
        .expect("snapshot exists");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list loads");
    let mut live = HashSet::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        for entry in manifest.entries() {
            if entry.is_alive() {
                live.insert(entry.data_file().file_path().to_string());
            }
        }
    }
    live
}

/// Derive the data file path the symmetric delete targets. The reject scenario targets a file S1
/// REMOVED (live at S0, gone at S1); the accept scenario targets a file that SURVIVED (live at both S0
/// and S1). Both are pure functions of the loaded table — no cross-engine path coupling.
async fn targeted_path(table: &Table, scenario: &Scenario) -> String {
    let s0 = root_snapshot_id(table);
    // S1 = the concurrent commit = the single non-root snapshot. Derive it structurally (not from
    // `current_snapshot`) so the removed-vs-survivor diff stays stable and mirrors the Java oracle's
    // `concurrentSnapshotId` exactly.
    let s1 = table
        .metadata()
        .snapshots()
        .find(|s| s.snapshot_id() != s0)
        .expect("a concurrent snapshot (S1) exists")
        .snapshot_id();
    let live_s0 = live_data_paths_at(table, s0).await;
    let live_s1 = live_data_paths_at(table, s1).await;

    match scenario.expected {
        Outcome::Reject => {
            // A file live at S0 but tombstoned by S1 — the concurrently-removed file.
            live_s0
                .difference(&live_s1)
                .next()
                .cloned()
                .expect("a data file live at S0 was removed by S1 (the overwrite victim)")
        }
        Outcome::Accept => {
            // A file live at BOTH S0 and S1 — a survivor the overwrite did not touch.
            live_s0
                .intersection(&live_s1)
                .next()
                .cloned()
                .expect("a data file survived from S0 to S1 (the overwrite spared it)")
        }
    }
}

/// Run the SYMMETRIC `delete_files` against `table` and return ACCEPT/REJECT. The delete targets the
/// derived path, pins `validate_from_snapshot(S0)`, and enables `validate_files_exist()`:
/// - reject scenario: the targeted file was removed by S1 ⇒ the files-exist check rejects.
/// - accept scenario: the targeted file survives ⇒ the delete commits.
///
/// A non-retryable `DataInvalid` (Java `ValidationException`) is REJECT; a successful commit is ACCEPT.
async fn deletefiles_outcome(
    catalog: &impl Catalog,
    table: &Table,
    scenario: &Scenario,
) -> Outcome {
    let from = root_snapshot_id(table);
    let target = targeted_path(table, scenario).await;

    let tx = Transaction::new(table);
    let action = tx
        .delete_files()
        .delete_file(target)
        .validate_from_snapshot(from)
        .validate_files_exist();
    let tx = action.apply(tx).expect("apply delete_files");

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
            // NOTE on axis isolation (deliberately NOT asserting the message here). Rust has TWO
            // rejection paths for an already-removed target that RACE non-deterministically: the
            // `validate_files_exist` axis ("Cannot commit, missing data files") and an UNCONDITIONAL
            // by-path check in `process_deletes` ("Missing required files to delete"). So in this
            // pre-built-history structure the message is not a stable discriminator. The axis itself
            // is isolated where it CAN be isolated: D2 (Java validates Rust's table) flips REJECT→
            // ACCEPT when Java's `validateFilesExist` flag is stripped (Java gates the missing-file
            // check on the flag; Rust does not — a documented mechanism divergence), and the Rust
            // unit tests (`delete_files.rs` `test_delete_files_exist_*`) pin the axis message directly.
            // D1 here corroborates the DECISION (Rust agrees reject/accept with Java), not the axis.
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
async fn test_deletefiles_conflict_gen_rust_writes_java_validatable_tables() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_deletefiles_conflict GEN — set \
             ICEBERG_INTEROP_DELETEFILES_CONFLICT_GEN_DIR \
             (run dev/java-interop/run-interop-deletefiles-conflict.sh)"
        );
        return;
    };

    for scenario in scenarios() {
        let scenario_dir = gen_dir.join(scenario.name);
        let warehouse = scenario_dir.to_string_lossy().to_string();
        let table_location = format!("{warehouse}/rust_table");
        let catalog = build_catalog(
            &format!("interop_deletefiles_conflict_gen_{}", scenario.name),
            &warehouse,
        )
        .await;

        let table = create_conflict_table(&catalog, &table_location).await;
        let table = build_scenario_table(&catalog, table).await;

        // Land final metadata at the known path BEFORE the sanity delete (which would otherwise write
        // further metadata versions). final.metadata.json reflects the clean S0+S1 history.
        let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
        table
            .metadata()
            .clone()
            .write_to(table.file_io(), &final_metadata_path)
            .await
            .expect("write final.metadata.json");

        // Sanity: Rust's OWN conflict decision must equal the hand-declared expected outcome.
        let outcome = deletefiles_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision must match the hand-declared expected outcome",
            scenario.name
        );

        println!(
            "interop_deletefiles_conflict GEN OK — scenario {} wrote {table_location}; \
             Rust decision = {:?} (expected {:?})",
            scenario.name, outcome, scenario.expected
        );
    }
}

// ===========================================================================================
// D1 — Rust validates the JAVA-written tables (register + symmetric delete_files).
// ===========================================================================================

/// Rust registers each JAVA-written `<dir>/<scenario>/table` into a `MemoryCatalog` and runs the
/// symmetric `delete_files`; the conflict decision must equal the scenario's hand-declared expected
/// outcome. This is DIRECTION 1: Rust's validation runs against Java's exact on-disk manifests. The
/// ACCEPT commit writes orphan `vN.metadata.json` files but never the fixed-name `final.metadata.json`,
/// so the fixture stays re-run-safe.
#[tokio::test]
async fn test_rust_validates_java_deletefiles_conflict_tables() {
    let Some(dir) = validate_dir() else {
        println!(
            "skipping interop_deletefiles_conflict D1 — set ICEBERG_INTEROP_DELETEFILES_CONFLICT_DIR \
             (run dev/java-interop/run-interop-deletefiles-conflict.sh)"
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
            &format!("interop_deletefiles_conflict_d1_{}", scenario.name),
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

        let outcome = deletefiles_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision on the JAVA table must match the expected outcome",
            scenario.name
        );

        println!(
            "interop_deletefiles_conflict D1 OK — scenario {}: Rust validated the Java table → \
             {:?} (expected {:?})",
            scenario.name, outcome, scenario.expected
        );
    }
}
