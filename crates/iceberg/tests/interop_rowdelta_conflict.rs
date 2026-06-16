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

//! CONFLICT-VALIDATION write-action interop (increment C3) — `RowDelta`'s THREE conflict axes proven
//! against Java `BaseRowDelta` on the conflict DECISION (ACCEPT vs REJECT) over a concurrent-commit
//! history. `RowDelta` is the RICHEST write-action conflict unit (GAP_MATRIX row 94): it is the
//! merge-on-read commit (adds DATA + DELETE files in one snapshot) and exposes three independent
//! conflict checks. This slice proves all three agree with Java for the same S0→S1 history.
//!
//! ## The three conflict axes
//!
//! 1. **`validate_no_conflicting_data_files()` + `conflict_detection_filter(F)`** — reject if a
//!    concurrent commit ADDED a DATA file matching `F` (inclusive-metrics, EXACTLY the C1
//!    `OverwriteFiles` slice's shape). Java `RowDelta.validateNoConflictingDataFiles` →
//!    `MergingSnapshotProducer.validateAddedDataFiles`.
//! 2. **`validate_no_conflicting_delete_files()` + filter** — reject if a concurrent commit ADDED a
//!    DELETE file matching `F` (RowDelta-specific). Java `RowDelta.validateNoConflictingDeleteFiles`
//!    → `MergingSnapshotProducer.validateNoNewDeleteFiles`. The concurrent DELETE is an EQUALITY
//!    delete keyed on `y` (equality_ids=[2]) so its real parquet carries the `y` bounds the
//!    inclusive-metrics evaluator reads — the SAME metrics narrowing as axis 1, on a delete file.
//! 3. **`validate_data_files_exist(paths)`** — reject if a concurrent commit REMOVED a data file the
//!    row-delta's deletes reference (concurrent-removal shape). Java
//!    `RowDelta.validateDataFilesExist` → `MergingSnapshotProducer.validateDataFilesExist`. The
//!    concurrent removal is an OVERWRITE (the `skipDeletes = true` DEFAULT op set `{OVERWRITE}`),
//!    so this rejects WITHOUT `validate_deleted_files()`.
//!
//! ## The insight that makes it tractable (same shape as C1/C4)
//!
//! Each axis's check depends ONLY on the table's S0→S1 history (the concurrent commit) plus the
//! symmetric row-delta's per-axis config — NOT on the row-delta's own ADDED payload. So both engines
//! run the SAME symmetric row-delta against the OTHER engine's table and the ACCEPT/REJECT outcome is
//! a pure function of the history + that config. The expected outcome is HAND-DECLARED identically on
//! both sides (anti-circular), with Java as the reference engine.
//!
//! For axis 3 the referenced path differs per engine (file names are random), so the validating
//! engine DERIVES it from the loaded table: `removed` = a data file live at S0 but tombstoned by S1;
//! `survivor` = a data file live at BOTH S0 and S1. The reject scenario references `removed`, the
//! accept scenario references `survivor` — both pure functions of the table, no cross-engine coupling.
//!
//! ## The scenario contract (hard-coded identically in Java + Rust)
//!
//! Schema `{1 id long required, 2 y long required}`, UNPARTITIONED, V2 (like C1).
//!
//! | scenario               | axis            | S1 concurrent commit                 | expected |
//! |------------------------|-----------------|--------------------------------------|----------|
//! | `data_conflict_reject` | data-conflict   | fast_append DATA y[60,70] (overlaps) | REJECT   |
//! | `data_conflict_accept` | data-conflict   | fast_append DATA y[10,20] (excluded) | ACCEPT   |
//! | `delete_conflict_reject`| delete-conflict| row_delta eq-delete on y[60,70]      | REJECT   |
//! | `delete_conflict_accept`| delete-conflict| row_delta eq-delete on y[10,20]      | ACCEPT   |
//! | `files_exist_reject`   | files-exist     | overwrite removes base f0 add f2     | REJECT   |
//! | `files_exist_accept`   | files-exist     | overwrite removes base f0 add f2     | ACCEPT   |
//!
//! The two `files_exist_*` scenarios share the SAME generated history (S0 appends `f0`+`f1`; S1
//! overwrite removes `f0`, adds `f2`). The reject scenario references the REMOVED `f0`; the accept
//! scenario references the SURVIVING `f1`. Both derived from the loaded table.
//!
//! The data/delete axes use the filter `y >= 50`: `[60,70]` overlaps (could match), `[10,20]` is
//! entirely below (the inclusive evaluator EXCLUDES it — the false-positive guard).
//!
//! ## Directions
//!
//! - **D1 (Rust validates Java's table):** `test_rust_validates_java_rowdelta_conflict_tables`. Rust
//!   `register_table`s `<dir>/<scenario>/table` (Java-built) into a `MemoryCatalog` and runs the
//!   symmetric row delta; the outcome must equal the scenario's hand-declared expected. (An ACCEPT
//!   commit writes new `vN.metadata.json` orphans but never the fixed-name `final.metadata.json`, so
//!   this is re-run-safe.)
//! - **D2 (Java validates Rust's table):** `test_rowdelta_conflict_gen_rust_writes_java_validatable_tables`
//!   writes `<dir>/<scenario>/rust_table` (S0+S1); Java's `verify-interop-rowdelta-conflict` loads it,
//!   runs the symmetric row delta, and asserts the same outcome.
//!
//! GATED on env vars (both unset ⇒ clean no-ops; offline `cargo test` gate stays green):
//! - `ICEBERG_INTEROP_ROWDELTA_CONFLICT_GEN_DIR` — Rust GEN (Rust writes the per-scenario tables)
//! - `ICEBERG_INTEROP_ROWDELTA_CONFLICT_DIR`     — D1 comparison (Rust validates Java's tables)

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use iceberg::expr::Reference;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, Datum, FormatVersion, ManifestContentType, NestedField,
    PrimitiveType, Schema, SortOrder, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation, TableIdent};

// ===========================================================================================
// The scenario contract — hand-declared IDENTICALLY in Java (RowDeltaConflictOracle) and here.
// ===========================================================================================

/// Which of `RowDelta`'s three conflict axes a scenario exercises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    /// `validate_no_conflicting_data_files()` + `conflict_detection_filter(y >= 50)` — a concurrent
    /// fast_append of a DATA file is the conflict (axis 1, the C1-shaped anchor).
    DataConflict,
    /// `validate_no_conflicting_delete_files()` + `conflict_detection_filter(y >= 50)` — a concurrent
    /// row_delta adding an equality-DELETE file (keyed on `y`) is the conflict (axis 2).
    DeleteConflict,
    /// `validate_data_files_exist([<referenced>])` — a concurrent OVERWRITE that REMOVED the
    /// referenced data file is the conflict (axis 3).
    FilesExist,
}

/// The expected conflict-validation outcome of the symmetric row delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The row delta committed — no conflicting concurrent commit under this axis.
    Accept,
    /// The row delta was rejected with a non-retryable `DataInvalid` (Java `ValidationException`).
    Reject,
}

/// One conflict scenario: its on-disk directory name, the axis it exercises, the concurrent S1
/// commit's `y` bounds (data/delete axes only), and the hand-declared expected outcome (Java is the
/// reference engine).
#[derive(Debug, Clone, Copy)]
struct Scenario {
    name: &'static str,
    axis: Axis,
    /// The concurrent file's `y` lower bound (data/delete axes); ignored for the files-exist axis.
    concurrent_lo: i64,
    /// The concurrent file's `y` upper bound (data/delete axes); ignored for the files-exist axis.
    concurrent_hi: i64,
    expected: Outcome,
}

/// The six scenarios — two per axis (a reject + an accept false-positive guard). Java's
/// `RowDeltaConflictOracle.SCENARIOS` MUST match this set exactly (same names, axes, bounds, expected
/// outcomes).
fn scenarios() -> Vec<Scenario> {
    vec![
        // Axis 1 (data-conflict, the C1-shaped anchor).
        Scenario {
            name: "data_conflict_reject",
            axis: Axis::DataConflict,
            concurrent_lo: 60,
            concurrent_hi: 70,
            expected: Outcome::Reject,
        },
        Scenario {
            name: "data_conflict_accept",
            axis: Axis::DataConflict,
            concurrent_lo: 10,
            concurrent_hi: 20,
            expected: Outcome::Accept,
        },
        // Axis 2 (delete-conflict, RowDelta-specific).
        Scenario {
            name: "delete_conflict_reject",
            axis: Axis::DeleteConflict,
            concurrent_lo: 60,
            concurrent_hi: 70,
            expected: Outcome::Reject,
        },
        Scenario {
            name: "delete_conflict_accept",
            axis: Axis::DeleteConflict,
            concurrent_lo: 10,
            concurrent_hi: 20,
            expected: Outcome::Accept,
        },
        // Axis 3 (files-exist, concurrent-removal shape). Bounds are inert; the referenced path
        // decides the outcome.
        Scenario {
            name: "files_exist_reject",
            axis: Axis::FilesExist,
            concurrent_lo: 0,
            concurrent_hi: 0,
            expected: Outcome::Reject,
        },
        Scenario {
            name: "files_exist_accept",
            axis: Axis::FilesExist,
            concurrent_lo: 0,
            concurrent_hi: 0,
            expected: Outcome::Accept,
        },
    ]
}

// ===========================================================================================
// Env-var gates.
// ===========================================================================================

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_ROWDELTA_CONFLICT_GEN_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn validate_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_ROWDELTA_CONFLICT_DIR")
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
/// the file's column bounds in parquet stats — the inputs the inclusive-metrics conflict check reads.
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

/// Write a REAL parquet EQUALITY-DELETE file keyed on `y` (equality_ids=[2]), carrying the two delete
/// keys `y = lo` and `y = hi` so the file's parquet `y` column bounds are `[lo,hi]` — the inputs the
/// inclusive-metrics conflict check reads when narrowing concurrently-added delete files by the
/// conflict filter. The delete-row schema is the single-column projection `{2 y long}`; Java derives
/// `equalityFieldIds = [2]` from `schema.select("y")` (same idiom).
async fn write_y_eq_delete_file(table: &Table, lo: i64, hi: i64) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let config = EqualityDeleteWriterConfig::new(vec![2], schema.clone())
        .expect("equality-delete writer config (equality_ids=[2], the y field)");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "yeqdel".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    // The parquet writer must use the PROJECTED schema (just `y`).
    let projected_iceberg_schema = Arc::new(
        iceberg::arrow::arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected arrow schema → iceberg schema"),
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        projected_iceberg_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(None)
        .await
        .expect("build equality-delete writer (keyed on y)");

    // A FULL-schema {id, y} batch carrying the two delete keys; the projector keeps only `y`. The two
    // `y` values bracket the delete file's parquet column bounds to exactly `[lo,hi]`.
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("schema → arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(vec![900_i64, 901])) as ArrayRef,
        Arc::new(Int64Array::from(vec![lo, hi])) as ArrayRef,
    ])
    .expect("build the y equality-delete key batch");
    writer.write(batch).await.expect("write eq-delete batch");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

// ===========================================================================================
// History builders — one per axis. Each returns the table at S1 (the validating engine pins S0).
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

/// Build the per-scenario table history for the data/delete/files-exist axes and return the table at
/// S1. The S0 history and the S1 concurrent commit differ by axis (see the module docs):
/// - data-conflict: S0 base file A0; S1 fast_append a DATA file `y[lo,hi]`.
/// - delete-conflict: S0 base file A0; S1 row_delta adds an EQUALITY delete keyed on `y[lo,hi]`.
/// - files-exist: S0 appends `f0` + `f1`; S1 overwrite removes `f0`, adds `f2`.
async fn build_scenario_table(catalog: &impl Catalog, table: Table, scenario: &Scenario) -> Table {
    match scenario.axis {
        Axis::DataConflict => {
            // S0: base file A0 — rows (1,0),(2,5) → y bounds [0,5].
            let file_a0 = write_yid_file(&table, vec![1, 2], vec![0, 5]).await;
            let table = append_files(catalog, &table, vec![file_a0]).await;
            // S1: the CONCURRENT DATA file — y bounds [lo,hi].
            let concurrent = write_yid_file(&table, vec![100, 101], vec![
                scenario.concurrent_lo,
                scenario.concurrent_hi,
            ])
            .await;
            append_files(catalog, &table, vec![concurrent]).await
        }
        Axis::DeleteConflict => {
            // S0: base file A0 — rows (1,0),(2,5) → y bounds [0,5].
            let file_a0 = write_yid_file(&table, vec![1, 2], vec![0, 5]).await;
            let table = append_files(catalog, &table, vec![file_a0]).await;
            // S1: a CONCURRENT row_delta adding an EQUALITY delete keyed on y, bounds [lo,hi].
            let concurrent_delete =
                write_y_eq_delete_file(&table, scenario.concurrent_lo, scenario.concurrent_hi)
                    .await;
            let tx = Transaction::new(&table);
            let tx = tx
                .row_delta()
                .add_deletes(vec![concurrent_delete])
                .apply(tx)
                .expect("apply concurrent row_delta add_deletes (S1)");
            tx.commit(catalog)
                .await
                .expect("commit concurrent delete S1")
        }
        Axis::FilesExist => {
            // S0: append two base files f0 (y[0,5]) + f1 (y[6,9]).
            let f0 = write_yid_file(&table, vec![1, 2], vec![0, 5]).await;
            let f1 = write_yid_file(&table, vec![3, 4], vec![6, 9]).await;
            let table = append_files(catalog, &table, vec![f0.clone(), f1]).await;
            // S1: an OVERWRITE that REMOVES f0 (in the {OVERWRITE} default op set) and adds f2.
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
    }
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
/// manifests). Used by the files-exist axis to derive the removed vs surviving path without coupling
/// to the other engine's file names.
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

/// For the files-exist axis: derive the referenced data file path the symmetric row delta pins. The
/// reject scenario references a file S1 REMOVED (live at S0, gone at S1); the accept scenario
/// references a file that SURVIVED (live at both S0 and S1). Both are pure functions of the loaded
/// table — no cross-engine path coupling.
async fn files_exist_referenced_path(table: &Table, scenario: &Scenario) -> String {
    let s0 = root_snapshot_id(table);
    let s1 = table
        .metadata()
        .current_snapshot()
        .expect("a current snapshot (S1) exists")
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

/// Run the SYMMETRIC row delta against `table` and return ACCEPT/REJECT. The row delta adds one fresh
/// local DELETE file (irrelevant to the conflict decision), pins `validate_from_snapshot(S0)`, and
/// enables the axis's validation:
/// - data-conflict: `validate_no_conflicting_data_files()` + `conflict_detection_filter(y >= 50)`.
/// - delete-conflict: `validate_no_conflicting_delete_files()` + `conflict_detection_filter(y >= 50)`.
/// - files-exist: `validate_data_files_exist([<derived path>])`.
///
/// A non-retryable `DataInvalid` (Java `ValidationException`) is REJECT; a successful commit is ACCEPT.
async fn rowdelta_outcome(catalog: &impl Catalog, table: &Table, scenario: &Scenario) -> Outcome {
    let from = root_snapshot_id(table);
    // The row delta's own added delete file is irrelevant to the conflict decision; a fresh eq-delete.
    let fresh_delete = write_y_eq_delete_file(table, 1, 1).await;

    let tx = Transaction::new(table);
    let action = tx.row_delta().add_deletes(vec![fresh_delete]);
    let action = match scenario.axis {
        Axis::DataConflict => action
            .validate_from_snapshot(from)
            .validate_no_conflicting_data_files()
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            ),
        Axis::DeleteConflict => action
            .validate_from_snapshot(from)
            .validate_no_conflicting_delete_files()
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            ),
        Axis::FilesExist => {
            let referenced = files_exist_referenced_path(table, scenario).await;
            action
                .validate_from_snapshot(from)
                .validate_data_files_exist([referenced])
        }
    };
    let tx = action.apply(tx).expect("apply row delta");

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
async fn test_rowdelta_conflict_gen_rust_writes_java_validatable_tables() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_rowdelta_conflict GEN — set \
             ICEBERG_INTEROP_ROWDELTA_CONFLICT_GEN_DIR \
             (run dev/java-interop/run-interop-rowdelta-conflict.sh)"
        );
        return;
    };

    for scenario in scenarios() {
        let scenario_dir = gen_dir.join(scenario.name);
        let warehouse = scenario_dir.to_string_lossy().to_string();
        let table_location = format!("{warehouse}/rust_table");
        let catalog = build_catalog(
            &format!("interop_rowdelta_conflict_gen_{}", scenario.name),
            &warehouse,
        )
        .await;

        let table = create_conflict_table(&catalog, &table_location).await;
        let table = build_scenario_table(&catalog, table, &scenario).await;

        // Land final metadata at the known path BEFORE the sanity row delta (which would otherwise
        // write further metadata versions). final.metadata.json reflects the clean S0+S1 history.
        let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
        table
            .metadata()
            .clone()
            .write_to(table.file_io(), &final_metadata_path)
            .await
            .expect("write final.metadata.json");

        // Sanity: Rust's OWN conflict decision must equal the hand-declared expected outcome.
        let outcome = rowdelta_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision must match the hand-declared expected outcome",
            scenario.name
        );

        println!(
            "interop_rowdelta_conflict GEN OK — scenario {} ({:?}) wrote {table_location}; \
             Rust decision = {:?} (expected {:?})",
            scenario.name, scenario.axis, outcome, scenario.expected
        );
    }
}

// ===========================================================================================
// D1 — Rust validates the JAVA-written tables (register + symmetric row delta).
// ===========================================================================================

/// Rust registers each JAVA-written `<dir>/<scenario>/table` into a `MemoryCatalog` and runs the
/// symmetric row delta; the conflict decision must equal the scenario's hand-declared expected
/// outcome. This is DIRECTION 1: Rust's validation runs against Java's exact on-disk manifests +
/// parquet metrics. The ACCEPT commit writes orphan `vN.metadata.json` files but never the fixed-name
/// `final.metadata.json`, so the fixture stays re-run-safe.
#[tokio::test]
async fn test_rust_validates_java_rowdelta_conflict_tables() {
    let Some(dir) = validate_dir() else {
        println!(
            "skipping interop_rowdelta_conflict D1 — set ICEBERG_INTEROP_ROWDELTA_CONFLICT_DIR \
             (run dev/java-interop/run-interop-rowdelta-conflict.sh)"
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
            &format!("interop_rowdelta_conflict_d1_{}", scenario.name),
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

        let outcome = rowdelta_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision on the JAVA table must match the expected outcome",
            scenario.name
        );

        println!(
            "interop_rowdelta_conflict D1 OK — scenario {} ({:?}): Rust validated the Java table → \
             {:?} (expected {:?})",
            scenario.name, scenario.axis, outcome, scenario.expected
        );
    }
}
