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

//! CONFLICT-VALIDATION write-action interop (increment C5) — `RewriteFiles`'s conflict validation
//! (`validate_no_new_deletes_for_data_files`, run automatically in `validate()`, gated by the preserved
//! `data_sequence_number`) proven against Java `BaseRewriteFiles.validate` on the conflict DECISION
//! (ACCEPT vs REJECT) over a concurrent-commit history. `RewriteFiles` is the MECHANICALLY most complex
//! conflict unit (GAP_MATRIX row 95): the seq-preservation + position-vs-equality-delete nuance.
//!
//! ## The rule (Java `BaseRewriteFiles.validate` L135-142, the shared `validateNoNewDeletesForDataFiles`)
//!
//! A rewrite atomically REPLACES a set of data files with a new set. A CONCURRENT DELETE file added since
//! the rewrite's starting snapshot that would APPLY to one of the replaced data files is a conflict (the
//! rewrite would resurrect deleted rows) — UNLESS the rewrite PRESERVED the replaced file's data sequence
//! number (`data_sequence_number`), in which case a concurrently-added EQUALITY delete is IGNORED (it
//! still applies to the rewritten file via the preserved seq — Java L500-517) but a POSITION delete
//! remains FATAL (path-scoped, the new file has a new path — Java L538-543). The shared helper takes
//! `ignore_equality_deletes = self.data_sequence_number.is_some()` (Java L475-479,
//! `newDataFilesDataSequenceNumber != null`). This validation is UNCONDITIONAL (Java has no opt-in flag);
//! the only knob is whether the seq was preserved. A conflict is a non-retryable `DataInvalid` (Java's
//! non-retryable `ValidationException`).
//!
//! ## The insight that makes it tractable (same shape as C1/C2/C3/C4)
//!
//! The check depends ONLY on the table's S0→S1 history (the concurrent delete) plus WHETHER the rewrite
//! preserves the seq — NOT on the rewrite's own added file (a fresh A' written with the same rows). So
//! both engines run the SAME symmetric `rewrite_files([A], [A'])` (with/without `data_sequence_number`)
//! against the OTHER engine's table and the ACCEPT/REJECT outcome is a pure function of the history + the
//! seq knob. The expected outcome is HAND-DECLARED identically on both sides (anti-circular), Java being
//! the reference engine.
//!
//! Because file names are random, the replaced file A and the validation start snapshot S0 are DERIVED
//! per engine from the loaded table: A = the live data file in partition `x = 0` as of the ROOT snapshot
//! S0; A' is written fresh in the same partition with the same rows; S0 is the root snapshot (no parent).
//! No cross-engine path coupling — both engines read the same on-disk table.
//!
//! ## The scenario contract (hard-coded identically in Java + Rust)
//!
//! Schema `{1 x long, 2 y long, 3 z long}`, V2, PARTITIONED by `identity(x)` (the rewrite_files.rs
//! conflict unit-test fixture). S0 fast-appends A (x=0) + B (x=1); S1 is a CONCURRENT row_delta adding a
//! delete file. The symmetric rewrite replaces A with a fresh A'.
//!
//! | scenario                  | preserve seq | S1 concurrent delete            | applies to | expected |
//! |---------------------------|--------------|---------------------------------|------------|----------|
//! | `no_seq_eq_delete_reject` | NO           | EQUALITY delete in A's partition | A          | REJECT   |
//! | `seq_eq_delete_accept`    | YES          | EQUALITY delete in A's partition | A (ignored)| ACCEPT   |
//! | `seq_position_delete_reject`| YES        | POSITION delete path-scoped on A | A          | REJECT   |
//! | `disjoint_delete_accept`  | (either)     | POSITION delete path-scoped on B | B (not A)  | ACCEPT   |
//!
//! Scenarios 1 and 2 share the SAME S1 history (a concurrent equality delete on A) and differ ONLY by the
//! seq knob — the load-bearing seq-preservation nuance. Scenario 3 keeps the seq yet a POSITION delete is
//! still fatal. Scenario 4 is the over-firing negative control (a delete on the disjoint file B does not
//! conflict with replacing A). `disjoint_delete_accept` preserves the seq too, so the only reason it
//! ACCEPTS is the disjointness — not the seq knob.
//!
//! ## Why each REJECT is genuine (the C2 vacuous-reject guard)
//!
//! A is live at BOTH S0 and S1 (the concurrent delete ADDS a delete file; it does not REMOVE A), so the
//! by-path `failMissingDeletePaths` resolution (and the `Files to delete cannot be empty` precondition)
//! both PASS — the only thing that can reject is `validate_no_new_deletes_for_data_files`. The harness's
//! sabotage battery mutation-confirms this: disabling the concurrent-delete validation flips the reject
//! scenarios to accept (see dev/java-interop/run-interop-rewritefiles-conflict.sh).
//!
//! ## Directions
//!
//! - **D1 (Rust validates Java's table):** `test_rust_validates_java_rewritefiles_conflict_tables`. Rust
//!   `register_table`s `<dir>/<scenario>/table` (Java-built) into a `MemoryCatalog` and runs the
//!   symmetric `rewrite_files`; the outcome must equal the scenario's hand-declared expected. (An ACCEPT
//!   commit writes new `vN.metadata.json` orphans but never the fixed-name `final.metadata.json`, so this
//!   is re-run-safe.)
//! - **D2 (Java validates Rust's table):** `test_rewritefiles_conflict_gen_rust_writes_java_validatable_tables`
//!   writes `<dir>/<scenario>/rust_table` (S0+S1); Java's `verify-interop-rewritefiles-conflict` loads it,
//!   runs the symmetric `rewrite_files`, and asserts the same outcome.
//!
//! GATED on env vars (both unset ⇒ clean no-ops; offline `cargo test` gate stays green):
//! - `ICEBERG_INTEROP_REWRITEFILES_CONFLICT_GEN_DIR` — Rust GEN (Rust writes the per-scenario tables)
//! - `ICEBERG_INTEROP_REWRITEFILES_CONFLICT_DIR`     — D1 comparison (Rust validates Java's tables)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType,
    NestedField, PartitionKey, PrimitiveType, Schema, SortOrder, Struct, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation, TableIdent};

// ===========================================================================================
// The scenario contract — hand-declared IDENTICALLY in Java (RewriteFilesConflictOracle) and here.
// ===========================================================================================

/// Which kind of concurrent DELETE file S1 adds, and which data file it targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConcurrentDelete {
    /// An EQUALITY delete in A's partition (`x = 0`) keyed on `y`. Applies to A by partition.
    EqualityOnA,
    /// A POSITION delete path-scoped on A (`referenced_data_file == A`). Applies to A by path.
    PositionOnA,
    /// A POSITION delete path-scoped on the disjoint file B (`x = 1`). Does NOT apply to A.
    PositionOnB,
}

/// The expected conflict-validation outcome of the symmetric rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The rewrite committed — no conflicting concurrent delete applies to the replaced file.
    Accept,
    /// The rewrite was rejected with a non-retryable `DataInvalid` (Java `ValidationException`).
    Reject,
}

/// One conflict scenario: its on-disk directory name, whether the symmetric rewrite preserves the data
/// sequence number, the concurrent S1 delete, and the hand-declared expected outcome (Java is the
/// reference engine). Java's `RewriteFilesConflictOracle.SCENARIOS` MUST match this set exactly.
#[derive(Debug, Clone, Copy)]
struct Scenario {
    name: &'static str,
    preserve_seq: bool,
    concurrent: ConcurrentDelete,
    expected: Outcome,
}

/// The four scenarios proving the seq-preservation + position-vs-equality-delete nuance.
fn scenarios() -> Vec<Scenario> {
    vec![
        // 1. No seq preservation + a concurrent EQUALITY delete on A ⇒ any applicable delete is a
        //    conflict (a fresh-seq rewrite would resurrect rows) ⇒ REJECT.
        Scenario {
            name: "no_seq_eq_delete_reject",
            preserve_seq: false,
            concurrent: ConcurrentDelete::EqualityOnA,
            expected: Outcome::Reject,
        },
        // 2. Seq preserved + the SAME concurrent EQUALITY delete on A ⇒ ignored (it still applies to A'
        //    via the preserved seq, no resurrection) ⇒ ACCEPT.
        Scenario {
            name: "seq_eq_delete_accept",
            preserve_seq: true,
            concurrent: ConcurrentDelete::EqualityOnA,
            expected: Outcome::Accept,
        },
        // 3. Seq preserved + a concurrent POSITION delete on A ⇒ position deletes are ALWAYS fatal (the
        //    path target dies with the replaced file) ⇒ REJECT.
        Scenario {
            name: "seq_position_delete_reject",
            preserve_seq: true,
            concurrent: ConcurrentDelete::PositionOnA,
            expected: Outcome::Reject,
        },
        // 4. A concurrent POSITION delete on the DISJOINT file B ⇒ does not apply to A ⇒ ACCEPT (the
        //    over-firing negative control; seq preserved so disjointness is the only reason it accepts).
        Scenario {
            name: "disjoint_delete_accept",
            preserve_seq: true,
            concurrent: ConcurrentDelete::PositionOnB,
            expected: Outcome::Accept,
        },
    ]
}

// ===========================================================================================
// Env-var gates.
// ===========================================================================================

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REWRITEFILES_CONFLICT_GEN_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn validate_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_REWRITEFILES_CONFLICT_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// ===========================================================================================
// Schema + partition spec (V2, partitioned by identity(x); the rewrite_files.rs unit-test fixture).
// ===========================================================================================

/// The fixture schema `{1 x long, 2 y long, 3 z long}` (the V2 minimal table schema).
fn conflict_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(3, "z", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("build the {x long, y long, z long} schema")
}

/// `identity(x)` unbound partition spec, spec id 0.
fn conflict_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(1, "x".to_string(), Transform::Identity)
        .expect("add identity(x) partition field")
        .build()
}

/// Build the `PartitionKey` for `x = <value>` over the table's bound spec + schema.
fn partition_key(table: &Table, x: i64) -> PartitionKey {
    PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(x))]),
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

/// Create a PARTITIONED V2 `{x, y, z}` (identity(x)) table at `table_location`.
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

// ===========================================================================================
// Real-parquet writer helpers (production paths — partitioned by identity(x)).
// ===========================================================================================

/// Write a REAL parquet `{x, y, z}` DATA file for partition `x = part_value` via the production
/// `DataFileWriter`. Each row is `(part_value, y, z)`.
async fn write_data_file(table: &Table, part_value: i64, rows: &[(i64, i64)]) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));

    let xs: Vec<i64> = rows.iter().map(|_| part_value).collect();
    let ys: Vec<i64> = rows.iter().map(|(y, _)| *y).collect();
    let zs: Vec<i64> = rows.iter().map(|(_, z)| *z).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(xs)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
        Arc::new(Int64Array::from(zs)) as ArrayRef,
    ])
    .expect("build the {x, y, z} data batch");

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
        .build(Some(partition_key(table, part_value)))
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

/// Write a REAL parquet EQUALITY-DELETE file keyed on `y` (equality_ids=[2]) in partition `x = part_value`,
/// carrying the delete key `y = delete_y`. Mirrors the rewrite_files.rs unit-test eq-delete helper.
async fn write_equality_delete_file(table: &Table, part_value: i64, delete_y: i64) -> DataFile {
    use iceberg::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};

    let schema = table.metadata().current_schema().clone();
    let config = EqualityDeleteWriterConfig::new(vec![2], schema.clone())
        .expect("equality-delete writer config (equality_ids=[2], the y field)");
    let delete_schema =
        Arc::new(arrow_schema_to_schema(config.projected_arrow_schema_ref()).unwrap());

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "yeqdel".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        delete_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(Some(partition_key(table, part_value)))
        .await
        .expect("build partitioned equality-delete writer");

    // A FULL-schema {x, y, z} batch carrying the delete key; the projector keeps only `y`.
    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).expect("schema → arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(vec![part_value])) as ArrayRef,
        Arc::new(Int64Array::from(vec![delete_y])) as ArrayRef,
        Arc::new(Int64Array::from(vec![0_i64])) as ArrayRef,
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

/// Write a REAL parquet POSITION-DELETE file in partition `x = part_value`, deleting `(target_path, pos)`.
/// The position delete carries `referenced_data_file == target_path` (path-scoped). Mirrors the
/// rewrite_files.rs unit-test position-delete helper.
async fn write_position_delete_file(
    table: &Table,
    part_value: i64,
    target_path: &str,
    pos: i64,
) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "posdel".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
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
        .build(Some(partition_key(table, part_value)))
        .await
        .expect("build partitioned position-delete writer");

    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(vec![target_path.to_string()])) as ArrayRef,
        Arc::new(Int64Array::from(vec![pos])) as ArrayRef,
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

// ===========================================================================================
// History builder — S0 appends A (x=0) + B (x=1); S1 row_delta adds the scenario's concurrent delete.
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

/// Build the scenario's S0→S1 history and return the table at S1:
/// - S0: fast-append A (x=0, y=[10,20]) + B (x=1, y=[60,70]) — seq 1.
/// - S1: a CONCURRENT row_delta adding the scenario's delete file — seq 2.
///
/// The concurrent delete (equality on A, position on A, or position on the disjoint B) is what the
/// rewrite's `validate` walks against the start snapshot S0. A and B both live at S0 and at S1 (the
/// delete ADDS a delete file, it does not REMOVE A) so the by-path delete resolution always succeeds —
/// the only thing that can reject is the concurrent-delete conflict.
async fn build_scenario_table(catalog: &impl Catalog, table: Table, scenario: &Scenario) -> Table {
    // S0: A in partition x=0, B in partition x=1.
    let a = write_data_file(&table, 0, &[(10, 100), (20, 200)]).await;
    let b = write_data_file(&table, 1, &[(60, 600), (70, 700)]).await;
    let b_path = b.file_path().to_string();
    let table = append_files(catalog, &table, vec![a, b]).await;

    // S1: a CONCURRENT row_delta adding the scenario's delete file.
    let delete = match scenario.concurrent {
        ConcurrentDelete::EqualityOnA => write_equality_delete_file(&table, 0, 20).await,
        ConcurrentDelete::PositionOnA => {
            let a_path = live_data_path_in_partition(&table, 0).await;
            write_position_delete_file(&table, 0, &a_path, 1).await
        }
        ConcurrentDelete::PositionOnB => write_position_delete_file(&table, 1, &b_path, 0).await,
    };
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![delete])
        .apply(tx)
        .expect("apply concurrent row_delta add_deletes (S1)");
    tx.commit(catalog)
        .await
        .expect("commit concurrent delete S1")
}

/// The ROOT snapshot (the one with no parent). After S0+S1 the root is S0, so S1 counts as the
/// concurrent commit when the rewrite pins `validate_from_snapshot(S0)`.
fn root_snapshot_id(table: &Table) -> i64 {
    table
        .metadata()
        .snapshots()
        .find(|s| s.parent_snapshot_id().is_none())
        .expect("a root snapshot (S0) exists")
        .snapshot_id()
}

/// The data sequence number of the root snapshot S0 — the seq the seq-preserving rewrite stamps on A'
/// (so an outstanding equality delete still applies). Mirrors Java's `startingDataSequenceNumber`.
fn root_sequence_number(table: &Table) -> i64 {
    let s0 = root_snapshot_id(table);
    table
        .metadata()
        .snapshot_by_id(s0)
        .expect("root snapshot exists")
        .sequence_number()
}

/// The path of the (single) live DATA file in partition `x = part_value` as of the ROOT snapshot S0 —
/// the file the symmetric rewrite replaces (A for partition 0). Derived from the loaded table, so there
/// is no cross-engine path coupling.
async fn live_data_path_in_partition(table: &Table, part_value: i64) -> String {
    derive_data_file_in_partition(table, part_value)
        .await
        .file_path()
        .to_string()
}

/// Derive the (single) live DATA [`DataFile`] in partition `x = part_value` as of the ROOT snapshot S0.
/// This is the EXACT `DataFile` the symmetric rewrite passes to `files_to_delete` (so the by-path
/// resolution against the current snapshot matches and `data_sequence_number` derivation is consistent).
async fn derive_data_file_in_partition(table: &Table, part_value: i64) -> DataFile {
    let s0 = root_snapshot_id(table);
    let snapshot = table
        .metadata()
        .snapshot_by_id(s0)
        .expect("root snapshot exists");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list loads");
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        for entry in manifest.entries() {
            if !entry.is_alive() || entry.data_file().content_type() != DataContentType::Data {
                continue;
            }
            if entry.data_file().partition()
                == &Struct::from_iter([Some(Literal::long(part_value))])
            {
                return entry.data_file().clone();
            }
        }
    }
    panic!("no live data file in partition x={part_value} at S0");
}

/// Run the SYMMETRIC rewrite against `table` and return ACCEPT/REJECT. The rewrite replaces A (derived
/// from S0, partition 0) with a fresh A' (same rows, new path), pins `validate_from_snapshot(S0)`, and
/// preserves the data seq when the scenario asks for it:
/// - `no_seq_eq_delete_reject`: NO seq ⇒ any applicable delete is a conflict ⇒ the equality delete rejects.
/// - `seq_eq_delete_accept`: seq preserved ⇒ the equality delete is ignored ⇒ accept.
/// - `seq_position_delete_reject`: seq preserved but the POSITION delete is still fatal ⇒ reject.
/// - `disjoint_delete_accept`: the delete targets B, not A ⇒ accept.
///
/// A non-retryable `DataInvalid` (Java `ValidationException`) is REJECT; a successful commit is ACCEPT.
async fn rewrite_outcome(catalog: &impl Catalog, table: &Table, scenario: &Scenario) -> Outcome {
    let from = root_snapshot_id(table);
    // A = the live data file in partition x=0 at S0 (the file the rewrite replaces).
    let a = derive_data_file_in_partition(table, 0).await;
    // A' = a fresh file with the same rows, in the same partition (a new path).
    let a_prime = write_data_file(table, 0, &[(10, 100), (20, 200)]).await;

    let tx = Transaction::new(table);
    let mut action = tx
        .rewrite_files(vec![a], vec![a_prime])
        .validate_from_snapshot(from);
    if scenario.preserve_seq {
        action = action.data_sequence_number(root_sequence_number(table));
    }
    let tx = action.apply(tx).expect("apply rewrite_files");

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
            // Pin the conflict to `validate_no_new_deletes_for_data_files` (NOT a pre-empting by-path /
            // missing-file / precondition error): the message MUST be the concurrent-delete conflict.
            // This is the C2 vacuous-reject guard at the assertion level — A is live at S0 AND S1, so
            // `failMissingDeletePaths` and `Files to delete cannot be empty` both pass, leaving only the
            // concurrent-delete validation able to reject.
            assert!(
                e.message()
                    .contains("found new delete for replaced data file")
                    || e.message()
                        .contains("found new position delete for replaced data file"),
                "the REJECT must come from validate_no_new_deletes_for_data_files (the concurrent-delete \
                 conflict), not a pre-empting check; got: {e}"
            );
            Outcome::Reject
        }
    }
}

// ===========================================================================================
// D2 GEN — Rust writes the per-scenario tables for Java's verify to validate.
// ===========================================================================================

/// Rust builds each scenario's `<gen_dir>/<scenario>/rust_table` (S0+S1) and lands
/// `final.metadata.json`. The sanity check confirms Rust's OWN conflict decision matches the scenario's
/// hand-declared expected outcome before handing the table to Java — so a Rust-side regression is caught
/// here, not silently shipped to the Java verify.
#[tokio::test]
async fn test_rewritefiles_conflict_gen_rust_writes_java_validatable_tables() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_rewritefiles_conflict GEN — set \
             ICEBERG_INTEROP_REWRITEFILES_CONFLICT_GEN_DIR \
             (run dev/java-interop/run-interop-rewritefiles-conflict.sh)"
        );
        return;
    };

    for scenario in scenarios() {
        let scenario_dir = gen_dir.join(scenario.name);
        let warehouse = scenario_dir.to_string_lossy().to_string();
        let table_location = format!("{warehouse}/rust_table");
        let catalog = build_catalog(
            &format!("interop_rewritefiles_conflict_gen_{}", scenario.name),
            &warehouse,
        )
        .await;

        let table = create_conflict_table(&catalog, &table_location).await;
        let table = build_scenario_table(&catalog, table, &scenario).await;

        // Land final metadata at the known path BEFORE the sanity rewrite (which would otherwise write
        // further metadata versions). final.metadata.json reflects the clean S0+S1 history.
        let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
        table
            .metadata()
            .clone()
            .write_to(table.file_io(), &final_metadata_path)
            .await
            .expect("write final.metadata.json");

        // Sanity: Rust's OWN conflict decision must equal the hand-declared expected outcome.
        let outcome = rewrite_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision must match the hand-declared expected outcome",
            scenario.name
        );

        println!(
            "interop_rewritefiles_conflict GEN OK — scenario {} (preserve_seq={}, {:?}) wrote \
             {table_location}; Rust decision = {:?} (expected {:?})",
            scenario.name, scenario.preserve_seq, scenario.concurrent, outcome, scenario.expected
        );
    }
}

// ===========================================================================================
// D1 — Rust validates the JAVA-written tables (register + symmetric rewrite_files).
// ===========================================================================================

/// Rust registers each JAVA-written `<dir>/<scenario>/table` into a `MemoryCatalog` and runs the
/// symmetric `rewrite_files`; the conflict decision must equal the scenario's hand-declared expected
/// outcome. This is DIRECTION 1: Rust's validation runs against Java's exact on-disk manifests + delete
/// files. The ACCEPT commit writes orphan `vN.metadata.json` files but never the fixed-name
/// `final.metadata.json`, so the fixture stays re-run-safe.
#[tokio::test]
async fn test_rust_validates_java_rewritefiles_conflict_tables() {
    let Some(dir) = validate_dir() else {
        println!(
            "skipping interop_rewritefiles_conflict D1 — set ICEBERG_INTEROP_REWRITEFILES_CONFLICT_DIR \
             (run dev/java-interop/run-interop-rewritefiles-conflict.sh)"
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
            &format!("interop_rewritefiles_conflict_d1_{}", scenario.name),
            &warehouse,
        )
        .await;
        let namespace = NamespaceIdent::new("interop".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace for register");
        // The catalog derives the next metadata version from the registered file NAME, which must match
        // `<version>-<uuid>.metadata.json`. Java writes a fixed-name `final.metadata.json`, so register a
        // conventionally-named COPY (an ACCEPT commit then writes `<version+1>-…`; `final.metadata.json`
        // is left untouched, keeping the fixture re-run-safe).
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

        let outcome = rewrite_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision on the JAVA table must match the expected outcome",
            scenario.name
        );

        println!(
            "interop_rewritefiles_conflict D1 OK — scenario {} (preserve_seq={}, {:?}): Rust validated \
             the Java table → {:?} (expected {:?})",
            scenario.name, scenario.preserve_seq, scenario.concurrent, outcome, scenario.expected
        );
    }
}
