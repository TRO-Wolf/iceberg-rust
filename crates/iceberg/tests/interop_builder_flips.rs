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

//! BUILDER-FLIPS write-action interop (Wave 3) — the two "builder flip" capabilities of `DeleteFiles`
//! proven bidirectionally against Java `StreamingDelete` (1.10.0): `deleteFromRowFilter(Expression)`
//! (GAP_MATRIX row 135) and `caseSensitive(boolean)` (row 134). Both ride the SAME builder vehicle —
//! Java `DeleteFiles.deleteFromRowFilter(...).caseSensitive(...)` → `StreamingDelete` →
//! `MergingSnapshotProducer.deleteByRowFilter` / `ManifestFilterManager.{caseSensitive,
//! PartitionAndMetricsEvaluator}`, mirrored by Rust `DeleteFiles::delete_from_row_filter` /
//! `case_sensitive` → the shared `SnapshotProducer::resolve_filter_deletes` / `build_residual_evaluator`
//! → `ResidualEvaluator::of(spec, expr, caseSensitive)` + the strict/inclusive metrics evaluators. This
//! is the FIRST interop that exercises the by-row-filter delete RESOLUTION (which file set is removed),
//! not just a conflict ACCEPT/REJECT decision.
//!
//! ## Why `DeleteFiles.deleteFromRowFilter` is the right vehicle for BOTH rows
//!
//! The `case_sensitive(bool)` flag's ONLY observable effect is the column-name binding
//! `predicate.bind(schema, case_sensitive)` (Java `Binder.bind` / `caseSensitive`). On `DeleteFiles`,
//! `OverwriteFiles`, and `RowDelta` it feeds the SAME `bind(...)` call through the shared
//! `snapshot.rs` binding sites. `delete_from_row_filter` drives `resolve_filter_deletes` — the
//! most-shared binding site (it ALSO backs `OverwriteFiles.overwrite_by_row_filter`) — so proving the
//! flag load-bearing here exercises the exact code path the other actions reach, AND co-proves the
//! row-135 delete-by-filter RESOLUTION. The conflict-filter binding-site family
//! (`first_conflicting_file`, shared by `OverwriteFiles`/`RowDelta` `conflict_detection_filter`) is the
//! SAME `bind(schema, case_sensitive)` call reached from a different entry point; it is already pinned
//! by the 25 mutation-proven `case_sensitive` unit tests and the C1/C3 conflict-validation interop.
//!
//! ## The fixture
//!
//! A PARTITIONED V2 table `identity(category)`, schema `{1 id long, 2 Category string, 3 y long}` — the
//! MIXED-CASE column `Category` (capital `C`) is the case-sensitivity discriminator. Two real-parquet
//! data files: cat=`a` (y[10,20]) and cat=`b` (y[60,70]); a third helper file straddles `y=55` for the
//! partial-match scenario. Each engine GENERATES its own history, then runs the symmetric
//! `delete_from_row_filter(...).case_sensitive(...)` against the OTHER engine's table and asserts the
//! SAME hand-declared outcome (anti-circular).
//!
//! ## The scenario contract (hard-coded identically in Java + Rust)
//!
//! | scenario                  | filter (column ref) | case_sensitive | expected                       |
//! |---------------------------|---------------------|----------------|--------------------------------|
//! | `filter_delete_partition` | `Category == "a"`   | true (default) | survivors = {b} (DELETE cat=a) |
//! | `filter_keep_complement`  | `Category == "b"`   | true (default) | survivors = {a} (DELETE cat=b, KEEP cat=a) |
//! | `filter_partial_error`    | `y >= 55`           | true (default) | ERROR — partial (cat=b straddles) |
//! | `case_insensitive_match`  | `category == "a"`   | FALSE          | survivors = {b} (binds case-insensitively) |
//! | `case_sensitive_reject`   | `category == "a"`   | true (default) | ERROR — wrong-case bind fails  |
//!
//! - `filter_delete_partition` is row 135's headline: DELETE the strictly-covered partition (residual
//!   `alwaysTrue` for cat=a), KEEP the disjoint partition (residual `alwaysFalse` for cat=b).
//! - `filter_keep_complement` is the SYMMETRIC partition-residual proof: `Category == "b"` DELETES cat=b
//!   (residual `alwaysTrue`), KEEPS cat=a (residual `alwaysFalse`). survivors = {a} — the complement of
//!   `filter_delete_partition`, so the two together pin that the surviving set is driven by the FILTER
//!   (not a hard-coded "always keep b"). Partition-residual based ⇒ no column-metrics dependency.
//! - `filter_partial_error` proves the PARTIAL→error path (Java's "some, but not all, rows match"). The
//!   `y >= 55` filter against the straddling cat=b helper (y[40,60]: 40 < 55 ≤ 60) is some-but-not-all ⇒
//!   non-retryable error; the helper is appended for THIS scenario only.
//! - `case_insensitive_match` is row 134's load-bearing FALSE direction: a wrong-cased `category`
//!   binds to the schema's `Category` under `case_sensitive(false)` and deletes cat=a.
//! - `case_sensitive_reject` is row 134's boundary: the SAME wrong-cased `category` under the DEFAULT
//!   (`case_sensitive(true)`) fails to bind ⇒ ERROR, nothing deleted.
//!
//! ## Named divergence NOT in the scenario set (live-oracle finding, 2026-06-16)
//!
//! A row filter that matches NO live data file (empty resolved delete set) DIVERGES: Rust's
//! `SnapshotProducer` REJECTS the truly-empty commit (pinned by `delete_files.rs`
//! `test_delete_from_row_filter_matching_nothing_is_rejected_and_deletes_nothing`), whereas Java's
//! `StreamingDelete.deleteFromRowFilter` COMMITS a successful no-op Delete snapshot (nothing deleted).
//! This was surfaced by the live D2 verify and is deliberately KEPT OUT of the bidirectional scenario
//! set (it is an empty-commit edge case orthogonal to rows 134/135). The KEEP semantics are proven by
//! `filter_keep_complement` / `filter_delete_partition` (non-empty commits both engines agree on).
//!
//! ## DELETE/KEEP correctness signal (engine-independent)
//!
//! File paths are random per engine, so the surviving file set is expressed as the set of partition
//! CATEGORY VALUES still live after the delete (identity(category) ⇒ each file's category is its
//! partition value). `survivors = {b}` means "exactly the cat=b file(s) survive" — the row-135 "Rust
//! deletes EXACTLY Java's file set" contract, decoupled from path naming.
//!
//! ## Documented row-135 residue NOT exercised (by design)
//!
//! The inherited fail-safe `markedForDelete` divergence (a by-path delete of a file that ALSO partially
//! matches the row filter: Rust errors, Java deletes) is deliberately NOT built here — none of the
//! scenarios combine `delete_file(path)` with a partial-filter overlap on the SAME file. It remains
//! pinned by the unit test `delete_files.rs`
//! `test_delete_from_row_filter_bypath_and_partial_match_diverges_failsafe`.
//!
//! ## Directions
//!
//! - **D1 (Rust validates Java's table):** `test_rust_runs_builder_flips_on_java_tables`. Rust
//!   `register_table`s `<dir>/<scenario>/table` (Java-built) and runs the symmetric
//!   `delete_from_row_filter(...).case_sensitive(...)`; the outcome must equal the hand-declared
//!   expected.
//! - **D2 (Java validates Rust's table):** `test_builder_flips_gen_rust_writes_java_validatable_tables`
//!   writes `<dir>/<scenario>/rust_table`; Java's `verify-interop-builder-flips` loads it, runs the
//!   symmetric delete, and asserts the same outcome.
//!
//! GATED on env vars (both unset ⇒ clean no-ops; offline `cargo test` gate stays green):
//! - `ICEBERG_INTEROP_BUILDER_FLIPS_GEN_DIR` — Rust GEN (Rust writes the per-scenario tables)
//! - `ICEBERG_INTEROP_BUILDER_FLIPS_DIR`     — D1 comparison (Rust validates Java's tables)

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use iceberg::expr::{Predicate, Reference};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, Datum, FormatVersion, Literal, ManifestContentType, NestedField,
    PartitionKey, PartitionSpec, PrimitiveLiteral, PrimitiveType, Schema, SchemaRef, SortOrder,
    Struct, Transform, Type, UnboundPartitionSpec,
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
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

// ===========================================================================================
// The scenario contract — hand-declared IDENTICALLY in Java (BuilderFlipsOracle) and here.
// ===========================================================================================

/// The expected outcome of the symmetric `delete_from_row_filter(...).case_sensitive(...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Expected {
    /// The delete committed; EXACTLY the listed category values survive (identity(category)).
    Survivors(&'static [&'static str]),
    /// The delete failed (partial-match error, wrong-case bind, or empty no-op) and removed nothing.
    Error,
}

/// One builder-flips scenario: its on-disk directory name, the row-filter column reference + the
/// equality literal (or the `y >= 55` partial probe), the `case_sensitive` flag, whether the partial
/// straddling file is appended, and the hand-declared expected outcome (Java is the reference engine).
#[derive(Debug, Clone)]
struct Scenario {
    name: &'static str,
    /// The column the equality row-filter references (mixed-case discriminator for row 134).
    filter_column: &'static str,
    /// The string literal the equality row-filter tests `filter_column == filter_value`. `None` ⇒ the
    /// `y >= 55` partial-probe filter is used instead (the partial-match scenario).
    filter_value: Option<&'static str>,
    /// Java `caseSensitive(...)` / Rust `case_sensitive(...)`. The DEFAULT (`true`) is still set
    /// explicitly on both engines so the symmetric delete is unambiguous.
    case_sensitive: bool,
    /// Append the straddling cat=b `y[40,60]` helper file so a `y >= 55` filter is PARTIAL.
    with_partial_file: bool,
    expected: Expected,
}

/// The five scenarios. Java's `BuilderFlipsOracle.SCENARIOS` MUST match this set exactly (same names,
/// filters, flags, expected outcomes).
fn scenarios() -> Vec<Scenario> {
    vec![
        // Row 135 headline: `Category == "a"` strictly covers cat=a (residual alwaysTrue) ⇒ DELETE;
        // cat=b residual is alwaysFalse ⇒ KEEP. Survivor = {b}.
        Scenario {
            name: "filter_delete_partition",
            filter_column: "Category",
            filter_value: Some("a"),
            case_sensitive: true,
            with_partial_file: false,
            expected: Expected::Survivors(&["b"]),
        },
        // Symmetric partition-residual proof: `Category == "b"` DELETES cat=b (residual alwaysTrue),
        // KEEPS cat=a (residual alwaysFalse). Survivor = {a}; the complement of filter_delete_partition,
        // so the two together pin that the surviving set tracks the FILTER (not a hard-coded "keep b").
        Scenario {
            name: "filter_keep_complement",
            filter_column: "Category",
            filter_value: Some("b"),
            case_sensitive: true,
            with_partial_file: false,
            expected: Expected::Survivors(&["a"]),
        },
        // PARTIAL→error: with the straddling cat=b `y[40,60]` helper present, `y >= 55` matches SOME
        // but not all of its rows ⇒ non-retryable "some, but not all, rows match" error.
        Scenario {
            name: "filter_partial_error",
            filter_column: "y",
            filter_value: None,
            case_sensitive: true,
            with_partial_file: true,
            expected: Expected::Error,
        },
        // Row 134 FALSE direction: wrong-cased `category` (schema column is `Category`) binds
        // case-insensitively under `case_sensitive(false)` ⇒ deletes cat=a. Survivor = {b}.
        Scenario {
            name: "case_insensitive_match",
            filter_column: "category",
            filter_value: Some("a"),
            case_sensitive: false,
            with_partial_file: false,
            expected: Expected::Survivors(&["b"]),
        },
        // Row 134 boundary: the SAME wrong-cased `category` under the DEFAULT (`case_sensitive(true)`)
        // fails to bind ⇒ ERROR, nothing deleted.
        Scenario {
            name: "case_sensitive_reject",
            filter_column: "category",
            filter_value: Some("a"),
            case_sensitive: true,
            with_partial_file: false,
            expected: Expected::Error,
        },
    ]
}

// ===========================================================================================
// Env-var gates.
// ===========================================================================================

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_BUILDER_FLIPS_GEN_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn validate_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_BUILDER_FLIPS_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// ===========================================================================================
// Schema + partitioned-table + real-parquet helpers (V2 identity(Category), {id, Category, y}).
// ===========================================================================================

/// The fixture schema `{1 id long, 2 Category string, 3 y long}` — `Category` is MIXED-CASE on purpose
/// (the case-sensitivity discriminator for row 134).
fn flips_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "Category", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("build the {id long, Category string, y long} schema")
}

/// `identity(Category)` unbound partition spec, spec id 0.
fn flips_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "Category".to_string(), Transform::Identity)
        .expect("add identity(Category) partition field")
        .build()
}

/// Build the `PartitionKey` for `Category = <value>` over the bound spec + schema.
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

/// Create a PARTITIONED V2 `identity(Category)` `{id, Category, y}` table at `table_location`.
async fn create_flips_table(catalog: &impl Catalog, table_location: &str) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(flips_schema())
        .partition_spec(flips_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create builder-flips rust_table")
}

/// Write a REAL parquet `{id, Category, y}` DATA file for ONE partition via the production
/// `DataFileWriter`. The `y` values become the file's parquet column bounds — the discriminating input
/// for the strict/inclusive metrics evaluators (DELETE vs PARTIAL on the `y` probe).
async fn write_partition_file(
    table: &Table,
    partition_key: &PartitionKey,
    category: &str,
    ids: Vec<i64>,
    ys: Vec<i64>,
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let row_count = ids.len();
    let categories: Vec<&str> = std::iter::repeat_n(category, row_count).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(categories)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
    ])
    .expect("build the per-partition {id, Category, y} data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "fdata".to_string(),
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

/// Build the per-scenario history and return the committed table. Always appends cat=a (y[10,20]) and
/// cat=b (y[60,70]); for the partial-match scenario ALSO appends a straddling cat=b file (y[40,60]) so
/// the `y >= 55` probe matches SOME but not all of partition b's rows.
async fn build_scenario_table(catalog: &impl Catalog, table: Table, scenario: &Scenario) -> Table {
    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().as_ref().clone();

    let key_a = partition_key(schema.clone(), spec.clone(), "a");
    let key_b = partition_key(schema.clone(), spec.clone(), "b");

    let file_a = write_partition_file(&table, &key_a, "a", vec![1, 2], vec![10, 20]).await;
    let file_b = write_partition_file(&table, &key_b, "b", vec![3, 4], vec![60, 70]).await;
    let mut files = vec![file_a, file_b];
    if scenario.with_partial_file {
        // A SECOND cat=b file whose y straddles 55 (y[40,60]) so `y >= 55` is a PARTIAL match.
        let file_b_straddle =
            write_partition_file(&table, &key_b, "b", vec![5, 6], vec![40, 60]).await;
        files.push(file_b_straddle);
    }
    append_files(catalog, &table, files).await
}

/// The `y` metrics-probe threshold (`y >= Y_THRESHOLD`), used only by `filter_partial_error`. Against
/// the straddling cat=b helper (y[40,60]: 40 < 55 ≤ 60) it is a PARTIAL match (some-but-not-all rows)
/// ⇒ the non-retryable "some, but not all, rows match" error. MUST match the Java oracle's Y_THRESHOLD.
const Y_THRESHOLD: i64 = 55;

/// Build the scenario's row filter: `filter_column == filter_value` for an equality scenario, or the
/// `y >= Y_THRESHOLD` metrics probe when `filter_value` is `None`.
fn scenario_filter(scenario: &Scenario) -> Predicate {
    match scenario.filter_value {
        Some(value) => Reference::new(scenario.filter_column).equal_to(Datum::string(value)),
        None => Reference::new(scenario.filter_column)
            .greater_than_or_equal_to(Datum::long(Y_THRESHOLD)),
    }
}

/// The set of live DATA-file CATEGORY values in the table's current snapshot — the engine-independent
/// DELETE/KEEP signal (identity(Category) ⇒ each file's category is its partition value). Reads the
/// partition struct's single string field.
async fn live_categories(table: &Table) -> HashSet<String> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list loads");
    let mut categories = HashSet::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let value = match entry.data_file().partition().fields().first() {
                Some(Some(Literal::Primitive(PrimitiveLiteral::String(s)))) => s.clone(),
                _ => panic!("identity(Category) string partition value present"),
            };
            categories.insert(value);
        }
    }
    categories
}

/// Run the SYMMETRIC `delete_from_row_filter(filter).case_sensitive(flag)` against `table` and return
/// the OBSERVED outcome — `Survivors(<surviving category set>)` on commit, `Error` on any failure
/// (partial-match, wrong-case bind, or empty no-op rejection). The table reference reloads via the
/// catalog after a failure so the surviving set is read from the persisted (unchanged) state.
async fn flips_outcome(
    catalog: &impl Catalog,
    table: &Table,
    scenario: &Scenario,
) -> ObservedOutcome {
    let filter = scenario_filter(scenario);
    let tx = Transaction::new(table);
    let action = tx
        .delete_files()
        .case_sensitive(scenario.case_sensitive)
        .delete_from_row_filter(filter);
    let tx = action.apply(tx).expect("apply delete_from_row_filter");

    match tx.commit(catalog).await {
        Ok(committed) => {
            let survivors: Vec<String> = {
                let mut v: Vec<String> = live_categories(&committed).await.into_iter().collect();
                v.sort();
                v
            };
            ObservedOutcome::Survivors(survivors)
        }
        Err(_) => ObservedOutcome::Error,
    }
}

/// The observed outcome of the symmetric delete, normalized to the comparable form (sorted category
/// list, or `Error`).
#[derive(Debug, Clone, PartialEq, Eq)]
enum ObservedOutcome {
    Survivors(Vec<String>),
    Error,
}

/// Compare an observed outcome against the scenario's hand-declared expected outcome.
fn matches_expected(observed: &ObservedOutcome, expected: &Expected) -> bool {
    match (observed, expected) {
        (ObservedOutcome::Survivors(observed_set), Expected::Survivors(expected_set)) => {
            let mut want: Vec<String> = expected_set.iter().map(|s| s.to_string()).collect();
            want.sort();
            observed_set == &want
        }
        (ObservedOutcome::Error, Expected::Error) => true,
        _ => false,
    }
}

// ===========================================================================================
// D2 GEN — Rust writes the per-scenario tables for Java's verify to validate.
// ===========================================================================================

/// Rust builds each scenario's `<gen_dir>/<scenario>/rust_table` and lands `final.metadata.json`. The
/// sanity check confirms Rust's OWN outcome matches the scenario's hand-declared expected BEFORE handing
/// the table to Java — so a Rust-side regression is caught here, not silently shipped to the Java verify.
#[tokio::test]
async fn test_builder_flips_gen_rust_writes_java_validatable_tables() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_builder_flips GEN — set ICEBERG_INTEROP_BUILDER_FLIPS_GEN_DIR \
             (run dev/java-interop/run-interop-builder-flips.sh)"
        );
        return;
    };

    for scenario in scenarios() {
        let scenario_dir = gen_dir.join(scenario.name);
        let warehouse = scenario_dir.to_string_lossy().to_string();
        let table_location = format!("{warehouse}/rust_table");
        let catalog = build_catalog(
            &format!("interop_builder_flips_gen_{}", scenario.name),
            &warehouse,
        )
        .await;

        let table = create_flips_table(&catalog, &table_location).await;
        let table = build_scenario_table(&catalog, table, &scenario).await;

        // Land final metadata at the known path BEFORE the sanity delete (which would otherwise write
        // further metadata versions). final.metadata.json reflects the clean pre-delete history.
        let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
        table
            .metadata()
            .clone()
            .write_to(table.file_io(), &final_metadata_path)
            .await
            .expect("write final.metadata.json");

        // Sanity: Rust's OWN outcome must equal the hand-declared expected outcome.
        let observed = flips_outcome(&catalog, &table, &scenario).await;
        assert!(
            matches_expected(&observed, &scenario.expected),
            "scenario {}: Rust's own outcome {:?} must match the hand-declared expected {:?}",
            scenario.name,
            observed,
            scenario.expected
        );

        println!(
            "interop_builder_flips GEN OK — scenario {} wrote {table_location}; \
             Rust outcome = {:?} (expected {:?})",
            scenario.name, observed, scenario.expected
        );
    }
}

// ===========================================================================================
// D1 — Rust validates the JAVA-written tables (register + symmetric delete_from_row_filter).
// ===========================================================================================

/// Rust registers each JAVA-written `<dir>/<scenario>/table` into a `MemoryCatalog` and runs the
/// symmetric `delete_from_row_filter(...).case_sensitive(...)`; the outcome must equal the scenario's
/// hand-declared expected. This is DIRECTION 1: Rust's resolution runs against Java's exact on-disk
/// manifests + parquet stats. An ACCEPT commit writes orphan `vN.metadata.json` files but never the
/// fixed-name `final.metadata.json`, so the fixture stays re-run-safe.
#[tokio::test]
async fn test_rust_runs_builder_flips_on_java_tables() {
    let Some(dir) = validate_dir() else {
        println!(
            "skipping interop_builder_flips D1 — set ICEBERG_INTEROP_BUILDER_FLIPS_DIR \
             (run dev/java-interop/run-interop-builder-flips.sh)"
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
            &format!("interop_builder_flips_d1_{}", scenario.name),
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
            .expect("register the Java-written builder-flips table");

        let observed = flips_outcome(&catalog, &table, &scenario).await;
        assert!(
            matches_expected(&observed, &scenario.expected),
            "scenario {}: Rust's outcome on the JAVA table {:?} must match the expected {:?}",
            scenario.name,
            observed,
            scenario.expected
        );

        println!(
            "interop_builder_flips D1 OK — scenario {}: Rust validated the Java table → \
             {:?} (expected {:?})",
            scenario.name, observed, scenario.expected
        );
    }
}
