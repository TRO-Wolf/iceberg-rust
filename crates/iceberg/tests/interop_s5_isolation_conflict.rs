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

//! ENGINE_CONTRACT §5 isolation-recipe CROSS-ENGINE interop — the three §5 cells whose covering
//! conflict scenario was previously Rust-unit-level only (the named residue of the 2026-07-09
//! DRAFT→NORMATIVE flip):
//!
//! 1. **COW / snapshot** — `OverwriteFiles.validate_no_conflicting_deletes()` on a rewrite: a
//!    concurrent DELETE file applying to a REWRITTEN data file would be silently dropped with it
//!    ⇒ REJECT. Java `BaseOverwriteFiles.validate` → `validateNoNewDeletesForDataFiles`
//!    (`core/src/main/java/org/apache/iceberg/BaseOverwriteFiles.java` L174-177 at the
//!    `apache-iceberg-1.10.0` tag) — the `SparkWrite.commitWithSnapshotIsolation` recipe
//!    (`SparkWrite.java` L490-509).
//! 2. **Dynamic overwrite / snapshot** — `ReplacePartitions.validate_no_conflicting_deletes()`: a
//!    concurrent DELETE file landing in a REPLACED partition ⇒ REJECT. Java
//!    `BaseReplacePartitions.validate` → `validateNoNewDeleteFiles` over `replacedPartitions`
//!    (`BaseReplacePartitions.java` L99-108) — the `SparkWrite.DynamicOverwrite.commit` SNAPSHOT
//!    arm (`SparkWrite.java` L329-330).
//! 3. **Static overwrite-by-filter** — `OverwriteFiles.overwrite_by_row_filter(expr)`:
//!    - *snapshot*: `validate_no_conflicting_deletes()` ⇒ a concurrent DELETE file that can apply
//!      to rows matching the ROW FILTER ⇒ REJECT (`BaseOverwriteFiles.java` L168-172:
//!      `validateNoNewDeleteFiles(base, start, rowFilter)` when no explicit conflict filter).
//!    - *serializable*: above + `validate_no_conflicting_data()` where the ROW FILTER is the
//!      DEFAULT conflict-detection filter (`BaseOverwriteFiles.dataConflictDetectionFilter()`
//!      L181-189) ⇒ a concurrent DATA file matching the filter ⇒ REJECT — the
//!      `SparkWrite.OverwriteByFilter.commit` recipe (`SparkWrite.java` L371-375). This cell runs
//!      on the PARTITIONED shape with the partition-scoped row filter `category = "a"` so the
//!      validation is genuinely LOAD-BEARING (mutation-proven RED): with a metrics-decided filter
//!      the operation's own delete-by-row-filter resolution would mask a dropped validation in
//!      Rust — strict-metrics inequality currently NEVER proves a full match on columns without
//!      nan counts (`may_contain_nan` treats ABSENT as may-contain-NaN;
//!      `expr/visitors/strict_metrics_evaluator.rs` L105-111), diverging from Java
//!      `StrictMetricsEvaluator.canContainNaNs` (L483-486: absent ⇒ CANNOT) — a NAMED follow-up
//!      defect, out of this increment's file scope.
//!
//! ## The insight that makes it tractable (same shape as C1/C3/C4)
//!
//! Each cell's conflict decision depends ONLY on the table's S0→S1 history plus the symmetric
//! operation's validation config — NOT on the operation's own payload (the COW cell's rewritten
//! data file is DERIVED from the loaded table: the single live cat=a data file). So both engines
//! run the SAME symmetric operation against the OTHER engine's table; the expected ACCEPT/REJECT
//! is HAND-DECLARED identically on both sides (anti-circular), with Java as the reference engine.
//!
//! ## The scenario contract (hard-coded identically in Java `S5IsolationOracle` + here)
//!
//! Two fixture shapes, both V2:
//! - **P** (cells 1+2 and the serializable by-filter cell): `{1 id long, 2 category string}`
//!   PARTITIONED by `identity(category)`. S0: `fast_append` base files cat=a (id=1) + cat=b
//!   (id=2). S1: a CONCURRENT `row_delta` adds a partition-scoped EQUALITY-delete file (keyed on
//!   `id`) into ONE partition (cells 1+2), or a CONCURRENT `fast_append` adds a DATA file into
//!   ONE partition (serializable by-filter).
//! - **U** (the snapshot by-filter cell): `{1 id long, 2 y long}` UNPARTITIONED. S0:
//!   `fast_append` base A0 (y bounds `[0,5]`). S1: a CONCURRENT eq-DELETE keyed on `y` (bounds
//!   `[lo,hi]`).
//!
//! | scenario                   | cell                    | S1 concurrent commit      | expected |
//! |----------------------------|-------------------------|---------------------------|----------|
//! | `cow_delete_on_rewritten`  | COW snapshot            | eq-delete in cat=a        | REJECT   |
//! | `cow_delete_on_other`      | COW snapshot            | eq-delete in cat=b        | ACCEPT   |
//! | `dyn_delete_in_replaced`   | dynamic-overwrite snap. | eq-delete in cat=a        | REJECT   |
//! | `dyn_delete_in_other`      | dynamic-overwrite snap. | eq-delete in cat=b        | ACCEPT   |
//! | `byfilter_delete_matching` | by-filter snapshot      | eq-delete keyed y[60,70]  | REJECT   |
//! | `byfilter_delete_excluded` | by-filter snapshot      | eq-delete keyed y[10,20]  | ACCEPT   |
//! | `byfilter_data_matching`   | by-filter serializable  | DATA file in cat=a        | REJECT   |
//! | `byfilter_data_excluded`   | by-filter serializable  | DATA file in cat=b        | ACCEPT   |
//!
//! The symmetric operations (the §5 recipes, verbatim):
//! - COW snapshot: `overwrite_files().delete_data_files([<derived live cat=a data file>])
//!   .add_file(<fresh cat=a>).validate_from_snapshot(S0).validate_no_conflicting_deletes()`.
//! - Dynamic snapshot: `replace_partitions().add_file(<fresh cat=a>)
//!   .validate_from_snapshot(S0).validate_no_conflicting_deletes()`.
//! - By-filter snapshot: `overwrite_files().overwrite_by_row_filter(y >= 50)
//!   .add_file(<fresh y=1>).validate_from_snapshot(S0).validate_no_conflicting_deletes()`.
//! - By-filter serializable: `overwrite_files().overwrite_by_row_filter(category = "a")
//!   .add_file(<fresh cat=a>).validate_from_snapshot(S0).validate_no_conflicting_deletes()
//!   .validate_no_conflicting_data()` — deliberately NO explicit `conflict_detection_filter`,
//!   pinning the row-filter-as-default-conflict-filter contract cross-engine.
//!
//! ## Directions
//!
//! - **D1 (Rust validates Java's table):** `test_rust_validates_java_s5_isolation_tables`. Rust
//!   `register_table`s `<dir>/<scenario>/table` (Java-built) into a `MemoryCatalog` and runs the
//!   symmetric operation; the outcome must equal the scenario's hand-declared expected. (An ACCEPT
//!   commit writes new `vN.metadata.json` orphans but never the fixed-name `final.metadata.json`,
//!   so this is re-run-safe.)
//! - **D2 (Java validates Rust's table):** `test_s5_isolation_gen_rust_writes_java_validatable_tables`
//!   writes `<dir>/<scenario>/rust_table` (S0+S1); Java's `verify-interop-s5-isolation` loads it,
//!   runs the symmetric operation, and asserts the same outcome.
//!
//! GATED on env vars (both unset ⇒ clean no-ops; offline `cargo test` gate stays green):
//! - `ICEBERG_INTEROP_S5_ISOLATION_GEN_DIR` — Rust GEN (Rust writes the per-scenario tables)
//! - `ICEBERG_INTEROP_S5_ISOLATION_DIR`     — D1 comparison (Rust validates Java's tables)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use iceberg::expr::Reference;
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, Literal, ManifestContentType,
    NestedField, PartitionKey, PartitionSpec, PrimitiveType, Schema, SchemaRef, SortOrder, Struct,
    Transform, Type, UnboundPartitionSpec,
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
// The scenario contract — hand-declared IDENTICALLY in Java (S5IsolationOracle) and here.
// ===========================================================================================

/// Which ENGINE_CONTRACT §5 cell a scenario exercises (the three formerly unit-level-only cells;
/// the by-filter cell splits into its snapshot and serializable rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    /// COW / snapshot — `OverwriteFiles` rewrite + `validate_no_conflicting_deletes()`.
    CowSnapshot,
    /// Dynamic overwrite / snapshot — `ReplacePartitions` + `validate_no_conflicting_deletes()`.
    DynamicSnapshot,
    /// Static overwrite-by-filter / snapshot — `overwrite_by_row_filter(y >= 50)` +
    /// `validate_no_conflicting_deletes()`.
    ByFilterSnapshot,
    /// Static overwrite-by-filter / serializable — above + `validate_no_conflicting_data()` with
    /// the ROW FILTER as the default conflict-detection filter (no explicit filter set).
    ByFilterSerializable,
}

/// The expected conflict-validation outcome of the symmetric operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The operation committed — no conflicting concurrent commit for this cell.
    Accept,
    /// The operation was rejected with a non-retryable `DataInvalid` (Java `ValidationException`).
    Reject,
}

/// One §5-cell scenario: its on-disk directory name, the cell it exercises, the concurrent S1
/// commit's partition (P-shape cells) or `y` bounds (U-shape cells), and the hand-declared
/// expected outcome (Java is the reference engine).
#[derive(Debug, Clone, Copy)]
struct Scenario {
    name: &'static str,
    cell: Cell,
    /// The partition the concurrent S1 eq-delete lands in (P-shape cells); ignored for U-shape.
    concurrent_category: &'static str,
    /// The concurrent file's `y` lower bound (U-shape cells); ignored for P-shape.
    concurrent_lo: i64,
    /// The concurrent file's `y` upper bound (U-shape cells); ignored for P-shape.
    concurrent_hi: i64,
    expected: Outcome,
}

/// The eight scenarios — a reject + an accept (false-positive guard) per §5 cell. Java's
/// `S5IsolationOracle.SCENARIOS` MUST match this set exactly (same names, cells, partitions,
/// bounds, expected outcomes).
fn scenarios() -> Vec<Scenario> {
    vec![
        // Cell 1: COW / snapshot — a concurrent delete file on the REWRITTEN data file conflicts.
        Scenario {
            name: "cow_delete_on_rewritten",
            cell: Cell::CowSnapshot,
            concurrent_category: "a",
            concurrent_lo: 0,
            concurrent_hi: 0,
            expected: Outcome::Reject,
        },
        Scenario {
            name: "cow_delete_on_other",
            cell: Cell::CowSnapshot,
            concurrent_category: "b",
            concurrent_lo: 0,
            concurrent_hi: 0,
            expected: Outcome::Accept,
        },
        // Cell 2: dynamic overwrite / snapshot — a concurrent delete file in a REPLACED partition
        // conflicts.
        Scenario {
            name: "dyn_delete_in_replaced",
            cell: Cell::DynamicSnapshot,
            concurrent_category: "a",
            concurrent_lo: 0,
            concurrent_hi: 0,
            expected: Outcome::Reject,
        },
        Scenario {
            name: "dyn_delete_in_other",
            cell: Cell::DynamicSnapshot,
            concurrent_category: "b",
            concurrent_lo: 0,
            concurrent_hi: 0,
            expected: Outcome::Accept,
        },
        // Cell 3a: by-filter / snapshot — a concurrent delete file that can apply to rows matching
        // the ROW FILTER conflicts (inclusive-metrics narrowing on the delete file's y bounds).
        Scenario {
            name: "byfilter_delete_matching",
            cell: Cell::ByFilterSnapshot,
            concurrent_category: "",
            concurrent_lo: 60,
            concurrent_hi: 70,
            expected: Outcome::Reject,
        },
        Scenario {
            name: "byfilter_delete_excluded",
            cell: Cell::ByFilterSnapshot,
            concurrent_category: "",
            concurrent_lo: 10,
            concurrent_hi: 20,
            expected: Outcome::Accept,
        },
        // Cell 3b: by-filter / serializable — a concurrent DATA file matching the ROW FILTER
        // (`category = "a"`, the DEFAULT conflict-detection filter — no explicit filter is set)
        // conflicts.
        Scenario {
            name: "byfilter_data_matching",
            cell: Cell::ByFilterSerializable,
            concurrent_category: "a",
            concurrent_lo: 0,
            concurrent_hi: 0,
            expected: Outcome::Reject,
        },
        Scenario {
            name: "byfilter_data_excluded",
            cell: Cell::ByFilterSerializable,
            concurrent_category: "b",
            concurrent_lo: 0,
            concurrent_hi: 0,
            expected: Outcome::Accept,
        },
    ]
}

/// Whether a cell runs on the PARTITIONED `{id, category}` fixture (vs the unpartitioned
/// `{id, y}` one).
fn is_partitioned_cell(cell: Cell) -> bool {
    matches!(
        cell,
        Cell::CowSnapshot | Cell::DynamicSnapshot | Cell::ByFilterSerializable
    )
}

// ===========================================================================================
// Env-var gates.
// ===========================================================================================

fn gen_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_S5_ISOLATION_GEN_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn validate_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_S5_ISOLATION_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// ===========================================================================================
// Fixture shapes: P = partitioned {id, category} identity(category); U = unpartitioned {id, y}.
// ===========================================================================================

/// The P-shape schema `{1 id long required, 2 category string required}`.
fn schema_p() -> Schema {
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
fn spec_p() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category".to_string(), Transform::Identity)
        .expect("add identity(category) partition field")
        .build()
}

/// The U-shape schema `{1 id long required, 2 y long required}`.
fn schema_u() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("build the {id long, y long} schema")
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

/// Create the scenario's V2 `rust_table` at `table_location` — P-shape (partitioned) for the
/// COW/dynamic cells, U-shape (unpartitioned) for the by-filter cells.
async fn create_scenario_table(catalog: &impl Catalog, table_location: &str, cell: Cell) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = if is_partitioned_cell(cell) {
        TableCreation::builder()
            .name("rust_table".to_string())
            .location(table_location.to_string())
            .schema(schema_p())
            .partition_spec(spec_p())
            .sort_order(SortOrder::unsorted_order())
            .format_version(FormatVersion::V2)
            .build()
    } else {
        TableCreation::builder()
            .name("rust_table".to_string())
            .location(table_location.to_string())
            .schema(schema_u())
            .sort_order(SortOrder::unsorted_order())
            .format_version(FormatVersion::V2)
            .build()
    };
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create scenario rust_table")
}

// ===========================================================================================
// Real-parquet writers (production writer stack on both shapes).
// ===========================================================================================

/// Write a REAL parquet `{id, category}` data file for ONE identity(category) partition via the
/// production `DataFileWriter` (P-shape).
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
        "s5data".to_string(),
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

/// Write a REAL parquet PARTITION-SCOPED EQUALITY-delete file for the P-shape, keyed on field id
/// 1 (`id`), deleting the given `ids`, stamped with the caller's identity(category) partition —
/// the routing key the delete-conflict validations index on.
async fn write_category_eq_delete(
    table: &Table,
    partition_key: &PartitionKey,
    category: &str,
    ids: &[i64],
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
        .expect("equality-delete writer config (equality_ids = [1])");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "s5eqdel".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
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
        .build(Some(partition_key.clone()))
        .await
        .expect("build partition-scoped equality-delete writer");

    // A FULL-schema {id, category} batch carrying the delete keys; the projector keeps only `id`.
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("schema → arrow"));
    let categories: Vec<&str> = std::iter::repeat_n(category, ids.len()).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(categories)) as ArrayRef,
    ])
    .expect("build the partition-scoped equality-delete key batch");
    writer
        .write(batch)
        .await
        .expect("write equality-delete batch");
    writer
        .close()
        .await
        .expect("close equality-delete writer")
        .into_iter()
        .next()
        .expect("one equality-delete file")
}

/// Write a REAL parquet `{id, y}` DATA file via the production `DataFileWriter` (U-shape). The `y`
/// values become the file's column bounds in parquet stats — the inclusive-metrics inputs.
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
        "s5ydata".to_string(),
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

/// Write a REAL parquet UNPARTITIONED EQUALITY-delete file keyed on `y` (equality_ids=[2]),
/// carrying the two delete keys `y = lo` and `y = hi` so the file's parquet `y` column bounds are
/// `[lo,hi]` — the inclusive-metrics inputs the by-filter delete validation narrows on.
async fn write_y_eq_delete_file(table: &Table, lo: i64, hi: i64) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let config = EqualityDeleteWriterConfig::new(vec![2], schema.clone())
        .expect("equality-delete writer config (equality_ids=[2], the y field)");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "s5yeqdel".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
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

    // A FULL-schema {id, y} batch carrying the two delete keys; the projector keeps only `y`.
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
// History builders + commit helpers.
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

/// Commit a `row_delta` carrying only the given DELETE files and return the committed table.
async fn commit_row_delta_deletes(
    catalog: &impl Catalog,
    table: &Table,
    deletes: Vec<DataFile>,
) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(deletes)
        .apply(tx)
        .expect("apply row_delta add_deletes");
    tx.commit(catalog).await.expect("commit row_delta deletes")
}

/// Build the per-scenario table history and return the table at S1 (see the module docs):
/// - P-shape COW/dynamic cells: S0 base cat=a (id=1) + cat=b (id=2); S1 `row_delta` adds a
///   partition-scoped eq-delete into the scenario's partition.
/// - P-shape by-filter serializable: same S0; S1 `fast_append` a DATA file (id=100) into the
///   scenario's partition.
/// - U-shape by-filter snapshot: S0 base A0 y[0,5]; S1 `row_delta` adds a y-keyed eq-delete
///   with bounds `[lo,hi]`.
async fn build_scenario_table(catalog: &impl Catalog, table: Table, scenario: &Scenario) -> Table {
    match scenario.cell {
        Cell::CowSnapshot | Cell::DynamicSnapshot => {
            let schema = table.metadata().current_schema().clone();
            let bound_spec = table.metadata().default_partition_spec().as_ref().clone();
            let pk_a = partition_key(schema.clone(), bound_spec.clone(), "a");
            let pk_b = partition_key(schema.clone(), bound_spec.clone(), "b");

            // S0: base files cat=a (id=1) + cat=b (id=2).
            let base_a = write_category_file(&table, &pk_a, "a", vec![1]).await;
            let base_b = write_category_file(&table, &pk_b, "b", vec![2]).await;
            let table = append_files(catalog, &table, vec![base_a, base_b]).await;

            // S1: a CONCURRENT row_delta adds an eq-delete into the scenario's partition,
            // deleting that partition's base row (id=1 in cat=a, id=2 in cat=b).
            let cat = scenario.concurrent_category;
            let pk = partition_key(schema, bound_spec, cat);
            let deleted_id = if cat == "a" { 1 } else { 2 };
            let eq_delete = write_category_eq_delete(&table, &pk, cat, &[deleted_id]).await;
            assert_eq!(eq_delete.content_type(), DataContentType::EqualityDeletes);
            commit_row_delta_deletes(catalog, &table, vec![eq_delete]).await
        }
        Cell::ByFilterSerializable => {
            let schema = table.metadata().current_schema().clone();
            let bound_spec = table.metadata().default_partition_spec().as_ref().clone();
            let pk_a = partition_key(schema.clone(), bound_spec.clone(), "a");
            let pk_b = partition_key(schema.clone(), bound_spec.clone(), "b");

            // S0: base files cat=a (id=1) + cat=b (id=2).
            let base_a = write_category_file(&table, &pk_a, "a", vec![1]).await;
            let base_b = write_category_file(&table, &pk_b, "b", vec![2]).await;
            let table = append_files(catalog, &table, vec![base_a, base_b]).await;

            // S1: a CONCURRENT fast_append of a DATA file (id=100) into the scenario's partition.
            let cat = scenario.concurrent_category;
            let pk = partition_key(schema, bound_spec, cat);
            let concurrent = write_category_file(&table, &pk, cat, vec![100]).await;
            append_files(catalog, &table, vec![concurrent]).await
        }
        Cell::ByFilterSnapshot => {
            // S0: base file A0 — rows (1,0),(2,5) → y bounds [0,5].
            let file_a0 = write_yid_file(&table, vec![1, 2], vec![0, 5]).await;
            let table = append_files(catalog, &table, vec![file_a0]).await;
            // S1: a CONCURRENT row_delta adding an eq-delete keyed on y, bounds [lo,hi].
            let concurrent_delete =
                write_y_eq_delete_file(&table, scenario.concurrent_lo, scenario.concurrent_hi)
                    .await;
            commit_row_delta_deletes(catalog, &table, vec![concurrent_delete]).await
        }
    }
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

/// Derive the single LIVE cat=a DATA file from the loaded table — the file the COW cell's
/// symmetric overwrite rewrites. A pure function of the table (walk the current snapshot's data
/// manifests), so there is no cross-engine path coupling.
async fn find_live_category_a_data_file(table: &Table) -> DataFile {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("a current snapshot exists");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list loads");
    let expected_partition = Struct::from_iter([Some(Literal::string("a"))]);
    let mut found: Option<DataFile> = None;
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        for entry in manifest.entries() {
            if entry.is_alive()
                && entry.data_file().content_type() == DataContentType::Data
                && *entry.data_file().partition() == expected_partition
            {
                assert!(
                    found.is_none(),
                    "expected exactly ONE live cat=a data file, found a second: {}",
                    entry.data_file().file_path()
                );
                found = Some(entry.data_file().clone());
            }
        }
    }
    found.expect("a live cat=a data file exists (the S0 base file)")
}

// ===========================================================================================
// The symmetric §5-recipe operation, per cell.
// ===========================================================================================

/// Run the scenario's SYMMETRIC operation (the §5 recipe, verbatim) against `table` and return
/// ACCEPT/REJECT. A non-retryable `DataInvalid` (Java `ValidationException`) is REJECT; a
/// successful commit is ACCEPT.
async fn s5_outcome(catalog: &impl Catalog, table: &Table, scenario: &Scenario) -> Outcome {
    let from = root_snapshot_id(table);

    let tx = Transaction::new(table);
    let tx = match scenario.cell {
        Cell::CowSnapshot => {
            // The COW rewrite: remove the (derived) live cat=a data file, add a fresh cat=a file.
            let rewritten = find_live_category_a_data_file(table).await;
            let schema = table.metadata().current_schema().clone();
            let bound_spec = table.metadata().default_partition_spec().as_ref().clone();
            let pk_a = partition_key(schema, bound_spec, "a");
            let fresh = write_category_file(table, &pk_a, "a", vec![999]).await;
            tx.overwrite_files()
                .delete_data_files(vec![rewritten])
                .add_file(fresh)
                .validate_from_snapshot(from)
                .validate_no_conflicting_deletes()
                .apply(tx)
                .expect("apply COW overwrite")
        }
        Cell::DynamicSnapshot => {
            // The dynamic overwrite: replace partition cat=a with a fresh file.
            let schema = table.metadata().current_schema().clone();
            let bound_spec = table.metadata().default_partition_spec().as_ref().clone();
            let pk_a = partition_key(schema, bound_spec, "a");
            let fresh = write_category_file(table, &pk_a, "a", vec![999]).await;
            tx.replace_partitions()
                .add_file(fresh)
                .validate_from_snapshot(from)
                .validate_no_conflicting_deletes()
                .apply(tx)
                .expect("apply replace_partitions")
        }
        Cell::ByFilterSnapshot => {
            // The static filter overwrite, snapshot isolation: no explicit conflict filter — the
            // row filter itself scopes the delete validation.
            let fresh = write_yid_file(table, vec![999], vec![1]).await;
            tx.overwrite_files()
                .overwrite_by_row_filter(
                    Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
                )
                .add_file(fresh)
                .validate_from_snapshot(from)
                .validate_no_conflicting_deletes()
                .apply(tx)
                .expect("apply by-filter overwrite (snapshot)")
        }
        Cell::ByFilterSerializable => {
            // Serializable = the snapshot recipe + validate_no_conflicting_data(); deliberately NO
            // explicit conflict_detection_filter — the ROW FILTER (`category = "a"`) is the
            // default conflict filter. The static overwrite rewrites everything matching the
            // filter (the whole cat=a partition scope) and adds a fresh cat=a file.
            let schema = table.metadata().current_schema().clone();
            let bound_spec = table.metadata().default_partition_spec().as_ref().clone();
            let pk_a = partition_key(schema, bound_spec, "a");
            let fresh = write_category_file(table, &pk_a, "a", vec![999]).await;
            tx.overwrite_files()
                .overwrite_by_row_filter(Reference::new("category").equal_to(Datum::string("a")))
                .add_file(fresh)
                .validate_from_snapshot(from)
                .validate_no_conflicting_deletes()
                .validate_no_conflicting_data()
                .apply(tx)
                .expect("apply by-filter overwrite (serializable)")
        }
    };

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
async fn test_s5_isolation_gen_rust_writes_java_validatable_tables() {
    let Some(gen_dir) = gen_dir() else {
        println!(
            "skipping interop_s5_isolation GEN — set ICEBERG_INTEROP_S5_ISOLATION_GEN_DIR \
             (run dev/java-interop/run-interop-s5-isolation.sh)"
        );
        return;
    };

    for scenario in scenarios() {
        let scenario_dir = gen_dir.join(scenario.name);
        let warehouse = scenario_dir.to_string_lossy().to_string();
        let table_location = format!("{warehouse}/rust_table");
        let catalog = build_catalog(
            &format!("interop_s5_isolation_gen_{}", scenario.name),
            &warehouse,
        )
        .await;

        let table = create_scenario_table(&catalog, &table_location, scenario.cell).await;
        let table = build_scenario_table(&catalog, table, &scenario).await;

        // Land final metadata at the known path BEFORE the sanity operation (which would otherwise
        // write further metadata versions). final.metadata.json reflects the clean S0+S1 history.
        let final_metadata_path = format!("{table_location}/metadata/final.metadata.json");
        table
            .metadata()
            .clone()
            .write_to(table.file_io(), &final_metadata_path)
            .await
            .expect("write final.metadata.json");

        // Sanity: Rust's OWN conflict decision must equal the hand-declared expected outcome.
        let outcome = s5_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision must match the hand-declared expected outcome",
            scenario.name
        );

        println!(
            "interop_s5_isolation GEN OK — scenario {} ({:?}) wrote {table_location}; \
             Rust decision = {:?} (expected {:?})",
            scenario.name, scenario.cell, outcome, scenario.expected
        );
    }
}

// ===========================================================================================
// D1 — Rust validates the JAVA-written tables (register + symmetric §5 operation).
// ===========================================================================================

/// Rust registers each JAVA-written `<dir>/<scenario>/table` into a `MemoryCatalog` and runs the
/// scenario's symmetric operation; the conflict decision must equal the scenario's hand-declared
/// expected outcome. This is DIRECTION 1: Rust's validation runs against Java's exact on-disk
/// manifests + parquet metrics. The ACCEPT commit writes orphan `vN.metadata.json` files but never
/// the fixed-name `final.metadata.json`, so the fixture stays re-run-safe.
#[tokio::test]
async fn test_rust_validates_java_s5_isolation_tables() {
    let Some(dir) = validate_dir() else {
        println!(
            "skipping interop_s5_isolation D1 — set ICEBERG_INTEROP_S5_ISOLATION_DIR \
             (run dev/java-interop/run-interop-s5-isolation.sh)"
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
            &format!("interop_s5_isolation_d1_{}", scenario.name),
            &warehouse,
        )
        .await;
        let namespace = NamespaceIdent::new("interop".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace for register");
        // The catalog derives the next metadata version from the registered file NAME, which must
        // match `<version>-<uuid>.metadata.json`. Java writes a fixed-name `final.metadata.json`,
        // so register a conventionally-named COPY (an ACCEPT commit then writes `<version+1>-…`;
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
            .expect("register the Java-written s5-isolation table");

        let outcome = s5_outcome(&catalog, &table, &scenario).await;
        assert_eq!(
            outcome, scenario.expected,
            "scenario {}: Rust's conflict decision on the JAVA table must match the expected outcome",
            scenario.name
        );

        println!(
            "interop_s5_isolation D1 OK — scenario {} ({:?}): Rust validated the Java table → \
             {:?} (expected {:?})",
            scenario.name, scenario.cell, outcome, scenario.expected
        );
    }
}
