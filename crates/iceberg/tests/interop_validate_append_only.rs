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

//! SAME-COMMIT append-only ASSERTION interop (GAP_MATRIX row 144) — `ReplacePartitions`'
//! `validate_append_only()` proven 1:1 against Java `BaseReplacePartitions.validateAppendOnly()`.
//!
//! ## Why this is a behavior-equivalence (not a byte-level) interop
//!
//! `validateAppendOnly()` is a pure engine-agnostic **iceberg-core** API (NOT a Spark-surface
//! class), so the real Java action CAN be driven directly — but when the guard FIRES it throws and
//! nothing is written, so there is no on-disk artifact to round-trip. The 1:1 evidence for a
//! validation guard is therefore "**Java throws ⇔ Rust rejects** under IDENTICAL table+commit
//! shapes". Both engines run a self-contained battery of the SAME four cases and assert each
//! case's THROW/COMMIT outcome against the SAME HAND-DECLARED expectation (anti-circular: neither
//! side derives the other's expectation).
//!
//! - **D1 (Java drives the real core API):** `ValidateAppendOnlyOracle` in
//!   `dev/java-interop/.../InteropOracle.java` builds each case's table and runs
//!   `table.newReplacePartitions().addFile(..)[.validateAppendOnly()].commit()`, asserting THROW
//!   (`ManifestFilterManager$DeleteException`, a `ValidationException` — NON-retryable) vs COMMIT.
//! - **D2 (this Rust mirror):** builds the equivalent tables and runs
//!   `replace_partitions().add_file(..)[.validate_append_only()]`, asserting REJECT
//!   (`ErrorKind::DataInvalid`, non-retryable) vs COMMIT.
//!
//! ## The Java 1.10.0 contract (decoded from the iceberg-core jar via `javap -c`)
//!
//! `BaseReplacePartitions.validateAppendOnly()` → `MergingSnapshotProducer.failAnyDelete()` →
//! `ManifestFilterManager.failAnyDelete = true`. During `filterManifestWithDeletedFiles`, the moment
//! a live entry would be dropped while `failAnyDelete` is set, it throws
//! `new ManifestFilterManager$DeleteException(spec.partitionToPath(file.partition()))`.
//! `DeleteException extends ValidationException extends RuntimeException` — NOT
//! `CommitFailedException`, i.e. **non-retryable**. Rust mirrors this with a non-retryable
//! `ErrorKind::DataInvalid` raised the moment the resolved partition-deletes are non-empty.
//!
//! ## The four cases (hand-declared IDENTICALLY in Java + here)
//!
//! | case                         | table                    | replace add | flag | expected |
//! |------------------------------|--------------------------|-------------|------|----------|
//! | `matching_partition`         | partitioned, cat=a live  | cat=a       | YES  | THROW    |
//! | `empty_new_partition`        | partitioned, cat=a live  | cat=b       | YES  | COMMIT   |
//! | `unpartitioned_full_replace` | unpartitioned, non-empty | (full)      | YES  | THROW    |
//! | `matching_partition_no_flag` | partitioned, cat=a live  | cat=a       | NO   | COMMIT   |
//!
//! - `matching_partition`: the replace removes a live file in cat=a ⇒ THROW (not append-only).
//! - `empty_new_partition`: the replace fills a previously-empty partition (cat=b) ⇒ removes
//!   nothing ⇒ COMMIT (the false-positive guard: an over-eager assertion must not reject a pure add).
//! - `unpartitioned_full_replace`: an unpartitioned replace removes ALL existing files (Java
//!   `deleteByRowFilter(alwaysTrue)`) ⇒ THROW.
//! - `matching_partition_no_flag`: the SAME table+payload as `matching_partition` WITHOUT the flag ⇒
//!   the gate is open ⇒ COMMIT (proves the FLAG is what rejects, not the replace itself).
//!
//! GATED on `ICEBERG_INTEROP_VALIDATE_APPEND_ONLY_DIR` (unset ⇒ a clean offline no-op so the
//! `cargo test` gate stays green); set by `dev/java-interop/run-interop-validate-append-only.sh`,
//! which also drives the Java D1 oracle and a fail-closed sabotage battery.

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
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};

// ===========================================================================================
// The battery contract — hand-declared IDENTICALLY in Java (ValidateAppendOnlyOracle) and here.
// ===========================================================================================

/// The expected outcome of running the (possibly append-only) replace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expected {
    /// The commit must be REJECTED (Java throws `DeleteException`/`ValidationException` —
    /// non-retryable; Rust raises a non-retryable `DataInvalid`).
    Throw,
    /// The commit must SUCCEED (the replace is purely additive, or the flag is off).
    Commit,
}

/// One battery case: its name, whether the table is partitioned by `identity(category)`, whether the
/// `validate_append_only()` flag is set, the partition the replace's added file lands in (the
/// `category` value, or `None` for the unpartitioned table), and the hand-declared expected outcome.
#[derive(Debug, Clone, Copy)]
struct Case {
    name: &'static str,
    partitioned: bool,
    validate_append_only: bool,
    add_category: Option<&'static str>,
    expected: Expected,
}

/// The four cases. Java's `ValidateAppendOnlyOracle.CASES` MUST match this set exactly (same names,
/// same shapes, same expected outcomes). The expectation is HAND-DECLARED, never derived from Java.
fn cases() -> Vec<Case> {
    vec![
        // (A) replace a NON-EMPTY matching partition (cat=a) with the flag set -> a live file is
        //     removed -> THROW.
        Case {
            name: "matching_partition",
            partitioned: true,
            validate_append_only: true,
            add_category: Some("a"),
            expected: Expected::Throw,
        },
        // (B) replace a brand-new EMPTY partition (cat=b) with the flag set -> nothing removed (pure
        //     add) -> COMMIT (the false-positive guard).
        Case {
            name: "empty_new_partition",
            partitioned: true,
            validate_append_only: true,
            add_category: Some("b"),
            expected: Expected::Commit,
        },
        // (C) full-replace a NON-EMPTY UNPARTITIONED table with the flag set -> ALL files removed ->
        //     THROW.
        Case {
            name: "unpartitioned_full_replace",
            partitioned: false,
            validate_append_only: true,
            add_category: None,
            expected: Expected::Throw,
        },
        // (D) the SAME non-empty matching-partition replace WITHOUT the flag -> the gate is open ->
        //     COMMIT (proves the flag is what rejects).
        Case {
            name: "matching_partition_no_flag",
            partitioned: true,
            validate_append_only: false,
            add_category: Some("a"),
            expected: Expected::Commit,
        },
    ]
}

// ===========================================================================================
// Env-var gate.
// ===========================================================================================

fn run_dir() -> Option<PathBuf> {
    std::env::var_os("ICEBERG_INTEROP_VALIDATE_APPEND_ONLY_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

// ===========================================================================================
// Schema + spec + real-parquet helpers ({id long, category string}).
// ===========================================================================================

/// The fixture schema `{1 id long required, 2 category string required}` (matches the Java oracle).
fn schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, category string} schema")
}

/// `identity(category)` unbound partition spec, spec id 0 (for the partitioned cases).
fn partitioned_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "category".to_string(), Transform::Identity)
        .expect("add identity(category) partition field")
        .build()
}

/// The empty (unpartitioned) spec, spec id 0 (for the full-replace case).
fn unpartitioned_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder().with_spec_id(0).build()
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

/// Create a V2 `{id, category}` table at `table_location`, partitioned by `identity(category)` iff
/// `partitioned`.
async fn create_table(catalog: &impl Catalog, table_location: &str, partitioned: bool) -> Table {
    let namespace = NamespaceIdent::new("interop".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let spec = if partitioned {
        partitioned_spec()
    } else {
        unpartitioned_spec()
    };
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location.to_string())
        .schema(schema())
        .partition_spec(spec)
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table")
}

/// Build a `PartitionKey` for `category = <value>` over the bound spec + schema (partitioned cases).
fn partition_key(schema: SchemaRef, spec: PartitionSpec, category: &str) -> PartitionKey {
    PartitionKey::new(
        spec,
        schema,
        Struct::from_iter([Some(Literal::string(category))]),
    )
}

/// Write a REAL parquet `{id, category}` data file via the production `DataFileWriter`. When
/// `category` is `Some`, the file is routed to that `identity(category)` partition; when `None` the
/// table is unpartitioned and the file carries no partition.
async fn write_file(
    table: &Table,
    partition_key: Option<&PartitionKey>,
    category: Option<&str>,
    ids: Vec<i64>,
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema → arrow"));
    let row_count = ids.len();
    // Unpartitioned rows still need a non-null category value; "x" matches the Java oracle.
    let category_value = category.unwrap_or("x");
    let categories: Vec<&str> = std::iter::repeat_n(category_value, row_count).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(categories)) as ArrayRef,
    ])
    .expect("build the {id, category} data batch");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "vaodata".to_string(),
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
        .build(partition_key.cloned())
        .await
        .expect("build data file writer");
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

/// Build the case's table (S0 base live file) and run the (possibly append-only) replace; return
/// the observed outcome. Mirrors `ValidateAppendOnlyOracle.runReplaceAndDetect`.
async fn run_replace_and_detect(catalog: &impl Catalog, table: Table, case: &Case) -> Expected {
    let schema = table.metadata().current_schema().clone();
    let bound_spec = table.metadata().default_partition_spec().as_ref().clone();

    // S0: the base live file. For the partitioned cases it lands in cat=a (so a cat=a replace
    // removes it, a cat=b replace does not). For the unpartitioned case it is the single non-empty
    // file a full replace removes.
    let base = if case.partitioned {
        let pk_a = partition_key(schema.clone(), bound_spec.clone(), "a");
        write_file(&table, Some(&pk_a), Some("a"), vec![1]).await
    } else {
        write_file(&table, None, None, vec![1]).await
    };
    let table = append_files(catalog, &table, vec![base]).await;

    // The replace's added file. cat=a removes the base file; cat=b fills an empty partition; the
    // unpartitioned add is a full replace.
    let fresh = if case.partitioned {
        let category = case.add_category.expect("partitioned case has a category");
        let pk = partition_key(schema, bound_spec, category);
        write_file(&table, Some(&pk), Some(category), vec![999]).await
    } else {
        write_file(&table, None, None, vec![999]).await
    };

    let tx = Transaction::new(&table);
    let mut action = tx.replace_partitions().add_file(fresh);
    if case.validate_append_only {
        action = action.validate_append_only();
    }
    let tx = action.apply(tx).expect("apply replace_partitions");

    match tx.commit(catalog).await {
        Ok(_) => Expected::Commit,
        Err(e) => {
            assert_eq!(
                e.kind(),
                ErrorKind::DataInvalid,
                "an append-only rejection must be a non-retryable DataInvalid, got: {e}"
            );
            assert!(
                !e.retryable(),
                "the append-only assertion failure must be NON-retryable (Java ValidationException), got: {e}"
            );
            assert!(
                e.message().contains("validateAppendOnly") || e.message().contains("append-only"),
                "the error must name the append-only assertion, got: {}",
                e.message()
            );
            Expected::Throw
        }
    }
}

// ===========================================================================================
// D2 — the Rust mirror battery.
// ===========================================================================================

/// Run the four-case battery: for each case, build the equivalent table and exercise
/// `validate_append_only()`, asserting the observed THROW/COMMIT outcome equals the HAND-DECLARED
/// expectation. A clean offline no-op when `ICEBERG_INTEROP_VALIDATE_APPEND_ONLY_DIR` is unset.
#[tokio::test]
async fn test_validate_append_only_mirror() {
    let Some(dir) = run_dir() else {
        println!(
            "skipping interop_validate_append_only mirror — set \
             ICEBERG_INTEROP_VALIDATE_APPEND_ONLY_DIR \
             (run dev/java-interop/run-interop-validate-append-only.sh)"
        );
        return;
    };

    for case in cases() {
        let case_dir = dir.join(format!("rust_{}", case.name));
        let warehouse = case_dir.to_string_lossy().to_string();
        let table_location = format!("{warehouse}/rust_table");
        let catalog = build_catalog(
            &format!("interop_validate_append_only_{}", case.name),
            &warehouse,
        )
        .await;

        let table = create_table(&catalog, &table_location, case.partitioned).await;
        let outcome = run_replace_and_detect(&catalog, table, &case).await;

        assert_eq!(
            outcome, case.expected,
            "case {}: Rust's validate_append_only outcome must match the hand-declared expected",
            case.name
        );

        println!(
            "interop_validate_append_only mirror OK — case {} (partitioned={}, flag={}, add={:?}) \
             → {:?} (expected {:?})",
            case.name,
            case.partitioned,
            case.validate_append_only,
            case.add_category,
            outcome,
            case.expected
        );
    }
}
