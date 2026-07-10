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

//! ENGINE_CONTRACT §5 isolation-recipe pins (G4, 2026-07-09) — the SERIALIZABLE-vs-SNAPSHOT
//! distinction proven BEHAVIORALLY for both write modes, exactly as the §5 table prescribes and as
//! decoded from Java 1.10.0 (source @ the `apache-iceberg-1.10.0` tag, commit 2114bf6):
//!
//! - **COW (`OverwriteFiles`)** — `SparkWrite.CopyOnWriteOperation`: snapshot isolation enables
//!   `validateNoConflictingDeletes()` only (`commitWithSnapshotIsolation`, SparkWrite.java L490-509);
//!   serializable ADDS `validateNoConflictingData()` (`commitWithSerializableIsolation`, L467-488).
//!   So a CONCURRENT INSERT matching the command's conflict filter commits fine under snapshot
//!   isolation but is a serializability violation (lost update / write skew) under serializable.
//! - **MoR (`RowDelta`)** — `SparkPositionDeltaWrite.PositionDeltaBatchWrite.commit`: the base
//!   (all commands) is `conflictDetectionFilter` + `validateDataFilesExist` + `validateFromSnapshot`
//!   (L240-249); serializable ADDS `validateNoConflictingDataFiles()` (L256-258). NOTE the DELETE
//!   command does NOT enable `validateDeletedFiles()`/`validateNoConflictingDeleteFiles()` — those
//!   are UPDATE/MERGE-only (L251-254) — so the snapshot-isolation DELETE leg here deliberately runs
//!   the CORRECTED (validate-deleted-files-free) §5 DELETE recipe.
//!
//! Each test runs the SAME scenario twice (two independent tables with identical histories): the
//! snapshot-isolation recipe must COMMIT (with the post-commit live set asserted — the observable),
//! and the serializable recipe must FAIL with the exact validation the cell prescribes (kind
//! `DataInvalid`, non-retryable, message naming the conflicting-files validation — not just "some
//! error"). Mutation pins (each proven RED during G4): drop the serializable leg's
//! `validate_no_conflicting_data[_files]()` ⇒ the reject assert fails; give the snapshot leg the
//! serializable validation (the swapped-mapping / over-broadened guard) ⇒ the accept assert fails.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
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
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};

/// The fixture schema `{1 id long required, 2 y long required}` (the C1 conflict-interop shape).
fn recipe_schema() -> Schema {
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

/// Create an UNPARTITIONED V2 `{id, y}` table named `table_name` at `table_location`.
async fn create_recipe_table(
    catalog: &impl Catalog,
    table_name: &str,
    table_location: &str,
) -> Table {
    let namespace = NamespaceIdent::new("recipes".to_string());
    if !catalog
        .namespace_exists(&namespace)
        .await
        .expect("check namespace")
    {
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace");
    }
    let creation = TableCreation::builder()
        .name(table_name.to_string())
        .location(table_location.to_string())
        .schema(recipe_schema())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create recipe table")
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
        "rdata".to_string(),
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

/// Write a REAL parquet UNPARTITIONED position-delete file deleting `position` of `data_file_path`.
async fn write_position_delete_file(
    table: &Table,
    data_file_path: &str,
    position: i64,
) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");

    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
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
        .build(None)
        .await
        .expect("build unpartitioned position-delete writer");

    let paths = StringArray::from(vec![data_file_path]);
    let positions = Int64Array::from(vec![position]);
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

/// The §5 scenario history, built fresh per leg: S0 = `fast_append` base file (rows `(1,0)`,
/// `(2,5)` ⇒ y bounds `[0,5]`); S1 = a CONCURRENT INSERT (`fast_append`, rows `(100,60)`,
/// `(101,70)` ⇒ y bounds `[60,70]`, OVERLAPPING the command's `y >= 50` conflict filter).
/// Returns `(table_at_s1, s0_snapshot_id, base_file_path)`.
async fn build_recipe_history(
    catalog: &impl Catalog,
    table_name: &str,
    table_location: &str,
) -> (Table, i64, String) {
    let table = create_recipe_table(catalog, table_name, table_location).await;

    // S0: the base file the row-level command read (its scan snapshot).
    let base = write_yid_file(&table, vec![1, 2], vec![0, 5]).await;
    let base_path = base.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![base])
        .apply(tx)
        .expect("apply fast_append S0");
    let table = tx.commit(catalog).await.expect("commit S0");
    let s0 = table
        .metadata()
        .current_snapshot()
        .expect("S0 snapshot")
        .snapshot_id();

    // S1: the CONCURRENT INSERT — matches the conflict filter (y bounds [60,70] vs `y >= 50`).
    let concurrent = write_yid_file(&table, vec![100, 101], vec![60, 70]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![concurrent])
        .apply(tx)
        .expect("apply fast_append S1 (concurrent insert)");
    let table = tx.commit(catalog).await.expect("commit S1");

    (table, s0, base_path)
}

/// Scan the table and return the live `(id, y)` rows sorted by id — the post-commit observable.
async fn scan_live_rows(table: &Table) -> Vec<(i64, i64)> {
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to_arrow")
        .try_collect()
        .await
        .expect("collect scan batches");
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = batch.column(0).as_primitive::<Int64Type>();
        let ys = batch.column(1).as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), ys.value(i)));
        }
    }
    rows.sort_unstable();
    rows
}

/// Assert `err` is the §5-prescribed NON-RETRYABLE conflicting-DATA validation failure — the error
/// must NAME the added-data-files validation (Java `ValidationException` "Found conflicting files
/// that can contain records matching ..."), so a generic commit failure cannot green this pin.
fn assert_conflicting_data_rejection(err: &iceberg::Error) {
    assert_eq!(
        err.kind(),
        ErrorKind::DataInvalid,
        "the conflict must surface as non-retryable DataInvalid (Java ValidationException), got: {err}"
    );
    assert!(
        !err.retryable(),
        "the §5 validation failure must be NON-retryable (the engine surfaces it, never loops), got: {err}"
    );
    assert!(
        err.message()
            .contains("Found conflicting files that can contain records matching"),
        "the failure must name the conflicting-DATA validation, got: {err}"
    );
}

/// §5 COW DELETE/UPDATE/MERGE — the serializable-vs-snapshot distinction, behaviorally.
///
/// Same scenario twice (S0 base read by the command; S1 = concurrent INSERT overlapping the
/// command's `y >= 50` condition):
/// - **snapshot-isolation recipe** (`validate_from_snapshot(S0)` + `conflict_detection_filter` +
///   `validate_no_conflicting_deletes()` — SparkWrite.java `commitWithSnapshotIsolation` L490-509)
///   ⇒ the COW rewrite COMMITS; the post-commit scan shows the rewrite AND the concurrent insert.
/// - **serializable recipe** (above + `validate_no_conflicting_data()` —
///   `commitWithSerializableIsolation` L467-488) ⇒ REJECTED with the conflicting-data validation.
///
/// Risk pinned: an engine that omits `validate_no_conflicting_data()` under serializable silently
/// accepts a lost-update/write-skew history; an engine that ADDS it under snapshot isolation
/// spuriously aborts legal MVCC histories.
#[tokio::test]
async fn test_s5_cow_serializable_rejects_concurrent_insert_snapshot_isolation_commits() {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = build_catalog("s5_cow", &warehouse).await;

    // --- Leg 1: SNAPSHOT isolation ⇒ the COW commit MUST succeed. ---
    {
        let (table, s0, base_path) = build_recipe_history(
            &catalog,
            "cow_snapshot",
            &format!("{warehouse}/cow_snapshot"),
        )
        .await;
        // The COW rewrite of the base file: survivors (id=2 kept, id=1 deleted by the command).
        let rewritten = write_yid_file(&table, vec![2], vec![5]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file(&base_path)
            .add_file(rewritten)
            .validate_from_snapshot(s0)
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_no_conflicting_deletes();
        let tx = action
            .apply(tx)
            .expect("apply snapshot-isolation overwrite");
        let table = tx
            .commit(&catalog)
            .await
            .expect("§5 COW snapshot-isolation recipe MUST commit over a concurrent insert");

        // The post-commit observable: the rewrite landed AND the concurrent insert survives.
        assert_eq!(
            scan_live_rows(&table).await,
            vec![(2, 5), (100, 60), (101, 70)],
            "post-commit live set = COW survivors + the concurrent insert"
        );
    }

    // --- Leg 2: SERIALIZABLE ⇒ the SAME commit MUST be rejected by validate_no_conflicting_data. ---
    {
        let (table, s0, base_path) = build_recipe_history(
            &catalog,
            "cow_serializable",
            &format!("{warehouse}/cow_serializable"),
        )
        .await;
        let rewritten = write_yid_file(&table, vec![2], vec![5]).await;

        let tx = Transaction::new(&table);
        let action = tx
            .overwrite_files()
            .delete_file(&base_path)
            .add_file(rewritten)
            .validate_from_snapshot(s0)
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_no_conflicting_deletes()
            .validate_no_conflicting_data();
        let tx = action.apply(tx).expect("apply serializable overwrite");
        let err = tx
            .commit(&catalog)
            .await
            .expect_err("§5 COW serializable recipe MUST reject the concurrent matching insert");
        assert_conflicting_data_rejection(&err);
    }
}

/// §5 MoR DELETE — the serializable-vs-snapshot distinction, behaviorally, on the CORRECTED
/// DELETE cell (NO `validate_deleted_files()` — Java enables it for UPDATE/MERGE only,
/// SparkPositionDeltaWrite.java L251-254).
///
/// Same scenario twice (S0 base read by the DELETE; S1 = concurrent INSERT overlapping `y >= 50`):
/// - **snapshot-isolation recipe** (base only: `validate_from_snapshot(S0)` +
///   `conflict_detection_filter` + `validate_data_files_exist([base])` —
///   SparkPositionDeltaWrite.java L240-249) ⇒ the `RowDelta` COMMITS; the post-commit scan omits
///   exactly the position-deleted row and keeps the concurrent insert.
/// - **serializable recipe** (above + `validate_no_conflicting_data_files()` — L256-258) ⇒
///   REJECTED with the conflicting-data validation.
///
/// Risk pinned: the same lost-update/write-skew asymmetry as the COW test, on the `RowDelta`
/// surface — plus the corrected-cell guarantee that a MoR DELETE needs no delete-file validations
/// to commit under snapshot isolation.
#[tokio::test]
async fn test_s5_merge_on_read_delete_serializable_rejects_concurrent_insert_snapshot_isolation_commits()
 {
    use tempfile::TempDir;

    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let catalog = build_catalog("s5_merge_on_read", &warehouse).await;

    // --- Leg 1: SNAPSHOT isolation ⇒ the RowDelta MUST commit. ---
    {
        let (table, s0, base_path) = build_recipe_history(
            &catalog,
            "merge_on_read_snapshot",
            &format!("{warehouse}/merge_on_read_snapshot"),
        )
        .await;
        // The MoR DELETE: position-delete row 0 (id=1) of the base file.
        let pos_delete = write_position_delete_file(&table, &base_path, 0).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![pos_delete])
            .validate_from_snapshot(s0)
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_data_files_exist(vec![base_path.clone()]);
        let tx = action
            .apply(tx)
            .expect("apply snapshot-isolation row delta");
        let table = tx
            .commit(&catalog)
            .await
            .expect("§5 MoR DELETE snapshot-isolation recipe MUST commit over a concurrent insert");

        // The post-commit observable: exactly the deleted row is gone; the concurrent insert lives.
        assert_eq!(
            scan_live_rows(&table).await,
            vec![(2, 5), (100, 60), (101, 70)],
            "post-commit live set omits exactly the position-deleted row (1,0)"
        );
    }

    // --- Leg 2: SERIALIZABLE ⇒ rejected by validate_no_conflicting_data_files. ---
    {
        let (table, s0, base_path) = build_recipe_history(
            &catalog,
            "merge_on_read_serializable",
            &format!("{warehouse}/merge_on_read_serializable"),
        )
        .await;
        let pos_delete = write_position_delete_file(&table, &base_path, 0).await;

        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .add_deletes(vec![pos_delete])
            .validate_from_snapshot(s0)
            .conflict_detection_filter(
                Reference::new("y").greater_than_or_equal_to(Datum::long(50)),
            )
            .validate_data_files_exist(vec![base_path.clone()])
            .validate_no_conflicting_data_files();
        let tx = action.apply(tx).expect("apply serializable row delta");
        let err = tx.commit(&catalog).await.expect_err(
            "§5 MoR DELETE serializable recipe MUST reject the concurrent matching insert",
        );
        assert_conflicting_data_rejection(&err);
    }
}
