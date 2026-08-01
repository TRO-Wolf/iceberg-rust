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

//! Tests for [`AuditPartitionKeys`] / [`RepairPartitionKeys`].
//!
//! The corrupted fixtures are built at the MANIFEST level (write a real parquet file, then commit a
//! manifest entry stamped with a deliberately wrong partition tuple through the public
//! [`DataFileBuilder`](crate::spec::DataFileBuilder)) — after Unit 1 the engine can no longer
//! produce this corruption, and the commit path validates a tuple's ARITY and TYPES but never its
//! VALUES, so a same-typed wrong value commits cleanly. That is the whole exposure.
//!
//! The two miskeyed fixtures are deliberately split by transform family — file A is wrong ONLY in
//! its `identity(dept)` component, file B ONLY in its `truncate[3](name)` component — so the
//! "read the file, not the manifest tuple" decision is falsifiable on its own: with the identity
//! constants left in place, A becomes invisible and B does not.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray};
use futures::TryStreamExt;
use tempfile::TempDir;

use super::*;
use crate::arrow::schema_to_arrow_schema;
use crate::expr::{Predicate, Reference};
use crate::io::LocalFsStorageFactory;
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataFileBuilder, Datum, FormatVersion, Literal, NestedField, PartitionSpec, PrimitiveType,
    Schema, Transform, Type,
};
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use crate::writer::file_writer::{FileWriter, FileWriterBuilder};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

// =================================================================================================
// Harness — a local-fs memory catalog (REAL parquet on disk)
// =================================================================================================

async fn local_fs_catalog() -> (impl Catalog, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let warehouse = temp_dir
        .path()
        .to_str()
        .expect("utf8 temp path")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([("warehouse".to_string(), warehouse)]),
        )
        .await
        .expect("load local-fs memory catalog");
    (catalog, temp_dir)
}

/// `id: long`, `dept: string`, `name: string` — the miskeying fixture schema.
fn id_dept_name_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "dept",
                Type::Primitive(PrimitiveType::String),
            )),
            Arc::new(NestedField::required(
                3,
                "name",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build schema")
}

async fn create_table(
    catalog: &impl Catalog,
    schema: Schema,
    spec: Option<PartitionSpec>,
) -> Table {
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let builder = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .format_version(FormatVersion::V2);
    let creation = match spec {
        Some(spec) => builder.partition_spec(spec).build(),
        None => builder.build(),
    };
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

/// The miskeying fixture table: `identity(dept)` + `truncate[3](name)`.
///
/// Two families, ONE of them identity — the split that makes the identity-constant decision
/// independently falsifiable.
async fn create_two_family_table(catalog: &impl Catalog) -> Table {
    let schema = id_dept_name_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("dept", "dept", Transform::Identity)
        .expect("add identity(dept)")
        .add_partition_field("name", "name_trunc", Transform::Truncate(3))
        .expect("add truncate[3](name)")
        .build()
        .expect("build spec");
    create_table(catalog, schema, Some(spec)).await
}

/// An `(id, dept, name)` batch in the fixture schema.
fn id_dept_name_batch(table: &Table, rows: &[(i64, &str, &str)]) -> RecordBatch {
    let arrow_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let ids: Vec<i64> = rows.iter().map(|(id, _, _)| *id).collect();
    let depts: Vec<&str> = rows.iter().map(|(_, dept, _)| *dept).collect();
    let names: Vec<&str> = rows.iter().map(|(_, _, name)| *name).collect();
    RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(depts)) as ArrayRef,
        Arc::new(StringArray::from(names)) as ArrayRef,
    ])
    .expect("build record batch")
}

/// Write ONE real parquet data file holding `batch`, and stamp its manifest entry with
/// `recorded_partition` — the manifest-level corruption injection. `recorded_partition` is written
/// verbatim; nothing recomputes it (only its arity/types are checked at commit).
async fn write_data_file_with_recorded_partition(
    table: &Table,
    file_name: &str,
    batch: &RecordBatch,
    recorded_partition: Struct,
) -> DataFile {
    let schema = table.metadata().current_schema();
    let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
    let output = table
        .file_io()
        .new_output(file_path)
        .expect("new parquet output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder
        .build(output)
        .await
        .expect("build parquet writer");
    writer.write(batch).await.expect("write parquet batch");
    let builders: Vec<DataFileBuilder> = writer.close().await.expect("close parquet writer");

    let mut builder = builders
        .into_iter()
        .next()
        .expect("the parquet writer produced a data-file builder");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(table.metadata().default_partition_spec_id())
        .partition(recorded_partition)
        .build()
        .expect("build data file")
}

/// Write `batch` through the SAME machinery the engine write path uses — compute the partition
/// values, split, and fan out — so the recorded tuples are genuinely computed and manifest
/// round-tripped.
async fn write_computed_files(table: &Table, batch: &RecordBatch) -> Vec<DataFile> {
    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().clone();
    let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(schema.clone(), spec)
        .expect("build splitter");
    let location_generator =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_generator = DefaultFileNameGenerator::new(
        "clean".to_string(),
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
        location_generator,
        file_name_generator,
    );
    let mut writer = FanoutWriter::new(DataFileWriterBuilder::new(rolling));
    for (partition_key, partition_batch) in splitter.split(batch).expect("split batch") {
        writer
            .write(partition_key, partition_batch)
            .await
            .expect("write partition batch");
    }
    writer.close().await.expect("close fanout writer")
}

async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    let tx = action.apply(tx).expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

/// Write a real parquet position-delete file removing `(path, pos)` pairs under `partition`.
async fn write_position_delete_file(
    table: &Table,
    partition: Struct,
    deletes: &[(String, i64)],
) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position delete config");
    let location_generator =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_generator = DefaultFileNameGenerator::new(
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
        location_generator,
        file_name_generator,
    );
    let partition_key = crate::spec::PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        partition,
    )
    .expect("PartitionKey::new: valid partition tuple");
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key))
        .await
        .expect("build position delete writer");
    let paths: Vec<&str> = deletes.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("position delete batch");
    writer.write(batch).await.expect("write position deletes");
    writer
        .close()
        .await
        .expect("close position delete writer")
        .into_iter()
        .next()
        .expect("one position delete file")
}

async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.row_delta().add_deletes(deletes);
    let tx = action.apply(tx).expect("apply row delta");
    tx.commit(catalog).await.expect("commit row delta")
}

/// Every `(id, dept, name)` row the scan hands back, sorted.
async fn scan_rows(table: &Table, filter: Option<Predicate>) -> Vec<(i64, String, String)> {
    let mut builder = table.scan().select(["id", "dept", "name"]);
    if let Some(filter) = filter {
        builder = builder.with_filter(filter);
    }
    let stream = builder
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect batches");

    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id is Int64")
            .clone();
        let depts = batch
            .column_by_name("dept")
            .expect("dept column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("dept is Utf8")
            .clone();
        let names = batch
            .column_by_name("name")
            .expect("name column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("name is Utf8")
            .clone();
        for row in 0..batch.num_rows() {
            rows.push((
                ids.value(row),
                depts.value(row).to_string(),
                names.value(row).to_string(),
            ));
        }
    }
    rows.sort();
    rows
}

/// `(id, dept, name)` triples, sorted — the expected-row literal helper.
fn rows_of(rows: &[(i64, &str, &str)]) -> Vec<(i64, String, String)> {
    let mut rows: Vec<(i64, String, String)> = rows
        .iter()
        .map(|(id, dept, name)| (*id, (*dept).to_string(), (*name).to_string()))
        .collect();
    rows.sort();
    rows
}

/// The recorded partition tuple of every live data file, keyed by path.
async fn recorded_partitions(table: &Table) -> HashMap<String, Struct> {
    live_data_files_by_path(table)
        .await
        .expect("live data files")
        .into_iter()
        .map(|(path, file)| (path, file.partition().clone()))
        .collect()
}

fn tuple(dept: &str, name_trunc: &str) -> Struct {
    Struct::from_iter([
        Some(Literal::string(dept)),
        Some(Literal::string(name_trunc)),
    ])
}

/// The A/B/C fixture: A miskeyed in its identity component only, B in its truncate component only,
/// C correct. Returns `(table, path_of_a, path_of_b, path_of_c)`.
async fn miskeyed_fixture(catalog: &impl Catalog) -> (Table, String, String, String) {
    let table = create_two_family_table(catalog).await;

    let batch_a = id_dept_name_batch(&table, &[(1, "eng", "alpha1"), (2, "eng", "alpha2")]);
    let file_a = write_data_file_with_recorded_partition(
        &table,
        "a.parquet",
        &batch_a,
        // TRUE tuple is ("eng", "alp") — only the identity component is wrong.
        tuple("sales", "alp"),
    )
    .await;

    let batch_b = id_dept_name_batch(&table, &[(3, "ops", "beta1"), (4, "ops", "beta2")]);
    let file_b = write_data_file_with_recorded_partition(
        &table,
        "b.parquet",
        &batch_b,
        // TRUE tuple is ("ops", "bet") — only the truncate component is wrong.
        tuple("ops", "zzz"),
    )
    .await;

    let batch_c = id_dept_name_batch(&table, &[(5, "fin", "sigma1")]);
    let file_c =
        write_data_file_with_recorded_partition(&table, "c.parquet", &batch_c, tuple("fin", "sig"))
            .await;

    let paths = (
        file_a.file_path().to_string(),
        file_b.file_path().to_string(),
        file_c.file_path().to_string(),
    );
    let table = append_files(catalog, &table, vec![file_a, file_b, file_c]).await;
    (table, paths.0, paths.1, paths.2)
}

fn finding_for<'a>(result: &'a AuditPartitionKeysResult, path: &str) -> &'a PartitionKeyFinding {
    result
        .findings
        .iter()
        .find(|finding| finding.data_file_path == path)
        .unwrap_or_else(|| {
            panic!(
                "expected a finding for {path}; got {:?}",
                result.corrupt_file_paths()
            )
        })
}

// =================================================================================================
// Detection
// =================================================================================================

/// A table written through the real compute-split-fanout path is CLEAN, across all four
/// recomputable transform families (identity / bucket / truncate / temporal). This also pins the
/// manifest ROUND TRIP: the recorded tuples are read back off disk and must equal what the
/// transforms recompute — a representation drift in any family would surface as a false finding.
#[tokio::test]
async fn test_clean_table_audits_clean_across_all_four_transform_families() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let schema = Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "dept",
                Type::Primitive(PrimitiveType::String),
            )),
            Arc::new(NestedField::required(
                3,
                "name",
                Type::Primitive(PrimitiveType::String),
            )),
            Arc::new(NestedField::required(
                4,
                "ts",
                Type::Primitive(PrimitiveType::Timestamp),
            )),
        ])
        .build()
        .expect("build schema");
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("dept", "dept", Transform::Identity)
        .expect("identity(dept)")
        .add_partition_field("id", "id_bucket", Transform::Bucket(4))
        .expect("bucket[4](id)")
        .add_partition_field("name", "name_trunc", Transform::Truncate(3))
        .expect("truncate[3](name)")
        .add_partition_field("ts", "ts_day", Transform::Day)
        .expect("day(ts)")
        .build()
        .expect("build spec");
    let table = create_table(&catalog, schema, Some(spec)).await;

    let arrow_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let base_micros: i64 = 1_700_000_000_000_000;
    let day_micros: i64 = 86_400_000_000;
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(vec![1_i64, 2, 3, 4, 5, 6])) as ArrayRef,
        Arc::new(StringArray::from(vec![
            "eng", "eng", "ops", "ops", "fin", "fin",
        ])) as ArrayRef,
        Arc::new(StringArray::from(vec![
            "alpha1", "alpha2", "beta1", "beta2", "sigma1", "sigma2",
        ])) as ArrayRef,
        Arc::new(TimestampMicrosecondArray::from(vec![
            base_micros,
            base_micros + day_micros,
            base_micros,
            base_micros + day_micros,
            base_micros,
            base_micros + day_micros,
        ])) as ArrayRef,
    ])
    .expect("build record batch");

    let files = write_computed_files(&table, &batch).await;
    assert!(
        files.len() > 1,
        "the fixture must span several partitions; wrote {} file(s)",
        files.len()
    );
    let file_count = files.len();
    let table = append_files(&catalog, &table, files).await;

    let result = AuditPartitionKeys::new(table)
        .execute()
        .await
        .expect("audit a clean table");

    assert!(
        result.is_clean(),
        "a table written through the compute path must audit clean; findings: {:?}",
        result.findings
    );
    assert_eq!(result.data_files_examined, file_count);
    assert_eq!(result.data_files_skipped, 0);
    assert_eq!(result.rows_examined, 6);
}

/// The audit flags EXACTLY the miskeyed files and names both tuples. File A is wrong only in its
/// `identity` component, file B only in its `truncate` component, file C is correct.
#[tokio::test]
async fn test_audit_flags_exactly_the_miskeyed_files_and_names_both_tuples() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let (table, path_a, path_b, path_c) = miskeyed_fixture(&catalog).await;

    let result = AuditPartitionKeys::new(table)
        .execute()
        .await
        .expect("audit the miskeyed fixture");

    assert_eq!(result.data_files_examined, 3);
    assert_eq!(result.data_files_skipped, 0);
    assert_eq!(result.rows_examined, 5);
    assert_eq!(
        result.findings.len(),
        2,
        "expected exactly the two miskeyed files; got {:?}",
        result.corrupt_file_paths()
    );
    assert!(
        !result.corrupt_file_paths().contains(&path_c.as_str()),
        "the correctly-keyed file must not be flagged"
    );

    let finding_a = finding_for(&result, &path_a);
    assert_eq!(finding_a.recorded_partition, tuple("sales", "alp"));
    assert_eq!(finding_a.computed_partitions, vec![tuple("eng", "alp")]);
    assert_eq!(finding_a.rows_examined, 2);
    assert_eq!(finding_a.mismatched_rows, 2);
    assert_eq!(finding_a.partition_spec_id, 0);

    let finding_b = finding_for(&result, &path_b);
    assert_eq!(finding_b.recorded_partition, tuple("ops", "zzz"));
    assert_eq!(finding_b.computed_partitions, vec![tuple("ops", "bet")]);
    assert_eq!(finding_b.mismatched_rows, 2);
}

/// The read the audit issues must NOT be told the manifest tuple: a planned task carries
/// `partition`/`partition_spec` (identity constants that OVERRIDE the file's own column), and the
/// recompute would then compare the recorded tuple against itself.
#[tokio::test]
async fn test_prepare_read_task_clears_the_partition_constants_and_the_residual() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let (table, path_a, _path_b, _path_c) = miskeyed_fixture(&catalog).await;

    let tasks = plan_whole_file_tasks(&table).await.expect("plan tasks");
    let planned = tasks
        .iter()
        .find(|task| task.data_file_path() == path_a)
        .expect("a planned task for file A");

    assert_eq!(
        planned.partition.as_ref(),
        Some(&tuple("sales", "alp")),
        "the planner attaches the RECORDED tuple — that is what makes stripping it load-bearing"
    );
    assert!(planned.partition_spec.is_some());

    let stripped = prepare_read_task(planned, false);
    assert!(stripped.partition.is_none());
    assert!(stripped.partition_spec.is_none());
    assert!(stripped.predicate.is_none());
    assert!(stripped.deletes.is_empty());
    assert_eq!(stripped.data_file_path(), path_a);

    let with_deletes = prepare_read_task(planned, true);
    assert_eq!(with_deletes.deletes.len(), planned.deletes.len());
    assert!(with_deletes.partition.is_none());
}

/// The harm the audit exists to find, asserted on the read path: a miskeyed file's rows vanish
/// from a partition-filtered query, and an unfiltered query hands back the RECORDED value for the
/// identity-partitioned column instead of the value stored in the file.
#[tokio::test]
async fn test_miskeyed_file_loses_rows_on_a_filtered_read_and_returns_the_recorded_value() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let (table, _path_a, _path_b, _path_c) = miskeyed_fixture(&catalog).await;

    let eng_rows = scan_rows(
        &table,
        Some(Reference::new("dept").equal_to(Datum::string("eng"))),
    )
    .await;
    assert!(
        eng_rows.is_empty(),
        "the two rows physically stored with dept='eng' are lost to a dept='eng' query: {eng_rows:?}"
    );

    assert_eq!(
        scan_rows(&table, None).await,
        rows_of(&[
            (1, "sales", "alpha1"),
            (2, "sales", "alpha2"),
            (3, "ops", "beta1"),
            (4, "ops", "beta2"),
            (5, "fin", "sigma1"),
        ]),
        "the identity-partitioned column comes back as the RECORDED value, not the stored one"
    );
}

/// A file under an UNPARTITIONED spec is skipped, never flagged.
#[tokio::test]
async fn test_unpartitioned_table_is_skipped_not_flagged() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog, id_dept_name_schema(), None).await;
    let batch = id_dept_name_batch(&table, &[(1, "eng", "alpha1")]);
    let file =
        write_data_file_with_recorded_partition(&table, "u.parquet", &batch, Struct::empty()).await;
    let table = append_files(&catalog, &table, vec![file]).await;

    let result = AuditPartitionKeys::new(table)
        .execute()
        .await
        .expect("audit an unpartitioned table");

    assert!(result.is_clean());
    assert_eq!(result.data_files_skipped, 1);
    assert_eq!(result.data_files_examined, 0);
    assert_eq!(result.rows_examined, 0);
}

/// A table with no snapshot audits clean without touching storage.
#[tokio::test]
async fn test_table_with_no_snapshot_audits_clean() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table = create_two_family_table(&catalog).await;

    let result = AuditPartitionKeys::new(table)
        .execute()
        .await
        .expect("audit an empty table");

    assert_eq!(result, AuditPartitionKeysResult::default());
    assert!(result.is_clean());
}

// =================================================================================================
// Repair
// =================================================================================================

/// The repair rewrites each miskeyed file's rows under their COMPUTED key: the audit goes clean,
/// the row-level data is intact, and the previously-lost partition-filtered read returns its rows.
#[tokio::test]
async fn test_repair_rewrites_under_the_computed_key_and_the_audit_goes_clean() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let (table, path_a, path_b, path_c) = miskeyed_fixture(&catalog).await;

    let result = RepairPartitionKeys::new(table.clone())
        .execute(&catalog)
        .await
        .expect("repair the miskeyed fixture");

    assert_eq!(result.repaired_data_files_count, 2);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.repaired_rows_count, 4);

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload the repaired table");

    let audit = AuditPartitionKeys::new(table.clone())
        .execute()
        .await
        .expect("re-audit after repair");
    assert!(
        audit.is_clean(),
        "the repaired table must audit clean; findings: {:?}",
        audit.findings
    );
    assert_eq!(audit.data_files_examined, 3);
    assert_eq!(audit.rows_examined, 5);

    // The miskeyed entries are gone and the new ones carry the TRUE tuples; the correctly-keyed
    // file was never touched.
    let recorded = recorded_partitions(&table).await;
    assert!(!recorded.contains_key(&path_a));
    assert!(!recorded.contains_key(&path_b));
    assert_eq!(recorded.get(&path_c), Some(&tuple("fin", "sig")));
    assert_eq!(
        recorded.values().cloned().collect::<HashSet<Struct>>(),
        HashSet::from([
            tuple("eng", "alp"),
            tuple("ops", "bet"),
            tuple("fin", "sig")
        ])
    );

    // Every row survives, with the values that were physically written.
    assert_eq!(
        scan_rows(&table, None).await,
        rows_of(&[
            (1, "eng", "alpha1"),
            (2, "eng", "alpha2"),
            (3, "ops", "beta1"),
            (4, "ops", "beta2"),
            (5, "fin", "sigma1"),
        ])
    );

    // The read that lost rows before the repair now returns them.
    assert_eq!(
        scan_rows(
            &table,
            Some(Reference::new("dept").equal_to(Datum::string("eng")))
        )
        .await,
        rows_of(&[(1, "eng", "alpha1"), (2, "eng", "alpha2")])
    );
}

/// A miskeyed file whose rows belong to SEVERAL true partitions (the reordered-source-column
/// symptom) is split into one correctly-keyed file per true partition.
#[tokio::test]
async fn test_repair_splits_a_file_whose_rows_span_two_true_partitions() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table = create_two_family_table(&catalog).await;

    let batch = id_dept_name_batch(&table, &[
        (1, "eng", "alpha1"),
        (2, "ops", "beta1"),
        (3, "eng", "alpha2"),
    ]);
    let file = write_data_file_with_recorded_partition(
        &table,
        "mixed.parquet",
        &batch,
        tuple("sales", "zzz"),
    )
    .await;
    let mixed_path = file.file_path().to_string();
    let table = append_files(&catalog, &table, vec![file]).await;

    let audit = AuditPartitionKeys::new(table.clone())
        .execute()
        .await
        .expect("audit the mixed file");
    let finding = finding_for(&audit, &mixed_path);
    assert_eq!(finding.mismatched_rows, 3);
    assert_eq!(
        finding
            .computed_partitions
            .iter()
            .cloned()
            .collect::<HashSet<Struct>>(),
        HashSet::from([tuple("eng", "alp"), tuple("ops", "bet")]),
        "both true partitions must be reported"
    );

    let result = RepairPartitionKeys::new(table.clone())
        .execute(&catalog)
        .await
        .expect("repair the mixed file");
    assert_eq!(result.repaired_data_files_count, 1);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.repaired_rows_count, 3);

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload the repaired table");
    let recorded = recorded_partitions(&table).await;
    assert!(!recorded.contains_key(&mixed_path));
    assert_eq!(
        recorded.values().cloned().collect::<HashSet<Struct>>(),
        HashSet::from([tuple("eng", "alp"), tuple("ops", "bet")])
    );
    assert!(
        AuditPartitionKeys::new(table.clone())
            .execute()
            .await
            .expect("re-audit")
            .is_clean()
    );
    assert_eq!(
        scan_rows(&table, None).await,
        rows_of(&[
            (1, "eng", "alpha1"),
            (2, "ops", "beta1"),
            (3, "eng", "alpha2")
        ])
    );
}

/// The repair reads LIVE rows: a row deleted by a position delete stays deleted, and is not
/// resurrected into the rewritten file.
#[tokio::test]
async fn test_repair_does_not_resurrect_a_position_deleted_row() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table = create_two_family_table(&catalog).await;

    let batch = id_dept_name_batch(&table, &[(1, "eng", "alpha1"), (2, "eng", "alpha2")]);
    let file =
        write_data_file_with_recorded_partition(&table, "d.parquet", &batch, tuple("sales", "alp"))
            .await;
    let data_path = file.file_path().to_string();
    let table = append_files(&catalog, &table, vec![file]).await;

    // Delete row 0 (id = 1). The delete file carries the SAME (wrong) tuple the data file does,
    // which is what an engine writing against the corrupted table would have produced.
    let delete_file =
        write_position_delete_file(&table, tuple("sales", "alp"), &[(data_path.clone(), 0)]).await;
    let table = add_deletes(&catalog, &table, vec![delete_file]).await;

    // Detection sees BOTH physical rows (delete files are stripped for detection).
    let audit = AuditPartitionKeys::new(table.clone())
        .execute()
        .await
        .expect("audit");
    let finding = finding_for(&audit, &data_path);
    assert_eq!(finding.rows_examined, 2);
    assert_eq!(finding.mismatched_rows, 2);

    let result = RepairPartitionKeys::new(table.clone())
        .execute(&catalog)
        .await
        .expect("repair");
    assert_eq!(result.repaired_data_files_count, 1);
    assert_eq!(
        result.repaired_rows_count, 1,
        "only the LIVE row may be rewritten"
    );

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    assert_eq!(
        scan_rows(&table, None).await,
        rows_of(&[(2, "eng", "alpha2")])
    );
    assert!(
        AuditPartitionKeys::new(table)
            .execute()
            .await
            .expect("re-audit")
            .is_clean()
    );
}

/// A clean table is left alone: no commit, no new snapshot.
#[tokio::test]
async fn test_repair_of_a_clean_table_commits_nothing() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table = create_two_family_table(&catalog).await;
    let batch = id_dept_name_batch(&table, &[(1, "eng", "alpha1"), (2, "ops", "beta1")]);
    let files = write_computed_files(&table, &batch).await;
    let table = append_files(&catalog, &table, files).await;
    let snapshot_before = table
        .metadata()
        .current_snapshot()
        .expect("a snapshot")
        .snapshot_id();

    let result = RepairPartitionKeys::new(table.clone())
        .execute(&catalog)
        .await
        .expect("repair a clean table");

    assert_eq!(result, RepairPartitionKeysResult::default());
    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    assert_eq!(
        reloaded
            .metadata()
            .current_snapshot()
            .expect("a snapshot")
            .snapshot_id(),
        snapshot_before,
        "a clean table must not be committed to"
    );
}
