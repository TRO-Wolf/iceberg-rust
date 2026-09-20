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

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use datafusion::arrow::array::Int64Array;
use datafusion::common::stats::Precision;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{Expr, SessionConfig, SessionContext, col, lit};
use datafusion::scalar::ScalarValue;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::expr::BoundPredicate;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::scan::{
    CombinedScanTask, FileScanTask, PartitionWork, ScanFilterMode, assign_partition_work,
};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, NestedField,
    PrimitiveLiteral, PrimitiveType, Schema as IcebergSchema, Struct, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::transform::create_transform_function;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

use crate::IcebergCatalogProvider;
use crate::physical_plan::scan::{IcebergTableScan, ScanKnobs};
use crate::physical_plan::scan_helpers::exact_table_row_count;
use crate::physical_plan::scan_knobs::ensure_iceberg_scan_options;

const A_ROWS: u64 = 100;
const B_ROWS: u64 = 40;

struct Fixture {
    catalog: Arc<MemoryCatalog>,
    _warehouse: TempDir,
}

fn table_schema() -> IcebergSchema {
    IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "part", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema")
}

fn ts_schema() -> IcebergSchema {
    IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "ts", Type::Primitive(PrimitiveType::Timestamp)).into(),
        ])
        .build()
        .expect("schema")
}

async fn fixture(partitioned: bool, properties: HashMap<String, String>) -> Fixture {
    let partition_spec = partitioned.then(|| {
        UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "part", Transform::Identity)
            .expect("partition field")
            .build()
    });
    fixture_with(table_schema(), partition_spec, properties).await
}

async fn fixture_with(
    schema: IcebergSchema,
    partition_spec: Option<UnboundPartitionSpec>,
    properties: HashMap<String, String>,
) -> Fixture {
    let warehouse = TempDir::new().expect("warehouse");
    let catalog = Arc::new(
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse.path().to_str().expect("utf8").to_string(),
                )]),
            )
            .await
            .expect("catalog"),
    );
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(format!("{}/t", warehouse.path().to_str().expect("utf8")))
        .schema(schema)
        .properties(properties)
        .partition_spec_opt(partition_spec)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("table");
    Fixture {
        catalog,
        _warehouse: warehouse,
    }
}

fn part(value: &str) -> Struct {
    Struct::from_iter([Some(Literal::string(value.to_string()))])
}

fn day_part(date: &str) -> Struct {
    let days = match Datum::date_from_str(date).expect("valid date").literal() {
        PrimitiveLiteral::Int(value) => *value,
        other => panic!("expected an Int date literal, got {other:?}"),
    };
    Struct::from_iter([Some(Literal::date(days))])
}

fn ts_micros(datetime: &str) -> i64 {
    match Datum::timestamp_from_str(datetime)
        .expect("valid timestamp")
        .literal()
    {
        PrimitiveLiteral::Long(value) => *value,
        other => panic!("expected a Long timestamp literal, got {other:?}"),
    }
}

fn bucket_of(value: i32, num_buckets: u32) -> i32 {
    let transform =
        create_transform_function(&Transform::Bucket(num_buckets)).expect("bucket transform");
    match transform
        .transform_literal(&Datum::int(value))
        .expect("bucket literal")
        .expect("bucket value")
        .literal()
    {
        PrimitiveLiteral::Int(bucket) => *bucket,
        other => panic!("expected an Int bucket literal, got {other:?}"),
    }
}

fn data_file(path: &str, records: u64, size: u64, partition: Struct) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(size)
        .record_count(records)
        .partition_spec_id(0)
        .partition(partition)
        .build()
        .expect("data file")
}

fn position_delete_file(path: &str, partition: Struct) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(partition)
        .build()
        .expect("position delete file")
}

fn equality_delete_file(path: &str, partition: Struct) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::EqualityDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(partition)
        .equality_ids(Some(vec![1]))
        .build()
        .expect("equality delete file")
}

fn task(path: &str, file_records: Option<u64>, predicate: Option<BoundPredicate>) -> FileScanTask {
    FileScanTask {
        file_size_in_bytes: 4096,
        start: 0,
        length: 4096,
        record_count: file_records,
        file_record_count: file_records,
        data_file_path: Arc::from(path),
        data_file_format: DataFileFormat::Parquet,
        schema: Arc::new(table_schema()),
        project_field_ids: Arc::from(vec![1]),
        predicate: predicate.map(Arc::new),
        deletes: Arc::from(vec![]),
        partition: None,
        partition_spec: None,
        name_mapping: None,
        case_sensitive: true,
        split_offsets: None,
        first_row_id: None,
        file_sequence_number: None,
    }
}

fn work(tasks: Vec<FileScanTask>) -> Vec<PartitionWork> {
    assign_partition_work(
        0,
        ScanFilterMode::Residual,
        vec![CombinedScanTask::new(tasks)],
        1,
    )
}

async fn append(catalog: &MemoryCatalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply");
    tx.commit(catalog).await.expect("commit")
}

async fn add_deletes(catalog: &MemoryCatalog, table: &Table, deletes: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(deletes)
        .apply(tx)
        .expect("apply");
    tx.commit(catalog).await.expect("commit")
}

async fn load(fixture: &Fixture) -> Table {
    fixture
        .catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new("ns".to_string()),
            "t".to_string(),
        ))
        .await
        .expect("load table")
}

async fn doctor_total_records(table: &Table, value: Option<&str>) {
    let location = table
        .metadata_location()
        .expect("metadata location")
        .to_string();
    let file_io = table.file_io();
    let raw = file_io
        .new_input(&location)
        .expect("input")
        .read()
        .await
        .expect("read metadata");
    let text = String::from_utf8(raw.to_vec()).expect("utf8 metadata");
    let current = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .summary()
        .additional_properties
        .get("total-records")
        .cloned()
        .expect("summary total-records");
    let needle = format!("\"total-records\":\"{current}\"");
    assert!(text.contains(&needle), "metadata must carry {needle}");
    let edited = match value {
        Some(v) => text.replace(&needle, &format!("\"total-records\":\"{v}\"")),
        None => {
            let removed = text
                .replace(&format!("{needle},"), "")
                .replace(&format!(",{needle}"), "");
            assert!(
                !removed.contains("\"total-records\""),
                "the key must be gone after removal"
            );
            removed
        }
    };
    file_io
        .new_output(&location)
        .expect("output")
        .write(Bytes::from(edited))
        .await
        .expect("write metadata");
}

async fn plan(table: &Table, filters: &[Expr], knobs: ScanKnobs) -> IcebergTableScan {
    plan_snapshot(table, None, filters, knobs).await
}

async fn plan_snapshot(
    table: &Table,
    snapshot_id: Option<i64>,
    filters: &[Expr],
    knobs: ScanKnobs,
) -> IcebergTableScan {
    IcebergTableScan::plan(
        table.clone(),
        snapshot_id,
        false,
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema")),
        None,
        filters,
        None,
        knobs,
    )
    .await
    .expect("plan")
}

fn num_rows(scan: &IcebergTableScan) -> Precision<usize> {
    scan.partition_statistics(None)
        .expect("partition statistics")
        .num_rows
}

fn planned_paths(scan: &IcebergTableScan) -> Vec<String> {
    scan.partition_work()
        .iter()
        .flat_map(|work| work.tasks())
        .map(|task| task.data_file_path().to_string())
        .collect()
}

fn find_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<&IcebergTableScan> {
    if let Some(scan) = plan.downcast_ref::<IcebergTableScan>() {
        return Some(scan);
    }
    plan.children().into_iter().find_map(find_scan)
}

#[tokio::test]
async fn doctored_total_records_reports_planned_file_count() {
    let fixture = fixture(false, HashMap::new()).await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![
        data_file("data/a.parquet", A_ROWS, 4096, Struct::empty()),
        data_file("data/b.parquet", B_ROWS, 4096, Struct::empty()),
    ])
    .await;
    doctor_total_records(&table, Some("0")).await;
    let table = load(&fixture).await;

    let scan = plan(&table, &[], ScanKnobs::default()).await;
    assert_eq!(
        num_rows(&scan),
        Precision::Exact(usize::try_from(A_ROWS + B_ROWS).expect("count fits usize")),
        "the exact count must come from the planned files, not the doctored summary"
    );
}

#[tokio::test]
async fn missing_total_records_reports_planned_file_count() {
    let fixture = fixture(false, HashMap::new()).await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![
        data_file("data/a.parquet", A_ROWS, 4096, Struct::empty()),
        data_file("data/b.parquet", B_ROWS, 4096, Struct::empty()),
    ])
    .await;
    doctor_total_records(&table, None).await;
    let table = load(&fixture).await;

    let scan = plan(&table, &[], ScanKnobs::default()).await;
    assert_eq!(
        num_rows(&scan),
        Precision::Exact(usize::try_from(A_ROWS + B_ROWS).expect("count fits usize")),
        "the exact count must come from the planned files when the summary has no total"
    );
}

#[tokio::test]
async fn partition_pruned_scan_sums_planned_files_only() {
    let fixture = fixture(true, HashMap::new()).await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![
        data_file("data/a.parquet", A_ROWS, 4096, part("a")),
        data_file("data/b.parquet", B_ROWS, 4096, part("b")),
    ])
    .await;

    let filters = [col("part").eq(lit("a"))];
    let scan = plan(&table, &filters, ScanKnobs::default()).await;
    assert_eq!(
        planned_paths(&scan),
        vec!["data/a.parquet".to_string()],
        "partition pruning must plan only the 'a' file"
    );
    assert_eq!(
        num_rows(&scan),
        Precision::Exact(usize::try_from(A_ROWS).expect("count fits usize")),
        "a partition-satisfied filter still yields an exact count over the surviving files"
    );
}

#[tokio::test]
async fn day_interior_range_reports_exact() {
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "ts_day", Transform::Day)
        .expect("partition field")
        .build();
    let fixture = fixture_with(ts_schema(), Some(spec), HashMap::new()).await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![data_file(
        "data/a.parquet",
        A_ROWS,
        4096,
        day_part("2021-01-15"),
    )])
    .await;

    let filters = [
        col("ts").gt_eq(lit(ScalarValue::TimestampMicrosecond(
            Some(ts_micros("2021-01-01T00:00:00")),
            None,
        ))),
        col("ts").lt_eq(lit(ScalarValue::TimestampMicrosecond(
            Some(ts_micros("2021-01-31T00:00:00")),
            None,
        ))),
    ];
    let scan = plan(&table, &filters, ScanKnobs::default()).await;
    assert_eq!(
        planned_paths(&scan),
        vec!["data/a.parquet".to_string()],
        "the interior-day file must survive pruning"
    );
    assert!(
        scan.partition_work()
            .iter()
            .flat_map(|work| work.tasks())
            .all(|task| matches!(task.predicate(), Some(BoundPredicate::AlwaysTrue))),
        "a day-interior range must leave an AlwaysTrue residual on the surviving task"
    );
    assert_eq!(
        num_rows(&scan),
        Precision::Exact(usize::try_from(A_ROWS).expect("count fits usize")),
        "an AlwaysTrue residual removes zero rows, so the count stays exact"
    );
}

#[tokio::test]
async fn bucket_equality_keeps_residual_and_refuses_exact() {
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(1, "id_bucket", Transform::Bucket(16))
        .expect("partition field")
        .build();
    let fixture = fixture_with(table_schema(), Some(spec), HashMap::new()).await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![data_file(
        "data/a.parquet",
        A_ROWS,
        4096,
        Struct::from_iter([Some(Literal::int(bucket_of(5, 16)))]),
    )])
    .await;

    let filters = [col("id").eq(lit(5i32))];
    let scan = plan(&table, &filters, ScanKnobs::default()).await;
    assert_eq!(
        planned_paths(&scan),
        vec!["data/a.parquet".to_string()],
        "the file holding id=5's bucket must survive pruning"
    );
    assert!(
        scan.partition_work()
            .iter()
            .flat_map(|work| work.tasks())
            .any(|task| !matches!(task.predicate(), None | Some(BoundPredicate::AlwaysTrue))),
        "bucket equality must leave a real residual on the surviving task"
    );
    assert!(
        matches!(num_rows(&scan), Precision::Absent),
        "a residual-bearing task must report nothing exact"
    );
}

#[tokio::test]
async fn row_filtered_scan_reports_nothing_exact() {
    let fixture = fixture(false, HashMap::new()).await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![data_file(
        "data/a.parquet",
        A_ROWS,
        4096,
        Struct::empty(),
    )])
    .await;

    let filters = [col("id").gt_eq(lit(5i32))];
    let scan = plan(&table, &filters, ScanKnobs::default()).await;
    assert!(
        scan.partition_work()
            .iter()
            .flat_map(|work| work.tasks())
            .any(|task| task.predicate().is_some()),
        "the fixture must produce a real residual on the surviving task"
    );
    assert!(
        matches!(num_rows(&scan), Precision::Absent),
        "a scan whose predicate is not fully partition-satisfied must report nothing exact"
    );
}

#[tokio::test]
async fn position_deletes_report_nothing_exact() {
    let fixture = fixture(false, HashMap::new()).await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![data_file(
        "data/a.parquet",
        A_ROWS,
        4096,
        Struct::empty(),
    )])
    .await;
    let table = add_deletes(&fixture.catalog, &table, vec![position_delete_file(
        "data/a-pos-del.parquet",
        Struct::empty(),
    )])
    .await;

    let scan = plan(&table, &[], ScanKnobs::default()).await;
    assert!(
        scan.partition_work()
            .iter()
            .flat_map(|work| work.tasks())
            .any(|task| !task.deletes.is_empty()),
        "the fixture must attach position deletes to the data task"
    );
    assert!(
        matches!(num_rows(&scan), Precision::Absent),
        "a file with deletes must report nothing exact"
    );
}

#[tokio::test]
async fn equality_deletes_report_nothing_exact() {
    let fixture = fixture(false, HashMap::new()).await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![data_file(
        "data/a.parquet",
        A_ROWS,
        4096,
        Struct::empty(),
    )])
    .await;
    let table = add_deletes(&fixture.catalog, &table, vec![equality_delete_file(
        "data/a-eq-del.parquet",
        Struct::empty(),
    )])
    .await;

    let scan = plan(&table, &[], ScanKnobs::default()).await;
    assert!(
        scan.partition_work()
            .iter()
            .flat_map(|work| work.tasks())
            .any(|task| !task.deletes.is_empty()),
        "the fixture must attach equality deletes to the data task"
    );
    assert!(
        matches!(num_rows(&scan), Precision::Absent),
        "a file with equality deletes must report nothing exact"
    );
}

#[tokio::test]
async fn time_travel_scan_sums_the_scanned_snapshots_files() {
    let fixture = fixture(false, HashMap::new()).await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![data_file(
        "data/a.parquet",
        A_ROWS,
        4096,
        Struct::empty(),
    )])
    .await;
    let first_snapshot = table
        .metadata()
        .current_snapshot()
        .expect("first snapshot")
        .snapshot_id();
    let table = append(&fixture.catalog, &table, vec![data_file(
        "data/b.parquet",
        B_ROWS,
        4096,
        Struct::empty(),
    )])
    .await;
    assert_ne!(
        table
            .metadata()
            .current_snapshot()
            .expect("current snapshot")
            .snapshot_id(),
        first_snapshot,
        "the second append must create a new current snapshot"
    );

    let scan = plan_snapshot(&table, Some(first_snapshot), &[], ScanKnobs::default()).await;
    assert_eq!(
        planned_paths(&scan),
        vec!["data/a.parquet".to_string()],
        "the older snapshot must plan only its own files"
    );
    assert_eq!(
        num_rows(&scan),
        Precision::Exact(usize::try_from(A_ROWS).expect("count fits usize")),
        "time travel must sum the scanned snapshot's planned files"
    );
}

#[tokio::test]
async fn split_file_counts_once() {
    let fixture = fixture(
        false,
        HashMap::from([("read.split.target-size".to_string(), "64".to_string())]),
    )
    .await;
    let table = load(&fixture).await;
    let table = append(&fixture.catalog, &table, vec![data_file(
        "data/a.parquet",
        A_ROWS,
        4096,
        Struct::empty(),
    )])
    .await;

    let scan = plan(&table, &[], ScanKnobs::default()).await;
    let tasks: Vec<String> = planned_paths(&scan);
    assert!(
        tasks.len() > 1,
        "the small split target must fan the file out into ranged tasks, got {tasks:?}"
    );
    assert!(
        tasks.iter().all(|path| path == "data/a.parquet"),
        "every ranged task must carry the same data-file path, got {tasks:?}"
    );
    assert_eq!(
        num_rows(&scan),
        Precision::Exact(usize::try_from(A_ROWS).expect("count fits usize")),
        "one file split into {} tasks must count once, not {} times",
        tasks.len(),
        tasks.len()
    );
}

#[tokio::test]
async fn count_star_sql_returns_planned_rows_on_doctored_summary() {
    let fixture = fixture(false, HashMap::new()).await;
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(fixture.catalog.clone())
            .await
            .expect("catalog provider"),
    );
    let mut config = SessionConfig::new();
    ensure_iceberg_scan_options(&mut config);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_catalog("catalog", provider);
    ctx.sql("INSERT INTO catalog.ns.t VALUES (1, 'a', 'x'), (2, 'b', 'y'), (3, 'c', 'z')")
        .await
        .expect("insert plan")
        .collect()
        .await
        .expect("insert");

    let table = load(&fixture).await;
    doctor_total_records(&table, Some("0")).await;

    let select_plan = ctx
        .sql("SELECT * FROM catalog.ns.t")
        .await
        .expect("select plan")
        .create_physical_plan()
        .await
        .expect("select physical plan");
    let iceberg_scan = find_scan(&select_plan).expect("the select plan must contain the scan");
    assert_eq!(
        iceberg_scan
            .partition_statistics(None)
            .expect("partition statistics")
            .num_rows,
        Precision::Exact(3),
        "the scan must report Exact so the folded count(*) is the planned-file sum, \
         not a row-count fallback"
    );

    let batches = ctx
        .sql("SELECT count(*) FROM catalog.ns.t")
        .await
        .expect("count plan")
        .collect()
        .await
        .expect("count");
    let total: i64 = batches
        .iter()
        .map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("count int64")
                .value(0)
        })
        .sum();
    assert_eq!(
        total, 3,
        "count(*) must come from the planned files, not the doctored total-records"
    );
}

#[test]
fn empty_plan_reports_zero() {
    assert_eq!(exact_table_row_count(&[]), Some(0));
}

#[test]
fn ranged_split_tasks_count_the_file_once() {
    let mut first = task("data/a.parquet", Some(A_ROWS), None);
    first.start = 0;
    first.length = 2048;
    first.record_count = None;
    let mut second = task("data/a.parquet", Some(A_ROWS), None);
    second.start = 2048;
    second.length = 2048;
    second.record_count = None;

    assert_eq!(
        exact_table_row_count(&work(vec![first, second])),
        usize::try_from(A_ROWS).ok(),
        "two ranged tasks over one file must sum its file record count once"
    );
}

#[test]
fn unknown_file_record_count_refuses_exact() {
    let works = work(vec![
        task("data/a.parquet", Some(A_ROWS), None),
        task("data/b.parquet", None, None),
    ]);
    assert_eq!(exact_table_row_count(&works), None);
}

#[test]
fn always_true_residual_is_countable() {
    let works = work(vec![task(
        "data/a.parquet",
        Some(A_ROWS),
        Some(BoundPredicate::AlwaysTrue),
    )]);
    assert_eq!(exact_table_row_count(&works), usize::try_from(A_ROWS).ok());
}

#[test]
fn non_trivial_residual_refuses_exact() {
    let works = work(vec![task(
        "data/a.parquet",
        Some(A_ROWS),
        Some(BoundPredicate::AlwaysFalse),
    )]);
    assert_eq!(exact_table_row_count(&works), None);
}

#[test]
fn overflow_refuses_exact() {
    let works = work(vec![
        task("data/a.parquet", Some(u64::MAX), None),
        task("data/b.parquet", Some(1), None),
    ]);
    assert_eq!(exact_table_row_count(&works), None);
}
