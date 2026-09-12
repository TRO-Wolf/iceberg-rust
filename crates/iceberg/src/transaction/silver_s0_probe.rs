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

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, TimeUnit};

use crate::arrow::{
    RecordBatchPartitionSplitter, UTC_TIME_ZONE, arrow_type_to_type, schema_to_arrow_schema,
    type_to_arrow_type,
};
use crate::expr::Predicate;
use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Operation, PrimitiveType,
    Struct, Type,
};
use crate::table::Table;
use crate::transaction::action::TransactionAction;
use crate::transaction::tests::make_v2_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, ErrorKind, TableRequirement};

const RUN_KEY: &str = "repark.silver.run-key";

fn synthetic_data_file(path: &str, part_value: i64, record_count: u64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(record_count)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(part_value))]))
        .build()
        .unwrap()
}

async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

async fn write_parquet_data_file(
    table: &Table,
    file_stem: &str,
    x: i64,
    y: i64,
    z: i64,
) -> DataFile {
    let schema = table.metadata().current_schema();
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(vec![x])) as ArrayRef,
        Arc::new(Int64Array::from(vec![y])) as ArrayRef,
        Arc::new(Int64Array::from(vec![z])) as ArrayRef,
    ])
    .unwrap();
    let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
    let file_name_gen = DefaultFileNameGenerator::new(
        file_stem.to_string(),
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
    let builder = DataFileWriterBuilder::new(rolling).with_partition_spec(spec);
    let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
        schema.clone(),
        table.metadata().default_partition_spec().clone(),
    )
    .unwrap();
    let splits = splitter.split(&batch).unwrap();
    assert_eq!(splits.len(), 1);
    let (partition_key, partition_batch) = splits.into_iter().next().unwrap();
    let mut writer = builder.build(Some(partition_key)).await.unwrap();
    writer.write(partition_batch).await.unwrap();
    writer.close().await.unwrap().into_iter().next().unwrap()
}

fn snapshot_id_for_run_key(table: &Table, value: &str) -> Option<i64> {
    table.metadata().snapshots().find_map(|snapshot| {
        snapshot
            .summary()
            .additional_properties
            .get(RUN_KEY)
            .filter(|found| found.as_str() == value)
            .map(|_| snapshot.snapshot_id())
    })
}

fn summary_prop(table: &Table, key: &str) -> Option<String> {
    table
        .metadata()
        .current_snapshot()
        .unwrap()
        .summary()
        .additional_properties
        .get(key)
        .cloned()
}

#[tokio::test]
async fn parquet_data_file_write_does_not_advance_snapshot_and_later_commit_reuses_files() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let ident = table.identifier().clone();
    let before = table.metadata().current_snapshot_id();
    let data_file = write_parquet_data_file(&table, "silver-stage", 1, 2, 3).await;
    let reloaded = catalog.load_table(&ident).await.unwrap();
    assert_eq!(reloaded.metadata().current_snapshot_id(), before);
    assert!(data_file.file_path().ends_with(".parquet"));
    assert_eq!(data_file.record_count(), 1);
    let staged_path = data_file.file_path().to_string();
    let committed = append_files(&catalog, &reloaded, vec![data_file]).await;
    assert_ne!(committed.metadata().current_snapshot_id(), before);
    let live = committed
        .metadata()
        .current_snapshot()
        .unwrap()
        .summary()
        .additional_properties
        .get("added-data-files")
        .cloned();
    assert_eq!(live.as_deref(), Some("1"));
    assert!(staged_path.contains("data"));
}

#[tokio::test]
async fn overwrite_without_conflict_validation_rebases_over_concurrent_append() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/base.parquet",
        0,
        1,
    )])
    .await;
    let expected_base = table.metadata().current_snapshot_id();
    let staged = synthetic_data_file("test/staged.parquet", 1, 3);
    let concurrent = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/concurrent.parquet",
        2,
        5,
    )])
    .await;
    assert_ne!(concurrent.metadata().current_snapshot_id(), expected_base);
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(Predicate::AlwaysTrue)
        .add_files(vec![staged]);
    let tx = action.apply(tx).unwrap();
    let published = tx.commit(&catalog).await.unwrap();
    assert_ne!(published.metadata().current_snapshot_id(), expected_base);
    assert_ne!(
        published.metadata().current_snapshot_id(),
        concurrent.metadata().current_snapshot_id()
    );
    assert_eq!(
        published
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        Operation::Overwrite
    );
    assert_eq!(
        summary_prop(&published, "deleted-data-files").as_deref(),
        Some("2")
    );
    assert_eq!(
        summary_prop(&published, "added-data-files").as_deref(),
        Some("1")
    );
}

#[tokio::test]
async fn overwrite_with_validate_no_conflicting_data_refuses_concurrent_append() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/base.parquet",
        0,
        1,
    )])
    .await;
    let expected_base = table.metadata().current_snapshot_id();
    let staged = synthetic_data_file("test/staged.parquet", 1, 3);
    let concurrent = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/concurrent.parquet",
        2,
        5,
    )])
    .await;
    assert_ne!(concurrent.metadata().current_snapshot_id(), expected_base);
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(Predicate::AlwaysTrue)
        .add_files(vec![staged])
        .validate_no_conflicting_data();
    let tx = action.apply(tx).unwrap();
    let error = tx.commit(&catalog).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(!error.retryable());
    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        concurrent.metadata().current_snapshot_id()
    );
}

#[tokio::test]
async fn concurrent_append_before_staging_then_overwrite_without_validation_still_rebases() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/base.parquet",
        0,
        1,
    )])
    .await;
    let expected_base = table.metadata().current_snapshot_id();
    let concurrent = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/concurrent.parquet",
        2,
        5,
    )])
    .await;
    let staged = synthetic_data_file("test/staged.parquet", 1, 3);
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(Predicate::AlwaysTrue)
        .add_files(vec![staged]);
    let tx = action.apply(tx).unwrap();
    let published = tx.commit(&catalog).await.unwrap();
    assert_ne!(published.metadata().current_snapshot_id(), expected_base);
    assert_ne!(
        published.metadata().current_snapshot_id(),
        concurrent.metadata().current_snapshot_id()
    );
    assert_eq!(
        summary_prop(&published, "deleted-data-files").as_deref(),
        Some("2")
    );
}

#[tokio::test]
async fn zero_commit_retries_still_rebases_overwrite_over_concurrent_append() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .update_table_properties()
        .set("commit.retry.num-retries".to_string(), "0".to_string())
        .apply(tx)
        .unwrap();
    let table = tx.commit(&catalog).await.unwrap();
    let table = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/base.parquet",
        0,
        1,
    )])
    .await;
    let staged = synthetic_data_file("test/staged.parquet", 1, 3);
    let concurrent = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/concurrent.parquet",
        2,
        5,
    )])
    .await;
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(Predicate::AlwaysTrue)
        .add_files(vec![staged]);
    let tx = action.apply(tx).unwrap();
    let published = tx.commit(&catalog).await.unwrap();
    assert_ne!(
        published.metadata().current_snapshot_id(),
        concurrent.metadata().current_snapshot_id()
    );
    assert_eq!(
        summary_prop(&published, "deleted-data-files").as_deref(),
        Some("2")
    );
}

#[tokio::test]
async fn run_key_in_snapshot_summary_is_findable_from_table_metadata() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/base.parquet",
        0,
        1,
    )])
    .await;
    let run_value = "plan-abc/bronze-snap-1";
    let mut properties = HashMap::new();
    properties.insert(RUN_KEY.to_string(), run_value.to_string());
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(Predicate::AlwaysTrue)
        .add_files(vec![synthetic_data_file("test/silver.parquet", 1, 2)])
        .set_snapshot_properties(properties);
    let tx = action.apply(tx).unwrap();
    let published = tx.commit(&catalog).await.unwrap();
    let found = snapshot_id_for_run_key(&published, run_value);
    assert_eq!(found, published.metadata().current_snapshot_id());
    let reloaded = catalog.load_table(published.identifier()).await.unwrap();
    assert_eq!(
        snapshot_id_for_run_key(&reloaded, run_value),
        reloaded.metadata().current_snapshot_id()
    );
}

#[tokio::test]
async fn expire_snapshot_id_drops_run_key_from_metadata() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/base.parquet",
        0,
        1,
    )])
    .await;
    let run_value = "plan-expire/bronze-snap-1";
    let mut properties = HashMap::new();
    properties.insert(RUN_KEY.to_string(), run_value.to_string());
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(Predicate::AlwaysTrue)
        .add_files(vec![synthetic_data_file("test/marked.parquet", 1, 2)])
        .set_snapshot_properties(properties);
    let tx = action.apply(tx).unwrap();
    let marked = tx.commit(&catalog).await.unwrap();
    let marked_id = marked.metadata().current_snapshot_id().unwrap();
    assert_eq!(snapshot_id_for_run_key(&marked, run_value), Some(marked_id));
    let successor = append_files(&catalog, &marked, vec![synthetic_data_file(
        "test/successor.parquet",
        3,
        1,
    )])
    .await;
    assert!(snapshot_id_for_run_key(&successor, run_value).is_some());
    let tx = Transaction::new(&successor);
    let tx = tx
        .expire_snapshots()
        .expire_snapshot_id(marked_id)
        .apply(tx)
        .unwrap();
    let expired = tx.commit(&catalog).await.unwrap();
    assert!(snapshot_id_for_run_key(&expired, run_value).is_none());
    assert!(expired.metadata().snapshot_by_id(marked_id).is_none());
    assert_eq!(
        expired.metadata().current_snapshot_id(),
        successor.metadata().current_snapshot_id()
    );
}

#[tokio::test]
async fn fast_append_commit_emits_uuid_and_ref_snapshot_requirements() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/base.parquet",
        0,
        1,
    )])
    .await;
    let expected_uuid = table.metadata().uuid();
    let expected_snapshot = table.metadata().current_snapshot_id();
    let action = Transaction::new(&table)
        .fast_append()
        .add_data_files(vec![synthetic_data_file("test/next.parquet", 1, 1)]);
    let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
    let requirements = action_commit.take_requirements();
    assert!(requirements.iter().any(|requirement| matches!(
        requirement,
        TableRequirement::UuidMatch { uuid } if *uuid == expected_uuid
    )));
    assert!(requirements.iter().any(|requirement| matches!(
        requirement,
        TableRequirement::RefSnapshotIdMatch { snapshot_id, .. }
            if *snapshot_id == expected_snapshot
    )));
}

#[tokio::test]
async fn overwrite_by_always_true_replaces_all_live_files_in_one_snapshot() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![
        synthetic_data_file("test/a.parquet", 0, 1),
        synthetic_data_file("test/b.parquet", 1, 1),
    ])
    .await;
    let before = table.metadata().current_snapshot_id();
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(Predicate::AlwaysTrue)
        .add_files(vec![synthetic_data_file("test/new.parquet", 2, 4)]);
    let tx = action.apply(tx).unwrap();
    let published = tx.commit(&catalog).await.unwrap();
    assert_ne!(published.metadata().current_snapshot_id(), before);
    assert_eq!(
        published
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        Operation::Overwrite
    );
    assert_eq!(
        summary_prop(&published, "deleted-data-files").as_deref(),
        Some("2")
    );
    assert_eq!(
        summary_prop(&published, "added-data-files").as_deref(),
        Some("1")
    );
    assert_eq!(
        summary_prop(&published, "added-records").as_deref(),
        Some("4")
    );
    assert_eq!(
        summary_prop(&published, "deleted-records").as_deref(),
        Some("2")
    );
}

#[test]
fn mvp_iceberg_arrow_types_round_trip() {
    let cases = [
        (Type::Primitive(PrimitiveType::Int), DataType::Int32),
        (Type::Primitive(PrimitiveType::Long), DataType::Int64),
        (Type::Primitive(PrimitiveType::String), DataType::Utf8),
        (Type::Primitive(PrimitiveType::Boolean), DataType::Boolean),
        (Type::Primitive(PrimitiveType::Date), DataType::Date32),
        (
            Type::Primitive(PrimitiveType::Timestamptz),
            DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into())),
        ),
        (
            Type::Primitive(PrimitiveType::Decimal {
                precision: 10,
                scale: 2,
            }),
            DataType::Decimal128(10, 2),
        ),
    ];
    for (iceberg_type, arrow_type) in cases {
        assert_eq!(type_to_arrow_type(&iceberg_type).unwrap(), arrow_type);
        assert_eq!(arrow_type_to_type(&arrow_type).unwrap(), iceberg_type);
    }
}
