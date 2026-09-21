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

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray, StructArray};
use futures::TryStreamExt;
use iceberg::inspect::{
    EntriesTable, FilesTable, ManifestsTable, PartitionsTable, PositionDeletesTable,
};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::scan::ArrowRecordBatchStream;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PrimitiveType,
    Schema, SortOrder, Struct, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};
use tempfile::TempDir;

const UNKNOWN_SNAPSHOT_ID: i64 = 999_999_999_999;

fn gen_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the {id long, data string} schema")
}

async fn empty_table(name: &str) -> (TempDir, MemoryCatalog, Table) {
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let namespace = NamespaceIdent::new("snapshot_scope".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location)
        .schema(gen_schema())
        .partition_spec(UnboundPartitionSpec::builder().build())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table");
    (tmp, catalog, table)
}

async fn write_data_file(table: &Table, ids: &[i64], filename: &str) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow"));
    let data: Vec<String> = ids.iter().map(|id| format!("d{id}")).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(data)) as ArrayRef,
    ])
    .expect("build the data batch");
    let file_path = format!("{}/data/{filename}", table.metadata().location());
    let output = table.file_io().new_output(file_path).expect("new output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder
        .build(output)
        .await
        .expect("build parquet writer");
    writer.write(&batch).await.expect("write data batch");
    let builders = writer.close().await.expect("close parquet writer");
    let mut builder = builders.into_iter().next().expect("one data file builder");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("build unpartitioned data file")
}

async fn two_snapshot_table() -> (TempDir, MemoryCatalog, Table, i64, i64, String, String) {
    let (tmp, catalog, table) = empty_table("snapshot_scope").await;
    let file_a = write_data_file(&table, &[1], "00000-snap-scope-a.parquet").await;
    let path_a = file_a.file_path().to_owned();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_a.clone()])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");
    let snap1 = table
        .metadata()
        .current_snapshot_id()
        .expect("snap1 is current");
    let file_b = write_data_file(&table, &[2], "00000-snap-scope-b.parquet").await;
    let path_b = file_b.file_path().to_owned();
    let tx = Transaction::new(&table);
    let tx = tx
        .rewrite_files(vec![file_a], vec![file_b])
        .apply(tx)
        .expect("apply rewrite files");
    let table = tx.commit(&catalog).await.expect("commit rewrite files");
    let snap2 = table
        .metadata()
        .current_snapshot_id()
        .expect("snap2 is current");
    assert_ne!(snap1, snap2);
    (tmp, catalog, table, snap1, snap2, path_a, path_b)
}

async fn collect_batches(stream: ArrowRecordBatchStream) -> Vec<RecordBatch> {
    stream
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect inspection batches")
}

fn string_column(batches: &[RecordBatch], name: &str) -> Vec<String> {
    let mut values = Vec::new();
    for batch in batches {
        let col = batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("{name} column"))
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("{name} is Utf8"));
        for i in 0..batch.num_rows() {
            values.push(col.value(i).to_string());
        }
    }
    values.sort();
    values
}

fn long_column(batches: &[RecordBatch], name: &str) -> Vec<i64> {
    let mut values = Vec::new();
    for batch in batches {
        let col = batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("{name} column"))
            .as_primitive::<Int64Type>();
        for i in 0..batch.num_rows() {
            values.push(col.value(i));
        }
    }
    values.sort();
    values
}

fn entries_file_status(batches: &[RecordBatch]) -> Vec<(String, i32)> {
    let mut rows = Vec::new();
    for batch in batches {
        let status = batch
            .column_by_name("status")
            .unwrap_or_else(|| panic!("status column"))
            .as_primitive::<Int32Type>();
        let data_file = batch
            .column_by_name("data_file")
            .unwrap_or_else(|| panic!("data_file column"))
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("data_file is a struct");
        let paths = data_file
            .column_by_name("file_path")
            .unwrap_or_else(|| panic!("file_path field"))
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("file_path is Utf8");
        for i in 0..batch.num_rows() {
            rows.push((paths.value(i).to_string(), status.value(i)));
        }
    }
    rows.sort();
    rows
}

#[tokio::test]
async fn files_at_snapshot_reads_history_not_current() {
    let (_tmp, _catalog, table, snap1, snap2, path_a, path_b) = two_snapshot_table().await;
    let at_snap1 = collect_batches(
        FilesTable::try_all_at_snapshot(&table, snap1)
            .expect("files at snap1")
            .scan()
            .await
            .expect("scan files at snap1"),
    )
    .await;
    assert_eq!(string_column(&at_snap1, "file_path"), vec![path_a]);
    let at_snap2 = collect_batches(
        FilesTable::try_all_at_snapshot(&table, snap2)
            .expect("files at snap2")
            .scan()
            .await
            .expect("scan files at snap2"),
    )
    .await;
    assert_eq!(string_column(&at_snap2, "file_path"), vec![path_b.clone()]);
    let live = collect_batches(
        table
            .inspect()
            .files()
            .scan()
            .await
            .expect("scan live files"),
    )
    .await;
    assert_eq!(string_column(&live, "file_path"), vec![path_b]);
}

#[tokio::test]
async fn data_and_delete_files_at_snapshot_follow_content() {
    let (_tmp, _catalog, table, snap1, _snap2, path_a, _path_b) = two_snapshot_table().await;
    let data = collect_batches(
        FilesTable::try_data_at_snapshot(&table, snap1)
            .expect("data files at snap1")
            .scan()
            .await
            .expect("scan data files at snap1"),
    )
    .await;
    assert_eq!(string_column(&data, "file_path"), vec![path_a]);
    let deletes = collect_batches(
        FilesTable::try_deletes_at_snapshot(&table, snap1)
            .expect("delete files at snap1")
            .scan()
            .await
            .expect("scan delete files at snap1"),
    )
    .await;
    assert!(string_column(&deletes, "file_path").is_empty());
}

#[tokio::test]
async fn entries_at_snapshot_reads_history_not_current() {
    let (_tmp, _catalog, table, snap1, _snap2, path_a, path_b) = two_snapshot_table().await;
    let at_snap1 = collect_batches(
        EntriesTable::try_at_snapshot(&table, snap1)
            .expect("entries at snap1")
            .scan()
            .await
            .expect("scan entries at snap1"),
    )
    .await;
    assert_eq!(entries_file_status(&at_snap1), vec![(path_a.clone(), 1)]);
    let live = collect_batches(
        table
            .inspect()
            .entries()
            .scan()
            .await
            .expect("scan live entries"),
    )
    .await;
    let live_rows = entries_file_status(&live);
    assert!(
        live_rows.iter().any(|row| row.0 == path_b),
        "live entries must contain file B"
    );
    assert!(
        !live_rows.iter().any(|row| row.0 == path_a && row.1 == 1),
        "file A must not be live at current"
    );
}

#[tokio::test]
async fn manifests_at_snapshot_reads_history_not_current() {
    let (_tmp, _catalog, table, snap1, _snap2, _path_a, _path_b) = two_snapshot_table().await;
    let at_snap1 = collect_batches(
        ManifestsTable::at_snapshot(&table, snap1)
            .scan()
            .await
            .expect("scan manifests at snap1"),
    )
    .await;
    let snap1_paths = string_column(&at_snap1, "path");
    assert!(!snap1_paths.is_empty());
    assert_eq!(long_column(&at_snap1, "added_snapshot_id"), vec![snap1]);
    let live = collect_batches(
        table
            .inspect()
            .manifests()
            .scan()
            .await
            .expect("scan live manifests"),
    )
    .await;
    let live_paths = string_column(&live, "path");
    assert!(!live_paths.is_empty());
    assert!(
        snap1_paths.iter().all(|path| !live_paths.contains(path)),
        "snap1 manifests must not be referenced by current"
    );
}

#[tokio::test]
async fn unknown_snapshot_id_fails_loud() {
    let (_tmp, _catalog, table, _snap1, _snap2, _path_a, _path_b) = two_snapshot_table().await;
    let err = match FilesTable::try_all_at_snapshot(&table, UNKNOWN_SNAPSHOT_ID)
        .expect("construct files at unknown id")
        .scan()
        .await
    {
        Ok(_) => panic!("unknown snapshot id must not scan files"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(err.message().contains("Cannot find snapshot"));
    let err = match EntriesTable::try_at_snapshot(&table, UNKNOWN_SNAPSHOT_ID)
        .expect("construct entries at unknown id")
        .scan()
        .await
    {
        Ok(_) => panic!("unknown snapshot id must not scan entries"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(err.message().contains("Cannot find snapshot"));
    let err = match ManifestsTable::at_snapshot(&table, UNKNOWN_SNAPSHOT_ID)
        .scan()
        .await
    {
        Ok(_) => panic!("unknown snapshot id must not scan manifests"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(err.message().contains("Cannot find snapshot"));
}

#[tokio::test]
async fn empty_table_without_snapshot_scans_empty() {
    let (_tmp, _catalog, table) = empty_table("snapshot_scope_empty").await;
    assert!(table.metadata().current_snapshot_id().is_none());
    let files = collect_batches(table.inspect().files().scan().await.expect("scan files")).await;
    assert!(string_column(&files, "file_path").is_empty());
    let entries = collect_batches(
        table
            .inspect()
            .entries()
            .scan()
            .await
            .expect("scan entries"),
    )
    .await;
    assert!(entries_file_status(&entries).is_empty());
    let manifests = collect_batches(
        table
            .inspect()
            .manifests()
            .scan()
            .await
            .expect("scan manifests"),
    )
    .await;
    assert!(string_column(&manifests, "path").is_empty());
}

fn data_spec() -> UnboundPartitionSpec {
    UnboundPartitionSpec::builder()
        .add_partition_field(2, "data", Transform::Identity)
        .expect("bind identity(data)")
        .build()
}

async fn partitioned_empty_table(name: &str) -> (TempDir, MemoryCatalog, Table) {
    let tmp = TempDir::new().expect("temp dir");
    let warehouse = tmp.path().to_string_lossy().to_string();
    let table_location = format!("{warehouse}/rust_table");
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            name,
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.clone())]),
        )
        .await
        .expect("build MemoryCatalog over local FS");
    let namespace = NamespaceIdent::new("snapshot_scope".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("rust_table".to_string())
        .location(table_location)
        .schema(gen_schema())
        .partition_spec(data_spec())
        .sort_order(SortOrder::unsorted_order())
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create rust_table");
    (tmp, catalog, table)
}

async fn write_partitioned_data_file(
    table: &Table,
    ids: &[i64],
    values: &[&str],
    filename: &str,
    partition: Struct,
) -> DataFile {
    use iceberg::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(values.to_vec())) as ArrayRef,
    ])
    .expect("build the data batch");
    let file_path = format!("{}/data/{filename}", table.metadata().location());
    let output = table.file_io().new_output(file_path).expect("new output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder
        .build(output)
        .await
        .expect("build parquet writer");
    writer.write(&batch).await.expect("write data batch");
    let builders = writer.close().await.expect("close parquet writer");
    let mut builder = builders.into_iter().next().expect("one data file builder");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(partition)
        .build()
        .expect("build partitioned data file")
}

async fn partition_swap_table() -> (TempDir, MemoryCatalog, Table, i64, i64) {
    let (tmp, catalog, table) = partitioned_empty_table("snapshot_scope_partitions").await;
    let file_a = write_partitioned_data_file(
        &table,
        &[1],
        &["a"],
        "00000-part-a.parquet",
        Struct::from_iter([Some(Literal::string("a"))]),
    )
    .await;
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_a.clone()])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");
    let snap1 = table
        .metadata()
        .current_snapshot_id()
        .expect("snap1 is current");
    let file_b = write_partitioned_data_file(
        &table,
        &[2],
        &["b"],
        "00000-part-b.parquet",
        Struct::from_iter([Some(Literal::string("b"))]),
    )
    .await;
    let tx = Transaction::new(&table);
    let tx = tx
        .rewrite_files(vec![file_a], vec![file_b])
        .apply(tx)
        .expect("apply rewrite files");
    let table = tx.commit(&catalog).await.expect("commit rewrite files");
    let snap2 = table
        .metadata()
        .current_snapshot_id()
        .expect("snap2 is current");
    assert_ne!(snap1, snap2);
    (tmp, catalog, table, snap1, snap2)
}

fn partition_data_values(batches: &[RecordBatch]) -> Vec<String> {
    let mut values = Vec::new();
    for batch in batches {
        let structs = batch
            .column_by_name("partition")
            .unwrap_or_else(|| panic!("partition column"))
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("partition is a struct");
        let data = structs
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("data child is Utf8");
        for i in 0..batch.num_rows() {
            values.push(data.value(i).to_string());
        }
    }
    values.sort();
    values
}

async fn write_pos_deletes(table: &Table, pairs: &[(String, i64)]) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("position-delete writer config");
    let location_gen = DefaultLocationGenerator::new(table.metadata()).expect("location generator");
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
        .unpartitioned()
        .build(None)
        .await
        .expect("build position-delete writer");
    let paths: Vec<&str> = pairs.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("build position-delete batch");
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

async fn posdel_two_snapshot_table() -> (TempDir, MemoryCatalog, Table, i64, i64, String) {
    let (tmp, catalog, table) = empty_table("snapshot_scope_posdel").await;
    let file_a = write_data_file(&table, &[1], "00000-posdel-a.parquet").await;
    let path_a = file_a.file_path().to_owned();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file_a])
        .apply(tx)
        .expect("apply fast append");
    let table = tx.commit(&catalog).await.expect("commit fast append");
    let snap1 = table
        .metadata()
        .current_snapshot_id()
        .expect("snap1 is current");
    let deletes = write_pos_deletes(&table, &[(path_a.clone(), 0)]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![deletes])
        .apply(tx)
        .expect("apply row delta");
    let table = tx.commit(&catalog).await.expect("commit row delta");
    let snap2 = table
        .metadata()
        .current_snapshot_id()
        .expect("snap2 is current");
    assert_ne!(snap1, snap2);
    (tmp, catalog, table, snap1, snap2, path_a)
}

#[tokio::test]
async fn partitions_at_snapshot_reads_history_not_current() {
    let (_tmp, _catalog, table, snap1, _snap2) = partition_swap_table().await;
    let at_snap1 = collect_batches(
        PartitionsTable::try_at_snapshot(&table, snap1)
            .expect("partitions at snap1")
            .scan()
            .await
            .expect("scan partitions at snap1"),
    )
    .await;
    assert_eq!(partition_data_values(&at_snap1), vec!["a".to_string()]);
    let live = collect_batches(
        table
            .inspect()
            .partitions()
            .scan()
            .await
            .expect("scan live partitions"),
    )
    .await;
    assert_eq!(partition_data_values(&live), vec!["b".to_string()]);
}

#[tokio::test]
async fn position_deletes_at_snapshot_reads_history_not_current() {
    let (_tmp, _catalog, table, snap1, _snap2, path_a) = posdel_two_snapshot_table().await;
    let at_snap1 = collect_batches(
        PositionDeletesTable::try_at_snapshot(&table, snap1)
            .expect("position deletes at snap1")
            .scan()
            .await
            .expect("scan position deletes at snap1"),
    )
    .await;
    assert!(string_column(&at_snap1, "file_path").is_empty());
    let live = collect_batches(
        table
            .inspect()
            .position_deletes()
            .scan()
            .await
            .expect("scan live position deletes"),
    )
    .await;
    assert_eq!(string_column(&live, "file_path"), vec![path_a]);
    assert_eq!(long_column(&live, "pos"), vec![0]);
}

#[tokio::test]
async fn partitions_and_position_deletes_unknown_snapshot_id_fails_loud() {
    let (_tmp, _catalog, table, _snap1, _snap2, _path_a, _path_b) = two_snapshot_table().await;
    let err = match PartitionsTable::try_at_snapshot(&table, UNKNOWN_SNAPSHOT_ID)
        .expect("construct partitions at unknown id")
        .scan()
        .await
    {
        Ok(_) => panic!("unknown snapshot id must not scan partitions"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(err.message().contains("Cannot find snapshot"));
    let err = match PositionDeletesTable::try_at_snapshot(&table, UNKNOWN_SNAPSHOT_ID)
        .expect("construct position deletes at unknown id")
        .scan()
        .await
    {
        Ok(_) => panic!("unknown snapshot id must not scan position deletes"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(err.message().contains("Cannot find snapshot"));
}

#[tokio::test]
async fn snapshotless_scoped_tables_fail_loud_but_unscoped_scan_empty() {
    let (_tmp, _catalog, table) = empty_table("snapshot_scope_empty_scoped").await;
    assert!(table.metadata().current_snapshot_id().is_none());
    let err = match PartitionsTable::try_at_snapshot(&table, UNKNOWN_SNAPSHOT_ID)
        .expect("construct partitions at unknown id")
        .scan()
        .await
    {
        Ok(_) => panic!("snapshot-less scoped partitions must fail"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(err.message().contains("Cannot find snapshot"));
    let err = match PositionDeletesTable::try_at_snapshot(&table, UNKNOWN_SNAPSHOT_ID)
        .expect("construct position deletes at unknown id")
        .scan()
        .await
    {
        Ok(_) => panic!("snapshot-less scoped position deletes must fail"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(err.message().contains("Cannot find snapshot"));
    let partitions = collect_batches(
        table
            .inspect()
            .partitions()
            .scan()
            .await
            .expect("scan partitions"),
    )
    .await;
    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].num_rows(), 0);
    let posdel = collect_batches(
        table
            .inspect()
            .position_deletes()
            .scan()
            .await
            .expect("scan position deletes"),
    )
    .await;
    assert!(posdel.iter().all(|batch| batch.num_rows() == 0));
}
