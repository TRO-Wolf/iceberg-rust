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

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;

use crate::error::ErrorKind;
use crate::maintenance::RewriteJobOrder;
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    append_files, config_for, create_partitioned_table, live_data_file_paths, local_fs_catalog,
    scan_rows, synthetic_spec_and_schema, synthetic_task, write_data_file,
};
use crate::maintenance::rewrite_data_files_plan::{
    format_java_double, input_split_size, plan_commit_batches, plan_read_tasks,
};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Literal, NestedField, PartitionKey, PartitionSpec,
    PrimitiveType, Schema, Struct, Transform, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, NamespaceIdent, TableCreation};

pub(crate) async fn two_partitions_by_four_files(catalog: &impl Catalog) -> crate::table::Table {
    let table = create_partitioned_table(catalog, crate::spec::FormatVersion::V2).await;
    for part in [0i64, 1] {
        for index in 0..4i64 {
            let file = write_data_file(&table, &format!("p{part}-{index}.parquet"), part, &[(
                part,
                part * 100 + index,
                index,
            )])
            .await;
            append_files(catalog, &table, vec![file]).await;
        }
    }
    catalog
        .load_table(table.identifier())
        .await
        .expect("reload oracle-shape table")
}

fn l001_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "p",
                Type::Primitive(PrimitiveType::Int),
            )),
            Arc::new(NestedField::required(
                3,
                "v",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build (id, p, v) schema")
}

async fn create_l001_table(catalog: &impl Catalog, name: &str) -> Table {
    let schema = l001_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("p", "p", Transform::Identity)
        .expect("add partition field")
        .build()
        .expect("build spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name(name.to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(crate::spec::FormatVersion::V2)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

async fn write_l001_file(table: &Table, tag: &str, p: i32, start: i64) -> DataFile {
    use crate::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from_iter_values(start..start + 50)) as ArrayRef,
        Arc::new(Int32Array::from(vec![p; 50])) as ArrayRef,
        Arc::new(StringArray::from(vec!["x".repeat(20); 50])) as ArrayRef,
    ])
    .expect("build (id, p, v) batch");
    let location_gen = DefaultLocationGenerator::new(table.metadata()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        format!("data-{tag}"),
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
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        schema.clone(),
        Struct::from_iter([Some(Literal::int(p))]),
    )
    .expect("build partition key");
    let mut writer = DataFileWriterBuilder::new(rolling)
        .build(Some(partition_key))
        .await
        .expect("build data file writer");
    writer.write(batch).await.expect("write data batch");
    writer
        .close()
        .await
        .expect("close data writer")
        .into_iter()
        .next()
        .expect("exactly one data file")
}

async fn l001_dropped_table(catalog: &impl Catalog, name: &str, order: &[i32]) -> Table {
    let table = create_l001_table(catalog, name).await;
    let mut next_id = 0i64;
    for (file_index, p) in order.iter().enumerate() {
        let file = write_l001_file(&table, &format!("{name}-{file_index}"), *p, next_id).await;
        next_id += 50;
        append_files(catalog, &table, vec![file]).await;
    }
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload oracle-shape table");
    drop_partition_field_named(catalog, &table, "p").await
}

async fn drop_partition_field_named(catalog: &impl Catalog, table: &Table, name: &str) -> Table {
    let action = Transaction::new(table)
        .update_partition_spec()
        .remove_field(name);
    let tx = Transaction::new(table);
    action
        .apply(tx)
        .expect("apply spec drop")
        .commit(catalog)
        .await
        .expect("commit spec drop")
}

async fn scan_l001_rows(table: &Table) -> Vec<(i64, i32, String)> {
    let stream = table
        .scan()
        .select(["id", "p", "v"])
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect batches");
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("id")
            .expect("id column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id is long");
        let ps = batch
            .column_by_name("p")
            .expect("p column")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("p is int");
        let vs = batch
            .column_by_name("v")
            .expect("v column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("v is string");
        for index in 0..batch.num_rows() {
            rows.push((
                ids.value(index),
                ps.value(index),
                vs.value(index).to_string(),
            ));
        }
    }
    rows.sort();
    rows
}

async fn live_data_spec_ids(table: &Table) -> HashSet<i32> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    let mut ids = HashSet::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                ids.insert(entry.data_file().partition_spec_id());
            }
        }
    }
    ids
}

async fn drop_partition_field(catalog: &impl Catalog, table: &Table) -> Table {
    let action = Transaction::new(table)
        .update_partition_spec()
        .remove_field("x");
    let tx = Transaction::new(table);
    action
        .apply(tx)
        .expect("apply spec drop")
        .commit(catalog)
        .await
        .expect("commit spec drop")
}

fn snapshot_count(table: &Table) -> usize {
    table.metadata().snapshots().count()
}

#[tokio::test]
async fn test_default_commits_all_groups_in_one_snapshot() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = two_partitions_by_four_files(&catalog).await;
    assert_eq!(
        snapshot_count(&table),
        8,
        "fixture: 8 single-file appends, one snapshot each"
    );
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 8);

    let result = RewriteDataFiles::new(table.clone())
        .min_input_files(1)
        .execute(&catalog)
        .await
        .expect("compaction must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.file_groups.len(), 2);
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        9,
        "default rewrites all groups in ONE commit (Java RewriteDataFilesCommitManager without partial progress)"
    );
    assert_eq!(live_data_file_paths(&table).await.len(), 2);
    assert_eq!(scan_rows(&table).await, rows_before);
}

#[tokio::test]
async fn test_rewrite_all_selects_and_qualifies_everything() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
    let rows: Vec<(i64, i64, i64)> = (0..100).map(|n| (0, n, n)).collect();
    let first = write_data_file(&table, "w1.parquet", 0, &rows).await;
    let size = first.file_size_in_bytes();
    let second = write_data_file(&table, "w2.parquet", 0, &rows).await;
    let table = append_files(&catalog, &table, vec![first, second]).await;
    let rows_before = scan_rows(&table).await;

    let well_sized = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(size)
        .min_file_size_bytes(size / 2)
        .max_file_size_bytes(size * 2);
    let idle = well_sized
        .execute(&catalog)
        .await
        .expect("empty plan must succeed");
    assert_eq!(
        idle,
        crate::maintenance::RewriteDataFilesResult::default(),
        "fixture: two well-sized files are selected by nothing at defaults"
    );

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(size)
        .min_file_size_bytes(size / 2)
        .max_file_size_bytes(size * 2)
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("rewrite_all must succeed");
    assert_eq!(result.rewritten_data_files_count, 2);
    assert!(result.added_data_files_count >= 1);
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(scan_rows(&table).await, rows_before);

    let lone_table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
    let lone = write_data_file(&lone_table, "lone.parquet", 0, &[(0, 1, 1)]).await;
    let lone_table = append_files(&catalog, &lone_table, vec![lone]).await;
    let lone_result = RewriteDataFiles::new(lone_table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("rewrite_all rewrites even a lone file");
    assert_eq!(lone_result.rewritten_data_files_count, 1);
}

#[tokio::test]
async fn test_partial_progress_commits_one_group_per_commit() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = two_partitions_by_four_files(&catalog).await;
    let rows_before = scan_rows(&table).await;

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .partial_progress(true)
        .execute(&catalog)
        .await
        .expect("partial progress must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.file_groups.len(), 2);
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        10,
        "partial progress commits each of the 2 groups alone"
    );
    assert_eq!(scan_rows(&table).await, rows_before);
}

#[tokio::test]
async fn test_partial_progress_max_commits_1_folds_into_one_commit() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = two_partitions_by_four_files(&catalog).await;

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .partial_progress(true)
        .partial_progress_max_commits(1)
        .execute(&catalog)
        .await
        .expect("partial progress with max-commits 1 must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.file_groups.len(), 2);
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        9,
        "ceil(2/1) = 2 groups per commit, so one commit"
    );
}

#[tokio::test]
async fn test_partial_progress_4_groups_max_commits_3_needs_2_commits() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
    for part in [0i64, 1, 2, 3] {
        for index in 0..2i64 {
            let file = write_data_file(&table, &format!("q{part}-{index}.parquet"), part, &[(
                part, index, index,
            )])
            .await;
            append_files(&catalog, &table, vec![file]).await;
        }
    }
    let table = catalog.load_table(table.identifier()).await.unwrap();
    let rows_before = scan_rows(&table).await;

    let result = RewriteDataFiles::new(table.clone())
        .min_input_files(2)
        .partial_progress(true)
        .partial_progress_max_commits(3)
        .execute(&catalog)
        .await
        .expect("partial progress over 4 groups must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 4);
    assert_eq!(result.file_groups.len(), 4);
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        10,
        "ceil(4/3) = 2 groups per commit, so two commits"
    );
    assert_eq!(scan_rows(&table).await, rows_before);
}

#[tokio::test]
async fn test_partial_progress_max_commits_0_fails_when_enabled() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

    let error = RewriteDataFiles::new(table.clone())
        .partial_progress(true)
        .partial_progress_max_commits(0)
        .execute(&catalog)
        .await
        .expect_err("max-commits 0 with partial progress must fail");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot set partial-progress.max-commits to 0, the value must be positive when partial-progress.enabled is true"
    );

    let mut files = Vec::new();
    for index in 0..5i64 {
        files.push(
            write_data_file(&table, &format!("m-{index}.parquet"), 0, &[(
                0, index, index,
            )])
            .await,
        );
    }
    let table = append_files(&catalog, &table, files).await;
    RewriteDataFiles::new(table.clone())
        .partial_progress_max_commits(0)
        .target_file_size_bytes(1_000_000)
        .execute(&catalog)
        .await
        .expect("max-commits 0 with progress OFF is accepted");
}

#[tokio::test]
async fn test_output_spec_id_0_after_drop_writes_two_files_under_spec_0() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
    for part in [0i64, 1] {
        for index in 0..2i64 {
            let file = write_data_file(&table, &format!("s{part}-{index}.parquet"), part, &[(
                part, index, index,
            )])
            .await;
            append_files(&catalog, &table, vec![file]).await;
        }
    }
    let table = catalog.load_table(table.identifier()).await.unwrap();
    let table = drop_partition_field(&catalog, &table).await;
    assert_eq!(table.metadata().default_partition_spec_id(), 1);
    let rows_before = scan_rows(&table).await;

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .output_spec_id(0)
        .execute(&catalog)
        .await
        .expect("output-spec-id 0 must succeed");

    assert_eq!(result.rewritten_data_files_count, 4);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(
        result.file_groups.len(),
        1,
        "files of spec 0 mismatch the current spec 1, so Java groups them in one empty-struct bucket"
    );
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        5,
        "4 appends plus one rewrite commit; the spec drop adds no snapshot"
    );
    assert_eq!(live_data_spec_ids(&table).await, HashSet::from([0]));
    assert_eq!(scan_rows(&table).await, rows_before);
}

#[tokio::test]
async fn test_default_spec_after_drop_writes_one_file_under_spec_1() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
    for part in [0i64, 1] {
        for index in 0..2i64 {
            let file = write_data_file(&table, &format!("d{part}-{index}.parquet"), part, &[(
                part, index, index,
            )])
            .await;
            append_files(&catalog, &table, vec![file]).await;
        }
    }
    let table = catalog.load_table(table.identifier()).await.unwrap();
    let table = drop_partition_field(&catalog, &table).await;
    let rows_before = scan_rows(&table).await;

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("default-spec rewrite must succeed");

    assert_eq!(result.rewritten_data_files_count, 4);
    assert_eq!(result.added_data_files_count, 1);
    assert_eq!(result.file_groups.len(), 1);
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        5,
        "4 appends plus one rewrite commit; the spec drop adds no snapshot"
    );
    assert_eq!(live_data_spec_ids(&table).await, HashSet::from([1]));
    assert_eq!(scan_rows(&table).await, rows_before);
}

#[tokio::test]
async fn test_output_spec_id_unknown_fails() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

    let error = RewriteDataFiles::new(table.clone())
        .output_spec_id(99)
        .execute(&catalog)
        .await
        .expect_err("unknown output spec id must fail");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot use output spec id 99 because the table does not contain a reference to this spec-id."
    );
}

#[tokio::test]
async fn test_rewrite_job_order_bytes_desc_orders_file_groups() {
    let build = || async {
        let (catalog, temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
        for index in 0..2i64 {
            let file = write_data_file(&table, &format!("small-{index}.parquet"), 0, &[(
                0, index, index,
            )])
            .await;
            append_files(&catalog, &table, vec![file]).await;
        }
        let big_rows: Vec<(i64, i64, i64)> = (0..100).map(|n| (1, n, n)).collect();
        for index in 0..2i64 {
            let file = write_data_file(&table, &format!("big-{index}.parquet"), 1, &big_rows).await;
            append_files(&catalog, &table, vec![file]).await;
        }
        let table = catalog.load_table(table.identifier()).await.unwrap();
        (catalog, temp, table)
    };

    let (catalog, _temp, table) = build().await;
    let desc = RewriteDataFiles::new(table.clone())
        .min_input_files(2)
        .rewrite_job_order(RewriteJobOrder::BytesDesc)
        .execute(&catalog)
        .await
        .expect("bytes-desc must succeed");
    assert_eq!(desc.file_groups.len(), 2);
    assert!(
        desc.file_groups[0].rewritten_bytes_count > desc.file_groups[1].rewritten_bytes_count,
        "bytes-desc lists the bigger group first: {:?}",
        desc.file_groups
    );

    let (catalog, _temp, table) = build().await;
    let asc = RewriteDataFiles::new(table.clone())
        .min_input_files(2)
        .rewrite_job_order(RewriteJobOrder::FilesAsc)
        .execute(&catalog)
        .await
        .expect("files-asc must succeed");
    assert_eq!(asc.file_groups.len(), 2);
    assert_eq!(asc.rewritten_data_files_count, 4);
    assert_eq!(
        asc.rewritten_bytes_count, desc.rewritten_bytes_count,
        "job order changes commit order, not the rewritten set"
    );
}

#[tokio::test]
async fn test_rewrite_job_order_parse() {
    assert_eq!(
        RewriteJobOrder::from_str("none").unwrap(),
        RewriteJobOrder::None
    );
    assert_eq!(
        RewriteJobOrder::from_str("bytes-asc").unwrap(),
        RewriteJobOrder::BytesAsc
    );
    assert_eq!(
        RewriteJobOrder::from_str("BYTES-DESC").unwrap(),
        RewriteJobOrder::BytesDesc
    );
    assert_eq!(
        RewriteJobOrder::from_str("Files_Asc").unwrap(),
        RewriteJobOrder::FilesAsc
    );
    assert_eq!(
        RewriteJobOrder::from_str("files-desc").unwrap(),
        RewriteJobOrder::FilesDesc
    );
    let error = RewriteJobOrder::from_str("bogus").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(error.message(), "Invalid rewrite job order name: bogus");
}

#[tokio::test]
async fn test_max_concurrent_0_fails_positive_runs_sequentially() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = two_partitions_by_four_files(&catalog).await;

    let error = RewriteDataFiles::new(table.clone())
        .max_concurrent_file_group_rewrites(0)
        .execute(&catalog)
        .await
        .expect_err("max-concurrent 0 must fail");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot set max-concurrent-file-group-rewrites to 0, the value must be positive."
    );

    let result = RewriteDataFiles::new(table.clone())
        .min_input_files(1)
        .max_concurrent_file_group_rewrites(4)
        .execute(&catalog)
        .await
        .expect("positive max-concurrent is accepted");
    assert_eq!(result.rewritten_data_files_count, 8);
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        9,
        "a positive value runs (sequentially) under the one-commit default"
    );
}

#[tokio::test]
async fn test_delete_ratio_threshold_2_renders_java_float() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

    let error = RewriteDataFiles::new(table.clone())
        .delete_ratio_threshold(2.0)
        .execute(&catalog)
        .await
        .expect_err("ratio above 1 must fail");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'delete-ratio-threshold' is set to 2.0 but must be <= 1"
    );

    let error = RewriteDataFiles::new(table.clone())
        .min_input_files(0)
        .execute(&catalog)
        .await
        .expect_err("min-input-files 0 must fail");
    assert_eq!(
        error.message(),
        "'min-input-files' is set to 0 but must be > 0"
    );
}

#[test]
fn test_commit_batching_math() {
    assert_eq!(plan_commit_batches(0, false, 10), Vec::<usize>::new());
    assert_eq!(plan_commit_batches(2, false, 10), vec![2]);
    assert_eq!(plan_commit_batches(2, true, 10), vec![1, 1]);
    assert_eq!(plan_commit_batches(2, true, 1), vec![2]);
    assert_eq!(plan_commit_batches(4, true, 3), vec![2, 2]);
    assert_eq!(plan_commit_batches(5, true, 10), vec![1, 1, 1, 1, 1]);
    assert_eq!(plan_commit_batches(7, true, 3), vec![3, 3, 1]);
}

#[test]
fn test_format_java_double() {
    assert_eq!(format_java_double(2.0), "2.0");
    assert_eq!(format_java_double(0.0), "0.0");
    assert_eq!(format_java_double(100.0), "100.0");
    assert_eq!(format_java_double(0.3), "0.3");
    assert_eq!(format_java_double(f64::NAN), "NaN");
    assert_eq!(format_java_double(f64::INFINITY), "Infinity");
    assert_eq!(format_java_double(f64::NEG_INFINITY), "-Infinity");
}

#[tokio::test]
async fn test_output_spec_id_0_partial_progress_matches_java_single_group() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = l001_dropped_table(&catalog, "out0_pp", &[0, 0, 0, 0, 1, 1, 1, 1]).await;
    assert_eq!(
        snapshot_count(&table),
        8,
        "fixture: 8 appends; the spec drop adds no snapshot"
    );
    let rows_before = scan_l001_rows(&table).await;
    assert_eq!(rows_before.len(), 400);

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .output_spec_id(0)
        .partial_progress(true)
        .execute(&catalog)
        .await
        .expect("output-spec-id 0 with partial progress must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(
        result.file_groups.len(),
        1,
        "Java groups files of spec 0 against the current spec 1 into one bucket"
    );
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        9,
        "8 appends plus one rewrite commit; the spec drop adds no snapshot"
    );
    assert_eq!(live_data_spec_ids(&table).await, HashSet::from([0]));
    assert_eq!(scan_l001_rows(&table).await, rows_before);
}

#[tokio::test]
async fn test_output_spec_id_0_size_split_fans_mixed_bins_out_by_partition() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = l001_dropped_table(&catalog, "out0_pp_small", &[0, 1, 0, 0, 1, 1, 0, 1]).await;
    let rows_before = scan_l001_rows(&table).await;
    assert_eq!(rows_before.len(), 400);

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .output_spec_id(0)
        .partial_progress(true)
        .max_file_group_size_bytes(4000)
        .execute(&catalog)
        .await
        .expect("output-spec-id 0 with split groups must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(
        result.file_groups.len(),
        4,
        "the limit must pack the files two per bin: {:?}",
        result.file_groups
    );
    assert_eq!(
        result.added_data_files_count, 6,
        "two mixed bins fan out to two files each, two pure bins to one: {:?}",
        result.file_groups
    );
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        12,
        "8 appends plus 4 rewrite commits; the spec drop adds no snapshot"
    );
    assert_eq!(live_data_spec_ids(&table).await, HashSet::from([0]));
    assert_eq!(scan_l001_rows(&table).await, rows_before);
}

#[tokio::test]
async fn test_default_spec_partial_progress_matches_java_single_file() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = l001_dropped_table(&catalog, "cur_pp", &[0, 0, 0, 0, 1, 1, 1, 1]).await;
    let rows_before = scan_l001_rows(&table).await;
    assert_eq!(rows_before.len(), 400);

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .partial_progress(true)
        .execute(&catalog)
        .await
        .expect("default-spec rewrite with partial progress must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 1);
    assert_eq!(result.file_groups.len(), 1);
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        snapshot_count(&table),
        9,
        "8 appends plus one rewrite commit; the spec drop adds no snapshot"
    );
    assert_eq!(live_data_spec_ids(&table).await, HashSet::from([1]));
    assert_eq!(scan_l001_rows(&table).await, rows_before);
}

#[tokio::test]
async fn test_target_small_output_count_follows_java_read_splits() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_l001_table(&catalog, "granularity-target").await;
    let mut sizes: Vec<(i32, u64)> = Vec::new();
    for index in 0..8i64 {
        let p = (index % 2) as i32;
        let file = write_l001_file(&table, &format!("g-{index}"), p, index * 50).await;
        sizes.push((p, file.file_size_in_bytes()));
        append_files(&catalog, &table, vec![file]).await;
    }
    let table = catalog.load_table(table.identifier()).await.unwrap();
    println!("fork-written file sizes: {sizes:?}");

    let (spec, schema) = synthetic_spec_and_schema();
    let config = config_for(2_000, 1_500, 3_600, 1);
    let mut expected_outputs = 0usize;
    for p in [0i32, 1] {
        let tasks: Vec<crate::scan::FileScanTask> = sizes
            .iter()
            .filter(|(pp, _)| *pp == p)
            .enumerate()
            .map(|(i, (_, size))| {
                synthetic_task(&format!("s{p}-{i}"), *size, i64::from(p), 0, &spec, &schema)
            })
            .collect();
        let input_size: u64 = tasks.iter().map(|task| task.length).sum();
        let split_size = input_split_size(input_size, &config);
        println!("p={p}: input={input_size} inputSplitSize={split_size}");
        expected_outputs += plan_read_tasks(tasks, split_size).unwrap().len();
    }
    println!("java-formula expected output files: {expected_outputs}");
    assert!(
        expected_outputs > 2,
        "the fixture must force more than one read task per group, got {expected_outputs}"
    );

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .target_file_size_bytes(2_000)
        .execute(&catalog)
        .await
        .expect("target_small rewrite must succeed");
    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.file_groups.len(), 2, "one group per partition");
    assert_eq!(
        result.added_data_files_count, expected_outputs,
        "the output count must follow Java's read-split planning, not one file per group"
    );
}
