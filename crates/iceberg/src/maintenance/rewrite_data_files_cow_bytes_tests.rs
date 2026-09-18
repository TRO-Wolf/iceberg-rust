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
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};

use crate::Catalog;
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, create_partitioned_table, live_delete_file_paths, local_fs_catalog,
    scan_rows, write_data_file,
};
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, MetricsConfig, PartitionKey,
    Struct,
};
use crate::table::Table;
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
    position_delete_writer_properties,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

async fn write_position_delete(
    table: &Table,
    part_value: i64,
    deletes: &[(String, i64)],
    metrics_config: MetricsConfig,
) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("pos-delete config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder =
        ParquetWriterBuilder::new(position_delete_writer_properties(), config.schema().clone())
            .with_metrics_config(metrics_config);
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(part_value))]),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key))
        .await
        .expect("build pos-delete writer");
    let paths: Vec<&str> = deletes.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("pos-delete batch");
    writer.write(batch).await.expect("write pos-delete");
    writer
        .close()
        .await
        .expect("close pos-delete")
        .into_iter()
        .next()
        .expect("one pos-delete file")
}

async fn cow_bytes_shape(catalog: &impl Catalog) -> (Table, Vec<String>, u64) {
    let table = create_partitioned_table(catalog, FormatVersion::V2).await;
    let mut files = Vec::new();
    let mut paths = Vec::new();
    let mut target = 0u64;
    for part in 0..2i64 {
        for fileno in 0..4i64 {
            let base = (part * 4 + fileno) * 50;
            let rows: Vec<(i64, i64, i64)> =
                (0..50).map(|n| (part, base + n, base + n)).collect();
            let file = write_data_file(
                &table,
                &format!("part-{part}-file-{fileno}.parquet"),
                part,
                &rows,
            )
            .await;
            target = target.max(file.file_size_in_bytes());
            paths.push(file.file_path().to_string());
            files.push(file);
        }
    }
    let table = append_files(catalog, &table, files).await;
    (table, paths, target)
}

async fn delete_partition_zero(
    catalog: &impl Catalog,
    table: Table,
    paths: &[String],
    last_id_exclusive: i64,
    metrics_config: MetricsConfig,
) -> (Table, String) {
    let deletes: Vec<(String, i64)> =
        (0..last_id_exclusive).map(|pos| (paths[0].clone(), pos)).collect();
    let pos_delete = write_position_delete(&table, 0, &deletes, metrics_config).await;
    let delete_path = pos_delete.file_path().to_string();
    let table = add_deletes(catalog, &table, vec![pos_delete]).await;
    (table, delete_path)
}

fn file_scoped_metrics() -> MetricsConfig {
    MetricsConfig::for_position_delete()
}

fn partition_scoped_metrics() -> MetricsConfig {
    MetricsConfig::from_properties(&HashMap::from([(
        "write.metadata.metrics.column.file_path".to_string(),
        "none".to_string(),
    )]))
}

async fn live_data_sequences(table: &Table) -> Vec<(String, i64, i64)> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut out = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                out.push((
                    entry.file_path().to_string(),
                    entry.sequence_number().expect("data seq"),
                    entry.file_sequence_number.expect("file seq"),
                ));
            }
        }
    }
    out
}

async fn assert_output_sequences(
    table: &Table,
    data_seq: i64,
    file_seq: i64,
    output_count: usize,
) {
    let outputs: Vec<(i64, i64)> = live_data_sequences(table)
        .await
        .into_iter()
        .filter(|(_, _, file)| *file == file_seq)
        .map(|(_, data, file)| (data, file))
        .collect();
    assert_eq!(
        outputs,
        vec![(data_seq, file_seq); output_count],
        "every rewrite output must carry data seq {data_seq} and file seq {file_seq}"
    );
}

#[tokio::test]
async fn test_cow_bytes_partition_delete_threshold_keeps_applicable_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, paths, target) = cow_bytes_shape(&catalog).await;
    let (table, delete_path) =
        delete_partition_zero(&catalog, table, &paths, 4, partition_scoped_metrics()).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 396);

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .delete_file_threshold(1)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");

    assert_eq!(result.rewritten_data_files_count, 4);
    assert_eq!(result.added_data_files_count, 1);
    assert_eq!(result.removed_delete_files_count, 0);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([delete_path]),
        "the still-applicable partition-scoped position delete must stay live"
    );
    assert_output_sequences(&table, 2, 3, 1).await;
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

#[tokio::test]
async fn test_cow_bytes_file_scoped_delete_threshold_keeps_applicable_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, paths, target) = cow_bytes_shape(&catalog).await;
    let (table, delete_path) =
        delete_partition_zero(&catalog, table, &paths, 4, file_scoped_metrics()).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 396);

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .delete_file_threshold(1)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");

    assert_eq!(result.rewritten_data_files_count, 1);
    assert_eq!(result.added_data_files_count, 1);
    assert_eq!(result.removed_delete_files_count, 0);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([delete_path]),
        "the still-applicable file-scoped position delete must stay live"
    );
    assert_output_sequences(&table, 2, 3, 1).await;
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

#[tokio::test]
async fn test_cow_bytes_rewrite_all_keeps_applicable_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, paths, target) = cow_bytes_shape(&catalog).await;
    let (table, delete_path) =
        delete_partition_zero(&catalog, table, &paths, 30, file_scoped_metrics()).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 370);

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.removed_delete_files_count, 0);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([delete_path]),
        "the still-applicable position delete must stay live"
    );
    assert_output_sequences(&table, 2, 3, 2).await;
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

#[tokio::test]
async fn test_cow_bytes_remove_dangling_keeps_applicable_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, paths, target) = cow_bytes_shape(&catalog).await;
    let (table, delete_path) =
        delete_partition_zero(&catalog, table, &paths, 30, file_scoped_metrics()).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 370);

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .rewrite_all(true)
        .remove_dangling_deletes(true)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.removed_delete_files_count, 0);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([delete_path]),
        "delete seq 2 is not below partition min data seq 2, so it is not dangling"
    );
    assert_output_sequences(&table, 2, 3, 2).await;
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

#[tokio::test]
async fn test_cow_bytes_remove_dangling_single_row_keeps_applicable_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, paths, target) = cow_bytes_shape(&catalog).await;
    let (table, delete_path) =
        delete_partition_zero(&catalog, table, &paths, 4, file_scoped_metrics()).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 396);

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .rewrite_all(true)
        .remove_dangling_deletes(true)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.removed_delete_files_count, 0);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([delete_path]),
        "the still-applicable position delete must stay live"
    );
    assert_output_sequences(&table, 2, 3, 2).await;
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

#[tokio::test]
async fn test_cow_bytes_partition_delete_survives_dangling_cleanup() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, paths, target) = cow_bytes_shape(&catalog).await;
    let (table, delete_path) =
        delete_partition_zero(&catalog, table, &paths, 30, partition_scoped_metrics()).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 370);

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .rewrite_all(true)
        .remove_dangling_deletes(true)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.removed_delete_files_count, 0);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([delete_path]),
        "the still-applicable partition-scoped position delete must stay live"
    );
    assert_output_sequences(&table, 2, 3, 2).await;
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

#[tokio::test]
async fn test_cow_bytes_new_sequence_keeps_delete_without_cleanup() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, paths, target) = cow_bytes_shape(&catalog).await;
    let (table, delete_path) =
        delete_partition_zero(&catalog, table, &paths, 30, file_scoped_metrics()).await;
    let rows_before = scan_rows(&table).await;

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .rewrite_all(true)
        .use_starting_sequence_number(false)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.removed_delete_files_count, 0);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([delete_path]),
        "the now-dangling delete stays live while the sub-action is off"
    );
    assert_output_sequences(&table, 3, 3, 2).await;
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

#[tokio::test]
async fn test_cow_bytes_new_sequence_dangling_cleanup_removes_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, paths, target) = cow_bytes_shape(&catalog).await;
    let (table, _delete_path) =
        delete_partition_zero(&catalog, table, &paths, 30, file_scoped_metrics()).await;
    let rows_before = scan_rows(&table).await;

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .rewrite_all(true)
        .use_starting_sequence_number(false)
        .remove_dangling_deletes(true)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");

    assert_eq!(result.rewritten_data_files_count, 8);
    assert_eq!(result.added_data_files_count, 2);
    assert_eq!(result.removed_delete_files_count, 1);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert!(
        live_delete_file_paths(&table).await.is_empty(),
        "delete seq 2 is below the new partition min data seq 3, so it dangles"
    );
    assert_output_sequences(&table, 3, 3, 2).await;
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}
