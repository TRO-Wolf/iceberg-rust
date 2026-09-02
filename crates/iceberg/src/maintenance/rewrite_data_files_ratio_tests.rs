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

//! Execute-path pins for `delete_ratio_threshold` and V3 DV removal on `RewriteDataFiles`.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};

use crate::Catalog;
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, create_partitioned_table, live_data_file_paths,
    live_delete_file_paths, local_fs_catalog, scan_rows, write_data_file,
    write_position_delete_file,
};
use crate::maintenance::rewrite_data_files::{RewriteDataFiles, RewriteDataFilesResult};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType,
    MetricsConfig, PartitionKey, Struct,
};
use crate::table::Table;
use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig, position_delete_writer_properties,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

fn ten_rows() -> Vec<(i64, i64, i64)> {
    (0..10).map(|n| (0, n, n * 10)).collect()
}

async fn write_deletion_vectors(table: &Table, deletes: &[(&str, &[u64])]) -> Vec<DataFile> {
    let dv_path = format!(
        "{}/data/dv-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(&dv_path).expect("dv output");
    let mut writer = DVFileWriter::new(output).unpartitioned();
    for (target_path, positions) in deletes {
        let partition_key = PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::long(0))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        for &pos in *positions {
            writer
                .delete(target_path, pos, Some(&partition_key))
                .expect("record DV position");
        }
    }
    writer.close().await.expect("close DV writer")
}

fn well_sized_action(table: Table, data_size: u64) -> RewriteDataFiles {
    RewriteDataFiles::new(table)
        .target_file_size_bytes(data_size)
        .min_file_size_bytes(data_size / 2)
        .max_file_size_bytes(data_size.saturating_mul(2).max(data_size + 1))
}

async fn write_file_scoped_position_delete(
    table: &Table,
    part_value: i64,
    deletes: &[(String, i64)],
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
            .with_metrics_config(MetricsConfig::for_position_delete());
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

/// Default 0.3: a well-sized V3 file with a 50% DV is rewritten and the DV is dropped.
#[tokio::test]
async fn test_default_ratio_rewrites_well_sized_file_and_drops_its_dv() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    let data = write_data_file(&table, "laden.parquet", 0, &ten_rows()).await;
    let data_path = data.file_path().to_string();
    let data_size = data.file_size_in_bytes();
    let table = append_files(&catalog, &table, vec![data]).await;

    let dvs = write_deletion_vectors(&table, &[(&data_path, &[0, 1, 2, 3, 4])]).await;
    assert_eq!(dvs.len(), 1);
    let table = add_deletes(&catalog, &table, dvs).await;

    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 5);

    let result = well_sized_action(table.clone(), data_size)
        .execute(&catalog)
        .await
        .expect("compaction must succeed");

    assert_eq!(result.rewritten_data_files_count, 1);
    assert_eq!(
        result.removed_delete_files_count, 1,
        "the DV is dropped in the rewrite commit with remove_dangling_deletes off"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert!(
        !live_data_file_paths(&table).await.contains(&data_path),
        "the delete-laden file was rewritten"
    );
    assert!(
        live_delete_file_paths(&table).await.is_empty(),
        "the DV that referenced the rewritten file is gone"
    );
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

/// Default 0.3: a well-sized file with a 20% DV is left alone.
#[tokio::test]
async fn test_default_ratio_under_threshold_is_a_noop() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    let data = write_data_file(&table, "ok.parquet", 0, &ten_rows()).await;
    let data_path = data.file_path().to_string();
    let data_size = data.file_size_in_bytes();
    let table = append_files(&catalog, &table, vec![data]).await;

    let dvs = write_deletion_vectors(&table, &[(&data_path, &[0, 1])]).await;
    let table = add_deletes(&catalog, &table, dvs).await;
    let deletes_before = live_delete_file_paths(&table).await;

    let result = well_sized_action(table.clone(), data_size)
        .execute(&catalog)
        .await
        .expect("execute must succeed (no-op)");

    assert_eq!(result, RewriteDataFilesResult::default());
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert!(live_data_file_paths(&table).await.contains(&data_path));
    assert_eq!(live_delete_file_paths(&table).await, deletes_before);
}

/// A custom 0.8 threshold does not rewrite a 50% DV file.
#[tokio::test]
async fn test_custom_ratio_threshold_0_8_leaves_half_deleted_file() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    let data = write_data_file(&table, "half.parquet", 0, &ten_rows()).await;
    let data_path = data.file_path().to_string();
    let data_size = data.file_size_in_bytes();
    let table = append_files(&catalog, &table, vec![data]).await;
    let dvs = write_deletion_vectors(&table, &[(&data_path, &[0, 1, 2, 3, 4])]).await;
    let table = add_deletes(&catalog, &table, dvs).await;

    let result = well_sized_action(table.clone(), data_size)
        .delete_ratio_threshold(0.8)
        .execute(&catalog)
        .await
        .expect("execute must succeed (no-op)");
    assert_eq!(result, RewriteDataFilesResult::default());
}

#[tokio::test]
async fn test_bounds_only_file_scoped_parquet_fires_ratio() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let data = write_data_file(&table, "v2.parquet", 0, &ten_rows()).await;
    let data_path = data.file_path().to_string();
    let data_size = data.file_size_in_bytes();
    let table = append_files(&catalog, &table, vec![data]).await;

    let deletes: Vec<(String, i64)> = (0..9).map(|pos| (data_path.clone(), pos)).collect();
    let pos_delete = write_file_scoped_position_delete(&table, 0, &deletes).await;
    let table = add_deletes(&catalog, &table, vec![pos_delete]).await;

    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 1);

    let result = well_sized_action(table.clone(), data_size)
        .execute(&catalog)
        .await
        .expect("compaction must succeed");

    assert_eq!(result.rewritten_data_files_count, 1);
    assert_eq!(result.removed_delete_files_count, 1);
    assert_eq!(result.added_data_files_count, 1);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert!(
        !live_data_file_paths(&table).await.contains(&data_path),
        "the 90% deleted in-band file was rewritten"
    );
    assert!(
        live_delete_file_paths(&table).await.is_empty(),
        "the file-scoped parquet position delete left with the rewritten file"
    );
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

#[tokio::test]
async fn test_fully_deleted_in_band_parquet_file_is_rewritten_and_drops_its_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let data = write_data_file(&table, "dead.parquet", 0, &ten_rows()).await;
    let data_path = data.file_path().to_string();
    let data_size = data.file_size_in_bytes();
    let table = append_files(&catalog, &table, vec![data]).await;

    let deletes: Vec<(String, i64)> = (0..10).map(|pos| (data_path.clone(), pos)).collect();
    let pos_delete = write_file_scoped_position_delete(&table, 0, &deletes).await;
    let table = add_deletes(&catalog, &table, vec![pos_delete]).await;
    assert!(scan_rows(&table).await.is_empty());

    let result = well_sized_action(table.clone(), data_size)
        .execute(&catalog)
        .await
        .expect("compaction must succeed");

    assert_eq!(result.rewritten_data_files_count, 1);
    assert_eq!(result.added_data_files_count, 0);
    assert_eq!(result.removed_delete_files_count, 1);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert!(
        !live_data_file_paths(&table).await.contains(&data_path),
        "the 100% deleted in-band file was rewritten away"
    );
    assert!(
        live_delete_file_paths(&table).await.is_empty(),
        "its parquet position delete left with it"
    );
    assert!(scan_rows(&table).await.is_empty());
}

#[tokio::test]
async fn test_default_ratio_under_threshold_parquet_is_a_noop() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let data = write_data_file(&table, "ok.parquet", 0, &ten_rows()).await;
    let data_path = data.file_path().to_string();
    let data_size = data.file_size_in_bytes();
    let table = append_files(&catalog, &table, vec![data]).await;

    let deletes: Vec<(String, i64)> = (0..2).map(|pos| (data_path.clone(), pos)).collect();
    let pos_delete = write_file_scoped_position_delete(&table, 0, &deletes).await;
    let table = add_deletes(&catalog, &table, vec![pos_delete]).await;
    let deletes_before = live_delete_file_paths(&table).await;

    let result = well_sized_action(table.clone(), data_size)
        .execute(&catalog)
        .await
        .expect("execute must succeed (no-op)");

    assert_eq!(result, RewriteDataFilesResult::default());
    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert!(live_data_file_paths(&table).await.contains(&data_path));
    assert_eq!(live_delete_file_paths(&table).await, deletes_before);
}

#[tokio::test]
async fn test_fully_deleted_2500_row_in_band_parquet_file_ends_at_zero_delete_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let target: u64 = 64 * 1024;
    let min = (target as f64 * 0.75) as u64;
    let max = (target as f64 * 1.80) as u64;
    let mut row_count: i64 = 2500;
    let (data, data_path) = loop {
        let rows: Vec<(i64, i64, i64)> = (0..row_count).map(|n| (0, n, n)).collect();
        let data = write_data_file(&table, &format!("dead-{row_count}.parquet"), 0, &rows).await;
        let data_size = data.file_size_in_bytes();
        if data_size >= min && data_size <= max {
            let data_path = data.file_path().to_string();
            break (data, data_path);
        }
        assert!(
            row_count < 20_000,
            "could not land a data file in the 64 KiB band (last size={data_size}, min={min}, max={max})"
        );
        row_count += 500;
    };
    let table = append_files(&catalog, &table, vec![data]).await;

    let deletes: Vec<(String, i64)> = (0..row_count).map(|pos| (data_path.clone(), pos)).collect();
    let pos_delete = write_file_scoped_position_delete(&table, 0, &deletes).await;
    let table = add_deletes(&catalog, &table, vec![pos_delete]).await;
    assert!(scan_rows(&table).await.is_empty());
    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .execute(&catalog)
        .await
        .expect("compaction must succeed");
    assert_eq!(result.rewritten_data_files_count, 1);
    assert_eq!(result.added_data_files_count, 0);
    assert_eq!(result.removed_delete_files_count, 1);

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert!(live_delete_file_paths(&table).await.is_empty());
    assert!(scan_rows(&table).await.is_empty());
}

#[tokio::test]
async fn test_planner_selects_bounds_only_parquet_because_referenced_data_file_location_is_set() {
    use futures::TryStreamExt;

    use crate::delete_file_index::referenced_data_file_location;
    use crate::maintenance::rewrite_data_files_dv::file_scoped_delete_paths;
    use crate::maintenance::rewrite_data_files_plan::{
        group_qualifies, is_candidate, too_high_delete_ratio,
    };

    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let data = write_data_file(&table, "probe.parquet", 0, &ten_rows()).await;
    let data_path = data.file_path().to_string();
    let data_size = data.file_size_in_bytes();
    let table = append_files(&catalog, &table, vec![data]).await;
    let deletes: Vec<(String, i64)> = (0..10).map(|pos| (data_path.clone(), pos)).collect();
    let pos_delete = write_file_scoped_position_delete(&table, 0, &deletes).await;
    assert!(
        pos_delete.referenced_data_file().is_none(),
        "v2 parquet position deletes do not set referenced_data_file"
    );
    assert_eq!(
        referenced_data_file_location(&pos_delete).as_deref(),
        Some(data_path.as_str()),
        "equal file_path bounds make Java isFileScoped true"
    );
    let table = add_deletes(&catalog, &table, vec![pos_delete]).await;

    let tasks: Vec<_> = table
        .scan()
        .build()
        .expect("scan")
        .plan_files()
        .await
        .expect("plan_files")
        .try_collect()
        .await
        .expect("collect tasks");
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(task.deletes.len(), 1);
    assert!(
        task.deletes[0].referenced_data_file.is_none(),
        "scan-task projection keeps the raw null referenced_data_file field"
    );
    assert_eq!(task.deletes[0].record_count, Some(10));
    assert_eq!(task.record_count, Some(10));

    let mut config = crate::maintenance::rewrite_data_files::tests::config_for(
        data_size,
        data_size / 2,
        data_size.saturating_mul(2).max(data_size + 1),
        5,
    );
    assert!(
        !too_high_delete_ratio(task, &config),
        "without the bounds-derived set the ratio misses the parquet delete"
    );
    assert!(!is_candidate(task, &config));

    config.file_scoped_delete_paths = file_scoped_delete_paths(&table)
        .await
        .expect("collect file-scoped delete paths");
    assert!(
        config
            .file_scoped_delete_paths
            .contains(&task.deletes[0].file_path)
    );
    assert!(too_high_delete_ratio(task, &config));
    assert!(is_candidate(task, &config));
    assert!(group_qualifies(std::slice::from_ref(task), &config));
}

#[tokio::test]
async fn test_absent_path_bounds_two_path_parquet_pos_delete_does_not_fire_ratio() {
    use crate::delete_file_index::referenced_data_file_location;
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;

    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let file_a = write_data_file(&table, "a.parquet", 0, &ten_rows()).await;
    let file_b = write_data_file(&table, "b.parquet", 0, &ten_rows()).await;
    let path_a = file_a.file_path().to_string();
    let path_b = file_b.file_path().to_string();
    let data_size = file_a.file_size_in_bytes().max(file_b.file_size_in_bytes());
    let table = append_files(&catalog, &table, vec![file_a, file_b]).await;

    let mut deletes: Vec<(String, i64)> = (0..9).map(|pos| (path_a.clone(), pos)).collect();
    deletes.push((path_b, 0));
    let pos_delete = write_position_delete_file(&table, 0, &deletes).await;
    assert!(
        pos_delete.referenced_data_file().is_none(),
        "Spark PARTITION shape: referenced_data_file is null"
    );
    assert!(
        !pos_delete
            .lower_bounds()
            .contains_key(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
        "the default-metrics helper writes absent file_path bounds, not unequal ones"
    );
    assert!(
        !pos_delete
            .upper_bounds()
            .contains_key(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
    );
    assert!(referenced_data_file_location(&pos_delete).is_none());
    let table = add_deletes(&catalog, &table, vec![pos_delete]).await;

    let result = well_sized_action(table.clone(), data_size)
        .execute(&catalog)
        .await
        .expect("execute must succeed (no-op)");
    assert_eq!(
        result,
        RewriteDataFilesResult::default(),
        "absent file_path bounds are not file-scoped, so the ratio must not fire"
    );
}

#[tokio::test]
async fn test_unequal_path_bounds_two_path_parquet_pos_delete_does_not_fire_ratio() {
    use crate::delete_file_index::referenced_data_file_location;
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;

    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let file_a = write_data_file(&table, "a.parquet", 0, &ten_rows()).await;
    let file_b = write_data_file(&table, "b.parquet", 0, &ten_rows()).await;
    let path_a = file_a.file_path().to_string();
    let path_b = file_b.file_path().to_string();
    let data_size = file_a.file_size_in_bytes().max(file_b.file_size_in_bytes());
    let table = append_files(&catalog, &table, vec![file_a, file_b]).await;

    let mut deletes: Vec<(String, i64)> = (0..9).map(|pos| (path_a.clone(), pos)).collect();
    deletes.push((path_b, 0));
    let pos_delete = write_file_scoped_position_delete(&table, 0, &deletes).await;
    assert!(pos_delete.referenced_data_file().is_none());
    let lower = pos_delete
        .lower_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        .expect("Full metrics write a file_path lower bound");
    let upper = pos_delete
        .upper_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        .expect("Full metrics write a file_path upper bound");
    assert_ne!(
        lower, upper,
        "two data-file paths must produce unequal file_path bounds"
    );
    assert!(
        referenced_data_file_location(&pos_delete).is_none(),
        "unequal bounds are not file-scoped"
    );
    let table = add_deletes(&catalog, &table, vec![pos_delete]).await;

    let result = well_sized_action(table.clone(), data_size)
        .execute(&catalog)
        .await
        .expect("execute must succeed (no-op)");
    assert_eq!(
        result,
        RewriteDataFilesResult::default(),
        "unequal file_path bounds are not file-scoped, so the ratio must not fire"
    );
}

#[tokio::test]
async fn test_partition_scoped_delete_survives_partial_rewrite() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let file_a = write_data_file(&table, "a.parquet", 0, &ten_rows()).await;
    let file_b = write_data_file(&table, "b.parquet", 0, &ten_rows()).await;
    let path_a = file_a.file_path().to_string();
    let path_b = file_b.file_path().to_string();
    let data_size = file_a.file_size_in_bytes().max(file_b.file_size_in_bytes());
    let table = append_files(&catalog, &table, vec![file_a, file_b]).await;

    let file_scoped_a: Vec<(String, i64)> = (0..9).map(|pos| (path_a.clone(), pos)).collect();
    let scoped_a = write_file_scoped_position_delete(&table, 0, &file_scoped_a).await;
    let scoped_a_path = scoped_a.file_path().to_string();
    let shared =
        write_position_delete_file(&table, 0, &[(path_a.clone(), 0), (path_b.clone(), 0)]).await;
    let shared_path = shared.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![scoped_a, shared]).await;

    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 10, "A has 1 live row, B has 9");
    assert!(
        !rows_before.iter().any(|&(_, y, _)| y == 0),
        "B's y=0 row is deleted by the shared partition-scoped file"
    );

    let result = well_sized_action(table.clone(), data_size)
        .execute(&catalog)
        .await
        .expect("compaction must succeed");

    assert_eq!(
        result.rewritten_data_files_count, 1,
        "only A is a ratio candidate"
    );
    assert_eq!(
        result.removed_delete_files_count, 1,
        "only A's file-scoped parquet delete is dropped"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    let live_data = live_data_file_paths(&table).await;
    assert!(!live_data.contains(&path_a), "A was rewritten");
    assert!(live_data.contains(&path_b), "B was left in place");
    let live_deletes = live_delete_file_paths(&table).await;
    assert!(
        !live_deletes.contains(&scoped_a_path),
        "A's file-scoped delete left with A"
    );
    assert!(
        live_deletes.contains(&shared_path),
        "the shared two-path delete must survive"
    );
    assert_eq!(live_deletes.len(), 1);
    assert_eq!(
        scan_rows(&table).await,
        rows_before,
        "B's deleted row stays deleted through the surviving shared delete"
    );
}

/// Rewriting one file's data rewrites the sibling blob of its Puffin and keeps it applying.
#[tokio::test]
async fn test_rewriting_one_file_keeps_sibling_dv_in_same_puffin() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    let file_a = write_data_file(&table, "a.parquet", 0, &ten_rows()).await;
    let file_b = write_data_file(&table, "b.parquet", 0, &ten_rows()).await;
    let path_a = file_a.file_path().to_string();
    let path_b = file_b.file_path().to_string();
    let size_a = file_a.file_size_in_bytes();
    let size_b = file_b.file_size_in_bytes();
    let table = append_files(&catalog, &table, vec![file_a, file_b]).await;

    let dvs = write_deletion_vectors(&table, &[(&path_a, &[0, 1, 2, 3, 4]), (&path_b, &[0])]).await;
    assert_eq!(dvs.len(), 2, "one Puffin, two DV entries");
    let puffin_path = dvs[0].file_path().to_string();
    assert_eq!(dvs[1].file_path(), puffin_path);
    let table = add_deletes(&catalog, &table, dvs).await;
    let sibling_seq_before = live_dv_sequence(&table, &path_b).await;

    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 14, "5 live on A, 9 live on B");

    let data_size = size_a.max(size_b);
    let result = well_sized_action(table.clone(), data_size)
        .execute(&catalog)
        .await
        .expect("compaction must succeed");

    assert_eq!(
        result.rewritten_data_files_count, 1,
        "only A is a ratio candidate"
    );
    assert_eq!(
        result.removed_delete_files_count, 1,
        "only A's DV is dropped"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert!(
        !live_data_file_paths(&table).await.contains(&path_a),
        "A was rewritten"
    );
    assert!(
        live_data_file_paths(&table).await.contains(&path_b),
        "B was left in place"
    );
    let remaining = live_delete_file_paths(&table).await;
    assert_eq!(remaining.len(), 1, "B's DV survives");
    assert!(
        !remaining.contains(&puffin_path),
        "the old Puffin must not remain: plan.removed carries the sibling entry too"
    );
    assert_eq!(
        scan_rows(&table).await,
        rows_before,
        "B's sibling DV still applies; A physically applied its deletes"
    );
    assert_eq!(
        live_dv_sequence(&table, &path_b).await,
        sibling_seq_before,
        "the rewritten sibling keeps its original data sequence number"
    );
}

async fn live_dv_sequence(table: &Table, referenced_data_file: &str) -> i64 {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load delete manifest");
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let data_file = entry.data_file();
            if data_file.content_type() != DataContentType::PositionDeletes
                || data_file.file_format() != DataFileFormat::Puffin
            {
                continue;
            }
            if data_file.referenced_data_file().as_deref() == Some(referenced_data_file) {
                return entry.sequence_number().expect("DV sequence is inherited");
            }
        }
    }
    panic!("no live DV for {referenced_data_file}");
}

#[tokio::test]
async fn test_delete_ratio_threshold_bounds_rejected() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let error = RewriteDataFiles::new(table.clone())
        .delete_ratio_threshold(0.0)
        .execute(&catalog)
        .await
        .expect_err("delete_ratio_threshold 0 must be rejected");
    assert!(
        error
            .message()
            .contains("'delete-ratio-threshold' is set to 0")
    );
    let error = RewriteDataFiles::new(table)
        .delete_ratio_threshold(1.1)
        .execute(&catalog)
        .await
        .expect_err("delete_ratio_threshold > 1 must be rejected");
    assert!(error.message().contains("must be <= 1"));
}

fn with_file_scoped_ratio(
    mut task: crate::scan::FileScanTask,
    data_records: u64,
    deleted_records: u64,
) -> crate::scan::FileScanTask {
    use std::sync::Arc;

    use crate::scan::FileScanTaskDeleteFile;
    use crate::spec::{DataContentType, DataFileFormat};

    task.record_count = Some(data_records);
    task.deletes = Arc::from(vec![FileScanTaskDeleteFile {
        file_path: format!("{}.dv", task.data_file_path()),
        file_size_in_bytes: 1,
        file_type: DataContentType::PositionDeletes,
        partition_spec_id: 0,
        equality_ids: None,
        file_format: DataFileFormat::Puffin,
        referenced_data_file: Some(task.data_file_path().to_string()),
        content_offset: Some(4),
        content_size_in_bytes: Some(40),
        record_count: Some(deleted_records),
    }]);
    task
}

/// Java `tooHighDeleteRatio`: file-scoped deletes / data records, `>=` the threshold.
#[test]
fn test_too_high_delete_ratio_predicate() {
    use std::sync::Arc;

    use crate::maintenance::rewrite_data_files::tests::{
        config_for, synthetic_spec_and_schema, synthetic_task,
    };
    use crate::maintenance::rewrite_data_files_plan::{
        group_qualifies, is_candidate, too_high_delete_ratio,
    };
    use crate::scan::FileScanTaskDeleteFile;
    use crate::spec::{DataContentType, DataFileFormat};

    let (spec, schema) = synthetic_spec_and_schema();
    let config = config_for(100, 75, 180, 5);

    let half = with_file_scoped_ratio(synthetic_task("h", 100, 0, 0, &spec, &schema), 10, 5);
    assert!(too_high_delete_ratio(&half, &config));
    assert!(is_candidate(&half, &config));
    assert!(group_qualifies(std::slice::from_ref(&half), &config));

    let fifth = with_file_scoped_ratio(synthetic_task("f", 100, 0, 0, &spec, &schema), 10, 2);
    assert!(!too_high_delete_ratio(&fifth, &config));
    assert!(!is_candidate(&fifth, &config));

    let exact = with_file_scoped_ratio(synthetic_task("e", 100, 0, 0, &spec, &schema), 10, 3);
    assert!(too_high_delete_ratio(&exact, &config));

    let empty = synthetic_task("z", 100, 0, 0, &spec, &schema);
    assert!(!too_high_delete_ratio(&empty, &config));

    let mut equality =
        with_file_scoped_ratio(synthetic_task("q", 100, 0, 0, &spec, &schema), 10, 9);
    equality.deletes = Arc::from(vec![FileScanTaskDeleteFile {
        file_path: "eq".to_string(),
        file_size_in_bytes: 1,
        file_type: DataContentType::EqualityDeletes,
        partition_spec_id: 0,
        equality_ids: Some(vec![2]),
        file_format: DataFileFormat::Parquet,
        referenced_data_file: None,
        content_offset: None,
        content_size_in_bytes: None,
        record_count: Some(9),
    }]);
    assert!(!too_high_delete_ratio(&equality, &config));

    let mut partition_scoped = synthetic_task("p", 100, 0, 1, &spec, &schema);
    partition_scoped.record_count = Some(10);
    let mut deletes = partition_scoped.deletes.to_vec();
    deletes[0].record_count = Some(9);
    partition_scoped.deletes = Arc::from(deletes);
    assert!(!too_high_delete_ratio(&partition_scoped, &config));

    let mut bounds_only = synthetic_task("b", 100, 0, 1, &spec, &schema);
    bounds_only.record_count = Some(10);
    let mut deletes = bounds_only.deletes.to_vec();
    deletes[0].record_count = Some(9);
    deletes[0].referenced_data_file = None;
    bounds_only.deletes = Arc::from(deletes);
    assert!(!too_high_delete_ratio(&bounds_only, &config));
    let mut config_with_bounds = config_for(100, 75, 180, 5);
    config_with_bounds
        .file_scoped_delete_paths
        .insert(bounds_only.deletes[0].file_path.clone());
    assert!(too_high_delete_ratio(&bounds_only, &config_with_bounds));
    assert!(is_candidate(&bounds_only, &config_with_bounds));
    assert!(group_qualifies(
        std::slice::from_ref(&bounds_only),
        &config_with_bounds
    ));
}
