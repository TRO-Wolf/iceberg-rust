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

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};

use crate::Catalog;
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, live_data_file_paths, live_delete_file_paths, local_fs_catalog,
    scan_rows,
};
use crate::maintenance::rewrite_data_files::{RewriteDataFiles, RewriteDataFilesResult};
use crate::spec::{DataContentType, DataFile, DataFileFormat, FormatVersion, MetricsConfig};
use crate::table::Table;
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig, position_delete_writer_properties,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

async fn write_unpartitioned_data_file(
    table: &Table,
    file_name: &str,
    rows: &[(i64, i64, i64)],
) -> DataFile {
    use crate::arrow::schema_to_arrow_schema;
    use crate::spec::Struct;
    use crate::writer::file_writer::{FileWriter, FileWriterBuilder};

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let xs: Vec<i64> = rows.iter().map(|(x, _, _)| *x).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, y, _)| *y).collect();
    let zs: Vec<i64> = rows.iter().map(|(_, _, z)| *z).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(xs)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
        Arc::new(Int64Array::from(zs)) as ArrayRef,
    ])
    .expect("data batch");
    let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
    let output = table.file_io().new_output(file_path).expect("output");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.expect("parquet writer");
    writer.write(&batch).await.expect("write rows");
    let mut builder = writer
        .close()
        .await
        .expect("close")
        .into_iter()
        .next()
        .expect("one data file");
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .expect("data file")
}

async fn write_unpartitioned_position_delete(
    table: &Table,
    deletes: &[(String, i64)],
    full_path_bounds: bool,
) -> DataFile {
    let config = PositionDeleteWriterConfig::new().expect("pos-delete config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = if full_path_bounds {
        ParquetWriterBuilder::new(position_delete_writer_properties(), config.schema().clone())
            .with_metrics_config(MetricsConfig::for_position_delete())
    } else {
        ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            config.schema().clone(),
        )
    };
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let spec = table.metadata().default_partition_spec().as_ref().clone();
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .with_partition_spec(spec)
        .build(None)
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

async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().expect("a snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() != DataContentType::Data {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

async fn mw7_seed(catalog: &impl Catalog, full_path_bounds: bool) -> (Table, u64, i64) {
    use crate::maintenance::rewrite_data_files_evolved_spec_tests::create_unpartitioned_table;

    let table = create_unpartitioned_table(catalog, FormatVersion::V2).await;
    let target: u64 = 64 * 1024;
    let min = (target as f64 * 0.75) as u64;
    let max = (target as f64 * 1.80) as u64;
    let mut row_count: i64 = 2500;
    let (data, data_path) = loop {
        let rows: Vec<(i64, i64, i64)> = (0..row_count).map(|n| (0, n, n)).collect();
        let data =
            write_unpartitioned_data_file(&table, &format!("dead-{row_count}.parquet"), &rows)
                .await;
        let size = data.file_size_in_bytes();
        if size >= min && size <= max {
            let path = data.file_path().to_string();
            break (data, path);
        }
        assert!(
            row_count < 20_000,
            "could not land an unpartitioned data file in the 64 KiB band"
        );
        row_count += 500;
    };
    let table = append_files(catalog, &table, vec![data]).await;

    let deletes: Vec<(String, i64)> = (0..row_count).map(|pos| (data_path.clone(), pos)).collect();
    let pos_delete = write_unpartitioned_position_delete(&table, &deletes, full_path_bounds).await;
    assert!(
        pos_delete.referenced_data_file().is_none(),
        "the MW-7 shape is PARTITION-scoped: referenced_data_file must be null"
    );
    let table = add_deletes(catalog, &table, vec![pos_delete]).await;
    assert!(scan_rows(&table).await.is_empty(), "all rows are deleted");
    (table, target, row_count)
}

#[tokio::test]
async fn test_mw7_unpartitioned_single_file_partition_scoped_full_bounds_is_reclaimed() {
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;

    let (catalog, _temp) = local_fs_catalog().await;
    let (table, target, _rows) = mw7_seed(&catalog, true).await;

    let delete = live_delete_files(&table).await;
    let delete = delete.first().expect("one live delete file");
    let lower = delete
        .lower_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        .expect("Full metrics keep a file_path lower bound");
    let upper = delete
        .upper_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        .expect("Full metrics keep a file_path upper bound");
    assert_eq!(
        lower, upper,
        "one covered data file must produce EQUAL file_path bounds"
    );

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
    assert!(live_data_file_paths(&table).await.is_empty());
    assert!(scan_rows(&table).await.is_empty());
}

#[tokio::test]
async fn test_mw7_unpartitioned_single_file_without_path_bounds_is_a_noop() {
    use crate::delete_file_index::referenced_data_file_location;
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;

    let (catalog, _temp) = local_fs_catalog().await;
    let (table, target, _rows) = mw7_seed(&catalog, false).await;

    let delete = live_delete_files(&table).await;
    let delete = delete.first().expect("one live delete file");
    assert!(
        !delete
            .lower_bounds()
            .contains_key(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
        "default metrics leave no exact file_path lower bound"
    );
    assert!(
        referenced_data_file_location(delete).is_none(),
        "without equal bounds the delete is NOT file-scoped, exactly as Java judges it"
    );

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(target)
        .execute(&catalog)
        .await
        .expect("execute must succeed (no-op)");
    assert_eq!(
        result,
        RewriteDataFilesResult::default(),
        "Java counts only FILE-SCOPED deletes, so this shape is a no-op in both engines"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(live_delete_file_paths(&table).await.len(), 1);
}
