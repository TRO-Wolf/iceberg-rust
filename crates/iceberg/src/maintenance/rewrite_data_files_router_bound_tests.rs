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

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;

use crate::Catalog;
use crate::error::ErrorKind;
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, create_partitioned_table, live_data_file_paths,
    live_delete_file_paths, local_fs_catalog, scan_rows, write_data_file,
    write_equality_delete_file,
};
use crate::maintenance::rewrite_data_files_evolved_spec_tests::{
    compact, compact_action, create_unpartitioned_table, evolve_spec, live_data_files,
    scan_pruned_rows, write_current_spec_file,
};
use crate::maintenance::rewrite_data_files_write::write_compacted_files;
use crate::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use crate::scan::FileScanTask;
use crate::spec::{
    DataFile, DataFileFormat, FormatVersion, Literal, MetricsConfig, PartitionKey, Struct,
};
use crate::table::Table;
use crate::transaction::Transaction;
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

fn parquet_paths(root: &Path) -> BTreeSet<PathBuf> {
    let mut out = BTreeSet::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "parquet") {
                out.insert(path);
            }
        }
    }
    out
}

async fn plan_tasks(table: &Table) -> Vec<FileScanTask> {
    let stream = table
        .scan()
        .build()
        .expect("scan")
        .plan_files()
        .await
        .expect("plan_files");
    stream.try_collect().await.expect("collect tasks")
}

async fn scan_lineage(table: &Table) -> Vec<(i64, i64, i64)> {
    let stream = table
        .scan()
        .select([
            "y",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .expect("scan")
        .to_arrow()
        .await
        .expect("to_arrow");
    let batches: Vec<_> = stream.try_collect().await.expect("collect");
    let mut rows = Vec::new();
    for batch in batches {
        let ys = batch
            .column_by_name("y")
            .expect("y")
            .as_primitive::<arrow_array::types::Int64Type>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id")
            .as_primitive::<arrow_array::types::Int64Type>();
        let seqs = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("_last_updated_sequence_number")
            .as_primitive::<arrow_array::types::Int64Type>();
        for index in 0..batch.num_rows() {
            assert!(row_ids.is_valid(index), "row must have _row_id");
            assert!(seqs.is_valid(index), "row must have last_updated_seq");
            rows.push((ys.value(index), row_ids.value(index), seqs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
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
    .expect("partition key");
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

async fn write_dv(table: &Table, part_value: i64, deletes: &[(&str, &[u64])]) -> Vec<DataFile> {
    let dv_path = format!(
        "{}/data/dv-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(&dv_path).expect("dv output");
    let mut writer = DVFileWriter::new(output).unpartitioned();
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(part_value))]),
    )
    .expect("partition key");
    for (target_path, positions) in deletes {
        for &pos in *positions {
            writer
                .delete(target_path, pos, Some(&partition_key))
                .expect("record DV");
        }
    }
    writer.close().await.expect("close DV")
}

#[tokio::test]
async fn default_max_open_partition_writers_is_64_and_peak_obeys_it() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100)]).await;
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field("y"),
    )
    .await;

    let action = compact_action(table.clone());
    assert_eq!(
        action
            .resolved_max_open_partition_writers()
            .expect("resolve"),
        64
    );

    let tasks = plan_tasks(&table).await;
    let compacted = write_compacted_files(&table, &tasks, 1_000_000, 64)
        .await
        .expect("write");
    assert!(
        compacted.peak_open_partition_writers <= 64,
        "peak {} must obey default 64",
        compacted.peak_open_partition_writers
    );

    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);
    assert_eq!(scan_rows(&table).await, vec![(1, 10, 100), (2, 20, 200)]);
}

#[tokio::test]
async fn zero_max_open_partition_writers_is_data_invalid_before_write() {
    let (catalog, temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100)]).await;
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field("y"),
    )
    .await;

    let snapshot_before = table.metadata().current_snapshot_id();
    let files_before = live_data_file_paths(&table).await;
    let parquet_before = parquet_paths(temp.path());
    let err = compact_action(table.clone())
        .max_open_partition_writers(0)
        .execute(&catalog)
        .await
        .expect_err("zero bound must fail");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.message().contains("max-open-partition-writers"),
        "unexpected message: {}",
        err.message()
    );

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    assert_eq!(table.metadata().current_snapshot_id(), snapshot_before);
    assert_eq!(live_data_file_paths(&table).await, files_before);
    let parquet_after = parquet_paths(temp.path());
    assert_eq!(
        parquet_after, parquet_before,
        "zero bound must not create an output parquet file"
    );
    assert!(
        parquet_after
            .iter()
            .filter(|path| path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("compacted-")))
            .count()
            == parquet_before
                .iter()
                .filter(|path| path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("compacted-")))
                .count(),
        "zero bound must not write compacted-*.parquet"
    );
}

#[tokio::test]
async fn high_cardinality_eviction_keeps_rows_and_obeys_bound() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;
    let max_open = 2usize;
    let key_count = 10 * max_open;
    let first: Vec<(i64, i64, i64)> = (0..key_count / 2)
        .map(|x| (x as i64, x as i64 * 10, 1))
        .collect();
    let second: Vec<(i64, i64, i64)> = (key_count / 2..key_count)
        .map(|x| (x as i64, x as i64 * 10, 2))
        .collect();
    let a = write_current_spec_file(&table, "a", &first).await;
    let b = write_current_spec_file(&table, "b", &second).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .add_field("x"),
    )
    .await;
    let before = scan_rows(&table).await;
    assert_eq!(before.len(), key_count);

    let tasks = plan_tasks(&table).await;
    let compacted = write_compacted_files(&table, &tasks, 1_000_000, max_open)
        .await
        .expect("bounded write");
    assert!(
        compacted.peak_open_partition_writers <= max_open,
        "peak {} exceeds bound {max_open}",
        compacted.peak_open_partition_writers
    );
    assert_eq!(
        compacted.peak_open_partition_writers, max_open,
        "high cardinality must fill the bound so eviction is load-bearing"
    );

    let ident = table.identifier().clone();
    let result = compact_action(table)
        .max_open_partition_writers(max_open)
        .execute(&catalog)
        .await
        .expect("compact");
    assert!(result.rewritten_data_files_count >= 2);
    let table = catalog.load_table(&ident).await.expect("reload");
    assert_eq!(scan_rows(&table).await, before);
    let files = live_data_files(&table).await;
    let spec_id = table.metadata().default_partition_spec().spec_id();
    let tuples: std::collections::HashSet<_> = files
        .iter()
        .map(|file| {
            assert_eq!(file.partition_spec_id(), spec_id);
            file.partition().clone()
        })
        .collect();
    assert_eq!(tuples.len(), key_count);
    for x in 0..key_count as i64 {
        assert_eq!(scan_pruned_rows(&table, "x", x).await, vec![(
            x,
            x * 10,
            if x < key_count as i64 / 2 { 1 } else { 2 }
        )]);
    }
}

#[tokio::test]
async fn v3_evolved_spec_rewrite_keeps_row_id_and_last_updated_seq() {
    let (catalog, _temp) = local_fs_catalog().await;
    let mut table = create_partitioned_table(&catalog, FormatVersion::V3).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100)]).await;
    table = append_files(&catalog, &table, vec![a]).await;
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    table = append_files(&catalog, &table, vec![b]).await;
    let before = scan_lineage(&table).await;
    assert_eq!(before.len(), 2);

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field("y"),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);
    let after = scan_lineage(&table).await;
    assert_eq!(
        after, before,
        "evolved-spec rewrite must keep _row_id and last_updated_seq"
    );
}

#[tokio::test]
async fn evolved_spec_rewrite_applies_equality_deletes() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100), (1, 11, 110)]).await;
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let eq = write_equality_delete_file(&table, 1, &[11]).await;
    let table = add_deletes(&catalog, &table, vec![eq]).await;
    assert_eq!(scan_rows(&table).await, vec![(1, 10, 100), (2, 20, 200)]);

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field("y"),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);
    assert_eq!(
        scan_rows(&table).await,
        vec![(1, 10, 100), (2, 20, 200)],
        "equality-deleted row must stay gone"
    );
}

#[tokio::test]
async fn evolved_spec_rewrite_drops_file_scoped_position_deletes() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100), (1, 11, 110)]).await;
    let a_path = a.file_path().to_string();
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;
    let pos = write_file_scoped_position_delete(&table, 1, &[(a_path, 1)]).await;
    let pos_path = pos.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![pos]).await;
    assert_eq!(scan_rows(&table).await, vec![(1, 10, 100), (2, 20, 200)]);

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field("y"),
    )
    .await;
    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);
    assert_eq!(scan_rows(&table).await, vec![(1, 10, 100), (2, 20, 200)]);
    let deletes = live_delete_file_paths(&table).await;
    assert!(
        !deletes.contains(&pos_path),
        "file-scoped position delete of a rewritten file must be removed"
    );
}

#[tokio::test]
async fn evolved_spec_rewrite_drops_file_scoped_dv_and_keeps_sibling() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;
    let a = write_data_file(&table, "a.parquet", 1, &[(1, 10, 100), (1, 11, 110)]).await;
    let a_path = a.file_path().to_string();
    let b = write_data_file(&table, "b.parquet", 2, &[(2, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let table = evolve_spec(
        &catalog,
        &table,
        Transaction::new(&table)
            .update_partition_spec()
            .remove_field("x")
            .add_field("y"),
    )
    .await;
    let kept_rows: Vec<(i64, i64, i64)> = (0..10).map(|n| (9, 99, 300 + n)).collect();
    let kept = write_current_spec_file(&table, "kept", &kept_rows).await;
    let kept_path = kept.file_path().to_string();
    let table = append_files(&catalog, &table, vec![kept]).await;

    let dvs = write_dv(&table, 99, &[
        (a_path.as_str(), &[1u64][..]),
        (kept_path.as_str(), &[0u64][..]),
    ])
    .await;
    assert_eq!(dvs.len(), 2, "one DeleteFile per referenced data file");
    assert_eq!(
        dvs[0].file_path(),
        dvs[1].file_path(),
        "both DV entries must share one puffin"
    );
    let table = add_deletes(&catalog, &table, dvs).await;
    let before = scan_rows(&table).await;
    assert_eq!(before.len(), 2 + 9);
    assert!(before.contains(&(1, 10, 100)));
    assert!(before.contains(&(2, 20, 200)));
    assert!(!before.contains(&(9, 99, 300)));

    let deletes_before = live_delete_file_paths(&table).await.len();
    assert_eq!(deletes_before, 1);

    let (table, result) = compact(&catalog, table).await;
    assert!(result.rewritten_data_files_count >= 2);
    assert_eq!(
        scan_rows(&table).await,
        before,
        "file-scoped DV on rewritten A is applied; sibling DV on kept still applies"
    );
    let live_data = live_data_file_paths(&table).await;
    assert!(
        live_data.contains(&kept_path),
        "current-spec sibling data file must not be rewritten away"
    );
    assert!(
        !live_delete_file_paths(&table).await.is_empty(),
        "rewritten sibling puffin must remain"
    );
}
