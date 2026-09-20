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

use std::collections::HashSet;

use super::{BinDisposition, bin_disposition};
use crate::Catalog;
use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestContentType,
    ManifestFile, ManifestStatus, Operation, Struct,
};
use crate::table::Table;
use crate::transaction::tests::{
    make_v2_minimal_table_in_catalog, make_v3_minimal_table_in_catalog,
};
use crate::transaction::{ApplyTransactionAction, Transaction};

#[test]
fn test_bin_disposition_size_one_is_kept() {
    assert_eq!(bin_disposition(1, false, 2), BinDisposition::Keep);
    assert_eq!(
        bin_disposition(1, true, 2),
        BinDisposition::Keep,
        "even the bin WITH first is kept at size 1"
    );
}

#[test]
fn test_bin_disposition_two_without_first_merges_below_min_count() {
    assert_eq!(
        bin_disposition(2, false, 100),
        BinDisposition::Merge,
        "a >=2 bin WITHOUT first merges even far below min-count"
    );
}

#[test]
fn test_bin_disposition_with_first_respects_min_count_on_the_boundary() {
    assert_eq!(
        bin_disposition(2, true, 3),
        BinDisposition::Keep,
        "bin WITH first below min-count (2 < 3) is kept"
    );
    assert_eq!(
        bin_disposition(2, true, 2),
        BinDisposition::Merge,
        "bin WITH first AT min-count (2 == 2, gate is strict `<`) merges"
    );
    assert_eq!(
        bin_disposition(3, true, 2),
        BinDisposition::Merge,
        "bin WITH first above min-count merges"
    );
}

fn data_file(path: &str, part_value: i64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(part_value))]))
        .build()
        .unwrap()
}

async fn fast_append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

async fn merge_append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.merge_append().add_data_files(files);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

async fn set_table_property(
    catalog: &impl Catalog,
    table: &Table,
    key: &str,
    value: &str,
) -> Table {
    let tx = Transaction::new(table);
    let action = tx
        .update_table_properties()
        .set(key.to_string(), value.to_string());
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

async fn current_manifests(table: &Table) -> Vec<ManifestFile> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap()
        .entries()
        .to_vec()
}

async fn live_file_paths(table: &Table) -> HashSet<String> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    let mut live = HashSet::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() {
                live.insert(entry.file_path().to_string());
            }
        }
    }
    live
}

async fn live_entry(
    table: &Table,
    path: &str,
) -> (ManifestStatus, Option<i64>, Option<i64>, Option<i64>) {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() && entry.file_path() == path {
                return (
                    entry.status(),
                    entry.snapshot_id(),
                    entry.sequence_number(),
                    entry.file_sequence_number,
                );
            }
        }
    }
    panic!("no live entry for {path}");
}

fn summary_prop(table: &Table, prop: &str) -> Option<String> {
    table
        .metadata()
        .current_snapshot()
        .unwrap()
        .summary()
        .additional_properties
        .get(prop)
        .cloned()
}

async fn data_manifest_count(table: &Table) -> usize {
    current_manifests(table)
        .await
        .iter()
        .filter(|m| m.content == ManifestContentType::Data)
        .count()
}

#[tokio::test]
async fn test_merge_append_below_min_count_does_not_merge() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;

    let table = fast_append(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
    let sa = table.metadata().current_snapshot().unwrap().snapshot_id();
    let table = fast_append(&catalog, &table, vec![data_file("test/b.parquet", 0)]).await;
    let sb = table.metadata().current_snapshot().unwrap().snapshot_id();

    let table = merge_append(&catalog, &table, vec![data_file("test/c.parquet", 0)]).await;

    assert_eq!(
        data_manifest_count(&table).await,
        3,
        "below the default min-count, the three data manifests are not merged"
    );
    let (status_a, snap_a, _, _) = live_entry(&table, "test/a.parquet").await;
    let (status_b, snap_b, _, _) = live_entry(&table, "test/b.parquet").await;
    assert_eq!(snap_a, Some(sa), "a keeps its original adding snapshot id");
    assert_eq!(snap_b, Some(sb), "b keeps its original adding snapshot id");
    assert_eq!(status_a, ManifestStatus::Added);
    assert_eq!(status_b, ManifestStatus::Added);
    let merge_snap = table.metadata().current_snapshot().unwrap().snapshot_id();
    let (status_c, snap_c, _, _) = live_entry(&table, "test/c.parquet").await;
    assert_eq!(status_c, ManifestStatus::Added);
    assert_eq!(snap_c, Some(merge_snap));
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/a.parquet".to_string(),
            "test/b.parquet".to_string(),
            "test/c.parquet".to_string(),
        ])
    );
}

#[tokio::test]
async fn test_merge_append_at_threshold_merges_and_preserves_provenance() {
    use crate::spec::Manifest;

    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.min-count-to-merge", "2").await;

    let table = fast_append(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
    let sa = table.metadata().current_snapshot().unwrap().snapshot_id();
    let (_, _, a_seq, a_fseq) = live_entry(&table, "test/a.parquet").await;
    let table = fast_append(&catalog, &table, vec![data_file("test/b.parquet", 0)]).await;
    let sb = table.metadata().current_snapshot().unwrap().snapshot_id();
    let (_, _, b_seq, b_fseq) = live_entry(&table, "test/b.parquet").await;
    assert!(a_seq.is_some() && a_fseq.is_some() && b_seq.is_some() && b_fseq.is_some());

    let table = merge_append(&catalog, &table, vec![data_file("test/c.parquet", 0)]).await;
    let merge_snap = table.metadata().current_snapshot().unwrap().snapshot_id();

    let manifests = current_manifests(&table).await;
    let data_manifests: Vec<&ManifestFile> = manifests
        .iter()
        .filter(|m| m.content == ManifestContentType::Data)
        .collect();
    assert_eq!(
        data_manifests.len(),
        1,
        "at/above min-count the three data manifests merge into one"
    );
    let merged_list_seq = data_manifests[0].sequence_number;

    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/a.parquet".to_string(),
            "test/b.parquet".to_string(),
            "test/c.parquet".to_string(),
        ])
    );

    let (status_a, snap_a, seq_a, fseq_a) = live_entry(&table, "test/a.parquet").await;
    assert_eq!(status_a, ManifestStatus::Existing, "a becomes Existing");
    assert_eq!(snap_a, Some(sa), "a keeps ORIGINAL adding snapshot id");
    assert_eq!(seq_a, a_seq, "a keeps ORIGINAL data seq");
    assert_eq!(fseq_a, a_fseq, "a keeps ORIGINAL file seq");
    let (status_b, snap_b, seq_b, fseq_b) = live_entry(&table, "test/b.parquet").await;
    assert_eq!(status_b, ManifestStatus::Existing, "b becomes Existing");
    assert_eq!(snap_b, Some(sb), "b keeps ORIGINAL adding snapshot id");
    assert_eq!(seq_b, b_seq, "b keeps ORIGINAL data seq");
    assert_eq!(fseq_b, b_fseq, "b keeps ORIGINAL file seq");

    let (status_c, snap_c, seq_c, _) = live_entry(&table, "test/c.parquet").await;
    assert_eq!(status_c, ManifestStatus::Added, "c stays Added");
    assert_eq!(snap_c, Some(merge_snap), "c is added by the merge snapshot");
    assert_eq!(
        seq_c,
        Some(merged_list_seq),
        "c re-inherits the merge snapshot's seq"
    );

    let bytes = table
        .file_io()
        .new_input(&data_manifests[0].manifest_path)
        .unwrap()
        .read()
        .await
        .unwrap();
    let (_, raw_entries) = Manifest::try_from_avro_bytes(&bytes).unwrap();
    assert_eq!(
        raw_entries.len(),
        3,
        "all three entries in the merged manifest"
    );
    for entry in &raw_entries {
        match entry.file_path() {
            "test/a.parquet" => {
                assert_eq!(entry.status(), ManifestStatus::Existing);
                assert_eq!(
                    entry.sequence_number(),
                    a_seq,
                    "a: original data seq stored EXPLICITLY on disk"
                );
                assert_eq!(entry.file_sequence_number, a_fseq);
                assert_ne!(
                    entry.sequence_number(),
                    Some(merged_list_seq),
                    "a's on-disk seq must NOT be the merge snapshot's seq (the resurrection bug)"
                );
            }
            "test/b.parquet" => {
                assert_eq!(entry.status(), ManifestStatus::Existing);
                assert_eq!(entry.sequence_number(), b_seq);
                assert_eq!(entry.file_sequence_number, b_fseq);
            }
            "test/c.parquet" => {
                assert_eq!(entry.status(), ManifestStatus::Added);
                assert_eq!(
                    entry.sequence_number(),
                    None,
                    "c: this-commit added entry stores NULL seq on disk ⇒ re-inherits at commit"
                );
                assert_eq!(entry.file_sequence_number, None);
            }
            other => panic!("unexpected entry {other}"),
        }
    }

    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        Operation::Append,
        "merge_append records Operation::Append"
    );
    assert_eq!(
        summary_prop(&table, "manifests-created").as_deref(),
        Some("1"),
        "merge_append emits manifests-created for the merged manifest"
    );
    assert_eq!(summary_prop(&table, "manifests-kept").as_deref(), Some("0"));
    assert_eq!(
        summary_prop(&table, "manifests-replaced").as_deref(),
        Some("2"),
        "the two carried source manifests consumed by the merge count as replaced"
    );
    assert_eq!(
        summary_prop(&table, "total-data-files").as_deref(),
        Some("3"),
        "cumulative total-data-files is correct"
    );
    assert_eq!(
        summary_prop(&table, "total-records").as_deref(),
        Some("3"),
        "cumulative total-records is correct"
    );
}

#[tokio::test]
async fn test_merge_append_suppresses_old_tombstones() {
    use crate::spec::Manifest;

    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.min-count-to-merge", "2").await;

    let table = fast_append(&catalog, &table, vec![
        data_file("test/a.parquet", 0),
        data_file("test/b.parquet", 0),
    ])
    .await;
    let tx = Transaction::new(&table);
    let action = tx
        .delete_files()
        .delete_files(vec!["test/a.parquet".to_string()]);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();
    let delete_snap = table.metadata().current_snapshot().unwrap().snapshot_id();

    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from(["test/b.parquet".to_string()]),
        "after delete, only b is live"
    );
    let mut tombstone_with_live = false;
    for manifest_file in current_manifests(&table).await {
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        let has_a_tombstone = manifest.entries().iter().any(|entry| {
            entry.file_path() == "test/a.parquet" && entry.status() == ManifestStatus::Deleted
        });
        let has_b_live = manifest
            .entries()
            .iter()
            .any(|entry| entry.file_path() == "test/b.parquet" && entry.is_alive());
        if has_a_tombstone {
            let tombstone = manifest
                .entries()
                .iter()
                .find(|entry| entry.file_path() == "test/a.parquet")
                .unwrap();
            assert_eq!(
                tombstone.snapshot_id(),
                Some(delete_snap),
                "the tombstone's adding snapshot is the delete snapshot"
            );
            tombstone_with_live = has_b_live;
        }
    }
    assert!(
        tombstone_with_live,
        "the rewritten manifest carries a's tombstone alongside live b (so it reaches the merge)"
    );

    let table = merge_append(&catalog, &table, vec![data_file("test/c.parquet", 0)]).await;

    let manifests = current_manifests(&table).await;
    let data_manifests: Vec<&ManifestFile> = manifests
        .iter()
        .filter(|m| m.content == ManifestContentType::Data)
        .collect();
    assert_eq!(
        data_manifests.len(),
        1,
        "the data manifests merged into one at min-count 2"
    );

    for data_manifest in &data_manifests {
        let bytes = table
            .file_io()
            .new_input(&data_manifest.manifest_path)
            .unwrap()
            .read()
            .await
            .unwrap();
        let (_, raw_entries) = Manifest::try_from_avro_bytes(&bytes).unwrap();
        assert!(
            raw_entries
                .iter()
                .all(|e| e.file_path() != "test/a.parquet"),
            "the prior-snapshot DELETED tombstone for a is suppressed from the merged manifest"
        );
    }

    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from(["test/b.parquet".to_string(), "test/c.parquet".to_string()]),
        "a stays deleted, b and c live"
    );
}

#[tokio::test]
async fn test_merge_append_never_merges_across_specs_higher_spec_first() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.min-count-to-merge", "2").await;

    let table = fast_append(&catalog, &table, vec![data_file("test/spec0a.parquet", 0)]).await;
    let table = fast_append(&catalog, &table, vec![data_file("test/spec0b.parquet", 0)]).await;

    let tx = Transaction::new(&table);
    let action = tx.update_partition_spec().add_field("y");
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();
    let new_spec_id = table.metadata().default_partition_spec_id();
    assert_ne!(new_spec_id, 0);

    let spec1_file = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path("test/spec1a.parquet".to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(new_spec_id)
        .partition(Struct::from_iter([
            Some(Literal::long(0)),
            Some(Literal::long(0)),
        ]))
        .build()
        .unwrap();
    let table = fast_append(&catalog, &table, vec![spec1_file]).await;

    let spec1_new = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path("test/spec1b.parquet".to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(new_spec_id)
        .partition(Struct::from_iter([
            Some(Literal::long(0)),
            Some(Literal::long(0)),
        ]))
        .build()
        .unwrap();
    let table = merge_append(&catalog, &table, vec![spec1_new]).await;

    let manifests = current_manifests(&table).await;
    let data_manifests: Vec<&ManifestFile> = manifests
        .iter()
        .filter(|m| m.content == ManifestContentType::Data)
        .collect();
    for manifest in &data_manifests {
        let loaded = manifest.load_manifest(table.file_io()).await.unwrap();
        let specs: HashSet<i32> = loaded
            .entries()
            .iter()
            .map(|e| e.data_file().partition_spec_id)
            .collect();
        assert_eq!(
            specs.len(),
            1,
            "a merged manifest must carry exactly one partition spec id"
        );
        assert!(specs.contains(&manifest.partition_spec_id));
    }

    assert!(
        data_manifests[0].partition_spec_id >= new_spec_id,
        "the higher spec id group is emitted first (got {})",
        data_manifests[0].partition_spec_id
    );

    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/spec0a.parquet".to_string(),
            "test/spec0b.parquet".to_string(),
            "test/spec1a.parquet".to_string(),
            "test/spec1b.parquet".to_string(),
        ])
    );
}

#[tokio::test]
async fn test_merge_append_tiny_target_keeps_all_size_one_bins() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.min-count-to-merge", "2").await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.target-size-bytes", "1").await;

    let table = fast_append(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
    let table = fast_append(&catalog, &table, vec![data_file("test/b.parquet", 0)]).await;
    let table = merge_append(&catalog, &table, vec![data_file("test/c.parquet", 0)]).await;

    assert_eq!(
        data_manifest_count(&table).await,
        3,
        "a 1-byte target makes every manifest its own size-1 bin ⇒ no merge despite min-count=2"
    );
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/a.parquet".to_string(),
            "test/b.parquet".to_string(),
            "test/c.parquet".to_string(),
        ])
    );
}

#[tokio::test]
async fn test_merge_append_carries_delete_manifest_and_delete_still_applies() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.min-count-to-merge", "2").await;

    let real_data =
        write_real_data_file(&table, "rows.parquet", 0, &[(0, 10), (0, 20), (0, 30)]).await;
    let data_path = real_data.file_path().to_string();
    let table = fast_append(&catalog, &table, vec![real_data]).await;
    let real_data2 = write_real_data_file(&table, "rows2.parquet", 0, &[(0, 40)]).await;
    let table = fast_append(&catalog, &table, vec![real_data2]).await;

    let delete_file = write_real_position_delete(&table, 0, &[(data_path.clone(), 1)]).await;
    let tx = Transaction::new(&table);
    let action = tx.row_delta().add_deletes(vec![delete_file]);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let delete_manifest_count_before = current_manifests(&table)
        .await
        .iter()
        .filter(|m| m.content == ManifestContentType::Deletes)
        .count();
    assert_eq!(
        delete_manifest_count_before, 1,
        "one delete manifest exists"
    );
    assert_eq!(
        scan_y_values(&table).await,
        HashSet::from([10, 30, 40]),
        "the delete drops y=20 before the merge"
    );

    let real_data3 = write_real_data_file(&table, "rows3.parquet", 0, &[(0, 50)]).await;
    let table = fast_append(&catalog, &table, vec![real_data3]).await;
    let real_data4 = write_real_data_file(&table, "rows4.parquet", 0, &[(0, 60)]).await;
    let table = merge_append(&catalog, &table, vec![real_data4]).await;

    let delete_manifests: Vec<ManifestFile> = current_manifests(&table)
        .await
        .into_iter()
        .filter(|m| m.content == ManifestContentType::Deletes)
        .collect();
    assert_eq!(
        delete_manifests.len(),
        1,
        "the delete manifest is carried forward unchanged (count pin)"
    );

    assert_eq!(
        scan_y_values(&table).await,
        HashSet::from([10, 30, 40, 50, 60]),
        "after the merge the position delete still drops y=20"
    );
}

#[tokio::test]
async fn test_merge_append_cumulative_totals_correct() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.min-count-to-merge", "2").await;

    let table = fast_append(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
    let table = fast_append(&catalog, &table, vec![data_file("test/b.parquet", 0)]).await;
    let table = merge_append(&catalog, &table, vec![data_file("test/c.parquet", 0)]).await;

    assert_eq!(
        summary_prop(&table, "total-data-files").as_deref(),
        Some("3"),
        "three data files total after two appends + one merge_append"
    );
    assert_eq!(
        summary_prop(&table, "total-records").as_deref(),
        Some("3"),
        "three records total"
    );
    assert_eq!(
        summary_prop(&table, "added-data-files").as_deref(),
        Some("1"),
        "the merge_append added exactly one file this commit"
    );
}

#[tokio::test]
async fn test_merge_append_empty_is_rejected() {
    use std::sync::Arc;

    use crate::transaction::TransactionAction;

    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let tx = Transaction::new(&table);
    let action = tx.merge_append().add_data_files(vec![]);
    assert!(
        Arc::new(action).commit(&table).await.is_err(),
        "a merge_append with no data files and no properties must be rejected"
    );
}

async fn write_real_data_file(
    table: &Table,
    file_name: &str,
    part_value: i64,
    rows: &[(i64, i64)],
) -> DataFile {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};

    use crate::arrow::schema_to_arrow_schema;
    use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());
    let xs: Vec<i64> = rows.iter().map(|(x, _)| *x).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, y)| *y).collect();
    let zs: Vec<i64> = rows.iter().map(|(_, y)| *y * 10).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(xs)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
        Arc::new(Int64Array::from(zs)) as ArrayRef,
    ])
    .unwrap();

    let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
    let output = table.file_io().new_output(file_path).unwrap();
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.unwrap();
    writer.write(&batch).await.unwrap();
    let mut builder = writer.close().await.unwrap().into_iter().next().unwrap();
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(part_value))]))
        .build()
        .unwrap()
}

async fn write_real_position_delete(
    table: &Table,
    part_value: i64,
    deletes: &[(String, i64)],
) -> DataFile {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};

    use crate::spec::PartitionKey;
    use crate::writer::base_writer::position_delete_writer::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};

    let config = PositionDeleteWriterConfig::new().unwrap();
    let location_gen = DefaultLocationGenerator::new(table.metadata()).unwrap();
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
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(part_value))]),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key))
        .await
        .unwrap();

    let paths: Vec<&str> = deletes.iter().map(|(p, _)| p.as_str()).collect();
    let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap().into_iter().next().unwrap()
}

async fn scan_y_values(table: &Table) -> HashSet<i64> {
    use arrow_array::{Int64Array, RecordBatch};
    use futures::TryStreamExt;

    let stream = table
        .scan()
        .select(["y"])
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap();
    let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
    let mut values = HashSet::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..col.len() {
            values.insert(col.value(i));
        }
    }
    values
}

#[tokio::test]
async fn test_properties_only_merge_append_keeps_first_existing_bin_below_min_count() {
    use std::collections::HashMap;

    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table = fast_append(&catalog, &table, vec![data_file("test/audit-a.parquet", 0)]).await;
    let table = fast_append(&catalog, &table, vec![data_file("test/audit-b.parquet", 0)]).await;
    assert_eq!(current_manifests(&table).await.len(), 2);

    let tx = Transaction::new(&table);
    let action = tx.merge_append().set_snapshot_properties(HashMap::from([(
        "audit-pin".to_string(),
        "true".to_string(),
    )]));
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let manifests = current_manifests(&table).await;
    assert_eq!(
        manifests.len(),
        2,
        "a properties-only merge_append below min-count must keep the existing manifests un-merged (Java stream-head `first`)"
    );
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/audit-a.parquet".to_string(),
            "test/audit-b.parquet".to_string()
        ])
    );
}

#[path = "merge_append/tests/merge_append_extracted.rs"]
mod merge_append_extracted;
