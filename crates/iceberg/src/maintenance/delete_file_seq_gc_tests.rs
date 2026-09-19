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

use crate::Catalog;
use crate::maintenance::RemoveDanglingDeleteFiles;
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, create_partitioned_table, live_delete_file_paths, local_fs_catalog,
    scan_rows, write_data_file, write_equality_delete_file,
};
use crate::maintenance::rewrite_data_files_cow_bytes_tests::{
    file_scoped_metrics, partition_scoped_metrics, write_position_delete,
};
use crate::maintenance::rewrite_data_files_router_bound_tests::write_dv;
use crate::maintenance::rewrite_position_delete_files::RewritePositionDeleteFiles;
use crate::spec::{DataContentType, DataFile, FormatVersion};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};

fn rows(part: i64, first: i64, count: i64) -> Vec<(i64, i64, i64)> {
    (first..first + count).map(|y| (part, y, y)).collect()
}

async fn live_data_file_paths(table: &Table) -> HashSet<String> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut paths = HashSet::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                paths.insert(entry.file_path().to_string());
            }
        }
    }
    paths
}

fn summary_value(table: &Table, key: &str) -> Option<String> {
    table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .summary()
        .additional_properties
        .get(key)
        .cloned()
}

async fn commit(catalog: &impl Catalog, tx: Transaction) -> Table {
    tx.commit(catalog).await.expect("commit must succeed")
}

#[tokio::test]
async fn test_seq_gc_residue_rpd_then_rdf_reaches_zero_delete_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let mut files = Vec::new();
    for part in 0..2i64 {
        for fileno in 0..8i64 {
            let first = (part * 8 + fileno) * 20;
            let name = format!("p{part}-f{fileno}.parquet");
            files.push(write_data_file(&table, &name, part, &rows(part, first, 20)).await);
        }
    }
    let table = append_files(&catalog, &table, files.clone()).await;

    let mut deletes = Vec::new();
    for (index, file) in files.iter().enumerate() {
        let part = if index < 8 { 0 } else { 1 };
        let positions: Vec<(String, i64)> = (0..20)
            .step_by(2)
            .map(|pos| (file.file_path().to_string(), pos))
            .collect();
        deletes.push(write_position_delete(&table, part, &positions, file_scoped_metrics()).await);
    }
    let table = add_deletes(&catalog, &table, deletes).await;
    let rows_before = scan_rows(&table).await;
    assert_eq!(rows_before.len(), 160);
    assert!(rows_before.iter().all(|(_, y, _)| y % 2 == 1));
    assert_eq!(live_delete_file_paths(&table).await.len(), 16);

    let rpd = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect("rewrite_position_delete_files must succeed");
    assert_eq!(rpd.rewritten_delete_files_count(), 16);
    assert_eq!(rpd.added_delete_files_count(), 2);
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after rpd");
    assert_eq!(live_delete_file_paths(&table).await.len(), 2);
    assert_eq!(scan_rows(&table).await, rows_before);

    let rdf = RewriteDataFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect("rewrite_data_files must succeed");
    assert_eq!(rdf.rewritten_data_files_count, 16);
    assert_eq!(rdf.added_data_files_count, 2);
    assert_eq!(rdf.removed_delete_files_count, 0);
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after rdf");

    assert_eq!(live_data_file_paths(&table).await.len(), 2);
    assert!(
        live_delete_file_paths(&table).await.is_empty(),
        "Spark 4.1.2 + Iceberg 1.11.0 ends the residue sequence at zero delete files"
    );
    assert_eq!(scan_rows(&table).await, rows_before);
    assert_eq!(
        summary_value(&table, "removed-position-delete-files").as_deref(),
        Some("2")
    );
    assert_eq!(
        summary_value(&table, "removed-delete-files").as_deref(),
        Some("2")
    );
    assert_eq!(
        summary_value(&table, "total-delete-files").as_deref(),
        Some("0")
    );
    assert_eq!(
        summary_value(&table, "total-position-deletes").as_deref(),
        Some("0")
    );
}

#[tokio::test]
async fn test_seq_gc_keeps_position_delete_at_the_minimum_live_sequence() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let old = write_data_file(&table, "old.parquet", 0, &rows(0, 0, 10)).await;
    let table = append_files(&catalog, &table, vec![old.clone()]).await;

    let fresh = write_data_file(&table, "fresh.parquet", 1, &rows(1, 100, 10)).await;
    let fresh_path = fresh.file_path().to_string();
    let delete = write_position_delete(
        &table,
        1,
        &[(fresh_path.clone(), 0), (fresh_path, 1)],
        partition_scoped_metrics(),
    )
    .await;
    let delete_path = delete.file_path().to_string();
    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_data_files([fresh])
        .add_deletes([delete])
        .apply(tx)
        .expect("row delta");
    let table = commit(&catalog, tx).await;
    assert_eq!(scan_rows(&table).await.len(), 18);

    let tx = Transaction::new(&table);
    let tx = tx
        .delete_files()
        .delete_data_files([old])
        .apply(tx)
        .expect("delete files");
    let table = commit(&catalog, tx).await;

    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([delete_path]),
        "a delete whose sequence equals the minimum live data sequence still applies"
    );
    assert_eq!(scan_rows(&table).await, rows(1, 102, 8));
    assert_eq!(summary_value(&table, "removed-delete-files"), None);
}

#[tokio::test]
async fn test_seq_gc_keeps_equality_delete_applying_to_older_live_data() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let older = write_data_file(&table, "older.parquet", 0, &rows(0, 0, 10)).await;
    let table = append_files(&catalog, &table, vec![older]).await;
    let newer = write_data_file(&table, "newer.parquet", 1, &rows(1, 100, 10)).await;
    let table = append_files(&catalog, &table, vec![newer.clone()]).await;
    let eq_delete = write_equality_delete_file(&table, 0, &[3]).await;
    let eq_path = eq_delete.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .delete_files()
        .delete_data_files([newer])
        .apply(tx)
        .expect("delete files");
    let table = commit(&catalog, tx).await;

    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([eq_path])
    );
    let expected: Vec<(i64, i64, i64)> = rows(0, 0, 10)
        .into_iter()
        .filter(|(_, y, _)| *y != 3)
        .collect();
    assert_eq!(scan_rows(&table).await, expected);
}

struct StaleDelete {
    table: Table,
    a: DataFile,
    b: DataFile,
    delete_path: String,
}

async fn stale_delete_table(catalog: &impl Catalog) -> StaleDelete {
    let table = create_partitioned_table(catalog, FormatVersion::V2).await;
    let ghost = format!("{}/data/ghost.parquet", table.metadata().location());
    let delete = write_position_delete(&table, 0, &[(ghost, 0)], partition_scoped_metrics()).await;
    let delete_path = delete.file_path().to_string();
    let table = add_deletes(catalog, &table, vec![delete]).await;
    let a = write_data_file(&table, "a.parquet", 0, &rows(0, 0, 10)).await;
    let b = write_data_file(&table, "b.parquet", 1, &rows(1, 100, 10)).await;
    let table = append_files(catalog, &table, vec![a.clone(), b.clone()]).await;
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([delete_path.clone()])
    );
    StaleDelete {
        table,
        a,
        b,
        delete_path,
    }
}

async fn assert_retired(table: &Table, expected_rows: Vec<(i64, i64, i64)>) {
    assert!(
        live_delete_file_paths(table).await.is_empty(),
        "a delete older than every live data file is retired by a merging commit"
    );
    assert_eq!(
        summary_value(table, "removed-position-delete-files").as_deref(),
        Some("1")
    );
    assert_eq!(
        summary_value(table, "removed-delete-files").as_deref(),
        Some("1")
    );
    assert_eq!(
        summary_value(table, "total-delete-files").as_deref(),
        Some("0")
    );
    assert_eq!(scan_rows(table).await, expected_rows);
}

async fn assert_kept(table: &Table, delete_path: &str) {
    assert!(
        live_delete_file_paths(table).await.contains(delete_path),
        "this commit must not retire the stale delete"
    );
    assert_eq!(summary_value(table, "removed-delete-files"), None);
}

fn a_and(extra: Vec<(i64, i64, i64)>) -> Vec<(i64, i64, i64)> {
    let mut all = rows(0, 0, 10);
    all.extend(extra);
    all.sort_unstable();
    all
}

#[tokio::test]
async fn test_seq_gc_delete_files_retires_stale_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let stale = stale_delete_table(&catalog).await;
    let tx = Transaction::new(&stale.table);
    let tx = tx
        .delete_files()
        .delete_data_files([stale.b])
        .apply(tx)
        .expect("delete files");
    let table = commit(&catalog, tx).await;
    assert_retired(&table, a_and(vec![])).await;
}

#[tokio::test]
async fn test_seq_gc_overwrite_files_retires_stale_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let stale = stale_delete_table(&catalog).await;
    let b2 = write_data_file(&stale.table, "b2.parquet", 1, &rows(1, 200, 5)).await;
    let tx = Transaction::new(&stale.table);
    let tx = tx
        .overwrite_files()
        .delete_data_files([stale.b])
        .add_file(b2)
        .apply(tx)
        .expect("overwrite files");
    let table = commit(&catalog, tx).await;
    assert_retired(&table, a_and(rows(1, 200, 5))).await;
}

#[tokio::test]
async fn test_seq_gc_replace_partitions_retires_stale_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let stale = stale_delete_table(&catalog).await;
    let b2 = write_data_file(&stale.table, "b2.parquet", 1, &rows(1, 200, 5)).await;
    let tx = Transaction::new(&stale.table);
    let tx = tx
        .replace_partitions()
        .add_file(b2)
        .apply(tx)
        .expect("replace partitions");
    let table = commit(&catalog, tx).await;
    assert_retired(&table, a_and(rows(1, 200, 5))).await;
}

#[tokio::test]
async fn test_seq_gc_row_delta_removing_data_retires_stale_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let stale = stale_delete_table(&catalog).await;
    let tx = Transaction::new(&stale.table);
    let tx = tx
        .row_delta()
        .remove_rows(stale.b)
        .apply(tx)
        .expect("row delta");
    let table = commit(&catalog, tx).await;
    assert_retired(&table, a_and(vec![])).await;
}

#[tokio::test]
async fn test_seq_gc_rewrite_files_retires_stale_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let stale = stale_delete_table(&catalog).await;
    let b2 = write_data_file(&stale.table, "b2.parquet", 1, &rows(1, 100, 10)).await;
    let tx = Transaction::new(&stale.table);
    let tx = tx
        .rewrite_files([stale.b], [b2])
        .apply(tx)
        .expect("rewrite files");
    let table = commit(&catalog, tx).await;
    assert_retired(&table, a_and(rows(1, 100, 10))).await;
}

#[tokio::test]
async fn test_seq_gc_fast_append_keeps_stale_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let stale = stale_delete_table(&catalog).await;
    let c = write_data_file(&stale.table, "c.parquet", 1, &rows(1, 300, 5)).await;
    let table = append_files(&catalog, &stale.table, vec![c]).await;
    assert_kept(&table, &stale.delete_path).await;
}

#[tokio::test]
async fn test_seq_gc_merge_append_keeps_stale_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let stale = stale_delete_table(&catalog).await;
    let c = write_data_file(&stale.table, "c.parquet", 1, &rows(1, 300, 5)).await;
    let tx = Transaction::new(&stale.table);
    let tx = tx
        .merge_append()
        .add_data_files([c])
        .apply(tx)
        .expect("merge append");
    let table = commit(&catalog, tx).await;
    assert_kept(&table, &stale.delete_path).await;
}

#[tokio::test]
async fn test_seq_gc_row_delta_adding_deletes_only_keeps_stale_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let stale = stale_delete_table(&catalog).await;
    let extra = write_position_delete(
        &stale.table,
        0,
        &[(stale.a.file_path().to_string(), 0)],
        partition_scoped_metrics(),
    )
    .await;
    let table = add_deletes(&catalog, &stale.table, vec![extra]).await;
    assert_kept(&table, &stale.delete_path).await;
    assert_eq!(live_delete_file_paths(&table).await.len(), 2);
}

#[tokio::test]
async fn test_remove_dangling_keeps_a_foreign_partition_dv_whose_data_file_is_live() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;
    let a = write_data_file(&table, "a.parquet", 1, &rows(1, 10, 3)).await;
    let a_path = a.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a]).await;
    let dv = write_dv(&table, 2, &[(a_path.as_str(), &[1])]).await;
    assert_eq!(dv.len(), 1);
    assert_eq!(
        dv[0].referenced_data_file().as_deref(),
        Some(a_path.as_str())
    );
    let dv_path = dv[0].file_path().to_string();
    let table = add_deletes(&catalog, &table, dv).await;
    let expected = vec![(1, 10, 10), (1, 12, 12)];
    assert_eq!(scan_rows(&table).await, expected);

    let result = RemoveDanglingDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect("remove dangling deletes");
    assert_eq!(
        result.removed_dvs_count(),
        0,
        "a DV whose referenced data file is live still masks a row and must not be collected"
    );
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after remove dangling");
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([dv_path])
    );
    assert_eq!(scan_rows(&table).await, expected);
}
