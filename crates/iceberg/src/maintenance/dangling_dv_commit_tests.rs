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

use crate::Catalog;
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, create_partitioned_table, local_fs_catalog, scan_rows,
    write_data_file,
};
use crate::maintenance::rewrite_data_files_cow_bytes_tests::{
    file_scoped_metrics, write_position_delete,
};
use crate::maintenance::rewrite_data_files_router_bound_tests::write_dv;
use crate::spec::{DataContentType, DataFile, FormatVersion, ManifestContentType};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};

fn rows(part: i64, first: i64, count: i64) -> Vec<(i64, i64, i64)> {
    (first..first + count).map(|y| (part, y, y)).collect()
}

async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
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

struct SharedPuffin {
    table: Table,
    a: DataFile,
    a2_rows: Vec<(i64, i64, i64)>,
    b_dv: DataFile,
    live_rows: Vec<(i64, i64, i64)>,
}

async fn shared_puffin_table(catalog: &impl Catalog) -> SharedPuffin {
    let table = create_partitioned_table(catalog, FormatVersion::V3).await;
    let a = write_data_file(&table, "a.parquet", 0, &rows(0, 0, 3)).await;
    let b = write_data_file(&table, "b.parquet", 1, &rows(1, 10, 3)).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = append_files(catalog, &table, vec![a.clone(), b.clone()]).await;
    let dvs = write_dv(&table, 0, &[
        (a_path.as_str(), &[1]),
        (b_path.as_str(), &[1]),
    ])
    .await;
    assert_eq!(dvs.len(), 2);
    assert_eq!(dvs[0].file_path(), dvs[1].file_path());
    assert_eq!(
        dvs[0].referenced_data_file().as_deref(),
        Some(a_path.as_str())
    );
    assert_eq!(
        dvs[1].referenced_data_file().as_deref(),
        Some(b_path.as_str())
    );
    let table = add_deletes(catalog, &table, dvs.clone()).await;
    let live_rows = vec![(0, 0, 0), (0, 2, 2), (1, 10, 10), (1, 12, 12)];
    assert_eq!(scan_rows(&table).await, live_rows);
    SharedPuffin {
        table,
        a,
        a2_rows: rows(0, 20, 2),
        b_dv: dvs[1].clone(),
        live_rows,
    }
}

async fn assert_only_b_dv_lives(
    table: &Table,
    b_dv: &DataFile,
    expected_rows: Vec<(i64, i64, i64)>,
) {
    let live = live_delete_files(table).await;
    assert_eq!(
        live.len(),
        1,
        "only the sibling blob of the shared Puffin file may stay live"
    );
    assert_eq!(live[0].file_path(), b_dv.file_path());
    assert_eq!(live[0].content_offset(), b_dv.content_offset());
    assert_eq!(
        live[0].content_size_in_bytes(),
        b_dv.content_size_in_bytes()
    );
    assert_eq!(live[0].referenced_data_file(), b_dv.referenced_data_file());
    assert_eq!(summary_value(table, "removed-dvs").as_deref(), Some("1"));
    assert_eq!(
        summary_value(table, "removed-delete-files").as_deref(),
        Some("1")
    );
    assert_eq!(
        summary_value(table, "removed-position-deletes").as_deref(),
        Some("1")
    );
    assert_eq!(
        summary_value(table, "total-delete-files").as_deref(),
        Some("1")
    );
    assert_eq!(
        summary_value(table, "total-position-deletes").as_deref(),
        Some("1")
    );
    assert_eq!(scan_rows(table).await, expected_rows);
}

async fn assert_both_dvs_live(table: &Table, expected_rows: Vec<(i64, i64, i64)>) {
    assert_eq!(live_delete_files(table).await.len(), 2);
    assert_eq!(summary_value(table, "removed-delete-files"), None);
    assert_eq!(scan_rows(table).await, expected_rows);
}

#[tokio::test]
async fn test_dangling_dv_delete_files_drops_dv_of_removed_data_file() {
    let (catalog, _temp) = local_fs_catalog().await;
    let st = shared_puffin_table(&catalog).await;
    let tx = Transaction::new(&st.table);
    let tx = tx
        .delete_files()
        .delete_data_files([st.a])
        .apply(tx)
        .expect("delete files");
    let table = commit(&catalog, tx).await;
    assert_only_b_dv_lives(&table, &st.b_dv, vec![(1, 10, 10), (1, 12, 12)]).await;
}

#[tokio::test]
async fn test_dangling_dv_overwrite_files_drops_dv_of_removed_data_file() {
    let (catalog, _temp) = local_fs_catalog().await;
    let st = shared_puffin_table(&catalog).await;
    let a2 = write_data_file(&st.table, "a2.parquet", 0, &st.a2_rows).await;
    let tx = Transaction::new(&st.table);
    let tx = tx
        .overwrite_files()
        .delete_data_files([st.a])
        .add_file(a2)
        .apply(tx)
        .expect("overwrite files");
    let table = commit(&catalog, tx).await;
    let mut expected = st.a2_rows.clone();
    expected.extend([(1, 10, 10), (1, 12, 12)]);
    expected.sort_unstable();
    assert_only_b_dv_lives(&table, &st.b_dv, expected).await;
}

#[tokio::test]
async fn test_dangling_dv_replace_partitions_drops_dv_of_removed_data_file() {
    let (catalog, _temp) = local_fs_catalog().await;
    let st = shared_puffin_table(&catalog).await;
    let a2 = write_data_file(&st.table, "a2.parquet", 0, &st.a2_rows).await;
    let tx = Transaction::new(&st.table);
    let tx = tx
        .replace_partitions()
        .add_file(a2)
        .apply(tx)
        .expect("replace partitions");
    let table = commit(&catalog, tx).await;
    let mut expected = st.a2_rows.clone();
    expected.extend([(1, 10, 10), (1, 12, 12)]);
    expected.sort_unstable();
    assert_only_b_dv_lives(&table, &st.b_dv, expected).await;
}

#[tokio::test]
async fn test_dangling_dv_row_delta_drops_dv_of_removed_data_file() {
    let (catalog, _temp) = local_fs_catalog().await;
    let st = shared_puffin_table(&catalog).await;
    let tx = Transaction::new(&st.table);
    let tx = tx
        .row_delta()
        .remove_rows(st.a)
        .apply(tx)
        .expect("row delta");
    let table = commit(&catalog, tx).await;
    assert_only_b_dv_lives(&table, &st.b_dv, vec![(1, 10, 10), (1, 12, 12)]).await;
}

#[tokio::test]
async fn test_dangling_dv_rewrite_files_drops_dv_of_removed_data_file() {
    let (catalog, _temp) = local_fs_catalog().await;
    let st = shared_puffin_table(&catalog).await;
    let a2 = write_data_file(&st.table, "a2.parquet", 0, &st.a2_rows).await;
    let tx = Transaction::new(&st.table);
    let tx = tx
        .rewrite_files([st.a], [a2])
        .apply(tx)
        .expect("rewrite files");
    let table = commit(&catalog, tx).await;
    let mut expected = st.a2_rows.clone();
    expected.extend([(1, 10, 10), (1, 12, 12)]);
    expected.sort_unstable();
    assert_only_b_dv_lives(&table, &st.b_dv, expected).await;
}

#[tokio::test]
async fn test_dangling_dv_kept_when_removed_data_file_carries_no_dv() {
    let (catalog, _temp) = local_fs_catalog().await;
    let st = shared_puffin_table(&catalog).await;
    let c = write_data_file(&st.table, "c.parquet", 1, &rows(1, 30, 2)).await;
    let table = append_files(&catalog, &st.table, vec![c.clone()]).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .delete_files()
        .delete_data_files([c])
        .apply(tx)
        .expect("delete files");
    let table = commit(&catalog, tx).await;
    assert_both_dvs_live(&table, st.live_rows.clone()).await;
}

#[tokio::test]
async fn test_dangling_dv_fast_append_keeps_dvs() {
    let (catalog, _temp) = local_fs_catalog().await;
    let st = shared_puffin_table(&catalog).await;
    let c = write_data_file(&st.table, "c.parquet", 1, &rows(1, 30, 2)).await;
    let table = append_files(&catalog, &st.table, vec![c]).await;
    let mut expected = st.live_rows.clone();
    expected.extend(rows(1, 30, 2));
    expected.sort_unstable();
    assert_both_dvs_live(&table, expected).await;
}

#[tokio::test]
async fn test_dangling_dv_merge_append_keeps_dvs() {
    let (catalog, _temp) = local_fs_catalog().await;
    let st = shared_puffin_table(&catalog).await;
    let c = write_data_file(&st.table, "c.parquet", 1, &rows(1, 30, 2)).await;
    let tx = Transaction::new(&st.table);
    let tx = tx
        .merge_append()
        .add_data_files([c])
        .apply(tx)
        .expect("merge append");
    let table = commit(&catalog, tx).await;
    let mut expected = st.live_rows.clone();
    expected.extend(rows(1, 30, 2));
    expected.sort_unstable();
    assert_both_dvs_live(&table, expected).await;
}

#[tokio::test]
async fn test_dangling_dv_row_delta_adding_deletes_only_keeps_dvs() {
    let (catalog, _temp) = local_fs_catalog().await;
    let st = shared_puffin_table(&catalog).await;
    let c = write_data_file(&st.table, "c.parquet", 1, &rows(1, 30, 2)).await;
    let table = append_files(&catalog, &st.table, vec![c.clone()]).await;
    let extra = write_dv(&table, 0, &[(c.file_path(), &[0])]).await;
    assert_eq!(extra.len(), 1);
    let table = add_deletes(&catalog, &table, extra).await;
    assert_eq!(live_delete_files(&table).await.len(), 3);
    assert_eq!(summary_value(&table, "removed-delete-files"), None);
    assert_eq!(scan_rows(&table).await, vec![
        (0, 0, 0),
        (0, 2, 2),
        (1, 10, 10),
        (1, 12, 12),
        (1, 31, 31)
    ]);
}

async fn delete_manifest_referenced_paths(table: &Table) -> Vec<Vec<String>> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut manifests = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        let mut referenced: Vec<String> = manifest
            .entries()
            .iter()
            .filter_map(|entry| entry.data_file().referenced_data_file())
            .collect();
        referenced.sort_unstable();
        manifests.push(referenced);
    }
    manifests
}

#[tokio::test]
async fn test_dangling_dv_delete_manifest_order_survives_concurrent_loads() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;
    let s1 = write_data_file(&table, "s1.parquet", 0, &rows(0, 400, 1)).await;
    let s2 = write_data_file(&table, "s2.parquet", 0, &rows(0, 401, 1)).await;
    let keep = write_data_file(&table, "keep.parquet", 1, &rows(1, 500, 1)).await;
    let mut big = Vec::new();
    for index in 0..128i64 {
        big.push(
            write_data_file(
                &table,
                &format!("big{index}.parquet"),
                0,
                &rows(0, 10_000 + index, 1),
            )
            .await,
        );
    }
    let mut all = vec![s1.clone(), s2.clone(), keep.clone()];
    all.extend(big.iter().cloned());
    let table = append_files(&catalog, &table, all).await;

    let s1_path = s1.file_path().to_string();
    let s2_path = s2.file_path().to_string();
    let s1_dvs = write_dv(&table, 0, &[(s1_path.as_str(), &[0])]).await;
    let table = add_deletes(&catalog, &table, s1_dvs).await;
    let s2_dvs = write_dv(&table, 0, &[(s2_path.as_str(), &[0])]).await;
    let table = add_deletes(&catalog, &table, s2_dvs).await;
    let big_pairs: Vec<(&str, &[u64])> = big
        .iter()
        .map(|file| (file.file_path(), &[0][..]))
        .collect();
    let big_dvs = write_dv(&table, 0, &big_pairs).await;
    assert_eq!(big_dvs.len(), 128);
    let table = add_deletes(&catalog, &table, big_dvs).await;

    let pre = delete_manifest_referenced_paths(&table).await;
    assert_eq!(
        pre.len(),
        3,
        "three row-delta commits leave three delete manifests"
    );
    assert_eq!(
        pre[0].len(),
        128,
        "the fat manifest sits first so its slower load would invert completion order"
    );

    let tx = Transaction::new(&table);
    let mut removed = vec![s1, s2];
    removed.extend(big.iter().cloned());
    let tx = tx
        .delete_files()
        .delete_data_files(removed)
        .apply(tx)
        .expect("delete files");
    let table = commit(&catalog, tx).await;

    let post = delete_manifest_referenced_paths(&table).await;
    assert_eq!(
        post, pre,
        "concurrent loads must not reorder the delete manifests in the committed list"
    );
    assert_eq!(live_delete_files(&table).await.len(), 0);
    assert_eq!(scan_rows(&table).await, vec![(1, 500, 500)]);
}

#[tokio::test]
async fn test_dangling_dv_parquet_position_delete_not_dropped_by_this_rule() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let a = write_data_file(&table, "a.parquet", 0, &rows(0, 0, 3)).await;
    let b = write_data_file(&table, "b.parquet", 1, &rows(1, 10, 3)).await;
    let a_path = a.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a.clone(), b]).await;
    let pd = write_position_delete(&table, 0, &[(a_path, 1)], file_scoped_metrics()).await;
    let pd_path = pd.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![pd]).await;
    assert_eq!(scan_rows(&table).await, vec![
        (0, 0, 0),
        (0, 2, 2),
        (1, 10, 10),
        (1, 11, 11),
        (1, 12, 12)
    ]);

    let tx = Transaction::new(&table);
    let tx = tx
        .overwrite_files()
        .delete_data_files([a])
        .apply(tx)
        .expect("overwrite files");
    let table = commit(&catalog, tx).await;
    let live = live_delete_files(&table).await;
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].file_path(), pd_path);
    assert_eq!(summary_value(&table, "removed-delete-files"), None);
    assert_eq!(scan_rows(&table).await, vec![
        (1, 10, 10),
        (1, 11, 11),
        (1, 12, 12)
    ]);
}
