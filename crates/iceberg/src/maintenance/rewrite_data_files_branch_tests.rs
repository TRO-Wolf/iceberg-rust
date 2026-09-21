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

use arrow_array::{Int64Array, RecordBatch};
use futures::TryStreamExt;

use crate::Catalog;
use crate::error::ErrorKind;
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    append_files, create_partitioned_table, local_fs_catalog, write_data_file,
};
use crate::spec::{DataContentType, DataFile, FormatVersion};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};

async fn create_branch(
    catalog: &impl Catalog,
    table: &Table,
    name: &str,
    snapshot_id: i64,
) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .manage_snapshots()
        .create_branch(name, snapshot_id)
        .apply(tx)
        .expect("apply create_branch");
    tx.commit(catalog).await.expect("commit create_branch")
}

async fn append_branch_files(
    catalog: &impl Catalog,
    table: &Table,
    branch: &str,
    files: Vec<DataFile>,
) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .to_branch(branch)
        .apply(tx)
        .expect("apply branch append");
    tx.commit(catalog).await.expect("commit branch append")
}

fn ref_snapshot_id(table: &Table, name: &str) -> Option<i64> {
    table.metadata().refs.get(name).map(|r| r.snapshot_id)
}

async fn live_data_file_paths_for_snapshot(table: &Table, snapshot_id: i64) -> HashSet<String> {
    let snapshot = table
        .metadata()
        .snapshot_by_id(snapshot_id)
        .expect("snapshot exists");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    let mut paths = HashSet::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                paths.insert(entry.file_path().to_string());
            }
        }
    }
    paths
}

fn column_i64<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int64Array {
    let index = batch.schema().index_of(name).expect("column exists");
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("long column")
}

async fn scan_rows_for_ref(table: &Table, ref_name: &str) -> Vec<(i64, i64, i64)> {
    let stream = table
        .scan()
        .use_ref(ref_name)
        .select(["x", "y", "z"])
        .build()
        .expect("build ref scan")
        .to_arrow()
        .await
        .expect("ref scan to arrow");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect ref batches");
    let mut rows: Vec<(i64, i64, i64)> = Vec::new();
    for batch in batches {
        let xs = column_i64(&batch, "x");
        let ys = column_i64(&batch, "y");
        let zs = column_i64(&batch, "z");
        for index in 0..xs.len() {
            rows.push((xs.value(index), ys.value(index), zs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}

#[tokio::test]
async fn branch_rewrite_commits_to_branch_and_leaves_main_untouched() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let ident = table.identifier().clone();
    let first = write_data_file(&table, "branch-main-0.parquet", 0, &[(0, 0, 0)]).await;
    let table = append_files(&catalog, &table, vec![first]).await;
    let second = write_data_file(&table, "branch-main-1.parquet", 0, &[(0, 1, 1)]).await;
    let table = append_files(&catalog, &table, vec![second]).await;
    let main_id = table.metadata().current_snapshot_id().expect("main head");
    let table = create_branch(&catalog, &table, "b1", main_id).await;
    let branch_before = ref_snapshot_id(&table, "b1").expect("branch head");
    let rows_before = scan_rows_for_ref(&table, "b1").await;
    assert_eq!(rows_before.len(), 2, "branch fixture holds two rows");

    let result = RewriteDataFiles::new(table)
        .rewrite_all(true)
        .branch("b1")
        .execute(&catalog)
        .await
        .expect("branch rewrite executes");
    assert_eq!(
        result.rewritten_data_files_count, 2,
        "both branch files are rewritten"
    );
    assert!(
        result.added_data_files_count >= 1,
        "the rewrite adds compacted files"
    );

    let table = catalog.load_table(&ident).await.expect("reload table");
    assert_eq!(
        table.metadata().current_snapshot_id(),
        Some(main_id),
        "main does not move"
    );
    let branch_after = ref_snapshot_id(&table, "b1").expect("branch still exists");
    assert_ne!(branch_after, branch_before, "the branch head moves");
    let branch_snapshot = table
        .metadata()
        .snapshot_by_id(branch_after)
        .expect("branch head snapshot");
    assert_eq!(
        branch_snapshot.parent_snapshot_id(),
        Some(branch_before),
        "the rewrite commits onto the branch head"
    );
    assert_eq!(
        scan_rows_for_ref(&table, "b1").await,
        rows_before,
        "branch rows survive the rewrite"
    );
}

#[tokio::test]
async fn branch_rewrite_plans_from_branch_files_not_main_files() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let ident = table.identifier().clone();
    let first = write_data_file(&table, "diverged-main-0.parquet", 0, &[(0, 0, 0)]).await;
    let table = append_files(&catalog, &table, vec![first]).await;
    let second = write_data_file(&table, "diverged-main-1.parquet", 0, &[(0, 1, 1)]).await;
    let table = append_files(&catalog, &table, vec![second]).await;
    let main_id = table.metadata().current_snapshot_id().expect("main head");
    let table = create_branch(&catalog, &table, "b1", main_id).await;
    let third = write_data_file(&table, "diverged-b1-0.parquet", 1, &[(1, 2, 2)]).await;
    let table = append_branch_files(&catalog, &table, "b1", vec![third]).await;
    let fourth = write_data_file(&table, "diverged-b1-1.parquet", 1, &[(1, 3, 3)]).await;
    let table = append_branch_files(&catalog, &table, "b1", vec![fourth]).await;
    let branch_before = ref_snapshot_id(&table, "b1").expect("diverged branch head");
    assert_ne!(branch_before, main_id, "branch must diverge from main");
    let main_paths_before = live_data_file_paths_for_snapshot(&table, main_id).await;
    assert_eq!(main_paths_before.len(), 2, "main holds two files");
    let rows_before = scan_rows_for_ref(&table, "b1").await;
    assert_eq!(rows_before.len(), 4, "branch holds four rows");

    let result = RewriteDataFiles::new(table)
        .rewrite_all(true)
        .branch("b1")
        .execute(&catalog)
        .await
        .expect("branch rewrite executes");
    assert_eq!(
        result.rewritten_data_files_count, 4,
        "the plan covers all four branch files, not main's two"
    );

    let table = catalog.load_table(&ident).await.expect("reload table");
    assert_eq!(
        table.metadata().current_snapshot_id(),
        Some(main_id),
        "main does not move"
    );
    assert_eq!(
        live_data_file_paths_for_snapshot(&table, main_id).await,
        main_paths_before,
        "main's live files are unchanged"
    );
    let branch_after = ref_snapshot_id(&table, "b1").expect("branch still exists");
    assert_ne!(branch_after, branch_before, "the branch head moves");
    assert_eq!(
        scan_rows_for_ref(&table, "b1").await,
        rows_before,
        "branch rows survive the rewrite"
    );
}

#[tokio::test]
async fn partial_progress_commits_every_batch_to_branch() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let ident = table.identifier().clone();
    let seed = write_data_file(&table, "pp-seed.parquet", 0, &[(0, 0, 0)]).await;
    let table = append_files(&catalog, &table, vec![seed]).await;
    let main_id = table.metadata().current_snapshot_id().expect("main head");
    let mut table = create_branch(&catalog, &table, "b1", main_id).await;
    for (part, tag) in [
        (0i64, "pp-a"),
        (0i64, "pp-b"),
        (1i64, "pp-c"),
        (1i64, "pp-d"),
    ] {
        let file = write_data_file(&table, &format!("{tag}.parquet"), part, &[(
            part,
            part + 10,
            part + 20,
        )])
        .await;
        table = append_branch_files(&catalog, &table, "b1", vec![file]).await;
    }
    let table = catalog.load_table(&ident).await.expect("reload table");
    let branch_before = ref_snapshot_id(&table, "b1").expect("diverged branch head");
    assert_ne!(branch_before, main_id, "branch must diverge from main");

    let result = RewriteDataFiles::new(table)
        .rewrite_all(true)
        .branch("b1")
        .partial_progress(true)
        .partial_progress_max_commits(2)
        .execute(&catalog)
        .await
        .expect("branch rewrite executes");
    assert_eq!(
        result.rewritten_data_files_count, 5,
        "all five branch files are rewritten across batches"
    );

    let table = catalog.load_table(&ident).await.expect("reload table");
    assert_eq!(
        table.metadata().current_snapshot_id(),
        Some(main_id),
        "main does not move under partial progress"
    );
    let branch_after = ref_snapshot_id(&table, "b1").expect("branch still exists");
    let first_parent = table
        .metadata()
        .snapshot_by_id(branch_after)
        .expect("branch head snapshot")
        .parent_snapshot_id()
        .expect("first batch commit");
    let second_parent = table
        .metadata()
        .snapshot_by_id(first_parent)
        .expect("first batch snapshot")
        .parent_snapshot_id()
        .expect("branch base");
    assert_eq!(
        second_parent, branch_before,
        "two batch commits land on the branch"
    );
}

#[tokio::test]
async fn unknown_branch_fails_before_planning() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let ident = table.identifier().clone();
    let file = write_data_file(&table, "ghost-0.parquet", 0, &[(0, 0, 0)]).await;
    let table = append_files(&catalog, &table, vec![file]).await;
    let main_id = table.metadata().current_snapshot_id().expect("main head");

    let error = RewriteDataFiles::new(table)
        .rewrite_all(true)
        .branch("ghost")
        .execute(&catalog)
        .await
        .expect_err("unknown branch must fail");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(error.message(), "snapshot ref 'ghost' not found");

    let table = catalog.load_table(&ident).await.expect("reload table");
    assert_eq!(
        table.metadata().current_snapshot_id(),
        Some(main_id),
        "a refused rewrite commits nothing"
    );
    assert!(
        ref_snapshot_id(&table, "ghost").is_none(),
        "a refused rewrite creates no ref"
    );
}

#[tokio::test]
async fn explicit_main_branch_behaves_like_default() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let ident = table.identifier().clone();
    let first = write_data_file(&table, "explicit-main-0.parquet", 0, &[(0, 0, 0)]).await;
    let table = append_files(&catalog, &table, vec![first]).await;
    let second = write_data_file(&table, "explicit-main-1.parquet", 0, &[(0, 1, 1)]).await;
    let table = append_files(&catalog, &table, vec![second]).await;
    let main_before = table.metadata().current_snapshot_id().expect("main head");

    let result = RewriteDataFiles::new(table)
        .rewrite_all(true)
        .branch("main")
        .execute(&catalog)
        .await
        .expect("explicit main rewrite executes");
    assert_eq!(result.rewritten_data_files_count, 2);

    let table = catalog.load_table(&ident).await.expect("reload table");
    let main_after = table.metadata().current_snapshot_id().expect("main head");
    assert_ne!(main_after, main_before, "explicit main moves main");
    assert_eq!(
        ref_snapshot_id(&table, "main"),
        Some(main_after),
        "the main ref tracks the rewrite"
    );
}
