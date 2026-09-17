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
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, create_partitioned_table, current_snapshot_id,
    live_delete_file_paths, local_fs_catalog, scan_rows, write_data_file,
    write_position_delete_file,
};
use crate::maintenance::rewrite_data_files::{RewriteDataFiles, RewriteDataFilesResult};
use crate::spec::DataFile;
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};

/// Drops `removed` data files in one `RewriteFiles` commit.
async fn remove_data_files(catalog: &impl Catalog, table: &Table, removed: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    // A delete-only rewrite adds nothing.
    let action = tx.rewrite_files(removed, Vec::new());
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

/// A fixture whose lone position delete genuinely dangles after compaction.
///
/// Everything sits in partition `x = 0`, so the table is one bin-pack group. Sequence 1 appends
/// five files, sequence 2 adds a position delete, and sequence 3 appends a sixth. The rewrite
/// starts from sequence 3, so the restamped data lifts the partition minimum to 3 and the
/// delete at 2 falls under Java's strict `<` dangling clause.
async fn dangling_after_compaction_fixture(catalog: &impl Catalog) -> (Table, String) {
    let table = create_partitioned_table(catalog, crate::spec::FormatVersion::V2).await;

    let mut files = Vec::new();
    let two_row =
        write_data_file(&table, "two-row.parquet", 0, &[(0, 11, 110), (0, 22, 220)]).await;
    let two_row_path = two_row.file_path().to_string();
    files.push(two_row);
    for index in 0..4i64 {
        files.push(
            write_data_file(&table, &format!("one-{index}.parquet"), 0, &[(
                0,
                30 + index,
                300,
            )])
            .await,
        );
    }
    let table = append_files(catalog, &table, files).await;

    let pos_delete = write_position_delete_file(&table, 0, &[(two_row_path, 0)]).await;
    let pos_delete_path = pos_delete.file_path().to_string();
    let table = add_deletes(catalog, &table, vec![pos_delete]).await;

    // This bump is what makes the delete dangle once the data is restamped.
    let later = write_data_file(&table, "later.parquet", 0, &[(0, 99, 990)]).await;
    let table = append_files(catalog, &table, vec![later]).await;

    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([pos_delete_path.clone()]),
        "fixture: exactly one live delete file before compaction"
    );
    (table, pos_delete_path)
}

/// The flag defaults off, so no caller gets a delete-file GC pass it did not ask for. On a
/// genuinely dangling fixture the count stays 0, the delete file stays live, and exactly one
/// snapshot is added.
#[tokio::test]
async fn test_remove_dangling_deletes_defaults_off() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, pos_delete_path) = dangling_after_compaction_fixture(&catalog).await;

    let rows_before = scan_rows(&table).await;
    let snapshots_before = table.metadata().snapshots().count();

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .execute(&catalog)
        .await
        .expect("compaction must succeed");

    assert_eq!(
        result.rewritten_data_files_count, 6,
        "fixture: all 6 files formed one group and were rewritten"
    );
    assert_eq!(
        result.removed_delete_files_count, 0,
        "the sub-action did not run, so nothing was removed"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([pos_delete_path]),
        "the dangling delete file survives (population: the table's 1 delete file)"
    );
    assert_eq!(
        table.metadata().snapshots().count(),
        snapshots_before + 1,
        "exactly one new snapshot — the lone group's rewrite commit, no GC commit \
         (population: 1 partition ⇒ 1 group ⇒ 1 commit)"
    );
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

/// The flag must compose something, not just be accepted. With it set, the count is 1, the
/// delete file is gone, a second snapshot lands, and the rows read identically.
#[tokio::test]
async fn test_remove_dangling_deletes_on_removes_the_dangling_delete() {
    let (catalog, _temp) = local_fs_catalog().await;
    let (table, pos_delete_path) = dangling_after_compaction_fixture(&catalog).await;

    let rows_before = scan_rows(&table).await;
    let snapshots_before = table.metadata().snapshots().count();

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .remove_dangling_deletes(true)
        .execute(&catalog)
        .await
        .expect("compaction + dangling removal must succeed");

    assert_eq!(
        result.rewritten_data_files_count, 6,
        "fixture: all 6 files formed one group and were rewritten"
    );
    assert_eq!(
        result.removed_delete_files_count, 1,
        "the one dangling delete file was removed (population: the table's 1 delete file)"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert!(
        live_delete_file_paths(&table).await.is_empty(),
        "no delete file is live any more; the removed one was {pos_delete_path}"
    );
    assert_eq!(
        table.metadata().snapshots().count(),
        snapshots_before + 2,
        "two new snapshots: the group's rewrite commit, then the GC commit"
    );
    assert_eq!(
        scan_rows(&table).await,
        rows_before,
        "row conservation: dangling-delete GC never changes the read result"
    );
}

/// The flag must not force an empty extra snapshot, nor remove a delete Java keeps. Without the
/// sequence bump the data restamps to the delete's own number, so Java's strict `<` clause does
/// not fire even though the referenced data file is gone. The sub-action finds and commits
/// nothing.
#[tokio::test]
async fn test_remove_dangling_deletes_on_with_nothing_dangling_commits_no_snapshot() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

    let mut files = Vec::new();
    let two_row =
        write_data_file(&table, "two-row.parquet", 0, &[(0, 11, 110), (0, 22, 220)]).await;
    let two_row_path = two_row.file_path().to_string();
    files.push(two_row);
    for index in 0..4i64 {
        files.push(
            write_data_file(&table, &format!("one-{index}.parquet"), 0, &[(
                0,
                30 + index,
                300,
            )])
            .await,
        );
    }
    let table = append_files(&catalog, &table, files).await;
    let pos_delete = write_position_delete_file(&table, 0, &[(two_row_path, 0)]).await;
    let pos_delete_path = pos_delete.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![pos_delete]).await;

    let rows_before = scan_rows(&table).await;
    let snapshots_before = table.metadata().snapshots().count();

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(1_000_000)
        .remove_dangling_deletes(true)
        .execute(&catalog)
        .await
        .expect("compaction must succeed");

    assert_eq!(
        result.rewritten_data_files_count, 5,
        "fixture: all 5 files formed one group and were rewritten"
    );
    assert_eq!(
        result.removed_delete_files_count, 0,
        "nothing dangled by Java's predicate (population: the table's 1 delete file)"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([pos_delete_path]),
        "the same-sequence delete is KEPT — Java's position clause is STRICT `<`"
    );
    assert_eq!(
        table.metadata().snapshots().count(),
        snapshots_before + 1,
        "exactly one new snapshot: the group's rewrite commit. The sub-action ran and found \
         nothing, and an empty dangling set commits NOTHING (Java commits only when the set is \
         non-empty) — so there is no empty GC snapshot"
    );
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}

/// An empty plan must not run the GC pass, because Java returns its empty result first. The
/// table carries a genuinely dangling delete, so "nothing ran" is observable: a non-empty plan
/// would remove that same delete.
#[tokio::test]
async fn test_empty_plan_skips_the_dangling_step_entirely() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

    // Partition x=0: one well-sized file that is not a rewrite candidate.
    let rows: Vec<(i64, i64, i64)> = (0..100).map(|n| (0, n, n)).collect();
    let well_sized = write_data_file(&table, "ok.parquet", 0, &rows).await;
    let well_sized_size = well_sized.file_size_in_bytes();
    // Partition x=1: a small file that will be dropped, orphaning its position delete.
    let doomed = write_data_file(&table, "doomed.parquet", 1, &[(1, 5, 50), (1, 6, 60)]).await;
    let doomed_path = doomed.file_path().to_string();
    let table = append_files(&catalog, &table, vec![well_sized, doomed.clone()]).await;

    let pos_delete = write_position_delete_file(&table, 1, &[(doomed_path, 0)]).await;
    let pos_delete_path = pos_delete.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![pos_delete]).await;
    let table = remove_data_files(&catalog, &table, vec![doomed]).await;

    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([pos_delete_path.clone()]),
        "fixture: the delete file is live and its partition now has NO live data"
    );

    let rows_before = scan_rows(&table).await;
    let snapshots_before = table.metadata().snapshots().count();
    let snapshot_id_before = current_snapshot_id(&table);

    let result = RewriteDataFiles::new(table.clone())
        .target_file_size_bytes(well_sized_size)
        .min_file_size_bytes(well_sized_size / 2)
        .max_file_size_bytes(well_sized_size * 2)
        .remove_dangling_deletes(true)
        .execute(&catalog)
        .await
        .expect("execute must succeed (no-op)");

    assert_eq!(
        result,
        RewriteDataFilesResult::default(),
        "an empty plan returns a zero-count result even with the flag on"
    );

    let table = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_delete_file_paths(&table).await,
        HashSet::from([pos_delete_path]),
        "the dangling delete is UNTOUCHED — the sub-action never ran (population: the \
         table's 1 delete file)"
    );
    assert_eq!(
        table.metadata().snapshots().count(),
        snapshots_before,
        "no snapshot at all was committed"
    );
    assert_eq!(
        current_snapshot_id(&table),
        snapshot_id_before,
        "the current snapshot is unchanged"
    );
    assert_eq!(scan_rows(&table).await, rows_before, "row conservation");
}
