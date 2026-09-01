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

use datafusion::arrow::array::{Array, AsArray, StringArray};
use futures::TryStreamExt;
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_COL_NAME_ROW_ID,
};
use iceberg::spec::DataFileFormat;
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};

use super::harness::{
    NS, TBL, commit_shared_puffin, delete_data_sequence, harness, live_data_files,
    live_delete_files, live_ids, load_table, row_positions, seed_two_file_shared_puffin, sql_count,
};

/// C-015 / T1 / T9: DELETE id=1 on a two-file shared Puffin must not resurrect id=5.
#[tokio::test]
async fn delete_of_one_file_must_not_resurrect_shared_puffin_sibling() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    assert_eq!(
        live_ids(&harness.ctx).await,
        vec![1, 3, 4, 6],
        "shared Puffin hides id 2 and id 5"
    );

    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    assert_eq!(deleted, 1, "exactly one row matches id = 1");

    let ids = live_ids(&harness.ctx).await;
    assert_eq!(
        ids,
        vec![3, 4, 6],
        "id 5 must stay deleted (shared-Puffin sibling). live={ids:?}"
    );

    let table = load_table(&harness.catalog).await;
    let deletes = live_delete_files(&table).await;
    assert!(
        deletes
            .iter()
            .all(|file| file.file_format() == DataFileFormat::Puffin),
        "V3 forbids new position-delete files"
    );
    let referenced: Vec<_> = deletes
        .iter()
        .filter_map(|file| file.referenced_data_file())
        .collect();
    assert_eq!(
        referenced.len(),
        2,
        "both blobs must stay live after the DELETE, got {referenced:?}"
    );
    let table_after = load_table(&harness.catalog).await;
    let data_files = live_data_files(&table_after).await;
    for delete in &deletes {
        let referenced = delete
            .referenced_data_file()
            .expect("DV names its data file");
        let data = data_files
            .iter()
            .find(|file| file.file_path() == referenced)
            .unwrap_or_else(|| panic!("live data file {referenced}"));
        assert_eq!(
            delete.partition_spec_id(),
            data.partition_spec_id(),
            "T10: replacement DV keeps the data file spec"
        );
        assert_eq!(
            delete.partition(),
            data.partition(),
            "T10: replacement DV keeps the data file partition"
        );
        assert!(
            delete.content_offset().is_some() && delete.content_size_in_bytes().is_some(),
            "T10: replacement DV has blob coordinates"
        );
    }
    let sibling = deletes
        .iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(b.as_str()))
        .expect("sibling DV for books");
    assert_eq!(
        sibling.record_count(),
        1,
        "T8: untouched sibling cardinality stays 1"
    );
}

/// T2: UPDATE one file in a shared Puffin. The sibling stays deleted; the updated row is live.
#[tokio::test]
async fn update_of_one_file_must_not_resurrect_shared_puffin_sibling() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    let table_before = load_table(&harness.catalog).await;
    let before = lineage_id_rows(&table_before).await;
    let id1_before = before.iter().find(|row| row.0 == 1).expect("id 1 before");

    let updated = sql_count(
        &harness.ctx,
        &format!("UPDATE catalog.{NS}.{TBL} SET data = 'z' WHERE id = 1"),
    )
    .await;
    assert_eq!(updated, 1, "exactly one row matches id = 1");

    let ids = live_ids(&harness.ctx).await;
    assert_eq!(
        ids,
        vec![1, 3, 4, 6],
        "id 5 must stay deleted after UPDATE. live={ids:?}"
    );
    let batches = harness
        .ctx
        .sql(&format!("SELECT data FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("select updated")
        .collect()
        .await
        .expect("collect updated");
    let data = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("data utf8")
        .value(0);
    assert_eq!(data, "z", "updated row must carry the new value");

    let table = load_table(&harness.catalog).await;
    let next = table.metadata().next_row_id();
    let rows = lineage_id_rows(&table).await;
    assert_eq!(rows.len(), 4, "every live row must still project lineage");
    let id1 = rows.iter().find(|row| row.0 == 1).expect("id 1");
    let id3 = rows.iter().find(|row| row.0 == 3).expect("id 3");
    let id3_before = before.iter().find(|row| row.0 == 3).expect("id 3 before");
    assert_eq!(id1.1, id1_before.1, "updated row keeps _row_id");
    assert!(
        id1.2 > id1_before.2,
        "updated row last_updated_seq must advance"
    );
    assert_eq!(id3.1, id3_before.1);
    assert_eq!(id3.2, id3_before.2);
    let deletes = live_delete_files(&table).await;
    assert!(
        deletes
            .iter()
            .any(|file| file.file_format() == DataFileFormat::Puffin),
        "sibling DV blobs must remain as a live Puffin"
    );
    assert!(
        next >= 6,
        "the six inserted rows assigned ids 0..5; next_row_id={next}"
    );
}

/// T12: a no-match DELETE must not write a Puffin or bump the snapshot.
#[tokio::test]
async fn no_match_delete_is_a_snapshot_noop() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    let before = load_table(&harness.catalog).await;
    let snapshot_before = before
        .metadata()
        .current_snapshot_id()
        .expect("snapshot before");
    let deletes_before = live_delete_files(&before).await.len();

    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = -1"),
    )
    .await;
    assert_eq!(deleted, 0);

    let after = load_table(&harness.catalog).await;
    assert_eq!(
        after
            .metadata()
            .current_snapshot_id()
            .expect("snapshot after"),
        snapshot_before
    );
    assert_eq!(live_delete_files(&after).await.len(), deletes_before);
    assert_eq!(live_ids(&harness.ctx).await, vec![1, 3, 4, 6]);
}

/// T4: two Puffins; touching one container leaves the other Puffin path unchanged.
#[tokio::test]
async fn delete_in_one_puffin_does_not_rewrite_the_other() {
    let harness = harness().await;
    super::harness::run_sql(
        &harness.ctx,
        &format!(
            "INSERT INTO catalog.{NS}.{TBL} VALUES \
             (1, 'a', 'electronics'), (2, 'b', 'electronics'), (3, 'c', 'electronics'), \
             (4, 'd', 'books'), (5, 'e', 'books'), (6, 'f', 'books')"
        ),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    let rows = row_positions(&table).await;
    let two = rows.iter().find(|row| row.id == 2).expect("id 2");
    let five = rows.iter().find(|row| row.id == 5).expect("id 5");
    commit_shared_puffin(&harness.catalog, &table, &[(
        two.file.clone(),
        u64::try_from(two.pos).expect("pos 2"),
    )])
    .await;
    let table = load_table(&harness.catalog).await;
    commit_shared_puffin(&harness.catalog, &table, &[(
        five.file.clone(),
        u64::try_from(five.pos).expect("pos 5"),
    )])
    .await;
    let before = load_table(&harness.catalog).await;
    let books_puffin = live_delete_files(&before)
        .await
        .into_iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(five.file.as_str()))
        .expect("books DV")
        .file_path()
        .to_string();
    sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    assert_eq!(live_ids(&harness.ctx).await, vec![3, 4, 6]);
    let after = load_table(&harness.catalog).await;
    let books_after = live_delete_files(&after)
        .await
        .into_iter()
        .find(|file| file.referenced_data_file().as_deref() == Some(five.file.as_str()))
        .expect("books DV after");
    assert_eq!(
        books_after.file_path(),
        books_puffin,
        "untouched Puffin path must not be rewritten"
    );
}

/// T3: touching both referenced files still leaves one live DV per file.
#[tokio::test]
async fn delete_touching_both_files_keeps_one_dv_each() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1 OR id = 4"),
    )
    .await;
    assert_eq!(deleted, 2);
    assert_eq!(live_ids(&harness.ctx).await, vec![3, 6]);
    let table = load_table(&harness.catalog).await;
    let deletes = live_delete_files(&table).await;
    assert_eq!(deletes.len(), 2, "one live DV per referenced file");
}

/// T13: an untouched sibling keeps its original data sequence.
#[tokio::test]
async fn untouched_sibling_keeps_original_data_sequence() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    let before = load_table(&harness.catalog).await;
    let seq_before = delete_data_sequence(&before, &b).await;
    sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    let after = load_table(&harness.catalog).await;
    let seq_after = delete_data_sequence(&after, &b).await;
    assert_eq!(
        seq_after, seq_before,
        "sibling data sequence must not inherit the DELETE snapshot"
    );
}

/// T17: concurrent DeleteFiles of untouched sibling B rejects the frozen DELETE.
#[tokio::test]
async fn delete_rejects_concurrent_delete_of_untouched_sibling() {
    let harness = harness().await;
    let (_a, b) = seed_two_file_shared_puffin(&harness).await;
    let plan = harness
        .ctx
        .sql(&format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("logical")
        .create_physical_plan()
        .await
        .expect("frozen physical plan");

    let table = load_table(&harness.catalog).await;
    let tx = Transaction::new(&table);
    tx.delete_files()
        .delete_files([b.clone()])
        .apply(tx)
        .expect("apply delete_files")
        .commit(harness.catalog.as_ref())
        .await
        .expect("concurrent DeleteFiles of sibling B");

    let err = datafusion::physical_plan::collect(plan, harness.ctx.task_ctx())
        .await
        .expect_err("DELETE must reject concurrent Delete of sibling B");
    let message = err.to_string();
    assert!(
        message.contains("missing data files") || message.contains("conflicting delete"),
        "expected files-exist rejection of sibling B, got {message}"
    );
}

/// T18: concurrent DeleteFiles of touched file A also rejects the frozen DELETE.
#[tokio::test]
async fn delete_rejects_concurrent_delete_of_touched_file() {
    let harness = harness().await;
    let (a, _b) = seed_two_file_shared_puffin(&harness).await;
    let plan = harness
        .ctx
        .sql(&format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("logical")
        .create_physical_plan()
        .await
        .expect("frozen physical plan");
    let table = load_table(&harness.catalog).await;
    let tx = Transaction::new(&table);
    tx.delete_files()
        .delete_files([a.clone()])
        .apply(tx)
        .expect("apply delete_files")
        .commit(harness.catalog.as_ref())
        .await
        .expect("concurrent DeleteFiles of A");
    let err = datafusion::physical_plan::collect(plan, harness.ctx.task_ctx())
        .await
        .expect_err("DELETE must reject concurrent Delete of touched A");
    let message = err.to_string();
    assert!(
        message.contains("missing data files") || message.contains("conflicting delete"),
        "expected files-exist rejection of A, got {message}"
    );
}

/// T23: concurrent DeleteFiles of an unrelated file is outside the replacement set.
#[tokio::test]
async fn delete_allows_concurrent_delete_of_unrelated_file() {
    let harness = harness().await;
    seed_two_file_shared_puffin(&harness).await;
    super::harness::run_sql(
        &harness.ctx,
        &format!("INSERT INTO catalog.{NS}.{TBL} VALUES (7, 'g', 'toys')"),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    let toys = live_data_files(&table)
        .await
        .into_iter()
        .find(|file| {
            !file.file_path().contains("electronics") && !file.file_path().contains("books")
        })
        .expect("toys data file")
        .file_path()
        .to_string();
    let plan = harness
        .ctx
        .sql(&format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"))
        .await
        .expect("logical")
        .create_physical_plan()
        .await
        .expect("frozen physical plan");
    let table = load_table(&harness.catalog).await;
    let tx = Transaction::new(&table);
    tx.delete_files()
        .delete_files([toys])
        .apply(tx)
        .expect("apply delete_files")
        .commit(harness.catalog.as_ref())
        .await
        .expect("concurrent DeleteFiles of unrelated C");
    datafusion::physical_plan::collect(plan, harness.ctx.task_ctx())
        .await
        .expect("DELETE must succeed when only unrelated C was removed");
    assert_eq!(live_ids(&harness.ctx).await, vec![3, 4, 6]);
}

async fn lineage_id_rows(table: &Table) -> Vec<(i32, i64, i64)> {
    let batches: Vec<_> = table
        .scan()
        .select([
            "id",
            RESERVED_COL_NAME_ROW_ID,
            RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER,
        ])
        .build()
        .expect("lineage scan")
        .to_arrow()
        .await
        .expect("to_arrow")
        .try_collect()
        .await
        .expect("collect lineage");
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column_by_name("id")
            .expect("id")
            .as_primitive::<datafusion::arrow::datatypes::Int32Type>();
        let row_ids = batch
            .column_by_name(RESERVED_COL_NAME_ROW_ID)
            .expect("_row_id")
            .as_primitive::<datafusion::arrow::datatypes::Int64Type>();
        let seqs = batch
            .column_by_name(RESERVED_COL_NAME_LAST_UPDATED_SEQUENCE_NUMBER)
            .expect("seq")
            .as_primitive::<datafusion::arrow::datatypes::Int64Type>();
        for index in 0..batch.num_rows() {
            assert!(row_ids.is_valid(index));
            assert!(seqs.is_valid(index));
            rows.push((ids.value(index), row_ids.value(index), seqs.value(index)));
        }
    }
    rows.sort_unstable();
    rows
}
