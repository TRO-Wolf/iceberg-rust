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
use std::sync::atomic::{AtomicU32, Ordering};

use crate::catalog::MockCatalog;
use crate::expr::{Predicate, Reference};
use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, Struct,
};
use crate::table::Table;
use crate::transaction::tests::make_v2_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, Error, ErrorKind};

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

fn data_file_with_y_bounds(path: &str, part_value: i64, y_lower: i64, y_upper: i64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(part_value))]))
        .lower_bounds(HashMap::from([(2, Datum::long(y_lower))]))
        .upper_bounds(HashMap::from([(2, Datum::long(y_upper))]))
        .build()
        .unwrap()
}

fn delete_file(path: &str, part_value: i64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(part_value))]))
        .build()
        .unwrap()
}

fn x_equals(value: i64) -> Predicate {
    Reference::new("x").equal_to(Datum::long(value))
}

async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

async fn commit_row_delta_deletes(
    catalog: &impl Catalog,
    table: &Table,
    files: Vec<DataFile>,
) -> Table {
    let tx = Transaction::new(table);
    let action = tx.row_delta().add_deletes(files);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

async fn live_file_paths(table: &Table) -> HashSet<String> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table should have a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list should load");
    let mut live = HashSet::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest should load");
        for entry in manifest.entries() {
            if entry.is_alive() {
                live.insert(entry.file_path().to_string());
            }
        }
    }
    live
}

fn snapshot_len(table: &Table) -> usize {
    table.metadata().snapshots().len()
}

#[tokio::test]
async fn row_delta_serializable_disjoint_partition_append_commits() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/base.parquet", 1)]).await;
    let base_snapshots = snapshot_len(&table);

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 1)])
        .conflict_detection_filter(x_equals(1))
        .validate_no_conflicting_data_files();
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![data_file(
        "test/other-part.parquet",
        0,
    )])
    .await;

    let table = tx.commit(&catalog).await.expect(
        "a concurrent append into a disjoint partition must not conflict under filter x = 1",
    );
    let live = live_file_paths(&table).await;
    assert!(live.contains("test/op.parquet"));
    assert!(live.contains("test/other-part.parquet"));
    assert!(live.contains("test/base.parquet"));
    assert_eq!(snapshot_len(&table), base_snapshots + 2);
}

#[tokio::test]
async fn row_delta_serializable_matching_partition_append_conflicts() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/base.parquet", 1)]).await;

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 1)])
        .conflict_detection_filter(x_equals(1))
        .validate_no_conflicting_data_files();
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![data_file(
        "test/same-part.parquet",
        1,
    )])
    .await;

    let err = tx.commit(&catalog).await.expect_err(
        "a concurrent append into the filtered partition must conflict under filter x = 1",
    );
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(!err.retryable());
    assert!(err.message().contains("conflicting files"));
    assert!(err.message().contains("test/same-part.parquet"));

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let live = live_file_paths(&reloaded).await;
    assert!(live.contains("test/same-part.parquet"));
    assert!(!live.contains("test/op.parquet"));
}

#[tokio::test]
async fn row_delta_serializable_metrics_excluded_file_in_matching_partition_commits() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/base.parquet", 1)]).await;

    let filter = Reference::new("x")
        .equal_to(Datum::long(1))
        .and(Reference::new("y").greater_than_or_equal_to(Datum::long(50)));
    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 1)])
        .conflict_detection_filter(filter)
        .validate_no_conflicting_data_files();
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![data_file_with_y_bounds(
        "test/low-y.parquet",
        1,
        10,
        20,
    )])
    .await;

    let table = tx
        .commit(&catalog)
        .await
        .expect("a same-partition file whose metrics exclude the filter must not conflict");
    let live = live_file_paths(&table).await;
    assert!(live.contains("test/op.parquet"));
    assert!(live.contains("test/low-y.parquet"));
}

#[tokio::test]
async fn row_delta_referenced_file_replaced_by_rewrite_conflicts() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let victim = data_file("test/victim.parquet", 1);
    let table = append_files(&catalog, &table, vec![
        victim.clone(),
        data_file("test/other.parquet", 0),
    ])
    .await;

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 1)])
        .validate_data_files_exist(["test/victim.parquet"]);
    let tx = action.apply(tx).unwrap();

    let rewrite_tx = Transaction::new(&table);
    let rewrite_action =
        rewrite_tx.rewrite_files(vec![victim], vec![data_file("test/victim-r.parquet", 1)]);
    let rewrite_tx = rewrite_action.apply(rewrite_tx).unwrap();
    let _concurrent = rewrite_tx.commit(&catalog).await.unwrap();

    let err = tx
        .commit(&catalog)
        .await
        .expect_err("a rewrite replacing the referenced file must conflict the files-exist check");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(!err.retryable());
}

#[tokio::test]
async fn row_delta_rewrite_of_unrelated_file_commits() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![
        data_file("test/victim.parquet", 1),
        data_file("test/other.parquet", 0),
    ])
    .await;

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 1)])
        .validate_data_files_exist(["test/victim.parquet"]);
    let tx = action.apply(tx).unwrap();

    let rewrite_tx = Transaction::new(&table);
    let rewrite_action =
        rewrite_tx.rewrite_files(vec![data_file("test/other.parquet", 0)], vec![data_file(
            "test/other-r.parquet",
            0,
        )]);
    let rewrite_tx = rewrite_action.apply(rewrite_tx).unwrap();
    let _concurrent = rewrite_tx.commit(&catalog).await.unwrap();

    let table = tx
        .commit(&catalog)
        .await
        .expect("a rewrite of an unrelated file must not conflict the files-exist check");
    let live = live_file_paths(&table).await;
    assert!(live.contains("test/op.parquet"));
    assert!(live.contains("test/victim.parquet"));
    assert!(live.contains("test/other-r.parquet"));
    assert!(!live.contains("test/other.parquet"));
}

#[tokio::test]
async fn row_delta_serializable_matching_delete_file_conflicts() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/base.parquet", 1)]).await;

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_deletes(vec![delete_file("test/my-del.parquet", 1)])
        .conflict_detection_filter(x_equals(1))
        .validate_no_conflicting_delete_files();
    let tx = action.apply(tx).unwrap();

    let _concurrent = commit_row_delta_deletes(&catalog, &table, vec![delete_file(
        "test/concurrent-del.parquet",
        1,
    )])
    .await;

    let err = tx.commit(&catalog).await.expect_err(
        "a concurrent delete file in the filtered partition must conflict under filter x = 1",
    );
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(!err.retryable());
    assert!(err.message().contains("conflicting delete files"));
    assert!(err.message().contains("test/concurrent-del.parquet"));
}

#[tokio::test]
async fn row_delta_serializable_nonmatching_delete_file_commits() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/base.parquet", 1)]).await;
    let base_snapshots = snapshot_len(&table);

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_deletes(vec![delete_file("test/my-del.parquet", 1)])
        .conflict_detection_filter(x_equals(1))
        .validate_no_conflicting_delete_files();
    let tx = action.apply(tx).unwrap();

    let _concurrent = commit_row_delta_deletes(&catalog, &table, vec![delete_file(
        "test/concurrent-del.parquet",
        0,
    )])
    .await;

    let table = tx.commit(&catalog).await.expect(
        "a concurrent delete file in a disjoint partition must not conflict under filter x = 1",
    );
    assert_eq!(snapshot_len(&table), base_snapshots + 2);
}

#[tokio::test]
async fn row_delta_snapshot_isolation_rebases_over_matching_append() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/base.parquet", 1)]).await;

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![data_file("test/op.parquet", 1)])
        .conflict_detection_filter(x_equals(1));
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![data_file(
        "test/same-part.parquet",
        1,
    )])
    .await;

    let table = tx.commit(&catalog).await.expect(
        "without the conflicting-data flag the row delta keeps snapshot isolation and rebases",
    );
    let live = live_file_paths(&table).await;
    assert!(live.contains("test/op.parquet"));
    assert!(live.contains("test/same-part.parquet"));
}

#[tokio::test]
async fn overwrite_serializable_disjoint_partition_append_commits() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![
        data_file("test/a.parquet", 1),
        data_file("test/b.parquet", 0),
    ])
    .await;

    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .delete_file("test/a.parquet")
        .add_file(data_file("test/c.parquet", 1))
        .conflict_detection_filter(x_equals(1))
        .validate_no_conflicting_data();
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![data_file(
        "test/other-part.parquet",
        0,
    )])
    .await;

    let table = tx.commit(&catalog).await.expect(
        "a concurrent append into a disjoint partition must not conflict under filter x = 1",
    );
    let live = live_file_paths(&table).await;
    assert!(live.contains("test/c.parquet"));
    assert!(live.contains("test/other-part.parquet"));
    assert!(!live.contains("test/a.parquet"));
}

#[tokio::test]
async fn overwrite_serializable_matching_partition_append_conflicts() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 1)]).await;

    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .delete_file("test/a.parquet")
        .add_file(data_file("test/c.parquet", 1))
        .conflict_detection_filter(x_equals(1))
        .validate_no_conflicting_data();
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![data_file(
        "test/same-part.parquet",
        1,
    )])
    .await;

    let err = tx.commit(&catalog).await.expect_err(
        "a concurrent append into the filtered partition must conflict under filter x = 1",
    );
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(!err.retryable());
    assert!(err.message().contains("conflicting files"));
    assert!(err.message().contains("test/same-part.parquet"));
}

#[tokio::test]
async fn overwrite_snapshot_isolation_rebases_over_matching_append() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 1)]).await;

    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .delete_file("test/a.parquet")
        .add_file(data_file("test/c.parquet", 1))
        .conflict_detection_filter(x_equals(1));
    let tx = action.apply(tx).unwrap();

    let _concurrent = append_files(&catalog, &table, vec![data_file(
        "test/same-part.parquet",
        1,
    )])
    .await;

    let table = tx.commit(&catalog).await.expect(
        "without the conflicting-data flag the overwrite keeps snapshot isolation and rebases",
    );
    let live = live_file_paths(&table).await;
    assert!(live.contains("test/c.parquet"));
    assert!(live.contains("test/same-part.parquet"));
    assert!(!live.contains("test/a.parquet"));
}

#[tokio::test]
async fn overwrite_row_filter_rewrite_of_matching_file_conflicts() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![
        data_file("test/a.parquet", 1),
        data_file("test/b.parquet", 0),
    ])
    .await;

    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(x_equals(1))
        .add_file(data_file("test/c.parquet", 1))
        .validate_added_files_match_overwrite_filter()
        .validate_no_conflicting_deletes();
    let tx = action.apply(tx).unwrap();

    let rewrite_tx = Transaction::new(&table);
    let rewrite_action =
        rewrite_tx.rewrite_files(vec![data_file("test/a.parquet", 1)], vec![data_file(
            "test/a-r.parquet",
            1,
        )]);
    let rewrite_tx = rewrite_action.apply(rewrite_tx).unwrap();
    let _concurrent = rewrite_tx.commit(&catalog).await.unwrap();

    let err = tx
        .commit(&catalog)
        .await
        .expect_err("a rewrite replacing the row-filtered file must conflict the deletes check");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(!err.retryable());
}

#[tokio::test]
async fn overwrite_row_filter_rewrite_of_unrelated_file_commits() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![
        data_file("test/a.parquet", 1),
        data_file("test/b.parquet", 0),
    ])
    .await;

    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .overwrite_by_row_filter(x_equals(1))
        .add_file(data_file("test/c.parquet", 1))
        .validate_added_files_match_overwrite_filter()
        .validate_no_conflicting_deletes();
    let tx = action.apply(tx).unwrap();

    let rewrite_tx = Transaction::new(&table);
    let rewrite_action =
        rewrite_tx.rewrite_files(vec![data_file("test/b.parquet", 0)], vec![data_file(
            "test/b-r.parquet",
            0,
        )]);
    let rewrite_tx = rewrite_action.apply(rewrite_tx).unwrap();
    let _concurrent = rewrite_tx.commit(&catalog).await.unwrap();

    let table = tx.commit(&catalog).await.expect(
        "a rewrite replacing a file outside the row filter must not conflict the deletes check",
    );
    let live = live_file_paths(&table).await;
    assert!(live.contains("test/c.parquet"));
    assert!(live.contains("test/b-r.parquet"));
    assert!(!live.contains("test/a.parquet"));
    assert!(!live.contains("test/b.parquet"));
}

#[tokio::test]
async fn fast_append_racing_append_rebases_and_commits() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/base.parquet", 0)]).await;
    let base_snapshots = snapshot_len(&table);

    let first_tx = Transaction::new(&table);
    let first_action = first_tx
        .fast_append()
        .add_data_files(vec![data_file("test/first.parquet", 0)]);
    let first_tx = first_action.apply(first_tx).unwrap();

    let second_tx = Transaction::new(&table);
    let second_action = second_tx
        .fast_append()
        .add_data_files(vec![data_file("test/second.parquet", 0)]);
    let second_tx = second_action.apply(second_tx).unwrap();

    let _first = first_tx.commit(&catalog).await.unwrap();
    let table = second_tx
        .commit(&catalog)
        .await
        .expect("a plain append racing an append must rebase and commit, not conflict");
    let live = live_file_paths(&table).await;
    assert!(live.contains("test/base.parquet"));
    assert!(live.contains("test/first.parquet"));
    assert!(live.contains("test/second.parquet"));
    assert_eq!(snapshot_len(&table), base_snapshots + 2);
}

#[tokio::test]
async fn unknown_outcome_data_commit_surfaces_unretried() {
    let memory_catalog = Arc::new(new_memory_catalog().await);
    let table = make_v2_minimal_table_in_catalog(memory_catalog.as_ref()).await;

    let fast_status_check = Transaction::new(&table);
    let fast_status_check = fast_status_check
        .update_table_properties()
        .set("commit.retry.num-retries".to_string(), "2".to_string())
        .set("commit.retry.min-wait-ms".to_string(), "1".to_string())
        .set("commit.retry.max-wait-ms".to_string(), "5".to_string())
        .set(
            "commit.status-check.num-retries".to_string(),
            "0".to_string(),
        )
        .set(
            "commit.status-check.min-wait-ms".to_string(),
            "1".to_string(),
        )
        .set(
            "commit.status-check.max-wait-ms".to_string(),
            "5".to_string(),
        )
        .set(
            "commit.status-check.total-timeout-ms".to_string(),
            "1000".to_string(),
        )
        .apply(fast_status_check)
        .expect("stage fast status-check knobs");
    let table = fast_status_check
        .commit(memory_catalog.as_ref())
        .await
        .expect("stage status-check knobs");

    let mut mock_catalog = MockCatalog::new();
    let load_table = table.clone();
    mock_catalog.expect_load_table().returning_st(move |_| {
        let table = load_table.clone();
        Box::pin(async move { Ok(table) })
    });
    let update_calls = Arc::new(AtomicU32::new(0));
    let update_calls_in_mock = Arc::clone(&update_calls);
    mock_catalog
        .expect_update_table()
        .times(1)
        .returning_st(move |_| {
            let calls = Arc::clone(&update_calls_in_mock);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(Error::new(
                    ErrorKind::CommitStateUnknown,
                    "connection reset after the update request was sent",
                )
                .with_retryable(true))
            })
        });

    let tx = Transaction::new(&table);
    let action = tx
        .fast_append()
        .add_data_files(vec![data_file("test/unknown.parquet", 0)]);
    let tx = action.apply(tx).unwrap();
    let err = tx
        .commit(&mock_catalog)
        .await
        .expect_err("an unknown-outcome commit must surface, never retry");
    assert_eq!(err.kind(), ErrorKind::CommitStateUnknown);
    assert!(err.message().contains("connection reset"));
    assert_eq!(
        update_calls.load(Ordering::SeqCst),
        1,
        "exactly one update_table attempt: the kind gate stops the retry even flagged retryable"
    );
}

#[tokio::test]
async fn fast_append_conflicted_first_attempt_retries_and_commits() {
    let memory_catalog = Arc::new(new_memory_catalog().await);
    let table = make_v2_minimal_table_in_catalog(memory_catalog.as_ref()).await;

    let fast_retry = Transaction::new(&table);
    let fast_retry = fast_retry
        .update_table_properties()
        .set("commit.retry.num-retries".to_string(), "2".to_string())
        .set("commit.retry.min-wait-ms".to_string(), "1".to_string())
        .set("commit.retry.max-wait-ms".to_string(), "5".to_string())
        .apply(fast_retry)
        .expect("stage fast retry knobs");
    let table = fast_retry
        .commit(memory_catalog.as_ref())
        .await
        .expect("stage retry knobs");
    let base_snapshots = snapshot_len(&table);

    let mut mock_catalog = MockCatalog::new();
    let load_table = table.clone();
    mock_catalog.expect_load_table().returning_st(move |_| {
        let table = load_table.clone();
        Box::pin(async move { Ok(table) })
    });
    let update_calls = Arc::new(AtomicU32::new(0));
    let update_calls_in_mock = Arc::clone(&update_calls);
    let update_delegate = Arc::clone(&memory_catalog);
    mock_catalog
        .expect_update_table()
        .returning_st(move |commit| {
            let catalog = Arc::clone(&update_delegate);
            let calls = Arc::clone(&update_calls_in_mock);
            Box::pin(async move {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(Error::new(
                        ErrorKind::CatalogCommitConflicts,
                        "injected concurrent commit conflict on first attempt",
                    )
                    .with_retryable(true));
                }
                catalog.update_table(commit).await
            })
        });

    let tx = Transaction::new(&table);
    let action = tx
        .fast_append()
        .add_data_files(vec![data_file("test/retried.parquet", 0)]);
    let tx = action.apply(tx).unwrap();
    let table = tx
        .commit(&mock_catalog)
        .await
        .expect("a retryable first-attempt conflict must retry and commit");
    assert_eq!(
        update_calls.load(Ordering::SeqCst),
        2,
        "exactly two update_table attempts: one conflicted, one committed"
    );
    assert!(
        live_file_paths(&table)
            .await
            .contains("test/retried.parquet")
    );
    assert_eq!(snapshot_len(&table), base_snapshots + 1);
}
