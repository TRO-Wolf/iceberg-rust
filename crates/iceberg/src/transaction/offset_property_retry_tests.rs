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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use super::occ_scoped_tests::{data_file, live_file_paths, snapshot_len};
use crate::catalog::MockCatalog;
use crate::memory::tests::new_memory_catalog;
use crate::spec::Operation;
use crate::table::Table;
use crate::transaction::tests::make_v2_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, TableIdent};

const OFFSET_KEY: &str = "streaming.offsets";
const EPOCH_KEY: &str = "streaming.batch-epoch";

fn summary_extras() -> HashMap<String, String> {
    HashMap::from([
        (EPOCH_KEY.to_string(), "7".to_string()),
        (OFFSET_KEY.to_string(), "src=42".to_string()),
    ])
}

async fn seeded_table(catalog: &impl Catalog) -> Table {
    let table = make_v2_minimal_table_in_catalog(catalog).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .update_table_properties()
        .set("commit.retry.num-retries".to_string(), "2".to_string())
        .set("commit.retry.min-wait-ms".to_string(), "1".to_string())
        .set("commit.retry.max-wait-ms".to_string(), "5".to_string())
        .apply(tx)
        .unwrap();
    let table = tx.commit(catalog).await.unwrap();
    let tx = Transaction::new(&table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![data_file("test/base.parquet", 0)])
        .apply(tx)
        .unwrap();
    tx.commit(catalog).await.unwrap()
}

#[derive(Clone, Copy)]
enum Writer {
    RowDelta,
    Overwrite,
}

fn offset_commit(table: &Table, writer: Writer) -> Transaction {
    let tx = Transaction::new(table);
    let tx = match writer {
        Writer::RowDelta => tx
            .row_delta()
            .add_data_files(vec![data_file("test/batch.parquet", 0)])
            .set_snapshot_properties(summary_extras())
            .apply(tx)
            .unwrap(),
        Writer::Overwrite => tx
            .overwrite_files()
            .delete_file("test/base.parquet")
            .add_file(data_file("test/batch.parquet", 0))
            .set_snapshot_properties(summary_extras())
            .apply(tx)
            .unwrap(),
    };
    tx.update_table_properties()
        .set(OFFSET_KEY.to_string(), "src=42".to_string())
        .apply(tx)
        .unwrap()
}

fn snapshots_carrying_epoch(table: &Table) -> usize {
    table
        .metadata()
        .snapshots()
        .filter(|snapshot| {
            snapshot
                .summary()
                .additional_properties
                .contains_key(EPOCH_KEY)
        })
        .count()
}

fn assert_offset_commit(table: &Table) {
    let current = table.metadata().current_snapshot().unwrap();
    assert_eq!(current.summary().operation, Operation::Overwrite);
    let extras = &current.summary().additional_properties;
    assert_eq!(extras.get(EPOCH_KEY).map(String::as_str), Some("7"));
    assert_eq!(extras.get(OFFSET_KEY).map(String::as_str), Some("src=42"));
    assert_eq!(snapshots_carrying_epoch(table), 1);
    assert_eq!(
        table
            .metadata()
            .properties()
            .get(OFFSET_KEY)
            .map(String::as_str),
        Some("src=42")
    );
}

async fn combined_commit_is_one_metadata_version(writer: Writer) {
    let catalog = new_memory_catalog().await;
    let base = seeded_table(&catalog).await;
    let base_log = base.metadata().metadata_log().len();
    let base_snapshots = snapshot_len(&base);

    let committed = offset_commit(&base, writer).commit(&catalog).await.unwrap();

    assert_eq!(committed.metadata().metadata_log().len(), base_log + 1);
    assert_eq!(snapshot_len(&committed), base_snapshots + 1);
    assert_ne!(
        committed.metadata_location_result().unwrap(),
        base.metadata_location_result().unwrap()
    );
    assert_offset_commit(&committed);
    let reloaded = catalog.load_table(base.identifier()).await.unwrap();
    assert_offset_commit(&reloaded);
}

#[tokio::test]
async fn row_delta_and_property_update_commit_as_one_metadata_version() {
    combined_commit_is_one_metadata_version(Writer::RowDelta).await;
}

#[tokio::test]
async fn overwrite_and_property_update_commit_as_one_metadata_version() {
    combined_commit_is_one_metadata_version(Writer::Overwrite).await;
}

struct RacingCatalog {
    mock: MockCatalog,
    update_calls: Arc<AtomicU32>,
}

fn racing_catalog<C: Catalog + 'static>(memory: Arc<C>, racer: Transaction) -> RacingCatalog {
    let mut mock = MockCatalog::new();
    let load_delegate = Arc::clone(&memory);
    mock.expect_load_table()
        .returning_st(move |ident: &TableIdent| {
            let catalog = Arc::clone(&load_delegate);
            let ident = ident.clone();
            Box::pin(async move { catalog.load_table(&ident).await })
        });
    let update_calls = Arc::new(AtomicU32::new(0));
    let calls = Arc::clone(&update_calls);
    let pending_racer = Arc::new(Mutex::new(Some(racer)));
    mock.expect_update_table().returning_st(move |commit| {
        let catalog = Arc::clone(&memory);
        calls.fetch_add(1, Ordering::SeqCst);
        let racer = pending_racer.lock().unwrap().take();
        Box::pin(async move {
            if let Some(racer) = racer {
                racer.commit(catalog.as_ref()).await.unwrap();
            }
            catalog.update_table(commit).await
        })
    });
    RacingCatalog { mock, update_calls }
}

fn unrelated_append(table: &Table) -> Transaction {
    let tx = Transaction::new(table);
    tx.fast_append()
        .add_data_files(vec![data_file("test/concurrent.parquet", 1)])
        .apply(tx)
        .unwrap()
}

async fn retried_commit_carries_both_exactly_once(writer: Writer) {
    let memory = Arc::new(new_memory_catalog().await);
    let base = seeded_table(memory.as_ref()).await;
    let base_log = base.metadata().metadata_log().len();
    let base_snapshots = snapshot_len(&base);
    let racing = racing_catalog(Arc::clone(&memory), unrelated_append(&base));

    let committed = offset_commit(&base, writer)
        .commit(&racing.mock)
        .await
        .unwrap();

    assert_eq!(racing.update_calls.load(Ordering::SeqCst), 2);
    assert_eq!(committed.metadata().metadata_log().len(), base_log + 2);
    assert_eq!(snapshot_len(&committed), base_snapshots + 2);
    let current = committed.metadata().current_snapshot().unwrap();
    let parent = committed
        .metadata()
        .snapshot_by_id(current.parent_snapshot_id().unwrap())
        .unwrap();
    assert!(
        !parent
            .summary()
            .additional_properties
            .contains_key(EPOCH_KEY)
    );
    assert_ne!(
        Some(parent.snapshot_id()),
        base.metadata().current_snapshot_id()
    );
    assert_eq!(
        parent.parent_snapshot_id(),
        base.metadata().current_snapshot_id()
    );
    assert_offset_commit(&committed);
    let live = live_file_paths(&committed).await;
    assert!(live.contains("test/concurrent.parquet"));
    assert!(live.contains("test/batch.parquet"));
    let reloaded = memory.load_table(base.identifier()).await.unwrap();
    assert_offset_commit(&reloaded);
}

#[tokio::test]
async fn row_delta_property_commit_survives_a_retry_over_an_unrelated_append() {
    retried_commit_carries_both_exactly_once(Writer::RowDelta).await;
}

#[tokio::test]
async fn overwrite_property_commit_survives_a_retry_over_an_unrelated_append() {
    retried_commit_carries_both_exactly_once(Writer::Overwrite).await;
}

fn same_key_writer(table: &Table, with_append: bool) -> Transaction {
    let tx = Transaction::new(table);
    let tx = if with_append {
        tx.fast_append()
            .add_data_files(vec![data_file("test/concurrent.parquet", 1)])
            .apply(tx)
            .unwrap()
    } else {
        tx
    };
    tx.update_table_properties()
        .set(OFFSET_KEY.to_string(), "src=99".to_string())
        .apply(tx)
        .unwrap()
}

async fn same_key_race(with_append: bool) {
    let memory = Arc::new(new_memory_catalog().await);
    let base = seeded_table(memory.as_ref()).await;
    let base_log = base.metadata().metadata_log().len();
    let racing = racing_catalog(Arc::clone(&memory), same_key_writer(&base, with_append));

    let committed = offset_commit(&base, Writer::RowDelta)
        .commit(&racing.mock)
        .await
        .unwrap();

    assert_eq!(racing.update_calls.load(Ordering::SeqCst), 2);
    assert_eq!(committed.metadata().metadata_log().len(), base_log + 2);
    let reloaded = memory.load_table(base.identifier()).await.unwrap();
    assert_offset_commit(&reloaded);
    let history = reloaded.metadata().metadata_log();
    let racer_version = &history[history.len() - 1];
    let racer_metadata =
        crate::spec::TableMetadata::read_from(reloaded.file_io(), &racer_version.metadata_file)
            .await
            .unwrap();
    assert_eq!(
        racer_metadata
            .properties()
            .get(OFFSET_KEY)
            .map(String::as_str),
        Some("src=99")
    );
}

#[tokio::test]
async fn retried_commit_overwrites_a_concurrent_same_key_property_without_notice() {
    same_key_race(false).await;
}

#[tokio::test]
async fn retried_commit_overwrites_a_concurrent_append_and_same_key_property_without_notice() {
    same_key_race(true).await;
}
