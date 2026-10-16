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

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use super::occ_scoped_tests::{data_file, live_file_paths, snapshot_len};
use crate::catalog::MockCatalog;
use crate::memory::tests::new_memory_catalog;
use crate::table::Table;
use crate::transaction::tests::make_v2_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, Error, ErrorKind, TableIdent};

const OURS: &str = "test/ours.parquet";
const THEIRS: &str = "test/theirs.parquet";

fn conflict() -> Error {
    Error::new(ErrorKind::CatalogCommitConflicts, "stale base").with_retryable(true)
}

async fn seeded_table(catalog: &impl Catalog) -> Table {
    let table = make_v2_minimal_table_in_catalog(catalog).await;
    let tx = Transaction::new(&table);
    let tx = tx
        .update_table_properties()
        .set("commit.retry.num-retries".to_string(), "3".to_string())
        .set("commit.retry.min-wait-ms".to_string(), "1".to_string())
        .set("commit.retry.max-wait-ms".to_string(), "5".to_string())
        .apply(tx)
        .expect("stage the retry properties");
    tx.commit(catalog)
        .await
        .expect("commit the retry properties")
}

fn our_append(table: &Table, check_duplicate: bool) -> Transaction {
    let tx = Transaction::new(table);
    tx.fast_append()
        .with_check_duplicate(check_duplicate)
        .add_data_files(vec![data_file(OURS, 0)])
        .apply(tx)
        .expect("stage our append")
}

async fn live_entries(table: &Table, path: &str) -> usize {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the manifest list");
    let mut count = 0;
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load a manifest");
        count += manifest
            .entries()
            .iter()
            .filter(|entry| entry.is_alive() && entry.file_path() == path)
            .count();
    }
    count
}

#[derive(Clone, Copy)]
enum FirstUpdate {
    LandsThenConflict,
    OtherWriterWins,
}

struct ScriptedCatalog {
    mock: MockCatalog,
    update_calls: Arc<AtomicU32>,
}

fn scripted_catalog<C: Catalog + 'static>(
    memory: Arc<C>,
    first: FirstUpdate,
    racer: Option<Transaction>,
    failing_loads_after_first_update: u32,
) -> ScriptedCatalog {
    let update_calls = Arc::new(AtomicU32::new(0));
    let failing_loads = Arc::new(AtomicU32::new(0));
    let mut mock = MockCatalog::new();
    let load_delegate = Arc::clone(&memory);
    let load_calls = Arc::clone(&update_calls);
    let load_failures = Arc::clone(&failing_loads);
    mock.expect_load_table()
        .returning_st(move |ident: &TableIdent| {
            let catalog = Arc::clone(&load_delegate);
            let ident = ident.clone();
            let after_first_update = load_calls.load(Ordering::SeqCst) > 0;
            let fail = after_first_update
                && load_failures.fetch_add(1, Ordering::SeqCst) < failing_loads_after_first_update;
            Box::pin(async move {
                if fail {
                    return Err(
                        Error::new(ErrorKind::Unexpected, "refresh failed").with_retryable(true)
                    );
                }
                catalog.load_table(&ident).await
            })
        });
    let calls = Arc::clone(&update_calls);
    let pending_racer = Arc::new(Mutex::new(racer));
    mock.expect_update_table().returning_st(move |commit| {
        let catalog = Arc::clone(&memory);
        let first_call = calls.fetch_add(1, Ordering::SeqCst) == 0;
        let racer = pending_racer.lock().expect("racer slot").take();
        Box::pin(async move {
            if !first_call {
                return catalog.update_table(commit).await;
            }
            match first {
                FirstUpdate::LandsThenConflict => {
                    catalog.update_table(commit).await?;
                }
                FirstUpdate::OtherWriterWins => {
                    if let Some(racer) = racer {
                        racer.commit(catalog.as_ref()).await?;
                    }
                }
            }
            Err(conflict())
        })
    });
    ScriptedCatalog { mock, update_calls }
}

fn their_append(table: &Table) -> Transaction {
    let tx = Transaction::new(table);
    tx.fast_append()
        .add_data_files(vec![data_file(THEIRS, 1)])
        .apply(tx)
        .expect("stage their append")
}

async fn landed_then_rejected(check_duplicate: bool, failing_loads: u32) {
    let memory = Arc::new(new_memory_catalog().await);
    let base = seeded_table(memory.as_ref()).await;
    let base_snapshots = snapshot_len(&base);
    let scripted = scripted_catalog(
        Arc::clone(&memory),
        FirstUpdate::LandsThenConflict,
        None,
        failing_loads,
    );

    let committed = our_append(&base, check_duplicate)
        .commit(&scripted.mock)
        .await
        .expect("a landed attempt rejected as a conflict is a committed append");

    assert_eq!(scripted.update_calls.load(Ordering::SeqCst), 1);
    assert_eq!(snapshot_len(&committed), base_snapshots + 1);
    let reloaded = memory.load_table(base.identifier()).await.expect("reload");
    assert_eq!(snapshot_len(&reloaded), base_snapshots + 1);
    assert_eq!(live_entries(&reloaded, OURS).await, 1);
}

#[tokio::test]
async fn landed_attempt_rejected_as_conflict_is_not_reapplied() {
    landed_then_rejected(false, 0).await;
}

#[tokio::test]
async fn landed_attempt_rejected_as_conflict_commits_under_the_duplicate_check() {
    landed_then_rejected(true, 0).await;
}

#[tokio::test]
async fn landed_attempt_is_found_after_a_failed_refresh() {
    landed_then_rejected(false, 1).await;
}

#[tokio::test]
async fn other_writer_wins_and_our_append_rebases_and_lands_once() {
    let memory = Arc::new(new_memory_catalog().await);
    let base = seeded_table(memory.as_ref()).await;
    let base_snapshots = snapshot_len(&base);
    let scripted = scripted_catalog(
        Arc::clone(&memory),
        FirstUpdate::OtherWriterWins,
        Some(their_append(&base)),
        0,
    );

    let committed = our_append(&base, true)
        .commit(&scripted.mock)
        .await
        .expect("a genuine conflict re-bases and lands");

    assert_eq!(scripted.update_calls.load(Ordering::SeqCst), 2);
    assert_eq!(snapshot_len(&committed), base_snapshots + 2);
    let live = live_file_paths(&committed).await;
    assert!(live.contains(OURS));
    assert!(live.contains(THEIRS));
    assert_eq!(live_entries(&committed, OURS).await, 1);
}
