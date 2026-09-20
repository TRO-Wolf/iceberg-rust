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

use std::sync::Arc;

use super::stage_only_tests::{
    STAGED_WAP_ID_PROP, append_main, data_file, live_file_paths, non_current_snapshot_ids,
    publish_changes, publish_changes_err, stage_fast_append, staged_base,
};
use crate::memory::tests::new_memory_catalog;
use crate::spec::{MAIN_BRANCH, ManifestContentType};
use crate::transaction::action::TransactionAction;
use crate::transaction::tests::make_v3_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, PublishChangesAction, Transaction};
use crate::{ErrorKind, TableUpdate};

#[tokio::test]
async fn publish_changes_fast_forwards_the_staged_snapshot() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let table = stage_fast_append(&catalog, &table, "test/staged.parquet", 0, "wap-ff").await;
    let staged_id = *non_current_snapshot_ids(&table)
        .first()
        .expect("a staged snapshot exists");
    let snapshot_count_before = table.metadata().snapshots().count();
    let base_live = live_file_paths(&table, ManifestContentType::Data).await;

    let table = publish_changes(&catalog, &table, "wap-ff").await;

    assert_eq!(
        table.metadata().current_snapshot_id(),
        Some(staged_id),
        "a head-parented staged snapshot publishes by fast-forwarding main"
    );
    assert_eq!(
        table.metadata().snapshots().count(),
        snapshot_count_before,
        "a fast-forward produces no new snapshot"
    );
    let published = table
        .metadata()
        .current_snapshot()
        .expect("the publish left a current snapshot");
    assert_eq!(
        published
            .summary()
            .additional_properties
            .get(STAGED_WAP_ID_PROP)
            .map(String::as_str),
        Some("wap-ff"),
        "the fast-forwarded snapshot keeps its wap.id, as Java does"
    );
    let live = live_file_paths(&table, ManifestContentType::Data).await;
    assert!(
        live.contains("test/staged.parquet"),
        "a read of main sees the staged data: {live:?}"
    );
    assert!(
        base_live.is_subset(&live),
        "the base files survive the publish: {live:?}"
    );
}

#[tokio::test]
async fn publish_changes_replays_and_sets_source_and_published_wap_id() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let table = stage_fast_append(&catalog, &table, "test/staged.parquet", 0, "wap-replay").await;
    let staged_id = *non_current_snapshot_ids(&table)
        .first()
        .expect("a staged snapshot exists");
    let table = append_main(&catalog, &table, vec![data_file("test/head.parquet", 9)]).await;
    let snapshot_count_before = table.metadata().snapshots().count();

    let table = publish_changes(&catalog, &table, "wap-replay").await;

    assert_eq!(
        table.metadata().snapshots().count(),
        snapshot_count_before + 1,
        "a staged snapshot whose parent is not the head replays into a NEW snapshot"
    );
    let published = table
        .metadata()
        .current_snapshot()
        .expect("the publish left a current snapshot");
    assert_ne!(
        published.snapshot_id(),
        staged_id,
        "the published snapshot is new, not the staged one"
    );
    let props = &published.summary().additional_properties;
    assert_eq!(
        props.get("source-snapshot-id"),
        Some(&staged_id.to_string()),
        "Java tags the replayed snapshot with the staged snapshot id"
    );
    assert_eq!(
        props.get("published-wap-id").map(String::as_str),
        Some("wap-replay"),
        "Java tags the replayed snapshot with the published wap id"
    );
    let live = live_file_paths(&table, ManifestContentType::Data).await;
    assert!(
        live.contains("test/staged.parquet") && live.contains("test/head.parquet"),
        "a read of main sees both the staged and the head data: {live:?}"
    );
}

#[tokio::test]
async fn publish_changes_unknown_wap_id_has_java_message() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;

    let error = publish_changes_err(&catalog, &table, "nope").await;
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot apply unknown WAP ID 'nope'",
        "Java's unknown-WAP-id text, verbatim"
    );
}

#[tokio::test]
async fn publish_changes_twice_has_java_message() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let table = stage_fast_append(&catalog, &table, "test/staged.parquet", 0, "wap-twice").await;
    let table = publish_changes(&catalog, &table, "wap-twice").await;

    let error = publish_changes_err(&catalog, &table, "wap-twice").await;
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Duplicate request to cherry pick wap id that was published already: wap-twice",
        "Java's DuplicateWAPCommitException text, verbatim"
    );
}

#[tokio::test]
async fn publish_changes_binds_the_wap_id_once_per_attempt() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let table = stage_fast_append(&catalog, &table, "test/s1.parquet", 0, "wap-bind").await;
    let bound_id = *non_current_snapshot_ids(&table)
        .first()
        .expect("a staged snapshot exists");

    let action = Arc::new(PublishChangesAction::new("wap-bind"));
    action
        .clone()
        .validate(None, &table)
        .await
        .expect("the first resolve binds the staged snapshot");

    let table = stage_fast_append(&catalog, &table, "test/s2.parquet", 1, "wap-bind").await;
    assert_eq!(
        non_current_snapshot_ids(&table).len(),
        2,
        "a second staged snapshot sharing the wap id landed between validate and commit"
    );

    let mut commit = action
        .commit(&table)
        .await
        .expect("the commit publishes the bound id, not a re-resolved one");
    match commit.take_updates().as_slice() {
        [
            TableUpdate::SetSnapshotRef {
                ref_name,
                reference,
            },
        ] => {
            assert_eq!(ref_name, MAIN_BRANCH);
            assert_eq!(
                reference.snapshot_id, bound_id,
                "Java cherry-picks the snapshot id bound at resolve time"
            );
        }
        updates => panic!("a fast-forward publish emits one main-ref update: {updates:?}"),
    }
}

#[tokio::test]
async fn publish_changes_bound_snapshot_expired_fails_unknown_snapshot_id() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let table = stage_fast_append(&catalog, &table, "test/s1.parquet", 0, "wap-gone").await;
    let bound_id = *non_current_snapshot_ids(&table)
        .first()
        .expect("a staged snapshot exists");

    let action = Arc::new(PublishChangesAction::new("wap-gone"));
    action
        .clone()
        .validate(None, &table)
        .await
        .expect("the first resolve binds the staged snapshot");

    let tx = Transaction::new(&table);
    let tx = tx
        .expire_snapshots()
        .expire_snapshot_id(bound_id)
        .apply(tx)
        .expect("apply expire");
    let table = tx.commit(&catalog).await.expect("commit expire");
    assert!(
        table.metadata().snapshot_by_id(bound_id).is_none(),
        "the bound staged snapshot is expired out of metadata"
    );
    let table = stage_fast_append(&catalog, &table, "test/s2.parquet", 1, "wap-gone").await;

    let error = action
        .commit(&table)
        .await
        .err()
        .expect("Java fails the bound id unknown rather than publishing the replacement");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        format!("Cannot cherry-pick unknown snapshot ID: {bound_id}"),
        "Java's cherry-pick unknown-snapshot-id text, verbatim"
    );
}

#[tokio::test]
async fn publish_changes_replay_assigns_fresh_row_ids_on_v3() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table = stage_fast_append(&catalog, &table, "test/staged.parquet", 0, "wap-rowid").await;
    let staged_id = *non_current_snapshot_ids(&table)
        .first()
        .expect("a staged snapshot exists");
    let staged_first_row_id = table
        .metadata()
        .snapshot_by_id(staged_id)
        .expect("the staged snapshot is in metadata")
        .first_row_id()
        .expect("a v3 staged snapshot carries a first row id");
    let table = append_main(&catalog, &table, vec![data_file("test/head.parquet", 9)]).await;

    let table = publish_changes(&catalog, &table, "wap-rowid").await;

    let published = table
        .metadata()
        .current_snapshot()
        .expect("the publish left a current snapshot");
    let published_first_row_id = published
        .first_row_id()
        .expect("a v3 published snapshot carries a first row id");
    assert!(
        published_first_row_id > staged_first_row_id,
        "the publish assigned a FRESH row range ({published_first_row_id}) rather than copying \
         the staged snapshot's ({staged_first_row_id})"
    );
    let manifest_list = published
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load the published manifest list");
    let mut stored = Vec::new();
    for manifest_file in manifest_list.entries() {
        let bytes = table
            .file_io()
            .new_input(&manifest_file.manifest_path)
            .expect("open the manifest")
            .read()
            .await
            .expect("read the manifest bytes");
        let manifest = crate::spec::Manifest::parse_avro(&bytes).expect("parse the manifest avro");
        for entry in manifest.entries() {
            if entry.file_path() == "test/staged.parquet" {
                stored.push(entry.data_file().first_row_id());
            }
        }
    }
    assert_eq!(
        stored,
        vec![None],
        "the replayed file is stored with no first_row_id, so a fresh one is assigned"
    );
}
