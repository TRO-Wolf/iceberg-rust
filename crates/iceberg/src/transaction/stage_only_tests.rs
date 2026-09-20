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

use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, MAIN_BRANCH,
    ManifestContentType, Operation, Struct,
};
use crate::table::Table;
use crate::transaction::tests::make_v2_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction, staged_snapshot_for_wap_id};
use crate::{Catalog, Error, ErrorKind};

pub(crate) const STAGED_WAP_ID_PROP: &str = "wap.id";

pub(crate) fn data_file(path: &str, part_value: i64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(part_value))]))
        .build()
        .expect("build fixture data file")
}

pub(crate) fn pos_delete_file(path: &str, part_value: i64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(part_value))]))
        .build()
        .expect("build fixture delete file")
}

pub(crate) fn wap_properties(wap_id: &str) -> HashMap<String, String> {
    HashMap::from([(STAGED_WAP_ID_PROP.to_string(), wap_id.to_string())])
}

pub(crate) async fn append_main(
    catalog: &impl Catalog,
    table: &Table,
    files: Vec<DataFile>,
) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .expect("apply base append");
    tx.commit(catalog).await.expect("commit base append")
}

pub(crate) async fn snapshot_live_file_paths(
    table: &Table,
    snapshot_id: i64,
    content: ManifestContentType,
) -> HashSet<String> {
    let snapshot = table
        .metadata()
        .snapshot_by_id(snapshot_id)
        .expect("snapshot readable by id");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    let mut live = HashSet::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != content {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load manifest");
        for entry in manifest.entries() {
            if entry.is_alive() {
                live.insert(entry.file_path().to_string());
            }
        }
    }
    live
}

pub(crate) async fn live_file_paths(
    table: &Table,
    content: ManifestContentType,
) -> HashSet<String> {
    let Some(snapshot_id) = table.metadata().current_snapshot_id() else {
        return HashSet::new();
    };
    snapshot_live_file_paths(table, snapshot_id, content).await
}

pub(crate) fn main_ref_id(table: &Table) -> Option<i64> {
    table
        .metadata()
        .snapshot_for_ref(MAIN_BRANCH)
        .map(|snapshot| snapshot.snapshot_id())
}

pub(crate) fn snapshot_count(table: &Table) -> usize {
    table.metadata().snapshots().count()
}

pub(crate) fn non_current_snapshot_ids(table: &Table) -> Vec<i64> {
    let current = table.metadata().current_snapshot_id();
    table
        .metadata()
        .snapshots()
        .map(|snapshot| snapshot.snapshot_id())
        .filter(|id| Some(*id) != current)
        .collect()
}

pub(crate) fn summary_prop(table: &Table, snapshot_id: i64, key: &str) -> Option<String> {
    table
        .metadata()
        .snapshot_by_id(snapshot_id)
        .and_then(|snapshot| snapshot.summary().additional_properties.get(key).cloned())
}

#[track_caller]
pub(crate) fn assert_staged_invariants(
    table: &Table,
    base_current: Option<i64>,
    base_main: Option<i64>,
    base_history_len: usize,
    base_count: usize,
    wap_id: &str,
) -> i64 {
    assert_eq!(
        snapshot_count(table),
        base_count + 1,
        "a staged commit must add exactly one snapshot to metadata"
    );
    assert_eq!(
        table.metadata().current_snapshot_id(),
        base_current,
        "stage_only must not advance current-snapshot-id"
    );
    assert_eq!(
        main_ref_id(table),
        base_main,
        "stage_only must not move the main ref"
    );
    assert_eq!(
        table.metadata().history().len(),
        base_history_len,
        "stage_only must not add a snapshot-log entry"
    );
    let staged_ids = non_current_snapshot_ids(table);
    assert_eq!(
        staged_ids.len(),
        1,
        "the staged snapshot must be the only snapshot off main"
    );
    let staged_id = *staged_ids.first().expect("a staged snapshot id exists");
    assert_eq!(
        summary_prop(table, staged_id, STAGED_WAP_ID_PROP).as_deref(),
        Some(wap_id),
        "the staged snapshot's summary must carry the wap.id"
    );
    staged_id
}

pub(crate) async fn staged_base(catalog: &impl Catalog) -> Table {
    let table = make_v2_minimal_table_in_catalog(catalog).await;
    append_main(catalog, &table, vec![data_file("test/base.parquet", 9)]).await
}

pub(crate) async fn stage_fast_append(
    catalog: &impl Catalog,
    table: &Table,
    path: &str,
    part_value: i64,
    wap_id: &str,
) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .set_snapshot_properties(wap_properties(wap_id))
        .add_data_files(vec![data_file(path, part_value)])
        .stage_only()
        .apply(tx)
        .expect("apply staged append");
    tx.commit(catalog).await.expect("commit staged append")
}

pub(crate) async fn cherry_pick(catalog: &impl Catalog, table: &Table, snapshot_id: i64) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .cherry_pick(snapshot_id)
        .apply(tx)
        .expect("apply cherry-pick");
    tx.commit(catalog).await.expect("commit cherry-pick")
}

pub(crate) async fn publish_changes(catalog: &impl Catalog, table: &Table, wap_id: &str) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .publish_changes(wap_id)
        .apply(tx)
        .expect("apply publish-changes");
    tx.commit(catalog).await.expect("commit publish-changes")
}

pub(crate) async fn publish_changes_err(
    catalog: &impl Catalog,
    table: &Table,
    wap_id: &str,
) -> Error {
    let tx = Transaction::new(table);
    let tx = tx
        .publish_changes(wap_id)
        .apply(tx)
        .expect("apply publish-changes");
    tx.commit(catalog)
        .await
        .expect_err("publish-changes should fail")
}

#[tokio::test]
async fn merge_append_stage_only_adds_snapshot_without_moving_main() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let base_current = table.metadata().current_snapshot_id();
    let base_main = main_ref_id(&table);
    let base_history_len = table.metadata().history().len();
    let base_count = snapshot_count(&table);
    let base_live = live_file_paths(&table, ManifestContentType::Data).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .merge_append()
        .set_snapshot_properties(wap_properties("wap-merge"))
        .add_data_files(vec![data_file("test/staged.parquet", 0)])
        .stage_only()
        .apply(tx)
        .expect("apply staged merge append");
    let staged_table = tx
        .commit(&catalog)
        .await
        .expect("commit staged merge append");

    let staged_id = assert_staged_invariants(
        &staged_table,
        base_current,
        base_main,
        base_history_len,
        base_count,
        "wap-merge",
    );
    assert_eq!(
        live_file_paths(&staged_table, ManifestContentType::Data).await,
        base_live,
        "a read of main must be unchanged by the staged merge append"
    );
    assert_eq!(
        staged_table
            .metadata()
            .snapshot_by_id(staged_id)
            .expect("staged snapshot readable by id")
            .summary()
            .operation,
        Operation::Append,
        "a staged merge append records an append snapshot"
    );
}

#[tokio::test]
async fn overwrite_files_stage_only_adds_snapshot_without_moving_main() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let base_current = table.metadata().current_snapshot_id();
    let base_main = main_ref_id(&table);
    let base_history_len = table.metadata().history().len();
    let base_count = snapshot_count(&table);
    let base_live = live_file_paths(&table, ManifestContentType::Data).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .overwrite_files()
        .set_snapshot_properties(wap_properties("wap-ow"))
        .add_file(data_file("test/new.parquet", 0))
        .delete_file("test/base.parquet")
        .stage_only()
        .apply(tx)
        .expect("apply staged overwrite");
    let staged_table = tx.commit(&catalog).await.expect("commit staged overwrite");

    assert_staged_invariants(
        &staged_table,
        base_current,
        base_main,
        base_history_len,
        base_count,
        "wap-ow",
    );
    assert_eq!(
        live_file_paths(&staged_table, ManifestContentType::Data).await,
        base_live,
        "a read of main must be unchanged by the staged overwrite"
    );
}

#[tokio::test]
async fn replace_partitions_stage_only_adds_snapshot_without_moving_main() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let base_current = table.metadata().current_snapshot_id();
    let base_main = main_ref_id(&table);
    let base_history_len = table.metadata().history().len();
    let base_count = snapshot_count(&table);
    let base_live = live_file_paths(&table, ManifestContentType::Data).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .replace_partitions()
        .set_snapshot_properties(wap_properties("wap-rp"))
        .add_file(data_file("test/new.parquet", 9))
        .stage_only()
        .apply(tx)
        .expect("apply staged replace partitions");
    let staged_table = tx
        .commit(&catalog)
        .await
        .expect("commit staged replace partitions");

    assert_staged_invariants(
        &staged_table,
        base_current,
        base_main,
        base_history_len,
        base_count,
        "wap-rp",
    );
    assert_eq!(
        live_file_paths(&staged_table, ManifestContentType::Data).await,
        base_live,
        "a read of main must be unchanged by the staged replace"
    );
}

#[tokio::test]
async fn row_delta_stage_only_adds_snapshot_without_moving_main() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let base_current = table.metadata().current_snapshot_id();
    let base_main = main_ref_id(&table);
    let base_history_len = table.metadata().history().len();
    let base_count = snapshot_count(&table);
    let base_live = live_file_paths(&table, ManifestContentType::Data).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .set_snapshot_properties(wap_properties("wap-rd"))
        .add_data_files(vec![data_file("test/new.parquet", 0)])
        .add_deletes(vec![pos_delete_file("test/del.parquet", 9)])
        .stage_only()
        .apply(tx)
        .expect("apply staged row delta");
    let staged_table = tx.commit(&catalog).await.expect("commit staged row delta");

    assert_staged_invariants(
        &staged_table,
        base_current,
        base_main,
        base_history_len,
        base_count,
        "wap-rd",
    );
    assert_eq!(
        live_file_paths(&staged_table, ManifestContentType::Data).await,
        base_live,
        "a read of main must be unchanged by the staged row delta"
    );
    assert!(
        live_file_paths(&staged_table, ManifestContentType::Deletes)
            .await
            .is_empty(),
        "a read of main must not see the staged delete file"
    );
}

#[tokio::test]
async fn delete_files_stage_only_adds_snapshot_without_moving_main() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let base_current = table.metadata().current_snapshot_id();
    let base_main = main_ref_id(&table);
    let base_history_len = table.metadata().history().len();
    let base_count = snapshot_count(&table);
    let base_live = live_file_paths(&table, ManifestContentType::Data).await;

    let tx = Transaction::new(&table);
    let tx = tx
        .delete_files()
        .set_snapshot_properties(wap_properties("wap-del"))
        .delete_file("test/base.parquet")
        .stage_only()
        .apply(tx)
        .expect("apply staged delete");
    let staged_table = tx.commit(&catalog).await.expect("commit staged delete");

    assert_staged_invariants(
        &staged_table,
        base_current,
        base_main,
        base_history_len,
        base_count,
        "wap-del",
    );
    assert_eq!(
        live_file_paths(&staged_table, ManifestContentType::Data).await,
        base_live,
        "a read of main must be unchanged by the staged delete"
    );
}

#[tokio::test]
async fn staged_snapshot_for_wap_id_finds_the_staged_snapshot() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let table = stage_fast_append(&catalog, &table, "test/staged.parquet", 0, "wap-find").await;
    let staged_id = *non_current_snapshot_ids(&table)
        .first()
        .expect("a staged snapshot exists");

    let found = staged_snapshot_for_wap_id(table.metadata(), "wap-find")
        .expect("the staged snapshot resolves by wap id");
    assert_eq!(found.snapshot_id(), staged_id);
    assert_eq!(
        found
            .summary()
            .additional_properties
            .get(STAGED_WAP_ID_PROP)
            .map(String::as_str),
        Some("wap-find"),
    );
}

#[tokio::test]
async fn staged_snapshot_for_wap_id_unknown_id_has_java_message() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;

    let error = staged_snapshot_for_wap_id(table.metadata(), "nope")
        .expect_err("an unknown wap id must fail");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot apply unknown WAP ID 'nope'",
        "Java's unknown-WAP-id text, verbatim"
    );
}

#[tokio::test]
async fn staged_snapshot_for_wap_id_non_unique_has_java_message() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let table = stage_fast_append(&catalog, &table, "test/s1.parquet", 0, "wap-multi").await;
    let table = stage_fast_append(&catalog, &table, "test/s2.parquet", 1, "wap-multi").await;
    assert_eq!(
        non_current_snapshot_ids(&table).len(),
        2,
        "the fixture holds two staged snapshots sharing one wap id"
    );

    let error = staged_snapshot_for_wap_id(table.metadata(), "wap-multi")
        .expect_err("a non-unique wap id must fail");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot apply non-unique WAP ID. Found multiple snapshots with WAP ID 'wap-multi'",
        "Java's non-unique-WAP-id text, verbatim"
    );
}

#[tokio::test]
async fn staged_snapshot_for_wap_id_already_published_has_java_message() {
    let catalog = new_memory_catalog().await;
    let table = staged_base(&catalog).await;
    let table = stage_fast_append(&catalog, &table, "test/staged.parquet", 0, "wap-dup").await;
    let staged_id = *non_current_snapshot_ids(&table)
        .first()
        .expect("a staged snapshot exists");

    let table = cherry_pick(&catalog, &table, staged_id).await;
    assert_eq!(
        table.metadata().current_snapshot_id(),
        Some(staged_id),
        "the staged snapshot fast-forwarded onto main"
    );

    let error = staged_snapshot_for_wap_id(table.metadata(), "wap-dup")
        .expect_err("an already-published wap id must fail");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Duplicate request to cherry pick wap id that was published already: wap-dup",
        "Java's DuplicateWAPCommitException text, verbatim"
    );
}
