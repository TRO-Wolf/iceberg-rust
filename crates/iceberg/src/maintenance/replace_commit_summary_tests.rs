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
use std::sync::Arc;

use super::rewrite_data_files::tests::{
    add_deletes, append_files, create_partitioned_table, local_fs_catalog, write_data_file,
    write_equality_delete_file, write_position_delete_file,
};
use super::rewrite_data_files_evolved_spec_tests::{
    literal_from_long_transform, write_current_spec_file,
};
use super::rewrite_data_files_router_bound_tests::write_dv;
use super::{RewriteDataFiles, RewriteJobOrder, RewritePositionDeleteFiles};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestStatus,
    NestedField, Operation, PartitionSpec, PrimitiveType, Schema, Snapshot, Struct, Transform,
    Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, NamespaceIdent, TableCreation, TableIdent};

const PARTITION: i64 = 0;

fn rows(x: i64, y_start: i64, count: i64) -> Vec<(i64, i64, i64)> {
    (0..count).map(|offset| (x, y_start + offset, 0)).collect()
}

async fn merge_commit(
    catalog: &impl Catalog,
    table: &Table,
    data_files: Vec<DataFile>,
    delete_files: Vec<DataFile>,
) -> Table {
    let tx = Transaction::new(table);
    let action = tx
        .row_delta()
        .add_data_files(data_files)
        .add_deletes(delete_files);
    let tx = action.apply(tx).expect("apply row delta");
    tx.commit(catalog).await.expect("commit row delta")
}

async fn seed_mor_table(catalog: &impl Catalog, format_version: FormatVersion) -> Table {
    let table = create_partitioned_table(catalog, format_version).await;
    let mut data_files = Vec::new();
    for index in 0..4i64 {
        data_files.push(
            write_data_file(
                &table,
                &format!("data-{index}.parquet"),
                PARTITION,
                &rows(PARTITION, index * 250, 250),
            )
            .await,
        );
    }
    let deleted_path = data_files[0].file_path().to_string();
    let mut table = append_files(catalog, &table, data_files).await;
    for merge in 0..3i64 {
        let merge_file = write_data_file(
            &table,
            &format!("merge-{merge}.parquet"),
            PARTITION,
            &rows(PARTITION, 1000 + merge * 200, 200),
        )
        .await;
        let delete_file =
            write_position_delete_file(&table, PARTITION, &[(deleted_path.clone(), merge)]).await;
        table = merge_commit(catalog, &table, vec![merge_file], vec![delete_file]).await;
    }
    table
}

async fn upgrade_to_v3(catalog: &impl Catalog, table: &Table) -> Table {
    let tx = Transaction::new(table);
    let action = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3);
    let tx = action.apply(tx).expect("apply upgrade");
    tx.commit(catalog).await.expect("commit upgrade")
}

fn long_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                3,
                "z",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("build schema")
}

async fn create_bucketed_table(catalog: &impl Catalog, format_version: FormatVersion) -> Table {
    let schema = long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("x", "x_bucket", Transform::Bucket(8))
        .expect("add bucket field")
        .build()
        .expect("build spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let table_ident = TableIdent::new(namespace.clone(), "t".to_string());
    let creation = TableCreation::builder()
        .name(table_ident.name().to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

fn props_of(snapshot: &Snapshot) -> HashMap<String, String> {
    snapshot.summary().additional_properties.clone()
}

fn prop_or_zero(props: &HashMap<String, String>, key: &str) -> u64 {
    props
        .get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn prop_u64(props: &HashMap<String, String>, key: &str) -> u64 {
    props
        .get(key)
        .unwrap_or_else(|| panic!("missing summary key '{key}' in {props:?}"))
        .parse()
        .unwrap_or_else(|_| panic!("summary key '{key}' is not a number in {props:?}"))
}

fn assert_summary_keys(
    props: &HashMap<String, String>,
    required: &[&str],
    forbidden: &[&str],
    context: &str,
) {
    for key in required {
        assert!(
            props.contains_key(*key),
            "{context}: required key '{key}' absent from {props:?}"
        );
    }
    for key in forbidden {
        assert!(
            !props.contains_key(*key),
            "{context}: forbidden key '{key}' present in {props:?}"
        );
    }
}

fn summary_content_size(data_file: &DataFile) -> u64 {
    if data_file.content_type() == DataContentType::PositionDeletes
        && data_file.file_format() == DataFileFormat::Puffin
    {
        data_file
            .content_size_in_bytes
            .map_or(data_file.file_size_in_bytes, |size| size.max(0) as u64)
    } else {
        data_file.file_size_in_bytes
    }
}

async fn live_files(
    table: &Table,
    snapshot: &Snapshot,
    content: ManifestContentType,
) -> Vec<DataFile> {
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != content {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

async fn snapshot_added_files(
    table: &Table,
    snapshot: &Snapshot,
    content: ManifestContentType,
) -> Vec<DataFile> {
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != content
            || manifest_file.added_snapshot_id != snapshot.snapshot_id()
        {
            continue;
        }
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.status() == ManifestStatus::Added {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

fn live_position_delete_records(delete_files: &[DataFile]) -> u64 {
    delete_files
        .iter()
        .filter(|file| file.content_type() == DataContentType::PositionDeletes)
        .map(|file| file.record_count)
        .sum()
}

fn live_equality_delete_records(delete_files: &[DataFile]) -> u64 {
    delete_files
        .iter()
        .filter(|file| file.content_type() == DataContentType::EqualityDeletes)
        .map(|file| file.record_count)
        .sum()
}

async fn assert_replace_totals_match_live_files(table: &Table, snapshot: &Snapshot) {
    let props = props_of(snapshot);
    assert_eq!(snapshot.summary().operation, Operation::Replace);
    let data_files = live_files(table, snapshot, ManifestContentType::Data).await;
    let delete_files = live_files(table, snapshot, ManifestContentType::Deletes).await;
    let added_data = snapshot_added_files(table, snapshot, ManifestContentType::Data).await;
    let added_deletes = snapshot_added_files(table, snapshot, ManifestContentType::Deletes).await;

    let live_size: u64 = data_files
        .iter()
        .chain(delete_files.iter())
        .map(summary_content_size)
        .sum();
    let added_size: u64 = added_data
        .iter()
        .chain(added_deletes.iter())
        .map(summary_content_size)
        .sum();

    assert_eq!(
        prop_u64(&props, "added-data-files"),
        added_data.len() as u64,
        "added-data-files vs snapshot-added live data files"
    );
    assert_eq!(
        prop_u64(&props, "added-records"),
        added_data.iter().map(|file| file.record_count).sum::<u64>(),
        "added-records vs snapshot-added record count"
    );
    assert_eq!(
        prop_u64(&props, "added-files-size"),
        added_size,
        "added-files-size vs snapshot-added content size"
    );
    assert_eq!(
        prop_u64(&props, "total-data-files"),
        data_files.len() as u64,
        "total-data-files vs live data-file count"
    );
    assert_eq!(
        prop_u64(&props, "total-records"),
        data_files.iter().map(|file| file.record_count).sum::<u64>(),
        "total-records vs live data-file record sum"
    );
    assert_eq!(
        prop_u64(&props, "total-files-size"),
        live_size,
        "total-files-size vs live content size"
    );
    assert_eq!(
        prop_u64(&props, "total-delete-files"),
        delete_files.len() as u64,
        "total-delete-files vs live delete-file count"
    );
    assert_eq!(
        prop_u64(&props, "total-position-deletes"),
        live_position_delete_records(&delete_files),
        "total-position-deletes vs live position-delete records"
    );
    assert_eq!(
        prop_u64(&props, "total-equality-deletes"),
        live_equality_delete_records(&delete_files),
        "total-equality-deletes vs live equality-delete records"
    );
}

const RDF_BASE_KEYS: &[&str] = &[
    "added-data-files",
    "added-files-size",
    "added-records",
    "changed-partition-count",
    "deleted-data-files",
    "deleted-records",
    "removed-files-size",
    "manifests-created",
    "manifests-kept",
    "manifests-replaced",
    "total-data-files",
    "total-delete-files",
    "total-records",
    "total-files-size",
    "total-position-deletes",
    "total-equality-deletes",
];

const RDF_DELETE_DROP_KEYS: &[&str] = &[
    "removed-delete-files",
    "removed-position-delete-files",
    "removed-position-deletes",
];

const RDF_FORBIDDEN: &[&str] = &[
    "added-delete-files",
    "added-dvs",
    "added-position-delete-files",
    "added-position-deletes",
    "added-equality-delete-files",
    "added-equality-deletes",
    "removed-dvs",
    "removed-equality-delete-files",
    "removed-equality-deletes",
    "entries-processed",
];

fn rdf_required_keys(delete_drop: bool) -> Vec<&'static str> {
    let mut keys = RDF_BASE_KEYS.to_vec();
    if delete_drop {
        keys.extend(RDF_DELETE_DROP_KEYS);
    }
    keys
}

fn assert_rdf_key_set(props: &HashMap<String, String>, delete_drop: bool, context: &str) {
    assert_summary_keys(
        props,
        &rdf_required_keys(delete_drop),
        RDF_FORBIDDEN,
        context,
    );
}

async fn manifest_list_counts(table: &Table, snapshot: &Snapshot) -> (u64, u64) {
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut created = 0u64;
    let mut kept = 0u64;
    for manifest in manifest_list.entries() {
        if manifest.added_snapshot_id == snapshot.snapshot_id() {
            created += 1;
        } else {
            kept += 1;
        }
    }
    (created, kept)
}

fn assert_manifest_counts(
    table_props: &HashMap<String, String>,
    expected: (u64, u64, u64),
    context: &str,
) {
    let (created, kept, replaced) = expected;
    assert_eq!(
        prop_u64(table_props, "manifests-created"),
        created,
        "{context}: manifests-created"
    );
    assert_eq!(
        prop_u64(table_props, "manifests-kept"),
        kept,
        "{context}: manifests-kept"
    );
    assert_eq!(
        prop_u64(table_props, "manifests-replaced"),
        replaced,
        "{context}: manifests-replaced"
    );
}

async fn set_table_property(
    catalog: &impl Catalog,
    table: &Table,
    key: &str,
    value: &str,
) -> Table {
    let tx = Transaction::new(table);
    let action = tx
        .update_table_properties()
        .set(key.to_string(), value.to_string());
    let tx = action.apply(tx).expect("apply property update");
    tx.commit(catalog).await.expect("commit property update")
}

#[tokio::test]
async fn append_commits_stamp_manifest_counts() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let file_a =
        write_data_file(&table, "app-a.parquet", PARTITION, &rows(PARTITION, 0, 250)).await;
    let table = append_files(&catalog, &table, vec![file_a]).await;
    let first = table
        .metadata()
        .current_snapshot()
        .expect("first append snapshot")
        .as_ref()
        .clone();
    assert_eq!(first.summary().operation, Operation::Append);
    let (created, kept) = manifest_list_counts(&table, &first).await;
    assert_eq!((created, kept), (1, 0), "first append writes one manifest");
    assert_manifest_counts(&props_of(&first), (1, 0, 0), "first append");

    let file_b = write_data_file(
        &table,
        "app-b.parquet",
        PARTITION,
        &rows(PARTITION, 250, 250),
    )
    .await;
    let table = append_files(&catalog, &table, vec![file_b]).await;
    let second = table
        .metadata()
        .current_snapshot()
        .expect("second append snapshot")
        .as_ref()
        .clone();
    assert_eq!(second.summary().operation, Operation::Append);
    let (created, kept) = manifest_list_counts(&table, &second).await;
    assert_eq!(
        (created, kept),
        (1, 1),
        "second append adds one manifest and keeps the first"
    );
    assert_manifest_counts(&props_of(&second), (1, 1, 0), "second append");
    assert_eq!(
        prop_u64(&props_of(&second), "manifests-replaced"),
        0,
        "a fast_append that merges nothing replaces no manifests"
    );
}

#[tokio::test]
async fn row_delta_merge_commit_stamps_manifest_counts() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let seed = write_data_file(&table, "mseed.parquet", PARTITION, &rows(PARTITION, 0, 250)).await;
    let deleted_path = seed.file_path().to_string();
    let table = append_files(&catalog, &table, vec![seed]).await;
    let merge_file = write_data_file(
        &table,
        "mmerge.parquet",
        PARTITION,
        &rows(PARTITION, 250, 200),
    )
    .await;
    let delete_file = write_position_delete_file(&table, PARTITION, &[(deleted_path, 0)]).await;
    let table = merge_commit(&catalog, &table, vec![merge_file], vec![delete_file]).await;

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("merge snapshot")
        .as_ref()
        .clone();
    assert_eq!(snapshot.summary().operation, Operation::Overwrite);
    let (created, kept) = manifest_list_counts(&table, &snapshot).await;
    assert_eq!(
        (created, kept),
        (2, 1),
        "merge writes one data manifest and one delete manifest, keeps the append manifest"
    );
    assert_manifest_counts(&props_of(&snapshot), (2, 1, 0), "row-delta merge");
}

#[tokio::test]
async fn cow_overwrite_commit_stamps_manifest_counts() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let file_a = write_data_file(&table, "ow-a.parquet", PARTITION, &rows(PARTITION, 0, 250)).await;
    let file_b = write_data_file(
        &table,
        "ow-b.parquet",
        PARTITION,
        &rows(PARTITION, 250, 250),
    )
    .await;
    let removed = file_a.clone();
    let table = append_files(&catalog, &table, vec![file_a, file_b]).await;

    let replacement = write_data_file(
        &table,
        "ow-new.parquet",
        PARTITION,
        &rows(PARTITION, 500, 250),
    )
    .await;
    let tx = Transaction::new(&table);
    let action = tx
        .overwrite_files()
        .add_file(replacement)
        .delete_data_files(vec![removed]);
    let tx = action.apply(tx).expect("apply overwrite");
    let table = tx.commit(&catalog).await.expect("commit overwrite");

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("overwrite snapshot")
        .as_ref()
        .clone();
    assert_eq!(snapshot.summary().operation, Operation::Overwrite);
    let (created, kept) = manifest_list_counts(&table, &snapshot).await;
    assert_eq!(
        (created, kept),
        (2, 0),
        "overwrite writes the added manifest plus the tombstoning rewrite of the source manifest"
    );
    assert_manifest_counts(&props_of(&snapshot), (2, 0, 1), "CoW overwrite");
}

#[tokio::test]
async fn delete_only_commit_stamps_manifest_counts() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let seed = write_data_file(&table, "dseed.parquet", PARTITION, &rows(PARTITION, 0, 250)).await;
    let deleted_path = seed.file_path().to_string();
    let table = append_files(&catalog, &table, vec![seed]).await;
    let delete_file = write_position_delete_file(&table, PARTITION, &[(deleted_path, 0)]).await;
    let table = add_deletes(&catalog, &table, vec![delete_file]).await;

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("delete snapshot")
        .as_ref()
        .clone();
    assert_eq!(snapshot.summary().operation, Operation::Delete);
    let (created, kept) = manifest_list_counts(&table, &snapshot).await;
    assert_eq!(
        (created, kept),
        (1, 1),
        "delete commit writes one delete manifest and keeps the data manifest"
    );
    assert_manifest_counts(&props_of(&snapshot), (1, 1, 0), "delete-only commit");
}

#[tokio::test]
async fn merge_append_stamps_merge_side_replaced_count() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let file_a = write_data_file(&table, "ma-a.parquet", PARTITION, &rows(PARTITION, 0, 250)).await;
    let table = append_files(&catalog, &table, vec![file_a]).await;
    let file_b = write_data_file(
        &table,
        "ma-b.parquet",
        PARTITION,
        &rows(PARTITION, 250, 250),
    )
    .await;
    let table = append_files(&catalog, &table, vec![file_b]).await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.min-count-to-merge", "2").await;

    let file_c = write_data_file(
        &table,
        "ma-c.parquet",
        PARTITION,
        &rows(PARTITION, 500, 250),
    )
    .await;
    let tx = Transaction::new(&table);
    let action = tx.merge_append().add_data_files(vec![file_c]);
    let tx = action.apply(tx).expect("apply merge append");
    let table = tx.commit(&catalog).await.expect("commit merge append");

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("merge append snapshot")
        .as_ref()
        .clone();
    assert_eq!(snapshot.summary().operation, Operation::Append);
    let (created, kept) = manifest_list_counts(&table, &snapshot).await;
    assert_eq!(
        (created, kept),
        (1, 0),
        "the merge bin-packs all three manifests into one"
    );
    assert_manifest_counts(&props_of(&snapshot), (1, 0, 2), "merge append");
}

async fn rewrite_and_reload(
    catalog: &impl Catalog,
    action: RewriteDataFiles,
    ident_table: &Table,
) -> Table {
    action.execute(catalog).await.expect("rewrite data files");
    catalog
        .load_table(ident_table.identifier())
        .await
        .expect("reload table")
}

#[tokio::test]
async fn rdf_replace_summary_counts_added_files_and_live_totals() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_mor_table(&catalog, FormatVersion::V2).await;

    let table = rewrite_and_reload(&catalog, RewriteDataFiles::new(table.clone()), &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    assert_rdf_key_set(&props_of(&snapshot), true, "rdf default");
}

#[tokio::test]
async fn rdf_rewrite_all_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_mor_table(&catalog, FormatVersion::V2).await;

    let action = RewriteDataFiles::new(table.clone()).rewrite_all(true);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    assert_rdf_key_set(&props_of(&snapshot), true, "rdf rewrite-all");
}

#[tokio::test]
async fn rdf_fresh_sequence_numbers_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_mor_table(&catalog, FormatVersion::V2).await;

    let action = RewriteDataFiles::new(table.clone()).use_starting_sequence_number(false);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    assert_rdf_key_set(&props_of(&snapshot), true, "rdf fresh-seq");
}

#[tokio::test]
async fn rdf_partial_progress_chains_totals_across_commits() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let mut data_files = Vec::new();
    for index in 0..4i64 {
        data_files.push(
            write_data_file(
                &table,
                &format!("d0-{index}.parquet"),
                0,
                &rows(0, index * 250, 250),
            )
            .await,
        );
    }
    for index in 0..3i64 {
        data_files.push(
            write_data_file(
                &table,
                &format!("d1-{index}.parquet"),
                1,
                &rows(1, index * 250, 250),
            )
            .await,
        );
    }
    let path_x0 = data_files[0].file_path().to_string();
    let path_x1 = data_files[4].file_path().to_string();
    let table = append_files(&catalog, &table, data_files).await;

    let pd_x1 = write_position_delete_file(&table, 1, &[(path_x1, 0)]).await;
    let table = add_deletes(&catalog, &table, vec![pd_x1]).await;
    let pd_x0 = write_position_delete_file(&table, 0, &[(path_x0, 0)]).await;
    let table = add_deletes(&catalog, &table, vec![pd_x0]).await;

    RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .partial_progress(true)
        .partial_progress_max_commits(2)
        .rewrite_job_order(RewriteJobOrder::FilesDesc)
        .execute(&catalog)
        .await
        .expect("rewrite data files");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");

    let last = table
        .metadata()
        .current_snapshot()
        .expect("head snapshot")
        .as_ref()
        .clone();
    let batch_one = table
        .metadata()
        .snapshot_by_id(last.parent_snapshot_id().expect("batch-2 parent id"))
        .expect("batch-1 snapshot")
        .as_ref()
        .clone();
    let seed_head = table
        .metadata()
        .snapshot_by_id(batch_one.parent_snapshot_id().expect("batch-1 parent id"))
        .expect("seed head snapshot");
    assert_eq!(
        seed_head.summary().operation,
        Operation::Delete,
        "exactly two partial-progress commits follow the seeded delete commit"
    );

    assert_eq!(batch_one.summary().operation, Operation::Replace);
    assert_replace_totals_match_live_files(&table, &batch_one).await;
    assert_rdf_key_set(&props_of(&batch_one), false, "rdf partial batch-1");

    assert_replace_totals_match_live_files(&table, &last).await;
    assert_rdf_key_set(&props_of(&last), true, "rdf partial batch-2");
    assert_eq!(
        prop_u64(&props_of(&last), "removed-position-delete-files"),
        1,
        "batch-2 expires the seq-2 position delete; the seq-3 delete survives because \
         batch-1's compacted file carries the starting sequence number"
    );
    assert_eq!(
        prop_u64(&props_of(&last), "removed-delete-files"),
        1,
        "exactly one delete file expires in batch-2"
    );

    let before = props_of(&batch_one);
    let after = props_of(&last);
    for (total, added, removed) in [
        ("total-data-files", "added-data-files", "deleted-data-files"),
        (
            "total-delete-files",
            "added-delete-files",
            "removed-delete-files",
        ),
        ("total-records", "added-records", "deleted-records"),
        ("total-files-size", "added-files-size", "removed-files-size"),
        (
            "total-position-deletes",
            "added-position-deletes",
            "removed-position-deletes",
        ),
        (
            "total-equality-deletes",
            "added-equality-deletes",
            "removed-equality-deletes",
        ),
    ] {
        assert_eq!(
            prop_or_zero(&after, total),
            prop_or_zero(&before, total) + prop_or_zero(&after, added)
                - prop_or_zero(&after, removed),
            "chain {total} = parent + added - removed through batch-2"
        );
    }
}

#[tokio::test]
async fn rdf_bucketed_table_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_bucketed_table(&catalog, FormatVersion::V2).await;

    let mut data_files = Vec::new();
    for index in 0..4i64 {
        data_files.push(
            write_current_spec_file(
                &table,
                &format!("bdata-{index}"),
                &rows(PARTITION, index * 250, 250),
            )
            .await,
        );
    }
    let deleted_path = data_files[0].file_path().to_string();
    let table = append_files(&catalog, &table, data_files).await;

    let bucket_partition = Struct::from_iter([Some(literal_from_long_transform(
        Transform::Bucket(8),
        PARTITION,
    ))]);
    let mut table = table;
    for merge in 0..3i64 {
        let merge_file = write_current_spec_file(
            &table,
            &format!("bmerge-{merge}"),
            &rows(PARTITION, 1000 + merge * 200, 200),
        )
        .await;
        let delete_file =
            write_pos_del_in_partition(&table, &bucket_partition, &[(deleted_path.clone(), merge)])
                .await;
        table = merge_commit(&catalog, &table, vec![merge_file], vec![delete_file]).await;
    }

    let action = RewriteDataFiles::new(table.clone()).rewrite_all(true);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    assert_rdf_key_set(&props_of(&snapshot), true, "rdf bucketed");
    assert_eq!(
        prop_u64(&props_of(&snapshot), "changed-partition-count"),
        1,
        "one bucket partition was rewritten"
    );
}

async fn write_pos_del_in_partition(
    table: &Table,
    partition: &Struct,
    deletes: &[(String, i64)],
) -> DataFile {
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};

    use crate::spec::PartitionKey;
    use crate::writer::base_writer::position_delete_writer::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};

    let config = PositionDeleteWriterConfig::new().expect("pos-del config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location gen");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        partition.clone(),
    )
    .expect("partition key");
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key))
        .await
        .expect("pos-del writer");

    let paths: Vec<&str> = deletes.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("pos-del batch");
    writer.write(batch).await.expect("write pos-del");
    writer
        .close()
        .await
        .expect("close pos-del")
        .into_iter()
        .next()
        .expect("pos-del file")
}

#[tokio::test]
async fn rdf_v3_dv_replace_summary_reports_removed_dvs() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    let mut data_files = Vec::new();
    for index in 0..4i64 {
        data_files.push(
            write_data_file(
                &table,
                &format!("dvdata-{index}.parquet"),
                PARTITION,
                &rows(PARTITION, index * 250, 250),
            )
            .await,
        );
    }
    let positions: Vec<u64> = (0..25).collect();
    let paths: Vec<String> = data_files
        .iter()
        .map(|file| file.file_path().to_string())
        .collect();
    let targets: Vec<(&str, &[u64])> = paths
        .iter()
        .map(|path| (path.as_str(), positions.as_slice()))
        .collect();
    let table = append_files(&catalog, &table, data_files).await;
    let dvs = write_dv(&table, PARTITION, &targets).await;
    let table = add_deletes(&catalog, &table, dvs).await;

    let action = RewriteDataFiles::new(table.clone()).rewrite_all(true);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();
    let props = props_of(&snapshot);

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    let mut required = rdf_required_keys(false);
    required.extend([
        "removed-dvs",
        "removed-delete-files",
        "removed-position-deletes",
    ]);
    assert_summary_keys(
        &props,
        &required,
        &[
            "removed-position-delete-files",
            "added-delete-files",
            "added-dvs",
            "added-position-delete-files",
            "added-position-deletes",
            "added-equality-delete-files",
            "added-equality-deletes",
            "removed-equality-delete-files",
            "removed-equality-deletes",
            "entries-processed",
        ],
        "rdf v3 dv",
    );
    assert_eq!(prop_u64(&props, "removed-dvs"), 4, "four DV blobs dropped");
    assert_eq!(
        prop_u64(&props, "removed-position-deletes"),
        100,
        "100 positions were deleted across the four DVs"
    );
    assert_eq!(prop_u64(&props, "total-records"), 900);
    assert_eq!(prop_u64(&props, "total-delete-files"), 0);
    assert_eq!(prop_u64(&props, "total-position-deletes"), 0);
}

#[tokio::test]
async fn rdf_v3_rewrite_all_removes_equality_and_dv_delete_files() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    let mut data_files = Vec::new();
    for index in 0..4i64 {
        data_files.push(
            write_data_file(
                &table,
                &format!("edata-{index}.parquet"),
                PARTITION,
                &rows(PARTITION, index * 250, 250),
            )
            .await,
        );
    }
    let positions: Vec<u64> = (0..25).collect();
    let paths: Vec<String> = data_files
        .iter()
        .map(|file| file.file_path().to_string())
        .collect();
    let targets: Vec<(&str, &[u64])> = paths
        .iter()
        .map(|path| (path.as_str(), positions.as_slice()))
        .collect();
    let table = append_files(&catalog, &table, data_files).await;
    let eq_del = write_equality_delete_file(&table, PARTITION, &[1000, 1001]).await;
    let table = add_deletes(&catalog, &table, vec![eq_del]).await;
    let dvs = write_dv(&table, PARTITION, &targets).await;
    let table = add_deletes(&catalog, &table, dvs).await;

    let action = RewriteDataFiles::new(table.clone()).rewrite_all(true);
    let table = rewrite_and_reload(&catalog, action, &table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();
    let props = props_of(&snapshot);

    assert_replace_totals_match_live_files(&table, &snapshot).await;
    let mut required = rdf_required_keys(false);
    required.extend([
        "removed-dvs",
        "removed-delete-files",
        "removed-position-deletes",
        "removed-equality-delete-files",
        "removed-equality-deletes",
    ]);
    assert_summary_keys(
        &props,
        &required,
        &[
            "removed-position-delete-files",
            "added-delete-files",
            "added-dvs",
            "added-position-delete-files",
            "added-position-deletes",
            "added-equality-delete-files",
            "added-equality-deletes",
            "entries-processed",
        ],
        "rdf v3 dv+eq",
    );
    assert_eq!(prop_u64(&props, "removed-dvs"), 4);
    assert_eq!(prop_u64(&props, "removed-equality-delete-files"), 1);
    assert_eq!(prop_u64(&props, "removed-delete-files"), 5);
    assert_eq!(prop_u64(&props, "total-equality-deletes"), 0);
}

async fn seed_rpd_table(catalog: &impl Catalog, delete_commits: i64) -> Table {
    let table = create_partitioned_table(catalog, FormatVersion::V2).await;
    let data_a =
        write_data_file(&table, "rpd-a.parquet", PARTITION, &rows(PARTITION, 0, 250)).await;
    let data_b = write_data_file(
        &table,
        "rpd-b.parquet",
        PARTITION,
        &rows(PARTITION, 250, 250),
    )
    .await;
    let deleted_path = data_a.file_path().to_string();
    let mut table = append_files(catalog, &table, vec![data_a, data_b]).await;
    for commit in 0..delete_commits {
        let delete_file =
            write_position_delete_file(&table, PARTITION, &[(deleted_path.clone(), commit)]).await;
        table = add_deletes(catalog, &table, vec![delete_file]).await;
    }
    table
}

const RPD_BASE_KEYS: &[&str] = &[
    "added-delete-files",
    "added-files-size",
    "added-position-deletes",
    "changed-partition-count",
    "removed-delete-files",
    "removed-files-size",
    "removed-position-deletes",
    "manifests-created",
    "manifests-kept",
    "manifests-replaced",
    "total-data-files",
    "total-delete-files",
    "total-records",
    "total-files-size",
    "total-position-deletes",
    "total-equality-deletes",
];

const RPD_FORBIDDEN: &[&str] = &[
    "added-data-files",
    "added-records",
    "deleted-data-files",
    "deleted-records",
    "added-equality-delete-files",
    "added-equality-deletes",
    "removed-equality-delete-files",
    "removed-equality-deletes",
    "entries-processed",
];

#[tokio::test]
async fn rpd_v2_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_rpd_table(&catalog, 3).await;

    RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("rewrite position deletes");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();
    let props = props_of(&snapshot);

    assert_eq!(snapshot.summary().operation, Operation::Replace);
    let mut required = RPD_BASE_KEYS.to_vec();
    required.push("added-position-delete-files");
    required.push("removed-position-delete-files");
    let mut forbidden = RPD_FORBIDDEN.to_vec();
    forbidden.extend(["added-dvs", "removed-dvs"]);
    assert_summary_keys(&props, &required, &forbidden, "rpd v2");

    assert_eq!(prop_u64(&props, "added-delete-files"), 1);
    assert_eq!(prop_u64(&props, "added-position-delete-files"), 1);
    assert_eq!(prop_u64(&props, "added-position-deletes"), 3);
    assert_eq!(prop_u64(&props, "removed-delete-files"), 3);
    assert_eq!(prop_u64(&props, "removed-position-delete-files"), 3);
    assert_eq!(prop_u64(&props, "removed-position-deletes"), 3);
    assert_eq!(prop_u64(&props, "total-delete-files"), 1);
    assert_eq!(prop_u64(&props, "total-position-deletes"), 3);
    assert_eq!(prop_u64(&props, "total-data-files"), 2);
    assert_eq!(prop_u64(&props, "total-records"), 500);
}

#[tokio::test]
async fn rpd_v3_replace_summary_reports_added_dv() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_rpd_table(&catalog, 2).await;
    let table = upgrade_to_v3(&catalog, &table).await;

    RewritePositionDeleteFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("rewrite position deletes");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .clone();
    let props = props_of(&snapshot);

    assert_eq!(snapshot.summary().operation, Operation::Replace);
    let mut required = RPD_BASE_KEYS.to_vec();
    required.extend(["added-dvs", "removed-position-delete-files"]);
    let mut forbidden = RPD_FORBIDDEN.to_vec();
    forbidden.extend(["added-position-delete-files", "removed-dvs"]);
    assert_summary_keys(&props, &required, &forbidden, "rpd v3");

    assert_eq!(prop_u64(&props, "added-delete-files"), 1);
    assert_eq!(prop_u64(&props, "added-dvs"), 1);
    assert_eq!(prop_u64(&props, "added-position-deletes"), 2);
    assert_eq!(prop_u64(&props, "removed-delete-files"), 2);
    assert_eq!(prop_u64(&props, "removed-position-delete-files"), 2);
    assert_eq!(prop_u64(&props, "removed-position-deletes"), 2);
    assert_eq!(prop_u64(&props, "total-delete-files"), 1);
    assert_eq!(prop_u64(&props, "total-position-deletes"), 2);
}

const RM_REQUIRED_KEYS: &[&str] = &[
    "changed-partition-count",
    "entries-processed",
    "manifests-created",
    "manifests-kept",
    "manifests-replaced",
    "total-data-files",
    "total-delete-files",
    "total-records",
    "total-files-size",
    "total-position-deletes",
    "total-equality-deletes",
];

const RM_FORBIDDEN_KEYS: &[&str] = &[
    "added-data-files",
    "added-records",
    "added-files-size",
    "added-delete-files",
    "added-dvs",
    "added-position-delete-files",
    "added-position-deletes",
    "added-equality-delete-files",
    "added-equality-deletes",
    "deleted-data-files",
    "deleted-records",
    "removed-delete-files",
    "removed-dvs",
    "removed-files-size",
    "removed-position-delete-files",
    "removed-position-deletes",
    "removed-equality-delete-files",
    "removed-equality-deletes",
];

fn assert_rm_totals_carried(snapshot: &Snapshot, parent: &Snapshot) {
    let props = props_of(snapshot);
    let parent_props = props_of(parent);
    for key in [
        "total-data-files",
        "total-delete-files",
        "total-records",
        "total-files-size",
        "total-position-deletes",
        "total-equality-deletes",
    ] {
        assert_eq!(
            prop_u64(&props, key),
            prop_u64(&parent_props, key),
            "{key} carried unchanged through manifest rewrite"
        );
    }
    assert_eq!(snapshot.summary().operation, Operation::Replace);
    assert_eq!(
        prop_u64(&props, "changed-partition-count"),
        0,
        "manifest rewrite changes no partitions"
    );
}

#[tokio::test]
async fn rm_data_only_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let mut table = table;
    for index in 0..3i64 {
        let file = write_data_file(
            &table,
            &format!("rm-{index}.parquet"),
            PARTITION,
            &rows(PARTITION, index * 250, 250),
        )
        .await;
        table = append_files(&catalog, &table, vec![file]).await;
    }

    let tx = Transaction::new(&table);
    let action = tx.rewrite_manifests().cluster_by(|_| "all".to_string());
    let tx = action.apply(tx).expect("apply rewrite manifests");
    let table = tx.commit(&catalog).await.expect("commit rewrite manifests");

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("head snapshot")
        .as_ref()
        .clone();
    let parent = table
        .metadata()
        .snapshot_by_id(snapshot.parent_snapshot_id().expect("parent id"))
        .expect("parent snapshot")
        .as_ref()
        .clone();
    let props = props_of(&snapshot);

    assert_summary_keys(&props, RM_REQUIRED_KEYS, RM_FORBIDDEN_KEYS, "rm data-only");
    assert_rm_totals_carried(&snapshot, &parent);
    assert_eq!(prop_u64(&props, "manifests-created"), 1);
    assert_eq!(prop_u64(&props, "manifests-kept"), 0);
    assert_eq!(prop_u64(&props, "manifests-replaced"), 3);
    assert_eq!(prop_u64(&props, "entries-processed"), 3);
}

#[tokio::test]
async fn rm_delete_manifests_replace_summary_matches_java_keys() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let data_a = write_data_file(
        &table,
        "rmdel-a.parquet",
        PARTITION,
        &rows(PARTITION, 0, 250),
    )
    .await;
    let deleted_path = data_a.file_path().to_string();
    let mut table = append_files(&catalog, &table, vec![data_a]).await;
    let data_b = write_data_file(
        &table,
        "rmdel-b.parquet",
        PARTITION,
        &rows(PARTITION, 250, 250),
    )
    .await;
    table = append_files(&catalog, &table, vec![data_b]).await;
    let delete_file = write_position_delete_file(&table, PARTITION, &[(deleted_path, 0)]).await;
    let table = add_deletes(&catalog, &table, vec![delete_file]).await;

    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .cluster_by(|_| "all".to_string())
        .rewrite_delete_manifests(true);
    let tx = action.apply(tx).expect("apply rewrite manifests");
    let table = tx.commit(&catalog).await.expect("commit rewrite manifests");

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("head snapshot")
        .as_ref()
        .clone();
    let parent = table
        .metadata()
        .snapshot_by_id(snapshot.parent_snapshot_id().expect("parent id"))
        .expect("parent snapshot")
        .as_ref()
        .clone();
    let props = props_of(&snapshot);

    assert_summary_keys(
        &props,
        RM_REQUIRED_KEYS,
        RM_FORBIDDEN_KEYS,
        "rm with deletes",
    );
    assert_rm_totals_carried(&snapshot, &parent);
    assert_eq!(prop_u64(&props, "manifests-created"), 2);
    assert_eq!(prop_u64(&props, "manifests-kept"), 0);
    assert_eq!(prop_u64(&props, "manifests-replaced"), 3);
    assert_eq!(prop_u64(&props, "entries-processed"), 3);
}
