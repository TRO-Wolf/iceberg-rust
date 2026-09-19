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

use super::RewriteDataFiles;
use super::rewrite_data_files::tests::{
    append_files, create_partitioned_table, local_fs_catalog, write_data_file,
    write_position_delete_file,
};
use crate::Catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestStatus,
    Operation,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};

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

fn summary_props(table: &Table) -> HashMap<String, String> {
    table
        .metadata()
        .current_snapshot()
        .expect("current snapshot")
        .summary()
        .additional_properties
        .clone()
}

fn prop_u64(props: &HashMap<String, String>, key: &str) -> u64 {
    props
        .get(key)
        .unwrap_or_else(|| panic!("missing summary key '{key}' in {props:?}"))
        .parse()
        .unwrap_or_else(|_| panic!("summary key '{key}' is not a number in {props:?}"))
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

async fn live_files(table: &Table, content: ManifestContentType) -> Vec<DataFile> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
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

async fn snapshot_added_files(table: &Table, content: ManifestContentType) -> Vec<DataFile> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
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

async fn assert_replace_totals_match_live_files(table: &Table) {
    let props = summary_props(table);
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .expect("current snapshot")
            .summary()
            .operation,
        Operation::Replace
    );
    let data_files = live_files(table, ManifestContentType::Data).await;
    let delete_files = live_files(table, ManifestContentType::Deletes).await;
    let added_data = snapshot_added_files(table, ManifestContentType::Data).await;
    let added_deletes = snapshot_added_files(table, ManifestContentType::Deletes).await;

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

#[tokio::test]
async fn rdf_replace_summary_counts_added_files_and_live_totals() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = seed_mor_table(&catalog, FormatVersion::V2).await;

    RewriteDataFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect("rewrite data files");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");

    assert_replace_totals_match_live_files(&table).await;
}
