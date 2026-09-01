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
use crate::error::ErrorKind;
use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Operation, Struct,
};
use crate::table::Table;
use crate::transaction::tests::make_v3_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};

fn data_file(path: &str, record_count: u64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(record_count)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(0))]))
        .build()
        .expect("build fixture data file")
}

async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    let tx = action.apply(tx).expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn metadata_avro_count(table: &Table) -> usize {
    let prefix = format!("{}/metadata", table.metadata().location());
    table
        .file_io()
        .list(&prefix)
        .await
        .expect("list metadata directory")
        .into_iter()
        .filter(|info| info.location.ends_with(".avro"))
        .count()
}

async fn live_file_paths(table: &Table) -> HashSet<String> {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("load manifest list");
    let mut live = HashSet::new();
    for manifest_file in manifest_list.entries() {
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

#[tokio::test]
async fn replace_rejects_added_records_greater_than_deleted_records() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let original = data_file("test/original.parquet", 3);
    let table = append_files(&catalog, &table, vec![original.clone()]).await;

    let before_snapshot = table
        .metadata()
        .current_snapshot()
        .expect("append left a current snapshot")
        .snapshot_id();
    let before_metadata = table
        .metadata_location()
        .expect("catalog table has a metadata pointer")
        .to_string();
    let before_avro = metadata_avro_count(&table).await;
    assert!(
        before_avro > 0,
        "append must have written at least one manifest or manifest-list object"
    );

    let replacement = data_file("test/replacement.parquet", 5);
    let tx = Transaction::new(&table);
    let action = tx.rewrite_files(vec![original], vec![replacement]);
    let tx = action.apply(tx).expect("apply rewrite_files");
    let error = tx
        .commit(&catalog)
        .await
        .expect_err("replacing 3 rows with 5 rows must be DataInvalid");

    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        !error.retryable(),
        "the REPLACE record-count guard is not retryable"
    );
    assert_eq!(
        error.message(),
        "Invalid REPLACE operation: 5 added records > 3 replaced records"
    );

    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("reload after refused rewrite");
    assert_eq!(
        reloaded
            .metadata()
            .current_snapshot()
            .expect("snapshot still present")
            .snapshot_id(),
        before_snapshot
    );
    assert_eq!(reloaded.metadata_location(), Some(before_metadata.as_str()));
    assert_eq!(
        metadata_avro_count(&reloaded).await,
        before_avro,
        "refused REPLACE must not write a new manifest or manifest-list object"
    );
    assert_eq!(
        live_file_paths(&reloaded).await,
        HashSet::from(["test/original.parquet".to_string()])
    );
}

#[tokio::test]
async fn replace_commits_when_added_records_equal_deleted_records() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let original = data_file("test/original.parquet", 3);
    let table = append_files(&catalog, &table, vec![original.clone()]).await;

    let replacement = data_file("test/replacement.parquet", 3);
    let tx = Transaction::new(&table);
    let action = tx.rewrite_files(vec![original], vec![replacement]);
    let tx = action.apply(tx).expect("apply equal rewrite");
    let table = tx
        .commit(&catalog)
        .await
        .expect("equal 3-to-3 rewrite commits");

    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .expect("rewrite produced a snapshot")
            .summary()
            .operation,
        Operation::Replace
    );
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from(["test/replacement.parquet".to_string()])
    );
}

#[tokio::test]
async fn replace_commits_when_added_records_trail_deleted_records() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let original = data_file("test/original.parquet", 5);
    let table = append_files(&catalog, &table, vec![original.clone()]).await;

    let replacement = data_file("test/replacement.parquet", 3);
    let tx = Transaction::new(&table);
    let action = tx.rewrite_files(vec![original], vec![replacement]);
    let tx = action.apply(tx).expect("apply shrinking rewrite");
    let table = tx
        .commit(&catalog)
        .await
        .expect("shrinking 5-to-3 rewrite commits");

    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .expect("rewrite produced a snapshot")
            .summary()
            .operation,
        Operation::Replace
    );
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from(["test/replacement.parquet".to_string()])
    );
}

#[tokio::test]
async fn rewrite_manifests_replace_commits_when_record_count_keys_are_absent() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 1)]).await;
    let table = append_files(&catalog, &table, vec![data_file("test/b.parquet", 1)]).await;
    let before = live_file_paths(&table).await;

    let tx = Transaction::new(&table);
    let action = tx.rewrite_manifests().cluster_by(|_| "all".to_string());
    let tx = action.apply(tx).expect("apply rewrite_manifests");
    let table = tx
        .commit(&catalog)
        .await
        .expect("RewriteManifests REPLACE with absent record-count keys must commit");

    let summary = table
        .metadata()
        .current_snapshot()
        .expect("rewrite_manifests produced a snapshot")
        .summary();
    assert_eq!(summary.operation, Operation::Replace);
    assert!(
        !summary.additional_properties.contains_key("added-records"),
        "RewriteManifests must omit added-records so the guard treats it as zero"
    );
    assert!(
        !summary
            .additional_properties
            .contains_key("deleted-records"),
        "RewriteManifests must omit deleted-records so the guard treats it as zero"
    );
    assert_eq!(live_file_paths(&table).await, before);
}

#[tokio::test]
async fn replace_still_refuses_after_the_base_refreshes() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let original = data_file("test/original.parquet", 3);
    let table = append_files(&catalog, &table, vec![original.clone()]).await;

    let replacement = data_file("test/replacement.parquet", 5);
    let tx = Transaction::new(&table);
    let action = tx.rewrite_files(vec![original], vec![replacement]);
    let tx = action.apply(tx).expect("apply rewrite_files");

    let _refreshed = append_files(&catalog, &table, vec![data_file(
        "test/concurrent.parquet",
        1,
    )])
    .await;

    let error = tx
        .commit(&catalog)
        .await
        .expect_err("retried invalid REPLACE after a refreshed base must still be DataInvalid");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        !error.retryable(),
        "the REPLACE record-count guard is not retryable after refresh"
    );
    assert_eq!(
        error.message(),
        "Invalid REPLACE operation: 5 added records > 3 replaced records"
    );
}
