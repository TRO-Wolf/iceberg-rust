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

use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestContentType,
    ManifestStatus, Struct,
};
use crate::table::Table;
use crate::transaction::tests::make_v2_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};

fn data_file(path: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(0))]))
        .build()
        .expect("build the fixture data file")
}

fn position_delete_file(path: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(0))]))
        .build()
        .expect("build the fixture delete file")
}

async fn manifest_contents(table: &Table) -> Vec<ManifestContentType> {
    manifest_list(table)
        .await
        .into_iter()
        .map(|(content, _)| content)
        .collect()
}

async fn manifest_list(table: &Table) -> Vec<(ManifestContentType, Vec<String>)> {
    let metadata = table.metadata();
    let snapshot = metadata
        .current_snapshot()
        .expect("the committed table has a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await
        .expect("load the manifest list");

    let mut described = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("load the manifest");
        let mut live: Vec<String> = manifest
            .entries()
            .iter()
            .filter(|entry| entry.status() != ManifestStatus::Deleted)
            .map(|entry| entry.file_path().to_string())
            .collect();
        live.sort();
        described.push((manifest_file.content, live));
    }
    described
}

#[tokio::test]
async fn carried_forward_data_manifests_keep_their_source_order() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;

    let transaction = Transaction::new(&table);
    let transaction = transaction
        .fast_append()
        .add_data_files(vec![
            data_file("test/a1.parquet"),
            data_file("test/a2.parquet"),
        ])
        .apply(transaction)
        .expect("apply the first append");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the first append");

    let transaction = Transaction::new(&table);
    let transaction = transaction
        .fast_append()
        .add_data_files(vec![data_file("test/b.parquet")])
        .apply(transaction)
        .expect("apply the second append");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the second append");

    let before: Vec<Vec<String>> = manifest_list(&table)
        .await
        .into_iter()
        .map(|(_, live)| live)
        .collect();
    assert_eq!(
        before,
        vec![vec!["test/b.parquet".to_string()], vec![
            "test/a1.parquet".to_string(),
            "test/a2.parquet".to_string()
        ]],
        "fixture precondition: two data manifests, newest first"
    );

    let transaction = Transaction::new(&table);
    let transaction = transaction
        .delete_files()
        .delete_file("test/a1.parquet".to_string())
        .apply(transaction)
        .expect("apply the delete");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the delete");

    let after: Vec<Vec<String>> = manifest_list(&table)
        .await
        .into_iter()
        .map(|(_, live)| live)
        .collect();
    assert_eq!(
        after,
        vec![vec!["test/b.parquet".to_string()], vec![
            "test/a2.parquet".to_string()
        ]],
        "the carried-forward manifest and the rewritten one kept their source-list order"
    );
}

#[tokio::test]
async fn every_data_manifest_precedes_every_delete_manifest() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;

    let transaction = Transaction::new(&table);
    let transaction = transaction
        .fast_append()
        .add_data_files(vec![data_file("test/d1.parquet")])
        .apply(transaction)
        .expect("apply the append");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the append");

    let transaction = Transaction::new(&table);
    let transaction = transaction
        .row_delta()
        .add_deletes(vec![position_delete_file("test/d1-pos-del.parquet")])
        .apply(transaction)
        .expect("apply the row delta");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the row delta");
    assert_eq!(
        manifest_contents(&table).await,
        vec![ManifestContentType::Data, ManifestContentType::Deletes],
        "fixture precondition: the table now carries a DELETE manifest"
    );

    let transaction = Transaction::new(&table);
    let transaction = transaction
        .fast_append()
        .add_data_files(vec![data_file("test/d2.parquet")])
        .apply(transaction)
        .expect("apply the second append");
    let table = transaction
        .commit(&catalog)
        .await
        .expect("commit the second append");

    let contents = manifest_contents(&table).await;
    assert_eq!(
        contents,
        vec![
            ManifestContentType::Data,
            ManifestContentType::Data,
            ManifestContentType::Deletes
        ],
        "the manifest list is all DATA then all DELETES"
    );
}
