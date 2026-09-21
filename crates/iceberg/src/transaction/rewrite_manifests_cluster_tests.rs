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
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestContentType,
    ManifestFile, Struct,
};
use crate::table::Table;
use crate::transaction::rewrite_manifests::tests::{append_files, current_manifests, summary_prop};
use crate::transaction::tests::make_v2_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};

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
        .expect("build fixture data file")
}

async fn live_file_paths(table: &Table) -> HashSet<String> {
    let mut live = HashSet::new();
    for manifest_file in current_manifests(table).await {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest loads");
        for entry in manifest.entries() {
            if entry.is_alive() {
                live.insert(entry.file_path().to_string());
            }
        }
    }
    live
}

async fn data_manifests(table: &Table) -> Vec<ManifestFile> {
    current_manifests(table)
        .await
        .into_iter()
        .filter(|m| m.content == ManifestContentType::Data)
        .collect()
}

async fn live_paths_in_manifest(table: &Table, manifest: &ManifestFile) -> Vec<String> {
    let mut paths: Vec<String> = manifest
        .load_manifest(table.file_io())
        .await
        .expect("manifest loads")
        .entries()
        .iter()
        .filter(|entry| entry.is_alive())
        .map(|entry| entry.file_path().to_string())
        .collect();
    paths.sort_unstable();
    paths
}

async fn three_manifest_fixture(catalog: &impl Catalog) -> Table {
    let table = make_v2_minimal_table_in_catalog(catalog).await;
    let table = append_files(catalog, &table, vec![data_file(
        "test/cluster-a.parquet",
        0,
    )])
    .await;
    let table = append_files(catalog, &table, vec![data_file(
        "test/cluster-b.parquet",
        0,
    )])
    .await;
    let table = append_files(catalog, &table, vec![data_file(
        "test/cluster-c.parquet",
        1,
    )])
    .await;
    let before = data_manifests(&table).await;
    assert_eq!(before.len(), 3, "one manifest per append");
    table
}

async fn rewrite_with_columns(
    table: &Table,
    catalog: &impl Catalog,
    columns: Vec<String>,
) -> Table {
    let tx = Transaction::new(table);
    let action = tx
        .rewrite_manifests()
        .cluster_by_columns(columns)
        .expect("valid columns build");
    let tx = action.apply(tx).expect("apply rewrite");
    tx.commit(catalog).await.expect("commit rewrite")
}

#[tokio::test]
async fn cluster_by_columns_groups_entries_by_partition_value() {
    let catalog = new_memory_catalog().await;
    let table = three_manifest_fixture(&catalog).await;

    let table = rewrite_with_columns(&table, &catalog, vec!["x".to_string()]).await;

    let after = data_manifests(&table).await;
    assert_eq!(after.len(), 2, "two partition values yield two manifests");
    let mut groups: Vec<Vec<String>> = Vec::new();
    for manifest in &after {
        groups.push(live_paths_in_manifest(&table, manifest).await);
    }
    groups.sort_unstable();
    assert_eq!(
        groups,
        vec![
            vec![
                "test/cluster-a.parquet".to_string(),
                "test/cluster-b.parquet".to_string()
            ],
            vec!["test/cluster-c.parquet".to_string()],
        ],
        "entries sharing x land in one manifest"
    );
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/cluster-a.parquet".to_string(),
            "test/cluster-b.parquet".to_string(),
            "test/cluster-c.parquet".to_string(),
        ]),
        "the live set is unchanged"
    );
    assert_eq!(
        summary_prop(&table, "manifests-replaced").as_deref(),
        Some("3"),
        "all three source manifests are replaced"
    );
    assert_eq!(
        summary_prop(&table, "manifests-created").as_deref(),
        Some("2"),
        "two clustered manifests are created"
    );
}

#[tokio::test]
async fn cluster_by_columns_differs_from_constant_clustering() {
    let catalog = new_memory_catalog().await;
    let keyed = three_manifest_fixture(&catalog).await;
    let keyed = rewrite_with_columns(&keyed, &catalog, vec!["x".to_string()]).await;
    let keyed_groups: Vec<Vec<String>> = {
        let mut groups = Vec::new();
        for manifest in data_manifests(&keyed).await {
            groups.push(live_paths_in_manifest(&keyed, &manifest).await);
        }
        groups.sort_unstable();
        groups
    };

    let flat = three_manifest_fixture(&catalog).await;
    let tx = Transaction::new(&flat);
    let action = tx.rewrite_manifests().cluster_by(|_| "same".to_string());
    let tx = action.apply(tx).expect("apply constant rewrite");
    let flat = tx.commit(&catalog).await.expect("commit constant rewrite");
    let flat_groups: Vec<Vec<String>> = {
        let mut groups = Vec::new();
        for manifest in data_manifests(&flat).await {
            groups.push(live_paths_in_manifest(&flat, &manifest).await);
        }
        groups.sort_unstable();
        groups
    };

    assert_eq!(flat_groups.len(), 1, "a constant key fuses one manifest");
    assert_eq!(keyed_groups.len(), 2, "the column key splits two manifests");
    assert_ne!(
        keyed_groups, flat_groups,
        "the column clustering differs from an unkeyed rewrite"
    );
}

#[tokio::test]
async fn cluster_by_columns_rejects_empty_input() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let tx = Transaction::new(&table);
    let Err(error) = tx.rewrite_manifests().cluster_by_columns(Vec::new()) else {
        panic!("empty columns must fail");
    };
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(error.message(), "sort_by must not be empty when provided");
}

#[tokio::test]
async fn cluster_by_columns_names_an_unknown_column() {
    let catalog = new_memory_catalog().await;
    let table = three_manifest_fixture(&catalog).await;
    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .cluster_by_columns(vec!["nope".to_string()])
        .expect("unknown columns build, resolution runs at commit");
    let tx = action.apply(tx).expect("apply rewrite");
    let error = tx
        .commit(&catalog)
        .await
        .expect_err("an unknown column must fail the commit");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.message().contains("'nope'"),
        "the error names the column, got: {}",
        error.message()
    );
    let unchanged = data_manifests(&table).await;
    assert_eq!(unchanged.len(), 3, "a refused rewrite keeps its manifests");
}

#[tokio::test]
async fn cluster_by_columns_combines_two_partition_values() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let tx = Transaction::new(&table);
    let action = tx.update_partition_spec().add_field("y");
    let tx = action.apply(tx).expect("apply spec evolution");
    let table = tx.commit(&catalog).await.expect("commit spec evolution");
    let spec_id = table.metadata().default_partition_spec_id();
    assert_ne!(spec_id, 0, "evolution creates a new default spec");
    let wide = |path: &str, x: i64, y: i64| {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(spec_id)
            .partition(Struct::from_iter([
                Some(Literal::long(x)),
                Some(Literal::long(y)),
            ]))
            .build()
            .expect("build wide data file")
    };
    let table = append_files(&catalog, &table, vec![wide("test/wide-a.parquet", 0, 0)]).await;
    let table = append_files(&catalog, &table, vec![wide("test/wide-b.parquet", 0, 1)]).await;
    assert_eq!(data_manifests(&table).await.len(), 2);

    let table =
        rewrite_with_columns(&table, &catalog, vec!["x".to_string(), "y".to_string()]).await;
    let after = data_manifests(&table).await;
    assert_eq!(
        after.len(),
        2,
        "files differing only in y stay split under a two-column key"
    );
    let mut groups: Vec<Vec<String>> = Vec::new();
    for manifest in &after {
        groups.push(live_paths_in_manifest(&table, manifest).await);
    }
    groups.sort_unstable();
    assert_eq!(groups, vec![vec!["test/wide-a.parquet".to_string()], vec![
        "test/wide-b.parquet".to_string()
    ],]);
}

#[tokio::test]
async fn last_setter_wins_between_cluster_by_and_columns() {
    let catalog = new_memory_catalog().await;
    let table = three_manifest_fixture(&catalog).await;
    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .cluster_by(|_| "same".to_string())
        .cluster_by_columns(vec!["x".to_string()])
        .expect("columns build");
    let tx = action.apply(tx).expect("apply rewrite");
    let table = tx.commit(&catalog).await.expect("commit rewrite");
    assert_eq!(
        data_manifests(&table).await.len(),
        2,
        "columns set last override the closure"
    );

    let table = three_manifest_fixture(&catalog).await;
    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .cluster_by_columns(vec!["x".to_string()])
        .expect("columns build")
        .cluster_by(|_| "same".to_string());
    let tx = action.apply(tx).expect("apply rewrite");
    let table = tx.commit(&catalog).await.expect("commit rewrite");
    assert_eq!(
        data_manifests(&table).await.len(),
        1,
        "a closure set last overrides the columns"
    );
}
