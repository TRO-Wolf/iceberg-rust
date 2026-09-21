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
    ManifestFile, PrimitiveLiteral, Struct,
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

fn wide_file(path: &str, spec_id: i32, x: i64, y: i64) -> DataFile {
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
        .expect("build wide fixture data file")
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

async fn live_paths_in_order(table: &Table, manifest: &ManifestFile) -> Vec<String> {
    manifest
        .load_manifest(table.file_io())
        .await
        .expect("manifest loads")
        .entries()
        .iter()
        .filter(|entry| entry.is_alive())
        .map(|entry| entry.file_path().to_string())
        .collect()
}

async fn live_keys_in_order(table: &Table, manifest: &ManifestFile) -> Vec<i64> {
    manifest
        .load_manifest(table.file_io())
        .await
        .expect("manifest loads")
        .entries()
        .iter()
        .filter(|entry| entry.is_alive())
        .map(
            |entry| match entry.data_file().partition().fields()[0].clone() {
                Some(Literal::Primitive(PrimitiveLiteral::Long(value))) => value,
                other => panic!("fixture partition holds a long, got {other:?}"),
            },
        )
        .collect()
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

async fn rewrite_with_sort(table: &Table, catalog: &impl Catalog, columns: Vec<String>) -> Table {
    let tx = Transaction::new(table);
    let action = tx
        .rewrite_manifests()
        .sort_by_columns(columns)
        .expect("valid columns build");
    let tx = action.apply(tx).expect("apply rewrite");
    tx.commit(catalog).await.expect("commit rewrite")
}

#[tokio::test]
async fn sort_by_columns_sorts_entries_into_one_manifest() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/sort-c.parquet", 1)]).await;
    let table = append_files(&catalog, &table, vec![data_file("test/sort-a.parquet", 0)]).await;
    let table = append_files(&catalog, &table, vec![data_file("test/sort-b.parquet", 0)]).await;

    let table = rewrite_with_sort(&table, &catalog, vec!["x".to_string()]).await;

    let after = data_manifests(&table).await;
    assert_eq!(after.len(), 1, "the default target packs one manifest");
    assert_eq!(
        live_paths_in_order(&table, &after[0]).await,
        vec![
            "test/sort-a.parquet".to_string(),
            "test/sort-b.parquet".to_string(),
            "test/sort-c.parquet".to_string(),
        ],
        "entries land in partition-value order, stable within a key"
    );
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/sort-a.parquet".to_string(),
            "test/sort-b.parquet".to_string(),
            "test/sort-c.parquet".to_string(),
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
        Some("1"),
        "one sorted manifest is created"
    );
}

#[tokio::test]
async fn sort_by_columns_packs_three_keys_into_two_manifests() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![
        data_file("test/sort-f1.parquet", 2),
        data_file("test/sort-f2.parquet", 0),
    ])
    .await;
    let table = append_files(&catalog, &table, vec![
        data_file("test/sort-f3.parquet", 1),
        data_file("test/sort-f4.parquet", 2),
    ])
    .await;
    let table = append_files(&catalog, &table, vec![
        data_file("test/sort-f5.parquet", 0),
        data_file("test/sort-f6.parquet", 1),
        data_file("test/sort-f7.parquet", 0),
    ])
    .await;
    let total: u64 = data_manifests(&table)
        .await
        .iter()
        .map(|m| m.manifest_length.max(0) as u64)
        .sum();
    assert!(total >= 2, "the fixture carries measurable bytes");
    let table = set_table_property(
        &catalog,
        &table,
        "commit.manifest.target-size-bytes",
        &total.div_ceil(2).to_string(),
    )
    .await;

    let table = rewrite_with_sort(&table, &catalog, vec!["x".to_string()]).await;

    let after = data_manifests(&table).await;
    assert_eq!(after.len(), 2, "ceil(total/target) packs two manifests");
    let mut groups: Vec<Vec<i64>> = Vec::new();
    for manifest in &after {
        groups.push(live_keys_in_order(&table, manifest).await);
    }
    groups.sort_by_key(|group| group[0]);
    assert_eq!(
        groups,
        vec![vec![0, 0, 0, 1, 1], vec![2, 2]],
        "each manifest holds a contiguous run of the sorted keys"
    );
    let first: HashSet<i64> = groups[0].iter().copied().collect();
    let second: HashSet<i64> = groups[1].iter().copied().collect();
    assert!(
        first.intersection(&second).next().is_none(),
        "the two manifests cover non-overlapping key ranges"
    );
    assert_eq!(
        live_file_paths(&table).await.len(),
        7,
        "all seven files stay live"
    );
    assert_eq!(
        summary_prop(&table, "manifests-replaced").as_deref(),
        Some("3"),
        "all three source manifests are replaced"
    );
    assert_eq!(
        summary_prop(&table, "manifests-created").as_deref(),
        Some("2"),
        "two sorted manifests are created"
    );
}

#[tokio::test]
async fn sort_by_columns_compares_the_second_key() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let tx = Transaction::new(&table);
    let action = tx.update_partition_spec().add_field("y");
    let tx = action.apply(tx).expect("apply spec evolution");
    let table = tx.commit(&catalog).await.expect("commit spec evolution");
    let spec_id = table.metadata().default_partition_spec_id();
    assert_ne!(spec_id, 0, "evolution creates a new default spec");
    let table = append_files(&catalog, &table, vec![wide_file(
        "test/sort-y1.parquet",
        spec_id,
        0,
        1,
    )])
    .await;
    let table = append_files(&catalog, &table, vec![wide_file(
        "test/sort-y0.parquet",
        spec_id,
        0,
        0,
    )])
    .await;

    let table = rewrite_with_sort(&table, &catalog, vec!["x".to_string(), "y".to_string()]).await;

    let after = data_manifests(&table).await;
    assert_eq!(after.len(), 1, "the default target packs one manifest");
    assert_eq!(
        live_paths_in_order(&table, &after[0]).await,
        vec![
            "test/sort-y0.parquet".to_string(),
            "test/sort-y1.parquet".to_string(),
        ],
        "the second key orders entries the first key ties"
    );
}

#[tokio::test]
async fn sort_by_columns_rejects_empty_input() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let tx = Transaction::new(&table);
    let Err(error) = tx.rewrite_manifests().sort_by_columns(Vec::new()) else {
        panic!("empty columns must fail");
    };
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(error.message(), "sort_by must not be empty when provided");
}

#[tokio::test]
async fn sort_by_columns_names_an_unknown_column() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/sort-a.parquet", 0)]).await;
    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .sort_by_columns(vec!["nope".to_string()])
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
    assert_eq!(unchanged.len(), 1, "a refused rewrite keeps its manifests");
}

#[tokio::test]
async fn sort_by_columns_rejects_a_zero_target_size() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/sort-a.parquet", 0)]).await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.target-size-bytes", "0").await;
    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .sort_by_columns(vec!["x".to_string()])
        .expect("valid columns build");
    let tx = action.apply(tx).expect("apply rewrite");
    let error = tx
        .commit(&catalog)
        .await
        .expect_err("a zero target size must fail the commit");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("commit.manifest.target-size-bytes"),
        "the error names the property, got: {}",
        error.message()
    );
}

#[tokio::test]
async fn last_setter_wins_between_sort_and_columns() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/sort-a.parquet", 0)]).await;
    let table = append_files(&catalog, &table, vec![data_file("test/sort-b.parquet", 1)]).await;
    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .cluster_by_columns(vec!["x".to_string()])
        .expect("columns build")
        .sort_by_columns(vec!["x".to_string()])
        .expect("sort builds");
    let tx = action.apply(tx).expect("apply rewrite");
    let table = tx.commit(&catalog).await.expect("commit rewrite");
    assert_eq!(
        data_manifests(&table).await.len(),
        1,
        "sort set last packs one manifest"
    );

    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/sort-a.parquet", 0)]).await;
    let table = append_files(&catalog, &table, vec![data_file("test/sort-b.parquet", 1)]).await;
    let tx = Transaction::new(&table);
    let action = tx
        .rewrite_manifests()
        .sort_by_columns(vec!["x".to_string()])
        .expect("sort builds")
        .cluster_by_columns(vec!["x".to_string()])
        .expect("columns build");
    let tx = action.apply(tx).expect("apply rewrite");
    let table = tx.commit(&catalog).await.expect("commit rewrite");
    assert_eq!(
        data_manifests(&table).await.len(),
        2,
        "columns set last cluster two manifests"
    );
}
