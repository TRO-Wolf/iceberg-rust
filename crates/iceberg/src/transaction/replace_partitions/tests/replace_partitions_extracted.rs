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

use super::{
    append_files, data_file, live_file_paths, make_v3_minimal_table_in_catalog, new_memory_catalog,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};

/// Pins: replacing a single partition with MULTIPLE new files removes the old file and adds all the
/// new ones in that partition. Risk: only the first added file landing (lost new files).
#[tokio::test]
async fn test_replace_partition_with_multiple_new_files() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;

    let table = append_files(&catalog, &table, vec![
        data_file("test/old.parquet", 0),
        data_file("test/keep.parquet", 1),
    ])
    .await;

    // Replace x=0 with TWO files; keep.parquet in x=1 must survive.
    let tx = Transaction::new(&table);
    let action = tx.replace_partitions().add_files(vec![
        data_file("test/new1.parquet", 0),
        data_file("test/new2.parquet", 0),
    ]);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/new1.parquet".to_string(),
            "test/new2.parquet".to_string(),
            "test/keep.parquet".to_string(),
        ]),
        "x=0 now holds new1+new2 (old replaced), x=1 keep survives"
    );
}

async fn replace_commit_with(properties: Option<HashMap<String, String>>) -> Table {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/old.parquet", 0)]).await;

    let tx = Transaction::new(&table);
    let action = tx
        .replace_partitions()
        .add_file(data_file("test/new.parquet", 0));
    let action = match properties {
        Some(properties) => action.set_snapshot_properties(properties),
        None => action,
    };
    let tx = action.apply(tx).unwrap();
    tx.commit(&catalog).await.unwrap()
}

fn summary_property(table: &Table, key: &str) -> Option<String> {
    table
        .metadata()
        .current_snapshot()
        .unwrap()
        .summary()
        .additional_properties
        .get(key)
        .cloned()
}

#[tokio::test]
async fn test_replace_partitions_summary_caller_false_wins() {
    let table = replace_commit_with(Some(HashMap::from([(
        "replace-partitions".to_string(),
        "false".to_string(),
    )])))
    .await;

    assert_eq!(
        summary_property(&table, "replace-partitions").as_deref(),
        Some("false"),
        "a caller-provided replace-partitions=false must override the action default (Java `set` order)"
    );
}

#[tokio::test]
async fn test_replace_partitions_summary_caller_true_wins() {
    let table = replace_commit_with(Some(HashMap::from([(
        "replace-partitions".to_string(),
        "true".to_string(),
    )])))
    .await;

    assert_eq!(
        summary_property(&table, "replace-partitions").as_deref(),
        Some("true")
    );
}

#[tokio::test]
async fn test_replace_partitions_summary_unrelated_key_keeps_default() {
    let table = replace_commit_with(Some(HashMap::from([(
        "k".to_string(),
        "v".to_string(),
    )])))
    .await;

    assert_eq!(
        summary_property(&table, "replace-partitions").as_deref(),
        Some("true"),
        "the default marker still lands when the caller sets only unrelated keys"
    );
    assert_eq!(summary_property(&table, "k").as_deref(), Some("v"));
}

#[tokio::test]
async fn test_replace_partitions_summary_default_is_true() {
    let table = replace_commit_with(None).await;

    assert_eq!(
        summary_property(&table, "replace-partitions").as_deref(),
        Some("true"),
        "no caller properties still lands the replace-partitions marker"
    );
}
