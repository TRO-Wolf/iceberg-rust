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

use std::collections::{BTreeSet, HashMap};

use iceberg::table::Table;

use super::harness::{
    Harness, NS, TBL, harness, live_delete_files, live_ids, load_table, row_positions, run_sql,
    sql_count,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DvEntry {
    pub(crate) container: String,
    pub(crate) offset: Option<i64>,
    pub(crate) size: Option<i64>,
}

pub(crate) async fn dv_entries_by_reference(table: &Table) -> HashMap<String, DvEntry> {
    live_delete_files(table)
        .await
        .into_iter()
        .filter_map(|file| {
            file.referenced_data_file().map(|referenced| {
                (referenced, DvEntry {
                    container: file.file_path().to_string(),
                    offset: file.content_offset(),
                    size: file.content_size_in_bytes(),
                })
            })
        })
        .collect()
}

pub(crate) fn summary_count(table: &Table, key: &str) -> i64 {
    table
        .metadata()
        .current_snapshot()
        .expect("snapshot")
        .summary()
        .additional_properties
        .get(key)
        .map(|value| value.parse::<i64>().expect("numeric summary value"))
        .unwrap_or(0)
}

pub(crate) async fn seed_two_blob_container(harness: &Harness) -> (String, String) {
    run_sql(
        &harness.ctx,
        &format!(
            "INSERT INTO catalog.{NS}.{TBL} VALUES \
             (1, 'a', 'electronics'), (2, 'b', 'electronics'), (3, 'c', 'electronics'), \
             (4, 'd', 'books'), (5, 'e', 'books'), (6, 'f', 'books')"
        ),
    )
    .await;
    let table = load_table(&harness.catalog).await;
    let rows = row_positions(&table).await;
    let touched = rows
        .iter()
        .find(|row| row.id == 1)
        .expect("id 1")
        .file
        .clone();
    let sibling = rows
        .iter()
        .find(|row| row.id == 5)
        .expect("id 5")
        .file
        .clone();
    assert_ne!(
        touched, sibling,
        "id 1 and id 5 must live in different files"
    );
    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 2 OR id = 5"),
    )
    .await;
    assert_eq!(deleted, 2);
    let table = load_table(&harness.catalog).await;
    let entries = dv_entries_by_reference(&table).await;
    assert_eq!(entries.len(), 2, "one DV per referenced file");
    let containers: BTreeSet<&str> = entries
        .values()
        .map(|entry| entry.container.as_str())
        .collect();
    assert_eq!(containers.len(), 1, "both blobs share ONE Puffin container");
    (touched, sibling)
}

#[tokio::test]
async fn touched_blob_moves_and_the_sibling_entry_stays_put() {
    let harness = harness().await;
    let (touched, sibling) = seed_two_blob_container(&harness).await;
    let before = load_table(&harness.catalog).await;
    let entries_before = dv_entries_by_reference(&before).await;
    let sibling_before = entries_before.get(&sibling).expect("sibling DV").clone();
    let touched_before = entries_before.get(&touched).expect("touched DV").clone();

    let deleted = sql_count(
        &harness.ctx,
        &format!("DELETE FROM catalog.{NS}.{TBL} WHERE id = 1"),
    )
    .await;
    assert_eq!(deleted, 1);

    let after = load_table(&harness.catalog).await;
    let entries_after = dv_entries_by_reference(&after).await;
    assert_eq!(entries_after.len(), 2, "still one DV per referenced file");
    let sibling_after = entries_after
        .get(&sibling)
        .expect("sibling DV after")
        .clone();
    let touched_after = entries_after
        .get(&touched)
        .expect("touched DV after")
        .clone();

    assert_eq!(
        sibling_after, sibling_before,
        "untouched sibling entry must keep its container path, offset and size"
    );
    assert_ne!(
        touched_after.container, touched_before.container,
        "the touched blob must move into a NEW container"
    );
    let containers: BTreeSet<&str> = entries_after
        .values()
        .map(|entry| entry.container.as_str())
        .collect();
    assert_eq!(
        containers.len(),
        2,
        "two live containers after the second DELETE"
    );

    assert_eq!(summary_count(&after, "removed-delete-files"), 1);
    assert_eq!(summary_count(&after, "removed-dvs"), 1);
    assert_eq!(summary_count(&after, "added-delete-files"), 1);
    assert_eq!(live_ids(&harness.ctx).await, vec![3, 4, 6]);
}
