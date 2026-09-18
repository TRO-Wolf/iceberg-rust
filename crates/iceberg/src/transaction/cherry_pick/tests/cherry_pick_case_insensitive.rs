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

use super::{
    Catalog, ErrorKind, Operation, REPLACE_PARTITIONS_PROP, SOURCE_SNAPSHOT_ID_PROP, Table,
    Transaction, append, cherry_pick, cherry_pick_err, current_summary_prop, data_file,
    live_file_paths, make_v3_minimal_table_in_catalog, new_memory_catalog, set_current,
    snapshot_count,
};
use crate::transaction::ApplyTransactionAction;

async fn stage_replace_partitions_with_marker(
    catalog: &impl Catalog,
    table: &Table,
    marker_value: &str,
) -> (Table, i64) {
    let table = append(catalog, table, vec![
        data_file("test/a.parquet", 0),
        data_file("test/b.parquet", 1),
    ])
    .await;
    let s0 = table.metadata().current_snapshot_id().unwrap();

    let tx = Transaction::new(&table);
    let action = tx
        .replace_partitions()
        .add_file(data_file("test/a2.parquet", 0))
        .set_snapshot_properties(HashMap::from([(
            REPLACE_PARTITIONS_PROP.to_string(),
            marker_value.to_string(),
        )]));
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(catalog).await.unwrap();
    let staged_id = table.metadata().current_snapshot_id().unwrap();

    let table = set_current(catalog, &table, s0).await;
    let table = append(catalog, &table, vec![data_file("test/c.parquet", 2)]).await;
    (table, staged_id)
}

#[tokio::test]
async fn test_cherrypick_replace_partitions_marker_is_case_insensitive() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let (table, staged_id) = stage_replace_partitions_with_marker(&catalog, &table, "TRUE").await;

    let before = snapshot_count(&table);
    let table = cherry_pick(&catalog, &table, staged_id).await;

    assert_eq!(
        snapshot_count(&table),
        before + 1,
        "Java parseBoolean reads \"TRUE\" as true: the staged overwrite replays like the \"true\" cell"
    );
    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        Operation::Overwrite,
        "the publish records the picked OVERWRITE operation"
    );
    assert_eq!(
        current_summary_prop(&table, SOURCE_SNAPSHOT_ID_PROP),
        Some(staged_id.to_string())
    );
    let live = live_file_paths(&table).await;
    assert!(
        live.contains("test/a2.parquet"),
        "the replayed add is live: {live:?}"
    );
    assert!(
        !live.contains("test/a.parquet"),
        "the replaced-partition old file is dropped: {live:?}"
    );
    assert!(live.contains("test/b.parquet") && live.contains("test/c.parquet"));
}

#[tokio::test]
async fn test_cherrypick_replace_partitions_marker_leading_space_is_not_replace() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let (table, staged_id) = stage_replace_partitions_with_marker(&catalog, &table, " true").await;

    let err = cherry_pick_err(&catalog, &table, staged_id).await;
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(!err.retryable(), "a validation conflict is non-retryable");
    assert!(
        err.message()
            .contains("not append, dynamic overwrite, or fast-forward"),
        "parseBoolean does not trim: \" true\" is not a replace: {}",
        err.message()
    );
}
