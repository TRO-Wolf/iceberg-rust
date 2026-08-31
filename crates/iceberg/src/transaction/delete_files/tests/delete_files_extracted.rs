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

use super::{
    append_files, data_file, live_file_paths, make_v3_minimal_table_in_catalog, new_memory_catalog,
};
use crate::spec::Operation;
use crate::transaction::{ApplyTransactionAction, Transaction};

/// Deleting one of several files in the SAME manifest must remove exactly that file and leave the
/// others. A wrong live set is data loss.
#[tokio::test]
async fn test_delete_files_removes_only_targeted_file_from_live_scan() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;

    let table = append_files(&catalog, &table, vec![
        data_file("test/a.parquet", 0),
        data_file("test/b.parquet", 0),
        data_file("test/c.parquet", 0),
    ])
    .await;
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/a.parquet".to_string(),
            "test/b.parquet".to_string(),
            "test/c.parquet".to_string(),
        ])
    );

    // Delete B.
    let tx = Transaction::new(&table);
    let action = tx.delete_files().delete_file("test/b.parquet");
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        Operation::Delete
    );
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from(["test/a.parquet".to_string(), "test/c.parquet".to_string()])
    );
}
