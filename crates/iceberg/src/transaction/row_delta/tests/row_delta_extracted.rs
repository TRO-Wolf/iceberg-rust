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

use super::{
    append_files, make_v2_minimal_table_in_catalog, new_memory_catalog, synthetic_data_file,
    synthetic_delete_file,
};
use crate::spec::Operation;
use crate::transaction::{ApplyTransactionAction, Transaction};

/// A deletes-only row delta is allowed and records `Delete`. The mutant: the producer's
/// empty-commit precondition rejects it.
#[tokio::test]
async fn test_row_delta_add_deletes_only_allowed() {
    let catalog = new_memory_catalog().await;
    let table = make_v2_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![synthetic_data_file(
        "test/a.parquet",
        0,
    )])
    .await;

    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_deletes(vec![synthetic_delete_file("test/a-pos-del.parquet", 0)]);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    assert_eq!(
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .operation,
        Operation::Delete,
        "an add-deletes-only row delta records Delete (Java BaseRowDelta.operation())"
    );
}
