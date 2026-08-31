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
use crate::Catalog;
use crate::error::ErrorKind;
use crate::transaction::{ApplyTransactionAction, Transaction};

/// Deleting a file that is NOT in the current snapshot must error (Java `failMissingDeletePaths`) and
/// must not add the added file. A silent drop keeps the add and loses the removal.
#[tokio::test]
async fn test_rewrite_delete_absent_file_errors() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table = append_files(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;

    let tx = Transaction::new(&table);
    let action = tx.rewrite_files(vec![data_file("test/does-not-exist.parquet", 0)], vec![
        data_file("test/b.parquet", 0),
    ]);
    let tx = action.apply(tx).unwrap();
    let error = tx
        .commit(&catalog)
        .await
        .expect_err("absent delete file must error");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.message().contains("Missing required files to delete"),
        "unexpected error message: {}",
        error.message()
    );

    // The failed rewrite did not add b.parquet.
    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        live_file_paths(&reloaded).await,
        HashSet::from(["test/a.parquet".to_string()])
    );
}
