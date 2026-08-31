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
use crate::spec::{DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Struct};
use crate::transaction::tests::make_v3_minimal_table_in_catalog;
use crate::transaction::{ApplyTransactionAction, Transaction};

#[tokio::test]
async fn test_fast_append_with_row_lineage() {
    // Helper function to create a data file with specified number of rows
    fn file_with_rows(record_count: u64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("test/{record_count}.parquet"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(record_count)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .build()
            .unwrap()
    }
    let catalog = new_memory_catalog().await;

    let table = make_v3_minimal_table_in_catalog(&catalog).await;

    // Check initial state - next_row_id should be 0
    assert_eq!(table.metadata().next_row_id(), 0);

    // First fast append with 30 rows
    let tx = Transaction::new(&table);
    let data_file_30 = file_with_rows(30);
    let action = tx.fast_append().add_data_files(vec![data_file_30]);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    // Check snapshot and table state after first append
    let snapshot = table.metadata().current_snapshot().unwrap();
    assert_eq!(snapshot.first_row_id(), Some(0));
    assert_eq!(table.metadata().next_row_id(), 30);

    // Check written manifest for first_row_id
    let manifest_list = table
        .metadata()
        .current_snapshot()
        .unwrap()
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();

    assert_eq!(manifest_list.entries().len(), 1);
    let manifest_file = &manifest_list.entries()[0];
    assert_eq!(manifest_file.first_row_id, Some(0));

    // Second fast append with 17 and 11 rows
    let tx = Transaction::new(&table);
    let data_file_17 = file_with_rows(17);
    let data_file_11 = file_with_rows(11);
    let action = tx
        .fast_append()
        .add_data_files(vec![data_file_17, data_file_11]);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    // Check snapshot and table state after second append
    let snapshot = table.metadata().current_snapshot().unwrap();
    assert_eq!(snapshot.first_row_id(), Some(30));
    assert_eq!(table.metadata().next_row_id(), 30 + 17 + 11);

    // Check written manifest for first_row_id
    let manifest_list = table
        .metadata()
        .current_snapshot()
        .unwrap()
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    assert_eq!(manifest_list.entries().len(), 2);
    // The NEW manifest comes first (Java `FastAppend.apply`: `writeNewManifests()` then
    // `snapshot.allManifests()`), and the manifest-list writer assigns ranges in that order.
    let new_manifest = &manifest_list.entries()[0];
    assert_eq!(new_manifest.added_files_count, Some(2));
    assert_eq!(new_manifest.first_row_id, Some(30));
    let carried_manifest = &manifest_list.entries()[1];
    assert_eq!(carried_manifest.added_files_count, Some(1));
    assert_eq!(carried_manifest.first_row_id, Some(0));
}
