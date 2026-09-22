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

use arrow_array::cast::AsArray;
use arrow_array::Array;

use super::files::tests::{scan_single_batch, setup_data_and_delete_manifests};
use crate::scan::tests::TableTestFixture;

#[tokio::test]
async fn test_pos_delete_row_renders_null_field_counts() {
    let fixture = TableTestFixture::new();
    setup_data_and_delete_manifests(&fixture).await;
    let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;
    let paths = batch
        .column_by_name("file_path")
        .unwrap()
        .as_string::<i32>();
    let mut by_suffix = HashMap::new();
    for index in 0..paths.len() {
        let suffix = paths.value(index).rsplit('/').next().unwrap().to_string();
        by_suffix.insert(suffix, index);
    }
    let delete_row = by_suffix["delete-1.parquet"];
    for name in ["value_counts", "null_value_counts", "nan_value_counts"] {
        let maps = batch.column_by_name(name).unwrap().as_map();
        assert!(maps.is_null(delete_row));
    }
    let column_sizes = batch.column_by_name("column_sizes").unwrap().as_map();
    assert!(!column_sizes.is_null(delete_row));
    assert_eq!(column_sizes.value_length(delete_row), 0);
}

#[tokio::test]
async fn test_data_row_renders_empty_counts_as_empty_maps() {
    let fixture = TableTestFixture::new();
    setup_data_and_delete_manifests(&fixture).await;
    let batch = scan_single_batch(fixture.table.inspect().files().scan().await.unwrap()).await;
    let paths = batch
        .column_by_name("file_path")
        .unwrap()
        .as_string::<i32>();
    let mut by_suffix = HashMap::new();
    for index in 0..paths.len() {
        let suffix = paths.value(index).rsplit('/').next().unwrap().to_string();
        by_suffix.insert(suffix, index);
    }
    let data_row = by_suffix["3.parquet"];
    for name in ["value_counts", "null_value_counts", "nan_value_counts"] {
        let maps = batch.column_by_name(name).unwrap().as_map();
        assert!(!maps.is_null(data_row));
        assert_eq!(maps.value_length(data_row), 0);
    }
}
