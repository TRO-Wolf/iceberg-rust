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
    data_file, data_manifest_count, fast_append, live_file_paths, make_v3_minimal_table_in_catalog,
    merge_append, new_memory_catalog, set_table_property,
};

// 3. PROPERTY-DISABLED PASSTHROUGH (Risk: merge fires when disabled). merge-enabled=false + min-count=2
// ⇒ no merge even though the threshold is met.
#[tokio::test]
async fn test_merge_append_disabled_does_not_merge() {
    let catalog = new_memory_catalog().await;
    let table = make_v3_minimal_table_in_catalog(&catalog).await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest.min-count-to-merge", "2").await;
    let table =
        set_table_property(&catalog, &table, "commit.manifest-merge.enabled", "false").await;

    let table = fast_append(&catalog, &table, vec![data_file("test/a.parquet", 0)]).await;
    let table = fast_append(&catalog, &table, vec![data_file("test/b.parquet", 0)]).await;
    let table = merge_append(&catalog, &table, vec![data_file("test/c.parquet", 0)]).await;

    assert_eq!(
        data_manifest_count(&table).await,
        3,
        "merge-enabled=false leaves all three manifests un-merged"
    );
    assert_eq!(
        live_file_paths(&table).await,
        HashSet::from([
            "test/a.parquet".to_string(),
            "test/b.parquet".to_string(),
            "test/c.parquet".to_string(),
        ])
    );
}
