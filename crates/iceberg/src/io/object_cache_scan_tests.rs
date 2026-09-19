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

use std::sync::Arc;

use futures::TryStreamExt;

use crate::io::object_cache::ObjectCache;
use crate::scan::tests::TableTestFixture;
use crate::table::Table;

#[tokio::test]
async fn p7_scan_plan_fetches_each_manifest_once_through_object_cache() {
    let mut fixture = TableTestFixture::new();
    fixture.setup_manifest_files().await;

    let object_cache = Arc::new(ObjectCache::new_with_capacity(
        fixture.table.file_io().clone(),
        1 << 20,
    ));
    let table = Table::builder()
        .metadata(fixture.table.metadata_ref())
        .identifier(fixture.table.identifier().clone())
        .file_io(fixture.table.file_io().clone())
        .metadata_location(
            fixture
                .table
                .metadata_location()
                .expect("fixture location"),
        )
        .object_cache(Arc::clone(&object_cache))
        .build()
        .expect("table with object cache");

    let first_plan: Vec<_> = table
        .scan()
        .build()
        .expect("first scan")
        .plan_files()
        .await
        .expect("first plan")
        .try_collect()
        .await
        .expect("collect first plan");
    assert!(!first_plan.is_empty(), "scan must yield file tasks");
    let cold_fetches = object_cache.body_fetches();
    assert!(cold_fetches > 0, "a cold plan must read manifest bodies");

    let second_plan: Vec<_> = table
        .scan()
        .build()
        .expect("second scan")
        .plan_files()
        .await
        .expect("second plan")
        .try_collect()
        .await
        .expect("collect second plan");
    assert_eq!(first_plan.len(), second_plan.len());
    assert_eq!(
        object_cache.body_fetches(),
        cold_fetches,
        "a second plan over the same snapshot must fetch zero manifest bodies"
    );
}
