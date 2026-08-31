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
use std::sync::Arc;

use super::super::tests::make_v2_minimal_table;
use crate::TableUpdate;
use crate::transaction::{Transaction, TransactionAction};

#[tokio::test]
async fn test_append_snapshot_properties() {
    let table = make_v2_minimal_table();
    let tx = Transaction::new(&table);

    let mut snapshot_properties = HashMap::new();
    snapshot_properties.insert("key".to_string(), "val".to_string());

    let action = tx
        .fast_append()
        .set_snapshot_properties(snapshot_properties);
    let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
    let updates = action_commit.take_updates();

    // Check customized properties is contained in snapshot summary properties.
    let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
        snapshot
    } else {
        unreachable!()
    };
    assert_eq!(
        new_snapshot
            .summary()
            .additional_properties
            .get("key")
            .unwrap(),
        "val"
    );
}
