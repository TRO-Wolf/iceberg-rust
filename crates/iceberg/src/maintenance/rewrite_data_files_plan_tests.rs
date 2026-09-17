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
use std::sync::Arc;

use crate::maintenance::rewrite_data_files::tests::{
    config_for, synthetic_spec_and_schema, synthetic_task,
};
use crate::maintenance::rewrite_data_files_plan::plan_file_groups;
use crate::spec::{Literal, PartitionSpec, Struct, Transform};

/// Partition grouping. Different partition values never share a group, and a task of a
/// non-default spec buckets as unpartitioned.
#[test]
fn test_plan_file_groups_partition_isolation_and_incompatible_spec() {
    let (spec, schema) = synthetic_spec_and_schema();
    let config = config_for(100, 75, 180, 2);

    // 2 undersized files in x=0, 2 in x=1 ⇒ two groups, one per partition.
    let tasks = vec![
        synthetic_task("p0a", 10, 0, 0, &spec, &schema),
        synthetic_task("p0b", 10, 0, 0, &spec, &schema),
        synthetic_task("p1a", 10, 1, 0, &spec, &schema),
        synthetic_task("p1b", 10, 1, 0, &spec, &schema),
    ];
    let groups = plan_file_groups(tasks, &config, &spec);
    assert_eq!(groups.len(), 2, "two partitions ⇒ two groups");
    for group in &groups {
        let partitions: HashSet<String> = group
            .iter()
            .map(|task| format!("{:?}", task.partition))
            .collect();
        assert_eq!(
            partitions.len(),
            1,
            "each group holds ONE partition value only"
        );
    }

    // A task of an incompatible spec buckets under the empty struct.
    let old_spec = Arc::new(
        PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_partition_field("y", "y", Transform::Identity)
            .unwrap()
            .build()
            .unwrap(),
    );
    // Both tasks carry the byte-identical partition struct `[0]`, so a naive "always key by
    // partition" co-groups them. Correct bucketing keeps them apart.
    let mut incompatible = synthetic_task("old", 10, 0, 0, &old_spec, &schema);
    incompatible.partition = Some(Struct::from_iter([Some(Literal::long(0))]));
    let current_file = synthetic_task("cur", 10, 0, 0, &spec, &schema);
    // A co-grouped 2-file bucket would qualify at min_input_files 2. Correct bucketing gives
    // two single-file buckets and zero groups, so dropping the spec check reddens this.
    let groups = plan_file_groups(vec![incompatible, current_file], &config, &spec);
    assert!(
        groups.is_empty(),
        "an incompatible-spec file and a current-spec file with the SAME partition struct are \
         bucketed SEPARATELY (incompatible ⇒ empty struct), never merged into a qualifying group"
    );
}
