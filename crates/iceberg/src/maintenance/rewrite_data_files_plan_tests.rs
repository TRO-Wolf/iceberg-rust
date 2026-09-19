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
use crate::maintenance::rewrite_data_files_plan::{
    expected_output_files, input_split_size, pack_bins, plan_file_groups, plan_read_tasks,
    write_max_file_size,
};
use crate::scan::FileScanTask;
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

#[test]
fn test_plan_read_tasks_spark_sizes_pack_two_files_per_split() {
    let (spec, schema) = synthetic_spec_and_schema();
    let config = config_for(2_000, 1_500, 3_600, 1);
    let tasks = vec![
        synthetic_task("p0a", 1_153, 0, 0, &spec, &schema),
        synthetic_task("p0b", 1_153, 0, 0, &spec, &schema),
        synthetic_task("p0c", 1_153, 0, 0, &spec, &schema),
        synthetic_task("p0d", 1_153, 0, 0, &spec, &schema),
        synthetic_task("p1a", 1_153, 1, 0, &spec, &schema),
        synthetic_task("p1b", 1_153, 1, 0, &spec, &schema),
        synthetic_task("p1c", 1_153, 1, 0, &spec, &schema),
        synthetic_task("p1d", 1_153, 1, 0, &spec, &schema),
    ];
    let groups = plan_file_groups(tasks, &config, &spec);
    assert_eq!(groups.len(), 2, "one group per partition");
    let mut total_read_tasks = 0usize;
    for group in &groups {
        let input_size: u64 = group.iter().map(|task| task.length).sum();
        assert_eq!(input_size, 4_612);
        let split_size = input_split_size(input_size, &config);
        assert_eq!(
            split_size, 2_800,
            "4612/3 + 5120 = 6657 clamps at writeMaxFileSize 2000 + (3600-2000)*0.5"
        );
        let read_tasks = plan_read_tasks(group.clone(), split_size).unwrap();
        assert_eq!(
            read_tasks.len(),
            2,
            "4 x 1153 B files pack two per 2800 B read split"
        );
        total_read_tasks += read_tasks.len();
    }
    assert_eq!(total_read_tasks, 4, "Spark's four output files");
}

#[test]
fn test_plan_read_tasks_default_target_is_one_task_per_group() {
    let (spec, schema) = synthetic_spec_and_schema();
    let config = config_for(512 * 1024 * 1024, 384 * 1024 * 1024, 966_367_641, 5);
    let tasks = vec![
        synthetic_task("a", 1_153, 0, 0, &spec, &schema),
        synthetic_task("b", 1_153, 0, 0, &spec, &schema),
        synthetic_task("c", 1_153, 0, 0, &spec, &schema),
        synthetic_task("d", 1_153, 0, 0, &spec, &schema),
    ];
    let input_size: u64 = tasks.iter().map(|task| task.length).sum();
    let split_size = input_split_size(input_size, &config);
    assert_eq!(
        split_size,
        512 * 1024 * 1024,
        "input below target ⇒ the split size is the target itself"
    );
    let read_tasks = plan_read_tasks(tasks, split_size).unwrap();
    assert_eq!(
        read_tasks.len(),
        1,
        "one read task per group ⇒ one output file, the pre-F-RDF-GRANULARITY-1 answer"
    );
}

#[test]
fn test_expected_output_files_remainder_rule_cells() {
    let config = config_for(2_000, 1_500, 3_600, 1);
    let cases: Vec<(u64, u64)> = vec![
        (6_776, 4),
        (4_612, 3),
        (1_999, 1),
        (2_000, 1),
        (4_000, 2),
        (18_000, 9),
        (3_500, 2),
        (4_398, 2),
        (4_400, 3),
        (17_500, 8),
    ];
    for (input_size, expected) in cases {
        assert_eq!(
            expected_output_files(input_size, &config),
            expected,
            "expected_output_files({input_size})"
        );
    }
}

#[test]
fn test_input_split_size_between_target_and_write_max() {
    let config = config_for(1_000_000, 750_000, 1_800_000, 1);
    let write_max = write_max_file_size(1_000_000, 1_800_000);
    assert_eq!(write_max, 1_400_000);
    let cases: Vec<(u64, u64)> = vec![(10_500_000, 1_055_120), (2_994_000, 1_003_120)];
    for (input_size, expected) in cases {
        let split = input_split_size(input_size, &config);
        assert_eq!(split, expected, "input_split_size({input_size})");
        assert!(
            split > 1_000_000 && split < write_max,
            "split {split} must sit strictly between target and writeMaxFileSize (unclamped)"
        );
    }
}

#[test]
fn test_pack_bins_forward_first_fit() {
    let (spec, schema) = synthetic_spec_and_schema();
    let sizes_of = |bins: &[Vec<FileScanTask>]| -> Vec<Vec<u64>> {
        bins.iter()
            .map(|bin| bin.iter().map(|task| task.file_size_in_bytes).collect())
            .collect()
    };

    let tasks: Vec<FileScanTask> = [3u64, 3, 3, 3]
        .iter()
        .enumerate()
        .map(|(index, &size)| synthetic_task(&format!("f{index}"), size, 0, 0, &spec, &schema))
        .collect();
    assert_eq!(
        sizes_of(&pack_bins(tasks, |task| task.file_size_in_bytes, 6)),
        vec![vec![3, 3], vec![3, 3]]
    );

    let tasks: Vec<FileScanTask> = [4u64, 3, 3]
        .iter()
        .enumerate()
        .map(|(index, &size)| synthetic_task(&format!("g{index}"), size, 0, 0, &spec, &schema))
        .collect();
    assert_eq!(
        sizes_of(&pack_bins(tasks, |task| task.file_size_in_bytes, 6)),
        vec![vec![4], vec![3, 3]]
    );

    let tasks: Vec<FileScanTask> = [7u64, 2, 2]
        .iter()
        .enumerate()
        .map(|(index, &size)| synthetic_task(&format!("h{index}"), size, 0, 0, &spec, &schema))
        .collect();
    assert_eq!(
        sizes_of(&pack_bins(tasks, |task| task.file_size_in_bytes, 6)),
        vec![vec![7], vec![2, 2]]
    );
}
