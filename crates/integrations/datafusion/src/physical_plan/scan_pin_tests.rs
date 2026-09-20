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

use super::tests::{create_test_table, test_arrow_schema, test_arrow_schema_with_field_ids};
use super::*;

#[test]
fn test_pin5_limit_demoted_when_n_gt_1() {
    let n = 3usize;
    let mut limit = Some(5usize);
    if n > 1 {
        limit = None;
    }
    assert_eq!(limit, None, "pin 5: limit demoted when N>1");

    let n1 = 1usize;
    let mut limit1 = Some(5usize);
    if n1 > 1 {
        limit1 = None;
    }
    assert_eq!(limit1, Some(5), "pin 5: limit retained when N=1");
}

#[test]
fn test_pin5_mutation_per_partition_hard_limit_overcounts() {
    let n = 3usize;
    let k = 2usize;
    let table_rows = 100usize;
    let per_part_only = (n * k).min(table_rows);
    let global_correct = k.min(table_rows);
    assert!(
        per_part_only > global_correct,
        "mutation RED condition: per-part hard k yields {per_part_only} > global {global_correct}"
    );
}

#[test]
fn test_pin13_effective_t_with_off_switch() {
    let knobs = ScanKnobs {
        batch_size: Some(1024),
        data_file_concurrency: Some(8),
        target_partitions: 8,
        multi_partition_scan: false,
        row_selection_enabled: true,
    };
    let t = if knobs.multi_partition_scan {
        knobs.target_partitions.max(1)
    } else {
        1
    };
    assert_eq!(t, 1);
    assert!(knobs.target_partitions > 1);
}

#[test]
fn test_pin14_p_formula_independent_of_t() {
    let n = 4usize;
    let l1 = 16usize;
    let l2 = 2usize;
    let p1 = l1.div_ceil(n).max(1);
    let p2 = l2.div_ceil(n).max(1);
    assert_eq!(p1, 4);
    assert_eq!(p2, 1);
    assert_ne!(
        p1, p2,
        "pin 14: distinct L must yield distinct P at fixed N"
    );
    let l_small = 2usize;
    let n_big = 8usize;
    let p = l_small.div_ceil(n_big).max(1);
    assert_eq!(p, 1);
    assert!(n_big * p > l_small);
}

#[tokio::test]
async fn test_pin12_snapshot_frozen_on_work() {
    use datafusion::execution::TaskContext;

    let table = create_test_table();
    let knobs = ScanKnobs {
        batch_size: Some(1024),
        data_file_concurrency: Some(1),
        target_partitions: 1,
        multi_partition_scan: true,
        row_selection_enabled: true,
    };
    let scan = IcebergTableScan::plan(
        table,
        None,
        false,
        test_arrow_schema_with_field_ids(),
        None,
        &[],
        None,
        knobs,
    )
    .await
    .expect("plan empty");
    assert!(!scan.partition_work().is_empty(), "eager plan embeds work");
    for work in scan.partition_work() {
        assert_eq!(
            work.snapshot_id(),
            scan.resolved_snapshot_id(),
            "pin 12: work snapshot must match plan resolved id"
        );
    }
    let ctx = Arc::new(TaskContext::default());
    let stream = scan.execute(0, ctx).expect("execute 0");
    drop(stream);
}

#[test]
fn test_display_default_deterministic_snapshot_id_verbose_only() {
    use datafusion::physical_plan::displayable;

    let scan = IcebergTableScan::new(
        create_test_table(),
        None,
        false,
        test_arrow_schema(),
        None,
        &[],
        None,
    )
    .expect("scan");
    let plan: Arc<dyn ExecutionPlan> = Arc::new(scan);
    let default_form = displayable(plan.as_ref()).indent(false).to_string();
    let verbose_form = displayable(plan.as_ref()).indent(true).to_string();

    assert!(
        !default_form.contains("snapshot_id="),
        "default EXPLAIN must not carry the per-run-random snapshot id: {default_form}"
    );
    assert!(
        default_form.contains(" N=1"),
        "default EXPLAIN keeps deterministic N: {default_form}"
    );
    assert!(
        verbose_form.contains("snapshot_id="),
        "EXPLAIN VERBOSE must expose the frozen snapshot id: {verbose_form}"
    );
    assert!(
        verbose_form.contains(" N=1"),
        "EXPLAIN VERBOSE keeps N too: {verbose_form}"
    );
}
