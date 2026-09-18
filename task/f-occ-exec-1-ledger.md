<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Ledger — F-OCC-EXEC-1: the DataFusion DELETE/UPDATE exec scopes conflict validation by its own scan filter

**Ledger id:** `F-OCC-EXEC-1-2026-09-18`
**Branch:** `fix/occ-exec-1` (cut off `origin/main` = `8fb44a39`)
**Scope:** devin-worker brief F-OCC-EXEC-1
**Model:** swe-2
**Extends:** `task/f-occ-scoped-1-ledger.md` (fork #291 armed the `conflict_detection_filter`
seam; this unit makes the engine side pass its own scan filter through it).

## 1. Commit-path map (fork, at `8fb44a39`)

| Step | File:line | What lives there |
|---|---|---|
| DML planning | `crates/integrations/datafusion/src/table/mod.rs` | `IcebergTableProvider::delete_from` / `.update` build `prune = convert_filters_to_predicate(&filters)` and the exact DataFusion row `predicate`, then construct `IcebergDeleteExec` / `IcebergUpdateExec` carrying both |
| Filter → predicate | `crates/integrations/datafusion/src/physical_plan/expr_to_predicate.rs` | `convert_filters_to_predicate`: each supported `Expr` converts; unsupported ones are dropped; successes combine with `AND`. `None` only when no filter was supplied |
| Target scan (MoR) | `crates/integrations/datafusion/src/physical_plan/mor_scan.rs` | `mor_scan_stream` → `table.scan().with_file_prune_only(prune)` |
| Target scan (CoW) | `crates/integrations/datafusion/src/physical_plan/row_lineage.rs` | `cow_scan_stream` → `table.scan().with_file_prune_only(prune)` |
| Prune-only scan | `crates/iceberg/src/scan/mod.rs` | `with_file_prune_only` runs `predicate.rewrite_not()` and uses it for manifest/file pruning only; it attaches no residual row filter, so every row of a surviving file reaches the exact DataFusion predicate |
| MoR DELETE commit | `crates/integrations/datafusion/src/physical_plan/delete.rs:~532` | `row_delta().conflict_detection_filter(Predicate::AlwaysTrue).validate_data_files_exist(..)[.validate_deleted_files()].validate_no_conflicting_data_files()` |
| CoW DELETE commit | `crates/integrations/datafusion/src/physical_plan/delete.rs:~671` | `overwrite_files().conflict_detection_filter(Predicate::AlwaysTrue).validate_no_conflicting_deletes()[.validate_no_conflicting_data()]` |
| MoR UPDATE commit | `crates/integrations/datafusion/src/physical_plan/delete.rs:~988` | `row_delta().conflict_detection_filter(Predicate::AlwaysTrue).validate_data_files_exist(..).validate_deleted_files().validate_no_conflicting_delete_files()[.validate_no_conflicting_data_files()]` |
| CoW UPDATE commit | `crates/integrations/datafusion/src/physical_plan/delete.rs:~1124` | `overwrite_files().conflict_detection_filter(Predicate::AlwaysTrue).validate_no_conflicting_deletes()[.validate_no_conflicting_data()]` |
| Filter-scoped check | `crates/iceberg/src/transaction/snapshot/conflict_filter.rs` | `first_conflicting_file`: partition projection per spec → `InclusiveMetricsEvaluator`; `None` filter = `AlwaysTrue`; unknown spec fails closed |

## 2. Java rule matched

`SparkPositionDeltaWrite` / `SparkCopyOnWriteOperation`: the serializable conflict-detection
filter is the conjunction of the filters pushed into the operation's scan; `alwaysTrue` only
when nothing was pushed. RePark run 21a (Spark 4.1.2 + Iceberg 1.11.0): four concurrent
partition-local `UPDATE … WHERE k = '<key>' AND id < 8` on `PARTITIONED BY (k)` commit 4/4 in
Spark, 1/4 in RePark — the three losers fail
`Found conflicting files that can contain records matching TRUE` naming another partition's file.

## 3. Defect

Each of the four commit sites hard-codes `Predicate::AlwaysTrue` although the exec already
carries `prune: Option<Predicate>` — the same predicate the exec pushes into its target scan
for file pruning. Under `serializable` every concurrently added data file therefore
might-matches and the commit aborts, even for provably disjoint partitions (the F-OCC-SCOPED-1
partition-projection gate never sees the real filter).

## 4. Red tests (step 2)

Harness: `crates/integrations/datafusion/src/physical_plan/occ_exec_tests.rs` (new file —
`delete_tests.rs` is at 986/1000 lines). Wired as `mod occ_exec_tests` inside `delete.rs`.
Fixture: memory catalog, V2 table `PARTITIONED BY (k)`, seeded rows in partitions `a` and `b`
(`mor` flag selects merge-on-read vs copy-on-write properties). Race injection plans the DML
through the real `TableProvider` entry point (`delete_from` / `update`), commits a concurrent
transaction against the same base snapshot, then executes the plan. `WHERE k = 'a' AND id < 8`
gives `prune = k = 'a' AND id < 8`.

`cargo test -p iceberg-datafusion --lib occ_exec_tests` before the fix:

```text
test result: FAILED. 9 passed; 4 failed; 0 ignored; 0 measured; 229 filtered out

failures:
    physical_plan::delete::occ_exec_tests::cow_delete_disjoint_partition_commit_commits
    physical_plan::delete::occ_exec_tests::cow_update_disjoint_partition_commit_commits
    physical_plan::delete::occ_exec_tests::mor_delete_disjoint_partition_commit_commits
    physical_plan::delete::occ_exec_tests::mor_update_disjoint_partition_commit_commits
```

Each disjoint case fails with the defect's own message:

```text
a concurrent commit in a disjoint partition must not conflict the k = 'a' <DML>:
External(DataInvalid => Found conflicting files that can contain records matching TRUE:
test/b-new.parquet)
```

The 4 disjoint cases inject a concurrent append into `b` plus — for the two MoR paths — a
concurrent `b` position-delete file in the same `row_delta` commit. The 9 passing controls:

- `mor_delete_matching_partition_commit_conflicts` — concurrent append into `a` aborts.
- `cow_delete_matching_partition_commit_conflicts` — same.
- `mor_update_matching_partition_commit_conflicts` — same.
- `cow_update_matching_partition_commit_conflicts` — same.
- `mor_update_matching_partition_delete_file_conflicts` — concurrent `a` delete file aborts
  the UPDATE (the UPDATE's `row_delta` arms `validate_no_conflicting_delete_files`; check 2a
  catches it through the rewritten `a` data files).
- `mor_delete_no_predicate_keeps_always_true` — `DELETE FROM t` aborts on any concurrent
  commit, error contains `matching TRUE`.
- `cow_delete_no_predicate_keeps_always_true` — same.
- `mor_update_no_predicate_keeps_always_true` — `UPDATE` with no filter aborts, `matching TRUE`.
- `cow_update_no_predicate_keeps_always_true` — same.

Observed while probing controls: the MoR DELETE commit does not arm
`validate_no_conflicting_delete_files` (`delete.rs:532-543`), so a concurrent `a` delete file
never conflicts it — a test asserting that conflict was dropped; that is the armed surface,
not a defect of this unit.
