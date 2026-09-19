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

# Ledger — F-RPD-COMMITS-1: `rewrite_position_delete_files` commits once and keeps file scope, as Java does

**Ledger id:** `F-RPD-COMMITS-1`
**Branch:** `fix/f-rpd-commits-1`
**RePark row:** ICE-RDF-RPD-COMMITS-1
**Oracle:** the run-23a RPD oracle (Spark 4.1.2 + Iceberg runtime 1.11.0)
**Model:** swe-2-high

| Step | Commit | Subject |
|---|---|---|
| 1 | `ac9a53a1` | `test: F-RPD-COMMITS-1 — red single-commit, file-scoped cells` |
| 2 | `8e7fde35` | `fix: F-RPD-COMMITS-1 — position-delete rewrite commits once and keeps file scope, as Java does` |
| 3 | this commit | `docs: F-RPD-COMMITS-1 — ledger, mutation proof` |

## The oracle shape (run-23a)

Table `(id BIGINT, p INT, v STRING) PARTITIONED BY (p)`, format v2,
`write.delete.mode = merge-on-read`; 2 partitions × 4 single-INSERT data files of 50 rows
(`id` consecutive from 0); `DELETE … WHERE id % 2 = 0`; then `rewrite_position_delete_files`.

The DELETE writes **8 position-delete files, one per data file** — each holds 25 positions,
references exactly one data file (file-scoped), data sequence number 9.

| Cell | Result |
|---|---|
| `rpd_rewrite_all` (`rewrite-all=true`) | `rewritten_delete_files_count = 8`, `added_delete_files_count = 8`, `rewritten_bytes_count = 12675 == added_bytes_count`; ONE `replace` snapshot (ops end `append, delete, replace`); 8 output delete files, one per data file, 25 positions each, data sequence 9 preserved, file sequence 10; 200 live rows |
| `rpd_min_input_files_1` (`min-input-files=1`) | identical rewrite shape |
| `rpd_baseline` (no options) | nothing rewritten, no new snapshot |

`rewritten_bytes_count == added_bytes_count` is a data coincidence on the oracle, not a Java
rule: `RewritePositionDeletesGroup.asResult()` computes `rewrittenBytesCount =
inputFilesSizeInBytes()` (sum of INPUT delete-file sizes) and `addedBytesCount = addedBytes()`
(sum of OUTPUT sizes). Spark's delete write and rewrite write produce byte-identical files for
the same 25 positions, so the sums coincide. The tests assert each count against its own
derivation (inputs sum / outputs sum), not the equality.

## Java rules (all bytecode-verified on the 1.11.0 spark-runtime jar)

| Rule | Class / method | Detail |
|---|---|---|
| One commit for the whole rewrite by default | `RewritePositionDeleteFilesSparkAction.doExecute` | All groups rewritten via `Tasks.foreach(...).stopOnFailure().noRetry()`, then `RewritePositionDeletesCommitManager.commitOrClean(allGroups)` — a single `RewriteFiles`/`replace` commit over every group |
| Conflict behavior | same + `RewritePositionDeletesCommitManager.commitOrClean` | `validateFromSnapshot(startingSnapshotId)`; a `ValidationException`/`CommitFailedException` is caught as `CleanableFailure`, the batch's new files are deleted, and the failure propagates — **the whole rewrite fails, no retry** |
| Failed group write, non-partial | `doExecute` | first group failure stops the run; already-rewritten groups' new files are aborted (deleted) via `suppressFailureWhenFinished().run(group::abort)`; nothing commits |
| Partial progress | `doExecuteWithPartialProgress` | `groupsPerCommit = ceil(totalGroups / maxCommits)`; a `CommitService` commits batches of *completed* rewrites; group write failures are suppressed (the group is skipped); a failed batch is aborted and excluded from results |
| Partial-progress defaults | `RewritePositionDeleteFiles` interface | `partial-progress.enabled` default false; `partial-progress.max-commits` default 10; `max-commits` must be positive when enabled |
| Dangling positions dropped | `SparkRewritePositionDeleteRunner.doRewrite` | delete rows `leftsemi` join `dataFiles(partitionType, partition)` (the `files` metadata table filtered `eqNullSafe` on the group's partition fields) on `file_path` — a position whose referenced data file is not live **in that partition** is never written; a group whose positions all drop still has its input files removed by the commit |
| Output ordering | same | `sortWithinPartitions("file_path", "pos")` |
| Output granularity | `SparkWriteConf.deleteGranularity` + `SparkPositionDeletesRewrite.DeleteWriter` | option `delete-granularity`, else table property `write.delete.granularity`, else **`DeleteGranularity.FILE`** (the action's own default; `TableProperties.DELETE_GRANULARITY_DEFAULT` is `"partition"` but this path overrides it). FILE → `ClusteredPositionDeleteWriter` + `FileScopedPositionDeleteWriter`: one rolling output chain per referenced `file_path`, each rolling at `write.delete.target-file-size-bytes`. PARTITION → one output chain per partition group |
| Per-path output files are file-scoped | `RollingPositionDeleteWriter` / `PositionDeleteWriter` | referenced paths are aggregated per writer; equal `file_path` lower/upper bounds mark the file file-scoped (`ContentFileUtil.referencedDataFile`); the `referenced_data_file` field itself is not set |
| Sequence stamping | `RewritePositionDeletesCommitManager.commitOrClean` → `RewriteFiles` | each added delete file is stamped with **its own group's** `maxRewrittenDataSequenceNumber` — preserves the input data sequence number |

## Fork measurement (before the fix)

**Does the fork's own DELETE write file-scoped deletes?** No. The DataFusion DELETE path
(`crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs`,
`group_pairs_by_partition`) groups `(path, pos)` pairs by the `(spec_id, partition)` of the
referenced data file and writes **one position-delete file per partition group** — partition-scoped.
It does not consult `write.delete.granularity`. The Spark-shape inputs are therefore built with the
existing test writer `write_file_scoped_position_delete_file`, which produces one file-scoped
position-delete file per data file (equal `file_path` bounds), exactly the oracle's input shape.

**Fixture for the cells:** 2 partitions (`x = 0`, `x = 1`) × 4 data files × 50 rows
(y = a distinct value per row); 8 file-scoped position-delete files, one per data file, each
masking 25 positions. One `fast_append` snapshot (data files), one `row_delta` snapshot
(the deletes, data sequence number = that snapshot's).

**Before-fix behavior, measured on the red run** (`cargo test -p iceberg --lib
rewrite_position_delete_files`, 100 tests, 9 failed / 91 passed):

| Measure | Fork (before) | Oracle |
|---|---|---|
| `rewritten_delete_files_count` | 8 | 8 |
| `added_delete_files_count` | **2** | 8 |
| new snapshots | **2 `replace`** | 1 (`replace`) |
| output files | **2, partition-scoped** (bounds span 4 paths each; `referenced_data_file_location` = `None`) | 8, file-scoped |
| positions per output | **100** (all pairs of the partition in one file) | 25 |
| referenced data files per output | **4** | 1 |

Failing assertions (each is the fork's answer vs the oracle's):

- `test_rewrite_all_commits_once_and_keeps_file_scope` — `added == 8`: left 2, right 8.
- `test_min_input_files_1_commits_once_and_keeps_file_scope` — `added == 8`: left 2, right 8.
- `test_one_replace_commit_for_all_bins` — `snapshots == 1`: left 2, right 1.
- `test_partition_granularity_writes_partition_scoped_outputs_in_one_commit` — `snapshots == 1`: left 2, right 1.
- `test_dangling_positions_are_dropped_not_rewritten` — output `file-scoped == Some(a_path)`: left `None` (partition-scoped output carrying the dead path's positions).
- `test_bin_failure_aborts_the_whole_rewrite` — `snapshots.is_empty()`: bin 1 had already committed.
- `test_admitted_bin_with_zero_pairs_loses_its_inputs` — `rewritten == 10`: left 5, right 10 (the zero-pairs bin was skipped, inputs left live).
- `test_partial_progress_commits_one_batch_per_commit` — `added == 8`: left 2, right 8 (2 commits happened to match, the outputs did not).
- `test_partial_progress_max_commits_1_batches_all_bins_into_one_commit` — `added == 8`: left 2, right 8 (and snapshots were 2, not 1).

Controls green before and after: `test_baseline_declines_every_bin_and_commits_nothing`
(oracle `rpd_baseline`), `test_partial_progress_max_commits_zero_is_rejected`.

## The fix

`crates/iceberg/src/maintenance/rewrite_position_delete_files.rs` +
new sibling module `rewrite_position_delete_files_commit.rs` (the file crossed the
1000-line default ceiling, so the commit-path code moved there, the same pattern
`rewrite_position_delete_files_v3.rs` already uses; the tests-file legacy ceiling was
lowered 4716 → 4608 to match the shrink — ceilings only move down).

1. **Staged rewrite, batched commits.** `compact_group` (read → write → commit per bin) is
   replaced by `rewrite_bin` (read → drop dangling → sort → write, no commit) returning a
   `RewrittenBin { deleted, added: [(file, data_seq)] }`, and `commit_bins`, which folds a
   batch of staged bins into ONE `RewriteFiles`/`replace` transaction. `execute` drives
   `per_commit = plan_commit_batches(bins, partial_progress, max_commits).first()` — the same
   shared helper `RewriteDataFiles` uses, which returns one batch covering every bin when
   partial progress is off and `ceil(total/max_commits)`-sized batches when it is on.
2. **Atomicity.** A non-partial bin failure cleans up every already-written output file
   (`delete_uncommitted_files`) and propagates — Java's `Tasks.stopOnFailure()` +
   `commitOrClean` abort semantics. An `apply`/`commit` failure deletes that batch's output
   files and propagates — Java's `CleanableFailure`, whole rewrite fails, no retry.
   Partial progress suppresses a bin's write failure exactly as Java's
   `suppressFailureWhenFinished` does (the bin is skipped; only *successful* rewrites count
   toward a batch, as `BaseCommitService` offers only completed groups).
3. **File scope.** `ResolvedConfig` gains `delete_granularity`, parsed from the new
   `TableProperties::PROPERTY_DELETE_GRANULARITY` (`write.delete.granularity`) with default
   **file** — `SparkWriteConf.deleteGranularity`'s answer on this path (Java's
   `TableProperties.DELETE_GRANULARITY_DEFAULT` is `"partition"`, but the RPD write path
   overrides it; this fork has no other reader of the property). Under `file`,
   `write_group_outputs` splits the sorted pairs into runs of equal `file_path` and opens a
   fresh `write_compacted_file` chain per run — Java's `FileScopedPositionDeleteWriter`
   behaviour (one rolling chain per referenced path, rolling at write-max). Equal
   `file_path` bounds then mark each output file-scoped via
   `referenced_data_file_location` leg 3 — the same leg Java's `PositionDeleteWriter` relies
   on; `referenced_data_file` (field 134) is not stamped, matching Java. Under `partition`
   the whole bin goes through one writer as before.
4. **Dangling positions.** `collect_position_delete_groups` now also returns the live
   data-file paths per `(spec_id, partition)` group key (Java's `leftsemi` join on
   `file_path` against the group's live `files` scan). `rewrite_bin` retains only pairs
   whose path is live in that group; a bin whose pairs all drop contributes its inputs to
   the remove set and nothing to the add set — Java removes the empty group's input files
   in the commit, it does not skip the group.
5. **Counts.** `rewritten_*` counts come from the `deleted` sets (all admitted inputs,
   empty-output bins included); `added_*` from the `added` sets — the same quantities
   `RewritePositionDeletesGroup.asResult()` reports.
6. **v3 unchanged.** `rewrite_to_deletion_vectors` was not touched; it already commits once
   and already drops positions for dead data files (`test_v3_position_naming_a_non_live_data_file_is_dropped`
   stays green). `partial_progress` has no effect on the v3 arm, matching the Java arm where
   the DV rewrite is a separate code path inside the same single-commit manager.

## Red / green / mutation output

Command for every row: `cargo test -p iceberg --lib rewrite_position_delete_files`
(100 tests in the filter).

**RED** (baseline, commit `ac9a53a1`): 91 passed / 9 failed — the nine cells listed above.

**GREEN** (commit `8e7fde35`): 100 passed / 0 failed.

**MUTATION A — single-commit half reverted** (`per_commit = 1`, i.e. one commit per bin):
91 passed / **9 failed**, all for the commit-shape reason:

- `commits_tests::test_rewrite_all_commits_once_and_keeps_file_scope` FAILED
- `commits_tests::test_min_input_files_1_commits_once_and_keeps_file_scope` FAILED
- `commits_tests::test_partial_progress_max_commits_1_batches_all_bins_into_one_commit` FAILED
- `commits_tests::test_partition_granularity_writes_partition_scoped_outputs_in_one_commit` FAILED
- `test_one_replace_commit_for_all_bins` FAILED
- `test_bin_failure_aborts_the_whole_rewrite` FAILED
- `test_admitted_bin_with_zero_pairs_loses_its_inputs` FAILED
- `test_partition_isolation_compacts_each_group_separately` FAILED
- `test_admission_max_file_group_size_splits_partition_into_bins` FAILED

**MUTATION B — file-scope half reverted** (dangling `retain` removed + the per-path run
split in `write_group_outputs` removed, so every bin writes one partition-scoped output):
93 passed / **7 failed**:

- `commits_tests::test_dangling_positions_are_dropped_not_rewritten` FAILED
- `commits_tests::test_rewrite_all_commits_once_and_keeps_file_scope` FAILED
- `commits_tests::test_min_input_files_1_commits_once_and_keeps_file_scope` FAILED
- `commits_tests::test_partial_progress_commits_one_batch_per_commit` FAILED
- `commits_tests::test_partial_progress_max_commits_1_batches_all_bins_into_one_commit` FAILED
- `test_multi_file_grouping_one_partition` FAILED
- `test_unpartitioned_group_compacts` FAILED

**RESTORED**: 100 passed / 0 failed.

## RePark strict-xfail forecast

| RePark cell | Fork answer now | Oracle | Forecast |
|---|---|---|---|
| `rpd_rewrite_all` value cell | `rewritten 8 / added 8`, outputs file-scoped 25-pos each, data seq 9, file seq = rewrite snapshot | identical | **xpass** — the strict xfail will fire and can flip to pass |
| `rpd_rewrite_all` snapshot cell | ops end `append, delete, replace` — one `replace`, so 10 snapshots not 11 | identical | **xpass** |
| `rpd_min_input_files_1` value cell | identical to `rpd_rewrite_all` | identical | **xpass** |
| `rpd_min_input_files_1` snapshot cell | identical | identical | **xpass** |

All four strict xfails should now fire XPASS against the bumped fork and retire.
The `rpd_baseline` cell already passed (declined bins commit nothing) and is untouched.
