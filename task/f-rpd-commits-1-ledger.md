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
| 1 | TBD | `test: F-RPD-COMMITS-1 — red single-commit, file-scoped cells` |
| 2 | TBD | `fix: F-RPD-COMMITS-1 — position-delete rewrite commits once and keeps file scope, as Java does` |
| 3 | TBD | `docs: F-RPD-COMMITS-1 — ledger, mutation proof` |

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

TBD — written after the fix lands.

## Red / green / mutation output

TBD — pasted from the runs.

## RePark strict-xfail forecast

TBD — written at the end.
