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

# Ledger — F-OCC-SCOPED-1: filter-scoped conflict detection and retry-on-rebase (fork half)

**Ledger id:** `F-OCC-SCOPED-1-2026-09-17`
**Branch:** `fix/occ-scoped-1` (cut off `origin/main` = `96fc9f1f`)
**Scope:** muse-worker brief FORK UNIT F-OCC-SCOPED-1, run 20a, 2026-09-17
**Model:** muse-spark-1.3-contributor
**Matrix rows touched:** R103 (`OverwriteFiles`), R106 (`RowDelta`), R110 (OCC retry), R146 (conflict-detection builder surface), R157 (commit-outcome taxonomy) — status cells only, no flip claimed here.
**Rating input:** `/tmp/oc-worker/ice-rating/report.md` row V2-20a + §7 item 3; `/tmp/oc-worker/ice-rating/worker-findings.md` V2-20a, DML-5, p_race, p_retry. RePark rows DML-5, G3-E8, ICE-COMMIT-UNKNOWN-1 in `/tmp/ia-build` (read only).

## 1. Commit-path map (fork, at `96fc9f1f`)

| Step | File:line | What lives there |
|---|---|---|
| Retry loop | `crates/iceberg/src/transaction/mod.rs:382-422` | `Transaction::commit`: `backon` retry; gate `e.retryable() && kind != CommitStateUnknown`; unknown arm reconciles by refresh |
| Retry knobs | `crates/iceberg/src/transaction/mod.rs:500-510` + `crates/iceberg/src/spec/table_properties.rs:111-129` | `build_backoff` from table props; defaults `commit.retry.num-retries=4`, `min-wait-ms=100`, `max-wait-ms=60000`, `total-timeout-ms=1800000` — the Java names and defaults |
| Rebase + validate + re-apply | `crates/iceberg/src/transaction/mod.rs:512-640` | `do_commit`: clear attempt evidence, `load_table`, re-base on stale pointer, run each action `validate(starting, refreshed)`, re-apply commits, capture `latest_attempt_snapshot_ids`, `update_table` |
| Unknown reconciliation | `crates/iceberg/src/transaction/mod.rs:434-498` + `crates/iceberg/src/transaction/commit_status.rs` | `reconcile_unknown_commit_outcome`: search reloaded snapshot set for attempted ids; landed ⇒ `Ok`, else original unknown surfaces; metadata-only commits skip |
| Validate seam | `crates/iceberg/src/transaction/action.rs` | `TransactionAction::validate` (default no-op) + `target_ref()` for branch-aware walks |
| Shared conflict checks | `crates/iceberg/src/transaction/snapshot.rs:2194-2339` | `validate_no_conflicting_added_data_files_on`, `validate_no_conflicting_added_delete_files_on`, `validate_deleted_data_files_on`, shared `first_conflicting_file` (metrics-only — the defect, §3) |
| Concurrent-commit walks | `crates/iceberg/src/transaction/snapshot.rs:1749-1900` | `files_after` + `added_data_files_after_on` / `added_delete_files_after_on` / `deleted_data_files_after_on`, branch-aware |
| Row-delta action | `crates/iceberg/src/transaction/row_delta.rs:255-341`, `645-730` | `validate_no_conflicting_data_files`, `validate_no_conflicting_delete_files`, `conflict_detection_filter`, `validate_from_snapshot`, `validate_data_files_exist`, `validate_deleted_files`; `validate` runs checks 1/2a/2b/3/4/5 |
| Overwrite action | `crates/iceberg/src/transaction/overwrite_files.rs:231-289`, `434-514` | `validate_no_conflicting_data`, `validate_no_conflicting_deletes` (branches A/B), `conflict_detection_filter`, `validate_from_snapshot`, `data_conflict_detection_filter` (explicit filter, else row filter when no explicit deletes, else `None` = TRUE) |
| Conflict errors | `crates/iceberg/src/error.rs:79-124` | `ErrorKind::CatalogCommitConflicts` (retryable flag set by catalogs) vs `ErrorKind::DataInvalid` (non-retryable validation conflicts) vs `ErrorKind::CommitStateUnknown` (never retried, kind-gated) |
| Memory catalog CAS | `crates/iceberg/src/catalog/memory/catalog.rs:255`, `542` | stale-base `update_table` ⇒ retryable `CatalogCommitConflicts` |
| Partition-eval pattern to reuse | `crates/iceberg/src/transaction/overwrite_files.rs:340-377` + `crates/iceberg/src/expr/visitors/expression_evaluator.rs:36-51` | `InclusiveProjection` → bind to per-spec partition schema → `ExpressionEvaluator::eval(&DataFile)` |

## 2. Java rule matched

No Java sources on this box (`/tmp/iceberg-java-ref` absent, no `iceberg-core*sources*` jar, no `~/.m2`). Rule stated from class/method names and the documented contract (RePark registry DML-5 audit M15/M20, ICE-COMMIT-UNKNOWN-1; rating report §6):

- C1. `BaseRowDelta` / `BaseOverwriteFiles` / `MergingSnapshotProducer` (`core/`, Iceberg 1.11): `validateAddedDataFiles`, `validateNoConflictingDeleteFiles` (`validateNoNewDeleteFiles` + `validateNoNewDeletesForDataFiles`), `validateDeletedDataFiles` scope concurrent files to the operation's `conflictDetectionFilter` via `ManifestGroup.filterData` — partition evaluation first, inclusive metrics second. Default filter is `alwaysTrue`.
- C2. `SnapshotProducer.commit()`: `onlyRetryOn(CommitFailedException)`; `CommitStateUnknownException` is never retried, never cleaned, and surfaces after `checkCommitStatus[Strict]` reconciliation.
- C3. Retry knobs `commit.retry.num-retries=4`, `min-wait-ms=100`, `max-wait-ms=60000`, `total-timeout-ms=1800000`; a retry re-applies the operation on the new base and re-runs validation from the operation's starting snapshot.

## 3. Defect analysis (measured 2026-09-16, memory catalog)

- D1. `first_conflicting_file` (`snapshot.rs:2315`) binds the filter to the table schema and runs `InclusiveMetricsEvaluator` only. No partition projection runs. A filter on a partition column can therefore exclude a concurrent file only when that file happens to carry usable metrics on the column; files without usable bounds always might-match. With no filter set the check is `AlwaysTrue` by design (Java default) — the RePark DML-5 shape that aborts a serializable MERGE on any concurrent append.
- D2. Retry-on-rebase (`mod.rs:382-422,512-640`), Java retry defaults (`table_properties.rs:111-129`), retryable `CatalogCommitConflicts` (`memory/catalog.rs:255,542`), and never-retry-unknown (`mod.rs:409,417-421`) are already ported. The 16-writer storm exhausting the budget is expected under contention; the fork-half gap is D1 only.
- Fix scope: gate each candidate file in `first_conflicting_file` on its own spec's partition projection (`InclusiveProjection` + `ExpressionEvaluator`, same pattern as `overwrite_files.rs:340-377`) before the metrics check. `None` filter stays `AlwaysTrue`. Unknown-spec files stay conflicting (fail-closed). No new public API: `RowDeltaAction::conflict_detection_filter` (`row_delta.rs:286`) and `OverwriteFilesAction::conflict_detection_filter` (`overwrite_files.rs:255`) are the seam the RePark side must call, each paired with its `validate_no_conflicting_*` arming flag.

## 4. Red tests (step 2)

Harness: fault-injected (load table, commit a concurrent snapshot between base load and operation commit, commit operation). File: `crates/iceberg/src/transaction/occ_scoped_tests.rs` (new).

`cargo test -p iceberg --lib transaction::occ_scoped` at `37924577` + wiring, before fix:

```text
test result: FAILED. 11 passed; 4 failed; 0 ignored; 0 measured; 3723 filtered out

failures:
    transaction::occ_scoped_tests::overwrite_row_filter_rewrite_of_unrelated_file_commits
    transaction::occ_scoped_tests::overwrite_serializable_disjoint_partition_append_commits
    transaction::occ_scoped_tests::row_delta_serializable_disjoint_partition_append_commits
    transaction::occ_scoped_tests::row_delta_serializable_nonmatching_delete_file_commits
```

Representative failure (`row_delta_serializable_disjoint_partition_append_commits`):

```text
a concurrent append into a disjoint partition must not conflict under filter x = 1:
DataInvalid => Found conflicting files that can contain records matching x = 1:
test/other-part.parquet
```

The 4 failures are all and only the disjoint-partition commit cases: the
metrics-only `first_conflicting_file` cannot exclude a concurrent file by
partition, so every concurrent file might-matches. The 11 passes cover the
matching-partition conflicts, metrics exclusion, files-exist, snapshot-isolation
rebase, append-vs-append retry, and unknown-no-retry.

## 5. Fix (step 3)

`first_conflicting_file` moved from `crates/iceberg/src/transaction/snapshot.rs:2315`
to the new `crates/iceberg/src/transaction/snapshot/conflict_filter.rs` (the move keeps
`snapshot.rs` under its 3490-line legacy ceiling: 3490 → 3454). Each candidate file is
now gated on its own spec's partition projection (`InclusiveProjection` + `ExpressionEvaluator`,
the `overwrite_files.rs:340-377` pattern) before the unchanged `InclusiveMetricsEvaluator`
check. `None` filter stays `AlwaysTrue`; unknown-spec files stay conflicting (fail-closed).
The three `validate_*_on` wrappers keep their signatures; no public API changed
(`RowDeltaAction::conflict_detection_filter`, `OverwriteFilesAction::conflict_detection_filter`
are the engine seam). Result: `cargo test -p iceberg --lib transaction::occ_scoped` →
16 passed, 0 failed (the 16th, `fast_append_conflicted_first_attempt_retries_and_commits`,
was added during mutation, §6).

## 6. Mutation (step 4, one knob at a time)

- A. Filter neutralised (partition gate dropped, metrics-only): 4 red out of 15 — exactly the
  disjoint-partition commit cases (`row_delta_serializable_disjoint_partition_append_commits`,
  `row_delta_serializable_nonmatching_delete_file_commits`,
  `overwrite_serializable_disjoint_partition_append_commits`,
  `overwrite_row_filter_rewrite_of_unrelated_file_commits`). Restored → green.
- B. Retry neutralised (`.when(|_| false)` in `Transaction::commit`): first attempt 0 red out
  of 16 — the sequential race rebases on its first attempt and never reaches the retry loop,
  so the battery could not observe the retry. Recorded as unkillable-through-harness, not as
  coverage. Added `fast_append_conflicted_first_attempt_retries_and_commits` (MockCatalog fails
  attempt 1 with retryable `CatalogCommitConflicts`, delegates attempt 2): re-applied B gives
  1 red out of 16 (that test only). Restored → 16 green.

## 7. Gates (step 5)

- `cargo test -p iceberg --lib`: 3731 passed, 0 failed, 8 ignored.
- `cargo clippy -p iceberg --all-targets -- -D warnings`: clean, no warnings.
- `cargo test -p iceberg-datafusion --lib`: 228 passed, 0 failed, 1 ignored.
- `cargo test -p iceberg-datafusion --tests` (all 32 non-Docker integration targets;
  interop suites run their offline legs, Java legs no-op without env): 264 passed,
  0 failed, 6 ignored.
- `cargo clippy -p iceberg-datafusion --all-targets -- -D warnings`: clean, no warnings.
- `cargo fmt --all -- --check`: clean. `typos .`: clean.
- `./scripts/check_rust_file_size.sh`: 485 files clean, 99 legacy ceilings
  (`snapshot.rs` ceiling lowered 3490 → 3454; `mod.rs` wiring moved to `action.rs`
  to hold its 1948 ceiling).
- `./scripts/check_comment_blocks.sh`: OK. `./scripts/check_agent_artifacts.sh`: OK.
- `./scripts/check_matrix_anchors.sh`: OK, 85 rows anchored.
- GAP_MATRIX: no row flipped (unit-only evidence; conflict-validation interop residue on
  rows R103/R106/R110/R146 remains). `task/todo.md`: ACTIVE F-OCC-SCOPED-1 section added.

## 8. Open questions

None yet.
