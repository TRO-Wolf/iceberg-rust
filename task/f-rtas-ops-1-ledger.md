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

# F-RTAS-OPS-1 ledger — RTAS records overwrite/delete, not append

Model: muse-spark-1.3-contributor | Run 20c, 2026-09-17 | Base 75da2b58

## Finding

RePark `CREATE OR REPLACE TABLE AS SELECT` over an existing table commits
`append, append`. Spark 4.1.2 + Iceberg 1.11.0 records `append, overwrite`.
Oracle: `/tmp/oc-worker/ic-build/write_fidelity_spark.json`, key `rtas_ops`.

## Oracle cells (Spark 4.1.2 + Iceberg 1.11.0)

- `ctas_then_rtas`: CTAS → `append`; RTAS (100 rows) → `overwrite`,
  summary `added-data-files 5, added-records 100, total-records 100,
  total-data-files 5`, NO `deleted-*` keys.
- `rtas_new_table`: RTAS on missing table → `overwrite` (10 rows).
- `rtas_empty_new`: RTAS empty SELECT on new table → `delete`, no `added-*`.
- `rtas_empty_twice`: again → `delete`, `delete`.
- CTAS of rows → `append` (unchanged).

## Java reason (to verify)

Spark RTAS writes through `OverwriteByExpression(alwaysTrue)` on the staged
replace transaction (`newOverwrite()`); operation is `overwrite` with added
files, `delete` with none. CTAS writes through `newAppend()`. Bytecode
targets: `org.apache.iceberg.BaseOverwriteFiles.operation`,
`org.apache.iceberg.spark.source.SparkWrite$OverwriteByFilter`.
Verification record below.

## Clauses

- C-001: staged commit with replace-write semantics: `overwrite` with added
  files, `delete` with none; plain create keeps `append`; empty CTAS behavior
  measured, unchanged without a Java reason.
- C-002: replace-semantics API reachable for new and existing tables.
- C-003: replace snapshot summary key set equals Java (no `deleted-*`,
  `changed-partition-count`, `total-*` as Java).
- C-004: existing callers/tests unchanged (CTAS, staged publish/reload, Glue).

## Steps

1. Ledger skeleton. Commit. (this file)
2. Red-first tests beside staged_table tests. Run on base, paste reds. Commit.
3. Implement C-001..C-003. Commit.
4. GAP_MATRIX row, maps, gate output. Gates green. Commit.

## Bytecode verification

Verified 2026-09-17 against
`/tmp/ic-build/.ivy2/jars/org.apache.iceberg_iceberg-spark-runtime-4.1_2.13-1.11.0.jar`
with `/usr/lib/jvm/zulu-17-amd64/bin/javap -c -p`:

- `BaseOverwriteFiles.operation()`: `delete` iff deletes && !adds; `append`
  iff adds && !deletes; else `overwrite`.
- `MergingSnapshotProducer.deletesDataFiles()` forwards to
  `ManifestFilterManager.containsDeletes()`: true when deletePaths non-empty,
  deleteFiles non-empty, deleteExpression != alwaysFalse, or dropPartitions
  non-empty. A SET row filter counts as a delete BEFORE any file resolves.
- `SparkWrite$OverwriteByFilter.commit` calls `table.newOverwrite()`,
  then `overwriteByRowFilter(overwriteExpr)`, then `addFile` per file.
- `SnapshotProducer` bytecode carries no empty-commit rejection string: Java
  commits the filter-requested empty overwrite as a `delete` snapshot. The
  fork's `manifest_file` truly-empty guard is Rust-side and must yield when a
  row filter was requested.

## Red run (base tree)

`cargo test -p iceberg --lib transaction::staged_table` on the base tree
plus the new `staged_table_rtas_ops_tests.rs` (6 tests, API absent):

- 4x `error[E0599]: no method named with_replace_write found for struct
  StagedTableTransaction` (the four replace-write tests).
- The two no-flag tests
  (`create_without_replace_write_stays_append`,
  `create_empty_without_flag_commits_no_snapshot`) compile; they pin C-004
  and measure empty-CTAS behavior.

## Implementation notes

- `staged_table.rs`: new `replace_write` flag (default false) + builder
  `with_replace_write(bool)`. `materialize_pending` with the flag commits
  pending files through `overwrite_files().overwrite_by_row_filter(AlwaysTrue)`
  instead of `fast_append`. Without the flag the path is byte-identical to
  before; empty non-flag commits still produce no snapshot (measured, kept).
- `overwrite_files.rs`: `OverwriteFilesOperation::allows_empty_commit`
  returns `row_filter.is_some()`. Classification already matched Java
  (`containsDeletes` counts a set filter before resolution): adds + filter
  gives `overwrite`, filter-only gives `delete`.
- `snapshot.rs`: new `SnapshotProduceOperation::allows_empty_commit`
  (default false) derives the truly-empty guard. Only the filter-requested
  overwrite passes it; `test_empty_overwrite_is_rejected` still holds.
- Totals work with no code change: the staged replace reset `main`, so the
  producer seeds from zero. Added-only gives `total-*` equal to added and no
  `deleted-*` keys; the empty commit gives `total-*` zero plus
  `changed-partition-count` zero and no `added-*` keys.
- C-002: the flag works from both `begin_create` and `begin_replace`; the
  caller states it. C-004: no existing caller touches the flag.

## Mutation proof (test-adequacy, one knob at a time)

- Baseline: `cargo test -p iceberg --lib transaction::staged_table::rtas`,
  6 passed.
- M1 `materialize_pending`: `if self.replace_write` forced to `if false`:
  4 red out of 6 (the four replace-write tests; the two CTAS pins green).
- M2 `allows_empty_commit` override forced to `false`: 2 red out of 6 (the
  two empty-delete tests; the rest green).
- Both restored; suite back to 6 green.

## Gate output

Round 1, all green 2026-09-17 (`CARGO_BUILD_JOBS=10 RUST_TEST_THREADS=8`):

- `cargo test -p iceberg --lib transaction`: 652 passed, 0 failed.
- `cargo test -p iceberg --lib catalog`: 190 passed, 0 failed.
- `cargo test -p iceberg-catalog-glue --lib`: 50 passed, 0 failed.
- `cargo clippy -p iceberg --all-targets -- -D warnings`: clean.
- `make check`: exit 0 (fmt, clippy, taplo, machete, agent-artifacts,
  matrix-anchors, comment-blocks, file-size all OK).
- `typos .`: exit 0.

## Round 2 (orchestrator review of round 1)

- Q-20c-1: ceilings only move DOWN. Restored the three round-1 raises and
  moved code out instead: `staged_table.rs` inline `mod tests` (minus the F-1
  test) into `staged_table_tests.rs` wired with `#[path]` (`mod
  staged_tests`); `FirstRowIdPolicy` into
  `snapshot/first_row_id_policy.rs` with a re-export;
  `OverwriteFilesOperation` into `overwrite_files_operation.rs` as a
  `pub(crate)` child-module struct. The F-1 test (whose 7-line comment block
  the comment gate counts as added when moved) stays inline in
  `staged_table.rs` verbatim with the two helpers it needs. Final ceilings:
  staged_table row REMOVED (510 lines, under the 1000 default), snapshot
  3490 to 3486, overwrite_files 3429 to 3383.
- Q-20c-2: `allows_empty_commit` is now an explicit opt-in on
  `OverwriteFilesAction` (default false, builder `allow_empty_commit()`),
  threaded into the operation; only `StagedTableTransaction` sets it, and
  only on the `replace_write` path. Pin test
  `test_empty_overwrite_by_row_filter_is_rejected_without_opt_in` proves a
  plain filter-only empty overwrite still fails `PreconditionFailed`.
  Mutation proof: action default flipped to true gives 2 red out of 2 (the
  pin plus the pre-existing truly-empty rejection — the flag permits any
  empty commit when set, which only the staged path requests).
- Rebase on `origin/main` (`5a0666b9`): clean, no conflicts. #287 took row
  R171, so the RTAS row is now row R172 (map reference fixed; anchors green).

## Round 3 (critic-logic NEEDS_REMEDIATION, rtas-logic-1-report.md)

Base: orchestrator rebase onto fork main 96fc9f1f, pushed as f8e80333.
Shipping code correct; pins too weak. Remediation: L-001 live keep-set pins
with critic-mutation reds, L-002 covered by L-001, L-003 v3 row-id and
partitioned/sort pins.

## Round 2 gate output

All green 2026-09-17 (`CARGO_BUILD_JOBS=10 RUST_TEST_THREADS=8`):

- `cargo test -p iceberg --lib transaction`: 653 passed, 0 failed.
- `cargo test -p iceberg --lib catalog`: 190 passed, 0 failed.
- `cargo test -p iceberg-datafusion`: all suites green (228 + 20 + 7 + 1 +
  6 + 1 + 5 + 4 + 7 passed, 0 failed).
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D
  warnings`: clean.
- `make check`: exit 0 (fmt, clippy, taplo, machete, agent-artifacts,
  matrix-anchors, comment-blocks, file-size 481 clean / 98 ceilings).
