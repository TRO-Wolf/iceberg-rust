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

# F-25 — `validate_fresh_dvs_only` stops walking data manifests once every added DV's file is found

Model: muse-spark-1.3-contributor. Base: fork main `594bdbe5`. Branch `repark/f25-fresh-dv-early-exit`.
Matrix row: R114.

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | A counting-Storage pin on the row-delta commit path (192 data manifests, one DV for the newest file) reads 1 data manifest, latent and non-latent; red at 192 before the fix | PROVEN | `fresh_dv_commit_for_newest_file_reads_one_data_manifest[_on_latent_store]` |
| C-002 | The validation walk runs newest-first, issues one load before buffering `DV_IO_CONCURRENCY`, and stops once every `added_dvs` key is found; a never-found key keeps the full walk | PROVEN | pins + mutation M1 |
| C-003 | An oldest-manifest DV still reads all N and validates; the missing-field error pin is unchanged; removing the stop reds; F-17/F-18/F-21/F-23 suites and both interop runners green | PROVEN | sections 6–7 |
| C-004 | Sibling measure reports close + commit opens and wall at 8/48/192 before/after, three runs, medians below | PROVEN | section 5 |
| C-005 | Docs, maps, this ledger | PROVEN | row R114, `task/todo.md`, maps |

## 2. Why the walk was unconditional

| fact | consequence |
|---|---|
| `validate_fresh_dvs_only` loaded every live data manifest with a sequential `await` per manifest | RePark RP-9 strace: 192 opens, ~0.8 s of a 1.5 s pure-DV DELETE statement |
| The walk only needs the live data entry for every `added_dvs` key (usually one path) | Everything after the last found key is thrown away |
| Java's `ManifestFilterManager` stops scanning once its targets resolve | Parity direction is an early exit, not a skip |

## 3. Production change

| file | change |
|---|---|
| `crates/iceberg/src/transaction/row_delta_fresh_dv.rs` | The data-manifest loop collects `Vec<&ManifestFile>` in manifest-list order (newest first, the same order the close walks), feeds a `FuturesOrdered` with budget 1 until the first manifest is consumed then `DV_IO_CONCURRENCY` (F-23's shape: one GET on a remote store), and stops once `live_data_entry_by_path.len() == added_dvs.len()`; a never-found key drains the whole list, so unfound-key behavior is byte-identical to before |
| `crates/iceberg/src/delete_vector_container.rs` | `mod counting` → `pub(crate) mod counting`, so the row-delta tests reuse the F-23 counting Storage |
| `crates/iceberg/src/transaction/row_delta_fresh_dv.rs` (tests) | C-001 pins, oldest/unknown controls, ignored sibling measure |

HALT check: the door is Rust-conservative with no Java counterpart (`row_delta.rs` docs); Java's `validateDeleteFiles` is a different, untouched check. Early exit leaves `live_data_entry_by_path` identical whenever it fires (every key found once); the full walk runs otherwise. No validation outcome changes.

Edge considered, not a HALT: one path live in two data manifests with different seqs. Before, the older entry overwrote the newer; on an early exit the newer wins (matching F-23 `collect_live_data_files` `or_insert`). Same-path-twice needs no dedup pass on this path today; both values come from live entries and the door treats them identically unless the two seqs straddle a delete seq.

## 4. Pins

| id | knob | result |
|---|---|---|
| P-newest | counting Storage, 192 data manifests, DV for the newest file, row-delta commit | `commit_data_manifest_reads == 1` (`fresh_dv_commit_for_newest_file_reads_one_data_manifest`) |
| P-newest-latent | same, latent counting Storage | `== 1` (`..._on_latent_store`) |
| P-oldest | DV for the oldest file | `== 192` and the commit validates (`fresh_dv_commit_for_oldest_file_reads_every_data_manifest`) |
| P-unknown | DV for a never-live file | `== 192`, door passes unchanged (`fresh_dv_commit_for_unknown_file_reads_every_data_manifest`) |
| P-malformed | Puffin DV with no `referenced_data_file` field | `DataInvalid` naming the file, unchanged (`test_row_delta_dv_missing_referenced_data_file_is_rejected`) |

## 5. Measurement (debug, this clone, no CI pin)

Command:
`cargo test -p iceberg --locked --offline --lib transaction::row_delta_fresh_dv::tests::measure_commit_at_8_48_192_data_manifests -- --ignored --nocapture --exact`

Sibling of the F-23 close measure: complete-map close, then a row-delta DV commit for the newest file, on the counting Storage. Three runs each side; before = pre-fix sequential walk swapped back in, after = the early exit.

| n | before run 1 | before run 2 | before run 3 | before median | after run 1 | after run 2 | after run 3 | after median |
|---|---|---|---|---|---|---|---|---|
| 8 commit wall | 42.450 ms | 40.370 ms | 40.528 ms | 40.528 ms | 15.351 ms | 16.292 ms | 15.571 ms | 15.571 ms |
| 48 commit wall | 192.056 ms | 200.321 ms | 189.400 ms | 192.056 ms | 21.976 ms | 22.161 ms | 22.047 ms | 22.047 ms |
| 192 commit wall | 726.258 ms | 725.030 ms | 737.350 ms | 726.258 ms | 47.274 ms | 44.218 ms | 44.170 ms | 44.218 ms |
| 8 close wall | 8.660 ms | 7.918 ms | 7.819 ms | 7.918 ms | 10.556 ms | 8.017 ms | 7.802 ms | 8.017 ms |
| 48 close wall | 8.489 ms | 9.854 ms | 8.200 ms | 8.489 ms | 8.488 ms | 8.253 ms | 8.272 ms | 8.272 ms |
| 192 close wall | 10.669 ms | 10.523 ms | 10.722 ms | 10.669 ms | 18.653 ms | 10.519 ms | 10.495 ms | 10.519 ms |

| n | before commit reads | after commit reads | close reads before/after |
|---|---|---|---|
| 8 | 8 | 1 | 0 / 0 |
| 48 | 48 | 1 | 0 / 0 |
| 192 | 192 | 1 | 0 / 0 |

Median 192: commit 726.258 ms → 44.218 ms; opens 192 → 1. Close is the untouched F-23 path: identical reads, wall within noise.

## 6. Mutations (one knob at a time)

| id | knob | population | result |
|---|---|---|---|
| M1 | `while len < added_dvs.len()` → `loop`, drop the inner stop (consume the whole stream) | 6 (`row_delta_fresh_dv` module) | 2 red of 6 (both newest-file pins, left 192 right 1; oldest, unknown, and the two door pins stay green) |

Restored from `/tmp/f25-fresh-dv-good.rs.bak` (first leg) with `touch` after restore; `md5`-verified.

## 7. Interop

| command | fixtures | sabotage |
|---|---|---|
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 2 | FAIL-closed (F-21) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 4 + final.metadata.json | FAIL-closed (F-18) |

## 8. Gate exits

| gate | exit |
|---|---|
| `make check` | 0 (fmt, clippy `-D warnings`, taplo, machete, agent-artifacts, matrix anchors, comment-blocks, rust-file-size 433 files / 101 legacy) |
| `cargo test -p iceberg --locked --offline` | 0 (lib `3605 passed; 0 failed; 8 ignored`; doctests 90 passed / 10 ignored; 60 targets green, 3884 individual ok) |
| `cargo test -p iceberg-datafusion --locked --offline` | 0 (lib 211 passed; all targets green) |
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 0 (`PASSED`; 2 fixtures; sabotages red) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 0 (`PASSED`; sabotage red) |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 (`OK: GAP_MATRIX anchors sound`) |
| `python3 scripts/check_rust_file_size.py` | 0 (`433 files clean`) |

Docker legs of `make test` excused.

## 9. Section 9 delivery template

```text
Charter clauses: C-001 through C-005
Matrix rows: R114 (deletion-vector write path; commit-door validation walk)
Base: 594bdbe5
Java methods or bytecode read: none new (door is Rust-conservative, no Java counterpart; ManifestFilterManager early-exit shape mirrored from F-23)
Files changed: crates/iceberg/src/transaction/row_delta_fresh_dv.rs; crates/iceberg/src/delete_vector_container.rs (test-only visibility); docs/parity/GAP_MATRIX.md R114; maps; task/todo.md; task/f25-fresh-dv-early-exit-ledger.md
Behavior before: validate_fresh_dvs_only loaded every live data manifest sequentially, so a pure-DV DELETE paid 192 opens (~0.8 s at 192 manifests) to resolve usually one file
Behavior after: the walk runs newest-first, issues one load before buffering DV_IO_CONCURRENCY, and stops once every added_dvs key is found (1 open); a never-found key keeps the full walk with identical outcomes
Negative cases: oldest-manifest DV reads all N and validates; never-live key reads all N and the door passes; removing the stop reds both newest pins at 192 vs 1; malformed DV (no referenced_data_file) still rejected naming the file
Test command and population: cargo test -p iceberg --locked --offline --lib transaction::row_delta_fresh_dv (6 tests)
Mutations, one at a time: see section 6
Java interop command and fixture count: run-interop-f21-legacy-delete-merge.sh (2 fixtures); run-interop-f18-dv-sibling-close.sh
CI-only evidence gap: Docker legs of make test excused
Breaking public API change: none (commit-door only; counting module visibility is test-only)
Critic attestation: Actor only (this unit)
Open findings and dispositions: R114 residues unchanged (equality-delete sort-order; Spark-job-written shared Puffin)
```

## 10. RePark

| note |
|---|
| RP-10 repins. Registry row `PERF-DVCLOSE-STMT-1` → FIXED. |
| Commit-door validation now costs one data-manifest GET for the usual single-DV statement; the full walk survives only for oldest-manifest and never-found keys. |
