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

# F-23 — DV container close walks data manifests only when it needs them

Model: grok-4.6. Base: fork main `c1d6c9de`. Branch `repark/f23-conditional-data-walk`.
Matrix row: R114.

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | Skip `collect_live_data_files` when `pending_legacy` is empty AND `known_partitions` covers every touched path; walk when either fails. `data_sequence_numbers` is total after a walk, empty when skipped | PROVEN | `a_supplied_partition_map_reads_no_data_manifest`; `a_supplied_partition_map_still_reads_each_data_manifest_once`; field `///` |
| C-002 | The walk stops once every wanted path is found | PROVEN | `a_touched_file_in_the_first_data_manifest_stops_the_walk` |
| C-003 | Before/after at 8/48/192 data manifests on the pure-DV complete-map fixture, three runs; legacy arm still walks | PROVEN | section 5 |
| C-004 | Docs, maps, this ledger | PROVEN | row R114, `task/todo.md`, maps, F-22 §9 |

## 2. Why the walk was unconditional

| fact | consequence |
|---|---|
| F-22 always called `collect_live_data_files` so `data_sequence_numbers` was total | RePark measured +39 % at 192 data manifests on the pure-DV path |
| Only in-close consumer of `data_sequence_numbers` is `seq_applies` inside `if !legacy_arcs.is_empty()` | On the pure-DV complete-map path the whole pass is thrown away |
| RePark never reads the field | Skip is safe when `pending_legacy` is empty and the map covers every touched path |

## 3. Production change

| file | change |
|---|---|
| `crates/iceberg/src/delete_vector_container.rs` | `walk_data` is true when `pending_legacy` is non-empty or any touched path is missing from `known_partitions`. `collect_live_data_files` uses `.buffered(DV_IO_CONCURRENCY)` and stops at `files.len() == wanted.len()`. Field `///` corrected: total after a walk, empty when skipped |
| `crates/iceberg/src/delete_vector_container/tests.rs` | F-18 zero-read pin restored; F-22 still-reads-once is the legacy arm; first-of-192 early-exit pin |
| `crates/iceberg/src/delete_vector_container/counting.rs` | counting Storage extracted so tests stay under the file-size ceiling |
| `crates/iceberg/src/delete_vector_container/measure.rs` | ignored 8/48/192 close wall |

HALT check: skip fires only when `pending_legacy` is empty. F-21 batteries have live parquet deletes, so the walk still runs and `seq_applies` still sees sequence numbers.

## 4. Pins

| id | knob | result |
|---|---|---|
| P-skip | counting Storage, 3 data manifests, complete map, no legacy | `data_manifest_reads == 0`; `data_sequence_numbers` empty (`a_supplied_partition_map_reads_no_data_manifest`) |
| P-legacy | counting Storage, 2 data manifests, complete map, live parquet deletes | `data_manifest_reads == 2`; sequences contain both paths (`a_supplied_partition_map_still_reads_each_data_manifest_once`) |
| P-exit | counting Storage, 192 data manifests, no map, touch the newest file (first in added-data list order) | `0 < reads < 192` and `reads <= 2 * DV_IO_CONCURRENCY` (`a_touched_file_in_the_first_data_manifest_stops_the_walk`) |

## 5. Measurement (debug, this clone, no CI pin)

Command:
`cargo test -p iceberg --locked --offline --lib delete_vector_container::measure::measure_close_at_8_48_192_data_manifests -- --ignored --nocapture --exact`

Pure-DV, complete `known_partitions`, 0 legacy. Three debug runs.

| n | before run 1 | before run 2 | before run 3 | before median | after run 1 | after run 2 | after run 3 | after median |
|---|---|---|---|---|---|---|---|---|
| 8 | 29.545 ms | 29.952 ms | 29.678 ms | 29.678 ms | 1.239 ms | 1.219 ms | 1.296 ms | 1.239 ms |
| 48 | 188.956 ms | 173.318 ms | 185.052 ms | 185.052 ms | 1.663 ms | 1.667 ms | 2.072 ms | 1.667 ms |
| 192 | 838.047 ms | 712.756 ms | 691.531 ms | 712.756 ms | 3.830 ms | 4.075 ms | 4.343 ms | 4.075 ms |

Before: `sequences=1` (walk ran). After: `sequences=0` (walk skipped). Median 192: 712.756 ms → 4.075 ms.

Early-exit count (no map, 192 data manifests, touch first list entry): `data_manifest_reads=1`. M3 without the stop is `reads=192`.

Legacy arm cost: the walk still runs (`P-legacy` `data_manifest_reads == 2`).

## 6. Mutations (one knob at a time)

| id | knob | population | result |
|---|---|---|---|
| M1 | make `collect_live_data_files` unconditional again | 2 (`a_supplied_partition_map_*`) | 1 red of 2 (`a_supplied_partition_map_reads_no_data_manifest`; left 3 right 0) |
| M2 | drop the `!pending_legacy.is_empty()` disjunct (skip the legacy arm) | 2 (`a_supplied_partition_map_*`) | 1 red of 2 (`a_supplied_partition_map_still_reads_each_data_manifest_once`; left 0 right 2) |
| M3 | consume the whole data-manifest stream (no `wanted.len()` stop) | 1 (`a_touched_file_in_the_first_data_manifest_stops_the_walk`) | 1 red of 1 (`reads=192`) |

Restored from `/tmp/oc-f23-dvc.rs.bak` after each; `touch` after restore.

## 7. Interop

| command | fixtures | sabotage |
|---|---|---|
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 2 | FAIL-closed (F-21) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 4 + final.metadata.json | FAIL-closed (F-18) |

## 8. Gate exits

Filled from PROGRESS.md after each gate is run. A gate not run is NOT RUN.

| gate | exit |
|---|---|
| `make check` | 0 (fmt, clippy `-D warnings`, taplo, machete, agent-artifacts, matrix anchors, comment-blocks, rust-file-size 433 files / 101 legacy) |
| `cargo test -p iceberg --locked` | 0 (`3599 passed; 0 failed; 7 ignored` lib; doctests 90 passed / 10 ignored) |
| `cargo test -p iceberg-datafusion --locked` | 0 (`211 passed` lib; `f21_legacy_delete_merge` 6 passed; `shared_puffin_dv` 20 passed / 3 ignored) |
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 0 (`PASSED`; 2 fixtures; sabotages red) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 0 (`PASSED`; sabotage red) |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 (`OK: GAP_MATRIX anchors sound`) |
| `python3 scripts/check_rust_file_size.py` | 0 (`433 files clean`; `delete_vector_container.rs` 568) |

Docker legs of `make test` excused.

## 9. RePark

| note |
|---|
| RP-9 repins. RePark's `known_partitions` sink is what makes the skip fire — keep it. |
| `data_sequence_numbers` is empty on the pure-DV complete-map path; do not assume totality. |
| On the legacy path the walk still runs and sequences are total. |

## 10. Section 9 delivery template

```text
Charter clauses: C-001 through C-004
Matrix rows: R114 (F-23 conditional data-manifest walk)
Base: c1d6c9de
Java methods or bytecode read: none new (F-21 seq_applies / F-18 skip restored)
Files changed: crates/iceberg/src/delete_vector_container.rs; crates/iceberg/src/delete_vector_container/tests.rs; crates/iceberg/src/delete_vector_container/counting.rs; crates/iceberg/src/delete_vector_container/measure.rs; docs/parity/GAP_MATRIX.md R114; maps; task/todo.md; task/f22-legacy-scan-one-pass-ledger.md §9; task/f23-conditional-data-walk-ledger.md
Behavior before: close always walked every data manifest so data_sequence_numbers was total, including with a complete known_partitions map and no legacy
Behavior after: the walk is skipped when pending_legacy is empty and known_partitions covers every touched path (data_sequence_numbers empty); otherwise it is buffered and stops once every wanted path is found
Negative cases: a complete map with live parquet deletes still walks; a missing map still walks; removing the wanted.len() stop reads all 192
Test command and population: cargo test -p iceberg --locked --lib delete_vector_container
Mutations, one at a time: see section 6
Java interop command and fixture count: run-interop-f21-legacy-delete-merge.sh (2 fixtures); run-interop-f18-dv-sibling-close.sh
CI-only evidence gap: Docker legs of make test excused
Breaking public API change: none (data_sequence_numbers contract narrowed: empty when the walk is skipped)
Critic attestation: Actor only (this unit)
Open findings and dispositions: R114 residues unchanged (equality-delete sort-order; Spark-job-written shared Puffin)
```
