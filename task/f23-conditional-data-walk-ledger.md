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

Model: grok-4.6 (round 1). Model: claude-opus-5 (medium) (round 2). Base: fork main `c1d6c9de`. Branch `repark/f23-conditional-data-walk`.
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
| `crates/iceberg/src/delete_vector_container.rs` | The walk runs when `pending_legacy` is non-empty or a touched path is missing from `known_partitions`. `collect_live_data_files` stops at `files.len() == wanted.len()`; r2 loads the first data manifest alone, buffers the tail into a `FuturesOrdered`, borrows the manifest entries, and takes only the paths the map missed as `wanted` on the no-legacy arm. Field `///` corrected: total with legacy deletes, else only the paths the map missed |
| `crates/iceberg/src/delete_vector_container/tests.rs` | F-18 zero-read pin restored; F-22 still-reads-once is the legacy arm; first-of-192 early-exit pin |
| `crates/iceberg/src/delete_vector_container/counting.rs` | counting Storage extracted so tests stay under the file-size ceiling |
| `crates/iceberg/src/delete_vector_container/measure.rs` | ignored 8/48/192 close wall |

HALT check: skip fires only when `pending_legacy` is empty. F-21 batteries have live parquet deletes, so the walk still runs and `seq_applies` still sees sequence numbers.

## 4. Pins

| id | knob | result |
|---|---|---|
| P-skip | counting Storage, 3 data manifests, complete map, no legacy | `data_manifest_reads == 0`; `data_sequence_numbers` empty (`a_supplied_partition_map_reads_no_data_manifest`) |
| P-legacy | counting Storage, 2 data manifests, complete map, live parquet deletes | `data_manifest_reads == 2`; sequences contain both paths (`a_supplied_partition_map_still_reads_each_data_manifest_once`) |
| P-exit | counting Storage, 192 data manifests, no map, touch the newest file (first in added-data list order) | `reads == 1` exactly, r2 (`a_touched_file_in_the_first_data_manifest_stops_the_walk`) |

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
| r2: with no legacy deletes and a partial map, sequences cover only the paths the map missed, not every touched path. |

## 10. Section 9 delivery template

```text
Charter clauses: C-001 through C-004
Matrix rows: R114 (F-23 conditional data-manifest walk)
Base: c1d6c9de
Java methods or bytecode read: none new (F-21 seq_applies / F-18 skip restored)
Files changed: crates/iceberg/src/delete_vector_container.rs; crates/iceberg/src/delete_vector_container/tests.rs; crates/iceberg/src/delete_vector_container/counting.rs; crates/iceberg/src/delete_vector_container/measure.rs; docs/parity/GAP_MATRIX.md R114; maps; task/todo.md; task/f22-legacy-scan-one-pass-ledger.md §9; task/f23-conditional-data-walk-ledger.md
Behavior before: close always walked every data manifest so data_sequence_numbers was total, including with a complete known_partitions map and no legacy
Behavior after: the walk is skipped when pending_legacy is empty and known_partitions covers every touched path (data_sequence_numbers empty); otherwise the first data manifest is loaded alone, the tail is buffered, and the walk stops once every wanted path is found; with no legacy deletes only the paths the map missed are wanted (r2)
Negative cases: a complete map with live parquet deletes still walks; a missing map still walks; removing the wanted.len() stop reads all 192; prefilling the buffer issues 8 GETs on a latent store; wanting the full touched set reads all 192 (r2)
Test command and population: cargo test -p iceberg --locked --lib delete_vector_container
Mutations, one at a time: see section 6
Java interop command and fixture count: run-interop-f21-legacy-delete-merge.sh (2 fixtures); run-interop-f18-dv-sibling-close.sh
CI-only evidence gap: Docker legs of make test excused
Breaking public API change: none (data_sequence_numbers contract narrowed: total only on the legacy arm, else only the paths known_partitions missed, empty when the walk is skipped)
Critic attestation: Actor only (this unit); round 2 is the Rust perf review's findings
Open findings and dispositions: R114 residues unchanged (equality-delete sort-order; Spark-job-written shared Puffin). r2: the iceberg-datafusion caller still passes an empty known_partitions, so the skip is RePark-only — task/todo.md row
```

## 11. Round 2 — Rust perf review (Opus 5 HIGH, 2026-09-03)

Model: claude-opus-5 (medium). Commits on top of `409e81b3`.

| sev | finding | disposition |
|---|---|---|
| S1 | `buffered(DV_IO_CONCURRENCY)` prefills 8 `load_manifest` futures, so a wanted set satisfied by the newest manifest issues 8 GETs on any store whose read is not ready on the first poll | FIXED — `FuturesOrdered` with a budget of 1 until the first manifest is consumed, `DV_IO_CONCURRENCY` after |
| S2 | `wanted` was every touched path even when `known_partitions` resolved all but one, so the `files.len() < wanted_len` exit could not fire | FIXED — with no legacy deletes only the unresolved paths are wanted; the legacy arm keeps the full set (`seq_applies` needs a sequence for every `item.touched`) |
| S2 | the early-exit pin was a range | FIXED — exact `== 1`, plus two latent-store pins |
| S2 | `delete_legacy_merge.rs` passes an empty `known_partitions`, so the skip never fires for iceberg-datafusion's own V3 DELETE/UPDATE | DOCUMENTED — handing partitions down means calling `live_data_file_partitions`, itself a full unstoppable walk, so it would be a regression; `task/todo.md` row + R114 + `physical_plan/map.md` |
| S3 | `Vec<ManifestFile>` clone per walk | FIXED — `Vec<&ManifestFile>`, `&FileIO` borrowed |
| S3 | `buffered` head-of-line blocking on the full-walk arm | TRADE RECORDED — an ordered buffer is the price of a deterministic stop; unordered would make the early exit non-reproducible and the count pins flaky. Cost is bounded: one extra round trip at the head |

### Why local FS hid S1

| step | local FS | latent store |
|---|---|---|
| `Buffered::poll_next` prefills 8 futures | yes | yes |
| `FuturesUnordered::poll_next` returns at the FIRST completion | the first read completes on its first poll, so 7 futures are never polled and never counted | every read returns Pending, so all 8 are polled and all 8 GETs are issued |

`CountingStorage::read` gained a `latent` flag: `yield_now().await` after `count(path)`, so the counter records an issued GET the way a remote store would.

### Round-2 pins

| id | knob | result |
|---|---|---|
| P-exit | 192 data manifests, no map, touch the newest file | `data_manifest_reads == 1` exactly (`a_touched_file_in_the_first_data_manifest_stops_the_walk`) |
| P-latent | same, latent counting Storage | `data_manifest_reads == 1` (`a_latent_store_reads_one_data_manifest_for_a_newest_file_hit`) |
| P-wanted | 192 data manifests, latent, touched = {newest, oldest}, map covers the oldest only | `data_manifest_reads == 1` and `data_sequence_numbers` keys `== [newest]` (`only_the_paths_the_map_misses_are_wanted_without_legacy_deletes`) |

### Round-2 mutations (one knob at a time)

| id | knob | population | result |
|---|---|---|---|
| M4 | `let budget = DV_IO_CONCURRENCY;` (prefill again) | 3 (the early-exit pins) | 2 red of 3 (`a_latent_store_…` left 8 right 1; `only_the_paths_…` left 8 right 1; the non-latent pin stays green — that is the point) |
| M5 | want the full touched set unless nothing is unresolved | 3 | 1 red of 3 (`only_the_paths_…` left 192 right 1) |
| M6 | drop the `files.len() < wanted_len` stop (`loop`) | 3 | 3 red of 3 (all left 192 right 1) |

Restored from `/tmp/oc-f23/.dvc-r2-good.rs` after each; `touch` after restore.

### Round-2 measurement (same fixture and command as section 5)

| n | run 1 | run 2 | run 3 | r2 median | r1 median |
|---|---|---|---|---|---|
| 8 | 1.200 ms | 1.219 ms | 1.214 ms | 1.214 ms | 1.239 ms |
| 48 | 1.660 ms | 1.702 ms | 1.706 ms | 1.702 ms | 1.667 ms |
| 192 | 3.843 ms | 3.880 ms | 3.871 ms | 3.871 ms | 4.075 ms |

`sequences=0` on every cell (the skip still fires). The legacy arm is unchanged: `P-legacy` still reads each of its 2 data manifests once.

### Round-2 gate exits

| gate | exit |
|---|---|
| `make check` | 0 (fmt, clippy `-D warnings`, taplo, machete, agent-artifacts, matrix anchors, comment-blocks, rust-file-size 433 files / 101 legacy) |
| `cargo test -p iceberg --locked --offline` | 0 (`3601 passed; 0 failed; 7 ignored` lib; doctests 90 passed / 10 ignored) |
| `cargo test -p iceberg-datafusion --locked --offline` | 0 (`211 passed` lib; `f21_legacy_delete_merge` 6 passed; `shared_puffin_dv` 20 passed / 3 ignored) |
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 0 (`PASSED`; 2 fixtures; sabotages red) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 0 (`PASSED`; sabotage red) |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 (`OK: GAP_MATRIX anchors sound`) |
| `python3 scripts/check_rust_file_size.py` | 0 (`433 files clean`; `delete_vector_container.rs` 572, `tests.rs` 930) |
