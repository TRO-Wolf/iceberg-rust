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

# F-21 — DataFusion V3 DV merges legacy parquet position deletes (Java BaseDVFileWriter.loadPreviousDeletes)

Model: muse-spark-1.2-contributor (rounds 1–2), grok-4.6 (rounds 3–5)

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | V2 MoR + file-scoped parquet delete (id 2) upgraded to V3, DELETE id 3 refuses before IO (red) | PROVEN | `f21_legacy_delete_merge.rs::test_f21_base_cell_delete_merges_parquet_into_dv` red at `ff4764d3e` (guard `BaseDVFileWriter.loadPreviousDeletes is deferred`), green after |
| C-002 | V3 DELETE/UPDATE merges file-scoped parquet deletes via DV and removes the superseded file; partition-scoped merges per-file positions but keeps the parquet live (Spark-equal) | PROVEN | `write_deletion_vectors` collects `referenced_data_file_location` file-scoped vs partition-scoped, loads positions by reserved ids 2147483546/2147483545 filtered to the data file, only `delete_seq >= data_seq`, union once, `DVFileWriter::with_previous_deletes` + `RowDelta::remove_deletes_many` for file-scoped, `close_touched_dv_containers` for siblings; `validate_fresh_dvs_only` now only blocks file-scoped |
| C-003 | Pins: base cell rc 2, two file-scoped parquet deletes on one data file, UPDATE, untouched-file control, partition-scoped merge-and-keep, sequence-number control; Java interop proves byte-level DV and rows plus partition-scoped coexistence | PROVEN | `f21_legacy_delete_merge.rs` 6 tests (base, two-file-scoped rc 3, UPDATE, untouched, partition-scoped rc 2 + parquet kept, seq not apply on warehouse parquet); `interop_f21_legacy_delete_merge` file-scoped seed → DV rc 2 parquet 0 rows (1,4); partition seed → Java `{4=d}` one puffin + one parquet; `run-interop-f21-legacy-delete-merge.sh` 2 fixtures; three sabotages red |

## 2. Java decode

| class / method | decisive | consequence |
|---|---|---|
| `core/.../deletes/BaseDVFileWriter` `loadPreviousDeletes` (L114-129) | for each touched path, `PositionDeleteIndex` filtered by `dataFile.path()`, `delete_seq >= data_seq` | legacy positions are per-file, sequence-filtered |
| `api/.../ContentFileUtil.isFileScoped` / `referencedDataFile` | `referenced_data_file != null` OR equal `file_path` lower/upper bounds | file-scoped vs partition-scoped |
| `api/.../DeleteFileIndex` `findDV` / `findPosPartitionDeletes` | DV hit early-returns, position deletes partition-keyed | DV supersedes position deletes per-file |
| `core/.../Transaction` `RowDelta.removeDeletes` + `validate_fresh_dvs_only` | `remove_deletes_many` in same commit escapes the door | superseded file must be removed in same commit |

## 3. Production change

| file | change |
|---|---|
| `crates/iceberg/src/arrow/delete_file_loader.rs` | `BasicDeleteFileLoader` stays `pub(crate)`; pub `load_position_deletes_by_path` (reserved ids, interned path keys, `Vec<u64>`) |
| `crates/iceberg/src/transaction/row_delta_fresh_dv.rs` | door now only blocks file-scoped (`referenced_data_file_location.is_some()`), message no longer "deferred" |
| `crates/integrations/datafusion/src/physical_plan/delete_legacy_merge.rs` | NEW: legacy collect, filtered load, union once, file-scoped removal via `DvContainerClose::removed` |
| `crates/integrations/datafusion/src/physical_plan/delete.rs` | split (1700), delegates to `delete_legacy_merge::write_deletion_vectors`, test imports trimmed |
| `scripts/check_rust_file_size.py` | ceiling `delete.rs` 2071 -> 1700; `row_delta.rs` restored 6385 -> 6366 |

## 4. Pins

| id | knob | result |
|---|---|---|
| M1 | skip `extra.extend` (no legacy positions merged) | 4 red of 6 (`test_f21_base_cell_delete_merges_parquet_into_dv`, `test_f21_two_file_scoped_parquet_deletes_merge_into_one_dv`, `test_f21_partition_scoped_merge_keeps_parquet`, `test_f21_update_merges_parquet_into_dv`) |
| M2 | `file_scoped_to_remove` not populated | 3 red of 6 (`test_f21_base_cell_delete_merges_parquet_into_dv`, `test_f21_two_file_scoped_parquet_deletes_merge_into_one_dv`, `test_f21_update_merges_parquet_into_dv`) |
| M3 | `if item.referenced.is_some()` → `if true` (partition-scoped also removed) | 1 red of 6 (`test_f21_partition_scoped_merge_keeps_parquet`) |
| M4 | `seq_applies` always true | 1 red of 6 (`test_f21_sequence_number_not_apply`: DV `record_count` 2 vs 1) |
| C2 | merge every position of the delete file into the touched DV | 1 red of 6 (`test_f21_partition_scoped_merge_keeps_parquet`: rc 3 vs 2) |
| M5 | Java sabotage (corrupt `expected_rows.json`; corrupt `added-dvs`; corrupt `expected_part_rows.json`) | oracle red (runner) |
| M6 | `loaded.get(item.file.file_path())` → `loaded.values().next()` | 1 red of 6 (`test_f21_two_file_scoped_parquet_deletes_merge_into_one_dv`: DV `record_count` 2 vs 3) |

## 4b. Perf (round 4, debug, this clone, no CI pin)

Command:
`cargo test -p iceberg-datafusion --locked --test f21_legacy_delete_merge_measure -- --ignored --nocapture --exact test_f21_measure_k8_partition_scoped_100k`
`cargo test -p iceberg-datafusion --locked --test f21_legacy_delete_merge_measure -- --ignored --nocapture --exact test_f21_measure_row_column_100k`

Before = cache off (`seen_load.insert` or true, reload the same parquet per touched file) and full parquet read (`parquet_to_batch_stream`). After = load-once + `parquet_positional_delete_batch_stream`. Three runs each.

| knob | before (this clone, cache/projection off) | after (this clone) |
|---|---|---|
| P1-a K=8 one partition-scoped delete of 100k positions, one DELETE touching all eight | 928.018 / 912.377 / 917.150 ms | 282.916 / 286.601 / 291.029 ms |
| P1-c file-scoped 100k positions with a 200-byte `row` column | 102.440 / 108.586 / 101.711 ms | 96.846 / 99.225 / 98.515 ms |

## 5. Interop

| command | fixtures | sabotage |
|---|---|---|
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 2 (`expected_rows.json` + `expected_part_rows.json`) | (6) `expected_rows.json` id 1→999; (6b) `added-dvs` 1→99; (8) `expected_part_rows.json` id→999; all three oracle red |

## 6. Gate exits

Filled from PROGRESS.md after each gate is run. A gate not run is NOT RUN.

| gate | exit |
|---|---|
| `make check` | 0 (`430 files clean`; GAP_MATRIX 5-pipe audit green) |
| `cargo test -p iceberg --locked --lib` | 0 (`test result: ok. 3582 passed; 0 failed; 2 ignored`) |
| `cargo test -p iceberg-datafusion --locked --test f21_legacy_delete_merge` | 0 (`test result: ok. 6 passed; 0 failed; 0 ignored`) |
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 0 (`PASSED`; 2 fixtures; row sabotage red; added-dvs sabotage red; partition `{4=d}` parquet live; partition row sabotage red) |
| `typos .` | 0 |
| `python3 scripts/check_rust_file_size.py` | 0 (`430 files clean`) |

## 7. RePark

| note |
|---|
| repin unit closes `V3-UPGRADE-DV-PLAIN-1`; keep `V3-UPGRADE-DV-PART-1` |

## 8. Section 9 delivery template

```text
Charter clauses: C-001 through C-004
Matrix rows: R114 (F-21 legacy delete merge)
Java methods or bytecode read: BaseDVFileWriter.loadPreviousDeletes (L114-129), ContentFileUtil.isFileScoped / referencedDataFile, DeleteFileIndex.findDV / findPosPartitionDeletes, RowDelta.removeDeletes + validate_fresh_dvs_only
Files changed: crates/iceberg/src/arrow/delete_file_loader.rs; crates/iceberg/src/transaction/map.md; crates/iceberg/src/transaction/row_delta.rs; crates/iceberg/src/transaction/row_delta_fresh_dv.rs; crates/iceberg/src/transaction/to_branch.rs; crates/integrations/datafusion/src/physical_plan/cow_affected.rs; crates/integrations/datafusion/src/physical_plan/delete.rs; crates/integrations/datafusion/src/physical_plan/delete_legacy_merge.rs; crates/integrations/datafusion/src/physical_plan/map.md; crates/integrations/datafusion/src/physical_plan/mod.rs; crates/integrations/datafusion/tests/f21_legacy_delete_merge.rs; crates/integrations/datafusion/tests/f21_legacy_delete_merge_measure.rs; crates/integrations/datafusion/tests/integration_datafusion_test.rs; crates/integrations/datafusion/tests/interop_f21_legacy_delete_merge.rs; crates/integrations/datafusion/tests/map.md; dev/java-interop/map.md; dev/java-interop/run-interop-f21-legacy-delete-merge.sh; dev/java-interop/src/main/java/org/apache/iceberg/InteropOracle.java; docs/parity/GAP_MATRIX.md; scripts/check_rust_file_size.py; scripts/run_interop_suites.sh; task/f21-legacy-delete-merge-ledger.md; task/todo.md
Behavior before: V3 `DELETE`/`UPDATE` on a table upgraded from V2 with a live parquet position delete refused pre-IO with "BaseDVFileWriter.loadPreviousDeletes is deferred"
Behavior after: for each touched data file, live file-scoped parquet deletes are loaded by reserved ids filtered to the file, only `delete_seq >= data_seq`, unioned once into the new DV (rc 2), and the superseded parquet file is removed in the same RowDelta; partition-scoped deletes are merged per-file but kept live (Spark-equal)
Negative cases: untouched file's parquet stays live; old delete_seq < data_seq does not apply; partition-scoped kept; file-scoped door still refuses unless removed in the same commit; two file-scoped deletes on one data file union into one DV
Test command and population: `dev/java-interop/run-interop-f21-legacy-delete-merge.sh`; `cargo test -p iceberg-datafusion --locked --test f21_legacy_delete_merge` (6 passed)
Mutations, one at a time: `cargo test -p iceberg-datafusion --locked --test f21_legacy_delete_merge`; M1 4 red of 6; M2 3 red of 6; M3 1 red of 6; M4 1 red of 6; C2 1 red of 6; M5 oracle red (runner); M6 1 red of 6
Java interop command and fixture count: `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` — 2 fixtures; three sabotages FAIL-closed
CI-only evidence gap: Docker legs of make test excused
Breaking public API change: none (`BasicDeleteFileLoader` stays `pub(crate)`; `load_position_deletes_by_path` is the pub free function)
Critic attestation: round 4 closed R1–R8; round 5 closed B1–B5
Open findings and dispositions: R114 residues: equality-delete sort-order, Spark-job-written shared Puffin
```
