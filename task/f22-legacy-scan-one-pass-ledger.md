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

# F-22 — deletion-vector container close reports live non-DV position deletes (one-pass)

Model: grok-4.6. Base: F-21 tip `4537cb261`. Branch `repark/f22-legacy-scan-one-pass`.
Matrix row: R114.

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | One delete-manifest pass returns live non-Puffin position deletes that name a touched data file, plus touched data sequence numbers; optional pre-loaded `ManifestList` | PROVEN | `collect_delete_index` on `manifest_stream(Deletes)`; `DvContainerClose::legacy_deletes` / `data_sequence_numbers`; `preloaded_manifest_list_skips_the_list_reread` |
| C-002 | `load_legacy_positions` reads only `pos` (plus `file_path` when not file-scoped) through the projected parquet reader | PROVEN | `load_legacy_positions_projects_past_the_row_column`: bytes read < file size |
| C-003 | DataFusion delete exec uses the close result; no second delete-manifest walk | PROVEN | `delete_legacy_merge.rs::write_deletion_vectors` calls `close_touched_dv_containers_with_partitions` only; F-21 battery green |
| C-004 | Delete manifests read once with and without legacy; projected pin; wall at 8/192 recorded; mutations one knob at a time | PROVEN | section 4 / 5 |
| C-005 | Docs, maps, this ledger | PROVEN | row R114, `task/todo.md`, three `map.md` files |

## 2. Java decode

| class / method | decisive | consequence |
|---|---|---|
| `BaseDVFileWriter.loadPreviousDeletes` L114-129 | per touched path, `PositionDeleteIndex` filtered by path, `delete_seq >= data_seq` | merge belongs in the DV close, not a second caller walk |
| `ContentFileUtil.referencedDataFile` / `isFileScoped` | field or equal `file_path` bounds | file-scoped vs partition-scoped |
| F-21 `write_deletion_vectors` (this fork) | walked every delete manifest after close's collect | the second walk this unit removes |

## 3. Production change

| file | change |
|---|---|
| `crates/iceberg/src/delete_vector_container.rs` | one-pass collect of touched DVs + live parquet position deletes; `legacy_deletes` + `data_sequence_numbers`; optional `&ManifestList`; merge via `load_legacy_index` at `DV_IO_CONCURRENCY` |
| `crates/iceberg/src/delete_vector_container/legacy.rs` | `LegacyPositionDelete`; `load_legacy_positions`; pos-only vs path+pos projection |
| `crates/iceberg/src/arrow/delete_file_loader.rs` | name-based projection fallback for pos-only |
| `crates/integrations/datafusion/src/physical_plan/delete_legacy_merge.rs` | no manifest walk; close merges |

Concurrency bound: `DV_IO_CONCURRENCY = 8` (now `pub`).

## 4. Pins

| id | knob | result |
|---|---|---|
| P-del | counting Storage, 8 delete manifests, no applicable legacy | 1 red of 1 if count ≠ N (`delete_manifests_are_read_once_without_legacy_deletes`) |
| P-leg | same with 8 file-scoped parquet deletes | 1 red of 1 if count ≠ N (`delete_manifests_are_read_once_with_legacy_deletes`) |
| P-proj | pos-delete with reserved `row` column | 1 red of 1 if bytes ≥ file size |
| P-seq | close without partition map | `data_sequence_numbers` contains the touched path |
| P-list | pre-loaded `ManifestList` | delete manifests still once; list not re-read |

## 5. Measurement (debug, this clone, no CI pin)

Command: `cargo test -p iceberg --locked --offline --lib delete_vector_container::tests::measure_close_at_8_and_192_delete_manifests -- --ignored --nocapture --exact`

| n delete manifests | sequential extra walk (old caller) | close after F-22 (one pass) |
|---|---|---|
| 8 | 41.300 ms | 36.666 ms |
| 192 | 718.106 ms | 727.287 ms |

RePark V3-12 (2026-09-02) on a MERGE-delete statement: fork close 38 ms / 643 ms vs RePark's own walk 27 ms / 626 ms at 8 / 192. After F-22 the statement does not pay the walk.

## 6. Mutations (one knob at a time)

| id | knob | result |
|---|---|---|
| M1 | skip merging loaded legacy positions into `positions_to_write` | 4 red of 6 (`test_f21_base_cell_delete_merges_parquet_into_dv`, `test_f21_partition_scoped_merge_keeps_parquet`, `test_f21_two_file_scoped_parquet_deletes_merge_into_one_dv`, `test_f21_update_merges_parquet_into_dv`) |
| M2 | never push file-scoped parquet onto `close.removed` | 3 red of 6 (base, two-file-scoped, UPDATE; partition-scoped stays green) |
| M3 | `seq_applies` always true | 1 red of 6 (`test_f21_sequence_number_not_apply`) |
| M4 | do not collect pending parquet deletes | 4 red of 6 (same set as M1) |

## 7. Interop

| command | fixtures | sabotage |
|---|---|---|
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 2 | FAIL-closed (F-21) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 4 + final.metadata.json | FAIL-closed (F-18) |

## 8. Gate exits

Filled from PROGRESS.md after each gate is run. A gate not run is NOT RUN.

| gate | exit |
|---|---|
| `make check` | 0 |
| `cargo test -p iceberg --locked` | 0 (`3587 passed; 0 failed; 3 ignored` lib; doctests 90 passed / 10 ignored) |
| `cargo test -p iceberg-datafusion --locked` | 0 (`211 passed` lib; `f21_legacy_delete_merge` 6 passed; `shared_puffin_dv` 20 passed / 3 ignored) |
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 0 (`PASSED`; 2 fixtures; sabotages red) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 0 (`PASSED`; sabotage red) |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 (`OK: GAP_MATRIX anchors sound`) |
| `python3 scripts/check_rust_file_size.py` | 0 (`431 files clean`; `delete_vector_container.rs` 571) |

Docker legs of `make test` excused.

## 9. RePark

| note |
|---|
| Delete `legacy_deletes.rs`'s own delete-manifest walk. Consume `DvContainerClose::legacy_deletes` + `load_legacy_positions`. |
| Do not re-walk data manifests for touched files' data sequence numbers; consume `DvContainerClose::data_sequence_numbers`. Pass statement-only positions into close (close merges). |

## 10. Section 9 delivery template

```text
Charter clauses: C-001 through C-005
Matrix rows: R114 (F-22 one-pass legacy scan)
Java methods or bytecode read: BaseDVFileWriter.loadPreviousDeletes (L114-129), ContentFileUtil.referencedDataFile / isFileScoped
Files changed: crates/iceberg/src/delete_vector_container.rs; crates/iceberg/src/delete_vector_container/legacy.rs; crates/iceberg/src/delete_vector_container/tests.rs; crates/iceberg/src/arrow/delete_file_loader.rs; crates/integrations/datafusion/src/physical_plan/delete_legacy_merge.rs; docs/parity/GAP_MATRIX.md R114; maps; task/todo.md; task/f22-legacy-scan-one-pass-ledger.md
Behavior before: close collected only Puffin DVs; DataFusion (and RePark) walked every delete manifest a second time for live parquet position deletes, and walked data manifests for sequence numbers
Behavior after: one delete-manifest pass returns legacy_deletes and data_sequence_numbers; load_legacy_positions projects pos (and file_path when not file-scoped); DataFusion write_deletion_vectors does not walk; file-scoped parquet is merged and removed in the same close
Negative cases: file-scoped parquet that does not name the touched file is not collected; delete_seq < data_seq is not merged and not removed; projected read of a row-column delete is smaller than the file
Test command and population: cargo test -p iceberg --locked --lib delete_vector_container; cargo test -p iceberg-datafusion --locked --test f21_legacy_delete_merge (6 passed); shared_puffin_dv 20 passed / 3 ignored
Mutations, one at a time: see section 6
Java interop command and fixture count: run-interop-f21-legacy-delete-merge.sh (2 fixtures); run-interop-f18-dv-sibling-close.sh
CI-only evidence gap: Docker legs of make test excused
Breaking public API change: close_touched_dv_containers_with_partitions gains Option<&ManifestList>; DvContainerClose gains legacy_deletes and data_sequence_numbers
Critic attestation: Actor only (this unit)
Open findings and dispositions: R114 residues unchanged (equality-delete sort-order; Spark-job-written shared Puffin)
```
