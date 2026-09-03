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

Model: grok-4.6. Base: F-21 tip `63d3d09f4` (round 2 parent). Branch `repark/f22-legacy-scan-one-pass`.
Matrix row: R114.

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | One delete-manifest pass returns live non-Puffin position deletes that name a touched data file, plus touched data sequence numbers; optional pre-loaded `ManifestList` | PROVEN | `collect_delete_index` on `manifest_stream(Deletes)`; `DvContainerClose::legacy_deletes` / `data_sequence_numbers`; `preloaded_manifest_list_skips_the_list_reread` |
| C-002 | `load_legacy_positions_by_path` is one projected parquet read per delete file; `load_legacy_positions` looks up a path on that index | PROVEN | `load_legacy_positions_projects_past_the_row_column`; P1-a 64-path 14.6 ms |
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
| `crates/iceberg/src/delete_vector_container.rs` | one delete-manifest pass; one data-manifest pass seeding partitions and sequences; overlay instead of cloning `new_positions`; Arc merge; sort `removed` after extending; `seq: Option<i64>` with F-21 `_ => true` |
| `crates/iceberg/src/delete_vector_container/legacy.rs` | `load_legacy_positions_by_path`; pos-only `values()` path; HashSet touched; `finalize_legacy` groups by `(spec_id, partition)`; file-scoped requires `touched.len()==1` |
| `crates/iceberg/src/arrow/delete_file_loader.rs` | name-based projection fallback for pos-only |
| `crates/integrations/datafusion/src/physical_plan/delete_legacy_merge.rs` | no manifest walk; close merges |

Concurrency bound: `DV_IO_CONCURRENCY = 8` (now `pub`).

## 4. Pins

| id | knob | result |
|---|---|---|
| P-del | counting Storage, 8 delete manifests, no applicable legacy | 1 red of 1 if count ≠ N (`delete_manifests_are_read_once_without_legacy_deletes`) |
| P-leg | same with 8 file-scoped parquet deletes | 1 red of 1 if count ≠ N (`delete_manifests_are_read_once_with_legacy_deletes`) |
| P-proj | pos-delete with reserved `row` column | 1 red of 1 if bytes ≥ file size |
| P-seq | close without partition map AND with a complete map and no legacy | `data_sequence_numbers` contains the touched path (C2 / P1-f) |
| P-list | pre-loaded `ManifestList` | `snapshot_list_reads == 0` with `Some(&list)`, `== 1` with `None` (C1 / P3-b) |
| P-data | partial `known_partitions` + legacy, two data manifests | `data_manifest_reads == 2` (C4 / P1-e) |
| P-scope | `finalize_legacy` / `partition_matches` / `file_path_bounds_admit` | other partition empty; other spec_id empty; bounds exclude empty; bounds include collected (C3) |

## 5. Measurement (debug, this clone, no CI pin)

Commands:
`cargo test -p iceberg --locked --offline --lib delete_vector_container::tests::measure_close_at_8_and_192_delete_manifests -- --ignored --nocapture --exact`
`cargo test -p iceberg --locked --offline --lib delete_vector_container::measure -- --ignored --nocapture`

Round 2 close wall (debug, this clone, `with_legacy=false` — the extra walk is sequential delete-manifest IO only, not legacy parquet loads):

| n delete manifests | extra walk | close (one pass) | honest total before (walk+close) |
|---|---|---|---|
| 8 | 35.758 ms | 44.796 ms | 80.554 ms |
| 48 | 179.135 ms | 192.450 ms | 371.585 ms |
| 192 | 693.921 ms | 723.409 ms | 1417.330 ms → 723.409 ms |

Round 1 table at 192 quoted walk 718 ms vs close 727 ms without adding them; C10: the statement used to pay both (≈1417 ms), now only close (723 ms). Fixture has no live parquet to load.

| cell | after (this clone) |
|---|---|
| P1-a 64-path `load_legacy_positions_by_path` | 14.551 ms (review before: 63.3 s / 91 MB for per-path re-read) |
| P1-c 512k file-scoped | 35.771 ms |
| P1-d touched=128 partition-scoped 512k rows | 661.133 ms |

P2-d: pos-only projection vs a two-column delete is ~0.09 %; the C-002 pin's fixture carries an ~800 KB `row` column, which is the win. Every parquet open still prefetches a ~512 KB tail.

## 6. Mutations (one knob at a time)

| id | knob | result |
|---|---|---|
| M1 | skip merging loaded legacy positions into `positions_to_write` | 4 red of 6 (`test_f21_base_cell_delete_merges_parquet_into_dv`, `test_f21_partition_scoped_merge_keeps_parquet`, `test_f21_two_file_scoped_parquet_deletes_merge_into_one_dv`, `test_f21_update_merges_parquet_into_dv`) |
| M2 | never push file-scoped parquet onto `close.removed` | 3 red of 6 (base, two-file-scoped, UPDATE; partition-scoped stays green) |
| M3 | `seq_applies` always true | 1 red of 6 (`test_f21_sequence_number_not_apply`) |
| M4 | do not collect pending parquet deletes | 4 red of 6 (same set as M1) |
| C1 | ignore `Option<&ManifestList>` and always load the list | 1 red of 13 (`preloaded_manifest_list_skips_the_list_reread`: list_reads 1 vs 0) |
| C3-bounds | `file_path_bounds_admit` always true | 1 red of 13 (`bounds_range_excluding_the_path_is_not_admitted`) |
| C3-part | `partition_matches` always true | 2 red of 13 (other partition; other spec_id) |

## 6b. Round 2 (P / C)

| id | closed by |
|---|---|
| P1-a / C9 | `load_legacy_positions_by_path`; file-scoped requires `touched.len()==1` |
| P1-b | reserve from current batch, not `record_count` |
| P1-c | `null_count()==0` then `pos_col.values()` |
| P1-d | `HashSet<&str>` of touched; `get_mut` before insert |
| P1-e / C4 | one `collect_live_data_files` of all touched paths |
| P1-f / C2 | `data_sequence_numbers` filled with a complete partition map and no legacy |
| P2-a / C6 | overlay HashMap only when legacy deletes exist |
| P2-b | `Arc<LegacyPositionDelete>` in the merge stream |
| P2-c | group touched by `(spec_id, partition)` |
| P2-d | ledger: pos-only win is the `row` column fixture |
| P3-a | `referenced_location_ref` allocates only for retained entries |
| P3-b / C1 | `snapshot_list_reads` 0 vs 1 |
| P3-c | RePark consumes `Option<&ManifestList>` |
| C5 | `removed.extend` then sort |
| C7 | `data_sequence_number: Option<i64>` and F-21 `_ => true` |
| C8 | ledger base `63d3d09f4` |
| C10 | honest walk+close total in §5 |

## 7. Interop

| command | fixtures | sabotage |
|---|---|---|
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 2 | FAIL-closed (F-21) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 4 + final.metadata.json | FAIL-closed (F-18) |

## 8. Gate exits

Filled from PROGRESS.md after each gate is run. A gate not run is NOT RUN.

| gate | exit |
|---|---|
| `make check` | 0 (round 2) |
| `cargo test -p iceberg --locked` | 0 (`3593 passed; 0 failed; 6 ignored` lib; doctests 90 passed / 10 ignored) |
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
| Delete `legacy_deletes.rs`'s own delete-manifest walk. Consume `DvContainerClose::legacy_deletes` + `load_legacy_positions_by_path` (one read per delete file). |
| Do not re-walk data manifests for sequence numbers; `data_sequence_numbers` is filled for every touched path even with a complete `known_partitions` and no legacy. Pass statement-only positions into close (close merges). |
| `Option<&ManifestList>` has no in-tree DataFusion consumer (`delete_legacy_merge.rs` passes `None`); RePark is the consumer (P3-c). |

## 10. Section 9 delivery template

```text
Charter clauses: C-001 through C-005; round 2 P1-a..f P2-a..d P3-a..c C1-C10
Matrix rows: R114 (F-22 one-pass legacy scan)
Base: 63d3d09f4
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
