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

# F-19 + F-20 — files-exist narrowing, sibling-rewrite removal, FanoutWriter drain order

Model: grok-4.6. Base: fork `main` `ff4764d3e`. Branch `repark/f19-f20-fanout-order`.
Matrix rows: R114, R115, R135. R136 re-audited unchanged.

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | F-19a over-strict files-exist and F-20 hash-order drain are reproducible | PROVEN | table in section 2 |
| C-002 | production: drop sibling refs from files-exist; delete sibling rewrite; sort drain | PROVEN | section 3 |
| C-003 | pins, mutations, F-18 Java interop, 1e6 timing | PROVEN | sections 4–6 |
| C-004 | docs in lockstep | PROVEN | rows R114 / R115 / R135 / R136; `task/todo.md`; map.md files; this ledger |

## 2. Red-first

| pin | base (`ff4764d3e` drain / retained_references) | after |
|---|---|---|
| `extra::delete_allows_concurrent_replace_of_untouched_sibling` | RED: `Cannot commit, missing data files: …/category=books/…parquet` | GREEN: DELETE commits; live ids `[3, 4, 5, 6]` |
| `test_fanout_close_drains_identity_int_partitions_ascending` | RED: `[Some(1), Some(3), Some(0), Some(2), Some(4)]` | GREEN: `[Some(0)…Some(4)]` |
| `test_fanout_close_drains_null_partition_first` | RED: `[Some(0), None]` | GREEN: `[None, Some(0)]` |
| `fanout_insert_order` run 0 shuffle `[3,1,4,0,2]` | RED: manifest order `[4,0,3,1,2]` | GREEN: `[0,1,2,3,4]` ten times |

## 3. Production change

| file | change |
|---|---|
| `crates/iceberg/src/delete_vector_container.rs` | deleted `retained_references`, `DvDropPlan`, `rewrite_siblings_for_dropped_references`, `collect_live_dvs`; `referenced_data_files()` is replacement blobs only; `collect_touched_dvs` collects touched DVs only |
| `crates/iceberg/src/maintenance/rewrite_data_files_dv.rs` | `plan_dv_removal` drops DVs and file-scoped parquet whose referenced path was rewritten; no sibling copy |
| `crates/iceberg/src/maintenance/rewrite_data_files.rs` | no `add_delete_file_with_sequence_number` of rewritten siblings |
| `crates/iceberg/src/writer/partitioning/fanout_writer.rs` | `close` sorts keys with `ascending_partition_order` (nulls first, spec-field order, primitive literal ascending) |
| `crates/integrations/datafusion/src/physical_plan/delete.rs` | files-exist comment/module row no longer claims sibling coverage |
| `crates/integrations/datafusion/tests/shared_puffin_dv/{extra,live}.rs` | concurrent sibling Replace/Delete now COMMIT; T20/T22 inject failure on touched file A |
| `crates/iceberg/src/maintenance/rewrite_data_files_ratio_tests.rs` | sibling stays at the original Puffin path |
| `crates/integrations/datafusion/tests/fanout_insert_order.rs` | NEW. ten shuffled identity-int INSERT statements |

Deleted: `DvContainerClose::retained_references`, `rewrite_siblings_for_dropped_references`, `DvDropPlan`, `LiveDv`, `collect_live_dvs`, `collect_dv_index` (renamed `collect_touched_dvs`), `StampedDeleteFile` (collapsed to `Vec<DataFile>`).

Id 5 is live after the F-19a pin: the harness `rewrite_data_file` is a byte-copy `RewriteFiles` that does not apply DV_B, and the fork's conservative dangling-delete carry-forward leaves DV_B pointing at the old books path. Java `isDanglingDV` would drop DV_B and land the same rows.

## 4. Measurement (debug, same clone, alternated knob)

| cell | hash drain | sorted keys |
|---|---|---|
| 1e6 rows / 8 partitions close_ms | 109 / 110 / 115 | 81 / 101 |

No measurable change. The sort is 8 keys; parquet flush dominates. A warm/cold pair is not a before/after.

Command: `cargo test -p iceberg --locked --lib writer::partitioning::fanout_writer::tests::measure_fanout_one_million_rows_eight_partitions -- --ignored --nocapture --exact`.

## 4b. Critic remediations (2026-09-02)

| item | change |
|---|---|
| R115 / §4 timing | alternated knob: hash 109/110/115 ms vs sorted 81/101 ms; no measurable change |
| F-19a pin | `assert_eq!(ids, vec![3, 4, 5, 6])`; id 5 live because harness byte-copy RewriteFiles + dangling-delete carry-forward (Java `isDanglingDV` drops DV_B, same rows) |
| R114 | F-18 `retained_references` sentence past-tense / superseded below |
| `StampedDeleteFile` | collapsed to `Vec<DataFile>`; `apply_dv_container_close` always `add_deletes` |
| `FanoutWriter::close` comments | restored the two upstream `//` lines byte-exact |
| §6 / §10 interop | 4 fixtures; sabotage leg red |
| files-exist set | run-length pass on sorted `(path, pos)` pairs. 1e6 rows / 8 paths (release, alternated): clone-all 42/42/42 ms vs run-length 11/11/12 ms |
| `plan_dv_removal` / `file_scoped_delete_paths` | one cached walk per `RewriteDataFiles::execute`; DELETE manifests via `buffer_unordered(8)`; path-only variant clones no `DataFile` |
| `FanoutWriter::close` | sorts `Struct` keys, then `remove` by key |
| `collect_touched_dvs` | `referenced_data_file_ref() -> Option<&str>`; owned String only for touched `BlobWrite` |

## 5. Mutations (one knob at a time)

| id | knob | result |
|---|---|---|
| M1 | `FanoutWriter::close` drains `HashMap` without `sort_by` | 3 red out of 3 (two unit pins + `fanout_insert_order`) |
| M2 | `referenced_data_files()` unions sibling referenced paths again | 4 red out of 20 (`delete_allows_concurrent_replace_of_untouched_sibling`, `update_allows_concurrent_replace_of_untouched_sibling`, `update_allows_concurrent_delete_of_untouched_sibling`, `delete_allows_concurrent_delete_of_untouched_sibling`) |
| M3 | `plan_dv_removal` also drops every live DV that shares a Puffin path with a dropped blob | 1 red out of 1 (`test_rewriting_one_file_keeps_sibling_dv_in_same_puffin`: `removed_delete_files_count` 2 vs 1) |

Restored after each.

## 6. Interop

```bash
dev/java-interop/run-interop-f18-dv-sibling-close.sh
```

F-18 sibling-close runner: 4 fixtures; sabotage leg red (`kept=0 moved=2`). PASSED.

## 7. Gate exits

| gate | exit |
|---|---|
| `make check` | 0 |
| `cargo test -p iceberg --locked` | 0 |
| `cargo test -p iceberg-datafusion --locked` | 0 |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 |
| `scripts/check_rust_file_size.py` | 0 (ceilings ratcheted DOWN: `rewrite_data_files.rs` 2663 -> 2659, `physical_plan/delete.rs` 2075 -> 2071) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 0 |

Docker legs of `make test` excused: Docker is unavailable on this box.

## 8. RePark note

Repin unit RP-8: `F-v3-10-partition-file-order` closes. Registry row `V3-FILEORDER-1` gains "fork INSERT path now ascending too".

## 9. Follow-ups

| item | owner |
|---|---|
| RePark RP-8 / `V3-FILEORDER-1` | RePark |
| V3 DataFusion still walks DELETE manifests twice per statement | fork |

## 10. Section 9 delivery template

```text
Charter clauses: C-001 through C-004
Matrix rows: row R114, row R115, row R135 (row R136 re-audit unchanged)
Java methods or bytecode read: RowDelta.validateDataFilesExist (referenced files only); ManifestFilterManager.isDanglingDV; DeleteFileSet triple (F-18, unchanged)
Files changed: crates/iceberg/src/delete_vector_container.rs; crates/iceberg/src/maintenance/rewrite_data_files_dv.rs; crates/iceberg/src/maintenance/rewrite_data_files.rs; crates/iceberg/src/writer/partitioning/fanout_writer.rs; crates/integrations/datafusion/src/physical_plan/delete.rs; crates/integrations/datafusion/tests/shared_puffin_dv/{extra,live}.rs; crates/integrations/datafusion/tests/fanout_insert_order.rs; crates/iceberg/src/maintenance/rewrite_data_files_ratio_tests.rs; docs/parity/GAP_MATRIX.md rows R114 R115 R135 R136; docs/ENGINE_CONTRACT.md; map.md files; task/todo.md; this ledger
Behavior before: validate_data_files_exist included untouched sibling references (over-strict vs Java); RewriteDataFiles copied live siblings out of a Puffin whose other blob was dropped; FanoutWriter::close drained HashMap in hash order (two INSERT orders across 10 runs)
Behavior after: files-exist covers replacement blobs only; sibling blob stays in the original Puffin; FanoutWriter::close drains ascending (nulls first, spec-field order, primitive literal ascending)
Negative cases: F-19a pin was red on missing-data-files; F-20 pins were red on hash order; mutations recorded in section 5
Test command and population: cargo test -p iceberg --locked; cargo test -p iceberg-datafusion --locked; shared_puffin_dv is 20 passed / 3 ignored
Mutations, one at a time: see section 5
Java interop command and fixture count: dev/java-interop/run-interop-f18-dv-sibling-close.sh — 4 fixtures; sabotage leg red (`kept=0 moved=2`)
CI-only evidence gap: Docker make test legs excused (Docker unavailable)
Breaking public API change: DvContainerClose loses retained_references; rewrite_siblings_for_dropped_references and DvDropPlan are deleted; StampedDeleteFile collapsed to Vec<DataFile> (the Some(sequence) stamp arm was unreachable after F-19b)
Open findings and dispositions: RePark RP-8 / V3-FILEORDER-1
```
