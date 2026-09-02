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

# F-18 — deletion-vector container close: Spark-equal layout, lazy data-file walk

Model: claude-opus-5 (medium). Base: fork `main` `d4ef080ac`. Branch `repark/f18-dv-container-close`.
Plan of record: `task/iceberg-v3-production-work-plan-2026-09-01.md`. Matrix row: R114.

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | The fork's shared-Puffin close diverges from Spark's layout, reproducibly | PROVEN | `container.rs::touched_blob_moves_and_the_sibling_entry_stays_put` red on `d4ef080ac`: sibling moved from `dv-...-afd4-...puffin@4+42` to `dv-...-b01f-...puffin@4+42` |
| C-002 | Only the touched blob is rewritten; the sibling entry is untouched | PROVEN | same test green; the interop leg reads the same layout through Java |
| C-003 | The data-file walk is lazy and the manifest list is loaded once | PROVEN | `close_touched_dv_containers_at` computes `fresh` first and calls `collect_live_data_files` only when it is non-empty, filtered to those paths |
| C-004 | The rewrite amplification is closed and measured | PROVEN | table in section 5 |
| C-005 | Java writes the seed, Rust the second delete, Java reads the result | PROVEN | `run-interop-f18-dv-sibling-close.sh` PASSED, sabotage red |
| C-006 | Docs, `map.md` and this ledger are in lockstep | PROVEN | row R114, `task/todo.md`, four `map.md` files |

## 2. Java 1.10.0 bytecode (`javap -c -p`, jars under `~/.m2/repository/org/apache/iceberg/`)

| class / method | decisive instructions | consequence |
|---|---|---|
| `iceberg-api` `util/DeleteFileSet$DeleteFileWrapper.equals` | offsets 25/49/73 compare `location()`, `contentOffset()`, `contentSizeInBytes()` | DV identity is a TRIPLE, not a path |
| `iceberg-api` `util/DeleteFileSet$DeleteFileWrapper.hashCode` | `Objects.hash(location, contentOffset, contentSizeInBytes)` | same |
| `iceberg-core` `MergingSnapshotProducer.delete(DeleteFile)` | offset 5 `invokevirtual ManifestFilterManager.delete:(ContentFile)` | `removeDeletes` routes to `delete(F)`, not `delete(CharSequence)` |
| `iceberg-core` `ManifestFilterManager.delete(F)` | offset 45 `deleteFiles.add(file)` (the `DeleteFileSet`); `deletePaths` untouched | removal is per-blob |
| `iceberg-core` `ManifestFilterManager.lambda$filterManifestWithDeletedFiles$3` | offsets 37/54 `deletePaths.contains(location)` then `deleteFiles.contains(file)` | the per-entry match uses the triple for delete manifests |

The fork keyed BOTH `resolve_delete_file_paths` and `process_deletes` on the path alone. That is why
F-17 had to carry live siblings: removing the touched blob tombstoned the sibling entry too.

## 3. Production change

| file | change |
|---|---|
| `crates/iceberg/src/transaction/snapshot/removal_targets.rs` | NEW. `RemovalTargets` / `RemovalHits`: DATA entries match by path, DELETE entries by the `DeleteFileSet` triple |
| `crates/iceberg/src/transaction/snapshot.rs` | `resolve_delete_file_paths` -> `resolve_removed_delete_files` (triple-keyed, no path expansion); `process_deletes` takes the removed DATA and DELETE sets separately and matches each manifest kind on its own key |
| `crates/iceberg/src/delete_vector_container.rs` | `close_touched_dv_containers_at` rewrites only touched blobs into ONE new container per statement; `DvContainerClose::retained_references` carries the untouched siblings of a rewritten container into `referenced_data_files()`; one manifest-list load shared by both collectors; `buffer_unordered(8)` manifest and DV reads; `collect_live_data_files` runs only for paths with no previous DV and keeps only those entries |
| `crates/iceberg/src/writer/base_writer/deletion_vector_writer.rs` | `delete` resolves with `get_mut` first and allocates the path only on insert |
| `crates/integrations/datafusion/src/physical_plan/delete.rs` | position-delete arm uses `..DvContainerClose::default()` |

Concurrency bound: `DV_IO_CONCURRENCY = 8`.

## 4. Layout now written

| statement | before (base) | after |
|---|---|---|
| `DELETE` touching two files with no DV | ONE Puffin, two blobs | unchanged |
| later `DELETE` touching ONE of them | both blobs rewritten into ONE new container, sibling path moves, `removed-dvs 2` | touched blob into a new container, sibling entry unchanged, TWO containers, `removed-delete-files 1 / removed-dvs 1 / added-delete-files 1` |
| `DELETE` touching two files in two different containers | two new containers | ONE new container (Spark writes one Puffin per write) |

Java-measured instance (interop step 5): seed `00001-1-...puffin` holds A@4+42 and B@46+44; after the
Rust DELETE, A is at `dv-00000-...puffin@4+44` and B is still at `00001-1-...puffin@46+44`.

## 5. Measurement (same clone, `git stash` of the source change only; debug unless stated)

| cell | before `d4ef080ac` | after |
|---|---|---|
| later single-row DELETE, 16 blobs: containers / blobs in the new container / bytes | 1 / 16 / 5,006 | 2 / 1 / 388 |
| later single-row DELETE, 64 blobs | 1 / 64 / 19,830 | 2 / 1 / 388 |
| six single-row `DELETE` statements, 8 live data files | 850 ms | 731 ms |
| six single-row `DELETE` statements, 64 live data files | 922 ms | 841 ms |
| six single-row `DELETE` statements, 192 live data files | 1,152 ms | 1,138 ms |
| the same three cells, `--release` | 250 / 247 / 275 ms | 238 / 233 / 294 ms |

Command: `cargo test -p iceberg-datafusion --locked --offline --test shared_puffin_dv -- --ignored --nocapture --test-threads=1`.
No wall-clock pin is in CI. Recorded honestly: the byte amplification is closed (51x at 64 blobs);
the wall-clock cells move only slightly in this harness, whose 192 data files live in ONE data
manifest, so the walk the reviewer measured was already cheap here. The reviewer's larger
wall-clock numbers are not reproduced by this fixture and are not claimed.

## 6. Mutations (one knob at a time, `cargo test -p iceberg-datafusion --locked --offline --test shared_puffin_dv`)

| id | knob | result |
|---|---|---|
| M1 | `RemovalTargets::matches` DELETE arm compares the path only | 7 red out of 20 (`container::touched_blob_moves...`, `measure::a_later_delete...`, both resurrection pins, `untouched_sibling_keeps_original_data_sequence`, `equality_delete_survives...`, `delete_allows_concurrent_delete_of_unrelated_file`) |
| M2 | `close.retained_references` never populated | 6 red out of 20 (the four concurrent-Replace/Delete sibling pins plus both post-output-failure pins) |
| M3 | `resolve_removed_delete_files` matches on the path only | 7 red out of 20 (same set as M1) |
| M4 | interop: both seed offsets shifted by one in `before_dvs.json` | oracle red, `kept=0 moved=2` |

Restored after each; the suite is 20 passed / 2 ignored green.

## 7. Interop command and fixture count

```bash
dev/java-interop/run-interop-f18-dv-sibling-close.sh
```

Six steps. Java `generate-interop-dv-table` writes the V3 seed (two partitioned data files, one
Puffin, two `BaseDVFileWriter` blobs); Rust `interop_f18_dv_sibling_close.rs` runs the second
DELETE; the runner asserts **4** fixture files (`before_dvs.json`, `after_dvs.json`,
`summary.json`, `expected_rows.json`) plus `rust_table/metadata/final.metadata.json`;
`verify-interop-f18-sibling-close` (`F18SiblingCloseOracle`) reads the live rows with
`IcebergGenerics`, walks the delete manifests, and asserts two containers, one moved blob, one
byte-identical sibling entry, Java/Rust agreement on every entry, and the summary counts; step 6
sabotages a COPY and requires the oracle to go red. `SUITE_FLOOR_DEFAULT` 63 -> 64.

## 8. Gate exits

| gate | exit |
|---|---|
| `make check` | 0 |
| `cargo test -p iceberg --locked` | 0 |
| `cargo test -p iceberg-datafusion --locked` | 0 |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 |
| `scripts/check_rust_file_size.py` | 0 (ceilings ratcheted DOWN: `transaction/snapshot.rs` 3502 -> 3491, `deletion_vector_writer.rs` 1426 -> 1425) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | 0 |

Docker legs of `make test` excused: Docker is unavailable on this box.

## 9. Follow-ups

| item | owner |
|---|---|
| RULING: retire the F-17 `DELETE`-side broadening of Java's skip-delete behavior. Untouched siblings are no longer rewritten, so `validate_data_files_exist` over them is now pure over-strictness. This unit KEPT it (`retained_references`) so every F-17 pin stays green; row R114 still records the divergence | owner |
| `rewrite_siblings_for_dropped_references` (maintenance) still copies siblings forward. With triple-keyed removal that copy is no longer required for correctness, only wasteful | fork |
| RePark repin: `crates/repark-spark/src/tests/v3e4.rs` shared-Puffin cell now sees TWO containers and an unchanged sibling entry; registry row `V3-DV-1`; repin unit RP-7 | RePark |

## 10. Section 9 delivery template (lift into the PR body)

```text
Charter clauses: C-001 through C-006
Matrix rows: row R114
Java methods or bytecode read: DeleteFileSet$DeleteFileWrapper.equals/hashCode (location, contentOffset, contentSizeInBytes); MergingSnapshotProducer.delete(DeleteFile) -> ManifestFilterManager.delete(F) -> deleteFiles.add; ManifestFilterManager.lambda$filterManifestWithDeletedFiles$3
Files changed: crates/iceberg/src/transaction/snapshot.rs; crates/iceberg/src/transaction/snapshot/removal_targets.rs; crates/iceberg/src/delete_vector_container.rs; crates/iceberg/src/delete_vector_lookup.rs; crates/iceberg/src/writer/base_writer/deletion_vector_writer.rs; crates/integrations/datafusion/src/physical_plan/delete.rs; crates/integrations/datafusion/tests/shared_puffin_dv/{container,measure,extra,main}.rs; crates/integrations/datafusion/tests/interop_f18_dv_sibling_close.rs; dev/java-interop/src/main/java/org/apache/iceberg/InteropOracle.java; dev/java-interop/run-interop-f18-dv-sibling-close.sh; scripts/run_interop_suites.sh SUITE_FLOOR_DEFAULT 64; scripts/check_rust_file_size.py; docs/parity/GAP_MATRIX.md row R114; four map.md; task/todo.md; task/f18-dv-container-close-ledger.md
Behavior before: a DELETE touching one blob of a shared Puffin rewrote every blob in it, moved the sibling entry, and reported removed-dvs 2. A 64-blob container cost 19,830 rewritten bytes per later statement.
Behavior after: only the touched blob is rewritten, into ONE new container per statement; the sibling entry keeps its path, content_offset, content_size_in_bytes and data sequence; summary removed-delete-files 1 / removed-dvs 1 / added-delete-files 1; 388 bytes per later statement. Removal of a DELETE file is keyed by the Java DeleteFileSet triple.
Negative cases: a sabotaged seed offset turns the Java oracle red (kept=0 moved=2); three source mutations each turn the pins red
Test command and population: cargo test -p iceberg --locked; cargo test -p iceberg-datafusion --locked; shared_puffin_dv is 20 passed / 2 ignored
Mutations, one at a time: M1 7 red of 20; M2 6 red of 20; M3 7 red of 20; M4 interop oracle red
Java interop command and fixture count: dev/java-interop/run-interop-f18-dv-sibling-close.sh — 4 fixtures + final.metadata.json; sabotage FAIL-closed
CI-only evidence gap: Docker make test legs excused (Docker unavailable)
Breaking public API change: DvContainerClose gains retained_references and removed now carries only the touched blobs
Critic attestation: pending
Open findings and dispositions: the F-17 DELETE-side skip-delete divergence is KEPT and re-raised as a RULING; maintenance sibling copy is now redundant; RePark shared-Puffin cell needs a repin
```
