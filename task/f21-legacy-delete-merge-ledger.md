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

Model: muse-spark-1.2-contributor

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | V2 MoR + file-scoped parquet delete (id 2) upgraded to V3, DELETE id 3 refuses before IO (red) | PROVEN | `f21_legacy_delete_merge.rs::test_f21_base_cell_delete_merges_parquet_into_dv` red at `ff4764d3e` (guard `BaseDVFileWriter.loadPreviousDeletes is deferred`), green after |
| C-002 | V3 DELETE/UPDATE merges file-scoped parquet deletes via DV and removes the superseded file; partition-scoped merges per-file positions but keeps the parquet live (Spark-equal) | PROVEN | `write_deletion_vectors` collects `referenced_data_file_location` file-scoped vs partition-scoped, loads positions by reserved ids 2147483546/2147483545 filtered to the data file, only `delete_seq >= data_seq`, union once, `DVFileWriter::with_previous_deletes` + `RowDelta::remove_deletes_many` for file-scoped, `close_touched_dv_containers` for siblings; `validate_fresh_dvs_only` now only blocks file-scoped |
| C-003 | Pins: base cell rc 2, UPDATE, untouched-file control, partition-scoped merge-and-keep, sequence-number control; Java interop proves byte-level DV and rows | PROVEN | `f21_legacy_delete_merge.rs` 5 tests (base, UPDATE, untouched, partition-scoped rc 2 + parquet kept, seq not apply); `interop_f21_legacy_delete_merge` Java seed V2 parquet delete -> Rust V3 DV rc 2, parquet 0, rows (1,4), summary `added-dvs 1/added-position-deletes 2`, sabotage red; `run-interop-f21-legacy-delete-merge.sh` 1 fixture + final.metadata.json |

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
| `crates/iceberg/src/arrow/delete_file_loader.rs` | `BasicDeleteFileLoader` now `pub`, `load_position_delete_pairs` (reserved ids, production read path) |
| `crates/iceberg/src/transaction/row_delta_fresh_dv.rs` | door now only blocks file-scoped (`referenced_data_file_location.is_some()`), message no longer "deferred" |
| `crates/integrations/datafusion/src/physical_plan/delete_legacy_merge.rs` | NEW: legacy collect, filtered load, union once, file-scoped removal via `DvContainerClose::removed` |
| `crates/integrations/datafusion/src/physical_plan/delete.rs` | split (1704), delegates to `delete_legacy_merge::write_deletion_vectors`, test imports trimmed |
| `crates/iceberg/src/writer/base_writer/deletion_vector_writer/file_scope.rs` | `is_file_scoped` now `pub(super)` -> `pub(crate)` via `referenced_data_file_location` |
| `scripts/check_rust_file_size.py` | ceiling `delete.rs` 2075 -> 1704 (split) |

## 4. Pins

| id | knob | result |
|---|---|---|
| M1 | `write_deletion_vectors` not merging legacy positions (remove the legacy load) | 1 red of 5 (`test_f21_base_cell_delete_merges_parquet_into_dv`) |
| M2 | `file_scoped_to_remove` not populated (keep parquet) | 1 red of 5 (base cell: parquet 1 vs 0, DV rc 1 vs 2) |
| M3 | `referenced_data_file_location` check removed (partition-scoped also removed) | 1 red of 5 (`test_f21_partition_scoped_merge_keeps_parquet`: parquet 0 vs 1) |
| M4 | `delete_seq >= data_seq` check removed (old delete merges for new file) | 1 red of 5 (`test_f21_sequence_number_not_apply`: DV rc 2 vs 1) |
| M5 | Java sabotage (corrupt expected_rows) | oracle red |

## 5. Interop

| command | fixtures | sabotage |
|---|---|---|
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 1 (`expected_rows.json`) + `rust_table/metadata/final.metadata.json` | corrupt `expected_rows.json` id 1→999, oracle red |

## 6. Gate exits

| gate | exit |
|---|---|
| `make check` | 0 |
| `cargo test -p iceberg --locked` | 0 |
| `cargo test -p iceberg-datafusion --locked` | 0 |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 |
| `scripts/check_rust_file_size.py` | 0 (delete.rs 1704, delete_legacy_merge.rs 387) |
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | 0 |
| `cargo test -p iceberg-datafusion --test f21_legacy_delete_merge` | 5 passed |
| `cargo test -p iceberg-datafusion --test interop_f21_legacy_delete_merge` | 1 passed |

## 7. RePark

| note |
|---|
| repin unit closes `V3-UPGRADE-DV-PLAIN-1`; keep `V3-UPGRADE-DV-PART-1` |

## 8. Section 9 delivery template

```text
Charter clauses: C-001 through C-004
Matrix rows: R114 (F-21 legacy delete merge), R166 (next-row-id 4, rows (1,0,1),(4,3,1) provenance)
Java methods or bytecode read: BaseDVFileWriter.loadPreviousDeletes (L114-129), ContentFileUtil.isFileScoped / referencedDataFile, DeleteFileIndex.findDV / findPosPartitionDeletes, RowDelta.removeDeletes + validate_fresh_dvs_only
Files changed: crates/iceberg/src/arrow/delete_file_loader.rs; crates/iceberg/src/transaction/row_delta_fresh_dv.rs; crates/integrations/datafusion/src/physical_plan/delete.rs; crates/integrations/datafusion/src/physical_plan/delete_legacy_merge.rs; crates/integrations/datafusion/tests/f21_legacy_delete_merge.rs; crates/integrations/datafusion/tests/interop_f21_legacy_delete_merge.rs; dev/java-interop/src/main/java/org/apache/iceberg/InteropOracle.java; dev/java-interop/run-interop-f21-legacy-delete-merge.sh; scripts/run_interop_suites.sh SUITE_FLOOR_DEFAULT 65; scripts/check_rust_file_size.py; docs/parity/GAP_MATRIX.md R114; task/f21-legacy-delete-merge-ledger.md
Behavior before: V3 `DELETE`/`UPDATE` on a table upgraded from V2 with a live parquet position delete refused pre-IO with "BaseDVFileWriter.loadPreviousDeletes is deferred"
Behavior after: for each touched data file, live file-scoped parquet deletes are loaded by reserved ids filtered to the file, only `delete_seq >= data_seq`, unioned once into the new DV (rc 2), and the superseded parquet file is removed in the same RowDelta; partition-scoped deletes are merged per-file but kept live (Spark-equal)
Negative cases: untouched file's parquet stays live; old delete_seq < data_seq does not apply; partition-scoped kept
Test command and population: cargo test -p iceberg-datafusion --test f21_legacy_delete_merge (5 passed); cargo test -p iceberg-datafusion --test interop_f21_legacy_delete_merge (1 passed, Java seed V2 parquet -> Rust V3 DV)
Mutations, one at a time: M1 1 red of 5; M2 1 red of 5; M3 1 red of 5; M4 1 red of 5; M5 oracle red
Java interop command and fixture count: dev/java-interop/run-interop-f21-legacy-delete-merge.sh — 1 fixture + final.metadata.json; sabotage FAIL-closed
CI-only evidence gap: Docker legs of make test excused
Breaking public API change: BasicDeleteFileLoader now pub, load_position_delete_pairs pub
Critic attestation: not yet run (delegated tier)
Open findings and dispositions: none for F-21; R114 residues (equality-delete sort-order, Spark-written shared Puffin) remain
```
