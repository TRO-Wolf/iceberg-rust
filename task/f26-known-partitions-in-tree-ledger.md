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

# F-26 — the in-tree MoR v3 DELETE/UPDATE hands `known_partitions` to the DV container close

Model: muse-spark-1.3. Base: fork main `189a73ed86c9bd29888fbd545f7957df8df25f18`. Branch `repark/f26-known-partitions-in-tree`.
Matrix row: R114.

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | Counting-Storage pins on the in-tree DataFusion MoR v3 DELETE/UPDATE path (48 data manifests, one row from the oldest file) asserting close reads `== 0`; red at 48 | PROVEN | `mor_delete_close_with_complete_partitions_reads_no_data_manifest`; `mor_update_close_with_complete_partitions_reads_no_data_manifest` |
| C-002 | The delete plan carries `(spec_id, partition)` per touched `file_path` from its own scan tasks into `write_deletion_vectors` as `known_partitions`; no new manifest walk; `live_data_file_partitions` untouched; outcomes identical | PROVEN | `mor_scan_stream` + `to_arrow_with_file_partitions`; batteries + interop §7 |
| C-003 | The two new pins; a partial-map control still walks with identical outcome; end-to-end threading pins; mutations | PROVEN | §4; §6 |
| C-004 | Before/after close reads/opens/wall at 8/48/192 through the in-tree DELETE, three runs, medians | PROVEN | §5 |
| C-005 | GAP_MATRIX R114 dated note; `task/todo.md` item checked with ledger link; `map.md` lockstep; this ledger | PROVEN | row R114; `task/todo.md`; `scan/map.md`; `physical_plan/map.md` |

Brief deviation, recorded: the brief names the newest manifest, but the F-23 r2 newest-first early exit resolves a newest touch in exactly 1 read, so the pins touch the oldest file to hold the brief's `red at 48` number. Scenario (one row, 48 manifests, `== 0`) is unchanged.

## 2. Why the skip never fired in tree

| fact | consequence |
|---|---|
| `delete_legacy_merge.rs` passed `&HashMap::new()` as `known_partitions` | F-23 skip dead code in tree; every DV commit walked all N data manifests |
| F-23 r2 ruled handing partitions down via `live_data_file_partitions` a regression (a full unstoppable walk of its own) | The map must come from work the delete plan already does |
| The delete scan already plans every affected data file with its manifest-entry partition tuple | `FileScanTask.partition` is always `Some(entry.partition)` and `partition_spec` resolves per manifest from the same spec id the close discovers, so the carried map is byte-identical to a fresh walk |

## 3. Production change

| file | change |
|---|---|
| `crates/iceberg/src/scan/partition_work.rs` | New `TableScan::to_arrow_with_file_partitions`: one `plan_files` pass feeds the Arrow reader (same knobs and within-file expand as `to_arrow`) and fills the per-path `(spec_id, partition)` map; entries without a resolvable spec are skipped, leaving a partial map the close walks for |
| `crates/integrations/datafusion/src/physical_plan/delete.rs` | New `mor_scan_stream` shared by both MoR paths; both pass the scan map to `write_merge_on_read_deletes`; both MoR fns return `(count, DvContainerClose)` so tests pin the threading; `apply_dv_container_close` borrows (clones two small vecs per statement) |
| `crates/integrations/datafusion/src/physical_plan/delete_legacy_merge.rs` | `write_deletion_vectors` takes `known_partitions` and forwards it instead of the empty map |
| `crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs` | Size-gate split (new): the V2 parquet position-delete writers, verbatim |
| `crates/iceberg/src/delete_vector_container.rs` | `pub use counting::CountingStorageFactory` alongside the crate-private module |
| `crates/iceberg/src/delete_vector_container/counting.rs` | `CountingStorageFactory` and its fields promoted to `pub` with one-line docs; test-only helpers and imports `#[cfg(test)]` |
| `scripts/check_rust_file_size.py` | `delete.rs` ceiling 1700 → 1163 (two splits: tests, then V2 writers) |

HALT check: carrying the partitions changes no DELETE/UPDATE outcome Java produces. Task values equal the close's own discovery by construction (same manifest entry; spec id parsed from the same manifest). The added DV stamp is pinned `(0, empty)` on the unpartitioned fixture. Row sets, snapshot summaries and delete-file kinds are covered by the unchanged F-17/F-18/F-21 batteries plus both Java interop runners (§7).

## 4. Pins

| id | knob | result |
|---|---|---|
| P-del | counting Storage, 48 data manifests, oldest-file row, complete scan map | `data_manifest_reads == 0`; `added == 1`; referenced `== [oldest]`; DV stamp `(0, empty)` (`mor_delete_close_with_complete_partitions_reads_no_data_manifest`) |
| P-upd | same, update-shaped projection with lineage columns | `data_manifest_reads == 0`; `added == 1` (`mor_update_close_with_complete_partitions_reads_no_data_manifest`) |
| P-partial | same, map minus the oldest path | `data_manifest_reads == 48`; `added == 1`; referenced `== [oldest]` (`mor_delete_close_with_partial_partitions_still_walks_and_matches`) |
| P-thread-del | real `merge_on_read_delete` end to end (commit included) | `deleted == 1`; `added == 1`; `data_sequence_numbers` empty (`mor_delete_threads_scan_partitions_to_the_close`) |
| P-thread-upd | real `merge_on_read_update` end to end (commit included) | `updated == 1`; `added == 1`; `data_sequence_numbers` empty (`mor_update_threads_scan_partitions_to_the_close`) |

## 5. Measurement (debug, this clone, no CI pin)

Command:
`cargo test -p iceberg-datafusion --locked --offline --lib physical_plan::delete::tests::measure_mor_delete_close_at_8_48_192 -- --ignored --nocapture --exact`

In-tree DELETE scan plus close, oldest-file row, empty map (before) vs scan-carried map (after). Three runs per cell.

| n | before reads | before wall r1 / r2 / r3 | before median | after reads | after wall r1 / r2 / r3 | after median |
|---|---|---|---|---|---|---|
| 8 | 8 | 37.584 ms / 36.105 ms / 103.112 ms | 37.584 ms | 0 | 7.664 ms / 7.888 ms / 8.100 ms | 7.888 ms |
| 48 | 48 | 185.195 ms / 182.145 ms / 202.175 ms | 185.195 ms | 0 | 7.920 ms / 12.979 ms / 13.178 ms | 12.979 ms |
| 192 | 192 | 687.166 ms / 685.405 ms / 684.052 ms | 685.405 ms | 0 | 9.738 ms / 9.649 ms / 10.139 ms | 9.738 ms |

Opens (`opens` counter) are 0 in every cell before and after: the pure-DV close opens no parquet. Median 192: 685.405 ms → 9.738 ms.

## 6. Mutations (one knob at a time)

| id | knob | population | result |
|---|---|---|---|
| M1 | production MoR sites pass `&HashMap::new()` again | 5 (`mor_*`) | 2 red of 5 (both threading pins; helper pins and control green) |
| M2 | `write_deletion_vectors` ignores its map parameter | 5 (`mor_*`) | 4 red of 5 (both `== 0` pins at 48 plus both threading pins; partial control green at 48) |

Restored from `/tmp/delete-good2.rs` / `/tmp/merge-good.rs` after each; full `mor_*` set re-run green after restore.

## 7. Interop

| command | result |
|---|---|
| `dev/java-interop/run-interop-f21-legacy-delete-merge.sh` | PASSED (2 fixtures; partition sabotage red) |
| `dev/java-interop/run-interop-f18-dv-sibling-close.sh` | PASSED (summary leg; sabotage red) |

## 8. Gate exits

| gate | exit |
|---|---|
| `make check` | 0 |
| `cargo test -p iceberg --locked --offline` | 0 |
| `cargo test -p iceberg-datafusion --locked --offline` | 0 |
| F-18 and F-21 interop runners | 0 (both PASSED, §7) |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 |
| `python3 scripts/check_rust_file_size.py` | 0 (437 files clean) |

Targeted batteries inside the full runs: `delete_vector_container` 20 passed / 5 ignored; `row_delta` 103 passed / 1 ignored; `shared_puffin_dv` 20 passed / 3 ignored; `f21_legacy_delete_merge` 6 passed.

## 9. Delivery template (section 9)

```text
Charter clauses: C-001 through C-005
Matrix rows: R114 (F-26 in-tree known_partitions)
Base: 189a73ed86c9bd29888fbd545f7957df8df25f18
Java methods or bytecode read: none new (F-23 seq_applies / F-18 skip reused)
Files changed: crates/iceberg/src/scan/partition_work.rs; crates/integrations/datafusion/src/physical_plan/delete.rs; crates/integrations/datafusion/src/physical_plan/delete_legacy_merge.rs; crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs (new); crates/integrations/datafusion/src/physical_plan/delete_tests.rs; crates/iceberg/src/delete_vector_container.rs; crates/iceberg/src/delete_vector_container/counting.rs; scripts/check_rust_file_size.py; docs/parity/GAP_MATRIX.md R114; maps; task/todo.md; task/f26-known-partitions-in-tree-ledger.md
Behavior before: in-tree MoR V3 DELETE/UPDATE passed an empty known_partitions map, so every DV commit walked all N data manifests
Behavior after: the scan-carried map covers every touched path, so the F-23 skip fires (0 reads); a partial map walks only the missed paths; V2 arm and live_data_file_partitions unchanged
Negative cases: partial map still walks all 48 with identical outcome; empty map at production sites 2 red of 5; ignored map in the helper 4 red of 5
Test command and population: cargo test -p iceberg-datafusion --locked --offline --lib physical_plan::delete:: (23 passed); measure_mor_delete_close_at_8_48_192 (ignored)
Mutations, one at a time: see section 6
Java interop command and fixture count: run-interop-f21-legacy-delete-merge.sh (2 fixtures); run-interop-f18-dv-sibling-close.sh
CI-only evidence gap: Docker legs of make test excused
Breaking public API change: additive only — TableScan::to_arrow_with_file_partitions (new) and CountingStorageFactory promoted to pub (test-support surface for the in-tree engine); MoR fn tuple returns are pub(crate)
Critic attestation: Actor only (this unit)
Open findings and dispositions: R114 residues unchanged (equality-delete sort-order; Spark-job-written shared Puffin)
```

## 10. RePark

| note |
|---|
| No repin needed unless RePark's own map is partial — the in-tree map is now complete by construction. |

## 11. Round 2 (critic follow-ups, 2026-09-04)

| item | change | evidence |
|---|---|---|
| S2 reader | One `TableScan::configure_reader` helper feeds `to_arrow` and `to_arrow_with_file_partitions`; the latter streams tasks through a recording tap instead of `try_collect`ing them first | `scan/partition_work.rs`; `scan/mod.rs` shrinks 13 lines |
| S2 pin | New-method batches `==` `to_arrow` batches (data-only; schema field-metadata order is nondeterministic); map names every planned file | `to_arrow_with_file_partitions_matches_to_arrow_and_names_every_file` |
| S3 scope | `dv_partitions_for` (new `mor_scan.rs` size-gate split): DV arm snapshots the shared map and retains touched paths; V2 `PositionDeletes` arm passes empty | `mor_delete_*` / `mor_update_*` pins green |
| S3 split | `mor_scan.rs` extracted; `delete.rs` ceiling 1163 → 1149, `scan/mod.rs` 6892 → 6879 | `check_rust_file_size.py` 438 files clean |
| S3 prose | One-line F-26 corrections in `writer/map.md:36`, `transaction/map.md:48`, `todo.md:69`; F-7 V3-DANGLE-1 ticked (RePark registry FIXED by V3-5 2026-08-31; fork `rewrite_data_files_dv.rs` apply path drops DVs) | prose commit |
| S1 un-publish | HALTED for ruling, see §12 | Q1 |

Re-measured n=48 after-wall cell, five samples (same command as §5; n=8/192 stay at three runs):

| n=48 run | before wall | after wall |
|---|---|---|
| 1 | 182.728 ms | 9.984 ms |
| 2 | 179.541 ms | 10.393 ms |
| 3 | 211.851 ms | 8.009 ms |
| 4 | 178.561 ms | 7.899 ms |
| 5 | 178.869 ms | 13.548 ms |
| median | 179.541 ms | 9.984 ms |

Reads unchanged: before 48 / after 0 every run; opens 0 every run.

Round-2 gate exits (2026-09-04, this clone):

| gate | exit |
|---|---|
| `make check` | 0 |
| `cargo test -p iceberg --locked --offline` | 0 |
| `cargo test -p iceberg-datafusion --locked --offline` | 0 |
| F-21 interop `run-interop-f21-legacy-delete-merge.sh` | 0 |
| F-18 interop `run-interop-f18-dv-sibling-close.sh` | 0 |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 |
| `python3 scripts/check_rust_file_size.py` | 0 (438 files clean) |

## 12. Round-2 questions

| id | class | question | premise | lean |
|---|---|---|---|---|
| Q1 | RULING | S1 demands `delete_tests.rs` gets its own counting storage factory while `counting.rs` reverts to `#[cfg(test)] pub(crate)`. A local factory must implement the `#[typetag::serde]` `StorageFactory` trait (methods take `bytes::Bytes`; registration needs `serde` derives), but `typetag`, `serde` and `bytes` are not direct dependencies of `iceberg-datafusion`, and dependency-file changes fail the unit. Allow adding the three as `[dev-dependencies]` (workspace-pinned, lockfile regenerated offline), or recast the pins onto the `data_sequence_numbers` emptiness signal (no factory anywhere), or keep the factory `pub` in `iceberg`? | `#[cfg(test)]` items in `iceberg` are invisible cross-crate; no re-export path for the three crates exists; `FileIO`/`MemoryCatalog` offer no observer hook | Allow the three workspace-pinned dev-deps; it keeps every pin and count identical |
