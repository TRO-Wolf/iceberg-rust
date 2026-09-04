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

# F-24 — the v3 parquet-to-DV arm honours min-input-files

Model: muse-spark-1.3-contributor. Base: fork main `594bdbe5`. Branch `repark/f24-rewrite-pos-deletes-floor`.
Matrix row: R136.

## 1. Propositions

| id | proposition | verdict | evidence |
|---|---|---|---|
| C-001 | Three below-floor cells (2 deletes, 1 delete, 1 partition-scoped delete covering 2 files) return four zeros with live parquet on defaults | PROVEN red then green | `floor_tests::test_v3_two_parquet_deletes_below_the_five_floor_stay_parquet`; `floor_tests::test_v3_one_parquet_delete_below_the_five_floor_stays_parquet`; `floor_tests::test_v3_one_partition_scoped_delete_covering_two_files_stays_parquet` (3 red of 3 pre-fix) |
| C-002 | The v3 arm gates admitted legacy deletes by `(spec_id, partition)` through the shared candidate/pack/group predicates; `rewrite_all` bypasses both filters with packing kept, on both arms | PROVEN | `rewrite_position_delete_files_v3.rs`; `plan_bins` declined return; `ResolvedConfig.rewrite_all` |
| C-003 | Five file-scoped deletes at the floor convert to 5 DVs with second-run zeros; lone-file bypass converts; admitted-shadows-declined refuses naming the gate | PROVEN | `floor_tests::test_v3_five_file_scoped_deletes_at_the_floor_convert_to_one_dv_per_data_file`; `floor_tests::test_v3_rewrite_all_converts_a_lone_parquet_delete_below_the_floor`; `floor_tests::test_v3_refuses_when_an_admitted_vector_would_shadow_a_gate_declined_delete`; `floor_tests::test_admission_rewrite_all_admits_lone_sub_min_file` |
| C-004 | R136 note (V-1 retired), todo, map, this ledger | PROVEN | GAP_MATRIX R136 F-24 note; `task/todo.md`; `maintenance/map.md` |

## 2. Java mechanism

| fact | source |
|---|---|
| `SparkPositionDeletesRewrite` plans v3 groups through `SizeBasedFileRewritePlanner`; below `min-input-files` (default 5) a group is not rewritten | RePark B-MOR-3 measured (PySpark 4.1.2 plus Iceberg 1.11.0, 2026-09-03): 2-file, 1-file, and partition-scoped-2 cells return four zeros with parquet live; 5-file cell converts to 5 PUFFIN |
| `planFileGroups` is `rewriteAll ? tasks : filterFiles`, then pack, then `rewriteAll ? bins : filterFileGroups` | size-gate ledger bytecode (`BinPackRewritePositionDeletePlanner`, offsets 4 and 47); packing runs unconditionally |
| `rewrite-all` has no threshold emulation: `min=0, max=MAX` empties the candidate set, the inverse of the bypass | size-gate ledger C-012 (retired by this unit: the option is now ported) |

## 3. Production change

| file | change |
|---|---|
| `maintenance/rewrite_position_delete_files.rs` | `rewrite_all` field plus builder (default false); `ResolvedConfig.rewrite_all`; `plan_bins` honors the bypass and returns declined entries; v2 caller takes admitted only |
| `maintenance/rewrite_position_delete_files_v3.rs` | new child module: the four arm methods plus inventory, plans, and both shadow refusals; `rewrite_to_deletion_vectors` gates by `(spec_id, partition)`, routes declined into the shadow closure, and returns honest zeros with no commit when nothing is admitted |
| `maintenance/rewrite_position_delete_files_tests.rs` | `write_file_scoped_position_delete_file` uses `position_delete_writer_properties` (truncate off) so the helper is file-scoped as named; 12 collided v3 pins repinned (`min_input_files(2)` or `rewrite_all(true)`); white-box configs gain the field |
| `maintenance/rewrite_position_delete_files_floor_tests.rs` | new child module: honest-zeros pin plus the 7 F-24 pins |
| `tests/interop_rewrite_pos_deletes.rs` | V3 pair topped to 5 deletes per partition (10 total, 2 DVs) via `top_up_to_floor`; `build_pre_world` returns both data paths |
| `tests/interop_v3_upgrade.rs`, `tests/interop_v3_maintenance.rs` | lone/small-shape conversions use `rewrite_all(true)` (u3 offline, u3 GEN, M3) |
| `dev/java-interop/run-interop-rewrite-pos-deletes.sh` | closing echo updated to ten deletes |
| `scripts/check_rust_file_size.py` | `.rs` legacy row removed (946 lines, within default); tests ceiling 4817 lowered to 4716 |

## 4. Pins

| pin | asserts |
|---|---|
| 3 below-floor cells | four zeros, snapshot unchanged, parquet count unchanged, read identity |
| 5-file control | rewritten 5, added 5, 0 parquet, 5 Puffin, read identity, second run zeros |
| lone-file bypass (v3 and v2 arms) | rewritten 1, added 1, read identity |
| admitted-shadows-declined | `DataInvalid`, names the size gate and rewrite-all, fail closed with rows unchanged |
| 12 repinned conversion/closure tests | unchanged expectations under `min_input_files(2)` or `rewrite_all(true)`; partition/spec pin additionally asserts the survivor is Puffin |
| helper file-scoping | every control-test delete satisfies `referenced_data_file_location().is_some()` |

## 5. Mutations (one knob at a time, battery `maintenance::rewrite_position_delete_files`)

| mutation | red | verdict |
|---|---|---|
| Admit every group (drop the gate call) | 4 red of 92: the 3 floor pins plus the gate-shadow pin | PROVEN post-split (pre-split: 4 red of 91) |
| Resolve `rewrite_all` to false (drop the bypass) | 13 red of 92: 12 v3 bypass pins plus the v2 bypass pin | PROVEN post-split (pre-split: 11 red of 91; the partition/spec pin stayed green until hardened with the survivor-format assert, then 12 red of 91) |
| Restore | md5-verified against the pre-mutation snapshot plus `touch`; battery re-run green before exiting | PROVEN |

## 6. Interop

| leg | result |
|---|---|
| `run-interop-rewrite-pos-deletes.sh` V2 | PRE live ids match; read identity; PRE pos 4 fused into POST pos 2 |
| `run-interop-rewrite-pos-deletes.sh` V3 | V3 PRE live ids match; V3 read identity; PRE parquet 10 became POST puffin 2; 0 failures |
| sabotage battery | read-identity breaker, V3 read-identity breaker, and truncate legs all fail closed |
| u3 / M3 GEN direction | offline suites green (`interop_v3_upgrade`, `interop_v3_maintenance`); Java-verify legs unchanged |

## 7. Gate exits

| gate | exit |
|---|---|
| `make check` | 0 (fmt, clippy `-D warnings`, taplo, machete, agent-artifacts, matrix anchors, comment-blocks, rust-file-size 435 files clean / 100 legacy) |
| `cargo test -p iceberg --locked --offline` | 0 (lib 3607 passed / 7 ignored; doctests 90 passed / 10 ignored; every integration target ok) |
| `cargo test -p iceberg-datafusion --locked --offline` | 0 (lib 211 passed; every integration target ok) |
| `dev/java-interop/run-interop-rewrite-pos-deletes.sh` | 0 (V3 PRE parquet 10 became POST puffin 2; sabotage battery closed) |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 (84 rows anchored, 5-pipe audit green) |
| `python3 scripts/check_rust_file_size.py` | 0 (435 files clean) |

Docker legs of `make test` excused.

## 8. File-size split

| file | lines | ceiling | note |
|---|---|---|---|
| `rewrite_position_delete_files.rs` | 946 | 1000 default | legacy row removed; v3 arm plus its inventory/plans/closures moved out |
| `rewrite_position_delete_files_v3.rs` | 507 | 1000 default | new; pure move plus one `pub(super)` on the parent-called entry |
| `rewrite_position_delete_files_tests.rs` | 4716 | 4716 lowered | honest-zeros pin plus 7 floor pins moved out |
| `rewrite_position_delete_files_floor_tests.rs` | 493 | 1000 default | new; pure move plus the v2 bypass pin |

## 9. RePark

| note |
|---|
| Repin closes `B-MOR-3-FLOOR-1`: below-floor parquet stays parquet with honest zeros, the 5-file cell converts, and `rewrite-all` bypasses exactly as Java's option does. |

## 10. Section 9 delivery template

```text
Charter clauses: C-001 through C-004
Matrix rows: R136 (F-24 v3 min-input-files floor; divergence V-1 retired)
Base: 594bdbe5
Java methods or bytecode read: SparkPositionDeletesRewrite plus SizeBasedFileRewritePlanner (RePark B-MOR-3 measured); planFileGroups rewriteAll branches (size-gate ledger offsets 4 and 47)
Files changed: crates/iceberg/src/maintenance/rewrite_position_delete_files.rs; crates/iceberg/src/maintenance/rewrite_position_delete_files_v3.rs; crates/iceberg/src/maintenance/rewrite_position_delete_files_tests.rs; crates/iceberg/src/maintenance/rewrite_position_delete_files_floor_tests.rs; crates/iceberg/tests/interop_rewrite_pos_deletes.rs; crates/iceberg/tests/interop_v3_upgrade.rs; crates/iceberg/tests/interop_v3_maintenance.rs; dev/java-interop/run-interop-rewrite-pos-deletes.sh; scripts/check_rust_file_size.py; docs/parity/GAP_MATRIX.md R136; maps; task/todo.md; task/f24-rewrite-pos-deletes-floor-ledger.md
Behavior before: the v3 arm converted every admitted legacy parquet delete regardless of group size (2-file, 1-file, and partition-scoped-2 cells all converted); no rewrite-all option existed
Behavior after: the v3 arm groups admitted legacy deletes by (spec_id, partition) and runs the shared candidate/pack/group gate, so below-floor groups stay parquet with honest zeros and no commit; rewrite_all(true) bypasses both filters with packing kept on both arms; gate-declined deletes join the shadow closure with a gate-naming refusal
Negative cases: a fully declined run commits nothing; an admitted vector shadowing a declined delete fails closed; dropping the gate reds 4 of 92; dropping the bypass reds 13 of 92
Test command and population: cargo test -p iceberg --locked --offline --lib -- maintenance::rewrite_position_delete_files (92 passed)
Mutations, one at a time: see section 5
Java interop command and fixture count: run-interop-rewrite-pos-deletes.sh (V2 pair 4 deletes; V3 pair 10 deletes five per partition; sabotage battery 3 legs)
CI-only evidence gap: Docker legs of make test excused
Breaking public API change: one additive builder method (rewrite_all); no signature removed
Critic attestation: Actor only (this unit)
Open findings and dispositions: R136 residues (e), (f), (i) and limits (g), (j), (k) unchanged; grouping-key divergence (a) still named
```
