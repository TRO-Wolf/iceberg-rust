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

# PR-6B ledger — V3 merge-on-read UPDATE row lineage on a branch (Java interop cell)

Model: claude-opus-5 (medium)

Plan of record: `task/iceberg-v3-production-work-plan-2026-09-01.md`, PR-6B. Matrix row R168 (the MoR UPDATE lineage cell PR-6A named as its residue). Branch `repark/pr6b`, based on `repark/pr3` (`cfe3bc979`).

## 1. Clauses

| Id | Proposition | Result |
|---|---|---|
| C-006 (PR-6B cell) | On a diverged branch a V3 merge-on-read UPDATE keeps the updated row's `_row_id`, advances its `_last_updated_sequence_number` once per UPDATE, leaves unmatched rows' lineage untouched, leaves `main` (rows, current snapshot id, `main` ref, `main` file set) untouched, and advances `next-row-id` by the one row each UPDATE adds. Java's production scan of the branch head reads back exactly what Rust wrote. | PROVEN. Measured on the branch head: `_row_id` 3 stable across both UPDATE statements, sequence 2 → 3 → 4, rows 1/2/3/11 unchanged, `main` at `1=0=1 2=1=1 3=2=1` with its single file and its snapshot/ref unmoved, `next-row-id` 5 → 6 → 7. |
| C-007 (PR-6B slice) | The cell is proven by a runner that hard-fails on any missing prerequisite, asserts the fixture count, and carries a sabotage pass so the lineage assertion is non-vacuous. | PROVEN. `dev/java-interop/run-interop-mor-branch-lineage.sh`, six steps, 1 Java fixture asserted twice (`fixture_count.json` and the metadata glob), 7 Rust GEN artifacts asserted non-empty, sabotage RED on a one-knob sequence-number bend. |

## 2. Java 1.10.0 decode

Jars: `~/.m2/repository/org/apache/iceberg/iceberg-{api,core,data}/1.10.0/`.

### `IcebergGenerics$ScanBuilder` (iceberg-data)

`javap -p` shows no `useRef`; the branch head is reached by snapshot id. `javap -c -p` shows both `project(Schema)` and `useSnapshot(long)` reassign the same private `tableScan` field and `areturn` `this` (`TableScan.project` at offsets 0-18; `TableScan.useSnapshot` at 0-15), so `project(schemaWithRowLineage(schema)).useSnapshot(branchHead)` composes — the lineage projection survives the snapshot pin. That is the call the verify mode uses.

### `MetadataColumns` (iceberg-core)

`ROW_ID`, `LAST_UPDATED_SEQUENCE_NUMBER`, and `schemaWithRowLineage(Schema)` are the public surface the oracle projects. Same reader PR-3's `MorUpdateLineageOracle` uses, now driven at a non-current snapshot.

### `ManageSnapshots` / `SnapshotUpdate` (iceberg-api)

`createBranch(String)` exists as a default single-argument overload (branch at the current head) alongside `createBranch(String, long)`. `SnapshotUpdate.toBranch(String)` is a default method, so `newAppend().appendFile(f).toBranch("b").commit()` is the seed's branch-append path.

### `SnapshotProducer` (iceberg-core)

`javap -c -p` shows the row-lineage assignment reading `TableMetadata.nextRowId()` and `ManifestListWriter.nextRowId()` with no branch predicate — table-level, not per-ref. Measured confirmation: the seed's branch append advanced `next-row-id` 3 → 5 and stamped rows 10/11 with `_row_id` 3/4 at sequence 2, exactly as a `main` append would.

## 3. Measured Java seed

`generate-interop-mor-branch-lineage` writes one V3 merge-on-read table (`branch_table`): `main` rows 1/2/3 (`a`/`b`/`c`), then `createBranch("b")`, then `newAppend().toBranch("b")` for rows 10/11 (`x`/`y`).

| Java seed observation | Value |
|---|---|
| branch `b` lineage (`id=_row_id=_last_updated_sequence_number`) | `1=0=1 2=1=1 3=2=1 10=3=2 11=4=2` |
| `main` lineage | `1=0=1 2=1=1 3=2=1` |
| `main` file basenames | `00000-main.parquet` |
| branch file basenames | `00000-main.parquet`, `00001-branch.parquet` |
| `main` snapshot id | `1` |
| `next-row-id` after the seed | `5` |

Direction 2 is judged against these Java-side observations as well as against the Rust GEN pins, so a Rust bug cannot move both sides of the comparison at once.

## 4. Dependency (resolved) and the PR-3 re-pin

**DV closure.** Before PR-6A merged, the first MoR UPDATE on branch `b` failed `DataInvalid => deletion-vector: data file ... is not a live file of the current snapshot`: `close_touched_dv_containers` resolved live data files from `metadata.current_snapshot()` (`main`), so a branch-only data file was not "live". PR-6A's `close_touched_dv_containers_at(table, positions, snapshot_id)` plus `write_deletion_vectors` passing `scan_snapshot_id` is the fix. Both are on fork `main` under this rebase, so the cell is now green on its own branch.

**`next-row-id` re-pin (2026-09-01, after PR-3's final form).** PR-3 reverted the rewrite-aware allocation this unit's first draft was written against. The rule is plain Iceberg, identical in 1.10.0 and 1.11.0: every V3 DATA manifest a snapshot writes with `first_row_id == null` is assigned `next-row-id` and advances it by `existing + added` rows; a carried assigned manifest advances 0. A single-row merge-on-read UPDATE writes one replacement row into a fresh unassigned data manifest, so it advances `next-row-id` by 1. Reproduced red at `interop_mor_branch_lineage.rs:359`, left 7 / right 5.

`next-row-id` is a **table** counter, not a per-ref one, so a branch UPDATE moves it even though `main` does not move. The unit now pins `main`'s snapshot id, `main` ref, `main` file set and `main` lineage as unchanged, and pins `next-row-id` absolutely at 6 after the first UPDATE and 7 after the second.

**Live Spark cross-check.** Spark 4.1.2 + Iceberg 1.11.0 (`_live_parity.build_spark_iceberg_engine`, catalog `local`, Hadoop warehouse under `/tmp`), same layout — V3 table with `write.update.mode=merge-on-read`, `INSERT` 1/2/3 on `main`, `ALTER TABLE ... CREATE BRANCH b`, `INSERT` 10/11 into `branch_b`, then two `UPDATE local.ns.t.branch_b SET val = ... WHERE id = 10`:

| Stage | Spark 4.1.2 / Iceberg 1.11.0 | This fork |
|---|---|---|
| `next-row-id` after the `main` insert | 3 | 3 |
| `next-row-id` after the branch insert | 5 | 5 |
| seed branch `(id, _row_id, seq)` | `(1,0,1) (2,1,1) (3,2,1) (10,3,2) (11,4,2)` | identical |
| seed `main` | `(1,0,1) (2,1,1) (3,2,1)` | identical |
| after UPDATE 1 | id 10 → `_row_id` 3, seq 3; `next-row-id` 6 | identical |
| after UPDATE 2 | id 10 → `_row_id` 3, seq 4; `next-row-id` 7 | identical |
| `main` after both branch UPDATE statements | `(1,0,1) (2,1,1) (3,2,1)` | identical |

`NEXT_ROW_ID_SEQUENCE 5 6 7` on both sides. Nothing in the fork's numbers diverges from the live oracle at this layout.

## 5. Files

- `dev/java-interop/src/main/java/org/apache/iceberg/InteropOracle.java` — `MorBranchLineageOracle` plus the `generate-interop-mor-branch-lineage` / `verify-interop-mor-branch-lineage` modes (a separate mode from PR-6A's `BranchDmlOracle`, so the two PRs do not collide).
- `dev/java-interop/run-interop-mor-branch-lineage.sh` — the six-step runner and its sabotage pass.
- `crates/integrations/datafusion/tests/interop_mor_branch_lineage.rs` — the offline reproduction, the Direction-1 read, the Direction-2 GEN.
- `scripts/run_interop_suites.sh` — `SUITE_FLOOR_DEFAULT` ratcheted to 60, the real discovered count on this branch.
- `docs/parity/GAP_MATRIX.md` — row R168 only.
- `dev/java-interop/map.md`, `crates/integrations/datafusion/tests/map.md`, `task/todo.md`, this ledger.

## 6. Tests

| Test | Result |
|---|---|
| `mor_update_on_branch_keeps_row_id_and_advances_seq_twice` (offline) | updated row keeps one `_row_id` across two branch UPDATE statements; sequence advances each time; unmatched branch rows unchanged; `main` snapshot, `main` ref, files and lineage unchanged; `next-row-id` 5 → 6 → 7 pinned absolutely and as seed + 2; branch file set is `main` plus three |
| `rust_reads_java_branch_lineage` (Direction 1, env-gated) | Rust's branch-head and `main` lineage scans equal Java's seed observations; the Java branch diverges |
| `rust_updates_java_branch_lineage_gen` (Direction 2, env-gated) | Rust updates the Java-created branch twice and writes `rust_after/` for Java to judge |
| Java `verify-interop-mor-branch-lineage` | 0 failures: `main_snapshot` 1, `next_row_id` `5 -> 6 -> 7 (one added row per UPDATE)`, `main_files`, `branch_files`, `main_lineage`, `branch_lineage` `10=3=4`, `updated_row_id` 3, `seq_advanced_twice` 2 → 3 → 4 |

## 7. Mutations (`N red out of M`) — **14 red out of 14**

Rust, command `cargo test -p iceberg-datafusion --test interop_mor_branch_lineage --locked mor_update_on_branch`:

| # | Knob | Result |
|---|---|---|
| R1 | Revert PR-6A's `close_touched_dv_containers_at` to `close_touched_dv_containers` | RED: `DataInvalid => ... is not a live file of the current snapshot` (measured before the rebase; PR-6A is now on `main`) |
| R2 | `lineage_on_ref` drops `use_ref(ref_name)` (scans `main`) | RED: seed branch rows 3, expected 5 |
| R3 | The first UPDATE targets `main` (`with_commit_branch` → `None`) | RED at the `main` snapshot-unmoved assertion |
| R4 | The second UPDATE targets id 11 instead of id 10 | RED at "second update must advance the sequence again" |

Java verify, command `mvn -o -q compile exec:java -Dexec.args=verify-interop-mor-branch-lineage`, one pinned file bent per run and restored:

| # | Knob | Result |
|---|---|---|
| J1 | `expected_branch_lineage.txt`: id 1 sequence 1 → 2 (the runner's own sabotage pass) | RED `FAIL mor-branch-lineage/branch_lineage: id=1 live=0/1 expected=0/2` |
| J2 | `expected_branch_lineage.txt`: id 10 `_row_id` 3 → 99 | RED `... id=10 live=3/4 expected=99/4` |
| J3 | `expected_branch_files.txt`: one basename bent | RED `FAIL mor-branch-lineage/branch_files` |
| J4 | `java_seed_main_snapshot_id.txt` 1 → 2 | RED `FAIL mor-branch-lineage/main_snapshot: current=1 main_ref=1 seed=2` |
| J5 | `java_seed_next_row_id.txt` 5 → 4 | RED `FAIL mor-branch-lineage/next_row_id: seed 4 then 6 then 7 is not one added row per UPDATE` |
| J6 | `java_seed_main_lineage.txt`: id 2 sequence 1 → 9 | RED `FAIL mor-branch-lineage/main_lineage: id=2 live=1/1 expected=1/9` |
| J7 | `updated_id.txt` 10 → 11 | RED twice (`second_update_seq`, `unmatched: id=10`) |
| J8 | `first_update_seq.txt` 3 → 4 | RED `FAIL mor-branch-lineage/second_update_seq: 4 did not advance past the first update 4` |
| J9 | `expected_next_row_id.txt` 7 → 5 | RED `FAIL mor-branch-lineage/next_row_id: 7 expected the Rust pin 5` |
| J10 | `first_update_next_row_id.txt` 6 → 5 | RED `FAIL mor-branch-lineage/next_row_id: seed 5 then 5 then 7 is not one added row per UPDATE` |

Control re-run after each restore: `verify-interop-mor-branch-lineage: 0 failures`, offline test green.

## 8. Interop command and fixture count

```
bash dev/java-interop/run-interop-mor-branch-lineage.sh
```

Java generate: **1** fixture (`branch_table`), asserted by `fixture_count.json` = `{"count":1}` and by the `*/branch_table/metadata/final.metadata.json` count. Rust GEN: **7** artifacts (`rust_table/metadata/final.metadata.json`, `expected_branch_lineage.txt`, `expected_branch_files.txt`, `updated_id.txt`, `first_update_seq.txt`, `expected_next_row_id.txt`, `first_update_next_row_id.txt`), each asserted non-empty. Sabotage step is fail-closed: the runner exits non-zero if the bend does not produce `FAIL mor-branch-lineage/branch_lineage`.

**Reverse leg:** iceberg-core 1.10.0 has no SQL-shaped merge-on-read UPDATE (that surface is Spark's `SparkPositionDelta`, off this oracle's classpath), so the reverse direction stays what PR-3 recorded — Java seeds and judges, Rust performs the MoR UPDATE.

## 9. Gate exits

Recorded at commit time in this unit's final report. Docker legs of `make test` excused (no Docker on this host).

## 10. Section 9 delivery template (lift into the PR body)

```text
Charter clauses: C-006 (MoR UPDATE lineage on a branch cell); C-007 for this unit's runner and sabotage pass
Matrix rows: row R168
Java methods or bytecode read: IcebergGenerics$ScanBuilder.project(Schema) and .useSnapshot(long) (both reassign the same tableScan field and return this, so the lineage projection survives the snapshot pin); MetadataColumns.schemaWithRowLineage / ROW_ID / LAST_UPDATED_SEQUENCE_NUMBER; ManageSnapshots.createBranch(String) default overload; SnapshotUpdate.toBranch(String) default method; SnapshotProducer row-lineage assignment reads TableMetadata.nextRowId with no branch predicate
Files changed: dev/java-interop/src/main/java/org/apache/iceberg/InteropOracle.java (MorBranchLineageOracle + two modes); dev/java-interop/run-interop-mor-branch-lineage.sh; crates/integrations/datafusion/tests/interop_mor_branch_lineage.rs; scripts/run_interop_suites.sh SUITE_FLOOR_DEFAULT 57; docs/parity/GAP_MATRIX.md row R168; dev/java-interop/map.md; crates/integrations/datafusion/tests/map.md; task/todo.md; task/pr6b-mor-branch-lineage-ledger.md
Behavior before: V3 MoR UPDATE lineage on a branch had no interop cell; PR-6A named it as residue
Behavior after: Java seeds a V3 MoR table with a diverged branch b, Rust updates id 10 twice through with_commit_branch("b"), and Java's production scan of the branch head reads _row_id 3 stable with the sequence advanced twice while main's snapshot, ref, file set and lineage are untouched and the table counter next-row-id advances 5 -> 6 -> 7
Negative cases: unmatched branch rows 1/2/3/11 must keep both lineage values; main rows, main ref, current snapshot id and main file set must not move; next-row-id must advance by exactly one per UPDATE, re-derived from Java's own seed rather than trusted from the Rust pin alone; the runner hard-fails on a missing mvn, JDK 11, cargo or python3, on a fixture-count mismatch, on a missing GEN artifact, and when the sabotage bend fails to turn verify red
Test command and population: cargo test -p iceberg-datafusion --test interop_mor_branch_lineage --locked (3 tests; the two env-gated ones are no-ops offline); cargo test -p iceberg-datafusion --locked; bash dev/java-interop/run-interop-mor-branch-lineage.sh
Mutations, one at a time: 14 red out of 14 — 4 Rust (revert the branch-aware DV closure; drop use_ref; update main instead of the branch; update the wrong id second) and 10 Java-verify one-knob bends of the pinned lineage, file set, main snapshot id, seed next-row-id, both next-row-id pins, updated id and first-update sequence
Java interop command and fixture count: bash dev/java-interop/run-interop-mor-branch-lineage.sh — 1 Java fixture + 7 Rust GEN artifacts; sabotage FAIL-closed; reverse MoR UPDATE not in iceberg-core 1.10.0
CI-only evidence gap: Docker make test legs excused; the suite is nightly-only (run_interop_suites.sh floor raised to 57)
Breaking public API change: none — the commit adds no crate source
Critic attestation: independent Critic (Opus 5) NOT PASS on 83bfceeac for record-keeping only (duplicated map rows, stale floor prose, tracked scratch file), substance PASS (every value asserted on both sides, bytecode claims confirmed, 6/6 mutations); fixed by the orchestrator (d7131e00e); re-pinned after PR-3 final with live-Spark identity (32c1992e4); merged as #256.
Open findings and dispositions: none. The PR-6A dependency is satisfied on this rebase. next-row-id is a table counter and moves 5 -> 6 -> 7 across the two branch UPDATE statements; main's snapshot, ref, file set and lineage are the invariants, not main's counter. Cross-checked against live Spark 4.1.2 + Iceberg 1.11.0 at the same layout.
```
