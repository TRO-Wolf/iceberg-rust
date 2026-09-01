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

# PR-6A evidence ledger — branch read and commit interop (row R168 / C-006)

Plan of record: `task/iceberg-v3-production-work-plan-2026-09-01.md` (section 4 PR-6 split, section 11.3 order). Base `00cdde0`.

## 1. Clauses

| Id | Proposition | Result |
|---|---|---|
| C-006 | Every required V3 reference operation reads and commits the named branch without moving `main`. Java and Rust agree. | PROVEN, 9/9 cases both directions including main vs branch file sets on every row-asserting verify |
| C-007 (this unit) | Claimed tests prove the cited behavior. Negative guards have a mutation that turns them red. Interop hard-fails when the fixture cannot run. | PROVEN, mutations 3 red out of 3; sabotage A truncate + sabotage B file-set (ids unchanged) FAIL-closed |

## 2. Required cases (plan section 4 PR-6; MoR UPDATE lineage is PR-6B)

| # | Case | Evidence |
|---|---|---|
| 1 | Rust reads a Java-created branch whose head diverges from `main` | `rust_reads_java_diverged_branch` + `rust_reads_java_v3_diverged_branch`; main vs branch file sets and ids `{1,2}` vs `{1,2,10,11}` |
| 2 | Rust appends through `with_commit_branch`; Java verifies `main` did not move | `gen_rust_append` + Java `verifyAppend` |
| 3 | Rust COW DELETE and COW UPDATE on the branch; Java reads both refs | `gen_rust_cow`; branch ids `{1,2,11}` data `z`; main `{1,2}` |
| 4 | Rust MoR DELETE and MoR UPDATE (rows only) | `gen_rust_mor` + offline `v3_merge_on_read_delete_on_diverged_branch_uses_branch_files`; Java production scan applies DVs |
| 5 | Missing branch fails on read and UPDATE; error names the ref | `missing_branch_fails_on_read_and_update` + `java_missing_branch_and_tag_match_producer` |
| 6 | INSERT-only creates a missing branch (Java `SnapshotProducer` / `latestSnapshot` fallback + `setBranchSnapshot`) | `insert_creates_missing_branch_per_snapshot_producer` + `gen_rust_insert_create` |
| 7 | Tag target refuses writes | `tag_target_refuses_writes`; message is Java `SnapshotProducer.targetBranch` |
| 8 | Retry reloads the named branch head, never `main` | `gen_rust_retry`; pending parent = winner id ≠ main |
| 9 | Java reads a Rust-created branch: parent snapshot + ref kind/retention | `gen_rust_created`; Java `verifyCreated` |

## 3. Java 1.10.0 bytecode

Jar: `~/.m2/repository/org/apache/iceberg/iceberg-core/1.10.0/iceberg-core-1.10.0.jar` and `iceberg-api/1.10.0`.

| Class#method | Decisive instructions |
|---|---|
| `SnapshotProducer.targetBranch(String)` | null rejected (`Invalid branch name: null`). If `base.ref(name)` exists and `!isBranch()`, fail: `%s is a tag, not a branch. Tags cannot be targets for producing snapshots`. Missing ref is allowed. |
| `FastAppend.toBranch` | `invokevirtual targetBranch:(String)V`; return this. |
| `SnapshotProducer.apply()` | `refresh()` then `SnapshotUtil.latestSnapshot(base, targetBranch)` as parent. Retry therefore re-resolves the named ref. |
| `SnapshotUtil.latestSnapshot(TableMetadata, String)` | null or `"main"` → `currentSnapshot()`. Else `ref(name)`; if null → `currentSnapshot()` (create-on-missing parent). Else `snapshot(ref.snapshotId())`. |
| `SnapshotProducer` commit lambda | existing snapshot id → `setBranchSnapshot(id, targetBranch)`; else if not `stageOnly` → `setBranchSnapshot(snapshot, targetBranch)` which creates the branch. |
| `SnapshotScan.useRef` | `"Cannot find ref %s"` when `table.snapshot(name)` is null. `"Cannot override ref, already set snapshot id=%s"` if a snapshot id is already set. |
| Spark `Cannot write to table with time travel` | not on iceberg-core 1.10.0. Core tag refusal is the `targetBranch` message above. |

## 4. Production fix in this unit

V3 MoR DELETE/UPDATE on a diverged branch failed: `close_touched_dv_containers` listed live data files from `current_snapshot()` (main), so a branch-only file was "not live". Additive `close_touched_dv_containers_at(table, positions, snapshot_id)` walks the scanned snapshot. DataFusion `write_deletion_vectors` passes `scan_snapshot_id`. Existing `close_touched_dv_containers` still targets current (maintenance).

## 5. Mutations (N red out of M)

| # | Knob | Command | Result |
|---|---|---|---|
| 1 | `resolve_scan_snapshot_id` named-ref arm → `current_snapshot_id()` | `cargo test -p iceberg-datafusion --test interop_branch_dml -- rust_reproduces_java_diverged_branch_table` | RED: branch SELECT `{1,2}` vs `{1,2,10,11}` |
| 2 | `close_touched_dv_containers_at(..., scan_snapshot_id)` → `None` | `... -- v3_merge_on_read_delete_on_diverged_branch_uses_branch_files` | RED: `not a live file of the scanned snapshot` |
| 3 | Delete `with_target_branch` tag guard | `... -- tag_target_refuses_writes` | RED: INSERT on tag committed |

**3 red out of 3** (scan fallback, DV snapshot `None`, tag guard deleted). Restored from `cp` backup + `touch`; restore re-run green.

Sabotage A: truncate `rust_append/metadata/final.metadata.json` → `FAIL branch-dml/rust_append: missing ...`.

Sabotage B (Critic S2-1): rewrite one basename in `rust_created/expected_branch_files.txt` (live row ids unchanged) → `FAIL branch-dml/rust_created/branch_files`. That is the no-op file-set rewrite: ids stay, file set pin goes red.

## 6. Interop command and fixture count

```
bash dev/java-interop/run-interop-branch-dml.sh
```

Java generate: **4** fixtures (`diverged`, `v3_diverged`, `tag`, `no_branch`). Hard-fail if count ≠ 4.

Rust GEN: **6** tables (`rust_append`, `rust_cow`, `rust_mor`, `rust_created`, `rust_insert_create`, `rust_retry`).

Docker `make test` legs excused (Docker unavailable). Offline `cargo test -p iceberg-datafusion --test interop_branch_dml` and the runner are the evidence.

## 7. Gate exits

Recorded at commit time in this unit's final report.

## 8. Section 9 delivery template (lift into the PR body)

```text
Charter clauses: C-006; C-007 for this unit's tests and interop harness
Matrix rows: row R168
Java methods or bytecode read: SnapshotProducer.targetBranch; FastAppend.toBranch; SnapshotProducer.apply + commit lambda setBranchSnapshot; SnapshotUtil.latestSnapshot(TableMetadata, String); SnapshotScan.useRef ("Cannot find ref %s")
Files changed: crates/iceberg/src/delete_vector_container.rs; crates/integrations/datafusion/src/physical_plan/delete.rs; crates/integrations/datafusion/tests/interop_branch_dml.rs; dev/java-interop/src/main/java/org/apache/iceberg/InteropOracle.java; dev/java-interop/run-interop-branch-dml.sh; docs/parity/GAP_MATRIX.md row R168; maps; task/pr6a-branch-interop-ledger.md; scripts/run_interop_suites.sh SUITE_FLOOR_DEFAULT 56
Behavior before: DataFusion with_commit_branch had unit pins only. V3 MoR DELETE on a diverged branch failed because DV container close listed live files from main.
Behavior after: Java/Rust agree on the nine PR-6A cases. MoR on a named branch closes DVs against the scanned snapshot. main does not move.
Negative cases: missing-ref SELECT and UPDATE name the ref; tag INSERT refused with Java's tag message; truncated rust_append metadata turns Java verify FAIL
Test command and population: cargo test -p iceberg-datafusion --test interop_branch_dml --locked (14 tests; env-gated no-ops offline); cargo test -p iceberg --locked; cargo test -p iceberg-datafusion --locked
Mutations, one at a time: 3 red out of 3 (scan fallback; DV snapshot None; tag guard deleted)
Java interop command and fixture count: dev/java-interop/run-interop-branch-dml.sh — 4 Java fixtures + 6 Rust tables; sabotage FAIL-closed
CI-only evidence gap: Docker make test legs excused; no credentialed catalog
Breaking public API change: additive close_touched_dv_containers_at; close_touched_dv_containers unchanged
Critic attestation: pending independent Critic
Open findings and dispositions: none from Actor. Residue named in row R168 (WAP/stage_only, RewriteManifests/CherryPick throwing default, catalog/session overrides). MoR UPDATE lineage columns are PR-6B.
```
