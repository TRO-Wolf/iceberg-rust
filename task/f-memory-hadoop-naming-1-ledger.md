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

# Ledger — F-MEMORY-HADOOP-NAMING-1: opt-in Hadoop metadata naming on `MemoryCatalog` (slice 1)

**Ledger id:** `F-MEMORY-HADOOP-NAMING-1-2026-09-23`
**Branch:** `feat/memory-catalog-hadoop-naming` (cut off fork `main` = `962f130c`)
**Scope:** work order prb-fork-r1, ruling Q-55-7
**Matrix rows touched:** R167 (cell note only)
**Model:** Claude Opus 5.5

## 1. Measured gap

The Spark harness's `type=hadoop` catalog answers `v4.metadata.json` after CREATE TABLE plus three
INSERT statements. RePark maps `type=hadoop` to a fork `MemoryCatalog`, which answered
`00003-<uuid>.metadata.json`. Java `HadoopTableOperations` names the first file `v1.metadata.json`
and keeps `metadata/version-hint.text` at the current version.

## 2. Clauses

| Clause | Statement | Proven by (`catalog/memory/hadoop_naming_tests.rs`) |
|---|---|---|
| C-1 | `metadata-naming=hadoop`: create writes `v1.metadata.json` and `version-hint.text` = `1` | `test_hadoop_create_writes_v1_and_hint` |
| C-2 | Three commits reach `v4`; `v1`..`v4` exist; hint = `4` | `test_hadoop_three_commits_reach_v4_and_hint` |
| C-3 | `load_table` returns the current `vN` location | `test_hadoop_load_round_trips_vn_location` |
| C-4 | No property, or `uuid`: `00000-<uuid>` then `00001-<uuid>`, no hint file | `test_default_naming_is_uuid_without_hint`, `test_explicit_uuid_naming_matches_default` |
| C-5 | `Hadoop`, `HADOOP`, `v`, empty string refused at load with `DataInvalid` naming property and value | `test_near_miss_naming_values_refused` |
| C-6 | Hadoop mode plus `write.metadata.path` refused with the `rebased()` message, nothing written, nothing registered | `test_hadoop_refuses_write_metadata_path_before_writing` |
| C-7 | A registered uuid pointer stays uuid-named after a commit; no hint | `test_hadoop_register_uuid_location_stays_uuid` |
| C-8 | A rejected duplicate create returns `TableAlreadyExists` before writing; the registered `v1` bytes, hint and schema are unchanged | `test_hadoop_duplicate_create_keeps_registered_v1_bytes` |
| C-9 | Two racing creates of one name: exactly one `Ok`; the registered `v1` bytes are the winner's (uuid, schema); hint = `1`; run at 8 scheduling offsets | `test_hadoop_concurrent_create_registers_winner_bytes` |
| C-10 | A `v2` hint write that finishes after a later `v3` commit cannot leave the hint behind the pointer | `test_hadoop_hint_follows_pointer_when_older_hint_write_finishes_last` |
| C-11 | Six commits raced from one base with conflict retry all land; pointer `v7`, hint = `7` | `test_hadoop_racing_commits_from_one_base_end_with_hint_at_pointer` |
| C-12 | Default mode, registered `v3` pointer, one commit: `v4`, no hint file | `test_default_naming_register_vn_pointer_writes_no_hint` |
| C-13 | Hadoop create whose hint write fails returns `Err` and registers nothing | `test_hadoop_create_fails_and_registers_nothing_when_hint_write_fails` |

Mutation check: skipping the post-commit hint write turns C-2 red; bypassing the relocation
refusal turns C-6 red.

Round r2fix mutation checks (each restored, suite green after):

| Mutation | Red |
|---|---|
| Pre-fix create: `write_to` for Hadoop `v1`, no registered-name check | C-8, C-9 |
| No registered-name check, exclusive `v1` kept | C-8 |
| Registered-name check kept, `v1` written with `write_to` | C-9 |
| Hint written after `drop(root_namespace_state)` in `update_table` | C-10 |
| `self != Self::Hadoop` guard removed from `advance_version_hint` | C-12 |
| Create ignores a failed hint write | C-13 |

C-9 and C-10 use `SteppingStorage`, a test `MemoryStorage` wrapper that yields before every
operation (and, for C-10, holds the `2` hint write for 200 ms), so the interleavings are
deterministic on the current-thread test runtime.

## 3. Decisions

- D-1: the hint is written only when the catalog is in Hadoop mode AND the new location is
  `vN`-named. A default-mode catalog that registered a `vN` pointer (the F-ICE-HADOOP-VN-1 pins)
  keeps writing no hint, so default mode stays byte-for-byte unchanged.
- D-2: a hint write failure after the pointer swap is logged with `tracing::warn!` and the commit
  returns `Ok`; at create it is an error because nothing is registered yet.
- D-3 (revised in round r2fix, critic V-001): in Hadoop mode create first checks the name is free
  under the catalog lock (`NamespaceState::ensure_table_name_free`, the same errors
  `insert_new_table` returns), then writes `v1` through the exclusive
  `TableMetadata::write_commit_metadata`. A duplicate create fails before writing; a racing
  create that passes the check fails at the exclusive write with `CatalogCommitConflicts` and
  registers nothing. Uuid mode still writes with `write_to`.
- D-5 (round r2fix, critic V-002): `update_table` writes `version-hint.text` before it drops the
  `root_namespace_state` lock that ordered the pointer swap, so hint writes land in pointer order.
  No second mutex. The trade: the hint write is inside the catalog-wide critical section.
- D-4: `metadata-naming` stays in the catalog properties handed to FileIO, like every other
  non-warehouse property.

## 4. Residue

- Staged create (`StagedTableTransaction::begin_create`, CTAS) still names `00000-<uuid>` in
  Hadoop mode; a later slice.
- `publish_replace_table` writes no hint.
- `drop_table` deletes only the current metadata file, so earlier `vK` files and the hint survive a
  non-purge drop. Since D-3 was revised, re-creating a table dropped at `v2` or later fails at
  create with `CatalogCommitConflicts` on the leftover `v1` (Java `HadoopCatalog.dropTable` removes
  the whole table directory). Filed by the orchestrator as a question; drop is out of scope.
- No reader consults `version-hint.text`; the catalog pointer stays authoritative.

## 5. Class sweep (round r2fix)

Class: Hadoop-mode deterministic file names turn unordered or overwriting writes into lost updates.
Every deterministic-path write in `git diff origin/main...HEAD`:

| Write | Site | After r2fix |
|---|---|---|
| `v1.metadata.json` | `create_table` via `MetadataNaming::write_first_metadata` | Exclusive (`write_commit_metadata`), after a registered-name check |
| `version-hint.text` = `1` | `create_table` via `write_first_metadata` | Only after the exclusive `v1` succeeds and before registration, so no commit of that table can run yet |
| `vN.metadata.json` | `update_table` via `write_commit_metadata` (existing seam) | Exclusive |
| `version-hint.text` = `N` | `update_table` via `advance_version_hint` | Ordered under the `root_namespace_state` lock that serialises the pointer swap |

No other write in the diff targets a deterministic path. `register_table` writes nothing. Staged
create stays uuid-named. Staged replace (`publish_replace_table`) writes its `vN` exclusively but no
hint (residue above).
