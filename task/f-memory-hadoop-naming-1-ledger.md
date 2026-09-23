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

Mutation check: skipping the post-commit hint write turns C-2 red; bypassing the relocation
refusal turns C-6 red.

## 3. Decisions

- D-1: the hint is written only when the catalog is in Hadoop mode AND the new location is
  `vN`-named. A default-mode catalog that registered a `vN` pointer (the F-ICE-HADOOP-VN-1 pins)
  keeps writing no hint, so default mode stays byte-for-byte unchanged.
- D-2: a hint write failure after the pointer swap is logged with `tracing::warn!` and the commit
  returns `Ok`; at create it is an error because nothing is registered yet.
- D-3: create writes `v1` with `TableMetadata::write_to`, the same overwrite write the uuid path
  uses, not the exclusive `write_commit_metadata` create.
- D-4: `metadata-naming` stays in the catalog properties handed to FileIO, like every other
  non-warehouse property.

## 4. Residue

- Staged create (`StagedTableTransaction::begin_create`, CTAS) still names `00000-<uuid>` in
  Hadoop mode; a later slice.
- `publish_replace_table` writes no hint.
- `drop_table` deletes only the current metadata file, so a re-created table in Hadoop mode can
  collide with a leftover `v2` on its first commit (Java `HadoopCatalog.dropTable` removes the
  whole table directory).
- No reader consults `version-hint.text`; the catalog pointer stays authoritative.
