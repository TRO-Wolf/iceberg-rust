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

# Ledger — F-MEMORY-HADOOP-NAMING-2: Hadoop-mode `drop_table` removes the metadata chain (HMETA-DROP)

**Ledger id:** `F-MEMORY-HADOOP-NAMING-2-2026-09-24`
**Branch:** `feat/memory-catalog-hadoop-drop` (cut off fork `main` = `84f92593`, the #347 merge)
**Scope:** work order hmeta-drop, owner ruling run29 route (a), 2026-09-23
**Matrix rows touched:** R167 (cell note only)
**Model:** Claude Opus 5.5

## 1. Measured gap

With `metadata-naming=hadoop`, a non-purge `drop_table` deleted only the current `vN.metadata.json`.
`v1..v(N-1)` and `version-hint.text` stayed, so re-creating the same ident failed at the exclusive
`v1` write. RED, measured before the fix (`hadoop_drop_then_recreate_starts_at_v1`):

```
recreate: CatalogCommitConflicts => Cannot commit table metadata to /tmp/.tmpCSX30h/ns/t/metadata/v1.metadata.json: version file already exists (/tmp/.tmpCSX30h/ns/t/metadata/v1.metadata.json)
```

## 2. Clauses

| Clause | Statement | Proven by (`catalog/memory/hadoop_drop_tests.rs`, module `hadoop_naming_tests::drop_tests`) |
|---|---|---|
| C-1 | Hadoop: create, two commits (`v3`), drop, create the same ident: `v1`, loads, hint `1`; one commit reaches `v2` with hint `2` | `hadoop_drop_then_recreate_starts_at_v1` |
| C-2 | Hadoop drop at `v3`: no `vK.metadata.json` and no `version-hint.text` left in `metadata/`; a data file under `data/` and a manifest list in `metadata/` survive; the table and metadata directories survive | `hadoop_drop_removes_chain_and_hint_keeps_data` |
| C-3 | Hadoop register of an external `v5` (no `v1..v4`, no hint), then drop: `Ok`, no `vK` file left, table directory kept | `hadoop_drop_after_register_of_vn` |
| C-4 | Uuid mode drop: the current file goes, the earlier `00000-<uuid>` file, a hand-placed `v1.metadata.json` and `version-hint.text` stay | `uuid_drop_leaves_hand_placed_hadoop_files` |
| C-5 | Hadoop drop leaves `v1.metadata.json.bak` and `other.metadata.json` untouched | `hadoop_drop_leaves_near_miss_names` |

Mutation check (measured, restored, suite green after): with the chain and hint deletes skipped in
`MetadataNaming::drop_metadata_chain`, C-1, C-2 and C-5 go red; C-3 and C-4 stay green (C-3's only
file is the current one, which `drop_table` deletes itself; C-4 is uuid mode).

## 3. Decisions

- D-1: `drop_table` keeps its current-file delete and then calls
  `MetadataNaming::drop_metadata_chain`. Uuid mode returns at once, so default mode is unchanged.
  Hadoop mode parses the dropped location with `MetadataLocation::from_file_path`; a uuid-named
  pointer (a registered uuid table in Hadoop mode) or an unparsable one is a no-op.
- D-2: for a `vN` pointer the helper deletes `v1..vN.metadata.json` in order, then
  `version-hint.text`, through `FileIO::delete`, which treats a missing file as success. The paths
  come from `MetadataLocation::hadoop_chain`, a lazy iterator. Only exact
  `v<K>.metadata.json` names in the pointer's own metadata directory are touched; gzip siblings,
  data files, manifests and the directories stay.
- D-3: a delete error propagates as the `drop_table` error, as the current-file delete already did.
  The pointer is removed first, so the table is dropped either way.

## 4. Residue

- The chain delete issues one delete per version up to `N`. A registered pointer with a very large
  `N` costs that many deletes; listing `metadata/` instead would bound it by the files present.
- Purge through maintenance `DeleteReachableFiles` is unchanged (ledger 1, section 8).
- `drop_namespace` still removes pointers without deleting files (ledger 1, section 8).
