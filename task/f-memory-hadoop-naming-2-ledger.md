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
**Scope:** work order hmeta-drop, owner ruling run29 route (a), 2026-09-23; round r2 (orchestrator ruling tick 47): the chain delete is bounded by a listing
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
| C-2 | Hadoop drop at `v3`: `v1`, `v2`, `v3` `.metadata.json` and `version-hint.text` are each absent by name; a data file under `data/` and a manifest list in `metadata/` survive; the table and metadata directories survive | `hadoop_drop_removes_chain_and_hint_keeps_data` |
| C-3 | Hadoop register of an external `v5` (`v1..v4` and the hint absent by name), then drop: `Ok`, `v5` and the hint absent by name, table directory kept | `hadoop_drop_after_register_of_vn` |
| C-4 | Uuid mode drop: the current file goes, the earlier `00000-<uuid>` file, a hand-placed `v1.metadata.json` and `version-hint.text` stay | `uuid_drop_leaves_hand_placed_hadoop_files` |
| C-5 | (r2) Hadoop drop of a `v3` table: `v1..v3` and the hint absent by name; these survive with their content, each by name: `v0.metadata.json` (parses, K = 0), `V1.metadata.json` (parse error), `v1.metadata.json.bak` (parse error), `v4.metadata.json` (K > N), `00001-<uuid>.metadata.json` (uuid convention), `metadata/sub/v1.metadata.json` (not the pointer's directory), `other.metadata.json` (parse error). Measured parser accepts, so deleted as K <= N: `v01.metadata.json` (K = 1) and `v2.gz.metadata.json` (K = 2) | `hadoop_drop_leaves_near_miss_names` |
| C-6 | (r2) Register a hand-placed `v2000000000.metadata.json` with a hand-placed hint and `v1`, then drop on its own thread and runtime under a 10 s `tokio::time::timeout` on the result: it completes; the pointer, the hint and `v1` are absent | `hadoop_drop_of_registered_huge_version_completes` |
| C-7 | (r2) The listing returns storage-native locations (`LocalFsStorage` without `file://`, `MemoryStorage` without `memory://`); a `file://` warehouse on local fs and a `memory:///warehouse` on memory storage both drop `v1..v3` and the hint and re-create at `v1`, then commit `v2` | `hadoop_drop_then_recreate_with_scheme_qualified_warehouses` |
| C-8 | (r2) A storage whose `list` fails `FeatureUnsupported` falls back to the `1..=N` walk: `v1..v3` and the hint absent, re-create at `v1` | `hadoop_drop_walks_versions_when_listing_is_unsupported` |
| C-9 | (r2) A storage whose `list` fails with any other kind: `drop_table` returns that error; the pointer is gone, `v3` is gone (current-file delete), `v1`, `v2` and the hint stay | `hadoop_drop_propagates_other_list_errors` |

Mutation check (measured, restored, suite green after): with the chain and hint deletes skipped in
`MetadataNaming::drop_metadata_chain` (renamed `drop_metadata` in r3), C-1, C-2 and C-5 go red; C-3 and C-4 stay green (C-3's only
file is the current one, which `drop_table` deletes itself; C-4 is uuid mode).

Round r2 mutation checks (each measured and restored; suite green after):

| Mutation | Red |
|---|---|
| Listing replaced by the `1..=N` walk | C-6 (times out at 10 s), C-5 (`v2.gz` survives), C-9 (`Ok` instead of the list error) |
| Directory match by exact string instead of scheme-insensitive | C-7 |
| No directory match | C-5 (`metadata/sub/v1.metadata.json` deleted) |

C-6 first awaited `tokio::time::timeout` directly on `drop_table`. Under the walk mutation that
test hung rather than failing: `LocalFsStorage::delete` never yields, so the timer never ran. The
drop now runs on its own thread and runtime, and the test awaits a oneshot under the timeout.

## 3. Decisions

- D-1 (r3 shape): `drop_table` makes one call, `MetadataNaming::drop_metadata`, which deletes the
  current file and then, in Hadoop mode only, the chain and the hint. Uuid mode returns right after
  the current-file delete, so default mode is unchanged. r3 moved the current-file delete out of
  `drop_table` and factored the two cache-invalidate blocks into `MemoryCatalog::cache_invalidate`
  (`caches.rs`), so `catalog.rs` stays under its unchanged 3153-line ceiling.
  Hadoop mode parses the dropped location with `MetadataLocation::from_file_path`; a uuid-named
  pointer (a registered uuid table in Hadoop mode) or an unparsable one is a no-op.
- D-2 (revised in r2): for a `vN` pointer the helper lists the pointer's metadata directory once
  (`FileIO::list`, recursive) and deletes each listed location that `MetadataLocation::from_file_path`
  parses as Hadoop convention with `1 <= K <= N`, and whose directory is the pointer's directory.
  The directory match ignores a leading `scheme://` or `file:` and leading `/`: `FileInfo::location`
  is storage-native, so local fs and memory storage list without the scheme the catalog wrote.
  Then it deletes `version-hint.text`. `FileIO::delete` treats a missing file as success. The
  parser accepts gzip siblings (`vK.gz.metadata.json`, `vK.metadata.json.gz`), leading zeros
  (`v01`) and a leading `+` (`v+1`, measured, not pinned), so these go when K <= N. Data files,
  manifests, other names and the directories stay. r1 walked `1..=N` without listing, so a
  registered `v2000000000` never finished.
- D-4 (r2): when `list` fails with `FeatureUnsupported` (the default `Storage::list`), the helper
  falls back to the r1 walk over `MetadataLocation::hadoop_chain` (`v1..vN.metadata.json`). Any
  other list error propagates.
- D-3: a delete error propagates as the `drop_table` error, as the current-file delete already did.
  The pointer is removed first, so the table is dropped either way.

## 4. Residue

- (Fixed in r2, D-2) The r1 walk issued one delete per version up to `N`. The walk remains only
  as the fallback for a storage that cannot list (D-4), where a huge `N` is still unbounded.
- A `drop_table` that returns `Err` after the pointer is removed leaves files, and the drop cannot
  be retried (`NoSuchTable`). Case 1: the current-file delete fails, so the chain and the hint
  stay. Case 2: the list fails with a kind other than `FeatureUnsupported`; the chain and the hint
  stay (C-9). Case 3: a chain delete fails part-way, in listing or walk order; the rest of the
  chain and the hint stay. Case 4: the hint delete fails and the hint stays; the next create
  overwrites it. In cases 1 to 3, a remaining `v1` makes a re-create fail `CatalogCommitConflicts`.
- Purge through maintenance `DeleteReachableFiles` is unchanged (ledger 1, section 8).
- `drop_namespace` still removes pointers without deleting files (ledger 1, section 8).
