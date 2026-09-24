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
| C-5 | (r2) Hadoop drop of a `v3` table: `v1..v3` and the hint absent by name; these survive with their content, each by name: `v0.metadata.json` (parses, K = 0), `V1.metadata.json` (parse error), `v1.metadata.json.bak` (parse error), `v4.metadata.json` (K > N), `00001-<uuid>.metadata.json` (uuid convention), `metadata/sub/v1.metadata.json` (not the pointer's directory), `other.metadata.json` (parse error). Measured parser accepts, so deleted as K <= N, each absent by name (r6 adds the last two): `v01.metadata.json` (K = 1), `v2.gz.metadata.json` (K = 2), `v1.metadata.json.gz` (K = 1) and `v+1.metadata.json` (K = 1) | `hadoop_drop_leaves_near_miss_names` |
| C-6 | (r2) Register a hand-placed `v2000000000.metadata.json` with a hand-placed hint and `v1`, then drop on its own thread and runtime under a 10 s `tokio::time::timeout` on the result: it completes; the pointer, the hint and `v1` are absent | `hadoop_drop_of_registered_huge_version_completes` |
| C-7 | (r2) The listing returns storage-native locations (`LocalFsStorage` without `file://`, `MemoryStorage` without `memory://`); a `file://` warehouse on local fs and a `memory:///warehouse` on memory storage both drop `v1..v3` and the hint and re-create at `v1`, then commit `v2` | `hadoop_drop_then_recreate_with_scheme_qualified_warehouses` |
| C-8 | (r5, replaces the r2 walk clause) A storage whose `list` fails `FeatureUnsupported`, drop at `v3`: `v3` and the hint absent; `v1` and `v2` still hold their pre-drop bytes (read back); a re-create at the same location fails `CatalogCommitConflicts` on the leftover `v1` | `hadoop_drop_without_listing_removes_only_current_file_and_hint` |
| C-9 | (r2) A storage whose `list` fails with any other kind: `drop_table` returns that error; the pointer is gone, `v3` is gone (current-file delete), `v1`, `v2` and the hint stay; (r6) a re-create fails `CatalogCommitConflicts` | `hadoop_drop_propagates_other_list_errors` |
| C-10 | (r5) Registered `v2000000000.metadata.json` on a storage whose `list` fails `FeatureUnsupported`: drop returns `Ok` after exactly 2 deletes (the current file and the hint) and 1 list call; the pointer is absent. The counting storage fails any delete past 64 per catalog, so a reintroduced walk fails instead of hanging | `hadoop_drop_of_huge_registered_version_without_listing_is_bounded` |
| C-11 | (r5, critic V-003) Listing path, drop at `v3` on a counting pass-through storage: exactly 1 list call and 4 deletes (`v3`, `v1`, `v2`, hint); all four absent; re-create at `v1` | `hadoop_drop_lists_metadata_once` |
| C-12 | (r6) Hadoop mode, a registered uuid-named pointer and, separately, a registered unparsable pointer (`custom.json`): drop deletes only the pointer; a hand-placed `v1.metadata.json` and `version-hint.text` keep their content | `hadoop_drop_of_uuid_or_unparsable_pointer_deletes_only_the_pointer` |
| C-13 | (r6) The current-file delete fails at `v3`: `drop_table` returns that error; the pointer is gone and a second drop is `TableNotFound`; `v1`, `v2`, `v3` and the hint stay; a re-create fails `CatalogCommitConflicts` | `hadoop_drop_current_file_delete_error_leaves_chain_and_hint` |
| C-14 | (r6) The `v1` delete fails at `v3`: the error propagates; the pointer is gone, a second drop is `TableNotFound`; `v3` is gone; `v1` and the hint stay (`v2` depends on listing order, not asserted); a re-create fails `CatalogCommitConflicts` | `hadoop_drop_chain_delete_error_leaves_rest_of_chain_and_hint` |
| C-15 | (r6) The hint delete fails at `v3`: the error propagates; the pointer is gone, a second drop is `TableNotFound`; `v1..v3` are gone; the hint still reads `3`; a re-create succeeds at `v1` and overwrites the hint with `1` | `hadoop_drop_hint_delete_error_leaves_only_the_hint` |
| C-16 | (r6, critic V-002) A drop whose listing never completes, cancelled by a 100 ms timeout: the pointer is gone, a second drop is `TableNotFound`; `v3` is gone; `v1`, `v2` and the hint stay | `hadoop_drop_cancelled_at_listing_leaves_chain_without_pointer` |

Mutation check (measured, restored, suite green after): with the chain and hint deletes skipped in
`MetadataNaming::drop_metadata_chain` (renamed `drop_metadata` in r3), C-1, C-2 and C-5 go red; C-3 and C-4 stay green (C-3's only
file is the current one, which `drop_table` deletes itself; C-4 is uuid mode).

Round r2 mutation checks (each measured and restored; suite green after):

| Mutation | Red |
|---|---|
| Listing replaced by the `1..=N` walk | C-6 (times out at 10 s), C-5 (`v2.gz` survives), C-9 (`Ok` instead of the list error) |
| Directory match by exact string instead of scheme-insensitive | C-7 |
| No directory match | C-5 (`metadata/sub/v1.metadata.json` deleted) |

Round r5 mutation checks (each measured on `drop_metadata` / `chain_member`, then reverted with
`git checkout`; suite green after):

| Mutation | Red |
|---|---|
| (a) `file_io.list(metadata_dir)` called twice | C-10, C-11 (list count 2) |
| (b) `chain_member` without the directory match | C-5 (`metadata/sub/v1.metadata.json` deleted) |
| (c) `(1..=version)` widened to `(1..)` | C-5 (`v4.metadata.json` deleted) |
| (d) the `FeatureUnsupported` arm walks `1..=version` again | C-8 (`v1` deleted), C-10 (delete budget exceeded) |
| (e, r6) `chain_member` also requires `!listed.ends_with(".metadata.json.gz")` | C-5 (`v1.metadata.json.gz` must not exist) |
| (f, r6) `chain_member` also requires `!listed.ends_with(".gz.metadata.json")` | C-5 (`v2.gz.metadata.json` must not exist) |
| (g, r6) `chain_member` rejects file names starting `v+` | C-5 (`v+1.metadata.json` must not exist) |
| (h, r6) `chain_member` rejects file names starting `v0` | C-5 (`v01.metadata.json` must not exist); `v0.metadata.json` survives either way (K = 0) |
| (i, r6) a uuid-named pointer is treated as version `i32::MAX` with the directory's hint | C-12 |
| (j, r6) an unparsable pointer deletes the directory's hint | C-12 |
| (k, r6) chain delete errors are ignored | C-14 |
| (l, r6) the hint delete error is ignored | C-15 |
| (m, r6) the current-file delete error is ignored | C-13 |
| (n, r6) `drop_table` removes the pointer after the file deletes instead of before | C-9, C-13, C-14, C-15, C-16 |
| (o, r6) `MemoryCatalog::cache_invalidate` does nothing | `pointer_cache_tests::test_fk4_1_drop_table_evicts_cache_entry`, `register_cache_tests::l2_register_insert_failure_evicts_stale_entry` |

C-6 first awaited `tokio::time::timeout` directly on `drop_table`. Under the walk mutation that
test hung rather than failing: `LocalFsStorage::delete` never yields, so the timer never ran. The
drop now runs on its own thread and runtime, and the test awaits a oneshot under the timeout.

## 3. Decisions

- D-1 (r3 shape): `drop_table` makes one call, `MetadataNaming::drop_metadata`, which deletes the
  current file and then, in Hadoop mode only, the chain and the hint (C-2, C-11). Uuid mode returns
  right after the current-file delete, so default mode is unchanged (C-4). r3 moved the
  current-file delete out of `drop_table` and factored the two cache-invalidate blocks into
  `MemoryCatalog::cache_invalidate` (`caches.rs`; mutation (o)); r4 ratcheted the `catalog.rs`
  size ceiling down to its measured 3147 lines (held by `scripts/check_rust_file_size.py`). The
  one-call shape is code structure, not a behaviour, and is not pinned by a test.
  Hadoop mode parses the dropped location with `MetadataLocation::from_file_path`; a uuid-named
  pointer (a registered uuid table in Hadoop mode) or an unparsable one is a no-op after the
  current-file delete (C-12).
- D-2 (revised in r2): for a `vN` pointer the helper lists the pointer's metadata directory once
  (`FileIO::list`, recursive) and deletes each listed location that `MetadataLocation::from_file_path`
  parses as Hadoop convention with `1 <= K <= N`, and whose directory is the pointer's directory.
  The directory match ignores a leading `scheme://` or `file:` and leading `/`: `FileInfo::location`
  is storage-native, so local fs and memory storage list without the scheme the catalog wrote.
  Then it deletes `version-hint.text`. `FileIO::delete` treats a missing file as success. The
  parser accepts gzip siblings (`vK.gz.metadata.json`, `vK.metadata.json.gz`), leading zeros
  (`v01`) and a leading `+` (`v+1`), so these go when K <= N; all four forms are pinned by name in C-5
  (r6 ruling: Java `Integer.parseInt` accepts a leading `+`, and `HadoopTableOperations` reads both
  gzip suffixes). Data files,
  manifests, other names and the directories stay. r1 walked `1..=N` without listing, so a
  registered `v2000000000` never finished.
- D-4 (r5 ruling, replaces the r2 walk fallback): when `list` fails with `FeatureUnsupported`,
  `drop_metadata` deletes no chain entry beyond the current file (already deleted) and then deletes
  `version-hint.text`. There is no `1..=N` walk anywhere, so a drop never issues more deletes than
  one listing returns, plus the current file and the hint. `MetadataLocation::hadoop_chain` lost its
  only caller and is deleted. Measured by reading the code on 2026-09-24, not pinned by a test in
  this PR: every in-tree production `Storage` implements `list` (`io/storage/memory.rs`,
  `io/storage/local_fs.rs`, OpenDAL `storage_impl.rs`), and only the default trait body answers
  `FeatureUnsupported`, so this arm serves out-of-tree storages only. Java `HadoopCatalog` has no
  per-version walk either (external evidence). Any other list error propagates (C-9).
- D-3: a delete error propagates as the `drop_table` error, as the current-file delete already did
  (C-13 current file, C-14 chain entry, C-15 hint). The pointer is removed first, so the table is
  dropped either way (C-13 to C-16 assert `table_exists` false and a `TableNotFound` retry).

## 4. Residue

- (Fixed in r2, D-2; fallback removed in r5, D-4) The r1 walk issued one delete per version up to
  `N`.
- A storage without `list` leaves `v1..v(N-1)`; a re-create at that location fails
  `CatalogCommitConflicts` (pinned by `hadoop_drop_without_listing_removes_only_current_file_and_hint`).
- A `drop_table` that returns `Err` after the pointer is removed leaves files, and the drop cannot
  be retried (`TableNotFound`). Case 1: the current-file delete fails, so the chain and the hint
  stay (C-13). Case 2: the list fails with a kind other than `FeatureUnsupported`; the chain and the
  hint stay (C-9). Case 3: a chain delete fails part-way, in listing order; the rest of the
  chain and the hint stay (C-14). Case 4: the hint delete fails and the hint stays; the next create
  overwrites it (C-15). In cases 1 to 3, a remaining `v1` makes a re-create fail
  `CatalogCommitConflicts` (C-13, C-9, C-14).
- (critic V-002) A `drop_table` future cancelled at an await after the pointer is removed leaves the
  remaining chain and the hint, with no pointer for a retry (same end state as cases 1–3; C-16
  pins the cancellation at the listing await).
- Purge through maintenance `DeleteReachableFiles` is unchanged (ledger 1, section 8); this PR does
  not touch that path and does not pin it.
- `drop_namespace` still removes pointers without deleting files (ledger 1, section 8); this PR does
  not touch that path and does not pin it.
