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
**Scope:** work order hmeta-drop, owner ruling run29 route (a), 2026-09-23; round r2 (orchestrator ruling tick 47): the chain delete is bounded by a listing; round r8 (HMETA-DROPNS): Hadoop-mode `drop_namespace` refuses a namespace that holds tables
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
| C-7 | (r2, table-driven in r7) The listing returns storage-native locations (`LocalFsStorage` without `file:`, `MemoryStorage` without `memory:`). For every warehouse form each storage accepts, create, commit to `v3`, drop: `v1..v3` and the hint absent by name, re-create at `v1`, commit `v2`. Memory storage: `memory:///warehouse`, `memory://warehouse`, `memory:/warehouse` (red before the r7 fix: `memory:/warehouse: memory:/warehouse/ns/t/metadata/v1.metadata.json must not exist`), `/warehouse`, `warehouse`. Local fs: `<tmp>`, `file://<tmp>` (= `file:///tmp/..`), `file://<tmp without leading />`, `file:<tmp>`, `file:<tmp without leading />` | `hadoop_drop_then_recreate_over_every_accepted_warehouse_form` |
| C-8 | (r5, replaces the r2 walk clause) A storage whose `list` fails `FeatureUnsupported`, drop at `v3`: `v3` and the hint absent; `v1` and `v2` still hold their pre-drop bytes (read back); a re-create at the same location fails `CatalogCommitConflicts` on the leftover `v1` | `hadoop_drop_without_listing_removes_only_current_file_and_hint` |
| C-9 | (r2) A storage whose `list` fails with any other kind: `drop_table` returns that error; the pointer is gone, `v3` is gone (current-file delete), `v1`, `v2` and the hint stay; (r6) a re-create fails `CatalogCommitConflicts` | `hadoop_drop_propagates_other_list_errors` |
| C-10 | (r5) Registered `v2000000000.metadata.json` on a storage whose `list` fails `FeatureUnsupported`: drop returns `Ok` after exactly 2 deletes (the current file and the hint) and 1 list call; the pointer is absent. The counting storage fails any delete past 64 per catalog, so a reintroduced walk fails instead of hanging | `hadoop_drop_of_huge_registered_version_without_listing_is_bounded` |
| C-11 | (r5, critic V-003; prefix r7, critic V-006) Listing path, drop at `v3` on a counting pass-through storage: exactly 1 list call, and its recorded prefix is exactly the pointer's metadata directory as the catalog passes it (`memory:///warehouse/ns/t/metadata`); 4 deletes (`v3`, `v1`, `v2`, hint); all four absent; re-create at `v1` | `hadoop_drop_lists_metadata_once` |
| C-12 | (r6) Hadoop mode, a registered uuid-named pointer and, separately, a registered unparsable pointer (`custom.json`): drop deletes only the pointer; a hand-placed `v1.metadata.json` and `version-hint.text` keep their content | `hadoop_drop_of_uuid_or_unparsable_pointer_deletes_only_the_pointer` |
| C-13 | (r6) The current-file delete fails at `v3`: `drop_table` returns that error; the pointer is gone and a second drop is `TableNotFound`; `v1`, `v2`, `v3` and the hint stay; a re-create fails `CatalogCommitConflicts` | `hadoop_drop_current_file_delete_error_leaves_chain_and_hint` |
| C-14 | (r6) The `v1` delete fails at `v3`: the error propagates; the pointer is gone, a second drop is `TableNotFound`; `v3` is gone; `v1` and the hint stay (`v2` depends on listing order, not asserted); a re-create fails `CatalogCommitConflicts` | `hadoop_drop_chain_delete_error_leaves_rest_of_chain_and_hint` |
| C-15 | (r6) The hint delete fails at `v3`: the error propagates; the pointer is gone, a second drop is `TableNotFound`; `v1..v3` are gone; the hint still reads `3`; a re-create succeeds at `v1` and overwrites the hint with `1` | `hadoop_drop_hint_delete_error_leaves_only_the_hint` |
| C-16 | (r6, critic V-002) A drop whose listing never completes, cancelled by a 100 ms timeout: the pointer is gone, a second drop is `TableNotFound`; `v3` is gone; `v1`, `v2` and the hint stay | `hadoop_drop_cancelled_at_listing_leaves_chain_without_pointer` |
| C-17 | (r7) Default (uuid) mode, a registered `v3.metadata.json` pointer with hand-placed `v1`, `v2` and hint: drop deletes only `v3`; `v1`, `v2` and the hint keep their content | `uuid_mode_drop_of_registered_vn_pointer_deletes_only_the_pointer` |
| C-18 | (r8, HMETA-DROPNS) Hadoop mode, namespace `db` with a table: `drop_namespace` fails `NamespaceNotEmpty` with the full message `Namespace db is not empty.`; `db` still exists, the table still loads at the same location, and its `v1.metadata.json` still exists. Red before the fix: `namespace holds a table: ()` | `hadoop_mode_drop_namespace_with_table_is_refused` |
| C-19 | (r8) Hadoop mode, empty namespace: `drop_namespace` succeeds and the namespace is gone | `hadoop_mode_drop_of_empty_namespace_succeeds` |
| C-20 | (r8) Hadoop mode, `a.b` holds a table: `drop_namespace(a)` fails `NamespaceNotEmpty` `Namespace a is not empty.`, `drop_namespace(a.b)` fails `Namespace a.b is not empty.`; both namespaces and the table remain | `hadoop_mode_drop_namespace_refuses_a_table_in_a_descendant` |
| C-21 | (r8) Hadoop mode, `a` with only an empty child `a.b`: `drop_namespace(a)` succeeds and both are gone | `hadoop_mode_drop_namespace_with_only_an_empty_child_succeeds` |
| C-22 | (r8) Hadoop mode, the namespace's only table dropped first: `drop_namespace` succeeds | `hadoop_mode_drop_namespace_after_its_table_was_dropped_succeeds` |
| C-23 | (r8) Uuid mode, namespace with a table: `drop_namespace` succeeds as before; the namespace is gone, `load_table` fails `NamespaceNotFound`, and the metadata file stays on storage | `uuid_mode_drop_namespace_with_table_still_succeeds` |
| C-24 | (r8) Hadoop mode, missing `missing` and `db.missing`: `NamespaceNotFound` with the same message as uuid mode, `No such namespace: <NamespaceIdent debug>` | `hadoop_mode_drop_of_missing_namespace_keeps_the_not_found_error` |

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
| (p, r7) `storage_path` loses the `memory:` prefix strip | C-7 (`memory:/warehouse: memory:/warehouse/ns/t/metadata/v1.metadata.json must not exist`) |
| (q, r7) the drop lists the table directory (parent of `metadata/`) | C-11 (prefix `memory:///warehouse/ns/t` instead of `.../metadata`) |
| (r, r7) the drop lists the warehouse root | C-11 (prefix `memory:///warehouse` instead of `.../metadata`) |
| (s, r7) the hint delete is skipped | C-2, C-5, C-6, C-7, C-8, C-10, C-11, C-15 |
| (t, r7) the drop also deletes `<table>/data` | C-2 (data file) |
| (t2, r7) `chain_member` also accepts `*.avro` | C-2 (manifest list) |
| (t3, r7) the drop deletes the whole `metadata/` directory | C-2, C-5, C-8, C-15 |
| (t4, r7) the drop deletes the whole table directory | C-2, C-3, C-5, C-8, C-15 |
| (u, r7) the current-file delete is skipped | C-4, C-8, C-9, C-10, C-12, C-16, C-17 |
| (v, r7) the hint delete first reads the hint (a missing hint becomes an error) | C-3, C-10 |
| (w, r7) the `self != Hadoop` early return is removed | C-17 |
| (x, r7) uuid mode lists the directory and deletes every `*.metadata.json` | C-4 (earlier `00000-<uuid>` file gone), C-17 (`v1` gone) |
| (y, r7) `(1..=version)` widened to `(0..=version)` | C-5 (`v0.metadata.json`) |
| (z, r7) `/V` read as `/v` before parsing | C-5 (`V1.metadata.json`) |
| (z2, r7) a `.bak` suffix stripped before parsing | C-5 (`v1.metadata.json.bak`) |
| (z3, r7) a parse error counts as K = 1 | C-2, C-5, C-11, C-14, C-15 |
| (aa, r7) a uuid-named listed entry counts as K = 1 | C-5 (`00001-<uuid>.metadata.json`) |
| (bb, r7) the no-list arm returns before the hint delete | C-8, C-10 |
| (cc, r7) `FeatureUnsupported` propagates as an error | C-8, C-10 |
| (dd, r7) other list errors are swallowed | C-9 |
| (ee, r7) the hint is deleted twice | C-10, C-11 (delete counts) |
| (gg, r7) `storage_path` loses the `file:` strip | C-7 (`file:/tmp/..` form) |
| (hh, r7) `storage_path` keeps leading `/` | C-7, C-11, C-14, C-15 |
| (ff, r7) `storage_path` also strips any `<x>:` prefix | none: not discriminated by a drop test (see D-2) |
| (o, r6) `MemoryCatalog::cache_invalidate` does nothing | `pointer_cache_tests::test_fk4_1_drop_table_evicts_cache_entry`, `register_cache_tests::l2_register_insert_failure_evicts_stale_entry` |

Round r8 (HMETA-DROPNS) mutation checks on `NamespaceState::ensure_droppable` / `holds_tables`, one
per run, each reverted with `git checkout`:

| Mutation | Red |
|---|---|
| (ns-p1) the refusal never applies (`false &&` on the guard) | C-18 (`namespace holds a table: ()`), C-20 |
| (ns-p2) the refusal also applies in uuid mode | C-23 (`drop namespace: NamespaceNotEmpty => Namespace db is not empty.`) |
| (ns-p3) only direct tables count, not descendants | C-20 (`Namespace a is not empty.: ()`) |
| (ns-p4) the namespace renders with `{:?}` | C-18, C-20 (`Namespace NamespaceIdent(["db"]) is not empty.`) |
| (ns-p5) `holds_tables` always true | C-19, C-21, C-22 |
| (ns-p6) a child namespace counts as a table | C-21 |
| (ns-p7) Hadoop mode answers a missing namespace with its own message | C-24 |

C-6 first awaited `tokio::time::timeout` directly on `drop_table`. Under the walk mutation that
test hung rather than failing: `LocalFsStorage::delete` never yields, so the timer never ran. The
drop now runs on its own thread and runtime, and the test awaits a oneshot under the timeout.

## 3. Decisions

- D-1 (r3 shape): `drop_table` makes one call, `MetadataNaming::drop_metadata`, which deletes the
  current file and then, in Hadoop mode only, the chain and the hint (C-2, C-11; Hadoop-only pinned
  by C-17, mutation (w)). Uuid mode returns right after the current-file delete, so default mode is
  unchanged (C-4, C-17). r3 moved the
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
  The directory match ignores a leading `scheme://`, a `file:` or `memory:` prefix (r7, critic
  V-005: `memory:/x` is a form `MemoryStorage` accepts) and leading `/` (C-7, mutations (p), (gg),
  (hh)). No other `<x>:` prefix is stripped, so a Windows drive letter such as `C:` survives; this
  is not pinned: every listed entry shares the pointer's prefix, so both sides of the comparison
  normalize alike and no drop test can tell a generic strip apart (mutation (ff) stays green). Only
  a unit test on `storage_path` inside `metadata_naming.rs` would pin it. `FileInfo::location` is
  storage-native, so local fs and memory storage list without the scheme the catalog wrote (C-7).
  Then it deletes `version-hint.text` (mutation (s)). `FileIO::delete` treats a missing file as
  success (C-3, mutation (v)). The
  parser accepts gzip siblings (`vK.gz.metadata.json`, `vK.metadata.json.gz`), leading zeros
  (`v01`) and a leading `+` (`v+1`), so these go when K <= N; all four forms are pinned by name in C-5
  (r6 ruling: Java `Integer.parseInt` accepts a leading `+`, and `HadoopTableOperations` reads both
  gzip suffixes). Data files,
  manifests, other names and the directories stay (C-2, C-5; mutations (t) to (t4), (y) to (aa)).
  The listing is recursive; that is a storage property
  (`io/storage/local_fs_tests.rs` `test_list_returns_exact_recursive_file_set_with_sizes_and_times`),
  and its consequence for the drop, that `metadata/sub/` is not touched, is C-5, mutation (b). r1 walked `1..=N` without listing, so a
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
- D-5 (r8, HMETA-DROPNS, owner ruling in run29 claims line 157, date fixed at line 158): in Hadoop
  mode `drop_namespace` refuses a namespace that holds a table, directly or in any descendant
  namespace, with `ErrorKind::NamespaceNotEmpty` and the message `Namespace <ns> is not empty.`,
  where `<ns>` is the levels joined with `.` (C-18, C-20; ns-p1, ns-p3, ns-p4). The check is
  `NamespaceState::ensure_droppable`, run under the same lock as the removal, and `holds_tables`
  walks the subtree with an explicit stack, not recursion; both are code structure and not pinned
  (no test here can race a table insert between the check and the removal). An empty namespace, one with only empty child
  namespaces, and one whose tables were dropped are still removed (C-19, C-21, C-22; ns-p5, ns-p6).
  A missing namespace falls through to `remove_existing_namespace` and keeps its error (C-24,
  ns-p7). Uuid mode is unchanged (C-23, ns-p2). The message is the one the owner measured from
  Java `HadoopCatalog.dropNamespace` (external evidence, not pinned by a Rust test).
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
- (critic V-002, narrowed in r7) A `drop_table` future cancelled during the listing await leaves
  the chain below the current file and the hint, with no pointer for a retry (C-16). Cancellation
  at a per-file delete await or at the hint delete await is not pinned: in this harness a delete
  that completes before the cancel is observed cannot be told apart from one that does not.
- Purge through maintenance `DeleteReachableFiles` is unchanged (ledger 1, section 8); this PR does
  not touch that path and does not pin it.
- (r8) Uuid-mode `drop_namespace` still removes a namespace with its table pointers and deletes no
  files (C-23). Hadoop mode now refuses instead (D-5). Views are not counted: a Hadoop-mode
  namespace that holds only views is still dropped with them; this is not pinned.
