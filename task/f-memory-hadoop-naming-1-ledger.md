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
| C-4 | No property, or `uuid`: `00000-<uuid>` then `00001-<uuid>`; no hint file right after create or after the commit (r8fix: absence by path) | `test_default_naming_is_uuid_without_hint`, `test_explicit_uuid_naming_matches_default` |
| C-5 | `Hadoop`, `HADOOP`, `v`, empty string refused at load with `DataInvalid` naming property and value | `test_near_miss_naming_values_refused` |
| C-6 | Hadoop mode plus `write.metadata.path` refused with the `rebased()` message, nothing written, nothing registered | `test_hadoop_refuses_write_metadata_path_before_writing` |
| C-7 | A registered uuid pointer stays uuid-named after a commit; no hint right after register or after the commit (r8fix: absence by path) | `test_hadoop_register_uuid_location_stays_uuid` |
| C-8 | A rejected duplicate create returns `TableAlreadyExists` before writing; the registered `v1` bytes, hint and schema are unchanged | `test_hadoop_duplicate_create_keeps_registered_v1_bytes` |
| C-9 | Two racing creates of one name: exactly one `Ok`; the registered `v1` bytes are the winner's (uuid, schema); hint = `1`; run at 8 scheduling offsets | `test_hadoop_concurrent_create_registers_winner_bytes` |
| C-10 | A `v2` hint write that finishes after a later `v3` commit cannot leave the hint behind the pointer | `test_hadoop_hint_follows_pointer_when_older_hint_write_finishes_last` |
| C-11 | Six commits raced from one base with conflict retry all land; pointer `v7`, hint = `7` | `test_hadoop_racing_commits_from_one_base_end_with_hint_at_pointer` |
| C-12 | Default mode, registered `v3` pointer, one commit: `v4`; no hint file right after register or after the commit (r8fix: absence by path) | `test_default_naming_register_vn_pointer_writes_no_hint` |
| C-13 | (r6fix, replaces the r3/r4 Err rule) A Hadoop create whose hint write fails (hint path obstructed by a directory) returns `Ok`, registered at an existing `v1`; a second create of the name returns `TableAlreadyExists`; with the obstruction gone, the first commit reaches `v2` with hint = `2` | `test_hadoop_create_with_failed_hint_registers_v1_and_commits_on` |
| C-14 | `create_table(T)` raced against `register_table(T, v3)` at T's default location, 32 rounds on fresh tempdirs: exactly one wins; a hint, if present, equals the registered pointer's version; a losing create leaves no `v1`; each side wins at least once | `test_hadoop_create_racing_register_keeps_hint_at_registered_pointer` |
| C-15 | (r6fix) A hint write that puts `1` at the final hint path and then fails: `create_table` returns `Ok`, registered at an existing `v1`; the hint reads `1`, the bytes the storage left | `test_hadoop_create_succeeds_when_hint_write_fails_after_bytes` |
| C-16 | (r6fix) The same failure over a pre-existing `version-hint.text` (`7`): `Ok`, registered at `v1`; the hint file still exists and reads `1` | `test_hadoop_create_succeeds_over_pre_existing_hint_when_hint_write_fails` |
| C-17 | Hadoop `rename_table(t, u)` fails `FeatureUnsupported` with the full message `Cannot rename Hadoop tables`; `t` still exists and loads at `v1`; `u` does not exist | `test_hadoop_rename_refused_without_state_change` |
| C-18 | After the refused rename, `create_table(u)` succeeds at `ns/u/metadata/v1.metadata.json` with hint `1`; `t` keeps its pointer, uuid and hint | `test_hadoop_create_at_target_name_after_refused_rename` |
| C-19 | Uuid mode, the near miss: rename `t -> u` succeeds, then `create_table(t)` succeeds with a fresh `00000-<uuid>` file; no hint after either create or after a commit (r8fix) | `test_uuid_rename_then_create_at_old_name_succeeds` |
| C-20 | Hadoop rename with a missing source returns the refusal first, as Java does, not `NoSuchTable`; `u` does not exist | `test_hadoop_rename_of_missing_source_is_refused_first` |

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
| Create ignores a failed hint write | C-13 (superseded in r6fix: ignoring the failure is now the rule, D-2) |

Round r3fix mutation checks (each restored, suite green after):

| Mutation | Red |
|---|---|
| No `v1` delete after a failed create-time hint write | C-13 (superseded in r6fix: the delete path is gone, D-2) |
| Hadoop create releases the lock between the name check and the insert (r2fix shape) | C-14 (round `delay_register true, yields 0`: hint `1`, pointer `v3`) |

Round r5fix mutation check (restored, suite green after): with the Hadoop refusal in `rename_table`
ignored, C-17, C-18 and C-20 go red. C-17 and C-18 fail because the rename succeeds. C-20 fails
on the error kind, because the missing source then reaches the pointer move and its not-found
error.

Round r4fix mutation checks (each restored, suite green after):

| Mutation | Red |
|---|---|
| No hint delete after a failed create-time hint write | C-15 (superseded in r6fix: the delete path is gone, D-2) |
| Hint deleted even when it existed before the create | C-16 (superseded in r6fix: the delete path is gone, D-2) |

Round r8fix (critic r8 V-001): the test helper `hint()` used to map every read error to `None`, so
`hint(..) == None` could pass for an unreadable hint. Absence is now asserted with
`symlink_metadata` returning `NotFound` (`assert_absent`, `assert_no_hint`, both
`#[track_caller]`). A positive `hint()` read panics on a read error, and the race test reads an
optional hint with `hint_if_present`, which returns `None` only on `NotFound`. Mutation, measured
and reverted: a uuid-mode create that also writes `version-hint.text` makes C-4 (both tests) and
C-19 go red, at the absence check right after create.

Round r6fix mutation checks (each restored, suite green after):

| Mutation | Red |
|---|---|
| (M-a) Create returns the hint write's error again | C-13, C-15, C-16 |
| (M-b) `stage_and_publish` returns `Err` when the post-link temp removal fails | `test_stage_and_publish_temp_removal_failure_after_link_is_ok` |

C-15 and C-16 use `SteppingStorage` with `failing_hint` set: it writes the hint bytes through the
local-fs storage, then returns an error.

C-9, C-10 and C-14 use `SteppingStorage`, a test `MemoryStorage` wrapper that yields before every
operation (and, for C-10, holds the `2` hint write for 200 ms), so the interleavings are
deterministic on the current-thread test runtime. C-14 wraps the local-fs storage, C-9 and C-10
the in-memory storage.

## 3. Decisions

- D-1: the hint is written only when the catalog is in Hadoop mode AND the new location is
  `vN`-named. A default-mode catalog that registered a `vN` pointer (the F-ICE-HADOOP-VN-1 pins)
  keeps writing no hint, so default mode stays byte-for-byte unchanged.
- D-2 (revised in round r6fix; replaces the r3fix and r4fix cleanup rules): in Hadoop mode a hint
  write failure is warn-only at create, exactly as after a commit. Once `v1` is published, the
  create registers the table at `v1` and returns `Ok`. `advance_version_hint` is the single hint
  writer for both and logs the path and error with `tracing::warn!`. There is no delete of `v1`
  or of the hint on any create path. Java evidence, measured by the orchestrator with javap of
  `org/apache/iceberg/hadoop/HadoopTableOperations.class` in iceberg-spark-runtime-4.1_2.13-1.11.0.jar
  (`/tmp/xo-xo-opus64/wo/jprobe/hto.javap`). Lines 270-300: `doCommit`, used for create and
  commit, calls `renameToFinal`, then `writeVersionHint(nextVersion)`. Lines 465-506:
  `writeVersionHint` writes a temp file, deletes the old hint, renames, catches `IOException`
  and logs "Failed to update version hint". A Java Hadoop create whose hint write fails
  succeeds. The r3fix/r4fix rule (fail the create, delete `v1` and a newly written hint) could
  return `Err` with `v1` left behind whenever its best-effort delete failed, and the retry then
  hit `CatalogCommitConflicts` (critic r6 V-002).
- D-8 (round r6fix, critic r6 V-001): `LocalFsStorage` exclusive publish (`stage_and_publish`)
  returns `Ok` once `hard_link` has placed the destination. A failure to remove the temp link
  afterwards is logged with `tracing::warn!` (temp, destination, error) and is not an error. The
  staging-failure and link-failure branches keep their errors (`AlreadyExists` stays
  `PreconditionFailed`). The post-link removal goes through the private
  `stage_and_publish_with(.., remove_published_temp)` so a test can inject the failure.
- D-9 (round r6fix): the Hadoop create holds the name as a `hash_map::VacantEntry`
  (`NamespaceState::vacant_table_slot`, which replaces `ensure_table_name_free` and returns the
  same errors as `insert_new_table`) across the `v1` write, under the catalog lock. The
  registration after a published `v1` is `VacantEntry::insert`, which cannot fail.
- D-3 (revised in round r2fix, critic V-001): in Hadoop mode create first checks the name is free
  under the catalog lock (`NamespaceState::ensure_table_name_free`, the same errors
  `insert_new_table` returns; `vacant_table_slot` since r6fix, D-9), then writes `v1` through the exclusive
  `TableMetadata::write_commit_metadata`. A duplicate create fails before writing; a racing
  create that passes the check fails at the exclusive write with `CatalogCommitConflicts` and
  registers nothing. Uuid mode still writes with `write_to`.
- D-7 (round r5fix, critic r5 V-001): in Hadoop mode `rename_table` returns
  `FeatureUnsupported` "Cannot rename Hadoop tables" before it takes the catalog lock
  (`MetadataNaming::ensure_rename_supported`). Java `HadoopCatalog.renameTable` throws
  `UnsupportedOperationException("Cannot rename Hadoop tables")`; the orchestrator measured this
  from the `HadoopCatalog.class` string table in iceberg-spark-runtime-4.1_2.13-1.11.0.jar. A
  pointer-only rename left `v1.metadata.json` at the old name's derived location, so a later
  create at that name failed `CatalogCommitConflicts`. Uuid mode rename is unchanged.
- D-6 (round r3fix, critic V-007): in Hadoop mode `create_table` takes the `root_namespace_state`
  lock once and holds it across the name check, the `v1` write, the hint write and the insert, so
  no `register_table` or `create_table` can take the name in between. Same trade as D-5. Uuid mode
  keeps its writes outside the lock.
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
  the whole table directory). Filed by the orchestrator as a question; drop is out of scope
  (HMETA-DROP; fixed in [F-MEMORY-HADOOP-NAMING-2](f-memory-hadoop-naming-2-ledger.md)). The same holds for `drop_namespace`, which removes a namespace together with its
  table pointers and deletes no files; see section 8 (Hadoop mode refuses since
  [F-MEMORY-HADOOP-NAMING-2](f-memory-hadoop-naming-2-ledger.md) D-5).
- No reader consults `version-hint.text`; the catalog pointer stays authoritative.
- A failed create-time or post-commit hint write (warn-only, D-2) can leave an empty or truncated
  `version-hint.text` on local fs, because `LocalFsStorage::write` is `fs::write`. A partial write
  is a prefix of the decimal digits, so it is empty or parses to a smaller version. Java
  `HadoopTableOperations` 1.10.0 was checked locally from `javap` bytecode only, not by running
  Java against a Rust-written table. `findVersion` catches any `Exception` from the hint read and
  parse; an empty hint throws there, so it falls back to listing `metadata/` and taking the
  highest `vN` whose file exists. For a smaller parsed version `K`, `refresh` loads `vK`, then
  walks forward while `v(K+1)` exists. The Rust crate never deletes older `vK` files, so both
  cases reach the newest contiguous version. If `vK` is missing, Java throws `ValidationException`
  "Metadata file for version K is missing". This cannot happen for a hint MemoryCatalog wrote,
  unless something outside the crate deletes `vK`. Other Java versions and non-local backends
  were not checked; object-store PUTs do not leave partial objects.

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

## 6. Class sweep (round r3fix)

Class: Hadoop create's durable writes are not atomic with registration, through a check-then-act
across a lock release, or a fallible step after a durable write with no undo. Hadoop-mode paths in
`git diff origin/main...HEAD`:

| Path | Durable write | Later fallible step | (a) Under the pointer lock | (b) Undone or harmless on a later failure |
|---|---|---|---|---|
| `create_table` | `v1.metadata.json` (exclusive) | hint write | Yes (D-6) | Superseded in r6fix: the hint failure only warns and the create registers `v1` (D-2) |
| `create_table` | `version-hint.text` = `1` | `insert_new_table` | Yes (D-6) | Harmless: the insert cannot fail (since r6fix it is `VacantEntry::insert`, D-9) |
| `create_table` | none after insert | `table_builder().build()` | n/a | Harmless: every required field is set; code unchanged from `main` |
| `update_table` | `version-hint.text` = `N` | none | Yes (D-5) | Last step; failure only warns (V-002 ruling) |
| `register_table` | none | — | Insert under lock | Nothing to undo |
| `rename_table` | none (pointer move only) | — | Yes | The hint lives in the table directory, which does not move |
| staged create | `00000-<uuid>` (not deterministic) | — | — | Outside the class; not in this diff |
| staged replace | none in this diff | — | — | Accepted residue: no hint |

No other site of the class is in this PR's code. Pre-existing on `main`, not in this diff:
`update_table` and staged replace write `v(N+1)` exclusively before the final pointer CAS, so a
CAS failure leaves an orphan `v(N+1)`. R167 already records this: the next commit fails loud, and
re-registering at the newest version recovers.

## 7. Class sweep (round r4fix)

Class: a durable write in a Hadoop path whose failure, including a partial write of that same file,
leaves bytes a later reader or retry can see. Hadoop-mode paths in `git diff origin/main...HEAD`:

| Path | Durable write | (a) Partial bytes at the final path | (b) Undone or harmless |
|---|---|---|---|
| `create_table` | `v1.metadata.json` via `write_commit_metadata` | No on local fs (temp file plus `hard_link`; the temp is removed on failure), no on memory (single lock), no on OpenDAL object stores (atomic PUT) | A failed exclusive write leaves nothing of ours; a conflict leaves the other writer's file, which must not be deleted |
| `create_table` | `version-hint.text` = `1` (the write that fails) | Yes on local fs (`fs::write` truncates, then writes) | Superseded in r6fix: not undone; the create succeeds at `v1` and the hint is warn-only, as after a commit (D-2); a partial hint is read as in the residue line above |
| `create_table` | cleanup deletes | n/a | Removed in r6fix |
| `update_table` | `vN.metadata.json` via the seam | Same as `v1` | Pre-existing on `main`; not in this diff |
| `update_table` | `version-hint.text` = `N` (`advance_version_hint`) | Yes on local fs | Not undone, by ruling (warn-only). Harmless for a Java 1.10.0 reader (residue line above) |
| `register_table` | none | — | — |
| `rename_table` | none | — | — |
| staged create / replace | not changed by this diff | — | Staged replace writes no hint (accepted residue) |

No other site inside create has the class.

## 8. Class sweep (round r5fix)

Class: a Hadoop-mode catalog operation that frees or re-binds a table name while files stay at a
location a later create derives, so the later create collides or reads foreign files. A later
Hadoop create writes `v1` exclusively (D-3), so it cannot overwrite; the question is what else it
meets.

| Operation | Leaves files where a later Hadoop create derives a location | What happens then |
|---|---|---|
| `create_table`, derived location | Writes `v1` and the hint at `<ns location or warehouse/ns>/<name>`; frees no name | A second create of the same location, under another name via an explicit location, fails `CatalogCommitConflicts` on `v1` |
| `create_table`, explicit location `L` | Same files at `L` | A later create deriving `L` fails `CatalogCommitConflicts`. If `L` already holds `vK` files but no `v1` (below), the create succeeds beside them |
| `rename_table` | Was: freed the old name, files stayed. Now refused (D-7) before any state change | Fixed |
| `drop_table` | Deletes only the current file; earlier `vK` and the hint stay | A table dropped at `v2+` blocks re-create with `CatalogCommitConflicts` on `v1`. A table dropped at `v1` leaves only a hint, which the next create overwrites. Fixed in F-MEMORY-HADOOP-NAMING-2 |
| purge (maintenance `DeleteReachableFiles`) | Deletes the current file, the `metadata_log` entries and the hint | A table with more history than `metadata_log` keeps (`write.metadata.previous-versions-max`) can keep `v1`, so re-create fails `CatalogCommitConflicts`. Residue, HMETA-DROP |
| `drop_namespace` | Removes the namespace and every table pointer in it; deletes no files | Re-creating the namespace and a table of the same name at the same derived location fails `CatalogCommitConflicts` on `v1` (for a table that reached `v1`). Not changed in this round; reported. Hadoop mode refuses since F-MEMORY-HADOOP-NAMING-2 D-5 (HMETA-DROPNS) |
| `register_table(T, P)` | Writes nothing; binds `T` to `P`, whose table location can be any `L` | If `P` is `vK` with `K > 1` and `L` has no `v1`, a later create deriving `L` succeeds. It writes `v1`, overwrites the registered table's hint with `1`, and shares `metadata/` with it. Its commits then fail `CatalogCommitConflicts` on reaching `vK`. That is silent overwrite of a foreign hint and a shared directory, reported and not fixed |
| staged create | Writes `00000-<uuid>` (Hadoop mode keeps uuid for staged create) | No deterministic name to collide; a later Hadoop create at the same location succeeds beside it |
| staged replace | Writes `v(N+1)` of the same table at its own location; frees no name | None |

## 9. Class sweep (round r6fix)

Class: an operation reports failure after its durable effect is already published, or undoes it
best-effort, so the caller's retry collides with the leftover.

| Path and step | Durable after | Can a later step still return `Err` | A retry then meets |
|---|---|---|---|
| Hadoop `create_table` (derived or explicit location): lock, `vacant_table_slot` | nothing | Yes, `TableAlreadyExists` / `ViewAlreadyExists` / `NoSuchNamespace`, before any write | nothing of ours |
| ... `write_commit_metadata(v1)` | `v1` published | The write itself fails only before publication (local fs: staging or link error; `AlreadyExists` is someone else's `v1`); a post-link temp-removal failure is `Ok` (D-8) | nothing of ours, or the other writer's `v1` (`CatalogCommitConflicts`, correct) |
| ... hint write | `v1` and maybe hint bytes | No, warn-only (D-2) | `TableAlreadyExists`: the name is registered |
| ... `VacantEntry::insert`, `cache_put`, `table_builder().build()` | pointer registered | No: `insert` and `cache_put` cannot fail; `build` fails only on a missing `file_io`, `metadata` or `identifier`, and all three are set | — |
| Hadoop `update_table` commit: `write_commit_metadata(vN)` outside the lock, then the lock, a CAS check, `commit_table_update`, then the hint | `vN` published before the final CAS | Yes: a CAS conflict after `vN` is published returns `CatalogCommitConflicts` and leaves an orphan `vN` | The next commit from the new base targets the same `vN`, finds it and fails loud; re-registering at the newest version recovers. Pre-existing on `main` (R167); residue, not in this diff |
| `register_table` | nothing written; pointer insert under the lock | The insert can fail (`TableAlreadyExists`); nothing durable to leak | — |
| staged create (`begin_create`) | `00000-<uuid>` written before the pointer publish | Yes, the publish can fail | A fresh uuid name, so no collision on retry; the orphan file is harmless. Not in this diff |
| staged replace (`begin_replace`, `publish_replace_table`) | `v(N+1)` published before the pointer CAS | Yes, a CAS conflict leaves an orphan `v(N+1)` | As for `update_table`. Pre-existing on `main`; residue |
| `stage_and_publish`: staging (`create_new` plus `write_all`) | nothing at `dest` | `Err` (Unexpected); the temp is removed best-effort | no `dest`; a stray temp at worst, never read |
| `stage_and_publish`: `hard_link` `AlreadyExists` | nothing of ours at `dest` | `Err` (`PreconditionFailed`) | the other writer's `dest` (correct) |
| `stage_and_publish`: `hard_link` other error | nothing at `dest` | `Err` (Unexpected) | no `dest` |
| `stage_and_publish`: post-link temp removal | `dest` published | No since r6fix (D-8) | — |

Residue, not fixed: an OpenDAL object-store conditional put (`if_not_exists`) can time out or lose
its response after the server committed. The caller sees `Err` with the file already published,
and a retry meets `PreconditionFailed` / `CatalogCommitConflicts`. The same applies to the
exists-then-write fallback backends. This belongs to the storage seam, not to this PR.
