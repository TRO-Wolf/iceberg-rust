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

# Ledger — F-ICE-RESIDUES-21B: run-21b fork-residue lane, five closures against Java 1.10.0

**Ledger id:** `F-ICE-RESIDUES-21B-2026-09-17`
**Branch:** `fix/ice-residues-21b` (cut off fork `main` = `8fb44a39`, F-OCC-SCOPED-1 / PR #291)
**Scope:** run 21b · FORK RESIDUE LANE, round 1 — five items, one commit each
**Matrix rows touched:** R94, R98, R110, R158, R167 (dated cell sentences only — no status flips)
**Model:** swe-2-high

| Item | Tag | Commit | Subject |
|---|---|---|---|
| 1 | F-HADOOP-VN-REPLACE-1 | `f91b1287` | staged replace on a `vN` base stages `vN+1` under exclusive create |
| 2 | F-CHERRYPICK-WAP-ORDER-1 | `83584125` | duplicate-WAP check runs at dispatch, before the already-picked check |
| 3 | F-UPDATE-SCHEMA-SAME-1 | `f0ea7e9b` | a structurally equal schema update commits nothing |
| 4 | Q-20b-C | `d3979d80` | `rollback_to_time` on a snapshotless table fails with Java's message |
| 5 | R-01 | `a7849bf5` | reuse the validated cherry-pick plan instead of planning twice |
| 6 | ledger | this commit | this file + `task/todo.md` entry + GAP_MATRIX dated sentences |

## Item 1 — F-HADOOP-VN-REPLACE-1: staged replace keeps `vN` naming under exclusive create

### Defect

`StagedTableTransaction::begin_replace` named the staged replacement with
`MetadataLocation::with_next_version_fresh_id()` — a fresh UUID — for EVERY base convention. On a
Hadoop-style `vN.metadata.json` base the staged file therefore never collided with a concurrent
winner's `v(N+1).metadata.json`: two stagers each wrote a distinct uuid file and both catalog
pointer CASes could succeed, leaving the loser visible only through a stale handle — the same
split-brain F-ICE-HADOOP-VN-1 (#286) closed for ordinary update commits, left open on the replace
path. Java `HadoopTableOperations.commit` writes a temp file and `renameToFinal` fails when `vN`
exists; it never overwrites, on replace or on append.

### Clauses

| Clause | Statement | Proven by |
|---|---|---|
| R1-1 | A staged replace on a `vN` base stages `v(N+1).metadata.json` (convention preserved) | `replace_stages_next_version_after_a_hadoop_named_pointer` |
| R1-2 | The staged write is exclusive-create: a second replace onto an existing `v(N+1)` fails retryable `CatalogCommitConflicts` and the winner's bytes are unchanged | `concurrent_replace_from_a_hadoop_pointer_fails_on_exclusive_create`; `hadoop_staged_replace_second_stager_fails_and_preserves_winner` |
| R1-3 | Hive/REST `<version>-<uuid>` bases keep a fresh uuid staged name (control) | `concurrent_replaces_from_a_uuid_pointer_stage_distinct_files`, `replace_stages_next_version_after_a_hive_named_pointer` |
| R1-4 | Unparsable bases / caller-supplied relocation restart at `00000-<uuid>` | `replace_restarts_versioning_when_base_pointer_does_not_parse`, `replace_restarts_versioning_under_a_different_caller_location` |

### Fix

`begin_replace` now derives the staged location with `with_next_version()` (convention-preserving)
and writes it through `TableMetadata::write_commit_metadata` — the same exclusive-create seam
`MemoryCatalog::update_table` took in F-ICE-HADOOP-VN-1. `with_next_version_fresh_id` had no
remaining caller and was deleted. `Transaction::apply_locally` already used this seam; only the
staged entry point diverged.

### Red-first evidence (unfixed tree, `f0ea7e9b`~ = `83584125`)

`cargo test -p iceberg --lib staged_table_version_tests` + `--test hadoop_version_commit`:
the Hadoop-pointer staging tests FAILED — the staged name was a fresh-uuid file rather than
`v(N+1).metadata.json`, and the second stager's write landed (no `CatalogCommitConflicts`, winner
bytes overwritten). The uuid-pointer control passed unchanged, pinning R1-3 against the fix.

## Item 2 — F-CHERRYPICK-WAP-ORDER-1: WAP-duplicate validation runs at dispatch

### Defect

`CherryPickAction::validate` checked already-ancestor / already-picked BEFORE
`validate_wap_publish`, so a cherry-pick of a snapshot whose `wap.id` was already published
surfaced "Cannot cherrypick snapshot %s: already an ancestor" instead of Java's
`Duplicate request to cherry pick wap id that was published already: <id>`.

### Java citation (1.10.0 source, `core/CherryPickOperation.java`)

`cherrypick(long)` runs `WapUtil.validateWapPublish` at BUILDER time inside the APPEND branch
(L80-83) and the replace-partitions branch (L108-114 — there AFTER the parent-ancestor check),
both before `apply()`'s fast-forward precedence (L193-204). `validate()` keeps the later order:
`validateNonAncestor` (L166) → `validateReplacedPartitions` (L167-168) → `validateWapPublish`
(L169). The fork's `plan()` is the dispatch-time analogue of `cherrypick()`; its `validate` is
Java's `validate`.

### Clauses

| Clause | Statement | Proven by |
|---|---|---|
| R2-1 | APPEND shape: WAP-dup decided before fast-forward/already-picked | `test_cherrypick_both_dedup_paths_wap_error_fires_first` |
| R2-2 | Replace-partitions: parent-ancestor check precedes the WAP check | `test_cherrypick_replace_partitions_rejects_non_ancestor_parent` |
| R2-3 | Non-WAP already-picked snapshot keeps the ancestor message | `test_cherrypick_double_publish_is_deduped_via_source_snapshot_id` (re-pinned with an empty `wap.id`) |
| R2-4 | Later validation order unchanged: nonAncestor → replacedPartitions → WAP | existing validate pins, 23/23 green |

### Fix

`plan()` now extracts operation + WAP id before the fast-forward test and calls
`validate_wap_publish` inside the APPEND and replace-partitions dispatch arms (RP arm: after the
parent-ancestor check), exactly Java's builder-time placement. `validate()`'s later ordering is
unchanged.

### Red-first evidence (unfixed tree)

`test_cherrypick_both_dedup_paths_wap_error_fires_first` FAILED: the combined
already-ancestor + duplicate-WAP pick surfaced
`Cannot cherrypick snapshot <id>: already an ancestor`; Java surfaces the duplicate-WAP
message. Post-fix: `Duplicate request to cherry pick wap id that was published already: wap-Z`,
23/23 module tests green. One pre-existing test pinned the stale order — corrected per the
oracle, and its non-WAP sibling re-pinned the ancestor message with an empty `wap.id`.

## Item 3 — F-UPDATE-SCHEMA-SAME-1: a structurally equal schema update commits nothing

### Defect (two layers)

`UpdateSchemaAction::commit` unconditionally emitted `AddSchema` + `SetCurrentSchema`, so a
no-op batch (e.g. a MOVE that leaves sibling order unchanged) minted a new schema id and wrote a
new metadata version. Java `SchemaUpdate.apply` replays ops onto the current schema,
`TableMetadata.Builder.addSchemaInternal` reuses an existing structurally equal schema, and
`ops.commit` early-returns on `base == metadata` — a no-op update commits NOTHING.

Root-cause find under red: `Schema::is_same_schema` compared `identifier_field_ids()` — an
iterator over a `HashSet` — element-by-element with `Iterator::eq`, which is ORDER-SENSITIVE. Two
equal sets with different iteration order compared false, so the V2 fixture
(`identifier-field-ids: [1, 2]`) never compared equal to its own rebuild. Fixed by comparing the
`HashSet`s directly — this also repairs `TableMetadataBuilder.reuse_or_create_new_schema_id` for
any schema with more than one identifier field.

### Clauses

| Clause | Statement | Proven by |
|---|---|---|
| R3-1 | A no-op MOVE emits no `AddSchema`, no `SetCurrentSchema`, no new schema id, no new metadata version, no pointer move | `noop_move_commits_nothing`, `test_no_op_rebuilds_equal_schema` (now asserts empty updates) |
| R3-2 | A real MOVE still commits and reorders | `real_move_still_commits` |
| R3-3 | ADD + no-op MOVE in one batch keeps the ADD and does not lose the MOVE | `add_plus_noop_move_keeps_the_add` |
| R3-4 | A no-op decided on a stale schema is re-validated against refreshed metadata | `stale_base_noop_move_revalidates_against_refreshed_schema` |
| R3-5 | Doc-only and default-only reapplies are no-ops | `same_doc_reapply_commits_nothing`, `same_default_reapply_commits_nothing` |
| R3-6 | Equality covers ids, names, types, nullability, order, doc, defaults, identifier-field ids as an unordered set | `is_same_schema` set comparison + the V2 two-identifier-field fixture |

### Fix

- `UpdateSchemaAction::commit` returns an empty `ActionCommit` when the evolved schema
  `is_same_schema` the current one.
- `Transaction::do_commit` skips `catalog.update_table` entirely when the accumulated updates are
  empty — the Java `ops.commit` `base == metadata` short-circuit (previously an empty commit still
  wrote a metadata file and bumped the pointer).
- `Schema::is_same_schema` compares identifier-field `HashSet`s directly.
- Three pre-existing unit tests that asserted the old emit-on-no-op contract were updated to the
  Java contract (zero updates).

### Red-first evidence (unfixed tree)

`cargo test -p iceberg --test update_schema_noop` (the file has since been folded into
`tests/hadoop_version_commit.rs::update_schema_noop` — see "Round 2" below): `noop_move_commits_nothing`,
`same_doc_reapply_commits_nothing`, `same_default_reapply_commits_nothing` FAILED — the commit
emitted `AddSchema`/`SetCurrentSchema`, wrote a `v2` metadata file, and moved the pointer. The
unit-level red surfaced a second layer: the empty-op rebuild compared UNEQUAL on the two-
identifier-field fixture until `is_same_schema` was fixed. After both changes: 6/6 integration
pins green, `transaction::` suite 677 passed / 0 failed / 1 ignored.

## Item 4 — Q-20b-C: `rollback_to_time` ancestry walk on `ManageSnapshots`

### Premise check against the oracle

The walk the brief asks to move into the fork ALREADY lives there:
`ManageSnapshotsAction` replays `SnapshotOp::RollbackToTime` through
`find_latest_ancestor_older_than` (`manage_snapshots.rs`), which walks ONLY the current snapshot's
parent chain, keeps the maximum qualifying timestamp, uses STRICT `<`, and fails with Java's exact
message. The brief's "`<=`" is a spec typo — Java 1.10.0 `SetSnapshotOperation.rollbackToTime` →
`findLatestAncestorOlderThan` uses `snapshot.timestampMillis() < timestampMillis`, and the fork's
`test_rollback_to_time_strict_less_than_skips_equal_timestamp` plus the interop strict-boundary
pin already prove it. All three required pins predate this round: ordinary
(`test_rollback_to_time_picks_newest_older_ancestor`), no-ancestor
(`test_rollback_to_time_before_first_snapshot_fails`), lateral-jump
(`test_rollback_to_time_never_selects_a_sibling`).

### Residual gap found and closed this round

Java resolves `ancestorIds(currentSnapshot)` — for a table with NO current snapshot the ancestry
is EMPTY, `findLatestAncestorOlderThan` returns null, and the failure is still
`Cannot roll back, no valid snapshot older than: %s`. The fork short-circuited first with a
different message, `Cannot roll back: table has no current snapshot`. The `RollbackToTime` arm now
treats a missing `main` ref as an empty ancestry and falls through to the Java message
(`SnapshotOp::RollbackTo`'s own missing-current message is untouched — Java checks the snapshot
id there, a separate pre-existing wording gap noted as residue).

### Red-first evidence (unfixed tree)

`cargo test -p iceberg --test interop_manage_snapshots`:

```text
test test_rollback_to_time_on_snapshotless_table_fails_with_java_message ... FAILED

---- test_rollback_to_time_on_snapshotless_table_fails_with_java_message stdout ----
unexpected message: Cannot roll back: table has no current snapshot
```

Post-fix: the snapshotless-table pin (driven end-to-end through a `MemoryCatalog` commit over
`TableMetadataV2ValidMinimal.json`, which carries no snapshots) fails with Java's message;
37/37 `manage_snapshots` unit tests + 5/5 interop tests green.

### Recorded semantic nuance (not a defect)

Java resolves the target at `rollbackToTime()` CALL time against the builder base, then
`apply()` re-validates the picked id is still a current ancestor on the refreshed base. The fork
resolves once, at commit-time replay, against the refreshed table — so in the narrow window where
ancestry changes between call and commit the fork picks on the NEWER ancestry rather than failing
on a stale pick. Both reject a lateral jump; the fork cannot surface Java's "not an ancestor of
the current table state" re-validation error because its pick is always on the ancestry it
commits against. Same outcome on every reachable single-writer shape; recorded, not changed.

## Item 5 — R-01: one `plan()` per cherry-pick base

### Defect

`CherryPickAction::validate` and `commit` each called `plan()`, and `plan` reads the picked
snapshot's manifest list via `picked_snapshot_changes` — one commit paid for the same manifest IO
twice.

### Why the calls are not unconditionally redundant (and what that implies)

`do_commit` runs every action's `validate` against the refreshed base, then each `commit` against
`current_table`, which `Self::apply` REBUILDS after every action (`with_metadata(Arc::new(..))`
unconditionally). For a single-action transaction both calls see the identical `TableMetadata`
object — the second plan is pure recompute. For a multi-action transaction, `commit` legitimately
plans against a different base than `validate` saw, so a plan cannot be reused blindly.

### Fix

`cached_plan` holds a `Mutex<Option<(TableMetadataRef, CherryPickPlan)>>` on the action. The
cached plan is reused only while `Arc::ptr_eq` proves the base metadata is the SAME OBJECT
`validate` planned against; any `Self::apply` rebuild or retry refresh produces a new Arc and
re-plans. `CherryPickPlan` derives `Clone`; the commit arm consumes a clone. The lock is never
held across `.await` and uses the repo's poison-tolerant
`lock().unwrap_or_else(|p| p.into_inner())` convention (`metrics/mod.rs`). Net effect: the
manifest list is read once per base instead of once per call — matching Java, where
`cherrypick()` computes `cherrypickChanges` once at builder time and `validate()` reuses it.

### Before/after evidence

`cargo test -p iceberg --lib transaction::cherry_pick` — BEFORE (tree at `83584125`): 23 passed,
0 failed. AFTER (`a7849bf5`): 23 passed, 0 failed. Identical results; no test text changed.
The stale-plan path is structurally unreachable (the key is object identity), so no new pin was
added; the reuse itself is exercised by every one of the 23 tests, each of which drives
`validate` then `commit` through `do_commit`.

## Gates and counts (2026-09-17)

| Gate | Result |
|---|---|
| `cargo test -p iceberg --lib transaction::` | 677 passed, 0 failed, 1 ignored (item 3 final run) |
| `cargo test -p iceberg --lib transaction::cherry_pick` | 23 passed before AND after R-01 |
| `cargo test -p iceberg --lib transaction::manage_snapshots` | 37 passed after item 4 |
| `cargo test -p iceberg --test interop_manage_snapshots` | 5 passed (incl. new snapshotless pin) |
| `cargo test -p iceberg --test hadoop_version_commit update_schema_noop` | 6 passed (as `update_schema_noop::*` inside `hadoop_version_commit.rs`) |
| `cargo test -p iceberg --test hadoop_version_commit` | 8 passed (incl. staged-replace pin) |
| `cargo clippy -p iceberg --all-targets -- -D warnings` | clean (one `collapsible_if` fixed in R-01) |
| `cargo fmt --all -- --check` | clean |
| Rule-1 comment grep over branch | 0 hits on `.rs`/`.toml`/`.yaml`/`.yml` (ASF header on the one new test file is the named exception) |
| `scripts/check_rust_file_size.py` | ceilings lowered to match: `cherry_pick.rs` 2132→2116→2115, `update_schema.rs` 3606→3595, `manage_snapshots.rs` 1254→1250 |

## Final gates (item 7, 2026-09-17 ~22:30 EDT)

| Gate | Result |
|---|---|
| `cargo test -p iceberg --lib` | 3762 passed, 0 failed, 8 ignored (381s) |
| `cargo test -p iceberg --test interop_cherrypick --test interop_wap_data --test interop_staged_wap` | 2 + 2 + 2 passed, 0 failed |
| `cargo test -p iceberg --test hadoop_version_commit --test update_schema_noop --test interop_manage_snapshots` | 8 + 6 + 5 passed, 0 failed (pre-move; post-move `hadoop_version_commit` alone runs 14/14) |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo clippy --all-targets --all-features --workspace -- -D warnings` (make-check flavor) | clean |
| `cargo fmt --all -- --check` | clean |
| `taplo check` / `cargo machete` | clean / no unused deps |
| `check_agent_artifacts.sh` / `check_matrix_anchors.sh` / `check_comment_blocks.sh` | all OK |
| `check_rust_file_size.py` | 495 files clean (98 legacy ceilings) |
| `typos` on the ledger/todo | clean |
| `git log origin/main..HEAD` | 6 commits, author `TRO-Wolf <64240326+TRO-Wolf@users.noreply.github.com>`, sole trailer `Authored-By: Devin SWE-2 (swe-2-high) <noreply@cognition.ai>` on each |
| Rule-1 grep over `origin/main` diff | 0 non-header hits; the 16 matched lines are the verbatim ASF header of the one new test file (named exception) |
| `git status --short` | clean |

## Round 2 (2026-09-17 late): mechanical gate rejection — file fold + subject fix

The orchestrator's mechanical comment-ban gate (`python3 /tmp/oc-worker/_lib/comment_ban.py
/tmp/kb-fork origin/main`) rejected round 1 on the 16 ASF-header lines of the NEW file
`crates/iceberg/tests/update_schema_noop.rs` — the named exception is not honored by that
gate. Resolution: the 6 pins moved VERBATIM into `crates/iceberg/tests/hadoop_version_commit.rs`
as `mod update_schema_noop { ... }` (reusing that file's `new_local_catalog`/`tempfile`
infrastructure; only `iceberg::spec::Literal` needed importing), the standalone file was
deleted, and every commit subject lost its ` (#292)` suffix (that PR belongs to another run's
branch; this branch has no PR). `cargo test -p iceberg --test hadoop_version_commit` now runs
14/14 — the same 8 + the same 6 under `update_schema_noop::*`. References updated in this
ledger, `task/todo.md`, `crates/iceberg/src/transaction/map.md`, and GAP_MATRIX R94.

## Round 3 (2026-09-18 early): the Grok reviews — logic NEEDS_REMEDIATION, perf PASS with two P2s

Round 2 was accepted and pushed as fork PR #293 (draft). Two Grok reviews read it:
`/tmp/oc-worker/kb-rv/reviews/fork-logic-report.md` (read the Java 1.10.0 sources directly and
mutation-tested every item) and `/tmp/oc-worker/kb-rv/reviews/fork-rustperf-report.md`. Verdicts:
logic NEEDS_REMEDIATION on four findings (L-001..L-004); Rust perf PASS with two P2
recommendations (R-01 Arc cache, R-02 empty-apply early return).

| Item | Tag | Commit | Subject |
|---|---|---|---|
| 7 | L-001 | `e2524fb5` | deterministic `is_same_schema` identifier-set pins + schema-id reuse |
| 8 | L-002 + L-004 | `e4ce33a2` | a staged replace publishes the staged version, not the next one |
| 9 | R-01/R-02 + L-003 | `3d143992` | `Arc` plan cache, empty-updates early return, cache-consume pin |
| 10 | ledger | this commit | round-3 evidence + rulings |

### L-001 (P1) — the `is_same_schema` set fix had a hollow pin

The logic critic reverted ONLY the `HashSet ==` in `Schema::is_same_schema` back to
`Iterator::eq` and `test_no_op_rebuilds_equal_schema` stayed green — `identifier_field_ids:
[1, 2]` iterates in the same order on both sides under most hash seeds, so the pin could not
fail. Java citation (the critic's reading of 1.10.0): `Schema.sameSchema` compares
`identifierFieldIds` as a `Set<Integer>` — `Set.equals` is order-insensitive.

Pins added in `spec/schema/utils.rs` (the only schema file with headroom under its ceiling):

- `test_is_same_schema_identifier_ids_compare_as_set` — two schemas with identical 64 fields
  and identifier ids inserted `1..=32` vs `32..=1` into independent `HashSet`s; 512 fresh pairs
  per process run; asserts `is_same_schema` is true.
- `test_reused_schema_id_for_equal_schema_with_reordered_identifier_ids` — 64 iterations
  through `TableMetadataBuilder::new_from_metadata(...).add_schema(...)` asserting
  `reuse_or_create_new_schema_id` keeps one schema (the id is reused, not appended).

Mutation evidence: with `is_same_schema` reverted to `Iterator::eq`, BOTH pins failed on 5/5
consecutive `cargo test -p iceberg --lib identifier_ids` process runs (each process re-seeds
`RandomState`; the failures panic inside the first loop iterations). With `HashSet ==`
restored, both pass — 2/2. Recorded runs: 5 process runs × (512 + 64 fresh-set iterations).

### L-002 (P2) — a staged Hadoop replace with pending files consumed two version slots

`begin_replace` staged `v(N+1)` under exclusive create, but `commit` → `materialize_pending` →
`Transaction::apply_locally` called `with_next_version` AGAIN off the staged location and
published `v(N+2)`, leaving `v(N+1)` a never-current orphan. Java `BaseTransaction` keeps the
replacement in memory and writes exactly one `nextVersion` at commit
(`BaseMetastoreTableOperations.commit` / `HadoopTableOperations.commit`).

Fix: the action loop is extracted into `run_actions_locally`; `apply_locally` keeps its
`with_next_version` contract for direct callers (the stale-`v3` collision pin depends on it),
and `materialize_pending` uses the new `pub(crate) apply_locally_in_place`, which OVERWRITES
the transaction's own staged file via `write_to`. Safe because the exclusive create at stage
time already proved this transaction owns that name — no other writer can hold it; a
concurrent writer that beat us to `v(N+1)` still fails our `begin_replace` retryable, and the
catalog CAS at publish is unchanged. The same path covers `begin_create` (the orphan
`00000-<uuid>` is likewise gone) and the `replace_write` overwrite arm.

Pins (`staged_table_tests.rs`): `hadoop_replace_with_files_publishes_only_next_version` (v2
pointer → `add_data_files` → commit publishes `v3`, the appended snapshot is present, no `v4`
exists) and `hadoop_replace_without_files_publishes_only_next_version` (same minus files).
The exclusive-create conflict pins
(`concurrent_replace_from_a_hadoop_pointer_fails_on_exclusive_create`,
`hadoop_staged_replace_second_stager_fails_and_preserves_winner`) pass unchanged. Mutation:
reverting `materialize_pending` to `apply_locally` fails the with-files pin exactly —
published `v4`, expected `v3` (`left: ".../v4.metadata.json", right: ".../v3.metadata.json"`).

### R-01 (P2) — the cached cherry-pick plan is `Arc<CherryPickPlan>`

`plan_cache` now holds `Arc<CherryPickPlan>`: a `validate`/`commit` cache hit is a refcount
bump, not a clone of every `DataFile` in the replay set. `commit` clones only the fields the
`SnapshotProducer` consumes, once, in the Replay arm; `CherryPickPlan` dropped its `Clone`
derive (the enum itself is no longer cloned). Key unchanged — `Arc::ptr_eq` on the base
`TableMetadataRef`, the entry retaining the base Arc so a rebuilt or refreshed base still
misses; the lock is never held across `.await`.

### R-02 (P2) — `Transaction::apply` returns the table unchanged on zero updates

`Self::apply` still runs every requirement's `check` against the current metadata and extends
`existing_updates`/`existing_requirements`, but skips `update_table_metadata` when the action's
update list is empty — previously it rebuilt `TableMetadata` unconditionally, bumping the
in-memory `last_updated_ms` even on a no-op action. This is the in-memory complement of the
`do_commit` empty-union skip (which suppresses the catalog write); requirement validation is
preserved at apply time, matching Java's `ops.commit` `base == metadata` early return which
fires before the catalog write while Java's requirement checks happen earlier on the action's
`validate`/`commit` path.
Pin: `staged_table_tests.rs::apply_locally_empty_updates_keep_the_metadata_arc` asserts
`Arc::ptr_eq` between the input and `apply_locally` output metadata for a no-op
`update_schema` move — red under revert (the unconditional rebuild produces a fresh Arc).

Before/after (identical, per the brief): `transaction::cherry_pick` ran 23 tests before and
the same 23 pass after, plus the new L-003 pin (24 total);
`cargo test -p iceberg --test hadoop_version_commit` ran 14/14 before and after (the schema
no-op suite unchanged).

### L-003 (P3) — plan-once pin, landed with no new production API

`cherry_pick.rs::tests::commit_uses_the_plan_cached_by_validate`: `validate` populates
`plan_cache`, the test poisons the entry (same base Arc, sentinel
`FastForward { picked_id: 7 }`), and asserts `commit` emits the poisoned `SetSnapshotRef`
shape — proof that `commit` consumes the cached plan rather than re-planning. The same-module
test reaches the private field directly, so no production seam was added; the ~20 lines were
funded at the file's exact 2115-line ceiling by compacting six private-helper doc blocks —
their Java citations are preserved in `transaction/map.md` (cherry_pick row) and this ledger.
Mutation: making `cached_plan` unconditionally re-plan fails the pin.

### L-004 (P3) — gzip-base staged naming

`staged_table_version_tests.rs::replace_stages_uncompressed_next_version_after_a_gzip_hadoop_pointer`:
a `v7.gz.metadata.json` base stages `v8.metadata.json` (uncompressed). `MetadataLocation`
already parsed both gzip spellings and `with_next_version` already emitted the uncompressed
next version (pinned at that level by `gzip_hadoop_next_version_is_uncompressed`); this pin
covers the `begin_replace` level, matching Java
`HadoopTableOperations.metadataFilePath` writing `v` + `version` + `.metadata.json`.

### Q2 — RULED FINAL by the Grok logic review

The critic confirmed the item-3 widening: Java `TableMetadata.commit` early-returns on
`base == metadata` for EVERY action — `BaseMetastoreTableOperations.commit` and
`HadoopTableOperations.commit` both open with `if (base == metadata) return;`. The fork's
`do_commit` empty-union skip is the same signal at the same seam, so the enumeration below
stands as the complete zero-update surface. R-02 adds the complementary in-memory seam: even
before `do_commit`, `Self::apply` no longer rebuilds metadata on an empty action.

## Round 4 (2026-09-18): the verification critic — one P2, V-001

Round 3 was accepted and pushed (#293 head `8c43a027`; fork CI 14/14 green). The verification
critic (`/tmp/oc-worker/kb-rv/reviews/fork-verify-report.md`) re-proved every item by mutation —
L-001 (10/10 red), L-002, L-003, R-01, R-02 all CLOSED — and found one new P2 introduced by the
L-002 fix.

| Item | Tag | Commit | Subject |
|---|---|---|---|
| 11 | V-001 pin (red-first) | `a3c06a8c` | a losing staged replace never overwrites a live vN |
| 12 | V-001 fix | `436e7bde` | a Hadoop staged replace writes its vN once, at commit, with final bytes |
| 13 | ledger | this commit | round-4 evidence + decision |

### V-001 (P2) — `apply_locally_in_place` `write_to` clobbered a live `v(N+1)` before the CAS

The L-002 fix made `commit` → `materialize_pending` → `apply_locally_in_place` write the final
metadata with `TableMetadata::write_to` — an OVERWRITE — onto the staged `v(N+1)` path, before
`publish_replace_table`'s pointer CAS. The exclusive create at `begin_replace` proved we owned
the NAME, but not that the file stayed uncommitted: a writer that landed `v(N+1)` as the live
pointer between `begin_replace` and `commit` had its bytes overwritten, after which our CAS
failed retryable — the live pointer then named a file holding this transaction's uncommitted
metadata (critic's reproduction: `v3_unchanged=false`, winner property gone, our appended
snapshot present). Java never overwrites a version file: `HadoopTableOperations.commit`
(L157-162) writes a temp file and `renameToFinal` (L368-377) refuses an existing destination
(`CommitFailedException("Version %d already exists")`); `BaseTransaction` keeps the replacement
in memory until one `ops.commit` (L516-526).

**Decision — option (a), the Java shape.** On a Hadoop-named staged target `begin_replace`
writes nothing: it probes `v(N+1)` and its gzip siblings (`ensure_staged_version_absent`) for
an early retryable `CatalogCommitConflicts` — preserving stage-time failure when the slot is
already taken — and otherwise holds the metadata in memory like `BaseTransaction`. `commit`
then writes the FINAL materialized bytes exactly once through `write_commit_metadata`'s
exclusive create (sibling pre-check + `write_new`) before the pointer CAS; a winner occupying
the slot makes that write fail retryable with the winner's bytes untouched. Uuid-named targets
are untouched: `write_commit_metadata` dispatches to `write_to` for them, so `begin_create`,
uuid/relocated bases, and the eager stage write all behave exactly as before. Option (b) —
materialize to a different slot — was rejected outright (reintroduces L-002's two-slot
publish); an early-reservation plus pointer-guard was rejected because any second write to a
`vN` name is an overwrite the rule forbids, and the stage-time file cannot hold final bytes
when pending files arrive later.

**Caller survey** (`grep -rn "begin_replace|StagedTableTransaction|apply_locally_in_place|
publish_replace_table" crates/`): the only production caller surface is `commit` →
`publish_create_table`/`publish_replace_table`. MemoryCatalog CASes the pointer under one
lock; S3 Tables does a pointer-only `GetTable` + version-token CAS to
`table.metadata_location()`; Glue's `replace_publish` read-validates the staged file before
`UpdateTable` — all three read the staged file (if at all) only at publish time, and under the
deferred design the file exists by then (written inside `materialize_pending`, before the
publish call). `apply_locally_in_place` is `pub(crate)` with `materialize_pending` as its
sole caller. No caller reads the staged file between `begin_replace` and `commit`.

**Red-first evidence:** `tests/hadoop_version_commit.rs::hadoop_staged_replace_never_overwrites_a_live_next_version`
(commit `a3c06a8c`) seeds a `v2` pointer, opens a staged replace, has a winner write `v3` and
CAS the pointer `v2→v3`, then commits the staged replace with pending files. On head
`8c43a027` it FAILED — the file held the loser's replace+append bytes (an `append` snapshot
with `added-data-files: 1`), not the winner's pristine metadata. Under the fix it passes:
`commit`'s exclusive create of `v3` fails retryable `CatalogCommitConflicts` before the CAS
and the winner's `v3` is byte-identical; no `v4` is written; the pointer stays `v3`.

**Pin mechanics adjusted to the deferred shape** (assertions preserved): in
`hadoop_staged_replace_second_stager_fails_and_preserves_winner` the `winner_bytes` read moved
after the winner's `commit` (the second `begin_replace` still fails retryable, now via the
probe); `concurrent_replace_from_a_hadoop_pointer_fails_on_exclusive_create` seeds the landed
`v4` explicitly through `write_commit_metadata` before asserting the second `begin_replace`
fails; `replace_stages_uncompressed_next_version_after_a_gzip_hadoop_pointer` now asserts the
staged target does NOT exist at begin (the deferred-write contract) — the
`v7.gz → v8.metadata.json` naming pin is unchanged.

### Q2 ruling addendum — every action that can emit ZERO updates, and what happens to each

`Transaction::do_commit` now returns `Ok(current_table)` without calling
`catalog.update_table` whenever the union of all action updates (`existing_updates`) is empty —
the Java `ops.commit` `base == metadata` early return at the transaction seam. The actions
that can reach that state:

| Action | Zero-update path | What the skip does |
|---|---|---|
| `UpdateSchemaAction` | The rebuilt schema `is_same_schema` the current one (item 3 fix) — `update_schema.rs` returns `ActionCommit::new(vec![], vec![])` | No `AddSchema`/`SetCurrentSchema`, no metadata file, no catalog-pointer move; `current-schema-id` and the schema list untouched |
| `ExpireSnapshotsAction` | (a) the table has zero snapshots — `expire_snapshots.rs` "Java `internalApply`: a table with no snapshots is a no-op"; (b) nothing expired and no refs to remove — the existing no-op-suppression return | Previously wrote a fresh metadata file bumping `last-updated-ms` and moved the pointer; now commits nothing. Both empty paths also carry zero requirements, so no guard is skipped |
| `ManageSnapshotsAction` | Per-ref net no-op suppression: every resolved ref equals its original (e.g., `rollback_to_time` resolving to the CURRENT snapshot, `set_current` to the already-current id, a create-then-remove within one action) leaves `updates` empty | Same skip: no file, no pointer move; the per-ref `RefSnapshotIdMatch` requirements are also not emitted (they are pushed only for refs that changed), so nothing is lost |
| `UpdateStatisticsAction` | `statistics_to_set` empty (action queued with no `set`/`remove` calls) | Emitted `ActionCommit::new(vec![], vec![])` — no requirements either; now writes nothing |
| `UpdatePartitionStatisticsAction` | `statistics_to_set` empty | Emits empty updates BUT a non-empty `UuidMatch` requirement; on the skip that requirement is never checked — which is also Java's order (`BaseTransaction` skips `ops.commit` on `base == metadata` BEFORE requirement validation, which lives inside `ops.commit`). Reachable only by queueing the action with zero stat entries; named here for the Critic to attack |
| Multi-action transaction | Every queued action emits zero updates (e.g., a no-op `update_schema` + a no-op `expire_snapshots` in one `Transaction`) | The union is empty → one skip; a non-empty sibling still commits normally (the skip is on the UNION, not per action) |

On the skip `do_commit` still returns the re-applied `current_table` — `Self::apply` over empty
updates produces identical metadata in memory; no `AddSnapshot` means no `CreateSnapshotEvent`,
no `latest_attempt_snapshot_ids` captured, and no reconciliation surface (there is nothing to
reconcile — no write was attempted).

## Open questions

1. **`task/map.md` does not exist.** RULED 2026-09-17 (orchestrator, round-2 brief): accepted as
   leaned — no `task/map.md`; the `task/todo.md` ACTIVE entry stands. (Original note: the brief
   asked the ledger be added to `task/map.md`; the repo's actual index convention for lane
   ledgers is an `## ACTIVE` section in `task/todo.md`, and map.md files live only in
   `.agents/skills`, the two task archives, `crates/sketches`, and per-directory code maps.)
2. **R110 adjacency — RULED FINAL 2026-09-18 (Grok logic review, round-3 brief):** the
   `do_commit` empty-updates skip (item 3) is confirmed as Java's `base == metadata` early
   return — `BaseMetastoreTableOperations.commit` / `HadoopTableOperations.commit` open with
   `if (base == metadata) return;` for every action. Keep it. The full enumeration of
   zero-update-capable actions and the consequence for each lives in "Q2 ruling addendum"
   above; R-02 (round 3) adds the complementary `Self::apply` early return.
3. `rollback_to` / `set_current` message wordings still diverge from Java's
   (`Cannot roll back to snapshot, not an ancestor of the current state: %s` /
   `Cannot roll back to unknown snapshot id: %s`) — pre-existing, outside this item's scope,
   noted for a future residue pass.
