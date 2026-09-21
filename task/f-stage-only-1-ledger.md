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

# F-STAGE-ONLY-1 — staged (WAP) commits on every write action, and the publish primitive

RePark parity inventory IPI-05 (oracle brief `IPI-05.md`, Spark cells P-CHERRYPICK-WAP,
P-PUBLISH-CHANGES, P-PUBLISH-CHANGES-MISSING-ERR, W-INSERT-WAP-ID). Measured Java/Spark
contract:

- `spark.wap.id` set ⇒ Java calls `SnapshotProducer.stageOnly()`: the write commits
  `AddSnapshot` alone — the snapshot lands in `snapshots` with summary `wap.id=<id>`
  but NO ref moves (`current-snapshot-id`, `main`, snapshot log unchanged).
- `cherrypick_snapshot(table, id)` publishes the staged snapshot onto `main`:
  fast-forward when the staged parent is the current head (same snapshot, keeps
  `wap.id`, no new summary keys), else replay into a NEW snapshot carrying
  `source-snapshot-id` and `published-wap-id`.
- `publish_changes(table, wap_id)` (Java `PublishChangesProcedure`) scans ALL table
  snapshots for summary `wap.id == wap_id`: 0 matches →
  `ValidationException: Cannot apply unknown WAP ID '<id>'`; >1 →
  `Cannot apply non-unique WAP ID. Found multiple snapshots with WAP ID '<id>'`;
  exactly 1 → cherry-picks that snapshot. An already-published id fails inside
  cherry-pick's `WapUtil.validateWapPublish` with `DuplicateWAPCommitException`:
  `Duplicate request to cherry pick wap id that was published already: <id>`
  (the check walks the CURRENT ancestry of `main` and matches both `wap.id` and
  `published-wap-id`).

## Public snapshot-producing action survey (pre-change state)

| Action (Transaction factory) | Produces snapshots | Could stage before | Now |
|---|---|---|---|
| `fast_append` → `FastAppendAction` | yes (`Append`) | YES — `stage_only()` + producer flag | unchanged |
| `merge_append` → `MergeAppendAction` | yes (`Append`) | no — wired this unit | `stage_only()` added |
| `overwrite_files` → `OverwriteFilesAction` | yes (`Overwrite`) | no — wired this unit | `stage_only()` added |
| `replace_partitions` → `ReplacePartitionsAction` | yes (`Overwrite` + `replace-partitions=true`) | no — wired this unit | `stage_only()` added |
| `row_delta` → `RowDeltaAction` | yes (`Overwrite`/`Delete`-bearing) | no — wired this unit | `stage_only()` added |
| `delete_files` → `DeleteFilesAction` | yes (`Delete`) | YES — `stage_only()` already wired | unchanged |
| `rewrite_files` → `RewriteFilesAction` | yes (`Replace`) | no — maintenance producer, not a RePark row-write path | out of enumerated scope |
| `rewrite_manifests` → `RewriteManifestsAction` | yes (`Replace`) | no — same maintenance class | out of enumerated scope |
| `cherry_pick` → `CherryPickAction` | yes (publish replay) | n/a — cherry-pick IS the publish; staging it is meaningless | n/a |
| `manage_snapshots` / ref ops | no new snapshot | n/a | n/a |

`rewrite_files` / `rewrite_manifests` produce snapshots through the same producer, so
`stage_only` could be added later by the same pattern; the brief enumerates the five
RePark write paths, and Java Spark only calls `stageOnly()` on the row-write actions
during a `spark.wap.id` write (procedure commits publish normally).

## Cherry-pick survey vs Java `CherryPickOperation`

Already implemented and pinned in `transaction/cherry_pick.rs` (read, no changes
needed): fast-forward precedence (staged parent == current head, or both null);
APPEND replay; `replace-partitions=true` OVERWRITE replay (removed paths must be
live, `failMissingDeletePaths`); `source-snapshot-id` always on replay,
`published-wap-id` iff the staged snapshot had non-empty `wap.id`; duplicate checks —
`validateNonAncestor`/`isCurrentAncestor`, `lookupAncestorBySourceSnapshot`,
`is_wap_id_published` matching BOTH `wap.id` and `published-wap-id` in current
ancestry with the verbatim `DuplicateWAPCommitException` message; rejection of
staged `Delete` ops and non-replace overwrites on the replay path; multi-spec
replay. The FF-vs-replay WAP-dedup ordering was bytecode-verified in
F-CHERRYPICK-WAP-ORDER-1. No cherry-pick gap found.

## Design

- `stage_only: bool` + ctor init + `.with_stage_only(self.stage_only)` on the four
  unwired actions, the same producer flag `FastAppendAction`/`DeleteFilesAction`
  already drive. Field is `pub(crate)` (the `target_branch` precedent) so the four
  `stage_only()` builder methods live together in `publish_changes.rs` — the
  `to_branch.rs` pattern of one home per cross-cutting producer knob. The three
  ceiling-capped files therefore grow only 3 lines each, reclaimed by item-boundary
  blank-line compaction (the f-exact-count-1 approach); `merge_append.rs` has headroom.
- `staged_snapshot_for_wap_id(&TableMetadata, &str) -> Result<SnapshotRef>` —
  `PublishChangesProcedure`'s lookup ONLY (orchestrator ruling Q-26d-2, closing
  L-02): a full `metadata.snapshots()` scan, the non-unique and unknown messages
  verbatim, and `Ok(snapshot)` on a unique match — including an already-published
  one. The procedure's lambda never calls `WapUtil`; `DuplicateWAPCommitException`
  lives in `CherryPickAction::validate_wap_publish` on the append and
  replace-partitions replay arms only, so a second publish of an FF-published
  Delete or non-replace Overwrite gets CherryPickOperation's FF-only message, as
  Java does. `cherry_pick.rs::is_wap_id_published` stays `pub(crate)` — one home
  for the dedup rule, matching both `wap.id` and `published-wap-id` on current
  ancestry like `WapUtil.isWapIdPublished` (offsets 53-71).
- `PublishChangesAction` + `Transaction::publish_changes(wap_id)` — resolves the
  wap id ONCE per attempt into a `PublishBinding { snapshot_id, cherry_pick }`
  held in a `Mutex` on the action (Java binds `snapshotId` once in
  `PublishChangesProcedure.lambda$call$0`, offsets 99-105, then
  `manageSnapshots().cherrypick(snapshotId).commit()`; `SnapshotProducer.commit`
  retries `apply()` with that bound id and never re-scans `snapshots()`). Every
  `validate`/`commit`/retry re-checks that the bound snapshot still exists and
  still carries `wap.id == wap_id`: a second same-id staged snapshot appearing
  mid-attempt still publishes the bound id; expire-and-restage fails
  `Cannot cherry-pick unknown snapshot ID: <bound id>` rather than publishing
  the replacement. `main`-only: Java's procedure publishes to `main`, so
  `target_ref` keeps the default.
- `mod publish_changes`, `mod stage_only_tests` and `mod
  stage_only_publish_tests` are `#[path]`-wired from `action.rs` — `mod.rs` sits
  on its 1936-line legacy ceiling; the escape valve is the same one
  `occ_scoped_tests.rs` uses. The publish pins split into their own file under
  the 1000-line ceiling, sharing fixtures through `pub(crate)` helpers.

## Clauses

| id | behaviour | verdict | pins |
|---|---|---|---|
| C-001 | `spark.wap.id` ⇒ staged commit emits `AddSnapshot` alone: `current-snapshot-id`, `main`, snapshot log and every other ref unmoved; main's live read unchanged | PROVEN | `merge_append_stage_only_adds_snapshot_without_moving_main`, `overwrite_files_stage_only_adds_snapshot_without_moving_main`, `replace_partitions_stage_only_adds_snapshot_without_moving_main`, `row_delta_stage_only_adds_snapshot_without_moving_main`, `delete_files_stage_only_adds_snapshot_without_moving_main` (stage_only_tests.rs) |
| C-002 | The staged snapshot is readable by id, carries `wap.id`, the action's Operation, `parent_snapshot_id` == base head, and the staged live file set (overwrite removes the base file; replace-partitions carries `replace-partitions=true` and drops the replaced file; row delta's staged snapshot contains the delete file) | PROVEN | same five stage pins (stage_only_tests.rs) |
| C-003 | A staged snapshot consumes a sequence number at stage time: `staged.sequence_number() == base last_sequence_number + 1`, `metadata.last_sequence_number` advances, `current-snapshot-id` unchanged (Java `TableMetadata.Builder.addSnapshot` on the stageOnly path) | PROVEN | `stage_only_consumes_a_sequence_number_at_stage_time` |
| C-004 | Staged commits still run conflict validation against the refreshed head — neither `OverwriteFilesAction::validate` nor `RowDeltaAction::validate` reads `stage_only`, and Java has no stageOnly branch in validation | PROVEN | `staged_overwrite_conflict_validation_still_rejects_a_concurrent_append`, `staged_row_delta_conflict_validation_still_rejects_a_concurrent_delete` |
| C-005 | `staged_snapshot_for_wap_id` is the procedure lookup only: full `metadata.snapshots()` scan; 0 → `Cannot apply unknown WAP ID '<id>'`; >1 → `Cannot apply non-unique WAP ID. Found multiple snapshots with WAP ID '<id>'`; a unique already-published match returns `Ok` (Q-26d-2) | PROVEN | `staged_snapshot_for_wap_id_finds_the_staged_snapshot`, `staged_snapshot_for_wap_id_unknown_id_has_java_message`, `staged_snapshot_for_wap_id_non_unique_has_java_message`, `staged_snapshot_for_wap_id_already_published_returns_the_staged_snapshot` |
| C-006 | `publish_changes` binds the wap id once per attempt and reuses the bound `CherryPickAction` for validate/commit/retries; a second same-id snapshot appearing mid-attempt still publishes the bound id; expire-and-restage fails `Cannot cherry-pick unknown snapshot ID: <id>` | PROVEN | `publish_changes_binds_the_wap_id_once_per_attempt`, `publish_changes_bound_snapshot_expired_fails_unknown_snapshot_id` |
| C-007 | FF publish (staged parent == head) moves `main` to the staged snapshot verbatim, keeping `wap.id`, adding no new summary keys | PROVEN | `publish_changes_fast_forwards_the_staged_snapshot` |
| C-008 | Replay publish (head advanced) creates a new snapshot stamped `source-snapshot-id` + `published-wap-id` carrying the staged live set | PROVEN | `publish_changes_replays_and_sets_source_and_published_wap_id` |
| C-009 | The duplicate-WAP check lives in `CherryPickAction::validate_wap_publish`, matching BOTH `wap.id` and `published-wap-id` on current ancestry (`WapUtil.isWapIdPublished` offsets 53-71): second publish of a published Append → `Duplicate request to cherry pick wap id that was published already: <id>`; of a published Delete → `Cannot cherry-pick snapshot <id>: not append, dynamic overwrite, or fast-forward`; a replay-published id is deduped through `published-wap-id` | PROVEN | `publish_changes_twice_has_java_message`, `publish_changes_twice_of_a_published_delete_has_ff_only_message`, `publish_changes_replay_publish_blocks_the_wap_id_via_published_wap_id` |
| C-010 | V3 publish assigns a FRESH `first_row_id`: `published.first_row_id() == metadata.next_row_id()` captured immediately before `publish_changes`, and `!= staged.first_row_id()` | PROVEN | `publish_changes_replay_assigns_fresh_row_ids_on_v3` |
| C-011 | Per-action publish coverage: merge append FF + replay; replace-partitions replay (old partition file gone, new present, `published-wap-id` on the replay); overwrite FF + Java-matching replay refusal; row delta FF; delete FF | PROVEN | `publish_changes_merge_append_fast_forwards`, `publish_changes_merge_append_replays`, `publish_changes_replace_partitions_replays`, `publish_changes_overwrite_files_fast_forwards`, `publish_changes_overwrite_files_replay_refuses_with_java_message`, `publish_changes_row_delta_fast_forwards`, `publish_changes_delete_files_fast_forwards` |
| C-012 | Named divergence (Q-26d-3): a data-only `row_delta().stage_only()` records `Operation::Overwrite` (the 1.10.0 freeze on `RowDeltaOperation::operation`), so `publish_changes` after a concurrent append fails with the FF-only message — Java 1.11.0 records `append` and replays | DIVERGENCE-DECLARED | `publish_changes_data_only_row_delta_records_overwrite_and_refuses_replay` |

## Pins

Stage + lookup pins live in `stage_only_tests.rs`; publish pins live in
`stage_only_publish_tests.rs`, both `#[path]`-wired from `action.rs` and sharing
`pub(crate)` fixtures. The clause table above names every pin.

## 7. Mutation evidence

### Critic-run mutations (rv-verify.json, round-1 head — a pre-rebase branch tip, no longer resolvable; no post-rebase commit carries that tree state)

| # | mutation | result |
|---|---|---|
| VM-1 | `merge_append.rs` `with_stage_only(self.stage_only)` → `with_stage_only(false)` | REDDENED — `merge_append_stage_only_adds_snapshot_without_moving_main` |
| VM-2 | `overwrite_files.rs` same | REDDENED — `overwrite_files_stage_only_adds_snapshot_without_moving_main` |
| VM-3 | `replace_partitions.rs` same | REDDENED — `replace_partitions_stage_only_adds_snapshot_without_moving_main` |
| VM-4 | `row_delta.rs` same | REDDENED — `row_delta_stage_only_adds_snapshot_without_moving_main` |
| VM-5 | `staged_snapshot_for_wap_id`: swap 0-match and non-unique messages | REDDENED — unknown/non-unique/publish-unknown pins |
| VM-6 | `staged_snapshot_for_wap_id`: ancestry scan instead of `metadata.snapshots()` | REDDENED — 6 pins |
| VM-7 | `staged_snapshot_for_wap_id`: >1 match returns first instead of erroring | REDDENED — non-unique pin |
| VM-8 | `publish_changes`: commit resolves against metadata cloned at validate (stale) | UNCAUGHT → closed by C-006 binding (L-01/V-02/V-06) |
| VM-9 | cherry-pick replay: drop `source-snapshot-id` + `published-wap-id` stamping | REDDENED — 11 pins |
| VM-10 | `is_wap_id_published`: match only `wap.id`, drop `published-wap-id` arm | REDDENED on cherry_pick pins only; all 14 stage-only pins stayed green → V-01, closed by `publish_changes_replay_publish_blocks_the_wap_id_via_published_wap_id` |
| VM-11 | `staged_snapshot_for_wap_id` published-check: local `wap.id`-only ancestry walk | UNCAUGHT → closed by the same V-01 pin |

### Round-2 actor mutations

| # | mutation | result |
|---|---|---|
| AM-1 | `bound_cherry_pick` re-resolves the wap id unconditionally on every call (no binding) | REDDENED — `publish_changes_binds_the_wap_id_once_per_attempt` (second same-id snapshot made it fail non-unique instead of publishing the bound id) and `publish_changes_bound_snapshot_expired_fails_unknown_snapshot_id` (replacement published instead of unknown-snapshot-id) |
| AM-2 | `replace_partitions.rs`: drop the automatic `replace-partitions=true` summary insertion | REDDENED — the staged-marker assertion in `replace_partitions_stage_only_adds_snapshot_without_moving_main` and the replay-shape assertion in `publish_changes_replace_partitions_replays` |
| AM-3 | `OverwriteFilesAction::validate` early-returns `Ok(())` when `stage_only` | REDDENED — `staged_overwrite_conflict_validation_still_rejects_a_concurrent_append` (staged commit went green instead of `DataInvalid`) |
| AM-4 | `RowDeltaAction::validate` early-returns `Ok(())` when `stage_only` | REDDENED — `staged_row_delta_conflict_validation_still_rejects_a_concurrent_delete` (same green-instead-of-reject shape) |
| AM-5 | `is_wap_id_published` matches only `wap.id` (V-01's published-wap-id-arm mutation, re-run against the new pin) | REDDENED — `publish_changes_replay_publish_blocks_the_wap_id_via_published_wap_id` fell through to the ancestor-dedup message instead of the duplicate-WAP text |

Round-1 actor mutations (kept for the record): `merge_append.rs` stage flag ignored
→ reddened its stage pin; the lookup-level duplicate check bypassed → reddened the
then-existing already-published pin. That check has since moved out of the lookup
per Q-26d-2; the duplicate-WAP text is now pinned at `publish_changes` of a
published Append (`publish_changes_twice_has_java_message`).

## Findings — round 2

| id | critic | disposition | where |
|---|---|---|---|
| V-01 | verify | CLOSED — replay-publish dedup pin added: stage append, advance main, `publish_changes` stamps `published-wap-id`, then `publish_changes` of the same id gives the duplicate-WAP text; AM-5 proves the `published-wap-id` arm is load-bearing | `publish_changes_replay_publish_blocks_the_wap_id_via_published_wap_id` (stage_only_publish_tests.rs); pins: f-stage-only-1/C-009 |
| V-02 | verify | CLOSED — the stale-metadata resolve is moot: the wap id now binds once per attempt into `PublishBinding`, and `commit` uses the bound `CherryPickAction` against the refreshed table | `PublishChangesAction::bound_cherry_pick` (publish_changes.rs); pins: f-stage-only-1/C-006 |
| V-03 | verify | CLOSED — every stage pin now asserts Operation, `parent_snapshot_id` == base head, and the staged live file set (overwrite drops the base file; replace-partitions marker + replaced file gone; row delta carries the delete file); per-action publish coverage added; the round-1 ledger's overclaim is corrected by the clause table | five stage pins (stage_only_tests.rs), seven per-action publish pins (stage_only_publish_tests.rs); pins: f-stage-only-1/C-002, C-011 |
| V-04 | verify | CLOSED — staged overwrite + `validate_no_conflicting_data` rejects a concurrent main append; staged row delta + `validate_no_conflicting_delete_files` rejects a concurrent delete commit; AM-3/AM-4 prove skipping validation under `stage_only` reddens | stage_only_tests.rs; pins: f-stage-only-1/C-004 |
| V-05 | verify | CLOSED — `staged.sequence_number() == base last_sequence_number + 1`, `last_sequence_number` advances while `current-snapshot-id` is unchanged; v3 publish `first_row_id == next_row_id` captured immediately before `publish_changes` and `!= staged.first_row_id` | `stage_only_consumes_a_sequence_number_at_stage_time`, `publish_changes_replay_assigns_fresh_row_ids_on_v3`; pins: f-stage-only-1/C-003, C-010 |
| V-06 | verify | CLOSED — one `CherryPickAction` per attempt: bound in a `Mutex` on `PublishChangesAction`, re-checked for existence and `wap.id` on every use; AM-1 proves re-resolving reddens both pins | publish_changes.rs `PublishBinding`/`bound_cherry_pick`; pins: f-stage-only-1/C-006 |
| L-01 | logic | CLOSED — same fix as V-06: Java binds `snapshotId` once (`lambda$call$0` offsets 99-105); the fork now mirrors that binding and never re-scans `snapshots()` mid-attempt | pins: f-stage-only-1/C-006 |
| L-02 | logic | CLOSED per ruling Q-26d-2 — `staged_snapshot_for_wap_id` is the procedure lookup only; the duplicate check stays in `CherryPickAction::validate_wap_publish` on the append and replace-partitions arms; published-Delete second publish gets the FF-only message | `staged_snapshot_for_wap_id_already_published_returns_the_staged_snapshot`, `publish_changes_twice_of_a_published_delete_has_ff_only_message`; pins: f-stage-only-1/C-005, C-009 |
| L-03 | logic | CLOSED — same pins as V-03 plus the exact `next_row_id` equality oracle | pins: f-stage-only-1/C-002, C-010, C-011 |
| L-04 | logic | DIVERGENCE-DECLARED per ruling Q-26d-3 — see below | `publish_changes_data_only_row_delta_records_overwrite_and_refuses_replay`; pins: f-stage-only-1/C-012 |

## Named divergences

- **Q-26d-3 — `RowDeltaOperation::operation` 1.10.0 freeze (declared 2026-09-20).**
  Java 1.11.0 `BaseRowDelta.operation()` returns `append` when the delta adds data
  files and no delete files (critic bytecode citation: offsets 0-23 `ldc #83
  'append'`; `addsDeleteFiles && !addsDataFiles → 'delete'` offsets 24-40; else
  `overwrite` 41-43). The fork's 1.10.0-frozen `operation()` has no leading append
  branch, so a data-only `row_delta().stage_only()` records `Operation::Overwrite`
  where Java 1.11.0 records `append`. Consequence pinned: `publish_changes` of
  that staged snapshot after a concurrent append fails with CherryPickOperation's
  `Cannot cherry-pick snapshot <id>: not append, dynamic overwrite, or
  fast-forward` where Java would replay it. Changing `operation()` is its own
  unit — fork ask F-ROWDELTA-OP-1 — because it moves the `operation` value of
  every row-delta snapshot, which the parity inventory records.

## Critic questions — recorded

- rv-verify Q-01: `gh pr view 330 -R TRO-Wolf/iceberg-rust` was denied by the local
  permission policy; the critic used the ledger and the `origin/main..HEAD` diff,
  not the GitHub PR body. Any contract living only in the PR body was unchecked.
- rv-verify Q-02: no live Spark write was run against the 1.11.0 runtime jar;
  `spark.wap.id → stageOnly` was measured from `SparkWrite` bytecode (offsets
  115-149: `wapEnabled && wapId != null` ⇒ `set("wap.id")` +
  `SnapshotUpdate.stageOnly()`), which applies to any `SnapshotUpdate`.
- rv-logic Q-01 → orchestrator ruling Q-26d-3 (above): the 1.10.0 freeze stands for
  this PR; the divergence is pinned, not fixed.
- rv-logic Q-02 → orchestrator ruling Q-26d-2: `staged_snapshot_for_wap_id` is the
  procedure lookup only; implemented.

COVERAGE_ATTESTATION: 12 clauses (C-001..C-012): 11 PROVEN, 1
DIVERGENCE-DECLARED (C-012, pinned, named and dated). Findings V-01..V-06 CLOSED,
L-01..L-03 CLOSED, L-04 DIVERGENCE-DECLARED under ruling Q-26d-3. 11 critic +
5 actor mutations recorded above; every mutation run against product code either
reddened a named pin or is mapped to a closed finding. The non-staged write path
is untouched (`stage_only: false` defaults pass through unchanged).

## Gates

Round 2 (re-measured 2026-09-21 after the rebase onto fork main `21c27eb5`):

- `cargo fmt --all -- --check` — clean
- `CARGO_BUILD_JOBS=6 cargo clippy -q -p iceberg --all-targets -- -D warnings` — clean
- `CARGO_BUILD_JOBS=6 RUST_TEST_THREADS=6 cargo test -q -p iceberg --lib transaction`
  — 818 passed; 0 failed; 1 ignored (29 stage-only pins inside)
- `python3 scripts/check_rust_file_size.py` — 634 files clean (90 legacy ceilings)
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/xf-wap origin/main HEAD` —
  `comment-ban hits=0`
- `git log --format='%B' origin/main..HEAD | grep -i co-authored` — no matches

## Base integration

2026-09-21: fork main advanced to
`b466ad6003bc61a3a9d692326da28d67f2ce105e` (F-RDF-DETERMINISTIC-ORDER-1), which
touches only `crates/iceberg/src/maintenance/`. It is merged into this branch so
the branch contains current fork main; the merge carries no changes of its own
and leaves every file this PR owns byte-identical.

Round 1 (for the record):

- `cargo fmt --all` — clean
- `cargo clippy -p iceberg --all-targets -- -D warnings` — clean
- `cargo test -p iceberg --lib stage_only_tests` — 14 passed; 0 failed
- `bash scripts/check_rust_file_size.sh` — 593 files clean (90 legacy ceilings);
  ceilings LOWERED for `replace_partitions.rs` (2783 → 2782) and `mod.rs`
  (1937 → 1936) after blank-line reclamation
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/qb-fork2 origin/main` —
  `comment-ban hits=0`

## SLICE F-STAGE-ONLY-2 — thread stage_only and snapshot_properties from IcebergTableProvider to every DML commit (2026-09-21)

Branch `feat/f-stage-only-2` off fork main
`3f5a9289cffecd82c9ed2ea53187d9b6eb8f3a6d` (the squashed F-STAGE-ONLY-1
commit); it now also contains fork main
`97f9b8a32226e03e21bec053ce39f3ec9cb10771` through merge
`0c2386b0d90aa7205722eccbc1b4e2f14ed0a227`, which changes one slice file
(`table/mod.rs`) only inside `metadata_table()` (the F-META-COLS
`snapshot_id` parameter) and leaves the seam below byte-identical.
Model: muse-spark-1.3-contributor. Round 1: `IcebergTableProvider` had
`with_commit_branch` but no staged-commit knob, so a DataFusion INSERT always
moved the table ref and RePark could not do write-audit-publish. Round 1
adds the exact analogue `with_stage_only(bool)`, threads it through
`IcebergCommitExec` the way `commit_branch` is threaded, and applies the
public no-arg `.stage_only()` to the Append and Overwrite actions. Round 2
adds the second seam `with_snapshot_properties(HashMap<String, String>)`
on the same path: caller snapshot properties merge into the exec-stamped
map on both arms. Round 3 pins `refreshed()` carry-through for all four
provider knobs. Round 4 pins `from_planning_load` defaults for all four
commit knobs.

### Seam

Provider (`table/mod.rs`): `stage_only: bool` field beside `commit_branch`,
default false in `try_new`, clone-through in `refreshed`, `with_stage_only`
builder beside `with_commit_branch`, hand-on in `insert_into`. Both
builders carry no doc comment (owner comment ban; the `with_output_spec_id`
precedent; this crate has no `missing_docs` deny). Exec (`commit.rs`): field,
default false, `pub(crate) with_stage_only`, hand-on in `with_new_children`,
local copy in `execute`, conditional `.stage_only()` in the
`InsertOp::Append` and `InsertOp::Overwrite` arms. `InsertOp::Replace` stays
untouched. `loaded.rs::from_planning_load` takes the default-false line
although the brief did not list the file: the struct literal does not compile
without it. `delete_from` / `update` are deliberately NOT threaded:
`IcebergDeleteExec::new` and `IcebergUpdateExec::new` take `commit_branch` as
a positional constructor argument, so handing them `stage_only` would mean
editing `delete.rs` / `update.rs`, which the brief's file list forbids. That
is observed out-of-scope, not a silent drop.

Round 2 adds the second seam `with_snapshot_properties` beside
`with_stage_only`: provider field `snapshot_properties:
HashMap<String, String>`, default empty map in `try_new` and in
`loaded.rs::from_planning_load`, clone-through in `refreshed()` (now
pinned, round 3), hand-on through `insert_into` into
`IcebergCommitExec`, and through `with_new_children`. In BOTH the
`InsertOp::Append` and `InsertOp::Overwrite` arms the caller map is
MERGED INTO the exec-built map (never replaces it): the exec first
stamps `{OPERATION_ID_PROP: uuid}`, then inserts every caller entry
except `OPERATION_ID_PROP`, so the exec-stamped operation id WINS over a
caller-supplied one. `InsertOp::Replace` errors `NotImplemented`, and
`delete_from` / `update` stay unthreaded as in round 1.

Round 3 pins `refreshed()`: it carries `commit_branch`, `stage_only`,
`snapshot_properties` and `output_spec_id` from `self`, and drops
`planning_table` so scans on the fresh instance load current state
instead of reusing the planning snapshot.

Round 4 pins the third constructor, `from_planning_load`, with no
production line changes: it already carried the same four defaults
(`commit_branch: None`, `stage_only: false`, an empty
`snapshot_properties`, `output_spec_id: None`) with `planning_table:
Some`, and the new pin asserts exactly that with no builder calls in
between.

### Files

Final line counts (`wc -l` at the round-4 class-A commit
`a22ee54cd9a5fc62d6906d090ec503ce610204e3`):
`table/mod.rs` 357, `table/loaded.rs` 63, `table/tests.rs` 971 (round 3
added the two `refreshed` pins and round 4 the `from_planning_load` pin
in place, under the 1000-line ceiling),
`commit.rs` 484, `commit_tests.rs` 946,
`commit_stage_only_tests.rs` 197, `commit_snapshot_properties_tests.rs`
330 (new in round 2), `tests/stage_only.rs` 610, `tests/map.md` 106.
The unit pins live in their own `#[path]`-wired modules sharing
`pub(crate)` fixtures from `commit_tests.rs` (the F-STAGE-ONLY-1
pattern): that file sits at 946 of its 1000-line ceiling and neither pin
set fits inside it.

### Pins

Round-1 unit (`commit_stage_only_tests.rs`, 4 pins):

- `test_provider_stage_only_defaults_false_and_threads` — default provider
  plans an exec with `stage_only == false`; `with_stage_only(true)` plans
  `true` (downcast through `insert_into`)
- `test_append_stage_only_adds_snapshot_without_moving_current`
- `test_overwrite_stage_only_adds_snapshot_without_moving_current`
- `test_stage_only_with_commit_branch_leaves_branch_unmoved`

Round-1 end to end (`tests/stage_only.rs`, 6 pins, SQL through `SessionContext`):

- `insert_stage_only_adds_snapshot_without_moving_main` (staged op/parent
  pinned, main scan still `[1]`)
- `insert_overwrite_stage_only_adds_snapshot_without_moving_main` (same shape)
- `stage_only_with_commit_branch_leaves_branch_unmoved`
- `insert_without_stage_only_advances_main` (near miss)
- `insert_overwrite_without_stage_only_advances_main` (near miss)
- `commit_branch_without_stage_only_still_targets_branch` (near miss)

Round-2 unit (`commit_snapshot_properties_tests.rs`, 7 pins):

- `test_provider_snapshot_properties_default_empty_and_threads` —
  default provider plans an exec with an empty map; the builder value
  threads through (downcast through `insert_into`)
- `test_default_commit_stamps_operation_id_without_caller_keys` —
  an empty caller map still stamps a non-empty `OPERATION_ID_PROP`
- `test_append_merges_caller_snapshot_properties`
- `test_overwrite_merges_caller_snapshot_properties`
- `test_append_caller_operation_id_does_not_win`
- `test_overwrite_caller_operation_id_does_not_win`
- `test_stage_only_snapshot_properties_land_on_staged_snapshot`
  (Append + `stage_only`: caller `wap.id` lands on the staged
  snapshot, main unmoved)

Round-2 end to end (`tests/stage_only.rs`, 4 pins):

- `insert_with_snapshot_properties_stamps_current_summary`
- `insert_overwrite_with_snapshot_properties_stamps_current_summary`
- `snapshot_properties_caller_operation_id_does_not_win` (Append path)
- `stage_only_with_snapshot_properties_stamps_staged_snapshot`

Round-3 unit (`table/tests.rs`, 2 pins):

- `test_refreshed_carries_commit_knobs_and_drops_planning_table` —
  all four knobs survive `refreshed()`, `planning_table` drops, the
  original instance is unchanged
- `test_refreshed_of_default_provider_carries_defaults` — a plain
  provider refreshes to `None` / `false` / empty / `None`

Round-4 unit (`table/tests.rs`, 1 pin):

- `test_from_planning_load_defaults_commit_knobs` —
  `from_planning_load` with no builder calls yields `None` / `false` /
  empty / `None`, and keeps `planning_table`

### Mutation arithmetic (one knob at a time, restored and re-greened after each)

1. Append arm `if stage_only` → `if !stage_only`: unit
   `physical_plan::commit::stage_only` 2 red out of 4 (exactly the two Append
   pins; overwrite and provider pins stayed green); `--test stage_only` 6 red
   out of 6, because every end-to-end test seeds through the flipped Append
   arm so `seed()` itself stages and the head assertion fails. Per-arm
   precision comes from the unit run.
2. Overwrite arm `if stage_only` → `if !stage_only`: unit 1 red out of 4
   (exactly the overwrite pin); `--test stage_only insert_overwrite` 2 red
   out of 2.
3. Restored: unit 4/4 and end-to-end 6/6 green again.

Round 2 (measured in round 3 against the round-2 threading — round 2
recorded no counts of its own; unit population
`physical_plan::commit::snapshot_properties`, 7 tests; end-to-end
`--test stage_only`, 10 tests):

4. Append arm `if key != OPERATION_ID_PROP` → `if false` (the caller
   map never merges): unit 3 red out of 7 (exactly the three
   Append-path pins: `test_append_merges_caller_snapshot_properties`,
   `test_append_caller_operation_id_does_not_win`,
   `test_stage_only_snapshot_properties_land_on_staged_snapshot`);
   end-to-end 3 red out of 10 (exactly the three Append-path pins:
   `insert_with_snapshot_properties_stamps_current_summary`,
   `snapshot_properties_caller_operation_id_does_not_win`,
   `stage_only_with_snapshot_properties_stamps_staged_snapshot`).
5. Overwrite arm, same flip: unit 2 red out of 7 (exactly the two
   Overwrite pins); end-to-end 1 red out of 10 (exactly
   `insert_overwrite_with_snapshot_properties_stamps_current_summary`).
6. Append arm `if key != OPERATION_ID_PROP` → `if true` (a caller
   operation id wins): unit 1 red out of 7 (exactly
   `test_append_caller_operation_id_does_not_win`); end-to-end 1 red
   out of 10 (exactly
   `snapshot_properties_caller_operation_id_does_not_win`).
7. Overwrite arm, same flip: unit 1 red out of 7 (exactly
   `test_overwrite_caller_operation_id_does_not_win`); end-to-end 0
   red out of 10 — the end-to-end precedence pin covers the Append
   path only, so per-arm precedence precision comes from the unit run.
8. Restored: unit 7/7 and end-to-end 10/10 green again.

Round 3, class A (population `table::tests::test_refreshed`, 2 tests;
the defaults pin stayed green in all four):

9. `refreshed()` `stage_only: self.stage_only` → `false`: 1 red out
   of 2 (`test_refreshed_carries_commit_knobs_and_drops_planning_table`).
10. `refreshed()` `snapshot_properties` clone → `HashMap::new()`: 1
    red out of 2 (same pin).
11. `refreshed()` `commit_branch` clone → `None`: 1 red out of 2
    (same pin).
12. `refreshed()` `output_spec_id` → `None`: 1 red out of 2 (same pin).
13. Restored: 2/2 green again (`git diff` shows only the test additions).

Round 4, class A (population the lib target, 401 tests;
`from_planning_load` is also the `IcebergSchemaProvider::table`
constructor, so the `stage_only` and `commit_branch` flips red
collateral tests that resolve providers through it too):

14. `from_planning_load` `stage_only: false` → `true`: 35 red out
    of 401, including the new pin (at its `!stage_only` assertion).
15. `from_planning_load` `snapshot_properties` empty → `{k: v}`: 1
    red out of 401 (exactly the new pin, at its `is_empty`
    assertion).
16. `from_planning_load` `commit_branch: None` → `Some("b")`: 20
    red out of 401, including the new pin (at its `commit_branch`
    assertion).
17. `from_planning_load` `output_spec_id: None` → `Some(0)`: 1 red
    out of 401 (exactly the new pin, at its `output_spec_id`
    assertion).
18. Restored: 400 passed, 0 failed, 1 ignored (`git diff` on
    `loaded.rs` empty).

### Gates

Measured at
`c0d7af35bd98976c6dfd518362aba3a132bc1689` (the docs commit after it adds
only prose):

- `CARGO_BUILD_JOBS=6 cargo test -q -p iceberg-datafusion` — exit 0;
  lib 399 passed, 0 failed, 1 ignored; every test binary 0 failed
- `CARGO_BUILD_JOBS=6 cargo clippy -p iceberg-datafusion --all-targets --
  -D warnings` — exit 0
- `cargo fmt --all -- --check` — exit 0
- `typos .` — exit 0
- `python3 scripts/check_rust_file_size.py` — 646 files clean (90 legacy
  ceilings), exit 0
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/xf-wap origin/main
  HEAD` — `comment-ban hits=0`, exit 0

Round 4 measured at
`a22ee54cd9a5fc62d6906d090ec503ce610204e3` (the docs commit after it
adds only prose):

- `CARGO_BUILD_JOBS=6 cargo test -q -p iceberg-datafusion --lib` —
  exit 0; 400 passed, 0 failed, 1 ignored
- `CARGO_BUILD_JOBS=6 cargo test -q -p iceberg-datafusion --test
  stage_only` — exit 0; 10 passed, 0 failed
- `CARGO_BUILD_JOBS=6 cargo clippy -p iceberg-datafusion --all-targets
  -- -D warnings` — exit 0
- `cargo fmt --all -- --check` — exit 0
- `typos .` — exit 0
- `python3 scripts/check_rust_file_size.py` — 646 files clean (90 legacy
  ceilings), exit 0
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/xf-wap origin/main
  HEAD` — `comment-ban hits=0`, exit 0

### Commits

A docs commit that writes a slice cannot cite its own sha, so each docs
commit is named by the following slice instead.

- `4f7a05fc727623efd55603f2a00ff10f4b1cca72` feat: F-STAGE-ONLY-2 thread
  stage_only from provider to commit arms
- `a831ccd6e95a2d7ee467cce465ea3c733df85c00` test: F-STAGE-ONLY-2 unit pins
  for provider default, both arms, branch composition
- `ca9e7e47f0fab21e4768cb9886ee5f757d32a1c4` test: F-STAGE-ONLY-2 end-to-end
  stage_only pins plus tests map row
- `1ab175724cd1f957cf62b7c167dacd7758bbf413` docs: F-STAGE-ONLY-2 ledger
  slice and plan
- `81c8e86f38f9369c9ced51dc4a9dfa16a87a8b35` feat: F-STAGE-ONLY-2b thread
  snapshot_properties from provider to commit arms
- `8a57a60843ee2d48d4e1e896e5b8252ff4ee16ba` test: F-STAGE-ONLY-2b unit pins
  for snapshot_properties default, both arms, precedence, stage_only
  composition
- `bb4ef734c7dcd519af90ee6be74a38be9ab263ed` test: F-STAGE-ONLY-2b end-to-end
  snapshot_properties pins plus tests map row
- `0c2386b0d90aa7205722eccbc1b4e2f14ed0a227` Merge remote-tracking branch
  'origin/main' into feat/f-stage-only-2
- `c0d7af35bd98976c6dfd518362aba3a132bc1689` test: F-STAGE-ONLY-2c pin
  refreshed() carry-through of all four commit knobs
- `b579c4eb473554464e240ab5289f36eaa514048b` docs: F-STAGE-ONLY-2
  round-3 ledger and plan to round 2 plus refreshed() pins
- `a22ee54cd9a5fc62d6906d090ec503ce610204e3` test: F-STAGE-ONLY-2d pin
  from_planning_load defaults for all four commit knobs
