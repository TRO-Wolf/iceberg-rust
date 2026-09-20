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
  `PublishChangesProcedure`'s lookup verbatim (full `snapshots()` scan, non-unique
  and unknown messages verbatim), PLUS the already-published check (`is_wap_id_published`
  on current ancestry → `DuplicateWAPCommitException` message), so one call gives the
  caller the whole Java publish-eligibility decision. Lookup-then-duplicate order is
  Java's: a published-but-expired id reports "unknown", matching the procedure's
  lookup-first failure. `cherry_pick.rs::is_wap_id_published` went `pub(crate)` —
  one home for the dedup rule, reused, not duplicated.
- `PublishChangesAction` + `Transaction::publish_changes(wap_id)` — a stateless
  `TransactionAction` that resolves the staged snapshot off the REFRESHED table on
  every `validate`/`commit` call (retry-safe, same discipline `CherryPickAction`
  documents) and delegates to `CherryPickAction::new(staged_id)`. `main`-only:
  Java's procedure publishes to `main`, so `target_ref` keeps the default.
- `mod publish_changes` and `mod stage_only_tests` are `#[path]`-wired from
  `action.rs` — `mod.rs` sits on its 1937-line legacy ceiling; the escape valve is
  the same one `occ_scoped_tests.rs` uses.

## Pins (`stage_only_tests.rs`, wired from `action.rs`)

Stage pins (one per RePark write action): a staged commit adds exactly one snapshot,
`current-snapshot-id`/`main` ref/snapshot log unchanged, a main read unchanged
(live file sets), the staged snapshot readable by id carrying `wap.id`.

Lookup pins: found returns the staged snapshot; unknown → Java text; non-unique →
Java text; already-published → Java `DuplicateWAPCommitException` text.

Publish pins: FF publish moves `main` to the staged snapshot verbatim keeping
`wap.id`; replay publish (head advanced after staging) produces a new snapshot with
`source-snapshot-id` + `published-wap-id` and the staged data in the live set;
per-action publish coverage (append, merge append, overwrite, replace-partitions
replay removing the replaced partition's old file, row delta, delete); unknown id;
double-publish; V3 row lineage — the published snapshot's `first_row_id` is assigned
FRESH at publish (Java `MergingSnapshotProducer.add` suppresses the staged id and the
manifest-list writer assigns from `next-row-id`), so it equals `next_row_id` captured
just before publish and differs from the staged snapshot's own `first_row_id`.

## Mutation evidence (executed)

Baseline: `cargo test -p iceberg --lib stage_only_tests` — 14 passed; 0 failed.

Mutation 1 — stage flag ignored in `merge_append.rs` (`.with_stage_only(self.stage_only)`
→ `.with_stage_only(self.stage_only && false)`): **1 red out of 14** —
`merge_append_stage_only_adds_snapshot_without_moving_main`, failing assertion:

```text
assertion `left == right` failed: stage_only must not advance current-snapshot-id
  left: Some(4951073193712195810)
 right: Some(3567619747308661495)
```

Restored; rerun 14 passed; 0 failed.

Mutation 2 — duplicate-publish check bypassed in `publish_changes.rs`
(`if is_wap_id_published(metadata, wap_id)` → `if false && is_wap_id_published(metadata, wap_id)`):
**1 red out of 14** — `staged_snapshot_for_wap_id_already_published_has_java_message`,
failing assertion:

```text
an already-published wap id must fail: Snapshot { snapshot_id: 5436803876107560008, ... }
```

(the helper returned `Ok(staged)` for an already-published wap id). Restored; rerun
14 passed; 0 failed.

Notable: `publish_changes_twice_has_java_message` stayed GREEN under mutation 2 —
`CherryPickAction::validate`'s own `validate_wap_publish` catches the duplicate on the
replay path independently. That is Java's shape too (`WapUtil.validateWapPublish` is
invoked by both `PublishChangesProcedure`'s resolved cherry-pick and the direct
`CherrypickOperation`), so the lookup's check is the procedure-level duplicate guard
while cherry-pick's is the action-level one; the helper pin is the load-bearing test
for the former.

## Gates

- `cargo fmt --all` — clean
- `cargo clippy -p iceberg --all-targets -- -D warnings` — clean
- `cargo test -p iceberg --lib stage_only_tests` — 14 passed; 0 failed
- `bash scripts/check_rust_file_size.sh` — 593 files clean (90 legacy ceilings);
  ceilings LOWERED for `replace_partitions.rs` (2783 → 2782) and `mod.rs`
  (1937 → 1936) after blank-line reclamation
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/qb-fork2 origin/main` —
  `comment-ban hits=0`
