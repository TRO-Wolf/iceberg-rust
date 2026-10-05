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

# F-COMMIT-OFFSET-PROPERTY-1 — a property update committed with a row delta or overwrite survives the retry path

**Date:** 2026-10-05. **Base:** fork `main` `e1d74bef` (the RePark pin), branch
`test/commit-offset-property-retry`. **Model:** Claude Opus 5.5 (`claude-opus-5-5`), medium
effort. **Card:** RePark `task/roadmap/mid-term/f-commit-offset-property-1-2026-10-05.md`
(RePark PR #953). **RePark consumers:** MB-2a (sink and offsets, plan D-1) and MB-2c (replay and
reconcile, plan D-4, CC-9).

This ledger retires when RePark repins past it, or when the owner removes it.

## Outcome

**Test-only.** The combined commit and the retry path both hold at `e1d74bef`; no fix was
needed. The same-key race overwrites the concurrent writer's value **without notice**, and the
fork has no `TableRequirement` that could refuse it. MB-2c must fence it in RePark.

## Harness

`crates/iceberg/src/transaction/offset_property_retry_tests.rs`, wired from `action.rs` as the
other OCC batteries are (`mod.rs` sits at its legacy ceiling). It reuses `occ_scoped_tests`'
`data_file` / `live_file_paths` / `snapshot_len`.

- **Base table:** V2 minimal, on the memory catalog. `commit.retry.*` is set to 2 retries and
  1–5 ms, and one base append adds `test/base.parquet`.
- **The offset commit:** one `Transaction` with two actions.
  - The first is `row_delta().add_data_files([batch])` or
    `overwrite_files().delete_file(base).add_file(batch)`. Either sets
    `set_snapshot_properties({streaming.batch-epoch: 7, streaming.offsets: src=42})`.
  - The second is `update_table_properties().set(streaming.offsets, src=42)`.
- **The racer:** `MockCatalog`. It sends `load_table` straight to the memory catalog. On the
  FIRST `update_table` it commits a prepared racer transaction to the memory catalog, then
  forwards the offset commit. That puts the racer after this attempt's load and before its
  commit, so the first attempt fails with a retryable `CatalogCommitConflicts`. Which check
  fires depends on the racer (corrected after the scoped verifier's probe, 2026-10-05):
  - When the racer appends (the two retry pins and the append-plus-property race), the
    branch-snapshot requirement fails first: "Branch or tag `main`'s snapshot has changed".
  - Only the property-only racer reaches the memory catalog's location CAS
    (`check_no_concurrent_modification`, Java `InMemoryTableOperations.doCommit`).
  `Transaction::commit` then retries: `do_commit` reloads, re-bases and re-applies every
  action.

A row delta that adds data records `Operation::Overwrite`, as Java `BaseRowDelta.operation()`
does. My first draft expected `Append` and failed 4 tests on that wrong expectation. That was a
mistake in the test, not a product defect, and it was corrected before any measurement was
recorded.

## Pins and the metadata versions they observed

| Pin | Observed |
|---|---|
| `row_delta_and_property_update_commit_as_one_metadata_version` | `metadata_log` +1, snapshots +1, metadata location moved once; current snapshot `overwrite` with both summary keys; table property `src=42`; exactly 1 snapshot carries the epoch key; the same on a fresh `load_table` |
| `overwrite_and_property_update_commit_as_one_metadata_version` | the same, for `overwrite_files()` |
| `row_delta_property_commit_survives_a_retry_over_an_unrelated_append` | `update_table` called 2× (conflict, then commit); `metadata_log` +2 and snapshots +2 (racer + ours); our snapshot's parent is the racer's append: its parent is the base snapshot, and it carries no epoch key; both summary keys and the property are present once; the racer's file and ours are both live |
| `overwrite_property_commit_survives_a_retry_over_an_unrelated_append` | the same, for `overwrite_files()` |
| `retried_commit_overwrites_a_concurrent_same_key_property_without_notice` | racer = property-only `streaming.offsets=src=99`; `update_table` 2×, no error; the final value is OURS (`src=42`); the racer's version in `metadata_log` holds `src=99` — clobbered silently |
| `retried_commit_overwrites_a_concurrent_append_and_same_key_property_without_notice` | racer = append + `src=99`; the same result |

**Green:** `cargo test -p iceberg --lib transaction::action::offset_property_retry_tests`: exit
0, 6 passed.

**Mutation** (the card's): in `Transaction::do_commit`, skip `UpdatePropertiesAction` when
re-applying on a re-based (stale) base. Exit 101, 4 failed, 2 passed:
- both retry pins went red, with `left: None, right: Some("src=42")`;
- both same-key pins went red, with `left: Some("src=99")`;
- the two no-race combined pins stayed green, as they should.

`mod.rs` was restored from a copy, and `git diff` was empty afterwards.

## Same-key race — the finding for MB-2c

- **What happens today.** A retried commit re-applies `UpdatePropertiesAction` on the refreshed
  base, and `SetProperties` overwrites the key. The conflict check (the branch-snapshot requirement, or the location CAS for
  a property-only racer) only forces the re-base; it never surfaces the overwrite. With a property-only racer, nothing else can conflict either. The
  row-delta and overwrite actions validate data conflicts only, and only when asked.
- **Why the preferred refusal can't be built.** The card wants a typed conflict naming the key.
  `UpdatePropertiesAction::commit` emits `ActionCommit::new(updates, vec![])`, with no
  requirements. The full `TableRequirement` set is `NotExist`, `UuidMatch`,
  `RefSnapshotIdMatch`, `LastAssignedFieldIdMatch`, `CurrentSchemaIdMatch`,
  `LastAssignedPartitionIdMatch`, `DefaultSpecIdMatch` and `DefaultSortOrderIdMatch`. None of
  them asserts a property value, so per the brief this item stops at measurement, and no new
  requirement type was added.
- **Consequence for RePark.** MB-2c's generation fencing (CC-9) must do the check in RePark,
  before commit: read the offset property on the refreshed table and compare it with the
  expected generation. Note that the fork's retry loop re-bases inside `commit()`. A check run
  before calling `commit()` therefore does not cover a race that lands during the retry window.
  The fencing has to tolerate that, or the fork needs an opt-in `validate` hook on
  `UpdatePropertiesAction` that compares the refreshed base's value. A `validate` failure is
  non-retryable, and `do_commit` runs it against the refreshed base. Building that hook is a
  separate card; it is not done here.

## Gates

Recorded in the round's hand-back (`/tmp/oc-worker/direct/wo/fork-microbatch/handback.json`)
with real exit codes.
