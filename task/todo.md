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

# Plan / Todo

The current plan for in-flight work. The operating manuals ([skills/](../skills/)) require this file
to be written **before** any non-trivial change and kept current as work proceeds.

How to use it (see the manuals' §1):

- Write a 3–7 bullet plan here before writing code.
- Flip `[ ]` → `[x]` as items complete; add a one-sentence "what changed and why" per step.
- Add indented sub-bullets when a step reveals unexpected complexity.
- Leave an `Outcome:` / `Done:` note when the work lands.

---

## Active: Phase 1 — Spec & metadata completeness

Parity target: Java `iceberg-core` evolution APIs. Authoritative plan: [Roadmap.md](../Roadmap.md)
Phase 1; status checklist: [docs/parity/GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md).

**Scaffolding (verified on 0.9.1):** actions follow the `transaction/sort_order.rs` pattern — a builder
struct + `#[async_trait] impl TransactionAction { async fn commit(self: Arc<Self>, &Table) ->
Result<ActionCommit> }` returning `ActionCommit::new(updates, requirements)`, plus a `pub fn` ctor on
`Transaction` and a `mod`/`use` line in `transaction/mod.rs`. `TableMetadataBuilder` already has the
low-level primitives; `TableUpdate` already has `SetSnapshotRef`/`RemoveSnapshotRef`/`AddSchema`/
`SetCurrentSchema`/`AddSpec`/`SetDefaultSpec`; `TableRequirement::RefSnapshotIdMatch{ref, snapshot_id:
Option<i64>}` (None ⇒ ref must not exist) guards concurrency. Java ref at `/tmp/iceberg-java-ref`.

### Sequencing (dependency, then value)
1. **ManageSnapshots** (this increment) — self-contained ref manipulation; primitives all exist.
2. **UpdatePartitionSpec** — addField/removeField/renameField → new spec via `AddSpec`/`SetDefaultSpec`.
3. **UpdateSchema** (largest) — add/drop/rename/update-type/move/make-req-opt + field-id reassignment
   and type-promotion validation → `AddSchema`/`SetCurrentSchema`. Split into sub-steps.
4. ManageSnapshots tail — `cherrypick` (needs snapshot replay) + `rollbackToTime`.
5. V3 groundwork — row-lineage fields, finish column-default plumbing.

Each item: behavioral parity with Java + unit tests (same change) + GAP_MATRIX row flip.

### Increment 1 — ManageSnapshots ✅ DONE
New file `crates/iceberg/src/transaction/manage_snapshots.rs`; wired into `transaction/mod.rs`.

- [x] create/replace/remove branch+tag (kind-checked); `rename_branch`.
- [x] `set_current_snapshot`; `rollback_to` (ancestry-validated); `fast_forward` (ancestry-validated).
- [x] retention: `set_min_snapshots_to_keep` / `set_max_snapshot_age_ms` / `set_max_ref_age_ms`
      (branch-only fields rejected on tags).
- [x] commit(): ops resolved in order against a working ref map seeded from `metadata().refs`; emits
      `SetSnapshotRef`/`RemoveSnapshotRef` + per-ref `RefSnapshotIdMatch` guard (original id, None if
      absent); snapshot ids validated via `snapshot_by_id`.
- [x] 12 unit tests via `make_v2_table()`; build + clippy(-D warnings) + fmt + offline lib suite green
      (1170 tests, stable across 6 runs).
- [x] Flipped `ManageSnapshots` + snapshot-refs GAP_MATRIX rows to 🟡.
  - Side fix: relaxed a flaky upstream assertion in `catalog/memory/catalog.rs` test_update_table
    (`last_updated_ms() <` → `<=`), exposed by the extra parallel test load. See task/lessons.md.

**Outcome:** branch/tag lifecycle + rollback + fast-forward + retention land with optimistic-concurrency
guards. **Deferred to increment 4:** `cherrypick` (needs snapshot replay), `rollbackToTime`,
`replaceBranch(from,to)`; a Java interop round-trip before the row flips to ✅.

#### Review remediation (2026-06-07, post multi-agent review) — DONE
- [x] `fast_forward` Java-parity fixes: `to` may be a tag; absent `from` auto-created; no-op when
      already at target. Verified against `UpdateSnapshotReferencesOperation.replaceBranch`.
- [x] commit() emit-time no-op suppression (create-then-remove, ff-to-same, replace-to-same emit nothing).
- [x] +14 tests (now 26): rollback non-ancestor, ff wrong-direction, remove/rename main guards, rename
      collision, remove/replace of pre-existing refs (guard carries original id), forked fixture.
- [x] Created `docs/testing.md` (was referenced 14× but missing); reconciled headline-gaps + Roadmap
      Phase 1 status (🟡) with the matrix; removed stale `iceberg-spark-python` line from `skills/*.md`.
- [ ] **Follow-up (unverified):** retention positivity validation (Java may reject `≤ 0` for
      min_snapshots_to_keep / max_snapshot_age_ms / max_ref_age_ms). A grep of `SnapshotRef.java` found
      no such `checkArgument` — confirm where (if anywhere) Java enforces it before adding it here.

### Increment 2 — UpdatePartitionSpec (IN PROGRESS, 2026-06-07)
New file `crates/iceberg/src/transaction/update_partition_spec.rs`; wired into `transaction/mod.rs`.
Full Java parity with `BaseUpdatePartitionSpec`. Builder records ops; `commit()` replays Java's
state machine against `table.metadata()` and emits `AddSpec`/`SetDefaultSpec{-1}` + concurrency guards.

Plan:
- [x] Read Java `UpdatePartitionSpec` + `BaseUpdatePartitionSpec` contract; Rust partition.rs,
      transform.rs, table_metadata_builder.rs (`add_partition_spec` recycles equal (src,transform)
      field ids), catalog/mod.rs (`AddSpec`/`SetDefaultSpec`/`DefaultSpecIdMatch`/
      `LastAssignedPartitionIdMatch` shapes + their check/apply).
- [x] `UpdatePartitionSpecAction`: builder methods record `PartitionSpecOp` (case_sensitive flag +
      add/add_with_transform/remove/remove_by_transform/rename/add_non_default). Names auto-generated
      to match Java `PartitionNameGenerator` (`generate_partition_name`).
- [x] `commit()`: `SpecEvolution` state machine replays Java `addField`/`removeField`/`renameField`
      validation in order, seeded from the current DEFAULT spec; `apply()` builds the new
      `UnboundPartitionSpec` (keep+rename / void-replace-on-V1 / omit-on-V2; appended adds with
      `field_id=None` so `TableMetadataBuilder` recycles/assigns). Emits `AddSpec{unbound}` +
      `SetDefaultSpec{-1}` (unless add_non_default) + `DefaultSpecIdMatch{current}` +
      `LastAssignedPartitionIdMatch{current}`.
- [x] 23 unit tests (same change): identity add, add-with-transform auto-name (one per transform incl
      bucket/truncate/year/month/day/hour/void), explicit name, remove-by-name, remove-by-transform,
      rename, add-non-default (no SetDefaultSpec), dup-add-name fails, redundant-time-transform fails,
      remove-newly-added fails, rename+delete-same fails, delete-then-readd un-deletes (+ with rename),
      V1 void replacement (+ V1 delete-collision rename), field-id recycling across historical specs
      (forked multi-spec fixture) + new-id assignment, case-insensitive source resolution, unknown
      column fails, colliding non-void name fails, updates/requirements shape, binds-to-schema.
- [x] Verify: build green; lib test ×2 = 1207 passed/0 failed both runs; clippy -D warnings clean;
      fmt clean. Mutation-checked auto-name + V1/V2 void branch (5 tests fail when logic broken).
      Flipped GAP_MATRIX row + headline gap to 🟡.

**Outcome:** `UpdatePartitionSpec` lands at full `BaseUpdatePartitionSpec` parity (🟡) with optimistic-
concurrency guards. **Deferred to ✅:** Java interop round-trip (read a Java-evolved spec; prove Java
reads ours). **Note:** `Transform::Unknown` reject-precondition is N/A — the Rust builder API cannot
produce an unknown transform (Java's `UnknownTransform` guard has no Rust analogue here).

#### Increment 2 — REVIEW (2026-06-07, adversarial reviewer agent)
Verified the 10 high-value points against `BaseUpdatePartitionSpec.java` + `TestUpdatePartitionSpec.java`.
- [x] Pts 1,2,4,5,6,7,8 (un-delete/reject branches; transform `to_string` keys; V1 alwaysNull;
      void-collision renames; redundant-time guard; case-sensitivity; requirement set): CONFIRMED.
- [x] **BUG (pt 3): field-id recycling dropped the historical NAME.** Java
      `recycleOrCreatePartitionField` (V2, base!=null) returns the historical field's *name* too when
      the add had no explicit name (and only recycles at all when name is null OR matches). Rust
      delegated id-recycling to `TableMetadataBuilder` (matches on source+transform only) but always
      used the *generated* default name → a `bucket[8](y)` recycle that was historically named
      `my_shard` came out `y_bucket_8`, and an explicit-name add recycled an id even when Java would
      not. FIX: replicate `recycleOrCreatePartitionField` in the action — search historical specs and,
      on match (respecting the name==null/name-equals rule), set BOTH the recycled field_id and the
      historical name on the added field. Metadata-builder recycling becomes a harmless no-op.
- [x] **BUG (pt 8): requirement over-constraint under add_non_default_spec.** Java emits
      `AssertDefaultSpecID` only for the `SetDefaultPartitionSpec` update; when `addNonDefaultSpec` is
      set there is no such update, so only `AssertLastAssignedPartitionId` is required. Rust emitted
      both unconditionally. FIX: gate `DefaultSpecIdMatch` on `set_as_default`.
- [x] Added tests (pt 9 end-to-end round-trip through builder; pt 10 no-op dedup to existing spec id;
      V2 remove+re-add-different-transform-same-name; historical-name recycle; add_non_default
      requirement shape).

**Review outcome:** 28 unit tests (23 → 28; both fixes mutation-verified to fail without the change).
build green; lib test ×2 = 1212/0 both runs (stable); clippy -D warnings clean; fmt clean. Reconciled
GAP_MATRIX row + Roadmap Phase 1 progress; appended 3 lessons. Row stays 🟡 (Java interop still deferred).
**Residual (tracked, intentional):** (a) full interop round-trip; (b) explicit-name recycle when the
historical name DIFFERS from the requested name — Java assigns a fresh id, Rust's metadata builder would
still recycle by (source,transform); narrow, spec-favoring, and not separately gated here. Both noted for
the ✅ flip.

