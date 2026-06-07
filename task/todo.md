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

### Increment 3 (prereq) — Type-promotion helper for UpdateSchema (IN PROGRESS, 2026-06-07, Actor A2)
New file `crates/iceberg/src/spec/schema/type_promotion.rs`; wired via one `mod` + `pub(crate) use`
line in `crates/iceberg/src/spec/schema/mod.rs`. Pure, self-contained spec module — no transaction dep.

Contract: mirror Java `TypeUtil.isPromotionAllowed(Type from, Type.PrimitiveType to)`
(`/tmp/iceberg-java-ref/.../types/TypeUtil.java` lines 440–466), verified against the source (not the
digest paraphrase). The switch has EXACTLY three branches + an identity short-circuit:
- `from.equals(to)` (Type-level) ⇒ allowed (no-op / identity).
- `INTEGER` ⇒ allowed iff `to == LONG`.
- `FLOAT`  ⇒ allowed iff `to == DOUBLE`.
- `DECIMAL`⇒ allowed iff `to` is DECIMAL AND `from.scale == to.scale` AND `from.precision <= to.precision`.
- everything else ⇒ forbidden.

Plan:
- [x] Implement `ensure_promotion_allowed(from: &Type, to: &PrimitiveType) -> Result<()>` (+ boolean
      core `is_promotion_allowed`): Ok(()) when allowed; Err(ErrorKind::DataInvalid,
      "{from} cannot be promoted to {to}") otherwise, mirroring Java `CheckCompatibility.primitive`.
- [x] Wired into schema/mod.rs as `mod type_promotion;` + `pub use self::type_promotion::{...}` (both
      fns are `pub` — task asked for a *public* helper, matches Java's `public static`, and avoids the
      dead-code warning that `pub(crate)` would raise since UpdateSchema does not exist yet).
- [x] 21 unit tests (6 allowed + 15 rejected; `rejects_struct_from_to_primitive` covers struct/list/map).
      Each names the risk; rejection tests assert exact ErrorKind::DataInvalid + exact message.
      Mutation-verified: dropping `from_scale == to_scale` fails `rejects_decimal_scale_change...`.

**Outcome (2026-06-07, Actor A2):** Shipped `crates/iceberg/src/spec/schema/type_promotion.rs` +
two lines in `schema/mod.rs`. Public API: `is_promotion_allowed(&Type, &PrimitiveType) -> bool` and
`ensure_promotion_allowed(&Type, &PrimitiveType) -> Result<()>` (re-exported via `spec::*`). Verify:
build clean; lib suite 1233/0 ×2; clippy -D warnings clean; fmt clean. **DEVIATION:** task brief listed
`date->timestamp` as allowed — Java's `TypeUtil.isPromotionAllowed` has NO DATE branch, so it is
implemented as REJECTED (matches Java + digest §6 pt5). Flagged in final report. No interop test (no
Java-written fixture for a pure type-check helper; the UpdateSchema interop test will exercise it).

**DEVIATION FROM TASK PROMPT (flag in final report):** the task brief said "cover ... date->timestamp"
as an allowed promotion. The authoritative Java source `TypeUtil.isPromotionAllowed` has NO `DATE` case
— `date -> timestamp` is FORBIDDEN in Java (confirmed by reading the source + the run's digest §6 pt 5).
Implementing it would diverge from Java and fail interop. Per the contract ("verify against the Java
source, not intuition"), I implement the real Java matrix (date→timestamp REJECTED) and surface this.

### Increment 3 — UpdateSchema transaction action (IN PROGRESS, 2026-06-07, Actor A1 Opus)
New file `crates/iceberg/src/transaction/update_schema.rs`; wired into `transaction/mod.rs`
(`mod` + `use` + `pub fn update_schema()`). Full Java parity with `SchemaUpdate`. Builder records ops;
`commit()` replays Java's state machine against `metadata().current_schema()`, builds the new `Schema`
via a recursive `ApplyChanges` walk (mirrors Java `SchemaUpdate.ApplyChanges` SchemaVisitor), and emits
`AddSchema` + `SetCurrentSchema{-1}` guarded by `LastAssignedFieldIdMatch` + `CurrentSchemaIdMatch`.

Inputs/outputs/contract: builder methods (add_column/add_required_column/rename_column/update_column/
update_column_doc/make_column_optional/require_column/delete_column/move_*/set_identifier_fields/
union_by_name_with/case_sensitive/allow_incompatible_changes) record `SchemaOp`s; `commit(&Table)`
resolves them in order against the current schema and returns `ActionCommit::new(updates, requirements)`.

Edge cases / risks (Risk-First): fresh field-id assignment for nested adds (reuse `ReassignFieldIds`);
type-promotion accept+reject (call `ensure_promotion_allowed`); add-required gating on
`allow_incompatible_changes`; optional→required gating (+ defaulted-add path); identifier-field
validation (exists/required/primitive/not-deleted/not-nested-in-map-or-list); move self-ref / cross-struct
/ non-struct-parent rejections; delete vs add/update/rename conflict matrix; map-key immutability;
case-insensitive resolution. DEVIATION: Rust `TableUpdate::AddSchema` has NO `last_column_id` field
(only `ViewUpdate::AddSchema` does) — the builder's `add_schema` derives last_column_id from
`schema.highest_field_id()`. Requirements: Java `UpdateRequirements` attaches BOTH
`AssertLastAssignedFieldId(base.lastColumnId())` (for AddSchema) AND `AssertCurrentSchemaID`
(for SetCurrentSchema) — confirmed in `UpdateRequirements.java` lines 131-142. We emit both.

Plan:
- [x] Builder + `SchemaOp` enum recording all op families; case_sensitive + allow_incompatible flags.
- [x] `SchemaEvolution` state machine (mirrors `SchemaUpdate` fields: deletes/updates/added-name-to-id/
      parent-to-added-ids/moves/identifier-field-names/last_column_id) replaying ops with Java's
      precondition checks; fresh-id assignment via a self-contained `assign_fresh_ids` walk (the schema
      module's `ReassignFieldIds` is crate-private and unreachable from `transaction/`, and is for
      whole-schema reassignment; a local walk mirrors Java `TypeUtil.assignFreshIds` for a single add).
- [x] `apply()` recursive rebuild → new `Schema` (deletes drop, updates replace, adds append to parent
      struct, moves reorder; map-key immutability + list/map-value rules); identifier-field validation
      delegated to `Schema::builder` (spec rules) after re-resolving names to fresh ids.
- [x] `commit()` emits AddSchema + SetCurrentSchema{-1} + LastAssignedFieldIdMatch(last_column_id) +
      CurrentSchemaIdMatch(current_schema_id) — both guards per Java `UpdateRequirements` (lines 131-142).
- [x] 46 unit tests (≥1 happy + ≥1 negative per op family); nested fixture built fresh via
      `TableMetadataBuilder::new` (no stale sort order / identifier set).
- [x] Verify: build clean; lib test ×2 = 1279/0 both runs (stable); clippy -D warnings clean; fmt clean.

**Outcome (2026-06-07, Actor A1 Opus):** `UpdateSchemaAction` lands at SchemaUpdate parity (builder +
state machine + recursive apply). Public ctor `Transaction::update_schema()`. **DEVIATIONS (flagged):**
(1) Rust `TableUpdate::AddSchema` has NO `last_column_id` field (only `ViewUpdate::AddSchema` does) — the
brief said emit `AddSchema{schema, last_column_id}`; we emit `AddSchema{schema}` and the metadata builder
derives last_column_id from `schema.highest_field_id()`. (2) Column DEFAULTS are not plumbed through the
builder API yet (the `addColumn(..., Literal)` / `updateColumnDefault` / `addRequiredColumn(..., default)`
Java overloads) — `NestedField` supports initial/write defaults but the action's builder methods do not
take a default, so the "required add WITH default needs no flag" and "defaulted-add can be made required"
Java paths are present in the state machine (`is_defaulted_add`) but unreachable from the public API. (3)
`union_by_name_with` is a pragmatic name-driven merge over struct fields (add-new / promote-widening /
apply-doc / keep-wider-on-narrowing), NOT a port of Java's full `UnionByNameVisitor` (which also handles
list/map element promotion and required→optional). **Deferred to ✅:** column-default builder overloads;
full UnionByNameVisitor parity (nested list/map element merge); Java interop round-trip.

#### Increment 3 — REVIEW remediation (2026-06-07, Opus actor, post 3-critic review)
Three perspective-diverse critics reviewed the UpdateSchema action. Verified each VERIFIED-REAL finding
against the Java source before acting. Scope: `transaction/update_schema.rs`, `spec/schema/mod.rs`
(lowercase index), docs. Plan:

- [x] **BLOCKER — nested field-id order (depth-first → level-order).** Rewrote `assign_fresh_ids`:
      struct assigns ALL immediate ids in pass 1 then recurses in pass 2; map assigns key_id then value_id
      FIRST then recurses key then value. Confirmed against `AssignFreshIds`/`CustomOrderSchemaVisitor` +
      `testAddNestedMapOfStructs`. Pinned by 3 exact-id tests (map<struct,struct>, list<struct>,
      struct{struct,prim}). Mutation-verified: depth-first map arm gives value-id 8 vs Java's 4.
- [x] **MAJOR — union_by_name full `UnionByNameVisitor` parity** (findings #1,#3,#4,#5,#6,#7,#8). Rewrote
      `union_struct` into `union_update_existing` + `union_recurse_into` + `union_nested_member` +
      free fn `is_ignorable_type_update`. Routes existing-field changes through `update_column`/
      `update_column_requirement`/`update_column_doc`; rejects incompatible primitive + complex↔primitive
      changes ("Cannot change column type"); relaxes required→optional; recurses struct/list/map. Added
      9 union tests (nested struct add, list<struct> nested add, required→optional, incompatible-primitive
      reject, list→primitive reject, mirrored no-op, doc-only). Mutation-verified the reject path.
- [x] **MAJOR — case-insensitive lowercase-name collision** (`spec/schema/mod.rs`). Added
      `build_lowercase_name_index` rejecting collisions with the exact Java message (smaller field-id
      first → `data and DATA collide`). 2 tests (collision rejected + distinct-lowercase accepted).
      Mutation-verified against the old `.collect()`.
- [x] **MINOR (test rigor)** — added exact-`ErrorKind::DataInvalid`+message asserts on 3 high-value
      negatives (ambiguous-name, delete-with-updates, move-before-itself); identifier id-stability tests
      (rename + move keep the identifier id); list-element struct move; delete+re-add+move.
- [x] **SKIP/TRACK — column defaults (#10).** Not fixed: plumbing `initial/write` defaults is a
      signature-changing API expansion beyond the named scope (§6). Tracked as the residual gap for the
      ✅ flip; row stays 🟡. (Finding #10's own recommendation explicitly allows tracking it as a gap.)
- [x] Reconciled GAP_MATRIX (UpdateSchema ❌ → 🟡) + headline gaps; Roadmap Phase 1 progress +
      snapshot/"next move"/current-state lines; appended dated lessons. Verification gate run last.

**Outcome (2026-06-07, Opus remediation):** all 9 VERIFIED-REAL code findings fixed (1 blocker, 4 majors
collapsed into 2 rewrites — union + lowercase, plus 4 test-coverage/rigor findings); 1 finding (#10
column defaults) tracked as a scoped-out parity gap. Files touched: `transaction/update_schema.rs`,
`spec/schema/mod.rs`, `transaction/mod.rs` (NOT edited — already wired), GAP_MATRIX, Roadmap, todo,
lessons. **Verify:** build clean; lib test ×2 = 1298/0 both runs (stable); clippy -D warnings clean; fmt
clean. update_schema 46 → 63 tests; +2 schema-build tests. Row stays 🟡 (defaults + Java interop deferred).

### Increment 4 — ManageSnapshots tail (rollbackToTime + retention `>0`) + UpdateSchema defaults (IN PROGRESS, 2026-06-07, BUILDER Opus)
Three Phase-1 metadata items at full Java parity. cherrypick is OUT OF SCOPE (it extends
`MergingSnapshotProducer` / replays data files → Phase 2; reclassified in docs only). Files touched:
`transaction/manage_snapshots.rs`, `transaction/update_schema.rs`, `docs/parity/GAP_MATRIX.md`,
`Roadmap.md`, `task/todo.md`, `task/lessons.md`.

**Java rules verified against `/tmp/iceberg-java-ref` source (not intuition):**
- **A (rollback_to_time):** `SetSnapshotOperation.rollbackToTime` → `findLatestAncestorOlderThan(base, ts)`
  walks `SnapshotUtil.ancestorIds(currentSnapshot)` (the parent chain of MAIN's current snapshot — exactly
  the existing `is_ancestor_of` walk) and picks the snapshot with the MAX `timestampMillis` that is
  STRICTLY `< ts`; errors "Cannot roll back, no valid snapshot older than: {ts}" if none. Then sets MAIN to
  it (same emit path as `rollback_to`). Note the STRICT `<`: ts == current's timestamp picks the next-older
  ancestor; ts > current keeps current (no-op, suppressed at emit).
- **B (retention >0):** `api/SnapshotRef.java` Builder setters (lines 154-177) DO reject non-positive:
  `minSnapshotsToKeep` `value == null || value > 0` "Min snapshots to keep must be greater than 0";
  `maxSnapshotAgeMs` "Max snapshot age must be greater than 0 ms"; `maxRefAgeMs` "Max reference age must be
  greater than 0". (Resolves the earlier unverified follow-up — the prior grep checked the wrong file.)
- **C (UpdateSchema defaults):** `core/SchemaUpdate.internalAddColumn` line 160: a required add is allowed
  when `defaultValue != null || isOptional || allowIncompatibleChanges` (default backfills existing rows).
  Add sets BOTH `withInitialDefault(default)` AND `withWriteDefault(default)` (lines 181-182).
  `updateColumnDefault(name, lit)` sets ONLY `writeDefault` on an existing field (no-op if already equal),
  line 339. Java `Types.NestedField` ctor `castDefault` validates: reject non-null default for a nested
  type; for a primitive, `defaultValue.to(type)` must succeed. Rust `with_initial_default`/`with_write_default`
  do NOT validate, so validate via `literal.try_into_json(&field_type)` (the canonical serde compatibility
  check — passing it guarantees no later serialization panic) before setting.

Plan:
- [x] **A.** `SnapshotOp::RollbackToTime { timestamp_ms: i64 }` + `pub fn rollback_to_time(self, i64)`;
      resolver `find_latest_ancestor_older_than` walks MAIN's parent chain, picks newest with
      `timestamp_ms < arg`, error if none; `set_main` to it. 5 tests (newest-older-ancestor, strict-`<`
      equal boundary, before-first→error, after-current→noop, sibling-never-chosen).
- [x] **B.** `validate_retention_positive` at the head of `apply_retention` (the one place all three
      fields flow through): rejects `<= 0` with exact Java messages; `ErrorKind::DataInvalid`. 6 negatives
      (zero+negative × 3 fields); existing positive retention tests stay green.
- [x] **C.** plumbed `Option<Literal>` default through new `add_column_with_default` /
      `add_required_column_with_default` / `add_column_to_with_default` /
      `add_required_column_to_with_default` builders + new `update_column_default(name, Literal)` builder +
      `SchemaOp::AddColumn.default` / `SchemaOp::UpdateColumnDefault`. `add_column` apply: required-without-
      default guard now relaxed when a default is present (`required && default.is_none() && !flag`); sets
      BOTH initial+write defaults on the new field; `update_column_default` sets ONLY write_default on an
      existing field (no-op if equal). `validate_default` rejects non-primitive defaults ("Invalid default
      value...") and type-mismatched primitives ("Cannot cast default value to...") via `try_into_json`
      (the canonical serde-compat check — passing it guarantees no later serialization panic). 9 tests.
- [x] Docs: GAP_MATRIX (ManageSnapshots row — rollbackToTime done, retention >0, cherrypick Phase-2-gated;
      UpdateSchema row — defaults dropped from "Pending ✅"; headline gaps reconciled); Roadmap Phase 1
      progress + increment-4 entry + headline gaps + current-state snapshot; this todo; lessons (6 entries).
- [x] Verify: build clean; lib test ×2 = 1318/0 both runs (stable, was 1298 baseline → +20 new tests);
      clippy -D warnings clean; fmt clean (one reflow applied via `cargo fmt`).

**Outcome (2026-06-07, Increment 4, BUILDER Opus):** all three items land at Java-source-verified parity.
**A** `rollback_to_time` (5 tests), **B** retention `>0` rejection (6 tests), **C** UpdateSchema column
defaults — `add_*_with_default` builders + `update_column_default` (9 tests). `cherrypick` reclassified
Phase-2-gated in docs (NOT implemented — it extends `MergingSnapshotProducer`). Files touched exactly the
6 allowed: `transaction/manage_snapshots.rs`, `transaction/update_schema.rs`, GAP_MATRIX, Roadmap, todo,
lessons. Nothing touched outside the allowed set; no `spec/datatypes.rs` change needed
(`with_initial_default`/`with_write_default` already exist). Rows stay 🟡 — Java interop round-trip is the
only remaining gate before ✅. An Opus REVIEWER verifies next.

#### Increment 4 — REVIEW (2026-06-07, Opus REVIEWER)
Adversarially verified points 1–5 against the Java source (`/tmp/iceberg-java-ref`), not the Rust comments.
- [x] **Pt 1 (rollback_to_time): CONFIRMED.** `findLatestAncestorOlderThan` (SetSnapshotOperation.java:146)
      walks `currentAncestors` (main parent chain), strict `<` on `timestampMillis`, picks the MAX. Rust
      `find_latest_ancestor_older_than` matches exactly. Boundary `ts == ancestor's timestamp` → that
      ancestor is excluded (strict `<`); `ts == current` → next-older selected; `ts > current` → current
      (no-op suppressed). Java test `testAttemptToRollbackToCurrentSnapshot` mirrors the Rust no-op test.
      Benign divergence noted (not a bug): Java seeds `snapshotTimestamp=0` so a snapshot with
      `timestampMillis <= 0` is never chosen; Rust has no such floor. Real ms timestamps are always > 0,
      so unreachable in practice — tracked, not fixed.
- [x] **Pt 2 (retention >0): CONFIRMED.** SnapshotRef.java Builder setters (lines 154-177) enforce
      `value == null || value > 0` with the three exact messages; Rust reproduces them verbatim in
      `validate_retention_positive`, called at the head of `apply_retention`. `null`/unset still allowed
      (the builder API always sets a concrete value, so only `<= 0` occurs). Existing positive-retention
      tests stay green.
- [x] **Pt 3 (UpdateSchema add/update defaults): CONFIRMED.** `internalAddColumn` (SchemaUpdate.java:160)
      gates on `defaultValue != null || isOptional || allowIncompatibleChanges` (Rust De Morgan equivalent
      `required && default.is_none() && !flag`); add sets BOTH `withInitialDefault`+`withWriteDefault`
      (lines 181-182); `updateColumnDefault` (line 339) sets ONLY `withWriteDefault`. All three match.
- [x] **Pt 4 (default type-validation): CONFIRMED (safety) + divergence noted.** The serde `From<NestedField>`
      path (datatypes.rs:591-592) calls `try_into_json(&field_type).expect(...)`. `validate_default` runs the
      SAME `try_into_json` at add/update time, so it is a PERFECT predictor of the panic: anything it passes
      cannot panic later. Parity-precision divergences vs Java `defaultValue.to(type)` exist only on
      deliberately type-mismatched `Literal`s (e.g. a UUID/binary literal accepted for any primitive via the
      `(_, UInt128|Binary)` wildcard arms = too lenient; an int literal rejected for a long column = too
      strict). These require the caller to hand-build a mismatched literal; the Rust `Literal` is already
      strongly typed so the natural usage matches the column. No corruption, no panic — tracked, not fixed.
- [x] **Pt 5 (builder naming/arity): CONFIRMED.** Every Java overload semantic
      ({optional|required}×{top-level|nested}×{doc}×{default}) is reachable via the 8 `add_*` builders +
      `update_column_default`. Top-level optional-with-doc uses `add_column_to(None, ..)` (doc preserved;
      bypasses the dotted-name guard — ergonomic, not a lost semantic).
- [x] **TEST GAP FOUND + FIXED (not a code bug):** the `is_defaulted_add` make-required branch
      (update_schema.rs:802) was UNTESTED. Java pins it with two tests: `testAddColumnWithDefaultToRequired
      Column` (optional add WITH default → requireColumn succeeds without the flag) and
      `testAddColumnWithUpdateColumnDefaultToRequiredColumn` (add + updateColumnDefault sets only write_default
      → requireColumn FAILS, since initial_default is still null). Added both as Rust tests. Mutation-verified:
      dropping `&& field.initial_default.is_some()` from `is_defaulted_add` makes the negative test pass-when-
      it-should-fail (caught); the positive test fails if the defaulted-add relaxation is removed.
- [x] Verify gate run; rows stay 🟡 (Java interop deferred); cherrypick stays Phase-2-gated. Files touched:
      `transaction/update_schema.rs` (+2 tests), todo, lessons. Nothing outside the allowed set.

### Increment 5 — UpdateSchema INTEROP PILOT (Phase-1 bidirectional Java round-trip, BUILDER Opus, 2026-06-07)
Goal: prove byte-/field-id-level UpdateSchema compatibility with Java `iceberg-core` 1.10.0 in BOTH
directions, so the GAP_MATRIX `UpdateSchema` row can flip 🟡 → ✅. Java is a TEST-ONLY ORACLE under
`dev/java-interop/` (a dev tool like `dev/spark/`) — NOT a crate, NOT in the Cargo graph; `cargo
build`/`cargo test` never invoke Java. Durable artifacts = committed JSON fixtures + Rust tests.

Scope (new files only; doc edits to GAP_MATRIX/Roadmap/todo/lessons):
- `dev/java-interop/{pom.xml,README.md,run.sh}` + `src/main/java/org/apache/iceberg/InteropOracle.java`
  (package `org.apache.iceberg` to reach the `@VisibleForTesting SchemaUpdate(Schema,int)` ctor).
- `crates/iceberg/testdata/interop/update_schema/<scenario>/{base,java_evolved,rust_evolved}.metadata.json`.
- `crates/iceberg/tests/interop_update_schema.rs` (auto-discovered; no Cargo.toml edit).

Environment confirmed: `/opt/maven/bin/mvn` 3.9.9, Java 11.0.31 (iceberg 1.10 needs Java 11+, OK).
Rust `TableMetadata` round-trips via plain `serde_json` (its serde impl). Java testing ctor + `apply()`
confirmed at SchemaUpdate.java:75/467; `TableMetadata.newTableMetadata`/`buildFrom`/`setCurrentSchema
(Schema,int)`/`TableMetadataParser.{toJson,read}` confirmed. Literal factories: Rust
`Literal::{int,long,...}`, Java `Literal.of(...)`. Compare by PARSING both into the Rust model and
asserting structural equality (Schema/NestedField PartialEq + explicit field-ids + current-schema-id +
last-column-id) — NOT raw JSON bytes (Jackson vs serde key-order differ).

Scenarios (≥7, identical names Java+Rust): `add_top_level_columns`; `add_nested_struct_and_map` (THE
level-order fresh-field-id case — assert exact nested ids); `rename_and_move`; `update_type_promotion`
(int→long, float→double, decimal widen); `make_optional_and_delete`; `set_identifier_fields`;
`add_required_with_default_and_update_default`.

Plan:
- [x] 1. Java oracle: `pom.xml` (iceberg-core+api 1.10.0, exec-maven-plugin) + `InteropOracle.java`
      with `generate` (build base+java_evolved metadata via testing ctor → write JSON) and `verify`
      (read rust_evolved.metadata.json, assert structural schema equality vs java_evolved).
      Compiles + runs under Maven 3.9.9 / Java 11. Used `TableMetadataParser.fromJson(location, json)`
      (no 3-arg `read`); compared via `Schema.asStruct().equals` + `identifierFieldIds().equals`.
- [x] 2. Rust test `interop_update_schema.rs`: Dir-1 + Dir-2 producer. **DEVIATION from the brief's
      "apply emitted updates through TableMetadataBuilder":** `TransactionAction::commit` is `pub(crate)`
      and the `update_schema` module is private, so an external integration test cannot reach the action's
      raw updates. Drove the evolution through the PUBLIC API instead — a real `MemoryCatalog`
      (`LocalFsStorageFactory` over a tempdir) register-table + `Transaction` + `ApplyTransactionAction` +
      `commit` — which is strictly stronger (it also exercises the optimistic-concurrency requirement
      checks) and still applies the emitted `TableUpdate`s through `TableMetadataBuilder` internally. No
      production code touched.
- [x] 3. Runner `run.sh` (java generate → rust gen+assert → java verify) + `README.md` (TEST ORACLE).
- [x] 4. Generated all fixtures; Dir-1 green; `mvn verify` 7/7 PASS. Committed base/java_evolved/
      rust_evolved JSON for all 7 scenarios.
- [x] 5. Verify gate: interop test 3/3; lib suite 1320/0 ×2 (was 1318 baseline → +0 lib, the interop
      lives in tests/ not lib); clippy -D warnings clean; fmt --check clean. Flipped GAP_MATRIX
      `UpdateSchema` 🟡→✅; reconciled Roadmap + headline gaps + provenance.

**Outcome (2026-06-07, Increment 5 INTEROP PILOT, BUILDER Opus):** bidirectional Java interop for
`UpdateSchema` lands; row ✅. Harness layout: `dev/java-interop/{pom.xml,run.sh,README.md,src/main/java/
org/apache/iceberg/InteropOracle.java}` (TEST ORACLE, not a crate) + `crates/iceberg/tests/
interop_update_schema.rs` + 7×3 committed JSON fixtures under `crates/iceberg/testdata/interop/
update_schema/`. **Both directions green: Dir-1 (Rust reproduces Java) 3 Rust tests pass offline;
Dir-2 (`mvn ... verify`) 7/7 PASS.** Mutation-checked the verify gate (corrupting one identifier-field
set → exit 1 + FAIL line). **Key findings:** (a) Java rejects non-null INITIAL defaults on V2 metadata
(V3-only) — the two default scenarios use V3 base metadata to match Java's contract; the Rust side does
NOT enforce this V3-only rule (a latent parity gap, flagged in the report, out of scope here). (b) Java
resolves move targets by ORIGINAL name (renames live in `updates`, not name resolution) — the
`rename_and_move` scenario moves `email`, not `email_address`; Rust mirrors this. (c) evolved
last-column-id = `max(base.lastColumnId, evolved.highestFieldId)` (a delete never lowers it). Files
touched exactly the allowed set: new files under `dev/java-interop/**`, `crates/iceberg/testdata/
interop/**`, `crates/iceberg/tests/interop_update_schema.rs`; doc edits to GAP_MATRIX, Roadmap, todo,
lessons. NO Cargo.toml/lockfile/production-code edits.

#### Increment 5 — REVIEW (2026-06-07, Opus REVIEWER, DELEGATED)
Adversarially verified the 6 brief points against the Java source (`/tmp/iceberg-java-ref`) and by
running the full pipeline + mutation tests. Plan:
- [x] **Pt 1 (bidirectional, not tautological): CONFIRMED.** Dir-1 compares Rust-evolved vs the
      Java-WRITTEN `java_evolved.metadata.json` (recursive `StructType: PartialEq` over id/name/type/
      required/doc/default + identifier ids + current-schema-id + last-column-id); Dir-2 (`mvn verify`)
      reads the Rust-WRITTEN `rust_evolved.metadata.json` and compares against a fresh Java re-evolution.
      Proved non-trivial by mutation: (a) `count`→`kount` in a `java_evolved` fixture FAILS the Rust
      Dir-1 struct assertion (panic shows left=Rust vs right=Java, defaults included); (b) shrinking a
      `rust_evolved` identifier set FAILS `mvn verify` with exit 1. Neither side compares a file to
      itself; all 7 `rust_evolved` differ byte-wise from `java_evolved` (independent serializers).
- [x] **Pt 2 (all 7 scenarios both sides): CONFIRMED.** The "Dir-1 3 passed" is the test-FUNCTION count;
      `test_update_schema_interop_all_scenarios` LOOPS over all 7 (`SCENARIOS` const) — so all 7 run
      Dir-1, and `mvn verify` runs all 7 Dir-2 (7/7 PASS). No scenario silently skipped. Java
      `scenarios()` and Rust `apply_scenario_ops` mirror op-for-op.
- [x] **Pt 3 (nested level-order ids): CONFIRMED + strong.** Java `java_evolved` fixture pins map id=2,
      key-id=3, value-id=4, key struct 5–8, value struct 9–10 (level-order). The Rust
      `test_add_nested_struct_and_map_assigns_level_order_ids` asserts those EXACT ids. A depth-first
      regression would yield value-id=8 (key struct 4–7), which the `value_field.id == 4` + `key_ids ==
      [5,6,7,8]` assertions reject. Mentally broken: depth-first fails this test. No strengthening needed.
- [x] **Pt 4 (✅ flip — V2-default rule): GAP CONFIRMED, ✅ KEPT WITH HONEST CAVEAT + new test.** Java
      `Schema.checkCompatibility` (api/Schema.java:619, called on every add-schema build path,
      TableMetadata.java:1617) rejects a non-null `initialDefault` when `formatVersion < 3`
      (`DEFAULT_VALUES_MIN_FORMAT_VERSION = 3`), exact message "...non-null default (...) is not
      supported until v3" (Java test `TestSchema.testUnsupportedInitialDefault`). Rust has NO
      `check_compatibility` anywhere (`add_schema` in table_metadata_builder.rs does not call it;
      `validate_default` only checks type-convertibility). The existing unit test
      `test_add_required_column_with_default_succeeds_without_flag` ALREADY emits a non-null default on a
      V2 base — so Rust produces V2 metadata Java refuses to read. Decision: keep ✅ (the 7-scenario
      BIDIRECTIONAL proof is real; the hole is narrow + conditional — only V1/V2 + a column default, and
      it is a missing guard, not corruption of a valid op), but (a) rewrote the GAP_MATRIX note to state
      the divergence honestly, (b) added a test-pinned divergence regression
      (`test_v2_default_is_emitted_without_v3_guard_known_divergence`) so the hole is tracked, not
      silent, and (c) tracked the production fix (a `Schema::check_compatibility(format_version)` guard)
      as a follow-up — NOT fixed here (it belongs in `spec/`, beyond a one-line guard, out of reviewer
      scope).
- [x] **Pt 5 (deviation — catalog/Transaction path): CONFIRMED sound.** `Transaction::update_schema()`
      is `pub fn` returning `UpdateSchemaAction`; the test queues it via `ApplyTransactionAction::apply`
      and drives `Transaction::commit(&catalog)` — genuinely exercising `UpdateSchemaAction::commit()`
      through the catalog seam (strictly stronger: runs the optimistic-concurrency requirements too).
      `TransactionAction`/`commit` remain `pub(crate)` — visibility NOT widened. Only tracked-file edits
      are docs; no production `.rs` added/modified.
- [x] **Pt 6 (reproducibility): CONFIRMED.** Ran `mvn compile` (clean), `mvn generate` (7), Rust gen +
      Dir-1 (3/7), `mvn verify` (7/7) from a clean fixtures state — all green. `dev/java-interop/target/`
      is git-ignored (root `.gitignore` `target`); only `?? dev/java-interop/` is untracked (no build
      cruft staged). **Wart (tracked, not blocking):** regeneration churns `base`/`java_evolved` on the
      non-deterministic `table-uuid` + `last-updated-ms` + time-logs (structure identical) — noisy
      diffs on every `run.sh`. Harmless because the comparison is structural.
- [x] Verify gate: interop 3/3; lib ×2 = 1320/0 + the new divergence test (interop file); clippy -D
      warnings clean; fmt clean; `mvn` PASS table above.

**Review outcome (2026-06-07, Opus REVIEWER):** all 6 points adjudicated. Row stays **✅** with an
honest V2-default caveat in its note; +1 divergence-pinning test; production V3-default guard tracked as
a follow-up (below). Files touched: `crates/iceberg/tests/interop_update_schema.rs` (+1 test), GAP_MATRIX
(honest caveat), Roadmap (caveat), todo, lessons. No production `.rs`, no Cargo edits.

### Increment 6 — V3 initial-default guard (close the UpdateSchema parity hole, BUILDER Opus, 2026-06-07)
Close the only known `UpdateSchema` parity gap surfaced by the interop pilot: Rust emits Java-incompatible
metadata because `TableMetadataBuilder::add_schema` has no V3-only guard on column INITIAL defaults. Mirror
Java `Schema.checkCompatibility(Schema, formatVersion)` (`api/.../Schema.java:604-637`): for EVERY field
(incl. nested) with a non-null `initialDefault` when `formatVersion < DEFAULT_VALUES_MIN_FORMAT_VERSION (=3)`,
reject. Gate **initial_default ONLY — NOT write_default** (Java doesn't). Do NOT implement the broader V3-only
TYPE gate (`MIN_FORMAT_VERSIONS`) — flag it as a follow-up only.

**Java rule verified against source** (`api/src/main/java/org/apache/iceberg/Schema.java`):
- `checkCompatibility(Schema, int formatVersion)` (lines 604-637) iterates `schema.lazyIdToField().values()`
  (ALL fields, incl. nested). For each with `field.initialDefault() != null && formatVersion < 3`, records
  `"Invalid initial default for %s: non-null default (%s) is not supported until v%s"` (col name, value, 3);
  throws `IllegalStateException("Invalid schema for v%s:\n- %s", formatVersion, joined-problems)`.
- `DEFAULT_VALUES_MIN_FORMAT_VERSION = 3` (line 61). Called from the metadata builder's add-schema path
  (`addSchemaInternal`). It ALSO gates V3-only TYPES via `MIN_FORMAT_VERSIONS` (lines 64-70) — SEPARATE,
  broader parity item, NOT built here (flagged below).

**Choke point** (verified): `TableUpdate::AddSchema::apply` (catalog/mod.rs:611) calls
`builder.add_schema(schema)`, so wiring the guard into `TableMetadataBuilder::add_schema` covers the
UpdateSchema action's emitted `AddSchema`, CTAS, and every catalog commit — matching Java's
`addSchemaInternal`. The Rust analogue of Java `lazyIdToField()` is `Schema::field_id_to_fields()` (built by
`index_by_id`, a recursive `SchemaVisitor` walk over struct/list/map → reaches ALL nested fields); dotted
column names come from `field_id_to_name_map()`.

**Blast-radius (Risk-First) verified:** the guard fires ONLY for `initial_default.is_some() &&
format_version < V3`. `UpdateSchemaAction::commit` does NOT call `add_schema` (it just emits the `AddSchema`
update), so `run()`-based tests are unaffected; only tests that APPLY updates through the builder (or the
catalog/interop path) hit it. Across the crate, the only existing tests that drive a V2 initial-default
through the builder are `update_schema.rs::{test_emitted_schema_round_trips_defaults,
test_add_required_column_with_default_succeeds_without_flag}` + the interop divergence test. The two interop
default scenarios use a V3 base (confirmed in fixtures) so the round-trip is unaffected. V1/V2 serde fixtures
with defaults parse directly into `TableMetadata` (not via `add_schema`), so they are unaffected.

Plan:
- [x] Helper `Schema::check_compatibility(format_version)` in `spec/schema/mod.rs`: iterate
      `field_id_to_fields()`; for each field with `initial_default.is_some()` when
      `format_version < FormatVersion::V3`, push the Java-mirrored message (sorted by field id for
      determinism, like Java's TreeMap); `Err(ErrorKind::DataInvalid, "Invalid schema for v{N}:\n- {...}")`
      if any. Gates initial_default ONLY (write_default untouched). `DEFAULT_VALUES_MIN_FORMAT_VERSION = 3`
      named constant.
- [x] Wire into `TableMetadataBuilder::add_schema` (after field-name validation, before id assignment) —
      `schema.check_compatibility(self.metadata.format_version)?`.
- [x] Reconcile `update_schema.rs::test_add_required_column_with_default_succeeds_without_flag` → move base
      to V3 (the rule it targets — "required add WITH default needs no flag" — is legal on V3); add a V2
      sibling that asserts the defaulted add is REJECTED at apply time. Move
      `test_emitted_schema_round_trips_defaults` base to V3 too (it applies through the builder).
- [x] Flip `interop_update_schema.rs::test_v2_default_is_emitted_without_v3_guard_known_divergence` → assert
      the V2 defaulted add is now REJECTED with the guard's error (kind + "not supported until v3" substring).
- [x] New focused unit tests in `table_metadata_builder.rs`: V2 + top-level initial_default → rejected
      (kind + message substring); V3 + initial_default → allowed; V2 + initial_default on a NESTED field →
      rejected; V2 + NO default → unaffected (sanity); V2 + write_default-only → allowed (write_default NOT
      gated).
- [x] Docs: GAP_MATRIX `UpdateSchema` row → clean ✅ (note: V3 initial-default guard enforced, mirrors
      `Schema.checkCompatibility`, interop-proven both directions); close this follow-up; Roadmap; lessons.
- [x] Verify gate from repo root.

**Outcome (2026-06-07, Increment 6, BUILDER Opus):** the Increment-5 V3-guard follow-up is **CLOSED**.
Guard lives in `Schema::check_compatibility(format_version)` (`spec/schema/mod.rs`), wired into
`TableMetadataBuilder::add_schema` (the single choke point — `TableUpdate::AddSchema::apply` calls it, so
it covers the UpdateSchema action's emitted `AddSchema`, CTAS, and every catalog commit, matching Java's
`addSchemaInternal`). Message mirrors Java: per offending field `"Invalid initial default for {col}:
non-null default ({value:?}) is not supported until v3"` (ordered by field id, like Java's TreeMap) under
an `"Invalid schema for v{N}:"` header; `ErrorKind::DataInvalid`. Nested fields reached via
`field_id_to_fields()` (the recursive id→field index = Java `lazyIdToField()`); dotted names from
`field_id_to_name_map()`. Gates `initial_default` ONLY — `write_default` untouched (Java parity).
Reconciled tests: `update_schema.rs::test_add_required_column_with_default_succeeds_without_flag` →
`..._on_v3` (V3 base, applied through builder) + new `..._rejected_on_v2` sibling;
`test_emitted_schema_round_trips_defaults` moved to a V3 base; interop
`test_v2_default_is_emitted_without_v3_guard_known_divergence` → `test_v2_default_is_rejected_by_v3_guard`
(asserts rejection). New tests: 5 in `table_metadata_builder.rs` (V2 top-level reject / V3 allow / V2
nested reject / V2 no-default unaffected / V2 write-default-only allow) + 2 `Schema::check_compatibility`
unit tests in `schema/mod.rs`. Guard mutation-verified (disabling it fails the 3 rejection tests, incl. the
nested case). **Verify:** build clean; lib ×2 = 1328/0 both runs (was 1320 baseline → +8); interop 4/4
(7-scenario round-trip green — V3 default scenarios unaffected); clippy -D warnings clean; fmt clean. Java
`mvn` side NOT re-run — the guard does not change V3 output, so the committed V3 fixtures are unchanged.
Files touched exactly the allowed set: `spec/schema/mod.rs`, `spec/table_metadata_builder.rs`,
`transaction/update_schema.rs`, `tests/interop_update_schema.rs`, GAP_MATRIX, Roadmap, todo, lessons. No
Cargo.toml/lockfile/other edits. Row is now a clean ✅.

#### Increment 6 — REVIEW (2026-06-07, Opus REVIEWER, DELEGATED)
Adversarially verified points 1–5 against the Java source (`api/.../Schema.java` `checkCompatibility` +
`DEFAULT_VALUES_MIN_FORMAT_VERSION=3`) and `addSchemaInternal` (TableMetadata.java:1610-1652), and re-ran
the gate + two-direction mutation tests. No code GAPs found; one doc sharpen + lessons.
- [x] **Pt 1 (mirrors Java): CONFIRMED.** Gates `initial_default` ONLY — `write_default` untouched
      (Java checks `field.initialDefault() != null`); fires iff `format_version < V3`
      (`DEFAULT_VALUES_MIN_FORMAT_VERSION`); iterates ALL fields incl. nested via `field_id_to_fields()`
      (= Java `lazyIdToField()`); message carries dotted col name (`payload.flag`) + `{value:?}` + "is not
      supported until v3" under "Invalid schema for v{N}:" header (`FormatVersion` Display = `v2`/`v3`,
      matching Java verbatim). Both break-it cases pinned: nested-default-on-V2 → REJECTED
      (`test_add_schema_with_nested_initial_default_rejected_on_v2`); write-default-only-on-V2 → ALLOWED
      (`test_add_schema_with_write_default_only_allowed_on_v2`).
- [x] **Pt 2 (central choke point, no over/under-fire): CONFIRMED.** Guard in `add_schema` (table_metadata_
      builder.rs:649); `TableUpdate::AddSchema::apply` (catalog/mod.rs:611) → `add_schema`, and
      `add_current_schema`/`new()` → `add_schema`, so it covers the UpdateSchema action's emitted AddSchema,
      CTAS, and every catalog commit (exactly Java `addSchemaInternal`). Suite 1328/0 because no V1/V2 fixture
      carries an initial default — WHY confirmed by mutation: forcing the guard to early-`Ok` fails EXACTLY 4
      rejection tests and nothing else; over-broadening it to gate `write_default` fails EXACTLY 1 (the
      write-default-only test). READ/parse path UNAFFECTED: `TableMetadataV2 → TableMetadata` `TryFrom`
      constructs directly, never via `add_schema`/`check_compatibility` — only build/commit trips it (Java
      `checkCompatibility` is likewise in `addSchemaInternal`, not the parser).
- [x] **Pt 3 (reconciled tests honest): CONFIRMED.** `..._succeeds_without_flag_on_v3` drives the allow-path
      on a real `v3_table()` and APPLIES through the builder (`apply_updates`), asserting both defaults land
      — not skipped. `..._rejected_on_v2` applies the emitted AddSchema through a V2 builder and asserts
      kind=DataInvalid + "is not supported until v3" + col name; the interop `test_v2_default_is_rejected_by_v3
      _guard` drives a real `MemoryCatalog` register + `Transaction::commit` and asserts end-to-end rejection
      (not a shallow check). The default-bearing interop scenario uses a V3 base (verified in the fixture) so
      the 4/4 round-trip stays green.
- [x] **Pt 4 (✅ hole-free? + type gate tracked): ADJUDICATED — ✅ stands for the initial-default rule.**
      No other UpdateSchema case emits Java-incompatible metadata via DEFAULTS (write_default is not
      format-gated in Java, so Rust matching is correct). The V3-only TYPE gate (`MIN_FORMAT_VERSIONS`) IS a
      narrow residual hole that is LIVE TODAY for `timestamp_ns` (it exists in Rust + no type-version guard
      anywhere in `spec/`) — `add_column(timestamp_ns)` on V2 emits Java-rejected metadata now; the other
      four V3 types are unimplemented so future-only. The ✅ never claimed the type gate; properly TRACKED in
      todo (sharpened this review to say the `timestamp_ns` slice is live, not future). ✅ is justified.
- [x] **Pt 5 (message quality): CONFIRMED.** `{:?}` (Debug) on `Literal` is acceptable — `Literal` has no
      `Display`; Java renders via `toString()`; value rendering is expected to differ by language. All
      assertions key on STABLE substrings ("is not supported until v3", "Invalid schema for v2", the column
      name) — never the Debug value text. Actionable + non-brittle.
- [x] Verify (repo root, pinned nightly nightly-2025-10-27): build clean; lib ×2 = 1328/0 + 1328/0 (stable);
      interop 4/4; clippy -D warnings clean (forced rebuild, 12.8s); fmt --check clean. `mvn` NOT re-run (the
      guard does not alter V3 output; committed V3 fixtures unchanged — consistent with the builder's note).

**Review outcome (2026-06-07, Opus REVIEWER):** all 5 points adjudicated CONFIRMED; ✅ stands as a clean ✅
**for the column initial-default rule** (the named scope of this fix), with the V3-only TYPE gate the only
residual UpdateSchema parity item — narrowly LIVE for `timestamp_ns`, future for the other four V3 types,
and properly tracked. No code GAP found (guard mutation-verified load-bearing AND non-over-firing). Files
touched: `task/todo.md`, `task/lessons.md` only (doc sharpen + 2 lessons). No production `.rs`, no test, no
Cargo edits needed — the builder's implementation and tests are correct as shipped.

**FLAGGED follow-up (NOT built — scope control):** the V3-only TYPE gate (`MIN_FORMAT_VERSIONS`:
`timestamp_ns`/`variant`/`unknown`/`geometry`/`geography` require v3) lives in the SAME Java
`checkCompatibility` method but is a separate, broader parity item. Tracked below, not implemented here.

- [x] **Follow-up (CLOSED by Increment 7, 2026-06-07):** V3-only TYPE gate in `Schema::check_compatibility`
      mirroring Java `MIN_FORMAT_VERSIONS` (`Schema.java:64-70`) — reject `timestamp_ns`/`timestamptz_ns`
      fields when `format_version < 3`, message `"Invalid type for {col}: {type} is not supported until v{min}"`.
      The live `timestamp_ns`-on-V2 hole is now closed: the same `check_compatibility` method that gates
      initial-defaults also gates V3-only types in one pass. **Residual (genuinely future):** the other four
      Java `MIN_FORMAT_VERSIONS` types — `variant`/`unknown`/`geometry`/`geography` — are NOT representable in
      the Rust `Type`/`PrimitiveType` enums yet; each gets a one-line `min_format_version` arm
      (`PrimitiveType::Variant => Some(FormatVersion::V3)`, …) when the type lands. The helper is shaped for
      exactly that one-line addition.

### Increment 7 — V3-only TYPE gate (close the type-version follow-up above, BUILDER Opus, 2026-06-07)
Extend `Schema::check_compatibility(format_version)` to ALSO gate V3-only TYPES, fully mirroring Java
`Schema.checkCompatibility` — closes the live `add_column(timestamp_ns)`-on-V2 hole tracked above.

**Java rule verified against source** (`api/.../Schema.java:604-637`, re-read for this increment): in the
SAME `for (NestedField field : schema.lazyIdToField().values())` loop, BEFORE the initial-default check,
`Integer minFormatVersion = MIN_FORMAT_VERSIONS.get(field.type().typeId()); if (minFormatVersion != null
&& formatVersion < minFormatVersion) problems.put(fieldId, "Invalid type for %s: %s is not supported until
v%s")`. `MIN_FORMAT_VERSIONS = {TIMESTAMP_NANO:3, VARIANT:3, UNKNOWN:3, GEOMETRY:3, GEOGRAPHY:3}`. Both
checks accumulate into the SAME `TreeMap<fieldId,String>` and throw one combined `IllegalStateException`
`"Invalid schema for v%s:\n- %s"`. Of the five Java types, only `TIMESTAMP_NANO` is representable in Rust
today — `PrimitiveType::{TimestampNs, TimestamptzNs}` (both map to Java `TIMESTAMP_NANO`); `variant`/
`unknown`/`geometry`/`geography` are NOT in the Rust `Type`/`PrimitiveType` enums yet.

**Blast-radius (Risk-First) verified:** the type branch fires ONLY when a field's type min-version
exceeds the table `format_version` (i.e. an ns type on a <v3 builder/commit). Confirmed across the crate
that NO existing test builds a `<v3 TableMetadata` with a `timestamp_ns`/`timestamptz_ns` column via
`add_schema`/`TableMetadataBuilder::new`/`add_current_schema`/`from_table_creation`: the only ns usages
are transform/arrow/avro/manifest/equality-delete/datum tests that build a bare `Schema` (which does NOT
call `check_compatibility`) or match on the type — none flow through the metadata builder. So there are
NO existing tests to reconcile. The READ/parse path (`TableMetadataV2 → TableMetadata` `TryFrom`) never
calls `add_schema`, so V3 metadata fixtures with `timestamp_ns` parse unaffected.

Plan:
- [x] Added `fn min_format_version(ty: &Type) -> Option<FormatVersion>` returning `V3` for
      `PrimitiveType::{TimestampNs, TimestamptzNs}`, else `None`, with the comment noting the four
      not-yet-representable Java types get a one-line arm each when they land.
- [x] Restructured `check_compatibility` to iterate `field_id_to_fields()` ONCE, pushing the type problem
      (`"Invalid type for {col}: {type} is not supported until v{min}"`) when
      `min_format_version(type) > format_version` AND the existing initial-default problem into the SAME
      `Vec<(field_id, message)>`, stable-sorted by field id (type problem precedes default problem for a
      shared field), joined into the one `"Invalid schema for v{N}:"` error. Removed the v3 early-return so
      both rules share the same pass (the two minima coincide at v3, but the structure now mirrors Java).
- [x] +5 tests in `spec/schema/mod.rs`: V2 + `timestamp_ns` → rejected; V2 + `timestamptz_ns` → rejected;
      V3 + `timestamp_ns` → allowed; NESTED `timestamp_ns` on V2 → rejected (dotted `payload.captured_at`
      in message); V2 with BOTH a V3 type (field 2) AND a non-null initial_default (field 3) → BOTH
      problems in the single error, type-before-default by field id. Exact `DataInvalid` kind +
      message-substring asserts. Mutation-verified: forcing `min_format_version` to `None` fails exactly the
      4 type-gate tests, the 3 default/allow tests still pass.
- [x] Docs: GAP_MATRIX (`timestamp_ns` row + `UpdateSchema` row note now say BOTH rules enforced; the only
      open sub-item is variant/geo/unknown); Roadmap (snapshot + Phase-1 entry + headline); closed the
      follow-up above; appended a dated lesson.
- [x] Verify gate from repo root: build clean; lib ×2 = 1333/0 both runs (was 1328 → +5); interop 4/4;
      clippy -D warnings clean; fmt --check clean (one fmt reflow applied).

#### Increment 7 — REVIEW (2026-06-07, Opus REVIEWER, DELEGATED)
Adversarially verified points 1–5 against the Java source (`api/.../Schema.java:604-637` +
`MIN_FORMAT_VERSIONS`), ran the full gate + two-direction mutation tests. One GAP found and fixed.
- [x] **Pt 1 (mirrors Java's type gate): CONFIRMED.** `min_format_version` returns `Some(V3)` for BOTH
      `TimestampNs` and `TimestamptzNs` (= Java `TIMESTAMP_NANO`), `None` otherwise; fires iff
      `format_version < min` (`FormatVersion` Ord by u8 repr, so V1/V2 < V3); message `"Invalid type for
      {col}: {type} is not supported until v{min}"` with `FormatVersion` Display = `v3` (verbatim Java
      `v%s`), dotted nested name via `name_by_field_id`. Iterates ALL fields via `field_id_to_fields()`
      (= Java `lazyIdToField()`). Break-it cases all pinned: nested ns on V1/V2 → rejected; `timestamptz_ns`
      on V2 → rejected; a v2 type on V2 → `None` → unaffected. Type gate mutation-verified load-bearing
      (forcing `min_format_version`→`None` fails exactly the 5 type-dependent tests, leaves the 3
      default/allow tests green).
- [x] **Pt 2 (no silent breakage; ZERO-tests claim REAL): CONFIRMED independently.** Crate-wide grep of
      `TimestampNs`/`TimestamptzNs`/`Nanosecond` across src + tests + glue/hms/datafusion/integration_tests:
      every usage is a type-conversion fn (`type_to_string`), a transform/arrow/avro/manifest/datum unit
      test building a bare `Schema`/`Datum`, or datafusion predicate-pushdown (`Datum::timestamp_nanos`) —
      NONE routes an ns column through `add_schema`/`from_table_creation`/`create_table` on a <v3 base. The
      one builder-path test helper (glue `create_metadata` → `from_table_creation`, defaults V2) is never
      called with an ns schema. Full lib suite 1333/0 ×2; all-targets unit+interop green.
- [x] **Pt 3 (TreeMap-vs-Vec): ADJUDICATED — KEEP both-report `Vec`.** Java's `TreeMap` last-wins
      collapses a single-field-both to ONE (default) line; Rust's `Vec` keeps BOTH. Decision: keep it
      (accept/reject identical, strictly more informative, message text already language-divergent, case
      vanishingly narrow; cross-field order already matches Java). **GAP FOUND + FIXED:** the builder kept
      the `Vec` and documented the divergence in prose but only added the CROSS-field test (two distinct
      ids — cannot collide, cannot distinguish the designs); the single-field-both case was unpinned. Added
      `test_check_compatibility_single_field_both_type_and_default_reports_both_lines`. Mutation-verified:
      emulating Java last-wins (dedup to BTreeMap) fails EXACTLY the new test, leaves the cross-field test
      green.
- [x] **Pt 4 (build-path-only, no over-fire): CONFIRMED.** Only production call site of
      `Schema::check_compatibility` is `table_metadata_builder.rs:649` (inside `add_schema`); the parse
      path `TryFrom<TableMetadataV2> for TableMetadata` constructs `TableMetadata { .. }` directly and
      never calls it — reading existing V3 `timestamp_ns` metadata is unaffected. v2-and-below schemas
      without V3 types/defaults are completely untouched (suite is 1333/0 because no <v3 fixture carries an
      ns type or initial default through the builder — confirmed by the Pt 2 audit and the mutation gate).
- [x] **Pt 5 (✅ honesty + tracking): CONFIRMED.** With both halves landed (defaults + types),
      `UpdateSchema`/`timestamp_ns` are hole-free for the representable compatibility class. GAP_MATRIX is
      accurate: `timestamp_ns` ✅ with the format-version gate; `UpdateSchema` ✅ noting both rules; the
      ONLY open residual is `variant`/`unknown`/`geometry`/`geography` (not representable in the Rust
      `Type`/`PrimitiveType` enums — each a one-line `min_format_version` arm when it lands), properly
      tracked in the follow-up and the matrix.
- [x] Fixed a stale comment in `table_metadata_builder.rs:644` (said "currently the column initial-default
      rule" — now also covers the V3-only type rule).
- [x] Verify (repo root, pinned nightly): build clean; lib ×2 = 1334/0 + 1334/0 (was 1333 → +1 review
      test); interop 4/4; clippy -D warnings clean; fmt --check clean. All-targets: only the 5 pre-existing
      `tokio::main` doctest-compile failures (reproduced with my changes stashed — environmental, not mine).

**Review outcome (2026-06-07, Opus REVIEWER):** all 5 points adjudicated; the TreeMap divergence decided
(keep both-report) + the missing single-field-both test added (mutation-verified both directions); the
type gate confirmed load-bearing and build-path-only; the ZERO-tests claim independently re-confirmed; ✅
stands with the residual (variant/geo/unknown) correctly tracked. Files touched: `spec/schema/mod.rs`
(+1 test), `spec/table_metadata_builder.rs` (stale comment), `task/todo.md`, `task/lessons.md`. No Cargo
edits, no commit.

**Outcome (2026-06-07, Increment 7, BUILDER Opus):** the V3-only TYPE gate follow-up is **CLOSED**.
`Schema::check_compatibility` now fully mirrors Java `Schema.checkCompatibility` — one pass over
`field_id_to_fields()` (all fields incl. nested) records a type problem (`MIN_FORMAT_VERSIONS`) and/or an
initial-default problem (`DEFAULT_VALUES_MIN_FORMAT_VERSION`) per field, accumulated into a single
`"Invalid schema for v{N}:"` error ordered by field id (Java's TreeMap). Helper `fn min_format_version(ty:
&Type) -> Option<FormatVersion>` returns `V3` for `PrimitiveType::{TimestampNs, TimestamptzNs}` (Java
`TIMESTAMP_NANO`); the four Java types not representable in Rust (`variant`/`unknown`/`geometry`/`geography`)
get a one-line arm each when they land. The guard stays on the BUILD path (`add_schema`) — the parse path
(`TableMetadataV2 → TableMetadata` `TryFrom`) never calls it, so reading existing V3 `timestamp_ns` metadata
is unaffected. **No existing tests needed reconciling** — a crate-wide audit confirmed no test builds a
`<v3 TableMetadata` with a `timestamp_ns`/`timestamptz_ns` column via `add_schema`/`from_table_creation`
(the ns usages are transform/arrow/avro/manifest/datum tests that build bare `Schema`s, which do not call
`check_compatibility`). Files touched exactly the allowed set: `crates/iceberg/src/spec/schema/mod.rs`
(helper + restructured method + 5 tests), GAP_MATRIX, Roadmap, todo, lessons. No production code outside
`check_compatibility`/its helper; no Cargo/lockfile edits. An Opus REVIEWER verifies next.
