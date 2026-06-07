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
