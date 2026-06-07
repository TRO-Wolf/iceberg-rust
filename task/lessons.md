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

# Lessons

Accumulated DO / DO NOT lessons. The operating manuals ([skills/](../skills/)) require reading this
file **in full at the start of every session**, and appending to it after **any** correction from
the user.

How to use it (see the manuals' §2):

- After any correction, append a **date-stamped** entry immediately.
- Write each as a concrete **DO** or **DO NOT** statement with the *why* and how to apply it.
- Supersede an outdated rule with a dated note (`_superseded YYYY-MM-DD: see ..._`) rather than
  editing the original in place.

---

<!-- Newest entries at the bottom. Example shape:

### YYYY-MM-DD
- **DO** carry context on every fallible Rust call (`.with_context(...)` / `.expect("msg")`).
  *Why:* a bare `.unwrap()` panic gives the operator no cause from logs alone.
- **DO NOT** edit upstream crate files to land a fork feature when an additive module would do.
  *Why:* it makes the next upstream merge conflict-prone. Prefer additive changes.
-->

### 2026-06-07
- **DO** run a new test module against the *full* parallel lib suite (`cargo test -p iceberg --lib`),
  not just its own filter, before declaring green. *Why:* adding ManageSnapshots' 12 tests increased
  parallel load and surfaced a latent flaky assertion in an unrelated test
  (`catalog/memory/catalog.rs::test_update_table`). The new code was correct; a pre-existing race only
  became visible under load. A filtered run (`... transaction::manage_snapshots`) hid it.
- **DO** treat strict `<` comparisons on millisecond wall-clock timestamps as flaky, and assert `<=`
  (plus a structural check that the change happened, e.g. metadata-log growth). *Why:* two metadata
  versions can legitimately share a `last_updated_ms`. Fixed `test_update_table` accordingly — a
  legitimate fix since we own the fork, and upstreamable.
- **Pattern: adding a transaction action.** Mirror `transaction/sort_order.rs` — builder struct that
  records intent, `#[async_trait] impl TransactionAction { commit(self: Arc<Self>, &Table) ->
  Result<ActionCommit> }` resolving against `table.metadata()` and returning
  `ActionCommit::new(updates, requirements)`; add `mod x;` + `use ...XAction;` + a `pub fn x()` ctor in
  `transaction/mod.rs`. The `TransactionAction` trait must be `use`d in tests to call `.commit()`.

### 2026-06-07 (multi-agent review remediation)
- **DO** verify a parity contract against the Java *source*, not intuition, before implementing the
  Rust side. *Why:* the first `fast_forward` cut wrongly required `to` to be a branch and rejected an
  absent `from`; Java `UpdateSnapshotReferencesOperation.replaceBranch` only requires `from` to be a
  branch and **auto-creates** an absent `from`. Read `/tmp/iceberg-java-ref` for the exact precondition
  checks (`Preconditions.checkArgument(...)`) and the early-return no-op cases.
- **DO** emit only refs whose final state differs from their original (no-op suppression) in
  metadata-mutating commit actions. *Why:* create-then-remove, fast-forward-to-same, replace-to-same
  should produce zero `TableUpdate`s — matches Java's "only changed refs" and keeps the commit
  idempotent at the catalog layer. Compare `metadata.refs.get(name) == working.get(name)` (both
  `SnapshotReference: PartialEq`).
- **DO** flag an agent finding you could NOT confirm rather than acting on it. *Why:* a reviewer
  claimed Java rejects non-positive retention (`> 0`); a grep of `SnapshotRef.java` showed no such
  `checkArgument`. Left unimplemented and tracked in `task/todo.md` as a follow-up — don't add
  validation that may diverge from Java on an unverified claim.
- **DO** test pre-existing-ref paths with a fixture that actually contains those refs. *Why:* the
  straight-line `make_v2_table()` has only `main`, so remove/replace/rollback-non-ancestor were
  untestable; build a forked fixture (`add_snapshot` a sibling + `set_ref` a branch/tag). `add_snapshot`
  validates the timestamp against the metadata's `last-updated-ms` (not just snapshot timestamps) — set
  the grafted snapshot's `timestamp_ms` after it.
- **DO** keep summary/headline sections in sync with the detail table when flipping a status. *Why:*
  the GAP_MATRIX + Roadmap "Headline gaps" kept listing `timestamp_ns`/column-defaults as missing after
  the matrix body flipped them to ✅. When you change a row's status, grep for the capability name and
  reconcile every mention.

### 2026-06-07 (UpdatePartitionSpec review remediation)
- **DO** replicate Java's `recycleOrCreatePartitionField` in the partition-spec *action*, not just lean
  on `TableMetadataBuilder.reuse_partition_field_ids`. *Why:* the builder recycles a historical field's
  *id* (matching on `(source_id, transform)`) but NOT its *name* — so an add with no explicit name came
  out with the generated default name (`y_bucket_8`) instead of the historical name (`y_bucket`) Java
  reuses. Java returns the whole historical field when `name == null || field.name().equals(name)`. Fix:
  the action searches `metadata.partition_specs_iter()` and, on match, sets BOTH the recycled `field_id`
  and the historical `name` on the `UnboundPartitionField`; the builder's id-recycling then no-ops on the
  pre-set id. Pin it with a multi-spec fixture whose historical field has a *custom* name (id-only checks
  pass even when the name is wrong).
- **DO** derive a transaction action's `TableRequirement`s from the updates it actually emits, the way
  Java's `UpdateRequirements` visitor does. *Why:* `UpdatePartitionSpec` emitted both
  `LastAssignedPartitionIdMatch` AND `DefaultSpecIdMatch` unconditionally, but under `add_non_default_spec`
  there is no `SetDefaultSpec` update, so Java attaches only `AssertLastAssignedPartitionId`. Emitting the
  default-spec guard anyway over-constrains the commit. Gate each guard on the update that induces it
  (`AddSpec` ⇒ last-assigned-partition-id; `SetDefaultSpec` ⇒ default-spec-id).
- **DO** prove a metadata-mutating action end-to-end by driving its emitted `TableUpdate`s through
  `TableMetadataBuilder` (`update.apply(builder)`), not just by inspecting the unbound `apply()` shape.
  *Why:* the unbound spec test never exercises the metadata layer's spec dedup, `LAST_ADDED` resolution,
  or default-spec switch — a no-op evolution (remove-then-re-add the same field) must dedup back to the
  existing spec id and NOT advance `last_partition_id`; only a round-trip catches a regression there.

### 2026-06-07 (UpdateSchema review remediation, Opus)
- **DO** assign fresh nested field ids LEVEL-ORDER (pre-order) when mirroring Java `TypeUtil.assignFreshIds`
  — it runs as a `CustomOrderSchemaVisitor` whose `struct` assigns ALL immediate field ids before the
  child futures evaluate, and whose `map` assigns key-id then value-id before either future. A natural
  depth-first Rust walk (assign this field's id, then immediately recurse its type) diverges the instant a
  nested field has a following sibling: for `map<struct,struct>` it yields value-id = key-id + (size of key
  subtree) instead of key-id + 1. *Why:* the ids are observable and Java pins them
  (`TestSchemaUpdate.testAddNestedMapOfStructs`: key=3, value=4, then 5..8, 9..10); divergent ids break
  interop and round-trip parity. A `last_column_id`-only assertion CANNOT catch this — assert per-field ids.
- **DO NOT** gate a union-by-name (or any Java-visitor port) type change on "apply only if it's a legal
  promotion." *Why:* Java `UnionByNameVisitor.updateColumn` computes `needsTypeUpdate = !isIgnorableType
  Update` and calls `api.updateColumn` UNCONDITIONALLY for a non-ignorable change, so an *incompatible*
  change reaches `updateColumn`'s guard and throws "Cannot change column type". Skipping the call (the old
  Rust behavior) silently drops a change Java rejects. `isIgnorableTypeUpdate` is: existing-primitive →
  ignorable iff incoming is a primitive that is a *narrowing* (`isPromotionAllowed(incoming, existing)`,
  reversed); existing-complex → ignorable iff incoming is also complex (the recursion handles inner edits).
- **DO** recurse a union/partner visitor THROUGH list elements and map key+values, not just structs.
  *Why:* Java reaches `<path>.element` / `<path>.key` / `<path>.value` via `PartnerIdByNameAccessors`, so a
  new field nested inside an existing `list<struct>`/`map<.,struct>` is added and an element/value type
  change is validated. A struct-only recursion silently no-ops those.
- **DO** reject case-insensitive lowercase-name collisions when building a Rust schema index, mirroring
  Java `TypeUtil.indexByLowerCaseName` ("Cannot build lower case index: a and b collide"). *Why:* Rust's
  `lowercase_name_to_id = name_to_id.iter().map(to_lowercase).collect()` silently overwrites colliding keys,
  so a case-insensitive add of `DATA` after `data` builds a corrupt schema; Java throws. Report the smaller
  field-id name first so the message is deterministic despite `HashMap` order (matches Java's first-visited
  ordering for sequential adds).
- **DO NOT** write a test that calls `.expect_err()` directly on a `commit()` future whose `Ok` type is
  `ActionCommit` — `ActionCommit` is not `Debug`. Insert `.map(drop)` before `.expect_err("…")` (or assert
  on `result.is_err()` + re-extract the error). *Why:* `Result::expect_err` requires `T: Debug`.

### 2026-06-07 (Increment 4 — ManageSnapshots tail + UpdateSchema defaults, BUILDER Opus)
- **DO** locate `rollbackToTime` in `core/SetSnapshotOperation.java` (`SnapshotManager.rollbackToTime`
  delegates to `transaction.setBranchSnapshot().rollbackToTime`). The rule is
  `findLatestAncestorOlderThan(base, ts)`: walk `SnapshotUtil.ancestorIds(currentSnapshot)` (the MAIN
  parent chain — the existing `is_ancestor_of` walk) and pick the ancestor with the MAX `timestampMillis`
  that is **strictly `<` ts**; error "Cannot roll back, no valid snapshot older than: {ts}" if none. *Why
  the strict `<` matters:* ts == current's own timestamp does NOT select current (it picks the next-older
  ancestor); ts > current selects current (a no-op, suppressed at emit). A `<=` would wrongly keep current
  on the exact-equal boundary. Non-ancestor siblings are never visited by the parent-chain walk, so they
  can never be chosen — no extra guard needed.
- **DO** find the retention positivity checks in `api/SnapshotRef.java` **Builder setters** (lines
  ~154–177), NOT in the `SnapshotRef` ctor — that is why an earlier grep "of `SnapshotRef.java`" missed
  them (it likely scanned the ctor/equals region). Each setter has `Preconditions.checkArgument(value ==
  null || value > 0, "…")` with these EXACT messages: "Min snapshots to keep must be greater than 0",
  "Max snapshot age must be greater than 0 ms" (note the trailing " ms"), "Max reference age must be
  greater than 0" (no unit). `null` is allowed (clears the field); our `set_*` API always sets a concrete
  value, so only the `<= 0` case occurs. *Supersedes the 2026-06-07 "unverified retention `>0`" follow-up
  — it IS in Java, just in the Builder.*
- **DO** allow a **required column add WITH a default WITHOUT `allow_incompatible_changes`** —
  `core/SchemaUpdate.internalAddColumn` gates on `defaultValue != null || isOptional ||
  allowIncompatibleChanges` (line ~160). The default backfills existing rows, so it is a compatible change.
  A required add WITHOUT a default to a non-empty schema stays incompatible. *Why it's easy to get wrong:*
  the obvious guard `required && !flag` rejects the legal defaulted case; the correct guard is `required &&
  default.is_none() && !flag`.
- **DO** set BOTH `initial_default` AND `write_default` on an *added* column (Java `addColumn` calls
  `.withInitialDefault(default).withWriteDefault(default)`), but on `updateColumnDefault` set ONLY
  `write_default` (Java comment: "write default is always set and initial default is only set if the field
  requires one"). *Why:* the initial default is the existing-row backfill (fixed at add time); a later
  default change only affects future writes.
- **DO** validate a Rust `Literal` default against the column type via
  `literal.clone().try_into_json(&field_type)` — the Rust `NestedField::with_initial_default` /
  `with_write_default` setters do NOT validate, but the serde `From<NestedField>` path calls
  `try_into_json` with a panicking `.expect`. Running `try_into_json` at add/update time is the canonical
  compatibility check (mirrors Java `castDefault`'s `defaultValue.to(type)`): a non-primitive type rejects
  with "Invalid default value… (must be null)" and an incompatible primitive rejects with "Cannot cast
  default value to…". Passing it guarantees no later serialization panic. *Note:* `Type` has `is_primitive`
  / `is_struct` / `is_nested` but NOT `is_list` / `is_map`; use `!is_primitive()` for "is nested".
- **DO** reclassify `cherrypick` as **Phase-2-gated**, not a metadata op: Java `SnapshotManager.cherrypick`
  → `transaction.cherryPick()` whose operation **extends `MergingSnapshotProducer`** and replays data
  files. It belongs to the write engine (Phase 2), so it is out of scope for the metadata-only
  `ManageSnapshots` surface even though Java co-locates it on the same API.

### 2026-06-07 (Increment 4 — REVIEWER Opus)
- **DO** add a test for EVERY conditional relaxation branch, not just its happy/sad headline. *Why:* the
  Increment-4 builder shipped `add_*_with_default` (defaulted required add) and `update_column_default`
  (write-default only) but left the *interaction* branch `is_defaulted_add` (an added field's
  `initial_default.is_some()` lets `require_column` skip the incompatible-change gate) UNTESTED. Java pins it
  with `testAddColumnWithDefaultToRequiredColumn` (defaulted add → require succeeds) AND
  `testAddColumnWithUpdateColumnDefaultToRequiredColumn` (add + updateColumnDefault → require FAILS, because
  updateColumnDefault sets only write_default, not initial_default). The two together are the only thing that
  distinguishes the two default-setting paths at the require boundary; mutation-verify by swapping
  `initial_default` → `write_default` (negative test catches it) and by forcing `is_defaulted_add = false`
  (positive test catches it). When porting a Java API, grep its test file for the *combinations* of the new
  methods, not just each method alone.
- **DO** recognize when a same-call validation makes a downstream `.expect` panic-proof, and say so instead
  of hunting for a stronger check. *Why:* `validate_default` runs `Literal::try_into_json(&field_type)` —
  the EXACT call the serde `From<NestedField>` path (datatypes.rs:591-592) later runs under a panicking
  `.expect`. Identical input + identical type ⇒ if validation passes, the serde call returns `Ok`, so the
  panic is unreachable. That is the load-bearing safety property; the residual Rust-vs-Java parity gaps
  (the `(_, UInt128|Binary)` wildcard arms accept a UUID/binary literal for any primitive = too lenient; the
  strict primitive-pair match rejects an int literal on a long column = too strict) only bite a caller who
  hand-builds a type-mismatched `Literal`, which the strongly-typed Rust `Literal` makes unnatural. Track as
  a parity note, do not "fix" by diverging from the serde contract.

### 2026-06-07 (Increment 5 — UpdateSchema interop pilot, BUILDER Opus)
- **DO** drive an interop test's evolution through the PUBLIC API (`MemoryCatalog` register-table +
  `Transaction` + `ApplyTransactionAction::apply` + `Transaction::commit`), not the action's `commit()`
  directly. *Why:* `TransactionAction::commit` is `pub(crate)` and the `transaction::update_schema` module
  is private, so an external integration test in `crates/iceberg/tests/` CANNOT name `UpdateSchemaAction`
  or call `.commit()` (only in-crate unit tests can). The catalog path is the only public way to apply an
  action offline and read the evolved metadata — and it is strictly stronger (it also runs the
  optimistic-concurrency `TableRequirement` checks). Do NOT widen production visibility to make the test
  compile; route through the catalog instead.
- **DO** use a `MemoryCatalog` built with `with_storage_factory(Arc::new(LocalFsStorageFactory))` over a
  `tempfile::tempdir()` warehouse when an interop test must register a table from an EXACT pre-written
  metadata file. *Why:* the default `MemoryCatalog` storage is in-memory (`MemoryStorageFactory`) and
  `register_table` reads the metadata via the catalog's OWN FileIO — it never sees a file you wrote with a
  separate FileIO. Two gotchas the catalog enforces: the metadata file must live under a `metadata/`
  subdirectory of the table location, AND be named `<version>-<uuid>.metadata.json` (e.g.
  `00000-00000000-0000-0000-0000-000000000000.metadata.json`) — a bare `base.metadata.json` fails with
  "Invalid metadata file name format". (`crates/iceberg/src/catalog/metadata_location.rs`.)
- **DO** generate Java interop fixtures at format version 3 for any scenario with a column INITIAL default.
  *Why:* Java `iceberg-core` 1.10.0 rejects a non-null initial default on V2 metadata ("non-null default
  (...) is not supported until v3") at `TableMetadata` build time. The Rust side does NOT enforce this
  V3-only rule (`validate_default` only checks type-convertibility), so it would happily emit a default on
  V2 metadata that Java then refuses to read — a real latent parity gap the interop test surfaces. Match
  Java's contract (V3 base for defaults) and flag the missing Rust V3-gate as a tracked parity note.
- **DO** reference renamed columns by their ORIGINAL name in a move within the SAME UpdateSchema sequence.
  *Why:* Java `SchemaUpdate.findForMove` resolves names against `addedNameToId` then the ORIGINAL schema —
  a rename is recorded in `updates` (keyed by field id) and is NOT visible to name resolution until the
  schema is rebuilt. So `renameColumn("email","email_address").moveFirst("email")` is correct;
  `moveFirst("email_address")` throws "Cannot move missing column". The Rust `find_for_move` mirrors this
  exactly — a scenario that moved by the new name was the test's bug, not the code's.
- **DO** compute an interop fixture's evolved last-column-id as `max(base.lastColumnId,
  evolved.highestFieldId)`, never the evolved schema's `highestFieldId()` alone. *Why:* a delete lowers
  `highestFieldId()` but Iceberg NEVER reuses or lowers `lastColumnId` (ids are permanently retired);
  `TableMetadata.buildFrom(base).setCurrentSchema(evolved, evolved.highestFieldId())` throws "Invalid last
  column ID: 2 < 3 (previous last column ID)" on a delete scenario. Java's `addSchema` does the same
  `Math.max` internally.
- **DO** compare interop metadata by PARSING both files into the Rust model and asserting
  `Schema::as_struct() == other.as_struct()` (+ explicit identifier ids / current-schema-id /
  last-column-id), NOT raw JSON bytes. *Why:* `StructType: PartialEq` recurses field id + name + type +
  required + doc + default in one shot, which IS the field-id-level identity contract; Jackson and
  `serde_json` differ in key order and whitespace, so byte-equality would be both too strict (false
  failures) and beside the point. Add a scenario-specific exact-id assertion for the nested level-order
  case — a `last_column_id`-only check cannot catch an interior-id divergence that lands on the same max.

### 2026-06-07 (Increment 5 — UpdateSchema interop pilot, REVIEWER Opus)
- **DO** mutation-test BOTH directions of an interop harness before trusting a ✅, by corrupting the
  fixture each direction reads and confirming the assertion fires. *Why:* an interop test that compares a
  file against itself, or whose "verify" recomputes the same value it checks, passes tautologically.
  Confirmed Dir-1 by editing a Java-written `java_evolved` field name (`count`→`kount`) → the Rust struct
  assertion fails (panic shows left=Rust vs right=Java); Dir-2 by shrinking a Rust-written identifier set
  → `mvn verify` exits 1. Both `rust_evolved` and `java_evolved` differ byte-wise (independent
  serializers), so neither side compares a file to itself.
- **DO** distinguish a test-FUNCTION count from a SCENARIO count when a reviewer flags "3 passed but 7
  scenarios." *Why:* the UpdateSchema Dir-1 reports "3 tests" but one of them
  (`test_update_schema_interop_all_scenarios`) LOOPS over all 7 in a `SCENARIOS` const — every scenario
  IS exercised. Read the loop, don't infer coverage from the headline number.
- **DO** treat "Rust emits metadata Java would REJECT" as a real parity hole even when the interop test is
  green — green only proves the cases the fixtures EXERCISE. *Why:* the UpdateSchema pilot generates the
  two default-bearing scenarios at V3 because Java `Schema.checkCompatibility` (`api/Schema.java:619`,
  called via `TableMetadata$Builder.addSchemaInternal`) rejects a non-null `initial_default` on
  `format_version < 3`. Rust has NO such guard (`TableMetadataBuilder::add_schema` never calls
  `check_compatibility`; `validate_default` only checks type-convertibility), so a defaulted add on a V2
  table SUCCEEDS and emits Java-unreadable metadata — and the interop test sidesteps it by using V3. Pin
  the hole with a divergence test (`test_v2_default_is_emitted_without_v3_guard_known_divergence`) and an
  HONEST GAP_MATRIX caveat; do NOT let a "V3 base for defaults" fixture note quietly imply the Rust side
  enforces the rule. ✅ is defensible only because the hole is narrow + conditional (V1/V2 + a column
  default) and every other surface is bidirectionally interop-proven; a wider or unconditional hole would
  force 🟡.
- **DO** verify a TEST-ONLY oracle never enters the Cargo graph and its build dir is git-ignored. *Why:*
  `dev/java-interop/` is Java driven by Maven; `cargo build`/`cargo test` must never invoke it, and its
  `target/` must not be committed. Confirmed: `git check-ignore dev/java-interop/target/classes` →
  ignored (root `.gitignore` `target`); `git status` shows only `?? dev/java-interop/` (no staged
  cruft); `TransactionAction`/`commit` stayed `pub(crate)` (visibility not widened to make the test
  compile — the public `Transaction::update_schema()` + catalog-commit path is used instead).
- **DO** expect non-deterministic interop-fixture churn from Java's `newTableMetadata` (`table-uuid`,
  `last-updated-ms`, time-logs regenerate every run) and confirm a regen is STRUCTURALLY identical, not
  byte-identical. *Why:* `run.sh` re-running mutates the committed `base`/`java_evolved` on those fields
  only; the structural comparison the tests assert is unaffected, but it produces noisy diffs. Track it;
  don't mistake it for a logic regression.

### 2026-06-07 (Increment 6 — V3 initial-default guard, BUILDER Opus)
- **DO** wire a Java-parity schema guard into `TableMetadataBuilder::add_schema`, not into the transaction
  action's `commit()`. *Why:* `UpdateSchemaAction::commit` only EMITS a `TableUpdate::AddSchema` (it does
  not call `add_schema`); the guard belongs at the single choke point every add-schema path flows through —
  `TableUpdate::AddSchema::apply` (catalog/mod.rs) calls `builder.add_schema(schema)`, so a guard there
  covers the UpdateSchema action, CTAS, and every catalog commit at once, exactly mirroring Java's
  `TableMetadata$Builder.addSchemaInternal` calling `Schema.checkCompatibility`. A guard in `commit()` would
  miss CTAS/raw-builder paths.
- **CONSEQUENCE: a guard in `add_schema` only fires on the APPLY path, not on `run()`-style action tests.**
  *Why:* tests that call `action.commit()` and merely inspect the emitted `AddSchema` updates (the `run()`
  helper in `update_schema.rs`) never reach the metadata builder, so they're unaffected by the guard; ONLY
  tests that drive updates through `TableMetadataBuilder` (the `apply_updates()` helper, or a full catalog
  commit) hit it. Before adding such a guard, grep the test file for which tests APPLY vs. merely INSPECT —
  the blast radius is the apply-path subset, not every test that uses the relevant feature. Here only
  `test_emitted_schema_round_trips_defaults` (and the catalog-driven interop tests) needed reconciling
  beyond the one the brief named.
- **DO** reach ALL fields (incl. nested struct/list/map descendants) via `Schema::field_id_to_fields()` —
  the recursive id→field index built by `index_by_id` (a `SchemaVisitor` walk) — when mirroring a Java rule
  that iterates `schema.lazyIdToField().values()`. *Why:* a hand-rolled top-level-only loop silently lets a
  default (or any per-field violation) buried inside a nested struct through. `field_id_to_name_map()` gives
  the dotted column name (`payload.flag`) for the error, matching Java's `findColumnName`. Pin the nested
  reach with a test asserting the dotted path appears in the rejection message — a top-level-only guard can
  never produce it.
- **DO** gate `initial_default` ONLY, never `write_default`, when mirroring Java
  `Schema.checkCompatibility`'s `DEFAULT_VALUES_MIN_FORMAT_VERSION` check (`api/Schema.java:619`). *Why:* Java
  checks `field.initialDefault() != null` only — a write default affects future writes, not how existing
  rows are read, so it is legal on v1/v2. A test that adds a v2 schema with only a `write_default` must SUCCEED
  (`test_add_schema_with_write_default_only_allowed_on_v2`); gating it would wrongly reject legal metadata.
- **DO** order accumulated per-field problem messages by field id before joining them, mirroring Java's
  `Maps.newTreeMap()`. *Why:* iterating a Rust `HashMap` yields nondeterministic order, so a multi-field
  violation would produce a flaky error message; Java accumulates into a `TreeMap` keyed by field id for a
  stable order. Sort by id, then join with the Java separator (`"\n- "`).
- **DO** use `{:?}` (Debug) to render a Rust `Literal` in an error message — it has no `Display` impl. *Why:*
  Java renders the offending default via `Literal.toString()`; the Rust `Literal` only derives `Debug`, so
  `{:?}` (e.g. `Primitive(Long(7))`) is the faithful analogue. The message STRUCTURE (col name, value,
  "not supported until v3") mirrors Java; the value rendering differs by language and that is expected — assert
  on the substring `"is not supported until v3"` + the column name, not the exact value text.
- **DO** flag (not build) a co-located-but-broader parity item. *Why:* Java's `checkCompatibility` gates BOTH
  V3-only initial-defaults AND V3-only TYPES (`MIN_FORMAT_VERSIONS`: `timestamp_ns`/`variant`/`unknown`/
  `geometry`/`geography`) in one method. The type gate is tied to landing those V3 types and is a separate
  capability; implementing it here would balloon the scope. Left `Schema::check_compatibility` structured so
  the type gate slots into the same method later, and tracked it as a follow-up in `task/todo.md` — don't
  build adjacent scope just because the Java method co-locates it.

### 2026-06-07 (Increment 6 — V3 initial-default guard, REVIEWER Opus)
- **DO** check whether a "tracked future" gate is ALREADY live for a partially-landed feature before
  characterizing it as future-only. *Why:* the Increment-6 follow-up called the V3-only TYPE gate
  (`MIN_FORMAT_VERSIONS`) "tied to landing those V3 types," but `PrimitiveType::TimestampNs` is ALREADY in
  Rust (GAP_MATRIX "V3 types: timestamp_ns" ✅) and no type-version guard exists anywhere in `spec/` — so
  `UpdateSchema.add_column("ts", timestamp_ns)` on a V2 table emits Java-rejected metadata TODAY ("Invalid
  type for ts: ... is not supported until v3"). The other four V3 types are unimplemented, so for them the
  gate really is future. Sharpened the follow-up to say the `timestamp_ns` slice is LIVE, not future, so the
  next agent doesn't deprioritize it. The `UpdateSchema` ✅ stands for the **initial-default** rule only (it
  never claimed the type gate); the type gate is a distinct, narrowly-live residual.
- **DO** mutation-verify a guard from BOTH failure directions when reviewing: disable it (the rejection tests
  must fail → it is load-bearing) AND over-broaden it (a legal case must fail → it does not over-fire).
  *Why:* forcing `check_compatibility` to early-`Ok` fails exactly the 4 rejection tests (top-level/nested V2
  + the unit test + the action-path V2 test) and NOTHING else; widening it to also gate `write_default` fails
  exactly `test_add_schema_with_write_default_only_allowed_on_v2`. The two mutations together prove the guard
  fires iff `initial_default.is_some() && format < v3` — the precise Java contract — not merely "fires
  sometimes." A one-direction mutation (disable only) would miss an over-firing guard.

### 2026-06-07 (Increment 7 — V3-only TYPE gate in check_compatibility, BUILDER Opus)
- **DO** model Java's `Schema.MIN_FORMAT_VERSIONS` (a `TypeID → minVersion` map) as a small
  `fn min_format_version(ty: &Type) -> Option<FormatVersion>` returning `Some(V3)` for the V3-only types and
  `None` otherwise, then gate on `format_version < min` per field. *Why:* it isolates the "which types are
  V3-only" knowledge in one place, keeps the per-field check a one-liner, and makes adding a future V3 type
  (`variant`/`unknown`/`geometry`/`geography`) a single `match` arm. Only `PrimitiveType::{TimestampNs,
  TimestamptzNs}` are representable in Rust today (both = Java `TIMESTAMP_NANO`); the other four are not in the
  Rust `Type`/`PrimitiveType` enums, so leave a comment, not a stub.
- **DO** fold a co-located Java check into the SAME single field-iteration pass and the SAME problem
  accumulator, in Java's order. *Why:* Java's `checkCompatibility` loops `lazyIdToField().values()` ONCE,
  checking the type rule (`MIN_FORMAT_VERSIONS`) BEFORE the initial-default rule, putting both into one
  `TreeMap<fieldId,String>` and throwing one combined `IllegalStateException`. The Rust mirror iterates
  `field_id_to_fields()` once, pushes the type problem then the default problem into one `Vec<(field_id,
  String)>`, stable-sorts by field id, and joins into the one `"Invalid schema for v{N}:"` error. A field that
  violates both rules surfaces both lines (type first); two different fields surface in id order. Pin it with a
  test that builds a V2 schema with a V3-typed field AND a separately-defaulted field and asserts BOTH
  substrings appear, type-before-default — proves the shared accumulator, mirroring Java's TreeMap. (NOTE: in
  Java, because both `problems.put(fieldId, …)` use the SAME field id when one field has BOTH a V3 type and a
  V3 default, the second put OVERWRITES the first — only the default message survives for that one field. The
  Rust `Vec` keeps both for a single field; this is a benign over-reporting divergence that only differs when
  ONE field is simultaneously a V3 type and carries a default, which the strongly-typed builder makes
  unusual. The cross-FIELD accumulation — the case the brief asked to prove — matches Java exactly.)
- **DO** confirm a build-path guard does NOT trip the parse path before declaring the blast radius safe.
  *Why:* `check_compatibility` is called only from `TableMetadataBuilder::add_schema` (the build/commit
  path); the `TableMetadataV2 → TableMetadata` `TryFrom` (parse path) constructs directly and never calls it,
  so reading an existing V3 `timestamp_ns` metadata fixture is unaffected — exactly like Java, where
  `checkCompatibility` lives in `addSchemaInternal`, not the parser. A guard that fired on parse would reject
  metadata that is already legally on disk.
- **DO** audit for tests that build a `<v3 TableMetadata` with the newly-gated type via
  `add_schema`/`from_table_creation`/`add_current_schema` BEFORE assuming reconciliation is needed. *Why:* the
  Increment-7 brief expected breakage, but a crate-wide grep showed every `timestamp_ns`/`timestamptz_ns`
  usage was in transform/arrow/avro/manifest/datum tests that build a bare `Schema` (via
  `Schema::builder().build()`, which does NOT call `check_compatibility`) or match on the type — none flowed
  through the metadata builder below v3. So ZERO existing tests needed changing. The build path vs. the
  bare-`Schema`-build path is the distinction that bounds the blast radius.

### 2026-06-07 (Increment 7 — V3-only TYPE gate in check_compatibility, REVIEWER Opus)
- **DECISION (TreeMap-vs-Vec divergence): KEEP Rust's both-report `Vec`, do NOT match Java's TreeMap
  last-wins.** Java keys `problems` by field id in a `TreeMap`, so a SINGLE field that is BOTH a V3-only
  type AND carries a non-null initial default collapses to ONE message (the second `put` overwrites the
  first — only the default line survives). Rust's `Vec<(field_id, String)>` reports BOTH lines for that
  one field. *Justification:* accept/reject is identical (both reject); the extra line is strictly more
  informative (Java arbitrarily hides the type problem behind the default); message text already differs
  by language (Rust renders `Literal` via `Debug`, Java via `toString`) so byte-identical messages were
  never the contract; the case is vanishingly narrow (one field simultaneously a V3 type AND defaulted on
  a <v3 table, unnatural for the strongly-typed builder). The CROSS-field case (two different fields →
  field-id order) — the parity-load-bearing one — already matches Java exactly. *Why it must be pinned:*
  a kept divergence that is UNtested is indistinguishable from an accidental one; if a later refactor
  silently drops one line, nobody notices. Added
  `test_check_compatibility_single_field_both_type_and_default_reports_both_lines` (one `timestamp_ns`
  field with an `initial_default` on V2 → BOTH lines, type before default). Mutation-verified: emulating
  Java's last-wins (dedup to a `BTreeMap` keyed by field id) fails EXACTLY this test and leaves the
  cross-field test green — proving it pins the both-report, not just "rejects."
- **DO add the single-field-both test even when the BUILDER documented the divergence in prose.** *Why:*
  the Increment-7 builder correctly chose the `Vec` and wrote the divergence into lessons/todo, but only
  added the CROSS-field accumulation test (two distinct field ids — which CANNOT collide in either Java or
  Rust and so cannot distinguish the two designs). The one test that actually exercises the divergence
  (a single field hitting BOTH rules) was missing — a documented-but-unpinned divergence. A prose note is
  not a regression guard; the test is.
- **DO `git stash` the production source and re-run a failing target on the clean tree before blaming
  your change.** *Why:* `cargo test -p iceberg` (all-targets) shows 5 doctest COMPILE failures
  (`lib.rs:24`, `writer/mod.rs:{42,117,257,321}`) — but all are `tokio::main` "default runtime flavor is
  `multi_thread`, but `rt-multi-thread` is disabled" errors, reproduced IDENTICALLY with the
  `check_compatibility` change stashed (and under `--all-features`). They are a pre-existing
  environment/feature artifact, NOT introduced by the guard. A compile error in an unrelated writer
  doctest is a strong tell that the failure is environmental, since a runtime guard cannot cause a
  doctest not to compile.
