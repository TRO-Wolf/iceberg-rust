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


> **Archival log.** Last pass: 2026-06-12 (size trigger — 2,358 lines; pass 3) →
> [todo-archive/2026-06_wave3-wave4-overnight.md](todo-archive/2026-06_wave3-wave4-overnight.md)
> (24 sections: 2 kept live, 22 archived to the one pass-scoped file incl. the superseded wave
> planning section; open items lifted into the fresh ACTIVE section). Prior passes: 2026-06-11
> (pass 2 — 1,381 lines → phase files + ops-hardening) and 2026-06-09 (pass 1 — 4,344 lines →
> phase files). Procedure: [skills/compaction.md](../skills/compaction.md) §Todo Archival.
> Archives are not read by default.

## ACTIVE (2026-06-12): Z1 — staged-WAP interop fixture (BUILDER Sonnet, wt-interop3)

**Structure choice:** a SIBLING script `dev/java-interop/run-interop-staged-wap.sh` (separate from
`run-interop-cherrypick.sh`) so the new staged-state-comparison steps do not disturb the existing
3-fixture chain that is already green.

**Java-emitter enumeration verdict (PRE-DECIDED CAVEAT):** `SnapshotMetaOracle.emit()` at line
6479 iterates `metadata.snapshots()` — the FULL metadata snapshot list, NOT `currentAncestors` or
`history`. A staged ref-less snapshot IS included. Confirmed all-snapshots — no STOP needed.

**Plan:**

- [x] Record plan (this section).
- [x] **Step 1: Java `StagedWapOracle` (new inner class in InteropOracle.java).** Three fixtures
  (S-ff / S-replay / S-dedup). Java uses REAL `stageOnly()`. S-dedup redesigned to the
  `testDuplicateCherrypick` same-parent pattern (stage BOTH w3 off S0 before any cherry-pick;
  first cherry-pick FF, second cherry-pick REPLAY → validate_wap_publish → DuplicateWAPCommitException).
  Both views per fixture + `wap_dedup_expected_rejection.json`. `generate-interop-staged-wap` +
  `verify-interop-staged-wap` modes landed.
- [x] **Step 2: Rust interop test `crates/iceberg/tests/interop_staged_wap.rs`.** Both directions,
  both views. S-dedup redesigned to match Java's same-parent pattern. Direction 2 assertions updated
  (staged count=3, final count=3, no source-snapshot-id on FF current, wap.id=w3 on current).
- [x] **Step 3: Shell script `dev/java-interop/run-interop-staged-wap.sh`.** 7 steps landed.
  Sabotage 7d redesigned from "inject main ref" (ineffective — canonical view omits refs) to
  "remove staged snapshot from metadata" (canonical view loses 1 ordinal → diverges → PASS).
- [x] **Step 4: map.md updates** (dev/java-interop/map.md, crates/iceberg/tests/map.md).
- [x] **Step 5: GAP_MATRIX.md update** (cherrypick row — staged-WAP interop ✅ 2026-06-12 landed).
- [x] **Step 6: Run full chain ×2, paste output; run verbatim gate ×2.**
  Full chain ×2 green (0 failures, all 4 sabotages closed). Verbatim gate ×2: 2168 lib tests.
- [x] **Step 7: task/lessons.md update.** Three lessons added: S-dedup same-parent pattern,
  canonical view does not include refs/current-snapshot-id (sabotage redesign), `cargo fmt` order.

**Outcome:** Z1 staged-WAP interop fixture landed 2026-06-12. Three fixtures × two views × two
directions green. The S-dedup same-parent correction was the key insight; the sabotage redesign
was secondary.

**Z1 OPUS REVIEWER (2 of 2), 2026-06-12 — cold-start verification:**

- [x] Cold-start required reading (CLAUDE.md, Opus.md + Sonnet addendum, lessons, git diff full,
      the three new/changed pieces, the canonical view, cherry_pick.rs + stage_only surface).
- [x] RUN the chain ×2 + 4-sabotage battery (both green; 7a/7b/7c/7d closed). Ref-state on BOTH
      sides: D1 verify + D2 test both assert it; FOUND S-replay D2 staged current loose → tightened.
- [x] S-dedup rejection arm: verbatim "Duplicate request to cherry pick wap id..." + DataInvalid +
      non-retryable on BOTH sides; FF-first (count unchanged) → REPLAY-second fires dedup. Confirmed.
- [x] wap.id coverage: canonical view excludes it; FOUND D1 value-pin missing (corruption passed) →
      added expectedStagedWapId pin + FF current-wap pin; mutation now fails closed BOTH directions.
- [x] ROW-STATUS JUDGMENT (headline): verdict = ✅-scoping HONEST (surface-scoped, row stays 🟡,
      data-level residue named); made airtight by the D1 wap.id-value pin I added.
- [x] 5 mutations beyond the battery: wap.id-value (D1+D2), ordinal/seq perturbation, current-id move
      (ref-state), cross-fixture swap — all fail closed.
- [x] Verbatim gate ×2 (2168 lib tests ×2, typos/fmt/clippy clean). Cross-chain: cherrypick + expire
      green; write-data running.
- [x] GAP_MATRIX pipe audit clean (5 pipes/row). Tier-ledger data point recorded in final report.

## ACTIVE (2026-06-12): Near-full-parity direction — open queue (planning record)

Directive (user, 2026-06-11): run this fork's Roadmap to **almost the full 1:1 Java replacement**;
DataFusion/RePark tabled. Waves 3–4 + the overnight session landed PRs #28–#37 (multi-spec writes,
constants-map, ExpireSnapshots + interop, DeleteOrphanFiles, RewriteDataFiles,
RemoveDanglingDeleteFiles, the variant arc end-to-end, stage_only + WAP dedup, ComputePartitionStats,
data-level interop fixtures A–G, cherrypick interop). Statuses live ONLY in the GAP_MATRIX.

- [ ] **THIS BRANCH (Wave 5 Group Z, SONNET-builder + OPUS-critic per the calibrated split,
      user-approved 2026-06-12): the named interop items.** Z1: the staged-WAP fixture (V1's
      `stage_only()` + the cherrypick chain; THE EXIT-AUDIT CAVEAT IS BINDING — the Java
      `SnapshotMetaOracle` enumerates ALL metadata snapshots, NOT ancestry/history, and a staged
      ref-less snapshot must appear identically on both sides); Z2: the multi-spec fixture
      (comparator groundwork RESOLVED in W3 — spec_id is already the final tiebreaker on all
      three view copies); Z3: Java-reads-our-partition-stats-file (PartitionStatsHandler
      .readPartitionStatsFile as the Direction-2 oracle). Worktree `wt-interop3`, parallel to
      Group Y (`phase6/compute-table-stats`, Opus) and Group U (`phase5/view-ops`, Opus).
- [ ] **Partition-stats residue:** the INCREMENTAL compute path; time/uuid/fixed/binary partition
      values in stats files (loud errors today).
- [ ] **`ComputeTableStats` (NDV/theta sketches) — DEPENDENCY-GATED, user decision:** needs a
      DataSketches-equivalent crate; Cargo frozen.
- [ ] **Reported divergences awaiting their increments:** manifest-list carried-vs-new entry ORDER
      differs from Java (cosmetic — readers + the canonical oracle reconcile by seq; O1 reviewer);
      F1 pre-existing global flags (Java lowercases ALL type names on parse, Rust exact-lowercase;
      Rust sort orders unbound/unvalidated on metadata parse; struct map-value Avro record naming
      shares the duplicate-name hazard variant's fix closed).
- [ ] **THIS BRANCH (Wave 5 Group U, Opus actor-critic, user-approved 2026-06-12): views in
      catalogs.** U1: view-metadata + view-ops completeness vs Java 1.10.0 (`ViewMetadata`
      builder parity, `ReplaceViewVersion`/`UpdateViewProperties` ops, version-log semantics,
      the `replace` dialect rules — bytecode-pinned); U2: catalog view CRUD parity (the `Catalog`
      view surface across MemoryCatalog + the SQL catalog; REST view shapes verified against the
      existing REST machinery; Glue/S3Tables deferred to the credentialed sprint). Worktree
      `wt-views`, parallel to Group Y (`phase6/compute-table-stats`) and Group Z
      (`interop/wave5`).
### Wave-5 Group U / U1 — view metadata + view ops + in-tree catalog CRUD (BUILDER, wt-views)

**Survey verdict (read before assuming a gap):** the SPEC side is already a faithful 1:1 port of
Java's `ViewMetadata.Builder` — `spec/view_metadata.rs` + `view_version.rs` +
`view_metadata_builder.rs` (58 KB) carry version-id assignment, the identical-version REUSE
(`reuse_or_create_new_view_version_id` == Java `reuseOrCreateNewViewVersionId`/`sameViewVersion`),
schema interning by id, version-log append + expiry (`version.history.num-entries`, default 10),
dialect rules (unique-per-version, replace-must-not-drop unless `replace.drop-dialect.allowed`),
parser via serde `_serde::ViewMetadataV1`. **What is MISSING:** (a) no `View` type, (b) no
view methods on the `Catalog` trait, (c) no view CRUD in any catalog, (d) no `ViewCommit` /
`ViewRequirement` / view-ops seam, (e) no `ViewMetadata::read_from`/`write_to`.

- [x] **U1a — spec IO + wire-format pin:** added `ViewMetadata::read_from`/`write_to` (gzip-aware,
      mirrors `TableMetadata`); pinned the Java `ViewMetadataParser`/`ViewVersionParser` FIELD SET
      round-trip + read/write FileIO round-trip. (Field order DOES differ Rust↔Java — see lesson.)
- [x] **U1b — `ViewRequirement` + `ViewUpdate::apply`:** `ViewRequirement::{NotExist, UuidMatch}`
      with `check(Option<&ViewMetadata>)` (case-insensitive uuid per `AssertViewUUID`) + serde tags
      `assert-create`/`assert-view-uuid`; `ViewUpdate::apply(ViewMetadataBuilder)`. Requirement set
      = `[AssertViewUUID]` ALONE, bytecode-confirmed (`forReplaceView`: null base ⇒ AddSchema arm
      no-ops, no AddViewVersion arm).
- [x] **U1c — `View` type + `ViewCommit`:** `crates/iceberg/src/view.rs` — `View` + `ViewBuilder`,
      `ViewCommit{ident, requirements, updates}` with `apply(View)->View` (check reqs, apply updates,
      bump metadata-file version via `MetadataLocation::with_next_version`).
- [x] **U1d — view ops seam:** `ReplaceViewVersionAction` (Java `ViewVersionReplace.internalApply`
      — requires query/schema/namespace, version-id `max+1` then builder REUSE) +
      `UpdateViewPropertiesAction` (set/remove + can't-set-and-remove guard), both emit `ViewCommit`.
- [x] **U1e — `Catalog` trait view surface + MemoryCatalog:** seven default-erroring view methods
      on the trait + full MemoryCatalog impl (new `view_metadata_locations` namespace state +
      `ViewNotFound`/`ViewAlreadyExists` ErrorKinds). REST view shapes + SQL catalog views deferred
      to U2/Task-3 (NOT this increment).
- [x] **U1f — tests (18, each risk-named) + GAP_MATRIX row 107 + gate ×2 (2168→2186, both green).**

Outcome: U1 landed. Spec builder was already a 1:1 Java port (the survey's key finding); the gap was
the entire catalog-facing view surface. Pre-existing flag: SQL catalog `lib.rs` doctest fails to
compile (tokio `rt-multi-thread` feature gate) — unrelated, SQL crate untouched, its 52 unit tests
pass. Deferred per scope: SQL/REST/Glue/S3Tables view CRUD, view interop (next wave).

U1 REVIEWER (2026-06-12, adversarial vs 1.10.0 bytecode): forReplaceView=`[AssertViewUUID]` ALONE
re-derived from bytecode (null base ⇒ AddSchema no-ops, no AddViewVersion arm) — CONFIRMED, pinned at
the commit object. Replace dialect-drop guard + escape, identical-version reuse, the missing-required-
field serde pin, and Java-parser tolerance of Rust's wire format all VERIFIED (Java reads Rust's JSON
today — order-insensitive + tolerates always-present empty `properties`; cosmetic divergence only).
FIXED one real divergence: table↔view name collisions were silently allowed (Java rejects both
directions) — added symmetric cross-guards to `insert_new_table`/`insert_new_view` + 2 tests
(fail-before/pass-after). REPORTED, not fixed (pre-existing, shared with `update_table`): the in-tree
`update_view` has no base-location CAS, so a stale concurrent commit lands last-write-win (Java's
`doCommit` throws CommitFailedException) — belongs in a dedicated concurrency-parity increment.
Gate green ×2 (2186→2188 with the 2 collision tests). Verdict: APPROVE with the concurrency-CAS gap
flagged for the orchestrator.

### Wave-5 Group U / U2 — SQL catalog views + REST view shapes (BUILDER Opus, wt-views)

**JDBC storage-scheme verdict (1.10.0 bytecode, `JdbcUtil`):** views live in the SAME `iceberg_tables`
table, discriminated by the EXISTING `iceberg_type` column = `'VIEW'` (constant `VIEW_RECORD_TYPE`).
Tables use `'TABLE'` OR-NULL (V0 backcompat); views are V1-only (`JdbcViewOperations.doCommit` uses
`SchemaVersion.V1` unconditionally — exact-match `iceberg_type = 'VIEW'`, no OR-NULL). The Rust SQL
catalog already creates the V1 schema (the `iceberg_type` column exists), so it is V1-native — mirror
the posture, no schema-version flag needed. The view SQL constants (`GET_VIEW_SQL`, `LIST_VIEW_SQL`,
`RENAME_VIEW_SQL`, `DROP_VIEW_SQL`, `V1_DO_COMMIT_VIEW_SQL`, `V1_DO_COMMIT_CREATE_SQL`) are the exact
table-CRUD SQL with `AND iceberg_type = 'VIEW'`. CAS: `V1_DO_COMMIT_VIEW_SQL` = `UPDATE ... WHERE ...
AND metadata_location = ?` (0 rows = conflict, `CommitFailedException "Cannot commit %s: metadata
location %s has changed from %s"`). Collisions: create-view checks `tableExists` → `AlreadyExists
"Table with same name already exists: %s"`; create-table checks `viewExists` → `"View with same name
already exists: %s"`; rename cross-checks the other map.

**REST verdict (1.10.0 bytecode):** routes `GET/POST /v1/{prefix}/namespaces/{ns}/views`,
`GET/POST/DELETE/HEAD .../views/{view}`, `POST /v1/{prefix}/views/rename`. Wire types: CreateViewRequest
`{name, location, view-version, schema, properties}`; LoadViewResponse `{metadata-location, metadata,
config}`; the replace/commit reuses `UpdateTableRequest` shape `{identifier, requirements, updates}` but
with VIEW requirements/updates (`UpdateRequirements.forReplaceView`). `ViewUpdate`/`ViewRequirement`
already carry the correct REST serde tags. Rename reuses the `{source, destination}` shape.

- [x] **U2a — SQL catalog view CRUD (7 methods):** done. `iceberg_type = 'VIEW'` discriminator
      (`CATALOG_FIELD_VIEW_RECORD_TYPE`), view error helpers, all 7 methods, collision guards both
      directions (added the table-side guards to `create_table`/`rename_table` too), location-CAS in
      `update_view` (`AND metadata_location = ?` → 0 rows = `CatalogCommitConflicts`).
- [x] **U2b — SQL tests (10, the full U1 lifecycle e2e ported + risk-named):** create/load/list;
      duplicate-fails; full lifecycle; update_properties; identical-version reuse; load-missing;
      collision BOTH directions; rename-onto-table rejected; list_views scoping; the CONCURRENT
      location-CAS conflict (barrier-synchronized two-task race — deterministic, ran ×5).
- [x] **U2c — REST view shapes:** `CreateViewRequest`/`LoadViewResult`/`CommitViewRequest` in
      `types.rs` + 3 serde round-trip tests vs hand-pinned Java-wire JSON.
- [x] **U2d — REST view methods:** all 7 wired (endpoint builders + `build_view_from_load_result`
      helper). Full impls landed (machinery cheap) + 7 mockito route/status-mapping tests.
- [x] **U2e — gate ×2 (SQL 62×2), GAP_MATRIX row 107 (5-pipe audit clean), lessons, todo.**

Outcome: U2 landed. SQL catalog views are a faithful JDBC-scheme port (`iceberg_type='VIEW'` in the
shared `iceberg_tables`, V1-schema exact-match — the Rust catalog is already V1-native). The SQL
`update_view` IMPLEMENTS the JDBC location-CAS (Java `V1_DO_COMMIT_VIEW_SQL`), a deliberate per-catalog
divergence from MemoryCatalog's no-CAS posture. REST landed BOTH the typed shapes AND full method
impls (the machinery made full impls cheap). One flagged shared-type fix: `ViewRepresentations` had no
public constructor, so `ViewCreation` was unconstructable from out-of-crate catalogs — added
`ViewRepresentations::new` in `catalog/mod.rs` (in-scope shared-type escape). Gate: iceberg 2188, SQL
62×2, REST 51, clippy/fmt/typos clean. Deferred per scope: Glue/S3Tables views (credentialed), view
interop (next wave).

U2 REVIEWER (2026-06-12, adversarial vs 1.10.0 bytecode): JDBC verbatim-ness CONFIRMED — re-derived all
six `JdbcUtil` view constants; Rust queries are SEMANTICALLY verbatim (conjunct SET identical; only
AND-clause/SET-column order differs, commutative-safe), CAS clause present, exact-match VIEW posture vs
TABLE-or-NULL backcompat mirrored, collision messages byte-verbatim. CROSS-CATALOG error-kind
consistency HOLDS (Memory U1 + SQL U2 both: view-over-table→TableAlreadyExists,
table-over-view→ViewAlreadyExists). CAS race test NON-VACUOUS — drop `AND metadata_location = ?` ⇒ both
win ×5 (corruption); loser kind/retryable match the table-side `update_table` convention; auto-rebase
off the freshly-reloaded base verified. REST wire EXACT vs bytecode (`ResourcePaths` routes,
`CreateViewRequestParser` keys, `RESTViewOperations`→`UpdateTableRequest.create` commit body). FIXED two
real gaps in touched files: (1) REST `update_view` 5xx now maps to CommitStateUnknown (Java
`ViewCommitErrorHandler` 500/502/503/504; was folded into the generic default arm, diverging from the
table side) + test; (2) tightened the body-blind commit mockito test with a hand-pinned body matcher.
ADDED two permanent pins the builder lacked: `rename_view` rejects a TABLE source (discriminator-on-
rename, mutation-proven), and a compile-level out-of-crate accessibility probe (pins the
`ViewRepresentations::new` gap CLASS forever — removing the fix fails the SQL test build). REPORTED, not
fixed (benign/unreachable): create-view collision-check ORDER is reversed vs Java (only matters if both
a table AND view of the same name exist — guards make that unconstructible). Mutations ×5 all killed
(CAS-drop, list_views filter-drop, REST route typo, rename discriminator-drop, accessibility-fix
removal). Gate ×2 green: iceberg 2188, SQL 64, REST 52, clippy/fmt/typos clean; GAP_MATRIX 5-pipe
audit clean; tree = allowed set + the flagged mod.rs line. Verdict: APPROVE.

- [ ] **Scheduled with the user:** real-catalog (Glue + S3 Tables) hardening — needs credentials.
- [ ] **Opus-queue (post-handoff or parallel):** ORC/Avro breadth, view ops + SessionCatalog +
      LockManager (Sonnet-builder + Opus-critic per the calibrated split), incremental-scan interop,
      scan completion (BatchScan / CDC / split planning).

## Carried-forward open items (full context in todo-archive/)

**Explicitly NOT decided:** the "platform cut line" through the GAP_MATRIX (which rows block the
user's trading platform vs continuous-parity backlog, incl. re-ordering maintenance actions ahead of
Phase-4 format exotica) was proposed but is an **open user decision — do not assume it.**
  _RESOLVED-AS-TABLED 2026-06-11: the user tabled the DataFusion/RePark direction and redirected
  the fork to near-full 1:1 Java parity — recorded in Roadmap.md (decision record item 5 + the
  re-sequenced headline areas). Originating narrative:
  [todo-archive/2026-06_ops-hardening.md](todo-archive/2026-06_ops-hardening.md)._


## Archived increment narratives

Completed-increment narratives moved verbatim out of this file (see [skills/compaction.md](../skills/compaction.md)
§Todo Archival). Not session-start reading — grep/open on demand.

- [todo-archive/phase1.md](todo-archive/phase1.md) — Phase 1 spec & metadata completeness (schema /
  partition / snapshot evolution + spec-read robustness).
- [todo-archive/phase2.md](todo-archive/phase2.md) — Phase 2 write engine (write actions + the
  concurrent-commit conflict-validation cluster, incl. the merged write-validation PR #9).
- [todo-archive/phase3.md](todo-archive/phase3.md) — Phase 3 scan parity (residual evaluation,
  inspection tables, scan-metrics emission, and inspection / scan-execution interop).
- [todo-archive/2026-06_ops-hardening.md](todo-archive/2026-06_ops-hardening.md) — the doc-infrastructure / hardening meta-sprints (not phase work).
- [todo-archive/2026-06_wave3-wave4-overnight.md](todo-archive/2026-06_wave3-wave4-overnight.md) — Waves 3–4 + the overnight session (PRs #25–#37; pass-scoped).
- Index: [todo-archive/map.md](todo-archive/map.md).
