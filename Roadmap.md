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

# Roadmap — Rust Iceberg (Java `iceberg-core` parity)

> **What this is.** The aggregate, project-manager plan for this repository: a **Rust-native**
> implementation of Apache Iceberg targeting **1:1 capability parity with the Java
> `iceberg-core` / `iceberg-api`** library — the engine-agnostic *table-format* core, **not** the
> Spark engine integration. It sequences all work from the current base through full parity, names the
> gate and exit criteria per phase, and is the entry point for any session (human or agent) picking up
> work here.
>
> **Authority.** This file is the **plan** (altitude + sequencing). The **living capability checklist**
> is [docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md). Repo conventions + read order live in
> [CLAUDE.md](CLAUDE.md); the testing contract is [docs/testing.md](docs/testing.md). When this file and
> the GAP_MATRIX disagree on a capability's status, the **GAP_MATRIX** (re-audited against the live base)
> wins and this file is corrected.

---

## North star

A **Rust-native** Apache Iceberg implementation with **1:1 capability parity** with the Java
`iceberg-core` / `iceberg-api` library. We **fork and own** these crates and maintain them indefinitely;
mergeability with upstream `apache/iceberg-rust` is **not** a constraint (we sync up from it and
cherry-pick wins, but diverge freely in service of parity). **Glue + S3 Tables** are the first-priority
catalogs. Python / PySpark is **deferred** (no PyIceberg, no PySpark layer) and the existing Python
layers are removed in Phase 0.

### Locked decisions

| Decision | Choice |
|---|---|
| Parity scope | Java **core library** (`iceberg-core` / `iceberg-api`), not the Spark surface |
| Core ownership | **Fork & own** the crates; drop the upstream-mergeability constraint |
| Deliverable | **Rust-native library** only; Python / PySpark deferred |
| Catalog priority | **Glue + S3 Tables first**, then REST, then Hive / JDBC / Nessie |
| Base | **Sync to upstream `iceberg` 0.9.x first**, then own from there |
| Python layers | **Delete** `iceberg-spark-python/`, `iceberg-spark-pyspark/`, `bindings/python/` |

---


## For a new session — start here

1. Read [CLAUDE.md](CLAUDE.md) (intent, prohibitions, conventions, read order) → this `Roadmap.md` →
   [docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md) (per-capability status — the ONLY status
   record) → [docs/testing.md](docs/testing.md) → [task/lessons.md](task/lessons.md) in full →
   [task/todo.md](task/todo.md) (the live plan).
2. **`main` is the owned 0.9.1 base.** Start from `main` or a short-lived feature branch off it.
   Where each phase stands: the one-line Status on each phase below; where each CAPABILITY stands:
   the GAP_MATRIX. Increment-level narrative lives in [task/todo-archive/](task/todo-archive/) and
   [task/lessons-archive/](task/lessons-archive/) — grep on demand, never required reading.
3. Verify the build before and after each change: `make build` / `make check` / `make test`, or the
   cargo commands in [docs/testing.md](docs/testing.md) (which also owns the gate-widening rules —
   workspace build for cross-crate changes; run `iceberg-datafusion` tests for read-path changes).
4. **Tests land with the code, same change.** A capability is ✅ only with unit tests AND an interop
   round-trip (see Definition of done).

### Sub-agent note
Per [CLAUDE.md](CLAUDE.md) `<subagent_policy>` the default is single-agent. The two heavy phases —
**Phase 2 (write engine)** and **Phase 4 (formats & types)** — are the natural fan-out candidates if the
policy is lifted; everything else is comfortably single-agent.

---

## Operational hardening sprint (2026-06-09) — decision record

A frontier-tier (Fable) review of the repo on 2026-06-09 concluded the **engineering discipline is
sound but the planning documents have outgrown their own read contract**: `task/todo.md` (380 KB),
`task/lessons.md` (256 KB), this file (72 KB), and the GAP_MATRIX (132 KB) totalled ~840 KB of
mandated session-start reading — several context windows — and the same increment status was being
written in triplicate (Roadmap current-state, Roadmap phase sections, GAP_MATRIX cells). The user
approved a hardening sprint **before further parity work**. Live plan + checkboxes:
[task/todo.md](task/todo.md) §"Operational hardening & Opus handoff".

**The decisions, for future sessions:**

1. **Model-tier handoff context.** Frontier-tier sessions are available only until **2026-06-22**;
   thereafter **Opus is the default maintainer tier**. Judgment-heavy work (write-engine interop
   semantics: sequence-number inheritance, delete manifests, conflict validation) is prioritized
   while the frontier tier remains; mechanical breadth (inspection interop, scenario fill-out, ORC,
   V3 exotica) is deliberately left for Opus — it is well-templated by the existing harness.
2. **The interop oracle is the project's objective verifier and its model-tier equalizer.** A
   weaker model's reasoning error cannot survive a bidirectional Java round-trip. Protect
   `dev/java-interop/` above other assets; every 🟡→✅ flip goes through it.
3. **Doc-mass discipline (sprint increments A–D):** `map.md` coverage for the hot source
   directories (A); the `skills/compaction.md` pass on lessons.md (B — it is 3–5× over its own
   trigger); a todo-archival convention + pass (C); then **one home per fact** (D): the GAP_MATRIX
   becomes the only status record with terse cells, this file's current-state section shrinks to
   ~30 lines, and increment narratives live in `task/todo-archive/`. After D, never write the same
   status in two places — link instead.
4. **Interop-debt budget (sprint increment E).** The 🟡-with-deferred-interop pattern is accepted,
   but the debt is paid down in risk order (RowDelta metadata semantics → the rewrite-family four →
   inspection tables) rather than accumulating indefinitely.
5. **Platform cut line — RESOLVED-AS-TABLED (2026-06-11); RE-OPENED & SUPERSEDED 2026-06-20 (see
   "Direction re-anchor" below).** The user tabled the downstream
   platform / DataFusion-SQL (RePark) direction to think it over, and redirected this fork's
   mission to **near-full 1:1 Java `iceberg-core`/`iceberg-api` replacement** — the phases run to
   completion in dependency-then-value order, NOT re-ranked by platform need. Maintenance actions
   (Phase 6) slot next after the Phase-2/3 residue on ordinary dependency grounds (their commit
   primitives now exist), not because of a platform cut. When the user re-opens the
   DataFusion/RePark discussion, the format-adjacent items it needs (`_file`/`_pos` metadata
   columns through the TableProvider, DataFusion write-path breadth) are ordinary GAP_MATRIX
   work that the near-full-parity path covers anyway.

> **Sprint status (2026-06-10):** A (maps), B (lessons compaction), C (todo archival), D (this
> de-triplication), and E3 (inspection interop) are DONE; E1/E2 (write-action metadata interop)
> remain. The size warnings above are resolved; the one-home-per-fact rule below is now in force.

---
---

## Direction re-anchor (2026-06-20) — the named consumer  ·  **engine-first closeout (2026-06-21)**

**Supersedes the "Platform cut line — RESOLVED-AS-TABLED (2026-06-11)" item above.** On 2026-06-20 the
user re-opened and resolved the platform direction: the terminal goal is no longer near-full 1:1 Java
parity *for its own sake* but **"be the Rust Iceberg core a named consumer pulls."** The named consumer
(option A) is a **downstream Apache DataFusion-wrapped custom query engine** issuing row-level
DELETE/UPDATE/MERGE. **Seam:** the `iceberg` CORE crate is the stable engine-facing contract — the
downstream builds its OWN DataFusion `TableProvider` over it; `crates/integrations/datafusion` is a
reference impl, not the product.

**DML foundation — DELIVERED & PROVEN (2026-06-21), through the PUBLIC surface, offline AND vs Java:**
#114 nameable write/commit action types · #115 the reserved `_pos` row-position column on Parquet scans ·
#116 the engine-boundary proof (scan `_file`/`_pos` → write position-delete → `RowDelta` → re-scan omits
exactly those rows) · #117 the public `iceberg::arrow::DeleteFilter` (mirrors Java
`org.apache.iceberg.data.DeleteFilter`) · the custom-scan **equivalence proof**
(`test/engine-mor-equivalence`): an engine's OWN raw read + `DeleteFilter::apply` reproduces the built-in
`to_arrow()` merge-on-read scan EXACTLY (position / equality / combined). A downstream engine can BYO
physical scan and get iceberg-correct rows. (These are engine-integration surface, not GAP_MATRIX parity
rows — no capability status changed.)

**Engine-first closeout (chosen 2026-06-21) — the new priority order.** The engine-serving core is now
substantively done (scan: `BatchScan` / `plan_tasks` / MoR deletes / `_pos` / `DeleteFilter`; write: the
full commit surface). Remaining parity work is re-ranked by **what the query engine actually pulls**:
- **Harden next (engine-relevant), in priority order:** (1) the **merge-on-read DELETE-apply residue** —
  multi-file-per-partition + non-identity partition transforms (the delete-application correctness the
  engine's own scan + `DeleteFilter` depend on; the "Read: merge-on-read apply" 🟡 row); directly extends
  the 2026-06-21 equivalence proof to realistic layouts. (2) **CDC row-level changelog** —
  `UpdateBefore`/`UpdateAfter` + accepting ranges that carry row-level DELETE manifests
  (`IncrementalChangelogScan` is whole-data-file-level only today). (3) the **ORC/Avro DATA-read residue**
  (footer codec / nested + V3 types / the Avro `timestamptz` mapping) — only if the engine queries
  non-parquet tables.
- **PULL-BASED (only on a concrete engine need):** `TransactionAction` made `pub` (custom commit actions);
  the transaction-side DV-on-DV auto-merge; the `iceberg-datafusion` `InsertOp` reference impl.
- **DEMOTED to opportunistic (a query engine does not pull these):** encryption, geometry/geography,
  ORC/Avro data **write**, `SnapshotTable`/`MigrateTable`. The PAR-05 near-full-parity march below is
  PAUSED in favor of this lens.

This changes SEQUENCING and PRIORITY only — the phase plan, the Definition of Done (interop bar), and
every GAP_MATRIX capability status are unchanged.

---
---

## Current state (one screen — details live in the GAP_MATRIX)

> **Corrected 2026-06-13** (post R1/R2/R3 audit); **prose refreshed 2026-06-16** (`caseSensitive` /
> `deleteFromRowFilter` builders built #51; conflict-validation interop proven for all 5 write actions
> #64-#69); **resynced 2026-06-17** (blocks 1-3 landings + the `validateAppendOnly` interop — 16 stale
> prose under-claims corrected against the matrix); **resynced 2026-06-19** (`BatchScan` ✅, ORC+Avro
> DATA READ 🟡, `RewriteTablePath` 🟡, events/listeners ✅ reflected; audit P1 panic-hardening + QUAL-01
> tracing landed). Per-capability status lives ONLY in the
> [GAP_MATRIX](docs/parity/GAP_MATRIX.md); this screen is a one-paragraph orientation that links there.

**Base:** upstream `iceberg` 0.9.1 (datafusion 52.2 / arrow 57.3 / parquet 57.3, MSRV 1.92), owned
fork on `main` since 2026-06-07. No Python layers. Offline lib suite green (**2,593 `#[test]`/`#[tokio::test]`
in the `iceberg` crate `src/`; ≈3,125 workspace-wide** — measured 2026-06-19);
Docker suites via `make test`; `sqllogictest` needs `protoc`.

**Interop-proven ✅:** the Phase-1 evolution surface (`UpdateSchema`, `UpdatePartitionSpec`,
`ManageSnapshots` ref-ops + snapshot refs + `cherrypick`/WAP); merge-on-read DATA-level scan
execution (full {position, equality} × {Java-writes, Rust-writes} × {unpartitioned, partitioned});
the deletion-vector writer + its DV row-delta commit (both directions); **`merge_append` data-level
interop both directions** (fixtures A + G, 2026-06-11 — moved out of the deferred bucket); scan
PLANNING; the COMPLETE inspection-table set incl. `readable_metrics`; the landed maintenance actions'
interop (`ExpireSnapshots` `ReachableFileCleanup` A3, partition-stats Z3/R2/R3, `ComputeTableStats`
theta-blob I1); **incremental append + changelog scans both directions (2026-06-17, rows 120/121 —
row-level CDC remains named residue)**. See the matrix for the exact ✅ rows + flip dates.

**Built but interop-deferred 🟡:** the metadata-level write actions (`DeleteFiles`, `OverwriteFiles`,
`ReplacePartitions`, `RewriteFiles`, `RowDelta`, `RewriteManifests` — Java-judged via the 8-step
chain); their **conflict validations are now interop-proven both directions (2026-06-15/16, PRs
#64-#69)** — the remaining 🟡 residue on those rows is the row-filter / multi-spec-DATA /
DELETE-file-rewrite interop slices, not conflict validation; residual evaluation;
scan-metrics model + emission; views (memory + REST + SQL landed, interop'd I2 — Glue/S3Tables view
ops are parity-correct-unsupported, NOT a gap — rows 124/125); `variant` (binary read+write byte-exact
both sides; shredded-parquet FILE I/O externally blocked by the parquet 57.3 pin); **ORC + Avro
DATA-file READ** (landed + Java→Rust interop-proven — rows 116/117; the WRITE half is the 🟡 residue).
Per-row status + residue: the matrix.

**Missing ❌:** ORC/Avro DATA-file **write** (READ landed 🟡 — rows 116/117), `geometry`/`geography`
types, encryption, `SessionCatalog` (assessed-deferred dead surface — row 126), the maintenance
residue (`SnapshotTable`/`MigrateTable` only — they need external sources), and
`commitTransaction(List<TableCommit>)` (REST multi-table commit, split out). *(No longer missing —
flipped 2026-06-17: `RewritePositionDeleteFiles` ✅ (134); `ComputePartitionStats` action +
`UpdatePartitionStatistics` ✅ (138); the `unknown` V3 type ✅ (89); `validateAppendOnly` ✅ (144); the
Catalog accessors `name()`/`properties()`/`invalidate*` 🟡 (149); the `conflictDetectionFilter`-on-
`DeleteFiles`/`ReplacePartitions` items are VOID — `javap`-proven not in Java 1.10.0. And flipped
2026-06-19: `BatchScan` ✅ (122); ORC+Avro DATA **read** 🟡 (116/117); `RewriteTablePath` 🟡 built+interop
(137); events/listeners ✅ (142); `LockManager` ❌→🟡 in-memory impl + tests (129). See GAP_MATRIX.)*

**Row-by-row truth:** [docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md).

---

## Working principles

- **Tests land with the code** (same change), plus **interop tests**: read tables Java wrote, and prove
  Java can read what we write — the only true 1:1 evidence.
- **The Java repo is the spec-by-example.** Keep a reference checkout of `apache/iceberg`; re-crawl on
  each Java release.
- **Re-audit after every upstream sync and every phase** — keep the GAP_MATRIX live.
- **Order by dependency, then value:** metadata correctness underpins writes; writes underpin
  maintenance actions.
- Engineering floor (no bare `.unwrap()` in prod paths, `thiserror`/`anyhow`, `tracing`, house style,
  `map.md` navigation): [CLAUDE.md](CLAUDE.md).

---


## Phase plan

Each phase: **Goal · Gates on · Key deliverables · Exit criteria · Status.** Granular per-capability
detail and live status live in [docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md).


### Phase 0 — Repo reset & base sync  ·  **Status: ✅ complete (2026-06-07)**
- **Goal:** a clean, owned, Rust-native base on upstream 0.9.x before any parity feature work.
- **Gates on:** —
- **Key deliverables:**
  - **Sync** to upstream `iceberg` 0.9.x; bump datafusion / arrow / parquet / object_store / opendal /
    AWS SDK / MSRV / toolchain to the family 0.9.x targets (≈ datafusion 52 / arrow 57); regenerate
    `Cargo.lock`; `cargo build` + `cargo test` green.
  - **Re-audit** the GAP_MATRIX against the 0.9.x base; strike rows already solved by 0.8 / 0.9.
  - **Delete** `iceberg-spark-python/`, `iceberg-spark-pyspark/`, `bindings/python/` and their CI/workspace
    references; workspace still builds.
  - **Rewrite** `PROJECT.md` + `CLAUDE.md` to this north star (remove the legacy Spark-drop-in framing,
    dead references, and any fake version pins); the CLAUDE.md ownership banner flags this rewrite as owed.
- **Exit criteria:** workspace builds + tests on 0.9.x; GAP_MATRIX reflects 0.9.x reality; Python layers
  gone; contract docs match reality; one clean commit per workstream (sync / re-audit / wipe / docs).
- **Sequencing within the phase:** sync → re-audit → wipe Python → rewrite contracts (rewrite last so it
  documents the *real* synced versions). **Recommended human checkpoints:** after the sync (the riskiest,
  most conflict-prone step) and before the irreversible Python deletion.

### Phase 1 — Spec & metadata completeness  ·  **Status: ✅ effectively complete (2026-06-07; `cherrypick` reclassified Phase-2)**
- **Goal:** the metadata-evolution surface that writes depend on.
- **Gates on:** Phase 0.
- **Key deliverables:** `UpdateSchema`, `UpdatePartitionSpec`, `ManageSnapshots` (branch/tag CRUD,
  rollback, rollback-to-time, set-current, fast-forward), full snapshot-ref handling, V3 groundwork.
- **Exit criteria:** each action matches the Java contract with unit + interop tests; GAP_MATRIX
  rows ✅. **Met for the entire surface** (all three capabilities bidirectionally interop-proven).
  V3 groundwork is largely closed here: `timestamp_ns`/`timestamptz_ns` ✅ with the
  `MIN_FORMAT_VERSIONS` gate enforced, `variant` schema-type entry + gate landed, and `unknown` is now a fully interop-proven ✅ row (row
  89 — schema-type entry + V3 gate + metadata round-trip interop; the gate now covers `timestamp_ns`/`variant`/`unknown` through
  the SAME `min_format_version`/`check_compatibility` mechanism); the remaining `MIN_FORMAT_VERSIONS`
  arms (`geometry`/`geography`) are Phase-4 type-breadth items, not Phase-1 groundwork. Increment
  narratives: [task/todo-archive/phase1.md](task/todo-archive/phase1.md).

### Phase 2 — Write engine  ·  **Status: 🟡 nearly complete (the FULL action set + the COMPLETE DV write surface [row ✅ 2026-06-11] + `cherrypick`; metadata-level interop Java-judged throughout. Remaining: real-catalog hardening, multi-spec writes, data-level write-action interop, `stageOnly`/`removeRows` residue)**
- **Goal:** the full commit/write surface beyond fast-append.
- **Gates on:** Phase 1.
- **Key deliverables:** `DeleteFiles`, `OverwriteFiles`, `ReplacePartitions`, `RewriteFiles`,
  `RowDelta` + position-delete/DV writers, `RewriteManifests`, merge append, multi-op transactions +
  optimistic-concurrency retry validated against Glue + S3 Tables.
- **Where it stands:** per-action status: GAP_MATRIX (the only status record). The DV writer + its
  row-delta commit are ✅ (interop both directions, 2026-06-11); `merge_append` data-level interop is
  now ✅ both directions; conflict-validation interop is proven for all 5 write actions (#64-#69).
  Remaining in-phase: real-catalog (Glue + S3 Tables) hardening, data-level interop for the remaining
  write actions, and the builder surface is now fully ported (GAP_MATRIX row 144 ✅): `validateAppendOnly`
  landed on `ReplacePartitions` (interop-proven 2026-06-17), and `conflictDetectionFilter` on
  `DeleteFiles`/`ReplacePartitions` is VOID — not in the Java 1.10.0 API.
  (`caseSensitive`/`deleteFromRowFilter` landed #51.)
- **Exit criteria:** each write action commits correctly through the real catalogs with conflict
  detection, with interop round-trips vs Java. Narratives:
  [task/todo-archive/phase2.md](task/todo-archive/phase2.md).

### Phase 3 — Scan parity  ·  **Status: 🟡 far along (inspection COMPLETE + interop'd; incremental scans interop-proven; reporting wired)**
- **Goal:** full read/scan capability + reporting + inspection.
- **Gates on:** Phase 1; benefits from Phase 2 (delete files to scan).
- **Key deliverables:** metrics/residual evaluators; incremental append/changelog scans +
  `BatchScan`; split planning; `ScanReport`/`MetricsReporter`; the full inspection-table set.
- **Where it stands:** inspection tables COMPLETE and interop-proven; data-level scan execution
  interop-proven both directions; residual evaluation wired into planning; incremental append +
  changelog scans interop-proven both directions (✅ 2026-06-17, rows 120/121; row-level CDC is named
  residue); `BatchScan` ✅ (row 122, interop-proven 2026-06-17); metrics model + opt-in emission landed.
  Remaining: CDC-merge (row-level), split planning (row 146), strict-evaluator completion (per-row
  status: GAP_MATRIX).
- **Exit criteria:** scans match Java result-for-result with reporting parity. Narratives:
  [task/todo-archive/phase3.md](task/todo-archive/phase3.md).

### Phase 4 — Format & type breadth  ·  **Status: 🟡 partial (V3-type front advanced; data-file formats untouched)**
- **Goal:** data-file format and V3 type coverage on par with Java `data/`.
- **Gates on:** Phase 1 (types in spec).
- **Key deliverables:** ORC + Avro **data** file read/write; remaining V3 types end-to-end — variant
  (incl. shredding), geometry/geography + geospatial predicates, `unknown`.
- **Where it stands:** `timestamp_ns` ✅ and column default values ✅ landed in the 0.9.1 base;
  `variant` is 🟡 (binary format read+write byte-exact BOTH sides — shredded-parquet FILE I/O is
  externally blocked by the parquet 57.1 pin, no variant support upstream). `unknown` is ✅ at the
  metadata level (schema-type entry + V3 gate + metadata-only schema round-trip interop; data-file
  always-null I/O deferred-loud — a no-physical-column type's contract is the metadata round-trip).
  Genuinely ❌: `geometry`/`geography` + geospatial predicates, ORC + Avro DATA files. Per-row status:
  GAP_MATRIX (the only status record).
- **Exit criteria:** read/write parity for ORC + Avro data; V3 types round-trip and interop with Java.

### Phase 5 — Catalog & views  ·  **Status: 🟡**
- **Goal:** view support + catalog completeness, Glue + S3 Tables first.
- **Gates on:** Phase 1.
- **Key deliverables:** `ViewCatalog` + view operations (create/replace/drop/list, view
  versions/representations) on Glue + S3 Tables, then REST; `SessionCatalog`; `LockManager`
  (in-memory impl + tests landed 2026-06-19; JVM-conformance interop is the ✅ gate); Glue + S3 Tables hardening.
- **Where it stands:** view ops landed + interop'd on memory + REST + SQL (I2); Glue + S3 Tables expose
  `FeatureUnsupported` view stubs, which is PARITY-CORRECT, not a gap (verified 2026-06-17 — Java
  `GlueCatalog` does not implement `ViewCatalog` (issue #12488, an OPEN feature request) and the S3
  Tables service has no view API; rows 124/125). `LockManager`: in-memory impl + tests landed (row 129 — see GAP_MATRIX); `SessionCatalog` is ❌
  but assessed-deferred as a dead surface (row 126 — revisit only for a multi-tenant single-instance
  use-case). Per-row status: GAP_MATRIX.
- **Exit criteria:** view lifecycle works on the priority catalogs with interop tests; session/lock gaps
  closed.

### Phase 6 — Maintenance actions & encryption  ·  **Status: 🟡 partial (9 of 12 ActionsProvider actions built — only `SnapshotTable`/`MigrateTable`/`RewriteTablePath` remain ❌; encryption ❌)**
- **Goal:** the engine-agnostic action layer + encryption.
- **Gates on:** Phase 2 (writes) and Phase 3 (scans).
- **Key deliverables:** `ExpireSnapshots`, `DeleteOrphanFiles`, `RewriteDataFiles` (compaction),
  `RewritePositionDeleteFiles`, `RemoveDanglingDeleteFiles`, `ComputeTableStats`/`ComputePartitionStats`,
  `SnapshotTable`/`MigrateTable`/`RewriteTablePath`; encryption (`EncryptionManager`, KMS client, encrypted
  FileIO + encrypted manifests/data, V3); metrics reporting + events/listeners.
- **Where it stands:** landed (several interop-proven): `ExpireSnapshots` (`ReachableFileCleanup`
  interop A3), `DeleteOrphanFiles`, `RewriteDataFiles`, `RemoveDanglingDeleteFiles`,
  `RewritePositionDeleteFiles` (✅ 2026-06-17), `DeleteReachableFiles` + `ConvertEqualityDeleteFiles`
  (✅, interop-proven), `ComputeTableStats` (theta-NDV, interop I1), and the `ComputePartitionStats`
  ACTION + `UpdatePartitionStatistics` commit surface (✅ 2026-06-17) over the partition-stats COMPUTE
  core + on-disk write (interop Z3/R2/R3). Genuinely ❌: `SnapshotTable`/`MigrateTable`/`RewriteTablePath`
  (need external sources), encryption. Per-row status: GAP_MATRIX.
- **Exit criteria:** maintenance actions match Java behavior with tests; encryption round-trips.

### Phase 7 — Continuous parity  ·  **Status: ❌ (ongoing)**
- **Goal:** keep parity from drifting as Java evolves.
- **Gates on:** Phases 1–6 maturing.
- **Key deliverables:** automation tracking Java release tags → re-crawl new features into the GAP_MATRIX;
  a differential conformance suite vs Java-produced tables run in CI; selective adoption of upstream
  `iceberg-rust` improvements.
- **Exit criteria:** CI fails on a parity regression vs Java; new Java features land as GAP_MATRIX rows
  automatically.

---


## Headline gap AREAS (ranked by effort × value — statuses live in the GAP_MATRIX)

> **Priority superseded 2026-06-21 — see "Direction re-anchor (2026-06-20)" above.** Under the
> engine-first closeout, this list is re-ranked by **what the named DataFusion consumer pulls**: harden
> CDC row-level (row 121) + the ORC/Avro **read** residue next; encryption / geometry / ORC-Avro **write**
> / `SnapshotTable`/`MigrateTable` are DEMOTED to opportunistic. The PAR-05 two-track text below is
> retained for its per-item detail but no longer sets the priority order.

Sequenced for the near-full-parity directive (2026-06-11); **re-steered 2026-06-19 (PAR-05) to a
two-track hybrid** now that the post-audit backlog is cleared and the work is Opus-paced (Fable
parked). **Track 1 (now — zero/low on-disk-format risk, ✅ momentum):** `LockManager` JVM
behavioral-conformance interop (in-memory impl + tests landed 2026-06-19 #109 — now 🟡; the
conformance test is the ✅ gate — pure catalog code, no format risk, unblocks Phase-5 catalog
completeness), then the ORC + Avro **read** merge-on-read residue (multi-file-per-partition /
non-identity transforms) — draining these cheap closeouts lifts the census from its live 38 ✅ floor.
*(Scan completion — `BatchScan` + split planning — already landed ✅ 2026-06-17 #87/#88; no longer
Track-1 work.)* **Track 2 (deliberate, format-risk-gated):** open
**encryption** (`EncryptionManager` / KMS / encrypted FileIO + manifests + data) as a scheduled
multi-block effort under the "do not break the on-disk format" approval + Java↔Rust interop
discipline, then geometry/geography + geospatial predicates, then ORC/Avro data-file **write**.
**Parked / not owed (do not "sequence" these):** variant shredding (externally blocked by the
parquet pin — binary format already byte-exact both sides), `SnapshotTable`/`MigrateTable` (need an
external source-table surface this library does not expose), `SessionCatalog` (dead surface),
Glue/S3 Tables views (parity-correct-unsupported). Within each track, judgment-heavy /
format-sensitive work still leads; well-templated breadth follows.

1. **Phase-2/3 closeout (frontier-first):** the conflict-detection + `caseSensitive` builder surfaces on
   `DeleteFiles`/`OverwriteFiles`/`RowDelta`/`ReplacePartitions` are now mostly ported, and their
   **conflict-validation interop is PROVEN both directions** (2026-06-15/16, PRs #64-#69): the
   `MergingSnapshotProducer.validate` engine was already implemented; `caseSensitive(bool)` LANDED on
   `DeleteFiles`/`OverwriteFiles`/`RowDelta` (narrowed out of `ReplacePartitions` per the Java API) and
   `DeleteFiles.deleteFromRowFilter(Expression)` LANDED (both 2026-06-13 #51; interop-proven ✅ 2026-06-16,
   rows 144/145). Now resolved: `validateAppendOnly` LANDED on `ReplacePartitions` (interop-proven ✅
   2026-06-17, row 144); `conflictDetectionFilter` on `DeleteFiles`/`ReplacePartitions` is VOID
   (`javap`-proven not in Java 1.10.0); the constants-map increment was activated 2026-06-11 (row 119).
   The genuinely STILL-open sub-items: the writer-layer multi-spec threading (a WIRING gap —
   `MergeManifestProcess` is not routed into the non-append merging actions; see GAP_MATRIX row 94), and
   the `dv_seq >= data_seq` index validation residue. (Real-catalog hardening needs user credentials —
   scheduled with the user; data-level write-action interop is templated → Opus.)
2. **Maintenance actions — LANDED:** `ExpireSnapshots`, `DeleteOrphanFiles`, `RewriteDataFiles`,
   `RewritePositionDeleteFiles` (✅), `RemoveDanglingDeleteFiles` (✅), `DeleteReachableFiles`,
   `ConvertEqualityDeleteFiles`, `Compute{Table,Partition}Stats` (built, interop-proven). Remaining
   ❌: `SnapshotTable`/`MigrateTable` (need external sources); `RewriteTablePath` is 🟡 (built + interop — row 137).
3. **Format & type breadth:** variant (incl. shredding — frontier; exact-byte class) and
   geometry/geography (`unknown` is ✅ at the metadata level — row 89); ORC + Avro data files
   (READ landed 🟡 — rows 116/117; the WRITE half remains; templated breadth → Opus).
4. **Scan completion:** CDC-merge (row-level) is the remaining open slice; split planning (row 146),
   `BatchScan` (row 122), and incremental-scan interop (rows 120/121) all LANDED ✅ 2026-06-17;
   strict-evaluator completion mostly templated → Opus.
5. **`LockManager`** (in-memory impl + tests landed 2026-06-19 — JVM-conformance interop is the ✅ gate) + the `ViewCatalog`
   byte-exact round-trip residue (row 125). Glue/S3 Tables view ops are parity-correct-unsupported
   (rows 124/125, NOT owed); `SessionCatalog` is assessed-deferred as a dead surface (row 126).
6. **Encryption** (`EncryptionManager`, KMS, encrypted FileIO / manifests — frontier-grade format
   work; schedule against the remaining frontier window or accept Opus pace).
7. **Phase 7 — continuous parity automation** (Java-release tracking, differential conformance in
   CI) — begins once 1-5 are substantially ✅.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| GAP_MATRIX drifts from reality as work lands or Java evolves | One home per fact (statuses ONLY in the matrix); re-audit after every sync and every phase; Phase 7 automates Java-release tracking. |
| Parity claimed without true 1:1 evidence | Definition of Done requires an interop test (Java↔Rust round-trip) before a row flips to ✅. |
| Status narratives regrow in this file | The de-triplication rule in [CLAUDE.md](CLAUDE.md): never write a capability's status outside the matrix — link instead. Archived narrative: [task/todo-archive/](task/todo-archive/). |

---

## Definition of done (per capability)

A GAP_MATRIX row flips to ✅ only when **(1)** the Rust API matches the Java contract's behavior,
**(2)** unit tests ship with it (same change), and **(3)** an interop test proves byte-level table
compatibility with Java in both directions where applicable.

---

## Cross-references

- [docs/parity/GAP_MATRIX.md](docs/parity/GAP_MATRIX.md) — the living capability audit (the checklist this
  roadmap drives).
- [CLAUDE.md](CLAUDE.md) — repository intent, prohibitions, conventions, read order, sub-agent policy.
- [docs/testing.md](docs/testing.md) — the testing contract (tests-with-code + interop tests).
- [README.md](README.md) — project front door.
