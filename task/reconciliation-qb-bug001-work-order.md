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

# Work order — line reconciliation + position-delete attach correctness (QB + BUG-001)

**Date:** 2026-08-03 · **Status:** DRAFT — awaiting user sign-off on gates D1–D4 below
**Author:** fork orchestrator session (independent verification pass over the 08-02/08-03 Desktop audits + the Opus meta-review)
**Repos:** `TRO-Wolf/iceberg-rust` (fork) · `BigRustSparkRebuild` (RePark, consumer) · PrimarySync (hub)
**On sign-off:** this file becomes the executing brief (reference it from `task/todo.md`; commit a copy to `task/` with the first unit).

---

## 1 · State of the world (all items verified 2026-08-03, evidence in §8)

| Fact | Value |
|---|---|
| Fork `main` | `0421ae15` (#183 FK MoR perf campaign — merged this morning) on top of `a966055e` (#182 slate) |
| RePark pin | `b009ac158…` on **all five** `[patch]` revs = tip of `chore/df54-family-bump` |
| Divergence | merge-base `9f2bf661` (#181); **neither line is an ancestor of the other** (18 commits on df54) |
| df54 contents | pre-squash copy of **WG0/WG1 + G1** (byte-identical `partition_work.rs` vs #182; DF provider carries full `multi_partition` wiring) **+** the real payload: DF 54.1 / Arrow 58.4 / parquet 58.4 / MSRV 1.94 / nightly-2026-03-05 (3 commits: `34192fcd`, `996a999e`, `d77bedae`) |
| Consequence | RePark **ships G1 multi-partition scan today** and lives on DF 54.1/Arrow 58.4 — it **cannot repin to main** (still DF 52.2/Arrow 57.1) until the bump lands on main. Pin discipline (R5, single line) is broken until then. |
| Breaking surfaces on main, **unannounced** | G2 `PartitionKey::new → Result` (#182) · FK2.1 `FileScanTask` Arc-shared fields (#183) · FK3 `DeleteFilter::deleted_row_positions → Option<Arc<DeleteVector>>` (#183). RoadMapSync's last fork rows are 2026-07-29/30 — the G2 coordination duty from the 07-31 work order was **not discharged**. |
| BUG-001 (08-03 audit) | **Confirmed, line-exact, survives #183.** `delete.rs:1007` fast path stamps every position delete `partition_key = None` off the table **default** spec; silently-wrong-results class (resurrected rows). Present on **both** lines (predates the divergence). Details §5. |
| Hazard-1 (2026-08-01 review) | OVERLAP-vs-MIDPOINT row-group selection — **open on both lines**; exposure = offsets-less manifests, which is exactly the DuckDB shape RePark already hit (QD). |
| Hazard-2 | `_pos`-over-ranged-task guard exists on main (FK rider `a74c4c54`) but **not on the pin**. Public-seam-reachable only. |
| Signed queue (D7, 2026-07-26) | QA ✅ #176 → Unit 3 + QC ✅ (delivered by slate G2, #182) → **QB is next** → H7-S2 → H7-P1. BUG-001 folds into QB naturally (same capability, other leg). |

## 2 · Decision gates (user)

- **D1 — Approve the dependency family bump landing on main** (Unit 2): workspace `Cargo.toml` + `Cargo.lock` → DF 54.1 / Arrow 58.4 / parquet 58.4, `rust-version` 1.92→**1.94**, `rust-toolchain.toml` nightly → 2026-03-05. This is the CLAUDE.md "never edit dependency files without explicit approval" gate. *(CI's MSRV job reads `rust-version` via `get-msrv`, so no workflow edit expected — verified at unit start.)*
- **D2 — Fix shape for BUG-001** (Unit 1): **Option A (recommended)** — condition the fast path on `partition_specs.len() == 1 && default_spec.is_unpartitioned()` (metadata-only, keeps the fast path for never-evolved unpartitioned tables, correct for all evolved shapes). **Option B** — delete the fast path outright (simpler; costs one manifest-list scan per DELETE on unpartitioned tables).
- **D3 — Sequencing**: recommend U0 now → U1 and U2 in parallel (independent trees) → U3 → RePark repins once. Alternative: strictly serial U0→U1→U2→U3.
- **D4 — Cadence per unit**: recommend AC·OO Mode A for U1 and U3 (correctness-bearing); U2 may run overnight-octo **plus a mandatory independent morning pass** (the original bump commits only ever had octo convergence, and Arrow 57→58 on the in-memory currency is exactly where an independent pass pays).

## 3 · Unit 0 — Owed comms (no code; do first; ~30 min)

Add RoadMapSync rows (draft text, edit freely):

1. **2026-08-03 · iceberg-rust** — "#182 (`a966055e`, 07-31 slate: WG1 prune-only, G1 multi-partition plan_tasks, WG2 within-file parallel reads, G2 PartitionKey::new→Result + toHumanString, G3 HMS) and #183 (`0421ae15`, FK1–FK5 MoR perf campaign) MERGED. **Three breaking surfaces for consumers:** `PartitionKey::new` now returns `Result` (G2); `FileScanTask` fields Arc-shared — `Arc<str>` / `Arc<[T]>` / `Option<Arc<BoundPredicate>>` (FK2.1, serde wire shape unchanged); `DeleteFilter::deleted_row_positions → Option<Arc<DeleteVector>>` (FK3). **RePark: do NOT repin yet** — pin `b009ac15` sits on a diverged line (DF 54.1); one repin after the Unit-2 re-cut lands (below)."
2. **2026-08-03 · fork + RePark** — "Line reconciliation plan: df54 family bump re-cut onto main as its own PR (Unit 2); pinned line carries G1 unchanged (`partition_work.rs` byte-identical to #182) so no scan-behavior delta at repin from the reconciliation itself. Known-on-pin: BUG-001 fast-path stamp (fix = Unit 1, will be on main at repin); hazard-2 `_pos`-ranged guard is main-only until repin — do not expose `_pos` over ranged `PartitionWork` windows meanwhile; hazard-1 midpoint row-group selection open on both lines (fix = Unit 3, targeted before/with repin — DuckDB offsets-less manifests are a live consumer shape)."

Also: claim-board rows (R2) for Units 1–3 before any edits.

## 4 · Unit 1 — QB + BUG-001: position-delete attach, both legs *(signed-queue next unit; main-side; no dependency on U2)*

**Branch:** `fix/qb-posdelete-bounds-and-partition-stamp` (from `0421ae15`, in `iceberg-rust-ws`) · **PR tag:** `[fork]`

**Leg A — partition stamp (BUG-001).** Per D2. Also fix the doc comment at `delete.rs:989–992`, which conflates "unpartitioned table" with "unpartitioned default spec". Note `is_unpartitioned()` = `is_empty() || all-Void` (`partition.rs:101`), so the V1 all-Void drop-partition shape takes the same path — cover it.

**Leg B — writer path bounds (QB proper, row R113 — re-verify anchor at start).** Position-delete writers must emit **full, untruncated** `file_path` lower/upper bounds so equal-bounds `referenced_data_file` derivation works (parquet-rs truncates byte-array stats at 64 bytes; realistic S3 paths never yield equal bounds today, which kills the path-routing leg that would otherwise mask Leg A). Scope per the signed queue brief `task/post-bundle-queue-2026-07-26-brief.md`.

**Why one unit:** the two legs are the two attach paths for the same delete files — path leg (B) and partition leg (A). Fixing either alone leaves the other silently load-bearing.

**Tests (ship with the change):**
- Unit: metadata-builder-constructed table with spec evolution (partitioned spec 0 → unpartitioned default spec 1; plus V1 all-Void variant) → DataFusion DELETE → scan: **zero resurrection**; assert stamped `(spec_id, partition)` equals each data file's own.
- Mutations (all must go RED, restore + re-green): restore the unconditional fast path; truncate bounds back to 64 bytes; (if Option A) weaken the condition to default-spec-only.
- Interop (`dev/java-interop`): Spark creates partitioned table → `ALTER TABLE … DROP PARTITION FIELD` → Rust DELETE → **Java reads back** zero resurrection; reverse leg (Java DELETE on evolved table → Rust read) pinned as a control.
- Matrix: update R113 cell (+ note in R117 that path-leg routing now has real bounds); `make check-matrix-anchors`.

**Gate:** full chain (`typos .` first) + lib + datafusion + slt 9/9 + ndf + interop subset touching deletes; anti-stale-rlib touch protocol.
**Estimate:** 4–6 h with interop. **Cadence:** per D4.

## 5 · Unit 2 — df54 family bump re-cut onto main *(gated on D1)*

**Branch:** `chore/df54-recut-on-main` (from `0421ae15`) · **PR tag:** `[fork]` · **Do not merge the 18-commit branch wholesale** — 15 of its commits are already-merged slate content under different SHAs; re-cut only the payload:

| Source commit | Payload |
|---|---|
| `34192fcd` | the family bump: workspace Cargo.toml/lock, `rust-toolchain.toml` → nightly-2026-03-05, DF-integration API adaptations (24 files) |
| `996a999e` | `PageIndexPolicy` import fix in `page_index_evaluator` tests |
| `d77bedae` | nested-insert re-pin, clippy-nightly cleanups, tokio doctest fix |

**Expect new adaptation work, not a clean cherry-pick:** FK1–FK5 and the post-#182 EXPLAIN fix were written against Arrow 57.1/DF 52.2 and have never compiled under 58.4/54.1 (arrow kernel and parquet API deltas land exactly on the FK-touched files). Budget for it.

**Rides along:** CLAUDE.md snapshot line (MSRV 1.92 → 1.94, version family), AGENTS.md/docs version mentions, `task/v1-df54-family-bump-ledger.md` carried forward with a re-cut note.

**Gate:** full battery **at the re-cut tip** — lib, datafusion all-targets, slt 9/9 (workspace form), opendal, ndf, `check-msrv` at 1.94, clippy `--all-features` on the new nightly, **and the full interop sweep (floor 52)** — the bump's original 52/52 claim was at the old base and proves nothing here. Independent morning-pass review mandatory (D4).
**Estimate:** 3–6 h. **After merge:** announce repin target in RoadMapSync; delete `chore/df54-family-bump` + `docs/df54-churn-map` only after RePark's repin SQMs (R4).

## 6 · Unit 3 — hazard-1: midpoint row-group selection *(before or with the repin)*

**Branch:** `fix/ranged-read-midpoint-rowgroups` · Scope per memory `ranged-read-rowgroup-overlap-trap`:
- `ArrowReader::filter_row_groups_by_byte_range` → keep a row group iff its **midpoint** ∈ `[start, end)` (parquet-mr `filterFileMetaDataByMidpoint` / `RangeMetadataFilter.contains`, bytecode-verified), using **real** row-group start positions from footer metadata, not the `4 + Σ compressed_size` model.
- Pins: straddling-RG fixture where fixed-size splits currently read the same group twice (mutation: revert to OVERLAP → duplicate rows → RED); exactly-once tiling property across a full split set.
- Exposure note for the PR body: offsets-aligned splits (fork/Java writers) unaffected; offsets-less external manifests (DuckDB class) are the live consumer shape.

**Estimate:** 3–4 h. **Cadence:** AC·OO (silent-duplication class).

## 7 · RePark repin (their side, after U2 merges — one repin, checklist for the announcement)

Adapt: `PartitionKey::new` → `Result` (their fixtures per RePark-Response 2026-07-25 already expect it) · `FileScanTask` Arc field construction (FK2.1) · `DeleteFilter` accessor (FK3). Still owed from the 07-26 triage: 4-vs-2 pin arithmetic reconciliation; identity "data-loss" pin rename; revert of the Group-AA bypass at repin. Their side also fixes CONSUMING.md's stale embedded rev (their 08-02 audit's own P1). New capability at repin: `iceberg.multi_partition_scan` knobs; hazard-1 status per Unit 3.

## 8 · Evidence appendix (verified 2026-08-03)

- Pin: `BigRustSparkRebuild/Cargo.toml` — five `[patch]` entries, all `rev = "b009ac158f7584a956fa9292c0e9675a411ecf0d"`.
- Topology: `git merge-base main b009ac15` = `9f2bf661`; `--is-ancestor` false both directions; `rev-list --count` = 18.
- G1-on-pin: `git diff b009ac15 a966055e -- crates/iceberg/src/scan/partition_work.rs` = empty; pinned DF `scan.rs` has 26 `multi_partition` hits.
- Versions: df54 `Cargo.toml` = rust-version 1.94 / arrow 58.4 / datafusion 54.1.0 / parquet 58.4; main = 1.92 / 57.1 / 52.2 / 57.1; toolchains nightly-2026-03-05 vs nightly-2025-10-27.
- BUG-001: fast path `crates/integrations/datafusion/src/physical_plan/delete.rs:1007`; general branch 1013–1072 (per-group check :1062); `is_unpartitioned` `crates/iceberg/src/spec/partition.rs:101`; read-side spec-equality `crates/iceberg/src/delete_file_index.rs:459,508`; `referenced_data_file` set only by `writer/base_writer/deletion_vector_writer.rs` (index doc :184 confirms Java `PositionDeleteWriter` never sets it).
- Comms gap: RoadMapSync grep — no #182/#183/PartitionKey/df54 rows; RePark-Response-2026-07-25 line 139 shows the break still "expected", not known-landed.
- Audit cross-check: 08-02 OTH-002 = pin-skew High; 08-03 OTH-002 = RePark PR-CI-skips-tests (recycled IDs — adopt carry-forward table + stable IDs in the audit prompt template).

## 9 · Answer ledger — Grok Q1–Q18 (2026-08-03; supersedes §2–§6 where noted)

**Amendments to the brief body:**
- **D3 AMENDED → strict serial** U0→U1→U2→U3 with the FK-campaign teardown/`df -h` disk protocol (100 GiB floor). Dissolves Q3: U2's base = post-U1 main tip; U1 is never rebased. *(Pending user ratification — Q2.)*
- **U1 Leg B fix shape (supersedes §4 Leg B prose):** two truncation layers, both levers in-tree, **no Cargo change under any expected shape**: (a) pos-delete writer's `WriterProperties` sets `set_statistics_truncate_length(None)` so parquet footer stats carry full `file_path` values (counters the 64-byte default); (b) pos-delete writer's `MetricsConfig` = **Full for the `file_path` column** via the existing override seam (`parquet_writer.rs:89`), which is character-for-character Java's mechanism (`MetricsConfig.forPositionDelete()` puts `MetricsModes.Full` on `DELETE_FILE_PATH` — cite the bytecode in the PR). The exactness guard already exists (`parquet_writer.rs:327/:338`) — today it correctly fail-closes on truncated stats; the fix makes them exact+full. **STOP bar:** if `set_statistics_truncate_length` is absent at parquet 57.1, land Leg A alone (legs are test-separable), Leg B becomes a rider on U2 (58.4 has it). No dep edits inside U1, ever.
- **U1 interop recipe (supersedes §4 interop):** drive evolution via the **Java core API oracle**, not Spark SQL — new suite modeled on `run-interop-multi-spec.sh`: Java `InteropOracle` creates partitioned table + data → `updateSpec().removeField(...)` (default spec now unpartitioned) → Rust DataFusion DELETE writes pos deletes → **Java reads back zero-resurrection**; control leg reversed. Fallback (only if the oracle recipe stalls): pure-Rust metadata-builder evolution + Java read of the Rust-written table, Spark-authored variant seeded as follow-up.

**Locks (mine, technical):**
| Q | Lock |
|---|---|
| Q3 | Dissolved by serial D3. Order-of-landing: U1 → main, then U2 re-cut from the new tip. |
| Q4 | **Option A locked** (single-spec condition `partition_specs.len()==1 && default_spec.is_unpartitioned()`); Option B only on explicit user preference. |
| Q6 | Standing policy confirmed: three per-unit `[fork]` Mode-A PRs; agent pushes + opens PRs; **user merges**. No mega PR. |
| Q7 | See Leg B amendment above (two-lever, in-tree, explicit STOP bar). |
| Q8 | **Leg B = bounds/derivation leg ONLY.** Do NOT set `referenced_data_file` on V2 pos-delete DataFiles — Java `PositionDeleteWriter` never sets it (`delete_file_index.rs:184`); writing it would be a written-metadata divergence. |
| Q9 | Feasible today — see interop recipe amendment. Harness already has multi-spec suites. |
| Q10 | Prose/residue by default; **flip R113 in the U1 PR only if both legs + the Java-read interop leg are green at tip** (flip rule: unit tests AND interop, per parity mandate). Re-verify the anchor at unit start. |
| Q11 | **U3 = midpoint RG selection only.** Hazard-2 stays loud-reject (FK rider); reject→suppress UX upgrade and FK5 ordinals are named seeds, not scope. |
| Q13 | Paste the two rows directly into RoadMapSync (hub-doc precedent, no PR) — but **only after the user locks the D-gates** (the rows reference the plan). Claim rows (R2) land in the same hub commit. |
| Q14 | Claim workstream name: `iceberg-rust (fork workstream)` — FK precedent. |
| Q17 | Confirmed: re-cut commits are new commits by this workstream → `[fork]` only; provenance preserved in the PR body ("re-cut from `chore/df54-family-bump`") + the carried ledger file. Never `[marci]`. |
| Q18 | Yes: copy this work order to `task/reconciliation-qb-bug001-work-order.md` on the first unit commit, with this §10 as the proposition ledger (locked answers = PROVEN props, FK pattern). |
| n1 | U2 budget widened to **4–8 h**; STOP-and-report if FK-code adaptation thrashes past it. |
| n2 | Interop floor definition (add to every gate): the sweep must **discover ≥52 suites and pass all discovered, 0 failed** — the floor guards against silent suite-loss, not just pass-rate. |
| n3 | Anti-stale-rlib protocol stays; between-unit boundaries are the risk point (the U2 family bump itself forces full rebuilds). |

**User gates (pending your word):** Q1/D1 (grant full family + MSRV 1.94 + nightly-2026-03-05, scoped to U2's branch only — recommend grant-in-full; a partial grant creates a third hybrid state nobody tests) · Q2 (ratify the D3 serial amendment) · Q5/D4 (ratify D11-style cadence for this slate: Grok octo Actor + critic-octo 8× `early_stop=false` per unit, in-octo mutation-RED execution, my morning independent pass re-runs named mutations at tip before any merge recommendation) · Q12 (recommend **A** — U3 merges before the repin announcement; RePark's DuckDB-class tables are the live exposure) · Q15 (accept: pin keeps BUG-001 until one post-U2 repin; mitigations — U0 row discloses the trigger shape to RePark; serial order puts U1 first; a cherry-pick onto their pinned line is available on request) · Q16 (approve one-shot `rm -rf org/` in U0 — untracked `javap` artifact from QC, regenerable from the jar).

## 10 · Deferred / riders (not units)

- Hygiene rider on U1 or U3: `Literal::into_any` → `Result` (`literal.rs:821/823`), `file_io.rs:116` → `.expect("set above")` house-style, delete-or-finish the dead `parquet_files_to_data_files` helper.
- REST catalog SEC cluster (SEC-004/005/006/010/011): deferred — Glue + S3 Tables are the priority catalogs; Postgres TLS class already adjudicated parity-shared (2026-07-17).
- `rm -rf org/` stray in the primary checkout (QC bytecode artifact) — one command, any time.
- Audit-harness change (user's template): mandatory carry-forward table with per-finding disposition + stable IDs across runs.

---

## 10 · Proposition ledger (Q1–Q18 answers — PROVEN 2026-08-03)

| Q | Decision |
|---|---|
| Q1/D1 | GRANTED full family for **U2 only** (DF 54.1 / Arrow 58.4 / parquet 58.4 / MSRV 1.94 / nightly-2026-03-05). U1/U3 zero-dep. |
| Q2/D3 | **Strict serial** U0→U1→U2→U3 + FK disk protocol (100 GiB floor). |
| Q3 | Dissolved by serial: U1 merges first; U2 re-cuts from post-U1 main. |
| Q4 | **Option A LOCKED** — `partition_specs.len()==1 && default_spec.is_unpartitioned()`. |
| Q5/D4 | D11: Grok octo Actor + critic-octo **8× early_stop=false**; mutation-RED in-octo for U1/U3; morning = independent Opus-class. |
| Q6 | Three per-unit `[fork]` PRs; user merges only. |
| Q7 | Leg B: `set_statistics_truncate_length(None)` + Full MetricsConfig; STOP if API missing (API present at 57.3). No Cargo in U1. |
| Q8 | Leg B = equal-bounds only; never set `referenced_data_file` on v2 pos-deletes. |
| Q9 | Java core API oracle preferred; pure-Rust evolution + Java read acceptable fallback. |
| Q10 | Prose default; flip R113 ✅ only if both legs + Java-read interop green. |
| Q11 | U3 = midpoint RG only. |
| Q12 | U3 before RePark repin announce. |
| Q13–18 | RoadMapSync paste; workstream `iceberg-rust (fork workstream)`; pin keeps BUG-001 until post-U2 repin; `rm -rf org/` in U0; `[fork]` only; brief in `task/`. |

Status: U0 DELIVERED 2026-08-03. U1 in flight.
