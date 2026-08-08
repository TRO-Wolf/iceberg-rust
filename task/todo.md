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

## IN FLIGHT — U3 / hazard-1: midpoint row-group selection (branch `fix/ranged-read-midpoint-rowgroups`)

Spec: [reconciliation-qb-bug001-work-order.md](reconciliation-qb-bug001-work-order.md) §6. Ledger:
[u3-midpoint-rowgroup-ledger.md](u3-midpoint-rowgroup-ledger.md). Zero-dependency-change unit.

- [x] **P1 — Replace the selection rule.** `ArrowReader::filter_row_groups_by_byte_range`: keep a
      row group iff `rg_start + compressed_size/2 ∈ [start, start+length)`, with `rg_start` =
      Java `getOffset(columns[0])` = `min(data_page_offset, dictionary_page_offset)` read from the
      REAL footer. Delete the `4 + Σ compressed_size` accumulator entirely; no fallback branch.
      Typed `DataInvalid` on zero-column row groups, negative offsets, and midpoint overflow.
- [x] **P2 — New discriminating pins** (T1 straddling exactly-once, T2 bloom-padded offset drift,
      T3 exactly-once partition property over a stride sweep, T4 `getOffset`/boundary/error unit
      matrix). Expectations derived from real footer metadata, never from the synthetic model.
- [x] **P3 — Repair the three self-blind tests** that build their windows with the same
      `4 + Σ compressed_size` model the production code used (`reader.rs` ~3459 / ~4949 / ~5180).
- [x] **P4 — Mutation proof** M1 (OVERLAP rule) · M2 (synthetic offsets, midpoint rule kept) ·
      M3/M4 (boundary flips) · M5 (`getOffset` → dict-wins) · M6 control (must stay GREEN).
- [x] **P5 — Amplifier 4 measured and reported**: annotate the `scan/partition_work.rs`
      split-size-1024 fixture as NON-discriminating (single row group; no live duplication pin).
- [x] **P6 — RIDER (h), reported separately**: make
      `fk5_pos_oracle_sparse_pos_deletes_multi_rg` discriminating (it is green today with
      `max_row_group_row_count = None`) and mutation-prove it RED.
- [x] **P7 — Gate + ledger + `[fork]` commit** in one `&&` chain. Interop leg: see ledger
      §Residue.
- [x] **P8 — Remediation cycle 2** (Critic S3s + Falsifier counterexamples; ledger §11): make the
      fabricated footer fixture multi-column and give it a distinct `total_byte_size` so
      `columns().first()→last()` and `compressed_size()→total_byte_size()` are RED offline; add an
      ODD-size case so `/2 → div_ceil(2)` is RED; add a REAL multi-column pin; fix the stale
      "duplicate rows" rationale in `scan/mod.rs`; re-point the amplifier-4 annotation at the
      load-bearing quantity; name the silent-row-loss and fail-closed-divergence residue.
- [x] **P9 — Remediation cycle 3** (Falsifier counterexamples; ledger §12). Plan:
      - [x] **P9a — AVRO/ORC ranged splits duplicate every row (HIGH, live, same hazard class).**
            `is_splittable` calls AVRO/ORC splittable (the Java `FileFormat` port) and
            `plan_tasks` splits unconditionally, but `process_avro_file_scan_task` /
            `process_orc_file_scan_task` never read `task.start`/`task.length` — every sub-task
            re-reads the whole file. Stop the planner emitting ranged AVRO/ORC tasks, add a
            fail-closed read guard (the `_pos` guard's shape) as defence in depth, and split the
            `scan/map.md` Debug row that currently attributes the symptom solely to parquet.
      - [x] **P9b — Pin the negative-`compressed_size` guard (MZ1 survived).** Add the case to
            the `(a)`–`(g)` semantics matrix and delete the in-tree comment claiming the public
            builder cannot construct a negative size — it can, and the Falsifier did.
      - [x] **P9c — `compressed_size()` overflows before any guard runs.** parquet-rs sums the
            column `total_compressed_size` values with an unchecked `i64` `sum()`; a corrupt
            footer panics (debug) or wraps (release). Sum it here with `checked_add` → typed
            `DataInvalid`.
      - [x] **P9d — Narrow the over-stated residue sentence.** An understated manifest
            `file_size_in_bytes` fails LOUDLY at footer decode, not silently; the silent-row-loss
            residue is real only for under-covering windows / non-tiling `split_offsets`.
      - [x] **P9e — Re-run the full gate and the whole mutation sweep in an ISOLATED tree**
            (`git archive | tar -x`, own `CARGO_TARGET_DIR`, `touch` after extraction) — the
            shared worktree was carrying a sibling agent's uncommitted mutation during cycle 2.
- [x] **P10 — Remediation cycle 4** (Critic C-S2/C-S3 + Falsifier F-1..F-4; ledger §13). Plan:
      - [x] **P10a — F-1: `split` evaporated a whole-file `length == 0` task (HIGH, silent total
            row loss).** `split_fixed_size` starts `remaining = self.length` and loops
            `while remaining > 0`, so the legacy sentinel returned ZERO sub-tasks and
            `mod.rs`'s `split_tasks.extend(...)` dropped the file — `plan_files` returned it,
            `plan_tasks` read 0 rows, no error. Cycle 3 made it an ASYMMETRY (AVRO passes such a
            task through; both reader guards bless the spelling). Fixed by returning `[self]`,
            pinned at the unit AND read level, mutation-proven RED.
      - [x] **P10b — F-2: pin the byte-range ENTRY gate** (`task.start != 0 || task.length != 0`).
            Weakening it to `task.length != 0` was GREEN across 3,146 tests: it turns the empty
            window `[start, start)` into a whole-file read.
      - [x] **P10c — F-3: pin the START half of both whole-file guards.** `start = 1,
            length = file_size_in_bytes` is a genuine window that
            `reject_ranged_whole_file_task` (and the copy-pasted `_pos` guard, which had the same
            gap) would ACCEPT without the `task.start == 0 &&` clause.
      - [x] **P10d — F-4: distinguish the two parquet-mr offset helpers.** `dict Some(0)` must
            still win (`ParquetMetadataConverter.getOffset`, no `> 0`), unlike
            `ColumnChunkMetaData.getStartingPos` which the split-offset WRITER uses.
      - [x] **P10e — C-S2: GAP_MATRIX row R148** — corrected the `FileScanTask::split`
            parenthetical and added the NAMED divergence for the AVRO/ORC decline + the sentinel
            passthrough; anchors green at 75 rows.
      - [x] **P10f — C-S3: lessons entry** — never pipe a gate command into `tail`/`grep`/`head`
            in a verification `&&` chain (the pipeline's status is the LAST command's).

- [x] **P11 — Remediation cycle 5** (Critic ORC-S2 + Falsifier F5..F9; ledger §14). Plan:
      - [x] **P11a — F5: `split` RELOCATED an already-ranged task's window (HIGH, wrong bytes).**
            Both real branches anchor at 0, so re-splitting a `start != 0` parent read bytes it
            never owned and dropped its tail (measured: parent ids 20..59, products ids 0..59).
            Fixed by returning `[self]` for `start != 0` — Java forecloses the shape structurally
            (`SplitScanTask` is not `SplittableScanTask`); pinned at the unit AND read level.
      - [x] **P11b — F7: `plan_tasks` split unconditionally under a `_pos` projection**, which the
            reader then rejects — a total outage of `_pos` on the `plan_tasks` / `PartitionWork`
            seam while `to_arrow()` worked. Suppression hoisted to `plan_tasks`.
      - [x] **P11c — Critic S2 / F9: pin the ORC call site** of
            `reject_ranged_whole_file_task` (deleting it was GREEN; only AVRO was pinned).
      - [x] **P11d — F6: second interop sabotage leg** — mutate the OFFSET SOURCE to the synthetic
            `4 + Σ compressed_size` model; the D2 JAVA verify must go RED with a per-window
            comparison signal. Both Rust legs are blind to it. Plus `JAVA_ROWS` declared instead
            of derived in `assert_exactly_once`.
      - [x] **P11e — notes promoted**: direct four-arm tests for `is_splittable` /
            `reader_honors_byte_range`; the negative-split-offset typed error.
      - [x] **P11f — F8 and the `can_expand` pin DECLINED with executed equivalence proofs**
            (mutants E1/E2/E3 GREEN by construction — ledger §14.6).

---


> **Archival log.** Last pass: 2026-07-26 (pass 6 — size trigger, 2,012 lines; run by the RePark
> workstream under the hub concurrency-protocol claim of the same date) →
> [todo-archive/2026-07_audit-hardening-engine-trust.md](todo-archive/2026-07_audit-hardening-engine-trust.md)
> (17 narratives, 2026-07-01 → 07-26) +
> [todo-archive/2026-06_charter-8hour-blocks.md](todo-archive/2026-06_charter-8hour-blocks.md)
> (18 narratives — the June charter / 8-hour blocks / superseded queue). The 2026-07-01 open queue
> kept live and reconciled in place; 7 buried open items lifted to Carried-forward. Prior passes:
> 2026-06-13 (pass 5 — Wave-6/7 → the wave6-wave7 file), 2026-06-12 (pass 4 → wave5), 2026-06-12
> (pass 3 → wave3-wave4), 2026-06-11 (pass 2), 2026-06-09 (pass 1). Procedure:
> [skills/compaction.md](../skills/compaction.md) §Todo Archival.

## POST-BUNDLE QUEUE (2026-07-26, signed off) — D7 order + D8 toHumanString approval

Spec: [post-bundle-queue-2026-07-26-brief.md](post-bundle-queue-2026-07-26-brief.md). Signed
order: QA (R117 cross-task over-delete) → Unit 3 breaking → QB (writer path bounds) → H7-S2 →
H7-P1, with QC (R161-completion toHumanString parity, format-visible, D8-APPROVED) alongside
Unit 3. Mode A per-unit PRs; SEPMO v2.3 duties. Context at signing: nightly interop FULLY GREEN
(scan-plan arc live-proven); all review branches pruned; main `a08a0957`.

- [x] **QA — per-task DeleteFilter scope** (S1 read correctness; branch
      `fix/delete-filter-per-task-scope`): scope delete APPLICATION to `task.deletes()`
      Java-exact while keeping load caching; RED-first the recorded category=b repro (id 30
      resurrects correctly); eq-delete path checked same-class; R117 🟡→✅ IF this was the sole
      blocker (Actor adjudicates; anchors gate).
      **MERGED #176 (`14921e78`) 2026-07-26**, content-verified R4 (per-task resolve +
      contribution install on main; R117 cell reads ✅). AC converged cycle 1, zero S1/S2 (2 S3
      residues: eq KEYSET fast-path pin coverage; theoretical claim-key namespace collision).
      Java-exact shape: per-SOURCE contribution maps (load-once preserved; G8 claim/notify/
      `Failed` machinery byte-untouched) + per-task application over `task.deletes()` with a
      per-(file, claims) memo; the defect's unattributed path-keyed API REMOVED, `reader.rs`
      unchanged. Eq-deletes PROVEN not-same-class (pinned + mutation-proven); DVs covered.
      Interop crosstask leg added (id 30 survives == Java `{10,30,40,60}`). Lib 2977.
- [ ] **Unit 3 — breaking** (`PartitionKey::new -> Result` + `CurrentFileStatus`): after QA
      merges; RoadMapSync warning to RePark BEFORE their next repin.
- [ ] **QC — toHumanString parity** (D8-approved, format-visible; FIXED/BINARY hex→base64 +
      Unknown taxonomy + identity(float/double)): alongside Unit 3; format-stability attestation.
- [x] **QB — delete-writer file_path bounds** (fork file-scoped deletes must self-identify;
      parquet-rs 64-byte stat truncation; investigate-first, STOP on any Cargo.toml need).
      **MERGED #184 (`7e26c2a0`) 2026-08-03**, content-verified R4. Landed with BUG-001 (the
      evolved-spec partition stamp) as one PR — both legs of position-delete attachment were
      broken simultaneously and each masked the other. No Cargo.toml need materialised
      (`set_statistics_truncate_length` was already on the pinned parquet). R113 stays 🟡 (owes
      the Java-read interop leg on the evolved-DROP shape); R117 note added.
- [ ] **H7-S2 → H7-P1** (re-scope at signing; P1's footgun pre-condition unchanged).

**Queue state 2026-08-05.** Since signing, the line has also absorbed: QD/QE (#178/#179, the two
RePark filings — manifest schema tolerance + s3tables replace), the ledger archive (#177), interop
weekly cadence (#180), perf waves A–E (#181), the 07-31 slate (#182), the FK1–FK5 MoR perf campaign
(#183), the V0 DF 52→54 churn map (#185), and the **DF 54.1 / arrow 58.4 family bump re-cut
(#187)** — which moved MSRV 1.92 → 1.94 and toolchain to nightly-2026-03-05.

**Remaining in signed order: Unit 3 (breaking) + QC alongside → H7-S2 → H7-P1.**

Two things now owed that were not at signing:

1. **RoadMapSync comms for MSRV 1.92 → 1.94 and the DF/arrow/parquet floors** (#187). Downstream-
   visible; RePark must see this before its next repin. Unit 3's own RoadMapSync warning
   (`PartitionKey::new -> Result` + `CurrentFileStatus`) should go out in the same message.
2. **RePark is still pinned at `b009ac15`** — the tip of the superseded pre-recut df54 branch,
   which predates BUG-001 (#184) and the FK campaign (#183). That branch was deleted 2026-08-05;
   the pinned commit is preserved by tag `archive/df54-family-bump-b009ac15` so the pin stays
   reachable. **Retire that tag once RePark repins to main.** Repin target: `3f63a6c7` (#187).

## ACTIVE (2026-07-01): Engine-first closeout — re-ranked open queue

Supersedes the 2026-06-13 queue below. **One home for PRIORITY: this list** (the Roadmap's
re-anchor carries a "Priority home" pointer here; do not grow ranked lists elsewhere). Re-ranked
after the 2026-07-01 review pass, which reconciled the old queue (most items had landed) and
surfaced two new items. Statuses live ONLY in
[docs/parity/GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md).

- [x] **1. Commit-outcome taxonomy (`CommitStateUnknown`)** — DONE, merged #144 (2026-07-08);
      narrative archived in
      [todo-archive/2026-07_audit-hardening-engine-trust.md](todo-archive/2026-07_audit-hardening-engine-trust.md).
      *(Reconciled 2026-07-26, archival pass 6 — the box was never flipped.)* Was: NEW, GAP_MATRIX row R157. An
      unknown-outcome `ErrorKind` (or flag) honored by the retry gate + sent-vs-unsent
      transport-error classification in the Glue / S3 Tables / REST / SQL catalogs +
      surfaced-no-retry-no-cleanup semantics matching Java + mock-catalog tests. Buildable
      WITHOUT AWS creds. Slots ahead of CDC: the named consumer commits continuously against
      S3 Tables, whose service-side maintenance ALSO commits concurrently — an ambiguous outcome
      today risks a duplicate commit (see the row cell). The credentialed conformance slice
      stays with item 6.
- [ ] **2. CDC row-level changelog** (re-anchor item 2) — **RE-CHARACTERIZED 2026-07-31 (G3
      ledger):** mostly **parity-correct as-is**. `ChangelogOperation::UpdateBefore` /
      `UpdateAfter` are declared for API parity (`scan/task.rs`) but are **never emitted by the
      core planner** — Java 1.10.0 `BaseIncrementalChangelogScan` only produces INSERT/DELETE
      task kinds; collapsing delete+insert into update pairs is an **engine-side** step (Spark
      `ChangelogIterator`), not owed by `iceberg-core`/`iceberg-api`. Residual (if any engine
      pull) is accepting ranges that carry row-level DELETE manifests
      (`IncrementalChangelogScan` is whole-data-file-level today) — engine-gated, not a
      library correctness hole.
- [ ] **3. ORC/Avro DATA-read residue** (re-anchor item 3) — footer codec / nested + V3 types /
      the Avro `timestamptz` mapping — pull only if the engine queries non-parquet tables.
- [ ] **4. ENGINE_CONTRACT.md recipes → NORMATIVE** — bytecode/oracle-verify the
      isolation-level → validation table (DRAFT landed 2026-07-01,
      [docs/ENGINE_CONTRACT.md](../docs/ENGINE_CONTRACT.md)) against Java 1.10.0
      `SparkWrite` / `SparkCopyOnWriteOperation` / `SparkPositionDeltaWrite`, one interop
      conflict scenario per cell.
- [x] **5. Nightly interop CI** — DONE: the Nightly Interop workflow runs the suites on `main`
      on a schedule and is fully green as of 2026-07-26 (scan-plan arc live proof landed).
      *(Reconciled 2026-07-26, archival pass 6 — the box was never flipped.)* Was: run the
      `dev/java-interop/` suites on a schedule. The oracle is
      the model-tier equalizer only if it runs unprompted; this is the cheap 80% of Phase 7.
- [ ] **6. Real-catalog hardening (credentialed)** — Glue + S3 Tables conflict/retry conformance
      + item 1's real-catalog classification slice; scheduled with the user (needs AWS creds).

**In-flight (off-matrix, user-gated — staged work, not ranked above): H7 DML
streaming/pushdown** on the DataFusion reference impl (scope converged 2026-06-30; engine-first
hardening of the #124 DML loop, flips no matrix row). **H7-S1** (MoR DELETE/UPDATE streaming) is
PUSHED — branch `parity/h7-s1-mor-streaming` (d2fecef6), gate green, all Critics/audits
converged — awaiting user merge. Remaining stages, each its own ladder when the user resumes:
**H7-S2** (COW streaming — the two `copy_on_write_*` fns, two-pass→bounded refactor) and
**H7-P1** (pushdown pruning — must FIRST thread the raw `Vec<Expr>` through both exec structs,
and resolve the `NOT`-over-dropped-conjunct under-delete footgun before any
`with_filter(convert_filters_to_predicate)`; pushdown may ONLY prune, never replace the exact
post-scan filter).

PULL-BASED / DEMOTED: unchanged from the Roadmap re-anchor — link, do not restate.

## Carried-forward open items (detail in todo-archive/)

Lifted verbatim from archived narratives by archival pass 6 (2026-07-26). Status caveat for the
three 07-17 audit units below: **no merged PR names them** (sibling unit D from the same list
shipped as #159), but they were never explicitly closed either — verify against the code /
GAP_MATRIX before starting one. Full context:
[todo-archive/2026-07_audit-hardening-engine-trust.md](todo-archive/2026-07_audit-hardening-engine-trust.md)
(the "OVERNIGHT BLOCK (2026-07-17)" section) and
[todo-archive/2026-06_charter-8hour-blocks.md](todo-archive/2026-06_charter-8hour-blocks.md)
(the "SUPERSEDED 2026-07-01" queue + "BLOCK 10").

- [x] **B (OO max): MoR eq-delete panic/hang** — **LANDED** (reconciled 2026-07-31 G3). Cite
      `caching_delete_file_loader.rs` (`equality_ids: None` → `DataInvalid`, not unwrap) and
      `delete_filter.rs` (oneshot sender-drop → terminal `Failed` + `notify_waiters`). Merged
      with the 07-18 audit bundle (#160 / follow-ons); content-verified on main.
- [x] **C (OO max): predicate serde arity validation (SAF-004)** — **LANDED** (reconciled
      2026-07-31 G3). Custom `Deserialize` on Unary/Binary/Set validates op/arity at the wire
      boundary; visitor dispatch returns typed `Err` instead of `panic!`. Pins in
      `expr/predicate.rs` `serde_arity_pins`.
- [x] **E (OO max): typed error kinds** — **LANDED** (reconciled 2026-07-31 G3). SQL helpers
      (`no_such_*` / `*_already_exists_*`) emit typed kinds; HMS thrift mappers in
      `crates/catalog/hms/src/error.rs` + call-site wiring in `catalog.rs` map
      `NoSuchObjectException` / `AlreadyExistsException` (and drop-namespace not-empty) to
      `NamespaceNotFound` / `TableNotFound` / `NamespaceAlreadyExists` / `TableAlreadyExists` /
      `NamespaceNotEmpty`. Config: empty required fields → `DataInvalid`; malformed/
      unresolvable address and missing StorageFactory → `Unexpected`. Unit G3
      (`fix/hms-typed-error-kinds`) closed the ledger + residual config pins; mapper unit
      tests offline.
- [ ] **2. Multi-spec write interop** — STILL OPEN (reconciled 2026-07-01; citations corrected
      same day). TWO distinct residues: (a) the manifest-merge LAYOUT gap —
      `MergeManifestProcess` is not routed into the non-append merging actions (the `RowDelta`
      row, currently row R106 — the old "row 94" pointer was dead); (b) the writer-layer spec
      threading — `DataFileWriter`/`DeletionVectorWriter` stamp the table default spec (row R110)
      — plus the multi-spec-DATA interop slices on the merging actions (one slice landed: #69,
      multi-spec RowDelta DELETE-commit); `fast_append` multi-spec is ✅ (Z2 — the template).
- [ ] **Multi-spec MERGING-path wiring gap** — the companion detail of item 2(a) above: route
      `MergeManifestProcess` into the non-append merging actions. The RE-CHARACTERIZED
      2026-06-16 narrative (what is already ported vs the real gap) is in the "BLOCK 10"
      section of the 2026-06 archive — read it before scoping; the earlier framing was a
      phantom bug.
- [ ] **4. geometry / geography types** — HALF DONE (reconciled 2026-07-01): `unknown` landed ✅
      2026-06-17 (interop-proven); geometry/geography remain ❌ and are DEMOTED to opportunistic
      by the 2026-06-21 engine-first re-anchor (a query engine does not pull them).
- [ ] **7. [PARKED] encryption** — reconciled 2026-07-01: the Glue / S3Tables VIEWS half is
      RESOLVED as parity-correct-unsupported (rows R126/R127, verified 2026-06-17 — NOT owed);
      encryption remains ❌ and is DEMOTED to opportunistic by the engine-first re-anchor. The
      credentialed real-catalog hardening piece moved to the 2026-07-01 queue (item 6).

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
- [todo-archive/2026-06_wave5.md](todo-archive/2026-06_wave5.md) — Wave 5 (PRs #39–#41; pass-scoped).
- [todo-archive/2026-06_wave6-wave7.md](todo-archive/2026-06_wave6-wave7.md) — Waves 6–7 (PRs #43–#47; pass-scoped): the I1/I2/I3 interop increments + O1/O2/O3 + R1/R2/R3.
- [todo-archive/2026-07_audit-hardening-engine-trust.md](todo-archive/2026-07_audit-hardening-engine-trust.md)
  — the 2026-07-01 → 07-26 audit / hardening / engine-trust era (pass 6).
- [todo-archive/2026-06_charter-8hour-blocks.md](todo-archive/2026-06_charter-8hour-blocks.md)
  — the 2026-06-13 → 06-19 charter / 8-hour blocks / superseded queue (pass 6).
- Index: [todo-archive/map.md](todo-archive/map.md).
