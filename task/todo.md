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


> **Archival log.** Last pass: 2026-06-13 (pass 5 — Wave-6/Wave-7 union, 466 lines) →
> [todo-archive/2026-06_wave6-wave7.md](todo-archive/2026-06_wave6-wave7.md) (9 spent increment
> narratives — R1/R2/R3, I1/I2/I3, O1/O2/O3, PRs #43–#47; the open queue refreshed in place to the
> 2026-06-13 re-audit). Prior passes: 2026-06-12 (pass 4 — post-Wave-5 union, 680 lines → the
> wave5 file), 2026-06-12 (pass 3 — 2,358 lines → the wave3-wave4 file), 2026-06-11 (pass 2),
> 2026-06-09 (pass 1). Procedure: [skills/compaction.md](../skills/compaction.md) §Todo Archival.

## ACTIVE (2026-07-01): Engine-first closeout — re-ranked open queue

Supersedes the 2026-06-13 queue below. **One home for PRIORITY: this list** (the Roadmap's
re-anchor carries a "Priority home" pointer here; do not grow ranked lists elsewhere). Re-ranked
after the 2026-07-01 review pass, which reconciled the old queue (most items had landed) and
surfaced two new items. Statuses live ONLY in
[docs/parity/GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md).

- [ ] **1. Commit-outcome taxonomy (`CommitStateUnknown`)** — NEW, GAP_MATRIX row R157. An
      unknown-outcome `ErrorKind` (or flag) honored by the retry gate + sent-vs-unsent
      transport-error classification in the Glue / S3 Tables / REST / SQL catalogs +
      surfaced-no-retry-no-cleanup semantics matching Java + mock-catalog tests. Buildable
      WITHOUT AWS creds. Slots ahead of CDC: the named consumer commits continuously against
      S3 Tables, whose service-side maintenance ALSO commits concurrently — an ambiguous outcome
      today risks a duplicate commit (see the row cell). The credentialed conformance slice
      stays with item 6.
- [ ] **2. CDC row-level changelog** (re-anchor item 2) — `UpdateBefore`/`UpdateAfter` + accepting
      ranges that carry row-level DELETE manifests (`IncrementalChangelogScan` is
      whole-data-file-level today).
- [ ] **3. ORC/Avro DATA-read residue** (re-anchor item 3) — footer codec / nested + V3 types /
      the Avro `timestamptz` mapping — pull only if the engine queries non-parquet tables.
- [ ] **4. ENGINE_CONTRACT.md recipes → NORMATIVE** — bytecode/oracle-verify the
      isolation-level → validation table (DRAFT landed 2026-07-01,
      [docs/ENGINE_CONTRACT.md](../docs/ENGINE_CONTRACT.md)) against Java 1.10.0
      `SparkWrite` / `SparkCopyOnWriteOperation` / `SparkPositionDeltaWrite`, one interop
      conflict scenario per cell.
- [ ] **5. Nightly interop CI** — run the `dev/java-interop/` suites on a schedule. The oracle is
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

## ACTIVE UNIT (2026-07-01b): review follow-ups 1+2 — gate needles + stable row anchors

User-directed (2026-07-01, post-merge of #140/#141/#142): implement follow-ups 1 and 2 from the
review-series closeout. One PR, branch `infra/review-followups-2026-07-01`. Also carries the
user's seam-status decision record (datafusion integration promoted to supported product surface —
ENGINE_CONTRACT §1 + Roadmap, committed first as its own decision commit).

- [x] **1. Broaden the artifact gate** (`scripts/check_agent_artifacts.sh`) — Critic LOW-1:
      add the function_results tag family + bare opening tags (`invoke name=` / `parameter name=`,
      concatenation-assembled as before); case-insensitive matching (uppercase variants);
      `<result>`/`<output>` deliberately EXCLUDED as too generic (false-positive risk — document);
      built-in SELF-TEST that plants every needle via a TEMP-COPY index (`GIT_INDEX_FILE`) and
      hard-fails if any needle goes undetected (a gate that cannot detect its own probe is
      vacuous — the sabotage-must-hard-fail doctrine). Red/green re-proof per new needle class.
- [x] **2. Stable matrix row anchors** — the durable fix for [citation drift]. Stamp every
      capability row's first cell with a permanent ID: `| R<n> · <name> |` where n = the row's
      file line number at stamping time (so every live citation just renumbered 2026-07-01 maps
      1:1). New rows take the next unused ID (R158+), insertable anywhere; IDs never reused.
      New `scripts/check_matrix_anchors.sh` (make target + CI step, mirroring the artifacts gate):
      (a) every data row anchored exactly once, (b) IDs unique, (c) every `row R<n>` citation in
      the live docs resolves to an existing anchor, (d) the 5-pipe audit AUTOMATED (was manual
      per CLAUDE.md). Convention note added BELOW the table (zero row-line movement). Sabotage
      proofs: duplicate ID / unstamped row / dead citation / 6-pipe row each proven RED.
- [x] **3. Citation migration** — live docs (Roadmap.md, docs/, todo ACTIVE + the 2026-07-01
      reconciliation lines, CLAUDE.md convention bullet) move to `row R<n>` form; quoted-historical
      spots and dated archives deliberately stay bare-N (they cite historical numbering epochs).
- [ ] **4. Verify + Critic** — verification fan-out (gate probes incl. new classes; anchor +
      migration correctness) → independent Critic → push on CONVERGED (user merges).

NAMED FOLLOW-UP (Critic LOW, 2026-07-01 — not this PR): ~20 stale bare `GAP_MATRIX row N`
citations live in `crates/` source/test comments (e.g. `scan/task_group.rs`,
`tests/interop_scan_plan.rs`, `aggregate_evaluator.rs`), most drifted under current numbering —
migrate them to `row R<n>` form and add `crates/` to the anchor checker's scan pathspec (touches
Rust files, so it rides a code PR, not this docs/CI one).

---

## SUPERSEDED 2026-07-01 — was ACTIVE (2026-06-13): Near-full-parity open queue

> Priority now lives ONLY in the 2026-07-01 queue above. This section is retained as the
> reconciliation record; landed items are flipped below with pointers (per this file's own
> flip-the-checkbox rule) — statuses live ONLY in the GAP_MATRIX.

Directive (user, 2026-06-11): run this fork's Roadmap to **almost the full 1:1 Java replacement**.
Waves 3–7 landed PRs #28–#47 (write-engine closeout; maintenance actions end-to-end incl.
Compute{Table,Partition}Stats + the iceberg-sketches crate; the variant arc; stage_only + WAP; views
end-to-end; SQL-catalog CAS; and the theta/view/WAP/partition-stats interop chains). This queue is the
**2026-06-13 re-audit's** ranked next-work; **statuses live ONLY in
[docs/parity/GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md)** — link, do not restate cells.

> **Queue re-audited 2026-06-15 against the live suite + GAP_MATRIX (Opus).** The prior #1
> ("write-action DATA-level interop FIRST") was already DONE — `interop_write_data.rs` +
> `run-interop-write-data.sh` landed the data-level round-trips for delete/overwrite/replace/rewrite
> (+ partitioned) and merge (one-bin + multi-bin), both directions, 2026-06-11 (GAP_MATRIX rows
> 90-95). The residue that actually keeps rows 91-95 at 🟡 is the phrase repeated in every cell:
> **"multi-spec / conflict-validation paths NOT covered."** The queue below is re-ranked to that
> reality. Statuses live ONLY in [docs/parity/GAP_MATRIX.md](../docs/parity/GAP_MATRIX.md).

Ranked, highest-value first:

- [x] **1. Conflict-validation interop** — DONE 2026-06-15/16 (PRs #64–#68; #69 was the
      multi-spec Wave-2 slice, item 2 — range corrected 2026-07-01): proven BOTH
      directions for all 5 write actions (C1 OverwriteFiles first, then DeleteFiles / RowDelta /
      ReplacePartitions / RewriteFiles). Reconciled 2026-07-01 — the checkbox had never been
      flipped.
- [ ] **2. Multi-spec write interop** — STILL OPEN (reconciled 2026-07-01; citations corrected
      same day). TWO distinct residues: (a) the manifest-merge LAYOUT gap —
      `MergeManifestProcess` is not routed into the non-append merging actions (the `RowDelta`
      row, currently row R106 — the old "row 94" pointer was dead); (b) the writer-layer spec
      threading — `DataFileWriter`/`DeletionVectorWriter` stamp the table default spec (row R110)
      — plus the multi-spec-DATA interop slices on the merging actions (one slice landed: #69,
      multi-spec RowDelta DELETE-commit); `fast_append` multi-spec is ✅ (Z2 — the template).
- [x] **3. Builder-surface interop flips** — DONE 2026-06-16: `case_sensitive` +
      `delete_from_row_filter` interop-proven ✅ (the rows this queue numbered 134/135 under the
      2026-06-13 line numbering). Reconciled 2026-07-01.
- [ ] **4. geometry / geography types** — HALF DONE (reconciled 2026-07-01): `unknown` landed ✅
      2026-06-17 (interop-proven); geometry/geography remain ❌ and are DEMOTED to opportunistic
      by the 2026-06-21 engine-first re-anchor (a query engine does not pull them).
- [x] **5. `RewritePositionDeleteFiles` + the `ActionsProvider` maintenance wrappers** — DONE
      2026-06-17 (✅ per the Maintenance rows; `DeleteReachableFiles` + `ConvertEqualityDeleteFiles`
      interop-proven). Reconciled 2026-07-01.
- [x] **6. `BatchScan` / `ScanTaskGroup` + `ExpressionParser` JSON** — DONE 2026-06-17 (all
      three ✅, interop-proven: `BatchScan`, `planTasks` split planning, the JSON expression
      (de)serializer). Reconciled 2026-07-01.
- [ ] **7. [PARKED] encryption** — reconciled 2026-07-01: the Glue / S3Tables VIEWS half is
      RESOLVED as parity-correct-unsupported (rows R126/R127, verified 2026-06-17 — NOT owed);
      encryption remains ❌ and is DEMOTED to opportunistic by the engine-first re-anchor. The
      credentialed real-catalog hardening piece moved to the 2026-07-01 queue (item 6).

Recently landed (2026-06-11 → 06-13) — status lives in the GAP_MATRIX rows; pointers only:

- Write-action DATA-level interop (delete/overwrite/replace/rewrite + partitioned + merge one-bin /
  multi-bin), both directions — `interop_write_data.rs` + `run-interop-write-data.sh`. Rows 90-95.
- `case_sensitive(bool)` on DeleteFiles/OverwriteFiles/RowDelta (default true; narrowed out of
  ReplacePartitions) — row 134. Interop deferred.
- `DeleteFiles.delete_from_row_filter(Predicate)` — row 135. Interop deferred.
- `RewriteFiles` DELETE-file ADD surface (`add_delete_file` / `_with_sequence_number` + 4-arg
  `rewrite_files_with_deletes`, third precondition reachable) — rows 95/140. Interop deferred.

See the 2026-06-13 GAP_MATRIX provenance block for per-row status and residue.

## CHARTER (2026-06-15, Opus): conflict-validation + multi-spec interop — AC·OO groups

User-approved 8h charter (2026-06-15). Close the "conflict-validation paths NOT covered" residue on
the write-action rows (C1 OverwriteFiles ✅ #64), then multi-spec. EVERY sequence is one **AC·OO
group** = a coherent PR-unit run as **Opus Actor → Opus Critic** (the user lifted the single-agent
default + named the tier). **One PR per group**; rebase on freshly-merged `main` between groups. The
C1 increment (`interop_overwrite_conflict.rs` + `OverwriteConflictOracle` + `run-interop-overwrite-conflict.sh`)
is the harness template; per-group loop = Actor builds + drives the LIVE oracle to green + offline
gate → Critic adversarially re-verifies (sabotage truly fails, D1/D2 independence, claim-vs-Java-1.10.0,
done-bar = unit + interop both directions, de-triplication, no over-scope) → finalize + PR.

Wave 1 — conflict-validation closeout (order locked):

- [x] **AC·OO #1 — C4 ReplacePartitions** (row 92) — **DONE 2026-06-15.** Partition-scoped conflict
      (`file_in_replaced_partition`, no filter); 2 scenarios (replaced-partition→REJECT, other→ACCEPT)
      both directions + fail-closed sabotage. Opus Actor → Opus Critic converged (2 cycles; cycle-1
      caught a masked-sabotage defect, fixed + independently re-verified). Files: `interop_replace_partitions_conflict.rs`,
      `run-interop-replace-partitions-conflict.sh`, `ReplacePartitionsConflictOracle`. Row 92 stays 🟡.
- [x] **AC·OO #2 — C3 RowDelta** (row 94) — **DONE 2026-06-15.** All THREE axes proven both directions:
      `validate_no_conflicting_data_files` (filter/metrics, C1 shape) + `_delete_files` (concurrent
      y-keyed eq-delete add) + `validate_data_files_exist` (concurrent OVERWRITE removal). 6 scenarios
      (reject+accept per axis). Opus Actor → Opus Critic converged in 1 cycle; Critic disabled each
      axis's validation in turn to prove per-axis non-vacuity. Files: `interop_rowdelta_conflict.rs`,
      `run-interop-rowdelta-conflict.sh`, `RowDeltaConflictOracle`. Row 94 stays 🟡.
- [x] **AC·OO #3 — C2 DeleteFiles** (row 93) — **DONE 2026-06-16.** Single `validate_files_exist`
      axis; 2 scenarios (same-file→REJECT, different-file→ACCEPT) both directions + sabotage
      (semantic-rollback + truncate). **Critic caught a MEDIUM** (D1 vacuous on the axis) but
      wrongly marked CONVERGED; orchestrator caught the contradiction. The Critic's fix (assert the
      reject message) proved FLAKY on my mutation test — Rust has TWO racing reject paths for a
      removed target (the `validate_files_exist` axis vs an UNCONDITIONAL by-path `process_deletes`
      check). Resolved HONESTLY: D2 isolates the axis (strip Java's flag → ACCEPT; Java gates the
      check on the flag, Rust's by-path is unconditional — a documented mechanism divergence), D1
      corroborates the DECISION, the axis is pinned by `delete_files.rs` unit tests. Row 93 stays 🟡.
      Files: `interop_deletefiles_conflict.rs`, `run-interop-deletefiles-conflict.sh`, `DeleteFilesConflictOracle`.
- [x] **AC·OO #4 — C5 RewriteFiles** (row 95) — **DONE 2026-06-16.** `validate_no_new_deletes_for_data_files`,
      the seq-preservation + position-vs-equality nuance: 4 scenarios both directions —
      no-seq+eq→REJECT, seq+eq→ACCEPT (ignored), seq+position→REJECT (always fatal), disjoint→ACCEPT.
      Actor engineered AROUND the C2 trap (A live at S0 AND S1, so only the axis can reject; confirmed
      by axis-message assertion + 2 mutation-swaps). Hardened loop; Critic source-disabled the
      validation to prove non-vacuity → converged 1 cycle, NO_FINDINGS. Files:
      `interop_rewritefiles_conflict.rs`, `run-interop-rewritefiles-conflict.sh`, `RewriteFilesConflictOracle`.
      Row 95 stays 🟡. **Wave 1 COMPLETE — all 5 write-action conflict rows interop-proven.**

Wave 2 — multi-spec write interop (stretch):

- [x] **AC·OO #5 — MS** — **DONE 2026-06-16.** RowDelta multi-spec DELETE commit: one `row_delta`
      adding position deletes under spec 0 AND spec 1 → TWO per-spec DELETE manifests, canonical view
      byte-matches Java 1.10.0 (3 directions + 4 sabotages incl. SB4 wrong-spec rendering). Closes the
      "multi-spec delete commits" residue on row 94. Converged 1 cycle, NO_FINDINGS; Critic wrote its
      own collapse probe to confirm `grouping_is_load_bearing`. Files: `interop_multispec_merge.rs`,
      `run-interop-multispec-merge.sh`, `MultiSpecMergeOracle`.
      **⚠ NEW PARITY FINDING (follow-on, surfaced by the Actor):** on the MERGING path Java
      force-merges every spec group NOT containing the iterator-`first` manifest (order-dependent,
      ignores min-count-to-merge) — Rust's merging producer does NOT mirror this. Documented in row 94.
      The multi-spec DATA cases (overwrite/rewrite carrying old-spec + adding new-spec) are deferred
      behind this asymmetry. **→ tracked as new queue item below.**
      **⚠ RE-CHARACTERIZED 2026-06-16 (post-review, code-verified): the framing above is IMPRECISE —
      `merge_append.rs` DOES port the force-merge faithfully; the real gap is that the NON-APPEND actions
      route through `DefaultManifestProcess` (no merge). See the corrected follow-on item below + the
      re-characterized GAP_MATRIX row 94.**

Wave 3 — builder-surface flips (stretch, only if 1+2 beat estimates):

- [x] **AC·OO #6 — BF** — **DONE 2026-06-16 (#TBD).** DeleteFiles 2-for-1 vehicle
      (`delete_from_row_filter` + `case_sensitive`) proven bidirectionally vs Java 1.10.0: 5 scenarios
      (filter DELETE / KEEP-complement / PARTIAL-error + case-insensitive-match + case-sensitive-reject),
      live oracle GREEN (D1 + D2 + semantic-rename & truncate sabotages, exit 0). **Row 135
      (`deleteFromRowFilter`) → ✅** — the FIRST ✅ flip since Wave 1 began (2 named fail-safe
      divergences: `markedForDelete` short-circuit + empty-match no-op, both Rust-stricter, kept out of
      the set). **Row 134 stays 🟡** — its `caseSensitive` SLICE is now ✅ interop-proven (shared
      `bind(schema, case_sensitive)` site; the conflict-filter family argued-equivalent via the same bind
      call + 25 unit tests + C1/C3 interop), but the row's conflict-detection surfaces
      (`validateNoConflictingData`/`conflictDetectionFilter`, ReplacePartitions `conflict_detection_filter`,
      `validateAppendOnly`) remain unported. Converged 1 cycle; Critic CONVERGED (both non-vacuity gates
      mutation-proven; orchestrator re-ran the live oracle + offline gate + fixed 2 LOW nits). Files:
      `interop_builder_flips.rs`, `run-interop-builder-flips.sh`, `BuilderFlipsOracle`.
      **Wave 3 COMPLETE — the 8h AC·OO charter (Waves 1 + 2 + 3) is fully landed.**

## POST-CHARTER (2026-06-16, Opus) — maintenance/actions surface (AC·OO continues)

Stance-review-ranked next surface (ActionsProvider + the cheap maintenance wrappers — orchestrate ✅
primitives). AC·OO via Workflow, one PR per unit, same finalize cadence. Track A (multi-spec MERGING-path
wiring gap) DEPRIORITIZED: recon confirmed `min-count-to-merge` default = 100, so the divergence is
dormant for realistic single commits (narrow non-`first` ≥2-manifest sub-case only).

- [x] **PC #1 — RemoveDanglingDeleteFiles → ✅** — **DONE 2026-06-16 (#TBD).** Interop-only (the action +
      22 tests already existed; only interop was deferred). Converged 2 cycles (cycle-1 LOW: the position
      at-exact-min boundary was unit-pinned only → cycle-2 restructured `pk` to a TRUE at-exact-min position
      delete via data+pos-del in one `row_delta`, so BOTH off-by-one boundaries are now interop-pinned).
      Proves 3 anti-circular engine-agnostic claims (Java's actual action is Spark-surface, N/A): semantics
      match Java's independent `findDanglingDeletes`, API-contract counters/survivors, and CORRUPTION-SAFETY
      read-identity (MoR live-id set identical before↔after, both directions). DV-REMOVE is now a real e2e
      Puffin-DV fixture (closed the prior pure-fn-only gap). Live oracle GREEN (D1+D2+3 sabotages, exit 0);
      all 6 non-vacuity gates mutation-proven by the Critic. **GAP_MATRIX row 135 🟡→✅.** Two infra crashes
      first (529 overload) — hardened the workflow loop with null-guards (see [[parity-next-work]]). Files:
      `interop_remove_dangling.rs`, `run-interop-remove-dangling.sh`, `RemoveDanglingOracle`.
- [x] **PC #2 — ActionsProvider** — **DONE 2026-06-16 (#TBD).** Rust `ActionsProvider` trait (12
      snake_case methods mirroring Java `api/actions/ActionsProvider` 1.10.0, javap-confirmed) + a concrete
      `Actions` factory (`Actions::get`) overriding the 6 built actions (delete_orphan_files,
      rewrite_data_files, compute_table_stats, remove_dangling_delete_files via `X::new(table)`;
      expire_snapshots, rewrite_manifests via the transaction seam — required re-exporting the 2 seam types).
      Unbuilt actions return `Result<NoAction>` over an UNINHABITED empty enum (Ok arm statically
      unreachable ⇒ no stub can masquerade as real) → typed `FeatureUnsupported`. ORACLE-INDEPENDENT
      (factory has no byte-level round-trip; offline gate IS the verification). Converged 1 cycle; Critic
      javap-confirmed parity + mutation-tested the wiring (breaking a factory method fails 2 tests incl. a
      live MemoryCatalog execute smoke test). **GAP_MATRIX row 151 ❌→🟡** (partial). Underlying actions
      UNCHANGED. Files: `maintenance/actions_provider.rs` + 2 mod re-exports.
- [x] **PC #3 — DeleteReachableFiles** — **DONE 2026-06-16 (#TBD).** The DROP-TABLE-PURGE engine:
      `DeleteReachableFiles::new(metadata_location: &str)` (Java String arg shape) collects the FULL
      reachable set across ALL snapshots — categorized into the 6 javap-verified Java `Result` buckets
      (manifest lists, manifests, data, position-deletes [DVs fold here by content-type], equality-deletes,
      + current/all-previous metadata.json + version-hint + statistics + partition-statistics) — and deletes
      each via FileIO. Reuses the `DeleteOrphanFiles::collect_valid_files` walk shape WITHOUT changing it
      (separate categorizing collector). Interop-proven both directions against Java's ENGINE-AGNOSTIC
      `ReachableFileUtil` (clean non-circular oracle): Rust reachable set == Java's, + delete-completeness
      (no orphan-leak/under-delete, no data-loss/over-delete) + under-count sabotage. **Wired into
      ActionsProvider** (`delete_reachable_files` FeatureUnsupported→real; factory now 7 supported / 5
      unsupported). Converged 1 cycle; all 6 non-vacuity gates mutation-proven (each reachable category +
      all-snapshots + the deletion + the provider override). `DeleteOrphanFiles` + Cargo UNCHANGED.
      **GAP_MATRIX row 151 stays 🟡** (DeleteReachableFiles portion now ✅+interop). Files:
      `delete_reachable_files.rs`, `interop_delete_reachable.rs`, `run-interop-delete-reachable.sh`,
      `DeleteReachableOracle`.
## 8-HOUR PLAN (2026-06-16, Opus, signed off) — 4 sequential AC·OO PRs

Grounded by a 9-unit parallel scoping pass (each scoped vs the live code + 1.10.0 jars). Front-load the
three low-risk OFFLINE-gated wins (near-zero 529 exposure), then the one hard capstone. Each is
independent → its own PR; run strictly one-at-a-time (rebase on freshly-merged main between groups).
Expected outcome: 3 rows flip ✅ (144, 138, 151), ActionsProvider 7/5→8/4, eq→pos capability lands;
parity ~25→28 ✅.

- [x] **G1 — `ReplacePartitions.validateAppendOnly()`** — **DONE 2026-06-16 (#TBD).** **row 144 🟡→✅.**
      One bool + builder on `ReplacePartitionsAction`; guard = `!resolved.is_empty()` on the existing
      `resolve_partition_deletes` result (snapshot.rs:703). Critic javap-verified + CORRECTED the wrong
      residue: `conflictDetectionFilter` is NOT in Java 1.10.0 on DeleteFiles/ReplacePartitions (void),
      `validateAppendOnly` is ReplacePartitions-only — built ONLY that, no anti-parity surface. 4 unit
      tests + mutation-proven guard (disabling it fails the 2 reject tests). Converged 1 cycle, offline
      gate green. Files: `replace_partitions.rs`, `transaction/map.md`, GAP_MATRIX row 144.
      **CRITICAL: build ONLY validateAppendOnly** — javap proves `conflictDetectionFilter` on
      DeleteFiles/ReplacePartitions does NOT exist in Java 1.10.0; the row-144 residue list is factually
      wrong — ALSO correct the row text (the 2 void items + this 1 real port ⇒ row flips ✅).
- [x] **G2 — `ComputePartitionStats` action + `UpdatePartitionStatistics` commit seam** — **DONE
      2026-06-17 (#TBD).** **row 138 🟡→✅**; ActionsProvider `compute_partition_stats`
      FeatureUnsupported→real (**8/4**). New `transaction/update_partition_statistics.rs` seam (clone of
      `UpdateStatisticsAction` over `PartitionStatisticsFile`, emits Set/RemovePartitionStatistics +
      UuidMatch) + `maintenance/compute_partition_stats.rs` action (clone of `ComputeTableStats`). Snags
      resolved: register_partition_stats_file REWRITTEN to delegate through the new seam (ONE commit path,
      no duplicate); Ok(None)→typed DataInvalid; UuidMatch attached. Converged 1 cycle; Critic
      mutation-proved 3 wirings + confirmed the delegated commit is byte-identical to the proven path.
      Orchestrator RE-RAN `run-interop-partition-stats.sh` GREEN (both chains) to confirm the refactor
      preserved the Z3/R2/R3-proven bytes. Offline gate green (2314 lib). Files: `update_partition_statistics.rs`,
      `compute_partition_stats.rs`, + seam/mod/provider/partition_stats wiring.
- [x] **G3 — `SupportsNamespaces` partial property set/remove** — **DONE 2026-06-17 (#TBD).**
      SupportsNamespaces component ✅ (row 151 STAYS 🟡 until G4). 3 default `Catalog` methods:
      `update_namespace_properties` (overlap-reject DataInvalid → get → clone → remove → extend →
      full-replace `update_namespace`) + `set`/`remove_namespace_properties` wrappers (1:1 with Java's two
      public methods; Result<()> not bool). 6 memory tests; 2 mutations proven (drop-remove, drop-overlap).
      **SQL fallback (good judgment):** the "preferred 1-line SQL delete-absent-keys fix" proved UNSAFE —
      the SQL catalog uses an `exists=true` SENTINEL property row as its namespace-existence anchor, so
      deleting absent keys makes namespaces VANISH (broke 2 pre-existing SQL tests). Reverted; documented
      the divergence in-code + GAP_MATRIX; confined tests to the memory catalog (faithful full-replace).
      SQL diff is comment-only (behavior unchanged; 68 SQL tests green). Converged 1 cycle, no findings.
      Files: `catalog/mod.rs`, `catalog/memory/catalog.rs`, `catalog/sql/catalog.rs` (NOTE only), GAP_MATRIX row 151.
- [x] **G4 — `ConvertEqualityDeleteFiles`** — **DONE 2026-06-17 (#TBD). COMPLETES row 151 🟡→✅.** The
      capstone: NEW eq→pos write logic, 1:1 port of Java `api/actions/ConvertEqualityDeleteFiles` (free-standing,
      not a provider method). Per eq-delete: build the survival predicate → applicable LIVE data files
      (strictly-lower data-seq, same partition / global) → read with ABSOLUTE `_pos` → collect MATCHING
      positions → sort → write pos-deletes stamped with the eq-delete's data-seq → RewriteFiles 4-set replace.
      All FOUR corruption-stallers (absolute-pos, seq-stamp, applicability, matching-not-surviving)
      mutation-proven by the Critic (each breaks read-identity). 9 offline read-identity tests + no-Spark
      Java-MoR interop GREEN (live ids identical before-eq/after-pos). Converged 1 cycle. Read-path files
      touched VISIBILITY-ONLY (`parse_equality_deletes_record_batch_stream`/`try_cast_literal` → pub(crate);
      full 2329 lib suite green = no regression). Orchestrator re-ran the interop oracle (GREEN) + full lib.
      Files: `convert_equality_delete_files.rs` (+tests), `interop_convert_eq_delete.rs`, `run-interop-convert-eq-delete.sh`.

> **8-HOUR PLAN COMPLETE (2026-06-17).** All 4 sequential AC·OO PRs landed/pushed, each converged in 1
> cycle: G1 row 144 ✅ (#75), G2 row 138 ✅ + provider 8/4 (#76), G3 SupportsNamespaces (#77), G4 row 151
> ✅ (capstone). **3 rows flipped to ✅ (144, 138, 151)** + ConvertEqualityDeleteFiles capability; parity
> ~25✅→28✅; ActionsProvider 8/12. Near-zero 529 exposure (all offline-gated). Floor held + capstone landed.

## BLOCK 2 (8-HOUR PLAN, 2026-06-17, Opus, signed off) — 3 sequential AC·OO PRs

Grounded by an 8-unit parallel scoping pass (vs main #78 + 1.10.0 jars). Highest ✅-flip density yet: 4
rows. All med-risk with strong reuse/templating (no HIGH capstone). Each independent → own PR, run
one-at-a-time. Expected: rows 134/89/120/121 → ✅, ActionsProvider 9/12, parity ~28→32 ✅.
(Pruned by scoping: AggregateEvaluator's BoundExtract = frontier-parked variant-shredding → only 🟡;
SnapshotTable/MigrateTable need an external-table source → stay ❌; both deferred.)

- [x] **G1 — `RewritePositionDeleteFiles`** — **DONE 2026-06-17 (#TBD). row 134 ❌→✅; provider 9/3.**
      V2 parquet pos-delete compaction (V3 DV/Puffin OUT, documented), a strict subset of
      `convert_equality_delete_files`. NEW parquet-pos-delete reader by RESERVED FIELD ID (2147483546/2147483545).
      Seq-stamp = group MAX rewritten data-seq via `add_delete_file_with_sequence_number` — mutation-proven 3
      ways (max→min, explicit→inherit both caught by an exact on-disk-seq assertion). 10 offline read-identity
      tests + no-Spark Java interop GREEN (Java MoR identical {100,130,200,230} before 4 pos-deletes/after 2
      compacted; sabotage battery HARD-FAILs). Converged 1 cycle, NO findings; Critic ran all 8
      non-vacuity/staller mutations itself. `rewrite_position_deletes` flipped FeatureUnsupported→real (provider
      8/4→9/3, mandatory — it IS a Java provider method). Orchestrator re-ran interop GREEN + full lib (2340).
      Files: `rewrite_position_delete_files.rs`(+tests), `interop_rewrite_pos_deletes.rs`, `run-interop-rewrite-pos-deletes.sh`.
      _Superseded plan note:_ near-complete blueprint = `convert_equality_delete_files.rs`. Front-loaded
      (highest reuse → highest convergence confidence).
- [x] **G2 — `unknown` V3 primitive type** — **DONE 2026-06-17 (#TBD). row 89 ❌→✅.** `PrimitiveType::Unknown`
      arm (Java-faithful PRIMITIVE, not a top-level Type; serde "unknown" free) + the V3 `min_format_version`
      gate (mutation-proven: removing it reds 3 V2-reject tests) + 9 compiler-forced arms (arrow→Null,
      avro→null, datum/glue/hms/inspect/partition_stats reject-loud). DEFERRED-LOUD: data-file always-null
      I/O (FeatureUnsupported, no silent wrong bytes). **Legality doors matched Java 1.10.0 bytecode EXACTLY
      (Critic-confirmed) — NOT mirror-Variant: `identity(unknown)` is ACCEPTED as a partition source (Java
      `Identity.UNSUPPORTED_TYPES` excludes UNKNOWN), value-producing transforms reject, identifier accepts.**
      Atomic full-workspace compile (iceberg+glue+hms); 2351 lib + 2 interop + 15 glue + 15 hms green.
      Metadata-only interop GREEN both directions (Java writes V3 unknown schema → Rust reads+writes → Java
      verifies; caught+fixed a field-id-reindex bug). Committed Java fixtures under testdata/interop/unknown_type/.
      Converged 1 cycle, NO findings. Files: `datatypes.rs` + 8 arm sites + `interop_unknown.rs` + `run-interop-unknown.sh`.
- [x] **G3 — `IncrementalAppendScan` + `IncrementalChangelogScan` interop** — **DONE 2026-06-17 (#TBD).
      rows 120 + 121 🟡→✅** (TWO rows). Interop-only (scans built; scan/incremental.rs UNCHANGED). 4-snapshot
      fixture (S1-3 appends + S4 overwrite), compared by data-file BASENAME (anti-circular). Append: 3 ranges
      (excl {b,c} / incl {a,b,c} / to-current {c}) — the incl/excl boundary pinned (a.parquet the only diff).
      Changelog: data-file-level {+b,+c,−a,+d} vs Java IncrementalDataTableScan. Both proven D1+D2 vs Java's
      REAL scans. Off-by-one boundary sabotage fails closed; PRODUCTION-level non-vacuity (mutating the
      inclusive→parent resolution at incremental.rs:256 reds the D1 test). Row 121 ✅ for the DATA-FILE
      changelog with row-level/CDC + BatchScan residue NAMED (matches Java current; not over-claimed).
      Converged 1 cycle, NO findings. Files: `interop_incremental_scans.rs`, `run-interop-incremental-scans.sh`, `IncrementalScanOracle`.

> **BLOCK 2 COMPLETE (2026-06-17).** All 3 sequential AC·OO PRs landed/pushed, each converged in 1 cycle,
> ZERO findings across the block: G1 RewritePositionDeleteFiles (row 134 ✅, provider 9/3, #79), G2 unknown
> V3 type (row 89 ✅, #80), G3 incremental-scans interop (rows 120+121 ✅). **4 rows flipped ✅ (134, 89, 120,
> 121)**; parity ~28✅→32✅; ActionsProvider 9/12. Notable judgment: G2's legality doors matched Java bytecode
> (identity(unknown) accepted, NOT mirror-Variant). Next: pick a block-3 (stretch: ExpressionParser-JSON 147;
> Catalog-accessors offline; or the deferred BatchScan-U1 / RewriteTablePath / AggregateEvaluator partials).

## BLOCK 3 (8-HOUR PLAN, 2026-06-17, Opus, signed off — the SPINE) — 3 sequential AC·OO PRs

Grounded by an 8-unit parallel scoping pass (vs main #81). **The easy ✅ flips are spent** — block 3 trades
✅-density for capability-advancement: only ExpressionParser is a clean ❌→✅ in one unit; the rest are
❌→🟡 advances or matrix corrections. Each independent → own PR, run one-at-a-time. Expected: 1 ✅ (147) +
2 ❌→🟡 (148, 149) + matrix corrections; parity 32→33✅, ❌ 14→12. Front-loaded with the marquee ✅; only
G1 is oracle-dependent.

- [x] **G1 — `ExpressionParser` JSON (toJson/fromJson)** — **DONE 2026-06-17 (#TBD). row 147 ❌→✅** + retired
      the ScanReport `filter` divergence (row 123 annotated, stays 🟡). Canonical codec over `Predicate`
      (`expr/expression_parser.rs`): byte-exact wire shape + op hyphen-map + SingleValueParser value forms;
      schema-aware `from_json(_, &Schema)` recovers the typed Datum (the typed-vs-untyped staller — handled);
      transform/aggregate terms rejected; depth-limited read recursion; wired into ScanReport.filter via custom
      serde. Converged 2 cycles (cycle-1 MEDIUM = float/double byte-parity → cycle-2 ported Java
      `Float/Double.toString` formatting, byte-confirmed vs the jar). Live interop D1+D2 byte-exact over 34
      expressions (0 failures) + 4-sabotage battery fails closed; Critic ran 4 source mutations (op-map,
      date-codec, float-E, binary-hex). **NAMED RESIDUE (honest, documented in row 147 + pinned by a unit
      test):** JDK-11 `FloatingDecimal` non-minimal floats (~0.33% large-magnitude) — Rust emits the minimal
      form (== JDK 19+), diverging only from the JDK-11 oracle; non-finite floats rejected. 2 LOW findings
      ACCEPTED-as-is (write-side depth limit — input already bounded by the read-side cap; signed-zero
      round-trip test gap — write preserves it). Files: `expression_parser.rs`, `interop_expression.rs`, `run-interop-expression.sh`.
- [x] **G2 — `AggregateEvaluator` (count/min/max pushdown)** — **DONE 2026-06-17 (#TBD). row 148 ❌→🟡.**
      UnboundAggregate{count_star/count/min/max}→bind→BoundAggregate + AggregateEvaluator folding from manifest
      DataFile metrics, NO scan. Critic DECOMPILED Java 1.10.0 bytecode (AggregateEvaluator/NullSafeAggregator/
      CountStar/CountNonNull/Min/MaxAggregate) — formulas match EXACTLY: count(*)=Σrecord_count,
      count(col)=Σ(value_count−null_count) [corrected from the plan's record_count−null], min/max via typed
      `Datum::partial_cmp`, the has_value AND/OR predicates + allNull short-circuit. STALLER mutation-proven:
      dropping the latched `is_valid=false` invalidation fails 5 cant-push tests (missing metric ⇒ not-pushable,
      never a silently-wrong partial). Bound/UnboundExtract CUT (the aggregate term is `Option<Reference>` — no
      extract type to construct; zero `*Extract` defs). 17 unit tests; full lib 2392. Converged 1 cycle. 2 LOW
      accepted (min/max NaN-ordering + partial_cmp→None=DataInvalid conservative — part of the 🟡 residue,
      addressed at the later interop ✅). Offline (529-light). Files: `expr/visitors/aggregate_evaluator.rs`.
- [x] **G3 — Catalog accessors + the Glue/S3Tables-views matrix correction** — **DONE 2026-06-17 (#TBD).
      row 149 ❌→🟡.** Four non-breaking DEFAULT `Catalog` methods (name/properties/invalidate_table/
      invalidate_view), overridden per impl from held config (REST/Glue/HMS/S3Tables/SQL) + the MemoryCatalog
      retain-name+props fix; `commitTransaction(List)` split out (deferred). `properties()` honestly disclaimed
      as a Rust-convenience (not a Java Catalog-interface method). Matrix correction: rewrote rows 124(a)/125
      (Glue/S3Tables view-unsupported = parity-correct — S3Tables SDK-verified zero view ops; Glue via #12488
      + Rust no-override) + annotated row 126 SessionCatalog assessed-deferred (dead surface). Converged 1
      cycle; Critic javap-confirmed parity + matrix accuracy + ran accessor mutations. Offline gate green
      (iceberg 2399 + glue 18 + hms 15 + rest 55 + s3tables 23 + sql 71). 2 LOW doc-accuracy nits the Critic
      caught (#12488 is OPEN not closed; properties() also on SessionCatalog) — FIXED by orchestrator before
      commit. HMS accessor compile-only (socket-resolving new()). Files: `catalog/mod.rs` + 6 impl files.

> **BLOCK 3 COMPLETE (2026-06-17).** 3 sequential AC·OO PRs: G1 ExpressionParser JSON (row 147 ❌→✅ +
> ScanReport divergence retired, #82), G2 AggregateEvaluator (row 148 ❌→🟡, #83), G3 Catalog accessors +
> views matrix correction (row 149 ❌→🟡). **1 ✅ flip (147) + 2 ❌→🟡 (148, 149) + matrix corrections**
> (views-false-premise 124/125, SessionCatalog 126 deferred); parity ~32✅→33✅, ❌ 14→12. Lower ✅-density
> as forecast (easy flips spent). Notable: G1's JDK-11-non-minimal-float named residue (Rust matches JDK
> 19+); G2 matched Java's real count(col)=value−null via bytecode; G3 corrected a false-premise residue +
> its own 2 doc nits. Next block: BatchScan U1/U2 (scan completion), RewriteTablePath, or Avro-data-read.

## ROADMAP CHECK (2026-06-17, Opus) — audit + 1 integrity fill-in

A verify-driven workflow audited Roadmap/GAP_MATRIX/live-code alignment + adversarially mutation-tested
the recent ✅ flips. **Verdict: on track.** Matrix accurate (~33✅/24🟡/11❌, pipe-clean); ActionsProvider
genuinely 9/3 in code; **6 of 7 recent greens (134/89/147/120-121/151/138) held under hostile
mutation-testing.** One crack found + fixed:

- [x] **VAO — `ReplacePartitions.validateAppendOnly` interop** — **DONE 2026-06-17.** The skeptic refuted
      row 144's ✅: `validateAppendOnly` had flipped on unit tests ALONE (#75), no interop — and unlike our
      no-Spark-oracle cases it has a real engine-agnostic Java oracle (core-API, not Spark). Built the
      missing bidirectional interop (`ValidateAppendOnlyOracle` + `run-interop-validate-append-only.sh` +
      `interop_validate_append_only.rs`): 4-case behavior-equivalence battery, Rust REJECTS exactly where
      Java THROWS `DeleteException`, COMMITS exactly where Java commits; `javap -c` re-decode confirmed the
      Rust guard already matches Java — **NO Rust fix needed.** AC·OO converged 1 cycle, Critic refutation
      FAILED (guard-neuter reds 2 unit tests + the mirror). Row 144 ✅ now meets the unit-tests-AND-interop
      bar. _Orchestrator re-ran the oracle (D1 4/4, D2 1-pass, sabotage fail-closed) + offline gate._

- [x] **Doc-drift correction PR** — **DONE 2026-06-17 (#86).** Resynced 16 Roadmap under-claims (stale ❌
      for RewritePositionDeleteFiles/ComputePartitionStats/Catalog-accessors/validateAppendOnly/unknown;
      "5/8 actions"→9/12; incremental scans interop-deferred→✅) + 3 GAP_MATRIX nits (row 105 xref 140→150;
      row 145 xref 134→144; row 138 stale 8/4→9/3 dropped per one-home). Docs-only, no glyph changed. _VAO
      interop merged #85._

## BLOCK 4 (8-HOUR PLAN, 2026-06-17, Opus, signed off) — BatchScan: 2 sequential AC·OO PRs

Grounded by a 3-agent scoping pass (Java javap contract + live Rust scan module + matrix rows 122/146).
**Decisive finding:** in Java 1.10.0 `Table.newBatchScan()` is a thin `BatchScanAdapter` delegating
`planTasks()`/`planFiles()` 1:1 to `BaseTableScan` — so porting `planTasks()` IS porting BatchScan; rows
122 + 146 are ONE gap, 146 subsumes 122. `planTasks()` = `splitFiles(planFiles(), target)` →
bin-pack(`largestBinFirst=true`). Props: `read.split.target-size`(128MiB)/`planning-lookback`(10)/
`open-file-cost`(4MiB). DataFusion uses `to_arrow()` not tasks → no forced ripple. Statuses live ONLY in
[GAP_MATRIX](../docs/parity/GAP_MATRIX.md).

- [x] **U1 — `plan_tasks()` + planning structures + Java oracle → row 146 ❌→🟡** — **DONE 2026-06-17.**
      AC·OO converged 1 cycle, Critic refutation FAILED (mutation-tested largestBinFirst/weight-floor/offsets-split
      → reds the right tests). Landed `scan/task_group.rs` + `scan/bin_pack.rs` + `FileScanTask::split` + flagged
      `split_offsets` field (benign `split_offsets:None` ripple into arrow/* + rewrite_data_files test literals) +
      `TableScan::plan_tasks()` ABOVE an unchanged `plan_files()`. 30 offline tests + bidirectional `ScanPlanOracle`
      (D1 11 groups / D2 16 groups / sabotage 11→1 + 8→2). **HONEST FLIP: 146 ❌→🟡 not ✅** — 146 *subsumes* the
      typed `BatchScan` surface (row 122, still ❌ until U2), so the core planTasks/split/bin-pack is interop-proven
      but the row is not fully ✅ yet. Orchestrator fixed 1 LOW (stale off-by-one sabotage comment → large-target),
      re-ran the oracle + offline gate, verified the arrow ripple is field-default-only + Cargo untouched.
      `DataTask` = metadata-tables (separate surface, deferred). _Original plan said ❌→✅; corrected to ❌→🟡 for the
      subsumption-honesty reason above._
  _Delivered spec (reference):_ `ScanTaskGroup`/`CombinedScanTask` + `SplittableScanTask::split(target)` (offsets-aware: one
      sub-task per strictly-ascending split-offset, target ignored; else fixed-size `min(target,remaining)`;
      non-splittable→no split; sub-tasks clone deletes/residual/partition) + `BinPacking` port (largestBinFirst
      eviction; weight `max(len+deleteBytes, (1+#deletes)·openFileCost)`). `TableScan::plan_tasks()` sits ABOVE
      `plan_files()` (preserve its byte-unchanged/no-reporter invariant); builder knobs w/ Java defaults+override;
      thread `split_offsets` from manifest entry into `FileScanTask` (flagged additive public field). **Interop
      (real bidirectional, NOT no-Spark):** `ScanPlanOracle` drives `newScan().planTasks()` over a fixture
      exercising fixed-size+bin-pack (+offsets-aware +MoR-delete-weight); compare multiset of per-group
      `{(path,start,length)}` sets + group count, both directions; anti-circular target/lookback/cost; fail-closed
      sabotage (±1B target re-pack; drop split-offset). RISK: MoR (every sub-task keeps same path+pos deletes),
      offset fidelity, i64/u64 no-`as`, the plan_files invariant.
- [x] **U2 — typed `BatchScan` scan-kind → rows 122 ❌→✅ AND 146 🟡→✅** — **DONE 2026-06-17.** AC·OO
      converged 1 cycle, Critic refutation FAILED (mutation-tested `as_of_time` `<=` and the delegation →
      reds the right tests; re-decoded `BatchScanAdapter`/`SnapshotUtil.snapshotIdAsOfTime` via javap).
      `scan/batch.rs` (`Table::batch_scan()`) — thin `BatchScanAdapter`-shaped adapter delegating
      `plan_files`/`plan_tasks` to the U1 `TableScan` pipeline (REUSED, not forked) + `use_snapshot`/`use_ref`/
      `as_of_time` selectors (greatest `timestamp_ms <= ms`, first-wins conflict). 11 offline tests + 2
      mutation-baits. Oracle EXTENDED to drive `table.newBatchScan().planTasks()` == `newScan().planTasks()`
      (Java adapter delegation) == Rust, both directions (D1 11 / D2 16, 0 failures). Orchestrator fixed 1 LOW
      (matrix `core/`→`api/BatchScanAdapter`), left 1 LOW cosmetic (conflict-msg id not embedded for
      as_of_time/ref — behavior/kind/tests unaffected), re-ran the oracle + offline gate (125 scan + 2440 lib,
      U1 unregressed), Cargo/datafusion untouched.

> **BLOCK 4 COMPLETE (2026-06-17).** BatchScan in 2 sequential AC·OO PRs: U1 `plan_tasks()` split+bin-pack+oracle
> (146 ❌→🟡, #87) → U2 typed `BatchScan` surface (146 🟡→✅ + 122 ❌→✅). **2 ✅ flips (122, 146)**; both
> interop-proven (real bidirectional `planTasks()` group-shape oracle — NOT a no-Spark case). Census
> 32✅/26🟡/10❌ → **34✅/25🟡/9❌**. Honesty note: U1 deliberately flipped 146 to 🟡 (not ✅) because 146
> subsumes the still-❌ BatchScan surface; U2 closed both together. `DataTask` (metadata-tables) carved out as
> a separate surface (the capability exists via inspection tables). NEXT-BLOCK options: `RewriteTablePath`
> (137 ❌, provider 9→10/3→2) · Avro-data-READ (117 🟡, own ~6.5h) · the `SnapshotTable`/`MigrateTable` pair
> (137, need external sources).

Sequencing (done): U1 (146 ❌→🟡) → merge #87 → rebase → U2 (146 🟡→✅ + 122 ❌→✅). Both interop-proven.
Parity after block 4: **34✅, ❌ 9** (U1: ❌ 11→10 via 146→🟡; U2: 122 ❌→✅ + 146 🟡→✅).

## BLOCK 5 (RewriteTablePath, 2026-06-17, Opus, signed off) — 1 AC·OO PR

Grounded by a 3-agent scoping pass. **Key finding: `org.apache.iceberg.RewriteTablePathUtil` is engine-agnostic
iceberg-core (no Spark dep) — ~95% portable 1:1 with a REAL bidirectional oracle** (only version-diff walking +
CSV serialization + parallelism are the Spark "shell" the Rust port supplies). Statuses live ONLY in
[GAP_MATRIX](../docs/parity/GAP_MATRIX.md).

- [x] **RewriteTablePath (FULL-rewrite) → row 137 ❌→🟡, provider 9/3→10/2** — **DONE 2026-06-17.** AC·OO
      converged 1 cycle, Critic refutation FAILED (re-decoded the bytecode — partition_statistics passthrough at
      offset 142, location=replaceFirst; mutation-tested un-rewritten-path + copy-plan-direction-flip +
      partition-stats-symmetric-rewrite → each reds its test). `maintenance/rewrite_table_path.rs` +
      `_tests.rs` (15 offline) + provider 9/3→10/2 (lockstep) + bidirectional `RewriteTablePathOracle` (Java
      DRIVES real `RewriteTablePathUtil`; D2 graph=7/plan=7 == Java, D1 0 failures, direction-flip sabotage
      fails closed). Orchestrator re-ran the oracle + offline gate, verified format-stability (only path strings;
      add_existing_file preserves seq/snapshot ids), Cargo untouched. **3 LOW residues (all named, non-blocking):**
      pos-delete `col2` (optional `row`) dropped — fork's writer is (file_path,pos); `location` literal-vs-regex
      (identical for absolute path prefixes; `regex` is dev-only, no Cargo edit); Puffin-DV pos-delete →
      FeatureUnsupported. **DEFERRED:** incremental (startVersion/endVersion + version-diff + version-hint) + the
      Spark CSV file-list. **HONEST FLIP 137 ❌→🟡** (SnapshotTable/MigrateTable stay ❌ — external sources).

> **BLOCK 5 COMPLETE (2026-06-17).** RewriteTablePath in 1 AC·OO PR: FULL-rewrite port of core `RewriteTablePathUtil`
> + copy-plan + provider **9/3→10/2** + real bidirectional oracle. **Row 137 ❌→🟡** (1 of 3 bundled — SnapshotTable/
> MigrateTable need external sources, stay ❌). Census **34✅/26🟡/8❌**. NEXT-BLOCK options: Avro-data-READ (117 🟡,
> own ~6.5h) · the SnapshotTable/MigrateTable pair (137, need external Hive/fs source ingest — bigger sprint) ·
> incremental RewriteTablePath (additive follow-up). Easy ✅-flips long spent; remaining ❌ (8) are the big surfaces
> (geometry/geography 87, ORC 116, Avro 117, SessionCatalog 126 deferred, LockManager 127, encryption 128, events 142,
> SnapshotTable/MigrateTable 137-residue).

## BLOCK 6 (Avro data-file READ, 2026-06-18, Opus, signed off) — 2 sequential AC·OO PRs

Grounded by a 3-agent scoping pass. **Row 117 is ❌ on the live matrix** (the memory's "🟡" was WRONG — zero Avro
data code; `avro/` is manifest-schema only). Java data path = `PlannedDataReader` (id-based partner-visitor read
plan: id→pos in FILE order, null pos → skip, missing ids → constant/initial-default/IS_DELETED=false/ROW_POSITION/
optional=null/else error; read-time int→long & float→double promotion; logical types date=int-days/time+ts(tz)-micros=
long/ts(tz)-nanos=long/decimal=BE-two's-complement+scale/uuid=fixed16/fixed[L]/binary=bytes; optional=union[null,T]).
**Reuse:** `avro/schema.rs` already converts Avro↔Iceberg by field-id; `apache_avro::Reader` is a row Iterator<Value>
(NO Cargo edit); `RecordBatchTransformer` + delete-filter are format-agnostic. **Missing:** an Avro-Value→Arrow-array
builder + a `FileFormat` dispatch (today `arrow/reader.rs::process_file_scan_task` is Parquet-ONLY — an Avro file
fails as a corrupt-Parquet-footer error). Statuses live ONLY in [GAP_MATRIX](../docs/parity/GAP_MATRIX.md).

- [x] **U1 — Avro-`Value`→Arrow reader CORE + offline goldens (no matrix flip; engine only)** — **DONE 2026-06-18.**
      AC·OO converged 3 cycles. **Cycle-1 Critic caught a HIGH** (nested structs resolved POSITIONALLY not by
      field-id — would break on any reordered/projected/defaulted nested struct); Actor rebuilt the read-plan
      RECURSIVELY (by-field-id at EVERY struct level) + added 4 nested-by-id tests (subset/reorder/skip-extra+missing-
      optional/list-of-struct/map-of-struct-value); cycles 2-3 cleared a V3-row-lineage MEDIUM down to named residue.
      `arrow/avro_reader.rs` (`read_avro_data_file`/`read_avro_data_bytes`) + `_tests.rs` (29 offline goldens) +
      `arrow/mod.rs`. 6-rung missing-default priority + int→long/float→double promotion + decimal-BE-two's-complement +
      uuid-fixed16 + all logical types, bytecode-verified. Critic mutation-tested decimal-sign-ext/projection-skip/
      missing-default → each reds. Orchestrator re-ran offline gate (29 module + 2484 lib, fmt, clippy), confirmed NO
      Cargo edit + NO scan-path edit (pure module) + matrix UNTOUCHED. **Reconciled:** `read_data_file_stream` is
      Parquet-only → the Avro reader is a justified disjoint path, not a parallel reader. **3 LOW (named):** timestamp
      tz-by-expected-type (parity-correct, re-prove at U2 interop); positional list/map element (parity-faithful);
      Enum-as-string liberality. **DEFERRED to U2/later:** name-mapping fallback, V3 row-lineage present-field readers,
      variant read. **Row 117 STAYS ❌** (engine not scan-wired — U2 flips it).
  _Delivered spec (reference):_ `apache_avro::Reader` datum stream → field-id read-plan (projection/skip/missing-defaults via `avro_schema_to_schema`)
      → Arrow `RecordBatch`es (via `schema_to_arrow_schema` + array builders). ALL primitives + logical types + nested
      struct/list/map + null/union + int→long/float→double promotion. Offline tests: checked-in GOLDEN Avro files +
      hand-declared expected rows + mutation baits. Pure module, no scan dependency. **RECONCILE with
      `convert_equality_delete_files.rs` `read_data_file_stream`** (scope flagged it may already touch the data-read
      path — do NOT build a parallel reader). Row 117 STAYS ❌ (engine not scan-wired yet — honest, like BatchScan U1).
- [x] **U2 — scan read-path wiring + interop → row 117 ❌→🟡** — **DONE 2026-06-18.** AC·OO converged 1 cycle,
      Critic refutation FAILED (mutation-tested dispatch-break → 5/6 Avro tests RED as corrupt-Parquet-footer, and
      delete-disable → 3 MoR tests RED). `process_file_scan_task` dispatches on `task.data_file_format`: Parquet body
      byte-UNCHANGED (moved verbatim into `process_parquet_file_scan_task`), Avro → U1 `read_avro_data_file` + the SAME
      `RecordBatchTransformer` + MoR deletes POST-materialization (positional via new `DeleteVector::contains`, equality+
      residual via a survival mask), **ORC errors CLEANLY** (fixes the latent silent wrong-format read, row 116). **De-forked** the
      field-id predicate evaluator into shared `arrow/record_batch_predicate.rs`, now used by BOTH the Avro path AND
      `ConvertEqualityDeleteFiles` (one impl). 6 scan tests + mutation baits. **Interop ✅ D1** (`AvroDataOracle`: Java
      `GenericAppenderFactory.newDataWriter(FileFormat.AVRO)` writes a V2 table, MoR pos-delete removes id=20; Rust scan →
      row-identity over every primitive+logical+optional/null; sabotage fail-closed exit 101). Orchestrator FULLY
      re-verified after a Critic git-checkout incident (it reconstructed `reader.rs` mid-mutation): full compile + 6 avro
      scan + 29 U1 decode + **9 de-forked ConvertEq** + 29 delete_vector + Parquet 40 + datafusion-lib 80 + interop all
      GREEN; clippy/fmt clean; Cargo untouched; row 117 🟡 pipe-5 census 34✅/27🟡/7❌; fixed 2 stale "nested" comments.
      **NAMED residue (🟡): WRITE half** (no Avro DataFileWriter → D2 absent → later 🟡→✅) + name-mapping fallback + V3
      row-lineage + variant-read + nested-not-in-oracle (Java's own AVRO+PlannedDataReader can't round-trip nested; nested
      READ proven by U1 offline tests). **Pre-existing (disclosed, NOT introduced):** an iceberg-datafusion DOCTEST in the
      untouched `table_provider_factory.rs` fails on clean main (tokio rt-multi-thread feature); datafusion lib/integration
      tests pass.

> **BLOCK 6 COMPLETE (2026-06-18).** Avro data-file READ in 2 sequential AC·OO PRs: U1 reader core (no flip, #90) → U2
> scan wiring + interop (row 117 ❌→🟡). **1 flip (117 ❌→🟡)**; READ interop-proven D1 (Java writes Avro, Rust scans,
> row-identity + MoR delete). Census **34✅/27🟡/7❌**. WRITE half (Avro DataFileWriter + D2) is the residue for a later
> 🟡→✅ block. NEXT options: the Avro WRITE half (completes 117→✅) · ORC read (116, own block) · SnapshotTable/MigrateTable
> (137 residue, need external sources). Remaining ❌ (7): geometry/geography 87, ORC 116, SessionCatalog 126 (deferred),
> LockManager 127, encryption 128, events/listeners 142, SnapshotTable/MigrateTable 137-residue.

Sequencing (done): U1 (reader core + offline goldens, 117 stayed ❌, #90) → merge → rebase → U2 (scan wiring + interop,
117 ❌→🟡). Census 34✅/26🟡/8❌ → **34✅/27🟡/7❌**.

## BLOCK 7 (Avro data-file WRITE, 2026-06-18, Opus, signed off) — 2 sequential AC·OO PRs → completes row 117 🟡→✅

Mirrors the U1/U2 read cadence. Deep-scope workflow decoded the Java 1.10.0 Avro write contract (DataWriter /
ValueWriters / GenericWriters / AvroSchemaUtil / AvroMetrics) from ~/.m2 jars and mapped it onto the writer seam.
The honest flip REQUIRES a real production writer (the row-117 residue literally names "there is NO Avro DataFileWriter"),
so a test-only encode would be an overclaim — W1 builds the engine, W2 proves it with Direction-2 interop + flips.

- [x] **W1 — production `AvroWriter` (engine; no matrix flip)** — **DONE 2026-06-18.** AC·OO converged cycle 2 (cycle 1
      CONVERGED with 2 MEDIUM test-strength findings; cycle 2 closed them). NEW `writer/file_writer/avro_writer.rs`
      (`AvroWriterBuilder`/`AvroWriter`) implementing the `FileWriterBuilder`/`FileWriter` RPITIT seam, mirroring
      `parquet_writer.rs`, slotting into `DataFileWriter`/`RollingFileWriter` with a 2-line `mod.rs` seam only. Encoder
      reuses the proven `arrow_struct_to_literal` → `RawLiteral` → `schema_to_avro_schema` path (field-ids stamped).
      **Metrics = record_count + file_size ONLY** (Java `AvroMetrics.fromWriter` = `Metrics(rowCount,null,null,null,null)`;
      all six column maps EMPTY — positively asserted, the false-parity mutation-bait). Codec default null, deflate
      supported; variant/unknown rejected (reader-symmetric); empty-input deletes file + returns vec![]. 11 module tests
      (all-types + nested + nested-optional U1 round-trips, raw-byte union/decimal-width pin, metrics-empty, via-DataFileWriter,
      empty, variant-reject, deflate, live-size). Critic re-decoded Java per-type encoding byte-faithful + mutation-tested
      all 5 traps (union-flip / decimal-width / populated-metric / dropped-field-id / tz-shift — each bit a test) + re-ran
      gate. **Orchestrator-verified:** renamed placeholder `AvroWriter1`→`AvroWriter` (inner `apache_avro::Writer` aliased
      `OcfWriter`) + explicit re-export; re-ran gate MYSELF — lib **2501** passed/0, clippy 0 warnings, fmt + typos clean;
      Cargo/lock/GAP_MATRIX untouched; ASF header present. **3 LOW (named, none in W1 new code):** `arrow/value.rs:59`
      required-child-under-null-parent (pre-existing shared reader converter — carry to W2/future); `current_written_size`
      uncompressed underestimate (sound roll signal, disclosed); `decimal_required_bytes` log10 `-1` nuance (pre-existing
      shared spec fn, no divergence in tested precisions). NO matrix flip (engine only).
- [x] **W2 — Direction-2 interop (row 117 STAYS 🟡 — tz-fix parked, user decision)** — **DONE 2026-06-18.** AC·OO
      converged 1 cycle (Critic CONVERGED, 3 LOW). Rust GEN test (`tests/interop_avro_write.rs`) writes
      `00000-rust-data.avro` THROUGH the W1 production writer; Java (`AvroWriteOracle` + `verify-interop-avro-write` +
      `run-interop-avro-write.sh`) reads the RAW file via `Avro.read(...).project(schema).createReaderFunc(PlannedDataReader::create)`
      and asserts row-identity vs TWO anchors (Java's OWN constants AND the Rust JSON — anti-circular, .avro the only crossing
      artifact, Rust never reads its own write). Flat fixture, no delete. Fail-closed sabotage (corrupt one decimal → anchor (b)
      diverges, anchor (a) still passes; HARD-FAIL never SKIP, scratch-copy restored). **Orchestrator-verified:** re-ran the live
      oracle MYSELF (RC=0, Java reads all 5 rows = both anchors, sabotage diverged) + offline gate (GEN no-op, lib 2501,
      clippy 0, fmt, typos); diff scoped to 4 files; Cargo untouched. **The oracle SURFACED a real divergence** → brought to
      the user, who chose **keep 🟡 + PARK the fix**: Rust's shared `avro/schema.rs::schema_to_avro_schema` maps both
      `timestamp` and `timestamptz` to Avro `timestamp-micros` WITHOUT `adjust-to-utc=true`, so Java reads a Rust `timestamptz`
      as `LocalDateTime` not `OffsetDateTime` (INSTANT value identical, tz-attribute does not round-trip → not yet 1:1). Named
      as THE 🟡-blocker in row 117; emitting `adjust-to-utc` is a separate on-disk-format unit deliberately not pursued. Census
      UNCHANGED **34✅/27🟡/7❌** (117 was already 🟡 from U2). Note: the strict `| ✅ |` grep UNDERCOUNTS (~7 status cells
      carry extra content) — the robust column-2 tally is 34✅/27🟡/7❌.

> **BLOCK 7 COMPLETE (2026-06-18).** Avro data-file WRITE in 2 sequential AC·OO PRs: W1 production `AvroWriter` (engine, no
> flip, #92) → W2 Direction-2 interop (Rust-writes-Java-reads value-identity proven). **Row 117 STAYS 🟡** (user parked the
> `adjust-to-utc` tz-fix — value crosses identically but Java reads `timestamptz` as the wrong type `LocalDateTime`). The Avro read+write
> engine + BOTH-direction interop now exist; the only remaining 1:1 gap is the tz-attribute (+ exotic read-side deferrals).
> Census **34✅/27🟡/7❌**. NEXT (user: move to a fresh capability): ORC read (116, own block) · SnapshotTable/MigrateTable
> (137 residue, external sources) · the parked Avro tz-fix (on-disk-format unit, needs approval) if revisited.

## BLOCK 8 (Type utilities `TypeUtil` completion, 2026-06-18, Opus, signed off) — 1 AC·OO PR → row 143 🟡→✅

SnapshotTable/MigrateTable (137) was SCOPED then TABLED — the actions are Spark-only (api interfaces + empty core markers; only iceberg-spark has impls); the engine-agnostic core is `TableMigrationUtil` (directory→DataFiles+metrics) which needs a non-Iceberg source-catalog reader (Hive Metastore) this library lacks, NOT a query engine. Row 137 can't reach ✅ regardless (MigrateTable = destructive in-place source replace). User pivoted to an actionable green flip.

- [x] **`TypeUtil` completion → row 143 🟡→✅** — **DONE 2026-06-18.** Ported the missing engine-agnostic `TypeUtil` fns
      (all pure, no deps, interop N/A) into `spec/schema/`: **assign-ids** (`assign_fresh_ids` level-order, now the SHARED
      engine consumed by `transaction/update_schema.rs` — its private copy DELETED; `assign_fresh_ids_with_base` name-reuse
      overload; `assign_increasing_fresh_ids`; `assign_ids`), **reassign** (`reassign_ids`/`_or_refresh` align-by-name,
      matched-name structural type-mismatch THROWS per Java `Preconditions.checkArgument`, `reassign_doc`,
      `refresh_identifier_fields`), **check-compat** (`compat.rs`: `validate_write_schema`/`validate_schema`/
      `check_schema_compatibility` porting `CheckCompatibility`), **projection** (`project`/`select`/`select_not`/
      `get_projected_ids` — the project-vs-select subtree distinction pinned), **peripherals** (`join`, `estimate_size`,
      `index_quoted_name_by_id`). 40+ unit tests; `CustomOrderSchemaVisitor` named-residue (every consumer ported as explicit
      recursion). **PROCESS NOTE — did NOT converge in 3 AC·OO cycles; orchestrator remediated.** The cycle-3 Critic (which
      ran LIVE Java 1.10.0) correctly REFUSED to converge: (1) **mustFix** — `check-compat` wiring was INVERTED
      (`write/typeCompatibilityErrors` swapped `checkOrdering`↔`checkNullability`; Java's `CheckCompatibility(schema,
      checkOrdering, checkNullability)` ctor + `if(checkNullability) writeCompat(.., checkOrdering)` — write/type differ ONLY
      in nullability, the passed bool is ALWAYS checkOrdering), and 2 tests pinned behavior Java never produces; (2) **HIGH**
      — `estimate_size` map arm was `12+5·(k+v)`, Java is `12+5·(12+k+v)` (map<int,string>=362 not 302); (3) **LOW** — a
      `base_id_for` comment misdiagnosed a borrow workaround as an "optimizer miscompile". **Orchestrator fix (verified each
      against the 1.10.0 bytecode MYSELF):** corrected the compat wiring (helpers take `check_ordering`, hardcode
      `check_nullability` true/false; `check_schema_compatibility` branches on `check_nullability`; `validate_write_schema`
      forwards correctly), fixed the 2 wrong tests to the live-Java truth table (assertions intact), fixed the map formula +
      added a map=362 test, removed the misdiagnosis comment, corrected the matrix narrative (formula + wiring). Re-ran gate
      MYSELF: lib **2544** passed/0, clippy 0, fmt + typos clean; row 143 pipe-5; only row 143 changed; Cargo untouched;
      `compat.rs` tracked. Census **34✅/27🟡/7❌ → 35✅/26🟡/7❌**.

> **BLOCK 8 COMPLETE (2026-06-18).** Engine-agnostic `TypeUtil` completed (assign/reassign/check-compat/project + cheap
> peripherals), row 143 🟡→✅. Census **35✅/26🟡/7❌**. Did NOT converge in 3 cycles — the live-Java Critic caught a real
> inverted check-compat wiring + a wrong estimate_size map formula; orchestrator remediated (each fix re-verified vs 1.10.0
> bytecode), so the ✅ is honest. LESSON: the 3-cycle cap + an adversarial Critic that runs live Java is what catches a
> plausible-but-wrong port; never trust the Actor's "all green" self-report over the final Critic verdict.
>
> **ADDENDUM 2026-06-20 — DEMOTED ✅→🟡.** Post-merge audit confirmed the port is genuine (unit-tested + real consumer,
> no dead code) but it was flipped ✅ on unit-tests-only, missing the **bidirectional Java interop round-trip** the matrix
> legend demands of a fork-flipped green. Per the legend's strict bar (and a user decision 2026-06-20), the row is now 🟡
> with that interop named as the residue. The historical ✅ flip above is left as the record of what BLOCK 8 did; the
> authoritative live status is the GAP_MATRIX (`Type utilities` row). The interop round-trip is the remaining ✅ gate.

## BLOCK 9 (ORC data-file READ, 2026-06-18, Opus, signed off) — 2 sequential AC·OO PRs → row 116 ❌→🟡 (read-only)

Scoped first: orc-rust crate compat (0.8=arrow58 INCOMPATIBLE → pin **0.7.0 = arrow 57.3**, unifies with workspace 57.1, Apache-2.0)
+ iceberg-orc 1.10.0 fetchable (interop unblocked). **Make-or-break:** orc-rust 0.7 DISCARDS the ORC `iceberg.id` type
attributes (mod proto private, from_proto drops them) → Iceberg's by-field-id read isn't directly available. **User chose
FIELD-ID-CORRECT (parse the ORC footer ourselves)** over name-based (silent-misread on optional renamed cols) or a vendored
fork. User approved the orc-rust crate; the zlib-decode (deflate) dep was already present.

- [x] **U1 — field-id-correct ORC reader core (engine; no flip)** — **DONE 2026-06-18.** AC·OO converged cycle 2 (cycle 1
      CONVERGED + 4 findings, cycle 2 closed the MEDIUM/LOW). NEW `arrow/orc_reader.rs` + `orc_reader/footer.rs` (hand-rolled
      minimal ORC footer protobuf parser — PostScript→Footer.types[].attributes for `iceberg.id`; NONE + ZLIB=raw-deflate
      with the 3-byte ORC compression-chunk framing; other footer codecs → clear FeatureUnsupported, named residue) →
      field-id→ORC-index map → by-id projection mirroring Java `buildOrcProjection` (promotion int→long/float→double/
      decimal-precision-at-same-scale; missing required-no-default = hard error; optional = null-fill; reorder) → orc-rust
      decode (sync under spawn_blocking) → ORC→Iceberg Arrow mapping (ts **ns→µs** + tz None/UTC by resolved type, decimal
      scale, uuid/fixed→FixedSizeBinary, **LONG-vs-TIME**/BINARY-three-way by resolved Iceberg type), stamping
      PARQUET_FIELD_ID_META_KEY. API `read_orc_data_file`/`read_orc_data_bytes` (mirrors Avro U1). Committed a REAL
      Java-Iceberg-1.10.0 ORC golden fixture (`testdata/orc/iceberg_primitives.orc`, 1.7KB) → 21 tests incl. field-id-correct
      reads of every primitive+logical+null, projection/reorder, missing-optional/required, promotions, batch-size, nested→
      FeatureUnsupported. **Critic validated the footer parser against TWO real Java ORC files (committed + a scrambled-id
      one, no overfit) via a real Java org.apache.orc.Reader as ground truth; all 5 mutations bit.** **Orchestrator-verified:**
      re-ran gate MYSELF (lib **2565** passed/0, clippy 0, fmt+typos clean); Cargo edit = **orc-rust only** (the zlib-decode crate pre-existing),
      **arrow single-major 57 (no v58 leak)**; GAP_MATRIX + scan reader.rs untouched (U2 wires them); `.orc` fixture is
      license-eye-auto-skipped (binary, like the existing `.avro`/`.bin` fixtures). 2 LOW (named, unreachable on Iceberg ORC):
      CHAR/VARCHAR + ns-timestamp convert arms untested; int-narrowing + decimal-scale-strict are deliberate safe-direction
      guards (Java accepts then setScale-to-file-scale, a latent Java mismatch) — documented vs bytecode. NO matrix flip (engine).
- [x] **U2 — scan wiring + Direction-1 interop → row 116 ❌→🟡** — **DONE 2026-06-18.** AC·OO converged 1 cycle. Added
      `process_orc_file_scan_task` (verbatim clone of the Avro arm — materialize via U1 `read_orc_data_file`, SAME
      `RecordBatchTransformer` + `build_avro_expected_schema`/`avro_survival_mask` positional+equality delete machinery);
      replaced the `Orc` FeatureUnsupported arm. `OrcDataOracle` (Java `GenericAppenderFactory` FileFormat.ORC writes a real
      Apache ORC data file + parquet pos-delete, self-checks) + `run-interop-orc-data.sh` + `interop_orc_data.rs` + the pom
      `iceberg-orc` test dep. **FIELD-ID PROOF (load-bearing):** `orc_scan_resolves_by_field_id_not_name` reads a Java ORC file
      (col named `id`, field-id 1) with an expected schema renaming id-1 to `renamed_id` and asserts the values land there — a
      name-based reader FAILS it (Critic mutation-confirmed: name-resolution → "Missing required field: renamed_id"). Deletes
      mutation-confirmed (invert keep-mask → both delete tests RED). **Orchestrator-verified:** re-ran the LIVE oracle MYSELF
      (step-3 clean: 4 rows {10,30,40,50} id=20 deleted column-identical; step-4 sabotage diverged exit 101; RC=0) + offline
      gate (lib **2570**/0, clippy 0, fmt+typos clean, interop no-op); Cargo untouched; row 116 pipe-5; only row 116 changed;
      census **35✅/26🟡/7❌ → 35✅/27🟡/6❌**. **Critic git-checkout-clobbered the Actor's uncommitted reader.rs + reconstructed
      it** (same incident as Avro U2) — I re-verified the reconstruction MYSELF (full compile + all 6 ORC scan tests + live
      oracle green) before trusting it. Flip row 116 ❌→🟡 (WRITE absent = residue, like Avro).

> **BLOCK 9 COMPLETE (2026-06-18).** ORC data-file READ in 2 sequential AC·OO PRs: U1 field-id-correct reader core (footer-parse
> for `iceberg.id`, engine, #95) → U2 scan wiring + Direction-1 interop (row 116 ❌→🟡). **1 flip (116 ❌→🟡)**; field-id-correct
> READ interop-proven (Java writes real Apache ORC, Rust scans, row-identity + MoR delete + a renamed-column proof). Solved the
> orc-rust `iceberg.id`-discard blocker by parsing the ORC footer ourselves. Census **35✅/27🟡/6❌**. WRITE half (no ORC writer)
> + footer-codecs-beyond-ZLIB + nested/V3 = named residue keeping it 🟡. PROCESS: the Critic-git-checkout-clobber recurred (now
> TWICE: Avro U2 + ORC U2) — orchestrator MUST independently re-verify the reconstruction, never trust "tree is coherent now".
> NEXT (fresh capability): LockManager (127 ❌→✅, bounded util, no deps) · Events/listeners (142 ❌→✅, +emit wiring).

## BLOCK 10 (Events / listeners, 2026-06-19, Opus, signed off) — 1 AC·OO PR → row 142 ❌→✅

Scoped: Java events package = exactly 5 classes (`Listener<E>`, `Listeners` with `register`+`notifyAll` only — NO `notify`/no-arg;
keyed by EXACT event.getClass(), no try/catch in notifyAll), `ScanEvent`/`IncrementalScanEvent` (api) + `CreateSnapshotEvent`
(core). Honest ✅ requires the EMIT WIRING (registry alone = dead 🟡).

- [x] **Events/listeners → row 142 ❌→✅** — **DONE 2026-06-19.** NEW `events/mod.rs`: `Listener<E>: Send+Sync` trait + global
      `register<E>`/`notify_all<E>` over `LazyLock<RwLock<HashMap<TypeId, Vec<Box<dyn Any+Send+Sync>>>>>` (clone the matching
      Arcs + DROP the read guard BEFORE calling listeners → re-entrant-safe, no-lock-across-callback; reentrant test passes) +
      the 3 event structs (Java-exact fields). **Wired all 3 emit sites** (the real-✅ part): ScanEvent @ `scan/mod.rs`
      plan_files (after the snapshotless guard, UNBOUND filter default AlwaysTrue, table_name threaded from `identifier()`);
      CreateSnapshotEvent @ `transaction/mod.rs` do_commit AFTER `catalog.update_table` Ok (ONE per AddSnapshot; BEST-EFFORT
      via `catch_unwind` so a panicking listener never fails the commit; scan site propagates — matching Java's
      catch-on-commit / no-catch-on-scan); IncrementalScanEvent @ `scan/incremental.rs` plan_files from BOTH append (present→
      inclusive=false, absent→oldest-ancestor+inclusive=true) AND the changelog scan (shared helper). Emit-fire tests prove
      events GENUINELY fire on real scans/commits (empty scan fires none; property-only commit fires none). **AC·OO 3 cycles —
      cycle 2 INTRODUCED a contract bug** (injected `operation` into CreateSnapshotEvent.summary; Java's in-memory
      `snapshot.summary()` EXCLUDES operation — it's a SEPARATE field/accessor; `SnapshotSummary.Builder` builds only
      total-*/added-*/changed-partition-count); cycle-3 Critic caught it + reverted (summary = `additional_properties`,
      operation-free; test asserts `!contains_key("operation")`). **Orchestrator-verified:** re-ran gate MYSELF (lib **2584**/0,
      clippy 0, fmt+typos clean); confirmed the summary contract vs bytecode; Cargo untouched (std LazyLock, NO dep); only row
      142 changed (pipe-5); census **35✅/27🟡/6❌ → 36✅/27🟡/5❌**. **The Critic-git-checkout-clobber did NOT recur** — the
      hardening worked (Critic instructed to revert mutations via surgical inverse Edits, never git checkout). 3 LOW residue
      (named, accepted): table_name = `namespace.name` (TableIdent) vs Java catalog-qualified `table.name()`; best-effort swallow
      is SILENT (the iceberg crate does NOT declare `tracing` — it's a workspace dep only; metrics/mod.rs defers it identically;
      a `tracing::warn` would need a Cargo edit + approval); a multi_thread event test would need explicit registry-arming.

> **BLOCK 10 COMPLETE (2026-06-19).** Events/listeners ported 1:1 (global `Listeners` registry + `Listener` trait + 3 event
> types) AND wired to fire on real scans/commits/incremental+changelog scans — row 142 ❌→✅, the honest (not-dead-registry)
> flip. Census **36✅/27🟡/5❌** (down to 5 reds). LESSON: cycle 2 introduced a plausible-but-wrong contract (operation-in-summary)
> that cycle 3 caught — the adversarial Critic earns its cost even on a "simple" port. The mutation-revert-via-surgical-Edit
> hardening STOPPED the recurring git-checkout-clobber. NEXT remaining ❌ (5): geometry/geography 87 + encryption 128 (frontier-
> parked), SessionCatalog 126 (dead surface), SnapshotTable/MigrateTable 137 (Spark/HMS-coupled, tabled), LOW-backlog 152.
> LockManager (127) is now the last clean red→green (bounded util, available-but-unwired caveat). Many 🟡→✅ completions remain.

  _Delivered spec (reference):_ `maintenance/rewrite_table_path.rs`: `Table::rewrite_table_path().rewrite_location_prefix(src,
      tgt).staging_location(dir).execute(io)` → `Result{staging_location, copy_plan, latest_version}`, a STAGE-AND-PLAN
      action (rewrites the metadata graph into staging + emits a `(source,target)` copy-plan; does NOT copy data).
      Ports `RewriteTablePathUtil`:
      - **metadata** (`replace_paths`): `location` (Java uses regex `replaceFirst` — the ONE asymmetry vs `newPath`),
        snapshots' `manifest_list`, metadata-log, the 4 `write.*.path` props, `statistics` (Puffin). **Mirror the
        divergences: `partition_statistics` PASSED THROUGH un-rewritten in 1.10.0**; refs/schemas/specs/sort-orders verbatim.
      - **manifest-list/manifests**: rewrite each `manifest_path`, each entry `file_path` + `referenced_data_file`;
        re-emit via `add_existing_file` (preserve seq/snapshot ids — SEMANTIC round-trip, not byte-identical; thread
        format_version). Precondition: path not under `sourcePrefix` → typed error (no panic).
      - **pos-deletes** are the ONLY content-rewritten payload (rewrite col-0 file_path + `replacePathBounds`);
        **eq-deletes verbatim**. **copy-plan DIRECTION differs by class**: staged→target (manifests/lists/pos-deletes)
        vs source→target (data/eq-deletes).
      - **ActionsProvider**: override `rewrite_table_path`→real action, move UNSUPPORTED→SUPPORTED (9/3→10/2), update
        arrays + doc table + partition-12 test IN LOCKSTEP.
      - **Interop (real bidirectional, NOT no-Spark):** `RewriteTablePathOracle` drives core `RewriteTablePathUtil`;
        compare the rewritten path GRAPH + the copy-plan `(source,target)` set+count, both directions; anti-circular
        prefixes; fail-closed sabotage (un-rewritten path / dropped plan entry / wrong copy direction → red). Offline
        unit tests (prefix boundary, idempotence, no-double-rewrite, the 4 props, partition-stats pass-through,
        pos-delete content+bounds, referenced_data_file, copy-plan direction, precondition errors) + mutation baits.
      - **DEFER (named residue):** incremental `startVersion`/`endVersion` + version-diff + version-hint write
        (additive via the core overloads, not a redesign).
      - **HONEST FLIP: 137 ❌→🟡 not ✅** — 137 bundles `SnapshotTable`/`MigrateTable` which ingest an EXTERNAL
        Hive/filesystem source the Rust core has no path for → they stay ❌ (1 of 3 delivered). Census 34✅/25🟡/9❌ →
        **34✅/26🟡/8❌**.

Block-3 stretch / deferred: BatchScan-U1 (ScanTaskGroup/bin-pack, 146 🟡, offline) · RewriteTablePath
(137 🟡, provider 10/2, 4.5h — full TableMetadata rebuild) · Avro-data-READ (own ~6.5h block, 117 🟡).

Block-2 stretch (own PRs, if the spine beats estimates): `ExpressionParser` JSON (row 147 ✅ + retires the
ScanReport filter divergence; L/3.5h/MED/3cy — type-erasure schema-overload risk) · Catalog accessors
name/properties/invalidate* (❌→🟡; M/2h/**LOW/offline** — the parked "swap-in for lower 529 exposure"
option, to revisit). Deferred to a later block: BatchScan U1 (ScanTaskGroup/bin-pack) · RewriteTablePath
(137 🟡) · AggregateEvaluator (148 🟡, Extract parked).

Stretch / next (own PRs, if the front beats estimates): `RewritePositionDeleteFiles` (134 🟡, provider
9/3, MED) · `ExpressionParser` JSON (147 ✅ + retires the ScanReport filter divergence, oracle, MED) ·
`unknown` V3 type (89 ✅, oracle, MED) · Catalog accessors name/properties/invalidate* (❌→🟡, LOW).
Deferred (XL, split into 2 PRs later): `BatchScan` + `ScanTaskGroup`/`planTasks`.

Follow-on residue (surfaced mid-charter 2026-06-16, see GAP_MATRIX row 94):

- [ ] **Multi-spec MERGING-path: route `MergeManifestProcess` into the non-append actions (WIRING gap —
      RE-CHARACTERIZED 2026-06-16 post-review, Rust-side code-verified).** The earlier framing ("Rust's
      merging producer doesn't mirror Java's `first`-relative force-merge") was IMPRECISE and would have
      sent a future session chasing a phantom bug. `merge_append.rs::bin_disposition` ALREADY ports Java's
      `mergeGroup` force-merge of non-`first` spec bins faithfully (`bin_len==1` keep; `contains_first &&
      < min_count_to_merge` keep; else MERGE), with a passing multi-spec test. The REAL gap: the non-append
      merging actions (`RowDelta`/`OverwriteFiles`/`RewriteFiles`/`ReplacePartitions`/`DeleteFiles`) all
      commit through `DefaultManifestProcess` (a no-op), so they NEVER merge manifests, while Java's extend
      `MergingSnapshotProducer` (merge-capable past `min-count-to-merge`). Impact = manifest-list LAYOUT
      only (NOT data / seq / partition), DORMANT below threshold. Steps: (a) FIRST confirm via the Java
      reference checkout that `BaseRowDelta`/`BaseOverwriteFiles` actually merge multi-spec manifests at
      default settings (the asymmetry may be narrower than assumed); (b) if so, reuse the existing
      `MergeManifestProcess` in those actions; (c) THEN the multi-spec DATA cases (overwrite/rewrite
      carrying old-spec + adding new-spec) can be interop'd.

See [[parity-next-work]] (memory) for the reusable harness gotchas (register_table `<version>-<uuid>`
name; LocalTableOperations re-seed; final.metadata.json untouched).

## DONE (2026-06-15, Opus): OverwriteFiles conflict-validation interop (C1 — first conflict unit)

Goal: prove the FIRST slice of the rows 91-95 residue ("conflict-validation paths NOT covered").
Show `OverwriteFiles.validate_no_conflicting_data()` + `conflict_detection_filter(Predicate)` matches
Java `BaseOverwriteFiles.validate` → `validateAddedDataFiles` (`validateNoConflictingData`) on the
SAME concurrent-commit scenario — Rust rejects exactly when Java rejects, accepts exactly when Java
accepts, including the filter boundary. INTEROP-ONLY (no production change expected). Scope:
`dev/java-interop/src/main/java/org/apache/iceberg/InteropOracle.java` (new `OverwriteConflictOracle`),
`dev/java-interop/run-interop-overwrite-conflict.sh` (new), `crates/iceberg/tests/interop_overwrite_conflict.rs`
(new), committed fixtures, + GAP_MATRIX row 91 (annotate the conflict-validation evidence) and the two
map.md rows (dev/java-interop, crates/iceberg/tests). Java 1.10.0 oracle; mvn + JDK11 confirmed live.

- [ ] Read the EXISTING OverwriteFiles conflict unit tests (overwrite_files.rs) + the shared walk in
      snapshot.rs (`added_data_files_after` / `validate_no_conflicting_added_data_files` /
      `first_conflicting_file`) — the template for the Rust side. Re-confirm Java 1.10.0 semantics
      (`BaseOverwriteFiles.validate` → `validateAddedDataFiles`) against the jar before coding.
- [ ] Scenario matrix (≥3): (REJECT) concurrent add MATCHES `conflict_detection_filter`;
      (ACCEPT) concurrent add does NOT match the filter; (ACCEPT) no concurrent add. Each: base S0 →
      build overwrite capturing S0 → concurrent S1 add → commit overwrite → record ACCEPT|REJECT.
- [ ] Java `OverwriteConflictOracle`: `generate` (build history + emit expected-outcome JSON) +
      `verify` (read the Rust-built history, run the same overwrite, assert SAME outcome, emit the
      `verify-…: N failures` sentinel). Mirror `WriteActionsOracle` structure.
- [ ] Rust `interop_overwrite_conflict.rs`: GEN test (build the concurrent history via the catalog +
      attempt the validated overwrite; assert a REJECT is non-retryable `DataInvalid` + `!retryable()`)
      and comparison test (Rust validates the Java-built history → outcome == Java's expected JSON).
      Env-var gated (unset ⇒ clean no-op; offline `cargo test` gate stays green).
- [ ] Shell `run-interop-overwrite-conflict.sh`: reset → Java generate → Rust GEN → Java verify (D2)
      → Rust compare (D1) → 2nd pass → SABOTAGE (flip a REJECT scenario's filter so the conflict
      vanishes ⇒ verify must FAIL closed; HARD-FAIL, never SKIP, if the mutation cannot apply).
- [ ] Run the LIVE oracle end-to-end (mvn) + commit the generated fixtures. Gate in ONE `&&` chain to
      the commit: `typos . && cargo fmt --all --check && cargo clippy --all-targets -p iceberg --
      -D warnings && cargo test -p iceberg --test interop_overwrite_conflict`.
- [ ] Annotate GAP_MATRIX row 91 with the conflict-validation interop evidence (cell edit + link
      ONLY); update the two map.md rows. Done-bar: ✅ for the OverwriteFiles conflict-validation
      SLICE (unit + interop); row 91 stays 🟡 until its multi-spec + row-filter residue also closes;
      rows 92-95 conflict-validation are follow-on units (queue item 1).

> **Done (2026-06-15):** all steps landed. New files: `crates/iceberg/tests/interop_overwrite_conflict.rs`
> (GEN + D1) and `dev/java-interop/run-interop-overwrite-conflict.sh`; new `OverwriteConflictOracle`
> (+ 2 dispatch cases) in `InteropOracle.java`. Live harness GREEN end-to-end: Java-gen → Rust-gen →
> D2 (Java validates Rust, `0 failures`) → D1 (Rust validates Java, register_table) → sabotage battery
> (semantic-swap + truncate, both fail-closed; control-gated). Both directions agree on all 3
> scenarios. Offline gate green (typos / fmt / clippy / `cargo test --test interop_overwrite_conflict`
> = clean no-op skip). GAP_MATRIX row 91 annotated (stays 🟡 — multi-spec + row-filter conflict
> interop still open); both map.md rows added. Next (queue item 1): replicate to DeleteFiles / RowDelta
> / ReplacePartitions / RewriteFiles conflict interop.

## LANDED (2026-06-13) — status in GAP_MATRIX rows 95/140; clears next archival pass: RewriteFiles DELETE-file ADD surface

Goal: port the unported DELETE-file ADD surface on `RewriteFiles` — `addFile(DeleteFile)` /
`addFile(DeleteFile, long)` (explicit-seq overload) + the 4-set
`rewriteFiles(data_to_replace, delete_to_replace, data_to_add, delete_to_add)` — and lift the third
precondition (`addsDeleteFiles() ⇒ deletesDeleteFiles()`) into reachability. Java spec from 1.10.0
bytecode (`BaseRewriteFiles`, `MergingSnapshotProducer.add(DeleteFile)/(DeleteFile,long)`,
`Delegates.PendingDeleteFile`, `SnapshotProducer.writeDeleteFileGroup`). Files: `rewrite_files.rs`,
`snapshot.rs` ONLY. Done-bar 🟡 (unit-tested, interop deferred).

- [ ] **snapshot.rs** — model the `PendingDeleteFile` per-file optional explicit data-seq: store added
      delete files as `Vec<(DataFile, Option<i64>)>` (None = inherit). Keep
      `with_added_delete_files(Vec<DataFile>)` mapping each to `(file, None)` (RowDelta unchanged); add
      `with_added_delete_files_with_seq(...)`. Stamp the explicit seq in `write_added_delete_manifests`
      (mirror `writeDeleteFileGroup`: `add(file, seq)` if Some, else `add(file)` = inherit). Update the
      validation/empty-check/summary read sites to destructure the pair.
- [ ] **rewrite_files.rs** — `add_delete_file(DeleteFile)` / `add_delete_files(...)` (inherited seq),
      `add_delete_file_with_sequence_number(DeleteFile, i64)` (Java `addFile(DeleteFile, long)`),
      `rewrite_files_with_deletes(4 sets)` (Java 4-arg). Make precondition (3) reachable
      (`adds_delete_files = !added_delete_files.is_empty()`). Content-guard + negative-seq guard on
      added delete files. Route added deletes through `with_added_delete_files_with_seq`.
- [ ] **Tests** (rewrite_files.rs): crown-jewel rewrite a delete file into a NEW delete file + post-commit
      MoR scan (no resurrection); explicit-seq overload stamps the given seq (on-disk pre-inheritance via
      the manifest reader); 4-arg atomic (data AND delete sets in ONE Replace snapshot); precondition (3)
      both-directions; content + negative-seq guards. Mutation: seq-strip → resurrection test fails.
- [ ] **Gate**: `typos . && cargo fmt --check && cargo clippy -D warnings && cargo test -p iceberg --lib`
      (twice). Update `transaction/map.md` rewrite_files row + the third-precondition note.

## LANDED (2026-06-13) — status in GAP_MATRIX row 134; clears next archival pass: `caseSensitive(bool)` on the expression-binding write actions

Add `case_sensitive(bool)` (DEFAULT TRUE = Java default, 1.10.0 bytecode-confirmed:
`MergingSnapshotProducer` ctor `iconst_1; putfield caseSensitive`; `ManifestFilterManager` ctor same)
to `DeleteFiles` / `OverwriteFiles` / `RowDelta` and thread through the shared snapshot.rs binding
sites. Scope: `delete_files.rs`, `overwrite_files.rs`, `row_delta.rs`, `replace_partitions.rs`,
`snapshot.rs` ONLY. Java refs: `api/{DeleteFiles,OverwriteFiles,RowDelta}.caseSensitive(boolean)`
present; `api/ReplacePartitions` has NO `caseSensitive` (javap-confirmed) — narrow it out.

- [x] snapshot.rs: threaded `case_sensitive` into `resolve_filter_deletes` (+`build_residual_evaluator`
      →`ResidualEvaluator::of`) and `validate_no_new_deletes_for_data_files`. The `eval(..., true)`
      (include_empty_files) calls left untouched. Forced 1-token out-of-scope edit: `rewrite_files.rs`
      passes `true` (inert — its conflict filter is `None`; documented).
- [x] delete_files.rs: `case_sensitive: bool` field (default true) + `case_sensitive(bool)` builder;
      threaded via `DeleteFilesOperation`. Deferred doc comment rewritten.
- [x] overwrite_files.rs: field + builder; threaded into `resolve_filter_deletes`, the 4 conflict
      helpers, and the StrictMetricsEvaluator row-filter bind in
      `check_added_files_match_overwrite_filter`. Java-faithful: partition-projection binds stay `true`
      (Java uses the single-arg `Projections`/two-arg `Evaluator`; only the StrictMetricsEvaluator takes
      `isCaseSensitive()` — bytecode-verified).
- [x] row_delta.rs: field + builder; threaded into the conflict helpers, `validate_added_dvs`, and
      `validate_no_new_deletes_for_data_files`. `validate_fresh_dvs_only` left (by-path/partition).
- [x] replace_partitions.rs: NARROWED — `javap -p` confirms no `caseSensitive` in the Java public API +
      validate path is partition-set-based (no `Predicate::bind`). Documented, no builder added.
- [x] Tests: 9 total (3/action). Mutation-verified BOTH directions at BOTH shared sites, failing all 3
      actions' tests simultaneously (ignore-flag ⇒ false-direction tests fail; hard-code-false ⇒
      boundary tests fail).

> **Done (2026-06-13):** `case_sensitive(bool)` landed on DeleteFiles/OverwriteFiles/RowDelta (DEFAULT
> TRUE), narrowed out of ReplacePartitions per Java 1.10.0 API. Gate green (typos/fmt/clippy + 2× lib
> @ 2258). Interop deferred → row 134 stays 🟡. GAP_MATRIX rows 134/135 updated.

## LANDED (2026-06-13) — status in GAP_MATRIX row 135; clears next archival pass: `DeleteFiles.deleteFromRowFilter(Expression)` delete-by-predicate

Close the deferral in `delete_files.rs` L30-32. Java bytecode-confirmed (`javap -p -c` on
iceberg-api/core 1.10.0): `StreamingDelete.deleteFromRowFilter(Expression)` → `MergingSnapshotProducer.deleteByRowFilter`
→ the SAME `ManifestFilterManager.manifestHasDeletedFiles` path `OverwriteFiles.overwriteByRowFilter`
already ports via `SnapshotProducer::resolve_filter_deletes`. `StreamingDelete.operation()` is the
CONSTANT `"delete"` (NOT dynamic). PARTIAL ⇒ "Cannot delete file where some, but not all, rows match
filter %s: %s" (verbatim string in the 1.10.0 jar). Scope: `crates/iceberg/src/transaction/delete_files.rs` ONLY.

- [x] Add `delete_from_row_filter(Predicate)` builder method (stores `Option<Predicate>` row filter).
- [x] Thread the row filter into `DeleteFilesOperation`; its `delete_files` unions `resolve_delete_paths`
      with `resolve_filter_deletes(row_filter)` (de-dupe by path) — mirroring `OverwriteFilesOperation`.
      `operation()` stays `Operation::Delete` (StreamingDelete constant).
- [x] Tests: A strictly-covered (deleted), B provably-cannot-match (kept), C partial (ERROR, nothing
      committed); residual KEEP/DELETE/PARTIAL pins; negative residual-non-match; combine-with-by-path.
- [x] Update `delete_files.rs` module doc (remove the deferral note) + the map.md row 39.
- Done-bar: 🟡 (unit-tested; interop deferred — flagged for the critic). `caseSensitive(bool)` is a
  SEPARATE GAP_MATRIX row — explicitly OUT of this increment (filter bound case-sensitive `true`, the
  Java default, as the precedent does).
- Outcome (2026-06-13): landed in `delete_files.rs` (`delete_from_row_filter` builder + `row_filter`
  field threaded into `DeleteFilesOperation`, unioned with by-path via the SHARED
  `resolve_filter_deletes` — no fork). 8 new tests; full gate green (typos/fmt/clippy + lib ×2 =
  2246 passed). Two mutations verified-then-reverted (residual→full-predicate caught by 3 tests incl.
  the dedicated partition-residual pin; strict→inclusive over-broaden caught by the crown-jewel partial
  test). DEFERRED for the reviewer/orchestrator: flip the GAP_MATRIX `deleteFromRowFilter` row ❌→🟡
  (outside the explicit modify-list), and data-level Java↔Rust interop.

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
- [todo-archive/2026-06_wave5.md](todo-archive/2026-06_wave5.md) — Wave 5 (PRs #39–#41; pass-scoped).
- [todo-archive/2026-06_wave6-wave7.md](todo-archive/2026-06_wave6-wave7.md) — Waves 6–7 (PRs #43–#47; pass-scoped): the I1/I2/I3 interop increments + O1/O2/O3 + R1/R2/R3.
- Index: [todo-archive/map.md](todo-archive/map.md).
