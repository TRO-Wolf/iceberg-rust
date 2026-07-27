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

# Todo archive — audit / hardening / engine-trust era (archival pass 6)

Increment plans/outcomes moved verbatim by todo-archival pass 6 (2026-07-26; run by the RePark
workstream under the hub concurrency-protocol claim of the same date). Covers 2026-07-01 →
2026-07-26. Genuinely-open boxes (B/C/E of the 07-17 overnight list) are ALSO lifted to the live
file's Carried-forward section — the sanctioned overlap; every other unticked box here was
stale-audited at pass time (superseded or shipped under later increments — e.g. the POST-D1
WG2–WG4 boxes merged via #170/#171, overnight unit D via #159, the BACK-TO-GOAL Unit-3/4/5 boxes
re-homed in the live POST-BUNDLE QUEUE). Not session-start reading.

---

## POST-D1 BLOCK (2026-07-24) — 4 parallel work groups, AC·OO per group

Follow-on from the scan-plan D1 root-cause + merge fix (fork #169, `c88888c3`, row R148). Four
independent groups, one branch each; this tracker rides ONLY the WG1 branch (each group edits its
own files, so a shared tracker would collide).

- [x] **WG1 — `fix/g3-scan-plan-merge-interop-pin`** (row R148): interop-pin the adjacent-split
  merge that #169 landed with offline unit tests only. Spec: [g3-scan-plan-merge-interop-brief.md](g3-scan-plan-merge-interop-brief.md).
  Two dedicated fixture files (`merge.parquet` co-binned contiguous pair ⇒ ONE spanning member;
  `gap.parquet` co-binned NON-contiguous pair ⇒ stays two), each planned under an isolating
  metrics-prunable filter at the delete-free append snapshot; both directions; non-vacuity proven
  on both sides by the exact plan-SHAPE asserts (the span inequalities are degenerate-fixture
  guards); new Java sabotage leg + TWO production-source mutation legs (merge-removal and
  adjacency-removal, each RED on a real assertion signal, restored md5-identical). Chain [1/7]…[7/7]. ZERO production-code changes.
  _What changed and why:_ the oracle could always REPRESENT a merge divergence but its fixture never
  reached the branch — measured: with `merge_tasks` removed the old unfiltered comparison still
  passes at 14 groups. See [lessons.md](lessons.md) 2026-07-24 (coverage-vs-granularity).
- [ ] **WG2 — `fix/df-provider-live-schema`** (BUG-005 / BUG-011): the DataFusion `TableProvider`
  freezes the Iceberg schema at construction, so a post-construction schema evolution is invisible
  to planning.
- [ ] **WG3 — `fix/safety-joinhandle-channel`** (SAF-006 / SAF-007): `JoinHandle` / channel
  panic-and-drop hardening.
- [ ] **WG4 — `fix/rest-secret-debug-ssrf`** (SEC-010 / SEC-011): REST catalog secret-in-`Debug`
  exposure + SSRF surface.

## BACK-TO-GOAL BLOCK (2026-07-25, signed off) — consumer fork-queue remediation, 3+2 units

Spec (single home, supersedes the triage scratchpad):
[back-to-goal-2026-07-25-brief.md](back-to-goal-2026-07-25-brief.md). Decisions D1-D6 recorded
there (WG1 = children-bearing PhysicalExpr; WG3 non-breaking first; WG2 + remediation tooling;
WG1 solo Mode A then Mode B bundle; bundle-first satisfied = #170/#171 merged, content-verified;
R161 URL-escaping APPROVED). RePark work order delivered:
`~/Desktop/repark-work-order-2026-07-25.md`. Goal frame: gate 1 data-trust → gate 2 engine-trust
→ gate 3 scale.

- [x] **Unit 1 — WG1 honest-children `PartitionExpr` (S1, Mode A solo PR, AC·OO)** — branch
      `fix/partition-expr-honest-children` off `ce2affc9`. FORK-O7+O8 are ONE defect (optimizer
      re-parents the children-less expr; positional read hits the wrong batch). Fix shape D1 +
      core-seam guard + T1-T10/M1-M5 per brief. Claim OPENED on the PrimarySync board
      2026-07-25. Close-out: RoadMapSync announce → RePark repins + flips its 2 detector pins.
      **ACTOR DONE 2026-07-25** (two code commits on the branch: the fix+tests commit and the
      `93e31b2b` interop null-tuple leg — Java read the NULL tuple back, full chain green; RED
      evidence + mutation matrix in the PR body at the scratchpad
      `pr-bodies/WG1-partition-expr-honest-children.md`).
      **CONVERGED 2026-07-25, cycle 1 — DUAL independent Fable-max Critics, zero S1/S2.**
      Critic 1 (adversarial): re-derived the DataFusion 52.2 mechanism from vendored sources,
      re-ran the full gate (lib 2865, datafusion 218) + all 5 mutations + 6 novel probes
      (filter/limit-between-projections, multi-partition-field, 3-column rotation — the only
      M2/M3-killing probe, 0-row source, nested CASE-in-CAST) + the full interop chain
      independently. Critic 2 (contract/honesty): re-derived the ORIGINAL RED at true base
      `ce2affc9` in a temp worktree (verbatim failure messages), format-stability spot-check
      (correct-input tuples byte-identical at base and tip), pub-surface grep (exactly one
      added pub item), caller-compat sweep. 3 S3 total: PR-body gate-number refresh (applied
      at close-out), duplicate Column-list style + second sabotage leg (recorded as residue).
      PUSHED for PR — merge is the user's; RoadMapSync announce + RePark repin follow merge.
      **MERGED #172 (`a6199ca5`) 2026-07-25, content-verified R4; RoadMapSync repin+pin-flip
      announcement posted; claim released. FORK-O7/O8 CLOSED.**
- [x] **Unit 2 — engine-trust bundle (Mode B, one branch, final bundle Critic) — STARTED
      2026-07-25, MERGED 2026-07-26**, branch `fix/engine-trust-bundle-2026-07` off
      `a6199ca5`; user-directed
      Opus-max AC per group + Fable-max closing bundle Critic. **G0 T10 nullability-widening
      residue (+ ADV-1/ADV-2 riders; spec added to the brief)** → G1 WG2 detector+remediation → G2
      WG4a stamp probe/fix + WG4c contract normativity (R113) → G3 WG3 L2+L3 non-breaking →
      G4 R161 escaping (format-stability attestation) → G5 WG4b path-keyed pos-delete routing
      (+ remove_dangling same-unit; R117 demote→re-promote) → G6 WG5 null-bit family → G7 WG6
      mechanical → G8 delete_filter lost-wakeup (3 sites + `PosDelState::Failed`).
      **BUNDLE CONVERGED 2026-07-26** (36 commits, 16h run): 8 groups converged in-ladder (G2
      cycle 1, G8 cycle 2, rest cycle 0); **G4 parked-not-converged on matrix-cell wording, its
      reset was classifier-blocked, and the Fable-max closing Critic caught it as S1** → CLOSE
      remediation corrected the R161 residue ledger (truncate FIXED/BINARY hex-vs-base64
      divergence named from an independent Java re-decode), ratcheted the interop floor to 52,
      and the re-attestation ruled the WHOLE bundle CONVERGED zero S1/S2 — with a live 22/22
      byte-match vs Java's own partitionToPath. Gate at tip: lib 2974 (+109), datafusion 246
      (+28), anchors 75 rows. ONE user merge-gate rides the PR (closer CR-1): **waive or strip
      G4** (strip = surgical: 5 of its 10 files overlap later groups, 7 of 10 by branch tip incl. the closing chain). PUSHED for PR.
      **MERGED #173 (`4063007b`) 2026-07-26, G4 WAIVED; R161 escaping landed** — the user took
      the waive branch of the closer's CR-1 merge gate at SQM (the R13-remand substance had
      already been independently re-attested by the closing chain), so G4's escaping fix ships
      with the bundle. Disposition appended to the incident section in
      [sepmo-metrics.md](sepmo-metrics.md); the canon gap it exposed is closed as v2.3 (below).
      **TRIAGE RIDER 2026-07-26 (docs/evidence only, NO behavior change — the bundle needed none):**
      the RePark consumer's response to the 5-action fork work order was adjudicated and absorbed.
      (a) EXPOSURE AUDIT — CLOSED as an audit, not a probability: they delivered an inventory
      (real-AWS surfaces write exclusively via CTAS + MERGE INTO, never the provider; targets
      unpartitioned; scratch Glue namespace disposable; provider `insert_into` consumers are
      ephemeral test fixtures), so "none exist" is answered. The `RoadMapSync.md` 2026-07-25 row
      still reads "the exposure-audit ask stays OPEN" and needs updating; one narrow follow-up
      question remains (their inventory is scoped to `insert_into` — confirm no OTHER RePark
      `PhysicalExpr` reads its input positionally while declaring no children, which is the general
      defect class). (b) WG4b FIXTURE — DELIVERED, and now WIRED: `interop_spark_mor_fixtures.rs`
      (env-gated, fixture NOT committed here). (c) IDENTITY-SPEC CORRECTION — ABSORBED with the
      mechanism RULED: their observation is right, their inference is not. The computed values ARE
      in the parquet (the write path drops no columns); the READ path substitutes the manifest
      tuple over the file's own column for identity transforms only, so the values are MASKED, not
      lost, and `RepairPartitionKeys` recovers them. The detector's severity model therefore needs
      no change; the recipe + contract wording landed in this rider. Honest boundaries sent back:
      history is not healed, and a pure manifest-tuple rewrite is NOT always sufficient (that half
      of their claim stands). (d) PIN PROTOCOL — their Group-AA guard makes their divergence MATRIX
      a RePark-with-guard measurement until repin; their bypass pins stay fork-direct, and the
      removal checklist must revert BOTH the bypass and the cell relabel in the same change. Open
      asks back to them: reconcile the pin arithmetic (the doc describes four new pins, the repin
      posture says two), rename the identity pin away from "data-loss", and note the FROM-less
      literal pin's post-repin expectation is schema-dependent (Unit 1 + required columns = Ok;
      Unit 1 + optional = still a loud `Plan(...)` rejection until this bundle's G0).
- [ ] **Unit 3 — breaking follow-up (Mode A, after Unit 2)** — `PartitionKey::new -> Result`
      (58 sites/34 files incl. 6 `no_run` doc fences) + `CurrentFileStatus` unwraps.
- [ ] **Unit 4 — H7-S2 COW streaming** · **Unit 5 — H7-P1 pushdown** (footgun precondition) —
      re-scope at signing.
- Watch: nightly-interop dispatch on main (#169/#170 live proof); branch pruning (classifier-
      blocked this session); tail backlog in the brief.

## ACTIVE UNIT (2026-07-26): SEPMO canon v2.3 re-bind — branch `infra/sepmo-v2-3-rebind`

Closes the canon gap the G4 parking incident exposed (Unit 2 above). CCR
[sepmo-ccr-2026-07-26-g4-incident.md](sepmo-ccr-2026-07-26-g4-incident.md) **RATIFIED by the
user 2026-07-26**. Docs-only: no crate code, no matrix row. STANDARD path (governance surface,
multi-file, >150 lines — LIGHT criteria 1/3 fail).

- [x] **1. Master home raised to v2.3** — `~/Desktop/Sepmo` (not under version control; pre-change
      state archived at `~/Desktop/Sepmo-v2.2-archive`): spine `version: "2.3"` + R11/R12/R13 after
      R10 + the widened *Incident retrospectives* trigger (Amendment D) + the v2.3 changelog head;
      refs 02/03/05/06/08 amended; template gained the optional `critic_engine` row (Amendment E,
      runtime-neutral). Refs 01/04/07 byte-identical to the archive.
- [x] **2. This instance re-bound** — spine replaced with a **byte-identical copy** of the master
      (`cmp`-verified, closing the 2026-07-13 install's transcription caveat); the same v2.3 deltas
      applied **in place** to the fork's references, preserving their ASF headers and their own
      (legitimately diverged) lineage wording — the two-lineage reconciliation the CCR settled.
      Lineage-local adaptations: the new vigilance trigger is **T10** here (the fork numbers
      triggers T1–T9, the master numbers watch items W1–W9), and ref 05's engine constraint (3)
      cites "the canonical categories" rather than the master's "ten" (this lineage tables nine).
- [x] **3. Manifest + map** — `spine_version: v2.3` (frontmatter fact + I-2, now carrying the
      byte-identity proof); `contingency_mechanics` SLIMMED to project values + an R11–R13 pointer
      (its own recorded duty — canon owns the rules now); NEW `critic_engine` row binding the
      **default** (the spine's own Critic stage via the standing sub-agent hard break, no external
      engine); the Debug canon-gap filing flipped to RESOLVED; `map.md` in lockstep (v2.3,
      R1–R13, tunables list).
- [x] **4. Ledger + tracker** — the G4 incident section's open disposition completed (WAIVED,
      #173); this block added.
- [ ] **5. Independent Critic (fresh context, per the sub-agent policy row)** — dispatched;
      convergence is the Critic's call (R4). Flip on CONVERGED, then push for PR.
      **Cycle 1 → CHANGES_REQUIRED** (1 S2 + 5 S3); cycle 2 dispositions, all APPLIED:
      **S2** — the CCR's own `Status:` still read "DRAFTED, awaiting user ratification" while the
      manifest, this file and the master all said RATIFIED; fixed, together with the stale
      *What already landed* close. **S3-1** — the template's "the ten canonical ones" is false in
      a lineage tabling nine; fixed **at the master** (portable canon may not hard-code a
      per-lineage count) and re-copied. **S3-2** — the changelog's "W9 in this lineage" travels
      into instances numbering T1–T9; fixed at the master to "each lineage's next unused id
      (**W9** in the master's references)", keeping instance ids out of portable canon.
      **S3-3** — the spine's *PRE_EXECUTION_REVIEW* list omitted the fifth confirmation that
      ref 02 gained; appended at the master, spine re-copied (I-2 byte count 41,957 → 42,055).
      **S3-4** — the two-lineage decision lived only in this file, which archives out of required
      reading; now durable in the manifest's *Debug* list and the metrics feed-forward
      dispositions. **S3-5** — the `contingency_mechanics` row's residual R11(a)/(b) glosses
      trimmed to project selections.

## FOLLOW-UPS BUNDLE (2026-07-18, signed off) — audit follow-up ledger, 5 groups, OO-High AC, ONE branch

Branch `fix/audit-followups-bundle` (Mode B; user-directed: Opus-high Actor/Critic per group, Fable
closing Critic). Closes the overnight block's follow-up ledger. All groups converged:

- [x] **G1 residue closure** (`ae5d4385`, 1 cycle): `get_delete_vector_for_path` poison fail-open →
      `recover_poison` (resurrection class closed; lock-site sweep — `apply`'s fail-CLOSED mutex
      justified-not-converted); exhaustive op-class partition guard test (E0004 compile-break
      proven); R160 cell-text corrections. Critic: CONVERGED, 3 LOW.
- [x] **G2 `ErrorKind::NamespaceNotEmpty`** (`5a7e5790`, 1 cycle): additive core variant (public-API
      addition flagged); SQL not-empty drop flipped; HMS `drop_database` O2 arm FLIPPED on the
      Actor's bytecode adjudication (`cascade=false` ⇒ Java maps `InvalidOperationException`
      unconditionally to `NamespaceNotEmptyException`, offsets 41-60 — Critic third-decoded,
      upheld). Critic: CONVERGED, 1 LOW (prose).
- [x] **G3 config Debug sweep + Glue flip** (`775e2c24` + c2 `61948eff` + c3 `db26147a` + c4
      `6ec5a765`, **4 cycles**): four catalog configs redacted-Debug via pub-promoted
      `is_secret_prop_key` (public-API addition); SqlCatalog raw-props leak found+fixed (Actor
      deviation, upheld); G3b Glue not-empty flip. **The DSN redactor took 3 refutations** —
      c1 truncate-order leak (`/?#`-in-password), c2 acceptance-scan leak one layer deeper
      (`p@x/y@host`), c3 sound span rule but unanchored scheme-strip leak (`user:p@ss://host`),
      c4 anchored `[A-Za-z][A-Za-z0-9+.-]*` strip ruled AIRTIGHT (27-probe attack, zero leaks).
      Lesson promoted: security-sensitive string handling starts from the coarse PROVABLE rule.
      Cumulative Critic: CONVERGED.
- [x] **G4 vended storage credentials, R160 ❌→🟡** (`3acf671e` + c2 `20c36b23`, 2 cycles):
      longest-prefix/vended-wins/silent-no-match decoded from `S3FileIO` bytecode and wired into
      `load_file_io`; c1 Critic MEDIUM = honesty (per-accessed-path vs once-at-metadata_location
      granularity collapse unnamed in the cell) → c2 named it (cell + call-site comment + both
      divergence cases). Cumulative Critic: CONVERGED.
- [x] **G5 OAuth2 auto-refresh, R159 ❌→🟡** (`4a00ae94`, 1 cycle): lazy single-flight
      refresh-before-use reproducing Java `scheduleTokenRefresh` math byte-exactly
      (`min(ttl/10, 5min)` window, 10ms floor); missing-`expires_in` ⇒ never refresh (Java-exact —
      the 3600s default is token-prop-only, Critic-verified); refresh failure keeps old token;
      disabled = legacy verbatim. 8-concurrent→1-hit single-flight pin. Two grant divergences
      disclosed (client_credentials vs token-exchange; config-token clients not refreshed —
      Critic F1: disclosure broader than "no credential", rewording queued). Critic: CONVERGED,
      2 LOW.
- [x] **Closing pass: Fable bundle Critic** over the full branch — **CONVERGED** 2026-07-18,
      zero MEDIUM+. Full-workspace clippy (protoc present) + all six lib suites green (iceberg
      2831 · sql 80 · hms 43 · glue 30 · s3tables 28 · rest 82); cross-group seams verified
      (G2→G3 variant dependency ordered; G4×G5 rest-crate hunks disjoint; novel probe: vended
      creds × token refresh compose in one mockito session, endpoint hit exactly 2×); public-API
      ledger = exactly the two flagged additions; matrix/tracker ruled truthful; 3 cross-group
      mutations re-run RED + md5-identical restores. 4 LOW (rest-crate local needle list
      pre-existing/not unified; G5's expect nit + DEVIATIONS rewording queued; G1 residue
      descriptions live in its Critic report → pasted into the PR body). Pushed after flip.

## OVERNIGHT BLOCK (2026-07-17, signed off) — audit-2026-07-17 remediation, 5 units, mixed FF/OO-max

Source: `~/Desktop/repo-audit-iceberg-rust-2026-07-17.md` (5-agent external audit, 74 facets/~34 roots),
verified same day by 3 parallel verification agents + inline P0 checks (memory:
`audit-2026-07-17-verification.md`). Verdicts: 0 stale, 1 invalid (BUG-005 — `?` propagates), Critical
BUG-001 confirmed, null family BUG-002/003/011 bytecode-confirmed vs Java nulls-first total-order.
**User sign-off: run ALL units incl. E; A2 design call LOCKED = Java nulls-first parity (DataFusion's
Inexact re-filter stays authoritative for SQL 3VL consumers).** Ladders: A1/A2 = FF (Fable–Fable);
bundle-final Critic + B/C/D/E = OO at max effort. No merges overnight — all SQMs are the user's in the
morning. This tracker rides branch A only (5 parallel branches editing one todo region would conflict
at SQM); B–E record themselves in PR bodies + the morning memory append.

- [x] **A1 (FF): BUG-001 real NaN evaluation** — branch `fix/audit-nan-null-residual-parity`,
      charter `task/a1-nan-residual-brief.md`. `is_nan`→always-true / `not_nan`→always-false on
      present columns in `arrow/reader.rs` PredicateConverter (RowFilter — rows dropped at read) AND
      `arrow/record_batch_predicate.rs` (eq-delete application). DF maps `isnan` in
      (`expr_to_predicate.rs:229`); `NOT isnan` over-drops through SQL. Fix = real per-row NaN checks;
      Java oracle (`Evaluator`/`NaNUtil`, cached jars) decides null-cell + missing-col + non-float
      binding arms.
      *Done 2026-07-17 (Fable Actor): shared two-valued `is_nan_row_mask`/`not_nan_row_mask`
      (`record_batch_predicate.rs`, imported by both evaluators) — elementwise `is_valid && is_nan`,
      NULL cell ⇒ not-NaN (self-decoded `NaNUtil.isNaN` bytecode: null ⇒ `iconst_0` at offsets 0-5;
      non-float ⇒ false at 42-43 mirrored for the bind-unreachable arm). Missing-column arms
      confirmed already-Java-correct (unchanged). Both engines reject non-float binds (Java
      `bindUnaryOperation` 158-171/202-215 ValidationException ≙ Rust `predicate.rs:399` DataInvalid)
      — no binding deviation. Pins: 4 unit truth-table tests (incl. crafted NaN-under-null-slot
      buffer), full-path `TableScan::to_arrow` scan pin (new `new_nan_floats` fixture, f64+f32 both
      directions + before-column-existed file), DF e2e `isnan`/`NOT isnan` (the silent-zero-rows
      regression). 9 mutations each independently RED, restores byte-verified. Gate green
      (typos/fmt/clippy×2/lib 2789/DF 53+88/artifacts); DF doc-test FAIL = the known pre-existing
      `-p` rt-multi-thread artifact (2026-07-10 A3 note). Charter file reworded minimally to pass
      the typos gate (SHA lengthened, two hyphenated words joined) — disclosed deviation.*
- [x] **A2 (FF): null-semantics family BUG-002/003/011** — same branch, stacked. Port Java
      nulls-first total-order (bytecode truth table: null<lit=T, null<=lit=T, null!=lit=T,
      null==lit=F, null>lit=F) consistently across `reader.rs` PredicateConverter,
      `record_batch_predicate.rs`, and `expr/visitors/expression_evaluator.rs` (null partition
      `<`/`<=` currently over-prunes). Missing-col `not_eq` → true. Pin matrix = every op ×
      {null cell, missing col, null partition}.
      *Done 2026-07-17 (Fable Actor): self-decoded the full oracle (NullsFirst compare(null,x)
      = -1 at offsets 19-20; in→false via HashSet + CharSequenceSet instanceof-guard;
      startsWith ifnull 38 → false; partition oracle CONFIRMED = ManifestReader.evaluator()
      applying EvalVisitor to file().partition()). Design: EVERY leaf mask two-valued via new
      shared `null_filled(mask, verdict)` (`record_batch_predicate.rs`) — the Java-null=FALSE
      ops too, because bind() PRESERVES `Predicate::Not` and `NOT(eq)` over a 3VL eq mask
      would drop where Java says TRUE (composition pin proves it). Fixed: reader missing-col
      `not_eq` false→true (BUG-002) + all 8 present-column closures null-filled (BUG-003);
      record-batch `not_eq` verdict + in/not_in fills; partition `<`/`<=` None→true (BUG-011).
      Eq-delete consequence (Java StructLikeSet-correct): a NULL key cell now SURVIVES value
      deletes — 2 delete_filter pins updated to the Java verdict, equality_delete_set bail-doc
      rewritten (fast path stays conservative). Pins: 22-case mask truth table + NOT-composition
      + buffer-under-null-slot; full-path scan `!=`/`<`/`<=` (null row + schema-evolved file);
      13-case null-partition sweep; DF e2e 3VL-refilter documentation pin. 11 mutations each
      independently RED, restores byte-verified (md5). Lib 2789→2795, DF suite 54.*
- [x] **A-final: bundle Critic (Opus max)** over the stacked branch; then push + PR body.
      Done 2026-07-18: **CONVERGED, zero blocking findings.** Third oracle decode matched all 13
      citations; both adjudications ruled FOR the Actor (eq-delete `[T,F,F]` flip = Java-correct per
      `Deletes$EqualitySetDeleteFilter`/`StructLikeComparator` decode, necessary consequence of the
      in-scope not_eq fix; null-filling coinciding ops = safe over-delivery, "necessity" overstated —
      LOW). 5/5 mutations re-run RED + restored md5-identical; 2 novel probes (OR/double-NOT
      composition; A1×A2 NaN+NULL not_in cross-cut) pass two-valued. Gate re-run green (2795 lib;
      DF 88+54+1+4). 3 LOW residues named (notEq offset attribution; necessity framing; 3-consumer
      blast-radius narrative gap — all verified Java-correct by the Critic itself).
- [ ] **B (OO max): MoR eq-delete panic/hang** — branch `fix/audit-mor-eqdel-panic-hang`.
      `equality_ids.unwrap()` (`caching_delete_file_loader.rs:298`) → DataInvalid; oneshot
      sender-drop must reach a terminal state + notify (kills the forever-hang,
      `delete_filter.rs:340`); unify the 13 poison-unwrap sites in
      `delete_filter.rs`/`delete_file_index.rs` onto the crate's poison-recovery policy;
      `scan/cache.rs` (SAF-006) in-scope stretch.
- [ ] **C (OO max): predicate serde arity validation (SAF-004)** — branch
      `fix/audit-predicate-serde-arity`. Derived `Deserialize` on Unary/Binary/Set expressions
      bypasses ctor debug_asserts (gone in release) → wire-reachable visitor `panic!`s. Validate
      op/arity at deserialize; visitor dispatch panics → typed `Err`. Fallible `new` = out of scope
      (breaking surface).
- [ ] **D (OO max): hardening quick-wins** — branch `infra/audit-hardening-quickwins`. Redacted
      `Debug` for `StorageConfig` (closes SEC-003→SEC-002 credential chain) + `HttpClient`
      `extra_headers` (SEC-008); SAF-008 default-literal validation (error, not serialize-panic);
      GAP_MATRIX rows (fresh R-ids) for the two verification-discovered parity gaps: Java OAuth
      auto-refresh (`OAuth2Manager` keepRefreshed) and vended `storage_credentials` consumption.
- [ ] **E (OO max, run even over budget per user): typed error kinds** — branch
      `fix/audit-typed-error-kinds`. CQ-002/CQ-003: SQL + HMS catalogs emit
      TableNotFound/NamespaceNotFound/TableAlreadyExists instead of Unexpected.
- Deferred (charters later, matrix rows tonight via D): token-refresh + vended-creds
  implementations; parity-shared-with-Java hardening ledger (SEC-001/003/004/009/012 etc.).

## ACTIVE UNIT (2026-07-17d): G3 HMS type-string parity — branch `fix/g3-hms-type-string-parity`

User-signed 2026-07-17: **FF AC (Fable Actor / independent Fable Critic)**. The HMS
sibling of #153/#155 with a DIFFERENT oracle: Java `HiveSchemaUtil.convertToTypeString`
THROWS where Glue lowercases, and timestamptz is Hive-version-gated (client-classpath
detection in Java → a config knob here, default Hive 3+). Spec:
[g3-hms-type-string-parity-brief.md](g3-hms-type-string-parity-brief.md) (C-1…C-7,
oracle pre-decoded from `iceberg-hive-metastore-1.10.0.jar` bytecode). Includes a
disclosed capability regression toward parity: `"timestamp_ns"` emission removed (Java
throws for nano).

- [x] **Build** — DONE 2026-07-17 (353bf920, Fable Actor): C-1 `hive_version` knob
      (bare-snake-case per fork HMS conventions; first-digit parse; default Hive3Plus;
      invalid ⇒ loud DataInvalid at `load()` BEFORE construction; key filtered from
      pass-through) · C-2 gated tz strings byte-exact · C-3 nano+Unknown one
      FeatureUnsupported arm, `"timestamp_ns"` emission removed (disclosed regression
      toward parity) · C-4 separator `","`, lambda verified · C-5 message
      `"{type} is not supported"` — the Actor's bytecode read CORRECTED the brief's
      wrong message shape · C-6 10 pins, 4 mutations · C-7 R91 HMS clause. 24 hms tests
      (was 20).
- [x] **Critic** — CONVERGED 2026-07-17 (independent Fable, fresh context, zero blocking
      findings). Settled the message dispute FOR the Actor (BootstrapMethods recipe
      `" is not supported"`; brief was wrong); verified Rust Display byte-identity
      for all three rejected types vs Java `Type.toString()` bytecode. Call-site sweep:
      exactly one production `from_iceberg` caller (create_table via
      `convert_to_hive_table`), knob-threaded — no hardcoded-default site. 6 mutations
      (incl. both-branches-same + accept-"5" + static-message vacuity probes) + a novel
      `map<string,timestamptz>` execution both versions. Residue (LOW): Java's `"1"`-major
      handling is finer-grained (HIVE_1_2/NOT_SUPPORTED buckets — behaviorally identical
      under the only consulted gate); pre-existing `thrift_transport` silent-fallback
      inconsistency (out of scope, future hygiene pass).
- [x] **Close-out** — tracker flipped, pushed, PR body delivered. G-series (G1/G2/G3)
      COMPLETE.

## DONE 2026-07-17 (merged #156): G2 incremental-scan name-mapping pin — was branch `fix/g2-incremental-name-mapping-pin`

User-signed 2026-07-17: **FF AC (Fable Actor / independent Fable Critic)** — the user's
chosen mode for G2+G3. Test-only unit closing the #154 Critic's residue (incremental
wiring correct but unpinned). Spec:
[g2-incremental-name-mapping-pin-brief.md](g2-incremental-name-mapping-pin-brief.md)
(C-1…C-5; escape hatch only for a proven-broken wiring).

- [x] **Build** (Fable Actor): C-1 plan-level pin · C-2 e2e contrast via incremental
      stream · C-3 absent-property fallback pin · C-4 live mutation proof (incremental
      site RED, snapshot pins GREEN) · C-5 reuse #154 fixtures/helpers.
      *Done 2026-07-17: 3 pins beside the incremental tests (`scan/incremental.rs`),
      reusing the #154 fixture/helpers (`NAME_MAPPING_X1_Y2` + `decode_int64_column`
      promoted to `pub` in `scan/mod.rs` tests). Mutation RED set = exactly the 2 new
      C-1/C-2 pins; all 5 snapshot pins + C-3 stayed green; wiring proven correct — no
      production change, no matrix edit (R143 does not claim incremental coverage).*
- [x] **Critic** — CONVERGED 2026-07-17 (independent Fable, fresh context, zero blocking
      findings). Test-only claim PROVEN (every hunk inside `#[cfg(test)]`; compiled crate
      bit-for-bit unaffected). Re-ran the Actor's mutation (exact unique-guard RED set
      confirmed) + 3 novel probes: expected-array swap (C-2 discriminates live values),
      property-forced-onto-C-3-fixture (RED at the `is_none` assert — pins the fallback,
      not "a read succeeded"), degenerate range (file-outside-range cannot pass). Reader-
      path fidelity verified: test defaults byte-identical to `to_arrow`'s knobs. Residue
      (LOW): value asserts read `batches[0]` only (safe at the 4-row fixture); the
      to_arrow-path docstring holds for default-configured consumers.
- [x] **Close-out** — tracker flipped, pushed, PR body delivered. G3 (HMS timestamptz, FF)
      follows after merge.

## DONE 2026-07-17 (merged #155): G1 Glue type-string byte-parity — was branch `fix/g1-glue-type-string-byte-parity`

User-signed 2026-07-17 (the "G1→G2→G3" follow-up sequence after #152/#153/#154 merged;
G2 = BUG-002 incremental-scan pin, G3 = HMS timestamptz design unit — queued next). OO AC.
Spec: [g1-glue-type-string-byte-parity-brief.md](g1-glue-type-string-byte-parity-brief.md)
(C-1…C-7, oracle pre-decoded from `iceberg-aws-1.10.0.jar` bytecode). Closes the #153
Fable-max Critic residues: struct-join separator (fork `", "` vs Java `","`), naive-nano
string (`"timestamp_ns"` vs Java's uniform `"timestamp_nano"`), plus the newly-surfaced
Unknown question (Java renders `"unknown"`, never throws; fork rejects).

- [x] **Build** — DONE 2026-07-17 (0aa61074, Opus Actor): C-1 separator `", "`→`","` (all
      struct pins updated) · C-2 both nano variants → `"timestamp_nano"` (freeze lifted) ·
      C-3 Unknown flipped reject→`"unknown"` (branch (a): UNKNOWN absent from the
      `$SwitchMap` → Java's lowercase default, never throws; no fork dependency on the
      reject) · C-4 field lambda decoded (`%s:%s`, fork already matched; citation added) ·
      C-5 citations bytecode-verified; #153 byte-false comment corrected · C-6 byte-exact
      pins incl. two-depth struct-in-struct; 3 mutations proven · C-7 R91 cell corrected
      (glue renders "unknown"; hms fact left to G3). 27 crate tests.
- [x] **Critic** — CONVERGED 2026-07-17 (independent Opus, fresh context, zero findings at
      the S2 floor). Third independent bytecode decode: zero string divergence, all comment
      offsets byte-accurate. Re-ran M1–M3 + extra M4 (field-lambda format corrupt → 4 pins
      RED — coverage confirmed). Duty-3 verdict: Java's `toColumns` reaches `toTypeString`
      with NO pre-converter validation, so end-to-end behavior matches (an unknown column
      publishes "unknown" on both sides; V2-vs-V3 gate identical both sides). Residue
      (LOW): live-Glue acceptance of "unknown"/"timestamp_nano" strings unverifiable
      offline (parity holds either way — both engines emit byte-identical strings); Java's
      unescaped `name:type` ambiguity for colon-bearing field names is shared, not a fork
      defect.
- [x] **Close-out** — tracker flipped, pushed, PR body delivered. NEXT: G2 (BUG-002
      incremental-scan pin) after this merges, then G3 (HMS timestamptz design unit).

## ACTIVE UNIT (2026-07-17): BUG-002 name-mapping scan wiring — branch `fix/bug-002-name-mapping-scan-wiring`

User-signed 2026-07-17: OO AC (Opus Actor / independent Opus Critic). Spec:
[bug-002-name-mapping-scan-brief.md](bug-002-name-mapping-scan-brief.md) (C-1…C-7); this
section is the tracker. The bug (external audit 2026-07-10, deferred backlog): the scan
hardcodes `name_mapping: None` (`scan/context.rs` TODO) while the downstream
(`FileScanTask.name_mapping` → ArrowReader `apply_name_mapping_to_arrow_schema`) is fully
built — ID-less-Parquet tables with `schema.name-mapping.default` read via position
fallback (wrong-data class) instead of the mapping (Java-divergent).

- [x] **Build** — DONE 2026-07-17 (8680f149 fix+pins+R143 cell, 3aaca2fa interop suite,
      4d559190 empty/whitespace pins; Opus Actor). All 7 clauses landed, PLUS the e2e pin
      exposed TWO further wrong-column bugs the brief's escape hatch authorized fixing, each
      uniquely mutation-pinned: (a) `arrow/reader.rs` used the positional projection mask
      even when a mapping applied; (b) `record_batch_transformer.rs::compare_schemas`
      compared by position → reordered name-mapped files relabeled in place. C-4 decided
      delete-task sites stay `None` (javap: Java `GenericReader`/`DeleteFilter`/
      `BaseDeleteLoader` have zero name-mapping references — engine data reads only).
      Deviation (flagged for merge): `dev/java-interop/pom.xml` +2 test-oracle deps
      (parquet-avro 1.16.0, hadoop-client-api 3.3.6) to write ID-less parquet.
- [x] **Critic** — CONVERGED 2026-07-17 (independent Opus, fresh context, zero findings at
      the S2 floor). Re-decoded the Java jars itself (C-3/C-4 confirmed; novel non-array
      probes matched Java beyond the enumerated partition), re-ran all 3 mutations (each fix
      distinctly+uniquely pinned), blast-radius analysis: both downstream fixes are
      no-op for embedded-id and no-mapping files. Interop: 50/50 discovery, D1 content-equal,
      sabotage RED via the id-less differential (embedded ids would have made it vacuous-
      green — RED proves the mapping is load-bearing). Residue (LOW): incremental-scan
      wiring correct but untested (F-1); driver floor comment staleness (F-2, fixed in
      close-out).
- [x] **Close-out** — tracker flipped, floor comment refreshed, pushed, PR body delivered.

## ACTIVE UNIT (2026-07-16): R158 Java interop battery (🟡→✅) — branch `parity/r158-staged-txn-interop`

User-signed 2026-07-16: OO AC (Opus Actor / Opus Critic, both at MAX effort) via the Workflow
ladder. Spec: [r158-staged-txn-interop-brief.md](r158-staged-txn-interop-brief.md) (C-1…C-9 +
the E-INV replace-invariant enumeration); this section is the tracker.

- [x] **Build** — DONE 2026-07-16 (9b421afa, Opus-max Actor): suite `run-interop-staged-txn.sh`
      (7 steps: Java d1 gen → Rust d2 gen → C-5 cross-check → Java verify → Rust verify →
      3 sabotages) + `InteropOracle.java` scenario (REAL `Transactions.create/replaceTableTransaction`
      over a committable `LocalTableOperations`, mirroring `BaseMetastoreCatalog` — no hand-rolled
      metadata) + `tests/interop_staged_txn.rs`; all C-1…C-9, E-INV(1–7) pinned per-cycle BOTH
      directions; V1-stays-V1 + property-directed upgrade both directions; SB1–SB3 RED +
      hard-fail-never-skip (pattern-absent ⇒ exit 3 ⇒ restore + exit 1); floor 48→49; zero
      production-code change (no parity bug exposed — the D1/D4 fixes held against real Java).
- [x] **Critic ladder** — CONVERGED cycle 1, ZERO findings at the S2 floor (Opus-max, fresh
      context). Critic re-ran the full gate + suite + selftest + 49-discovery itself, and
      mutation-tested SIX verifier assertions across BOTH directions (spliced UUID → Rust RED;
      fmtv leak + last_column_id reduction → Java RED; committed SB1–SB3 observed RED) — proving
      the cross-engine wiring non-tautological (mutating d1 flips only Rust, d2 only Java).
      Java fidelity verified against /tmp/iceberg-java-ref source (buildReplacement,
      persistedProperties, assignFreshIds seeding). Full taxonomy attestation in the run record.
- [x] **Flip + close** — R158 ✅ (residues (1)–(3) retained verbatim; dated 2026-07-16),
      ENGINE_CONTRACT §8a item 5 → PROVEN. Pushed for PR.

## ACTIVE UNIT (2026-07-15): fork-atomicity remediation (R158 staged create/replace) — branch `feat/replace-table-transaction`

SEPMO Actor–Critic unit hardening the just-landed R158 staged transaction (tip 9280320b). Two
findings, both correctness/atomicity, tests in the same commit as the fix; mutation-proved RED on
revert. No `--all-features` (per unit charter); no push.

- [x] **C1 (N1) — create-publish is not atomic.** `MemoryCatalog::register_table` (the
  `publish_create_table` default) inserted the pointer THEN read the metadata; a reload failure
  (staged metadata written through a FileIO the catalog cannot read) left `table_exists=true` +
  `load_table` erroring — a half-created table breaking `IF NOT EXISTS` retry. Fix: read metadata
  BEFORE inserting, under the one catalog lock. Guarantee documented on `publish_create_table` (for
  other Catalog impls) + ENGINE_CONTRACT §8a. Pin: `create_publish_reload_failure_leaves_no_catalog_entry`.
- [x] **C2 (N2) — CREATE OR REPLACE location drift.** `begin_replace` baked
  `"{existing}__staged_replace"` into the new metadata's `location()` and never reset it, so every
  replace relocated the table and compounded the suffix. Fix: keep the stable existing/caller
  location (staging = deferring the pointer swap, not a separate dir). Pin:
  `replace_cycle_keeps_location_stable_and_reads_latest` (triple cycle; location == original each
  publish; reads expose the latest replace's data).

CLOSED 2026-07-15 after a 4-cycle independent Opus Critic ladder (OO AC) over the full branch:

- [x] **CF-1 (D1, MEDIUM) — replace built fresh metadata, inverting Java `buildReplacement`.**
  `begin_replace` used `from_table_creation` (fresh UUID, empty snapshots/metadata-log). Fix
  4b944152: seed `new_from_metadata` — UUID + snapshot history + metadata log retained, ONLY the
  main ref removed, format version never downgraded. Pin:
  `replace_retains_uuid_history_and_metadata_log` (mutation-proven vs fresh-UUID seed AND dropped
  `remove_ref(MAIN)`).
- [x] **CF-4 (D4, MEDIUM) — replace silently upgraded V1→V2** via `max(previous,
  creation.format_version)` with the builder's V2 default. Fix 8c7d2c02: version derived ONLY from
  the `format-version` property (popped before `set_properties`, mirroring Java
  `persistedProperties`); absent ⇒ keep existing; downgrade/unparsable ⇒ DataInvalid;
  `creation.format_version` ignored on replace. Pins:
  `replace_default_creation_preserves_v1_format_version` (mutation-proven),
  `replace_upgrades_format_version_by_property`, `replace_downgrade_attempt_errors_and_keeps_original`.
- [x] **CF-5 (D5, LOW) — docs over-claimed `assignFreshIds` parity**: corrected to caller-ids-as-is
  + named residue (base-aware fresh-id helper = follow-up); new pin
  `replace_with_different_schema_keeps_caller_ids`.
- [x] **F-1 (MEDIUM, found by cycle-3 Critic) — the "unparsable format-version ⇒ DataInvalid"
  claim had ZERO tests** (silent-fallback mutation survived the suite). Fix 2e08a6e4 (test-only):
  `replace_invalid_format_version_property_errors_and_keeps_original` over 8 invalid values incl.
  `"2 "` (anti-trim), each pinning DataInvalid + original unchanged; mutation-proven RED.
- CF-2/CF-3 (LOW) accepted as NAMED residue in the R158 cell: replace-publish lacks
  read-before-swap validation (create has it); staged replace restarts metadata versioning at
  v0/v1. Cycle-4 Critic **CONVERGED** (zero findings; gate 2769 lib green + fmt/clippy/anchors/
  typos). Pushed for PR.

## ACTIVE UNIT (2026-07-13): SEPMO canon v2.2 upgrade + manifest re-instantiation

User-directed 2026-07-13 ("We have updated SEPMO rules we need to implement"): bring the repo's
SEPMO installation (pre-ledger lineage, installed 2026-06-15/25) up to the new master canon
**v2.2** the user supplied (spine + binding-manifest template). One branch
`infra/sepmo-canon-v2.2`, one PR; STANDARD path (governance surface; >5 files and >150 lines —
LIGHT criteria 1/3 fail; uncertain routes STANDARD anyway). Docs-only: no crate code, no matrix
row.

- [x] **1. Canon lands verbatim** — `skills/sepmo/SKILL.md` replaced with the v2.2 spine
      (frontmatter `version: "2.2"`); NEW `skills/sepmo/binding-manifest.template.md` (the
      portable template; ASF header prepended per the sibling convention — the one sanctioned
      local touch, matching how every references/ file carries it).
      - Caveat, disclosed for the PR: the canon text was transcribed from the user's message,
        not copied from a master file — the user should eyeball the SKILL.md diff against the
        master before merge. Canon defect FILED (manifest Debug): the spine's *Model
        assumption* carries a "For this repo ... single-agent default" instantiation artifact;
        not patched locally, does not bind (manifest + CLAUDE.md win).
- [x] **2. Manifest re-instantiated** — `skills/sepmo/binding-manifest.md` rebuilt per the
      template protocol: role rows all BIND (existing homes), `spine_version: v2.2`, tunables
      bound (two-tier `green_commands` + CI-only exception record + parity-guard-absence
      justification, `severity_floor: S2` raise with provenance, hard-break
      `context_break_mechanics`, `s0_fresh_execution: N/A` per the standing hard break,
      `metrics_ledger_location: task/sepmo-metrics.md`, `taxonomy_extensions: java-parity +
      format-stability`), instantiation checklist I-1…I-10 proven.
      - Note: `severity_floor: S2` is bar-PRESERVING (the old install blocked MEDIUM+ ≈ S2),
        so it lands at instantiation under the asymmetric feed-forward rule rather than
        waiting for a retrospective; provenance stamped in the row.
- [x] **3. Reference amendments** (the v2.1 + v2.2 required sets + spine coherence): 01
      proposition ledger + enumeration obligation + reworked examples; 02 PRE_EXECUTION_REVIEW
      / states renumbered / S-scale / LIGHT-STANDARD rubric / R3 input restriction / R7
      readiness incl. pre-merge gate + exception record; 03 doubles as the one-time
      pre-execution review format; 04 per-element pinning + R5 regression proof + R6
      dispositions; 05 canonical taxonomy + coverage attestation + span check + fresh-execution
      step + S-scale; 06 Invariant V reframe + unledgered-claim/silent-gate-skip watch items;
      07 state 5 + R8 embedded-evidence verification + flag disclosure; 08 metrics ledger
      (incl. `environment_drift_events`) + incident retrospective + asymmetric feed-forward.
- [x] **4. New artifacts + navigation** — CREATE `task/sepmo-metrics.md` (the bound metrics
      ledger, ref-08 metric set, no sections yet — first one lands with the first
      retrospective; the 2026-07-11 nightly-interop CI failure is pre-named as the first
      `environment_drift_events` candidate); refresh `skills/sepmo/map.md` in the same change.
      Fixed in passing (in-scope files): two pre-existing broken `../../CLAUDE.md` links in
      the refs 04/05 repo-note boxes (needed three levels up from `references/`).
- [x] **5. Gate + independent Critic → push** — DONE 2026-07-13: round-2 re-attestation
      **CONVERGED** — the Critic independently re-ran all four gates green on b440629b,
      re-executed the no-default-features compile itself (exit 0, byte-identical to
      ci.yml:149), re-swept every `row *…*` citation to resolution, and re-falsified I-4
      (now true). One NEW S3 advisory filed and accepted-open (F-SEPMO22-3: exception
      entry 5 over-states the platform residual — CI's `tests` job is ubuntu-only, so the
      gap belongs to build/no-default-features + check-on-macOS; conservative-direction
      error, "never blocks" per the spine; tighten at the next touch of the row —
      disclosed in the PR body). Pushed; merge is the user's. Meta-note worth keeping:
      round 1's S1 was the new machinery catching a real binding defect in its own
      install — the R7 silent-skip rule fired on the very unit that introduced it.
      - Round 1 (2026-07-13): Critic CHANGES_REQUIRED with 2 findings, both fixed same day —
        **F-SEPMO22-1 (S1, binding defect):** CI's `build_with_no_default_features` job was
        neither mirrored nor excepted; REMEDIATED by adding
        `cargo build -p iceberg --no-default-features` to the pre-merge gate (run live:
        green, 20.8s) + a fifth exception entry (non-Linux runners platform residual).
        **F-SEPMO22-2 (S2):** three reference citations still named the old manifest row
        `Capability status (SSOT)` after the template-aligned rename; REMEDIATED to
        `Status SSOT` (refs 02 ×1, 07 ×2; repo-wide grep now clean). Regression proof for
        both is structural (grep + live command), not test-expressible — R5 justification.
        Sent back for re-attestation.

## ACTIVE UNIT (2026-07-10): AUDIT TIER 1 Mode B bundle — A1→A3→A2→A4, one branch, one PR

User-approved 2026-07-10 triage of the external five-agent audit (run on the overnight branch;
orchestrator spot-verified all four roots in-tree before scoping). Full ladder-ready briefs:
[audit-2026-07-10-tier1-tier3-briefs.md](audit-2026-07-10-tier1-tier3-briefs.md) — the briefs
file is the spec; this section is the tracker. **Mode B** per [pr-per-work-cycle]: one bundle
branch `fix/audit-tier1-2026-07`, four SEQUENTIAL parity-increment ladders, orchestrator
gates+commits after each unit, ONE final independent SEPMO bundle Critic over the whole branch
diff; push on CONVERGED; single PR. Execution order **A1 → A3 → A2 → A4** (small corruption
fixes first; A4 last — it loosens a must-match guard and deserves the freshest scrutiny).
Tier 3 (ops) already landed separately as `infra/audit-ops-2026-07` (Critic CONVERGED, pushed).
Contingency: a unit whose ladder cannot converge is parked on `fix/audit-tier1-parked-A<n>`
and the bundle resets to the last good commit; the bundle ships with the units that converged.

- [x] **A1 — zero-width / oversized partition transforms** (BUG-001/SAF-001/BUG-013): reject
  `bucket[N]`/`truncate[W]` outside `1..=i32::MAX` at parse per Java preconditions; kill the
  `rem_euclid(0)` panic and the `mod_n as i32` wrap; defense-in-depth typed error at apply.
  - Outcome (ladder SHIP_WITH_NITS, 0 remediation rounds, mutations 5/5 RED, 16 tests): three
    independent doors — `Transform::validate()` in FromStr (the ONLY serde route, so metadata/
    spec/sort-order/manifest deserialization all covered), fallible `Bucket::new`/
    `Truncate::new` at the apply door (deliberately duplicated guards), and both
    partition-spec builders. `Bucket` now stores `mod_n: i32` (checked at construction;
    `bucket_n` cast-free). Java precondition text pinned verbatim from 1.10.0 jar bytecode;
    `Transforms.fromString` parses via Java int, confirming the 1..=i32::MAX parity bound.
    Crown jewel: hostile table-metadata JSON with `bucket[0]`/`bucket[2147483648]` fails at
    deserialization with DataInvalid (control `bucket[16]` parses). Argued deviation: the
    Java message text is asserted at the PartitionSpec serde door, not TableMetadata level
    (untagged-enum serde swallows inner messages; kind stays DataInvalid there).
- [x] **A3 — negative/null position-delete positions** (BUG-005): fail closed with DataInvalid
  at `caching_delete_file_loader.rs` (3 sites + a `.unwrap()`); checked `u64::try_from`.
  CLOSED: ladder SHIP, 0 remediation rounds, critic mutations 5/5 — details in the builder
  notes below (audit's :918 claim corrected: test-oracle code, not production).
    - Builder plan (2026-07-10, live-audited): the audit's ":918 `.unwrap() as u64`" site is
      INSIDE `#[cfg(test)]` (the M5 per-row reference oracle), NOT a production path — and the
      production null guard already exists (L516-522, typed DataInvalid, but names neither the
      delete file nor the column). Real production bugs = the two `pos as u64` wraps
      (L526/L537). Java oracle (source + 1.10.0 jar bytecode): `BitmapPositionDeleteIndex.
      delete(long)` → `RoaringPositionBitmap.set` → `validatePosition`
      (RoaringPositionBitmap.java L311-316) throws IllegalArgumentException for pos < 0;
      `pos` is a REQUIRED column (MetadataColumns.java L70-74) and Java NPEs unboxing a null
      (Deletes.java L146). Plan: (1) thread `delete_file_path` into
      `parse_positional_deletes_record_batch_stream` for error context; (2) split the null
      let-else into per-column typed errors naming the delete file; (3) per-branch checked
      conversion via a `checked_delete_position` helper (kept per-site so each brief-mandated
      mutation stays independently RED); (4) fix the test-module reference oracle's bare
      unwraps; (5) tests: negative-in-run (L526 pin), negative-first-row (L537 pin),
      null-position via the FULL `load_deletes` production path, null-file-path direct-parse
      pin, happy-path control including boundary pos=0 (over-broaden direction). Named
      divergence to record in-test: Java also caps positions at MAX_POSITION
      (0x7FFF_FFFE_8000_0000, roaring key-space); Rust RoaringTreemap takes full u64 — only
      the negative bound is ported.
    - Builder outcome (2026-07-10, pre-critic): LANDED as planned — per-site
      `checked_delete_position` (both insert branches), split null let-else guards naming
      the delete file + column, `delete_file_path` threaded, test-oracle unwraps fixed.
      5 new tests (2 negative-site pins via the FULL `load_deletes` parquet path, null-pos
      full-path, null-file-path direct-parse, pos=0 boundary control), 5 mutations ALL
      independently RED (mutation 2's failure output showed the exact corruption:
      RoaringTreemap<[18446744073709551611]>). Gate green: typos/fmt/clippy -D warnings/
      lib 2745×2 (+5 over the A1 baseline 2740); `cargo test -p iceberg-datafusion` unit+
      integration green (the one doc-test FAIL is a pre-existing `#[tokio::main]`
      rt-multi-thread feature-unification artifact of `-p` isolation, untouched crate).
      Flagged, not fixed (scope): L298 `task.equality_ids.clone().unwrap()` (production
      bare unwrap, eq-delete column — not a position/path column); Java's MAX_POSITION
      upper bound not mirrored (named in doc + test comments).
- [x] **A2 — Fixed/Binary single-value JSON** (BUG-004/OTH-007): implement both `todo!()` arms
  per Java `SingleValueParser`/spec Appendix D; verify emit case vs Java base16 (possible
  two-sided interop bug); Fixed length enforcement.
  CLOSED: ladder SHIP, critic mutations 7/7, zero `todo!()` left in spec/values/ — details in
  the builder notes below. Orchestrator notes: the round-trip test alone CANNOT catch an emit
  case flip (parse is case-insensitive per Java) — the exact-emit-string test is the sole
  case pin; round-2 remediation fired on the harness out-of-scope matcher false positive
  (tests.rs vs the `spec/values/` directory allow entry) with an EMPTY issue list —
  verification-only round, 2 mutations re-proven RED, no code changed after round 0.
    - Builder outcome (2026-07-10, pre-critic): CONFIRMED two-sided — the old emit catch-all
      was `{x:x}` (lowercase AND unpadded: 0x0A → "a", undecodable by Java's strict
      `BaseEncoding.base16()`). Both `try_from_json` arms implemented (mixed-case accept per
      Java `toUpperCase(Locale.ROOT)`, SingleValueParser.java L169/L175; Fixed pre-decode
      string-length == 2·L check per L160-167); emit replaced with explicit Fixed
      (length-enforced per L331-337) + Binary arms, UPPERCASE `{b:02X}`; other
      (type, Binary) combos now fall to the DataInvalid catch-all (was: silently hex-encoded
      under any type). 6 new tests incl. the crown-jewel Java-written schema-with-defaults
      deserialization (previously PANICKED via `SerdeNestedField`); 7 mutations ALL
      independently RED (error-arm, case-flip, pad-drop, parse+emit length-check drops,
      odd-length accept, non-hex-as-0, over-broadened/inverted length guard — both
      directions per testing.md). Java fixtures from `TestSingleValueParser` L53-54,
      L117-123. Gate green: typos/fmt/clippy -D warnings/lib 2751×2 (+6 over A3's 2745).
      Flagged, not fixed (scope): `SerdeNestedField→NestedField` swallows parse errors via
      `.ok()` (malformed default hex now yields default=None silently, panic before —
      pre-existing seam behavior for all types); `hex_str_to_bytes` duplicates
      `expr/expression_parser.rs::hex_to_bytes` (out-of-scope file, dedup deferred);
      interop round-trip vs a live Java oracle deferred (🟡 done-bar).
    - Remediation r1 (2026-07-10): critic issue list parsed EMPTY and no report file found —
      instead of guessing, independently re-verified the increment: all four hex arms
      bytecode-confirmed against the 1.10.0 `SingleValueParser.class` line table (fromJson
      FIXED length-precheck + `toUpperCase(Locale.ROOT)` decode at offsets 576-643, BINARY
      644-678; toJson FIXED `remaining()==length()` check + `base16().encode` at 455-520,
      BINARY 523-552); 3 spot mutations re-run RED (emit case-flip, emit + parse Fixed
      length-check drops) with byte-identical restore; full gate re-run green (lib 2751×2).
      No code changes this round.
    - Remediation r2 (2026-07-10): issue list parsed EMPTY again; searched scratchpad +
      task/ — no critic report file exists anywhere. Self-critique of the accumulated diff
      found no open defect (no `todo!()` remains; non-Fixed/Binary `(type, Binary)` combos
      fall to the DataInvalid catch-all; `is_multiple_of` is within MSRV 1.92). 2 mutations
      re-run RED (emit case-flip → `json_binary_fixed_emit_uppercase_padded_java_compatible`
      RED; parse-side Fixed length check disabled via `if false &&` →
      `json_fixed_length_mismatch_is_data_invalid` RED), restored byte-identical (`cmp`
      verified). Full gate re-run green (typos/fmt/clippy -D warnings/lib 2751×2). No code
      changes this round.
- [x] **A4 — StrictMetricsEvaluator absent-NaN inversion** (found by our G4): absent NaN
  counts ⇒ CANNOT contain, matching Java cell-by-cell; over-loosening pin required; close the
  ENGINE_CONTRACT §9 open item in the same change.
    - Outcome (2026-07-10): `may_contain_nan` absent arm flipped to CANNOT (Java
      `canContainNaNs` 1.10.0 L483-486, jar-bytecode-verified: absent map/key ⇒ `iconst_0`);
      the Java `gtEq` NaN-lower-bound guard (L285-291, bytecode offsets 93-105) that the
      loosening makes REACHABLE ported in the same change — `Datum` orders NaN largest
      (`total_cmp`), so without it `NaN >= x` would wrongly prove ROWS_MUST_MATCH (the
      over-claim/data-loss direction). Every helper consumer matched cell-by-cell vs the Java
      visitor (lt/ltEq/gt/gtEq/eq/in consult the pair guard; notEq/notIn/isNaN/notNaN use the
      containsOnly helpers — all match). 5 new tests: crown-jewel int-column provable sweep
      (RED pre-fix), eq+in absent-arm consumers, over-loosening guard across all 5 consumers
      (nan_count>0 + bounds that would otherwise prove), float-absent Java-verdict pin
      (MUST_MATCH, bytecode-provenance), NaN-poisoned-bounds never-prove pin. 3 mutations RED
      (absent-arm revert / nan>0-arm drop / gtEq-guard drop), byte-identical restore.
      NAMED findings (recorded, NOT fixed here): (1) `may_contain_null` diverges from Java
      `canContainNulls` in the map-present-key-ABSENT case (Java: cannot; Rust: may —
      conservative/under-fires; Rust's single HashMap cannot represent Java's null-map vs
      empty-map split); (2) Rust has NO `isNestedColumn` short-circuit (Java returns
      MIGHT_NOT_MATCH for nested columns in every arm). §9 bullet closed. Deferred: the
      cross-engine metrics-decided full-match interop sweep (done-bar 🟡).
- [x] **A5 — bundle close**: independent SEPMO bundle Critic over `main..HEAD` → on CONVERGED
  flip this section, push, PR body to scratchpad.
  - Outcome (2026-07-10): bundle Critic **CONVERGED**, ZERO findings, NO units parked. Gate
    re-run in full by the Critic (typos/fmt/clippy/lib 2756/matrix-anchors/agent-artifacts);
    4/4 cross-unit mutations re-proven RED with byte-identical restores; Java citations for
    all four units re-verified against the 1.10.0 reference (incl. confirming `greater_than`
    already had its NaN guard on main — A4 correctly added only the missing gtEq one);
    behavioral-break sweep ruled the A2 emit change and A1 rejections corrections TOWARD the
    Java-written format, not format breaks (no in-tree dependents of the old behavior). Two
    accepted LOWs: the narrow pre-existing-in-kind serialize `.expect` surface in
    datatypes.rs:691 for malformed in-memory defaults (the flagged `.ok()` seam's sibling —
    future unit), and cosmetic `bind().unwrap()` in A4 test helpers matching module
    convention. Pushed `fix/audit-tier1-2026-07`; PR body at scratchpad
    `pr-body-audit-tier1-2026-07.md`. Tier 3 companion branch `infra/audit-ops-2026-07`
    (Critic CONVERGED after 2 MEDIUM prose corrections) pushed earlier the same day.

## ACTIVE UNIT (2026-07-09): OVERNIGHT Mode B bundle — G1→G4, one branch, one PR

User-directed 2026-07-09 ("run G1 to G4 in sequential groups without needing a PR for each") —
**Mode B** per [pr-per-work-cycle]: one bundle branch `parity/overnight-2026-07-09`, four
SEQUENTIAL parity-increment ladders (each: builder → tailored Opus critic, mutation-gated →
independent gate → bounded remediation), the orchestrator gates+commits after each unit, then ONE
final independent SEPMO bundle Critic over the whole branch diff; push on CONVERGED; single PR
for the user in the morning. Execution order **G1 → G2 → G4 → G3** (G3 last so the nightly
workflow enumerates any interop suites G4 adds). Statuses live ONLY in the GAP_MATRIX.

- [x] **G1. CDC row-level changelog** (queue item 2; rows R122/R123 named residue) —
      `ChangelogOperation::{UpdateBefore, UpdateAfter}` + handling ranges that carry row-level
      DELETE manifests (today: `FeatureUnsupported`, matching Java's data-file changelog).
      JAVA-FIRST scoping is mandatory: decode what 1.10.0 CORE (`BaseIncrementalChangelogScan`)
      actually defines vs what lives Spark-side (`ChangelogIterator` net-change pairing is NOT
      core) — parity claims only for the core surface; anything beyond is engine-first and
      labeled so (DML-foundation direction). Done bar: partial (interop slice may defer).
    - Builder plan (2026-07-09, bytecode-audited): 1.10.0 core REJECTS every delete-manifest
      range (`javap` offsets 86–95) and never constructs `BaseDeletedRowsScanTask` — so
      row-level acceptance is ENGINE-FIRST behind an opt-in builder flag
      (`with_row_level_deletes`), default = exact Java rejection surface. Port the api
      taxonomy as core parity: `ChangelogOperation` gains `UpdateBefore`/`UpdateAfter`
      (declared, never emitted by the planner — pairing is Spark-side, DEFERRED);
      `ChangelogScanTask` gains `kind` (AddedRows/DeletedDataFile/DeletedRows, operation()
      derived) + `added_deletes`/`existing_deletes`. Row-level planning per snapshot: split
      its delete manifests into added-in-S vs pre-existing `DeleteFileIndex`es; own-added
      data entries → AddedRows (with added deletes)/DeletedDataFile (with existing deletes);
      live NOT-added-in-S data files hit by added deletes → DeletedRows (added+existing
      split). Tests: crown-jewel MoR chain mirroring the `DeletedDataFileScanTask` javadoc
      example, added-vs-preexisting split, same-snapshot fold, pure-append control,
      replace-consumes-no-ordinal, rejection unchanged. Arrow read: no core-defined
      semantics (reading is engine-side) — AddedRows/DeletedDataFile tasks readable via the
      existing MoR `FileScanTask.deletes` machinery; DeletedRows projection deferred.
      Outcome (2026-07-09): LANDED as planned — `scan/task.rs` taxonomy (breaking:
      `ChangelogScanTask.operation` field → `kind`, `operation()` now derived; 2 new enum
      variants break downstream exhaustive matches), `scan/incremental.rs` planner
      (opt-in row-level mode; default path output-identical, guard mutation-proven both
      ways), 6 new/extended tests + 6 targeted mutations ALL RED
      (guard-disable / added-existing-swap / fold-drop / ordinal-reverse / kind-swap /
      commit-misstamp), R123 residue re-written (matrix gate green, 71 rows), R122
      untouched (no row-level residue lives there), scan/map.md refreshed. Deferred:
      interop slice (Java oracle for the row-level mode is meaningless — 1.10.0 core
      cannot plan those ranges; the existing data-file changelog interop stands as the
      control), DeletedRows Arrow projection (engine-side), UPDATE_BEFORE/UPDATE_AFTER
      pairing (Spark-side, not core parity).
- [x] **G2. Reconciliation-by-refresh** (R157 residue; `BaseMetastoreTableOperations.
      checkCommitStatus` / `CommitStatus` SUCCESS·FAILURE·UNKNOWN) — on `CommitStateUnknown`,
      re-read the catalog with bounded retries and decide landed (⇒ success) / absent (⇒ real
      failure, re-thrown per Java) / still-unknown (⇒ surface unknown). Mock tests for all three
      outcomes; the credentialed real-catalog slice stays with queue item 6.
      Outcome (2026-07-09): LANDED with one JAVA-FIRST rescope — the brief's "absent ⇒ re-thrown
      CommitFailed" is NOT 1.10.0 production behavior: the only production callers (Glue L174,
      DynamoDb L136) use the NON-strict `checkCommitStatus`, which converts strict-FAILURE ⇒
      UNKNOWN (bytecode offsets 11-34; `checkCommitStatusStrict` has zero non-test callers)
      because a pending in-flight request may still land after the check — declaring failure and
      re-running is the double-commit corruption class. Shipped: `transaction/commit_status.rs`
      (strict classifier, `commit.status-check.*` knobs with Java names/defaults, n+1 attempts,
      2.0-factor clamped backoff) + `Transaction::reconcile_unknown_commit_outcome` (non-strict
      conversion at the catalog-agnostic seam; snapshot-id evidence searched in the reloaded
      snapshot SET — history-tolerant to concurrent writers). 11 new/updated tests (crown jewel
      reconciles-to-success-without-reapply; buried-under-concurrent-writer; absent ⇒ unknown
      never success/retry; bounded-by-property; CommitFailed-control never reconciles;
      metadata-only skip; invalid-knob surfaces unknown; 4 unit pins) + 7 mutations ALL RED.
      Named divergences (matrix cell + module docs): snapshot-id evidence vs Java's
      metadata-location; metadata-only commits not reconciled; REST/SQL unknowns also reconciled
      (Java's REST/JDBC ops never do — strictly outcome-improving, read-only). R157 stays 🟡
      (credentialed slice remains); ENGINE_CONTRACT §8 manual reconciliation downgraded to the
      two residual cases.
- [x] **G4. ENGINE_CONTRACT §5 DRAFT→NORMATIVE** (queue item 4) — verify the isolation-level →
      validation table against Java 1.10.0 `SparkWrite`/`SparkCopyOnWriteOperation`/
      `SparkPositionDeltaWrite` (bytecode where jars exist, else the reference-checkout source —
      cite which); one interop conflict scenario per cell; + the owed non-identity
      DeleteFilter-equivalence test.
      Outcome (2026-07-09): §5 flipped NORMATIVE — every cell verified against the
      `apache-iceberg-1.10.0` SOURCE (Spark jars absent from `~/.m2`; oracle form cited per
      cell; api/core surfaces additionally javap-verified). TWO cells CORRECTED: (1) MoR DELETE
      does NOT enable `validate_deleted_files` (UPDATE/MERGE-only, `SparkPositionDeltaWrite`
      L251-254) — the draft prescribed it; (2) `case_sensitive` is NOT part of the Java base
      recipe (neither Spark writer calls it — engine policy). Base clarified: MoR
      `validate_data_files_exist` is unconditional (all commands, both isolation levels, L243);
      scan==null ⇒ NO validation; static overwrite-by-filter rows ADDED (`OverwriteByFilter`).
      Per-cell covering scenarios cited (C1-C5 interop arc + named unit tests); NEW
      `engine_contract_isolation_recipes.rs` pins the serializable-vs-snapshot distinction
      behaviorally for BOTH modes (snapshot leg COMMITS + post-commit live set; serializable leg
      REJECTS naming the validation; 3 recipe mutations RED). Owed non-identity DeleteFilter
      test LANDED (`test_engine_deletefilter_nonidentity_partition_equivalence`, offline
      truncate[10](id) pos+eq deletes, production-mutation RED). §9 R157 bullet un-staled
      (reconciliation-by-refresh landed G2). No matrix row touched.
      Remediation 2 (2026-07-09): the unit-only residue CLOSED — NEW cross-engine suite
      `interop_s5_isolation_conflict.rs` + `S5IsolationOracle` + `run-interop-s5-isolation.sh`
      covers the three formerly unit-level cells (COW/snapshot deletes, dynamic-overwrite/
      snapshot, static overwrite-by-filter snapshot+serializable): 8 scenarios (4 REJECT +
      4 ACCEPT guards), BOTH directions green + sabotage fail-closed on the local Java 11 run;
      4 recipe mutations RED (each cell's isolation-distinguishing validation dropped ⇒ GEN
      self-check fails). FOUND + NAMED (out of increment file scope, ENGINE_CONTRACT §9 open
      item): Rust `StrictMetricsEvaluator::may_contain_nan` treats ABSENT nan counts as
      may-contain-NaN (Java `canContainNaNs` 1.10.0 L483-486: absent ⇒ CANNOT), so strict
      inequalities never prove a full match on non-float columns —
      `overwrite_by_row_filter`/`DeleteFiles`-by-filter rejects ("some, but not all, rows
      match") files Java deletes cleanly; the serializable by-filter interop cell therefore
      runs partition-scoped (`category = "a"`) to keep `validate_no_conflicting_data`
      load-bearing. Follow-up: fix `expr/visitors/strict_metrics_evaluator.rs` L105-111 +
      an interop pin on a metrics-decided full-match sweep.
- [x] **G3. Nightly interop CI** (queue item 5) — scheduled workflow running the
      `dev/java-interop/` suites unprompted (cron precedent: audit/codeql/stale.yml); enumerate
      suites, doc the runner requirements (Java/protoc/docker), local one-shot proof of the
      entry point; the "runs unprompted" proof is next night's run.
      Outcome (2026-07-10): LANDED — `scripts/run_interop_suites.sh` (dynamic glob discovery,
      floor 48 with ratchet-on-add rule, prereq HARD-FAIL never-skip, continue-across-suites
      per-suite PASS/FAIL summary + step summary, `--only` local subset flag that logs every
      exclusion, `--selftest` battery), `make interop`/`interop-selftest`,
      `.github/workflows/nightly_interop.yml` (cron 06:43 UTC + workflow_dispatch; apt JDK 11 +
      `/opt/maven` symlink because all 48 suites default to those paths — 47 hardcode them
      outright, only `run-interop-aggregate.sh` reads `$MVN`/`$JAVA_HOME` — and must not be
      modified; online `~/.m2` priming because 47 of 48 suites run `mvn -o` (only
      `run-interop-scan-exec.sh` is online); full set only — no subset flag or env hook
      reachable from the YAML), map.md/README rows. Proofs: selftest 9/9 green + 7 driver
      mutations RED (exit-on-fail / floor / prereq / exclusion-log / empty-`--only` /
      empty-run-set / fake-prereq-wiring guards each turn a case red); real-dir battery —
      planted failing suite ⇒ exit 1 with the other suite still run+reported, renamed suite ⇒
      floor error before running anything, PATH-without-cargo + void-mvn ⇒ prereq hard-fail,
      YAML safe_load green + broken-copy red (non-vacuous); GREEN real-suite subset runs
      exit 0, 48 discovered. Remediation R1 (2026-07-10; critic report unrecoverable ⇒
      self-audit): (1) `--only ""` silently ran the FULL set (bounded request became
      unbounded — reproduced live) ⇒ parse-time hard-fail + selftest ST7; (2) a zero-suite
      run greened ("0 passed, 0 failed" ⇒ exit 0, reachable via the floor-0 test hooks) ⇒
      empty-run-set guard in `run_suites` + ST8; (3) the selftest was NOT hermetic (needed a
      real `/opt/maven` + JDK 11 on the machine) ⇒ fake prereqs wired through `drive()`,
      ST3 now isolates ONE missing prereq per case, wiring mutation-proven (6 cases red when
      the fake mvn path is broken); (4) `--help` used a hardcoded `sed '19,66p'` line range
      that drifts on any header edit ⇒ marker-based awk; (5) corrected wrong counts shipped
      in 5 places (was "29 hardcode / 19 offline"; measured truth: 48 default, 47 hardcode
      outright, 47/48 offline). NAMED RESIDUE: the
      "runs unprompted" proof is inherently NEXT night's live run (cron fires only once this
      file is on the default branch); the CI-runner provisioning (apt/symlink/m2-priming +
      the 350-min job bound vs the full 48-suite wall time) is NOT locally verifiable — first
      nightly is the proof. Deferred: `run.sh` + `run-inspection-manifests.sh` (outside the
      `run-interop-*.sh` glob, named in map.md/README); no log artifact upload (step summary
      only — no pinned upload-artifact action precedent in this repo). Remediation R2
      (2026-07-10; critic verdict SHIP — 5/5 mutations caught, zero bugs/over-claims; closed
      its one named test-strength nit): ST1's failing fake sorted LAST, so a
      bookkeeping-clean abort-on-first-failure mutation greened the whole battery 9/9
      (reproduced live — worse than the critic's own summary-needle-caught variant); renamed
      it `run-interop-aa-fail.sh` (sorts FIRST, before both passers), so the a/b `.ran`
      marker check now pins continue-AFTER-failure directly, independent of summary wording;
      the same mutation goes RED at 2 checks post-fix, clean battery 9/9 green. The critic's
      two blind-spot claims were resolution:refuted by its own probes (bash>=4.4 empty-array
      expansion; an independent sort-first continue-across probe against production).
    - Builder plan (2026-07-10, live-audited): 48 `run-interop-*.sh` suites exist (the brief
      said ~31 — floor set to the LIVE count 48); 29 hardcode `/opt/maven/bin/mvn` +
      `JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64` and 19 run `mvn -o` (offline ⇒ CI must
      prime `~/.m2`) [counts corrected in R1: those greps were style-narrow — truth is 48
      default / 47 hardcode outright / 47 of 48 offline], so the workflow installs apt
      `openjdk-11-jdk-headless` (noble carries
      11.0.31) + `maven` and symlinks `/opt/maven` rather than setup-java (the suites must
      not be modified). Deliverables: `scripts/run_interop_suites.sh` (dynamic glob discovery
      + floor 48 + hard-fail prereqs + continue-across-suites + per-suite PASS/FAIL summary +
      step summary + `--only` LOCAL subset flag that logs exclusions + `--selftest` sabotage
      battery), `make interop`/`interop-selftest`, `.github/workflows/nightly_interop.yml`
      (cron + workflow_dispatch, full set only — no subset flag reachable), map.md/README
      rows, local green subset proof + sabotage battery RED proofs. No matrix row touched
      (infra; no capability status changes).
- [x] **G5. Bundle close** — DONE 2026-07-10: independent SEPMO bundle Critic (fresh context,
      Opus) over `main..HEAD` **CONVERGED**, zero HIGH/MEDIUM findings ("Recommendation:
      push"). Cross-unit checks all clean: G1's breaking `ChangelogScanTask` change has ZERO
      external consumers (workspace-wide grep + build); G2's reconciliation composes correctly
      with the #144 unknown-kind retry gate (absent ⇒ original error, Java non-strict); all 7
      spot-checked §5 citations resolve; G3's floor (48) matches the live suite count incl.
      G4's new suite; todo notes accurate. 3 cross-unit mutations re-proven RED. 2 LOWs
      accepted (selftest count understated 9→10; interop coverage disclosed as
      claim-of-existence pending Java/Maven + first nightly). Pushed; merge is the user's.
      NO groups parked — the contingency was never needed.

CONTINGENCY (unattended): if a group's ladder cannot converge (workflow remediation exhausted +
one orchestrator remediation), park its work on `parity/overnight-parked-G<n>`, reset the bundle
branch to the last good unit commit (own unpushed branch; work preserved on the parked branch),
continue the chain, and report the parked group in the morning. Gate note (2026-07-08): the
typos step excludes the two untracked scratch briefs (`.typos.toml` decision still the user's).

## DONE 2026-07-08 (merged #144): queue item 1 — commit-outcome taxonomy (row R157)

User-directed 2026-07-08 ("proceed with your recommendation"). One PR, branch
`parity/commit-state-unknown`. Ladder: parity-increment workflow (builder → tailored adversarial
critic, mutation-gated → verification gate → bounded remediation), then the independent SEPMO
Critic before push. Status flips live ONLY in the GAP_MATRIX (row R157).

- [x] **1. Unknown-outcome error class** — `ErrorKind::CommitStateUnknown` (`error.rs`) +
      `Transaction::commit`'s gate refuses the KIND regardless of the `retryable` flag (Java
      1.10.0 bytecode: `onlyRetryOn(CommitFailedException.class)` + dedicated
      `CommitStateUnknownException` rethrow ahead of the cleanup catch). Gate mutation-proven
      BOTH directions (flag-only gate + unknown-retried gate each turn a pin red).
- [x] **2. Sent-vs-unsent transport classification** — REST (`query_catalog_for_commit`
      transport split + 500/502/503/504 ⇒ unknown + 200-with-lost-response ⇒ unknown, tables
      AND views), SQL (`from_sqlx_commit_error`: Io/Protocol/WorkerCrashed ⇒ unknown; CAS
      conflict stays retryable; NOTE — the previously-DISCARDED SQL-transaction `COMMIT` error
      now propagates for all statements), Glue + S3 Tables (`SdkError` dispatch classification
      + `InternalService`/`OperationTimeout`/`InternalServerError` ⇒ unknown;
      `ConcurrentModification`/`Conflict` stay retryable). REST/SQL/Glue classifiers
      mutation-proven.
- [x] **3. Mock-catalog tests** — crown jewel (`transaction/mod.rs`): durably-landed-but-
      unacknowledged commit against a real in-memory catalog ⇒ surfaced intact, exactly 1
      `update_table` call, exactly 1 snapshot, file appears ONCE, manifests NOT cleaned up;
      + flag-defense test (unknown-with-retryable-flag still not retried); + Error API
      kind-survives-wrapping test; existing retryable/terminal tests unchanged (control pins).
- [x] **4. Rider: crates/ citation migration** — 26 bare-citation sites migrated across ~24
      files (each target row VERIFIED by cell content — drift was NOT uniformly +2: e.g.
      93/94/95→R105/R106/R107, 100→R100, 129→R129, 152→R152, builder-flips 134/135→R146
      merged row); `crates` added to check-4's pathspec AND the asserted scan-target list;
      anchor grep made case-sensitive on the `R` (test prose "rows r1" false-positive);
      red-proof: planted a dead `R9999` anchor citation in a crates/ comment ⇒ gate exits
      non-zero; removed ⇒ green.
- [x] **5. Matrix + docs** — row R157 flipped ❌→🟡 (2026-07-08; residue named:
      reconciliation-by-refresh `checkCommitStatus` NOT ported, credentialed slice with queue
      item 6); ENGINE_CONTRACT §8 rewritten around catching `ErrorKind::CommitStateUnknown`
      (mitigation (a)-(c) stands until reconciliation lands); `make check-matrix-anchors`
      green (71 rows).
- [x] **R1. Remediation rounds 1–2 (2026-07-08)** — (a) typos: round 1 reworded 7
      typos-cli-1.47.2 false-positives in the untracked scratch briefs
      `task/a1-cow-partition-brief.md` + `task/h7-dml-streaming-scope.md`; round 2 REVERTED
      that as out-of-scope (user scratch, not increment files). RESOLUTION: the commit gate
      runs `typos` over TRACKED files (exactly what CI certifies on a clean checkout — the
      untracked briefs never enter any commit); a `.typos.toml` exclude vs rewording the
      briefs is the user's call (flagged in the PR); (b) the REST 200-with-unparsable-body
      OK arms are now PINNED:
      `test_update_table_200_unparsable_body_maps_to_commit_state_unknown` (full
      `Transaction::commit` stack, POST `expect(1)`) +
      `test_update_view_200_unparsable_body_maps_to_commit_state_unknown` — both
      mutation-proven RED on OK-arm kind → `Unexpected`, green restored (REST lib 64→66).
- [x] **6. Gate + independent Critic → push** — DONE 2026-07-08: gate green in ONE chain with
      commit 4bffcc82 (typos·fmt·clippy -D warnings·lib tests 2706/66/74/23/26·both integrity
      gates); independent SEPMO Critic (fresh context, Opus) **CONVERGED** — bytecode-verified
      the Java contract (`onlyRetryOn(CommitFailedException.class)`; unknown rethrown ahead of
      cleanup; 409→CommitFailed, 500/502/503/504→unknown), 6/6 mutations RED, ALL rider
      citations content-verified, zero blocking findings (2 LOWs accepted: REST-test bare
      unwraps house-consistent; 200-unparsable-body arm is a disclosed safer-than-Java
      extension). Pushed; merge is the user's.

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
- [x] **4. Verify + Critic** — DONE 2026-07-01: 2-auditor fan-out (287 claims; found the 4th
      drift wave + 8 hardening findings, all fixed) → independent Critic CHANGES REQUIRED
      (1 MEDIUM: check-4 git-grep false-green — fixed c195b616) → re-review CONVERGED. Pushed.

NAMED FOLLOW-UP (Critic LOW, 2026-07-01 — not this PR): ~20 stale bare `GAP_MATRIX row N`
citations live in `crates/` source/test comments (e.g. `scan/task_group.rs`,
`tests/interop_scan_plan.rs`, `aggregate_evaluator.rs`), most drifted under current numbering —
migrate them to `row R<n>` form and add `crates/` to the anchor checker's scan pathspec (touches
Rust files, so it rides a code PR, not this docs/CI one).

---

## 2026-07-10 — Zero-width / oversized bucket-truncate transforms: reject at parse, never panic at apply (BUG-001/SAF-001/BUG-013)

Plan (delegated BUILDER; done-bar 🟡 unit-tested, interop deferred):

- [x] Java contract (bytecode-verified, 1.10.0 jar): `Bucket.get(int)` bytecode `<= 0` reject-branch + msg
  `"Invalid number of buckets: %s (must be > 0)"` (Bucket.java:41-42); `Truncate.get(int)` bytecode
  `<= 0` reject-branch + msg `"Invalid truncate width: %s (must be > 0)"` (Truncate.java:42);
  `Transforms.fromString` parses via `Integer.parseInt` (Transforms.java:39,45) so values >
  `Integer.MAX_VALUE` are unrepresentable ⇒ parity bound is `1..=i32::MAX` for both.
- [x] `spec/transform.rs`: add `Transform::validate()` (Java-precondition messages); call it from
  `FromStr` bucket/truncate arms (covers serde/Deserialize ⇒ table-metadata JSON, partition specs,
  sort orders). Tests: reject 0 + `i32::MAX as u64 + 1` on both; boundary-legal 1 and `i32::MAX`
  accepted (over-broaden pin); JSON serde rejection.
- [x] `transform/bucket.rs`: `Bucket::new(u32) -> Result<Self>`, store `mod_n: i32` via zero-check +
  `i32::try_from` (drops the lossy `as i32` in `bucket_n` — wrong-bucket wrap for N > i32::MAX).
- [x] `transform/truncate.rs`: `Truncate::new(u32) -> Result<Self>` with the same 1..=i32::MAX guard
  (division/modulo-by-zero in `truncate_i32/i64/decimal_i128` becomes unreachable).
- [x] `transform/mod.rs`: `create_transform_function` propagates the fallible constructors (`?`) —
  the apply-door defense; direct `Transform::Bucket(0)` (public enum payload, not blockable) now
  errors instead of panicking.
- [x] `spec/partition.rs`: `transform.validate()` in both builders (`add_partition_field_internal`
  unbound + `add_unbound_field` bound) — the programmatic route Java rejects at construction.
  Crown-jewel test: minimal V2 TableMetadata JSON, control (bucket[16]) parses, sabotage
  (bucket[0] / truncate[0]) fails with `ErrorKind::DataInvalid` via the production
  `serde_json → Error::from` conversion (`read_from` path).
- [x] Gate: typos + fmt + clippy + `cargo test -p iceberg --lib` ×2; mutation-check the guards both
  directions (disable ⇒ rejection tests red; over-broaden ⇒ boundary-legal tests red).
- Outcome (2026-07-10): landed. Parse door = `Transform::validate()` called from `FromStr`
  (covers Deserialize -> table metadata / partition specs / sort orders); apply door =
  fallible `Bucket::new` / `Truncate::new` (independent guards) propagated by
  `create_transform_function`; builder door = `validate()` in both partition-spec builders.
  `bucket_n` now stores a construction-proven `i32` (lossy `as i32` removed). 16 new tests incl.
  the crown-jewel TableMetadata fixture (control-first; Java-text pinned at the PartitionSpec
  serde door - the untagged TableMetadataEnum swallows inner messages). Mutations M1 (parse bound
  off: 6 tests RED, apply door stayed green - layer independence), M2 (apply guards off +
  unchecked cast restored: 8 tests RED incl. the project-panic pin), M3 (over-broadened bounds:
  3 boundary-legal tests RED) - all restored, full gate green (typos/fmt/clippy + lib 2740 x2).
  Done-bar partial: interop (Java-side cross-validation of the rejection) deferred; GAP_MATRIX
  untouched (hardening, no capability row).



**Explicitly NOT decided:** the "platform cut line" through the GAP_MATRIX (which rows block the
user's trading platform vs continuous-parity backlog, incl. re-ordering maintenance actions ahead of
Phase-4 format exotica) was proposed but is an **open user decision — do not assume it.**
  _RESOLVED-AS-TABLED 2026-06-11: the user tabled the DataFusion/RePark direction and redirected
  the fork to near-full 1:1 Java parity — recorded in Roadmap.md (decision record item 5 + the
  re-sequenced headline areas). Originating narrative:
  [todo-archive/2026-06_ops-hardening.md](todo-archive/2026-06_ops-hardening.md)._


