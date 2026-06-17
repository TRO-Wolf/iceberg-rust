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

## ACTIVE (2026-06-13): Near-full-parity direction — open queue (ranked next-work)

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

- [ ] **1. Conflict-validation interop** — prove the `validateNoConflictingData` /
      `validateNewDeletes` / `validateDataFilesExist` family on real concurrent-commit Java↔Rust
      scenarios (Rust rejects exactly when Java rejects). Gates rows 91-95. **Start: OverwriteFiles**
      (the C1 active increment below), then DeleteFiles / RowDelta / ReplacePartitions / RewriteFiles.
- [ ] **2. Multi-spec write interop** — the merging actions (overwrite / replace / row-delta) under
      more than one partition spec; `fast_append` multi-spec is already ✅ (Z2 — the template).
- [ ] **3. Builder-surface interop flips** — `case_sensitive` (row 134) + `delete_from_row_filter`
      (row 135): code landed 2026-06-13, interop deferred → flip to proven.
- [ ] **4. `unknown` → geometry / geography types** — the V3 type-breadth follow-on.
- [ ] **5. `RewritePositionDeleteFiles` + the cheap `ActionsProvider` maintenance wrappers.**
- [ ] **6. `BatchScan` / `ScanTaskGroup` + `ExpressionParser` JSON** — scan-completion (task-group /
      `planTasks` split planning) and the JSON expression (de)serializer.
- [ ] **7. [FRONTIER — parked until Fable returns] Glue / S3Tables views + encryption** — the
      credentialed real-catalog view surface (needs AWS creds) and frontier-grade encryption. Held
      out of the Opus queue per the 2026-06-15 tier decision (Fable off limits).

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
- [ ] **G2 — `ComputePartitionStats` action + `UpdatePartitionStatistics` commit seam** (2h, LOW, offline)
      → **row 138 🟡→✅**; ActionsProvider `compute_partition_stats` FeatureUnsupported→real (7/5→8/4).
      Both halves exist + interop-proven (Z3/R2/R3): clone `update_statistics.rs` for the seam + `ComputeTableStats`
      for the action over `compute_and_write_stats_file`/`register_partition_stats_file`.
- [ ] **G3 — `SupportsNamespaces` partial property set/remove** (1.25h, LOW, offline) → SupportsNamespaces
      component ✅ (row 151 stays 🟡 until G4). ~25-line default `Catalog::update_namespace_properties`
      (overlap-reject + get→mutate→full-replace). Optionally fix the latent SQL full-replace bug the
      remove-test exposes (catalog.rs:612).
- [ ] **G4 — `ConvertEqualityDeleteFiles`** (3.5h, HIGH, offline + corroborating no-Spark interop) → with
      G3, **completes row 151 🟡→✅**. NEW write logic: per eq-delete, scan referenced data emitting ABSOLUTE
      `_pos`, materialize MATCHING positions, write pos-deletes stamped with the eq-delete's data-seq
      (`add_delete_file_with_sequence_number`), commit via the RewriteFiles 4-set. Corruption-class proof =
      Java MoR reads the table IDENTICALLY before (eq) / after (pos) conversion. NOT an ActionsProvider
      method (javap-confirmed). Capstone risk: absolute-pos, seq-stamping, applicability-scope,
      matching-vs-surviving inversion. Floor: if it lands 🟡, G1-G3 still delivered 2 ✅ flips + provider 8/4.

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
