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
- [ ] **AC·OO #2 — C3 RowDelta** (row 94). Richest: `validate_no_conflicting_data_files` +
      `_delete_files` + `validate_data_files_exist` — concurrent data AND delete-file conflicts.
- [ ] **AC·OO #3 — C2 DeleteFiles** (row 93). `validate_files_exist()` — concurrent REMOVAL of a file
      this delete targets (missing-path shape, not added-data).
- [ ] **AC·OO #4 — C5 RewriteFiles** (row 95). `validate_no_new_deletes_for_data_files` — a concurrent
      delete that would apply to the rewritten files (seq-preservation angle).

Wave 2 — multi-spec write interop (stretch):

- [ ] **AC·OO #5 — MS** merging actions (overwrite/replace/row-delta) under >1 partition spec; extend
      the Z2 `run-interop-multi-spec.sh` template (fast_append multi-spec already ✅).

Wave 3 — builder-surface flips (stretch, only if 1+2 beat estimates):

- [ ] **AC·OO #6 — BF** `case_sensitive` (row 134) + `delete_from_row_filter` (row 135) → interop-proven.

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
