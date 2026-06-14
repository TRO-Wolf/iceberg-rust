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

Ranked, highest-value first:

- [ ] **1. Write-action DATA-level interop FIRST** — `DeleteFiles` / `RowDelta` / `RewriteFiles`
      proven through real Java↔Rust byte-level round-trips (the merge-append data-level chain is the
      template). The write engine is feature-complete; the residue is *interop evidence*, not code.
- [ ] **2. Conflict-detection + `caseSensitive` builder surfaces** — the validation-builder family
      (`conflictDetectionFilter` / `validateNoConflicting*` wiring) and the `caseSensitive` toggles on
      the scan/expression builders, both newly-tracked rows from the re-audit.
- [ ] **3. Multi-spec writes** — writing under more than one partition spec in a single table's history.
- [ ] **4. `unknown` → geometry / geography types** — the V3 type-breadth follow-on.
- [ ] **5. `RewritePositionDeleteFiles` + the cheap maintenance wrappers** — the position-delete
      rewrite action plus the remaining thin `ActionsProvider` maintenance wrappers.
- [ ] **6. Glue / S3Tables views + encryption** — the credentialed real-catalog view surface (needs
      AWS creds, scheduled with the user) and the frontier-grade encryption work.
- [ ] **7. `BatchScan` / `ScanTaskGroup` + `ExpressionParser` JSON** — scan-completion (task-group /
      `planTasks` split planning) and the JSON expression (de)serializer.

See the 2026-06-13 GAP_MATRIX provenance block for the per-row status and residue of every item above.

## ACTIVE (2026-06-13, Opus builder): `caseSensitive(bool)` on the expression-binding write actions

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

## ACTIVE (2026-06-13, Opus builder): `DeleteFiles.deleteFromRowFilter(Expression)` delete-by-predicate

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
