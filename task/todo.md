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


> **Archival log.** Last pass: 2026-06-11 (size trigger — 1,381 lines; pass 2) → [todo-archive/](todo-archive/) (phase1/phase2/phase3 + 2026-06_ops-hardening). 16 `##` sections: 2 kept live, 14 archived (9 → phase2, 2 → phase1, 1 → phase3, 2 → ops-hardening); the platform-cut-line paragraph lifted into Carried-forward. Prior pass: 2026-06-09 (size trigger — 4,344 lines) → [todo-archive/](todo-archive/) (phase1/phase2/phase3). Completed-increment narratives moved verbatim; this file keeps the active sprint + open items + archive pointers. Procedure: [skills/compaction.md](../skills/compaction.md) §Todo Archival. Archives are not read by default.

## DONE (2026-06-11): Lessons compaction pass 2 (branch `docs/lessons-compaction-pass-2`, user-approved)

- Trigger: SIZE — 1,369 lines / 128 KB on the settled post-#24 main (trigger ~800 / 50 KB).
- Tally: **42 entries → 17 KEEP / 25 ARCHIVE / 6 rules PROMOTED** (the 25-entry archive set and
  all six promotion diffs are byte-identical to the version presented for approval pre-merges;
  the KEEP set absorbed every 2026-06-10/11 arc entry — current work feeding the platform plan).
- Archive: `task/lessons-archive/2026-06_phase2-completion.md` (+ archive map row). Promotions:
  2 → docs/testing.md, 2 → dev/java-interop/map.md#debug, 1 → transaction/map.md#debug,
  1 → CLAUDE.md. Conservation: 42 == 17 + 25, no duplicates. Active file 583 lines.

## DONE (2026-06-11): Todo archival pass 2 (branch `docs/todo-archival-pass-2`, stacked on the lessons pass)

- Trigger: SIZE — 1,381 lines on the settled post-#24 main (target < ~500).
- 16 `##` sections: 2 KEEP live (the lessons-pass record + the pointer section), 14 ARCHIVE
  verbatim — 9 → phase2.md (the write-engine arc ×2 incl. the union-merge bare-header artifact,
  the DV arc, the overnight plan + morning report, Arc F, E1, E2, the OverwriteFiles branch-A
  increment, the superseded RewriteManifests sketch), 2 → phase1.md (Arc G + the closed
  carried-forward items), 1 → phase3.md (readable_metrics interop), 2 → the NEW
  `2026-06_ops-hardening.md` (increment D + the hardening meta-sprint — meta work, deliberately
  not phase-filed; deviation documented in that file's header).
- LIFTED (the sanctioned carve-out): the platform-cut-line open decision → Carried-forward below.
- Conservation: every pre-pass `##` heading in exactly one place; no checkbox flipped; no
  paraphrase. typos clean.
- Stale-box audit (the rule's verify-before-deciding step): 8 unticked `[ ]` boxes ride the
  archived sections, ALL stale-done, verified — Arc E (merged PR #22), Arc-E Inc 1 (#22), Arc F
  (#23), Arc G (#24), the morning report (delivered via the wrapper-root file), increment C
  (archival pass 1, 2026-06-09, recorded in the archival log), E / E3 (inspection interop
  COMPLETE per the GAP_MATRIX row). None surfaced as live work; preserved unflipped in context.

## ACTIVE (2026-06-11): Near-full-parity direction — next arcs (planning record)

Directive (user, 2026-06-11): table DataFusion/RePark; run this fork's Roadmap to **almost the
full 1:1 Java replacement**. Sequencing in Roadmap.md "Headline gap AREAS" (handoff-aware:
judgment-heavy → frontier window before 2026-06-22; templated breadth → Opus).

- [ ] **Next arc proposal (awaiting user green-light): Phase-2/3 closeout** — multi-spec writes
      (producer per-spec manifests; unlocks the documented default-spec-only divergences),
      the constants-map increment (reverted 2026-06-08; known latent type bugs; gated on
      datafusion + integration read tests), `removeRows` apply-side, the `dv_seq >= data_seq`
      index-validation residue.
- [ ] **Then: maintenance actions** (`ExpireSnapshots` first — the GC-safety judgment increment).
- [ ] **Scheduled with the user:** real-catalog (Glue + S3 Tables) hardening — needs credentials.
- [ ] **Opus-queue (post-handoff or parallel):** data-level write-action interop paydown,
      cherrypick interop + `stageOnly`, ORC/Avro breadth, view ops, incremental-scan interop.
## DONE (2026-06-11): Multi-spec writes — producer per-spec grouping (BUILDER, Group A, wt-closeout)

Goal: lift the Rust `SnapshotProducer` from DEFAULT-SPEC-ONLY to Java-parity PER-SPEC manifest groups.

- [x] **Producer grouping (snapshot.rs):** `write_added_manifests`/`write_added_delete_manifests` group
  `added_data_files` / `added_delete_files` by `partition_spec_id` (helper `group_files_by_spec`, spec-id
  DESCENDING) and write one manifest per (content × spec) via `new_cluster_manifest_writer(spec, content)`
  (generalized to take content; the two existing callers pass `Data`). The explicit-data-seq (RewriteFiles)
  + V1 snapshot-id stamping paths preserved. Removed the now-dead default-spec `new_manifest_writer` (which
  also carried a bare `.unwrap()`).
- [x] **Validation lift (snapshot.rs):** `validate_added_data_files`/`validate_added_delete_files` now check
  spec EXISTENCE via `partition_type_for_added_file` with Java's exact "Cannot find partition spec %s for
  {data,delete} file: %s"; partition-value compat against the FILE's own spec.
- [x] **Summary ripple:** `summary()` passes each file's own spec via `file_partition_spec(file)` (Java
  `addedFile(spec(file.specId()), file)`), not the default — the changed-partition-summaries fix.
- [x] **Cherrypick conversion:** `test_cherrypick_multispec_replay_fails_loud` →
  `test_cherrypick_multispec_replay_produces_per_spec_manifest` (replay SUCCEEDS, manifest stamped spec 0,
  scan correct); module-doc note rewritten to the per-spec parity contract.
- [x] **Tests:** 6 producer tests in `snapshot::multispec_tests` (two-spec data + delete manifests,
  unknown-spec data + delete rejection, wrong-spec-type, cumulative totals) + the cherrypick conversion.
  Renamed `row_delta::test_row_delta_rejects_partition_spec_mismatch` →
  `test_row_delta_rejects_unknown_partition_spec` (stale default-spec assertion fixed).
- [x] **Mutations:** grouping-revert (default-spec-only) ⇒ all 4 grouping tests fail (`zip_eq` tuple-arity
  panic = partition corruption); validation-revert (default fallback) ⇒ all 3 unknown-spec tests fail
  (door message gone). Both restored from /tmp/wtA_snapshot_pre_mutation.rs.
- [x] **Docs:** GAP_MATRIX (multi-op row + cherrypick cell), transaction/map.md, lessons.

**Outcome:** Producer is Java-parity per-spec. Verification: typos clean, fmt clean, clippy `-D warnings`
clean (workspace ex-sqllogictest), `cargo test -p iceberg --lib` 1804 passed ×2 (was 1798 baseline +6 new
−1 renamed... net 1798→1804 = +6 producer +6 unchanged... 1798+6=1804), `iceberg-datafusion` lib+integration
9/9 + write-path insert tests green. PRE-EXISTING unrelated failure flagged: an `iceberg-datafusion` DOCTEST
(`table_provider_factory.rs:41`) fails to compile (`#[tokio::main]` multi_thread w/o `rt-multi-thread`) — not
touched by this increment. Deferred (flagged): WRITER-LAYER spec threading; `OverwriteFiles::validate_added_files`
default-spec (Java's `dataSpec()` rejects multi-spec there anyway); multi-spec Java↔Rust interop. No commit.

**REVIEWER PASS (Group A, 2026-06-11, wt-closeout).** Verdict: APPROVE with two added pins.
- **THE MISSING SUMMARY PIN (point 1) — confirmed gap, fixed.** The builder shipped NO test that fails
  CLEANLY under a summary-collector revert. The summary-revert mutation only crashed the 3 arity-differing
  manifest tests via a `partition_to_path` index-out-of-bounds PANIC (the "lucky" version) — a same-arity
  different-NAME multi-spec commit would silently render the WRONG `partitions.{path}` key with NO panic.
  Added `test_fast_append_multispec_partition_summary_keys_use_file_spec` (spec0=`identity(x)`, spec1=
  `identity(y)` via a same-arity rename; both files partition value 5): asserts `partitions.x=5` present
  (NOT `partitions.y=5`-only) AND `changed-partition-count=2` (the default-spec bug collapses both onto
  `y=5` ⇒ 1). Fails CLEANLY under the summary-revert (asserted, not panic); passes on fixed. Verified Java
  `SnapshotSummary.Builder.addedFile(spec(file.specId()), file)` → `updatePartitions` → `partitionToPath`
  uses the FILE's spec (1.10.0 bytecode).
- **V1 multi-spec (point 4) — probed, WORKS.** Added `test_v1_fast_append_two_specs_produces_per_spec_data_manifests`:
  a V1 two-spec DATA append produces one V1 manifest per spec (not fail-loud) — Java parity.
- **Mutations re-run (point 6):** grouping-revert ⇒ 3 manifest tests fail (zip_eq); validation-revert ⇒ 3
  unknown-spec tests fail (door message gone, deeper failure confirms defense-in-depth); cherrypick
  default-spec-stamp ⇒ conversion test fails (zip_eq) — pins per-spec, not just success. NEW reviewer
  mutation: `group_files_by_spec` file-LOSS (truncate to 1 group) ⇒ caught by the two-spec manifest tests
  (count + per-file presence). NOTE: `test_fast_append_multispec_cumulative_totals` does NOT catch file-loss
  — `added-data-files`/`total-data-files` come from `added_data_files` BEFORE grouping, so its docstring
  ("a dropped spec group would under-count") slightly overclaims; the manifest tests are the real loss guard.
- **Ordering (point 2) — FLAG for future interop (view NOT changed).** `snapshot_meta_view.rs` manifest sort
  tuple (L113) is `(content_rank, seq, min_seq, 6×counts)` and does NOT include `partition_spec_id`; the
  emitted manifest JSON also omits it. Two same-content/same-seq/same-counts manifests of DIFFERENT specs
  TIE on the whole tuple ⇒ array order falls back to manifest-LIST position (Rust spec-descending vs Java
  HashMap order). NO current interop fixture is multi-spec single-commit, so nothing is broken today; a
  FUTURE multi-spec interop fixture must either add spec id to the comparator tuple or assert the manifest
  SET (order-insensitively).
- **Pre-existing, untouched:** `validate_partition_value` has two near-duplicate messages (L843 "...not
  compatible WITH partition type" arity branch vs L859 "...not compatible partition type" per-field branch);
  both present in HEAD, the increment's test asserts the variant it triggers. Cosmetic; out of scope.
- Stale cherry_pick.rs banner comment (L1442 "fail-loud divergence") corrected to the converted contract.
  GAP_MATRIX/todo test count 6→8. Gate clean: typos, fmt, clippy -D warnings (workspace ex-sqllogictest),
  `cargo test -p iceberg --lib` 1806 ×2, `iceberg-datafusion` lib 80 + integration 9. Tree clean, no commit.

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
- Index: [todo-archive/map.md](todo-archive/map.md).
