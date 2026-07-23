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

# Lessons

Accumulated DO / DO NOT lessons. The operating manuals ([skills/](../skills/)) require reading this
file **in full at the start of every session**, and appending to it after **any** correction from
the user.

How to use it (see the manuals' §2):

- After any correction, append a **date-stamped** entry immediately.
- Write each as a concrete **DO** or **DO NOT** statement with the *why* and how to apply it.
- Supersede an outdated rule with a dated note (`_superseded YYYY-MM-DD: see ..._`) rather than
  editing the original in place.

> **Compaction log.** Last pass: 2026-06-13 (pass 5 — Wave-6/Wave-7 compaction, 959 lines;
> 1 KEEP-added / 17 ARCHIVE / 0 promoted) →
> [lessons-archive/2026-06_wave6-wave7.md](lessons-archive/2026-06_wave6-wave7.md). All 17 dated
> increment narratives (Wave-4 F2 → Wave-7 R3, incl. 8 nested REVIEWER sub-sections) archived
> verbatim; one new cross-cutting KEEP added (squash-merge content validation). Promotion candidate
> flagged in this pass's notes (the SKIP-false-green sabotage rule), user-approval-gated. Prior
> passes: 2026-06-12 (pass 4 — post-Wave-5 merge union, 4/25/2) →
> [lessons-archive/2026-06_wave5.md](lessons-archive/2026-06_wave5.md);
> 2026-06-12 (pass 3 — 14/47/3) →
> [lessons-archive/2026-06_wave3-wave4-overnight.md](lessons-archive/2026-06_wave3-wave4-overnight.md);
> 2026-06-11 (pass 2 — 17/25/6); 2026-06-09 (pass 1 — 31 promoted). Archives are not read by
> default — see [skills/compaction.md](../skills/compaction.md).

---

<!-- Newest entries at the bottom. Example shape:

### YYYY-MM-DD
- **DO** carry context on every fallible Rust call (`.with_context(...)` / `.expect("msg")`).
  *Why:* a bare `.unwrap()` panic gives the operator no cause from logs alone.
- **DO NOT** edit upstream crate files to land a fork feature when an additive module would do.
  *Why:* it makes the next upstream merge conflict-prone. Prefer additive changes.
-->

### 2026-06-13 — Prove a squash-merged branch's content landed before pruning it (squash-merge defeats `--is-ancestor`)

- **DO, before deleting a branch/worktree that was merged into the trunk via SQUASH-merge, prove its
  content landed with `git diff <branch-tip> <squash-commit>` — an EMPTY diff means every line is in
  the trunk and the branch is safe to prune.** *Why:* a squash-merge replays the branch's combined
  changes as ONE brand-new commit with NO parent link to the branch, so the branch tip is NOT an
  ancestor of the trunk. The usual safety checks then LIE: `git merge-base --is-ancestor <tip> main`
  returns false (non-zero), `git branch --merged` omits the branch, and `git cherry main <branch>`
  reports every commit as `+` (unmerged) — all three imply "you'd lose work by deleting this," when
  in fact the work is fully present under a different commit identity. The content diff is the truth;
  ancestry is not. (Came up 2026-06-13 pruning the merged Wave-6/Wave-7 worktrees: `--is-ancestor`
  flagged them as unmerged, but `git diff wt-tip <squash-sha>` was empty for each, confirming a safe
  prune.) *Apply:* for a squash-merged branch use the content diff, not ancestry, as the
  delete-safety gate; reserve `--is-ancestor` for true (non-squash) merges where the parent link
  exists. If the diff is NON-empty, the branch carries commits the squash did not capture — stop and
  reconcile before deleting.

### 2026-06-15 — When a stacked PR's base is squash-merged, rebase `--onto` to drop the now-redundant base commits

- **DO, when the base branch of a stacked PR gets SQUASH-merged into the trunk, recover the dependent
  branch with `git rebase --onto origin/main <old-base-tip> <dependent-branch>` — this replays ONLY
  the dependent branch's own commits and lets the base's now-redundant commits fall away.** *Why:* the
  squash lands the base PR's content as ONE new trunk commit with no parent link to the base branch
  (see the 2026-06-13 entry — ancestry now lies). The dependent branch still carries the base's
  ORIGINAL commits, so the same content lives under two unrelated histories; opening or merging the
  dependent PR against the trunk then throws add/add conflicts on every shared file. A plain `merge` or
  a normal re-push does NOT fix it — you must replay your commits onto the new trunk so the base
  commits drop out. *Apply:* (1) `git fetch`, then confirm the trunk truly carries the base's content —
  an EMPTY `git diff <base-branch> origin/main -- <base's paths>` (per the 2026-06-13 entry) — before
  trusting the squash; (2) `git rebase --onto origin/main <old-base-tip> <dependent-branch>` (the range
  `<old-base-tip>..<dependent-branch>` is exactly your own commits, so only they replay); (3) verify
  `HEAD^ == origin/main` and that the PR diff is only your files; (4) update the remote with
  `git push --force-with-lease` — the lease refuses if anyone else pushed since your fetch, and a
  force-push needs explicit user approval per [CLAUDE.md](../CLAUDE.md) *Absolute prohibitions*; (5)
  retarget the GitHub PR base to the trunk. (Came up 2026-06-15: PR-0 squash-merged as #52 while PR-1
  was stacked on it; PR-1 showed conflicts until rebased `--onto origin/main`, dropping PR-0's two
  commits and replaying only the one SKILL.md commit — a clean one-file diff with no conflicts.)
  *Corollary:* when two pending PRs touch DISJOINT files, branch each independently off the trunk
  instead of stacking — they merge in any order and never hit this trap.

### 2026-07-15 — Catalog publish must be all-or-nothing; a staged replace must not relocate the table

- **DO order catalog-pointer mutations AFTER every fallible step (read/validate first, insert last)
  under the one catalog lock.** `MemoryCatalog::register_table` (the `publish_create_table` default)
  inserted the pointer THEN read the metadata; a reload failure — staged metadata written through a
  FileIO the catalog cannot read — left a half-created table (`table_exists`=true, `load_table`
  errors) and broke `CREATE TABLE IF NOT EXISTS` retry. *Why:* the in-memory catalog's whole
  register/update body already runs under one lock, so reordering read-before-insert makes
  create-publish atomic at zero concurrency cost. Pin the *during-commit* failure explicitly (a
  publish whose reload fails), not just the pre-commit abort.
- **DO NOT bake a transient stage path into a replace table's `metadata().location()`.**
  `begin_replace` derived `"{existing_location}__staged_replace"` and never reset it, so every CREATE
  OR REPLACE relocated the table and COMPOUNDED the suffix
  (`orders__staged_replace__staged_replace…`), sending future writers to a drifted path. *Why:*
  staging isolation comes from deferring the catalog pointer swap until `commit`, NOT from a separate
  directory — keep `location()` equal to the stable existing/caller location; the new metadata gets a
  fresh version+UUID under it and only becomes current at publish. Never move already-written data
  (manifests carry absolute paths).
- **DO re-run `cargo fmt`/`clippy` against a feature-branch tip before building on it** — tip
  9280320b (the R158 commit) was committed without `make check` and failed both `cargo fmt --check`
  and `cargo clippy -D warnings` (a `collapsible_if` in `publish_replace_table`). A remediation unit
  that gates cleanly must normalize the pre-existing violations in the files it already touches and
  disclose it.

### 2026-07-23 — When a nightly cross-engine assertion fails UNREPRODUCIBLY, INSTRUMENT the upstream facts before "fixing"; do NOT paper over a same-file divergence with a fixture knob

Context: the `Nightly Interop` `scan-plan` D1 leg (`interop_scan_plan.rs`) had failed EVERY run since the
workflow's first (2026-07-11), reporting Rust splitting `big.parquet` into 8 sub-tasks vs Java 5 — the
split member keys `(basename,start,length)` ARE the row-group offsets. It could NOT be reproduced locally
(deterministically green), and the nightly uploads no artifacts + raw Actions logs need auth.

- **DO instrument a cross-engine equality assertion with the UPSTREAM facts, not just the two outputs,
  when the CI channel is only the driver's `tail -40`.** The D1 leg now, on mismatch, `panic!`s with the
  manifest field-132 `split_offsets` Rust plans from + `big.parquet`'s PHYSICAL parquet-footer offsets +
  `created_by` (the parquet-mr build) — so one `tail -40` localizes whether the manifest, the physical
  grid, or the emitted Java plan is the odd one out, on the next nightly. *Why:* the failing suite's
  self-printed tail is the ONLY channel that reaches CI logs here; put the evidence where it will be read.
  Keep such diagnostics best-effort — they run only on the already-failed path, so they must never panic
  themselves (String-map every error to `<unavailable: …>`; NEVER `unwrap`).
- **DO NOT assert an unproven failure mechanism as fact, and do NOT mask a possible same-file parity
  divergence with a fixture-determinism knob.** It was tempting to write "different parquet-mr row-group
  grids on the CI runner vs a dev box broke the exact-offset assertion" — but the independent Critic
  DISPROVED that as a *mechanism*: `java_scan_plan.json` is REGENERATED each run (never a committed
  golden), so within a run BOTH engines plan the SAME `big.parquet`, and Rust `plan_tasks` == Java
  `planTasks` at EVERY grid tried (2/3/4/8 offsets — both split one-sub-task-per-field-132-offset and
  bin-pack identically). A differing grid ALONE therefore cannot make the plans diverge; the reported
  Rust≠Java over the same file is a genuine `plan_tasks` PARITY anomaly (or a CI-only write/read
  inconsistency), which a grid pin would MASK, not fix. *Why:* a green nightly that hides a real parity
  gap is worse than a red one (the anti-false-green norm). The `PARQUET_ROW_GROUP_SIZE_BYTES 1024→64`
  change was kept ONLY as fixture hygiene (a byte target far below the ~1 KiB buffered-at-100-rows size
  forces parquet-mr's flush at its 100-row check FLOOR ⇒ a deterministic 8×100-row grid; byte-identical
  on a dev box) — explicitly NOT claimed as the fix. If the nightly goes green after this, that does NOT
  confirm the mechanism; the diagnostics (not the knob) are the deliverable. (Watch the unit trap: the
  Java `64` is BYTES → 8 groups; the Rust GEN side's `set_max_row_group_size(64)` is ROWS → 13 groups —
  the shared literal is coincidental, not a "mirror.")
