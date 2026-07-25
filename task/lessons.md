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
- **RESOLVED 2026-07-23 (the instrument-first move paid off; #164 diagnostics → precise fix).** The very
  first nightly on the instrumented `main` printed the answer in one `tail -40`: manifest field-132 ==
  physical footer == the SAME 8 offsets (byte-identical to a dev box, `created_by` = the same parquet-mr
  1.16.0 build 402c3810), Rust correctly planned 8 — but Java's `java_scan_plan.json` was 5. So it was
  NOT a Rust bug, NOT the grid, and NOT dependency drift: the Java **oracle planned its IN-MEMORY table**,
  whose `DataFile.splitOffsets()` returned only 5 offsets — a coarse SUBSET of big.parquet's 8 physical row
  groups — **on the GitHub runner only** (a parquet-mr-1.16 in-memory-vs-footer quirk; the 5-split plan is
  a strict subset of the 8, and Java's split layer is provably one-sub-task-per-offset), while the
  PERSISTED manifest correctly kept all 8 — exactly what Rust reads. **Fix: emit `java_scan_plan.json` from the table RELOADED off the
  persisted `final.metadata.json`** (reusing `ScanPlanOracle.verify()`'s disk-load path), so both engines
  plan the identical persisted manifest — apples-to-apples. *Why this is the RIGHT fix, not a mask:* the
  persisted manifest is authoritative and is what any reader sees; comparing Rust's persisted-table plan to
  Java's *in-memory* plan was the apples-to-oranges bug. **DO, in a cross-engine oracle, have BOTH sides
  plan the PERSISTED artifact — never plan a freshly-built in-memory table against the other engine's
  on-disk read** (writer in-memory state can legitimately differ from the flushed footer). Local stays a
  no-op (in-memory == persisted == 8). The `block=64` fixture-hygiene knob was orthogonal (kept, harmless).

### 2026-07-24 — A merged-span START masquerades as an offset SUBSET; decompose the LENGTHS before diagnosing a split-count divergence

Context: the same nightly D1 (`interop_scan_plan.rs`) that the 2026-07-23 entry "resolved" via oracle-reload was
STILL diverging on the runner. The 2026-07-23 diagnosis — "Java's in-memory `DataFile.splitOffsets()` returned a
coarser 5-offset SUBSET of big.parquet's 8 physical row groups" — was the **#165 WRONG diagnosis**. The real cause,
found by decoding the 1.10.0 jar: `TableScanUtil.planTasks` maps every bin through `BaseCombinedScanTask(List)`,
whose ctor calls `TableScanUtil.mergeTasks` (javap: `REF_newInvokeSpecial BaseCombinedScanTask.<init>:(List)`),
which MERGES adjacent contiguous same-file splits within a bin (`SplitScanTask.canMerge` = same `file()` +
`offset+len==next.start`; `merge` sums lengths). Rust never merged, so on the runner — where MoR delete-byte
pack-weights pushed 2 splits into one bin — Java emitted merged spans and Rust emitted both members.

- **DO decompose a "fewer members on one engine" divergence by SUMMING the suspect side's member LENGTHS against the
  other side's, not by comparing member STARTS as a set.** Java's members `(4,939),(943,943),(1886,947)` are EXACTLY
  Rust's `{(4,469),(473,470)},{(943,470),(1413,473)},{(1886,474),(2360,473)}` pairwise-merged (`4+469==473`,
  `469+470==939`; …). The merged STARTS `{4,943,1886}` are a SUBSET of the 8 split starts
  `{4,473,943,1413,1886,2360,…}`, so a start-only comparison SCREAMS "coarser grid / dropped offsets" when the truth
  is "adjacent spans coalesced." *Why:* a merge preserves the run's FIRST start and its TOTAL length; the interior
  starts vanish, mimicking a subset. Only the length arithmetic (`Σ member.len == merged.len`) distinguishes "merged"
  from "re-gridded." Apply BEFORE attributing a split-count gap to parquet-mr, in-memory-vs-persisted state, or
  dependency drift.
- **DO treat a same-file split-count parity gap as a MERGE-LAYER question first** (`mergeTasks` in the
  `CombinedScanTask`/`ScanTaskGroup` ctor): Java merges in the group constructor, not the split layer — the split
  layer is one-sub-task-per-offset on both engines, so a divergence over ONE file is almost always the merge.
- **DO NOT trust a delete-file's pack-weight to be environment-stable when the delete embeds an absolute path.** The
  D1 fixture's `big-deletes.parquet` records big.parquet's ABSOLUTE path; its serialized length (hence the bin-pack
  WEIGHT `length + contentSizeInBytes(deletes)`) differs by warehouse root, so the number of splits that land in one
  bin sits on a `target=4096` knife edge — single-member bins locally (merge a no-op, D1 green), 2-member bins on the
  runner (merge fires, Rust≠Java). *Why:* the fix must be environment-INDEPENDENT — implement the merge itself, not a
  fixture knob that only re-centres the knife edge. The merge is a no-op on single-member bins, so local plans are
  byte-unchanged (the no-regression proof) while the runner now matches.

### 2026-07-24 — Assertion GRANULARITY and fixture COVERAGE are independent axes; audit both before calling an interop row proven

Context: the `interop_scan_plan` oracle compares plans as a multiset of per-group member-key SETS, where a member
key is `(basename,start,length)` — so a merge divergence (`[(big,0,512),(big,512,512)]` vs `[(big,0,1024)]`) IS
representable and WOULD be caught. The oracle's granularity was fine. What was missing was fixture COVERAGE: with
`big/mid/small` + one MoR delete at `target=4096`, no bin ever deterministically held ≥2 ADJACENT splits of one
file, so `TableScanUtil.mergeTasks` was never exercised on either side and the comparison was silently vacuous with
respect to it. R148 first went ✅ after auditing only the first axis. Measured proof of the vacuity, from the
mutation leg added in this unit: with `merge_tasks` DELETED from `CombinedScanTask::new`, the pre-existing
UNFILTERED D1 comparison still passes at 14 groups — only the new merge-filtered leg reds.

- **DO audit an interop pin along BOTH axes: (1) can the comparison REPRESENT the divergence, and (2) does the
  FIXTURE drive the code path that produces it?** *Why:* the two are independent, and a green bidirectional oracle
  says nothing about a branch its fixture never reaches. State the second one explicitly in the row's evidence
  ("the fixture forces X"), not just the first ("the comparison compares X").
- **DO make a coverage claim EXECUTABLE — assert the branch fired, in the same test — and be precise about WHICH
  assertion does the discriminating.** Here the proof is an exact `assert_eq` on the plan SHAPE (ONE group holding
  ONE whole-file-spanning member) read against an invariant of the layer BELOW: the offsets-aware splitter emits
  exactly one sub-task per split offset and ignores the target, so `>= 2` offsets means `>= 2` splits and a single
  spanning member is only producible by the merge. *Why:* it is easy to ship a plausible-looking numeric guard that
  is arithmetically ALWAYS TRUE — the first draft here asserted "the merged member's length exceeds the largest
  single row-group span", which given `>= 2` ascending offsets is implied, and named THAT the non-vacuity
  mechanism. The Critic caught the misnaming, not a hole. Keep such checks (they catch a degenerate fixture) but
  label them as fixture guards, and let a SHAPE equality plus a named lower-layer invariant carry the proof.
- **DO give a branch-coverage fixture its own ISOLATING scan rather than tuning it into the shared plan.** The
  merge fixtures are planned under a metrics-prunable row filter (disjoint `id` ranges) at the DELETE-FREE append
  snapshot, so their splits meet an EMPTY bin-packer with weights equal to their lengths. *Why:* co-binning inside
  the shared plan depends on how every other file happened to pack AND on the delete file's size (which embeds an
  absolute path, so it varies by checkout root) — that is the same 4096 knife edge that made the original failure
  runner-only. Isolate the precondition instead of tuning toward it, and ASSERT the sizing (`fileLength < TARGET`;
  outer spans sum `<= TARGET`; middle span `> TARGET`) so a future fixture edit fails loudly.
- **DO NOT use low-entropy filler when a fixture's size is the thing being asserted.** `gap.parquet`'s middle row
  group must exceed the split target on BOTH engines, which write parquet with different default codecs (Rust
  uncompressed, Iceberg-Java zstd); zero-padded digits would have compressed away on the Java side only. A
  deterministic LCG over a 62-char alphabet, mirrored on both sides, keeps the byte size a property of the fixture.
- **DO pair a new interop leg with a PRODUCTION-SOURCE mutation leg in the same harness run** (mutate → expect RED
  → restore → md5-verify), hard-failing when the mutation pattern is absent, and add ONE MUTATION PER CLAIM. *Why:*
  the Java-side sabotage legs prove the fixture is discriminating; only removing the Rust code under test proves the
  RUST assertions are. One mutation is not enough when the pin makes two claims: deleting the merge outright reds
  the "merge fires" leg but NOT the "adjacency is respected" leg (with no merge at all, the non-contiguous pair
  survives for the WRONG reason) — that one needs its own mutation, dropping the contiguity clause so `can_merge`
  degenerates to group-by-file.
- **DO make the mutation leg's assertion-signal grep GATING, not decorative.** A mutant that fails to COMPILE also
  exits non-zero, so gating on the exit code alone turns a broken mutation into a green "the test is
  load-bearing" — the classic mutation false-green. Require the run's output to match
  `panicked at|assertion .*failed|test result: FAILED`; no signal ⇒ restore and hard-fail as INCONCLUSIVE.
- **DO `touch` the restored file after an in-place source mutation, and re-run the leg GREEN before exiting.**
  Caught the hard way in this unit: `cp -p` + `mv` restore the ORIGINAL mtime, which leaves the restored source
  OLDER than the mutant's build artifacts — cargo's mtime staleness check then silently reuses the MUTANT lib for
  every later build in that checkout (the next full chain run failed in GEN with the merge visibly absent, from a
  byte-perfect working tree). *Why:* an md5-identical restore is necessary but NOT sufficient; the build cache is
  part of the state a mutation leg perturbs. `touch` costs nothing (content, hence the md5, is unchanged) and the
  green re-run both proves the restore and evicts the mutant.

### 2026-07-25 — A pre-warmed `target/` in a fresh worktree can serve STALE rlibs that cargo calls fresh: verify a post-fix symbol is IN the linked artifact before believing any RED

Context (G0, engine-trust bundle): the first RED-first run in the `iceberg-rust-ws` worktree failed **seven** tests,
including four (`T2`/`T3`/`T4`/`T10`) that #172 had just made green on the branch's own parent commit — with the
exact pre-#172 signature (`record_batch_projector.rs:183: index out of bounds`, tuples `[books, electronics]`).
The source on disk was unambiguously post-#172. The linked artifact was not: `strings
target/debug/deps/libiceberg-<hash>.rlib | grep -c "Column index out of bounds for batch in RecordBatchProjector"`
returned **0**, and cargo had printed no `Compiling iceberg` line — it considered the crate fresh. `find
crates/iceberg/src crates/integrations/datafusion/src -name '*.rs' -exec touch {} +` forced the rebuild; the same
grep then returned 1 and the four tests were green. (Provenance of the warm `target/` was not determined; the
observable is that cargo's freshness check passed over artifacts built from different bytes.)

- **DO prove the artifact matches the source before trusting a RED (or a GREEN) in a worktree with an inherited
  `target/`:** pick a string that exists ONLY in the post-fix source — an error message, a new symbol name — and
  `strings` the linked rlib for it (`cargo test … --no-run -v | grep -oE '\-\-extern <crate>=[^ ]*'` names the exact
  file). *Why:* a stale-but-"fresh" rlib fabricates a RED that looks exactly like the bug you are about to fix, and
  the natural response — "good, the defect reproduces" — writes a whole unit against a phantom. The same failure
  mode in the other direction (stale rlib still containing the fix) manufactures a false GREEN, which ships.
- **DO NOT infer freshness from `cargo` recompiling SOMETHING.** The run that produced the phantom RED did print
  `Compiling iceberg-datafusion` — it was recompiling only the integration-test target, linking two stale lib
  rlibs underneath. A `Compiling` line for the crate you edited says nothing about its dependencies.

### 2026-07-25 — A verifier that reads through the normal read path is fed the very metadata it is trying to falsify

Context (G1, engine-trust bundle): the partition-tuple detector re-reads each data file and recomputes its
partition tuple to compare against the manifest entry. The obvious implementation — plan the scan, read the
tasks, recompute — is **vacuous for identity transforms**: a `FileScanTask` carries `partition` +
`partition_spec`, and `RecordBatchTransformer` materializes identity-partitioned columns as CONSTANTS from
that tuple which **override the column physically stored in the file** (`constant_overrides_file_column`,
Iceberg "Column Projection" rule 1 / Java `PartitionUtil.constantsMap`). Recomputing `identity(col)` from
such a read returns the RECORDED value by construction, so a corrupted file audits clean — and the matching
*repair* would have written that wrong value into the rewritten rows. Both halves were mutation-proven:
clearing the two task fields is the fix; restoring them drops the identity-only finding (2 findings → 1) and
turns the repaired data from `{(eng,alp),(ops,bet)}` into `{(sales,alp),(sales,bet)}`.

- **DO, when building a tool that verifies metadata M against the data, audit every layer between the bytes
  and your comparison for a path that injects M.** A read path optimized to *trust* metadata (constants,
  pruning, projection short-cuts, cached stats) is exactly the wrong substrate for *checking* it. Enumerate
  what the task/handle you pass down carries, and strip anything derived from M — here `partition`,
  `partition_spec` (constants) and `predicate` (a residual that would drop rows). *Why:* the failure is
  silent and total — the verifier passes on genuinely corrupt input, which is worse than not having one.
- **DO make the vacuity independently falsifiable by SPLITTING the fixture along the axis the trap follows.**
  Two corrupt files were built, one wrong ONLY in its `identity(dept)` component and one wrong ONLY in its
  `truncate[3](name)` component. The mutation then yields a *precise, explainable* delta (the identity-only
  finding disappears, the truncate-only one survives) instead of a mushy "some tests fail". A single
  all-families-wrong fixture would have gone RED under the mutation too — for reasons that do not
  discriminate. *Apply:* when one transform/branch/type is the trap, give it a fixture no other branch can
  cover for it.
- **DO build the corrupted fixture at the layer BELOW the one that was fixed.** After the engine fix (#172)
  no engine path can produce a wrong tuple, so the fixture is written through the public `DataFileBuilder` +
  `fast_append`: the commit path validates a partition tuple's ARITY and TYPES but never its VALUES (Java
  identical), so a same-typed wrong value commits cleanly. *Why:* "we can't reproduce it any more" is not a
  reason to skip the regression fixture — it is a reason to inject one layer down, and the injection point
  doubles as the honest statement of what the format does and does not enforce.

### 2026-07-25 — A "silent corruption" fixture must be chosen so the corruption STAYS silent; and assert the row-level outcome BEFORE the metadata field that caused it

Context (G2, engine-trust bundle): the delete writers stamped `partition_spec_id` only when a `PartitionKey`
was present, else `DEFAULT_PARTITION_SPEC_ID` (0). Two fixtures were built for the same defect. Fixture A
(spec 0 UNPARTITIONED, evolved to a partitioned spec 1): the wrong stamp COMMITS — spec 0's partition type is
empty, exactly the tuple the file carries — and the delete then never applies; every row survives, nothing
errors. Fixture B (spec 0 partitioned, evolved to an UNPARTITIONED spec whose id is 1): the same wrong stamp
is caught LOUDLY at commit (`Partition value is not compatible with partition type`), because spec 0's type
has a field the empty tuple cannot fill.

- **DO pick the fixture whose claimed-spec partition type is ARITY- AND TYPE-COMPATIBLE with the tuple the
  file carries when the claim is "this corruption is silent."** Commit validation checks the tuple against
  *the spec the file claims*, never against *which* spec the file belongs to (Java identical) — so the same
  wrong id is loud or silent purely as a function of the two specs' shapes. A fixture in which it is loud
  proves the opposite of what the unit is about, and a reviewer reading only the test name cannot tell. State
  which shape each fixture exercises in its doc comment.
- **DO order an end-to-end test so the ROW-LEVEL outcome is asserted BEFORE the metadata field you believe
  causes it, and label the field assertion as a corroborating guard.** The first draft asserted
  `delete.partition_spec_id() == cur_spec_id` and only then the post-delete row set; every mutation that
  changed the stamp therefore reddened on the *stamp* assertion, leaving the row-set assertion never
  executed under any mutation — i.e. unproven. *Why:* a metadata assertion placed upstream of the behavior
  short-circuits every mutation you would use to prove the behavioral assertion is load-bearing.
- **DO add a WITHIN-FIXTURE positive control when the headline assertion is "nothing happened".** "The rows
  survived the delete" is also what you would see if deletes never worked in that fixture at all. The same
  table, the same data file, the same positions, re-deleted with the correct `PartitionKey` — and now
  emptied — makes the spec stamp the only difference between the two outcomes. This is a controlled A/B, and
  it is stronger than a mutation here because no in-scope production mutation can hold the commit legal while
  breaking the pairing.
- **DO key a "does this spec need a partition tuple?" guard on partition-field ARITY, never on
  `PartitionSpec::is_unpartitioned()`** — the latter is also `true` for an ALL-VOID spec (`fields.is_empty()
  || all fields are Void`), whose partition TYPE still has fields, so a tuple is still required.
  `is_unpartitioned()` would wave exactly that case through into the commit-time arity failure the guard
  exists to prevent. Mutation-proven: swapping the predicate reds the all-void leg and nothing else.

### 2026-07-25 — A fixture that differs on TWO dimensions cannot attribute the outcome to one; and never state a read-side consequence without executing it

Same unit (G2, engine-trust bundle), found by the independent Critic. Two separate defects in what was
otherwise a correct, mutation-proven change — both of them in the *claims*, not the code.

- **DON'T pin a claim of the form "X is what excludes it" with a fixture that also differs in Y.** The
  wrong-spec delete e2e differed from its data in BOTH halves of the read-side `(spec_id, partition)` key:
  spec 0 vs spec 1 AND `Struct::empty()` vs `{"eng"}`. The exclusion therefore happened at the partition
  BUCKET LOOKUP and never reached the `partition_spec_id` comparison the test was advertised as proving —
  deleting that comparison from `delete_file_index.rs` left the test GREEN. *Fix pattern:* find two specs
  whose TRANSFORMS AGREE on the fixture value, so the tuple is byte-identical and the id is the only
  variable — here `truncate[5](dept)` and `identity(dept)`, both `{"eng"}` for `"eng"`. Then the
  condition-deleting mutation reds. *Cheap detector:* mutate the exact production line you are claiming
  credit for; if the new test does not red, the fixture is not isolating it.
- **DON'T write a normative read-side consequence into a contract doc (or public rustdoc) from the writer's
  side of the seam — run it.** §7a asserted a wrong-spec delete "is never applied to any data file, for both
  equality and position deletes". True for position deletes; INVERTED for the equality-delete shape the same
  paragraph told engines to avoid: a writer with no `PartitionKey` also emits an EMPTY partition tuple, and
  per the Iceberg spec an equality delete stored with an unpartitioned spec is a GLOBAL delete — Rust routes
  on the empty tuple (`PopulatedDeleteFileIndex::new`), Java on `spec.isUnpartitioned()`
  (`DeleteFileIndex.Builder.add`), and the global bucket is consulted with no spec-id and no partition
  condition at all. The doc told engines "rows resurrect" where the real behavior is a table-wide
  OVER-delete. *Why it matters more than a typo:* the two hazards need opposite mitigations, and the doc was
  the deliverable. A contract sentence about what the READER does is only as good as the e2e that executed
  it; write the probe, then write the sentence.

### 2026-07-25 — When a phase runs FIRST and cannot fail, its panics preempt every typed error downstream

G3 (engine-trust bundle), totalising `PartitionSpec::partition_to_path`. Two reusable findings:

- **Locate the abort by PHASE ORDER, not by which check "should" have caught it.** `SnapshotProducer::commit`
  runs `summary()` BEFORE `manifest_file()`, and `summary()` is infallible. So a commit over broken metadata
  died in the summary — the typed `DataInvalid` the manifest rewriter would have produced for the SAME input
  was unreachable. An in-code comment at the fabricating call site even claimed "the validation will reject
  them" — validation never ran. *Detector:* before asserting which error a hardening test observes, read the
  order of the phases in `commit()`; the first infallible phase that touches the bad input is the one that
  must be made total, and the test then observes the SECOND phase's error.
- **An infallible signature you cannot change is hardened by a `try_` sibling with an EQUALITY contract, not by
  a second opinion.** `partition_to_path` must stay infallible (the public `LocationGenerator` trait, plus the
  summary path), so the fix pairs it with `try_partition_to_path` whose contract is "returns EXACTLY the same
  string whenever it returns `Ok`". That single sentence makes the leniency mutation-provable: the total path's
  behavior is pinned to the fallible one everywhere except the anomaly branches, and one test asserts the
  equality directly. Reuse an EXISTING predicate for the anomaly test (`PrimitiveType::compatible`, already the
  commit path's `validate_partition_value` rule) instead of mirroring the formatter's match arms — then add a
  matrix test that EXECUTES every accepted pair through the formatter as the drift alarm.

### 2026-07-25 — Java's partition-path escaper is FORM encoding, not percent encoding

G4 (engine-trust bundle), R161. Java `PartitionSpec.escape` decodes to a one-liner —
`java.net.URLEncoder.encode(s, "UTF-8")` — and `partitionToPath` runs BOTH the field name and the
transform's human string through it. Two reusable findings:

- **DON'T reach for the percent-encoding default when a Java call site says "URL encode".**
  `URLEncoder` is `application/x-www-form-urlencoded`: `-`, `_`, `.` and `*` pass through, and a
  SPACE becomes `+`, not `%20`. An RFC-3986-style fix (e.g. a `NON_ALPHANUMERIC` set) would have
  churned the layout of every ordinary partition path (`us-east-1`, `2024-01-31`, `a.b`) — the exact
  opposite of the "only special values move" promise the format-stability attestation makes. The
  space rule also has a trap of its own: mapping space to `+` is only sound BECAUSE a literal `+` is
  escaped to `%2B` in the same pass; do one without the other and the distinct values `"a b"` and
  `"a+b"` collapse onto the SAME path. *Detector that costs one JVM invocation:*
  sweep `0x20..=0x7E` through the real Java method and pin the surviving set as a constant; the
  sweep is 95 assertions and refutes every guessed safe-set in one run.
- **A behavior-preserving control is only load-bearing if a plausible WRONG fix reds it.** The two
  "safe-set values are byte-identical to pre-change output" regressions look vacuous next to a
  no-escaping mutation (they stay green — correctly, since safe values are unchanged either way).
  They earn their place under the OVER-escaping mutation: restricting the pass-through set to
  alphanumerics reds them and nothing else reds them. *Pattern:* pair every "nothing changed for the
  common case" pin with a mutation that changes the common case, or the pin is decoration.

### 2026-07-25 — Mutating a shared seam at ALL its call sites proves only "at least one is covered"

G4 remediation (engine-trust bundle), R161. The escaping fix funnels every `name=value` pair through one
helper, and the increment's mutation for the `name=null` family unescaped **all three** call sites at once:
1 test went RED, which reads like coverage. Mutating them **individually** showed two of the three were
completely unpinned — `partition_to_path`'s lenient fallback and the `void`-past-end-of-tuple branch each
left the full 2920-test lib suite GREEN. Both are reachable in three lines, and the first sits on the commit
path (`SnapshotProducer::summary` pairs the CURRENT schema with a file's OLDER spec), so a regression there
would put a raw `/` straight into a `partitions.` summary key — the exact defect the change existed to remove.

- **A single-seam refactor CONCENTRATES the code but MULTIPLIES the call sites you must pin.** "One helper,
  so it cannot be missed" is an argument about today's structure, not about what the suite enforces; the next
  editor changes a call site, not the helper. *Rule:* when a fix routes N branches through one helper, the
  mutation budget is N+1 — one per call site, plus the helper itself — and the union-mutation is a summary,
  never the evidence.
- **The tell is arithmetic:** an N-site mutation that reds strictly fewer than N distinct tests has at least
  one unpinned site. Cheap detector, no reasoning required — count the sites, count the distinct RED tests.
- *Corollary for docs:* a rustdoc/matrix sentence like "no branch can skip it" is only earned once each
  branch reds alone; otherwise state what the suite actually enforces.
