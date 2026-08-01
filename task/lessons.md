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

### 2026-07-25 — A byte-stability claim is about the rendered OUTPUT type, never the transform's name

G4 remediation cycle 2 (engine-trust bundle), R161. The format-stability attestation for the
partition-path escaper asserted that "every `bucket` / `truncate` / `day` output is byte-stable" —
affirmatively false, and the PR's own shipped interop evidence refuted it: Java's oracle case
`truncate_string` is `s_trunc=a%2Fb+c`. `Transform::result_type` returns `input_type.clone()` for
`Truncate`, so `truncate(string, N)` renders a STRING and is exactly as escaper-sensitive as
`identity(string)` — and `truncate` over a high-cardinality string column is the single most likely
real-table shape to move.

- **Group the claim by what is RENDERED, not by the transform that produced it.** `bucket` → int and
  `day` → date really are stable because their OUTPUT type is; `truncate` is not a peer of theirs, it
  is type-preserving. *Detector:* before writing "transform X is stable", read `result_type` for X —
  if it can return the source type, the claim must be stated per source type.
- **A universally-quantified sentence in an attestation is a testable claim; make it a pin.** The
  wrong clause survived a full Critic cycle because it lived only in prose. It is now
  `truncate_is_byte_stable_except_over_string`, offline, and it reds under 5 of the 9 mutations.
- **A "measured, not derived" sweep is only as complete as its type list.** The same sentence named
  three always-moving temporal types; `timestamp_ns` / `timestamptz_ns` move identically and were
  missed by two consecutive sweeps. *Rule:* enumerate from `PrimitiveType`'s own variants, not from
  the types the fixture happened to use.

### 2026-07-25 — A per-site mutation anchor that is a SUBSTRING of another site's line silently re-runs that site

Same increment. Three `name=null` call sites share the text `return Ok(escaped_partition_pair(&field.name,
"null"));` at different indentations; the 12-space anchor is a substring of the 16-space line, so a
first-match string replace mutated the WRONG site and the "site 3" leg silently re-ran "site 2".

- **The tell is that a per-site mutation reds a DIFFERENT site's test.** If leg N reds leg M's pin, the
  anchor missed — a per-site mutation must red the test named for that site, and nothing else. Treat a
  mismatch as a harness bug, never as evidence.
- **Anchor on the preceding distinguishing line, not on the mutated line alone** (here `let Some(literal)
  = slot.as_ref() else {`). Indentation is not identity.

### 2026-07-25 — A guard inside a SHARED predicate can be dead for one caller and load-bearing for another; mutate the helper and check EVERY caller's tests

WG4b (path-keyed position-delete routing). `referenced_data_file_location` (Java
`ContentFileUtil.referencedDataFile`) opens with "equality deletes are never file-scoped". A read-path
unit test was written to pin it and passed — but the mutation that DELETES that early return produced
zero failures, because the index consults the helper only inside its `PositionDeletes` match arm, so
the guard is unreachable from that caller. The guard is genuinely load-bearing for the OTHER caller,
`RemoveDanglingDeleteFiles`, where an equality delete judged by reference instead of by the partition
min-sequence rule is removed while it still applies — an irreversible resurrection.

- **DO run each mutation against the WHOLE suite, not the test you wrote it for, and read the failure
  set.** An empty failure set is the finding: the code you mutated is dead for that path, or nothing
  covers it. Here the empty set bought a real corruption-direction test that did not exist.
- **DO NOT let a test's doc comment claim a mutation it does not actually detect.** The corrected
  comment now names the mutation that DOES red it (content-blind routing) and points at the sibling
  test in the other module that owns the early return.
- **Detector:** a `pub(crate)` predicate with more than one caller. Enumerate the callers before
  writing the mutation list, and give each caller its own leg.

### 2026-07-25 — parquet-rs truncates byte-array statistics at 64 bytes by DEFAULT, so a bounds-derived fixture can silently carry no bounds

Same increment. A test that needed a position delete carrying EQUAL `file_path` lower/upper bounds
wrote one with `MetricsConfig::for_position_delete()` (which forces that column to FULL) and still got
no bounds at all: `DEFAULT_STATISTICS_TRUNCATE_LENGTH = Some(64)` in parquet-rs truncates the min/max,
`Statistics::min_is_exact()` then returns false, and the Iceberg metrics aggregator drops the bound.
Java's parquet-mr does not truncate row-group statistics, which is why the same fixture shape works
there — a real fork-vs-Java asymmetry, not just a test problem.

- **DO assert the fixture's own precondition when the fixture depends on emitted METRICS** (here:
  both bounds present AND equal AND naming the intended file). Without it the test would have gone
  green the moment the routing stopped being exercised.
- **DO reach for `set_statistics_truncate_length(None)` when a test needs exact bounds**, and say in
  the comment why — otherwise the next reader deletes it as noise.

### 2026-07-25 — a NULL-parent bug is invisible to any test whose fixture masks its children

G6 (WG5, null-bit propagation). Arrow does **not** require a null struct slot to mask its children —
`StructArray::try_new` only enforces the *reverse* containment (a non-nullable field's own nulls must
be masked by the parent). So the class of defect is: detach a nested child from its parent, or judge
it on its own, and you read whatever bytes happen to sit under a logically-absent row. Three separate
live sites had it, and **none** of the pre-existing tests could see it: the projector's only nested
test has no nulls at all, and the equality-delete fixture's nested key column happened to be non-null.

- **DO build the fixture with the child LIVE and the parent NULL.** That is the only shape that
  distinguishes "propagates validity" from "does not". A fixture where the child is null too passes
  either way, and a fixture where the parent is live never enters the branch.
- **DO make the negative pin a MINIMAL PAIR of the positive one** — same arrays, one flipped null
  bit. For the "required field has a null value" check, the pair (parent NULL ⇒ accept) /
  (parent LIVE ⇒ still reject) is what proves the check moved rather than disappeared; a mutation
  that deletes the check entirely reds only the second one.
- **DO NOT assume the union is the whole fix.** Unioning the parent's validity into the child makes
  a `required` child under a NULL parent *more* null, so the required-check had to move to the
  parent-aware callback in the same change; the union alone would have made the false rejection
  worse, not better.
- **Detector:** any `-> Result<&'a P>` accessor over a nested container, and any `.column(i)` /
  `.columns()[i]` whose result outlives the parent binding.

### 2026-07-25 — a fixture's own expected string can be asserting the bug

Same increment. `test_delete_file_loader_parse_equality_deletes` asserted `sa != 4` for a key column
that lives inside struct `s`. `Schema::name_to_id` indexes nested fields by their FULL dotted path,
so `sa` is unbindable — the fixture had frozen an unbindable reference as the contract, and the real
fix flipped it to `s.sa`.

- **DO treat a pre-existing expectation that flips as EVIDENCE, not as breakage** — but only after
  deciding from the spec (here `Schema::name_to_id` + `Reference::bind`) which side is right, and
  saying so in the test comment.
- **DO check whether the assertion ever exercised the consequence.** A predicate string test never
  binds the predicate; adding a `bind()` leg turned a cosmetic-looking rename into a proven
  scan-failure fix.

### 2026-07-25 — A conversion that feeds a PUSHDOWN filter must be exact or absent; there is no safe rounding direction

G7 (engine-trust bundle). `scalar_value_to_datum` turned a DataFusion `Date64` literal
(milliseconds) into an Iceberg `date` (days) with `(millis / MILLIS_PER_DAY) as i32` — a wrap AND a
truncation, in a function whose result becomes the scan's only filter.

- **DO answer `None` for any input the target type cannot represent EXACTLY when the consumer is a
  filter.** The comparison operator lives at a different call site, and every rounding direction is
  under-inclusive for some operator: for `millis = 1 day + 1 ms`, flooring makes `col < millis`
  match `{0}` where the truth is `{0, 1}`, and rounding up breaks `>` symmetrically. A converter
  that already returns `Option` has the correct answer built in — "cannot be pushed down" — and the
  engine then evaluates the predicate itself.
- **DO NOT assume `Inexact` pushdown makes a wrong predicate harmless.**
  `TableProviderFilterPushDown::Inexact` means the engine re-checks the rows the scan RETURNS; it
  cannot resurrect rows the scan pruned. Over-inclusive is free, under-inclusive is silent row loss
  — that asymmetry is what decides between "convert approximately" and "do not push down at all".
- *Detector:* any `-> Option<Datum>` / `-> Option<Literal>` converter whose result becomes a
  comparison's right-hand side. Read every `as` cast inside it first; each is a wrap waiting for an
  out-of-range literal, and the RED test writes itself (one day past `i32::MAX` became `i32::MIN`,
  i.e. a far-future bound pushed down as a far-past one).

### 2026-07-25 — When you cannot build the adversarial input, pin the HELPER — then spend the freed budget on the hazard the EDIT introduces

Same increment, `transform/bucket.rs` + `truncate.rs`: 19 `downcast_ref::<ConcreteArray>().unwrap()`
sites, each inside a `match input.data_type()` arm. For arrow's own arrays the pairing is exact
(`make_array` maps each `DataType` to exactly one struct), but `transform` takes `Arc<dyn Array>` and
`Array` is a public trait, so it is not enforceable. Writing a lying `Array` impl to prove the branch
reachable turned out to be impossible from this crate: the trait requires
`fn to_data(&self) -> ArrayData`, and `arrow-data` is not a direct dependency (adding one needs
Cargo approval).

- **DO factor the unprovable guard into a named helper and unit-test the HELPER with an ordinary
  mismatched pair** (a `StringArray` asked for as an `Int32Array`). The guard then REDs under a real
  mutation (`ok_or_else` → `.unwrap()`) even though no call site can reach it — far better than a
  prose claim that it is unreachable.
- **DO spend the freed test budget on the hazard the EDIT introduces, not the one the code had.**
  Rewriting 19 downcasts risks pairing a concrete type with the WRONG arm; `Bucket`'s array path had
  tests for 2 of its 13 arms. A per-arm sweep whose oracle is the independent `transform_literal`
  match is what actually protects the change, and it REDs under two different arm-swap mutations.
- **A mutation that does not COMPILE is evidence too.** Swapping truncate's `Int64` arm to
  `Int32Array` is rejected by the annotated result type, so that arm's mapping is owned by the
  compiler, not by a test — record which arms have that guarantee instead of "adding coverage" the
  type system already provides.

### 2026-07-26 — A blanket `#[allow(unused_variables)]` can hide a DEAD SYNCHRONIZATION HANDLE; and a publish-then-await test cannot tell "armed under the lock" from "armed one line later"

The delete-filter lost-wakeup unit (G8, `arrow/delete_filter.rs`) turned up three notifier bugs where
the brief predicted two. Three things generalise:

- **Remove a module-wide `#[allow(unused_variables)]` BEFORE trusting anything about its concurrency.**
  `impl CachingDeleteFileLoader` carried one. Under it, `try_start_eq_del_load` returned an
  `Arc<Notify>` that the caller bound and never used — while the very next line replaced the state
  entry with a SECOND notifier. A waiter that armed on the first one in the window between the two
  calls was woken by nothing, ever. A dead binding of a notifier / lock guard / channel handle is not
  lint noise, it is a liveness bug the compiler was already pointing at. Cost of removing the allow
  here: three lines (two other dead bindings, both genuinely dead, deleted).
- **The mutation that proves an arming fix must cross the function boundary the test observes.** The
  test shape is "publish FIRST, await SECOND", which pins *the future existed before the
  notification*. Moving the arming a few lines later — still inside the claiming function, just after
  the lock is released — does NOT fail that test, so it proves nothing. The honest revert is the BASE
  CONTRACT: hand back the raw `Arc<Notify>` and create the `Notified` at the AWAIT SITE. That one
  REDs (`Elapsed(())`), and it is exactly what the pre-fix tree did — so the same test also gives a
  genuine pre-group RED when the old API can express it.
- **Arm the drop guard in the CLAIM, not in the publisher's registration.** Every `?` between "insert
  Loading" and "register the publisher" is a stranded waiter. Returning the guard *from* the claiming
  call makes that window unrepresentable, and letting the spawned publisher CAPTURE the guard (rather
  than construct it inside its `async move`) is what covers the never-polled teardown — a future
  dropped before its first poll runs no local destructors. Those are two separate pins with two
  separate mutations; one passing does not imply the other.

### 2026-07-26 — When one fix rewrites TWO call sites of shared machinery, the SECOND site needs its own pin — the whole suite stays green without it

The G8 remediation (independent Critic, `arrow/caching_delete_file_loader.rs`). The lost-wakeup fix
rewrote both positional-delete claim sites: the parquet path (`{file path}` key) and the
deletion-vector path (`{puffin path}@{offset}` key). Only the parquet path got a production-path
test. Two independent regressions in the DV half — leaking the claim on the blob-read failure, and
consulting the wait state under `&task.file_path` instead of the composite key — each left ALL 2967
lib tests green, while one of them reproduced the exact hang the group's headline test exists to
catch. Generalisations:

- **Enumerate the call sites your diff touched and check each one appears in a test name.** "The
  mechanism is pinned" is not the bar; the bar is "every site that instantiates the mechanism is
  pinned". A composite claim key is its own risk: a guard that publishes under the wrong key leaves
  the real entry `Loading` forever, and no test of the other site can see it.
- **A terminal failure state should carry its CAUSE from the first design pass.** The failing task is
  the only one that sees its own error; every waiter and every later caller reads the state instead.
  `Failed` as a unit variant means they learn THAT a load died, never WHY — and on a caching loader
  the later caller has no other channel. Recording `error.to_string()` at each failure site (and a
  generic reason from `Drop`, which has none) costs one `map_err` per site and is mutation-provable
  per site.
- **Bound EVERY test await that can reach a wait path, not just the ones about waiting.** A
  pre-existing predicate test with a bare `.await` turned a lost-wakeup mutation into a hung CI job
  instead of a red test — and hid a fourth RED in the mutation matrix. `tokio::time::timeout(5s)`
  around the await converts it into evidence.

### 2026-07-26 — Per-test wall-clock bounds RELOCATE a lost-wakeup hang; and a "did the waiter WAIT?" property is invisible to any test that reads the result after both sides joined

Two refinements to the previous entry, both from the G8 cycle-1 Critic and both proved by mutation
rather than argued.

- **A race-shaped concurrency test cannot pin an ORDERING property.** Two concurrent `load_deletes`
  calls for one delete file DO drive the real `WaitFor` arm, but the assertions run after both
  futures have joined — by which point the claiming task has published either way. So "the waiter
  observed the populated vector" is unfalsifiable there: deleting the wait entirely (`drop(notified)`
  in place of the `.await`) left the whole suite green while every deleted row RESURRECTED. Drive it
  DETERMINISTICALLY instead — claim the loader's own key first, so the production path must take
  `WaitFor`, then assert the load is STILL PENDING while the claim is unpublished. The same fixture
  then pins the fail-loud half: record a sentinel cause, drop the claim, and assert the woken
  waiter's error carries it (that one also kills a hard-coded post-wake reason). Corollary for
  reviewing an assert message: if the message names a property the test's own timing cannot observe,
  the message is the bug.
- **`let _ = ...await;` is a distinct mutation class from deleting the await, and on a wait path it
  is the WORSE one.** Removing the await returns early; swallowing the result returns AFTER the wait
  but as though it had succeeded, so a waiter whose claimant DIED proceeds with no deletes at all.
  Both are one-token slips at a `?`. Run both against every `.await?` a diff introduces on a
  synchronisation path.
- **Do NOT try to close "a lost-wakeup regression must not hang CI" with per-test timeouts.** Bounding
  the awaits inside the tests that exist to pin the wait contract is right — the bound is part of the
  pin. Spraying bounds elsewhere is not: with the eq-delete arming mutation applied, the full lib
  suite hung in one compaction test; bounding that module's read helper simply MOVED the hang to the
  next test in the same module (the one that reaches the delete-applied read through
  `RewriteDataFiles::execute` rather than the helper). Any test that reads merge-on-read deletes can
  hang, so the class-correct instrument is a HARNESS-level per-test timeout (`cargo nextest`
  `slow-timeout` + `terminate-after`, or a CI job timeout), not N test edits. Scope the claim in the
  PR body to what the bounds actually cover and name the rest as a residue.

### 2026-07-26 — A per-group gate that TRIPS must be settled before the next group starts; a residue ledger can contradict itself two clauses apart; and an `assert_ne!` alarm needs its own non-vacuity mutation

From the closing-Critic remediation of the engine-trust bundle.

- **"Parked, not converged" is not a state a branch can hold.** The bundle brief says a group whose
  ladder cannot converge in two remediation cycles is reset to its last good commit and the bundle
  ships without it. G4's cycle-2 Critic returned CHANGES_REQUIRED; the group was recorded PARKED —
  and its four commits were left on the branch while five more groups stacked on top. Nothing was
  hidden and no one lied, but by the time the bundle Critic re-read the rule, the cheap remedy
  (reset, one commit deep) had become a conflict-heavy rebase across six groups that invalidates
  every per-group attestation sha. A tripped gate has exactly two settlements, and both are cheap
  ONLY at the moment it trips: actually reset the branch, or record the waiver in writing. Deferring
  the choice silently converts "ship without it" into "ship it".
- **A residue inventory is a promise the next agent will act on — file each divergence on the right
  side of the ledger.** Row R161 said `truncate` "already matches" Java in one clause and "the binary
  leg's human string is hex" two clauses earlier; Java renders base64 there. Every mechanical check
  passed (anchors, 5-pipe audit, citation resolution) because the contradiction is semantic. The
  only instrument that catches it is measuring EACH claim of the inventory against the JVM, one row
  at a time — and the failure mode it protects against is specific: the next agent closes the named
  residue, skips the item filed as matching, and flips the row.
- **An `assert_ne!` alarm is a dead comparison until you prove the compared string is REACHABLE.**
  The alarm that keeps a residue non-skippable ("when this becomes equal, the residue is closed and
  the row must be updated") passes trivially forever if the expectation is a string the code can
  never produce. Its non-vacuity mutation is the inverse of the usual one: simulate the residue being
  CLOSED (here, make `display_bytes` emit Java's base64 for the pinned bytes) and prove the alarm
  goes RED. Order it BEFORE the value pin in the same test, or the byte-stability `assert_eq!` fires
  first and the alarm's own coverage stays unproven.

### 2026-07-31 — D11: Grok octo Actor + critic-octo 8× early_stop=false for the plan_tasks slate (deviation from AC·OO)

Recorded on WG0 (rebase + re-gate of parked WG1/WG2) before flagship G1 lands, so the cadence
deviation is on the ledger even if G1 never touches `task/`.

- **DO treat D11 as a written, slate-scoped deviation from `CLAUDE.md` §subagent_policy AC·OO.**
  For units WG0, G1 (plan_tasks multi-partition), G2 (PartitionKey::Result + toHumanString), and
  G3 (HMS typed errors): the Actor is Grok-class under `/sepmo-octo`, and the Critic engine is
  standalone `/critic-octo` with **8 cycles and `early_stop=false`**. Convergence claims without
  green named-RED mutation runs at the **actual tip** are slate-failing (a mutation that was RED
  three commits ago is not RED).
- **DO NOT invent an external Opus Critic mid-run for this slate.** The orchestrator's morning
  review is the independent Opus-class pass before any merge recommendation. Grok critic-octo is
  intentional overnight hardening, not a substitute for that pass.
- **Why:** QD/QE + perf-wave A–E precedent on this fork; G1's merge gate is objective (14 pins +
  named mutations HARD-FAIL never SKIP). Source: `~/.claude/plans/2026-08-02-grok-fork-plantasks-slate.md`
  D11; dossier `~/Desktop/iceberg-rust-overnight-work-slate-2026-07-31.md`.
