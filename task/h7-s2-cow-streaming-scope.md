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

# H7-S2 — COW streaming: scope (re-scoped 2026-08-05 against `403625eb`)

**Unit:** H7-S2, next in the signed queue after Unit 3 / QC / QB (all verified landed 2026-08-05).
**Targets:** `copy_on_write_delete` (`physical_plan/delete.rs:544`) and `copy_on_write_update`
(`:1735`).
**Predecessor:** H7-S1 (MoR streaming) merged #140 — the streaming idiom to match lives at
`merge_on_read_delete:387`.
**Cadence:** Mode A, AC·OO with an independent Critic. **NOT YET SIGNED — this document is the
scoping deliverable; §6 open questions need answers before Actor work begins.**

## 1. What is actually buffered today — two distinct buffers, not one

The brief says "two-pass → bounded refactor". Measured at `403625eb`, there are **two** unbounded
buffers and they need different fixes:

| # | Buffer | Site | Grows with |
|---|---|---|---|
| **B1** | `batches: Vec<RecordBatch>` — the **entire live table**, held across both passes | `delete.rs:565-575` (`try_collect()`), twin at `:1756` | total live rows |
| **B2** | `survivors_to_rewrite: Vec<RecordBatch>` — every surviving row from every affected file, held until the writer runs | `delete.rs:650`, `:711` | surviving rows in affected files |

The existing comment at `:577` ("Collect all batches into memory (documented: full-table buffer,
fits in executor memory)") is the honest statement of the current contract. This unit retires it.

**B2 is nearly free to fix.** `write_partitioned_data_files` (`:969`) is already a thin
`&[RecordBatch]` wrapper over `StreamingDataFileWriter::{write_batch, finish}` — the sink is
*already* streaming-capable. Pass 2 can drive the writer directly and B2 disappears.

**B1 is the real unit**, and §2 is why it cannot be removed the way S1's was.

## 2. Why a single-pass stream is impossible here (S1's shape does not transfer)

S1 (MoR) streams in one pass because a row's fate depends only on the row: predicate TRUE → emit a
position delete. COW's does not. A surviving row must be rewritten **iff its source file is
affected**, and a file becomes affected the moment *any* of its rows matches — possibly the last
row of the last batch. So the affected set must be **complete** before the first survivor may be
emitted. Two passes are inherent to COW, not an artifact of the current code.

This also means S1's precedent is a style guide, not a template. Note S1 is itself only partly
bounded — it accumulates `pairs: Vec<(String, i64)>`, one entry per *deleted* row — which is
inherent to MoR (a position delete per deleted row). Do not cite S1 as "already bounded".

## 3. Proposed shape — two pinned scans

- **Pass 1** streams the scan; accumulates **only** `affected: HashSet<String>` and `deleted: u64`.
  Bounded by the number of *files*, not rows. Predicate evaluation is unchanged (the exact
  `PhysicalExpr`, on a by-name table-column sub-batch — never pushed into the scan; see the module
  note on inexact Iceberg pushdown, and H7-P1's footgun).
- **Early exit** on `deleted == 0` — unchanged, and now avoids pass 2 entirely.
- **Pass 2** re-scans, and for each batch emits `!deleted && affected.contains(path)` rows straight
  into `StreamingDataFileWriter`. B2 gone.
- Commit path (`resolve_affected_data_files`, `overwrite_files`, the §5 validations) **untouched**.

Peak memory becomes `O(#affected files)` + one batch + writer buffers.

## 4. The snapshot hazard — INVESTIGATED 2026-08-05, and it is NOT real

An earlier revision of this scope led with a blocking hazard: that splitting one buffered scan into
two scans replaces one snapshot with two, so a concurrent commit between the passes could make pass 2
miss a survivor that pass 1 saw — and since the commit deletes every affected source file and adds
only what pass 2 wrote, that survivor would be silently destroyed.

**That analysis was wrong.** Traced through the source rather than reasoned about:

1. `copy_on_write_delete(table: &Table, ..)` receives a **frozen handle**. `Table.metadata` is a
   plain `TableMetadataRef` field (`table.rs:157-164`) with **no interior mutability**, and the only
   mutator, `with_metadata` (`:168`), takes `mut self` by value — so nothing can change it through a
   shared `&Table` for the lifetime of the call.
2. `TableScanBuilder::build()` resolves an unpinned scan's snapshot from
   `self.table.metadata().current_snapshot()` (`scan/mod.rs:487`) — the frozen metadata, **never a
   fresh catalog read**.

Therefore two `table.scan()` calls on the same handle resolve the **identical** snapshot no matter
what commits concurrently. The two-scan refactor is snapshot-consistent by construction, and the
S5 suite already covers the real concurrency surface (see §6).

**Residual, minor and pre-existing:** pass 2 re-reads the same physical data files. A concurrent
`expire_snapshots` + orphan-file deletion could delete them mid-DML, failing pass 2's reads. That
hazard applies to *any* scan, including today's single-pass one, and fails **loud** (an IO error),
never silently. Not a blocker; note it in the PR body.

**Optional, recommended anyway:** pass `.snapshot_id(scan_snapshot_id)` on both scans explicitly.
It is a no-op given the above, but it documents the invariant at the call site and makes the
property robust if `Table` ever gains a refresh path. Costs nothing; do not present it as a fix for
a live bug.

## 5. Named costs and non-goals

- **Double read.** Pass 2 re-reads the whole table. This is the accepted price of bounded memory
  and must be stated in the PR body, not buried. Restricting pass 2 to affected files only is
  **out of scope**: the public scan API has no file-set filter, and `_file` is a reserved metadata
  column, not a pushdown-able predicate term. Note it as a follow-up seed.
- **Not a perf unit.** The bar is behavior-invariance plus a memory-profile change. Any wall-clock
  regression from the second read is expected and acceptable.
- **H7-P1 stays out.** No `with_filter` / pushdown work here. P1's pre-condition (the
  NOT-over-dropped-conjunct under-delete footgun, and threading the raw `Vec<Expr>` through both
  exec structs) is unchanged and remains mandatory *for P1*.
- **`copy_on_write_update` gets the same treatment in the same PR** — it is the same shape plus
  updated values (`:1735`). Splitting them would leave a known unbounded twin on main.

## 6. Open questions — answer before signing

Questions 1 and 2 of the earlier revision are **CLOSED by the §4 investigation**; recorded here so
the reasoning is not redone.

- ~~*Does any caller depend on float-to-latest between passes?*~~ **Moot** — the handle is frozen,
  so there is no float-to-latest behavior to depend on.
- ~~*Is there an OCC test that would catch a wrongly-pinned pass 2?*~~ **No such test exists, and
  none is needed.** The S5 isolation suite (`integration_datafusion_test.rs:5490-6230`, 15 tests
  incl. 5 CoW-specific) commits concurrently between `s5_freeze_plan` (physical-plan creation,
  which freezes the table handle) and `s5_execute_frozen` — i.e. **before pass 1**, exercising the
  §5 commit-time OCC validations. That is the correct surface for those tests and it is unaffected
  by this refactor. A between-the-passes test would be testing an impossibility.

**Remaining open — one question:**

1. **How does the bounded-memory claim get evidence?** "It streams now" is not testable, and the
   existing COW suites will pass identically before and after (they are small-fixture functional
   tests), so a green run proves nothing about memory. Options: (a) a row-count-scaled test that
   asserts peak retained batches stays O(1) via an instrumented counter on the scan stream;
   (b) a `#[cfg(test)]` high-water-mark counter incremented in the pass loops; (c) accept a
   code-shape argument and state explicitly in the PR that memory is unproven. **(a) or (b) —
   #187's Critic caught exactly this shape of tautology (identical test counts on a change that
   added zero tests), and a "streaming" claim with no memory evidence is the same error.**

**Consequence for scoping:** with §4 closed, H7-S2 is a genuine mechanical refactor plus one
memory-evidence test. It no longer needs a RED-first concurrency pin, and the unit is materially
smaller than the earlier revision implied.

## 7. Test plan (draft)

- **Memory evidence** (per §6.1) — the one test that justifies the unit.
- Behavior-invariance: the existing COW DELETE/UPDATE suites pass unchanged.
- Empty-table / `deleted == 0` early exit (pass 2 never runs).
- Every-row-deleted (affected files fully emptied → writer produces zero files — already handled
  at `:723`, keep the pin).
- Partitioned COW through the partition-aware writer.
- Mutation proof: drop the `affected.contains` term → an over-rewrite pin reds (rows from
  unaffected files get needlessly rewritten).

## 8. Gate

Standing chain: `typos . && cargo fmt --all -- --check && clippy -D warnings && iceberg lib +
iceberg-datafusion all-targets && --no-default-features`, anchors if the matrix moves.
**No GAP_MATRIX row flips expected** — this is an engine-side memory-profile change, not a
capability. `make interop` not required unless the commit path changes (it should not).
