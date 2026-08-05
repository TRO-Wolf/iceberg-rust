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

## 4. THE hazard this refactor introduces — snapshot pinning (blocking, must be designed first)

**One scan sees one snapshot. Two scans do not.** `table.scan()` resolves the current snapshot at
build time, so a concurrent commit landing between pass 1 and pass 2 gives pass 2 a *different*
table. The commit then deletes every affected source file and adds only what pass 2 wrote —
so any row pass 1 counted as a survivor that pass 2 does not see is **silently destroyed**.

That is a data-loss class the buffered version structurally cannot have. It is the single reason
this unit is not a mechanical refactor, and it must be closed in the design, not discovered in
review.

**Mitigation:** pin *both* passes to the snapshot already captured for the §5 anchor —
`let scan_snapshot_id = table.metadata().current_snapshot_id();` (`:552`) — via
`TableScanBuilder::snapshot_id` (`scan/mod.rs:268`). Two details:

1. `scan_snapshot_id` is `Option<i64>`; `None` means an empty table at read time. Decide the branch
   explicitly — an empty table has no rows, so pass 1 finds nothing and `deleted == 0` exits before
   pass 2. Assert that rather than leaving it implicit.
2. Pinning pass **1** as well is not optional. If pass 1 floats and pass 2 is pinned, the affected
   set can reference files absent from the pinned snapshot, and `resolve_affected_data_files`
   ("every affected path MUST resolve … a missing path is an internal invariant breach", `:757`)
   turns a concurrency race into an internal error.

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

1. **Isolation semantics of the double read.** Pinning both scans makes the DML read a fixed
   snapshot, which is *stronger* and matches Java (`SparkWrite` reads the scan snapshot). Confirm
   no fork caller depends on the current float-to-latest behavior between passes.
2. **Is there an existing OCC test that would catch a wrongly-pinned pass 2?** If not, this unit owes a
   RED-first concurrency pin — commit between the passes, assert survivors are not lost. That pin is
   the deliverable that justifies the unit; without it the hazard is only argued.
3. **Memory assertion strategy.** A bounded-memory claim needs evidence. Options: a row-count-scaled
   test asserting peak batch retention is O(1), or an instrumented counter. Decide, because "it
   streams now" is not testable and would repeat the tautology finding from #187.

## 7. Test plan (draft)

- **RED-first concurrency pin** (per §6.2): concurrent commit between passes → survivors intact.
  Must red on an unpinned pass 2.
- Behavior-invariance: the existing COW DELETE/UPDATE suites pass unchanged.
- Empty-table / `deleted == 0` early exit (pass 2 never runs).
- Every-row-deleted (affected files fully emptied → writer produces zero files — already handled
  at `:723`, keep the pin).
- Partitioned COW through the partition-aware writer.
- Mutation proofs: unpin pass 2 → concurrency pin reds; drop the `affected.contains` term → an
  over-rewrite pin reds.

## 8. Gate

Standing chain: `typos . && cargo fmt --all -- --check && clippy -D warnings && iceberg lib +
iceberg-datafusion all-targets && --no-default-features`, anchors if the matrix moves.
**No GAP_MATRIX row flips expected** — this is an engine-side memory-profile change, not a
capability. `make interop` not required unless the commit path changes (it should not).
