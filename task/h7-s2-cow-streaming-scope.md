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
**Cadence:** Mode A, AC·OOO — one Opus Actor, **two** independent Opus Critics with distinct lenses
(the H7-S1 build-phase ladder). **SIGNED 2026-08-05** — all §6 questions are closed; the
memory-evidence form is decided in §6.1 and specified in §7.

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

### 6.1 How the bounded-memory claim gets evidence — DECIDED 2026-08-05

The existing COW suites assert *which rows survive*; those assertions are byte-identical before and
after this refactor, so the suite goes green either way. A "COW now streams" claim backed by that
green run is exactly the tautology #187's Critic caught — evidence that would look the same if the
claim were false.

**The requirement is therefore not "add a memory test" but: produce a test that is RED against the
buffered code and GREEN against the streaming code.** That criterion selects the form.

**Chosen: a counting global allocator in a dedicated test binary, two data scales, assertion on the
_marginal_ peak.** Specified in §7.1.

Rejected, with reasons (do not re-litigate):

| Option | Why not |
|---|---|
| Absolute byte threshold (`peak < 50 MB`) | A magic number encoding machine + allocator + arrow version. Flakes, gets bumped, stops discriminating. |
| `#[cfg(test)]` high-water counter in the pass loops | Mutation-provable, but measures *the proxy we chose to instrument*, not the claim — blind to buffering inside the writer, inside the scan stream, or in any future regression. A code-shape argument wearing a test's clothes. |
| DataFusion `MemoryPool` with a hard limit | **Cannot work.** Pool accounting only sees operators that register a `MemoryConsumer`; `try_collect()` into a plain `Vec` is invisible to it. Making the COW path hold a `MemoryReservation` is a real idea but is production-code scope creep plus a new failure mode — follow-up seed, not this unit. |
| `dhat` | The standard tool, but a new `[dev-dependencies]` entry, and CLAUDE.md forbids dependency-file edits without explicit approval. The hand-rolled allocator is ~40 lines of `std::alloc` and needs no approval. |

**Consequence for scoping:** with §4 closed, H7-S2 is a genuine mechanical refactor plus one
memory-evidence test. It no longer needs a RED-first concurrency pin, and the unit is materially
smaller than the earlier revision implied.

## 7. Test plan

### 7.1 The memory-evidence test — the one deliverable that justifies the unit

**Location:** a NEW binary, `crates/integrations/datafusion/tests/cow_memory_bound.rs`. A `tests/*.rs`
file compiles to its own binary and runs in its own process, so a `#[global_allocator]` declared
there has **zero blast radius** on the library or any other test. The crate already carries four such
binaries. There is no `#[global_allocator]` anywhere in `crates/` today (verified) — nothing to
collide with.

**Instrument.** `std::alloc` only, no new dependency:

```rust
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            PEAK.fetch_max(LIVE.fetch_add(l.size(), Relaxed) + l.size(), Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}
```
(`realloc` must be handled too — either leave it to the default `GlobalAlloc` provided impl, which
routes through `alloc`/`dealloc`, or account it explicitly. Whichever the Actor picks, say which.)

**Assertion — marginal, never absolute.** Run the *same* COW DELETE at N and 4N rows: same schema,
same file count, same deleted fraction, same affected-file fraction. Only the row count varies.

```
peak(4N) − peak(N)   <   ¼ × (added live bytes)
```

Constant overheads cancel exactly, so there is no magic number and no machine dependence. At
N = 128k / 4N = 512k and ~128 B/row (an i64, a couple of i32s, a ~100-char string) the added volume
is ≈ 48 MB:

- **Buffered (today):** `batches` and `survivors_to_rewrite` are **both live at the peak** — pass 2
  reads `&batches` while filling the survivor vec — so the delta lands at 48–96 MB against a 12 MB
  threshold. **Fails by 4–8×.**
- **Streaming (after):** the delta is a `HashSet<String>` of file paths plus one batch — well under
  1 MB. **Passes with ~10× headroom.**

That margin is deliberate: the counter races under a multi-thread runtime and the measurement is
approximate by construction. An order-of-magnitude gap means the approximation cannot flip the
verdict. **Warm up with one discarded small run** so tokio/planner/parquet one-time init is paid
before either measurement.

**Two fixture constraints, both to be restated in the PR body:**

1. **Unpartitioned table.** `StreamingDataFileWriter` wraps `TaskWriter<DmlDataFileWriterBuilder>`
   (`delete.rs:826`); a fanout task writer holds an open writer per partition, so partition
   cardinality is a **second** memory variable. Holding it at zero isolates the row-count claim.
   High-cardinality partitioned writes are a genuine separate unbounded-writer question — **name it
   as a follow-up seed, do not fold it in.**
2. **One test function per memory binary.** The counter is process-global, so anything concurrent in
   the same binary pollutes it. DELETE-small / DELETE-large / UPDATE-small / UPDATE-large run
   sequentially inside one `#[tokio::test]` — no lock needed and no `--test-threads` requirement to
   forget. (There is no `serial_test` precedent in the repo; separate-binary isolation is the
   mechanism.)

**Cost:** the COW suites build on `MemoryCatalog` + `TempDir` — no Docker, no MinIO — so this stays a
fast local test. It is **not** `#[ignore]`d: an ignored memory test is a false-green in CI, the exact
failure mode being closed here.

**The deliverable is the mutation proof, not the test.** Reinstating `try_collect()` is a one-line
revert, so the PR body must record the RED run **with real observed numbers**. Without that, the
test is unfalsifiable decoration and reproduces the #187 error one level up.

*Amortization note:* H7-P1 and the QB writer-bounds unit both make streaming claims that today have
no way to be evidenced. This harness serves them.

### 7.2 The rest

- Behavior-invariance: the existing COW DELETE/UPDATE suites pass unchanged.
- Empty-table / `deleted == 0` early exit (pass 2 never runs).
- Every-row-deleted (affected files fully emptied → writer produces zero files — already handled
  at `:723`, keep the pin).
- Partitioned COW through the partition-aware writer (functional, small fixture — **not** the memory
  test, per §7.1 constraint 1).
- Mutation proof: drop the `affected.contains` term → an over-rewrite pin reds (rows from
  unaffected files get needlessly rewritten).

## 8. Gate

Standing chain: `typos . && cargo fmt --all -- --check && clippy -D warnings && iceberg lib +
iceberg-datafusion all-targets && --no-default-features`, anchors if the matrix moves.
**No GAP_MATRIX row flips expected** — this is an engine-side memory-profile change, not a
capability. `make interop` not required unless the commit path changes (it should not).
