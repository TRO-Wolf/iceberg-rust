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

# H7-S2 — copy-on-write DML streaming (ledger)

**Branch:** `feat/h7-s2-cow-streaming` (worktree `iceberg-rust-ws`), cut from main @ `403625eb`.
**Scope:** the signed H7-S2 scope, §§1–8. Mode A, AC·OOO.
**Matrix:** no GAP_MATRIX row flips — this is an engine-side memory-profile change, not a capability.

## 1. What changed, and why

`copy_on_write_delete` and `copy_on_write_update` each held two unbounded buffers: **B1**, the entire
live table (`try_collect()` into `Vec<RecordBatch>`, held across both passes), and **B2**, every
rewrite row from every affected file, held until the writer ran. Copy-on-write UPDATE held a **third**,
`batch_masks` — one cached `BooleanArray` per batch (the "M7" optimisation).

Copy-on-write is inherently **two-pass**: a source file is "affected" the moment *any* of its rows
matches, possibly the last row of the last batch, so the affected set must be complete before the
first survivor may be emitted. The fix is therefore not single-pass streaming. It is:

* **Pass 1** streams the scan and retains only `affected: HashSet<String>` (one path per affected
  FILE) and the row counter. B1 gone.
* **Early exit** on `deleted == 0` / `updated == 0` unchanged, and it now skips the second scan
  entirely — a zero-match copy-on-write DML reads the table exactly once.
* **Pass 2 RE-SCANS** the same snapshot and feeds each batch's rewrite rows straight into
  `StreamingDataFileWriter`. B2 gone.
* **B3 (`batch_masks`) deleted, not adapted.** It was indexed by batch *position*; pass 2 is now an
  independent scan whose batch boundaries and arrival order are not guaranteed to match pass 1's, so
  indexing across them would silently apply one batch's mask to another batch's rows. Pass 2 now
  re-evaluates the predicate over `affected_batch`. This is the same value by the identity the M7
  comment itself asserted (`match_mask` is row-wise; `filter` preserves order); the traded cost is one
  extra predicate evaluation per batch.
* **The commit path is untouched** — `resolve_affected_data_files`, `overwrite_files`, and every §5
  validation are byte-identical.

Peak becomes `O(#affected files)` + one batch + the writer's own buffers.

### Snapshot consistency (settled in scope §4 — not re-litigated)

`Table` is a frozen handle: `metadata` is a plain `TableMetadataRef` with no interior mutability, and
the only mutator takes `mut self` by value. An unpinned `TableScanBuilder::build()` resolves from that
frozen metadata, never a fresh catalog read. Two `table.scan()` calls on one handle therefore resolve
the IDENTICAL snapshot regardless of concurrent commits.

Both scans additionally pass `.snapshot_id(scan_snapshot_id)` explicitly, via the new `cow_scan_stream`
helper. **This is a documented NO-OP, not a bug fix** — it records the invariant at the call site and
keeps it true if `Table` ever gains a refresh path. `None` (a snapshotless table) is left unpinned,
which yields the same empty scan.

### Collateral

* `write_partitioned_data_files` had exactly two callers, both of them the functions refactored here.
  It is now dead and was **removed**; its two stale doc references were updated (the one on
  `merge_on_read_update` was already wrong — that path has streamed since H7-S1).
* `test_m7_filtered_mask_equals_reeval` was **retired** with the code it justified. It is pure array
  algebra over hand-built masks and stays green either way, so it would not have caught the M7 removal;
  keeping it would have left a test whose doc comment describes code that no longer exists. *Judgement
  call — flagged for the Critic.*
* Copy-on-write DELETE previously open-coded what `table_column_batch` and `match_mask` already do,
  twice each. Both passes now use the shared helpers, which is what makes the DELETE and UPDATE passes
  structurally identical. This touches lines the minimal diff would not have.

## 2. Files changed

| File | Reason |
|---|---|
| `crates/integrations/datafusion/src/physical_plan/delete.rs` | The refactor: both COW paths streamed, `cow_scan_stream` added, `write_partitioned_data_files` + `batch_masks` + the M7 test removed, module/function docs corrected. |
| `crates/integrations/datafusion/tests/cow_memory_bound.rs` | **New** test binary: counting global allocator, marginal-peak memory evidence for both COW paths. |

No `Cargo.toml` edit was required or made — the crate has no `[[test]]` sections and relies on cargo
auto-discovery of `tests/*.rs`; `tempfile` and `tokio` (`macros` + `rt-multi-thread`) are already
dev-dependencies, and `MemoryCatalog` comes from the `iceberg` normal dependency.

## 3. The memory-evidence test

`tests/cow_memory_bound.rs`, one `#[tokio::test]`, **not** `#[ignore]`d. A `#[global_allocator]` in a
test binary has zero blast radius — the binary is its own process, and there is no other
`#[global_allocator]` in `crates/`.

**`realloc` handling:** accounted **explicitly** — delegated to `System.realloc` with the counters
adjusted by the size delta, rather than left to the `GlobalAlloc` default (which would route through
`alloc`/`dealloc` and always memcpy). It under-counts the transient inside a copying `System.realloc`;
that is the SAFE direction, since it can only make a measured peak *smaller* — i.e. make the buffered
form look better, never the streaming form.

### The assertion, and why it needed a measured baseline

The scope's form is `peak(4N) − peak(N) < ¼ × added_live_bytes`. **That form does not work in this
codebase, and the reason was measured, not guessed:** the Iceberg scan itself has memory that grows
with table size. `to_arrow` reads up to `concurrency_limit_data_files` files at once (default
`num_cpus` = **64** on this machine) and materializes a Parquet row group for each, plus per-file
scan-task overhead. On the *fixed, streaming* code that term alone measured between **4 MB and 58 MB**
depending on the fixture's file layout — enough to blow the ¼ threshold by itself, and it varies with
the host's core count.

The shipped assertion therefore subtracts a **measured baseline**:

```
[peak(4N) − peak(N)]  −  baseline_delta   <   ¼ × added_live_bytes
```

`baseline_delta` is the same marginal quantity for a zero-match **merge-on-read** DELETE over an
identical fixture. Merge-on-read already streams (H7-S1) and is untouched by this unit; matching
nothing, it writes no file and commits nothing, so what it measures is the scan. **Subtracting it is
sound against the mutation the test exists to catch** — reinstating `try_collect` changes only the
copy-on-write path, moving the measured delta by a whole live row set while leaving the baseline
exactly where it was. This is a deviation from the signed formula and is called out as such.

Rejected on measurement: a DataFusion `SELECT` control (delta 39.6 MB vs the DML's 14.1 MB — the
SELECT plan adds its own repartitioning, so it is not a clean baseline), and a zero-match *copy-on-write*
DELETE control (it shares the `try_collect`, so the subtraction would cancel the mutation and the test
would go green on buffered code — checked, not assumed).

### Fixture and its stated conditions

`N = 128_000`, `4N = 512_000`; `{id int, payload string}`, **unpartitioned**, `ROWS_PER_FILE = 2_000`
loaded as one data file per DataFusion input partition in a single INSERT, verified by counting live
manifest entries (64 files at N, 256 at 4N — pinned by an assertion, not assumed). Payload values are
**distinct per row**: a constant filler dictionary-encodes to almost nothing and can return from the
scan as a dictionary/REE array, which would make `ROW_BYTES_FLOOR` a wild over-estimate — a *loose*
threshold, the dangerous direction.

Three conditions, each isolating a memory axis this unit does **not** address:

1. **Unpartitioned.** A fanout `TaskWriter` holds one open writer per partition until close, so
   partition cardinality is its own axis.
2. **File size constant across scales, exactly ONE file affected.** The Parquet writer accumulates a
   row group before flushing, so peak also scales with rows *written*. Holding both constant makes the
   written volume identical at both scales so it cancels.
3. **One test function.** The counters are process-global.

Conditions 1 and 2 are load-bearing, not cosmetic — see the M2 measurement below.

**Runtime cost:** ~21–31 s for this binary. The rest of the crate's suite is ~0.5 s, so this is the
dominant test in the crate. That is the price of real evidence; it needs no Docker and is not gated.

## 4. Mutation proof — real observed numbers

All runs on the same machine (64 cores), `cargo test -p iceberg-datafusion --test cow_memory_bound`.
Threshold is `¼ × added_live_bytes` = **10,368,000 B** (added = 41,472,000 B) in every row.

### M1 — reinstate B1 (the `try_collect()` full-table buffer) in COW DELETE

| Run | baseline_delta | peak(N) | peak(4N) | delta | **excess** | verdict |
|---|---|---|---|---|---|---|
| Fixed (streaming) | 14,037,322 | 30,407,762 | 44,460,384 | 14,052,622 | **15,300** | GREEN (~678× headroom) |
| **M1 mutant** | 14,045,930 | 44,029,766 | 87,602,628 | 43,572,862 | **29,526,932** | **RED — fails by 2.85×** |

The mutant's UPDATE excess stayed at **6,363** (green), confirming the mutation is localized to DELETE
and that the test attributes it correctly.

### M2 — reinstate B2 (the survivor vector) alone, B1 left streaming

| Run | excess | verdict |
|---|---|---|
| M2 mutant, shipped fixture (1 file affected) | **4,791** | **GREEN — the test does NOT discriminate B2** |

**This is an honest negative result and it is not a pass.** Under the shipped fixture exactly one
constant-size file is affected, so the survivor volume is identical at both scales and B2 cannot grow
with `N`. Re-measured with an all-files-affected predicate to find out whether B2 is observable *at
all*:

| All-files-affected variant | excess |
|---|---|
| Fixed code (no B2) | **15,108,231** |
| M2 mutant (B2 present) | **13,914,483** |

The mutant is *lower* than the fixed code — i.e. B2 is **not observable**, within noise. The reason is
structural: B2 holds exactly the rows the Parquet writer is already retaining in its open row group
until flush, so removing B2 removes a redundant *second* copy whose magnitude is capped by a buffer
that remains. Note also that this variant's excess (~15 MB) exceeds the ¼ threshold on the *fixed*
code — which is the measurement proving condition 2 above is necessary rather than convenient.

**Conclusion stated plainly: this test proves B1. It does not prove B2.** B2's removal is justified by
code reading and is required by the streaming shape, but it has no memory evidence and should not be
claimed to have any until the writer-side row-group buffering is bounded (QB writer-bounds unit).

### M3 — behavioural: drop the `affected.contains` term (scope §7.2)

| Mutation | Result |
|---|---|
| COW **DELETE** pass 2 keep-mask loses `affected.contains(paths[row])` | **RED — 4 failed / 69 passed**: `test_delete_cow_multi_file_per_partition_only_affected_rewritten` (the named over-rewrite pin), `test_delete_cow_partitioned`, `test_delete_cow_non_identity_transform_truncate`, `test_delete_cow_path_matching_and_manifest_inspection` |
| COW **UPDATE** pass 2 `keep_affected` forced all-true | **RED — 4 failed / 69 passed**: `test_update_cow_multi_file_per_partition_only_affected_rewritten` (the pin), `test_update_cow_partitioned`, `test_update_cow_partitioned_moves_partition`, `test_update_cow_row_conservation_and_manifest_inspection` |

### M4 — behavioural: the M7 replacement mask (the riskiest part of this change)

| Mutation | Result |
|---|---|
| COW UPDATE passes `None` instead of `Some(&affected_match_mask)` to `apply_assignments` | **RED — 8 failed / 65 passed**, incl. `test_update_cow_null_predicate_three_valued_logic`, `test_update_copy_on_write`, `test_update_cow_unpartitioned_exact_filter_preserved`, `test_update_cow_row_conservation_and_manifest_inspection` |

So the freshly-evaluated pass-2 mask that replaced the M7 cache is well pinned by the existing suite.

## 5. Gate

| Command | Result |
|---|---|
| `typos .` | **pass** (exit 0, clean tree) |
| `cargo fmt --all -- --check` | **pass** |
| `cargo clippy --all-targets --workspace -- -D warnings` | **pass**, zero warnings |
| `cargo test -p iceberg-datafusion --all-targets` | **268 passed, 0 failed** (176 lib + 1 cow_memory_bound + 73 integration + 2 interop + 4 lazy + 12 partitioned_insert_select) |
| `cargo test -p iceberg --lib` | **3137 passed, 0 failed, 1 ignored** |
| `cargo check --workspace --no-default-features` | **pass** |

Lib tests went 177 → 176: exactly one test removed (`test_m7_filtered_mask_equals_reeval`), plus the
new `cow_memory_bound` binary's 1.

## 6. Named costs

* **The double read.** Pass 2 re-reads the whole live table. This is the accepted price of bounded
  memory. Restricting pass 2 to affected files is **out of scope**: the public scan API has no file-set
  filter and `_file` is a reserved metadata column, not a pushdown-able predicate term.
* **Not a perf unit.** Wall-clock regression from the second read is expected and acceptable.
* **One extra predicate evaluation per batch in pass 2 of UPDATE**, from dropping the M7 mask cache.
  DELETE already re-evaluated.
* **UPDATE newly inherits the volatile-predicate divergence DELETE already had.** With a
  non-deterministic predicate (`random()`, `now()`) the pass-1 count and the pass-2 row set may
  disagree. DELETE had this before this change (it re-evaluated in pass 2); UPDATE did not, because it
  cached. Pre-existing for DELETE, new for UPDATE, and inherent to the two-pass shape.
* **Error timing moved after the first write → orphan staged files.** Previously `apply_assignments`
  ran for every batch before any Parquet I/O, so the required-column-NULL guard fired with zero files
  written; now batch 1 may already be in an open writer when batch 2 trips it. No commit occurs either
  way, so this is not a correctness break, and `merge_on_read_update` already behaves this way. **COW
  has no test covering the NULL-into-required path** (the only one,
  `test_update_null_into_required_is_rejected`, builds a merge-on-read table), so nothing pinned the
  old behaviour and nothing reds on the change.
* **Output row order within rewritten files changes** — pass-2 arrival order rather than pass-1 buffer
  order. Both are already nondeterministic (`try_flatten_unordered`, and within-file parallel task
  expansion is enabled because COW does not project `_pos`). No Iceberg contract depends on it.
* **Pass 2 re-reads physical files.** A concurrent `expire_snapshots` + orphan cleanup between the
  passes fails pass 2's reads **loud** (an IO error), never silently. That hazard applies to any scan,
  including the single-pass one this replaces.

## 7. Follow-up seeds

1. **Writer-side row-group buffering is unbounded** — the Parquet writer accumulates a whole row group
   (arrow-rs default `max_row_group_size` = 1M rows) before flushing, and the rolling writer's target
   is 512 MB. Measured at ~15 MB marginal on an all-files-affected 512k-row DELETE. **This is now the
   dominant row-proportional term on the COW path**, and it is what makes B2's removal unprovable.
   → QB writer-bounds unit.
2. **Fanout writer holds one open writer per partition** until close, never evicting. Partition
   cardinality is an unbounded axis for partitioned DML. Options: clustered writer + a partition sort,
   or a bounded-writer LRU with mid-stream flush. → same QB unit.
3. **The scan's own peak grows with table size** — up to `concurrency_limit_data_files` (default
   `num_cpus`, 64 here) open files each materializing a row group, plus per-file scan-task overhead.
   Measured 4–58 MB marginal depending on layout. Not reachable from the DML path today
   (`with_concurrency_limit_data_files` is not threaded through), and it dwarfs everything H7-S2
   removed on a many-core host.
4. **Restrict pass 2 to affected files.** Needs a file-set filter on the scan API. Would remove most of
   the double-read cost as well as the second scan's share of seed 3.
5. **`MemoryReservation` / DataFusion `MemoryPool` integration** for the DML paths — a real idea, named
   in scope §6.1 as production scope creep, still open.
6. ~~**COW has no NULL-into-required-column UPDATE test.**~~ **CLOSED in remediation (§8, R4)** —
   `test_update_cow_null_into_required_is_rejected` added.
7. **COW 3VL had no non-vacuous guard test** — CLOSED in remediation (§8, R3): the Falsifier proved the
   `=`-only COW tests cannot falsify `match_mask`'s `is_valid` guard.
8. **This memory harness is reusable** — H7-P1 and the QB writer-bounds unit both make streaming claims
   with no way to evidence them today.

## 8. Remediation — 2026-08-05 (two independent Critics + Falsifier)

Both Critics **CONVERGED with zero blocking findings**; the Falsifier applied 10 mutations, reproduced
the headline M1 number independently (29,528,102 B vs the ledger's 29,526,932 B), **closed the UPDATE-arm
attribution gap for B1** (the UPDATE arm of `assert_marginal_bound` reds at 29,527,334 B when B1 is
reinstated in `copy_on_write_update`, so both arms are now falsifiable and cleanly attributed —
note this is *not* §4/M2's gap, which is that the test cannot discriminate **B2**, the survivor
vector; that one remains open by design and §4's "this test proves B1, it does not prove B2"
conclusion stands), and held the negative
control (removing the explicit `.snapshot_id()` pin left 268/268 green — the "documented no-op, not a
bug fix" framing is honest). One mutation exposed a real coverage gap (R3 below).

### Dispositions

| # | Finding (source) | Disposition |
|---|---|---|
| **R1** | Pass 1's exhausted scan stream is *shadowed*, not dropped, so it lives through pass 2 (correctness Critic) | **FIXED** — explicit `drop(stream)` after pass 1 in BOTH `copy_on_write_delete` and `copy_on_write_update`, with a comment naming the shadowing rule. The module doc's "peak is O(#affected files) + one batch" is now literally true. |
| **R2** | The memory binary leaks ~490 MB of `/tmp` per run (correctness Critic) | **FIXED** — `temp_path()` (drop-the-guard idiom) replaced by `temp_dir() -> (String, TempDir)`; `setup` returns the guards, `measure_dml_mode` drops the context and then the guards **after** `end_measure`, so teardown allocations cannot enter the measurement. Verified: `du -sm /tmp` unchanged (8431 MB → 8431 MB) across a full run, and the assertion stays green. |
| **R3** | The COW 3VL tests are `=`-only and cannot falsify `match_mask`'s `is_valid` guard — **Falsifier M-B went RED only via a merge-on-read test** (Critic + Falsifier finding 1) | **FIXED — mandatory coverage gap.** Added `test_delete_cow_null_neq_predicate_isvalid_guard` and `test_update_cow_null_neq_predicate_isvalid_guard` (`<>` over a NULL operand ⇒ validity=false, value=TRUE, so the guard is load-bearing). **Mutation-proved:** dropping `is_valid` from `match_mask` now reds **3** tests — both new COW tests plus the pre-existing MoR one — where before it red only the MoR one. |
| **R4** | Mid-pass-2 failure orphans staged Parquet files; the behaviour is safe but COW had no NULL-into-required test, so nothing pinned either the old or the new timing (both Critics; seed 6) | **FIXED (scheduled, not seeded)** — `test_update_cow_null_into_required_is_rejected` added: the statement must error AND the table must be byte-unchanged (`[1, 2]` intact). The staged-file *timing* change is real and remains disclosed in §6; what the test pins is the invariant that survived it (error + no commit + no partial rewrite). |
| **R5** | `StreamingDataFileWriter::try_new` moved before the pass-2 loop — a new error surface on a COW DML that fully empties every affected file (correctness Critic) | **FIXED by restoring the old behaviour** rather than by disclosure. DELETE's pass-2 writer is now built lazily on the first batch that actually has survivors, exactly as the deleted `write_partitioned_data_files` did (it returned `Ok(vec![])` before touching `DefaultLocationGenerator::new` / `PartitionValueCalculator::try_new`). UPDATE keeps eager construction and the reason is now in a code comment: `updated > 0` ⇒ some file is affected ⇒ every row of that file is rewritten ⇒ the writer is always fed, so its construction was always reached before this change too. |
| **R6** | `DELETE FROM t` (no predicate) pays a full second scan that provably yields nothing (correctness Critic) | **FIXED** — `predicate.is_none()` short-circuits pass 2 entirely (`new_files = vec![]`). Exact, not heuristic: with no predicate `match_mask` is all-true by construction, so `!deleted && affected.contains(..)` is false for every row. **Mutation-proved non-vacuous:** widening the condition to `true` reds **6** tests (`test_delete_from_copy_on_write`, `..._unpartitioned_exact_filter_preserved`, `..._non_identity_transform_truncate`, both COW-DELETE 3VL tests, `test_s5_cow_delete_snapshot_allows_concurrent_append`). This also removes the single most common shape from R5's residue. |
| **R7** | COW DML now emits two `ScanReport`s per statement for catalogs with a metrics reporter (evidence Critic) | **FIXED (disclosure)** — named in the module-level *Memory* note beside the double read, and carried into the PR body. No reporter is installed anywhere in `crates/integrations/datafusion`, so no test moves. |
| **R8** | UPDATE newly inherits the volatile-predicate divergence (correctness Critic) | **DECLINED — already disclosed, no code change.** §6 states it plainly. Bounded harm: no row is lost (a row pass 2 judges non-matching is still rewritten with its original values); only the reported count can disagree. Removing it would require re-introducing a cross-pass mask cache, which is exactly what this unit retired as unsound across independent batch boundaries. |
| **R9** | The memory assertion has only ever run on this 64-core host; the baseline subtraction cancels 99.89% of the measured quantity (both Critics) | **DECLINED as a code change; ACCEPTED as a claim limit.** The evidence Critic independently measured the baseline term collapsing to 0.40 MB under `taskset -c 0-3`, i.e. discrimination *improves* on small hosts; the FALSE-RED direction remains unmeasured off this host. Seed 3 already names it. The PR body claims only what was measured and flags the first CI run as the real proof. Not `#[ignore]`-ing it stays correct per scope §7.1. |
| **R10** | Falsifier M-D (removing the `keep.true_count() == 0` early-continue) stayed **GREEN** | **DECLINED — not a coverage gap, and the Falsifier agrees.** It chased the green rather than accepting it: a stacked probe proved zero-row batches genuinely reach the writer under M-D (6 tests hit it), yet the manifest-level assertions still held — the `TaskWriter` stack already swallows zero-row batches and commits no data file. So §3's zero-empty-file claim is true and independently pinned; the guard is an optimization with no distinct behavioural signature. Adding a test for it would pin an implementation detail, not a contract. |
| **R11** | `task/todo.md` still lists H7-S2 as pending (evidence Critic) | **DECLINED in-unit, by repo convention** — the queue file is reconciled at merge, and editing it here would put the unit's status in two places (CLAUDE.md's one-home-per-fact rule). Flagged for the merge commit. |

### Gate after remediation

| Command | Result |
|---|---|
| `typos .` | **pass** (exit 0, clean tree) |
| `cargo fmt --all -- --check` | **pass** |
| `cargo clippy --all-targets --workspace -- -D warnings` | **pass**, zero warnings |
| `cargo test -p iceberg-datafusion --all-targets` | **271 passed, 0 failed** (176 lib + 1 `cow_memory_bound` + 76 integration + 2 interop + 4 lazy + 12 partitioned_insert_select) — 268 → 271, exactly the three tests added by R3/R4 |
| `cargo test -p iceberg --lib` | **3137 passed, 0 failed, 1 ignored** |
| `cargo check -p iceberg-datafusion --no-default-features` | **pass** |

Memory assertion after remediation (same host, unchanged verdict, `/tmp` no longer growing):

```text
added=41,472,000 B, threshold=10,368,000 B
BASELINE (merge-on-read, zero match): peak(N)=30,172,720 peak(4N)=44,219,858 delta=14,047,138
DELETE: peak(N)=30,403,418 peak(4N)=44,457,037 delta=14,053,619 excess=6,481
UPDATE: peak(N)=30,437,808 peak(4N)=44,492,173 delta=14,054,365 excess=7,227
```

## 9. Remediation round 2 — 2026-08-05 (post-remediation review: 2 Critics + Falsifier)

**Both Critics CONVERGED with ZERO blocking findings** ("I would let this merge" / "I found no blocking
defect"). Twelve non-blocking findings were raised across the two lenses (several are the same item seen
twice). The **Falsifier report in this round was executed against `7e04b7b5` — the pre-remediation
commit** — so its one substantive finding (the COW 3VL guard gap) is the finding that round 1's R3
already fixed. That is not taken on trust: see R2.1.

### Dispositions

| # | Finding (source) | Disposition |
|---|---|---|
| **R2.1** | Falsifier finding 1 — M-B (dropping `is_valid` from `match_mask`) reds only a *merge-on-read* test; both COW 3VL tests stay GREEN because they are `=`-only (the repo's known vacuity trap). Mandatory coverage gap. | **ALREADY FIXED by round-1 R3, and RE-PROVED AT HEAD this round.** The Falsifier ran at `7e04b7b5`; the `<>` COW tests landed in `78c87af4`. Re-applied the exact M-B mutation (`.map(\|row\| raw.is_valid(row) && raw.value(row))` → `.map(\|row\| raw.value(row))`) on the current tree and re-ran the 8 NULL-predicate tests: **3 RED — `test_delete_cow_null_neq_predicate_isvalid_guard` (left 3, right 2), `test_update_cow_null_neq_predicate_isvalid_guard` (left 3, right 2), `test_update_mread_null_neq_predicate_isvalid_guard`** — 5 passed. Mutation reverted, tree verified clean by `git status --porcelain` + grep. The COW arms are now load-bearing pins, not coincidental passengers on the MoR test. |
| **R2.2** | Falsifier M-D stayed GREEN (the `keep.true_count() == 0` early-continue) | **DECLINED — unchanged from round-1 R10**, and the Falsifier itself concurs after stacking a probe: zero-row batches genuinely reach the writer under M-D, the `TaskWriter` stack swallows them, and the manifest-level assertions independently pin the no-empty-file contract. The guard is an optimization with no distinct behavioural signature. |
| **R2.3** | Module doc's "Peak is O(#affected files) + one batch + the writer's own buffers" omits the dominant measured term — the scan's own in-flight row groups (evidence Critic) | **FIXED (doc).** The sentence now says explicitly that this is the *DML path's own contribution*, not the total, and names the scan term: up to `concurrency_limit_data_files` (default `num_cpus`) in-flight Parquet row groups plus per-file task state, dominant on a many-core host, **bounded by concurrency not by row count — which is exactly why it cancels out of the marginal assertion**. This repo's rule is that a wrong reason in a doc actively misdirects; the fix is one clause and costs nothing. |
| **R2.4** | Ledger §8 says the Falsifier "closed the M2 gap" — it closed a different gap (evidence Critic) | **FIXED (ledger).** §8 now reads "closed the UPDATE-arm attribution gap for B1" and states in-line that §4/M2's gap — the test cannot discriminate **B2**, the survivor vector — **remains open by design**, with §4's "this test proves B1; it does not prove B2" standing. The correction matters: a reader must not come away believing B2 acquired memory evidence it does not have. |
| **R2.5** | Memory test verified host-robust beyond the single-host claim: GREEN under `taskset -c 0-3` **and** `taskset -c 0,1`; baseline_delta collapses 14,047,138 → 394,467 → 393,654 B, and the `assert_fixture` file-count pins (64 files at N, 256 at 4N) hold at 2 cores, so DataFusion does not coalesce the MemTable partitions on a small runner (both Critics) | **ACCEPTED — no code change; claim STRENGTHENED and its residual named.** Round-1 R9 declined host-dependence as unmeasured; it is now measured at 2, 4 and 64 cores and discrimination *improves* as cores fall. Residual is FALSE-GREEN only: `excess = delta.saturating_sub(baseline_delta)` floors at zero, so a host where the merge-on-read baseline over-measures relative to the COW run by >29 MB would mask a reinstated B1. Not observed at 2/4/64 cores. → PR body. |
| **R2.6** | The DELETE arm's printed `excess` is a `saturating_sub` clamp to 0 (`delta = 13,984,083` vs `baseline_delta = 14,046,987`), so the *printed* line shows no positive margin (evidence Critic) | **ACCEPTED — no code change.** The clamp is the streaming signature, not a measurement failure, and the DELETE arm's discriminating power is established directly by mutation M1 (excess 29,528,102 B, 2.85× the threshold), not by an in-band margin. → PR body, so the printed `0` is not read as "nothing measured". Removing the clamp would make a *negative* excess printable but would not add evidence. |
| **R2.7** | UPDATE newly inherits the volatile-predicate divergence (correctness Critic, confirming round-1 R8) | **DECLINED as a code change — disclosure only, unchanged.** Both Critics independently reached round-1 R8's conclusion: harm is bounded to the reported count (a row pass 2 judges non-matching is still rewritten carrying its ORIGINAL values, because its file is affected — no row lost or duplicated), and the only alternative is the position-indexed cross-pass cache this unit retired as unsound across two independent scans. DELETE already had it. → PR body. |
| **R2.8** | Mid-pass-2 failure orphans staged Parquet files (both Critics, twice) | **DECLINED as a code change — disclosed + invariant pinned.** No commit occurs either way, so the table is unchanged; `merge_on_read_update` has always had this shape; `test_update_cow_null_into_required_is_rejected` (round-1 R4) pins what matters (statement errors, table byte-unchanged). `DeleteOrphanFiles` (R134) is the recovery path. Pinning the *absence* of staged files would pin an implementation detail of writer construction order. → PR body. |
| **R2.9** | Two `ScanReport`s per COW statement — the one externally-visible non-memory behaviour change (correctness Critic) | **ACCEPTED — already in the module doc (round-1 R7); PROMOTED to the PR body** as its own line rather than a clause, since it is the only behaviour change a downstream consumer can observe without measuring memory. |
| **R2.10** | `cow_memory_bound` dominates crate test wall-clock (~20–21 s vs ~0.7 s for the other 270), I/O-bound not core-bound (20.68 s @64c, 19.68 s @4c, 19.89 s @2c), writes ~490 MB of temp Parquet per run (both Critics) | **ACCEPTED — no change.** Correctly NOT `#[ignore]`d per scope §7.1 (an ignored memory test is the exact false-green this unit closes); the temp files are cleaned up (round-1 R2, re-confirmed by the evidence Critic — the `TempDir` guards do travel out of `setup` and drop after `end_measure`). → PR body, so the CI-time jump is a chosen price and not something a future reader investigates. |
| **R2.11** | Output row order within rewritten data files changes (pass-2 scan order, not pass-1 buffer order) (correctness Critic) | **NO ACTION — already in §6.** Both orders were already nondeterministic (`try_flatten_unordered` + within-file parallel task expansion), no Iceberg contract depends on it, the table has no sort order, and the pre-existing code honoured none. The Critic re-ran `interop_partitioned_dml` (the Java-reads-Rust COW direction) and it passes, so nothing byte-pinned depends on it. |
| **R2.12** | `task/todo.md` still lists H7-S2 as pending (evidence Critic, accepting round-1 R11's reasoning but requiring it be flagged) | **DECLINED in-unit, ESCALATED to the PR.** Unchanged reasoning (one-home-per-fact; the queue file is reconciled at merge). The Critic's condition is met by listing it explicitly in the PR body as a merge-time obligation. |

### What the PR body must state (consolidated)

1. The double read — pass 2 re-reads the whole live table; accepted price of bounded memory (scope §5).
2. **Two `ScanReport`s per COW statement** — the only externally-visible non-memory behaviour change.
3. Volatile predicates (`random()`, `now()`) in a COW UPDATE `WHERE` can now make the reported count
   disagree with the applied row set; no row is lost or duplicated. DELETE already behaved this way.
4. Mid-pass-2 failure can leave staged, uncommitted Parquet files (DELETE and UPDATE); the table is
   unchanged; `DeleteOrphanFiles` (R134) recovers.
5. `cow_memory_bound` costs ~20 s of CI wall-clock and ~490 MB of transient temp space (cleaned up),
   and is deliberately not `#[ignore]`d.
6. The memory test is GREEN at 2, 4 and 64 cores with discrimination *improving* as cores fall; the
   residual risk is false-GREEN only (`saturating_sub` clamp), never false-RED as first feared.
7. The DELETE arm's printed `excess = 0` is a clamp artifact of the streaming signature; that arm's
   discrimination is proven by mutation M1 (29,528,102 B), not by the printed margin.
8. Row order within rewritten data files changes; both orders were already nondeterministic.
9. Pass 2 re-reading the same physical files means a concurrent `expire_snapshots` + orphan deletion
   fails pass 2 LOUD (IO error), never silently — pre-existing for any scan (scope §4).
10. **Merge-time obligation:** reconcile `task/todo.md`'s H7-S2 entry.
11. No GAP_MATRIX row flips — engine-side memory profile, not a capability (scope §8).

### Gate after round 2

| Command | Result |
|---|---|
| `typos .` | **pass** |
| `cargo fmt --all -- --check` | **pass** |
| `cargo clippy --all-targets --workspace -- -D warnings` | **pass**, zero warnings |
| `cargo test -p iceberg-datafusion --all-targets` | **271 passed, 0 failed** (unchanged — round 2 changed only doc/ledger prose) |
| `cargo check -p iceberg-datafusion --no-default-features` | **pass** |
| M-B re-mutation at HEAD (R2.1) | **3 RED / 5 pass**, reverted, tree clean |
