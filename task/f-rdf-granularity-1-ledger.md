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

# Ledger — F-RDF-GRANULARITY-1: `rewrite_data_files` output count follows Java's read-split planning

**Ledger id:** `F-RDF-GRANULARITY-1-2026-09-18`
**Branch:** `fix/f-rdf-granularity-1` (cut off fork `main`)
**Scope:** RePark row ICE-RDF-GRANULARITY-1 — one defect, one fix, one ledger
**Model:** swe-2-high

| Item | Tag | Commits | Subject |
|---|---|---|---|
| 1 | F-RDF-GRANULARITY-1 | `34dea9bb` / `d635ccf9` / `7c4b45db` | red output-count cells, per-read-split output planning, ledger (round 1, rebased onto `9f36da97`) |
| 2 | F-RDF-GRANULARITY-1 round 2 | `9cf50d1b` / `9ff89f5d` / this commit | footer-map + delete-loader sharing, planner remainder cells, ledger |

## 1. Step 0 — measured sizes (fork writer, RePark shape)

Shape: `(id BIGINT, p INT, v STRING) PARTITIONED BY (p)`, format v2, 2 partitions × 4
single-append files × 50 rows (`id` consecutive from 0, `v = repeat('x', 20)`), written through
the fork's own writer path (`write_l001_file`, the same `DataFileWriterBuilder` + parquet route
the existing rewrite tests use). Measured `file_size_in_bytes`, printed by the red cell:

```
fork-written file sizes: [(0, 1694), (1, 1694), (0, 1694), (1, 1694),
                          (0, 1694), (1, 1694), (0, 1694), (1, 1694)]
```

Eight files of **1,694 B** each (Spark's were ≈1,153 B — a writer difference, not a planning
input). Per-partition input 6,776 B.

## 2. Java's answer on the measured 1,694 B files

Java 1.11.0 rules (verified against the cached
`iceberg-spark-runtime-4.1_2.13-1.11.0.jar` bytecode and the `apache-iceberg-1.11.0` source):

- `SizeBasedFileRewritePlanner.expectedOutputFiles(inputSize)`: `input < target ⇒ 1`; else
  `withRemainder = ceil(input/target)`, `withoutRemainder = floor(input/target)`,
  `avg = input/withoutRemainder`; `input % target > minFileSize ⇒ withRemainder`, else
  `avg < min(1.1*target, writeMaxFileSize) ⇒ withoutRemainder`, else `withRemainder`.
- `SizeBasedFileRewritePlanner.inputSplitSize(inputSize)`:
  `estimated = input/expected + SPLIT_OVERHEAD` (`SPLIT_OVERHEAD = 5*1024 = 5,120` — bytes,
  not the "5 MB" the row shorthand said); `estimated < target ⇒ target`, else
  `min(estimated, writeMaxFileSize)`.
- `SizeBasedFileRewritePlanner.writeMaxFileSize()`:
  `(long)(target + (max - target) * 0.5)` — here `2000 + (3600-2000)*0.5 = 2800`.
- `SparkBinPackFileRewriteRunner.doRewrite`: `DataFrameReader.option("split-size",
  group.inputSplitSize())` and `option("file-open-cost", "0")`; writer
  `option("target-file-size-bytes", group.maxOutputFileSize())` where
  `maxOutputFileSize = writeMaxFileSize()` (set in `BinPackRewriteFilePlanner.newRewriteGroup`).
- `TableScanUtil.planTaskGroups`: split each task at `splitSize`, pack with
  `BinPacking.PackingIterable` at `lookback = read.split.planning-lookback` (default **10**),
  `largestBinFirst = true` (bytecode `iconst_1`), weight `task.weight(openFileCost)` =
  `max(length + deleteBytes, (1 + deletes) * openFileCost)`; merge contiguous same-file splits.
- `SizeBasedFileRewritePlanner.inputSize(group) = Σ ContentScanTask.length`.
- `BaseFileScanTask.sizeBytes() = length + deletesSizeBytes`; `BinPacking.ListPacker` for group
  planning stays lookback-1, `largestBinFirst = false` — unchanged, already mirrored by
  `pack_bins`.

Per cell, on the fork's 1,694 B files (target 2,000 ⇒ min 1,500, max 3,600, writeMax 2,800):

| cell | grouping | inputSplitSize | read tasks | Java answer |
|---|---|---|---|---|
| `target_small` | 2 groups of 4 (6,776 B each) | `expected(6776)=4` (mod 776 ≤ min 1500; avg 2258 ≥ 2200 ⇒ withRemainder); `6776/4+5120=6814` → clamp 2800 | 1694×2=3388 > 2800 ⇒ 1 file/task ⇒ 4 per partition | **8 added, 1 commit** |
| `max_group_size` | 8 groups of 1 (1,694+1,694=3,388 > 2,500) | `expected(1694)=1`; `1694+5120` → clamp 2800 | 1 task per group | **8 added, 1 commit** |
| `partial_progress_groups` | same 8 groups | same | same | **8 added, 3 commits** (`ceil(8/3)=3`/commit) |

Fork answers before the fix: `target_small` **2**, `max_group_size` 8, `partial_progress` 8/3
commits. So `max_group_size` and `partial_progress` **already answer Java's count on the
fork's file sizes** — their RePark deltas are file-size differences (1,694 vs Spark's 1,153 B),
not planning defects, exactly as the brief predicted for files > 1,250 B. `target_small`
diverges: 2 vs Java's 8. **The defect is real on one cell** → proceed.

## 3. Defect

`write_compacted_files` (`crates/iceberg/src/maintenance/rewrite_data_files_write.rs`) streamed
each rewrite group through ONE writer chain rolling at `target_file_size_bytes`. Java instead
reads each group through `TableScanUtil.planTaskGroups` at `inputSplitSize(group input)` and
writes each resulting read task's rows through its own writer rolling at `writeMaxFileSize()`.
On small parquet the fork's buffered bytes never crossed the target before close, so one group
produced one file — 8→2 where Java gives 8→8 on the same input.

## 4. Fix

- `rewrite_data_files_plan.rs` gained the Java formulas: `SPLIT_OVERHEAD`,
  `write_max_file_size`, `expected_output_files`, `input_split_size`, and `plan_read_tasks`.
  `plan_read_tasks` splits each task (`FileScanTask::split`, honoring row-group
  `split_offsets`), packs the splits through the existing parameterized
  `crate::scan::bin_pack::PackingIterator` (`scan/mod.rs`: `mod bin_pack` → `pub(crate) mod
  bin_pack`) at lookback 10 (`PROPERTY_SPLIT_LOOKBACK_DEFAULT`), `largestBinFirst = true`,
  weight `task.weight(0)` (open cost 0), then `merge_tasks` — the same port `TableScan::plan_tasks`
  uses; no second bin-packer was written.
- `rewrite_data_files_write.rs`: `write_compacted_files(table, group, config, output_spec)`
  (signature now takes `&ResolvedConfig`; call sites in `rewrite_data_files.rs` and
  `rewrite_data_files_router_bound_tests.rs` updated). The rolling writer target is
  `writeMaxFileSize`; the group is planned into read tasks and each read task gets its own
  reader stream and writer chain — sorted path still sorts and stamps `sort_order_id` per
  output file, partition routing still bounded, lineage/delete handling unchanged.
- Bounded memory preserved: one read task's stream at a time, the sorted-path run buffer
  bounded by `write_max` per task (previously by target per group), the partition router's
  open-writer bound unchanged; `peak_open_partition_writers` is now the max across read tasks.
- `rewrite_data_files.rs` shrank 2,449 → 2,440 (signature simplification); the legacy ceiling
  in `scripts/check_rust_file_size.py` was lowered to match (ceilings only move down).
- Lineage note: `FileScanTask::split` branch (1c) keeps `_pos`/`_row_id`-projecting tasks
  whole, so V3 lineage-carrying tasks are never split — each lands as a whole unit in the
  packing (spill bin if oversized), matching the reader's whole-file requirement.

## 5. Red → green → mutation

- RED cells (commit `34dea9bb`):
  - `rewrite_data_files_options_tests::test_target_small_output_count_follows_java_read_splits`
    — RePark shape on fork-written files; prints measured sizes and computes the expected count
    from `input_split_size` + `plan_read_tasks`. **RED:** `left: 2, right: 8`.
  - `rewrite_data_files_plan_tests::test_plan_read_tasks_spark_sizes_pack_two_files_per_split`
    — synthetic planner cell on Spark's measured sizes (8 × 1,153 B, 2 partitions, target
    2,000): asserts split 2,800, 2 read tasks per partition, 4 total. Green at RED commit —
    it pins the new planning function, not the write path.
  - `rewrite_data_files_plan_tests::test_plan_read_tasks_default_target_is_one_task_per_group`
    — control: default 512 MiB target ⇒ 1 read task per group. Green at RED commit (control).
- GREEN (commit `d635ccf9`): `cargo test -p iceberg --lib rewrite_data_files` — **99 passed, 0
  failed**; `test_target_small_output_count_follows_java_read_splits` reports
  `added_data_files_count = 8` (= the Java formula's answer on the measured sizes).
- MUTATION: fix-commit files restored to `34dea9bb` (uncommitted) →
  `test_target_small_output_count_follows_java_read_splits` **FAILED** with the identical
  signature (`left: 2, right: 8`); restored to `d635ccf9` → 99/99 green. Revert not committed.
  The planner-level cells stay green under the revert by construction (they exercise
  `plan_read_tasks`, which the revert leaves in place); the E2E cell is the load-bearing one
  and it is the one that went red.

## 6. RePark xfail prognosis

On RePark-written (fork-writer) files ≈1,694 B, the fork now answers **8 added / 8 added /
8 added + 3 commits** — Java's own answers on those inputs. Spark measured **4 / 4 / 4 + 2
commits** on Spark-written ≈1,153 B files. All three xfails therefore **stay**: the residual
divergence is input parquet size, not planning. If RePark is ever pointed at Spark-written
files, all three flip — the synthetic planner cell already pins the 1,153 B ⇒ 4 answer.

## 7. Out of scope observed

- The fork's parquet writer emits ≈47% larger files than Spark's for this shape (1,694 vs
  1,153 B). That is a writer-compression property, not a rewrite-planning property; left
  unchanged.
- `pack_bins` (group planning) stays a separate lookback-1 packer rather than being re-expressed
  through `PackingIterator`; behavior identical, refactor not needed for this lane.

## 8. Round 2 — review follow-ups (fork #302)

Round 1 was accepted, rebased onto fork `main` `9f36da97` (replay ids `34dea9bb`,
`d635ccf9`, `7c4b45db` — corrected throughout this ledger), and pushed as fork #302. The
Grok logic review passed; the perf review returned two P2s and one P3 (plus one ledger-only
P3). This round lands all of them.

### 8.1 Perf R-01 (P2) — footer map shared, not cloned per read task

Pre-fix, `write_compacted_files` built a fresh `ArrowReader` inside the read-task loop and
passed `input_footers.clone()` — an O(group files) `HashMap` clone per read task (the
reviewer's arithmetic: 100k one-file tasks × a 100k-entry map). Fix: `ArrowReader`'s
`prefetched_parquet_metadata` field is now `Arc<HashMap<Arc<str>, Arc<ParquetMetaData>>>`
(`arrow/reader.rs`; `ArrowReaderBuilder::build` wraps the builder's map in `Arc`), and the
write path builds ONE reader per group (`reader.rs` `read(self)` still consumes `self`, so
the loop calls `reader.clone().read(task_stream)` — the clone is now O(1): `FileIO`,
`CachingDeleteFileLoader`, `ParquetReadOptions` (Copy) and the footer `Arc`). The builder's
public `with_prefetched_parquet_metadata(HashMap)` signature is unchanged; the write path
was its only caller.

### 8.2 Perf R-02 (P2) — one delete loader per group, not per read task

The same hoist fixes the loader regression the read-split planning introduced: a new
`ArrowReaderBuilder` per read task meant a new `CachingDeleteFileLoader` per task, so a
partition-scoped delete that loaded once per GROUP loaded once per read TASK.
`CachingDeleteFileLoader`'s `delete_filter` is `Arc<RwLock<DeleteFileFilterState>>` — clones
share the claim/loaded state — so `reader.clone()` per read task gives every task the same
cache while `DeleteFilter::resolve_delete_vector` still scopes application to each task's
own delete list. No behavior change; the count below is the proof.

Proof cell `rewrite_data_files_delete_loader_tests.rs` (new `#[cfg(test)]` sibling wired in
`maintenance/mod.rs`; the counting `Storage`/`StorageFactory` wraps `LocalFsStorage` and
records `Storage::reader(path)` per path — the one seam every delete-file load crosses,
via `BasicDeleteFileLoader` → `open_parquet_file` → `new_input().reader()`, exactly once
per load): V2 table, 4 × 50-row files (2,104 B each) in partition `x=0`, one
partition-scoped equality delete on `y`. `input_split_size` on the measured sizes is
`8416/4 + 5120 →` clamp 2,800 < 2 × 2,104 ⇒ **4 read tasks**. The counter is cleared after
the fixture's pre-rewrite MoR scan so only rewrite loads are measured. Asserts:
`added_data_files_count = 4`, `reader(eq-del path) == 1`, and `scan_rows` after == before
(199 live rows — the delete drops `y=20`).

- MUTATION R-02: per-task `ArrowReaderBuilder` restored inside the loop (uncommitted;
  `input_footers.clone()` kept so only the loader sharing is reverted) → the cell **FAILED**
  `left: 4, right: 1` — one load per read task, the exact regression signature. Restored →
  green.

### 8.3 Logic L-001 (P3) — planner cells for the remainder rule and unclamped split size

The E2E fixture always clamps `input_split_size` to `writeMaxFileSize`, so it cannot catch a
wrong remainder rule. Two planner-level cells were added in
`rewrite_data_files_plan_tests.rs`:

- `test_expected_output_files_remainder_rule_cells` — the reviewer's nine cases at
  target 2,000 / min 1,500 / max 3,600 (writeMax 2,800): 6776→4, 4612→3, 1999→1, 2000→1,
  4000→2, 18000→9, 3500→2, 4398→2, 4400→3 — **plus 17500→8**, the discriminating case
  explained below.
- `test_input_split_size_between_target_and_write_max` — with a larger target (1,000,000,
  min 750,000, max 1,800,000 ⇒ writeMax 1,400,000), two inputs whose
  `input/expected + 5120` lands strictly inside (target, writeMax): 10,500,000 → expected 10
  → split **1,055,120**; 2,994,000 → remainder 994,000 > min ⇒ expected 3 → split
  **1,003,120**. Both asserted exactly AND asserted `target < split < write_max`.

- MUTATION L-001: `input_size % target > min_file_size` flipped to `>=` →
  `expected_output_files(17500)` **FAILED** `left: 9, right: 8`. Restored → green.
  **Correction to the review premise:** the review expected the *3500* case to redden. It
  cannot: 3500's remainder (1,500) equals `min_file_size`, so `>=` takes the first branch —
  but 3500's else-branch *also* returns `withRemainder = 2` (avg-without 3,500 ≥ 2,200), so
  the operator flip is invisible there. The discriminating input needs `rem == min` AND
  `avg_without < min(1.1·target, writeMax)` — i.e. `n·2000 + 1500` with `1500/n < 200`, first
  satisfiable at `n = 8` → 17,500: `>` gives `without = 8` (avg 2,187.5 < 2,200), `>=` gives
  `with = 9`. That cell is what the mutation killed.

### 8.4 Logic L-002 (P3 residual, no code change)

`plan_read_tasks` packs read splits at `PROPERTY_SPLIT_LOOKBACK_DEFAULT` (**10**) and does
not read `read.split.planning-lookback` from table properties; Java's Spark runner honors
that property in `TableScanUtil.planTaskGroups`. Recorded as a residual — the default
matches Java's own default, so behavior diverges only when a table overrides the property.

### 8.5 Round-2 residual notes

- The exactly-once guarantee is a property of the shared `DeleteFilter` state inside the
  cloned `CachingDeleteFileLoader` — the load-count cell is the regression alarm if loader
  cloning ever stops sharing that `Arc<RwLock>` state.
- Gates after round 2: `cargo fmt --all`, `cargo clippy -p iceberg -p iceberg-datafusion
  --all-targets -- -D warnings`, `python3 scripts/check_rust_file_size.py`, and
  `cargo test -p iceberg --lib rewrite_data_files` → **102 passed, 0 failed**.
