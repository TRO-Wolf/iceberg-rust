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

# map.md — crates/iceberg/src/scan/

## Purpose

Table-scan planning and execution (Java `SnapshotScan` / `DataTableScan` / `BaseFileScanTask`):
snapshot → manifest list → manifest pruning → `FileScanTask` stream → Arrow record batches (the
Arrow read itself lives in `../arrow/`). Includes merge-on-read delete application inputs, residual
propagation, and opt-in scan-metrics reporting.

## Contents

| File | What it does |
|---|---|
| `mod.rs` | `TableScanBuilder` (filter / **`with_file_prune_only`** metrics-only prune without residual / select / snapshot_id / use_ref / concurrency limits / **within-file read parallelism** (split_offsets expand in `to_arrow`) / range-fetch knobs / row-group filtering / row selection / `with_metrics_reporter` / `with_split_size`·`with_split_lookback`·`with_split_open_file_cost`) + `TableScan::plan_files` (**FK2.2:** delete-entry + data-entry streams run concurrently; data parks on `DeleteFileIndex` `Notify` until the index is published — no correctness barrier) + `TableScan::plan_tasks` (split + bin-pack, Java `Scan.planTasks`) + `to_arrow` |
| `partition_work.rs` | Multi-partition SELECT seam: `PartitionWork` / `FileScanTaskId` / `ScanFilterMode` + `assign_partition_work` (strip empties + fixed-T round-robin) + `stream_partition_work` (legacy single-stream ArrowReader; **no within-file expand** — the `to_arrow` expand does not apply on this path) + `TableScan::plan_partition_work`. Cover at task-range grain `(path,start,length)`. Wired by `iceberg-datafusion` `IcebergTableScan::plan` |
| `batch.rs` | `BatchScan` (Java `BatchScan` / `BatchScanAdapter`) — the typed adapter built by `Table::batch_scan()`. Wraps a `TableScanBuilder`; adds the time-travel selectors `use_snapshot` / `use_ref` / `as_of_time` (Java `SnapshotScan` selectors, FIRST-WINS mutual exclusion; `as_of_time` = greatest `timestamp_ms <= ms` over `history()`, Java `SnapshotUtil.snapshotIdAsOfTime`); re-exposes the inherited filter/select/case-sensitive/split knobs; `plan_files`/`plan_tasks` DELEGATE 1:1 to the `TableScan` pipeline (no reimplemented split/bin-pack) |
| `context.rs` | `PlanContext` / `ManifestFileContext` / `ManifestEntryContext`: per-manifest evaluator caches (manifest evaluator, partition filter, residual evaluator built per manifest), `into_file_scan_task` (evaluates the **partition-reduced residual** per file via `ResidualEvaluator::residual_bound_for` with per-partition memo, Java `residuals.residualFor(file.partition())`; **Arc-clones** the memoized residual onto the task — no deep clone of the tree (FK2.1); threads `split_offsets` from the manifest entry onto the task). Partition filters use the scan-level `snapshot_bound_predicate` (no re-bind per manifest); `BoundPredicates` fields are `Arc` shared trees |
| `task.rs` | `FileScanTask` (+ delete-file attachments for merge-on-read, + flagged-additive `split_offsets`); **Arc-shared innards** (`data_file_path: Arc<str>`, `project_field_ids: Arc<[i32]>`, `predicate: Option<Arc<BoundPredicate>>`, `deletes: Arc<[FileScanTaskDeleteFile]>`) so `split`/`sub_task` pointer-clone path/projection/residual/deletes (FK2.1; JSON serde still emits plain string/array shapes); `FileScanTask::split(target)` (Java `BaseContentScanTask.split`: non-splittable / offsets-aware / fixed-size) + `weight(open_file_cost)` (Java `lambda$planTasks$3` = `max(length+deleteBytes, (1+#deletes)*openFileCost)`) |
| `task_group.rs` | `ScanTaskGroup` trait + `CombinedScanTask` (`Vec<FileScanTask>`; `size_bytes` = Σ lengths, `files_count`) — the `plan_tasks` group output (Java `ScanTaskGroup` / `CombinedScanTask`). `CombinedScanTask::new` MERGES adjacent contiguous same-file splits via `task.rs::merge_tasks` (Java `BaseCombinedScanTask(List)` ctor → `TableScanUtil.mergeTasks`), so contiguous byte-range splits of one file sharing a bin become ONE spanning member. Cross-engine pinned by the `merge.parquet` / `gap.parquet` fixtures in `run-interop-scan-plan.sh` (both directions, non-vacuity asserted, + a production-source merge-removal mutation leg) |
| `bin_pack.rs` | `PackingIterator` — a faithful Java `BinPacking.PackingIterable` port (FIFO `findBin`, `largestBinFirst` eviction, drain FIFO); self-contained + unit-testable |
| `cache.rs` | Per-partition-spec evaluator caches shared across one scan (`PartitionFilterCache` / `ManifestEvaluatorCache` / `ExpressionEvaluatorCache`). `PartitionFilterCache::get` takes `&BoundPredicate` (bind once at scan build; project only on miss). A poisoned `RwLock` is RECOVERED (`PoisonError::into_inner`, the crate-wide policy) — the values are pure derived data, so one panicked task cannot brick subsequent scans (SAF-003 hardening, 2026-07-18; previously `map_err(..).unwrap()` aborted) |
| `incremental.rs` | `IncrementalAppendScan` (Java `BaseIncrementalAppendScan`: appended files in `(from, to]`, append-only walk) + `IncrementalChangelogScan` (Java `BaseIncrementalChangelogScan`: `ChangelogScanTask`s with oldest→0 ordinals; default = Java 1.10.0 data-file changelog, REJECTS delete-manifest ranges; opt-in ENGINE-FIRST `with_row_level_deletes(true)` emits the Java-api row-level taxonomy — AddedRows fold / DeletedDataFile / DeletedRows with added-vs-existing delete split — beyond what 1.10.0 core implements) |
| `metrics_collector.rs` | `ScanMetricsCollector` (Arc'd `AtomicI64`) — opt-in; report emitted ONCE on full stream consumption |

## I want to...

| I want to... | go to |
|---|---|
| Touch split / bin-pack grouping (`plan_tasks`) | `task.rs::split` + `weight` (Java `BaseContentScanTask.split` / `lambda$planTasks$3`), `bin_pack.rs` (Java `BinPacking.PackingIterable`), and `mod.rs::plan_tasks` (drives `plan_files` UNCHANGED, then split + pack). The split target/lookback/open-file-cost resolve from table props + scan-option overrides (Java `BaseScan.targetSplitSize`/`splitLookback`/`splitOpenFileCost`). Cross-engine pinned by `tests/interop_scan_plan.rs` + `dev/java-interop/run-interop-scan-plan.sh` |
| Parallel reads inside one multi-RG Parquet file | `TableScanBuilder::with_within_file_read_parallelism` (default on) + data-file concurrency > 1: `to_arrow` expands whole-file tasks with `split_offsets` into concurrent ranged sub-tasks. Column-chunk range fetch: `with_range_fetch_concurrency` / `ArrowFileReader::get_byte_ranges`. Suppressed when projecting `_pos`. |
| Change manifest/file pruning | `context.rs` (evaluator construction + the prune point) and [../expr/visitors/map.md](../expr/visitors/map.md) for the evaluators themselves |
| File prune without residual (COW MERGE) | `TableScanBuilder::with_file_prune_only` — plan-time partition + inclusive-metrics prune only; `PlanContext.apply_residual_filter = false` so tasks carry `predicate: None` (survivors kept). Do **not** use `with_filter` for COW target scans |
| Multi-partition DF SELECT (`plan_tasks` → N) | `partition_work.rs` + DF `IcebergTableScan::plan` — fixed-T RR assignment; knobs `iceberg.multi_partition_scan` / `iceberg.data_file_concurrency` (`T`/`L`/`P`) |
| Touch residual handling | `context.rs` + `../expr/visitors/residual_evaluator.rs` — residual evaluator built per manifest from the file's spec; `residual_bound_for` evaluates per partition, binds back to the snapshot schema, and **memoizes by partition** so co-partition files share one residual (skipped when `apply_residual_filter` is false) |
| Add a scan metric | `metrics_collector.rs` + the count sites in `mod.rs`/`context.rs`; populated-vs-`None` counters are documented in `../metrics/mod.rs` |
| Understand merge-on-read reads | `task.rs` delete attachments → applied in `../arrow/` (delete_file_index / delete_vector at crate root). **FK2.3:** `../delete_file_index.rs` keys partition maps by nested `(spec_id → partition)` and seq-sorts lists once so lookup can `partition_point` the applicable tail (wrong key = delete resurrection; multi-spec identical-result pins; zero-clone `get(partition)` hot path) |
| Touch the changelog task taxonomy or row-level changelog mode | `task.rs` (`ChangelogScanTask` / `ChangelogTaskKind` / `ChangelogOperation` — Java api split, `operation()` derived from kind) + `incremental.rs` (`plan_snapshot_change_tasks` / `plan_deleted_rows_tasks` / `build_snapshot_delete_indexes`; the default-mode rejection guard lives in `ordered_changelog_snapshots` and must keep matching Java 1.10.0 — mutation-pinned) |

## Pointers

- **Up:** [crates/iceberg/src/](..) · **Related:** [../expr/visitors/map.md](../expr/visitors/map.md)
  (evaluators), `../arrow/` (batch reading + delete application), `../metrics/` (report model),
  [../../tests/map.md](../../tests/map.md) (`interop_scan_exec.rs` — the data-level MoR interop)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Task carries the full snapshot filter instead of a reduced one | The residual must be evaluated against the file's partition in `into_file_scan_task` — mutation-pinned; check the per-manifest evaluator wiring |
| Identity-partition constants wrong / type errors in record batches | The `PartitionUtil.constantsMap` constant-materialization path is **active**: each task carries its manifest's `partition_spec` (`context.rs::create_manifest_file_context`), and `arrow/record_batch_transformer` materializes identity-partition columns as PLAIN arrays of the declared scan-schema type (never REE) coerced to the column's Iceberg type (`Datum::to`). A constant field present in the file is forced down the `Modify` path so the constant OVERRIDES the file column (`constant_overrides_file_column`). Wrong value ⇒ check the spec/partition threading and `constants_map`; type/REE error ⇒ check the plain-array + `Datum::to` coercion in the transformer |
| Metrics report emitted on a dropped stream / per task | The report fires ONCE on full consumption (`None` from the stream); early-drop emits NOTHING (pinned by test) |
| No-reporter path changed | With no reporter there must be **no collector, no timer, no wrapper** — the plan path is byte-unchanged (structural test pins this) |
| `plan_tasks` groups diverge from Java | Check the WEIGHT (`task.rs::weight` — delete bytes via `content_size_in_bytes`: DV→blob size, else file size; floor `(1+#deletes)*openFileCost`), the SPLIT branch order (offsets-aware IGNORES target; non-splittable = Puffin only), the bin-pack eviction (`largestBinFirst`, ties → first-inserted; drain FIFO), AND the adjacent-split MERGE (`task.rs::merge_tasks` in `CombinedScanTask::new` — Java `mergeTasks` in the `BaseCombinedScanTask(List)` ctor). EVERY split sub-task inherits the parent deletes (so deletes are charged per sub-task). If ONE file shows FEWER members on Java than Rust, suspect the merge FIRST: Java coalesces contiguous same-file splits that co-bin (a merged span's start is the run's first start + total length, so it masquerades as an offset SUBSET — decompose the lengths, don't compare starts as a set). Pinned by `bin_pack.rs`/`task.rs`/`task_group.rs` unit tests + the `interop_scan_plan` oracle |
| Wrong rows after partition evolution | The residual (and any per-file logic) must use the FILE's own spec (`partition_spec_by_id(manifest.partition_spec_id)`), never `default_partition_spec()`. A single-spec fixture is structurally BLIND to this swap — use the 2-spec `new_with_evolved_default_spec` fixture to pin it |
| MoR plan hangs forever on delete index | Data tasks park on `DeleteFileIndex` `Notify` (FK2.2 overlap). Lost-wakeup if `Notified` is armed AFTER the read lock (upstream #2696 class); failed-populate hang if state stays `Populating` without `Failed` terminal. Pins: `test_fk2_2_*` + SAF-007 suite in `delete_file_index.rs` |
| Deletes resurrect after partition-spec evolution | Partition maps must key by `(spec_id, partition)` (FK2.3), not `Struct` alone — same partition values under two specs must not cross-attach. Pins: `test_fk2_3_multi_spec_*` |

### First checks

- Reproduce with a single-manifest fixture; check which layer prunes (manifest evaluator vs
  partition filter vs metrics evaluator) before touching code.
- For wrong-row results on MoR tables, confirm whether the bug is planning (here) or delete
  application (`../arrow/`) by scanning without deletes.

### Escalate to

- Evaluator semantics → [../expr/visitors/map.md#debug](../expr/visitors/map.md#debug).
- Cross-engine result divergence → the scan-exec interop harness,
  [dev/java-interop/map.md#debug](../../../../dev/java-interop/map.md#debug).
