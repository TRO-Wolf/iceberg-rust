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

# Ledger — F-FORK-ASKS-22A: three RePark fork asks (hour on ns, COW UPDATE sort+stamp, binpack sort+stamp)

**Ledger id:** `F-FORK-ASKS-22A-2026-09-17`
**Branch:** `fix/fork-asks-22a` (cut off fork `main`)
**Scope:** three asks, one commit group per ask, smallest first; one ledger for the lane
**Model:** swe-2-high

| Item | Tag | Commits | Subject |
|---|---|---|---|
| 1 | F-TSNS-HOUR-1 | `52bb9f387` / `9aba4ebe5` | red cells, then `Hour::transform` accepts nanosecond timestamp arrays |
| 2 | F-COW-UPDATE-STAMP-1 | `b4851f80b` / `7e945ef4b` | red cells, then the COW UPDATE rewrite sorts by the default order and stamps `sort_order_id` |
| 3 | F-RDF-SORT-STAMP-1 | `165760ab0` / `3ffe3b92e` | red cells, then binpack output is sorted by the default order and stamped |
| 4 | ledger | this commit | this file + `task/todo.md` entry |

## Item 1 — F-TSNS-HOUR-1: the `hour` transform accepts nanosecond timestamps

### Defect

`Hour::transform` in `crates/iceberg/src/transform/temporal.rs` matched only
`DataType::Timestamp(TimeUnit::Microsecond, _)`; an array transform on a `timestamp_ns` /
`timestamptz_ns` column refused with
`FeatureUnsupported => Unsupported data type for hour transform: Timestamp(Nanosecond, _)`. The
literal arm (`transform_literal`) and the Day/Month/Year array transforms already accepted
nanoseconds, so a write into a table partitioned by `hours(ts)` on an ns column failed while a
literal partition computation succeeded.

### Java/spec answer

`hour = floor(ns / 3_600_000_000_000)` — floor (Euclidean), not truncation, so pre-epoch values
round down: `-1ns → -1h`, `-3_600_000_000_001ns → -2h`. Timezone metadata does not change the
arithmetic.

### Fix

Added the `Timestamp(Nanosecond, _)` arm to `Hour::transform`, mapping to
`hour_timestamp_nano` (`div_euclid(NANOSECONDS_PER_HOUR)`), mirroring the microsecond arm and the
`Day` transform's structure. `temporal.rs` sat exactly on its 2796-line legacy ceiling, so the two
identical unsupported-type error constructions in `Hour`'s `_` arms were deduplicated into
`Hour::unsupported` (net −4 lines → 2792) and the sanctioned ceiling in
`scripts/check_rust_file_size.py` was lowered to match (the script's prescribed move for a
shortened file). Unsupported types still error; nulls pass through.

### Red → green → mutation

- RED `test_hour_transform_accepts_nanosecond_timestamp_arrays` (in `transform/mod.rs`'s test
  module — `temporal.rs` was at its ceiling): `Timestamp(Nanosecond, None)` and
  `Timestamp(Nanosecond, Some("UTC"))` arrays over 0, 3_599_999_999_999, 3_600_000_000_000, −1,
  −3_600_000_000_001, and null, asserted equal to `transform_literal` on the same values.
  Pre-fix: `FAILED — FeatureUnsupported => Unsupported data type for hour transform:
  Timestamp(Nanosecond, None)`.
- GREEN: same test passes; `cargo test -p iceberg --lib hour` 6/6.
- MUTATION: `temporal.rs` restored to `52bb9f387` (no ns arm) → test red with the same
  `FeatureUnsupported`; restored to `HEAD` → green. Reverts not committed.

## Item 2 — F-COW-UPDATE-STAMP-1: the COW UPDATE rewrite sorts by the default order and stamps it

### Defect

`copy_on_write_update` in `crates/integrations/datafusion/src/physical_plan/delete.rs` rewrote
data files through `StreamingDataFileWriter` with neither a default-order sort nor
`with_sort_order_id`: output kept scan order and carried `sort_order_id = None`. The INSERT path
already did both (`physical_plan/write.rs` via `sort.rs::write_sort_plan`).

### Measured Spark 4.1.2 answer

On `(id BIGINT, p INT) PARTITIONED BY (p)` with `WRITE ORDERED BY (id)`,
`UPDATE t SET id = id WHERE p = 0` (and `SET id = 42 WHERE p = 0`) rewrites files SORTED by id,
stamped `sort_order_id = 1`. A float sort field orders `NULLS FIRST` per the field's null order,
values ascending, NaN last.

### Fix

The sort+stamp was added inside the shared `StreamingDataFileWriter`
(`physical_plan/row_lineage.rs`), which is the writer every DML rewrite path funnels through —
`delete.rs` itself needed no change beyond what it already had. The writer clones the `Table`,
computes `write_sort_plan` lazily from the first prepared batch's schema (the plan needs the
arrow schema), buffers batches only when the plan carries sort expressions, and at `finish()`
concatenates, evaluates the `PhysicalSortExpr`s, `lexsort_to_indices` + `take` for the ordered
batch, and writes it through the normal `TaskWriter` with `with_sort_order_id` applied.
Unsorted tables (and unresolvable plans, per `write_sort_plan`'s existing fallbacks) stream
unchanged and stamp `0`, so `tests/cow_memory_bound.rs`'s no-live-row-buffering invariant still
holds.

### Audited sibling paths (same writer ⇒ same behavior)

| Path | Verdict |
|---|---|
| `copy_on_write_update` (delete.rs ~1070) | sorted + stamped — the ask |
| `copy_on_write_delete` survivor files (delete.rs ~620) | sorted + stamped via the same `StreamingDataFileWriter` |
| `merge_on_read_update` new-row files (delete.rs ~922) | sorted + stamped via the same writer |
| `merge_on_read_delete` / `write_position_deletes` | writes position-delete files only — no data files, stamp N/A |

### Red → green → mutation

- RED (in `tests/sorted_insert.rs` — `delete_tests.rs` at 986/1000 and `commit_tests.rs` at 873
  lacked headroom; this module owns the sort+stamp harness):
  `cow_update_rewrites_file_sorted_by_default_order_and_stamps_it` (output `[5,1,4,2,3]` vs
  `[1,2,3,4,5]`), `cow_update_desc_nulls_last_orders_nulls_last` (unsorted), and
  `cow_update_on_unsorted_table_stamps_zero_and_keeps_scan_order` (`None` vs `Some(0)`).
- GREEN: all 3 pass; datafusion `--lib` 149 passed, `cow_memory_bound` 1 passed,
  `sorted_insert` file 16/16, promoted-type DML tests 4/4.
- MUTATION: `row_lineage.rs` restored to `b4851f80b` → all 3 red with the original failures;
  restored to `HEAD` → green. Reverts not committed.

## Item 3 — F-RDF-SORT-STAMP-1: binpack `rewrite_data_files` output sorted and stamped

### Defect

`write_compacted_files` in `crates/iceberg/src/maintenance/rewrite_data_files_write.rs` built
`DataFileWriterBuilder::new(rolling_builder).with_partition_spec(spec)` — no sort, no
`with_sort_order_id` — so binpack output preserved input order and carried no order id.

### Measured Spark 4.1.2 answer

On a table with `WRITE ORDERED BY (id)`, binpack re-sorts each OUTPUT file by the table's default
order and stamps it: 6 sorted 100-row inputs compact to 2 files, `sort_order_id = 1`, id ascending
within each (heads `[0,1,2,6,7,8]` and `[31,32,33,37,38,39]`).

### Fix

Core `iceberg` cannot depend on DataFusion, so `rewrite_sort_plan` mirrors
`physical_plan/sort.rs::write_sort_plan` against plain arrow: unsorted order → no keys, stamp `0`;
unresolvable field/transform/order-id → the same silent degrade to no-keys + stamp `0` the INSERT
path performs; all-Void field list → no keys but stamps the order id. Each sort field resolves its
source column (top-level name + nested struct path), applies non-identity transforms through
`create_transform_function`, canonicalizes NaN payloads on float keys (matching
`CanonicalFloatExpr`), and maps direction/null-order onto `SortOptions`. When the plan has keys,
the group's batches are concatenated once, `lexsort_to_indices` + `take_record_batch` produce one
sorted batch, and that batch flows through the unchanged downstream (single writer, or splitter +
`BoundedPartitionRouter` — a partition's slice of a globally sorted batch is itself sorted, and
the rolling writer still rolls at `ROWS_DIVISOR` boundaries, so each output file is a sorted
range). `with_sort_order_id(sort.stamp)` stamps every produced file, including `0` for unsorted
tables.

Memory bound: the collect happens only when a real sort key exists, and holds one file group's
rows — the unit the binpack planner already materialises and hands to `write_compacted_files` per
call. Unsorted tables keep the streaming path byte-for-byte; the partitioned router's open-writer
bound is unchanged (each partition is written once, so evicted writers are never reopened).

### Audited sibling paths

| Path | Verdict |
|---|---|
| `write_compacted_files` (binpack data files) | sorted + stamped — the ask |
| `rewrite_position_delete_files` | `PositionDeleteFileWriterBuilder`, `PositionDeletes` content — not data files, stamp N/A |
| `convert_equality_delete_files` | equality-delete content — not data files, stamp N/A |
| Unpartitioned output branch | sorted identically (single writer over the sorted batch) |

### Red → green → mutation

- RED (in `rewrite_data_files_lineage_tests.rs`, 176→640 lines — it owns "what rewritten output
  files carry"; `options_tests` at 819 lacked headroom):
  `binpack_output_files_sorted_by_default_order_and_stamped` (3×700-row files, union unsorted,
  `target_file_size_bytes(1)` forcing rolled output; `None` vs `Some(1)`),
  `binpack_desc_nulls_last_orders_nulls_last`,
  `binpack_float_sort_places_nan_last_and_honors_nulls_first` (unpartitioned; NaN sorts after all
  values, before NULLS-last / after NULLS-first), and
  `binpack_unsorted_table_stamps_zero_and_keeps_row_union` (`None` vs `Some(0)`).
  All 4 failed on `sort_order_id == None`.
- GREEN: all 4 pass; `cargo test -p iceberg --lib rewrite_data_files` 96/96.
- MUTATION: `rewrite_data_files_write.rs` restored to `165760ab0` → all 4 red on the missing
  stamp; restored to `HEAD` → green. Reverts not committed.

## Gates

| Gate | Result |
|---|---|
| `cargo test -p iceberg --lib binpack_` | 4/4 |
| `cargo test -p iceberg --lib rewrite_data_files` | 96/96 |
| `cargo test -p iceberg --lib hour` | 6/6 |
| `cargo test -p iceberg-datafusion --test sorted_insert` | 16/16 |
| `cargo test -p iceberg-datafusion --lib` | 149 passed, 1 ignored |
| `cargo test -p iceberg-datafusion --test cow_memory_bound` | 1/1 |
| `scripts/check_rust_file_size.py` | `temporal.rs` ceiling lowered 2796→2792 alongside the shrink; clean |
| `cargo fmt --all` / `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings` | see close-out run |
