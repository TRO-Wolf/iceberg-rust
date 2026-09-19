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

# F-PAGE-PRUNE-2 — a non-pruning predicate must not pay page-selection cost

Scope: the RePark bench (run 24a) measured a filtered scan whose predicate can prune no
page getting slower after F-PAGE-PRUNE-1: `category = 'cat_7'`, warm 659 → 776 ms, cold
834 → 1,000 ms. Page bytes read are identical before and after (1,428,284,966 B), and the
page-index bytes were already fetched under the old `Required` policy. The regression is
CPU inside the reader. Target: ON/OFF ratio ≤ 1.03 for the non-prunable query, with a
prunable query keeping its win.

## Mechanism (pre-change reading)

`ArrowReader::process_parquet_file_scan_task` (`crates/iceberg/src/arrow/reader.rs`)
hands the evaluated `RowSelection` to `ParquetRecordBatchStreamBuilder::with_row_selection`
unconditionally. In parquet-rs 58.4 (`arrow/in_memory_row_group.rs::fetch_ranges`), the
mere presence of a selection plus an offset index switches the column-chunk fetch from
`Dense` (one range per column) to `Sparse`: `selection.scan_ranges(page_locations)` emits
one range per page, `expand_to_batch_boundaries` walks the selectors, `merge_ranges` then
coalesces the contiguous pages back into one fetch (hence identical page bytes), and
`ColumnChunkData::Sparse` serves every page read through a `binary_search` over a
`Vec<(usize, Bytes)>` instead of a single slice offset. An all-keep `RowSelection` pays all
of this and prunes nothing. parquet-rs itself documents the invariant in
`read_plan.rs::with_predicate_options`: when a pushed predicate selects every row it keeps
`selection` as `None`, "which enables coalesced page fetches". The fork never applied the
same rule to an externally supplied selection.

The `PageIndexEvaluator` itself walks every page and decodes each bound into a `Datum`
(`apply_to_pages`), so it also pays per-page CPU even when it can prove nothing; whether it
is material against the target ratio is a measurement question.

## Measurement

Ignored release test `arrow::page_prune_perf_tests::perf_page_prune_cost_breakdown`
(`cargo test -p iceberg --release --lib page_prune_perf -- --ignored --nocapture`).
Fixture: fork `DataFileWriter` output, 50 files × 128,000 rows,
`data_page_row_count_limit(8,000)` → 16 pages per column chunk, one row group per file.
Columns: `id` i64 ascending, `category` string cycling `cat_0..=cat_7` (every page holds
every value — `category = 'cat_7'` can prune no page), plus `v1` f64 / `v2` i32. Scan
collects all columns, batch size 8,192, median of 5 after a warmup pass.

(numbers to be recorded here)

## Fix

(to be recorded)

## Mutation evidence

(to be recorded)

## Gates

(to be recorded)
