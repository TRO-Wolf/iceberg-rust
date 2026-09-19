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

Recorded runs (each printed scan number is already a median of 5):

Before the fix (selection handed to parquet unconditionally):

```
(a) evaluator total 168-255µs over 50 files (~5µs/file)
(b) nonprunable selection keeps every row: true
(c) decode/file: all-select RowSelection ~2.0ms, no selection ~1.89ms, no index ~1.84ms
(d) footer+index parse: Optional ~20µs, Skip ~13µs
scan nonprunable: ON 241.4ms OFF 189.4ms ratio 1.274
scan prunable: ON ~28ms OFF ~44ms ratio ~0.63
```

The evaluator (~5µs/file) and the index parse (~7µs/file) are noise; the all-keep
`RowSelection` costs ~150µs/file in the sparse-fetch path. After landing the
`effective_row_selection` drop alone the ratio improved to ~1.17 but a residual ~33ms
(≈660µs/file) remained.

Residual isolation: forcing `should_load_page_index` off produced ON/OFF = 1.011 —
the residual is the *presence* of the loaded index, not the selection. Byte-level read
recording shows identical fetch plans ON vs OFF (3 calls, same ranges; the coalesced
data read covers the index bytes anyway). The mechanism is inside parquet-rs:
`InMemoryRowGroup::column_chunks` hands `page_locations` to `SerializedPageReader`
whenever `offset_index` is present, flipping it from `Values` (incremental header
parse) to `Pages` (per-page `get_bytes`) mode. Measured directly in `decode_one`:

```
decode/file, no selection: filter + index loaded 3.42-3.64ms vs filter + no index 2.82-2.94ms
```

i.e. ~0.7ms/file — with a `RowFilter` present, `Pages` mode is slower than header
parsing. That is a parquet-rs 58.4.0 internal; the fix must keep the index out of the
decode metadata for the all-keep case.

## Fix

Two changes, both keyed off the *evaluated* selection, never the static leaf class:

1. `effective_row_selection` (`crates/iceberg/src/arrow/open_parquet.rs`) drops a
   `RowSelection` that selects rows but skips none (`selects_any() &&
   skipped_row_count() == 0`), applied to the final intersected selection before
   `with_row_selection`. Empty selections (`!selects_any()`) and any selection with a
   skipped row — including every delete-derived selection — are preserved.
2. `ArrowReader::prune_indexed_metadata_for_scan` (same file) runs before the stream
   builder exists: only when there are no deletes, row selection is on, the predicate
   is statically prunable, and no `_pos`/`_row_id` ordinals are projected, it evaluates
   row-group filtering and the filter's `RowSelection` on `arrow_metadata` and, when
   the result is all-keep, rebuilds the `ArrowReaderMetadata` over a
   `ParquetMetaData` whose column and offset indexes are stripped
   (`into_builder().set_column_index(None).set_offset_index(None)`), preserving the
   stamped/coerced arrow schema. The decode then runs `Values` mode exactly as the
   pre-F-PAGE-PRUNE-1 path did. Prunable selections keep the index (sparse fetch is
   the win); deletes never take this path and always keep the index.

After the fix (medians of 5, two runs):

```
scan nonprunable: ON 184.2/184.3ms OFF 183.7/190.3ms ratio 1.003 / 0.969   (target ≤1.03)
scan prunable:    ON 28.0/26.5ms  OFF 43.4/41.9ms  ratio 0.645 / 0.632    (win retained)
```

Behavior pins in `open_parquet_tests.rs` use a thread-local seam that counts only
selections actually handed to parquet (`ROW_SELECTIONS_APPLIED`) and one that counts
index-stripped metadata (`PAGE_INDEX_STRIPS`):

- all-keep predicate → 0 selections applied, 1 strip, all rows returned
- pruning predicate → 1 selection applied, 0 strips, pruned rows
- position deletes → 1 selection applied, 0 strips, deleted rows removed
- `effective_row_selection` unit pins: all-keep dropped, partial kept, empty kept

## Mutation evidence

- `effective_row_selection` mutated to a passthrough: `all_keep_row_selection_is_dropped`
  and `all_keep_predicate_hands_no_selection_to_parquet` both go red (selection applied
  count 1, expected 0).
- `prune_indexed_metadata_for_scan` mutated to return before stripping:
  `all_keep_predicate_hands_no_selection_to_parquet` goes red on the strip assert
  (0 strips, expected 1).

Both restored; all pins green again.

## Round 2 — review remediation

Review reports: logic PASS (`rv-pp2-logic-report.md`), perf APPROVE
(`rv-pp2-perf-out.json`).

| Finding | Severity | Remediation | Evidence |
|---|---|---|---|
| L-001 — the strip arm's `selected_row_group_indices` return and the caller's `with_row_groups` had no oracle | S2 | Two pins added in `open_parquet_tests.rs` | `ranged_all_keep_scan_returns_only_split_row_groups`, `all_keep_pages_return_only_selected_row_groups`; mutation below turns the second pin red |
| R-01 — deep `ParquetMetaData` clone on the strip path | P3 | `Arc::try_unwrap` + `into_builder().set_column_index(None).set_offset_index(None)` when the metadata `Arc` is unique; otherwise `ParquetMetaData::new(file_metadata.clone(), row_groups.to_vec())`, which carries no indexes | all strip pins stay green |

Pin shapes:

- `ranged_all_keep_scan_returns_only_split_row_groups`: 4 row groups of 128,
  split task starting at RG2's data offset through EOF, predicate `id >= 0`
  (all-keep at page level). Asserts exactly 256 rows and 1 strip. This pin guards
  the caller's `with_row_groups` application: the strip arm's indices are
  byte-range derived, so the caller re-derives them when the arm returns `None`
  (`reader.rs` byte-range gate) and this pin alone cannot see that mutation.
- `all_keep_pages_return_only_selected_row_groups`: 4 row groups of 128, `s`
  all-null in RG0–RG1 and `"a"` in RG2–RG3, predicate `s < 'x'`. Row-group
  metrics drop the all-null groups (`contains_nulls_only` → `null_count ==
  num_rows`); the surviving groups are all-keep at page level, so the strip path
  runs with `Some([2, 3])`. Asserts exactly 256 rows and 1 strip. Losing the
  indices re-reads the excluded groups, and the nulls-first `lt` residual keeps
  their NULL rows: 512 rows.

Round-2 mutation evidence:

- Strip arm mutated to `Ok((arrow_metadata, None, row_selection))`:
  `all_keep_pages_return_only_selected_row_groups` fails (512 rows returned,
  256 expected) — red as designed; `ranged_all_keep_scan_returns_only_split_row_groups`
  stays green because the caller re-derives the byte-range restriction, which is
  itself the contract that pin documents. Restored; both pins green.

## Round 3 — verification remediation

Review report: verification critic NEEDS_REMEDIATION — no wrong-row bug found, two
stated invariants unpinned in the committed suite.

| Finding | Severity | Remediation | Evidence |
|---|---|---|---|
| V-001 — strip-when-deletes unpinned: removing `task.deletes.is_empty()` from `decide_early` left every committed pin green | S2 | `all_keep_predicate_with_position_deletes_keeps_index` in `open_parquet_tests.rs` | pin red under the deletes-guard mutation below |
| V-002 — all-keep strip must not poison a shared footer-cache entry | S2 | `v_all_keep_strip_leaves_cached_index_for_later_prune` in `footer_cache_v_tests.rs` | pin red under the cache write-back mutation below |

Pin shapes:

- `all_keep_predicate_with_position_deletes_keeps_index`: `id >= 0` (all-keep,
  prunable) on a file with two position deletes. Asserts `ROWS - 2` rows, one
  selection applied (the intersected delete selection), zero strips.
- `v_all_keep_strip_leaves_cached_index_for_later_prune`: through one shared
  `ParquetFooterCache`, an `id >= 0` scan of file F (takes the strip path,
  1 strip), then a direct `footer_or_fetch` probe asserting the cached entry
  still carries column and offset indexes, then an `id = 64` scan of F asserting
  1 row, rows equal to an uncached scan, and exactly one selection applied.

Round-3 mutation evidence:

- V-001: `task.deletes.is_empty()` dropped from `decide_early` → the pin fails on
  the strip count (1 vs 0); rows stay `ROWS - 2` because the intersected delete
  selection is still applied. Restored; pin green.
- V-002, equivalent mutation: strip via `Arc::make_mut` in place instead of
  `try_unwrap` + `ParquetMetaData::new` copy → the pin stays green. `make_mut`
  clones the inner `ParquetMetaData` while the cache holds a ref, so by
  construction no mutation of the cached Arc can occur; this mutation class
  cannot be detected by any scan-level pin and is recorded as equivalent.
- V-002, poisoning mutation: `prune_indexed_metadata_for_scan` made async, given
  the `TableFooterCache`, and the strip arm writes the stripped
  `ArrowReaderMetadata` back into the shared entry keeping the entry's index
  flags → the pin fails on the cached-index assert (the served entry's column
  index is `None`); a pruning rescan would also see zero selections applied.
  Restored; pin green.

## Gates

- `cargo fmt --all` clean; `cargo clippy -p iceberg -p iceberg-datafusion
  --all-targets -- -D warnings` clean
- filtered suites: `open_parquet` 19, `page_prune` 31+1 ignored,
  `footer_cache` 24, `spark_fixture` 10, `iceberg-datafusion page_prune` 6 —
  all green
- comment-ban gate `hits=0`; `scripts/check_rust_file_size.py` 573 files clean
- `crates/iceberg/src/arrow/reader.rs` 10,139 lines (ceiling 10,139)
