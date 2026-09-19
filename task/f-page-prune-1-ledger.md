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

# F-PAGE-PRUNE-1 — page-level row selection on the DataFusion scan

Scope: `iceberg.row_selection_enabled` on `IcebergScanOptions`, carried through `ScanKnobs` into
both DataFusion execution paths, default `true` once every clause below is green. The work is
correctness first: a wrongly excluded page is a silent wrong answer because the `RowFilter`
residual only re-applies the predicate to surviving rows.

## The two execution paths

| Path | Where | Row selection reaches the reader via |
|---|---|---|
| Multi-partition | `IcebergTableScan::execute` → `stream_partition_work(file_io, &work, concurrency, batch_size, true, row_selection_enabled)` (`crates/integrations/datafusion/src/physical_plan/scan.rs`, `crates/iceberg/src/scan/partition_work.rs`) | `ArrowReaderBuilder::with_row_selection_enabled` |
| Single-stream | `IcebergTableScan::execute` (empty `partition_work`) → `get_batch_stream` → `TableScanBuilder::with_row_selection_enabled` (`crates/iceberg/src/scan/mod.rs`) | same builder inside `TableScan::to_arrow` |

The session door is `IcebergTableProvider::scan` → `scan_knobs_from_context(&state.task_ctx())`
(`crates/integrations/datafusion/src/table/mod.rs`).

## How the reader loads the page index

- `ArrowReader::process_parquet_file_scan_task` (`crates/iceberg/src/arrow/reader.rs`) computes
  `should_load_page_index = (row_selection_enabled && task.predicate.is_some()) || !task.deletes.is_empty()`
  and sets `parquet_read_options.preload_page_index` accordingly. Unfiltered scans without
  deletes must pay nothing.
- `open_parquet_file` (`crates/iceberg/src/arrow/open_parquet.rs`): with prefetched footers it
  calls `ParquetMetaDataReader::new_with_metadata(...).load_page_index(...)`; otherwise the
  `ArrowFileReader::get_metadata` call parses the footer with the same policies.
- `PageIndexPolicy::from(bool)` maps `true` to **`Required`** (parquet-rs), which errors when the
  file lacks an offset index. `preload_column_index` / `preload_offset_index` default to `true`
  in `ParquetReadOptions`, so today the column and offset indexes are loaded and required on
  every task — including unfiltered scans. The unit maps preloads to `Optional` when requested
  and `Skip` when not, so an index-less file never fails at metadata load and an unfiltered
  scan never fetches index bytes.
- `get_row_selection_for_filter_predicate` drives `PageIndexEvaluator`
  (`crates/iceberg/src/expr/visitors/page_index_evaluator.rs`) per selected row group and zips
  page selectors into a `RowSelection`; the `RowSelection` coordinate space covers only the
  selected row groups (parquet-rs applies `with_row_groups` before `with_row_selection`).
  Position/equality deletes intersect via `build_deletes_row_selection`.
- Java parity references: `org.apache.iceberg.parquet.ParquetMetricsRowGroupFilter` (row-group
  level; metrics truncation uses `is_{min,max}_value_exact` on column-chunk Statistics) and
  parquet-mr `ColumnIndexFilter` (page level; a column without a column index contributes "all
  pages", never an error). parquet-rs writes column-index min as a prefix bound and max as an
  *incremented* upper bound (`truncate_min_value` / `truncate_max_value` in
  `parquet::column::writer`), so stored bounds stay valid under the 64-byte truncate.

## Clause matrix

| Clause | Test (planned name) | File |
|---|---|---|
| W writer emits page index | `w_writer_emits_column_and_offset_index` | `src/arrow/page_prune_tests.rs` |
| M missing page index | `m_scan_succeeds_without_column_index`, `m_scan_succeeds_without_offset_index`, `m_prefetched_footer_without_index` | same |
| L row lineage | `l_row_lineage_columns_match_unfiltered` | same |
| D deletes | `d_position_deletes_inside_skipped_pages`, `d_equality_deletes_null_key_and_nonkeyset`, `d_deletion_vector_v3` | same |
| S schema evolution | `s_added_column_predicates`, `s_renamed_column`, `s_drop_readd_new_id`, `s_int_to_long`, `s_float_to_double`, `s_decimal_widening` | same |
| N nulls | `n_is_null`, `n_is_not_null`, `n_eq`, `n_not_eq`, `n_not_in` | same |
| F NaN | `f_is_nan`, `f_not_nan`, `f_lt`, `f_gt`, `f_not_lt`, `f_eq_nan` | same |
| T truncated bounds | `t_eq`, `t_lt`, `t_ge`, `t_starts_with`, `t_not_starts_with`, `t_binary_column` | same |
| R split tasks | `r_ranged_task_multi_row_group` | same |
| DF DataFusion door | `df_row_selection_knob_both_paths`, `df_set_knob_reaches_plan` | `physical_plan` tests |

Each clause asserts: the fixture file has >= 4 pages in the filtered column,
`PageIndexEvaluator::eval` yields a `RowSelection` with at least one skip and one select
selector, and knob-ON rows/values equal knob-OFF (and the in-memory filter of the unfiltered
scan).

## Findings (pre-change audit)

- `get_row_selection_for_filter_predicate` returns `Err("does not contain a column index")`
  when `ParquetMetaData::column_index()` or `offset_index()` is `None`, and `Required` policy
  fails earlier at `load_page_index` on files with no offset index. parquet-mr semantics: a
  missing index means "select all pages" — never an error (clause M).
- `apply_predicate_to_column_index` BYTE_ARRAY arm decodes bounds with
  `String::from_utf8(...).unwrap()` — panics on binary columns and non-UTF8 bounds; FIXED and
  INT96 arms return `FeatureUnsupported` errors that abort the scan. Decimal-on-FIXED columns
  (precision > 9) therefore fail any filtered scan with row selection on. Conservative decode
  via `Datum::try_from_bytes` (keep the page when a bound cannot decode) matches parquet-mr's
  "keep pages you cannot disprove" rule.
- `preload_column_index` / `preload_offset_index` default `true` → `Required` → index bytes are
  fetched on every task, so the brief's "unfiltered scans pay nothing" is not true today. All
  three preload flags will be gated on `should_load_page_index` and map to `Optional`/`Skip`.
- parquet-rs writers exclude NaN from page statistics (`is_nan` skip in
  `parquet::column::writer`), and all-NaN pages carry no min/max bound; a foreign writer could
  still emit a NaN bound, so the evaluator gets a defensive "NaN bound → keep page" guard.
- Writer defaults: `WriterProperties::builder().build()` carries no `data_page_row_count_limit`
  (page size limit 1 MiB, `column_index_truncate_length` 64), so a default fork write produces
  one page per column chunk for realistic test files. `write.parquet.page-row-limit` and
  `write.parquet.page-size-bytes` table properties are NOT honoured (Java honours both) —
  recorded here, not fixed in this unit.
- `_pos` / `_row_id` projections route to a whole-file path (`needs_physical_ordinals`), which
  rejects ranged split tasks and bypasses RowFilter/RowSelection/row-group pruning entirely;
  the knob cannot change `_pos`/`_row_id`/`_file` results. `_last_updated_sequence_number`
  alone stays on the normal path as a transformer constant.
- Residual contract for missing columns (from `PredicateConverter`): `is_null`/`lt`/`le`/
  `not_eq`/`not_in`/`not_starts_with`/`not_nan` → always-true, `not_null`/`gt`/`ge`/`eq`/
  `starts_with`/`in`/`is_nan` → always-false. The evaluator's `MissingColBehavior` arms agree.
- DataFusion never pushes a NaN literal: `expr_to_predicate.rs` rewrites NaN equality/IN to
  `is_nan` (F-ICE-NAN-PUSHDOWN-1). Core-level `eq(NaN)` is consistent ON==OFF because the Arrow
  `eq` residual matches nothing either.

## Mutation evidence

(to be filled — one leg per fix, `N red out of M` recorded per leg)
