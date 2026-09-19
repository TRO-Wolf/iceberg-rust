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

## Clause matrix — all green (29 core + 5 DataFusion tests)

Core tests live in `crates/iceberg/src/arrow/page_prune_tests.rs` (W/M/L/D) and
`page_prune_tests_2.rs` (S/N/F/T/R); shared builders in `page_prune_fixture.rs`. Each fixture
writes 512 rows at `data_page_row_count_limit(64)` → 8 pages per column in one row group;
`assert_prunes` proves the evaluator's `RowSelection` skips at least one page, and every
correctness clause compares knob-ON rows against knob-OFF on the same task.

| Clause | Result | Tests |
|---|---|---|
| W writer emits page index | green | `w_data_file_writer_emits_column_and_offset_index` |
| M missing page index | green | `m_filtered_scan_succeeds_without_any_page_index`, `m_filtered_scan_succeeds_without_offset_index`, `m_filtered_scan_succeeds_with_chunk_only_statistics`, `m_prefetched_footer_without_index_scans_filtered_rows` |
| L row lineage (v3) | green | `l_row_lineage_columns_match_unfiltered`, `l_stored_row_lineage_columns_match_unfiltered`, `l_last_updated_sequence_number_alone_matches_unfiltered`, `l_pos_and_file_columns_match_unfiltered` |
| D deletes | green | `d_position_deletes_inside_kept_and_skipped_pages`, `d_equality_deletes_keyset_path`, `d_equality_deletes_null_key_and_nonkeyset`, `d_deletion_vector_inside_kept_and_skipped_pages` |
| S schema evolution | green | `s_added_column_predicates_match_unfiltered`, `s_renamed_column_predicate_matches_unfiltered`, `s_readded_name_with_new_field_id_matches_unfiltered`, `s_int_to_long_promotion_matches_unfiltered`, `s_float_to_double_promotion_matches_unfiltered`, `s_decimal_widening_matches_unfiltered`, `s_decimal_on_fixed_matches_unfiltered` |
| N nulls | green | `n_is_null_matches_unfiltered`, `n_is_not_null_matches_unfiltered`, `n_eq_not_eq_not_in_match_unfiltered` |
| F NaN | green | `f_is_nan_and_not_nan_match_unfiltered`, `f_lt_gt_not_match_unfiltered`, `f_eq_nan_matches_unfiltered` |
| T truncated bounds | green | `t_truncated_string_bounds_match_unfiltered`, `t_binary_column_matches_unfiltered` |
| R split tasks | green | `r_ranged_task_intersects_row_group_and_page_selection` |
| DF DataFusion door | green | `multi_partition_filtered_scan_row_selection_on_matches_off`, `single_stream_filtered_scan_row_selection_on_matches_off`, `single_stream_scan_builder_receives_row_selection_knob`, `scan_knobs_from_context_wires_row_selection_enabled`, `plan_carries_row_selection_enabled_to_multi_partition_path` (`crates/integrations/datafusion/src/physical_plan/page_prune_tests.rs`) |

The DataFusion door is proven through `IcebergTableProvider` end-to-end (row equality ON vs
OFF on each path), through the `ScanKnobs` seam (`scan_knobs_from_context` reflects `SET
iceberg.row_selection_enabled=false`, `build_table_scan` stores the flag on the `TableScan`),
and through `IcebergTableScan::plan` carrying the flag into the multi-partition exec. A
footer-prefetch byte count cannot serve as the seam: the 512 KiB `metadata_size_hint` default
always covers the index bytes of a test file.

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

## Fixes (commit 16bf797f)

1. `get_row_selection_for_filter_predicate` returns `Ok(None)` when column or offset index
   metadata is absent; the caller treats `None` as "scan all pages" (parquet-mr
   `ColumnIndexFilter` semantics — a missing index never errors, never excludes a page).
   Moved to `open_parquet.rs` in the size-gate refactor.
2. `page_index_policy(needed)` maps preload flags to `PageIndexPolicy::Optional` when page
   selection or deletes need them and `Skip` otherwise, in both the prefetched-footer and the
   footer-load paths (`open_parquet.rs`). `Required` (the `From<bool>` mapping) fails on
   index-less files before the evaluator can fall back.
3. `apply_predicate_to_column_index` decodes every supported index variant through
   `Datum::try_from_bytes`; unsupported variants (INT96) and undecodable bounds return `None`
   per page, which the visitors translate to "keep the page". BOOLEAN/INT32/INT64/FLOAT/
   DOUBLE use the generic `PrimitiveColumnIndex`; BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY share
   `ByteArrayColumnIndex`.
4. `bound_datum` normalizes decimal INT32/INT64 index bounds to `Int128` so they compare
   against Iceberg decimal literals; `Datum::physical` (the old, narrower converter) was
   removed as dead.
5. `visit_inequality` keeps a page when `partial_cmp` returns `None` (incomparable bound
   types, e.g. truncated or foreign-writer bounds) — the old code treated it as no-match and
   silently skipped the page.

## Mutation evidence — every fix is load-bearing

| Mutation | Result |
|---|---|
| M1 revert fix 1 (`Err` on missing index) | `m_filtered_scan_succeeds_without_any_page_index`, `m_filtered_scan_succeeds_without_offset_index` red |
| M2 revert bound decode (pre-fix `unwrap`/`FeatureUnsupported` arms) | `s_decimal_on_fixed_matches_unfiltered`, `t_binary_column_matches_unfiltered` red |
| M3 revert decimal Int64→Int128 normalization | `s_decimal_widening_matches_unfiltered` red |
| M4 `Optional` → `Required` index policy | `m_filtered_scan_succeeds_without_offset_index` red |
| M5 DataFusion default `true` → `false` | `scan_knobs_from_context_wires_row_selection_enabled` red |

## Findings recorded during implementation

- parquet-rs emits `ColumnIndexMetaData::NONE` for FLOAT and DOUBLE columns in this build —
  float predicates cannot prune and conservatively keep every page (correct, and the F/N
  clauses prove row equality; pruning coverage for floats depends on parquet-rs emitting
  those indexes or on foreign fixtures).
- parquet writers round truncated max bounds UP (`truncate_max_value`), so a shared >64-byte
  prefix legitimately keeps every page on `=`; the T clause uses distinct short-prefix
  binaries for a load-bearing prune and a long-prefix string for the keep-correctness case.
- NaN orders above all ordinary values in `Datum::partial_cmp` (Java `Double.compare`
  parity), so `!(f < 100)` and `f >= 100` correctly include the all-NaN page.
- Writer defaults: `WriterProperties::builder().build()` carries no page row limit (page
  size 1 MiB, `column_index_truncate_length` 64); `write.parquet.page-row-limit` and
  `write.parquet.page-size-bytes` are not honoured (Java honours both) — recorded, not fixed.
- The file-size gate (`check_rust_file_size.sh`) forced three extractions, all
  behavior-preserving: `get_row_selection_for_filter_predicate` and `page_index_policy` →
  `arrow/open_parquet.rs`; `TableScan::row_selection_enabled` getter →
  `scan/partition_work.rs`; `build_table_scan` → `physical_plan/scan_knobs.rs`. Ceilings were
  lowered to the new sizes (reader 10185, evaluator 1347, DF scan 1592), never raised.

## Gates

- `cargo fmt --all -- --check` — clean
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings` — clean
- `make check` — clean (fmt, workspace clippy, taplo, cargo-machete, agent-artifacts,
  matrix-anchors, comment-blocks, rust-file-size)
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/pb-fork origin/main HEAD` — `comment-ban hits=0`
- `cargo test -p iceberg --lib arrow` — 460 passed, 1 ignored (includes all 29 clause tests)
- `cargo test -p iceberg-datafusion --lib` — 286 passed, 1 ignored (includes all 5 DF clause
  tests)
