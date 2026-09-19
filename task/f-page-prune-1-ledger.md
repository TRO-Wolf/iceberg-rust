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
  lowered to the new sizes (reader 10162 after the round-2 `build_expected_schema` move,
  evaluator 1347, DF scan 1592), never raised.

## Round 2 — Spark-written fixtures

Provenance: five tables written by Spark 4.1.2 + iceberg-spark-runtime 1.11.0 (parquet-mr
writers), `write.parquet.page-row-limit=100`, `write.parquet.row-group-size-bytes=16384`,
2,000 rows over 4 data files per seed table, merge-on-read delete/update settings. Paths were
rewritten to the neutral prefix `/iceberg-fixtures/page-prune` and checked in under
`crates/iceberg/testdata/interop/page_prune/` (data + metadata + `page_prune_truth.json`,
Spark's recorded answer per query). Tests map the prefix onto the checked-in tree through a
`Storage` wrapper over `LocalFsStorage` (`PrefixStorage` in
`crates/iceberg/src/arrow/spark_fixture_tests.rs`), build each `Table` from its latest
metadata file, translate every truth predicate into an Iceberg `Predicate`, and assert row
equality with row selection ON and OFF.

| Table | Metadata | Contents |
|---|---|---|
| `base_v2` | v2 | 4 data files, 2,000 rows |
| `base_v3` | v2 | + `_row_id`, `_last_updated_sequence_number` assertions |
| `del_v2` | v3 | + 3 position-delete parquet files |
| `del_v3` | v6 | + deletion vectors (puffin), MoR update `i += 1_000_000` on ids 1000–1020 (`_last_updated_sequence_number` 5) |
| `evo_v2` | v10 | `i` INT→BIGINT, `f` FLOAT→DOUBLE, `dec` (9,2)→(18,2), `s`→`s2` rename, `n` drop/re-add (new id), `addc` added, +300 rows |

Coverage: 21 predicates × 4 tables + 11 × evo_v2 = **95 query assertions** at the core door,
each compared ON and OFF against Spark's recorded rows (ids; ids + lineage on the v3 tables).
Selective-page assertions: `id_eq`, `id_range`, `ts_range`, `i_gt`, `f_gt` skip pages on
`base_v2`; `id_eq`, `id_range`, `ts_range` on `base_v3`; `spark_base_v2_per_file_selection_prunes`
proves `id_eq` skips pages on ≥1 file and keeps on ≥1. DataFusion SQL door: 15 cases across
`base_v2`, `del_v3`, `evo_v2` — `count/sum/min/max(id)` equal Spark truth with the knob on and
off (`crates/integrations/datafusion/src/physical_plan/spark_fixture_tests.rs`).

### Findings fixed in round 2

- **Whole-file/lineage path dropped predicate columns (found via `base_v3.i_gt`/`del_v3.i_gt`
  → 0 rows).** The `_pos`/`_row_id` whole-file path decoded only the projected columns, so a
  residual on a non-projected column reached `survival_mask` on a batch missing the column and
  evaluated all-false (`i > 1800` → 0 rows under a `_row_id` projection). Fix
  (`crates/iceberg/src/arrow/reader.rs`): the decode projection is widened with the
  predicate's referenced field ids and the equality-delete `equality_ids`, and `survival_mask`
  evaluates the residual/eq masks on the decoded (pre-transform) batch — the same batch shape
  the normal path's `RowFilter` residual sees. `build_expected_schema` (the Avro/ORC decode
  schema) carries the same widening and moved to `open_parquet.rs` for the file-size ceiling.
- **Stale `file_size_in_bytes` broke the footer read (found via `del_v2`).** The fixture's
  path rewrite changed delete-file contents, so the manifest records sizes ~17–36 bytes larger
  than the files on disk; `open_parquet_file` passed the manifest size to
  `ParquetMetaDataReader::load_and_finish`, which then read past EOF ("failed to fill whole
  buffer"). Java reads the actual object length (`SeekableInputStream.getLength`) and never
  uses the manifest size for the footer. Fix (`crates/iceberg/src/arrow/open_parquet.rs`): on
  a failed open, stat the file and retry once with the real size; the fast path keeps the
  manifest size so the no-stat optimization stands.

### Documented semantics (not defects)

- **Spark SQL vs Java `Evaluator` on NULLs.** `n != 900` and `n NOT IN (900, 901)`: Spark's
  three-valued filter drops NULL rows; the residual evaluator is Java-faithful
  (`Evaluator.notEq`/`notIn` are `!eq`/`!in` under the nulls-first comparator, so NULLs pass —
  the audit BUG-002 semantics already encoded in `record_batch_predicate.rs`). The expected
  sets are Spark ∪ `n IS NULL` (base_v2/base_v3 1999/1998, del_v2/del_v3 1902/1901), and the
  tests assert exactly that. Through the DataFusion door the provider marks pushdown
  `Inexact`, DataFusion re-filters with SQL three-valued logic, and the SQL answer matches
  Spark verbatim (asserted: 1499/1474).
- **`s_eq` cannot prune on this fixture.** Every `s` value shares an 80-char prefix and
  parquet-mr truncates column-index bounds at 64 bytes, so every page's `s` bounds are the
  same truncated prefix — the literal can never be disproved and the evaluator keeps all
  2,000 rows (asserted by `spark_s_eq_keeps_all_pages_under_degenerate_truncated_bounds`; a
  foreign-writer case of the T clause's truncated-bound finding). Row results are still
  asserted Spark-exact in the query tests.
- **parquet-mr clause-M case confirmed.** The truth file records that the `d` DOUBLE column
  has no column index in the file holding NaNs — the missing-index tolerance from round 1
  carries it (those queries pass).

### Round-2 mutation evidence

| Mutation | Result |
|---|---|
| M6 `needs_physical_ordinals` decode widening off (`if false`) | `spark_base_v3_queries_match_with_lineage` red (`i_gt` → 0) |
| M7 `survival_mask(&transformed)` instead of decoded `&batch` | `spark_base_v3_queries_match_with_lineage` red (`i_gt` → 0) |
| M8 stale-size retry disabled (`if true`) | `spark_del_v2_queries_match_position_deletes` red (short read) |

## Gates

- `cargo fmt --all -- --check` — clean
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings` — clean
- `make check` — clean (fmt, workspace clippy, taplo, cargo-machete, agent-artifacts,
  matrix-anchors, comment-blocks, rust-file-size)
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/pb-fork origin/main HEAD` — `comment-ban hits=0`
- `cargo test -p iceberg --lib arrow` — 469 passed, 1 ignored (29 clause tests + 9 Spark-fixture
  tests)
- `cargo test -p iceberg-datafusion --lib` — 289 passed, 1 ignored (5 DF clause tests + 3
  Spark-fixture SQL-door tests)

## Round 3 — review remediation

Reviewer reports `rv-pp-logic-out.json` (logic) and `rv-pp-perf-out.json` / `rv-pp-perf-report.md`
(perf), remediated per orchestrator rulings.

### Findings table

| ID | Ruling | Pin | Commits |
|---|---|---|---|
| L-001 | Not a defect — declared divergence (Q-24b-1); evaluator unchanged | `n_all_null_pages_skipped_for_lt_declared_divergence` (core door), `sql_door_nulls_under_lt_le_not_gt_three_valued` (DF door) | `a6a7bdd5` |
| L-002 (P2) | `eq`/`IN` must keep pages whose column-index min or max is NaN | `f_eq_in_nan_bound_keeps_page` | `39a1bc3f` (test, RED), `a74abafb` (fix) |
| L-003 (P3) | Stale-size retry only after a footer-read failure at the manifest size; first error stays the source | `footer_error_at_real_size_does_not_retry`, `retry_failure_reports_first_error_as_source`, `footer_short_read_retries_with_real_size`, `page_index_error_does_not_retry` | `869b7182` (test, RED), `27f0654d` (fix) |
| R-01/R-05 (P2) | Load the page index only when the bound predicate has ≥1 page-prunable leaf; deletes still force it | `not_eq_only_scan_reads_no_index_bytes`, `eq_scan_reads_index_bytes`, `deletes_force_index_load` (counting storage) | `e2166af5` (test), `956e34d2` (fix) |
| R-02 (P3, optional if cheap) | Borrow `&[usize]` from the row-count cache; `calc_row_counts` capacity = page count | covered by existing evaluator/page-prune suite | `f6b6873a` |
| R-02 (Datum-free compare) | Deferred — threading raw typed bounds into every leaf predicate's `Datum`-ordering semantics is not the cheap part of the finding | — | — |
| R-03 (perf report, reader decode widening) | Kept — correctness fix proven by M6/M7; optional memory-only array drop not taken | — | — |
| R-04 | 1,347-line source ceiling preserved — evaluator tests split into `page_index_evaluator_tests.rs` | — | `a74abafb` |

### Declared: nulls under `<` at the core door

`PageIndexEvaluator::visit_inequality` skips all-null pages for `<` / `<=` (and, via
`rewrite_not`, for `NOT (s > _)` / `NOT (s >= _)`). Ruling Q-24b-1 declares this correct:

- The fork's own row-group evaluator on main does the same at the row-group level —
  `row_group_metrics_evaluator.rs` `visit_inequality` returns `ROWS_CANNOT_MATCH` when
  `contains_nulls_only(field_id)` (readable in this tree).
- Java `ParquetMetricsRowGroupFilter.lt`/`ltEq` return `ROWS_CANNOT_MATCH` when a column chunk
  contains only nulls, and parquet-mr's column-index filter treats NULL as not satisfying `<`
  (UNMEASURED — the Iceberg Java 1.11 source and parquet-mr are not in this tree; cited from
  the reviewer report and the orchestrator ruling).
- SQL three-valued logic and Spark's answer agree: `s < 'v999'`, `s <= 'z'`, and
  `NOT (s > 'a')` exclude NULL rows — pinned at the DataFusion door ON and OFF by
  `sql_door_nulls_under_lt_le_not_gt_three_valued` (the provider marks pushdown `Inexact`, so
  DataFusion re-filters with SQL semantics; `NOT (s > 'a')` reaches the page index as
  `s <= 'a'` through `rewrite_not`).

The genuinely inconsistent piece is the core `RecordBatch` residual's nulls-first `<`, which
keeps NULL rows (`record_batch_predicate.rs` module doc — readable; it carries the iceberg-api
1.10.0 bytecode citations for Java's `Comparators.nullsFirst` total order and the per-op
`null_verdict` table where `<`/`<=` resolve NULL to `true`). That is pre-existing BUG-002
semantics this unit does not change. Pinned at the core door by
`n_all_null_pages_skipped_for_lt_declared_divergence`: ON returns 384 rows — the page index
skips the two all-null pages (128 rows) and the residual then keeps the 64 NULLs on mixed
pages; OFF returns all 512. The core-door scan result therefore differs from SQL by exactly
the NULL rows on kept pages — matching what the row-group evaluator already produces for
all-null groups (skipped) vs NULLs inside kept groups (kept by the residual).

### Round-3 mutation evidence

| Mutation | Result |
|---|---|
| L-002 pin vs pre-fix evaluator (`eq`/`IN` skip on NaN bound) | `f_eq_in_nan_bound_keeps_page` red (NaN min bound wrongly skipped) |
| L-003 pins vs pre-fix retry (retried after index error; retry error lost first source) | `page_index_error_does_not_retry` red (3 opens, not 1), `retry_failure_reports_first_error_as_source` red |
| R-01/R-05 pin vs pre-fix gate (index loaded for any predicate) | `not_eq_only_scan_reads_no_index_bytes` red |
| M9 `AllNull` skip removed from `visit_inequality` | `n_all_null_pages_skipped_for_lt_declared_divergence` red |
| M10 DF pushdown `Inexact` → `Exact` | `sql_door_nulls_under_lt_le_not_gt_three_valued` red (512 rows incl. NULLs through the door) |

### Round-3 gates

- `cargo fmt --all -- --check` — clean
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings` — clean
- `make check` — clean (fmt, workspace clippy, taplo, cargo-machete, agent-artifacts,
  matrix-anchors, comment-blocks, rust-file-size; `reader.rs` ceiling lowered 10162 → 10157)
- `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/pb-fork origin/main HEAD` — `comment-ban hits=0`
- `cargo test -p iceberg --lib arrow` — 478 passed, 1 ignored
- `cargo test -p iceberg --lib page_index` — 16 passed
- `cargo test -p iceberg-datafusion --lib` — 290 passed, 1 ignored

## Round 4 — verification pin

The Grok verification critic (`rv-pp-verify-out.json`) passed the unit with one finding:

- **V-001 (S3):** mutating `visit_inequality` to *skip* a page when `bound.partial_cmp(datum)`
  returns `None` left every pin green — the "incomparable bounds keep the page" rule was
  unpinned. `Datum::to` normalizes the literal to the column type at bind time, so the `None`
  arm is unreachable through a bound predicate; the per-page decision is extracted into
  `PageIndexEvaluator::inequality_keeps_page` (same pattern as `eq_keeps_page`/`in_keeps_page`)
  and pinned directly by `inequality_incomparable_bound_keeps_page` — an `Int` bound vs a
  `String` literal (`partial_cmp` → `None`) must keep the page for `<`, `<=`, `>`, `>=`, plus a
  comparable-out-of-range bound must skip. Mutation applied → the pin red; reverted → green.
- **V-002 (noted):** the Spark fixtures are one row group per file, so the only two-row-group
  ranged cell is the synthetic `r_ranged_task_intersects_row_group_and_page_selection` pin.

### Round-4 mutation evidence

| Mutation | Result |
|---|---|
| V-001 `partial_cmp → None` skips the page instead of keeping it | `inequality_incomparable_bound_keeps_page` red |
