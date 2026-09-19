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

# F-TS-PUSHDOWN-1 — timezone-bearing timestamp literals must reach the Iceberg scan predicate

**Date:** 2026-09-21. **Branch:** `fix/f-ts-pushdown-1`.
**Base:** `43fcd243`. **Model:** swe-2-high.
**Path:** STANDARD because this changes the DataFusion-to-Iceberg filter
conversion and touches scan pruning.

This ledger retires when the fork change merges or the owner removes the unit.

## The defect

Measured on RePark's bench at fork pin `43fcd243`: a Spark-door filter
`ts >= CAST(1703150000 AS TIMESTAMP)` on an Iceberg `timestamptz` column
reaches `IcebergTableScan` with `predicate: []`. The literal arrives as
`ScalarValue::TimestampMicrosecond(Some(v), Some("UTC"))` and
`scalar_value_to_datum`
(`crates/integrations/datafusion/src/physical_plan/expr_to_predicate.rs`)
converts every timestamp scalar to the zone-less `Datum::timestamp_micros` /
`Datum::timestamp_nanos`. `literal_binds_soundly` then converts the datum to
the field type; a `Timestamp` datum does not round-trip through
`Datum::to(Timestamptz)`, so `convert_filters_to_predicate` drops the
conjunct. No manifest, file, row-group or page is pruned on `ts`. Pushdown is
`TableProviderFilterPushDown::Inexact`, so DataFusion re-filters above the
scan and returned rows stay correct — the cost is pruning only.

A second defect falls out of the same code: a *zone-bearing* literal against
a *zone-less* `timestamp` column pushes `Datum::timestamp_micros` with the
zone silently discarded (`tsn >= lit(us, Some("UTC"))` pushed
`Timestamp`-typed datum on a `Timestamp` field and bound). The zone marker is
type information, not decoration: `timestamp` and `timestamptz` are different
Iceberg types, and a comparison DataFusion typed as cross-type must not be
re-bound as same-type by dropping the marker.

## Observed expressions

From `record_observed_filter_exprs` in
`crates/integrations/datafusion/tests/ts_tz_pushdown.rs`, recording every
expression the provider sees in `supports_filters_pushdown` and `scan`
(`--nocapture`, datafusion 54.1.0):

| SQL filter | expression handed to the provider |
|---|---|
| `ts >= CAST(1703150000 AS TIMESTAMP)` | `ts >= TimestampMicrosecond(1703150000000000, Some("UTC"))` |
| `ts < CAST(1703150000 AS TIMESTAMP)` | `ts < TimestampMicrosecond(1703150000000000, Some("UTC"))` |
| `ts = CAST(1703150000 AS TIMESTAMP)` | `ts = TimestampMicrosecond(1703150000000000, Some("UTC"))` |
| `ts BETWEEN CAST(a AS TIMESTAMP) AND CAST(b AS TIMESTAMP)` | two conjuncts `ts >= ..(a*10^6, Some("UTC"))`, `ts <= ..(b*10^6, Some("UTC"))` |
| `ts IN (CAST(a AS TIMESTAMP), CAST(b AS TIMESTAMP))` | `ts = ..(Some("UTC")) OR ts = ..(Some("UTC"))` |
| `tsn >= CAST(1703150000 AS TIMESTAMP)` | `tsn >= TimestampMicrosecond(1703150000000000, None)` |
| `ts >= TIMESTAMP '2023-12-21 10:33:20'` | `ts >= TimestampMicrosecond(1703154800000000, Some("UTC"))` |
| `ts >= CAST(1703150000 AS TIMESTAMP(6))` | `ts >= TimestampMicrosecond(1703150000, Some("UTC"))` — integer taken in the declared unit, not seconds |
| `id = 5` | `id = Int64(5)` |

Facts this establishes:

- The column is never wrapped in `Cast` for these queries; DataFusion coerces
  the *literal* to the column's Arrow type, stamping `Some("UTC")` on
  comparisons against the `timestamptz` column and `None` against the
  `timestamp` column.
- `CAST(int AS TIMESTAMP)` (no precision) interprets the integer as seconds
  and multiplies by 10^9; `CAST(int AS TIMESTAMP(p))` takes the integer in
  unit `p`.
- The pushed literal is the session execution timezone's zone string;
  `"UTC"` and `"+00:00"` are both spellings of the same zone the Iceberg
  Arrow mapping writes, and a non-UTC zone string on a `Timestamp` scalar is
  still the same UTC instant — Arrow timestamps store the instant, the zone
  is a display label.

## Clauses

- **C-1** `TimestampMicrosecond(v, Some(_))` converts to
  `Datum::timestamptz_micros(v)`; `TimestampNanosecond(v, Some(_))` to
  `Datum::timestamptz_nanos(v)`. The zone string never changes the value:
  `"UTC"`, `"+00:00"` and `"America/New_York"` all map the same `v` to the
  same datum, because the scalar already holds the UTC instant.
- **C-2** `TimestampMicrosecond(v, None)` / `TimestampNanosecond(v, None)`
  keep converting to `Datum::timestamp_micros` / `Datum::timestamp_nanos`.
- **C-3** `TimestampMillisecond` / `TimestampSecond` literals, if DataFusion
  ever delivers them un-coerced, convert by exact scaling
  (`*1_000`, `*1_000_000`, `checked_mul` — overflow is `None`, never a
  wrapped value) to the micros datum of the matching zoned/zone-less flavor.
- **C-4** Zone-bearing literals pushed onto `timestamptz` / `timestamptz_ns`
  columns bind and prune; the same literal against a `timestamp` /
  `timestamp_ns` column does not bind and the conjunct stays unpushed
  (DataFusion re-filters). Symmetrically, a zone-less literal against a
  zoned column stays unpushed. `timestamp` vs `timestamptz` is a cross-type
  comparison, not a same-instant one.
- **C-5** A `Timestamp(u1, z1) -> Timestamp(u2, z2)` cast wrapped around a
  column strips only when `u1 == u2` and `z1.is_some() == z2.is_some()` —
  i.e. the cast changes at most the zone *spelling* (`"UTC"` vs `"+00:00"`)
  and therefore changes no instant. Casts that change the unit or add or
  drop a zone are not stripped; the conjunct stays unpushed. This tightens
  the previous `time_unit_widens` rule for timestamps, which ignored the
  zone entirely and stripped `Timestamp(us,_) -> Timestamp(ns,_)`.
- **C-6** A `Timestamp`/`Timestamptz` datum never binds the other flavor's
  field; nothing in this change weakens `literal_binds_soundly`. A pushed
  predicate is exact or absent — there is no safe rounding direction, and
  `Inexact` pushdown does not make a wrong predicate safe (over-inclusive is
  free, under-inclusive is silent row loss).
- **C-7** Rows returned equal the in-memory reference for `>=`, `<`, `=`,
  `BETWEEN`, `IN`, a boundary microsecond, and a negative (pre-1970)
  literal, on both the multi-partition and single-stream scan paths.
- **C-8** `BETWEEN` arrives as `>=`/`<=` top-level conjuncts, and each pushes
  independently. `IN` arrives as `OR`ed `=` terms inside one expression: an
  `OR` pushes only when every term converts, so a mixed-zone `IN` list is
  dropped whole (DataFusion re-filters above the scan). That is the safe
  direction; no part of such a list is pushed. Corrected after the Grok logic
  review of #312.

## RED evidence (pre-fix, base 43fcd243)

`cargo test -p iceberg-datafusion --lib ts_tz` — 7 red of 11:

- `zoned_timestamp_literals_map_to_timestamptz_datums` — produced
  `Datum{type: Timestamp}` instead of `Timestamptz`.
- `milli_and_second_literals_widen_exactly` — `TimestampMillisecond` /
  `TimestampSecond` returned `None` (unhandled arm).
- `zoned_literals_push_onto_timestamptz_columns` — predicate `None` (the
  defect: `Timestamp` datum cannot bind `Timestamptz`).
- `cross_zone_timestamp_comparisons_stay_unpushed` —
  `tsn >= lit(us, Some("UTC"))` *pushed* `Timestamp`-datum (zone silently
  dropped; second defect).
- `zone_string_only_timestamp_cast_on_a_column_strips` —
  `CAST(ts AS Timestamp(us, "+00:00"))` not stripped
  (`time_unit_widens(us,us)` is false), predicate `None`.
- `cast_wrapped_timestamp_literal_converts_then_pushes` — `None`.
- `single_stream_scan_reads_rows_behind_a_zoned_literal` —
  `scan.predicates()` `None`.

Green pre-fix (correct already): `zoneless_timestamp_literals_keep_zoneless_datums`,
`zoneless_literals_still_push_onto_zoneless_columns`,
`zone_or_unit_changing_timestamp_casts_stay_unpushed`,
`milli_and_second_literals_past_the_micros_range_are_not_pushed`.

`cargo test -p iceberg-datafusion --test ts_tz_pushdown` — 3 red of 7:

- `zoned_literals_push_a_predicate_and_prune_files` — `predicates()` `None`,
  all 4 files planned instead of 2 (the RePark-measured defect, end to end
  through the catalog provider).
- `cross_zone_and_cross_unit_comparisons_stay_unpushed` —
  `tsn >= lit(us, Some("UTC"))` reached the scan predicate.
- `boundary_row_survives_a_zoned_literal` — reads all 4 rows instead of
  `{2,3}`: nothing is pushed, so the scan emits every file's row
  (scan-level collection, pre-DataFusion re-filter).

Green pre-fix: `unfiltered_scan_plans_every_file`,
`zoneless_column_filters_still_push_and_prune`,
`filtered_rows_match_the_in_memory_reference` (rows are correct without
pushdown — DataFusion re-filters; the pin guards against the fix changing
results),
`record_observed_filter_exprs` (prints the table above).

## Fix shape

In `expr_to_predicate.rs`:

- `scalar_value_to_datum` — match on the scalar's timezone field:
  `Some(_)` -> `timestamptz_*` datum, `None` -> `timestamp_*` datum; add the
  `TimestampSecond`/`TimestampMillisecond` arms with `checked_mul` scaling.
- `cast_strips_lossless` — the `Timestamp -> Timestamp` arm becomes
  `from_unit == to_unit && from_tz.is_some() == to_tz.is_some()`;
  `time_unit_widens` remains for `Time32/Time64`.

No change to `literal_binds_soundly`: a `timestamptz` datum on a `timestamp`
field (and vice versa) still fails `Datum::to` round-trip and stays unpushed,
which is exactly the C-4 contract.

## Pins

Unit (`--lib ts_tz`, module `expr_to_predicate::ts_tz_tests`):

- `zoned_timestamp_literals_map_to_timestamptz_datums` — C-1.
- `zoneless_timestamp_literals_keep_zoneless_datums` — C-2 incl. `None` value.
- `milli_and_second_literals_widen_exactly` /
  `milli_and_second_literals_past_the_micros_range_are_not_pushed` — C-3.
- `zoned_literals_push_onto_timestamptz_columns` — C-1/C-4, all zone
  spellings, `>=`, `<=`, `=`, `<`, `IN`, `BETWEEN`-shaped AND, `timestamptz_ns`,
  negative literal.
- `zoneless_literals_still_push_onto_zoneless_columns` — C-2/C-4.
- `cross_zone_timestamp_comparisons_stay_unpushed` — C-4 both directions,
  plus a mixed-zone IN list.
- `zone_string_only_timestamp_cast_on_a_column_strips` — C-5 positive arm,
  `us` and `ns` columns, `+00:00` and `America/New_York`.
- `zone_or_unit_changing_timestamp_casts_stay_unpushed` — C-5 negative arm:
  zone drop, zone add, unit widen, unit narrow.
- `cast_wrapped_timestamp_literal_converts_then_pushes` — cast on the
  literal side still converts.
- `single_stream_scan_reads_rows_behind_a_zoned_literal` — C-7 legacy path:
  builds an `IcebergTableScan` directly over a memory-catalog table,
  asserts the pushed predicate and the surviving row.

E2E (`--test ts_tz_pushdown`, through `IcebergCatalogProvider` over a
MemoryCatalog table `t(id, ts timestamptz, tsn timestamp)` with four data
files at ts V1=1_700_000_000_000_000us, V2=1_703_000_000_000_000us,
V3=1_706_000_000_000_000us, NEG=-86_400_000_000us):

- `zoned_literals_push_a_predicate_and_prune_files` — `>=`/`=`/`BETWEEN`/`IN`
  with `UTC`, `+00:00`, `America/New_York` literals push the exact datum and
  plan 2 of 4 files (`<` plans the complement).
- `zoneless_column_filters_still_push_and_prune` — `tsn` zone-less literal.
- `cross_zone_and_cross_unit_comparisons_stay_unpushed` — every cross-type
  and unit-changing form reaches the scan with `predicates() == None` and
  plans all 4 files.
- `filtered_rows_match_the_in_memory_reference` — C-7, `target_partitions`
  4 and 1, nine filter forms against an in-memory `ref` MemTable.
- `boundary_row_survives_zoned_literal` — C-7 boundary: `ts >= V2` returns
  exactly `{2,3}` on both paths; the row at the boundary microsecond is
  read.
- `unfiltered_scan_plans_every_file` — baseline: no filter plans all 4.
- `record_observed_filter_exprs` — prints the observed-expression table
  above.

## Mutation evidence

Arithmetic is `N red of M` against the populations above. Mutations were
applied one at a time and the tree was restored and re-run green between
and after them.

- **M-1 revert the fix**
  (`git checkout <red-commit> -- crates/integrations/datafusion/src/physical_plan/expr_to_predicate.rs`,
  all pins kept at HEAD):
  - `cargo test -p iceberg-datafusion --lib expr_to_predicate`:
    **8 red of 82** — the seven ts_tz pins
    (`zoned_timestamp_literals_map_to_timestamptz_datums`,
    `milli_and_second_literals_widen_exactly`,
    `zoned_literals_push_onto_timestamptz_columns`,
    `cross_zone_timestamp_comparisons_stay_unpushed`,
    `zone_string_only_timestamp_cast_on_a_column_strips`,
    `cast_wrapped_timestamp_literal_converts_then_pushes`,
    `single_stream_scan_reads_rows_behind_a_zoned_literal`) plus
    `tests::test_scalar_value_to_datum_timestamp` (pins the s/ms widening).
  - `cargo test -p iceberg-datafusion --test ts_tz_pushdown`:
    **3 red of 7** — `zoned_literals_push_a_predicate_and_prune_files`
    (predicate `None`, all 4 files planned),
    `cross_zone_and_cross_unit_comparisons_stay_unpushed`
    (`tsn >= lit(us, "UTC")` pushed a `Timestamp` datum),
    `boundary_row_survives_a_zoned_literal` (scan emitted all 4 rows).
- **M-2 `timestamp`/`timestamptz` value mutation** — one line in
  `timestamp_micros_datum`: the zoned arm pushed
  `Datum::timestamptz_micros(micros + 18_000_000_000)`, i.e. the literal
  went through a zone "conversion" that shifted the instant five hours.
  The datum still binds `timestamptz`, so the wrong predicate prunes.
  - `cargo test -p iceberg-datafusion --lib ts_tz`: **6 red of 11**.
  - `cargo test -p iceberg-datafusion --test ts_tz_pushdown`:
    **4 red of 7**, including the boundary pin:
    `boundary_row_survives_a_zoned_literal` returned `[3]` for
    `ts >= V2@UTC` (expected `[2,3]` — the boundary row was silently
    pruned by the shifted literal, a wrong answer DataFusion's Inexact
    re-filter cannot repair) and
    `filtered_rows_match_the_in_memory_reference` returned `[3]` for
    `WHERE ts >= CAST(1703000000 AS TIMESTAMP)`.
  - This is the load-bearing check for C-1: the zone string must be a
    display label only — any mutation that routes the literal through a
    `timestamp` <-> `timestamptz` value conversion moves the pushed
    boundary and is caught end to end.
- **Restore:** `git checkout HEAD -- expr_to_predicate.rs`; re-ran
  `--lib ts_tz` (11/11 ok) and `--test ts_tz_pushdown` (7/7 ok).
  `git status` clean afterwards.

## Gates

Final state, branch `fix/f-ts-pushdown-1`:

| command | result |
|---|---|
| `cargo fmt --all -- --check` | clean (exit 0) |
| `cargo clippy -p iceberg-datafusion -p iceberg --all-targets -- -D warnings` | clean (exit 0) |
| `cargo test -p iceberg-datafusion --lib ts_tz` | 11 passed, 0 failed |
| `cargo test -p iceberg-datafusion --lib expr_to_predicate` | 82 passed, 0 failed |
| `cargo test -p iceberg-datafusion --lib predicate` | 86 passed, 0 failed |
| `cargo test -p iceberg-datafusion --lib scan` | 42 passed, 0 failed |
| `cargo test -p iceberg-datafusion --lib table` | 57 passed, 0 failed |
| `cargo test -p iceberg-datafusion --test ts_tz_pushdown` | 7 passed, 0 failed |
| `make check` | exit 0 — fmt --check, workspace clippy `-D warnings`, taplo, cargo-machete, agent-artifacts, matrix-anchors, comment-blocks, rust-file-size all OK |
| `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/pb-fork2 origin/main HEAD` | `comment-ban hits=0` |

Environment for every test run: `CARGO_BUILD_JOBS=6 RUST_TEST_THREADS=6`.

One pre-existing pin updated as part of the fix:
`tests::test_scalar_value_to_datum_timestamp` asserted `TimestampSecond`
and `TimestampMillisecond` scalars convert to `None`; with the C-3
exact-widening arms they now convert to scaled `timestamp_micros` datums,
and the pin asserts the exact scaled value (and `i64::MAX`/`MIN` still
return `None` via `checked_mul`, covered in
`milli_and_second_literals_past_the_micros_range_are_not_pushed`).
