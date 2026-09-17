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

# F-SORTED-INSERT-1 ledger

Model: muse-spark-1.3-contributor. Branch `fix/ice-sorted-insert-1`, from fork main `edc38c6a`.
RePark rating 2026-09-16 row V2-12 claim C-7: plain `INSERT INTO` into a table with a declared
default sort order writes UNSORTED files with manifest `sort_order_id` NULL.

## Contract

- C-001: every data file `insert_into` writes into a table with a non-empty default sort order is
  sorted by that order (direction and null order honoured, per writer stream — Spark's
  `LOCALLY ORDERED` half; no global range distribution).
- C-002: each written data file's `sort_order_id` is the table's default sort order id
  (0 for unsorted tables, as Java/Spark).
- C-003: identity, multi-key and nulls-first/last cells.
- C-004: transform orders (`bucket`, `truncate`, `year/month/day/hour`) sort by the transformed
  value. RePark side names this `WRITE-ORDER-TRANSFORM-1`.
- C-005: no change for tables without a sort order (no SortExec added, plan unchanged).

## Findings

- Write path: `IcebergTableProvider::insert_into`
  (`crates/integrations/datafusion/src/table/mod.rs:173`) builds
  `project_with_partition` → `repartition` → fanout branch (`sort_by_partition` when fanout is
  off) → `IcebergWriteExec` → `CoalescePartitionsExec` → `IcebergCommitExec`.
- `DataFileWriter::close` (`crates/iceberg/src/writer/base_writer/data_file_writer.rs:201`)
  stamps `partition_spec_id` on every file but never `sort_order_id`, so it stays `None` (NULL).
  The manifest/commit path carries whatever the `DataFile` holds through JSON
  (`serialize_data_file_to_json` / `deserialize_data_file_from_json`), so stamping at the base
  writer is sufficient; no commit change is needed.
- C-002 zero-stamp evidence: the brief asserts Spark stamps 0 for unsorted tables. In-repo
  corroboration: row R114 records that Java's equality-delete builder defaults an absent
  `sort_order_id` to `SortOrder.unsorted().orderId()` (0, not null), and
  `spec/manifest/_serde.rs` carries `sort_order_id: Some(0)` as the Java shape. Decision: stamp
  `Some(default_order_id)` on every file `insert_into` writes, including `Some(0)` for unsorted
  tables. The `DataFileWriterBuilder` default stays `None` so all other callers
  (maintenance, deletes) keep today's behaviour.
- C-004 decision: every fork `Transform` except `Unknown` evaluates through
  `iceberg::transform::create_transform_function`, so the sort keys evaluate the transform per
  row (RePark `distribution.rs` `PartitionTransformExpr` shape, mirrored as `SortTransformExpr`
  with honest children). `Void` fields are skipped (constant-null key, no ordering content; an
  all-void order stamps its id with no SortExec). `Unknown`, an unresolvable source id, or a
  non-top-level (nested) source id falls back to today's plan with stamp `None` — a typed refusal
  is rejected by the brief because INSERT succeeds today. Nested struct-child sort keys are not
  implemented (no oracle cell needs them).
- ClusteredWriter (fanout off) rejects a re-appearing partition, so the sort key list is
  `_partition`-major (`_partition` first with the existing default options, then the default-order
  keys): partition runs stay contiguous and sorted within. For the fanout path the same prefix is
  harmless (per-partition stream order is still key-sorted by restriction) and keeps one code path.
- Optimizer survival (measured 2026-09-16, DataFusion 54.1.0): an explicit `SortExec` below
  `IcebergWriteExec` is REMOVED by `EnforceSorting` unless the write declares a matching
  `required_input_ordering`. Mechanism: `ensure_sorting` at the `CoalescePartitionsExec` above the
  write sees unordered write output and runs `remove_corresponding_sort_from_sub_plan`, which digs
  through the write to the `SortExec`; the dig is skipped only when the write's own requirement is
  `Some` (`propagates_ordering = maintains && required.is_none()` in
  `update_sort_ctx_children_data`). This is also why the pre-existing `sort_by_partition` survives
  only on the non-fanout path. Fix shape: `write_sort_plan` resolves the keys once;
  `sort_for_write` builds the explicit `SortExec` (`preserve_partitioning = true`, parallel), and
  `IcebergWriteExec::input_requirements` declares the identical ordering so the optimizer keeps
  the node. Proven: 4-partition source yields `Write(4) <- SortExec(4)` post-optimization, i.e.
  the explicit node (not an optimizer-added single-partition funnel) survives.
- Deletion safeguard: the requirement is the correctness anchor — with the explicit sort forced
  off, the optimizer inserts its own sort from the requirement and data stays sorted; with the
  requirement dropped, the explicit sort is stripped and files land unsorted (mutation M1).
- `table/mod.rs` sits exactly at its legacy file-size ceiling (2243 = 2243), so the `insert_into`
  edit must be net-negative: the fanout-property block moves into `physical_plan/sort.rs`
  (`sort_for_write`), which has headroom (240/1000).
- Oracle `sort_transform_bucket` records `"sorted": false` with first rows `[[0],[1],[2]]`: the
  flag is a raw-id check, while the brief states Spark sorts by `(bucket, id)`. The fork test
  asserts the `(bucket, id)` property directly.

## Red phase (base tree)

`cargo test -p iceberg-datafusion --test sorted_insert` on the base tree: 0 passed, 8 failed.

- `insert_into_table_with_asc_order_writes_one_sorted_file_with_order_stamp` — file rows not
  ascending; stamp `None`, expected `Some(1)`.
- `insert_into_table_with_desc_order_writes_descending_file` — file rows not descending; stamp
  `None`, expected `Some(1)`.
- `insert_into_two_key_order_honours_direction_and_null_order` — rows land in INPUT order
  `[(Some(2), Some(1)), (None, Some(5)), ...]`, expected
  `[(Some(0), Some(7)), (Some(1), None), ...]`; stamp `None`.
- `insert_into_partitioned_table_sorts_within_each_file` — stamp `None`, expected `Some(1)`
  (2 files land, unsorted).
- `insert_into_partitioned_table_sorts_every_file_across_streams` — stamp `None`, expected
  `Some(1)`.
- `insert_into_bucket_order_sorts_by_bucket_then_id` — `(bucket, id)` order broken.
- `insert_into_sorted_table_feeds_write_through_sort` — write input is not a `SortExec`.
- `insert_into_unsorted_table_adds_no_sort_and_stamps_zero` — plan shape holds on base
  (`Some(false)`), stamp `None`, expected `Some(0)`.

## Test adequacy (one knob at a time, population M = 8 `sorted_insert` tests)

- M1 — `write.rs::input_requirements` ignores `write_sort_plan` (requirement back to base,
  explicit sort + stamps intact): 7 red out of 8 (only `insert_into_unsorted_table_...` passes).
  The declared requirement is load-bearing; without it the optimizer strips the explicit sort.
- M2 — `write.rs::execute` skips `with_sort_order_id`: 7 red out of 8 (only the stamp-free
  plan-shape test passes). The stamp path is load-bearing.
- Restore + re-run: 8 passed out of 8.
- Vacuity sweep: sortedness asserts production-written file order against an independently sorted
  copy (falsifiable — red on base); stamps compare table-metadata read-back against manifest
  entries (independent values); shape asserts (`files.len`, `SortExec` presence) pin the action,
  not just read identity. No `=`-null predicates. The single-constant-row plan shape was dropped:
  the optimizer provably eliminates that sort, so the plan-shape test uses a 100-row source.

## Gates

- `cargo test -p iceberg-datafusion --test sorted_insert`: 8 passed, 0 failed.
- `cargo test -p iceberg-datafusion --lib`: 216 passed, 0 failed, 1 ignored.
- `cargo test -p iceberg --lib writer`: 162 passed, 0 failed, 1 ignored.
- `cargo clippy -p iceberg-datafusion -p iceberg --all-targets -- -D warnings`: clean
  (one `needless_range_loop` in the new test fixed with `iter().enumerate()`).
- `make check` with `CARGO_BUILD_JOBS=10 RUST_TEST_THREADS=8`: exit 0 (464 files clean;
  `table/mod.rs` legacy ceiling lowered 2243 to 2220 after the net-negative edit, as the
  checker requires).
- Neighbour suites: `insert_distribution` 7 passed, `fanout_insert_order` 1 passed,
  `partitioned_insert_select_test` 12 passed.

## Round 2 (remediation, 2026-09-17, base `27afcba4`)

Model: muse-spark-1.3-contributor. Critic report `/tmp/oc-worker/ic-rv/sort-logic-1-report.md`
(Grok 4.6 critic-logic, run 19c). Oracle float measurement
`/tmp/oc-worker/ic-build/nan_sort_spark.json` (probe `nan_sort_probe.py` beside it; PySpark 4.1.2
+ Iceberg 1.11.0, single task): table `(k INT, f FLOAT|DOUBLE)` `WRITE ORDERED BY f ASC NULLS
FIRST`, appended k→f 1→NaN, 2→-1.0, 3→-NaN (`0xFFC00000` / `0xFFF8000000000000`), 4→1.0, 5→+Inf,
6→-0.0, 7→0.0, 8→NULL, 9→-Inf. Spark file order by k: **[8, 9, 2, 6, 7, 4, 5, 1, 3]** for both
types — NULL, -Inf, -1, -0.0, 0.0, 1, +Inf, then every NaN (any sign/payload) as one greatest
value, ties in input order.

- L-01 [P1]: float/double keys now sort through `CanonicalFloatExpr`
  (`physical_plan/sort.rs`), which maps every NaN to +NaN before the sort and passes all other
  values (including -0.0) through. The wrap applies to any key whose Arrow type is Float32/Float64
  (identity today; no Iceberg transform yields a float — bucket→int, temporal→int/date,
  truncate rejects float/double — so the wrap is future-proofing, not a live second path). ASC
  puts the canonical NaN last under Arrow totalOrder; DESC (same key, reversed `SortOptions`)
  puts it first, which is Java's reversed comparator. Pins: `insert_into_float/double_order_sorts_nan_last_asc`
  (first seven keys exact `[8, 9, 2, 6, 7, 4, 5]`, NaN pair as a set) and the DESC twins
  (`insert_into_float/double_order_sorts_nan_first_desc`: NaN pair first as a set, tail exact
  `[5, 4, 7, 6, 2, 9, 8]`). The NaN pair is asserted as a set because Arrow `sort` is documented
  unstable and Iceberg requires no tie stability (critic §attack-1, not a finding); the measured
  run orders them `[1, 3]`, matching the oracle's input-order tie.
- L-02 [P2]: `write_sort_plan` resolves `source_id` with `Schema::field_by_id` (nested struct
  children included) and locates the top-level Arrow column through `sort_source_path`, an
  explicit-stack walk that returns the top name plus the child path (iterative, so no depth
  argument is needed). A nested key sorts through `NestedFieldExpr`, which extracts the child
  per row and nulls the key where any enclosing struct is null (`nullif` with the parent
  validity, mirroring Java `SortKey` on a null parent). A two-key order with one nested key
  sorts by both keys. Pins: `insert_into_nested_identity_order_sorts_by_child` (null parent
  sorts as null first; the garbage child value 999 under the null parent proves the
  propagation) and `insert_into_two_key_order_with_nested_key_sorts_by_both`. Fanout-off
  regression fixed: `write_input_without_default_sort` returns the input as-is when the plan
  carries no `_partition` column (unpartitioned tables never project one), instead of erroring
  `_partition not found`. Pin: `insert_into_unknown_transform_order_writes_with_zero_stamp`
  runs with `fanout.enabled=false` on an unpartitioned table.
- L-03 [P2]: new pins in `sorted_insert_types.rs` (string, decimal, timestamptz, boolean, binary
  identity; truncate on string and decimal; day/hour on timestamp; NULL through bucket) and
  `sorted_insert_writer.rs` (nested ×2, rolling split, Overwrite, fanout disabled). Shared
  fixture/readers live in `tests/sorted_insert_shared/mod.rs`, reused by all three binaries via
  `mod sorted_insert_shared` (the `rewrite_size_shared` pattern). `javap -c -p` of
  `org.apache.iceberg.types.Comparators` in
  `/tmp/ic-build/.ivy2/jars/org.apache.iceberg_iceberg-spark-runtime-4.1_2.13-1.11.0.jar`
  (this lane, 1.11.0): `<clinit>` maps Boolean/Integer/Long/Float/Double/Date/Time/Timestamp
  with and without zone (plus nano variants)/UUID to `naturalOrder`, String to `charSequences`,
  Binary to `unsignedBytes`, Decimal to `naturalOrder` (the `forType(PrimitiveType)` fallback),
  Unknown to `naturalOrder` wrapped `nullsFirst`. `CharSeqComparator.compare` walks `charAt`
  with a high-surrogate bias (lone high surrogate sorts after a non-surrogate; both-high or
  neither compares `Character.compare`), which orders well-formed strings exactly like UTF-8
  unsigned bytes — the string pin (`a < b < ä < 𝄞`, BMP before supplementary) rests on that
  equivalence, verified, not recalled. Spark's file order is Spark SQL's ordering for the sort
  expression, not the Iceberg comparator: for identity keys over these types both orders agree
  (Spark float ordering is `Float.compare`; Spark 4 strings are `UTF8_BINARY`; decimals compare
  numerically; timestamps compare instants; booleans `false < true`; binaries compare unsigned).
  Where they could differ: bucket/temporal/truncate keys sort by the Spark expression result —
  the fork sorts the same Iceberg transform result, and the bucket/decimal-truncate pins assert
  sortedness-by-key (same-function key oracle, independent order property), not Spark's exact
  bytes. No Spark oracle cell covers these types in this lane; the pins are unit-level.
- L-04 [P3]: every unresolvable-order fallback in `write_sort_plan` now stamps `Some(0)` (was
  `None`), matching Java `DataFiles.Builder` defaulting `sortOrderId` to 0. After L-02 the only
  reachable fallback is `Unknown` (plus truly missing columns, which a bound default order
  cannot name). Pin: the unknown-transform test asserts `Some(0)` with no `SortExec`.

## Round 2 red phase (base `27afcba4`, before the fix)

- `insert_into_float_order_sorts_nan_last_asc`: tail `[1, 5]` vs `{1, 3}` — +NaN and +Inf last,
  -NaN (k=3) displaced forward (Arrow totalOrder). All four float/double ASC/DESC pins red.
- `insert_into_nested_identity_order_sorts_by_child` and
  `insert_into_two_key_order_with_nested_key_sorts_by_both`: file rows in input order, stamp
  `None` (nested miss aborted the whole order).
- `insert_into_unknown_transform_order_writes_with_zero_stamp`: `Error during planning:
  Partition column '_partition' not found in schema` (fanout-off unpartitioned fallback).
- Green on base (regression pins, not divergence pins): string, decimal, timestamptz, boolean,
  day, hour, truncate string/decimal, null-through-bucket, rolling, overwrite, fanout-off,
  plus all 8 round-1 tests. Two test-authoring defects found while going red, both fixed in the
  tests (no production change): a helper that returned file paths after dropping the fixture's
  `TempDir` (paths dangle — helper now returns the fixture), and a `BinaryArray` downcast where
  the parquet read-back is `LargeBinary`.

## Round 2 test adequacy (one knob at a time)

- M1 — `CanonicalFloatExpr` wrap removed (key passes through): 4 red out of 13
  (`sorted_insert`; exactly the float/double ASC/DESC pins).
- M2 — nested resolution reverted to top-level-only lookup: 2 red out of 5
  (`sorted_insert_writer`; exactly the two nested pins).
- M3 — unknown-transform fallback stamp back to `None`: 1 red out of 13 (the stamp assert).
- M4 — `_partition`-presence guard removed: 1 red out of 13, failing with the exact
  `_partition not found` planning error.
- Restore + re-run after each: 13/10/5 green.
- Vacuity sweep: order asserts compare production-written file bytes against hand-computed
  oracle orders (float k-order, nested pairs, type value orders); same-function key oracles
  (bucket, decimal-truncate) assert the independent sortedness-by-key property plus null
  placement; stamps compare manifest entries against table-metadata read-back; shape asserts
  (file counts, row counts, `SortExec` absence) distinguish action from read identity
  (overwrite asserts the replaced row count 100, rolling asserts ≥2 files and 20 000 rows).
  No `=`-null predicates. The NaN tie pair is asserted as a set (Arrow unstable sort; Iceberg
  requires no stability) — recorded, not hidden.

## Round 2 gates

- `cargo test -p iceberg-datafusion --test sorted_insert`: 13 passed, 0 failed.
- `cargo test -p iceberg-datafusion --test sorted_insert_types`: 10 passed, 0 failed.
- `cargo test -p iceberg-datafusion --test sorted_insert_writer`: 5 passed, 0 failed.
- (Neighbour lib/integration gates re-run at commit time; see handback.)
