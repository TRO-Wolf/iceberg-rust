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

# F-PROMOTE-READ-1 — manifest values written before a type promotion read under the promoted type

**Date:** 2026-09-16. **Branch:** `fix/ice-promote-read-1`.
**Base:** `edc38c6aa5cdbe132235f4fefc01d3066d6cff23`.
**Model:** claude-opus-5.
**Path:** HIGH because the defect is silent row loss and silent duplicate writes on a
spec-legal schema evolution, and the fix touches scan planning, both metrics evaluators,
the partition accessor, `PartitionKey::new` and the replace-partitions resolver.

This ledger retires when the fork change merges and RePark's pin bump consumes it
(RePark unit ICE-PROMOTE-READ-1, run 19a).

## The defect

After a legal Iceberg promotion (`int -> long`, `float -> double`,
`decimal(P,S) -> decimal(P',S)`), a table holds manifests written under the old
schema. The fork decodes every manifest under the schema embedded in that manifest
(`spec/manifest/mod.rs` `try_from_avro_bytes_with_schema_fallback` passes
`metadata.schema` to `_serde.rs` `parse_bytes_entry` and to the partition struct
decode). A pre-promotion data file therefore carries `Datum{Int}` lower/upper bounds
and an `Int` partition literal, while a scan or a write binds its predicate and its
partition type under the current schema (`Long`).

Java never meets the mismatch: `ManifestReader` projects the Avro file onto the
table's current specs (Avro promotes `int` to `long` on read), and
`InclusiveMetricsEvaluator` / `StrictMetricsEvaluator` decode each bound with
`Conversions.fromByteBuffer(ref.type(), bytes)`, which widens a 4-byte bound under a
`long` or `double` reference.

Measured consequences in the fork, one per seam:

| Seam | Base behaviour | Consequence |
|---|---|---|
| `InclusiveMetricsEvaluator` `visit_inequality` / `in` | `PartialOrd::lt(Datum{Int}, Datum{Long})` is `None`, so `cmp_fn` answers `false` and the arm answers `ROWS_CANNOT_MATCH` | every pre-promotion file is pruned for `<`, `<=`, `>`, `>=`, `IN` — silent row loss; `=` survives because its arm uses the negative form |
| `StrictMetricsEvaluator` `not_eq` / `not_in` | `lower.literal() > datum.literal()` compares `PrimitiveLiteral` by derived variant order (`Int < Long` always); `not_in` retains nothing | `ROWS_MUST_MATCH` claimed for a file that still holds the excluded value — a whole-file drop through `overwrite_by_row_filter` / `delete_from_row_filter` |
| `StructAccessor::get` | `PrimitiveType::Long.compatible(Int)` is false | every partition filter, residual and `overwrite_by_row_filter` on a promoted identity/truncate partition source fails loud `Literal Int(7) … not compatible with accessor type long` |
| scan planning `FileScanTask.partition` and `DeleteFileIndex` keys | tuples keep the manifest type | an equality delete written after the promotion (`Long(7)`) never finds a pre-promotion data file (`Int(7)`) — deleted rows come back; RePark's MoR DELETE/UPDATE hand `Int` tuples to `PartitionKey::new` |
| `PartitionKey::new` | validates the tuple kind against the partition type under the given schema | MoR position-delete and DV writers refuse `Partition value for field … is not compatible with its partition type long` |
| `SnapshotProducer::resolve_partition_deletes` | matches `(spec_id, partition)` by exact `Struct` equality | `ReplacePartitions` after a promotion keeps the old partition — silent duplicate rows |
| `PageIndexEvaluator` (row selection on) | builds `Datum::new(Long, PrimitiveLiteral::Int(v))` from an INT32 column index | every page of a pre-promotion file is skipped |

## Decisions

- **D-1** Promotion follows Java `TypeUtil.isPromotionAllowed` exactly
  (`spec/schema/type_promotion.rs`): `int -> long`, `float -> double`,
  `decimal(P,S) -> decimal(P',S)` with `P <= P'`. Any other kind mismatch keeps today's
  behaviour (loud error or no pruning); this unit adds no new coercion.
- **D-2** Bounds are promoted where they meet a bound reference (the Java
  `fromByteBuffer(ref.type(), …)` point), not at manifest decode, so every caller of
  the two metrics evaluators — scan pruning, conflict detection, row-filter deletes —
  is covered without touching the ~100 manifest-load call sites.
- **D-3** Partition tuples are promoted at the three places where a manifest tuple meets
  a current-schema partition type: the accessor, the scan-planning entry context (which
  feeds `FileScanTask.partition` and the delete-file index for data and delete entries
  alike), and `PartitionKey::new`. `resolve_partition_deletes` compares promoted tuples.
- **D-4** No on-disk format change: manifests are never rewritten by a read.
- **D-5** File-size ceilings only move down: production edits in capped files are
  line-neutral; helpers live in the new `spec/promotion.rs`, pins in
  `spec/promotion_tests.rs`.

## Implemented fix

`crates/iceberg/src/spec/promotion.rs` (new, `pub(crate)` only) holds the promotion helpers,
all built on `is_promotion_allowed` and the existing `PrimitiveLiteral::promote_to`:

- `Datum::promoted_to(&Type) -> Cow<Datum>` — the Java `fromByteBuffer(ref.type(), …)` step:
  a legal promotion returns the widened datum, anything else borrows the original.
- `DataFile::promoted_lower_bound` / `promoted_upper_bound(&BoundReference)` — a bound read
  under the reference's type.
- `Struct::promoted_to(&StructType) -> Option<Struct>` — re-types each slot whose literal is
  not compatible with its partition field but becomes compatible after `promote_to`;
  `None` when nothing changes, so the common path allocates nothing.
- `PartitionSpec::validated_promoted_partition` — promote, then the unchanged
  `validate_partition_data`.
- `TableMetadata::current_partition_key(&DataFile)` — `(spec_id, tuple promoted to the spec's
  partition type under the current schema)`.
- `ManifestEntry::with_promoted_partition` — `Arc::clone` unless the tuple changes.
- `Datum::physical(&PrimitiveType, PrimitiveLiteral)` — a page-index bound built under the
  field type from the column index's physical literal.

Seams (production edits in capped files are line-neutral or shrink):

| File | Change |
|---|---|
| `expr/visitors/inclusive_metrics_evaluator.rs` | `lower_bound` / `upper_bound` take the `BoundReference` and return the promoted `Cow`; comparisons use `.as_deref()` |
| `expr/visitors/strict_metrics_evaluator.rs` | same, renamed `lower` / `upper` so the two tuple `if let` lines fit one line; the `field_id` locals only those calls used are gone (1928 → 1922, ceiling lowered) |
| `expr/accessor.rs` | a second arm returns the promoted datum when `promote_to` makes the literal compatible |
| `spec/partition.rs` | `PartitionKey::new` stores `validated_promoted_partition(data, schema)` |
| `transaction/snapshot.rs` | `resolve_partition_deletes` keys data files with `current_partition_key` |
| `expr/visitors/page_index_evaluator.rs` | INT32 and FLOAT column-index arms build bounds with `Datum::physical` |
| `scan/context.rs` | the partition type of each manifest's spec under the scan schema is computed once; every streamed entry (data and delete manifests) carries `with_promoted_partition` |

Two existing pins encoded the pre-fix rule that an `int` literal under a `long` type is refused,
which is exactly the Java-legal promotion this unit implements. Their rejection coverage now uses
a genuinely incompatible kind instead: `accessor::tests::test_accessor_rejects_representation_incompatible_primitives`
(`(Long, Literal::int(7))` → `(Long, Literal::string("7"))`) and
`partition::partition_path_totalisation_tests::partition_key_new_rejects_incompatible_literal`
(`Literal::int(7)` → `Literal::string("7")` in the `long` slot). The `(Int, Literal::long(7))`
narrowing case stays refused.

Named residue (not reachable from RePark's writers, recorded for a later unit):

- `ReplacePartitionsAction::drop_partitions` keys the ADDED files' tuples as given. RePark and
  every in-tree writer type new tuples under the current schema, so the promoted old side matches;
  a caller that hands an `int` tuple for a promoted field would not.
- `ConflictScope::contains` compares a concurrently committed file's tuple by exact equality; a
  stale writer that committed an `int` tuple after the promotion would escape the
  replace-partitions conflict check. Neither action has table metadata in scope today.
- `get_parquet_stat_min_as_datum` answers `None` for `(Long, Int32)` / `(Double, Float)` row-group
  statistics — correct (no pruning), not promoted.

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Status |
|---|---|---|---|
| C-001 | `InclusiveMetricsEvaluator` keeps a file whose bounds were written as `int`/`float` for `<`, `<=`, `>`, `>=` and `IN` under a `long`/`double` reference, and still prunes it when the promoted bounds exclude the literal. | `inclusive_metrics_keep_an_int_bounded_file_for_long_predicates`, `inclusive_metrics_keep_a_float_bounded_file_for_double_predicates`; red on base. | EXECUTION PROVEN |
| C-002 | `StrictMetricsEvaluator` never claims `ROWS_MUST_MATCH` for `<>`/`NOT IN` on a pre-promotion file that holds the literal, and does claim it when the promoted bounds prove every row matches. | `strict_metrics_decide_an_int_bounded_file_under_long_predicates`; red on base. | EXECUTION PROVEN |
| C-003 | `StructAccessor::get` reads an `Int`/`Float` partition literal under a `long`/`double` accessor as the promoted datum and still refuses an unrelated kind. | `partition_accessor_reads_pre_promotion_literals_under_the_promoted_type`; red on base. | EXECUTION PROVEN |
| C-004 | `PartitionKey::new` accepts a tuple written before a legal promotion and stores the promoted tuple. | `partition_key_new_promotes_a_pre_promotion_tuple`; red on base. | EXECUTION PROVEN |
| C-005 | A mixed-era table (two pre-promotion files, one post-promotion file) answers `<`, `>`, long `IN` and — with row selection on — `<=` with every matching row; with row selection on, `f < 2.0` over a float column promoted to `double` keeps the pre-promotion page. | `mixed_era_range_and_in_filters_return_pre_promotion_rows` (red on base), `row_selection_keeps_float_pages_under_a_promoted_double` (added after the fix; red under the FLOAT-arm mutation). | EXECUTION PROVEN |
| C-006 | A promoted identity partition source answers `=` and `<` and plans every `FileScanTask.partition` as `long`. | `promoted_identity_partition_source_filters_and_plans_long_partitions`; red on base. | EXECUTION PROVEN |
| C-007 | An equality delete written after the promotion applies to a pre-promotion data file in the same partition. | `equality_delete_written_after_promotion_applies_to_a_pre_promotion_partition`; red on base. | EXECUTION PROVEN |
| C-008 | `ReplacePartitions` after the promotion drops the pre-promotion file of the replaced partition; `overwrite_by_row_filter(id = 7)` replaces the promoted identity partition and keeps the other one. | `replace_partitions_after_promotion_drops_the_pre_promotion_partition`, `overwrite_by_row_filter_on_a_promoted_identity_partition_replaces_it`; red on base. | EXECUTION PROVEN |
| C-010 | `iceberg-datafusion` `UPDATE` and `DELETE`, copy-on-write and merge-on-read, on a table holding only pre-promotion files update and delete the matching rows instead of failing `column types must match schema types, expected Int64 but found Int32`. | `tests/promoted_type_dml.rs` (4 pins); red on `7e027cca`. | EXECUTION PROVEN |
| C-009 | Gates: the pins green after the fix, `cargo test -p iceberg --lib`, `cargo clippy -p iceberg --all-targets -- -D warnings`, `cargo fmt`, the Rust file-size check, the comment fence. | Command -> result below. | EXECUTION PROVEN |

## Base-red evidence

`CARGO_BUILD_JOBS=10 cargo test -p iceberg --lib spec::promotion_tests` on base
`edc38c6a` with only the pins added exited 101 — **10 of 10 red**, each on its own seam:

```
test spec::promotion_tests::partition_accessor_reads_pre_promotion_literals_under_the_promoted_type ... FAILED
test spec::promotion_tests::inclusive_metrics_keep_an_int_bounded_file_for_long_predicates ... FAILED
test spec::promotion_tests::inclusive_metrics_keep_a_float_bounded_file_for_double_predicates ... FAILED
test spec::promotion_tests::partition_key_new_promotes_a_pre_promotion_tuple ... FAILED
test spec::promotion_tests::strict_metrics_decide_an_int_bounded_file_under_long_predicates ... FAILED
test spec::promotion_tests::overwrite_by_row_filter_on_a_promoted_identity_partition_replaces_it ... FAILED
test spec::promotion_tests::equality_delete_written_after_promotion_applies_to_a_pre_promotion_partition ... FAILED
test spec::promotion_tests::replace_partitions_after_promotion_drops_the_pre_promotion_partition ... FAILED
test spec::promotion_tests::mixed_era_range_and_in_filters_return_pre_promotion_rows ... FAILED
test spec::promotion_tests::promoted_identity_partition_source_filters_and_plans_long_partitions ... FAILED

an int literal reads under long: DataInvalid => Literal Int(7) at position 0 is not compatible with accessor type long
assertion failed: might_match(Reference::new("id").less_than(Datum::long(2)))
assertion failed: might_match(Reference::new("f").less_than(Datum::double(2.0)))
a tuple written before a legal promotion constructs a key: DataInvalid => Partition value for field `id` is not compatible with its partition type `long`
assertion failed: !must_match(Reference::new("id").not_equal_to(Datum::long(1)))
commit: DataInvalid => Literal Int(7) at position 0 is not compatible with accessor type long
equality delete:  left: [(7, 100), (7, 101)]  right: [(7, 101)]
replace partitions:  left: [(7, 100), (7, 999)]  right: [(7, 999)]
mixed-era range:  left: []  right: [(1, 10)]
collect: Unexpected => file scan task generate failed
Source: DataInvalid => Literal Int(1) at position 0 is not compatible with accessor type long

test result: FAILED. 0 passed; 10 failed; 0 ignored; 0 measured; 3685 filtered out
```

The accessor, `PartitionKey::new` and range messages are byte-identical to the RePark
run-19a reproduction (`p_promote_partition`, `p_promote_suspects`, `p_promote_read`).

## Execution evidence

`CARGO_BUILD_JOBS=10 cargo test -p iceberg --lib spec::promotion_tests` after the fix:

```
test spec::promotion_tests::partition_accessor_reads_pre_promotion_literals_under_the_promoted_type ... ok
test spec::promotion_tests::partition_key_new_promotes_a_pre_promotion_tuple ... ok
test spec::promotion_tests::inclusive_metrics_keep_a_float_bounded_file_for_double_predicates ... ok
test spec::promotion_tests::inclusive_metrics_keep_an_int_bounded_file_for_long_predicates ... ok
test spec::promotion_tests::strict_metrics_decide_an_int_bounded_file_under_long_predicates ... ok
test spec::promotion_tests::equality_delete_written_after_promotion_applies_to_a_pre_promotion_partition ... ok
test spec::promotion_tests::replace_partitions_after_promotion_drops_the_pre_promotion_partition ... ok
test spec::promotion_tests::mixed_era_range_and_in_filters_return_pre_promotion_rows ... ok
test spec::promotion_tests::overwrite_by_row_filter_on_a_promoted_identity_partition_replaces_it ... ok
test spec::promotion_tests::promoted_identity_partition_source_filters_and_plans_long_partitions ... ok
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 3685 filtered out; finished in 0.12s
```

First full run after the fix: `3685 passed; 2 failed` — the two existing pins named under
"Implemented fix" still asserted that an `int` literal under a `long` type is refused. After
narrowing them to a string literal:

```
cargo test -p iceberg --lib                               -> 3687 passed; 0 failed; 8 ignored
cargo clippy -p iceberg --all-targets -- -D warnings      -> exit 0
cargo fmt --all -- --check                                -> exit 0
python3 scripts/check_rust_file_size.py                   -> rust-file-size: 465 files clean (100 legacy ceilings)
typos (touched files)                                     -> exit 0
make check-comment-blocks check-agent-artifacts check-matrix-anchors -> all OK
```

Mutation proof — each seam reverted alone, `cargo test -p iceberg --lib spec::promotion_tests`,
file restored from a byte copy (`/tmp/oc-worker/ia-build/fork-mutations.py`, a pattern that does
not apply exactly once hard-fails):

| Mutation | Red pins |
|---|---|
| M1 inclusive evaluator reads raw bounds | 4 — both inclusive unit pins, the mixed-era scan, the promoted identity partition scan |
| M2 strict evaluator reads raw bounds | 1 — `strict_metrics_decide_an_int_bounded_file_under_long_predicates` |
| M3 accessor promotion arm removed | 2 — the accessor pin, `overwrite_by_row_filter_on_a_promoted_identity_partition_replaces_it` |
| M4 `PartitionKey::new` validates the raw tuple | 1 — `partition_key_new_promotes_a_pre_promotion_tuple` |
| M5 `resolve_partition_deletes` compares raw tuples | 1 — `replace_partitions_after_promotion_drops_the_pre_promotion_partition` |
| M6 page index builds `Datum::new(Long, Int)` | 1 — `mixed_era_range_and_in_filters_return_pre_promotion_rows` (the row-selection `<=` assertion) |
| M7 scan planning streams raw tuples | 2 — the cross-era equality delete, the promoted identity partition plan |

Consumer evidence (RePark ICE-PROMOTE-READ-1, local path override, not committed there): the
RePark ledger records the Python pins against the recorded Spark 4.1.2 oracle.

## Second seam — the DataFusion DML execs (found by the RePark pins)

The RePark pin module on the first fix went 157 of 169 green; the 12 red cells were every
single-era `UPDATE`. RePark routes a plain `UPDATE` to this crate's `IcebergUpdateExec`, whose
copy-on-write and merge-on-read paths scan at the snapshot (`Int32` for a pre-promotion file)
and rebuild each batch under the current table schema without widening
(`physical_plan/delete.rs` `table_column_batch`, and the merge-on-read delete's predicate
batch). The same rebuild serves this crate's own `DELETE`.

`CARGO_BUILD_JOBS=10 cargo test -p iceberg-datafusion --test promoted_type_dml` on `7e027cca`:

```
test copy_on_write_delete_after_a_promotion_removes_the_matching_row ... FAILED
test merge_on_read_delete_after_a_promotion_removes_the_matching_row ... FAILED
test copy_on_write_update_after_a_promotion_updates_pre_promotion_rows ... FAILED
test merge_on_read_update_after_a_promotion_updates_pre_promotion_rows ... FAILED
run DELETE FROM catalog.ns.t WHERE id = 6: Arrow error: Invalid argument error: column types must match schema types, expected Int64 but found Int32 at column index 0
run UPDATE catalog.ns.t SET s = 'u5' WHERE id = 5: Arrow error: Invalid argument error: column types must match schema types, expected Int64 but found Int32 at column index 0
test result: FAILED. 0 passed; 4 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.61s
```

Fix: `physical_plan/promotion.rs` (new) `widened_batch(table_schema, columns)` widens a legally
promoted column (`Int32 → Int64`, `Float32 → Float64`, `Decimal128(p,s) → Decimal128(p',s)`) and
then calls `RecordBatch::try_new`, so every other mismatch still fails there. `delete.rs` calls it
at both rebuild sites (line-neutral; the file stays at its 1149 ceiling).

```
cargo test -p iceberg-datafusion --lib --test promoted_type_dml --test integration_datafusion_test
  --test h7_p1_dml_prune --test commit_branch --test row_lineage_cow --test row_lineage_mor
  --test count_star_fold
    lib 216 passed (1 ignored); commit_branch 20; count_star_fold 7; h7_p1_dml_prune 5;
    integration_datafusion_test 87; promoted_type_dml 4; row_lineage_cow 14; row_lineage_mor 5
cargo clippy -p iceberg-datafusion --all-targets -- -D warnings -> exit 0
```

Mutation (`/tmp/oc-worker/ia-build/fork-df-mutation.py`): both `widened_batch` calls reverted to
`RecordBatch::try_new` → `0 passed; 4 failed`; restored.

## Refinement (self-review)

- `Struct::promoted_to` first asks, without allocating, whether any slot needs a promotion
  (`promotable_slot`), and returns `None` before building a tuple otherwise. The first version
  collected a cloned tuple for every manifest entry of every scan and dropped it when nothing
  changed.
- The FLOAT column-index arm of the page-index evaluator had no pin.
  `row_selection_keeps_float_pages_under_a_promoted_double` writes a `float` file, promotes the
  column to `double`, appends a post-promotion file and scans `f < 2.0` with row selection on.

```
cargo test -p iceberg --lib spec::promotion_tests           -> 11 passed
FLOAT arms reverted to Datum::new(field_type.clone(), Float) -> 10 passed; 1 failed
                                                               (row_selection_keeps_float_pages_under_a_promoted_double)
cargo test -p iceberg --lib                                 -> 3688 passed; 0 failed; 8 ignored
cargo clippy -p iceberg --all-targets -- -D warnings        -> exit 0
```
