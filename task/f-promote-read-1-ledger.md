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

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Status |
|---|---|---|---|
| C-001 | `InclusiveMetricsEvaluator` keeps a file whose bounds were written as `int`/`float` for `<`, `<=`, `>`, `>=` and `IN` under a `long`/`double` reference, and still prunes it when the promoted bounds exclude the literal. | `inclusive_metrics_keep_an_int_bounded_file_for_long_predicates`, `inclusive_metrics_keep_a_float_bounded_file_for_double_predicates`; red on base. | OPEN |
| C-002 | `StrictMetricsEvaluator` never claims `ROWS_MUST_MATCH` for `<>`/`NOT IN` on a pre-promotion file that holds the literal, and does claim it when the promoted bounds prove every row matches. | `strict_metrics_decide_an_int_bounded_file_under_long_predicates`; red on base. | OPEN |
| C-003 | `StructAccessor::get` reads an `Int`/`Float` partition literal under a `long`/`double` accessor as the promoted datum and still refuses an unrelated kind. | `partition_accessor_reads_pre_promotion_literals_under_the_promoted_type`; red on base. | OPEN |
| C-004 | `PartitionKey::new` accepts a tuple written before a legal promotion and stores the promoted tuple. | `partition_key_new_promotes_a_pre_promotion_tuple`; red on base. | OPEN |
| C-005 | A mixed-era table (two pre-promotion files, one post-promotion file) answers `<`, `>`, long `IN` and — with row selection on — `<=` with every matching row. | `mixed_era_range_and_in_filters_return_pre_promotion_rows`; red on base. | OPEN |
| C-006 | A promoted identity partition source answers `=` and `<` and plans every `FileScanTask.partition` as `long`. | `promoted_identity_partition_source_filters_and_plans_long_partitions`; red on base. | OPEN |
| C-007 | An equality delete written after the promotion applies to a pre-promotion data file in the same partition. | `equality_delete_written_after_promotion_applies_to_a_pre_promotion_partition`; red on base. | OPEN |
| C-008 | `ReplacePartitions` after the promotion drops the pre-promotion file of the replaced partition; `overwrite_by_row_filter(id = 7)` replaces the promoted identity partition and keeps the other one. | `replace_partitions_after_promotion_drops_the_pre_promotion_partition`, `overwrite_by_row_filter_on_a_promoted_identity_partition_replaces_it`; red on base. | OPEN |
| C-009 | Gates: the pins green after the fix, `cargo test -p iceberg --lib`, `cargo clippy -p iceberg --all-targets -- -D warnings`, `cargo fmt`, the Rust file-size check, the comment fence. | Command -> result below. | OPEN |

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
