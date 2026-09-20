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

# Ledger — F-CONTAINER-ACCESSOR-1: `IS NULL` on a list, map or struct column (Q-23a-C / IPI-34)

**Ledger id:** `F-CONTAINER-ACCESSOR-1`
**Branch:** `fix/f-container-accessor-1` (cut off fork `main` = `9cb63641`)
**Scope:** devin-worker brief F-CONTAINER-ACCESSOR-1
**Model:** swe-2-high

## 1. The gap

`Schema::build_accessors` (`crates/iceberg/src/spec/schema/mod.rs`) builds a
`StructAccessor` only for `Type::Primitive` fields and for primitive descendants inside
`Type::Struct` fields (`build_accessors_nested`). A `Type::List`, `Type::Map`, the
struct-typed field itself, and nested list/map/struct fields get **no accessor**.
`Reference::bind` (`crates/iceberg/src/expr/term.rs`) requires an accessor for every
named field, so binding `xs IS NULL`, `mp IS NULL`, `st IS NULL`, `deep.inner IS NULL`,
or `deep.inner.ys IS NULL` fails with `DataInvalid: Accessor for Field <name> not found`.
RePark therefore has to refuse these predicates (gap rows TY-STRUCT, TY-ARRAY,
TY-MAP-LIST, TY-NESTED-DEEP; IPI-34).

## 2. Java evidence (measured, `javap -c -p` on the Iceberg 1.11.0 runtime jar)

Class `org.apache.iceberg.Accessors$BuildPositionAccessors`:

- `struct(Types$StructType, List<Map<Integer,Accessor>>)` — for **every** field `i` of
  the struct it executes `map.put(field.fieldId(), Accessors.newAccessor(i, field.type()))`
  (bytecode offsets 132–151: `NestedField.fieldId()` at 133–138, `NestedField.type()` at
  141–145, `Accessors.newAccessor(I, Type)` at 148). The accessor is built from the
  field's full `Type`, whatever that type is — list, map, struct, variant included.
- For each entry of the child's own accessor map it additionally wraps the nested
  accessor: `map.put(childId, Accessors.newAccessor(i, field.isOptional(), inner))`
  (offsets 51–129; `isOptional` read at 103–110, `newAccessor(I,Z,Accessor)` at 120).
- `variant(...)` returns `null` — a variant field has no child accessor map but still
  receives its own `PositionAccessor` via the unconditional put at 132–151.
- `TypeUtil$SchemaVisitor` base methods `list`, `map`, `field`, `primitive` all return
  `null` (verified: each is `aconst_null; areturn`). So a list/map field contributes no
  *child* accessor map — **element/key/value field ids never get accessors** — while the
  list/map field itself does.

`Accessors$PositionAccessor.get(StructLike)` is `struct.get(position, javaClass)` where
`javaClass` is the field type's Java class (`List`, `Map`, `StructLike`) — a container
accessor returns the raw container value (bytecode offsets 0–14).

`org.apache.iceberg.expressions.UnboundPredicate.bindUnaryOperation` (measured):

- `IS_NULL` folds to `Expressions.alwaysFalse()` iff `!term.producesNull()` **and**
  `allAncestorFieldsAreRequired(struct, term.ref().fieldId())` (offsets 40–69);
  `NOT_NULL` folds to `alwaysTrue()` symmetrically (offsets 101–130).
  `BoundReference.producesNull()` is `field.isOptional()`, and
  `allAncestorFieldsAreRequired` is `TypeUtil.ancestorFields(schema, fieldId).stream()
  .allMatch(NestedField::isRequired)` (the private method, offsets 0–23).
  **Consequence:** a *required* leaf under an *optional* parent does **not** fold — Java
  binds it and evaluates it. The fork today folds on `field.required` alone, which is a
  silent wrong answer for required-under-optional fields (`deep.inner`, `st.a` when
  required). This change ports the ancestor rule.
- `IS_NAN`/`NOT_NAN` reject non-floating terms with `ValidationException` (offsets
  162–251) — the fork already mirrors this in `bind_leaf`.

`org.apache.iceberg.expressions.NamedReference.bind` resolves the `NestedField` and calls
`Schema.accessorForField(fieldId)` — a `Map.get` that returns `null` when absent, which
`BoundReference` stores unchecked; evaluation then NPEs. So for `xs.element` (no accessor
in Java either) Java fails at eval; the fork's typed `DataInvalid` at bind is the
strictly-better equivalent and stays.

**Comparisons on container columns:** `UnboundPredicate.bind` routes binary/set literals
through `Literal.to(term.type())`, which returns `null` for a non-primitive target;
`bind` then raises `ValidationException`. So `xs = ...`, `st < ...` **fail at bind** in
Java — they are never evaluated. The fork's `Datum::to(&field.field_type)` already
returns `Err(DataInvalid)` for non-primitive targets in `bind_leaf` (binary and set
arms), which is the same bind-time typed failure. **Decision: keep it — no container
comparison semantics are invented.**

## 3. Spark oracle (run-25d container-null oracle, Iceberg 1.11.0, format v2 and v3 — identical)

Table `(id BIGINT, st STRUCT<a STRING, b INT>, xs ARRAY<INT>, mp MAP<STRING,INT>,
deep STRUCT<inner STRUCT<x STRING, ys ARRAY<INT>>>)`, four rows:

| id | st | xs | mp | deep |
|----|----|----|----|------|
| 1 | `{a:'aa',b:1}` | `[1,2]` | `{k:1}` | `{inner:{x:'xx',ys:[1]}}` |
| 2 | NULL | NULL | NULL | NULL |
| 3 | `{a:NULL,b:NULL}` | `[]` | `{}` | `{inner:{x:NULL,ys:NULL}}` |
| 4 | `{a:'dd',b:4}` | `[NULL]` | `{k:NULL}` | `{inner:NULL}` |

Note row 3: an **empty** array/map is NOT NULL, and a struct whose fields are all NULL
is NOT NULL. Row 4: a NULL struct child inside a live parent is NULL.

| cell | Spark ids |
|---|---|
| `st IS NULL` | [2] |
| `st IS NOT NULL` | [1,3,4] |
| `xs IS NULL` | [2] |
| `xs IS NOT NULL` | [1,3,4] |
| `mp IS NULL` | [2] |
| `mp IS NOT NULL` | [1,3,4] |
| `st.a IS NULL` | [2,3] |
| `st.a IS NOT NULL` | [1,4] |
| `deep.inner IS NULL` | [2,4] |
| `deep.inner.x IS NULL` | [2,3,4] |
| `deep.inner.ys IS NULL` | [2,3,4] |
| `st IS NULL OR id = 1` | [1,2] |
| `st IS NOT NULL AND id > 2` | [3,4] |

## 4. What the fork does today (per cell)

Scratch bind probe (temporary `#[test]`, since removed) printed:

```text
st IS NULL -> DataInvalid Accessor for Field st not found
xs IS NULL -> DataInvalid Accessor for Field xs not found
mp IS NULL -> DataInvalid Accessor for Field mp not found
st.a IS NULL -> binds today (nested primitive leaf accessor exists)
```

| cell | today |
|---|---|
| `st`/`xs`/`mp` `IS NULL` / `IS NOT NULL` | `DataInvalid` at bind — accessor missing |
| `st.a` null tests | **bind OK**, then the Parquet `RowFilter` (`PredicateConverter.project_column`) resolves the leaf's projection to the top-level `st` StructArray and fails `ArrowError: "Does not support struct column yet."`; on the post-decode residual path (`evaluate_predicate_to_mask`, Avro/`_pos`) `st.a` is absent from the top-level field-id map → `is_null` → all-`true` — **silent wrong answer** |
| `deep.inner IS NULL` | `DataInvalid` at bind (inner is a struct — no accessor) |
| `deep.inner.x` null tests | binds (primitive leaf accessor exists), same RowFilter struct error / residual all-`true` as `st.a` |
| `deep.inner.ys` null tests | `DataInvalid` at bind (ys is a list — no accessor) |
| `st IS NULL OR id = 1`, `st IS NOT NULL AND id > 2` | `DataInvalid` at bind via the `st` arm |
| comparisons on `st`/`xs`/`mp` | `DataInvalid` at bind via `Datum::to` (Java parity — keep) |

## 5. Design (what the implementation does, and why)

1. **`Schema::build_accessors`/`build_accessors_nested`** — build a `StructAccessor` for
   **every** struct field (primitive, struct, list, map, variant) at its position and
   full `Type`, wrapping nested accessors for struct descendants — the Java shape above.
   List element / map key / map value field ids still get **no** accessor (Java parity).
2. **`StructAccessor`** — `r#type` widens `PrimitiveType` → `Type`. For
   **optional** primitives the serialized form is byte-identical to `main`
   (`SerdeType::Primitive` renders the same bare type-name string, and the new
   `is_optional` key is skipped when it holds the default `true` — proven by
   `cmp` against a `main` probe, 220 bytes, §7 M16); a **required** primitive
   carries one explicit `"is_optional":false` key, and wire JSON without the key
   still deserializes via the `true` default. `get()` keeps its
   exact behavior on primitive leaves; on a non-primitive leaf it fails `DataInvalid`
   (it is unreachable through bound predicates — comparisons fail at bind, null tests go
   through `is_present`). New `is_present(&Struct) -> Result<bool>`: presence semantics —
   `None`/null parent ⇒ `false`, matching-shape literal ⇒ `true`, wrong-shape literal ⇒
   typed error (Java's `PositionAccessor.get` + `term.eval(struct) == null`).
3. **`bind_leaf` unary fold** — port `allAncestorFieldsAreRequired`: fold `IS NULL`/
   `NOT NULL` for `field.required` only when **all** ancestors are required too.
   Top-level fields have no ancestors, so existing primitive behavior is unchanged
   there; required-under-optional nested fields now bind and evaluate (Java) instead of
   folding to a wrong constant.
4. **`ExpressionEvaluatorVisitor.is_null`/`not_null`** (partition eval) — use
   `is_present` so a container field in a projected predicate answers presence rather
   than erroring on `get`.
5. **`PredicateConverter` (parquet `RowFilter`)** — resolve references by a parquet
   **name path** through the projected batch: leaf field ids keep the existing leaf-map
   resolution (same map, incl. the id-less fallback); non-leaf field ids resolve through
   a new field-id → name-path map built by walking the parquet schema's group nodes.
   `project_column` descends `StructArray`s by name, unioning each parent's validity
   into the child via `arrow::null_propagation::array_with_parent_validity` — a NULL
   struct parent must read as a NULL child (Arrow never propagates validity downward).
   The `RowFilter` mask gains the whole subtree's leaves for group predicates.
6. **`RecordBatchPredicateEvaluator`** (post-decode residual: Avro/ORC/`_pos`/delete
   paths) — same path resolution over the batch's own field-id metadata (nested arrow
   fields carry `PARQUET_FIELD_ID_META_KEY` too), same parent-validity propagation, so
   `is_null`/`not_null` answer on list/map/struct arrays and nested leaves.
7. **Pruning** — no semantic change needed: manifest projection already erases
   non-partition container columns (`AlwaysTrue`), inclusive/strict metrics default to
   might-match on absent stats, row-group metrics and the page index treat unmapped
   field ids as might-match. Pins prove a container-column filter never drops a file
   holding a matching row.
8. **`object_cache` charge accounting** — `schema_accessor_charge` is updated to the new
   accessor tree shape (every field owns an accessor; every nested field's accessor is a
   `Box` node at its depth; struct accessors clone their `StructType` field list).

## 6. Red-first pins (added before the implementation)

- `table.scan().with_filter(...)` over a fixture table of the oracle shape — all 13
  cells above.
- Accessor-existence unit pins: `accessor_by_field_id` returns `Some` for list, map,
  struct, nested struct, and nested list/map field ids, and `None` for element/key/value
  ids.
- `is_present` unit pins incl. empty-list/map and all-null-struct presence, and
  null-parent propagation.

## 7. Mutation evidence

Protocol: one mutation at a time, applied to a clean tree (only this ledger
uncommitted), restored by file copy afterwards; restore confirmed by
`git diff --name-only` plus a green re-run of the reddened tests. Each entry names
the command, the population `M` that run reported, and the tests that went red.
Baselines this session: `cargo test -q -p iceberg --lib` → 4285 passed, 0 failed,
9 ignored; `cargo test -q -p iceberg-datafusion --lib` → 376 passed, 1 failed (the
RULE-5 out-of-scope `list_null_tests::select_where_xs_is_null_returns_the_null_row`),
1 ignored. The verification critic's list was not in the repo, so M1–M8 below are the
round-2b worker's own one-per-knob set, each actually executed.

| # | mutation (file) | command / population | red |
|---|---|---|---|
| M1 | `build_accessors` primitive-only (`spec/schema/mod.rs`) | iceberg `--lib` / 4294 | 11: `record_batch_predicate_container_tests::container_null_predicates_match_spark_oracle_on_materialized_batch`, `predicate_container_tests::test_bind_comparison_on_container_column_fails_at_bind`, `predicate_container_tests::test_bind_is_null_on_container_columns`, 3× `object_cache::charge_tests::test_schema_accessor_charge_*`, 4× `scan::container_null_tests::*spark_oracle*`, `accessor_tests::test_build_accessors_includes_container_and_struct_fields` |
| M1 | same | datafusion `--lib` / 378 | 4: `is_null_on_a_list/map/struct_column_is_pushed`, `is_null_on_a_container_column_composes_with_its_neighbours`. NOTE: the out-of-scope `list_null` test flips red→green here (see §9 V-note) |
| M2 | ancestor-optional dropped, leaf required-ness only (`spec/schema/mod.rs`) | iceberg `--lib expr::` / 416 | 2: both `*_required_leaf_under_optional_parent_does_not_fold` |
| M3a | partition evaluator `is_present`→`get` (`expression_evaluator.rs`) | iceberg `--lib expr::` / 416 + `scan::` / 254 | 0 — PIN GAP, closed by the direct pin below |
| M3b | M3 re-run after the pin | iceberg `--lib expr::` / 417 | 1: `predicate_container_tests::test_partition_evaluator_answers_container_null_tests_through_presence` |
| M4 | group leaf-lists dropped from `RowFilter` plan (`row_filter_plan.rs`) | iceberg `--lib arrow::` / 504 + `scan::` / 254 | 2 (scan only): `test_filter_on_arrow_container_null_predicates_match_spark_oracle`, `..._under_page_index_row_selection` |
| M5 | parent-validity propagation dropped (`record_batch_predicate.rs`) | iceberg `--lib arrow::` / 504 + `scan::` / 254 | 1 (arrow only): `..._on_materialized_batch` (row 2's valid-but-empty `ys` under NULL parents) |
| M6 | post-decode residual never applied (`arrow/reader.rs`) | iceberg `--lib scan::` / 254 + `arrow::` / 504 | 2 (scan only): `..._without_field_ids`, `..._with_name_mapping` |
| M7 | `Residual` fallback forced to Push (`row_filter_plan.rs`) | iceberg `--lib scan::` / 254 + `arrow::` / 504 | 2 (scan only): `..._without_field_ids`, `..._with_name_mapping` |
| M8 | push gate requires primitive (`expr_to_predicate.rs`) | datafusion `--lib` / 378 | 4: the same push pins as M1. NOTE: out-of-scope `list_null` flips red→green here too |
| M9 | page-index group fail-open removed (`page_index_evaluator.rs`) | iceberg `--lib page_index` / 18 + `scan::` / 254 | 1 in each pop, same test: `..._under_page_index_row_selection` |
| M10 | fallback stamping positional-only, name match dropped (`arrow/reader.rs`) | iceberg `--lib scan::` / 254 + `arrow::` / 504 | 1 (scan only): `..._without_field_ids` (mapping case greens: that branch stamps real ids) |
| M11a | row-3 fixture back to `{a:NULL,b:3}` (both oracle fixtures) | iceberg `--lib container` / 49 | 0 — PIN GAP, closed by the `st.b` cells below |
| M11b | M11 re-run after the pins | `scan::container` / 4 + `record_batch_predicate_container` / 1 | 5: all four scan oracle tests + `..._on_materialized_batch` |
| M12 | charge skips non-primitive accessors (`io/object_cache.rs`) | iceberg `--lib object_cache` / 25 | 3: `test_schema_accessor_charge_counts_actual_arc_and_box_nodes`, `..._grows_with_every_accessor_map_entry`, `..._matches_the_accessor_map_schema_builds` |
| M13 | `accessor_by_field_id` falls back to any accessor (`spec/schema/mod.rs`) | iceberg `--lib` / 4295 | 2: `test_build_accessors_omits_element_key_value_and_container_nested_ids`, `test_bind_is_null_on_element_key_and_value_paths_fails_at_accessor_lookup` |
| M13 | same | datafusion `--lib` / 378 | 5: `binary_on_an_accessorless_leaf_is_not_pushed`, `in_list_on_an_accessorless_leaf_is_not_pushed`, `is_null_on_a_list_element_name_is_not_pushed`, `is_null_on_an_accessorless_leaf_still_drops_only_its_own_conjunction`, + baseline-red `list_null` (stays red) |
| M14 | `is_present` `None` arm → `true` (`expr/accessor.rs`) | iceberg `--lib expr::` / 417 + `scan::` / 254 | 4 (expr only): both `accessor::tests::test_is_present_*`, `test_partition_evaluator_answers_container_null_tests_through_presence`, pre-existing `expression_evaluator::tests::test_null_partition_value_truth_table_nulls_first` |
| M15 | `is_present` wrong-shape arm → `true` (`expr/accessor.rs`) | iceberg `--lib expr::` / 417 + `scan::` / 254 | 1 (expr only): `test_is_present_propagates_null_parent_and_rejects_wrong_shape` |
| M16 | `skip_serializing_if` removed (`expr/accessor.rs`) | iceberg `--lib expr::` / 417 | 1: `test_optional_primitive_bound_predicate_json_omits_default_is_optional` |
| M17 | `is_optional` forced `true` (`spec/schema/mod.rs`) | iceberg `--lib` / 4295 | 15: `test_required_primitive_bound_predicate_json_carries_explicit_is_optional`, 9× `predicate::tests::*` (8 `test_bind_*` + `test_bound_predicate_rewrite_not_always_true_false`), 2× `term::tests::test_bind_reference*`, `bound_predicate_visitor::tests::test_not_null`, `inclusive_metrics_evaluator::test::test_required_column`, `residual_nested_name_tests::tests::a_nested_residual_keeps_its_full_column_name` |

Gap-closure pins written during this step (in the step-7 commit): M3 →
`test_partition_evaluator_answers_container_null_tests_through_presence`
(`predicate_container_tests.rs`); M11 → `st.b IS NULL` / `st.b IS NOT NULL` cells in
`container_oracle_cases()` (all four scan oracle tests) and in
`container_null_predicates_match_spark_oracle_on_materialized_batch`. The `st.b`
expectations derive from the fixture's row-3 definition (`{a:NULL,b:NULL}` → `st.b`
null exactly in rows 2–3), not from a fresh Spark run — §3 stays the measured oracle.

## 8. Residual gaps observed (out of scope for this slice)

- ~~Parquet files **without** embedded field ids ...~~ **WITHDRAWN 2026-09-20 (round
  2a, `ec6da3c9`):** id-less files now stamp top-level ids by table-schema name match
  with a Java-counter fallback (`add_fallback_field_ids_to_arrow_schema`), project
  unmapped groups by id (`unmapped_group_leaf_indices`), and route
  present-but-unmapped predicates to a post-decode residual (`RowFilterPlan::Residual`)
  instead of a wrong Push. Pinned by the id-less oracle cases in
  `scan/container_null_tests.rs` (§7 M6/M7/M10).
- `ExpressionEvaluatorVisitor` (partition evaluation) answers null tests through
  `is_present`, pinned directly at the visitor (§7 M3). Partition specs cannot legally
  contain non-primitive source fields, so the container arm stays unreachable in
  production; the direct pin guards the contract, not a reachable path.
- Recursive name-mapping application (`ApplyNameMapping` descends; the fork maps
  top-level only) — see divergence (ii) in §13.

## 9. Round-2 findings dispositions

The critic reports are not in the repo; the mapping below is reconstructed from the
round-2a/2b commits, one row per finding id named in the round-2 brief.

| finding | disposition | where |
|---|---|---|
| V-01 (page-index group ids used `CantMatch`, dropping matching rows under row selection) | CLOSED | `page_index_evaluator.rs::field_id_names_a_group` fail-open (`4b85af4a`) + `test_container_null_predicates_match_spark_oracle_under_page_index_row_selection` (§7 M9) |
| V-02 (element/key/value accessor absence + `xs.element` bind failure unpinned) | CLOSED | `test_build_accessors_omits_element_key_value_and_container_nested_ids` + `test_bind_is_null_on_element_key_and_value_paths_fails_at_accessor_lookup` (round 2b step 5, §7 M13) |
| V-03 (`is_present` pinned only through `table.scan()`, never directly) | CLOSED | `test_is_present_treats_empty_containers_and_all_null_struct_as_present` + `test_is_present_propagates_null_parent_and_rejects_wrong_shape` (round 2b step 5, §7 M14/M15) |
| V-04 (id-less files answer container/nested null tests wrong; §8 parked it) | CLOSED | mapping/positional stamping + unmapped→Residual (`ec6da3c9`), §8 parking withdrawn above (§7 M6/M7/M10) |
| V-05 / Q-26d-7 (JSON "byte-identical" claim false: `is_optional` always serialized) | CLOSED | `skip_serializing_if` on the default + frozen 220-byte string pin + `cmp` proof vs `main` (round 2b step 6, §7 M16/M17) |
| L-01 (stale `is_optional=true` expectation in term bind tests, red since round 1) | CLOSED | `term.rs` bind tests expect `false` for required `bar` (`40bcf119`) |
| L-02 (row-3 fixture `{a:NULL,b:3}`, not the oracle's all-NULL struct; no `st.a IS NOT NULL` pin) | CLOSED | all-NULL row 3 + `st.a IS NOT NULL` cell (`1a586446`); `st.b` cell added round 2b to kill M11 (§7 M11) |

V-note (out of scope, for the `list_null` clerk): the RULE-5 red test
`physical_plan::list_null_tests::select_where_xs_is_null_returns_the_null_row` flips
red→green under both M1 (primitive-only accessors) and M8 (primitive-only push gate),
and stays red under M13 (accessor fallback). Its red state on this branch depends on
container accessors existing and pushing — evidence it is branch-caused, not red on
`main`. Untouched here per RULE 5.

## 10. Q-26d-4 missing-column audit

What each layer answers when the predicate's column (or its stats) is absent from the
file being pruned. "Fail open" = keep the file/row (never drop a match).

| layer | absent input | answer | fail-open pin |
|---|---|---|---|
| manifest evaluator | partition summary absent | might-match | existing suite |
| inclusive/strict metrics | bounds/counts absent | might-match | existing suite |
| row-group metrics | field id unmapped | might-match | existing suite |
| page index | leaf id unmapped, `IsNull`/`NotNull` | `MightMatch` select-all / `CantMatch` skip-all by op | existing suite |
| page index | **group** id unmapped (list/map/struct) | select-all (`field_id_names_a_group`, V-01 fix) | `..._under_page_index_row_selection` |
| RowFilter leaf map | field id unmapped, ids present | id ignored (schema evolution) | existing suite |
| id-less RowFilter plan | predicate id present-but-unmapped | `Residual`, never a partial Push (V-04 fix) | id-less oracle cases |
| record-batch evaluator | column absent from batch | NULL semantics (`is_null`→true) | `record_batch_predicate` suite |
| DataFusion push gate | no accessor for the term | not pushed (`term_binds_soundly`) | `*_is_not_pushed` pins |

## 11. Clause close-out

| clause | status | pin |
|---|---|---|
| C-001 container/struct/nested-container ids own an accessor | PROVEN | pins: f-container-accessor-1/C-001 (`spec::schema::accessor_tests::test_build_accessors_includes_container_and_struct_fields`) |
| C-002 element/key/value/container-nested ids own none; `xs.element`/`mp.key`/`mp.value` bind fails `DataInvalid` at the accessor lookup | PROVEN | pins: f-container-accessor-1/C-002 (`test_build_accessors_omits_element_key_value_and_container_nested_ids`, `test_bind_is_null_on_element_key_and_value_paths_fails_at_accessor_lookup`) |
| C-003 `IS NULL`/`NOT NULL` fold only when the leaf and all ancestors are required | PROVEN | pins: f-container-accessor-1/C-003 (`test_bind_is_null_required_leaf_under_optional_parent_does_not_fold`, `test_bind_is_not_null_required_leaf_under_optional_parent_does_not_fold`) |
| C-004 `is_present`: empty list/map present, all-NULL struct present, `None` absent, wrong shape `DataInvalid` | PROVEN | pins: f-container-accessor-1/C-004 (`test_is_present_treats_empty_containers_and_all_null_struct_as_present`, `test_is_present_propagates_null_parent_and_rejects_wrong_shape`) |
| C-005 partition evaluator answers container null tests through presence | PROVEN | pins: f-container-accessor-1/C-005 (`test_partition_evaluator_answers_container_null_tests_through_presence`) |
| C-006 `RowFilter` + batch evaluator match the Spark oracle (15 scan cells + 12 batch cells, incl. `st.b` both parities) | PROVEN | pins: f-container-accessor-1/C-006 (`test_filter_on_arrow_container_null_predicates_match_spark_oracle`, `container_null_predicates_match_spark_oracle_on_materialized_batch`) |
| C-007 id-less files answer container/nested null tests via stamping + post-decode residual | PROVEN | pins: f-container-accessor-1/C-007 (id-less oracle cases in `scan/container_null_tests.rs`) |
| C-008 pruning never drops a matching row (manifest/metrics/row-group/page-index fail open) | PROVEN | pins: f-container-accessor-1/C-008 (`test_container_null_predicates_match_spark_oracle_under_page_index_row_selection`) |
| C-009 optional-primitive `BoundPredicate` JSON byte-identical to `main`; required carries explicit `false`; keyless wire JSON reads | PROVEN | pins: f-container-accessor-1/C-009 (`test_optional_primitive_bound_predicate_json_omits_default_is_optional`, `test_required_primitive_bound_predicate_json_carries_explicit_is_optional`) |
| C-010 container comparisons fail at bind; charge covers the new tree; DF pushes container null tests | PROVEN | pins: f-container-accessor-1/C-010 (`test_bind_comparison_on_container_column_fails_at_bind`, `object_cache_charge_tests`, `is_null_on_a_*_column_is_pushed`) |

## 12. Coverage attestation

COVERAGE_ATTESTATION:
AT-1 covers C-001 via the container-accessor existence pin;
AT-2 covers C-002 via the element/key/value absence and bind-failure pins;
AT-3 covers C-003 via the required-under-optional no-fold pins;
AT-4 covers C-004 via the direct `is_present` pins;
AT-5 covers C-005 via the direct partition-evaluator pin;
AT-6 covers C-006 via the Spark-oracle scan and batch pins;
AT-7 covers C-007 via the id-less oracle pins;
AT-8 covers C-008 via the page-index row-selection oracle pin;
AT-9 covers C-009 via the frozen-JSON and explicit-false pins;
AT-10 covers C-010 via the bind-rejection, charge, and pushdown pins.

## 13. Declared divergences

| # | divergence | evidence |
|---|---|---|
| (i) | DIVERGENCE-DECLARED — Branch-3 (no field ids, no name mapping) on NON-DENSE top-level ids: the fork stamps by table position with a Java-counter fallback and projects by id; Java's `pruneColumnsFallback` misaligns and returns unusable answers (bytecode-proven against iceberg-parquet 1.10.0). Flat/dense stays Java-identical. | round-2a bytecode evidence; id-less oracle pins §7 M10 |
| (ii) | DIVERGENCE-DECLARED — Branch-2 nested predicates go to a post-decode residual instead of a pushed RowFilter, because the fork applies name mappings top-level-only while Java's `ApplyNameMapping` recurses. Same answers, less pushdown; recursive mapping is a declared future gap (§8). | id-less nested oracle pins §7 M6/M7 |
