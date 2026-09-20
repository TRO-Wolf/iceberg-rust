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
2. **`StructAccessor`** — `r#type` widens `PrimitiveType` → `Type`. For primitive types
   the serialized form is byte-identical (`SerdeType::Primitive` renders the same bare
   type-name string), so `BoundPredicate` JSON round-trips unchanged. `get()` keeps its
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

(filled in during step 5 — one sabotage at a time, red-count recorded)

## 8. Residual gaps observed (out of scope for this slice)

- Parquet files **without** embedded field ids: the `RowFilter` resolves leaf predicates
  by the position fallback map, but non-leaf field ids have no leaf to map to, so a
  container-column filter keeps every row at the RowFilter (always-`true`); the scan's
  normal path applies no post-decode residual, so such a file answers a container null
  test incorrectly. Pre-existing class of gap (same shape as nested-leaf predicates on
  id-less files); fixing it needs name-mapping-aware reference resolution in the
  RowFilter.
- `ExpressionEvaluatorVisitor` (partition evaluation) is the only `is_present` consumer
  today; partition specs cannot legally contain non-primitive source fields, so the
  container arm is defensive consistency, not a reachable production path.
