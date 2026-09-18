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

# F-LIST-INSERT-1 — nested Arrow field relabel on writes (ledger)

**Branch:** `fix/list-insert-1` (worktree of the fork clone).
**Scope:** writes into Iceberg `ARRAY` columns failed when the engine's batch had correct top-level
`PARQUET:field_id` metadata but nested fields (list element, map key/value, struct children inside a
list) carried no ids or a different name (`item` vs `element`).
**Commits:** `3f090a8eb` red cells · `1db44546e` fix.

## 1. The defect

`apply_write_defaults` (`crates/iceberg/src/writer/write_defaults.rs`) built the output batch as
`RecordBatch::try_new(target_schema, columns)` with the input columns pushed through unchanged. The
target Arrow schema from `schema_to_arrow_schema` stamps the Iceberg field id on every nested field,
so an engine batch whose top-level fields carry `PARQUET:field_id` but whose nested fields carry none
(or a different name) failed `try_new` validation:

```
column types must match schema types, expected List(Int32, field: 'element',
metadata: {"PARQUET:field_id": "3"}) but found List(Int32)
```

Second half: `batch_matches_schema_order` compared only top-level field ids, so a batch with unstamped
nested fields took the zero-cost borrowed path and was written as-is. The parquet footer itself stays
correct — it derives from `writer_arrow_schema`, not the batch schema (verified by probe below) — but
the borrowed path skips the compatibility check entirely, so a batch whose nested PHYSICAL type is
incompatible could reach the writer with no validation.

## 2. The rule implemented

Every column the writer writes is relabelled to the Iceberg target Arrow type: same physical data,
the target's nested field names, nullability flags and `PARQUET:field_id` metadata at every level
(list / large list / fixed-size list element, map entries/key/value, struct children, any depth). The
relabel is a metadata-only `ArrayData` rebuild (`into_builder().data_type(target).child_data(…)`),
no value copy and no Arrow `cast`. A column already equal to the target type is returned untouched;
a whole batch equal to the target schema (complete `Field` equality — names, nullability, metadata,
nested descriptors) keeps the zero-cost `Cow::Borrowed` path. Anything physically incompatible is
refused with `ErrorKind::DataInvalid`, never silently rewritten.

## 3. The site

All in `crates/iceberg/src/writer/write_defaults.rs`:

- `apply_write_defaults` — builds `target_schema` first; the borrowed fast path now requires complete
  field equality; supplied columns go through `relabel_column` before `RecordBatch::try_new`.
- `batch_matches_schema_order` — `batch.schema().fields().iter().eq(target_fields.iter())` — deep
  `Field` equality, so a top-level-id match with unstamped nested fields goes through the relabel.
- `relabel_column` — recursive relabel of an `ArrayRef` to a target `DataType`, bounded by
  `MAX_RELABEL_DEPTH = 128`.
- `nested_fields` — the child `Field`s of Struct/List/LargeList/FixedSizeList/Map data types.
- `compatible_layout` — same-container-shape check (struct arity, list kind, fixed-size width, map
  sortedness) before recursion.
- `incompatible_type` — the `DataInvalid` error constructor.

### Design rulings

- **Struct children are matched by name, not relabelled.** Struct field identity in Iceberg is
  name-keyed; a struct child whose name differs from the target's is a different field, not a label
  difference, so `relabel_column` refuses. List/map wrapper child names (`element`, `item`,
  `entries`, `key`, `value`) are container plumbing and always take the target's name.
- **`ArrayDataBuilder::build()` supplies the required-nullability refusal.** Rebuilding with the
  target type validates child nullability against the actual buffers, so a required element whose
  data contains nulls fails at build time; the Arrow error is wrapped as the source of a
  `DataInvalid`.

## 4. Test cells

Schema fixture (`nested_ids_schema`, built by `serde_json`): `id` 1, `nums` `list<int>` id 2 element
id 3, `pairs` `list<struct<a:int id6, b:string id7>>` id 4 element id 5, `props`
`map<string, list<int ele id11>>` id 8 key id 9 value id 10. Fixture batch `nested_batch(element_name,
top_ids)` builds all four columns with no nested metadata; `top_ids` controls whether top-level ids
are stamped. Cells live in `write_defaults.rs::tests` and
`base_writer/data_file_writer.rs::test`, sharing `pub(crate)` fixtures.

| Cell | Test | Pin |
|---|---|---|
| (a) `list<int>` `element` + `item` | `unstamped_nested_fields_are_relabelled_to_iceberg_types` | `Cow::Owned` + `column(1).data_type() == target` |
| (b) `list<struct<a,b>>` unstamped children | same test, columns 2 | same pin |
| (c) `map<string,list<int>>` unstamped key/value/element | same test, column 3 | same pin |
| (a)–(c) via name fallback, no ids at all | same test, `nested_batch("item", false)` | `Cow::Owned` |
| (d) data-file round trip | `data_file_writer_stamps_nested_field_ids_in_parquet_footer` | writes `nested_batch("item", false)`, asserts footer nested ids 3,5,6,7,9,10,11 via `assert_nested_field_ids` on the reader schema; null list, empty list, null element, null map row survive |
| (e1) `list<int>` fed `List(Int64)` | `incompatible_nested_data_is_data_invalid` | `DataInvalid` |
| (e2) required element fed a null element | same test, `nested_ids_schema_with(true)` | `DataInvalid` |

(d) uses the idless batch deliberately: a top-id batch under the full revert takes the borrowed path
and writes successfully (probe result below), so only a batch that must take the fill path proves the
write actually goes through the relabel.

## 5. Red output (pre-fix, commit `3f090a8eb`)

```
running 3 tests
test writer::write_defaults::tests::unstamped_nested_fields_are_relabelled_to_iceberg_types ... FAILED
test writer::write_defaults::tests::incompatible_nested_data_is_data_invalid ... FAILED
test writer::base_writer::data_file_writer::test::data_file_writer_stamps_nested_field_ids_in_parquet_footer ... FAILED

---- unstamped_nested_fields_are_relabelled_to_iceberg_types stdout ----
assertion failed: matches!(filled, Cow::Owned(_))
(top-level ids matched → borrowed path taken, batch returned unstamped)

---- incompatible_nested_data_is_data_invalid stdout ----
assertion `left == right` failed  left: Unexpected  right: DataInvalid
(Int64 list reached try_new / the required-element batch borrowed through)

---- data_file_writer_stamps_nested_field_ids_in_parquet_footer stdout ----
write: Unexpected => Arrow Schema Error
Source: Invalid argument error: column types must match schema types, expected
List(Int32, field: 'element', metadata: {"PARQUET:field_id": "3"}) but found
List(Int32) at column index 1
```

## 6. Green output (post-fix)

```
test writer::write_defaults::tests::incompatible_nested_data_is_data_invalid ... ok
test writer::write_defaults::tests::unstamped_nested_fields_are_relabelled_to_iceberg_types ... ok
test writer::base_writer::data_file_writer::test::data_file_writer_stamps_nested_field_ids_in_parquet_footer ... ok
test result: ok. 3 passed; 0 failed

cargo test -p iceberg --lib writer
test result: ok. 168 passed; 0 failed; 1 ignored
```

## 7. Mutation proof

**Mutation A — full fix reverted** (top-level-id fast path + raw column push, tests kept):

```
test writer::write_defaults::tests::unstamped_nested_fields_are_relabelled_to_iceberg_types ... FAILED
  assertion failed: matches!(filled, Cow::Owned(_))
test writer::write_defaults::tests::incompatible_nested_data_is_data_invalid ... FAILED
  left: Unexpected  right: DataInvalid
test writer::base_writer::data_file_writer::test::data_file_writer_stamps_nested_field_ids_in_parquet_footer ... FAILED
  write: Unexpected => Arrow Schema Error: column types must match schema types,
  expected List(Int32, field: 'element', metadata: {"PARQUET:field_id": "3"})
  but found List(Int32) at column index 1
```

**Mutation B — only the fast-path tightening reverted** (relabel still active):

```
test writer::write_defaults::tests::unstamped_nested_fields_are_relabelled_to_iceberg_types ... FAILED
  assertion failed: matches!(filled, Cow::Owned(_))
  (top-id batches borrowed instead of relabelled)
test writer::write_defaults::tests::incompatible_nested_data_is_data_invalid ... FAILED
  must refuse: RecordBatch { ... }
  (the required-element batch borrowed through untouched; the Int64 two-column
  batch still fails because its column count forces the fill path)
test ...data_file_writer_stamps_nested_field_ids_in_parquet_footer ... ok
```

(d) is green under mutation B by construction: its idless batch can never satisfy a top-level-id
borrow check, so the borrowed path never applies. A probe switching (d) to `nested_batch("item",
true)` was also green under mutation B — a borrowed top-id batch writes successfully because the
parquet footer derives from `writer_arrow_schema`. The borrowed-path tightening is therefore pinned
by the unit cells' `Cow::Owned` / `expect_err` assertions, which is what the red output above shows.

**Restored:** both mutations reverted to `HEAD`; all 3 cells green again.

## 8. Refusal cases and why each refuses

| Input | Refusal | Why |
|---|---|---|
| `List(Int64)` vs target `List(Int32)` | `DataInvalid` | different primitive leaf type — `compatible_layout` false |
| `list<struct>` element with nulls, element required | `DataInvalid` | `ArrayDataBuilder::build()` nullability validation; Arrow error kept as source |
| struct child name differs from target | `DataInvalid` | struct fields are name-keyed in Iceberg; a different name is a different field |
| `List` vs `LargeList`, different `FixedSizeList` width, different map `sorted` flag | `DataInvalid` | different physical container shape |
| child-data count vs declared children mismatch | `DataInvalid` | malformed array structure |
| recursion past depth 128 | `DataInvalid` | bound on hostile nesting depth |

## 9. Audit — other `RecordBatch::try_new(target_schema, …)` write-path sites

| Site | Verdict |
|---|---|
| `write_defaults.rs:75` `apply_write_defaults` | **the fix site** |
| `file_writer/parquet_writer.rs:844` `maybe_normalize_utc_alias` | unaffected — rebuilds under a schema derived from the batch's OWN fields (only the UTC-alias column type changes); runs after `apply_write_defaults` on already-normalised columns |
| `arrow/record_batch_projector.rs:166` `project_batch` | same defect CLASS, different entry point — equality-delete/reader projection rebuilds under the projected schema with batch columns as-is; an unstamped nested equality-field batch would fail the same way. Deferred — out of scope for this lane (defect measured on ARRAY data writes; equality deletes with nested equality fields are a separate surface) |
| `base_writer/position_delete_writer.rs`, `equality_delete_writer.rs`, `position_delete_writer_spec_stamp.rs`, `transaction/*`, `scan/*` | all remaining hits are `#[cfg(test)]` code or test helpers — no production target-schema `try_new` |

## 10. Notes

- The fork's file-size gate (`scripts/check_rust_file_size.py`, default 1000, no headroom anywhere
  in the tree) forced fixture compression: the Iceberg schema fixture is `serde_json` text, the
  round-trip cell lives in `data_file_writer.rs`, and shared helpers are `pub(crate)` in
  `write_defaults.rs::tests`. Both files sit at 999/997 lines.
- `Fields` equality used by the fast path is deep `Field` equality — name, nullability, metadata,
  nested children — so the borrowed path is taken only for a byte-identical schema.
