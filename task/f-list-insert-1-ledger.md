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
**Commits:** `3f090a8eb` red cells · `1db44546e` fix · `7fa2217f3` round-1 ledger ·
`6b11d0dd9` round-2 red cells · `63a907d82` round-2 fix · `74d6b23ec` R-01..R-04 remediation.

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

---

## Round 2 — INSERT VALUES

**Scope:** the orchestrator's end-to-end RePark measurement showed `writeTo().append()`,
`write.insertInto`, `saveAsTable(append)` and `INSERT INTO t SELECT …` all write after round 1, but
`INSERT INTO t VALUES (1, array(1,2))` still failed. Two distinct failure modes were measured:

```
(1) Arrow error: Invalid argument error: column types must match schema types,
    expected List(Int32, field: 'element', metadata: {"PARQUET:field_id": "3"})
    but found List(Int32, field: 'element') at column index 1
(2) It is not possible to concatenate arrays of different data types
    (List(Int32, field: 'element'), List(Int32, field: 'element',
    metadata: {"PARQUET:field_id": …}))
```

### 1. The site and the cause

The failure is **not** in the Iceberg writer — `apply_write_defaults` never runs. Both modes happen
inside DataFusion's SQL insert planning/execution, driven by `TableProvider::schema()`:

- `insert_to_plan` (`datafusion-54.1.0 sql/planner.rs`) builds the target `DFSchema` from
  `table_source.schema()`, types each `VALUES` expression against it, and inserts
  `cast_to(field.data_type())` per column. With the fully stamped schema that means
  `PARQUET:field_id`-carrying nested types enter the plan.
- `MemorySourceConfig::try_new_as_values` (`datafusion-datasource-54.1.0 memory.rs:294`) evaluates the
  scalar literals into arrays and rebuilds them under the `Values` node's schema —
  `RecordBatch::try_new_with_options(schema, columns)`. A list literal evaluates to
  `List(Int32, field: 'element')` with no metadata; the schema wants the stamped type → failure (1).
- A `NULL` literal is typed *with* the stamped type while `array(6)` is typed without it, so
  concatenating the two rows fails earlier inside DataFusion array concat → failure (2).
- A nested-struct literal additionally hits DataFusion's nested-cast validator at plan time
  (`Unsupported CAST …`), because nullable literal children can't cast to non-nullable stamped
  children — a separate DataFusion limitation, not this lane's defect (the fixture uses an optional
  struct child for that reason).

### 2. The fix

`IcebergTableProvider::schema()` (`crates/integrations/datafusion/src/table/mod.rs`) now returns
`strip_metadata_from_schema(&self.schema)` — the same Arrow projection minus every metadata key at
every nesting level — while the provider's internal `self.schema` stays stamped.

**Why strip at the provider boundary instead of relabelling at a site:** every failure site is inside
DataFusion itself (`insert_to_plan`, `Values` execution); there is no Iceberg-owned `try_new` on the
SQL `VALUES` path to fix. The stamped schema is a *write-side* detail — `insert_into` reloads the
Iceberg schema itself and the write path (`TaskWriter` → `DataFileWriter` → `apply_write_defaults`)
keeps the stamped schema, so top-level and nested field ids still reach the Parquet footer (proven by
the leaf-id assertions below). Scan correctness is unaffected: `matches_arrow_schema` compares field
names only, and nested sort keys resolve by name path (`NestedFieldExpr`), not metadata.

### 3. Test cells

Both live in `crates/integrations/datafusion/tests/evo_schema_dml.rs`, driven through a real
`SessionContext` with `IcebergCatalogProvider` + `IcebergTableProvider` registered:

| Cell | Test | Pin |
|---|---|---|
| list column via `INSERT … VALUES` (single row, two-row, `NULL` row, null element) | `insert_values_into_a_list_column_writes_and_stamps_the_element_id` | SELECT reads back `[1, 2]`, `[3]`, `NULL`, `[5, , 7]`; parquet footer leaf ids = `[1, 5, 7, 8, 9, 11]` |
| `list<struct<a,b>>` + `map<string,list<int>>` via `named_struct`/`map` in VALUES | `insert_values_into_list_struct_and_map_columns_writes` | SELECT reads back struct/map rows; same footer pin |

### 4. Red output (pre-fix, commit `6b11d0dd9`)

```
insert_values_into_a_list_column_writes_and_stamps_the_element_id ... FAILED
  column types must match schema types, expected List(Int32, field: 'element',
  metadata: {"PARQUET:field_id": "5"}) but found List(Int32, field: 'element')
insert_values_into_list_struct_and_map_columns_writes ... FAILED
  Unsupported CAST from List(Struct("a": Int32, "b": Utf8)) to List(Struct(
  "a": non-null Int32, metadata: {"PARQUET:field_id": "7"}, …))   [plan-time]
```

### 5. Green output (post-fix, commit `63a907d82`)

```
cargo test -p iceberg-datafusion --test evo_schema_dml — 14 passed, 0 failed
cargo test -p iceberg-datafusion --lib                    — 228 passed, 0 failed, 1 ignored
cargo test -p iceberg --lib writer                        — 168 passed, 0 failed, 1 ignored
```

### 6. Mutation proof

Reverted only the round-2 fix (`schema()` returns `self.schema.clone()` — the stamped schema):

```
insert_values_into_a_list_column_writes_and_stamps_the_element_id ... FAILED
  column types must match schema types, expected List(Int32, field: 'element',
  metadata: {"PARQUET:field_id": "5"}) but found List(Int32, field: 'element')
insert_values_into_list_struct_and_map_columns_writes ... FAILED
  It is not possible to concatenate arrays of different data types (List(Struct(
  "a": Int32, metadata: {"PARQUET:field_id": "7"}, …), field: 'element'),
  List(…, field: 'element', metadata: {"PARQUET:field_id": "6"}))
```

Both measured failure modes reproduce; the other 12 tests stay green. Restored; 14/14 green again.

### 7. Audit — `RecordBatch::try_new(<target schema>, …)` sites

| Site | Path | Verdict |
|---|---|---|
| `write_defaults.rs::apply_write_defaults` | write | the round-1 fix site; now takes the cached `target_schema` |
| `integrations/datafusion physical_plan/promotion.rs::widened_batch` | **read** — `delete.rs::table_column_batch`, the merge-on-read delete predicate batch | same defect class but a read-side site owned by F-PROMOTE-READ-1; not a write path — no change |
| `arrow/record_batch_projector.rs::project_batch` | **write** — `EqualityDeleteFileWriter::write` | audited, no fix: `EqualityDeleteWriterConfig::new` rejects `is_nested()` fields in the id fetch func, so equality-delete projection only ever emits primitive leaf columns — the nested-mismatch class cannot fire here |
| `physical_plan/project.rs::project_with_partition` | write (partitioned insert) | already strips metadata via `strip_metadata_from_schema` — unaffected |
| `MemorySourceConfig::try_new_as_values` (datafusion-datasource) | SQL `VALUES` exec | the round-2 site — fixed at the provider schema boundary |
| `parquet_writer.rs::normalize_utc_alias_timestamps` | write | rebuilds under a schema derived from the batch's OWN fields — unaffected |

---

## Round 2 remediation — Grok review R-01..R-04 (commit `74d6b23ec`)

The full test module moved out of `write_defaults.rs` into
`writer/write_defaults_tests.rs` (`#[cfg(test)] pub(crate) mod` in `writer/mod.rs`); the source file
was at the 1000-line ceiling with no headroom for the remediation code.

### R-01 — encoding/alias compatibility restored

Before round 1 the borrowed path compared only top-level field ids, so any column whose parquet leaf
type matched the writer schema wrote fine; the parquet writer's own checks decided acceptability.
Round 1's strict equality + `compatible_layout` turned several of those cases into `DataInvalid`:

| Input → target | Before round 1 | Round 1 | After R-01 |
|---|---|---|---|
| `Timestamp(u, "+00:00")` → `Timestamp(u, "UTC")` (either alias direction) | borrowed → `normalize_utc_alias_timestamps` relabels in `ParquetWriter` | `DataInvalid` | `compatible_layout` accepts same-unit UTC-alias pairs (`is_utc_time_zone`, the same helper `ParquetWriter` uses); the relabel rebuild stamps the target spelling |
| `Utf8View`/`LargeUtf8`/`Utf8` cross-pairs | borrowed → parquet leaf-compat writes | `DataInvalid` | `arrow_cast::cast` to the target type, then writes |
| `BinaryView`/`LargeBinary`/`Binary` cross-pairs | borrowed → writes | `DataInvalid` | cast, then writes |
| `Dictionary(_, v)` → `v` | borrowed → writes (parquet unwraps dicts to the value leaf) | `DataInvalid` | cast unpacks to `v`, then writes |
| `Timestamp(u, None)` → `Timestamp(u, "UTC")` | borrowed → fails at `ArrowWriter` schema check | `DataInvalid` | `DataInvalid` — unchanged refusal, earlier and typed |
| `List(Int64)` → `List(Int32)` | borrowed → fails at `ArrowWriter` schema check | `DataInvalid` | `DataInvalid` — unchanged refusal |

Cells: `utc_alias_timestamptz_is_relabelled_to_canonical_zone` (`"+00:00"` batch → `UTC` column,
values bit-identical) and `view_and_dictionary_encodings_write_as_plain_leaves` (`Utf8View`→`Utf8`,
`BinaryView`→`LargeBinary` — the fork maps `binary` to `LargeBinary`, `Dictionary(Int8,Utf8)`→`Utf8`).

### R-02 — cached target schema

`apply_write_defaults` takes `target_schema: &ArrowSchemaRef`; `DataFileWriter` computes it once in
`build()` (`schema_to_arrow_schema(iceberg_schema)` → `Option<ArrowSchemaRef>`), the same shape as the
existing `writer_arrow_schema` cache. The free-function signature keeps the Iceberg `Schema` too —
`fill_missing_column` still needs `NestedField`s. Test call sites keep `apply_write_defaults(&schema,
&batch)` via a same-named wrapper in `write_defaults_tests.rs`.

### R-03 — borrowed-path equality relaxed to what matters

`batch_matches_schema_order` now uses `borrow_field_eq`/`borrow_type_eq`: field name, nullability,
`PARQUET:field_id` at every level, and recursive data-type structure (Struct children, List/LargeList/
FixedSizeList elements — with width, Map entries — with sorted flag, Dictionary value types).
Iceberg `doc` strings and any other metadata keys no longer push a semantically identical batch onto
the owned rebuild path.

### R-04 — unchecked rebuild under the layout proof

`relabel_column` rebuilds with `ArrayDataBuilder::build_unchecked()` instead of `build()`, removing
`validate_offsets_full`'s O(rows) scan on List/Map data.

**Unsafe justification** (recorded here per the repo rule — no code comments): `data` is `to_data()`
of a valid `ArrayRef`, so its buffers, offsets and null buffers already satisfy their layout. The
only mutation is `data_type(target)` under `compatible_layout(actual, target)`, which proves the
target has the *identical physical layout* — same container variant, same struct arity, same
`FixedSizeList` width, same `Map` sorted flag, same `Timestamp` unit — so the swap changes nested
`Field` names/nullability/metadata (and the timestamp zone string) only, never buffer count, order
or interpretation. Children are recursively rebuilt `ArrayData` of proven-valid layout. What
`build()` would additionally check is child nullability against the nested `Field`s — that is
reinstated explicitly by `disallowed_nulls`, the same per-container rule arrow's validator uses
(strict `null_count` for List/Map entries, parent-mask containment — expanded for `FixedSizeList` —
for Struct/FixedSizeList children), which is O(children), not O(rows). The e2 regression cell
(required element fed a null → `DataInvalid`) still passes.

#### Remaining write-path `try_new` re-check

After the remediation the only `RecordBatch::try_new(target_schema, …)` in the write funnel is the
`apply_write_defaults` rebuild itself; the sites table in Round 2 §7 still stands.

## Round 2 file-size remediation

`table/mod.rs` grew past its recorded legacy ceiling (2220 → 2222) under the round-2 `schema()`
fix, and `data_file_writer.rs` crossed 1000 under R-02/R-04. Resolutions, no ceiling raised:

- `base_writer/data_file_writer.rs`: the 720-line test module extracted to
  `base_writer/data_file_writer_tests.rs` (`#[cfg(test)] pub(crate) mod` in `base_writer/mod.rs`),
  production file now 288 lines; the legacy ceiling row was already removed.
- `table/mod.rs`: the ~1900-line `mod tests` extracted into `table/tests.rs` (provider
  construction, static provider, partitioning/sort/limit cells — 943 lines) and
  `table/schema_evo_tests.rs` (schema-evolution cells — 999 lines). Shared fixtures are
  `pub(super)` in `tests.rs`; `schema_evo_tests.rs` reaches them via `use super::tests::*`.
  `mod.rs` is now 307 lines; its obsolete `2220` legacy-ceiling row was deleted from
  `scripts/check_rust_file_size.py`.
- Verification after the split: `cargo test -p iceberg-datafusion --lib` 228/228 + 1 ignored,
  `cargo test -p iceberg-datafusion --test evo_schema_dml` 14/14, fmt/clippy clean, file-size gate
  green across 506 files.
