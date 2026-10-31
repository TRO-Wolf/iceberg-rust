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


# F-STRUCT-EMPTY-CHILD-PANIC-1 — a struct whose every child was dropped reads back without a panic and with the specification's per-row answer

**Date:** 2026-10-31. **Base:** fork `main` `935cc55c`. **Branch:**
`fix/struct-empty-child-panic-1`. Commits: `13ee562b` (item 1, fix),
`d170d74d` (item 2, pins), `a47b9e01` (item 3, records); fix round 1:
`94e9c43f` (V2 fix and pins) and the fix-1 records commit `90114dda`;
fix round 2 is records only.

**Model:** builder Claude Opus 5.5 (`claude-opus-5-5`).

**Outcome in one line.** A scan no longer panics inside the library after
every child of a struct is dropped. The struct reads back with the
specification's per-row answer: a non-null empty struct where the data file
row was non-null, and NULL where it was NULL. Spark 4.1.2 answers `(1, Row())`
for the AUD-008 row; on a file holding a NULL row of the struct Spark
collapses every row to NULL, a divergence the owner ruled on 2026-10-31
(S-13).

## The fault at `935cc55c`

Independent re-audit 2026-10-29, finding AUD-008 (High), reproduced through
RePark against the fork at `935cc55c`; pre-existing on the base. Ledger
F-EQDEL-DROPPED-KEY-1 row F2-6 recorded the same fault.

A v2 table `(id INT, s STRUCT<k: INT>)` with one row `(1, {k: 10})`. After
`s.k` is dropped, a scan of every column panics:

```
called `Result::unwrap()` on an `Err` value: InvalidArgumentError("use StructArray::try_new_with_length or
StructArray::new_empty_fields to create a struct array with no fields so that the length can be set correctly")
  at crates/iceberg/src/arrow/value.rs:1056  create_primitive_array_repeated   (StructArray::new with zero fields)
  at crates/iceberg/src/arrow/nested_projection.rs:710  create_constant_column
  at crates/iceberg/src/arrow/record_batch_transformer.rs:874, :960  transform_columns
  at crates/iceberg/src/arrow/reader.rs:861  process_parquet_file_scan_task
```

The call chain: the Parquet projection is leaf-based
(`ArrowReader::include_leaf_field_id` collects primitive ids only), so a
struct with no surviving child contributes no leaf. arrow-rs drops the whole
group from the batch. The transformer finds no source column for the struct's
field id and falls to `ColumnSource::Add` with no default, which asks
`create_primitive_array_repeated` for a NULL struct with zero fields. Had the
arm not panicked it would have answered a NULL struct: a silent wrong value.

The same leaf gap gave wrong answers without a panic at the base:

| Shape | Base answer | Spark 4.1.2 answer (measured 2026-10-31, S-13) | Fork answer after this unit |
|---|---|---|---|
| struct nested in a struct, inner children all dropped (pin (f): the file holds NULL inner and NULL outer rows) | inner struct NULL on every row (`PlanNode::Fill` null) | inner struct NULL on every row; after `o.a` is dropped too, the outer struct NULL on every row | per row from the file: inner `[t, f, f]`, outer `[t, t, f]` |
| `LIST<STRUCT<k>>`, `k` dropped (pin (g): the file holds a NULL list row) | the whole list NULL on every row | the whole list NULL on every row | per row from the file: list `[t, t, f]` with empty-struct elements |
| `MAP<INT, STRUCT<v>>`, `v` dropped (pin (g): the file holds a NULL map row) | `DataInvalid => Map field must have exactly 2 fields` (arrow-rs `visit_map` needs key and value projected; the pruned map fails `arrow_schema_to_schema`) | fails: `IllegalArgumentException ... [m, key_value, key] ... is not in the store` | per row from the file: map `[t, f]` with an empty-struct value |
| only the emptied struct projected, a file with no NULL row | no leaf at all, `ProjectionMask::all()`, then `StructArray::try_new` with zero fields fails `DataInvalid` | non-null empty struct | non-null empty struct |

On the mixed-null rows the base equalled Spark only because both answered NULL
on every row; the fork now keeps the non-null rows non-null (owner ruling
2026-10-31, S-13).

The Iceberg specification, Column Projection: "Values for field ids which are
not present in a data file must be resolved according the following rules:
... 4. Return null in all other cases." The struct's field id IS present in
the data file (as a group), so rule 4 does not reach it: its value comes from
the file, with the file's per-row validity. Dropping a field changes the
schema only; existing data files keep the parent struct's definition levels.
Rule 4 still covers a struct added after the file was written (its id is
absent from the file): NULL on every row, in both engines.

## The fix (ruling 2)

The struct's validity is recovered from the file. No scan-planner and no
`FileScanTask` change.

- `row_filter_plan::validity_leaves` walks the task schema from each projected
  id, post-order, through structs, lists and maps, bounded by
  `MAX_NESTED_PROJECTION_DEPTH`. A struct none of whose task-schema leaves is
  present inside the file's group for its id is a leafless struct: every child
  dropped, or (fix round 1, V2) every task child added after the file was
  written. The file is indexed only when the projection holds a struct. For each one whose field id is a struct group in
  the file's Arrow schema (`index_file_groups`, which mirrors the leaf order of
  arrow's `Fields::filter_leaves`, the order `reader.rs` already relies on),
  it borrows the group's first file leaf, unless an earlier borrow already
  covers the group. Post-order makes an inner struct borrow first, so an outer
  leafless struct needs no second leaf.
- `ArrowReader::get_arrow_projection_mask_with_field_ids` starts its leaf
  index list from those borrowed leaves, and keeps them in the projected
  Arrow schema it converts for the type-promotion check (so a map with an
  emptied value struct keeps both children). The second filter pass still
  rejects them, because their ids are not in the task schema; they enter the
  mask only through the borrowed list. Both mask call sites (the initial
  projection and the residual re-projection) go through this function. A
  projection of only emptied structs now reads one leaf per struct, not the
  whole file.
- `nested_projection::PlanNode::build_struct` plans a target struct with no
  fields as an empty `Struct` node, also for an id-less source (which used to
  fall to a struct cast that cannot drop children). Every rebuilt struct uses
  `StructArray::try_new_with_length` with the source length and the source
  null buffer, so the empty struct carries the file's validity. The output
  schema is the target schema: the borrowed leaf never shows.
- `null_propagation::null_struct` builds an all-NULL struct of explicit length
  through `try_new_with_length` and maps the error. `value.rs` uses it in both
  NULL-struct arms, so a struct added after the file and later emptied answers
  NULL per row instead of panicking.
- File-size ceilings: `reader.rs` 9547 to 9546 and `value.rs` 1118 to 1111.
  `record_batch_transformer.rs` is unchanged.

## Sweep of `StructArray::new(` under `crates/iceberg/src/arrow`

| Site at base | Can it see an empty field list | Verdict |
|---|---|---|
| `value.rs:888` `create_primitive_array_single_element` NULL struct | yes: an empty struct type with `None` | same panic shape, fixed through `null_struct`; pinned by `test_create_primitive_array_single_element_empty_struct_null` |
| `value.rs:1056` `create_primitive_array_repeated` NULL struct | yes: the AUD-008 site | fixed through `null_struct`; pinned by `test_create_primitive_array_repeated_empty_struct_null` and `struct_added_after_the_file_reads_null_per_row` |
| `nan_val_cnt_visitor.rs:175` batch viewed as a struct | yes: a zero-column batch | fixed with `try_new_with_length(..., batch.num_rows())`, error mapped |

No production `StructArray::new(` remains in the directory; the rest are in
test files. The `StructArray::try_new` sites (`avro_reader.rs`,
`record_batch_projector.rs`, `null_propagation.rs`,
`partition_value_calculator.rs`, `caching_delete_file_loader.rs`) return an
`Err` on an empty field list, never a panic.

## Pins

All in `crates/iceberg/tests/struct_empty_child.rs` (710 lines) except the
unit pins. Each integration pin builds a v2 table through the memory catalog,
the Parquet data-file writer, a fast-append transaction, `update_schema`
transactions and the Arrow scan.

- `one_child_struct_dropped_reads_non_null_empty_struct` (a): drop `s.k`; a
  scan of every column answers one row, `id = 1`, `s` typed `Struct([])` and
  non-null; a scan of `id` alone answers `[1]`; a scan of `s` alone answers
  one non-null empty struct.
- `two_child_struct_dropped_one_at_a_time_reads_non_null_empty_struct` (b):
  drop `s.k`, then `s.j`; same answers.
- `null_struct_row_stays_null_beside_non_null_empty_struct` (c): rows
  `(1, {k: 10})` and `(2, NULL)`; after the drop the validity is
  `[true, false]`. Spark answers per file here (S-13).
- `file_written_between_drops_scans_with_the_old_file` (d): file 1 written
  with `s<k, j>`, drop `s.k`, file 2 written with `s<j>` (one non-null, one
  NULL row), drop `s.j`; both files scan together, validity
  `[true, true, false]`. Spark answers per file here (S-13).
- `writing_a_struct_with_no_children_refuses_without_panic` (d, second half):
  a file written AFTER every child is dropped cannot exist in Parquet
  (`Parquet does not support writing empty structs`); the production writer
  refuses with an `Error`. Green on the base and after the fix.
- `struct_added_after_the_file_reads_null_per_row` (e): a struct added after
  the file reads NULL; after its only child is dropped it still reads NULL
  (the `value.rs` arm), and the untouched `s` stays non-null.
- `struct_nested_in_struct_with_children_dropped_keeps_file_validity` (f):
  rows `{a: 1, i: {x: 5}}`, `{a: 2, i: NULL}`, `NULL`; drop `o.i.x`: `o`
  validity `[t, t, f]`, `o.a` values kept, `o.i` typed `Struct([])` with
  validity `[t, f, f]`; then drop `o.a`: `o` has one child and the same
  validities. Spark answers per file here (S-13).
- `list_of_struct_with_children_dropped_keeps_elements` (g, ruling 4):
  `[{k:1}, {k:2}]`, `[]`, `NULL`; drop `l.element.k`: list validity
  `[t, t, f]`, lengths 2 and 0, two non-null empty-struct elements. Spark
  answers per file here (S-13).
- `map_of_struct_with_children_dropped_keeps_entries` (g, ruling 4):
  `{7: {v: 1}}`, `NULL`; drop `m.value.v`: map validity `[t, f]`, one entry
  with a non-null empty-struct value. Spark answers per file here (S-13);
  on this shape Spark fails the read.
- Fix round 1 (V2), `struct s<d>`, add `s.k`, drop `s.d`, so the task struct
  `s<k>` has a leaf the file never held:
  `struct_whose_task_children_are_all_absent_from_the_file_reads_non_null`
  (one row `{d: 1}`: `s` typed `Struct<k>`, non-null, `k` NULL; a scan of `s`
  alone answers the same),
  `struct_whose_task_children_are_all_absent_keeps_a_null_row_null`
  (`{d: 1}` and NULL: validity `[t, f]`, `k` NULL on both),
  `nested_struct_whose_task_children_are_all_absent_keeps_file_validity`
  (`o<s<d>>`, add `o.s.k`, drop `o.s.d`; rows `{s: {d: 1}}`, `{s: NULL}`,
  NULL: `o` `[t, t, f]`, `o.s` typed `Struct<k>` `[t, f, f]`, `k` NULL) and
  `struct_whose_task_children_are_partly_in_the_file_is_unchanged`
  (`s<d, e>`, add `s.k`, drop `s.d`: `e` `[2, NULL]`, `k` NULL, validity
  `[t, f]`; green on `a47b9e01` and after the fix, a no-regression pin).
- Unit pins in `crates/iceberg/src/arrow/value_tail_tests.rs`:
  `test_create_primitive_array_repeated_empty_struct_null` (`Ok`, length 3,
  3 nulls, type `Struct([])`) and
  `test_create_primitive_array_single_element_empty_struct_null` (`Ok`,
  length 1, 1 null).
- Unit pins in `crates/iceberg/src/arrow/nested_projection_evo_pin_tests.rs`:
  `idless_source_struct_to_empty_target_keeps_validity` (a file struct whose
  child carries no field id projects onto an empty target struct with the
  file's validity; at base the plan fell to a struct cast and failed
  `DataInvalid => failed to cast nested column to Struct()`) and
  `inner_struct_absent_from_file_and_emptied_fills_null` (an inner struct
  absent from the file and later emptied fills a typed NULL column through
  `PlanNode::Fill`; green on the base and after the fix, a no-regression pin).

## Red runs (pins against the unmodified `935cc55c` sources)

Integration binary, before any source file was edited:

```
test one_child_struct_dropped_reads_non_null_empty_struct ... FAILED
test two_child_struct_dropped_one_at_a_time_reads_non_null_empty_struct ... FAILED
test null_struct_row_stays_null_beside_non_null_empty_struct ... FAILED
test file_written_between_drops_scans_with_the_old_file ... FAILED
test struct_added_after_the_file_reads_null_per_row ... FAILED
  panicked at arrow-array-58.4.0/src/array/struct_array.rs:91:46   (all five: the library panic)
test struct_nested_in_struct_with_children_dropped_keeps_file_validity ... FAILED
  left: [false, false, false]  right: [true, false, false]          (silent NULL)
test list_of_struct_with_children_dropped_keeps_elements ... FAILED
  left: [false, false, false]  right: [true, true, false]           (silent NULL list)
test map_of_struct_with_children_dropped_keeps_entries ... FAILED
  collect scan batches: DataInvalid => Map field must have exactly 2 fields
test writing_a_struct_with_no_children_refuses_without_panic ... ok
test result: FAILED. 1 passed; 8 failed
```

The (e) pin's last assertion was corrected after the red run (it asserted the
untouched `s` as empty); the red failure was the library panic on the first
scan after `added.x` was dropped, before that assertion.

Unit pins. Run 1, before any source file was edited:

```
test arrow::value::test::test_create_primitive_array_repeated_empty_struct_null ... FAILED
test arrow::value::test::test_create_primitive_array_single_element_empty_struct_null ... FAILED
  panicked at arrow-array-58.4.0/src/array/struct_array.rs:91:46
```

Run 2, after the fix was written: the seven fix-commit source files checked
out at `935cc55c` (`git show HEAD:<path>`), the four unit pins run, then the
fix restored and its diff compared byte-for-byte with a saved copy:

```
test arrow::value::test::test_create_primitive_array_single_element_empty_struct_null ... FAILED
test arrow::value::test::test_create_primitive_array_repeated_empty_struct_null ... FAILED
test arrow::nested_projection::test::inner_struct_absent_from_file_and_emptied_fills_null ... ok
test arrow::nested_projection::test::idless_source_struct_to_empty_target_keeps_validity ... FAILED
  DataInvalid => failed to cast nested column to Struct()
test result: FAILED. 1 passed; 3 failed
```

## Green runs

At `d170d74d`, `cargo test -p iceberg --test struct_empty_child`:

```
test one_child_struct_dropped_reads_non_null_empty_struct ... ok
test two_child_struct_dropped_one_at_a_time_reads_non_null_empty_struct ... ok
test null_struct_row_stays_null_beside_non_null_empty_struct ... ok
test file_written_between_drops_scans_with_the_old_file ... ok
test writing_a_struct_with_no_children_refuses_without_panic ... ok
test struct_added_after_the_file_reads_null_per_row ... ok
test struct_nested_in_struct_with_children_dropped_keeps_file_validity ... ok
test list_of_struct_with_children_dropped_keeps_elements ... ok
test map_of_struct_with_children_dropped_keeps_entries ... ok
test result: ok. 9 passed; 0 failed; 0 ignored
```

The four unit pins pass in the `iceberg` lib run below.

## Clauses

| Id | Claim | Verdict | Pins |
|---|---|---|---|
| S-1 | No library panic on a struct whose every child was dropped, at the top level, nested, in a list or in a map | PROVEN | (a), (b), (c), (d), (e), (f), (g) |
| S-2 | A struct the file holds by id reads its validity from the file: non-null empty where the row was non-null, NULL where NULL (ruling 2, the specification's per-row answer) | PROVEN | `one_child_struct_dropped_reads_non_null_empty_struct`, `two_child_struct_dropped_one_at_a_time_reads_non_null_empty_struct`, `null_struct_row_stays_null_beside_non_null_empty_struct`, `file_written_between_drops_scans_with_the_old_file` |
| S-3 | The output schema does not show the borrowed leaf | PROVEN | every pin asserts the column type `Struct([])`; (f) asserts `o` keeps one child after `o.a` is dropped |
| S-4 | A struct nested in a struct keeps the file's validity at both levels, per row (the specification's per-row answer) | PROVEN | `struct_nested_in_struct_with_children_dropped_keeps_file_validity` |
| S-5 | A struct added after the file was written stays NULL per row, also once emptied; it never refuses | PROVEN | `struct_added_after_the_file_reads_null_per_row` |
| S-6 | Ruling 4: a struct inside a list or a map whose children were all dropped reads the specification's per-row answer | PROVEN | `list_of_struct_with_children_dropped_keeps_elements`, `map_of_struct_with_children_dropped_keeps_entries` |
| S-7 | `create_primitive_array_repeated` and `create_primitive_array_single_element` answer `Ok` for an empty struct type with `None` | PROVEN | `test_create_primitive_array_repeated_empty_struct_null`, `test_create_primitive_array_single_element_empty_struct_null` |
| S-8 | No `StructArray::new` over a possibly empty field list remains in production code under `crates/iceberg/src/arrow` | PROVEN | Sweep above (by inspection; the two `value.rs` sites are pinned) |
| S-9 | Writing a Parquet file for a schema whose struct has no children refuses with an `Error` | PROVEN | `writing_a_struct_with_no_children_refuses_without_panic` (no-regression; the parquet writer refuses, as parquet-mr does) |
| S-10 | Avro and ORC data files with an emptied struct | OPEN | Not in this unit. The Avro reader builds the struct with `StructArray::try_new` (loud `DataInvalid`, not a panic, not a wrong row); the ORC path is unmeasured |
| S-11 | An id-less Parquet file whose name mapping gives the dropped child no id. Split by the verifier's round-2 probe (F3): an emptied id-less struct (every child dropped) is correct end to end at `90114dda` (an id-less `s<d>`, drop `s.d`, full scan answers `Struct([])` with validity `[t, f]`; the base panicked), and an id-less struct whose task children are all absent from the file is wrong, see S-16 | OPEN | The emptied case has no end-to-end pin; its unit pin is `idless_source_struct_to_empty_target_keeps_validity` (the empty-target plan over an unstamped source), and the borrowed leaf is chosen by group id and position. The absent-children case is unpinned and carded |
| S-12 | No neighbouring answer changed | PROVEN | Gates below |
| S-13 | Dated Spark divergence (2026-10-31). Measured by the orchestrator on Spark 4.1.2 with Iceberg 1.11.0, hadoop catalog, vectorization on and off: Spark decides an emptied struct's validity PER DATA FILE. A file with no NULL row of the struct (or of an enclosing struct or list) reads a non-null empty struct on every row; a file with any NULL row of it reads NULL on every row, including rows the file holds as non-null. Measured: `(1, {k: 10}), (2, NULL)` in one file, drop `s.k`, Spark reads `[NULL, NULL]`; two files `[(1, {10})]` and `[(2, {20}), (3, NULL)]` read `[Row(), NULL, NULL]`; `(1, {10}), (2, {k: NULL})` reads `[Row(), Row()]` (a struct with a NULL child is non-null). On the pin (f) shape Spark reads the inner struct NULL on all rows and, after the second drop, the outer struct NULL on all rows; on the pin (g) list shape Spark reads the whole list NULL on every row; on the pin (g) map shape Spark FAILS with `IllegalArgumentException ... [m, key_value, key] ... is not in the store`. The engines agree on: a single non-null row (the AUD-008 case), `{k: NULL}` rows, a required struct, and an added struct across pre-add and post-add files. Owner ruling 2026-10-31 (option 1): the fork keeps the per-row answer from the file's definition levels, the specification's reading; Spark's per-file collapse is a lossy reader shortcut (a non-null row becomes NULL) and the fork does not copy it | PROVEN | the per-row assertions of (c), (d), (f), (g) |
| S-14 | Schema-change divergence: the fork's `update_schema` accepts `drop m.key.a` on `map<struct<a>, int>`; Spark refuses the ALTER with `Unsupported table change: Cannot alter map keys`. The head then reads the map correctly (empty-struct keys, values `[7, 8]`, validity `[t, f]`); the base refused the scan. Loud or correct is met on the read; the schema change itself is not fixed here | OPEN | carded on the RePark side by the RP-63 records round |
| S-15 | Fix round 1 (V2), on data files that carry embedded field ids: a struct whose task-schema children are all absent from the file (add `s.k`, drop `s.d`) reads its validity from the file per row, children filled per the existing rules; a struct whose task children are partly in the file is unchanged. The id-less case is S-16 | PROVEN | `struct_whose_task_children_are_all_absent_from_the_file_reads_non_null`, `struct_whose_task_children_are_all_absent_keeps_a_null_row_null`, `nested_struct_whose_task_children_are_all_absent_keeps_file_validity`, `struct_whose_task_children_are_partly_in_the_file_is_unchanged` |
| S-16 | **F1 (S2, pre-existing at `935cc55c`, `a47b9e01` and `90114dda` alike; outside this unit's charter, reproduced by the orchestrator).** On a Parquet data file WITHOUT embedded field ids (the fallback-id path, with or without a name mapping, because `apply_name_mapping_to_arrow_schema` and `add_fallback_field_ids_to_arrow_schema` stamp top-level fields only), the V2 shape is wrong: a two-leaf file `s<d, e>` with rows `[{d: 1, e: 2}, NULL]`, add `s.k`, drop `s.d` and `s.e`, full scan reads `s` validity `[t, f]` and `k = [Some(1), None]`, so the dropped `d`'s value reads under `k`; a one-leaf file `s<d>`, add `s.k`, drop `s.d`, full scan returns a `RecordBatch` whose schema field is `s: Struct<k>` but whose column is `Struct<d>` with values `[1, null]`, and `concat_batches` then fails with `column types must match schema types ... expected Struct(k) but found Struct(d)`. Cause: `nested_projection::build_struct` sees a source struct with no child ids and takes `PlanNode::Cast`, and the whole top-level group is read through `unmapped_group_leaf_indices`. Disposition (orchestrator, 2026-10-31): CARDED on the RePark side as F-IDLESS-STRUCT-CHILD-REMAP-1 by the RP-63 records round, not folded; S-15 and the map rows narrow to files with embedded field ids. The fix lives in `nested_projection::build_struct`'s id-less Cast path and in the top-level-only stamping of `apply_name_mapping_to_arrow_schema` and `add_fallback_field_ids_to_arrow_schema`. | OPEN | None: the two id-less probe shapes are unpinned; the RePark card is F-IDLESS-STRUCT-CHILD-REMAP-1 |

## Gates

Measured at `d170d74d`, every command exit 0.

- `cargo test -p iceberg --test struct_empty_child`: 9 passed, 0 failed.
- `cargo test -p iceberg`: lib 4646 passed, 0 failed, 9 ignored (the
  F-EQDEL-DROPPED-KEY-1 fold-2 figure 4642 plus the 4 new unit pins);
  71 result lines in all, 5014 passed, 0 failed, 21 ignored (fold-2 5001 plus
  the 4 unit pins and the 9 integration pins).
- `cargo test -p iceberg-datafusion`: lib 463 passed, 0 failed, 5 ignored;
  43 result lines in all, 802 passed, 0 failed, 15 ignored (unchanged).
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings`.
- `cargo fmt --all -- --check`.
- `make check-rust-file-size`: 705 files clean, 89 legacy ceilings.
- `make check-comment-blocks` and `make check-agent-artifacts`: OK.
- Comment gate: 0 added comment lines in the fix commit; in the pins commit
  only the 16-line Apache licence header of the new test file.

## Verifier round 1 (Opus 5.5 on `a47b9e01`): FAIL

- **V1 (S1, a ruling).** Spark decides an emptied struct's validity per data
  file, the fork per row (the measurements are in S-13). Owner ruling
  2026-10-31, option 1: the fork keeps the specification's per-row answer and
  the divergence is recorded, dated, as S-13. The code of the fix did not
  change for V1; pins (c), (d), (f) and (g) keep their per-row assertions.
- **V2 (S2, pre-existing, folded into this unit by the ruling).** A struct
  whose task-schema children are all absent from the file (`s<d>`, one file
  `[(1, {d: 1})]`, add `s.k`, drop `s.d`) read NULL with type `Struct<k>`, the
  same at the base: `validity_leaves` borrowed a leaf only for a struct with no
  task-schema leaf. Fixed in this round (S-15).
- **V3 (S3, outside this diff).** `update_schema` accepts `drop m.key.a` on
  `map<struct<a>, int>`, which Spark refuses; the read is correct. Recorded
  OPEN as S-14.
- **V4 (S3).** The `crates/iceberg/src/arrow/map.md` row for
  `nan_val_cnt_visitor.rs` claimed a zero-column batch is an `Error`; it views
  as a zero-field struct of the batch length and only another construction
  failure is mapped. Row reworded in this round.

## Fix round 1

This round changed: `row_filter_plan::validity_leaves` judges a struct
leafless against the file (none of its task-schema leaves present inside the
file's group for its id) and indexes the file only when the projection holds a
struct (`94e9c43f`, with four pins); the outcome line, the title, S-2, S-4,
S-6 and the base-vs-Spark table re-worded for the ruling; S-13, S-14 and S-15
added; the `map.md` rows for `nan_val_cnt_visitor.rs`, `row_filter_plan.rs`
and `struct_empty_child.rs` updated.

Red run, the four new pins against the `a47b9e01` sources (test file only
changed), `cargo test -p iceberg --test struct_empty_child`:

```
test struct_whose_task_children_are_all_absent_from_the_file_reads_non_null ... FAILED
  left: [false]  right: [true]
test struct_whose_task_children_are_all_absent_keeps_a_null_row_null ... FAILED
  left: [false, false]  right: [true, false]
test nested_struct_whose_task_children_are_all_absent_keeps_file_validity ... FAILED
  left: [false, false, false]  right: [true, true, false]
test struct_whose_task_children_are_partly_in_the_file_is_unchanged ... ok
test result: FAILED. 10 passed; 3 failed; 0 ignored
```

Green run at `94e9c43f`: `test result: ok. 13 passed; 0 failed; 0 ignored`.

Gates at `94e9c43f`, every command exit 0:

- `cargo test -p iceberg --test struct_empty_child`: 13 passed, 0 failed.
- `cargo test -p iceberg`: lib 4646 passed, 0 failed, 9 ignored; 71 result
  lines in all, 5018 passed, 0 failed, 21 ignored (round 0 5014 plus the 4
  new pins).
- `cargo test -p iceberg-datafusion`: lib 463 passed, 0 failed, 5 ignored;
  43 result lines in all, 802 passed, 0 failed, 15 ignored (unchanged).
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings`.
- `cargo fmt --all -- --check`.
- `make check-rust-file-size`: 705 files clean, 89 legacy ceilings
  (`reader.rs` and `row_filter_plan.rs` did not need a ceiling;
  `row_filter_plan.rs` is 508 lines, the test file 710).
- `make check-comment-blocks` and `make check-agent-artifacts`: OK.
- Comment gate: 0 added comment lines in both fix-round commits.

## Verifier round 2 (Opus 5.5 on `90114dda`): FAIL

- **F1 (S2, pre-existing at `935cc55c`, `a47b9e01` and `90114dda` alike; outside this unit's charter, reproduced by the orchestrator).** On a Parquet data file WITHOUT embedded field ids (the fallback-id path, with or without a name mapping, because `apply_name_mapping_to_arrow_schema` and `add_fallback_field_ids_to_arrow_schema` stamp top-level fields only), the V2 shape is wrong: a two-leaf file `s<d, e>` with rows `[{d: 1, e: 2}, NULL]`, add `s.k`, drop `s.d` and `s.e`, full scan reads `s` validity `[t, f]` and `k = [Some(1), None]`, so the dropped `d`'s value reads under `k`; a one-leaf file `s<d>`, add `s.k`, drop `s.d`, full scan returns a `RecordBatch` whose schema field is `s: Struct<k>` but whose column is `Struct<d>` with values `[1, null]`, and `concat_batches` then fails with `column types must match schema types ... expected Struct(k) but found Struct(d)`. Cause: `nested_projection::build_struct` sees a source struct with no child ids and takes `PlanNode::Cast`, and the whole top-level group is read through `unmapped_group_leaf_indices`. Disposition (orchestrator, 2026-10-31): CARDED on the RePark side as F-IDLESS-STRUCT-CHILD-REMAP-1 by the RP-63 records round, not folded; S-15 and the map rows narrow to files with embedded field ids.
- **F2 (S3).** The ledger header names fix-round commit `162e80ac`, which does not exist: the V2 fix commit was amended to `94e9c43f` before it was pushed (a clippy `needless_lifetimes` fix). The Fix round 1 section names `94e9c43f` correctly.
- **F3 (S3).** S-11 says no end-to-end pin reads an id-less file. The verifier's probe did: an id-less `s<d>`, drop `s.d`, full scan answers `Struct([])` with validity `[t, f]` at `90114dda` and panics at `935cc55c`. So the EMPTIED-struct id-less case is correct end to end and only lacks a pin, while the V2 id-less case is wrong (F1).

## Fix round 2 (records)

This round changed the records only: the ledger header now names the V2 fix commit `94e9c43f` and the fix-1 records commit `90114dda` (F2); S-11 is split into its emptied case (correct, unpinned end to end) and its absent-children case (wrong, S-16) and stays OPEN (F3); S-15 is narrowed to data files with embedded field ids and stays PROVEN with its pins; S-16 (OPEN) records F1 with its cause and disposition, carded as F-IDLESS-STRUCT-CHILD-REMAP-1 by the RP-63 records round; the `row_filter_plan.rs` row of `crates/iceberg/src/arrow/map.md` and the `struct_empty_child.rs` row of `crates/iceberg/tests/map.md` narrow their V2 sentence to files with embedded field ids. No source, test, manifest or workflow file changed, so no cargo gate ran in this round; the round-1 gates at `94e9c43f` stand.

The round-2 measured facts: both id-less V2 probes read wrong (F1), the id-less emptied `s<d>` read answers `Struct([])` with validity `[t, f]` (F3), and 12 verifier probes are green at `90114dda` (10 red at `a47b9e01`) with every other claim held.

## Follow-ups (out of scope)

1. Avro and ORC data files (S-10): pin both readers on an emptied struct.
   The Avro fix is the same `try_new_with_length` change in
   `avro_reader.rs` (at its legacy ceiling).
2. S-11: an end-to-end pin over an id-less Parquet file with a name mapping
   that omits the dropped child.
