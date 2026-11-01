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

# F-IDLESS-STRUCT-CHILD-REMAP-1 — an id-less Parquet file gets field ids at every depth, and a struct child that was dropped or added reads by the table schema

**Date:** 2026-11-01. **Base:** fork `main` `14a93602`. **Branch:**
`fix/idless-struct-child-remap-1`. Commits: `51f39a29` (item 1, fix),
`8a66aa7c` (item 2, pins), `bc6a2e67` (item 3, this ledger's first records
commit), `cc213112` (fix round 1, F4 pruning), `9c4d43a9` (fix round 1, F1 to
F3 and ruling 7), `85687ce0` (fix round 1, pins), and the fix-round records
commit that adds this text.

**Model:** builder Claude Opus 5.5 (`claude-opus-5-5`), round 0 and fix round
1. Records clerk Claude Haiku 5.5 (`claude-haiku-5-5`), round 0 and fix round 1.

**Outcome in one line.** An id-less Parquet file gets field ids at every depth.
A nested child takes the id of the child with its name in the matched parent,
and nothing nested is stamped by a counter or by position. A dropped or added
struct child never reads a neighbour's value, an absent or unmapped child reads
NULL per row, and pruning on an id-less file uses the ids the reader stamped.
At `85687ce0` the 18 pins of `struct_idless_child.rs`, the 29 pins of
`struct_idless_child_2.rs`, the 13 `struct_empty_child` pins and the 10 unit
pins are green. Verifier round 1 (Opus 5.5) FAILED `8a66aa7c` with F1 to F4,
and the Muse MAX critic PASSED it with three findings; all of them are fixed in
fix round 1. The round-2 verdicts are not yet recorded, so S-17 stays OPEN.

## The fault at `14a93602`

A Parquet data file without embedded field ids is read through one of two
routes: a name mapping (`schema.name-mapping.default`) or the fallback
stamping by name and position. Both routes stamped top-level fields only, so
a struct child had no id. Three shapes failed on the base. The table column
`s` is a struct, and the data file has no field ids.

| Shape | Table change | Base answer |
|---|---|---|
| (i) two-leaf | `s<d, e>`: add `s.k`, drop `s.d` and `s.e` | `k = [Some(1), None]`: the dropped `d` reads under `k` |
| (ii) one-leaf | `s<d>`: add `s.k`, drop `s.d` | batch column `Struct<d>` under field `Struct<k>`; `concat_batches` refuses |
| (iii) added child | `s<d>`: add `s.k` alone | `DataInvalid`: `Incorrect number of arrays for StructArray fields, expected 2 got 1` |

The nested shape (f) (`s<t<d>>`, add `s.t.k`, drop `s.t.d`) failed the same
way on the base. The list shape (g) (`l<item<d>>`, add `l.element.k`, drop
`l.element.d`) failed with `failed to rebuild projected list`.

The parent ledger's round-2 verifier found this fault as its F1 (S-16 of
F-STRUCT-EMPTY-CHILD-PANIC-1), and the RP-63 records round carded it on the
RePark side as F-IDLESS-STRUCT-CHILD-REMAP-1. The causes are the positional
`Cast` in the id-less arm of `nested_projection::build_struct` and the
top-level-only stamping in `apply_name_mapping_to_arrow_schema` and
`add_fallback_field_ids_to_arrow_schema`.

## The fix

No scan-planner and no `FileScanTask` change.

The bullets below record the round-0 fix at `51f39a29` and `8a66aa7c`. Fix
round 1 superseded three of them, marked inline: the nested counter, the
id-less arm of `build_struct`, and `unmapped_group_leaf_indices`. The Fix
round 1 section is the current state.

- `crates/iceberg/src/arrow/field_id_stamping.rs` (new; 332 lines at head).
  Both stamping functions move here from `reader.rs` and become recursive over
  Struct, List, LargeList, FixedSizeList and Map. The Map `entries` struct
  takes no id. The recursion is depth-limited by `MAX_NESTED_PROJECTION_DEPTH`,
  and both functions return `Result`.
  - Mapping route: a struct child takes the id of the mapped field among its
    parent's `MappedField::fields()` whose names contain the child name. A list
    element is looked up as `element`, a map key as `key` and a map value as
    `value`. A node with no mapping takes no id, and neither does its subtree.
  - Fallback route, round 0 (superseded in fix round 1): pass 1 matched by name
    and position through matched parents, and pass 2 gave each unmatched node a
    pre-order counter from `top_level_count + 1`. Fix round 1 removed the
    counter: below the top level a nested child takes the id of the child with
    its name in its matched parent, and nothing nested is stamped by position.
    The top level keeps name and position, else `position + 1` unless that id
    is claimed. `build_fallback_field_id_map` agreed with it at round 0; F4
    removed that map from the pruning path.
  - A claimed id is never reused.
- `crates/iceberg/src/arrow/field_id_stamping_tests.rs` (new; 347 lines at
  head): ten unit pins, see Pins.
- `crates/iceberg/src/arrow/reader.rs`: both functions and their doc and `//`
  comments are removed. The import and two `?` change. The file goes from 9546
  to 9442 lines at round 0, and 9433 lines at head.
- `scripts/check_rust_file_size.py`: the legacy ceiling of `reader.rs` lowers
  from 9546 to 9442 at round 0, and to 9433 at head. It is never raised.
- `crates/iceberg/src/arrow/row_filter_plan.rs`: round 0 added
  `unmapped_group_leaf_indices` (superseded in fix round 1: deleted, together
  with `collect_unstamped_leaves` and `collect_unstamped_type`).
- `crates/iceberg/src/arrow/nested_projection.rs`: round 0 passed a source struct
  through on an exact name, type and nullability match, and refused otherwise
  (superseded in fix round 1, ruling 2: a struct whose source children carry
  no ids fills every target child with NULL or its initial default, and the
  passthrough and `same_struct_shape` are deleted).
- `crates/iceberg/src/arrow/mod.rs` (92 to 95 lines): `mod field_id_stamping;`
  and `#[cfg(test)] mod field_id_stamping_tests;`.

## Pins

Integration pins: `crates/iceberg/tests/struct_idless_child.rs` (18) and
`crates/iceberg/tests/struct_idless_child_2.rs` (29, fix round 1). Unit pins:
`crates/iceberg/src/arrow/field_id_stamping_tests.rs` (10 at head; 7 at round
0).

Red run on `14a93602` (`cargo test -p iceberg --test struct_idless_child`,
exit 101): 5 passed, 13 failed. Green run on `8a66aa7c`: 18 passed.

| Shape | Fallback pin | Mapping pin | Base `14a93602` | Fix `8a66aa7c` |
|---|---|---|---|---|
| (i) two-leaf, dropped children, new child | `idless_two_leaf_struct_dropped_children_new_child_reads_null_fallback` | `idless_two_leaf_struct_dropped_children_new_child_reads_null_mapping` | FAILED: `[Some(1), None]` against `[None, None]` | ok |
| (ii) one-leaf, dropped child, new child | `idless_one_leaf_struct_dropped_child_new_child_matches_schema_fallback` | `idless_one_leaf_struct_dropped_child_new_child_matches_schema_mapping` | FAILED: column `Struct<d>` against field `Struct<k>` | ok |
| unchanged struct (e'), batch type and concatenation | `idless_struct_batches_carry_the_schema_type_and_concatenate_fallback` | `idless_struct_batches_carry_the_schema_type_and_concatenate_mapping` | FAILED: `concat_batches` type mismatch | ok |
| (iii) added child beside a file child | `idless_struct_with_added_child_reads_file_child_and_null_fallback` | `idless_struct_with_added_child_reads_file_child_and_null_mapping` | FAILED: `DataInvalid` cast | ok |
| emptied id-less struct | `idless_emptied_struct_reads_empty_struct_per_row_validity_fallback` | `idless_emptied_struct_reads_empty_struct_per_row_validity_mapping` | ok (correct on the base, unpinned) | ok |
| all children in the task schema | `idless_struct_with_every_child_in_the_task_schema_is_unchanged_fallback` | `idless_struct_with_every_child_in_the_task_schema_is_unchanged_mapping` | ok | ok |
| nested struct (f) | `idless_nested_struct_dropped_child_new_child_reads_null_per_row_fallback` | `idless_nested_struct_dropped_child_new_child_reads_null_per_row_mapping` | FAILED: `t.d` against `t.k` | ok |
| list element struct (g) | `idless_list_element_struct_dropped_child_new_child_reads_null_fallback` | `idless_list_element_struct_dropped_child_new_child_reads_null_mapping` | FAILED: `ListArray` element type | ok |
| no child mapping, same shape (renamed in fix round 1) | `idless_struct_without_child_mapping_and_the_same_shape_reads_by_shape` at round 0; `idless_struct_without_child_mapping_reads_its_unmapped_child_as_null` at head (no mapping route) | not applicable | ok | round 0 answer `d = [1, None]`; head `d = [None, None]` |
| no child mapping, different shape (renamed in fix round 1) | `idless_struct_without_child_mapping_and_a_different_shape_refuses` at round 0; `idless_struct_without_child_mapping_and_an_added_child_reads_null` at head (no mapping route) | not applicable | FAILED: `DataInvalid` with no refusal text | round 0 refused; head `d = [None, None]`, `k = [None, None]` |

Round-0 unit pins (seven, all green in the `iceberg` lib run at `8a66aa7c`;
their base state was not measured): `nested_name_mapping_stamps_struct_children_and_list_elements`,
`nested_name_mapping_leaves_an_unmapped_child_without_an_id`,
`fallback_stamps_a_nested_child_by_name_and_position`,
`fallback_gives_an_unmatched_nested_child_its_preorder_counter`,
`fallback_matches_a_list_element_structurally`,
`fallback_counter_that_collides_with_a_claimed_id_yields_no_id`,
`fallback_keeps_the_top_level_counter_and_name_match`. Fix round 1 removed the
two counter pins and replaced the by-position pin with
`fallback_stamps_a_nested_child_by_name_at_any_position`; the ten unit pins at
head are listed in the Fix round 1 section.

## Clauses

| Id | Claim | Verdict | Pins and evidence |
|---|---|---|---|
| S-1 | The fault reproduces red on `14a93602`: 13 of the 18 integration pins fail there. The other 5 pass there as no-regression pins | PROVEN | Red run on `14a93602` (see Pins) |
| S-2 | Mapping route: each struct child, list element, map key and map value of an id-less file takes its id from the name mapping at every depth. An unmapped node and its subtree take no id. Fix round 1 keeps the round-0 mapping-route answers, except the partial mapping (F3), which now reads NULL (S-19) | PROVEN | The `_mapping` pins of shapes (i), (ii), (f) and (g); units `nested_name_mapping_stamps_struct_children_and_list_elements` and `nested_name_mapping_leaves_an_unmapped_child_without_an_id` |
| S-3 | Fallback route, rewritten by fix round 1 (ruling 1). A top-level field takes the table field at its position with its name, else `position + 1` unless that id is claimed. A nested child takes the id of the child with its name in its matched parent's type, at any position and at every depth. A list element, map key and map value take the table's element, key and value ids when the parent matched. A nested child whose name matches nothing takes no id, and neither does its subtree. Nothing nested is stamped from a counter or by position. A claimed id is never reused | PROVEN | The `_fallback` pins of shapes (i), (ii), (f) and (g); the pins of S-18; units `fallback_stamps_a_nested_child_by_name_at_any_position`, `fallback_leaves_an_unmatched_nested_child_and_its_subtree_without_an_id`, `fallback_never_stamps_a_nested_child_from_a_counter`, `fallback_top_level_counter_field_stamps_its_children_by_name`, `fallback_matches_map_key_and_value_and_their_children_by_name`, `fallback_matches_a_list_element_structurally` and `fallback_keeps_the_top_level_counter_and_name_match` |
| S-4 | Shape (i): a dropped child's value never reads under a new child. The new child reads NULL per row. Validity comes from the file | PROVEN | The two-leaf pins, both routes |
| S-5 | Shape (ii), and the batch of an unchanged struct (e'): the batch column has the schema field's type, with the nested field-id metadata, and `concat_batches` succeeds. The `matches_schema` pins and the batch pins stay green at `85687ce0` after the round-0 passthrough was removed | PROVEN | The one-leaf `matches_schema` pins; the `idless_struct_batches_carry_the_schema_type_and_concatenate` pins, both routes, red on the base because the passed-through column lacked the nested field-id metadata |
| S-6 | Shape (iii): an added child beside a file child reads the file child's values and NULL for the new child | PROVEN | The `idless_struct_with_added_child_reads_file_child_and_null` pins, both routes |
| S-7 | An emptied id-less struct, every child dropped, reads a non-null empty struct where the file row was non-null and NULL where it was NULL. This closes the emptied half of the parent ledger's S-11 by citation | PROVEN | The `idless_emptied_struct_reads_empty_struct_per_row_validity` pins, both routes. Green on the base too: it was correct there and unpinned |
| S-8 | No regression: a struct whose task children are all in the file is unchanged, on both routes | PROVEN | The `idless_struct_with_every_child_in_the_task_schema_is_unchanged` pins. Green on the base |
| S-9 | The nested shape (f) and the list shape (g) read the table's shape per row, with the file's validity at each level | PROVEN | The `idless_nested_struct_dropped_child_new_child_reads_null_per_row` pins and the `idless_list_element_struct_dropped_child_new_child_reads_null` pins, both routes |
| S-10 | Round-0 claim: without a child mapping, a file struct whose children differ from the table refuses with `DataInvalid`, naming the struct path and `schema.name-mapping.default`; a file struct of the same shape reads by shape | REJECTED | The round-0 refusal claim was withdrawn by ruling 2 of fix round 1 and replaced by S-19 (the NULL answer). The two no-mapping pins were rewritten at `85687ce0`; the refusal is gone from the id-less route |
| S-11 | Spark 4.1.2 answers for the id-less shapes, measured 2026-11-01 (see the Spark section, including the fix round 1 additions). The NULL-row answer differs from the fork's; that difference is the per-write NULL collapse of the struct ledger's S-13, ruled 2026-10-31 and not copied. Spark and RePark's `add_files` install `schema.name-mapping.default`, so every id-less file either of them reads takes the mapping route. The claim holds for the mapping route. On the fallback route a rename reads NULL (S-18), which Spark never reaches | PROVEN (mapping route) | Orchestrator measurement 2026-11-01, copied; the mapping-route pins of S-18 and S-19 |
| S-12 | Java note, restated: Java's `ParquetSchemaUtil.addFallbackIds` stamps top-level ordinals only, and its fallback read builder reads nested structs by position. This is the builder's recollection and stands unverified. The measured fact is that Spark always reads an id-less file through the mapping `add_files` installs, and the fork's fallback rule is by name with no counter (fix round 1), so the counter-order question no longer changes any fork answer | OPEN | Not checked against the Java source. No Java checkout was used |
| S-13 | Pruning: on an id-less file, row-group and page-index pruning map field ids to Parquet columns through the ids the reader stamped on the Arrow schema (the mapping or the fallback rule), at every depth. A column with no stamped id has no entry, so a filter on it keeps every row group. Before fix round 1 the map was positional (`build_fallback_field_id_map`), which read the wrong column's statistics when a mapping or the fallback rule gave other ids (F4) | PROVEN | Fix round 1 (`cc213112`). Integration pins `idless_row_group_pruning_uses_the_mapped_ids` and `idless_row_group_pruning_uses_the_fallback_ids_after_a_leading_drop`, both red at `bc6a2e67`; unit `stamped_field_id_map_indexes_stamped_leaves_at_every_depth` |
| S-14 | File moves and ceiling: both stamping functions and their tests move out of `reader.rs` (9546 to 9442 lines at round 0, 9433 at head). The `reader.rs` ceiling in `scripts/check_rust_file_size.py` lowers at each commit and is never raised. `build_fallback_field_id_map` moved to the test-only `page_prune_fixture.rs` | PROVEN | `git diff --stat 14a93602..85687ce0`. `make check-rust-file-size` exit 0 (builder, fix round 1) |
| S-15 | Neighbours unchanged: the 13 `struct_empty_child` pins, the `interop_name_mapping` binary, and the full `iceberg` and `iceberg-datafusion` suites pass with no failure, at round 0 and at `85687ce0` | PROVEN | Builder gate runs (see Gates) |
| S-16 | NULL-free parity: an id-less file with no NULL struct row reads Spark's NULL-free answers in S-11, shapes (ii) and (iii), on both routes | PROVEN | Fix round 1 (`85687ce0`), ruling 4: `idless_null_free_struct_dropped_child_new_child_reads_null_fallback` and `_mapping` (shape ii); `idless_null_free_struct_added_child_reads_file_child_and_null_fallback` and `_mapping` (shape iii). All four green at head |
| S-17 | Verification: the union of the PASS verdicts of an Opus 5.5 verifier and a Muse MAX critic on the current head is recorded | OPEN | The round-1 verdicts (Opus 5.5 FAIL; Muse MAX PASS with findings) are in the Verifier round 1 section. Round-2 verdicts on `85687ce0` are not yet recorded. The closures of the parent ledger's S-11 and S-16 depend on this clause |
| S-18 | Fallback route, by name (ruling 1), F1 and F2 shapes, both routes. A rename, a double rename, a move, reordered file children, a dropped child beside an added one, an extra file child then an added child, a list element's children and a two-deep struct with a rename each read the table's child by name, and a child with no history reads NULL, never a neighbour's value | PROVEN | Fallback pins, red at `bc6a2e67` unless noted: `idless_renamed_child_after_a_nested_sibling_reads_null_fallback` (green at `bc6a2e67` by design: round 1 already read NULL); `idless_two_renamed_children_never_read_a_neighbour_fallback`; `idless_moved_child_reads_by_name_fallback`; `idless_file_children_in_another_order_read_by_name_fallback`; `idless_dropped_child_never_lends_its_value_to_a_sibling_fallback`; `idless_extra_file_child_never_reads_under_an_added_child_fallback`; `idless_list_element_children_read_by_name_fallback`; `idless_struct_two_deep_reads_by_name_and_a_rename_reads_null_fallback`. Mapping-route no-regression pins, green at `bc6a2e67` and at head: `idless_renamed_child_after_a_nested_sibling_keeps_its_value_mapping`, `idless_two_renamed_children_keep_their_values_mapping`, `idless_moved_child_reads_by_name_mapping`, `idless_file_children_in_another_order_read_by_name_mapping`, `idless_dropped_child_never_lends_its_value_to_a_sibling_mapping`, `idless_extra_file_child_never_reads_under_an_added_child_mapping`, `idless_list_element_children_read_by_name_mapping`, `idless_struct_two_deep_reads_by_name_and_a_rename_keeps_its_value_mapping`. Source: `struct_idless_child_2.rs` |
| S-19 | Unmapped child (ruling 2, F3): on an id-less file, a child of a mapped struct that the mapping or the name leaves unstamped reads NULL per row, on both routes, and is never refused and never projected. Where the file has no embedded ids at all, an unmapped child is never a refusal | PROVEN | Red at `bc6a2e67`: `idless_partial_nested_mapping_reads_the_unmapped_child_as_null` (the F3 shape: mapping `s -> [d]`, file `s<d = 10, e = 20>`, answer `{d: 10, e: NULL}`); `idless_file_children_with_no_matching_name_read_null_mapping` and `idless_file_children_with_no_matching_name_read_null_fallback`. Renamed no-mapping pins, green at head: `idless_struct_without_child_mapping_reads_its_unmapped_child_as_null` and `idless_struct_without_child_mapping_and_an_added_child_reads_null` (`struct_idless_child.rs`) |
| S-20 | Top-level positional misread, OPEN and not fixed by this unit. On the fallback route, a table `(x, a, b)`, no drop, and an id-less file `(a, b)` with rows `a = [1, 2]`, `b = [40, 30]` read `x = [1, 2]` (a's values), `a = [40, 30]` (b's values) and `b = [NULL, NULL]`. The top-level rule (name and position, else `position + 1`) is unchanged by this unit and pre-existing at `14a93602`. The Muse MAX critic's out-of-scope note, identical on base and head: a top-level reorder on the fallback route fails with a `Struct -> Int32` cast error. Spark never reaches this shape because `add_files` installs a mapping; RePark's `add_files` installs one too, so the shape needs a table registered from foreign metadata with no mapping | OPEN | Measured once at `85687ce0` by the builder (scratch test, since deleted). Carded as F-IDLESS-TOPLEVEL-POSITIONAL-1, to be filed on the RePark side by the RP-64 records round, not fixed here |
| S-21 | Ruling 7 (the Muse MAX critic's F2, folded): on an id-less file, a nested leaf whose file type cannot be promoted to the table type refuses `DataInvalid` naming the field path and both types, in both directions and on both routes, instead of collapsing the struct's validity to `[f, f]`. A top-level incompatible type keeps its old behaviour (the column is excluded and reads NULL) | PROVEN | Red at `bc6a2e67` (the scan read batches where it must refuse): `idless_nested_int_under_a_string_child_refuses_fallback` and `_mapping`; `idless_nested_string_under_an_int_child_refuses_fallback` and `_mapping`. Green at head. The refusal is raised in `get_arrow_projection_mask_with_field_ids` (`reader.rs`) |
| S-22 | A name mapping whose list or map entry lacks its `element`, `key` or `value` child now reads that list or map as NULL where round 1 read its values. Judged spec-consistent (an unmapped field is not projected) | OPEN | Builder's question, pending the round-2 verifier. No pin is named in the builder's notes |
| S-23 | An incompatible nested type on a file WITH embedded field ids (table `s<d: string>`, file `s<d: int>` with embedded ids `1, 2, 3`, values `[1, NULL]`) still collapses the struct validity to `[false, false]` with a NULL `StringArray` child at `85687ce0`. The item-7 refusal is gated on `missing_field_ids`, so this case is outside this unit's scope and the reader behaves as at round 0 | OPEN | Measured at `85687ce0`. A candidate card, a follow-up to the card of S-20 or to a new unit |

VERDICT (2026-11-01, fix round 1 records): 17 PROVEN, 5 OPEN (S-12, S-17,
S-20, S-22, S-23), 1 REJECTED (S-10), of 23 clauses. S-17 stays OPEN.

## Spark 4.1.2 measurement (2026-11-01, S-11)

Spark 4.1.2 with Iceberg 1.11.0, hadoop catalog. The orchestrator measured
it on 2026-11-01. The id-less file came from Spark's plain Parquet writer with
no field id at any level, and `add_files` admitted it. Each shape ran with no
name mapping, and with `schema.name-mapping.default` naming `id`, `s` and
`s.d`. Both routes gave the same answers.

- Shape (ii), NULL-free file `[(1, {d: 1}), (2, {d: 2})]`, `ADD COLUMN s.k`,
  `DROP COLUMN s.d`: `[(1, Row(k=None)), (2, Row(k=None))]`. `s IS NULL` is
  false on both rows. `WHERE s.k = 1` matches no row.
- Shape (ii), file with a NULL struct row `[(1, {d: 1}), (2, NULL)]`:
  `[(1, NULL), (2, NULL)]`. The fork answers `[Row(k=None), NULL]`, which the
  one-leaf pins assert. The difference is the per-write NULL collapse of the
  struct ledger's S-13, corrected by the errata in that ledger. The owner
  ruling of 2026-10-31 stands, and the fork keeps the per-row answer.
- Shape (iii), `ADD COLUMN s.k` alone on the NULL-free file:
  `[(1, Row(d=1, k=None)), (2, Row(d=2, k=None))]`.

Spark does not read a dropped child's value under a new child on an id-less
file. The orchestrator reads the fork's NULL-free answers as the same. S-16
records that the round-1 pins cover them.

### Fix round 1 additions (orchestrator, 2026-11-01)

Measured on Spark 4.1.2 with Iceberg 1.11.0. Copied, not re-measured by the
clerk.

- `CALL system.add_files` on an id-less file installs `schema.name-mapping.default`
  built from the table schema (measured as a table property). Iceberg's schema
  updates maintain it: `rename s.u -> v` leaves the names `[u, v]` on the field,
  and `add s.k` appends `k`. So through Spark every id-less file is read on the
  mapping route, and the no-mapping fallback never happens.
- RePark's `CALL <catalog>.system.add_files` also installs the mapping, measured
  from the table schema. After `RENAME COLUMN s.d TO v`, RePark reads `{v: 1}`
  through the fork at `14a93602`'s mapping route. The RePark route is the mapping
  route; the fallback route is reached only by a table whose files carry no ids
  and whose properties carry no mapping (a table registered from foreign
  metadata, or the property removed).
- With that mapping Spark reads: rename `s.u -> v` as `v = 7`; the double rename
  as `uu = 7, ww = 8`; `move s.w first` as `[8, {d: 1}, 7]`; file `s<b = 10, a = 20>`
  under table `s<a, b>` as `{a: 20, b: 10}`; file `s<d = 1, x = 99>` then `add s.k`
  as `{d: 1, k: NULL}`; the partial mapping `s -> [d]` as `{d: 10, e: NULL}`; and
  the two pruning filters each return their one row.
- Spark never guesses a nested position on an id-less file, because it always has
  the mapping. The round-0 nested fallback by position is what produced F1 and F2.

## Verifier round 1 (Opus 5.5 on `8a66aa7c`): FAIL

Verdict FAIL with four findings. The orchestrator reproduced F1, F2 and F4 at
`8a66aa7c`, and the findings are copied verbatim.

- **F1 (S1, regression against `14a93602`, fallback route).** Table
  `(id, s STRUCT<t STRUCT<d>, u, w>)`, one id-less file row
  `(1, {t: {d: 1}, u: 7, w: 8})`, no mapping. After `rename s.u -> v` the head
  reads `{t: {d: 1}, v: NULL}`; after `rename s.u -> uu` and `s.w -> ww` it
  reads `{uu: NULL, ww: 7}`: `ww` gets `u`'s value. Cause: the nested counter is
  pre-order while Iceberg numbers all of a struct's children before recursing,
  and `FallbackIds::claim` refuses only ids already claimed in the file schema,
  not live task-schema ids whose field did not match, so a counter id lands on
  a live field.
- **F2 (S1, positional misreads that were loud at the base and are silent now,
  fallback route).** Table `s<a, b>` with file `s<b = 10, a = 20>` reads
  `{a: 10, b: 20}`; after `move s.w first` on the F1 table the head reads
  `{w: 7, u: NULL}`; table `s<d>` with file `s<d = 1, x = 99>` then `add s.k`
  reads `{d: 1, k: 99}`. Expected `{a: 20, b: 10}`, `{w: 8, u: 7}`, `{d: 1, k: NULL}`.
  The mapping route is correct on all three.
- **F3 (S2, mapping route).** Table `s<d, e>`, file `s<d = 10, e = 20>`, mapping
  `s -> [d]` with no `e` entry: `DataInvalid: file mixes stamped and unstamped
  nested field ids in struct 's' at child 'e'; set a name mapping ...`, a
  circular hint since a mapping is set. Expected `{d: 10, e: NULL}`: an unmapped
  field is not projected (the specification; Spark 4.1.2 measured `[10, NULL]`).
- **F4 (S1, predates this diff, identical at `14a93602`).** Row-group and
  page-index pruning on an id-less file use `build_fallback_field_id_map`
  (position + 1) even when the mapping or the fallback rule stamped different
  ids. Measured: an id-less file with two row groups and columns in order
  `(b, a)`, mapping `a -> 1, b -> 2`, filters `a = 1` and `b = 10` return `[]`;
  on the fallback route a table `(x, a, b)`, `drop x`, file `(a, b)`, filter
  `a = 1` returns `[]`; both correct with row-group filtering off. Silent
  missing rows.

The Muse MAX critic's round-1 verdict on `8a66aa7c` was PASS (52 scratch
probes, suites and red state matching the builder), with three findings: its
F1 is the Opus F4 pruning fault; its F2 (S1, invalid-migration state only) is
an id-less file whose nested child type is incompatible with the table (file
int under table string, and the reverse), where the struct validity collapsed
to `[f, f]` where the file says `[t, f]`, reproduced by the orchestrator and
folded into fix round 1 as ruling 7; its F3 is the Opus F3 partial-mapping
refusal. Its out-of-scope notes: the top-level reorder on the fallback route
fails with a `Struct -> Int32` cast error, identical on base and head (goes to
the top-level card, S-20); and a counter coincidence (file `s<x, y>` under
table `s<a>` read `a = 20`) that the by-name rule removes.

### Rulings

1. **Nested fallback rule: by name within the matched parent, at every depth,
   with no counter.** Top level unchanged. A nested child whose name matches
   nothing gets no id, and neither does its subtree. Landed in `9c4d43a9`
   (F1, F2); pins S-18.
2. **F3: an unmapped child under a mapped struct reads NULL, never a refusal.**
   The mixed-ids refusal stays only for files that arrived with embedded ids,
   and its hint no longer tells a user who has a mapping to set one. Landed in
   `9c4d43a9` (F3); pins S-19. The two round-0 no-mapping pins were rewritten.
3. **F4: pruning uses the stamped ids.** A column with no stamped id keeps every
   row group. Landed in `cc213112`; pins S-13.
4. **NULL-free pins on both routes** for shape (ii) and shape (iii), the shapes
   the Spark measurement covers. Landed in `85687ce0`; pins S-16.
5. **Pins for F1, F2 and F3 on both routes where a ruling gives an answer**,
   red first at `bc6a2e67`, in the sibling `struct_idless_child_2.rs` (under 1000
   lines). Landed in `85687ce0`; pins S-18, S-19, S-21.
6. **Gates** as listed in the Gates section.
7. **Incompatible nested type on an id-less file** (the Muse F2, folded): refuse
   `DataInvalid`, never a collapsed validity. Landed in `9c4d43a9`; pins S-21.

## Fix round 1 (`85687ce0`, on `bc6a2e67`)

Commits (each ends with exactly one `Authored-By: Claude (claude-opus-5-5) <noreply@anthropic.com>`):

- `cc213112` fix(arrow): fix 1 pruning on an id-less file uses the stamped field ids (F4)
- `9c4d43a9` fix(arrow): fix 1 the nested fallback rule is by name with no counter and an unmapped child reads NULL (F1 to F3); also carries item 7
- `85687ce0` test(arrow): fix 1 pins: renames, moves, reorders, an extra child, a partial mapping, pruning and NULL-free files on both routes

Commit order: F4 is first because `reader.rs` must stay at or under its ceiling
at every commit (the F4 commit moves `build_fallback_field_id_map` out of
`reader.rs`, which pays for the item-7 lines). Item 7 sits in the F1-to-F3
commit because it replaces the same hunk of
`get_arrow_projection_mask_with_field_ids`.

### The rule as implemented

- **Fallback route, nested (ruling 1)**, in `field_id_stamping.rs`: a nested
  struct child takes the id of the child with the same name in its matched
  parent's table type (`StructType::field_by_name`), at any position. A list
  element, map key and map value take the table's element, key and value ids
  when the parent matched. A child whose name matches nothing gets no id, and
  neither does its subtree. There is no counter and no position below the top
  level: `FallbackIds` lost `next_counter`, `take_counter` and `skip_subtree`,
  and `Slot::Member` lost its position.
- **Fallback route, top level: unchanged.** A field takes the id of the table
  field at the same position with the same name; otherwise `position + 1`, unless
  that id is already claimed. Pass 1 pre-claims the matched subtrees by name. A
  top-level field that took `position + 1` stamps its children by name under the
  table field with that id, and each child id goes through `claim`.
- **Unmapped children (ruling 2)**: `unmapped_group_leaf_indices` and
  `collect_unstamped_*` are deleted from `row_filter_plan.rs`, so an id-less file
  no longer decodes a leaf under an unstamped node. In
  `nested_projection::build_struct`, the round-0 id-less arm is gone. A struct
  whose source children ALL lack ids fills every target child with NULL or its
  `initial_default`. The mixed stamped and unstamped refusal fires only when some
  siblings carry ids; its hint now reads `...; the data file must carry a field
  id on every nested field`. The `stamped and unstamped` pins in
  `nested_projection_evo_tests.rs` and `nested_projection_evo_pin_tests.rs` stay
  green.
- **Pruning (ruling 3)**: `build_field_id_set_and_map` in `reader.rs` takes the
  stamped Arrow schema. When the Parquet descriptor has no ids it uses
  `stamped_field_id_map` (in `field_id_stamping.rs`), which walks the stamped
  leaves with `Fields::filter_leaves` and maps each stamped leaf id to its leaf
  index at every depth. Both callers pass the stamped schema: the reader's
  row-group path, and `prune_indexed_metadata_for_scan` in `open_parquet.rs`
  (row groups and page index). A column with no stamped id has no entry, and the
  row-group evaluator keeps that row group.
- **Item 7 (ruling 7)**: in `get_arrow_projection_mask_with_field_ids`, a
  projected leaf on an id-less file (`missing_field_ids`) that is nested and fails
  `type_promotion_is_valid` refuses `DataInvalid`:
  `nested field '<path>' in the data file has type <file type>, which cannot be read as the table type <table type>`.
  A top-level incompatible type keeps its old behaviour.

### Red run at `bc6a2e67` (new pins present, code at `bc6a2e67`)

`cargo test -p iceberg --test struct_idless_child_2` exit 101: 13 passed,
16 failed. The 16 failures:

`idless_file_children_with_no_matching_name_read_null_mapping`,
`idless_extra_file_child_never_reads_under_an_added_child_fallback`,
`idless_dropped_child_never_lends_its_value_to_a_sibling_fallback`,
`idless_file_children_with_no_matching_name_read_null_fallback`,
`idless_file_children_in_another_order_read_by_name_fallback`,
`idless_list_element_children_read_by_name_fallback`,
`idless_nested_int_under_a_string_child_refuses_fallback`,
`idless_nested_int_under_a_string_child_refuses_mapping`,
`idless_moved_child_reads_by_name_fallback`,
`idless_nested_string_under_an_int_child_refuses_fallback`,
`idless_nested_string_under_an_int_child_refuses_mapping`,
`idless_row_group_pruning_uses_the_fallback_ids_after_a_leading_drop`,
`idless_row_group_pruning_uses_the_mapped_ids`,
`idless_partial_nested_mapping_reads_the_unmapped_child_as_null`,
`idless_struct_two_deep_reads_by_name_and_a_rename_reads_null_fallback`,
`idless_two_renamed_children_never_read_a_neighbour_fallback`.

The 13 that pass at `bc6a2e67` are listed under Pins at head: the mapping-route
no-regression pins, the single-rename fallback NULL and the four NULL-free pins.
The pruning pins fail on the base with `a = 1` returning 0 rows where 1 is
expected; the `b = 10` assertion of the mapping pin is not reached.

Unit pins (`field_id_stamping_tests.rs`), run against the pre-fix
`field_id_stamping.rs` with `stamped_field_id_map` appended so the file
compiles: exit 101, 5 passed, 5 failed. The five failures are
`fallback_leaves_an_unmatched_nested_child_and_its_subtree_without_an_id`,
`fallback_top_level_counter_field_stamps_its_children_by_name`,
`fallback_stamps_a_nested_child_by_name_at_any_position`,
`fallback_matches_map_key_and_value_and_their_children_by_name` and
`fallback_never_stamps_a_nested_child_from_a_counter`.

### Green run at `85687ce0`

`cargo test -p iceberg --test struct_idless_child --test struct_idless_child_2 --test struct_empty_child`
exit 0: `struct_idless_child` 18 passed, `struct_idless_child_2` 29 passed,
`struct_empty_child` 13 passed, 0 failed in each.

### Pins at head

- `struct_idless_child_2.rs` (766 lines at head; 29 pins, 16 red at `bc6a2e67`):
  the list in the red run above, and the 13 green at `bc6a2e67`:
  `idless_renamed_child_after_a_nested_sibling_reads_null_fallback`,
  `idless_renamed_child_after_a_nested_sibling_keeps_its_value_mapping`,
  `idless_two_renamed_children_keep_their_values_mapping`,
  `idless_moved_child_reads_by_name_mapping`,
  `idless_dropped_child_never_lends_its_value_to_a_sibling_mapping`,
  `idless_file_children_in_another_order_read_by_name_mapping`,
  `idless_extra_file_child_never_reads_under_an_added_child_mapping`,
  `idless_list_element_children_read_by_name_mapping`,
  `idless_struct_two_deep_reads_by_name_and_a_rename_keeps_its_value_mapping`,
  `idless_null_free_struct_dropped_child_new_child_reads_null_fallback`,
  `idless_null_free_struct_dropped_child_new_child_reads_null_mapping`,
  `idless_null_free_struct_added_child_reads_file_child_and_null_fallback`,
  `idless_null_free_struct_added_child_reads_file_child_and_null_mapping`.
- `struct_idless_child.rs` (706 lines at head; 18 pins): the two renamed
  no-mapping pins (see Pins table), 16 others unchanged.
- `field_id_stamping_tests.rs` (347 lines at head; 10 unit pins, all green in the
  full `cargo test -p iceberg` run): `nested_name_mapping_stamps_struct_children_and_list_elements`,
  `nested_name_mapping_leaves_an_unmapped_child_without_an_id`,
  `fallback_stamps_a_nested_child_by_name_at_any_position`,
  `fallback_leaves_an_unmatched_nested_child_and_its_subtree_without_an_id`,
  `fallback_never_stamps_a_nested_child_from_a_counter`,
  `fallback_top_level_counter_field_stamps_its_children_by_name`,
  `fallback_matches_a_list_element_structurally`,
  `fallback_matches_map_key_and_value_and_their_children_by_name`,
  `fallback_keeps_the_top_level_counter_and_name_match`,
  `stamped_field_id_map_indexes_stamped_leaves_at_every_depth`.

### Per-shape answers after the fix (head `85687ce0`)

The table is `s<t<d>, u, w>` with ids `s=2, t=3, u=4, w=5, d=6`. The file row is
`(1, {t: {d: 1}, u: 7, w: 8})`. The mapping is built from the table schema at add
time (`create_name_mapping`, the shape Spark and RePark install).

| Shape | Fallback (head) | Mapping (head) | `bc6a2e67` fallback |
|---|---|---|---|
| rename `s.u -> v` | `{t: {d: 1}, v: NULL, w: 8}` | `{t: {d: 1}, v: 7, w: 8}` | same as head |
| rename `s.u -> uu`, `s.w -> ww` | `{uu: NULL, ww: NULL}` | `{uu: 7, ww: 8}` | `uu: 7`, a neighbour's value (red) |
| `move s.w first` | `{w: 8, t: {d: 1}, u: 7}` | same | `w: 7` (red) |
| `add s.k`, `drop s.u` | `{t: {d: 1}, w: 8, k: NULL}` | same | `w: 7` (red) |
| `s<a, b>`, file `s<b=10, a=20>` | `{a: 20, b: 10}` | same | `a: 10` (red) |
| `s<d>`, file `s<d=1, x=99>`, `add s.k` | `{d: 1, k: NULL}` | same | `k: 99` (red) |
| `s<a>`, file `s<x=10, y=20>` (coincidence) | `{a: NULL}`, validity `[t]` | same | fallback `a: 10`; mapping refused (red both) |
| partial mapping `s -> [d]`, file `s<d=10, e=20>` | not applicable | `{d: 10, e: NULL}` | refused, mixed ids (red) |
| `s<t<a, b>>`, file `t<b=2, a=1>`, rename `s.t.b -> bb` | `{a: 1, bb: NULL}` | `{a: 1, bb: 2}` | `a: 2` (red) |
| `l: list<struct<a, b>>`, file element `<b, a>` | `a = [20, 21]`, `b = [10, 11]` | same | swapped (red) |
| NULL-free (ii) `[(1,{d:1}),(2,{d:2})]`, `add s.k`, `drop s.d` | `Row(k=None)` twice, validity `[t, t]` | same | same |
| NULL-free (iii), `add s.k` | `d = [1, 2]`, `k = [None, None]`, validity `[t, t]` | same | same |
| item 7: table `s<d: string>`, file `s<d: int>` `[1, NULL]`, validity `[t, f]` | `DataInvalid` naming `s.d`, `int`, `string` | same | validity `[f, f]` (red) |
| item 7 reverse: table `s<d: int>`, file `s<d: string>` | `DataInvalid` | same | validity `[f, f]` (red) |
| mapping without child entries, `s<d>` unchanged (round-1 pin rewritten) | not applicable | `d = [None, None]`, validity `[t, f]` | `d = [1, None]` by shape |
| mapping without child entries, `add s.k` (round-1 pin rewritten) | not applicable | `d = [None, None]`, `k = [None, None]` | refused |
| pruning, mapping `a -> 1, b -> 2`, file `(b, a)`, 2 row groups | not applicable | `a = 1`: 1 row; `b = 10`: 1 row, with pruning on and off | 0 rows (red) |
| pruning, fallback `(x, a, b)`, `drop x`, file `(a, b)` | `a = 1`: 1 row; `b = 10`: 1 row | not applicable | 0 rows (red) |

All 18 round-0 pins in `struct_idless_child.rs` keep their round-0 answers on
both routes, except the two no-mapping pins, which ruling 2 changed. Those were
renamed (see the Pins table).

### Top-level positional measurement (card F-IDLESS-TOPLEVEL-POSITIONAL-1)

Measured once by the builder at `85687ce0` on the fallback route, with a scratch
test since deleted.

- Setup: table `(x=1, a=2, b=3)`, no drop, id-less file `(a, b)` with rows
  `a = [1, 2]`, `b = [40, 30]`.
- Result: `x = [Some(1), Some(2)]` (a's values), `a = [Some(40), Some(30)]`
  (b's values), `b = [None, None]`.
- Cause: the top-level `position + 1` counter lands on live, unmatched table ids.
- Not fixed by this unit (fix-round brief, ruling 1). Recorded as S-20 OPEN for the
  RP-64 records round.

### Item 7 on a file WITH embedded ids (not fixed here)

- Table `s<d: string>`, file `s<d: int>` with embedded ids `1, 2, 3`, values
  `[1, NULL]`, validity `[t, f]`.
- Result: struct validity `[false, false]`, and `d` is a NULL `StringArray`. The
  validity collapses the same way.
- The item-7 refusal is gated on `missing_field_ids`, so this case is unchanged.
  Recorded as S-23 OPEN, a candidate card.

### Ledger clauses changed by fix round 1

- **S-3** rewritten: the nested counter is gone; the by-name rule is in place.
- **S-5** kept PROVEN, with a note that the round-0 passthrough was removed.
- **S-10** REJECTED, replaced by S-19.
- **S-11** restated for the mapping route, with the fix round 1 Spark additions.
- **S-12** restated; still OPEN.
- **S-13** PROVEN with the F4 pins.
- **S-14**, **S-15** recounted at head.
- **S-16** PROVEN with the four NULL-free pins.
- **S-17** stays OPEN.
- **S-18** to **S-23** new.

## Gates

Round-0 builder runs, every command exit 0 (at `8a66aa7c`):

- `cargo test -p iceberg --test struct_idless_child`: 18 passed, 0 failed.
- `cargo test -p iceberg --test struct_empty_child`: 13 passed, 0 failed.
- `cargo test -p iceberg --test interop_name_mapping`: 1 passed. The test skips
  without its interop directory variable, and the builder host has no JVM.
- `cargo test -p iceberg`: 5043 passed, 0 failed, 21 ignored, over 72 result
  lines. The lib run is 4653 passed and 9 ignored, and it includes the round-0
  unit pins.
- `cargo test -p iceberg-datafusion`: 802 passed, 0 failed, 15 ignored.
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings`:
  no warnings.
- `cargo fmt --all -- --check`: exit 0.
- `make check-rust-file-size`: exit 0. 708 files clean, 89 legacy ceilings.
- `make check-comment-blocks` and `make check-agent-artifacts`: exit 0.
- Comment gate: only the 16-line Apache licence header of each new `.rs` file.

Fix round 1 builder runs, at `85687ce0`; every cargo command under the flock,
environment and `systemd-run` wrapper, exit 0:

| Command | Exit | Counts |
|---|---|---|
| `cargo test -p iceberg --test struct_idless_child --test struct_idless_child_2 --test struct_empty_child` | 0 | 18 + 29 + 13 passed, 0 failed |
| `cargo test -p iceberg` (`--no-fail-fast`) | 0 | 73 result lines, 5075 passed, 0 failed, 21 ignored |
| `cargo test -p iceberg-datafusion` (`--no-fail-fast`) | 0 | 43 result lines, 802 passed, 0 failed, 15 ignored |
| `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | none |
| `make check-rust-file-size` | 0 | 709 files clean (89 legacy ceilings); `reader.rs` 9433 |
| `BASE_REF=14a93602 make check-comment-blocks` | 0 | no added block over 6 lines |
| `make check-agent-artifacts` | 0 | clean (11 + 4 needles) |

Comment gate per commit (added `//` or `/*` lines in the `.rs` diff): `cc213112`
0, `9c4d43a9` 0, `85687ce0` 16 (the licence header of the new file).

The reader went from 9442 to 9424 at `cc213112` and to 9433 at head; the legacy
ceiling was lowered at each commit and never raised.

Records gates for this commit (clerk), every command exit 0:
`BASE_REF=14a93602 make check-comment-blocks` (no added block over 6 lines);
`make check-agent-artifacts` (11 + 4 needles clean); `make check-matrix-anchors`
(88 GAP_MATRIX rows anchored, IDs unique). The staged diff touches only markdown.

## Verification

OPEN (S-17). Round 1 is FAIL (Opus 5.5) and PASS-with-findings (Muse MAX
critic), both on `8a66aa7c`; every finding is fixed in fix round 1. Round 2 on
`85687ce0`: an Opus 5.5 re-verifier and a Muse MAX critic. Their verdicts are
not yet recorded, and the union of their PASS is what S-17 needs.

## Open items

1. S-12: check the Java note against the Java source, if the Java note matters
   for the closure.
2. S-17: the union of the round-2 PASS verdicts is needed before the closures of
   the parent ledger's S-11 and S-16 in its errata stand.
3. S-20: F-IDLESS-TOPLEVEL-POSITIONAL-1, to be filed on the RePark side by the
   RP-64 records round.
4. S-22: the list and map entry child reads NULL. The round-2 verifier decides.
5. S-23: the embedded-ids item-7 collapse. A candidate card or a unit of its own.
