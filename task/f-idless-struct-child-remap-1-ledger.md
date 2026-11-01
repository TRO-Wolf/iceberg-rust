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
`8a66aa7c` (item 2, pins) and the records commit that adds this ledger
(item 3).

**Model:** builder Claude Opus 5.5 (`claude-opus-5-5`). Records clerk
Claude Haiku 5.5 (`claude-haiku-5-5`).

**Outcome in one line.** An id-less Parquet file gets field ids at every
depth. A dropped struct child never reads under a new child, and an absent
child reads NULL per row. The 18 pins are green on `8a66aa7c`. Verification
is OPEN (S-17).

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

- `crates/iceberg/src/arrow/field_id_stamping.rs` (new, 348 lines). Both
  stamping functions move here from `reader.rs` and become recursive over
  Struct, List, LargeList, FixedSizeList and Map. The Map `entries` struct
  takes no id. The recursion is depth-limited by `MAX_NESTED_PROJECTION_DEPTH`,
  and both functions return `Result`.
  - Mapping route: a struct child takes the id of the mapped field among its
    parent's `MappedField::fields()` whose names contain the child name. A list
    element is looked up as `element`, a map key as `key` and a map value as
    `value`. A node with no mapping takes no id, and neither does its subtree.
  - Fallback route: pass 1 matches by name and position through matched
    parents. At the top level this is the table field at the same position with
    the same name, as before. Pass 2 gives each unmatched node a pre-order
    counter from `top_level_count + 1` unless that id is already claimed in the
    file schema. A colliding node and its subtree take no id. The top level
    keeps `position + 1`, so `build_fallback_field_id_map` still agrees. A node
    that took a counter id matches its children against the table field with
    that id.
  - A claimed id is never reused.
- `crates/iceberg/src/arrow/field_id_stamping_tests.rs` (new, 243 lines): seven
  unit pins.
- `crates/iceberg/src/arrow/reader.rs`: both functions and their doc and `//`
  comments are removed. The import and two `?` change. The file goes from 9546
  to 9442 lines.
- `scripts/check_rust_file_size.py`: the legacy ceiling of `reader.rs` lowers
  from 9546 to 9442. It is never raised.
- `crates/iceberg/src/arrow/row_filter_plan.rs` (508 to 584 lines):
  `unmapped_group_leaf_indices` returns `Result`. It adds only the leaves under a
  node of a projected top-level group that has no stamped id. A fully stamped
  group is read leaf by leaf through `filter_leaves`, so a dropped child is no
  longer decoded. If the Arrow leaf walk and the Parquet leaf count disagree, it
  falls back to the whole group. New helpers: `collect_unstamped_leaves` and
  `collect_unstamped_type`.
- `crates/iceberg/src/arrow/nested_projection.rs` (726 to 779 lines): the
  id-less arm of `build_struct` passes a source struct through only on an exact
  name, type and nullability match at every depth. Metadata is ignored, and so
  are list and map wrapper names. Otherwise it refuses with `DataInvalid`,
  naming the struct path and `schema.name-mapping.default`. It never casts by
  position. New helpers: `same_struct_shape` and `same_field_shape`.
- `crates/iceberg/src/arrow/mod.rs` (92 to 95 lines): `mod field_id_stamping;`
  and `#[cfg(test)] mod field_id_stamping_tests;`.

## Pins

Integration pins: `crates/iceberg/tests/struct_idless_child.rs` (18). Unit
pins: `crates/iceberg/src/arrow/field_id_stamping_tests.rs` (7).

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
| no child mapping, same shape | `idless_struct_without_child_mapping_and_the_same_shape_reads_by_shape` (no mapping route) | not applicable | ok | ok |
| no child mapping, different shape | `idless_struct_without_child_mapping_and_a_different_shape_refuses` (no mapping route) | not applicable | FAILED: `DataInvalid` with no refusal text | ok |

Unit pins, all green in the `iceberg` lib run. The notes record no red run for
them, so their base state is not measured: `nested_name_mapping_stamps_struct_children_and_list_elements`,
`nested_name_mapping_leaves_an_unmapped_child_without_an_id`,
`fallback_stamps_a_nested_child_by_name_and_position`,
`fallback_gives_an_unmatched_nested_child_its_preorder_counter`,
`fallback_matches_a_list_element_structurally`,
`fallback_counter_that_collides_with_a_claimed_id_yields_no_id`,
`fallback_keeps_the_top_level_counter_and_name_match`.

## Clauses

| Id | Claim | Verdict | Pins and evidence |
|---|---|---|---|
| S-1 | The fault reproduces red on `14a93602`: 13 of the 18 integration pins fail there. The other 5 pass there as no-regression pins | PROVEN | Red run on `14a93602` (see Pins) |
| S-2 | Mapping route: each struct child, list element, map key and map value of an id-less file takes its id from the name mapping at every depth. An unmapped node and its subtree take no id | PROVEN | The `_mapping` pins of shapes (i), (ii), (f) and (g); units `nested_name_mapping_stamps_struct_children_and_list_elements` and `nested_name_mapping_leaves_an_unmapped_child_without_an_id` |
| S-3 | Fallback route: a nested node takes its name and position match through matched parents, else a pre-order counter from `top_level_count + 1`. A claimed id is never reused. A colliding node and its subtree take no id. The top level keeps `position + 1` | PROVEN | The `_fallback` pins of shapes (i), (ii), (f) and (g); units `fallback_stamps_a_nested_child_by_name_and_position`, `fallback_gives_an_unmatched_nested_child_its_preorder_counter`, `fallback_matches_a_list_element_structurally`, `fallback_counter_that_collides_with_a_claimed_id_yields_no_id` and `fallback_keeps_the_top_level_counter_and_name_match` |
| S-4 | Shape (i): a dropped child's value never reads under a new child. The new child reads NULL per row. Validity comes from the file | PROVEN | The two-leaf pins, both routes |
| S-5 | Shape (ii), and the batch of an unchanged struct (e'): the batch column has the schema field's type, with the nested field-id metadata, and `concat_batches` succeeds | PROVEN | The one-leaf `matches_schema` pins; the `idless_struct_batches_carry_the_schema_type_and_concatenate` pins, both routes, red on the base because the passed-through column lacked the nested field-id metadata |
| S-6 | Shape (iii): an added child beside a file child reads the file child's values and NULL for the new child | PROVEN | The `idless_struct_with_added_child_reads_file_child_and_null` pins, both routes |
| S-7 | An emptied id-less struct, every child dropped, reads a non-null empty struct where the file row was non-null and NULL where it was NULL. This closes the emptied half of the parent ledger's S-11 by citation | PROVEN | The `idless_emptied_struct_reads_empty_struct_per_row_validity` pins, both routes. Green on the base too: it was correct there and unpinned |
| S-8 | No regression: a struct whose task children are all in the file is unchanged, on both routes | PROVEN | The `idless_struct_with_every_child_in_the_task_schema_is_unchanged` pins. Green on the base |
| S-9 | The nested shape (f) and the list shape (g) read the table's shape per row, with the file's validity at each level | PROVEN | The `idless_nested_struct_dropped_child_new_child_reads_null_per_row` pins and the `idless_list_element_struct_dropped_child_new_child_reads_null` pins, both routes |
| S-10 | Without a child mapping, a file struct whose children differ from the table refuses with `DataInvalid`, naming the struct path and `schema.name-mapping.default`. A file struct of the same shape reads by shape | PROVEN | `idless_struct_without_child_mapping_and_a_different_shape_refuses` and `idless_struct_without_child_mapping_and_the_same_shape_reads_by_shape` (green on the base). The refusal text is in the builder's notes |
| S-11 | Spark 4.1.2 answers for the id-less shapes, measured 2026-11-01 (see the next section). The NULL-row answer differs from the fork's. That difference is the per-write NULL collapse of the struct ledger's S-13, ruled 2026-10-31 and not copied | PROVEN | Orchestrator measurement 2026-11-01, copied |
| S-12 | Java note: Java's `ParquetSchemaUtil.addFallbackIds` stamps top-level ordinals only, and its fallback read builder reads nested structs by position. This is the builder's recollection. The fork's nested fallback rule is its own design, so an absent child reads NULL and never a neighbour's value. Spark 4.1.2 gives the same answers for the measured shapes. The counter order differs from Iceberg's fresh-id order for a struct that holds a nested struct followed by a sibling. An unmatched node takes an id absent from the task schema and is not projected, so the order changes no answer | OPEN | Not checked against the Java source. No Java checkout was used |
| S-13 | Pruning: on an id-less file, row-group and page-index pruning use `build_fallback_field_id_map` (positional ids), even when a name mapping gave different top-level ids. A filter on a mapped column whose id differs from its position could prune on the wrong column's statistics | OPEN | To be measured by the round-1 verifier. Carded on the RePark side if it holds. No pin in this unit |
| S-14 | File moves and ceiling: both stamping functions and their tests move out of `reader.rs` (9546 to 9442 lines). The `reader.rs` ceiling in `scripts/check_rust_file_size.py` lowers to 9442 and is never raised | PROVEN | `git diff --stat 14a93602..8a66aa7c`. `make check-rust-file-size` exit 0 (builder) |
| S-15 | Neighbours unchanged: the 13 `struct_empty_child` pins, the `interop_name_mapping` binary, and the full `iceberg` and `iceberg-datafusion` suites pass with no failure | PROVEN | Builder gate runs (see Gates) |
| S-16 | NULL-free parity: an id-less file with no NULL struct row reads Spark's NULL-free answers in S-11, shapes (ii) and (iii), on both routes | OPEN | No pin. All nine `write_idless` fixtures in the integration file hold a NULL struct or list row, so no pin covers a NULL-free file. The orchestrator reads the fork's answers as equal to Spark's on every NULL-free write by mechanism, not by a pin |
| S-17 | Verification: an Opus 5.5 verifier and a Muse MAX critic PASS on `8a66aa7c`, and the union of their PASS is recorded | OPEN | Not yet recorded. The closures of the parent ledger's S-11 and S-16 depend on this clause |

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
records that no pin covers them.

## Gates

Builder runs, every command exit 0. The full-suite counts are from the fix
tree. The three named binaries were re-run on `8a66aa7c`.

- `cargo test -p iceberg --test struct_idless_child`: 18 passed, 0 failed.
- `cargo test -p iceberg --test struct_empty_child`: 13 passed, 0 failed.
- `cargo test -p iceberg --test interop_name_mapping`: 1 passed. The test skips
  without its interop directory variable, and the builder host has no JVM.
- `cargo test -p iceberg`: 5043 passed, 0 failed, 21 ignored, over 72 result
  lines. The lib run is 4653 passed and 9 ignored, and it includes the seven
  unit pins.
- `cargo test -p iceberg-datafusion`: 802 passed, 0 failed, 15 ignored.
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings`:
  no warnings.
- `cargo fmt --all -- --check`: exit 0.
- `make check-rust-file-size`: exit 0. 708 files clean, 89 legacy ceilings.
- `make check-comment-blocks` and `make check-agent-artifacts`: exit 0.
- Comment gate: only the 16-line Apache licence header of each new `.rs` file.

## Verification

OPEN (S-17). An Opus 5.5 verifier and a Muse MAX critic are running on
`8a66aa7c`. The union of their PASS is not yet recorded.

## Open items

1. S-13: the pruning probe, for the round-1 verifier. Carded on the RePark side
   if it holds.
2. S-12: check the Java note against the Java source, if the Java note matters
   for the closure.
3. S-16: add a NULL-free pin on each route, or accept the mechanism. This is the
   orchestrator's decision.
4. S-17: the union of PASS verdicts is needed before the closures of the parent
   ledger's S-11 and S-16 in its errata stand.
