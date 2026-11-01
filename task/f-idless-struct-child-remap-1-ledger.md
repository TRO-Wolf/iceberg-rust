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

**Errata (2026-11-01).** S-26's mapping-route sentence ("The mapping route reads NULL, which is correct") is superseded by F-NAME-MAPPING-SCHEMA-UPDATE-1 (branch `fix/name-mapping-schema-update-1`, ledger `task/f-name-mapping-schema-update-1-ledger.md`). A schema update now rewrites `schema.name-mapping.default` as Java's `SchemaUpdate.commit` does, so a child dropped and re-added by name moves to the new field id on the mapping route and reads its old value under the re-added field (`{a: 1, b: 2}`). That is Spark 4.1.2's answer (`[(1, {a: 1, b: 2}, 5)]`), not NULL. The no-mapping route and the rest of S-26 stand.

**Date:** 2026-11-01. **Base:** fork `main` `14a93602`. **Branch:**
`fix/idless-struct-child-remap-1`. Commits: `51f39a29` (item 1, fix),
`8a66aa7c` (item 2, pins), `bc6a2e67` (item 3, this ledger's first records
commit), `cc213112` (fix round 1, F4 pruning), `9c4d43a9` (fix round 1, F1 to
F3 and ruling 7), `85687ce0` (fix round 1, pins), `b47fdda1` and `b16def80` (fix round 1 records),
`9f928be2` (fix round 2, the gated NULL fill), `2f0c16f6` (fix round 2, the
incomplete list or map mapping), `e0bfed03` (fix round 2, pins), `fc8ff8fe`
(fix round 2 records), `72aaa7c5` (fix round 3, V3-F1, the any-depth id gate),
`e7fb600d` (fix round 3, V3-F2, the mark kept off the output), `7db298ed` (fix
round 3, V3-N1, the projected-only refusal), `d856d06e` (fix round 3, V3-N2,
the list and map refusal in a file with ids), `c9fd7dc4` (fix round 3, pins),
and the fix-round 3 records commit that adds this text.

**Model:** builder Claude Opus 5.5 (`claude-opus-5-5`), round 0, fix round 1,
fix round 2 and fix round 3. Records clerk Claude Haiku 5.5
(`claude-haiku-5-5`), round 0, fix round 1, fix round 2, fix round 3 and the round-4 closure.

**Outcome in one line.** An id-less Parquet file gets field ids at every depth.
A nested child takes the id of the child with its name in the matched parent,
and nothing nested is stamped by a counter or by position. A dropped or added
struct child never reads a neighbour's value, an absent or unmapped child reads
NULL per row, and pruning on an id-less file uses the ids the reader stamped.
A file has ids when a field id is present at any depth (fix round 3, Java
`HasIds`); only a file with none is stamped. At `c9fd7dc4` the 18 pins of
`struct_idless_child.rs`, the 29 of `struct_idless_child_2.rs`, the 14 of
`struct_idless_child_3.rs`, the 16 of `struct_idless_child_4.rs`, the 13
`struct_empty_child` pins and the unit pins of `field_id_stamping_tests.rs` are
green. Verifier round 1 (Opus 5.5) FAILED `8a66aa7c` with F1 to F4, and the Muse
MAX critic PASSED it with three findings; all of them are fixed in fix round 1.
Verifier round 2 on `85687ce0` FAILED from both verifiers (Opus 5.5 and Muse
MAX). Fix round 2 gates the NULL fill of an unstamped struct on a file that
carries no field ids, and refuses a list or map that the mapping names without
its entries. The one open limitation of round 2 is a re-added child on the
no-mapping route (S-26). Verifier round 3 on `e0bfed03` FAILED (Opus 5.5: V3-F1,
V3-F2, V3-N1, V3-N2) and PASSED with notes (Muse MAX). Fix round 3 takes all four
findings (S-28 to S-31) and records the Avro and ORC consequence of fix round 2
as unreachable (S-32). The round-4 verdicts on `c9fd7dc4` are recorded (a Muse MAX critic PASS with
zero findings and an Opus 5.5 re-verifier PASS), so S-17 is PROVEN and the
parent ledger's closures of S-11 and S-16 stand.

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
`crates/iceberg/tests/struct_idless_child_2.rs` (29, fix round 1), and
`crates/iceberg/tests/struct_idless_child_3.rs` (14, fix round 2; 6 red at
`b16def80`). Unit pins:
`crates/iceberg/src/arrow/field_id_stamping_tests.rs` (10 at head; 7 at round 0). The fix round 2 pins are listed in the Fix round 2
section.

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
| S-2 | Mapping route: each struct child, list element, map key and map value of an id-less file takes its id from the name mapping at every depth. An unmapped node and its subtree take no id. Fix round 1 keeps the round-0 mapping-route answers, except the partial mapping (F3), which now reads NULL (S-19). Fix round 2 refuses a list or map that the mapping names without its entries (S-25) | PROVEN | The `_mapping` pins of shapes (i), (ii), (f) and (g); units `nested_name_mapping_stamps_struct_children_and_list_elements` and `nested_name_mapping_leaves_an_unmapped_child_without_an_id` |
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
| S-14 | File moves and ceiling: both stamping functions and their tests move out of `reader.rs` (9546 to 9442 lines at round 0, 9433 at fix round 2, 9429 at fix round 3). The `reader.rs` ceiling in `scripts/check_rust_file_size.py` lowers at each commit and is never raised. `build_fallback_field_id_map` moved to the test-only `page_prune_fixture.rs` | PROVEN | `git diff --stat 14a93602..85687ce0`. `make check-rust-file-size` exit 0 (builder, fix round 1; builder, fix round 3 at `c9fd7dc4`, ceiling 9433 lowered to 9429) |
| S-15 | Neighbours unchanged: the 13 `struct_empty_child` pins, the `interop_name_mapping` binary, and the full `iceberg` and `iceberg-datafusion` suites pass with no failure, at round 0, at `85687ce0`, at `e0bfed03` (fix round 2 gates) and at `c9fd7dc4` (fix round 3 gates) | PROVEN | Builder gate runs (see Gates) |
| S-16 | NULL-free parity: an id-less file with no NULL struct row reads Spark's NULL-free answers in S-11, shapes (ii) and (iii), on both routes | PROVEN | Fix round 1 (`85687ce0`), ruling 4: `idless_null_free_struct_dropped_child_new_child_reads_null_fallback` and `_mapping` (shape ii); `idless_null_free_struct_added_child_reads_file_child_and_null_fallback` and `_mapping` (shape iii). All four green at head |
| S-17 | Verification: the union of the PASS verdicts of an Opus 5.5 verifier and a Muse MAX critic on the current head is recorded | PROVEN | Round 4 on `c9fd7dc4`, the fix-round-3 code head, copied from the two verifiers' reports (the clerk ran no probe): the Muse MAX critic PASS with zero findings (32 scratch probes and 135 neighbour runs; the builder's two questions confirmed: a projected list or map with unstamped slots refuses on both routes and through a row filter, an unprojected one never refuses; a hand-built task projecting only a map descendant refuses loudly); the Opus 5.5 re-verifier PASS (every round-3 finding closed on both routes, on plain scans, through row filters and with `_pos`; its extra probes: a list element struct with an id and id-less children refuses naming `struct l.element`, `list<list<int>>` with an id-less inner slot refuses naming `list l.element`, and an id-less list of structs, a map of structs and a nested list read their values on both routes; gates match the builder; red state exact). Rounds 1 to 3 are in their sections. The records commits `80f88e0f` and this one sit on `c9fd7dc4` with no code change. The three Opus notes are S-33 to S-35. |
| S-18 | Fallback route, by name (ruling 1), F1 and F2 shapes, both routes. A rename, a double rename, a move, reordered file children, a dropped child beside an added one, an extra file child then an added child, a list element's children and a two-deep struct with a rename each read the table's child by name, and a child with no history reads NULL, never a neighbour's value | PROVEN | Fallback pins, red at `bc6a2e67` unless noted: `idless_renamed_child_after_a_nested_sibling_reads_null_fallback` (green at `bc6a2e67` by design: round 1 already read NULL); `idless_two_renamed_children_never_read_a_neighbour_fallback`; `idless_moved_child_reads_by_name_fallback`; `idless_file_children_in_another_order_read_by_name_fallback`; `idless_dropped_child_never_lends_its_value_to_a_sibling_fallback`; `idless_extra_file_child_never_reads_under_an_added_child_fallback`; `idless_list_element_children_read_by_name_fallback`; `idless_struct_two_deep_reads_by_name_and_a_rename_reads_null_fallback`. Mapping-route no-regression pins, green at `bc6a2e67` and at head: `idless_renamed_child_after_a_nested_sibling_keeps_its_value_mapping`, `idless_two_renamed_children_keep_their_values_mapping`, `idless_moved_child_reads_by_name_mapping`, `idless_file_children_in_another_order_read_by_name_mapping`, `idless_dropped_child_never_lends_its_value_to_a_sibling_mapping`, `idless_extra_file_child_never_reads_under_an_added_child_mapping`, `idless_list_element_children_read_by_name_mapping`, `idless_struct_two_deep_reads_by_name_and_a_rename_keeps_its_value_mapping`. Source: `struct_idless_child_2.rs` |
| S-19 | Unmapped child (ruling 2, F3): on an id-less file, a child of a mapped struct that the mapping or the name leaves unstamped reads NULL per row, on both routes, and is never refused and never projected. Where the file has no embedded ids at all, an unmapped child is never a refusal | PROVEN | Red at `bc6a2e67`: `idless_partial_nested_mapping_reads_the_unmapped_child_as_null` (the F3 shape: mapping `s -> [d]`, file `s<d = 10, e = 20>`, answer `{d: 10, e: NULL}`); `idless_file_children_with_no_matching_name_read_null_mapping` and `idless_file_children_with_no_matching_name_read_null_fallback`. Renamed no-mapping pins, green at head: `idless_struct_without_child_mapping_reads_its_unmapped_child_as_null` and `idless_struct_without_child_mapping_and_an_added_child_reads_null` (`struct_idless_child.rs`) |
| S-20 | Top-level positional misread, OPEN and not fixed by this unit. On the fallback route, a table `(x, a, b)`, no drop, and an id-less file `(a, b)` with rows `a = [1, 2]`, `b = [40, 30]` read `x = [1, 2]` (a's values), `a = [40, 30]` (b's values) and `b = [NULL, NULL]`. The top-level rule (name and position, else `position + 1`) is unchanged by this unit and pre-existing at `14a93602`. The Muse MAX critic's out-of-scope note, identical on base and head: a top-level reorder on the fallback route fails with a `Struct -> Int32` cast error. Spark never reaches this shape because `add_files` installs a mapping; RePark's `add_files` installs one too, so the shape needs a table registered from foreign metadata with no mapping | OPEN | Measured once at `85687ce0` by the builder (scratch test, since deleted). Carded as F-IDLESS-TOPLEVEL-POSITIONAL-1, to be filed on the RePark side by the RP-64 records round, not fixed here. Fix round 3 (V3-F2) found positional values of the same kind in one more shape, an id-less file `(a, c)` under a mapping that names only `zz`, reading `a = 1`, `c = 2`; the builder records them as pre-existing at the base and not fixed by fix round 3 |
| S-21 | Ruling 7 (the Muse MAX critic's F2, folded): on an id-less file, a nested leaf whose file type cannot be promoted to the table type refuses `DataInvalid` naming the field path and both types, in both directions and on both routes, instead of collapsing the struct's validity to `[f, f]`. A top-level incompatible type keeps its old behaviour (the column is excluded and reads NULL) | PROVEN | Red at `bc6a2e67` (the scan read batches where it must refuse): `idless_nested_int_under_a_string_child_refuses_fallback` and `_mapping`; `idless_nested_string_under_an_int_child_refuses_fallback` and `_mapping`. Green at head. The refusal is raised in `get_arrow_projection_mask_with_field_ids` (`reader.rs`) |
| S-22 | A name mapping whose list or map entry lacks its `element`, `key` or `value` child now reads that list or map as NULL where round 1 read its values. Judged spec-consistent (an unmapped field is not projected) | REJECTED | Verifier round 2 V2-F3 (S2): the list or map reads NULL for present values, and ruling 3 of round 2 refuses instead, so the claim is withdrawn. S-25 (PROVEN) replaces it and closes this question by citation |
| S-23 | An incompatible nested type on a file WITH embedded field ids (table `s<d: string>`, file `s<d: int>` with embedded ids `1, 2, 3`, values `[1, NULL]`) still collapses the struct validity to `[false, false]` with a NULL `StringArray` child at `85687ce0`. The item-7 refusal is gated on `missing_field_ids`, so this case is outside this unit's scope and the reader behaves as at round 0. Fix round 3 did not measure this shape; the file carries an id on its top-level field, so the first-field gate and the any-depth gate read it the same way | OPEN | Measured at `85687ce0`; both round-2 verifiers confirm it is pre-existing at `14a93602` (Q4). A candidate card, a follow-up to the card of S-20 or to a new unit |
| S-24 | Fix round 2, ruling 1: on an id-less file, the NULL fill for a source struct whose children all lack ids applies only when every child carries the stamping pass's in-memory mark. A file that carries ids on some fields and none on a nested struct's children refuses `DataInvalid`, naming the struct path and that the data file must carry a field id on every nested field, on every shape: same names, swapped, extra child. Never a silent NULL for a present value, never a positional read | PROVEN | Fix round 2 (`9f928be2`). Red at `b16def80`: `top_level_ids_only_file_with_same_named_children_refuses`, `top_level_ids_only_file_with_swapped_children_refuses` and `top_level_ids_only_file_with_an_extra_child_refuses` (`struct_idless_child_3.rs`), and the unit `unstamped_children_of_a_file_with_ids_refuse_without_a_name_mapping_hint`. Green at both: the no-regression pins `idless_file_with_same_named_children_reads_its_values_fallback` and `_mapping`, `idless_file_with_swapped_children_reads_by_name_fallback` and `_mapping`, `idless_file_with_an_extra_child_ignores_it_fallback` and `_mapping`; the units `children_the_reader_left_unstamped_fill_null` and `mixed_ids_refusal_hint_asks_for_ids_and_never_for_a_name_mapping` (the V2-N1 hint pin). A file with ids at every depth is unchanged: `struct_empty_child` 13 green at head `e0bfed03`. Restated for the any-depth gate (fix round 3, ruling 1): the stamping pass, and so the NULL fill, now run only on a file with no field id at any depth; the three top-level-ids-only pins are unchanged and green at `c9fd7dc4` |
| S-25 | Fix round 2, ruling 3: a name mapping that names a list without an `element` entry carrying a field id, or a map without `key` and `value` entries carrying field ids, refuses `DataInvalid`, naming the list or map path and the missing entry. An entry without a field id counts as missing. A list or map the mapping does not name, and a list or map whose mapping carries its entries, is unchanged. Never a NULL column for present values, never a positional read | PROVEN | Fix round 2 (`2f0c16f6`), closes S-22 by citation. Red at `b16def80`: `mapping_naming_a_list_without_its_element_entry_refuses` (`l` reads `[null, null]`), `mapping_naming_a_map_without_its_key_and_value_entries_refuses` (`m` reads `[null, null]`) and `mapping_naming_a_map_with_its_key_entry_only_refuses` (`DataInvalid: Map field must have exactly 2 fields`), all in `struct_idless_child_3.rs`. Green at both: `fully_mapped_list_and_map_read_their_values` and `list_of_structs_mapping_naming_the_element_and_one_child_reads_null_for_the_rest`. Head `e0bfed03`. Amended by S-30 (fix round 3, ruling 3): the refusal fires only for a projected list or map |
| S-26 | Fallback route with no mapping, ruling-inherent: a child dropped and then re-added by name reads its old value. Table `(id, s<a, b>)`, id-less file `{a: 1, b: 2}`, `drop s.b` then `add s.b` reads `{a: 1, b: 2}`. The base did the same, and so does the top-level rule (`drop c`, `add c` reads `c = 5`). The mapping route reads NULL, which is correct | OPEN | Verifier round 2 V2-F1 (S1 under the round-2 rule, pre-existing, fallback route only). A by-name rule without schema history cannot tell a re-added field from the original, so this is a limitation of the no-mapping route and is not fixed here. It joins the top-level card, S-20 (F-IDLESS-TOPLEVEL-POSITIONAL-1) |
| S-27 | Fix round 2 also clears four base bugs that the round-2 verifier found as side effects of the fix: (1) a fabricated value for a REQUIRED child the file lacks; (2) rows lost by pruning on `c IS NULL`; (3) rows lost with row selection on a nested promoted filter; (4) rows lost by pruning on a partially stamped file | PROVEN | Opus 5.5 verifier round 2 measurement, copied and not re-measured by the clerk. The builder's notes list the four without re-measuring them, and no pin of this unit names them |
| S-28 | Fix round 3, ruling 1 (V3-F1): a file has ids when a `PARQUET:field_id` is present on ANY field at ANY depth (Java `HasIds`, `file_has_field_ids`), and only a file with no id anywhere takes the stamping pass. The stamping pass never overwrites an id a field already carries and never marks such a field. The V3-F1 file reads its struct children by their embedded ids on both routes, `{e: 10, d: 20}`; its id-less first column is not projected, so the table's `x` reads NULL, never a position | PROVEN | Red at `fc8ff8fe`: `a_first_field_without_an_id_keeps_the_embedded_nested_ids_mapping` and `..._fallback` in `struct_idless_child_4.rs` (both read the swapped `e = [20]`, `d = [10]`), and the unit `stamping_never_overwrites_an_embedded_nested_id` (`left: Some(4)`, `right: Some(3)`). Green at `c9fd7dc4`, with the unit `a_field_id_at_any_depth_means_the_file_has_ids` (it does not compile at `fc8ff8fe`, so it has no red run) |
| S-29 | Fix round 3, ruling 2 (V3-F2): no output field at any depth carries the in-memory mark `iceberg.reader.left-unstamped`. `stamp_mapped` returns an unmapped top-level field unchanged and unmarked, and the mark sits at depth 1 and below only; the fallback route never marked a top-level field. The output schema is clean on both routes for two unmapped columns, one unmapped column and an unmapped struct child, and the values are unchanged from `e0bfed03` | PROVEN | Red at `fc8ff8fe`: `the_reader_mark_never_reaches_the_output_with_two_unmapped_columns_mapping` (the mark reached `["a", "c"]`) and `..._one_unmapped_column_mapping` (`["c"]`). Controls green at `fc8ff8fe`: the two `_fallback` pins and `the_reader_mark_on_an_unmapped_struct_child_never_reaches_the_output` (a nested mark makes the source type differ from the target type, so the table's field is output). The values the pins assert were read at `fc8ff8fe`, a records commit on `e0bfed03` with no code change; a print run with the assertions swapped gave 5 passed, 0 failed |
| S-30 | Fix round 3, ruling 3 (V3-N1): the incomplete list or map mapping refusal of S-25 fires only for a projected list or map. Projected means the column, or an ancestor, is in the task's projected field ids. A list or map the task does not select, or has dropped, never refuses, and `select(["id"])` reads `id`. An unprojected list or map is stamped as far as its mapping reaches and is not read. Amends S-25 | PROVEN | Red at `fc8ff8fe`: `a_dropped_map_with_an_incomplete_mapping_never_refuses` and `an_unprojected_map_with_an_incomplete_mapping_reads_the_selected_column` (both refused on the missing `key` entry), and `an_unprojected_list_with_an_incomplete_mapping_reads_the_selected_column` (refused on the missing `element` entry). Control, green on both: `a_projected_map_with_an_incomplete_mapping_still_refuses` |
| S-31 | Fix round 3, ruling 4 (V3-N2), TAKEN: on a file that carries ids, a projected list or map whose element, key or value slot carries neither a field id nor the reader mark refuses `DataInvalid` naming the list or map path (`list 'l' ...`, `map 'm' ...`), the struct refusal's text with the kind swapped. The check runs at the start of `validity_leaves` (`row_filter_plan.rs`), because a list or map with no id-carrying slot never reaches the plan. An unprojected list never refuses, and a slot carrying the mark keeps its old answer. The clause covers the slots; the children of an element that carries an id are not walked (Open items, 5) | PROVEN | Red at `fc8ff8fe`: `a_list_of_ints_without_an_element_id_in_a_file_with_ids_refuses`, `a_list_of_structs_without_nested_ids_in_a_file_with_ids_refuses` and `a_map_without_key_and_value_ids_in_a_file_with_ids_refuses` (each read the column as NULL instead of refusing). Controls green on both: `an_unprojected_list_without_nested_ids_in_a_file_with_ids_never_refuses` and `an_idless_list_of_ints_still_reads_its_values_on_the_fallback_route` |
| S-32 | Fix round 2's gated fill refuses an Avro or ORC struct whose children all lack ids, where fix round 1 filled NULL (the fix round 2 mechanism, "a stricter, loud result"). The claim: this is a reachable regression on those formats | REJECTED | Opus 5.5 verifier round 3 measurement, copied and not re-measured by the clerk: the fork's writer always writes `field-id`, so a hand-built Avro file without it refuses `FeatureUnsupported` identically on head and base; ORC refuses every nested type. The claim is unreachable |
| S-33 | Round-4 note V4-N1: a file whose only field ids sit on nested leaves, with none at the top level, now refuses with arrow-rs's generic `Field id not found in metadata`, where `e0bfed03` and the base read the values and Java would read the id-less top-level fields as NULL. The refusal is loud and never a wrong value; a path-naming message is a follow-up | OPEN | Opus 5.5 round-4 measurement, copied |
| S-34 | Round-4 note V4-N2: a map value struct that carries an id but whose children carry none refuses with arrow-rs's `Map field must have exactly 2 fields`, with no path, because the walk descends only into children that carry an id and the projection then selects the key leaf alone. The list counterpart names its path. The refusal is loud; a path-naming message is a follow-up | OPEN | Opus 5.5 round-4 measurement, copied |
| S-35 | Round-4 note (c): projecting only a descendant of a list or map through a hand-built `FileScanTask` is always loud, on every route and even with complete ids, with `DataInvalid` or `Unexpected field not found`. A pre-existing limitation of the reader, not fixed here | OPEN | Opus 5.5 round-4 measurement, copied |

VERDICT (2026-11-01, round-4 closure): 25 PROVEN, 7 OPEN (S-12, S-20, S-23, S-26, S-33, S-34, S-35), 3 REJECTED (S-10, S-22, S-32), of 35 clauses. S-17 is PROVEN on the round-4 union of PASS at `c9fd7dc4`.

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

## Verifier round 2 (on `85687ce0`): FAIL from both verifiers

An Opus 5.5 re-verifier and a Muse MAX critic each read fix round 1 on
`85687ce0`, and both returned FAIL. The orchestrator reproduced Q1 and V2-F3 at
`85687ce0`. The findings below are copied as the orchestrator recorded them for
fix round 2.

Opus 5.5 round-2 verifier:

- **V2-F2 (S1, the same shape as the Muse critic's Q1).** A Parquet file that
  carries `PARQUET:field_id` on its top-level fields (`id = 1`, `s = 2`) and none
  on the nested children, table `s<d, e>`, one row `(1, {d: 10, e: 20})`: the head
  reads `validity [true]`, `d = [None]`, `e = [None]`, a silent NULL for two present
  values; the base `bc6a2e67` refused `DataInvalid: struct 's' has no field ids and
  children differ`. The critic measured three shapes: same-named children (`d, e`),
  swapped (`s<b = 10, a = 20>` under `s<a, b>`), and an extra child (`s<d = 1, x = 99>`
  under `s<d>`; the base read `d = 1`). Cause: the NULL fill of ruling 2 in
  `nested_projection::build_struct` is universal, and on a file that has ids an
  unstamped nested source struct means a partially stamped file, not an unmapped one.
- **V2-F3 (S2, mapping route).** A name mapping that names a list but has no
  `element` entry, or a map with no `key` and `value` entries: `l: list<int>` reads
  `[NULL, NULL]` with validity `[f, f]` where the base and round 1 read
  `[[1, 2], [3]]`; a map whose mapping has no fields reads NULL; a map mapping that
  names `key` only fails with `DataInvalid: Map field must have exactly 2 fields`.
  Cause: `unmapped_group_leaf_indices` was deleted, so an unmapped element or
  key/value leaf is no longer decoded and the whole column collapses.
- **V2-F1 (S1 under the round-2 rule, ruling-inherent, pre-existing, fallback route
  only).** Table `(id, s<a, b>)`, id-less file `{a: 1, b: 2}`, `drop s.b` then
  `add s.b`: the fallback route reads `{a: 1, b: 2}`, because the re-added field takes
  the old value by name. The base did the same, and so does the top-level rule
  (`drop c`, `add c` reads `c = 5`). The mapping route reads NULL, correctly.
- **V2-N1 (S3).** The mixed-ids refusal in `build_struct` never fires end to end: a
  file with ids on `s.d` only reads `{d: 10, e: NULL}` on head and base, which is
  spec-consistent. No pin asserts the new hint text.
- **V2-N2.** An orchestrator scratch file in the tree. No finding.

Muse MAX critic, round 2:

- **Q1 (S1)** is V2-F2: the three shapes, reproduced twice by the critic.
- **Q4** is an incompatible nested type on a file with embedded ids, pre-existing at
  `14a93602` and confirmed (S-23).
- **Q5** is V2-F3. The critic judged the NULL answer spec-consistent; ruling 3
  overrules that judgement.

Four base bugs that the fix also clears, measured by the Opus 5.5 verifier and
copied (S-27), not re-measured here: a fabricated value for a REQUIRED child the
file lacks; rows lost by pruning on `c IS NULL`; rows lost with row selection on a
nested promoted filter; rows lost by pruning on a partially stamped file.

Rulings for fix round 2:

1. **Gate the fill.** The NULL fill for a source struct whose children all lack ids
   applies only when the file carried no field ids at all. A file that carries ids
   on some fields and none on a nested struct's children refuses `DataInvalid`,
   naming the struct path, on every shape. Never a silent NULL for a present value,
   never a positional read. The flag travels from the reader inside the stamped
   schema, so `reader.rs` stays at or under 9433 lines and no ceiling is raised.
2. **Pins red first at `b16def80`**, in a sibling file `struct_idless_child_3.rs`,
   because adding them to `struct_idless_child_2.rs` (766 lines) would have taken it
   past 1000. The three top-level-ids-only shapes refuse. The same three shapes on a
   file with no ids keep their round-1 answers. A file with ids at every depth is
   unchanged (`struct_empty_child.rs` green).
3. **A list or map named by the mapping without its `element`, or `key` and `value`,
   entries refuses**: `DataInvalid`, naming the list or map path and the missing
   entry; never a NULL column for present values and never a positional read. A list
   or map whose mapping carries the entries is unchanged, and so is a list of structs
   whose mapping names the element but not all of its children (round-1 ruling 2).

V2-F1 is not fixed. It is recorded as S-26, the limitation of the no-mapping route.

## Fix round 2 (`e0bfed03`, on `b16def80`)

Commits:

- `9f928be2` fix(arrow): fix 2 the NULL fill for an unstamped source struct applies only to a file without field ids (Q1)
- `2f0c16f6` fix(arrow): fix 2 a list or map the mapping names without its element, key or value entry refuses (V2-F3)
- `e0bfed03` test(arrow): fix 2 pins: a file with top-level ids only refuses, an id-less file keeps its answers, an incomplete list or map mapping refuses, the hint text

### Mechanism (ruling 1, `9f928be2`)

- The stamping pass, which runs only on a file without field ids, on both routes,
  tags each field it returns without an id with the in-memory Arrow metadata key
  `iceberg.reader.left-unstamped` (const `LEFT_UNSTAMPED_META_KEY` in
  `nested_projection.rs`). On the mapping route the tag is set by the early return of
  `stamp_mapped`; on the fallback route, by the two `None` arms of
  `FallbackIds::stamp_children`. The subtree under a tagged field is not tagged, and
  the plan never descends into it.
- `build_struct` fills NULL or `initial_default` only when every child of the source
  struct carries the tag. Otherwise it refuses: `DataInvalid => struct 's' has no
  field ids on its nested fields in a data file that carries field ids; the data file
  must carry a field id on every nested field`. A struct with no source children
  still fills, so `struct_empty_child.rs` is unchanged.
- The refusal is the default for any reader that runs no stamping pass: Parquet with
  embedded ids, and also Avro and ORC. An Avro or ORC struct whose children all lack
  ids now refuses where fix round 1 filled NULL. The orchestrator accepted this
  stricter, loud result.
- The tag never reaches a file. It survives `ArrowReaderMetadata::try_new` and the
  INT96 visitor, which copies field metadata, and it never reaches the output,
  because target fields come from the table schema and a tagged source never equals a
  target type.
- The mixed stamped and unstamped refusal and its hint are unchanged.
- Corrected by fix round 3 (V3-F2, S-29): the claim above that the tag never reaches the output is false on the mapping route.
- Why a tag and not a parameter: `reader.rs` was at its ceiling (9433),
  `record_batch_transformer.rs` at 999 of 1000, and `generate_batch_transform` already
  takes seven parameters, the clippy limit. The Avro, ORC and Parquet `_pos` paths
  share `build_scan_task_transformer`, so a bool would have needed lines in both full
  files. `reader.rs` and `record_batch_transformer.rs` are untouched, and no ceiling
  moved.

### Mechanism (ruling 3, `2f0c16f6`)

- `stamp_mapped` carries the field path. When a mapped list's `element` slot, or a
  mapped map's `key` or `value` slot, has no mapping entry with a field id, the scan
  refuses:
  - `DataInvalid => the name mapping names list 'l' but has no 'element' entry with a field id for it`
  - `DataInvalid => the name mapping names map 'm' but has no 'key' entry with a field id for it` (or `'value'`)
- The first missing slot is the one reported, so a map with no entries names `'key'`.
  An entry without a field id counts as missing.
- A list or map the mapping does not name keeps its old behaviour, and so does one
  whose mapping carries its entries.

### Pins (`e0bfed03`)

Red run at `b16def80`, `cargo test -p iceberg --test struct_idless_child_3`, exit
101: 8 passed, 6 failed. What the red pins read at `b16def80`:

| Pin | Read or outcome at `b16def80` |
|---|---|
| `top_level_ids_only_file_with_same_named_children_refuses` | struct valid, `d = null`, `e = null` |
| `top_level_ids_only_file_with_swapped_children_refuses` | struct valid, `a = null`, `b = null` |
| `top_level_ids_only_file_with_an_extra_child_refuses` | struct valid, `d = null` |
| `mapping_naming_a_list_without_its_element_entry_refuses` | `l` = `[null, null]` |
| `mapping_naming_a_map_without_its_key_and_value_entries_refuses` | `m` = `[null, null]` |
| `mapping_naming_a_map_with_its_key_entry_only_refuses` | `DataInvalid => Map field must have exactly 2 fields` |

Green at both `b16def80` and head, as no-regression pins (8): `fully_mapped_list_and_map_read_their_values`,
`idless_file_with_same_named_children_reads_its_values_fallback` and `_mapping`,
`idless_file_with_swapped_children_reads_by_name_fallback` and `_mapping`,
`idless_file_with_an_extra_child_ignores_it_fallback` and `_mapping`, and
`list_of_structs_mapping_naming_the_element_and_one_child_reads_null_for_the_rest`.

Unit pins, in `nested_projection_evo_pin_tests.rs`. Red run at `b16def80`,
`cargo test -p iceberg --lib -- mixed_ids_refusal_hint_asks_for_ids unstamped_children_of_a_file_with_ids children_the_reader_left_unstamped`,
exit 101: 2 passed, 1 failed. The failure is
`unstamped_children_of_a_file_with_ids_refuse_without_a_name_mapping_hint`.
`mixed_ids_refusal_hint_asks_for_ids_and_never_for_a_name_mapping` (V2-N1, the hint,
which fix round 1 had already changed) and `children_the_reader_left_unstamped_fill_null`
(the fill on a tagged child, which goes through `add_fallback_field_ids_to_arrow_schema`
and needs no new name) are green by design.

Green run at `e0bfed03`: `struct_idless_child_3` 14 ok, and the three unit pins ok.

### Ledger clauses changed by fix round 2

- **S-2** gains the list and map refusal of S-25; still PROVEN.
- **S-15** recounted at `e0bfed03`; still PROVEN.
- **S-17** stays OPEN. The round-2 verdicts are recorded in this ledger; the
  round-3 verdicts on `e0bfed03` are not yet.
- **S-22** REJECTED. Ruling 3 refuses the NULL column the clause described, and S-25
  closes the question by citation.
- **S-23** gains the note that both round-2 verifiers confirm Q4 as pre-existing.
  Still OPEN.
- **S-24** to **S-27** new: the gated fill and the partially stamped refusal
  (PROVEN), the incomplete list or map mapping refusal (PROVEN), the re-added child
  on the no-mapping route (OPEN), and the four cleared base bugs (PROVEN, copied).

## Verifier round 3 (on `e0bfed03`): FAIL (Opus 5.5), PASS with notes (Muse MAX)

The orchestrator reproduced V3-F1 and V3-F2 at `e0bfed03`. The findings are
copied from the round-3 fix brief. The last sentence of V3-F2 is adjusted from
the builder's address to the ledger's reading.

- **V3-F1 (S1, present since the round-0 nested stamping, unusual file shape).**
  The "file has ids" gate at `reader.rs` line 478 region looks at the FIRST
  top-level field only. Java `HasIds` treats a field id at ANY depth as "the file
  has ids". File: `x` (no id), `s` (id 2) `<d id 3 = 10, e id 4 = 20>`; table
  `s<e = 3, d = 4>` (the two children renamed into each other's names): head
  reads `{e: 20, d: 10}`, silently swapped, on both routes; the base read
  `{e: 10, d: 20}`. Cause: the by-name stamping pass runs on this file and
  overwrites its embedded nested ids.
- **V3-F2 (S2, introduced by `9f928be2`).** The mark `iceberg.reader.left-unstamped`
  reaches the OUTPUT batch schema: mapping route, table `(a = 1, c = 2)`, id-less
  file `(a, c)`, mapping naming only `zz`: the unmapped top-level fields are
  marked, no projected leaf has an id, the mask falls to `ProjectionMask::all()`,
  `compare_schemas` returns Equivalent and `PassThrough` hands back the source
  schema with the mark. A one-column table with a one-column file leaks it too.
  Nested marks cannot leak (nested types compare strictly). The fork's Parquet
  writer does not persist the mark, a plain arrow-rs `ArrowWriter` does.
  Separately, the positional values in that shape (`a = 1, c = 2` although the
  mapping names neither) are pre-existing at the base and belong to the top-level
  card, not this unit (S-20).
- **V3-N1 (S3).** The V2-F3 refusal fires for a list or map the scan never reads:
  `select(["id"])` with a mapping naming `m` without `key`/`value` refuses, both
  when the table has `m` and when it dropped `m`; the base read `id`.
- **V3-N2 (S2, pre-existing at the base).** On a file with top-level ids only,
  `list<struct<a, b>>` and `list<int>` read the whole list NULL (validity
  `[f, f]`) on head and base alike; `validity_leaves` borrows a leaf only for
  leafless structs; the Q1 refusal covers struct children only.

The Opus 5.5 verifier also measured the Avro and ORC consequence of fix round 2
as unreachable (S-32), and held Q1 and V2-F3 on every shape.

The Muse MAX round-3 verdict on `e0bfed03` is PASS (49 scratch probes; every Q1
and V2-F3 refusal with and without a mapping; id-less values unchanged; no tag
in its output shapes nor in a table rewritten from a scanned batch; the stamping
pass bypassed for a file with top-level ids; Avro and ORC unchanged against
`b16def80`; red state exactly the refusal assertions; gates matching the
builder). Its out-of-scope notes, reported and not re-measured by the clerk: (a)
a file whose first field lacks an id but whose other top-level field carries an
embedded id read silent NULL at `b16def80` and now refuses loud (the Q1
direction); (b) list and map files with top-level ids only and an incomplete
mapping read NULL on head and `b16def80` alike (pre-existing, the V3-N2 family,
taken by S-31).

Rulings for fix round 3 (orchestrator, round-3 fix brief):

1. The gate follows Java `HasIds`: a file has ids when a field id is present on
   any field at any depth. Only a file with no id anywhere takes the stamping
   pass. The stamping pass never overwrites an id a field already carries (a
   unit pin).
2. The mark never reaches the output: unmapped top-level fields stay unmarked on
   both routes, and no output field at any depth carries the mark (a recursive
   pin).
3. The incomplete-mapping refusal fires only for a projected list or map.
4. V3-N2: extend the Q1 refusal to lists and maps if it stays small, else record
   it OPEN with the top-level card. Taken (S-31).
5. Pins red first at `fc8ff8fe` in the sibling `struct_idless_child_4.rs`
   (`struct_idless_child_3.rs` is 604 lines; the sixteen pins with their helpers
   would pass 1000).
6. Gates as listed in the Gates section.

## Fix round 3 (`c9fd7dc4`, on `fc8ff8fe`)

Commits (each ends with exactly one `Authored-By: Claude (claude-opus-5-5) <noreply@anthropic.com>`):

- `72aaa7c5` fix(arrow): fix 3 a file has ids when any field at any depth carries one, and stamping never overwrites an embedded id (V3-F1)
- `e7fb600d` fix(arrow): fix 3 the reader mark stays off unmapped top-level fields and never reaches the output schema (V3-F2)
- `7db298ed` fix(arrow): fix 3 the incomplete-mapping refusal fires only for a projected list or map (V3-N1)
- `d856d06e` fix(arrow): fix 3 a list or map without nested ids in a file with ids refuses (V3-N2)
- `c9fd7dc4` test(arrow): fix 3 pins: the any-depth id gate, the mark, the unprojected column, the list and map refusal

Each code commit was checked with `cargo check -p iceberg --all-targets` and the
comment gate (0 added comment lines) before it was made.

### Mechanism

- **Ruling 1, V3-F1 (`72aaa7c5`).** `file_has_field_ids` in `field_id_stamping.rs`
  walks every field at every depth (struct children, list element, map key and
  value) with an explicit stack, and returns true on the first `PARQUET:field_id`.
  This is Java `HasIds`. `reader.rs` calls it in one line,
  `let missing_field_ids = !file_has_field_ids(arrow_metadata.schema());`, and the
  first-field check and its two `debug_assert!`s are gone. `with_field_id` keeps
  an id the field already carries (`entry().or_insert_with`), and `left_unstamped`
  returns a field that carries an id unchanged, with no mark. Both routes share
  those helpers. The V3-F1 file now takes the ids branch: it reads `{e: 10, d: 20}`
  on both routes, and its id-less first column is not projected, so the table's
  `x` reads NULL, never a position.
- **Ruling 2, V3-F2 (`e7fb600d`).** `stamp_mapped` returns an unmapped depth-0
  field unchanged; only depth 1 and deeper get the mark. `build_struct` reads the
  marks on children only, so the NULL fill is unchanged. The fallback route never
  marked a top-level field. No strip was needed: the recursive output-walk pin is
  green on both routes. A nested mark makes the source type differ from the
  target type, so the output field comes from the table schema.
- **Ruling 3, V3-N1 (`7db298ed`).** `apply_name_mapping_to_arrow_schema` takes a
  third argument, the task's `project_field_ids()`. `stamp_mapped` carries
  `projected` (ancestor-or-self is projected) down the tree. The element, key and
  value refusal fires only when `projected` is true. The scan builder projects
  top-level columns only, so ancestor-or-self is the whole rule. An unprojected
  list or map is stamped as far as the mapping reaches; its missing slot is marked
  like any unmapped nested child, and it is not read. The `reader.rs` call gained
  one argument line. The `reader.rs` ceiling in `scripts/check_rust_file_size.py`
  was lowered from 9433 to 9429 in this commit, the file's final size.
- **Ruling 4, V3-N2 (`d856d06e`), TAKEN.** The struct check lives in the plan
  (`build_struct`). A list or map whose slots carry no ids never reaches the plan,
  because the projection selects no leaf of it and the transformer fills the column
  NULL. So the check sits in `validity_leaves` (`row_filter_plan.rs`), which the
  projection builder already calls with the projected ids.
  `refuse_collections_without_nested_ids` walks each projected top-level field down
  through the fields that carry ids. At a list (element) or map (key, value) node,
  a slot with neither an id nor the reader mark refuses `DataInvalid` with the text
  `list 'l' has no field ids on its nested fields in a data file that carries field
  ids; the data file must carry a field id on every nested field` (`map 'm' ...`
  for a map): the struct refusal's text with the kind swapped. About 50 lines in
  `row_filter_plan.rs`; no `reader.rs` change. A slot carrying the mark keeps its
  old answer, and an unprojected list never refuses.
- **Correction to fix round 2.** The fix round 2 mechanism's claim that the tag
  never reaches the output was false on the mapping route. V3-F2 measured it, and
  `e7fb600d` with S-29 supersedes it.

Builder deviations, recorded as the builder recorded them: one `cargo fmt --all`
ran outside the flock and `systemd-run` wrapper, right after the ruling-1 edit;
every later cargo command ran under the wrapper. At `72aaa7c5` and `e7fb600d` the
`reader.rs` size was 9428 under the then-ceiling 9433, so the size ratchet reports
a shrink on those two intermediate commits only; the ceiling was lowered once, in
`7db298ed`.

### Red run at `fc8ff8fe` (pins present, code at `fc8ff8fe`)

`cargo test -p iceberg --test struct_idless_child_4`, exit 101:

```
test a_dropped_map_with_an_incomplete_mapping_never_refuses ... FAILED
test an_idless_list_of_ints_still_reads_its_values_on_the_fallback_route ... ok
test a_projected_map_with_an_incomplete_mapping_still_refuses ... ok
test a_first_field_without_an_id_keeps_the_embedded_nested_ids_mapping ... FAILED
test a_map_without_key_and_value_ids_in_a_file_with_ids_refuses ... FAILED
test a_first_field_without_an_id_keeps_the_embedded_nested_ids_fallback ... FAILED
test a_list_of_ints_without_an_element_id_in_a_file_with_ids_refuses ... FAILED
test a_list_of_structs_without_nested_ids_in_a_file_with_ids_refuses ... FAILED
test an_unprojected_list_without_nested_ids_in_a_file_with_ids_never_refuses ... ok
test the_reader_mark_never_reaches_the_output_with_two_unmapped_columns_fallback ... ok
test the_reader_mark_never_reaches_the_output_with_one_unmapped_column_fallback ... ok
test an_unprojected_map_with_an_incomplete_mapping_reads_the_selected_column ... FAILED
test an_unprojected_list_with_an_incomplete_mapping_reads_the_selected_column ... FAILED
test the_reader_mark_on_an_unmapped_struct_child_never_reaches_the_output ... ok
test the_reader_mark_never_reaches_the_output_with_two_unmapped_columns_mapping ... FAILED
test the_reader_mark_never_reaches_the_output_with_one_unmapped_column_mapping ... FAILED
test result: FAILED. 6 passed; 10 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s
```

What the red cases read at `fc8ff8fe` (builder's table):

- first field without an id, both routes: `e = [20]`, `d = [10]` (swapped);
- mark, two unmapped columns, mapping: the reader mark reached the output at `["a", "c"]`; one unmapped column, mapping: at `["c"]`;
- unprojected map, `select(["id"])`, and the dropped map: the `key` refusal;
- unprojected list, `select(["id"])`: the `element` refusal;
- list of ints, list of structs and map, top-level ids only: read NULL (`l` = `[null, null]`, `m` NULL).

Controls green at `fc8ff8fe` by design: the two fallback mark pins (the fallback
route never marked a top-level field), the nested-mark pin, the projected-map
refusal, the unprojected list in a file with ids, and the id-less list values.

Unit red runs: `cargo test -p iceberg --lib -- stamping_never_overwrites_an_embedded_nested_id`, exit 101, `left: Some(4)`, `right: Some(3)`. `a_field_id_at_any_depth_means_the_file_has_ids` does not compile at `fc8ff8fe` (`file_has_field_ids` does not exist there), so it has no red run.

Values control for the mark pins: the arrow source was set to `fc8ff8fe` and only
`assert_no_reader_mark` was swapped for a print. `-- the_reader_mark` then gave 5
passed and 0 failed, so the asserted values are those of `fc8ff8fe`, a records
commit on `e0bfed03` with no code change. The source and the test were restored
with `git checkout HEAD`.

### Green run at `c9fd7dc4`

`struct_idless_child_4`: all 16 `ok`. `field_id_stamping_tests`: 12 `ok`, including
both new unit pins.

### Ledger clauses changed by fix round 3

- **S-14** ceiling: `reader.rs` at 9429 at head; the legacy ceiling lowered, never raised.
- **S-15** recounted at `c9fd7dc4`; still PROVEN.
- **S-17** stays OPEN (the round-4 verdicts are not yet recorded).
- **S-20** gains the V3-F2 positional values as a second shape of the same top-level card; still OPEN.
- **S-23** not measured at round 3. The file carries an id on its top-level field, so the first-field gate and the any-depth gate read it the same way; still OPEN.
- **S-24** restated for the any-depth gate; still PROVEN.
- **S-25** amended by S-30 (the refusal fires only for a projected list or map); still PROVEN.
- **S-28** to **S-32** new (below).

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

Fix round 2 builder runs, at `e0bfed03`. The clerk ran no cargo; the rows are
copied from the builder's notes, and every cargo command ran under the flock,
environment and `systemd-run` wrapper of the brief:

| Command | Exit | Counts |
|---|---|---|
| `cargo test -p iceberg --test struct_idless_child --test struct_idless_child_2 --test struct_idless_child_3 --test struct_empty_child` | 0 | 18 + 29 + 14 + 13 passed, 0 failed |
| `cargo test -p iceberg --no-fail-fast` | 0 | 74 result lines, 5092 passed, 0 failed, 21 ignored |
| `cargo test -p iceberg-datafusion --no-fail-fast` | 0 | 43 result lines, 802 passed, 0 failed, 15 ignored |
| `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | none |
| `make check-rust-file-size` | 0 | 710 files clean (89 legacy ceilings); `reader.rs` 9433, `record_batch_transformer.rs` 999 |
| `BASE_REF=14a93602 make check-comment-blocks` | 0 | no added block over 6 lines |
| `make check-agent-artifacts` | 0 | clean (11 + 4 needles) |

Comment gate per commit (added `//` or `/*` lines in the `.rs` diff): `9f928be2` 0,
`2f0c16f6` 0, `e0bfed03` 16 (the licence header of the new file).

Records gates for this commit (clerk), every command exit 0: `BASE_REF=14a93602 make check-comment-blocks` (no added block over 6 lines); `make check-agent-artifacts` (11 + 4 needles clean); `make check-matrix-anchors` (88 GAP_MATRIX rows anchored, IDs unique). The staged diff touches only markdown.

Fix round 3 builder runs, at `c9fd7dc4`. The clerk ran no cargo; the rows are
copied from the builder's notes, and every cargo command ran under the flock,
environment and `systemd-run` wrapper of the brief:

| Command | Exit | Counts |
| --- | --- | --- |
| `cargo test -p iceberg --test struct_idless_child --test struct_idless_child_2 --test struct_idless_child_3 --test struct_idless_child_4 --test struct_empty_child` | 0 | 18 + 29 + 14 + 16 + 13 passed, 0 failed |
| `cargo test -p iceberg --no-fail-fast` | 0 | 75 result lines, 5110 passed, 0 failed, 21 ignored |
| `cargo test -p iceberg-datafusion --no-fail-fast` | 0 | 43 result lines, 802 passed, 0 failed, 15 ignored |
| `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | none |
| `make check-rust-file-size` | 0 | 711 files clean (89 legacy ceilings); `reader.rs` ceiling lowered 9433 to 9429 |
| `BASE_REF=14a93602 make check-comment-blocks` | 0 | no added block over 6 lines |
| `make check-agent-artifacts` | 0 | clean (11 + 4 needles) |

Comment gate per commit (added `//` or `/*` lines in the `.rs` diff): `72aaa7c5` 0,
`e7fb600d` 0, `7db298ed` 0, `d856d06e` 0, `c9fd7dc4` 16 (the licence header of the
new file).

Line counts at head: `reader.rs` 9429; `field_id_stamping.rs` 377;
`field_id_stamping_tests.rs` 424; `row_filter_plan.rs` 520;
`record_batch_transformer.rs` 999, untouched; `struct_idless_child_4.rs` 627 (new);
`struct_idless_child_3.rs` 604.

Records gates for this commit (clerk), every command exit 0:
`BASE_REF=14a93602 make check-comment-blocks`; `make check-agent-artifacts`;
`make check-matrix-anchors`. The staged diff touches only markdown.

## Verification

PROVEN (S-17, 2026-11-01). Round 1 on `8a66aa7c` is Opus 5.5 FAIL and Muse MAX
PASS with findings, and every finding is fixed in fix round 1. Round 2 on
`85687ce0` is FAIL from both verifiers, and every finding is fixed in fix
round 2 except V2-F1 (OPEN, S-26) and Q4 (pre-existing, S-23). Round 3 on
`e0bfed03` is Opus 5.5 FAIL (V3-F1, V3-F2, V3-N1, V3-N2) and Muse MAX PASS
with notes, and all four findings are fixed in fix round 3 (S-28 to S-31).
Round 4 on `c9fd7dc4` is an Opus 5.5 re-verifier PASS and a Muse MAX critic
PASS with zero findings, and its three Opus notes are recorded as OPEN clauses
S-33 to S-35.

## Open items

1. S-12: check the Java note against the Java source, if the Java note matters for
   the closure.
2. S-17 is closed (2026-11-01): PROVEN on the round-4 union of PASS at
   `c9fd7dc4`, so the parent ledger's closures of S-11 and S-16 stand.
3. S-20 and S-26: F-IDLESS-TOPLEVEL-POSITIONAL-1, and the re-added child on the
   no-mapping route, both to be filed on the RePark side by the RP-64 records
   round. The V3-F2 positional values join S-20.
4. S-23: the embedded-ids item-7 collapse. A candidate card or a unit of its own.
5. S-31 scope: the walk checks a list's element and a map's key and value slots,
   and descends only through children that carry an id. An element struct that
   carries an id, with a child that has no id, is not walked by the refusal; it
   reaches the round-2 struct refusal in `build_struct` only if the plan reads that
   struct, which the clerk did not measure. No pin name in this round covers that
   shape. Ruling 4 named the children too. The walk is also not gated on
   `missing_field_ids`; the builder's reading is that the stamping pass gives every
   slot of an id-less file an id or the mark, which the clerk did not verify. The
   round-4 verifiers should measure both.
