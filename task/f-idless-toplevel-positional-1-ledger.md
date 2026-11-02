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

# F-IDLESS-TOPLEVEL-POSITIONAL-1 — a top-level column of an id-less Parquet file reads by position on the no-mapping route, refuses a type or nesting-kind change, and reads NULL when unmapped

**Date:** 2026-11-02 (the work spans 2026-11-01 and 2026-11-02). **Base:** fork `main` `56ff5545` (the `#380` squash). **Branch:** `fix/idless-toplevel-positional-1`, not pushed. Commits, by round: round 1 `1d9decdd` (pins) and `414f544d` (fix); round 2 `e8568134` (re-pins), `90d304e0` (pins) and `9facc9a7` (fix); round 3 `c1b4c54b` (pins) and `91a72650` (fix). The records commit that adds this text follows `91a72650`.

**Model:** builder Claude Opus 5.5 (`claude-opus-5-5`), verified from the transcript, rounds 1 to 3. Orchestrator `claude-fable-5-1`. Records clerk Claude Haiku 5.5 (`claude-haiku-5-5`). The Opus 5.5 verifier and the Muse MAX critic have not run on `91a72650` at the time of this text.

**Outcome in one line.** On the no-mapping route of an id-less Parquet file, a top-level column reads by position (`position + 1`), as Java's fallback stamping does. A column whose file type or nesting kind would change under its positional table field refuses, on projected, predicate and equality-delete key columns. An unmapped top-level column under a partial mapping reads NULL. At `91a72650` the 20 pins of `idless_toplevel_positional.rs` and the 6 of `idless_toplevel_positional_2.rs` are green.

## Why

**The fault.** On the no-mapping route, a top-level file column matched a table column by name and position, and otherwise took `position + 1`. A positional read into another type was cast silently, to NULL or to a wrong value. An unmapped top-level column under a partial mapping was read by position. Java's `ParquetSchemaUtil.addFallbackIds` assigns `position + 1` only, and Spark raises on a type mismatch. The RePark card is F-IDLESS-TOPLEVEL-POSITIONAL-1. The top-level shapes of the remap ledger's S-20 and the top-level half of its S-26 are carried here.

**The ruling.** The owner ruled on 2026-11-01 that this residue blocks RePark's format-v2 label under section 1 of its order, and chartered this unit for the top level only. The nested half of the remap ledger's S-26 is a ruled divergence and stands.

**Spark measurements.** Spark 4.1.2 with Iceberg 1.11.0, copied from the orchestrator's notes and not re-measured by the clerk. The fork's answer before the fix is in brackets.

- `(a INT, b STRING)` over `(b STRING, a INT)` [`[(None, '1'), (None, '2')]`]: Spark raises.
- `(a INT, b BIGINT)` over `(b, a)` [`[(None, 1), (None, 2)]`]: Spark raises on `SELECT a, b` and reads `[(1,), (2,)]` on `SELECT b`.
- `(a INT, b DATE)` [a date read as an int]: Spark's answer is not recorded in the notes.
- `drop c`, then `add c` [the file's `c = 5` read]: Spark `[(1, None)]`.
- A mapping naming only `zz` over file `(a, c)` [`(1, 2)`]: Spark `(None, None)`.
- After `DROP COLUMN x` on table `(x, a, b)` over file `(a, b)`: Spark's projection is positional, and every filtered read returns zero rows. The fork answers coherently. This is a kept divergence (S-14).

**The fix, three files.** `field_id_stamping.rs` stamps a top-level file column `position + 1` always, and leaves unstamped a position whose id only a nested table field holds. `guarded_field_ids` gathers the projected, predicate and equality-delete key ids, and `refuse_positional_type_change` refuses a type or nesting-kind change on them. `reader.rs` passes the guarded ids in place of the projected ids. `record_batch_transformer.rs` `compare_schemas` compares field ids exactly, so a None against a Some is a difference, and an unmapped column reads NULL.

## Clauses

| Clause | Statement | Verdict | Evidence |
|---|---|---|---|
| S-1 | The fault reproduces red on `56ff5545`: the no-mapping route matches a top-level column by name and position, else takes `position + 1`; a positional read into another type is cast silently (NULL or a wrong value); a partial mapping reads an unmapped top-level column by position | PROVEN | Red run on `56ff5545` for the round-1 set of 13 pins: 7 red, 6 green, verbatim in the builder's notes (step 1). The Spark side is the orchestrator's and the RP-64 verifier's measurement, copied; the clerk did not re-run Spark |
| S-2 | A top-level file column takes id `position + 1` whatever its name, as Java's `ParquetSchemaUtil.addFallbackIds` does on the no-mapping route, so a same-type positional read is by position | PROVEN | Unit `fallback_stamps_position_plus_one_where_name_and_position_match_another_id` (red on the base: `left: Some(5)`, `right: Some(1)`; green at head). Control `a_same_type_swap_reads_by_position_control` (green on both). The Java source was not read in this unit, so the attribution to Java is the builder's reading; the PROVEN verdict is for the fork's behaviour, pinned |
| S-3 | A top-level position whose id is held only by a nested table field gets no id, and that column is never read: table `(x INT, s STRUCT<a, b>)` over file `(x, s, extra)` reads `s` as `Row(a=None, b=None)`, with an INT or a STRING `extra` | PROVEN | Pins `an_int_column_whose_position_is_a_nested_id_is_never_read` and `a_string_column_whose_position_is_a_nested_id_is_never_read` (green at `414f544d` and on the base: no regression). Spark's answer, copied: `[(1, Row(a=None, b=None)), (2, Row(a=None, b=None))]`; `SELECT x, s.a` gives `(1, None), (2, None)`; `WHERE s.a = 77` returns no rows; with a STRING `extra` the same, with no error |
| S-4 | Type guard on projected columns: a top-level column whose file type is neither equal nor an allowed promotion to its positional table field's type refuses with `DataInvalid`, naming the file column, its position and type, the table field with its id and type, and the hint that the file has no field ids and the table no name mapping. Timestamp-family pairs and unconvertible types are not guarded (S-20) | PROVEN | Pins `a_positional_long_column_landing_on_an_int_field_refuses`, `a_positional_string_column_landing_on_an_int_field_refuses`, `a_positional_date_column_landing_on_an_int_field_refuses` and `a_positional_read_that_casts_into_a_wrong_value_refuses` (red on the base, green at head); control `a_positional_int_column_landing_on_a_long_field_reads_promoted` (green on both). Spark raises on the string and long shapes (copied). The date shape's Spark answer is not recorded, so no Spark parity is claimed for it |
| S-5 | Type guard on predicate columns: the same guard covers the columns the scan's predicate references, so a filter on a mismatched column refuses where the projection alone would not | PROVEN | Pins `a_filtered_projection_over_a_type_changing_positional_read_refuses` (names `'a'` and `id 1`) and `a_count_filtered_on_a_type_changing_positional_column_refuses` (both filters). Red at `414f544d`: the scan of `[]` where `a = 1` read 0 rows, verbatim. Spark raises on `SELECT b ... WHERE a = 1`, `count(*) ... WHERE a = 1`, `count(*) ... WHERE b = 'p'` and `SELECT a ... WHERE b = 'p'` (orchestrator, 2026-11-02, copied). Controls green at head: `an_unfiltered_count_over_a_type_changing_positional_read_reads_every_row` (2, the Spark answer for `count(*)` with no filter) and `a_filter_on_a_same_type_positional_column_reads_control` (1 and `"30"`, the Spark control on `(a INT, b INT)`) |
| S-6 | The guard covers the equality-delete key ids of the task's delete files, so a positional key column that changes type refuses | PROVEN | Pin `an_equality_delete_key_fed_by_a_type_changing_positional_column_refuses`: red at `9facc9a7` (the scan read `["", ""]`), green at `91a72650`. The ids come from `FileScanTask.deletes[*].equality_ids` where the guarded set is built. Spark's answer is not measured: S-7 |
| S-7 | Spark's answer on an equality delete whose key column is read by position into another type is not known: whether Spark raises | OPEN | Not measured. The pin asserts the fork's refusal, as the ruling asked. Open item 1 |
| S-8 | Kind guard: a top-level column whose nesting kind (struct, list, map or primitive) differs from its positional table field's kind refuses, on full and filtered reads. Struct over list, list over struct, struct over map and nested over primitive all refuse. A struct read into a scalar field goes through the same refusal, where the base raised an Arrow cast error | PROVEN | Pins `a_positional_struct_column_landing_on_a_list_field_refuses`, `a_filter_on_a_list_field_fed_by_a_positional_struct_refuses` (`IS NULL` and `IS NOT NULL`), `a_positional_list_column_landing_on_a_struct_field_refuses` (full read and `WHERE s IS NULL`), `a_positional_struct_column_landing_on_a_map_field_refuses` and `a_positional_struct_column_landing_on_an_int_field_refuses` (the last red on the base as an Arrow cast error). The five round-3 pins are red at `9facc9a7` (reads `["1, NULL", "2, NULL"]`, `[]`), green at `91a72650`. Spark raises on `SELECT *`, `SELECT a WHERE xs IS NULL` and `SELECT a WHERE xs IS NOT NULL` for a struct over a list; on `SELECT *` and `SELECT a WHERE s IS NULL` for a list over a struct; and on `SELECT *` for a struct over a map (orchestrator, 2026-11-02, copied) |
| S-9 | Same kind over same kind (struct over struct, list over list, map over map) keeps the nested by-name rule. Spark reads list elements and map entries on this route but no struct child; the fork reads struct children by name. This is the ruled divergence, in RePark's favour | PROVEN | Control `a_positional_struct_over_a_struct_with_other_children_reads_by_name_control` (a file struct `t{j}` over a table struct `s{i}` reads `1, {i: }` and `2, {i: }`), green at head. Spark's answers, copied: `s` as `Row(a=None, b=None)`, `xs` as `[1, 2]`, `mp` as `{'k': 1}`. A list over a list of another element type is not pinned: the card gives no Spark answer for it |
| S-10 | Lists and maps read on the no-mapping route as Spark reads them: table `(id, xs list<int>, mp map<string,int>)` over an id-less file in schema order reads `("1", "[1, 2]", "{k: 1}")` | PROVEN | Control `lists_and_maps_read_on_the_no_mapping_route_control`, green at head. Spark's `xs` as `[1, 2]` and `mp` as `{'k': 1}` (copied) |
| S-11 | A dropped and re-added top-level name reads NULL on the no-mapping route: `drop c`, `add c`, then the file's `c = 5` reads NULL, as Spark does. This closes the top-level half of the remap ledger's S-26 | PROVEN | Pin `a_re_added_top_level_name_reads_null_on_the_no_mapping_route`: red on the base (`left: [[Some("1"), Some("5")]]`, `right: [[Some("1"), None]]`), green at head. Spark's `[(1, None)]` (copied). The mapping route still reads the re-added name's value: control `a_re_added_top_level_name_reads_its_value_on_the_mapping_route`, green on both |
| S-12 | An unmapped top-level column under a partial mapping reads NULL, never its value by position: the mapping names some columns and the file holds one more | PROVEN | Pin `an_unmapped_top_level_column_under_a_partial_mapping_reads_null`: red on the base (`[Some("1"), Some("2")]` against `[None, None]`), green at head. Mechanism: `compare_schemas` in `record_batch_transformer.rs` returned Different only when both sides carried ids; a None against a Some is now a difference, so the column is sourced by id and reads NULL. Spark's answer for a mapping naming only `zz` over `(a, c)` is `(None, None)` (copied) |
| S-13 | The nested half of the remap ledger's S-26 stands as a ruled divergence: a child dropped and re-added by name under a struct, on the no-mapping route, reads its old value, because the by-name rule keeps no schema history. This unit does not change it and adds no pin | PROVEN | Fact clause. The fork's answer is the remap ledger's round-2 measurement (V2-F1), copied and not re-measured here. The ruling that it stands is the orchestrator's, as the brief records it |
| S-14 | Spark's zero-row filtered reads after a drop are a kept divergence in the fork's favour. After `DROP COLUMN x` on table `(x, a, b)` over file `(a, b)`, Spark's projection is positional (`a` holds `b`'s values, `b` NULL), yet `count(*) WHERE a = 1` is 0, `WHERE a = 40` is 0, `WHERE b IS NULL` is 0 and `SELECT a WHERE a > 25` is empty. The fork answers coherently: `a = 40` one row, `a = 1` none, `b IS NULL` four | PROVEN | Fact clause. Spark's answers are the orchestrator's measurement of 2026-11-02, copied. The fork's answers are pinned by the re-pin in S-16 |
| S-15 | Re-pin `struct_idless_child_4.rs`: the two pins `the_reader_mark_never_reaches_the_output_with_two_unmapped_columns_mapping` and `the_reader_mark_never_reaches_the_output_with_one_unmapped_column_mapping` keep their names and their reader-mark check, and now expect NULL for the unmapped columns, under S-12. Reason: the partial-mapping rule, and Spark's `(None, None)` for a mapping naming only `zz` over `(a, c)` | PROVEN | Re-pin commit `e8568134` (round-2 ruling 1). Red at `414f544d` (`struct_idless_child_4`: 14 passed, 2 failed, the two pins), green at `91a72650` (16 passed) |
| S-16 | Re-pin `struct_idless_child_2.rs` `idless_row_group_pruning_uses_the_fallback_ids_after_a_leading_drop`: the projection is now `a` = `[40, 30, 20, 10]` and `b` NULL, which is Spark's projection, and the filters are coherent with it, pruned and unpruned. The old pin read `a = 1` as one row, the by-name answer of the table before the drop | PROVEN | Re-pin commit `e8568134`. Red at `414f544d` (the pin failed, `a = 1` finding 0 rows where the old pin expected 1), green at `91a72650`. Spark's projection `[(10, None), (20, None), (30, None), (40, None)]` (copied) matches the fork's. The filter answers are S-14 |
| S-17 | Re-pin `src/scan/container_null_tests.rs` `test_container_null_predicates_match_spark_oracle_without_field_ids`: the fixture's top-level ids 9, 10, 13, 15 and 18 are matched against the file's 13 positional columns. `id` and `st` keep their answers. `xs` and `mp` land on nested-only ids and read NULL. `deep` lands on the table's `xs` list, so the cases that filter `xs` expect the kind refusal naming `deep` and `xs` | PROVEN | Re-pin commit `e8568134` (positional answers) and `91a72650` (the kind refusal). Red at `9facc9a7`: the first `xs` case read 1 batch where it must refuse. Green at `91a72650` (6 passed) |
| S-18 | The container fixture's `xs` cases are not run through Spark, so their refusal rests on the kind rule of S-8, not on a measured answer for this fixture | OPEN | The round-3 Spark measurement covers the three kind shapes of S-8, not this fixture. Open item 2 |
| S-19 | The remap ledger's S-20 is closed for the type-changing top-level shapes: a positional read into another type refuses (S-4, S-8) or reads NULL (S-11, S-12), where the base cast silently. A same-type positional read keeps its positional answer, Java's rule, pinned as a control, and Spark never reaches that route because `add_files` installs a mapping | PROVEN | Pins S-4, S-8, S-11 and S-12, and the control `a_same_type_swap_reads_by_position_control` (green on both). The statement about Spark's route is the remap ledger's S-20, copied |
| S-20 | Timestamp-family positional mismatches, unconvertible file types and the variant type are not guarded: the guard leaves them to the INT96 coercion and to the existing path | OPEN | No pin of this unit covers these shapes and Spark is not measured for them. The scope is the builder's. Open item 3 |
| S-21 | The `s3:/` debris: the three interop tests that register Java-written metadata with an `s3://` bucket location into a memory catalog over local storage write a relative `s3:/` tree under the crate directory. Unchanged by this unit; the builder removed the tree after the gates | OPEN | The builder's notes. Open item 4 |
| S-22 | Verification: the union of the PASS verdicts of an Opus 5.5 verifier and a Muse MAX critic on the same head is recorded | OPEN | Follows the fork CI. Neither has run on `91a72650`. Open item 5 |

**VERDICT (2026-11-02, at build, before verification):** 17 PROVEN, 5 OPEN (S-7, S-18, S-20, S-21, S-22), 0 REJECTED, of 22 clauses.

## Verification

Builder gates at `91a72650`, every command exit 0. Copied from the builder's notes; the clerk ran no cargo.

| Command | Exit | Counts |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | none |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 0 | no warnings |
| `cargo test --no-fail-fast -p iceberg` | 0 | 5161 passed, 0 failed, 21 ignored; lib 4673 passed, 9 ignored |
| `cargo test --no-fail-fast -p iceberg-datafusion` | 0 | 802 passed, 0 failed, 15 ignored |
| `cargo test --doc -p iceberg` | 0 | 90 passed, 10 ignored |
| `python3 scripts/check_rust_file_size.py` | 0 | 717 files clean (89 legacy ceilings) |
| comment gate, each commit | prints nothing | |

The orchestrator re-ran the four pin suites at `91a72650`: 20, 6, 29 and 16 green.

Records gates, run by the clerk at the records commit: `BASE_REF=56ff5545 make check-comment-blocks` and `make check-agent-artifacts`, both exit 0.

Not yet run: the Opus 5.5 verifier and the Muse MAX critic, after the fork CI. Verdict OPEN (S-22).

## Open items

1. S-7: measure whether Spark raises on an equality delete whose key column is read by position into another type.
2. S-18: run the container fixture's `xs` cases through Spark, or accept the kind rule of S-8 as its basis.
3. S-20: pin or scope the timestamp-family, unconvertible and variant shapes.
4. S-21: the `s3:/` debris belongs to a later unit, which should stop the three interop tests from writing an `s3:/` tree under the crate directory.
5. S-22: record the verifier and critic verdicts on `91a72650`.
