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

# F-NAME-MAPPING-SCHEMA-UPDATE-1 — a schema update maintains `schema.name-mapping.default` as Java's `SchemaUpdate` does, so a renamed, added or re-added name reads its value on the mapping route

**Date:** 2026-11-01. **Base:** fork `main` `ee592dcf` (the `#379` squash). **Branch:**
`fix/name-mapping-schema-update-1`, not pushed. Commits: `dfa0776f` (pins, red on the base),
`6e8593ee` (fix, green at head), and the records commit that adds this text.

**Model:** builder Claude Opus 5.5 (`claude-opus-5-5`), round 0. Records clerk Claude Haiku 5.5
(`claude-haiku-5-5`). The verifiers have not run on `6e8593ee` at the time of this text.

**Outcome in one line.** A schema update on a table with a name mapping now rewrites
`schema.name-mapping.default` in the same commit, as Java's `SchemaUpdate.commit` does. A renamed
child keeps its old name and gains the new one, an added field maps to its new id under its parent,
and a name dropped and re-added moves to the new id. On the mapping route an id-less file then reads
these children under the table's field, where the base read NULL with no error. At head the 14 new
pins are green; 9 of them are red on the base and 5 are controls green on both.

## Why

The fault: `UpdateSchemaAction::commit` never touched `schema.name-mapping.default`. The RePark RP-64
Opus 5.5 verifier found it on 2026-11-01 (finding F1, severity S1, confirmed by the Muse MAX critic),
and the orchestrator reproduced it through RePark on both pins and on Spark 4.1.2 with Iceberg 1.11.0.
The shapes and answers:

- Rename `s.d` to `v` on `(id INT, s STRUCT<d: INT>)`. An id-less file `(1, {d: 1})` is admitted by
  `add_files`, which installs the mapping. After `RENAME COLUMN s.d TO v`, a second id-less file
  `(2, {v: 2})` is read. Spark maps field 3 to names `[d, v]` and reads `(2, {v: 2})`. The fork at
  `ee592dcf` left the mapping at `[d]`, and because `#379` applies the mapping at every depth, it read
  `(2, {v: NULL})`, a silent NULL. At `14a93602` the same shape read `{v: 2}` through the then
  top-level-only stamping.
- After `ADD COLUMN s.k`, Spark maps `k` to id 4, and a file holding `k = 22` reads 22. The fork read NULL.
- Top-level `ADD COLUMN y`, then a file with `y = 7`: Spark reads 7; the fork read NULL on both pins.

## The fix

`crates/iceberg/src/transaction/update_schema_mapping.rs` (a private module, `name_mapping_update`)
ports Java's `MappingUtil.update`, as the builder read it. `UpdateSchemaAction::commit` calls it with
the replayed updates, the parent-to-added-ids map and the added fields before `apply`, and appends one
`TableUpdate::SetProperties` for the single key after `SetCurrentSchema`. The requirements are
unchanged. `spec/name_mapping/create.rs::nested_mapping` became `pub(crate)` and gives the mapping of an
added type. The three rename unit tests moved unchanged from `update_schema.rs` into
`update_schema_mapping_tests.rs`; `update_schema.rs` drops from 3595 to 3551 lines, and its ceiling is
lowered in the same commit.

## Pins

Integration `crates/iceberg/tests/name_mapping_schema_update.rs`, 8 pins, 6 red on `ee592dcf`:

- (a) `rename_of_a_nested_child_maps_the_new_name_and_keeps_the_old_one`, red: `[Some(1), None]` against `[Some(1), Some(2)]`.
- (b) `added_nested_child_maps_to_its_new_id`, red: `[None, None, None]` against `[None, Some(22), None]`.
- (c) `added_top_level_column_maps_to_its_new_id`, red: `[None, None]` against `[None, Some(7)]`.
- (d) `dropped_then_re_added_name_moves_to_the_new_id`, red: `[None]` against `[Some(2)]`.
- (e) `added_struct_column_maps_its_children`, red: `[false, false]` against `[false, true]`.
- (f) `added_child_of_a_list_element_maps_under_the_element`, red: `[None, None]` against `[None, Some(20)]`.
- Controls, green on both: (g) `table_without_a_mapping_gets_none_from_a_schema_update`, (h) `type_promotion_and_move_leave_the_mapping_unchanged`.

Unit `crates/iceberg/src/transaction/update_schema_mapping_tests.rs`, 6 new pins:

- (i) `rename_commit_holds_the_schema_updates_and_the_mapping_with_the_same_requirements`, red: two updates against three.
- `mapped_field_without_an_id_is_kept` and `add_to_a_map_value_struct_maps_under_the_value`, red on the base because no property update is written.
- Controls, green on both: `commit_without_a_mapping_emits_no_property_update`,
  `type_doc_requirement_and_move_changes_emit_no_property_update`,
  `unparsable_mapping_is_left_unchanged_and_the_schema_still_commits`.

The three rename tests moved from `update_schema.rs` are not new pins; they are green at head.

Sabotage (not committed): the sibling-name removal of the add was forced to return the field
unchanged. Pin (d) failed (`left: [None]`, `right: [Some(2)]`) and the other seven passed. The source
was restored and checked with `cmp`.

## Clauses

| Id | Claim | Verdict | Pins and evidence |
|---|---|---|---|
| S-1 | The fault reproduces red on `ee592dcf`: a schema update never rewrites `schema.name-mapping.default`, so the mapping keeps the pre-evolution names. Six of the eight integration pins fail there, the two controls pass, and three of the six new unit pins fail | PROVEN | Red run on `ee592dcf` with the pins (see Pins). The orchestrator's measurements and the RP-64 verifier's F1 (see Why), copied; the clerk did not re-run them |
| S-2 | Rename: a rename adds the new name to the mapped field and keeps the old names, so a renamed nested child reads under both names with its id | PROVEN | Pin (a) `rename_of_a_nested_child_maps_the_new_name_and_keeps_the_old_one`, red on the base. Mapping JSON (a) of the builder's notes holds `names [d, v]` for field 3 after the rename |
| S-3 | Add: an added nested child, an added top-level column, an added struct column's children and a child of a list element each map to the id the update assigned, under their parent's mapped field | PROVEN | Pins (b), (c), (e) and (f), red on the base (see Pins). Mapping JSON (b) of the builder's notes shows `k` as field 4 with names `[k]` under `s` |
| S-4 | Add into a map value struct: the added child maps under the value entry | PROVEN | Unit `add_to_a_map_value_struct_maps_under_the_value`, red on the base (no property written), green at head |
| S-5 | Add name move: an add removes a reused name from the sibling mapped fields under the same parent, then appends the new mapped field. A name dropped and re-added moves to the new id | PROVEN | Pin (d) `dropped_then_re_added_name_moves_to_the_new_id`, red on the base. The sabotage that removes the sibling-name step turns (d) red and leaves the other seven green (Pins) |
| S-6 | A mapping entry with no field id is kept unchanged by an update | PROVEN | Unit `mapped_field_without_an_id_is_kept`, red on the base (no property written), green at head |
| S-7 | No-op changes: a type promotion, a doc change, a requirement change (a column made optional), a move and a delete emit no property update, so the mapping keeps its names | PROVEN | Unit `type_doc_requirement_and_move_changes_emit_no_property_update`, green on both, with the delete in its action list; pin (h) `type_promotion_and_move_leave_the_mapping_unchanged`, green on both, reads the mapping after the changes |
| S-8 | A default change (`update_column_default`) emits no property update and leaves the mapping unchanged. The brief's no-op list names defaults | OPEN | No pin covers a default change: the unit pin's action list has type, doc, requirement, move and delete, and no default. Not measured |
| S-9 | No property: a table without `schema.name-mapping.default` gets no property update, and its mapping lookups stay None. The no-mapping route of `#379` S-26 is unchanged by this rule | PROVEN | Unit `commit_without_a_mapping_emits_no_property_update` and pin (g) `table_without_a_mapping_gets_none_from_a_schema_update`, both green on both |
| S-10 | Unparsable property: a property that does not parse is left unchanged, and the schema update still commits. The fork logs a `tracing::warn!`; the pin does not assert the log | PROVEN | Unit `unparsable_mapping_is_left_unchanged_and_the_schema_still_commits`, green on both |
| S-11 | One commit shape: a rename commit emits `AddSchema`, `SetCurrentSchema` and one `SetProperties` with the single key, under the same two requirements as before the change | PROVEN | Unit `rename_commit_holds_the_schema_updates_and_the_mapping_with_the_same_requirements`, red on the base (two updates, against three) |
| S-12 | The property is written as compact JSON (`serde_json::to_string`); Java writes pretty JSON. The fork's reader parses the compact property in every mapping-route pin, so the bytes differ from Java's and the fork reads them. The orchestrator ruled on 2026-11-01 that the byte difference is accepted | PROVEN | Fact clause (orchestrator ruling, 2026-11-01). The reader side is shown by pins (a) to (f). Java's parse of these bytes is not measured |
| S-13 | The new pins are red on `ee592dcf` with the pins only and green on `6e8593ee`: 9 red (6 integration, 3 unit) and 5 controls green on both | PROVEN | Red and green runs of the builder, verbatim (see Pins). The clerk ran no cargo |
| S-14 | Gates at `6e8593ee` pass: fmt; clippy with no warnings; `cargo test -p iceberg` (5124 passed, 0 failed); `cargo test -p iceberg-datafusion` (802 passed, 0 failed); the file-size ratchet (`update_schema.rs` lowered from 3595 to 3551, no ceiling raised); the `#379` suites unchanged and green | PROVEN | Builder runs (see Gates). The clerk ran no cargo |
| S-15 | Spark parity on the measured shapes: the fork's answers equal Spark 4.1.2's on the rename (`(2, {v: 2})`), the nested add (`k` reads 22), the top-level add (`y = 7` reads 7) and the drop then re-add (`(1, {a: 1, b: 2}, 5)`) | PROVEN | Spark side: the orchestrator's measurement and the RP-64 verifier's Spark answer, copied. Fork side: pins (a), (b), (c) and (d), and mapping JSON (a), (b) and (d) of the builder's notes. The clerk did not re-run Spark |
| S-16 | `#379` S-26 consequence: on the mapping route, a child dropped and re-added by name reads its old value under the re-added field (`{a: 1, b: 2}`), which is Spark's answer, where `#379` S-26 said NULL. The no-mapping route is unchanged (S-9) | PROVEN | Pin (d), green at head. Spark's answer `[(1, {a: 1, b: 2}, 5)]` from the RP-64 verifier, 2026-11-01, copied. The `#379` ledger carries an errata paragraph under its title |
| S-17 | The port follows Java's `MappingUtil.update` (rename, add with the name move, delete, no-op changes, no property, unparsable property). The clerk did not check these rules against the Java source; the builder's reading of Java is in its notes | OPEN | The Java source was not checked in this unit, as with `#379` S-12 |
| S-18 | An add whose parent has no mapped field in the mapping adds nothing, as in Java's visitor, which visits mapped fields only. A partial mapping stays partial | OPEN | Not pinned |
| S-19 | A rename onto a name that a dropped sibling's mapped field still holds is ambiguous: the new name is added and the dropped sibling keeps it; the reader takes the first match, as in Java | OPEN | Not pinned |
| S-20 | Names keep their order with the new one appended; existing duplicates are not removed, where Java keeps a Set | OPEN | Not pinned |
| S-21 | RePark's own `ALTER TABLE` reaches this through the fork's action, so RePark's answers change only when RePark repins (RP-64 fix round) | OPEN | Not measured in this unit; the repin is the RP-64 fix round's work |
| S-22 | Verification: the union of the PASS verdicts of an Opus 5.5 verifier and a Muse MAX critic on `6e8593ee` is recorded | OPEN | Not yet run at build time; recorded here so that the closure can cite it |

VERDICT (2026-11-01, at build, before verification): 15 PROVEN, 7 OPEN (S-8, S-17 to S-22), 0 REJECTED, of 22 clauses.

## Gates

Builder runs at `6e8593ee`. The clerk ran no cargo; the rows are copied from the builder's notes. Every
cargo command ran under the flock, environment and wrapper of the brief.

| Command | Exit | Counts |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | none |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 0 | no warnings |
| `cargo test -p iceberg` | 0 | 76 result lines; 5124 passed, 0 failed, 21 ignored; lib 4667 passed, 9 ignored; doctests 90 passed, 10 ignored |
| `cargo test -p iceberg-datafusion` | 0 | 43 result lines; 802 passed, 0 failed, 15 ignored |
| `python3 scripts/check_rust_file_size.py` | 0 | 714 files clean (89 legacy ceilings); `update_schema.rs` ceiling lowered 3595 to 3551; `transaction/mod.rs` stays at 1928 |
| comment gate (both commits) | — | only the 16-line ASF header of each new `.rs` file (two in `dfa0776f`, one in `6e8593ee`) |
| `#379` suites, unchanged and green | 0 | `struct_idless_child` 18, `_2` 29, `_3` 14, `_4` 16, `interop_name_mapping` 1 |

Records gates for this commit (clerk), every command exit 0: `BASE_REF=ee592dcf make check-comment-blocks`;
`make check-agent-artifacts`. The staged diff touches only markdown: the two ledgers and the two map files.

## Verification

OPEN (S-22). An Opus 5.5 verifier and a Muse MAX critic have not yet run on `6e8593ee`. When both report
PASS on the same head, S-22 becomes PROVEN by citation, as S-17 of `#379` did.

## Open items

1. S-17: check the port against Java's `MappingUtil.update` in the Java source, if the closure should cite the source and not the builder's reading.
2. S-8: a pin for the default change, or a scope line saying it is out of this unit.
3. S-18 to S-20: OPEN and not pinned; each needs a pin or a stated scope before it closes.
4. S-21: RePark's answers change only when RePark repins (the RP-64 fix round).
5. S-22: the verifier union, as above.
