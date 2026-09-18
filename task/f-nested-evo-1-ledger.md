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

# F-NESTED-EVO-1 — struct children added after old files were written read as NULL by field id

**Date:** 2026-09-17. **Branch:** `fix/nested-evo-1` (from `origin/main` `4151b488`).
**Model:** muse-spark-1.3-contributor.
**Consumer:** RePark row V2-10d (`p_isolate_nested` `st_nested_add`).

## The defect

Spark creates `t (id INT, s STRUCT<a: INT>)`, inserts `(1, {a:1})`, runs
`ALTER TABLE t ADD COLUMN s.b STRING`, inserts `(2, {a:2, b:'y'})`. Spark reads
`[(1, {a:1, b:None}), (2, {a:2, b:'y'})]`. The fork fails every read with
`Unexpected => Arrow Schema Error, source: Invalid argument error: Incorrect
number of arrays for StructArray fields, expected 2 got 1`. Same shape for a
list-element struct child (`ADD COLUMN arrs.element.y INT`). Nested DROP and
RENAME already read equal; plain structs, arrays, maps already read equal.

Probable site: `crates/iceberg/src/arrow/record_batch_transformer.rs` — a file
struct with fewer children than the table struct falls into
`ColumnSource::Promote` and `arrow_cast::cast` rejects the arity mismatch (and
would project positionally even when it succeeds, breaking rename/reorder).

## Decisions

- **D-1** New module `crates/iceberg/src/arrow/nested_projection.rs` holds the
  recursive nested projector plus the six clause tests. `record_batch_transformer.rs`
  sat exactly at its 2475-line ceiling, so the hook there is minimal (one import,
  one `ColumnSource` variant, one branch, one match arm) and `create_column` moved
  into the new module unchanged; ceiling 2475 → 2457.
- **D-2** Match nested children by `PARQUET:field_id`, never name or position.
  A source child without an id is dropped; a target child without a file match
  is NULL-filled with `new_null_array` of the child's Arrow type.
- **D-3** Every same-variant nested pair (struct/list/map) routes through the
  projector, even when `equals_datatype` holds: that predicate ignores struct
  field names, so a nested rename otherwise passes through with stale file names.
  Whole-batch `PassThrough` still covers the fully identical common case.
- **D-4** A source struct whose children carry no ids at all (legacy id-less
  files, top-level fallback path) keeps the old behavior: passthrough when
  `equals_datatype`, arrow cast otherwise. Found via
  `arrow::reader::tests::test_read_parquet_without_field_ids_with_struct`,
  which the first cut broke (`Found unmasked nulls for non-nullable field`).
- **D-5** Recursion carries a depth counter capped at 32 (`DataInvalid` past it).

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Status |
|---|---|---|---|
| C-001 | File struct lacks a child the table struct has → child reads NULL, kept values read by field id. | `struct_child_added_after_file_written_reads_null` plus end-to-end `parquet_file_missing_struct_child_reads_null_through_reader` (real parquet footer ids → mask → transformer), red then green. | PROVEN |
| C-002 | File list-element struct lacks a child the table element struct has → NULL-fill through the list. | `list_element_struct_child_added_after_file_written_reads_null`, red then green. | PROVEN |
| C-003 | File map-value struct lacks a child the table value struct has → NULL-fill through the map. | `map_value_struct_child_added_after_file_written_reads_null`, red then green. | PROVEN |
| C-004 | Two levels of nesting with a leaf added at the bottom → NULL-fill at depth. | `struct_child_added_two_levels_deep_reads_null`, red then green. | PROVEN |
| C-005 | Nested child renamed (same id, new name) reads the old file's values under the new name. | `renamed_struct_child_reads_by_field_id`, red then green. | PROVEN |
| C-006 | Nested children reordered (same ids, new order) read by field id, not position. | `reordered_struct_children_read_by_field_id`, green before and after (arrow cast already matched by name; kept as guard). | PROVEN |
| C-007 | Disabling the nested null-fill turns the new tests red; restoring turns them green. | mutation run pasted below. | PROVEN |
| C-008 | Gates hold: `cargo test -p iceberg --lib`, clippy `-D warnings`, `cargo test -p iceberg-datafusion --lib`, fmt, fork scripts. | gates table below. | PROVEN |

## Evidence

### RED (red commit `f502f5ff`, transformer tests)

`cargo test -p iceberg --lib arrow::nested_projection` → 1 passed, 5 failed:

```
Source: Invalid argument error: Incorrect number of arrays for StructArray fields, expected 2 got 1
```

Failing: struct-child-added, list-element-child-added, map-value-child-added,
two-levels-deep, renamed. Passing pre-fix: reordered (arrow's struct cast
matches children by name, so the reorder already read correctly; the rename
failed its schema-vs-column name assertion).

### Mutation

`nested_projection_applies` forced to `false` → 1 passed, 5 failed (same five;
4 with the `Incorrect number of arrays` source error). Restored → 6 passed.
The reorder guard passes in both states, as in RED. The end-to-end parquet test
was verified red the same way (fails with `Incorrect number of arrays ...
expected 2 got 1` through the real reader) and green after restore — 7/7.

### Gates

| Command | Exit |
|---|---|
| `cargo test -p iceberg --lib` | 0 — 3744 passed, 0 failed, 8 ignored (final tree) |
| `cargo clippy -p iceberg --all-targets -- -D warnings` | 0 |
| `cargo test -p iceberg-datafusion --lib` | 0 — 228 passed, 0 failed, 1 ignored |
| `cargo fmt --all -- --check` | 0 |
| `python3 scripts/check_rust_file_size.py` | 0 — 492 files clean (ceiling 2475 → 2457) |
| `bash scripts/check_agent_artifacts.sh` | 0 |
| `bash scripts/check_comment_blocks.sh` | 0 |
| comment grep `git diff --cached \| grep -P '^\+\s*(//\|#(?!\[))'` | nothing but the new file's ASF header (allowed exception) |

## Open questions

None inside this unit. Out of scope by brief: the RePark side (adopting a
Spark-evolved table; nested CREATE / `ADD COLUMN s.b` DDL) belongs to the next
run.

## Round 2 — review remediation (devin-worker / SWE-2, 2026-09-17)

Two read-only reviews of `136c3da09` (`/tmp/oc-worker/ka-rv/reviews/nest-logic-report.md`,
`nest-rustperf-report.md`) returned findings L-001..L-011 and R-01..R-04. This
section records the round-2 remediation: red-first evidence, per-finding
decisions (continuing the D-n numbering), and gate output.

### RED (test commit, before the fix)

`cargo test -p iceberg --lib arrow::nested_projection` on `136c3da09` plus the
new tests in `nested_projection_evo_tests.rs` → 8 passed, 10 failed:

| Test | Clause | Observed failure |
|---|---|---|
| `nested_child_with_initial_default_reads_default` | L-001 | `b` read `""` (NULL), not default `"x"` |
| `mixed_field_id_struct_matches_idless_child_by_name` | L-002 | id-less child `b` read `""` (NULL) instead of file values |
| `zero_child_file_struct_fills_target_children` | L-002 | vacuous `all()` → legacy cast → `StructArray` arity error |
| `map_key_struct_child_added_after_file_written_reads_null` | L-003 | `failed to cast map keys`, `expected 2 got 1` |
| `deeply_nested_struct_child_add_projects_within_bound` | L-004 | `nested schema projection exceeds depth 32` |
| `required_nested_child_missing_without_default_errors` | L-005 | Arrow's `Found unmasked nulls for non-nullable StructArray field`, not the Iceberg missing-required error |
| `required_nested_child_missing_with_default_reads_default` | L-005 | same Arrow error instead of reading the default |
| `required_list_element_with_null_projected_values_errors` | L-006 | panic inside `ListArray::new` (`Non-nullable field ... cannot contain nulls`) |
| `list_source_projects_into_large_list_target` | L-007 | `failed to cast nested column`, `expected 2 got 1` |
| `fixed_size_list_rebuild_uses_target_size` | L-007 | returned `Ok` with a `FixedSizeList(2)` against a `FixedSizeList(3)` target |
| `nested_add_with_sibling_promotion_and_decimal_widen` | L-008 | passed pre-fix (pin for untested composition, kept as guard) |
