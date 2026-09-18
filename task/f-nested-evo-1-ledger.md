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

### Decisions (round 2)

- **D-6 (L-001)** — nested fills are schema-aware. `NestedProjectionPlan::build` resolves each
  missing target child's `NestedField` by field id via `Schema::field_by_id`; the fill is the
  field's `initial_default` primitive literal when present, null for an optional field, and a
  `DataInvalid` "Missing required field" error for a required field without one — the same
  priority the Avro reader's `missing_column_source` applies, minus the identity-partition
  constant step (partition values only exist at the top level).
- **D-7 (L-002, amended by D-24)** — an id-less source child in a partially id-carrying struct
  matches its target child by NAME. Java's fallback-id path (`ApplyNameMapping` /
  `addFallbackFieldIds`, already ported in this crate's reader for the top level) assigns
  synthetic ids so that name-identical children line up; matching the id-less child by name
  reproduces that observable behavior without inventing ids. The name fallback cannot leak a
  dropped-then-readded name: `source_by_name` holds only id-less source children, the fallback
  runs only when the target child's id lookup misses, and — added in round 3 — the target
  child's id must not exceed the largest stamped sibling id (a field id allocated after every
  id the file carries could not have existed when the file was written, so the id-less child is
  a stale same-named field and null-fills; see D-24). The vacuous-`all()` hole is closed by an
  explicit `!source_fields.is_empty()` guard: a zero-child file struct fills every target child
  instead of taking the legacy cast path.
- **D-8 (L-003)** — map keys project through the same `PlanNode` machinery as map values. The
  `DataType::Map` plan arm builds independent key and value plans, so an added key-struct child
  null-fills and a key-struct child rename resolves by field id.
- **D-9 (L-004)** — the depth bound is 128, matching the crate's other schema-walk bounds
  (`spec::schema::visitor` `MAX_SCHEMA_NESTING_DEPTH`, `arrow::null_propagation`,
  `variant::value`). The bound is enforced at plan build; a 40-deep schema projects and a
  130-deep schema errors at build, pinning both sides of the boundary.
- **D-10 (L-005)** — a required missing nested child without a default fails `DataInvalid` at
  plan build, before any batch is touched; a required missing child with `initial_default` reads
  the default. Both directions pinned.
- **D-11 (L-006)** — every container rebuild uses Arrow's `try_new` (`StructArray`,
  `ListArray`, `LargeListArray`, `FixedSizeListArray`, `MapArray`) and wraps Arrow errors in
  `ErrorKind::DataInvalid`. No panic path remains on projected input: a required list element
  whose projected values carry a null returns an error instead of panicking inside
  `ListArray::new`.
- **D-12 (L-007)** — `nested_projection_applies` accepts the `List` ↔ `LargeList` pair in both
  directions; the apply path rebuilds the values array and constructs fresh `OffsetBuffer`s of
  the target width (i32 ↔ i64 conversion is checked inside `try_new`). `FixedSizeList` rebuilds
  with the TARGET size; a size mismatch surfaces as a `try_new` error rather than a wrong-typed
  array.
- **D-13 (L-008)** — composition pinned by `nested_add_with_sibling_promotion_and_decimal_widen`:
  one transformer run carries a nested struct ADD, a sibling int→long promotion and a decimal
  precision widen together.
- **D-14 (L-009)** — container element/key/value FIELD ids stay un-compared and positional:
  `ListArray` has exactly one element field and `MapArray::try_new` enforces key-then-value, so
  there is nothing to match those ids against; Java's `PruneColumns` treats them as structural
  too. Element field NAME differences are ignored by construction (pinned by
  `list_element_field_named_differently_still_projects`).
- **D-15 (L-010)** — the reorder clause is no longer vacuous: the `==` guard (D-17) only
  short-circuits byte-identical fields, so a reordered struct takes `NestedProject` and the
  projector itself performs the id-based reorder. New pins cover the remaining unmeasured
  nestings the review named: null parent bitmap propagation
  (`null_parent_struct_propagates_null_rows`), drop-then-readd same-name-new-id
  (`nested_field_dropped_then_readded_same_name_new_id_reads_null`), differently-named element
  field, and list-of-list-of-struct recursion. Two files written at different schema versions in
  one scan stay unmeasured: `RecordBatchTransformer` builds its `BatchTransform` lazily from the
  first batch's schema per instance, so each file gets its own plan — correct by construction.
- **D-16 (L-011)** — OPEN. Duplicate nested field ids keep last-wins `HashMap` behavior;
  unparseable `PARQUET:field_id` metadata degrades to "no id" and now reaches the name fallback
  rather than an immediate null-fill. Corrupt-input edge the review rated P3; the top-level id
  map errors on unparseable ids while the nested path stays quiet — deferred hardening, no
  writer-produced file is affected.
- **D-17 (R-01)** — `generate_transform_operations` selects `NestedProject` only when
  `nested_projection_applies(source_type, target_type) && source_type != target_type`;
  byte-identical fields (name, type, nullability and metadata all equal) take `PassThrough` as
  before. `equals_datatype` is deliberately NOT used for this guard — it ignores field names and
  would smuggle the rename clause back onto the pass-through path. The rename and reorder tests
  pin the distinction.
- **D-18 (R-02)** — the field-id maps and per-child plans are built once per file inside
  `NestedProjectionPlan::build` and stored on `ColumnSource::NestedProject(NestedProjectionPlan,
  usize)`; `process_record_batch` mutably borrows the cached transform (`as_mut`) so the plan
  persists across batches.
- **D-19 (R-03)** — a `source_type == target_type` pair inside the plan produces
  `PlanNode::Passthrough`; `apply` returns the source `ArrayRef` by `Arc` clone. Identical
  leaves no longer pay a `cast` round trip.
- **D-20 (R-04)** — `PlanNode::Fill` caches the materialized fill array keyed by row count, so
  repeated same-length batches reuse the null or constant column instead of rebuilding it.
- **D-21 (R-05)** — source-child lookup is a `HashMap` by field id plus a `HashMap` by name for
  id-less children, both built once at plan time: O(children) build, O(1) expected lookup.
- **D-22 (R-06)** — recursion happens at plan build, once per file, bounded at 128; `apply`
  walks the pre-built plan with the same bound. The stack bound matches the crate's other
  schema-walk limits.

### Gates (final tree)

| Command | Exit |
|---|---|
| `cargo test -p iceberg --lib` | 0 — 3761 passed, 0 failed, 8 ignored |
| `cargo test -p iceberg --lib nested_projection` | 0 — 24 passed, 0 failed |
| `cargo fmt --all -- --check` | 0 |
| `make check` | 0 — fmt, workspace clippy `-D warnings`, taplo, cargo-machete, agent-artifacts, matrix-anchors, comment-blocks all OK; `rust-file-size: 495 files clean (98 legacy ceilings)` |
| `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/ka-fork origin/main HEAD` | 64 hits, all ASF license headers of the four new files (the allowed exception) |

### What is not closed

L-011 (D-16) — duplicate/unparseable nested field-id hardening deferred; current behavior is
deterministic (last-wins on duplicates; unparseable degrades to the id-less name fallback).
Everything else in L-001..L-010 and R-01..R-06 is closed on the final tree.

## Round 3 — verification remediation (devin-worker / SWE-2)

The verification critic's mutation pass on `c1bc78864`
(`/tmp/oc-worker/ka-rv/reviews/nest-verify-report.md`) confirmed every round-2 pin
turns red under revert, and left R-01 without a correctness pin plus findings
V-01..V-03. This section records the round-3 remediation: red-first evidence,
per-finding decisions (continuing the D-n numbering), and gate output.

### RED (round 3)

| Test | Finding | Observed failure on `c1bc78864` |
|---|---|---|
| `identical_nested_column_on_a_modify_batch_uses_pass_through` | R-01 | correctness pin, not a red-first fix: on the guarded tree it passes; reverting `source_type != target_type` turns it red (`operations[1]` becomes `NestedProject`) |
| `idless_source_child_named_like_a_readded_field_reads_null` | V-01 | the id-less file child `b` read `"old"` into the re-added field id 5 |
| `nested_time_initial_default_reads_time64` | V-02 | `unexpected target column type Time64(Microsecond)` |
| `nested_uuid_initial_default_reads_fixed_size_binary` | V-02 | `unexpected target column type FixedSizeBinary(16)` |
| `nested_binary_initial_default_reads_large_binary` | V-02 | `unexpected target column type LargeBinary` |
| `nested_fixed_initial_default_reads_fixed_size_binary` | V-02 | `unexpected target column type FixedSizeBinary(4)` |
| `top_level_time_initial_default_reads_time64` | V-02 | same unsupported-type error through the top-level `ColumnSource::Add` path |
| `later_batch_with_a_richer_nested_layout_rebuilds_the_plan` | V-03 | second batch's `b` read `""` (null-filled by the first batch's cached plan) instead of `"keep"` |

### Decisions (round 3)

- **D-23 (R-01)** — the `PassThrough` guard gains a correctness pin:
  `identical_nested_column_on_a_modify_batch_uses_pass_through` forces a `Modify`
  batch with a reordered top-level projection (`[2, 1]`) and asserts every
  operation, including the byte-identical nested struct, is
  `ColumnSource::PassThrough`. Reverting the `source_type != target_type` guard
  was verified to turn the pin red. `RecordBatchTransformer::generate_batch_transform`
  and `BatchTransform` are `pub(crate)` so the test observes the source choice
  directly rather than inferring it from timing.
- **D-24 (V-01)** — the id-less name fallback is bounded by write-time id space.
  `build_struct` tracks `max_source_id`, the largest stamped field id among the
  source struct's children; an id-less source child may bind by name only to a
  target child whose id does not exceed it. A target id newer than every id the
  file carries could not have existed when the file was written, so the
  same-named file child belongs to a dropped field and null-fills — the critic's
  repro (`b:utf8` id-less in file, `b:string` id 5 in table with `a` id 3 the
  only stamped sibling) now reads null. The L-002 fixture gained a stamped
  sibling `c` id 5 so the intended fallback (target `b` id 4, within the file's
  observed id space) stays exercised. D-7's wording is amended to match.
- **D-25 (V-02)** — `create_primitive_array_repeated` covers the remaining
  Iceberg primitive defaults: `Time64(Microsecond)` from a `Long` literal,
  `LargeBinary` from a `Binary` literal, and `FixedSizeBinary(n)` from `Binary`
  or `UInt128` (the `PrimitiveLiteral` representation of Iceberg `uuid`, which
  maps to `FixedSizeBinary(16)`); a null literal produces `new_null_array` for
  the fixed-width type. `fixed_size_binary_column` validates the literal byte
  length against the target width instead of trusting it. The helper is shared
  with top-level `ColumnSource::Add`, so the top-level path gained the same
  types — pinned by `top_level_time_initial_default_reads_time64`. The
  `value.rs` tail tests moved to `value_tail_tests.rs` (include-split, same
  pattern as the nested-projection test files) and its ceiling moved down with
  it.
- **D-26 (V-03)** — the transformer stores its `BatchTransform` together with
  the `SchemaRef` it was built for. Each `process_record_batch` compares the
  incoming schema by pointer first (`Arc::ptr_eq` — the unchanged-schema
  per-batch cost is one pointer compare) and rebuilds only when the pointer
  differs AND the schema value differs, so a differently-allocated but equal
  schema still reuses the plan. The critic's two-batch repro is pinned. The
  row-lineage test block moved to
  `record_batch_transformer_row_lineage_tests.rs` (include-split) and the
  transformer's legacy ceiling moved down with it.

### Gates (final tree, round 3)

| Command | Exit |
|---|---|
| `cargo test -p iceberg --lib` | 0 — 3769 passed, 0 failed, 8 ignored |
| `cargo test -p iceberg --lib nested_projection` | 0 — 32 passed, 0 failed |
| `cargo fmt --all -- --check` | 0 |
| `make check` | 0 — fmt, workspace clippy `-D warnings`, taplo, cargo-machete, agent-artifacts, matrix-anchors, comment-blocks all OK; `rust-file-size: 497 files clean (98 legacy ceilings)` |
| `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/ka-fork origin/main HEAD` | 96 hits, all ASF license headers of the six new files (the allowed exception); the two pre-existing comments the V-03 change touched were removed outright rather than edited, keeping every hit inside the header exception |

### What is not closed (round 3)

L-011 (D-16) — unchanged, still deferred: duplicate/unparseable nested field-id
hardening. Everything the verification critic left open (R-01 pin, V-01..V-03)
is closed on the final tree.
