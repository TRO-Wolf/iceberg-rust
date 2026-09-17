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

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Status |
|---|---|---|---|
| C-001 | File struct lacks a child the table struct has → child reads NULL, kept values read by field id. | transformer red test (a), green after fix. | OPEN |
| C-002 | File list-element struct lacks a child the table element struct has → NULL-fill through the list. | transformer red test (b), green after fix. | OPEN |
| C-003 | File map-value struct lacks a child the table value struct has → NULL-fill through the map. | transformer red test (c), green after fix. | OPEN |
| C-004 | Two levels of nesting with a leaf added at the bottom → NULL-fill at depth. | transformer red test (d), green after fix. | OPEN |
| C-005 | Nested child renamed (same id, new name) reads the old file's values under the new name. | transformer red test, green after fix. | OPEN |
| C-006 | Nested children reordered (same ids, new order) read by field id, not position. | transformer red test, green after fix. | OPEN |
| C-007 | Disabling the nested null-fill turns the new tests red; restoring turns them green. | mutation run pasted below. | OPEN |
| C-008 | Gates hold: `cargo test -p iceberg --lib`, clippy `-D warnings`, `cargo test -p iceberg-datafusion --lib`, fmt, fork scripts. | gates table below. | OPEN |

## Evidence

### RED (base `4151b488`, transformer tests)

(paste after run)

### Mutation

(paste after run)

### Gates

| Command | Exit |
|---|---|
| (paste after run) | |

## Open questions

None inside this unit. Out of scope by brief: the RePark side (adopting a
Spark-evolved table; nested CREATE / `ADD COLUMN s.b` DDL) belongs to the next
run.
