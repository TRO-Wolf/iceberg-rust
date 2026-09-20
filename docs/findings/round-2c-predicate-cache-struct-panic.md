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

# Round 2c — `select_where_xs_is_null_returns_the_null_row` redness

## Symptom

`physical_plan::list_null_tests::select_where_xs_is_null_returns_the_null_row`
(`iceberg-datafusion`) panics on branch head with parquet 58.4
`StructArrayReader::consume_batch` (`struct_array.rs:142`):
`child with nullable parents must have definition level`,
reached through `ReadPlanBuilder::with_predicate_options` in the async
(push-decoder) read path. Green on base `0cb6ebd30`, red on `1a586446`
and identically on `c159b8d4`.

## Root cause

The standing hypothesis (predicate mask selects a leaf without its
ancestor path) is wrong. The mask is correct: for `xs IS NULL` on
`struct<a:int>` it selects exactly the `a` leaf, and the sync parquet
reader decodes that projection without error. The defect is parquet
58.4's predicate-result cache in the async path. The push decoder
wraps every predicate leaf that also appears in the scan projection in
a `CachedArrayReader`, except leaves excluded by
`ProjectionMask::without_nested_types` — and that filter only excludes
roots spanning several leaves plus `LIST` roots. A single-child
`STRUCT` root passes the filter, so the `a` leaf is cached;
`CachedArrayReader::get_def_levels` returns `None` (it documents no
support for nullable parents), and the nullable `xs`
`StructArrayReader` panics on its `.expect`. List and map shapes pass
because `LIST` roots are excluded explicitly and map roots always span
two leaves; the base passed because it never mapped a container id to
a leaf, so the predicate mask was empty and nothing was cached.

## Confirming measurements

- Probe of `plan_row_filter` during the failing test: 13 calls before
  the panic — 8 list (all pass), 4 map (all pass), then the first
  struct call panics. Predicate mask `[1]`, ids `{2}`.
- The failing scan projects `[1, 2]`, so the predicate leaf overlaps
  the projection and parquet caches it.
- Iceberg-level scan of `{id, xs struct{a}}` with `xs IS NULL`:
  projecting `[1]` passes, projecting `[1, 2]` panics — the overlap
  is the trigger.
- The passing `delete_where_xs_is_null` test never calls
  `plan_row_filter` (the DML path does not push the filter), which is
  why exactly one test is red.

## Fix direction

Keep the pushdown. When the pushed predicate mask contains a leaf
whose root is a single-leaf non-`LIST` root and whose ancestor path
holds an `OPTIONAL` plain (non-list, non-map) group, disable parquet's
predicate cache for that scan with
`with_max_predicate_cache_size(0)`. Raw readers carry definition
levels, so the nullable struct parent decodes normally.
