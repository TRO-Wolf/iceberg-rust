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

# F-EQDEL-DROPPED-KEY-1 — equality deletes survive a dropped, renamed or re-added key column

**Date:** 2026-10-28. **Base:** fork `main` `ea0b7741`. **Branch:**
`fix/eqdel-dropped-key-1`, commits `ee5e0c97` (fix) and `68ebe908` (pins).

**Model:** builder Muse Spark (`muse-spark-1.3-contributor`).

**Outcome in one line.** A scan no longer fails with `Unexpected => field not
found` after the key column of a committed equality delete is dropped. The
delete keeps applying to old data files by field id, post-drop files keep
their rows, and a re-added same-name column never matches the old delete.

## The fault at `ea0b7741`

A v2 `{id: long, data: string}` table with five rows and a committed equality
delete on `id` for 20 and 40 scans three rows. After
`Transaction::update_schema().delete_column("id")` commits, scanning `data`
fails with `Unexpected => field not found`.

Two name- or current-schema couplings cause it. `evolve_schema` in
`crates/iceberg/src/arrow/delete_file_loader.rs` builds a
`RecordBatchTransformer` against the CURRENT table schema with the historical
`equality_ids`, so a dropped id has no field to resolve. Had that succeeded,
`DeleteFilter::build_equality_delete_predicate` binds the per-file survival
predicates by NAME against the task schema, so a dropped name fails to bind
and a re-added same-name column would bind the old delete to the NEW id.

The Iceberg specification, Equality Delete Files (verified by the audit
2026-10-28): a delete's equality field ids are applied by id, and the column
keeps being used for the delete even after it is dropped from the table
schema. Expected post-drop answer: `a, c, e`.

## The fix

Each equality-delete file is evolved and bound against its own per-file
schema, built in the loader by `BasicDeleteFileLoader::equality_delete_schema`
(`crates/iceberg/src/arrow/delete_file_loader.rs`): an equality id present in
the current schema keeps its current definition (rename-safe, since the
transformer renames the delete column by id); an absent id is recovered from
the delete file's own Arrow schema by Parquet field id, with the name
disambiguated against the other equality fields so a re-added same-name
column can never collide.

- The caching loader's equality arm opens the delete file once, returning the
  stream and the Arrow schema (`parquet_stream_and_arrow_schema`), evolves
  against the per-file schema, and the `FreshEqDel` channel carries the
  per-file bind schema to the filter. The trait's `read_delete_file` equality
  arm takes the same path. The transformer itself is unchanged: the minimal
  schema makes its existing id lookup succeed.
- `EqDelState::Loaded` stores the bind schema with the predicate, and
  `build_equality_delete_predicate` binds each file's predicate against its
  own schema with the task's case sensitivity, then ANDs the bound results.
- On the data side nothing is added to the scan projection. The pushed
  `RowFilter` maps predicate field ids to Parquet leaves from the data file's
  own embedded ids, so a dropped key is decoded as a filter-only column from
  files that still hold it. Files written after the drop carry no such column;
  the batch evaluator treats a missing column as all-NULL, and under the
  survival predicate (`col != v`) NULL rows are kept. No `scan/` change, no
  `FileScanTask` change, no `reader.rs` change.
- `plan_row_filter` fails loud with `FeatureUnsupported` when a predicate id
  is absent from the task schema outside the embedded-id Push path (forced
  residual such as `_deleted` scans, and id-less files). Without the guards
  those paths would flip from loud errors to silent under-deletes.
- The page-index evaluator treats a predicate id absent from the snapshot
  schema as unevaluable and selects all rows. Reachable only with opt-in page
  pruning plus NULL cells in the delete file. The row-group evaluator is
  intentionally untouched: the survival predicate rewrites to `!=` / null
  leaves only, and its `not_eq` arm returns might-match without consulting the
  schema while the null arms use stats alone. The pins pass with default-on
  row-group pruning and that file unmodified.
- Two pure-move splits keep the file-size ceilings honest (moved code sheds
  its comments per the owner rule): positional-delete stream parsing moves
  from the caching loader to `arrow/pos_delete_parse.rs`, and the
  positional-claim key plus failure renderer move from the delete filter to
  `arrow/pos_del_claim.rs`. Ceilings lower to the new counts (caching loader
  3270 to 3148, delete filter 2938 to 2915).

## Pins

All three live in `crates/iceberg/tests/eqdel_dropped_key.rs` (351 lines).
Each builds a v2 table through the production catalog, writers, transactions
and Arrow scan, and asserts a baseline of 3 rows before evolving the schema.

- `equality_delete_survives_drop_of_key_column`: drop `id`, scan `data`,
  answers `a, c, e` in file order.
- `equality_delete_survives_rename_of_key_column`: rename `id` to
  `identifier`, scan `data`, answers `a, c, e`. Green before and after the
  fix: a no-regression pin over the refactored evolve-and-bind path.
- `equality_delete_survives_readd_and_new_delete_scopes_to_new_rows`: drop
  `id`, re-add `id` (new id asserted different), scan `data` for `a, c, e`;
  append post-re-add rows `(20, f)` and `(60, g)`; commit a new delete on the
  new id for `[60, 10]`; scan `data` for sorted `[a, c, e, f]`. Row `f`
  (value 20 in the NEW column) surviving the old delete and row `a` (value 10
  in the OLD column) surviving the new delete prove id-keyed, never
  name-keyed, application in both directions.

## Red runs

Run 1, the ported pins against the unmodified base tree: drop and re-add
fail, rename passes.

```
test equality_delete_survives_drop_of_key_column ... FAILED
test equality_delete_survives_readd_and_new_delete_scopes_to_new_rows ... FAILED
collect scan batches: Unexpected => field not found
test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out
```

Run 2, the brief's revert procedure after the pins commit (fix-commit sources
checked out at their base revisions, new sibling modules left unreferenced in
the tree, then restored): identical red, 2 failed out of 3, and 3 passed after
the restore.

Neighbour baseline: `cargo test -p iceberg --lib` at base gives 4639 passed,
0 failed, 9 ignored; the same command at head gives the identical 4639
passed, 0 failed, 9 ignored.

## Gates (builder, measured at `68ebe908`)

- `cargo test -p iceberg --test eqdel_dropped_key`: exit 0, 3 passed.
- `cargo test -p iceberg`: exit 0. Lib 4639 passed, 0 failed, 9 ignored; every
  integration binary ok with 0 failed; doctests 90 passed, 0 failed,
  10 ignored.
- `cargo clippy -p iceberg --all-targets -- -D warnings`: exit 0.
- `cargo fmt --all -- --check`: first run red on 4 call sites the test
  repoint shortened past the fmt width; repaired with `cargo fmt`
  (whitespace-only, plus the caching-loader ceiling ratchet 3157 to 3148),
  then clean. The repair rides in the records commit with this note, since a
  rebase is prohibited on this lane.
- `python3 scripts/check_rust_file_size.py`: 701 files clean.

## Follow-ups (out of scope)

1. Avro and ORC scans, the `_pos` whole-file path, and compaction
   (`convert_equality_delete_files.rs`) still fail LOUD on dropped equality
   keys, exactly as at base. Making them apply the deletes needs the dropped
   column in their projections, which is scan-projection surgery the brief
   excludes. No silent flips: every one of these paths errors rather than
   under-deletes.
2. `_deleted` projections and id-less Parquet files fail loud with
   `FeatureUnsupported` naming the dropped ids when such deletes are present.
   Same status as item 1: loud by design, applied later if the projection
   work is ever scoped.
3. The per-file bind schema is transient (never persisted). A future
   name-based consumer of stored eq-delete predicates must resolve through
   the delete file's own schema, not the current table schema.
