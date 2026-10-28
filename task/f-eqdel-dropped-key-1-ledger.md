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

**Date:** 2026-10-28. **Base:** fork `main` `ea0b7741`, rebased in fold-2 onto
`56aa8913` (F-POSDEL-FORMAT-READ-1, fork PR #375). **Branch:**
`fix/eqdel-dropped-key-1`. Fold-1 commits (rebased): `988bebdf` (fix),
`0e209dea` (pins), `02474b31` (records). Fold-2 commits: `d6f0180b` (item 0,
rebase follow-through), `414f617f` (item 1), `01e4eb7f` (item 2), `bfec8a9a`
(item 3), `1afda5ec` (item 4, pins) and the records commit (item 5).

**Model:** fold-1 builder Muse Spark (`muse-spark-1.3-contributor`); fold-2
builder Claude Opus 5.5 (`claude-opus-5-5`).

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

1. Avro and ORC DATA-file scans and compaction
   (`convert_equality_delete_files.rs`) still fail LOUD on dropped equality
   keys, as at base (`build_expected_schema` refuses the absent id). The
   fold-1 claim that the `_pos` whole-file path also failed loud, and that
   there were "no silent flips", was refuted by the verifier (V-EQKEY-1,
   V-EQKEY-4): that path answered every row. Fold-2 item 1 makes it apply the
   delete by field id.
2. `_deleted` projections over the Push/Residual path and id-less Parquet
   files fail loud with `FeatureUnsupported` naming the dropped ids when such
   deletes are present: loud by design.
3. The per-file bind schema is transient (never persisted). A future
   name-based consumer of stored eq-delete predicates must resolve through
   the delete file's own schema, not the current table schema.

## Fold-2 — rebase onto `56aa8913`

`git rebase origin/main` replayed the three fold-1 commits. Conflicts and their
resolution:

- `caching_delete_file_loader.rs`, equality arm: #375 replaced the Parquet
  opener with `equality_delete_batch_stream(task, ..)`; this branch opened the
  Parquet file directly for its per-file schema. Resolved by making
  `equality_delete_batch_stream` return the stream and the per-file schema
  for every format, so #375's dispatch and Puffin refusal survive.
- `delete_file_loader.rs`, imports and `read_delete_file`: both sides kept;
  the trait door routes equality deletes through the same dispatched
  function, position deletes through `position_delete_batch_stream`.
- `scripts/check_rust_file_size.py`: the lower of each ceiling pair
  (caching loader 3157, delete filter 2915).
- `arrow/map.md`, `tests/map.md`: rows from both sides kept.

At the replayed head the Avro and ORC equality arms still used #375's
current-schema expected schema (the interim refusal of a dropped key).
Fold-2 item 0 (`d6f0180b`) routes them through the per-file resolution:
the delete file's own schema comes from the Avro writer schema or the ORC
footer (`orc_top_level_schema`, which now reads the root struct's names and
the `iceberg.*` type attributes), a dropped key is recovered from it, and
neither arm returns an empty stream for an unresolvable key. The resolver moves
to `arrow/eq_delete_schema.rs` (the loader would otherwise pass its 1000-line
ceiling); the caching-loader ceiling lowers to 3130. Before any new work
`eqdel_dropped_key` was 3/3 and `mor_eqdel_format` 4/4 on the replayed head.

## Verifier round 1 (Opus 5.5 at `40fa72f7`): FAIL

- **V-EQKEY-1, S1.** With a key the equality fast path does not take (a
  `decimal` key, or a composite key with a NULL in an earlier column), a scan
  projecting `_pos` or `_row_id` after the key is dropped answered all five
  rows: the whole-file projection path returns before `plan_row_filter`, so
  the guards never ran, and `evaluate_predicate_to_mask` read the absent
  column as all-NULL under `col != v`. The base failed loud. The DELETE,
  UPDATE and MERGE position-delete writers scan with `_pos` and `_file`.
- **V-EQKEY-2, S1.** A spec-shaped equality-delete file keyed on a nested
  field (`s.k`, equality id of the child) was lifted to the top level by the
  per-file schema; the transformer filled it with NULL and the delete became
  `k IS NULL`: the targets were kept and the NULL-key row deleted. The base
  failed loud.
- **V-EQKEY-3, S2, pre-existing.** An equality id present in the table but
  absent from the delete file's own schema read as NULL and over-deleted.
- **V-EQKEY-4, S3.** The ledger's "no silent flips" and "the `_pos` path
  errors as at base" sentences were refuted by V-EQKEY-1 (corrected above in
  Follow-ups).
- Side observation: dropping the only child of a struct and then scanning
  panics in the transformer (`StructArray::try_new` unwrap), same on the base.

## Fold-2 clauses

| Id | Claim | Verdict | Pins |
|---|---|---|---|
| F2-0 | Avro and ORC equality-delete files resolve a dropped key from their own schema and answer the survivors | PROVEN | `test_avro_equality_delete_with_dropped_key_answers_survivors`, `test_orc_equality_delete_with_dropped_key_answers_survivors` (`mor_eqdel_format.rs`; were the interim refusal pins, red against the replayed head with `DataInvalid ... absent from the current schema`) |
| F2-1 | V-EQKEY-1: the `_pos` / `_row_id` whole-file path applies a dropped equality key by field id | PROVEN | `decimal_key_dropped_pos_projection_answers_survivors`, `decimal_key_dropped_row_id_projection_answers_survivors`, `composite_key_with_null_first_column_dropped_pos_projection_answers_survivors`, `long_key_dropped_pos_projection_keeps_post_drop_rows` |
| F2-2 | V-EQKEY-2: a nested key keeps its struct path; nothing is flattened | PROVEN | `nested_key_spec_shaped_delete_answers_survivors`, `nested_key_rust_writer_delete_answers_survivors` (no-regression), `nested_key_with_struct_dropped_is_loud_or_correct` |
| F2-3 | V-EQKEY-3: an equality id absent from the delete file refuses `DataInvalid` naming the file and the id, in every format | PROVEN | `parquet_delete_file_missing_key_column_refuses`, `avro_delete_file_missing_key_column_refuses`, `orc_delete_file_missing_key_column_refuses` |
| F2-4 | V-EQKEY-4: the ledger's refuted sentences are corrected | PROVEN | Follow-ups above |
| F2-5 | No neighbouring answer changed | PROVEN | Gates below: `iceberg` lib 4642/0 (base fold-1 4639 plus #375's 3), every integration binary 0 failed, `iceberg-datafusion` 802/0 |
| F2-6 | Dropping the only child of a struct then scanning panics in the transformer (`StructArray::try_new` unwrap), same on the base | OPEN | Carded on the RePark side; not touched here |
| F2-7 | A Rust `EqualityDeleteWriterConfig` keyed on a nested id writes the key flattened to the top level, not under its struct (the Java shape) | OPEN | Read path accepts both shapes (F2-2); writer shape unchanged |
| F2-8 | A key promoted `int` to `long` after the delete fails loud under any `_pos` or `_row_id` projection, and DataFusion merge-on-read DELETE and UPDATE on such tables leave the table unchanged; pre-existing on the base and on fold-1 | OPEN | Scans without `_pos` answer correctly; promoting then dropping the key is correct; carded on the RePark side: the key-set fast path should compare after type promotion or fall back to the predicate path |
| F2-9 | A key inside a struct inside a list is loud with the wrong message | OPEN | `DataInvalid` naming the file and the id as absent from the delete file's own schema, though the column is in the file (`parquet_key_schema` does not descend into lists); carded on the RePark side |

### Design choice for ruling 1

The first option was taken: the dropped key is read from the data file as a
filter-only column by field id. `row_filter_plan::dropped_equality_key_mask`
maps every equality id absent from the task schema to the data file's
Parquet leaves through the embedded field ids (the map the Push row filter
uses), and the `_pos` / `_row_id` branch of `reader.rs` unions those leaves
into its own file-level projection mask. That is the reader's own file-level
read, not the scan projection: `FileScanTask`, `scan/` and the output
transformer are untouched, and the extra column never reaches the output
batch. A data file without embedded ids refuses with `FeatureUnsupported`.
The keyset fast path now returns `None` (predicate oracle) instead of an
error when a key column is absent from the batch, so a post-drop data file
keeps its rows (the absent column is all-NULL under `col != v`) and a nested
key, which the keyset resolves only at the top level, goes to the
path-resolving predicate. `eq_delete_key_fields_projected` moved from
`reader.rs` to `equality_delete_set.rs` (comments shed) to keep `reader.rs`
under its ceiling, which lowers to 9547.

### Design of the per-file schema (rulings 2 and 3)

`eq_delete_schema::equality_delete_schema` compares each equality id's struct
path in the current schema with its path in the delete file's own schema:

| Current schema | Delete file | Per-file entry |
|---|---|---|
| present, same path | present | the current schema pruned to the id, struct path kept |
| present, nested | top level (Rust writer) | a top-level entry with the current definition |
| absent | top level | a top-level entry recovered from the file |
| absent | nested | `FeatureUnsupported` |
| any | absent | `DataInvalid` naming the file and the id |
| present, another struct path | | `FeatureUnsupported` |
| present, inside a list or map | present in the file | loud with `DataInvalid` naming the file and the id as absent from the delete file's own schema, though the column is in the file (`parquet_key_schema` does not descend into lists); wrong message (OPEN, F2-9) |

The delete stream is evolved on the per-file schema's top-level ids, and each
file's predicate binds against it.

## Fold-2 pins

`crates/iceberg/tests/eqdel_key_paths.rs` (10 tests, new) and the two
updated `mor_eqdel_format.rs` pins, listed in the clause table.

## Fold-2 red runs

Run R1, the new and updated pins on the replayed head before any fold-2
code: `eqdel_key_paths` 1 passed, 8 failed (the decimal `_pos` and `_row_id`
pins and the composite pin answered all five rows; the spec-shaped nested
pin answered `a, b, c, d, e` (targets kept, NULL-key row `f` deleted); the
struct-dropped pin answered wrong rows; the three missing-key pins answered
`a, c` (the NULL-id row over-deleted)); the Rust-writer nested pin passed
(no-regression); `mor_eqdel_format` 2 passed, 2 failed (`DataInvalid ...
absent from the current schema`).

Run R2, the same with the long-key post-drop pin and the nested rename added,
fold-2 sources stashed back to the replayed head: `eqdel_key_paths` 1 passed,
9 failed; the long-key pin failed loud with `Unexpected => equality-delete
set fast path: key column 'id' (field id 1) is absent from the data batch`.

Per-commit greens: after item 0, `mor_eqdel_format` 4/4 and
`eqdel_key_paths` 1/10; after item 1, 5/10 (the four `_pos` / `_row_id`
pins green); after item 2, 7/10 (nested pins green); after item 3, 10/10.

## Fold-2 gates (measured at `1afda5ec`)

- `cargo test -p iceberg --test eqdel_dropped_key`: exit 0, 3 passed.
- `cargo test -p iceberg --test eqdel_key_paths`: exit 0, 10 passed.
- `cargo test -p iceberg --test mor_eqdel_format`: exit 0, 4 passed.
- `cargo test -p iceberg`: exit 0; lib 4642 passed, 0 failed, 9 ignored;
  70 result lines in all, 5001 passed, 0 failed, 21 ignored.
- `cargo test -p iceberg-datafusion`: exit 0; lib 463 passed, 0 failed,
  5 ignored; 43 result lines in all, 802 passed, 0 failed, 15 ignored.
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings`: exit 0.
- `cargo fmt --all -- --check`: exit 0.
- `python3 scripts/check_rust_file_size.py`: exit 0, 704 files clean.
- Comment gate: 0 added comment lines in every fold-2 commit beyond the
  Apache licence headers of the two new files.

## Verifier round 2 (Opus 5.5 at `60b0f3da`): PASS, no S1

Every round-1 shape and every hunt answered the survivors: decimal,
null-carrying composite and plain long keys under `[data]`, `_pos`, `_file`
and `_row_id` projections before and after the drop; post-drop rows kept; a
re-added id with new rows; a key present in some data files and absent in
others; `_pos` with a row-selection filter; an identity-partitioned table;
two-level nested keys under renames and sibling drops; the Rust-writer
flattened shape; dropped single and composite keys through Parquet, Avro and
ORC delete files; a key column under a different name with the right field
id applies, the right name with a wrong id refuses; DataFusion merge-on-read
DELETE and UPDATE on long and decimal keys before and after the drop, 16
cases correct.

Red on the fold-1 head: `eqdel_key_paths` 1 passed, 9 failed, with the exact
wrong-row shapes of round 1.

V2-EQKEY-1 (S2, pre-existing on the base and on fold-1): a key promoted
`int` to `long` after the delete fails loud under any `_pos` or `_row_id`
projection (`equality-delete set fast path: expected Int64Array, got
Int32`), which also fails DataFusion merge-on-read DELETE and UPDATE on such
tables with the table unchanged; scans without `_pos` answer correctly;
promoting then dropping the key is correct. Card-worthy on the RePark side:
the key-set fast path should compare after type promotion or fall back to
the predicate path. Recorded as OPEN F2-8; no code touched here.

V2-EQKEY-2 (S3): the fold-1 Debug row (`crates/iceberg/src/arrow/map.md`
line 92) still says the Avro, ORC and `_pos` paths reject a schema-absent id
loudly and only the Parquet pushdown path applies dropped-key deletes; after
fold-2 only Avro and ORC DATA files stay loud (`DataInvalid: Projected field
id ... is not present in the scan schema`). The map row is narrowed
accordingly in this commit.

V2-EQKEY-3 (S3): the fold-1 "The fix" text above (lines 60 and 83) still
says the per-file schema is built by
`BasicDeleteFileLoader::equality_delete_schema` in `delete_file_loader.rs`,
and that there is no `reader.rs` change. Both lines are superseded, and the
fold-1 text above is left as written: the resolver now lives in
`eq_delete_schema::equality_delete_schema` (see "Design of the per-file
schema"), and fold-2 item 1 changed `reader.rs` (see "Design choice for
ruling 1").

V2-EQKEY-4 (S3): a key inside a struct inside a list is loud, but with
`DataInvalid` naming the file and the id as absent from the delete file's
own schema, though the column is in the file (`parquet_key_schema` does not
descend into lists); the per-file schema table above said
`FeatureUnsupported` naming the list path. The table row is re-worded to
what is measured, and the misleading error kind is recorded as OPEN F2-9; no
code touched here.
