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

# Ledger — F-LIST-NULL-ACCESSOR-2: compound predicates over a nested column reach `bind` unguarded

**Ledger id:** `F-LIST-NULL-ACCESSOR-2-2026-09-18`
**Branch:** `fix/f-list-null-accessor-2` (cut off fork `main` = `29ea7f6d`)
**Scope:** devin-worker brief F-LIST-NULL-ACCESSOR-2 (follows defect #299, `9f36da97`)
**Model:** swe-2
**Found by:** RePark run 23b pinned to fork `50350e33` — copy-on-write `DELETE` with
`id > 1 AND xs IS NULL` or `xs IS NULL OR id = 1` failed `DataInvalid => Accessor for
Field xs not found` for `xs` a `list<int>`, `list<struct>`, `map` or `struct` column,
v2 and v3; merge-on-read and UPDATE were fine; Spark 4.1.2 answered every cell.

## 1. Defect

#299 (`9f36da97`, F-LIST-NULL-ACCESSOR-1) put the soundness gate inside the DataFusion
converter: `predicate_binds_soundly` (`expr_to_predicate.rs`) drops any pushed-down
term whose `Reference` has no `Schema::accessor_by_field_id` — list/map/struct roots
and collection children have no accessors (`f-list-null-accessor-1-ledger.md` §1),
so `Reference::bind` (`expr/term.rs:326`) raises `DataInvalid => Accessor for Field
<name> not found`.

That gate only covers predicates that pass through `convert_filters_to_predicate`.
Iceberg's public `Predicate`-accepting surfaces take a raw `Predicate` and hand it to
`Predicate::bind` directly, with no equivalent check:

- `TableScanBuilder::with_filter` / `with_file_prune_only` → `scan/mod.rs` binds
  `self.filter` at `build()`.
- `IncrementalAppendScanBuilder::with_filter` → `scan/incremental.rs`, same shape.
- `conflict_detection_filter` on `OverwriteFilesAction` / `RowDeltaAction` →
  `transaction/snapshot/conflict_filter.rs` `first_conflicting_file` and the two
  `row_delta.rs` validation binds.

Any caller that builds `id > 1 AND xs IS NULL` itself — RePark's copy-on-write
affected-file/conflict path does — skips the converter entirely and raises at bind.
The fork's own SQL cells passed because the DataFusion converter was already sound.

### Reproducing shape

The defect is **entry-point**, not fixture, shaped. Every SQL-driven fixture axis was
varied and none reproduced it: four single-row `INSERT`s (four data files, the RePark
seed shape), v3 tables, `identity(id)` partitioning, `target_partitions` > 1, and
`write.metadata.metrics.default` on/off — all green, because `delete_from` routes the
filters through `convert_filters_to_predicate` before they reach any bind. The red
cells therefore drive the public APIs the way a non-DataFusion caller does:

- `table.scan().with_file_prune_only(id > 1 AND xs IS NULL).build().plan_files()`
- `table.scan().with_filter(id > 1 AND xs IS NULL).build().plan_files()`
- `first_conflicting_file(files, table, Some(id > 1 AND xs IS NULL), true)`

Each raised `DataInvalid => Accessor for Field xs not found` pre-fix, at
`TableScanBuilder::build` (`scan/mod.rs`) and `first_conflicting_file`'s bind
(`conflict_filter.rs`) respectively.

## 2. Audit — every predicate conversion/binding site

| Site | Path | Predicate source | Gate |
|---|---|---|---|
| `convert_filters_to_predicate` / `predicate_binds_soundly` / `term_binds_soundly` / `literal_binds_soundly` | `crates/integrations/datafusion/src/physical_plan/expr_to_predicate.rs` | DataFusion `Expr` filters | SOUND — drops unbindable terms since #299; `And`/`Or` recurse |
| `scan_predicates` (SELECT pushdown) | `crates/integrations/datafusion/src/physical_plan/scan.rs` | same converter | SOUND |
| `rebind_filters` | `expr_to_predicate.rs` | same converter | SOUND |
| `IcebergTableProvider::delete_from` → `prune` | `table/mod.rs` `delete_from` → `IcebergDeleteExec` → `cow_scan_stream` / `mor_scan_stream` | converter output | SOUND; the exact `PhysicalExpr` stays the row contract, Iceberg side is prune-only |
| `IcebergTableProvider::update` → `prune` | `table/mod.rs` `update` | converter output | SOUND, same construction |
| `conflict_detection_filter(prune)` on the DML commit path | `delete.rs` / `update.rs` → `OverwriteFilesAction` | converter output | SOUND upstream of this fix |
| `TableScanBuilder::build` | `crates/iceberg/src/scan/mod.rs` `snapshot_bound_predicate` | raw `Predicate` (`with_filter`, `with_file_prune_only`) | FIXED — now `bind_pruning` |
| `IncrementalAppendScanBuilder::build` | `crates/iceberg/src/scan/incremental.rs` `snapshot_bound_predicate` | raw `Predicate` (`with_filter`) | FIXED — now `bind_pruning` |
| `first_conflicting_file` | `crates/iceberg/src/transaction/snapshot/conflict_filter.rs` | raw `conflict_detection_filter` | FIXED — `.rewrite_not().bind_pruning(...)` |
| added-DV conflict validation | `crates/iceberg/src/transaction/row_delta.rs` (`validate_added_dvs`) | raw `conflict_detection_filter` | FIXED — `bind_pruning` |
| removed-data-files delete validation | `crates/iceberg/src/transaction/row_delta.rs` (`validate_no_new_deletes_for_data_files_on`) | raw `conflict_detection_filter` | FIXED — `bind_pruning` |
| residual bind per task | `crates/iceberg/src/scan/context.rs` | residual of the already-bound snapshot predicate | n/a — always bindable once the snapshot predicate is sanitized |
| `PartitionFilterCache` / `ExpressionEvaluatorCache` / `ManifestEvaluatorCache` rebinds | `crates/iceberg/src/scan/cache.rs` | projections of bound predicates | n/a |
| equality-delete row predicate | `crates/iceberg/src/arrow/delete_filter.rs` | built internally from delete-file equality ids | n/a — internally constructed, always bindable |
| `record_batch_predicate` | `crates/iceberg/src/arrow/record_batch_predicate.rs` | bound predicates | n/a |
| `check_added_files_match_overwrite_filter` | `crates/iceberg/src/transaction/overwrite_files.rs` | raw `row_filter()` | UNCHANGED — see §6 |
| maintenance binds (`rewrite_position_delete_files`, `convert_equality_delete_files`), `table_metadata_builder` | `crates/iceberg/src/maintenance/`, `src/spec/` | internally built | n/a |

## 3. Choice — `Predicate::drop_unbindable_terms` + `Predicate::bind_pruning`

New file `crates/iceberg/src/expr/sanitize.rs` (`scan/mod.rs`, `incremental.rs`,
`row_delta.rs`, `predicate.rs` are all at frozen legacy line ceilings — every edit
there had to be line-neutral, so the sanitizer lives in a new module and each call
site is a one-line swap to `bind_pruning`).

`drop_unbindable_terms(&self, schema, case_sensitive) -> Predicate` rewrites a
predicate conservatively for **prune** use:

- A leaf (`Unary`/`Binary`/`Set`) whose `Reference` resolves to an existing field
  that has **no accessor** becomes `AlwaysTrue`. `And`/`Or` fold via the existing
  combinators, so under `AND` the dropped term vanishes and the sound conjunct is
  kept (`id > 1 AND xs IS NULL` → `id > 1` — a file that cannot match `id > 1`
  cannot match the conjunction either, so the prune stays exact); under `OR` the
  `AlwaysTrue` absorbs (`xs IS NULL OR id = 1` → `AlwaysTrue` — keeping only the
  sound disjunct would wrongly exclude files the unbindable disjunct might match).
- A `Not` node whose subtree contains any unbindable term becomes `AlwaysTrue` —
  a negation cannot be partially salvaged without De Morgan rewriting, and widening
  stays a sound over-approximation.
- A leaf whose name **does not resolve** (missing field, wrong case) is kept, so
  `bind` still fails loudly — the row-delta case-sensitivity contract
  (`test_row_delta_*_wrong_case_fails_to_bind`, 9 tests) is preserved verbatim.
  Type-mismatched literals likewise stay loud.
- `bind_pruning(schema, case_sensitive)` = `drop_unbindable_terms` then `bind` —
  the single call the five prune/conflict sites now make.

The exact row contract is untouched: in the DataFusion DML path the
`PhysicalExpr` built from the original filters still decides which rows delete;
the Iceberg predicate was and remains prune-only.

## 4. Red (commit `b32b218c`)

| Cell | File | Pre-fix failure |
|---|---|---|
| `cow_prune_scan_drops_the_unbindable_null_term` | `crates/integrations/datafusion/src/physical_plan/list_null_tests.rs` | `DataInvalid => Accessor for Field xs not found` at `build()` |
| `filtered_scan_residual_drops_the_unbindable_null_term` | same | same, at `build()` |
| `unbindable_conflict_filter_widens_instead_of_failing_to_bind` | `crates/iceberg/src/transaction/snapshot/conflict_filter.rs` | same, at `bind` |

Fixture: `NullFixture` now retains `Arc<MemoryCatalog>` so the test loads the table
and drives `Table` APIs directly; `seeded_per_row` issues one `INSERT` per row so the
four rows land in four data files and file-level pruning is observable; `NullShape::ALL`
(list<int>, list<struct>, map, struct) × `FormatVersion::{V2,V3}`.

## 5. Green (commit `c9e24310`)

- `cow_prune_scan_drops_the_unbindable_null_term`: the AND prune plans exactly 3
  tasks (the `id = 1` file is pruned by its metrics — the sound conjunct still
  discriminates); the OR prune plans all 4 (conservative widening).
- `filtered_scan_residual_drops_the_unbindable_null_term`: every task's residual
  equals bound `id > 1` — the unbindable term never reaches the residual either.
- `unbindable_conflict_filter_widens_instead_of_failing_to_bind`: a data file with
  `upper(id)=4` stays conflicting, `upper(id)=1` does not — the sound conjunct still
  narrows conflict detection.
- 9 `expr::sanitize` unit tests pin the rewrite contract (AND keeps, OR widens,
  OR-inside-AND, NOT leaf and NOT compound widen, missing field and wrong-case stay
  bind errors, bindable predicate unchanged, `bind_pruning` binds).
- Surviving-ids matrix is held by the existing SQL cells (all 12 `list_null` tests
  green): `delete_where_id_and_xs_is_null_composes_with_a_primitive_conjunct` →
  survivors `[1,3,4]`; `delete_where_xs_is_null_or_id_eq_1_keeps_matching_rows` →
  `[3,4]`; each across all four shapes, v2+v3, copy-on-write and merge-on-read.
- Regression: `row_delta` 120/120, `incremental` 39/39, `predicate` 187/187,
  `conflict_filter` 9/9, `list_null` 12/12.

## 6. Intentionally unchanged

- `OverwriteFilesAction::row_filter` / `check_added_files_match_overwrite_filter`
  (`transaction/overwrite_files.rs`): that bind enforces an **exact-match**
  contract — added files must verify against the declared overwrite filter.
  Widening an unverifiable term to `AlwaysTrue` would silently accept files the
  filter cannot check, weakening a correctness validation rather than a prune.
  Strict bind stays; a caller can sanitize before setting the filter if it means
  prune semantics.
- `Expr`/`Predicate` conversion in `expr_to_predicate.rs`: already sound since
  #299; unchanged.
- `PlanContext.predicate` (raw filter kept for `ScanEvent` observability): never
  bound; unchanged.
- `FileScanTask::split`: propagates the already-bound task predicate; no bind;
  unchanged.

## 7. Mutation proof

Revert = `git checkout b32b218c --` the five fix files + remove `expr/sanitize.rs`
(tests intact):

- `cow_prune_scan_drops_the_unbindable_null_term` — RED, `DataInvalid => Accessor
  for Field xs not found`.
- `filtered_scan_residual_drops_the_unbindable_null_term` — RED, same.
- `unbindable_conflict_filter_widens_instead_of_failing_to_bind` — RED, same.

Restore = `git checkout HEAD --` the same files:

- All three cells GREEN; the 9 `expr::sanitize` unit tests GREEN.

## 8. Notes for the next round

- The first sanitizer draft widened **any** bind failure and regressed four
  `row_delta` wrong-case tests on its first run — the field-resolves-but-no-accessor
  split (§3) is the contract, not an implementation detail.
- `Not` subtrees are deliberately widened whole rather than De Morgan-rewritten:
  `Predicate::rewrite_not` exists if a future caller needs the tighter form, but
  every current prune site accepts the conservative result.
