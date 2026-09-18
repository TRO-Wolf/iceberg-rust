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

# Ledger — F-OCC-EXEC-1: the DataFusion DELETE/UPDATE exec scopes conflict validation by its own scan filter

**Ledger id:** `F-OCC-EXEC-1-2026-09-18`
**Branch:** `fix/occ-exec-1` (cut off `origin/main` = `8fb44a39`)
**Scope:** devin-worker brief F-OCC-EXEC-1
**Model:** swe-2
**Extends:** `task/f-occ-scoped-1-ledger.md` (fork #291 armed the `conflict_detection_filter`
seam; this unit makes the engine side pass its own scan filter through it).

## 1. Commit-path map (fork, at `8fb44a39`)

| Step | File:line | What lives there |
|---|---|---|
| DML planning | `crates/integrations/datafusion/src/table/mod.rs` | `IcebergTableProvider::delete_from` / `.update` build `prune = convert_filters_to_predicate(&filters)` and the exact DataFusion row `predicate`, then construct `IcebergDeleteExec` / `IcebergUpdateExec` carrying both |
| Filter → predicate | `crates/integrations/datafusion/src/physical_plan/expr_to_predicate.rs` | `convert_filters_to_predicate`: each supported `Expr` converts; unsupported ones are dropped; successes combine with `AND`. `None` only when no filter was supplied |
| Target scan (MoR) | `crates/integrations/datafusion/src/physical_plan/mor_scan.rs` | `mor_scan_stream` → `table.scan().with_file_prune_only(prune)` |
| Target scan (CoW) | `crates/integrations/datafusion/src/physical_plan/row_lineage.rs` | `cow_scan_stream` → `table.scan().with_file_prune_only(prune)` |
| Prune-only scan | `crates/iceberg/src/scan/mod.rs` | `with_file_prune_only` runs `predicate.rewrite_not()` and uses it for manifest/file pruning only; it attaches no residual row filter, so every row of a surviving file reaches the exact DataFusion predicate |
| MoR DELETE commit | `crates/integrations/datafusion/src/physical_plan/delete.rs:~532` | `row_delta().conflict_detection_filter(Predicate::AlwaysTrue).validate_data_files_exist(..)[.validate_deleted_files()].validate_no_conflicting_data_files()` |
| CoW DELETE commit | `crates/integrations/datafusion/src/physical_plan/delete.rs:~671` | `overwrite_files().conflict_detection_filter(Predicate::AlwaysTrue).validate_no_conflicting_deletes()[.validate_no_conflicting_data()]` |
| MoR UPDATE commit | `crates/integrations/datafusion/src/physical_plan/delete.rs:~988` | `row_delta().conflict_detection_filter(Predicate::AlwaysTrue).validate_data_files_exist(..).validate_deleted_files().validate_no_conflicting_delete_files()[.validate_no_conflicting_data_files()]` |
| CoW UPDATE commit | `crates/integrations/datafusion/src/physical_plan/delete.rs:~1124` | `overwrite_files().conflict_detection_filter(Predicate::AlwaysTrue).validate_no_conflicting_deletes()[.validate_no_conflicting_data()]` |
| Filter-scoped check | `crates/iceberg/src/transaction/snapshot/conflict_filter.rs` | `first_conflicting_file`: partition projection per spec → `InclusiveMetricsEvaluator`; `None` filter = `AlwaysTrue`; unknown spec fails closed |

## 2. Java rule matched

`SparkPositionDeltaWrite` / `SparkCopyOnWriteOperation`: the serializable conflict-detection
filter is the conjunction of the filters pushed into the operation's scan; `alwaysTrue` only
when nothing was pushed. RePark run 21a (Spark 4.1.2 + Iceberg 1.11.0): four concurrent
partition-local `UPDATE … WHERE k = '<key>' AND id < 8` on `PARTITIONED BY (k)` commit 4/4 in
Spark, 1/4 in RePark — the three losers fail
`Found conflicting files that can contain records matching TRUE` naming another partition's file.

## 3. Defect

Each of the four commit sites hard-codes `Predicate::AlwaysTrue` although the exec already
carries `prune: Option<Predicate>` — the same predicate the exec pushes into its target scan
for file pruning. Under `serializable` every concurrently added data file therefore
might-matches and the commit aborts, even for provably disjoint partitions (the F-OCC-SCOPED-1
partition-projection gate never sees the real filter).

## 4. Red tests (step 2)

Harness: `crates/integrations/datafusion/src/physical_plan/occ_exec_tests.rs` (new file —
`delete_tests.rs` is at 986/1000 lines). Wired as `mod occ_exec_tests` inside `delete.rs`.
Fixture: memory catalog, V2 table `PARTITIONED BY (k)`, seeded rows in partitions `a` and `b`
(`mor` flag selects merge-on-read vs copy-on-write properties). Race injection plans the DML
through the real `TableProvider` entry point (`delete_from` / `update`), commits a concurrent
transaction against the same base snapshot, then executes the plan. `WHERE k = 'a' AND id < 8`
gives `prune = k = 'a' AND id < 8`.

`cargo test -p iceberg-datafusion --lib occ_exec_tests` before the fix:

```text
test result: FAILED. 9 passed; 4 failed; 0 ignored; 0 measured; 229 filtered out

failures:
    physical_plan::delete::occ_exec_tests::cow_delete_disjoint_partition_commit_commits
    physical_plan::delete::occ_exec_tests::cow_update_disjoint_partition_commit_commits
    physical_plan::delete::occ_exec_tests::mor_delete_disjoint_partition_commit_commits
    physical_plan::delete::occ_exec_tests::mor_update_disjoint_partition_commit_commits
```

Each disjoint case fails with the defect's own message:

```text
a concurrent commit in a disjoint partition must not conflict the k = 'a' <DML>:
External(DataInvalid => Found conflicting files that can contain records matching TRUE:
test/b-new.parquet)
```

The 4 disjoint cases inject a concurrent append into `b` plus — for the two MoR paths — a
concurrent `b` position-delete file in the same `row_delta` commit. The 9 passing controls:

- `mor_delete_matching_partition_commit_conflicts` — concurrent append into `a` aborts.
- `cow_delete_matching_partition_commit_conflicts` — same.
- `mor_update_matching_partition_commit_conflicts` — same.
- `cow_update_matching_partition_commit_conflicts` — same.
- `mor_update_matching_partition_delete_file_conflicts` — concurrent `a` delete file aborts
  the UPDATE (the UPDATE's `row_delta` arms `validate_no_conflicting_delete_files`; check 2a
  catches it through the rewritten `a` data files).
- `mor_delete_no_predicate_keeps_always_true` — `DELETE FROM t` aborts on any concurrent
  commit, error contains `matching TRUE`.
- `cow_delete_no_predicate_keeps_always_true` — same.
- `mor_update_no_predicate_keeps_always_true` — `UPDATE` with no filter aborts, `matching TRUE`.
- `cow_update_no_predicate_keeps_always_true` — same.

Observed while probing controls: the MoR DELETE commit does not arm
`validate_no_conflicting_delete_files` (`delete.rs:532-543`), so a concurrent `a` delete file
never conflicts it — a test asserting that conflict was dropped; that is the armed surface,
not a defect of this unit.

## 5. Fix (step 3)

All four commit sites now pass the exec's own scan predicate:

```rust
.conflict_detection_filter(prune.unwrap_or(Predicate::AlwaysTrue))
```

- `merge_on_read_delete` and `merge_on_read_update` pass `prune.clone()` into
  `mor_scan_stream` and consume the original at the commit site.
- `copy_on_write_delete` and `copy_on_write_update` already cloned `prune` into both
  `cow_scan_stream` calls; the commit site consumes the original.
- The two stale comments were edited in place (same line counts): the module doc's
  "Conflict filter is `AlwaysTrue`" and the MoR-DELETE recipe's "`AlwaysTrue` is Java-exact
  because this path pushes no filter". Both now state the filter is the scan's `prune`,
  `AlwaysTrue` when nothing was pushed.
- `case_sensitive` needs no threading: the scan builder defaults `case_sensitive: true`
  (`scan/mod.rs`) and `RowDeltaAction` / `OverwriteFilesAction` default the same
  (`row_delta.rs:159` and the overwrite sibling), so the predicate binds identically on
  both sides.

### Soundness of `prune` as the conflict filter

`prune` is `convert_filters_to_predicate(&filters)` (`expr_to_predicate.rs`): each DataFusion
`Expr` that converts contributes one conjunct; anything that cannot convert (type-promoted
comparisons, NaN-sensitive comparisons, unsupported function forms, one side of a partial
`AND`) is DROPPED, never approximated downward. A conjunction can only lose clauses, so the
result is logically weaker than or equal to the exact `WHERE` — a superset of the rows the
DML reads. `None` iff the caller pushed no filter, which is exactly Java's `alwaysTrue`
case.

The scan consumes it through `with_file_prune_only` (`scan/mod.rs`): `rewrite_not()` pushes
`NOT` inward before binding, the predicate prunes manifests/files with INCLUSIVE evaluators
(partition projection, then `InclusiveMetricsEvaluator` — the same evaluators
`first_conflicting_file` runs on the validation side), and no residual row filter is
attached. Every row of every surviving file reaches the exact DataFusion `predicate`, which
alone decides matches. So `prune` cannot exclude a file whose rows the DML can match — it
is a superset filter by construction, and passing it to `conflict_detection_filter` narrows
concurrent-file validation to the same file set the operation actually read. No
construction of `prune` can be narrower than the rows read; no `AlwaysTrue` fallback is
needed.

## 6. Existing pins updated (step 4 fallout)

Three `tests/integration_datafusion_test.rs` pins asserted the defect itself: an
unpartitioned table, `WHERE foo1 = 1`, concurrent `INSERT (3,'c')` — the appended file's
metrics cannot match `foo1 = 1`, so Java (`filterData` → inclusive metrics) commits. Under
`AlwaysTrue` they rejected; under the fix they committed and the tests failed:

- `test_s5_cow_delete_serializable_default_rejects_concurrent_append`
- `test_s5_cow_update_serializable_default_rejects_concurrent_append`
- `test_s5_merge_on_read_delete_serializable_default_rejects_concurrent_append`

Each now inserts `(1,'c')` — a file whose metrics DO match the filter — so the rejection
pin is preserved as a same-scope control rather than deleted. The
"matching the AlwaysTrue conflict filter" comment was edited to stay true. No other test
carried the premise (`INSERT OVERWRITE`'s row filter is genuinely `AlwaysTrue`).

Clippy (`err_expect`) rewrote `.err().expect(..)` to `.expect_err(..)`; `cargo fmt`
re-wrapped; `mod occ_exec_tests` moved from `delete.rs` (1149-line legacy ceiling, +4 over)
to `physical_plan/mod.rs` where the sibling file needs no `#[path]`.

## 7. Gates (step 4)

- `CARGO_BUILD_JOBS=10 cargo test -p iceberg-datafusion`: 521 passed, 0 failed, 12 ignored
  (241 lib + 34 test targets + doctests).
- `cargo test -p iceberg --lib conflict`: 70 passed, 0 failed.
- `make check`: fmt clean; clippy `-D warnings` clean; taplo clean; cargo-machete clean;
  `check_agent_artifacts` OK; `check_matrix_anchors` OK (88 rows); `check_comment_blocks` OK;
  `check_rust_file_size` 495 files clean.
- `python3 /tmp/oc-worker/_lib/comment_ban.py . origin/main HEAD`: hits=22 — 16 ASF-header
  lines in the new test file + 6 stay-true edits of existing comments (the two the brief
  names in `delete.rs`, one in `integration_datafusion_test.rs`).

## 8. Clauses

- C-001 PROVEN — MoR DELETE scopes validation by `prune`:
  `mor_delete_disjoint_partition_commit_commits` — RED
  (`Found conflicting files … matching TRUE: test/b-new.parquet`) → GREEN, `b` append +
  `b` delete file committed concurrently.
- C-002 PROVEN — CoW DELETE: `cow_delete_disjoint_partition_commit_commits` — same RED
  message → GREEN.
- C-003 PROVEN — MoR UPDATE: `mor_update_disjoint_partition_commit_commits` — same RED
  message → GREEN.
- C-004 PROVEN — CoW UPDATE: `cow_update_disjoint_partition_commit_commits` — same RED
  message → GREEN.
- C-005 PROVEN — the fix does not over-narrow (per-path control):
  `mor_delete_matching_partition_commit_conflicts`,
  `cow_delete_matching_partition_commit_conflicts`,
  `mor_update_matching_partition_commit_conflicts`,
  `cow_update_matching_partition_commit_conflicts` — concurrent `a` appends still abort,
  naming `a-new.parquet`; and `mor_update_matching_partition_delete_file_conflicts` — a
  concurrent `a` delete file still aborts the UPDATE.
- C-006 PROVEN — `prune == None` keeps `AlwaysTrue`:
  `mor_delete_no_predicate_keeps_always_true`,
  `cow_delete_no_predicate_keeps_always_true`,
  `mor_update_no_predicate_keeps_always_true`,
  `cow_update_no_predicate_keeps_always_true` — `DELETE FROM t` / predicate-less `UPDATE`
  abort on any concurrent commit, error contains `matching TRUE`.
- C-007 PROVEN — `prune` is a superset filter by construction (§5 soundness): conversion
  drops unconvertible conjuncts; `with_file_prune_only` is inclusive and attaches no
  residual; the exact DataFusion predicate remains the row contract. No fallback needed.
- C-008 PROVEN — `case_sensitive` parity: scan and action both bind case-sensitively by
  default; nothing threads an alternate setting.
- C-009 PROVEN — SQL-level controls still bite: the three s5 serializable pins reject a
  concurrent append whose file metrics match `foo1 = 1` (§6).

## Round 2 — F-01: a literal that does not convert exactly to the column type is not pushed

Commits: `d8dbaf944` (red cells), `85e30d001` (fix).

### Defect (critic F-01)

`DELETE FROM t WHERE f < 1e-50` on a `FLOAT` column `f`. DataFusion coerces the
comparison to `CAST(f AS DOUBLE) < Float64(1e-50)` — or leaves the column bare; the
logical plan observed in the e2e probe carried no CAST at all. Either way the
converter pushes `f < Datum::double(1e-50)`. At bind,
`Datum::to` (`datum.rs:1143`, Double→Float) bounds-checks to ±f32::MAX and then
`as f32`-narrows: `1e-50 → 0.0`, `1.00000001 → 1.0`, `0.1 → 0.10000000149…`.
The bound predicate can therefore be *stronger* than what was pushed — `f <
1.00000001` binds as `f < 1.0` — and the inclusive metrics evaluator prunes a file
whose only row satisfies the exact DataFusion predicate. Result: silent wrong
DELETE/UPDATE. Since round 1 the same predicate scopes serializable conflict
detection, the narrowing can also under-scope OCC and miss a real conflict.

The defect is not CAST-specific: any pushed literal whose `Datum::to` conversion
loses precision has the same failure shape (bare Float64 vs Float, Int64 vs
Float/Double, timestamp strings beyond microsecond precision).

### Site, and why

`convert_filters_to_predicate` now takes the Iceberg `&Schema` the produced
predicate will bind against and runs a `predicate_binds_soundly` post-pass over
every converted conjunct: each leaf's column must resolve in that schema, each
literal must convert under `Datum::to` without error, and a literal whose value
changes in conversion must round-trip exactly. `AboveMax`/`BelowMin` sentinels are
kept — binding folds them to the correct constant result.

Sites rejected:

- **`Datum::to` itself.** Java `Literals.*Literal.to` deliberately rounds
  (`datum.rs:1108-1110` ports that accept-set); other callers may rely on it.
  Pushdown is the only path that needs exactness.
- **The `Expr::Cast` arm.** The narrowing happens at bind against the *Iceberg*
  type, which the cast arm cannot see; a bare (uncasted) literal has the same bug.
- **Checking in `Expr` space.** A post-pass on the converted `Predicate` covers
  every producer uniformly: cast-stripped columns, bare literals, IN-list
  elements, and nested `AND`/`OR`/`NOT` (any unsound leaf drops the whole
  conjunct — partial drops inside `OR`/`NOT` would change semantics).

Callers pass the schema the predicate actually binds: `delete_from`/`update` use
`table.metadata().current_schema()`; `IcebergTableScan::new` selects the pinned
snapshot's schema when pinned, else the current schema — the same selection its
binding path makes.

### Exactness rule per pair (`converts_exactly`)

Only pairs where `Datum::to` rewrites the literal reach the table; unchanged
literals (`Int→Date`, `Long→Timestamp[tz]`, `Int128→Decimal`, `Binary↔Fixed`) are
accepted by identity, and a `Datum::to` error drops the conjunct.

| Pair | `Datum::to` does | Pushed iff |
|---|---|---|
| Int → Long / Int → Double / Float → Double | widening | always |
| Int → Float | `v as f32` | `(v as f32) as f64 == v` |
| Long → Int | `i64_to_i32` sentinels | in range; else sentinel kept |
| Long → Date | sentinel or `v as i32` | in range; else sentinel kept |
| Long → Float | `v as f32` | `(v as f32) as i128 == v` |
| Long → Double | `v as f64` | `(v as f64) as i128 == v` |
| Double → Float | sentinel beyond ±f32::MAX, else `v as f32` | `(v as f32) as f64 == v` |
| String → Timestamp/Timestamptz/Time | parse → micros `Long` | ≤ 6 sub-second digits (a 7th truncates) |
| String → Date / Uuid | parse or error | the parsed datum is canonical; parse errors drop |

The `Long→{Float,Double}` arms compare through `i128` because `v as f64` itself
rounds for |v| > 2^53 — comparing two rounded values would admit `2^53+1` as
"exact" (`cast_wrapped_double_column_with_long_literal_checks_exactness` pins
this). NaN never round-trips (`f64::from(NaN) != NaN`) so NaN literals drop — the
explicit `isnan` path stays the only NaN producer (nan tests green). `as f32`
preserves the sign of zero, so `±0.0` literals push with the sign intact, which
the totalOrder metrics comparison respects.

### Red output (commit `d8dbaf944`)

Unit — `cargo test -p iceberg-datafusion --lib expr_to_predicate`:

```text
test result: FAILED. 51 passed; 4 failed
    cast_wrapped_float_column_with_inexact_literal_is_not_pushed
      — pushed Some(f < Datum{Double(1e-50)}) where None required
    bare_float_column_with_inexact_literal_is_not_pushed
    cast_wrapped_float_column_in_list_with_inexact_element_is_not_pushed
    cast_wrapped_int_column_with_double_literal_is_not_pushed
```

E2E — the `f = 0.0` / `f < 1e-50` fixture was *masked*: the written file's lower
bound is `-0.0` and `Datum` comparison is IEEE totalOrder, so `-0.0 < 0.0` kept
the file. The load-bearing red uses `f = 1.0` vs `f < 1.00000001`
(`1.00000001 as f32` rounds to `1.0`; the bound `f < 1.0` prunes the file):

```text
test result: FAILED. 13 passed; 2 failed
    delete_where_float_lt_inexact_double_deletes_the_row — deleted 0, not 1
    update_where_float_lt_inexact_double_updates_the_row — updated 0, not 1
    (assertion: merge_on_read=true: f = 1.0 satisfies f < 1.00000001)
```

### Green output

```text
cargo test -p iceberg-datafusion --lib expr_to_predicate  → 56 passed, 0 failed
cargo test -p iceberg-datafusion --lib occ_exec_tests     → 15 passed, 0 failed
cargo test -p iceberg-datafusion --lib                    → 251 passed, 0 failed, 1 ignored
cargo test -p iceberg --lib expr                          → 406 passed, 0 failed
```

### Mutation output

Reverted only the fix (removed the `.filter(predicate_binds_soundly)` call; tests
and signature kept), uncommitted:

```text
expr_to_predicate: FAILED. 51 passed; 5 failed
    bare_float_column_with_inexact_literal_is_not_pushed
    cast_wrapped_double_column_with_long_literal_checks_exactness
    cast_wrapped_float_column_in_list_with_inexact_element_is_not_pushed
    cast_wrapped_float_column_with_inexact_literal_is_not_pushed
    cast_wrapped_int_column_with_double_literal_is_not_pushed
occ_exec_tests:    FAILED. 13 passed; 2 failed
    delete_where_float_lt_inexact_double_deletes_the_row
    update_where_float_lt_inexact_double_updates_the_row
```

Restored → 56 / 15 green.

### Cast-shape audit (every other conversion shape and its verdict)

- `CAST(foo AS INT64)` vs out-of-range Int64 literal → `i64_to_i32` sentinel →
  kept; binding folds `x < AboveMax` / `x > BelowMin` to the correct constant —
  pruning degenerates to keep-all, which is sound.
- `CAST(foo AS FLOAT64)` vs `Float64(2.5)`/`Float64(2.0)` → `Datum::to(Int)`
  errors → dropped. Previously the same shape was a loud mid-scan bind error;
  dropping is strictly safer and now also covers the OCC path.
- IN lists: one inexact element drops the whole set predicate — dropping a single
  element would change semantics.
- `String → Timestamp` beyond 6 sub-second digits → micros truncation → dropped;
  ≤ 6 digits pushed.
- Decimal literal vs Decimal column → `Datum::to` returns `self` (Java ignores
  target scale) → unchanged literal → pushed; Java-identical.
- `Long → Timestamp[tz]`, `Int → Date` reinterpret the i64/i32 verbatim → pushed.
- Any leaf on a column the binding schema lacks (unary included) → dropped;
  previously a bind error.
- ±0.0: sign survives `as f32` and the pushed datum carries it → pushed, correct
  under totalOrder metrics.
- No pre-existing test expectation was weakened; every prior test passes
  unmodified.

