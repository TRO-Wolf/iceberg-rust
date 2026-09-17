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

# F-ICE-NAN-PUSHDOWN-1 — NaN equality and IN push down as is_nan

**Date:** 2026-09-16. **Branch:** `fix/ice-nan-pushdown-1`.
**Base:** `edc38c6a`. **Model:** muse-spark-1.3-contributor.
**Path:** STANDARD because this changes the DataFusion-to-Iceberg filter
conversion and touches scan pruning.

This ledger retires when the fork change merges or the owner removes the unit.

## The defect

Measured 2026-09-16 on RePark main at fork pin `edc38c6a`: on an Iceberg
table with a double column `d` holding NaN, 1.0, NULL,
`SELECT id FROM t WHERE d = CAST('NaN' AS DOUBLE)` and
`WHERE d IN (CAST('NaN' AS DOUBLE))` return NO rows.
Spark 4.1.2 + Iceberg 1.11.0 return the NaN rows.
`isnan(d)`, `d <=> NaN` and `d = d` are correct, and the same query on a
non-Iceberg view is correct, so the rows are lost in the scan:
`to_iceberg_predicate` turns `d = NaN` into Iceberg `Eq(d, Datum::double(NaN))`,
and the manifest / row-group / row evaluators never match a NaN literal.
Pushdown is `TableProviderFilterPushDown::Inexact`, so over-inclusion is safe
and under-inclusion is the bug.

Java reference: Iceberg never builds a NaN literal (`Literals.from` rejects
NaN); Spark's filter conversion (`SparkV2Filters` / `SparkFilters`, Iceberg
1.11.0) rewrites equality with a NaN value to `isNaN` and inequality to
`notNaN`. Spark SQL semantics: NaN = NaN is true; NaN sorts above every
non-NaN double, so `d > NaN` is false, `d >= NaN` and `d = NaN` match exactly
the NaN rows, `d < NaN` and `d != NaN` match the non-NaN non-null rows; NULL
never matches.

## Clauses

- **C-1** Eq with a NaN literal on either side (`d = NaN`, `NaN = d`) converts
  to `IsNan(d)`, never to `Eq(d, NaN-datum)`.
- **C-2** EqNullSafe (`d <=> NaN`, DataFusion `IsNotDistinctFrom`) with a NaN
  literal converts to `IsNan(d)`. `NULL <=> NaN` is false in DataFusion and
  `isNaN(NULL)` is false in Java `NaNUtil`, so the pushed predicate is exact.
- **C-3** NotEq with a NaN literal (`d != NaN`, `NaN != d`) converts to
  `NotNan(d)`. `NotNan` keeps NULL rows at prune level per Java
  `notNaN(null) == true`; the e2e pin proves DataFusion's Inexact re-filter
  still drops them, else this clause falls back to NotTransformed.
- **C-4** Lt / LtEq / Gt / GtEq with a NaN literal on either side convert to
  NotTransformed (no pushdown; DataFusion evaluates). No range rewrite is
  attempted.
- **C-5** IN with NaN in the list: NaN alone converts to `IsNan(d)`; NaN mixed
  with non-NaN values converts to `IsNan(d) OR In(non-NaN values)`.
- **C-6** NOT IN with any NaN in the list converts to NotTransformed.
- **C-7** float32 (`Float`) behaves like float64: an `f32::NAN` literal takes
  the same arms as an `f64::NAN` literal.
- **C-8** A NaN literal under a CAST (`d = CAST('NaN' AS DOUBLE)`) returns the
  NaN rows end to end through DataFusion over the provider.
- **C-9** Nesting: NaN comparisons compose under AND / OR / NOT without
  reintroducing a NaN literal.

## Fix shape

In `crates/integrations/datafusion/src/physical_plan/expr_to_predicate.rs`:
a NaN check on the converted literal side runs before the operator mapping in
the binary arm, and the IN arm partitions NaN datums out of the set.
`scalar_value_to_datum` is unchanged: it still materializes the NaN datum so
the check has something to see. LIKE patterns accept only strings, the
`isnan()` scalar arm takes no literal, and bare column / literal arms build no
predicate, so the binary and IN arms are the only sites that can place a NaN
literal into a predicate. `Datum::is_nan` covers both `Float` and `Double`.

## Evidence

### Red-first unit pins (unfixed tree)

`cargo test -p iceberg-datafusion --lib
physical_plan::expr_to_predicate::nan_tests` on the unfixed tree:
3 passed, 9 failed. The three passes are the no-NaN control guards.
Sample failures:

```
left: Some(Binary(BinaryExpression { op: LessThan, term: Reference { name: "qux" },
  literal: Datum { type: Double, literal: Double(NaN) } }))
right: None
```

```
left: Some(Set(SetExpression { op: In, term: Reference { name: "qux" },
  literal: {Datum { type: Double, literal: Double(NaN) },
            Datum { type: Double, literal: Double(1.0) }} }))
right: Some(Or(LogicalExpression { inputs: [Unary(UnaryExpression { op: IsNan, ... }),
  Set(SetExpression { op: In, ... Double(1.0) ... })] }))
```

Every failing pin shows a NaN datum reaching the pushed predicate, or a
missing `IsNan` rewrite.

### Red-first e2e pins (unfixed tree)

`cargo test -p iceberg-datafusion --test nan_pushdown` on the unfixed tree:

```
assertion `left == right` failed: WHERE d = CAST('NaN' AS DOUBLE)
  left: []
 right: [1, 5]
```

The table holds NaN rows (ids 1, 5) across two data files plus finite and
NULL rows; the query returns no rows. This is the reported defect,
reproduced end to end.

### Post-fix gates

- `cargo test -p iceberg-datafusion` (whole crate): exit 0. Lib 228
  passed, 0 failed; every integration target green, including the new
  `nan_pushdown` e2e and the 12 new `nan_tests` conversion pins
  (48 conversion tests green in `physical_plan::expr_to_predicate`).
- `cargo clippy -p iceberg-datafusion --all-targets --all-features --
  -D warnings`: exit 0, no warnings.
- `cargo fmt --all -- --check`: clean (one formatting pass applied to the
  new unit file before the final runs).
- `./scripts/check_matrix_anchors.sh`: `OK: GAP_MATRIX anchors sound
  (84 rows anchored ...)`.
- `./scripts/check_agent_artifacts.sh`: OK.
- `./scripts/check_comment_blocks.sh`: OK (base `origin/main`).
- `python3 -B -m unittest scripts/check_rust_file_size_test.py` plus
  `./scripts/check_rust_file_size.sh`: `465 files clean`.
- `taplo check`: clean. `cargo machete`: no unused dependencies.
  `typos` on the touched files: no findings.
- Not run: `make test` (needs Docker), Java interop (no new fixture;
  the e2e asserts the Spark 4.1.2 answers measured on RePark main).

## Post-fix result

`cargo test -p iceberg-datafusion --test nan_pushdown`: 1 passed. All ten
filter legs return the Spark 4.1.2 answers: `=` and `IN (NaN)` return ids
1 and 5; `IN (NaN, 1.0)` returns 1, 2 and 5; `<` returns 2 and 4; `>=`
returns 1 and 5; `!=` returns 2 and 4; `<=>` returns 1 and 5; the three
float legs return 1 and 4, 1 and 4, and 1, 2 and 4.

C-3 settles on `NotNan`: the `!=` leg proves the NULL row (id 3) is still
re-checked away by DataFusion's Inexact re-filter, so the `NotNan` pushdown
stays.

## Open questions

None. Every decision the brief left open (NotEq as NotNan vs NotTransformed)
is settled by the e2e pin in this ledger.
