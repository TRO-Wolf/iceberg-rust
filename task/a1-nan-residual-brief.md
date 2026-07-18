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

# Unit A1 charter — real NaN evaluation on the residual paths (audit BUG-001, Critical)

Branch: `fix/audit-nan-null-residual-parity` (this unit is the first of two stacked groups; keep
your commits scoped to A1 — a follow-up unit A2 will land the broader null-semantics port on the
same branch AFTER you finish. Do not fix null-comparison semantics beyond what this charter names).

## The defect (verified 2026-07-17, main @ 4ba52db7)

Both row-level predicate evaluators compile `is_nan` / `not_nan` on a PRESENT float column to
constants:

1. `crates/iceberg/src/arrow/reader.rs` — `PredicateConverter::is_nan` (~L2079): column present →
   `build_always_true()`; `not_nan` (~L2091): column present → `build_always_false()`. This
   converter's closures become an `ArrowPredicateFn` inside the parquet `RowFilter`
   (`get_row_filter`, ~L1416-1443; applied ~L612-624) — rows failing the predicate are DROPPED
   during read. So `not_nan` on a present column silently drops EVERY row; `is_nan` returns finite
   values as if they were NaN.
2. `crates/iceberg/src/arrow/record_batch_predicate.rs` — `is_nan`/`not_nan` (~L176-189): same
   constant shapes (`all_true()`/`all_false()`). `evaluate_predicate_to_mask` is used by the reader
   post-filter path and by equality-delete application (`arrow/delete_filter.rs`), so eq-delete
   predicates involving NaN-relevant floats can mis-apply.

DataFusion maps `isnan(col)` → `Reference::new(col).is_nan()`
(`crates/integrations/datafusion/src/physical_plan/expr_to_predicate.rs:229`) and pushdown is
`Inexact`: DF's re-filter masks over-INCLUSION but cannot restore rows the RowFilter over-DROPPED —
`SELECT ... WHERE NOT isnan(x)` returns zero rows through SQL today.

## The fix

Implement real per-row NaN evaluation in BOTH evaluators for Float32/Float64 columns (elementwise
`f.is_nan()` over the arrow array — build the BooleanArray yourself or via arrow compute; no `as`
casts; nulls handled per the Java oracle below). The two implementations must agree; if you can
share a helper across the two files without contorting either, do so.

## Java oracle — decode, don't trust priors (including this brief's)

The parity oracle is Java 1.10.0, cached locally (NO network):
`~/.m2/repository/org/apache/iceberg/iceberg-api/1.10.0/iceberg-api-1.10.0.jar` (+ iceberg-core
same tree). `JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`, unzip into your scratch dir, decode
with `javap -p -c`. A previous unit's brief mis-stated a decoded oracle and the Actor's own decode
was right — treat every claim in this section as a hypothesis until your own decode confirms it.

Decode and cite (class + method + decisive bytecode offsets) each of:

1. `Evaluator$EvalVisitor.isNaN` / `notNaN` — expected shape: `NaNUtil.isNaN(term.eval(struct))`
   and its negation. Confirm.
2. `NaNUtil.isNaN` — how a NULL value evaluates (hypothesis: null → false for isNaN, hence true
   for notNaN). Your NULL-cell arm must byte-match this outcome in both Rust evaluators. Note the
   interplay: `record_batch_predicate` results currently pass through `coerce_nulls_to_false`
   (`reader.rs` ~L1979) — make sure the END-TO-END outcome for a null cell matches Java, whatever
   the internal mask encoding is.
3. Binding legality: can Java even bind `isNaN` to a non-floating column? (Check
   `UnboundPredicate.bind` / wherever IS_NAN validation lives — likely a ValidationException for
   non-float types.) Compare with Rust `Predicate::bind` (`crates/iceberg/src/expr/predicate.rs`) —
   if Rust binding already rejects non-float `is_nan`, the evaluator only ever sees float columns
   and needs no non-float arm; if Rust binding is looser than Java, note it as a DEVIATION finding
   (do NOT widen this unit to fix binding).
4. Missing (schema-evolved) column arm: Java evaluates the term against a null value. Confirm the
   current Rust arms (`is_nan` missing → always_false, `not_nan` missing → always_true) match
   Java's outcome; fix them only if the decode says they diverge.

## Required pins (tests land with the change, RED-under-mutation proven)

- A parquet fixture with a Float64 (and a Float32) column containing {NaN, finite, null} rows,
  scanned through the FULL path (`TableScan::to_arrow` with `is_nan` filter, then `not_nan`):
  content-asserted row sets both directions. This pins the RowFilter path.
- The same truth table through `evaluate_predicate_to_mask` directly (unit-level), covering
  null-cell behavior per the decoded oracle.
- A DataFusion e2e pin in `crates/integrations/datafusion` tests: `WHERE NOT isnan(x)` returns the
  finite rows (this is the silent-zero-rows regression) and `WHERE isnan(x)` returns only NaN rows.
- Missing-column arms: a file written before the float column existed, scanned with
  `is_nan`/`not_nan` filters — outcomes per the decoded Java answer.
- Mutation-proof every new arm: revert each behavioral change one at a time (constant-true,
  constant-false, dropped null-handling) and name the exact pins that go RED; restore GREEN.
  A pin that stays green under its mutation is vacuous — rework it.

## Gates (chain to commit in ONE `&&` chain; never `git add -A`)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg --lib --tests -- -D warnings` · `cargo test -p iceberg --lib` ·
if the datafusion crate is touched: `cargo clippy -p iceberg-datafusion --all-targets -- -D
warnings && cargo test -p iceberg-datafusion` · `./scripts/check_agent_artifacts.sh`.

## Constraints

- CLAUDE.md + AGENTS.md rules bind: no bare `.unwrap()` in production paths, typed errors with
  context, no `as` numeric casts, tracing not println, house banners only where the module already
  uses them.
- NO Cargo.toml/Cargo.lock edits. NO edits outside: the two evaluator files, test files/fixtures,
  and (if a shared helper) `crates/iceberg/src/arrow/` module wiring. `task/` and map.md updates
  only as the conventions require.
- Do not touch the null-comparison semantics of other operators (`lt`/`not_eq`/... — that is unit
  A2, out of your scope even where you notice divergence; NAME what you noticed instead).
- Commit trailers exactly:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`
  (use `git commit -F <file>` from your scratch dir if your message has characters the tool
  rejects).

## Report back (your final message is the record)

Structured: (1) oracle decodes with offsets for all four questions; (2) what changed, file:line;
(3) pin inventory; (4) mutation table (mutation → RED pins → restored GREEN); (5) gate transcript
summary; (6) DEVIATIONS — anything you did beyond/despite this charter, including brief errors you
corrected; (7) named residues you chose not to touch.
