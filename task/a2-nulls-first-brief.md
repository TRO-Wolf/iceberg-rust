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

# Unit A2 charter — Java nulls-first null semantics on the row/partition evaluators (audit BUG-002/003/011)

Branch: `fix/audit-nan-null-residual-parity` — stacked ON TOP of unit A1 (tip `2024ab76`, real NaN
evaluation). A1's shared-helper pattern in `record_batch_predicate.rs` (two-valued, `is_valid`-aware
masks; see `nan_row_mask` and its pins) is the house pattern to follow. Do not modify A1's arms.

## Design decision (LOCKED by user sign-off 2026-07-17 — do not relitigate)

The row-level and partition-level evaluators port **Java's nulls-first total-order semantics**, not
SQL three-valued logic. Rationale on record: the parity mandate is the north star; DataFusion's
`Inexact` pushdown re-filter remains authoritative for SQL-3VL consumers (it re-drops null rows the
scan now keeps), while direct library consumers get the Java `Evaluator` contract.

## The defect family (verified 2026-07-17 vs decoded bytecode)

Java `Evaluator$EvalVisitor` (iceberg-api 1.10.0) computes every comparison via
`literal.comparator().compare(term.eval(struct), literal.value())` where the comparator chain is
`Comparators.nullsFirst().thenComparing(naturalOrder)` — a TOTAL ORDER with null smallest. Decoded
truth table for a NULL value vs a non-null literal (re-verify, see oracle section):

| op | Java result |
|---|---|
| `<` , `<=` | **TRUE** |
| `>` , `>=` | FALSE |
| `==` | FALSE |
| `!=` | **TRUE** |

Rust divergences to fix (each verified at the current branch):

1. **`crates/iceberg/src/arrow/reader.rs` `PredicateConverter`** (RowFilter — divergent rows are
   silently DROPPED during read):
   - missing-column `not_eq` (~L2219) → `build_always_false()`; Java ⇒ true. (Missing `lt`/`lt_eq`
     → always_true already match; verify `gt`/`gt_eq`/`eq`/`in`/`not_in`/`starts_with`/
     `not_starts_with` missing arms against your decode and fix any that diverge.)
   - present-column NULL cells: the closures use Arrow kleene kernels whose NULL results the
     parquet `RowFilter` treats as drop — wrong for every op whose Java-null result is TRUE
     (`lt`, `lt_eq`, `not_eq`, and whichever of `not_in`/`not_starts_with` your decode confirms).
2. **`crates/iceberg/src/arrow/record_batch_predicate.rs`** (post-filter + eq-delete application):
   same family — `binary_cmp(..., on_missing, kernel)` arms and the missing-column table in the
   module doc. NULL results pass through `coerce_nulls_to_false` (`reader.rs` ~L1990s) → dropped;
   ops with Java-null=TRUE must yield a definite TRUE in the mask (two-valued, A1 pattern).
   Ops with Java-null=FALSE (`eq`,`gt`,`gt_eq`,`in`,`starts_with` per decode) already coincide
   end-to-end via the coercion — for those, prefer leaving the kernels untouched and PIN the
   end-to-end outcome rather than churning working code.
3. **`crates/iceberg/src/expr/visitors/expression_evaluator.rs`** (partition pruning — divergence
   silently PRUNES files = lost rows): `less_than` (~L131-141) and `less_than_or_eq` (~L143-153)
   return `Ok(false)` for a null partition value; Java ⇒ true (keep the file). `not_eq`
   (~L191-199 → true) and `eq`/`gt`/`gt_eq` (→ false) already match; verify and fix the
   `in`/`not_in`/`starts_with`/`not_starts_with` null arms per your decode.

## Java oracle — re-decode everything; treat this brief's table as hypothesis

Cached jars (NO network): `~/.m2/repository/org/apache/iceberg/iceberg-api/1.10.0/` (+ iceberg-core
sibling). `JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`, `javap -p -c`. Decode and cite offsets:

1. `Evaluator$EvalVisitor.lt/ltEq/gt/gtEq/eq/notEq` — the compare-then-branch shapes.
2. `Literals$ComparableLiteral` comparator chain + `Comparators$NullsFirst.compare` +
   `NullSafeChainedComparator` (second comparator only when both non-null) — derive the six-op
   truth table yourself.
3. **`in`/`notIn`** null handling (`Evaluator` inSet path — hypothesis: `literalSet.contains(null)`
   = false ⇒ `in`→false, `notIn`→true) and **`startsWith`/`notStartsWith`** null handling
   (hypothesis: there is an explicit null guard — find it; if `startsWith` would NPE on null in
   Java, the parity answer for that arm is "unreachable in Java" and the Rust arm should keep its
   current safe behavior, DOCUMENTED).
4. Confirm which class is the partition-pruning oracle: iceberg-core `ManifestGroup` builds
   `new Evaluator(spec.partitionType(), Projections.inclusive(spec).project(filter))` over
   `file.partition()` — i.e., the SAME EvalVisitor semantics apply to `expression_evaluator.rs`.

The A1 report's decodes (offsets in `task/todo.md` A1 flip and the A1 commit message) are
precedent for citation style.

## Required pins (RED-under-mutation proven; every new arm mutation-proofed)

- **Mask-level truth table** (`evaluate_predicate_to_mask`): every comparison op ×
  {NULL cell, missing column} against the decoded Java table — including the ops that already
  coincide (those pins guard against future kernel churn; they must be RED-able by mutating the
  coincidence away, e.g. flipping the coercion — prove at least one such mutation).
- **Full-path scan pins** (`TableScan::to_arrow`, extend A1's NaN fixture or add a sibling): a file
  with NULL cells scanned under `!=`, `<`, `<=` keeps the null rows (BUG-003 headline); a
  schema-evolved file (column absent) scanned under `!=` keeps all rows (BUG-002 headline —
  today it returns zero).
- **Partition-pruning pins** (`expression_evaluator.rs` unit tests): null partition value ×
  every op → keep/prune per the table; mutation `less_than` back to `Ok(false)` → RED.
- **DataFusion documentation pin**: one e2e showing DF's SQL 3VL re-filter still drops null rows
  under `!=` (i.e., SQL semantics preserved for SQL consumers) — this is the pin that proves the
  design decision is safe, document it in-test.
- `is_valid`-guard pin per the A1 pattern (buffer values under invalid slots) for any new
  two-valued kernel you write.

## Gates (ONE `&&` chain to commit; never `git add -A`)

Same as A1: typos (tracked) · `cargo fmt --all -- --check` · `cargo clippy -p iceberg --lib
--tests -- -D warnings` · `cargo test -p iceberg --lib` · if datafusion touched: `cargo clippy -p
iceberg-datafusion --all-targets -- -D warnings && cargo test -p iceberg-datafusion --tests` ·
`./scripts/check_agent_artifacts.sh`.

## Constraints

- CLAUDE.md + AGENTS.md bind (no bare unwrap in production, no `as` casts, typed errors).
- NO Cargo.toml/Cargo.lock edits. Edit surface: the three named files, `arrow/` module wiring for
  shared helpers, tests/fixtures, `task/todo.md` (flip your bullet), this brief only for
  typos-gate compliance. NOTHING else — the metrics evaluators
  (`InclusiveMetricsEvaluator`/strict), `PageIndexEvaluator`, and manifest evaluator are OUT of
  scope even where you notice the same family (NAME them as residue instead).
- Preserve A1's behavior byte-for-byte (its pins must stay green untouched).
- Commit trailers exactly:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`
  (`git commit -F` from your scratch dir if needed).

## Report back (your final message is the record)

(1) decoded truth table with offsets, incl. the in/notIn/startsWith answers and the
partition-oracle confirmation; (2) per-file changes with line numbers, and the
leave-untouched-vs-change decision per op; (3) pin inventory; (4) mutation table (mutation → RED
pins → restored GREEN); (5) gate summary; (6) DEVIATIONS (incl. any brief errors you corrected);
(7) named residues.
