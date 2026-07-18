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

# Unit C charter — predicate op/arity validation on the serde boundary (audit SAF-004)

Branch: `fix/audit-predicate-serde-arity` (created from main; this charter is the tip commit).

## The defect (verified 2026-07-17, main @ 4ba52db7062ba67d95d2599833b852071491b49f)

The predicate visitor dispatchers panic on op/arity mismatch:
`crates/iceberg/src/expr/visitors/predicate_visitor.rs:172,196,207` and
`bound_predicate_visitor.rs:188,212,223` — `panic!("Unexpected op for unary/binary/set
predicate")`. The constructors (`expr/predicate.rs:147-148,200-201,277-278` —
`UnaryExpression::new` etc.) guard with `debug_assert!` ONLY (compiled out in release). And the
bypass is wire-reachable: `UnaryExpression`/`BinaryExpression`/`SetExpression`/`Predicate`/
`BoundPredicate` all derive `Deserialize` (`predicate.rs:104,164,241,320,706`) with no arity
validation, and `FileScanTask` (`scan/task.rs` — field `predicate: Option<BoundPredicate>`) is
`Serialize`/`Deserialize`. A crafted/corrupt serialized task whose JSON encodes e.g.
`Unary { op: LessThan }` bypasses `new()` entirely and panics the scan in release. (The JSON
*expression_parser* path is NOT affected — it validates arity at :592-636; the hole is the derived
serde on the types themselves.)

## The fix — two layers

1. **Validate at the serde boundary.** Each of the three expression shapes rejects mismatched
   operators at deserialize time with a serde error naming the op and expected arity class.
   `PredicateOperator` already knows its class (`is_unary()`/`is_binary()`/`is_set()` — verify
   those exist/are complete in `expr/mod.rs`; extend privately if a class test is missing). Use
   `#[serde(try_from = "...")]` with a raw shadow struct, or a manual `Deserialize` impl —
   whichever produces the smaller, idiomatic diff. Bound AND unbound variants. `Serialize` stays
   derived (we never emit invalid shapes).
2. **Visitor dispatch panics → typed `Err`.** The visit functions already return `Result`: replace
   the six `panic!`s with `Error::new(ErrorKind::DataInvalid, ...)` carrying the op. This is
   defense-in-depth for programmatic construction in release builds (the `debug_assert`s stay —
   making `new()` fallible is a breaking surface change and OUT of scope).

Java note (context, no decode needed): Java's `Expressions` factory methods validate eagerly and
`ExpressionParser` validates arity; Java has no equivalent unvalidated wire path, so tightening is
parity-consistent, not divergence.

## Required pins (RED-under-mutation proven)

- Round-trip: every valid op class serializes → deserializes → equals (spot per class, not
  exhaustive 30-op matrix; include one bound variant via a bound `FileScanTask` predicate if a
  fixture exists cheaply).
- Rejection: for each shape (unary/binary/set, unbound + bound), a JSON payload with a wrong-class
  op fails to deserialize with the typed message (assert error content, not just is_err). Mutation:
  drop the validation → pins RED.
- Visitor fallback: construct an invalid predicate in-memory (debug_assert bypass: build the struct
  via deserialization from the raw shadow in a test, or `#[cfg(test)]` constructor if the shadow is
  private) → `visit` returns the typed error, does not panic. Mutation: restore one `panic!` → pin
  RED.
- Existing suites green, including expression_parser round-trip tests (`cargo test -p iceberg
  --lib`) — the parser path must be behaviorally untouched.

## Gates (ONE `&&` chain to commit; never `git add -A`)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg --lib --tests -- -D warnings` · `cargo test -p iceberg --lib` ·
`./scripts/check_agent_artifacts.sh`.

## Constraints

CLAUDE.md + AGENTS.md bind. NO Cargo.toml/Cargo.lock edits. Edits confined to:
`expr/predicate.rs`, the two visitor files, `expr/mod.rs` (only if an op-class helper is missing),
and tests. Do NOT change the serialized FORMAT of valid predicates (wire compatibility with
already-serialized tasks is load-bearing — prove format stability with a fixture: a JSON string
serialized by the CURRENT code must still deserialize identically after your change). Commit
trailers exactly:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`.

## Report back (final message = the record)

(1) changes with file:line, including which serde mechanism you chose and why; (2) proof of wire
format stability for valid payloads; (3) pin inventory; (4) mutation table; (5) gate summary;
(6) DEVIATIONS; (7) named residues.
