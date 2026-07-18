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

# Group G2 charter — `ErrorKind::NamespaceNotEmpty` + the not-empty drop paths

Branch: `fix/audit-followups-bundle` (Mode B bundle, group 2 of 5; stacked on G1 — do not touch
G1's surface). Closes the #160 unit-E Critic's L1 residue.

## The gap

Java throws `NamespaceNotEmptyException` when dropping a non-empty namespace; the fork has no
equivalent `ErrorKind`, so:
- SQL `drop_namespace` (`crates/catalog/sql/src/catalog.rs`, the "Namespace is not empty" path,
  ~L763) emits `Unexpected`.
- HMS `drop_database`'s `InvalidOperationException` arm (`crates/catalog/hms/src/error.rs`, the
  drop_database mapper) stays `Unexpected`.

## The fix

1. **Core**: add `ErrorKind::NamespaceNotEmpty` to `crates/iceberg/src/error.rs` — the enum is
   `#[non_exhaustive]`, so this is additive/non-breaking (still: place it with the other
   namespace variants, doc-comment in the house style, add the Display arm — study how the
   existing variants spell theirs). This is the bundle's ONE sanctioned core-error edit.
2. **SQL**: flip the not-empty path to the new kind (message preserved).
3. **HMS — decode BEFORE flipping**: Java `HiveCatalog.dropNamespace` (cached
   `iceberg-hive-metastore-1.10.0.jar`, JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64,
   `javap -p -c`) — determine what Java maps the Hive `InvalidOperationException` to. Only if
   the bytecode shows it becomes `NamespaceNotEmptyException` (or the not-empty semantic is
   otherwise provably the sole meaning of that arm on drop_database) may the HMS mapper arm
   flip; if the exception is semantically broader, it STAYS `Unexpected` and you record the
   bytecode-grounded reasoning in the report + a code comment. Do not guess.

## Contract check (before flipping anything)

Sweep for callers matching on `Unexpected` from these paths and for any `retryable()` coupling —
the #160 unit-E sweep found zero conditional matchers and retry provably kind-independent;
re-verify briefly on the current tree rather than trusting that record.

## Required pins (RED-under-mutation proven)

- SQL: memory-backed test — create namespace, create table in it, `drop_namespace` → assert the
  NEW kind (+ message content). Mutation: revert kind → RED.
- HMS: mapper-level unit test for whichever outcome the decode dictates (flip → typed pin;
  stay → a stays-Unexpected pin with the doc comment). Mutation accordingly.
- Core: Display arm pin (kind renders its name) if the existing variants have such pins —
  follow the file's precedent.

## Gates (ONE `&&` chain to commit; never `git add -A`)

typos (tracked) · `cargo fmt --all -- --check` · `cargo clippy -p iceberg --lib --tests -- -D
warnings` · `cargo clippy -p iceberg-catalog-sql --all-targets -- -D warnings` · `cargo clippy -p
iceberg-catalog-hms --all-targets -- -D warnings` · `cargo test -p iceberg --lib` · `cargo test
-p iceberg-catalog-sql --lib` · `cargo test -p iceberg-catalog-hms --lib` ·
`./scripts/check_agent_artifacts.sh`.

## Constraints

CLAUDE.md + AGENTS.md bind. NO Cargo.toml/Cargo.lock edits. Edit surface: core `error.rs` (the
one variant + Display arm ONLY), sql catalog/error files, hms error.rs (+ catalog.rs only if the
flip requires wiring), tests. The new variant is additive — call it out in the commit message
body as a public-API addition so downstream pins can follow. Commit trailers exactly:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`.

## Report back (final message = the record)

(1) the HMS bytecode adjudication with offsets and the flip/stay decision; (2) changes file:line;
(3) contract-check result; (4) pin inventory; (5) mutation table; (6) gate summary;
(7) DEVIATIONS; (8) named residues.
