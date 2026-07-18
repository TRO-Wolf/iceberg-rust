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

# Unit E charter — typed error kinds in SQL + HMS catalogs (audit CQ-002/CQ-003)

Branch: `fix/audit-typed-error-kinds` (created from main; this charter is the tip commit).

## The defect (verified 2026-07-17, main @ 4ba52db7062ba67d95d2599833b852071491b49f)

Callers cannot branch on catalog errors:

- SQL: `crates/catalog/sql/src/error.rs:63-82` — `no_such_namespace_err`, `no_such_table_err`,
  `table_already_exists_err` emit `ErrorKind::Unexpected`, while the VIEW helpers in the same file
  (:84-96) use typed kinds. The typed variants exist in the core crate
  (`crates/iceberg/src/error.rs:48-57`: TableNotFound, NamespaceNotFound / their actual names —
  read the enum, use what's there, do NOT add new variants without flagging).
- HMS: `crates/catalog/hms/src/error.rs:28-55` — every thrift failure and every
  not-found/already-exists condition collapses to `ErrorKind::Unexpected`; zero typed-kind usage
  in the whole crate.

## The fix

1. SQL: flip the three helpers to the typed kinds, mirroring the view helpers' shape exactly
   (message text preserved unless factually wrong — behavior-compatible messages, typed kinds).
2. HMS: map the specific hive_metastore thrift exception types to typed kinds where the thrift
   IDL distinguishes them (`NoSuchObjectException` → table/namespace-not-found by call-site
   context, `AlreadyExistsException` → already-exists; read
   `crates/catalog/hms/src/error.rs` + the call sites in `catalog.rs` to see what type info
   survives). Where the generic `from_thrift_error` genuinely cannot know, leave Unexpected —
   precision at the call sites that DO know beats a lossy global remap. Sweep every
   `catalog.rs` call site that matches on "no such"/"already exists" semantics.
3. Java parity anchor (no bytecode needed): Java catalogs throw `NoSuchTableException`/
   `NoSuchNamespaceException`/`AlreadyExistsException` — typed kinds are the Rust-native
   equivalent; this is convergence, not divergence.

## CRITICAL contract check before you start

Grep the workspace for tests and callers matching on the CURRENT `ErrorKind::Unexpected` from
these paths (e.g. `assert!(matches!(...Unexpected...))` in sql/hms tests, loader crate, transaction
retry logic `retryable()`, datafusion integration). Every caller that branches on the old kind is
in scope to update — a kind flip that breaks a hidden `match` is the classic regression here. Also
check: does `Error::retryable()` or the transaction retry path treat Unexpected differently from
the typed kinds? If flipping a kind would change retry behavior, STOP and record it as a
DEVIATIONS finding with your recommendation instead of silently changing retry semantics.

## Required pins (RED-under-mutation proven)

- Per flipped helper: an operation that hits the path asserts the TYPED kind (memory-backed SQL
  catalog tests exist — extend them; HMS lib tests are offline/mock — extend what's there, note
  what is only reachable via docker integration and pin those at the error-mapper unit level).
- Mutation: revert each helper to Unexpected → its pin RED, restore GREEN.
- Full workspace check for kind-match regressions: `cargo test -p iceberg-catalog-sql --lib` +
  `cargo test -p iceberg-catalog-hms --lib` + `cargo test -p iceberg --lib` green.

## Gates (ONE `&&` chain to commit; never `git add -A`)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg-catalog-sql --all-targets -- -D warnings` · `cargo clippy -p
iceberg-catalog-hms --all-targets -- -D warnings` · the three test suites above ·
`./scripts/check_agent_artifacts.sh`.

## Constraints

CLAUDE.md + AGENTS.md bind. NO Cargo.toml/Cargo.lock edits. Edits confined to the two catalog
crates (error.rs + catalog.rs + tests); core `error.rs` READ-ONLY (no new variants — if a needed
variant is missing, record it in DEVIATIONS instead). Commit trailers exactly:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`.

## Report back (final message = the record)

(1) changes with file:line; (2) the caller/contract sweep results (who matched on Unexpected, what
you updated, the retryable() answer); (3) HMS thrift-exception mapping table (exception × call-site
→ kind); (4) pin inventory; (5) mutation table; (6) gate summary; (7) DEVIATIONS; (8) named
residues (e.g. docker-gated paths not pinned offline).
