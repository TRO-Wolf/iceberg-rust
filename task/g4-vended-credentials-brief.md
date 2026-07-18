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

# Group G4 charter — consume vended storage credentials (GAP_MATRIX row R160, ❌ → flip)

Branch: `fix/audit-followups-bundle` (Mode B bundle, group 4 of 5; stacked on G1-G3 — do not
touch their surfaces). Implements the R160 parity gap found by the 2026-07-17 verification: Java
consumes `LoadTableResult.storage-credentials`; the fork parses the field
(`crates/catalog/rest/src/types.rs`, `storage_credentials`) and never reads it.

## Java oracle — decode the integration semantics FIRST (no network; cached jars)

`~/.m2/repository/org/apache/iceberg/` (iceberg-core-1.10.0 has `RESTSessionCatalog`,
`credentials/Credential`, `LoadTableResponse`; check for a `LoadTableResponse.credentials()`
consumer chain). `JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`, `javap -p -c`. Decode and cite:

1. `RESTSessionCatalog.newFileIO(SessionContext, Map, List<Credential>)` — how the credential
   list reaches FileIO properties. Expected shape (VERIFY, do not assume): each `Credential` has
   `prefix()` + `config()`; Java selects/overlays config for the table's storage — find the
   actual selection rule (longest-prefix match against the table location? single-credential
   fast path? what wins on key collision: vended config or table/catalog config?).
2. Where Java calls it from (`loadTable` path) and what map it passes as the base properties —
   this defines the overlay ORDER you must mirror.
3. Any expiry/refresh handling on vended credentials in 1.10.0 (if present, note it — likely
   OUT of scope for this group; name it as residue rather than building it).

## The fix

Wire `storage_credentials` into `load_file_io` (`crates/catalog/rest/src/catalog.rs`) mirroring
the decoded Java semantics: selection rule, overlay order, collision winners. Byte-parity is not
measurable here (property maps, not serialized artifacts) — the bar is BEHAVIORAL parity with
the decoded rules, each rule pinned. Loud behavior on unmatchable prefixes per Java (decode what
Java does when no credential matches — silent skip vs error — and mirror it).

SECURITY constraints (this is the SEC-001-adjacent surface):
- Vended credential values must NOT appear in Debug/log output — they flow into FileIO props,
  which land in `StorageConfig` (redacted Debug since #159; G3 extended the pattern). Verify the
  needle list covers the vended key names (e.g. `s3.access-key-id`, `s3.secret-access-key`,
  `s3.session-token`) and pin one composition case.
- No new logging of credential material anywhere on the path.

## Matrix (R160)

Flip status with HONEST evidence grade: implementation + behavioral pins offline ⇒ likely 🟡
with the named residue "live-cloud round-trip credentials-gated" (✅ requires credentialed
interop we cannot run offline — state which you shipped and why in the cell, date-stamped
2026-07-18). ALSO fix the G1-Critic-flagged wording in the same cell: the wire-key constant is
`STORAGE_CREDENTIALS` (screaming-snake, in `LoadTableResponseParser`), not a camelCase symbol.
MANDATORY: `./scripts/check_matrix_anchors.sh` after the edit; anchor R160 unchanged.

## Required pins (RED-under-mutation proven)

- Selection rule: multiple credentials with distinct prefixes → the decoded-rule winner applies
  to the FileIO for that table location (content-assert the resulting props).
- Overlay order: a colliding key between vended config and the base props → the decoded winner
  wins (pin the exact key).
- No-match behavior per the decode (skip vs error) — pinned either way.
- Zero-credentials path: behavior identical to today (regression pin — the common case must not
  change).
- Redaction composition: a FileIO built from vended creds Debug-prints without the secret value.
- Mutation per rule: invert the selection (e.g. shortest-prefix), swap the overlay order, drop
  the wiring entirely → each pins RED; restore byte-identical.

## Gates (ONE `&&` chain to commit; never `git add -A`)

typos (tracked) · `cargo fmt --all -- --check` · `cargo clippy -p iceberg-catalog-rest
--all-targets -- -D warnings` · `cargo test -p iceberg-catalog-rest --lib` · `cargo test -p
iceberg --lib` (composition pin may live core-side) · `./scripts/check_matrix_anchors.sh` ·
`./scripts/check_agent_artifacts.sh`.

## Constraints

CLAUDE.md + AGENTS.md bind. NO Cargo.toml/Cargo.lock edits. Edit surface:
`crates/catalog/rest/src/catalog.rs` (+ `types.rs` only if accessors are needed),
`docs/parity/GAP_MATRIX.md` row R160, tests; core needle list ONLY if a vended key name is
genuinely uncovered (flag it). Commit trailers exactly:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`.

## Report back (final message = the record)

(1) decoded Java semantics with offsets (selection rule, overlay order, no-match, expiry note);
(2) changes file:line; (3) the R160 cell text you shipped and the status grade rationale;
(4) pin inventory; (5) mutation table; (6) gate summary; (7) DEVIATIONS; (8) named residues.
