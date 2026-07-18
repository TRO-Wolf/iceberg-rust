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

# Unit D charter — hardening quick-wins + parity-gap matrix rows (audit SEC-002/003 chain, SEC-008, SAF-008; verification-discovered gaps)

Branch: `infra/audit-hardening-quickwins` (created from main; this charter is the tip commit).

## D1 — redacted Debug for the raw-props layer (SEC-002 plus the SEC-003 chain)

`crates/iceberg/src/io/storage/config/mod.rs:52-56`: `StorageConfig` derives plain `Debug` over a
raw property `HashMap` — and the REST catalog clones its full runtime props (including
`credential`/`token`/`client_secret`) into `FileIO` (`catalog/rest/src/catalog.rs:499-537`), so a
consumer `{:?}`-ing a `FileIO`/`StorageConfig` prints live credentials. The typed configs
(`S3Config` `config/s3.rs:136-143`, `GcsConfig`, `AzdlsConfig`, `OssConfig`) and
`RestCatalogConfig` (`catalog.rs:145-149,157-174`, SECRET_PROP_KEYS) already hand-redact — mirror
that pattern: a manual `Debug` for `StorageConfig` that redacts values whose keys match a
secret-key list (adapt the existing SECRET_PROP_KEYS approach; cover at minimum: credential,
token, secret, key, password substrings — check what the typed configs treat as secret and stay a
superset). Also sweep `FileIO`/`FileIOBuilder` (`io/file_io.rs:61,210`) — if their derived Debug
prints the embedded `StorageConfig`, the fix composes; verify with a pin.

Same pattern for `HttpClient`'s Debug (`catalog/rest/src/client.rs:50-56`): `extra_headers`
printed raw leaks `header.*`-configured auth headers (SEC-008). Redact header VALUES for sensitive
header names (reuse the redaction list at `client.rs:344-360`).

Do NOT touch the glue/hms/s3tables/sql catalog config Debugs in this unit (bigger sweep, named
residue).

## D2 — default-literal validation instead of serialize-panic (SAF-008)

`crates/iceberg/src/spec/datatypes.rs`: `From<NestedField> for SerdeNestedField` (~:690-691)
`.expect()`s that the default literal converts to JSON, claiming "we should have checked this in
NestedField::with_initial_default" — but the setters (~:755-763) perform NO validation.
`NestedField::required(1,"f",Int).with_initial_default(Literal::bool(true))` + schema serialize =
panic. Fix WITHOUT breaking the builder API: keep `with_initial_default`/`with_write_default`
signatures, move the failure to serialization time as a proper a serde serialization Error (custom
`Serialize` or a `TryFrom` shadow with `#[serde(try_from/into)]` — check what SerdeNestedField's
current wiring allows) so an invalid (literal, type) pair errors typed instead of panicking.
Verify against Java for message spirit only (Java validates in the fluent setter via
`Preconditions` — if you can match Java by validating in the setter WITHOUT an API break, i.e. the
setters already return `Self` by value and could... they cannot return Result without breaking;
note this as the parity residue).

## D3 — GAP_MATRIX rows for the two verification-discovered parity gaps

Add two NEW capability rows to `docs/parity/GAP_MATRIX.md` (next unused `R<id>` anchors — check
the current max; IDs are never reused):

1. **REST OAuth automatic token refresh** — status ❌. Java 1.10.0 `OAuth2Manager`
   `keepRefreshed`/`RefreshingAuthManager` refreshes automatically (`token-refresh-enabled`
   default true); fork caches the token forever (`catalog/rest/src/client.rs:217` TODO, manual
   `regenerate_token` only). Cite the Java classes; date-stamp provenance 2026-07-17
   (bytecode-verified).
2. **Vended storage credentials (`LoadTableResult.storage_credentials`)** — status ❌. Java
   consumes vended credentials into FileIO (`RESTSessionCatalog.newFileIO(..., List<Credential>)`);
   fork parses the field (`catalog/rest/src/types.rs:229-238`) and never reads it. Same provenance
   stamp.

Row format: study neighboring rows first; terse cells; 5-pipe discipline (raw pipes inside code
spans split cells). MANDATORY after the matrix edit: `./scripts/check_matrix_anchors.sh` green.

## Required pins (RED-under-mutation proven)

- D1: Debug output of a `StorageConfig` (and a `FileIO` embedding it) built with a secret-bearing
  prop map does NOT contain the secret value but DOES name the keys (match the existing redaction
  pin style, e.g. RestCatalogConfig's). Same shape for `HttpClient` with a `header.authorization`
  extra header. Mutation: revert to derived Debug → pins RED.
- D2: the invalid (literal, type) default → schema JSON serialization returns Err (content
  asserted), no panic; a VALID default still round-trips byte-identically (format stability).
  Mutation: restore the expect → pin RED (catch_unwind or should_panic contrast — prefer asserting
  the Err).
- Matrix: `check_matrix_anchors.sh` is the gate; no unit test.

## Gates (ONE `&&` chain to commit; never `git add -A`)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg --lib --tests -- -D warnings` · `cargo clippy -p iceberg-catalog-rest
--all-targets -- -D warnings` · `cargo test -p iceberg --lib` · `cargo test -p
iceberg-catalog-rest --lib` · `./scripts/check_matrix_anchors.sh` ·
`./scripts/check_agent_artifacts.sh`.

## Constraints

CLAUDE.md + AGENTS.md bind. NO Cargo.toml/Cargo.lock edits. Edits confined to:
`io/storage/config/mod.rs`, `io/file_io.rs` (Debug only), `catalog/rest/src/client.rs` (Debug
only), `spec/datatypes.rs` (+ `spec/values/literal.rs` ONLY if a try_into_json helper tweak is
unavoidable — flag it), `docs/parity/GAP_MATRIX.md`, tests. On-disk/wire format of VALID data must
not change — prove with the D2 round-trip pin. Commit trailers exactly:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`.

## Report back (final message = the record)

(1) changes with file:line per sub-unit; (2) the secret-key list you chose and why it's a superset
of the typed configs'; (3) new matrix row anchors + cell text; (4) pin inventory; (5) mutation
table; (6) gate summary; (7) DEVIATIONS; (8) named residues (include: glue/hms/s3tables/sql config
Debug sweep deferred; setter-time validation parity residue if applicable).
