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

# Group G3 charter — catalog-config redacted Debug sweep (Glue / HMS / S3Tables / SQL)

Branch: `fix/audit-followups-bundle` (Mode B bundle, group 3 of 5; stacked on G1+G2 — do not
touch their surfaces). Closes the #159 unit-D deferred residue: the four remaining catalog
config types derive plain `Debug` over secret-bearing property maps / DSNs.

## The surfaces (verify each on the current tree; line refs are from the 2026-07-17 audit)

- `GlueCatalogConfig` (`crates/catalog/glue/src/catalog.rs` ~L136) — raw `props: HashMap` may
  carry AWS creds; printed via `GlueCatalog`'s Debug too.
- `HmsCatalogConfig` (`crates/catalog/hms/src/catalog.rs` ~L156) — props map; printed via
  `HmsCatalog` Debug.
- `S3TablesCatalogConfig` (`crates/catalog/s3tables/src/catalog.rs` ~L46) — props map.
- `SqlCatalogConfig` (`crates/catalog/sql/src/catalog.rs` ~L215) — **the DSN/uri can embed
  credentials** (`postgres://user:pass@host`) — decide and document: redact the userinfo
  password portion of the URI (preferred; keep scheme/host/db visible) or the whole value;
  props map same as the others. `SqlCatalog` itself derives Debug with a `fileio: FileIO` field
  — FileIO's Debug is already redacted (#159), verify composition rather than re-fixing.

## The pattern

Mirror #159's `StorageConfig` approach: manual `Debug`, keys visible, secret values `"***"`,
case-insensitive substring needle matching. The needle list lives in
`crates/iceberg/src/io/storage/config/mod.rs` (`SECRET_PROP_KEY_SUBSTRINGS` +
`is_secret_prop_key`) — REUSE it if visibility allows (it may need `pub` promotion from
`pub(crate)`; that is sanctioned, flag it), otherwise replicate minimally per crate with a
comment naming the canonical copy. Do NOT invent a new needle set; if a catalog needs an extra
needle (e.g. `dsn`, `uri`-embedded), ADD it with justification.

## G3b — Glue not-empty drop kind flip (G2 spillover, sanctioned)

G2 added `ErrorKind::NamespaceNotEmpty` and flipped SQL+HMS; the G2 Actor found Glue's own
not-empty drop path (`crates/catalog/glue/src/catalog.rs` ~L475, message
`"Database with name: {} is not empty"`) still emitting a non-typed kind. Flip it to
`NamespaceNotEmpty` (message preserved), with a pin asserting the kind and that the namespace
survives the refused drop, and a revert mutation → RED. This is the ONE functional change in G3;
everything else stays Debug-format-only. Also sanctioned (comment-only, G2 Critic residue F1):
tighten the `from_drop_database_exception` doc comment in `crates/catalog/hms/src/error.rs`
(~L187) — the guarantee is Java's UNCONDITIONAL mapping of the caught exception (the not-empty
class also covers functions/materialized views), not "tables are the sole meaning".

## Required pins (RED-under-mutation proven, per config type)

- Debug output with a secret-bearing map does not contain the secret value, does name the keys.
- The catalog-level composition (e.g. `GlueCatalog`'s Debug renders the redacted config).
- SQL: a DSN with an embedded password — password absent from output, host/db still visible
  (or whole-value redaction if you chose that — pin whichever you shipped).
- One mutation per config type (revert to derived/raw) → its pin RED; restore byte-identical.

## Gates (ONE `&&` chain to commit; never `git add -A`)

typos (tracked) · `cargo fmt --all -- --check` · clippy `-D warnings` for: `iceberg` (only if
the needle helper visibility changed), `iceberg-catalog-glue`, `iceberg-catalog-hms`,
`iceberg-catalog-sql`, `iceberg-catalog-s3tables` (all `--all-targets`) · `cargo test --lib` for
those four catalog crates (+ `-p iceberg --lib` if core touched) · `./scripts/check_agent_artifacts.sh`.

## Constraints

CLAUDE.md + AGENTS.md bind. NO Cargo.toml/Cargo.lock edits (if reusing the core needle helper
requires a dependency the catalog crates lack, STOP that route and replicate instead — do not
touch manifests). Behavior changes are Debug-format ONLY — no functional edits — EXCEPT the sanctioned G3b kind
flip above. Edit surface:
the four catalog crates' config/catalog files + tests, plus core `io/storage/config/mod.rs`
visibility ONLY if reusing. Commit trailers exactly:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`.

## Report back (final message = the record)

(1) per-config changes file:line + the reuse-vs-replicate decision; (2) the SQL DSN redaction
design choice and why; (3) pin inventory; (4) mutation table; (5) gate summary; (6) DEVIATIONS;
(7) named residues.
