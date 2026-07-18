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

# Group G5 charter — REST OAuth2 automatic token refresh (GAP_MATRIX row R159, ❌ → flip)

Branch: `fix/audit-followups-bundle` (Mode B bundle, group 5 of 5; stacked on G1-G4 — do not
touch their surfaces). Implements the R159 parity gap: Java 1.10.0 refreshes OAuth tokens
automatically; the fork exchanges once and caches forever (`crates/catalog/rest/src/client.rs`,
`authenticate`, the `# TODO: Support automatic token refreshing` — manual
`regenerate_token`/`invalidate_token` only).

## Java oracle — decode FIRST (cached iceberg-core-1.10.0; JAVA_HOME=java-11; javap -p -c)

1. `OAuth2Manager` / `RefreshingAuthManager` / `OAuth2Util` (+ `OAuth2Util$AuthSession`,
   `OAuth2Properties`): the refresh trigger model (Java schedules background refresh — find the
   schedule rule: fraction of expires_in? fixed skew?), the refresh GRANT type (client
   credentials re-exchange vs token-exchange with the current token as subject — decode
   `refreshToken`/`refreshExpiredToken` paths), and the config knobs with their defaults
   (`token-refresh-enabled` default, `token-expires-in-ms` default, session timeout).
2. What Java does when the token response carries no `expires_in` (refresh disabled for that
   token? default applied?) — mirror it.
3. What happens on refresh FAILURE in Java (keep old token and retry later? invalidate? error
   the request?) — mirror the observable behavior.

## Design adaptation (sanctioned, disclose in code + report)

Java runs a background scheduler thread. This library is async-Rust; a spawned background task
per catalog is heavier machinery than the contract needs. SANCTIONED SHAPE: **lazy
refresh-before-use** — store the token with its expiry instant; on each authenticated request
path, if refresh is enabled and the token is within the skew window (derive the skew from
Java's schedule rule) or expired, refresh it (single-flight: concurrent requests must not
stampede N parallel refreshes — use the existing lock/once machinery or a tokio Mutex, no
lock held across the actual HTTP await unless bounded-and-justified per CLAUDE.md concurrency
rules). The observable contract must match Java: no request is sent with a known-expired token;
tokens refresh proactively near expiry; `token-refresh-enabled=false` restores today's behavior
exactly. If your decode reveals this adaptation cannot match some observable Java behavior,
STOP and record it in DEVIATIONS with your recommendation instead of building a divergent thing.

## Security constraints (SEC-001-adjacent — the token endpoint is server-influencable)

- The refresh path must reuse `get_token_endpoint()` exactly as `authenticate` does — no NEW
  endpoint resolution, no new place credentials are POSTed. (The server-override-of-
  `oauth2-server-uri` exposure is parity-shared with Java and out of scope — do not widen it,
  do not fix it here.)
- No token/credential material in logs, Debug output, or error contexts on the new path (the
  existing OAuth error paths attach only `response_body_len` — keep that discipline).
- Clock handling: no `SystemTime` arithmetic that panics on skew; use `Instant` monotonic math
  with saturating operations.

## Testability

`mockito` is already a dev-dependency of the rest crate — use it for the refresh pins (short
`expires_in`, assert the second request re-hits the token endpoint; a refresh-failure case;
a refresh-disabled case). Do NOT add dependencies. If expiry timing needs control, prefer a
tiny injectable-clock seam over sleeps; sub-second real sleeps in a couple of pins are
acceptable if bounded (<2s total suite impact).

## Matrix (R159)

Flip with honest evidence grade (likely 🟡: behavioral pins offline via mockito; live
long-session soak is credentialed/docker-gated — say so in the cell, date-stamp 2026-07-18).
Anchor R159 unchanged; `./scripts/check_matrix_anchors.sh` mandatory.

## Required pins (RED-under-mutation proven)

- Near-expiry request triggers exactly ONE refresh (single-flight under concurrency — spawn
  N concurrent requests at the expiry boundary, count token-endpoint hits via mockito).
- Fresh-token request does NOT re-hit the token endpoint (no gratuitous refresh).
- `token-refresh-enabled=false` → zero refresh traffic ever (today's behavior regression pin).
- Missing `expires_in` → the decoded Java behavior.
- Refresh failure → the decoded Java behavior (pinned).
- Mutations: disable the expiry check (never refresh) → near-expiry pin RED; drop single-flight
  → stampede pin RED; invert the enabled flag → regression pin RED. Restore byte-identical.

## Gates (ONE `&&` chain to commit; never `git add -A`)

typos (tracked) · `cargo fmt --all -- --check` · `cargo clippy -p iceberg-catalog-rest
--all-targets -- -D warnings` · `cargo test -p iceberg-catalog-rest --lib` ·
`./scripts/check_matrix_anchors.sh` · `./scripts/check_agent_artifacts.sh`.

## Constraints

CLAUDE.md + AGENTS.md bind (typed errors, no bare unwrap in production, documented lock order,
no guard across await unless bounded-justified). NO Cargo.toml/Cargo.lock edits. Edit surface:
`crates/catalog/rest/src/client.rs` (+ `catalog.rs` only for knob threading), rest tests,
`docs/parity/GAP_MATRIX.md` row R159. Commit trailers exactly:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_011wbhUQi8Ts3CAdgvbQm2r5`.

## Report back (final message = the record)

(1) decoded Java semantics with offsets (schedule rule, grant type, no-expires_in, failure
behavior, knob defaults); (2) the lazy-refresh adaptation and its observable-contract argument;
(3) changes file:line; (4) the R159 cell + grade rationale; (5) pin inventory incl. the
single-flight concurrency proof; (6) mutation table; (7) gate summary; (8) DEVIATIONS;
(9) named residues.
