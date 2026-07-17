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

# Unit brief — G2: incremental-scan name-mapping pin (test-only)

**Charter (2026-07-17, user-signed; FF Actor–Critic).** Close the single named residue
from the #154 (BUG-002) Critic: the incremental-append scan's name-mapping wiring
(`scan/incremental.rs`, `name_mapping: parse_name_mapping(self.table.metadata())?` in
`IncrementalAppendScanBuilder::build`) is correct but has NO test — every #154 pin
drives the snapshot scan, so a refactor could drop the incremental parse with zero red
tests. This unit is **TEST-ONLY**: it adds the missing pins; production code changes
are OUT of scope unless a pin proves the wiring is actually broken (in which case: fix
+ own pin + loud callout, same escape hatch as #154).

## Proposition ledger

- **C-1 (plan-level pin):** an `IncrementalAppendScan` planned over a table whose
  metadata carries `schema.name-mapping.default` yields tasks that ALL carry the parsed
  mapping (assert mapping CONTENT — field names/ids — not just `is_some()`), mirroring
  the snapshot-scan pin from #154.
- **C-2 (e2e contrast pin):** an ID-less Parquet data file appended WITHIN the
  incremental range of a name-mapped table is read to the CORRECT columns through the
  incremental scan's stream, in a scenario where position-fallback would produce
  provably WRONG data (reuse the #154 contrast-fixture technique from
  `scan/mod.rs` — mapped names deliberately misaligned with naive positional order).
- **C-3 (absent-property pin):** the incremental scan over the same shape WITHOUT the
  property still reads via position fallback (regression guard for the `None` path).
- **C-4 (mutation proof, run live):** hardcode `name_mapping: None` at the
  `incremental.rs` wiring site → C-1 and C-2 MUST go RED while the #154 snapshot pins
  stay GREEN (proving the new pins uniquely guard the INCREMENTAL site); restore →
  GREEN. Record the exact RED sets.
- **C-5 (placement/style):** pins live beside the existing incremental-scan tests
  (find them — `scan/incremental.rs` tests or the scan test module used by #154) and
  reuse the #154 helper/fixture code rather than duplicating it (extract a shared
  helper if needed — test-code-only refactor is in scope).

## Boundaries (not in scope)

- NO production code changes (escape hatch: only a proven-broken wiring, per above).
- NO interop suite changes, NO floor bump (the #154 suite already covers cross-engine;
  this is an in-crate coverage close). NO matrix edit unless a cell claims incremental
  name-mapping coverage that is false today (verify row R143's cell wording; edit at
  most that one cell if it over-claims).
- NO Cargo.toml/Cargo.lock/pom.xml. NO Avro/ORC.

## Gate (unit)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg --lib --tests -- -D warnings` · `cargo test -p iceberg --lib`
(2781+ baseline, no regressions) · `./scripts/check_matrix_anchors.sh` ·
`./scripts/check_agent_artifacts.sh`. Chain gate→commit in ONE `&&` chain; explicit
paths, never `git add -A`.
