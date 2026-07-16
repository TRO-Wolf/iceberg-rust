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

# Unit brief — R158 staged create/replace Java interop battery (🟡→✅)

**Charter (2026-07-16, user-signed execution mode: OO AC, both Opus at max effort).**
Prove bidirectional Java-1.10.0 interop for the staged create/replace table transaction
(GAP_MATRIX row R158, merged #149) through the interop oracle, and flip the row ✅.
`dev/java-interop/README.md` is the authoritative harness contract (directions, fixture
flow, structural-comparison semantics); this brief scopes WHAT to prove, not HOW the
harness works.

## Proposition ledger

- **C-1 (Direction 1, Java→Rust — create):** Java `Catalog.newCreateTableTransaction`
  (HadoopCatalog or equivalent local-fs catalog, per harness precedent) stages a table
  with appended data and commits; Rust reads the published metadata and scan-verifies
  row content. Success: Rust sees exactly the committed table (schema, properties,
  location, one publish — no intermediate catalog states).
- **C-2 (Direction 1, Java→Rust — replace):** Java `newReplaceTableTransaction` over an
  existing table with ≥2 snapshots, run for ≥2 replace cycles; Rust reads each published
  metadata and asserts the **replace invariant set** (enumeration below, `E-INV`).
- **C-3 (Direction 2, Rust→Java — create):** Rust `StagedTableTransaction::begin_create`
  + `add_data_files` + `commit` against a local-fs-backed catalog publishes metadata +
  data that Java (`TableMetadataParser` + a table scan) reads and verifies: schema,
  row content, and single-publish semantics.
- **C-4 (Direction 2, Rust→Java — replace):** Rust `begin_replace` over an existing
  table, ≥2 cycles; Java reads each published metadata and asserts the same `E-INV` set,
  proving Rust's `buildReplacement` port is Java-readable and semantically identical.
- **C-5 (Cross-check):** for one equivalent create+replace scenario executed
  independently by both sides, the two published metadata files are **structurally
  equivalent** (harness comparison semantics — field-level, not byte-for-byte;
  timestamps/UUID values excluded, UUID *stability across replace* included).
- **C-6 (Format-version directive):** both directions prove the property-directed
  format-version contract: absent property ⇒ existing version preserved (a V1 table
  stays V1 through a replace on BOTH sides); explicit `format-version=2` ⇒ upgrade, and
  the directive does NOT appear in the published table properties on either side.
- **C-7 (Sabotage, hard-fail-never-skip):** ≥2 sabotage steps prove the verifiers are
  non-vacuous — e.g. corrupt the retained UUID (or splice a fresh one) in a copied
  fixture and prove the verifier goes RED; drop the retained snapshot history and prove
  RED. A sabotage step that cannot be applied must exit non-zero, never SKIP (CLAUDE.md
  working convention).
- **C-8 (Standing regression net):** one new suite `dev/java-interop/run-interop-staged-txn.sh`
  discovered by `scripts/run_interop_suites.sh`; `SUITE_FLOOR_DEFAULT` bumped 48→49 in
  the same change (driver contract); `dev/java-interop/map.md` + README scenario table
  updated in lockstep.
- **C-9 (Status flip):** GAP_MATRIX row R158 🟡→✅ (date-stamped, interop suite cited),
  named residue (1)–(3) retained verbatim in the ✅ cell; `docs/ENGINE_CONTRACT.md` §8a
  point 5 ("Java interop battery for R158 is a follow-up (status 🟡)") updated to done.
  Edit ONLY the R158 row cell; `./scripts/check_matrix_anchors.sh` green.

### E-INV — the replace invariant set (enumeration for C-2/C-4; complete because it is
the full guarantee list of the R158 cell + ENGINE_CONTRACT §8a replace contract)

1. `table_uuid` identical before/after every replace cycle.
2. Pre-replace snapshots present in the published metadata (history retained).
3. `metadata_log` grows (appended, never truncated) across cycles.
4. `main` ref reset: current snapshot is the replace's own new state (or none before
   first append), never a pre-replace snapshot; reads through main expose ONLY the
   latest replace's rows.
5. `location()` stable across cycles (no staged suffix, no drift).
6. Format version preserved absent an explicit directive (C-6 overlaps; assert here per
   cycle).
7. `last_column_id` monotonic (never reduced below the pre-replace value).

## Boundaries (not in scope)

- NO real-catalog (`REST/Glue/S3Tables/SQL`) `publish_replace_table` wiring — residue (1)/(2)
  stay named residue.
- NO `assignFreshIds` base-aware helper — residue (3) stays named; the cross-check
  scenario uses name/id-aligned schemas.
- NO changes to `staged_table.rs` production code UNLESS an interop failure proves a
  genuine parity bug (that is the point of the battery — if found, fix ships in this
  unit with its own pin, and the finding is called out in the build summary).
- NO Cargo.toml/Cargo.lock edits. Java side lives in the existing Maven test module.

## Gate (unit)

`git ls-files -z | xargs -0 typos --force-exclude` · `cargo fmt --all -- --check` ·
`cargo clippy -p iceberg --lib --tests -- -D warnings` · `cargo test -p iceberg --lib` ·
the new suite green end-to-end via `scripts/run_interop_suites.sh --only run-interop-staged-txn.sh`
(and `--selftest` untouched-green) · `./scripts/check_matrix_anchors.sh` ·
`./scripts/check_agent_artifacts.sh`. Chain gate→commit in ONE `&&` chain; explicit
paths, never `git add -A`.
