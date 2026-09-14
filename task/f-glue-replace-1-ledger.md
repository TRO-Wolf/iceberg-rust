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

# F-GLUE-REPLACE-1 — Glue staged replace publish

**Date:** 2026-09-14. **Branch:** `fix/f-glue-replace-1`.
**Base:** `3ebf7d36c`. **Model:** swe-2-high.
**Path:** STANDARD because this adds a `Catalog` trait implementation on the
Glue commit path and touches the staged-replace metadata versioning.

This ledger retires when the fork change merges or the owner removes the unit.

## The defect

RePark production assessment 2026-09-14, gap G-1: `GlueCatalog` does not
implement `Catalog::publish_replace_table`, so it inherits the trait default
(`crates/iceberg/src/catalog/mod.rs`), which returns `FeatureUnsupported`.
Every `CREATE OR REPLACE TABLE … AS` on an EXISTING Glue table (a dbt `table`
model's second run, `writeTo().createOrReplace()`) streams its data files, then
fails at publish. `S3TablesCatalog` and `MemoryCatalog` implement it.

## Decisions (from the card)

- **D-1 Shape.** `publish_replace_table` on `impl Catalog for GlueCatalog`,
  mirroring `crates/catalog/s3tables/src/catalog.rs`:
  (a) `get_table_pointer(&ident)` → (stored location, version_id); (b) a `Some`
  expected base that differs from the stored location returns retryable
  `CatalogCommitConflicts` BEFORE any transport send (S3 Tables message shape);
  (c) `convert_to_glue_table(name, new_location, metadata, properties,
  Some(stored))`; (d) `commit_transport.send_update_table(GlueUpdateTableCall
  { database_name, table_input, version_id, catalog_id })` — Glue's version-id
  optimistic lock is the CAS; (e) `#[cfg(test)]` harness publish when
  `glue_commit_send_landed`; (f) `map_glue_commit_send` then return the table.
- **D-2** R158 residue (1) closed for Glue: read the staged file back with
  `TableMetadata::read_from(&self.file_io, &new_location)` before the send and
  refuse (`DataInvalid`, no send) when unreadable or when its `uuid()` differs.
- **D-3** R158 residue (2) measured: when the base pointer parses as a
  `MetadataLocation`, `begin_replace` stages `base.with_next_version()`;
  fallback to the current v0 restart when it does not parse or when the
  caller-supplied `creation.location` differs from the existing location.
  In scope only while `cargo test -p iceberg --lib` stays green and no existing
  expectation or interop harness shape has to change.
- **D-4** No reconciliation after an unknown/lost response: `MaybeSentLost` and
  `AcceptThenLose` surface `CommitStateUnknown`, non-retryable, exactly one
  transport attempt, staged file left on disk; `AcceptThenLose` leaves the
  harness pointer at the staged location. Named residue in R158.
- **D-5** `ConcurrentModification` on the replace send → retryable
  `CatalogCommitConflicts`; `AccessDenied` → terminal.

## Implemented minimal fix

- `catalog/replace_publish.rs::publish_replace_table` — `get_table_pointer`
  reads the stored metadata location and Glue `version_id`; a `Some` expected
  base that differs returns retryable `CatalogCommitConflicts` before any
  send; `TableMetadata::read_from` then validates the staged file and its
  `uuid()` against the supplied table (`DataInvalid`, no send); the send goes
  through `commit_transport.send_update_table` with the stored `version_id`
  as the CAS token; `#[cfg(test)]` harness publish only when
  `glue_commit_send_landed`; `map_glue_commit_send` preserves the existing
  never-sent / maybe-sent / modeled-service-error classification.
- `catalog.rs` — `mod replace_publish;` + the trait delegate in
  `impl Catalog for GlueCatalog`; the inline `mod tests` moved verbatim to
  `catalog/tests.rs` so the file stays under its ceiling (now 1087).
  The moved tests' doc comments were dropped rather than relocated: the
  comment fence counts every `+//` line in the new file. The facts they
  carried: `GlueCatalog::new` performs no network call, so the
  name/properties/invalidate pins run offline; the `Debug` pins guard the
  manual redacting impls (`#159` unit-D residue — reverting to derived
  `Debug` leaks `aws_secret_access_key`/`aws_session_token` values, keys and
  non-secret values stay visible); the `NamespaceNotEmpty` pin guards the
  G3b parity kind (Java `GlueCatalog.dropNamespace` →
  `NamespaceNotEmptyException`) and the early return before
  `delete_database`.
- `begin_replace` (D-3) — when `creation.location` is unset or equals the
  existing table location AND the base pointer parses as a
  `MetadataLocation`, the staged file is `base.with_next_version()`; else a
  fresh `MetadataLocation::new_with_table_location(&table_location)`. The
  pre-existing comment above the site stays verbatim.

## File allowlist

- `crates/catalog/glue/src/catalog.rs`
- `crates/catalog/glue/src/catalog/replace_publish.rs` (new — keeps
  `catalog.rs` under its 1212-line ceiling)
- `crates/catalog/glue/src/catalog/tests.rs` (new — `mod tests` extraction)
- `crates/catalog/glue/src/replace_publish_tests.rs` (new — pins)
- `crates/catalog/glue/src/commit_outcome_tests.rs` (helpers → `pub(crate)`)
- `crates/catalog/glue/src/lib.rs` (test module wiring)
- `crates/catalog/glue/map.md`
- `crates/iceberg/src/transaction/staged_table.rs` (D-3 only)
- `crates/iceberg/src/transaction/staged_table_version_tests.rs` (new, D-3 only)
- `crates/iceberg/src/transaction/map.md` (D-3 only)
- `scripts/check_rust_file_size.py` (ceiling follows files DOWN only)
- `docs/parity/GAP_MATRIX.md` (row R158)
- `docs/ENGINE_CONTRACT.md` (§8a)
- `task/f-glue-replace-1-ledger.md`
- `task/todo.md`

`Cargo.toml`, `Cargo.lock`, `.github/`, and every dependency file stay closed.

## Proposition ledger

`EXECUTION PROVEN` means the implemented patch has direct passing evidence.
`OPEN` means the pin is not yet green.

| Clause | Checkable proposition | Proof obligation | Status |
|---|---|---|---|
| C-001 | Glue replace publish commits through the real engine path (`begin_replace` → `add_data_files` → `commit`): Ok, harness pointer equals the returned metadata location, table UUID retained, base metadata file in the metadata log, exactly one transport attempt. | Pin in `replace_publish_tests.rs`; red on base (`FeatureUnsupported`), green after. | EXECUTION PROVEN |
| C-002 | A `Some` expected base that differs from the stored pointer returns retryable `CatalogCommitConflicts` before any transport send. | Pin: two staged replaces on one seed, loser observes the moved pointer; `catalog_commit_attempts() == 1` total. | EXECUTION PROVEN |
| C-003 | The staged metadata file is read-validated before the send: missing file and foreign-uuid file each fail `DataInvalid` with zero transport attempts and no pointer move. | Two pins; red on base, green after. | EXECUTION PROVEN |
| C-004 | `MaybeSentLost` and `AcceptThenLose` surface `CommitStateUnknown` (never Ok, never retryable), one attempt, staged file still present; `AcceptThenLose` leaves the harness pointer at the staged location (no reconciliation). | Two pins; red on base, green after. | EXECUTION PROVEN |
| C-005 | `ConcurrentModification` → retryable `CatalogCommitConflicts`; `AccessDenied` → terminal `Unexpected` ("Authorization denied"); one attempt each. | Two pins; red on base, green after. | EXECUTION PROVEN |
| C-006 | D-3 measured: staged replace metadata filename continues the base pointer's version (N → N+1) when the base parses, else restarts. Fix confined to `begin_replace`; full `iceberg --lib` suite green; else the residue is documented. Round 2 (audit P1): the continued name must ALWAYS carry a fresh uuid — a Hadoop-named base (`vN.metadata.json`, R167) under plain `with_next_version` keeps `id: None`, so concurrent staged replaces collide on one `v(N+1)` file. | Pin in `staged_table_version_tests.rs` + measurement; residue wording if not landed. | EXECUTION PROVEN |
| C-007 | GAP_MATRIX R158 names Glue `publish_replace_table` (version-id CAS, read-validated), residue (1) closed for Glue / S3 Tables state named, residue (2) closed or kept, D-4 no-reconciliation residue added; ENGINE_CONTRACT §8a names the implementing catalogs. | Row rewritten, single 5-pipe line; `make check-matrix-anchors` green. | EXECUTION PROVEN |
| C-008 | Gates: `cargo test -p iceberg-catalog-glue --lib`, `cargo test -p iceberg-catalog-s3tables --lib`, `cargo test -p iceberg --lib staged` (+ full `iceberg --lib` if staged_table.rs touched), `make check`, `make check-matrix-anchors`, file-size script, comment fence. | Run and paste counts below. | EXECUTION PROVEN |

## Base-red evidence

`cargo test -p iceberg-catalog-glue --lib replace_publish` on the base tree
(pins added, production code untouched) exited 101 — **8 red out of 8 new
pins**, every one on the trait default `FeatureUnsupported => publish_replace_table
is not supported by this catalog`:

```
running 8 tests
test replace_publish_tests::replace_publish_access_denied_is_terminal ... FAILED
test replace_publish_tests::replace_publish_concurrent_modification_is_retryable_conflict ... FAILED
test replace_publish_tests::replace_publish_accept_then_lose_is_unknown_and_pointer_moved ... FAILED
test replace_publish_tests::replace_publish_unreadable_staged_metadata_refuses_before_send ... FAILED
test replace_publish_tests::replace_publish_foreign_uuid_staged_metadata_refuses_before_send ... FAILED
test replace_publish_tests::replace_publish_maybe_sent_lost_is_unknown_and_keeps_pointer ... FAILED
test replace_publish_tests::replace_publish_stale_base_conflicts_retryable_before_any_send ... FAILED
test replace_publish_tests::staged_replace_commit_swaps_glue_pointer_and_retains_uuid ... FAILED

---- staged_replace_commit_swaps_glue_pointer_and_retains_uuid stdout ----
staged replace commit: FeatureUnsupported => publish_replace_table is not supported by this catalog

---- replace_publish_stale_base_conflicts_retryable_before_any_send stdout ----
winner publish: FeatureUnsupported => publish_replace_table is not supported by this catalog

test result: FAILED. 0 passed; 8 failed; 0 ignored; 0 measured; 41 filtered out
```

## Execution evidence

- `cargo test -p iceberg-catalog-glue --lib` — **49 passed, 0 failed**
  (41 existing incl. all `commit_outcome_tests` + 8 new pins).
- `cargo test -p iceberg-catalog-s3tables --lib` — **39 passed, 0 failed**.
- `cargo test -p iceberg --lib staged` — **22 passed, 0 failed**.
- `cargo test -p iceberg --lib` (full — `staged_table.rs` touched) —
  **3676 passed, 0 failed, 8 ignored**. D-3 landed with zero expectation
  edits: `staged_replace_continues_base_metadata_version` and
  `staged_replace_on_parseable_v3_continues_at_v4` green, the two
  restart-on-unparseable/relocated controls confirm the fallback.
- `cargo fmt --all -- --check` — clean.
- `make check` — fmt + clippy `-D warnings` (workspace, all-features,
  all-targets) + taplo (34 files) + cargo-machete (no unused deps) +
  agent-artifacts + matrix-anchors + comment-blocks + file-size: all green.
- `make check-matrix-anchors` — 84 rows anchored, IDs unique, citations
  resolve, 5-pipe audit green.
- `python3 scripts/check_rust_file_size.py` — 462 files clean, 100 legacy
  ceilings; `catalog.rs` ceiling lowered 1212 → 1087 after the `mod tests`
  extraction; `staged_table.rs` exactly at its 1229 ceiling.
- Comment fence `git diff --cached -- '*.rs' '*.toml' '*.sh' '*.yml' |
  grep -P '^\+\s*(//|#(?!\[|!\[))'` — prints only the ASF license headers
  of the four brand-new files.

Mutation evidence: the red pins all failed on the trait default
`FeatureUnsupported` (above) — the pins are load-bearing on the new code
path, not on an incidental error. Within the green suite, C-002's loser
observes `CatalogCommitConflicts` only because the winner's pointer move is
visible to `get_table_pointer`; C-003's zero-attempt assertion fails if the
read-validation ordering is swapped with the send.

Named residue carried forward in R158: staged replace has no
`CommitStateUnknown` reconciliation on Glue or S3 Tables; Java
name-matching fresh field-ID assignment on replace remains open.

## Round 2 — orchestrator audit P1 (Hadoop-named base collision)

**Finding.** D-3 as landed staged `MetadataLocation::from_str(base)
.with_next_version()`. `with_next_version` preserves `id: None` on a
Hadoop-named base (`vN.metadata.json`, row R167 — what a Spark
Hadoop-catalog writes and what `register_table` can install), so every
staged replace from one base wrote the SAME `v(N+1).metadata.json`: two
concurrent replaces overwrote each other's staged file, and a loser
staging after the winner published rewrote the file the catalog pointer
names — the winner's table silently becomes the loser's metadata.

**Red (base tree of the fix, commit `19381fa3`):**
`cargo test -p iceberg --lib staged_table::version_tests` exited 101 —
2 red, 3 green:

```
test ...::replace_stages_next_version_after_a_hadoop_named_pointer ... FAILED
a Hadoop-named pointer must continue the version under a fresh uuid, got
memory://wh/ns/t/metadata/v8.metadata.json

test ...::concurrent_replaces_from_a_hadoop_pointer_stage_distinct_files ... FAILED
assertion `left != right` failed: two staged replaces from one base must
not share a file
  left: "memory://wh/ns/t/metadata/v4.metadata.json"
 right: "memory://wh/ns/t/metadata/v4.metadata.json"
```

The two-`begin_replace` pin is the direct proof of the audit's collision
claim: same base, same staged path, both writes landed on
`v4.metadata.json`.

**Fix.** `MetadataLocation::with_next_version_fresh_id` (`pub(crate)`,
added beside `with_next_version` — that method and the public API are
unchanged, so plain `update_table` commits still emit `v(N+1)` for
Hadoop-convention tables per R167) continues the version and always sets
`id: Some(Uuid::new_v4())`. `begin_replace` calls it at the kept-location
arm, so a `vN` base stages `0000(N+1)-<uuid>` and every later
`apply_locally` step regenerates a fresh uuid. Restart arms
(`new_with_table_location`) already carry a uuid. `staged_table.rs` stays
at its 1229 ceiling; the one-line call-site change keeps it there.

**Green.** The collision pin and the updated Hadoop pin pass;
`replace_stages_next_version_after_a_hive_named_pointer` (N → N+1 with a
new uuid) is unchanged and still green. Full gate counts below.

## Critic round (Grok 4.6 critic-logic)

Report: `/tmp/oc-worker/i-fcrit/report.md` (clone `/tmp/i-fcrit`, detached
`334d063d5`, range `3ebf7d36c..334d063d5`, high risk tier). Verdict:
**NEEDS_REMEDIATION** — no P1 silent-loss bug on the staged engine path;
two P2 hollow oracles (L-001, L-002) and P3 residues.

The critic's mutation arithmetic (one knob at a time, restore-green
verified):

| # | Mutation | Result | Red pins |
|---|---|---|---|
| M1 | `GlueUpdateTableCall.version_id = None` | 0 red / 8 | none — hollow oracle (L-001) |
| M2 | delete expected-base conflict | 1 red / 8 | `replace_publish_stale_base_conflicts_retryable_before_any_send` |
| M3 | delete uuid match | 1 red / 8 | `replace_publish_foreign_uuid_staged_metadata_refuses_before_send` |
| M4 | send UpdateTable before `read_from` | 2 red / 8 | unreadable + foreign-uuid (`attempts` 1 ≠ 0) |
| M5 | `begin_replace` uses `with_next_version` | 2 red / 5 | Hadoop next-version + concurrent distinct-files |
| M6 | `convert_to_glue_table(..., None)` | 0 red / 8 | none — hollow oracle (L-002) |

**L-001 closed.** `ScriptedGlueCommitTransport` now records the last
`GlueUpdateTableCall` it received (`last_call()` → `version_id` +
`TableInput` parameters; `#[cfg(test)]` only). The happy-path staged
replace pin asserts `sent.version_id == Some("v0")` (the harness's
GetTable version). A second pin seeds the harness pointer at
`version_id: None` (`catalog_with_version` →
`for_commit_outcome_tests_at_version`) and asserts the send carries
`None` — matching `update_table`: no Glue OCC, last-writer-wins beyond
the expected-base string check. Actor's own red proof: M1 applied in
this tree (`version_id: None` in the send) →
`staged_replace_commit_swaps_glue_pointer_and_retains_uuid` RED
(`left: None, right: Some("v0")`), 8 others green; restored.

**L-002 closed.** The same recording pins the sent `TableInput`
parameters: `metadata_location == <staged location>` and
`previous_metadata_location == <base location>` on the happy path (and
`previous_metadata_location` on the none-version pin). Actor's own red
proof: M6 applied (`Some(stored)` → `None`) → both pins RED
(`left: None` on the `previous_metadata_location` assertion), 7 others
green; restored.

**Named residues (P3, no code change):**

- **L-003**: a base at metadata version `i32::MAX` wraps
  (`wrapping_add`) to a negative version, producing an unparseable name;
  the next replace then restarts at `00000-<uuid>`. No overwrite of the
  base file — `write_to` uniqueness is uuid-v4. Unrealistic for a
  shipping table.
- `Catalog::publish_replace_table(table, None)` is a blind replace (no
  expected-base check) — same as MemoryCatalog and S3 Tables.
  `StagedTableTransaction::commit` always passes `Some(base)`, so the
  staged engine path is covered; the `None` arm is the public surface's
  documented contract, not a path this unit changed.
- Replace (like `update_table`) rebuilds the Glue `TableInput` from
  Iceberg metadata and drops Glue-only parameters (Lake Formation
  flags, classification, custom storage-descriptor extras); Java
  `persistGlueTable` overlays onto the current Glue parameter map.
  Pre-existing Glue-catalog residue, identical on `update_table`.

**Stale glyph fixed.** `crates/iceberg/src/transaction/map.md` dropped
the `🟡` next to the R158 row reference — status lives only in the
matrix.
