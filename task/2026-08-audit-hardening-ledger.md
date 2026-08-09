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

# 2026-08 audit hardening — bundle ledger

Branch `fix/2026-08-audit-hardening`, rebased onto `e4f7f010` (#190) on 2026-08-09.
Charter: [todo.md](todo.md) § "ACTIVE (2026-08-08)" — clauses C-001…C-008, frozen; the user approved
the 8/8 proposition ledger on 2026-08-08.
Source audit: `~/Desktop/IcebergAudits/repo-audit-iceberg-rust-2026-08-08.md` (5 agents, 68
facet-level findings, 0 Critical / 10 High).

**This file is the artifact for the bundle.** SEPMO's artifact rule
([binding-manifest.md](../skills/sepmo/binding-manifest.md)) requires every gate to be a checkable
record rather than a self-report; its absence for G1 was itself a filed S2 (R-09 below). Nothing in
this file is taken from an agent's summary — every gate line was re-run, and every Critic verdict
cited here was produced by a fresh-context agent that reviewed the commit, not the report.

---

## 1. Provenance — a mid-flight handover, and why remediation units exist

G1 was committed and G2 left uncommitted by a prior session, which reported both converged. On
2026-08-09 a fresh session picked the work up and, before extending it, ran an independent review
(5 lenses × adversarial verification, 20 agents) over the committed G1 and the uncommitted G2.
**11 of 15 raised findings survived refutation** (§3). They are why this bundle carries remediation
units R1–R3 on top of G1/G2.

Two handover facts, recorded because both bear on what the prior "converged" claims were worth:

- **`cargo fmt --check` was RED at that tip** — import ordering in three files and four wrapped call
  sites. The gate had not been run on the uncommitted G2 work.
- **The G1 close-out claim had no artifact** — no ledger, no Critic disposition, no gate log, while
  every comparable prior unit in `task/` has one.

## 2. Scope amendments (Invariant V)

| Date | File | Why | Authority |
|---|---|---|---|
| 2026-08-09 | `crates/iceberg/src/expr/predicate.rs` | Audit finding **SAF-003**'s Location field literally reads `predicate_visitor.rs, bound_predicate_visitor.rs, Predicate::bind`, and C-003 says "bounded **predicate** … recursion". `Predicate::bind` recurses with no visitor on the path and runs first on untrusted filters (`scan/mod.rs` binds before any `visit`). G2's file list simply failed to enumerate the file its own clause names. R-03's panic is also unfixable without it. | User, explicit |
| 2026-08-09 | `crates/iceberg/src/spec/schema/id_reassigner.rs` | C-003's schema-evolution leg is undelivered without it: G2 hardened a walk already bounded by `SchemaBuilder`, while the one reachable from the public `add_column` was unbounded. | User, explicit |

**No dependency file was touched at any point** — `git diff --name-only e4f7f010..HEAD` contains no
`Cargo.toml`, no `Cargo.lock`, no crate manifest. Re-verified independently by four Critics.

### 2.1 Files touched outside a declared list — adjudicated, not waived

| File | Group | Adjudication |
|---|---|---|
| `crates/iceberg/src/spec/values/decimal_utils.rs` | R1 | ACCEPTED. `i128_from_be_bytes` is the decimal read primitive C-001 governs; the clause names the behaviour, the list missed the file. Filed as S3 by R1's Critic and carried here. |
| `crates/catalog/rest/src/catalog.rs` | G5 | ACCEPTED, test module only. G5's brief required the SEC-001 residue pin to be **inverted, not deleted**; that pin lives in `catalog.rs`, not `client.rs`. "Adjacent tests" covers it. |
| `crates/iceberg/src/io/storage/config/mod.rs` | G5 | ACCEPTED — the charter explicitly permits "a narrowly scoped shared redaction helper if required". |
| `crates/iceberg/src/expr/visitors/mod.rs` | R2 | ACCEPTED. Mechanical consequence of the iterative `rewrite_not`: `RewriteNotVisitor` lost its only production caller, so it and the unbound `predicate_visitor` became `#[cfg(test)]`. **No public API removed** — `expr/mod.rs` re-exports only `predicate::*` and `term::*`; the modules were `pub(crate)` and `PredicateVisitor` has no user outside them. `BoundPredicateVisitor` is untouched and still production. |
| `crates/iceberg/src/expr/visitors/map.md` | R2 | ALWAYS IN SCOPE — CLAUDE.md's `map.md` lockstep rule. |
| `crates/integrations/cache-moka/README.md` | G4 | ACCEPTED, documentation of a behaviour the clause mandates. Filed as S3-3(a) by G4's Critic and carried here. |

## 3. The 2026-08-09 independent review — findings and disposition

Severity is the **verifier's** re-grade after adversarial refutation, not the finder's.

| id | sev | where | claim | closed by |
|---|---|---|---|---|
| R-01 | S2 | `spec/values/datum.rs` | canonical-minimal decimal byte gate rejects padded bounds Java accepts; one bad bound fails the WHOLE manifest via `parse_bytes_entry`'s `?` | R1 |
| R-02 | S2 | `spec/datatypes.rs` | `decimal(P,S)` narrowing rejects type strings Java loads → table unopenable | R1 |
| R-03 | S2 | `expr/predicate.rs` | `rewrite_not`'s `.expect` turns the new depth error into a live panic from the public scan API | R2 |
| R-04 | S2 | `expr/visitors/predicate_visitor.rs` | `MAX_PREDICATE_DEPTH=100` below what the eq-delete fold builds | R2 |
| R-05 | S3 | `expr/visitors/predicate_visitor.rs` | `negate` / `Display` unbounded; doc overstates coverage | R2 (bounded, not narrowed) |
| R-06 | S2 | `transaction/update_schema.rs` | `assign_fresh_ids` unbounded on the caller-controlled `add_column` path | R3 |
| R-07 | S3 | `transaction/update_schema.rs` | the `index_parents` rationale is false for this codebase | R3 |
| R-08 | S2 | `expr/predicate.rs` | file outside G2's list, no Invariant V raised | §2 |
| R-09 | S2 | `task/todo.md` | G1's "four Critic cycles, CONVERGED" has zero artifact backing | this file |
| R-10 | S2 | `spec/datatypes.rs` | new rejection class recorded in no GAP_MATRIX row; R87 claims exact Java mirroring | R1 |
| R-11 | S2 | `expr/visitors/predicate_visitor.rs` | 102+-conjunct DataFusion filters now error; Java has no limit | R2 |

Refuted, recorded so they are not re-raised:

- *`run_end_encoded_decimal_rejects_wrapping_precision_scale` is vacuous* — the named mutation is an
  **equivalent mutant**: after `validate_decimal()` the arm is reachable only where `p as u8` /
  `s as i8` are lossless. A redundant guard, not a coverage hole.
- *`MetadataStripVisitor`'s new hooks silently change `strip_metadata_from_schema`* — behaviour does
  change (list/map schemas went `Err("field stack underflow")` → `Ok`), but an out-of-repo probe
  proved the new output correct: names, nullability and metadata-stripping all right.
- *rebase onto post-#190 main is not clean* — true, but caused by #190 landing after the branch was
  cut. Only `task/lessons.md` and `task/todo.md` conflicted (both append-only); no source overlap.
  Resolved combine-both 2026-08-09.
- *commit-message convention* — `CONTRIBUTING.md:43` exempts branch commits; only the PR title must
  conform, because the repo squash-merges.

## 4. Gate records

Per-unit gates were run by the Actor as one unpiped `&&` chain ending in the commit, and
independently re-run by that unit's Critic. Bundle gate in §4.2.

| unit | commits | cycles | Critic | residue |
|---|---|---|---|---|
| G1 (prior session) | `8b3ceef4` | — | no artifact (R-09) | superseded by R1 |
| G2 first pass | `de3961da` | — | fmt was RED; committed by the 08-09 session after `cargo fmt --all` | superseded by R2/R3 |
| R1 decimal parity | `ff53c252` `3b89c93a` | 2 | CONVERGED, zero S1/S2 | 6× S3 |
| R2 predicate depth | `fb11dc66` | 1 | CONVERGED, zero S1/S2 | 6× S3 |
| R3 schema recursion | `006dc721` `340fa4ea` | 2 | CONVERGED, zero S1/S2 | 5× S3 |
| G4 cache bytes | `cb489615` `73ff5157` | 2 | CONVERGED, zero S1/S2 | 6× S3 |
| G3 namespaces | `d99e56d6` `2556e109` | 2 | CONVERGED, zero S1/S2 | 5× S3 |
| G5 secrets | `879bf55e` | 3 | CONVERGED, zero S1/S2 | 3× S3 |

**Every one of the six units' first Critic cycle blocked on the same defect class**: a test whose
doc comment named a mutation it did not actually catch. In each case the Critic applied the named
mutation and the full suite stayed green. That is a systematic weakness in Actor self-certification,
not six coincidences — see the lessons entry of 2026-08-09.

### 4.1 What each unit actually changed

- **R1** — split the decimal surface into a documented READ door (Java-exact: `precision <= 38` and
  non-empty buffer, nothing else) and a WRITE/CONSTRUCTION door (keeps the anti-truncation gate on
  `Datum::to_bytes`, which really can silently corrupt). Java facts re-derived from
  `iceberg-api-1.10.0.jar` bytecode plus a compiled probe; the decisive one is that
  `Conversions.toByteBuffer` **writes** minimal while `fromByteBuffer` **reads** anything, which is
  exactly why a minimality gate on read was wrong. GAP_MATRIX **R87** cell corrected (not flipped).
- **R2** — `rewrite_not` ×2, `negate` ×2 and both `Display` impls are now explicit-stack, so the
  infallible `pub` surface is stack-safe at any depth with **no signature change**;
  `TableScanBuilder::with_filter` stays infallible and nothing ripples into DataFusion or a
  downstream pin. `MAX_PREDICATE_DEPTH` 100 → **1000**, re-derived by measurement: bisecting the
  depth that dies on the guard page at known `stack_size` gives `bind` 5,504 B/level dev, 964
  release; splitting the leaf arms into `#[inline(never)]` helpers first took that from 12,336 →
  5,504 (dev) and 1,800 → 964 (release), which is what makes a four-figure limit affordable. The
  arithmetic (2 MiB tokio worker ÷ 2 reserve ÷ 964 = 1,087 → 1000) is in the constant's doc.
  Cross-check: 1000 admits ~985 realistic eq-delete files versus ~87 before.
- **R3** — bounded `assign_fresh_ids` at 128, consistent with the crate's existing family
  (`MAX_SCHEMA_NESTING_DEPTH`, `MAX_AVRO_SCHEMA_DEPTH`, variant `MAX_NESTING_DEPTH`,
  `MAX_ARROW_SCHEMA_NESTING_DEPTH`), proved on a small-stack thread. Corrected the false
  `index_parents` rationale: that walk is reached only with an already-bounded `Schema`, so its
  explicit-stack form is defence in depth, not the fix for the reachable hazard.
- **G4** — byte weighting via `weigher` + `max_capacity`, mirroring `io/object_cache.rs`. Note the
  aggregate default ceiling is now 2 × 32 MiB = **64 MiB**, twice the core cache's single budget for
  the same two object kinds (merging them is ARCH-004, excluded) — a real number an operator
  inherits, documented on the constant, both constructors and the README.
- **G3** — nested namespaces discovered by explicit-queue BFS; identity preserved by joining on
  **U+001F**, the convention `NamespaceIdent::to_url_string()` already uses and REST + S3 Tables
  already rely on, so there is one flattening in the fork and `split('\u{1f}')` is a total inverse.
  A dot **alias** restores SQL typeability without making the canonical name ambiguous.
- **G5** — SEC-001 closed by replacing the raw `serde_json::Error` source with a
  `SanitizedJsonError` carrying category and position but nothing derived from the body, so the
  chain obligation survives; the former residue pin in `rest/catalog.rs` is **inverted**, not
  deleted. SEC-002/SEC-009 closed via `is_secret_prop_key` per-key redaction.

### 4.2 Bundle gate (C-008)

Run 2026-08-09 at tip `879bf55e` + this commit, in `iceberg-rust-ws`, each step unpiped.

| # | step | result |
|---|---|---|
| 1 | `typos` | **OK** — see caveat below |
| 2 | `make check` (fmt · clippy `--all-targets --all-features --workspace -D warnings` · taplo · cargo-machete · agent-artifacts · matrix-anchors) | **OK** — anchors sound, 75 rows, IDs unique, citations resolve, 5-pipe audit green |
| 3 | `make check-msrv` (`cargo +1.94 check --workspace`) | **OK** |
| 4 | `cargo build -p iceberg --no-default-features` | **OK** |
| 5 | `cargo deny check advisories` | **OK** |
| 6a | `cargo test --no-fail-fast --doc --all-features --workspace` | **OK**, rc 0 |
| 6b | `cargo test --no-fail-fast --all-targets --all-features --workspace` | **4210 passed / 45 failed** — every failure infrastructure, see below |
| 7 | `make test` | **NOT RUN** — see below |

Per-crate unit tests for everything this bundle touched, all green:
`iceberg` **3232**/0 · `iceberg-datafusion` 191/0 · `iceberg-catalog-rest` 105/0 ·
`iceberg-cache-moka` 10/0. (The last of these is the gap G4's Critic filed as S3-2: the charter's
declared gate ends in `cargo test -p iceberg --lib`, which filters to the `iceberg` package and
never runs cache-moka's ten tests. Closed here by running the whole workspace.)

**The 45 failures — classified, not waived.** The Docker daemon was unavailable at bundle close
(client present, context `desktop-linux`, daemon down), so `make test`'s `docker-up` prerequisite
could not be satisfied and step 7 was replaced by 6a+6b. All 45 failures land in exactly eight
`tests/*.rs` integration binaries and **none in any `unittests src/lib.rs`**:
`conflict_commit_test`, `file_io_gcs_test`, `file_io_s3_test`, `glue_catalog_test`,
`hms_catalog_test`, `read_evolved_schema`, `read_positional_deletes`, `rest_catalog_test`.
Every one fails with `Connection refused` / `Os { code: 111 }` against the REST fixture, MinIO,
fake-gcs or HMS. 119 `Connection refused` occurrences and zero assertion failures.

Docker enters this gate transitively and at exactly one point: `Makefile:80` is `test: docker-up`,
and `docker-up` (`Makefile:95`) brings up `dev/docker-compose.yaml`. No earlier unit gate touches it.
The six services map one-to-one onto the eight failing binaries — `minio` → `file_io_s3_test`,
`fake-gcs-server` → `file_io_gcs_test`, `rest` → `rest_catalog_test` + `conflict_commit_test`,
`spark-iceberg`+`provision` → `read_evolved_schema` + `read_positional_deletes`, `moto` →
`glue_catalog_test`, `hive-metastore` → `hms_catalog_test` — so the shortfall is bounded and
enumerated, not open-ended.

**Runner caveat on the substitution.** `make test` runs `cargo nextest run --all-targets
--all-features --workspace`; step 6b ran `cargo test --no-fail-fast` with the same target, feature
and workspace scope. Same test SELECTION, different runner: nextest executes each test in its own
PROCESS, `cargo test` uses threads within one process per binary. That cannot change which tests
run, but it can mask a test that depends on process isolation. Re-running under nextest is part of
the pre-merge `make test`, not an extra step.

**What that leaves genuinely unverified**, stated plainly rather than assumed benign: the REST
catalog end-to-end suite is the one that exercises G5's changed error paths against a live server,
so SEC-001/SEC-009 are pinned by unit tests and not by an end-to-end run. `read_evolved_schema` and
`read_positional_deletes` exercise read paths adjacent to R1's decimal decode. Re-run
`make test` with Docker up before merge.

**typos caveat.** The invocation was `typos --exclude repark-grok-catchup .`. That file is an
untracked ~37k-line transcript in the repo root, unrelated to this work and never staged; a bare
`typos .` fails on it. CI is unaffected — it only sees tracked files — and `git ls-files | xargs
typos` is not equivalent because passing explicit paths bypasses `.typos.toml`'s
`extend-exclude = ["**/testdata", "CHANGELOG.md"]`.

## 5. Named residue — 31 S3 items carried forward

Full text in the per-unit Critic verdicts; the load-bearing ones:

- **No interop artifact for R1.** Everything is unit-level plus a live Java oracle run in-session.
  The parity mandate requires an interop test for a ✅ flip — which is why R87 was **corrected, not
  flipped**. The natural artifact is a Java-written table with a zero-padded decimal bound plus a
  manifest-level scan assertion; that needs a test in `spec/manifest/_serde.rs` or a fixture, both
  outside every declared list.
- **`Drop` is still recursive.** R2 made five walks iterative, but the derived `Drop` on a very deep
  `Predicate` still recurses. Nothing in this bundle can protect it.
- **`with_manifest_list_cache` is unpinned** (G4 S3-1) — the caller-builder-wins contract the commit
  newly documents is unverified; its manifest twin *is* pinned. The precondition is now satisfied
  (`manifest_list_with_entries(n)` exists), so this is a two-line addition.
- **The charter's declared gate never runs G4's tests** (G4 S3-2) — they live in
  `iceberg-cache-moka`, and `cargo test -p iceberg --lib` filters to `iceberg`. `clippy
  --all-targets` compiles but does not run them. Addressed for real in §4.2's bundle gate; the
  charter text itself still carries the gap.
- **`RawLiteral::try_from`'s `Int128` write arm** no longer enforces a decimal type while its
  comment and error message still say it does (R1 S3-1). Not a regression against the merge base
  (`e4f7f010` had no check at all there), and it fails closed downstream at Avro schema resolution.
- **`Table`'s `{:?}` is not wholesale credential-safe.** G5 closed `metadata.properties` and the
  FileIO config, but `snapshots[*].summary.additional_properties`, `encryption_keys[*].properties`
  and `statistics[*].blob_metadata[*].properties` still render in clear. The rustdoc now says so
  explicitly instead of asserting a blanket invariant. (`partition_statistics` was named in the
  finding but is **clean** — it carries no property map; the doc says that too.)

## 6. Follow-up seeds

- `DeleteFilter::build_equality_delete_predicate` left-folds one conjunct per eq-delete file
  (`delete_filter.rs:737`), so predicate depth is linear in delete-file count. A balanced fold makes
  it logarithmic and removes the pressure on `MAX_PREDICATE_DEPTH` entirely. `delete_filter.rs` is
  in no group's list.
- Docker was unavailable at bundle close, so the integration suites in `make test` did not run —
  see §4.2 for exactly what that leaves unverified.
- The untracked `repark-grok-catchup` transcript in the repo root breaks a bare `typos .` locally
  (CI is unaffected — it only sees tracked files). Move it out of the working tree.
