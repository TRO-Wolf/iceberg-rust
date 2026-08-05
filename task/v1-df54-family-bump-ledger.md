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

# V1 — DF 54 family bump ledger

**Unit:** V1 FORK DF-54 FAMILY BUMP  
**Branch:** `chore/df54-family-bump-recut`  
**Base:** `bde2e95d` (main @ #185) — **re-cut 2026-08-05**, see below  
**Floors:** datafusion 54.1.0 / arrow*+parquet 58.4 / package identity 0.9.1  

## Coupled deps (family-adjacent)

| Dep | Change | Why |
|---|---|---|
| `home` | `=0.5.11` → `0.5.12` | datafusion-cli 54 / rustyline |
| `orc-rust` | 0.7 → 0.8 | arrow 58 only |
| `serde_arrow` feature | arrow-57 → arrow-58 | dual-version RecordBatch break |
| `sqllogictest` | 0.28.3 → 0.29 | match datafusion-sqllogictest 54 |
| `rust-toolchain` | nightly-2025-10-27 → nightly-2026-03-05 | AWS SDK / fastnum MSRV 1.94+ |
| `rust-version` | 1.92 → 1.94 | align MSRV |

## Cherry-pick / hand-port

| Upstream SHA | Applied how |
|---|---|
| `477a1e525` (#2206 DF53+Arrow58) | Pattern port (floors, row-group API, date deprecations) |
| `875fdb746` (#2648 DF54) | Pattern port (`as_any` strip, Cast.field, Arc PlanProperties) |
| `e8460eee` (#2872 DF54.1) | Floors 54.1.0 |
| G1 / delete / write / ScanKnobs | Hand-port compile fixes only |

## Status (HISTORICAL — pre-recut epoch, superseded)

> Everything from here to "Re-cut onto main" records the ORIGINAL branch off `bc4ffa19`, which was
> superseded on 2026-08-05. Its counts and states do NOT describe the shipped tree — the live gate
> is "Re-cut gate (at tip)" below. Retained for provenance only.

- Actor build: in progress → cargo check --workspace --all-targets green (pre-test)
- Octo: pending

## Actor gate (pre-octo, HISTORICAL)

| Gate | Result |
|---|---|
| `cargo check --workspace --all-targets` | green |
| `cargo test -p iceberg --lib --all-features` | **3068 passed** (original branch; the shipped tree is 3137 — see the re-cut gate) |
| `cargo test -p iceberg-datafusion --all-features` | green (incl. nested insert re-pin + doctest) |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | green (new nightly lints auto-fixed) |
| `make interop` | deferred to octo / pre-push (needs JDK/mvn) |

### Behavior re-pins (cited)

1. **test_insert_into_nested** — DF54 field-aware `CastExpr` + `validate_field_compatibility` rejects nullable SQL `named_struct` leaves → non-null nested Iceberg required fields. Fixture leaves under `address` re-pinned OPTIONAL; zip literal cast to INT; expect! snapshot updated. Citation: DF 54 upgrade guide (CastColumnExpr → field-aware CastExpr); datafusion-common `nested_struct.rs` nullability rule. Required-nested SQL insert follow-up out of family-bump scope.

## Octo (HISTORICAL)

Pending: 8× critic-octo early_stop=false — never run on the original branch; superseded by the
independent Critic pass on the re-cut (see "Independent Critic" below).

## Re-cut onto main (2026-08-05)

The original branch forked from `9f2bf661` (#181) and carried 18 commits, of which 15 were dropped:
**9 were the WG0/G1 `plan_tasks` scan work that had already landed on main via #182**, and 6 were
this unit's own ledger/stamp commits. Merging it produced 7 conflicts that were almost entirely a
stale duplicate colliding with its own merged descendant — `partition_work.rs` was an add/add whose
whole diff was 11 lines of FK2.1 `Arc` sharing.

**Evidence that nothing unique was discarded.** The Actor's original check — that the set of `fn`
names on the branch is a subset of main's — does NOT establish this: it is blind to changed function
bodies, removed tests, consts, struct fields, match arms, and to files that did not conflict. The
claim was re-proven properly by the independent Critic, which computed the cumulative tree change of
the 9 dropped scan commits (`git diff 9f2bf661 bc4ffa19`, 13 files, ~2314 insertions) and diffed it
forward into #182: `bc4ffa19 → a966055e` removes **exactly 13 lines**, every one accounted for
in-hunk — `PartitionKey::new` became fallible (`.transpose()?`), two `.expect(..)` additions, two
`scan/map.md` rows rewritten with strictly more content, an import that gained `DataFileFormat`,
`to_arrow` split into `expand_within_file_parallel_tasks` (`scan/mod.rs:1006-1014`, strictly
expanded), and 5 `DisplayAs::fmt_as` lines deliberately superseded by #182, which moved
`snapshot_id` to Verbose-only with a rationale comment (`physical_plan/scan.rs:535-551`). All 72
functions added by the dropped commits exist on main; no removed line sits inside a test body.

The branch was therefore re-cut as the 3 genuine bump commits (`34192fcd`, `996a999e`, `d77bedae`)
cherry-picked onto `bde2e95d`. That reduced 7 conflicts to **1**: a clippy `useless_conversion`
auto-fix colliding with FK2.1's `Arc` types in an `arrow/reader.rs` test fixture. Resolved keeping
both (Arc types + the `.into_iter()` removal); no semantic payload either way.

### Residue new on the #182–#185 base (absent from the original branch)

| Item | Cause | Resolution |
|---|---|---|
| `set_max_row_group_size` deprecated (`arrow/reader.rs:3830`) | main-side code from #183/#184; parquet 58 deprecation | migrated to `set_max_row_group_row_count(Some(..))`, matching the 8 sites the bump already moved |
| `useless_conversion` in `storage/opendal/src/lib.rs:846` | nightly-2026-03-05 lint | `.zip(ready_meta)` |
| 4 sqllogictest schedules failing (10 plan blocks) | the `EnsureCooperative` rule was rewritten in DF54 to skip leaves that already sit under a cooperative ancestor — `CooperativeExec` is gone from EXPLAIN output | fixtures updated (node removed, subtree renumbered/dedented) |

**On the `CooperativeExec` removal — cosmetic, not a yield regression.**

*Correction (2026-08-05, independent Critic).* An earlier revision of this ledger attributed the
removal to `RepartitionExec` "now" declaring `SchedulingType::Cooperative` in DF54. **That is
false** — DF52 already declared it at `repartition/mod.rs:1261`, byte-identical to DF54's `:1560`.
The actual cause is the rule itself: DF52's `EnsureCooperative` (`ensure_coop.rs:70-86`) was a plain
`transform_up` that wrapped **every** non-cooperative leaf unconditionally; DF54 (`:70-119`) uses
`transform_down_up` with an ancestry stack and adds the `&& !is_under_cooperative_context` clause.

The safety conclusion is unchanged and rests on that clause directly: wrapping is the **fallback**,
so a leaf with no cooperative ancestor is still wrapped. `IcebergTableScan` declares no scheduling
type and stays `NonCooperative` (the default); in these plans it sits under `RepartitionExec`, so
budget is consumed at the repartition boundary instead. Both paths keep the yield guarantee, and no
fork-specific starvation hazard exists. Fixture mechanics verified exact: 10 insertions / 20
deletions = 10 blocks × (−2/+1), only the `CooperativeExec` line removed, indentation checked
against DF's EXPLAIN convention (`display.rs:451`, `indent*2`) and the sqllogictest normalizer
(`:104-115`); no stale block remains anywhere in `crates/`.

**The original branch never ran sqllogictest.** Its gate table omits it; these 4 schedules would
have failed there too. Recorded so the omission is not repeated.

### Re-cut gate (at tip)

| Gate | Result |
|---|---|
| `typos .` | green |
| `cargo fmt --all -- --check` | green |
| `cargo clippy --all-targets --all-features --workspace -- -D warnings` | green |
| `cargo test -p iceberg --lib` | **3137 passed** / 1 ignored (3135 = main @ #184, +2 F1 remediation pins) |
| `cargo test -p iceberg-datafusion --all-targets` | 177 / 73 / 12 / 4 / 2 passed, 0 failed (= main @ #184) |
| `cargo test -p iceberg-storage-opendal --lib` | **41 passed** |
| `cargo test --doc --workspace` | green |
| `cargo test --workspace --all-features --test sqllogictests` | **9/9 schedules** |
| `cargo check --workspace --no-default-features` | green |
| `make check-matrix-anchors` | OK (75 rows) |
| `make interop` | **52 passed, 0 failed** / 52 discovered (floor 52) — run twice, Actor + Critic independently |

**On what the matching counts do and do not prove.** An earlier revision offered "3135 and
177/73/12/4/2, identical to main @ #184" as behavior-invariance evidence. The independent Critic
correctly refuted that framing: the diff adds and removes **zero** test functions, so the counts
*could not* have changed — it is a tautology, not evidence. What the run does establish is that
every pre-existing assertion still holds under the new dependency family, which is meaningful but
weaker. Note also that net assertion strength went slightly **down**: `test_insert_into_nested`
re-pinned required→optional nested leaves, and 10 slt plan blocks lost a node.

The real format-stability evidence is (a) `make interop` 52/52 — run independently twice, by the
Actor and by the Critic — and (b) the Critic's direct diff of parquet writer defaults: every
encoding-relevant default is byte-identical between 57.3.1 and 58.4.0. The only change is
`DEFAULT_BLOOM_FILTER_NDV` (1,000,000 → 1,048,576), and this repo never enables bloom filters; every
in-repo path builds `WriterProperties::builder().build()`. **No on-disk encoding change.**

### Security posture (`cargo audit`)

Run both ways at the re-cut tip: **branch 6 advisories, main 13**. The bump is a net security
improvement — it clears anyhow, crossbeam-epoch, event-listener, quinn-proto, rkyv ×4 and spin. The
only addition is a third vulnerable `quick-xml` copy (0.39.4), still under the ≥0.41 fix threshold,
so the quick-xml exposure is unchanged in kind. Not a blocker; recorded because 1207/1430 lines of
`Cargo.lock` churn warrant it.

### Owed follow-up: R89 interop pin

parquet 58 broadens `LogicalType::Unknown` → `DataType::Null` to all physical types
(`parquet-58.4.0/src/arrow/schema/primitive.rs:118-121`). Row R89 (`unknown`) is ✅, so a scan of an
`unknown` column another engine wrote as BYTE_ARRAY / INT64 / FLBA now yields a different Arrow type
than before. `run-interop-unknown.sh` passes, which makes this theoretical rather than live — but no
in-tree pin covers the non-INT32 physical encodings. Named, not hidden.

### Independent Critic

One independent Opus Critic, fresh context, read-only. First pass: **NOT CONVERGED** — four blocking
items (F1 `.unwrap()`s, F2 stale precedence docs, F3 tokio widening, F4/F5 ledger defects), no
data-correctness defect found. It independently re-ran every gate rather than trusting this table,
independently confirmed `make interop` 52/52, retired the on-disk-format question with direct
evidence, and mutation-proved the Date32 pushdown guard non-vacuous. All four items are remediated
above and in the commits that follow.

Its one carried-forward observation, **pre-existing on main and not introduced here**: mutating
`max_row_group_row_count` to `None` leaves all tests green, so
`fk5_pos_oracle_sparse_pos_deletes_multi_rg` is vacuous w.r.t. row-group count. Filed for a later
unit; out of scope for a dependency bump.

### Remediation (2026-08-05, post-Critic)

| Finding | Fix |
|---|---|
| F1 | `date_literal_to_naive` helper returns `ErrorKind::DataInvalid` instead of three bare `.unwrap()`s at `transform/temporal.rs`; 2 tests added, mutation-proven RED by restoring the `unwrap` |
| F2 | `CLAUDE.md`, `Roadmap.md` (×5), `docs/testing.md`, GAP_MATRIX base line + row R88, `crates/iceberg/src/variant/map.md` |
| F3 | tokio `macros` + `rt-multi-thread` moved to `[dev-dependencies]` (only consumer is a `#[tokio::main]` doctest) |
| F4/F5 | this file |
| F6/F7 | recorded above |

**R88 adjudicated (the finding behind F2).** The stale text said variant shredded-parquet was
"blocked at the pinned parquet 57", which implied this bump would unblock it. It does not:
`LogicalType::Variant` and the arrow-side extension path are equivalent across parquet 57.3.1 and
58.4.0, gated behind the opt-in `variant_experimental` feature that pulls
`parquet-variant{,-json,-compute}` — absent from our lockfile. Precisely: `src/basic.rs` is
**byte-identical** between the two (`diff -q` silent), while `src/arrow/schema/extension.rs` differs
(187 → 197 lines) by a behavior-preserving refactor only — `try_extension_type` →
`has_valid_extension_type` and a `mut arrow_field` restructure. The blocker is a feature the
workspace does not enable and that upstream marks experimental, **not** a version floor. Enabling it
would be necessary but not sufficient: fork-side work is also owed (no Iceberg→`ShreddedSchemaBuilder`
bridge, no file-level interop leg, and `arrow/schema.rs`'s `variant()` visitor hard-errors by
design). R88 stays 🟡 with a corrected reason.

*(The "identically" in an earlier revision of this paragraph was itself an unverified supporting
claim — the third such in this unit, after the `fn`-name subset proof and the `RepartitionExec`
causal story. All three had correct conclusions and unchecked evidence. Recorded as the pattern to
watch, not just three separate corrections.)*

### Carried forward by the re-cut

BUG-001 (position-delete partition stamp, #184) and the FK1–FK5 perf campaign (#183) were **absent
from the original branch**, which is what RePark pins at `b009ac15`. Re-cutting inherits them.
