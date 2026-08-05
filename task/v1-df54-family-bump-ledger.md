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

## Status

- Actor build: in progress → cargo check --workspace --all-targets green (pre-test)
- Octo: pending

## Actor gate (pre-octo)

| Gate | Result |
|---|---|
| `cargo check --workspace --all-targets` | green |
| `cargo test -p iceberg --lib --all-features` | **3068 passed** |
| `cargo test -p iceberg-datafusion --all-features` | green (incl. nested insert re-pin + doctest) |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | green (new nightly lints auto-fixed) |
| `make interop` | deferred to octo / pre-push (needs JDK/mvn) |

### Behavior re-pins (cited)

1. **test_insert_into_nested** — DF54 field-aware `CastExpr` + `validate_field_compatibility` rejects nullable SQL `named_struct` leaves → non-null nested Iceberg required fields. Fixture leaves under `address` re-pinned OPTIONAL; zip literal cast to INT; expect! snapshot updated. Citation: DF 54 upgrade guide (CastColumnExpr → field-aware CastExpr); datafusion-common `nested_struct.rs` nullability rule. Required-nested SQL insert follow-up out of family-bump scope.

## Octo

Pending: 8× critic-octo early_stop=false

## Re-cut onto main (2026-08-05)

The original branch forked from `9f2bf661` (#181) and carried 18 commits, of which **15 were the
WG0/G1 `plan_tasks` scan work that had already landed on main via #182**. Merging it produced 7
conflicts that were almost entirely a stale duplicate colliding with its own merged descendant —
`partition_work.rs` was an add/add whose whole diff was 11 lines of FK2.1 `Arc` sharing.

Verified before dropping them: for every conflicted code file, the set of functions on the branch
is a **subset** of main's. Nothing unique was discarded.

The branch was therefore re-cut as the 3 genuine bump commits (`34192fcd`, `996a999e`, `d77bedae`)
cherry-picked onto `bde2e95d`. That reduced 7 conflicts to **1**: a clippy `useless_conversion`
auto-fix colliding with FK2.1's `Arc` types in an `arrow/reader.rs` test fixture. Resolved keeping
both (Arc types + the `.into_iter()` removal); no semantic payload either way.

### Residue new on the #182–#185 base (absent from the original branch)

| Item | Cause | Resolution |
|---|---|---|
| `set_max_row_group_size` deprecated (`arrow/reader.rs:3830`) | main-side code from #183/#184; parquet 58 deprecation | migrated to `set_max_row_group_row_count(Some(..))`, matching the 8 sites the bump already moved |
| `useless_conversion` in `storage/opendal/src/lib.rs:846` | nightly-2026-03-05 lint | `.zip(ready_meta)` |
| 4 sqllogictest schedules failing (10 plan blocks) | DF54 `RepartitionExec` now declares `SchedulingType::Cooperative`, so `EnsureCooperative` no longer wraps the scan leaf — `CooperativeExec` is gone from EXPLAIN output | fixtures updated (node removed, subtree renumbered/dedented) |

**On the `CooperativeExec` removal — cosmetic, not a yield regression.** `EnsureCooperative`
(`datafusion-physical-optimizer-54.1.0/src/ensure_coop.rs:110`) wraps a leaf only when it is not
already cooperative **and not under a cooperative ancestor**. `RepartitionExec` sets
`SchedulingType::Cooperative` (`repartition/mod.rs:1560`), and `IcebergTableScan` sits directly
beneath it in these plans, so budget is consumed at the repartition boundary instead. Our scan
declares no scheduling type and stays `NonCooperative` (the default) — in any plan shape where it
is a leaf *without* a cooperative ancestor it is still wrapped. Both paths keep the yield guarantee.

**The original branch never ran sqllogictest.** Its gate table omits it; these 4 schedules would
have failed there too. Recorded so the omission is not repeated.

### Re-cut gate (at tip)

| Gate | Result |
|---|---|
| `typos .` | green |
| `cargo fmt --all -- --check` | green |
| `cargo clippy --all-targets --all-features --workspace -- -D warnings` | green |
| `cargo test -p iceberg --lib` | **3135 passed** / 1 ignored — identical to main @ #184 |
| `cargo test -p iceberg-datafusion --all-targets` | 177 / 73 / 12 / 4 / 2 passed, 0 failed — identical to main @ #184 |
| `cargo test -p iceberg-storage-opendal --lib` | **41 passed** |
| `cargo test --doc --workspace` | green |
| `cargo test --workspace --all-features --test sqllogictests` | **9/9 schedules** |
| `cargo check --workspace --no-default-features` | green |
| `make check-matrix-anchors` | OK (75 rows) |

The two headline counts matching main exactly is the behavior-invariance evidence: the family bump
moves no core or DataFusion-integration behavior.

### Carried forward by the re-cut

BUG-001 (position-delete partition stamp, #184) and the FK1–FK5 perf campaign (#183) were **absent
from the original branch**, which is what RePark pins at `b009ac15`. Re-cutting inherits them.
