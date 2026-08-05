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

# DF 52 → 54 family catch-up — churn map (V0 RECON)

**Unit:** V0 (docs-only)  
**Branch:** `docs/df54-churn-map`  
**Date:** 2026-08-01  
**Fork base (this branch):** `9f2bf661` (origin/main — perf waves A–E #181)  
**V1 stack base (G1 tip):** `bc4ffa19` (`feat/plan-tasks-multi-partition`, pushed-unmerged)  
**Target family (locked greylight):** `datafusion` + `datafusion-spark` **54.1.0**, `arrow*` / `parquet` **58.4**  
**Package identity:** fork stays **0.9.1** (family-only; 0.10 full sync is a different slate)  
**Scope audit:** greylight Q&A 2026-08-01 (no proposition ledger)

---

## 1. Sequencing (hard)

| Guard | State at recon |
|---|---|
| G1 `feat/plan-tasks-multi-partition` | **Pushed, not merged** — tip `bc4ffa19` (9 commits on main). V1 **stacks off G1** (do not wait). |
| V2 start | Only after V1's **pushed** rev exists. |
| Single pin write | V2 RePark `[patch]` rev is the only RePark pin commit. |
| Family floors | DF/spark **54.1.0** + arrow/parquet **58.4** — not 54.0.0, not arrow 59. |

---

## 2. Family version table (before → target)

| Crate / pin | Fork today (G1 tip) | RePark today (`cfd2416`) | Target |
|---|---|---|---|
| `datafusion` | 52.2 (lock → 52.x) | 52.2 (lock **52.5.0**) | **54.1.0** |
| `datafusion-spark` | n/a (fork) | 52.2 (lock **52.5.0**) | **54.1.0** |
| `datafusion-cli` / `datafusion-sqllogictest` | 52.2 | n/a | **54.1.0** (fork only) |
| `arrow*` | 57.1 | 57.1 (lock **57.3.1**) | **58.4** |
| `parquet` | 57.1 | (via iceberg/DF) | **58.4** |
| iceberg package version | 0.9.1 | 0.9.1 labels | **stay 0.9.1** |
| RePark `[patch]` rev | — | `4723104b` (**behind** fork main `9f2bf661`) | **V1 tip after push** |
| `pyo3` (fork python bindings) | (upstream path bumped 0.26→0.28 in DF53 commit; fork may lag) | RePark **0.29** (abi3-py312) | RePark: bump only if Arrow/PyO3 coupling forces it; not a free-ride |
| `serde_arrow` (fork `iceberg`) | `0.14` feature **`arrow-57`** | n/a | Must gain **`arrow-58`** feature (or equivalent) with the family bump |

---

## 3. Upstream leverage (apache/iceberg-rust)

Upstream `main` is already on **DF 54.1.0 / Arrow 58.4** (crate line 0.10.x). Harvest these commits — **do not hand-port what upstream already fixed**.

### 3.1 Cherry-pick candidates (ordered, DF path only)

| Order | Full SHA | Subject | What it does for us |
|---|---|---|---|
| 1 | `477a1e525b4915895388a4f45557b825ea541ef2` | deps: upgrade DataFusion to 53.0, Arrow to 58 (#2206) | **Primary port.** Arrow 57→58, DF 52→53; physical-plan executor adapt; `set_max_row_group_size` → `set_max_row_group_row_count(Some(...))`; SLT expectation updates; (skip or surgically adapt `bindings/python` — we own a different python story). |
| 2 | `1b6400956da02d6b564db527edf5d7ea8feca0d3` | chore: bump datafusion to 53.1.0 (#2350) | Pin-only (lock + floors). |
| 3 | `7e84157818bb09e9d6bdfc7d3bdc4061a6d078b6` | Bump datafusion-cli 53.0→53.1 (#2373) | Pin-only (fork tooling). |
| 4 | `d7c647ff792649460dc3ac38193bcbe6882990da` | Bump datafusion-sqllogictest 53.0→53.1 (#2369) | Pin-only. |
| 5 | `875fdb7463f0a2d685fa3c61c70527469bcd4407` | chore(deps): Bump DataFusion to 54.0.0 (#2648) | **Primary port.** Remove `as_any` from Catalog/Schema/Table/ExecutionPlan/PhysicalExpr impls; `Expr::Cast` → `c.field.data_type()`; downcast via trait `downcast_ref` not `.as_any()`. |
| 6 | `e8460eee8725c7f66d53e26b89b3fef578f90ce5` | deps: bump to DataFusion 54.1.0 (#2872) | Pin-only to **54.1.0** (our locked floor). |

**Not in V1 charter (0.10 / feature surface):** metadata columns (`_partition`, `_pos`, `_spec_id`), history metadata table, encryption AES-256, etc. Those ride the future 0.10 sync / upstreaming workstream.

**Python binding commits** (`f13bd41b` py DF 54, pyo3 capsule churn in #2206): **skip by default** unless the fork's `bindings/python` is in the V1 gate path and fails without them. Prefer minimal hand-port if needed; RePark does **not** consume fork py bindings.

### 3.2 Cherry-pick applicability vs fork divergence

| Area | Upstream applies cleanly? | Fork-divergent remainder |
|---|---|---|
| Workspace `Cargo.toml` floors | Yes (edit numbers) | Keep **0.9.1** package versions (upstream moved to 0.10.0). |
| `iceberg-datafusion` `as_any` removal | Yes (same trait surface) | Fork has **extra** exec nodes (`delete.rs`, G1 scan knobs, `ScanKnobs`, multi-partition `plan_tasks` path) — strip `as_any` there too. |
| `expr_to_predicate` Cast field | Yes | Same file on fork still uses `c.data_type` at G1 tip — apply #2648 pattern. |
| `set_max_row_group_*` | Yes pattern | Fork still has many `set_max_row_group_size` call sites (reader tests, parquet_writer, convert_eq_delete tests, interop_scan_plan). |
| `serde_arrow` feature | Manual | Upstream will have moved feature flag; fork still `arrow-57`. |
| G1 `plan_tasks` / `ScanKnobs` / fail-closed demote | **No upstream equivalent** | Hand-port only DF trait compile breaks (`as_any`, stats API if touched, partition props). **Do not** regress G1 pin battery. |
| Perf-wave scan prune / `with_file_prune_only` | Fork-only lineage under G1 | Same — compile-fix only. |

**Strategy recommendation for V1 Actor:**  
1) Bump workspace floors + lock to 54.1/58.4 in one commit.  
2) Apply mechanical API fixes by **replaying upstream #2206 + #2648 hunks** onto divergent files (prefer three-way cherry-pick with conflict resolution over blind copy).  
3) Fix remaining compile errors on G1-only surfaces.  
4) Behavior-drift tests: re-pin only with changelog citation in the commit message.

---

## 4. Breaking-change matrix (engine → our code)

Sources: DF 53.0.0 blog + changelog; DF 54.0.0 upgrade guide + changelog; Arrow 58 parquet writer rename (via upstream #2206).

### 4.1 High-likelihood compile breaks (must fix in V1 and/or V2)

| Change | Where it hits | Repo |
|---|---|---|
| Remove `as_any` from `ExecutionPlan`, `TableProvider`, `SchemaProvider`, `CatalogProvider`, `PhysicalExpr`, `ScalarUDFImpl`, `AggregateUDFImpl`, `WindowUDFImpl` | All impl sites; downcasts become `trait.downcast_ref::<T>()` | **Both** |
| `Expr::Cast` carries `Field` (`c.field.data_type()` not `c.data_type`) | `expr_to_predicate.rs` (fork); any RePark expr walkers | **Both** (fork certain) |
| `partition_statistics` → `Result<Arc<Statistics>>` | Custom `ExecutionPlan`s | Fork if impl'd; RePark **postgres** still has old `fn statistics` — DF53 removed `statistics()` API (#20319) → **must** migrate |
| `PlanProperties` / Arc-wrapped immutable plan parts (DF53) | Exec constructors, `with_new_children` | Both if custom plans |
| Parquet `WriterProperties::set_max_row_group_size` → `set_max_row_group_row_count(Option<usize>)` | iceberg arrow reader tests, writers | **Fork** |
| `serde_arrow` feature `arrow-57` → `arrow-58` | `crates/iceberg/Cargo.toml` | **Fork** |
| Numeric-preferring comparison coercion (DF54) | Possible SLT / golden result drift | Both tests |
| Physical EXPLAIN aggregate display changes | Any EXPLAIN string pins | Both tests |
| `ScalarUDFImpl` / registry higher-order methods (DF54) | Custom `FunctionRegistry` / `Session` / `TaskContext::new` if used | RePark session path — check constructors |
| `MemoryPool: Any + 'static` | Custom memory pools | RePark session memory pool if non-static |

### 4.2 Behavior / scoreboard risk (re-pin only with citation)

| Change | Risk to us | Notes |
|---|---|---|
| Join filter pushdown + dynamic filters (DF53) | Plan shape / TPC-H wall times | **Win candidate** for Q21 class |
| Sort-merge join speedups + spilling NLJ (DF54) | Memory / hang class | **Win candidate** for Q72 multi-join hang |
| Morsel-driven Parquet scans (DF54) | Scan walls | Scoreboard walls may drop without code change |
| Scalar subquery multi-row now errors | Rare SQL surface | Loud error vs silent multi-row — document if tests hit |
| Filter predicate reorder | Fallible cast-after-regex patterns | Unlikely in our banked suite |
| `arrays_zip` field names `c0`→`1` | If any Spark path uses arrays_zip field names | RePark / spark functions census |
| Unnest / LATERAL (DF54) | F3 explode path may simplify later | Retirement only if oracle green (greylight §9) |

### 4.3 Explicitly out of scope for this slate

- Full apache 0.10 feature sync  
- Capability matrix flips from “version bump alone”  
- Engine productization of Q58/Q59 decimal seeds (unless free win)  
- AWS live MERGE profile re-run (orchestrator post-merge)

---

## 5. Fork module churn map

### 5.1 `iceberg-datafusion` (integrations/datafusion)

| Module | DF/Arrow touch | V1 action |
|---|---|---|
| `table/mod.rs` | `TableProvider`, G1 `plan_tasks` multi-partition, `ScanKnobs` | Remove `as_any`; keep G1 fail-closed battery green |
| `table/metadata_table.rs` | `TableProvider` | Remove `as_any` |
| `table/table_provider_factory.rs` | factory APIs | Compile-fix |
| `catalog.rs` / `schema.rs` | Catalog/Schema providers | Remove `as_any` |
| `physical_plan/scan.rs` | `IcebergTableScan` + knobs | Remove `as_any` / `as_any_mut` if trait dropped; preserve metrics |
| `physical_plan/delete.rs` | DELETE exec (fork-heavy) | Remove `as_any`; hand-port |
| `physical_plan/write.rs`, `commit.rs` | write/commit exec | Remove `as_any` (upstream #2648 pattern) |
| `physical_plan/metadata_scan.rs`, `project.rs`, `repartition.rs`, `sort.rs` | PhysicalExpr / plans | `as_any` strip; `downcast_ref` in tests |
| `physical_plan/expr_to_predicate.rs` | `Expr::Cast` | `field.data_type()` |
| `physical_plan/mod.rs`, `error.rs`, `lib.rs`, `task_writer.rs` | glue | Compile-fix |
| Tests (`integration_datafusion_test`, interop DML, partitioned insert) | SLT + integration | Re-pin with citation if behavior drift |

### 5.2 `iceberg` core (arrow / parquet / scan)

| Module | Touch | V1 action |
|---|---|---|
| `arrow/reader.rs` | parquet writer props in tests | `set_max_row_group_row_count` |
| `writer/file_writer/parquet_writer.rs` | writer props | same rename |
| `maintenance/*_tests.rs`, `tests/interop_scan_plan.rs` | row-group sizing | same rename |
| `scan/*` (G1 `plan_tasks`, batch, context) | mostly iceberg-native | Only if Arrow types in signatures break |
| `transform/*`, `inspect/*` | Arrow arrays | Compile against arrow 58 |
| `serde_arrow` dep feature | `arrow-57` | → `arrow-58` |

### 5.3 Fork gate (V1 CONVERGED)

From fork `Makefile` / charter:

- `make check` (fmt, clippy `-D warnings`, toml, machete, agent-artifacts, matrix-anchors)  
- `make test` / nextest workspace  
- **`make interop` locally** (weekly CI won't catch in-window)  
- Matrix: **version notes only** — no capability cell flips claimed from the bump  
- Behavior re-pins need **changelog citation in commit message**

---

## 6. RePark module churn map (V2)

### 6.1 Pin commit (ONE commit)

| File | Change |
|---|---|
| `Cargo.toml` `[workspace.dependencies]` | `datafusion`/`datafusion-spark` **54.1.0**, `arrow*` **58.4** |
| `Cargo.toml` `[patch.crates-io]` | **all** `iceberg*` → V1 pushed rev (single writer) |
| `Cargo.lock` | resolve family together |
| `uv.lock` | only if py deps move |
| `AGENTS.md` / pin-contract comments | update verified family table |
| `deny.toml` | if quick-xml ≥0.41 unlocks, **remove** RUSTSEC-2026-0194/0195 ignores; else record survival |

### 6.2 Fix-forward surfaces

| Crate / path | Likely breaks | Notes |
|---|---|---|
| `repark-functions` | `as_any` on `ScalarUDFImpl`/`AggregateUDFImpl` (datetime, string `SparkConcat`, collection, aggregate `SparkAvgWithRetract`, lib stubs) | Mechanical strip; keep name-overwrite order after datafusion-spark |
| `repark-ta` | `as_any` on window UDFs | Mechanical |
| `repark-postgres` | `TableProvider`/`ExecutionPlan` `as_any`; **`fn statistics` → partition_statistics Arc** | Highest DF-API risk after functions |
| `repark-sql` | MERGE / time_travel / metadata / call / spark_ast; test UDFs with `as_any` | Compile + EXPLAIN pins |
| `repark-write` | MERGE exec / scan prune / writer props | Arrow/parquet writer API |
| `repark-session` | SessionContext, memory pool, spark register | TaskContext / higher-order if constructing manually |
| `repark-python` | Arrow FFI / RecordBatch; downcasts using `.as_any()` on arrays are **Arrow** `Array::as_any` — **keep** those | Only DF trait `as_any` removals change |
| `repark-catalog` | downcasts | Review `.as_any().downcast_ref` on DF traits → trait downcast |

### 6.3 Shim-retirement census (V2, gated)

Retirement requires **(a)** datafusion-spark 54 verifiably covers the case **and** **(b)** existing oracle battery passes **unchanged**. Same-named upstream ≠ coverage.

| Shim / workaround | Why it exists | Retirement candidate? |
|---|---|---|
| `SparkConcat` Utf8 force (`repark-functions/string.rs`) | datafusion-spark 52 plan Utf8 / kernel Utf8View (Q5/80/84) | **Only if** 54 SparkConcat emits Utf8 (or view consistently) **and** concat oracle + TPC-DS Q5/80/84 stay green |
| Date shim (`spark_date_shim_functions`) | Spark date semantics gaps | Inventory vs datafusion-spark 53/54 additions (`date_diff`, timestamp casts, etc.); retire only per-function with oracle |
| `SparkAvgWithRetract` | datafusion-spark SparkAvg lacks `retract_batch` | Check 54 SparkAvg for retract; keep if still missing |
| F1 select global-agg sticky / SQL path | Facade plan routing, not DF missing fn | **Not** a DF-spark shim — leave unless DF plan behavior makes sticky unnecessary (unlikely) |
| F3 explode → unnest rewrite | DF unnest null/empty/ordinality gaps | DF54 LATERAL + unnest improvements are **retirement candidates** only if explode oracle battery green without rewrite |
| Function census LOUD unsupported | Engine-missing | May flip to supported if spark 54 adds them — census-driven |

---

## 7. Win census (changelog → banked seeds)

Baseline **(A)** banked as-of r13 (same machine, days old). Anomalous after-rows → spot re-measure that query on pre-repin main.

| Claim | Banked status | Upstream source | How V2 verifies |
|---|---|---|---|
| **TPC-H SF10 Q21** join/memory TIMEOUT | TIMEOUT @ 300s, ~8.4 GiB RSS (tpch-report 2026-07-31) | DF53 join filter pushdown; DF54 SMJ + repartition + spilling NLJ | Scoreboard SF1 both legs always; SF10 Q21 optional/deferred if window dies — at minimum SF1 walls + call out Q21 class |
| **TPC-DS SF1 Q72** multi-join hang DIED | SEEDED (D2); planner/exec hang | Same join/spill/pushdown class | Full SF1 deferred OK for tip; smoke + spot Q72; full table is SQM hard gate |
| **TPC-DS Q5/80/84** concat Schema | FIXED D2 via SparkConcat | DF53/54 concat perf + Utf8View paths | Must stay OK after repin; retirement of shim only if oracle green without our UDF |
| **TPC-DS Q58/Q59** decimal vs DuckDB float | SEEDED | Unlikely fixed by DF bump (oracle type class) | Expect **unchanged**; if “win”, spot re-measure attribution |
| **F1** select global-agg | Shipped | N/A (facade) | Facade suite green; no capability claim |
| **F3** explode unnest | Shipped partial (posexplode STOP) | DF54 LATERAL / unnest | Retirement only with oracle evidence |
| **datafusion-spark shim retirements** | date shim, avg retract, concat | DF53 ~20 spark fns; DF54 more (round/floor/ceil/array_contains/…) | Per-function: delete ours only if (a)+(b) |
| **TPC-H SF1** 22/22 | Green parquet + Iceberg | Scan/join speedups | Re-run SF1 **both legs**; expect OK or cited re-pin |
| **Fuzzer** | D3 5000/5000 empty corpus | Random SQL | Smoke + optional 1000q seeded; corpus growth is not V2 mandate |

### 7.1 Scoreboard deliverable shape (V2 ledger)

| Suite | Before (banked) | After (V2) | Notes |
|---|---|---|---|
| TPC-H SF1 parquet | 22/22 OK | … | required for tip if time |
| TPC-H SF1 Iceberg | 22/22 OK | … | required for tip if time |
| TPC-DS SF1 parquet | post-D2: ~96 OK class (3 fixed, 3 seeded) | … | full SF1 **may be deferred** with loud ledger |
| TPC-DS Iceberg | out of scope | — | confirmed greylight |
| Fuzzer smoke | seed 42 harness green | … | required |
| Fuzzer 1000q | — | … | may defer |
| Facade + wheel-smoke | green on main | … | required gate |

---

## 8. V1 Actor checklist (fork `chore/df54-family-bump` @ G1)

1. Stack base: `bc4ffa19` (already branched in worktree).  
2. One family bump commit: floors 54.1.0 / 58.4 + lock.  
3. Port upstream #2206 + #2648 mechanical hunks; hand-fix G1/delete/write remainder.  
4. `serde_arrow` feature `arrow-58`.  
5. Full gate + `make interop`.  
6. Critic-octo 8× `early_stop=false`.  
7. Push on `OCTO-CONVERGED`; open `[marci]` PR stacked on G1; merge stays user.  
8. PR body: family table, cherry-picked vs hand-ported list, re-pin citations, RoadMapSync draft for RePark.

---

## 9. V2 Actor checklist (RePark `chore/grok-df54-repin`)

1. Wait for V1 **pushed** rev.  
2. ONE pin commit (`[patch]` + DF/arrow floors + locks).  
3. Fix-forward churn surfaces; shim retirement only under greylight §9.  
4. Scoreboard: smoke-level evidence at tip OK; full SF1 TPC-DS + 1000q fuzz marked deferred if needed.  
5. Baseline (A) + anomalous spot re-measure.  
6. Audit: quick-xml ignore sunset check.  
7. Critic-octo 8× `early_stop=false`.  
8. **Local tip only — no push** (operator verifies pin before origin).  
9. Import slate brief to `briefs/` + greylight appendix on first V2 commit; ledger points at this churn map (no file mirror).

---

## 10. RoadMapSync draft (for V1 PR → RePark)

> Fork tip after V1 will be DF **54.1.0** / Arrow **58.4** while remaining package identity **0.9.1**.  
> Breaking surface for consumers: remove `as_any` on all iceberg-datafusion providers/plans; `Expr::Cast` field-aware; parquet writer row-group API rename; possible physical plan partition count behavior from G1 `plan_tasks` (pre-existing on stack base).  
> RePark must repin `[patch]` to V1 rev and bump the DF family in the **same** pin commit — never half-bump.

---

## 11. References

- Greylight plan: `~/.claude/plans/2026-08-01-grok-df54-catchup-slate.md`  
- DF 53 blog: https://datafusion.apache.org/blog/2026/04/02/datafusion-53.0.0/  
- DF 54 blog: https://datafusion.apache.org/blog/output/2026/06/12/datafusion-54.0.0/  
- DF 54 upgrade guide: https://datafusion.apache.org/library-user-guide/upgrading/54.0.0.html  
- DF 53 changelog: https://github.com/apache/datafusion/blob/branch-53/dev/changelog/53.0.0.md  
- DF 54 changelog: https://github.com/apache/datafusion/blob/branch-54/dev/changelog/54.0.0.md  
- Upstream bump SHAs: §3.1  
- Banked scoreboards: RePark `task/tpch-report-2026-07-31.md`, `task/tpcds-report-2026-07-31.md`, `task/d2-tpcds-fixes-ledger.md`
