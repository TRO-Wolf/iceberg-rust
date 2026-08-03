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

# Rust performance optimization ideas

- **Date:** 2026-08-02
- **Scan roots:**
  - `/home/john/CodeRepos/BigRustSparkRebuild` (repark)
  - `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust` (TRO-Wolf fork; **note:** `../iceberg-rust` relative to repark does not exist — resolved to this checkout)
- **Agent count:** **10** (user override; skill default clamp is 2–8)
- **Focus:** full Rust surface (both trees), deep / granular multi-lens scout
- **Mode:** ideas only — **no implementations in this run**

## Executive summary

- **MERGE is the highest-leverage repark system path:** multi-phase `ctx.sql` re-parse/re-plan/re-scan, MoR full `collect`, path-string explosion (`String` per mutated row), sequential manifest walks, and giant `_file IN (...)` SQL all stack on the same critical path.
- **Iceberg MoR read/plan pays heavy ownership tax:** residual/`FileScanTask` deep clones, eq-delete keysets that still materialize `Datum` per cell, optional-but-unused page selection, delete→data planning barrier, and mutex-guarded immutable delete vectors.
- **“Zero-copy Arrow” is strong on engine→Python stream export; weak on ingest and several actions:** createDataFrame still does IPC + `to_vec` + decode; pandas/polars go through Python tuples; `to_arrow`/`collect` fully materialize.
- **TA / ML execute kernels leave clear compute wins:** multi-output TA recomputes full kernels 2–3×; TA densifies to `Vec<f64>` every partition; ML fit densifies per row and re-executes plans each IRLS/Lloyd pass.
- **I/O chat:** commit retries amplify Glue GetTable + metadata GET/PUT; no session metadata-pointer cache; dual S3 stacks (OpenDAL vs DF `object_store`); sequential manifest GETs on MERGE resolve.
- **Build/API surface:** OpenDAL default features always pull FS+memory+S3; iceberg core always links ORC; repark-session always links Postgres + dual AWS catalogs; `Column.sql` spins a fresh session+runtime per call.
- Many paths are already strong (Parquet pos-delete `RowSelection`, residual memo, COW stream-out, C-stream export producer). Wins below are **gaps**, not a rewrite mandate.

## Ranked ideas

### 1. Collapse MERGE into fewer plans / shared target+source materialization
- **Lens / agents:** DS algorithms repark (A7); I/O (A6); Alloc repark (A1)
- **Location:** `BigRustSparkRebuild/crates/repark-write/src/merge.rs` — multi `ctx.sql` / `run_sql` / `stream_sql` (~848–998, ~1094–1101, ~1161–1167, ~1776–1778); `TargetScanStream`
- **Evidence:** Each phase rebuilds SQL strings, re-plans, and re-executes target scan; multi-insert is one plan per clause; Stage A already folded cardinality but still multiplies I/O.
- **Idea:** Lower `MergeSpec` once to DF logical plans; materialize/register source once; share pin + planned tasks; sibling projections for discovery / rewrite / inserts.
- **Expected benefit:** throughput + latency (S3 GETs and optimizer work)
- **Confidence:** high
- **Validate with:** existing target-scan pass counters; wall-clock multi-file MERGE; span counts on `merge.target_scan` / `merge.join`
- **Risks:** first-match-wins, 3VL, cardinality, streaming bounds must stay bit-identical

### 2. MoR Stage B: stream matched work (stop full `collect` + path `HashSet<String>`)
- **Lens / agents:** Alloc repark (A1); Hot compute repark (A3); DS repark (A7)
- **Location:** `repark-write/src/merge.rs` — `matched_work_mor` (~1261–1366); pairs as `(String, i64)` in `position_delete.rs`
- **Evidence:** Full join result collected to `Vec<RecordBatch>`; every path `to_string()` into sets; COW already streams.
- **Idea:** Batch-stream SQL result; intern paths (`Arc<str>` / dictionary / path index); filter with masks not full `take` retention; pipeline into writers.
- **Expected benefit:** memory (primary) + CPU
- **Confidence:** high
- **Validate with:** large MoR UPDATE peak RSS vs COW; cardinality fail-loud pins
- **Risks:** first-match / dedup semantics; writer APIs wanting owned strings

### 3. Eq-delete apply without per-cell `Datum` clone storm (+ skip unused predicate trees)
- **Lens / agents:** Hot compute iceberg (A4); DS iceberg (A8); Alloc iceberg (A2); Zero-copy (A10)
- **Location:** `iceberg-rust/crates/iceberg/src/arrow/equality_delete_set.rs` (~174–251); `caching_delete_file_loader.rs` (~724–837); `delete_filter.rs` (~671–738)
- **Evidence:** “O(R)” path still decodes columns to `Vec<Option<Datum>>` and clones into `HashSet<Vec<…>>`; parse always builds Θ(E) survival predicate trees even when keyset is primary.
- **Idea:** Columnar / primitive key hashing; decode keys once per batch for multi-set probe; defer/lazy-build predicate tree for fallback only.
- **Expected benefit:** CPU + memory on MoR reads
- **Confidence:** high
- **Validate with:** 1M rows × 100k–1M eq-deletes; delete_filter equivalence harness; null/float gates
- **Risks:** Java nulls / float soundness is P0; harness must stay green

### 4. createDataFrame: use Arrow C-stream (kill IPC + `to_vec` + tuple explode)
- **Lens / agents:** Zero-copy (A10); Alloc repark (A1)
- **Location:** `python/repark/src/repark/session.py` (~2879–3032); `repark-python/src/session.rs` (~276–357, ~584–652)
- **Evidence:** CDF builds pyarrow Table → IPC stream → Rust `ipc_bytes.to_vec()` → decode MemTable; C-stream register already exists for mapInArrow; pandas/polars still go through Python tuples.
- **Idea:** CDF → `register_arrow_stream_as_temp_view`; frame inputs via `Table.from_pandas` / polars Arrow export with cast/null rules on Arrow types.
- **Expected benefit:** memory + latency on ingest
- **Confidence:** high (path exists); medium for pandas parity edge cases
- **Validate with:** large CDF peak RSS; full CDF oracle (null/NaT/decimal/refuse)
- **Risks:** null-witness / type refuse semantics; GIL/stream abort footguns

### 5. Share `FileScanTask` fields / residuals via `Arc` (stop plan-time clone fan-out)
- **Lens / agents:** Alloc iceberg (A2); API surface (A9); DS iceberg (A8)
- **Location:** `iceberg/src/scan/task.rs` (`split`/`sub_task`); `scan/context.rs` `into_file_scan_task` (~222–263); residual memo then deep clone (~280–290)
- **Evidence:** Residual memoized as `Arc` then deep-cloned into every task; splits reclone path, deletes, projection, partition; paths `to_string()` from already-Arc manifest entries.
- **Idea:** `Option<Arc<BoundPredicate>>`; shared parent task + offset window; `Arc<str>` paths; `Arc` project field ids / deletes.
- **Expected benefit:** memory + plan latency (file/split cardinality)
- **Confidence:** high
- **Validate with:** plan-only heaptrack on multi-RG many-delete tables
- **Risks:** public `FileScanTask` type/serde; engine consumers expecting owned trees

### 6. Concurrent manifest resolve + one path→DataFile index for MERGE/pos-deletes
- **Lens / agents:** I/O (A6); DS repark (A7); DS iceberg (A8)
- **Location:** `repark-write/src/merge.rs` `resolve_affected_data_files` (~1111–1147); `position_delete.rs` `data_file_partitions` (~193–221); fork `transaction/snapshot.rs` sequential walks
- **Evidence:** Plain serial `for` + `await` per manifest after Stage A; MoR and COW each walk; pos-delete groups re-walk.
- **Idea:** Bounded concurrent load (scan-like); build once path→(DataFile, partition) shared by resolve, stamp, rewrite allowlist.
- **Expected benefit:** latency (S3 RTT serialization)
- **Confidence:** high
- **Validate with:** N manifests wall time serial vs concurrent; ObjectCache hit on second consumer
- **Risks:** peak parse memory; schema-fallback cache keys

### 7. Session metadata-pointer cache + cut commit I/O amplification
- **Lens / agents:** I/O (A6)
- **Location:** Glue catalog `load_table` / `update_table` (fork `catalog/glue`); repark write always `load_table`; commit retry loop `transaction/mod.rs`
- **Evidence:** Every write/alter pays GetTable + metadata JSON; commit retry redoes full stack; double GetTable structural on update path.
- **Idea:** Session cache `(metadata_location → Arc<TableMetadata>)` with pointer check; skip re-parse when pointer matches; soft reuse on rebase when UUID data paths unique.
- **Expected benefit:** latency multi-statement + contended commits
- **Confidence:** high
- **Validate with:** two loads unchanged pointer → zero body GET; CAS injector op counts
- **Risks:** stale metadata across writers — fail closed on pointer mismatch

### 8. Multi-output TA: compute once per partition (stop 2–3× kernel recompute)
- **Lens / agents:** Hot compute repark (A3); Alloc repark (A1)
- **Location:** `repark-ta/src/udf.rs` — BBANDS/MACD/STOCH/AROON/MAMA arms (~350–423+)
- **Evidence:** Each sibling UDF runs full multi-output kernel and discards unused outputs.
- **Idea:** Partition-local multi-output cache or fused multi-column window UDF; share densified series once.
- **Expected benefit:** CPU
- **Confidence:** high
- **Validate with:** microbench three BBANDS cols vs one; bit-exact goldens
- **Risks:** cache key identity across partitions/params

### 9. TA bridge: stop `column_to_f64` densify + reuse buffers
- **Lens / agents:** Hot compute repark (A3); Alloc repark (A1)
- **Location:** `repark-ta/src/udf.rs` `evaluate_all` / `column_to_f64` (~630–669)
- **Evidence:** Cast + row map to owned `Vec<f64>` (NULL→NaN) per series per partition; multi-series multiplies.
- **Idea:** Null-free Float64 → `values()` slice; scratch reuse; kernel write into Arrow builders.
- **Expected benefit:** memory + CPU bandwidth
- **Confidence:** high
- **Validate with:** heaptrack on 1e6–1e7 partition windows
- **Risks:** NULL→NaN contract; golden bit-exactness

### 10. ML fit: batch `observe_dense` + avoid multi-pass plan re-exec
- **Lens / agents:** Hot compute repark (A3); Zero-copy (A10); Alloc repark (A1)
- **Location:** `repark-python/src/ml.rs` (~113–535); `repark-ml` accumulators
- **Evidence:** Per-row `list.value` + `Vec<f64>`; width discovery full stream; IRLS/Lloyd re-open DF plan each iter.
- **Idea:** Flatten FixedSizeList/dense batches once; materialize dense feature matrix when it fits (else stream); iterate math on buffers not re-scan.
- **Expected benefit:** throughput + allocator pressure
- **Confidence:** high
- **Validate with:** n=1e6 p=32 FixedSizeList timing + RSS
- **Risks:** memory blow-up on huge fits — keep streaming fallback

### 11. Process-wide Tokio + per-batch `block_on` on Python boundary
- **Lens / agents:** Concurrency (A5)
- **Location:** `repark-python/src/session.rs`, `dataframe.rs`, `ml.rs` — `OnceLock<Runtime>` + `block_on`
- **Evidence:** Shared multi-thread runtime; every action/`__arrow_c_stream__` batch pays enter/exit + GIL attach/detach cycles.
- **Idea:** Configurable worker threads; long-lived drain task + bounded channel for streams; avoid per-batch `block_on`.
- **Expected benefit:** latency under concurrent Python threads
- **Confidence:** high
- **Validate with:** N-thread collect/stream p99; tokio-console queue depth
- **Risks:** nested block_on / SAF-008; oversubscription vs DF partitions

### 12. Freeze pos-delete vectors without `Mutex` after publish
- **Lens / agents:** Concurrency (A5); Hot compute iceberg (A4 — DV merge clone)
- **Location:** `delete_filter.rs` `Arc<Mutex<DeleteVector>>`; merge via `vector.clone()` before OR
- **Evidence:** Comments say immutable post-install; apply still locks; merge clones full roaring maps.
- **Idea:** `Arc<DeleteVector>` freeze; OR-by-reference without clone.
- **Expected benefit:** CPU under parallel MoR
- **Confidence:** high (lock unnecessary); medium absolute win
- **Validate with:** multi-file MoR throughput; DV merge microbench
- **Risks:** any post-publish mutation becomes a data race — audit load path

### 13. MemoryCatalog: do not hold global mutex across FileIO
- **Lens / agents:** Concurrency (A5)
- **Location:** `iceberg/src/catalog/memory/catalog.rs` load/register/update
- **Evidence:** Single mutex held through metadata read/write awaits.
- **Idea:** Short critical sections + optimistic CAS pointer flip; I/O outside lock.
- **Expected benefit:** latency under parallel local DDL/DML
- **Confidence:** high for multi-session; medium for single-thread scripts
- **Validate with:** parallel load/update latency histograms
- **Risks:** half-create / atomicity invariants

### 14. Overlap delete-manifest and data-manifest planning
- **Lens / agents:** DS iceberg (A8)
- **Location:** `iceberg/src/scan/mod.rs` (~718–771); `delete_file_index` wait/`Notify` already exists
- **Evidence:** All delete entries fully processed before data entry processing starts, despite waiter support.
- **Idea:** Parallelize populate; data tasks park on `Notify` until index ready.
- **Expected benefit:** plan latency ≈ max(T_del, T_data)
- **Confidence:** high (barrier not correctness-required)
- **Validate with:** many delete+data manifests `plan_files` wall time; lost-wakeup tests
- **Risks:** thundering herd; Failed populate hang classes

### 15. Delete-index: composite `(spec_id, partition)` keys + seq-sorted lists
- **Lens / agents:** DS iceberg (A8)
- **Location:** `delete_file_index.rs` maps and `get_deletes_for_data_file`
- **Evidence:** Partition maps key on `Struct` only; `spec_id` post-filter linear; global lists full-scanned by seq predicate.
- **Idea:** Key by `(spec_id, Struct)`; sort lists once; `partition_point` for applicable tail.
- **Expected benefit:** plan attach CPU on MoR
- **Confidence:** high structural; medium magnitude
- **Validate with:** multi-spec large delete microbench; identical result sets
- **Risks:** wrong key → resurrection; seq `None` fixtures

### 16. `_pos` projection: restore pushdown / stream (don’t whole-file collect)
- **Lens / agents:** Hot compute iceberg (A4); Alloc iceberg (A2)
- **Location:** `arrow/reader.rs` (~514–534, finish whole-file ~873–925)
- **Evidence:** `_pos` disables RowFilter/RowSelection/RG prune and `try_collect`s entire file.
- **Idea:** Track absolute ordinals under selection; at least stream batches; Avro/ORC same streaming finish.
- **Expected benefit:** memory + IO on MoR identity scans
- **Confidence:** high impact; medium engineering difficulty
- **Validate with:** dense pos-deletes + residual; row set + `_pos` oracle
- **Risks:** wrong `_pos` corrupts written position deletes

### 17. Page-level residual selection + DF multi-partition Iceberg scan
- **Lens / agents:** Hot compute iceberg (A4)
- **Location:** page selection default off `reader.rs`/`scan/mod.rs`; DF scan `UnknownPartitioning(1)` in `integrations/datafusion`
- **Evidence:** RG metrics prune on; page selection opt-in and DF never enables; single DF partition, concurrency only inside scan.
- **Idea:** Adaptive page selection; map task groups to N DF partitions.
- **Expected benefit:** throughput selective scans / CPU post-decode
- **Confidence:** medium–high
- **Validate with:** selective multi-RG table; cores used vs wall time
- **Risks:** page-index overhead on tiny files; global limit/sort semantics

### 18. Prefix-negation / assignment maps / drop redundant path `IN` lists
- **Lens / agents:** DS algorithms repark (A7)
- **Location:** `merge.rs` `prior_clauses_do_not_apply`, `rewrite_column`, `rewrite_sql_from`
- **Evidence:** First-match encoded as O(C²) SQL text; assignment linear `.find` per col×clause; rewrite always embeds path literals even with file-scoped allowlist.
- **Idea:** Single clause_id CASE; assignment HashMaps; drop IN when allowlist already scopes tasks; else path MemTable semi-join.
- **Expected benefit:** plan time + SQL size
- **Confidence:** high (complexity); medium runtime depends on clause count
- **Validate with:** 20 clauses × 100 cols gen/plan time; survivor-row pins
- **Risks:** NULL first-match semantics

### 19. Single parse + single analyze fixpoint on `spark.sql` passthrough
- **Lens / agents:** DS repark (A7)
- **Location:** `repark-sql/src/lib.rs`, `spark_ast.rs` execute_passthrough; CTAS double analyze
- **Evidence:** Router parses Databricks dialect then passthrough re-parses session dialect; CTAS may analyze again.
- **Idea:** Thread parsed Statement; one analyze fixpoint shared with write schema.
- **Expected benefit:** TTFP / large-query plan latency
- **Confidence:** high for double parse; medium for CTAS analyze collapse
- **Validate with:** large SELECT parse+plan timing; UNION int-division pins
- **Risks:** dialect mismatch; set-op fixpoint

### 20. Constant-pattern datetime + substring without full `Vec<char>`
- **Lens / agents:** Hot compute repark (A3); Alloc repark (A1); DS repark (A7)
- **Location:** `repark-functions/src/datetime.rs`, `string.rs` SparkSubstring
- **Evidence:** Pattern re-tokenized / `format!` per row; substring collects all chars then slices.
- **Idea:** Compile scalar patterns once; char_indices / ASCII fast path; StringBuilder pre-size.
- **Expected benefit:** CPU on SQL ETL
- **Confidence:** high micro; medium e2e
- **Validate with:** 1e7 row microbench + Spark parity
- **Risks:** Spark char (not byte) edges; pattern error loudness

### 21. Write `conform_batch` cast elision + hoist name index
- **Lens / agents:** Alloc repark (A1); DS repark (A7)
- **Location:** `append.rs` conform_batch; merge cast helpers
- **Evidence:** Always cast+new RecordBatch; rebuilds CaseInsensitiveColumnIndex every batch.
- **Idea:** Identity fast path when types already match; cache permutation/cast plan per schema fingerprint.
- **Expected benefit:** CPU bandwidth on append/CTAS streams
- **Confidence:** medium
- **Validate with:** already-conforming stream append; negative cast overflow tests
- **Risks:** Utf8View/dict/timezone silent mismatch

### 22. Dual S3 stacks + OpenDAL feature defaults / always-on ORC
- **Lens / agents:** I/O (A6); API/build (A9)
- **Location:** OpenDAL FileIO vs `repark-session` `object_store_s3`; `iceberg-storage-opendal` default features; `iceberg` unconditional `orc-rust`
- **Evidence:** Two HTTP clients/pools; OpenDAL default still enables memory+fs when only s3 requested; ORC always linked for Parquet-first engines.
- **Idea:** Unify or share client; `default-features = false`; feature-gate ORC/expect-test.
- **Expected benefit:** cold start, wheel size, compile time, connection reuse
- **Confidence:** high for features; medium for dual-stack ROI
- **Validate with:** `cargo tree` / `cargo bloat` / wheel size; socket counts mixed workload
- **Risks:** consumer feature matrix; FileIO scheme parity

### 23. Feature-gate cold repark surfaces + gate Arrow prettyprint
- **Lens / agents:** API/build (A9)
- **Location:** `repark-session` always postgres+glue+s3tables+ta; `repark-python` arrow `prettyprint` for `show()` only
- **Evidence:** Full product always linked; prettyprint only for text show.
- **Idea:** `catalog-glue` / `postgres` / `ta` / slim maturin profile; optional `show-text`.
- **Expected benefit:** wheel size / link time
- **Confidence:** high for size; product may keep full default
- **Validate with:** full vs slim wheel MB
- **Risks:** API expectation “import repark gets everything”

### 24. `Column.sql` reuses planner context (no per-call SessionContext+runtime)
- **Lens / agents:** API/build (A9)
- **Location:** `repark-python/src/column.rs` (~227–252)
- **Evidence:** Fresh session, full function registration, current-thread runtime, `block_on` per expression.
- **Idea:** Process/thread-local expr planner; batch API.
- **Expected benefit:** latency for many Column.sql constructions
- **Confidence:** high cost; medium product heat
- **Validate with:** 1k× `Column.sql("1+1")`
- **Risks:** global state isolation

### 25. Honest streaming consumer APIs + schema cache on PyDataFrame
- **Lens / agents:** Zero-copy (A10)
- **Location:** `python/repark/.../dataframe.py` to_arrow/collect/toLocalIterator; `repark-python` `analyzed_arrow_schema_native` re-analyzes every call
- **Evidence:** Producer is O(batch); facade drains full table; schema re-analyze+clone on every metadata/export open.
- **Idea:** `to_arrow_batches` / true streaming iterator; cache SchemaRef per handle.
- **Expected benefit:** memory honesty + interactive metadata latency
- **Confidence:** high
- **Validate with:** multi-GB result RSS stream vs to_arrow vs collect
- **Risks:** Spark wording parity for iterator memory

### 26. Adaptive write fanout K / small-file control
- **Lens / agents:** I/O (A6)
- **Location:** `append.rs` / `merge.rs` concurrency; pos-delete per partition files
- **Evidence:** K independent writers can multi-file same partition; tiny objects amplify future plan cost.
- **Idea:** Adaptive K from estimated bytes; optional coalesce; metrics for compaction.
- **Expected benefit:** future scan/plan throughput
- **Confidence:** medium–high
- **Validate with:** small-batch append K=1 vs 4 file counts
- **Risks:** upload parallelism tradeoff

### 27. Enable page selection wisely; evaluate Parquet blooms
- **Lens / agents:** Hot compute iceberg (A4) — related to #17
- **Location:** residual page prune gate; no bloom usage under crates/
- **Evidence:** Selective filters still decode dead pages; blooms unused.
- **Idea:** Adaptive enable; writer bloom stats if parquet-rs supports.
- **Expected benefit:** IO bytes selective point lookups
- **Confidence:** medium
- **Validate with:** TPC-DS-like selective scan byte counters
- **Risks:** false prune must be zero

### 28. LINEARREG/CCI/MIDPRICE algorithmic TA costs (golden-gated)
- **Lens / agents:** Hot compute repark (A3)
- **Location:** `repark-ta` statistic/momentum/overlap kernels
- **Evidence:** O(n·period) re-sum/rescan; midprice lacks trailing-index used by min/max.
- **Idea:** Incremental sums / trailing extrema where goldens allow; SIMD reduce for fixed buffers.
- **Expected benefit:** CPU factor pipelines
- **Confidence:** medium (bit-exact hard)
- **Validate with:** TA goldens must not drift
- **Risks:** product prioritizes TA-Lib bit-identity

### 29. Postgres pool Drop `try_lock` drops live connections
- **Lens / agents:** Concurrency (A5)
- **Location:** `repark-postgres/src/pool.rs`
- **Evidence:** Contended Drop discards client; only permit returns → reconnect storms.
- **Idea:** Async return channel / await lock outside Drop critical path.
- **Expected benefit:** tail latency multi-partition JDBC
- **Confidence:** medium–high under concurrency
- **Validate with:** max concurrency checkout/return reconnect counts
- **Risks:** Drop must stay non-blocking

### 30. OpenDAL list incomplete entries: concurrent stat; Buffer→Bytes copies
- **Lens / agents:** I/O (A6); Alloc iceberg (A2)
- **Location:** `storage/opendal` list/read paths
- **Evidence:** Sequential HEAD for incomplete list rows; `to_bytes()` may consolidate copies.
- **Idea:** Concurrent incomplete stats; prefer contiguous zero-copy Bytes.
- **Expected benefit:** orphan/GC latency; IO bandwidth
- **Confidence:** medium–high
- **Validate with:** 10k-key list HEAD counts
- **Risks:** rate limits

---

## By lens (raw coverage)

| Agent | Lens | Areas scanned (high level) | Idea count (raw) | Gaps / notes |
|------:|------|----------------------------|------------------:|--------------|
| A1 | Alloc — repark | write/merge, ta udf, ml binder, functions, python session | 8 | Deprioritized catalog cold clones |
| A2 | Alloc — iceberg | scan tasks, residual, delete index, OpenDAL, loaders | 10 | Writers/metadata clone lower urgency |
| A3 | Hot compute — repark | TA kernels, ML, string/datetime, MERGE post-join | 10 | Analyzer setup noted as non-row |
| A4 | Hot compute — iceberg | eq-delete, pos-delete, pushdown, DF scan | 9 | RG prune / pos RowSelection already good |
| A5 | Concurrency / locks | Python runtime, MemoryCatalog, DV mutex, pools, REST OAuth | 8 | MERGE test mutex skipped |
| A6 | I/O catalog/S3 | MERGE manifests, commit, dual S3, metadata cache, fanout | 10 | — |
| A7 | DS/algos — repark | MERGE plans/SQL, sql parse, pos-delete, registration | 10 | Prune AST latent until residual re-enabled |
| A8 | DS/algos — iceberg | delete index, planning barrier, path deletes, residual cache | 10 | Schema id maps already hashed |
| A9 | API/build/logging | features, dyn IO, Column.sql, prettyprint | 8 | Error Display polish low ROI |
| A10 | Zero-copy FFI/Arrow | CDF, stream export, mapInArrow, ML, eq-delete, projector | 10 | Export producer already strong |

**All 10 agents completed successfully.**

## Dedup notes

Near-duplicate themes collapsed in ranking:
- Path-string ownership (A1/A3/A7/A9) → ideas **2**, **5**
- Eq-delete Datum/tree (A2/A4/A8/A10) → idea **3**
- MERGE multi-scan/sql (A6/A7) → ideas **1**, **6**, **18**
- TA densify + multi-output (A1/A3) → **8**, **9**
- ML densify/multi-pass (A1/A3/A10) → **10**
- createDataFrame IPC (A1/A10) → **4**
- Metadata load storms (A6) → **7**

## Out of scope / deferred

- **Python-only facade logic** scanned only where it defines the Rust FFI contract (CDF, collect, mapInArrow).
- **Measured speedups** — none claimed; all ideas are structural.
- **Java Iceberg interop changes** that would break packing order / summary ordering without deliberate decision.
- **Distributed execution / Ballista** — out of product v1 scope.
- Relative path `../iceberg-rust` from repark: **missing**; fork used at `openSource/apacheIcebergRust/iceberg-rust`.

## Suggested next measurements (optional)

1. **MERGE e2e flamegraph + S3 GET counters** (Stage A vs rewrite vs MoR collect RSS) — validates #1, #2, #6, #18.
2. **MoR read criterion:** eq-delete apply ns/row + allocs — validates #3.
3. **CDF / stream peak RSS** large table — validates #4, #25.
4. **plan_files heaptrack** many files + deletes — validates #5, #14, #15.
5. **TA multi-column microbench + goldens** — validates #8, #9.
6. **ML fit FixedSizeList** — validates #10.
7. **`cargo tree` / wheel bloat** OpenDAL features + orc — validates #22, #23.
8. **Multi-thread Python collect p99** — validates #11.

---

*Report generated by `/rust-performance-opt-ideas` (N=10). **Nothing was implemented.***
