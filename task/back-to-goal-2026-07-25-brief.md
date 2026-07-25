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

# Back-to-goal block — signed off 2026-07-25

The standing spec for the queue that takes the fork from "consumer findings filed" back to the
RePark end-goal: a trustworthy table-format core under a DataFusion-wrapped query engine.
Supersedes the 2026-07-25 10-agent triage scratchpad document (this file is now the single home);
tracker rides `task/todo.md`. Companion work order for the consumer workstream:
`~/Desktop/repark-work-order-2026-07-25.md`.

## Decisions (user, 2026-07-25)

| # | Decision |
|---|---|
| D1 | WG1 fix shape = **children-bearing `PhysicalExpr`** (not a custom ExecutionPlan node). |
| D2 | WG3 = **harden non-breakingly first** (L2+L3 now); the L1 `PartitionKey::new -> Result` break lands as a follow-up unit. |
| D3 | WG2 = detection recipe **plus remediation tooling**. |
| D4 | Cadence = **WG1 alone as a Mode A PR**; everything else bundled Mode B. |
| D5 | Merge the pending bundle first — **satisfied**: merged as #170 (`37d035d6`) + #171 (`ce2affc9`), content-verified per protocol R4. |
| D6 | Partition-path URL-escaping divergence: **explicit approval granted** (format-visible change; new GAP_MATRIX row R161). |

## State of main at signing (`ce2affc9`, verified 2026-07-25)

- The post-D1 bundle is merged; all WG1 target files are **byte-stable** from the triage rev
  (`a4d3b92e` → `c88888c3` diff empty for `arrow/` + `datafusion/`; `c88888c3` → `ce2affc9` touches
  only `scan.rs`/`write.rs`/`avro_reader.rs`/`orc_reader.rs`). `PartitionExpr::children()` still
  returns `vec![]` at `project.rs:149-151` — every triage citation remains valid at tip.
- The `is_nan`/`not_nan` Critical from the 2026-07-17 audit is **already remediated** (overnight
  units A1/A2, merged #158-#160) — no unit needed.
- R159 (OAuth token refresh) / R160 (vended credentials) are **🟡 with named residues** — backlog,
  not RePark-blocking for Glue/S3 Tables.
- `PosDelState` still has **no `Failed` variant** (`delete_filter.rs:58`; only `EqDelState` has one)
  — the lost-wakeup unit stands as scoped.
- Nightly-interop live proof of #169/#170 (scan-plan split merge + pin) = **next dispatch on main**
  (watch item; needs `gh`/web — not runnable from this environment).

## Goal frame — three gates

1. **Data-trust** (nothing silently corrupt in either direction): Unit 1 + bundle groups G1/G2.
2. **Engine-trust** (no panic, hang, or wrong pushdown in a long-running engine): bundle G3–G8 +
   Unit 3.
3. **Scale**: Units 4–5 (H7 remainder).

---

## Unit 1 — WG1: honest-children `PartitionExpr` (S1, Mode A solo PR)

**Branch** `fix/partition-expr-honest-children` off `ce2affc9`, in worktree `iceberg-rust-ws`.
**Ladder:** AC·OO (Opus Actor / independent Opus Critic) minimum, per CLAUDE.md. **Claim opened**
on the PrimarySync board 2026-07-25.

**Defect (one mechanism, two filed symptoms — FORK-O7 + FORK-O8):**
`PartitionExpr` (`crates/integrations/datafusion/src/physical_plan/project.rs:103-190`) declares
`children() = vec![]` while reading the input batch positionally
(`evaluate()` → `calculator.calculate(batch)` → `projector.project_column(batch.columns())`,
`partition_value_calculator.rs:135-137`). DataFusion's `ProjectionPushdown` (runs twice, after the
provider hook) fuses adjacent projections; `update_expr` walks `children()`, finds none, and copies
the expr verbatim onto a **different child** — the partition value is then computed from the wrong
batch. Consequences: any computed SELECT item (`CASE`, `CAST`, `coalesce`, arithmetic, UDF) writes
a real-but-wrong manifest partition tuple (silent row loss on pruned reads, Rust AND Java/Spark;
spec-invalid when the true value is NULL); same-typed column permutation picks the wrong column
entirely; the FROM-less literal re-parents onto `PlaceholderRowExec` (1-row/0-col batch) and panics
at `record_batch_projector.rs:183`. The core crate is **not** at fault (splitter, transforms,
null-union all correct); the consumer's proposed bounds-check fix and the instinctive
schema-equality guard are both **false fixes** — name them rejected in the PR body.

**Fix (D1 shape):**
1. Carry `children: Vec<Arc<dyn PhysicalExpr>>` — one `Column::new(name, i)` per field of the
   validated input schema, built in `project_with_partition`.
2. `children()` returns them; `with_new_children()` rebuilds (typed error on arity mismatch — not
   the current silent `Ok(self)`).
3. `evaluate()` evaluates the children and feeds those arrays to the calculator — never
   `batch.columns()`; normalise via `into_array(batch.num_rows())` (a substituted child can be a
   `Scalar`).
4. Additive core helper `PartitionValueCalculator::calculate_from_columns(&[ArrayRef], num_rows)`;
   `calculate()` becomes a wrapper.
5. Include `children` in `PartialEq`/`Hash` (`project.rs:119-126`, `:183-189` are pointer-equality
   today) — omitting this makes structurally different exprs compare equal.

Correct whether or not DF fuses: with children present, DF's anti-fusion guard
(`count > 1 && !is_expr_trivial`) refuses to unify exactly when the child expr is non-trivial;
trivial children unify and `update_expr` re-indexes correctly.

**Hardening shipped alongside — labelled NOT the fix:**
- `record_batch_projector.rs:183`: bounds-checked index + drop the bare `.unwrap()`. Note for
  review: SAF-005 (#168) cleared the *unwrap*; the unchecked **slice index** is a distinct hazard
  that audit did not consider.
- `PartitionValueCalculator::calculate`: validate incoming batch width/schema against the schema the
  projector's indices were derived from — the only guard catching right-count/wrong-column, and it
  protects the **core-crate seam RePark consumes**.

**Tests (RED-first is the acceptance gate):** T1 hermetic two-projection plan through
`ProjectionPushdown::optimize` then **execute**, asserting `_partition` values (never plan shape);
T2 e2e `INSERT … SELECT CASE WHEN … THEN NULL` asserting the **manifest partition tuple**; T3 same
with a non-NULL divergent value (`THEN 'zzz'`) — kills the "just add a null check" false fix,
non-optional; T4 column permutation; T5/T6 plain-column + `VALUES` controls (verified green before
the fix too); T7 children/arity/eq-hash units; T10 FROM-less literal partitioned INSERT (panics
unfixed — cannot go falsely green). Optional interop leg:
`crates/integrations/datafusion/tests/interop_partitioned_dml.rs` + a
`dev/java-interop/run-interop-partitioned-dml.sh` (Java reads back the null partition tuple).
**Mutations:** M1 `children()` → `vec![]`; M2 `evaluate()` → `batch.columns()`; M3
`with_new_children` → `Ok(self)` (unit-level; record honestly if e2e stays green); M4 revert
eq/hash (T7 only — known residual); M5 sabotage T3's expectation.

**Coverage hole that let this ship (fix in this unit):** no `INSERT … SELECT <expr>` into a
partitioned table exists anywhere in the workspace — every partitioned insert test uses `VALUES`.

**Size:** ~120-180 LoC production, ~250-400 test. **API:** additive only (one new public core
method; `PartitionExpr` is private). **Close-out:** announce in `RoadMapSync.md`; RePark repins and
flips its two divergence pins (`overwrite_null_partition_source_lands_in_wrong_slot_fork_gap`,
`overwrite_fromless_literal_source_panics_on_partitioned_table_fork_gap`) — that flip is the
cross-repo done-signal.

---

## Unit 2 — engine-trust bundle (Mode B, one branch, sequential groups, final bundle Critic)

**Branch** `fix/engine-trust-bundle-2026-07` off main after Unit 1 merges. Per-group AC ladders
(user-directed 2026-07-25: **Opus-max Actor + independent Opus-max Critic per group**), sequential
in the ws worktree, ONE final independent Fable-max bundle Critic over the whole diff; push on
CONVERGED; single PR. A group whose ladder cannot converge in 2 remediation cycles is reset to the
last good commit and the bundle ships without it.

**G0 — the Unit 1 T10 residue: provider INSERT nullability widening (added 2026-07-25 post-#172).**
`project_with_partition`'s input-schema validation requires exact Arrow field equality INCLUDING
nullability, so an input plan carrying non-nullable fields (FROM-less literals, non-null `VALUES`,
`SELECT` from required columns) into a table whose target column is OPTIONAL fails with
Plan("Input schema does not match Iceberg table schema … nullable Utf8 vs Utf8") before the
partition machinery runs (first T10 run, 2026-07-25, recorded as Unit 1 residue). **Fix:** relax
exactly the SAFE direction — input field non-nullable where the table field is nullable — applied
RECURSIVELY (nested struct fields, list elements, map values); everything else stays strict:
nullable input into a required target keeps failing loudly, and names/types/order are unchanged.
Anchor: required-into-optional is the standard write-compatible direction (Spark accepts it; Java
Iceberg write compatibility treats it as legal; verify the DataFusion-side behavior in-tree rather
than by citation). **RED-first tests:** the original T10 optional-column shape (FROM-less literal
into an optional-column partitioned table — must succeed post-fix with correct manifest tuples and
NULL legality intact); non-null `VALUES` into optional; `SELECT` required-source into optional;
NEGATIVE pins: nullable input → required target still rejected with the existing loud error;
record the unpartitioned-path behavior for symmetry. **Mutations:** revert the widening (strict
equality restored) → the new positives RED; over-widen (nullable→required accepted) → the negative
pin RED. **Riders:** ADV-1 — deduplicate the double `Column` list construction in
`project_with_partition` (same file, style-only, from the Unit 1 Critic); ADV-2 — add the second
interop sabotage leg truncating `rust_table_nulltuple`'s metadata in
`dev/java-interop/run-interop-partitioned-dml.sh`, HARD-FAIL-never-SKIP pattern.

**G1 — WG2: exposure detector + remediation tooling (D3).** Detector (offline, mechanical): for
each live data file, re-read partition-source columns, recompute the transform, compare to the
manifest tuple (identity/bucket/truncate/temporal all recomputable). Remediation: rewrite affected
files under correct partition keys (lean on the existing rewrite/maintenance machinery), replacing
manifest entries atomically. Exposure window to document verbatim: commits via the DataFusion
provider's `insert_into` into a partitioned table where the SELECT list contained a computed or
reordered partition-source column; `VALUES` and plain passthrough are clean. **Fixture note:** after
Unit 1 the engine can no longer produce corruption — build the corrupted-table fixture via
manifest-level APIs, not the fixed engine.

**G2 — WG4a + WG4c: delete-writer spec-id stamp (probe first) + contract normativity.**
`position_delete_writer.rs:222` stamps `partition_spec_id` only when a `PartitionKey` is present;
builder default is `DEFAULT_PARTITION_SPEC_ID = 0`; same `if let Some(pk)` pattern at
`data_file_writer.rs:99` + `equality_delete_writer.rs:179`. Java's writer requires a
`PartitionSpec` and always stamps from it. **Probe the silent shape before sizing:** spec 0
unpartitioned + current spec id ≠ 0 → delete commits stamped 0 → read-side matcher never applies it
(rows resurrect); most other shapes fail loudly at commit. Touches row R113. WG4c rides along:
`docs/ENGINE_CONTRACT.md:224-226` — make normative: the key MUST come from the spec of the **data
files being deleted from**, never the table default; note a same-arity compatible-type wrong-spec
stamp passes commit validation and silently never applies.

**G3 — WG3 (L2+L3 only, per D2): totalise the partition-path walk.** Four abort vectors: V1
`partition.rs:168` → `struct_value.rs:59` (tuple shorter than spec); V2 `partition.rs:161` unwrap
(source column absent); V3 `transform.rs:189` (`Void` over non-primitive); V4 `datum.rs:347-349`
`unreachable!()` (literal kind ≠ derived type). Two are reachable from the fork's own **commit
path**: `SnapshotProducer::summary` (`transaction/snapshot.rs:1396-1440`) pairs `current_schema()`
with each file's own possibly-old spec (Java substitutes `UnknownType`, `PartitionSpec.java:136-139`);
the `removed_*_files` loops (`:1420`, `:1435`) bypass `validate_partition_value`; and
`file_partition_spec` (`:1356-1363`) substitutes the default spec for an unknown id. The in-code
"never panics" comment is false. **L2 (non-breaking):** totalise `partition_to_path` (Java is
lenient — `PartitionData.get(pos)` returns null past the end, renders `name=null`); add
`try_partition_to_path -> Result` for callers that can handle it; `tracing::warn!` on the lenient
fallback. **L3:** fix the two fabricated bad inputs — `physical_plan/delete.rs:1041-1048`
(`Struct::empty()` for a partitioned default spec) and `maintenance/rewrite_data_files.rs:577-586`
(`unwrap_or_else(Struct::empty)`). **Traps:** all-void specs are `is_unpartitioned() == true`
(`partition.rs:101-103`) — a naive arity rule rejects a legitimate `(void_spec, Struct::empty())`
pair; needs its own test leg + a mutation proving arity and void-case independence. NULL partition
values must stay legal — write `partition_key_new_accepts_null_value` BEFORE any validator.
Preserve the two error strings in `transaction/snapshot.rs:914-950` verbatim (asserted at `:3141`).
The *filed* fix (fallible `to_path`) stays rejected — it breaks the public `LocationGenerator`
trait, which has an out-of-crate impl.

**G4 — R161: partition-path URL-escaping parity (D6 — APPROVED; format-visible).** Java escapes
BOTH sides of each `name=value` pair (`PartitionSpec.java:205-211`, `:225`); Rust emits raw. Visible
in data-file locations AND snapshot-summary `changed-partition` keys (`snapshot_summary.rs:136`).
Implement Java-parity escaping; add GAP_MATRIX row **R161** (next unused id); byte-compare a
special-char layout against Java (interop leg or jar-oracle pin); note tables written pre-fix keep
raw paths (readers follow manifest paths — layout divergence only, not a read break).
**Format-stability attestation REQUIRED** (SEPMO v2.2 taxonomy duty). Sequenced directly after G3
because both rework the same function.

**G5 — WG4b: Java's path-keyed position-delete routing (parity; R117 demotion → re-promotion).**
Java routes any position delete with a referenced data file into a path-keyed map consulted with NO
spec/partition condition (`DeleteFileIndex.java:190-197`, `:513-529`); the fork has no such map, so
those deletes fall under the partition+spec guard — and Spark's default write granularity is FILE
(`SparkWriteConf.java:720-727`), so Java-written MoR tables routinely contain exactly this class
(silent under-delete: rows resurrect). Requirements: (a) the routing predicate is NOT just "field
set" — Java's `ContentFileUtil.referencedDataFile` (`:63-88`) also derives the path from the
`file_path` column's equal lower/upper bounds (machinery exists:
`RESERVED_FIELD_ID_DELETE_FILE_PATH`, `replace_path_bounds` at `rewrite_table_path.rs:795`);
(b) `remove_dangling_delete_files.rs:266/:352` groups only by `(spec_id, partition)` — update in
the SAME unit or a silent read miss becomes an irreversible metadata delete; (c) expect
`FileScanTask.deletes` → bin-pack weights to move — re-baseline the nightly consciously (the same
4096 knife edge #169 fixed); (d) governance: R117 is ✅ with interop evidence — demote honestly in
the same change, re-promote with a Spark-shaped FILE-granularity interop fixture (ask RePark — see
the Desktop work order). The **filed** DELSPEC validation stays CLOSED-REFUTED: read-side dual key
is line-for-line Java, the write-side guard is deliberate and Java-cited
(`transaction/snapshot.rs:410-430`), and the proposed check is a fork invention unreachable for
every delete file the fork currently writes.

**G6 — WG5: null-bit propagation family (one shared helper; matrix touch).** Root:
`arrow/value.rs:598` + `caching_delete_file_loader.rs:888` hand back a struct child WITHOUT the
parent's validity, while `record_batch_projector.rs:184-198` does the same walk correctly. Members:
(a) `optional struct<required int>` with a null outer row falsely rejected on the Avro write path
(RED-provable parity break today); (b) same accessor yields `= <garbage>` instead of `IS NULL` in
the equality-delete key path (nested keys separately broken one line away); (c) `null_count()`
where `logical_null_count()` is required (`physical_plan/delete.rs:1442`) — a dictionary-encoded
NULL slips past the required-column guard and is **persisted**; (d) `.value(i)` without `is_valid`
on the `_file`/`_pos` decode. Lift the `NullBuffer::union` walk into one shared helper. ~4-6h.

**G7 — WG6 (mechanical half only).** `record_batch_transformer.rs:85`;
`caching_delete_file_loader.rs:546/552` column arity; ~25 `downcast_ref().unwrap()` in
`transform/bucket.rs` + `truncate.rs`; and `expr_to_predicate.rs:311`'s truncating `as i32` on a
caller-supplied `Date64` — a wrong pushdown predicate → silently wrong scan results (the function
already returns `Option`). (`record_batch_projector` bounds ride Unit 1; API-breaking members ride
Unit 3.)

**G8 — lost-wakeup class in `delete_filter.rs` (bigger than the advertised cherry-pick).**
Upstream #2859 covers 1 of 3 sites and does not apply cleanly (context drift); its test does not
port (no runtime plumbing in our `DeleteFilter`); a second caller on the DV path stops compiling;
and `PosDelState` has no `Failed` variant — a loader that dies before publishing parks every waiter
forever (verified still true at `ce2affc9`; `EqDelState` has the variant, `PosDelState` does not).
Port the WaitFor pattern from the #171 `delete_file_index.rs` work (same file family), add
`PosDelState::Failed` + the eq-delete pair fix.

---

## Unit 3 — breaking follow-up (Mode A, after Unit 2 merges)

**WG3-L1:** `PartitionKey::new -> Result` validating arity, spec/schema binding, per-value type
compatibility — Java-parity (an invalid tuple is unrepresentable: array sized from the spec,
`StructTransform.java:49/63`, throws on missing accessor `:57-58`). Churn: 58 call sites across 34
files including **6 inside `rust,no_run` doc fences** in `writer/mod.rs` (`no_run` still compiles —
missing them breaks `make unit-test` with an unrelated-looking error). Plus WG6's API-breaking
members: `CurrentFileStatus`'s three post-`close()` unwraps (`data_file_writer.rs:125/129/133`).
Call out every breaking surface in the PR body for downstream pins.

## Units 4-5 — scale gate (H7 remainder, own signed-off scopes)

**Unit 4 — H7-S2:** copy-on-write streaming DML. **Unit 5 — H7-P1:** DML pushdown — the
NOT-over-dropped-conjunct under-delete footgun MUST be fixed first (pre-condition inside the unit).
Scope brief: `task/h7-dml-streaming-scope.md`. Sequenced after the correctness gates; re-scope at
signing time against whatever landed in between.

## Watch items / tail backlog (no units scheduled)

- Nightly Interop dispatch on main = live proof of #169/#170 (first run after 2026-07-24).
- Prune leftover merged local/remote review branches (blocked from this session by the permission
  classifier; command in the session wrap-up).
- Decimal eq-delete read; M8 (marginal) — H7 ledger.
- R159/R160 residues (token-exchange grant; per-path vended credential selection).
- Core `Error` source-rendering rework (closes the #171 rest-crate DOCUMENTED-LEAK pin) + core
  property-map redaction sweep.
- Core DML-rescan name-binding (loud-fail residual named by #171 df-provider work).
- F2 partial-delete-set under-delete window; F3 tokio feature under-declaration (needs Cargo.toml
  approval).
- SessionCatalog: deferred-as-dead-surface (unchanged).

## Per-unit protocol (all units)

Persistent worktree `iceberg-rust-ws`; claim on the PrimarySync board BEFORE first edit; PR title
carries the workstream tag (R3); content-verify before acting on "merged" (R4); RePark repin
announcements in `RoadMapSync.md` after each merge that moves their pins. Gate = `typos .` + `cargo
fmt --all -- --check` + `cargo clippy --all-targets --all-features -D warnings` + lib/crate suites
+ `cargo build -p iceberg --no-default-features` + `make check-matrix-anchors` after any matrix
edit, chained into the commit in ONE `&&` chain. SEPMO v2.2 duties: java-parity attestation per
unit; format-stability attestation for G4/R161 (and any unit that changes written bytes); metrics
ledger `task/sepmo-metrics.md`; independent Critic per PR, convergence is the Critic's call.
