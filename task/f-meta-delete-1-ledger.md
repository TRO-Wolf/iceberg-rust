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

# F-META-DELETE-1 — metadata-only DELETE decision (ledger)

**Branch:** `fix/f-meta-delete-1`, cut from fork main. RePark slate `IPI-08`.
**Scope:** port Java `SparkTable.canDeleteUsingMetadata` (iceberg-spark 1.11.0) as a public fork
API on `Table`. RePark routing that calls the API is run 24c's half — untouched here.
**Oracle:** the run-24d metadata-delete oracle — Spark 4.1.2 + Iceberg 1.11.0, 18 shapes ×
v2/v3 × CoW/MoR. Verbatim copies of bytecode dumps and probe output live below, not in code.

## 1. The Java rule — verified from `iceberg-spark-runtime-4.1_2.13-1.11.0.jar` bytecode

`javap -c -p org.apache.iceberg.spark.source.SparkTable` on the 1.11.0 jar gives
`private boolean canDeleteUsingMetadata(Expression, String branch)`:

```
caseSensitive = SparkUtil.caseSensitive(spark)            // spark.sql.caseSensitive
if ExpressionUtil.selectsPartitions(deleteExpr, table(), caseSensitive) -> return true
scan = table().newScan()
        .filter(deleteExpr)
        .caseSensitive(caseSensitive)
        .includeColumnStats()
        .ignoreResiduals()
if branch != null: scan = scan.useRef(branch)
else if this.snapshot != null: scan = scan.useSnapshot(this.snapshot.snapshotId())
tasks = scan.planFiles()
evaluatorCache: Map<Integer, Evaluator>
strict = new StrictMetricsEvaluator(this.schema /* SparkTable.schema field */, deleteExpr)
return Iterables.all(tasks, task ->
    evaluatorCache.computeIfAbsent(task.spec().specId(),
        id -> new Evaluator(spec.partitionType(), Projections.strict(spec).project(deleteExpr)))
        .eval(task.file().partition())
    || strict.eval(task.file()))
```

Verified bytecode details that refine the brief's pseudocode:

1. **Metrics schema is `this.schema`, not `SnapshotUtil.schemaFor(table, branch)`.** Bytecode
   reads `getfield SparkTable.schema` (instance field, = `table.schema()` for a normal SparkTable)
   even on the `useRef` path. The strict metrics evaluator therefore binds against the CURRENT
   table schema regardless of the requested ref. Fork analog: `metadata().current_schema()`.
2. **`caseSensitive` comes from the Spark session** (`SparkUtil.caseSensitive(spark)` reads
   `spark.sql.caseSensitive`). Fork analog: an explicit `case_sensitive: bool` parameter.
3. **`selectsPartitions` checks ALL specs**, not just the current one:
   `table.specs().values().stream().allMatch(spec -> selectsPartitions(expr, spec, cs))`.
4. **`selectsPartitions(expr, spec, cs)`**:
   `spec.isUnpartitioned()` → `false`; else
   `equivalent(Projections.inclusive(spec, cs).project(expr),
              Projections.strict(spec, cs).project(expr),
              spec.partitionType(), cs)`.
5. **`equivalent(a, b, struct, cs)`** =
   `bind(struct, rewriteNot(a), cs).isEquivalentTo(bind(struct, rewriteNot(b), cs))` — structural
   equality of the two projections bound to the partition type.
6. **`Projections.strict(spec)` (1-arg) binds the unbound predicate with caseSensitive = true**
   (`BaseProjectionEvaluator` stores `caseSensitive`; `UnboundPredicate.bind(spec.schema().asStruct(),
   caseSensitive)`; `project()` runs `RewriteNot` on the input first). Same for the
   1-arg `Evaluator(StructType, Expression)` — binds with `true`. So the partition arm and the
   metrics arm both bind `rewriteNot(deleteExpr)` to the (current) table schema with cs=true;
   only the inclusive filter arm and the projections' own bind in `selectsPartitions` honor the
   session flag.
7. **`StrictMetricsEvaluator(Schema, Expression)`** ctor =
   `BoundPredicates.and(...) via rewriteNot then Binder.bind(schema.asStruct(), expr, true)` —
   rewrite-not + case-sensitive bind, then a `StrictMetricsEvaluator` over the bound tree.
8. **Per-task order: partition arm first, metrics arm second** (`eval(...) || strict.eval(...)` —
   short-circuit skips metrics when partition evaluation is true).
9. **`Iterables.all` is vacuous-true**: a scan that plans zero files returns true. This is what
   produces Spark's empty `delete` snapshot for `no_match` (`id = 99`).
10. **Delete files attached to a task do not enter the decision** — only `task.file()` (the data
    file's partition tuple + column metrics) and `task.spec()` are read.
11. **`Projections` binds unbound exprs to `spec.schema()`** (the schema carried by the spec,
    Java 1.11 `PartitionSpec.schema()`), not the table schema. Fork `PartitionSpec` carries no
    schema; the closest analog is the snapshot/current schema — named divergence candidate,
    see §5 D3.
12. **`StrictProjection.predicate(bound)`** ORs `transform.projectStrict(field.name, pred)` over
    every partition field whose source id matches; a source field with no partition fields yields
    `alwaysFalse`. `and`/`or` compose structurally; `not` throws `[BUG]` (never reached —
    `project()` rewrite-nots first).

### The Spark caller (CoW + MoR both)

`OptimizeMetadataOnlyDeleteFromTable` rewrites `DeleteFromTable` to `DeleteFromIcebergTable`
with `canDeleteWhere`-style checks; when `canDeleteUsingMetadata` is true the committed action is
`table.newDelete().deleteFromRowFilter(expr)` → `DeleteFiles` → snapshot `operation = "delete"`,
`deleted-data-files`, no delete file. Verified `BaseOverwriteFiles.operation()` returns
`"delete"` when the overwrite deletes files but adds none — so the two oracle cells the recorder
labels `op: delete` on the **CoW row-level path** (`not_in_whole_v2_cow`,
`prior_deletes_then_rest_v2_cow`) are delete-only overwrites, NOT metadata deletes; the decision
is `false` there, consistent with the bytecode.

## 2. Fork mapping — what exists, what is missing

| Java piece | Fork piece | Status |
|---|---|---|
| `Table.newScan().filter(e).caseSensitive(cs).includeColumnStats().ignoreResiduals()[.useRef(r)]` | `TableScanBuilder::with_filter(Predicate)` + `with_case_sensitive(bool)` + `select_all()` + `with_file_prune_only(Predicate)` + `use_ref(String)` | exists. `with_file_prune_only` = Java `ignoreResiduals`: plan-time pruning still runs, no residual lands on tasks. `includeColumnStats` is a Spark-load hint; the fork always carries file metrics — no-op. |
| `scan.planFiles()` → `FileScanTask` with `file()` metrics | `FileScanTask` carries partition + spec but NOT the full `DataFile` metrics | **missing** — the decision needs manifest-entry-level `DataFile`s. New `PlanContext::plan_matching_manifest_entries` reuses `build_manifest_file_contexts_from_files` + a new `ManifestFileContext::fetch_manifest_entries` + a new `ManifestEntryContext::survives_plan_filter` (extracted verbatim from `TableScan::process_data_manifest_entry`'s is_alive / content / partition-filter / inclusive-metrics checks). |
| `ExpressionUtil.selectsPartitions` | nothing equivalent | **missing** — new `selects_partitions` helper in `table.rs`. |
| `Projections.inclusive/strict(spec)` | `InclusiveProjection::new(spec).project(&bound)` / `StrictProjection::new(spec).strict_project(&bound)` | exists (`expr/visitors/{inclusive,strict}_projection.rs`). Fork projections take a `BoundPredicate`; caller binds `rewrite_not()` first. |
| `Expressions.equivalent` (rewriteNot + bind + isEquivalentTo) | `BoundPredicate` derives `PartialEq` | exists. Fork bind already simplifies `and(p, AlwaysTrue)` → `p`, `and(p, AlwaysFalse)` → `AlwaysFalse`, `or(p, AlwaysFalse)` → `p`, so projected trees normalize the same way `Expressions.and/or` do. |
| `Evaluator(partitionType, projExpr)` | `ExpressionEvaluator::new(bound)` + `.eval(&DataFile)` | exists — evaluates the data file's partition tuple. |
| `StrictMetricsEvaluator(schema, expr)` + `eval(file)` | `StrictMetricsEvaluator::eval(&BoundPredicate, &DataFile)` | exists (currently `#[allow(dead_code)]` — this is its first consumer). |
| per-spec evaluator cache `Map<Integer, Evaluator>` | plain `HashMap<i32, ExpressionEvaluator>` local to the call | trivial. |
| `SparkUtil.caseSensitive` | `case_sensitive: bool` param | API surface. |
| `branch` (`useRef`) | `use_ref` + `Option<&str>` param | exists. |
| `this.snapshot` (SparkTable time-travel) | `snapshot_id` scan option — not part of this API | out of scope. |

## 3. Public API

```rust
impl Table {
    pub async fn can_delete_using_metadata(
        &self,
        predicate: &Predicate,
        branch: Option<&str>,
        case_sensitive: bool,
    ) -> Result<bool>;
}
```

Placement: `crates/iceberg/src/table.rs` (544/1000 lines — `scan/mod.rs`,
`transaction/mod.rs`, and `transaction/delete_files.rs` are all pinned at their exact legacy
ceilings). `plan_matching_manifest_entries` + `fetch_manifest_entries` +
`survives_plan_filter` go in `scan/context.rs` (576/1000).

## 4. Oracle table (run-24d, measured on Spark 4.1.2 + Iceberg 1.11.0)

Table `(id INT, cat STRING, v STRING)`; file1 `(1,'x','a'),(2,'x','b'),(3,'y','c')`;
file2 `(7,'g','x')`; file3 `(8,'g','y'),(9,'g','z')` where named.

| shape | predicate | decision | oracle evidence |
|---|---|---|---|
| whole_one_file | `id = 7` | TRUE | `delete`, deleted-data-files 1 |
| whole_two_files | `id >= 7` (3 files) | TRUE | `delete`, 2 files |
| mixed_partial_and_whole | `id >= 3` | FALSE | MoR delete files; CoW `overwrite` |
| partial_only | `id = 2` | FALSE | MoR delete files; CoW `overwrite` |
| all_rows_true / no-pred | `true` / none | TRUE | `delete`, 2 files |
| no_match | `id = 99` | TRUE (vacuous) | `delete`, empty snapshot |
| string_eq_whole | `cat = 'g'` | TRUE | `delete` |
| partition_select | spec `cat`, `cat = 'x'` | TRUE | `delete` (selectsPartitions) |
| partition_select_bucket | spec `bucket(4,id)`, `id = 7` | TRUE | `delete` (selectsPartitions) |
| partition_plus_metrics | spec `cat`, `cat='x' AND id<=2` | TRUE | `delete` (metrics arm) |
| partition_partial | spec `cat`, `cat='x' AND id=1` | FALSE | NOT metadata |
| prior_deletes_then_whole | `id=1` then `id<=3` | TRUE | v2 keeps old delete file; v3 drops DV |
| prior_deletes_then_rest | `id=1` then `id IN (2,3)` | FALSE | MoR: strict IN on [1,3] false. CoW `op:delete` = delete-only OVERWRITE (row-level), decision still false |
| is_null_whole | `v IS NULL`, file2 all-null | TRUE | null counts |
| or_whole | `id = 7 OR id = 99` | TRUE | `delete` |
| not_in_whole | `id NOT IN (1,2,3)` | FALSE | file1 candidate, strict NOT_IN false |
| nondeterministic_like | `v LIKE 'x%'` → startsWith | TRUE | `delete` — needs REAL strict starts_with |

## 5. Named divergences / residues found during measurement

- **D1 — strict `starts_with` is a fork stub.** `strict_metrics_evaluator.rs` `starts_with` /
  `not_starts_with` return `ROWS_MIGHT_NOT_MATCH` unconditionally; Java implements prefix-bounds
  logic. Without it the `nondeterministic_like` pin is FALSE vs Java TRUE. **Fix in scope** — the
  file is at its 1922-line ceiling, so the Java logic lands in a new sibling file
  (`expr/visitors/strict_prefix_eval.rs`), the evaluator's metric accessors go `pub(crate)` (net
  line delta zero), and the two visitor arms delegate.
  Java semantics ported from `StrictMetricsEvaluator$MetricsEvalVisitor` bytecode:
  `startsWith(ref, lit)`: nested column or `canContainNulls` → might-not-match; requires BOTH
  lower and upper bounds; `lower.length() < prefix.length()` or `upper.length() < prefix.length()`
  → might-not-match; must-match iff `lower.subSequence(0, prefixLen) == prefix` AND
  `upper.subSequence(0, prefixLen) == prefix` (char-sequence compare).
  `notStartsWith(ref, lit)`: nested → might-not-match; `containsNullsOnly` → MUST-match; else
  `trunc = min(prefix.len, bound.len)`; must-match iff `lower.subSequence(0, trunc) > prefix` OR
  `upper.subSequence(0, trunc) < prefix`; otherwise might-not-match.
- **D2 — `no_match` commit path.** The DECISION is vacuous-true (Java + fork alike), but the
  fork's `delete_from_row_filter` with a predicate matching zero files currently rejects the
  empty commit (existing `DeleteFiles` validation) while Spark commits an empty `delete`
  snapshot. The decision pin asserts `true`; the commit-shape pin records the fork's actual
  `Err` today and names the divergence (RePark routing in run 24c decides whether to follow).
- **D3 — projection bind schema.** Java binds the delete predicate to `spec.schema()` inside
  `Projections` and to `this.schema` in the metrics evaluator; the fork binds to
  `metadata().current_schema()` for both. Identical for every fixture here; a renamed-column +
  evolved-spec edge could diverge (OPEN, recorded, not chased).
- **D4 — `isEquivalentTo` vs `PartialEq`.** Java's bound-expression equivalence for And/Or is
  order-sensitive structural equality; fork `PartialEq` on `BoundPredicate` is the same shape
  (binary `LogicalExpression<_,2>`). Equivalent for every projection output pair seen here.
- **D5 — CoW `delete` ops are delete-only overwrites** (`BaseOverwriteFiles.operation()` dynamic
  op), not metadata deletes. The decision is false in both cells; recorded so run 24c does not
  mistake them for metadata-delete commits.

## 6. Implementation plan (steps 2–4 of the brief)

1. `ManifestEntryContext::survives_plan_filter` — extract lines `scan/mod.rs` 971–1011 verbatim
   into `scan/context.rs`; `process_data_manifest_entry` calls it (frees ~36 lines in a
   ceiling-pinned file).
2. `ManifestFileContext::fetch_manifest_entries(&self) -> Result<Vec<ManifestEntryContext>>` —
   same manifest load + promoted-partition mapping as `fetch_manifest_and_stream_manifest_entries`,
   collected into a `Vec` (no channel).
3. `PlanContext::plan_matching_manifest_entries(&self) -> Result<Vec<ManifestEntryContext>>` —
   manifest list → `build_manifest_file_contexts_from_files` (data manifests only get fetched) →
   `fetch_manifest_entries` → `survives_plan_filter`.
4. `Table::can_delete_using_metadata` — `selects_partitions` early-out; then scan built with
   `with_case_sensitive` + `with_file_prune_only` + `use_ref`; `rewrite_not + bind(current_schema,
   true)` once → feeds both `StrictMetricsEvaluator::eval` and `StrictProjection::strict_project`
   (Java binds both to the table schema with cs=true); per-entry
   `ExpressionEvaluator` on the strict projection bound to `spec.partition_type(schema)` cs=true,
   partition arm first.
5. Strict `starts_with` / `not_starts_with` prefix-bounds logic (D1).

### Mutation obligations

- Drop the partition arm → `partition_plus_metrics`/`partition_select`-family pins must stay green
  via metrics where Java says so, but `partition_partial`-style partition-only proofs go red — the
  named pin is the bucket/identity whole-partition shape that metrics alone cannot prove.
- Drop the metrics arm → `whole_one_file` (metrics-only on unpartitioned spec) goes red.
- Drop `use_ref` → the branch pin (decision differs main vs branch) goes red.

## 7. Test matrix (file `transaction/delete_files/tests/metadata_delete_tests.rs`)

Fixtures: real parquet data files via the existing `write_data_file`/`write_position_delete_file`
helpers + a stats-precise synthetic-`DataFile` builder where bound/null-count control matters
(the synthetic path is required: only it can set `value_counts`/`null_value_counts`/`lower`/`upper`
exactly, e.g. all-null `v`, or `id IN (2,3)` on bounds `[1,3]`).

| test | fixture | predicate | expected |
|---|---|---|---|
| `whole_one_file` | unpart, f1 y[1,3], f2 y[7,7] | `y = 7` | true |
| `whole_two_files` | + f3 y[8,9] | `y >= 7` | true |
| `mixed_partial_and_whole` | f1+f2 | `y >= 3` | false |
| `partial_only` | f1+f2 | `y = 2` | false |
| `all_rows_true` | unpart | `AlwaysTrue` | true |
| `no_match_vacuous_true` | unpart | `y = 99` | true + commit-shape pin (D2) |
| `string_eq_whole` | unpart, cat bounds | `cat = 'g'` | true |
| `partition_select_identity` | spec identity(x) | `x = <part>` | true |
| `partition_select_bucket` | spec bucket(4,x) | `x = 7` | true |
| `partition_plus_metrics` | spec identity(x) | `x = <part> AND y <= 2` | true |
| `partition_partial` | spec identity(x) | `x = <part> AND y = 1` | false |
| `prior_deletes_then_whole` | f1 + pos-del, f2 | `y <= 3` | true |
| `prior_deletes_then_rest` | f1 + pos-del | `y IN (2,3)` | false |
| `prior_deletes_then_rest_cow` | f1' y[2,3], no del | `y IN (2,3)` | false |
| `is_null_whole` | f2 all-null z | `z IS NULL` | true |
| `or_whole` | unpart | `y = 7 OR y = 99` | true |
| `not_in_whole` | unpart | `y NOT IN (1,2,3)` | false |
| `nondeterministic_like` | unpart, f2 v=[x,x] | `v STARTS WITH 'x'` | true (needs D1 fix) |
| `branch_ref` | main f1, branch adds partial f2 | `y <= 3` | main true, branch false |
| `case_insensitive` | column `Y` vs pred `y` | cs=false vs true | pins binding flag |
| `delete_summary_pins` | each TRUE shape | `delete_from_row_filter` commit | op `delete`, deleted-data-files N, no delete manifests |

## 8. Gate + mutation record

Recorded at HEAD `c5af541a` (four commits over `origin/main`: `c63772d5` ledger, `1ed5048d`
red-first pins, `f82a4e46` implementation, `c5af541a` gate fixes — rustfmt layout + two
`#[allow]` attributes, zero behavioral change).

### Mutations (each reverted after measurement; suite re-verified 23/23 green at HEAD)

- **Partition arm dropped** (`if evaluator.eval(file)` kept, metrics OR removed): 22/23 pass,
  `partition_arm_only_proof` red — the file provable only by strict partition evaluation is
  the named red. Partition arm is load-bearing — **PROVEN**.
- **Metrics arm dropped** (`StrictMetricsEvaluator::eval` OR removed): 11/23 pass, 12 red —
  `whole_one_file`, `whole_two_files`, `string_eq_whole`, `is_null_whole`, `or_whole`,
  `not_in_whole`'s sibling `partition_select_bucket`, `partition_plus_metrics`,
  `nondeterministic_like`, `case_sensitivity_binds_the_filter`, `prior_deletes_then_whole_v2`,
  `prior_deletes_then_whole_v3`, `uses_the_named_ref`. Metrics arm is load-bearing — **PROVEN**.
  (`partition_select_bucket` red is expected and Java-consistent: bucket `strict_project`
  yields no projection for `eq` — same as Java's `projectStrict`, which only projects
  NotEq/NotIn through collision-prone transforms — so `selectsPartitions` does not fire and
  the file is proven by the metrics arm in both implementations.)
- **`use_ref` dropped** (branch argument ignored): 22/23 pass, `uses_the_named_ref` red —
  named-reference handling is load-bearing — **PROVEN**.

### Gates at HEAD

| gate | command | result |
|---|---|---|
| fmt | `cargo fmt --all -- --check` | clean — PROVEN |
| clippy | `cargo clippy -p iceberg --all-targets -- -D warnings` | clean — PROVEN |
| size | `./scripts/check_rust_file_size.sh` | 524 files clean (96 legacy ceilings) — PROVEN |
| comment ban | `comment_ban.py <clone> origin/main HEAD` | hits=0 — PROVEN |
| artifacts | `./scripts/check_agent_artifacts.sh` | OK — PROVEN |
| comment blocks | `./scripts/check_comment_blocks.sh` | OK — PROVEN |
| matrix anchors | `./scripts/check_matrix_anchors.sh` | 88 rows anchored — PROVEN |
| toml | `taplo check` | clean — PROVEN |
| unused deps | `cargo machete` | none — PROVEN |
| spelling | `typos` | clean — PROVEN |
| pins | `cargo test -p iceberg --lib can_delete_using_metadata` | 23/23 green at HEAD — PROVEN |
| delete regressions | `cargo test -p iceberg --lib delete_files` | 203/203 green at HEAD — PROVEN |
| scan regressions | `cargo test -p iceberg --lib scan` | 292/292 green at HEAD — PROVEN |

### Residues

- D3 (bind-schema source on renamed-column + evolved-spec edge) — **OPEN**, recorded in §5.
- D2 (`no_match` commit path returns `Err` where Spark commits an empty `delete` snapshot) —
  pinned as fork-today behavior; the decision itself is correct vacuous-true — **OPEN**,
  routed to run 24c's routing decision.
- Independent Critic pass — not run in this lane; deferred to the PR-level review step of the
  owner's shipping workflow — **OPEN**.
