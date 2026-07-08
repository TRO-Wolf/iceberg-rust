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

# Engine Integration Contract — the `iceberg` core ↔ downstream query engine

> **Status: DRAFT (2026-07-01).** This is the written half of the seam the 2026-06-20 direction
> re-anchor declared: *"the `iceberg` CORE crate is the stable engine-facing contract — the
> downstream builds its OWN DataFusion `TableProvider` over it."* The equivalence proof
> (`crates/iceberg/tests/interop_scan_exec.rs`) is the executable half; this file names everything
> else a DELETE/UPDATE/MERGE engine must know. **DoD to flip DRAFT → NORMATIVE:** every row in
> §5 bytecode-verified against Java 1.10.0 (`SparkWrite` / `SparkCopyOnWriteOperation` /
> `SparkPositionDeltaWrite` / `SparkRowLevelOperationBuilder`) per the house `javap`/live-oracle
> discipline, plus ONE interop conflict scenario per cell. Tracked in `task/todo.md`
> §"ACTIVE (2026-07-01)".

---

## 1. The seam

- **Core owns:** table metadata + spec semantics, scan planning, merge-on-read delete application,
  the full commit/write action surface, conflict validation, transaction retry.
- **The engine owns:** SQL planning (including the MERGE rewrite — §6), expression translation,
  physical execution, distribution + ordering before write (§7), COW-vs-MoR mode selection, and
  the isolation-level policy that picks which validations to enable (§5).
- `crates/integrations/datafusion` is the **reference implementation** of this contract. Note
  (2026-07-01): it is currently ahead of this document — partitioned copy-on-write DELETE/UPDATE
  and partitioned merge-on-read DELETE/UPDATE landed there (#131, #133) with a partition-aware
  `TaskWriter`. Treat its `physical_plan/{delete,write,commit}.rs` as worked examples of §4–§7.
- **Status change (2026-07-01, decided with the named consumer):** `crates/integrations/datafusion`
  is **promoted from reference implementation to a supported product surface**. RePark consumes it
  directly (rev-pinned `[patch.crates-io]`; RePark ADR-0003 mirrors this decision): scan + INSERT +
  DELETE + UPDATE flow through `IcebergCatalogProvider` / `IcebergTableProvider`'s standard
  DataFusion `TableProvider` methods rather than being re-implemented downstream — the H7 ladder
  already treats this crate as load-bearing. Consequences: public-API changes in this crate are
  **consumer-breaking** (the DML exec types stay `pub(crate)`, crossing the boundary only as opaque
  `ExecutionPlan`s); the ownership split above is UNCHANGED — the MERGE rewrite, COW-vs-MoR mode
  policy, and isolation-level selection remain engine-owned (§5–§6). The boundary rule: anything
  provable by the Java interop oracle lives in this fork (including engine-generic DataFusion
  execs); anything Spark-flavored lives in the consumer.

## 2. Read surface

Two supported paths, proven equivalent:

1. **Built-in:** `Table::scan().to_arrow()` is the all-in-one read — merge-on-read
   position/equality deletes and V3 deletion vectors are applied internally. When the engine
   drives per-task reading instead, use `Table::scan()` / `Table::batch_scan()` → `plan_tasks()`
   (split planning / `ScanTaskGroup`) and feed the task groups to your own reader — `to_arrow()`
   is a method on `TableScan`, not on the `plan_tasks()` output. *(Corrected 2026-07-01: the two
   chains were previously drawn as one pipeline.)*
2. **BYO physical scan:** the engine reads data files itself and applies
   `iceberg::arrow::DeleteFilter` (public since #117; mirrors Java
   `org.apache.iceberg.data.DeleteFilter`). The equivalence proof
   (`crates/iceberg/tests/interop_scan_exec.rs`, `test_engine_deletefilter_*`) shows a raw read +
   `DeleteFilter::apply` reproduces the built-in merge-on-read scan EXACTLY — position, equality,
   and combined; multi-file-per-partition (identity-partitioned tables). Non-identity partition
   transforms are proven for the BUILT-IN scan only (`test_nonidentity_scan_exec_*` vs the Java
   oracle); a `DeleteFilter`-equivalence test over a non-identity layout is still owed
   *(corrected 2026-07-01 — previously over-claimed as deletefilter coverage)*.

Incremental reads: `IncrementalAppendScan` / `IncrementalChangelogScan` (whole-data-file level;
row-level CDC is an open queue item). Branch/tag reads via `use_ref`.

## 3. Reserved metadata columns

Source of truth: `crates/iceberg/src/metadata_columns.rs` (names + field-id constants there;
request by field id through `project_field_ids`). The DML-critical set:

| Column | Field id const | Engine use |
|---|---|---|
| `_file` | `RESERVED_FIELD_ID_FILE` | identify source data file per row (COW affected-file detection; position-delete targeting) |
| `_pos` | `RESERVED_FIELD_ID_POS` | row ordinal within its file (position deletes; landed #115) |
| `_deleted` | `RESERVED_FIELD_ID_DELETED` | changelog / deleted-row exposure |
| `_spec_id` | `RESERVED_FIELD_ID_SPEC_ID` | partition-spec provenance |
| `_partition` | `RESERVED_FIELD_ID_PARTITION` | partition struct (routing; delete-file stamping) |
| changelog trio | `RESERVED_FIELD_ID_CHANGE_TYPE` / `RESERVED_FIELD_ID_CHANGE_ORDINAL` / `RESERVED_FIELD_ID_COMMIT_SNAPSHOT_ID` (columns `_change_type` / `_change_ordinal` / `_commit_snapshot_id`) | CDC reads |
| delete-file schema | `RESERVED_FIELD_ID_DELETE_FILE_PATH` (`file_path`) / `RESERVED_FIELD_ID_DELETE_FILE_POS` (`pos`) — NOT `RESERVED_FIELD_ID_POS`, which is the different `_pos` metadata column | writing position-delete files (2147483546 / 2147483545) |

The engine-boundary proof (#116): scan `_file`/`_pos` → write position-delete → `RowDelta` commit
→ re-scan omits exactly those rows.

## 4. Write surface — operation → mode → core action

| Engine operation | Mode | Core action(s) |
|---|---|---|
| INSERT / append | — | fast append (`Transaction` append) or merge append |
| DELETE | copy-on-write | rewrite affected files' survivors; commit `OverwriteFiles` `.delete_files(affected).add_files(rewritten)` |
| DELETE | merge-on-read | write position-delete files (or V3 DVs), one per `(spec_id, partition)` group, stamped with the matching `PartitionKey`; commit `RowDelta` |
| UPDATE | copy-on-write | rewrite affected files in full (matched rows take SET values); `OverwriteFiles` as above; a partition-key-changing UPDATE re-routes rows via the partition-aware writer |
| UPDATE / MERGE | merge-on-read | position/equality deletes for matched rows + new data files; ONE `RowDelta` commit (added deletes inherit the commit's sequence number) |
| INSERT OVERWRITE (dynamic) | — | `ReplacePartitions` |
| compaction commit | — | `RewriteFiles` (the action layer's `RewriteDataFiles` wraps it) |

## 5. Isolation level → validation recipes  ·  **DRAFT — verify each cell before relying on it**

Rust builder methods, verbatim from `transaction/{row_delta,overwrite_files,replace_partitions}.rs`. **Base for every
row-level op, both modes:** `validate_from_snapshot(scan_snapshot_id)` +
`conflict_detection_filter(command_condition)` + `case_sensitive(engine_setting)`.

| Command | Mode | Isolation | Enable (beyond base) |
|---|---|---|---|
| DELETE | COW (`OverwriteFiles`) | snapshot | `validate_no_conflicting_deletes()` — a concurrent delete file touching a rewritten data file would be silently dropped with it |
| DELETE | COW | serializable | above + `validate_no_conflicting_data()` — a concurrent insert matching the condition violates serializability |
| DELETE | MoR (`RowDelta`) | snapshot | `validate_data_files_exist(referenced_files)` + `validate_deleted_files()` — referenced files must not have been compacted away or deleted concurrently |
| DELETE | MoR | serializable | above + `validate_no_conflicting_data_files()` |
| UPDATE / MERGE | COW | snapshot | `validate_no_conflicting_deletes()` |
| UPDATE / MERGE | COW | serializable | above + `validate_no_conflicting_data()` |
| UPDATE / MERGE | MoR | snapshot | `validate_data_files_exist(...)` + `validate_deleted_files()` + `validate_no_conflicting_delete_files()` — the op READ rows to produce output; concurrent deletes of those rows conflict |
| UPDATE / MERGE | MoR | serializable | above + `validate_no_conflicting_data_files()` |
| INSERT OVERWRITE | `ReplacePartitions` | append-only guard | `validate_append_only()` where the engine requires it (landed 2026-06-17, row R146) |

Provenance note: this table is reconstructed from the Java Spark row-level-operation builders (the
logic deliberately OUT of `iceberg-core` scope, hence out of the GAP_MATRIX) — it is exactly the
knowledge a downstream engine gets wrong without a written recipe. Each cell carries DRAFT weight
until the §DoD verification lands. A validation failure is **non-retryable** by design (Java's
`ValidationException`) and propagates out of the retry loop — the engine should surface it, not
loop.

## 6. MERGE is engine-owned

DataFusion at the pinned family (52.x) plans DELETE and UPDATE but has **no MERGE INTO planner**.
The engine therefore owns the MERGE rewrite end-to-end: source↔target join, the
matched / not-matched / not-matched-by-source clause application, row-splitting into
(position-deletes + inserted rows) for MoR or (rewritten files) for COW, and the **cardinality
check** (error when >1 source row matches a target row — Spark's `MERGE_CARDINALITY_VIOLATION`
error condition).
Core's contribution is §4/§5: `RowDelta`/`OverwriteFiles` + validations. Do not wait for core to
grow MERGE semantics; it will not (out of parity scope).

## 7. Distribution & ordering expectations

- Cluster or fan out rows by partition before writing: the projection stage (reference impl
  `physical_plan/project.rs`) computes `_partition` via `PartitionValueCalculator`; the
  partition-aware `TaskWriter` (`task_writer.rs`) consumes the precomputed `_partition` column
  and routes per row. Clustered input may use the cheaper clustered writer.
- Position-delete files: **one file per `(spec_id, partition)` group**, stamped with the matching
  `PartitionKey` (see #131 U3 — an unstamped delete file on a partitioned table fails
  `validate_partition_value` at commit). Keep `MetricsConfig::for_position_delete` so
  `file_path`/`pos` bounds stay Full (pruning precision).
- Sort-order application before write is engine-owned; core records the order.

## 8. Commit semantics

- Retry: `Transaction::commit` wraps `do_commit` in `backon` exponential backoff driven by the
  `commit.retry.*` table properties (`num-retries`, `min-wait-ms`, `max-wait-ms`,
  `total-timeout-ms`), retrying only `Error::retryable()` — and NEVER an
  `ErrorKind::CommitStateUnknown`, regardless of the flag. Each attempt REFRESHES the table,
  re-runs every action's `validate` against the refreshed base (non-retryable on failure), then
  re-applies.
- **Ambiguous commit outcome (GAP_MATRIX row R157, 🟡 since 2026-07-08).** A catalog failure
  AFTER the update request may have durably landed (timeout awaiting the response, 5xx,
  connection reset mid-response) now surfaces as **`ErrorKind::CommitStateUnknown`** — never
  auto-retried, no cleanup, surfaced (Java `CommitStateUnknownException` semantics). All four
  catalogs (REST, SQL, Glue, S3 Tables) classify commit-path transport failures sent-vs-unsent.
  **The engine MUST catch this kind and reconcile before re-running** — the library does not yet
  port Java's reconciliation-by-refresh (`checkCommitStatus`), so the mitigation stands:
  **(a)** stamp every commit with a unique summary property (e.g. `engine.operation-id`) via the
  snapshot-summary surface; **(b)** on `CommitStateUnknown`, reload the table and scan recent
  snapshots' summaries for that id BEFORE re-running; **(c)** never delete the files a failed
  attempt wrote until (b) confirms the commit is truly absent.
- S3 Tables runs **service-side maintenance** (compaction, snapshot expiry) that commits
  concurrently with the engine — treat `CommitFailed` requirement mismatches as routine there,
  and expect `validate_data_files_exist` trips when service compaction rewrites files referenced
  by in-flight position deletes.

## 9. Open items (tracked in `task/todo.md` §"ACTIVE (2026-07-01)")

- [ ] Bytecode-verify §5 against Java 1.10.0 + one interop conflict scenario per cell → NORMATIVE.
- [ ] Commit-outcome taxonomy (row R157): the unknown-outcome class LANDED 2026-07-08 (🟡, §8
      updated); the remaining rewrite waits on reconciliation-by-refresh + the credentialed
      real-catalog slice.
- [ ] `TransactionAction` `pub` (pull-based, per the re-anchor) → document custom commit actions.
- [ ] Row-level CDC changelog lands → extend §2/§3 changelog guidance.
