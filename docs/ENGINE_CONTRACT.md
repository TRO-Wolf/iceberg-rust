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

> **Status: NORMATIVE for §5 since 2026-07-09** (DRAFT 2026-07-01 → §5 oracle-verified per the DoD
> below). This is the written half of the seam the 2026-06-20 direction re-anchor declared: *"the
> `iceberg` CORE crate is the stable engine-facing contract — the downstream builds its OWN
> DataFusion `TableProvider` over it."* The equivalence proof
> (`crates/iceberg/tests/interop_scan_exec.rs`) is the executable half; this file names everything
> else a DELETE/UPDATE/MERGE engine must know. **DoD (met 2026-07-09):** every §5 cell verified
> against Java 1.10.0 (`SparkWrite` / `SparkCopyOnWriteOperation` / `SparkPositionDeltaWrite`),
> each cell citing its oracle form + a named covering conflict scenario, and — since the
> remediation the same day — every cell carries a CROSS-ENGINE interop scenario (see the §5
> provenance note). Tracked in `task/todo.md` §"ACTIVE UNIT (2026-07-09)" G4.

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
- **Lazy, failure-tolerant catalog registration (2026-07-17).** `IcebergCatalogProvider::try_new`
  lists namespaces and table *names* only — it reads **no** table metadata. Each table's metadata is
  loaded on first reference in `SchemaProvider::table` (async), so a single foreign / unreadable /
  IAM-blocked table cannot brick session construction (startup is O(#tables to *list*), not
  O(#tables to *load*)); a good table queries while an unloadable one coexists; and the unloadable
  table errors loud — **by name** — only when it is referenced. This mirrors Java/Spark lazy-by-name
  resolution; `table_names()` / `table_exist()` report the full listing regardless of loadability.

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
   oracle); the `DeleteFilter`-equivalence proof over a non-identity layout landed 2026-07-09
   (`test_engine_deletefilter_nonidentity_partition_equivalence` — offline: `truncate[10](id)`
   partitions, a transform-scoped position delete + a transform-scoped equality delete, engine
   path == built-in scan, live set pinned exactly) *(2026-07-01 note: previously over-claimed as
   deletefilter coverage, then recorded as owed; now landed)*.

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

**Write-batch timezone coercion (F-A2-3).** The Parquet write funnel (`ParquetWriter::write`, the
sole `FileWriter` impl every data / delete / partitioning writer routes through) normalizes a record
batch whose columns differ from the file (writer) schema ONLY by a UTC-alias timezone string on a
**top-level** timestamp — `Timestamp(_, "UTC")` (as Spark tags Iceberg `timestamptz`) vs the crate's
canonical `Timestamp(_, "+00:00")` (`arrow::schema` `UTC_TIME_ZONE`) — via a metadata-only relabel
(instants bit-identical), mirroring Java Iceberg's coercion of write batches to the file schema. The
engine may therefore hand `"UTC"`-tagged batches directly. The alias set is CLOSED (`"UTC"` /
`"+00:00"` only): a genuinely different timezone (`"+05:00"`), a naive-vs-`timestamptz` mismatch, or a
**nested** alias mismatch (inside a struct/list) is NOT coerced and fails loud — nested normalization
is a deferred fork follow-up.

## 5. Isolation level → validation recipes  ·  **NORMATIVE (oracle-verified 2026-07-09)**

Rust builder methods, verbatim from `transaction/{row_delta,overwrite_files,replace_partitions}.rs`.

**Base for every row-level op (DELETE / UPDATE / MERGE), both modes, both isolation levels:**

- `validate_from_snapshot(scan_snapshot_id)` — Java sets it only when the scan captured a snapshot
  (`scan.snapshotId() != null`; a table that was empty at read time has none:
  `SparkWrite.java` L470-472 / L493-495, `SparkPositionDeltaWrite.java` L246-249). When the
  planner replaced the command's scan with an empty relation (`scan == null` — e.g. a `false`
  condition), Java runs **NO validation at all** (`SparkWrite.java` L446-447,
  `SparkPositionDeltaWrite.java` L236-238): the command never depended on table state.
- `conflict_detection_filter(F)` where `F` = the AND of the scan's **pushed** filter expressions,
  `alwaysTrue()` when none (`SparkWrite.CopyOnWriteOperation.conflictDetectionFilter()` L417-428,
  `SparkPositionDeltaWrite` L284-292) — i.e. the command condition *as pushed to the scan*, not
  necessarily the verbatim SQL predicate.
- **MoR only:** `validate_data_files_exist(referenced_files)` for **all** commands, DELETE
  included (`SparkPositionDeltaWrite.java` L243, unconditional when the scan exists) — the files
  the position deltas reference must not have been compacted away or deleted concurrently.
- *Correction (2026-07-09):* `case_sensitive(...)` is **not** part of the Java recipe — neither
  Spark writer calls `caseSensitive(...)` (grep over both files at the 1.10.0 tag); binding uses
  the operation default (case-sensitive). The Rust builders expose `case_sensitive()` for engines
  whose SQL layer resolves names case-insensitively; calling it is engine policy, outside the
  verified recipe. *(The previous DRAFT base row over-claimed it as base.)*

| Command | Mode | Isolation | Enable (beyond base) | Java 1.10.0 oracle · covering conflict scenario |
|---|---|---|---|---|
| DELETE / UPDATE / MERGE (COW does not branch on command: the `switch` is on isolation level only, `SparkWrite.java` L448-456) | COW (`OverwriteFiles`) | snapshot | `validate_no_conflicting_deletes()` — a concurrent delete file touching a rewritten data file would be silently dropped with it | `SparkWrite.commitWithSnapshotIsolation` L490-509 (L499) · interop `interop_s5_isolation_conflict.rs` (`cow_delete_on_{rewritten,other}`, both directions) + unit `overwrite_files.rs::test_overwrite_rejects_concurrent_delete_for_removed_data_file` (+ legal case `test_overwrite_allows_concurrent_delete_in_other_partition`, NO-OVERRIDE `test_overwrite_rejects_concurrent_delete_using_tx_captured_starting_snapshot`); commit-leg observable in `engine_contract_isolation_recipes.rs` |
| DELETE / UPDATE / MERGE | COW | serializable | above + `validate_no_conflicting_data()` — a concurrent insert matching the condition violates serializability | `SparkWrite.commitWithSerializableIsolation` L467-488 (L476-477) · interop `interop_overwrite_conflict.rs` (C1, both directions) + the serializable-vs-snapshot distinction pin `engine_contract_isolation_recipes.rs::test_s5_cow_serializable_rejects_concurrent_insert_snapshot_isolation_commits` |
| DELETE | MoR (`RowDelta`) | snapshot | *base only* — **correction (2026-07-09): the previous draft prescribed `validate_deleted_files()` here; Java enables it for UPDATE/MERGE ONLY** (`command == UPDATE \|\| command == MERGE`, `SparkPositionDeltaWrite.java` L251-254). A MoR DELETE tolerates a concurrent DELETE-op removal of a referenced file being outside the `{OVERWRITE}` op set | `SparkPositionDeltaWrite.commit` L240-249 · interop `interop_rowdelta_conflict.rs` files-exist axis (`files_exist_{reject,accept}`) + unit `row_delta.rs::test_row_delta_files_exist_skip_deletes_default_excludes_delete_op_snapshot` (half A = the corrected no-`validate_deleted_files` behavior); commit-leg observable in `engine_contract_isolation_recipes.rs` |
| DELETE | MoR | serializable | above + `validate_no_conflicting_data_files()` | `SparkPositionDeltaWrite.commit` L256-258 · interop `interop_rowdelta_conflict.rs` data-conflict axis (`data_conflict_{reject,accept}`) + the distinction pin `engine_contract_isolation_recipes.rs::test_s5_merge_on_read_delete_serializable_rejects_concurrent_insert_snapshot_isolation_commits` |
| UPDATE / MERGE | MoR | snapshot | above (DELETE row) + `validate_deleted_files()` + `validate_no_conflicting_delete_files()` — the op READ rows to produce output; concurrent deletes of those rows conflict | `SparkPositionDeltaWrite.commit` L251-254 · interop `interop_rowdelta_conflict.rs` delete-conflict axis (`delete_conflict_{reject,accept}`) + unit `row_delta.rs::test_row_delta_files_exist_skip_deletes_default_excludes_delete_op_snapshot` (half B = `validate_deleted_files()` widens the op set) |
| UPDATE / MERGE | MoR | serializable | above + `validate_no_conflicting_data_files()` | `SparkPositionDeltaWrite.commit` L256-258 · interop `interop_rowdelta_conflict.rs` data-conflict axis |
| INSERT OVERWRITE (dynamic) | `ReplacePartitions` | snapshot | `validate_no_conflicting_deletes()`. NOT base-shaped: `validate_from_snapshot` only when the engine tracked one (Java guards on `isolationLevel != null && validateFromSnapshotId != null`, L322-324); there is NO `conflict_detection_filter` / `case_sensitive` on `ReplacePartitions` (absent from the interface — `javap` `iceberg-api-1.10.0.jar`) | `SparkWrite.DynamicOverwrite.commit` L318-331 (SNAPSHOT arm L329-330) · interop `interop_s5_isolation_conflict.rs` (`dyn_delete_in_{replaced,other}`, both directions) + unit `replace_partitions.rs::test_replace_partitions_rejects_concurrent_added_delete_in_replaced_partition` (+ legal case `..._allows_concurrent_added_delete_in_other_partition`, NO-OVERRIDE `..._rejects_concurrent_added_delete_using_tx_captured_start`) |
| INSERT OVERWRITE (dynamic) | `ReplacePartitions` | serializable | above + `validate_no_conflicting_data()` | `SparkWrite.DynamicOverwrite.commit` L326-328 · interop `interop_replace_partitions_conflict.rs` (C4, both directions) |
| INSERT OVERWRITE (by filter / static) | `OverwriteFiles` + `overwrite_by_row_filter(expr)` | snapshot | `validate_no_conflicting_deletes()` | `SparkWrite.OverwriteByFilter` (class L346-387; SNAPSHOT arm L374-375) · interop `interop_s5_isolation_conflict.rs` (`byfilter_delete_{matching,excluded}`, both directions) + unit `overwrite_files.rs::test_overwrite_row_filter_rejects_concurrent_added_delete_file_matching_filter` (+ legal/default-filter pins `test_row_filter_is_default_conflict_filter_{matching_add_conflicts,outside_add_does_not_conflict}`) |
| INSERT OVERWRITE (by filter / static) | `OverwriteFiles` | serializable | above + `validate_no_conflicting_data()` — no explicit conflict filter is set; the row filter itself is the default conflict-detection filter | `SparkWrite.OverwriteByFilter.commit` L371-373 · interop `interop_s5_isolation_conflict.rs` (`byfilter_data_{matching,excluded}` — the row-filter-as-default-conflict-filter contract, both directions) + unit `overwrite_files.rs::test_row_filter_is_default_conflict_filter_matching_add_conflicts` |
| INSERT OVERWRITE (dynamic) | `ReplacePartitions` | append-only guard | `validate_append_only()` where the engine requires it (landed 2026-06-17, row R146) — engine policy: the method is core API (`ReplacePartitions.java` L56; `javap` `iceberg-api-1.10.0.jar`) with NO `SparkWrite` caller at 1.10.0 | api interface + `core/BaseReplacePartitions.java` L59 · interop `interop_validate_append_only.rs::test_validate_append_only_mirror` |

Provenance note (oracle form, 2026-07-09): the recipe logic lives in the Java **Spark** writers
(deliberately OUT of `iceberg-core` scope, hence out of the GAP_MATRIX) — it is exactly the
knowledge a downstream engine gets wrong without a written recipe. The Spark jars are not in the
local `~/.m2` mirror, so all `SparkWrite.java` / `SparkPositionDeltaWrite.java` /
`SparkCopyOnWriteOperation.java` citations above are **reference-checkout SOURCE at the
`apache-iceberg-1.10.0` tag** (commit `2114bf6`, `/tmp/iceberg-java-ref`,
`spark/v3.5/spark/src/main/java/org/apache/iceberg/spark/source/`); the core/api validation
surfaces are additionally **`javap`-verified** from `iceberg-api-1.10.0.jar` /
`iceberg-core-1.10.0.jar` (`ReplacePartitions` interface, `BaseRowDelta` validation fields +
methods). Covering-scenario form is named per cell: `interop_*` = cross-engine (the 2026-06-15/16
conflict-validation arc + the 2026-07-09 `interop_s5_isolation_conflict.rs` slice, driven by
`dev/java-interop/run-interop-s5-isolation.sh` — 8 scenarios, both directions, sabotage
fail-closed), `unit` = Rust-side. Since the 2026-07-09 remediation EVERY §5 cell carries a
cross-engine interop scenario (the former unit-only residue — COW/snapshot deletes,
dynamic-overwrite/snapshot, static overwrite-by-filter — is closed). A validation failure is
**non-retryable** by design (Java's
`ValidationException` ⇒ Rust non-retryable `DataInvalid`, message `Found conflicting files that
can contain records matching ...` / `Found conflicting deleted files ...` / `Cannot commit,
missing data files ...`) and propagates out of the retry loop — the engine should surface it, not
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
  connection reset mid-response) surfaces as **`ErrorKind::CommitStateUnknown`** — never
  auto-retried, no cleanup (Java `CommitStateUnknownException` semantics). All four catalogs
  (REST, SQL, Glue, S3 Tables) classify commit-path transport failures sent-vs-unsent.
  **The library now reconciles this in-process** (since 2026-07-09, Java
  `BaseMetastoreOperations.checkCommitStatus`): on an unknown outcome from a snapshot-producing
  commit, `Transaction::commit` re-reads the catalog — bounded by the `commit.status-check.*`
  table properties (`num-retries` default 3 ⇒ 4 read attempts, `min-wait-ms` 1000,
  `max-wait-ms` 60000, `total-timeout-ms` 1800000; Java's names and defaults) — and searches the
  reloaded snapshot history (not just the current pointer, so a commit buried under a concurrent
  writer is still found). **Landed ⇒ `commit` returns `Ok`** with the reloaded table (nothing is
  re-applied); **absent or still-unreadable ⇒ the original `CommitStateUnknown` surfaces**
  (Java's production semantics: an absent-after-refresh commit is NOT declared failed — the
  in-flight request may still land after the check, so re-running on "absent" is the
  double-commit window; do not treat a surfaced unknown as a license to blindly re-run).
  **Residual engine-side reconciliation** is needed only where the library cannot decide:
  **(a)** metadata-only commits (no snapshot added — no client-visible marker; the operation-id
  stamp is useless there too, since a summary property needs a snapshot) and **(b)** an unknown
  that SURVIVES the in-library reconciliation (catalog unreadable for the whole budget) — for
  (b) the prior manual recipe still applies: reload later, scan recent snapshot summaries for a
  unique `engine.operation-id` stamp BEFORE re-running, and never delete the files a failed
  attempt wrote until the commit is confirmed truly absent.
- S3 Tables runs **service-side maintenance** (compaction, snapshot expiry) that commits
  concurrently with the engine — treat `CommitFailed` requirement mismatches as routine there,
  and expect `validate_data_files_exist` trips when service compaction rewrites files referenced
  by in-flight position deletes.


## 8a. CTAS / CREATE OR REPLACE (staged table transaction)

Engine recipe for atomic `CREATE [OR REPLACE] TABLE … AS SELECT` (GAP_MATRIX **R158**):

1. Materialize or stream the SELECT into **data files** against a *staged* table handle from
   `StagedTableTransaction::begin_create` (new table) or `begin_replace` (existing table —
   original catalog pointer stays current).
2. `add_data_files` + `commit(catalog)` — FileIO work completes first; the catalog pointer is
   published in **one** step (`publish_create_table` / `publish_replace_table`).
3. Failure **before** `commit` returns: create → no table; replace → original snapshot current.
4. `MemoryCatalog` implements replace CAS against the base metadata location observed at
   `begin_replace`. Other catalogs default `publish_replace_table` to FeatureUnsupported until
   wired.
5. Bidirectional Java 1.10.0 interop is PROVEN (R158 ✅, 2026-07-16):
   `dev/java-interop/run-interop-staged-txn.sh` (`StagedTxnOracle` ⇄ `tests/interop_staged_txn.rs`)
   drives the engine-agnostic core surface these catalog methods wrap
   (`Transactions.createTableTransaction` / `replaceTableTransaction` over
   `ops.current().buildReplacement(...)`) and asserts, in BOTH directions, create single-publish +
   row content, the replace invariant set (uuid retained, history retained, `metadata_log` grows,
   `main` reset to the latest replace's rows only, location stable, format version preserved,
   `last_column_id` monotonic), a V1 table surviving a replace as V1 on both sides, and the
   `format-version` directive contract — behind a structural cross-check and a fail-closed sabotage
   battery.

**Atomicity guarantee (create-publish):** create-publish is all-or-nothing, even when the reload
fails. If publishing a staged create cannot reload the staged metadata (e.g. it was written through
a `FileIO` the catalog cannot read), the catalog is left with **no** pointer for that identifier —
`table_exists` stays false, so a retry / `CREATE TABLE IF NOT EXISTS` re-create of the same
identifier succeeds. The `MemoryCatalog` default reads the metadata **before** inserting the pointer
(`register_table`); any catalog overriding `publish_create_table` / `register_table` MUST preserve
that ordering.

**Replace contract (build-on-existing):** `begin_replace` builds the replacement metadata ON TOP OF
the existing table's metadata (Java `TableMetadata.buildReplacement`), not from scratch. It
**retains** the table UUID, the full snapshot history, and the metadata log (appended-to, never
truncated); it **resets** the `main` branch ref (no current snapshot) and applies the
`TableCreation`'s schema / partition spec / sort order / properties / location as the new current
ones. The replace-schema field-ids are taken **from the caller as provided**; `last_column_id`
advances monotonically (`max` of the existing value and the caller's highest id, never reduced).
This diverges from Java `TypeUtil.assignFreshIds`, which reassigns fresh ids by **name-matching**
the replacement schema against the base schema — a caller supplying field-ids misaligned with the
base schema's names diverges from Java (**named residue**: a base-aware fresh-id helper is the
follow-up; not corruption — per-snapshot schema binding keeps prior history readable). The format
version is **preserved** across a replace unless the `TableCreation`'s properties carry an explicit
`format-version` directive: absent ⇒ keep the existing version; a higher value ⇒ upgrade; equal ⇒
no-op; a lower value ⇒ hard `DataInvalid` error (never downgraded); unparsable ⇒ hard `DataInvalid`.
`creation.format_version` is **ignored** on the replace path (indistinguishable from the
`TableCreation::builder()` V2 default), and the `format-version` key is consumed as a directive and
not persisted (Java `persistedProperties` filters reserved properties out). Retaining the history
keeps time-travel raw material intact, while `main` exposes only the latest replace's data.

**Location guarantee (replace):** a published replace never relocates the table — it keeps the
existing root `location()` (or the caller-provided `creation.location`), so repeated CREATE OR
REPLACE leaves the location identical every time (no `__staged_replace` suffix drift, no compounding
`…__staged_replace__staged_replace…`). Future writers therefore keep landing under the stable path.
Data files already written under any other path stay readable (Iceberg manifests carry absolute
paths); a replace does **not** move data.

Do **not** drop-then-create-then-insert for OR REPLACE: that loses the original table if insert
fails after drop.

## 9. Open items (tracked in `task/todo.md` §"ACTIVE (2026-07-01)")

- [x] Verify §5 against Java 1.10.0 + a covering conflict scenario per cell → **NORMATIVE
      (2026-07-09)**. Oracle form + per-cell scenario names in the §5 provenance note. The
      unit-only residue (COW/snapshot deletes-validation, dynamic-overwrite/snapshot, static
      overwrite-by-filter) was CLOSED the same day by `interop_s5_isolation_conflict.rs` +
      `dev/java-interop/run-interop-s5-isolation.sh` (8 scenarios, both directions, sabotage
      fail-closed) — every §5 cell now has a cross-engine scenario.
- [x] **Strict-metrics NaN divergence (found by the §5 interop, 2026-07-09) → FIXED
      (2026-07-10):** `StrictMetricsEvaluator::may_contain_nan` now treats an ABSENT nan count
      as *CANNOT contain NaN*, matching Java `canContainNaNs`
      (`api/.../expressions/StrictMetricsEvaluator.java` L483-486 @ 1.10.0, bytecode-verified;
      "nan counts might be null for early version writers"), and the Java `gtEq`
      NaN-lower-bound guard (L285-291) that the loosening makes reachable was ported in the
      same change. Strict inequalities/eq/in can now prove a full match on columns without nan
      counts (every non-float/double column), so `overwrite_by_row_filter` /
      `DeleteFiles`-by-filter no longer need partition-scoped filters. Unit + mutation pinned;
      the cross-engine metrics-decided full-match sweep is the deferred interop slice.
- [ ] Commit-outcome taxonomy (row R157): the unknown-outcome class LANDED 2026-07-08 and
      **reconciliation-by-refresh LANDED 2026-07-09** (§8 updated — the library now decides
      landed/absent/still-unknown in-process); the row stays 🟡 only for the credentialed
      real-catalog slice.
- [ ] `TransactionAction` `pub` (pull-based, per the re-anchor) → document custom commit actions.
- [ ] Row-level CDC changelog lands → extend §2/§3 changelog guidance.
