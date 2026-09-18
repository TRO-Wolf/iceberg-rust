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

# Ledger — F-ROWID-ORDER-1: an INSERT commits its data files in ascending partition order

**Ledger id:** `F-ROWID-ORDER-1-2026-09-18`
**Branch:** `fix/f-rowid-order-1` (cut off fork `main`)
**Scope:** deterministic v3 `first_row_id` for INSERT/CTAS commits through the DataFusion execs
**Model:** swe-2-high

| Item | Commit | Subject |
|---|---|---|
| red cells | `1920c8833` | `test: F-ROWID-ORDER-1 — red partition-order first_row_id cells` |
| fix | `01b1a2f4` | `fix: F-ROWID-ORDER-1 — INSERT commits data files in ascending partition order` |
| ledger | this commit | `docs: F-ROWID-ORDER-1 — ledger, mutation proof` |

## Defect

`IcebergCommitExec::execute` in `crates/integrations/datafusion/src/physical_plan/commit.rs`
collects the serialized `DataFile` list from the coalesced multi-partition write stream in
arrival order and hands it to `fast_append` / `overwrite_files` unsorted. Task completion order
is a schedule, not data: on a v3 partitioned table the manifest entry order varies run to run,
and `first_row_id` — assigned in manifest order by the v3 manifest-list writer — varies with
it. RePark on fork main measured six distinct partition→row-id mappings in 12 runs of one
statement.

## Spark's measured answer

Run 23b's rowid measurement (Spark 4.1.2 + Iceberg 1.11.0, Hadoop and in-memory catalogs,
`local[8]`, four shuffle partitions, 12/12 runs): a v3 table
`(id BIGINT, cat STRING, v DOUBLE) PARTITIONED BY (cat)` under `INSERT INTO t SELECT …` over
300 rows with `cat = 'a'/'b'/'c'` by `id % 3` commits files ordered by `first_row_id` as
`[a, 100 rows, 0], [b, 100, 100], [c, 100, 200]` — partition → `min(_row_id)` is
`a: 0, b: 100, c: 200`. `INSERT … VALUES (1,'b'),(2,'a'),(3,'c'),(4,'a'),(5,'b'),(6,'c')`
gives `a: 0, b: 2, c: 4`; CTAS answers like the SELECT shape.

## Ordering rule and tie-break

The collected list is sorted once in the commit exec, before the append/overwrite action sees
it:

1. Primary: ascending partition value, reusing `ascending_partition_order` — the comparator
   `FanoutWriter::close` drains by (null partition values first, partition fields in
   partition-spec order, primitive literals ascending, shorter struct first). It was promoted
   from private to `pub` in `crates/iceberg/src/writer/partitioning/fanout_writer.rs` rather
   than duplicated, so the fork keeps one ascending-partition rule.
2. Tie-break inside one partition value: the write task's output-partition index, which
   `IcebergWriteExec` now emits as a second result column `write_partition_index` (UInt64,
   constant per task batch). The sort is stable (`slice::sort_by`), so files one task produced
   for one partition keep the order the writer produced them in within that input partition,
   and files from different tasks that share a partition value order by input partition index.
   File path was rejected as tie-break: generated paths embed a per-statement UUID and are
   nondeterministic across runs.

Under the standard partitioned INSERT the tie-break is latent — the `_partition` hash
distribution hands each write task a disjoint set of partition values — but it is the whole
order for unpartitioned tables (every file shares the empty partition and `RoundRobinBatch`
distribution) and for any path that reaches the write exec without the hash requirement; it
matches Spark's file ordering there (task index). The one collection feeds both
`InsertOp::Append` (INSERT INTO, and CTAS through the same `insert_into` seam) and
`InsertOp::Overwrite` (INSERT OVERWRITE), so both arms are sorted.

## Red → green → mutation

All cells live in `crates/integrations/datafusion/tests/insert_row_id_order.rs` and run the real
SQL path: `SessionContext` with `target_partitions = 4`, a four-partition `MemTable` source, a
multi-thread tokio runtime, v3 `(id BIGINT, cat STRING, v DOUBLE)` tables, 20 fresh tables per
cell, committed manifest entries read back through `load_manifest` (whose
`apply_manifest_list_context` assigns `first_row_id` on read).

- RED (code without the fix): `insert_select_first_row_id_follows_ascending_partition_order`
  FAILED at run 0 — committed `[c, 100, 0], [a, 100, 100], [b, 100, 200]`;
  `insert_overwrite_first_row_id_follows_ascending_partition_order` FAILED — `[b, 100, 0], …`;
  `insert_select_eight_partitions_first_row_id_ascending` FAILED — eight files scrambled across
  the racing write tasks. `insert_unpartitioned_first_row_id_tiles_the_write` and
  `insert_values_first_row_id_follows_ascending_partition_order` passed under the bug: the
  unpartitioned cell is a control (equal-count files tile `0, 75, 150, 225` under any
  collection order), and the six-row VALUES write's three tasks complete in submission order in
  these runs.
- GREEN (with the fix): all five cells pass, 20/20 runs each —
  `cargo test -p iceberg-datafusion --test insert_row_id_order` → 5 passed.
- MUTATION: `git revert --no-commit` of the fix commit → the same three partitioned cells FAIL
  again on assertion (`[c, 100, 0], [a, 100, 100], [b, 100, 200]` at run 0 for the SELECT cell;
  assertion failures, not compile failures); `git checkout HEAD -- .` restored → 5/5 green. The
  revert was not committed.

## Sibling audit — every exec that commits a collected data-file list

- `IcebergDeleteExec` / `IcebergUpdateExec` COW rewrites (`physical_plan/delete.rs`,
  `physical_plan/update.rs`): each exec drives ONE sequential `StreamingDataFileWriter` inside
  its single `execute(0)` and commits `data_writer.finish()` via `OverwriteFiles` / `RowDelta`.
  There is no coalesced multi-task stream, so no arrival order to leak; the committed order is
  scan/write order, deterministic. Left unchanged.
- MoR DELETE / UPDATE: commits `RowDelta`s of position-delete files from `DvContainerClose`
  (one sequential close); no collected data-file list. Left unchanged.
- No MERGE exec exists in this integration.
- Spark-order relevance: Java's COW row-level commits order added files by write task; the
  fork's single sequential writer is a degenerate one-task version of that order and already
  deterministic. Whether a COW rewrite's file order must additionally be partition-ascending is
  a separate parity question, out of scope for this lane.
- CTAS: reaches `insert_into` → `IcebergWriteExec` → `IcebergCommitExec` — the same collection
  this lane sorts; no separate path.

## Notes

- `IcebergCatalogProvider` resolves a namespace's table list at provider construction, so the
  test fixture creates all 20 fresh tables before registering the provider.
- Two stale doc comments are accepted under the comment ban, evidence routed here:
  `IcebergWriteExec`'s struct and `execute` docs still describe a one-column result batch; the
  result schema is now `(data_files Utf8, write_partition_index UInt64)`.
- `IcebergCommitExec` fails loud (`Internal`) when the `write_partition_index` column is absent
  or mistyped — the same contract style as the `data_files` column.
