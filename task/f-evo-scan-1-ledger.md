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

# F-EVO-SCAN-1 — unpinned scans and the DataFusion UPDATE/DELETE execs after ADD / RENAME COLUMN with no write since

**Date:** 2026-09-17. **Branch:** `fix/evo-scan-1` (stacked on `fix/ice-promote-read-1` `e8db2ac0`).
**Model:** muse-spark-1.3-contributor.
**Consumer:** RePark ICE-EVO-DML-1 residual red (33 plain-UPDATE cells).

## The defect

After `ALTER TABLE … ADD COLUMN extra` or `RENAME COLUMN w TO v` (or a name swap) with no
data write since, the current schema is newer than the current snapshot's schema.
`TableScanBuilder::build` (`crates/iceberg/src/scan/mod.rs:447`) binds the selected CURRENT
column names against the snapshot's schema, so an added or renamed-to name fails loud
(`DataInvalid => Column extra not found in table`), and a swapped name binds the other
field's id and reads the other column's values silent.

Java's contract: unpinned `table.newScan()` binds `table.schema()` (current); only
`useSnapshot(id)` binds `SnapshotUtil.schemaFor(table, id)`. Spark's DML scan does
`useSnapshot(id)…project(expectedSchema)` — the snapshot's files under the current
projection by field id. The fork's own compaction already uses that recipe (F-RDF-EVO-1,
`maintenance/rewrite_data_files_write.rs:104-116`: re-point `FileScanTask.schema` and
`project_field_ids` at the current schema; the Arrow reader NULL-fills added columns and
reads renamed columns by field id). RePark fixed its own DML scans RePark-side
(`current_schema_scan.rs`); the fork's DataFusion UPDATE/DELETE execs
(`mor_scan_stream`, `cow_scan_stream`) still pin the snapshot and select current names.

## Decisions

- **D-1** Unpinned scan (no `snapshot_id`, no `snapshot_ref`) binds the current schema.
  Pinned scan keeps the snapshot schema (time travel unchanged).
- **D-2** Pinned scans gain an opt-in projection of the current schema over the pinned
  snapshot's files (Spark `project(expectedSchema)`), used by the two DML scan seams.
  The SELECT path (`IcebergTableScan`) keeps translating names snapshot-side and is untouched.
- **D-3** `scan/mod.rs` sits exactly at its size ceiling, so the fix removes the
  name-validation loop that the field-id loop below already performs, and the ceiling
  follows the file down.

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Status |
|---|---|---|---|
| C-001 | Unpinned `table.scan().select(current names)` after ADD COLUMN with no write since reads NULL for the added column. | `scan::evo_scan_tests` add-column pin; red on base. | OPEN |
| C-002 | Same after RENAME COLUMN: the renamed column's values read by field id. | rename pin; red on base. | OPEN |
| C-003 | Same after a name swap: each swapped name reads its own field. | swap pin; red on base. | OPEN |
| C-004 | Pinned scan keeps binding the snapshot schema (time travel). | existing time-travel pins stay green. | OPEN |
| C-005 | SQL `UPDATE … WHERE id = 1` and `DELETE … WHERE id = 1` after each of the three evolutions, two data files, CoW and MoR, assert post-statement rows. | `tests/evo_schema_dml.rs` (12 pins); red on base. | OPEN |
| C-006 | Gates per brief step 5. | Command → result below. | OPEN |

## Red evidence

Pending.

## Execution evidence

Pending.

## Mutation evidence

Pending.

## Gates

Pending.

## Open questions

None.
