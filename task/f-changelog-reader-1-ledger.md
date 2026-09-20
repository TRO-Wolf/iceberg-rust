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

# F-CHANGELOG-READER-1 — an Arrow reader for `ChangelogScanTask`

## Finding (RePark parity inventory IPI-22, run 26c)

Spark answers `SELECT id, _change_type, _change_ordinal FROM ns.t.changes` and
`CALL sys.create_changelog_view(...)`; RePark refuses both. The planner half is already here
(`scan/incremental.rs`, GAP_MATRIX R123), but **nothing reads a `ChangelogScanTask`**: no code in
`crates/iceberg/src/arrow/` mentions one, and the three reserved columns defined in
`metadata_columns.rs` (`_change_type` `i32::MAX-104`, `_change_ordinal` `-105`,
`_commit_snapshot_id` `-106`) were never emitted by anything. An engine had to interpret
`file_scan_task()` plus the two delete lists itself.

Spark's own split is the same one this unit draws: `BaseIncrementalChangelogScan` plans
INSERT/DELETE tasks, `ChangelogRowReader` reads them into rows carrying the change columns, and
`CreateChangelogViewProcedure` does the carryover / update-image / net-change work in the ENGINE.
This unit ports the middle piece only.

## What landed

| Clause | Statement | Verdict | Evidence |
|---|---|---|---|
| C-001 | A `ChangelogScanTaskStream` reads into `RecordBatch`es carrying the table's columns plus `_change_type` (Utf8), `_change_ordinal` (Int32) and `_commit_snapshot_id` (Int64). | PROVEN | `changelog_reader_tests::changelog_rows_carry_the_window_snapshot_of_their_own_ordinal` |
| C-002 | Every row of a task carries **that task's** commit snapshot id — the window snapshot at its own `_change_ordinal`, never the scan's `to` id. | PROVEN | same test: the ordinal-0 rows carry `s1` while the scan's `to` is `s2`, and the test asserts `s1 != s2`. A reader filling the column with the scan's `to` id (or `0`) fails it. |
| C-003 | A `DeletedDataFile` task reads as `DELETE` rows of the snapshot that removed the file, beside the `INSERT` rows of the snapshot that added it. | PROVEN | `a_deleted_data_file_reads_as_delete_rows_of_its_commit_snapshot` |
| C-004 | The three appended fields carry the RESERVED field ids already defined in `metadata_columns.rs` — no second definition. | PROVEN | `the_three_reserved_columns_keep_their_reserved_field_ids` (asserts the `PARQUET:field_id` metadata equals the three `RESERVED_FIELD_ID_*` constants) |
| C-005 | The default (Java-parity) mode is unchanged: the reader never enables `with_row_level_deletes`, so a range holding delete manifests still refuses with the planner's `Delete files are currently not supported in changelog scans` — the same string Iceberg 1.11.0's `BaseIncrementalChangelogScan` carries. | PROVEN | untouched planner guard (`scan/incremental.rs`), pinned by the existing `test_changelog_rejects_range_with_delete_manifest`; the reader adds no flag |

## Design notes

- `ChangelogReader` wraps an `ArrowReader` rather than duplicating the parquet path: each task's
  `file_scan_task` is read through the existing reader (so delete files, name mapping, schema
  evolution and the footer cache all behave exactly as on a normal scan), and the three constant
  columns are appended per batch.
- `changelog_arrow_schema` / `changelog_arrow_fields` are public so an engine declares the same
  schema its provider will serve, instead of hand-building the three fields and risking a
  divergent field id.
- The module carries `#[allow(missing_docs)]` at its declaration instead of doc comments: the
  owner's comment ban applies to this fork too, and the crate's `#![deny(missing_docs)]` is
  satisfied by the allow.

## Not in this unit

Row-level changelog (`ChangelogTaskKind::DeletedRows`, applying `added_deletes` as a selector)
stays behind the existing opt-in flag and is NOT read here: Iceberg 1.11.0's `ChangelogRowReader`
implements `openAddedRowsScanTask` / `openDeletedDataFileScanTask` only, so exceeding it is an
owner ruling, not this unit's. `UPDATE_BEFORE` / `UPDATE_AFTER` pairing stays engine-side, as the
planner's own documentation says.

## Mutations (round 2, critic remediation 2026-09-20)

| # | Mutation | Expected | Observed |
|---|---|---|---|
| M-001 (V-001) | Delete the `DeletedRows` refusal at the top of `read_one_task` | `changelog_reader_refuses_deleted_rows_tasks` goes RED | RED, `cargo test -p iceberg --lib changelog_reader_refuses_deleted_rows` exit 101; restored, green again |
