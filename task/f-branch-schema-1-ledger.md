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

# F-BRANCH-SCHEMA-1 — a branch read projects the table's current schema; a tag or snapshot-id read keeps the snapshot's

## Finding (RePark IPI-07, measured by run 24c)

Spark 4.1.2 + Iceberg 1.11 resolves a BRANCH reference (including `main`) to its referenced
snapshot but exposes the table's CURRENT schema (Java `SnapshotUtil.schemaFor(table, branch)`:
branch → `table.schema()`; tag or snapshot id → the snapshot's schema). Columns added after the
branch snapshot read as NULL, dropped columns disappear, renamed columns appear under the current
name, promoted types surface at the current type, and filters on a post-snapshot column bind and
return rows (`WHERE z IS NULL`). A tag or an explicit snapshot id keeps the referenced snapshot's
schema. Unknown refs must fail with a typed error naming the ref.

The fork's `IcebergStaticTableProvider` had only `try_new_from_table` (current snapshot, current
schema) and `try_new_from_table_snapshot` (always the snapshot schema); the DataFusion scan bound
advertised columns and pushed filters through the snapshot schema only. The core
`TableScanBuilder` already carried `project_current_schema()`.

## Fix design

- `Table::snapshot_ref(ref_name) -> Option<&SnapshotReference>` (new public accessor in
  `crates/iceberg/src/table.rs`): the crate-internal `refs` map is `pub(crate)`, and DataFusion
  needs the ref KIND (branch vs tag), which `snapshot_for_ref` does not expose — it returns the
  snapshot, not the reference. `TableMetadata` sits exactly at its file-size ceiling
  (`scripts/check_rust_file_size.py`: 4552 = current), so the accessor lives on `Table`.
- `IcebergStaticTableProvider::try_new_from_table_ref(table, ref_name)`: resolves the ref;
  unknown name → `ErrorKind::DataInvalid` carrying the ref name (same wording as
  `resolve_scan_snapshot_id`). A branch → `snapshot_id = ref.snapshot_id`,
  `project_current_schema = true`, advertised schema = `current_schema`. A tag → same snapshot
  pin, flag false, advertised schema = the snapshot's schema.
- `project_current_schema: bool` threads through `IcebergTableScan::new`/`plan`,
  `resolve_bindings`, `scan_predicates`, `get_batch_stream`, and `build_table_scan`. When set:
  `resolve_bindings` names advertised field ids through the CURRENT schema (so `select` resolves
  current names against the current-schema builder — RENAME emits the new name, ADD's `z` is
  selected and the RecordBatchTransformer null-fills it where a file lacks the field);
  `scan_predicates` binds converted filters against the current schema (so `WHERE z IS NULL`
  survives `predicate_binds_soundly` and is pushed; the residual evaluates a file-missing column
  as all-null, Java semantics); the builder gets `.project_current_schema()` so the core scan
  reads the pinned snapshot's files under the current schema.
- `IcebergTableProvider::scan` (the writable provider) computes the flag from
  `commit_branch`: `Some(name)` and `table.snapshot_ref(name).is_some_and(is_branch)` → flag on.
  `resolve_scan_snapshot_id` already fails the read leg on a missing ref before the flag is
  computed.
- `get_batch_stream` moved from `scan.rs` to `scan_knobs.rs` next to `build_table_scan`: `scan.rs`
  is exactly at its 1498-line ceiling and the flag threading adds lines; the function is a thin
  `build_table_scan + to_arrow` wrapper. Re-exported through `scan.rs` so
  `crate::physical_plan::scan::get_batch_stream` is unchanged.

## Proposition ledger

| # | Proposition | Evidence | Status |
|---|---|---|---|
| P1 | branch ref → ref's snapshot + table current schema | `bs_add_ident_v{2,3}`, `bs_main_ident_v{2,3}` | verified |
| P2 | `VERSION AS OF 'b0'` shape → same branch rule | `bs_add_version_v{2,3}` | verified |
| P3 | tag ref → tagged snapshot's schema | `bs_add_tag_v{2,3}` | verified |
| P4 | snapshot-id read → snapshot schema (unchanged) | `bs_add_snapid_v{2,3}` | verified |
| P5 | dropped column absent on branch read | `bs_drop_v{2,3}` | verified |
| P6 | renamed column appears under current name | `bs_rename_v{2,3}` | verified |
| P7 | promoted type surfaces at current type | `bs_widen_v{2,3}` | verified |
| P8 | branch write before ADD still projects current schema | `bs_write_then_add_v{2,3}` | verified |
| P9 | branch write after ADD carries the new column | `bs_add_then_write_branch_v{2,3}` | verified |
| P10 | `WHERE z IS NULL` on the new column binds and is pushed | `bs_where_newcol_v{2,3}` | verified |
| P11 | unknown ref → typed error naming the ref | `bs_unknown_ref_v{2,3}` | verified |
| P12 | position delete committed on a branch applies to the branch read | `bs_delete_branch_v{2,3}` | verified |
| P13 | partitioned table branch read projects current schema | `bs_partitioned_branch_v{2,3}` | verified |
| P14 | `with_commit_branch` read follows the same rule | `bs_writable_provider_branch_v{2,3}` | verified |

## Red evidence

The suite (`crates/integrations/datafusion/src/table/branch_schema_tests.rs`) is written against
the only pre-unit read path: `provider_for_ref` resolves the ref's snapshot and calls
`try_new_from_table_snapshot`, i.e. every ref read projects the snapshot schema. Assertions are
the Spark cells' columns and rows. `cargo test -p iceberg-datafusion --lib bs_`:

- **11 passed / 16 failed.** Passed: the tag cells (`bs_add_tag_v{2,3}`), the snapshot-id cells
  (`bs_add_snapid_v{2,3}`), and `bs_add_then_write_branch_v{2,3}` — the branch write after the ADD
  lands a snapshot whose schema IS the current one, so the snapshot-schema path is incidentally
  right there. Those are the control pins; they stay green under the fix.
- Failed with the wrong columns, exactly the defect: `bs_add_ident`, `bs_add_version`,
  `bs_main_ident`, `bs_write_then_add` (v2+v3 each) report
  `["id:Int64","data:Utf8","cat:Utf8"]` where the cell expects `+ "z:Int32"`; `bs_drop` keeps the
  dropped `data`; `bs_rename` shows `data` instead of `payload`; `bs_widen` reports `id:Int32`
  where the cell expects `id:Int64`; `bs_where_newcol` fails at planning with
  `column 'z' not found` because the advertised schema lacks the post-snapshot column.

## Mutation evidence

Each leg was applied, the filtered suite run, then the mutation reverted and the suite re-run
green (`30 passed / 0 failed`).

- **branch → snapshot schema**: `try_new_from_table_ref` mutated so `is_branch` reads `false`
  (branch resolves its snapshot but advertises the snapshot schema, flag off). Result: the 20
  branch projections red — `bs_add_ident`, `bs_add_version`, `bs_main_ident`, `bs_drop`,
  `bs_rename`, `bs_widen`, `bs_write_then_add`, `bs_delete_branch`, `bs_partitioned_branch`,
  `bs_where_newcol` (v2+v3 each); tags, snapshot ids, unknown refs, and
  `bs_add_then_write_branch` stay green (that branch snapshot's own schema IS the current one).
  10 passed / 20 failed.
- **tag → current schema**: `is_branch` inverted (`!is_branch()`): a tag now advertises and
  projects the current schema. `bs_add_tag_v{2,3}` red on the extra `z:Int32` column (the leg
  also flips branch handling, so the branch pins redden too). 8 passed / 22 failed.
- **drop the scan-binding change**: `resolve_bindings` mutated to ignore
  `project_current_schema` and always bind through the snapshot schema.
  `bs_where_newcol_v{2,3}` red — advertised `z` has no field id in the snapshot schema, so the
  filter cannot bind and the pushed predicate the pin inspects is gone. `bs_rename_v{2,3}`
  also red: the binding resolves `payload` to the snapshot name `data`, which the
  current-schema core `select` then cannot resolve. 26 passed / 4 failed.

## Gates

- `cargo test -p iceberg-datafusion --lib branch_schema` — 30 passed / 0 failed.
- `cargo test -p iceberg-datafusion --lib "physical_plan::scan"` — 29 passed / 0 failed.
- `cargo test -p iceberg-datafusion --lib table::` — 71 passed / 0 failed.
- `cargo test -p iceberg-datafusion --lib physical_plan` — 243 passed / 0 failed (pre-split run).
- `cargo test -p iceberg --lib snapshot` — 249 passed / 0 failed.
- `cargo fmt --all -- --check` — clean.
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings` — clean.
- `python3 scripts/check_rust_file_size.py` — 577 files clean (the `scan.rs` legacy ceiling row
  removed: the file is 462 lines after its test module moved to `scan_tests.rs` +
  `scan_pin_tests.rs` siblings; moved test comments were deleted in the move per the comment
  ban).
- `typos` — clean.
- `python3 /tmp/oc-worker/_lib/comment_ban.py <clone> origin/main` — `comment-ban hits=0`
  after every commit.
