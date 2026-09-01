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

# F-6c evidence ledger — DataFusion branch-following reads (row R168)

Charter: when `IcebergTableProvider::with_commit_branch` is set, every scan this
provider plans — `TableProvider::scan` and the scan legs inside UPDATE / DELETE —
resolves its snapshot from the named branch head. Commit targeting is unchanged
from F-6b. Base `33c20da31` (F-6b in `main` as #245).

## 1. Measured edges

| Case | Measured |
|---|---|
| Named-branch SELECT | returns the branch head's rows, not `main` |
| Named-branch DELETE | deletes per the branch head; `main` snapshot-id, snapshot-log, and rows stay byte-unmoved |
| Named-branch INSERT SELECT | source scan reads the branch head; commit still `to_branch` |
| Default (no setter) | SELECT and DML stay on `main` (F-6b default pins kept) |
| Missing ref, SELECT | `DataInvalid` `snapshot ref '{name}' not found` (Java `TableScan.useRef` / core `use_ref`) |
| Missing ref, DELETE / UPDATE | same error on the read leg; the ref is **not** created |
| Missing ref, INSERT VALUES | still created at commit (F-6b pin; no target scan) |
| Tag as commit target | INSERT still rejected at commit with Java's tag message (F-6b) |
| Older schema-id on the branch | advertised schema stays current (catalog-backed contract). `IcebergTableScan` binds by field id and null-fills columns the snapshot lacks — the same rule as SELECT after `ADD COLUMN` (`test_scan_batches_match_advertised_schema_after_add_column`). Not `IcebergStaticTableProvider`, which re-advertises the snapshot schema because it is constructed from that snapshot. |
| OCC | `validate_from_snapshot` is the **scanned** snapshot. Named-ref → branch head. Unset → current. Missing-ref INSERT OVERWRITE passes `None` so F-6 `starting_snapshot_for` still creates the branch. Diverged serializable INSERT OVERWRITE pin kept. |

DML projection is still name-select of advertised columns. A branch snapshot that
lacks an advertised name fails at `TableScanBuilder::build`, same as DELETE after
`ADD COLUMN` on `main` today. Not a third schema rule.

## 2. Scan sites

| # | Site | How the snapshot is chosen |
|---|---|---|
| 1 | `IcebergTableProvider::scan` | `resolve_scan_snapshot_id` → `IcebergTableScan::plan` |
| 2 | MoR DELETE | `resolve_scan_snapshot_id` → `table.scan().snapshot_id` |
| 3 | CoW DELETE | `resolve_scan_snapshot_id` → `cow_scan_stream` + `resolve_affected_data_files` |
| 4 | MoR UPDATE | same as MoR DELETE |
| 5 | CoW UPDATE | same as CoW DELETE |
| 6 | INSERT OVERWRITE validate-from | `optional_ref_snapshot_id` (missing ref → `None`, no error) |

Live-file walks (`live_data_file_partitions`, `live_delete_vectors_by_data_file`,
`resolve_affected_data_files`) take the scanned snapshot, not `current_snapshot()`.

## 3. Clause pins

| Clause | Pin (`crates/integrations/datafusion/tests/commit_branch.rs`) |
|---|---|
| Diverged SELECT returns B not A | `scan_with_commit_branch_returns_branch_rows_not_main` |
| Diverged DELETE follows B, main untouched | `delete_with_commit_branch_follows_branch_rows_and_leaves_main_untouched` |
| INSERT SELECT reads B | `insert_select_with_commit_branch_reads_the_named_branch` |
| Default SELECT stays on main | `scan_with_commit_branch_returns_branch_rows_not_main` (second assert) |
| Default DML stays on main | F-6b `insert_without_target_advances_main` and the five `*_does_not_move_main` site pins |
| Missing-ref SELECT errors | `scan_missing_branch_errors_loudly` |
| Missing-ref DELETE errors, does not create | `delete_missing_branch_errors_on_the_read_leg` |
| Missing-ref INSERT creates | `insert_with_commit_branch_creates_missing_branch` |
| Older schema-id null-fills | `scan_older_branch_schema_null_fills_columns_added_on_main` |
| OCC scan == validate-from | `maybe_validate_from_snapshot_applies_the_scan_snapshot_when_set` + `insert_overwrite_on_diverged_branch_does_not_treat_branch_files_as_concurrent` |

## 4. Named residue

- WAP / `stage_only` + `to_branch` (F-6 residue).
- Catalog / session / SQL `AS OF` / per-statement branch.
- Java interop (row R168 stays 🟡).
- `IcebergTableProviderFactory` and `IcebergCatalogProvider` do not set a target.
- DML name-select vs IcebergTableScan field-id bind on a renamed column.

## 5. Gates

| Command | Exit |
|---|---|
| `cargo test -p iceberg-datafusion --test commit_branch --locked` | 0 (16 passed) |
| `cargo test -p iceberg -p iceberg-datafusion --locked` | 0 |
| `make check` | 0 |
| Docker / `make test` | excused (Docker unavailable) |

## 6. Self Logic Review

```yaml
SELF_LOGIC_REVIEW:
  id: SLR-f6c-1
  agent: Actor
  action: Branch-following reads on IcebergTableProvider when with_commit_branch is set
  charter_trace: F-6c / row R168
  preconditions:
    - F-6b with_commit_branch exists and commits via to_branch: SATISFIED
    - TableScanBuilder::use_ref errors on a missing ref: SATISFIED (scan/mod.rs)
    - IcebergTableScan binds advertised schema to a snapshot by field id: SATISFIED
  expected_output: scans and DML read legs resolve snapshot_for_ref; missing ref is loud; default path unchanged
  success_condition: diverged fixture SELECT/DELETE/INSERT-SELECT follow the branch; main log and rows unmoved
  step_risks:
    - CoW live-file walk still uses current_snapshot (main): HANDLED (snapshot_for_scan)
    - validate_from_snapshot(main) on a diverged branch: HANDLED (pass scanned snapshot)
    - Missing-ref DML creating the branch: HANDLED (read-leg error pin)
  tripwire_scan: CLEAN
  uncertainty: NONE
  verdict: PROCEED
  escalation: "—"
```
