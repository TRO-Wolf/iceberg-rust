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

# F-6b evidence ledger — DataFusion commit target (row R168)

Charter: provider-level `with_commit_branch` on `IcebergTableProvider`, handed to
`SnapshotUpdate::to_branch` at every snapshot-producing DataFusion DML site. WAP is out of
scope. Base `33be9a0f4` (F-6 `to_branch`, PR #244).

## 1. Measured `to_branch` semantics (read, not assumed)

Source: `crates/iceberg/src/transaction/to_branch.rs` tests +
[f6-branch-commit-target-ledger.md](f6-branch-commit-target-ledger.md).

| Case | Measured |
|---|---|
| Existing branch | named ref advances; `main` (`current_snapshot_id`) is byte-stable |
| Missing branch | created at commit; parent is current snapshot; `main` does not move |
| Empty table | `current_snapshot_id` stays `None`; the named ref is created |
| Tag | commit fails `ErrorKind::DataInvalid` with Java's tag message |
| Sibling refs | other branches/tags are byte-stable |
| Retry | pending commit parents off the named branch's new head, not `main` |
| Validate start | `starting_snapshot_for(target_ref())`: named-ref head at txn start, else main |

## 2. Enumerated DataFusion commit sites

Quantified clause: every snapshot-producing action under
`crates/integrations/datafusion/` must receive the provider's commit target.

Production sites (`rg '\.(fast_append|overwrite_files|row_delta|merge_append|replace_partitions|rewrite_files|delete_files)\(' crates/integrations/datafusion/src`):

| # | File | Action | DML |
|---|---|---|---|
| 1 | `physical_plan/commit.rs` | `fast_append` | INSERT INTO |
| 2 | `physical_plan/commit.rs` | `overwrite_files` | INSERT OVERWRITE |
| 3 | `physical_plan/delete.rs` | `row_delta` | merge-on-read DELETE |
| 4 | `physical_plan/delete.rs` | `overwrite_files` | copy-on-write DELETE |
| 5 | `physical_plan/delete.rs` | `row_delta` | merge-on-read UPDATE |
| 6 | `physical_plan/delete.rs` | `overwrite_files` | copy-on-write UPDATE |

Not sites: `metadata_table.rs` `delete_files()` (metadata-table scan, not a transaction);
`commit.rs` `append_files_direct` (test helper); schema-evolution `Transaction` in table tests.

Default `write.delete.mode` / `write.update.mode` is copy-on-write. Merge-on-read is
`write.*.mode = merge-on-read`.

## 3. API

`IcebergTableProvider::with_commit_branch(name)` stores `Option<String>` on the provider.
`None` (default) does not call `to_branch` — commits land on `main`, byte-identical to
`33be9a0f4`. `Some` is passed through `IcebergCommitExec` / `IcebergDeleteExec` /
`IcebergUpdateExec` into `to_branch`.

`refreshed()` copies the target. Catalog-level / session / per-statement overrides are
out of scope.

When a target is set, DataFusion does not call `validate_from_snapshot(current_snapshot_id)`
(that id is `main`). Conflict validation then uses `starting_snapshot_for(target_ref())`,
matching F-6. The default path still arms `validate_from_snapshot` on current.

Scans still use the table handle's current snapshot (`main`). This unit is a commit
target, not `table.asBranch`.

## 4. Clause pins

| Clause | Pin (`crates/integrations/datafusion/tests/commit_branch.rs`) |
|---|---|
| Today / default INSERT lands on `main` | `insert_without_target_advances_main` |
| Named branch INSERT advances branch, `main` unmoved | `insert_with_commit_branch_does_not_move_main` |
| Missing ref is created | `insert_with_commit_branch_creates_missing_branch` |
| Tag rejected | `insert_with_commit_branch_rejects_tag` |
| INSERT OVERWRITE site | `insert_overwrite_with_commit_branch_does_not_move_main` |
| CoW DELETE site | `copy_on_write_delete_with_commit_branch_does_not_move_main` |
| MoR DELETE site | `merge_on_read_delete_with_commit_branch_does_not_move_main` |
| CoW UPDATE site | `copy_on_write_update_with_commit_branch_does_not_move_main` |
| MoR UPDATE site | `merge_on_read_update_with_commit_branch_does_not_move_main` |

## 5. Named residue

- WAP / `stage_only` + `to_branch` (F-6 residue).
- Scan / prune / `validate_from_snapshot` still keyed off current snapshot when no
  target is set; with a target, validate-from uses the named ref, but the DML scan
  still reads `main`.
- Catalog / session / SQL `AS OF` / per-statement branch.
- Java interop (row R168 stays 🟡).
- `IcebergTableProviderFactory` and `IcebergCatalogProvider` do not set a target.

## 6. Gates

| Command | Exit |
|---|---|
| `cargo test -p iceberg-datafusion --test commit_branch --locked` | 0 (9 passed) |
| `cargo test -p iceberg -p iceberg-datafusion --locked` | 0 |
| `make check` | 0 |
| Docker / `make test` | excused (Docker unavailable) |

## 6a. Self Logic Review

```yaml
SELF_LOGIC_REVIEW:
  id: SLR-f6b-2
  agent: Actor
  action: Hand IcebergTableProvider commit_branch to to_branch at all six DataFusion DML sites
  charter_trace: F-6b / row R168 provider-level commit target
  preconditions:
    - to_branch exists on FastAppend, OverwriteFiles, RowDelta: SATISFIED (to_branch.rs)
    - Six production sites enumerated: SATISFIED (ledger §2)
  expected_output: with_commit_branch setter; default path does not call to_branch
  success_condition: named branch advances and main is unmoved on insert, overwrite, CoW/MoR delete, CoW/MoR update
  step_risks:
    - Missing one commit site: HANDLED (six pins)
    - Default path calls to_branch("main"): HANDLED (None skips to_branch)
    - validate_from_snapshot(main) overrides named-ref OCC: HANDLED (skipped when target set)
  tripwire_scan: CLEAN
  uncertainty: NONE
  verdict: PROCEED
  escalation: "—"
```

## 7. Self Logic Review

See session SLR-f6b-1 / SLR-f6b-2.
