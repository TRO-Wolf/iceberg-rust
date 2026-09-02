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

# map.md — crates/integrations/datafusion/tests/

## Purpose

Integration tests for `iceberg-datafusion`. They register an `IcebergTableProvider` against a
`MemoryCatalog` and run SQL (or the provider trait) through DataFusion.

## Contents

| File | What it pins |
|---|---|
| `commit_branch.rs` | `IcebergTableProvider::with_commit_branch` scopes scans and commits to the named branch (row R168). Diverged SELECT/DELETE/INSERT-SELECT follow the branch head; CoW/MoR UPDATE and CoW/MoR DELETE each have a branch-only (`id = 10`) pin so a main-read is red; missing-ref SELECT/DELETE error; INSERT VALUES still creates the ref; advertised schema stays current and field-id bind null-fills; default path stays on `main` |
| `integration_datafusion_test.rs` | Core provider SQL / scan / DML |
| `cow_memory_bound.rs` | Copy-on-write memory bound |
| `h7_p1_dml_prune.rs` | DML file prune |
| `interop_dv_sql.rs` / `interop_partitioned_dml.rs` | Interop DML |
| `interop_branch_dml.rs` | Branch read/commit Java interop (row R168 / PR-6A). Offline: Rust reproduces a Java diverged branch table; missing-ref SELECT/UPDATE; INSERT creates (ids, parent, main vs branch file sets); tag refuses; V3 MoR DELETE uses branch live files. GEN of 6 `rust_*` tables pins post-DML main vs branch file sets and writes `expected_*_files.txt`. Env `ICEBERG_INTEROP_BRANCH_DIR` / `_GEN_DIR` via `dev/java-interop/run-interop-branch-dml.sh` |
| `lazy_table_resolution_test.rs` | Catalog-backed lazy resolve |
| `partitioned_insert_select_test.rs` | Partitioned INSERT SELECT |
| `row_lineage_cow.rs` | V3 row lineage on CoW DML; Spark sequences at the fork's single-file layout (F-rp3-c7, row R166) |
| `row_lineage_mor.rs` | V3 merge-on-read UPDATE lineage, sequential/partitioned UPDATE, V2 control, commit-conflict |
| `interop_mor_update_lineage.rs` | GEN for `run-interop-mor-update-lineage.sh` (Java-created V3 tables; two MoR UPDATE statements + RePark COW UPDATE-then-DELETE) |
| `shared_puffin_dv/` | Shared-Puffin deletion-vector DML |

## I want to...

| I want to... | go to |
|---|---|
| Pin `with_commit_branch` scan + commit | `commit_branch.rs` |
| Prove Java/Rust branch DML interop | `interop_branch_dml.rs` |
| Pin default (no branch) DML | `integration_datafusion_test.rs` |

## Pointers

- **Up:** [crates/integrations/datafusion/](..) · **Related:** [crates/iceberg/src/scan/map.md](../../../iceberg/src/scan/map.md), [crates/iceberg/src/transaction/map.md](../../../iceberg/src/transaction/map.md), [task/f6c-branch-following-reads-ledger.md](../../../../task/f6c-branch-following-reads-ledger.md)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Named-branch SELECT returns `main` rows | `IcebergTableProvider::scan` passed `None` instead of `resolve_scan_snapshot_id` |
| Missing-ref DELETE creates the branch | the read leg did not call `resolve_scan_snapshot_id` (INSERT VALUES is the only create-on-missing path) |
| V3 MoR DELETE on a diverged branch fails "not a live file" | DV container close used `current_snapshot()`; see `close_touched_dv_containers_at` |

### First checks

- `cargo test -p iceberg-datafusion --test commit_branch --locked`

### Escalate to

- [../../iceberg/src/scan/map.md#debug](../../../iceberg/src/scan/map.md#debug)
