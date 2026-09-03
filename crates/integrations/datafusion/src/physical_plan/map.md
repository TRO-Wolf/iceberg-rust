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

# map.md — crates/integrations/datafusion/src/physical_plan/

## Purpose

Physical plans for Iceberg scans and DML. `metadata_scan.rs` is the inspect-table
exec (row R169).

## Contents

| File | Role |
|---|---|
| `scan.rs` | `IcebergTableScan` (data files; projection by field id) |
| `metadata_scan.rs` | `IcebergMetadataScan` — projects inspect batches |
| `project.rs` | partition-value projection |
| `commit.rs` / `write.rs` | INSERT commit |
| `delete.rs` / `update.rs` | DELETE / UPDATE |
| `delete_legacy_merge.rs` | F-22: thin wrapper; close collects legacy deletes in one pass and merges via `load_legacy_positions_by_path` (R114) |
| `repartition.rs` / `sort.rs` | writer helpers |
| `expr_to_predicate.rs` | filter pushdown |
| `row_lineage.rs` / `snapshot_target.rs` / `cow_affected.rs` | DML helpers. `row_lineage.rs` is the single lineage attach path for COW DELETE/UPDATE and MoR UPDATE (`attach_update_lineage`, `cow_scan_stream`). |
| `mod.rs` | module root |

## I want to...

| I want to... | go to |
|---|---|
| Honor metadata-table `projection` | `metadata_scan.rs` (schema at plan time, `RecordBatch::project` at execute) |
| Honor data-table `projection` | `scan.rs` `IcebergTableScan::new` |
| Change INSERT / DELETE / UPDATE | `commit.rs` / `delete.rs` / `update.rs` |
| Merge V2 parquet position deletes into a V3 DV | `delete_legacy_merge.rs` `write_deletion_vectors` → `close_touched_dv_containers_with_partitions` (`legacy_deletes` + `load_legacy_positions_by_path`) |
| Carry V3 `_row_id` on MoR UPDATE | `delete.rs` `merge_on_read_update` + `row_lineage.rs` |

## Pointers

- **Up:** [../map.md](../map.md) · **Related:** [../table/map.md](../table/map.md)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Metadata plan schema has every column | `IcebergMetadataScan::new` ignored `projection` |
| Empty projection panics or reports 0 rows | `RecordBatch::project(&[])` must keep `num_rows` |
| Reordered columns come back in table order | execute path is not applying `batch.project(indices)` |
| MoR UPDATE reassigns `_row_id` | `merge_on_read_update` omitted `push_lineage_scan_columns` / `attach_update_lineage` |
| COW rewrite drifts `next-row-id` | layout mismatch (file / manifest count between the compared trees); an unassigned V3 DATA manifest advances by existing+added rows, a carried assigned manifest advances 0 (`spec/manifest/map.md`) |
| V3 MoR DELETE on a diverged branch: data file is not a live file of the scanned snapshot | `close_touched_dv_containers` walked `current_snapshot()` (main). Pass the scan snapshot to `close_touched_dv_containers_at` |
| V3 DELETE on a V2-upgrade table with live parquet position deletes resurrects rows | close must return `legacy_deletes` from the same delete-manifest pass and merge via `load_legacy_positions`; file-scoped parquet is removed, partition-scoped is kept |
| Delete manifests are read twice per V3 DELETE | F-22: `write_deletion_vectors` must not walk manifests; consume `DvContainerClose::legacy_deletes` |
| Partition-scoped parquet delete is dropped after a one-file DELETE | `referenced_data_file_location` is None: merge per touched file, do not add the parquet to `DvContainerClose::removed` |
| Old parquet delete from an earlier snapshot is merged into a newer data file's DV | sequence filter: skip when `delete_seq < data_seq` |

### First checks

1. `IcebergMetadataScan` stores `projection` and builds `PlanProperties` from the projected schema.
2. `execute` maps each batch through `RecordBatch::project`.

### Escalate to

[docs/parity/GAP_MATRIX.md](../../../../../docs/parity/GAP_MATRIX.md) row R169
