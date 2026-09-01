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

# map.md — crates/iceberg/src/maintenance/

## Purpose

Engine-agnostic table maintenance actions. They read a committed `Table` and rewrite or
remove files. Status lives on GAP_MATRIX rows R133–R140.

## Contents

| File | What it does |
|---|---|
| `rewrite_data_files.rs` | Bin-pack compaction. Java `RewriteDataFiles`. Plans candidates, rewrites live rows, commits per group through `RewriteFiles`. |
| `rewrite_data_files_plan.rs` | Candidate and group predicates. Java `BinPackRewriteFilePlanner`. Size band, `tooManyDeletes`, `tooHighDeleteRatio`. |
| `rewrite_data_files_dv.rs` | Drop file-scoped deletes whose referenced data file this rewrite removes. Java `isDanglingDV` is DV-only; parquet file-scoped drops are a fork extension. Rewrite Puffin siblings. |
| `rewrite_data_files_write.rs` | Read a planned group with merge-on-read applied and write compacted data files. |
| `rewrite_data_files_ratio_tests.rs` | Execute-path pins for `delete_ratio_threshold` and file-scoped delete removal. |
| `remove_dangling_delete_files.rs` | Composed GC pass. Java `RemoveDanglingDeletes`. Opt-in on `RewriteDataFiles`, default off. |
| `rewrite_position_delete_files.rs` | Compact live parquet position deletes, or convert them to DVs on v3. |
| `actions_provider.rs` | Java `ActionsProvider` factory. |

## I want to...

| I want to... | go to |
|---|---|
| Change which data files compaction selects | `rewrite_data_files_plan.rs` (`is_candidate`, `group_qualifies`, `too_high_delete_ratio`) |
| Count file-scoped parquet position deletes toward the ratio | `rewrite_data_files_dv.rs::file_scoped_delete_paths` then `ResolvedConfig.file_scoped_delete_paths`. Scan-task deletes do not carry `file_path` bounds. |
| Drop deletes that targeted a rewritten data file | `rewrite_data_files_dv.rs::plan_dv_removal`. Java drops DVs only (`isDanglingDV`). The parquet file-scoped drop is a fork extension. |
| Change output rolling or the rewrite read | `rewrite_data_files_write.rs` |
| Pin delete-ratio or 100%-dead in-band rewrite | `rewrite_data_files_ratio_tests.rs` |

## Pointers

- **Up:** [crates/iceberg/src/](..) · **Related:** [../scan/map.md](../scan/map.md) (plan_files attachments), [../transaction/map.md](../transaction/map.md) (`RewriteFiles`), [../writer/map.md](../writer/map.md) (position-delete bounds), GAP_MATRIX row R135

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| A 100%-deleted in-band file survives `RewriteDataFiles` | The ratio counts only file-scoped deletes. Scan-task `referenced_data_file` is the raw field (null on v2 parquet). The planner must load `file_scoped_delete_paths` from `referenced_data_file_location` (equal `file_path` bounds). Probe: `test_planner_selects_bounds_only_parquet_because_referenced_data_file_location_is_set`. |
| Ratio fires but the parquet delete file stays | `plan_dv_removal` must also drop non-DV file-scoped position deletes whose referenced path was rewritten. `remove_dangling_deletes` defaults off and uses a seq `<` clause that often keeps the same-seq delete. |
| A below-threshold partial delete is rewritten | Size band, not the ratio. Check `min_file_size_bytes` / `max_file_size_bytes` against the file. Pin: `test_default_ratio_under_threshold_parquet_is_a_noop`. |
| A two-path parquet position delete rewrites both files | Absent bounds (Spark PARTITION) and unequal Full bounds are both not file-scoped. Pins: `test_absent_path_bounds_two_path_parquet_pos_delete_does_not_fire_ratio`, `test_unequal_path_bounds_two_path_parquet_pos_delete_does_not_fire_ratio`. |
| A shared partition-scoped delete vanishes when one file is rewritten | `plan_dv_removal` must drop only file-scoped parquet whose referenced path was rewritten. Pin: `test_partition_scoped_delete_survives_partial_rewrite`. |
| A single-file group is skipped | `enough_input_files` and `enough_content` require `size > 1`. `any_too_high_delete_ratio` does not. A lone needs-rewrite candidate must still qualify. |

### First checks

1. Read the planned `FileScanTask.deletes`: `referenced_data_file` vs `referenced_data_file_location` on the live `DataFile`.
2. Confirm `ResolvedConfig.file_scoped_delete_paths` contains the parquet delete path.
3. Confirm `group_qualifies` is true for a one-file group when the ratio fires.
4. After execute, assert `removed_delete_files_count` and `live_delete_file_paths`, not only row identity.

### Escalate to

- GAP_MATRIX row R135 · [../scan/map.md](../scan/map.md)#debug · [../delete_file_index.rs](../delete_file_index.rs) (`referenced_data_file_location`)
