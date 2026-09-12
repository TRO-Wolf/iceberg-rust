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
| `rewrite_data_files_write.rs` | Read a planned group with merge-on-read applied and write compacted data files under the current spec. **F-REWRITE-SIZE-1 step 2:** `dictionary_fallback_columns` reads the group's input footers (`ArrowFileReader::get_metadata`, `task.file_size_in_bytes` skips the stat) and `set_column_dictionary_enabled(path, false)`s only the columns whose chunks show fallback — dict page + PLAIN data pages, no dict page, or an uncompressed-bytes-per-value ≥ half the physical width (near-unique ⇒ the output dictionary would overflow its 1 MiB limit and write a dead page). **Remediation (S2-21):** the decision pass runs under `ParquetReadOptions` hint = `FOOTER_SIZE` with all index policies off, folds each footer into the column-decision map as it arrives, and returns `HashMap<Arc<str>, Arc<ParquetMetaData>>` keyed by path; `ArrowReaderBuilder::with_prefetched_parquet_metadata` fuses those footers into the scan so `open_parquet_file` never re-fetches a footer (indexes load lazily via `ParquetMetaDataReader::load_page_index` only when the scan needs them). |
| `rewrite_data_files_router.rs` | Bounded LRU partition router for rewrite output. Default 64 open writers. Private to maintenance. |
| `rewrite_data_files_evolved_spec_tests.rs` | Spec-evolution output routing pins: source-field, transform, unpartitioned, mixed specs. |
| `rewrite_data_files_evolved_schema_tests.rs` | Schema-evolution compaction pins: add(+spec), add-only, drop, rename, promote, v3-DV, unpartitioned controls. |
| `rewrite_data_files_router_bound_tests.rs` | Writer bound, eviction, V3 lineage, and evolved-spec delete-class pins. |
| `rewrite_data_files_ratio_tests.rs` | Execute-path pins for `delete_ratio_threshold` and file-scoped delete removal. |
| `rewrite_data_files_fuse_tests.rs` | S2-21 footer-fuse pins: counting `Storage`/`FileRead` harness over memory storage; decision footers are bounded to the group and carry no column/offset index, and the decision fetch is the 8-byte tail plus the exact footer. |
| `rewrite_data_files_mw7_tests.rs` | The MW-7 pair (unpartitioned v2, one in-band data file, one PARTITION-scoped position delete covering every row): reclaimed when the delete carries EQUAL exact `file_path` bounds, a no-op when it does not — the bounds leg of Java `ContentFileUtil.referencedDataFile`, which is how Spark reclaims the shape. pins: task/f16-residue-2-partition-scoped-ratio-ledger.md |
| `remove_dangling_delete_files.rs` | Composed GC pass. Java `RemoveDanglingDeletes`. Opt-in on `RewriteDataFiles`, default off. |
| `rewrite_position_delete_files.rs` | Compact live parquet position deletes, or convert them to DVs on v3. The v3 arm gates legacy deletes by `(spec_id, partition)` through the same candidate/pack/group predicates; below-floor groups stay parquet with honest zeros. `rewrite_all(true)` bypasses both filters on both arms. |
| `rewrite_position_delete_files_v3.rs` | The v3 parquet-to-DV arm: inventory, DV planning, shadow refusals. Child module of the action file (file-size split, no behavior seam). |
| `rewrite_position_delete_files_floor_tests.rs` | Below-floor, at-floor, bypass, and gate-shadow pins. Child module of the action tests (file-size split). |
| `partition_key_audit.rs` | Offline partition-key audit + repair: recomputes every live data file's partition tuple from its rows, and repair rewrites miskeyed files through `RewriteFiles`. Repair output compression comes from `parquet_compression_from_properties` (F-WRITE-COMPRESS-2). |
| `partition_key_audit_tests.rs` | Audit/repair pins incl. `test_repair_rewritten_files_carry_the_table_codec` (F-WRITE-COMPRESS-2). |
| `actions_provider.rs` | Java `ActionsProvider` factory. |

## I want to...

| I want to... | go to |
|---|---|
| Change which data files compaction selects | `rewrite_data_files_plan.rs` (`is_candidate`, `group_qualifies`, `too_high_delete_ratio`) |
| Count file-scoped parquet position deletes toward the ratio | `rewrite_data_files_dv.rs::file_scoped_delete_paths` then `ResolvedConfig.file_scoped_delete_paths`. Scan-task deletes do not carry `file_path` bounds. |
| Drop deletes that targeted a rewritten data file | `rewrite_data_files_dv.rs::plan_dv_removal`. Java drops DVs only (`isDanglingDV`). The parquet file-scoped drop is a fork extension. F-19b: no sibling rewrite; the sibling blob stays in the original Puffin. One DELETE-manifest walk is cached for planning and commit; `file_scoped_delete_paths` is path-only. |
| Change output rolling or the rewrite read | `rewrite_data_files_write.rs` |
| Change how rewritten rows are routed after spec evolution | `rewrite_data_files_write.rs` + `rewrite_data_files_router.rs` |
| Pin evolved-spec output tuples or the writer bound | `rewrite_data_files_evolved_spec_tests.rs`, `rewrite_data_files_router_bound_tests.rs` |
| Pin evolved-schema compaction (add/drop/rename/promote/v3-DV) | `rewrite_data_files_evolved_schema_tests.rs` |
| See why an all-void current spec (`void(x)`, one field, `is_unpartitioned`) fails rewrite | unsupported current-spec shape: `RecordBatchPartitionSplitter` refuses it (`Cannot create partition calculator for unpartitioned table`). Pin: `all_void_current_spec_is_refused` |
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
| After spec evolution, partition-pruned scans miss live rows | Output used `group.first()` under the current spec. Routing must recompute tuples from rows (`rewrite_data_files_write.rs`). Pin: `source_field_identity_x_to_identity_y_rewrites_two_old_partitions`. |
| Rewrite fails with `Cannot create partition calculator for unpartitioned table` | Current spec is all-void (`void(x)`, one field, `is_unpartitioned` but `fields()` is not empty). Unsupported current-spec shape. Pin: `all_void_current_spec_is_refused`. |
| Rewrite after schema evolution fails with a batch/expected width, name, or type mismatch | Tasks plan under the snapshot-pinned old schema while the calculator/writer build on the current schema. The write path re-points each task at the current schema with the full current projection (`rewrite_data_files_write.rs`), so the reader evolves every file's batches first. Pins: `rewrite_data_files_evolved_schema_tests::*`. |
| Manifest write after promoting a partition source fails with `value is not compatible with type` | Old carried tuples keep the narrow literal while summaries build against the current wide type. `PartitionFieldStats::update` widens via `PrimitiveLiteral::promote_to` (int→long, float→double; anything else still fails loud). Pin: `promote_partition_source_int_to_long_rewrites_old_files`. |

### First checks

1. Read the planned `FileScanTask.deletes`: `referenced_data_file` vs `referenced_data_file_location` on the live `DataFile`.
2. Confirm `ResolvedConfig.file_scoped_delete_paths` contains the parquet delete path.
3. Confirm `group_qualifies` is true for a one-file group when the ratio fires.
4. After execute, assert `removed_delete_files_count` and `live_delete_file_paths`, not only row identity.

### Escalate to

- GAP_MATRIX row R135 · [../scan/map.md](../scan/map.md)#debug · [../delete_file_index.rs](../delete_file_index.rs) (`referenced_data_file_location`)
