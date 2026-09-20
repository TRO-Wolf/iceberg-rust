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
| `rewrite_data_files.rs` | Compaction. Java `RewriteDataFiles`. Bin-pack by default; `strategy()` selects sort or z-order (RDF-SORT-1). Plans candidates, rewrites live rows, commits once by default and per batch under partial progress through `RewriteFiles`. |
| `rewrite_data_files_plan.rs` | Candidate and group predicates. Java `BinPackRewriteFilePlanner`. Size band, `tooManyDeletes`, `tooHighDeleteRatio`, `rewrite_all` bypass, current-spec grouping (output spec reaches only the write), `RewriteJobOrder`, commit batches. |
| `rewrite_data_files_dv.rs` | Drop Puffin DVs whose referenced data file this rewrite removes (Java `ManifestFilterManager.isDanglingDV`, DV-only). One delete-manifest walk returns the path set of every file-scoped position delete (planner ratio) and clones only the DVs (the drop candidates). |
| `rewrite_data_files_write.rs` | Read a planned group with merge-on-read applied and write compacted data files under the output spec. **F-REWRITE-SIZE-1 step 2:** `dictionary_fallback_columns` reads the group's input footers (`ArrowFileReader::get_metadata`, `task.file_size_in_bytes` skips the stat) and `set_column_dictionary_enabled(path, false)`s only the columns whose chunks show fallback — dict page + PLAIN data pages, no dict page, or an uncompressed-bytes-per-value ≥ half the physical width (near-unique ⇒ the output dictionary would overflow its 1 MiB limit and write a dead page). **Remediation (S2-21):** the decision pass runs under `ParquetReadOptions` hint = `FOOTER_SIZE` with all index policies off, folds each footer into the column-decision map as it arrives, and returns `HashMap<Arc<str>, Arc<ParquetMetaData>>` keyed by path; `ArrowReaderBuilder::with_prefetched_parquet_metadata` fuses those footers into the scan so `open_parquet_file` never re-fetches a footer (indexes load lazily via `ParquetMetaDataReader::load_page_index` only when the scan needs them). |
| `rewrite_data_files_sort.rs` | The public `RewriteStrategy` / `ZOrderSpec` surface, Java's strategy and option preconditions, the z tuple's column bind (the caller's own name against a TOP-LEVEL field, Java's `validZOrderColNames` + `df.schema().apply`; a dotted or differently-cased name is refused, never rebound), and the `sort_order_id` stamp (Java `SortOrderUtil.findTableSortOrder`: the first table sort order equal field for field, else 0). `shuffle-partitions-per-file` and `compression-factor` are accepted, validated and inert; bin-pack refuses them with Java's message. |
| `rewrite_data_files_sort_key.rs` | The byte-comparable row key both sort arms compare on: a null marker per field, sign-flipped integers, IEEE total-order floats with a canonical NaN, escaped variable-length bytes, descending by byte inversion. One encoding for the in-run sort and the k-way merge, so the two cannot disagree. |
| `rewrite_data_files_sort_run.rs` | The bounded-memory external merge sort. Runs fill to `sort-memory-budget-bytes` (default 128 MiB), spill as parquet under `<data location>/rewrite-sort-spill-<uuid>/` through the table's own `FileIO`, and merge k-way (fan-in 16, more runs merge in passes) into the writer. Every spill path is deleted on success AND on failure, and never reaches a commit. |
| `rewrite_data_files_zorder.rs` | Java `ZOrderByteUtils` + `SparkZOrderUDF`, byte for byte: 8-byte whole numbers, the arithmetic-shift float mask, `timestamptz` in SECONDS (Spark's `cast(ts AS LONG)`) against `timestamp` in micros, character-boundary string truncation, `min(sum, max-output-size)` interleave. |
| `rewrite_data_files_sort_tests.rs`, `rewrite_data_files_sort_bound_tests.rs`, `rewrite_data_files_zorder_tests.rs`, `rewrite_data_files_sort_key_tests.rs`, `rewrite_data_files_sort_harness.rs`, `rewrite_data_files_sort_vectors.rs` | RDF-SORT-1 pins: the run-25d Spark sort oracle cells replayed row for row, Java's own z bytes measured through the JVM, and the bounded-memory pins. `_sort_key_tests.rs` drives the two encoders directly, where an oracle cell's row order cannot see a clause break (the output-size cap, the sign flip, the escaping and terminator, the wide decimal). `_vectors.rs` is recorded data, not code. |
| `rewrite_data_files_router.rs` | Bounded LRU partition router for rewrite output. Default 64 open writers. Private to maintenance. |
| `rewrite_data_files_evolved_spec_tests.rs` | Spec-evolution output routing pins: source-field, transform, unpartitioned, mixed specs. |
| `rewrite_data_files_evolved_schema_tests.rs` | Schema-evolution compaction pins: add(+spec), add-only, drop, rename, promote, v3-DV, unpartitioned controls. |
| `rewrite_data_files_router_bound_tests.rs` | Writer bound, eviction, V3 lineage, and evolved-spec delete-class pins. |
| `rewrite_data_files_ratio_tests.rs` | Execute-path pins for `delete_ratio_threshold` and file-scoped delete removal. |
| `rewrite_data_files_options_tests.rs` | F-RDF-OPTIONS-1 pins: one-commit default, partial progress, `rewrite_all`, `output_spec_id`, job order, concurrency, Java float rendering. Round 2 adds the L-001 oracle cells (current-spec grouping, spec-0 fan-out). |
| `rewrite_data_files_plan_tests.rs` | Planner unit pins moved out of `rewrite_data_files.rs` (file-size split): partition isolation and incompatible-spec bucketing. |
| `rewrite_data_files_dangling_tests.rs` | Composed dangling-delete pins moved out of `rewrite_data_files.rs` (file-size split). |
| `rewrite_data_files_delete_loader_tests.rs` | F-RDF-GRANULARITY-1 round 2: counting `Storage`/`StorageFactory` over local fs counts `reader()` calls per path; a group split into 4 read tasks with one partition-scoped equality delete loads the delete file exactly once (the shared `CachingDeleteFileLoader` cache) and conserves rows. |
| `rewrite_data_files_fuse_tests.rs` | S2-21 footer-fuse pins: counting `Storage`/`FileRead` harness over memory storage; decision footers are bounded to the group and carry no column/offset index, and the decision fetch is the 8-byte tail plus the exact footer. |
| `rewrite_data_files_mw7_tests.rs` | The MW-7 pair (unpartitioned v2, one in-band data file, one PARTITION-scoped position delete covering every row): reclaimed when the delete carries EQUAL exact `file_path` bounds, a no-op when it does not — the bounds leg of Java `ContentFileUtil.referencedDataFile`, which is how Spark reclaims the shape. pins: task/f16-residue-2-partition-scoped-ratio-ledger.md |
| `delete_file_seq_gc_tests.rs` | Delete-file sequence GC pins (F-RDF-COW-BYTES-1 round 4): every merging commit retires a delete whose data seq is below every live data file's; fast append / merge append / adds-only commits keep it. |
| `dangling_dv_commit_tests.rs` | F-DANGLING-DV-COMMIT-1 cells: every merging commit that removes a data file drops its DV (Java `removeDanglingDeletesFor`); sibling-blob, live-reference, parquet-delete, and append/merge-append/row-delta-adds controls. |
| `remove_dangling_delete_files.rs` | Composed GC pass. Java `RemoveDanglingDeletes`. Opt-in on `RewriteDataFiles`, default off. |
| `rewrite_position_delete_files.rs` | Compact live parquet position deletes, or convert them to DVs on v3. The v3 arm gates legacy deletes by `(spec_id, partition)` through the same candidate/pack/group predicates; below-floor groups stay parquet with honest zeros. `rewrite_all(true)` bypasses both filters on both arms. |
| `rewrite_position_delete_files_v3.rs` | The v3 parquet-to-DV arm: inventory, DV planning, shadow refusals. Child module of the action file (file-size split, no behavior seam). |
| `rewrite_position_delete_files_floor_tests.rs` | Below-floor, at-floor, bypass, and gate-shadow pins. Child module of the action tests (file-size split). |
| `partition_key_audit.rs` | Offline partition-key audit + repair: recomputes every live data file's partition tuple from its rows, and repair rewrites miskeyed files through `RewriteFiles`. Repair output compression comes from `parquet_compression_from_properties` (F-WRITE-COMPRESS-2). |
| `partition_key_audit_tests.rs` | Audit/repair pins incl. `test_repair_rewritten_files_carry_the_table_codec` (F-WRITE-COMPRESS-2). |
| `actions_provider.rs` | Java `ActionsProvider` factory. |
| `add_files.rs` | Adopt existing parquet files into a table IN PLACE, in one `append` commit. Java `AddFilesProcedure` + `SparkTableUtil` + `TableMigrationUtil`, without Spark. Source is a recursively listed directory (hive `k=v` segments give the partition values) or an explicit file list with its own values. Carries `findCompatibleSpec`, the three `validatePartitionFilter` refusals, `filterPartitions`, the duplicate check against the live manifest entries, `ensureNameMappingPresent`, and a bounded-concurrency footer read. Free-standing (NOT an `ActionsProvider` method — Java's is a Spark procedure). Ledger: task/f-add-files-1-ledger.md |
| `add_files_datafile.rs` | One imported parquet file → one `DataFile`. Field ids by Java `ParquetUtil.getParquetTypeWithIds` (the file's embedded ids, else the table's name mapping, else positional) — a file carrying ids on SOME columns only is REFUSED (ledger D-17). Metrics from the footer under the table's `MetricsConfig` keyed by the FILE's column names, read with the column/offset/page indexes preloaded OFF. Partition values through the `Conversions.fromPartitionString` rules (`__HIVE_DEFAULT_PARTITION__` is NULL, BOOLEAN is `Boolean.valueOf` and never refuses, FLOAT/DOUBLE follow `Float.valueOf`), `sort_order_id` 0 and NO split offsets (what `DataFiles.Builder` leaves on this path). Also home to `unescape_hive_path_name`, Spark's `ExternalCatalogUtils.unescapePathName` — NOT the inverse of `spec::partition`'s `escape_partition_path_component`, which is `URLEncoder` and disagrees on `+` (ledger D-18). |
| `add_files_tests.rs`, `add_files_refusal_tests.rs`, `add_files_field_id_tests.rs`, `add_files_round2_tests.rs` | F-ADD-FILES-1 pins: the recorded Spark `add_files` oracle cells replayed, every refusal as a typed error carrying Java's message shape, and the field-id / metrics / partition-order cells a single-column single-order fixture cannot see (reversed source columns, embedded ids, a foreign id, a two-field spec, `MetricsConfig`). `add_files_round2_tests.rs` carries the reviewers' round-2 findings: the hive unescape table, the unescaped `partition_filter`, the mixed-field-id refusal, `Boolean.valueOf` / `Float.valueOf`, a delete file's path as a duplicate, the lowest-spec-id choice and the all-void spec. |

## I want to...

| I want to... | go to |
|---|---|
| Adopt existing parquet files into a table | `add_files.rs` (`AddFiles::new(table, source).execute(catalog)`). Files are NOT copied; every adopted `file_path` stays where it was. |
| Change how an imported file's field ids are resolved | `add_files_datafile.rs::resolve_field_id`. Embedded ids win for the WHOLE file when any column carries one (Java `ParquetSchemaUtil.hasIds`); a column that resolves to no id, or to an id the table schema lacks, is DROPPED and reads back NULL. A file whose top-level columns carry ids UNEVENLY is refused by `refuse_partial_field_ids` — the fork's own reader resolves such a file differently and the scan then fails (ledger D-17, `task/todo.md`). |
| Change which data files compaction selects | `rewrite_data_files_plan.rs` (`is_candidate`, `group_qualifies`, `too_high_delete_ratio`) |
| Count file-scoped parquet position deletes toward the ratio | `rewrite_data_files_dv.rs::file_scoped_delete_paths` then `ResolvedConfig.file_scoped_delete_paths`. Scan-task deletes do not carry `file_path` bounds. |
| Drop deletes that targeted a rewritten data file | `rewrite_data_files_dv.rs::plan_dv_removal`. Java drops DVs only (`isDanglingDV`). The same drop runs inside every merging commit (`transaction/snapshot/manifest_filter.rs::is_dangling_dv`, F-DANGLING-DV-COMMIT-1, Java `removeDanglingDeletesFor`), so a data-file-removing commit never carries a dangling DV forward. A parquet delete is retired only by the commit's sequence GC (`transaction/snapshot/manifest_filter.rs`, Java `dropDeleteFilesOlderThan`). F-19b: no sibling rewrite; the sibling blob stays in the original Puffin. One DELETE-manifest walk is cached for planning and commit; `file_scoped_delete_paths` is path-only. |
| Change output rolling or the rewrite read | `rewrite_data_files_write.rs` |
| Add a rewrite strategy, or change its preconditions | `rewrite_data_files_sort.rs` (`RewriteStrategy`, `resolve_strategy`) |
| Change what a sort or z-order rewrite orders rows by | `rewrite_data_files_sort_key.rs` (sort fields) or `rewrite_data_files_zorder.rs` (the z tuple) |
| Change how a sort spills, merges or cleans up | `rewrite_data_files_sort_run.rs` |
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
| Ratio fires but the parquet delete file stays | Expected since F-RDF-COW-BYTES-1: `plan_dv_removal` is Puffin-only (`is_deletion_vector`), so a file-scoped parquet delete stays live after its referenced file is rewritten, as Java's `ManifestFilterManager.isDanglingDV` does. The rewrite commit's sequence GC retires it only when its data seq is below every live data file's (Java `dropDeleteFilesOlderThan`); a delete at the starting sequence stays. `remove_dangling_deletes` defaults off. |
| A below-threshold partial delete is rewritten | Size band, not the ratio. Check `min_file_size_bytes` / `max_file_size_bytes` against the file. Pin: `test_default_ratio_under_threshold_parquet_is_a_noop`. |
| A two-path parquet position delete rewrites both files | Absent bounds (Spark PARTITION) and unequal Full bounds are both not file-scoped. Pins: `test_absent_path_bounds_two_path_parquet_pos_delete_does_not_fire_ratio`, `test_unequal_path_bounds_two_path_parquet_pos_delete_does_not_fire_ratio`. |
| A shared partition-scoped delete vanishes when one file is rewritten | `plan_dv_removal` drops only Puffin DVs (`is_deletion_vector`); a parquet delete that still applies to a live data file is never dropped by the rewrite commit. Pin: `test_partition_scoped_delete_survives_partial_rewrite`. |
| A sort rewrite writes independently sorted files instead of one order | The group went down the bin-pack path (`KeyPlan::build` returned `None`). Only `RewriteStrategy::BinPack` may. Pin: `sorted_output_files_hold_one_global_order_in_disjoint_ranges`. |
| A sort rewrite writes ONE file whatever `target-file-size-bytes` says | `RollingFileWriter` cuts a file only on a 1000-row boundary (`ROWS_DIVISOR`), so a group under 1000 rows never splits. Spark cuts at sampled range boundaries instead. Ledger: task/rdf-sort-1-ledger.md D-9. |
| `add_files` adopted a file but a column reads back NULL | The column resolved to no field id. Either the file carries embedded ids and that column has none, or the table's `schema.name-mapping.default` does not name it. This is Java's behaviour, not a fork bug. Pin: `a_source_column_the_target_lacks_is_dropped_and_reads_back_null`. |
| `add_files` refuses with "Cannot find any file to import under …" | The source prefix listed nothing. `FileIO::list` returns an EMPTY list for an absent prefix on a directory-semantics backend, so a typo in the source path lands here. Hidden `_`/`.` segments are skipped at every level. Ledger D-14. |
| `add_files` refuses with "Cannot find a partition spec … that matches the partition columns ([…])" | The source's hive directory names, UNESCAPED, lowercased and in order, equal no all-identity spec in the table (Java `SparkTableUtil.findCompatibleSpec`). A VOID-transform field disqualifies a spec, so a v1 table that dropped its only partition field matches nothing. Specs are walked by ascending `spec_id`, Java's metadata list order. |
| `add_files` refuses a source whose columns carry field ids unevenly | Intended (ledger D-17). Java would adopt it and drop the id-less columns; the fork refuses because its own reader would then fail the scan with "Found duplicate 'field.id'". |
| An adopted file carries no split offsets | Expected. Java `TableMigrationUtil.buildDataFile` never calls `withSplitOffsets`, and `DataFiles.Builder` leaves the field null. Pin: `an_adopted_file_carries_sort_order_id_zero_and_no_split_offsets`. |
| A `rewrite-sort-spill-*` directory survives a rewrite | The spill cleanup runs on both exits; a surviving file means the process died mid-rewrite. The files sit under the table's data location, so `delete_orphan_files` reclaims them. |
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
