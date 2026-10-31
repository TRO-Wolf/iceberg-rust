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

# map.md — crates/iceberg/src/arrow/

## Purpose

Arrow execution for scans: file-format readers (Parquet / Avro / ORC) that decode
`FileScanTask`s into `RecordBatch`es, the `RowFilter` / post-decode residual that applies the
scan predicate, schema projection/transformation onto the read schema, and merge-on-read delete
application. Scan planning lives in `../scan/`; this directory runs the plan.

## Contents

| File | What it does |
|---|---|
| `reader.rs` | `ArrowReader`: Parquet/Avro/ORC task execution — projection masks, `RowFilter` planning via `row_filter_plan`, row-group + page-index pruning, the post-decode residual (`RowFilterPlan::Residual` → `evaluate_predicate_to_mask` → prune extra columns), fallback field-id stamping for id-less files (`add_fallback_field_ids_to_arrow_schema`: table-name match, Java-counter fallback). The `_pos` / `_row_id` whole-file path unions dropped equality-key leaves into its own file-level projection mask so the survival predicate reads them by id; the scan projection is unchanged. `get_arrow_projection_mask_with_field_ids` starts its leaf list from `row_filter_plan::validity_leaves`, so a struct whose every child was dropped still reads its validity from the file (F-STRUCT-EMPTY-CHILD-PANIC-1) |
| `row_filter_plan.rs` | `plan_row_filter`: `Push` (leaf-index map incl. group ids) vs `Residual` (present-but-unmapped ids on id-less files); `top_level_ancestor_id`, `unmapped_group_leaf_indices`, `leaf_count`; `PushedRowFilter { filter, disable_predicate_cache }` + `apply_to_stream` (`pushed_mask_disables_predicate_cache`, `group_is_list`) turns the predicate cache off when the pushed mask selects the leaf of a single-leaf non-LIST group root, which would otherwise panic in parquet's cached array reader. Predicate ids absent from the task schema (equality deletes on dropped keys) fail loud with `FeatureUnsupported` outside the embedded-id `Push` path, which reads them from the data file by id. `dropped_equality_key_mask` gives the `_pos` / `_row_id` whole-file read the leaves of dropped equality keys by embedded field id (fold-2 V-EQKEY-1), refusing `FeatureUnsupported` on a data file with no embedded ids. `validity_leaves` (F-STRUCT-EMPTY-CHILD-PANIC-1): for every projected struct none of whose task-schema leaves is present inside the file's group for its id (all children dropped, or, on a data file that carries embedded field ids, every task child added after the file was written; the id-less case is ledger S-16, carded), at any depth and inside lists and maps, it picks the first leaf of the file's group with that field id (`index_file_groups` mirrors arrow's `filter_leaves` leaf order), so Parquet decodes the struct's definition levels; the nested projection drops the borrowed leaf from the output |
| `evolution_nested_null_tests.rs` | **test-only** schema-evolution pins: a STRUCT / MAP / LIST / struct-keyed MAP column added after a Parquet file was written reads as typed nulls for that file, and as values for files written after the ALTER |
| `row_filter_nested_tests.rs` | **test-only** red-first pin: `IS NULL` on an optional single-leaf struct with the struct projected returns only the null-struct row (parquet predicate-cache panic guard) |
| `record_batch_predicate.rs` | `evaluate_predicate_to_mask`: `BoundPredicate` → `BooleanArray` over one batch, resolving references by field-id path with parent-validity propagation (`column_at_path` + `null_propagation`) |
| `record_batch_predicate_container_tests.rs` | **test-only** Spark container-null oracle over a materialized batch (incl. `st.b` both parities) |
| `null_propagation.rs` | `array_with_parent_validity`: union a struct parent's nulls into a child column (Arrow never propagates validity downward). `null_struct`: an all-NULL `StructArray` of explicit length through `try_new_with_length`, safe for an empty field list |
| `record_batch_transformer.rs` (+ `*_tests.rs`) | `RecordBatchTransformer`: type promotion, defaults, reorder, partition constants, `_file`/`_pos`/row-lineage virtual columns |
| `partition_constant.rs` | `_partition` struct-constant synthesis: union partition type across specs, per-file tuple coercion, `ColumnSource::PartitionConstant` source + `StructArray` builder (NULL struct for unpartitioned tables and missing specs) |
| `record_batch_projector.rs` | batch projection onto the read schema |
| `record_batch_partition_splitter.rs` | split batches by partition value |
| `nested_projection.rs` (+ `*_tests.rs`) | nested-column projection with schema evolution. A target struct with no fields plans as an empty `Struct` node, and every rebuilt struct uses `try_new_with_length`, so an emptied struct keeps the source struct's null buffer |
| `schema.rs` | Arrow ↔ Iceberg schema conversion |
| `value.rs` (+ `value_tests.rs`, `value_tail_tests.rs`) | Arrow value conversion. NULL struct constants (`create_primitive_array_repeated`, `create_primitive_array_single_element`) build through `null_struct`, so an empty struct type answers a typed NULL column, never a panic |
| `partition_value_calculator.rs` | partition values from data batches |
| `nan_val_cnt_visitor.rs` | NaN / null-count metric collection; the batch-as-struct view uses `try_new_with_length` with the batch's row count, so a zero-column batch views as a zero-field struct of the batch length instead of panicking; any other construction failure maps to an `Error` |
| `delete_filter.rs` | merge-on-read delete filtering over batches; each equality-delete predicate binds against its own file's schema (dropped keys resolve by field id, never by name) |
| `delete_file_loader.rs` / `caching_delete_file_loader.rs` | delete-file loading (+ cached). F-POSDEL-FORMAT-READ-1: position and equality deletes open with the reader for the task's `file_format` (parquet as before; Avro/ORC through the whole-file readers over the reserved `file_path`+`pos` schema / the key-column subset schema; Puffin rows refuse `FeatureUnsupported`). F-EQDEL-DROPPED-KEY-1: `equality_delete_batch_stream` returns the evolved stream plus its per-file bind schema for every format, built by `eq_delete_schema.rs` |
| `eq_delete_schema.rs` | per-file equality-delete schema (F-EQDEL-DROPPED-KEY-1 fold-2): present ids at the same struct path keep it (the current schema pruned to them); a key the file holds at the top level gets a top-level entry (a dropped key recovered from the file, or a nested key a Rust writer flattened); an id absent from the delete file's own schema refuses `DataInvalid` naming the file and the id; a dropped nested key, a key under another struct path or inside a list/map refuses `FeatureUnsupported`. The file's own schema comes from its embedded Parquet ids, the Avro writer schema, or the ORC footer (`orc_top_level_schema`); `whole_file_equality_delete_batches` reads Avro/ORC delete files against it |
| `pos_delete_parse.rs` | positional-delete stream parsing, moved out of the caching loader: per-data-file delete vectors from path+pos batches; short arity, null cells and negative positions fail closed |
| `pos_del_claim.rs` | positional-delete load-claim key derivation (a parquet file claims under its path, a Puffin deletion vector under `{path}@{offset}`) + the terminal-failure error renderer |
| `equality_delete_set.rs` | equality-delete keyset matching; a key column absent from the batch (a post-drop file, a nested key) hands the batch to the predicate oracle; `eq_delete_key_fields_projected` (moved from `reader.rs`) gates the post-decode keyset path |
| `avro_reader.rs` (+ `avro_reader_tests.rs`) | Avro whole-file reader |
| `orc_reader.rs` (+ `orc_reader_tests.rs`, `orc_reader/`) | ORC whole-file reader; `orc_top_level_schema` maps the footer's top-level columns (names, `iceberg.id`, `iceberg.required`, and the `iceberg.long-type` / `binary-type` / `length` / `timestamp-unit` attributes) to an Iceberg schema |
| `open_parquet.rs` (+ `open_parquet_tests.rs`) | Parquet file opening |
| `footer_cache.rs` (+ `footer_cache_tests.rs`, `footer_cache_v_tests.rs`, `footer_cache_r2_tests.rs`) | Parquet footer caching |
| `ranges.rs` | byte-range fetching |
| `int96.rs` | Int96 conversion |
| `page_prune_fixture.rs` / `page_prune_tests.rs` / `page_prune_tests_2.rs` / `page_prune_perf_tests.rs` | **test-only** page-pruning fixtures |
| `spark_fixture_tests.rs` / `f_transform_arrow_types_1_tests.rs` | **test-only** fixtures |

## I want to...

| I want to... | go to |
|---|---|
| Change how the scan predicate filters Parquet rows | `row_filter_plan.rs` (Push-vs-Residual decision) + `reader.rs` (residual application + column pruning) |
| Change null-predicate evaluation on batches | `record_batch_predicate.rs` (`column_at_path` resolution, `null_propagation` for NULL parents) |
| Change id-less / name-mapped reads | `reader.rs::add_fallback_field_ids_to_arrow_schema` (stamping) + `get_arrow_projection_mask*` (projection) + `row_filter_plan.rs` (Residual routing) |
| Touch delete application | `delete_filter.rs`, `equality_delete_set.rs`, the delete-file loaders |
| Touch the transformer / projection | `record_batch_transformer.rs`, `record_batch_projector.rs`, `nested_projection.rs` |

## Pointers

- **Up:** [crates/iceberg/src/](../) · **Related:** [../scan/map.md](../scan/map.md)
  (planning + `FileScanTask`), [../expr/map.md](../expr/map.md) (predicates + binding),
  [../expr/visitors/map.md](../expr/visitors/map.md) (pruning evaluators)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Container / nested null predicate returns wrong rows on files WITH ids | `row_filter_plan::build_field_id_leaf_lists` must map group ids to their whole leaf span; `record_batch_predicate::column_at_path` must union parent validity — both are mutation-pinned (ledger F-CONTAINER-ACCESSOR-1 §7 M4/M5) |
| Wrong rows on id-less files | present-but-unmapped predicate ids must route to `RowFilterPlan::Residual` and apply post-decode; a partial Push reads wrong columns (pins: M6/M7); fallback stamping must name-match the table schema before the Java counter (M10) |
| Matching rows dropped under page-index row selection | unmapped GROUP ids must fail open (`field_id_names_a_group`), never `CantMatch` (pin: M9) |
| Library panic `use StructArray::try_new_with_length or StructArray::new_empty_fields`, or a NULL struct / list / map where the file row was non-null, after every child of a struct was dropped or, on files that carry embedded field ids, every task child of a struct is absent from the file (the id-less case is OPEN as S-16 of the F-STRUCT-EMPTY-CHILD-PANIC-1 ledger) | a leafless struct dropped out of the leaf-based Parquet mask; `row_filter_plan::validity_leaves` must borrow one file leaf and `nested_projection` must emit the empty struct with the source null buffer (pins: `crates/iceberg/tests/struct_empty_child.rs`, ledger F-STRUCT-EMPTY-CHILD-PANIC-1) |
| NULL struct parent reads as non-null child | Arrow validity does not propagate down; every struct descent must go through `array_with_parent_validity` |
| Rows duplicated on AVRO/ORC splits | whole-file readers ignore byte ranges — see [../scan/map.md#debug](../scan/map.md#debug) |
| `FeatureUnsupported` naming dropped column ids on a `_deleted` projection or an id-less file | equality deletes on dropped keys apply only through the embedded-id `Push` path; residual and id-less reads fail loud by design (F-EQDEL-DROPPED-KEY-1) instead of silently keeping deleted rows |
| Avro/ORC data-file scan errors after `DROP` of an equality-delete key | Avro and ORC data files refuse the schema-absent id loud with `DataInvalid` (`Projected field id ... is not present in the scan schema`), as does compaction; Parquet data files apply dropped-key deletes by field id — known residue, ledger F-EQDEL-DROPPED-KEY-1 § Follow-ups |

### First checks

- Reproduce on the materialized batch (`record_batch_predicate_container_tests`) to separate
  evaluator bugs from reader/planning bugs.
- For id-less files, check the stamped arrow schema metadata (`PARQUET_FIELD_ID_META_KEY`)
  before suspecting the evaluator.

### Escalate to

- Binding/term issues → [../expr/map.md#debug](../expr/map.md#debug).
- Planning/pruning divergence → [../scan/map.md#debug](../scan/map.md#debug).
