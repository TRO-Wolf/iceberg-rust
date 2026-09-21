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
| `reader.rs` | `ArrowReader`: Parquet/Avro/ORC task execution — projection masks, `RowFilter` planning via `row_filter_plan`, row-group + page-index pruning, the post-decode residual (`RowFilterPlan::Residual` → `evaluate_predicate_to_mask` → prune extra columns), fallback field-id stamping for id-less files (`add_fallback_field_ids_to_arrow_schema`: table-name match, Java-counter fallback) |
| `row_filter_plan.rs` | `plan_row_filter`: `Push` (leaf-index map incl. group ids) vs `Residual` (present-but-unmapped ids on id-less files); `top_level_ancestor_id`, `unmapped_group_leaf_indices`, `leaf_count`; `PushedRowFilter { filter, disable_predicate_cache }` + `apply_to_stream` (`pushed_mask_disables_predicate_cache`, `group_is_list`) turns the predicate cache off when the pushed mask selects the leaf of a single-leaf non-LIST group root, which would otherwise panic in parquet's cached array reader |
| `row_filter_nested_tests.rs` | **test-only** red-first pin: `IS NULL` on an optional single-leaf struct with the struct projected returns only the null-struct row (parquet predicate-cache panic guard) |
| `record_batch_predicate.rs` | `evaluate_predicate_to_mask`: `BoundPredicate` → `BooleanArray` over one batch, resolving references by field-id path with parent-validity propagation (`column_at_path` + `null_propagation`) |
| `record_batch_predicate_container_tests.rs` | **test-only** Spark container-null oracle over a materialized batch (incl. `st.b` both parities) |
| `null_propagation.rs` | `array_with_parent_validity`: union a struct parent's nulls into a child column (Arrow never propagates validity downward) |
| `record_batch_transformer.rs` (+ `*_tests.rs`) | `RecordBatchTransformer`: type promotion, defaults, reorder, partition constants, `_file`/`_pos`/row-lineage virtual columns |
| `partition_constant.rs` | `_partition` struct-constant synthesis: union partition type across specs, per-file tuple coercion, `ColumnSource::PartitionConstant` source + `StructArray` builder (NULL struct for unpartitioned tables and missing specs) |
| `record_batch_projector.rs` | batch projection onto the read schema |
| `record_batch_partition_splitter.rs` | split batches by partition value |
| `nested_projection.rs` (+ `*_tests.rs`) | nested-column projection with schema evolution |
| `schema.rs` | Arrow ↔ Iceberg schema conversion |
| `value.rs` (+ `value_tests.rs`, `value_tail_tests.rs`) | Arrow value conversion |
| `partition_value_calculator.rs` | partition values from data batches |
| `nan_val_cnt_visitor.rs` | NaN / null-count metric collection |
| `delete_filter.rs` | merge-on-read delete filtering over batches |
| `delete_file_loader.rs` / `caching_delete_file_loader.rs` | delete-file loading (+ cached) |
| `equality_delete_set.rs` | equality-delete keyset matching |
| `avro_reader.rs` (+ `avro_reader_tests.rs`) | Avro whole-file reader |
| `orc_reader.rs` (+ `orc_reader_tests.rs`, `orc_reader/`) | ORC whole-file reader |
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
| NULL struct parent reads as non-null child | Arrow validity does not propagate down; every struct descent must go through `array_with_parent_validity` |
| Rows duplicated on AVRO/ORC splits | whole-file readers ignore byte ranges — see [../scan/map.md#debug](../scan/map.md#debug) |

### First checks

- Reproduce on the materialized batch (`record_batch_predicate_container_tests`) to separate
  evaluator bugs from reader/planning bugs.
- For id-less files, check the stamped arrow schema metadata (`PARQUET_FIELD_ID_META_KEY`)
  before suspecting the evaluator.

### Escalate to

- Binding/term issues → [../expr/map.md#debug](../expr/map.md#debug).
- Planning/pruning divergence → [../scan/map.md#debug](../scan/map.md#debug).
