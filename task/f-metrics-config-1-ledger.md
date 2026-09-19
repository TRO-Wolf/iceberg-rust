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

# F-METRICS-CONFIG-1 — Java's table metrics config, everywhere the fork writes

**Date:** 2026-09-19. **Base:** `origin/main` `43fcd243` (fork tip at branch
`fix/f-metrics-config-1`). **Model:** swe-2-high. **Path:** step 1 MEASURE (first
commit), step 2 RED-FIRST pins, step 3 IMPLEMENT, step 4 MUTATION + gates (later
commits).

This ledger retires when the unit lands or the owner removes it.

## Defect

`write.metadata.metrics.default` / `write.metadata.metrics.column.<col>` are table
properties users set to keep sensitive values out of manifest bounds. RePark measured
them IGNORED: the fork's own writers build `ParquetWriterBuilder` without the table's
resolved config, so `metrics.default=none` data files are rewritten by compaction WITH
full bounds — a privacy regression Java does not have.

## Java 1.11.0, verified against bytecode

Jar: `iceberg-spark-runtime-4.1_2.13-1.11.0.jar`, read with `javap -c -p`.

### `MetricsConfig.forTable(Table)` / `forPositionDelete(Table)` / `from(props, schema, sortOrder)`

- `forTable(table)` = `from(table.properties(), table.schema(), table.sortOrder())`.
- `forPositionDelete(table)`: puts `file_path -> Full`, `pos -> Full` (the
  `MetadataColumns.DELETE_FILE_PATH` / `DELETE_FILE_POS` names), then re-keys every
  entry of `forTable(table).columnModes` as `row.<name>` and keeps the table's
  `defaultMode`. The `row.` entries serve position-delete schemas carrying a `row`
  struct; the fork's delete schema has only `file_path`/`pos`, so they are inert today
  but ported verbatim.
- `forPositionDelete()` (no-arg) is a static `POSITION_DELETE_MODE` = columnModes
  `{file_path: Full, pos: Full}` over `DEFAULT_MODE` — this is what the fork's current
  `MetricsConfig::for_position_delete()` ports.
- `from(props, schema, sortOrder)`:
  1. `limit = maxInferredColumnDefaults(props)` —
     `PropertyUtil.propertyAsInt(props, "write.metadata.metrics.max-inferred-column-defaults", 100)`;
     `propertyAsInt` is a bare `Integer.parseInt` with NO catch — a non-numeric value
     throws `NumberFormatException` out of `forTable` (hard failure at write time, not
     a warn-fallback). A negative parsed value warns and falls back to 100.
  2. `defaultMode`: if `write.metadata.metrics.default` is set, `parseMode` it (bad
     value warns, falls back to `DEFAULT_MODE` = `truncate(16)`); else if `schema ==
     null` or `getProjectedIds(schema).size() <= limit`, `DEFAULT_MODE`; else the
     wide-schema branch: `defaultMode = None` AND the first `limit` field ids (Java
     traversal order) each get an explicit `columnModes[findColumnName(id)] =
     DEFAULT_MODE` entry. (Mechanism matters: the surviving columns carry explicit
     entries, the default itself flips to `none`.)
  3. Sorted promotion: `sortedDefault = sortedColumnDefaultMode(defaultMode)` —
     `None` or `Counts` -> `Truncate(16)`, else the mode unchanged. Then every column
     of `SortOrderUtil.orderPreservingSortedColumns(sortOrder)` is put into
     `columnModes` with `sortedDefault` (unconditional put — overwrites inferred
     entries, is overwritten by explicit `column.*` overrides which run next).
  4. Column overrides: every `write.metadata.metrics.column.<name>` key; the name is
     `key.replaceFirst(PREFIX, "")` (dotted nested names kept verbatim); bad value
     warns and falls back to the resolved `defaultMode`.
- `orderPreservingSortedColumns(order)`: null order -> empty set; otherwise fields
  filtered by `transform.preservesOrder()`, mapped `sourceId ->
  order.schema().findColumnName(sourceId)`, nulls dropped. The Rust `SortOrder` is
  unbound, so the fork resolves names through `TableMetadata.current_schema()` — Java
  binds the order to the same schema at build (`SortOrderParser`/`checkCompatibility`).
- `preservesOrder()` per transform, verified: `Identity`, `Truncate`, and
  `TimeTransform` (Years/Months/Days/Hours) return true; `Void`, `Bucket`, `Unknown`
  false. The fork's `Transform::preserves_order()` already matches exactly.

### `limitFieldIds(schema, limit)` (`MetricsConfig$1`, a `CustomOrderSchemaVisitor`)

- `metricsEligible(type)` = `isPrimitiveType() || isVariantType()` — struct/list/map
  fields are NOT eligible themselves.
- `struct()`: FIRST scans its direct fields in order, adding each eligible field id
  while `idSet.size() < limit`; THEN iterates the children in field order, descending
  lazily (each child's subtree fully consumed before the next, each level re-checking
  the limit).
- `list()`: if under limit and `elementType` is eligible, adds `elementId`; then
  descends into the element type (so `list<struct>` element fields CAN consume slots).
- `map()`: adds `keyId` then `valueId` under the same guards; then descends into key
  and value types.
- Oracle confirmation: `(id,s,d,st,xs)` with limit 2 -> `{1,2}` — the direct-field
  scan stops at the limit before descending.

### `TypeUtil.getProjectedIds(schema)` (`GetProjectedIds`, includeStructIds = true)

Adds a field's id when its type is primitive, variant, or struct. List/map field ids
are NOT added; their element/key/value field ids are added by their own `field()`
visits (primitive/variant/struct-typed children). For the oracle schema:
`{1,2,3,4,6,7,8}` (7 ids — `st` included, `xs` excluded, `xs.element` included).
The `list()/map()` null-result fallbacks are unreachable in the post-order walk
(every child `field()` returns the set).

### `ParquetMetrics$MetricsVisitor` (1.11.0, field-id-keyed)

- `message()`: first pass puts `columnSizes[fieldId] = totalCompressedSize` for every
  projected field whose `effectiveMode != None` — INCLUDING list/map descendants;
  then `footerMetrics` descends the schema.
- `primitive()`: mode `None` -> no metrics; otherwise `metricsFromFooter`/`counts`.
  `metricsFromFooter` returns `null` for INT96, returns counts-only when
  `truncateLength <= 0`, and returns `null` (dropping the column ENTIRELY — value
  counts, null counts and bounds) when any row-group chunk's `getStatistics()` is
  absent or `isEmpty()`. `bounds()` applies the same statless-drop.
- `list()` and `map()` return EMPTY lists — every metric under a list or map is
  dropped except `column_sizes` (collected in the `message()` pass).
- `struct()` descends; `variant()` has its own handling.
- `MetricsUtil.metricsMode(config, fieldId, schema)` resolves modes by
  `schema.findColumnName(fieldId)` — the fork's `name_by_field_id` + `column_mode`
  is the same lookup.

### Mode parsing

`MetricsModes.fromString`: case-insensitive `none`/`counts`/`full`, `truncate(N)`
with `N > 0` parsed as `int` (overflow rejects). `parseMode` catches
`IllegalArgumentException`, warns, returns the fallback. The fork's
`MetricsMode::parse` already matches.

### `validateReferencedColumns(schema)`

Lives on `MetricsConfig` but is called from `PropertiesUpdate`/`TableMetadata`
(property-set time), NOT from `from` or the write path — so `for_table` performs no
validation and unknown `column.*` names are inert, matching Java's writer behavior.

## Oracle (the run-24d metrics oracle), extracted

Table `(id BIGINT, s STRING, d DOUBLE, st STRUCT<a: STRING, b: INT>, xs ARRAY<INT>)`
— field ids 1,2,3,4(st),6(st.a),7(st.b),5(xs),8(xs.element); two rows:
`id` 1/3, `s` "alpha-long-string-value-0001"/"zulu-long-string-value-0003", `d`
1.5/2.5, `st` ("aa",1)/("zz",3), `xs` [1,2]/[3]. The Spark file's
`xs.list.element` column DOES carry footer stats (min 1, max 3, nulls 0); Java drops
its metrics anyway (list/map rule above).

Per-cell expected map KEYS (`column_sizes` values are engine-dependent — pins assert
key sets only; `value_counts`/`null_value_counts`/`nan_value_counts` pins assert
keys + values; bounds pins assert keys + exact bytes):

| cell | props | expected |
|---|---|---|
| default | — | counts {1,2,3,6,7}, sizes {1,2,3,6,7,8}, nan {3:0}, bounds {1,2,3,6,7}, `s` 16 B truncated |
| none | default=none | every map empty |
| counts | default=counts | counts+nan+sizes as default; bounds empty |
| truncate4 | default=truncate(4) | as default; `s` bounds `616c7068`/`7a756c76` |
| full | default=full | as default; `s` bounds untruncated |
| col_none | column.s=none | field 2 absent from EVERY map (sizes too) |
| col_nested | default=none + column.st.a=full | only field 6, all six maps |
| max_inferred_2 | max-inferred=2 | only fields {1,2} in every map |
| max_inferred_2_default_set | max-inferred=2 + default=counts | limit inert; counts everywhere, no bounds |
| sorted_none | default=none + WRITE ORDERED BY s | only field 2, `s` bounds 16 B |
| sorted_counts | default=counts + WRITE ORDERED BY d | counts everywhere; bounds only field 3 |
| bad_mode | default=bogus | identical to default cell |

## Fork writer inventory — every production `ParquetWriterBuilder` site

Config reaches metrics only via `ParquetWriterBuilder::with_metrics_config` →
`ParquetWriter::parquet_to_data_file_builder` (the writer's own close path is the
only `parquet_to_data_file_builder` caller).

| site | file:line | writes | today | fix |
|---|---|---|---|---|
| compaction rewrite | `maintenance/rewrite_data_files_write.rs:102` | data files | builder default | `for_table` |
| partition-key repair | `maintenance/partition_key_audit.rs:518` | data files | builder default | `for_table` |
| eq-delete conversion | `maintenance/convert_equality_delete_files.rs:509` | pos-delete | `for_position_delete()` | `for_position_delete_table` |
| pos-delete compaction | `maintenance/rewrite_position_delete_files.rs:684` | pos-delete | `for_position_delete()` | `for_position_delete_table` |
| table-path rewrite | `maintenance/rewrite_table_path.rs:500` | pos-delete | `for_position_delete()` | `for_position_delete_table` |
| DataFusion INSERT | `integrations/.../physical_plan/write.rs:304` | data files | builder default | `for_table` |
| DataFusion DML | `integrations/.../physical_plan/row_lineage.rs:242` | data files | builder default | `for_table` |
| DataFusion DELETE | `integrations/.../physical_plan/delete_position_deletes.rs:148` | pos-delete | `for_position_delete()` | `for_position_delete_table` |
| dead helper | `writer/file_writer/parquet_writer.rs:489` | data files | `from_properties` | `for_table` |

The brief's defect list also names `remove_dangling_delete_files.rs`,
`delete_vector_lookup.rs`, `compute_table_stats.rs` — measured WRONG: every
`ParquetWriterBuilder`/`PositionDeleteFileWriterBuilder` hit in those files is inside
`#[cfg(test)]` (boundaries at lines 327, 85, 363). Nothing to wire there. All
`transaction/*.rs`, `writer/base_writer/*`, `task_writer.rs`, `fanout_writer.rs`,
`unpartitioned_writer.rs`, `clustered_writer.rs` hits are likewise test-only — those
layers take a caller-built rolling builder, so config flows in from the sites above.

## Measured fork output vs the oracle (default config)

Fixture replayed through `ParquetWriterBuilder` + `DataFileWriterBuilder`
(scratch probe, not committed):

| map | fork keys | oracle keys | verdict |
|---|---|---|---|
| column_sizes | {1,2,3,6,7,8} | {1,2,3,6,7,8} | match (values differ — engine-dependent compressed sizes) |
| value_counts | {1,2,3,6,7,**8**} | {1,2,3,6,7} | fork emits element count |
| null_value_counts | {1,2,3,6,7,**8**} | {1,2,3,6,7} | fork emits element count |
| nan_value_counts | {3} | {3} | match |
| lower_bounds | {1,2,3,6,7,**8**} | {1,2,3,6,7} | fork emits element bound `01000000` |
| upper_bounds | {1,2,3,6,7,**8**} | {1,2,3,6,7} | fork emits element bound `03000000` |

## Scope decisions (measured, not guessed)

- **List/map-descendant metrics drop is IN SCOPE.** The pins compare the
  fork-written `DataFile`'s six maps with Spark's recorded ones; every cell fails on
  the extra field-8 entries until `parquet_to_data_file_builder` drops counts/bounds
  for fields not reachable through structs — Java's `MetricsVisitor.list()/.map()`
  empty-return, measured above. `column_sizes` keeps list/map fields (Java's
  `message()` pass).
- **Statless-chunk drop: deferred.** Java drops a column's value/null counts and
  bounds when ANY chunk's statistics is absent/empty; the fork gates only bounds.
  No oracle cell exercises a statless column (Spark always writes stats); recorded
  here as a known residue, not fixed this unit.
- **`max-inferred-column-defaults` non-numeric**: Java throws; the fork warns and
  falls back to 100 (the `for_table` API is infallible; warn-and-continue is this
  module's established fallback idiom). Negative: warn + 100 (matches Java).
- **`row.<name>` overlays are inert** in the fork's two-column delete schema but are
  ported verbatim (a future `row` field resolves them exactly as Java does).
- **RePark's own writers** are run 24c's half — this unit adds the helpers and wires
  the fork's production sites only.

## Propositions

- [ ] P1 `for_table` resolves default/column/max-inferred/sorted rules exactly per
      bytecode; pinned red-first by the 12 oracle cells.
- [ ] P2 `for_position_delete_table` = `for_table` + `file_path`/`pos` Full + `row.*`
      re-key; pinned by the delete-overlay pin.
- [ ] P3 every production writer above applies the resolved config; pinned by the
      `rewrite_data_files` e2e none-cell.
- [ ] P4 list/map descendants drop counts/bounds but keep `column_sizes`; pinned by
      every oracle cell's key sets.
- [ ] P5 mutations (drop promotion / drop limit / drop wiring) turn named pins red.
