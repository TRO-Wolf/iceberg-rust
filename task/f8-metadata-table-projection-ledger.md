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

# F-8 evidence ledger — metadata-table projection + `table_names` synthesis

Base `33c20da31`. GAP_MATRIX rows R169 (projection) and R170 (listing).

## 1. Measured before (base)

### Projection

`IcebergMetadataTableProvider::scan` took `_projection: Option<&Vec<usize>>` and
dropped it. `IcebergMetadataScan::new` always built `PlanProperties` from
`provider.schema()` (the full inspect schema) and `execute` streamed unprojected
batches.

| Input | Observed at base |
|---|---|
| `projection = None` | full schema, full rows |
| `projection = Some([last, 0])` | full schema, full rows (engine shim had to re-project) |
| `projection = Some([])` (`SELECT count(*)`) | full schema, full rows (engine shim had to drop columns) |

Snapshots fixture `TableMetadataV2Valid.json`: 2 snapshots, 6 columns
(`committed_at`, `snapshot_id`, `parent_id`, `operation`, `manifest_list`,
`summary`).

### Listing

`IcebergSchemaProvider::table_names` chained every `MetadataTableType` onto each
catalog name (`{base}${suffix}`). `test_provider_list_table_names` at base
expected 17 names for one table (1 catalog + 16 metadata types).
`show_tables.slt` listed both catalog tables plus 16 twins each.

`table` / `table_exist` / `split_metadata_table_ref` (last-`$` + vocabulary)
already resolved `$` names. That path was not the bug.

## 2. Change

1. `IcebergMetadataScan::new` projects the advertised schema at plan time
   (`Schema::project`). `execute` applies `RecordBatch::project` (empty
   projection keeps `num_rows`).
2. `table_names` returns catalog directory keys only. Resolution is unchanged.

No public signature change. `IcebergMetadataScan` is `pub(crate)`.
`TableProvider::scan` already took `projection`.

## 3. Measured after

| Input | Observed |
|---|---|
| `projection = None` | 6 snapshot columns, 2 rows |
| `projection = Some([last, 0])` | columns `[summary, committed_at]` in that order; values match the full scan |
| `projection = Some([])` | 0 columns, 2 rows |
| index `999` | plan-time error |
| `table_names` for `my_table` / `orders` | `["my_table"]` / `["orders"]` |
| `table_names` for `a$b` | `["a$b"]` only (no `a$b$files`) |
| `table("orders$snapshots")` | still resolves; `snapshot_id` present |
| `table_exist("orders$snapshots")` | still true |
| `SHOW TABLES` (slt) | catalog tables only |

## 4. Pins

| Element | Test |
|---|---|
| subset + reorder + values | `test_metadata_table_scan_projects_subset_in_requested_order` |
| empty projection row count | `test_metadata_table_scan_empty_projection_preserves_row_count` |
| out-of-bounds projection | `test_metadata_table_scan_rejects_out_of_bounds_projection` |
| listing catalog-only + `$` resolve | `test_table_names_lists_catalog_entries_only_and_dollar_names_still_resolve` |
| `$` in the base name | `test_dollar_in_base_table_name_enumeration_exist_resolve_and_read` |
| SQL `$` resolve after listing change | `test_dollar_in_base_table_name_sql_read_and_metadata_twin` |
| lazy listing still names unloadable tables, not twins | `table_names_and_existence_come_from_listing_not_loading` |

## 5. Gates

- `make check` — exit 0 (fmt, clippy `-D warnings`, taplo, machete, agent-artifacts, matrix-anchors 84 rows, comment-blocks, rust-file-size 400 files / 101 legacy ceilings; `integration_datafusion_test.rs` ceiling 6913 → 6893).
- `cargo test -p iceberg-datafusion --locked` — exit 0 (lib + integration + doctests).
- Docker `make test` legs excused (docker unavailable in this session).

## 6. Out of scope

Consuming-engine shim deletion (next engine repin). Filters / limit on the
metadata-table scan stay unpushed.
