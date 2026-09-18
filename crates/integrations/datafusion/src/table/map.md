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

# map.md — crates/integrations/datafusion/src/table/

## Purpose

DataFusion `TableProvider` implementations. Metadata-table `scan` honors `projection`
(row R169).

## Contents

| File | Role |
|---|---|
| `mod.rs` | `IcebergTableProvider` (catalog-backed, writes). `schema()` advertises the metadata-free Arrow schema (`strip_metadata_from_schema`) because DataFusion's `insert_to_plan`/`Values` exec type literals against it and reject stamped nested field ids (F-LIST-INSERT-1 r2); the internal stamped schema still feeds `scan` and the write path |
| `loaded.rs` | `IcebergTableProvider` loaded-Table paths (`from_planning_load`, planning fast path) |
| `static_provider.rs` | `IcebergStaticTableProvider` (one snapshot, read-only) |
| `metadata_table.rs` | `IcebergMetadataTableProvider` — inspect tables as DataFusion tables |
| `table_provider_factory.rs` | DataFusion factory for `CREATE EXTERNAL TABLE` |
| `tests.rs` | `#[cfg(test)]` unit tests: provider construction, static provider, partitioning/sort/limit plans, shared fixtures (`pub(super)` helpers reused by `schema_evo_tests.rs`) |
| `schema_evo_tests.rs` | `#[cfg(test)]` unit tests: schema-evolution cells — stale providers, delete/update binding to the current schema, renames, nested evolution, pushdown after evolution |

## I want to...

| I want to... | go to |
|---|---|
| Project metadata-table columns | `metadata_table.rs` `TableProvider::scan` → [../physical_plan/map.md](../physical_plan/map.md) `metadata_scan.rs` |
| Bind a catalog table | `mod.rs` `IcebergTableProvider` |
| Time-travel a snapshot | `static_provider.rs` |

## Pointers

- **Up:** [../map.md](../map.md) · **Related:** [../physical_plan/map.md](../physical_plan/map.md),
  [../../../../iceberg/src/inspect/map.md](../../../../iceberg/src/inspect/map.md)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Projected metadata scan schema is the full schema | `TableProvider::scan` dropped `projection` (row R169) |
| `SELECT count(*)` over `$snapshots` is wrong | empty projection lost row count in `IcebergMetadataScan` |
| `schema()` on the provider is projected | advertised schema must stay full; only the plan schema projects |

### First checks

1. `IcebergMetadataTableProvider::scan` passes `projection` into `IcebergMetadataScan::new`.
2. `TableProvider::schema` still returns the full field set (metadata-free for SQL planning); the
   stamped schema is internal.

### Escalate to

[docs/parity/GAP_MATRIX.md](../../../../../docs/parity/GAP_MATRIX.md) row R169
