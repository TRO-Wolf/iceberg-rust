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

# map.md — crates/integrations/datafusion/src/

## Purpose

DataFusion catalog / schema / table-provider surface for Iceberg. Status: GAP_MATRIX
rows R164, R169, R170.

## Contents

| File | Role |
|---|---|
| `lib.rs` | crate root; re-exports catalog, table providers, scan knobs |
| `catalog.rs` | `IcebergCatalogProvider` (row R164 namespace scope) |
| `schema.rs` | `IcebergSchemaProvider` — `table_names` lists catalog entries only (row R170); `$`-name `table` / `table_exist` still resolve |
| `error.rs` | DataFusion error conversion |
| `task_writer.rs` | per-task data writers |
| `table/` | catalog-backed, static, and metadata table providers (row R169) |
| `physical_plan/` | scan, DML, metadata scan |

## I want to...

| I want to... | go to |
|---|---|
| Change how tables are listed | `schema.rs` `table_names` (row R170) |
| Resolve `<base>$snapshots` | `schema.rs` `split_metadata_table_ref` + `table` |
| Honor metadata-table projection | [table/map.md](table/map.md) |
| Push a data scan projection | [physical_plan/map.md](physical_plan/map.md) |

## Pointers

- **Up:** [crates/integrations/datafusion/](..) · **Related:** [table/map.md](table/map.md),
  [physical_plan/map.md](physical_plan/map.md), [docs/ENGINE_CONTRACT.md](../../../../docs/ENGINE_CONTRACT.md) §1

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| `SHOW TABLES` lists `t$snapshots` | `table_names` synthesizing `$` names again (row R170) |
| `t$snapshots` does not resolve | `split_metadata_table_ref` last-`$` + vocabulary drifted |
| Metadata scan returns every column under a projection | [table/map.md](table/map.md) / [physical_plan/map.md](physical_plan/map.md) (row R169) |

### First checks

1. `IcebergSchemaProvider::table_names` maps directory keys only.
2. `table` / `table_exist` still split on the last `$` against `MetadataTableType`.

### Escalate to

[docs/parity/GAP_MATRIX.md](../../../../docs/parity/GAP_MATRIX.md) rows R169 / R170
