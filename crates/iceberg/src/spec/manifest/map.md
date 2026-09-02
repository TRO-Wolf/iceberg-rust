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

# map.md — crates/iceberg/src/spec/manifest/

## Purpose

Manifest entries, data-file structs, and the V3 rewrite-aware first-row-id allocator.

## Contents

| File | Role |
|---|---|
| `entry.rs` | `ManifestEntry`, `assign_first_row_ids` (Java `ManifestReader.idAssigner`) |
| `data_file.rs` | on-disk data-file fields including `first_row_id` |
| `writer.rs` | `ManifestWriter` / `ManifestWriterBuilder` |
| `rewrite_aware.rs` | `RewriteAwareFirstRowIds`: per-file stored `_row_id` recovery after Suppress. Increment: mixed = new rows; stored source = holes; no removed files = 0; else Java `+= existing+added`. |
| `metadata.rs` | manifest metadata |

## I want to...

| I want to... | go to |
|---|---|
| Stamp stored `_row_id` files after Suppress | `rewrite_aware.rs` `apply_rewrite_aware_first_row_ids` |
| Inherit `first_row_id` on read | `entry.rs` `assign_first_row_ids` |

## Pointers

- **Up:** [../map.md](../map.md) if present · **Related:** [../../transaction/map.md](../../transaction/map.md)

## Debug

| Symptom | Likely cause |
|---|---|
| COW DELETE of id=1 vs id=2 disagrees on `next-row-id` | `source_has_stored_row_ids` not passed into `ManifestWriterBuilder` (`None` = no removed files, `Some(false)` = first materialization, `Some(true)` = stored source) |
| Mixed stored+new file advances by all rows | `unassigned_row_count` not set on `ManifestFile` |
| MoR UPDATE advances `next-row-id` | no-removed path must return `Some(0)`, not Java `+= added` |
