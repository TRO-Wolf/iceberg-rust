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

Manifest entries, data-file structs, and the V3 first-row-id reader/writer.

## Contents

| File | Role |
|---|---|
| `entry.rs` | `ManifestEntry`, `assign_first_row_ids` (Java `ManifestReader.idAssigner`), `apply_manifest_list_context` (list-entry inherit + assign per caller) |
| `data_file.rs` | on-disk data-file fields including `first_row_id` |
| `writer.rs` | `ManifestWriter` / `ManifestWriterBuilder`. EXISTING/DELETED entries copy `data_file.first_row_id` verbatim. Manifest `first_row_id` stays null for the list writer. |
| `metadata.rs` | manifest metadata |

## I want to...

| I want to... | go to |
|---|---|
| Inherit `first_row_id` on read | `entry.rs` `assign_first_row_ids` |
| See how `next-row-id` advances | `../manifest_list.rs` `ManifestListWriter::assign_first_row_id` (Java `+= existing+added` on unassigned DATA) |

## Pointers

- **Up:** [../map.md](../map.md) if present · **Related:** [../../transaction/map.md](../../transaction/map.md)

## Debug

| Symptom | Likely cause |
|---|---|
| Sequential COW `next-row-id` disagrees with a Spark notebook | layout mismatch (file count / manifest count); numbers compare only at matched layout |
| EXISTING survivor lost `first_row_id` | filtered rewrite did not copy the entry's `data_file` |
