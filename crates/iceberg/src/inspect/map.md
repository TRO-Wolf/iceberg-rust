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

# map.md — crates/iceberg/src/inspect/

## Purpose

Metadata-inspection tables (Java `core/.../MetadataTableType` + the `*Table` classes): virtual
tables projecting table metadata as Arrow `RecordBatch`es. The ported Java table set is the
complete `MetadataTableType` vocabulary — `position_deletes` is schema-only (scan refused
loud; row R142). Entry
point: `MetadataTable` (`metadata_table.rs`), reached via `table.inspect()`.

## Contents

| File | Java analogue | Notes |
|---|---|---|
| `metadata_table.rs` | `MetadataTableType` | `MetadataTable` dispatcher + `MetadataTableType` enum |
| `snapshots.rs` / `manifests.rs` | `SnapshotsTable` / `ManifestsTable` | pre-existing upstream pair; `manifests` count columns are **content-gated** (data vs delete) |
| `files.rs` | `BaseFilesTable` | `files` / `data_files` / `delete_files` — differ only by manifest-content filter (`FilesTableKind`); also hosts the `all_*` variants via `MetadataScope` |
| `entries.rs` | `ManifestEntriesTable` | every entry **including `Deleted` tombstones** (status==2); nests the shared data_file struct |
| `data_file.rs` | `DataFile.getType` | shared data_file projection (`files` flattens it, `entries` nests it). Partition-struct append: `Timestamptz` / `TimestamptzNs` project (row R162); `Uuid` / `Fixed` stay `FeatureUnsupported` |
| `manifest_source.rs` | `BaseAllMetadataTableScan.reachableManifests` | `MetadataScope {CurrentSnapshot, AllSnapshots}` — all-snapshot manifest union, dedup by `manifest_path` (files NOT deduped) |
| `all_manifests.rs` | `AllManifestsTable` | one row per (manifest × referencing snapshot), NO dedup; `reference_snapshot_id` column |
| `history.rs` / `refs.rs` / `metadata_log_entries.rs` | `HistoryTable` / `RefsTable` / `MetadataLogEntriesTable` | pure-metadata (no manifest IO); `is_current_ancestor` = membership in the current parent chain |
| `partitions.rs` | `PartitionsTable` | the aggregating table: live entries grouped by partition struct; `last_updated_*` via strict-`>` tie-break |
| `partition_summary.rs` | — | shared partition-summary builder (used by `manifests` + `all_manifests`) |
| `readable_metrics.rs` | `MetricsUtil.readableMetricsStruct` | virtual per-leaf-column typed-metrics struct; bounds decoded via `Datum::try_from_bytes`; field ids seeded at host table's `highestFieldId()` |
| `java_schema_shape.rs` | `MetadataTableUtils` / `BaseFilesTable.schema` / `PartitionsTable.schema` / `BaseEntriesTable.schema` / `MetricsUtil.readableMetricsSchema` | **test-only** battery: Java-cited field id/name/required pins. Increment 1 — files-family (all six analogues) + `partitions`. Increment 2 — `entries`/`all_entries` shape (`ManifestEntry.wrapFileSchema` interleave, nested `selectNot(102)`) and `readable_metrics` STRUCTURE (per-leaf-column struct, six `READABLE_METRIC_COLS`, by-name emit order, host-`highestFieldId` id seed). OUT: `position_deletes` rows, cross-spec unification. Interior field-id ORDER is a documented divergence, pinned as such (Java's HashMap order is not portable) |
| `position_deletes.rs` | `PositionDeletesTable.calculateSchema` | **schema only** (FB-2): fixed `MetadataColumns` ids, partition child-id reassignment, empty-partition `partition` drop; `scan()` refused loud (`FeatureUnsupported`) — `PositionDeletesBatchScan` unported (row R142) |

## I want to...

| I want to... | go to |
|---|---|
| Add/modify an inspection table | the matching file above; field **ids and order are verbatim from Java** — check the Java `*Table` class first |
| Change what counts as a "live" entry | `files.rs` (`is_alive()`) — mutation-pinned, touch with care |
| Understand current-vs-all snapshot scope | `manifest_source.rs` (`MetadataScope`) |
| See known divergences from Java | GAP_MATRIX inspection rows: unpartitioned `partition` drop now matches Java on files/entries/partitions (row R142, 2026-08-14); timestamptz / timestamptz_ns now project from `.files` / `.partitions` (row R162; interop pending); cross-spec `Partitioning.partitionType` unification deferred; `readable_metrics` sub-field order is ascending-field-id (Java: HashMap order); `position_deletes` metadata table is unported (row R142) |
| Lock Java files/partitions/entries/readable_metrics schema shape | `java_schema_shape.rs` (cite the Java class/method in the test) |

## Pointers

- **Up:** [crates/iceberg/src/](..) · **Related:** `spec/` (manifest/entry types),
  [../scan/map.md](../scan/map.md) (the data-path scan), [../../tests/map.md](../../tests/map.md)
  (`interop_inspection*.rs`)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Wrong rows in `files` vs `entries` | `files` excludes `Deleted` tombstones, `entries` includes them — check the status filter, not the projection |
| Count columns wrong in `manifests`/`all_manifests` | Counts must be **content-gated** (data counts only on data manifests, delete counts only on delete manifests) — a past real bug |
| Inspection interop mismatch vs Java | Materialize expectations from a **re-parsed** base: Rust matches Java's on-disk round-trip (e.g. `operation` split out of the summary map), not Java's in-memory object |
| Duplicate rows in `all_*` tables | Expected — Java javadoc says "may return duplicate rows"; manifests dedup by path, files do not |
| `field_builder::<T>(i)` returns `None` / "child builder has unexpected type" at runtime | `StructBuilder::from_fields` builds Map/List children as BOXED builders (`MapBuilder<Box<dyn ArrayBuilder>, …>`); downcast the boxed inner builders. Compiles fine with the wrong typed builder — only fails on first append, so run the tests, don't trust a green build |
| `FeatureUnsupported` `partition field type … is not supported in the data_file metadata projection` | `append_partition_field` still refuses `Uuid` / `Fixed` (row R162). `Timestamptz` / `TimestamptzNs` are projected; if those error, the append match drifted from `readable_metrics.rs` |
| Scan panics "unmasked nulls for non-nullable StructArray field" | Java metadata-table `required` flags are NOMINAL (its `Object[]` rows emit nulls unchecked); Arrow ENFORCES them. If the Java producer can return null (boxed `Boolean`, `@Nullable`), the Arrow field must be nullable regardless of Java's schema flag |
| Tie-break / per-snapshot test passes under inverted comparison | The fixture collapsed to one distinct commit time: `ManifestWriter::add_entry` RESTAMPS `snapshot_id` (and forces `Added`); write the older file with `add_existing_entry` to preserve the parent snapshot id |

### First checks

- Compare field ids + column order against the Java `*Table` class — ids are non-sequential and
  pinned by tests.
- For aggregation bugs (`partitions`): check the strict-`>` last-updated tie-break and `is_alive()`.

### Escalate to

- Manifest/entry decoding → `spec/` and `avro/`.
- Cross-direction verification → [dev/java-interop/map.md#debug](../../../../dev/java-interop/map.md#debug).
