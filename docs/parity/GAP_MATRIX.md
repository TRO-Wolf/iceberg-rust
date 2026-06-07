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

# Java ↔ Rust Capability Gap Matrix

> **Goal:** 1-to-1 capability parity between this Rust implementation and the Apache Iceberg
> **Java `iceberg-core` / `iceberg-api`** library (the engine-agnostic table-format library — *not*
> the Spark engine integration). This is a **living document**: re-run the audit after every upstream
> sync and after each parity phase lands.

## Status legend

- ✅ **present** — implemented to a usable degree
- 🟡 **partial** — exists but incomplete vs Java
- ❌ **missing** — not implemented

## Audit provenance

- **Rust base audited:** owned fork on upstream **`iceberg` 0.9.1** (datafusion 52.2, arrow 57.1,
  parquet 57.1, MSRV 1.92), **re-audited 2026-06-07** after the Phase 0 sync. Source of truth:
  `crates/iceberg/src/{spec,expr,scan,transaction,writer,arrow,io,inspect,puffin,catalog}` plus the
  new `crates/storage/opendal` FileIO crate.
- **Java reference:** `apache/iceberg` `main`, modules `api/` + `core/` + `data/` + `orc/` +
  `parquet/` + `arrow/`.
- **Re-audit policy:** re-run after every upstream sync and after each parity phase; date-stamp this
  block and strike the rows the sync/phase solved.
- **Interop oracle (test-only):** `dev/java-interop/` drives Java `iceberg-core` **1.10.0** as a
  read/write oracle for bidirectional fixtures (not shipped, not a Cargo dependency). It backs the
  first ✅-by-interop row, `UpdateSchema` (2026-06-07).
- **What the 0.7→0.9.1 sync changed (flipped rows below):** `timestamp_ns` type, column default
  values (`initial_default`/`write_default`), merge-on-read **read** application of position-deletes +
  deletion-vectors during scan, the `upgrade_format_version` transaction action, and a real
  `TransactionAction`/`ApplyTransactionAction` extension seam. The **headline gaps are unchanged**:
  write-engine actions, schema/partition/snapshot evolution, incremental scans, ORC/Avro data files,
  variant/geo/unknown types, catalog view ops, maintenance actions, encryption.

## Matrix

| Area | Status | Java reference | Rust location / note |
|---|---|---|---|
| Primitive + nested types | ✅ | `api/.../types/Types.java` | `spec/datatypes.rs` |
| V3 types: variant | ❌ | `api/.../variants/` | none |
| V3 types: geometry / geography | ❌ | `api/.../geospatial/` | none |
| V3 types: timestamp_ns | ✅ | `types/Types.java` | `spec/datatypes.rs` (`PrimitiveType::TimestampNs`/`TimestamptzNs`); **V3-only format-version gate enforced** — `Schema::check_compatibility` rejects a `timestamp_ns`/`timestamptz_ns` field (incl. nested) on a V1/V2 table (Java `Schema.MIN_FORMAT_VERSIONS`, "Invalid type for {col}: timestamp_ns is not supported until v3") |
| V3 types: unknown | ❌ | `types/Types.java` | none |
| Column default values (initial/write) | ✅ | `Schema`/`Types` | `spec/datatypes.rs` `NestedField` carries `initial_default`/`write_default` |
| Partition transforms (identity/bucket/truncate/year/month/day/hour/void) | ✅ | `api/.../transforms/` | `spec/transform.rs` |
| Schema evolution (`UpdateSchema`) | ✅ | `api/UpdateSchema.java`, `core/SchemaUpdate.java`, `schema/UnionByNameVisitor.java` | `transaction/update_schema.rs`: add/add-required (allow-incompatible gated), rename, update-type (promotion-gated), update-doc, make-optional/require, delete, move first/before/after (struct-local, cross-struct rejected), set-identifier-fields (exists/required/primitive/not-deleted rules; id-stable across rename/move), `union_by_name_with` at full `UnionByNameVisitor` parity (add new incl. nested under list/map structs; relax required→optional; legal promotion; reject incompatible primitive + complex↔primitive type changes; doc; mirrored no-op). Fresh nested field-id assignment is **level-order** (Java `AssignFreshIds`/`CustomOrderSchemaVisitor`: struct assigns all immediate ids then descends; map assigns key-id then value-id first) — pinned by `testAddNestedMapOfStructs`/`testAddNestedListOfStructs` exact-id tests. Case-insensitive lowercase-name collisions rejected at `Schema::build` (`Cannot build lower case index: a and b collide`, Java `TypeUtil.indexByLowerCaseName`). Column **initial/write defaults** plumbed through the builder API (Java `addColumn(..,Literal)` / `addRequiredColumn(..,default)` / `updateColumnDefault(..,Literal)` overloads): an add sets BOTH `initial_default` and `write_default`, a **required add WITH a default is allowed without `allow_incompatible_changes`** (the default backfills existing rows), and `update_column_default` sets only `write_default` on an existing field; defaults are type-validated (reject non-primitive "Invalid default value..."; reject mismatched primitive "Cannot cast default value to..."). A defaulted optional add can be made required without `allow_incompatible_changes` (its initial default backfills existing rows); `update_column_default` sets only the write default, so an add+`update_column_default`+require is still rejected (Java `testAddColumnWith[UpdateColumn]DefaultToRequiredColumn`). Emits `AddSchema`+`SetCurrentSchema{-1}` with `LastAssignedFieldIdMatch`+`CurrentSchemaIdMatch`. 75 unit tests (+ 2 schema-build collision tests + 7 `Schema::check_compatibility` tests [2 initial-default + 5 V3-type gate] + 5 `add_schema` V3-default-guard tests). **Interop ✅ (2026-06-07):** bidirectional Java round-trip via the `dev/java-interop/` oracle (Java `iceberg-core` 1.10.0, package-private `@VisibleForTesting SchemaUpdate(Schema,int)` ctor) + `crates/iceberg/tests/interop_update_schema.rs` over 7 committed scenarios (`add_top_level_columns`, `add_nested_struct_and_map` [the level-order nested-id case], `rename_and_move`, `update_type_promotion`, `make_optional_and_delete`, `set_identifier_fields`, `add_required_with_default_and_update_default`). Direction 1 (Rust reproduces Java's evolution) runs offline; Direction 2 (`mvn ... verify`) asserts Java reads the Rust-written metadata — 7/7 PASS both directions. **V3 initial-default guard enforced:** column initial defaults are V3-only in Java — `Schema.checkCompatibility(schema, formatVersion)` (called on every add-schema build path via `addSchemaInternal`) rejects a non-null `initialDefault` when `formatVersion < 3` ("...non-null default (...) is not supported until v3"). The Rust side now mirrors this exactly: `Schema::check_compatibility(format_version)` (`spec/schema/mod.rs`) is wired into `TableMetadataBuilder::add_schema` — the single choke point every add-schema path flows through — and iterates ALL fields incl. nested (via the recursive id→field index, the analogue of Java's `lazyIdToField()`), rejecting `add_required_column_with_default(..)` on a V1/V2 table with `ErrorKind::DataInvalid`. Only `initial_default` is gated (not `write_default`, matching Java). The two default-bearing interop scenarios use a V3 base (matching Java's contract); the V2 rejection is pinned both at the metadata-builder layer (`table_metadata_builder.rs`: top-level + nested rejected, V3 allowed, write-default-only allowed, no-default unaffected) and end-to-end through the catalog commit path (`interop_update_schema.rs::test_v2_default_is_rejected_by_v3_guard`). The capability is interop-proven in both directions with the guard in place. **V3-only type gate enforced too:** `Schema::check_compatibility` now mirrors Java `Schema.checkCompatibility` in FULL — the same single pass that gates initial-defaults also gates V3-only **types** via `MIN_FORMAT_VERSIONS`, rejecting a `timestamp_ns`/`timestamptz_ns` field (incl. nested, dotted path in the message) on a V1/V2 table ("Invalid type for {col}: timestamp_ns is not supported until v3"); a field that violates BOTH rules accumulates BOTH problems into the single combined "Invalid schema for v{N}:" error (ordered by field id, mirroring Java's TreeMap). _(Of Java's five `MIN_FORMAT_VERSIONS` types only `timestamp_ns`/`timestamptz_ns` (Java `TIMESTAMP_NANO`) are representable in Rust today; `variant`/`unknown`/`geometry`/`geography` get a one-line arm each in `min_format_version` when those types land — tracked in `task/todo.md`.)_ |
| Partition evolution (`UpdatePartitionSpec`) | 🟡 | `api/UpdatePartitionSpec.java`, `core/BaseUpdatePartitionSpec.java` | `transaction/update_partition_spec.rs`: add/add-with-transform (Java `PartitionNameGenerator` auto-naming), remove-by-name/by-transform, rename, `add_non_default_spec`, `case_sensitive`; full `BaseUpdatePartitionSpec` parity — dup-name/redundant-time-transform/remove-newly-added/rename+delete guards, delete-then-readd rewrite, V1 alwaysNull (void) replacement, and `recycleOrCreatePartitionField` (recycles a historical field's id AND name on a `(source,transform)` match). Emits `AddSpec`+`SetDefaultSpec{-1}` with `LastAssignedPartitionIdMatch` always + `DefaultSpecIdMatch` only when the spec is set as default (Java `UpdateRequirements`). Reviewed against `TestUpdatePartitionSpec.java` (28 unit tests incl. end-to-end builder round-trip + no-op dedup). **Pending ✅:** Java interop round-trip. |
| Sort order (`ReplaceSortOrder`) | ✅ | `api/ReplaceSortOrder.java` | `transaction/sort_order.rs` |
| Snapshot model + refs (branches/tags) | 🟡 | `api/Snapshot.java`, `SnapshotRef.java` | spec types + ref ops (`transaction/manage_snapshots.rs`) |
| Snapshot management (`ManageSnapshots`: branch/tag CRUD, rollback, rollback-to-time, set-current, fast-forward) | 🟡 | `api/ManageSnapshots.java`, `core/SetSnapshotOperation.java`, `api/SnapshotRef.java` | `transaction/manage_snapshots.rs`: create/replace/remove branch+tag, rename-branch, set-current, rollback (ancestry-checked), **rollback-to-time** (Java `SetSnapshotOperation.findLatestAncestorOlderThan`: newest ancestor with `timestamp_ms` strictly `<` the arg; errors "Cannot roll back, no valid snapshot older than" if none), fast-forward, retention — with **non-positive retention rejected** (`SnapshotRef.Builder` `> 0`: "Min snapshots to keep must be greater than 0" / "Max snapshot age must be greater than 0 ms" / "Max reference age must be greater than 0") and optimistic-concurrency `RefSnapshotIdMatch` guards + unit tests. **`cherrypick` is Phase-2-gated**, not a metadata op: Java `cherrypick` extends `MergingSnapshotProducer` and replays data files (gated on the write engine). **Pending ✅:** Java interop round-trip. |
| Manifest + manifest-list read/write | ✅ | `core/.../ManifestReader/Writer` | `spec/manifest`, `spec/manifest_list.rs` |
| `RewriteManifests` | ❌ | `api/RewriteManifests.java` | none |
| Write: fast append | ✅ | `api/AppendFiles.java` | `transaction/append.rs` |
| Write: merge append | ❌ | `AppendFiles` (merge mode) | none |
| Write: `OverwriteFiles` | ❌ | `api/OverwriteFiles.java` | none |
| Write: `ReplacePartitions` (dynamic/static overwrite) | ❌ | `api/ReplacePartitions.java` | none |
| Write: `DeleteFiles` | ❌ | `api/DeleteFiles.java` | none |
| Write: `RowDelta` (merge-on-read) | ❌ | `api/RowDelta.java` | none |
| Write: `RewriteFiles` (compaction commit) | ❌ | `api/RewriteFiles.java` | none |
| Transaction action extension seam | 🟡 | `core/.../BaseTransaction` | `transaction/action.rs` — `TransactionAction`/`ApplyTransactionAction` + `ActionCommit` exist (trait is `pub(crate)`; we own it → make `pub` in Phase 2) |
| Write: `upgrade_format_version` action | ✅ | format-version upgrade | `transaction/upgrade_format_version.rs` (new in 0.9) |
| Multi-op transactions + optimistic-concurrency retry | 🟡 | `api/Transaction.java` | `catalog.update_table`; needs validation against Glue/S3 Tables |
| Writer: data file | ✅ | `data/` | `writer/base_writer/data_file_writer.rs` |
| Writer: equality-delete | ✅ | `data/` | `writer/base_writer/equality_delete_writer.rs` |
| Writer: position-delete | 🟡 | `data/` | no dedicated `PositionDeleteWriter` in `writer/base_writer/`; read-side apply is done (see below) |
| Writer: deletion-vector (V3 puffin DV) | 🟡 | `core/.../deletes` | `delete_vector.rs` + `puffin/` (read solid; write side partial) |
| Writer: partitioning (fanout/clustered/unpartitioned) | ✅ | — | `writer/partitioning/` |
| Read: Parquet → Arrow | ✅ | `parquet/` | `arrow/reader.rs` |
| Read: merge-on-read apply (position-deletes + DVs during scan) | ✅ | `data/.../DeleteFilter` | `arrow/delete_filter.rs`, `arrow/caching_delete_file_loader.rs`, `delete_file_index.rs` (new in 0.8/0.9) |
| Read: ORC data files | ❌ | `orc/` | none |
| Read/write: Avro data files | ❌ | `core/.../avro` (data) | Avro is manifest-only here |
| Scan planning + partition pruning | ✅ | `api/TableScan.java` | `scan/` |
| Metrics evaluators (inclusive/strict) + residual evaluation | 🟡 | `expressions/` | `expr/visitors` (partial) |
| `IncrementalAppendScan` | ❌ | `api/IncrementalAppendScan.java` | none |
| `IncrementalChangelogScan` | ❌ | `api/IncrementalChangelogScan.java` | none |
| `BatchScan` | ❌ | `api/BatchScan.java` | none |
| Scan/commit metrics reporting (`ScanReport`, `MetricsReporter`) | ❌ | `metrics/` | none |
| Catalogs: REST, Hive, Glue, S3 Tables, SQL/JDBC, in-memory | ✅ | `core/.../{rest,jdbc,inmemory}`, `aws`, `hive-metastore` | `crates/catalog/*`, `catalog/memory` |
| `ViewCatalog` + view operations (create/replace/drop/list, versions) | 🟡 | `api/catalog/ViewCatalog.java`, `api/view/` | view metadata spec + builder (`spec/view_metadata*`, `view_version.rs`) and `ViewCreation`/`ViewUpdate` types in `catalog/mod.rs`; **no `ViewCatalog` trait / no catalog view ops** (REST/Glue/etc.) |
| `SessionCatalog` | ❌ | `api/catalog/SessionCatalog.java` | none |
| `LockManager` | 🟡 | `api/LockManager.java` | partial |
| Encryption (`EncryptionManager`, KMS, encrypted FileIO/manifests) | ❌ | `api/encryption/`, `core/.../encryption` | V3 `spec/encrypted_key.rs` stub only |
| FileIO (S3/GCS/Azure/OSS/fs/memory) | ✅ | `core/.../io`, cloud modules | `io/` + extracted `crates/storage/opendal` (OpenDAL) |
| Puffin read/write + blob types (theta NDV, DV) | 🟡 | `core/.../puffin`, `api/.../puffin` | `puffin/` (blob coverage partial) |
| Maintenance: `ExpireSnapshots` | ❌ | `api/actions/ExpireSnapshots.java` | none |
| Maintenance: `DeleteOrphanFiles` | ❌ | `api/actions/DeleteOrphanFiles.java` | none |
| Maintenance: `RewriteDataFiles` (compaction) | ❌ | `api/actions/RewriteDataFiles.java` | none |
| Maintenance: `RewritePositionDeleteFiles` | ❌ | `api/actions/RewritePositionDeleteFiles.java` | none |
| Maintenance: `RemoveDanglingDeleteFiles` | ❌ | `api/actions/RemoveDanglingDeleteFiles.java` | none |
| Maintenance: `ComputeTableStats` / `ComputePartitionStats` | ❌ | `api/actions/Compute*.java` | none |
| Maintenance: `SnapshotTable` / `MigrateTable` / `RewriteTablePath` | ❌ | `api/actions/` | none |
| Partition statistics (`UpdatePartitionStatistics`, `PartitionStatisticsScan`) | ❌ | `api/Partition*Statistics*.java` | table-level stats partial |
| Table-level statistics (`UpdateStatistics`) | ✅ | `api/UpdateStatistics.java` | `transaction/update_statistics.rs` |
| Metadata inspection tables | 🟡 | `core/.../*Table` (~15 variants) | `inspect/` has snapshots + manifests only |
| Name mapping (schema-less Parquet) | ✅ | `mapping/` | `spec/name_mapping/` |
| Events / listeners | ❌ | `api/events/`, `core/.../events` | none |
| Type utilities (prune/assign-ids/reassign/check-compat) | 🟡 | `types/TypeUtil.java` etc. | partial |

## Headline gaps (ranked by effort × value)

1. **Write engine** — everything beyond fast-append (`OverwriteFiles`, `ReplacePartitions`,
   `DeleteFiles`, `RowDelta`, `RewriteFiles`, `RewriteManifests`, merge append).
2. **Schema/partition evolution + snapshot management** — `UpdateSchema` is now **✅** (bidirectional
   Java interop round-trip landed 2026-06-07 via the `dev/java-interop/` oracle + the
   `crates/iceberg/tests/interop_update_schema.rs` Direction-1 tests; first interop-proven row of this
   group). `UpdatePartitionSpec` and `ManageSnapshots` remain 🟡 (landed with unit tests; **Java interop
   round-trip is the only thing left before ✅** for each — the UpdateSchema harness is the template).
   `UpdateSchema` column initial/write **defaults** are plumbed and `Schema::check_compatibility` (in
   `TableMetadataBuilder::add_schema`) now mirrors Java `Schema.checkCompatibility` in FULL — both the
   **V3-only initial-default guard** (rejects a non-null `initial_default` below v3) AND the **V3-only type
   gate** (rejects a `timestamp_ns`/`timestamptz_ns` field, incl. nested, below v3); `ManageSnapshots` now
   has `rollbackToTime` + non-positive-retention rejection (`cherrypick` is Phase-2-gated — it extends
   `MergingSnapshotProducer` / replays data files).
3. **Format & type breadth** — ORC + Avro data files; remaining V3 types (variant, geo, unknown). The
   V3-only **type** gate (Java `Schema.MIN_FORMAT_VERSIONS`: `timestamp_ns`/`variant`/`unknown`/`geometry`/
   `geography` require v3) is now **enforced for the representable types** (`timestamp_ns`/`timestamptz_ns`)
   in `Schema::check_compatibility`; `variant`/`unknown`/`geometry`/`geography` get a one-line
   `min_format_version` arm each when those types land (tracked in `task/todo.md`). (`timestamp_ns` and
   column default values already landed in the 0.8/0.9 base — see the matrix.)
4. **Views in catalogs** (`ViewCatalog` + view operations).
5. **Maintenance actions** (expire/orphan/compaction/rewrite-deletes/compute-stats/migrate).
6. **Encryption** (`EncryptionManager`, KMS, encrypted FileIO/manifests).
</content>
</invoke>
