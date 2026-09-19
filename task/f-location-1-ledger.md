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

# Ledger — F-LOCATION-1: Java location providers and `write.metadata.path`

**Ledger id:** `F-LOCATION-1`
**Branch:** `fix/f-location-1`
**RePark slate:** IPI-10
**Oracle:** the run-24d location oracle (Spark 4.1.2 + Iceberg runtime 1.11.0), keys
`locations` and `locations_inmemory`
**Model:** swe-2-high

| Step | Commit | Subject |
|---|---|---|
| 1 | this commit | `docs: F-LOCATION-1 — measurement ledger, Java rules, site inventory` |

## Java 1.11.0 rules (all bytecode-verified, `javap -c -p` on the spark-runtime jar)

### `LocationProviders.locationsFor(String location, Map properties)`

1. `location = LocationUtil.stripTrailingSlash(location)` FIRST — the table location is
   stripped before either provider sees it. `stripTrailingSlash` requires the input non-null
   and non-empty (`checkArgument`, `IllegalArgumentException`), then loops: while the string
   ends with `/` AND does not end with `://`, drop the last char. So `s3://b/` → `s3://b`,
   `s3://` → `s3://` (the `://` stop keeps a bare scheme+authority root intact).
2. If `properties` contains key `write.location-provider.impl` → the named Java class is
   instantiated reflectively (`DynConstructors`; two constructor shapes: `(String, Map)` or
   `()`); any miss throws `IllegalArgumentException`. **This takes precedence over
   `write.object-storage.enabled`.** Rust cannot instantiate a Java class name — this lane
   refuses the property loudly with a typed `FeatureUnsupported` error at generator
   resolution time (never a silent default fall-back).
3. `PropertyUtil.propertyAsBoolean(props, "write.object-storage.enabled", false)` →
   `ObjectStoreLocationProvider`; else `DefaultLocationProvider`. `propertyAsBoolean` is
   Java `Boolean.parseBoolean` on the raw string — only `equalsIgnoreCase("true")` is true.

### `LocationProviders$DefaultLocationProvider`

- `dataLocation = LocationUtil.stripTrailingSlash(dataLocation(properties, tableLocation))`.
- `dataLocation(properties, tableLocation)`: `getAndCheckLegacyLocation(write.data.path)` →
  if non-null return it; else `getAndCheckLegacyLocation(write.folder-storage.path)` →
  deprecated ⇒ if set, `IllegalArgumentException("Property 'write.folder-storage.path' has
  been deprecated and will be removed in 2.0.0, use 'write.data.path' instead.")`; else
  `tableLocation + "/data"`.
- Note the SHORT-CIRCUIT: when `write.data.path` is set, the deprecated
  `write.folder-storage.path` is never reached ⇒ no throw. The default provider never checks
  `write.object-storage.path` at all — that key is silently ignored on the default path.
- `newDataLocation(filename)` → `dataLocation + "/" + filename`.
- `newDataLocation(spec, data, filename)` → `dataLocation + "/" + partitionToPath + "/" + filename`.

### `LocationProviders$ObjectStoreLocationProvider`

- `storageLocation = LocationUtil.stripTrailingSlash(dataLocation(properties, tableLocation))`
  where its `dataLocation` checks, in order: `write.data.path` (non-deprecated → return if
  set), `write.object-storage.path` (deprecated → throw if set), `write.folder-storage.path`
  (deprecated → throw if set), else `tableLocation + "/data"`.
- `context`: `null` when `storageLocation.startsWith(tableLocation)` (the RAW, unstripped
  table location is the `startsWith` argument — the ctor receives the already-stripped
  `location` from `locationsFor`), else `pathContext(tableLocation)`.
- `pathContext(tableLocation)` = Hadoop `Path` semantics: `parent.getName() + "/" +
  path.getName()` when the parent exists, else `path.getName()` — i.e. the last two
  normalized path segments (`ns/t`), scheme and authority excluded; `checkState` the result
  does not end with `/`. Oracle confirms: table `{wh}/ns/l_object_storage_data_path`,
  `write.data.path={wh}/alt-data` ⇒ context `ns/l_object_storage_data_path`.
- `includePartitionPaths = propertyAsBoolean(props,
  "write.object-storage.partitioned-paths", true)` — DEFAULT TRUE.
- `computeHash(filename)` = `Hashing.murmur3_32_fixed().hashString(filename,
  UTF_8).asInt() | 0x80000000`, then `Integer.toBinaryString` (always exactly 32 chars
  because the sign bit is set), then the LAST 20 chars. `murmur3_32_fixed` is seed 0.
- `dirsFromHash(last20)`: chunks `hash[0..4]`, `hash[4..8]`, `hash[8..12]` joined by `/`,
  then if `len > 12` appends `/` + `hash[12..len]` — i.e. three 4-bit dirs + ONE trailing
  component holding the remaining 8 bits: `xxxx/xxxx/xxxx/xxxxxxxx`. (Not five 4-bit dirs.)
- `newDataLocation(filename)`:
  `context != null` → `{storage}/{hash}/{context}/{filename}`;
  else `includePartitionPaths` → `{storage}/{hash}/{filename}`;
  else `{storage}/{hash}-{filename}` (the hash string keeps its `/`-separated dirs).
- `newDataLocation(spec, data, filename)`:
  `includePartitionPaths` → `newDataLocation(partitionToPath + "/" + filename)` — the
  partition path is INSIDE the hashed name AND inside the emitted path;
  else → `newDataLocation(filename)` — partition path dropped entirely.

### `BaseMetastoreTableOperations` (metadata JSON path)

- `newTableMetadataFilePath(metadata, version)` →
  `metadataFileLocation(metadata, format("%05d-%s.metadata.json", version, UUID.randomUUID()))`;
  the Hadoop-convention variant uses `v{version}.metadata.json`.
- `metadataFileLocation(metadata, fileName)` (bytecode-verified):
  `String metadataLocation = metadata.properties().get("write.metadata.path");`
  `if (metadataLocation != null) return format("%s/%s",
  LocationUtil.stripTrailingSlash(metadataLocation), fileName);`
  `return format("%s/%s/%s", metadata.location(), "metadata", fileName)`.
  So `write.metadata.path` is a COMPLETE directory — `{stripTrailingSlash(prop)}/{file}` —
  never prefixed by the table location. Oracle confirms: `/alt-meta/<file>`.
- `SnapshotProducer` manifest files and manifest lists are written through
  `ops.metadataFileLocation(fileName)` → `<uuid>-m<n>.avro` and
  `snap-<snapshotId>-<attempt>-<uuid>.avro` land in the same metadata directory.

## The run-24d oracle cells

| Oracle key | Properties | Recorded data layout |
|---|---|---|
| `l_object_storage` | `write.object-storage.enabled=true`, unpartitioned | `{wh}/ns/t/data/1110/1011/1110/01000111/<file>` |
| `l_object_storage_p` | enabled, partitioned `cat` | `{wh}/ns/t/data/0001/1111/1111/10111111/cat=x/<file>` (hash over `cat=x/<file>`) |
| `l_object_storage_unpartitioned_paths` | enabled + `partitioned-paths=false`, unpartitioned | `{wh}/ns/t/data/0001/1001/0110/10000001-<file>` |
| `l_object_storage_unpartitioned_paths_p` | enabled + `partitioned-paths=false`, partitioned | `{wh}/ns/t/data/0110/1101/0111/11111000-<file>` (partition path dropped) |
| `l_object_storage_data_path` | enabled + `write.data.path={wh}/alt-data`, unpartitioned | `{wh}/alt-data/1111/0011/1100/11110011/ns/<table>/<file>` (context appended) |
| `l_object_storage_data_path_p` | enabled + data path, partitioned | `{wh}/alt-data/0101/1111/0100/00101010/ns/<table>/cat=x/<file>` |
| `l_data_path` | `write.data.path={wh}/alt-data2`, no object storage | `{wh}/alt-data2/<file>` |
| `l_data_path_p` | same, partitioned | `{wh}/alt-data2/cat=x/<file>` |
| `l_metadata_path*` | `write.metadata.path` cells | metadata files under the configured dir |
| `locations_inmemory.metadata_path` | `write.metadata.path=/alt-meta` | every `NNNNN-<uuid>.metadata.json`, `-mN.avro`, `snap-….avro` under `/alt-meta/`; data files still under the table location |

Hash-directory pins (verified by independent re-implementation of the Java algorithm
against every recorded file name — all match):

| File name (hashed input) | Dirs |
|---|---|
| `00000-28-3812449f-1cb2-488c-9cc3-8f96668c44cd-0-00001.parquet` | `1110/1011/1110/01000111` |
| `cat=x/00000-34-96eea29b-37d5-44d2-8f8e-790fbe133349-0-00001.parquet` | `0001/1111/1111/10111111` |
| `cat=y/00000-34-96eea29b-37d5-44d2-8f8e-790fbe133349-0-00002.parquet` | `1001/1011/1111/11111010` |
| `00000-39-afc298d7-6274-47c4-a229-c4bd98b470f7-0-00001.parquet` | `0001/1001/0110/10000001` |
| `00000-45-9205b7af-4125-4711-912d-a933510ba235-0-00001.parquet` | `0110/1101/0111/11111000` |
| `00000-45-9205b7af-4125-4711-912d-a933510ba235-0-00002.parquet` | `1010/0011/1100/11110001` |
| `00000-50-661c5ab6-de30-4594-858b-7f8d5fbc98e1-0-00001.parquet` | `1111/0011/1100/11110011` |
| `cat=x/00000-56-c7fb03ac-63e3-46aa-8da7-1608d002a1a4-0-00001.parquet` | `0101/1111/0100/00101010` |
| `cat=y/00000-56-c7fb03ac-63e3-46aa-8da7-1608d002a1a4-0-00002.parquet` | `0110/0010/1100/10100111` |

## Current fork behavior per oracle cell (measured, pre-change)

| Cell | Fork today | Status |
|---|---|---|
| object storage enabled (any shape) | `write.object-storage.enabled` is not read anywhere; `DefaultLocationGenerator` always emits `{data}/{partition?}/{file}` — NO hash dirs | DIVERGENT — PROVEN BELOW |
| `partitioned-paths=false` | property unread | DIVERGENT |
| `write.data.path` alone | honored by `DefaultLocationGenerator` (`{alt}/<file>`, `{alt}/cat=x/<file>`) — `location_generator.rs` | MATCHES already |
| object storage + `write.data.path` context | no context mechanism exists | DIVERGENT |
| `write.metadata.path` | property unread; all metadata under `{loc}/metadata` | DIVERGENT |
| `write.location-provider.impl` | property unread — silently ignored today | DIVERGENT (must become a loud typed error) |
| `write.folder-storage.path` | honored as fallback (pre-1.9 Java behavior; 1.11.0 throws) | RESIDUE — `DefaultLocationGenerator` keeps legacy leniency for existing callers; the table-property-resolved API rejects it as Java does |

## Every fork site that builds a data-file or metadata-file path

### Data-file paths (the `LocationGenerator` trait + `DefaultLocationGenerator`)

`crates/iceberg/src/writer/file_writer/location_generator.rs` —
`DefaultLocationGenerator::new(table_metadata)`; layout `{data}/{partition_path?}/{file}`;
`data` = `write.data.path` → `write.folder-storage.path` (lenient) → `{loc}/data`.

Production `DefaultLocationGenerator::new` call sites (all must route through the resolved
provider):

- `crates/iceberg/src/maintenance/rewrite_data_files_write.rs` (data-file rewrite output)
- `crates/iceberg/src/maintenance/rewrite_position_delete_files.rs` (`GroupWriteFactory`
  field type + construction)
- `crates/iceberg/src/maintenance/rewrite_position_delete_files_v3.rs` (V3 DV/pos-del writes)
- `crates/iceberg/src/maintenance/partition_key_audit.rs` (audit-driven writes)
- `crates/iceberg/src/maintenance/convert_equality_delete_files.rs`
- `crates/iceberg/src/delete_vector_container.rs` (DV blob file path)
- `crates/integrations/datafusion/src/physical_plan/write.rs` (`DmlDataFileWriterBuilder`)
- `crates/integrations/datafusion/src/physical_plan/row_lineage.rs` (`DmlRollingBuilder`
  type aliases)
- `crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs`
- `crates/iceberg/src/transaction/{row_delta,rewrite_manifests,rewrite_files,
  replace_partitions,overwrite_files,merge_append,delete_files}.rs` — via the writer
  builders used inside each action (all construct or receive a `DefaultLocationGenerator`)

Test/example call sites stay on `DefaultLocationGenerator` (the brief keeps it as the
preserved default-shaped generator).

### Metadata JSON paths

- `crates/iceberg/src/catalog/metadata_location.rs` — `MetadataLocation { table_location,
  version, id }` where `table_location` is really the directory (always `…/metadata`);
  `parse_metadata_path_prefix` REQUIRES a literal `/metadata` suffix; `with_next_version`,
  `new_with_table_location`, `Display`, `is_hadoop_convention`, `hadoop_version_siblings`.
- `crates/iceberg/src/catalog/memory/catalog.rs` — `create_table` (line ~415)
  `new_with_table_location(location)`; `create_view` (line ~674) same for views (views are
  out of scope — Java view ops do not consult `write.metadata.path`).
- `crates/iceberg/src/catalog/mod.rs` — `TableCommit::apply` (~665): parses the current
  pointer with strict `from_str`, `with_next_version`, writes via `write_commit_metadata`.
  Every metastore catalog update (memory, glue, s3tables, sql, hms, rest update paths)
  funnels here.
- `crates/iceberg/src/transaction/staged_table.rs` — `begin_create` (~109),
  `begin_replace` (~230: parse-or-fallback + `keeps_location`), `apply_locally` (~339),
  `hadoop_staged_location` (~405).
- `crates/iceberg/src/spec/table_metadata_commit.rs` — `is_hadoop_location` (~59),
  `is_hadoop_staged`/sibling logic (~77).
- `crates/iceberg/src/view.rs` (~281) — view metadata version advance (views: unchanged).
- `crates/catalog/hms/src/catalog.rs` (~593), `crates/catalog/glue/src/catalog.rs` (~728),
  `crates/catalog/s3tables/src/catalog.rs` (~629), `crates/catalog/sql/src/catalog.rs`
  (~1015 table create; ~1261 is a VIEW — unchanged).
- REST create — the create metadata file is staged server-side via `stage-create`;
  `update_table` funnels through `TableCommit::apply` (covered).

### Manifest + manifest-list paths

- `crates/iceberg/src/transaction/snapshot.rs` — `META_ROOT_PATH = "metadata"`;
  `new_cluster_manifest_writer` (~378): `{loc}/metadata/{uuid}-m{n}.avro`;
  `new_filtering_manifest_writer` (~1159): same shape;
  `generate_manifest_list_file_path` (~1317): `{loc}/metadata/snap-{id}-{attempt}-{uuid}.avro`.
  All three are `SnapshotProducer` sites — Java routes them through
  `ops.metadataFileLocation`, so `write.metadata.path` applies.

### Maintenance / special path sites

- `crates/iceberg/src/maintenance/rewrite_table_path.rs` — `StagedLocationGenerator`
  deliberately emits one exact staging path (path REWRITE action; its paths are inputs,
  not generated layout) — unchanged, documented here so the inventory is complete.
- `crates/iceberg/src/maintenance/remove_dangling_delete_files.rs`,
  orphan-file actions — they READ recorded paths, they do not build layout.

## Implementation plan (step 3 — recorded before writing, per red-first order)

1. `TableProperties` constants: `write.data.path`, `write.object-storage.enabled`,
   `write.object-storage.partitioned-paths`, `write.object-storage.path`,
   `write.folder-storage.path`, `write.metadata.path`, `write.location-provider.impl`.
2. `location_generator.rs`: keep `DefaultLocationGenerator` verbatim (existing callers,
   legacy leniency); add `ObjectStoreLocationGenerator` (Java-shaped: storageLocation,
   context, includePartitionPaths, `murmur3_32` hash via the existing `murmur3` dep);
   add `TableLocationGenerator` enum `Default | ObjectStore` with
   `TableLocationGenerator::new(&TableMetadata) -> Result<Self>` implementing Java's
   `locationsFor` precedence (`impl` → `FeatureUnsupported`; `object-storage.enabled` →
   object-store; deprecated-property checks per provider, Java short-circuit order).
3. `catalog/metadata_location.rs`: `MetadataLocation` field becomes `metadata_dir`;
  `from_file_path` (lenient parent) alongside strict `from_str`;
   `new_with_table_location_and_properties`; `rebased(table_location, properties)`;
   `write_metadata_dir`/`metadata_file_location` helpers (Java `metadataFileLocation`).
4. `catalog/mod.rs` `TableCommit::apply`: parse with `from_file_path`, `with_next_version`,
   `rebased` against the NEW metadata's location+properties.
5. `transaction/staged_table.rs`: same treatment at create/replace/apply_locally;
   `hadoop_staged_location` parses with `from_file_path`.
6. `spec/table_metadata_commit.rs`: hadoop-convention detection via `from_file_path`.
7. `snapshot.rs`: `metadata_file_location(metadata, name)` at the three manifest sites.
8. Every catalog `create_table`: `new_with_table_location_and_properties(location,
   metadata.properties())` — memory, hms, glue, s3tables, sql (view sites unchanged).
9. Wire `TableLocationGenerator::new` into the fork's own writers + maintenance
   (`rewrite_data_files_write`, `rewrite_position_delete_files{,_v3}`,
   `partition_key_audit`, `convert_equality_delete_files`, `delete_vector_container`,
   datafusion `write.rs`/`row_lineage.rs`/`delete_position_deletes.rs`).

## Proof status

| Proposition | Status |
|---|---|
| Java rules extracted from 1.11.0 bytecode | PROVEN (bytecode above) |
| Hash algorithm reproduces every oracle dir | PROVEN (independent check, 9/9 names) |
| Fork currently ignores the three properties | PROVEN (no readers exist) |
| Resolved provider answers every oracle data cell | OPEN — step 2/3 |
| `write.metadata.path` honored for JSON/manifests/manifest lists on the memory catalog | OPEN — step 2/3 |
| Relocated metadata pointer round-trips through `MetadataLocation` parsing | OPEN — step 2/3 |
| `write.location-provider.impl` refused with a typed error | OPEN — step 2/3 |
| Mutation pass (hash, flag, context, metadata path) drives pins red then green | OPEN — step 4 |
