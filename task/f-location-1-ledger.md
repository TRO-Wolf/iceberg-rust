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
   `for_metadata(&TableMetadata)` (Java `metadataFileLocation(newMetadata, …)` —
   the built metadata already carries location + properties, keeping every
   catalog call site line-neutral under the file-size ceilings);
   `rebased(&metadata)`; `write_metadata_dir`/`metadata_file_location` helpers.
4. `catalog/mod.rs` `TableCommit::apply`: parse with `from_file_path`, `with_next_version`,
   `rebased` against the NEW metadata's location+properties.
5. `transaction/staged_table.rs`: same treatment at create/replace/apply_locally;
   `hadoop_staged_location` parses with `from_file_path`. `keeps_location` retained:
   a relocated staged replace restarts at v0 under the new location (pre-existing
   fork rule, pinned by `replace_restarts_versioning_under_a_different_caller_location`).
6. `spec/table_metadata_commit.rs`: hadoop-convention detection via `from_file_path`.
7. `snapshot.rs`: `metadata_file_location(metadata, name)` at the three manifest sites.
8. Every catalog `create_table`: `MetadataLocation::for_metadata(&metadata)` —
   memory, hms, glue, s3tables, sql (view sites unchanged).
9. Wire `TableLocationGenerator::new` into the fork's own writers + maintenance
   (`rewrite_data_files_write`, `rewrite_position_delete_files{,_v3}`,
   `partition_key_audit`, `convert_equality_delete_files`, `delete_vector_container`,
   datafusion `write.rs`/`row_lineage.rs`/`delete_position_deletes.rs`).

## Implementation landed (step 3)

`TableLocationGenerator::new(&TableMetadata) -> Result<Self>` resolves the Java
provider: `write.location-provider.impl` present → `ErrorKind::FeatureUnsupported`
loud refusal; `write.object-storage.enabled=true` → `ObjectStoreLocationGenerator`;
otherwise `DefaultLocationGenerator` (kept public for existing callers and test
fixtures). Object-store paths are `storage/hash-dirs[/context][/partition]/file`
with `context` = last two table-location components when data storage is outside
the table location and suppressed under it; `partitioned-paths=false` yields
`storage/hash-file`. Deprecated `write.folder-storage.path` and (object-store arm)
`write.object-storage.path` are rejected unless `write.data.path` short-circuits
(Java constructor order). `MetadataLocation` now carries `metadata_dir` (the
`/metadata` suffix moved into the dir), `from_file_path` accepts ANY parent dir,
and every metadata-writing surface routes through `write_metadata_dir`:
`TableCommit::apply`, staged create/replace/`apply_locally`, the three
`SnapshotProducer` manifest/manifest-list sites, and all five catalog
`create_table` paths. Production call sites switched to the resolver: the five
maintenance writers, `delete_vector_container`, and the three DataFusion write
paths; test-only fixtures keep `DefaultLocationGenerator`. `memory/catalog.rs`
table create unchanged for views. Size ceilings ratcheted down in
`scripts/check_rust_file_size.py` for the five files that shrank.

## Proof status

| Proposition | Status |
|---|---|
| Java rules extracted from 1.11.0 bytecode | PROVEN (bytecode above) |
| Hash algorithm reproduces every oracle dir | PROVEN (independent check, 9/9 names) |
| Fork currently ignores the three properties | PROVEN (no readers exist) |
| Resolved provider answers every oracle data cell | PROVEN — `cargo test -p iceberg --lib location`, 14 `table_location_generator_*` pins green |
| `write.metadata.path` honored for JSON/manifests/manifest lists on the memory catalog | PROVEN — `write_metadata_path_relocates_metadata_files_on_memory_catalog` green |
| Relocated metadata pointer round-trips through `MetadataLocation` parsing | PROVEN — `from_file_path_accepts_relocated_metadata_dir`, `rebased_moves_dir_with_write_metadata_path` green |
| `write.location-provider.impl` refused with a typed error | PROVEN — `table_location_generator_refuses_java_provider_impl` green (`FeatureUnsupported`) |
| Mutation pass (hash, flag, context, metadata path) drives pins red then green | PROVEN — see below |
| No GAP_MATRIX row exists for location providers | PROVEN — grep of `docs/parity/GAP_MATRIX.md`; the ledger is this unit's home per the brief |

## Mutation evidence (step 4)

Each mutation applied alone, `cargo test -p iceberg --lib location_generator`
(or `write_metadata_path`) run, then reverted and confirmed green:

| Mutation | Pins driven red |
|---|---|
| murmur3 seed 0→1 in `compute_hash` | 8 red — every oracle-hash pin (`object_storage_unpartitioned`, `_partitioned`, `_unpartitioned_paths{,_partitioned}`, `_data_path_{unpartitioned,partitioned}`, `context_suppressed`, `rejects_deprecated_object_storage_path` short-circuit pin) |
| `partitioned-paths` default true→false | 4 red — `object_storage_partitioned`, `object_storage_data_path_partitioned`, `object_storage_unpartitioned` (its file name lost the partition segment so its pinned hash dirs moved), `context_suppressed` |
| `context` forced to `None` | 3 red — `object_storage_data_path_{unpartitioned,partitioned}` (no `ns/<table>` segment), `rejects_deprecated_object_storage_path` (context pin) |
| `generate_manifest_list_file_path` reverted to `{loc}/metadata/…` | 1 red — `write_metadata_path_relocates_metadata_files_on_memory_catalog` (manifest list escaped the relocated dir) |

All reverted; `cargo test -p iceberg --lib location` = 55/55 green.

## Gates

- `cargo fmt --all -- --check` — clean.
- `cargo clippy -p <crate> --all-targets -- -D warnings` — clean for iceberg,
  iceberg-datafusion, iceberg-catalog-{glue,hms,s3tables,sql}.
- `./scripts/check_rust_file_size.sh` — 522 files clean after ratcheting the
  five shrunk ceilings; `check_rust_file_size_test.py` 11/11.
- `./scripts/check_agent_artifacts.sh`, `check_matrix_anchors.sh`,
  `check_comment_blocks.sh` — all OK.
- `comment_ban.py` (the run-24d lane script) over `origin/main..HEAD` —
  `comment-ban hits=0`.
- `cargo test -p iceberg --lib location` — 55/55.
- `cargo test -p iceberg --test hadoop_version_commit` — 15/15.

## Round 2 — comment gate + rebase

- Three added doc-comment lines removed from `metadata_location.rs`
  (`for_metadata` gained `#[allow(missing_docs)]`; `from_file_path` is
  `pub(crate)` and needs no allowance). Gate now `comment-ban hits=0` over
  `origin/main..HEAD`.
- Rebased onto origin/main (`466bdbc7`, carries fork #308 F-AVRO-NAME-1 and
  F-TS-PUSHDOWN-1). One conflict: `scripts/check_rust_file_size.py`
  LEGACY_CEILINGS — kept the LOWER value per path (s3tables 1402, sql 3946,
  arrow/avro_reader 1255). Size checker clean post-merge (531 files).
- Post-rebase: `cargo test -p iceberg --lib location` 55/55;
  `hadoop_version_commit` 15/15; catalog lib suites glue 50, hms 48,
  s3tables 39, sql 81, datafusion 292 — all green. fmt, per-crate clippy,
  `typos .`, artifact/matrix/comment scripts all clean.

## Round 3 — logic P2s + perf (fork #314)

### L-001 — Hadoop `Path` context parity

`javap -c` on `LocationProviders$ObjectStoreLocationProvider.pathContext`
returns `parent.getName() + "/" + getName()` when a parent exists, else
`getName()`. Hadoop `Path` semantics (verified against hadoop-common
3.3.6): `s3://bucket/mytable` → name=`mytable`, parent=`s3://bucket/`
whose name is the empty string, so Java's context is `/mytable` and the
data path carries `//` before the table segment; `s3://bucket` (and
`s3://bucket/`) → name empty, parent null, context empty; `mytable` →
parent is the EMPTY path (not null), so the context is again `/mytable`;
multi-segment locations take the last two segments. `path_context` now
splits on `/`, drops empties, and yields `""` / `/{seg}` /
`{parent}/{name}` for 0 / 1 / 2+ segments. Pins:
`object_storage_context_{bucket_root_parent,single_segment_relative,bucket_only}`
plus the trailing-slash cases in `object_storage_unpartitioned` (the
`s3://bucket/mytable` and `mytable` cases pin the exact oracle string
`s3://alt-data/0111/1111/1110/11001100//mytable/f.parquet`).

### L-002 — S3 Tables owns the warehouse

Java S3 Tables catalog behavior: **UNMEASURED** — no S3 Tables catalog
jar exists under the run-24d ivy cache, so no bytecode evidence. The
offline Rust contract is pinned instead: the service generates the table
warehouse location, so (a) `create_table` refuses ANY
`write.metadata.path` / `write.data.path` creation property (nothing can
be verified under a warehouse that does not exist yet), and (b)
`update_table` / `publish_replace_table` refuse the staged metadata
location and any configured `write.metadata.path` / `write.data.path`
that is not `{warehouse}/`-prefixed — all typed `ErrorKind::DataInvalid`,
all before any metadata write or metadata-pointer CAS. Glue keeps
honoring external paths. Helpers `under_warehouse` /
`ensure_write_paths_under_warehouse` / `ensure_no_write_path_override`
live in `s3tables/src/utils.rs`; the four pins (create refusal, update
refusal, replace refusal, under-warehouse acceptance asserting no CAS on
refusal) live in `commit_outcome_tests.rs`.

### L-003 — `rewrite_table_path` loud failure pin

`relativize` already fails `DataInvalid` ("does not start with") when a
path escapes the source prefix; the new end-to-end pin
`execute_fails_loud_when_write_metadata_path_is_outside_the_source_prefix`
drives a full rewrite where `write.metadata.path` placed manifests and
the manifest list outside the table location and asserts the loud error
(never a silent rewrite).

### R-01 — object-store path generation allocs

`generate_location` no longer builds `file_name.to_string()` in the
unpartitioned arm nor a 32-char binary `String` for the hash: the 20
hash bits are written straight into the final path buffer via
`std::fmt::Write`. Hash dirs remain byte-identical to the oracle pins.

### R-02 — cached metadata dir on `SnapshotProducer`

`SnapshotProducer::new` resolves `write_metadata_dir` once (now returns
`Result<Self>`); both manifest writers and the manifest-list path use
the cached `metadata_dir` via `new_manifest_path` /
`generate_manifest_list_file_path`. The dead `metadata_file_location`
free fn was removed. `snapshot.rs` held at exactly its 3350 ceiling.

### R-03 — `DefaultLocationGenerator::new` borrows metadata

Signature is `impl Borrow<TableMetadata>`: every caller passes
`&TableMetadata` with no clone, AND the three `//!` doc-test lines in
`writer/mod.rs` that pass an owned `table.metadata().clone()` still
compile verbatim — required because any touched `//!` line is a
comment-ban hit (`&TableMetadata`-only would have forced edits onto
comment lines). ~50 call sites updated.

### R-04 — one generator per `execute`

`convert_equality_delete_files` builds its `TableLocationGenerator` once
in `execute` and threads `&TableLocationGenerator` through
`materialize_one` → `write_position_delete_file` (cloned only at the
rolling-writer builder) instead of one construction per equality delete.

### R-05 — `rebased` without struct clone

`MetadataLocation::rebased` now copies `version` and `id` explicitly and
recomputes only `metadata_dir` — no `..self.clone()`.

### Round-3 mutations

| Mutation | Pins driven red |
|---|---|
| `path_context` 1-segment arm → no leading slash | 2 red — `context_bucket_root_parent`, `context_single_segment_relative` |
| `under_warehouse` → always true | 2 red — `update_table_refuses_…`, `publish_replace_table_refuses_…` |
| `ensure_no_write_path_override` → no-op | 1 red — `create_table_refuses_…` |
| `relativize` prefix check → passthrough | 2 red — `execute_fails_loud_…_outside_the_source_prefix`, `relativize_errors_when_path_not_under_prefix` |

All reverted; filtered suites green again.

### Round-3 gates

- `cargo fmt --all -- --check` — clean.
- `cargo clippy -p iceberg -p iceberg-datafusion -p iceberg-catalog-s3tables
  -p iceberg-integration-tests --all-targets -- -D warnings` — clean.
- `./scripts/check_rust_file_size.sh` — 531 files clean; four shrunk test
  ceilings ratcheted down (interop_remove_dangling 1021→1018,
  interop_scan_exec 2592→2585, interop_scan_plan 1028→1027,
  interop_write_data 2115→2111); `snapshot.rs` held at 3350,
  `s3tables/catalog.rs` at 1402, `rewrite_table_path_tests.rs` at 1000.
- `comment_ban.py` over `origin/main..HEAD` — `comment-ban hits=0`.
- `./scripts/check_agent_artifacts.sh`, `check_matrix_anchors.sh`,
  `check_comment_blocks.sh`, `typos .` — all clean.
- `cargo test -p iceberg --lib -- location convert_equality
  rewrite_table_path` — 84/84; `cargo test -p iceberg-catalog-s3tables`
  — 43/43 + register_table + doctest.

## Round 4 — V-001: `rebased` dropped the hadoop pointer's directory

### Root cause

`MetadataLocation::rebased` routed every convention through
`write_metadata_dir`, so absent `write.metadata.path` the next file
always landed under `{metadata.location()}/metadata`. A Hadoop-named
`vN` pointer whose own directory differs from `metadata.location()/metadata`
(a `register_table` pointer, e.g. `seed_hadoop_v2`: v2 registered under
`{wh}/sales/orders/metadata` while the JSON `location` is
`{wh}/sales/seed`) lost its directory — `origin/main`'s `with_next_version`
kept the parsed pointer dir for every convention.

### Java 1.11.0 bytecode (spark-runtime 4.1_2.13-1.11.0.jar, `javap -c -p`)

- `HadoopTableOperations.commit`: offsets 71–92 —
  `checkArgument(!metadata.properties().containsKey("write.metadata.path"),
  "Hadoop path-based tables cannot relocate metadata")`. Hadoop path
  tables REFUSE `write.metadata.path` outright (and refuse any
  `metadata.location()` change at offsets 48–68, "cannot be relocated").
- `HadoopTableOperations.metadataRoot()` = `new Path(this.location,
  "metadata")` — `this.location` is the constructor's loaded-from table
  path, NOT `metadata.location()`. `metadataFilePath(version, codec)` =
  `{metadataRoot}/v{version}{ext}` — new metadata lands under the
  loaded-path root's `metadata/` = the fork's pointer directory.
- `BaseMetastoreTableOperations.metadataFileLocation(metadata, fileName)`
  re-verified: `write.metadata.path` → `{stripTrailingSlash(prop)}/{file}`;
  else `{metadata.location()}/metadata/{file}`.

### Fork rule after the fix (`MetadataLocation::rebased`)

- `write.metadata.path` set → `{strip(prop)}` for EVERY convention.
  Brief-mandated extension of the property to hadoop-convention pointers,
  where Java's `HadoopTableOperations` refuses the property — named
  residue (the fork's staged/metastore paths have no Java hadoop-ops
  analogue that could honor it; refusing would break the
  `write.metadata.path`-every-catalog contract).
- Hadoop convention (`id == None`), property absent → keep the pointer's
  own `metadata_dir`. Java: `HadoopTableOperations` writes under the
  loaded-from root's `metadata/`; restores `origin/main` behavior.
- Uuid convention, property absent → `{metadata.location()}/metadata`.
  Java `metadataFileLocation` else-arm — this is the metastore rule, so
  the "unless Java says otherwise" clause fires over the
  keep-the-pointer-dir default; the two differ only for a relocated uuid
  pointer, where `metadata.location()/metadata` is what Java writes.

### Per-catalog effect (all update paths funnel through `TableCommit::apply` → `rebased`)

- Memory / Glue / SQL / HMS / REST / S3 Tables `update_table`: uuid
  pointers write under `metadata.location()/metadata` absent the property
  (Java metastore parity; identical to the pointer dir whenever the
  pointer sits at the standard place); registered hadoop pointers keep
  their own dir (fork's hadoop-pointer feature, Java `HadoopTableOperations`
  semantics).
- `StagedTableTransaction::begin_replace` keeps-location path and
  `Transaction::apply_locally`: same rule — the two V-001 tests go green.
- Catalog `create_table` paths use `for_metadata` (unchanged); view
  version advance untouched; `for_metadata` and the three
  `SnapshotProducer` manifest sites still route through
  `write_metadata_dir` unconditionally.

### Pins and mutation evidence

- New unit pin `rebased_keeps_pointer_dir_for_hadoop_convention`:
  hadoop + no property → `/wh/sales/orders/metadata/v3.metadata.json`
  (pointer dir, not the seed location); hadoop + property →
  `/alt-meta/v3.metadata.json`.
- Existing pins cover the other arms:
  `rebased_moves_dir_with_write_metadata_path` (uuid + property →
  `/alt-meta/00001-*`; uuid + no property + moved location →
  `/wh/moved/metadata/00001-*`).
- Mutation: `rebased` reverted to unconditional `write_metadata_dir` →
  3 red out of 46 (`rebased_keeps_pointer_dir_for_hadoop_convention`,
  `hadoop_replace_with_files_publishes_only_next_version`,
  `hadoop_replace_without_files_publishes_only_next_version`); restored →
  46/46. The two end-to-end tests were already red on this head before
  the fix — the bug state is the mutation.

### Round-4 gates

- `cargo test -p iceberg --lib -- location_generator metadata_location
  snapshot staged_table rewrite_table_path catalog::memory` — 423/423.
- `cargo test -p iceberg-catalog-s3tables --lib --test register_table` —
  43/43 + 1/1.
- `cargo test -p iceberg-catalog-glue -p iceberg-catalog-hms
  -p iceberg-catalog-sql --lib` — 50/50, 48/48, 81/81.
- `cargo fmt --all -- --check` — clean.
- `cargo clippy -p iceberg --all-targets --all-features -- -D warnings` —
  clean (only `crates/iceberg` touched).
- `./scripts/check_rust_file_size.sh` — 539 files clean
  (`metadata_location.rs` 848 < 1000 default).
- `./scripts/check_agent_artifacts.sh`, `check_matrix_anchors.sh`,
  `check_comment_blocks.sh`, `typos .` — all clean.
- `comment_ban.py` over `origin/main..HEAD` — `comment-ban hits=0`.
