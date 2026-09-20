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

# F-ADD-FILES-1 (IPI-30) — the `add_files` action

The unit ledger. Unit: the FORK half of `CALL system.add_files` (gap-matrix row IPI-30 —
`P-ADD-FILES-PARTITIONED`, `-UNPARTITIONED`, `-PARTITION-FILTER`, `-CHECK-DUP`, `-PARALLELISM`).
The RePark CALL router is a later unit; the API here is shaped so a router passes Spark's five
procedure arguments straight through.

Every fact below is MEASURED — either against the recorded Spark add_files oracle (Spark 4.1 +
Iceberg 1.11.0 runtime, hadoop catalog, `local[1]`, 13 cells) or read out of the Iceberg 1.11.0
runtime jar with `javap -c -p`. Nothing here is recalled.

## 1. The oracle cells

Source tables, both written by Spark as plain parquet (no Iceberg field ids in the files):

| Source | Shape | Layout |
|---|---|---|
| `srcdb.flat_src` | `(id BIGINT, v STRING)` | one file, `.../srcdb.db/flat_src/part-...snappy.parquet`, 2 rows |
| `srcdb.part_src` | `(id BIGINT, v STRING, cat STRING)` PARTITIONED BY `cat` | three hive dirs `cat=x` (2 rows), `cat=y` (1), `cat=z` (1) |

Targets are Iceberg tables under a hadoop catalog `sc.ns.*`, v2 unless the cell says v3.

| # | Cell | CALL argument | Result row | Outcome |
|---|---|---|---|---|
| 1 | PARTITIONED | `source_table => 'srcdb.part_src'` | `added_files_count=3`, `changed_partition_count=NULL` | 3 files adopted, one per hive dir, partition values parsed from the dir names |
| 2 | UNPARTITIONED | `source_table => 'srcdb.flat_src'` | `1`, `1` | 1 file adopted |
| 3 | PARTITION-FILTER | `+ partition_filter => map('cat','x')` | `1`, `NULL` | only `cat=x` adopted; 2 rows visible |
| 4 | PARALLELISM | `+ parallelism => 2` | `3`, `NULL` | identical to cell 1 |
| 5 | CHECK-DUP-TRUE-TWICE | `+ check_duplicate_files => true`, run twice | `3`,`NULL` then ERROR | 2nd run: `java.lang.IllegalStateException`, "Cannot complete import because data files to be imported already exist within the target table: `<paths>`" |
| 6 | CHECK-DUP-FALSE-TWICE | `+ check_duplicate_files => false`, run twice | `3`,`NULL` twice | 6 file entries, every row duplicated, `total-data-files=6` |
| 7 | FILE-PATH-PARQUET | ``source_table => '`parquet`.`<flat_loc>`'`` | `1`, `NULL` | same as cell 2 except `changed_partition_count` |
| 8 | FILE-PATH-PARQUET-PARTITIONED | ``source_table => '`parquet`.`<part_loc>`'`` | `3`, `NULL` | same as cell 1 |
| 9 | PARTITION-FILTER-ON-UNPARTITIONED | `flat_src + partition_filter => map('cat','x')` | ERROR | `IllegalArgumentException`, "Cannot use partition filter with an unpartitioned table sc.ns.a_partition_filter_on_unpartitioned". Table left EMPTY (no snapshot) |
| 10 | SCHEMA-MISMATCH | target `(id BIGINT, other STRING)`, source `(id, v)` | `1`, `1` | SUCCEEDS. Rows read back `(10, NULL)`, `(11, NULL)` — the source column `v` is SILENTLY DROPPED |
| 11 | INTO-NONEMPTY | target seeded with one row | `1`, `1` | 2nd snapshot `added-records=2`, `total-records=3`, `total-data-files=2`; the seed file is `in_table_location=true`, the adopted file is not |
| 12 | MISSING-SOURCE | `source_table => 'srcdb.nosuch'` | ERROR | `org.apache.iceberg.exceptions.NoSuchTableException`, "Table \`srcdb\`.\`nosuch\` does not exist" (`SparkTableUtil.importSparkTable:594`) |
| 13 | V3-TARGET | `format-version=3` | `1`, `1` | identical to cell 2 |

Two facts hold across EVERY successful cell:

- **Adoption is IN PLACE.** Every adopted file's `file_path` is still its source path
  (`in_table_location=false`). Nothing is copied, rewritten or moved.
- **One `append` snapshot per call.** Every cell's snapshot `operation` is `append`.

### 1a. Why `changed_partition_count` is NULL on the partitioned cells

Measured, not guessed. `AddFilesProcedure.changedPartitionCount(Map)` is

```
0: aload_1  1: ldc #227 // String changed-partition-count
3: invokestatic #231 // PropertyUtil.propertyAsNullableLong(Map, String)
```

— it simply reads the snapshot summary key `changed-partition-count`, and returns `null` when the
key is absent. `addedFilesCount` reads `added-data-files` with default `0`.

The key is absent on the partitioned cells because the two import paths COMMIT DIFFERENTLY:

| Path | Java | Commit shape | Summary |
|---|---|---|---|
| unpartitioned | `SparkTableUtil.importUnpartitionedSparkTable` | `Table.newAppend()`, `files.forEach(append::appendFile)`, `commit()` | full — `added-files-size=686`, `changed-partition-count=1` |
| partitioned | `SparkTableUtil.importSparkPartitions` | executors write MANIFEST files, then `forEach(append::appendManifest)`, `commit()` | partial — NO `added-files-size`, `total-files-size=0`, NO `changed-partition-count` |

`importSparkPartitions` bytecode at 428–457: `Table.newAppend()` → `List<ManifestFile>.forEach(append::appendManifest)` → `commit()`. Appending a MANIFEST (rather than files) makes Java's
snapshot summary distrust the size/partition metrics, so those keys never reach the summary and the
procedure's second output column is `null`. This is a **Spark-layer artifact of `appendManifest`,
not a core-library rule** — see D-9.

## 2. The Java evidence (`javap -c -p`, iceberg-spark-runtime-4.1_2.13-1.11.0.jar)

### 2a. `AddFilesProcedure`

- Parameters and defaults (`static {}` at 0–94): `table` (required string), `source_table`
  (required string), `partition_filter` (optional string map), `check_duplicate_files` (optional
  boolean), `parallelism` (optional int). `call()` reads `check_duplicate_files` with default
  `Boolean.TRUE` (`iconst_1` at 70) and `parallelism` with default `1` (`iconst_1` at 86), then
  `checkArgument(parallelism > 0, "Parallelism should be larger than 0")`.
- `OUTPUT_TYPE` is exactly two `LongType` columns: `added_files_count` (non-nullable) and
  `changed_partition_count` (nullable).
- `lambda$importToIceberg$0` at 0: `ensureNameMappingPresent(table)` runs FIRST, before any import.
- `isFileIdentifier(ident)`: namespace length 1 and the single element `equalsIgnoreCase` one of
  `orc` / `parquet` / `avro` → the `` `parquet`.`<path>` `` form; otherwise a catalog table.
- `ensureNameMappingPresent` at 0–52: when `table.properties().get("schema.name-mapping.default")`
  is NULL, build `MappingUtil.create(table.schema())`, serialize with `NameMappingParser.toJson`,
  and `table.updateProperties().set("schema.name-mapping.default", json).commit()`. **A separate
  commit, before the append.** When the property is present it is left alone.
- `importFileTable` (the `` `parquet`.`path` `` form): infer the source's partition columns with
  `Spark3Util.getInferredSpec`, `SparkTableUtil.findCompatibleSpec(names, table)`,
  `SparkTableUtil.validatePartitionFilter(spec, filter, table.name())`, then
  `Spark3Util.getPartitions(...)`. On an UNPARTITIONED target it additionally asserts
  "Cannot add partitioned files to an unpartitioned table" and "Cannot use a partition filter when
  importingto an unpartitioned table" (sic — Java's own missing space), and imports ONE
  `SparkPartition(emptyMap, path, format)`. On a partitioned target:
  `checkArgument(!partitions.isEmpty(), "Cannot find any matching partitions in table %s")`.

### 2b. `SparkTableUtil`

- `validatePartitionFilter(spec, filter, tableName)` — three refusals, in this order:
  1. partitioned AND a filter: `checkArgument(spec.fields().size() >= filter.size(),
     "Cannot add data files to target table %s because that table is partitioned, but the number of
     columns in the provided partition filter (%s) is greater than the number of partitioned
     columns in table (%s)")`.
  2. then `checkArgument(<every filter key is a partition field name>, "Cannot add files to target
     table %s. %s is partitioned but the specified partition filter refers to columns that are not
     partitioned: %s . Valid partition columns: [%s]")`.
  3. NOT partitioned: `checkArgument(filter.isEmpty(), "Cannot use partition filter with an
     unpartitioned table %s")` — cell 9's message, verbatim.
- `findCompatibleSpec(List<String> sourceNames, Table)` — iterate `table.specs().values()`; a spec
  matches when EVERY field's `transform().isIdentity()` AND the field names lowercased
  (`Locale.ROOT`) equal the source names lowercased, IN ORDER. No match →
  `IllegalArgumentException("Cannot find a partition spec in Iceberg table %s that matches the
  partition columns (%s) in input table")`.
- `filterPartitions(partitions, filter)` — empty filter returns everything; otherwise keep a
  partition when `partition.getValues().entrySet().containsAll(filter.entrySet())`, i.e. exact
  STRING equality on both key and value.
- The duplicate check (identical in `importUnpartitionedSparkTable` at 108–247 and
  `importSparkPartitions` at 158–294): take the candidate file paths, join them against the
  target's `ENTRIES` metadata table filtered `status != 2` (every non-DELETED entry) on
  `data_file.file_path`, and
  `checkState(join.isEmpty(), DUPLICATE_FILE_MESSAGE, Joiner.on(",").join(join.take(10)))`.
  `checkState` → `java.lang.IllegalStateException`, matching cell 5. The message constant is

  > Cannot complete import because data files to be imported already exist within the target table:
  > %s.  This is disabled by default as Iceberg is not designed for multiple references to the same
  > file within the same table.  If you are sure, you may set 'check_duplicate_files' to false to
  > force the import.

  (two spaces after each sentence, as in the constant). **At most TEN paths are listed** (`take(10)`).
- Both paths read `MetricsConfig.forTable(table)` and
  `NameMappingParser.fromJson(properties.get("schema.name-mapping.default"))` (null when absent)
  and hand both to `TableMigrationUtil.listPartition`.

### 2c. `TableMigrationUtil.listPartition(Map values, String uri, String format, PartitionSpec spec, Configuration, MetricsConfig, NameMapping, ExecutorService)`

- Partition values: `spec.fields().stream().map(PartitionField::name).map(values::get).toList()` —
  for each SPEC FIELD, look the field's NAME up in the partition map. Order follows the SPEC.
- Listing: `new Path(uri).getFileSystem(conf).listStatus(path, HIDDEN_PATH_FILTER)` then
  `filter(FileStatus::isFile)`. `HIDDEN_PATH_FILTER` (`lambda$static$0`) rejects a path whose NAME
  starts with `_` or `.`. `listStatus` is ONE level — not recursive.
- Format dispatch is `format.contains("avro"|"parquet"|"orc")`, else
  `UnsupportedOperationException("Unknown partition format: %s")`.
- Concurrency: `Tasks.range(n).stopOnFailure().throwFailureWhenFinished()` plus
  `.executeWith(service)` when non-null. `migrationService(1)` returns NULL (no pool) —
  `parallelism => 1` is serial, and the pool is `ThreadPools.newFixedThreadPool("table-migration", n)`
  otherwise. The service is shut down on both exits.
- Parquet metrics: `ParquetUtil.fileMetrics(inputFile, metricsConfig, nameMapping)`.
- `buildDataFile(status, partitionValues, spec, metrics, format)`:
  `DataFiles.builder(spec).withPath(status.getPath().toString()).withFormat(format)
  .withFileSizeInBytes(status.getLen()).withMetrics(metrics).withPartitionValues(values).build()`.
  **No `withSplitOffsets`. No `withSortOrder`/`withSortOrderId`.**

### 2d. `DataFiles$Builder`

- Constructor defaults (41–131): `splitOffsets = null`, `sortOrderId =
  SortOrder.unsorted().orderId()` = **0**, `recordCount = -1`, `fileSizeInBytes = -1`.
- `build()` (522) asserts path / format / `fileSizeInBytes >= 0` / `recordCount >= 0`, then
  constructs `GenericDataFile` with `splitOffsets` and `sortOrderId` as they stand.
  **So every adopted file carries `sort_order_id = 0` and NO split offsets.**
- `withPartitionValues(List<String>)` → `DataFiles.fillFromValues(spec, values, data)`, which
  requires `values.size() == spec.fields().size()` ("Invalid partition data, expecting %s fields,
  found %s") and sets each position from
  `Conversions.fromPartitionString(partitionData.getType(i), values.get(i))`.

### 2e. `Conversions.fromPartitionString(Type, String)` — the hive value parser

```
0: aload_1  1: ifnull 13   4: ldc #35 // String __HIVE_DEFAULT_PARTITION__
6: invokevirtual String.equals  10: ifeq 15   13: aconst_null  14: areturn
```

- `null` OR the literal `__HIVE_DEFAULT_PARTITION__` → **NULL partition value**. This is Java's
  null-partition encoding, and the answer to the brief's "measure whether Java has one".
- Otherwise by type: BOOLEAN `Boolean.valueOf`; INTEGER `Integer.valueOf`; LONG `Long.valueOf`;
  FLOAT `Float.valueOf`; DOUBLE `Double.valueOf`; STRING as-is; UUID `UUID.fromString`; FIXED
  `Arrays.copyOf(utf8Bytes, length)`; BINARY utf8 bytes; DECIMAL `new BigDecimal(s)`; DATE
  `Literal.of(s).to(DateType)`.
- **Every other type throws** `UnsupportedOperationException` — notably TIME, TIMESTAMP and
  TIMESTAMPTZ have no hive-string parse.

### 2f. `ParquetUtil` — where the field ids come from

`footerMetrics` calls `getParquetTypeWithIds(footer, nameMapping)` then
`ParquetSchemaUtil.convertAndPrune` and computes the metrics against THAT (file-derived) schema:

```
8: ParquetSchemaUtil.hasIds(fileSchema)  12: ifeq 17   15: aload_2  16: areturn   // ids win
17: aload_1  18: ifnull 27   23: ParquetSchemaUtil.applyNameMapping(fileSchema, nameMapping)
27: aload_2  28: ParquetSchemaUtil.addFallbackIds(fileSchema)
```

Three branches, in order: **(1)** the file's own embedded ids, **(2)** the table's name mapping,
**(3)** positional fallback ids. `hasIds` (`ParquetSchemaUtil$HasIds`) is true when ANY field at
ANY depth carries an id — it is a file-level all-or-nothing switch. `convertAndPrune` DROPS every
column that came out without an id.

This is the mechanism behind cell 10's silent column drop: the name mapping built from the target
`(id, other)` names `id`→1 and `other`→2; applied to the source file `(id, v)`, `v` resolves to no
id and is pruned, so the adopted file carries `id` only and `other` reads back NULL.

The fork's READ path already implements the same three branches at TOP LEVEL
(`crates/iceberg/src/arrow/reader.rs`, `apply_name_mapping_to_arrow_schema` /
`add_fallback_field_ids_to_arrow_schema`).

### 2g. `MappingUtil.create(Schema)` — `MappingUtil$CreateMapping`

- `struct`: for each field i, `MappedField.of(field.fieldId(), field.name(), nested_i)`.
- `list`: one `MappedField.of(elementId, "element", nested)`.
- `map`: `MappedField.of(keyId, "key", …)` and `MappedField.of(valueId, "value", …)`.
- `variant` and `primitive`: `null` (no nested mapping).

Each mapped field carries exactly ONE name.

## 3. What the fork has today

| Needed | Present? | Where |
|---|---|---|
| `MetricsConfig::for_table` | yes | `crates/iceberg/src/spec/metrics_config.rs:323` |
| per-column metrics modes by field id | yes | `MetricsByFieldId` (`pub(crate)`), same file |
| metrics from a parquet footer → `DataFileBuilder` | yes | `ParquetWriter::parquet_to_data_file_builder` (`pub(crate)`), `writer/file_writer/parquet_writer.rs:489` |
| `NameMapping` / `MappedField` + the Java JSON shape | yes | `spec/name_mapping/mod.rs` |
| build a `NameMapping` FROM a schema (`MappingUtil.create`) | **no** | this unit adds it |
| recursive prefix listing with sizes through `FileIO` | yes | `FileIO::list` (`io/file_io.rs:206`), recursive, files only, carries `size` |
| one `append` commit | yes | `Transaction::merge_append` (`Table.newAppend()` is `MergeAppend` in Java) |
| set a table property in its own commit | yes | `Transaction::update_properties` |
| an `add_files` action | **no** | this unit adds it |

`ParquetWriter::parquet_files_to_data_files` exists but is `#[allow(dead_code)]`, unpartitioned-only
("TODO: support adding to partitioned table"), ignores the name mapping, and builds no commit. It is
not the action and is left untouched.

## 4. Design decisions

**D-1 — a free-standing action, not an `ActionsProvider` method.** Java's `ActionsProvider` surface
is twelve methods (javap-confirmed, recorded in `maintenance/actions_provider.rs`) and `add_files`
is not one of them: it is a Spark PROCEDURE delegating to `SparkTableUtil`. So `AddFiles` follows
`ConvertEqualityDeleteFiles`: `AddFiles::new(table)` … `.execute(catalog)`, no provider entry.

**D-2 — the source is `(path, partition values)` pairs, however they were discovered.** Java's two
source shapes both reduce to a list of `SparkPartition(values, uri, format)`: a catalog table takes
its values from the metastore, a `` `parquet`.`<path>` `` identifier takes them from Spark's
directory inference. The fork therefore exposes

- `AddFilesSource::Directory(String)` — list the prefix recursively and parse hive-style `k=v`
  path segments, and
- `AddFilesSource::Files(Vec<AddFilesEntry>)` — explicit paths, each with its own partition map
  (empty for an unpartitioned target),

so a RePark router hands `source_table` to whichever it already resolved, and neither shape needs a
Spark session.

**D-3 — field ids follow Java's three branches, at top level.** `hasIds`(file) → the file's ids;
else the table's `schema.name-mapping.default`; else positional 1-based fallback. Resolution is
TOP-LEVEL, matching the fork's existing read path (`reader.rs`) — a nested column takes its id from
the table's own field under the resolved top-level id. **Divergence D-3a:** Java's `convertAndPrune`
keeps a resolved id the TABLE schema does not have (converting the parquet type); the fork drops it,
because a bound keyed by a foreign id is unreadable by any scan of this table and the fork has no
type to interpret it with. Recorded, not silent.

**D-4 — metrics come from the footer under `MetricsConfig::for_table`, keyed by the FILE's column
names.** Java computes them against the file-derived schema, so `write.metadata.metrics.column.<name>`
is matched on the FILE's names, not the target's. The fork builds a resolved schema carrying the
file's top-level names with the resolved ids and the table's types, then reuses
`ParquetWriter::parquet_to_data_file_builder`.

**D-5 — `file_size_in_bytes` from the listing, `record_count` from the footer, `sort_order_id = 0`,
no split offsets.** Measured in 2c/2d: `DataFiles$Builder` defaults `sortOrderId` to
`SortOrder.unsorted().orderId()` and never sets `splitOffsets` on this path. The fork does not
"improve" on that — an adopted file that claims a sort order it was not written in would corrupt a
later sort-aware rewrite.

**D-6 — partition values parse through Java's `fromPartitionString`,
`__HIVE_DEFAULT_PARTITION__` included.** Per-type parse as in 2e, with `null` for the hive default
sentinel, and a typed refusal for TIME/TIMESTAMP/TIMESTAMPTZ, which Java refuses too.

**D-7 — the partition spec is chosen by Java's `findCompatibleSpec`.** All-identity, names
lowercased and compared IN ORDER against the source's partition column names; the first table spec
that matches wins; no match is the Java message. This means an adopted file may land on a
NON-default spec id, exactly as Java allows.

**D-8 — the duplicate check reads the live manifest entries, not the `ENTRIES` metadata table.**
Java joins against `ENTRIES` filtered `status != 2` because Spark has a Dataset to join with; the
fork walks the current snapshot's manifests and collects every non-DELETED entry's `file_path`.
Same set, no engine. Default ON, Java's message, at most ten paths listed.

**D-9 — one `append` snapshot, with an HONEST summary (a named divergence).** The fork appends the
`DataFile`s themselves through `Transaction::merge_append` (Java `Table.newAppend()` is
`MergeAppend`), so the snapshot summary carries `added-files-size` and `changed-partition-count` on
the PARTITIONED path too. Java's partitioned path loses those keys only because it appends
executor-written MANIFESTS (§1a) — a Spark distribution artifact, not a format rule. The fork has no
executors and no manifest-staging step, so reproducing the missing keys would mean writing a
manifest solely to degrade the summary. `added_files_count` is identical; a RePark router that must
return Spark's exact NULL for `changed_partition_count` on a partitioned target does so at the
router, from the same summary.

**D-10 — `ensure_name_mapping_present` mirrors Java, in its own commit.** When
`schema.name-mapping.default` is absent the action builds it with the Rust port of
`MappingUtil.create(schema)` and commits it through `Transaction::update_properties` BEFORE the
append, as `AddFilesProcedure` does. Without it a later scan of the id-less adopted file falls back
to POSITIONAL ids and silently returns the wrong columns — the same class of bug as cell 10, but
unbounded.

**D-11 — bounded concurrency, memory flat in the file count above the bound.** Footer reads run
through a bounded `buffered` stream at the caller's `parallelism` (default 1, Java's default and
Java's "1 means no pool"). Nothing collects the parquet footers; each one becomes a `DataFile` and
is dropped. The listing is one `FileIO::list` of the source prefix.

**D-12 — a failure leaves the table untouched.** Discovery, validation, the duplicate check and
every footer read complete BEFORE the append transaction is built, so any refusal happens with no
commit. The only ordering exception is D-10's name-mapping property commit, which is Java's order
too.

**D-13 — hidden paths are skipped at EVERY segment.** Java's `HIDDEN_PATH_FILTER` filters the LEAF
name only, because Spark's partition discovery already dropped `_`/`.` DIRECTORIES upstream. With
one recursive listing the fork applies the same `_`/`.` rule to every segment below the source root,
which is the composition of the two Java filters.

## 5. Clause coverage

| Clause | Status |
|---|---|
| in-place adoption, one append snapshot | pending |
| hive-layout partitioned source, one file per dir | pending |
| `partition_filter` selects one partition | pending |
| `partition_filter` on an unpartitioned table refuses | pending |
| `check_duplicate_files` true refuses the second import | pending |
| `check_duplicate_files` false adopts twice | pending |
| missing source refuses | pending |
| silent column drop (name mapping) | pending |
| name mapping created when absent | pending |
| `parallelism` bound | pending |
| v3 target | pending |
