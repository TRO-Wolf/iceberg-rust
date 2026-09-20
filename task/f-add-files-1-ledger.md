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

**D-14 — a source with no importable file is REFUSED, not committed as an empty snapshot.**
Java's catalog-table path reaches `table.newAppend().commit()` with an empty file list and writes
an EMPTY append snapshot (`importSparkTable` at 148–162), while its `` `parquet`.`<path>` `` path
refuses ("Cannot find any matching partitions in table %s"). The fork refuses, with
"Cannot find any file to import under <source>": the fork's snapshot producer already refuses a
no-op manifest write (`PreconditionFailed`), an empty append snapshot carries no information while
consuming a sequence number, and a caller that meant to import files and imported none must learn
so. A missing source directory reaches the same refusal, because `FileIO::list` over a
directory-semantics backend returns an EMPTY list for an absent prefix rather than an error.

**D-15 — the duplicate check runs BEFORE the footer reads.** Java builds every `DataFile` first and
checks duplicates after. The refusal and its message are identical; checking first avoids reading
every footer of an import that cannot commit. The only observable difference is which error a
caller sees when a source is BOTH a duplicate and unreadable — the fork reports the duplicate.

**D-16 — Java's second unpartitioned-filter check is DEAD, in Java too, and is not ported.**
`AddFilesProcedure.importFileTable` asserts "Cannot use a partition filter when importingto an
unpartitioned table" at bytecode 107–117, AFTER `validatePartitionFilter` at 64 has already thrown
"Cannot use partition filter with an unpartitioned table %s" for exactly that state. The fork keeps
only the reachable one, so no branch in this action looks live and is not.

**D-17 — a source whose columns carry field ids UNEVENLY is REFUSED (a named divergence).**
Java's `hasIds` is a file-level recursive ANY, so Java adopts such a file by its own ids and
`convertAndPrune` silently drops every id-less column. The fork refuses instead, because its own
reader resolves ids by a different rule (`arrow/reader.rs` looks at the FIRST top-level field) and
the two answers disagree on exactly that shape — measured in §9 L-002, where the pre-fix action
produced a table whose scan failed with `Found duplicate 'field.id' 2`. The rule: when ANY field at
any depth carries an id, EVERY TOP-LEVEL column must carry one. This also covers a file whose ids
live only below the top level, which the fork's top-level resolution (D-3) would adopt with no
metrics at all.

**D-18 — hive path values unescape through Spark's `unescapePathName`, not a URL decoder.**
The two escapes are different schemes and disagree on `+`; see §9 L-001. The decode is per code
unit, matching Java's `(char) code`.

**D-13 — hidden paths are skipped at EVERY segment.** Java's `HIDDEN_PATH_FILTER` filters the LEAF
name only, because Spark's partition discovery already dropped `_`/`.` DIRECTORIES upstream. With
one recursive listing the fork applies the same `_`/`.` rule to every segment below the source root,
which is the composition of the two Java filters.

## 5. Clause coverage

| Clause | Pin | Status |
|---|---|---|
| in-place adoption, one `append` snapshot | `unpartitioned_source_is_adopted_in_place_in_one_append_snapshot` | PROVEN |
| hive-layout partitioned source, one file per dir, values parsed | `partitioned_source_adopts_one_file_per_hive_directory` | PROVEN |
| `partition_filter` selects one partition | `partition_filter_adopts_only_the_named_partition` | PROVEN |
| `partition_filter` on an unpartitioned table refuses | `partition_filter_on_an_unpartitioned_table_is_refused` | PROVEN |
| `partition_filter` wider than the spec refuses | `a_partition_filter_wider_than_the_spec_is_refused` | PROVEN |
| `partition_filter` naming a non-partition column refuses | `a_partition_filter_naming_a_non_partition_column_is_refused` | PROVEN |
| `partition_filter` matching nothing refuses | `a_partition_filter_matching_no_partition_is_refused` | PROVEN |
| `check_duplicate_files` true refuses the second import | `check_duplicate_files_refuses_the_second_import`, `the_duplicate_refusal_is_a_typed_error_naming_every_duplicate` | PROVEN |
| `check_duplicate_files` false adopts twice | `check_duplicate_files_false_adopts_the_same_files_twice` | PROVEN |
| a missing source refuses | `a_missing_source_is_refused`, `a_missing_source_file_is_a_typed_error_naming_the_path` | PROVEN |
| a file that is not parquet refuses | `a_file_that_is_not_parquet_is_refused`, `a_non_parquet_file_refusal_names_the_file` | PROVEN |
| silent column drop (name mapping) | `a_source_column_the_target_lacks_is_dropped_and_reads_back_null` | PROVEN |
| name mapping created when absent | `the_default_name_mapping_is_created_when_absent` | PROVEN |
| `parallelism` bound, and `0` refused | `parallelism_two_adopts_the_same_files`, `parallelism_zero_is_refused` | PROVEN |
| v3 target | `a_v3_target_adopts_the_file` | PROVEN |
| `sort_order_id = 0`, no split offsets | `an_adopted_file_carries_sort_order_id_zero_and_no_split_offsets` | PROVEN |
| `__HIVE_DEFAULT_PARTITION__` is a NULL partition value | `the_hive_default_partition_directory_becomes_a_null_partition_value` | PROVEN |
| hidden `_`/`.` paths skipped | `a_hidden_directory_or_file_is_skipped` | PROVEN |
| `findCompatibleSpec` refusal | `a_source_whose_partition_columns_match_no_spec_is_refused` | PROVEN |
| an explicit file list carries its own values | `an_explicit_file_list_carries_its_own_partition_values` | PROVEN |
| a partition value that does not parse refuses | `a_partition_value_that_does_not_parse_for_its_type_is_refused` | PROVEN |
| a partition type with no hive-string parse refuses | `a_partition_type_java_cannot_parse_from_a_string_is_refused` | PROVEN |
| a non-`name=value` source directory refuses | `a_source_directory_that_is_not_a_partition_directory_is_refused` | PROVEN |
| a hive directory value is UNESCAPED the way Spark's discovery unescapes | `the_hive_path_unescape_is_spark_s_unescape_path_name`, `a_percent_escaped_hive_directory_adopts_spark_s_unescaped_value` | PROVEN |
| `partition_filter` matches the UNESCAPED value | `a_partition_filter_matches_the_unescaped_hive_value` | PROVEN |
| a source whose columns carry field ids unevenly is REFUSED | `a_source_carrying_field_ids_on_only_some_columns_is_refused`, `a_source_whose_field_ids_are_only_nested_is_refused` | PROVEN |
| an all-ids and an id-less source each adopt and scan back their rows | `an_all_ids_source_adopts_and_scans_back_its_rows`, `an_id_less_source_adopts_and_scans_back_its_rows` | PROVEN |
| a BOOLEAN hive value follows `Boolean.valueOf` (never refused) | `a_boolean_hive_value_follows_java_s_boolean_value_of` | PROVEN |
| a FLOAT/DOUBLE hive value follows `Float.valueOf` | `a_float_hive_value_follows_java_s_float_value_of` | PROVEN |
| the duplicate set spans the DELETE manifests | `a_path_already_referenced_by_a_delete_file_is_a_duplicate` | PROVEN |
| the spec choice is deterministic (lowest matching spec id) | `the_lowest_spec_id_wins_when_several_specs_match` | PROVEN |
| an all-VOID spec is `isUnpartitioned`, and matches no source | `a_partition_filter_over_an_all_void_spec_is_refused_as_unpartitioned`, `a_table_whose_only_matching_spec_is_void_is_refused` | PROVEN |
| an explicit file list keeps each file's values under `parallelism` | `an_explicit_file_list_keeps_each_file_s_partition_values_under_parallelism` | PROVEN |
| conflicting source directory structures refuse | `conflicting_source_directory_structures_are_refused` | PROVEN |

## 6. Mutation proof

Each mutation was applied to the product code alone, `cargo test -p iceberg --lib add_files` was
run, and the tree was restored. Every rule the action enforces has at least one pin that turns RED
when the rule is broken.

| # | Mutation | Pins that went RED |
|---|---|---|
| M1 | `__HIVE_DEFAULT_PARTITION__` is treated as an ordinary string | `the_hive_default_partition_directory_becomes_a_null_partition_value` |
| M2 | every partition value becomes NULL | `partitioned_source_adopts_one_file_per_hive_directory`, `partition_filter_adopts_only_the_named_partition`, `an_explicit_file_list_carries_its_own_partition_values`, `a_partition_value_that_does_not_parse_for_its_type_is_refused`, `a_partition_type_java_cannot_parse_from_a_string_is_refused` |
| M3 | the duplicate check never fires | `check_duplicate_files_refuses_the_second_import` |
| M4 | the name mapping is ignored (positional fallback) | `an_id_less_source_resolves_its_columns_by_name_not_by_position` |
| M5 | `MetricsConfig::default()` instead of `for_table` | `the_table_metrics_config_decides_the_adopted_bounds` |
| M6 | `sort_order_id` left unset | `an_adopted_file_carries_sort_order_id_zero_and_no_split_offsets` |
| M7 | the footer's split offsets are kept | `an_adopted_file_carries_sort_order_id_zero_and_no_split_offsets` |
| M8 | hidden `_`/`.` paths are not skipped | `a_hidden_directory_or_file_is_skipped` |
| M9 | `find_compatible_spec` accepts any spec | `a_source_whose_partition_columns_match_no_spec_is_refused` |
| M10 | the name-mapping property is never created | `the_default_name_mapping_is_created_when_absent`, `a_source_column_the_target_lacks_is_dropped_and_reads_back_null` |
| M11 | the partition filter is ignored | `partition_filter_adopts_only_the_named_partition`, `a_partition_filter_matching_no_partition_is_refused` |
| M12 | the file's embedded field ids are ignored | `a_source_that_carries_field_ids_is_resolved_by_those_ids` |
| M13 | a resolved id the table schema lacks is kept, not dropped | `a_field_id_the_table_schema_lacks_is_dropped_from_the_adopted_file` |
| M14 | the partition lookup ignores the spec field name | `a_two_column_partition_tuple_follows_the_spec_field_order` |
| M15 | `file_size_in_bytes` is not the listing's size | `an_adopted_file_carries_sort_order_id_zero_and_no_split_offsets` |
| M16 | the adopted file is stamped with the wrong spec id | 19 pins |
| M17 | `record_count` is not the footer's | `unpartitioned_source_is_adopted_in_place_in_one_append_snapshot`, `partitioned_source_adopts_one_file_per_hive_directory`, `the_table_metrics_config_decides_the_adopted_bounds` |

### 6a. Round 2 (the reviewers' findings)

| # | Mutation | Pins that went RED |
|---|---|---|
| M18 | the hive directory name and value are not unescaped | `a_percent_escaped_hive_directory_adopts_spark_s_unescaped_value`, `a_partition_filter_matches_the_unescaped_hive_value` |
| M19 | the unescape turns `+` into a space (URL decoding, not Spark's) | `the_hive_path_unescape_is_spark_s_unescape_path_name` |
| M20 | a `%XX` pair decodes as a UTF-8 BYTE instead of Java's `char` | `a_percent_escaped_hive_directory_adopts_spark_s_unescaped_value`, `the_hive_path_unescape_is_spark_s_unescape_path_name` |
| M21 | a source whose columns carry field ids unevenly is accepted | `a_source_carrying_field_ids_on_only_some_columns_is_refused`, `a_source_whose_field_ids_are_only_nested_is_refused` |
| M22 | a BOOLEAN hive value parses with Rust's strict `bool` | `a_boolean_hive_value_follows_java_s_boolean_value_of` |
| M23 | every non-empty BOOLEAN hive value becomes `true` | `a_boolean_hive_value_follows_java_s_boolean_value_of` |
| M24 | a FLOAT hive value parses with Rust's `FromStr` | `a_float_hive_value_follows_java_s_float_value_of` |
| M25 | the duplicate check skips the DELETE manifests | `a_path_already_referenced_by_a_delete_file_is_a_duplicate` |
| M26 | `find_compatible_spec` walks the spec map unordered | `the_lowest_spec_id_wins_when_several_specs_match` |
| M27 | `validate_partition_filter` tests `fields().is_empty()`, not `is_unpartitioned()` | `a_partition_filter_over_an_all_void_spec_is_refused_as_unpartitioned` |
| M28 | the no-matching-spec refusal drops Java's list brackets | `a_table_whose_only_matching_spec_is_void_is_refused`, `a_source_whose_partition_columns_match_no_spec_is_refused` |
| M29 | the footer read is handed size `0` instead of the listing's size | 8 pins, incl. `the_table_metrics_config_decides_the_adopted_bounds`, `a_source_that_carries_field_ids_is_resolved_by_those_ids` |
| M30 | the `Files` arm's stats run `buffer_unordered` instead of `buffered` | **GREEN — recorded, not a gap.** See below. |

Two mutations needed a note:

- **M19 was GREEN on the first attempt.** `unescape_hive_path_name` copies the prefix before the
  FIRST `%` verbatim and only loops after it, so a `+` that sits before any `%` never reaches the
  mutated branch. Every `+` case in the pin (`a+b`, `cat=a+b`) sat in that prefix. The pin now
  carries the MEASURED cases where a `+` follows a `%` — `+%20+` → `+ +`, `%20+` → ` +`,
  `a+b%20c` → `a+b c`, `%20a+b` → ` a+b`, `%2Ba` → `+a`, `a%2Bb` → `a+b` — and M19 is RED.
- **M30 is GREEN because the order is not observable, and cannot be.** Each `SourceFile` carries
  its own path AND its own partition values as one value, and `partition_names_of` compares every
  file against the first, so reordering the stat completions cannot mismatch a file with another
  file's values. The pin proves the parallel path adopts all four files with their own values; it
  does not — and no pin could — prove that `buffered` rather than `buffer_unordered` is what holds
  the order. Recorded as unpinnable rather than claimed as covered.

M4, M12, M13, M14 and M15 were GREEN on the first round. Each one added a pin, and each pin was
then proven RED under the same mutation:

- M4 was green because every fixture table's column ORDER matched its field-id order, so the
  positional fallback and the name mapping agreed. The new pin writes the source columns REVERSED.
- M12 was green because no fixture carried embedded field ids. The new pin writes a source whose
  columns are named `x`/`y` (which the name mapping cannot find) and carry ids 1/2.
- M13 was green because no fixture carried an id the table schema lacks. The new pin writes id 9.
- M14 was green because every partitioned fixture had ONE partition field. The new pin uses a
  two-field `(cat, dept)` spec.
- M15 was green because the size pin only asserted `> 0`. It now asserts the listing's exact size.

## 7. Gates

Run on the final tree (`CARGO_BUILD_JOBS=6 RUST_TEST_THREADS=6`, filtered tests only).

Round 2 re-ran every one on the final tree (head `a2ea1100` plus the ledger commit).

| Gate | Round 1 | Round 2 |
|---|---|---|
| `cargo test -p iceberg --lib add_files` | ok. 34 passed | ok. 48 passed; 0 failed |
| `cargo test -p iceberg --lib maintenance::` | ok. 499 passed | ok. 513 passed; 0 failed |
| `cargo test -p iceberg --lib scan::` | ok. 250 passed | ok. 250 passed; 0 failed |
| `cargo test -p iceberg --lib spec::name_mapping` | ok. 5 passed | ok. 5 passed; 0 failed |
| `cargo fmt --all -- --check` | clean | clean |
| `cargo clippy -p iceberg --all-targets -- -D warnings` | clean | clean |
| `python3 scripts/check_rust_file_size.py` | 590 files clean | rust-file-size: 591 files clean (92 legacy ceilings) |
| `typos .` | clean | clean |
| comment gate | `comment-ban hits=0` | `comment-ban hits=0` |

The round-1 table, for the record:

| Gate | Result |
|---|---|
| `cargo test -p iceberg --lib add_files` | ok. 34 passed; 0 failed |
| `cargo test -p iceberg --lib maintenance::` | ok. 499 passed; 0 failed |
| `cargo test -p iceberg --lib scan::` | ok. 250 passed; 0 failed |
| `cargo test -p iceberg --lib spec::name_mapping` | ok. 5 passed; 0 failed |
| `cargo fmt --all -- --check` | clean |
| `cargo clippy -p iceberg --all-targets -- -D warnings` | clean |
| `python3 scripts/check_rust_file_size.py` | rust-file-size: 590 files clean (92 legacy ceilings) |
| `typos .` | clean |
| comment gate | `comment-ban hits=0` |

`scan::` and `spec::name_mapping` are in the list because this unit touched
`crates/iceberg/src/scan/mod.rs` (the `context` module is now `pub(crate)` so the action reuses
`parse_name_mapping`) and added `spec/name_mapping/create.rs`.

## 8. What the RePark CALL router needs

`CALL system.add_files(table, source_table, partition_filter, check_duplicate_files, parallelism)`
maps onto the action with no reshaping:

| CALL argument | Action |
|---|---|
| `table` | the `Table` handed to `AddFiles::new` |
| `source_table` as `` `parquet`.`<path>` `` | `AddFilesSource::Directory(path)` |
| `source_table` as a catalog table | resolve the table's location (and, for a partitioned source, its metastore partitions) and pass either `AddFilesSource::Directory(location)` or `AddFilesSource::Files(entries)` with the metastore's values |
| `partition_filter => map(...)` | `.partition_filter(HashMap<String, String>)` |
| `check_duplicate_files => bool` | `.check_duplicate_files(bool)` (default true, as Java) |
| `parallelism => int` | `.parallelism(usize)` (default 1, as Java; `0` is refused with Java's message) |
| the result row | `AddFilesResult { added_files_count, changed_partition_count }` — the same two columns, the second nullable |

Two things the router owns, not the action:

1. **A missing SOURCE TABLE.** Java raises `NoSuchTableException("Table %s does not exist")` from
   the Spark session catalog before any Iceberg code runs. The action only sees paths, so a
   missing catalog table must be refused by the router's own lookup. A missing PATH lands on the
   action's "Cannot find any file to import under <source>".
2. **`changed_partition_count` on a PARTITIONED target.** Java returns NULL there only because its
   Spark path commits `appendManifest` (§1a). The fork commits the files, so the summary carries a
   real count. A router that must be byte-identical to Spark's result row suppresses it for a
   partitioned target; one that wants the honest number passes it through. See D-9.

## 9. Round 2 — the reviewers' findings

Round 1 was accepted; a logic critic and a Rust perf reviewer then ran. Every claim below is
MEASURED — three new Spark oracle scripts in the run-25d Spark add_files oracle directory
(`record_add_files_round2.py` cells A/A2/B/C/D/E, `record_add_files_round2b.py` cells F/G/H/I, and
`record_jvm_probe{,2,3}.py`, which call the Java methods directly through the pyspark JVM gateway)
plus `javap -c -p` on the 1.11.0 runtime jar.

### L-001 — the hive directory value is not URL-unescaped — FIXED, and the suggested home REFUTED

Measured (cells A, G, `unescapePathName` probe). Spark writes a partition value through
`ExternalCatalogUtils.escapePathName` and add_files reads it back through `unescapePathName`:

| value | directory Spark writes | what add_files adopts |
|---|---|---|
| `a b` | `cat=a b` | `a b` |
| `a/b` | `cat=a%2Fb` | `a/b` |
| `a%b` | `cat=a%25b` | `a%b` |
| `a=b` | `cat=a%3Db` | `a=b` |
| `a+b` | `cat=a+b` | `a+b` |
| `a:b` / `a?b` / `a#b` / `a*b` | `cat=a%3Ab` / `%3Fb` / `%23b` / `%2Ab` | the raw value |
| `abé` | `cat=abé` | `abé` |
| `` (empty) | `cat=__HIVE_DEFAULT_PARTITION__` | NULL |
| `a%20b` | `cat=a%2520b` | `a%20b` |

**The brief's suggestion — "add the inverse of `escape_partition_path_component`" — is REFUTED.**
That function is Iceberg's `PartitionSpec.escape`, which is `java.net.URLEncoder.encode` (a space
becomes `+`); Spark's hive layout uses a DIFFERENT scheme, and the two disagree on exactly the
character that matters. Measured: `escapePathName("a b")` = `a b` and `escapePathName("a+b")` =
`a+b`, and `unescapePathName("a+b")` = `a+b` — a URL decoder would have returned `a b` and adopted
the wrong value for every Spark-written partition containing a literal `+`. The inverse therefore
does NOT belong beside `escape_partition_path_component`; `unescape_hive_path_name` lives in
`add_files_datafile.rs`, the module that already owns the hive VALUE parsing.

The algorithm, measured pair by pair (`unescapePathName` probe): scan for `%`; when at least two
characters follow, parse them as hex (either case) and append **the character with that code
point**; otherwise append the `%` literally and advance one. Bad hex and a truncated escape are
left verbatim (`a%zzb` → `a%zzb`, `a%2` → `a%2`, `100%` → `100%`), `%%20` → `% `, `%2f` → `/`.

**The decode is per CODE UNIT, not per UTF-8 byte.** Measured (round-2c cell): `%C3%A9` →
`Ã©`, `ab%C3%A9` → `abÃ©`, and the three-byte `ab%E2%82%AC` → `abâ¬` — one character per
escape, never one character per UTF-8 sequence. Confirmed end to end through the
`` `parquet`.`path` `` form: directories `cat=ab%C3%A9` and `cat=abé` adopt as `abÃ©` and `abé`.
The fork matches with `char::from(byte)`, which is Java's `(char) code`. A UTF-8-byte decoder
would have been "more correct" and WRONG.

Both the directory NAME and the VALUE unescape, as Spark's `parsePartitionColumn` does. The
`partition_filter` compares against the unescaped value, measured in cell B:
`map('cat','a b')` selects `cat=a%20b`, `map('cat','a%20b')` selects `cat=a%2520b`, and
`map('cat','a/b')` selects `cat=a%2Fb`. The fork unescapes at discovery, so `filter_partitions`
compares unescaped for free. `AddFilesSource::Files` is untouched — a router's values arrive from a
metastore already decoded.

### L-002 — a source with field ids on SOME columns — FIXED by a loud refusal, and the fork's READER is wrong

Measured, three ways.

1. **Java is all-or-nothing per FILE, recursively.** `ParquetSchemaUtil$HasIds.struct` returns true
   if ANY child returned true, else `getId() != null`; `list`, `map` and `primitive` do the same.
   `ParquetUtil.getParquetTypeWithIds` branches on it once for the whole file, and
   `convertAndPrune` then DROPS every column that came out without an id. So Java adopts a mixed-id
   file using the file's own ids and silently loses the id-less columns.
2. **The fork's READER does NOT implement that.** `crates/iceberg/src/arrow/reader.rs:476-481`
   decides `missing_field_ids` from the FIRST top-level field alone
   (`.fields().iter().next().is_some_and(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none())`).
   Java's `hasIds` is a recursive ANY. The two disagree on exactly one shape: a file whose first
   column has no id and some later column does.
3. **What that costs, measured end to end.** Table `(id:1 long, v:2 string)`; a source file with
   columns `v` (no id) and `w` (id 2). Before the fix the action adopted it: the manifest carried
   `lower_bounds/upper_bounds {2: "z"}` from column `w`, while the reader — taking the first-field
   branch — applied the name mapping ON TOP of `w`'s embedded id and then failed the scan outright:

   > `DataInvalid => Found duplicate 'field.id' 2. Field ids must be unique.`

   The whole table became unreadable, not merely wrong. (A second shape, `id` with no id and `v`
   with id 1, failed inside the action with a leaked `Unexpected => Statistics {...} is not match
   with field type long` — an internal error, not a refusal.)

**The fix: the action REFUSES a mixed-id source**, with a typed `DataInvalid` naming the columns
that carry no id. The rule is stricter than Java's `hasIds` in the one direction that matters: when
ANY field at any depth carries an id, EVERY top-level column must carry one. That also catches the
nested-only file (ids below the top level, none above), which the fork's top-level resolution (D-3)
would otherwise adopt with no metrics at all.

This is a NAMED DIVERGENCE (D-17). Java accepts such a file and drops columns; the fork refuses.
It is the safe direction: the fork cannot produce an entry that its own reader misreads, and no
Spark or Hive writer emits a partially-id'd file — Iceberg writes ids everywhere, a migration
writes none.

**The reader is a separate fork bug and is NOT fixed here** (the brief's instruction, and it is a
scan-path change that wants its own unit and its own pins). Logged in `task/todo.md`.

### L-003 — a BOOLEAN hive value — FIXED, and the finding UNDERSTATED the divergence

Measured directly (`Conversions.fromPartitionString(Types.BooleanType.get(), s)` over the JVM
gateway). Java's BOOLEAN branch is `Boolean.valueOf`, which **never throws**:

| input | Java | fork before | fork now |
|---|---|---|---|
| `true`, `TRUE`, `True`, `tRuE` | `true` | `true` only for `true` | `true` |
| `false`, `FALSE`, `False` | `false` | `false` only for `false` | `false` |
| `yes`, `no`, `1`, `0`, `` , ` true`, `true ` | **`false`** | **REFUSED** | `false` |

So the critic's "`TRUE`/`FALSE` is refused where Java accepts it" was right but narrow: Java also
turns every unparsable string into `false`. The fork is now
`raw.eq_ignore_ascii_case("true")` — `"true"` has no character whose Unicode case folding differs
from ASCII, so `equalsIgnoreCase` and `eq_ignore_ascii_case` agree on every input.

**FLOAT/DOUBLE diverged too, and are fixed in the same pass** (measured, same probe). Java's
`Float.valueOf`/`Double.valueOf` TRIM whitespace, accept a trailing `f|F|d|D`, and accept exactly
`NaN` / `Infinity` (with an optional sign):

| input | Java | Rust `parse::<f32>` |
|---|---|---|
| `1.5f`, `1.5D`, ` 1.5`, `\t1.5\n` | 1.5 | REFUSED |
| `NaN`, `Infinity`, `-Infinity` | NaN, ±∞ | accepted |
| `nan`, `NAN`, `inf`, `infinity`, `INFINITY` | **REFUSED** | **accepted** |
| `1.`, `.5`, `1e5`, `+1.5` | accepted | accepted |
| `""`, `" "` | REFUSED | REFUSED |

`parse_java_floating` now trims Java's `String.trim()` set, strips one trailing `f/F/d/D`, maps a
signed `NaN`/`Infinity` onto Rust's spelling, and refuses any remaining string carrying an ASCII
letter other than `e`/`E`. That closes `inf`/`nan`/`infinity` (which the fork used to accept and
Java refuses) and opens `1.5f`/`1.5d` (which Java accepts and the fork used to refuse).

INTEGER, LONG, DATE and DECIMAL were measured in the same probe and already AGREE: Java refuses a
leading or trailing space on all four (` 1` → `For input string: " 1"`), accepts `+1` and `007`,
and refuses `2020-1-1`. No change.

### L-004 — the duplicate check skipped the DELETE manifests — FIXED

Measured (cell H). A v2 table with one merge-on-read `DELETE` over one data file:
`SELECT status, data_file.content, data_file.file_path FROM tbl.entries` returns TWO rows —
`(1, 0, <data file>)` and `(1, 1, <position delete file>)`. Java's duplicate check joins the
candidate paths against exactly that set (`ENTRIES` filtered `status != 2`), so a candidate whose
path is already a DELETE file's path is a duplicate for Java. The fork walked only
`ManifestContentType::Data`. The content filter is gone; the function is now `live_entry_paths`.
`entry.is_alive()` is the `status != 2` half and was already right.

### L-005 — `find_compatible_spec` walked a `HashMap` — FIXED

Measured (`javap` on `TableMetadata` → `PartitionUtil.indexSpecs`): `specsById` is an
`ImmutableMap` built by iterating the metadata's partition-spec LIST, so `table.specs().values()`
iterates in metadata list order — the order the specs were added, i.e. ascending spec id for every
spec list Java writes. The fork's `TableMetadata` keeps a `HashMap<i32, PartitionSpecRef>` and lost
that order at parse time, so two equivalent identity specs gave a non-deterministic `spec_id`.
`find_compatible_spec` now sorts by `spec_id` before walking, which reproduces Java's order for
every spec list Java can produce.

While matching the refusal the message was corrected too: Java formats the source names with
`String.format("...(%s)...", List<String>)`, so the list prints WITH brackets. Measured in cell F:
`Cannot find a partition spec in Iceberg table sc.ns.v1_void that matches the partition columns
([]) in input table`. The fork printed `()` / `(dept)`; it now prints `([])` / `([dept])`.

### The void-spec item — REFUTED as unreachable, and the inconsistency fixed anyway

The critic reported that `is_unpartitioned` and `validate_partition_filter` disagree for a spec
whose fields are all `Void`. Both halves were measured.

1. **`is_unpartitioned()` is right.** `PartitionSpec.isPartitioned()` is
   `fields.length > 0 && fields.stream().anyMatch(<not void>)`, so `isUnpartitioned()` is
   "empty OR all void" — the fork's definition exactly. Cell E confirms it end to end: a
   partition_filter over a table whose default spec has no live field takes the
   `Cannot use partition filter with an unpartitioned table sc.ns.void_filter` refusal.
2. **The disagreement is unreachable — in Java too.** An all-Void spec is not all-Identity, so
   `findCompatibleSpec` skips it, and `findCompatibleSpec` runs BEFORE `validatePartitionFilter`
   (`importSparkTable` bytecode 94 then 109). Measured in cell F on a **v1** table (v2's
   `DROP PARTITION FIELD` removes the field outright and leaves an EMPTY spec — only v1 leaves a
   `void` field, confirmed in the on-disk metadata): both `add_files` and
   `add_files + partition_filter` fail with `Cannot find a partition spec ... ([]) ...`, never with
   the unpartitioned-filter message. The fork's `validate_partition_filter` can only ever see the
   all-identity spec `find_compatible_spec` returned.

`validate_partition_filter` now tests `spec.is_unpartitioned()` anyway, so the two functions state
one rule instead of two. Because the branch is unreachable through `execute`, it is pinned by a
DIRECT unit test on the function (`a_partition_filter_over_an_all_void_spec_is_refused_as_unpartitioned`)
rather than by an end-to-end pin that would be vacuous — and the reachable half (an all-Void spec
matches no source) is pinned end to end against cell F's message.

### R-01 / R-02 / R-03 — FIXED. R-04 — DEFERRED

- **R-01.** `adopt_parquet_file` no longer stats the file it is about to read; it builds
  `FileMetadata { size: file_size_in_bytes }` from the size the listing (or the `Files` arm's own
  stat) already carries, as `rewrite_data_files_write::input_parquet_metadata` does. One RPC per
  file, gone.
- **R-02.** The footer read now runs with `preload_column_index(false)`,
  `preload_offset_index(false)` and `preload_page_index(false)`. Only row-group statistics are
  used and the split offsets are cleared afterwards, so the indexes were pure waste. The 512 KiB
  `metadata_size_hint` is KEPT — the rewrite path's 8-byte footer hint is not copied, per the
  brief.
- **R-03.** The `Files` arm's stats run through the same bounded `buffered` stream at the caller's
  `parallelism` instead of a serial `for`-await.
- **R-04 — DEFERRED, with the reason.** `Schema::build` and `MetricsByFieldId::new` are per file
  because the resolved schema is a function of THAT file's own top-level columns, which the action
  cannot know until it has read that file's footer. Hoisting them means a cache keyed on the file's
  `(name, id)` column list, shared across a `buffered` stream — a lock or a per-task clone — to
  save O(columns) allocations per file against one object-store round trip per file. The trade is
  not obviously positive and the change is not contained; it is recorded here rather than taken.

None of R-01…R-03 is directly pinnable: the workspace has no request-counting `FileIO`, so "one
fewer HEAD" has no observable. What IS pinned is the correctness coupling R-01 introduces — the
footer read now TRUSTS the listing's size — and M29 (hand the reader size `0`) turns eight pins
RED. R-03's pin proves the parallel path adopts every file with its own values; see M30 for what it
cannot prove.

## 10. Open


- **CLOSED (round 2): hive value escaping.** See §9 L-001. The decoder is
  `unescape_hive_path_name`, measured against `ExternalCatalogUtils.unescapePathName`.
- **A hex float literal is still refused.** Measured: `Float.valueOf("0x1p3")` = 8.0; the fork
  refuses it. Java's hex-significand grammar is the one form of `Float.valueOf` the fork does not
  implement. A hive directory named `f=0x1p3` is the only way to reach it.
- **DECIMAL scale is not re-scaled.** Measured: `fromPartitionString(decimal(9,2), "1.234")`
  returns `1.234` — Java does NOT rescale to the declared scale. Whether the fork's
  `Literal::decimal_from_str` agrees was not measured in this unit.
- **The fork's READER decides `hasIds` from the first top-level field only**
  (`arrow/reader.rs:476-481`), where Java's `ParquetSchemaUtil.hasIds` is a recursive ANY. This
  unit refuses to FEED the reader such a file (§9 L-002) but does not fix the reader; a migrated
  table can still hold one from another writer. Its own unit — logged in `task/todo.md`.
- **Nested-column resolution is TOP LEVEL**, matching the fork's read path
  (`arrow/reader.rs`). A struct/list/map column takes its nested ids from the table's own field
  under the resolved top-level id, so a file whose NESTED names differ from the table's carries no
  metrics for those leaves. No oracle cell covers it.
- **ORC and Avro sources.** Java's `TableMigrationUtil.listPartition` dispatches on
  `format.contains("avro"|"parquet"|"orc")`. This unit is parquet only; a non-parquet file is
  refused by the footer read.
