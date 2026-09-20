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

# Ledger — F-POSDEL-SCAN-1: the `position_deletes` metadata table scan

**Ledger id:** `F-POSDEL-SCAN-1`
**Branch:** `fix/f-posdel-scan-1`
**Parity row:** IPI-45 (GAP_MATRIX R142); unblocks `R-MT-POSITION-DELETES`,
`R-MT-POSITION-DELETES-V3`
**Oracle:** `/tmp/oc-worker/rd-oracle/posdel/posdel_truth.json`
(Spark 4.1.2 + `iceberg-spark-runtime-4.1_2.13:1.11.0`)
**Model:** swe-2-high

| Step | Commit | Subject |
|---|---|---|
| 1 | this commit | `docs: F-POSDEL-SCAN-1 — ledger skeleton, reuse map` |

## 1. The gap

`crates/iceberg/src/inspect/position_deletes.rs` already ports Java
`PositionDeletesTable.calculateSchema` — fixed `MetadataColumns` ids, partition child-id
reassignment, empty-partition drop — but `scan()` refuses with `FeatureUnsupported`. Spark
answers both `position_deletes` cells (v2 parquet delete files and v3 puffin deletion
vectors); a refusal is not an acceptable end state for a cell Spark answers. This unit
implements the scan only; the schema is already correct and is not changed except where a
measured fact says it is wrong.

## 2. Java evidence (Iceberg 1.11.0, `javap -c -p` on the spark-runtime jar)

### 2.1 Planning — `PositionDeletesTable$PositionDeletesBatchScan.doPlanFiles()`

1. Manifest source is `snapshot().deleteManifests(io)` — DELETE manifests of the scan's
   snapshot only. Not data manifests, not the all-snapshots union.
2. Each manifest is filtered by TWO `ManifestEvaluator.forRowFilter(...)` instances, both
   keyed on `manifest.partitionSpecId()`: one built from the scan `filter()` against the
   TRANSFORMED spec (`BaseMetadataTable.transformSpecs(tableSchema, table.specs())`), one
   built from `baseTableFilter` against the table's own spec. A manifest passes only when
   both evaluate true.
3. Each surviving manifest is read with `ManifestFiles.readDeleteManifest(manifest, io,
   transformedSpecs)`, `.caseSensitive(...)`, `.select(scanColumns())`,
   `.filterRows(filter())`, `.filterPartitions(Projections.inclusive(spec,
   caseSensitive).project(baseTableFilter))`, then `liveEntries()` — deleted (status 2)
   entries are excluded.
4. The live entries are filtered by `entry.file().content() == FileContent.POSITION_DELETES`
   (`PositionDeletesBatchScan$1.lambda$iterator$0`). Equality delete files are excluded
   (oracle cell `sc.ns.pd_eqdel_v2:DELETE-FILES-CONTENT` measured `content = 1` — i.e.
   Spark's MoR DELETE writes position deletes; the R-1 ruling targets equality-delete files
   written by other paths).
5. One task per surviving delete file, carrying the `DeleteFile`, the spec string, the
   schema string and a `ResidualEvaluator` for that spec (`alwaysTrue` when residuals are
   ignored).

Bytecode check on `PositionDeletesTable.transformSpec` (`javap` output, this session):
builds `PartitionSpec.builderFor(schema)`, preserves the original spec id, disables
conflict checking, reassigns partition child source ids through `Schema.idsToReassigned()`,
adds identity transforms for the projected partition fields, then `build(true)`.

### 2.2 Reading — `org.apache.iceberg.spark.source.PositionDeletesRowReader.open(task)`

1. `ContentFileUtil.isDV(deleteFile)` decides the branch.
   - DV (v3 puffin): rows from `DVIterator(inputFile, deleteFile, expectedSchema,
     constantsMap)` — the deletion vector materialises into one row per deleted position.
   - Otherwise (v2 positional delete file): the delete file is read as data with
     `newIterable(inputFile, format, task.start(), task.length(), residual,
     expectedSchema, constantsMap)`.
2. The residual pushed into the file read is
   `ExpressionUtil.extractByIdInclusive(task.residual(), expectedSchema, caseSensitive,
   nonConstantFieldIds)` where `nonConstantFieldIds` = the expected schema's field ids that
   are not in the constants map. Constant columns are filtered after the read, not inside.
3. The constants map (`constantsMap(task, expectedSchema)`) supplies, per task:
   `partition` (the file's partition tuple coerced into the table's unified partition
   type), `spec_id` (the delete file's spec id), `delete_file_path` (the delete file's
   location), and for a DV `content_offset` / `content_size_in_bytes` from the
   `DeleteFile` metadata. `file_path` and `pos` (and `row`, when present) come from the
   delete file's own rows; for a DV, `file_path` is the DV's referenced data file and
   `row` is null.

## 3. The oracle

Truth file `/tmp/oc-worker/rd-oracle/posdel/posdel_truth.json`, five tables:

| table | shape |
|---|---|
| `pd_part_v2` | `(id BIGINT, cat STRING, v INT)` partitioned by `cat`, v2 MoR; `DELETE WHERE id=2`, `DELETE WHERE id=4` → two parquet position-delete files |
| `pd_part_v3` | same, format-version 3 → deletes become puffin DVs |
| `pd_unpart_v2` | `(id BIGINT, v INT)` unpartitioned v2 MoR; `DELETE WHERE id=2` |
| `pd_evolved_v2` | partitioned by `cat`, one delete, `DROP PARTITION FIELD cat` + `ADD PARTITION FIELD bucket(4,id)`, insert, another delete |
| `pd_eqdel_v2` | unpartitioned v2 MoR, one delete — `delete_files` shows `content = 1` |

Measured facts the pins encode:

1. Schema v2 partitioned: `file_path string NOT NULL, pos bigint NOT NULL,
   row struct<id,cat,v> NULL, partition struct<cat:string> NOT NULL, spec_id int NOT NULL,
   delete_file_path string NOT NULL` — in that order.
2. Schema v3 partitioned: the same plus `content_offset bigint NULL,
   content_size_in_bytes bigint NULL` at the end.
3. Schema unpartitioned v2: `partition` column absent.
4. Schema spec-evolved v2: `partition` is the unified type
   `struct<cat:string, id_bucket_4:int>`; the one row's partition value is `["x", null]`.
5. Rows `pd_part_v2`: two rows, `pos = 1` both, partition `["x"]` and `["y"]`,
   `spec_id = 0`, `row` NULL, two distinct `delete_file_path`s, `file_path`s are the two
   data files under `cat=x` / `cat=y`.
6. Rows `pd_part_v3`: same two rows, `row` NULL, `content_offset = 4`,
   `content_size_in_bytes = 42`; delete files are `*-deletes.puffin` at the table data
   root.
7. Filters: `WHERE pos = 0` → empty; `WHERE spec_id = 0` → 2; `WHERE partition.cat='y'` → 1.
8. `pd_eqdel_v2`: `delete_files` reports `content = 1` for Spark's DELETE.

### 3.1 Oracle open question — `pd_evolved_v2` shows only ONE row (spec_id 0)

The recording captured one `position_deletes` row for `pd_evolved_v2` — the second DELETE
(after spec evolution to `bucket(4,id)`) produced no row. Measured answer: **PENDING**
(step 2/3 work inspects the recorded warehouse's delete manifests to settle whether Spark
copy-on-write rewrote the delete, the second delete file is an equality delete, or the
spec-1 manifest entry is non-live). To be resolved by reading the delete manifest list in
`/tmp/oc-worker/rd-oracle/posdel/wh/` directly; outcome recorded here and in the hand-back.

## 4. Reuse map — existing fork functions named (brief step 1 deliverable)

| Need | Reused function |
|---|---|
| (a) reading a delete manifest | `ManifestFile::load_manifest(&file_io)` + `manifest.consume_entries()` — same call the `files`/`entries` tables use in `crates/iceberg/src/inspect/files.rs`; delete manifests of the current snapshot come from `ManifestSource::current(&table).collect()` filtered to `content == ManifestContent::Deletes` (the `snapshot().deleteManifests(io)` equivalent) |
| (b) reading a v2 positional delete file | `BasicDeleteFileLoader::create_basic_read` + `project_positional_deletes`/`position_delete_field_ids` in `crates/iceberg/src/arrow/delete_file_loader.rs` — full parquet read + field-id projection with name fallback (`file_path`/`pos`) |
| (c) reading a v3 puffin DV into positions | `crate::delete_vector::load_delete_vector(&data_file, file_io)` → `DeleteVector::iter()` — performs the puffin validation, coordinate bounds check, ranged blob read, `deletion-vector-v1` decode and the record-count cross-check |
| manifest-level filter | `ManifestEvaluator::builder(bound_predicate).build()` + `InclusiveProjection::new(spec).project(...)` — same pair the data scan uses via `PartitionFilterCache`/`ManifestEvaluatorCache` (`crates/iceberg/src/scan/cache.rs`) |
| per-task residual | `ResidualEvaluator::of(spec, schema, bound_filter, case_sensitive)` + `residual_for(partition)` (`crates/iceberg/src/expr/visitors/residual_evaluator.rs`) |
| unified partition projection | `inspect::partition_values::append_partition` (field-id match + null fill) + `inspect::data_file::partition_field_ids_by_spec` — the same pair `files` uses |
| Arrow batch stream shape | `files.rs`/`entries.rs` idiom — build each output column with typed `*Builder`s inside `StructBuilder::from_fields`, `RecordBatch::try_new(arrow_schema, columns)` → `futures::stream::once` |

## 5. Design notes (decisions taken while reading — recorded for the Critic)

- **Row filter source.** `PositionDeletesTable::scan()` takes no filter and the DataFusion
  metadata-table provider ignores `_filters`
  (`crates/integrations/datafusion/src/table/metadata_table.rs` discards them). The Java
  walk is still built exactly as 2.1 prescribes — scan filter and `baseTableFilter` are
  both `AlwaysTrue` in this configuration, so both `ManifestEvaluator`s, the inclusive
  projections, the live-entry filter and the per-task `ResidualEvaluator` are real running
  machinery, not dead code. A later unit that wires a filter into `scan()` replaces the
  two `AlwaysTrue` constants; nothing else changes.
- **`transformSpecs` port.** Java's transformed spec needs conflict checking OFF (it is
  `checkConflicts(false)`); `PartitionSpecBuilder::add_unbound_field` always checks. The
  port adds a `pub(crate)` unchecked constructor on `PartitionSpec`
  (`new_unchecked(spec_id, fields)`) — the faithful analogue of `builderFor(...).
  checkConflicts(false)...build(true)` — used only by `inspect::position_deletes`'s
  `transform_specs`. `PartitionSpec` fields are private, so this is the only way to build
  it without the checks.
- **Residual pushdown.** The per-task residual is computed per 2.1.5. Java then pushes
  `ExpressionUtil.extractByIdInclusive(residual, expectedSchema, ..., nonConstantFieldIds)`
  into the file read; the fork has no `extractByIdInclusive`, and the only residual this
  port can produce is `AlwaysTrue` (no filter reaches `scan()`). The task carries the
  computed residual; the file read pushes `None` — behaviorally identical while the filter
  is vacuous. A residual that is not `AlwaysTrue` would need the extract port AND a filter
  parameter; both land with the later pushdown unit.
- **`row` column.** R-4: read from the delete file when the file carries it, null
  otherwise. A v2 delete file whose `row` struct type matches the expected Arrow type is
  surfaced verbatim; a type mismatch is `FeatureUnsupported` loud (never silently
  null-filled — silent null would fabricate "no row stored" for a file that stored one).
- **`partition` nullability.** The Arrow `partition` field is non-nullable (Java marks it
  required); `append_partition` always appends a struct, null-filling absent fields. An
  evolved-to-unpartitioned file still emits a non-null all-nulls struct, matching Java's
  `coercePartition` output.
- **DV `file_path`/`pos`.** `file_path` = `referenced_data_file` (validated non-null by
  `load_delete_vector`); `pos` = each u64 position from `DeleteVector::iter()` cast to i64.
- **Row ordering.** Emit rows in manifest-entry order per delete file, positions in file
  order (DV iter is ascending). Java does not sort; the oracle's observed order follows
  manifest order. Tests assert content, not incidental ordering.

## 6. Clauses

```yaml
LEDGER:
  - id: C-001
    proposition: >
      v2 partitioned schema matches oracle fact 1 — file_path/pos/row/partition/spec_id/
      delete_file_path in that order with the measured nullability (already pinned by the
      existing schema tests).
    verdict: OPEN
  - id: C-002
    proposition: >
      v3 partitioned schema appends nullable content_offset + content_size_in_bytes
      (oracle fact 2).
    verdict: OPEN
  - id: C-003
    proposition: >
      unpartitioned v2 schema drops the partition column (oracle fact 3).
    verdict: OPEN
  - id: C-004
    proposition: >
      spec-evolved table: partition column is the unified type and a file written under an
      older spec null-fills unified fields its own spec does not carry (oracle fact 4).
    verdict: OPEN
  - id: C-005
    proposition: >
      partitioned v2 scan emits one row per delete-file record with the measured pos,
      partition, spec_id, delete_file_path and referenced file_path values; row NULL when
      the file stores none (oracle facts 5 + R-4).
    verdict: OPEN
  - id: C-006
    proposition: >
      v3 puffin DV scan emits one row per DV position with file_path =
      referenced_data_file, row NULL, and content_offset / content_size_in_bytes populated
      from the DeleteFile metadata (oracle fact 6).
    verdict: OPEN
  - id: C-007
    proposition: >
      equality-delete files never appear in position_deletes — the
      DataContentType::PositionDeletes content filter (2.1.4, R-1).
    verdict: OPEN
  - id: C-008
    proposition: >
      deleted (status 2) manifest entries never appear — the liveEntries() equivalent
      (2.1.3).
    verdict: OPEN
  - id: C-009
    proposition: >
      the scan reads the current snapshot's DELETE manifests only — data manifests of the
      same snapshot contribute nothing (2.1.1, R-2).
    verdict: OPEN
  - id: C-010
    proposition: >
      planning runs both manifest evaluators keyed on manifest.partitionSpecId()
      (transformed-spec evaluator + own-spec evaluator), the live filter and the content
      filter, and produces one task per delete file carrying its spec and its residual
      (2.1.2-2.1.5).
    verdict: OPEN
  - id: C-011
    proposition: >
      format routing is ContentFileUtil.isDV-equivalent: Puffin + PositionDeletes takes the
      DV path (load_delete_vector), everything else takes the positional-delete parquet
      path (BasicDeleteFileLoader). The wrong branch errors loud, never silently.
    verdict: OPEN
  - id: C-012
    proposition: >
      pos comes from the delete file's pos column (DELETE_FILE_POS_ID) — not file_path,
      not the partition tuple; file_path comes from the delete file's file_path column
      (v2) or referenced_data_file (DV).
    verdict: OPEN
  - id: C-013
    proposition: >
      every mutation (a)-(g) turns at least one pin red with the same test population —
      a green mutation is a pin gap and is fixed, not rationalised.
    verdict: OPEN
  - id: C-014
    proposition: >
      module doc, inspect/map.md position_deletes row and GAP_MATRIX R142 no longer claim
      schema-only / scan-refused; the refusal text is deleted.
    verdict: OPEN
```

## 7. Mutation records (step 6 — filled as each sabotage runs)

| Mutation | Pins reddened | Arithmetic |
|---|---|---|
| (a) drop POSITION_DELETES content filter | — pending — | — |
| (b) data manifests instead of delete manifests | — pending — | — |
| (c) include status-2 (deleted) entries | — pending — | — |
| (d) partition from file's own spec, not unified type | — pending — | — |
| (e) DV branch for v2 file and vice versa | — pending — | — |
| (f) drop content_offset / content_size_in_bytes | — pending — | — |
| (g) pos from the wrong column | — pending — | — |

## 8. Gates (step 7 — filled as they run)

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | pending |
| `cargo clippy -q -p iceberg --all-targets -- -D warnings` | pending |
| `cargo test -q -p iceberg --lib inspect` | pending |
| `cargo test -q -p iceberg --lib position_delete` | pending |
| `python3 scripts/check_rust_file_size.py` | pending |
| `python3 scripts/check_ledger_grammar.py` (if exists — it does not) | n/a |
| `python3 /tmp/oc-worker/_lib/comment_ban.py /tmp/rd-posdel origin/main` | pending |
