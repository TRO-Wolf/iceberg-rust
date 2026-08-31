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

# Scope brief — F-7 slice 1: v3 row-lineage carry through rewrites

**Unit:** engine handoff item F-7, slice 1 (`V3-LINEAGE-1` + `V3-COW-1`). **Branch:**
`parity/f7-row-lineage-carry` off main. **Matrix rows:** [`R166`](../docs/parity/GAP_MATRIX.md)
(row lineage), [`R135`](../docs/parity/GAP_MATRIX.md) (`RewriteDataFiles`),
[`R103`](../docs/parity/GAP_MATRIX.md) (`OverwriteFiles`). Status stays in those cells.

## The problem

Format v3 assigns each live row a stable `_row_id` and a `_last_updated_sequence_number`. A
rewrite that copies a live row into a new data file must persist those values. The engine
measured the fork reassigning them twice: RP-2 (`ce92a7bf`) and RP-3 (`d408da42`) saw compaction
turn `_row_id` `0..11` into `12..23` and stamp every retained row with sequence 13.

F-7 U1 already suppresses `first_row_id` at the merging-producer add seam (Java
`Delegates.suppressFirstRowId`). That is the metadata half. This slice is the stored-column
half: a rewritten parquet file must carry the scanned lineage columns so the reader prefers
the stored values over `first_row_id + pos` / file sequence.

## The Java 1.10.0 / 1.11.0 oracle

Jars: `iceberg-core-1.10.0` under `~/.m2`, Spark runtime
`iceberg-spark-runtime-4.1_2.13-1.11.0` under `/tmp/ivy-dmlb`. Citations are first-hand
`javap`. Offsets live in [`f7-row-lineage-carry-ledger.md`](f7-row-lineage-carry-ledger.md).

- `TableUtil.supportsRowLineage(Table)` is format version ≥ 3 (and not a metadata table).
- `MetadataColumns.schemaWithRowLineage(Schema)` is `TypeUtil.join(schema, {_row_id, _last_updated_sequence_number})`.
- Compaction: `SparkRewriteTable.rewriteSchema` returns that joined schema when lineage is
  supported. Scan and write of a rewrite group both use it. Stored columns then win at read
  (`ValueReaders$RowIdReader` / `$LastUpdatedSeqReader`).
- COW DML: `SparkCopyOnWriteOperation.requiredMetadataAttributes` adds the two Spark metadata
  columns so the rewrite scan projects them. `SparkWrite$WriterFactory` always builds
  `ExtractRowLineage(writeSchema)`; when `_row_id` is in the write schema, the extractor joins
  the two lineage fields onto the data row before the parquet writer.
- `GenericAppenderFactory` still does not emit stored lineage. New rows (plain INSERT) stay
  computed. Only rewrite/COW write the stored pair.

## The ask (this slice)

1. Public `schema_with_row_lineage` (Java `MetadataColumns.schemaWithRowLineage`).
2. `RewriteDataFiles` on format v3: project the two reserved ids on each rewrite-group task,
   write through a lineage-extended schema. V1/V2 unchanged.
3. DataFusion COW `DELETE` / `UPDATE` (`OverwriteFiles`): the same project-and-write. A
   survivor keeps both values. An UPDATE-matched row keeps `_row_id` and stores a NULL
   `_last_updated_sequence_number` so the reader falls back to the new file sequence.
4. Pins: unit tests that fail without the stored columns, plus a Java-read interop leg on a
   fork-compacted V3 table.

## Explicitly OUT of scope

- Dangling-DV removal on compaction (`V3-DANGLE-1` / row R137). Named residue.
- Further `RewritePositionDeleteFiles` DV-aware remainder (row R136). Named residue.
- MoR UPDATE/DELETE `_row_id` carry on `RowDelta` inserts. Named residue: this slice owns the
  `OverwriteFiles` / `RewriteDataFiles` rewrite of existing rows.
- Flipping R166 / R135 / R103 off 🟡/✅. Narrow the cells; do not restated status.
- Java `SnapshotProducer.apply` REPLACE record-count precondition (row R107 residue).
