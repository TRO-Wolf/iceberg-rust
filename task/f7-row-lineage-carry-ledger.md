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

# Evidence ledger — F-7 slice 1: v3 row-lineage carry through rewrites

Home for bytecode evidence behind this slice. Capability status lives on GAP_MATRIX rows
R166 / R135 / R103. Java method names in doc comments; offsets here.

Oracle commands:

```
javap -p -c -constants -cp ~/.m2/repository/org/apache/iceberg/iceberg-core/1.10.0/iceberg-core-1.10.0.jar org.apache.iceberg.TableUtil
javap -p -c -constants -cp /tmp/ivy-dmlb/cache/org.apache.iceberg/iceberg-spark-runtime-4.1_2.13/jars/iceberg-spark-runtime-4.1_2.13-1.11.0.jar org.apache.iceberg.spark.source.SparkRewriteTable
```

## Java mechanism (decoded 2026-08-30)

| Class | Rule |
|---|---|
| `TableUtil.supportsRowLineage` | `formatVersion >= 3` after a null-table check and a `BaseMetadataTable` false. Offsets 15-37. |
| `MetadataColumns.schemaWithRowLineage` | `TypeUtil.join(schema, Schema(ROW_ID, LAST_UPDATED_SEQUENCE_NUMBER))`. Offsets 0-27. |
| `SparkRewriteTable.rewriteSchema` | if `supportsRowLineage` then `schemaWithRowLineage(table.schema())` else `table.schema()`. Offsets 0-23. Used as the Spark table schema for both the rewrite scan and `SparkRewriteWriteBuilder`. |
| `SparkCopyOnWriteOperation.requiredMetadataAttributes` | when lineage is supported, append `SparkMetadataColumns.ROW_ID` and `LAST_UPDATED_SEQUENCE_NUMBER` to the required metadata refs. Offsets 57-85. |
| `SparkWrite$WriterFactory.createWriter` | always constructs `ExtractRowLineage(writeSchema)` (offsets 145-156). `ExtractRowLineage.<init>` sets `rowLineageRequired` from `writeSchema.findField(ROW_ID.name()) != null`. `apply` returns null when not required; otherwise projects the two lineage fields. `decorateWithRowLineage` joins them onto the data row. |
| `ValueReaders$RowIdReader.read` / `$LastUpdatedSeqReader.read` | stored column wins; else `firstRowId + pos` / file sequence. See [f13-v3-row-lineage-ledger.md](f13-v3-row-lineage-ledger.md). |

`GenericAppenderFactory` does not emit stored `_row_id`. Plain INSERT stays computed. Rewrite and COW are the writers that persist the scanned pair.

## Why F-7 U1 is not this slice

U1 (`FirstRowIdPolicy::Suppress` at `SnapshotProducer::new`) clears the added file's
`first_row_id` so the manifest-list writer assigns a fresh range. That is required, and already
landed. Without stored columns the reader then computes `_row_id = new_first_row_id + pos`,
which is the 0..11 → 12..23 reassignment. Stored columns are the carry.

## Charter ledger

```yaml
LEDGER:
  - id: C-001
    proposition: >
      schema_with_row_lineage(schema) equals TypeUtil.join(schema, {_row_id, _last_updated_sequence_number})
      with the reserved ids and optional nullability Java declares.
    verdict: PROVEN
    proof: metadata_columns::schema_with_row_lineage_joins_the_two_reserved_fields
  - id: C-002
    proposition: >
      RewriteDataFiles on format v3 rewrites live rows with stored _row_id and
      _last_updated_sequence_number unchanged (V3-LINEAGE-1).
    verdict: PROVEN
    proof: >
      rewrite_data_files_lineage_tests::v3_compaction_keeps_row_id_and_last_updated_seq;
      interop D2 compact table Java-read
    enumeration:
      domain: rewrite entry points that copy existing rows
      partition:
        - RewriteDataFiles bin-pack write (V3-LINEAGE-1)
        - OverwriteFiles COW DELETE survivors (V3-COW-1)
        - OverwriteFiles COW UPDATE survivors and matched rows (V3-COW-1)
      complete_because: >
        Those are the fork surfaces that rewrite parquet for an existing row. RewriteFiles as a
        commit of caller-supplied files is the U1 metadata half and is already pinned. MoR
        RowDelta inserts are named residue.
  - id: C-003
    proposition: >
      COW DELETE survivors keep _row_id and last_updated_seq. COW UPDATE keeps _row_id for every
      rewritten row; unmatched survivors keep last_updated_seq; matched rows store a null
      last_updated_seq so the reader uses the new file sequence.
    verdict: PROVEN
    proof: iceberg-datafusion tests in row_lineage_cow.rs
  - id: C-004
    proposition: Format v2 RewriteDataFiles does not write the reserved lineage columns.
    verdict: PROVEN
    proof: v2_compaction_does_not_persist_row_lineage_columns
  - id: C-005
    proposition: >
      Java IcebergGenerics.project(schemaWithRowLineage) reads fork-compacted stored lineage
      equal to the pre-compaction ids.
    verdict: PROVEN
    proof: run-interop-row-lineage.sh compact D2 leg
```

## Residues (not built)

- V3-DANGLE-1 / row R137 dangling-DV drop on compaction.
- Row R136 RewritePositionDeleteFiles DV remainder.
- MoR UPDATE/DELETE `_row_id` carry on RowDelta-added data files.

## Risk

Writing stored `_row_id` on a new INSERT would freeze computed ids into the file and break
range assignment. This slice extends only rewrite/COW write schemas, never the INSERT writer
in `physical_plan/write.rs`.
