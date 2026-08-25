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

# F-13 V1+V2 — V3 row lineage · unit ledger

Home for the **bytecode evidence** behind the row-lineage units, per AGENTS.md's evidence-routing
table: doc comments name the mirrored Java method and the caller-visible divergence; offsets,
`javap` output and decode narrative live here; capability status lives on GAP_MATRIX row R166.

Oracle: `javap -p -c -constants -cp
/home/john/.m2/repository/org/apache/iceberg/iceberg-core/1.10.0/iceberg-core-1.10.0.jar <class>`
(api jar alongside). Offsets are BYTECODE offsets from that disassembly — never Java source line
numbers. (The first version of this file confused the two for `fileFieldReader`; a round-4 Critic
caught it. An offset here must be checkable by running the command above and reading the left-hand
column.)

---

## V1 — `ManifestReader.idAssigner` (→ `spec/manifest/entry.rs::assign_first_row_ids`)

`idAssigner(Long firstRowId)` returns one of two functions:

| Arm | Bytecode | Behaviour |
|---|---|---|
| `firstRowId == null` | `idAssigner` offsets 0-1 branch to 13; `lambda$idAssigner$2` | sets EVERY file's `firstRowId` to **null** (`setFirstRowId(null)` at offset 22), OVERWRITING a stored value. Guarded only by `instanceof BaseFile` (`ifeq 25` at offset 9) — no status guard, no already-assigned guard. |
| otherwise | `ManifestReader$1` | stateful counter `nextRowId = firstRowId` (ctor offsets 9-17) |

`ManifestReader$1.apply` — three guards, all jumping to the no-op exit at 66:

| Guard | Offsets |
|---|---|
| `file instanceof BaseFile` | 6-9 (`ifeq 66`) |
| `entry.status() != DELETED` | 12-21 (`if_acmpeq 66`) |
| `file.firstRowId() == null` | 34-39 (`if_acmpne 66`) |

Then `setFirstRowId(nextRowId)` at 42-50 and `nextRowId += file.recordCount()` at 53-63. The
counter does **not** advance for a skipped entry — the next assignable file takes the id the
skipped entry did not consume.

`ManifestReader.open()` applies the assigner over all raw entries before live filtering, which is
why the port is a pass over the whole slice.

**Divergence (fork-only, fail-closed):** `DataFile::first_row_id` is `i64` while the counter is
`u64`. The assignment converts with `i64::try_from` and the addition uses `checked_add`; both
return `DataInvalid`. Java's `long` wraps silently, which would mint NEGATIVE row ids that alias
live rows. The `i64` door is the reachable one (`i64::MAX` < `u64::MAX`).

## V2 — the two reader rules (→ `arrow/record_batch_transformer.rs`)

| Java | Bytecode | Rule |
|---|---|---|
| `ValueReaders$RowIdReader.read` | stored id returned at 34-39; else `firstRowId + pos` | `_row_id` = stored value, else `first_row_id + physical ordinal` |
| `ValueReaders$LastUpdatedSeqReader.read` | stored value returned at 15-20; else `fileSeqNumber` at 21-28 | `_last_updated_sequence_number` = stored value, else the file's sequence number |
| `ValueReaders.rowIds(Long, reader)` | `ifnull` at 1 → `constant(null)` at 14-18 | absent `first_row_id` ⇒ an ALL-NULL column, not an error |
| `ValueReaders.lastUpdated(Long rowIdConst, Long fileSeq, reader)` | `ifnull 21` at BOTH offsets 1 and 5; `constant(null)` at 21-25 | null if EITHER input is null — a V1/V2 file reports NULL, not its sequence number |
| `ValueReaders.fileFieldReader` | `ROW_ID` compare at 0-13 then `rowIds` at 32; `LAST_UPDATED_SEQUENCE_NUMBER` compare at 39-52 then `lastUpdated` at 93 | special-cases a PRESENT file field with either reserved id; dispatch on whether the FILE carries the column |
| `MetadataColumns.<clinit>` | `NestedField.optional` at 144 (`_row_id`, id 2147483540) and 159 (`_last_updated_sequence_number`, id 2147483539) | both reserved fields are **optional** |

The fork implements these once, in the shared `RecordBatchTransformer`, rather than per format.
Java's own oracle for the ABSENT-column arm is the parquet reader
(`ParquetValueReaders$RowIdReader`, whose ctor substitutes `nulls()` for a missing column and whose
`read` then returns `firstRowId + pos`); the Avro class instead takes `constant(firstRowId)` via
`createMissingFieldReader`. The behaviour the fork implements matches the parquet oracle, which is
the one that describes a file lacking the column.

**Divergence (fork-only, fail-closed):** the `_row_id` computation checks `i64` overflow on the
LAST row of the batch (`start + num_rows - 1`, not `start + num_rows` — the latter rejects a batch
whose final id is exactly `i64::MAX`). Java's `long` addition wraps.

## Why the reserved ids must reach both readers

`is_metadata_field` covers every reserved column, but only the row-lineage pair can be PHYSICALLY
PRESENT in a data file. Both readers therefore need an exemption, and they are separate code paths:

* Parquet — `project_field_ids_without_metadata` (`is_row_lineage_field`), plus reserved-id
  resolution in `get_arrow_projection_mask`'s leaf expansion and its type check;
* Avro / ORC — `build_expected_schema`.

Exempting only one leaves the other silently substituting a computed id for a stored one. Both
arms are pinned by real-reader tests (`stored_row_id_in_the_data_file_survives_the_real_reader`,
`stored_row_id_in_an_avro_file_survives_the_real_reader`,
`stored_last_updated_sequence_number_survives_the_real_reader`).

`compare_schemas` must also force the `Modify` path for these columns: once the reserved fields
are correctly OPTIONAL, a file column matches the target exactly and the pass-through fast paths
hand it back verbatim — nulls included — which is precisely what the fallback exists to replace.

## R88 — variant scope

`unzip -l` over `iceberg-core-1.10.0.jar` returns variant classes only under `avro/`
(`ValueReaders$VariantReader`, `ValueWriters$Variant{Binary,Metadata,Value,}Writer`,
`VariantConversion`, `VariantLogicalType`) and `variants/`. Java's parquet variant lives in
`iceberg-parquet`, outside the parity scope. `TypeUtil$SchemaVisitor.variant(VariantType)` is a
bare `throw new UnsupportedOperationException("Unsupported type: variant")`, offsets 0-9 (`athrow`
at 9). The fork's canonical-extension-type emission is therefore a deliberate EXTENSION beyond
Java, recorded on R88.

## F-13 U3b — `DeleteWriteResult` (→ `DVWriteResult::referenced_data_files`)

| Java | Bytecode | Rule |
|---|---|---|
| `BaseDVFileWriter.close` | `write(PuffinWriter, Deletes)` at 199, then `CharSequenceSet.add(deletes.path())` at 205, once per iterated `Deletes` | the referenced set has exactly one entry per written blob, so it is DERIVABLE from `delete_files` |
| `BaseDVFileWriter.close` empty arm | `CharSequenceSet.empty()` built at 11; `deletesByPath.isEmpty()` early return at 19-45 | no deletes ⇒ empty set, `referencesDataFiles()` false |
| `DeleteWriteResult.referencesDataFiles` | `!= null && !isEmpty()` | non-empty test |
| `FileMetadata$Builder.build` | the DV `referencedDataFile` precondition at 158-173 (its `ldc_w` for "Referenced data file is required for DV" at 170) | a Puffin file without `referenced_data_file` is rejected, so the derivation's `filter_map` is total |

