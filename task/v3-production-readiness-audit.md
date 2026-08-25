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

# V3 production readiness — SEPMO Phase-0 scope audit

**Question asked:** what remains before RePark has full Iceberg **v3 spec production**
capabilities on this fork?

**Method:** every claim below is either a grep over the live tree at `d62fe54bd` (the negative
claims name the exact search that came back empty) or a `javap` disassembly of the pinned 1.10.0
oracle jars. Nothing here is inferred from documentation.

---

## Verdict

**As first written: four axes closed, two open (both ROW LINEAGE, chained). BOTH have since been CLOSED — V1 and V2 landed 2026-08-24 on this same branch. The narrative below is preserved as the audit's findings AT THE TIME; the per-unit sections carry an outcome note.** Two more (variant
shredded-parquet I/O, geospatial types) are v3 type-system features that are NOT on the
production-capability path unless RePark writes those types.

| v3 axis | State | Evidence |
|---|---|---|
| Deletion vectors (Puffin `deletion-vector-v1`) | ✅ closed | R114; write, read, merge, commit door, and V3 engine MOR all landed (#219, #221) |
| V3 delete-file rules (no parquet position deletes at v3) | ✅ closed | `RowDelta` refuses them at v3; the DV writer's pre-IO check refuses a DV over a live bounds-scoped delete |
| Default values (`initial-default` / `write-default`) | ✅ closed **on the data-path formats** | APPLIED on the two data-path readers — `record_batch_transformer.rs`'s `generate_transform_operations` (parquet, the one RePark reads) and `avro_reader.rs`'s `initial_default` fill. ORC is the exception and is NOT an application site: `orc_reader.rs`'s `missing_column_source` REFUSES a field carrying a non-null `initial_default` with `FeatureUnsupported`. That is Java-faithful — Java's ORC reader throws the same way (`ORCSchemaUtil.buildOrcProjection`) — so it is parity, not a gap, but it is a REFUSAL and must not be read as support. `write_default` correctly NOT gated at v2 (`table_metadata_builder.rs:4284-4306`) |
| Multi-argument transforms | ✅ out of scope | NOT in the oracle: `javap org.apache.iceberg.PartitionField` (1.10.0) has a single `private final int sourceId`. Parity is Java core 1.10.0, so this is not a fork gap |
| `timestamp_ns` / `timestamptz_ns` | ✅ closed | R90 ✅ (R162 🟡 is a `data_file` metadata-projection residue, not a data-path gap) |
| **Row lineage — `first_row_id` inheritance** | ❌ OPEN *(as audited; BUILT since — status lives on row R166)* | see V1 |
| **Row lineage — `_row_id` / `_last_updated_sequence_number` materialization** | ❌ OPEN *(as audited; BUILT since — status lives on row R166)* | see V2 |
| `variant` | 🟡 R88 | binary format done both directions. SCOPE CORRECTED below: shredded-PARQUET variant is NOT in the parity scope (Java's is in `iceberg-parquet`, not `iceberg-core`). The `variant_experimental` feature has since been enabled and the canonical Arrow extension type wired both directions; file-level I/O remains owed |
| `geometry` / `geography` | ❌ R89 | nothing exists |

---

## V1 — `DataFile.first_row_id` inheritance on manifest read

> **OUTCOME (2026-08-24): BUILT.** `spec/manifest/entry.rs` `assign_first_row_ids`, wired into
> `ManifestFile::load_manifest_with_schema_fallback`. See GAP_MATRIX row R166. The gap statement
> below describes the state BEFORE that change.

**The gap.** Java assigns each data file's `first_row_id` **at manifest-read time**, exactly as it
inherits sequence numbers. The fork does not: `spec/manifest/_serde.rs:159` passes the stored value
straight through (`first_row_id: value.first_row_id`), and no assigner exists anywhere in the crate.

**The oracle rule** (`javap org.apache.iceberg.ManifestReader` + `ManifestReader$1`, 1.10.0):

`idAssigner(Long firstRowId)` returns one of two functions:

- `firstRowId == null` → a function setting EVERY file's `firstRowId` to **null**
  (`lambda$idAssigner$2`). Note this OVERWRITES a stored value rather than preserving it.
- otherwise → a stateful `ManifestReader$1` with `nextRowId = firstRowId`, applying per entry:
  - **only if** the file is a `BaseFile` **and** `entry.status() != DELETED` **and**
    `file.firstRowId() == null` — three guards, each jumping to a no-op exit;
  - then `setFirstRowId(nextRowId)` and `nextRowId += file.recordCount()`.
  - The counter therefore does **not** advance for a DELETED entry, nor for a file that already
    carries a `firstRowId`.

Exact offsets and opcodes: [task/f13-v3-row-lineage-ledger.md](f13-v3-row-lineage-ledger.md), which
is their single home per AGENTS.md's evidence-routing table.

**Why it matters for production.** Without it, every data file read back carries
`first_row_id: None`, so V2 below has nothing to compute `_row_id` from. It is also the axis on
which a wrong implementation is silently wrong: an off-by-one in the counter mislabels rows
without failing anything.

**Size:** small — one assigner, applied where the manifest entry stream is built. Its closed test
domain is the three guards crossed with the null/non-null manifest arm; the DELETED and
already-assigned cells are the ones a naive running-total gets wrong.

## V2 — `_row_id` / `_last_updated_sequence_number` materialization at scan

> **OUTCOME (2026-08-24): BUILT** for Parquet, Avro AND ORC, at the shared
> `RecordBatchTransformer` — ORC shares `build_expected_schema` with Avro, so it gets the arm too;
> what is unverified there is the STORED-column half, because no in-repo ORC writer stamps
> `iceberg.id`. **GAP_MATRIX row R166 is the single home for the status and the residue** — read it
> there rather than here; this file records the audit's findings at the time, not the outcome's
> status. The evidence sentence below — "`RESERVED_FIELD_ID_ROW_ID` appears in exactly
> one file" — was true at `d62fe54bd` and is NO LONGER true at tip; it is kept because it is the
> finding that motivated the unit.

**The gap.** `RESERVED_FIELD_ID_ROW_ID` appears in **exactly one file** in the whole workspace —
`crates/iceberg/src/metadata_columns.rs`, where it is defined. `grep -rl` over `crates/` returns
that file alone. The constants, the `NestedField`s, and the name↔id maps all exist; nothing
consumes them. Selecting `_row_id` on a scan yields nothing today. `arrow/avro_reader.rs:82-85`
already names this as a known deferred gap.

**The oracle rules** (both `javap`-decoded, 1.10.0):

- `ValueReaders$RowIdReader.read` — read the file's `_row_id` column; if non-null return it; otherwise return `firstRowId + pos`, where `pos` comes from
  `ValueReaders.positions()` and `firstRowId` is the **file's** assigned value (V1's output).
- `ValueReaders$LastUpdatedSeqReader.read` — read the file's column; if non-null return it; otherwise return the file's `fileSeqNumber`.
- Dispatch is `ValueReaders.fileFieldReader`, which special-cases a **present** file field whose id
  is `MetadataColumns.ROW_ID` or `LAST_UPDATED_SEQUENCE_NUMBER`; an **absent** field takes the
  constant path.

**Blocked on:** V1 (hard — `firstRowId` is V1's output, and with V1 missing the fallback arm
computes from `None`).

**Size:** medium, and it touches all three readers. The parquet path is the one RePark needs; the
Avro path is the one with a direct core-jar oracle (`ValueReaders`); ORC's Java reader is outside
`iceberg-core`, so its arm is a fork-side extension of the same rule rather than a port.

---

## What is NOT on the production path

- **`variant` shredded-PARQUET I/O (R88).** CORRECTED 2026-08-24, and the correction changes what
  the unit is. Parquet variant support does **not exist in the parity scope at all**: `unzip -l`
  over the 1.10.0 `iceberg-core` jar returns variant classes only under `avro/` and `variants/` —
  `ValueReaders$VariantReader`, `ValueWriters$Variant{Binary,Metadata,Value,}Writer`,
  `VariantConversion`, `VariantLogicalType`. Java's parquet variant lives in the **`iceberg-parquet`
  module, outside `iceberg-core`/`iceberg-api`**. So shredded-parquet variant I/O would be a
  FORK-ONLY EXTENSION beyond Java core, gated on the `parquet` crate's experimental
  `variant_experimental` feature — not a parity gap with an oracle.

  The related fork refusal WAS also correct at the time of the audit: `arrow/schema.rs`'s
  `variant()` arm threw, and so does Java — `TypeUtil$SchemaVisitor.variant(VariantType)` is a bare
  `throw new UnsupportedOperationException("Unsupported type: variant")` (see the ledger for offsets),
  and `ArrowSchemaUtil`'s converter does not override it.

- **`variant` over AVRO — the in-scope unit R88 should actually name.** This one HAS an oracle in
  `iceberg-core` (the five classes above) and needs no dependency change. The fork already has the
  whole binary format both directions (`variant/` — metadata, value, shredded, visitor, write) and
  the Avro SCHEMA shape (`avro/schema.rs` `avro_variant_schema`, `is_variant_record_shape`). What
  is missing is the Avro→Iceberg direction, which refuses at the `AvroSchemaVisitor::variant` DEFAULT, and the data
  read/write plumbing behind it.
- **`geometry` / `geography` (R89).** A whole type family with no fork presence: types, JSON and
  Avro serde, transforms, metrics bounds, and Arrow mapping. This is its own block, not a residue.

Neither blocks a v3 table that does not use those types. Both are honest gaps in "full v3 type
system" and neither is reachable by tightening something that already exists.

---

## What I could not determine, and why

1. **Whether RePark actually needs `_row_id` selectable, or only correct `first_row_id` metadata.**
   V2's cost is dominated by the three reader arms. If RePark only needs lineage to survive
   rewrites, V1 plus a metadata-level accessor is materially cheaper. Named as a question for the
   user rather than assumed — the audit does not know RePark's consumer.
2. **Whether ORC's `_row_id` arm is owed at all.** Java's ORC reader is outside the parity scope
   (`iceberg-orc`, not `iceberg-core`), so that arm has no oracle. Flagged, not decided.
