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

# F-PARQUET-SIZE-1 — why a fork-written 50-row parquet file is larger than Spark's, measured first

**Date:** 2026-09-18. **Base:** `origin/main` `587d3592` (fork tip at branch
`fix/f-parquet-size-1`). **Model:** swe-2-high. **Path:** steps 1–2 MEASURE + CLASSIFY
(first commit), steps 3–4 FIX + MUTATION (later commits).

This ledger retires when the unit's fix lands or the owner removes the unit.

## Defect

The RePark COW-bytes shape `(id BIGINT, p INT, v STRING) PARTITIONED BY (p)`, format v2,
2 partitions × 4 single-INSERT files of 50 rows (`id` consecutive, `v = repeat('x',20)`),
zstd both sides: the fork writes each file at ~1.7 KB; Spark 4.1.2 + Iceberg 1.11.0
writes the same rows at ~1.1–1.2 KB. The size difference is why the ICE-RDF-GRANULARITY-1
cells cannot match Spark's `rewritten_bytes_count`.

## Measurement

Spark evidence: the run-23a COW-bytes oracle warehouse — `ns/t0/data/p=1/` and `t1..t3`,
4 files of 50 rows each per partition directory, inspected with `pyarrow.parquet` 25.0.1.

Fork evidence: a throwaway integration test drove the exact RePark write path —
`ParquetWriterBuilder::new(props, schema)` + `DataFileWriterBuilder` + a `PartitionKey` —
with `props = WriterProperties::builder().set_compression(parquet_compression_from_properties(&{}))`
(zstd level 3, the table-default path of `rewrite_data_files_write.rs:95-102`). The
50-row batch `id=200..249, p=1, v='x'*20` matches the Spark file whose stats read
`min=200 max=249`. A second probe with `set_dictionary_enabled(false)` reproduces the
INSERT path (`physical_plan/write.rs:298-308`, `parquet.enable.dictionary` unset).

### Per-file comparison (ids 200–249, both zstd, both 1 row group, both 50 rows)

| item | fork | Spark | Δ bytes |
|---|---|---|---|
| total file | 1430 | 1146 | +284 |
| footer serialized size | 965 | 707 | +258 |
| data section (before footer) | 457 | 431 | +26 |
| kv `ARROW:schema` | present, 528 B value | absent | +528 |
| kv `iceberg.schema` | absent | present, 194 B | −194 |
| `created_by` | `parquet-rs version 58.4.0` (24 B) | `parquet-mr version 1.17.1 (build 78a8d…)` (71 B) | −47 |
| col `id` chunk | ZSTD, enc PLAIN+RLE+RLE_DICTIONARY, dict page, 130→189 comp | ZSTD, enc PLAIN+BIT_PACKED, no dict page, 130 comp | +59 |
| col `p` chunk | ZSTD, RLE_DICTIONARY+dict page, 55 comp | ZSTD, PLAIN_DICTIONARY+dict page, 72 comp | −17 |
| col `v` chunk | ZSTD, RLE_DICTIONARY+dict page, 62 comp | ZSTD, PLAIN_DICTIONARY+dict page, 79 comp | −17 |
| column index / offset index | both present, 1 page each | both present, 1 page each | 0 |
| column statistics (min/max/nulls) | present | present | 0 |
| bloom filter | absent | absent | 0 |
| data page version | v1 | v1 | 0 |
| dict-off INSERT probe | 1294 | — | (dict off: v→PLAIN 42 B, p→PLAIN 38 B, id→PLAIN 122 B) |

Accounting: +528 (ARROW:schema) −194 (iceberg.schema) −47 (created_by) +26 (data
section) −29 (residual thrift: encoding lists, stats serialization) = **+284** = the
whole measured gap. The `ARROW:schema` key-value blob alone is ~37 % of the fork file.

(The brief's 1 694 B figure came from a RePark campaign measurement; this unit's own
probe reproduces the same gap at 1 430 B vs 1 146 B on the cited id range — the
mechanism and the ranking of causes are identical.)

## Step 2 — Java vs fork, per difference

### (a) Fork defaults diverging from Java writer behaviour

**A1 — `ARROW:schema` embedded. +528 B.** Java never writes this key; it is arrow-rs's
`AsyncArrowWriter::try_new` default (`skip_arrow_metadata=false`), reached at
`crates/iceberg/src/writer/file_writer/parquet_writer.rs:664`. parquet-rs 58.4:
`arrow/arrow_writer/mod.rs:254-257` calls `add_encoded_arrow_schema_to_metadata` unless
`ArrowWriterOptions::with_skip_arrow_metadata(true)`. **FIXABLE** —
`AsyncArrowWriter::try_new_with_options` exists (async_writer/mod.rs:173).
Read-path safety: the fork reads Spark/Java-written files — which never carry the key —
through the same `ArrowReaderMetadata::load` path every day (`arrow/reader.rs:466-517`;
the whole `tests/interop_spark_mor_fixtures.rs` suite reads Spark-written warehouses
through the full scan). Field ids are not lost: they live in the parquet schema's
`field_id` attribute, which arrow-rs propagates to `PARQUET:field_id` field metadata on
inference (parquet-rs `arrow/schema/mod.rs:503-517`). `schema_to_arrow_schema` emits no
extension types (uuid→`FixedSizeBinary(16)`, schema.rs:917-919), so nothing non-inferable
was carried by the blob.

**A2 — `iceberg.schema` absent. −194 B.** Java writes it unconditionally:
`Parquet$WriteBuilder.build()` → `meta("iceberg.schema", SchemaParser.toJson(schema))`
(bytecode-verified in iceberg 1.11.0 `Parquet$WriteBuilder.class`, `ldc "iceberg.schema"`,
`SchemaParser.toJson`). `Avro$WriteBuilder` does the same. The fork writes no Iceberg kv
at all — verified in the probe footer (only `ARROW:schema` present). **FIXABLE** —
`WriterProperties::into_builder().set_key_value_metadata(...)` (parquet-rs
file/properties.rs:281,796). Reader-visible effect: none for Iceberg/arrow-rs readers —
Java's core reader does not consume the key (only the two WriteBuilders reference it);
it exists for external tools reading a file standalone. **Note:** the fork's `Schema`
serde emits `{"schema-id":0,"type":"struct","fields":[…]}` — same fields and same byte
length as Java's `{"type":"struct","schema-id":0,…}`, different top-level key order.

**A3 — INSERT-path `parquet.enable.dictionary` default.** `physical_plan/write.rs:298-303`
reads the right Java property name (verified: Java `Parquet$WriteBuilder$Context.dataContext`
reads `parquet.enable.dictionary` via `propertyAsBoolean(..., true)`) but defaults it to
**false** when unset; Java defaults **true**. Deliberate fork choice in #288
(F-TARGET-FILE-SIZE-1) — and flipping it would move INSERT output *away* from Spark's
effective bytes: parquet-mr's first-page `isCompressionSatisfying` fallback (below)
turns Java's dict-on into PLAIN exactly on the columns where the dictionary costs more
than it saves, which arrow-rs cannot replicate. On this shape dict-off (1294) is already
closer to Spark (1146) than dict-on (1430). **NOT FIXED** — recorded, deliberate, and
the rewrite path already propagates per-column fallback from input footers
(`dictionary_fallback_columns`, rewrite_data_files_write.rs:488-522).

### (b) arrow-rs vs parquet-mr implementation differences (no Java-visible property)

**B1 — `id` column: dict page vs PLAIN. +59 B on this chunk.** parquet-mr
`FallbackValuesWriter.getBytes()` falls back to PLAIN on the first page when
`!(encodedSize + dictionaryByteSize < rawSize)` — parquet-mr 1.17.1
`FallbackValuesWriter.java` (getBytes, firstPage branch) +
`DictionaryValuesWriter.isCompressionSatisfying` (`(encodedSize + dictionaryByteSize) <
rawSize`). For 50 consecutive int64: ~45+400 ≥ 400 → PLAIN. arrow-rs has no equivalent
heuristic — `set_dictionary_enabled(true)` writes the dictionary unconditionally. On the
rewrite path the fork already converges to Spark's answer: `dictionary_fallback_columns`
detects `missing_dictionary_page`/`plain_data_pages` in the INPUT footers and disables
dict on those columns (rewrite_data_files_write.rs:98-101, 536-547) — Spark's PLAIN `id`
propagates to the rewrite output. On a fresh INSERT it cannot (no input evidence).

**B2 — data-page dictionary encoding label.** parquet-mr writes `PLAIN_DICTIONARY`
dictionary-encoded data pages (v1); arrow-rs writes `RLE_DICTIONARY` for the same
layout. −17 B per dict column here — impl detail, not a property.

**B3 — `created_by` string.** `parquet-rs version 58.4.0` vs `parquet-mr version 1.17.1
(build …)` — provenance must stay honest. NOT a fix target.

**B4 — thrift footer residual.** Encoding lists, page-encoding stats, stats field
serialization differ (−29 B). Impl detail.

## Fix taken (step 3)

Class-(a) items that are Java-divergent defaults, Cargo.toml-untouched, reader-invariant:

1. **Drop `ARROW:schema`** — `AsyncArrowWriter::try_new_with_options` with
   `with_skip_arrow_metadata(true)`, inside `ParquetWriter`'s lazy init.
2. **Add `iceberg.schema`** — `serde_json`-serialized `Schema` merged into
   `WriterProperties.key_value_metadata` (replacing any same-key entry), inside the same
   options build. Applies to data files AND delete files — Java's WriteBuilder stamps it
   on both (the Spark delete file carries `iceberg.schema` + `delete-type`; the
   `delete-type` key is a second delete-side gap, ledgered, not fixed here — the
   ParquetWriter layer cannot tell a delete file from a data file without plumbing).

Net footer effect on this shape: −528 +~210 → file ≈ 1 098 B vs Spark 1 146 B; residual
−48 B is created_by + thrift/encoding detail (class b).

Adjacent observations, not fixed: `delete-type` kv on delete files (Java
`Parquet$DeleteWriteBuilder` stamps `position`/`equality`); `iceberg.schema` on the Avro
data-file writer (`Avro$WriteBuilder` stamps it too).

## Verification plan

- Red-first cell: footer kv of a `DataFileWriter` file == `{iceberg.schema}` exactly —
  `ARROW:schema` absent, `iceberg.schema` parses back to the written schema.
- Round-trip pins for every Iceberg primitive incl. `timestamptz`, `timestamp_ns`,
  `uuid`, `fixed`, `decimal`, nested list/map/struct — write → read → values + field ids.
- Size cell: the 50-row RePark shape measured before (1 430) and after.
- Mutation: revert the options → footer cell red.

## Step 3–4 results — landed

**Implementation.** New `file_writer/parquet_footer.rs::writer_options` builds the
`ArrowWriterOptions` for every `ParquetWriter` (lazy init at
`parquet_writer.rs:664-668`): `props.into_builder().set_key_value_metadata` injects
`iceberg.schema` (replacing any same-key entry a caller supplied), and
`with_skip_arrow_metadata(true)` suppresses the arrow-rs blob. `parquet_writer.rs` was
at its legacy 3 391-line ceiling, so the helper lives in its own module; the test cell
is `file_writer/parquet_footer_tests.rs`.

**After-measurement (same probe, same rows).**

| variant | before | after | Spark |
|---|---|---|---|
| RePark path (dict default-on), ids 200–249 | 1 430 | **1 098** | 1 146 |
| RePark path, ids 250–299 | 1 467 | **1 135** | 1 178 |
| INSERT path (dict off), ids 200–249 | 1 294 | **962** | 1 146 |

Footer: 965 → 633 B (Spark 707 B); kv is now `{iceberg.schema: 194 B}` exactly. The fork
file is now *smaller* than Spark's — residual −48 B is `created_by` (24 vs 71 B) plus
thrift/encoding detail (class b). No fork-side change to encodings, compression, page
version, statistics, or indexes.

**Cells.** `parquet_footer_tests.rs` holds four cells:
`footer_key_values_match_java` (full `DataFileWriter` path, kv == `{iceberg.schema}` and
the JSON re-parses to the written schema), `footer_iceberg_schema_round_trips_all_types`
(same pin over a 21-field schema covering every primitive + list/map/struct),
`all_types_values_round_trip_without_arrow_schema` (write → `ParquetRecordBatchReader`
read → value equality), and `repark_shape_file_size_matches_java_scale` (the 50-row
shape must land < 1 300 B; measured 1 098).

**Reader-safety evidence.** `test_all_type_for_write`,
`test_parquet_writer_with_complex_schema`, the `write_defaults_tests` matrix, the
equality/position-delete write-read cells, and `data_file_writer_tests` all read
fork-written files back through `ParquetRecordBatchReader` inference — the same path
that already reads Spark files (which never carried the key). One physical-type note:
Iceberg `binary` writes as `LargeBinary`; without the blob, BYTE_ARRAY infers to
`Binary` (parquet-rs `arrow/schema/primitive.rs:289`) — identical to what a Spark file
of the same column already yields. The two shared check helpers
(`writer::tests::check_parquet_data_file`, the equality-delete twin) now cast read
columns to the expected Arrow type before value comparison; byte content is unchanged
by the cast. `field_id`s still reach the reader via parquet `field_id` →
`PARQUET:field_id` field metadata.

**Mutation.** Reverting `try_new_with_options` → `try_new` turned all three footer/size
cells red (`["ARROW:schema"]` vs `["iceberg.schema"]`; file back to 1 430 B) while the
value round-trip stayed green — the cells fail on exactly the fixed mechanism. Fix
restored; `cargo test -p iceberg --lib writer::` 162/162 green.

**Test-run coverage.** `writer::` 162/162, `arrow:: + scan:: + maintenance:: +
inspect::` 1 181/1 181, `iceberg-datafusion` write-filtered 15/15.

## Round 2 — R-01 perf + optional delete-type parity

**R-01 (P2).** The per-file lazy init used to re-run `writer_options` — and with it
`serde_json::to_string(schema)` plus the name/id-map clone inside `Schema`
serialisation — for every rolled file. `ParquetWriterBuilder` now stores the prepared
`ArrowWriterOptions` built once in `new_with_match_mode` and clones it into each
`ParquetWriter`; the lazy init calls `try_new_with_options` with the clone. A rolling
writer producing N files serialises the schema once, not N times. The builder boundary
is non-fallible (`new_with_match_mode -> Self`), so the serialisation `Result`
collapsed to an `expect` inside `writer_options` — `Schema` JSON serialisation is
infallible in practice and the old site could only have propagated the same failure
as a writer-build error. `parquet_writer.rs` stayed at its 3 391-line ceiling by
inlining the one-use `async_writer` local into the call.

Footer bytes are unchanged by construction: the same `WriterProperties` kv list and
the same `skip_arrow_metadata` flag reach every file. Pinned by the new
`rolled_files_from_one_builder_have_identical_bytes` cell — two `ParquetWriter`s
built from one `ParquetWriterBuilder` produce byte-identical files.

**Optional item — delete-type, taken.** Java's `Parquet.DeleteWriteBuilder` stamps
`delete-type=position|equality` on delete-file footers (observed on the run-23a
oracle's position deletes: `{delete-type, iceberg.schema}`). Because `writer_options`
preserves caller key-values ahead of the canonical `iceberg.schema`, the key rides in
through `WriterProperties` with no plumbing through the writer layers:
`position_delete_writer_properties[_for]` stamp `delete-type=position`, and new
public `equality_delete_writer_properties[_for]` helpers (re-exported from
`file_writer`) stamp `delete-type=equality` — the latter needed because no in-tree
production path builds equality-delete properties yet, so there was no existing
function to stamp. Footer key order matches Java: `{delete-type, iceberg.schema}`.
Cells `position_delete_footer_carries_delete_type` and
`equality_delete_footer_carries_delete_type` write real delete files through the
rolling writer and assert the two-key footer.

**Mutation re-run.** Same revert as round 1 — `new_with_match_mode` bypassed to a
plain `ArrowWriterOptions::new().with_properties(props)`: 5 of 8 footer cells red
(`footer_key_values_match_java`, `footer_iceberg_schema_round_trips_all_types`,
`repark_shape_file_size_matches_java_scale`, and now both delete-type cells, since
each asserts `iceberg.schema` at kv position 1). `rolled_files…identical_bytes` and
`all_types_values_round_trip` stay green by design — files are identical either way
and values read back either way. Fix restored; 8/8 green.

**Gates.** `writer::` 165/165, `arrow:: + scan::` 655/655, `iceberg-datafusion`
write-filtered 15/15, fmt clean, `clippy -p iceberg --all-targets -- -D warnings`
clean, size checker clean (`parquet_writer.rs` back at the 3 391 ceiling),
comment-ban `hits=0`.
