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

# map.md — crates/iceberg/src/writer/file_writer/

## Purpose

The physical file layer: Arrow `RecordBatch` in, one data or delete file on storage out, plus the
`DataFileBuilder` that carries its manifest statistics. Everything here implements the two traits in
`mod.rs` — `FileWriterBuilder` (a cloneable recipe) and `FileWriter` (one open file). The layers
above (`../base_writer/`, `../partitioning/`, `rolling_writer.rs`) are generic over those traits, so
a format is chosen only by which concrete builder is instantiated.

## Contents

| Path | What it does |
|---|---|
| `mod.rs` | the `FileWriterBuilder` / `FileWriter` traits and the format re-exports |
| `parquet_writer.rs` | Parquet IO and its per-column metrics; the reference implementation of the per-column metrics contract |
| `parquet_compression.rs` | `write.parquet.compression-codec` / `-level` → `parquet::basic::Compression` |
| `parquet_footer.rs` | the Java-identical parquet footer key-values (`iceberg.schema`, `delete-type`) |
| `parquet_footer_tests.rs`, `parquet_writer_unsupported_tests.rs` | the `#[cfg(test)]` cells of the two above |
| `avro_writer.rs` | Avro OCF data files. Metrics are **row count + file size only** — Java `AvroMetrics.fromWriter` returns `Metrics(rowCount, null, null, null, null)`, and Spark's manifests confirm it (every `readable_metrics` field is NULL on an Avro data file). Adding column metrics here would be a divergence, not an improvement |
| `avro_reject.rs` | the shared variant / unknown refusal the Avro writer applies at `build()` |
| `orc_writer.rs` | ORC data files: `OrcWriterBuilder` / `OrcWriter`. Buffers each batch as Iceberg `Literal` rows (the same path `avro_writer.rs` uses), encodes them into ORC stripes, then writes the footer itself so every non-root type carries Java's `iceberg.id` / `iceberg.required` attributes (the root carries field names only, like Java). The `DataFileBuilder` carries record count + file size + split offsets, no column metrics |
| `orc_writer/orc_type.rs` | Iceberg schema → the pre-order ORC type list, with Java `ORCSchemaUtil`'s attributes (`iceberg.id`, `iceberg.required`, `iceberg.long-type`, `iceberg.binary-type`, `iceberg.length`, `iceberg.timestamp-unit`). `variant` and `unknown` are refused here by name |
| `orc_writer/encode.rs` | the ORC stream primitives: base-128 varints, byte RLE, boolean RLE, integer RLE **v1**, and the ORC compression-chunk framing (NONE and ZLIB = raw DEFLATE) |
| `orc_writer/column.rs` | the per-column stream builders. One recursive walk over the `Literal` row fills each column's PRESENT / DATA / LENGTH / SECONDARY buffers; a child only receives a value for the rows where its parent is present, which is ORC's nesting rule |
| `orc_writer/footer_write.rs` | the hand-rolled protobuf writer for `StripeFooter`, `Footer` and `PostScript` — the mirror image of `../../arrow/orc_reader/footer.rs`, which hand-parses them. `orc-rust` keeps its `writer` and `encoding` modules private and stamps `attributes: vec![]` on every type, so neither its writer nor its encoders can produce an Iceberg ORC file |
| `orc_writer/null_repair.rs` | forces list/map `Literal` slots to null wherever the Arrow column is null, recursing into structs; `schema_has_container` gates the repair |
| `orc_writer_tests.rs`, `orc_writer_layout_tests.rs`, `orc_writer/*_tests.rs` | the `#[cfg(test)]` cells for the ORC writer |
| `rolling_writer.rs` | size-based rolling over any `FileWriterBuilder` |
| `location_generator.rs` | file naming and placement; the extension comes from `DataFileFormat`'s `Display` |

## The ORC design, and why it is in-tree

`orc-rust` is the ORC **reader** this crate uses, and its writer cannot produce an Iceberg ORC file:

- `arrow_writer.rs` stamps `attributes: vec![]` on every serialized type in both 0.8.0 and 0.9.0, so
  no `iceberg.id` reaches the footer. The fork's ORC reader refuses a file without those attributes
  rather than resolving by name, so such a file is unreadable here and in Java-by-field-id.
- Its `serialize_schema` is flat and ends in `unimplemented!("unsupported datatype")`: `DECIMAL`,
  `LIST`, `MAP` and `STRUCT` panic rather than erroring.
- `mod writer` and `mod encoding` are **private** in 0.8.0 and 0.9.0, so no stripe writer and no
  encoder primitive can be reused from outside the crate.
- 0.9.0 needs arrow 59; this workspace is on arrow 58.4. `datafusion-orc` 0.10.0 is a DataFusion
  integration **over** `orc-rust` 0.9, not a second ORC implementation.

So the writer is in-tree and adds **no dependency**: the DEFLATE encoder (already used by the ORC footer reader)
covers Java's default `zlib` codec, and the protobuf is hand-written exactly as the reader's is. The
full evidence and the rejected alternatives are clause C-001 of
[task/f-orc-avro-write-1-ledger.md](../../../../../task/f-orc-avro-write-1-ledger.md).

Defaults that match Iceberg 1.11.0: stripe size 67108864 (`write.orc.stripe-size-bytes`),
compression `zlib`, file version [0, 12], and the `iceberg.*` type attribute names and encodings.
The 256 KiB compression block is the Apache ORC `bufferSize` default, not
`write.orc.block-size-bytes` (the 268435456 HDFS block). Two divergences stay: integer streams use
RLE **v1** (`ColumnEncoding.DIRECT`) where Java writes RLE v2 (`DIRECT_V2`), and the footer sets
`rowIndexStride = 0` with no `ROW_INDEX` streams where Java uses stride 10000. Both are ORC-spec
shapes and every reader selects the decoder from the declared encoding kind, so the values are
identical; only the byte layout differs. Clause C-002 of the ledger records the comparison.

## I want to...

| I want to... | go to |
|---|---|
| Add a physical format | implement `FileWriterBuilder` / `FileWriter` |
| Change what statistics a data file carries | `parquet_writer.rs` (parquet); Avro carries row count + file size only, ORC adds split offsets; neither carries column metrics |
| Change an ORC byte encoding | `orc_writer/encode.rs`, then re-run the round-trip tests — they decode with `orc-rust`, not with this crate's encoder |
| Understand how a nested ORC column is laid out | `orc_writer/column.rs` — the present/length/child recursion |
| Read an ORC file | `../../arrow/orc_reader.rs` (and its `footer.rs` for the field-id map) |

## Pointers

- **Up:** [crates/iceberg/src/writer/](../map.md) · **Related:** `../../arrow/orc_reader/`
  (the read half of the ORC footer contract),
  [../../transaction/map.md](../../transaction/map.md) (commits what this produces)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| `ORC data file carries no iceberg.id type attributes` when reading a file this crate wrote | The footer splice was skipped or a type lost its attributes. `orc_writer/footer_write.rs::encode_type` is the only place they are written |
| An ORC round trip returns the right rows but the wrong nulls | A child column received a value for a row where its parent was null. ORC gives a child one entry per **present** parent row; see `orc_writer/column.rs::append` |
| A timestamp before 1970 comes back one second off | ORC-763. The reader subtracts a second when the stored second is negative and the nanos exceed 999_999, so the writer stores the second truncated **toward zero**, which is what Java does. `split_timestamp` pins both halves |
| Java reads garbage integers from a Rust-written ORC file | The declared `ColumnEncoding` and the encoder disagree. This writer declares `DIRECT` everywhere and must therefore write RLE v1 |

### First checks

- Dump the footer of the written file and compare it to a Spark-written one: the type list, the
  attributes, and the stripe footer's stream list are where a divergence shows first.
- Round-trip through `orc-rust`'s own Arrow reader before blaming this crate's ORC reader — that
  separates an encoding bug from a projection bug.

### Escalate to

- Read-side questions → `../../arrow/orc_reader.rs`.
- Cross-engine readability → [dev/java-interop/map.md#debug](../../../../../dev/java-interop/map.md#debug).
