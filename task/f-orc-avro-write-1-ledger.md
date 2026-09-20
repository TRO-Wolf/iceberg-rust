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

# Ledger — F-ORC-AVRO-WRITE-1: hand-rolled ORC data-file writer (PR-1)

**Ledger id:** `F-ORC-AVRO-WRITE-1-2026-09-20`
**Branch:** `fix/f-orc-avro-write-1`
**Scope:** IPI-41 PR-1, ORC writer (record count + file size only; metrics are PR-2)
**Model:** muse-spark-1.3-contributor
**Found by:** Grok critic on `51f16bc60` (round 2: V-001..V-004)

## C-001. Why the ORC writer is hand-rolled

`orc-rust` 0.8 is already a workspace dependency and ships `ArrowWriterBuilder`,
but its writer cannot produce an Iceberg ORC file. Measured against the
vendored sources (`~/.cargo/registry/src/*/orc-rust-0.8.0` and `-0.9.0`):

1. `serialize_schema` stamps `attributes: vec![]` on every type (0.8.0
   `arrow_writer.rs:175`, 0.9.0 `arrow_writer.rs:209`) and exposes no builder
   to set them, so `iceberg.id` / `iceberg.required` can never be written.
   The fork reader refuses a file without those attributes.
2. `close(mut self) -> Result<()>` (0.8.0 `:136`, 0.9.0 `:160`, with
   `// TODO: return file metadata`) exposes no statistics, so per-column
   metrics are unreachable through it.
3. `serialize_schema` is flat, ends in `unimplemented!("unsupported datatype")`,
   and panics on `DECIMAL`, `LIST`, `MAP` and `STRUCT` rather than erroring.
4. `mod writer` and `mod encoding` are private (0.8.0 `lib.rs:58,72`), so no
   stripe writer and no encoder primitive can be reused from outside.
5. 0.9.0 keeps holes 1, 2 and 4 and needs `arrow 59.0` (`Cargo.toml:72`)
   while this workspace is on arrow 58.4. `datafusion-orc` 0.10.0 is a
   DataFusion integration over `orc-rust` 0.9, not a second ORC implementation.

So the writer is in-tree and adds no dependency: the DEFLATE encoder covers the `zlib`
codec and the protobuf is hand-written. `orc-rust` keeps the READ path and the
independent write oracle (the nested round-trip test decodes with its
`ArrowReader`, not with this crate's encoder).

## C-002. Java-default comparison (V-002)

Measured against Iceberg 1.11.0. These match: stripe size 67108864
(`write.orc.stripe-size-bytes`), compression `zlib`, file version [0, 12], and
the `iceberg.*` type attribute names and encodings (`iceberg.id`,
`iceberg.required`, `iceberg.long-type`, `iceberg.binary-type`,
`iceberg.length`, `iceberg.timestamp-unit`). The 256 KiB compression block is
the Apache ORC `bufferSize` default, NOT `write.orc.block-size-bytes` (the
268435456 HDFS block).

Two divergences stay, both ORC-spec shapes where every reader selects the
decoder from the declared kind, so values are identical and only the byte
layout differs. Row indexes and RLE v2 are NOT implemented in this PR:

| # | This writer | Java |
|---|---|---|
| 1 | Integer streams use RLE v1 (`ColumnEncoding.DIRECT`) | RLE v2 (`DIRECT_V2`) |
| 2 | Footer sets `rowIndexStride = 0` with no `ROW_INDEX` streams | Stride 10000 with row indexes |

## Round-2 remediation

| Finding | Fix |
|---|---|
| V-001 (P1) | The nested oracle now pins VALUES: list row 0 is `[1, 2, 3]`, row 1 is empty non-null, row 2 null; map row 0 is `k -> 1` with key and value asserted; struct row 0 is `x == 1, y == "z"`, row 1 children null, row 2 null |
| V-004 (P2) | `put_unbounded_varint_i128` zigzags in wrapping space; `encode_tests.rs` pins `i128::MIN`, `i128::MAX` and ±(10^38-1), and `test_orc_writer_round_trips_decimal_38_extremes` writes DECIMAL(38,9) extremes through `OrcWriter` |
| V-002 (P2) | This clause plus the exact map-card paragraph |
| V-003 (P3) | This ledger, shipped at the fork path; the map-card link now points here |
