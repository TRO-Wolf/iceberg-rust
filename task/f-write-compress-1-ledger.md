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

# F-WRITE-COMPRESS-1 — INSERT parquet compression from table properties

**Date:** 2026-09-11. **Branch:** `fix/f-write-compress-1`.
**Base:** `origin/main` `6232e39fe5f40830dd615d4371a8947a3c517767`.
**Model:** grok-4.6
**Path:** STANDARD because this changes a data-write path.

This ledger retires when the fork change merges or the owner removes the unit.

## Defect

`IcebergWriteExec::execute` built `ParquetWriterBuilder` with
`WriterProperties::default()`. parquet-rs default compression is UNCOMPRESSED, so
every `INSERT INTO` data file ignored `write.parquet.compression-codec` (Iceberg
default `zstd`) and `write.parquet.compression-level`. RePark run 7, ruling S2-16:
INSERT-grown footers had Σcompressed == Σuncompressed (ratio 1.000000) where Spark
writes zstd.

## Fix

- `TableProperties` constants: `PROPERTY_PARQUET_COMPRESSION_CODEC` /
  `PROPERTY_PARQUET_COMPRESSION_CODEC_DEFAULT` (`zstd`) /
  `PROPERTY_PARQUET_COMPRESSION_LEVEL` (no default).
- `parquet_compression_from_properties` in
  `crates/iceberg/src/writer/file_writer/parquet_compression.rs`. Codec names are
  case-insensitive. `lz4` / `lz4_raw` → `LZ4_RAW`. Level applies to zstd and gzip
  only. Unknown codec or non-integer / out-of-range level is `ErrorKind::DataInvalid`
  naming the key and value.
- `IcebergWriteExec::execute` builds
  `WriterProperties::builder().set_compression(<parsed>).build()` and maps parse
  errors through `to_datafusion_error`.

## File allowlist

- `crates/iceberg/src/spec/table_properties.rs`
- `crates/iceberg/src/writer/file_writer/parquet_compression.rs`
- `crates/iceberg/src/writer/file_writer/mod.rs`
- `crates/iceberg/src/writer/map.md`
- `crates/integrations/datafusion/src/physical_plan/write.rs`
- `crates/integrations/datafusion/src/physical_plan/map.md`
- `crates/integrations/datafusion/tests/insert_compression.rs`
- `crates/integrations/datafusion/tests/map.md`
- `task/f-write-compress-1-ledger.md`

`Cargo.toml`, `Cargo.lock`, every dependency file, `.github/`, and other writer
sites stay closed. Residue rows below list the skipped sites.

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Verdict | Evidence |
|---|---|---|---|---|
| C-001 | INSERT into a table with no codec property writes ZSTD column chunks. | `insert_into_default_table_writes_zstd` reads every live data-file footer. | PROVEN | Red on base: `unexpected column-chunk compression UNCOMPRESSED`. Green after fix: 4/4 `insert_compression` passed. |
| C-002 | `write.parquet.compression-codec` is honoured per codec; zstd level 3 is applied at parse and INSERT writes ZSTD. | `insert_into_honours_codec_property` (snappy / gzip / uncompressed) plus `insert_into_honours_zstd_level`. | PROVEN | Red: snappy/gzip/zstd-level all UNCOMPRESSED. Uncompressed is parquet-rs default, so that leg is green on base. Green after fix: SNAPPY, GZIP(_), UNCOMPRESSED, and ZSTD(_). Footer limitation below. |
| C-003 | Unknown codec / bad level fail loud, naming the property key and the bad value. | `insert_into_unknown_codec_fails_loud` (`brotli`); parse-helper bad-level unit pins. | PROVEN | Red: brotli INSERT committed `count=1` (silent fallback). Green: INSERT errors containing `write.parquet.compression-codec` and `brotli`. Unit pins: `abc`/`0`/`23` (zstd) and `nope`/`10` (gzip) are `DataInvalid` naming `write.parquet.compression-level`. |
| C-004 | The parse helper unit-pins default zstd, each codec, case-insensitivity, level honoured, bad level loud, unknown codec loud. | `cargo test -p iceberg --lib parquet_compression`. | PROVEN | 6 passed: `parquet_compression_default_is_zstd`, `parquet_compression_parses_each_codec`, `parquet_compression_codec_is_case_insensitive`, `parquet_compression_honours_zstd_and_gzip_level`, `parquet_compression_bad_level_fails_loud`, `parquet_compression_unknown_codec_fails_loud`. |

## Base-red evidence

`CARGO_BUILD_JOBS=16 cargo test -p iceberg-datafusion --test insert_compression -- --nocapture`
on the exact base (`WriterProperties::default()` still in `write.rs`) exited 101.

```
running 4 tests

thread 'insert_into_unknown_codec_fails_loud' panicked at
crates/integrations/datafusion/tests/insert_compression.rs:285:14:
brotli insert must fail: [RecordBatch { ... columns: [PrimitiveArray<UInt64> [ 1 ]] ... }]
test insert_into_unknown_codec_fails_loud ... FAILED

thread 'insert_into_honours_codec_property' panicked at
crates/integrations/datafusion/tests/insert_compression.rs:189:9:
unexpected column-chunk compression UNCOMPRESSED

thread 'insert_into_honours_zstd_level' panicked at
crates/integrations/datafusion/tests/insert_compression.rs:262:22:
expected ZSTD, got UNCOMPRESSED
test insert_into_honours_codec_property ... FAILED
test insert_into_honours_zstd_level ... FAILED

thread 'insert_into_default_table_writes_zstd' panicked at
crates/integrations/datafusion/tests/insert_compression.rs:189:9:
unexpected column-chunk compression UNCOMPRESSED
test insert_into_default_table_writes_zstd ... FAILED

test result: FAILED. 0 passed; 4 failed; 0 ignored; 0 measured; 0 filtered out
```

The `uncompressed` codec leg is the parquet-rs default. It is green on base. The
combined codec test failed on `snappy` first (`UNCOMPRESSED`), so that green leg
did not run in the red command. Mechanism: `WriterProperties::default()` is
UNCOMPRESSED.

## Footer limitation (measured)

parquet-rs 58.4 writes only the codec enum id into the column-chunk footer.
`ReadThrift` reconstructs `Compression::ZSTD(ZstdLevel::default())` (level 1) and
`Compression::GZIP(GzipLevel::default())`. After the INSERT path set
`ZSTD(ZstdLevel::try_new(3))`, `insert_into_honours_zstd_level` still read
`compression_level() == 1` from the footer:

```
assertion `left == right` failed: expected zstd compression_level 3, got 1
  left: 1
 right: 3
```

The INSERT pin therefore asserts footer `ZSTD(_)` plus
`parquet_compression_from_properties` returning `compression_level() == 3` for the
same properties `IcebergWriteExec` feeds to `set_compression`. Level 3 is pinned
on the parse helper (C-004). The footer cannot store the level.

## Residue (not changed this round)

Non-test sites in `crates/integrations/datafusion/src/` and `crates/iceberg/src/`
that still build a data-file or delete-file writer with `WriterProperties::default()`
or `WriterProperties::builder().build()`:

| Residue | Path:line | What it writes |
|---|---|---|
| F-WRITE-COMPRESS-1-R-001 | `crates/integrations/datafusion/src/physical_plan/row_lineage.rs:224` | COW DELETE/UPDATE and MoR UPDATE rewrite data files (`StreamingDataFileWriter`); `WriterProperties::default()` (UNCOMPRESSED) |
| F-WRITE-COMPRESS-1-R-002 | `crates/iceberg/src/maintenance/rewrite_data_files_write.rs:69` | `rewrite_data_files` compaction data files; `WriterProperties::builder().build()` (UNCOMPRESSED) |
| F-WRITE-COMPRESS-1-R-003 | `crates/iceberg/src/maintenance/partition_key_audit.rs:518` | partition-key repair rewrite data files; `WriterProperties::builder().build()` (UNCOMPRESSED) |
| F-WRITE-COMPRESS-1-R-004 | `crates/iceberg/src/writer/base_writer/position_delete_writer.rs:54` | `position_delete_writer_properties()` — parquet position-delete files; `WriterProperties::builder().set_statistics_truncate_length(None).build()` (UNCOMPRESSED). Callers reuse this helper. |

`crates/integrations/datafusion/src/task_writer.rs:364` and
`physical_plan/delete_tests.rs:616` are `#[cfg(test)]` and are not residue.

## Gates

| Command | Result |
|---|---|
| `cargo test -p iceberg-datafusion --test insert_compression` | exit 0; 4 passed, 0 failed |
| `cargo test -p iceberg --lib parquet_compression` | exit 0; 6 passed, 0 failed, 3671 filtered |
| `cargo test -p iceberg-datafusion --lib` | exit 0; 216 passed, 0 failed, 1 ignored (pre-existing measure) |
| `cargo test -p iceberg-datafusion --test insert_distribution` | exit 0; 7 passed, 0 failed |
| `make check` | exit 0 (fmt, clippy `-D warnings`, taplo, machete, agent-artifacts, matrix anchors 84 rows, comment-blocks, rust-file-size 451 files clean / 100 legacy) |

Docker-backed `make test` is excused (no Docker on this box).
