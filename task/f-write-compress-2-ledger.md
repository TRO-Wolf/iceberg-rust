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

# F-WRITE-COMPRESS-2 — maintenance/rewrite parquet compression from table properties

**Date:** 2026-09-11. **Branch:** `fix/f-write-compress-2`.
**Base:** `origin/main` `090bc8214` (F-WRITE-COMPRESS-1, #276).
**Model:** swe-2-high
**Path:** STANDARD because this changes data-write paths.

This ledger retires when the fork change merges or the owner removes the unit.

## Defect

`#276` (F-WRITE-COMPRESS-1) fixed `IcebergWriteExec` only. Four production writer
sites still built `WriterProperties` with the parquet-rs default, which is
UNCOMPRESSED, ignoring `write.parquet.compression-codec` (Iceberg default `zstd`)
and `write.parquet.compression-level`. RePark AP-1 re-measure: `rewrite_data_files`
compacted a 3.07 MB zstd bed into 7.93 MB of uncompressed output (~2.6x inflation).
The residue rows R-001…R-004 are listed in `task/f-write-compress-1-ledger.md`.

## Fix

Every site routes `table.metadata().properties()` through the existing
`parquet_compression_from_properties` parser (D-1 — no second parser) and sets the
result on `WriterProperties::builder().set_compression(..)`:

- `rewrite_data_files_write.rs` (R-002): the compaction writer's
  `WriterProperties::builder().build()` becomes `.set_compression(<parsed>)`.
- `row_lineage.rs` (R-001): `StreamingDataFileWriter` (COW DELETE/UPDATE and MoR
  UPDATE rewrite data files) parses the table properties, mapping errors through
  `to_datafusion_error`.
- `partition_key_audit.rs` (R-003): the repair rewrite writer takes the parsed
  codec.
- `position_delete_writer.rs` (R-004): new
  `position_delete_writer_properties_for(properties)` keeps
  `set_statistics_truncate_length(None)` verbatim (exact path bounds, D-2) and adds
  `.set_compression(parquet_compression_from_properties(properties)?)`. Its four
  production callers pass `table.metadata().properties()`:
  `delete_position_deletes.rs` (MoR DELETE position deletes),
  `rewrite_table_path.rs`, `rewrite_position_delete_files.rs`,
  `convert_equality_delete_files.rs`.

### Helper shape decision (D-2)

The card allowed either an argument or reading table properties. The chosen shape
is a sibling function `position_delete_writer_properties_for(&HashMap)` returning
`Result<WriterProperties>` — the RePark reference name — while the zero-arg
`position_delete_writer_properties()` stays verbatim (UNCOMPRESSED). Reasons:

- Every pre-existing test caller builds byte-size/roll fixtures measured under the
  old uncompressed default; the zero-arg helper keeps them compiling unchanged and
  byte-exact, so no test semantics silently shift.
- `docs/ENGINE_CONTRACT.md`, `docs/parity/GAP_MATRIX.md`, and existing comments
  name `position_delete_writer_properties`; keeping that symbol stable leaves no
  stale references.
- One parser: `_for` delegates to `parquet_compression_from_properties`.

Consequence pinned by tests: the shared fixture table in
`rewrite_position_delete_files_tests.rs` now declares
`write.parquet.compression-codec = uncompressed` explicitly, so the 92
byte-size/roll pins keep the exact base byte regime while production output
honours the table property.

## File allowlist

- `crates/iceberg/src/maintenance/rewrite_data_files_write.rs`
- `crates/iceberg/src/maintenance/partition_key_audit.rs`
- `crates/iceberg/src/maintenance/partition_key_audit_tests.rs`
- `crates/iceberg/src/maintenance/rewrite_table_path.rs`
- `crates/iceberg/src/maintenance/rewrite_position_delete_files.rs`
- `crates/iceberg/src/maintenance/rewrite_position_delete_files_tests.rs`
- `crates/iceberg/src/maintenance/convert_equality_delete_files.rs`
- `crates/iceberg/src/maintenance/map.md`
- `crates/iceberg/src/writer/base_writer/position_delete_writer.rs`
- `crates/iceberg/src/writer/map.md`
- `crates/integrations/datafusion/src/physical_plan/row_lineage.rs`
- `crates/integrations/datafusion/src/physical_plan/delete.rs`
- `crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs`
- `crates/integrations/datafusion/src/physical_plan/map.md`
- `crates/integrations/datafusion/tests/rewrite_compression.rs`
- `crates/integrations/datafusion/tests/map.md`
- `task/f-write-compress-2-ledger.md`

`Cargo.toml`, `Cargo.lock`, every dependency file, and `.github/` stay closed.

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Verdict | Evidence |
|---|---|---|---|---|
| C-001 | `rewrite_data_files` compaction output carries the table codec (default `zstd`) in every column chunk. | `rewrite_data_files_output_carries_table_codec` reads every rewritten data-file footer. | PROVEN | Red on base: `unexpected column-chunk compression UNCOMPRESSED in .../compacted-00000-01a09399-744d-7c71-8ba5-5f1f532790d5.parquet`. Green after fix: 3/3 `rewrite_compression` passed. |
| C-002 | COW DELETE rewritten data files carry the table codec in every column chunk (R-001, `row_lineage.rs`). | `cow_delete_rewrite_output_carries_table_codec` runs a copy-on-write DELETE and reads the rewritten file's footer. | PROVEN | Red on base: `unexpected column-chunk compression UNCOMPRESSED in .../01a09399-741e-78e3-8610-232e890ce803-00000.parquet`. Green after fix: ZSTD in every chunk. |
| C-003 | MoR DELETE position-delete files written through the shared helper carry the table codec (R-004). | `mor_delete_position_delete_file_carries_table_codec` runs a merge-on-read DELETE and reads the `pos-del-*.parquet` footer. | PROVEN | Red on base: `unexpected column-chunk compression UNCOMPRESSED in .../pos-del-00000-01a09399-740a-7a11-985f-a8015807cd71.parquet`. Green after fix: ZSTD in every chunk. |
| C-004 | Partition-key repair rewrite output carries the table codec (R-003). | `maintenance::partition_key_audit::tests::test_repair_rewritten_files_carry_the_table_codec` repairs a miskeyed fixture and reads the `repaired-*.parquet` footer. | PROVEN | Red on base: `unexpected column-chunk compression UNCOMPRESSED in .../t/data/dept=ops/name_trunc=bet/repaired-00000-01a0939a-3872-7962-8679-b259cb45fcfa.parquet`. Green after fix: 1 passed, 3677 filtered. |

## Base-red evidence

`CARGO_BUILD_JOBS=16 cargo test -p iceberg-datafusion --test rewrite_compression`
on the base tree (production sites reverted to `090bc8214`, pins in place)
exited 101:

```
---- cow_delete_rewrite_output_carries_table_codec stdout ----
thread 'cow_delete_rewrite_output_carries_table_codec' panicked at
crates/integrations/datafusion/tests/rewrite_compression.rs:166:13:
unexpected column-chunk compression UNCOMPRESSED in
/tmp/.tmp9iUvZf/target/data/01a09399-741e-78e3-8610-232e890ce803-00000.parquet

---- rewrite_data_files_output_carries_table_codec stdout ----
thread 'rewrite_data_files_output_carries_table_codec' panicked at
crates/integrations/datafusion/tests/rewrite_compression.rs:166:13:
unexpected column-chunk compression UNCOMPRESSED in
/tmp/.tmpqK60Nh/target/data/compacted-00000-01a09399-744d-7c71-8ba5-5f1f532790d5.parquet

---- mor_delete_position_delete_file_carries_table_codec stdout ----
thread 'mor_delete_position_delete_file_carries_table_codec' panicked at
crates/integrations/datafusion/tests/rewrite_compression.rs:166:13:
unexpected column-chunk compression UNCOMPRESSED in
/tmp/.tmp83N7WN/target/data/pos-del-00000-01a09399-740a-7a11-985f-a8015807cd71.parquet

failures:
    cow_delete_rewrite_output_carries_table_codec
    mor_delete_position_delete_file_carries_table_codec
    rewrite_data_files_output_carries_table_codec

test result: FAILED. 0 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out
```

`CARGO_BUILD_JOBS=16 cargo test -p iceberg --lib
test_repair_rewritten_files_carry_the_table_codec` on the same base exited 101:

```
---- maintenance::partition_key_audit::tests::test_repair_rewritten_files_carry_the_table_codec stdout ----
thread 'maintenance::partition_key_audit::tests::test_repair_rewritten_files_carry_the_table_codec' panicked at
crates/iceberg/src/maintenance/partition_key_audit_tests.rs:933:17:
unexpected column-chunk compression UNCOMPRESSED in
/tmp/.tmpZcBtXq/ns-d1e9a8a0-e1b0-4242-8b4e-cc05982dcf49/t/data/dept=ops/name_trunc=bet/repaired-00000-01a0939a-3872-7962-8679-b259cb45fcfa.parquet

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 3677 filtered out
```

Mechanism: each site built `WriterProperties::default()` (or a builder without
`set_compression`), and parquet-rs defaults to UNCOMPRESSED.

## Footer limitation (measured, carried from F-WRITE-COMPRESS-1)

parquet-rs 58.4 writes only the codec enum id into the column-chunk footer, so a
configured zstd/gzip level cannot be reconstructed from the footer. The pins
assert `ZSTD(_)` per column chunk; level parsing stays pinned on
`parquet_compression_from_properties` (F-WRITE-COMPRESS-1 C-004).

## Residue closed

| Residue (F-WRITE-COMPRESS-1 ledger) | Path | Closed by |
|---|---|---|
| R-001 | `crates/integrations/datafusion/src/physical_plan/row_lineage.rs` | C-002 |
| R-002 | `crates/iceberg/src/maintenance/rewrite_data_files_write.rs` | C-001 |
| R-003 | `crates/iceberg/src/maintenance/partition_key_audit.rs` | C-004 |
| R-004 | `crates/iceberg/src/writer/base_writer/position_delete_writer.rs` | C-003 + `_for` callers |

No remaining production writer site builds data-file or position-delete-file
`WriterProperties` without the table codec.

## Gates

| Command | Result |
|---|---|
| `cargo test -p iceberg-datafusion --test rewrite_compression` | exit 0; 3 passed, 0 failed |
| `cargo test -p iceberg --lib test_repair_rewritten_files_carry_the_table_codec` | exit 0; 1 passed, 0 failed, 3677 filtered |
| `cargo test -p iceberg --lib` | exit 0; 3670 passed, 0 failed, 8 ignored |
| `cargo test -p iceberg-datafusion --lib` | exit 0; 216 passed, 0 failed, 1 ignored (pre-existing measure) |
| `cargo test -p iceberg-datafusion --test insert_compression` | exit 0; 4 passed, 0 failed |
| `cargo test -p iceberg-datafusion --test insert_distribution` | exit 0; 7 passed, 0 failed |
| `cargo test -p iceberg --lib rewrite_position_delete_files` | exit 0; 92 passed, 0 failed |
| `make check` | exit 0 (fmt, clippy `-D warnings`, taplo, machete, agent-artifacts, matrix anchors 84 rows, comment-blocks, rust-file-size 452 files clean / 100 legacy) |

Docker-backed `make test` is excused (no Docker on this box).

## Out-of-scope observations

- `crates/integrations/datafusion/src/task_writer.rs:364` and
  `physical_plan/delete_tests.rs:616` remain `#[cfg(test)]` uncompressed writers —
  test fixtures, not residue (carried from the F-WRITE-COMPRESS-1 ledger).
- The maintenance `map.md` did not list `partition_key_audit.rs` /
  `partition_key_audit_tests.rs` (pre-existing gap); both rows were added in this
  change because this unit touches them.
