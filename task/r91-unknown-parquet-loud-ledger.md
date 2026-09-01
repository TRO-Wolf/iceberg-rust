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

# Evidence ledger — R91 parquet write refuses `unknown` loud

Capability status lives on GAP_MATRIX row R91. This file holds the red-leg
measurement, write-door enumeration, and pin names.

## Charter

```yaml
LEDGER:
  - id: C-001
    proposition: >
      DataFileWriter::write of a RecordBatch with an Arrow Null column for Iceberg
      unknown currently commits a parquet file that the Iceberg reader cannot
      visit (Cannot visit Arrow data type: Null). After the fix the same write
      returns FeatureUnsupported and names the unknown type (and the column).
    verdict: PROVEN
    evidence: >
      Red-leg at 33be9a0f4: data_file_writer_refuses_unknown_null_column_loud
      asserted silent write+close, file on disk, arrow_schema_to_schema failed
      naming Null (1 passed / 11 data_file_writer_ filter, 3541 filtered).
      After the guard: the same test asserts FeatureUnsupported, message contains
      "Writing the unknown column" and "unknown" and column 'u', empty temp dir.
  - id: C-002
    proposition: >
      A neighbouring DataFileWriter write of a batch with no unknown/Null column
      still writes a parquet file and reads the values back.
    verdict: PROVEN
    evidence: data_file_writer_writes_and_reads_back_int_string_batch
  - id: C-003
    proposition: >
      Every writer door that can reach parquet with an Arrow Null / Iceberg
      unknown column is enumerated. Reachable doors are pinned. Unreachable
      doors are recorded as such. Scope stays the loud-write refusal.
    verdict: PROVEN
    evidence: write-door table below plus
      an_unknown_schema_is_refused_before_any_bytes_are_written (all four
      container depths + top level) and
      data_file_writer_refuses_omitted_optional_unknown_column
```

## Red leg (base `33be9a0f4`)

Command: `cargo test -p iceberg --locked --lib data_file_writer_`

Result: 11 passed, 0 failed, 3541 filtered. Includes
`data_file_writer_refuses_unknown_null_column_loud` asserting:

1. `DataFileWriter::write` of `{id: Int32, u: Null}` against Iceberg
   `{id: int, u: unknown}` returned `Ok`.
2. `close()` returned one `DataFile`.
3. The parquet file existed on disk.
4. `arrow_schema_to_schema` on the parquet-arrow schema failed with a
   message containing `null` (`Cannot visit Arrow data type: Null`).

That is the silent-commit-then-unreadable-file bug row R91 claimed was
already deferred-loud via `arrow/value.rs` only.

## Flip

`ParquetWriterBuilder::build` calls `reject_unknown_write`, which walks the
Iceberg schema at every depth (struct / list element / map key / map value)
with the same 128 depth bound as `variant_path_within`. First `unknown`
returns `FeatureUnsupported`: `Writing the unknown column '{path}' is not
supported yet: unknown is always null and has no physical column`.

`DataFileWriter::write` hits this on the first batch because
`RollingFileWriter` builds the parquet writer lazily then. No bytes land.

Java `TypeToMessageType` emits no parquet column for unknown. The fork maps
unknown to Arrow `Null` and parquet-rs will write that type. Reader-side
always-null synth is the bigger change; this unit refuses the write instead
(brief: reader-accepts-Null is not chosen).

A table whose parquet writer schema still contains `unknown` cannot emit a
data file until that synth lands. That is the deferred-loud posture, not a
regression of a currently-correct path: today that path writes an unreadable
file.

## Write-door enumeration

| Door | Reaches parquet with Arrow Null / Iceberg unknown? | Pin |
|---|---|---|
| `ParquetWriterBuilder::build` | Yes — Iceberg schema is converted to Arrow `Null` | `an_unknown_schema_is_refused_before_any_bytes_are_written` (top, struct, list, map key, map value) |
| `DataFileWriter::write` | Yes — rolling writer builds parquet on first write. `apply_write_defaults` also fills a missing optional unknown as `NullArray` | `data_file_writer_refuses_unknown_null_column_loud`; `data_file_writer_refuses_omitted_optional_unknown_column` |
| `RollingFileWriter` | Passthrough; not a separate type door | covered by the two above |
| Fanout / clustered / unpartitioned writers | Wrap `DataFileWriter`; inherit the parquet builder schema | inherit |
| `EqualityDeleteFileWriter` | Only if the *projected* parquet schema carries unknown (equality id is the unknown field). Callers convert the projected Arrow schema back to Iceberg; a Null field fails `arrow_schema_to_schema` first. If the caller passes the full table schema including unused unknown, parquet `build` now refuses | not pinned as a separate public write of unknown equality ids (Unknown is not an eq-delete value type). Full-schema misuse is the parquet builder pin |
| `PositionDeleteFileWriter` | No — schema is reserved `file_path` + `pos` only | unreachable |
| `AvroWriterBuilder::build` | Already refused `PrimitiveType::Unknown` at build (`reject_unsupported_field`) | pre-existing; out of parquet scope |

## Mutation (one knob)

Removed only `reject_unknown_write(self.schema.as_ref())?;` in
`ParquetWriterBuilder::build`.

- `cargo test -p iceberg --locked --lib -- data_file_writer_refuses` → **2 red
  out of 2** executed (3552 filtered). Tests:
  `data_file_writer_refuses_unknown_null_column_loud`,
  `data_file_writer_refuses_omitted_optional_unknown_column`.
- `cargo test -p iceberg --locked --lib -- an_unknown_schema_is_refused_before_any_bytes_are_written`
  → **1 red out of 1** executed (3553 filtered).

Guard restored. Variant pin was not in this mutation set (separate function).

## Pins

- `data_file_writer_refuses_unknown_null_column_loud`
- `data_file_writer_refuses_omitted_optional_unknown_column`
- `data_file_writer_writes_and_reads_back_int_string_batch`
- `an_unknown_schema_is_refused_before_any_bytes_are_written`
- `a_variant_schema_is_refused_before_any_bytes_are_written` (moved, not
  behavior-changed)

## Out of scope

- Always-null parquet column omission + reader synth (Java `TypeToMessageType`
  / deferred `arrow/value.rs` read).
- RePark engine pin `fork_unknown_write_commits_then_scan_refuses_naming_null`
  (separate repo).
