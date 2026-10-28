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

# F-POSDEL-FORMAT-READ-1 — the delete-file loader reads Avro and ORC position deletes

**Date:** 2026-10-28. **Base:** fork `main` `ea0b7741`. **Branch:**
`fix/posdel-format-read-1`, commits `9cef1328` (item 1), `5efbeeb8` (item 2),
and this commit (item 3, records only).

**Model:** builder Muse Spark (`muse-spark-1.3-contributor`).

**Outcome in one line.** A v2 merge-on-read table with `write.format.default`
`avro` or `orc` survives a DELETE: the next scan reads the Avro/ORC
position-delete files instead of failing with a Parquet corrupt-footer error.
Parquet behavior is byte-identical.

## The fault at `ea0b7741`

`CachingDeleteFileLoader::load_file_for_task` in
`crates/iceberg/src/arrow/caching_delete_file_loader.rs` opened every
non-Puffin position-delete task with `parquet_positional_delete_batch_stream`,
whatever `task.file_format` said, and every equality-delete task with
`parquet_to_batch_stream_with_projection`. The writer in
`crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs`
writes the file in the resolved delete format, so after a DELETE on an Avro or
ORC table the next scan failed with `Failed to load Parquet metadata ...
Corrupt footer`. Independent audit 2026-10-28, AUD-003.

## The fix

All dispatch lives in `crates/iceberg/src/arrow/delete_file_loader.rs`. The
caching loader only re-points its two call sites, shrinking by 7 lines (3263;
the legacy ceiling moves 3270 to 3263 in the item-1 commit). No reader file
changed: both whole-file readers already project by field id, so the brief's
conditional read-fully-and-select-by-name arm was not needed.

- `BasicDeleteFileLoader::position_delete_batch_stream(&task)`: Parquet keeps
  the exact old call; Avro/ORC read through `read_avro_data_file` /
  `read_orc_data_file` over `pos_delete_schema()` (the reserved `file_path` +
  `pos` pair, extra writer columns skipped by id) and stream the batches in
  file order; Puffin refuses `FeatureUnsupported` naming the file (unreachable
  from the caching loader, which routes Puffin to the deletion-vector path
  first).
- `BasicDeleteFileLoader::equality_delete_batch_stream(&task, &schema, ids)`:
  Parquet keeps the exact old call; Avro/ORC read over
  `equality_delete_expected_schema` (the top-level fields containing the key
  ids, in table order; unknown ids are skipped, and an empty subset yields an
  empty stream, which the equality parse folds to `AlwaysTrue` exactly as its
  empty-columns arm does) and then run the unchanged `evolve_schema`; Puffin
  refuses `FeatureUnsupported` naming the file. A nested key column resolves
  on Avro (the reader descends by id) and refuses loud on ORC (the reader's v1
  scope is top-level primitives).
- The `DeleteFileLoader::read_delete_file` trait impl routes through the same
  two methods, so the trait door and the caching door cannot drift.

## Clauses

| Id | Claim | Verdict | Pins |
|---|---|---|---|
| C-001 | Avro MoR table: `DELETE WHERE id BETWEEN 1 AND 3` scans `[0, 4]`; second `DELETE` of id 4 scans `[0]` | PROVEN | `test_mor_delete_on_avro_writes_avro_delete_file`, `test_avro_position_deletes_read_back_two_columns_in_order` |
| C-002 | Same, ORC | PROVEN | `test_mor_delete_on_orc_writes_orc_delete_file`, `test_orc_position_deletes_read_back_two_columns_in_order` |
| C-003 | Parquet MoR table: same two-delete shape answers `[0, 4]` then `[0]`; no parquet behavior changed | PROVEN | `test_mor_delete_parquet_reads_back_two_deletes`; full suites green (§Gates) |
| C-004 | Equality deletes dispatch by format and keep parquet semantics | OPEN | No Avro/ORC equality-delete fixture exists in the suite (Spark writes parquet equality deletes; the fork never writes equality deletes, it converts them), and `delete_file_loader.rs` sits 36 lines under its ceiling, so no direct pin fit. The parquet arm is the unchanged call, covered by the existing equality pins, all green. The Avro/ORC arms share the proven pos-delete reader plumbing and the unchanged `evolve_schema`. |
| C-005 | A format the loader cannot read refuses `FeatureUnsupported` naming the file and the format | OPEN | Same ceiling reason as C-004: no pin fit for the two Puffin-row refusal arms. Both are unreachable in-suite (Puffin position deletes route to the deletion-vector path first; Puffin equality deletes are never written). Each is a 6-line error constructor with the file path and the format in the message. |
| C-006 | No neighboring answer changed | PROVEN | Gates: `iceberg --lib arrow::` 537/0, `iceberg-datafusion` 802/0 across 43 suites, clippy `-D warnings`, fmt. Base `mor_delete_format` run (11/11) matches the 10 pre-existing tests at head plus the new control. |

## Pins

Each unit pin is in `crates/iceberg/src/arrow/delete_file_loader.rs`; each
integration pin in `crates/integrations/datafusion/tests/mor_delete_format.rs`.

- `test_avro_position_deletes_read_back_two_columns_in_order`: a 3-row
  unsorted `(file_path, pos)` fixture written by `AvroWriterBuilder` over the
  production `PositionDeleteWriterConfig` schema reads back as two columns
  named `file_path`, `pos` with the rows in file order.
- `test_orc_position_deletes_read_back_two_columns_in_order`: same through
  `OrcWriterBuilder`.
- `test_mor_delete_on_avro_writes_avro_delete_file` (extended): fixture grows
  to ids 0 to 4; after `DELETE WHERE id BETWEEN 1 AND 3` the scan answers
  `[0, 4]`; after `DELETE WHERE id = 4` it answers `[0]`. Suffix, magic, and
  files-table asserts unchanged.
- `test_mor_delete_on_orc_writes_orc_delete_file` (extended): same shape.
- `test_mor_delete_parquet_reads_back_two_deletes` (new): the parquet control
  for the same shape, with suffix and `PAR1` asserts.

## Red runs

Run 1, the brief's red-first protocol: new integration pins against the
`HEAD~1` loader (`git checkout HEAD~1 --` the two loader files, restored
after). 2 red out of 12; both extended tests fail in `surviving_ids` with:

```
execute `SELECT id FROM catalog.mor_delete_format_orc.target ORDER BY id`:
External error: Unexpected => Failed to load Parquet metadata, source:
Parquet error: Invalid Parquet file. Corrupt footer
```

The Avro twin fails identically. The 10 other tests, including the new
parquet control, stay green. After `git checkout HEAD --` the same files, 12
out of 12 green.

Run 2, unit-pin mutation: the Avro/ORC arm of
`position_delete_batch_stream` temporarily routed through
`parquet_positional_delete_batch_stream` (the old assumption, restored after).
2 red out of 2 in the focused run (full lib population 4650); both pins fail.
Restored tree: 2 green.

## Gates (builder, measured at `5efbeeb8`)

- `cargo test -p iceberg --lib arrow::`: 537 passed, 0 failed, 2 ignored.
- `cargo test -p iceberg-datafusion --test mor_delete_format`: 12 passed.
- `cargo test -p iceberg-datafusion`: 802 passed, 0 failed, across 43 suites.
- `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D warnings`:
  exit 0.
- `cargo fmt --all -- --check`: exit 0.
- `scripts/check_rust_file_size.py`: 698 files clean (after the 3270 to 3263
  ceiling move).

## Follow-ups (out of scope)

1. C-004 and C-005 want pins once a ceiling split frees room in
   `delete_file_loader.rs`, or an Avro/ORC equality-delete fixture exists.
2. `crates/iceberg/src/inspect/position_deletes.rs` (`read_delete_file_rows`)
   keeps its own door: the `position_deletes` metadata table reads Parquet
   delete files and Puffin deletion vectors and refuses Avro/ORC with a typed
   `FeatureUnsupported`. Untouched: beyond the loader and the two readers, so
   the brief excludes it. It already fails loud, never corrupt-footer.
