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
  ids, in table order; an id absent from the current schema refuses
  `DataInvalid` naming the file and the field id, so a dropped key fails loud
  instead of applying no delete — fold-2 V-1) and then run the unchanged
  `evolve_schema`; Puffin refuses `FeatureUnsupported` naming the file. A
  nested key column resolves on Avro (the reader descends by id) and refuses
  loud on ORC (the reader's v1 scope is top-level primitives).
- The `DeleteFileLoader::read_delete_file` trait impl routes through the same
  two methods, so the trait door and the caching door cannot drift.

## Clauses

| Id | Claim | Verdict | Pins |
|---|---|---|---|
| C-001 | Avro MoR table: `DELETE WHERE id BETWEEN 1 AND 3` scans `[0, 4]`; second `DELETE` of id 4 scans `[0]` | PROVEN | `test_mor_delete_on_avro_writes_avro_delete_file`, `test_avro_position_deletes_read_back_two_columns_in_order` |
| C-002 | Same, ORC | PROVEN | `test_mor_delete_on_orc_writes_orc_delete_file`, `test_orc_position_deletes_read_back_two_columns_in_order` |
| C-003 | Parquet MoR table: same two-delete shape answers `[0, 4]` then `[0]`; no parquet behavior changed | PROVEN | `test_mor_delete_parquet_reads_back_two_deletes`; full suites green (§Gates) |
| C-004 | Equality deletes dispatch by format and keep parquet semantics | PROVEN | Live-key happy path, closed in fold-2: the verifier measured Avro-on-Parquet, ORC-on-Parquet, Avro-on-Avro, ORC-on-ORC and the Parquet control each answering `y = [0, 10, 40]` after an equality delete on `y in {20, 30}`, key-less projections included; pinned in-suite by `test_avro_equality_delete_with_live_key_applies_without_key_projection` and the ORC twin in `crates/iceberg/tests/mor_eqdel_format.rs`. Dropped-key refusal is C-007. |
| C-005 | A format the loader cannot read refuses `FeatureUnsupported` naming the file and the format | PROVEN | Verifier-measured at `53fc7177`: `position_delete_batch_stream` and the trait door refuse "Cannot read position deletes from Puffin file '<path>' as delete rows: ..."; `equality_delete_batch_stream` refuses "Cannot read equality deletes from Puffin file '<path>': ...". No committed pin: both arms are unreachable in-suite (Puffin position deletes route to the deletion-vector path first; Puffin equality deletes are never written). |
| C-006 | No neighboring answer changed | PROVEN | Gates: `iceberg --lib arrow::` 537/0, `iceberg-datafusion` 802/0 across 43 suites, clippy `-D warnings`, fmt. Base `mor_delete_format` run (11/11) matches the 10 pre-existing tests at head plus the new control. |
| C-007 | An equality key absent from the current schema refuses instead of applying no delete | PROVEN | Fold-2 V-1: `equality_delete_expected_schema` returns `DataInvalid` naming the file and the field id; the empty-expected-schema arm now only covers an empty id list. Red-first: pre-fix the Avro and ORC dropped-key scans answered all five rows; pinned by `test_avro_equality_delete_with_dropped_key_refuses` and the ORC twin in `crates/iceberg/tests/mor_eqdel_format.rs`. |
| C-008 | The legacy position-delete load dispatches on file format | PROVEN | Fold-2 V-2: `load_legacy_positions_by_path` builds a `FileScanTaskDeleteFile` and reads through `position_delete_batch_stream` instead of the Parquet loader. Pinned by `avro_position_delete_loads_positions_by_path` (red against the Parquet-only call, restored green). Parquet behavior unchanged: the partition-scoped call is identical, and the file-scoped projection difference is invisible to the pos-only parser. |
| C-009 | Parquet-only delete opens outside the dispatched doors stay open as follow-ups, not defects of this PR | OPEN | `maintenance/rewrite_position_delete_files.rs:617` skips non-Parquet position deletes with `continue` (Avro/ORC deletes uncompacted; data correct); `maintenance/rewrite_table_path.rs:446` refuses non-Parquet with a typed error; `maintenance/convert_equality_delete_files.rs:601` has no format gate (Avro/ORC fail loud with the Parquet footer error, by code reading). |
| C-010 | 2026-10-28: C-007's interim refusal of a dropped Avro or ORC equality key is superseded by F-EQDEL-DROPPED-KEY-1 fold-2: the per-file equality-delete schema now resolves the key from the delete file's own schema for every format, and the dropped-key sequences answer the survivors `a, c, e` | PROVEN | `test_avro_equality_delete_with_dropped_key_answers_survivors` and `test_orc_equality_delete_with_dropped_key_answers_survivors` in `crates/iceberg/tests/mor_eqdel_format.rs` (renamed from the two C-007 refusal pins and flipped); record in `task/f-eqdel-dropped-key-1-ledger.md` (clause F2-0) |

VERDICT (2026-10-28, appended by F-EQDEL-DROPPED-KEY-1 fold-2): counts unchanged for C-001 to C-009 (8 PROVEN, 1 OPEN: C-009); new row C-010 PROVEN, recording that C-007's interim refusal is superseded. C-007 is left as recorded.

## Pins

Unit pins live in `crates/iceberg/src/arrow/delete_file_loader.rs`; the
position-delete integration pins in
`crates/integrations/datafusion/tests/mor_delete_format.rs`; the
equality-delete format pins in `crates/iceberg/tests/mor_eqdel_format.rs`;
the legacy-door pin in `delete_vector_container/legacy.rs`.

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
  ceiling move; 699 at the fold-2 head with the new `mor_eqdel_format.rs`
  test file).

## Fold-2 — verifier round 1 (one S1, two S3)

**Date:** 2026-10-28. **Branch:** `fix/posdel-format-read-1`, commits
`e887c9ec` (item 1), `cb22348c` (item 2), `4e18f3c9` (item 3), `06835def`
(item 4, records), `53fc7177` (fmt fixup), and this commit (fold-3,
records only).

**Model:** builder Muse Spark (`muse-spark-1.3-contributor`).

- **V-1, S1.** An equality id absent from the current schema was skipped in
  `equality_delete_expected_schema`; with every key dropped the empty expected
  schema produced an empty stream that folded to `AlwaysTrue`, so Avro/ORC
  scans silently resurrected deleted rows. Fixed in `e887c9ec`: any
  unresolvable id refuses `DataInvalid` naming the file and the field id, and
  the empty-stream arm now only covers an empty id list (the Parquet-arm
  outcome for that edge). The field-id resolution that closes the sequence
  for every format lands in F-EQDEL-DROPPED-KEY-1, out of scope here.
- **V-2, S3.** `load_legacy_positions_by_path` opened every live non-DV
  position delete with the Parquet loader. Fixed in `4e18f3c9`: it builds a
  `FileScanTaskDeleteFile` and reads through `position_delete_batch_stream`.
  The two required doc lines on the touched items were reworded in place
  (parquet-only text was false after the dispatch); no net-new comments.
- **V-3, S3.** This ledger described the unknown-id skip as a design choice;
  it was the defect. C-004 is rewritten and closed PROVEN for the live-key
  happy path on the verifier's measurement plus the new pins; C-007
  (dropped-key refusal) and C-008 (legacy door) are new PROVEN rows. Map rows:
  `crates/iceberg/tests/map.md` gains the new test file;
  `crates/iceberg/src/arrow/map.md` gains the refusal clause.
  `delete_vector_container/` has no `map.md` (nor does `src/`), so per the
  navigation rule none was created for the V-2 door.

Red runs, fold-2: the four new `mor_eqdel_format` tests run 2 red / 2 green
pre-fix (both dropped-key scans answered all five rows; both live-key pins
already green), 4 green post-fix. The V-2 unit pin runs red against the
temporarily restored Parquet-only call and green after the restore.

Gate results for the fold-2 head live in the round hand-back, not here: the
brief runs the gate after the last commit.

## Verifier rounds

Round 1 (Opus 5.5): FAIL with one S1 and two S3 findings. V-1 S1: the
unknown-id skip resurrected deleted rows (fixed in `e887c9ec`, C-007). V-2
S3: the legacy door opened every live non-DV position delete with the
Parquet loader (fixed in `4e18f3c9`, C-008). V-3 S3: this ledger described
the unknown-id skip as a design choice; it was the defect (C-004 rewritten,
C-007/C-008 added).

Round 2 (fresh Opus 5.5 at `53fc7177`): PASS. V-1 closed: Avro and ORC
equality deletes refuse `DataInvalid` on a dropped key and apply live keys
under key-less projections, with renamed keys applying by field id; the
dropped-key pins went red on the fold-1 sources and the legacy pin went red
under a dispatch-call mutation. V-2 closed: the legacy door builds the task
from the `DataFile`, and an ORC position-delete probe read back the expected
partition-scoped and file-scoped positions. The Puffin arms measured as
stated in C-005. Three S3 wording items, all fixed in this fold-3 commit:
C-005 carries its own reason, the `## Pins` intro names every pin home, and
the fold-2 commit list plus the file-size count are corrected.

## Follow-ups (out of scope)

1. C-004 is pinned by `crates/iceberg/tests/mor_eqdel_format.rs` in
   fold-2 (the pins live outside `delete_file_loader.rs`, so no ceiling split
   was needed). C-005 still wants pins once a ceiling split frees room in
   `delete_file_loader.rs`, or a Puffin-row fixture exists.
2. `crates/iceberg/src/inspect/position_deletes.rs` (`read_delete_file_rows`)
   keeps its own door: the `position_deletes` metadata table reads Parquet
   delete files and Puffin deletion vectors and refuses Avro/ORC with a typed
   `FeatureUnsupported`. Untouched: beyond the loader and the two readers, so
   the brief excludes it. It already fails loud, never corrupt-footer.
