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

# Ledger — F-RPD-TARGET-SMALL-1: `rewrite_position_delete_files` with `target-file-size-bytes=2000` rewrites 8→8 where Spark rewrites 0

**Ledger id:** `F-RPD-TARGET-SMALL-1`
**Branch:** `fix/f-rpd-target-small-1`
**Base:** `43fcd243`
**Oracle:** the run-23a RPD oracle (Spark 4.1.2 + Iceberg runtime 1.11.0)
**Model:** swe-2-high

| Step | Commit | Subject |
|---|---|---|
| 1 | this commit | `docs: F-RPD-TARGET-SMALL-1 — measured selection rule and delete-file bytes against Spark` |

## The regression

The oracle shape: `(id BIGINT, p INT, v STRING) PARTITIONED BY (p)`, format v2 merge-on-read,
16 data files (2 partitions × 2 rounds × 4 single-INSERT files, 50 rows each,
`v = repeat('x',20)`), `DELETE FROM t WHERE id % 2 = 0` → 8 file-scoped position-delete files,
25 positions each, data seq 9. `rewrite_position_delete_files(target-file-size-bytes=2000)`,
then again with `min-input-files=1` added. Spark answers 0 rewritten, 0 added, no new snapshot
for both; the fork answers 8 → 8.

## M-a — selection rule, Java 1.11.0 vs fork — PROVEN

Java (`org.apache.iceberg.actions.SizeBasedFileRewritePlanner`, 1.11.0 spark-runtime jar,
source + bytecode verified; the RPD action drives
`BinPackRewritePositionDeletePlanner extends SizeBasedFileRewritePlanner`):

| Piece | Java |
|---|---|
| Ratios | `MIN_FILE_SIZE_DEFAULT_RATIO = 0.75`, `MAX_FILE_SIZE_DEFAULT_RATIO = 1.80` |
| Thresholds (`sizeThresholds`) | `defaultMin = (long)(target * 0.75)`; `defaultMax = (long)(target * 1.80)` → at target 2000: min 1500, max 3600 |
| Candidate filter (`filterFiles`) | `outsideDesiredFileSizeRange(file)`: `file.fileSizeInBytes() < minFileSize || > maxFileSize` — files inside `[1500, 3600]` are not candidates |
| Group filter (`filterFileGroups`) | `enoughInputFiles(group) || enoughContent(group) || tooMuchContent(group)` — `enoughInputFiles`: `group.numFiles() > 1 && >= minInputFiles` (default 5); `enoughContent`: `group.numFiles() > 1 && totalSize > targetSize`; `tooMuchContent`: `totalSize > maxFileGroupSize` (default 100 GiB) |
| RPD defaults | `write.delete.target-file-size-bytes`, default 67108864 |

Fork (`crates/iceberg/src/maintenance/rewrite_position_delete_files.rs`,
`rewrite_data_files_plan.rs`):

| Piece | Fork |
|---|---|
| Ratios | `MIN_FILE_SIZE_DEFAULT_RATIO = 0.75`, `MAX_FILE_SIZE_DEFAULT_RATIO = 1.80` (`rewrite_data_files_plan.rs`) |
| Thresholds | `d2l(target as f64 * 0.75)` = 1500, `d2l(target as f64 * 1.80)` = 3600 at target 2000 |
| Candidate filter (`is_candidate`) | `length < min_file_size_bytes || length > max_file_size_bytes` |
| Group filter (`group_qualifies`) | `size > 1 && size >= min_input_files` (default 5) OR `size > 1 && input_size > target` OR `input_size > max_file_group_size` |
| RPD defaults | `write.delete.target-file-size-bytes`, default 67108864 |

**Conclusion: the rule is identical.** `is_candidate` is `outsideDesiredFileSizeRange`;
`group_qualifies` is `enoughInputFiles || enoughContent || tooMuchContent`. No planner change.

## M-b — delete-file bytes, fork writer vs Spark oracle — PROVEN

The comparison file was written through the same fork path RePark's merge-on-read DELETE drives:
`PositionDeleteFileWriterBuilder` + `position_delete_writer_properties_for(table properties)`
+ `MetricsConfig::for_position_delete`, ZSTD, 25 positions `0,2,…,48`, every `file_path` value
the same 109-byte path as the inspected oracle delete file. Oracle file: a `p=0` deletes file
from the run-23a RPD oracle warehouse (parquet-mr 1.17.1).

| item | fork | Spark | bytes |
|---|---|---|---|
| total size | 1375 | 1590 | −215 |
| footer size | 874 | 1108 | −234 |
| body (pages + indexes) | 489 | 470 | +19 |
| `created_by` | `parquet-rs version 58.4.0` (25 B) | `parquet-mr version 1.17.1 (build 78a8d32…)` (74 B) | −49 |
| schema root name | `arrow_schema` | `table` | +7 |
| key-values | `delete-type` (8 B), `iceberg.schema` (303 B) | `delete-type` (8 B), `iceberg.schema` (287 B) | +16 |
| `iceberg.schema` `file_path` doc | `Path of a file, used in position-based delete files` | `Path of a file in which a deleted row is stored` | +4 |
| `iceberg.schema` `pos` doc | `Ordinal position of a row, used in position-based delete files` | `Ordinal position of a deleted row in the data file` | +12 |
| `file_path` encodings | PLAIN, RLE, RLE_DICTIONARY (dict page + data page) | BIT_PACKED, PLAIN_DICTIONARY (dict page + data page) | ≈0 |
| `file_path` codec / comp / uncomp | ZSTD / 152 / 148 | ZSTD / 168 / 159 | −16 |
| `pos` encodings | PLAIN, RLE, RLE_DICTIONARY (dict page + data page) | BIT_PACKED, PLAIN (data page only) | dict page −36 |
| `pos` codec / comp / uncomp | ZSTD / 137 / 255 | ZSTD / 101 / 225 | +36 |
| `file_path` statistics | `max_value`, `min_value`, `null_count`, exactness flags — 2 copies of the path | deprecated `min`, `max` + `max_value`, `min_value`, `null_count` — 4 copies of the path | −218 |
| `pos` statistics | same fields + `is_max_value_exact`, `is_min_value_exact` | `min`, `max`, `max_value`, `min_value`, `null_count` | +4 |
| column index | present (15 B + 11 B) | present (15 B + 12 B) | −1 |
| offset index | present | present | ≈0 |
| `size_statistics` | present (repetition+definition 2725) | present (2725) | 0 |
| rows / row groups | 25 / 1 | 25 / 1 | 0 |
| `delete-type` kv | `position` | `position` | 0 |

RePark-equivalent write (table properties as RePark's helper builds them — ZSTD,
no `delete-type` kv): **1348 B** — the same layout minus the `delete-type` kv (−27 B). The
`delete-type` kv is a RePark-side writer-properties gap, out of scope here (RePark does not use
`position_delete_writer_properties_for`); the fork writer itself emits it.

The −215 B decomposes into **Java-visible writer divergences** the fork can change:

1. `iceberg.schema` field docs — fork strings are paraphrases; Java `MetadataColumns` carries
   `Path of a file in which a deleted row is stored` / `Ordinal position of a deleted row in the
   data file` (bytecode-verified). −16 B. Any reader diffing the kv bytes or consuming the docs
   sees a divergence.
2. `pos` dictionary-encoded — parquet-mr emits INT64 `pos` as PLAIN (no dict page) while dict-
   encoding `file_path`; the same INT64→PLAIN pattern holds in the oracle's data files (`id`).
   The global `dictionaryEnabled` is on (`parquet.enable.dictionary`, default true); the PLAIN
   result is a parquet-mr-internal encoding outcome. Reader-transparent; matching it removes the
   dict page. ≈ −36 B.
3. Parquet schema root name `arrow_schema` — parquet-mr writes `table` for every Iceberg file
   (data and delete files alike). −7 B.

And **parquet-rs internals** that cannot change without a Cargo change:

4. parquet-rs omits the deprecated unsigned-order `min`/`max` stats fields on binary columns —
   parquet-mr writes all four min/max variants, carrying the 109-B path twice more. −218 B.
   This is the dominant residual and is deliberate parquet-rs behavior.
5. `created_by` — honestly reporting `parquet-rs` rather than `parquet-mr`. −49 B.
6. `RLE_DICTIONARY` vs `PLAIN_DICTIONARY` encoding ids, `is_*_value_exact` flags, thrift field
   order, zstd framing. ≈ −3 B.

## M-c — the fork's planner on the RePark shape — PROVEN

Fork test helpers, 16 data files / 8 file-scoped delete files / 25 positions each:

| Input `file_size_in_bytes` | Option set | Result |
|---|---|---|
| 1585 (Spark's measured size) | `target-file-size-bytes=2000` | **0 rewritten, 0 added; snapshots 2→2** — Java's answer |
| 1585 | `target-file-size-bytes=2000`, `min-input-files=1` | **0 rewritten, 0 added** — Java's answer |
| 1293 (fork-written files through the table path) | `target-file-size-bytes=2000` | 8 rewritten, 8 added (10344 B) |
| 1293 | `target-file-size-bytes=2000`, `min-input-files=1` | 8 rewritten, 8 added |

Spark-sized delete files are inside `[1500, 3600]` → no candidates → `min-input-files` never
applies → no snapshot. The fork's planner is already Java's; the regression is file bytes:
fork delete files (1293–1375 B at this shape) fall below 1500 and are selected.

## Decision

**The rule is Java's; the gap is bytes.** Fix the three Java-divergent writer defaults (1–3
above), pinned red-first by footer cells plus the measured file size; the residual (~−270 B:
deprecated stats fields, `created_by`, encoding ids) is parquet-rs internals unreachable without
a Cargo change. Post-fix predicted size ≈ 1316 B — still below 1500, so the byte gap to Spark
is reduced, not closed; the residual goes to the hand-back as a measured question.

Pins:

- **Planner guard** (green today, added to keep the rule honest): the 16-file shape with
  `file_size_in_bytes = 1585` selects 0 under both option sets and adds no snapshot.
- **Footer cells** (red first): `iceberg.schema` kv equals Java's 287-B JSON byte-for-byte;
  `pos` column carries no dictionary page (data page PLAIN); parquet schema root is `table`.

## Fix — landed (commit `ac1bcf21`)

All three Java-divergent writer defaults fixed, no Cargo change:

1. `metadata_columns.rs` — `file_path` / `pos` field docs now Java's bytecode-verified strings
   (`Path of a file in which a deleted row is stored`, `Ordinal position of a deleted row in
   the data file`).
2. `parquet_footer.rs` — `java_ordered_schema_json` serializes `iceberg.schema` in Java's key
   order (`type`, `schema-id`, `identifier-field-ids`, `fields`); `with_schema_root("table")`
   replaces `arrow_schema` on every file rolled through `ParquetWriterBuilder` (data and delete
   alike — parquet-mr names the root `table` for both).
3. `position_delete_writer.rs` — both property builders set
   `set_column_dictionary_enabled(ColumnPath::from("pos"), false)`; `file_path` stays
   dictionary-encoded, matching the oracle's observed layout. Covers every production path:
   the DataFusion MoR DELETE physical plan, `rewrite_position_delete_files` output,
   `remove_dangling_delete_files`, `rewrite_table_path`, `convert_equality_delete_files`.

## Post-fix measurements — PROVEN

| Variant | Before | After |
|---|---|---|
| 25-pos delete file via `position_delete_writer_properties_for` (equal path length) | 1375 B | **1297 B** |
| RePark-equivalent props | 1348 B | 1325 B |
| `iceberg.schema` kv | 303 B | 287 B — byte-identical to Java's |
| `pos` encoding | RLE_DICTIONARY + dict page | PLAIN, no dict page |
| parquet root | `arrow_schema` | `table` |

Residual vs Spark (1585 B): −288 B = deprecated unsigned-order `min`/`max` stats on `file_path`
(−218 B; parquet-rs writes them only when `sort_order().is_signed()` — hard-coded, no knob),
`created_by` (−44 B; parquet-rs honestly reports itself), thrift/encoding-id/page layout
(~−26 B). **Nothing Java-visible remains.**

## Test results — PROVEN

- `floor_tests::test_spark_sized_delete_files_inside_the_band_are_not_candidates` — the oracle
  shape with Spark-sized metadata selects 0 under both option sets, no snapshot (green).
- `floor_tests::test_sub_min_delete_files_in_the_same_shape_are_rewritten` — the eligibility
  control: real fork-written files are sub-min and rewrite 8→8 (green).
- `parquet_footer_tests::position_delete_footer_matches_java_layout` — all Java-layout cells
  (green post-fix; red pre-fix on the `table` root cell).
- Maintenance suite: 102/102 green. Writer surface: 166 green.
- The dict→PLAIN change shrank output bytes; three fixture tests that calibrate roll/tail
  boundaries off measured file sizes needed recalibration: the shared fixture writer now uses
  `position_delete_writer_properties()` (production-shaped inputs — previously raw
  `WriterProperties` produced dict-encoded `pos`, an asymmetry nothing writes in production),
  and `test_roll_bound_is_write_max_not_target` raised positions 12000→13000 to keep the
  measured total over its 30×16 KiB precondition floor.

## Mutation — PROVEN

- Footer pin killed the `pos` mutant: flipping `set_column_dictionary_enabled("pos", …)` back
  to `true` fails the pin on the dictionary-page cell (observed).
- The same pin was red on the `arrow_schema` cell before the fix (commit `988da6e0`), green
  after — the `table`-root and Java-ordered-JSON cells are load-bearing.
- Planner pin is non-vacuous: the eligibility control rewrites the same table with real
  sub-min files 8→8, so the zero-result test cannot pass by a dead planner.

## Gates — PROVEN

`cargo fmt --check` clean; `cargo clippy -p iceberg --all-targets -- -D warnings` clean;
`check_rust_file_size.sh` 522 files clean (the tests file's legacy ceiling ratcheted 4607→4604
after the pin moved to the floor-tests sibling); `check_comment_blocks.sh`,
`check_matrix_anchors.sh`, `check_agent_artifacts.sh` OK; comment-ban hits=0.

## Final answer to the brief

The selection rule needed no fix — it is already Java's, and the pin proves it. The residual
8→8 at the oracle shape is produced by parquet-rs-vs-parquet-mr byte internals, not by any
Iceberg-level writer default. What the fork writes now matches Java on every observable axis
that exists at the Iceberg layer: schema kv bytes, root name, field docs, `delete-type` kv,
column encodings (`file_path` dict / `pos` PLAIN), untruncated stats, ZSTD, indexes, page
layout reach. The remaining −288 B is engine internals a reader can only see by diffing thrift
footers against a parquet-mr file.
