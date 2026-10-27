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

# F-POSDEL-BATCH-1 — position-delete writes in bounded batches

**Date:** 2026-10-27. **Base:** fork `main` `d5951e45`. **Branch:** `fix/posdel-batch-1`, commits
`2bbf5bff` and `653fbfb3`. **Fork PR:** #374. **Consumer:** RePark RP-60, which repins to this fix.

**Model:** builder Muse Spark (`muse-spark-1.3-contributor`). Records clerk Claude Haiku 5.5
(`claude-haiku-5-5`), which writes markdown only.

**Outcome in one line.** A DELETE that writes position deletes no longer panics with `offset
overflow` past `i32::MAX` path bytes. The writer cuts the deletes into bounded batches. A guard
turns any remaining overflow into an error.

## The fault at `d5951e45`

`write_position_deletes_for_partition` in
`crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs` built one
`StringArray` per partition group. The array held the full data-file path of every deleted row.
Past `i32::MAX` path bytes, the Arrow 32-bit offsets overflow. The process panics with `offset
overflow`. The RePark owner measured this on a format-v2 merge-on-read table at roughly 10 to 15
million deleted rows.

## The fix

- `chunk_position_delete_pairs` cuts the sorted `(path, pos)` pairs into consecutive ranges. A
  range ends when it holds `write.delete.position-delete-batch-rows` rows, or when it would pass
  256 MiB (`POSITION_DELETE_BATCH_BYTE_BOUND`) of path bytes. One pair always fits its own range.
- `position_delete_chunk_batch` builds one `RecordBatch` per range. It sums the path bytes first.
  Past `i32::MAX` it returns `DataFusionError::Internal`, naming the row count and the byte count.
  The check uses `i64::try_from`. No `as` cast, `unwrap` or `expect` enters a production path.
- `write_position_deletes_for_partition` writes each range in order, then calls `close` once. The
  pair order across ranges does not change, so the file-then-position sort still holds. `Utf8` is
  kept.
- New table property `write.delete.position-delete-batch-rows`, default `1_048_576`. The constants
  `TableProperties::PROPERTY_WRITE_DELETE_POSITION_DELETE_BATCH_ROWS` and its `_DEFAULT` sit in
  `crates/iceberg/src/spec/table_properties.rs`. A value below 1, or a value that is not a number,
  fails with a `DataFusionError::Plan` that names the property.
- The rolling writer keeps its cumulative counters across `write` calls. Chunked writes roll where
  one batch would roll.

## Pins

Each pin is in `crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs` unless
the clause says otherwise.

- `chunker_splits_twenty_five_rows_by_ten`: 25 rows with a row bound of 10 cut into `0..10`,
  `10..20`, `20..25`.
- `chunker_emits_single_pairs_when_two_paths_exceed_bytes`: five 100-byte paths with a 150-byte
  bound cut into five single-pair ranges.
- `chunker_enumerates_ranges_only_past_two_gibibytes`: 3000 paths of 819,200 bytes, past `i32::MAX`
  in total, with row bound 1, cut into 3000 single-pair ranges. Round 1 changed this pin; see
  Verification round 1.
- `batch_rows_default_applies_when_unset`: an unset property gives 1,048,576.
- `batch_rows_accepts_positive_value`: the value `10` parses to 10.
- `batch_rows_rejects_zero_and_non_numeric`: the values `0` and `abc` each fail with a Plan error
  that names the property.
- `chunk_batch_refuses_path_bytes_past_i32_max`: one path of 2,200,000,000 bytes returns `Internal`
  naming "1 rows" and "2200000000". This is the guard branch. Round 1 made it `#[ignore]`; see Verification round 1.
- `chunk_batch_one_chunk_past_i32_max_returns_internal` (`#[ignore]`): the whole 3000 x 800 KiB
  slice as one range returns `Internal` naming "3000 rows" and "2457600000". It does not panic.
- `stress_chunked_batches_cover_partition_past_i32_max` (`#[ignore]`): the same 3000 paths cut at
  the default row bound and the 256 MiB byte bound. The cut gives at least 10 ranges. Each range
  stays within the byte bound and builds one batch. The ranges hold 3000 rows in total.
- `test_posdel_batch_rows_delete_all_reads_empty` in
  `crates/integrations/datafusion/tests/posdel_batch.rs`: merge-on-read, batch rows 10, 25 rows
  (three ranges). A full delete reads empty and leaves one delete file.
- `test_posdel_batch_rows_partial_delete_survivors_by_value`, same file: deleting ids 1 to 13 leaves
  ids 14 to 25, in one delete file.
- `test_posdel_batch_rows_zero_refuses_delete`, same file: a value of 0 fails the DELETE at
  execution. The error names the property.

The integration pins prove that chunking keeps the behavior. SQL cannot count the ranges. The unit
pins and the stress pin count them.

A check of three or more delete files from 25 rows cannot hold on this path. Twenty-five rows never
reach the 1000-row roll boundary. This path builds the rolling writer with
`new_with_default_file_size`, which ignores `write.delete.target-file-size-bytes`. The fixture sets
that property small, and it is inert here. The integration pins assert one delete file.

## Red runs

Run 1, the shape before the fix. A temporary test built one `StringArray` over 3000 synthetic
800 KiB paths, 2,457,600,000 path bytes. It panicked with:

```
panicked at arrow-array-58.4.0/src/array/byte_array.rs:236:45:
offset overflow
```

The temporary test was removed after the run. Nothing temporary remains in the tree.

Run 2, pin `chunk_batch_one_chunk_past_i32_max_returns_internal`, with the guard bypassed
(`if !within_limit` replaced by `if false`, restored after the run). It panicked with:

```
thread '...::chunk_batch_one_chunk_past_i32_max_returns_internal' panicked at
arrow-array-58.4.0/src/array/byte_array.rs:236:45:
offset overflow
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 466 filtered out; finished in 5.44s
```

The registry path prefix is trimmed in both runs.

## Gates (builder, measured at `653fbfb3`)

- `cargo test -p iceberg-datafusion`: exit 0. This includes the three integration pins, the guard
  and chunker pins, and the two ignored pins listed as run below.
- The two ignored pins, run once each with `--ignored --exact`: exit 0 each. Test module path:
  `physical_plan::delete::delete_position_deletes::tests`.
- `cargo test -p iceberg --lib`: 4639 passed, 0 failed, 9 ignored.
- `make check`: fmt, clippy with `-D warnings`, taplo, machete, check-agent-artifacts,
  check-matrix-anchors, check-comment-blocks, and the Rust file-size gate (698 files clean). Exit 0.

## Follow-ups (out of scope)

1. The owned `String` per deleted row is still built before batching. Batching bounds the Arrow
   array. It does not bound the Rust heap held per deleted row.
2. `write.delete.target-file-size-bytes` is inert on this path. The writer is built with
   `new_with_default_file_size`, and only the rewrite planner reads that key.

## Verification round 1 (2026-10-27)

**Verified head:** `653fbfb3`. **Fix round:** commits `bd11bf3f` and `2bc85bb8`. The fix round
changes no production line. It adds one pin, changes two pins, and reworks one map row.

### Verifier verdict

The Opus 5.5 verifier (`claude-opus-5-5`) ran at `653fbfb3`. It returned FAIL on one S2 finding,
V-POSDEL-1. No committed pin covered the production loop. A wiring mutant that cut all pairs into
one range kept every pin green.

Every product claim passed:

- The pre-fix single-batch shape panics with `offset overflow` through the real writer.
- The head returns the 3000 rows in input order.
- Sort order across chunks holds under partition and file granularity.
- The property refuses `0`, `-1`, `abc`, empty, padded, fractional and overflowing values. It
  accepts `1`, `+5` and the default.
- The guard is cast-free. It returns `Internal` at `i32::MAX` plus one byte.
- Only the shared per-partition loop changed.
- `make check` is green.

The property and guard probes are verifier evidence. None of them is a committed pin.

Two S3 notes:

- Two plain tests allocated over 2 GiB each (V-POSDEL-2). The fix round removes both allocations
  from the plain run.
- The equality-delete conversion site in `crates/iceberg/src/maintenance/convert_equality_delete_files.rs`
  builds one `StringArray` over all pairs (line 538 at `2bc85bb8`). It has the same fault class. It
  is outside this unit. It should be carded on the RePark side.

### Fix round: pins new or changed

Each name is in `crates/integrations/datafusion/src/physical_plan/delete_position_deletes.rs`.

- `writer_loop_chunks_partition_past_i32_max` (new, `#[ignore]`, `#[tokio::test]`). It calls
  `write_position_deletes` on an unpartitioned v2 in-memory table with 3000 identical 800 KiB paths
  (2,457,600,000 path bytes). It asserts `Ok`, record counts that sum to 3000, and read-back pairs
  equal to the input in order. Green once: 20.93 s test time, peak RSS 5,666,052 KiB (5.40 GiB).
  The builder replayed the one-chunk wiring mutant as `let chunks = vec![0..pairs.len()];`, for one
  run, then restored the tree. The pin failed in 3.42 s with:

    chunked writer loop covers the partition: Internal("position-delete batch of 3000 rows holds 2457600000 path bytes, past the Utf8 StringArray limit")

  The pin re-ran green after the restore.
- `chunker_enumerates_ranges_only_past_two_gibibytes` (changed, plain). The fixture shrinks to 30
  pairs with row bound 1, and it asserts 30 single-pair ranges. A byte-bound check now asserts that
  uniform and varied same-length paths chunk identically: fifteen 2-wide ranges at byte bound 25.
  That proves the chunker reads path bytes only through `len()`. The name stays. The fixture no
  longer passes 2 GiB.
- `chunk_batch_refuses_path_bytes_past_i32_max` (changed, now `#[ignore]`). It was a plain test over
  one path of 2,200,000,000 bytes. The ignored one-chunk pin still covers the guard.

The plain module run, `cargo test -p iceberg-datafusion --lib delete_position_deletes`, gives 6
passed, 4 ignored, and peak RSS 123,496 KiB (about 121 MiB).

The four ignored pins, each run green once at head:

    cargo test -p iceberg-datafusion --lib writer_loop_chunks -- --ignored --exact physical_plan::delete::delete_position_deletes::tests::writer_loop_chunks_partition_past_i32_max
    cargo test -p iceberg-datafusion --lib chunk_batch_refuses -- --ignored --exact physical_plan::delete::delete_position_deletes::tests::chunk_batch_refuses_path_bytes_past_i32_max
    cargo test -p iceberg-datafusion --lib chunk_batch_one_chunk -- --ignored --exact physical_plan::delete::delete_position_deletes::tests::chunk_batch_one_chunk_past_i32_max_returns_internal
    cargo test -p iceberg-datafusion --lib stress_chunked_batches -- --ignored --exact physical_plan::delete::delete_position_deletes::tests::stress_chunked_batches_cover_partition_past_i32_max

The other unit pins and the three integration pins in `posdel_batch.rs` are unchanged.

### Gates at `2bc85bb8`

Builder record, not re-run by the clerk: `cargo test -p iceberg-datafusion` exit 0.
`cargo test -p iceberg --lib` 4639 passed, 0 failed, 9 ignored. `make check` green.
