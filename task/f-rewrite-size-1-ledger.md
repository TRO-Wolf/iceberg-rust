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

# F-REWRITE-SIZE-1 — why `rewrite_data_files` output is ~1.5× its input's compressed bytes, and the fix

**Date:** 2026-09-13 (step 1, branch `measure/f-rewrite-size-1`); 2026-09-12 (step 2, branch
`fix/f-rewrite-size-1`).
**Base:** `origin/main` `41e25ba2b99a0d90caf4317b4236f5e1d4b683f5` (fork tip at branch).
**Model:** swe-2-high
**Path:** step 1 MEASURE — no product code change. Step 2 (the fix) landed on this
measurement; see the step-2 sections below.

This ledger retires when the unit's fix lands or the owner removes the unit.

## Defect (RePark run, ruling S2-24, 2026-09-12)

206 zstd `INSERT` files — Σ compressed 2 831 692 B, Σ uncompressed 7 529 566 B
(ratio 0.3762) — were compacted by `rewrite_data_files` into 20 zstd files of
4 474 081 B (uniform) / 4 136 632 B (skewed): ~1.5× the input's compressed bytes for
the same rows and the same codec. Compaction on a zstd table is a net-size loss.

## Measurement design

Bed (`crates/integrations/datafusion/tests/rewrite_size_shared/mod.rs`): the RePark
`bench/adaptpart` uniform bed copied field-for-field — `ts TIMESTAMP, grp STRING,
id BIGINT`, 206 batches of 2 000 rows, `grp = g{batch % 20}`, `id` and `ts` globally
monotonic (`ts = BASE_US + (base+row)*SPAN_US//total`), one `INSERT INTO` per batch
against a `PARTITIONED BY (grp)` table on a memory catalog + local-fs warehouse.

The RePark input files were written by **Spark**, whose parquet zstd default level is
3. The fork honors `write.parquet.compression-level`, so the bed is built with that
property set to `3` (the Spark-equivalent input), then the property is removed via a
transaction before `rewrite_data_files` runs — reproducing exactly what RePark
measured: input written at zstd 3, rewrite written at the fork's `ZstdLevel::default()`
= **1**. The probe also runs the symmetric controls in the flip table (zstd-3 rewrite
over the level-3 input, and the level-1 baseline).

Probe targets (all `#[ignore]`d; CI does not run them):

- `rewrite_size_probe` — full measurement + defect-reproduction assertions (green on
  base, because the defect is present).
- `rewrite_size_pin` — the forward pin for step 2: rewrite output ≤ 1.1× input
  compressed bytes. Red on base (evidence below).

## File allowlist

Step 1 (measure only):

- `crates/integrations/datafusion/tests/rewrite_size_probe.rs`
- `crates/integrations/datafusion/tests/rewrite_size_pin.rs`
- `crates/integrations/datafusion/tests/rewrite_size_shared/mod.rs`
- `crates/integrations/datafusion/tests/map.md`
- `task/f-rewrite-size-1-ledger.md`

Step 2 adds:

- `crates/iceberg/src/writer/file_writer/parquet_compression.rs`
- `crates/iceberg/src/maintenance/rewrite_data_files_write.rs`
- `crates/iceberg/src/maintenance/map.md`
- `crates/iceberg/src/writer/map.md`

No `Cargo.toml`/`Cargo.lock`, no `.github/`. `row_lineage.rs` was examined and left
untouched: `StreamingDataFileWriter` writes new DML rows, not a rewrite of identified
input files, so it has no input footers to consult; it inherits the D-2 level-3 default
through `parquet_compression_from_properties` like every other writer path.

## Proposition ledger

| Clause | Checkable proposition | Proof obligation | Verdict | Evidence |
|---|---|---|---|---|
| C-001 | The bed reproduces the RePark input ratio: Σ compressed ≈ 0.38 × Σ uncompressed over ≥150 small INSERT files. | `rewrite_size_probe` asserts `files ≥ 150` and ratio ∈ [0.30, 0.50]. | PROVEN | 206 files; Σ compressed **3 263 419 B**, Σ uncompressed **7 755 694 B**, ratio **0.4208**. Residual vs RePark's 0.3762 is the writer's encoding detail (the fork's writer dictionary-encodes the unique-valued `ts`; Spark's effective encoding differs), not the bed shape — uncompressed totals match within 3 % and row distributions are identical. At the fork's own default (zstd 1) the same bed lands at 4 279 399 B, ratio 0.5518 — level 3 is what moves the input toward Spark's number. |
| C-002 | Base-tree `rewrite_data_files` output is ≥ 1.4× the input's compressed bytes — the defect reproduces in the fork. | `rewrite_size_probe` asserts `out/in ≥ 1.4`; the `rewrite_size_pin` forward pin must FAIL on base. | PROVEN | Real `rewrite_data_files` (rewritten=206, added=20, groups=20): output Σ compressed **4 782 768–4 789 094 B** across runs → out/in **1.4656–1.4675**. Pin red output below. |
| C-003 | Table of every `WriterProperties` value each writer path reaches. | Source diff of `write.rs` vs `rewrite_data_files_write.rs` + runtime dump of the built `WriterProperties`. | PROVEN | Both paths construct `WriterProperties::builder().set_compression(<parsed>).build()` — identical objects. Full property table below. The only construction differences are outside `WriterProperties`: `FieldMatchMode::Name` (insert) vs `Id` (rewrite), DataFusion 2 000-row batches vs 8 192-row reader batches, and TaskWriter fanout vs the splitter/router. |
| C-004 | One-at-a-time flip table names the single property (or sort) bringing the rewrite within 10 % of input compressed bytes. | `manual_rewrite` replays the real input tasks through `ParquetWriterBuilder` with each candidate flipped once. | PROVEN | **`dictionary_enabled` is the named cause.** `set_dictionary_enabled(false)` alone: σ compressed **3 292 655 B → out/in 1.0090** (≤ 1.1). zstd level 3 alone: 1.1325; level 9: 1.1015 — the level mismatch contributes but does not reach 10 % by itself. Sorting by the input's `id` order: 1.4572 — order is not the cause. Every other candidate ≈ 1.472. Full table below. |
|| C-005 | D-1: the rewrite writer disables dictionary encoding only on columns whose input chunks show fallback; the high-cardinality bed's rewrite output stays ≤ 1.05× input. | `rewrite_size_pin::rewrite_output_within_five_percent_of_input` (un-ignored, tightened from 1.1 to 1.05) green. | PROVEN | Red on base: `got 1.4654 (4782370 vs 3263419)`. Green with the fix: output Σ compressed **2 181 668 B** vs input **3 263 419 B** → out/in **0.6686** (20 files). `ts`/`id` footers show no dictionary page; `grp` keeps it. |
|| C-006 | Low-cardinality columns keep the dictionary: a `grp`-like column with 8 values over 200 k rows rewrites ≤ 1.05× input with dict pages still on. | `rewrite_size_pin::rewrite_keeps_dictionary_on_low_cardinality_columns` green: size bound AND per-chunk dict-page/data-page-encoding assertions. | PROVEN | 100 input files × 2 000 rows (`id` unique, `grp` = `g{id % 8}`). Output: Σ compressed **208 559 B** vs input **500 834 B** → **0.4164**; every output `grp` chunk carries a dictionary page with dict-only data-page encodings (1/1); `id` chunks carry none (0/1). Red on base via the `id` leg (dict page present 1/1) — and on an earlier fix draft via size (1.0837) when the byte leg wrongly disabled `grp`. |
|| C-007 | D-2: an unset `write.parquet.compression-level` writes zstd level 3 (Java/Spark default), explicit levels still win. | `rewrite_size_pin::insert_without_level_property_writes_level_three_bytes` green: byte equality with an explicit level-3 write + unit test on the parser. | PROVEN | No-level INSERT Σ compressed **634 636 B** == explicit level-3 INSERT **634 636 B**; level-1 control **831 806 B** differs. `parquet_compression_default_is_zstd` asserts `compression_level() == 3`. Red on base: no-level bytes equalled level-1 (831 806), not level-3. Footer carries no level field — proven by bytes only. |
|| C-008 | Step-2 gates all green. | The card's gate list. | PROVEN | `rewrite_size_pin` 3 passed; `rewrite_compression` 3 passed; `insert_compression` 4 passed; `insert_distribution` 7 passed; `cargo test -p iceberg --lib` 3670 passed (8 ignored); `cargo test -p iceberg-datafusion --lib` 216 passed (1 ignored); `make check` exit 0 (fmt, clippy `-D warnings`, taplo, machete, artifacts, anchors, comment blocks, file size); comment fence clean. |

## Red evidence (forward pin, base tree)

`cargo test -p iceberg-datafusion --test rewrite_size_pin -- --ignored --nocapture`
exited 101 on the base tree:

```
== pin input: files=206 file_bytes=3505057 rows=412000 sigma_compressed=3263419 sigma_uncompressed=7755694 ratio=0.420777
== pin output: files=20 file_bytes=4810824 rows=412000 sigma_compressed=4786422 sigma_uncompressed=8142272 ratio=0.587848

thread 'rewrite_output_within_ten_percent_of_input' panicked at crates/integrations/datafusion/tests/rewrite_size_pin.rs:48:5:
rewrite output must stay within 10% of input compressed bytes, got 1.4667 (4786422 vs 3263419)
test result: FAILED. 0 passed; 1 failed
```

The pin lives in its own test target so the card's probe gate
(`--test rewrite_size_probe`) stays green while this file stays red until step 2.

## Red evidence (step-2 pins, base tree)

`cargo test -p iceberg-datafusion --test rewrite_size_pin` exited 101 on the base tree
(product code unchanged, pins un-ignored):

```
test insert_without_level_property_writes_level_three_bytes ... FAILED
  assertion `left == right` failed: an INSERT without write.parquet.compression-level
  must write the same compressed bytes as an explicit level-3 write (831806 vs 634636)
  (the no-level bytes equal the level-1 control exactly: 831806 == 831806)
test rewrite_keeps_dictionary_on_low_cardinality_columns ... FAILED
  assertion `left == right` failed: the near-unique id column must not write a
  dictionary page (1/1)
test rewrite_output_within_five_percent_of_input ... FAILED
  rewrite output must stay within 5% of input compressed bytes, got 1.4654
  (4782370 vs 3263419)
test result: FAILED. 0 passed; 3 failed
```

## Mechanism (measured, not guessed)

`dictionary_enabled = true` is a parquet-rs default both paths share — but it only
becomes expensive on the rewrite side:

- **INSERT files** hold 2 000 rows per column chunk. The `ts`/`id` dictionaries
  (~2 000 unique values) never approach `dictionary_page_size_limit` (1 MiB), so the
  dictionary page plus RLE indices is roughly break-even (ts chunk ≈ 10.9 KB
  compressed at level 3).
- **Rewrite files** hold ~22 000 rows per chunk of *globally unique* `ts`/`id`
  values. The dictionary grows past its 1 MiB limit, the column writer falls back to
  PLAIN for the data pages — **and still writes the dead dictionary page**. Footer
  evidence: in a compacted file the `ts` chunk's `data_page_offset` is **152 163**
  (vs **8 089** in an INSERT file) — ~144 KB of compressed dictionary page that no
  data page consumes. Per-column totals, rewrite baseline vs dictionary OFF:

  | column | dict ON compressed | dict ON uncompressed | dict OFF compressed | dict OFF uncompressed |
  |---|---|---|---|---|
  | ts | 3 573 059 | 4 070 532 | 2 815 439 | 3 296 794 |
  | grp | 1 622 | 1 208 | 1 652 | 2 884 742 |
  | id | 1 231 510 | 4 070 532 | 475 564 | 3 296 788 |

  `id` drops 61 %, `ts` drops 21 %. (`grp`'s uncompressed size balloons under PLAIN —
  expected: it stores strings instead of dict indices — but its compressed size is
  unchanged, so the totals fall.)

The secondary contributor is the **zstd level asymmetry**: the input bed was written
at level 3 (Spark/Hadoop's zstd default), the fork writes `ZstdLevel::default()` = 1
when `write.parquet.compression-level` is unset. Level 3 alone takes the rewrite to
1.13× — inside the defect's margin but above the 10 % bound.

**Row order is not the cause.** The scan plans files newest-manifest-first, so a
compacted `grp` file's rows arrive in reverse append-batch order (first 50 ids of the
`g00` output file are 400 000–400 049, vs 0–49 in the first INSERT file). Sorting the
rewrite input back to `id` order recovers only 4 755 410 B (1.4572×).

## WriterProperties table (identical on both paths)

Both `IcebergWriteExec` (`crates/integrations/datafusion/src/physical_plan/write.rs`)
and `write_compacted_files`
(`crates/iceberg/src/maintenance/rewrite_data_files_write.rs`) build
`WriterProperties::builder().set_compression(compression).build()`; everything else
is the parquet-rs 58.4 default:

| property | value (both paths) |
|---|---|
| compression | `ZSTD(ZstdLevel(1))` unless `write.parquet.compression-level` is set |
| dictionary_enabled | true |
| encoding | None (writer default) |
| dictionary_page_size_limit | 1 048 576 |
| data_page_size_limit | 1 048 576 |
| data_page_row_count_limit | 20 000 |
| write_batch_size | 1 024 |
| max_row_group_row_count | 1 048 576 |
| max_row_group_bytes | None |
| statistics_enabled | Page |
| write_page_header_statistics | false |
| statistics_truncate_length | Some(64) |
| column_index_truncate_length | Some(64) |
| writer_version | PARQUET_1_0 |
| created_by | `parquet-rs version 58.4.0` |
| offset_index_disabled | false |
| sorting_columns | None |
| bloom_filter_position / properties | AfterRowGroup / None |
| coerce_types | false |
| data_page_v2_compression_ratio_threshold | 1 |
| key_value_metadata | None |

Construction differences that are NOT `WriterProperties`:

| detail | insert path | rewrite path |
|---|---|---|
| ParquetWriterBuilder | `new_with_match_mode(FieldMatchMode::Name)` | `new(FieldMatchMode::Id)` |
| row routing | TaskWriter fanout, rows clustered by partition | ArrowReader → `RecordBatchPartitionSplitter` → `BoundedPartitionRouter` |
| input batches | DataFusion child exec batches (2 000 rows) | parquet reader batches (8 192-row default) |

The match-mode difference was flipped (`match_mode Name` row): no effect (1.4721 vs
baseline 1.4727).

## Flip table — rewrite σ compressed over the level-3 input (σ in = 3 263 419 B)

Two independent runs agree to ≤ 0.003 out/in; representative numbers from the probe
run:

| flip (one at a time) | σ compressed B | out/in |
|---|---|---|
| baseline (zstd-1, dict on, scan order) | 4 806 191 | 1.4727 |
| **dictionary OFF** | **3 292 655** | **1.0090** |
| zstd level 3 | 3 695 772 | 1.1325 |
| zstd level 9 | 3 594 605 | 1.1015 |
| zstd level 3 + dictionary OFF | 2 185 775 | 0.6698 |
| data_page_size_limit 256 KiB | 4 804 448 | 1.4722 |
| data_page_size_limit 8 MiB | 4 804 418 | 1.4722 |
| data_page_row_count_limit 1 Mi | 4 807 118 | 1.4730 |
| max_row_group_row_count 200 k | 4 803 812 | 1.4720 |
| write_batch_size 8 192 | 4 802 333 | 1.4716 |
| statistics NONE | 4 802 657 | 1.4717 |
| encoding PLAIN (dict still on) | 4 804 448 | 1.4722 |
| offset_index disabled | 4 804 448 | 1.4722 |
| match_mode Name | 4 804 063 | 1.4721 |
| sorted by id (INSERT order) | 4 755 410 | 1.4572 |
| sorted by id + dictionary OFF | 3 245 289 | 0.9944 |

**Named cause for step 2:** `dictionary_enabled` on the rewrite writer — the
dictionary-overflow fallback writes a ~144 KB dead dictionary page per
high-cardinality column chunk. A single `set_dictionary_enabled(false)` lands the
rewrite at 1.009× the input's compressed bytes.

Caveat for step 2's design: dictionary OFF trades away real dictionary wins on
low-cardinality beds (`grp` uncompressed balloons; on repeat-heavy data dict beats
PLAIN). The alternative fix shapes are (a) per-column dictionary control on
high-cardinality columns, or (b) making the fallback not emit the dead dictionary
page — a parquet-rs behavior question the fix step should verify against Java's
parquet-mr semantics before choosing. The zstd default-level gap (1 vs Java's
effective 3 when the property is absent) is a separate, smaller divergence recorded
here for the orchestrator.

## Footer limitation (measured)

parquet footers do not record the zstd level — `ReadThrift` reconstructs
`Compression::ZSTD(ZstdLevel::default())`, so every footer prints `level=1`
regardless of the written level (already documented in
`task/f-write-compress-1-ledger.md`). Written levels are proven by byte totals:
level-3 INSERT input = 3 263 419 B vs 4 279 399 B at level 1 for identical rows.

## Step 2 — the fix (2026-09-12, branch `fix/f-rewrite-size-1`)

Two ruled changes:

**D-2 — zstd default level 3.** `parquet_compression_from_properties` maps an unset
`write.parquet.compression-level` to `ZstdLevel::try_new(3)` instead of
`ZstdLevel::default()` (= 1). One line in
`crates/iceberg/src/writer/file_writer/parquet_compression.rs`; every writer path that
parses table properties inherits it. Proven by bytes (footers do not store the level):
no-level INSERT output equals an explicit level-3 write byte-for-byte (634 636 B) and
differs from level-1 (831 806 B) on identical 80 000-row beds.

**D-1 — per-column dictionary from the input footers.**
`rewrite_data_files_write.rs::dictionary_fallback_columns` loads the parquet footer of
every file in the rewrite group (`ArrowFileReader::get_metadata`;
`task.file_size_in_bytes` avoids a stat, with `input.metadata()` as the fallback), walks
every column chunk, and calls `set_column_dictionary_enabled(path, false)` on the output
`WriterProperties` for exactly the columns whose input chunks show the dictionary not
paying. Three signals, any one disables:

1. a chunk with a dictionary page whose data-page encoding mask contains PLAIN — the
   literal fallback signature (the dead dict page + PLAIN data pages);
2. a chunk with no dictionary page at all — the input writer already declined the dict;
3. `2 × Σ uncompressed_size ≥ Σ num_values × value_bytes` over the column's chunks —
   i.e. uncompressed bytes per value ≥ half the physical value width. A dictionary
   chunk's uncompressed size ≈ U×v + N×w (U uniques × value width + N indices), so this
   fires when U/N ≳ 0.5 — a near-unique column whose output-chunk dictionary would grow
   linearly past `dictionary_page_size_limit` (1 MiB) and die as a dead page. `v` is the
   physical width for fixed-width types; for BYTE_ARRAY it is `4 + (len(min)+len(max))/2`
   from column statistics (default 12 when stats are absent).

Signal 3 is the leg that catches the measured bed: the input files are 2 000-row chunks
whose dictionaries are healthy (no literal fallback), yet `ts`/`id` run at
~9.4 uncompressed B/row vs an 8-byte value — U/N ≈ 1, and the ~22 k-row output chunks
overflow as measured in step 1. An earlier draft used `dict_page_bytes ≥ data_bytes`
instead; it wrongly disabled `grp` on the low-cardinality bed (dict page ≈ 60 B vs
period-8 indices that zstd crushes to ≈ 50 B, U/N = 0.004) and the C-006 size leg went
red at 1.0837 — the per-value-width rule keeps `grp` at 0.43 B/row ≪ 3.5 B threshold.

Output ordering of `paths` is sorted for determinism; `buffered(8)` bounds footer-fetch
concurrency. Non-parquet group members are skipped.

## Gates

Step 1 (measure):

| gate | command | result |
|---|---|---|
| probe | `cargo test -p iceberg-datafusion --test rewrite_size_probe -- --ignored --nocapture` | exit 0 — `1 passed` (231 s) |
| pin (red on base, by design) | `cargo test -p iceberg-datafusion --test rewrite_size_pin -- --ignored --nocapture` | exit 101 — assertion `got 1.4667 (4786422 vs 3263419)` |
| lib | `cargo test -p iceberg-datafusion --lib` | exit 0 — `216 passed; 1 ignored` |
| workspace gates | `make check` | exit 0 — fmt, clippy `-D warnings`, taplo, cargo-machete, agent artifacts, matrix anchors, comment blocks, file-size all green |
| comment fence | `git diff --cached` fence | clean — the only added comment lines are ASF license headers |

Step 2 (fix):

| gate | command | result |
|---|---|---|
| pin | `cargo test -p iceberg-datafusion --test rewrite_size_pin` | exit 0 — `3 passed` (117–133 s); main pin out/in **0.6686**, low-card pin **0.4164** with `grp` dict on, level-3 byte-match exact |
| rewrite compression | `cargo test -p iceberg-datafusion --test rewrite_compression` | exit 0 — `3 passed` |
| insert compression | `cargo test -p iceberg-datafusion --test insert_compression` | exit 0 — `4 passed` |
| insert distribution | `cargo test -p iceberg-datafusion --test insert_distribution` | exit 0 — `7 passed` |
| iceberg lib | `cargo test -p iceberg --lib` | exit 0 — `3670 passed; 8 ignored` |
| datafusion lib | `cargo test -p iceberg-datafusion --lib` | exit 0 — `216 passed; 1 ignored` |
| workspace gates | `make check` | exit 0 — fmt, clippy `-D warnings`, taplo, cargo-machete, agent artifacts, matrix anchors, comment blocks, file-size all green |
| comment fence | `git diff --cached` fence | clean — no added comment lines outside the ASF header |

## Residue / observations

- `write.rs` builds `FieldMatchMode::Name` writers and the rewrite `Id`; measured no
  size effect on this bed.
- The rewrite path has no way to inherit `write.parquet.dictionary.*` table
  properties — the fork's writer surface reads only codec + level today.
- `BoundedPartitionRouter` writer lifecycle is irrelevant here: each partition's rows
  stay in one writer (one file, one row group per output file, 20 files total).
