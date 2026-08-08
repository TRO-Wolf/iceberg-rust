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

# U3 / hazard-1 — MIDPOINT row-group selection over a byte-range split

**Date:** 2026-08-07 · **Branch:** `fix/ranged-read-midpoint-rowgroups` (cut off main `bd9c1cd2`)
**Spec:** [reconciliation-qb-bug001-work-order.md](reconciliation-qb-bug001-work-order.md) §6 ·
**Cadence:** AC·OO Mode A · **Dependency changes:** NONE (and none permitted in this unit)

---

## 1 · The defect

`ArrowReader::filter_row_groups_by_byte_range` (`crates/iceberg/src/arrow/reader.rs`) kept a row
group when its byte range **OVERLAPPED** the scan window, over byte offsets **SYNTHESIZED** as
`4 + Σ compressed_size`. Java/parquet-mr keeps a row group iff its **MIDPOINT** falls in the
half-open `[start, start + length)`, computed from **real** row-group start positions in the footer.

Two independent consequences, both silent (no error, ever):

1. **The rule.** A row group straddling a split boundary is decoded by BOTH adjacent sub-tasks →
   duplicate rows.
2. **The offset source.** Any file whose row groups are not perfectly contiguous (padding, inline
   bloom filters, a non-4 first offset) drifts from the synthetic model → duplicates even for
   splits that were aligned to the file's own row-group offsets.

## 2 · The Java oracle (bytecode-verified, parquet-mr 1.16.0 ≡ 1.13.1)

`ParquetMetadataConverter.filterFileMetaDataByMidpoint`, reached from
`Parquet.ReadBuilder.split(start, length)` → `ParquetReadOptions.Builder.withRange(start, start + length)`
→ `ParquetMetadataConverter.range` → `RangeMetadataFilter` → the `MetadataFilterVisitor`. This is the
call `org.apache.iceberg.data.GenericReader.openFile` makes for every `FileScanTask`.

```text
startIndex = getOffset(rg.getColumns().get(0))       // REAL offset; rg.file_offset UNUSED here
totalSize  = rg.total_compressed_size ?: Σ columns.total_compressed_size
midPoint   = startIndex + totalSize / 2              // TRUNCATING integer division on the SIZE
keep iff   midPoint >= startOffset && midPoint < endOffset      // half-open

getOffset(cc) = min(data_page_offset, dictionary_page_offset)   // dict wins only when SET *and* SMALLER
```

**Regime B** (Java's `RowGroup.file_offset` + `invalidFileOffset` repair chain) is Java's
*error-recovery* path for footers with omitted inline `ColumnMetaData`, and it **is** the
`4 + Σ compressed_size` model. It is **not** ported: porting it as a fallback would reintroduce the
defect on exactly the files that trigger it, and it is unreachable here — parquet-rs
`validate_column_metadata` hard-errors on a missing `data_page_offset` unless the `encryption`
feature is on, and `encryption` is not enabled anywhere in this workspace.

## 3 · The fix

Stateless per row group; the running `4 + Σ compressed_size` accumulator is **deleted** (it is what
made padded files drift). New private helper `ArrowReader::parquet_column_chunk_offset` implements
Java's `getOffset` by hand — deliberately **not** `ColumnChunkMetaData::byte_range()`, which is
`dictionary_page_offset().unwrap_or(data_page_offset())` (no `min`, so it diverges from Java) and
which `assert!`s on negative offsets (a panic on corrupt metadata).

Typed `ErrorKind::DataInvalid`, never a panic, on: a row group with no column chunks (Java throws
`IndexOutOfBounds` there), a negative row-group offset, a negative compressed size, and
`start + length` overflow (the pre-existing guard, kept unchanged).

## 4 · Coverage

New (`crates/iceberg/src/arrow/reader.rs`), all expectations derived from **real footer metadata**:

| Test | Pins |
|---|---|
| `test_midpoint_selection_straddling_splits_read_each_row_exactly_once` | 3 row groups, 800-byte tiling that straddles them: per-window id sets + the union is every row EXACTLY once. Asserts the fixture actually straddles (non-vacuity). |
| `test_midpoint_selection_reads_real_offsets_on_padded_file` | Bloom-padded file, windows tiled at the file's OWN row-group starts. Asserts real starts ≠ `4 + Σ compressed_size` (non-vacuity) — the OFFSET-SOURCE pin. |
| `test_midpoint_selection_partitions_row_groups_over_stride_sweep` | Both fixture shapes × strides {256, 512, 800, 1024, 4096}: selected index sets must PARTITION `0..n`. Plus an adversarial tiling whose boundaries sit EXACTLY on row-group midpoints (only `[start, end)` partitions there). |
| `test_midpoint_selection_offset_and_boundary_semantics` | `getOffset` (a) dict smaller wins, (b) dict NOT smaller ⇒ data page offset (the `min`, not "dict wins" — the arm a naive port gets wrong), (c) no dict; (d) a midpoint exactly on a boundary belongs to the HIGHER window; (e) column-less row group ⇒ typed `DataInvalid`; (f) negative offset ⇒ typed `DataInvalid`, no panic; (g) extreme `i64` offsets still answer. |

Repaired (they built their windows with the same synthetic model the production code used, so they
were structurally incapable of catching offset drift): `test_file_splits_respect_byte_ranges`,
`test_position_delete_with_row_group_selection`, `test_position_delete_in_skipped_row_group`. All
three now derive starts from the footer; each records that its own fixture is contiguous.

Kept unchanged: `test_filter_row_groups_by_byte_range_start_plus_length_overflow`.

## 5 · Mutation proof

| # | Mutation | Result |
|---|---|---|
| M1 | Predicate → OVERLAP (keeping REAL offsets) | RED — straddling, stride-sweep, semantics (proves the RULE, independent of the offset source) |
| M2 | Midpoint rule kept, start restored to `4 + Σ compressed_size` | RED — padded-file pin + semantics (proves the OFFSET SOURCE is load-bearing) |
| M3 | Low bound strict (`midpoint > start`) | RED — semantics + stride sweep |
| M4 | High bound inclusive (`midpoint <= end`) | RED — semantics + stride sweep |
| M5 | `getOffset` → "dictionary wins whenever set" | RED — semantics (b) |
| M6 | `mid = (rg_start + rg_end) / 2` (the mean-of-endpoints form) | RED — 7 tests; the truncating-on-SIZE form is load-bearing, not cosmetic |
| M6c | `row_group_size / 2` → `row_group_size >> 1` (equivalent) | **GREEN** (control — the tests pin a value, not an expression) |
| interop | Predicate → OVERLAP, re-run the D1 interop leg | RED with a real assertion signal (fail-closed step 5 of the suite) |
| rider | `max_row_group_row_count` → one row group in `fk5_pos_oracle_sparse_pos_deletes_multi_rg` | RED (**GREEN on `HEAD` — verified by replaying the mutation against `git show HEAD:` — i.e. the pre-existing test was a false green for its own name**) |

**Correction to the plan:** M2 and M4 were forecast to also turn the stride-sweep property test RED.
They do not by themselves — the partition property is preserved by *any* monotone start model (M2),
and no midpoint in the fixed-stride fixtures lands exactly on a stride boundary (M4). The
adversarial midpoint-aligned tiling was added to the property test for exactly that reason, which
brings M3/M4 back under it; M2 remains caught by the padded-file pin alone, as designed.

## 6 · Interop (built — both directions, fail-closed)

`dev/java-interop/run-interop-ranged-read.sh` + `InteropOracle.RangedReadOracle` +
`crates/iceberg/tests/interop_ranged_read.rs`. Suite floor ratcheted **52 → 53** in the same change;
rows added to `dev/java-interop/map.md` and `crates/iceberg/tests/map.md`. Zero new Maven or Cargo
dependencies.

* **Anti-circular:** both engines tile `[0, fileLength)` at the HAND-DECLARED `STRIDE = 800`
  (`RangedReadOracle.STRIDE` mirrors `interop_ranged_read.rs::STRIDE`); the windows are never taken
  from either engine's splitter.
* **D1:** Java writes a tiny-row-group file the tiling straddles, reads every window through the real
  `Parquet.read(...).split(...)` filter, and asserts its own tiling is a partition of the rows. Rust
  reads the same windows and asserts identical id lists + exactly-once. **PASS.**
* **D2:** Rust writes `rust_contig.parquet` and `rust_padded.parquet` (bloom filters ⇒ parquet-rs
  writes a bloom section after each row group ⇒ real starts run ahead of the synthetic model; the
  drift is asserted so the leg cannot go vacuous), reads every window, and Java replays all 22
  windows. **PASS, 0 failures** — including every padded-file window.
* **Sabotage:** step 5 reverts the production predicate to OVERLAP, requires the D1 leg to go RED
  **with a real assertion signal** (a non-zero exit alone would let a non-compiling mutant score as a
  pass), hard-fails if the pattern is absent or ambiguous, restores + `touch`es + md5-verifies, and
  re-runs GREEN. **Confirmed RED then GREEN.**

## 7 · Corrections to the work order (state these in the PR body)

1. **§6's exposure note is WRONG.** "offsets-aligned splits (fork/Java writers) unaffected;
   offsets-less external manifests (DuckDB class) are the live consumer shape" — measured
   counterexample: with bloom filters enabled (parquet-rs's default position is `AfterRowGroup`) a
   3-row-group fixture has real starts 4/686/1368 while the synthetic model computes 4/542/1080. The
   fork's own writer emits `split_offsets = RowGroupMetaData::file_offset()`, i.e. the REAL starts,
   so offsets-aware windows tile at the real starts and the old rule duplicated rows there too.
   **Offsets-aligned splits over any non-contiguous file were affected. The blast radius was wider
   than scoped.**
2. **Amplifier 4 — MEASURED, and the answer is "no live pin."** The `scan/mod.rs`
   `write_parquet_data_files` fixture writes **one** row group per file (1,024 rows at the default
   row-group size, 3,003-byte file, row group `[4, 467)`); `with_split_size(1024)` tiles
   `[0,1024)/[1024,2048)/[2048,3003)` and the single midpoint 235 lands in window 0 under **both**
   rules. `main` did **not** duplicate rows through that fixture. The real risk was the opposite: the
   `partition_work.rs` "union of `stream_partition_work` bags ≡ `to_arrow`" pin being *cited* as
   coverage. It is not — annotated in-tree by
   `u3_annotation_planning_fixture_is_single_row_group_non_discriminating`, which asserts the
   one-row-group fact from the footer so it stays measured rather than assumed.
3. **Amplifier 2 — `is_strictly_ascending` was NOT touched, deliberately.** Its vacuity on a
   1-element slice matches Java `ArrayUtil.isStrictlyAscending`; the `to_arrow` expansion already
   guards with `offsets.len() > 1`, and a 1-element offsets list through `plan_tasks` yields one
   sub-task `[0, file_len)` whose window contains every midpoint (harmless and Java-identical).
   Changing it would diverge from Java.

## 8 · Named residue (neither justifies work)

1. Java prefers thrift `RowGroup.total_compressed_size` (field 6) over the column sum; parquet-rs
   deliberately does not decode field 6 (`thrift/mod.rs`: "we don't expose total_compressed_size"),
   so the fork always uses `Σ columns.total_compressed_size` — Java's else-branch. The two differ
   only in a malformed file whose declared row-group size contradicts its own column sizes. Closing
   it would require a parquet change, which this unit forbids.
2. Regime B (omitted inline `ColumnMetaData`) is not implemented because parquet-rs refuses to decode
   such a footer without the `encryption` feature. If `encryption` is ever enabled, Regime B becomes
   reachable and this function needs the `file_offset` + `invalidFileOffset` chain added.
3. The `checked_add` guard on the midpoint is **unreachable by construction** (both inputs are `i64`,
   so `offset + size/2 < 2^63 + 2^62 < u64::MAX`). It is kept as a defensive assertion and documented
   as such in the test.

## 9 · Scope fence

Hazard-2 (`_pos` over ranged windows) is **out**. `reader.rs` already rejects ranged `_pos` tasks
with a typed `FeatureUnsupported` error; that guard was not weakened, extended, or "improved" —
midpoint selection does **not** make `_pos` safe, because ordinals still restart at 0 per split.

## 10 · Rider (reported separately from the U3 core)

`fk5_pos_oracle_sparse_pos_deletes_multi_rg` was a proven false green for its own name: it passed
unchanged with `max_row_group_row_count = None` (verified by replaying that mutation against
`git show HEAD:`). Now: (a) it asserts the fixture's row-group count from the real footer, and (b) it
carries a decode-batch leg. The `_pos` path deliberately decodes with **no** row-skipping (no
`RowSelection` / `RowFilter` / row-group pruning), so the only way the multi-row-group shape reaches
the reader is through the decode BATCH boundaries — parquet-rs never spans a batch across a row
group, so batch size 17 over 100-row groups yields `[17;5] + [15]`, twice, and `_pos` must run 0..199
unbroken across that seam. (A row-group-*pruning* leg was drafted and **discarded as misleading**: it
would have claimed pruning that this path does not perform.)
