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
| `test_midpoint_selection_uses_first_column_chunk_on_real_file` | A REAL 2-column file: the row-group start is `columns()[0]`'s offset (Java `getColumns().get(0)`), not any later chunk's. Asserts the last chunk starts well after the first (non-vacuity), then pins a 1-byte window at each true midpoint. |
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
   coverage. It is not — annotated in-tree by `u3_annotation_planning_fixture_is_non_discriminating`,
   which asserts the *load-bearing* fact from the footer (no row group crosses a 1,024-byte split
   boundary) so it stays measured rather than assumed. **Cycle-2 correction:** the first cut asserted
   `num_row_groups == 1`, which is the wrong fact — the fixture's `DataFile` carries no
   `split_offsets`, so `plan_tasks(1024)` really does take the fixed-size branch and really does emit
   three windows; the pin is blind only because that one row group ends at byte 467, well inside the
   first window. Had the fixture's data grown past ~1 KiB, the old assertion would have stayed green
   while its stated reason silently became false.
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
4. **Midpoint selection converts an under-covering window from a harmless over-read into silent row
   loss** (Java-identical, therefore parity-correct, but new behaviour for this fork). A window that
   does not contain a row group's midpoint reads none of its rows: measured, a `[0, 1349)` window
   over a 3-row-group / 300-row file returns 200 rows with no error, where the deleted overlap rule
   returned all 300. Callers whose windows do not TILE `[0, file_size)` — a window narrower than the
   file with no sibling covering the rest, or a `split_offsets[0]` above the first midpoint — now
   lose rows silently. Named in the `filter_row_groups_by_byte_range` doc comment. **Corrected in
   cycle 3 (§12.4):** an understated manifest `file_size_in_bytes` is NOT one of those routes — the
   footer read is anchored at that value, so it fails loudly at metadata decode. And on the
   non-tiling `split_offsets` layout the deleted overlap rule loses the identical rows, so that
   route is not a regression either.
5. **Deliberate fail-closed divergences from Java on corrupt metadata.** A negative offset/size is a
   typed `DataInvalid` here; Java's `getOffset` has no non-negativity guard, so it computes a
   negative midpoint, fails `>= startOffset` and silently DROPS the row group. A column-less row
   group is likewise a typed error where Java throws `IndexOutOfBoundsException`. Rust is stricter in
   both cases and never silently under-reads. Named in the doc comment (cycle-2 addition). Cycle 3
   adds a third: the row-group size is summed with `checked_add` into a typed `DataInvalid` rather
   than through `RowGroupMetaData::compressed_size()`, whose unchecked `i64` `sum()` panics (debug)
   or wraps (release) on a footer declaring several chunks near `i64::MAX`; Java sums into a `long`
   and wraps silently. See §12.3.
6. **AVRO and ORC data files are never split** (cycle 3, §12.1). Java's `FileFormat` calls them
   splittable and parquet-mr's siblings seek to Avro block / ORC stripe boundaries inside the
   window; this fork's Avro and ORC readers materialize whole files, so the planner declines the
   split (`scan::task::reader_honors_byte_range`) rather than manufacture windows they cannot
   honour. The cost is intra-file parallelism for those two formats; the alternative was silent
   N-fold row duplication. Closing it means implementing block/stripe-range reads — a separate
   unit, at which point flip `reader_honors_byte_range`, never `is_splittable`.

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

## 11 · Remediation cycle 2 (2026-08-07) — closing the Critic/Falsifier findings

The independent Critic converged with three S3 findings and the Falsifier demonstrated three
SURVIVING mutations. Every one of them was a **coverage** gap, not a correctness gap — the shipped
rule was confirmed Java-exact by both reviewers, independently re-decoded from the jar. What the
suite failed to pin:

| Surviving mutation | Why it survived | Closed by |
|---|---|---|
| `columns().first()` → `.last()` | every fixture was SINGLE-COLUMN (real and fabricated) | `midpoint_test_metadata` now builds **three** column chunks 1 MiB apart (trailing chunks declare size 0, so `compressed_size()` is unchanged) + new real 2-column fixture `test_midpoint_selection_uses_first_column_chunk_on_real_file` |
| `compressed_size()` → `total_byte_size()` | the builder set `total_byte_size == total_compressed_size` | `total_byte_size` is now `4 × size + 7`, with an in-test `assert_ne!` guard so it cannot silently drift back |
| `row_group_size / 2` → `div_ceil(2)` | every fabricated size was EVEN, so truncation never differed | new case (d2): an ODD size 21 → midpoint 110, pinned from both sides (`[0,111)` selects, `[111,121)` does not) |

Each fixture guard is asserted in-test (`assert_ne!` on compressed-vs-uncompressed, `columns().len()
> 1` with strictly increasing offsets, `last_start > starts[idx]` on the real file), so a future
fixture change that re-degenerates the shape fails loudly instead of quietly restoring the blind
spot.

Also closed this cycle:

- **Stale comment asserting the deleted defect.** `scan/mod.rs`'s `expand_within_file_parallel_tasks`
  still justified its offsets-only guard with "non-aligned windows can re-select the same row group
  and duplicate rows" — true only under the OVERLAP rule this unit deleted, and contradicting the
  `scan/map.md` row added in the same commit. Rewritten to the real reason (byte-arbitrary windows
  can produce EMPTY sub-tasks under midpoint selection — wasted parallelism, not duplication).
- **The amplifier-4 annotation asserted the wrong fact** — see §7.2.
- **Two residue lines added** (§8.4 silent row loss on an under-covering window, §8.5 the
  fail-closed divergences), both also named in the production doc comment.

Declined, with reason: the Critic's S3 on the rider's hardcoded decode-batch vector
(`[17;5]+[15]` twice) — the Critic itself marked it "no action required to merge", the maintenance
cost is documented in the test's own doc comment, and weakening it to "some batch boundary falls at
row 100" would drop the `_pos`-continuity evidence that makes the rider non-vacuous. If a parquet-rs
bump ever fires it, weaken it then.

**Process note (from the Falsifier, worth keeping).** Two false-green mechanisms were hit during
review, neither in the product: (1) a concurrent sibling agent mutating the same shared worktree made
one baseline run RED — mutation work must be done in an isolated copy or under an exclusive claim;
(2) `git archive HEAD | tar -x` preserves COMMIT mtimes, so cargo skips the rebuild and runs a STALE
test binary (it reported the pre-fix count 3138). Any restore-by-content step must bump mtimes —
this cycle's harness `touch`es every restored file and md5-verifies against the pre-mutation baseline.

## 12 · Remediation cycle 3 (2026-08-07) — closing the Falsifier counterexamples

The independent Critic converged with **zero** findings and independently re-decoded parquet-mr,
confirming the shipped rule Java-exact on all four axes. The Falsifier reported `broke_it = true`
with one HIGH counterexample it actually RAN, two MEDIUM items, and one doc correction.

### 12.1 · HIGH — AVRO/ORC ranged splits duplicate every row (same hazard class, live, pre-existing)

`is_splittable` (`scan/task.rs`) calls AVRO/ORC splittable — a faithful port of the Java
`FileFormat` table — and `TableScan::plan_tasks` (`scan/mod.rs`) calls `task.split(split_size)`
unconditionally. But **only the Parquet reader reads the window**: a grep of every `task.start` /
`task.length` read in `arrow/reader.rs` lands exclusively inside `process_parquet_file_scan_task`
(the `_pos` guard and the byte-range filter call). `process_avro_file_scan_task` and
`process_orc_file_scan_task` materialize whole files and drop straight into
`finish_whole_file_scan_task`. So every sub-task re-emitted the entire file: the Falsifier measured
a 500-row Avro OCF split four ways returning **2,000 rows**, silently, with no error — precisely
the symptom this unit exists to eliminate, on a different format.

Verified independently here by the grep above and by the new end-to-end pin, which is RED without
the fix.

**Fix, two layers:**

1. **Planner (the real fix).** New `scan::task::reader_honors_byte_range(format)` — a property of
   the READ path, deliberately separate from `is_splittable`, which keeps porting Java faithfully.
   `FileScanTask::split` takes the passthrough branch when either predicate says no, so AVRO/ORC
   are never split. This is Java-divergent in PARALLELISM; splitting was Java-divergent in ROWS.
2. **Reader (defence in depth).** `ArrowReader::reject_ranged_whole_file_task` fails a ranged
   AVRO/ORC task closed with a typed `FeatureUnsupported`, covering the public `PartitionWork` /
   direct-reader seams exactly as the `_pos` guard does. Both whole-file spellings (`length == 0`
   legacy sentinel, `length == file_size_in_bytes`) still pass.

**Not fixed, named instead:** implementing real Avro block-range and ORC stripe-range reads is a
separate unit. Until then AVRO/ORC scans are single-task per file.

**Interop safety check.** The `run-interop-scan-plan` oracle's fixtures are all parquet
(`merge.parquet` / `gap.parquet` / `big.parquet`), so the Rust-vs-Java `planTasks` comparison is
untouched by the AVRO/ORC passthrough. If an AVRO or ORC data file is ever added to that oracle it
WILL diverge, by design — Java splits, this fork does not.

### 12.2 · MEDIUM — the negative-`compressed_size` guard was unpinned (MZ1 survived)

Replacing the guard with `.unwrap_or(0)` left the full lib suite green. Under that mutation a
corrupt size makes `midpoint == row_group_start`, i.e. **selection by START instead of midpoint** —
wrong rows, no error. Worse, the in-tree comment justified the gap with a claim that is simply
false: the public `ColumnChunkMetaData` builder *does* accept `set_total_compressed_size(-20)`.
Closed by case **(h)** of the semantics matrix (with a fixture guard asserting the fabricated size
really is negative, and a second window at the row-group start proving the error is the only thing
preventing selection-by-start); the false claim is deleted.

### 12.3 · MEDIUM — `RowGroupMetaData::compressed_size()` panics before any guard runs

That accessor is an **unchecked** `i64` `sum()` over the column chunks, and parquet-rs applies no
range validation when decoding the thrift field, so a corrupt footer declaring several chunks near
`i64::MAX` aborts with `attempt to add with overflow` (debug) or wraps to a bogus/negative size
(release) — reachable from a hostile file, and directly contradicting the doc comment's fail-closed
claim. The size is now summed **here** with `checked_add` into a typed `DataInvalid`. Semantics are
unchanged (still Java's else-branch, `Σ columns.total_compressed_size`). Pinned by case **(i)**.

### 12.4 · LOW — the residue sentence over-claimed one route

"a manifest whose `file_size_in_bytes` understates the file … loses rows silently" is wrong:
`ArrowFileReader` anchors the footer read at that value, so an understated size fails **loudly** at
metadata decode (`Invalid Parquet file. Corrupt footer`) long before selection. The silent-row-loss
residue is real only for an under-covering WINDOW and for a `split_offsets` list that does not tile
— and the Falsifier hand-computed the deleted overlap rule on the latter layout and got the same
rows, so it is not a regression either. The doc comment now says exactly that.

### 12.5 · Declined, with reason

- **MZ3** (midpoint `checked_add` → `wrapping_add`) survives. This is not a coverage gap: both
  inputs are `i64`-derived, so `offset + size/2 < 2^63 + 2^62 < u64::MAX` and the branch is
  unreachable by construction. The Falsifier agrees ("confirming the Actor's call rather than a
  coverage gap"). It stays as a defensive assertion — §8.6.
- **MZ5** (`data_page_offset > dict` → `>=`) survives and is genuinely equivalent (both arms return
  the same value when the two offsets are equal). The Falsifier ran it as its own harness control.
- **`is_strictly_ascending` vacuity on a 1-element slice** stays untouched: it matches Java
  `ArrayUtil.isStrictlyAscending`, and the resulting single sub-task `[0, file_len)` contains every
  midpoint. Out of scope per the work order's risk list.
- **Zero-size row group + `end - 1` underflow** in the `partition_work.rs` annotation (Falsifier's
  "minor nit"): unreachable for that fixture (a parquet row group always has a positive compressed
  size), and the assertion is a test annotation, not a production path.

### 12.6 · Process

Per the Falsifier's urgent process finding — the shared worktree was carrying a sibling agent's
uncommitted `midpoint <= end` mutation during cycle-2 review — the entire cycle-3 gate and mutation
sweep were run in an **isolated** `git archive | tar -x` tree with its own `CARGO_TARGET_DIR`, with
`touch` after extraction (commit mtimes otherwise make cargo run a stale binary) and md5 restore
verification after every mutant.

---

## 13 · Cycle 4 — the independent Critic's two doc items and the Falsifier's four survivors

Cycle 3's reviewers split cleanly: the Critic declined to converge on a matrix-accuracy item, and
the Falsifier demonstrated four mutation survivors by execution — three of them in the *guards*
this unit shipped, one a live silent-data-loss bug that cycle 3 had turned into an asymmetry.

### 13.1 · HIGH — `split` returned ZERO sub-tasks for a whole-file `length == 0` task (F-1)

`FileScanTask::split_fixed_size` starts `remaining = self.length` and loops `while remaining > 0`,
so a PARQUET task carrying the legacy whole-file sentinel (`start == 0, length == 0`) produced an
EMPTY sub-task vector. `scan/mod.rs`'s `split_tasks.extend(task.split(split_size)?)` then dropped
the file outright: `plan_files` returned it, `plan_tasks` read **zero rows** from it, and nothing
errored anywhere. The Falsifier demonstrated the pair — 60/60 rows whole-file, 0 sub-tasks and 0
rows after `task.split(1024)`.

It is externally reachable: `FileScanTask` is `pub` with `pub` fields and a derived `Deserialize`
(reproduced through a serde_json round-trip), so any caller or persisted plan can carry the
spelling.

Cycle 3 turned it into an **asymmetry**, which is what makes it a defect rather than a policy: the
new AVRO/ORC passthrough returns `[self]` for exactly the same input, and BOTH whole-file reader
guards — `reject_ranged_whole_file_task` (cycle 3) and the `_pos` guard — explicitly accept
`start == 0, length == 0` as a supported whole-file spelling. Three of the four sites agreed;
`split` was the outlier, and it was the one that lost rows.

**Fix:** `split` returns `[self]` for `length == 0`, ahead of both the offsets-aware and fixed-size
branches (the offsets-aware branch is no better — it would derive its last window from a zero file
length). Rejecting with a typed error was the alternative and was NOT chosen: three other sites
already treat the spelling as valid, so an error would break callers that are correct today, and
`[self]` loses nothing. Java never reaches this case at all — `BaseFileScanTask.length()` is always
`file.fileSizeInBytes()` — so this is a fork-local branch, NAMED in GAP_MATRIX row R148.

Pinned twice: `scan::task::tests::split_whole_file_length_sentinel_is_one_task_not_zero` (both
branch orders, with a non-vacuity control that the same geometry DOES split when `length > 0`) and
`arrow::reader::tests::test_whole_file_length_sentinel_survives_split_and_reads_every_row`, which
drives the real `split` → `ArrowReader::read` path and asserts the union is all 60 rows.

### 13.2 · MEDIUM — the byte-range ENTRY gate of this whole unit was unpinned (F-2)

Mutating `if task.start != 0 || task.length != 0` to `if task.length != 0` left the entire lib
suite green. It is not a no-op: it flips a `start > 0, length == 0` task from "select nothing"
(Java `withRange(start, start)`, whose `contains` is never true) to "read the ENTIRE file" —
the inverse of what this unit exists to guarantee. No test in the suite used that shape.
Closed by `test_byte_range_gate_fires_on_a_zero_length_window_at_a_nonzero_start`, which asserts
zero rows at three non-zero starts and keeps the `(0, 0)` sentinel full-file read as the
non-vacuity control.

### 13.3 · MEDIUM — both whole-file guards varied only the LENGTH axis (F-3)

Dropping `task.start == 0 &&` from `reject_ranged_whole_file_task` — code shipped in cycle 3 —
survived green, because the new test varied only `length`. A task with
`start > 0 && length == file_size_in_bytes` is a genuine ranged window that the mutant ACCEPTS, and
the Avro/ORC reader would then re-emit the whole file: precisely the silent-duplication class the
guard was written to stop. The Falsifier's tell was the asymmetry — the LENGTH-axis mutation was
RED with 12 killers, the START-axis one had none.

The `_pos` guard at the top of `process_parquet_file_scan_task` is a copy of the same predicate and
had the same gap; `fk5_pos_ranged_split_task_is_rejected_fail_loud` also varied only `length`. Both
tests now sweep both axes (`(1, file_size)` and, for `_pos`, `(1, 0)`).

### 13.4 · LOW — nothing distinguished the two parquet-mr offset helpers (F-4)

Adding `dictionary_page_offset > 0 &&` to `parquet_column_chunk_offset` survived green. parquet-mr
has two helpers differing at exactly that predicate, and they belong to different call sites:

* `ParquetMetadataConverter.getOffset(ColumnChunk)` — `isSet`, **no** `> 0` — drives
  `filterFileMetaDataByMidpoint`, i.e. the READ path this function serves.
* `ColumnChunkMetaData.getStartingPos()` — `dictionaryPageOffset > 0 &&` — is what Iceberg's
  split-offset WRITER uses.

The fork already had the right one; nothing proved it. Case **(d5)** of the semantics matrix closes
it: dict `Some(0)`, data 1000, size 100 ⇒ start 0, midpoint 50, selected by `[0, 51)` and NOT by
`[1000, 1100)`.

### 13.5 · Doc items from the independent Critic

* **C-S2 — GAP_MATRIX row R148** was factually wrong after cycle 3: the `FileScanTask::split` cell
  still read "(offsets-aware, Puffin non-splittable)" while split had begun declining AVRO and ORC,
  and the string AVRO appeared nowhere in the matrix. The parenthetical is corrected and a NAMED
  divergence clause added (why the decline exists, that `is_splittable` still ports Java's
  `FileFormat` table faithfully and the gate is the separate `reader_honors_byte_range` predicate,
  that the cost is intra-file parallelism only, and that closing it means flipping
  `reader_honors_byte_range` and never `is_splittable`), plus the F-1 sentinel clause. Anchors
  green at 75 rows.
* **C-S3 — lessons** now carries the rule that a gate command must never be piped into
  `tail`/`grep`/`head` inside a verification `&&` chain: the pipeline's exit status is the last
  command's, so the gate step cannot fail the chain. The cycle-3 gate did exactly that.

### 13.6 · Declined, with reason

The Falsifier's own equivalent-survivor classifications are accepted and NOT "fixed": **MF4**
(`split_at_offsets`' last window end `self.length` → `file_size_in_bytes` — equal for a whole-file
parent, which is the only shape that reaches the offsets-aware branch), **MF7** (dropping
`!is_splittable(...) ||` from `split`, redundant because `reader_honors_byte_range` is already false
for Puffin), and **MF8** (`midpoint < end` → `end.max(start)`, equal because `end >= start` is
already enforced by the `checked_add` guard).

### 13.7 · Process

This cycle was run STRICTLY SERIAL in the shared worktree, as the only agent touching it — cycle 3
had a Critic and a Falsifier mutating it concurrently and each `git checkout --`-ing the other's
in-flight mutant, which can turn a RED mutant into a false GREEN. Every mutation below was applied
alone, run, restored with `git checkout --`, and `git status --porcelain` verified EMPTY before the
next one.

## 14 · Cycle 5 — the Critic's ORC S2 and the Falsifier's F5–F9

Six items in, six closed: **two production fixes** (F5, F7), **three test pins** (the Critic's S2 /
F9, plus two notes from the Falsifier's list), **one interop leg** (F6), and **two DECLINED with an
executed equivalence proof** (F8 and the `can_expand` half of F5). Lib suite 3149 → **3155**.

### 14.1 · F5 (S2) — `split` RELOCATED an already-ranged task's window

`split`'s two real branches both treat the byte space as ABSOLUTE FROM ZERO: `split_fixed_size`
started its walk at a literal `0` and `split_at_offsets` takes the manifest offsets verbatim, ending
the last window at `self.length`. So a parent covering bytes 139..1185 of a 3-row-group / 60-row file
(ids 20..59, which the production reader serves correctly) came back as `[(0, 1046)]` — or
`[(0, 524), (524, 522)]` at a smaller target — and reading those products returned ids **0..59**:
20 rows the parent never owned, with its own tail dropped. Silent corruption, one level worse than
the row loss F-1 closed in cycle 4, and reachable through the same public route (`FileScanTask` is
`pub` with `pub` fields and a derived `Deserialize`; `split` is `pub`).

**Fix chosen: return `[self]` for `self.start != 0`**, ahead of both branches and of the cycle-4
`length == 0` guard. The alternative — teaching both branches to anchor at `self.start` and clip to
`[self.start, self.start + self.length)` — was rejected because Java forecloses the shape
STRUCTURALLY rather than handling it: `BaseFileScanTask` implements `SplittableScanTask`, but the
`SplitScanTask` its splitter emits does not, so a Java split product is never re-splittable. This
crate uses one type for both, so it must say something; `[self]` is the lossless thing to say, it is
the same answer branches (1) and (1b) already give, and it costs nothing in practice (no planner path
produces a ranged parent). `split_fixed_size` is nevertheless re-anchored at `self.start` so the
window stays absolute-correct if the guard ever moves — a documented no-op today (mutant **E3**).

Pinned twice: `scan::task::tests::split_of_an_already_ranged_task_is_a_passthrough_not_a_relocation`
(both branch orders, each with a non-vacuity control proving the same geometry DOES split from
start 0) and `arrow::reader::tests::test_split_of_a_ranged_task_reads_the_parents_rows_not_the_whole_file`
(drives the real `split` → `ArrowReader::read` path and asserts the union is exactly the parent's
ids 20..59, never 0..59).

### 14.2 · F7 (S3) — `plan_tasks` manufactured the shape the `_pos` reader refuses

`expand_within_file_parallel_tasks` (the `to_arrow` seam) has always suppressed splitting under a
`_pos` projection, because the `_pos` path decodes whole files with ordinals from 0. `plan_tasks`
split unconditionally, and the reader's `_pos` guard rejects every `start != 0` task — so a `_pos`
scan that `to_arrow()` served correctly was a TOTAL OUTAGE on the `plan_tasks` / `PartitionWork`
seam (measured: 0 rows and two `FeatureUnsupported` errors), and the error told the caller to "plan
without splitting" when the caller never chose to split. Fixed by hoisting the same suppression into
`plan_tasks`: one layer must never manufacture what the next layer rejects. Fail-loud, not corruption
— hence S3 — but a total outage of the FK5 row-identity surface on the documented multi-partition
seam. Pin: `scan::tests::test_plan_tasks_does_not_split_when_pos_is_projected`, with a non-vacuity
assertion that the same target DOES produce ranged sub-tasks without `_pos`.

### 14.3 · Critic S2 / Falsifier F9 — the ORC call site of the duplication guard

`reject_ranged_whole_file_task` is invoked from both `process_avro_file_scan_task` and
`process_orc_file_scan_task`, but only the AVRO call site was pinned: deleting the ORC line left the
whole suite green, so a future edit could re-open the N-copies-per-N-way-split class and ship. No
production change was needed — the guard is correct — the exposure was pure coverage, and it left
R148's and `scan/map.md`'s "fails a ranged AVRO/ORC task closed" claim resting on nothing for half
the formats it names. `orc_ranged_task_is_rejected_with_a_typed_error` mirrors the AVRO test
one-for-one: both axes of the predicate (`(0, len/2)` and `(1, len)`), `panic!`-with-row-count on an
unexpected success so a duplicating mutant reports the duplication it caused, and both whole-file
spellings kept as non-vacuity controls. `scan/map.md`'s Pins list now names it.

### 14.4 · F6 (S2) — the interop driver's sabotage proved only the PREDICATE, not the OFFSET SOURCE

Step [5/5] mutated one literal — the `midpoint >= start && midpoint < end` predicate — and re-ran
only the D1 leg. Both Rust-side legs are BLIND to the other half of the claim: reverting
`row_group_start` to the synthetic `4 + Σ compressed_size` model (the model the reader doc,
`scan/map.md` and R148 all say is gone from the selection path) leaves D1 green AND the D2 GEN leg
green, because the synthetic model still yields a partition of the rows, so `assert_exactly_once` and
the bloom-drift guard both pass. Only the JAVA comparison catches it. Under this repo's promoted rule
— a sabotage step that did not actually corrupt anything has proven nothing — the offset-source half
was unproven. The driver now has a **step [6/6]**: mutate the offset SOURCE to the synthetic model,
re-run the Rust GEN through the mutant, and require `verify-interop-ranged-read` to fail with a real
PER-WINDOW comparison signal (`FAIL ranged-read-d2 <file>.parquet [...`). The "missing json" /
"empty json" FAIL forms are explicitly NOT accepted, because a mutant that failed to compile would
produce exactly those. Same HARD-FAIL-if-the-literal-is-absent, `|| rc=$?`, md5-verified-restore
mechanics as step [5/6], then GEN + VERIFY re-run GREEN.

Secondary point in the same finding, also fixed: `assert_exactly_once("java fixture", …)` derived its
expected row count from the OBSERVED total, so a read that lost a suffix of ids satisfied it. It now
takes a declared `JAVA_ROWS` constant mirroring `InteropOracle.RangedReadOracle.ROWS`, with the
observed total asserted equal to it first.

### 14.5 · Notes promoted from the Falsifier's non-finding list

* `is_splittable` is behaviourally INERT as a single point (each format is masked by the other
  predicate), so R148's "ports Java's `FileFormat` table" claim had no executable evidence.
  `format_predicate_tables_match_java_and_the_read_path` asserts both four-arm tables directly;
  mutant **M5** (`Puffin → true`) is RED against it and was GREEN before.
* `split_at_offsets`' negative-offset typed error was unpinned (clamping to 0 survived) and is
  reachable from a corrupt manifest whose offsets are strictly ascending but negative.
  `split_negative_offsets_are_a_typed_error_not_a_clamp` pins it; mutant **M4** is RED.

### 14.6 · Declined, with an EXECUTED equivalence proof

* **F8 — the sentinel guard is "broader than its documentation".** Not fixed by narrowing the
  condition; fixed by ORDERING. The new `start != 0` passthrough (14.1) sits ABOVE the `length == 0`
  guard, so `start == 0` holds whenever the sentinel branch is reached and the shipped condition IS
  the documented `start == 0, length == 0` sentinel. Mutant **E1** (narrowing it to spell that out)
  is GREEN — an equivalent no-op, as claimed, not an unpinned choice. The three-layer disagreement
  F8 described is gone with it: `(start > 0, length == 0)` now takes the ranged passthrough.
* **F5's second half — "pin `can_expand`'s `task.start == 0` clause".** Impossible after the fix, and
  the impossibility is the proof: with `split` returning `[self]` for a ranged parent, dropping the
  clause routes the task into `task.split(target)` which returns `vec![task]` — byte-identical to the
  `else` branch. Mutant **E2** is GREEN by construction. The clause is retained as a local statement
  of the precondition and is now commented as defensive rather than load-bearing.
* MF4 / MF7 / MF8 remain accepted equivalent survivors (see §13.6). The Falsifier's correction to the
  MF4 rationale is recorded: a caller CAN construct `length != file_size_in_bytes` through the same
  public route as F5, so MF4 is an unreachable-IN-PRACTICE survivor, not a true equivalent.

### 14.7 · Process — and a lesson bought at the cost of a rewrite

Strictly serial, single agent in the worktree. **Restores were done from `cp` backups and verified
with `cmp`, never with `git checkout --`** — the first M1 attempt this cycle used
`git checkout -- crates/iceberg/src/scan/task.rs` and, because the F5 fix and its tests were still
uncommitted in that same file, the "restore" reverted the file to HEAD and DELETED the fix. The sweep
looked healthy (porcelain clean for that path is exactly what the exclusive-access protocol asks
for); the loss only surfaced when the NEXT mutant produced an extra, unrelated failure. All of
`scan/task.rs` was re-applied, the suite re-greened at 3155, and the whole sweep was re-run from
scratch with the `cp`/`cmp` harness. Filed in `task/lessons.md` (2026-08-08).

### 14.8 · Mutation table (all re-run after the rewrite)

| # | Mutation | Expected | Result |
|---|---|---|---|
| M1 | remove `if self.start != 0 { return Ok(vec![self.clone()]); }` from `split` | RED | RED — 2 killers (`split_of_an_already_ranged_task_is_a_passthrough_not_a_relocation`, `test_split_of_a_ranged_task_reads_the_parents_rows_not_the_whole_file`) |
| M2 | delete `Self::reject_ranged_whole_file_task(&task, "ORC")?;` | RED | RED — `orc_ranged_task_is_rejected_with_a_typed_error` (was GREEN before this cycle) |
| M3 | make `plan_tasks` split unconditionally again | RED | RED — `test_plan_tasks_does_not_split_when_pos_is_projected` |
| M4 | clamp a negative split offset to 0 instead of erroring | RED | RED — `split_negative_offsets_are_a_typed_error_not_a_clamp` |
| M5 | `is_splittable(Puffin) → true` | RED | RED — `format_predicate_tables_match_java_and_the_read_path` |
| E1 | narrow the sentinel guard to `start == 0 && length == 0` | GREEN (equivalent) | GREEN — 3155/0 |
| E2 | drop `task.start == 0 &&` from `can_expand` | GREEN (equivalent) | GREEN — 3155/0 |
| E3 | re-anchor `split_fixed_size` at `0u64` instead of `self.start` | GREEN (equivalent) | GREEN — 3155/0 |

Every mutant was applied alone, run against the full `cargo test -p iceberg --lib`, restored from its
`cp` backup, and the restore verified byte-for-byte with `cmp` before the next one; `git status
--porcelain` showed only this cycle's own intended files throughout, and the tree is clean at commit.
