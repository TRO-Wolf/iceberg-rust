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

# Ledger — F-RDF-COW-BYTES-1: a data rewrite keeps a position delete that still applies, as Java does

**Ledger id:** `F-RDF-COW-BYTES-1-2026-09-22`
**Branch:** `fix/f-rdf-cow-bytes-1` (cut off fork `main`)
**Scope:** one defect, one commit per brief step; one ledger for the lane
**Model:** swe-2-high

| Item | Commits | Subject |
|---|---|---|
| 0 | `66299118` | `docs: F-RDF-COW-BYTES-1 — step 0 measurement` |
| 1 | `194503d4` | `test: F-RDF-COW-BYTES-1 — red delete-survival cells` |
| 2 | `5e7455f2` | `fix: F-RDF-COW-BYTES-1 — the rewrite keeps a position delete that still applies, as Java does` |
| 3 | — | (mutation proof — no commit) |
| 4 | this commit | `docs: F-RDF-COW-BYTES-1 — ledger, mutation proof` |

## Defect

RePark registry rows ICE-RDF-COW-BYTES-1 ("`rewritten_bytes_count` misses a DELETE-written file")
and ICE-RDF-DANGLE-2 ("`removed_delete_files_count` 1 vs Spark 0") are one defect. RePark's
merge-on-read DELETE writes a parquet position-delete file with FULL `file_path` bounds
(`MetricsConfig::for_position_delete`), so a delete covering one data file has equal
`file_path` lower/upper bounds and is FILE-SCOPED under `referenced_data_file_location`
(`crates/iceberg/src/delete_file_index.rs`, leg 3 — Java
`ContentFileUtil.referencedDataFile`). The fork's `plan_dv_removal`
(`crates/iceberg/src/maintenance/rewrite_data_files_dv.rs`) then drops that delete in the
`RewriteFiles` commit when the referenced data file is rewritten — because it treats every
file-scoped position delete as a deletion vector. Java's DV-removal path
(`RewriteDataFilesSparkAction.danglingDVs` via `ContentFileUtil.isDV`) is Puffin-only, so
Spark keeps the parquet delete.

## Step 0 — measurement

### Fork shape used

`(x BIGINT, y BIGINT, z BIGINT)` partitioned by `x`, format v2 — the fork's own
`rewrite_data_files_dangling_tests.rs` shape (`create_partitioned_table`,
`write_data_file`, `append_files`, `add_deletes` from
`crates/iceberg/src/maintenance/rewrite_data_files.rs::tests`). Two partitions × 4 data
files of 50 rows each; one append commit (data seq 1), one `row_delta` commit carrying ONE
parquet position-delete file in partition `x = 0` (data seq 2). Two delete-file shapes:

- **file-scoped** (`MetricsConfig::for_position_delete` — full `file_path` bounds, equal
  lower/upper for a single covered file): what RePark's DELETE actually writes.
- **partition-scoped** (`write.metadata.metrics.column.file_path = none` — no `file_path`
  bounds, no `referenced_data_file` field): the shape the brief describes; attaches to all
  four partition-0 tasks.

### Pre-fix results (`cargo test -p iceberg --lib cow_bytes`, RUST_TEST_THREADS=6)

| cell | delete shape | options | fork result | delete after | removal path |
|---|---|---|---|---|---|
| `partition_delete_threshold` | partition-scoped | `delete-file-threshold=1` | rewritten 4, added 1, removed 0 | kept | — (already correct) |
| `file_scoped_delete_threshold` | file-scoped | `delete-file-threshold=1` | rewritten 1, added 1, removed **1** | dropped | `plan_dv_removal` in the rewrite commit |
| `rewrite_all` | file-scoped | `rewrite-all=true` | rewritten 8, added 2, removed **1** | dropped | `plan_dv_removal` |
| `remove_dangling` | file-scoped | `rewrite-all=true`, `remove-dangling-deletes=true` | rewritten 8, added 2, removed **1** | dropped | `plan_dv_removal` |
| `remove_dangling_single_row` | file-scoped | `rewrite-all=true`, `remove-dangling-deletes=true` | rewritten 8, added 2, removed **1** | dropped | `plan_dv_removal` |
| `partition_delete_survives_dangling_cleanup` | partition-scoped | `rewrite-all=true`, `remove-dangling-deletes=true` | rewritten 8, added 2, removed 0 | kept | — (already correct) |
| `new_sequence_keeps` | file-scoped | `rewrite-all=true`, `use-starting-sequence-number=false` | rewritten 8, added 2, removed **1** | dropped | `plan_dv_removal` |
| `new_sequence_dangling_cleanup` | file-scoped | `rewrite-all=true`, `use-starting-sequence-number=false`, `remove-dangling-deletes=true` | rewritten 8, added 2, removed 1 | dropped | seq rule (`2 < 3`) — correct, stays |

Under `use-starting-sequence-number=true` (default) the output data files carry data seq 2
(the starting snapshot's seq = the row-delta's seq) and file seq 3 — measured through
`live_data_sequences`. With `use-starting-sequence-number=false` they carry data seq 3 =
file seq 3. The fork honours the Java starting-sequence-number rule already
(`RewriteDataFiles::execute` → `RewriteFilesAction::data_sequence_number(starting)`);
step 0 shows NO divergence there — the seq rule itself is not the defect.

A partition-scoped delete is already kept everywhere: `plan_dv_removal` never sees it (no
derivable referenced path) and the seq rule keeps it (`2 < 2` is false). The defect is
exactly the file-scoped-parquet path.

### The attach asymmetry (why `delete_file_threshold` rewrites 1, not 4)

`PopulatedDeleteFileIndex::new` puts a file-scoped delete in `pos_deletes_by_path` only —
the maps are EXCLUSIVE — so it attaches to just its referenced file. Under
`delete-file-threshold=1` only that file is a candidate → rewritten 1. A partition-scoped
delete lands in `pos_deletes_by_partition`, attaches to all four partition-0 files →
rewritten 4 (Spark's exact cell). Java `DeleteFileIndex.forDataFile` routes the same way,
so rewritten=1 is the Java-correct answer for the file-scoped shape — Spark's 4 reflects
Spark's own partition-scoped delete files, not a planner divergence.

### Spark oracle (the run-23a COW-bytes oracle, `cow_bytes_truth.json`)

| cell | DELETE | options | Spark result | delete after |
|---|---|---|---|---|
| `delete_file_threshold` | `id = 3` | `delete-file-threshold=1` | rewritten 4, added 1, removed **0** | kept (seq 9) |
| `remove_dangling` | `id < 30` | `rewrite-all=true`, `remove-dangling-deletes=true` | rewritten 8, added 2, removed **0** | kept |
| `rewrite_all` | `id < 30` | `rewrite-all=true` | rewritten 8, added 2, removed **0** | kept |
| `use_start_seq_false` | — | `rewrite-all=true`, `use-starting-sequence-number=false` | rewritten 8, added 2, removed 0 | (no delete in cell) |

Spark's delete files carry NO derivable `referenced_data_file` (no equal `file_path`
bounds) → partition-scoped → attach to all 4 → rewritten 4. Spark's output data files
carry data seq 9 (the starting seq = the DELETE snapshot's seq); the delete's seq 9 is not
below the partition minimum data seq 9 → kept.

### Decisive Spark probe — the MERGE-based file-scoped-parquet cell

The earlier whole-file DELETE probe was invalid (Spark removed the data file entirely).
The corrected probe ran Spark 4.1.2 + Iceberg 1.11.0 under
`/tmp/oc-worker/_lib/jvm-lock.sh /tmp/sparkenv/bin/python` (driver 2g, UI off):

```python
# shape: unpartitioned (id BIGINT, v STRING), v2, write.delete.mode=merge-on-read,
# 4 single-INSERT files of 50 rows, then a MERGE that matches one row per file —
# Spark writes one file-scoped parquet delete per data file (equal file_path bounds),
# keeping the data files live.
spark.sql("""MERGE INTO ns.t target USING (SELECT id FROM ns.t WHERE id % 50 = 0) source
             ON target.id = source.id WHEN MATCHED THEN DELETE""")
spark.sql("""CALL sc.system.rewrite_data_files(
             table => 'ns.t', options => map('rewrite-all','true'))""").show()
```

Measured:

```text
{"rewritten_data_files_count": 4, "added_data_files_count": 0,
 "rewritten_bytes_count": 6229, "failed_data_files_count": 0,
 "removed_delete_files_count": 0}
```

After the rewrite the four parquet delete files REMAIN live even though every referenced
data file was rewritten away and no live rewritten data files carry matching paths.
`removed_delete_files_count = 0`. This REFUTES the f16 inference that Spark reclaims
file-scoped parquet deletes through a DV-shaped path — Java's reclaim is Puffin-only.

### Java rules confirmed from source (iceberg-core 1.11.0)

- `ContentFileUtil.isDV` — `deleteFile.format() == FileFormat.PUFFIN`. Only Puffin files
  are deletion vectors; parquet position deletes never are.
- `ContentFileUtil.referencedDataFile` — legs: `referenced_data_file` field, else equal
  `file_path` lower/upper bounds. Format-agnostic; decides INDEX ROUTING only.
- `RewriteDataFilesSparkAction.danglingDVs` / `isDanglingDV` — drops only `isDV` deletes
  whose referenced data file was rewritten. Puffin-only.
- `BaseRewriteFiles` / `MergingSnapshotProducer.apply` —
  `dropDeleteFilesOlderThan(minDataSequenceNumber)`: sequence-based GC of delete entries
  in rewritten manifests, not a format check.
- `RemoveDanglingDeletesSparkAction` — position deletes dangle when
  `data_sequence_number <` the partition's minimum live data seq; equality deletes when
  `<=`; a partition with no live data file drops all its deletes; a file-scoped
  (DV-shaped) reference that no live data file matches is dropped — but that reference
  arm applies to deletion vectors.
- `RewriteDataFilesCommitManager` / `use-starting-sequence-number` (default true):
  rewrite outputs are stamped with the starting snapshot's sequence number, so a
  delete written by the latest snapshot still applies (its seq is not below the new
  partition minimum).

## Root cause

1. `plan_dv_removal` collects `live_file_scoped_position_deletes` — every position
   delete with a derivable referenced path, ANY format — and drops the ones whose
   referenced path was rewritten. Java drops only `isDV` (Puffin) there.
2. `find_dangling_deletes` (`remove_dangling_delete_files.rs`) applies the
   referenced-path dangling arm to every position delete with a derivable path, ANY
   format. Java applies that arm to deletion vectors only.

Both must be restricted to `is_deletion_vector(data_file)` (Puffin).

## Step 1 — red cells

`crates/iceberg/src/maintenance/rewrite_data_files_cow_bytes_tests.rs` (new file, 432
lines, under the directory ceiling; ASF header copied from a sibling, no other
comments), wired in `crates/iceberg/src/maintenance/mod.rs`. Eight tests over the
shape above; the five file-scoped cells were red pre-fix (`removed_delete_files_count`
1 vs 0 — exact assertion: `left: 1, right: 0`), the two partition-scoped controls and
the genuinely-dangling sequence control were already green:

| test | cell | pre-fix |
|---|---|---|
| `test_cow_bytes_partition_delete_threshold_keeps_applicable_delete` | partition-scoped `delete-file-threshold=1` | green (rewritten 4, added 1, removed 0) |
| `test_cow_bytes_file_scoped_delete_threshold_keeps_applicable_delete` | file-scoped `delete-file-threshold=1` | RED (removed 1) |
| `test_cow_bytes_rewrite_all_keeps_applicable_delete` | file-scoped `rewrite-all=true` | RED (removed 1) |
| `test_cow_bytes_remove_dangling_keeps_applicable_delete` | file-scoped `rewrite-all` + `remove-dangling-deletes` | RED (removed 1) |
| `test_cow_bytes_remove_dangling_single_row_keeps_applicable_delete` | file-scoped, one deleted row | RED (removed 1) |
| `test_cow_bytes_partition_delete_survives_dangling_cleanup` | partition-scoped `rewrite-all` + cleanup | green |
| `test_cow_bytes_new_sequence_keeps_delete_without_cleanup` | file-scoped, `use-starting-sequence-number=false` | RED (removed 1 — Java keeps it: no cleanup asked) |
| `test_cow_bytes_new_sequence_dangling_cleanup_removes_delete` | file-scoped, new seq + cleanup | green (removed 1 — genuinely dangling, seq `2 < 3`) |

Each cell asserts the Spark/Java answer: the delete file stays in
`live_delete_file_paths`, `removed_delete_files_count == 0`, the deleted rows stay
deleted through `scan_rows`, and the output data files carry the starting snapshot's
data sequence number (2; file seq 3) under the default, or the new sequence (3/3)
under `use-starting-sequence-number=false`.

## Step 2 — the fix

Two removal paths restricted to `is_deletion_vector` (Puffin), matching Java's
`danglingDVs` / `ContentFileUtil.isDV`:

- `rewrite_data_files_dv.rs::plan_dv_removal` — the drop predicate gains
  `is_deletion_vector(delete_file) &&`: a file-scoped PARQUET position delete whose
  referenced data file was rewritten is no longer dropped in the `RewriteFiles`
  commit. Puffin DVs still are. The stale doc sentence claiming the parquet extension
  is deleted.
- `remove_dangling_delete_files.rs::find_dangling_deletes` — the reference arm is
  split: a Puffin DV with a dead or missing `referenced_data_file` dangles
  immediately (Java's left-join-null semantics for DVs); a NON-DV file-scoped delete
  with a LIVE referenced path is kept (the reader still honors it by path —
  `delete_file_index` routes it, resurrecting masked rows if dropped); a non-DV
  delete with a dead reference falls through to the Java partition `(spec_id,
  partition)` min-seq rule instead of being dropped on path alone. Net effect: Java's
  seq rule judges parquet deletes; the dead-reference shortcut judges only DVs. The
  change is line-negative (1804 → 1798; ceiling lowered in
  `scripts/check_rust_file_size.py`).

`map.md` rows updated in the same change (AGENTS.md `map_md_navigation`): the
"parquet delete file stays" symptom now documents the kept delete as expected
behavior, and the partition-scoped row records the Puffin-only drop.

### Pins re-pinned to the Java/Spark answer (names preserved, none deleted)

Each asserted the refuted f16 "Spark reclaims file-scoped parquet deletes" inference
and now asserts the kept delete:

- `rewrite_data_files_mw7_tests.rs::test_mw7_unpartitioned_single_file_partition_scoped_full_bounds_is_reclaimed`
  — `removed_delete_files_count` 1 → 0; the delete stays live.
- `rewrite_data_files_ratio_tests.rs::test_bounds_only_file_scoped_parquet_fires_ratio`
  — 1 → 0.
- `rewrite_data_files_ratio_tests.rs::test_fully_deleted_in_band_parquet_file_is_rewritten_and_drops_its_delete`
  — 1 → 0.
- `rewrite_data_files_ratio_tests.rs::test_fully_deleted_2500_row_in_band_parquet_file_ends_at_zero_delete_files`
  — 1 → 0.
- `rewrite_data_files_ratio_tests.rs::test_partition_scoped_delete_survives_partial_rewrite`
  — 1 → 0 (the file-scoped delete on the rewritten file also stays live).
- `rewrite_data_files_router_bound_tests.rs::evolved_spec_rewrite_drops_file_scoped_position_deletes`
  — the parquet position delete stays live; the DV sibling assertions unchanged.

Candidate `remove_dangling_delete_files.rs::test_dangling_position_delete_parquet_removed_after_data_rewritten_away`
was left UNCHANGED and stays green: its delete has no live referenced path AND no live
data file in its stamped partition, so the Java seq rule drops it either way — the
pin proves the seq arm still fires for a genuinely orphaned parquet delete.

Post-fix suite (`CARGO_BUILD_JOBS=6 RUST_TEST_THREADS=6`):

- `cargo test -p iceberg --lib cow_bytes` — 8/8 green.
- `cargo test -p iceberg --lib rewrite_data_files` — 104/104 green (includes every
  re-pinned test and all DV cells; `evolved_spec_rewrite_drops_file_scoped_dv_and_keeps_sibling`
  and `test_rewriting_one_file_keeps_sibling_dv_in_same_puffin` still prove the DV arm).
- `cargo test -p iceberg --lib remove_dangling` — 23/23 green.
- `cargo test -p iceberg --lib rewrite_position_delete` — 93/93 green (RPD's own DV
  handling untouched).

## Step 3 — mutation proof

`git checkout 194503d4 -- remove_dangling_delete_files.rs rewrite_data_files_dv.rs`
(reverts ONLY the fix; the re-pinned tests and new file stay), then
`cargo test -p iceberg --lib cow_bytes`:

RED, exactly the five file-scoped cells, all `removed_delete_files_count` 1 vs 0:

- `test_cow_bytes_file_scoped_delete_threshold_keeps_applicable_delete` — FAILED
- `test_cow_bytes_rewrite_all_keeps_applicable_delete` — FAILED
- `test_cow_bytes_remove_dangling_keeps_applicable_delete` — FAILED
- `test_cow_bytes_remove_dangling_single_row_keeps_applicable_delete` — FAILED
- `test_cow_bytes_new_sequence_keeps_delete_without_cleanup` — FAILED (the delete is
  dropped by `plan_dv_removal` in the commit itself — proof the rewrite-commit path,
  not only the cleanup sub-action, carried the defect)

GREEN through the mutation (correct controls):

- `test_cow_bytes_partition_delete_threshold_keeps_applicable_delete`
- `test_cow_bytes_partition_delete_survives_dangling_cleanup`
- `test_cow_bytes_new_sequence_dangling_cleanup_removes_delete` (the seq arm still
  removes a genuinely dangling delete — the mutation does not over-keep)

`git checkout HEAD -- <the two files>` → 8/8 green. Revert not committed.

## Step 4 — RePark xfail forecast after the bump

- `delete_file_threshold` VALUE cell (ICE-RDF-COW-BYTES-1): the vanished-sum now
  contains only the rewritten data file — the delete survives — so
  `rewritten_bytes_count` should match Spark. The rewritten-count half depends on
  RePark's delete shape (see the reconcile note): file-scoped → 1 vs Spark's 4 stays
  an xfail on the count; partition-scoped → 4/1 matches and the cell flips to pass.
- `remove_dangling` value cell + BOTH removed-count cells (ICE-RDF-DANGLE-2): flip to
  Spark's answer — `removed_delete_files_count` 0, the delete file live.
- `test_residue_matches_spark_zero_delete_files`: unchanged by this lane where the
  deletes are Puffin DVs (still removed when their referenced file goes) or
  partition-scoped (still seq-judged). If any residue cell relied on the over-broad
  removal of file-scoped PARQUET deletes to reach Spark's zero, it could newly
  diverge — flagged, not expected: Spark's zero comes from seq-GC
  (`dropDeleteFilesOlderThan`), a different mechanism.

## RePark reconcile note

The ICE-RDF-COW-BYTES-1 xfail text records "result counts match Spark (4/1)" for
`delete_file_threshold`. Under the current fork the faithful RePark delete shape
(bounds-file-scoped) attaches to one file → rewritten 1; a rewritten count of 4 requires
a partition-scoped attach, which nothing then removes. Both shapes' measurements are
recorded above; the xfail's rewritten count could not be reproduced for a file-scoped
delete and is believed to reflect Spark's own partition-scoped delete files (or a
stale measurement). The removed-count divergence (the actual defect) reproduces exactly.
