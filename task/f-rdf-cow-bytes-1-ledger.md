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
**Model:** swe-2-high (rounds 1–3); claude-opus-5 (round 4)

| Item | Commits | Subject |
|---|---|---|
| 0 | `66299118` | `docs: F-RDF-COW-BYTES-1 — step 0 measurement` |
| 1 | `194503d4` | `test: F-RDF-COW-BYTES-1 — red delete-survival cells` |
| 2 | `5e7455f2` | `fix: F-RDF-COW-BYTES-1 — the rewrite keeps a position delete that still applies, as Java does` |
| 3 | — | (mutation proof — no commit) |
| 4 | `292c4f71` | `docs: F-RDF-COW-BYTES-1 — ledger, mutation proof` |
| 5 | `b8002010` | `docs: F-RDF-COW-BYTES-1 — delete the stale module sentence; seq-GC audit` |
| R4-1 | `c7c45299` | `test: F-RDF-COW-BYTES-1 — red delete-file sequence GC cells` |
| R4-2 | `b273d097` | `fix: F-RDF-COW-BYTES-1 — merging commits retire delete files older than every live data file, as Java's dropDeleteFilesOlderThan` |
| R4-3 | `47d767d8` | `fix: F-RDF-COW-BYTES-1 — review items R-02, R-04; stale module line deleted` |
| R4-4 | — | (mutation proof — no commit) |
| R4-5 | this commit | `docs: F-RDF-COW-BYTES-1 — round 4` |

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
(`ManifestFilterManager.isDanglingDV` inside `MergingSnapshotProducer.apply`, via `ContentFileUtil.isDV`) is Puffin-only, so
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
the shared JVM lock wrapper with the Spark virtualenv's python (driver 2g, UI off):

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
- `ManifestFilterManager.isDanglingDV` (1.11.0 L493-495, reached through `MergingSnapshotProducer.apply` → `removeDanglingDeletesFor`, L994-995; round 4 corrected the round-1 name `RewriteDataFilesSparkAction.danglingDVs`, which does not exist in 1.11.0) — drops only `isDV` deletes
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
`ManifestFilterManager.isDanglingDV` / `ContentFileUtil.isDV`:

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

## Round 3 — comment-gate remediation + seq-GC audit (read-only)

The mechanical comment gate rejected round 2 on one hit: step 2 had truncated the
`rewrite_data_files_dv.rs` module-doc line `//! only. File-scoped parquet position
deletes are a fork extension of that predicate.` to `//! only.`, and an edited
comment line counts as an added one. Round 3 deletes the whole three-line sentence
(`//! Java 1.10.0 ManifestFilterManager.isDanglingDV is …`, `//!
removedDataFilePaths.contains(…). The apply path drops DVs`, `//! only. …`) — pure
deletion, no comment line added or changed. The fact the sentence carried is already
recorded in "Java rules confirmed from source" above (`ManifestFilterManager.isDanglingDV`
drops only `isDV` deletes whose referenced data file was rewritten; Puffin-only).

### Seq-GC audit — does the fork port `MergingSnapshotProducer.dropDeleteFilesOlderThan`?

**No.** Java's `MergingSnapshotProducer.apply` runs
`filterManager.dropDeleteFilesOlderThan(minDataSequenceNumber)` on every commit that
rewrites data: delete files whose data sequence number is below the minimum data
sequence number of the surviving live data files are dropped silently inside the
commit — `removed_delete_files_count` stays 0 because the drop never surfaces in the
action result. The fork's port point is `SnapshotProducer::current_manifests`
(`crates/iceberg/src/transaction/snapshot.rs`), whose own doc records the divergence
verbatim: Java's `apply` "also drops delete files older than the surviving data's
minimum sequence number, and removes DVs orphaned by the data files it deleted.
This port carries every delete manifest forward UNCHANGED." `process_deletes` (same
file) tombstones only entries for explicitly requested `removed_data_files` /
`removed_delete_files`; no sequence-based GC of delete entries exists anywhere in
the commit path. The only seq-based removal the fork has is
`RemoveDanglingDeleteFiles` / `find_dangling_deletes`, which runs only under the
`remove-dangling-deletes=true` option.

Consequence: Spark's `residue_rpd_then_rdf` sequence reaches ZERO delete files
through that Java channel (RPD re-stamps rewritten deletes at
`maxRewrittenDataSequenceNumber`, then the RDF commit's seq-GC drops them), which
the fork cannot reproduce until `dropDeleteFilesOlderThan` is ported. That port is
the remaining ICE-RDF-DANGLE-2 residue — the next fork ask, not this PR.

## RePark reconcile note

The ICE-RDF-COW-BYTES-1 xfail text records "result counts match Spark (4/1)" for
`delete_file_threshold`. Under the current fork the faithful RePark delete shape
(bounds-file-scoped) attaches to one file → rewritten 1; a rewritten count of 4 requires
a partition-scoped attach, which nothing then removes. Both shapes' measurements are
recorded above; the xfail's rewritten count could not be reproduced for a file-scoped
delete and is believed to reflect Spark's own partition-scoped delete files (or a
stale measurement). The removed-count divergence (the actual defect) reproduces exactly.

## Round 4 — port of Java's delete-file sequence GC (`dropDeleteFilesOlderThan`)

Perf review R-01 (P1): after round 1 the fork correctly keeps a file-scoped parquet
delete, but it had no port of the commit-path sequence GC. Deletes that apply to no live
row then stayed in every later snapshot, while Java retires them. Round 4 ports the GC.
It also closes R-02 and R-04 and the two logic nits, and records R-03.

### Java 1.11.0 lines relied on

Source: tag `apache-iceberg-1.11.0`, fetched from GitHub raw this round and diffed
against the logic reviewer's copies (identical).

| Fact | File:line |
|---|---|
| The minimum is over the FILTERED existing data manifests (`filterManager.filterManifests(snapshot.dataManifests())`), `ManifestFile::minSequenceNumber`, `UNASSIGNED_SEQ` skipped, reduced from `base.lastSequenceNumber()` | `core/.../MergingSnapshotProducer.java:977-989` |
| `deleteFilterManager.dropDeleteFilesOlderThan(minDataSequenceNumber)` | `MergingSnapshotProducer.java:990` |
| `removeDanglingDeletesFor(filterManager.filesToBeDeleted())` (the DV half) | `MergingSnapshotProducer.java:994-995` |
| The delete summary merges `deleteFilterManager.buildSummary(filteredDeletes)` | `MergingSnapshotProducer.java:1019` |
| `dropDeleteFilesOlderThan` stores the minimum (`>= 0` precondition) | `core/.../ManifestFilterManager.java:160-164` |
| A live delete entry is marked when `dataSequenceNumber() > 0 && dataSequenceNumber() < minSequenceNumber` (strict `<`, every delete content alike) | `ManifestFilterManager.java:462-467` (scan), `:516-522` (rewrite) |
| A marked entry is written with `writer.delete(entry)` and added to the per-manifest deleted set | `ManifestFilterManager.java:534-549` |
| `buildSummary` calls `summaryBuilder.deletedFile(spec, file)` for every file in that set | `ManifestFilterManager.java:254-269` |
| `deletedFile(spec, DeleteFile)` increments `removed-delete-files`, `removed-position-delete-files` / `removed-equality-delete-files` / `removed-dvs` and the record/size counters | `core/.../SnapshotSummary.java:144-146`, `:325-337` |
| The manifest-open gate: `filterManifest` → `canContainDeletedFiles` returns `false` for a manifest with no live files; with trusted manifest references only a referenced manifest is opened; else `canContainDroppedFiles` (true when `deletePaths` is non-empty, when `deleteFiles` overlap the manifest's partitions, or when `removedDataFilePaths` is non-empty), `canContainExpressionDeletes`, `canContainDroppedPartitions` | `ManifestFilterManager.java:368-376`, `:400-447`; trust rule `:241-246` |
| A manifest's `minSequenceNumber` counts LIVE entries only; none → `UNASSIGNED_SEQ` | `core/.../ManifestWriter.java:111-115`, `:222-223` |
| Merging operation set: `MergeAppend`, `BaseOverwriteFiles`, `BaseReplacePartitions`, `BaseRowDelta`, `StreamingDelete`, `BaseRewriteFiles`, `CherryPickOperation` extend `MergingSnapshotProducer`; `FastAppend` and `BaseRewriteManifests` extend `SnapshotProducer` | `MergeAppend.java:24`, `BaseOverwriteFiles.java:31`, `BaseReplacePartitions.java:26`, `BaseRowDelta.java:31`, `StreamingDelete.java:24`, `BaseRewriteFiles.java:26`, `CherryPickOperation.java:46`, `FastAppend.java:36`, `BaseRewriteManifests.java:49` |
| `isDanglingDV` = `ContentFileUtil.isDV(file) && removedDataFilePaths.contains(file.referencedDataFile())` | `ManifestFilterManager.java:493-495` |
| `RemoveDanglingDeletesSparkAction.findDanglingDeletes` filters `data_file.content != 0` with no format test, so DVs are judged by the partition minimum too | `spark/v4.1/.../RemoveDanglingDeletesSparkAction.java:125-176` |
| A DV whose data seq is below its data file's fails the scan: `DV data sequence number (%s) must be greater than or equal to data file sequence number (%s)` | `core/.../DeleteFileIndex.java:207-213` |

### What the port does

`crates/iceberg/src/transaction/snapshot/manifest_filter.rs` (new) now holds
`process_deletes` (moved out of `snapshot.rs`, comments dropped in the move) and the
GC. `snapshot.rs` 3450 → 3355 lines, ceiling lowered.

- **Operation set.** `SnapshotProduceOperation::drops_old_delete_files` defaults to
  `true`. `FastAppendOperation` and `RewriteManifestsOperation` return `false`: they are
  the two Java `SnapshotProducer` subclasses. Every other operation is a Java
  `MergingSnapshotProducer`: merge append, overwrite, replace partitions, row delta,
  delete files, rewrite files, and cherry-pick replay.
- **Minimum.** Data manifests are filtered first. The minimum is the fold of
  `min_sequence_number` over the filtered existing DATA manifests (rewritten or carried,
  dead ones included, before the keep rule), skipping `UNASSIGNED_SEQUENCE_NUMBER`, with
  identity `last_sequence_number()`. It is global, not per partition, as Java's. Added
  data files of this commit are not in it, as in Java (they are in
  `prepareNewDataManifests`, not `filtered`). The fork's `ManifestWriter` also counts
  only live entries (`spec/manifest/writer.rs`), so a fully-tombstoned rewritten
  manifest is skipped exactly as Java skips it.
- **Retirement.** A live entry with `0 < data seq < minimum` in an examined DELETE
  manifest becomes a `Deleted` entry (`add_delete_entry`, snapshot id and both
  sequence numbers kept) in the rewritten delete manifest. Delete manifests added by
  this commit are never examined.
- **Summary.** `process_deletes` returns the retired `DataFile`s. `manifest_file`
  appends them to `removed_delete_files`, and `commit` rebuilds the summary when that
  list grew. The snapshot therefore reports `removed-delete-files`,
  `removed-position-delete-files` / `removed-equality-delete-files` / `removed-dvs`,
  the removed record and size counters, and lower `total-*` values, like Java's
  `buildSummary`. The REPLACE record-count guard still runs on the first summary,
  before manifest IO. Retirement does not change `added-records` / `deleted-records`.
- **Scope (Java's manifest-open gate).** The fork has no `manifestLocation`, so it
  cannot model Java's trusted-reference test exactly. Rule: when the commit removes a
  data file, every live delete manifest is examined (Java: `removedDataFilePaths`
  non-empty and references not trusted). When it removes only delete files, only the
  delete manifests holding one of them are examined (Java: trusted references, the
  scan-derived callers — RPD, `RowDelta.removeDeletes`). When it removes nothing, no
  manifest is examined (merge append, add-only row delta). A delete manifest whose own
  `min_sequence_number` is not below the minimum is not read: none of its live entries
  can expire, so Java's rewrite of it would be a no-op too.
- **Soundness.** A position delete applies to data seq ≤ its own, an equality delete to
  data seq < its own. Below the minimum it applies to no live data file. Added files
  carry the new snapshot's seq, or with `use-starting-sequence-number` the starting
  snapshot's seq, which is ≥ every retired delete's seq only when the delete is older
  than the minimum. That case is exactly the one where the rewrite already read the
  delete when it produced the new file.

Named scope differences (all keep row safety; the fork examines fewer manifests, except
the first case):

1. `RowDelta` that removes data files AND delete files. Java trusts the references and
   opens only the manifests holding the removed deletes. The fork examines every
   delete manifest, so it retires more.
2. `RemoveDanglingDeleteFiles` commit (removes delete files only). Java's
   `SparkDeleteFile` has no manifest location, so Java opens every delete manifest whose
   partitions overlap a removed file. The fork examines only the manifests that hold a
   removed file.
3. `DeleteFiles.delete_from_row_filter` / `ReplacePartitions` that match no data file.
   Java still opens delete manifests through `deleteExpression` / `dropPartitions`. The
   fork examines none.
4. `removeDanglingDeletesFor` (Java drops a DV whose referenced data file ANY merging
   commit removes) is still ported only on the `RewriteDataFiles` path
   (`plan_dv_removal`). `DeleteFiles` / `OverwriteFiles` / `RowDelta` removing a data
   file keep its DV until the sequence GC or `RemoveDanglingDeleteFiles` retires it. This
   is the next unit, not this one.

No cell pins cherry-pick replay. It takes the default `true`, as Java's
`CherryPickOperation`.

### Red cells (R4-1, `c7c45299`)

New file `crates/iceberg/src/maintenance/delete_file_seq_gc_tests.rs` (maintenance/,
because cell (a) drives RPD and RDF; the other cells drive transaction actions
directly). The helpers `write_position_delete`, `file_scoped_metrics` and
`partition_scoped_metrics` in `rewrite_data_files_cow_bytes_tests.rs` became
`pub(super)`.

`cargo test -p iceberg --lib seq_gc` at `c7c45299`: 5 passed, 6 failed.

| test | brief item | pre-fix |
|---|---|---|
| `test_seq_gc_residue_rpd_then_rdf_reaches_zero_delete_files` | (a) v2, 2 partitions × 8 files, file-scoped parquet delete of every even `y` per file, RPD default (16 → 2), RDF default (16 → 2): 2 data + 0 delete files, rows unchanged, `removed_delete_files_count` 0, summary `removed-position-delete-files=2`, `total-delete-files=0` | RED at the zero-delete assert (every count before it matched) |
| `test_seq_gc_keeps_position_delete_at_the_minimum_live_sequence` | (b) a position delete at the minimum live seq still masks rows after a `DeleteFiles` commit | green |
| `test_seq_gc_keeps_equality_delete_applying_to_older_live_data` | (b) equality delete at seq 3 over live data at seq 1 survives a `DeleteFiles` commit | green |
| `test_seq_gc_fast_append_keeps_stale_delete` | (b)/(c) fast append | green |
| `test_seq_gc_merge_append_keeps_stale_delete` | (c) merge append (Java opens no delete manifest) | green |
| `test_seq_gc_row_delta_adding_deletes_only_keeps_stale_delete` | (c) add-only row delta | green |
| `test_seq_gc_delete_files_retires_stale_delete` | (c) `DeleteFiles` | RED |
| `test_seq_gc_overwrite_files_retires_stale_delete` | (c) `OverwriteFiles` | RED |
| `test_seq_gc_replace_partitions_retires_stale_delete` | (c) `ReplacePartitions` | RED |
| `test_seq_gc_row_delta_removing_data_retires_stale_delete` | (c) `RowDelta.remove_rows` | RED |
| `test_seq_gc_rewrite_files_retires_stale_delete` | (c) `RewriteFiles` | RED |

The (c) cells use a delete at seq 1 that names a data path the table never held, then
append the live data at seq 2. Each retiring cell asserts: no live delete file,
`removed-position-delete-files=1`, `removed-delete-files=1`, `total-delete-files=0`, and
the scan rows. The fast-append cell passes without the flag too: `FastAppend` has no
removal surface. It pins behaviour, not the flag.

### Fix (R4-2, `b273d097`) — existing pins re-examined

Three existing pins failed on the fix. Each asserted the pre-port carry-forward:

- `rewrite_position_delete_files_tests.rs::test_v3_non_superset_refusal_is_cleared_by_rewrite_data_files`.
  The shadowed parquet delete (seq 2) sits below the rewrite's minimum (4). Java's
  rewrite commit retires it. Re-pinned: after the default rewrite no delete file is
  live. With `remove_dangling_deletes(true)`, `removed_delete_files_count` is 1 (the DV
  only), because the dangling pass finds nothing. Rows are unchanged in both halves.
- `rewrite_data_files_dangling_tests.rs::test_remove_dangling_deletes_defaults_off` and
  `…_on_removes_the_dangling_delete`. The single-partition fixture put every data file
  in the rewrite, so the minimum rose to 3 and Java's commit GC would retire the delete
  (then the sub-action finds nothing). These tests exist to prove the opt-in sub-action
  composes. The fixture now appends one extra data file in partition `x = 1` at seq 1.
  The rewrite does not touch it (one file, under `min-input-files`), so the global
  minimum stays 1 and no sequence GC fires, in Java or here. The partition-`x = 0`
  minimum still rises to 3, so only `RemoveDanglingDeleteFiles` removes the delete.
  Both tests keep every assertion unchanged. The fixture's doc block (it said
  "Everything sits in partition `x = 0`") was deleted, not reworded.

Comment lines deleted because they became false: the `current_manifests` doc
paragraphs "Every DELETE manifest carries forward UNCHANGED" and "Conservative
dangling-delete posture", the `manifest_file` note "Manifests that contain none of the
target files are carried forward unchanged", and the `existing_manifest` blocks in
`rewrite_files.rs`, `delete_files.rs`, `overwrite_files_operation.rs`,
`replace_partitions.rs` (including its full-table-replace note, whose "this port keeps
them" is no longer true for deletes older than the last sequence) and `cherry_pick.rs`.
To fit the two opt-outs under the frozen ceilings, duplicated comment lines were
deleted in `append.rs` (the merge-append aside, which `merge_append.rs` already records,
and the "properties used to create SnapshotProducer" note) and `rewrite_manifests.rs`
(three lines restating the struct doc or line 308). Ceilings lowered: `snapshot.rs`
3355, `rewrite_files.rs` 2458, `delete_files.rs` 2257, `replace_partitions.rs` 2784,
`cherry_pick.rs` 2103, `rewrite_position_delete_files_tests.rs` 4715. `map.md`
(transaction, maintenance) rows updated.

### Review items (R4-3, `47d767d8`)

- **R-02 (P2).** `live_file_scoped_position_deletes` now returns
  `LiveFileScopedDeletes { paths, deletion_vectors }` from one walk. `paths` (every
  file-scoped position delete) feeds the planner's ratio. `deletion_vectors` clones only
  Puffin DVs, the only files `plan_dv_removal` can drop. Before, every file-scoped
  parquet `DataFile` was cloned as well. `plan_dv_removal` no longer re-tests
  `is_deletion_vector`: its input is DV-only. `file_scoped_delete_paths_from` is gone,
  and the test-only `file_scoped_delete_paths` reuses the walk.
- **R-04 (P3) — ruling: the reviewer is right under the fork's row-safety posture, not
  under Java parity.** Java's `findDanglingDeletes` judges DVs by the partition minimum
  too (no format filter). On a valid table, a DV with a live referenced file can never
  fall below its partition minimum: `DeleteFileIndex.java:207-213` rejects a DV whose
  seq is below its data file's. A red cell that tried to build that state failed with
  exactly that `DataInvalid` from the fork's planner. The fall-through therefore fires
  only for a DV stamped in a FOREIGN partition with no live data. In that case Java's
  left join gives `min IS NULL` and drops a DV that the reader still honors by path.
  The fork already keeps the parquet analogue
  (`test_file_scoped_position_delete_in_a_foreign_partition_applies_and_survives`).
  The `continue` makes DVs consistent with that. Cell
  `test_remove_dangling_keeps_a_foreign_partition_dv_whose_data_file_is_live`: a v3 DV
  for a file in `x = 1`, stamped `x = 2`, masks `y = 11`. `RemoveDanglingDeleteFiles`
  removes 0 DVs, the DV stays live, and the rows are unchanged. This is a named
  divergence from Java in the row-safe direction.
- **Logic nit (citation).** `RewriteDataFilesSparkAction.danglingDVs` does not exist in
  1.11.0. Every mention in this ledger and in `task/todo.md` now names
  `ManifestFilterManager.isDanglingDV` (reached through `removeDanglingDeletesFor`,
  `MergingSnapshotProducer.java:994-995`).
- **Logic nit (module doc).** The `rewrite_data_files_dv.rs` line `//! Drop file-scoped
  deletes that reference data files this rewrite removes.` and the `//!` line after it
  were deleted, not reworded.
- **R-03 (P2, recorded, not fixed).** RPD can pack small dead file-scoped parquet
  deletes into one partition-scoped output that then attaches to every data file of the
  partition, and every scan opens it (`rewrite_position_delete_files.rs` ~453-657). The
  sequence GC now retires such an output at the next merging commit that removes a data
  file, but only once the output's seq (the max rewritten seq) is below every live data
  file's. A different unit.
- The ledger's Spark-probe line named a machine-local wrapper path. It now describes
  the wrapper without the path.

### Mutation proof (R4-4, not committed)

1. **Revert the step-2 fix.** `git checkout c7c45299 --` `snapshot.rs`,
   `snapshot/removal_targets.rs`, `append.rs`, `rewrite_manifests.rs`, and
   `snapshot/manifest_filter.rs` moved aside. `cargo test -p iceberg --lib seq_gc`:
   6 passed, 6 FAILED — exactly the six retirement cells (the residue cell at "Spark
   4.1.2 + Iceberg 1.11.0 ends the residue sequence at zero delete files", the five (c)
   cells at "a delete older than every live data file is retired by a merging commit").
   Green: both (b) row-safety controls, the three keep controls, and the R-04 cell.
   `test_v3_non_superset_refusal_is_cleared_by_rewrite_data_files` FAILED at "the parquet
   position delete (seq 2) is below the rewrite's minimum live data seq (4)". Restored →
   12/12.
2. **`<` → `<=` in `DeleteFileExpiry::expires` alone.** 12/12 green. The manifest-level
   test `min_sequence_number < minimum` still prunes the delete manifest, so the entry
   test never runs. The mutation was incomplete, not the cell. **Both comparisons →
   `<=`:** 11 passed, 1 FAILED,
   `test_seq_gc_keeps_position_delete_at_the_minimum_live_sequence` ("a delete whose
   sequence equals the minimum live data sequence still applies"). Restored → 12/12.
3. **Revert R-04** (`git checkout b273d097 -- remove_dangling_delete_files.rs`).
   `test_remove_dangling_keeps_a_foreign_partition_dv_whose_data_file_is_live` FAILED,
   `left: 1` (the DV was collected). The other 11 green. Restored → 12/12,
   `remove_dangling` 24/24.

### Gates (round 4, `CARGO_BUILD_JOBS=6 RUST_TEST_THREADS=6`)

- `cargo test -p iceberg --lib seq_gc` 12/12; `--lib transaction` 685 passed, 1 ignored;
  `--lib rewrite_data_files` 104/104; `--lib rewrite_position_delete` 93/93;
  `--lib remove_dangling` 24/24; `--lib cow_bytes` 8/8; `--lib maintenance` 376/376.
- `cargo test -p iceberg-datafusion --lib delete` 38 passed, 1 ignored; `--lib update`
  11/11; `--lib merge` 2/2.
- `cargo fmt --all`; `cargo clippy -p iceberg -p iceberg-datafusion --all-targets -- -D
  warnings` clean; `python3 scripts/check_rust_file_size.py` clean (96 legacy
  ceilings); comment gate `comment-ban hits=0`.

### RePark forecast

- `test_residue_matches_spark_zero_delete_files`: expected to flip to pass. Cell (a)
  reproduces the sequence in-tree and reaches 2 data + 0 delete files. The fork's RPD
  stamps the max rewritten data seq (`rewrite_position_delete_files.rs` `compact_group`),
  so its outputs sit at the DELETE's seq. The default RDF starts from the RPD snapshot,
  and the commit's minimum is that snapshot's seq, so both outputs are retired.
  `removed_delete_files_count` stays 0, as Spark's. Caveat: if RePark's residue runs RDF
  with `use-starting-sequence-number=false` or a filter that leaves an older data file
  live, the minimum stays low and Java keeps the deletes too. Parity still holds.
- ICE-RDF-DANGLE-2 removed-count cells: unchanged from the round-1 forecast
  (`removed_delete_files_count` 0). In those cells the delete's seq equals the rewrite's
  starting seq and the commit minimum, so the strict `<` keeps it, as Java does. The
  eight `cow_bytes` cells are green without change.
- New observable: a merging commit that retires deletes now reports them in its snapshot
  summary (`removed-delete-files`, `removed-position-delete-files`, lower
  `total-delete-files`). RePark cells that compare summaries of such commits should now
  match Spark. Before, they differed by exactly the retired files.
