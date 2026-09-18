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
| 0 | this commit | step-0 measurement |
| 1 | | `test: F-RDF-COW-BYTES-1 — red delete-survival cells` |
| 2 | | `fix: F-RDF-COW-BYTES-1 — the rewrite keeps a position delete that still applies, as Java does` |
| 3 | | (mutation proof — no commit) |
| 4 | | `docs: F-RDF-COW-BYTES-1 — ledger, mutation proof` |

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

## Pins to re-pin (asserted the refuted f16 inference)

Pins asserting a parquet position delete is removed after its referenced file is
rewritten were written from the f16 "Spark reclaims file-scoped parquet deletes"
inference, which the MERGE probe above refutes end-to-end. Each is changed to the
Java/Spark answer (delete kept, `removed_delete_files_count` 0) in step 2 and named
here when changed — none is deleted.

Candidates identified so far (confirmed against the suite in step 2):

- `rewrite_data_files_mw7_tests.rs::test_mw7_unpartitioned_single_file_partition_scoped_full_bounds_is_reclaimed`
- `remove_dangling_delete_files.rs::test_dangling_position_delete_parquet_removed_after_data_rewritten_away`
- `rewrite_data_files_ratio_tests.rs::test_partition_scoped_delete_survives_partial_rewrite`
  (expects one file-scoped parquet delete removed)

## RePark reconcile note

The ICE-RDF-COW-BYTES-1 xfail text records "result counts match Spark (4/1)" for
`delete_file_threshold`. Under the current fork the faithful RePark delete shape
(bounds-file-scoped) attaches to one file → rewritten 1; a rewritten count of 4 requires
a partition-scoped attach, which nothing then removes. Both shapes' measurements are
recorded above; the xfail's rewritten count could not be reproduced for a file-scoped
delete and is believed to reflect Spark's own partition-scoped delete files (or a
stale measurement). The removed-count divergence (the actual defect) reproduces exactly.
