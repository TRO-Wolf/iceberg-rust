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

# F-RDF-SUMMARY-1 — every `replace` commit writes Java's summary keys with live totals

## Finding (RePark, fork `44834673`)

MoR table (format 2): INSERT 1,000 rows → three MERGEs that each write a data file and a
position-delete file (committed through `RowDelta`) → `RewriteDataFiles::execute` (defaults).
The `replace` snapshot summary recorded:

```
changed-partition-count=1, deleted-data-files=13, deleted-records=1600, removed-delete-files=2,
removed-files-size=19010, removed-position-delete-files=2, removed-position-deletes=400,
total-data-files=0, total-delete-files=1, total-equality-deletes=0, total-files-size=1715,
total-position-deletes=200, total-records=0
```

while the table's live files held ONE compacted data file with `record_count=1000` plus one
position-delete file. The `added-data-files` / `added-records` / `added-files-size` keys are
absent and every total that depends on them is wrong.

## Root cause

`SnapshotProducer::commit` (`crates/iceberg/src/transaction/snapshot.rs`):

1. `commit` resolves `removed_data_files` and `removed_delete_files`, then computes the summary
   via `self.summary(...)` (line ~1357).
2. `commit` then calls `self.manifest_file(...)` (line ~1390). Inside `manifest_file`,
   `write_added_manifests` (line ~918) and `write_added_delete_manifests` (line ~979) each do
   `std::mem::take` on `self.added_data_files` / `self.added_delete_files`, leaving them EMPTY.
3. `process_deletes` (`transaction/snapshot/manifest_filter.rs`) can extend
   `self.removed_delete_files` with sequence-number-expired delete files (the Java
   `dropDeleteFilesOlderThan(minDataSequenceNumber)` GC). A fully-tombstoned rewritten data
   manifest gets `min_sequence_number = UNASSIGNED` (the writer counts live entries only,
   `spec/manifest/writer.rs` line ~413), so `DeleteFileExpiry::new` folds down to
   `last_sequence_number` and every older delete file expires. On a compaction that rewrites
   all data files this fires every time.
4. When `removed_delete_files` grew, `commit` recomputes the summary (line ~1393-1398) — but the
   added-file vectors were already consumed by `mem::take`, so the recomputed collector sees no
   added data or delete files: every `added-*` key vanishes and totals degrade to
   `parent - removed`.

The asymmetry: `manifest_file` takes `removed_*` at its top but RESTORES them before returning
(lines ~1086-1088), precisely so the recompute still sees them; the added vectors were taken
without restore. Every shared `replace` path that drops old delete files (rewrite_data_files,
rewrite_position_delete_files, remove_dangling_deletes, V3 dangling-DV expiry) hits this.

## Step 1 — RED reproduction

`crates/iceberg/src/maintenance/replace_commit_summary_tests.rs`:
`rdf_replace_summary_counts_added_files_and_live_totals` builds the MoR shape with the fork's own
writers (four appended data files, then three `RowDelta` commits each adding a 200-row data file
and a one-row parquet position delete on `data-0.parquet`), runs `RewriteDataFiles::execute` with
defaults, reloads the table, and asserts the `replace` summary against the live manifest walk.

Red on `44834673` (`cargo test -p iceberg --lib replace_commit_summary`):

```
missing summary key 'added-data-files' in {"total-delete-files": "1", "total-records": "0",
"changed-partition-count": "1", "removed-position-delete-files": "2", "total-equality-deletes": "0",
"total-files-size": "1127", "deleted-data-files": "7", "removed-delete-files": "2",
"total-position-deletes": "1", "total-data-files": "0", "deleted-records": "1600",
"removed-files-size": "22950", "removed-position-deletes": "2"}
```

Same shape as the RePark measurement: no `added-*` keys, `total-records=0`, `total-data-files=0`,
`total-files-size` = only the surviving position-delete file. The sequence-number GC expired the
two oldest position deletes (`removed-position-delete-files=2`), which is the growth in
`removed_delete_files` that triggers the buggy recompute.

## Fix

(in progress)
