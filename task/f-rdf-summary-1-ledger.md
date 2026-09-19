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

Two commits, both in `crates/iceberg/src/transaction/snapshot.rs` (the shared producer — every
`replace` path runs through it: `RewriteFiles`, `RewriteDataFiles`, `RewritePositionDeleteFiles`,
`RemoveDanglingDeleteFiles`, and `RewriteManifests`):

- `b8acea60` — `write_added_manifests` (~line 918) and `write_added_delete_manifests` (~line 979)
  now CLONE `added_data_files` / `added_delete_files` instead of `std::mem::take`-ing them, so the
  post-`manifest_file` summary recompute (fired whenever `process_deletes` grows
  `removed_delete_files`) still sees every added file. `manifest_file` now returns
  `(Vec<ManifestFile>, usize)` where the second element is the count of manifests rewritten during
  filtering, computed inside `process_deletes` (`manifest_filter.rs`).
- `e2836a5f` — after `manifest_file`, `commit` injects Java's manifest counts on
  `Operation::Replace` only: `manifests-created` / `manifests-kept` counted from the FINAL manifest
  list by `added_snapshot_id == self.snapshot_id`, `manifests-replaced` filled from the filter
  count via `entry().or_insert` so `RewriteManifests` (which computes its own
  created/kept/replaced/entries-processed via `extend_snapshot_properties`) keeps its authoritative
  values.

Java comparison (`MergingSnapshotProducer.apply`): `addedDataSummary + addedDeleteSummary +
appendedManifestSummary + filteredDataManifestSummary + filteredDeleteManifestSummary`, then
`manifests-created/-kept` counted on the final list by `snapshotId` and
`manifests-replaced = filterManager.manifestsReplaced() + mergeManager.manifestsReplaced()`. The
fork's `commit` now produces the same shape: the summary is accumulated once (re-accumulated only
when the delete-side removal set grows — now with the added files still present) and the three
manifest counters are stamped from the final list, matching Java's `SnapshotSummary` semantics of
omitting zero-valued update counters while always writing the six `total-*` keys.

## Pins (step 4)

`crates/iceberg/src/maintenance/replace_commit_summary_tests.rs` — every test asserts the Java key
set (required + forbidden) AND the count invariants against a live manifest walk
(`total-*` = live file/record/size sums; `added-*` = snapshot-added files; chain
`total = parent + added − removed`):

- `rdf_replace_summary_counts_added_files_and_live_totals` — RDF defaults (the step-1 repro)
- `rdf_rewrite_all_replace_summary_matches_java_keys` — `rewrite_all(true)`
- `rdf_fresh_sequence_numbers_replace_summary_matches_java_keys` —
  `use_starting_sequence_number(false)`
- `rdf_partial_progress_chains_totals_across_commits` — `partial_progress` + `max_commits(2)` +
  `FilesDesc`; two commits, per-commit totals verified against each commit's own manifest list and
  the `parent + added − removed` chain; batch-2 exercises the recompute path (one seq-expired
  position delete)
- `rdf_bucketed_table_replace_summary_matches_java_keys` — `bucket(8, x)` spec
- `rdf_v3_dv_replace_summary_reports_removed_dvs` — v3, four DV blobs in one Puffin:
  `removed-dvs=4`, `removed-position-deletes=100`, `total-records=900`, `total-delete-files=0`
- `rdf_v3_rewrite_all_removes_equality_and_dv_delete_files` — v3 with an equality delete committed
  BEFORE the DVs so the seq-GC expires it during the rewrite (`removed-equality-delete-files=1`,
  `removed-dvs=4`, `removed-delete-files=5`) — this is the shape that drives the recompute on v3
- `rpd_v2_replace_summary_matches_java_keys` — 3 parquet position deletes merged to 1:
  `added-position-delete-files`, no `added-dvs`
- `rpd_v3_replace_summary_reports_added_dv` — v2 seeded, upgraded to v3: parquet deletes merge into
  ONE DV (`added-dvs=1`, no `added-position-delete-files`)
- `rm_data_only_replace_summary_matches_java_keys` — `cluster_by` rewrites 3 data manifests to 1:
  `manifests-created=1`, `manifests-kept=0`, `manifests-replaced=3`, `entries-processed=3`, totals
  carried
- `rm_delete_manifests_replace_summary_matches_java_keys` — `rewrite_delete_manifests(true)`:
  `manifests-created=2`, `manifests-replaced=3`, totals carried

## Mutation evidence (step 4)

- Reverted the `b8acea60` clone lines to `std::mem::take` (`cargo test -p iceberg --lib
  replace_commit_summary`): 6 of 11 pins red — `rdf_replace_summary_counts_added_files_and_live_totals`,
  `rdf_rewrite_all_replace_summary_matches_java_keys`,
  `rdf_fresh_sequence_numbers_replace_summary_matches_java_keys`,
  `rdf_partial_progress_chains_totals_across_commits`,
  `rdf_bucketed_table_replace_summary_matches_java_keys`,
  `rdf_v3_rewrite_all_removes_equality_and_dv_delete_files`. Failing assertion (all six):
  `missing summary key 'added-data-files'` with `total-records=0`, `total-data-files=0`. The
  `rdf_v3_dv` pin stays green because its DVs are removed by the targeted `plan_dv_removal` set
  (resolved BEFORE `summary()`), not by the post-`manifest_file` seq-GC growth — expected.
- Dropped the `manifests-created` insert in `commit` (rest of the fix intact): 9 of 11 pins red —
  every RDF and RPD key-set pin fails with `required key 'manifests-created' absent`; the two RM
  pins stay green because `RewriteManifests` stamps its own counts. Both mutations restored after
  the run; the suite re-verified 11/11 green.

## Gates

- `cargo fmt --all` — clean
- `cargo clippy -p iceberg --all-targets -- -D warnings` — clean
- `cargo test -p iceberg --lib replace_commit_summary` — 11 passed
- `cargo test -p iceberg --lib snapshot_summary` — 16 passed
- `cargo test -p iceberg --lib rewrite_` — 319 passed
- `cargo test -p iceberg --lib replace_record_count` — 5 passed
- the lane's comment-ban gate against `origin/main` — `comment-ban hits=0` after every commit

## Residue

- `manifests-*` keys are emitted only on `Operation::Replace` (the scoped operations). `append`,
  `overwrite`, and `delete` commits still do not write them; Java writes them only on
  merge/filter operations too, so this matches — flagging as observed scope, not a gap.
- `entries-processed` remains a `RewriteManifests`-only key, matching Java.
- `manifests-replaced` for `RewriteManifests` counts action-level rewritten+deleted manifests
  (its own `extend_snapshot_properties` values win via `or_insert`); for `RewriteFiles`-family
  commits it counts manifests rewritten by the delete-side filter — the fork has no
  manifest-merge manager, so there is no merge-side component to add.
