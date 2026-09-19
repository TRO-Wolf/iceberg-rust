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
- `b03bdab2` — round 2: the Spark oracle (`appends_rm_rdf`, `merges_rdf_only`) shows Java stamps
  the three counters on EVERY operation — plain `INSERT` (append) commits carry
  `manifests-created=1, manifests-kept=0..3, manifests-replaced=0` and MoR merges (overwrite) carry
  `manifests-created=3, manifests-kept=2, manifests-replaced=1`. The `Operation::Replace` guard is
  removed so every snapshot-producing commit is stamped. `manifests-replaced` is now the Java sum
  `filterManager.manifestsReplaced() + mergeManager.manifestsReplaced()`: `ManifestProcess` /
  `DefaultManifestProcess` / `MergeManifestProcess::process_manifests` return
  `(Vec<ManifestFile>, usize)` where the usize is the merge-side count — manifests consumed by a
  merged bin whose `added_snapshot_id` differs from the current snapshot (carried sources only,
  matching `ManifestMergeManager.replacedManifests`); `manifest_file` sums filter + merge counts.

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

Round-2 operation-level pins (counts derived from each commit's own manifest list, compared against
the oracle programs `appends_rm_rdf` / `merges_rdf_only` — key set and count shape, not Spark's
byte sizes):

- `append_commits_stamp_manifest_counts` — first append `created=1, kept=0, replaced=0`; second
  append `created=1, kept=1, replaced=0` (oracle 1st/2nd inserts)
- `row_delta_merge_commit_stamps_manifest_counts` — MoR merge adds one data + one delete manifest,
  keeps one data manifest: `created=2, kept=1, replaced=0`
- `cow_overwrite_commit_stamps_manifest_counts` — copy-on-write rewrite of one file: `created=2,
  kept=0, replaced=1` (filter rewrites the carried manifest)
- `delete_only_commit_stamps_manifest_counts` — a position-delete-only commit writes one delete
  manifest and keeps the data manifest untouched: `created=1, kept=1, replaced=0`
- `merge_append_stamps_merge_side_replaced_count` — `commit.manifest.min-count-to-merge=2`, two
  carried manifests bin-merged: `created=1, kept=0, replaced=2` (merge-side count, oracle
  `merges_rdf_only` shape); the plain `fast_append` under the same property merges nothing and
  keeps `manifests-replaced=0`

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

Round-2 mutations (run on `b03bdab2`):

- Counted kept manifests as created (`added_snapshot_id == id || != id` ⇒ `created += 1`):
  `append_commits_stamp_manifest_counts` red — `second append: manifests-created
  left: 2 right: 1`.
- Re-applied the round-1 `Operation::Replace` guard around the whole stamping block:
  `append_commits_stamp_manifest_counts` red — `missing summary key 'manifests-created'` on the
  append summary. Both restored after the run; the suite re-verified 16/16 green.

## Gates

- `cargo fmt --all` — clean
- `cargo clippy -p iceberg --all-targets -- -D warnings` — clean
- `cargo test -p iceberg --lib replace_commit_summary` — 16 passed
- `cargo test -p iceberg --lib snapshot_summary` — 16 passed
- `cargo test -p iceberg --lib rewrite_` — 319 passed
- `cargo test -p iceberg --lib replace_record_count` — 5 passed
- `cargo test -p iceberg --lib append` — 116 passed
- `cargo test -p iceberg --lib row_delta` — 123 passed
- `cargo test -p iceberg --lib overwrite` — 76 passed
- `cargo test -p iceberg --lib delete_files` — 222 passed
- the lane's comment-ban gate against `origin/main` — `comment-ban hits=0` after every commit

Key-set tests updated for the universal stamping (round-2 brief step 4):

- `transaction::merge_append::tests::test_merge_append_at_threshold_merges_and_preserves_provenance`
  pinned `manifests-created/-kept/-replaced` absent on an append-operation commit (the pre-parity
  shape). Updated to the Java-correct counts for its fixture (one merged manifest; two carried
  sources consumed): `manifests-created=1, manifests-kept=0, manifests-replaced=2` — the same
  values the `merge_append_stamps_merge_side_replaced_count` pin asserts.

## Residue

- `manifests-*` keys are now stamped on EVERY snapshot-producing commit (`append`, `overwrite`,
  `delete`, `replace`) matching the Spark oracle — the round-1 claim that Java only writes them on
  merge/filter operations was refuted by `rdf_summary_truth.json` (`appends_rm_rdf`,
  `merges_rdf_only`).
- `entries-processed` remains a `RewriteManifests`-only key, matching Java.
- `manifests-replaced` for `RewriteManifests` counts action-level rewritten+deleted manifests
  (its own `extend_snapshot_properties` values win via `or_insert`); for all other commits it is
  the Java `filter + merge` sum — filter-rewritten manifests from `process_deletes` plus
  carried-source manifests consumed by `MergeManifestProcess` bins.
