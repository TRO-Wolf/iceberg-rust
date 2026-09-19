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

# Ledger — F-DANGLING-DV-COMMIT-1: every merging commit drops the DVs of the data files it removes, as Java's `removeDanglingDeletesFor`

**Ledger id:** `F-DANGLING-DV-COMMIT-1-2026-09-19`
**Branch:** `fix/f-dangling-dv-commit-1` (cut off fork `main` at `587d3592`)
**Scope:** one defect, one commit per brief step; one ledger for the lane
**Model:** swe-2-high

| Item | Commits | Subject |
|---|---|---|
| 1 | `2c8bb888` | `test: F-DANGLING-DV-COMMIT-1 — red DV-drop cells on merging commits` |
| 2 | `9bcf2375` | `fix: F-DANGLING-DV-COMMIT-1 — merging commits drop the deletion vectors of the data files they remove, as Java does` |
| 3 | — | (mutation proof — no commit) |
| 4 | this commit | `docs: F-DANGLING-DV-COMMIT-1 — ledger, mutation proof` |

## Defect

A merging commit that removes a data file carried that file's deletion vectors forward. Java
1.11.0 `MergingSnapshotProducer.apply` calls
`deleteFilterManager.removeDanglingDeletesFor(filterManager.filesToBeDeleted())`
(`core/src/main/java/org/apache/iceberg/MergingSnapshotProducer.java` L994-995) right after
filtering the data manifests, so every DV whose `referenced_data_file` names a removed data
file is tombstoned in the same commit. The fork only did this inside `rewrite_data_files`
(`maintenance/rewrite_data_files_dv.rs::plan_dv_removal`); every other merging commit kept
the dangling DV — a metadata divergence: the blob still references a file the snapshot no
longer holds, inflates `total-delete-files`, and leaks the Puffin to `RemoveDanglingDeleteFiles`
cleanup instead of dying with its data file.

## Java references (1.11.0)

| Line | What it does |
|---|---|
| `MergingSnapshotProducer.java` L990 | `deleteFilterManager.dropDeleteFilesOlderThan(minDataSequenceNumber)` — the sequence GC ported in F-RDF-COW-BYTES-1 round 4 |
| `MergingSnapshotProducer.java` L994-995 | `deleteFilterManager.removeDanglingDeletesFor(filterManager.filesToBeDeleted())` — this unit |
| `MergingSnapshotProducer.java` L1019 | `deleteFilterManager.buildSummary(filteredDeletes)` — dropped DVs feed the summary through `deletedFiles` |
| `ManifestFilterManager.java` L170-173 | `removeDanglingDeletesFor` stores `removedDataFilePaths` |
| `ManifestFilterManager.java` L437-447 | `canContainDroppedFiles`: a live delete manifest is opened when `removedDataFilePaths` is non-empty |
| `ManifestFilterManager.java` L493-495 | `isDanglingDV`: `ContentFileUtil.isDV(file) && removedDataFilePaths.contains(file.referencedDataFile())` — Puffin only; parquet position deletes are NOT dropped by this rule |
| `ManifestFilterManager.java` L513-549 | `drop`-family: the dangling DV becomes a `Deleted` entry and lands in `deletedFiles` → `buildSummary` |
| `SnapshotSummary.UpdateMetrics.deletedFile` | a removed DV increments `removed-dvs` (not `removed-position-delete-files`), `removed-delete-files`, `removed-position-deletes` by record count, `removed-files-size` by `contentSizeInBytes`; `total-delete-files`/`total-position-deletes` decrement accordingly |

## Operation set

The gate is the same `SnapshotProduceOperation::drops_old_delete_files` the sequence GC uses:
`true` for every Java `MergingSnapshotProducer` subclass — `DeleteFiles`, `OverwriteFiles`,
`RowDelta`, `ReplacePartitions`, `RewriteFiles`, `MergeAppend` — and `false` for the two Java
`SnapshotProducer` subclasses, `FastAppend` (`append.rs` L178) and `RewriteManifests`
(`rewrite_manifests.rs` L688). On top of that the drop fires only when
`removed_data_files` is non-empty, matching Java's `removedDataFilePaths.isEmpty()` gate —
a merge append or a row delta that only adds can never drop a DV.

## Implementation

`crates/iceberg/src/transaction/snapshot/manifest_filter.rs`, beside `DeleteFileExpiry`:

- `is_dangling_dv(file, removed_data_paths)` = `is_deletion_vector(file)` &&
  `removed_data_paths.contains(referenced_data_file_location(file))` — the same two helpers
  `rewrite_data_files_dv.rs` uses (`crate::delete_file_index`), so the explicit
  `referenced_data_file` field and the equal-`file_path`-bounds leg resolve identically.
- `process_deletes` builds `dangling_dv_paths: HashSet<String>` from `removed_data_files`
  when `drops_old_delete_files && !removed_data_files.is_empty()`, and passes it to
  `filter_manifest` for DELETE manifests only (DATA manifests get `None`).
- `filter_manifest` opens a live delete manifest when `dangling_dv_paths` is present
  (`has_added_files() || has_existing_files()` — Java `canContainDroppedFiles`), then rewrites
  it when any live entry is a dangling DV.
- `rewrite_manifest_with_deletes` tombstones a dangling DV per ENTRY — so a sibling blob in
  the same Puffin file survives (Java `DeleteFileSet` triple semantics), and `hits.expire`s it
  so the summary counts it exactly like a sequence-GC retirement.
- A parquet position delete can never satisfy `is_dangling_dv` (not Puffin); it is judged
  only by `DeleteFileExpiry`.

## Summary counters

Verified against Java `SnapshotSummary.deletedFile` semantics and asserted in the cells:

- Dropping one DV blob: `removed-dvs=1`, `removed-delete-files=1`,
  `removed-position-deletes=<record_count>`, `total-delete-files`/`total-position-deletes`
  decrement. A two-blob shared Puffin losing one blob ends at `total-delete-files=1`
  (delete-file counts are per-entry, matching Java).
- The rewrite-commit summary is rebuilt through the existing `expired_delete_files` →
  `removed_delete_files` path in `snapshot.rs::commit`, so `update_snapshot_summaries`
  derives the totals arithmetically.

## Cells (all assert the live row set is unchanged by the drop)

`crates/iceberg/src/maintenance/dangling_dv_commit_tests.rs` (V3, shared-Puffin fixture:
files `a`/`b`, one Puffin path, one blob per file):

| Cell | Expectation |
|---|---|
| `test_dangling_dv_delete_files_drops_dv_of_removed_data_file` | `delete_files` removes `a` → a's blob tombstoned |
| `test_dangling_dv_overwrite_files_drops_dv_of_removed_data_file` | `overwrite_files` swaps `a`→`a2` → a's blob dropped |
| `test_dangling_dv_replace_partitions_drops_dv_of_removed_data_file` | `replace_partitions` rewrites partition `x=0` → a's blob dropped |
| `test_dangling_dv_row_delta_drops_dv_of_removed_data_file` | `row_delta().remove_rows(a)` → a's blob dropped |
| `test_dangling_dv_rewrite_files_drops_dv_of_removed_data_file` | `rewrite_files([a],[a2])` → a's blob dropped |
| `test_dangling_dv_kept_when_removed_data_file_carries_no_dv` | removing DV-less file `c` keeps both blobs |
| `test_dangling_dv_fast_append_keeps_dvs` | fast append keeps both blobs (op-set exclusion) |
| `test_dangling_dv_merge_append_keeps_dvs` | merge append adds only, keeps both blobs |
| `test_dangling_dv_row_delta_adding_deletes_only_keeps_dvs` | row delta adding a DV only keeps all entries |
| `test_dangling_dv_parquet_position_delete_not_dropped_by_this_rule` | V2 file-scoped parquet delete on `a` survives `a`'s overwrite (sequence GC decides it) |

Every drop cell asserts the SURVIVING entry is exactly `b`'s blob — same Puffin path, same
`content_offset`/`content_size_in_bytes` — the sibling-blob control is structural, not a
separate test.

`crates/integrations/datafusion/src/physical_plan/dangling_dv_delete_tests.rs`:

| Cell | Expectation |
|---|---|
| `test_dangling_dv_cow_delete_drops_dv_of_removed_data_file` | v3 CoW table, DV planted on file 1 via `row_delta`; `DELETE WHERE id < 4` removes the whole file → `live_delete_files` empty, `removed-dvs=1`, `removed-delete-files=1`, `removed-position-deletes=1`, `total-delete-files=0`, ids `[4,5,6]` |

Pre-existing pin flipped (was asserting the divergence):
`remove_dangling_delete_files.rs::test_dangling_deletion_vector_removed_after_referenced_data_rewritten_away`
now asserts `rewrite_files` drops the DV itself (`removed-dvs=1` on the rewrite snapshot) and
`RemoveDanglingDeleteFiles` finds nothing (`removed_delete_files` empty, no snapshot).

## Red (before the fix)

```
cargo test -p iceberg --lib dangling_dv            → FAILED: 5 failed / 5 passed of 10
cargo test -p iceberg-datafusion --lib dangling_dv → FAILED: 1 failed / 0 passed of 1
```

Every drop cell failed with `left: 2` live delete entries where `1` was expected — both
shared-Puffin blobs carried forward. All five controls were already green (the controls pin
what must NOT change).

## Green (after the fix)

```
cargo test -p iceberg --lib dangling_dv            → 10 passed
cargo test -p iceberg-datafusion --lib dangling_dv →  1 passed
cargo test -p iceberg --lib transaction            → 691 passed, 1 ignored
cargo test -p iceberg --lib seq_gc                 →  12 passed
cargo test -p iceberg --lib rewrite_data_files     → 110 passed
cargo test -p iceberg --lib remove_dangling        →  24 passed
cargo test -p iceberg --lib cow_bytes              →   8 passed
cargo test -p iceberg --lib rewrite_position_delete→ 101 passed
cargo test -p iceberg-datafusion --lib delete      →  43 passed, 1 ignored
cargo test -p iceberg-datafusion --lib update      →  15 passed
```

## Mutation proof

Mutation: `git checkout 2c8bb888 -- crates/iceberg/src/transaction/snapshot/manifest_filter.rs`
(full revert of the fix, the smallest mutation covering the whole mechanism).

```
cargo test -p iceberg --lib dangling_dv            → FAILED: 5 failed / 5 passed of 10
cargo test -p iceberg-datafusion --lib dangling_dv → FAILED: 1 failed / 0 passed of 1
```

**6 red out of 11** — exactly the five drop cells plus the e2e COW DELETE cell; all five
controls stayed green under the mutation. `git checkout HEAD -- manifest_filter.rs` restored
the fix; both filters re-ran green (10/10 and 1/1).

## Round 2 — perf remediations (Grok P2s, no behaviour change)

| Item | Commit | Subject |
|---|---|---|
| perf | `4f2902d0` | `perf: F-DANGLING-DV-COMMIT-1 — borrowed path matching, bounded-concurrency manifest loads` |
| docs | this commit | `docs: F-DANGLING-DV-COMMIT-1 — round 2` |

- **R-01** (`manifest_filter.rs::is_dangling_dv`, `delete_file_index.rs` ~228): the common leg now
  compares `referenced_data_file_ref()` — a `&str` — directly against the removed-path set, so a
  live DV hashes a borrowed path with no `String` allocation; `referenced_data_file_location`
  (the bounds leg, which allocates) runs only when `referenced_data_file` is `None`. The
  equality-delete leg of the helper is preserved ahead of both legs.
- **R-03**: `dangling_dv_paths` is now `Option<&HashSet<&str>>` borrowing
  `RemovalTargets::data_paths` (new `data_paths()` accessor on `removal_targets.rs`) — the set
  `RemovalTargets::new` already builds from `removed_data_files`; the second owned
  `HashSet<String>` is gone.
- **R-04**: the three post-load entry scans (`has_removal` / `has_expired` / `has_dangling`)
  folded into one pass over live entries in `scan_manifest`; the `expiry` second-stage filter
  (`every_manifest || has_removal`) applies after the pass and gates `has_expired`, matching
  the round-1 predicate order exactly.
- **R-02** (`process_deletes` ~110-121, ~151-156): delete-manifest load+scan is split into a
  pure `scan_manifest` (free fn, `&FileIO` only) run over
  `stream::iter(...).buffer_unordered(DELETE_MANIFEST_SCAN_CONCURRENCY = 8)` — the
  `rewrite_data_files_dv.rs` pattern — while the rewrite stays sequential because it needs
  `&mut self` (`new_filtering_manifest_writer` → `manifest_counter`) and `&mut hits`. Results
  carry their source index into a pre-sized `Vec<Option<ManifestScan>>`, so the rewrite and the
  output manifest list stay in manifest-list order — deterministic output, matching Java's
  worker-pool-open/serial-apply shape. `filter_manifest` (data-manifest path) now delegates to
  the same `scan_manifest` + rewrite pair.
- **Mutation rerun** (round-1 mutation, full fix-file revert to `2c8bb888`): `dangling_dv`
  5 failed / 5 passed of 10 iceberg + 1 failed / 1 datafusion — **6 red of 11**, all controls
  green, identical signatures; restore → 10/10 + 1/1 green.
- **Gates**: `dangling_dv` 10 + 1, `transaction` 691 (1 ignored), `seq_gc` 12,
  `rewrite_data_files` 110, `remove_dangling` 24, `cow_bytes` 8, datafusion `delete` 43
  (1 ignored); `fmt --check`, clippy `-D warnings`, size checker, comment-ban `hits=0`.

## Residue / scope decisions

- `RemoveDanglingDeleteFiles` stays necessary for tables whose metadata ALREADY carries a
  dangling DV (written by older forks or other writers); its test now proves the action is a
  committed no-op on a clean table rather than the drop path.
- The `spec/snapshot_summary.rs` comment at `UpdateMetrics::remove_file` still says `removed-dvs`
  is "reachable only through direct collector use" — stale BEFORE this change (sequence GC and
  `removeDeletes` already reach it); left untouched, the comment ban forbids rewording and the
  line is outside the diff.
- `rewrite_data_files_dv.rs::plan_dv_removal` remains the planner-side mechanism that feeds
  `removed_delete_files_count` on `RewriteDataFiles` results; the commit-level drop covers the
  same entries through the explicit-removal triple first, so there is no double counting.
- GAP_MATRIX write-operation rows (R103/R104/R106/R107) carry no per-commit DV-drop note;
  status updates on the matrix are the orchestrator's pass, not this lane's.
- External `tests/shared_puffin_dv` cells that remove DV-bearing data files were audited
  statically: every assertion is on live ids, orphan file sets, or commit success — none pins
  DV-entry liveness after a data-file removal, so the new parity does not refute them.
