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

# F-INTEROP-NIGHTLY-GREEN-1 — the weekly Java-interop run back to 65 of 65

**Date:** 2026-10-17. **Base:** fork `main` `409ff330e`. **Model:** Claude Opus 5.5
(`claude-opus-5-5`), high effort.

The Nightly Interop workflow (`.github/workflows/nightly_interop.yml`, Saturdays 06:43 UTC) failed
every week from 2026-08-29. Run 39011689293 (2026-10-17) passed 61 of 65 suites. Four suites failed:

| Suite | Red since | Cause | PR |
|---|---|---|---|
| `run-interop-validate-append-only.sh` | 08-29 | test infrastructure: writer builder | A |
| `run-interop-pr5a-catalog-commit-decode.sh` | 09-05 | test infrastructure: missing jar | A |
| `run-interop-remove-dangling.sh` | 09-26 | see PR B | B |
| `run-interop-v3-maintenance.sh` | 09-19 | see PR C | C |

The unit ships as three pull requests. Each branches from the same `main`, so each one reviews
alone. This ledger carries one section per pull request.

This ledger retires when the three pull requests land and the weekly run reports 65 of 65.

## PR C — `fix/v3-maintenance-next-row-id-1`: the Java confirm mirrors the one-commit rewrite

**Outcome in one line.** The fork already assigns row ids exactly as Java 1.10.0 does. The two
sides disagreed because the Java confirm ran TWO rewrite commits (one per partition) and the fork
ran ONE. The brief's premise, "the SAME RewriteDataFiles on the SAME layout", did not hold. The
fix is in the oracle, not in the product.

### Step 0 — measurement

The suite on `origin/main` fails at step 5 exactly as the weekly run did:
`FAIL v3-maintenance confirm plain/m1: Java next_row_id 30 != Rust 24 (before 12)`. Every manifest
list below was read through `StaticTable` from the suite's own fixture
(`dev/java-interop/target/interop-v3-maintenance/`). Columns: `first_row_id`, added rows,
existing rows, deleted rows, added files, existing files, deleted files, in manifest-list order.

**Seed `plain/m0`** (Rust: V2 seed upgraded, then one append; `next-row-id` 12; snapshot
first-row-id 0, added-rows 12):

| # | first_row_id | added r | existing r | deleted r | added f | existing f | deleted f |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 2 | 0 | 0 | 2 | 0 | 0 |
| 1 | 2 | 5 | 0 | 0 | 2 | 0 | 0 |
| 2 | 7 | 5 | 0 | 0 | 2 | 0 | 0 |

**Rust `plain/m1`** (`RewriteDataFiles`, one commit, both partitions; `next-row-id` 24; snapshot
first-row-id 12, added-rows 12):

| # | first_row_id | added r | existing r | deleted r | added f | existing f | deleted f |
|---|---|---|---|---|---|---|---|
| 0 | 12 | 12 | 0 | 0 | 2 | 0 | 0 |
| 1 | 24 | 0 | 0 | 2 | 0 | 0 | 2 |
| 2 | 24 | 0 | 0 | 5 | 0 | 0 | 2 |
| 3 | 24 | 0 | 0 | 5 | 0 | 0 | 2 |

**Java `m1` as the old confirm built it** (one `newRewrite` per partition). After the first
commit (grp 1), `next-row-id` 24, snapshot first-row-id 12, added-rows 12:

| # | first_row_id | added r | existing r | deleted r | added f | existing f | deleted f |
|---|---|---|---|---|---|---|---|
| 0 | 12 | 6 | 0 | 0 | 1 | 0 | 0 |
| 1 | 18 | 0 | 1 | 1 | 0 | 1 | 1 |
| 2 | 19 | 0 | 3 | 2 | 0 | 1 | 1 |
| 3 | 22 | 0 | 2 | 3 | 0 | 1 | 1 |

After the second commit (grp 2), `next-row-id` 30, snapshot first-row-id 24, added-rows 6:

| # | first_row_id | added r | existing r | deleted r | added f | existing f | deleted f |
|---|---|---|---|---|---|---|---|
| 0 | 24 | 6 | 0 | 0 | 1 | 0 | 0 |
| 1 | 12 | 6 | 0 | 0 | 1 | 0 | 0 |
| 2 | 30 | 0 | 0 | 1 | 0 | 0 | 1 |
| 3 | 30 | 0 | 0 | 3 | 0 | 0 | 1 |
| 4 | 30 | 0 | 0 | 2 | 0 | 0 | 1 |

These are the five data manifests the weekly log printed: (24,0,6), (12,0,6), (30,0,0) ×3.

**Java `m1` with the new confirm** (one `newRewrite` over both partitions): `next-row-id` 24,
snapshot first-row-id 12, added-rows 12, and a manifest list identical, row for row, to Rust's
`plain/m1` above.

### Step 1 — the rule

Spec (format/spec.md at tag `apache-iceberg-1.10.0`, "First Row ID Assignment" and "Table
Metadata"): a writer preserves the `first_row_id` of an existing manifest and assigns one only to
a data manifest that has none. Each assignment advances by the rows that will inherit an id from
it; "a simple and valid approach" is `added_rows_count + existing_rows_count`. `next-row-id`
grows by the same sum (spec lines 833–837 and 961–963 of that file). Java 1.10.0 implements
exactly that approach. `ManifestListWriter$V3Writer.prepare` (`ManifestListWriter.java:157-167`)
leaves a delete manifest, or a data manifest whose `firstRowId()` is set, unchanged. Any other
data manifest gets `firstRowId = nextRowId`, and `nextRowId += existingRowsCount() +
addedRowsCount()`. `SnapshotProducer.apply` starts the writer at `base.nextRowId()`
(`SnapshotProducer.java:264-271`) and records `assignedRows = writer.nextRowId() -
base.nextRowId()` (`:291-296`). `TableMetadata.Builder.addSnapshot` then does `nextRowId +=
snapshot.addedRows()` (`TableMetadata.java:1264-1273`). Which manifests arrive without an id:
`BaseRewriteFiles` is a `MergingSnapshotProducer` (`BaseRewriteFiles.java:26`). Its `apply`
concatenates the new data manifests with the filtered existing ones (`MergingSnapshotProducer.java:922-956`).
A filtered manifest that loses an entry is rewritten by `ManifestFilterManager.filterManifestWithDeletedFiles`
(`ManifestFilterManager.java:478-550`). That method writes every surviving entry as EXISTING and
every removed one as DELETED through `newManifestWriter`, which is `ManifestFiles.write(formatVersion,
spec, file, snapshotId)` with a null `firstRowId` (`SnapshotProducer.java:569-572`,
`ManifestFiles.java:180-185`, `ManifestWriter.toManifestFile` at `ManifestWriter.java:193-215`).
An untouched manifest is carried with its stored id. **So Java advances `next-row-id`, per
commit, by the added plus existing rows of every data manifest written in that commit (new and
filtered), and by nothing for a carried manifest.** On this layout one commit writes 12 added
rows and three filtered manifests with 0 existing rows: 12 → 24. Two commits write 6 added rows
plus 6 existing rows (grp 2's three files, re-counted by the filtered manifests: 1 + 3 + 2), then
6 added rows plus 0: 12 → 24 → 30. The existing grp-2 rows take a fresh range in the first commit
and give it up in the second. That is why 12 became 30.

**The fork's seam**, unchanged: `ManifestListWriter::assign_first_row_id`
(`crates/iceberg/src/spec/manifest_list.rs`, the `(Some(_), None)` arm adds
`existing_rows_count + added_rows_count`; the `(Some(_), Some(_))` arm keeps a carried manifest),
`SnapshotProducer::commit` (`transaction/snapshot.rs`, `with_row_range(first_row_id,
writer_next_row_id - base next_row_id)`) and `TableMetadataBuilder::add_snapshot`
(`spec/table_metadata_builder.rs`, `next_row_id += added_rows`). Each is Java's rule line for
line, so step 2's "conform the fork" is empty.

**Why it went red on 09-19.** F-RDF-OPTIONS-1 (`d73b914a4`, 2026-09-17) made `RewriteDataFiles`
commit once by default, as Java's action does, and moved `M1_NEXT_ROW_ID` / `M2_NEXT_ROW_ID` to
24 / 36 on a measured PySpark 4.1.2 + Iceberg 1.11.0 oracle (ruling Q-19c-3,
`task/f-rdf-options-1-ledger.md` §9e: default 24 in one snapshot, partial progress 30 in two).
That ledger named the per-partition Java confirm as a P3 residue and did not re-run it. The next
weekly run did.

`assert_distinct_row_ids` and every live file range stay within `next_row_id` on both shapes; the
"Java violates the spec" branch does not apply.

### Step 2 — the change

- `InteropOracle.java` `V3MaintenanceOracle.confirmRewrite`: `compactOnePartition` now only
  writes the compacted file and collects the delete and add sets. One
  `newRewrite().rewriteFiles(toDelete, toAdd, sequenceNumber)` commits both partitions, the
  one-commit default of Java's `RewriteDataFiles`. The PASS line says "after one rewrite commit
  over both partitions".
- `interop_v3_maintenance.rs`: the constants are derived from the rule, not from a printed number.
  `SEED_DATA_FILE_ROWS = 12` (record counts of the six seed files; the deletes world's data files
  carry the same 12 records). The upgrade's first V3 commit numbers every data manifest:
  `SEED = 12`. M1 writes one added manifest of 12 rows and filtered manifests with 0 existing:
  `M1 = SEED + 12 = 24`. M2 does the same again: `M2 = M1 + 12 = 36`. M3 writes no data manifest:
  `M3 = SEED = 12`. M4 writes one clustered manifest that re-counts 12 existing rows:
  `M4 = M3 + 12 = 24`. The values are unchanged; the derivation is now in the code.
- `assert_row_id_accounting` states the rule. A data manifest the current snapshot numbered is one
  whose `first_row_id` is at or above the snapshot's `first-row-id` (a carried manifest's id is
  always below it). The snapshot's `added-rows` must equal the added plus existing rows of those
  manifests, and `next_row_id` must equal `first-row-id` plus that sum. The two earlier checks
  (highest manifest range end, every live file range at or below `next_row_id`) stay.
- New unit pin `rewrite_assigns_row_ids_to_new_and_filtered_manifests_and_keeps_carried_ranges`
  in `transaction/snapshot_first_row_id_tests.rs`, no Java. Two fast appends make two manifests,
  {first, second} (ids 0–5) and {third} (ids 6–8), `next-row-id` 9. One `rewrite_files` replaces
  `first`. Pinned manifest list: new `(9, added 3)`, carried `(6, added 3)`, filtered `(12,
  existing 3, deleted 3)`; snapshot row range `(9, 6)`; `next-row-id` 15. The filtered manifest
  with a surviving existing entry is the case the interop layout does not reach in one commit.

### Step 3 — evidence (2026-10-17)

- `scripts/run_interop_suites.sh --only run-interop-v3-maintenance.sh,run-interop-row-lineage.sh,
  run-interop-mor-update-lineage.sh,run-interop-mor-branch-lineage.sh,run-interop-v3-upgrade.sh`:
  `TOTAL: 5 passed, 0 failed, 5 run, 65 discovered (floor 65)`. v3-maintenance: verify `0
  failures`; confirm `deletes/m4: Java and Rust agree, next_row_id 12 to 24` and `plain/m1: Java
  and Rust agree, next_row_id 12 to 24 after one rewrite commit over both partitions`,
  `confirm-interop-v3-maintenance: 0 failures`; 9 fixtures; all five sabotages PASS.
- `cargo test -p iceberg --test interop_v3_maintenance` (offline): 4 passed.
- `cargo test -p iceberg --lib snapshot_first_row_id_tests`: 5 passed.
- Mutation A, `assign_first_row_id` advances by `added` only (existing term dropped):
  1 red out of 5 in `snapshot_first_row_id_tests` (the new pin); 1 red out of 4 in
  `interop_v3_maintenance` offline (`test_v3_delete_matrix_converts_and_reclusters_without_losing_lineage`,
  the clustering stage). Restored, green.
- Mutation B, every data manifest re-numbered (a carried manifest loses its id): 1 red out of 5
  in `snapshot_first_row_id_tests` (the new pin). Restored, green.
