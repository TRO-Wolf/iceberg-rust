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

# PR-3 ledger — V3 row-DML lineage (MoR UPDATE carry + F-rp3-c7)

Plan of record: `task/iceberg-v3-production-work-plan-2026-09-01.md` (section 4 PR-3 as amended by section 11.2 / clause C-003). Matrix rows R114 and R166.

## Clauses

| Id | Proposition | Result |
|---|---|---|
| C-003 | Every V3 DataFusion UPDATE path keeps `_row_id`. An updated row advances `_last_updated_sequence_number`. | PROVEN. MoR UPDATE now projects both lineage columns, applies SET to user columns only, attaches the original `_row_id`, and writes null `_last_updated_sequence_number` so the reader resolves the new data file's sequence. COW UPDATE was already the reference (`row_lineage.rs`). |
| C-007 (PR-3 slice) | Rewritten rows that already carry a stored `_row_id` do not move `next-row-id`. | PROVEN as F-rp3-c7. A new V3 data manifest whose live files all have stored `_row_id` (parquet bounds, recovered after `FirstRowIdPolicy::Suppress`) sets `ManifestFile.first_row_id` so `ManifestListWriter$V3Writer` takes the already-assigned arm and does not add `existing_rows_count + added_rows_count`. |

## Java 1.10.0 decode

Jar: `~/.m2/repository/org/apache/iceberg/iceberg-core/1.10.0/iceberg-core-1.10.0.jar`.

### `ManifestListWriter$V3Writer.prepare`

- `content != DATA` OR `firstRowId != null`: `wrapper.wrap(manifest, null)` and return (keep the stored range; do not advance `nextRowId`).
- Else (`DATA` and `firstRowId == null`): `wrap(manifest, nextRowId)` then `nextRowId += existingRowsCount + addedRowsCount` (offsets: `ladd` / `ladd` after `existingRowsCount()` and `addedRowsCount()`).

This matches the fork's `assign_first_row_id` and confirms the plan's diagnosis: a new unassigned V3 data manifest advances the writer by every added and existing row.

### `ManifestFiles.write` / `ManifestWriter`

- `write(int, PartitionSpec, EncryptedOutputFile, Long snapshotId)` passes `aconst_null` as the writer's `firstRowId` (`newWriter` 5th argument).
- `toManifestFile` copies that constructor field onto the `ManifestFile`. New added-file manifests therefore hit the unassigned arm above.
- `copyRewriteManifest` is the path that passes a real `firstRowId` into `newWriter`.

### `MergingSnapshotProducer.add(DataFile)`

- Offset 73: `Delegates.suppressFirstRowId`. `Delegates$1.firstRowId` is `aconst_null; areturn`. Added files never carry a file-level `first_row_id` into the new manifest.

### `SnapshotProducer.apply`

- Builds `ManifestLists.write(..., table.nextRowId())`.
- `assignedRows = manifestListWriter.nextRowId() - table.nextRowId()` becomes the snapshot row range.

### Does Java contradict the plan?

No. Java iceberg-core 1.10.0 counts all added+existing rows of an unassigned data manifest. Stored `_row_id` lives in parquet and is invisible to that counter. Spark staying at 5 on the RePark recipe is the engine-side rewrite (stored `_row_id` in the replacement file) plus not consuming ids for those rows. The plan's rewrite-aware allocation is implemented by recovering the stored `_row_id` min from parquet metrics after Suppress and marking the manifest already-assigned, so the list writer takes Java's already-assigned arm. The increment formula for still-unassigned manifests (upgrade / plain append) is unchanged.

## Files

- `crates/integrations/datafusion/src/physical_plan/delete.rs` — MoR UPDATE projects lineage; uses `attach_update_lineage`.
- `crates/integrations/datafusion/src/physical_plan/row_lineage.rs` — `attach_update_lineage`, `cow_scan_stream`.
- `crates/iceberg/src/spec/manifest/entry.rs` — `stored_row_id_first`, `apply_rewrite_aware_first_row_ids`.
- `crates/iceberg/src/spec/manifest/writer.rs` — apply rewrite-aware range before serializing a V3 data manifest.
- Tests: `row_lineage_mor.rs`, sequential COW in `row_lineage_cow.rs`, shared-Puffin T2/T16.
- Interop: `dev/java-interop/run-interop-mor-update-lineage.sh`, `MorUpdateLineageOracle`, `tests/interop_mor_update_lineage.rs`.

## Tests

| Test | Result |
|---|---|
| One MoR UPDATE | updated row keeps `_row_id`; last-updated advances; unmatched keep both; `next-row-id` unchanged |
| Sequential MoR UPDATE | one `_row_id`; sequence advances twice |
| Partitioned MoR UPDATE | lineage correct across two partitions |
| Shared Puffin T2 | updating one row preserves sibling DV blobs and every live row's lineage |
| Commit conflict | frozen UPDATE after concurrent removal of the referenced data file refuses; no replacement DV |
| Sequential COW DELETE (F-rp3-c7) | overwrite then DELETE: `next-row-id` unchanged by the rewrite; survivors keep `_row_id` |
| Sequential COW UPDATE (F-rp3-c7) | overwrite then UPDATE: same, updated rows keep `_row_id`, sequence advances |
| V2 control | V2 still writes parquet position deletes; `_row_id` / last-updated are all-null; `next-row-id` is 0 |

## Mutations (`N red out of M`)

Exact command for 1–3: `cargo test -p iceberg-datafusion --locked --test row_lineage_mor -- --nocapture`

| # | Knob | Result |
|---|---|---|
| 1 | Remove `push_lineage_scan_columns` from MoR UPDATE | **3 red out of 5** (first, sequential, and partitioned UPDATE). Conflict and V2 stayed green. Restored + `touch`. |
| 2 | Attach a null `_row_id` array | **3 red out of 5** (row-id assertions). Restored + `touch`. |
| 3 | Preserve the old last-updated value on the modified row (`None => last_updated`) | **3 red out of 5** (sequence assertions). Restored + `touch`. |

Exact command for 4: `cargo test -p iceberg-datafusion --locked --test row_lineage_cow sequential_cow -- --nocapture`

| # | Knob | Result |
|---|---|---|
| 4 | Restore count-all-rows allocation (drop `apply_rewrite_aware_first_row_ids`) | **2 red out of 2** sequential COW tests. Restored + `touch`. Green re-run: 6/6 COW + 5/5 MoR. |

## Interop

Command: `bash dev/java-interop/run-interop-mor-update-lineage.sh`

- Fixture count: **2** (`mor_table`, `cow_table`), asserted in `fixture_count.json` and the runner.
- Java writes two 3-row V3 tables. Rust performs two MoR UPDATE statements on one and INSERT OVERWRITE then COW DELETE on the other. Java `IcebergGenerics.project(MetadataColumns.schemaWithRowLineage)` asserts stable ids, advancing sequences, and `next-row-id`.
- **Reverse MoR UPDATE:** iceberg-core 1.10.0 has no SQL / DataFusion-shaped merge-on-read UPDATE. That surface is Spark `SparkPositionDelta` / `SparkCopyOnWriteOperation`, not on this oracle's classpath. Reverse leg omitted.

Docker `make test` legs excused (no Docker).

## RePark

Liftable after this unit:

- `V3-COW-1` UPDATE half (already green on fork #243; this unit does not regress it).
- `V3-COW-1` sequential COW / next-row-id counter (F-rp3-c7). Recipe RePark must re-measure: 3-row V3 table → COW overwrite (`INSERT OVERWRITE` of the three rows) → COW DELETE of one row → `next-row-id` equals the post-overwrite value (Spark 5 vs pre-fix fork 6 on the original measurement; this unit holds the post-overwrite counter still).

Do not lift the guard until RePark re-measures that recipe on this SHA.

## Gates

Recorded at commit time in the ACTIVE todo section.

## Section 9 delivery template

```text
Charter clauses: C-003; C-007 PR-3 slice (F-rp3-c7)
Matrix rows: row R114, row R166
Java methods or bytecode read: ManifestListWriter$V3Writer.prepare (assign vs already-assigned; nextRowId += existingRowsCount + addedRowsCount); ManifestFiles.write aconst_null firstRowId; Delegates$1.firstRowId aconst_null; SnapshotProducer.apply assignedRows = writer.nextRowId - table.nextRowId
Files changed: physical_plan/delete.rs, physical_plan/row_lineage.rs, spec/manifest/entry.rs, spec/manifest/writer.rs, row_lineage_mor.rs, row_lineage_cow.rs, shared_puffin_dv live/extra, interop_mor_update_lineage.rs, run-interop-mor-update-lineage.sh, InteropOracle.MorUpdateLineageOracle, GAP_MATRIX rows R114 and R166, maps, this ledger
Behavior before: MoR UPDATE wrote replacement rows without _row_id / last-updated; COW rewrite of stored-_row_id files advanced next-row-id by every added row
Behavior after: MoR UPDATE keeps _row_id and nulls last-updated on the modified row; rewrite-aware manifests of fully stored-_row_id files do not move next-row-id
Negative cases: V2 still writes position deletes; concurrent removal of a referenced data file refuses the UPDATE; no replacement DV
Test command and population: cargo test -p iceberg --locked --lib spec::manifest::entry::first_row_id_tests; cargo test -p iceberg-datafusion --locked --test row_lineage_mor --test row_lineage_cow --test shared_puffin_dv; bash dev/java-interop/run-interop-mor-update-lineage.sh (2 fixtures)
Mutations, one at a time: (1) drop MoR lineage projections 3/5 red; (2) null attached row_id 3/5 red; (3) keep old last-updated 3/5 red; (4) drop rewrite-aware assignment 2/2 sequential COW red
Java interop command and fixture count: bash dev/java-interop/run-interop-mor-update-lineage.sh ; 2 fixtures; reverse MoR UPDATE not in iceberg-core 1.10.0
CI-only evidence gap: Docker make test legs excused
Breaking public API change: none
Critic attestation: pending independent Critic
Open findings and dispositions: none from Actor
```
