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
| C-007 (PR-3 slice) | Spark 4.1.2 + Iceberg 1.11.0 COW `next-row-id` on the 7-sequence table (not iceberg-core 1.10.0 write rules). | PROVEN. Bounds from parquet `_row_id` metrics only (not `DataFile.first_row_id`). First materialization (`Some(false)`: removed files exist, metrics incomplete) uses Java `nextRowId += existing+added`. Stored-source (`Some(true)`) advances by holes `(max-min+1)-count`. No removed files (`None`, MoR) consumes 0. Mixed added files advance only by rows that still need ids. 1.10.0 always `+= added` for unassigned DATA manifests (`ManifestFiles.write` passes null `firstRowId`). |

## Decode (Iceberg 1.11.0 in `/tmp/iceberg-spark-runtime-4.1_2.13-1.11.0.jar`)

`javap -c -p`. Sources also read from `apache-iceberg-1.11.0`.

| Class | Instruction | Meaning |
|---|---|---|
| `ManifestListWriter$V3Writer.prepare` | offset 9 `if_acmpne`; 18 `ifnull`; 52–71 `ladd`/`ladd`; 75 `putfield nextRowId` | DATA + `firstRowId==null` → `nextRowId += existing+added`. Already-assigned arm does not advance. Same formula in 1.10.0 and 1.11.0. |
| `ManifestFiles.write` | `aconst_null` into `newWriter` 5th arg | COW added manifests are unassigned. |
| `MergingSnapshotProducer.add(DataFile)` | offset 73 `Delegates.suppressFirstRowId` | File-level `firstRowId` cleared on add (only if non-null). |
| `Delegates.suppressFirstRowId` 1.11 | `if (file instanceof DataFile && file.firstRowId() != null)` wrap | No-op when `firstRowId` is already null. |
| `DataWriter.close` | `DataFiles.builder` without `withFirstRowId` | Spark does not stamp file `firstRowId` at write. |
| `ExtractRowLineage.apply` | `rowLineageRequired` from write schema; `meta==null` → empty lineage row | INSERT/OVERWRITE VALUES: no stored `_row_id`. COW DELETE/UPDATE: scan metadata joined onto rows. |
| `SparkWrite$CopyOnWriteOperation.commit` | `table.newOverwrite()` + `addFile` | COW DML is OverwriteFiles, not copyRewriteManifest. |

1.11 core write path does not encode holes. Spark-measured `next-row-id` is: Java `+= added` on first materialization (source parquet has no complete stored `_row_id`); later stored-source rewrites `+= (max-min+1)-count`.

## Spark 4.1.2 + Iceberg 1.11.0 oracle (INSERT 3 then …)

| Sequence | after step 1 | after step 2 |
|---|---|---|
| DELETE id=2 | (1,a,0,1),(3,c,2,1) next 5 | — |
| DELETE id=3 | (1,a,0,1),(2,b,1,1) next 5 | — |
| DELETE id=1 | (2,b,1,1),(3,c,2,1) next 5 | — |
| DELETE id=2, then DELETE id=1 | next 5 | (3,c,2,1) next 5 |
| UPDATE id=2 SET val='B', then DELETE id=1 | (1,a,0,1),(2,B,1,2),(3,c,2,1) next 6 | (2,B,1,2),(3,c,2,1) next 6 |
| UPDATE id=2 SET val='B', then DELETE id=2 | same, next 6 | (1,a,0,1),(3,c,2,1) next 7 |
| INSERT OVERWRITE VALUES then DELETE id=2 | (1,a,3,2),(2,b,4,2),(3,c,5,2) next 6 | (1,a,3,2),(3,c,5,2) next 8 |

## Files

- `crates/integrations/datafusion/src/physical_plan/delete.rs` — MoR UPDATE projects lineage; uses `attach_update_lineage`.
- `crates/integrations/datafusion/src/physical_plan/row_lineage.rs` — `attach_update_lineage`, `cow_scan_stream`.
- `crates/iceberg/src/spec/manifest/rewrite_aware.rs` — per-file stamp; holes vs first-materialization vs no-removed; mixed increment on `ManifestFile.unassigned_row_count`.
- `crates/iceberg/src/spec/manifest/writer.rs` — `source_has_stored_row_ids: Option<bool>` from removed files.
- `crates/iceberg/src/spec/manifest_list.rs` — increment uses `unassigned_row_count` when present.
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
| Spark 7-sequence COW table | `row_lineage_cow.rs` `spark_*` tests; all 7 cells green |
| Mixed ManifestWriter+ListWriter | `mixed_manifest_list_writer_advances_only_by_new_rows`: first_row_id 3, next 6, per-file [3, 0] |
| No-removed all-stored | `no_removed_files_all_stored_consumes_zero`: `None` source → `Some(0)` |
| V2 control | V2 still writes parquet position deletes; `_row_id` / last-updated are all-null; `next-row-id` is 0 |

## Mutations (`N red out of M`)

Exact command for 1–3: `cargo test -p iceberg-datafusion --locked --test row_lineage_mor -- --nocapture`

| # | Knob | Result |
|---|---|---|
| 1 | Remove `push_lineage_scan_columns` from MoR UPDATE | **3 red out of 5** (first, sequential, and partitioned UPDATE). Conflict and V2 stayed green. Restored + `touch`. |
| 2 | Attach a null `_row_id` array | **3 red out of 5** (row-id assertions). Restored + `touch`. |
| 3 | Preserve the old last-updated value on the modified row (`None => last_updated`) | **3 red out of 5** (sequence assertions). Restored + `touch`. |

Exact command for 4: `cargo test -p iceberg-datafusion --locked --test row_lineage_cow spark_ -- --nocapture`

| # | Knob | Result |
|---|---|---|
| 4 | Drop applying `apply_rewrite_aware_first_row_ids` (count-all-rows) | **3 red out of 7** Spark sequences (second-step stored-source rewrites: 6 vs 5, 8 vs 6, 8 vs 7). Restored + `touch`. |
| 5 | `unassigned_row_count` forced `None` | **1 red out of 1** mixed probe (`None` vs `Some(3)`). Restored. |

## Interop

Command: `bash dev/java-interop/run-interop-mor-update-lineage.sh`

- Fixture count: **2** (`mor_table`, `cow_table`), asserted in `fixture_count.json` and the runner.
- Java writes two 3-row V3 tables. Rust: two MoR UPDATE statements + Spark COW UPDATE id=2 then DELETE id=2. Java read asserts Spark numbers: MOR next 3; COW survivors `_row_id` {0,2}, last-updated 1, next 7. 1.10.0 harness cannot drive Spark DML; it only reads Rust output.
- **Reverse MoR UPDATE:** iceberg-core 1.10.0 has no SQL / DataFusion-shaped merge-on-read UPDATE. That surface is Spark `SparkPositionDelta` / `SparkCopyOnWriteOperation`, not on this oracle's classpath. Reverse leg omitted.

Docker `make test` legs excused (no Docker).

## RePark

Liftable after this unit:

- `V3-COW-1` UPDATE half (already green on fork #243; this unit does not regress it).
- `V3-COW-1` sequential COW / next-row-id (F-rp3-c7). Seven Spark sequences pinned.

Do not lift the guard until RePark re-measures that recipe on this SHA.

## Gates

| Command | Exit |
|---|---|
| `make check` | 0 |
| `cargo test -p iceberg -p iceberg-datafusion --locked` | 0 |
| `bash dev/java-interop/run-interop-mor-update-lineage.sh` | 0 (2 fixtures, 0 failures) |

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
