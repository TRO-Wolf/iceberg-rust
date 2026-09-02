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
| C-007 (PR-3 slice) | `next-row-id` follows the Java list-writer algorithm. Literal numbers compare only at matched physical layout. | PROVEN. Unassigned DATA: `nextRowId += existing+added`. Carried manifest with stored `first_row_id`: +0. TableMetadata `nextRowId` += listWriter.next − base. Survivors keep `_row_id` because rewritten parquet materialises the column. Same formula in 1.10.0 and 1.11.0. |

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
| `SnapshotProducer.apply` | `firstRowId=null` on added **and** filtered manifests | New snapshot writes unassigned DATA manifests. |

Layout: followup-2 Spark table used a two-file seed (rc=1+rc=2). Fork INSERT VALUES writes one 3-row file. `next-row-id` is a function of that layout, not a second allocation rule.

## Spark 4.1.2 + Iceberg 1.11.0, coalesce(1) single-file seed (fork layout)

| Sequence | after step 1 | after step 2 / 3 |
|---|---|---|
| DELETE any one of id=1,2,3 | survivors keep ids; next 5 | — |
| DELETE id=2, then DELETE id=1 | next 5 | (3,c,2,1) next 6 |
| UPDATE id=2 SET val='B', then DELETE id=1 | (1,a,0,1),(2,B,1,2),(3,c,2,1) next 6 | (2,B,1,2),(3,c,2,1) next 8 |
| UPDATE id=2 SET val='B', then DELETE id=2 | same, next 6 | (1,a,0,1),(3,c,2,1) next 8 |
| INSERT OVERWRITE VALUES then DELETE id=2 | (1,a,3,2),(2,b,4,2),(3,c,5,2) next 6 | (1,a,3,2),(3,c,5,2) next 8 |
| UPDATE id<=2 SET val='B', then DELETE id=3 | (1,B,0,2),(2,B,1,2),(3,c,2,1) next 6 | (1,B,0,2),(2,B,1,2) next 8 |
| DELETE id=2, INSERT (4,'d'), DELETE id=1 | next 5 | next 6; then (3,c,2,1),(4,d,5,3) next 7 |
| three single-row INSERT, DELETE id=2, DELETE id=1 | next 3 | next 3; then (3,c,2,3) next 3 |

## Files

- `crates/integrations/datafusion/src/physical_plan/delete.rs` — MoR UPDATE projects lineage; uses `attach_update_lineage`.
- `crates/integrations/datafusion/src/physical_plan/row_lineage.rs` — `attach_update_lineage`, `cow_scan_stream`.
- `crates/iceberg/src/spec/manifest_list.rs` — Java `+= existing+added` on unassigned DATA. No `unassigned_row_count`.
- `crates/iceberg/src/spec/manifest/writer.rs` — EXISTING/DELETED copy per-file `first_row_id`; manifest `first_row_id` stays null.
- Tests: `row_lineage_mor.rs`, sequential COW in `row_lineage_cow.rs`, shared-Puffin T2/T16.
- Interop: `dev/java-interop/run-interop-mor-update-lineage.sh`, `MorUpdateLineageOracle`, `tests/interop_mor_update_lineage.rs`.

## Tests

| Test | Result |
|---|---|
| One MoR UPDATE | updated row keeps `_row_id`; last-updated advances; unmatched keep both; `next-row-id` += added (Java) |
| Sequential MoR UPDATE | one `_row_id`; sequence advances twice; `next-row-id` += 1 per replacement |
| Partitioned MoR UPDATE | lineage correct across two partitions |
| Shared Puffin T2 | updating one row preserves sibling DV blobs and every live row's lineage |
| Commit conflict | frozen UPDATE after concurrent removal of the referenced data file refuses; no replacement DV |
| Spark single-file COW sequences | `row_lineage_cow.rs` `spark_*` (7 + OVERWRITE + 3 extra) |
| Filtered EXISTING/DELETED `first_row_id` | `filtered_manifest_copies_existing_and_deleted_first_row_id` |
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
| 4 | List-writer increment: skip `added_rows_count` (existing only) | **10 red out of 10** `spark_*`. Restored + `touch`. |

## Interop

Command: `bash dev/java-interop/run-interop-mor-update-lineage.sh`

- Fixture count: **2** (`mor_table`, `cow_table`), asserted in `fixture_count.json` and the runner.
- Java writes two 3-row V3 tables. Rust: two MoR UPDATE statements + COW UPDATE id=2 then DELETE id=2. Java production-scan reads the fork's own `next_row_id` back (MOR `_row_id`/`last_updated`; COW survivors `{0,2}` last-updated 1, next 8). No independent Java-written DML oracle. 1.10.0 cannot drive Spark DML.
- **Reverse MoR UPDATE:** iceberg-core 1.10.0 has no SQL / DataFusion-shaped merge-on-read UPDATE. That surface is Spark `SparkPositionDelta` / `SparkCopyOnWriteOperation`, not on this oracle's classpath. Reverse leg omitted.

Docker `make test` legs excused (no Docker).

## RePark

Liftable after this unit:

- `V3-COW-1` UPDATE half (already green on fork #243; this unit does not regress it).
- `V3-COW-1` sequential COW / next-row-id (F-rp3-c7) was a two-file Spark-seed layout artefact, not a defect. Re-measure `V3-COW-1` sequential-COW at matched layout.

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
Behavior after: MoR UPDATE keeps _row_id and nulls last-updated on the modified row; next-row-id is Java += existing+added on unassigned DATA at matched layout
Negative cases: V2 still writes position deletes; concurrent removal of a referenced data file refuses the UPDATE; no replacement DV
Test command and population: cargo test -p iceberg --locked --lib spec::manifest::entry::first_row_id_tests; cargo test -p iceberg-datafusion --locked --test row_lineage_mor --test row_lineage_cow --test shared_puffin_dv; bash dev/java-interop/run-interop-mor-update-lineage.sh (2 fixtures)
Mutations, one at a time: (1) drop MoR lineage projections 3/5 red; (2) null attached row_id 3/5 red; (3) keep old last-updated 3/5 red; (4) list-writer skip added_rows_count 10/10 spark_ red
Java interop command and fixture count: bash dev/java-interop/run-interop-mor-update-lineage.sh ; 2 fixtures; reverse MoR UPDATE not in iceberg-core 1.10.0
CI-only evidence gap: Docker make test legs excused
Breaking public API change: none
Critic attestation: pending independent Critic
Open findings and dispositions: none from Actor
```
