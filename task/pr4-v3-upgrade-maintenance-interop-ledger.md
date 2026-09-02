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

# PR-4 ledger — V3 upgrade and maintenance interoperability

Model: claude-opus-5 (medium)

Plan of record: `task/iceberg-v3-production-work-plan-2026-09-01.md` section 4 PR-4, clause C-004
and the PR-4 part of C-007. Matrix rows R109, R114, R135, R136, R166.

This unit proves COMPOSED behavior over PR-2 and PR-3. It adds no product code. Every cell runs
the shipped actions end to end and compares the result against Apache Iceberg Java 1.10.0.

## Clauses

| Id | Proposition | Result |
|---|---|---|
| C-004 | A V2 to V3 upgrade and every required V3 maintenance action preserve live rows, delete semantics, row lineage, and snapshot validity across Java and Rust. | PROVEN for the four upgrade cells and the five maintenance actions below. Both directions are green; no cell relies on a skipped oracle. |
| C-007 (PR-4 slice) | `RewriteManifests` over ordinary clustering keeps every live file's row-id range, and `next_row_id` stays above every live assigned range. | PROVEN. Rust asserts the per-file range map is byte-identical across the clustering and that `next_row_id` moves 12 to 24 under the assignment rule below. Java re-reads the clustered table and asserts one data manifest, six assigned ranges, and every range end at or below `next_row_id` 24; Java then runs the SAME `clusterBy` itself over the same input table and lands on the same 24. |

## Java 1.10.0 decode

Jar: `~/.m2/repository/org/apache/iceberg/iceberg-core/1.10.0/iceberg-core-1.10.0.jar`
(`javap -c -p`, JDK 11 at `/usr/lib/jvm/java-11-openjdk-amd64`).

### `TableMetadata.upgradeToFormatVersion(int)` and `TableMetadata$Builder.upgradeFormatVersion(int)`

`upgradeToFormatVersion` is public on `TableMetadata`, so the oracle can drive the upgrade without
a catalog. The builder method's bytecode is short and complete:

- offsets 0-15: `Preconditions.checkArgument(newVersion <= 4, "Cannot upgrade table to unsupported format version: v%s (supported: v%s)")`.
- offsets 18-39: `Preconditions.checkArgument(newVersion >= formatVersion, "Cannot downgrade v%s table to v%s")`.
- offsets 42-51: equal version returns `this` unchanged.
- offsets 52-56: `putfield formatVersion`.
- offsets 57-74: appends `MetadataUpdate$UpgradeFormatVersion`.

The method never touches `nextRowId`. `TableMetadata$Builder` seeds `nextRowId` from the existing
metadata (`getfield TableMetadata.nextRowId` then `putfield`), so a V2 table upgraded to V3 keeps
`next_row_id` 0 and no file carries a `first_row_id`. Both implementations were measured to agree
on that: after the upgrade every `_row_id` is null and `next_row_id` is 0.

### `org.apache.iceberg.deletes.BaseDVFileWriter` (iceberg-core)

`BaseDVFileWriter(OutputFileFactory, Function<String, PositionDeleteIndex>)`,
`delete(String path, long pos, PartitionSpec, StructLike)`, `result()` returning
`DeleteWriteResult`. The 1.10.0 constructor takes an `OutputFileFactory`, so the Puffin name comes
from the table's location provider. This is the writer the Java half of upgrade cell u4 uses.

### `org.apache.iceberg.data.BaseDeleteLoader` (iceberg-data)

`loadPositionDeletes(Iterable<DeleteFile>, CharSequence)` returns the `PositionDeleteIndex` for one
data file, filtering the delete rows by path exactly as Java's scan does. The Java conversion feeds
that index into `BaseDVFileWriter.delete` through `PositionDeleteIndex.forEach(LongConsumer)`, so
the fixture never hard-codes the positions.

### `RewriteFiles.rewriteFiles(Set<DataFile>, Set<DeleteFile>, Set<DataFile>, Set<DeleteFile>)`

The only 1.10.0 core API that removes delete files and adds delete files in one commit.
`RowDelta.removeDeletes` is a `default` method on the interface, so the four-argument
`RewriteFiles` overload is the load-bearing path for a Java-side conversion.

## Measured upgrade semantics (both implementations agree)

A metadata-only V2 to V3 upgrade leaves `next_row_id` at 0 and every `_row_id` null. The FIRST V3
commit assigns ranges to every DATA manifest in the new list that still lacks one, in list order.
For a table with three carried V2 rows and a two-row append, both Java and Rust produce:

```text
next_row_id = 5
appended rows id=4,5   -> _row_id 0,1
carried  rows id=1,2,3 -> _row_id 2,3,4
snapshot sequence numbers [1, 2]
```

The appended manifest sorts first in the manifest list, so the new rows take the lowest ids. That
is a Java-observed fact, not a fork choice: cell u1 (Rust upgrades) and cell u2 (Java upgrades)
produce byte-identical expectation documents.

## Upgrade matrix — per-cell verdicts

| Cell | Producer | Upgrade writer | First V3 operation | Verdict |
|---|---|---|---|---|
| u1 | Java V2 | Rust | append | PASS. Java's production scan of the Rust result matches the Rust expectation on format version, `next_row_id`, snapshot sequence numbers and every `(id, val, _row_id)` triple. |
| u2 | Rust V2 | Java | append | PASS. Rust reads the Java-upgraded table and matches Java's own expectation document exactly, including the row-id assignment above. |
| u3 | Java V2 with a parquet position delete | Rust | `RewritePositionDeleteFiles` to a deletion vector, then a merge-on-read UPDATE | PASS. Rust reads the Java V2 table with the delete applied, upgrades, converts 1 parquet position delete to 1 Puffin DV, then a DataFusion `UPDATE` keeps the replacement row's original `_row_id`. Java re-reads the result: rows `{1:a, 3:X, 4:d, 5:e}`, one deletion vector, and no parquet position delete. |
| u4 | Rust V2 with a parquet position delete | Java | Java `BaseDeleteLoader` + `BaseDVFileWriter` conversion | PASS. Rust reads the Java-converted table, matches Java's expectation document, sees the same live id set as before the conversion, finds no live parquet position delete, and every assigned row id stays below `next_row_id`. |

ORC and Avro legacy position deletes stay outside the envelope (plan section 6).

## Maintenance matrix — per-action verdicts

Two Java-written partitioned V2 seeds (`identity(grp)`, ten rows across four data files, five rows
per partition; the second seed adds two parquet position deletes). Rust upgrades each to V3 and
appends one file per partition, so the table reaches twelve rows in six data files, six rows per
partition, and every live row carries a row id before any action runs. The partitions are
deliberately symmetric: `RewriteDataFiles` commits one snapshot per bin and the bin order is not
fixed, so unequal partitions would make the `next_row_id` pins order-dependent.

### The row-id assignment rule and the per-stage derivation

Plain Iceberg, identical in 1.10.0 and 1.11.0: every V3 DATA manifest a snapshot writes with
`first_row_id == null` is assigned `next-row-id` and advances it by `existing + added` rows; a
carried, already-assigned manifest advances it by 0. `SnapshotProducer.newManifestWriter` calls the
four-argument `ManifestFiles.write`, whose `newWriter` call passes `aconst_null` for `firstRowId`
(offset 4), so EVERY manifest an ordinary commit writes — including the ones `RewriteManifests`
clustering produces and the filtered copies a rewrite makes of the manifests it touched — arrives
unassigned. `ManifestFiles.copyRewriteManifest` is the only path that carries a source
`first_row_id` through, and `BaseRewriteManifests` uses it from `copyManifest`, i.e. the
`addManifest` external-manifest path (H-3), never from `clusterBy`.

Per stage, over the twelve-row fixture:

| Stage | Manifests the snapshot writes unassigned | Rows they carry | `next_row_id` |
|---|---|---|---|
| `m0` seed | three append manifests | 2 + 4 + 6 | 12 |
| `m1` `RewriteDataFiles`, current spec | first bin: its new manifest plus the filtered copies of every manifest it touched (all 12 live rows); second bin: its own new manifest only | 12, then 6 | 30 |
| `m2` `RewriteDataFiles`, evolved spec | one new manifest holding all 12 rows | 12 | 42 |
| `m3` `RewritePositionDeleteFiles` | none — only DELETE manifests are written, and data manifests are carried assigned | 0 | 12 |
| `m4` `RewriteManifests` | one clustered data manifest holding all 12 rows as `existing` | 12 | 24 |
| `m5` `ExpireSnapshots` | none — expiry writes no manifest | 0 | 24 |

Each stage asserts that absolute value, and also the rule as a per-stage identity:
`next_row_id == max over live DATA manifests of (first_row_id + existing + added)`, plus every live
file's `first_row_id + record_count <= next_row_id`.

| Action | Stage | Verdict |
|---|---|---|
| `RewriteDataFiles` on the current spec | `plain/m1` | PASS. 4 files rewritten, the live data-file path set changes, rows and every `_row_id` are unchanged, no delete file appears. |
| `RewriteDataFiles` after spec evolution (`identity(grp)` to `identity(y)`) | `plain/m2` | PASS. Every live data file carries the evolved spec id; Java confirms the current spec is a single `y` field and the output tuples are `{10, 20}`. Rows and row ids unchanged. |
| `RewritePositionDeleteFiles` converting legacy parquet deletes to DVs | `deletes/m3` | PASS. 2 parquet position deletes in, 2 Puffin deletion vectors out, 0 parquet position deletes left live, rows and row ids unchanged. |
| `RewriteManifests` with data and delete manifests | `deletes/m4` | PASS. Data manifests go from more than one to exactly 1; both delete manifests survive; the per-file `(first_row_id, record_count)` map is identical to `m3` and `next_row_id` moves 12 to 24 exactly as the rule predicts; Java confirms 6 assigned ranges all ending at or below `next_row_id` 24. |
| `ExpireSnapshots` after the rewrite sequence | `deletes/m5` | PASS. Snapshots 6 to 1, the current snapshot id is unchanged, rows and row ids unchanged. |

Java re-reads all seven stage tables through `IcebergGenerics` projected on
`MetadataColumns.schemaWithRowLineage` and asserts the SAME row and row-id document at each one, so
no stage can drift silently.

### Java and Rust agreement on the counter, measured not argued

`confirm-interop-v3-maintenance` makes Java run the two counter-moving actions ITSELF, over the
same input table and at the same layout, and compares its `next_row_id` with the value Rust wrote
to `next_row_ids.json`:

- `deletes/m4`: Java loads Rust's `m3` table and runs `rewriteManifests().clusterBy(f -> "all")`.
  Java 12 to 24, Rust 12 to 24, one clustered data manifest on both sides. AGREE.
- `plain/m1`: Java loads Rust's `m0` table and runs one `newRewrite().rewriteFiles(deleted, added,
  seq)` per partition, mirroring Rust's two bins. Java 12 to 30, Rust 12 to 30. AGREE.

So ordinary clustering DOES advance the counter, in both implementations, and the fork has no
divergence here. The runner fails closed if the two ever disagree.

## Test adequacy

- `run-interop-v3-upgrade.sh` asserts the Java fixture count `{"count":2}` and a total of 9
  `final.metadata.json` files, and hard-fails on a missing Maven, JDK, oracle output or target
  file. `run-interop-v3-maintenance.sh` asserts the same Java fixture count and 9
  `final.metadata.json` files.
- No action can pass as a no-op. Each cell asserts a shape that only the action produces: a changed
  data-file path set plus a non-zero rewritten count, an evolved spec id and tuple set, a parquet
  count of 2 falling to 0 with a Puffin count rising, a data-manifest count falling to 1, and a
  strictly smaller snapshot count.
- Sabotage, seven in total (two upgrade, five maintenance), each on a scratch copy, hard-fail if the
  corruption cannot be applied:

| Runner | Sabotage | Required failure line |
|---|---|---|
| upgrade | u3 result replaced by the pre-conversion V2 table | `parquet position delete survived the V3 conversion` |
| upgrade | u1 result truncated | `FAIL v3-upgrade u1: unexpected error` |
| maintenance | `deletes/m3` replaced by `deletes/m0` | `parquet position delete survived the V3 conversion` |
| maintenance | `plain/m1` replaced by `plain/m0` | `rewrite left the live data-file set unchanged` |
| maintenance | `plain/m2` replaced by `plain/m1` | `FAIL v3-maintenance plain/m2: current spec is` |
| maintenance | `deletes/m4` replaced by `deletes/m3` | `clustered data manifests` |
| maintenance | `plain/m0` truncated | `FAIL v3-maintenance plain: unexpected error` |

## Mutations, one knob at a time

Each mutation edits one production expression, runs the four offline pins, then restores the file.
The offline population is 4: the two `interop_v3_upgrade.rs` pins and the two
`interop_v3_maintenance.rs` pins.

| Mutation | Knob | Result |
|---|---|---|
| MUT-1 delete-conversion arm | `rewrite_position_delete_files.rs`: `format_version() >= FormatVersion::V3` becomes `>` , so a V3 table never enters the deletion-vector arm | 2 red out of 4 (`test_upgraded_v3_converts_legacy_parquet_deletes_and_keeps_no_parquet_delete`, `test_v3_delete_matrix_converts_and_reclusters_without_losing_lineage`) |
| MUT-2 manifest row-id inheritance | `spec/manifest/entry.rs` `assign_first_row_ids`: every entry's `first_row_id` is forced to `None` | 3 red out of 4 |
| MUT-3 computed row-id base | `arrow/record_batch_transformer.rs`: `ColumnSource::RowId { first_row_id }` becomes `first_row_id: 0`, so a computed `_row_id` ignores the file's assigned range | 3 red out of 4 |
| MUT-4 stored row-id preference | `arrow/record_batch_transformer.rs`: the `RowIdFromFile` arm falls back to the computed `RowId` arm, so a stored `_row_id` is ignored | 1 red out of 4 (`test_v3_rewrite_matrix_preserves_rows_and_lineage`) |
| MUT-5 rewrite lineage carry | `maintenance/rewrite_data_files_write.rs`: `carry_lineage` forced to `false`, so a V3 rewrite stops projecting and writing the lineage columns | 1 red out of 4 (`test_v3_rewrite_matrix_preserves_rows_and_lineage`) |

MUT-4 and MUT-5 are exercised only by the rewrite matrix, so a single red is the whole reachable
population for those two rules. Each mutation also reaches the interop cells, because the runners
drive the same production code; the artifact-level non-vacuity of the Java verify is pinned
separately by the six sabotages above.

Control: the unmutated tree is 0 red out of 4.

MUT-1 was also run against the two runners end to end, so the interop cells are pinned by the same
knob, not only the offline halves: `run-interop-v3-upgrade.sh` and `run-interop-v3-maintenance.sh`
both exited 101 with the mutation applied (`gen_rust_converts_java_v2_position_deletes_after_upgrade`
and `gen_rust_runs_the_v3_delete_matrix_over_the_java_seed` assert-failed), and the source restored
clean.

## Critic S3 remediation

- S3-1: the Java verify compared only rows and `_row_id`, so swapping `plain/m1` for `plain/m0` stayed
  green. `V3MaintenanceOracle.verifyRewriteChangedFiles` now compares the live data-file path sets of
  `plain/m0` and `plain/m1` and fails when they are equal or either is empty ("live data-file set
  changed, 6 to 2 file(s)" on the clean run). A fifth sabotage, `no-op-rewrite`, applies that swap:
  1 red out of 1, and the sabotaged verify reports exactly one failure, through the new line — which
  also measures that the swap produced zero failures before this rule existed.
- S3-2: each action stage now asserts the committing snapshot's `operation` in the Rust matrix. `m0`
  is `Append` (so a no-op action cannot satisfy the stages after it) and `m1`, `m2`, `m3`, `m4` and
  `m5` are all `Replace`; `m5` keeps `m4`'s snapshot, so its operation is the clustering commit's.
- S3-3: `run_delete_matrix` asserts `m3_ranges.len() == 6` before the `m3`-to-`m4` range comparison,
  so the per-file range check cannot pass over an empty map.

## Deviations and notes

- The merge-on-read half of upgrade cell u3 lives in
  `crates/integrations/datafusion/tests/interop_v3_upgrade_mor.rs`, not in
  `crates/iceberg/tests/interop_v3_upgrade.rs`. The DataFusion SQL surface is not reachable from
  the `iceberg` crate's dev-dependencies and this unit does not edit a dependency file. That file
  builds its expectation JSON by hand for the same reason: `serde_json` is not a dev-dependency of
  `iceberg-datafusion`.
- `scripts/run_interop_suites.sh` `SUITE_FLOOR_DEFAULT` moves 56 to 60, the real discovered count
  after the two new runners.
- No PR-1, PR-2 or PR-3 product code changed. One suspected PR-2 defect was investigated and
  REFUTED: after the mutation battery restored `rewrite_data_files_write.rs`, `cargo` skipped the
  rebuild because the restored file's mtime was older than the mutated build, so the stale test
  binary still carried `carry_lineage = false` and failed with `batch_columns: 3,
  expected_columns: 5`. Touching the restored file and rebuilding turned it green. A mutation
  harness must refresh the mtime and rebuild after every restore, or the next clean run reports a
  phantom defect.

## Gate exits

| Gate | Command | Exit |
|---|---|---|
| Lint and static gates | `make check` | 0 |
| Crate tests | `cargo test -p iceberg -p iceberg-datafusion --locked` | 0 |
| Upgrade interop | `dev/java-interop/run-interop-v3-upgrade.sh` | 0 |
| Crate tests, remediation re-run | `cargo test -p iceberg --test interop_v3_upgrade --test interop_v3_maintenance --locked` | 0 |
| Maintenance interop | `dev/java-interop/run-interop-v3-maintenance.sh` | 0 (7 steps, 5 sabotages, both Java confirmation legs agree) |
| Prose | `typos .` | 0 |
| Docker legs of `make test` | not run | EXCUSED — no Docker in this environment. The offline lib and integration suites plus both interop runners were run. |

Rust file-size ceilings did not move: the three new test files are 735, 731 and 308 lines, all under
the 1000-line default. `rust-file-size: 417 files clean (101 legacy ceilings)`.

## Section 9 delivery template

```text
Charter clauses: C-004; the PR-4 slice of C-007.
Matrix rows: R109, R114, R135, R136, R166.
Java methods or bytecode read: TableMetadata.upgradeToFormatVersion; TableMetadata$Builder.upgradeFormatVersion; TableMetadata$Builder nextRowId seeding; deletes.BaseDVFileWriter (ctor, delete, result); data.BaseDeleteLoader.loadPositionDeletes; deletes.PositionDeleteIndex.forEach; RewriteFiles.rewriteFiles 4-arg; RowDelta.removeDeletes default.
Files changed: crates/iceberg/tests/interop_v3_upgrade.rs (new); crates/iceberg/tests/interop_v3_maintenance.rs (new); crates/integrations/datafusion/tests/interop_v3_upgrade_mor.rs (new); dev/java-interop/run-interop-v3-upgrade.sh (new); dev/java-interop/run-interop-v3-maintenance.sh (new); dev/java-interop/src/main/java/org/apache/iceberg/InteropOracle.java; scripts/run_interop_suites.sh; docs/parity/GAP_MATRIX.md; task/pr4-v3-upgrade-maintenance-interop-ledger.md (new); task/todo.md; three map.md files.
Behavior before: no bidirectional interop fixture existed for the V2 to V3 bump, for a legacy parquet position delete converted on an upgraded table, or for the five V3 maintenance actions run in sequence over a Java-written seed.
Behavior after: unchanged product behavior. Four upgrade cells and five maintenance actions are proven against Java 1.10.0 in both directions, with row, format-version, snapshot-sequence, delete-file and row-lineage comparison at every stage.
Negative cases: seven sabotages (two upgrade, five maintenance) each pinned to a specific failure line; every runner hard-fails on a missing environment, oracle output or fixture count.
Test command and population: cargo test -p iceberg -p iceberg-datafusion --locked; dev/java-interop/run-interop-v3-upgrade.sh (9 fixtures); dev/java-interop/run-interop-v3-maintenance.sh (9 fixtures).
Mutations, one at a time: see the mutation table in this ledger.
Java interop command and fixture count: run-interop-v3-upgrade.sh, 9 final.metadata.json; run-interop-v3-maintenance.sh, 9 final.metadata.json plus confirm-interop-v3-maintenance; both assert the Java-side {"count":2}.
CI-only evidence gap: the Docker legs of `make test` were not run (no Docker in this environment); the offline lib and integration suites plus both interop runners were.
Breaking public API change: none. No product code changed.
Critic attestation: independent Critic PASS with three S3 pin-adequacy findings; all three landed in the remediation commit (see the "Critic S3 remediation" section).
Open findings and dispositions: none open. The suspected PR-2 rewrite-lineage defect was refuted as a cargo mtime fingerprint artifact of the mutation harness and is recorded above.
```
