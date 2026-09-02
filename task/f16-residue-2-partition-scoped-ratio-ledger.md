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

# Evidence ledger — F-16 residue 2 (partition-scoped delete ratio, RePark RDF-1)

Model: claude-opus-5 (medium)

Capability status lives on GAP_MATRIX row R135. Verdict: **the RDF-1 premise is REFUTED.**
The fork already reclaims the MW-7 shape. No fix was written; inventing one would DIVERGE
from Java. Status: HALT pending a ruling.

## Charter

```yaml
LEDGER:
  - id: C-001
    proposition: >
      Decode which deletes Java's ratio clause counts, in iceberg-core 1.10.0 AND the
      Spark 4.1 runtime 1.11.0 the oracle runs, and find Spark's real reclaim route.
    verdict: PROVEN
    evidence: sections "Decode" and "Java measurement"
  - id: C-002
    proposition: >
      The MW-7 shape (v2 MOR, unpartitioned, one in-band data file, one PARTITION-scoped
      position delete covering 100% of its rows) is left live by the fork.
    verdict: REFUTED
    evidence: >
      test_mw7_unpartitioned_single_file_partition_scoped_full_bounds_is_reclaimed —
      rewritten=1, added=0, removed_delete_files_count=1, zero live deletes, zero rows.
      The red could not be reproduced at fork main ccaf527 nor, by code identity, at
      fb0cacfa (F-16r #248 6801659bf predates it).
  - id: C-003
    proposition: The fork needs a behavior change to match Spark on this shape.
    verdict: REFUTED
    evidence: >
      Java and the fork agree on BOTH bound shapes (full/equal -> reclaimed;
      truncated or absent -> no-op). Measured on both sides, with controls.
  - id: C-004
    proposition: The proven behavior is pinned and the pins are load-bearing.
    verdict: PROVEN
    evidence: two pins, five mutations, section "Mutations"
  - id: C-005
    proposition: No row resurrection — a partition-scoped delete still covering
      non-rewritten files survives.
    verdict: PROVEN (pre-existing, re-run green)
    evidence: test_partition_scoped_delete_survives_partial_rewrite (F-16r)
```

## Decode — `javap -c -p -v`, both jars

`~/.m2/.../iceberg-core/1.10.0/iceberg-core-1.10.0.jar` and
`/tmp/iceberg-spark-runtime-4.1_2.13-1.11.0.jar`.

| Site | Decisive instruction | 1.10.0 | 1.11.0 | Fork |
|---|---|---|---|---|
| `BinPackRewriteFilePlanner.tooHighDeleteRatio` filter | BSM `REF_invokeStatic ContentFileUtil.isFileScoped:(DeleteFile)Z` | same | same | `is_file_scoped_scan_delete` |
| ...mapper | BSM `REF_invokeInterface ContentFile.recordCount:()J` | same | same | `delete.record_count` |
| ...arithmetic | `min(sum, file().recordCount()) / file().recordCount() >= deleteRatioThreshold` | same | same | identical |
| `DELETE_RATIO_THRESHOLD_DEFAULT` | `ConstantValue: double 0.3d` | 0.3 | 0.3 | 0.3 |
| `DELETE_FILE_THRESHOLD_DEFAULT` | `ConstantValue: int 2147483647` | MAX | MAX | `usize::MAX` |
| `RewriteDataFiles.REMOVE_DANGLING_DELETES_DEFAULT` | `ConstantValue: int 0` | false | false | false |
| `ContentFileUtil.isFileScoped` | `referencedDataFile(f) != null` | same | same | same |
| `ContentFileUtil.referencedDataFile` | eq-deletes -> null; `referencedDataFile()` field; else `lowerBounds[PATH_ID].equals(upperBounds[PATH_ID])` -> decode; else null | same | same | `referenced_data_file_location` |
| `filterFiles` lambda | `outsideDesiredFileSizeRange \|\| tooManyDeletes \|\| tooHighDeleteRatio` | same | same | `is_candidate` |
| `filterFileGroups` lambda | `enoughInputFiles \|\| enoughContent \|\| tooMuchContent \|\| anyMatch(tooManyDeletes) \|\| anyMatch(tooHighDeleteRatio)` — no `size > 1` on either delete disjunct | same | same | `group_qualifies` |
| `MetricsConfig.forPositionDelete(Table)` | puts `DELETE_FILE_PATH -> MetricsModes$Full` and `DELETE_FILE_POS -> Full`; table column modes re-homed under a `row.` prefix by `lambda$forPositionDelete$0`, so user config CANNOT override | same | same | `MetricsConfig::for_position_delete` |
| `SparkFileWriterFactory` | calls `MetricsConfig.forPositionDelete()` / `(Table)` into `Parquet$DeleteWriteBuilder.metricsConfig` | — | present | every fork production site |
| `RewritePositionDeleteFilesSparkAction.execute` | DV early-return gated `TableUtil.formatVersion >= 3` (offset 66-70) | — | present | v2 unaffected |
| `SparkWriteConf.deleteGranularity` | option `delete-granularity`, table property `write.delete.granularity`, default `FILE` | — | present | n/a |

### Spark's reclaim route for MW-7

Not the DV path (v2), not `tooManyDeletes` (threshold MAX), not `remove-dangling-deletes` (off).
It is the ratio clause, reached through the **bounds** leg of `referencedDataFile`: Spark writes
position deletes with FULL untruncated `file_path` bounds, so a PARTITION-scoped delete that
happens to cover exactly ONE data file has `lower == upper`, is judged FILE-SCOPED, and its
3000/3000 = 1.0 ratio clears 0.3. The single-file group qualifies because the
`anyMatch(tooHighDeleteRatio)` disjunct carries no `size > 1` guard. The rewrite emits zero
output files and the delete falls in the same `RewriteFiles` commit.

## Java measurement — the control that decides it

Scratch probe (`/tmp`, no tree change) over the interop maven classpath, Iceberg 1.10.0,
`GenericAppenderFactory.newPosDeleteWriter`, 3000 positions naming ONE 139-char S3-style path.

| metrics mode | lower / upper `file_path` bound | equal | `ContentFileUtil.isFileScoped` | ratio clause |
|---|---|---|---|---|
| default (`truncate(16)`) | `s3://a-fairly-lo` / `s3://a-fairly-lp` | no | **false** | contributes 0 -> no-op |
| `full` (what `forPositionDelete` forces, and what `SparkFileWriterFactory` uses) | the whole 139-char path, twice | yes | **true** | 3000/3000 = 1.0 -> reclaimed |

`GenericAppenderFactory` is the generic/test writer and does NOT apply `forPositionDelete`;
`SparkFileWriterFactory` does. The oracle is Spark, so the oracle's deletes carry full bounds.

## Fork measurement — same shape, same verdict

| delete written with | `file_path` bounds | `RewriteDataFiles` result | live deletes after |
|---|---|---|---|
| `MetricsConfig::for_position_delete` (EVERY fork production site) | full, equal | rewritten=1, added=0, removed_delete_files=1 | 0 |
| default metrics (no fork production site emits this) | absent | all zeros, no-op | 1 |

Fork-vs-Java representational note, benign: given default metrics the fork DROPS the inexact
bound while Java STORES a truncated one. Both are not-file-scoped, so both no-op.

Every production position-delete writer was audited; all set `for_position_delete`:
`delete.rs` (the MOR DML path), `rewrite_position_delete_files.rs`, `rewrite_table_path.rs`,
`remove_dangling_delete_files.rs`, `convert_equality_delete_files.rs`. Only test helpers omit it.
`position_delete_writer_properties()` sets `set_statistics_truncate_length(None)`, so long S3
URIs keep exact bounds.

## Why the fix was not written

Attributing a partition-scoped delete's rows to target files by reading the delete's CONTENTS
would make the fork rewrite files Java leaves alone. Java uses bounds only — measured above.
That is a divergence, not parity, so it was not written. HALT per the brief.

## Pins

| Pin | Asserts |
|---|---|
| `test_mw7_unpartitioned_single_file_partition_scoped_full_bounds_is_reclaimed` | MW-7 exactly: unpartitioned v2, one in-band file, partition-scoped delete (`referenced_data_file` null) with EQUAL full bounds; rewritten=1, added=0, removed_delete_files=1, zero live deletes, zero live data files, zero rows |
| `test_mw7_unpartitioned_single_file_without_path_bounds_is_a_noop` | the same shape with no exact `file_path` bound is a no-op in BOTH engines |
| `test_partition_scoped_delete_survives_partial_rewrite` (F-16r, re-run) | no row resurrection: a shared partition-scoped delete survives a partial rewrite |

## Mutations, one at a time

`cargo test -p iceberg --lib test_mw7_`, each mutation applied alone to
`rewrite_data_files_plan.rs` and reverted.

| # | Mutation | full-bounds pin | no-op pin |
|---|---|---|---|
| M1 | `is_file_scoped_scan_delete` drops the bounds fallback | RED | green |
| M2b | ratio counts no deleted records (`.map(\|_\| 0)`) | RED | green |
| M3 | group gate requires `size > 1` on the ratio disjunct | RED | green |
| M5 | `is_candidate` drops the ratio clause | RED | green |
| M4 | every position delete judged file-scoped | green | RED |

`4 red out of 5` for the full-bounds pin, `1 red out of 5` for the no-op pin; each is green
under exactly the mutations it must not detect. M2 (threshold 1.5) was discarded: the builder's
range validation rejects it, so it proves nothing about the ratio.

## Consuming engine — RePark

The engine pin `test_delete_laden_in_band_file_survives_the_runbook` should go RED at the next
repin, and registry row RDF-1 retires with it. Before repinning, RePark must check the ONE fact
this ledger cannot check from here: whether the MW-7 delete file RePark wrote carries EQUAL
exact `file_path` bounds. If it does not, the gap is in whatever wrote it — Java's planner is
equally blind to that shape — and it is not a `RewriteDataFiles` parity gap.

## Gates

| Gate | Exit |
|---|---|
| `make check` | see PR body; Docker legs of `make test` excused (Docker unavailable) |
| `cargo test -p iceberg --locked` | 0 |
| `typos .` | 0 |
| `make check-matrix-anchors` | 0 |

No Java interop runner was added and `SUITE_FLOOR_DEFAULT` stays 63: the runner was specified as
evidence for a fix that this unit proves unnecessary. Ruling requested.

## 9. Delivery template

```text
Charter clauses: C-001 PROVEN, C-002 REFUTED, C-003 REFUTED, C-004 PROVEN, C-005 PROVEN
Matrix rows: R135
Java methods or bytecode read: BinPackRewriteFilePlanner.tooHighDeleteRatio / filterFiles /
  filterFileGroups / tooManyDeletes, SizeBasedFileRewritePlanner, ContentFileUtil.isFileScoped /
  referencedDataFile, MetricsConfig.forPositionDelete, SparkFileWriterFactory,
  RewritePositionDeleteFilesSparkAction.execute, SparkWriteConf.deleteGranularity
Files changed: crates/iceberg/src/maintenance/rewrite_data_files_ratio_tests.rs,
  docs/parity/GAP_MATRIX.md, task/todo.md, task/f16-residue-2-partition-scoped-ratio-ledger.md,
  crates/iceberg/src/maintenance/map.md, task/map.md
Behavior before: unchanged
Behavior after: unchanged — no production code was modified
Negative cases: no-op without exact path bounds; no row resurrection on a shared delete
Test command and population: cargo test -p iceberg --lib test_mw7_ (2 tests)
Mutations, one at a time: M1, M2b, M3, M5 red on the full-bounds pin; M4 red on the no-op pin
Java interop command and fixture count: none added — see Gates
CI-only evidence gap: Docker legs of make test
Breaking public API change: none
Critic attestation: pending independent Critic
Open findings and dispositions: RDF-1 premise refuted; ruling requested on the runner and floor
```
