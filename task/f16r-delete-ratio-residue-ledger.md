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

# Evidence ledger — F-16r delete-ratio residue

Capability status lives on GAP_MATRIX row R135. This file holds the
reproduction, the mechanism, and the pins.

## Charter

```yaml
LEDGER:
  - id: C-001
    proposition: >
      On a v2 MOR table with one in-band data file and parquet position
      deletes covering 100% of its rows, RewriteDataFiles at defaults
      rewrites the file and removes the delete file. Surviving rows stay
      intact (here: zero live rows).
    verdict: PROVEN
    evidence: >
      test_fully_deleted_in_band_parquet_file_is_rewritten_and_drops_its_delete
      and test_fully_deleted_2500_row_in_band_parquet_file_ends_at_zero_delete_files
  - id: C-002
    proposition: >
      A partially-deleted below-threshold in-band parquet file is not
      selected (no over-eager rewrite).
    verdict: PROVEN
    evidence: test_default_ratio_under_threshold_parquet_is_a_noop (2/10 = 0.2)
  - id: C-003
    proposition: >
      The mechanism is in this fork, not the consuming engine call.
    verdict: PROVEN
    evidence: >
      Probe test_planner_selects_bounds_only_parquet_because_referenced_data_file_location_is_set
  - id: C-004
    proposition: >
      Result counts are truthful: rewritten_data_files_count and
      removed_delete_files_count.
    verdict: PROVEN
    evidence: >
      100% dead pin asserts rewritten=1, added=0, removed_delete_files_count=1
  - id: C-005
    proposition: >
      A partition-scoped two-path parquet delete survives when a sibling
      file-scoped 90% file is rewritten. The shared delete stays live and
      the sibling's deleted row stays deleted. removed_delete_files_count
      is 1.
    verdict: PROVEN
    evidence: test_partition_scoped_delete_survives_partial_rewrite
```

## Contradiction

Fork #232 (8fdf04a4, 2026-08-27) landed `delete_ratio_threshold` default 0.3
and wired `tooHighDeleteRatio`. The consuming engine at rev `d408da42` still
measured: one in-band data file, 100% position-deleted, survives the full
maintenance sequence. Spark 4.0.1 + Iceberg 1.10.0 ends the same sequence at
zero delete files with `remove-dangling-deletes` off.

Java site: `BinPackRewriteFilePlanner.tooHighDeleteRatio` —
`knownDeletedRecordCount / recordCount >= 0.3`, candidacy regardless of size.
`SizeBasedFileRewritePlanner.filterFileGroups` admits a single-file group when
any member matches `tooHighDeleteRatio` (no `size > 1` on that disjunct).
`ContentFileUtil.isFileScoped` is `referencedDataFile(f) != null`, and
`referencedDataFile` falls back to equal `file_path` lower/upper bounds.

## Probe (mechanism)

Fixture: v2 partitioned table, ten rows in one parquet data file, ten
position-delete rows naming that path. `PositionDeleteFileWriter` writes equal
`file_path` bounds. The scan-task delete keeps `referenced_data_file: None`.

Measured on the planned `FileScanTask` (no prints):

| Fact | Value |
|---|---|
| task.deletes.len() | 1 (index already routed the file-scoped delete) |
| task.deletes[0].referenced_data_file | None |
| task.deletes[0].record_count | Some(10) |
| task.record_count | Some(10) |
| DataFile.referenced_data_file() | None |
| referenced_data_file_location(&DataFile) | Some(data_path) |
| too_high_delete_ratio without the bounds set | false |
| is_candidate without the bounds set | false (file is in-band; delete_file_threshold is usize::MAX) |
| too_high_delete_ratio with file_scoped_delete_paths | true |
| group_qualifies on the one-file group | true |

Rejected hypotheses:

- Denominator wrong: `record_count` is Some(10).
- Delete FILES vs delete RECORDS: the delete's `record_count` is 10, but it
  was not counted because `is_file_scoped_scan_delete` required the raw
  `referenced_data_file` field.
- DV vs parquet: DVs already fired (#232 execute pins). The engine fixture is
  v2 parquet position deletes.
- Sequence-number filtering: the delete is attached to the task, so it is
  applicable.
- Group-level `min_input_files` / `size > 1`: `any_too_high_delete_ratio` has
  no `size > 1` guard. The group never formed because the file was not a
  candidate.
- Executor rewrite that keeps the delete: the planner selected nothing, so
  execute returned the empty result. After the planner fix the fork also
  drops the file-scoped parquet file (`plan_dv_removal`). That parquet drop
  is a fork extension: Java 1.10.0 `ManifestFilterManager.isDanglingDV` is
  DV-only. The composed dangling-delete action defaults off and its seq `<`
  clause would keep a same-seq delete.

A second fixture defect sat under the residue: the rewrite-test helper
`write_position_delete_file` uses default parquet `WriterProperties` and
default metrics, so it does not write Full equal `file_path` bounds.
`referenced_data_file_location` is then None even on a single-path delete.
The ratio pins use `write_file_scoped_position_delete`, which matches
production `position_delete_writer_properties` +
`MetricsConfig::for_position_delete` (the Spark FILE-granularity shape).
2500 rows on this writer is ~48 KiB, just under the 64 KiB min (0.75 ×
64 KiB). The pin grows row count until the file is inside that band so
the size clause cannot select it.

## Fix

1. `RewriteDataFiles::execute` loads live file-scoped delete paths via
   `referenced_data_file_location` into `ResolvedConfig.file_scoped_delete_paths`.
2. `too_high_delete_ratio` counts a scan-task delete when that set contains
   its path, matching Java `ContentFileUtil.isFileScoped`.
3. `plan_dv_removal` also removes non-DV file-scoped position deletes whose
   referenced data file this group rewrote. Java's apply path does not:
   `isDanglingDV` requires `ContentFileUtil.isDV`. The parquet drop is a
   fork extension. Java's Result DOES count apply-path DV drops
   (`RewriteFileGroup.asResult` from `danglingDVs.size()`); the named
   divergence is the extra parquet count, not the DV count.

The scan-task field stays the raw null. `interop_spark_mor_fixtures` pins
that Spark FILE-granularity deletes have `referenced_data_file: None`.

## Before / after

Before: `test_bounds_only_file_scoped_parquet_does_not_fire_ratio` expected
`RewriteDataFilesResult::default()`. The 100%-dead in-band file stayed live.

After: the 90% and 100% in-band parquet files are rewritten. The delete file
is gone. 20% parquet is a no-op. Absent-bounds two-path parquet and
unequal-bounds two-path parquet are still no-ops. A shared partition-scoped
delete survives a partial rewrite
(`test_partition_scoped_delete_survives_partial_rewrite`).

## Consuming engine

The engine pin `test_delete_laden_in_band_file_survives_the_runbook` is
written to go RED at the repin that takes this fix. Registry row RDF-1
retires with that pin. This unit does not touch the engine.

## Gates

- `make check` (docker legs N/A)
- `cargo test -p iceberg --locked`
- Docker is unavailable; `make test` docker legs are excused.
