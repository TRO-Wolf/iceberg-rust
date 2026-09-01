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

# PR-2 partition-safe RewriteDataFiles ledger

Plan of record: `task/iceberg-v3-production-work-plan-2026-09-01.md` (clause C-001, section 4 PR-2, section 11 owner ruling). Matrix: row R135.

## Proposition table

| Id | Proposition | Result |
|---|---|---|
| C-001 | `RewriteDataFiles` writes each output row under the partition tuple computed from the selected output spec. It never copies an old tuple into a different spec. | PROVEN |
| P-plan | Candidate grouping stays Java `groupByPartition`: current-spec partition else empty bucket. Output routing is a separate pass. | PROVEN |
| P-split | Output uses `RecordBatchPartitionSplitter` on the current schema and default spec. | PROVEN |
| P-bound | `max_open_partition_writers` defaults to 64. Zero is `DataInvalid` before any output data file is written. LRU eviction closes a writer, keeps its files, and may reopen the same key. | PROVEN |
| P-seq | Starting snapshot data sequence number is still stamped when `use_starting_sequence_number`. | UNCHANGED (existing tests) |
| P-dv | File-scoped delete removal and shared-Puffin sibling rewrite stay in the same atomic replace. | PROVEN (evolved-spec delete-class tests) |

## Java bytecode

Jar: `~/.m2/repository/org/apache/iceberg/iceberg-core/1.10.0/iceberg-core-1.10.0.jar` and `iceberg-api/1.10.0`.

| Class | Method | Decisive instructions |
|---|---|---|
| `BinPackRewriteFilePlanner` | `groupByPartition` offsets 52-87 | `DataFile.specId()` vs `Table.spec().specId()`; equal → `DataFile.partition()`, else empty `GenericRecord` of the current partition type. |
| `BinPackRewriteFilePlanner` | `newRewriteGroup` offset 38 | `invokevirtual outputSpecId:()I` passed into `RewriteFileGroup.<init>`. |
| `SizeBasedFileRewritePlanner` | `outputSpecId(Map)` offsets 1-15 | `ldc "output-spec-id"` then `PropertyUtil.propertyAsInt` defaulting to `table.spec().specId()`. |
| `RewriteDataFiles` (api) | constants | `OUTPUT_SPEC_ID` exists. No `max-open-partition-writers` option. The bound is fork-original because Spark's rewrite runner is not on iceberg-core. |

Spark `RewriteDataFiles` is not on this classpath. Java D2 interop therefore evolves the spec with `updateSpec` and rewrites with core `RewriteFiles` + `GenericAppenderFactory` under the new spec. That is the engine-agnostic write of current-spec tuples, not the Spark action.

## Test commands

- `cargo test -p iceberg --locked --lib rewrite_data_files` — 60 passed (includes new evolved-spec and bound tests plus prior rewrite suite).
- `cargo test -p iceberg --locked --test interop_evolved_spec_rewrite` — env-gated; offline no-op.
- Interop: `dev/java-interop/run-interop-evolved-spec-rewrite.sh`
- Docker `make test` legs excused (Docker unavailable).

## Gates (each run alone)

| Command | Exit |
|---|---|
| `make check` | 0 |
| `make check-matrix-anchors` (via `make check`) | 0 |
| `cargo test -p iceberg --locked --offline` | 0 |
| `cargo test -p iceberg --locked --lib rewrite_data_files` | 0 (60 passed) |
| `dev/java-interop/run-interop-evolved-spec-rewrite.sh` | 0 (5 `final.metadata.json`) |
| Docker `make test` legs | excused (Docker unavailable) |

## Mutations (one knob at a time)

Command unless noted: `cargo test -p iceberg --locked --lib rewrite_data_files` (population 60 after critic-fix pins). Restored from `.bak` + `touch` after each run.

| # | Knob | Result | Tests that went red |
|---|---|---|---|
| 1 | Restore static `group.first()` tuple as the output stamp | **11 red out of 59** | `source_field_identity_x_to_identity_y_rewrites_two_old_partitions` and 10 other evolved-spec / bound tests |
| 2 | Skip the partition splitter (`if true \|\| spec.fields().is_empty()`) | **37 red out of 59** | includes `unpartitioned_to_partitioned_fans_out_to_recomputed_keys` |
| 3 | Change default `64` to `63` | **1 red out of 59** | `default_max_open_partition_writers_is_64_and_peak_obeys_it` |
| 4 | Accept zero (`max_open==0` unbounded) | **1 red out of 1** on `cargo test -p iceberg --locked --lib zero_max_open_partition_writers_is_data_invalid` (execute committed 2 added files) | `zero_max_open_partition_writers_is_data_invalid_before_write` |
| 5 | Remove the eviction path | **1 red out of 1** on `high_cardinality_eviction_keeps_rows` | peak assertion |
| 6 | Drop accumulated output from an evicted writer | **1 red out of 1** on `high_cardinality_eviction_keeps_rows` | row census |
| 7 | Drop lineage projection | **1 red out of 1** on `v3_evolved_spec_rewrite_keeps_row_id` | V3 lineage |
| 8 | Disable MoR (`task.deletes` cleared) | **3 red out of 4** on `evolved_spec_rewrite_` | equality, position, and DV delete-class tests; V3 lineage stayed green |
| 9 | Move zero check to after `router.close()` so compacted parquet is written first | **1 red out of 1** on `zero_max_open_partition_writers_is_data_invalid_before_write` | parquet census (`compacted-*.parquet` appeared under `y=10` / `y=20`) |

## Interop

Command: `dev/java-interop/run-interop-evolved-spec-rewrite.sh`

Fixtures: 5 `final.metadata.json` (D1 Java table, D1 compacted, D2 Rust table, D2 Java rewritten, V3 compacted) plus `expected_row_ids.json`.

## PR body (section 9 template)

```text
Charter clauses: C-001 (and the C-004 / C-007 parts named on PR-2)
Matrix rows: row R135
Java methods or bytecode read: BinPackRewriteFilePlanner.groupByPartition (specId cmp + empty GenericRecord); newRewriteGroup outputSpecId(); SizeBasedFileRewritePlanner.outputSpecId(Map) ldc output-spec-id default table.spec().specId(); RewriteDataFiles.OUTPUT_SPEC_ID. Spark rewrite runner not on iceberg-core.
Files changed: maintenance/rewrite_data_files.rs, rewrite_data_files_plan.rs, rewrite_data_files_write.rs, rewrite_data_files_router.rs (new), evolved-spec and bound test modules, InteropOracle EvolvedSpecRewriteOracle, run-interop-evolved-spec-rewrite.sh, crates/iceberg/tests/interop_evolved_spec_rewrite.rs, GAP_MATRIX row R135, map.md files, this ledger.
Behavior before: write_compacted_files stamped group.first() under the current default spec (arity-only check). Same-arity identity(x)->identity(y) wrote the old x value as y. Partition-pruned scans returned the wrong rows.
Behavior after: live rows are split with RecordBatchPartitionSplitter on the current spec. Each output file is stamped with that spec id and the recomputed tuple. Open writers are bounded (default 64; zero rejected).
Negative cases: max_open_partition_writers=0 is DataInvalid and writes no output file; high-cardinality input keeps peak open writers at the bound; evolved-spec equality / position / DV deletes keep only live rows.
Test command and population: cargo test -p iceberg --locked --lib rewrite_data_files (60 passed this session among that filter).
Mutations, one at a time: see ledger Mutations.
Java interop command and fixture count: run-interop-evolved-spec-rewrite.sh ; 5 final.metadata.json.
CI-only evidence gap: Docker make test legs excused.
Breaking public API change: none. Additive builder max_open_partition_writers.
Critic attestation: pending independent Critic.
Open findings and dispositions: none from Actor.
```
