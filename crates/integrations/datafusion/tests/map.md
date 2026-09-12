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

# map.md — crates/integrations/datafusion/tests/

## Purpose

Integration tests for `iceberg-datafusion`. They register an `IcebergTableProvider` against a
`MemoryCatalog` and run SQL (or the provider trait) through DataFusion.

## Contents

| File | What it pins |
|---|---|
| `commit_branch.rs` | `IcebergTableProvider::with_commit_branch` scopes scans and commits to the named branch (row R168). Diverged SELECT/DELETE/INSERT-SELECT follow the branch head; CoW/MoR UPDATE and CoW/MoR DELETE each have a branch-only (`id = 10`) pin so a main-read is red; missing-ref SELECT/DELETE error; INSERT VALUES still creates the ref; advertised schema stays current and field-id bind null-fills; default path stays on `main` |
| `integration_datafusion_test.rs` | Core provider SQL / scan / DML |
| `cow_memory_bound.rs` | Copy-on-write memory bound |
| `h7_p1_dml_prune.rs` | DML file prune |
| `interop_dv_sql.rs` / `interop_partitioned_dml.rs` | Interop DML |
| `interop_branch_dml.rs` | Branch read/commit Java interop (row R168 / PR-6A). Offline: Rust reproduces a Java diverged branch table; missing-ref SELECT/UPDATE; INSERT creates (ids, parent, main vs branch file sets); tag refuses; V3 MoR DELETE uses branch live files. GEN of 6 `rust_*` tables pins post-DML main vs branch file sets and writes `expected_*_files.txt`. Env `ICEBERG_INTEROP_BRANCH_DIR` / `_GEN_DIR` via `dev/java-interop/run-interop-branch-dml.sh` |
| `lazy_table_resolution_test.rs` | Catalog-backed lazy resolve |
| `partitioned_insert_select_test.rs` | Partitioned INSERT SELECT |
| `row_lineage_cow.rs` | V3 row lineage on CoW DML; Spark sequences at the fork's single-file layout (F-rp3-c7, row R166) |
| `row_lineage_mor.rs` | V3 merge-on-read UPDATE lineage, sequential/partitioned UPDATE, V2 control, commit-conflict |
| `interop_v3_upgrade_mor.rs` | GEN for `run-interop-v3-upgrade.sh` cell u3: the first V3 DML after Rust converts a Java parquet position delete to a deletion vector. Runs one merge-on-read `UPDATE`, asserts the replacement row keeps its original `_row_id` and that no parquet position delete is added, then lands the result table and the shared expectation document for the Java verify. |
| `interop_mor_update_lineage.rs` | GEN for `run-interop-mor-update-lineage.sh` (Java-created V3 tables; two MoR UPDATE statements + RePark COW UPDATE-then-DELETE) |
| `interop_mor_branch_lineage.rs` | V3 merge-on-read UPDATE lineage on a DIVERGED BRANCH (row R168 / PR-6B). Offline: Rust seeds `main` 1/2/3 + branch `b` 10/11, runs two MoR UPDATE statements of id 10 through `with_commit_branch("b")`, and pins stable `_row_id`, a sequence that advances on each UPDATE, unmatched branch rows unchanged, `main` snapshot / files / lineage untouched and `next-row-id` advancing by one added row per UPDATE (5 → 6 → 7). `ICEBERG_INTEROP_MOR_BRANCH_LINEAGE_DIR` adds the Direction-1 read of the Java fixture; `..._GEN_DIR` adds the Direction-2 GEN that writes `rust_after/` for Java. Both env vars are set by `dev/java-interop/run-interop-mor-branch-lineage.sh`. pins: R168/C-006 |
| `shared_puffin_dv/` | Shared-Puffin deletion-vector DML. `live.rs`/`extra.rs` are F-17 (T1–T23); F-19a re-aims concurrent sibling Replace/Delete to COMMIT (files-exist covers replacement blobs only). `container.rs` is F-18's Spark layout pin (touched blob moves, sibling entry unchanged, two containers, `removed-dvs`/`removed-delete-files`/`added-delete-files` = 1); `measure.rs` pins the rewrite amplification (a later single-row DELETE writes a ONE-blob container at 16 and 64 blobs) and carries the two `#[ignore]`d wall-clock/byte measurements |
| `fanout_insert_order.rs` | F-20: ten shuffled identity-int partitioned INSERT statements; the committed manifest data-file order is always ascending (row R115) |
| `interop_f18_dv_sibling_close.rs` | GEN for `run-interop-f18-dv-sibling-close.sh` (row R114 / F-18). Java `BaseDVFileWriter` writes the two-file seed and its two-blob delete; Rust runs the second DELETE and lands `before_dvs.json` / `after_dvs.json` / `summary.json` / `expected_rows.json` + `final.metadata.json` for the Java verify. Env `ICEBERG_INTEROP_F18_JAVA_SHARED`; a clean no-op when unset |
| `f21_legacy_delete_merge.rs` | F-21 V2 MoR parquet → V3 DV merge (row R114): two file-scoped deletes on one data file, partition-scoped keep, sequence skip, UPDATE, untouched file |
| `f21_legacy_delete_merge_measure.rs` | `#[ignore]` wall-clock: K=8 partition-scoped 100k positions; file-scoped 100k with a 200-byte `row` column. Not a CI pin |
| `interop_f21_legacy_delete_merge.rs` | GEN for `run-interop-f21-legacy-delete-merge.sh`: file-scoped merge plus partition-scoped coexistence (two data files, parquet stays live beside one DV) |
| `count_star_fold.rs` | F-27b (OFFLINE): `IcebergTableScan::partition_statistics` reports exact whole-table row counts (`total-records` + delete-free planned tasks) so DataFusion folds `count(*)` without a scan. Plain table: Exact(3), answer 3, physical plan has no `IcebergTableScan`; V3 MoR table with a DV: unknown, answer 2, plan still scans; residual filter / limit / per-partition query: unknown; empty table: Exact(0); COW DELETE (no delete files): Exact(2) |
| `parallel_small_scan.rs` | F-27d (OFFLINE): a sub-split-size table scans in parallel when the session allows it. 8 single-row-group files at `target_partitions=8` → 8 partitions with the 24-row set intact; at 1 → 1 partition; an empty projection stays 1 partition (the `count(*)` fold exemption); a LIMIT at N=8 is cleared on the scan node (stats stay Exact(24), every partition emits its full 3 rows) while SQL LIMIT still answers from the top (GlobalLimitExec owns the cap) |
| `insert_distribution.rs` | F-INSERT-DIST-1: optimized append keeps hash and clustered-order requirements across schema-preserving child replacement; identity/null and distinct bucket/day values from four source tasks reach one writer; zero-row, unpartitioned, and one-target controls stay intact; source failure commits no snapshot |
| `insert_compression.rs` | F-WRITE-COMPRESS-1: INSERT parquet footers honour `write.parquet.compression-codec` (default `zstd`; `snappy`/`gzip`/`uncompressed`) and `write.parquet.compression-level` for zstd; unknown codec fails naming the key and value |
| `rewrite_compression.rs` | F-WRITE-COMPRESS-2: `rewrite_data_files` output, CoW-DELETE rewritten data files, and MoR position-delete files all carry the table codec (default `zstd`) in every column chunk |
| `rewrite_size_shared/` | Shared machinery for the F-REWRITE-SIZE-1 measurement targets: the RePark-shaped bed (206 `INSERT INTO` files of `ts TIMESTAMP, grp STRING, id BIGINT`, `grp = g{batch % 20}`, `id`/`ts` globally monotonic), live-file footer inspection (`parquet::file::reader` per column chunk), the `WriterProperties` dump, and the `manual_rewrite` flip harness used to vary one parquet writer property at a time |
| `rewrite_size_probe.rs` | F-REWRITE-SIZE-1 step 1 measurement probe (`#[ignore]`d, run with `--ignored`): reproduces the RePark bed at `write.parquet.compression-level=3` (the Spark-side input), removes the property, runs the real `rewrite_data_files` at fork defaults, prints per-file/per-chunk footers, first-50-row order, the identical `WriterProperties` table for both paths, and the one-at-a-time flip table; asserts the bed ratio window and the ≥1.4× defect on the base tree |
| `rewrite_size_pin.rs` | F-REWRITE-SIZE-1 step 2 forward pin (`#[ignore]`d): same bed + real `rewrite_data_files`, asserts rewrite output ≤ 1.1× input compressed bytes. Red on the base tree (≈1.47); the step-2 fix makes it green |

## I want to...

| I want to... | go to |
|---|---|
| Pin `with_commit_branch` scan + commit | `commit_branch.rs` |
| Pin the Spark-equal DV container layout | `shared_puffin_dv/container.rs`, `interop_f18_dv_sibling_close.rs` |
| Pin fanout INSERT file order | `fanout_insert_order.rs` |
| Prove Java/Rust branch DML interop | `interop_branch_dml.rs` |
| Prove V3 MoR UPDATE lineage on a branch vs Java | `interop_mor_branch_lineage.rs` |
| Pin default (no branch) DML | `integration_datafusion_test.rs` |
| Pin INSERT parquet compression | `insert_compression.rs` |
| Pin rewrite/CoW/MoR parquet compression | `rewrite_compression.rs` |
| Measure why rewrite output is ~1.5× its input | `rewrite_size_probe.rs` (ledger: `task/f-rewrite-size-1-ledger.md`) |

## Pointers

- **Up:** [crates/integrations/datafusion/](..) · **Related:** [crates/iceberg/src/scan/map.md](../../../iceberg/src/scan/map.md), [crates/iceberg/src/transaction/map.md](../../../iceberg/src/transaction/map.md), [task/f6c-branch-following-reads-ledger.md](../../../../task/f6c-branch-following-reads-ledger.md)

## Debug

### Known failure modes

| Symptom | Likely cause |
|---|---|
| Named-branch SELECT returns `main` rows | `IcebergTableProvider::scan` passed `None` instead of `resolve_scan_snapshot_id` |
| Missing-ref DELETE creates the branch | the read leg did not call `resolve_scan_snapshot_id` (INSERT VALUES is the only create-on-missing path) |
| V3 MoR DELETE on a diverged branch fails "not a live file" | DV container close used `current_snapshot()`; see `close_touched_dv_containers_at` |
| MoR UPDATE on a diverged branch fails `not a live file of the current snapshot` | the DV container close resolved live data files from `current_snapshot()`; it must close at the scanned branch head (`close_touched_dv_containers_at`, PR-6A) |

### First checks

- `cargo test -p iceberg-datafusion --test commit_branch --locked`

### Escalate to

- [../../iceberg/src/scan/map.md#debug](../../../iceberg/src/scan/map.md#debug)
