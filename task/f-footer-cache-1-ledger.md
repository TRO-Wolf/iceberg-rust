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

# F-FOOTER-CACHE-1 — a bounded shared parquet footer cache feeding the reader

Unit ledger. Scope: one public `ParquetFooterCache` (moka `future::Cache`, byte-bounded by
`ParquetMetaData::memory_size()`), scope-aware via `CacheScope`, wired through
`ArrowReaderBuilder` → `Table`/`TableScan`/`stream_partition_work` → the three catalog
builders (memory, S3 Tables, Glue) → both DataFusion execution paths. Default OFF.

Status: implemented. All pins green; mutation evidence and gates below.

## Request counts today (pre-change, debug profile, counting storage wrapper)

A counting storage wrapper (`RecordingStorageFactory` in `page_prune_fixture.rs`) records
every `(path, byte-range)` read. A footer read is identified as the tail read whose
`range.end == file_size_in_bytes` (the read that carries the `Parquet` magic + footer).

| Scan | Footer reads | Footer bytes |
|---|---|---|
| Cold scan, 4-file table, no cache (pin C-9) | 4 | per-file tail read |
| Warm re-scan, same reader options, no cache | +4 (8 cumulative) | repeated |
| Cold scan, 100-file table, no cache | 100 | 344,600 B (~3.4 KiB/footer) |
| Warm re-scan, 100-file table, no cache | +100 (200 cumulative) | +344,600 B |

Every scan re-reads every footer: the page-prune/selection work added in
F-PAGE-PRUNE-1 pays the full footer cost on every query because the only consumer of
`with_prefetched_parquet_metadata` was the maintenance rewrite path.

## Orchestrator rulings restated

### FC-1 — bounded shared cache with stats and dedup

`ParquetFooterCache` (`crates/iceberg/src/arrow/footer_cache.rs`) is a public,
`Arc`-shareable wrapper over `moka::future::Cache<FooterKey, CachedFooter>`:

- **Bounded by bytes**: `.weigher(footer_weight)` where
  `footer_weight = u32::try_from(entry.metadata.memory_size()).unwrap_or(u32::MAX)`;
  `max_capacity` set by `ParquetFooterCache::with_max_bytes(u64)`;
  `new()` defaults to 64 MiB.
- **Stats**: `hits`, `misses`, `fetches` (cold footer reads initiated), `upgrades`
  (page-index loads that replaced an index-less entry), `evictions` (listener counts
  `cause.was_evicted()`). Snapshot via `ParquetFooterCache::stats()`.
- **Concurrent-miss deduplication**: `try_get_with` coalesces same-key cold opens —
  one `get_metadata` call serves every waiter; the init error is propagated to all
  waiters (reconstructed per waiter, since moka wraps it in `Arc`) and is never cached.

Pins: C-1, C-2, C-6 (dedup + error propagation), C-7 (byte bound).

### FC-2 — key = (CacheScope, path, file_size_in_bytes)

`FooterKey { scope: CacheScope, path: Arc<str>, file_size_in_bytes: u64 }` — Hash+Eq.
Same path + size inside one scope: shared hit. Same path different size: distinct key,
miss (a stale/foreign rewrite never hits). Two `CacheScope`s never share an entry —
the scope travels in `TableFooterCache`, the per-table handle binding
`Arc<ParquetFooterCache>` to the catalog's `CacheScope` (same construction as the
metadata cache: `CacheScope::for_catalog` / `isolated` / explicit context).

Pins: C-4 (size in key), C-5 (scope separation).

### FC-3 — upgrade, not duplicate

`CachedFooter { metadata: Arc<ParquetMetaData>, index_checked: bool }`.
`index_checked` records that the entry already went through the index-loading path
(page + column + offset index per `ParquetReadOptions`). A task needing the index that
finds `index_checked == false` enters `entry_by_ref(&key).and_try_compute_with(...)` —
moka serializes same-key computes, the closure double-checks `index_checked` inside the
critical section, loads the page index onto a cloned metadata
(`ParquetMetaDataReader::new_with_metadata(...).load_page_index(&mut reader)`), and
`compute::Op::Put`s the replacement under the **same key** (`upgrades += 1`). Entry
count is unchanged; later index-needing and index-free tasks both hit the upgraded
entry.

Pins: C-3 (one upgrade, `upgrades == 1`, `entry_count` unchanged, second filtered scan
loads zero index bytes).

### FC-4 — no deep clone on the hot path

A hit returns the cached `Arc<ParquetMetaData>` directly (pointer-shared —
`Arc::ptr_eq` proven in C-4). The only `ParquetMetaData::clone` left is inside the
upgrade compute, because `ParquetMetaDataReader::load_page_index` mutates in place and
the cached `Arc` must not be mutated under other readers. The deep-clone-per-open that
`open_parquet_file` performed on every prefetched open is removed from the scan path:
when the cache handle is present, `open_parquet_file_sized` calls
`footer_cache.footer_or_fetch(...)` and hands the shared `Arc` to
`ArrowReaderMetadata::try_new`; the `prefetched_metadata` arm is retained only as a
`seed(...)` into the cache (one path, the cache is the feeder).

Pins: C-1/C-3 (warm scans and second filtered scans perform zero footer/index reads —
a clone-per-open could not produce that), C-4 (`Arc::ptr_eq` on hit).

### FC-5 — wiring, default OFF

- `ArrowReaderBuilder::with_footer_cache(TableFooterCache)`; `ArrowReader` carries it
  into `process_parquet_file_scan_task` → `open_parquet_file_cached`.
- `TableBuilder::footer_cache(...)` / `Table::footer_cache()`; `Table::reader_builder()`
  attaches it; `TableScanBuilder::build()` reads it via `assemble_scan` (the two
  duplicated `TableScan { … }` literals collapsed into one helper).
- `TableScan::configure_reader` and `TableScan::stream_partition_work` pass it; the
  free `stream_partition_work` gained a trailing `footer_cache: Option<TableFooterCache>`
  parameter (all three callers updated).
- Catalog builders: `MemoryCatalogBuilder`, `S3TablesCatalogBuilder`,
  `GlueCatalogBuilder` each accept `with_shared_footer_cache(Arc<ParquetFooterCache>)`;
  `with_cache_options` carries it into the catalog; each `table_builder()` binds
  `TableFooterCache::new(cache, catalog.cache_scope)` when present.
- DataFusion: `IcebergTableScan`/`TableProvider` scan path and the partitioned
  `stream_partition_work` path both source the handle from `self.table.footer_cache()`.
- Default OFF: `None` everywhere → `open_parquet_file_sized` falls through to today's
  prefetched/plain open arms — identical request counts (C-9).

Pins: FC-5 propagation tests (table → reader_builder, scan → configure_reader,
partition work both entry points, memory catalog attach), C-9 (off-by-default).

## Pins (red-first contract → test names)

| Clause | Pin | Test |
|---|---|---|
| C-1 | warm re-scan of N files: 0 footer reads | `c1_warm_rescan_reads_zero_footers` (N=4) |
| C-2 | two tasks of one file in one scan: 1 footer read | `c2_same_file_two_tasks_one_footer_read` |
| C-3 | unfiltered→filtered upgrade: 1 index load, `upgrades==1`, `entry_count` stable; second filtered scan loads nothing | `c3_filtered_scan_upgrades_entry_once` |
| C-4 | same path, different `file_size_in_bytes` → miss | `c4_same_path_different_size_misses` |
| C-5 | two scopes, same path+size → 2 reads, 2 entries | `c5_two_scopes_same_path_two_entries` |
| C-6 | 16 concurrent cold opens → 1 footer read; injected error reaches all 16, not cached | `c6_concurrent_cold_opens_one_footer_read` |
| C-7 | byte bound → evictions counted, `weighted_size ≤ bound`, rescan of evicted file refetches and returns identical rows | `c7_byte_bound_evicts_and_rescan_identical` |
| C-8 | cache ON ≡ cache OFF rows; filtered scans still read the page index | `c8_cache_on_matches_off_and_still_prunes` + full `arrow::page_prune_*` + `open_parquet_tests` suites |
| C-9 | no handle → today's counts | `c9_no_cache_keeps_todays_counts` |
| FC-5 | handle propagation, every seam | `fc5_table_reader_builder_carries_cache`, `fc5_table_scan_configure_reader_carries_cache`, `fc5_stream_partition_work_carries_cache`, `fc5_memory_catalog_attaches_footer_cache` |
| measure | 100-file before/after | `measure_100_file_footer_requests` |

Note on red-first ordering: the implementation and pins were developed in the same
uncommitted tree (single delegated round). The pins' red evidence is provided by the
mutation runs below, which break each mechanism and observe the corresponding pin fail.

## Measurement (debug profile; counts only — wall-clock is the RePark half's job)

100-file table, 512 rows/file, `data_file_concurrency = 8`:

| Leg | Footer reads | Footer bytes |
|---|---|---|
| Uncached cold | 100 | 344,600 |
| Uncached warm | +100 (200 cumulative) | +344,600 |
| Cached cold | 100 | 344,600 |
| Cached warm | +0 (100 cumulative) | +0 |

Source: `measure_100_file_footer_requests` eprintln, run 1 (this round).

## Mutation evidence

Each mutation was applied to `footer_cache.rs`, the named pin was run, the failure
observed, then the file was reverted (`git checkout`). After all four reverts the
suite re-ran: 14/14 green.

| Mutation | Pin | Observed failure |
|---|---|---|
| dropped `file_size_in_bytes` from `FooterKey` + `key()` | C-4 | `c4_same_path_different_size_misses` FAILED — `different size must miss: left 0, right 1` (wrong-size lookup hit the seeded entry, no fetch) |
| dropped `scope` from `FooterKey` + `key()` | C-5 | `c5_two_scopes_same_path_two_entries` FAILED — `scopes must not share: left 1, right 2` (second scope hit scope A's entry) |
| upgrade compute ran under a different key (`wrapping_add(1)` on size) — the "second entry instead of replace" bug | C-3 | `c3_filtered_scan_upgrades_entry_once` FAILED — `upgrades: left 0, right 1` (the compute's `None` arm cold-fetched and stored under the foreign key; the real entry stayed index-less) |
| replaced `try_get_with` with `get` + `insert` (no coalescing) | C-6 | `c6_concurrent_cold_opens_one_footer_read` FAILED — `fetches: left 16, right 1` (every waiter read the footer) |

Note on the C-3 mutation shape: moka `insert` itself replaces under the same key, so
"insert instead of replace" is only observable as a *different-key* insert — the
mutation models exactly that bug class (upgrade stored under a key the lookup never
revisits), and the pin catches it three ways (`upgrades`, `entry_count`, index reads).

## Implementation decisions

- `moka::future::Cache` (repo already uses moka 0.12 via `TableMetadataCache`): the
  async `try_get_with`/`entry_by_ref().and_try_compute_with` pair gives per-key
  coalescing and per-key serialized compute without holding any lock across `.await`.
- Errors are reconstructed per waiter via `shared_error` (kind + message + retryable)
  because moka hands back `Arc<E>`; no backtrace/source chain survives — acceptable:
  the failure path is a cold read, not the hot path.
- `OpenParquetError` gained `#[derive(Debug)]` (test `expect` ergonomics); it is
  `pub(crate)`, no API change.
- `FooterKey` is private; `CachedFooter` is private; only `ParquetFooterCache`,
  `ParquetFooterCacheStats`, `TableFooterCache` are public (plus
  `ArrowReaderBuilder::with_footer_cache`, `TableBuilder::footer_cache`,
  `Table::footer_cache`, `*CatalogBuilder::with_shared_footer_cache`).
- `len`/`weighted_size`/`run_pending_tasks` are `#[cfg(test)]`-gated `pub(crate)`
  helpers — moka `entry_count`/`weighted_size` lag pending maintenance, so the pins
  drain it first.
- `merge_ranges` moved `reader.rs` → `arrow/ranges.rs` (pure function extraction) to
  keep `reader.rs` under its file-size ceiling after the cache plumbing; the DataFusion
  `scan.rs` likewise shed `exact_table_row_count`/`resolve_bindings`/`project_bindings`
  to the new `physical_plan/scan_helpers.rs`. Moved code's comments were deleted per
  the comment ban.
- File-size ceilings lowered (shrunk files) in `scripts/check_rust_file_size.py`:
  `arrow/reader.rs` 10157→10140, `scan/mod.rs` 6878→6874.

## Gate results

PENDING — filled at final gate run.
