// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use arrow_array::RecordBatch;
use futures::future::join_all;
use futures::{TryStreamExt, stream};

use super::footer_cache::{ParquetFooterCache, TableFooterCache};
use super::page_prune_fixture::*;
use super::reader::{ArrowFileReader, ArrowReaderBuilder, ParquetReadOptions};
use crate::catalog::CacheScope;
use crate::catalog::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use crate::expr::Reference;
use crate::io::{FileIO, FileMetadata};
use crate::scan::{
    CombinedScanTask, FileScanTask, FileScanTaskStream, ScanFilterMode, assign_partition_work,
    stream_partition_work,
};
use crate::spec::{
    Datum, FormatVersion, NestedField, PrimitiveType, Schema, SortOrder, TableMetadataBuilder,
    Type, UnboundPartitionSpec,
};
use crate::table::Table;
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableIdent};

fn footer_reads(ranges: &Mutex<Vec<(String, Range<u64>)>>, path: &str, file_size: u64) -> usize {
    ranges
        .lock()
        .expect("read ranges")
        .iter()
        .filter(|(p, r)| p == path && r.end == file_size)
        .count()
}

fn footer_bytes(ranges: &Mutex<Vec<(String, Range<u64>)>>, path: &str, file_size: u64) -> u64 {
    ranges
        .lock()
        .expect("read ranges")
        .iter()
        .filter(|(p, r)| p == path && r.end == file_size)
        .map(|(_, r)| r.end - r.start)
        .sum()
}

fn index_reads(
    ranges: &Mutex<Vec<(String, Range<u64>)>>,
    path: &str,
    index_ranges: &[Range<u64>],
) -> usize {
    ranges
        .lock()
        .expect("read ranges")
        .iter()
        .filter(|(p, read)| {
            p == path
                && index_ranges
                    .iter()
                    .any(|r| read.start < r.end && r.start < read.end)
        })
        .count()
}

fn opts_unfiltered() -> ParquetReadOptions {
    let mut options = ParquetReadOptions::builder().build();
    options.preload_column_index = false;
    options.preload_offset_index = false;
    options.preload_page_index = false;
    options
}

fn opts_filtered() -> ParquetReadOptions {
    let mut options = ParquetReadOptions::builder().build();
    options.preload_column_index = true;
    options.preload_offset_index = true;
    options.preload_page_index = true;
    options
}

async fn file_reader(
    io: &FileIO,
    path: &str,
    size: u64,
    options: ParquetReadOptions,
) -> ArrowFileReader {
    let input = io.new_input(path).expect("input");
    let reader = input.reader().await.expect("reader");
    ArrowFileReader::new(FileMetadata { size }, reader).with_parquet_read_options(options)
}

async fn read_tasks(
    tasks: Vec<FileScanTask>,
    file_io: FileIO,
    footer_cache: Option<TableFooterCache>,
    concurrency: usize,
    row_selection: bool,
) -> crate::Result<Vec<RecordBatch>> {
    let mut builder = ArrowReaderBuilder::new(file_io)
        .with_batch_size(37)
        .with_row_group_filtering_enabled(true)
        .with_row_selection_enabled(row_selection)
        .with_data_file_concurrency_limit(concurrency.max(1));
    if let Some(cache) = footer_cache {
        builder = builder.with_footer_cache(cache);
    }
    builder
        .build()
        .read(Box::pin(stream::iter(tasks.into_iter().map(Ok))) as FileScanTaskStream)?
        .try_collect::<Vec<RecordBatch>>()
        .await
}

fn id_tasks(paths: &[String], schema: &Arc<Schema>) -> Vec<FileScanTask> {
    paths
        .iter()
        .map(|p| task(p, schema.clone(), &[1], None))
        .collect()
}

#[tokio::test]
async fn c1_warm_rescan_reads_zero_footers() {
    let tmp = tmpdir();
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("c1"));
    let schema = id_schema();
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let mut paths = Vec::new();
    for i in 0..4 {
        let p = path(&tmp, &format!("c1-{i}.parquet"));
        write_id_pages(&p, &ids);
        paths.push(p);
    }
    let tasks = id_tasks(&paths, &schema);
    let sizes: Vec<u64> = tasks.iter().map(|t| t.file_size_in_bytes).collect();
    read_tasks(tasks.clone(), io.clone(), Some(handle.clone()), 4, true)
        .await
        .expect("cold scan");
    for (p, size) in paths.iter().zip(&sizes) {
        assert_eq!(footer_reads(&ranges, p, *size), 1, "cold scan: {p}");
    }
    read_tasks(tasks, io, Some(handle), 4, true)
        .await
        .expect("warm scan");
    for (p, size) in paths.iter().zip(&sizes) {
        assert_eq!(footer_reads(&ranges, p, *size), 1, "warm scan: {p}");
    }
    assert_eq!(cache.stats().fetches, 4);
}

#[tokio::test]
async fn c2_same_file_two_tasks_one_footer_read() {
    let tmp = tmpdir();
    let p = path(&tmp, "c2.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("c2"));
    let schema = id_schema();
    let t1 = task(&p, schema.clone(), &[1], None);
    let t2 = task(&p, schema, &[1], None);
    read_tasks(vec![t1, t2], io, Some(handle), 2, true)
        .await
        .expect("scan");
    assert_eq!(footer_reads(&ranges, &p, size), 1);
    assert_eq!(cache.stats().fetches, 1);
}

#[tokio::test]
async fn c3_filtered_scan_upgrades_entry_once() {
    let tmp = tmpdir();
    let p = path(&tmp, "c3.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let metadata = file_metadata(&p);
    let index_ranges = index_byte_ranges(&metadata);
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("c3"));
    let schema = id_schema();
    read_tasks(
        vec![task(&p, schema.clone(), &[1], None)],
        io.clone(),
        Some(handle.clone()),
        1,
        true,
    )
    .await
    .expect("unfiltered");
    assert_eq!(cache.stats().fetches, 1);
    assert_eq!(cache.stats().upgrades, 0);
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 1);
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let filtered = task(&p, schema.clone(), &[1], Some(predicate));
    read_tasks(
        vec![filtered.clone()],
        io.clone(),
        Some(handle.clone()),
        1,
        true,
    )
    .await
    .expect("filtered");
    assert_eq!(cache.stats().upgrades, 1);
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 1, "upgrade must not create a second entry");
    let index_reads_after_upgrade = index_reads(&ranges, &p, &index_ranges);
    assert!(index_reads_after_upgrade >= 1);
    read_tasks(vec![filtered], io, Some(handle), 1, true)
        .await
        .expect("filtered again");
    assert_eq!(cache.stats().fetches, 1);
    assert_eq!(cache.stats().upgrades, 1);
    assert_eq!(
        index_reads(&ranges, &p, &index_ranges),
        index_reads_after_upgrade,
        "a second filtered scan loads no index bytes"
    );
}

#[tokio::test]
async fn c4_same_path_different_size_misses() {
    let tmp = tmpdir();
    let p = path(&tmp, "c4.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let seeded = file_metadata(&p);
    let path_arc: Arc<str> = Arc::from(p.as_str());
    let (io, _opens, _ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("c4"));
    handle.seed(&path_arc, size, seeded.clone()).await;
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 1);
    let mut wrong_size_reader =
        file_reader(&io, &p, size + 16 * 1024 * 1024, opts_unfiltered()).await;
    let _ = handle
        .footer_or_fetch(
            &path_arc,
            size + 16 * 1024 * 1024,
            opts_unfiltered(),
            &mut wrong_size_reader,
        )
        .await;
    assert_eq!(cache.stats().fetches, 1, "different size must miss");
    let mut reader = file_reader(&io, &p, size, opts_unfiltered()).await;
    let hit = handle
        .footer_or_fetch(&path_arc, size, opts_unfiltered(), &mut reader)
        .await
        .expect("seeded entry must hit");
    assert_eq!(cache.stats().fetches, 1, "same size must hit");
    assert!(
        Arc::ptr_eq(hit.metadata(), &seeded),
        "a hit serves the stored Arc"
    );
}

#[tokio::test]
async fn c5_two_scopes_same_path_two_entries() {
    let tmp = tmpdir();
    let p = path(&tmp, "c5.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let (io, _opens, _ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let scope_a = TableFooterCache::new(cache.clone(), CacheScope::new("catalog:one", "creds-a"));
    let scope_b = TableFooterCache::new(cache.clone(), CacheScope::new("catalog:one", "creds-b"));
    let path_arc: Arc<str> = Arc::from(p.as_str());
    let mut r1 = file_reader(&io, &p, size, opts_unfiltered()).await;
    scope_a
        .footer_or_fetch(&path_arc, size, opts_unfiltered(), &mut r1)
        .await
        .expect("scope a");
    let mut r2 = file_reader(&io, &p, size, opts_unfiltered()).await;
    scope_b
        .footer_or_fetch(&path_arc, size, opts_unfiltered(), &mut r2)
        .await
        .expect("scope b");
    assert_eq!(cache.stats().fetches, 2, "scopes must not share");
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 2);
}

#[tokio::test]
async fn c6_concurrent_cold_opens_one_footer_read() {
    let tmp = tmpdir();
    let p = path(&tmp, "c6.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("c6"));
    let schema = id_schema();
    let tasks: Vec<FileScanTask> = (0..16)
        .map(|_| task(&p, schema.clone(), &[1], None))
        .collect();
    read_tasks(tasks, io.clone(), Some(handle), 16, true)
        .await
        .expect("scan");
    assert_eq!(footer_reads(&ranges, &p, size), 1);
    assert_eq!(cache.stats().fetches, 1);

    let err_cache = Arc::new(ParquetFooterCache::new());
    let err_handle = TableFooterCache::new(err_cache.clone(), CacheScope::isolated("c6-err"));
    let bogus = size + 16 * 1024 * 1024;
    let err_path: Arc<str> = Arc::from(p.as_str());
    let mut readers = Vec::new();
    for _ in 0..16 {
        readers.push(file_reader(&io, &p, bogus, opts_unfiltered()).await);
    }
    let results = join_all(
        readers
            .iter_mut()
            .map(|r| err_handle.footer_or_fetch(&err_path, bogus, opts_unfiltered(), r)),
    )
    .await;
    assert_eq!(results.len(), 16);
    assert!(
        results.iter().all(|r| r.is_err()),
        "the injected read error must reach every waiter"
    );
    assert!(
        err_cache.stats().fetches >= 1 && err_cache.stats().fetches <= 16,
        "each waiter retries its own uncached read: {}",
        err_cache.stats().fetches
    );
    assert_eq!(err_cache.len(), 0, "failed reads are never cached");
    let mut reader = file_reader(&io, &p, bogus, opts_unfiltered()).await;
    let _ = err_handle
        .footer_or_fetch(&err_path, bogus, opts_unfiltered(), &mut reader)
        .await;
    assert_eq!(err_cache.len(), 0, "a retried failure is still uncached");
}

#[tokio::test]
async fn c7_byte_bound_evicts_and_rescan_identical() {
    let tmp = tmpdir();
    let (io, _opens, _ranges) = recording_io();
    let schema = id_schema();
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let mut paths = Vec::new();
    for i in 0..3 {
        let p = path(&tmp, &format!("c7-{i}.parquet"));
        write_id_pages(&p, &ids);
        paths.push(p);
    }
    let probe_cache = Arc::new(ParquetFooterCache::new());
    let probe = TableFooterCache::new(probe_cache, CacheScope::isolated("c7-probe"));
    let size0 = std::fs::metadata(&paths[0]).expect("stat").len();
    let path0_arc: Arc<str> = Arc::from(paths[0].as_str());
    let mut probe_reader = file_reader(&io, &paths[0], size0, opts_unfiltered()).await;
    let probed = probe
        .footer_or_fetch(&path0_arc, size0, opts_unfiltered(), &mut probe_reader)
        .await
        .expect("probe");
    let bound = probed.metadata().memory_size() as u64;
    let cache = Arc::new(ParquetFooterCache::with_max_bytes(bound));
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("c7"));
    let tasks = id_tasks(&paths, &schema);
    let batches = read_tasks(tasks.clone(), io.clone(), Some(handle.clone()), 1, true)
        .await
        .expect("scan");
    cache.run_pending_tasks().await;
    assert!(cache.weighted_size() <= bound);
    assert!(
        cache.stats().evictions >= 2,
        "three footers over a one-entry bound must evict: {}",
        cache.stats().evictions
    );
    let fetches = cache.stats().fetches;
    let rescanned = read_tasks(tasks.clone(), io.clone(), Some(handle), 1, true)
        .await
        .expect("rescan");
    assert!(
        cache.stats().fetches > fetches,
        "an evicted file must refetch"
    );
    let uncached = read_tasks(tasks, io, None, 1, true)
        .await
        .expect("uncached scan");
    assert_eq!(dump(&rescanned), dump(&uncached));
    assert_eq!(dump(&batches), dump(&uncached));
}

#[tokio::test]
async fn c8_cache_on_matches_off_and_still_prunes() {
    let tmp = tmpdir();
    let p = path(&tmp, "c8.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let index_ranges = index_byte_ranges(&file_metadata(&p));
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let uncached = collect(
        task(&p, schema.clone(), &[1], Some(predicate.clone())),
        true,
    )
    .await;
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("c8"));
    let filtered = task(&p, schema.clone(), &[1], Some(predicate));
    let first = read_tasks(
        vec![filtered.clone()],
        io.clone(),
        Some(handle.clone()),
        1,
        true,
    )
    .await
    .expect("cached filtered");
    assert_eq!(dump(&first), dump(&uncached));
    assert!(
        index_reads(&ranges, &p, &index_ranges) >= 1,
        "the page index must be read for a filtered scan"
    );
    let second = read_tasks(vec![filtered], io.clone(), Some(handle.clone()), 1, true)
        .await
        .expect("cached filtered again");
    assert_eq!(dump(&second), dump(&uncached));
    let unfiltered_cached = read_tasks(
        vec![task(&p, schema.clone(), &[1], None)],
        io,
        Some(handle),
        1,
        true,
    )
    .await
    .expect("cached unfiltered");
    let unfiltered_uncached = collect(task(&p, schema, &[1], None), true).await;
    assert_eq!(dump(&unfiltered_cached), dump(&unfiltered_uncached));
    assert_eq!(cache.stats().fetches, 1);
    assert_eq!(
        cache.stats().upgrades,
        0,
        "a filtered cold open loads the index inline"
    );
}

#[tokio::test]
async fn c9_no_cache_keeps_todays_counts() {
    let tmp = tmpdir();
    let (io, _opens, ranges) = recording_io();
    let schema = id_schema();
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let mut paths = Vec::new();
    for i in 0..4 {
        let p = path(&tmp, &format!("c9-{i}.parquet"));
        write_id_pages(&p, &ids);
        paths.push(p);
    }
    let tasks = id_tasks(&paths, &schema);
    let sizes: Vec<u64> = tasks.iter().map(|t| t.file_size_in_bytes).collect();
    read_tasks(tasks.clone(), io.clone(), None, 4, true)
        .await
        .expect("cold scan");
    for (p, size) in paths.iter().zip(&sizes) {
        assert_eq!(footer_reads(&ranges, p, *size), 1, "cold scan: {p}");
    }
    read_tasks(tasks, io.clone(), None, 4, true)
        .await
        .expect("warm scan");
    for (p, size) in paths.iter().zip(&sizes) {
        assert_eq!(
            footer_reads(&ranges, p, *size),
            2,
            "uncached warm scan: {p}"
        );
    }
    assert!(
        ArrowReaderBuilder::new(io.clone()).footer_cache.is_none(),
        "ArrowReaderBuilder::new attaches no footer cache"
    );
    let bare_table = Table::builder()
        .metadata(minimal_metadata())
        .identifier(TableIdent::from_strs(["db", "t"]).expect("ident"))
        .file_io(io.clone())
        .metadata_location("memory://wh/t/metadata/v1.json")
        .build()
        .expect("bare table");
    assert!(
        bare_table.footer_cache().is_none(),
        "Table::builder attaches no footer cache"
    );
    let bare_scan = bare_table.scan().build().expect("bare scan");
    assert!(
        bare_scan.footer_cache.is_none(),
        "TableScan from a cache-less table attaches no footer cache"
    );
    assert!(
        bare_scan
            .configure_reader(ArrowReaderBuilder::new(io.clone()))
            .footer_cache
            .is_none(),
        "configure_reader attaches no footer cache"
    );
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                tmp.path().to_str().expect("utf8").to_string(),
            )]),
        )
        .await
        .expect("catalog");
    let namespace = NamespaceIdent::new("ns".into());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let created = catalog
        .create_table(
            &namespace,
            crate::TableCreation::builder()
                .name("t".to_string())
                .schema(id_schema().as_ref().clone())
                .build(),
        )
        .await
        .expect("create");
    let loaded = catalog
        .load_table(&TableIdent::new(namespace, "t".to_string()))
        .await
        .expect("load");
    for candidate in [&created, &loaded] {
        assert!(
            candidate.footer_cache().is_none(),
            "a catalog without a shared footer cache attaches none"
        );
    }
}

fn minimal_metadata() -> crate::spec::TableMetadata {
    let schema = Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("schema");
    let spec = UnboundPartitionSpec::builder().with_spec_id(0).build();
    TableMetadataBuilder::new(
        schema,
        spec,
        SortOrder::unsorted_order(),
        "memory://wh/t".to_string(),
        FormatVersion::V2,
        HashMap::new(),
    )
    .expect("builder")
    .build()
    .expect("metadata")
    .metadata
}

fn cached_table(io: FileIO, handle: TableFooterCache) -> Table {
    Table::builder()
        .metadata(minimal_metadata())
        .identifier(TableIdent::from_strs(["db", "t"]).expect("ident"))
        .file_io(io)
        .metadata_location("memory://wh/t/metadata/v1.json")
        .footer_cache(handle)
        .build()
        .expect("table")
}

#[tokio::test]
async fn fc5_table_reader_builder_carries_cache() {
    let tmp = tmpdir();
    let p = path(&tmp, "fc5-table.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("fc5-table"));
    let table = cached_table(io.clone(), handle);
    assert!(table.footer_cache().is_some());
    let schema = id_schema();
    for _ in 0..2 {
        table
            .reader_builder()
            .build()
            .read(
                Box::pin(stream::iter(vec![Ok(task(&p, schema.clone(), &[1], None))]))
                    as FileScanTaskStream,
            )
            .expect("read")
            .try_collect::<Vec<RecordBatch>>()
            .await
            .expect("collect");
    }
    assert_eq!(footer_reads(&ranges, &p, size), 1);
}

#[tokio::test]
async fn fc5_table_scan_configure_reader_carries_cache() {
    let tmp = tmpdir();
    let p = path(&tmp, "fc5-scan.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("fc5-scan"));
    let table = cached_table(io.clone(), handle);
    let scan = table.scan().build().expect("scan");
    let schema = id_schema();
    for _ in 0..2 {
        scan.configure_reader(ArrowReaderBuilder::new(io.clone()))
            .build()
            .read(
                Box::pin(stream::iter(vec![Ok(task(&p, schema.clone(), &[1], None))]))
                    as FileScanTaskStream,
            )
            .expect("read")
            .try_collect::<Vec<RecordBatch>>()
            .await
            .expect("collect");
    }
    assert_eq!(footer_reads(&ranges, &p, size), 1);
}

#[tokio::test]
async fn fc5_stream_partition_work_carries_cache() {
    let tmp = tmpdir();
    let p = path(&tmp, "fc5-partition.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("fc5-partition"));
    let table = cached_table(io.clone(), handle);
    let scan = table.scan().build().expect("scan");
    let schema = id_schema();
    let works = assign_partition_work(
        0,
        ScanFilterMode::Residual,
        vec![CombinedScanTask::new(vec![task(
            &p,
            schema.clone(),
            &[1],
            None,
        )])],
        1,
    );
    let work = works.first().expect("one work unit").clone();
    for _ in 0..2 {
        scan.stream_partition_work(&work)
            .expect("stream")
            .try_collect::<Vec<RecordBatch>>()
            .await
            .expect("collect");
    }
    assert_eq!(footer_reads(&ranges, &p, size), 1);
    let works2 = assign_partition_work(
        0,
        ScanFilterMode::Residual,
        vec![CombinedScanTask::new(vec![task(&p, schema, &[1], None)])],
        1,
    );
    let work2 = works2.first().expect("one work unit").clone();
    for _ in 0..2 {
        stream_partition_work(io.clone(), &work2, 4, None, true, true, None)
            .expect("stream")
            .try_collect::<Vec<RecordBatch>>()
            .await
            .expect("collect");
    }
    assert_eq!(footer_reads(&ranges, &p, size), 3, "no handle: two reads");
}

#[tokio::test]
async fn fc5_memory_catalog_attaches_footer_cache() {
    let tmp = tmpdir();
    let cache = Arc::new(ParquetFooterCache::new());
    let catalog = MemoryCatalogBuilder::default()
        .with_shared_footer_cache(cache.clone())
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                tmp.path().to_str().expect("utf8").to_string(),
            )]),
        )
        .await
        .expect("catalog");
    let namespace = NamespaceIdent::new("ns".into());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let table = catalog
        .create_table(
            &namespace,
            crate::TableCreation::builder()
                .name("t".to_string())
                .schema(id_schema().as_ref().clone())
                .build(),
        )
        .await
        .expect("create");
    let loaded = catalog
        .load_table(&TableIdent::new(namespace, "t".to_string()))
        .await
        .expect("load");
    for candidate in [&table, &loaded] {
        let bound = candidate
            .footer_cache()
            .expect("catalog tables carry the shared footer cache");
        assert!(Arc::ptr_eq(&bound.shared(), &cache));
    }
}

#[tokio::test]
async fn measure_100_file_footer_requests() {
    let tmp = tmpdir();
    let schema = id_schema();
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let mut paths = Vec::new();
    for i in 0..100 {
        let p = path(&tmp, &format!("m-{i:03}.parquet"));
        write_id_pages(&p, &ids);
        paths.push(p);
    }
    let tasks = id_tasks(&paths, &schema);
    let sizes: Vec<u64> = tasks.iter().map(|t| t.file_size_in_bytes).collect();

    let (off_io, _o1, off_ranges) = recording_io();
    read_tasks(tasks.clone(), off_io.clone(), None, 8, true)
        .await
        .expect("uncached cold");
    let uncached_cold_reads: usize = paths
        .iter()
        .zip(&sizes)
        .map(|(p, s)| footer_reads(&off_ranges, p, *s))
        .sum();
    let uncached_cold_bytes: u64 = paths
        .iter()
        .zip(&sizes)
        .map(|(p, s)| footer_bytes(&off_ranges, p, *s))
        .sum();
    read_tasks(tasks.clone(), off_io, None, 8, true)
        .await
        .expect("uncached warm");
    let uncached_warm_reads: usize = paths
        .iter()
        .zip(&sizes)
        .map(|(p, s)| footer_reads(&off_ranges, p, *s))
        .sum();
    let uncached_warm_bytes: u64 = paths
        .iter()
        .zip(&sizes)
        .map(|(p, s)| footer_bytes(&off_ranges, p, *s))
        .sum();

    let (on_io, _o2, on_ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache, CacheScope::isolated("measure"));
    read_tasks(tasks.clone(), on_io.clone(), Some(handle.clone()), 8, true)
        .await
        .expect("cached cold");
    let cached_cold_reads: usize = paths
        .iter()
        .zip(&sizes)
        .map(|(p, s)| footer_reads(&on_ranges, p, *s))
        .sum();
    let cached_cold_bytes: u64 = paths
        .iter()
        .zip(&sizes)
        .map(|(p, s)| footer_bytes(&on_ranges, p, *s))
        .sum();
    read_tasks(tasks, on_io, Some(handle), 8, true)
        .await
        .expect("cached warm");
    let cached_warm_reads: usize = paths
        .iter()
        .zip(&sizes)
        .map(|(p, s)| footer_reads(&on_ranges, p, *s))
        .sum();
    let cached_warm_bytes: u64 = paths
        .iter()
        .zip(&sizes)
        .map(|(p, s)| footer_bytes(&on_ranges, p, *s))
        .sum();

    eprintln!(
        "measurement 100 files | uncached cold {uncached_cold_reads} reads {uncached_cold_bytes}B \
         | uncached warm +{uncached_warm_reads} reads {uncached_warm_bytes}B \
         | cached cold {cached_cold_reads} reads {cached_cold_bytes}B \
         | cached warm +{cached_warm_reads} reads {cached_warm_bytes}B"
    );
    assert_eq!(uncached_cold_reads, 100);
    assert_eq!(uncached_warm_reads, 200);
    assert_eq!(cached_cold_reads, 100);
    assert_eq!(cached_warm_reads, 100);
}

mod r2 {
    include!("footer_cache_r2_tests.rs");
}

mod v {
    include!("footer_cache_v_tests.rs");
}
