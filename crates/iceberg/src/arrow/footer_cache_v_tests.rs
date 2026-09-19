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

use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use parquet::file::properties::WriterProperties;

use super::*;
use crate::arrow::footer_cache::{ParquetFooterCache, TableFooterCache};
use crate::arrow::open_parquet::{PAGE_INDEX_STRIPS, ROW_SELECTIONS_APPLIED};
use crate::arrow::reader::ArrowReader;
use crate::catalog::CacheScope;
use crate::expr::Reference;
use crate::spec::{Datum, MappedField, NameMapping, NestedField, PrimitiveType, Type};

fn write_two_cols_noids(path: &str, xs: &[i32], ys: &[i32]) {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("col_x", DataType::Int32, false),
        Field::new("col_y", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(xs.to_vec())) as ArrayRef,
        Arc::new(Int32Array::from(ys.to_vec())) as ArrayRef,
    ])
    .expect("batch");
    write_parquet(
        path,
        arrow_schema,
        &[batch],
        WriterProperties::builder().build(),
    );
}

fn mapping_for(parquet_name: &str) -> Arc<NameMapping> {
    Arc::new(NameMapping::new(vec![MappedField::new(
        Some(1),
        vec![parquet_name.to_string()],
        vec![],
    )]))
}

#[tokio::test]
async fn v_shared_cache_two_name_mappings_keep_own_rows() {
    let tmp = tmpdir();
    let p = path(&tmp, "hive-migrated.parquet");
    let xs: Vec<i32> = (0..ROWS as i32).collect();
    let ys: Vec<i32> = xs.iter().map(|v| v + 10_000).collect();
    write_two_cols_noids(&p, &xs, &ys);
    let schema = iceberg_schema(vec![NestedField::required(
        1,
        "id",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let mut map_x = task(&p, schema.clone(), &[1], None);
    map_x.name_mapping = Some(mapping_for("col_x"));
    let mut map_y = task(&p, schema, &[1], None);
    map_y.name_mapping = Some(mapping_for("col_y"));
    let uncached_x = collect(map_x.clone(), true).await;
    let uncached_y = collect(map_y.clone(), true).await;
    assert_ne!(
        dump(&uncached_x),
        dump(&uncached_y),
        "the two mappings must project different physical columns"
    );
    let (io, _opens, _ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("v-map"));
    let cached_x = read_tasks(
        vec![map_x.clone()],
        io.clone(),
        Some(handle.clone()),
        1,
        true,
    )
    .await
    .expect("mapped x");
    let cached_y = read_tasks(vec![map_y.clone()], io, Some(handle), 1, true)
        .await
        .expect("mapped y");
    assert_eq!(dump(&cached_x), dump(&uncached_x), "mapping col_x");
    assert_eq!(dump(&cached_y), dump(&uncached_y), "mapping col_y");
    assert_ne!(dump(&cached_x), dump(&cached_y));
    assert_eq!(cache.stats().fetches, 1, "one shared footer for both tables");
}

#[tokio::test]
async fn v_cached_indexless_then_filtered_pos_deletes_match_uncached() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let del_path = path(&tmp, "pos-deletes.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let delete = write_pos_delete_file(&del_path, &data_path, &[10, 70, 300, 500]);
    let unfiltered = task(&data_path, schema.clone(), &[1], None);
    let filtered_deletes = with_deletes(
        task(&data_path, schema, &[1], Some(predicate)),
        vec![delete],
    );
    let uncached = collect(filtered_deletes.clone(), true).await;
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let scope = CacheScope::isolated("v-del");
    let handle = TableFooterCache::new(cache.clone(), scope.clone());
    read_tasks(
        vec![unfiltered],
        io.clone(),
        Some(handle.clone()),
        1,
        true,
    )
    .await
    .expect("index-less warm");
    cache.run_pending_tasks().await;
    assert_eq!(cache.stats().fetches, 1);
    assert_eq!(cache.stats().upgrades, 0);
    let size = std::fs::metadata(&data_path).expect("stat").len();
    let path_arc: Arc<str> = Arc::from(data_path.as_str());
    assert_eq!(
        cache.probe_index_state(&scope, &path_arc, size).await,
        Some((false, false)),
        "unfiltered scan must leave an index-less entry"
    );
    let cached = read_tasks(vec![filtered_deletes], io, Some(handle), 1, true)
        .await
        .expect("filtered deletes");
    assert_eq!(dump(&cached), dump(&uncached));
    assert_eq!(cache.stats().upgrades, 1, "deletes must upgrade the index");
    assert_eq!(cache.stats().fetches, 1);
    let _ = ranges;
}

#[tokio::test]
async fn v_all_keep_strip_leaves_cached_index_for_later_prune() {
    let tmp = tmpdir();
    let p = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let path_arc: Arc<str> = Arc::from(p.as_str());
    let schema = id_schema();
    let all_keep = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(0)),
    );
    let pruning = bound(&schema, Reference::new("id").equal_to(Datum::int(64)));
    let uncached = collect(
        task(&p, schema.clone(), &[1], Some(pruning.clone())),
        true,
    )
    .await;
    let (io, _opens, _ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("v-strip"));
    let strips_before = PAGE_INDEX_STRIPS.with(|count| count.get());
    let first = read_tasks(
        vec![task(&p, schema.clone(), &[1], Some(all_keep))],
        io.clone(),
        Some(handle.clone()),
        1,
        true,
    )
    .await
    .expect("cached all-keep scan");
    assert_eq!(first.iter().map(|b| b.num_rows()).sum::<usize>(), ROWS);
    assert_eq!(
        PAGE_INDEX_STRIPS.with(|count| count.get()) - strips_before,
        1,
        "the all-keep cached scan must take the strip path"
    );
    cache.run_pending_tasks().await;
    let mut probe_reader = file_reader(&io, &p, size, opts_filtered()).await;
    let served = handle
        .footer_or_fetch(&path_arc, size, opts_filtered(), &mut probe_reader)
        .await
        .expect("warm hit");
    assert!(
        served.metadata().column_index().is_some() && served.metadata().offset_index().is_some(),
        "the strip must not remove the page index from the shared cache entry"
    );
    let applied_before = ROW_SELECTIONS_APPLIED.with(|count| count.get());
    let second = read_tasks(
        vec![task(&p, schema, &[1], Some(pruning))],
        io,
        Some(handle),
        1,
        true,
    )
    .await
    .expect("cached pruning scan");
    assert_eq!(second.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    assert_eq!(dump(&second), dump(&uncached));
    assert_eq!(
        ROW_SELECTIONS_APPLIED.with(|count| count.get()) - applied_before,
        1,
        "a pruning scan of the cached file must still hand parquet a RowSelection"
    );
}

#[tokio::test]
async fn v_stale_manifest_size_retries_and_later_tasks_hit_actual() {
    let tmp = tmpdir();
    let p = path(&tmp, "stale.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let stale = size + 512;
    let path_arc: Arc<str> = Arc::from(p.as_str());
    let schema = id_schema();
    let (io, _opens, ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let scope = CacheScope::isolated("v-stale");
    let handle = TableFooterCache::new(cache.clone(), scope.clone());
    let (_reader, metadata) = ArrowReader::open_parquet_file_cached(
        &path_arc,
        &io,
        stale,
        opts_unfiltered(),
        None,
        Some(handle.clone()),
    )
    .await
    .expect("stale size retries with the real file size");
    assert!(metadata.metadata().file_metadata().num_rows() > 0);
    let cold_fetches = cache.stats().fetches;
    assert!(
        cold_fetches >= 1,
        "the retry must fetch the footer at the actual size"
    );
    assert_eq!(
        cache.probe_index_state(&scope, &path_arc, stale).await,
        None,
        "a failed stale-size open must not cache under the manifest size"
    );
    assert!(
        cache
            .probe_index_state(&scope, &path_arc, size)
            .await
            .is_some(),
        "the retry must key the entry by the actual file size"
    );
    let footers_actual = footer_reads(&ranges, &p, size);
    assert!(footers_actual >= 1, "actual-size footer must be read once");
    ArrowReader::open_parquet_file_cached(
        &path_arc,
        &io,
        size,
        opts_unfiltered(),
        None,
        Some(handle.clone()),
    )
    .await
    .expect("actual-size hit");
    assert_eq!(
        cache.stats().fetches,
        cold_fetches,
        "a later actual-size task must hit, not miss forever"
    );
    ArrowReader::open_parquet_file_cached(
        &path_arc,
        &io,
        stale,
        opts_unfiltered(),
        None,
        Some(handle.clone()),
    )
    .await
    .expect("later stale-size task still opens");
    assert_eq!(
        cache.probe_index_state(&scope, &path_arc, stale).await,
        None,
        "later manifest-size tasks must not insert under the stale size"
    );
    assert_eq!(
        footer_reads(&ranges, &p, size),
        footers_actual,
        "later tasks must not re-read the actual-size footer"
    );
    let mut stale_task = task(&p, schema.clone(), &[1], None);
    stale_task.file_size_in_bytes = stale;
    let cached_rows = read_tasks(
        vec![stale_task],
        io.clone(),
        Some(handle),
        1,
        true,
    )
    .await
    .expect("stale-size scan");
    let uncached_rows = collect(task(&p, schema, &[1], None), true).await;
    assert_eq!(
        dump(&cached_rows),
        dump(&uncached_rows),
        "a stale-size cached scan must not serve the wrong bytes"
    );
}
