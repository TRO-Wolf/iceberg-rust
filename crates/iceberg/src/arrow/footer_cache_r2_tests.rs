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

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use parquet::basic::Compression;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::{EnabledStatistics, WriterProperties};

use super::*;
use crate::arrow::footer_cache::{ParquetFooterCache, TableFooterCache};
use crate::arrow::reader::ArrowReader;
use crate::catalog::CacheScope;
use crate::expr::Reference;
use crate::spec::{Datum, MappedField, NameMapping, NestedField, PrimitiveType, Type};

fn write_id_pages_noindex(path: &str, ids: &[i32]) {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "id",
        DataType::Int32,
        false,
        1,
    )]));
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
        ])
        .expect("batch");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .set_offset_index_disabled(true)
        .set_statistics_enabled(EnabledStatistics::None)
        .build();
    write_parquet(path, arrow_schema, &[batch], props);
}

fn write_id_pages_noids(path: &str, ids: &[i32]) {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
        ])
        .expect("batch");
    write_parquet(path, arrow_schema, &[batch], page_props());
}

fn write_int96_pages(path: &str) {
    use parquet::basic::{Repetition, Type as PhysicalType};
    use parquet::data_type::{Int32Type, Int96, Int96Type};
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type as SchemaType;

    const JULIAN_3333: u32 = 2_953_529;
    let ts = SchemaType::primitive_type_builder("ts", PhysicalType::INT96)
        .with_repetition(Repetition::OPTIONAL)
        .with_id(Some(1))
        .build()
        .expect("ts");
    let id = SchemaType::primitive_type_builder("id", PhysicalType::INT32)
        .with_repetition(Repetition::REQUIRED)
        .with_id(Some(2))
        .build()
        .expect("id");
    let schema = SchemaType::group_type_builder("schema")
        .with_fields(vec![Arc::new(ts), Arc::new(id)])
        .build()
        .expect("schema");
    let file = std::fs::File::create(path).expect("create");
    let mut writer =
        SerializedFileWriter::new(file, Arc::new(schema), Default::default()).expect("writer");
    let mut row_group = writer.next_row_group().expect("row group");
    {
        let values: Vec<Int96> = (0..3u32)
            .map(|i| {
                let mut v = Int96::new();
                v.set_data(0, i, JULIAN_3333);
                v
            })
            .collect();
        let mut col = row_group
            .next_column()
            .expect("column")
            .expect("column writer");
        col.typed::<Int96Type>()
            .write_batch(&values, Some(&[1; 3]), None)
            .expect("write ts");
        col.close().expect("close ts");
    }
    {
        let mut col = row_group
            .next_column()
            .expect("column")
            .expect("column writer");
        col.typed::<Int32Type>()
            .write_batch(&[0, 1, 2], None, None)
            .expect("write id");
        col.close().expect("close id");
    }
    row_group.close().expect("row group close");
    writer.close().expect("close");
}

fn file_metadata_noindex(path: &str) -> Arc<ParquetMetaData> {
    let file = std::fs::File::open(path).expect("open");
    let metadata = ParquetMetaDataReader::new()
        .with_page_index_policy(PageIndexPolicy::Skip)
        .with_column_index_policy(PageIndexPolicy::Skip)
        .with_offset_index_policy(PageIndexPolicy::Skip)
        .parse_and_finish(&file)
        .expect("metadata");
    Arc::new(metadata)
}

#[tokio::test]
async fn r01_hit_reuses_base_arrow_metadata() {
    let tmp = tmpdir();
    let p = path(&tmp, "r01.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let path_arc: Arc<str> = Arc::from(p.as_str());
    let (io, _opens, _ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache, CacheScope::isolated("r01"));
    let (_r1, first) = ArrowReader::open_parquet_file_cached(
        &path_arc,
        &io,
        size,
        opts_unfiltered(),
        None,
        Some(handle.clone()),
    )
    .await
    .expect("cold open");
    let (_r2, second) = ArrowReader::open_parquet_file_cached(
        &path_arc,
        &io,
        size,
        opts_unfiltered(),
        None,
        Some(handle),
    )
    .await
    .expect("warm open");
    assert!(Arc::ptr_eq(first.metadata(), second.metadata()));
    assert!(
        Arc::ptr_eq(first.schema(), second.schema()),
        "a hit must reuse the cached base ArrowReaderMetadata, not rebuild it"
    );
}

#[tokio::test]
async fn r01_per_task_rebuilds_still_apply() {
    let tmp = tmpdir();
    let ids: Vec<i32> = (0..ROWS as i32).collect();

    let mapped = path(&tmp, "r01-mapped.parquet");
    write_id_pages_noids(&mapped, &ids);
    let schema = id_schema();
    let mut mapped_task = task(&mapped, schema.clone(), &[1], None);
    mapped_task.name_mapping = Some(Arc::new(NameMapping::new(vec![MappedField::new(
        Some(1),
        vec!["id".to_string()],
        vec![],
    )])));
    let uncached_mapped = collect(mapped_task.clone(), true).await;
    let (io, _o, _r) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache, CacheScope::isolated("r01-rebuild"));
    for _ in 0..2 {
        let cached = read_tasks(
            vec![mapped_task.clone()],
            io.clone(),
            Some(handle.clone()),
            1,
            true,
        )
        .await
        .expect("cached name-mapped read");
        assert_eq!(dump(&cached), dump(&uncached_mapped));
    }

    let int96 = path(&tmp, "r01-int96.parquet");
    write_int96_pages(&int96);
    let ts_schema = iceberg_schema(vec![
        NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp)),
        NestedField::required(2, "id", Type::Primitive(PrimitiveType::Int)),
    ]);
    let int96_task = task(&int96, ts_schema, &[1, 2], None);
    let uncached_int96 = collect(int96_task.clone(), true).await;
    for _ in 0..2 {
        let cached = read_tasks(
            vec![int96_task.clone()],
            io.clone(),
            Some(handle.clone()),
            1,
            true,
        )
        .await
        .expect("cached int96 read");
        assert_eq!(dump(&cached), dump(&uncached_int96));
    }
}

#[tokio::test]
async fn r02_key_reuses_task_path_arc() {
    let tmp = tmpdir();
    let p = path(&tmp, "r02.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let schema = id_schema();
    let t = task(&p, schema, &[1], None);
    let size = t.file_size_in_bytes;
    let (io, _opens, _ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("r02"));
    let before = Arc::strong_count(&t.data_file_path);
    let mut reader = file_reader(&io, &p, size, opts_unfiltered()).await;
    handle
        .footer_or_fetch(&t.data_file_path, size, opts_unfiltered(), &mut reader)
        .await
        .expect("fetch");
    cache.run_pending_tasks().await;
    assert_eq!(
        Arc::strong_count(&t.data_file_path),
        before + 1,
        "the cache key must hold the task's Arc<str> path, not a fresh allocation"
    );
}

#[tokio::test]
async fn r03_upgrade_grows_weight_at_stable_count() {
    let tmp = tmpdir();
    let p = path(&tmp, "r03.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let schema = id_schema();
    let (io, _opens, _ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let handle = TableFooterCache::new(cache.clone(), CacheScope::isolated("r03"));
    read_tasks(
        vec![task(&p, schema.clone(), &[1], None)],
        io.clone(),
        Some(handle.clone()),
        1,
        true,
    )
    .await
    .expect("unfiltered");
    cache.run_pending_tasks().await;
    let weight_unindexed = cache.weighted_size();
    assert_eq!(cache.len(), 1);
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    read_tasks(
        vec![task(&p, schema, &[1], Some(predicate))],
        io,
        Some(handle),
        1,
        true,
    )
    .await
    .expect("filtered");
    cache.run_pending_tasks().await;
    assert_eq!(cache.len(), 1, "the upgrade replaces under the same key");
    assert!(
        cache.weighted_size() > weight_unindexed,
        "an index upgrade must grow the weighed entry: {weight_unindexed} -> {}",
        cache.weighted_size()
    );
}

#[tokio::test]
async fn r08_indexless_never_replaces_indexed() {
    let tmp = tmpdir();
    let p = path(&tmp, "r08.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&p, &ids);
    let size = std::fs::metadata(&p).expect("stat").len();
    let path_arc: Arc<str> = Arc::from(p.as_str());
    let (io, _opens, _ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let scope = CacheScope::isolated("r08");
    let handle = TableFooterCache::new(cache.clone(), scope.clone());
    let mut indexed_reader = file_reader(&io, &p, size, opts_filtered()).await;
    handle
        .footer_or_fetch(&path_arc, size, opts_filtered(), &mut indexed_reader)
        .await
        .expect("indexed fetch");
    assert_eq!(
        cache.probe_index_state(&scope, &path_arc, size).await,
        Some((true, true))
    );
    let mut plain_reader = file_reader(&io, &p, size, opts_unfiltered()).await;
    let served = handle
        .footer_or_fetch(&path_arc, size, opts_unfiltered(), &mut plain_reader)
        .await
        .expect("index-less open");
    assert!(
        served.metadata().column_index().is_some(),
        "the index-less open is served the indexed entry"
    );
    assert_eq!(
        cache.probe_index_state(&scope, &path_arc, size).await,
        Some((true, true)),
        "an index-less open must never replace an indexed entry"
    );
    handle
        .seed(&path_arc, size, file_metadata_noindex(&p))
        .await;
    assert_eq!(
        cache.probe_index_state(&scope, &path_arc, size).await,
        Some((true, true)),
        "an index-less prefetch seed must not downgrade an indexed entry"
    );
    assert_eq!(cache.stats().fetches, 1);
}

#[tokio::test]
async fn l002_index_checked_reflects_presence() {
    let tmp = tmpdir();
    let ids: Vec<i32> = (0..ROWS as i32).collect();

    let p = path(&tmp, "l002-noindex.parquet");
    write_id_pages_noindex(&p, &ids);
    let meta = file_metadata(&p);
    assert!(
        meta.column_index().is_none() && meta.offset_index().is_none(),
        "fixture must produce an index-less file"
    );
    let size = std::fs::metadata(&p).expect("stat").len();
    let path_arc: Arc<str> = Arc::from(p.as_str());
    let (io, _opens, _ranges) = recording_io();
    let cache = Arc::new(ParquetFooterCache::new());
    let scope = CacheScope::isolated("l002");
    let handle = TableFooterCache::new(cache.clone(), scope.clone());
    let mut filtered_reader = file_reader(&io, &p, size, opts_filtered()).await;
    handle
        .footer_or_fetch(&path_arc, size, opts_filtered(), &mut filtered_reader)
        .await
        .expect("filtered fetch on index-less file");
    assert_eq!(
        cache.probe_index_state(&scope, &path_arc, size).await,
        Some((false, true)),
        "index_checked reflects the loaded index, not the request"
    );
    let mut second_reader = file_reader(&io, &p, size, opts_filtered()).await;
    handle
        .footer_or_fetch(&path_arc, size, opts_filtered(), &mut second_reader)
        .await
        .expect("second filtered open");
    assert_eq!(cache.stats().fetches, 1, "an attempted entry must hit");
    assert_eq!(
        cache.stats().upgrades,
        0,
        "absent index is never an upgrade"
    );

    let p2 = path(&tmp, "l002-indexed.parquet");
    write_id_pages(&p2, &ids);
    let size2 = std::fs::metadata(&p2).expect("stat").len();
    let path2_arc: Arc<str> = Arc::from(p2.as_str());
    let mut plain_reader = file_reader(&io, &p2, size2, opts_unfiltered()).await;
    handle
        .footer_or_fetch(&path2_arc, size2, opts_unfiltered(), &mut plain_reader)
        .await
        .expect("footer-only fetch");
    assert_eq!(
        cache.probe_index_state(&scope, &path2_arc, size2).await,
        Some((false, false))
    );
    let mut filtered_reader2 = file_reader(&io, &p2, size2, opts_filtered()).await;
    handle
        .footer_or_fetch(&path2_arc, size2, opts_filtered(), &mut filtered_reader2)
        .await
        .expect("filtered fetch");
    assert_eq!(
        cache.probe_index_state(&scope, &path2_arc, size2).await,
        Some((true, true)),
        "the upgrade records actual index presence"
    );
    assert_eq!(cache.stats().upgrades, 1);
}
