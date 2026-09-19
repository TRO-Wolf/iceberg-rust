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

use std::error::Error as _;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Schema as ArrowSchema};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::basic::Compression;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;

use super::open_parquet::{PAGE_INDEX_STRIPS, ROW_SELECTIONS_APPLIED, effective_row_selection};
use super::page_prune_fixture::*;
use super::reader::{ArrowReader, ParquetReadOptions};
use crate::expr::Reference;
use crate::spec::Datum;

#[tokio::test]
async fn footer_short_read_retries_with_real_size() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let actual = std::fs::metadata(&data_path).expect("stat").len();
    let (io, new_input_calls, _ranges) = recording_io();
    let (_reader, metadata) = ArrowReader::open_parquet_file(
        &data_path,
        &io,
        actual + 512,
        ParquetReadOptions::builder().build(),
        None,
    )
    .await
    .expect("stale manifest size retries with real size");
    assert!(metadata.metadata().file_metadata().num_rows() > 0);
    assert_eq!(
        new_input_calls.load(std::sync::atomic::Ordering::Relaxed),
        3
    );
}

#[tokio::test]
async fn footer_error_at_real_size_does_not_retry() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "garbage.parquet");
    std::fs::write(&data_path, vec![0xABu8; 300]).expect("write");
    let actual = std::fs::metadata(&data_path).expect("stat").len();
    let (io, new_input_calls, _ranges) = recording_io();
    let err = ArrowReader::open_parquet_file(
        &data_path,
        &io,
        actual,
        ParquetReadOptions::builder().build(),
        None,
    )
    .await
    .err()
    .expect("garbage file fails");
    assert!(
        err.to_string().contains("Failed to load Parquet metadata"),
        "unexpected error: {err}"
    );
    assert_eq!(
        new_input_calls.load(std::sync::atomic::Ordering::Relaxed),
        2
    );
}

#[tokio::test]
async fn page_index_error_does_not_retry() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let corrupt_path = path(&tmp, "corrupt.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let prefetched = ParquetMetaDataReader::new()
        .with_page_index_policy(PageIndexPolicy::Skip)
        .parse_and_finish(&std::fs::File::open(&data_path).expect("open"))
        .expect("prefetched metadata");
    let column_index_offset = prefetched
        .row_group(0)
        .column(0)
        .column_index_offset()
        .expect("column index present") as u64;
    let column_index_length = prefetched
        .row_group(0)
        .column(0)
        .column_index_length()
        .expect("column index present") as usize;
    let mut bytes = std::fs::read(&data_path).expect("read");
    for byte in
        &mut bytes[column_index_offset as usize..column_index_offset as usize + column_index_length]
    {
        *byte = 0xAB;
    }
    std::fs::write(&corrupt_path, &bytes).expect("write corrupt");
    let actual = bytes.len() as u64;
    let (io, new_input_calls, _ranges) = recording_io();
    let mut options = ParquetReadOptions::builder().build();
    options.preload_page_index = true;
    let err = ArrowReader::open_parquet_file(
        &corrupt_path,
        &io,
        actual + 512,
        options,
        Some(Arc::new(prefetched)),
    )
    .await
    .err()
    .expect("corrupt index fails");
    assert!(
        err.to_string().contains("page index"),
        "unexpected error: {err}"
    );
    assert_eq!(
        new_input_calls.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[tokio::test]
async fn not_eq_only_scan_reads_no_index_bytes() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let metadata = file_metadata(&data_path);
    let index_ranges = index_byte_ranges(&metadata);
    assert!(!index_ranges.is_empty(), "fixture must carry a page index");
    let predicate = bound(&schema, Reference::new("id").not_equal_to(Datum::int(64)));
    let (io, _calls, read_ranges) = recording_io();
    let rows = collect_with_io(task(&data_path, schema, &[1], Some(predicate)), true, io, 8).await;
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), ROWS - 1);
    assert!(
        !any_read_intersects(&read_ranges, &data_path, &index_ranges),
        "a !=-only filtered scan must not read page-index bytes"
    );
}

#[tokio::test]
async fn eq_scan_reads_index_bytes() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let metadata = file_metadata(&data_path);
    let index_ranges = index_byte_ranges(&metadata);
    assert!(!index_ranges.is_empty(), "fixture must carry a page index");
    let predicate = bound(&schema, Reference::new("id").equal_to(Datum::int(64)));
    let (io, _calls, read_ranges) = recording_io();
    let rows = collect_with_io(task(&data_path, schema, &[1], Some(predicate)), true, io, 8).await;
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    assert!(
        any_read_intersects(&read_ranges, &data_path, &index_ranges),
        "an = filtered scan must read page-index bytes"
    );
}

#[tokio::test]
async fn deletes_force_index_load_under_not_eq() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let del_path = path(&tmp, "pos-deletes.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let metadata = file_metadata(&data_path);
    let index_ranges = index_byte_ranges(&metadata);
    assert!(!index_ranges.is_empty(), "fixture must carry a page index");
    let predicate = bound(&schema, Reference::new("id").not_equal_to(Datum::int(64)));
    let delete = write_pos_delete_file(&del_path, &data_path, &[10, 70]);
    let (io, _calls, read_ranges) = recording_io();
    let rows = collect_with_io(
        with_deletes(task(&data_path, schema, &[1], Some(predicate)), vec![
            delete,
        ]),
        true,
        io,
        8,
    )
    .await;
    assert_eq!(
        rows.iter().map(|b| b.num_rows()).sum::<usize>(),
        ROWS - 1 - 2
    );
    assert!(
        any_read_intersects(&read_ranges, &data_path, &index_ranges),
        "deletes must force the page-index load even for a non-prunable predicate"
    );
}

#[test]
fn all_keep_row_selection_is_dropped() {
    let all_keep = RowSelection::from(vec![RowSelector::select(128)]);
    assert!(effective_row_selection(Some(all_keep)).is_none());
}

#[test]
fn skipping_row_selection_is_kept() {
    let skipping = RowSelection::from(vec![RowSelector::skip(8), RowSelector::select(120)]);
    assert!(effective_row_selection(Some(skipping)).is_some());
}

#[test]
fn empty_row_selection_is_kept() {
    let empty = RowSelection::from(Vec::new());
    assert!(effective_row_selection(Some(empty)).is_some());
}

#[tokio::test]
async fn all_keep_predicate_hands_no_selection_to_parquet() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(0)),
    );
    let before = ROW_SELECTIONS_APPLIED.with(|count| count.get());
    let strips_before = PAGE_INDEX_STRIPS.with(|count| count.get());
    let rows = collect(task(&data_path, schema, &[1], Some(predicate)), true).await;
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), ROWS);
    assert_eq!(
        ROW_SELECTIONS_APPLIED.with(|count| count.get()) - before,
        0,
        "a predicate that keeps every page must not hand parquet a RowSelection"
    );
    assert_eq!(
        PAGE_INDEX_STRIPS.with(|count| count.get()) - strips_before,
        1,
        "a predicate that keeps every page must not let decode see the page index"
    );
}

#[tokio::test]
async fn pruning_predicate_hands_selection_to_parquet() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let predicate = bound(&schema, Reference::new("id").equal_to(Datum::int(64)));
    let before = ROW_SELECTIONS_APPLIED.with(|count| count.get());
    let strips_before = PAGE_INDEX_STRIPS.with(|count| count.get());
    let rows = collect(task(&data_path, schema, &[1], Some(predicate)), true).await;
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    assert_eq!(
        ROW_SELECTIONS_APPLIED.with(|count| count.get()) - before,
        1,
        "a predicate that skips pages must hand parquet a RowSelection"
    );
    assert_eq!(
        PAGE_INDEX_STRIPS.with(|count| count.get()) - strips_before,
        0,
        "a predicate that skips pages must keep the page index"
    );
}

#[tokio::test]
async fn position_deletes_hand_selection_to_parquet() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let del_path = path(&tmp, "pos-deletes.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let delete = write_pos_delete_file(&del_path, &data_path, &[10, 70]);
    let before = ROW_SELECTIONS_APPLIED.with(|count| count.get());
    let strips_before = PAGE_INDEX_STRIPS.with(|count| count.get());
    let rows = collect(
        with_deletes(task(&data_path, schema, &[1], None), vec![delete]),
        true,
    )
    .await;
    assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), ROWS - 2);
    assert_eq!(
        ROW_SELECTIONS_APPLIED.with(|count| count.get()) - before,
        1,
        "a delete selection must never be dropped"
    );
    assert_eq!(
        PAGE_INDEX_STRIPS.with(|count| count.get()) - strips_before,
        0,
        "deletes must keep the page index"
    );
}

#[tokio::test]
async fn retry_failure_reports_first_error_as_source() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "garbage.parquet");
    std::fs::write(&data_path, vec![0xABu8; 300]).expect("write");
    let actual = std::fs::metadata(&data_path).expect("stat").len();
    let (io, _calls, _ranges) = recording_io();
    let err = ArrowReader::open_parquet_file(
        &data_path,
        &io,
        actual + 512,
        ParquetReadOptions::builder().build(),
        None,
    )
    .await
    .err()
    .expect("garbage file fails");
    let source = err.source().expect("first error kept as source");
    assert!(
        source
            .to_string()
            .contains("Failed to load Parquet metadata"),
        "first error not preserved as source: {source}"
    );
}

#[tokio::test]
async fn ranged_all_keep_scan_returns_only_split_row_groups() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "id",
        DataType::Int32,
        false,
        1,
    )]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(ids)) as ArrayRef
    ])
    .expect("batch");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_data_page_row_count_limit(32)
        .set_write_batch_size(32)
        .set_max_row_group_row_count(Some(128))
        .build();
    write_parquet(&data_path, arrow_schema, &[batch], props);
    let metadata = file_metadata(&data_path);
    assert_eq!(metadata.num_row_groups(), 4);
    assert_page_count(&metadata, 3, 0, 3);
    let rg2_start = {
        let first_column = metadata.row_group(2).columns().first().expect("column");
        let data_offset = first_column.data_page_offset();
        match first_column.dictionary_page_offset() {
            Some(dict) if data_offset > dict => dict,
            _ => data_offset,
        }
    } as u64;
    let file_size = std::fs::metadata(&data_path).expect("stat").len();
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(0)),
    );
    let mut t = task(&data_path, schema, &[1], Some(predicate));
    t.start = rg2_start;
    t.length = file_size - rg2_start;
    let strips_before = PAGE_INDEX_STRIPS.with(|count| count.get());
    let rows = collect(t, true).await;
    assert_eq!(
        rows.iter().map(|b| b.num_rows()).sum::<usize>(),
        256,
        "the ranged split must return only row groups 2 and 3"
    );
    assert_eq!(
        PAGE_INDEX_STRIPS.with(|count| count.get()) - strips_before,
        1,
        "an all-keep ranged scan must take the strip path"
    );
}

#[tokio::test]
async fn all_keep_pages_return_only_selected_row_groups() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let strings: Vec<Option<String>> = (0..ROWS)
        .map(|i| (i >= 256).then(|| "a".to_string()))
        .collect();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, false, 1),
        field("s", DataType::Utf8, true, 2),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(strings)) as ArrayRef,
    ])
    .expect("batch");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_data_page_row_count_limit(32)
        .set_write_batch_size(32)
        .set_max_row_group_row_count(Some(128))
        .build();
    write_parquet(&data_path, arrow_schema, &[batch], props);
    let metadata = file_metadata(&data_path);
    assert_eq!(metadata.num_row_groups(), 4);
    assert_page_count(&metadata, 3, 1, 3);
    let schema = id_s_schema();
    let predicate = bound(&schema, Reference::new("s").less_than(Datum::string("x")));
    let strips_before = PAGE_INDEX_STRIPS.with(|count| count.get());
    let rows = collect(task(&data_path, schema, &[1, 2], Some(predicate)), true).await;
    assert_eq!(
        rows.iter().map(|b| b.num_rows()).sum::<usize>(),
        256,
        "row-group stats must drop the all-null row groups the nulls-first residual keeps"
    );
    assert_eq!(
        PAGE_INDEX_STRIPS.with(|count| count.get()) - strips_before,
        1,
        "all-keep pages over the surviving row groups must take the strip path"
    );
}
