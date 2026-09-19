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

use std::fs::File;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ArrowReaderMetadata, ArrowReaderOptions, RowFilter, RowSelection,
};
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::page_prune_fixture::{
    bound, collect_with_io, field, field_id_map, file_metadata, iceberg_schema, index_byte_ranges,
    recording_io, selected_rows, task, tmpdir,
};
use super::reader::{ArrowFileReader, ArrowReader, ArrowReaderBuilder};
use crate::expr::Reference;
use crate::io::{FileIO, FileMetadata};
use crate::scan::{FileScanTask, FileScanTaskStream};
use crate::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, SchemaRef, Type};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

const FILES: usize = 50;
const ROWS: usize = 128_000;
const PAGE_ROWS: usize = 8_000;
const REPS: usize = 5;
const BATCH_SIZE: usize = 8_192;
const DECODE_FILES: usize = 10;

fn perf_schema() -> SchemaRef {
    iceberg_schema(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)),
        NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)),
        NestedField::required(3, "v1", Type::Primitive(PrimitiveType::Double)),
        NestedField::required(4, "v2", Type::Primitive(PrimitiveType::Int)),
    ])
}

fn perf_batch() -> RecordBatch {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int64, false, 1),
        field("category", DataType::Utf8, false, 2),
        field("v1", DataType::Float64, false, 3),
        field("v2", DataType::Int32, false, 4),
    ]));
    RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from((0..ROWS as i64).collect::<Vec<i64>>())) as ArrayRef,
        Arc::new(StringArray::from(
            (0..ROWS)
                .map(|i| format!("cat_{}", i % 8))
                .collect::<Vec<String>>(),
        )) as ArrayRef,
        Arc::new(Float64Array::from(
            (0..ROWS).map(|i| i as f64 * 0.5).collect::<Vec<f64>>(),
        )) as ArrayRef,
        Arc::new(Int32Array::from(
            (0..ROWS).map(|i| (i % 997) as i32).collect::<Vec<i32>>(),
        )) as ArrayRef,
    ])
    .expect("batch")
}

async fn write_perf_files(tmp: &TempDir) -> Vec<String> {
    let file_io = FileIO::new_with_fs();
    let schema = perf_schema();
    let batch = perf_batch();
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(PAGE_ROWS)
        .build();
    let mut paths = Vec::with_capacity(FILES);
    for i in 0..FILES {
        let location_gen = DefaultLocationGenerator::with_data_location(
            tmp.path().to_str().expect("utf8").to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new(format!("perf{i}"), None, DataFileFormat::Parquet);
        let parquet_builder = ParquetWriterBuilder::new(props.clone(), schema.clone());
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );
        let mut writer = DataFileWriterBuilder::new(rolling)
            .unpartitioned()
            .build(None)
            .await
            .expect("writer");
        writer.write(batch.clone()).await.expect("write");
        let data_files = writer.close().await.expect("close");
        assert_eq!(data_files.len(), 1);
        paths.push(data_files[0].file_path.clone());
    }
    paths
}

fn perf_tasks(
    paths: &[String],
    schema: &SchemaRef,
    predicate: Option<crate::expr::BoundPredicate>,
) -> Vec<FileScanTask> {
    paths
        .iter()
        .map(|p| task(p, schema.clone(), &[1, 2, 3, 4], predicate.clone()))
        .collect()
}

async fn collect_timed(tasks: &[FileScanTask], row_selection: bool) -> (Duration, usize) {
    let stream =
        Box::pin(futures::stream::iter(Vec::from(tasks).into_iter().map(Ok))) as FileScanTaskStream;
    let reader = ArrowReaderBuilder::new(FileIO::new_with_fs())
        .with_batch_size(BATCH_SIZE)
        .with_row_group_filtering_enabled(true)
        .with_row_selection_enabled(row_selection)
        .build();
    let start = Instant::now();
    let batches = reader
        .read(stream)
        .expect("read")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect");
    (start.elapsed(), batches.iter().map(|b| b.num_rows()).sum())
}

async fn decode_one(
    path: &str,
    selection: Option<RowSelection>,
    load_index: bool,
    with_filter: bool,
) -> Duration {
    let file_io = FileIO::new_with_fs();
    let input = file_io.new_input(path).expect("input");
    let file_read = input.reader().await.expect("reader");
    let size = std::fs::metadata(path).expect("stat").len();
    let mut arrow_reader = ArrowFileReader::new(FileMetadata { size }, file_read);
    let policy = if load_index {
        PageIndexPolicy::Optional
    } else {
        PageIndexPolicy::Skip
    };
    let metadata = ParquetMetaDataReader::new()
        .with_page_index_policy(policy)
        .with_column_index_policy(policy)
        .with_offset_index_policy(policy)
        .load_and_finish(&mut arrow_reader, size)
        .await
        .expect("metadata");
    let arrow_metadata =
        ArrowReaderMetadata::try_new(Arc::new(metadata), ArrowReaderOptions::default())
            .expect("arrow metadata");
    let mut builder =
        ParquetRecordBatchStreamBuilder::new_with_metadata(arrow_reader, arrow_metadata)
            .with_batch_size(BATCH_SIZE)
            .with_projection(ProjectionMask::all());
    if let Some(selection) = selection {
        builder = builder.with_row_selection(selection);
    }
    if with_filter {
        let parquet_schema = builder.parquet_schema().clone();
        let projection = ProjectionMask::leaves(&parquet_schema, [1]);
        let predicate = ArrowPredicateFn::new(projection, |batch: RecordBatch| {
            let category = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("category");
            Ok(BooleanArray::from(
                category
                    .iter()
                    .map(|v| v == Some("cat_7"))
                    .collect::<Vec<bool>>(),
            ))
        });
        builder = builder.with_row_filter(RowFilter::new(vec![Box::new(predicate)]));
    }
    let start = Instant::now();
    let batches = builder
        .build()
        .expect("build")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("decode");
    let expected = if with_filter { ROWS / 8 } else { ROWS };
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        expected
    );
    start.elapsed()
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times[times.len() / 2]
}

async fn median_scan(tasks: &[FileScanTask], row_selection: bool, warmup: bool) -> Duration {
    if warmup {
        let _ = collect_timed(tasks, row_selection).await;
    }
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        times.push(collect_timed(tasks, row_selection).await.0);
    }
    median(&mut times)
}

#[tokio::test]
#[ignore]
async fn perf_page_prune_cost_breakdown() {
    let tmp = tmpdir();
    let schema = perf_schema();
    let paths = write_perf_files(&tmp).await;

    let mut min_pages = usize::MAX;
    for p in &paths {
        let metadata = file_metadata(p);
        assert_eq!(metadata.num_row_groups(), 1, "fixture wants one row group");
        let pages = metadata.offset_index().expect("offset index")[0][0]
            .page_locations()
            .len();
        min_pages = min_pages.min(pages);
    }
    assert!(
        min_pages >= 8,
        "fixture wants several pages, got {min_pages}"
    );
    println!("fixture: {FILES} files x {ROWS} rows, min pages/chunk {min_pages}");

    let nonprunable = bound(
        &schema,
        Reference::new("category").equal_to(Datum::string("cat_7")),
    );
    let prunable = bound(&schema, Reference::new("id").equal_to(Datum::long(12_345)));
    let tasks_np = perf_tasks(&paths, &schema, Some(nonprunable.clone()));
    let tasks_p = perf_tasks(&paths, &schema, Some(prunable.clone()));
    let tasks_unfiltered = perf_tasks(&paths, &schema, None);

    let mut eval_total = Duration::ZERO;
    let mut selections = Vec::with_capacity(FILES);
    let mut all_keep = true;
    for (p, t) in paths.iter().zip(&tasks_np) {
        let metadata = file_metadata(p);
        let map = field_id_map(&metadata);
        let start = Instant::now();
        let selection = ArrowReader::get_row_selection_for_filter_predicate(
            t.predicate.as_deref().expect("predicate"),
            &metadata,
            &None,
            &map,
            &schema,
        )
        .expect("eval");
        eval_total += start.elapsed();
        let selection = selection.expect("selection present");
        if selected_rows(&selection) != ROWS {
            all_keep = false;
        }
        selections.push(selection);
    }
    println!(
        "(a) evaluator total {eval_total:?} over {FILES} files ({:?}/file)",
        eval_total / FILES as u32
    );
    println!("(b) nonprunable selection keeps every row: {all_keep}");

    let mut decode_sel = Vec::new();
    let mut decode_none = Vec::new();
    let mut decode_no_index = Vec::new();
    let mut decode_filter_index = Vec::new();
    let mut decode_filter_no_index = Vec::new();
    for _ in 0..REPS {
        for (i, p) in paths.iter().take(DECODE_FILES).enumerate() {
            decode_sel.push(decode_one(p, Some(selections[i].clone()), true, false).await);
            decode_none.push(decode_one(p, None, true, false).await);
            decode_no_index.push(decode_one(p, None, false, false).await);
            decode_filter_index.push(decode_one(p, None, true, true).await);
            decode_filter_no_index.push(decode_one(p, None, false, true).await);
        }
    }
    let decode_sel = median(&mut decode_sel);
    let decode_none = median(&mut decode_none);
    let decode_no_index = median(&mut decode_no_index);
    let decode_filter_index = median(&mut decode_filter_index);
    let decode_filter_no_index = median(&mut decode_filter_no_index);
    println!(
        "(c) decode/file: all-select RowSelection {decode_sel:?}, no selection {decode_none:?}, no index {decode_no_index:?}, filter+index {decode_filter_index:?}, filter+no index {decode_filter_no_index:?}"
    );

    let p0 = &paths[0];
    let mut parse_optional = Vec::new();
    let mut parse_skip = Vec::new();
    for _ in 0..REPS {
        let start = Instant::now();
        let file = File::open(p0).expect("open");
        let m = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Optional)
            .with_column_index_policy(PageIndexPolicy::Optional)
            .with_offset_index_policy(PageIndexPolicy::Optional)
            .parse_and_finish(&file)
            .expect("parse");
        parse_optional.push(start.elapsed());
        assert!(m.offset_index().is_some());
        let start = Instant::now();
        let file = File::open(p0).expect("open");
        let m = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Skip)
            .with_column_index_policy(PageIndexPolicy::Skip)
            .with_offset_index_policy(PageIndexPolicy::Skip)
            .parse_and_finish(&file)
            .expect("parse");
        parse_skip.push(start.elapsed());
        assert!(m.offset_index().is_none());
    }
    println!(
        "(d) footer+index parse: Optional {:?}, Skip {:?}",
        median(&mut parse_optional),
        median(&mut parse_skip)
    );

    let (on_rows, off_rows) = (
        collect_timed(&tasks_np, true).await.1,
        collect_timed(&tasks_np, false).await.1,
    );
    assert_eq!(on_rows, off_rows, "ON and OFF must return the same rows");

    for on in [true, false] {
        let (io, _calls, ranges) = recording_io();
        let _ = collect_with_io(tasks_np[0].clone(), on, io, 512 * 1024).await;
        let recorded = ranges.lock().expect("ranges");
        let total_bytes: u64 = recorded.iter().map(|(_, r)| r.end - r.start).sum();
        println!(
            "single-file reads on={on}: {} calls, {total_bytes} B, ranges {:?}",
            recorded.len(),
            recorded
        );
        drop(recorded);
        let metadata = file_metadata(&paths[0]);
        let _ = index_byte_ranges(&metadata);
    }

    let on = median_scan(&tasks_np, true, false).await;
    let off = median_scan(&tasks_np, false, false).await;
    let unfiltered = median_scan(&tasks_unfiltered, true, false).await;
    println!(
        "scan nonprunable: ON {on:?} OFF {off:?} ratio {:.3} (unfiltered {:?})",
        on.as_secs_f64() / off.as_secs_f64(),
        unfiltered
    );

    let on_p = median_scan(&tasks_p, true, true).await;
    let off_p = median_scan(&tasks_p, false, false).await;
    println!(
        "scan prunable: ON {on_p:?} OFF {off_p:?} ratio {:.3}",
        on_p.as_secs_f64() / off_p.as_secs_f64()
    );
}
