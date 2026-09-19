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
use std::fs::File;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};
use parquet::file::properties::{EnabledStatistics, WriterProperties};

use super::page_prune_fixture::*;
use super::reader::ArrowReaderBuilder;
use crate::expr::Reference;
use crate::io::FileIO;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_FILE, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_FIELD_ID_POS,
    RESERVED_FIELD_ID_ROW_ID,
};
use crate::scan::FileScanTaskStream;
use crate::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Type};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

#[tokio::test]
async fn w_data_file_writer_emits_column_and_offset_index() {
    let tmp = tmpdir();
    let file_io = FileIO::new_with_fs();
    let location_gen = DefaultLocationGenerator::with_data_location(
        tmp.path().to_str().expect("utf8").to_string(),
    );
    let file_name_gen =
        DefaultFileNameGenerator::new("w".to_string(), None, DataFileFormat::Parquet);
    let schema = iceberg_schema(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::required(2, "s", Type::Primitive(PrimitiveType::String)),
    ]);
    let parquet_builder =
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone());
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
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, false, 1),
        field("s", DataType::Utf8, false, 2),
    ]));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from((0..400).collect::<Vec<i32>>())) as ArrayRef,
        Arc::new(StringArray::from(
            (0..400).map(|i| format!("v{i}")).collect::<Vec<String>>(),
        )) as ArrayRef,
    ])
    .expect("batch");
    writer.write(batch).await.expect("write");
    let data_files = writer.close().await.expect("close");
    assert_eq!(data_files.len(), 1);
    let metadata = file_metadata(&data_files[0].file_path);
    let column_index = metadata.column_index().expect("column index");
    let offset_index = metadata.offset_index().expect("offset index");
    assert_eq!(column_index.len(), metadata.num_row_groups());
    assert_eq!(offset_index.len(), metadata.num_row_groups());
    for group in 0..metadata.num_row_groups() {
        let columns = metadata.row_group(group).columns().len();
        assert_eq!(column_index[group].len(), columns);
        assert_eq!(offset_index[group].len(), columns);
    }
}

#[tokio::test]
async fn m_filtered_scan_succeeds_without_any_page_index() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "no-index.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "id",
        DataType::Int32,
        false,
        1,
    )]));
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(ids.clone())) as ArrayRef,
        ])
        .expect("batch");
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .set_offset_index_disabled(true)
        .set_statistics_enabled(EnabledStatistics::None)
        .build();
    write_parquet(&data_path, arrow_schema, &[batch], props);
    let metadata = file_metadata(&data_path);
    assert!(metadata.column_index().is_none() || metadata.offset_index().is_none());
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let rows = on_off(task(&data_path, schema, &[1], Some(predicate))).await;
    assert_eq!(rows.len(), ROWS - 256);
}

#[tokio::test]
async fn m_filtered_scan_succeeds_without_offset_index() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "no-offset-index.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "id",
        DataType::Int32,
        false,
        1,
    )]));
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(ids.clone())) as ArrayRef,
        ])
        .expect("batch");
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .set_offset_index_disabled(true)
        .build();
    write_parquet(&data_path, arrow_schema, &[batch], props);
    let metadata = file_metadata(&data_path);
    assert!(metadata.offset_index().is_none());
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let rows = on_off(task(&data_path, schema, &[1], Some(predicate))).await;
    assert_eq!(rows.len(), ROWS - 256);
}

#[tokio::test]
async fn m_filtered_scan_succeeds_with_chunk_only_statistics() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "chunk-stats.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "id",
        DataType::Int32,
        false,
        1,
    )]));
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(ids.clone())) as ArrayRef,
        ])
        .expect("batch");
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .build();
    write_parquet(&data_path, arrow_schema, &[batch], props);
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let rows = on_off(task(&data_path, schema, &[1], Some(predicate))).await;
    assert_eq!(rows.len(), ROWS - 256);
}

#[tokio::test]
async fn m_prefetched_footer_without_index_scans_filtered_rows() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let file = File::open(&data_path).expect("open");
    let metadata = ParquetMetaDataReader::new()
        .with_page_index_policy(PageIndexPolicy::Skip)
        .parse_and_finish(&file)
        .expect("metadata");
    assert!(metadata.column_index().is_none());
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let mut t = task(&data_path, schema, &[1], Some(predicate));
    let reader = ArrowReaderBuilder::new(FileIO::new_with_fs())
        .with_batch_size(37)
        .with_row_group_filtering_enabled(true)
        .with_row_selection_enabled(true)
        .with_prefetched_parquet_metadata(HashMap::from([(
            t.data_file_path.clone(),
            Arc::new(metadata),
        )]))
        .build();
    let on = reader
        .read(Box::pin(futures::stream::iter(vec![Ok(t.clone())])) as FileScanTaskStream)
        .expect("read")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect");
    t.predicate = None;
    let unfiltered_rows = dump(&collect(t, false).await);
    let expected: Vec<Vec<String>> = unfiltered_rows
        .into_iter()
        .filter(|row| row[0].parse::<i64>().expect("id") >= 256)
        .collect();
    assert_eq!(dump(&on), expected);
}

#[tokio::test]
async fn l_row_lineage_columns_match_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let mut t = task(
        &data_path,
        schema,
        &[
            1,
            RESERVED_FIELD_ID_ROW_ID,
            RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
        ],
        Some(predicate),
    );
    t.first_row_id = Some(1000);
    t.file_sequence_number = Some(9);
    let on = collect(t.clone(), true).await;
    let off = collect(t, false).await;
    assert_eq!(dump(&on), dump(&off));
    let row_id_col = on[0].schema().index_of("_row_id").expect("_row_id column");
    let row_ids = i64_column(&on, row_id_col);
    let expected: Vec<Option<i64>> = (256..ROWS as i64).map(|p| Some(1000 + p)).collect();
    assert_eq!(row_ids, expected);
    let seq_col = on[0]
        .schema()
        .index_of("_last_updated_sequence_number")
        .expect("seq column");
    for value in i64_column(&on, seq_col) {
        assert_eq!(value, Some(9));
    }
}

#[tokio::test]
async fn l_last_updated_sequence_number_alone_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let mut t = task(
        &data_path,
        schema,
        &[1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER],
        Some(predicate),
    );
    t.first_row_id = Some(1000);
    t.file_sequence_number = Some(9);
    let rows = on_off(t).await;
    assert_eq!(rows.len(), ROWS - 256);
    assert!(rows.iter().all(|row| row[1] == "9"));
}

#[tokio::test]
async fn l_pos_and_file_columns_match_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let t = task(
        &data_path,
        schema,
        &[1, RESERVED_FIELD_ID_POS, RESERVED_FIELD_ID_FILE],
        Some(predicate),
    );
    let on = collect(t.clone(), true).await;
    let off = collect(t, false).await;
    assert_eq!(dump(&on), dump(&off));
    let pos_col = on[0].schema().index_of("_pos").expect("_pos column");
    let positions = i64_column(&on, pos_col);
    let expected: Vec<Option<i64>> = (256..ROWS as i64).map(Some).collect();
    assert_eq!(positions, expected);
    let file_col = on[0].schema().index_of("_file").expect("_file column");
    for row in dump(&on) {
        assert_eq!(row[file_col], data_path);
    }
}

#[tokio::test]
async fn l_stored_row_lineage_columns_match_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, false, 1),
        field("_row_id", DataType::Int64, true, RESERVED_FIELD_ID_ROW_ID),
        field(
            "_last_updated_sequence_number",
            DataType::Int64,
            true,
            RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
        ),
    ]));
    let row_ids: Vec<Option<i64>> = ids.iter().map(|i| Some(5000 + i64::from(*i))).collect();
    let seqs: Vec<Option<i64>> = ids.iter().map(|_| Some(42)).collect();
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(ids.clone())) as ArrayRef,
        Arc::new(Int64Array::from(row_ids)) as ArrayRef,
        Arc::new(Int64Array::from(seqs)) as ArrayRef,
    ])
    .expect("batch");
    write_parquet(&data_path, arrow_schema, &[batch], page_props());
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let mut t = task(
        &data_path,
        schema,
        &[
            1,
            RESERVED_FIELD_ID_ROW_ID,
            RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
        ],
        Some(predicate),
    );
    t.first_row_id = Some(1000);
    t.file_sequence_number = Some(9);
    let on = collect(t.clone(), true).await;
    let off = collect(t, false).await;
    assert_eq!(dump(&on), dump(&off));
    let row_id_col = on[0].schema().index_of("_row_id").expect("_row_id column");
    let stored = i64_column(&on, row_id_col);
    let expected: Vec<Option<i64>> = (256..ROWS as i64).map(|p| Some(5000 + p)).collect();
    assert_eq!(stored, expected);
    let seq_col = on[0]
        .schema()
        .index_of("_last_updated_sequence_number")
        .expect("seq column");
    for value in i64_column(&on, seq_col) {
        assert_eq!(value, Some(42));
    }
}

#[tokio::test]
async fn d_position_deletes_inside_kept_and_skipped_pages() {
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
    let metadata = file_metadata(&data_path);
    assert_page_count(&metadata, 0, 0, NUM_PAGES - 1);
    let selection = page_selection(&metadata, &schema, &predicate, &None).expect("selection");
    assert_prunes(&selection);
    let delete = write_pos_delete_file(&del_path, &data_path, &[10, 70, 300, 500]);
    let t = with_deletes(task(&data_path, schema, &[1], Some(predicate)), vec![
        delete,
    ]);
    let rows = on_off(t).await;
    assert_eq!(rows.len(), ROWS - 256 - 2);
}

#[tokio::test]
async fn d_equality_deletes_null_key_and_nonkeyset() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let del_path = path(&tmp, "eq-deletes.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let strings: Vec<Option<String>> = ids.iter().map(|_| Some("x".to_string())).collect();
    write_id_s_pages(&data_path, &ids, &strings);
    let schema = iceberg_schema(vec![
        NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "s", Type::Primitive(PrimitiveType::String)),
    ]);
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let delete = write_eq_delete_file(&del_path, &[Some(300), None], vec![1]);
    let t = with_deletes(task(&data_path, schema, &[2], Some(predicate)), vec![
        delete,
    ]);
    let rows = on_off(t).await;
    assert_eq!(rows.len(), ROWS - 256 - 1);
}

#[tokio::test]
async fn d_equality_deletes_keyset_path() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let del_path = path(&tmp, "eq-deletes.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let delete = write_eq_delete_file(&del_path, &[Some(300), Some(500)], vec![1]);
    let t = with_deletes(task(&data_path, schema, &[1], Some(predicate)), vec![
        delete,
    ]);
    let rows = on_off(t).await;
    assert_eq!(rows.len(), ROWS - 256 - 2);
}

#[tokio::test]
async fn d_deletion_vector_inside_kept_and_skipped_pages() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let dv_path = path(&tmp, "deletes.puffin");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let file_io = FileIO::new_with_fs();
    let delete = write_dv_delete_file(&file_io, &dv_path, &data_path, &[5, 100, 300, 500]).await;
    let t = with_deletes(task(&data_path, schema, &[1], Some(predicate)), vec![
        delete,
    ]);
    let rows = on_off(t).await;
    assert_eq!(rows.len(), ROWS - 256 - 2);
}
