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

mod sorted_insert_shared;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use datafusion::arrow::array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StructArray};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use iceberg::spec::{
    NestedField, NullOrder, PrimitiveType, Schema, SortDirection, SortOrder, StructType,
    TableProperties, Transform, Type,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use sorted_insert_shared::{
    default_order_id, fixture, fixture_with_props, id_arrow_schema, id_batches, id_schema,
    live_files, live_row_count, read_long_column, run_insert, run_insert_overwrite, shuffled_ids,
    sort_field, unpartitioned_spec,
};

fn nested_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(
                1,
                "s",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(2, "x", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
        ])
        .build()
        .expect("nested schema")
}

fn nested_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![Field::new(
        "s",
        DataType::Struct(Fields::from(vec![Field::new("x", DataType::Int32, true)])),
        true,
    )]))
}

fn nested_batch(children: Vec<Option<i32>>, parent_valid: Vec<bool>) -> RecordBatch {
    let struct_array = StructArray::new(
        Fields::from(vec![Field::new("x", DataType::Int32, true)]),
        vec![Arc::new(Int32Array::from(children)) as ArrayRef],
        Some(NullBuffer::from(parent_valid)),
    );
    RecordBatch::try_new(nested_arrow_schema(), vec![Arc::new(struct_array)]).expect("nested batch")
}

fn read_struct_int_column(path: &str, index: usize, child: &str) -> Vec<Option<i32>> {
    let file = std::fs::File::open(sorted_insert_shared::local_path(path)).expect("open data file");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("parquet reader")
        .build()
        .expect("build reader");
    let mut values = Vec::new();
    for batch in reader {
        let batch = batch.expect("read batch");
        let parent = batch
            .column(index)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("struct column");
        let child_array = parent
            .column_by_name(child)
            .expect("struct child")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int child");
        for row in 0..batch.num_rows() {
            let value =
                (!parent.is_null(row) && !child_array.is_null(row)).then(|| child_array.value(row));
            values.push(value);
        }
    }
    values
}

#[tokio::test]
async fn insert_into_nested_identity_order_sorts_by_child() -> Result<()> {
    let schema = nested_schema();
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let fixture = fixture(
        "sorted_nested",
        "t",
        schema,
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let batch = nested_batch(vec![Some(30), Some(10), Some(999), Some(20)], vec![
        true, true, false, true,
    ]);
    let source = MemTable::try_new(nested_arrow_schema(), vec![vec![batch]])?;
    run_insert(&fixture, source, "SELECT s FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    assert_eq!(
        read_struct_int_column(&files[0].0, 0, "x"),
        vec![None, Some(10), Some(20), Some(30)],
        "rows sort by the nested child, a null parent sorts as null first"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

fn id_nested_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(
                2,
                "s",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(3, "x", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
        ])
        .build()
        .expect("id nested schema")
}

fn id_nested_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![Field::new("x", DataType::Int32, true)])),
            true,
        ),
    ]))
}

#[tokio::test]
async fn insert_into_two_key_order_with_nested_key_sorts_by_both() -> Result<()> {
    let schema = id_nested_schema();
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::Last,
        ))
        .with_sort_field(sort_field(
            3,
            Transform::Identity,
            SortDirection::Descending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let fixture = fixture(
        "sorted_nested_two_key",
        "t",
        schema,
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let struct_array = StructArray::new(
        Fields::from(vec![Field::new("x", DataType::Int32, true)]),
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(3), Some(1), None])) as ArrayRef],
        None,
    );
    let batch = RecordBatch::try_new(id_nested_arrow_schema(), vec![
        Arc::new(Int64Array::from(vec![2, 1, 1, 1])),
        Arc::new(struct_array),
    ])
    .expect("id nested batch");
    let source = MemTable::try_new(id_nested_arrow_schema(), vec![vec![batch]])?;
    run_insert(&fixture, source, "SELECT id, s FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let actual = read_long_column(&files[0].0, 0)
        .into_iter()
        .zip(read_struct_int_column(&files[0].0, 1, "x"))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        vec![(1, None), (1, Some(3)), (1, Some(1)), (2, Some(1)),],
        "file rows follow id ASC, then nested x DESC NULLS FIRST"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_sorted_table_with_rolling_split_keeps_every_file_sorted() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture_with_props(
        "sorted_rolling",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
        HashMap::from([
            (
                TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED.to_string(),
                "true".to_string(),
            ),
            (
                TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES.to_string(),
                "1024".to_string(),
            ),
        ]),
    )
    .await?;
    let ids = shuffled_ids(20_000);
    let partitions = ids.chunks(5_000).map(id_batches).collect::<Vec<_>>();
    let source = MemTable::try_new(id_arrow_schema(), partitions)?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert!(
        files.len() >= 2,
        "a tiny target file size rolls one sorted stream into several files"
    );
    assert_eq!(
        live_row_count(&fixture).await?,
        20_000,
        "every input row lands in exactly one file"
    );
    let expected_id = default_order_id(&fixture).await?;
    for (path, stamp) in &files {
        assert_eq!(*stamp, Some(expected_id), "data file stamps the order id");
        let rows = read_long_column(path, 0);
        let mut expected = rows.clone();
        expected.sort_unstable();
        assert_eq!(rows, expected, "file {path} rows are ascending by id");
    }
    Ok(())
}

#[tokio::test]
async fn insert_overwrite_into_sorted_table_writes_sorted_files() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture(
        "sorted_overwrite",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let seed = shuffled_ids(500);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&seed)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let fresh: Vec<i64> = shuffled_ids(100).iter().map(|id| id + 10_000).collect();
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&fresh)])?;
    run_insert_overwrite(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(
        live_row_count(&fixture).await?,
        100,
        "overwrite replaces the seed rows"
    );
    let expected_id = default_order_id(&fixture).await?;
    for (path, stamp) in &files {
        assert_eq!(*stamp, Some(expected_id), "data file stamps the order id");
        let rows = read_long_column(path, 0);
        let mut expected = rows.clone();
        expected.sort_unstable();
        assert_eq!(rows, expected, "file {path} rows are ascending by id");
        assert!(
            rows.iter().all(|id| *id >= 10_000),
            "file {path} holds only the overwrite rows"
        );
    }
    Ok(())
}

#[tokio::test]
async fn insert_into_sorted_table_with_fanout_disabled_writes_sorted_files() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture_with_props(
        "sorted_fanout_off",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
        HashMap::from([(
            TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED.to_string(),
            "false".to_string(),
        )]),
    )
    .await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(
        live_row_count(&fixture).await?,
        1000,
        "every input row lands in a file"
    );
    let expected_id = default_order_id(&fixture).await?;
    for (path, stamp) in &files {
        assert_eq!(*stamp, Some(expected_id), "data file stamps the order id");
        let rows = read_long_column(path, 0);
        let mut expected = rows.clone();
        expected.sort_unstable();
        assert_eq!(rows, expected, "file {path} rows are ascending by id");
    }
    Ok(())
}
