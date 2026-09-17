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
use datafusion::arrow::array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::MemTable;
use iceberg::spec::{
    NestedField, NullOrder, PrimitiveType, Schema, SortDirection, SortOrder, TableProperties,
    Transform, Type, UnboundPartitionSpec,
};
use iceberg::transform::create_transform_function;
use sorted_insert_shared::{
    default_order_id, fixture, fixture_with_props, id_arrow_schema, id_batches, id_schema,
    live_files, read_int_column, read_long_column, read_nullable_long_column, run_insert,
    shuffled_ids, sort_field, unpartitioned_spec, writer_input_is_sort,
};

fn nulls_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "b", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("nulls schema")
}

fn nulls_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int64, true),
    ]))
}

#[tokio::test]
async fn insert_into_table_with_asc_order_writes_one_sorted_file_with_order_stamp() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture(
        "sorted_asc",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let rows = read_long_column(&files[0].0, 0);
    assert_eq!(rows.len(), 1000);
    let mut expected = rows.clone();
    expected.sort_unstable();
    assert_eq!(rows, expected, "file rows are ascending by id");
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_table_with_desc_order_writes_descending_file() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Descending,
            NullOrder::Last,
        ))
        .build(&id_schema())?;
    let fixture = fixture(
        "sorted_desc",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let rows = read_long_column(&files[0].0, 0);
    assert_eq!(rows.len(), 1000);
    let mut expected = rows.clone();
    expected.sort_unstable_by(|left, right| right.cmp(left));
    assert_eq!(rows, expected, "file rows are descending by id");
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_unsorted_table_adds_no_sort_and_stamps_zero() -> Result<()> {
    let fixture = fixture(
        "sorted_none",
        "t",
        id_schema(),
        unpartitioned_spec(),
        None,
        1,
    )
    .await?;
    let plan = fixture
        .context
        .sql("INSERT INTO catalog.sorted_none.t SELECT CAST(1 AS BIGINT) AS id, 1 AS p")
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(
        writer_input_is_sort(&plan),
        Some(false),
        "no SortExec feeds the write when the table has no sort order"
    );
    fixture
        .context
        .sql("INSERT INTO catalog.sorted_none.t SELECT CAST(1 AS BIGINT) AS id, 1 AS p")
        .await?
        .collect()
        .await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1);
    for (_, stamp) in &files {
        assert_eq!(*stamp, Some(0), "unsorted table stamps order id 0");
    }
    Ok(())
}

#[tokio::test]
async fn insert_into_sorted_table_feeds_write_through_sort() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture(
        "sorted_plan",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let ids = shuffled_ids(100);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)]).expect("plan source");
    fixture
        .context
        .register_table("source", Arc::new(source))
        .expect("register plan source");
    let plan = fixture
        .context
        .sql("INSERT INTO catalog.sorted_plan.t SELECT id, p FROM source")
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(
        writer_input_is_sort(&plan),
        Some(true),
        "a SortExec feeds the write when the table declares a sort order"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_two_key_order_honours_direction_and_null_order() -> Result<()> {
    let schema = nulls_schema();
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::Last,
        ))
        .with_sort_field(sort_field(
            2,
            Transform::Identity,
            SortDirection::Descending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let fixture = fixture(
        "sorted_nulls",
        "t",
        schema,
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let rows: Vec<(Option<i32>, Option<i64>)> = vec![
        (Some(2), Some(1)),
        (None, Some(5)),
        (Some(1), None),
        (Some(0), Some(7)),
        (None, None),
        (Some(1), Some(3)),
        (Some(2), None),
        (Some(1), Some(1)),
    ];
    let batch = RecordBatch::try_new(nulls_arrow_schema(), vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|row| row.0).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.1).collect::<Vec<_>>(),
        )),
    ])
    .expect("nulls batch");
    let source = MemTable::try_new(nulls_arrow_schema(), vec![vec![batch]])?;
    run_insert(&fixture, source, "SELECT a, b FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let actual = read_int_column(&files[0].0, 0)
        .into_iter()
        .zip(read_nullable_long_column(&files[0].0, 1))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        vec![
            (Some(0), Some(7)),
            (Some(1), None),
            (Some(1), Some(3)),
            (Some(1), Some(1)),
            (Some(2), None),
            (Some(2), Some(1)),
            (None, None),
            (None, Some(5)),
        ],
        "file rows follow a ASC NULLS LAST, b DESC NULLS FIRST"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_partitioned_table_sorts_within_each_file() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "p", Transform::Identity)?
        .build();
    let fixture = fixture("sorted_part", "t", id_schema(), spec, Some(order), 1).await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 2, "one file per partition value");
    let expected_id = default_order_id(&fixture).await?;
    for (path, stamp) in &files {
        assert_eq!(*stamp, Some(expected_id), "data file stamps the order id");
        let rows = read_long_column(path, 0);
        assert_eq!(rows.len(), 500);
        let mut expected = rows.clone();
        expected.sort_unstable();
        assert_eq!(rows, expected, "file {path} rows are ascending by id");
        let parts = read_int_column(path, 1);
        assert!(
            parts.iter().all(|part| *part == parts[0]),
            "file {path} holds a single partition value"
        );
    }
    Ok(())
}

#[tokio::test]
async fn insert_into_partitioned_table_sorts_every_file_across_streams() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "p", Transform::Identity)?
        .build();
    let fixture = fixture("sorted_streams", "t", id_schema(), spec, Some(order), 4).await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert!(!files.is_empty(), "insert writes data files");
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
async fn insert_into_bucket_order_sorts_by_bucket_then_id() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Bucket(4),
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture(
        "sorted_bucket",
        "t",
        id_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let ids = shuffled_ids(1000);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let rows = read_long_column(&files[0].0, 0);
    assert_eq!(rows.len(), 1000);
    let buckets = create_transform_function(&Transform::Bucket(4))?
        .transform(Arc::new(Int64Array::from(rows.clone())))?;
    let buckets = buckets
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("bucket values are Int32");
    let mut previous = (i32::MIN, i64::MIN);
    for (row, id) in rows.iter().enumerate() {
        let key = (buckets.value(row), *id);
        assert!(
            key >= previous,
            "row {row} breaks (bucket, id) order: {key:?} after {previous:?}"
        );
        previous = key;
    }
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

fn float_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "k", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "f", Type::Primitive(PrimitiveType::Float)).into(),
        ])
        .build()
        .expect("float schema")
}

fn double_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "k", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "f", Type::Primitive(PrimitiveType::Double)).into(),
        ])
        .build()
        .expect("double schema")
}

fn float_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("f", DataType::Float32, true),
    ]))
}

fn double_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("f", DataType::Float64, true),
    ]))
}

fn spark_float_keys() -> Vec<Option<f32>> {
    vec![
        Some(f32::NAN),
        Some(-1.0),
        Some(f32::from_bits(0xFFC0_0000)),
        Some(1.0),
        Some(f32::INFINITY),
        Some(-0.0),
        Some(0.0),
        None,
        Some(f32::NEG_INFINITY),
    ]
}

fn spark_double_keys() -> Vec<Option<f64>> {
    vec![
        Some(f64::NAN),
        Some(-1.0),
        Some(f64::from_bits(0xFFF8_0000_0000_0000)),
        Some(1.0),
        Some(f64::INFINITY),
        Some(-0.0),
        Some(0.0),
        None,
        Some(f64::NEG_INFINITY),
    ]
}

fn float_source() -> MemTable {
    let batch = RecordBatch::try_new(float_arrow_schema(), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9])),
        Arc::new(Float32Array::from(spark_float_keys())),
    ])
    .expect("float batch");
    MemTable::try_new(float_arrow_schema(), vec![vec![batch]]).expect("float source")
}

fn double_source() -> MemTable {
    let batch = RecordBatch::try_new(double_arrow_schema(), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9])),
        Arc::new(Float64Array::from(spark_double_keys())),
    ])
    .expect("double batch");
    MemTable::try_new(double_arrow_schema(), vec![vec![batch]]).expect("double source")
}

fn read_key_order(path: &str) -> Vec<i32> {
    read_int_column(path, 0)
        .into_iter()
        .map(|key| key.expect("non-null k"))
        .collect()
}

#[tokio::test]
async fn insert_into_float_order_sorts_nan_last_asc() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&float_schema())?;
    let fixture = fixture(
        "sorted_float",
        "t",
        float_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    run_insert(&fixture, float_source(), "SELECT k, f FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let keys = read_key_order(&files[0].0);
    let mut nan_pair = keys[7..].to_vec();
    nan_pair.sort_unstable();
    assert_eq!(
        nan_pair,
        vec![1, 3],
        "both NaN rows sort last under ASC, sign and payload ignored"
    );
    assert_eq!(
        keys[..7],
        vec![8, 9, 2, 6, 7, 4, 5],
        "float ASC NULLS FIRST matches the Spark file order"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_double_order_sorts_nan_last_asc() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&double_schema())?;
    let fixture = fixture(
        "sorted_double",
        "t",
        double_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    run_insert(&fixture, double_source(), "SELECT k, f FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let keys = read_key_order(&files[0].0);
    let mut nan_pair = keys[7..].to_vec();
    nan_pair.sort_unstable();
    assert_eq!(
        nan_pair,
        vec![1, 3],
        "both NaN rows sort last under ASC, sign and payload ignored"
    );
    assert_eq!(
        keys[..7],
        vec![8, 9, 2, 6, 7, 4, 5],
        "double ASC NULLS FIRST matches the Spark file order"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_float_order_sorts_nan_first_desc() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Identity,
            SortDirection::Descending,
            NullOrder::Last,
        ))
        .build(&float_schema())?;
    let fixture = fixture(
        "sorted_float_desc",
        "t",
        float_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    run_insert(&fixture, float_source(), "SELECT k, f FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let keys = read_key_order(&files[0].0);
    let mut nan_pair = keys[..2].to_vec();
    nan_pair.sort_unstable();
    assert_eq!(
        nan_pair,
        vec![1, 3],
        "both NaN rows sort first under DESC, sign and payload ignored"
    );
    assert_eq!(
        keys[2..],
        vec![5, 4, 7, 6, 2, 9, 8],
        "float DESC NULLS LAST mirrors the ASC Spark order"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_double_order_sorts_nan_first_desc() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Identity,
            SortDirection::Descending,
            NullOrder::Last,
        ))
        .build(&double_schema())?;
    let fixture = fixture(
        "sorted_double_desc",
        "t",
        double_schema(),
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    run_insert(&fixture, double_source(), "SELECT k, f FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let keys = read_key_order(&files[0].0);
    let mut nan_pair = keys[..2].to_vec();
    nan_pair.sort_unstable();
    assert_eq!(
        nan_pair,
        vec![1, 3],
        "both NaN rows sort first under DESC, sign and payload ignored"
    );
    assert_eq!(
        keys[2..],
        vec![5, 4, 7, 6, 2, 9, 8],
        "double DESC NULLS LAST mirrors the ASC Spark order"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_unknown_transform_order_writes_with_zero_stamp() -> Result<()> {
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Unknown,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&id_schema())?;
    let fixture = fixture_with_props(
        "sorted_unknown",
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
    let plan = fixture
        .context
        .sql("INSERT INTO catalog.sorted_unknown.t SELECT CAST(1 AS BIGINT) AS id, 1 AS p")
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(
        writer_input_is_sort(&plan),
        Some(false),
        "an unresolvable order adds no SortExec"
    );
    let ids = shuffled_ids(100);
    let source = MemTable::try_new(id_arrow_schema(), vec![id_batches(&ids)])?;
    run_insert(&fixture, source, "SELECT id, p FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1);
    for (_, stamp) in &files {
        assert_eq!(*stamp, Some(0), "unresolvable order stamps order id 0");
    }
    Ok(())
}
