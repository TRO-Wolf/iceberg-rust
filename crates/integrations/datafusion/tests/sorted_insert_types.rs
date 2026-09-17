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

use std::sync::Arc;

use anyhow::Result;
use datafusion::arrow::array::{
    Array, BinaryArray, BooleanArray, Decimal128Array, Int32Array, Int64Array, RecordBatch,
    StringArray, TimestampMicrosecondArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use datafusion::datasource::MemTable;
use iceberg::spec::{
    NestedField, NullOrder, PrimitiveType, Schema, SortDirection, SortOrder, Transform, Type,
};
use iceberg::transform::create_transform_function;
use sorted_insert_shared::{
    Fixture, default_order_id, fixture, live_files, read_binary_column, read_boolean_column,
    read_decimal_column, read_long_column, read_nullable_long_column, read_string_column,
    read_timestamp_column, run_insert, sort_field, unpartitioned_spec,
};

fn single_schema(field: NestedField) -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![field.into()])
        .build()
        .expect("single column schema")
}

fn optional_primitive(id: i32, name: &str, primitive: PrimitiveType) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(primitive))
}

async fn insert_one_column(
    namespace: &str,
    schema: Schema,
    arrow_schema: Arc<ArrowSchema>,
    batch: RecordBatch,
    order: SortOrder,
    select: &str,
) -> Result<(Fixture, Vec<(String, Option<i32>)>)> {
    let fixture = fixture(namespace, "t", schema, unpartitioned_spec(), Some(order), 1).await?;
    let source = MemTable::try_new(arrow_schema, vec![vec![batch]]).expect("single column source");
    run_insert(&fixture, source, select).await?;
    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    for (_, stamp) in &files {
        assert_eq!(
            *stamp,
            Some(default_order_id(&fixture).await?),
            "data file stamps the default sort order id"
        );
    }
    Ok((fixture, files))
}

#[tokio::test]
async fn insert_into_string_order_sorts_utf8_bytes() -> Result<()> {
    let schema = single_schema(optional_primitive(1, "v", PrimitiveType::String));
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Utf8,
        true,
    )]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(StringArray::from(
        vec![Some("𝄞"), Some("b"), None, Some("ä"), Some("a")],
    ))])
    .expect("string batch");
    let (_fixture, files) = insert_one_column(
        "sorted_string",
        schema,
        arrow_schema,
        batch,
        order,
        "SELECT v FROM source",
    )
    .await?;
    assert_eq!(
        read_string_column(&files[0].0, 0),
        vec![
            None,
            Some("a".to_string()),
            Some("b".to_string()),
            Some("ä".to_string()),
            Some("𝄞".to_string()),
        ],
        "strings sort in UTF-8 byte order with NULLS FIRST"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_decimal_order_sorts_numeric() -> Result<()> {
    let schema = single_schema(optional_primitive(1, "v", PrimitiveType::Decimal {
        precision: 10,
        scale: 2,
    }));
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::Last,
        ))
        .build(&schema)?;
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Decimal128(10, 2),
        true,
    )]));
    let values = Decimal128Array::from(vec![Some(1025), Some(-200), None, Some(150), Some(0)])
        .with_precision_and_scale(10, 2)
        .expect("decimal precision");
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(values)]).expect("decimal batch");
    let (_fixture, files) = insert_one_column(
        "sorted_decimal",
        schema,
        arrow_schema,
        batch,
        order,
        "SELECT v FROM source",
    )
    .await?;
    assert_eq!(
        read_decimal_column(&files[0].0, 0),
        vec![Some(-200), Some(0), Some(150), Some(1025), None],
        "decimals sort numerically with NULLS LAST"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_timestamptz_order_sorts_instant() -> Result<()> {
    let schema = single_schema(optional_primitive(1, "v", PrimitiveType::Timestamptz));
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        true,
    )]));
    let values =
        TimestampMicrosecondArray::from(vec![Some(3_000_000_000), None, Some(0), Some(-1_000_000)])
            .with_timezone("UTC");
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(values)])
        .expect("timestamptz batch");
    let (_fixture, files) = insert_one_column(
        "sorted_timestamptz",
        schema,
        arrow_schema,
        batch,
        order,
        "SELECT v FROM source",
    )
    .await?;
    assert_eq!(
        read_timestamp_column(&files[0].0, 0),
        vec![None, Some(-1_000_000), Some(0), Some(3_000_000_000)],
        "timestamptz sorts by instant with NULLS FIRST"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_boolean_order_sorts_false_first() -> Result<()> {
    let schema = single_schema(optional_primitive(1, "v", PrimitiveType::Boolean));
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::Last,
        ))
        .build(&schema)?;
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Boolean,
        true,
    )]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(BooleanArray::from(
        vec![Some(true), Some(false), None, Some(false), Some(true)],
    ))])
    .expect("boolean batch");
    let (_fixture, files) = insert_one_column(
        "sorted_boolean",
        schema,
        arrow_schema,
        batch,
        order,
        "SELECT v FROM source",
    )
    .await?;
    assert_eq!(
        read_boolean_column(&files[0].0, 0),
        vec![Some(false), Some(false), Some(true), Some(true), None],
        "booleans sort false first with NULLS LAST"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_binary_order_sorts_unsigned_bytes() -> Result<()> {
    let schema = single_schema(optional_primitive(1, "v", PrimitiveType::Binary));
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Binary,
        true,
    )]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(BinaryArray::from(
        vec![
            Some(b"\xff".as_slice()),
            Some(b"\x00".as_slice()),
            Some(b"\x00\xff".as_slice()),
            None,
            Some(b"a".as_slice()),
        ],
    ))])
    .expect("binary batch");
    let (_fixture, files) = insert_one_column(
        "sorted_binary",
        schema,
        arrow_schema,
        batch,
        order,
        "SELECT v FROM source",
    )
    .await?;
    assert_eq!(
        read_binary_column(&files[0].0, 0),
        vec![
            None,
            Some(vec![0x00]),
            Some(vec![0x00, 0xff]),
            Some(b"a".to_vec()),
            Some(vec![0xff]),
        ],
        "binaries sort as unsigned bytes with NULLS FIRST"
    );
    Ok(())
}

fn id_string_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "s", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("id string schema")
}

fn id_string_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("s", DataType::Utf8, true),
    ]))
}

#[tokio::test]
async fn insert_into_truncate_string_order_sorts_by_prefix() -> Result<()> {
    let schema = id_string_schema();
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Truncate(2),
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .with_sort_field(sort_field(
            2,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let fixture = fixture(
        "sorted_truncate_string",
        "t",
        schema,
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let batch = RecordBatch::try_new(id_string_arrow_schema(), vec![
        Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        Arc::new(StringArray::from(vec![
            Some("bca"),
            Some("abc"),
            Some("ablaze"),
            None,
        ])),
    ])
    .expect("truncate string batch");
    let source = MemTable::try_new(id_string_arrow_schema(), vec![vec![batch]])?;
    run_insert(&fixture, source, "SELECT id, s FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    assert_eq!(
        read_long_column(&files[0].0, 0),
        vec![4, 2, 3, 1],
        "rows sort by the 2-char prefix, ties by the full string, null first"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_truncate_decimal_order_sorts_by_truncated_value() -> Result<()> {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(
                2,
                "d",
                Type::Primitive(PrimitiveType::Decimal {
                    precision: 10,
                    scale: 2,
                }),
            )
            .into(),
        ])
        .build()
        .expect("id decimal schema");
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Truncate(500),
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .with_sort_field(sort_field(
            1,
            Transform::Identity,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("d", DataType::Decimal128(10, 2), true),
    ]));
    let decimals = Decimal128Array::from(vec![Some(1234), Some(1789), Some(1105), None])
        .with_precision_and_scale(10, 2)
        .expect("decimal precision");
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        Arc::new(decimals),
    ])
    .expect("truncate decimal batch");
    let fixture = fixture(
        "sorted_truncate_decimal",
        "t",
        schema,
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let source = MemTable::try_new(arrow_schema, vec![vec![batch]])?;
    run_insert(&fixture, source, "SELECT id, d FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let ids = read_long_column(&files[0].0, 0);
    let truncated = create_transform_function(&Transform::Truncate(500))?
        .transform(Arc::new(
            Decimal128Array::from(read_decimal_column(&files[0].0, 1))
                .with_precision_and_scale(10, 2)
                .expect("decimal precision"),
        ))?
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("truncated decimals")
        .clone();
    assert_eq!(ids[0], 4, "null truncates to null and sorts first");
    let mut previous = (i128::MIN, i64::MIN);
    for (row, id) in ids.iter().enumerate().skip(1) {
        let key = (truncated.value(row), *id);
        assert!(
            key >= previous,
            "row {row} breaks (truncate, id) order: {key:?} after {previous:?}"
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

fn id_timestamp_schema(primitive: PrimitiveType) -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "ts", Type::Primitive(primitive)).into(),
        ])
        .build()
        .expect("id timestamp schema")
}

fn id_timestamp_arrow_schema(tz: bool) -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, tz.then(|| "UTC".into())),
            false,
        ),
    ]))
}

#[tokio::test]
async fn insert_into_day_order_sorts_by_calendar_day() -> Result<()> {
    let schema = id_timestamp_schema(PrimitiveType::Timestamp);
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Day,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let arrow_schema = id_timestamp_arrow_schema(false);
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int64Array::from(vec![1, 2, 3])),
        Arc::new(TimestampMicrosecondArray::from(vec![
            1_704_276_000_000_000,
            1_704_150_000_000_000,
            1_704_155_400_000_000,
        ])),
    ])
    .expect("day batch");
    let fixture = fixture(
        "sorted_day",
        "t",
        schema,
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let source = MemTable::try_new(arrow_schema, vec![vec![batch]])?;
    run_insert(&fixture, source, "SELECT id, ts FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    assert_eq!(
        read_long_column(&files[0].0, 0),
        vec![2, 3, 1],
        "rows sort by calendar day: Jan 1, Jan 2, Jan 3"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_hour_order_sorts_by_calendar_hour() -> Result<()> {
    let schema = id_timestamp_schema(PrimitiveType::Timestamptz);
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Hour,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let arrow_schema = id_timestamp_arrow_schema(true);
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int64Array::from(vec![1, 2, 3])),
        Arc::new(
            TimestampMicrosecondArray::from(vec![
                1_704_077_200_000_000,
                1_704_070_000_000_000,
                1_704_073_600_000_000,
            ])
            .with_timezone("UTC"),
        ),
    ])
    .expect("hour batch");
    let fixture = fixture(
        "sorted_hour",
        "t",
        schema,
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let source = MemTable::try_new(arrow_schema, vec![vec![batch]])?;
    run_insert(&fixture, source, "SELECT id, ts FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    assert_eq!(
        read_long_column(&files[0].0, 0),
        vec![2, 3, 1],
        "rows sort by calendar hour: 01:00, 02:00, 03:00"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}

#[tokio::test]
async fn insert_into_bucket_order_sorts_null_first_through_transform() -> Result<()> {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "g", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .expect("nullable bucket schema");
    let order = SortOrder::builder()
        .with_sort_field(sort_field(
            2,
            Transform::Bucket(4),
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .build(&schema)?;
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("g", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int64Array::from(vec![1, 2, 3])),
        Arc::new(Int64Array::from(vec![Some(10), None, Some(3)])),
    ])
    .expect("null bucket batch");
    let fixture = fixture(
        "sorted_bucket_null",
        "t",
        schema,
        unpartitioned_spec(),
        Some(order),
        1,
    )
    .await?;
    let source = MemTable::try_new(arrow_schema, vec![vec![batch]])?;
    run_insert(&fixture, source, "SELECT id, g FROM source").await?;

    let files = live_files(&fixture).await?;
    assert_eq!(files.len(), 1, "one writer stream writes one file");
    let ids = read_long_column(&files[0].0, 0);
    assert_eq!(
        ids[0], 2,
        "null survives the bucket transform and sorts first"
    );
    let groups: Vec<i64> = read_nullable_long_column(&files[0].0, 1)[1..]
        .iter()
        .map(|group| group.expect("non-null tail row"))
        .collect();
    assert_eq!(groups.len(), 2, "both non-null rows land after the null");
    let buckets = create_transform_function(&Transform::Bucket(4))?
        .transform(Arc::new(Int64Array::from(groups)))?
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("bucket values are Int32")
        .clone();
    assert!(
        buckets.value(0) <= buckets.value(1),
        "non-null tail stays bucket-sorted after the null"
    );
    assert_eq!(
        files[0].1,
        Some(default_order_id(&fixture).await?),
        "data file stamps the default sort order id"
    );
    Ok(())
}
