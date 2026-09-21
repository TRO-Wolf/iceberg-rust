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

fn schema_renumbered_with_long_string() -> Schema {
    Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            NestedField::optional(100, "c_id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(101, "c_name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("build the renumbered schema")
}

fn renumbered_batch(schema: &Schema) -> RecordBatch {
    let arrow_schema: ArrowSchemaRef = Arc::new(
        crate::arrow::schema_to_arrow_schema(schema).expect("iceberg schema to arrow schema"),
    );
    let long = "a".repeat(40);
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![Some(1), Some(2), None])),
        Arc::new(StringArray::from(vec![
            Some(long.as_str()),
            Some("short"),
            None,
        ])),
    ];
    RecordBatch::try_new(arrow_schema, columns).expect("build the renumbered batch")
}

fn schema_required_child() -> Schema {
    Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![NestedField::optional(
            20,
            "st",
            Type::Struct(StructType::new(vec![
                NestedField::required(21, "x", Type::Primitive(PrimitiveType::Int)).into(),
            ])),
        )
        .into()])
        .build()
        .expect("build the required-child schema")
}

fn required_child_batch(schema: &Schema) -> RecordBatch {
    let arrow_schema: ArrowSchemaRef = Arc::new(
        crate::arrow::schema_to_arrow_schema(schema).expect("iceberg schema to arrow schema"),
    );
    let DataType::Struct(struct_fields) = arrow_schema.field(0).data_type().clone() else {
        panic!("field 0 must be a struct");
    };
    let child = Arc::new(Int32Array::from(vec![Some(7), Some(0)])) as ArrayRef;
    let parent = Arc::new(StructArray::new(
        struct_fields,
        vec![child],
        Some(arrow_buffer::NullBuffer::from(vec![true, false])),
    )) as ArrayRef;
    RecordBatch::try_new(arrow_schema, vec![parent]).expect("build the required-child batch")
}

#[tokio::test]
async fn test_the_data_file_carries_the_full_java_metric_set() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let written = all_primitives_batch(&schema);

    let (_path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()),
        &file_io,
        &location_gen,
        "metrics",
        std::slice::from_ref(&written),
    )
    .await;
    let data_file = files
        .into_iter()
        .next()
        .expect("one data file")
        .build()
        .expect("build the data file");

    assert_eq!(data_file.record_count(), 3);
    assert_eq!(data_file.file_format(), DataFileFormat::Orc);
    assert!(data_file.file_size_in_bytes() > 0);

    for field_id in 1..=16i32 {
        assert_eq!(
            data_file.value_counts().get(&field_id),
            Some(&3),
            "field {field_id} must carry a value count"
        );
        assert_eq!(
            data_file.null_value_counts().get(&field_id),
            Some(&1),
            "field {field_id} has exactly one null row"
        );
        assert!(
            data_file.column_sizes().contains_key(&field_id),
            "field {field_id} must carry a column size"
        );
    }

    assert_eq!(data_file.nan_value_counts().get(&4), Some(&1));
    assert_eq!(data_file.nan_value_counts().get(&5), Some(&1));
    assert_eq!(
        data_file.nan_value_counts().len(),
        2,
        "only FLOAT and DOUBLE carry nan_value_count"
    );

    for field_id in [7i32, 15, 16] {
        assert!(
            !data_file.lower_bounds().contains_key(&field_id),
            "ORC binary statistics carry no bound for field {field_id}"
        );
    }
    for field_id in [1i32, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14] {
        assert!(
            data_file.lower_bounds().contains_key(&field_id),
            "field {field_id} must carry a lower bound"
        );
        assert!(
            data_file.upper_bounds().contains_key(&field_id),
            "field {field_id} must carry an upper bound"
        );
    }

    assert_eq!(
        data_file.lower_bounds().get(&2).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::Int(i32::MIN))
    );
    assert_eq!(
        data_file.upper_bounds().get(&2).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::Int(42))
    );
    assert_eq!(
        data_file.lower_bounds().get(&4).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::Float(3.5f32.into())),
        "NaN must not poison the float bound"
    );
    assert_eq!(
        data_file.upper_bounds().get(&4).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::Float(3.5f32.into())),
        "NaN must not poison the float upper bound"
    );
    assert_eq!(
        data_file.lower_bounds().get(&5).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::Double(4.5f64.into())),
        "NaN must not poison the double lower bound"
    );
    assert_eq!(
        data_file.upper_bounds().get(&5).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::Double(4.5f64.into())),
        "NaN must not poison the double upper bound"
    );
    assert_eq!(
        data_file.split_offsets(),
        Some(&[3i64][..]),
        "the single stripe starts after the 3-byte ORC magic"
    );
}

#[tokio::test]
async fn test_metrics_config_none_drops_the_column_from_the_data_file() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let written = all_primitives_batch(&schema);
    let config = crate::spec::MetricsConfig::from_properties(&HashMap::from([(
        format!(
            "{}c_string",
            crate::spec::METRICS_MODE_COLUMN_CONF_PREFIX
        ),
        "none".to_string(),
    )]))
    .expect("parse the metrics config");

    let (_path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()).with_metrics_config(config),
        &file_io,
        &location_gen,
        "metrics_none",
        std::slice::from_ref(&written),
    )
    .await;
    let data_file = files
        .into_iter()
        .next()
        .expect("one data file")
        .build()
        .expect("build the data file");

    assert!(!data_file.value_counts().contains_key(&6));
    assert!(!data_file.null_value_counts().contains_key(&6));
    assert!(!data_file.column_sizes().contains_key(&6));
    assert!(!data_file.lower_bounds().contains_key(&6));
    assert!(data_file.value_counts().contains_key(&2));
}

#[tokio::test]
async fn test_the_data_file_keys_column_sizes_by_field_id() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_renumbered_with_long_string());
    let written = renumbered_batch(&schema);

    let (_path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()),
        &file_io,
        &location_gen,
        "metrics_renumbered",
        std::slice::from_ref(&written),
    )
    .await;
    let data_file = files
        .into_iter()
        .next()
        .expect("one data file")
        .build()
        .expect("build the data file");

    assert_eq!(data_file.record_count(), 3);
    for field_id in [100i32, 101] {
        assert!(
            data_file.column_sizes().contains_key(&field_id),
            "field {field_id} must carry a column size"
        );
        assert_eq!(
            data_file.value_counts().get(&field_id),
            Some(&3),
            "field {field_id} must carry a value count"
        );
    }
    for orc_index in [1i32, 2] {
        assert!(
            !data_file.column_sizes().contains_key(&orc_index),
            "ORC index {orc_index} must not appear as a field id key"
        );
    }
}

#[tokio::test]
async fn test_metrics_config_counts_keeps_counts_and_drops_bounds_on_the_data_file() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let written = all_primitives_batch(&schema);
    let config = crate::spec::MetricsConfig::from_properties(&HashMap::from([(
        crate::spec::METRICS_MODE_DEFAULT_KEY.to_string(),
        "counts".to_string(),
    )]))
    .expect("parse the metrics config");

    let (_path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()).with_metrics_config(config),
        &file_io,
        &location_gen,
        "metrics_counts",
        std::slice::from_ref(&written),
    )
    .await;
    let data_file = files
        .into_iter()
        .next()
        .expect("one data file")
        .build()
        .expect("build the data file");

    assert_eq!(data_file.value_counts().get(&2), Some(&3));
    assert_eq!(data_file.null_value_counts().get(&2), Some(&1));
    assert!(
        data_file.lower_bounds().is_empty(),
        "counts mode must drop every lower bound"
    );
    assert!(
        data_file.upper_bounds().is_empty(),
        "counts mode must drop every upper bound"
    );
}

#[tokio::test]
async fn test_the_data_file_truncates_string_bounds_to_sixteen() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_renumbered_with_long_string());
    let written = renumbered_batch(&schema);

    let (_path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()),
        &file_io,
        &location_gen,
        "metrics_truncate",
        std::slice::from_ref(&written),
    )
    .await;
    let data_file = files
        .into_iter()
        .next()
        .expect("one data file")
        .build()
        .expect("build the data file");

    let lower = data_file
        .lower_bounds()
        .get(&101)
        .expect("a string lower bound");
    match lower.literal() {
        crate::spec::PrimitiveLiteral::String(value) => assert_eq!(
            value.len(),
            16,
            "the default metrics mode truncates string bounds to 16"
        ),
        other => panic!("expected a string bound, got {other:?}"),
    }
    assert_eq!(
        data_file.upper_bounds().get(&101).map(Datum::literal),
        Some(&crate::spec::PrimitiveLiteral::String("short".to_string())),
        "the short upper bound passes through untruncated"
    );
}

#[tokio::test]
async fn test_the_data_file_counts_a_required_child_under_a_null_parent_as_null() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_required_child());
    let written = required_child_batch(&schema);

    let (_path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()),
        &file_io,
        &location_gen,
        "metrics_required_child",
        std::slice::from_ref(&written),
    )
    .await;
    let data_file = files
        .into_iter()
        .next()
        .expect("one data file")
        .build()
        .expect("build the data file");

    assert_eq!(data_file.record_count(), 2);
    assert_eq!(data_file.value_counts().get(&21), Some(&2));
    assert_eq!(
        data_file.null_value_counts().get(&21),
        Some(&1),
        "Java ORC reports 0 nulls here because required children have no PRESENT stream; this collector follows Java Parquet where a null parent makes the child null"
    );
}
