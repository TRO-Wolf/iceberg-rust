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
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::types::Int8Type;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, DictionaryArray, Int32Array, Int64Array,
    ListArray, MapArray, RecordBatch, StringArray, StringViewArray, StructArray,
    TimestampMicrosecondArray, make_array,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use crate::ErrorKind;
use crate::arrow::schema_to_arrow_schema;
use crate::io::FileIO;
use crate::spec::{
    DataFileFormat, Literal, NestedField, PrimitiveLiteral, PrimitiveType, Schema, Type,
};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::write_defaults::*;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

fn apply_write_defaults<'a>(
    schema: &Schema,
    batch: &'a RecordBatch,
) -> crate::Result<Cow<'a, RecordBatch>> {
    crate::writer::write_defaults::apply_write_defaults(
        schema,
        &Arc::new(schema_to_arrow_schema(schema)?),
        batch,
    )
}

fn id_meta(id: i32) -> HashMap<String, String> {
    HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())])
}

fn schema_id_and_name() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                .with_write_default(Literal::string("anon"))
                .into(),
        ])
        .build()
        .expect("schema")
}

fn id_only_batch() -> RecordBatch {
    let arrow_schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
    ]);
    RecordBatch::try_new(Arc::new(arrow_schema), vec![Arc::new(Int32Array::from(
        vec![1, 2, 3],
    ))])
    .expect("id-only batch")
}

fn complete_batch() -> RecordBatch {
    let arrow_schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
        Field::new("name", DataType::Utf8, true).with_metadata(id_meta(2)),
    ]);
    RecordBatch::try_new(Arc::new(arrow_schema), vec![
        Arc::new(Int32Array::from(vec![1, 2])),
        Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
    ])
    .expect("complete batch")
}

fn nested_ids_schema_with(element_required: bool) -> Schema {
    serde_json::from_str(&format!(
        r#"{{"type":"struct","schema-id":0,"fields":[
            {{"id":1,"name":"id","required":true,"type":"int"}},
            {{"id":2,"name":"nums","required":false,"type":{{"type":"list","element-id":3,"element":"int","element-required":{element_required}}}}},
            {{"id":4,"name":"pairs","required":false,"type":{{"type":"list","element-id":5,"element-required":false,"element":{{"type":"struct","fields":[{{"id":6,"name":"a","required":true,"type":"int"}},{{"id":7,"name":"b","required":false,"type":"string"}}]}}}}}},
            {{"id":8,"name":"props","required":false,"type":{{"type":"map","key-id":9,"key":"string","value-id":10,"value-required":false,"value":{{"type":"list","element-id":11,"element":"int","element-required":false}}}}}}
        ]}}"#
    ))
    .expect("nested schema")
}

pub(crate) fn nested_ids_schema() -> Schema {
    nested_ids_schema_with(false)
}

fn int_list(element_name: &str, rows: Vec<Option<Vec<Option<i32>>>>) -> ArrayRef {
    let element = Arc::new(Field::new(element_name, DataType::Int32, true));
    make_array(
        ListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(rows)
            .into_data()
            .into_builder()
            .data_type(DataType::List(element))
            .build()
            .expect("named element list"),
    )
}

fn boxed<A: Array + 'static>(array: A) -> ArrayRef {
    Arc::new(array)
}

pub(crate) fn nested_batch(element_name: &str, top_ids: bool) -> RecordBatch {
    let fld = |name: &str, data_type: DataType, nullable: bool| {
        Arc::new(Field::new(name, data_type, nullable))
    };
    let stamp = |field: Field, id: i32| {
        let meta = if top_ids { id_meta(id) } else { HashMap::new() };
        field.with_metadata(meta)
    };
    let nums = int_list(element_name, vec![
        Some(vec![Some(1), None]),
        None,
        Some(vec![]),
    ]);
    let pair_values = StructArray::from(vec![
        (
            fld("a", DataType::Int32, false),
            boxed(Int32Array::from(vec![10, 20])),
        ),
        (
            fld("b", DataType::Utf8, true),
            boxed(StringArray::from(vec![Some("x"), None])),
        ),
    ]);
    let pairs = ListArray::new(
        fld(element_name, pair_values.data_type().clone(), true),
        OffsetBuffer::new(vec![0, 1, 1, 2].into()),
        Arc::new(pair_values),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let inner = int_list(element_name, vec![
        Some(vec![Some(5), Some(6)]),
        Some(vec![]),
    ]);
    let entries = StructArray::from(vec![
        (
            fld("key", DataType::Utf8, false),
            boxed(StringArray::from(vec!["k1", "k2"])),
        ),
        (fld("value", inner.data_type().clone(), true), inner),
    ]);
    let props = MapArray::new(
        fld("entries", entries.data_type().clone(), false),
        OffsetBuffer::new(vec![0, 1, 2, 2].into()),
        entries,
        Some(NullBuffer::from(vec![true, true, false])),
        false,
    );
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            stamp(Field::new("id", DataType::Int32, false), 1),
            stamp(Field::new("nums", nums.data_type().clone(), true), 2),
            stamp(Field::new("pairs", pairs.data_type().clone(), true), 4),
            stamp(Field::new("props", props.data_type().clone(), true), 8),
        ])),
        vec![
            boxed(Int32Array::from(vec![1, 2, 3])),
            nums,
            Arc::new(pairs),
            Arc::new(props),
        ],
    )
    .expect("unstamped batch")
}

pub(crate) fn assert_nested_field_ids(schema: &ArrowSchema) {
    for (path, id) in [
        (&[1, 0][..], 3),
        (&[2, 0], 5),
        (&[2, 0, 0], 6),
        (&[2, 0, 1], 7),
        (&[3, 0, 0], 9),
        (&[3, 0, 1], 10),
        (&[3, 0, 1, 0], 11),
    ] {
        let mut field = schema.fields()[path[0]].as_ref();
        for &index in &path[1..] {
            field = match field.data_type() {
                DataType::List(child) | DataType::Map(child, _) if index == 0 => child.as_ref(),
                DataType::Struct(children) => children[index].as_ref(),
                _ => panic!("expected a nested field"),
            };
        }
        assert_eq!(batch_field_id(field), Some(id));
    }
}

#[test]
fn unstamped_nested_fields_are_relabelled_to_iceberg_types() {
    let schema = nested_ids_schema();
    let target = schema_to_arrow_schema(&schema).expect("arrow schema");
    for (element_name, column) in [("element", 1), ("item", 1), ("element", 2), ("item", 3)] {
        let batch = nested_batch(element_name, true);
        let filled = apply_write_defaults(&schema, &batch).expect("fill");
        assert!(matches!(filled, Cow::Owned(_)));
        assert_eq!(
            filled.column(column).data_type(),
            target.field(column).data_type()
        );
    }
    let batch = nested_batch("item", false);
    assert!(matches!(
        apply_write_defaults(&schema, &batch).expect("fill"),
        Cow::Owned(_)
    ));
}

fn int64_nums_batch() -> RecordBatch {
    let int64_list: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("element", DataType::Int64, true)),
        OffsetBuffer::new(vec![0, 1, 1, 1].into()),
        Arc::new(Int64Array::from(vec![9])),
        None,
    ));
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
            Field::new("nums", int64_list.data_type().clone(), true).with_metadata(id_meta(2)),
        ])),
        vec![boxed(Int32Array::from(vec![1, 2, 3])), int64_list],
    )
    .expect("int64 batch")
}

#[test]
fn incompatible_nested_data_is_data_invalid() {
    for (schema, batch) in [
        (nested_ids_schema(), int64_nums_batch()),
        (nested_ids_schema_with(true), nested_batch("element", true)),
    ] {
        assert_eq!(
            apply_write_defaults(&schema, &batch)
                .expect_err("must refuse")
                .kind(),
            ErrorKind::DataInvalid
        );
    }
}

#[test]
fn missing_optional_write_default_is_filled() {
    let schema = schema_id_and_name();
    let batch = id_only_batch();
    let filled = apply_write_defaults(&schema, &batch).expect("fill");
    let names = filled
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name col");
    assert_eq!(names.value(0), "anon");
    assert_eq!(names.value(1), "anon");
    assert_eq!(names.value(2), "anon");
}

#[test]
fn supplied_column_is_not_replaced() {
    let schema = schema_id_and_name();
    let batch = complete_batch();
    let filled = apply_write_defaults(&schema, &batch).expect("fill");
    let names = filled
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name col");
    assert_eq!(names.value(0), "a");
    assert_eq!(names.value(1), "b");
    assert!(matches!(filled, Cow::Borrowed(_)));
}

#[test]
fn missing_required_without_write_default_fails() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let err = apply_write_defaults(&schema, &id_only_batch()).expect_err("must fail");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.message().contains("write-default"),
        "got {}",
        err.message()
    );
}

#[test]
fn missing_optional_without_write_default_is_null() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let batch = id_only_batch();
    let filled = apply_write_defaults(&schema, &batch).expect("null fill");
    let names = filled
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name col");
    assert!(names.is_null(0));
    assert!(names.is_null(1));
    assert!(names.is_null(2));
}

#[test]
fn missing_required_write_default_is_filled() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String))
                .with_write_default(Literal::string("x"))
                .into(),
        ])
        .build()
        .expect("schema");
    let batch = id_only_batch();
    let filled = apply_write_defaults(&schema, &batch).expect("fill");
    let names = filled
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name col");
    assert_eq!(names.value(0), "x");
}

#[test]
fn name_fallback_matches_when_field_id_is_absent() {
    let schema = schema_id_and_name();
    let arrow_schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]);
    let batch = RecordBatch::try_new(Arc::new(arrow_schema), vec![
        Arc::new(Int32Array::from(vec![9])),
        Arc::new(StringArray::from(vec![Some("kept")])),
    ])
    .expect("name-only batch");
    let filled = apply_write_defaults(&schema, &batch).expect("name match");
    let names = filled
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name col");
    assert_eq!(names.value(0), "kept");
}

#[test]
fn non_primitive_write_default_on_missing_field_fails() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(
                2,
                "s",
                Type::Struct(crate::spec::StructType::new(vec![
                    NestedField::optional(3, "n", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .with_write_default(Literal::Struct(crate::spec::Struct::from_iter([Some(
                Literal::int(1),
            )])))
            .into(),
        ])
        .build()
        .expect("schema");
    let err = apply_write_defaults(&schema, &id_only_batch()).expect_err("nested");
    assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    assert!(err.message().contains("non-primitive"));
}

#[tokio::test]
async fn data_file_writer_writes_write_default_into_parquet() {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = Arc::new(schema_id_and_name());
    let location_gen = DefaultLocationGenerator::with_data_location(
        temp_dir.path().to_str().expect("utf8 path").to_string(),
    );
    let file_name_gen =
        DefaultFileNameGenerator::new("wd".to_string(), None, DataFileFormat::Parquet);
    let parquet = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema);
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet,
        file_io.clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .unpartitioned()
        .build(None)
        .await
        .expect("build writer");
    writer
        .write(id_only_batch())
        .await
        .expect("write missing name");
    let data_files = writer.close().await.expect("close");
    assert_eq!(data_files.len(), 1, "one data file");

    let input = file_io
        .new_input(data_files[0].file_path.clone())
        .expect("input")
        .read()
        .await
        .expect("read parquet");
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)
        .expect("parquet reader")
        .build()
        .expect("build reader");
    let batches: Vec<_> = reader.map(|b| b.expect("batch")).collect();
    assert_eq!(batches.len(), 1);
    let names = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name col");
    assert_eq!(names.value(0), "anon");
    assert_eq!(names.value(1), "anon");
    assert_eq!(names.value(2), "anon");
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id col");
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);
}

#[test]
fn missing_binary_write_default_is_filled() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::Binary))
                .with_write_default(Literal::Primitive(PrimitiveLiteral::Binary(vec![
                    0x01, 0x02,
                ])))
                .into(),
        ])
        .build()
        .expect("schema");
    let batch = id_only_batch();
    let filled = apply_write_defaults(&schema, &batch).expect("binary fill");
    let col = filled
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
        .expect("large binary");
    assert_eq!(col.value(0), &[0x01, 0x02]);
    assert_eq!(col.value(1), &[0x01, 0x02]);
}

#[test]
fn missing_time_write_default_is_filled() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "t", Type::Primitive(PrimitiveType::Time))
                .with_write_default(Literal::Primitive(PrimitiveLiteral::Long(1_000)))
                .into(),
        ])
        .build()
        .expect("schema");
    let batch = id_only_batch();
    let filled = apply_write_defaults(&schema, &batch).expect("time fill");
    let col = filled
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::Time64MicrosecondArray>()
        .expect("time");
    assert_eq!(col.value(0), 1_000);
}

#[test]
fn missing_uuid_write_default_is_filled() {
    let uuid = uuid::Uuid::parse_str("ec5911be-b0a7-458c-8438-c9a3e53cffae").expect("uuid");
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Uuid))
                .with_write_default(Literal::Primitive(PrimitiveLiteral::UInt128(
                    uuid.as_u128(),
                )))
                .into(),
        ])
        .build()
        .expect("schema");
    let batch = id_only_batch();
    let filled = apply_write_defaults(&schema, &batch).expect("uuid fill");
    let col = filled
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
        .expect("uuid bytes");
    assert_eq!(col.value(0), uuid.as_bytes());
}

#[test]
fn missing_fixed_write_default_is_filled() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "f", Type::Primitive(PrimitiveType::Fixed(4)))
                .with_write_default(Literal::Primitive(PrimitiveLiteral::Binary(vec![
                    9, 8, 7, 6,
                ])))
                .into(),
        ])
        .build()
        .expect("schema");
    let batch = id_only_batch();
    let filled = apply_write_defaults(&schema, &batch).expect("fixed fill");
    let col = filled
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
        .expect("fixed");
    assert_eq!(col.value(0), &[9, 8, 7, 6]);
}

#[test]
fn zero_row_omitted_uuid_write_default_is_empty_not_error() {
    let uuid = uuid::Uuid::parse_str("ec5911be-b0a7-458c-8438-c9a3e53cffae").expect("uuid");
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Uuid))
                .with_write_default(Literal::Primitive(PrimitiveLiteral::UInt128(
                    uuid.as_u128(),
                )))
                .into(),
        ])
        .build()
        .expect("schema");
    let arrow_schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
    ]);
    let batch = RecordBatch::new_empty(Arc::new(arrow_schema));
    let filled = apply_write_defaults(&schema, &batch).expect("0-row uuid fill");
    assert_eq!(filled.num_rows(), 0);
    assert_eq!(filled.num_columns(), 2);
    assert_eq!(
        filled.schema().field(1).data_type(),
        &DataType::FixedSizeBinary(16)
    );
}

#[test]
fn zero_row_omitted_fixed_write_default_is_empty_not_error() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "f", Type::Primitive(PrimitiveType::Fixed(4)))
                .with_write_default(Literal::Primitive(PrimitiveLiteral::Binary(vec![
                    9, 8, 7, 6,
                ])))
                .into(),
        ])
        .build()
        .expect("schema");
    let arrow_schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
    ]);
    let batch = RecordBatch::new_empty(Arc::new(arrow_schema));
    let filled = apply_write_defaults(&schema, &batch).expect("0-row fixed fill");
    assert_eq!(filled.num_rows(), 0);
    assert_eq!(filled.num_columns(), 2);
    assert_eq!(
        filled.schema().field(1).data_type(),
        &DataType::FixedSizeBinary(4)
    );
}

#[test]
fn type_mismatched_write_default_is_data_invalid() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                .with_write_default(Literal::int(7))
                .into(),
        ])
        .build()
        .expect("schema");
    let batch = id_only_batch();
    let err = apply_write_defaults(&schema, &batch).expect_err("mismatch");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.message().contains("Cannot apply write-default"),
        "got {}",
        err.message()
    );
}

#[tokio::test]
async fn data_file_writer_writes_binary_write_default_into_parquet() {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::Binary))
                    .with_write_default(Literal::Primitive(PrimitiveLiteral::Binary(vec![
                        0xaa, 0xbb,
                    ])))
                    .into(),
            ])
            .build()
            .expect("schema"),
    );
    let location_gen = DefaultLocationGenerator::with_data_location(
        temp_dir.path().to_str().expect("utf8 path").to_string(),
    );
    let file_name_gen =
        DefaultFileNameGenerator::new("bin".to_string(), None, DataFileFormat::Parquet);
    let parquet = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema);
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet,
        file_io.clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .unpartitioned()
        .build(None)
        .await
        .expect("build writer");
    writer.write(id_only_batch()).await.expect("write");
    let data_files = writer.close().await.expect("close");
    let input = file_io
        .new_input(data_files[0].file_path.clone())
        .expect("input")
        .read()
        .await
        .expect("read");
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)
        .expect("parquet reader")
        .build()
        .expect("build reader");
    let batches: Vec<_> = reader.map(|b| b.expect("batch")).collect();
    let payload = batches[0].column(1);
    let bytes = if let Some(array) = payload.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        array.value(0).to_vec()
    } else if let Some(array) = payload
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        array.value(0).to_vec()
    } else {
        panic!("payload not binary, type {:?}", payload.data_type());
    };
    assert_eq!(bytes, vec![0xaa, 0xbb]);
}

#[tokio::test]
async fn equality_delete_writer_with_projected_schema_writes_only_equality_ids() {
    use crate::arrow::arrow_schema_to_schema;
    use crate::writer::base_writer::equality_delete_writer::{
        EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
    };

    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = Arc::new(schema_id_and_name());
    let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone()).expect("eq config");
    let projected = Arc::new(
        arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected iceberg schema"),
    );
    let parquet = ParquetWriterBuilder::new(WriterProperties::builder().build(), projected);
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet,
        file_io.clone(),
        DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().expect("utf8 path").to_string(),
        ),
        DefaultFileNameGenerator::new("eq".to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .unpartitioned()
        .build(None)
        .await
        .expect("build eq writer");
    writer
        .write(id_only_batch())
        .await
        .expect("id-only equality delete");
    let files = writer.close().await.expect("close");
    assert_eq!(files.len(), 1);
    let input = file_io
        .new_input(files[0].file_path.clone())
        .expect("input")
        .read()
        .await
        .expect("read");
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)
        .expect("parquet reader")
        .build()
        .expect("build reader");
    let batches: Vec<_> = reader.map(|b| b.expect("batch")).collect();
    assert_eq!(
        batches[0].num_columns(),
        1,
        "must not add write-default name"
    );
    assert_eq!(batches[0].schema().field(0).name(), "id");
}

#[tokio::test]
async fn equality_delete_writer_does_not_fill_omitted_equality_key() {
    use crate::arrow::arrow_schema_to_schema;
    use crate::writer::base_writer::equality_delete_writer::{
        EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
    };

    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = Arc::new(schema_id_and_name());
    let config = EqualityDeleteWriterConfig::new(vec![1, 2], schema.clone()).expect("eq config");
    let projected = Arc::new(
        arrow_schema_to_schema(config.projected_arrow_schema_ref())
            .expect("projected iceberg schema"),
    );
    let parquet = ParquetWriterBuilder::new(WriterProperties::builder().build(), projected);
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet,
        file_io,
        DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().expect("utf8 path").to_string(),
        ),
        DefaultFileNameGenerator::new("eq2".to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .unpartitioned()
        .build(None)
        .await
        .expect("build eq writer");
    let err = writer
        .write(id_only_batch())
        .await
        .expect_err("omitted equality key must not be filled from write-default");
    assert_ne!(err.kind(), ErrorKind::FeatureUnsupported);
    let rendered = format!("{err:?}");
    assert!(
        !rendered.contains("anon"),
        "must not have filled name=anon into equality keys, got {rendered}"
    );
}

fn encoding_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "ts", Type::Primitive(PrimitiveType::Timestamptz)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "payload", Type::Primitive(PrimitiveType::Binary)).into(),
            NestedField::required(4, "tag", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("encoding schema")
}

#[test]
fn utc_alias_timestamptz_is_relabelled_to_canonical_zone() {
    let schema = encoding_schema();
    let alias: ArrayRef =
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3]).with_timezone("+00:00"));
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("ts", alias.data_type().clone(), false).with_metadata(id_meta(1)),
            Field::new("name", DataType::Utf8, false).with_metadata(id_meta(2)),
            Field::new("payload", DataType::Binary, false).with_metadata(id_meta(3)),
            Field::new("tag", DataType::Utf8, false).with_metadata(id_meta(4)),
        ])),
        vec![
            alias,
            boxed(StringArray::from(vec!["a", "b", "c"])),
            boxed(BinaryArray::from_vec(vec![b"x", b"y", b"z"])),
            boxed(StringArray::from(vec!["t1", "t2", "t3"])),
        ],
    )
    .expect("alias batch");
    let filled = apply_write_defaults(&schema, &batch).expect("utc alias writes");
    assert_eq!(
        filled.column(0).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );
    let stamps = filled
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .expect("ts col");
    assert_eq!(stamps.iter().flatten().collect::<Vec<_>>(), [1, 2, 3]);
}

#[test]
fn view_and_dictionary_encodings_write_as_plain_leaves() {
    let schema = encoding_schema();
    let dictionary: DictionaryArray<Int8Type> = ["d1", "d2", "d1"].into_iter().collect();
    let canonical: ArrayRef =
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3]).with_timezone("UTC"));
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("ts", canonical.data_type().clone(), false).with_metadata(id_meta(1)),
            Field::new("name", DataType::Utf8View, false).with_metadata(id_meta(2)),
            Field::new("payload", DataType::BinaryView, false).with_metadata(id_meta(3)),
            Field::new("tag", dictionary.data_type().clone(), false).with_metadata(id_meta(4)),
        ])),
        vec![
            canonical,
            boxed(StringViewArray::from(vec!["a", "b", "c"])),
            boxed(BinaryViewArray::from_iter_values([b"x", b"y", b"z"])),
            boxed(dictionary),
        ],
    )
    .expect("encoding batch");
    let filled = apply_write_defaults(&schema, &batch).expect("encodings write");
    assert_eq!(filled.column(1).data_type(), &DataType::Utf8);
    assert_eq!(filled.column(2).data_type(), &DataType::LargeBinary);
    assert_eq!(filled.column(3).data_type(), &DataType::Utf8);
    assert_eq!(
        filled
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("utf8 col")
            .iter()
            .flatten()
            .collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
    assert_eq!(
        filled
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("unpacked dict")
            .iter()
            .flatten()
            .collect::<Vec<_>>(),
        ["d1", "d2", "d1"]
    );
}
