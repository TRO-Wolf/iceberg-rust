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
use std::io::Read as _;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, FixedSizeBinaryArray,
    Float32Array, Float64Array, Int32Array, Int64Array, LargeBinaryArray, ListArray, MapArray,
    RecordBatch, StringArray, StructArray, Time64MicrosecondArray, TimestampMicrosecondArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, SchemaRef as ArrowSchemaRef};
use uuid::Uuid;

use super::*;
use crate::arrow::orc_reader::read_orc_data_bytes;
use crate::arrow::UTC_TIME_ZONE;
use crate::io::FileIO;
use crate::spec::{ListType, MapType, NestedField, PrimitiveType, Schema, StructType, Type};
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
};

fn make_temp() -> (tempfile::TempDir, FileIO, DefaultLocationGenerator) {
    let temp_dir = tempfile::TempDir::new().expect("create a temp dir");
    let file_io = FileIO::new_with_fs();
    let location_gen = DefaultLocationGenerator::with_data_location(
        temp_dir
            .path()
            .to_str()
            .expect("temp dir path is utf-8")
            .to_string(),
    );
    (temp_dir, file_io, location_gen)
}

fn output_file(
    file_io: &FileIO,
    location_gen: &DefaultLocationGenerator,
    prefix: &str,
) -> (String, OutputFile) {
    let file_name_gen =
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Orc);
    let path = location_gen.generate_location(None, &file_name_gen.generate_file_name());
    let of = file_io
        .new_output(&path)
        .expect("create the ORC output file");
    (path, of)
}

async fn read_back_bytes(file_io: &FileIO, path: &str) -> Bytes {
    file_io
        .new_input(path)
        .expect("open the written ORC file")
        .read()
        .await
        .expect("read the written ORC file")
}

fn schema_all_primitives() -> Schema {
    Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            NestedField::optional(1, "c_bool", Type::Primitive(PrimitiveType::Boolean)).into(),
            NestedField::optional(2, "c_int", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(3, "c_long", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(4, "c_float", Type::Primitive(PrimitiveType::Float)).into(),
            NestedField::optional(5, "c_double", Type::Primitive(PrimitiveType::Double)).into(),
            NestedField::optional(6, "c_string", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(7, "c_binary", Type::Primitive(PrimitiveType::Binary)).into(),
            NestedField::optional(8, "c_date", Type::Primitive(PrimitiveType::Date)).into(),
            NestedField::optional(9, "c_time", Type::Primitive(PrimitiveType::Time)).into(),
            NestedField::optional(10, "c_ts", Type::Primitive(PrimitiveType::Timestamp)).into(),
            NestedField::optional(11, "c_tstz", Type::Primitive(PrimitiveType::Timestamptz)).into(),
            NestedField::optional(12, "c_ts_ns", Type::Primitive(PrimitiveType::TimestampNs))
                .into(),
            NestedField::optional(
                13,
                "c_tstz_ns",
                Type::Primitive(PrimitiveType::TimestamptzNs),
            )
            .into(),
            NestedField::optional(14, "c_dec", Type::Primitive(PrimitiveType::Decimal {
                precision: 10,
                scale: 2,
            }))
            .into(),
            NestedField::optional(15, "c_uuid", Type::Primitive(PrimitiveType::Uuid)).into(),
            NestedField::optional(16, "c_fixed", Type::Primitive(PrimitiveType::Fixed(10))).into(),
        ])
        .build()
        .expect("build the all-primitives schema")
}

fn all_primitives_batch(schema: &Schema) -> RecordBatch {
    let arrow_schema: ArrowSchemaRef = Arc::new(
        crate::arrow::schema_to_arrow_schema(schema).expect("iceberg schema to arrow schema"),
    );
    let columns: Vec<ArrayRef> = vec![
        Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
        Arc::new(Int32Array::from(vec![Some(42), Some(i32::MIN), None])),
        Arc::new(Int64Array::from(vec![Some(42), Some(i64::MIN), None])),
        Arc::new(Float32Array::from(vec![Some(3.5), Some(f32::NAN), None])),
        Arc::new(Float64Array::from(vec![Some(4.5), Some(f64::NAN), None])),
        Arc::new(StringArray::from(vec![Some("hello"), Some(""), None])),
        Arc::new(LargeBinaryArray::from_opt_vec(vec![
            Some(&[1u8, 2, 255][..]),
            Some(&[][..]),
            None,
        ])),
        Arc::new(Date32Array::from(vec![Some(19723), Some(2932896), None])),
        Arc::new(Time64MicrosecondArray::from(vec![
            Some(0),
            Some(86_399_999_999),
            None,
        ])),
        Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1_704_103_200_000_000),
            Some(4_102_444_799_999_999),
            None,
        ])),
        Arc::new(
            TimestampMicrosecondArray::from(vec![Some(1_704_103_200_000_000), Some(0), None])
                .with_timezone(UTC_TIME_ZONE),
        ),
        Arc::new(TimestampNanosecondArray::from(vec![
            Some(1_704_103_200_000_000_123),
            Some(-1_500_000_000),
            None,
        ])),
        Arc::new(
            TimestampNanosecondArray::from(vec![Some(1), Some(-2_000_000_001), None])
                .with_timezone(UTC_TIME_ZONE),
        ),
        Arc::new(
            Decimal128Array::from(vec![Some(1234), Some(-999_999_999), None])
                .with_precision_and_scale(10, 2)
                .expect("decimal(10,2)"),
        ),
        Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![
                    Some(Uuid::from_u128(7).as_bytes().to_vec()),
                    Some(Uuid::from_u128(0).as_bytes().to_vec()),
                    None,
                ]
                .into_iter(),
                16,
            )
            .expect("uuid column"),
        ),
        Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![
                    Some(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    Some(vec![0; 10]),
                    None,
                ]
                .into_iter(),
                10,
            )
            .expect("fixed(10) column"),
        ),
    ];
    RecordBatch::try_new(arrow_schema, columns).expect("build the all-primitives batch")
}

async fn write_orc(
    builder: OrcWriterBuilder,
    file_io: &FileIO,
    location_gen: &DefaultLocationGenerator,
    prefix: &str,
    batches: &[RecordBatch],
) -> (String, Vec<DataFileBuilder>) {
    let (path, of) = output_file(file_io, location_gen, prefix);
    let mut writer = builder.build(of).await.expect("build the ORC writer");
    for batch in batches {
        writer.write(batch).await.expect("write a batch");
    }
    let files = writer.close().await.expect("close the ORC writer");
    (path, files)
}

#[tokio::test]
async fn test_orc_writer_round_trips_every_primitive_through_the_fork_reader() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let written = all_primitives_batch(&schema);

    let (path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()),
        &file_io,
        &location_gen,
        "all_primitives",
        std::slice::from_ref(&written),
    )
    .await;
    assert_eq!(files.len(), 1, "one ORC data file must be produced");

    let bytes = read_back_bytes(&file_io, &path).await;
    let batches = read_orc_data_bytes(bytes, &schema, 1024).expect("read the ORC file back");
    let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(total, 3, "all three rows must come back");

    let read = arrow_select::concat::concat_batches(&written.schema(), &batches)
        .expect("concatenate the decoded batches");
    for (index, field) in written.schema().fields().iter().enumerate() {
        assert_eq!(
            read.column(index).as_ref(),
            written.column(index).as_ref(),
            "column '{}' must round trip through ORC",
            field.name()
        );
    }
}

#[tokio::test]
async fn test_orc_writer_round_trips_decimal_38_extremes() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(
                    1,
                    "c_dec38",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 38,
                        scale: 9,
                    }),
                )
                .into(),
            ])
            .build()
            .expect("build the decimal(38,9) schema"),
    );
    let arrow_schema: ArrowSchemaRef = Arc::new(
        crate::arrow::schema_to_arrow_schema(&schema).expect("iceberg schema to arrow schema"),
    );
    let written = RecordBatch::try_new(arrow_schema, vec![Arc::new(
        Decimal128Array::from(vec![
            Some(99_999_999_999_999_999_999_999_999_999_999_999_999i128),
            Some(-99_999_999_999_999_999_999_999_999_999_999_999_999i128),
            None,
        ])
        .with_precision_and_scale(38, 9)
        .expect("decimal(38,9)"),
    ) as ArrayRef])
    .expect("build the decimal batch");

    let (path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()),
        &file_io,
        &location_gen,
        "dec38",
        std::slice::from_ref(&written),
    )
    .await;
    assert_eq!(files.len(), 1, "one ORC data file must be produced");

    let bytes = read_back_bytes(&file_io, &path).await;
    let batches = read_orc_data_bytes(bytes, &schema, 1024).expect("read the decimal file back");
    let read = arrow_select::concat::concat_batches(&written.schema(), &batches)
        .expect("concatenate the decoded batches");
    assert_eq!(
        read.column(0).as_ref(),
        written.column(0).as_ref(),
        "the decimal(38,9) extremes must round trip exactly"
    );
}

#[tokio::test]
async fn test_the_written_footer_carries_the_iceberg_id_attributes_the_reader_requires() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let written = all_primitives_batch(&schema);
    let builder = OrcWriterBuilder::new_from_properties(
        schema.clone(),
        &HashMap::from([(
            PROPERTY_ORC_COMPRESSION_CODEC.to_string(),
            "none".to_string(),
        )]),
    )
    .expect("build an uncompressed ORC writer");
    let (path, _files) = write_orc(
        builder,
        &file_io,
        &location_gen,
        "ids",
        std::slice::from_ref(&written),
    )
    .await;

    let bytes = read_back_bytes(&file_io, &path).await;
    let text = String::from_utf8_lossy(bytes.as_ref());
    assert!(
        text.contains("iceberg.id"),
        "the ORC footer must carry the iceberg.id type attributes"
    );
    assert!(
        text.contains("iceberg.required"),
        "the ORC footer must carry the iceberg.required type attributes"
    );

    let projection = Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            NestedField::optional(2, "renamed_int", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("build a renamed projection");
    let batches =
        read_orc_data_bytes(bytes, &projection, 1024).expect("resolve the column by field id");
    let column = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("an int column");
    assert_eq!(column.value(0), 42, "the id, not the name, resolved it");
}

#[tokio::test]
async fn test_an_orc_file_without_iceberg_ids_is_rejected_loudly_by_the_fork_reader() {
    use orc_rust::arrow_writer::ArrowWriterBuilder;

    let arrow_schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
        "c_int",
        DataType::Int32,
        true,
    )]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Int32Array::from(
        vec![Some(1)],
    ))])
    .expect("build a bare arrow batch");
    let mut sink = Vec::new();
    let mut writer = ArrowWriterBuilder::new(&mut sink, arrow_schema)
        .try_build()
        .expect("build the orc-rust writer");
    writer.write(&batch).expect("write the bare batch");
    writer.close().expect("close the orc-rust writer");

    let schema = Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            NestedField::optional(2, "c_int", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("build the projection");
    let error = read_orc_data_bytes(Bytes::from(sink), &schema, 1024)
        .expect_err("a file with no iceberg.id attributes must be refused");
    assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
    assert!(
        error.message().contains("iceberg.id"),
        "the refusal must name the missing attribute: {error}"
    );
}

fn schema_nested() -> Schema {
    Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(
                2,
                "c_arr",
                Type::List(ListType {
                    element_field: NestedField::list_element(
                        5,
                        Type::Primitive(PrimitiveType::Int),
                        true,
                    )
                    .into(),
                }),
            )
            .into(),
            NestedField::optional(
                3,
                "c_map",
                Type::Map(MapType {
                    key_field: NestedField::map_key_element(
                        6,
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    value_field: NestedField::map_value_element(
                        7,
                        Type::Primitive(PrimitiveType::Int),
                        true,
                    )
                    .into(),
                }),
            )
            .into(),
            NestedField::optional(
                4,
                "c_struct",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(8, "x", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(9, "y", Type::Primitive(PrimitiveType::String)).into(),
                ])),
            )
            .into(),
        ])
        .build()
        .expect("build the nested schema")
}

fn nested_batch(schema: &Schema) -> RecordBatch {
    let arrow_schema: ArrowSchemaRef = Arc::new(
        crate::arrow::schema_to_arrow_schema(schema).expect("iceberg schema to arrow schema"),
    );

    let id = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;

    let DataType::List(element_field) = arrow_schema.field(1).data_type().clone() else {
        panic!("field 1 must be a list");
    };
    let list = ListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(vec![
        Some(vec![Some(1), Some(2), Some(3)]),
        Some(vec![]),
        None,
    ]);
    let list = Arc::new(ListArray::new(
        element_field,
        list.offsets().clone(),
        list.values().clone(),
        list.nulls().cloned(),
    )) as ArrayRef;

    let DataType::Map(entries_field, sorted) = arrow_schema.field(2).data_type().clone() else {
        panic!("field 2 must be a map");
    };
    let DataType::Struct(entry_fields) = entries_field.data_type().clone() else {
        panic!("map entries must be a struct");
    };
    let keys = Arc::new(StringArray::from(vec!["k"])) as ArrayRef;
    let values = Arc::new(Int32Array::from(vec![Some(1)])) as ArrayRef;
    let entries = StructArray::new(entry_fields, vec![keys, values], None);
    let offsets = arrow_buffer::OffsetBuffer::new(vec![0, 1, 1, 1].into());
    let map_nulls = arrow_buffer::NullBuffer::from(vec![true, true, false]);
    let map = Arc::new(MapArray::new(
        entries_field,
        offsets,
        entries,
        Some(map_nulls),
        sorted,
    )) as ArrayRef;

    let DataType::Struct(struct_fields) = arrow_schema.field(3).data_type().clone() else {
        panic!("field 3 must be a struct");
    };
    let nested_x = Arc::new(Int32Array::from(vec![Some(1), None, None])) as ArrayRef;
    let nested_y = Arc::new(StringArray::from(vec![Some("z"), None, None])) as ArrayRef;
    let nested = Arc::new(StructArray::new(
        struct_fields,
        vec![nested_x, nested_y],
        Some(arrow_buffer::NullBuffer::from(vec![true, true, false])),
    )) as ArrayRef;

    RecordBatch::try_new(arrow_schema, vec![id, list, map, nested])
        .expect("build the nested batch")
}

#[tokio::test]
async fn test_orc_writer_round_trips_list_map_and_struct_through_orc_rust() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_nested());
    let written = nested_batch(&schema);

    let (path, files) = write_orc(
        OrcWriterBuilder::new(schema.clone()),
        &file_io,
        &location_gen,
        "nested",
        std::slice::from_ref(&written),
    )
    .await;
    assert_eq!(files.len(), 1);

    let bytes = read_back_bytes(&file_io, &path).await;
    let reader = orc_rust::ArrowReaderBuilder::try_new(bytes)
        .expect("open the ORC file with orc-rust")
        .build();
    let batches: Vec<RecordBatch> = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .expect("decode every nested batch");
    let decoded = arrow_select::concat::concat_batches(&batches[0].schema(), &batches)
        .expect("concatenate the oracle batches");
    assert_eq!(
        decoded.num_rows(),
        3,
        "all three nested rows must come back"
    );

    let id = decoded
        .column_by_name("id")
        .expect("the id column")
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("an int array");
    assert_eq!(id.value(0), 1);
    assert_eq!(id.value(1), 2);
    assert_eq!(id.value(2), 3);

    let list = decoded
        .column_by_name("c_arr")
        .expect("the list column")
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("a list array");
    assert!(!list.is_null(0));
    assert!(!list.is_null(1), "the second list row is empty, not null");
    assert!(list.is_null(2), "the third list row is null");
    assert_eq!(list.value_length(0), 3);
    assert_eq!(list.value_length(1), 0);
    let row0 = list
        .value(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("int list elements")
        .clone();
    assert_eq!(row0.len(), 3);
    assert_eq!(row0.value(0), 1);
    assert_eq!(row0.value(1), 2);
    assert_eq!(row0.value(2), 3);

    let map = decoded
        .column_by_name("c_map")
        .expect("the map column")
        .as_any()
        .downcast_ref::<MapArray>()
        .expect("a map array");
    assert!(!map.is_null(0));
    assert!(!map.is_null(1), "the second map row is empty, not null");
    assert!(map.is_null(2), "the third map row is null");
    assert_eq!(map.value_length(0), 1);
    assert_eq!(map.value_length(1), 0);
    let entries = map.value(0);
    let entries = entries
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("map entries");
    let keys = entries
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("string map keys");
    let values = entries
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("int map values");
    assert_eq!(keys.value(0), "k");
    assert_eq!(values.value(0), 1);

    let nested = decoded
        .column_by_name("c_struct")
        .expect("the struct column")
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("a struct array");
    assert!(!nested.is_null(0));
    assert!(
        !nested.is_null(1),
        "the second struct row is valid with null children"
    );
    assert!(nested.is_null(2), "the third struct row is null");
    let x = nested
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("an int child");
    let y = nested
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("a string child");
    assert_eq!(x.value(0), 1);
    assert_eq!(y.value(0), "z");
    assert!(x.is_null(1), "the second row's struct child is null");
    assert!(y.is_null(1), "the second row's string child is null");
}

#[tokio::test]
async fn test_an_empty_writer_leaves_no_phantom_file() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let (path, of) = output_file(&file_io, &location_gen, "empty");
    let writer = OrcWriterBuilder::new(schema)
        .build(of)
        .await
        .expect("build the writer");
    let files = writer.close().await.expect("close an empty writer");
    assert!(files.is_empty(), "an empty ORC writer emits no data file");
    assert!(
        !file_io.exists(&path).await.expect("probe the path"),
        "no phantom ORC file may be left behind"
    );
}

#[tokio::test]
async fn test_a_small_stripe_size_produces_several_stripes_that_still_round_trip() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let batch = all_primitives_batch(&schema);
    let mut builder = OrcWriterBuilder::new(schema.clone());
    builder.stripe_size = 1;

    let batches = vec![batch.clone(), batch.clone(), batch.clone()];
    let (path, files) = write_orc(builder, &file_io, &location_gen, "stripes", &batches).await;
    let data_file = files
        .into_iter()
        .next()
        .expect("one data file")
        .build()
        .expect("build the data file");
    assert_eq!(data_file.record_count(), 9);
    let split_offsets = data_file
        .split_offsets()
        .expect("split offsets are set")
        .to_vec();
    assert_eq!(split_offsets.len(), 3, "one split offset per stripe");

    let bytes = read_back_bytes(&file_io, &path).await;
    let layout = test_file_layout(&bytes);
    assert_eq!(
        layout.stripe_rows,
        vec![3, 3, 3],
        "one 3-row batch per stripe in the emitted footer"
    );
    assert_eq!(
        layout.footer_rows,
        data_file.record_count(),
        "Footer.numberOfRows must equal the committed record count"
    );
    assert_eq!(
        layout.footer_rows,
        layout.stripe_rows.iter().sum::<u64>(),
        "Footer.numberOfRows must equal the sum of stripe row counts"
    );
    assert_eq!(
        data_file.file_size_in_bytes(),
        u64::try_from(bytes.len()).expect("the test file fits in a u64"),
        "file_size_in_bytes must equal the exact on-disk byte count"
    );
    assert_eq!(
        layout.header_length,
        u64::try_from(ORC_MAGIC.len()).expect("the test magic fits in a u64"),
        "the footer header length must cover the ORC magic"
    );
    assert_eq!(
        layout.content_length, layout.footer_start,
        "the footer content length must land on the real footer bytes"
    );
    assert_eq!(
        layout.row_index_stride, 0,
        "no row-index streams are written, so the emitted stride is 0"
    );
    assert_eq!(
        split_offsets[0],
        i64::try_from(ORC_MAGIC.len()).expect("the test magic fits in an i64"),
        "the first stripe starts right after the ORC magic"
    );
    let footer_offsets: Vec<i64> = layout
        .stripe_offsets
        .iter()
        .map(|offset| i64::try_from(*offset).expect("the test offsets fit in an i64"))
        .collect();
    assert_eq!(
        split_offsets, footer_offsets,
        "split offsets must match the footer stripe offsets"
    );
    assert_eq!(
        layout.footer_start
            + layout.postscript_footer_length
            + layout.postscript_length
            + 1,
        u64::try_from(bytes.len()).expect("the test file fits in a u64"),
        "the PostScript footer length must land on the real footer bytes"
    );
    assert_eq!(
        layout.postscript_compression, 1,
        "the default codec is zlib"
    );
    assert_eq!(
        layout.postscript_block_size,
        Some(
            u64::try_from(DEFAULT_COMPRESSION_BLOCK_SIZE)
                .expect("the test block size fits in a u64")
        ),
        "the PostScript must declare the compression block size"
    );
    assert_eq!(
        layout.postscript_version,
        vec![0, 12],
        "the PostScript must declare file version 0.12"
    );
    assert_eq!(
        layout.postscript_magic,
        b"ORC".to_vec(),
        "the PostScript must carry the ORC magic"
    );
    assert_eq!(
        layout.stripe_timezones,
        vec![b"UTC".to_vec(), b"UTC".to_vec(), b"UTC".to_vec()],
        "every stripe footer must declare the UTC writer timezone"
    );
    let decoded = read_orc_data_bytes(bytes, &schema, 1024).expect("read the multi-stripe file");
    let expected = arrow_select::concat::concat_batches(&batch.schema(), &batches)
        .expect("concatenate the written batches");
    let read = arrow_select::concat::concat_batches(&batch.schema(), &decoded)
        .expect("concatenate the decoded batches");
    assert_eq!(read.num_rows(), 9);
    for (index, field) in batch.schema().fields().iter().enumerate() {
        assert_eq!(
            read.column(index).as_ref(),
            expected.column(index).as_ref(),
            "column '{}' must round trip across stripes",
            field.name()
        );
    }
}

#[tokio::test]
async fn test_uncompressed_orc_round_trips_too() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let written = all_primitives_batch(&schema);
    let builder = OrcWriterBuilder::new_from_properties(
        schema.clone(),
        &HashMap::from([(
            PROPERTY_ORC_COMPRESSION_CODEC.to_string(),
            "none".to_string(),
        )]),
    )
    .expect("build an uncompressed ORC writer");

    let (path, _files) = write_orc(
        builder,
        &file_io,
        &location_gen,
        "plain",
        std::slice::from_ref(&written),
    )
    .await;
    let bytes = read_back_bytes(&file_io, &path).await;
    let layout = test_file_layout(&bytes);
    assert_eq!(
        layout.postscript_compression, 0,
        "the file must actually be uncompressed"
    );
    assert_eq!(
        layout.postscript_block_size, None,
        "an uncompressed file declares no compression block size"
    );
    assert_eq!(layout.footer_rows, 3, "all three rows reach the footer");
    assert_eq!(layout.stripe_rows, vec![3], "one stripe holds all rows");
    let decoded = read_orc_data_bytes(bytes, &schema, 1024).expect("read the uncompressed file");
    let read = arrow_select::concat::concat_batches(&written.schema(), &decoded)
        .expect("concatenate the decoded batches");
    assert_eq!(read.num_rows(), 3);
    for (index, field) in written.schema().fields().iter().enumerate() {
        assert_eq!(
            read.column(index).as_ref(),
            written.column(index).as_ref(),
            "column '{}' must round trip uncompressed",
            field.name()
        );
    }
}

#[tokio::test]
async fn test_the_default_codec_is_java_s_zlib() {
    let builder =
        OrcWriterBuilder::new_from_properties(Arc::new(schema_all_primitives()), &HashMap::new())
            .expect("build with no properties");
    assert_eq!(builder.compression, encode::OrcCompression::Zlib);
    assert_eq!(
        builder.stripe_size,
        PROPERTY_ORC_STRIPE_SIZE_BYTES_DEFAULT as usize
    );
}

#[tokio::test]
async fn test_an_unsupported_codec_is_a_typed_error_naming_it() {
    let error = OrcWriterBuilder::new_from_properties(
        Arc::new(schema_all_primitives()),
        &HashMap::from([(
            PROPERTY_ORC_COMPRESSION_CODEC.to_string(),
            "lzo".to_string(),
        )]),
    )
    .expect_err("lzo is not supported");
    assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
    assert!(error.message().contains("lzo"), "{error}");
}

#[tokio::test]
async fn test_a_variant_column_is_refused_at_build_time_without_a_panic() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(1, "v", Type::Variant).into(),
            ])
            .build()
            .expect("build a variant schema"),
    );
    let (_path, of) = output_file(&file_io, &location_gen, "variant");
    let error = OrcWriterBuilder::new(schema)
        .build(of)
        .await
        .map(|_| ())
        .expect_err("variant has no ORC mapping");
    assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
    assert!(error.message().contains("variant"), "{error}");
}

#[tokio::test]
async fn test_current_file_status_tracks_rows_and_bytes() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(schema_all_primitives());
    let batch = all_primitives_batch(&schema);
    let (path, of) = output_file(&file_io, &location_gen, "status");
    let mut writer = OrcWriterBuilder::new(schema)
        .build(of)
        .await
        .expect("build the writer");
    assert_eq!(writer.current_row_num(), 0);
    assert_eq!(writer.current_file_path(), path);
    writer.write(&batch).await.expect("write a batch");
    assert_eq!(writer.current_row_num(), 3);
    assert!(writer.current_written_size() > 0);
    let files = writer.close().await.expect("close");
    assert_eq!(files.len(), 1);
}

fn unicode_and_boundary_batch(schema: &Schema) -> RecordBatch {
    let arrow_schema: ArrowSchemaRef = Arc::new(
        crate::arrow::schema_to_arrow_schema(schema).expect("iceberg schema to arrow schema"),
    );
    RecordBatch::try_new(arrow_schema, vec![
        Arc::new(StringArray::from(vec![
            Some("ünïcödé☃"),
            Some(""),
            Some(&"x".repeat(5000)[..]),
        ])) as ArrayRef,
    ])
    .expect("build the unicode batch")
}

#[tokio::test]
async fn test_unicode_empty_and_long_strings_round_trip() {
    let (_temp, file_io, location_gen) = make_temp();
    let schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "s", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("build the string schema"),
    );
    let written = unicode_and_boundary_batch(&schema);
    let (path, _files) = write_orc(
        OrcWriterBuilder::new(schema.clone()),
        &file_io,
        &location_gen,
        "unicode",
        std::slice::from_ref(&written),
    )
    .await;

    let bytes = read_back_bytes(&file_io, &path).await;
    let decoded = read_orc_data_bytes(bytes, &schema, 1024).expect("read the string file");
    let read = arrow_select::concat::concat_batches(&written.schema(), &decoded)
        .expect("concatenate");
    assert_eq!(read.column(0).as_ref(), written.column(0).as_ref());
}
