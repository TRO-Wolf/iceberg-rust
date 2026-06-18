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

// Offline tests for the Avro data-file value reader.
//
// Golden Avro bytes are generated in-test with `apache_avro::Writer`, using the writer Avro schema
// produced by `crate::avro::schema_to_avro_schema` (which stamps the `field-id` props the reader
// resolves by). The real Java-written interop oracle is U2; this file proves the engine core
// against hand-declared expected rows plus mutation baits.

use apache_avro::types::Value as AvroWriteValue;
use apache_avro::{Decimal as AvroDecimal, Writer as AvroWriter};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type,
    Time64MicrosecondType, TimestampMicrosecondType, TimestampNanosecondType,
};
use arrow_array::{Array, BooleanArray};

use super::*;
use crate::arrow::UTC_TIME_ZONE;
use crate::avro::schema_to_avro_schema;
use crate::spec::{ListType, MapType, NestedField};

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

/// Build an Avro OCF holding `rows` written under the Avro schema derived from `write_schema`.
fn write_avro(write_schema: &Schema, rows: Vec<AvroWriteValue>) -> Vec<u8> {
    let avro_schema =
        schema_to_avro_schema("data", write_schema).expect("iceberg→avro schema conversion");
    let mut writer = AvroWriter::new(&avro_schema, Vec::new());
    for row in rows {
        writer
            .append_value_ref(&row)
            .expect("append a value-record row to the Avro writer");
    }
    writer.into_inner().expect("finalize the Avro OCF")
}

fn schema_of(fields: Vec<NestedField>) -> Schema {
    Schema::builder()
        .with_fields(fields.into_iter().map(|f| f.into()))
        .build()
        .expect("build iceberg schema")
}

fn rec(fields: Vec<(&str, AvroWriteValue)>) -> AvroWriteValue {
    AvroWriteValue::Record(
        fields
            .into_iter()
            .map(|(n, v)| (n.to_string(), v))
            .collect(),
    )
}

/// Wrap a present value in the non-null branch of an Iceberg optional union (`[null, T]`, so the
/// value branch is index 1). The Avro writer requires the explicit `Value::Union` form.
fn opt(v: AvroWriteValue) -> AvroWriteValue {
    AvroWriteValue::Union(1, Box::new(v))
}

/// The null branch of an Iceberg optional union (`[null, T]`, branch index 0).
fn none() -> AvroWriteValue {
    AvroWriteValue::Union(0, Box::new(AvroWriteValue::Null))
}

/// Decode once, asserting exactly one batch and returning it.
fn decode_one(bytes: &[u8], expected: &Schema) -> RecordBatch {
    let mut batches = read_avro_data_bytes(bytes, expected, 1024).expect("decode avro data bytes");
    assert_eq!(batches.len(), 1, "expected a single batch");
    batches.pop().expect("one batch")
}

// -------------------------------------------------------------------------------------------------
// Per-primitive-type coverage (round-trip: write golden bytes, read, assert)
// -------------------------------------------------------------------------------------------------

#[test]
fn test_all_primitive_types_roundtrip() {
    let schema = schema_of(vec![
        NestedField::required(1, "b", Type::Primitive(PrimitiveType::Boolean)),
        NestedField::required(2, "i", Type::Primitive(PrimitiveType::Int)),
        NestedField::required(3, "l", Type::Primitive(PrimitiveType::Long)),
        NestedField::required(4, "f", Type::Primitive(PrimitiveType::Float)),
        NestedField::required(5, "d", Type::Primitive(PrimitiveType::Double)),
        NestedField::required(6, "s", Type::Primitive(PrimitiveType::String)),
        NestedField::required(7, "bin", Type::Primitive(PrimitiveType::Binary)),
    ]);

    let bytes = write_avro(&schema, vec![
        rec(vec![
            ("b", AvroWriteValue::Boolean(true)),
            ("i", AvroWriteValue::Int(42)),
            ("l", AvroWriteValue::Long(9_000_000_000)),
            ("f", AvroWriteValue::Float(1.5)),
            ("d", AvroWriteValue::Double(2.25)),
            ("s", AvroWriteValue::String("hello".into())),
            ("bin", AvroWriteValue::Bytes(vec![1, 2, 3])),
        ]),
        rec(vec![
            ("b", AvroWriteValue::Boolean(false)),
            ("i", AvroWriteValue::Int(-7)),
            ("l", AvroWriteValue::Long(-1)),
            ("f", AvroWriteValue::Float(-3.5)),
            ("d", AvroWriteValue::Double(-4.5)),
            ("s", AvroWriteValue::String("world".into())),
            ("bin", AvroWriteValue::Bytes(vec![9])),
        ]),
    ]);

    let batch = decode_one(&bytes, &schema);
    assert_eq!(batch.num_rows(), 2);

    let b = batch
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("boolean column");
    assert!(b.value(0));
    assert!(!b.value(1));

    let i = batch.column(1).as_primitive::<Int32Type>();
    assert_eq!(i.value(0), 42);
    assert_eq!(i.value(1), -7);

    let l = batch.column(2).as_primitive::<Int64Type>();
    assert_eq!(l.value(0), 9_000_000_000);

    let f = batch.column(3).as_primitive::<Float32Type>();
    assert_eq!(f.value(0), 1.5);

    let d = batch.column(4).as_primitive::<Float64Type>();
    assert_eq!(d.value(0), 2.25);

    let s = batch.column(5).as_string::<i32>();
    assert_eq!(s.value(0), "hello");
    assert_eq!(s.value(1), "world");

    let bin = batch.column(6).as_binary::<i64>();
    assert_eq!(bin.value(0), &[1, 2, 3]);
}

#[test]
fn test_logical_types_roundtrip() {
    let schema = schema_of(vec![
        NestedField::required(1, "dt", Type::Primitive(PrimitiveType::Date)),
        NestedField::required(2, "tm", Type::Primitive(PrimitiveType::Time)),
        NestedField::required(3, "ts", Type::Primitive(PrimitiveType::Timestamp)),
        NestedField::required(4, "tstz", Type::Primitive(PrimitiveType::Timestamptz)),
        NestedField::required(5, "tsns", Type::Primitive(PrimitiveType::TimestampNs)),
        NestedField::required(6, "tstzns", Type::Primitive(PrimitiveType::TimestamptzNs)),
        NestedField::required(7, "fx", Type::Primitive(PrimitiveType::Fixed(4))),
        NestedField::required(8, "uu", Type::Primitive(PrimitiveType::Uuid)),
    ]);

    let uuid = uuid::Uuid::from_u128(0x0011_2233_4455_6677_8899_aabb_ccdd_eeff);
    // NOTE: the Iceberg→Avro schema converter collapses tz-ness — both `timestamp` and
    // `timestamptz` map to Avro `timestamp-micros` (and the `_ns` pair to `timestamp-nanos`). The
    // file therefore always carries `TimestampMicros`/`TimestampNanos`; tz-ness is decided at read
    // time by the EXPECTED Iceberg type. We write the non-local variants for every field.
    let bytes = write_avro(&schema, vec![rec(vec![
        ("dt", AvroWriteValue::Date(19_000)),
        ("tm", AvroWriteValue::TimeMicros(3_600_000_000)),
        ("ts", AvroWriteValue::TimestampMicros(1_600_000_000_000_000)),
        (
            "tstz",
            AvroWriteValue::TimestampMicros(1_600_000_000_000_000),
        ),
        (
            "tsns",
            AvroWriteValue::TimestampNanos(1_600_000_000_000_000_001),
        ),
        (
            "tstzns",
            AvroWriteValue::TimestampNanos(1_600_000_000_000_000_002),
        ),
        ("fx", AvroWriteValue::Fixed(4, vec![0xDE, 0xAD, 0xBE, 0xEF])),
        ("uu", AvroWriteValue::Uuid(uuid)),
    ])]);

    let batch = decode_one(&bytes, &schema);

    assert_eq!(
        batch.column(0).as_primitive::<Date32Type>().value(0),
        19_000
    );
    assert_eq!(
        batch
            .column(1)
            .as_primitive::<Time64MicrosecondType>()
            .value(0),
        3_600_000_000
    );

    let ts = batch.column(2).as_primitive::<TimestampMicrosecondType>();
    assert_eq!(ts.value(0), 1_600_000_000_000_000);
    // Non-tz timestamp must NOT carry a timezone; tz timestamp must carry UTC.
    assert_eq!(ts.timezone(), None);
    let tstz = batch.column(3).as_primitive::<TimestampMicrosecondType>();
    assert_eq!(tstz.value(0), 1_600_000_000_000_000);
    assert_eq!(tstz.timezone(), Some(UTC_TIME_ZONE));

    assert_eq!(
        batch
            .column(4)
            .as_primitive::<TimestampNanosecondType>()
            .value(0),
        1_600_000_000_000_000_001
    );
    let tstzns = batch.column(5).as_primitive::<TimestampNanosecondType>();
    assert_eq!(tstzns.value(0), 1_600_000_000_000_000_002);
    assert_eq!(tstzns.timezone(), Some(UTC_TIME_ZONE));

    let fx = batch.column(6).as_fixed_size_binary();
    assert_eq!(fx.value(0), &[0xDE, 0xAD, 0xBE, 0xEF]);

    let uu = batch.column(7).as_fixed_size_binary();
    assert_eq!(uu.value(0), uuid.as_bytes());
}

// -------------------------------------------------------------------------------------------------
// Decimal — exact scale + big-endian unscaled value
// -------------------------------------------------------------------------------------------------

#[test]
fn test_decimal_roundtrip() {
    let schema = schema_of(vec![NestedField::required(
        1,
        "amt",
        Type::Primitive(PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        }),
    )]);

    // 123.45 at scale 2 => unscaled 12345; -1.00 => unscaled -100.
    let bytes = write_avro(&schema, vec![
        rec(vec![(
            "amt",
            AvroWriteValue::Decimal(AvroDecimal::from(12345_i32.to_be_bytes().to_vec())),
        )]),
        rec(vec![(
            "amt",
            AvroWriteValue::Decimal(AvroDecimal::from((-100_i32).to_be_bytes().to_vec())),
        )]),
    ]);

    let batch = decode_one(&bytes, &schema);
    let col = batch.column(0).as_primitive::<Decimal128Type>();
    assert_eq!(col.value(0), 12345_i128);
    assert_eq!(col.value(1), -100_i128);
    // The Arrow decimal must carry the EXACT scale; a wrong-scale array would render values 100x off.
    assert_eq!(col.scale(), 2);
    assert_eq!(col.precision(), 10);
}

/// MUTATION BAIT: a negative decimal proves big-endian two's-complement sign extension. If
/// `be_bytes_to_i128` dropped the sign extension (e.g. zero-fill), -100 would decode as a large
/// positive number and this reds.
#[test]
fn test_decimal_negative_sign_extension_bait() {
    let schema = schema_of(vec![NestedField::required(
        1,
        "amt",
        Type::Primitive(PrimitiveType::Decimal {
            precision: 18,
            scale: 0,
        }),
    )]);
    // -1 unscaled, written as a single 0xFF byte (minimal two's-complement).
    let bytes = write_avro(&schema, vec![rec(vec![(
        "amt",
        AvroWriteValue::Decimal(AvroDecimal::from(vec![0xFFu8])),
    )])]);
    let batch = decode_one(&bytes, &schema);
    let col = batch.column(0).as_primitive::<Decimal128Type>();
    assert_eq!(col.value(0), -1_i128);
}

// -------------------------------------------------------------------------------------------------
// Type promotion: int→long and float→double driven by the EXPECTED type
// -------------------------------------------------------------------------------------------------

#[test]
fn test_int_to_long_and_float_to_double_promotion() {
    // The FILE is written with int + float...
    let file_schema = schema_of(vec![
        NestedField::required(1, "n", Type::Primitive(PrimitiveType::Int)),
        NestedField::required(2, "x", Type::Primitive(PrimitiveType::Float)),
    ]);
    let bytes = write_avro(&file_schema, vec![rec(vec![
        ("n", AvroWriteValue::Int(123)),
        ("x", AvroWriteValue::Float(1.5)),
    ])]);

    // ...but the EXPECTED (read) schema asks for long + double — Java widens at read time.
    let expected = schema_of(vec![
        NestedField::required(1, "n", Type::Primitive(PrimitiveType::Long)),
        NestedField::required(2, "x", Type::Primitive(PrimitiveType::Double)),
    ]);

    let batch = decode_one(&bytes, &expected);
    assert_eq!(
        batch.schema(),
        Arc::new(schema_to_arrow_schema(&expected).expect("arrow schema"))
    );
    assert_eq!(
        batch.column(0).as_primitive::<Int64Type>().value(0),
        123_i64
    );
    assert_eq!(
        batch.column(1).as_primitive::<Float64Type>().value(0),
        1.5_f64
    );
}

// -------------------------------------------------------------------------------------------------
// Null / union (optional fields)
// -------------------------------------------------------------------------------------------------

#[test]
fn test_optional_null_and_present() {
    let schema = schema_of(vec![
        NestedField::optional(1, "maybe_i", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "maybe_s", Type::Primitive(PrimitiveType::String)),
    ]);

    let bytes = write_avro(&schema, vec![
        rec(vec![
            ("maybe_i", opt(AvroWriteValue::Int(7))),
            ("maybe_s", none()),
        ]),
        rec(vec![
            ("maybe_i", none()),
            ("maybe_s", opt(AvroWriteValue::String("present".into()))),
        ]),
    ]);

    let batch = decode_one(&bytes, &schema);
    let i = batch.column(0).as_primitive::<Int32Type>();
    assert_eq!(i.value(0), 7);
    assert!(i.is_null(1));

    let s = batch.column(1).as_string::<i32>();
    assert!(s.is_null(0));
    assert_eq!(s.value(1), "present");
}

// -------------------------------------------------------------------------------------------------
// Projection + skip of unprojected file columns
// -------------------------------------------------------------------------------------------------

#[test]
fn test_projection_skips_unprojected_columns() {
    // File has 3 columns (ids 1,2,3).
    let file_schema = schema_of(vec![
        NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)),
        NestedField::required(2, "b", Type::Primitive(PrimitiveType::String)),
        NestedField::required(3, "c", Type::Primitive(PrimitiveType::Long)),
    ]);
    let bytes = write_avro(&file_schema, vec![rec(vec![
        ("a", AvroWriteValue::Int(10)),
        ("b", AvroWriteValue::String("skip-me".into())),
        ("c", AvroWriteValue::Long(99)),
    ])]);

    // Project only ids 3 then 1 (reordered) — id 2 is skipped (decode-and-discard).
    let expected = schema_of(vec![
        NestedField::required(3, "c", Type::Primitive(PrimitiveType::Long)),
        NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)),
    ]);

    let batch = decode_one(&bytes, &expected);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.column(0).as_primitive::<Int64Type>().value(0), 99);
    assert_eq!(batch.column(1).as_primitive::<Int32Type>().value(0), 10);
}

// -------------------------------------------------------------------------------------------------
// Missing-column defaults (priority order)
// -------------------------------------------------------------------------------------------------

#[test]
fn test_missing_optional_column_is_null() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "a",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let bytes = write_avro(&file_schema, vec![rec(vec![("a", AvroWriteValue::Int(5))])]);

    // id 2 is absent from the file and optional => null.
    let expected = schema_of(vec![
        NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "missing_opt", Type::Primitive(PrimitiveType::String)),
    ]);

    let batch = decode_one(&bytes, &expected);
    assert_eq!(batch.column(0).as_primitive::<Int32Type>().value(0), 5);
    assert!(batch.column(1).as_string::<i32>().is_null(0));
}

#[test]
fn test_missing_required_column_uses_initial_default() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "a",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let bytes = write_avro(&file_schema, vec![rec(vec![("a", AvroWriteValue::Int(5))])]);

    // id 2 is required, absent, but carries an initial-default => constant.
    let default_field =
        NestedField::required(2, "with_default", Type::Primitive(PrimitiveType::Long))
            .with_initial_default(Literal::long(777));
    let expected = schema_of(vec![
        NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)),
        default_field,
    ]);

    let batch = decode_one(&bytes, &expected);
    assert_eq!(batch.column(1).as_primitive::<Int64Type>().value(0), 777);
}

#[test]
fn test_missing_required_no_default_errors() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "a",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let bytes = write_avro(&file_schema, vec![rec(vec![("a", AvroWriteValue::Int(5))])]);

    let expected = schema_of(vec![
        NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)),
        NestedField::required(2, "needed", Type::Primitive(PrimitiveType::String)),
    ]);

    let err = read_avro_data_bytes(&bytes, &expected, 1024).expect_err("must error");
    assert!(
        err.to_string().contains("Missing required field: needed"),
        "unexpected error: {err}"
    );
}

/// IS_DELETED missing column => constant `false` (priority rung 3).
#[test]
fn test_missing_is_deleted_defaults_false() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "a",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let bytes = write_avro(&file_schema, vec![
        rec(vec![("a", AvroWriteValue::Int(1))]),
        rec(vec![("a", AvroWriteValue::Int(2))]),
    ]);

    let expected = schema_of(vec![
        NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)),
        NestedField::required(
            RESERVED_FIELD_ID_DELETED,
            "_deleted",
            Type::Primitive(PrimitiveType::Boolean),
        ),
    ]);

    let batch = decode_one(&bytes, &expected);
    let deleted = batch
        .column(1)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("boolean column");
    assert!(!deleted.value(0));
    assert!(!deleted.value(1));
}

/// ROW_POSITION missing column => running counter (priority rung 4), continuing across batches.
#[test]
fn test_missing_row_position_counter_across_batches() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "a",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let bytes = write_avro(&file_schema, vec![
        rec(vec![("a", AvroWriteValue::Int(10))]),
        rec(vec![("a", AvroWriteValue::Int(11))]),
        rec(vec![("a", AvroWriteValue::Int(12))]),
    ]);

    let expected = schema_of(vec![
        NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)),
        NestedField::required(
            RESERVED_FIELD_ID_POS,
            "_pos",
            Type::Primitive(PrimitiveType::Long),
        ),
    ]);

    // batch_size 2 => positions [0,1] then [2].
    let batches = read_avro_data_bytes(&bytes, &expected, 2).expect("decode");
    assert_eq!(batches.len(), 2);
    let p0 = batches[0].column(1).as_primitive::<Int64Type>();
    assert_eq!(p0.value(0), 0);
    assert_eq!(p0.value(1), 1);
    let p1 = batches[1].column(1).as_primitive::<Int64Type>();
    assert_eq!(p1.value(0), 2);
}

// -------------------------------------------------------------------------------------------------
// Batching
// -------------------------------------------------------------------------------------------------

#[test]
fn test_batching_splits_rows() {
    let schema = schema_of(vec![NestedField::required(
        1,
        "a",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let rows: Vec<AvroWriteValue> = (0..5)
        .map(|i| rec(vec![("a", AvroWriteValue::Int(i))]))
        .collect();
    let bytes = write_avro(&schema, rows);

    let batches = read_avro_data_bytes(&bytes, &schema, 2).expect("decode");
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>(),
        vec![2, 2, 1]
    );
    assert_eq!(batches[2].column(0).as_primitive::<Int32Type>().value(0), 4);
}

#[test]
fn test_zero_batch_size_errors() {
    let schema = schema_of(vec![NestedField::required(
        1,
        "a",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let bytes = write_avro(&schema, vec![rec(vec![("a", AvroWriteValue::Int(1))])]);
    assert!(read_avro_data_bytes(&bytes, &schema, 0).is_err());
}

// -------------------------------------------------------------------------------------------------
// Nested: struct, list, map
// -------------------------------------------------------------------------------------------------

#[test]
fn test_nested_struct() {
    let inner = Type::Struct(StructType::new(vec![
        NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
        NestedField::optional(11, "y", Type::Primitive(PrimitiveType::String)).into(),
    ]));
    let schema = schema_of(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)),
        NestedField::required(2, "s", inner),
    ]);

    let bytes = write_avro(&schema, vec![rec(vec![
        ("id", AvroWriteValue::Long(1)),
        (
            "s",
            rec(vec![
                ("x", AvroWriteValue::Int(99)),
                ("y", opt(AvroWriteValue::String("nested".into()))),
            ]),
        ),
    ])]);

    let batch = decode_one(&bytes, &schema);
    let s = batch.column(1).as_struct();
    assert_eq!(s.column(0).as_primitive::<Int32Type>().value(0), 99);
    assert_eq!(s.column(1).as_string::<i32>().value(0), "nested");
}

#[test]
fn test_nested_list() {
    let schema = schema_of(vec![NestedField::required(
        1,
        "nums",
        Type::List(ListType {
            element_field: NestedField::list_element(10, Type::Primitive(PrimitiveType::Int), true)
                .into(),
        }),
    )]);

    let bytes = write_avro(&schema, vec![
        rec(vec![(
            "nums",
            AvroWriteValue::Array(vec![
                AvroWriteValue::Int(1),
                AvroWriteValue::Int(2),
                AvroWriteValue::Int(3),
            ]),
        )]),
        rec(vec![(
            "nums",
            AvroWriteValue::Array(vec![AvroWriteValue::Int(9)]),
        )]),
    ]);

    let batch = decode_one(&bytes, &schema);
    let list = batch.column(0).as_list::<i32>();
    let row0 = list.value(0);
    assert_eq!(row0.as_primitive::<Int32Type>().values(), &[1, 2, 3]);
    let row1 = list.value(1);
    assert_eq!(row1.as_primitive::<Int32Type>().values(), &[9]);
}

#[test]
fn test_nested_map_string_keys() {
    let schema = schema_of(vec![NestedField::required(
        1,
        "m",
        Type::Map(MapType {
            key_field: NestedField::map_key_element(10, Type::Primitive(PrimitiveType::String))
                .into(),
            value_field: NestedField::map_value_element(
                11,
                Type::Primitive(PrimitiveType::Int),
                true,
            )
            .into(),
        }),
    )]);

    let mut entries = std::collections::HashMap::new();
    entries.insert("a".to_string(), AvroWriteValue::Int(1));
    let bytes = write_avro(&schema, vec![rec(vec![(
        "m",
        AvroWriteValue::Map(entries),
    )])]);

    let batch = decode_one(&bytes, &schema);
    let map = batch.column(0).as_map();
    assert_eq!(map.value_length(0), 1);
    let keys = map.keys().as_string::<i32>();
    assert_eq!(keys.value(0), "a");
    let vals = map.values().as_primitive::<Int32Type>();
    assert_eq!(vals.value(0), 1);
}

// -------------------------------------------------------------------------------------------------
// Nested struct: resolution BY FIELD-ID at every level (projection / reorder / skip / default)
//
// These are the tests that would have caught a positional nested read. Each writes the FILE with a
// different inner-struct shape than the EXPECTED projection, so positional == by-id never coincides.
// -------------------------------------------------------------------------------------------------

/// MUTATION BAIT (nested by-field-id projection): the file inner struct is `[id10=int, id11=string]`
/// but we project only the inner `id11=string`. A positional read would grab writer position 0
/// (the int) and either error or read the wrong child. By-field-id must resolve id11 → writer
/// position 1 → "the-string". This is the critic's exact reproduction.
#[test]
fn test_nested_struct_projects_inner_subset_by_id() {
    // FILE: outer {1: id long, 2: s {10: x int, 11: y string}}
    let file_schema = schema_of(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)),
        NestedField::required(
            2,
            "s",
            Type::Struct(StructType::new(vec![
                NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(11, "y", Type::Primitive(PrimitiveType::String)).into(),
            ])),
        ),
    ]);
    let bytes = write_avro(&file_schema, vec![rec(vec![
        ("id", AvroWriteValue::Long(7)),
        (
            "s",
            rec(vec![
                ("x", AvroWriteValue::Int(99)),
                ("y", AvroWriteValue::String("the-string".into())),
            ]),
        ),
    ])]);

    // EXPECTED: project only the inner id 11 (drop inner id 10). Outer order also reduced to just s.
    let expected = schema_of(vec![NestedField::required(
        2,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(11, "y", Type::Primitive(PrimitiveType::String)).into(),
        ])),
    )]);

    let batch = decode_one(&bytes, &expected);
    assert_eq!(batch.num_columns(), 1);
    let s = batch.column(0).as_struct();
    assert_eq!(s.num_columns(), 1, "only inner id 11 is projected");
    assert_eq!(
        s.column(0).as_string::<i32>().value(0),
        "the-string",
        "must resolve inner id 11 by field-id (writer position 1), not positionally"
    );
}

/// Nested struct children REORDERED relative to the file: file inner is `[10:x int, 11:y string]`,
/// expected inner is `[11:y string, 10:x int]`. A positional read would swap the columns.
#[test]
fn test_nested_struct_reordered_children() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(11, "y", Type::Primitive(PrimitiveType::String)).into(),
        ])),
    )]);
    let bytes = write_avro(&file_schema, vec![rec(vec![(
        "s",
        rec(vec![
            ("x", AvroWriteValue::Int(42)),
            ("y", AvroWriteValue::String("yy".into())),
        ]),
    )])]);

    // Expected inner reorders y before x.
    let expected = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(11, "y", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
        ])),
    )]);

    let batch = decode_one(&bytes, &expected);
    let s = batch.column(0).as_struct();
    assert_eq!(s.column(0).as_string::<i32>().value(0), "yy");
    assert_eq!(s.column(1).as_primitive::<Int32Type>().value(0), 42);
}

/// Nested struct with an EXTRA file child skipped + a MISSING nested optional filled with null.
/// File inner = `[10:x int, 12:extra string]`; expected inner = `[10:x int, 11:absent_opt string?]`.
/// id 12 is decoded-and-discarded; id 11 is absent from the file and optional → null. A positional
/// read would wrongly route id 12's string into id 11's slot.
#[test]
fn test_nested_struct_skip_extra_and_missing_optional() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(12, "extra", Type::Primitive(PrimitiveType::String)).into(),
        ])),
    )]);
    let bytes = write_avro(&file_schema, vec![rec(vec![(
        "s",
        rec(vec![
            ("x", AvroWriteValue::Int(5)),
            ("extra", AvroWriteValue::String("discard-me".into())),
        ]),
    )])]);

    let expected = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(11, "absent_opt", Type::Primitive(PrimitiveType::String)).into(),
        ])),
    )]);

    let batch = decode_one(&bytes, &expected);
    let s = batch.column(0).as_struct();
    assert_eq!(s.num_columns(), 2);
    assert_eq!(s.column(0).as_primitive::<Int32Type>().value(0), 5);
    assert!(
        s.column(1).as_string::<i32>().is_null(0),
        "missing nested optional id 11 must be null, not the skipped id 12 string"
    );
}

/// Nested struct with a MISSING required child carrying a V3 initial-default (priority rung 2).
#[test]
fn test_nested_struct_missing_required_uses_initial_default() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
        ])),
    )]);
    let bytes = write_avro(&file_schema, vec![rec(vec![(
        "s",
        rec(vec![("x", AvroWriteValue::Int(1))]),
    )])]);

    let expected = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(11, "dflt", Type::Primitive(PrimitiveType::Long))
                .with_initial_default(Literal::long(555))
                .into(),
        ])),
    )]);

    let batch = decode_one(&bytes, &expected);
    let s = batch.column(0).as_struct();
    assert_eq!(s.column(1).as_primitive::<Int64Type>().value(0), 555);
}

/// Nested struct with a MISSING required child and no default → "Missing required field".
#[test]
fn test_nested_struct_missing_required_no_default_errors() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
        ])),
    )]);
    let bytes = write_avro(&file_schema, vec![rec(vec![(
        "s",
        rec(vec![("x", AvroWriteValue::Int(1))]),
    )])]);

    let expected = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(10, "x", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(11, "need", Type::Primitive(PrimitiveType::String)).into(),
        ])),
    )]);

    let err = read_avro_data_bytes(&bytes, &expected, 1024).expect_err("must error");
    assert!(
        err.to_string().contains("Missing required field: need"),
        "unexpected error: {err}"
    );
}

/// Nested struct int→long promotion at the nested level (expected inner type widens the file's int).
#[test]
fn test_nested_struct_int_to_long_promotion() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(10, "n", Type::Primitive(PrimitiveType::Int)).into(),
        ])),
    )]);
    let bytes = write_avro(&file_schema, vec![rec(vec![(
        "s",
        rec(vec![("n", AvroWriteValue::Int(321))]),
    )])]);

    let expected = schema_of(vec![NestedField::required(
        1,
        "s",
        Type::Struct(StructType::new(vec![
            NestedField::required(10, "n", Type::Primitive(PrimitiveType::Long)).into(),
        ])),
    )]);

    let batch = decode_one(&bytes, &expected);
    let s = batch.column(0).as_struct();
    assert_eq!(s.column(0).as_primitive::<Int64Type>().value(0), 321);
}

// -------------------------------------------------------------------------------------------------
// Nested struct INSIDE a list element and INSIDE a map value — field-id resolution must hold there
// -------------------------------------------------------------------------------------------------

/// List of struct, where the element struct is projected to an inner subset BY FIELD-ID. File
/// element = `[20:a int, 21:b string]`; expected element = `[21:b string]`. Positional would read
/// the int.
#[test]
fn test_list_of_struct_projects_element_subset_by_id() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "items",
        Type::List(ListType {
            element_field: NestedField::list_element(
                10,
                Type::Struct(StructType::new(vec![
                    NestedField::required(20, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(21, "b", Type::Primitive(PrimitiveType::String)).into(),
                ])),
                true,
            )
            .into(),
        }),
    )]);
    let bytes = write_avro(&file_schema, vec![rec(vec![(
        "items",
        AvroWriteValue::Array(vec![
            rec(vec![
                ("a", AvroWriteValue::Int(1)),
                ("b", AvroWriteValue::String("first".into())),
            ]),
            rec(vec![
                ("a", AvroWriteValue::Int(2)),
                ("b", AvroWriteValue::String("second".into())),
            ]),
        ]),
    )])]);

    let expected = schema_of(vec![NestedField::required(
        1,
        "items",
        Type::List(ListType {
            element_field: NestedField::list_element(
                10,
                Type::Struct(StructType::new(vec![
                    NestedField::required(21, "b", Type::Primitive(PrimitiveType::String)).into(),
                ])),
                true,
            )
            .into(),
        }),
    )]);

    let batch = decode_one(&bytes, &expected);
    let list = batch.column(0).as_list::<i32>();
    let elems = list.value(0);
    let s = elems.as_struct();
    assert_eq!(s.num_columns(), 1, "only element inner id 21 is projected");
    let b = s.column(0).as_string::<i32>();
    assert_eq!(b.value(0), "first");
    assert_eq!(b.value(1), "second");
}

/// String-keyed map whose VALUE is a struct projected to an inner subset BY FIELD-ID. File value =
/// `[30:p int, 31:q string]`; expected value = `[31:q string]`.
#[test]
fn test_map_of_struct_value_projects_subset_by_id() {
    let file_schema = schema_of(vec![NestedField::required(
        1,
        "m",
        Type::Map(MapType {
            key_field: NestedField::map_key_element(10, Type::Primitive(PrimitiveType::String))
                .into(),
            value_field: NestedField::map_value_element(
                11,
                Type::Struct(StructType::new(vec![
                    NestedField::required(30, "p", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(31, "q", Type::Primitive(PrimitiveType::String)).into(),
                ])),
                true,
            )
            .into(),
        }),
    )]);
    let mut entries = std::collections::HashMap::new();
    entries.insert(
        "k".to_string(),
        rec(vec![
            ("p", AvroWriteValue::Int(9)),
            ("q", AvroWriteValue::String("val".into())),
        ]),
    );
    let bytes = write_avro(&file_schema, vec![rec(vec![(
        "m",
        AvroWriteValue::Map(entries),
    )])]);

    let expected = schema_of(vec![NestedField::required(
        1,
        "m",
        Type::Map(MapType {
            key_field: NestedField::map_key_element(10, Type::Primitive(PrimitiveType::String))
                .into(),
            value_field: NestedField::map_value_element(
                11,
                Type::Struct(StructType::new(vec![
                    NestedField::required(31, "q", Type::Primitive(PrimitiveType::String)).into(),
                ])),
                true,
            )
            .into(),
        }),
    )]);

    let batch = decode_one(&bytes, &expected);
    let map = batch.column(0).as_map();
    assert_eq!(map.value_length(0), 1);
    assert_eq!(map.keys().as_string::<i32>().value(0), "k");
    let vals = map.values().as_struct();
    assert_eq!(vals.num_columns(), 1, "only value inner id 31 is projected");
    assert_eq!(vals.column(0).as_string::<i32>().value(0), "val");
}

// -------------------------------------------------------------------------------------------------
// Mutation baits: date units & uuid byte order
// -------------------------------------------------------------------------------------------------

/// MUTATION BAIT (date units): the Avro/Arrow contract is DAYS since epoch. A value of 19000 days
/// must read as exactly 19000 (Date32 = days). If the reader treated the int as anything but raw
/// days (e.g. millis), this reds.
#[test]
fn test_date_is_days_bait() {
    let schema = schema_of(vec![NestedField::required(
        1,
        "dt",
        Type::Primitive(PrimitiveType::Date),
    )]);
    let bytes = write_avro(&schema, vec![rec(vec![(
        "dt",
        AvroWriteValue::Date(19_000),
    )])]);
    let batch = decode_one(&bytes, &schema);
    assert_eq!(
        batch.column(0).as_primitive::<Date32Type>().value(0),
        19_000
    );
}

/// MUTATION BAIT (uuid byte order): the 16 bytes must be preserved big-endian exactly. A
/// byte-swapped read would not equal `uuid.as_bytes()`.
#[test]
fn test_uuid_byte_order_bait() {
    let schema = schema_of(vec![NestedField::required(
        1,
        "uu",
        Type::Primitive(PrimitiveType::Uuid),
    )]);
    let uuid = uuid::Uuid::from_u128(0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10);
    let bytes = write_avro(&schema, vec![rec(vec![("uu", AvroWriteValue::Uuid(uuid))])]);
    let batch = decode_one(&bytes, &schema);
    let uu = batch.column(0).as_fixed_size_binary();
    assert_eq!(uu.value(0), &[
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10
    ]);
}

// -------------------------------------------------------------------------------------------------
// Output schema equals schema_to_arrow_schema(expected)
// -------------------------------------------------------------------------------------------------

#[test]
fn test_output_arrow_schema_matches_expected() {
    let schema = schema_of(vec![
        NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "b", Type::Primitive(PrimitiveType::String)),
    ]);
    let bytes = write_avro(&schema, vec![rec(vec![
        ("a", AvroWriteValue::Int(1)),
        ("b", opt(AvroWriteValue::String("x".into()))),
    ])]);
    let batch = decode_one(&bytes, &schema);
    let want = Arc::new(schema_to_arrow_schema(&schema).expect("arrow schema"));
    assert_eq!(batch.schema(), want);
}

// -------------------------------------------------------------------------------------------------
// No-field-id file errors (known gap: name-mapping fallback is deferred)
// -------------------------------------------------------------------------------------------------

#[test]
fn test_missing_field_id_errors() {
    // Hand-build an Avro schema WITHOUT field-id props.
    let avro_schema = apache_avro::Schema::parse_str(
        r#"{"type":"record","name":"data","fields":[{"name":"a","type":"int"}]}"#,
    )
    .expect("parse avro schema");
    let mut writer = AvroWriter::new(&avro_schema, Vec::new());
    writer
        .append_value_ref(&rec(vec![("a", AvroWriteValue::Int(1))]))
        .expect("write row");
    let bytes = writer.into_inner().expect("finalize");

    let expected = schema_of(vec![NestedField::required(
        1,
        "a",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let err = read_avro_data_bytes(&bytes, &expected, 1024).expect_err("must error on no field-id");
    assert!(
        err.to_string().contains("field-id"),
        "unexpected error: {err}"
    );
}
