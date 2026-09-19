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

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, TimestampMicrosecondArray, TimestampNanosecondArray};
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime};

use super::Bucket;
use crate::Result;
use crate::expr::PredicateOperator;
use crate::spec::PrimitiveType::{
    Binary, Date, Decimal, Fixed, Int, Long, String as StringType, Time, Timestamp, TimestampNs,
    Timestamptz, TimestamptzNs, Uuid,
};
use crate::spec::Type::{Primitive, Struct};
use crate::spec::decimal_utils::decimal_new;
use crate::spec::{Datum, NestedField, PrimitiveType, StructType, Transform, Type};
use crate::transform::TransformFunction;
use crate::transform::test::{TestProjectionFixture, TestTransformFixture};

#[test]
fn test_bucket_transform() {
    let trans = Transform::Bucket(8);

    let fixture = TestTransformFixture {
        display: "bucket[8]".to_string(),
        json: r#""bucket[8]""#.to_string(),
        dedup_name: "bucket[8]".to_string(),
        preserves_order: false,
        satisfies_order_of: vec![
            (Transform::Bucket(8), true),
            (Transform::Bucket(4), false),
            (Transform::Void, false),
            (Transform::Day, false),
        ],
        trans_types: vec![
            (Primitive(Binary), Some(Primitive(Int))),
            (Primitive(Date), Some(Primitive(Int))),
            (
                Primitive(Decimal {
                    precision: 8,
                    scale: 5,
                }),
                Some(Primitive(Int)),
            ),
            (Primitive(Fixed(8)), Some(Primitive(Int))),
            (Primitive(Int), Some(Primitive(Int))),
            (Primitive(Long), Some(Primitive(Int))),
            (Primitive(StringType), Some(Primitive(Int))),
            (Primitive(Uuid), Some(Primitive(Int))),
            (Primitive(Time), Some(Primitive(Int))),
            (Primitive(Timestamp), Some(Primitive(Int))),
            (Primitive(Timestamptz), Some(Primitive(Int))),
            (Primitive(TimestampNs), Some(Primitive(Int))),
            (Primitive(TimestamptzNs), Some(Primitive(Int))),
            (
                Struct(StructType::new(vec![
                    NestedField::optional(1, "a", Primitive(Timestamp)).into(),
                ])),
                None,
            ),
        ],
    };

    fixture.assert_transform(trans);
}

#[test]
fn test_projection_bucket_uuid() -> Result<()> {
    let value = uuid::Uuid::from_u64_pair(123, 456);
    let another = uuid::Uuid::from_u64_pair(456, 123);

    let fixture = TestProjectionFixture::new(
        Transform::Bucket(10),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::Uuid)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::uuid(value)),
        Some("name = 4"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::uuid(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::uuid(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::uuid(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThan, Datum::uuid(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThanOrEq, Datum::uuid(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::uuid(value),
            Datum::uuid(another),
        ]),
        Some("name IN (4, 6)"),
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::NotIn, vec![
            Datum::uuid(value),
            Datum::uuid(another),
        ]),
        None,
    )?;

    Ok(())
}

#[test]
fn test_projection_bucket_fixed() -> Result<()> {
    let value = "abcdefg".as_bytes().to_vec();
    let another = "abcdehij".as_bytes().to_vec();

    let fixture = TestProjectionFixture::new(
        Transform::Bucket(10),
        "name",
        NestedField::required(
            1,
            "value",
            Type::Primitive(PrimitiveType::Fixed(value.len() as u64)),
        ),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::fixed(value.clone())),
        Some("name = 4"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::fixed(value.clone())),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::fixed(value.clone())),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::fixed(value.clone())),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThan, Datum::fixed(value.clone())),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(
            PredicateOperator::GreaterThanOrEq,
            Datum::fixed(value.clone()),
        ),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::fixed(value.clone()),
            Datum::fixed(another.clone()),
        ]),
        Some("name IN (4, 6)"),
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::NotIn, vec![
            Datum::fixed(value.clone()),
            Datum::fixed(another.clone()),
        ]),
        None,
    )?;

    Ok(())
}

#[test]
fn test_projection_bucket_string() -> Result<()> {
    let value = "abcdefg";
    let another = "abcdefgabc";

    let fixture = TestProjectionFixture::new(
        Transform::Bucket(10),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::String)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::string(value)),
        Some("name = 4"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::string(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::string(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::string(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThan, Datum::string(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThanOrEq, Datum::string(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::string(value),
            Datum::string(another),
        ]),
        Some("name IN (9, 4)"),
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::NotIn, vec![
            Datum::string(value),
            Datum::string(another),
        ]),
        None,
    )?;

    Ok(())
}

#[test]
fn test_projection_bucket_decimal() -> Result<()> {
    let prev = "99.00";
    let curr = "100.00";
    let next = "101.00";

    let fixture = TestProjectionFixture::new(
        Transform::Bucket(10),
        "name",
        NestedField::required(
            1,
            "value",
            Type::Primitive(PrimitiveType::Decimal {
                precision: 9,
                scale: 2,
            }),
        ),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::decimal_from_str(curr)?),
        Some("name = 2"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::decimal_from_str(curr)?),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::decimal_from_str(curr)?),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(
            PredicateOperator::LessThanOrEq,
            Datum::decimal_from_str(curr)?,
        ),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(
            PredicateOperator::GreaterThan,
            Datum::decimal_from_str(curr)?,
        ),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(
            PredicateOperator::GreaterThanOrEq,
            Datum::decimal_from_str(curr)?,
        ),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::decimal_from_str(next)?,
            Datum::decimal_from_str(curr)?,
            Datum::decimal_from_str(prev)?,
        ]),
        Some("name IN (2, 6)"),
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::NotIn, vec![
            Datum::decimal_from_str(curr)?,
            Datum::decimal_from_str(next)?,
        ]),
        None,
    )?;

    Ok(())
}

#[test]
fn test_projection_bucket_long() -> Result<()> {
    let value = 100;
    let fixture = TestProjectionFixture::new(
        Transform::Bucket(10),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::Long)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::long(value)),
        Some("name = 6"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::long(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::long(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::long(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThan, Datum::long(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThanOrEq, Datum::long(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::long(value - 1),
            Datum::long(value),
            Datum::long(value + 1),
        ]),
        Some("name IN (8, 7, 6)"),
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::NotIn, vec![
            Datum::long(value),
            Datum::long(value + 1),
        ]),
        None,
    )?;

    Ok(())
}

#[test]
fn test_projection_bucket_integer() -> Result<()> {
    let value = 100;

    let fixture = TestProjectionFixture::new(
        Transform::Bucket(10),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::Int)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::int(value)),
        Some("name = 6"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::int(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::int(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::int(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThan, Datum::int(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThanOrEq, Datum::int(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::int(value - 1),
            Datum::int(value),
            Datum::int(value + 1),
        ]),
        Some("name IN (8, 7, 6)"),
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::NotIn, vec![
            Datum::int(value),
            Datum::int(value + 1),
        ]),
        None,
    )?;

    Ok(())
}

#[test]
fn test_hash() {
    assert_eq!(Bucket::hash_int(34), 2017239379);
    assert_eq!(Bucket::hash_long(34), 2017239379);
    assert_eq!(Bucket::hash_decimal(1420), -500754589);
    let date = NaiveDate::from_ymd_opt(2017, 11, 16).unwrap();
    assert_eq!(
        Bucket::hash_date(
            date.signed_duration_since(NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
                .num_days() as i32
        ),
        -653330422
    );
    let time = NaiveTime::from_hms_opt(22, 31, 8).unwrap();
    assert_eq!(
        Bucket::hash_time(
            time.signed_duration_since(NaiveTime::from_hms_opt(0, 0, 0).unwrap())
                .num_microseconds()
                .unwrap()
        ),
        -662762989
    );
    let timestamp =
        NaiveDateTime::parse_from_str("2017-11-16 22:31:08", "%Y-%m-%d %H:%M:%S").unwrap();
    assert_eq!(
        Bucket::hash_timestamp(
            timestamp
                .signed_duration_since(
                    NaiveDateTime::parse_from_str("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
                        .unwrap()
                )
                .num_microseconds()
                .unwrap()
        ),
        -2047944441
    );
    let timestamp = DateTime::parse_from_rfc3339("2017-11-16T14:31:08-08:00").unwrap();
    assert_eq!(
        Bucket::hash_timestamp(
            timestamp
                .signed_duration_since(
                    DateTime::parse_from_rfc3339("1970-01-01T00:00:00-00:00").unwrap()
                )
                .num_microseconds()
                .unwrap()
        ),
        -2047944441
    );
    assert_eq!(Bucket::hash_str("iceberg"), 1210000089);
    assert_eq!(
        Bucket::hash_bytes(
            [
                0xF7, 0x9C, 0x3E, 0x09, 0x67, 0x7C, 0x4B, 0xBD, 0xA4, 0x79, 0x3F, 0x34, 0x9C, 0xB7,
                0x85, 0xE7
            ]
            .as_ref()
        ),
        1488055340
    );
    assert_eq!(
        Bucket::hash_bytes([0x00, 0x01, 0x02, 0x03].as_ref()),
        -188683207
    );
}

#[test]
fn test_hash_decimal_with_negative_value() {
    assert_eq!(Bucket::hash_decimal(1), -463810133);
    assert_eq!(Bucket::hash_decimal(-1), -43192051);

    assert_eq!(Bucket::hash_decimal(0), Bucket::hash_decimal(0));
    assert_eq!(Bucket::hash_decimal(127), Bucket::hash_decimal(127));
    assert_eq!(Bucket::hash_decimal(-128), Bucket::hash_decimal(-128));

    assert_eq!(Bucket::hash_decimal(128), Bucket::hash_bytes(&[0x00, 0x80]));
    assert_eq!(
        Bucket::hash_decimal(-129),
        Bucket::hash_bytes(&[0xFF, 0x7F])
    );
}

#[test]
fn test_bucket_new_rejects_zero_buckets() {
    let error = Bucket::new(0).expect_err("bucket count 0 must be rejected");
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("Invalid number of buckets: 0 (must be > 0)"),
        "message must match the Java precondition text, got: {}",
        error.message()
    );
}

#[test]
fn test_bucket_new_rejects_count_above_java_int_max() {
    let error = Bucket::new(2147483648).expect_err("bucket count above i32::MAX must be rejected");
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        error.message().contains("must be <= 2147483647"),
        "message must name the Java int bound, got: {}",
        error.message()
    );
}

#[test]
fn test_bucket_at_java_int_max_accepted_and_produces_exact_value() {
    let bucket = Bucket::new(2147483647).expect("bucket[i32::MAX] is legal");
    assert_eq!(
        bucket
            .transform_literal(&Datum::int(34))
            .expect("int is bucketable")
            .expect("bucket of a non-null value is non-null"),
        Datum::int(2017239379)
    );
}

#[test]
fn test_int_literal() {
    let bucket = Bucket::new(10).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket.transform_literal(&Datum::int(34)).unwrap().unwrap(),
        Datum::int(9)
    );
}

#[test]
fn test_long_literal() {
    let bucket = Bucket::new(10).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket.transform_literal(&Datum::long(34)).unwrap().unwrap(),
        Datum::int(9)
    );
}

#[test]
fn test_decimal_literal() {
    let bucket = Bucket::new(10).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket
            .transform_literal(&Datum::decimal(decimal_new(1420, 0)).unwrap())
            .unwrap()
            .unwrap(),
        Datum::int(9)
    );
}

#[test]
fn test_date_literal() {
    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket
            .transform_literal(&Datum::date(17486))
            .unwrap()
            .unwrap(),
        Datum::int(26)
    );
}

#[test]
fn test_time_literal() {
    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket
            .transform_literal(&Datum::time_micros(81068000000).unwrap())
            .unwrap()
            .unwrap(),
        Datum::int(59)
    );
}

#[test]
fn test_timestamp_literal() {
    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket
            .transform_literal(&Datum::timestamp_micros(1510871468000000))
            .unwrap()
            .unwrap(),
        Datum::int(7)
    );
}

#[test]
fn test_str_literal() {
    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket
            .transform_literal(&Datum::string("iceberg"))
            .unwrap()
            .unwrap(),
        Datum::int(89)
    );
}

#[test]
fn test_uuid_literal() {
    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket
            .transform_literal(&Datum::uuid(
                "F79C3E09-677C-4BBD-A479-3F349CB785E7".parse().unwrap()
            ))
            .unwrap()
            .unwrap(),
        Datum::int(40)
    );
}

#[test]
fn test_binary_literal() {
    let bucket = Bucket::new(128).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket
            .transform_literal(&Datum::binary(b"\x00\x01\x02\x03".to_vec()))
            .unwrap()
            .unwrap(),
        Datum::int(57)
    );
}

#[test]
fn test_fixed_literal() {
    let bucket = Bucket::new(128).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket
            .transform_literal(&Datum::fixed(b"foo".to_vec()))
            .unwrap()
            .unwrap(),
        Datum::int(32)
    );
}

#[test]
fn test_timestamptz_literal() {
    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");
    assert_eq!(
        bucket
            .transform_literal(&Datum::timestamptz_micros(1510871468000000))
            .unwrap()
            .unwrap(),
        Datum::int(7)
    );
}

#[test]
fn test_timestamp_ns_literal() {
    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");
    let ns_value = 1510871468000000i64 * 1000;
    assert_eq!(
        bucket
            .transform_literal(&Datum::timestamp_nanos(ns_value))
            .unwrap()
            .unwrap(),
        Datum::int(7)
    );
}

#[test]
fn test_timestamptz_ns_literal() {
    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");
    let ns_value = 1510871468000000i64 * 1000;
    assert_eq!(
        bucket
            .transform_literal(&Datum::timestamptz_nanos(ns_value))
            .unwrap()
            .unwrap(),
        Datum::int(7)
    );
}

#[test]
fn test_transform_array_matches_literal_for_every_supported_arm() {
    use arrow_array::{
        BinaryArray, Date32Array, Decimal128Array, FixedSizeBinaryArray, Int64Array,
        LargeBinaryArray, LargeStringArray, StringArray, Time64MicrosecondArray,
        Time64NanosecondArray,
    };

    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");

    let micros = 1510871468000000i64;
    let time_micros = 81068000000i64;
    let bytes = b"\x00\x01\x02\x03".to_vec();

    let cases: Vec<(ArrayRef, Datum)> = vec![
        (Arc::new(Int32Array::from(vec![34])), Datum::int(34)),
        (Arc::new(Int64Array::from(vec![34i64])), Datum::long(34)),
        (
            Arc::new(
                Decimal128Array::from(vec![1420i128])
                    .with_precision_and_scale(20, 0)
                    .expect("decimal precision/scale"),
            ),
            Datum::decimal(decimal_new(1420, 0)).expect("decimal datum"),
        ),
        (Arc::new(Date32Array::from(vec![17486])), Datum::date(17486)),
        (
            Arc::new(Time64MicrosecondArray::from(vec![time_micros])),
            Datum::time_micros(time_micros).expect("time datum"),
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![micros])),
            Datum::timestamp_micros(micros),
        ),
        (
            Arc::new(Time64NanosecondArray::from(vec![time_micros * 1000])),
            Datum::time_micros(time_micros).expect("time datum"),
        ),
        (
            Arc::new(TimestampNanosecondArray::from(vec![micros * 1000])),
            Datum::timestamp_nanos(micros * 1000),
        ),
        (
            Arc::new(StringArray::from(vec!["iceberg"])),
            Datum::string("iceberg"),
        ),
        (
            Arc::new(LargeStringArray::from(vec!["iceberg"])),
            Datum::string("iceberg"),
        ),
        (
            Arc::new(BinaryArray::from_vec(vec![&bytes])),
            Datum::binary(bytes.clone()),
        ),
        (
            Arc::new(LargeBinaryArray::from_vec(vec![&bytes])),
            Datum::binary(bytes.clone()),
        ),
        (
            Arc::new(
                FixedSizeBinaryArray::try_from_iter(vec![b"foo".to_vec()].into_iter())
                    .expect("fixed size binary array"),
            ),
            Datum::fixed(b"foo".to_vec()),
        ),
    ];

    for (array, literal) in cases {
        let data_type = array.data_type().clone();
        let transformed = bucket
            .transform(array)
            .unwrap_or_else(|e| panic!("{data_type:?} must be bucketable: {e}"));
        let transformed = transformed
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("{data_type:?} must bucket to an Int32Array"));
        let expected = bucket
            .transform_literal(&literal)
            .unwrap_or_else(|e| panic!("{literal} must be bucketable: {e}"))
            .unwrap_or_else(|| panic!("{literal} must bucket to a value"));

        assert_eq!(
            Datum::int(transformed.value(0)),
            expected,
            "the {data_type:?} array arm must bucket {literal} exactly as the literal path"
        );
    }
}

#[test]
fn test_downcast_input_is_a_typed_error_not_a_panic() {
    use arrow_array::StringArray;

    let mislabelled: ArrayRef = Arc::new(StringArray::from(vec!["not an int"]));
    let error = super::downcast_input::<Int32Array>(&mislabelled)
        .expect_err("a StringArray is not an Int32Array");

    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        error.to_string().contains("Utf8"),
        "the error must name the data type it was given: {error}"
    );
    assert!(
        error.to_string().contains("Int32"),
        "the error must name the array type it expected: {error}"
    );
}

#[test]
fn test_transform_timestamp_nanos_and_micros_array_equivalence() {
    let bucket = Bucket::new(100).expect("bucket count is within 1..=i32::MAX");
    let micros_value = 1510871468000000;
    let nanos_value = micros_value * 1000;

    let micro_array = TimestampMicrosecondArray::from_iter_values(vec![micros_value]);
    let nano_array = TimestampNanosecondArray::from_iter_values(vec![nanos_value]);

    let transformed_micro: ArrayRef = bucket.transform(Arc::new(micro_array)).unwrap();
    let transformed_nano: ArrayRef = bucket.transform(Arc::new(nano_array)).unwrap();

    let micro_result = transformed_micro
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let nano_result = transformed_nano
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();

    assert_eq!(micro_result.value(0), nano_result.value(0));
}
