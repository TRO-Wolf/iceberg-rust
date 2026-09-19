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

use arrow_array::builder::PrimitiveBuilder;
use arrow_array::types::Decimal128Type;
use arrow_array::{ArrayRef, Decimal128Array, Int32Array, Int64Array};

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
fn test_truncate_transform() {
    let trans = Transform::Truncate(4);

    let fixture = TestTransformFixture {
        display: "truncate[4]".to_string(),
        json: r#""truncate[4]""#.to_string(),
        dedup_name: "truncate[4]".to_string(),
        preserves_order: true,
        satisfies_order_of: vec![
            (Transform::Truncate(4), true),
            (Transform::Truncate(2), false),
            (Transform::Bucket(4), false),
            (Transform::Void, false),
            (Transform::Day, false),
        ],
        trans_types: vec![
            (Primitive(Binary), Some(Primitive(Binary))),
            (Primitive(Date), None),
            (
                Primitive(Decimal {
                    precision: 8,
                    scale: 5,
                }),
                Some(Primitive(Decimal {
                    precision: 8,
                    scale: 5,
                })),
            ),
            (Primitive(Fixed(8)), None),
            (Primitive(Int), Some(Primitive(Int))),
            (Primitive(Long), Some(Primitive(Long))),
            (Primitive(StringType), Some(Primitive(StringType))),
            (Primitive(Uuid), None),
            (Primitive(Time), None),
            (Primitive(Timestamp), None),
            (Primitive(Timestamptz), None),
            (Primitive(TimestampNs), None),
            (Primitive(TimestamptzNs), None),
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
fn test_projection_truncate_string_rewrite_op() -> Result<()> {
    let value = "abcde";

    let fixture = TestProjectionFixture::new(
        Transform::Truncate(5),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::String)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::StartsWith, Datum::string(value)),
        Some(r#"name = "abcde""#),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotStartsWith, Datum::string(value)),
        Some(r#"name != "abcde""#),
    )?;

    let value = "abcdefg";
    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::StartsWith, Datum::string(value)),
        Some(r#"name STARTS WITH "abcde""#),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotStartsWith, Datum::string(value)),
        None,
    )?;

    Ok(())
}

#[test]
fn test_projection_truncate_string() -> Result<()> {
    let value = "abcdefg";

    let fixture = TestProjectionFixture::new(
        Transform::Truncate(5),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::String)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::string(value)),
        Some(r#"name <= "abcde""#),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::string(value)),
        Some(r#"name <= "abcde""#),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThan, Datum::string(value)),
        Some(r#"name >= "abcde""#),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThanOrEq, Datum::string(value)),
        Some(r#"name >= "abcde""#),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::string(value)),
        Some(r#"name = "abcde""#),
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::string(value),
            Datum::string(format!("{value}abc")),
        ]),
        Some(r#"name IN ("abcde")"#),
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::NotIn, vec![
            Datum::string(value),
            Datum::string(format!("{value}abc")),
        ]),
        None,
    )?;

    Ok(())
}

#[test]
fn test_projection_truncate_upper_bound_decimal() -> Result<()> {
    let prev = "98.99";
    let curr = "99.99";
    let next = "100.99";

    let fixture = TestProjectionFixture::new(
        Transform::Truncate(10),
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
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::decimal_from_str(curr)?),
        Some("name <= 9990"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(
            PredicateOperator::LessThanOrEq,
            Datum::decimal_from_str(curr)?,
        ),
        Some("name <= 9990"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(
            PredicateOperator::GreaterThanOrEq,
            Datum::decimal_from_str(curr)?,
        ),
        Some("name >= 9990"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::decimal_from_str(curr)?),
        Some("name = 9990"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::decimal_from_str(curr)?),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::decimal_from_str(prev)?,
            Datum::decimal_from_str(curr)?,
            Datum::decimal_from_str(next)?,
        ]),
        Some("name IN (9890, 9990, 10090)"),
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
fn test_projection_truncate_lower_bound_decimal() -> Result<()> {
    let prev = "99.00";
    let curr = "100.00";
    let next = "101.00";

    let fixture = TestProjectionFixture::new(
        Transform::Truncate(10),
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
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::decimal_from_str(curr)?),
        Some("name <= 9990"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(
            PredicateOperator::LessThanOrEq,
            Datum::decimal_from_str(curr)?,
        ),
        Some("name <= 10000"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(
            PredicateOperator::GreaterThanOrEq,
            Datum::decimal_from_str(curr)?,
        ),
        Some("name >= 10000"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::decimal_from_str(curr)?),
        Some("name = 10000"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::decimal_from_str(curr)?),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::decimal_from_str(prev)?,
            Datum::decimal_from_str(curr)?,
            Datum::decimal_from_str(next)?,
        ]),
        Some("name IN (10000, 10100, 9900)"),
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
fn test_projection_truncate_upper_bound_long() -> Result<()> {
    let value = 99i64;

    let fixture = TestProjectionFixture::new(
        Transform::Truncate(10),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::Long)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::long(value)),
        Some("name <= 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::long(value)),
        Some("name <= 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThanOrEq, Datum::long(value)),
        Some("name >= 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::long(value)),
        Some("name = 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::long(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::long(value - 1),
            Datum::long(value),
            Datum::long(value + 1),
        ]),
        Some("name IN (100, 90)"),
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
fn test_projection_truncate_lower_bound_long() -> Result<()> {
    let value = 100i64;

    let fixture = TestProjectionFixture::new(
        Transform::Truncate(10),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::Long)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::long(value)),
        Some("name <= 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::long(value)),
        Some("name <= 100"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThanOrEq, Datum::long(value)),
        Some("name >= 100"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::long(value)),
        Some("name = 100"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::long(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::long(value - 1),
            Datum::long(value),
            Datum::long(value + 1),
        ]),
        Some("name IN (100, 90)"),
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
fn test_projection_truncate_upper_bound_integer() -> Result<()> {
    let value = 99;

    let fixture = TestProjectionFixture::new(
        Transform::Truncate(10),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::Int)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::int(value)),
        Some("name <= 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::int(value)),
        Some("name <= 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThanOrEq, Datum::int(value)),
        Some("name >= 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::int(value)),
        Some("name = 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::int(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::int(value - 1),
            Datum::int(value),
            Datum::int(value + 1),
        ]),
        Some("name IN (100, 90)"),
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
fn test_projection_truncate_lower_bound_integer() -> Result<()> {
    let value = 100;

    let fixture = TestProjectionFixture::new(
        Transform::Truncate(10),
        "name",
        NestedField::required(1, "value", Type::Primitive(PrimitiveType::Int)),
    );

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThan, Datum::int(value)),
        Some("name <= 90"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::LessThanOrEq, Datum::int(value)),
        Some("name <= 100"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::GreaterThanOrEq, Datum::int(value)),
        Some("name >= 100"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::Eq, Datum::int(value)),
        Some("name = 100"),
    )?;

    fixture.assert_projection(
        &fixture.binary_predicate(PredicateOperator::NotEq, Datum::int(value)),
        None,
    )?;

    fixture.assert_projection(
        &fixture.set_predicate(PredicateOperator::In, vec![
            Datum::int(value - 1),
            Datum::int(value),
            Datum::int(value + 1),
        ]),
        Some("name IN (100, 90)"),
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
fn test_truncate_new_rejects_zero_width() {
    let error = super::Truncate::new(0).expect_err("truncate width 0 must be rejected");
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("Invalid truncate width: 0 (must be > 0)"),
        "message must match the Java precondition text, got: {}",
        error.message()
    );
}

#[test]
fn test_truncate_new_rejects_width_above_java_int_max() {
    let error = super::Truncate::new(2147483648)
        .expect_err("truncate width above i32::MAX must be rejected");
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        error.message().contains("must be <= 2147483647"),
        "message must name the Java int bound, got: {}",
        error.message()
    );
}

#[test]
fn test_truncate_at_java_int_max_accepted_and_produces_exact_value() {
    let truncate = super::Truncate::new(2147483647).expect("truncate[i32::MAX] is legal");
    assert_eq!(
        truncate
            .transform_literal(&Datum::int(1))
            .expect("int is truncatable")
            .expect("truncate of a non-null value is non-null"),
        Datum::int(0)
    );
}

#[test]
fn test_downcast_input_is_a_typed_error_not_a_panic() {
    let mislabelled: ArrayRef = Arc::new(arrow_array::StringArray::from(vec!["not an int"]));
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
fn test_truncate_simple() {
    let input = Arc::new(Int32Array::from(vec![1, -1]));
    let res = super::Truncate::new(10)
        .expect("truncate width is within 1..=i32::MAX")
        .transform(input)
        .unwrap();
    assert_eq!(
        res.as_any().downcast_ref::<Int32Array>().unwrap().value(0),
        0
    );
    assert_eq!(
        res.as_any().downcast_ref::<Int32Array>().unwrap().value(1),
        -10
    );

    let input = Arc::new(Int64Array::from(vec![1, -1]));
    let res = super::Truncate::new(10)
        .expect("truncate width is within 1..=i32::MAX")
        .transform(input)
        .unwrap();
    assert_eq!(
        res.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        0
    );
    assert_eq!(
        res.as_any().downcast_ref::<Int64Array>().unwrap().value(1),
        -10
    );

    let mut builder = PrimitiveBuilder::<Decimal128Type>::new()
        .with_precision_and_scale(20, 2)
        .unwrap();
    builder.append_value(1065);
    let input = Arc::new(builder.finish());
    let res = super::Truncate::new(50)
        .expect("truncate width is within 1..=i32::MAX")
        .transform(input)
        .unwrap();
    assert_eq!(
        res.as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap()
            .value(0),
        1050
    );

    let input = Arc::new(arrow_array::StringArray::from(vec!["iceberg"]));
    let res = super::Truncate::new(3)
        .expect("truncate width is within 1..=i32::MAX")
        .transform(input)
        .unwrap();
    assert_eq!(
        res.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap()
            .value(0),
        "ice"
    );

    let input = Arc::new(arrow_array::LargeStringArray::from(vec!["iceberg"]));
    let res = super::Truncate::new(3)
        .expect("truncate width is within 1..=i32::MAX")
        .transform(input)
        .unwrap();
    assert_eq!(
        res.as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
            .unwrap()
            .value(0),
        "ice"
    );

    let input = Arc::new(arrow_array::BinaryArray::from_vec(vec![b"iceberg"]));
    let res = super::Truncate::new(3)
        .expect("truncate width is within 1..=i32::MAX")
        .transform(input)
        .unwrap();
    assert_eq!(
        res.as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .unwrap()
            .value(0),
        b"ice"
    );
}

#[test]
fn test_string_truncate() {
    let test1 = "イロハニホヘト";
    let test1_2_expected = "イロ";
    assert_eq!(super::Truncate::truncate_str(test1, 2), test1_2_expected);

    let test1_3_expected = "イロハ";
    assert_eq!(super::Truncate::truncate_str(test1, 3), test1_3_expected);

    let test2 = "щщаεはчωいにπάほхεろへσκζ";
    let test2_7_expected = "щщаεはчω";
    assert_eq!(super::Truncate::truncate_str(test2, 7), test2_7_expected);

    let test3 = "\u{FFFF}\u{FFFF}";
    assert_eq!(super::Truncate::truncate_str(test3, 2), test3);

    let test4 = "\u{10000}\u{10000}";
    let test4_1_expected = "\u{10000}";
    assert_eq!(super::Truncate::truncate_str(test4, 1), test4_1_expected);
}

#[test]
fn test_literal_int() {
    let input = Datum::int(1);
    let res = super::Truncate::new(10)
        .expect("truncate width is within 1..=i32::MAX")
        .transform_literal(&input)
        .unwrap()
        .unwrap();
    assert_eq!(res, Datum::int(0),);

    let input = Datum::int(-1);
    let res = super::Truncate::new(10)
        .expect("truncate width is within 1..=i32::MAX")
        .transform_literal(&input)
        .unwrap()
        .unwrap();
    assert_eq!(res, Datum::int(-10),);
}

#[test]
fn test_literal_long() {
    let input = Datum::long(1);
    let res = super::Truncate::new(10)
        .expect("truncate width is within 1..=i32::MAX")
        .transform_literal(&input)
        .unwrap()
        .unwrap();
    assert_eq!(res, Datum::long(0),);

    let input = Datum::long(-1);
    let res = super::Truncate::new(10)
        .expect("truncate width is within 1..=i32::MAX")
        .transform_literal(&input)
        .unwrap()
        .unwrap();
    assert_eq!(res, Datum::long(-10),);
}

#[test]
fn test_decimal_literal() {
    let input = Datum::decimal(decimal_new(1065, 0)).unwrap();
    let res = super::Truncate::new(50)
        .expect("truncate width is within 1..=i32::MAX")
        .transform_literal(&input)
        .unwrap()
        .unwrap();
    assert_eq!(res, Datum::decimal(decimal_new(1050, 0)).unwrap(),);
}

#[test]
fn test_string_literal() {
    let input = Datum::string("iceberg".to_string());
    let res = super::Truncate::new(3)
        .expect("truncate width is within 1..=i32::MAX")
        .transform_literal(&input)
        .unwrap()
        .unwrap();
    assert_eq!(res, Datum::string("ice".to_string()),);
}
