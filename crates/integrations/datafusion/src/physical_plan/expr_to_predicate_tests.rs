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

use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::common::{Column, DFSchema};
use datafusion::logical_expr::expr::Cast;
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::prelude::{Expr, SessionContext, col, lit};
use datafusion::scalar::ScalarValue;
use iceberg::expr::{Predicate, Reference};
use iceberg::spec::{
    Datum, ListType, MapType, NestedField, PrimitiveType, Schema as IcebergSchema, SchemaRef,
    StructType, Type,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::convert_filters_to_predicate;

fn create_test_schema() -> DFSchema {
    let arrow_schema = Schema::new(vec![
        Field::new("foo", DataType::Int32, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("bar", DataType::Utf8, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
        Field::new("ts", DataType::Timestamp(TimeUnit::Second, None), true).with_metadata(
            HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "3".to_string())]),
        ),
        Field::new("qux", DataType::Float64, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "4".to_string(),
        )])),
        Field::new("flt", DataType::Float32, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "5".to_string(),
        )])),
        Field::new("bin", DataType::Binary, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "6".to_string(),
        )])),
        Field::new("d", DataType::Date32, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "7".to_string(),
        )])),
    ]);
    DFSchema::try_from_qualified_schema("my_table", &arrow_schema).unwrap()
}

pub(super) fn test_iceberg_schema() -> SchemaRef {
    IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "bar", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "ts", Type::Primitive(PrimitiveType::Timestamp)).into(),
            NestedField::optional(4, "qux", Type::Primitive(PrimitiveType::Double)).into(),
            NestedField::optional(5, "flt", Type::Primitive(PrimitiveType::Float)).into(),
            NestedField::optional(6, "bin", Type::Primitive(PrimitiveType::Binary)).into(),
            NestedField::optional(7, "d", Type::Primitive(PrimitiveType::Date)).into(),
        ])
        .build()
        .unwrap()
        .into()
}

fn convert_to_iceberg_predicate(sql: &str) -> Option<Predicate> {
    let df_schema = create_test_schema();
    let expr = SessionContext::new()
        .parse_sql_expr(sql, &df_schema)
        .unwrap();
    let exprs: Vec<Expr> = split_conjunction(&expr).into_iter().cloned().collect();
    convert_filters_to_predicate(&exprs[..], &test_iceberg_schema())
}

#[test]
fn test_predicate_conversion_with_single_condition() {
    let predicate = convert_to_iceberg_predicate("foo = 1").unwrap();
    assert_eq!(predicate, Reference::new("foo").equal_to(Datum::long(1)));

    let predicate = convert_to_iceberg_predicate("foo != 1").unwrap();
    assert_eq!(
        predicate,
        Reference::new("foo").not_equal_to(Datum::long(1))
    );

    let predicate = convert_to_iceberg_predicate("foo > 1").unwrap();
    assert_eq!(
        predicate,
        Reference::new("foo").greater_than(Datum::long(1))
    );

    let predicate = convert_to_iceberg_predicate("foo >= 1").unwrap();
    assert_eq!(
        predicate,
        Reference::new("foo").greater_than_or_equal_to(Datum::long(1))
    );

    let predicate = convert_to_iceberg_predicate("foo < 1").unwrap();
    assert_eq!(predicate, Reference::new("foo").less_than(Datum::long(1)));

    let predicate = convert_to_iceberg_predicate("foo <= 1").unwrap();
    assert_eq!(
        predicate,
        Reference::new("foo").less_than_or_equal_to(Datum::long(1))
    );

    let predicate = convert_to_iceberg_predicate("foo is null").unwrap();
    assert_eq!(predicate, Reference::new("foo").is_null());

    let predicate = convert_to_iceberg_predicate("foo is not null").unwrap();
    assert_eq!(predicate, Reference::new("foo").is_not_null());

    let predicate = convert_to_iceberg_predicate("foo in (5, 6)").unwrap();
    assert_eq!(
        predicate,
        Reference::new("foo").is_in([Datum::long(5), Datum::long(6)])
    );

    let predicate = convert_to_iceberg_predicate("foo not in (5, 6)").unwrap();
    assert_eq!(
        predicate,
        Reference::new("foo").is_not_in([Datum::long(5), Datum::long(6)])
    );

    let predicate = convert_to_iceberg_predicate("not foo = 1").unwrap();
    assert_eq!(predicate, !Reference::new("foo").equal_to(Datum::long(1)));
}

#[test]
fn test_predicate_conversion_with_single_unsupported_condition() {
    let predicate = convert_to_iceberg_predicate("foo + 1 = 1");
    assert_eq!(predicate, None);

    let predicate = convert_to_iceberg_predicate("length(bar) = 1");
    assert_eq!(predicate, None);

    let predicate = convert_to_iceberg_predicate("foo in (1, 2, foo)");
    assert_eq!(predicate, None);
}

#[test]
fn test_predicate_conversion_with_single_condition_rev() {
    let predicate = convert_to_iceberg_predicate("1 < foo").unwrap();
    assert_eq!(
        predicate,
        Reference::new("foo").greater_than(Datum::long(1))
    );
}

#[test]
fn test_predicate_conversion_with_and_condition() {
    let sql = "foo > 1 and bar = 'test'";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    let expected_predicate = Predicate::and(
        Reference::new("foo").greater_than(Datum::long(1)),
        Reference::new("bar").equal_to(Datum::string("test")),
    );
    assert_eq!(predicate, expected_predicate);
}

#[test]
fn test_predicate_conversion_with_and_condition_unsupported() {
    let sql = "foo > 1 and length(bar) = 1";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    let expected_predicate = Reference::new("foo").greater_than(Datum::long(1));
    assert_eq!(predicate, expected_predicate);
}

#[test]
fn test_predicate_conversion_with_and_condition_both_unsupported() {
    let sql = "foo in (1, 2, foo) and length(bar) = 1";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(predicate, None);
}

#[test]
fn test_predicate_conversion_with_or_condition_unsupported() {
    let sql = "foo > 1 or length(bar) = 1";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(predicate, None);
}

#[test]
fn test_predicate_conversion_with_or_condition_supported() {
    let sql = "foo > 1 or bar = 'test'";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    let expected_predicate = Predicate::or(
        Reference::new("foo").greater_than(Datum::long(1)),
        Reference::new("bar").equal_to(Datum::string("test")),
    );
    assert_eq!(predicate, expected_predicate);
}

#[test]
fn test_predicate_conversion_with_complex_binary_expr() {
    let sql = "(foo > 1 and bar = 'test') or foo < 0 ";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();

    let inner_predicate = Predicate::and(
        Reference::new("foo").greater_than(Datum::long(1)),
        Reference::new("bar").equal_to(Datum::string("test")),
    );
    let expected_predicate = Predicate::or(
        inner_predicate,
        Reference::new("foo").less_than(Datum::long(0)),
    );
    assert_eq!(predicate, expected_predicate);
}

#[test]
fn test_predicate_conversion_nested_partial_and_does_not_drop_a_side() {
    let sql = "(foo > 1 and length(bar) = 1 ) or foo < 0 ";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(
        predicate, None,
        "a nested AND with one unconverted side must not become the converted side"
    );
}

#[test]
fn test_predicate_conversion_not_over_partial_and_is_not_pushed() {
    let sql = "NOT (foo > 1 AND length(bar) = 1)";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(
        predicate, None,
        "NOT of a partial AND must not push NOT(foo > 1)"
    );
}

#[test]
fn test_predicate_conversion_with_complex_binary_expr_unsupported() {
    let sql = "(foo > 1 or length(bar) = 1 ) and foo < 0 ";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    let expected_predicate = Reference::new("foo").less_than(Datum::long(0));
    assert_eq!(predicate, expected_predicate);
}

#[test]
fn test_predicate_conversion_with_cast() {
    let sql = "ts >= timestamp '2023-01-05T00:00:00'";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    let expected_predicate =
        Reference::new("ts").greater_than_or_equal_to(Datum::string("2023-01-05T00:00:00"));
    assert_eq!(predicate, expected_predicate);
}

#[test]
fn test_predicate_conversion_with_date_cast() {
    let sql = "ts >= date '2023-01-05T11:00:00'";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(predicate, None);
}

#[test]
fn test_scalar_value_to_datum_timestamp() {
    use datafusion::common::ScalarValue;

    let ts_micros = 1672876800000000i64;
    let datum =
        super::scalar_value_to_datum(&ScalarValue::TimestampMicrosecond(Some(ts_micros), None));
    assert_eq!(datum, Some(Datum::timestamp_micros(ts_micros)));

    let ts_nanos = 1672876800000000500i64;
    let datum =
        super::scalar_value_to_datum(&ScalarValue::TimestampNanosecond(Some(ts_nanos), None));
    assert_eq!(datum, Some(Datum::timestamp_nanos(ts_nanos)));

    let datum = super::scalar_value_to_datum(&ScalarValue::TimestampMicrosecond(None, None));
    assert_eq!(datum, None);

    let ts_seconds = 1672876800i64;
    let datum = super::scalar_value_to_datum(&ScalarValue::TimestampSecond(Some(ts_seconds), None));
    assert_eq!(datum, None);

    let ts_millis = 1672876800000i64;
    let datum =
        super::scalar_value_to_datum(&ScalarValue::TimestampMillisecond(Some(ts_millis), None));
    assert_eq!(datum, None);
}

#[test]
fn test_scalar_value_to_datum_date64_day_aligned_in_range() {
    use datafusion::common::ScalarValue;

    let days = 19362i32;
    let millis = i64::from(days) * super::MILLIS_PER_DAY;

    let datum = super::scalar_value_to_datum(&ScalarValue::Date64(Some(millis)));
    assert_eq!(datum, Some(Datum::date(days)));

    assert_eq!(
        super::scalar_value_to_datum(&ScalarValue::Date64(Some(0))),
        Some(Datum::date(0))
    );
    assert_eq!(
        super::scalar_value_to_datum(&ScalarValue::Date64(Some(-super::MILLIS_PER_DAY))),
        Some(Datum::date(-1))
    );

    assert_eq!(
        super::scalar_value_to_datum(&ScalarValue::Date32(Some(days))),
        Some(Datum::date(days))
    );
    assert_eq!(
        super::scalar_value_to_datum(&ScalarValue::Date64(None)),
        None
    );
}

#[test]
fn test_scalar_value_to_datum_date64_out_of_date_range_is_not_pushed_down() {
    use datafusion::common::ScalarValue;

    let millis = (i64::from(i32::MAX) + 1) * super::MILLIS_PER_DAY;
    assert_eq!(
        ((millis / super::MILLIS_PER_DAY) as i32),
        i32::MIN,
        "fixture precondition: this value is exactly the one the old cast wrapped"
    );
    assert_eq!(
        super::scalar_value_to_datum(&ScalarValue::Date64(Some(millis))),
        None
    );

    let millis = (i64::from(i32::MIN) - 1) * super::MILLIS_PER_DAY;
    assert_eq!(
        super::scalar_value_to_datum(&ScalarValue::Date64(Some(millis))),
        None
    );
}

#[test]
fn test_scalar_value_to_datum_date64_not_day_aligned_is_not_pushed_down() {
    use datafusion::common::ScalarValue;

    for millis in [
        super::MILLIS_PER_DAY + 1,
        super::MILLIS_PER_DAY - 1,
        1,
        -1,
        -super::MILLIS_PER_DAY - 1,
        i64::MAX,
        i64::MIN,
    ] {
        assert_eq!(
            super::scalar_value_to_datum(&ScalarValue::Date64(Some(millis))),
            None,
            "millis {millis} is not a whole number of days and must not be pushed down"
        );
    }
}

#[test]
fn test_predicate_conversion_date64_out_of_range_is_dropped_not_wrapped() {
    use datafusion::common::ScalarValue;
    use datafusion::prelude::col;

    let millis = (i64::from(i32::MAX) + 1) * super::MILLIS_PER_DAY;
    let filter = col("d").lt(Expr::Literal(ScalarValue::Date64(Some(millis)), None));

    let predicate = convert_filters_to_predicate(&[filter], &test_iceberg_schema());
    assert_eq!(
        predicate, None,
        "an out-of-range Date64 must not reach the scan as a predicate"
    );

    let millis = 19362i64 * super::MILLIS_PER_DAY;
    let filter = col("d").lt(Expr::Literal(ScalarValue::Date64(Some(millis)), None));
    assert_eq!(
        convert_filters_to_predicate(&[filter], &test_iceberg_schema()),
        Some(Reference::new("d").less_than(Datum::date(19362)))
    );
}

#[test]
fn test_scalar_value_to_datum_binary() {
    use datafusion::common::ScalarValue;

    let bytes = vec![1u8, 2u8, 3u8];
    let datum = super::scalar_value_to_datum(&ScalarValue::Binary(Some(bytes.clone())));
    assert_eq!(datum, Some(Datum::binary(bytes.clone())));

    let datum = super::scalar_value_to_datum(&ScalarValue::LargeBinary(Some(bytes.clone())));
    assert_eq!(datum, Some(Datum::binary(bytes)));

    let datum = super::scalar_value_to_datum(&ScalarValue::Binary(None));
    assert_eq!(datum, None);
}

#[test]
fn test_predicate_conversion_with_binary() {
    let sql = "foo = 1 and bin = X'0102'";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    let expected_predicate = Reference::new("foo")
        .equal_to(Datum::long(1))
        .and(Reference::new("bin").equal_to(Datum::binary(vec![1u8, 2u8])));
    assert_eq!(predicate, expected_predicate);
}

#[test]
fn test_scalar_value_to_datum_boolean() {
    use datafusion::common::ScalarValue;

    let datum = super::scalar_value_to_datum(&ScalarValue::Boolean(Some(true)));
    assert_eq!(datum, Some(Datum::bool(true)));

    let datum = super::scalar_value_to_datum(&ScalarValue::Boolean(Some(false)));
    assert_eq!(datum, Some(Datum::bool(false)));

    let datum = super::scalar_value_to_datum(&ScalarValue::Boolean(None));
    assert_eq!(datum, None);
}

#[test]
fn test_predicate_conversion_with_like_starts_with() {
    let sql = "bar LIKE 'test%'";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    assert_eq!(
        predicate,
        Reference::new("bar").starts_with(Datum::string("test"))
    );
}

#[test]
fn test_predicate_conversion_with_not_like_starts_with() {
    let sql = "bar NOT LIKE 'test%'";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    assert_eq!(
        predicate,
        Reference::new("bar").not_starts_with(Datum::string("test"))
    );
}

#[test]
fn test_predicate_conversion_with_like_empty_prefix() {
    let sql = "bar LIKE '%'";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    assert_eq!(
        predicate,
        Reference::new("bar").starts_with(Datum::string(""))
    );
}

#[test]
fn test_predicate_conversion_with_like_complex_pattern() {
    let sql = "bar LIKE 'te%st'";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(predicate, None);
}

#[test]
fn test_predicate_conversion_with_like_underscore_wildcard() {
    let sql = "bar LIKE 'test_'";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(predicate, None);
}

#[test]
fn test_predicate_conversion_with_like_no_wildcard() {
    let sql = "bar LIKE 'test'";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(predicate, None);
}

#[test]
fn test_predicate_conversion_with_ilike() {
    let sql = "bar ILIKE 'test%'";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(predicate, None);
}

#[test]
fn test_predicate_conversion_with_like_and_other_conditions() {
    let sql = "bar LIKE 'test%' AND foo > 1";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    let expected_predicate = Predicate::and(
        Reference::new("bar").starts_with(Datum::string("test")),
        Reference::new("foo").greater_than(Datum::long(1)),
    );
    assert_eq!(predicate, expected_predicate);
}

#[test]
fn test_predicate_conversion_with_like_special_characters() {
    let sql = "bar LIKE 'test-abc_123%'";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(predicate, None);
}

#[test]
fn test_predicate_conversion_with_like_unicode() {
    let sql = "bar LIKE '测试%'";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    assert_eq!(
        predicate,
        Reference::new("bar").starts_with(Datum::string("测试"))
    );
}

#[test]
fn test_predicate_conversion_with_isnan() {
    let predicate = convert_to_iceberg_predicate("isnan(qux)").unwrap();
    assert_eq!(predicate, Reference::new("qux").is_nan());
}

#[test]
fn test_predicate_conversion_with_not_isnan() {
    let predicate = convert_to_iceberg_predicate("NOT isnan(qux)");
    assert_eq!(predicate, None);
}

#[test]
fn test_predicate_conversion_with_isnan_and_other_condition() {
    let sql = "isnan(qux) AND foo > 1";
    let predicate = convert_to_iceberg_predicate(sql).unwrap();
    let expected_predicate = Predicate::and(
        Reference::new("qux").is_nan(),
        Reference::new("foo").greater_than(Datum::long(1)),
    );
    assert_eq!(predicate, expected_predicate);
}

#[test]
fn test_predicate_conversion_with_isnan_unsupported_arg() {
    let sql = "isnan(qux + 1)";
    let predicate = convert_to_iceberg_predicate(sql);
    assert_eq!(predicate, None);
}

fn push(expr: Expr) -> Option<Predicate> {
    convert_filters_to_predicate(&[expr], &test_iceberg_schema())
}

fn cast_col(name: &str, data_type: DataType) -> Expr {
    Expr::Cast(Cast::new(Box::new(col(name)), data_type))
}

#[test]
fn cast_wrapped_float_column_with_inexact_literal_is_not_pushed() {
    for literal in [1e-50_f64, 0.1, 3.4e38, -1e-50] {
        let expr = cast_col("flt", DataType::Float64).lt(lit(literal));
        assert_eq!(
            push(expr),
            None,
            "literal {literal} does not round-trip through f32 and must not be pushed"
        );
    }
}

#[test]
fn cast_wrapped_float_column_with_exact_literal_is_pushed() {
    let expr = cast_col("flt", DataType::Float64).lt(lit(0.5_f64));
    assert_eq!(
        push(expr),
        Some(Reference::new("flt").less_than(Datum::double(0.5)))
    );
    let expr = cast_col("flt", DataType::Float64).gt_eq(lit(-2.0_f64));
    assert_eq!(
        push(expr),
        Some(Reference::new("flt").greater_than_or_equal_to(Datum::double(-2.0)))
    );
}

#[test]
fn cast_wrapped_float_column_out_of_range_literal_is_not_pushed() {
    let expr = cast_col("flt", DataType::Float64).gt(lit(3.5e38_f64));
    assert_eq!(push(expr), None);
    let expr = cast_col("flt", DataType::Float64).lt(lit(-3.5e38_f64));
    assert_eq!(push(expr), None);
}

#[test]
fn cast_wrapped_int_column_with_double_literal_is_not_pushed() {
    for literal in [2.5_f64, 2.0] {
        let expr = cast_col("foo", DataType::Float64).lt(lit(literal));
        assert_eq!(
            push(expr),
            None,
            "literal {literal} cannot reach Datum::to(Int) and must not be pushed"
        );
    }
}

#[test]
fn cast_wrapped_int_column_out_of_range_long_keeps_sentinel_push() {
    let expr = cast_col("foo", DataType::Int64).lt(lit(3_000_000_000_i64));
    assert_eq!(
        push(expr),
        Some(Reference::new("foo").less_than(Datum::long(3_000_000_000_i64)))
    );
    let expr = cast_col("foo", DataType::Int64).gt(lit(-3_000_000_000_i64));
    assert_eq!(
        push(expr),
        Some(Reference::new("foo").greater_than(Datum::long(-3_000_000_000_i64)))
    );
}

#[test]
fn cast_wrapped_double_column_to_int64_is_not_pushed() {
    let expr = cast_col("qux", DataType::Int64).lt(lit(9_007_199_254_740_993_i64));
    assert_eq!(push(expr), None);
    let expr = cast_col("qux", DataType::Int64).lt(lit(9_007_199_254_740_992_i64));
    assert_eq!(push(expr), None);
}

#[test]
fn cast_wrapped_float_column_in_list_with_inexact_element_is_not_pushed() {
    let expr =
        cast_col("flt", DataType::Float64).in_list(vec![lit(0.5_f64), lit(1e-50_f64)], false);
    assert_eq!(push(expr), None);
    let expr = cast_col("flt", DataType::Float64).in_list(vec![lit(0.5_f64), lit(-2.0_f64)], false);
    assert_eq!(
        push(expr),
        Some(Reference::new("flt").is_in([Datum::double(0.5), Datum::double(-2.0)]))
    );
}

#[test]
fn bare_float_column_with_inexact_literal_is_not_pushed() {
    let expr = col("flt").lt(lit(1e-50_f64));
    assert_eq!(push(expr), None);
}

#[test]
fn lossy_column_casts_are_not_stripped() {
    for (column, data_type) in [
        ("flt", DataType::Int32),
        ("flt", DataType::Utf8),
        ("qux", DataType::Float32),
        ("foo", DataType::Int16),
        ("flt", DataType::Int64),
        ("qux", DataType::Int64),
    ] {
        let expr = cast_col(column, data_type.clone()).eq(lit(2_i64));
        assert_eq!(
            push(expr),
            None,
            "CAST({column} AS {data_type}) is lossy and must not strip"
        );
    }
    let expr = Expr::Cast(Cast::new(
        Box::new(cast_col("foo", DataType::Int64)),
        DataType::Float64,
    ))
    .eq(lit(9_007_199_254_740_993_i64));
    assert_eq!(
        push(expr),
        None,
        "int64 -> float64 cast is not injective and must not strip"
    );
}

#[test]
fn lossless_column_casts_still_strip() {
    let expr = cast_col("foo", DataType::Int64).lt(lit(5_i64));
    assert_eq!(
        push(expr),
        Some(Reference::new("foo").less_than(Datum::long(5)))
    );
    let expr = cast_col("flt", DataType::Float64).lt(lit(0.5_f64));
    assert_eq!(
        push(expr),
        Some(Reference::new("flt").less_than(Datum::double(0.5)))
    );
    let expr = cast_col("qux", DataType::Float64).eq(lit(1.5_f64));
    assert_eq!(
        push(expr),
        Some(Reference::new("qux").equal_to(Datum::double(1.5)))
    );
}

#[test]
fn float_column_out_of_range_and_infinite_literals_are_not_pushed() {
    for expr in [
        cast_col("flt", DataType::Float64).gt(lit(3.5e38_f64)),
        cast_col("flt", DataType::Float64).lt(lit(-3.5e38_f64)),
        col("flt").eq(lit(f64::INFINITY)),
        col("flt").eq(lit(f64::NEG_INFINITY)),
        col("flt").lt(lit(f64::INFINITY)),
        col("flt").gt(lit(f64::NEG_INFINITY)),
    ] {
        assert_eq!(push(expr), None);
    }
}

#[test]
fn float_comparisons_with_zero_literals_are_not_pushed() {
    for expr in [
        col("flt").eq(lit(0.0_f64)),
        col("flt").lt_eq(lit(-0.0_f64)),
        col("flt").gt_eq(lit(0.0_f64)),
        col("flt").lt(lit(0.0_f64)),
        col("flt").not_eq(lit(0.0_f64)),
        col("qux").eq(lit(-0.0_f64)),
        col("flt").eq(lit(0_i64)),
        col("flt").eq(lit(0.0_f32)),
    ] {
        assert_eq!(push(expr), None);
    }
}

#[test]
fn not_over_float_comparison_is_not_pushed() {
    assert_eq!(push(Expr::Not(Box::new(col("flt").lt(lit(5.0_f64))))), None);
    assert_eq!(push(Expr::Not(Box::new(col("flt").eq(lit(5.0_f64))))), None);
    assert_eq!(
        push(Expr::Not(Box::new(col("foo").eq(lit(5_i64))))),
        Some(!Reference::new("foo").equal_to(Datum::long(5)))
    );
}

#[test]
fn timestamp_nanos_literal_against_micros_column_is_not_pushed() {
    let expr = col("ts").gt_eq(lit(ScalarValue::TimestampNanosecond(
        Some(1_672_876_800_000_000_500),
        None,
    )));
    assert_eq!(push(expr), None);
}

fn nested_iceberg_schema() -> SchemaRef {
    IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(
                2,
                "xs",
                Type::List(ListType::new(
                    NestedField::list_element(3, Type::Primitive(PrimitiveType::Int), false)
                        .into(),
                )),
            )
            .into(),
            NestedField::optional(
                4,
                "m",
                Type::Map(MapType::new(
                    NestedField::map_key_element(5, Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::map_value_element(
                        6,
                        Type::Primitive(PrimitiveType::Int),
                        false,
                    )
                    .into(),
                )),
            )
            .into(),
            NestedField::optional(
                7,
                "s",
                Type::Struct(StructType::new(vec![NestedField::optional(
                    8,
                    "a",
                    Type::Primitive(PrimitiveType::Int),
                )
                .into()])),
            )
            .into(),
        ])
        .build()
        .unwrap()
        .into()
}

fn push_nested(exprs: &[Expr]) -> Option<Predicate> {
    convert_filters_to_predicate(exprs, &nested_iceberg_schema())
}

#[test]
fn is_null_on_a_list_column_is_not_pushed() {
    assert_eq!(push_nested(&[Expr::IsNull(Box::new(col("xs")))]), None);
    assert_eq!(
        push_nested(&[Expr::IsNotNull(Box::new(col("xs")))]),
        None
    );
}

#[test]
fn is_null_on_a_map_column_is_not_pushed() {
    assert_eq!(push_nested(&[Expr::IsNull(Box::new(col("m")))]), None);
    assert_eq!(
        push_nested(&[Expr::IsNotNull(Box::new(col("m")))]),
        None
    );
}

#[test]
fn is_null_on_a_struct_column_is_not_pushed() {
    assert_eq!(push_nested(&[Expr::IsNull(Box::new(col("s")))]), None);
    assert_eq!(
        push_nested(&[Expr::IsNotNull(Box::new(col("s")))]),
        None
    );
}

#[test]
fn is_null_on_a_list_element_name_is_not_pushed() {
    let column = Column::new_unqualified("xs.element");
    assert_eq!(push_nested(&[Expr::IsNull(Box::new(Expr::Column(column)))]), None);
}

#[test]
fn is_null_on_a_nested_column_drops_only_its_own_conjunction() {
    assert_eq!(
        push_nested(&[
            col("id").gt(lit(1_i64)),
            Expr::IsNull(Box::new(col("xs")))
        ]),
        Some(Reference::new("id").greater_than(Datum::long(1)))
    );
    assert_eq!(
        push_nested(&[col("id")
            .gt(lit(1_i64))
            .and(Expr::IsNull(Box::new(col("xs"))))]),
        None
    );
    assert_eq!(
        push_nested(&[Expr::IsNull(Box::new(col("xs"))).or(col("id").eq(lit(1_i64)))]),
        None
    );
    assert_eq!(
        push_nested(&[Expr::Not(Box::new(Expr::IsNull(Box::new(col("xs")))))]),
        None
    );
}

#[test]
fn is_null_on_a_primitive_column_still_pushes() {
    assert_eq!(
        push_nested(&[Expr::IsNull(Box::new(col("id")))]),
        Some(Reference::new("id").is_null())
    );
    assert_eq!(
        push_nested(&[Expr::IsNotNull(Box::new(col("id")))]),
        Some(Reference::new("id").is_not_null())
    );
}

#[test]
fn is_null_on_a_struct_leaf_still_pushes() {
    let column = Column::new_unqualified("s.a");
    assert_eq!(
        push_nested(&[Expr::IsNull(Box::new(Expr::Column(column)))]),
        Some(Reference::new("s.a").is_null())
    );
}
