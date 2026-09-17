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

use datafusion::logical_expr::expr::BinaryExpr;
use datafusion::logical_expr::{Expr, Operator};
use datafusion::prelude::col;
use datafusion::scalar::ScalarValue;
use iceberg::expr::{Predicate, Reference};
use iceberg::spec::Datum;

use super::convert_filters_to_predicate;

fn nan_f64() -> Expr {
    Expr::Literal(ScalarValue::Float64(Some(f64::NAN)), None)
}

fn nan_f32() -> Expr {
    Expr::Literal(ScalarValue::Float32(Some(f32::NAN)), None)
}

fn double_lit(v: f64) -> Expr {
    Expr::Literal(ScalarValue::Float64(Some(v)), None)
}

fn long_lit(v: i64) -> Expr {
    Expr::Literal(ScalarValue::Int64(Some(v)), None)
}

fn null_safe_eq(left: Expr, right: Expr) -> Expr {
    Expr::BinaryExpr(BinaryExpr::new(
        Box::new(left),
        Operator::IsNotDistinctFrom,
        Box::new(right),
    ))
}

fn push(expr: Expr) -> Option<Predicate> {
    convert_filters_to_predicate(&[expr])
}

#[test]
fn nan_eq_either_side_converts_to_is_nan() {
    assert_eq!(
        push(col("qux").eq(nan_f64())),
        Some(Reference::new("qux").is_nan())
    );
    assert_eq!(
        push(nan_f64().eq(col("qux"))),
        Some(Reference::new("qux").is_nan())
    );
}

#[test]
fn nan_null_safe_eq_either_side_converts_to_is_nan() {
    assert_eq!(
        push(null_safe_eq(col("qux"), nan_f64())),
        Some(Reference::new("qux").is_nan())
    );
    assert_eq!(
        push(null_safe_eq(nan_f64(), col("qux"))),
        Some(Reference::new("qux").is_nan())
    );
}

#[test]
fn nan_not_eq_either_side_converts_to_not_nan() {
    assert_eq!(
        push(col("qux").not_eq(nan_f64())),
        Some(Reference::new("qux").is_not_nan())
    );
    assert_eq!(
        push(nan_f64().not_eq(col("qux"))),
        Some(Reference::new("qux").is_not_nan())
    );
}

#[test]
fn nan_range_comparisons_are_not_pushed() {
    assert_eq!(push(col("qux").lt(nan_f64())), None);
    assert_eq!(push(col("qux").lt_eq(nan_f64())), None);
    assert_eq!(push(col("qux").gt(nan_f64())), None);
    assert_eq!(push(col("qux").gt_eq(nan_f64())), None);
    assert_eq!(push(nan_f64().lt(col("qux"))), None);
    assert_eq!(push(nan_f64().gt_eq(col("qux"))), None);
}

#[test]
fn nan_only_in_list_converts_to_is_nan() {
    assert_eq!(
        push(col("qux").in_list(vec![nan_f64()], false)),
        Some(Reference::new("qux").is_nan())
    );
}

#[test]
fn nan_mixed_in_list_converts_to_is_nan_or_in_rest() {
    assert_eq!(
        push(col("qux").in_list(vec![nan_f64(), double_lit(1.0)], false)),
        Some(Predicate::or(
            Reference::new("qux").is_nan(),
            Reference::new("qux").is_in([Datum::double(1.0)])
        ))
    );
}

#[test]
fn nan_not_in_list_is_not_pushed() {
    assert_eq!(
        push(col("qux").in_list(vec![nan_f64(), double_lit(1.0)], true)),
        None
    );
    assert_eq!(push(col("qux").in_list(vec![nan_f64()], true)), None);
}

#[test]
fn float32_nan_takes_the_same_arms_as_float64_nan() {
    assert_eq!(
        push(col("flt").eq(nan_f32())),
        Some(Reference::new("flt").is_nan())
    );
    assert_eq!(
        push(null_safe_eq(col("flt"), nan_f32())),
        Some(Reference::new("flt").is_nan())
    );
    assert_eq!(
        push(col("flt").not_eq(nan_f32())),
        Some(Reference::new("flt").is_not_nan())
    );
    assert_eq!(push(col("flt").lt(nan_f32())), None);
    assert_eq!(
        push(col("flt").in_list(vec![nan_f32(), double_lit(1.5)], false)),
        Some(Predicate::or(
            Reference::new("flt").is_nan(),
            Reference::new("flt").is_in([Datum::double(1.5)])
        ))
    );
}

#[test]
fn nan_eq_composes_under_and_or_not() {
    assert_eq!(
        push(col("qux").eq(nan_f64()).and(col("foo").eq(long_lit(1)))),
        Some(Predicate::and(
            Reference::new("qux").is_nan(),
            Reference::new("foo").equal_to(Datum::long(1))
        ))
    );
    assert_eq!(
        push(col("qux").eq(nan_f64()).or(col("foo").eq(long_lit(1)))),
        Some(Predicate::or(
            Reference::new("qux").is_nan(),
            Reference::new("foo").equal_to(Datum::long(1))
        ))
    );
    assert_eq!(
        push(Expr::Not(Box::new(col("qux").eq(nan_f64())))),
        Some(!Reference::new("qux").is_nan())
    );
}

#[test]
fn non_nan_float_comparisons_still_push() {
    assert_eq!(
        push(col("qux").eq(double_lit(1.0))),
        Some(Reference::new("qux").equal_to(Datum::double(1.0)))
    );
    assert_eq!(
        push(col("qux").lt(double_lit(1.0))),
        Some(Reference::new("qux").less_than(Datum::double(1.0)))
    );
    assert_eq!(
        push(col("qux").in_list(vec![double_lit(1.0), double_lit(2.0)], false)),
        Some(Reference::new("qux").is_in([Datum::double(1.0), Datum::double(2.0)]))
    );
    assert_eq!(
        push(col("qux").in_list(vec![double_lit(1.0)], true)),
        Some(Reference::new("qux").is_not_in([Datum::double(1.0)]))
    );
}

#[test]
fn non_nan_null_safe_eq_is_not_pushed() {
    assert_eq!(push(null_safe_eq(col("qux"), double_lit(1.0))), None);
}

#[test]
fn column_to_column_eq_is_not_pushed() {
    assert_eq!(push(col("qux").eq(col("qux"))), None);
}
