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
use std::vec;

use datafusion::arrow::datatypes::{DataType, TimeUnit};
use datafusion::common::Column;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{Expr, Like, Operator};
use datafusion::scalar::ScalarValue;
use iceberg::expr::{BinaryExpression, Predicate, PredicateOperator, Reference, UnaryExpression};
use iceberg::spec::{Datum, PrimitiveLiteral, PrimitiveType, Schema, Type};
use iceberg::table::Table;

use crate::to_datafusion_error;

// A datafusion expression could be an Iceberg predicate, column, or literal.
enum TransformedResult {
    Predicate(Predicate),
    Column(Reference),
    Literal(Datum),
    NotTransformed,
}

enum OpTransformedResult {
    Operator(PredicateOperator),
    And,
    Or,
    NotTransformed,
}

/// Converts DataFusion filters ([`Expr`]) to an iceberg [`Predicate`].
/// If none of the filters could be converted, return `None` which adds no predicates to the scan operation.
/// If the conversion was successful, return the converted predicates combined with an AND operator.
pub fn convert_filters_to_predicate(filters: &[Expr], schema: &Schema) -> Option<Predicate> {
    filters
        .iter()
        .filter_map(|expr| {
            convert_filter_to_predicate(expr, schema)
                .filter(|predicate| predicate_binds_soundly(predicate, schema))
        })
        .reduce(Predicate::and)
}

fn predicate_binds_soundly(predicate: &Predicate, schema: &Schema) -> bool {
    match predicate {
        Predicate::AlwaysTrue | Predicate::AlwaysFalse => true,
        Predicate::And(expr) | Predicate::Or(expr) => {
            let [left, right] = expr.inputs();
            predicate_binds_soundly(left, schema) && predicate_binds_soundly(right, schema)
        }
        Predicate::Not(expr) => predicate_binds_soundly(expr.inputs()[0], schema),
        Predicate::Unary(expr) => term_binds_soundly(schema, expr.term()),
        Predicate::Binary(expr) => {
            term_binds_soundly(schema, expr.term())
                && literal_binds_soundly(schema, expr.term(), expr.literal())
        }
        Predicate::Set(expr) => {
            term_binds_soundly(schema, expr.term())
                && expr
                    .literals()
                    .iter()
                    .all(|literal| literal_binds_soundly(schema, expr.term(), literal))
        }
    }
}

fn term_binds_soundly(schema: &Schema, column: &Reference) -> bool {
    schema
        .field_by_name(column.name())
        .is_some_and(|field| schema.accessor_by_field_id(field.id).is_some())
}

fn literal_binds_soundly(schema: &Schema, column: &Reference, literal: &Datum) -> bool {
    let Some(field) = schema.field_by_name(column.name()) else {
        return false;
    };
    let Ok(converted) = literal.clone().to(&field.field_type) else {
        return false;
    };
    let float_column = matches!(
        field.field_type.as_ref(),
        Type::Primitive(PrimitiveType::Float | PrimitiveType::Double)
    );
    let untyped_literal = matches!(
        literal.data_type(),
        PrimitiveType::Boolean
            | PrimitiveType::Int
            | PrimitiveType::Long
            | PrimitiveType::Float
            | PrimitiveType::Double
            | PrimitiveType::String
            | PrimitiveType::Binary
    );
    if !untyped_literal && converted != *literal {
        return false;
    }
    match converted.literal() {
        PrimitiveLiteral::AboveMax | PrimitiveLiteral::BelowMin => !float_column,
        PrimitiveLiteral::Float(value) if float_column && value.0 == 0.0 => false,
        PrimitiveLiteral::Double(value) if float_column && value.0 == 0.0 => false,
        _ if converted == *literal => true,
        converted_literal => converts_exactly(literal.literal(), converted_literal),
    }
}

fn converts_exactly(original: &PrimitiveLiteral, converted: &PrimitiveLiteral) -> bool {
    match (original, converted) {
        (PrimitiveLiteral::Int(v), PrimitiveLiteral::Long(w)) => *w == i64::from(*v),
        (PrimitiveLiteral::Int(v), PrimitiveLiteral::Float(w)) => f64::from(w.0) == f64::from(*v),
        (PrimitiveLiteral::Int(v), PrimitiveLiteral::Double(w)) => w.0 == f64::from(*v),
        (PrimitiveLiteral::Long(v), PrimitiveLiteral::Int(w)) => i64::from(*w) == *v,
        (PrimitiveLiteral::Long(v), PrimitiveLiteral::Float(w)) => w.0 as i128 == i128::from(*v),
        (PrimitiveLiteral::Long(v), PrimitiveLiteral::Double(w)) => w.0 as i128 == i128::from(*v),
        (PrimitiveLiteral::Float(v), PrimitiveLiteral::Double(w)) => w.0 == f64::from(v.0),
        (PrimitiveLiteral::Double(v), PrimitiveLiteral::Float(w)) => f64::from(w.0) == v.0,
        (PrimitiveLiteral::String(s), PrimitiveLiteral::Long(_)) => sub_second_digits(s) <= 6,
        _ => false,
    }
}

fn sub_second_digits(s: &str) -> usize {
    match s.find('.') {
        Some(index) => s[index + 1..]
            .chars()
            .take_while(char::is_ascii_digit)
            .count(),
        None => 0,
    }
}

fn convert_filter_to_predicate(expr: &Expr, schema: &Schema) -> Option<Predicate> {
    match to_iceberg_predicate(expr, schema) {
        TransformedResult::Predicate(predicate) => Some(predicate),
        TransformedResult::Column(column) => {
            // A bare column in a filter context represents a boolean column check
            // Convert it to: column = true
            Some(Predicate::Binary(BinaryExpression::new(
                PredicateOperator::Eq,
                column,
                Datum::bool(true),
            )))
        }
        TransformedResult::Literal(_) => {
            // Literal values in filter context cannot be pushed down
            None
        }
        _ => None,
    }
}

fn to_iceberg_predicate(expr: &Expr, schema: &Schema) -> TransformedResult {
    match expr {
        Expr::BinaryExpr(binary) => {
            let left = to_iceberg_predicate(&binary.left, schema);
            let right = to_iceberg_predicate(&binary.right, schema);
            if let Some(nan) = nan_comparison(binary.op, &left, &right) {
                return nan;
            }
            let op = to_iceberg_operation(binary.op);
            match op {
                OpTransformedResult::Operator(op) => to_iceberg_binary_predicate(left, right, op),
                OpTransformedResult::And => to_iceberg_and_predicate(left, right),
                OpTransformedResult::Or => to_iceberg_or_predicate(left, right),
                OpTransformedResult::NotTransformed => TransformedResult::NotTransformed,
            }
        }
        Expr::Not(exp) => {
            let expr = to_iceberg_predicate(exp, schema);
            match expr {
                TransformedResult::Predicate(p) => {
                    if not_operand_is_sound(&p, schema) {
                        TransformedResult::Predicate(!p)
                    } else {
                        TransformedResult::NotTransformed
                    }
                }
                TransformedResult::Column(column) => {
                    // NOT of a bare boolean column: NOT col => col = false
                    TransformedResult::Predicate(Predicate::Binary(BinaryExpression::new(
                        PredicateOperator::Eq,
                        column,
                        Datum::bool(false),
                    )))
                }
                _ => TransformedResult::NotTransformed,
            }
        }
        Expr::Column(column) => TransformedResult::Column(Reference::new(column.name())),
        Expr::Literal(literal, _) => match scalar_value_to_datum(literal) {
            Some(data) => TransformedResult::Literal(data),
            None => TransformedResult::NotTransformed,
        },
        Expr::InList(inlist) => {
            let mut datums = vec![];
            for expr in &inlist.list {
                let p = to_iceberg_predicate(expr, schema);
                match p {
                    TransformedResult::Literal(l) => datums.push(l),
                    _ => return TransformedResult::NotTransformed,
                }
            }

            let expr = to_iceberg_predicate(&inlist.expr, schema);
            match expr {
                TransformedResult::Column(r) => in_list_predicate(r, datums, inlist.negated),
                _ => TransformedResult::NotTransformed,
            }
        }
        Expr::IsNull(expr) => {
            let p = to_iceberg_predicate(expr, schema);
            match p {
                TransformedResult::Column(r) => TransformedResult::Predicate(Predicate::Unary(
                    UnaryExpression::new(PredicateOperator::IsNull, r),
                )),
                _ => TransformedResult::NotTransformed,
            }
        }
        Expr::IsNotNull(expr) => {
            let p = to_iceberg_predicate(expr, schema);
            match p {
                TransformedResult::Column(r) => TransformedResult::Predicate(Predicate::Unary(
                    UnaryExpression::new(PredicateOperator::NotNull, r),
                )),
                _ => TransformedResult::NotTransformed,
            }
        }
        Expr::Cast(c) => {
            if let Expr::Literal(value, _) = c.expr.as_ref() {
                if matches!(c.field.data_type(), DataType::Timestamp(..))
                    && matches!(
                        value,
                        ScalarValue::Utf8(_) | ScalarValue::LargeUtf8(_) | ScalarValue::Utf8View(_)
                    )
                {
                    return to_iceberg_predicate(&c.expr, schema);
                }
                return match value.cast_to(c.field.data_type()) {
                    Ok(value) => match scalar_value_to_datum(&value) {
                        Some(datum) => TransformedResult::Literal(datum),
                        None => TransformedResult::NotTransformed,
                    },
                    Err(_) => TransformedResult::NotTransformed,
                };
            }
            let Some(source) = cast_source_type(&c.expr, schema) else {
                return TransformedResult::NotTransformed;
            };
            if !cast_strips_lossless(&source, c.field.data_type()) {
                return TransformedResult::NotTransformed;
            }
            to_iceberg_predicate(&c.expr, schema)
        }
        Expr::Like(Like {
            negated,
            expr,
            pattern,
            escape_char,
            case_insensitive,
        }) => {
            // Only support simple prefix patterns (e.g., 'prefix%')
            // Note: Iceberg's StartsWith operator is case-sensitive, so we cannot
            // push down case-insensitive LIKE (ILIKE) patterns
            // Escape characters are also not supported for pushdown
            if escape_char.is_some() || *case_insensitive {
                return TransformedResult::NotTransformed;
            }

            // Extract the pattern string
            let pattern_str = match to_iceberg_predicate(pattern, schema) {
                TransformedResult::Literal(d) => match d.literal() {
                    PrimitiveLiteral::String(s) => s.clone(),
                    _ => return TransformedResult::NotTransformed,
                },
                _ => return TransformedResult::NotTransformed,
            };

            // Check if it's a simple prefix pattern (ends with % and no other wildcards)
            if pattern_str.ends_with('%')
                && !pattern_str[..pattern_str.len() - 1].contains(['%', '_'])
            {
                // Extract the prefix (remove trailing %)
                let prefix = pattern_str[..pattern_str.len() - 1].to_string();

                // Get the column reference
                let column = match to_iceberg_predicate(expr, schema) {
                    TransformedResult::Column(r) => r,
                    _ => return TransformedResult::NotTransformed,
                };

                // Create the appropriate predicate
                let predicate = if *negated {
                    column.not_starts_with(Datum::string(prefix))
                } else {
                    column.starts_with(Datum::string(prefix))
                };

                TransformedResult::Predicate(predicate)
            } else {
                // Complex LIKE patterns cannot be pushed down
                TransformedResult::NotTransformed
            }
        }
        Expr::ScalarFunction(ScalarFunction { func, args }) => {
            scalar_function_to_iceberg_predicate(func.name(), args, schema)
        }
        _ => TransformedResult::NotTransformed,
    }
}

fn cast_source_type(expr: &Expr, schema: &Schema) -> Option<DataType> {
    match expr {
        Expr::Column(column) => schema
            .field_by_name(column.name())
            .and_then(|field| iceberg_arrow_type(&field.field_type)),
        Expr::Cast(cast) => Some(cast.field.data_type().clone()),
        _ => None,
    }
}

fn iceberg_arrow_type(field_type: &Type) -> Option<DataType> {
    let Type::Primitive(primitive) = field_type else {
        return None;
    };
    Some(match primitive {
        PrimitiveType::Boolean => DataType::Boolean,
        PrimitiveType::Int => DataType::Int32,
        PrimitiveType::Long => DataType::Int64,
        PrimitiveType::Float => DataType::Float32,
        PrimitiveType::Double => DataType::Float64,
        PrimitiveType::Date => DataType::Date32,
        PrimitiveType::Time => DataType::Time64(TimeUnit::Microsecond),
        PrimitiveType::Timestamp => DataType::Timestamp(TimeUnit::Microsecond, None),
        PrimitiveType::Timestamptz => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        }
        PrimitiveType::TimestampNs => DataType::Timestamp(TimeUnit::Nanosecond, None),
        PrimitiveType::TimestamptzNs => {
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
        }
        PrimitiveType::String => DataType::Utf8,
        PrimitiveType::Uuid => DataType::FixedSizeBinary(16),
        PrimitiveType::Fixed(length) => DataType::FixedSizeBinary(*length as i32),
        PrimitiveType::Binary => DataType::Binary,
        PrimitiveType::Decimal { precision, scale } => {
            DataType::Decimal128(*precision as u8, *scale as i8)
        }
        _ => return None,
    })
}

fn time_unit_widens(from: &TimeUnit, to: &TimeUnit) -> bool {
    matches!(
        (from, to),
        (
            TimeUnit::Second,
            TimeUnit::Millisecond | TimeUnit::Microsecond | TimeUnit::Nanosecond
        ) | (
            TimeUnit::Millisecond,
            TimeUnit::Microsecond | TimeUnit::Nanosecond
        ) | (TimeUnit::Microsecond, TimeUnit::Nanosecond)
    )
}

fn cast_strips_lossless(from: &DataType, to: &DataType) -> bool {
    if from == to {
        return true;
    }
    match (from, to) {
        (DataType::Int8, DataType::Int16 | DataType::Int32 | DataType::Int64)
        | (DataType::Int16, DataType::Int32 | DataType::Int64)
        | (DataType::Int32, DataType::Int64)
        | (DataType::UInt8, DataType::Int16 | DataType::Int32 | DataType::Int64)
        | (DataType::UInt16, DataType::Int32 | DataType::Int64)
        | (DataType::UInt32, DataType::Int64)
        | (DataType::UInt8, DataType::UInt16 | DataType::UInt32 | DataType::UInt64)
        | (DataType::UInt16, DataType::UInt32 | DataType::UInt64)
        | (DataType::UInt32, DataType::UInt64)
        | (
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32,
            DataType::Float64,
        )
        | (DataType::Float32, DataType::Float64)
        | (
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
        ) => true,
        (DataType::Timestamp(from_unit, from_tz), DataType::Timestamp(to_unit, to_tz)) => {
            from_unit == to_unit && from_tz.is_some() == to_tz.is_some()
        }
        (DataType::Time32(from_unit), DataType::Time64(to_unit))
        | (DataType::Time64(from_unit), DataType::Time64(to_unit)) => {
            time_unit_widens(from_unit, to_unit)
        }
        _ => false,
    }
}

fn float_field(schema: &Schema, column: &Reference) -> bool {
    schema.field_by_name(column.name()).is_some_and(|field| {
        matches!(
            field.field_type.as_ref(),
            Type::Primitive(PrimitiveType::Float | PrimitiveType::Double)
        )
    })
}

fn not_operand_is_sound(predicate: &Predicate, schema: &Schema) -> bool {
    match predicate {
        Predicate::AlwaysTrue | Predicate::AlwaysFalse => true,
        Predicate::And(expr) | Predicate::Or(expr) => {
            let [left, right] = expr.inputs();
            not_operand_is_sound(left, schema) && not_operand_is_sound(right, schema)
        }
        Predicate::Not(expr) => not_operand_is_sound(expr.inputs()[0], schema),
        Predicate::Unary(expr) => expr.op() != PredicateOperator::IsNan,
        Predicate::Binary(expr) => !float_field(schema, expr.term()),
        Predicate::Set(expr) => !float_field(schema, expr.term()),
    }
}

fn to_iceberg_operation(op: Operator) -> OpTransformedResult {
    match op {
        Operator::Eq => OpTransformedResult::Operator(PredicateOperator::Eq),
        Operator::NotEq => OpTransformedResult::Operator(PredicateOperator::NotEq),
        Operator::Lt => OpTransformedResult::Operator(PredicateOperator::LessThan),
        Operator::LtEq => OpTransformedResult::Operator(PredicateOperator::LessThanOrEq),
        Operator::Gt => OpTransformedResult::Operator(PredicateOperator::GreaterThan),
        Operator::GtEq => OpTransformedResult::Operator(PredicateOperator::GreaterThanOrEq),
        // AND OR
        Operator::And => OpTransformedResult::And,
        Operator::Or => OpTransformedResult::Or,
        // Others not supported
        _ => OpTransformedResult::NotTransformed,
    }
}

/// Translates a DataFusion scalar function into an Iceberg predicate.
/// Unlike dedicated Expr variants (e.g. `Expr::IsNull`), scalar functions are
/// identified by name at runtime, so we need to handle them here.
fn scalar_function_to_iceberg_predicate(
    func_name: &str,
    args: &[Expr],
    schema: &Schema,
) -> TransformedResult {
    match func_name {
        // TODO: support complex expression arguments to scalar functions
        "isnan" if args.len() == 1 => {
            let operand = to_iceberg_predicate(&args[0], schema);
            match operand {
                TransformedResult::Column(r) => TransformedResult::Predicate(Predicate::Unary(
                    UnaryExpression::new(PredicateOperator::IsNan, r),
                )),
                _ => TransformedResult::NotTransformed,
            }
        }
        _ => TransformedResult::NotTransformed,
    }
}

/// Both sides must convert. Dropping one side is a weaker AND at the top level, and
/// `convert_filters_to_predicate` already drops unconverted top-level conjuncts. Under `NOT`
/// a dropped side inverts and the prune keeps the wrong files.
fn to_iceberg_and_predicate(
    left: TransformedResult,
    right: TransformedResult,
) -> TransformedResult {
    match (left, right) {
        (TransformedResult::Predicate(left), TransformedResult::Predicate(right)) => {
            TransformedResult::Predicate(left.and(right))
        }
        _ => TransformedResult::NotTransformed,
    }
}

fn to_iceberg_or_predicate(left: TransformedResult, right: TransformedResult) -> TransformedResult {
    match (left, right) {
        (TransformedResult::Predicate(left), TransformedResult::Predicate(right)) => {
            TransformedResult::Predicate(left.or(right))
        }
        _ => TransformedResult::NotTransformed,
    }
}

fn nan_comparison(
    op: Operator,
    left: &TransformedResult,
    right: &TransformedResult,
) -> Option<TransformedResult> {
    let (column, is_nan) = match (left, right) {
        (TransformedResult::Column(r), TransformedResult::Literal(d)) => (r, d.is_nan()),
        (TransformedResult::Literal(d), TransformedResult::Column(r)) => (r, d.is_nan()),
        _ => return None,
    };
    if !is_nan {
        return None;
    }
    match op {
        Operator::Eq | Operator::IsNotDistinctFrom => {
            Some(TransformedResult::Predicate(column.clone().is_nan()))
        }
        _ => Some(TransformedResult::NotTransformed),
    }
}

fn in_list_predicate(column: Reference, datums: Vec<Datum>, negated: bool) -> TransformedResult {
    let has_nan = datums.iter().any(Datum::is_nan);
    if negated {
        if has_nan {
            TransformedResult::NotTransformed
        } else {
            TransformedResult::Predicate(column.is_not_in(datums))
        }
    } else if !has_nan {
        TransformedResult::Predicate(column.is_in(datums))
    } else {
        let rest: Vec<Datum> = datums.into_iter().filter(|d| !d.is_nan()).collect();
        let is_nan = column.clone().is_nan();
        if rest.is_empty() {
            TransformedResult::Predicate(is_nan)
        } else {
            TransformedResult::Predicate(Predicate::or(is_nan, column.is_in(rest)))
        }
    }
}

fn to_iceberg_binary_predicate(
    left: TransformedResult,
    right: TransformedResult,
    op: PredicateOperator,
) -> TransformedResult {
    let (r, d, op) = match (left, right) {
        (TransformedResult::NotTransformed, _) => return TransformedResult::NotTransformed,
        (_, TransformedResult::NotTransformed) => return TransformedResult::NotTransformed,
        (TransformedResult::Column(r), TransformedResult::Literal(d)) => (r, d, op),
        (TransformedResult::Literal(d), TransformedResult::Column(r)) => {
            (r, d, reverse_predicate_operator(op))
        }
        _ => return TransformedResult::NotTransformed,
    };
    TransformedResult::Predicate(Predicate::Binary(BinaryExpression::new(op, r, d)))
}

fn reverse_predicate_operator(op: PredicateOperator) -> PredicateOperator {
    match op {
        PredicateOperator::Eq => PredicateOperator::Eq,
        PredicateOperator::NotEq => PredicateOperator::NotEq,
        PredicateOperator::GreaterThan => PredicateOperator::LessThan,
        PredicateOperator::GreaterThanOrEq => PredicateOperator::LessThanOrEq,
        PredicateOperator::LessThan => PredicateOperator::GreaterThan,
        PredicateOperator::LessThanOrEq => PredicateOperator::GreaterThanOrEq,
        _ => unreachable!("Reverse {}", op),
    }
}

const MILLIS_PER_DAY: i64 = 24 * 60 * 60 * 1000;

/// Convert an Arrow `Date64` literal (milliseconds since the Unix epoch) to an Iceberg `date`
/// datum (days since the Unix epoch), or `None` when it cannot be represented as one.
///
/// Returning `None` means "not pushed down": the caller drops the comparison and DataFusion
/// evaluates it itself. That is always safe; an approximate day is not, because the predicate
/// this feeds is the ONLY filter the Iceberg scan applies. The provider reports
/// [`TableProviderFilterPushDown::Inexact`](datafusion::logical_expr::TableProviderFilterPushDown),
/// so DataFusion re-checks the rows the scan RETURNS but can never resurrect rows a wrong
/// predicate pruned away.
///
/// Two inputs are rejected:
///
/// * **Not a whole number of days.** Arrow requires `Date64` values to be evenly divisible by
///   86_400_000, so this is out-of-contract input. Rounding it to a day is unsound in a way that
///   depends on the comparison operator — which this function does not know. For
///   `millis = 1 day + 1 ms`, `col < millis` matches days `{0, 1}` (day 1 begins before the
///   literal) but the truncated `col < date(1)` matches only `{0}`, silently dropping every
///   day-1 row; rounding UP breaks `>` symmetrically.
/// * **Out of the `date` range.** The day count must fit `i32`. The previous `as i32` wrapped
///   instead: one day past `i32::MAX` became `i32::MIN`, turning a far-future bound into a
///   far-past one — with `<`, a filter matching every row became one matching none.
fn date64_millis_to_datum(millis: i64) -> Option<Datum> {
    if millis % MILLIS_PER_DAY != 0 {
        return None;
    }
    // Exact for a whole number of days: truncating and flooring division agree on multiples.
    i32::try_from(millis / MILLIS_PER_DAY).ok().map(Datum::date)
}

/// Convert a scalar value to an iceberg datum.
fn scalar_value_to_datum(value: &ScalarValue) -> Option<Datum> {
    match value {
        ScalarValue::Boolean(Some(v)) => Some(Datum::bool(*v)),
        ScalarValue::Int8(Some(v)) => Some(Datum::int(i32::from(*v))),
        ScalarValue::Int16(Some(v)) => Some(Datum::int(i32::from(*v))),
        ScalarValue::Int32(Some(v)) => Some(Datum::int(*v)),
        ScalarValue::Int64(Some(v)) => Some(Datum::long(*v)),
        ScalarValue::Float32(Some(v)) => Some(Datum::double(f64::from(*v))),
        ScalarValue::Float64(Some(v)) => Some(Datum::double(*v)),
        ScalarValue::Utf8(Some(v)) => Some(Datum::string(v.clone())),
        ScalarValue::LargeUtf8(Some(v)) => Some(Datum::string(v.clone())),
        ScalarValue::Binary(Some(v)) => Some(Datum::binary(v.clone())),
        ScalarValue::LargeBinary(Some(v)) => Some(Datum::binary(v.clone())),
        ScalarValue::Date32(Some(v)) => Some(Datum::date(*v)),
        ScalarValue::Date64(Some(v)) => date64_millis_to_datum(*v),
        ScalarValue::TimestampSecond(Some(v), timezone) => v
            .checked_mul(1_000_000)
            .map(|micros| timestamp_micros_datum(micros, timezone.is_some())),
        ScalarValue::TimestampMillisecond(Some(v), timezone) => v
            .checked_mul(1_000)
            .map(|micros| timestamp_micros_datum(micros, timezone.is_some())),
        ScalarValue::TimestampMicrosecond(Some(v), timezone) => {
            Some(timestamp_micros_datum(*v, timezone.is_some()))
        }
        ScalarValue::TimestampNanosecond(Some(v), timezone) => Some(if timezone.is_some() {
            Datum::timestamptz_nanos(*v)
        } else {
            Datum::timestamp_nanos(*v)
        }),
        _ => None,
    }
}

fn timestamp_micros_datum(micros: i64, zoned: bool) -> Datum {
    if zoned {
        Datum::timestamptz_micros(micros)
    } else {
        Datum::timestamp_micros(micros)
    }
}

pub(crate) fn scan_predicates(
    table: &Table,
    snapshot_id: Option<i64>,
    filters: &[Expr],
    bindings: &HashMap<String, Option<String>>,
) -> DFResult<Option<Predicate>> {
    let schema = match snapshot_id.and_then(|id| table.metadata().snapshot_by_id(id)) {
        Some(snapshot) => snapshot
            .schema(table.metadata())
            .map_err(to_datafusion_error)?,
        None => table.metadata().current_schema().clone(),
    };
    Ok(convert_filters_to_predicate(
        &rebind_filters(filters, bindings),
        &schema,
    ))
}

fn rebind_filters(filters: &[Expr], bindings: &HashMap<String, Option<String>>) -> Vec<Expr> {
    filters
        .iter()
        .filter_map(|filter| rebind_filter(filter, bindings))
        .collect()
}

fn rebind_filter(filter: &Expr, bindings: &HashMap<String, Option<String>>) -> Option<Expr> {
    let mut unbound = false;
    let rewritten = filter
        .clone()
        .transform(|node| {
            if let Expr::Column(column) = &node {
                match bindings.get(&column.name) {
                    Some(Some(scanned_name)) if scanned_name != &column.name => {
                        return Ok(Transformed::yes(Expr::Column(Column::new(
                            column.relation.clone(),
                            scanned_name,
                        ))));
                    }
                    Some(Some(_)) => {}
                    Some(None) | None => unbound = true,
                }
            }
            Ok(Transformed::no(node))
        })
        .ok()?;
    (!unbound).then_some(rewritten.data)
}

#[cfg(test)]
#[path = "expr_to_predicate_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "expr_to_predicate_nan_tests.rs"]
mod nan_tests;

#[cfg(test)]
#[path = "expr_to_predicate_ts_tz_tests.rs"]
mod ts_tz_tests;
