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

//! Canonical expression-JSON codec: a 1:1 port of Java `ExpressionParser` over the Rust
//! [`Predicate`] tree.
//!
//! | Node | Wire form |
//! |---|---|
//! | literal predicate | `{"type":<op>,"term":<term>,"value":<single-value>}` |
//! | set predicate (`in`/`not-in`) | `{"type":<op>,"term":<term>,"values":[<single-value>,...]}` |
//! | unary predicate | `{"type":<op>,"term":<term>}` |
//! | `and`/`or` | `{"type":"and"\|"or","left":<expr>,"right":<expr>}` |
//! | `not` | `{"type":"not","child":<expr>}` |
//! | `AlwaysTrue`/`AlwaysFalse` | the bare JSON booleans `true`/`false`, NOT an object |
//! | term | a bare JSON string holding the column name |
//! | op name | `Operation.toString()` with `_` mapped to `-`, lowercased |
//! | int, long, boolean | a JSON number or boolean |
//! | float, double | a JSON number whose TEXT is Java `Float.toString`/`Double.toString` |
//! | date, time, timestamp, timestamptz | an ISO-8601 string |
//! | decimal, uuid, binary, fixed | a string: plain decimal, lowercase uuid, uppercase base-16 |
//!
//! Java also emits `{"type":"transform",...}` terms. [`Reference`] is name-only, so this port
//! cannot write one and rejects one on the read side.
//!
//! The float text is NOT byte-for-byte 1:1. The JDK 11 `FloatingDecimal` is non-minimal and prints
//! more digits than the shortest round trip for some long-mantissa values, where Rust prints the
//! minimal form. JDK 19 adopted that minimal form. [`float_residue_is_jdk_nonminimal`] pins it.
//!
//! A Rust [`Predicate`] always carries a typed [`Datum`], but Java's schema-less `fromJson(String)`
//! collapses temporal and decimal literals. So [`from_json`] REQUIRES a [`Schema`]: it rebuilds
//! each literal from its bound field's type, the only round trip that preserves them.

use std::fmt::Write as _;

use fnv::FnvHashSet;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::expr::{
    BinaryExpression, LogicalExpression, Predicate, PredicateOperator, Reference, SetExpression,
    UnaryExpression,
};
use crate::spec::{Datum, PrimitiveLiteral, PrimitiveType, Schema};
use crate::{Error, ErrorKind, Result};

/// Maximum JSON nesting: malformed deep input must not overflow the stack. Java has no limit.
const MAX_DEPTH: u32 = 100;

// JSON field names — mirror the private `String` constants in Java `ExpressionParser`.
const TYPE: &str = "type";
const TERM: &str = "term";
const VALUE: &str = "value";
const VALUES: &str = "values";
const LEFT: &str = "left";
const RIGHT: &str = "right";
const CHILD: &str = "child";
const TRANSFORM: &str = "transform";

/// Serializes a [`Predicate`] to its canonical `ExpressionParser` JSON string.
///
/// ```rust
/// use iceberg::expr::Reference;
/// use iceberg::expr::expression_parser::to_json;
/// use iceberg::spec::Datum;
///
/// let pred = Reference::new("x").is_null();
/// assert_eq!(to_json(&pred).unwrap(), r#"{"type":"is-null","term":"x"}"#);
///
/// let pred = Reference::new("i").equal_to(Datum::int(42));
/// assert_eq!(
///     to_json(&pred).unwrap(),
///     r#"{"type":"eq","term":"i","value":42}"#
/// );
/// ```
pub fn to_json(predicate: &Predicate) -> Result<String> {
    // Built by hand to keep Java's field order: a `serde_json::Map` alphabetizes its keys here.
    let mut out = String::new();
    write_predicate(&mut out, predicate)?;
    Ok(out)
}

/// Parses a canonical `ExpressionParser` JSON string, recovering each literal's type from the
/// bound field of `schema`. Transform and aggregate terms are rejected.
///
/// ```rust
/// use std::sync::Arc;
///
/// use iceberg::expr::expression_parser::from_json;
/// use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
///
/// let schema = Schema::builder()
///     .with_fields(vec![Arc::new(NestedField::optional(
///         1,
///         "i",
///         Type::Primitive(PrimitiveType::Int),
///     ))])
///     .build()
///     .unwrap();
/// let pred = from_json(r#"{"type":"eq","term":"i","value":42}"#, &schema).unwrap();
/// assert_eq!(pred.to_string(), "i = 42");
/// ```
pub fn from_json(json: &str, schema: &Schema) -> Result<Predicate> {
    let value: JsonValue = serde_json::from_str(json).map_err(|e| {
        Error::new(ErrorKind::DataInvalid, "Failed to parse expression JSON").with_source(e)
    })?;
    value_to_predicate(&value, Some(schema), 0)
}

/// Parses a canonical `ExpressionParser` JSON string WITHOUT a schema, mirroring Java's untyped
/// `fromJson(String)`: an integral number becomes a `Long`, a floating number a `Double`. Date,
/// time, timestamp and decimal literals therefore COLLAPSE. It is the deserialize half of the
/// `ScanReport.filter` wire contract. Prefer [`from_json`] whenever a schema is available.
pub fn from_json_untyped(json: &str) -> Result<Predicate> {
    let value: JsonValue = serde_json::from_str(json).map_err(|e| {
        Error::new(ErrorKind::DataInvalid, "Failed to parse expression JSON").with_source(e)
    })?;
    value_to_predicate(&value, None, 0)
}

// Write side — Predicate to JSON (mirrors Java `$JsonGeneratorVisitor`).

/// Render the hyphenated lowercase op name (Java `operationType`).
fn op_type(op: PredicateOperator) -> &'static str {
    match op {
        PredicateOperator::IsNull => "is-null",
        PredicateOperator::NotNull => "not-null",
        PredicateOperator::IsNan => "is-nan",
        PredicateOperator::NotNan => "not-nan",
        PredicateOperator::LessThan => "lt",
        PredicateOperator::LessThanOrEq => "lt-eq",
        PredicateOperator::GreaterThan => "gt",
        PredicateOperator::GreaterThanOrEq => "gt-eq",
        PredicateOperator::Eq => "eq",
        PredicateOperator::NotEq => "not-eq",
        PredicateOperator::StartsWith => "starts-with",
        PredicateOperator::NotStartsWith => "not-starts-with",
        PredicateOperator::In => "in",
        PredicateOperator::NotIn => "not-in",
    }
}

/// Appends a JSON string literal, escaped by serde_json so it matches Jackson's output.
fn write_json_string(out: &mut String, s: &str) {
    // `to_string` on a `str` never fails, so the fallback is unreachable, not dead.
    let encoded = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string());
    out.push_str(&encoded);
}

fn write_predicate(out: &mut String, predicate: &Predicate) -> Result<()> {
    match predicate {
        // Java `alwaysTrue`/`alwaysFalse` write a bare boolean, not a typed object.
        Predicate::AlwaysTrue => out.push_str("true"),
        Predicate::AlwaysFalse => out.push_str("false"),
        Predicate::And(expr) => write_logical(out, "and", expr.inputs())?,
        Predicate::Or(expr) => write_logical(out, "or", expr.inputs())?,
        Predicate::Not(expr) => {
            let [child] = expr.inputs();
            out.push('{');
            write_field_name(out, TYPE);
            out.push_str("\"not\",");
            write_field_name(out, CHILD);
            write_predicate(out, child)?;
            out.push('}');
        }
        Predicate::Unary(expr) => write_unary(out, expr),
        Predicate::Binary(expr) => write_binary(out, expr)?,
        Predicate::Set(expr) => write_set(out, expr)?,
    }
    Ok(())
}

fn write_field_name(out: &mut String, key: &str) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
}

fn write_type_field(out: &mut String, ty: &str) {
    write_field_name(out, TYPE);
    write_json_string(out, ty);
    out.push(',');
}

fn write_logical(out: &mut String, ty: &str, inputs: [&Predicate; 2]) -> Result<()> {
    out.push('{');
    write_type_field(out, ty);
    write_field_name(out, LEFT);
    write_predicate(out, inputs[0])?;
    out.push(',');
    write_field_name(out, RIGHT);
    write_predicate(out, inputs[1])?;
    out.push('}');
    Ok(())
}

fn write_unary(out: &mut String, expr: &UnaryExpression<Reference>) {
    out.push('{');
    write_type_field(out, op_type(expr.op()));
    write_field_name(out, TERM);
    write_json_string(out, expr.term().name());
    out.push('}');
}

fn write_binary(out: &mut String, expr: &BinaryExpression<Reference>) -> Result<()> {
    out.push('{');
    write_type_field(out, op_type(expr.op()));
    write_field_name(out, TERM);
    write_json_string(out, expr.term().name());
    out.push(',');
    write_field_name(out, VALUE);
    write_value(out, expr.literal())?;
    out.push('}');
    Ok(())
}

fn write_set(out: &mut String, expr: &SetExpression<Reference>) -> Result<()> {
    out.push('{');
    write_type_field(out, op_type(expr.op()));
    write_field_name(out, TERM);
    write_json_string(out, expr.term().name());
    out.push(',');
    write_field_name(out, VALUES);
    out.push('[');
    for (i, datum) in expr.literals().iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_value(out, datum)?;
    }
    out.push(']');
    out.push('}');
    Ok(())
}

/// Appends a single value (Java `SingleValueParser.toJson`). Float and double take a bespoke
/// path: `serde_json::Number` cannot hold a token like `1.0E10`, so the token is appended raw.
fn write_value(out: &mut String, datum: &Datum) -> Result<()> {
    match datum.literal() {
        PrimitiveLiteral::Float(v) => {
            out.push_str(&format_java_float(f64::from(v.0), true)?);
        }
        PrimitiveLiteral::Double(v) => {
            out.push_str(&format_java_float(v.0, false)?);
        }
        _ => {
            let value = datum_to_value(datum)?;
            let encoded = serde_json::to_string(&value).map_err(|e| {
                Error::new(ErrorKind::Unexpected, "Failed to serialize literal value")
                    .with_source(e)
            })?;
            out.push_str(&encoded);
        }
    }
    Ok(())
}

// Value codec — Datum to JSON, byte-matching Java `SingleValueParser.toJson`.

fn datum_to_value(datum: &Datum) -> Result<JsonValue> {
    let value = match (datum.data_type(), datum.literal()) {
        (_, PrimitiveLiteral::Boolean(v)) => JsonValue::Bool(*v),
        (_, PrimitiveLiteral::Int(v)) if matches!(datum.data_type(), PrimitiveType::Int) => {
            JsonValue::Number((*v).into())
        }
        (_, PrimitiveLiteral::Long(v)) if matches!(datum.data_type(), PrimitiveType::Long) => {
            JsonValue::Number((*v).into())
        }
        // `write_value` emits floats itself, so reaching here means a value bypassed it.
        (_, lit @ PrimitiveLiteral::Float(_)) | (_, lit @ PrimitiveLiteral::Double(_)) => {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("Float/double literal {lit:?} must be serialized via write_value"),
            ));
        }
        (PrimitiveType::Date, PrimitiveLiteral::Int(_)) => JsonValue::String(datum.to_string()),
        (PrimitiveType::Time, PrimitiveLiteral::Long(v)) => JsonValue::String(format_iso_time(*v)),
        (PrimitiveType::Timestamp, PrimitiveLiteral::Long(v)) => {
            JsonValue::String(format_iso_timestamp(*v, false, false))
        }
        (PrimitiveType::Timestamptz, PrimitiveLiteral::Long(v)) => {
            JsonValue::String(format_iso_timestamp(*v, true, false))
        }
        (PrimitiveType::TimestampNs, PrimitiveLiteral::Long(v)) => {
            JsonValue::String(format_iso_timestamp(*v, false, true))
        }
        (PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(v)) => {
            JsonValue::String(format_iso_timestamp(*v, true, true))
        }
        // `Datum`'s Display already byte-matches Java for string, uuid, decimal and binary.
        (_, PrimitiveLiteral::String(v)) => JsonValue::String(v.clone()),
        (PrimitiveType::Uuid, PrimitiveLiteral::UInt128(_)) => JsonValue::String(datum.to_string()),
        (_, PrimitiveLiteral::Binary(_)) => JsonValue::String(datum.to_string()),
        (PrimitiveType::Decimal { .. }, PrimitiveLiteral::Int128(_)) => {
            JsonValue::String(datum.to_string())
        }
        (ty, lit) => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("Cannot serialize literal {lit:?} of type {ty} to expression JSON"),
            ));
        }
    };
    Ok(value)
}

/// Formats a finite float as the verbatim text of Java `Float.toString`/`Double.toString`. With
/// `is_float` the digits come from the `f32`, so the mantissa matches `Float.toString`. See the
/// module residue note. Non-finite values are rejected: Java quotes them or rejects them.
fn format_java_float(v: f64, is_float: bool) -> Result<String> {
    if !v.is_finite() {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!("Cannot serialize non-finite floating-point value {v} to expression JSON"),
        ));
    }
    Ok(crate::spec::java_to_string_float(v, is_float))
}

/// Formats `micros`-since-midnight as Java `DateTimeUtil.microsToIsoTime`, trailing zeros trimmed.
fn format_iso_time(micros: i64) -> String {
    let secs_total = micros / 1_000_000;
    let frac_micros = micros % 1_000_000;
    let hours = secs_total / 3_600;
    let minutes = (secs_total % 3_600) / 60;
    let seconds = secs_total % 60;
    let mut out = format!("{hours:02}:{minutes:02}:{seconds:02}");
    append_fraction(&mut out, frac_micros as u64, 6);
    out
}

/// Formats an epoch `value` as `yyyy-MM-ddTHH:mm:ss[.fraction]`, with `+00:00` when `tz`.
fn format_iso_timestamp(value: i64, tz: bool, nanos: bool) -> String {
    let (unit_per_sec, frac_width) = if nanos {
        (1_000_000_000i64, 9)
    } else {
        (1_000_000i64, 6)
    };
    // Floor division so the sub-second remainder is always non-negative (pre-epoch instants).
    let mut secs = value.div_euclid(unit_per_sec);
    let frac = value.rem_euclid(unit_per_sec);
    // The fraction is rendered here, not by chrono, to match the JDK's trimming.
    let naive = chrono::DateTime::from_timestamp(secs, 0)
        .map(|dt| dt.naive_utc())
        .unwrap_or_else(|| {
            // Defensive: `from_timestamp` returns None only far outside the calendar range.
            secs = 0;
            chrono::DateTime::UNIX_EPOCH.naive_utc()
        });
    let mut out = naive.format("%Y-%m-%dT%H:%M:%S").to_string();
    append_fraction(&mut out, frac as u64, frac_width);
    if tz {
        out.push_str("+00:00");
    }
    out
}

/// Appends a JDK-style fraction: `width` digits, trailing zeros stripped, no dot when zero.
fn append_fraction(out: &mut String, frac: u64, width: usize) {
    if frac == 0 {
        return;
    }
    let mut digits = format!("{frac:0width$}", width = width);
    while digits.ends_with('0') {
        digits.pop();
    }
    let _ = write!(out, ".{digits}");
}

// Read side — JSON to Predicate (mirrors Java `fromJson`/`predicateFromJson`).

fn value_to_predicate(value: &JsonValue, schema: Option<&Schema>, depth: u32) -> Result<Predicate> {
    if depth > MAX_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Expression JSON nesting exceeds the maximum depth of {MAX_DEPTH}"),
        ));
    }

    // A bare boolean node is AlwaysTrue / AlwaysFalse (Java `node.isBoolean()`).
    if let JsonValue::Bool(b) = value {
        return Ok(if *b {
            Predicate::AlwaysTrue
        } else {
            Predicate::AlwaysFalse
        });
    }

    let obj = value.as_object().ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot parse expression from non-object: {value}"),
        )
    })?;

    let ty = get_str(obj, TYPE)?;

    match ty {
        "and" | "or" => {
            let left = value_to_predicate(get_field(obj, LEFT)?, schema, depth + 1)?;
            let right = value_to_predicate(get_field(obj, RIGHT)?, schema, depth + 1)?;
            let logical = LogicalExpression::new([Box::new(left), Box::new(right)]);
            Ok(if ty == "and" {
                Predicate::And(logical)
            } else {
                Predicate::Or(logical)
            })
        }
        "not" => {
            let child = value_to_predicate(get_field(obj, CHILD)?, schema, depth + 1)?;
            Ok(Predicate::Not(LogicalExpression::new([Box::new(child)])))
        }
        // `literal` is a reserved type keyword in Java's grammar but never a top-level predicate.
        "literal" => Err(Error::new(
            ErrorKind::DataInvalid,
            "Cannot parse expression: 'literal' is not a valid predicate type",
        )),
        _ => predicate_from_obj(ty, obj, schema),
    }
}

fn predicate_from_obj(
    ty: &str,
    obj: &JsonMap<String, JsonValue>,
    schema: Option<&Schema>,
) -> Result<Predicate> {
    let op = op_from_type(ty)?;
    // Without a schema, `field_type` is `None` and the literal follows the JSON value's own shape.
    let (field_name, field_type) = read_term(obj, schema)?;
    let reference = Reference::new(field_name);

    if op.is_unary() {
        if obj.contains_key(VALUE) || obj.contains_key(VALUES) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot parse {ty} predicate: unexpected value/values field"),
            ));
        }
        Ok(Predicate::Unary(UnaryExpression::new(op, reference)))
    } else if op.is_set() {
        let values = get_field(obj, VALUES).map_err(|_| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot parse {ty} predicate: missing values"),
            )
        })?;
        let array = values.as_array().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot parse literals from non-array: {values}"),
            )
        })?;
        let literals = array
            .iter()
            .map(|v| value_to_datum(v, field_type.as_ref()))
            .collect::<Result<FnvHashSet<Datum>>>()?;
        Ok(Predicate::Set(SetExpression::new(op, reference, literals)))
    } else {
        let value = get_field(obj, VALUE).map_err(|_| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot parse {ty} predicate: missing value"),
            )
        })?;
        let datum = value_to_datum(value, field_type.as_ref())?;
        Ok(Predicate::Binary(BinaryExpression::new(
            op, reference, datum,
        )))
    }
}

/// Reads the `term` field, rejecting transform terms, and resolves the bound field's type.
fn read_term(
    obj: &JsonMap<String, JsonValue>,
    schema: Option<&Schema>,
) -> Result<(String, Option<PrimitiveType>)> {
    let term = get_field(obj, TERM)?;
    let name = match term {
        JsonValue::String(s) => s.clone(),
        JsonValue::Object(t) => {
            // A Rust `Predicate` cannot hold Java's UnboundTransform, so reject it, never drop it.
            if t.contains_key(TRANSFORM) {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    "Transform terms are not supported in Rust expression JSON",
                ));
            }
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Only bare-reference terms are supported in Rust expression JSON",
            ));
        }
        other => {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot parse term from: {other}"),
            ));
        }
    };

    let Some(schema) = schema else {
        return Ok((name, None));
    };

    let field = schema.field_by_name(&name).ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot bind term '{name}': field not found in schema"),
        )
    })?;
    let field_type = match &field.field_type.as_ref() {
        crate::spec::Type::Primitive(p) => p.clone(),
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("Cannot build a literal for non-primitive field '{name}' of type {other}"),
            ));
        }
    };
    Ok((name, Some(field_type)))
}

/// Reverse the hyphen-map (`-` → `_`) and resolve the operator (Java `fromType`).
fn op_from_type(ty: &str) -> Result<PredicateOperator> {
    Ok(match ty {
        "is-null" => PredicateOperator::IsNull,
        "not-null" => PredicateOperator::NotNull,
        "is-nan" => PredicateOperator::IsNan,
        "not-nan" => PredicateOperator::NotNan,
        "lt" => PredicateOperator::LessThan,
        "lt-eq" => PredicateOperator::LessThanOrEq,
        "gt" => PredicateOperator::GreaterThan,
        "gt-eq" => PredicateOperator::GreaterThanOrEq,
        "eq" => PredicateOperator::Eq,
        "not-eq" => PredicateOperator::NotEq,
        "starts-with" => PredicateOperator::StartsWith,
        "not-starts-with" => PredicateOperator::NotStartsWith,
        "in" => PredicateOperator::In,
        "not-in" => PredicateOperator::NotIn,
        other => {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Invalid operation type: {other}"),
            ));
        }
    })
}

/// Builds a [`Datum`]: with `Some(field_type)` typed, with `None` Java's untyped `asObject`.
fn value_to_datum(value: &JsonValue, field_type: Option<&PrimitiveType>) -> Result<Datum> {
    let Some(field_type) = field_type else {
        return value_to_datum_untyped(value);
    };
    let type_err = || {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot parse {field_type} value from JSON: {value}"),
        )
    };
    let as_str = |v: &JsonValue| v.as_str().map(str::to_string).ok_or_else(type_err);

    Ok(match field_type {
        PrimitiveType::Boolean => Datum::bool(value.as_bool().ok_or_else(type_err)?),
        PrimitiveType::Int => {
            let n = value.as_i64().ok_or_else(type_err)?;
            let n: i32 = n.try_into().map_err(|_| type_err())?;
            Datum::int(n)
        }
        PrimitiveType::Long => Datum::long(value.as_i64().ok_or_else(type_err)?),
        PrimitiveType::Float => {
            let n = value.as_f64().ok_or_else(type_err)?;
            Datum::float(n as f32)
        }
        PrimitiveType::Double => Datum::double(value.as_f64().ok_or_else(type_err)?),
        PrimitiveType::Date => Datum::date_from_str(as_str(value)?)?,
        PrimitiveType::Time => Datum::time_from_str(as_str(value)?)?,
        PrimitiveType::Timestamp => Datum::timestamp_from_str(as_str(value)?)?,
        PrimitiveType::Timestamptz => Datum::timestamptz_from_str(as_str(value)?)?,
        // No `*_nanos_from_str` constructor exists, so parse the ISO string to nanos directly.
        PrimitiveType::TimestampNs => Datum::timestamp_nanos(iso_to_nanos(&as_str(value)?, false)?),
        PrimitiveType::TimestamptzNs => {
            Datum::timestamptz_nanos(iso_to_nanos(&as_str(value)?, true)?)
        }
        PrimitiveType::String => Datum::string(as_str(value)?),
        PrimitiveType::Uuid => Datum::uuid_from_str(as_str(value)?)?,
        PrimitiveType::Fixed(_) => Datum::fixed(hex_to_bytes(&as_str(value)?)?),
        PrimitiveType::Binary => Datum::binary(hex_to_bytes(&as_str(value)?)?),
        PrimitiveType::Decimal { .. } => Datum::decimal_from_str(as_str(value)?)?,
        // The V3 `unknown` type has no literal representation in ExpressionParser JSON.
        PrimitiveType::Unknown => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Cannot build a literal for the 'unknown' type from expression JSON",
            ));
        }
    })
}

/// Builds a [`Datum`] with no type context. Temporal, decimal and binary types CANNOT be
/// recovered here, exactly as in Java's untyped `asObject`.
fn value_to_datum_untyped(value: &JsonValue) -> Result<Datum> {
    match value {
        JsonValue::Bool(b) => Ok(Datum::bool(*b)),
        JsonValue::String(s) => Ok(Datum::string(s)),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Datum::long(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Datum::double(f))
            } else {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("Cannot parse numeric literal from JSON: {value}"),
                ))
            }
        }
        other => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot parse untyped literal from JSON: {other}"),
        )),
    }
}

/// Decodes a base-16 string to bytes. Mixed case is accepted; odd-length or non-hex is rejected.
fn hex_to_bytes(s: &str) -> Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot parse hex string of odd length: {s}"),
        ));
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let nibble = |b: u8| -> Result<u8> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Invalid hex character: {}", b as char),
            )),
        }
    };
    for chunk in bytes.chunks_exact(2) {
        out.push((nibble(chunk[0])? << 4) | nibble(chunk[1])?);
    }
    Ok(out)
}

/// Parses an ISO-8601 timestamp to nanos since the epoch; with `tz` the input is zoned.
fn iso_to_nanos(s: &str, tz: bool) -> Result<i64> {
    let parse_err = |e: chrono::ParseError| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot parse ISO nanosecond timestamp from: {s}"),
        )
        .with_source(e)
    };
    let (secs, subsec_nanos) = if tz {
        let dt = s
            .parse::<chrono::DateTime<chrono::Utc>>()
            .map_err(parse_err)?;
        (dt.timestamp(), dt.timestamp_subsec_nanos())
    } else {
        let dt = s.parse::<chrono::NaiveDateTime>().map_err(parse_err)?;
        let utc = dt.and_utc();
        (utc.timestamp(), utc.timestamp_subsec_nanos())
    };
    secs.checked_mul(1_000_000_000)
        .and_then(|n| n.checked_add(i64::from(subsec_nanos)))
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("ISO nanosecond timestamp out of range: {s}"),
            )
        })
}

// JSON object helpers.

fn get_field<'a>(obj: &'a JsonMap<String, JsonValue>, key: &str) -> Result<&'a JsonValue> {
    obj.get(key).ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot parse missing field: {key}"),
        )
    })
}

fn get_str<'a>(obj: &'a JsonMap<String, JsonValue>, key: &str) -> Result<&'a str> {
    get_field(obj, key)?.as_str().ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot parse non-string field: {key}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::spec::{NestedField, Type};

    fn test_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::optional(
                    1,
                    "i",
                    Type::Primitive(PrimitiveType::Int),
                )),
                Arc::new(NestedField::optional(
                    2,
                    "l",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::optional(
                    3,
                    "s",
                    Type::Primitive(PrimitiveType::String),
                )),
                Arc::new(NestedField::optional(
                    4,
                    "b",
                    Type::Primitive(PrimitiveType::Boolean),
                )),
                Arc::new(NestedField::optional(
                    5,
                    "d",
                    Type::Primitive(PrimitiveType::Date),
                )),
                Arc::new(NestedField::optional(
                    6,
                    "t",
                    Type::Primitive(PrimitiveType::Time),
                )),
                Arc::new(NestedField::optional(
                    7,
                    "ts",
                    Type::Primitive(PrimitiveType::Timestamp),
                )),
                Arc::new(NestedField::optional(
                    8,
                    "tstz",
                    Type::Primitive(PrimitiveType::Timestamptz),
                )),
                Arc::new(NestedField::optional(
                    9,
                    "dec",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 9,
                        scale: 2,
                    }),
                )),
                Arc::new(NestedField::optional(
                    10,
                    "u",
                    Type::Primitive(PrimitiveType::Uuid),
                )),
                Arc::new(NestedField::optional(
                    11,
                    "bin",
                    Type::Primitive(PrimitiveType::Binary),
                )),
                Arc::new(NestedField::optional(
                    12,
                    "fx",
                    Type::Primitive(PrimitiveType::Fixed(3)),
                )),
                Arc::new(NestedField::optional(
                    13,
                    "f",
                    Type::Primitive(PrimitiveType::Float),
                )),
                // Java's `predicateFromJson` always binds the term, so `x`/`y` must exist here.
                Arc::new(NestedField::optional(
                    14,
                    "x",
                    Type::Primitive(PrimitiveType::Int),
                )),
                Arc::new(NestedField::optional(
                    15,
                    "y",
                    Type::Primitive(PrimitiveType::Int),
                )),
                Arc::new(NestedField::optional(
                    16,
                    "dbl",
                    Type::Primitive(PrimitiveType::Double),
                )),
            ])
            .build()
            .unwrap()
    }

    /// Assert `to_json` produces exactly `expected` and `from_json` round-trips back to `pred`.
    fn assert_roundtrip(pred: Predicate, expected: &str) {
        let json = to_json(&pred).unwrap();
        assert_eq!(json, expected, "to_json byte output");
        let back = from_json(&json, &test_schema()).unwrap();
        assert_eq!(back, pred, "from_json round-trip");
    }

    // Hand-pinned fixtures: the REAL Java output, so the test cannot be circular.

    #[test]
    fn pinned_is_null() {
        assert_roundtrip(
            Reference::new("x").is_null(),
            r#"{"type":"is-null","term":"x"}"#,
        );
    }

    #[test]
    fn pinned_not_null() {
        assert_roundtrip(
            Reference::new("x").is_not_null(),
            r#"{"type":"not-null","term":"x"}"#,
        );
    }

    #[test]
    fn pinned_is_nan_not_nan() {
        assert_roundtrip(
            Reference::new("f").is_nan(),
            r#"{"type":"is-nan","term":"f"}"#,
        );
        assert_roundtrip(
            Reference::new("f").is_not_nan(),
            r#"{"type":"not-nan","term":"f"}"#,
        );
    }

    #[test]
    fn pinned_eq_int() {
        assert_roundtrip(
            Reference::new("i").equal_to(Datum::int(42)),
            r#"{"type":"eq","term":"i","value":42}"#,
        );
    }

    #[test]
    fn pinned_eq_long() {
        assert_roundtrip(
            Reference::new("l").equal_to(Datum::long(9_000_000_000i64)),
            r#"{"type":"eq","term":"l","value":9000000000}"#,
        );
    }

    #[test]
    fn pinned_all_binary_ops() {
        assert_roundtrip(
            Reference::new("i").less_than(Datum::int(10)),
            r#"{"type":"lt","term":"i","value":10}"#,
        );
        assert_roundtrip(
            Reference::new("i").less_than_or_equal_to(Datum::int(10)),
            r#"{"type":"lt-eq","term":"i","value":10}"#,
        );
        assert_roundtrip(
            Reference::new("i").greater_than(Datum::int(10)),
            r#"{"type":"gt","term":"i","value":10}"#,
        );
        assert_roundtrip(
            Reference::new("i").greater_than_or_equal_to(Datum::int(10)),
            r#"{"type":"gt-eq","term":"i","value":10}"#,
        );
        assert_roundtrip(
            Reference::new("s").not_equal_to(Datum::string("hi")),
            r#"{"type":"not-eq","term":"s","value":"hi"}"#,
        );
        assert_roundtrip(
            Reference::new("s").starts_with(Datum::string("pre")),
            r#"{"type":"starts-with","term":"s","value":"pre"}"#,
        );
        assert_roundtrip(
            Reference::new("s").not_starts_with(Datum::string("pre")),
            r#"{"type":"not-starts-with","term":"s","value":"pre"}"#,
        );
    }

    #[test]
    fn pinned_eq_bool_str() {
        assert_roundtrip(
            Reference::new("b").equal_to(Datum::bool(true)),
            r#"{"type":"eq","term":"b","value":true}"#,
        );
        assert_roundtrip(
            Reference::new("s").equal_to(Datum::string("hello")),
            r#"{"type":"eq","term":"s","value":"hello"}"#,
        );
    }

    #[test]
    fn pinned_temporal_values() {
        assert_roundtrip(
            Reference::new("d").equal_to(Datum::date_from_str("2017-11-16").unwrap()),
            r#"{"type":"eq","term":"d","value":"2017-11-16"}"#,
        );
        assert_roundtrip(
            Reference::new("t").equal_to(Datum::time_from_str("13:14:15.000001").unwrap()),
            r#"{"type":"eq","term":"t","value":"13:14:15.000001"}"#,
        );
        assert_roundtrip(
            Reference::new("ts")
                .equal_to(Datum::timestamp_from_str("2017-11-16T14:15:16.123456").unwrap()),
            r#"{"type":"eq","term":"ts","value":"2017-11-16T14:15:16.123456"}"#,
        );
        assert_roundtrip(
            Reference::new("tstz")
                .equal_to(Datum::timestamptz_from_str("2017-11-16T14:15:16.123456+00:00").unwrap()),
            r#"{"type":"eq","term":"tstz","value":"2017-11-16T14:15:16.123456+00:00"}"#,
        );
    }

    #[test]
    fn iso_fraction_trimming_matches_jdk() {
        assert_eq!(
            to_json(&Reference::new("ts").equal_to(Datum::timestamp_micros(1_510_841_716_000_000)))
                .unwrap(),
            r#"{"type":"eq","term":"ts","value":"2017-11-16T14:15:16"}"#
        );
        // 0.5s -> ".5" (NOT ".500000"); JDK ISO_LOCAL_TIME trims trailing zeros.
        assert_eq!(
            to_json(&Reference::new("t").equal_to(Datum::time_micros(500_000).unwrap())).unwrap(),
            r#"{"type":"eq","term":"t","value":"00:00:00.5"}"#
        );
        assert_eq!(
            to_json(&Reference::new("tstz").equal_to(Datum::timestamptz_micros(0))).unwrap(),
            r#"{"type":"eq","term":"tstz","value":"1970-01-01T00:00:00+00:00"}"#
        );
    }

    #[test]
    fn pinned_decimal_uuid_binary_fixed() {
        assert_roundtrip(
            Reference::new("dec").equal_to(Datum::decimal_from_str("12.34").unwrap()),
            r#"{"type":"eq","term":"dec","value":"12.34"}"#,
        );
        assert_roundtrip(
            Reference::new("dec").equal_to(Datum::decimal_from_str("12.30").unwrap()),
            r#"{"type":"eq","term":"dec","value":"12.30"}"#,
        );
        assert_roundtrip(
            Reference::new("u")
                .equal_to(Datum::uuid_from_str("f79c3e09-677c-4bbd-a479-3f349cb785e7").unwrap()),
            r#"{"type":"eq","term":"u","value":"f79c3e09-677c-4bbd-a479-3f349cb785e7"}"#,
        );
        assert_roundtrip(
            Reference::new("bin").equal_to(Datum::binary(vec![0x01, 0xAB, 0xFF])),
            r#"{"type":"eq","term":"bin","value":"01ABFF"}"#,
        );
        assert_roundtrip(
            Reference::new("fx").equal_to(Datum::fixed(vec![0x0A, 0x0B, 0x0C])),
            r#"{"type":"eq","term":"fx","value":"0A0B0C"}"#,
        );
    }

    #[test]
    fn pinned_float_double_values() {
        // EXACT Java output: plain decimal inside a [-3, 7) exponent, scientific outside it.
        assert_roundtrip(
            Reference::new("dbl").equal_to(Datum::double(1234.5)),
            r#"{"type":"eq","term":"dbl","value":1234.5}"#,
        );
        assert_roundtrip(
            Reference::new("dbl").equal_to(Datum::double(1.0)),
            r#"{"type":"eq","term":"dbl","value":1.0}"#,
        );
        assert_roundtrip(
            Reference::new("dbl").equal_to(Datum::double(0.001)),
            r#"{"type":"eq","term":"dbl","value":0.001}"#,
        );
        assert_roundtrip(
            Reference::new("dbl").equal_to(Datum::double(1e10)),
            r#"{"type":"eq","term":"dbl","value":1.0E10}"#,
        );
        assert_roundtrip(
            Reference::new("dbl").equal_to(Datum::double(1e16)),
            r#"{"type":"eq","term":"dbl","value":1.0E16}"#,
        );
        assert_roundtrip(
            Reference::new("dbl").equal_to(Datum::double(1e-4)),
            r#"{"type":"eq","term":"dbl","value":1.0E-4}"#,
        );
        assert_roundtrip(
            Reference::new("dbl").equal_to(Datum::double(0.0005)),
            r#"{"type":"eq","term":"dbl","value":5.0E-4}"#,
        );
        assert_roundtrip(
            Reference::new("f").equal_to(Datum::float(2.5f32)),
            r#"{"type":"eq","term":"f","value":2.5}"#,
        );
        assert_roundtrip(
            Reference::new("f").equal_to(Datum::float(1e10f32)),
            r#"{"type":"eq","term":"f","value":1.0E10}"#,
        );
        assert_eq!(
            to_json(&Reference::new("dbl").equal_to(Datum::double(-0.0))).unwrap(),
            r#"{"type":"eq","term":"dbl","value":-0.0}"#,
        );
    }

    #[test]
    fn float_residue_is_jdk_nonminimal() {
        // NAMED RESIDUE GUARD. JDK 11 `Double.toString` is non-minimal, so its bytes differ from
        // Rust's minimal form for a narrow class of values. For bits 0x43d4f2247a5cdbed it prints
        // 17 significant digits and Rust prints 16. Both round-trip to the same f64.
        let v = f64::from_bits(0x43d4f2247a5cdbed);
        let json = to_json(&Reference::new("dbl").equal_to(Datum::double(v))).unwrap();
        assert_eq!(
            json, r#"{"type":"eq","term":"dbl","value":6.037235732340258E18}"#,
            "Rust emits the minimal shortest-round-trip form (JDK 19+); the JDK 11 oracle would \
             emit the non-minimal 6.0372357323402578E18 — this is the named residue, not parity",
        );
        // The minimal text parses back to the identical f64: the divergence is textual only.
        let back = from_json(&json, &test_schema()).unwrap();
        assert_eq!(back, Reference::new("dbl").equal_to(Datum::double(v)));

        // The f32 quirk is the same effect: JDK 11 prints `1.00000003E16`, Rust `1.0E16`.
        let json = to_json(&Reference::new("f").equal_to(Datum::float(1e16f32))).unwrap();
        assert_eq!(
            json, r#"{"type":"eq","term":"f","value":1.0E16}"#,
            "Rust emits the minimal f32 form; JDK 11 would emit the non-minimal 1.00000003E16",
        );
    }

    #[test]
    fn non_finite_float_rejected() {
        // Both are out of scope here and must fail loudly, never emit a divergent token.
        let err =
            to_json(&Reference::new("dbl").equal_to(Datum::double(f64::INFINITY))).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
        let err =
            to_json(&Reference::new("dbl").equal_to(Datum::double(f64::NEG_INFINITY))).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    }

    #[test]
    fn pinned_logical_and_or_not() {
        assert_roundtrip(
            Predicate::And(LogicalExpression::new([
                Box::new(Reference::new("i").equal_to(Datum::int(1))),
                Box::new(Reference::new("l").equal_to(Datum::long(2))),
            ])),
            r#"{"type":"and","left":{"type":"eq","term":"i","value":1},"right":{"type":"eq","term":"l","value":2}}"#,
        );
        assert_roundtrip(
            Predicate::Or(LogicalExpression::new([
                Box::new(Reference::new("x").is_null()),
                Box::new(Reference::new("y").is_not_null()),
            ])),
            r#"{"type":"or","left":{"type":"is-null","term":"x"},"right":{"type":"not-null","term":"y"}}"#,
        );
        assert_roundtrip(
            Predicate::Not(LogicalExpression::new([Box::new(
                Reference::new("i").equal_to(Datum::int(5)),
            )])),
            r#"{"type":"not","child":{"type":"eq","term":"i","value":5}}"#,
        );
    }

    #[test]
    fn pinned_always_true_false() {
        assert_roundtrip(Predicate::AlwaysTrue, "true");
        assert_roundtrip(Predicate::AlwaysFalse, "false");
    }

    #[test]
    fn set_predicates_in_not_in() {
        let pred = Predicate::Set(SetExpression::new(
            PredicateOperator::In,
            Reference::new("i"),
            [Datum::int(1), Datum::int(2), Datum::int(3)]
                .into_iter()
                .collect(),
        ));
        let json = to_json(&pred).unwrap();
        let back = from_json(&json, &test_schema()).unwrap();
        assert_eq!(back, pred);
        let v: JsonValue = serde_json::from_str(&json).unwrap();
        assert_eq!(v["type"], "in");
        assert_eq!(v["term"], "i");
        assert_eq!(v["values"].as_array().unwrap().len(), 3);

        let pred = Predicate::Set(SetExpression::new(
            PredicateOperator::NotIn,
            Reference::new("s"),
            [Datum::string("a"), Datum::string("b")]
                .into_iter()
                .collect(),
        ));
        let json = to_json(&pred).unwrap();
        assert_eq!(from_json(&json, &test_schema()).unwrap(), pred);
    }

    #[test]
    fn parse_pinned_fixtures_exactly() {
        let p = from_json(r#"{"type":"is-null","term":"x"}"#, &test_schema()).unwrap();
        assert_eq!(p, Reference::new("x").is_null());
        let p = from_json(
            r#"{"type":"eq","term":"d","value":"2017-11-16"}"#,
            &test_schema(),
        )
        .unwrap();
        assert_eq!(
            p,
            Reference::new("d").equal_to(Datum::date_from_str("2017-11-16").unwrap())
        );
    }

    #[test]
    fn deeply_nested_tree_round_trips_within_limit() {
        let mut pred = Reference::new("i").equal_to(Datum::int(0));
        for n in 1..30 {
            pred = Predicate::And(LogicalExpression::new([
                Box::new(pred),
                Box::new(Reference::new("i").equal_to(Datum::int(n))),
            ]));
        }
        let json = to_json(&pred).unwrap();
        assert_eq!(from_json(&json, &test_schema()).unwrap(), pred);
    }

    #[test]
    fn depth_limit_rejects_pathological_nesting() {
        // Construct JSON nested beyond MAX_DEPTH; must fail closed, not overflow the stack.
        let mut json = r#"{"type":"is-null","term":"x"}"#.to_string();
        for _ in 0..(MAX_DEPTH + 10) {
            json = format!(r#"{{"type":"not","child":{json}}}"#);
        }
        let err = from_json(&json, &test_schema()).unwrap_err();
        assert!(
            err.to_string().contains("maximum depth"),
            "expected depth-limit error, got: {err}"
        );
    }

    #[test]
    fn reject_transform_term() {
        let err = from_json(
            r#"{"type":"eq","term":{"type":"transform","transform":"bucket[16]","term":"i"},"value":1}"#,
            &test_schema(),
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
        assert!(err.to_string().contains("Transform terms"));
    }

    #[test]
    fn reject_unknown_op() {
        let err =
            from_json(r#"{"type":"between","term":"i","value":1}"#, &test_schema()).unwrap_err();
        assert!(err.to_string().contains("Invalid operation type"));
    }

    #[test]
    fn reject_missing_value_on_binary() {
        let err = from_json(r#"{"type":"eq","term":"i"}"#, &test_schema()).unwrap_err();
        assert!(err.to_string().contains("missing value"));
    }

    #[test]
    fn mutation_wrong_op_hyphen_fails_byte_assert() {
        // Assert the hyphenated form directly: `is_null` with an underscore must fail here.
        assert_eq!(op_type(PredicateOperator::IsNull), "is-null");
        assert_eq!(op_type(PredicateOperator::NotEq), "not-eq");
        assert_eq!(op_type(PredicateOperator::LessThanOrEq), "lt-eq");
        assert_eq!(op_type(PredicateOperator::NotIn), "not-in");
        assert_eq!(op_type(PredicateOperator::NotStartsWith), "not-starts-with");
    }

    #[test]
    fn mutation_date_not_serialized_as_int() {
        // A date must serialize as an ISO string, never the raw days int.
        let json =
            to_json(&Reference::new("d").equal_to(Datum::date_from_str("2017-11-16").unwrap()))
                .unwrap();
        assert!(json.contains(r#""value":"2017-11-16""#));
        assert!(!json.contains(r#""value":17486"#));
    }
}
