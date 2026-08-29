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

//! Typed literals with validation

use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};
use ordered_float::{Float, OrderedFloat};
use serde::de::{self, MapAccess};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

use super::decimal_utils::{
    Decimal, decimal_from_i128_with_scale, decimal_from_str_exact, decimal_mantissa, decimal_scale,
    i128_from_be_bytes, i128_to_be_bytes_min,
};
use super::literal::Literal;
use super::primitive::PrimitiveLiteral;
use super::serde::_serde::RawLiteral;
use super::temporal::{date, time, timestamp, timestamptz};
use crate::error::Result;
use crate::spec::MAX_DECIMAL_PRECISION;
use crate::spec::datatypes::{PrimitiveType, Type, ensure_java_decimal_precision};
use crate::{Error, ErrorKind, ensure_data_valid};

/// Maximum [`PrimitiveType::Time`] value in microseconds: one microsecond before 24 hours.
pub(crate) const MAX_TIME_VALUE: i64 = 24 * 60 * 60 * 1_000_000i64 - 1;

pub(crate) const INT_MAX: i32 = 2147483647;
pub(crate) const INT_MIN: i32 = -2147483648;

/// Metadata gate for the decimal encode paths. Requires `1 <= precision <= 38`. Java sets the
/// upper bound; the lower one is a fork addition, because [`Type::decimal_required_bytes`] has no
/// byte width for `0`. `scale <= precision` is NOT required: Java allows it, so would reject data
/// Java writes.
fn validate_decimal_type(r#type: &PrimitiveType) -> Result<()> {
    if let PrimitiveType::Decimal { precision, .. } = r#type {
        ensure_data_valid!(
            *precision > 0 && *precision <= MAX_DECIMAL_PRECISION,
            "PrimitiveType Decimal must have valid precision from 1 through {MAX_DECIMAL_PRECISION}, got {precision}",
        );
    }
    Ok(())
}

/// Validate a decimal's metadata and unscaled value against its declared precision.
///
/// Encode path only. Java never checks the magnitude; the fork must, because the encoder
/// otherwise truncates the two's-complement buffer to `decimal_required_bytes`. [`crate::inspect`]
/// and manifest re-writes reach it too, so an over-wide bound scans fine but fails those with
/// `DataInvalid`. That is deliberate: the alternative is a silently truncated, wrong bound.
pub(crate) fn validate_decimal_value(r#type: &PrimitiveType, value: i128) -> Result<()> {
    let PrimitiveType::Decimal { precision, .. } = r#type else {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Decimal value {value} has non-decimal type {}", r#type),
        ));
    };
    validate_decimal_type(r#type)?;

    let actual_precision = value.unsigned_abs().to_string().len();
    ensure_data_valid!(
        actual_precision <= usize::try_from(*precision)?,
        "Decimal value {value} is too large for precision {precision}",
    );
    Ok(())
}

pub(crate) fn validate_decimal_literal(
    r#type: &PrimitiveType,
    literal: &PrimitiveLiteral,
) -> Result<()> {
    match (r#type, literal) {
        (PrimitiveType::Decimal { .. }, PrimitiveLiteral::Int128(value)) => {
            validate_decimal_value(r#type, *value)
        }
        (PrimitiveType::Decimal { .. }, _) => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Decimal type {} requires an Int128 literal", r#type),
        )),
        _ => validate_decimal_type(r#type),
    }
}

/// A literal with its type. Construction checks the pair, so the type always matches the value.
/// A plain [`PrimitiveLiteral`] omits the type to save space; carry the type where a consumer
/// needs it, as an unbound expression does.
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub struct Datum {
    r#type: PrimitiveType,
    literal: PrimitiveLiteral,
}

impl Serialize for Datum {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        // No decimal value gate here: a Java-written bound can exceed its declared precision, and
        // a gate would make that scan task unserializable. `serialize_decimal` holds `<= 38`.
        let mut struct_ser = serializer
            .serialize_struct("Datum", 2)
            .map_err(serde::ser::Error::custom)?;
        struct_ser
            .serialize_field("type", &self.r#type)
            .map_err(serde::ser::Error::custom)?;
        struct_ser
            .serialize_field(
                "literal",
                &RawLiteral::try_from(
                    Literal::Primitive(self.literal.clone()),
                    &Type::Primitive(self.r#type.clone()),
                )
                .map_err(serde::ser::Error::custom)?,
            )
            .map_err(serde::ser::Error::custom)?;
        struct_ser.end()
    }
}

impl<'de> Deserialize<'de> for Datum {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Type,
            Literal,
        }

        struct DatumVisitor;

        impl<'de> serde::de::Visitor<'de> for DatumVisitor {
            type Value = Datum;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Datum")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where A: serde::de::SeqAccess<'de> {
                let r#type = seq
                    .next_element::<PrimitiveType>()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let value = seq
                    .next_element::<RawLiteral>()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let Literal::Primitive(primitive) = value
                    .try_into(&Type::Primitive(r#type.clone()))
                    .map_err(serde::de::Error::custom)?
                    .ok_or_else(|| serde::de::Error::custom("None value"))?
                else {
                    return Err(serde::de::Error::custom("Invalid value"));
                };

                Ok(Datum::new(r#type, primitive))
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<Datum, V::Error>
            where V: MapAccess<'de> {
                let mut raw_primitive: Option<RawLiteral> = None;
                let mut r#type: Option<PrimitiveType> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Type => {
                            if r#type.is_some() {
                                return Err(de::Error::duplicate_field("type"));
                            }
                            r#type = Some(map.next_value()?);
                        }
                        Field::Literal => {
                            if raw_primitive.is_some() {
                                return Err(de::Error::duplicate_field("literal"));
                            }
                            raw_primitive = Some(map.next_value()?);
                        }
                    }
                }
                let Some(r#type) = r#type else {
                    return Err(serde::de::Error::missing_field("type"));
                };
                let Some(raw_primitive) = raw_primitive else {
                    return Err(serde::de::Error::missing_field("literal"));
                };
                let Literal::Primitive(primitive) = raw_primitive
                    .try_into(&Type::Primitive(r#type.clone()))
                    .map_err(serde::de::Error::custom)?
                    .ok_or_else(|| serde::de::Error::custom("None value"))?
                else {
                    return Err(serde::de::Error::custom("Invalid value"));
                };
                Ok(Datum::new(r#type, primitive))
            }
        }
        const FIELDS: &[&str] = &["type", "literal"];
        deserializer.deserialize_struct("Datum", FIELDS, DatumVisitor)
    }
}

// Compare following iceberg float ordering rules:
//  -NaN < -Infinity < -value < -0 < 0 < value < Infinity < NaN
fn iceberg_float_cmp_f32(a: OrderedFloat<f32>, b: OrderedFloat<f32>) -> Option<Ordering> {
    Some(a.total_cmp(&b))
}

fn iceberg_float_cmp_f64(a: OrderedFloat<f64>, b: OrderedFloat<f64>) -> Option<Ordering> {
    Some(a.total_cmp(&b))
}

impl PartialOrd for Datum {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (&self.literal, &other.literal, &self.r#type, &other.r#type) {
            (
                PrimitiveLiteral::Boolean(val),
                PrimitiveLiteral::Boolean(other_val),
                PrimitiveType::Boolean,
                PrimitiveType::Boolean,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Int(val),
                PrimitiveLiteral::Int(other_val),
                PrimitiveType::Int,
                PrimitiveType::Int,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Long(val),
                PrimitiveLiteral::Long(other_val),
                PrimitiveType::Long,
                PrimitiveType::Long,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Float(val),
                PrimitiveLiteral::Float(other_val),
                PrimitiveType::Float,
                PrimitiveType::Float,
            ) => iceberg_float_cmp_f32(*val, *other_val),
            (
                PrimitiveLiteral::Double(val),
                PrimitiveLiteral::Double(other_val),
                PrimitiveType::Double,
                PrimitiveType::Double,
            ) => iceberg_float_cmp_f64(*val, *other_val),
            (
                PrimitiveLiteral::Int(val),
                PrimitiveLiteral::Int(other_val),
                PrimitiveType::Date,
                PrimitiveType::Date,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Long(val),
                PrimitiveLiteral::Long(other_val),
                PrimitiveType::Time,
                PrimitiveType::Time,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Long(val),
                PrimitiveLiteral::Long(other_val),
                PrimitiveType::Timestamp,
                PrimitiveType::Timestamp,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Long(val),
                PrimitiveLiteral::Long(other_val),
                PrimitiveType::Timestamptz,
                PrimitiveType::Timestamptz,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Long(val),
                PrimitiveLiteral::Long(other_val),
                PrimitiveType::TimestampNs,
                PrimitiveType::TimestampNs,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Long(val),
                PrimitiveLiteral::Long(other_val),
                PrimitiveType::TimestamptzNs,
                PrimitiveType::TimestamptzNs,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::String(val),
                PrimitiveLiteral::String(other_val),
                PrimitiveType::String,
                PrimitiveType::String,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::UInt128(val),
                PrimitiveLiteral::UInt128(other_val),
                PrimitiveType::Uuid,
                PrimitiveType::Uuid,
            ) => uuid::Uuid::from_u128(*val).partial_cmp(&uuid::Uuid::from_u128(*other_val)),
            (
                PrimitiveLiteral::Binary(val),
                PrimitiveLiteral::Binary(other_val),
                PrimitiveType::Fixed(_),
                PrimitiveType::Fixed(_),
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Binary(val),
                PrimitiveLiteral::Binary(other_val),
                PrimitiveType::Binary,
                PrimitiveType::Binary,
            ) => val.partial_cmp(other_val),
            (
                PrimitiveLiteral::Int128(val),
                PrimitiveLiteral::Int128(other_val),
                PrimitiveType::Decimal {
                    precision: _,
                    scale,
                },
                PrimitiveType::Decimal {
                    precision: _,
                    scale: other_scale,
                },
            ) => {
                let val = decimal_from_i128_with_scale(*val, *scale);
                let other_val = decimal_from_i128_with_scale(*other_val, *other_scale);
                val.partial_cmp(&other_val)
            }
            _ => None,
        }
    }
}

impl Display for Datum {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match (&self.r#type, &self.literal) {
            (_, PrimitiveLiteral::Boolean(val)) => write!(f, "{val}"),
            (PrimitiveType::Int, PrimitiveLiteral::Int(val)) => write!(f, "{val}"),
            (PrimitiveType::Long, PrimitiveLiteral::Long(val)) => write!(f, "{val}"),
            (_, PrimitiveLiteral::Float(val)) => write!(f, "{val}"),
            (_, PrimitiveLiteral::Double(val)) => write!(f, "{val}"),
            (PrimitiveType::Date, PrimitiveLiteral::Int(val)) => match date::days_to_date(*val) {
                Some(date) => write!(f, "{date}"),
                None => write!(f, "<invalid date: {val}>"),
            },
            // Formatting must never panic on out-of-range on-disk bytes, so render a placeholder.
            (PrimitiveType::Time, PrimitiveLiteral::Long(val)) => {
                match time::microseconds_to_time(*val) {
                    Some(time) => write!(f, "{time}"),
                    None => write!(f, "<invalid time: {val}>"),
                }
            }
            (PrimitiveType::Timestamp, PrimitiveLiteral::Long(val)) => {
                match timestamp::microseconds_to_datetime(*val) {
                    Some(ts) => write!(f, "{ts}"),
                    None => write!(f, "<invalid timestamp: {val}>"),
                }
            }
            (PrimitiveType::Timestamptz, PrimitiveLiteral::Long(val)) => {
                match timestamptz::microseconds_to_datetimetz(*val) {
                    Some(ts) => write!(f, "{ts}"),
                    None => write!(f, "<invalid timestamptz: {val}>"),
                }
            }
            (PrimitiveType::TimestampNs, PrimitiveLiteral::Long(val)) => {
                write!(f, "{}", timestamp::nanoseconds_to_datetime(*val))
            }
            (PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(val)) => {
                match timestamptz::nanoseconds_to_datetimetz(*val) {
                    Some(ts) => write!(f, "{ts}"),
                    None => write!(f, "<invalid timestamptz_ns: {val}>"),
                }
            }
            (_, PrimitiveLiteral::String(val)) => write!(f, r#""{val}""#),
            (PrimitiveType::Uuid, PrimitiveLiteral::UInt128(val)) => {
                write!(f, "{}", uuid::Uuid::from_u128(*val))
            }
            (_, PrimitiveLiteral::Binary(val)) => display_bytes(val, f),
            (
                PrimitiveType::Decimal {
                    precision: _,
                    scale,
                },
                PrimitiveLiteral::Int128(val),
            ) => {
                write!(f, "{}", decimal_from_i128_with_scale(*val, *scale))
            }
            (_, _) => {
                unreachable!()
            }
        }
    }
}

fn display_bytes(bytes: &[u8], f: &mut Formatter<'_>) -> std::fmt::Result {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02X}"));
    }
    f.write_str(&s)
}

impl From<Datum> for Literal {
    fn from(value: Datum) -> Self {
        Literal::Primitive(value.literal)
    }
}

impl From<Datum> for PrimitiveLiteral {
    fn from(value: Datum) -> Self {
        value.literal
    }
}

impl Datum {
    pub(crate) fn validate_decimal(&self) -> Result<()> {
        validate_decimal_literal(&self.r#type, &self.literal)
    }

    pub(crate) fn new(r#type: PrimitiveType, literal: PrimitiveLiteral) -> Self {
        Datum { r#type, literal }
    }

    /// Create an iceberg value from its [single-value binary encoding](https://iceberg.apache.org/spec/#binary-single-value-serialization).
    pub fn try_from_bytes(bytes: &[u8], data_type: PrimitiveType) -> Result<Self> {
        let literal = match data_type {
            PrimitiveType::Boolean => {
                if bytes.len() == 1 && bytes[0] == 0u8 {
                    PrimitiveLiteral::Boolean(false)
                } else {
                    PrimitiveLiteral::Boolean(true)
                }
            }
            PrimitiveType::Int => PrimitiveLiteral::Int(i32::from_le_bytes(bytes.try_into()?)),
            PrimitiveType::Long => {
                if bytes.len() == 4 {
                    // In the case of an evolved field
                    PrimitiveLiteral::Long(i32::from_le_bytes(bytes.try_into()?) as i64)
                } else {
                    PrimitiveLiteral::Long(i64::from_le_bytes(bytes.try_into()?))
                }
            }
            PrimitiveType::Float => {
                PrimitiveLiteral::Float(OrderedFloat(f32::from_le_bytes(bytes.try_into()?)))
            }
            PrimitiveType::Double => {
                if bytes.len() == 4 {
                    // In the case of an evolved field
                    PrimitiveLiteral::Double(OrderedFloat(
                        f32::from_le_bytes(bytes.try_into()?) as f64
                    ))
                } else {
                    PrimitiveLiteral::Double(OrderedFloat(f64::from_le_bytes(bytes.try_into()?)))
                }
            }
            PrimitiveType::Date => PrimitiveLiteral::Int(i32::from_le_bytes(bytes.try_into()?)),
            PrimitiveType::Time => PrimitiveLiteral::Long(i64::from_le_bytes(bytes.try_into()?)),
            PrimitiveType::Timestamp => {
                PrimitiveLiteral::Long(i64::from_le_bytes(bytes.try_into()?))
            }
            PrimitiveType::Timestamptz => {
                PrimitiveLiteral::Long(i64::from_le_bytes(bytes.try_into()?))
            }
            PrimitiveType::TimestampNs => {
                PrimitiveLiteral::Long(i64::from_le_bytes(bytes.try_into()?))
            }
            PrimitiveType::TimestamptzNs => {
                PrimitiveLiteral::Long(i64::from_le_bytes(bytes.try_into()?))
            }
            PrimitiveType::String => {
                PrimitiveLiteral::String(std::str::from_utf8(bytes)?.to_string())
            }
            PrimitiveType::Uuid => {
                PrimitiveLiteral::UInt128(u128::from_be_bytes(bytes.try_into()?))
            }
            PrimitiveType::Fixed(_) => PrimitiveLiteral::Binary(Vec::from(bytes)),
            PrimitiveType::Binary => PrimitiveLiteral::Binary(Vec::from(bytes)),
            PrimitiveType::Decimal { precision, .. } => {
                // Java's DECIMAL arm has no length check and no minimality check, so padded,
                // sign-extended, and over-precision buffers all decode. One rejected bound aborts
                // every scan. Java does reject an empty buffer and `precision > 38`, so these do.
                ensure_java_decimal_precision(precision)?;
                ensure_data_valid!(
                    !bytes.is_empty(),
                    "Zero length BigInteger: a decimal value must have at least one byte",
                );
                let value = i128_from_be_bytes(bytes).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Can't convert bytes to i128: {bytes:?}"),
                    )
                })?;
                PrimitiveLiteral::Int128(value)
            }
            // `unknown` values are always null, so there is no byte encoding to decode.
            PrimitiveType::Unknown => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot deserialize a value of the unknown type: unknown is always null and has no single-value encoding",
                ));
            }
        };
        Ok(Datum::new(data_type, literal))
    }

    /// Convert the value to its [single-value binary encoding](https://iceberg.apache.org/spec/#binary-single-value-serialization).
    pub fn to_bytes(&self) -> Result<ByteBuf> {
        // Keep the original decimal encoder's message for an invalid precision.
        if matches!(&self.literal, PrimitiveLiteral::Int128(_))
            && let PrimitiveType::Decimal { precision, .. } = &self.r#type
            && (*precision == 0 || *precision > MAX_DECIMAL_PRECISION)
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("PrimitiveType Decimal must has valid precision but got {precision}"),
            ));
        }
        validate_decimal_literal(&self.r#type, &self.literal)?;

        let buf = match &self.literal {
            PrimitiveLiteral::Boolean(val) => {
                if *val {
                    ByteBuf::from([1u8])
                } else {
                    ByteBuf::from([0u8])
                }
            }
            PrimitiveLiteral::Int(val) => ByteBuf::from(val.to_le_bytes()),
            PrimitiveLiteral::Long(val) => ByteBuf::from(val.to_le_bytes()),
            PrimitiveLiteral::Float(val) => ByteBuf::from(val.to_le_bytes()),
            PrimitiveLiteral::Double(val) => ByteBuf::from(val.to_le_bytes()),
            PrimitiveLiteral::String(val) => ByteBuf::from(val.as_bytes()),
            PrimitiveLiteral::UInt128(val) => ByteBuf::from(val.to_be_bytes()),
            PrimitiveLiteral::Binary(val) => ByteBuf::from(val.as_slice()),
            PrimitiveLiteral::Int128(val) => {
                let PrimitiveType::Decimal { precision, .. } = self.r#type else {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "PrimitiveLiteral Int128 must be PrimitiveType Decimal but got {}",
                            &self.r#type
                        ),
                    ));
                };

                // The spec requires the minimum number of bytes for the value.
                let required_bytes = Type::decimal_required_bytes(precision)?;

                // The literal is the unscaled value. Emit two's complement, big-endian.
                let mut bytes = i128_to_be_bytes_min(*val);
                bytes.truncate(required_bytes.try_into()?);

                ByteBuf::from(bytes)
            }
            PrimitiveLiteral::AboveMax | PrimitiveLiteral::BelowMin => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot convert AboveMax or BelowMin to bytes".to_string(),
                ));
            }
        };

        Ok(buf)
    }

    /// Creates a boolean value.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal, PrimitiveLiteral};
    /// let t = Datum::bool(true);
    ///
    /// assert_eq!(format!("{}", t), "true".to_string());
    /// assert_eq!(
    ///     Literal::from(t),
    ///     Literal::Primitive(PrimitiveLiteral::Boolean(true))
    /// );
    /// ```
    pub fn bool<T: Into<bool>>(t: T) -> Self {
        Self {
            r#type: PrimitiveType::Boolean,
            literal: PrimitiveLiteral::Boolean(t.into()),
        }
    }

    /// Creates a boolean value from a string, via `bool`'s `FromStr`.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal, PrimitiveLiteral};
    /// let t = Datum::bool_from_str("false").unwrap();
    ///
    /// assert_eq!(&format!("{}", t), "false");
    /// assert_eq!(
    ///     Literal::Primitive(PrimitiveLiteral::Boolean(false)),
    ///     t.into()
    /// );
    /// ```
    pub fn bool_from_str<S: AsRef<str>>(s: S) -> Result<Self> {
        let v = s.as_ref().parse::<bool>().map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Can't parse string to bool.").with_source(e)
        })?;
        Ok(Self::bool(v))
    }

    /// Creates an 32bit integer.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal, PrimitiveLiteral};
    /// let t = Datum::int(23i8);
    ///
    /// assert_eq!(&format!("{}", t), "23");
    /// assert_eq!(Literal::Primitive(PrimitiveLiteral::Int(23)), t.into());
    /// ```
    pub fn int<T: Into<i32>>(t: T) -> Self {
        Self {
            r#type: PrimitiveType::Int,
            literal: PrimitiveLiteral::Int(t.into()),
        }
    }

    /// Creates an 64bit integer.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal, PrimitiveLiteral};
    /// let t = Datum::long(24i8);
    ///
    /// assert_eq!(&format!("{t}"), "24");
    /// assert_eq!(Literal::Primitive(PrimitiveLiteral::Long(24)), t.into());
    /// ```
    pub fn long<T: Into<i64>>(t: T) -> Self {
        Self {
            r#type: PrimitiveType::Long,
            literal: PrimitiveLiteral::Long(t.into()),
        }
    }

    /// Creates an 32bit floating point number.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal, PrimitiveLiteral};
    /// use ordered_float::OrderedFloat;
    /// let t = Datum::float(32.1f32);
    ///
    /// assert_eq!(&format!("{t}"), "32.1");
    /// assert_eq!(
    ///     Literal::Primitive(PrimitiveLiteral::Float(OrderedFloat(32.1))),
    ///     t.into()
    /// );
    /// ```
    pub fn float<T: Into<f32>>(t: T) -> Self {
        Self {
            r#type: PrimitiveType::Float,
            literal: PrimitiveLiteral::Float(OrderedFloat(t.into())),
        }
    }

    /// Creates an 64bit floating point number.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal, PrimitiveLiteral};
    /// use ordered_float::OrderedFloat;
    /// let t = Datum::double(32.1f64);
    ///
    /// assert_eq!(&format!("{t}"), "32.1");
    /// assert_eq!(
    ///     Literal::Primitive(PrimitiveLiteral::Double(OrderedFloat(32.1))),
    ///     t.into()
    /// );
    /// ```
    pub fn double<T: Into<f64>>(t: T) -> Self {
        Self {
            r#type: PrimitiveType::Double,
            literal: PrimitiveLiteral::Double(OrderedFloat(t.into())),
        }
    }

    /// Creates date literal from number of days from unix epoch directly.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal, PrimitiveLiteral};
    /// // 2 days after 1970-01-01
    /// let t = Datum::date(2);
    ///
    /// assert_eq!(&format!("{t}"), "1970-01-03");
    /// assert_eq!(Literal::Primitive(PrimitiveLiteral::Int(2)), t.into());
    /// ```
    pub fn date(days: i32) -> Self {
        Self {
            r#type: PrimitiveType::Date,
            literal: PrimitiveLiteral::Int(days),
        }
    }

    /// Creates a date literal in `%Y-%m-%d` format, in UTC. See [`NaiveDate::from_str`].
    /// ```rust
    /// use iceberg::spec::{Datum, Literal};
    /// let t = Datum::date_from_str("1970-01-05").unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "1970-01-05");
    /// assert_eq!(Literal::date(4), t.into());
    /// ```
    pub fn date_from_str<S: AsRef<str>>(s: S) -> Result<Self> {
        let t = s.as_ref().parse::<NaiveDate>().map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Can't parse date from string: {}", s.as_ref()),
            )
            .with_source(e)
        })?;

        Ok(Self::date(date::date_from_naive_date(t)))
    }

    /// Creates a date literal from a calendar date. See [`NaiveDate::from_ymd_opt`].
    ///```rust
    /// use iceberg::spec::{Datum, Literal};
    /// let t = Datum::date_from_ymd(1970, 1, 5).unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "1970-01-05");
    /// assert_eq!(Literal::date(4), t.into());
    /// ```
    pub fn date_from_ymd(year: i32, month: u32, day: u32) -> Result<Self> {
        let t = NaiveDate::from_ymd_opt(year, month, day).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Can't create date from year: {year}, month: {month}, day: {day}"),
            )
        })?;

        Ok(Self::date(date::date_from_naive_date(t)))
    }

    /// Creates a time literal from microseconds. Fails if the value is negative or past 24 hours.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal};
    /// let micro_secs = {
    ///     1 * 3600 * 1_000_000 + // 1 hour
    ///     2 * 60 * 1_000_000 +   // 2 minutes
    ///     1 * 1_000_000 + // 1 second
    ///     888999 // microseconds
    /// };
    ///
    /// let t = Datum::time_micros(micro_secs).unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "01:02:01.888999");
    /// assert_eq!(Literal::time(micro_secs), t.into());
    ///
    /// let negative_value = -100;
    /// assert!(Datum::time_micros(negative_value).is_err());
    ///
    /// let too_large_value = 36 * 60 * 60 * 1_000_000; // Too large to fit in 24 hours.
    /// assert!(Datum::time_micros(too_large_value).is_err());
    /// ```
    pub fn time_micros(value: i64) -> Result<Self> {
        ensure_data_valid!(
            (0..=MAX_TIME_VALUE).contains(&value),
            "Invalid value for Time type: {}",
            value
        );

        Ok(Self {
            r#type: PrimitiveType::Time,
            literal: PrimitiveLiteral::Long(value),
        })
    }

    /// Creates time literal from [`chrono::NaiveTime`].
    fn time_from_naive_time(t: NaiveTime) -> Self {
        let duration = t - date::unix_epoch().time();
        // A span under 24 hours always fits in microseconds, so this cannot overflow.
        let micro_secs = duration.num_microseconds().unwrap();

        Self {
            r#type: PrimitiveType::Time,
            literal: PrimitiveLiteral::Long(micro_secs),
        }
    }

    /// Creates a time literal from a `%H:%M:%S%.f` string. See [`NaiveTime::from_str`].
    /// ```rust
    /// use iceberg::spec::{Datum, Literal};
    /// let t = Datum::time_from_str("01:02:01.888999777").unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "01:02:01.888999");
    /// ```
    pub fn time_from_str<S: AsRef<str>>(s: S) -> Result<Self> {
        let t = s.as_ref().parse::<NaiveTime>().map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Can't parse time from string: {}", s.as_ref()),
            )
            .with_source(e)
        })?;

        Ok(Self::time_from_naive_time(t))
    }

    /// Creates a time literal from hour, minute, second, and microsecond.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal};
    /// let t = Datum::time_from_hms_micro(22, 15, 33, 111).unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "22:15:33.000111");
    /// ```
    pub fn time_from_hms_micro(hour: u32, min: u32, sec: u32, micro: u32) -> Result<Self> {
        let t = NaiveTime::from_hms_micro_opt(hour, min, sec, micro)
            .ok_or_else(|| Error::new(
                ErrorKind::DataInvalid,
                format!("Can't create time from hour: {hour}, min: {min}, second: {sec}, microsecond: {micro}"),
            ))?;
        Ok(Self::time_from_naive_time(t))
    }

    /// Creates a timestamp from unix epoch in microseconds.
    /// ```rust
    /// use iceberg::spec::Datum;
    /// let t = Datum::timestamp_micros(1000);
    ///
    /// assert_eq!(&format!("{t}"), "1970-01-01 00:00:00.001");
    /// ```
    pub fn timestamp_micros(value: i64) -> Self {
        Self {
            r#type: PrimitiveType::Timestamp,
            literal: PrimitiveLiteral::Long(value),
        }
    }

    /// Creates a timestamp from unix epoch in nanoseconds.
    /// ```rust
    /// use iceberg::spec::Datum;
    /// let t = Datum::timestamp_nanos(1000);
    ///
    /// assert_eq!(&format!("{t}"), "1970-01-01 00:00:00.000001");
    /// ```
    pub fn timestamp_nanos(value: i64) -> Self {
        Self {
            r#type: PrimitiveType::TimestampNs,
            literal: PrimitiveLiteral::Long(value),
        }
    }

    /// Creates a timestamp from [`DateTime`].
    /// ```rust
    /// use chrono::{NaiveDate, NaiveDateTime, TimeZone, Utc};
    /// use iceberg::spec::Datum;
    /// let t = Datum::timestamp_from_datetime(
    ///     NaiveDate::from_ymd_opt(1992, 3, 1)
    ///         .unwrap()
    ///         .and_hms_micro_opt(1, 2, 3, 88)
    ///         .unwrap(),
    /// );
    ///
    /// assert_eq!(&format!("{t}"), "1992-03-01 01:02:03.000088");
    /// ```
    pub fn timestamp_from_datetime(dt: NaiveDateTime) -> Self {
        Self::timestamp_micros(dt.and_utc().timestamp_micros())
    }

    /// Parse a timestamp in `%Y-%m-%dT%H:%M:%S%.f` format. See [`NaiveDateTime::from_str`].
    /// ```rust
    /// use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime};
    /// use iceberg::spec::{Datum, Literal};
    /// let t = Datum::timestamp_from_str("1992-03-01T01:02:03.000088").unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "1992-03-01 01:02:03.000088");
    /// ```
    pub fn timestamp_from_str<S: AsRef<str>>(s: S) -> Result<Self> {
        let dt = s.as_ref().parse::<NaiveDateTime>().map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Can't parse timestamp.").with_source(e)
        })?;

        Ok(Self::timestamp_from_datetime(dt))
    }

    /// Creates a timestamp with timezone from unix epoch in microseconds.
    /// ```rust
    /// use iceberg::spec::Datum;
    /// let t = Datum::timestamptz_micros(1000);
    ///
    /// assert_eq!(&format!("{t}"), "1970-01-01 00:00:00.001 UTC");
    /// ```
    pub fn timestamptz_micros(value: i64) -> Self {
        Self {
            r#type: PrimitiveType::Timestamptz,
            literal: PrimitiveLiteral::Long(value),
        }
    }

    /// Creates a timestamp with timezone from unix epoch in nanoseconds.
    /// ```rust
    /// use iceberg::spec::Datum;
    /// let t = Datum::timestamptz_nanos(1000);
    ///
    /// assert_eq!(&format!("{t}"), "1970-01-01 00:00:00.000001 UTC");
    /// ```
    pub fn timestamptz_nanos(value: i64) -> Self {
        Self {
            r#type: PrimitiveType::TimestamptzNs,
            literal: PrimitiveLiteral::Long(value),
        }
    }

    /// Creates a timestamp with timezone from [`DateTime`].
    /// ```rust
    /// use chrono::{TimeZone, Utc};
    /// use iceberg::spec::Datum;
    /// let t = Datum::timestamptz_from_datetime(Utc.timestamp_opt(1000, 0).unwrap());
    ///
    /// assert_eq!(&format!("{t}"), "1970-01-01 00:16:40 UTC");
    /// ```
    pub fn timestamptz_from_datetime<T: TimeZone>(dt: DateTime<T>) -> Self {
        Self::timestamptz_micros(dt.with_timezone(&Utc).timestamp_micros())
    }

    /// Parse a timestamp with timezone in RFC3339 format. See [`DateTime::from_str`].
    /// ```rust
    /// use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime};
    /// use iceberg::spec::{Datum, Literal};
    /// let t = Datum::timestamptz_from_str("1992-03-01T01:02:03.000088+08:00").unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "1992-02-29 17:02:03.000088 UTC");
    /// ```
    pub fn timestamptz_from_str<S: AsRef<str>>(s: S) -> Result<Self> {
        let dt = DateTime::<Utc>::from_str(s.as_ref()).map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Can't parse datetime.").with_source(e)
        })?;

        Ok(Self::timestamptz_from_datetime(dt))
    }

    /// Creates a string literal.
    /// ```rust
    /// use iceberg::spec::Datum;
    /// let t = Datum::string("ss");
    ///
    /// assert_eq!(&format!("{t}"), r#""ss""#);
    /// ```
    pub fn string<S: ToString>(s: S) -> Self {
        Self {
            r#type: PrimitiveType::String,
            literal: PrimitiveLiteral::String(s.to_string()),
        }
    }

    /// Creates uuid literal.
    /// ```rust
    /// use iceberg::spec::Datum;
    /// use uuid::uuid;
    /// let t = Datum::uuid(uuid!("a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8"));
    ///
    /// assert_eq!(&format!("{t}"), "a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8");
    /// ```
    pub fn uuid(uuid: uuid::Uuid) -> Self {
        Self {
            r#type: PrimitiveType::Uuid,
            literal: PrimitiveLiteral::UInt128(uuid.as_u128()),
        }
    }

    /// Creates uuid from str. See [`uuid::Uuid::parse_str`].
    /// ```rust
    /// use iceberg::spec::Datum;
    /// let t = Datum::uuid_from_str("a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8").unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "a1a2a3a4-b1b2-c1c2-d1d2-d3d4d5d6d7d8");
    /// ```
    pub fn uuid_from_str<S: AsRef<str>>(s: S) -> Result<Self> {
        let uuid = uuid::Uuid::parse_str(s.as_ref()).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Can't parse uuid from string: {}", s.as_ref()),
            )
            .with_source(e)
        })?;
        Ok(Self::uuid(uuid))
    }

    /// Creates a fixed literal from bytes.
    /// ```rust
    /// use iceberg::spec::{Datum, Literal, PrimitiveLiteral};
    /// let t = Datum::fixed(vec![1u8, 2u8]);
    ///
    /// assert_eq!(&format!("{t}"), "0102");
    /// ```
    pub fn fixed<I: IntoIterator<Item = u8>>(input: I) -> Self {
        let value: Vec<u8> = input.into_iter().collect();
        Self {
            r#type: PrimitiveType::Fixed(value.len() as u64),
            literal: PrimitiveLiteral::Binary(value),
        }
    }

    /// Creates a binary literal from bytes.
    /// ```rust
    /// use iceberg::spec::Datum;
    /// let t = Datum::binary(vec![1u8, 100u8]);
    ///
    /// assert_eq!(&format!("{t}"), "0164");
    /// ```
    pub fn binary<I: IntoIterator<Item = u8>>(input: I) -> Self {
        Self {
            r#type: PrimitiveType::Binary,
            literal: PrimitiveLiteral::Binary(input.into_iter().collect()),
        }
    }

    /// Creates decimal literal from string.
    /// ```rust
    /// use iceberg::spec::Datum;
    /// let t = Datum::decimal_from_str("123.45").unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "123.45");
    /// ```
    pub fn decimal_from_str<S: AsRef<str>>(s: S) -> Result<Self> {
        let decimal = decimal_from_str_exact(s.as_ref())?;

        Self::decimal(decimal)
    }

    /// Try to create a decimal literal from [`Decimal`].
    /// ```rust
    /// use iceberg::spec::Datum;
    ///
    /// let t = Datum::decimal_from_str("1.23").unwrap();
    ///
    /// assert_eq!(&format!("{t}"), "1.23");
    /// ```
    pub fn decimal(value: Decimal) -> Result<Self> {
        let scale = decimal_scale(&value);

        let r#type = Type::decimal(MAX_DECIMAL_PRECISION, scale)?;
        if let Type::Primitive(p) = r#type {
            Ok(Self {
                r#type: p,
                literal: PrimitiveLiteral::Int128(decimal_mantissa(&value)),
            })
        } else {
            unreachable!("Decimal type must be primitive.")
        }
    }

    /// Try to create a decimal literal from [`Decimal`]. [`Datum::decimal`] uses precision 38.
    pub fn decimal_with_precision(value: Decimal, precision: u32) -> Result<Self> {
        let scale = decimal_scale(&value);
        let mantissa = decimal_mantissa(&value);

        // Metadata before value: `scale > precision` must fail even when the value fits a byte.
        validate_decimal_value(&PrimitiveType::Decimal { precision, scale }, mantissa)?;

        let available_bytes = usize::try_from(Type::decimal_required_bytes(precision)?)?;
        let actual_bytes = i128_to_be_bytes_min(mantissa);
        if actual_bytes.len() > available_bytes {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Decimal value {value} is too large for precision {precision}"),
            ));
        }

        let r#type = Type::decimal(precision, scale)?;
        if let Type::Primitive(p) = r#type {
            Ok(Self {
                r#type: p,
                literal: PrimitiveLiteral::Int128(mantissa),
            })
        } else {
            unreachable!("Decimal type must be primitive.")
        }
    }

    fn i64_to_i32<T: Into<i64> + PartialOrd<i64>>(val: T) -> Datum {
        if val > INT_MAX as i64 {
            Datum::new(PrimitiveType::Int, PrimitiveLiteral::AboveMax)
        } else if val < INT_MIN as i64 {
            Datum::new(PrimitiveType::Int, PrimitiveLiteral::BelowMin)
        } else {
            Datum::int(val.into() as i32)
        }
    }

    /// Convert the datum to `target_type`.
    pub fn to(self, target_type: &Type) -> Result<Datum> {
        match target_type {
            Type::Primitive(target_primitive_type) => {
                match (&self.literal, &self.r#type, target_primitive_type) {
                    (PrimitiveLiteral::Int(val), _, PrimitiveType::Int) => Ok(Datum::int(*val)),
                    (PrimitiveLiteral::Int(val), _, PrimitiveType::Date) => Ok(Datum::date(*val)),
                    (PrimitiveLiteral::Int(val), _, PrimitiveType::Long) => Ok(Datum::long(*val)),
                    (PrimitiveLiteral::Long(val), _, PrimitiveType::Int) => {
                        Ok(Datum::i64_to_i32(*val))
                    }
                    (PrimitiveLiteral::Long(val), _, PrimitiveType::Timestamp) => {
                        Ok(Datum::timestamp_micros(*val))
                    }
                    (PrimitiveLiteral::Long(val), _, PrimitiveType::Timestamptz) => {
                        Ok(Datum::timestamptz_micros(*val))
                    }

                    // Java `DecimalLiteral.to` accepts DECIMAL only, and `StringLiteral.to`
                    // rejects BOOLEAN, INTEGER, and LONG. Those must reach the catch-all `Err`.
                    (PrimitiveLiteral::String(val), _, PrimitiveType::Timestamp) => {
                        Datum::timestamp_from_str(val)
                    }
                    (PrimitiveLiteral::String(val), _, PrimitiveType::Timestamptz) => {
                        Datum::timestamptz_from_str(val)
                    }

                    // The arms below port the Java `Literals.*Literal.to(Type)` accept-set. Still
                    // unported: `-> decimal`, `timestamp[tz] -> date`, `long -> time`, every
                    // `timestamp_nano` arm, and `string -> {fixed,binary}`.

                    // IntegerLiteral.to: FLOAT / DOUBLE
                    (PrimitiveLiteral::Int(val), _, PrimitiveType::Float) => {
                        Ok(Datum::float(*val as f32))
                    }
                    (PrimitiveLiteral::Int(val), _, PrimitiveType::Double) => {
                        Ok(Datum::double(*val))
                    }

                    // LongLiteral.to: FLOAT / DOUBLE / DATE. DATE sentinels outside int range.
                    (PrimitiveLiteral::Long(val), _, PrimitiveType::Float) => {
                        Ok(Datum::float(*val as f32))
                    }
                    (PrimitiveLiteral::Long(val), _, PrimitiveType::Double) => {
                        Ok(Datum::double(*val as f64))
                    }
                    (PrimitiveLiteral::Long(val), _, PrimitiveType::Date) => {
                        Ok(if *val > INT_MAX as i64 {
                            Datum::new(PrimitiveType::Date, PrimitiveLiteral::AboveMax)
                        } else if *val < INT_MIN as i64 {
                            Datum::new(PrimitiveType::Date, PrimitiveLiteral::BelowMin)
                        } else {
                            Datum::date(*val as i32)
                        })
                    }

                    // FloatLiteral.to: DOUBLE
                    (PrimitiveLiteral::Float(val), _, PrimitiveType::Double) => {
                        Ok(Datum::double(f64::from(val.0)))
                    }

                    // DoubleLiteral.to: FLOAT (bounds-checked to ±Float.MAX_VALUE)
                    (PrimitiveLiteral::Double(val), _, PrimitiveType::Float) => {
                        Ok(if val.0 > f32::MAX as f64 {
                            Datum::new(PrimitiveType::Float, PrimitiveLiteral::AboveMax)
                        } else if val.0 < -(f32::MAX as f64) {
                            Datum::new(PrimitiveType::Float, PrimitiveLiteral::BelowMin)
                        } else {
                            Datum::float(val.0 as f32)
                        })
                    }

                    // DecimalLiteral.to: DECIMAL. Java returns `this`, ignoring the target scale.
                    (
                        PrimitiveLiteral::Int128(_),
                        PrimitiveType::Decimal { .. },
                        PrimitiveType::Decimal { .. },
                    ) => Ok(self),

                    // FixedLiteral.to: BINARY / BinaryLiteral.to: FIXED. A length mismatch rejects.
                    (
                        PrimitiveLiteral::Binary(val),
                        PrimitiveType::Fixed(_),
                        PrimitiveType::Binary,
                    ) => Ok(Datum::binary(val.clone())),
                    (
                        PrimitiveLiteral::Binary(val),
                        PrimitiveType::Binary,
                        PrimitiveType::Fixed(len),
                    ) => {
                        if val.len() as u64 == *len {
                            Ok(Datum::fixed(val.clone()))
                        } else {
                            Err(Error::new(
                                ErrorKind::DataInvalid,
                                format!(
                                    "Can't convert binary of {} bytes to fixed[{len}].",
                                    val.len()
                                ),
                            ))
                        }
                    }

                    // StringLiteral.to: UUID / DATE / TIME. A parse failure mirrors Java's `null`.
                    (PrimitiveLiteral::String(val), _, PrimitiveType::Uuid) => {
                        Datum::uuid_from_str(val)
                    }
                    (PrimitiveLiteral::String(val), _, PrimitiveType::Date) => {
                        Datum::date_from_str(val)
                    }
                    (PrimitiveLiteral::String(val), _, PrimitiveType::Time) => {
                        Datum::time_from_str(val)
                    }

                    // Identity (same primitive type).
                    (_, self_type, target_type) if self_type == target_type => Ok(self),
                    _ => Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Can't convert datum from {} type to {} type.",
                            self.r#type, target_primitive_type
                        ),
                    )),
                }
            }
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Can't convert datum from {} type to {} type.",
                    self.r#type, target_type
                ),
            )),
        }
    }

    /// Get the primitive literal from datum.
    pub fn literal(&self) -> &PrimitiveLiteral {
        &self.literal
    }

    /// Get the primitive type from datum.
    pub fn data_type(&self) -> &PrimitiveType {
        &self.r#type
    }

    /// Returns true if the literal is a float or double whose value is NaN.
    pub fn is_nan(&self) -> bool {
        match self.literal {
            PrimitiveLiteral::Double(val) => val.is_nan(),
            PrimitiveLiteral::Float(val) => val.is_nan(),
            _ => false,
        }
    }

    /// Java `Transform.toHumanString`. Float/double use `Float`/`Double.toString`.
    pub fn to_human_string(&self) -> String {
        match self.literal() {
            PrimitiveLiteral::String(s) => s.to_string(),
            PrimitiveLiteral::Binary(bytes)
                if matches!(self.r#type, PrimitiveType::Binary | PrimitiveType::Fixed(_)) =>
            {
                use base64::Engine as _;
                base64::engine::general_purpose::STANDARD.encode(bytes)
            }
            PrimitiveLiteral::Float(v) => super::java_float::java_to_string_f32(v.0),
            PrimitiveLiteral::Double(v) => super::java_float::java_to_string_f64(v.0),
            _ => self.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A temporal `Datum` can hold an out-of-range value read from corrupt on-disk bytes, and
    // `Display` must render a placeholder rather than panic. `Datum::new` bypasses the checks.

    #[test]
    fn test_display_time_out_of_range_does_not_panic() {
        // A negative microsecond-of-day wrapped through `as u32` and unwrapped a `None`.
        let negative = Datum::new(PrimitiveType::Time, PrimitiveLiteral::Long(-1));
        assert_eq!(negative.to_string(), "<invalid time: -1>");

        // One microsecond past 24 hours.
        let past_midnight = Datum::new(PrimitiveType::Time, PrimitiveLiteral::Long(86_400_000_001));
        assert_eq!(past_midnight.to_string(), "<invalid time: 86400000001>");
    }

    #[test]
    fn test_display_valid_time_unchanged() {
        // A valid value must still render normally.
        let valid = Datum::new(PrimitiveType::Time, PrimitiveLiteral::Long(1_000_000));
        assert_eq!(valid.to_string(), "00:00:01");
    }

    // An out-of-range days-since-epoch `Date` must render a placeholder, not panic. The mutation
    // this discriminates: make `days_to_date` return a bare `NaiveDate` from
    // `UNIX_EPOCH + TimeDelta::try_days(d).unwrap()` and format it with a bare `write!`. The
    // chrono `Add` then panics here on an extreme `i32`.
    #[test]
    fn test_display_date_out_of_range_does_not_panic() {
        let max = Datum::new(PrimitiveType::Date, PrimitiveLiteral::Int(i32::MAX));
        assert_eq!(max.to_string(), format!("<invalid date: {}>", i32::MAX));

        let min = Datum::new(PrimitiveType::Date, PrimitiveLiteral::Int(i32::MIN));
        assert_eq!(min.to_string(), format!("<invalid date: {}>", i32::MIN));
    }

    #[test]
    fn test_display_valid_date_unchanged() {
        // 19_723 days is 2024-01-01. A valid value must still render normally.
        let valid = Datum::new(PrimitiveType::Date, PrimitiveLiteral::Int(19_723));
        assert_eq!(valid.to_string(), "2024-01-01");
    }

    #[test]
    fn test_display_timestamp_out_of_range_does_not_panic() {
        // i64::MAX microseconds is past chrono's range.
        let datum = Datum::new(PrimitiveType::Timestamp, PrimitiveLiteral::Long(i64::MAX));
        let rendered = datum.to_string();
        assert_eq!(rendered, format!("<invalid timestamp: {}>", i64::MAX));
    }

    #[test]
    fn test_display_timestamptz_out_of_range_does_not_panic() {
        // The negative sub-second remainder wrapped through `as u32` and the unwrap panicked.
        let datum = Datum::new(PrimitiveType::Timestamptz, PrimitiveLiteral::Long(i64::MIN));
        let rendered = datum.to_string();
        assert_eq!(rendered, format!("<invalid timestamptz: {}>", i64::MIN));
    }

    #[test]
    fn test_display_timestamptz_ns_negative_remainder_does_not_panic() {
        // A small negative nanosecond value wrapped the sub-second remainder through `as u32`,
        // so `from_timestamp` gave `None` and the unwrap panicked. Every i64 nanosecond value is
        // inside chrono's range, so this renders normally and only proves the absence.
        let datum = Datum::new(PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(-1));
        assert_eq!(datum.to_string(), "1969-12-31 23:59:59.999999999 UTC");
    }
}
