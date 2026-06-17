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

/*!
 * Data Types
 */
use std::collections::HashMap;
use std::convert::identity;
use std::fmt;
use std::ops::Index;
use std::sync::{Arc, OnceLock};

use ::serde::de::{MapAccess, Visitor};
use serde::de::{Error, IntoDeserializer};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::Value as JsonValue;

use super::values::Literal;
use crate::ensure_data_valid;
use crate::error::Result;
use crate::spec::PrimitiveLiteral;
use crate::spec::datatypes::_decimal::{MAX_PRECISION, REQUIRED_LENGTH};

/// Field name for list type.
pub const LIST_FIELD_NAME: &str = "element";
/// Field name for map type's key.
pub const MAP_KEY_FIELD_NAME: &str = "key";
/// Field name for map type's value.
pub const MAP_VALUE_FIELD_NAME: &str = "value";

pub(crate) const MAX_DECIMAL_BYTES: u32 = 24;
pub(crate) const MAX_DECIMAL_PRECISION: u32 = 38;

mod _decimal {
    use once_cell::sync::Lazy;

    use crate::spec::{MAX_DECIMAL_BYTES, MAX_DECIMAL_PRECISION};

    // Max precision of bytes, starts from 1
    pub(super) static MAX_PRECISION: Lazy<[u32; MAX_DECIMAL_BYTES as usize]> = Lazy::new(|| {
        let mut ret: [u32; 24] = [0; 24];
        for (i, prec) in ret.iter_mut().enumerate() {
            *prec = 2f64.powi((8 * (i + 1) - 1) as i32).log10().floor() as u32;
        }

        ret
    });

    //  Required bytes of precision, starts from 1
    pub(super) static REQUIRED_LENGTH: Lazy<[u32; MAX_DECIMAL_PRECISION as usize]> =
        Lazy::new(|| {
            let mut ret: [u32; MAX_DECIMAL_PRECISION as usize] =
                [0; MAX_DECIMAL_PRECISION as usize];

            for (i, required_len) in ret.iter_mut().enumerate() {
                for j in 0..MAX_PRECISION.len() {
                    if MAX_PRECISION[j] >= ((i + 1) as u32) {
                        *required_len = (j + 1) as u32;
                        break;
                    }
                }
            }

            ret
        });
}

#[derive(Debug, PartialEq, Eq, Clone)]
/// All data types are either primitives or nested types, which are maps, lists, or structs —
/// plus `variant`, which is its own category.
///
/// `Variant` mirrors Java 1.10.0 `org.apache.iceberg.types.Types.VariantType`, which implements
/// the `Type` interface directly: it is **neither** a `Type.PrimitiveType` **nor** a
/// `Type.NestedType` (`isPrimitiveType()` and `isNestedType()` are both false). Placing it as its
/// own `Type` variant (not a `PrimitiveType`) preserves every Java non-primitive door for free:
/// variant is rejected as a partition source, sort key, and identifier field by the same
/// `is_primitive()` checks Java uses.
pub enum Type {
    /// Primitive types
    Primitive(PrimitiveType),
    /// Struct type
    Struct(StructType),
    /// List type.
    List(ListType),
    /// Map type
    Map(MapType),
    /// Variant type (format version 3+): semi-structured data as a (metadata, value) binary pair.
    ///
    /// The binary encoding lives in [`crate::variant`]; this enum entry is the schema-level type.
    /// Like Java's singleton `Types.VariantType`, it carries no parameters.
    Variant,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Primitive(primitive) => write!(f, "{primitive}"),
            Type::Struct(s) => write!(f, "{s}"),
            Type::List(_) => write!(f, "list"),
            Type::Map(_) => write!(f, "map"),
            // Java `VariantType.toString()` returns exactly "variant".
            Type::Variant => write!(f, "variant"),
        }
    }
}

impl Type {
    /// Whether the type is primitive type.
    #[inline(always)]
    pub fn is_primitive(&self) -> bool {
        matches!(self, Type::Primitive(_))
    }

    /// Whether the type is struct type.
    #[inline(always)]
    pub fn is_struct(&self) -> bool {
        matches!(self, Type::Struct(_))
    }

    /// Whether the type is nested type.
    ///
    /// `variant` is NOT a nested type (Java `Type.isNestedType()` defaults to false and
    /// `VariantType` does not override it).
    #[inline(always)]
    pub fn is_nested(&self) -> bool {
        matches!(self, Type::Struct(_) | Type::List(_) | Type::Map(_))
    }

    /// Whether the type is the variant type.
    ///
    /// Mirrors Java `Type.isVariantType()` (false everywhere except `Types.VariantType`).
    #[inline(always)]
    pub fn is_variant(&self) -> bool {
        matches!(self, Type::Variant)
    }

    /// Convert Type to reference of PrimitiveType
    pub fn as_primitive_type(&self) -> Option<&PrimitiveType> {
        if let Type::Primitive(primitive_type) = self {
            Some(primitive_type)
        } else {
            None
        }
    }

    /// Convert Type to StructType
    pub fn to_struct_type(self) -> Option<StructType> {
        if let Type::Struct(struct_type) = self {
            Some(struct_type)
        } else {
            None
        }
    }

    /// Return max precision for decimal given [`num_bytes`] bytes.
    #[inline(always)]
    pub fn decimal_max_precision(num_bytes: u32) -> Result<u32> {
        ensure_data_valid!(
            num_bytes > 0 && num_bytes <= MAX_DECIMAL_BYTES,
            "Decimal length larger than {MAX_DECIMAL_BYTES} is not supported: {num_bytes}",
        );
        Ok(MAX_PRECISION[num_bytes as usize - 1])
    }

    /// Returns minimum bytes required for decimal with [`precision`].
    #[inline(always)]
    pub fn decimal_required_bytes(precision: u32) -> Result<u32> {
        ensure_data_valid!(
            precision > 0 && precision <= MAX_DECIMAL_PRECISION,
            "Decimals with precision larger than {MAX_DECIMAL_PRECISION} are not supported: {precision}",
        );
        Ok(REQUIRED_LENGTH[precision as usize - 1])
    }

    /// Creates  decimal type.
    #[inline(always)]
    pub fn decimal(precision: u32, scale: u32) -> Result<Self> {
        ensure_data_valid!(
            precision > 0 && precision <= MAX_DECIMAL_PRECISION,
            "Decimals with precision larger than {MAX_DECIMAL_PRECISION} are not supported: {precision}",
        );
        Ok(Type::Primitive(PrimitiveType::Decimal { precision, scale }))
    }

    /// Check if it's float or double type.
    #[inline(always)]
    pub fn is_floating_type(&self) -> bool {
        matches!(
            self,
            Type::Primitive(PrimitiveType::Float) | Type::Primitive(PrimitiveType::Double)
        )
    }
}

impl From<PrimitiveType> for Type {
    fn from(value: PrimitiveType) -> Self {
        Self::Primitive(value)
    }
}

impl From<StructType> for Type {
    fn from(value: StructType) -> Self {
        Type::Struct(value)
    }
}

impl From<ListType> for Type {
    fn from(value: ListType) -> Self {
        Type::List(value)
    }
}

impl From<MapType> for Type {
    fn from(value: MapType) -> Self {
        Type::Map(value)
    }
}

/// Primitive data types
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Hash)]
#[serde(rename_all = "lowercase", remote = "Self")]
pub enum PrimitiveType {
    /// True or False
    Boolean,
    /// 32-bit signed integer
    Int,
    /// 64-bit signed integer
    Long,
    /// 32-bit IEEE 754 floating point.
    Float,
    /// 64-bit IEEE 754 floating point.
    Double,
    /// Fixed point decimal
    Decimal {
        /// Precision, must be 38 or less
        precision: u32,
        /// Scale
        scale: u32,
    },
    /// Calendar date without timezone or time.
    Date,
    /// Time of day in microsecond precision, without date or timezone.
    Time,
    /// Timestamp in microsecond precision, without timezone
    Timestamp,
    /// Timestamp in microsecond precision, with timezone
    Timestamptz,
    /// Timestamp in nanosecond precision, without timezone
    #[serde(rename = "timestamp_ns")]
    TimestampNs,
    /// Timestamp in nanosecond precision with timezone
    #[serde(rename = "timestamptz_ns")]
    TimestamptzNs,
    /// Arbitrary-length character sequences encoded in utf-8
    String,
    /// Universally Unique Identifiers, should use 16-byte fixed
    Uuid,
    /// Fixed length byte array
    Fixed(u64),
    /// Arbitrary-length byte array.
    Binary,
    /// Unknown type (format version 3+): a column whose values are always null and that has no
    /// physical storage.
    ///
    /// Mirrors Java 1.10.0 `org.apache.iceberg.types.Types.UnknownType`, which (unlike
    /// [`Type::Variant`]) **extends `Type.PrimitiveType`** — so it is a Rust `PrimitiveType` arm,
    /// not a top-level [`Type`] variant. It is a singleton (`UnknownType.get()`), its `toString()`
    /// is the bare string `"unknown"` (so the `rename_all = "lowercase"` serde gives the JSON
    /// `"unknown"` for free), and `Schema.MIN_FORMAT_VERSIONS` gates it at format version 3.
    ///
    /// `unknown` has no [`PrimitiveLiteral`](crate::spec::PrimitiveLiteral) form — its values are
    /// always null — so the datum/literal layer rejects it rather than carrying a value, and Java
    /// `TypeToMessageType` returns `null` for it (no physical parquet column). Data-file
    /// always-null write/read I/O is deferred (see the writer/value paths, which fail loudly).
    Unknown,
}

impl PrimitiveType {
    /// Check whether literal is compatible with the type.
    pub fn compatible(&self, literal: &PrimitiveLiteral) -> bool {
        matches!(
            (self, literal),
            (PrimitiveType::Boolean, PrimitiveLiteral::Boolean(_))
                | (PrimitiveType::Int, PrimitiveLiteral::Int(_))
                | (PrimitiveType::Long, PrimitiveLiteral::Long(_))
                | (PrimitiveType::Float, PrimitiveLiteral::Float(_))
                | (PrimitiveType::Double, PrimitiveLiteral::Double(_))
                | (PrimitiveType::Decimal { .. }, PrimitiveLiteral::Int128(_))
                | (PrimitiveType::Date, PrimitiveLiteral::Int(_))
                | (PrimitiveType::Time, PrimitiveLiteral::Long(_))
                | (PrimitiveType::Timestamp, PrimitiveLiteral::Long(_))
                | (PrimitiveType::Timestamptz, PrimitiveLiteral::Long(_))
                | (PrimitiveType::TimestampNs, PrimitiveLiteral::Long(_))
                | (PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(_))
                | (PrimitiveType::String, PrimitiveLiteral::String(_))
                | (PrimitiveType::Uuid, PrimitiveLiteral::UInt128(_))
                | (PrimitiveType::Fixed(_), PrimitiveLiteral::Binary(_))
                | (PrimitiveType::Binary, PrimitiveLiteral::Binary(_))
        )
    }
}

impl Serialize for Type {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where S: Serializer {
        let type_serde = _serde::SerdeType::from(self);
        type_serde.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Type {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: Deserializer<'de> {
        let type_serde = _serde::SerdeType::deserialize(deserializer)?;
        // Java `SchemaParser.typeFromJson` matches the object WRAPPER names with `String.equals`
        // (1.10.0 bytecode, offsets 41/55/69) — CASE-SENSITIVE: `{"type":"STRUCT"/"LIST"/"MAP"}`
        // falls through to `IllegalArgumentException("Cannot parse type from json: ...")`. The
        // untagged [`_serde::SerdeType`] matches a wrapper STRUCTURALLY (by its `fields` /
        // `element` / `key`+`value` shape) and ignores the `type` string, so without this guard
        // Rust would ACCEPT a wrong-cased or wrong wrapper name that Java rejects (a read-leniency
        // divergence found by the O3 REVIEWER). Re-assert Java's exact `String.equals` here so the
        // accepted set matches; Rust's own writer always emits the lowercase name, so this never
        // rejects a self-round-trip.
        if let Some((actual, expected)) = type_serde.wrapper_type_mismatch() {
            return Err(serde::de::Error::custom(format!(
                "Cannot parse type from json: expected wrapper type '{expected}', got '{actual}'"
            )));
        }
        Ok(Type::from(type_serde))
    }
}

impl<'de> Deserialize<'de> for PrimitiveType {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: Deserializer<'de> {
        // Case-fold the type NAME before matching, exactly like Java 1.10.0
        // `Types.fromTypeName(name)` does `name.toLowerCase(Locale.ROOT)` before consulting the
        // primitive-name map and the `fixed[..]` / `decimal(..)` regexes (1.10.0 bytecode:
        // `fromTypeName` offsets 0-7 lowercase, then the TYPES map / FIXED / DECIMAL all match the
        // lowercased string). So `"BOOLEAN"`, `"Decimal(9,2)"`, `"FIXED[16]"` all parse. The object
        // WRAPPER names (`struct`/`list`/`map`) are NOT folded by Java (`SchemaParser.typeFromJson`
        // matches them with `String.equals`) and are handled structurally by the untagged
        // [`_serde::SerdeType`], so this lowercasing is scoped to primitive names only.
        //
        // `to_lowercase` here is `char::to_lowercase` (Unicode), a superset of Java's
        // `Locale.ROOT` ASCII fold for the only inputs that can collide with a type name (ASCII
        // letters); every real type name is ASCII, so the two agree on every accepted input.
        let s = String::deserialize(deserializer)?.to_lowercase();
        if s.starts_with("decimal") {
            deserialize_decimal(s.into_deserializer())
        } else if s.starts_with("fixed") {
            deserialize_fixed(s.into_deserializer())
        } else {
            PrimitiveType::deserialize(s.into_deserializer())
        }
    }
}

impl Serialize for PrimitiveType {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where S: Serializer {
        match self {
            PrimitiveType::Decimal { precision, scale } => {
                serialize_decimal(precision, scale, serializer)
            }
            PrimitiveType::Fixed(l) => serialize_fixed(l, serializer),
            _ => PrimitiveType::serialize(self, serializer),
        }
    }
}

fn deserialize_decimal<'de, D>(deserializer: D) -> std::result::Result<PrimitiveType, D::Error>
where D: Deserializer<'de> {
    // Java 1.10.0 `Types.fromTypeName` matches `decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)` against the
    // lowercased name (the caller has already lowercased). The regex REQUIRES the literal `decimal(`
    // prefix and the closing `)`, and `\s*` allows whitespace around the precision/scale and the
    // comma. We mirror that: require the wrapping `decimal(` … `)` exactly, then `trim()` the inner
    // operands. A missing close paren (`decimal(38,2`) or trailing junk (`decimal(38,2)x`) is
    // rejected here just as Java's anchored `matches()` rejects it.
    let s = String::deserialize(deserializer)?;
    let inner = s
        .strip_prefix("decimal(")
        .and_then(|rest| rest.strip_suffix(')'))
        .ok_or_else(|| D::Error::custom(format!("Cannot parse type string to primitive: {s}")))?;
    let (precision, scale) = inner
        .split_once(',')
        .ok_or_else(|| D::Error::custom(format!("Decimal requires precision and scale: {s}")))?;

    Ok(PrimitiveType::Decimal {
        precision: parse_unsigned_digits(precision, &s)?,
        scale: parse_unsigned_digits(scale, &s)?,
    })
}

/// Parses the `\d+` capture of Java's `fixed`/`decimal` regex: ASCII digits only after a `trim()`.
///
/// Rust's `FromStr for u32`/`u64` accept a leading `+` (`"+16"` parses), but Java's `\d+` does not,
/// so we reject any non-digit (the `+`/`-`/hex prefixes Java's regex never matches) before parsing.
/// `original` is the full type string, carried only for the error message.
fn parse_unsigned_digits<T, E>(digits: &str, original: &str) -> std::result::Result<T, E>
where
    T: std::str::FromStr,
    E: serde::de::Error,
{
    let trimmed = digits.trim();
    if trimmed.is_empty() || !trimmed.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(E::custom(format!(
            "Cannot parse type string to primitive: {original}"
        )));
    }
    trimmed
        .parse()
        .map_err(|_| E::custom(format!("Cannot parse type string to primitive: {original}")))
}

fn serialize_decimal<S>(
    precision: &u32,
    scale: &u32,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&format!("decimal({precision},{scale})"))
}

fn deserialize_fixed<'de, D>(deserializer: D) -> std::result::Result<PrimitiveType, D::Error>
where D: Deserializer<'de> {
    // Java 1.10.0 `Types.fromTypeName` matches `fixed\[\s*(\d+)\s*\]` against the lowercased name.
    // The regex REQUIRES the literal `fixed[` prefix and the closing `]`, and `\s*` allows
    // whitespace around the length. We mirror that: require the wrapping `fixed[` … `]` exactly,
    // then `trim()` the inner length. A missing close bracket (`fixed[16`) is rejected just as
    // Java's anchored `matches()` rejects it, and inner whitespace (`fixed[ 16 ]`) is accepted.
    let s = String::deserialize(deserializer)?;
    let inner = s
        .strip_prefix("fixed[")
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or_else(|| D::Error::custom(format!("Cannot parse type string to primitive: {s}")))?;

    parse_unsigned_digits(inner, &s).map(PrimitiveType::Fixed)
}

fn serialize_fixed<S>(value: &u64, serializer: S) -> std::result::Result<S::Ok, S::Error>
where S: Serializer {
    serializer.serialize_str(&format!("fixed[{value}]"))
}

impl fmt::Display for PrimitiveType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PrimitiveType::Boolean => write!(f, "boolean"),
            PrimitiveType::Int => write!(f, "int"),
            PrimitiveType::Long => write!(f, "long"),
            PrimitiveType::Float => write!(f, "float"),
            PrimitiveType::Double => write!(f, "double"),
            PrimitiveType::Decimal { precision, scale } => {
                write!(f, "decimal({precision},{scale})")
            }
            PrimitiveType::Date => write!(f, "date"),
            PrimitiveType::Time => write!(f, "time"),
            PrimitiveType::Timestamp => write!(f, "timestamp"),
            PrimitiveType::Timestamptz => write!(f, "timestamptz"),
            PrimitiveType::TimestampNs => write!(f, "timestamp_ns"),
            PrimitiveType::TimestamptzNs => write!(f, "timestamptz_ns"),
            PrimitiveType::String => write!(f, "string"),
            PrimitiveType::Uuid => write!(f, "uuid"),
            PrimitiveType::Fixed(size) => write!(f, "fixed({size})"),
            PrimitiveType::Binary => write!(f, "binary"),
            // Java `UnknownType.toString()` returns exactly "unknown".
            PrimitiveType::Unknown => write!(f, "unknown"),
        }
    }
}

/// DataType for a specific struct
#[derive(Debug, Serialize, Clone, Default)]
#[serde(rename = "struct", tag = "type")]
pub struct StructType {
    /// Struct fields
    fields: Vec<NestedFieldRef>,
    /// Lookup for index by field id
    #[serde(skip_serializing)]
    id_lookup: OnceLock<HashMap<i32, usize>>,
    #[serde(skip_serializing)]
    name_lookup: OnceLock<HashMap<String, usize>>,
}

impl<'de> Deserialize<'de> for StructType {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: Deserializer<'de> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Type,
            Fields,
        }

        struct StructTypeVisitor;

        impl<'de> Visitor<'de> for StructTypeVisitor {
            type Value = StructType;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<StructType, V::Error>
            where V: MapAccess<'de> {
                let mut fields = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Type => {
                            let type_val: String = map.next_value()?;
                            if type_val != "struct" {
                                return Err(serde::de::Error::custom(format!(
                                    "expected type 'struct', got '{type_val}'"
                                )));
                            }
                        }
                        Field::Fields => {
                            if fields.is_some() {
                                return Err(serde::de::Error::duplicate_field("fields"));
                            }
                            fields = Some(map.next_value()?);
                        }
                    }
                }
                let fields: Vec<NestedFieldRef> =
                    fields.ok_or_else(|| de::Error::missing_field("fields"))?;

                Ok(StructType::new(fields))
            }
        }

        const FIELDS: &[&str] = &["type", "fields"];
        deserializer.deserialize_struct("struct", FIELDS, StructTypeVisitor)
    }
}

impl StructType {
    /// Creates a struct type with the given fields.
    pub fn new(fields: Vec<NestedFieldRef>) -> Self {
        Self {
            fields,
            id_lookup: OnceLock::new(),
            name_lookup: OnceLock::new(),
        }
    }

    /// Get struct field with certain id
    pub fn field_by_id(&self, id: i32) -> Option<&NestedFieldRef> {
        self.field_id_to_index(id).map(|idx| &self.fields[idx])
    }

    fn field_id_to_index(&self, field_id: i32) -> Option<usize> {
        self.id_lookup
            .get_or_init(|| {
                HashMap::from_iter(self.fields.iter().enumerate().map(|(i, x)| (x.id, i)))
            })
            .get(&field_id)
            .copied()
    }

    /// Get struct field with certain field name
    pub fn field_by_name(&self, name: &str) -> Option<&NestedFieldRef> {
        self.field_name_to_index(name).map(|idx| &self.fields[idx])
    }

    fn field_name_to_index(&self, name: &str) -> Option<usize> {
        self.name_lookup
            .get_or_init(|| {
                HashMap::from_iter(
                    self.fields
                        .iter()
                        .enumerate()
                        .map(|(i, x)| (x.name.clone(), i)),
                )
            })
            .get(name)
            .copied()
    }

    /// Get fields.
    pub fn fields(&self) -> &[NestedFieldRef] {
        &self.fields
    }
}

impl PartialEq for StructType {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

impl Eq for StructType {}

impl Index<usize> for StructType {
    type Output = NestedField;

    fn index(&self, index: usize) -> &Self::Output {
        &self.fields[index]
    }
}

impl fmt::Display for StructType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "struct<")?;
        for field in &self.fields {
            write!(f, "{}", field.field_type)?;
        }
        write!(f, ">")
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Eq, Clone)]
#[serde(from = "SerdeNestedField", into = "SerdeNestedField")]
/// A struct is a tuple of typed values. Each field in the tuple is named and has an integer id that is unique in the table schema.
/// Each field can be either optional or required, meaning that values can (or cannot) be null. Fields may be any type.
/// Fields may have an optional comment or doc string. Fields can have default values.
pub struct NestedField {
    /// Id unique in table schema
    pub id: i32,
    /// Field Name
    pub name: String,
    /// Optional or required
    pub required: bool,
    /// Datatype
    pub field_type: Box<Type>,
    /// Fields may have an optional comment or doc string.
    pub doc: Option<String>,
    /// Used to populate the field’s value for all records that were written before the field was added to the schema
    pub initial_default: Option<Literal>,
    /// Used to populate the field’s value for any records written after the field was added to the schema, if the writer does not supply the field’s value
    pub write_default: Option<Literal>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
#[serde(rename_all = "kebab-case")]
struct SerdeNestedField {
    pub id: i32,
    pub name: String,
    pub required: bool,
    #[serde(rename = "type")]
    pub field_type: Box<Type>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_default: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_default: Option<JsonValue>,
}

impl From<SerdeNestedField> for NestedField {
    fn from(value: SerdeNestedField) -> Self {
        NestedField {
            id: value.id,
            name: value.name,
            required: value.required,
            initial_default: value.initial_default.and_then(|x| {
                Literal::try_from_json(x, &value.field_type)
                    .ok()
                    .and_then(identity)
            }),
            write_default: value.write_default.and_then(|x| {
                Literal::try_from_json(x, &value.field_type)
                    .ok()
                    .and_then(identity)
            }),
            field_type: value.field_type,
            doc: value.doc,
        }
    }
}

impl From<NestedField> for SerdeNestedField {
    fn from(value: NestedField) -> Self {
        let initial_default = value.initial_default.map(|x| x.try_into_json(&value.field_type).expect("We should have checked this in NestedField::with_initial_default, it can't be converted to json value"));
        let write_default = value.write_default.map(|x| x.try_into_json(&value.field_type).expect("We should have checked this in NestedField::with_write_default, it can't be converted to json value"));
        SerdeNestedField {
            id: value.id,
            name: value.name,
            required: value.required,
            field_type: value.field_type,
            doc: value.doc,
            initial_default,
            write_default,
        }
    }
}

/// Reference to nested field.
pub type NestedFieldRef = Arc<NestedField>;

impl NestedField {
    /// Construct a new field.
    pub fn new(id: i32, name: impl ToString, field_type: Type, required: bool) -> Self {
        Self {
            id,
            name: name.to_string(),
            required,
            field_type: Box::new(field_type),
            doc: None,
            initial_default: None,
            write_default: None,
        }
    }

    /// Construct a required field.
    pub fn required(id: i32, name: impl ToString, field_type: Type) -> Self {
        Self::new(id, name, field_type, true)
    }

    /// Construct an optional field.
    pub fn optional(id: i32, name: impl ToString, field_type: Type) -> Self {
        Self::new(id, name, field_type, false)
    }

    /// Construct list type's element field.
    pub fn list_element(id: i32, field_type: Type, required: bool) -> Self {
        Self::new(id, LIST_FIELD_NAME, field_type, required)
    }

    /// Construct map type's key field.
    pub fn map_key_element(id: i32, field_type: Type) -> Self {
        Self::required(id, MAP_KEY_FIELD_NAME, field_type)
    }

    /// Construct map type's value field.
    pub fn map_value_element(id: i32, field_type: Type, required: bool) -> Self {
        Self::new(id, MAP_VALUE_FIELD_NAME, field_type, required)
    }

    /// Set the field's doc.
    pub fn with_doc(mut self, doc: impl ToString) -> Self {
        self.doc = Some(doc.to_string());
        self
    }

    /// Set the field's initial default value.
    pub fn with_initial_default(mut self, value: Literal) -> Self {
        self.initial_default = Some(value);
        self
    }

    /// Set the field's initial default value.
    pub fn with_write_default(mut self, value: Literal) -> Self {
        self.write_default = Some(value);
        self
    }

    /// Set the id of the field.
    pub(crate) fn with_id(mut self, id: i32) -> Self {
        self.id = id;
        self
    }
}

impl fmt::Display for NestedField {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: ", self.id)?;
        write!(f, "{}: ", self.name)?;
        if self.required {
            write!(f, "required ")?;
        } else {
            write!(f, "optional ")?;
        }
        write!(f, "{} ", self.field_type)?;
        if let Some(doc) = &self.doc {
            write!(f, "{doc}")?;
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
/// A list is a collection of values with some element type. The element field has an integer id that is unique in the table schema.
/// Elements can be either optional or required. Element types may be any type.
pub struct ListType {
    /// Element field of list type.
    pub element_field: NestedFieldRef,
}

impl ListType {
    /// Construct a list type with the given element field.
    pub fn new(element_field: NestedFieldRef) -> Self {
        Self { element_field }
    }
}

/// Module for type serialization/deserialization.
pub(super) mod _serde {
    use std::borrow::Cow;
    use std::fmt;

    use serde::{Deserializer, Serializer};
    use serde_derive::{Deserialize, Serialize};

    use crate::spec::datatypes::Type::Map;
    use crate::spec::datatypes::{
        ListType, MapType, NestedField, NestedFieldRef, PrimitiveType, StructType, Type,
    };

    /// Marker that deserializes the JSON string `"variant"` (case-insensitively) and serializes
    /// the lowercase `"variant"`.
    ///
    /// Java `SchemaParser.toJson` writes variant the same way it writes primitives — as the bare
    /// string `type.toString()` (`"variant"`) — and `typeFromJson` parses any textual node through
    /// `Types.fromTypeName`, which LOWERCASES its input (`toLowerCase(Locale.ROOT)`, 1.10.0
    /// bytecode) before consulting the TYPES map whose key is `"variant" -> VariantType`. So Java
    /// reads `"Variant"`/`"VARIANT"` as the variant type too; this marker matches case-insensitively
    /// to stay at parity with the primitive-name case fold (see [`PrimitiveType`]'s deserializer).
    /// It rejects every non-`variant` string so the untagged [`SerdeType`] falls through to
    /// [`SerdeType::Primitive`] for real primitive names.
    pub(super) struct VariantTypeName;

    impl serde::Serialize for VariantTypeName {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where S: Serializer {
            serializer.serialize_str("variant")
        }
    }

    impl<'de> serde::Deserialize<'de> for VariantTypeName {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where D: Deserializer<'de> {
            struct VariantTypeNameVisitor;

            impl serde::de::Visitor<'_> for VariantTypeNameVisitor {
                type Value = VariantTypeName;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("the string 'variant' (case-insensitive)")
                }

                fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
                where E: serde::de::Error {
                    // Case-insensitive to mirror Java `Types.fromTypeName`'s `toLowerCase`.
                    if value.eq_ignore_ascii_case("variant") {
                        Ok(VariantTypeName)
                    } else {
                        Err(E::custom(format!("expected 'variant', got '{value}'")))
                    }
                }
            }

            deserializer.deserialize_str(VariantTypeNameVisitor)
        }
    }

    /// List type for serialization and deserialization
    #[derive(Serialize, Deserialize)]
    #[serde(untagged)]
    pub(super) enum SerdeType<'a> {
        #[serde(rename_all = "kebab-case")]
        List {
            r#type: String,
            element_id: i32,
            element_required: bool,
            element: Cow<'a, Type>,
        },
        Struct {
            r#type: String,
            fields: Cow<'a, [NestedFieldRef]>,
        },
        #[serde(rename_all = "kebab-case")]
        Map {
            r#type: String,
            key_id: i32,
            key: Cow<'a, Type>,
            value_id: i32,
            value_required: bool,
            value: Cow<'a, Type>,
        },
        // The exact string "variant" lands here; `VariantTypeName` rejects everything else,
        // letting genuine primitive names fall through to `Primitive`. Correctness does NOT
        // depend on this arm's position: untagged serde tries arms until one succeeds, and the
        // `PrimitiveType` deserializer independently rejects "variant" (pinned by
        // `variant_is_rejected_as_a_primitive_type_string`), so either order parses identically
        // (mutation-verified by the reviewer).
        Variant(VariantTypeName),
        Primitive(PrimitiveType),
    }

    impl SerdeType<'_> {
        /// Returns `Some((actual, expected))` when a wrapper arm's `type` string is not Java's
        /// exact lowercase wrapper name, else `None`.
        ///
        /// Java `SchemaParser.typeFromJson` selects the wrapper handler with
        /// `String.equals("struct"/"list"/"map")` (1.10.0 bytecode) — case-sensitive — so a
        /// wrong-cased name (`"STRUCT"`) is rejected. The untagged enum here matches a wrapper by its
        /// field shape and never inspects the `type` string, so [`Type`]'s deserializer calls this
        /// to re-impose Java's check. Primitive/variant arms carry no wrapper `type` and return
        /// `None` (they are validated by their own deserializers).
        pub(super) fn wrapper_type_mismatch(&self) -> Option<(&str, &'static str)> {
            let (actual, expected) = match self {
                SerdeType::List { r#type, .. } => (r#type.as_str(), "list"),
                SerdeType::Struct { r#type, .. } => (r#type.as_str(), "struct"),
                SerdeType::Map { r#type, .. } => (r#type.as_str(), "map"),
                SerdeType::Variant(_) | SerdeType::Primitive(_) => return None,
            };
            (actual != expected).then_some((actual, expected))
        }
    }

    impl From<SerdeType<'_>> for Type {
        fn from(value: SerdeType) -> Self {
            match value {
                SerdeType::List {
                    r#type: _,
                    element_id,
                    element_required,
                    element,
                } => Self::List(ListType {
                    element_field: NestedField::list_element(
                        element_id,
                        element.into_owned(),
                        element_required,
                    )
                    .into(),
                }),
                SerdeType::Map {
                    r#type: _,
                    key_id,
                    key,
                    value_id,
                    value_required,
                    value,
                } => Map(MapType {
                    key_field: NestedField::map_key_element(key_id, key.into_owned()).into(),
                    value_field: NestedField::map_value_element(
                        value_id,
                        value.into_owned(),
                        value_required,
                    )
                    .into(),
                }),
                SerdeType::Struct { r#type: _, fields } => {
                    Self::Struct(StructType::new(fields.into_owned()))
                }
                SerdeType::Variant(_) => Self::Variant,
                SerdeType::Primitive(p) => Self::Primitive(p),
            }
        }
    }

    impl<'a> From<&'a Type> for SerdeType<'a> {
        fn from(value: &'a Type) -> Self {
            match value {
                Type::List(list) => SerdeType::List {
                    r#type: "list".to_string(),
                    element_id: list.element_field.id,
                    element_required: list.element_field.required,
                    element: Cow::Borrowed(&list.element_field.field_type),
                },
                Type::Map(map) => SerdeType::Map {
                    r#type: "map".to_string(),
                    key_id: map.key_field.id,
                    key: Cow::Borrowed(&map.key_field.field_type),
                    value_id: map.value_field.id,
                    value_required: map.value_field.required,
                    value: Cow::Borrowed(&map.value_field.field_type),
                },
                Type::Struct(s) => SerdeType::Struct {
                    r#type: "struct".to_string(),
                    fields: Cow::Borrowed(&s.fields),
                },
                Type::Variant => SerdeType::Variant(VariantTypeName),
                Type::Primitive(p) => SerdeType::Primitive(p.clone()),
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
/// A map is a collection of key-value pairs with a key type and a value type.
/// Both the key field and value field each have an integer id that is unique in the table schema.
/// Map keys are required and map values can be either optional or required.
/// Both map keys and map values may be any type, including nested types.
pub struct MapType {
    /// Field for key.
    pub key_field: NestedFieldRef,
    /// Field for value.
    pub value_field: NestedFieldRef,
}

impl MapType {
    /// Construct a map type with the given key and value fields.
    pub fn new(key_field: NestedFieldRef, value_field: NestedFieldRef) -> Self {
        Self {
            key_field,
            value_field,
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use uuid::Uuid;

    use super::*;
    use crate::spec::values::PrimitiveLiteral;

    fn check_type_serde(json: &str, expected_type: Type) {
        let desered_type: Type = serde_json::from_str(json).unwrap();
        assert_eq!(desered_type, expected_type);

        let sered_json = serde_json::to_string(&expected_type).unwrap();
        let parsed_json_value = serde_json::from_str::<serde_json::Value>(&sered_json).unwrap();
        let raw_json_value = serde_json::from_str::<serde_json::Value>(json).unwrap();

        assert_eq!(parsed_json_value, raw_json_value);
    }

    #[test]
    fn primitive_type_serde() {
        let record = r#"
    {
        "type": "struct",
        "fields": [
            {"id": 1, "name": "bool_field", "required": true, "type": "boolean"},
            {"id": 2, "name": "int_field", "required": true, "type": "int"},
            {"id": 3, "name": "long_field", "required": true, "type": "long"},
            {"id": 4, "name": "float_field", "required": true, "type": "float"},
            {"id": 5, "name": "double_field", "required": true, "type": "double"},
            {"id": 6, "name": "decimal_field", "required": true, "type": "decimal(9,2)"},
            {"id": 7, "name": "date_field", "required": true, "type": "date"},
            {"id": 8, "name": "time_field", "required": true, "type": "time"},
            {"id": 9, "name": "timestamp_field", "required": true, "type": "timestamp"},
            {"id": 10, "name": "timestamptz_field", "required": true, "type": "timestamptz"},
            {"id": 11, "name": "timestamp_ns_field", "required": true, "type": "timestamp_ns"},
            {"id": 12, "name": "timestamptz_ns_field", "required": true, "type": "timestamptz_ns"},
            {"id": 13, "name": "uuid_field", "required": true, "type": "uuid"},
            {"id": 14, "name": "fixed_field", "required": true, "type": "fixed[10]"},
            {"id": 15, "name": "binary_field", "required": true, "type": "binary"},
            {"id": 16, "name": "string_field", "required": true, "type": "string"}
        ]
    }
    "#;

        check_type_serde(
            record,
            Type::Struct(StructType {
                fields: vec![
                    NestedField::required(1, "bool_field", Type::Primitive(PrimitiveType::Boolean))
                        .into(),
                    NestedField::required(2, "int_field", Type::Primitive(PrimitiveType::Int))
                        .into(),
                    NestedField::required(3, "long_field", Type::Primitive(PrimitiveType::Long))
                        .into(),
                    NestedField::required(4, "float_field", Type::Primitive(PrimitiveType::Float))
                        .into(),
                    NestedField::required(
                        5,
                        "double_field",
                        Type::Primitive(PrimitiveType::Double),
                    )
                    .into(),
                    NestedField::required(
                        6,
                        "decimal_field",
                        Type::Primitive(PrimitiveType::Decimal {
                            precision: 9,
                            scale: 2,
                        }),
                    )
                    .into(),
                    NestedField::required(7, "date_field", Type::Primitive(PrimitiveType::Date))
                        .into(),
                    NestedField::required(8, "time_field", Type::Primitive(PrimitiveType::Time))
                        .into(),
                    NestedField::required(
                        9,
                        "timestamp_field",
                        Type::Primitive(PrimitiveType::Timestamp),
                    )
                    .into(),
                    NestedField::required(
                        10,
                        "timestamptz_field",
                        Type::Primitive(PrimitiveType::Timestamptz),
                    )
                    .into(),
                    NestedField::required(
                        11,
                        "timestamp_ns_field",
                        Type::Primitive(PrimitiveType::TimestampNs),
                    )
                    .into(),
                    NestedField::required(
                        12,
                        "timestamptz_ns_field",
                        Type::Primitive(PrimitiveType::TimestamptzNs),
                    )
                    .into(),
                    NestedField::required(13, "uuid_field", Type::Primitive(PrimitiveType::Uuid))
                        .into(),
                    NestedField::required(
                        14,
                        "fixed_field",
                        Type::Primitive(PrimitiveType::Fixed(10)),
                    )
                    .into(),
                    NestedField::required(
                        15,
                        "binary_field",
                        Type::Primitive(PrimitiveType::Binary),
                    )
                    .into(),
                    NestedField::required(
                        16,
                        "string_field",
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                ],
                id_lookup: OnceLock::default(),
                name_lookup: OnceLock::default(),
            }),
        )
    }

    #[test]
    fn struct_type() {
        let record = r#"
        {
            "type": "struct",
            "fields": [
                {
                    "id": 1,
                    "name": "id",
                    "required": true,
                    "type": "uuid",
                    "initial-default": "0db3e2a8-9d1d-42b9-aa7b-74ebe558dceb",
                    "write-default": "ec5911be-b0a7-458c-8438-c9a3e53cffae"
                }, {
                    "id": 2,
                    "name": "data",
                    "required": false,
                    "type": "int"
                }
            ]
        }
        "#;

        check_type_serde(
            record,
            Type::Struct(StructType {
                fields: vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Uuid))
                        .with_initial_default(Literal::Primitive(PrimitiveLiteral::UInt128(
                            Uuid::parse_str("0db3e2a8-9d1d-42b9-aa7b-74ebe558dceb")
                                .unwrap()
                                .as_u128(),
                        )))
                        .with_write_default(Literal::Primitive(PrimitiveLiteral::UInt128(
                            Uuid::parse_str("ec5911be-b0a7-458c-8438-c9a3e53cffae")
                                .unwrap()
                                .as_u128(),
                        )))
                        .into(),
                    NestedField::optional(2, "data", Type::Primitive(PrimitiveType::Int)).into(),
                ],
                id_lookup: HashMap::from([(1, 0), (2, 1)]).into(),
                name_lookup: HashMap::from([("id".to_string(), 0), ("data".to_string(), 1)]).into(),
            }),
        )
    }

    #[test]
    fn test_deeply_nested_struct() {
        let record = r#"
{
  "type": "struct",
  "fields": [
    {
      "id": 1,
      "name": "id",
      "required": true,
      "type": "uuid",
      "initial-default": "0db3e2a8-9d1d-42b9-aa7b-74ebe558dceb",
      "write-default": "ec5911be-b0a7-458c-8438-c9a3e53cffae"
    },
    {
      "id": 2,
      "name": "data",
      "required": false,
      "type": "int"
    },
    {
      "id": 3,
      "name": "address",
      "required": true,
      "type": {
        "type": "struct",
        "fields": [
          {
            "id": 4,
            "name": "street",
            "required": true,
            "type": "string"
          },
          {
            "id": 5,
            "name": "province",
            "required": false,
            "type": "string"
          },
          {
            "id": 6,
            "name": "zip",
            "required": true,
            "type": "int"
          }
        ]
      }
    }
  ]
}
"#;

        let struct_type = Type::Struct(StructType::new(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Uuid))
                .with_initial_default(Literal::Primitive(PrimitiveLiteral::UInt128(
                    Uuid::parse_str("0db3e2a8-9d1d-42b9-aa7b-74ebe558dceb")
                        .unwrap()
                        .as_u128(),
                )))
                .with_write_default(Literal::Primitive(PrimitiveLiteral::UInt128(
                    Uuid::parse_str("ec5911be-b0a7-458c-8438-c9a3e53cffae")
                        .unwrap()
                        .as_u128(),
                )))
                .into(),
            NestedField::optional(2, "data", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(
                3,
                "address",
                Type::Struct(StructType::new(vec![
                    NestedField::required(4, "street", Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::optional(5, "province", Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::required(6, "zip", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
        ]));

        check_type_serde(record, struct_type)
    }

    #[test]
    fn list() {
        let record = r#"
        {
            "type": "list",
            "element-id": 3,
            "element-required": true,
            "element": "string"
        }
        "#;

        check_type_serde(
            record,
            Type::List(ListType {
                element_field: NestedField::list_element(
                    3,
                    Type::Primitive(PrimitiveType::String),
                    true,
                )
                .into(),
            }),
        );
    }

    #[test]
    fn map() {
        let record = r#"
        {
            "type": "map",
            "key-id": 4,
            "key": "string",
            "value-id": 5,
            "value-required": false,
            "value": "double"
        }
        "#;

        check_type_serde(
            record,
            Type::Map(MapType {
                key_field: NestedField::map_key_element(4, Type::Primitive(PrimitiveType::String))
                    .into(),
                value_field: NestedField::map_value_element(
                    5,
                    Type::Primitive(PrimitiveType::Double),
                    false,
                )
                .into(),
            }),
        );
    }

    #[test]
    fn map_int() {
        let record = r#"
        {
            "type": "map",
            "key-id": 4,
            "key": "int",
            "value-id": 5,
            "value-required": false,
            "value": "string"
        }
        "#;

        check_type_serde(
            record,
            Type::Map(MapType {
                key_field: NestedField::map_key_element(4, Type::Primitive(PrimitiveType::Int))
                    .into(),
                value_field: NestedField::map_value_element(
                    5,
                    Type::Primitive(PrimitiveType::String),
                    false,
                )
                .into(),
            }),
        );
    }

    #[test]
    fn test_decimal_precision() {
        let expected_max_precision = [
            2, 4, 6, 9, 11, 14, 16, 18, 21, 23, 26, 28, 31, 33, 35, 38, 40, 43, 45, 47, 50, 52, 55,
            57,
        ];
        for (i, max_precision) in expected_max_precision.iter().enumerate() {
            assert_eq!(
                *max_precision,
                Type::decimal_max_precision(i as u32 + 1).unwrap(),
                "Failed calculate max precision for {i}"
            );
        }

        assert_eq!(5, Type::decimal_required_bytes(10).unwrap());
        assert_eq!(16, Type::decimal_required_bytes(38).unwrap());
    }

    #[test]
    fn test_primitive_type_compatible() {
        let pairs = vec![
            (PrimitiveType::Boolean, PrimitiveLiteral::Boolean(true)),
            (PrimitiveType::Int, PrimitiveLiteral::Int(1)),
            (PrimitiveType::Long, PrimitiveLiteral::Long(1)),
            (PrimitiveType::Float, PrimitiveLiteral::Float(1.0.into())),
            (PrimitiveType::Double, PrimitiveLiteral::Double(1.0.into())),
            (
                PrimitiveType::Decimal {
                    precision: 9,
                    scale: 2,
                },
                PrimitiveLiteral::Int128(1),
            ),
            (PrimitiveType::Date, PrimitiveLiteral::Int(1)),
            (PrimitiveType::Time, PrimitiveLiteral::Long(1)),
            (PrimitiveType::Timestamptz, PrimitiveLiteral::Long(1)),
            (PrimitiveType::Timestamp, PrimitiveLiteral::Long(1)),
            (PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(1)),
            (PrimitiveType::TimestampNs, PrimitiveLiteral::Long(1)),
            (
                PrimitiveType::Uuid,
                PrimitiveLiteral::UInt128(Uuid::new_v4().as_u128()),
            ),
            (PrimitiveType::Fixed(8), PrimitiveLiteral::Binary(vec![1])),
            (PrimitiveType::Binary, PrimitiveLiteral::Binary(vec![1])),
        ];
        for (ty, literal) in pairs {
            assert!(ty.compatible(&literal));
        }
    }

    // RISK: `unknown` is a Java PRIMITIVE (`Types.UnknownType extends Type.PrimitiveType`), so it
    // must be a `PrimitiveType` arm, not a top-level `Type` variant — the on-disk JSON is the bare
    // string `"unknown"` (Java `UnknownType.toString()` and `Types.fromTypeName("unknown")`). The
    // `rename_all = "lowercase"` serde must give that name for free; a wrong serialization would
    // corrupt schema round-trips for every V3 table that uses it. Covers all four placements.
    #[test]
    fn unknown_type_serde_round_trip_in_all_placements() {
        let record = r#"
        {
            "type": "struct",
            "fields": [
                {"id": 1, "name": "u", "required": false, "type": "unknown"},
                {
                    "id": 2,
                    "name": "s",
                    "required": false,
                    "type": {
                        "type": "struct",
                        "fields": [
                            {"id": 3, "name": "nested_u", "required": false, "type": "unknown"}
                        ]
                    }
                },
                {
                    "id": 4,
                    "name": "l",
                    "required": false,
                    "type": {
                        "type": "list",
                        "element-id": 5,
                        "element": "unknown",
                        "element-required": false
                    }
                },
                {
                    "id": 6,
                    "name": "m",
                    "required": false,
                    "type": {
                        "type": "map",
                        "key-id": 7,
                        "key": "string",
                        "value-id": 8,
                        "value": "unknown",
                        "value-required": false
                    }
                }
            ]
        }
        "#;

        let de: Type = serde_json::from_str(record).expect("unknown schema must deserialize");
        // Re-serialize and compare as JSON values (key order independent).
        let reser = serde_json::to_value(&de).expect("re-serialize");
        let orig = serde_json::from_str::<serde_json::Value>(record).expect("orig json");
        assert_eq!(
            reser, orig,
            "unknown must round-trip structurally in every placement"
        );
    }

    // RISK: the bare-string serde for `unknown` is the whole on-disk contract for a no-physical-
    // column type. The JSON token must be exactly "unknown" (lowercase), and `Display` must match
    // Java `UnknownType.toString()`.
    #[test]
    fn unknown_type_serializes_as_bare_lowercase_string() {
        check_type_serde(r#""unknown""#, Type::Primitive(PrimitiveType::Unknown));
        assert_eq!(PrimitiveType::Unknown.to_string(), "unknown");
        assert_eq!(
            Type::Primitive(PrimitiveType::Unknown).to_string(),
            "unknown"
        );
        // Java `fromTypeName` lowercases the name first, so an upper-cased token also parses.
        let upper: Type = serde_json::from_str(r#""UNKNOWN""#).expect("case-folded unknown");
        assert_eq!(upper, Type::Primitive(PrimitiveType::Unknown));
    }

    // RISK: `unknown` has NO `PrimitiveLiteral` form (its values are always null). `compatible`
    // must reject EVERY literal — a stray accept would let a value flow into a column that can
    // hold none. Mutation guard: a `(Unknown, _) => true` arm would flip this red.
    #[test]
    fn unknown_type_is_compatible_with_no_literal() {
        let literals = [
            PrimitiveLiteral::Boolean(true),
            PrimitiveLiteral::Int(1),
            PrimitiveLiteral::Long(1),
            PrimitiveLiteral::Float(1.0.into()),
            PrimitiveLiteral::Double(1.0.into()),
            PrimitiveLiteral::Int128(1),
            PrimitiveLiteral::UInt128(1),
            PrimitiveLiteral::String("x".to_string()),
            PrimitiveLiteral::Binary(vec![1]),
        ];
        for literal in &literals {
            assert!(
                !PrimitiveType::Unknown.compatible(literal),
                "unknown must reject {literal:?}",
            );
        }
    }

    // RISK: the variant SCHEMA-JSON contract is the on-disk format — Java `SchemaParser.toJson`
    // writes variant as the bare string "variant" (the `isPrimitiveType() || isVariantType()`
    // branch) and `typeFromJson` parses it via `Types.fromTypeName` (1.10.0 bytecode). A wrong
    // serialization here corrupts schema round-trips for every V3 table. Covers all four
    // placements: top-level field, nested struct, list element, map value.
    #[test]
    fn variant_type_serde_round_trip_in_all_placements() {
        let record = r#"
        {
            "type": "struct",
            "fields": [
                {"id": 1, "name": "v", "required": false, "type": "variant"},
                {
                    "id": 2,
                    "name": "payload",
                    "required": true,
                    "type": {
                        "type": "struct",
                        "fields": [
                            {"id": 3, "name": "nested_v", "required": false, "type": "variant"}
                        ]
                    }
                },
                {
                    "id": 4,
                    "name": "events",
                    "required": true,
                    "type": {
                        "type": "list",
                        "element-id": 5,
                        "element-required": false,
                        "element": "variant"
                    }
                },
                {
                    "id": 6,
                    "name": "tags",
                    "required": true,
                    "type": {
                        "type": "map",
                        "key-id": 7,
                        "key": "string",
                        "value-id": 8,
                        "value-required": false,
                        "value": "variant"
                    }
                }
            ]
        }
        "#;

        check_type_serde(
            record,
            Type::Struct(StructType::new(vec![
                NestedField::optional(1, "v", Type::Variant).into(),
                NestedField::required(
                    2,
                    "payload",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "nested_v", Type::Variant).into(),
                    ])),
                )
                .into(),
                NestedField::required(
                    4,
                    "events",
                    Type::List(ListType {
                        element_field: NestedField::list_element(5, Type::Variant, false).into(),
                    }),
                )
                .into(),
                NestedField::required(
                    6,
                    "tags",
                    Type::Map(MapType {
                        key_field: NestedField::map_key_element(
                            7,
                            Type::Primitive(PrimitiveType::String),
                        )
                        .into(),
                        value_field: NestedField::map_value_element(8, Type::Variant, false).into(),
                    }),
                )
                .into(),
            ])),
        )
    }

    // RISK: "variant" must NOT parse as a PrimitiveType — Java `Types.fromPrimitiveString`
    // throws "Cannot parse type string: variant is not a primitive type" (1.10.0 bytecode, the
    // literal constant). Accepting it would let variant sneak through every primitive-only door
    // (promotion targets, accessors, partition literals).
    #[test]
    fn variant_is_rejected_as_a_primitive_type_string() {
        let result = serde_json::from_str::<PrimitiveType>(r#""variant""#);
        assert!(
            result.is_err(),
            "'variant' must not deserialize as a PrimitiveType"
        );
    }

    // RISK (case posture, 1.10.0-bytecode-derived): Java `Types.fromTypeName(name)` does
    // `name.toLowerCase(Locale.ROOT)` BEFORE consulting the primitive-name map and the
    // `fixed[..]` / `decimal(..)` regexes, so `SchemaParser` reads `"BOOLEAN"`/`"Decimal(9,2)"`/
    // `"FIXED[16]"`/`"Variant"` exactly as their lowercase forms. Rust used to be lowercase-exact
    // and REJECTED every mixed-case name Java accepts — a read-tolerance divergence. The fix folds
    // the primitive name (incl. the parameterized forms) and the variant marker; this pins the
    // accepted set. Self-mutation: removing the `.to_lowercase()` in `PrimitiveType::deserialize`
    // (and the variant marker's `eq_ignore_ascii_case`) makes every assertion below fail.
    #[test]
    fn primitive_type_names_parse_case_insensitively_like_java_from_type_name() {
        // Plain primitive names — upper, mixed, and lowercase all reach the same type.
        for cased in ["BOOLEAN", "Boolean", "boolean"] {
            assert_eq!(
                serde_json::from_str::<Type>(&format!(r#""{cased}""#))
                    .unwrap_or_else(|error| panic!("'{cased}' must parse: {error}")),
                Type::Primitive(PrimitiveType::Boolean),
                "'{cased}' must fold to boolean"
            );
        }
        assert_eq!(
            serde_json::from_str::<Type>(r#""STRING""#).expect("STRING folds to string"),
            Type::Primitive(PrimitiveType::String)
        );
        assert_eq!(
            serde_json::from_str::<Type>(r#""TimestampTZ""#).expect("mixed-case timestamptz folds"),
            Type::Primitive(PrimitiveType::Timestamptz)
        );
        // Parameterized forms: Java lowercases the WHOLE string before the FIXED / DECIMAL regex.
        assert_eq!(
            serde_json::from_str::<Type>(r#""Decimal(9,2)""#).expect("Decimal(9,2) folds"),
            Type::Primitive(PrimitiveType::Decimal {
                precision: 9,
                scale: 2
            })
        );
        assert_eq!(
            serde_json::from_str::<Type>(r#""FIXED[16]""#).expect("FIXED[16] folds"),
            Type::Primitive(PrimitiveType::Fixed(16))
        );
        // The variant name folds too (Java's TYPES map key is `"variant"`, matched against the
        // lowercased input — so `"Variant"`/`"VARIANT"` read as the variant type).
        for cased in ["Variant", "VARIANT", "vArIaNt", "variant"] {
            assert_eq!(
                serde_json::from_str::<Type>(&format!(r#""{cased}""#))
                    .unwrap_or_else(|error| panic!("'{cased}' must parse as variant: {error}")),
                Type::Variant,
                "'{cased}' must fold to variant"
            );
        }
    }

    // RISK (negative pin, 1.10.0-bytecode-derived): the case fold is SCOPED to type NAMES. Java
    // `fromTypeName` does NOT recognize trailing junk or whitespace-padded names ( leading space,
    // `variant2`) — they fall through to its `IllegalArgumentException`. Rust must reject the same.
    #[test]
    fn case_fold_does_not_admit_non_type_name_strings() {
        for bad in [
            " variant", "variant2", "boolean_", "decimal", "fixed", "notatype", "VARIANTX",
        ] {
            assert!(
                serde_json::from_str::<Type>(&format!(r#""{bad}""#)).is_err(),
                "'{bad}' must not parse as a type"
            );
        }
    }

    // RISK (scope pin, 1.10.0-bytecode-derived): Java folds case ONLY for primitive names.
    // `SchemaParser.typeFromJson` matches the object WRAPPER names `struct`/`list`/`map` with
    // `String.equals` (CASE-SENSITIVE), so `{"type":"STRUCT", ...}` FAILS in Java
    // (`IllegalArgumentException`). The dedicated `StructType` deserializer already enforced this;
    // the production `Type` route did NOT until the O3 REVIEWER added `wrapper_type_mismatch`
    // (pinned end-to-end in `wrapper_type_names_are_case_sensitive_via_the_type_path_like_java_schema_parser`).
    // This pin documents that the primitive case fold did not leak into the wrappers (a lowercased
    // `struct` is still required) AND that the `StructType` deserializer stays case-sensitive.
    #[test]
    fn wrapper_type_names_are_not_folded_by_the_primitive_case_fix() {
        // The lowercase wrapper parses (baseline, unchanged).
        let struct_json =
            r#"{"type":"struct","fields":[{"id":1,"name":"x","required":true,"type":"long"}]}"#;
        let parsed = serde_json::from_str::<Type>(struct_json).expect("lowercase struct parses");
        assert!(
            parsed.is_struct(),
            "lowercase struct must parse as a struct"
        );
        // The dedicated StructType deserializer is case-SENSITIVE on the wrapper name, matching
        // Java's `String.equals("struct")` (it is NOT routed through the primitive case fold).
        let upper_struct = r#"{"type":"STRUCT","fields":[]}"#;
        assert!(
            serde_json::from_str::<StructType>(upper_struct).is_err(),
            "StructType deserializer must reject an upper-case wrapper name (Java String.equals)"
        );
    }

    // RISK: the Java type-hierarchy placement — `Types.VariantType implements Type` directly, so
    // `isPrimitiveType()`, `isNestedType()` are FALSE and `isVariantType()` is TRUE, and
    // `toString()` is exactly "variant". Every legality door (partition/sort/identifier) keys off
    // these predicates; a wrong category flips all of them at once.
    #[test]
    fn variant_type_category_mirrors_java_hierarchy() {
        let variant = Type::Variant;
        assert!(!variant.is_primitive(), "variant is not a primitive type");
        assert!(!variant.is_nested(), "variant is not a nested type");
        assert!(!variant.is_struct(), "variant is not a struct type");
        assert!(variant.is_variant(), "is_variant must report true");
        assert!(
            variant.as_primitive_type().is_none(),
            "variant has no primitive view"
        );
        assert_eq!(
            variant.to_string(),
            "variant",
            "Display must match Java VariantType.toString()"
        );
        assert!(
            !Type::Primitive(PrimitiveType::String).is_variant(),
            "non-variant types must not report is_variant"
        );
    }

    #[test]
    fn struct_type_with_type_field() {
        // Test that StructType properly deserializes JSON with "type":"struct" field
        // This was previously broken because the deserializer wasn't consuming the type field value
        let json = r#"
        {
            "type": "struct",
            "fields": [
                {"id": 1, "name": "field1", "required": true, "type": "string"}
            ]
        }
        "#;

        let struct_type: StructType = serde_json::from_str(json)
            .expect("Should successfully deserialize StructType with type field");

        assert_eq!(struct_type.fields().len(), 1);
        assert_eq!(struct_type.fields()[0].name, "field1");
    }

    #[test]
    fn struct_type_rejects_wrong_type() {
        // Test that StructType validation rejects incorrect type field values
        let json = r#"
        {
            "type": "list",
            "fields": [
                {"id": 1, "name": "field1", "required": true, "type": "string"}
            ]
        }
        "#;

        let result = serde_json::from_str::<StructType>(json);
        assert!(
            result.is_err(),
            "Should reject StructType with wrong type field"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("expected type 'struct'")
        );
    }

    // RISK (item O3(a) parameterized parse fidelity, 1.10.0-bytecode + live-Java-probed by the
    // O3 REVIEWER): Java `Types.fromTypeName` matches `fixed\[\s*(\d+)\s*\]` and
    // `decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)` (anchored `matches()`) against the lowercased name.
    // `\s*` allows whitespace INSIDE the brackets/parens, and the anchors REQUIRE the close
    // bracket/paren. The Rust helpers previously used `trim_end_matches` (a no-op when the close
    // char is absent) and never trimmed the inner content, so they DIVERGED from Java two ways:
    //   - too LENIENT: `fixed[16` / `decimal(38,2` (no close) parsed in Rust, Java rejects;
    //   - too STRICT: `fixed[ 16 ]` (inner whitespace) was rejected in Rust, Java accepts.
    // The REVIEWER fix (strip-prefix-and-suffix + trim) restores 1:1. This pins BOTH arms against
    // a live `Types.fromTypeName` oracle (every case below was probed in Java 1.10.0).
    #[test]
    fn parameterized_type_parse_matches_java_fixed_and_decimal_regex() {
        // Inner whitespace ACCEPTED (Java `\s*`), folding case too.
        for (cased, expected) in [
            ("fixed[ 16 ]", PrimitiveType::Fixed(16)),
            ("fixed[16 ]", PrimitiveType::Fixed(16)),
            ("fixed[ 16]", PrimitiveType::Fixed(16)),
            ("FIXED[ 16 ]", PrimitiveType::Fixed(16)),
            ("decimal( 38 , 2 )", PrimitiveType::Decimal {
                precision: 38,
                scale: 2,
            }),
            ("Decimal( 38, 2 )", PrimitiveType::Decimal {
                precision: 38,
                scale: 2,
            }),
        ] {
            assert_eq!(
                serde_json::from_str::<Type>(&format!(r#""{cased}""#))
                    .unwrap_or_else(|error| panic!("'{cased}' must parse like Java: {error}")),
                Type::Primitive(expected),
                "'{cased}' must parse with inner whitespace, like Java's `\\s*`"
            );
        }
        // Missing close bracket/paren REJECTED (Java anchored `matches()`), trailing junk too, and
        // the `\d+` capture rejects a SIGN/extra-arg/hex Rust's `FromStr` would otherwise accept
        // (`fixed[+16]`/`decimal( +38 , 2 )` parse in raw Rust but NOT in Java — live-probed).
        // Self-mutation: reverting `deserialize_fixed`/`deserialize_decimal` to `trim_end_matches`
        // makes `fixed[16`/`decimal(38,2` parse, and dropping the `parse_unsigned_digits` digit
        // guard makes `fixed[+16]` parse (too lenient both ways) — these assertions catch it.
        for bad in [
            "fixed[16",
            "fixed[16]x",
            "fixed16]",
            "xfixed[16]",
            "fixed[]",
            "fixed[+16]",
            "fixed[-16]",
            "fixed[0x10]",
            "decimal(38,2",
            "decimal(38,2)x",
            "decimal38,2)",
            "xdecimal(38,2)",
            "decimal()",
            "decimal(38,2,3)",
            "decimal( +38 , 2 )",
            "decimal(-38,2)",
        ] {
            assert!(
                serde_json::from_str::<Type>(&format!(r#""{bad}""#)).is_err(),
                "'{bad}' must be rejected like Java's anchored regex"
            );
        }
    }

    // RISK (item O3(a) wrapper read-leniency, 1.10.0-bytecode + live-Java-probed by the O3
    // REVIEWER): Java `SchemaParser.typeFromJson` selects the wrapper handler with
    // `String.equals("struct"/"list"/"map")` (1.10.0 bytecode offsets 41/55/69) — CASE-SENSITIVE —
    // so `{"type":"STRUCT"/"LIST"/"MAP"}` raises `IllegalArgumentException` (live-probed: lowercase
    // `struct` OK, `STRUCT`/`LIST`/`MAP` all ERR). The untagged `_serde::SerdeType` matches a
    // wrapper by its field SHAPE and ignores the `type` string, so the `Type` deserializer USED TO
    // accept a wrong-cased wrapper Java rejects — a read-leniency divergence the builder documented
    // but did not close (its scope test exercised the `StructType` deserializer, a path the `Type`
    // route never takes). The REVIEWER added `SerdeType::wrapper_type_mismatch` re-imposing Java's
    // `String.equals`. This pins BOTH arms via the production `Type` path. Self-mutation: removing
    // the `wrapper_type_mismatch` guard in `Type::deserialize` makes every upper-case case parse.
    #[test]
    fn wrapper_type_names_are_case_sensitive_via_the_type_path_like_java_schema_parser() {
        // Lowercase wrapper names parse (baseline) through the production `Type` route.
        for good in [
            r#"{"type":"struct","fields":[{"id":1,"name":"x","required":true,"type":"long"}]}"#,
            r#"{"type":"list","element-id":1,"element-required":true,"element":"long"}"#,
            r#"{"type":"map","key-id":1,"key":"long","value-id":2,"value-required":true,"value":"long"}"#,
        ] {
            assert!(
                serde_json::from_str::<Type>(good).is_ok(),
                "lowercase wrapper must parse via the Type path: {good}"
            );
        }
        // Wrong-cased wrapper names are REJECTED, matching Java's `String.equals` (live-probed
        // `IllegalArgumentException`). This is the production read path (a `Schema`/`Type` doc),
        // NOT the dedicated `StructType` deserializer the builder's scope test used.
        for bad in [
            r#"{"type":"STRUCT","fields":[{"id":1,"name":"x","required":true,"type":"long"}]}"#,
            r#"{"type":"Struct","fields":[{"id":1,"name":"x","required":true,"type":"long"}]}"#,
            r#"{"type":"LIST","element-id":1,"element-required":true,"element":"long"}"#,
            r#"{"type":"MAP","key-id":1,"key":"long","value-id":2,"value-required":true,"value":"long"}"#,
        ] {
            assert!(
                serde_json::from_str::<Type>(bad).is_err(),
                "wrong-cased wrapper must be rejected via the Type path (Java String.equals): {bad}"
            );
        }
    }
}
