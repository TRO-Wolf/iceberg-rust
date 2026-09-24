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

use arrow_array::builder::{
    BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder, Float64Builder, Int32Builder,
    Int64Builder, LargeBinaryBuilder, StringBuilder, StructBuilder, Time64MicrosecondBuilder,
    TimestampMicrosecondBuilder, TimestampNanosecondBuilder,
};

use crate::spec::{Literal, PrimitiveLiteral, PrimitiveType, Struct, StructType};
use crate::{Error, ErrorKind, Result};

/// Appends one partition tuple to the partition [`StructBuilder`].
///
/// Values match BY FIELD ID against the spec the tuple was written under
/// (Java `PartitionUtil.coercePartition`); absent fields null-fill. Each
/// matched value widens through a legal promotion before extract.
pub(crate) fn append_partition(
    builder: &mut StructBuilder,
    partition_type: &StructType,
    source_field_ids: &[i32],
    partition: &crate::spec::Struct,
) -> Result<()> {
    for (index, field) in partition_type.fields().iter().enumerate() {
        let primitive_type = field.field_type.as_primitive_type().ok_or_else(|| {
            Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "partition field '{}' has non-primitive type {:?}; not supported in the data_file metadata projection",
                    field.name, field.field_type
                ),
            )
        })?;
        let value = source_field_ids
            .iter()
            .position(|source_field_id| *source_field_id == field.id)
            .and_then(|source_index| partition.fields().get(source_index))
            .and_then(|value| value.as_ref());
        let promoted;
        let value = match value {
            Some(Literal::Primitive(literal)) if !primitive_type.compatible(literal) => {
                let widened = literal.promote_to(primitive_type);
                if primitive_type.compatible(&widened) {
                    promoted = Some(Literal::Primitive(widened));
                    promoted.as_ref()
                } else {
                    value
                }
            }
            _ => value,
        };
        append_partition_field(builder, index, primitive_type, value)?;
    }
    builder.append(true);
    Ok(())
}

/// Appends a single partition-field value (or null) to the struct child builder at `index`, dispatching
/// on the field's primitive type. Mirrors the Arrow types produced by `type_to_arrow_type`.
fn append_partition_field(
    builder: &mut StructBuilder,
    index: usize,
    primitive_type: &PrimitiveType,
    value: Option<&Literal>,
) -> Result<()> {
    let primitive = match value {
        Some(Literal::Primitive(primitive)) => Some(primitive),
        Some(other) => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("non-primitive partition literal {other:?} is not supported"),
            ));
        }
        None => None,
    };

    macro_rules! append_typed {
        ($builder_ty:ty, $extract:expr) => {{
            let child = builder.field_builder::<$builder_ty>(index).ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("partition child builder at index {index} has an unexpected type"),
                )
            })?;
            match primitive {
                Some(primitive) => child.append_value($extract(primitive)?),
                None => child.append_null(),
            }
        }};
    }

    match primitive_type {
        PrimitiveType::Boolean => append_typed!(BooleanBuilder, extract_bool),
        PrimitiveType::Int => append_typed!(Int32Builder, extract_i32),
        PrimitiveType::Long => append_typed!(Int64Builder, extract_i64),
        PrimitiveType::Float => append_typed!(Float32Builder, extract_f32),
        PrimitiveType::Double => append_typed!(Float64Builder, extract_f64),
        PrimitiveType::Date => append_typed!(Date32Builder, extract_i32),
        PrimitiveType::Time => append_typed!(Time64MicrosecondBuilder, extract_i64),
        PrimitiveType::Timestamp => append_typed!(TimestampMicrosecondBuilder, extract_i64),
        PrimitiveType::Timestamptz => append_typed!(TimestampMicrosecondBuilder, extract_i64),
        PrimitiveType::TimestampNs => append_typed!(TimestampNanosecondBuilder, extract_i64),
        PrimitiveType::TimestamptzNs => append_typed!(TimestampNanosecondBuilder, extract_i64),
        PrimitiveType::String => append_typed!(StringBuilder, extract_string),
        PrimitiveType::Binary => append_typed!(LargeBinaryBuilder, extract_binary),
        PrimitiveType::Decimal { .. } => append_typed!(Decimal128Builder, extract_i128),
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "partition field type {other:?} is not supported in the data_file metadata projection"
                ),
            ));
        }
    }
    Ok(())
}

fn type_mismatch(primitive: &PrimitiveLiteral) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("partition literal {primitive:?} does not match its partition field type"),
    )
}

fn extract_bool(primitive: &PrimitiveLiteral) -> Result<bool> {
    match primitive {
        PrimitiveLiteral::Boolean(value) => Ok(*value),
        other => Err(type_mismatch(other)),
    }
}

fn extract_i32(primitive: &PrimitiveLiteral) -> Result<i32> {
    match primitive {
        PrimitiveLiteral::Int(value) => Ok(*value),
        other => Err(type_mismatch(other)),
    }
}

fn extract_i64(primitive: &PrimitiveLiteral) -> Result<i64> {
    match primitive {
        PrimitiveLiteral::Long(value) => Ok(*value),
        other => Err(type_mismatch(other)),
    }
}

fn extract_f32(primitive: &PrimitiveLiteral) -> Result<f32> {
    match primitive {
        PrimitiveLiteral::Float(value) => Ok(value.into_inner()),
        other => Err(type_mismatch(other)),
    }
}

fn extract_f64(primitive: &PrimitiveLiteral) -> Result<f64> {
    match primitive {
        PrimitiveLiteral::Double(value) => Ok(value.into_inner()),
        other => Err(type_mismatch(other)),
    }
}

fn extract_string(primitive: &PrimitiveLiteral) -> Result<&str> {
    match primitive {
        PrimitiveLiteral::String(value) => Ok(value.as_str()),
        other => Err(type_mismatch(other)),
    }
}

fn extract_binary(primitive: &PrimitiveLiteral) -> Result<&[u8]> {
    match primitive {
        PrimitiveLiteral::Binary(value) => Ok(value.as_slice()),
        other => Err(type_mismatch(other)),
    }
}

fn extract_i128(primitive: &PrimitiveLiteral) -> Result<i128> {
    match primitive {
        PrimitiveLiteral::Int128(value) => Ok(*value),
        other => Err(type_mismatch(other)),
    }
}

/// Compares two partition tuples field-by-field for a deterministic row order.
///
/// Mirrors Java's `Comparators.forType(partitionType)` ordering for the common case: nulls sort first,
/// then each field's primitive value is compared via [`PrimitiveLiteral`]'s `PartialOrd`. Any incomparable
/// pair (e.g. a `NaN`, or a non-primitive partition literal — neither of which is a valid partition value)
/// falls back to `Equal`, so the order stays total + deterministic under a stable sort. The first field
/// that differs decides the order.
pub(super) fn compare_partition_values(left: &Struct, right: &Struct) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let left_fields = left.fields();
    let right_fields = right.fields();
    let len = left_fields.len().min(right_fields.len());
    for index in 0..len {
        let ordering = compare_partition_field(&left_fields[index], &right_fields[index]);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left_fields.len().cmp(&right_fields.len())
}

/// Compares one optional partition field value; `None` (null) sorts before any value.
fn compare_partition_field(left: &Option<Literal>, right: &Option<Literal>) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(Literal::Primitive(left)), Some(Literal::Primitive(right))) => {
            compare_primitive(left, right)
        }
        // Non-primitive partition literals are not valid partition values; keep order stable.
        _ => Ordering::Equal,
    }
}

/// Compares two [`PrimitiveLiteral`]s, falling back to `Equal` for an incomparable pair.
fn compare_primitive(left: &PrimitiveLiteral, right: &PrimitiveLiteral) -> std::cmp::Ordering {
    left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
}
