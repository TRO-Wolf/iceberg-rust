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

//! Fill missing Iceberg columns from `write-default` before a data file is written.
//!
//! Spec: a writer must emit every known field. A missing field takes `write-default`.
//! A required field with no `write-default` fails. An optional field with none writes null.

use std::borrow::Cow;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeBinaryArray, LargeBinaryArray, LargeListArray, ListArray, MapArray,
    RecordBatch, Time64MicrosecondArray, make_array, new_null_array,
};
use arrow_buffer::{BooleanBuffer, MutableBuffer, NullBuffer, bit_util};
use arrow_schema::{DataType, Field, Fields, SchemaRef as ArrowSchemaRef, TimeUnit};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use uuid::Uuid;

use crate::arrow::{create_primitive_array_repeated, is_utc_time_zone};
use crate::spec::{Literal, NestedField, PrimitiveLiteral, Schema};
use crate::{Error, ErrorKind, Result};

/// Project `batch` onto `schema`, filling any missing field from `write-default`.
///
/// A complete batch in Iceberg field order is returned borrowed. Extra batch columns
/// are dropped. Nested `write-default` fill is refused (row R92 residue).
///
/// # Errors
///
/// [`ErrorKind::DataInvalid`] when a required field is missing and has no `write-default`.
/// [`ErrorKind::FeatureUnsupported`] when a missing field's `write-default` is not primitive.
pub(crate) fn apply_write_defaults<'a>(
    schema: &Schema,
    target_schema: &ArrowSchemaRef,
    batch: &'a RecordBatch,
) -> Result<Cow<'a, RecordBatch>> {
    let iceberg_fields = schema.as_struct().fields();
    if batch.num_rows() == 0 {
        return Ok(Cow::Owned(RecordBatch::new_empty(target_schema.clone())));
    }
    if batch_matches_schema_order(target_schema.fields(), batch) {
        return Ok(Cow::Borrowed(batch));
    }

    let num_rows = batch.num_rows();
    let mut columns = Vec::with_capacity(iceberg_fields.len());
    for (iceberg_field, arrow_field) in iceberg_fields.iter().zip(target_schema.fields()) {
        if let Some(idx) = batch_column_index(batch, iceberg_field) {
            columns.push(relabel_column(
                batch.column(idx),
                arrow_field.data_type(),
                0,
            )?);
        } else {
            columns.push(fill_missing_column(iceberg_field, arrow_field, num_rows)?);
        }
    }

    Ok(Cow::Owned(RecordBatch::try_new(
        target_schema.clone(),
        columns,
    )?))
}

fn batch_matches_schema_order(target_fields: &Fields, batch: &RecordBatch) -> bool {
    let batch_schema = batch.schema();
    let batch_fields = batch_schema.fields();
    batch_fields.len() == target_fields.len()
        && batch_fields
            .iter()
            .zip(target_fields.iter())
            .all(|(batch_field, target_field)| borrow_field_eq(batch_field, target_field))
}

fn borrow_field_eq(batch_field: &Field, target_field: &Field) -> bool {
    batch_field.name() == target_field.name()
        && batch_field.is_nullable() == target_field.is_nullable()
        && batch_field_id(batch_field) == batch_field_id(target_field)
        && borrow_type_eq(batch_field.data_type(), target_field.data_type())
}

fn borrow_type_eq(batch_type: &DataType, target_type: &DataType) -> bool {
    match (batch_type, target_type) {
        (DataType::Struct(batch_fields), DataType::Struct(target_fields)) => {
            batch_fields.len() == target_fields.len()
                && batch_fields
                    .iter()
                    .zip(target_fields.iter())
                    .all(|(batch_field, target_field)| borrow_field_eq(batch_field, target_field))
        }
        (DataType::List(batch_field), DataType::List(target_field))
        | (DataType::LargeList(batch_field), DataType::LargeList(target_field)) => {
            borrow_field_eq(batch_field, target_field)
        }
        (
            DataType::FixedSizeList(batch_field, batch_len),
            DataType::FixedSizeList(target_field, target_len),
        ) => batch_len == target_len && borrow_field_eq(batch_field, target_field),
        (DataType::Map(batch_field, batch_sorted), DataType::Map(target_field, target_sorted)) => {
            batch_sorted == target_sorted && borrow_field_eq(batch_field, target_field)
        }
        (
            DataType::Dictionary(batch_key, batch_value),
            DataType::Dictionary(target_key, target_value),
        ) => batch_key == target_key && borrow_type_eq(batch_value, target_value),
        _ => batch_type == target_type,
    }
}

const MAX_RELABEL_DEPTH: usize = 128;

fn relabel_column(column: &ArrayRef, target: &DataType, depth: usize) -> Result<ArrayRef> {
    let actual = column.data_type();
    if actual == target {
        return Ok(column.clone());
    }
    if depth > MAX_RELABEL_DEPTH {
        return Err(incompatible_type(actual, target));
    }
    let actual_children = nested_fields(actual);
    let target_children = nested_fields(target);
    let data = column.to_data();
    if !compatible_layout(actual, target) || actual_children.len() != data.child_data().len() {
        return match cast_leaf_encoding(column, actual, target) {
            Some(array) => array,
            None => Err(incompatible_type(actual, target)),
        };
    }
    let mut children = Vec::with_capacity(target_children.len());
    for ((actual_field, target_field), child_data) in actual_children
        .iter()
        .zip(target_children.iter())
        .zip(data.child_data())
    {
        if matches!(target, DataType::Struct(_)) && actual_field.name() != target_field.name() {
            return Err(incompatible_type(actual, target));
        }
        let child_array = make_array(child_data.clone());
        if !target_field.is_nullable() && disallowed_nulls(target, column, &child_array) {
            return Err(incompatible_type(actual, target));
        }
        children.push(relabel_column(&child_array, target_field.data_type(), depth + 1)?.to_data());
    }
    Ok(make_array(unsafe {
        data.into_builder()
            .data_type(target.clone())
            .child_data(children)
            .build_unchecked()
    }))
}

fn disallowed_nulls(target: &DataType, column: &ArrayRef, child: &ArrayRef) -> bool {
    let Some(child_nulls) = child.nulls() else {
        return false;
    };
    if child_nulls.null_count() == 0 {
        return false;
    }
    let mut bits = MutableBuffer::new_null(child.len());
    let base = column.offset();
    match target {
        DataType::List(_) => {
            let Some(list) = column.as_any().downcast_ref::<ListArray>() else {
                return true;
            };
            let offsets = list.value_offsets();
            mark_valid_ranges(&mut bits, column, |i| {
                offsets[i] as usize..offsets[i + 1] as usize
            });
        }
        DataType::LargeList(_) => {
            let Some(list) = column.as_any().downcast_ref::<LargeListArray>() else {
                return true;
            };
            let offsets = list.value_offsets();
            mark_valid_ranges(&mut bits, column, |i| {
                offsets[i] as usize..offsets[i + 1] as usize
            });
        }
        DataType::FixedSizeList(_, width) => {
            let width = *width as usize;
            mark_valid_ranges(&mut bits, column, |i| {
                (base + i) * width..(base + i + 1) * width
            });
        }
        DataType::Map(_, _) => {
            let Some(map) = column.as_any().downcast_ref::<MapArray>() else {
                return true;
            };
            let offsets = map.value_offsets();
            mark_valid_ranges(&mut bits, column, |i| {
                offsets[i] as usize..offsets[i + 1] as usize
            });
        }
        DataType::Struct(_) => {
            mark_valid_ranges(&mut bits, column, |i| base + i..base + i + 1);
        }
        _ => return true,
    }
    let mask = NullBuffer::new(BooleanBuffer::new(bits.into(), 0, child.len()));
    !mask.contains(child_nulls)
}

fn mark_valid_ranges(
    bits: &mut MutableBuffer,
    parent: &ArrayRef,
    range: impl Fn(usize) -> std::ops::Range<usize>,
) {
    for i in 0..parent.len() {
        if parent.is_valid(i) {
            for j in range(i) {
                bit_util::set_bit(bits.as_mut(), j);
            }
        }
    }
}

fn nested_fields(data_type: &DataType) -> Vec<&Field> {
    match data_type {
        DataType::Struct(fields) => fields.iter().map(|field| field.as_ref()).collect(),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => vec![field.as_ref()],
        _ => vec![],
    }
}

fn compatible_layout(actual: &DataType, target: &DataType) -> bool {
    match (actual, target) {
        (DataType::Struct(a), DataType::Struct(b)) => a.len() == b.len(),
        (DataType::List(_), DataType::List(_))
        | (DataType::LargeList(_), DataType::LargeList(_)) => true,
        (DataType::FixedSizeList(_, a), DataType::FixedSizeList(_, b)) => a == b,
        (DataType::Map(_, a), DataType::Map(_, b)) => a == b,
        (
            DataType::Timestamp(actual_unit, Some(actual_tz)),
            DataType::Timestamp(target_unit, Some(target_tz)),
        ) => {
            actual_unit == target_unit
                && is_utc_time_zone(actual_tz.as_ref())
                && is_utc_time_zone(target_tz.as_ref())
        }
        _ => false,
    }
}

fn cast_leaf_encoding(
    column: &ArrayRef,
    actual: &DataType,
    target: &DataType,
) -> Option<Result<ArrayRef>> {
    if parquet_leaf_equivalent(actual, target) {
        Some(
            arrow_cast::cast(column, target)
                .map_err(|err| incompatible_type(actual, target).with_source(err)),
        )
    } else {
        None
    }
}

fn parquet_leaf_equivalent(actual: &DataType, target: &DataType) -> bool {
    let actual = match actual {
        DataType::Dictionary(_, value) => value.as_ref(),
        other => other,
    };
    if actual == target {
        return true;
    }
    matches!(
        (actual, target),
        (
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) | (
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        )
    )
}

fn incompatible_type(actual: &DataType, target: &DataType) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("Column type {actual} is not compatible with the table schema type {target}"),
    )
}

pub(crate) fn batch_field_id(field: &Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .and_then(|value| value.parse().ok())
}

fn batch_column_index(batch: &RecordBatch, field: &NestedField) -> Option<usize> {
    let schema = batch.schema();
    for (idx, arrow_field) in schema.fields().iter().enumerate() {
        if batch_field_id(arrow_field) == Some(field.id) {
            return Some(idx);
        }
    }
    schema.fields().iter().position(|arrow_field| {
        arrow_field.name() == field.name.as_str() && batch_field_id(arrow_field).is_none()
    })
}

fn fill_missing_column(
    field: &NestedField,
    arrow_field: &Field,
    num_rows: usize,
) -> Result<ArrayRef> {
    match field.write_default.as_ref() {
        Some(Literal::Primitive(prim)) => {
            repeat_write_default(arrow_field.data_type(), prim, num_rows).map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot apply write-default for field '{}' (id {}): {}",
                        field.name,
                        field.id,
                        err.message()
                    ),
                )
            })
        }
        Some(_) => Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "write-default fill for non-primitive field '{}' is not supported",
                field.name
            ),
        )),
        None if field.required => Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Required field '{}' (id {}) is missing from the write batch and has no write-default",
                field.name, field.id
            ),
        )),
        None => Ok(new_null_array(arrow_field.data_type(), num_rows)),
    }
}

fn repeat_write_default(
    data_type: &DataType,
    prim: &PrimitiveLiteral,
    num_rows: usize,
) -> Result<ArrayRef> {
    match (data_type, prim) {
        (DataType::Time64(TimeUnit::Microsecond), PrimitiveLiteral::Long(value)) => {
            Ok(Arc::new(Time64MicrosecondArray::from(vec![
                *value;
                num_rows
            ])))
        }
        (DataType::FixedSizeBinary(16), PrimitiveLiteral::UInt128(value)) => {
            let bytes = Uuid::from_u128(*value).into_bytes();
            let values: Vec<&[u8]> = vec![bytes.as_slice(); num_rows];
            Ok(Arc::new(FixedSizeBinaryArray::try_from_iter(
                values.into_iter(),
            )?))
        }
        (DataType::FixedSizeBinary(width), PrimitiveLiteral::Binary(bytes))
            if bytes.len() == *width as usize =>
        {
            let values: Vec<&[u8]> = vec![bytes.as_slice(); num_rows];
            Ok(Arc::new(FixedSizeBinaryArray::try_from_iter(
                values.into_iter(),
            )?))
        }
        (DataType::LargeBinary, PrimitiveLiteral::Binary(bytes)) => Ok(Arc::new(
            LargeBinaryArray::from_iter_values(std::iter::repeat_n(bytes.as_slice(), num_rows)),
        )),
        _ => create_primitive_array_repeated(data_type, &Some(prim.clone()), num_rows),
    }
}
