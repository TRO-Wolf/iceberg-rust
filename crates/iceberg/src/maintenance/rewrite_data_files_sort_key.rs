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

use arrow_arith::boolean::is_null;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Decimal256Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, Time64MicrosecondType, TimestampMicrosecondType, TimestampNanosecondType,
};
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{DataType, Schema as ArrowSchema, TimeUnit};
use arrow_select::nullif::nullif;

use crate::error::{Error, ErrorKind, Result};
use crate::maintenance::rewrite_data_files_sort::ResolvedStrategy;
use crate::maintenance::rewrite_data_files_zorder::ZOrderEncoder;
use crate::spec::{
    NestedFieldRef, NullOrder, Schema as IcebergSchema, SortDirection, Transform, Type,
};
use crate::transform::{BoxedTransformFunction, create_transform_function};

pub(super) struct SortKeyField {
    column: usize,
    nested_path: Vec<String>,
    transform: Option<BoxedTransformFunction>,
    descending: bool,
    nulls_first: bool,
}

pub(super) enum KeyPlan {
    Sort(Vec<SortKeyField>),
    ZOrder(ZOrderEncoder),
}

impl KeyPlan {
    pub(super) fn build(
        strategy: &ResolvedStrategy,
        iceberg_schema: &IcebergSchema,
        arrow_schema: &ArrowSchema,
    ) -> Result<Option<KeyPlan>> {
        match strategy {
            ResolvedStrategy::BinPack => Ok(None),
            ResolvedStrategy::Sort { order, .. } => {
                let mut fields = Vec::with_capacity(order.fields.len());
                for field in &order.fields {
                    if field.transform == Transform::Void {
                        continue;
                    }
                    let source = iceberg_schema.field_by_id(field.source_id).ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Cannot find source column for sort field: {field}"),
                        )
                    })?;
                    let (top_name, nested_path) = sort_source_path(iceberg_schema, field.source_id)
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                format!("Cannot find source column for sort field: {field}"),
                            )
                        })?;
                    let column = arrow_schema.index_of(&top_name).map_err(|error| {
                        Error::new(
                            ErrorKind::Unexpected,
                            format!("Sort column '{top_name}' is not in the rewrite's batches"),
                        )
                        .with_source(error)
                    })?;
                    let transform = if field.transform == Transform::Identity {
                        None
                    } else {
                        field.transform.result_type(source.field_type.as_ref())?;
                        Some(create_transform_function(&field.transform)?)
                    };
                    fields.push(SortKeyField {
                        column,
                        nested_path,
                        transform,
                        descending: field.direction == SortDirection::Descending,
                        nulls_first: field.null_order == NullOrder::First,
                    });
                }
                if fields.is_empty() {
                    return Ok(None);
                }
                Ok(Some(KeyPlan::Sort(fields)))
            }
            ResolvedStrategy::ZOrder {
                columns,
                var_length_contribution,
                max_output_size,
            } => Ok(Some(KeyPlan::ZOrder(ZOrderEncoder::build(
                columns,
                iceberg_schema,
                arrow_schema,
                *var_length_contribution,
                *max_output_size,
            )?))),
        }
    }

    pub(super) fn encode(&self, batch: &RecordBatch) -> Result<Vec<Vec<u8>>> {
        let mut keys = vec![Vec::new(); batch.num_rows()];
        match self {
            KeyPlan::Sort(fields) => {
                for field in fields {
                    let array = field_array(batch, field.column, &field.nested_path)?;
                    let array = match &field.transform {
                        Some(function) => function.transform(array)?,
                        None => array,
                    };
                    encode_sort_field(&array, field.descending, field.nulls_first, &mut keys)?;
                }
            }
            KeyPlan::ZOrder(encoder) => {
                encoder.encode(batch, &mut keys)?;
            }
        }
        Ok(keys)
    }
}

pub(super) fn field_array(
    batch: &RecordBatch,
    column: usize,
    nested_path: &[String],
) -> Result<ArrayRef> {
    let mut array = batch.column(column).clone();
    for segment in nested_path {
        let parent = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("sort key column '{segment}' is not a struct"),
                )
            })?;
        let child = parent.column_by_name(segment).ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("sort key struct field '{segment}' not found"),
            )
        })?;
        array = if parent.null_count() > 0 {
            let mask = is_null(parent).map_err(arrow_key_err)?;
            nullif(child.as_ref(), &mask).map_err(arrow_key_err)?
        } else {
            child.clone()
        };
    }
    Ok(array)
}

pub(super) fn sort_source_path(
    schema: &IcebergSchema,
    source_id: i32,
) -> Option<(String, Vec<String>)> {
    let mut stack: Vec<(Vec<String>, &[NestedFieldRef])> =
        vec![(Vec::new(), schema.as_struct().fields())];
    while let Some((prefix, fields)) = stack.pop() {
        for field in fields {
            if field.id == source_id {
                let mut full = prefix.clone();
                full.push(field.name.clone());
                let mut names = full.into_iter();
                return Some((names.next()?, names.collect()));
            }
            if let Type::Struct(inner) = field.field_type.as_ref() {
                let mut child_prefix = prefix.clone();
                child_prefix.push(field.name.clone());
                stack.push((child_prefix, inner.fields()));
            }
        }
    }
    None
}

fn encode_sort_field(
    array: &ArrayRef,
    descending: bool,
    nulls_first: bool,
    keys: &mut [Vec<u8>],
) -> Result<()> {
    let mut scratch: Vec<u8> = Vec::with_capacity(16);
    for (row, key) in keys.iter_mut().enumerate() {
        if array.is_null(row) {
            key.push(u8::from(!nulls_first));
            continue;
        }
        key.push(u8::from(nulls_first));
        scratch.clear();
        encode_value(array, row, &mut scratch)?;
        if descending {
            key.extend(scratch.iter().map(|byte| !byte));
        } else {
            key.extend_from_slice(&scratch);
        }
    }
    Ok(())
}

fn encode_value(array: &ArrayRef, row: usize, out: &mut Vec<u8>) -> Result<()> {
    match array.data_type() {
        DataType::Boolean => out.push(u8::from(array.as_boolean().value(row))),
        DataType::Int8 => push_int(out, i64::from(array.as_primitive::<Int8Type>().value(row))),
        DataType::Int16 => push_int(out, i64::from(array.as_primitive::<Int16Type>().value(row))),
        DataType::Int32 => push_int(out, i64::from(array.as_primitive::<Int32Type>().value(row))),
        DataType::Date32 => push_int(
            out,
            i64::from(array.as_primitive::<Date32Type>().value(row)),
        ),
        DataType::Int64 => push_int(out, array.as_primitive::<Int64Type>().value(row)),
        DataType::Time64(TimeUnit::Microsecond) => push_int(
            out,
            array.as_primitive::<Time64MicrosecondType>().value(row),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => push_int(
            out,
            array.as_primitive::<TimestampMicrosecondType>().value(row),
        ),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => push_int(
            out,
            array.as_primitive::<TimestampNanosecondType>().value(row),
        ),
        DataType::Float32 => {
            let value = array.as_primitive::<Float32Type>().value(row);
            push_float(
                out,
                f64::from(if value.is_nan() { f32::NAN } else { value }),
            );
        }
        DataType::Float64 => {
            let value = array.as_primitive::<Float64Type>().value(row);
            push_float(out, if value.is_nan() { f64::NAN } else { value });
        }
        DataType::Decimal128(_, _) => {
            let value = array.as_primitive::<Decimal128Type>().value(row);
            let mut bytes = value.to_be_bytes();
            bytes[0] ^= 0x80;
            out.extend_from_slice(&bytes);
        }
        DataType::Decimal256(_, _) => {
            let value = array.as_primitive::<Decimal256Type>().value(row);
            let mut bytes = value.to_be_bytes();
            bytes.reverse();
            bytes[0] ^= 0x80;
            out.extend_from_slice(&bytes);
        }
        DataType::Utf8 => push_escaped(out, array.as_string::<i32>().value(row).as_bytes()),
        DataType::LargeUtf8 => push_escaped(out, array.as_string::<i64>().value(row).as_bytes()),
        DataType::Utf8View => push_escaped(out, array.as_string_view().value(row).as_bytes()),
        DataType::Binary => push_escaped(out, array.as_binary::<i32>().value(row)),
        DataType::LargeBinary => push_escaped(out, array.as_binary::<i64>().value(row)),
        DataType::BinaryView => push_escaped(out, array.as_binary_view().value(row)),
        DataType::FixedSizeBinary(_) => {
            let values = array
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "fixed-size binary sort key is not a FixedSizeBinaryArray",
                    )
                })?;
            push_escaped(out, values.value(row));
        }
        other => {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("Cannot sort a rewrite by a column of arrow type {other}"),
            ));
        }
    }
    Ok(())
}

fn push_int(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&(value ^ i64::MIN).to_be_bytes());
}

fn push_float(out: &mut Vec<u8>, value: f64) {
    let bits = value.to_bits() as i64;
    let ordered = if bits < 0 {
        !(bits as u64)
    } else {
        (bits as u64) ^ (1u64 << 63)
    };
    out.extend_from_slice(&ordered.to_be_bytes());
}

fn push_escaped(out: &mut Vec<u8>, bytes: &[u8]) {
    for &byte in bytes {
        out.push(byte);
        if byte == 0 {
            out.push(0xFF);
        }
    }
    out.extend_from_slice(&[0, 0]);
}

fn arrow_key_err(error: arrow_schema::ArrowError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Failed to read a sort key column of the rewritten rows",
    )
    .with_source(error)
}
