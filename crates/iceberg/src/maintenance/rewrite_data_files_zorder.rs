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

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int32Type, Int64Type, Time64MicrosecondType,
    TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::Schema as ArrowSchema;
use uuid::Uuid;

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{PrimitiveType, Schema as IcebergSchema, Type};

pub(super) const PRIMITIVE_BUFFER_SIZE: usize = 8;

enum ZColumnKind {
    WholeNumber,
    FloatingPoint,
    TimestampSeconds,
    Boolean,
    Text,
    Bytes,
    Uuid,
}

struct ZColumn {
    column: usize,
    kind: ZColumnKind,
    width: usize,
}

pub(super) struct ZOrderEncoder {
    columns: Vec<ZColumn>,
    output_size: usize,
}

impl ZOrderEncoder {
    pub(super) fn build(
        names: &[String],
        iceberg_schema: &IcebergSchema,
        arrow_schema: &ArrowSchema,
        var_length_contribution: usize,
        max_output_size: usize,
    ) -> Result<ZOrderEncoder> {
        let mut columns = Vec::with_capacity(names.len());
        let mut total = 0usize;
        for name in names {
            let field = iceberg_schema.field_by_name(name).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Cannot find column '{name}' in table schema"),
                )
            })?;
            let (kind, width) = match field.field_type.as_ref() {
                Type::Primitive(primitive) => match primitive {
                    PrimitiveType::Int | PrimitiveType::Long | PrimitiveType::Date => {
                        (ZColumnKind::WholeNumber, PRIMITIVE_BUFFER_SIZE)
                    }
                    PrimitiveType::Time | PrimitiveType::Timestamp => {
                        (ZColumnKind::WholeNumber, PRIMITIVE_BUFFER_SIZE)
                    }
                    PrimitiveType::Timestamptz => {
                        (ZColumnKind::TimestampSeconds, PRIMITIVE_BUFFER_SIZE)
                    }
                    PrimitiveType::Float | PrimitiveType::Double => {
                        (ZColumnKind::FloatingPoint, PRIMITIVE_BUFFER_SIZE)
                    }
                    PrimitiveType::Boolean => (ZColumnKind::Boolean, PRIMITIVE_BUFFER_SIZE),
                    PrimitiveType::String => (ZColumnKind::Text, var_length_contribution),
                    PrimitiveType::Uuid => (ZColumnKind::Uuid, var_length_contribution),
                    PrimitiveType::Binary | PrimitiveType::Fixed(_) => {
                        (ZColumnKind::Bytes, var_length_contribution)
                    }
                    unsupported => {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Cannot use column {name} of type {unsupported} in ZOrdering, the type is unsupported"
                            ),
                        ));
                    }
                },
                unsupported => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot use column {name} of type {unsupported} in ZOrdering, the type is unsupported"
                        ),
                    ));
                }
            };
            let column = arrow_schema.index_of(name).map_err(|error| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Z-order column '{name}' is not in the rewrite's batches"),
                )
                .with_source(error)
            })?;
            total = total.saturating_add(width).min(max_output_size);
            columns.push(ZColumn {
                column,
                kind,
                width,
            });
        }
        Ok(ZOrderEncoder {
            columns,
            output_size: total,
        })
    }

    pub(super) fn encode(&self, batch: &RecordBatch, keys: &mut [Vec<u8>]) -> Result<()> {
        let arrays: Vec<&ArrayRef> = self
            .columns
            .iter()
            .map(|column| batch.column(column.column))
            .collect();
        let mut buffers: Vec<Vec<u8>> = self
            .columns
            .iter()
            .map(|column| vec![0u8; column.width])
            .collect();
        for (row, key) in keys.iter_mut().enumerate() {
            for (index, column) in self.columns.iter().enumerate() {
                let buffer = &mut buffers[index];
                buffer.fill(0);
                if !arrays[index].is_null(row) {
                    encode_column(arrays[index], row, &column.kind, buffer)?;
                }
            }
            interleave_bits(&buffers, self.output_size, key);
        }
        Ok(())
    }
}

fn encode_column(
    array: &ArrayRef,
    row: usize,
    kind: &ZColumnKind,
    buffer: &mut [u8],
) -> Result<()> {
    match kind {
        ZColumnKind::WholeNumber => {
            let value = match array.data_type() {
                arrow_schema::DataType::Int32 => {
                    i64::from(array.as_primitive::<Int32Type>().value(row))
                }
                arrow_schema::DataType::Date32 => {
                    i64::from(array.as_primitive::<Date32Type>().value(row))
                }
                arrow_schema::DataType::Int64 => array.as_primitive::<Int64Type>().value(row),
                arrow_schema::DataType::Time64(_) => {
                    array.as_primitive::<Time64MicrosecondType>().value(row)
                }
                arrow_schema::DataType::Timestamp(_, _) => {
                    array.as_primitive::<TimestampMicrosecondType>().value(row)
                }
                other => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!("Z-order whole-number column has arrow type {other}"),
                    ));
                }
            };
            buffer.copy_from_slice(&whole_number_ordered_bytes(value));
        }
        ZColumnKind::TimestampSeconds => {
            let micros = array.as_primitive::<TimestampMicrosecondType>().value(row);
            buffer.copy_from_slice(&whole_number_ordered_bytes(micros.div_euclid(1_000_000)));
        }
        ZColumnKind::FloatingPoint => {
            let value = match array.data_type() {
                arrow_schema::DataType::Float32 => {
                    f64::from(array.as_primitive::<Float32Type>().value(row))
                }
                arrow_schema::DataType::Float64 => array.as_primitive::<Float64Type>().value(row),
                other => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!("Z-order floating-point column has arrow type {other}"),
                    ));
                }
            };
            buffer.copy_from_slice(&floating_point_ordered_bytes(value));
        }
        ZColumnKind::Boolean => {
            buffer[0] = if array.as_boolean().value(row) {
                0x81
            } else {
                0
            };
        }
        ZColumnKind::Text => {
            let text = match array.data_type() {
                arrow_schema::DataType::Utf8 => array.as_string::<i32>().value(row),
                arrow_schema::DataType::LargeUtf8 => array.as_string::<i64>().value(row),
                arrow_schema::DataType::Utf8View => array.as_string_view().value(row),
                other => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!("Z-order string column has arrow type {other}"),
                    ));
                }
            };
            string_to_ordered_bytes(text, buffer);
        }
        ZColumnKind::Uuid => {
            let bytes = fixed_bytes(array, row)?;
            let uuid = Uuid::from_slice(bytes).map_err(|error| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Z-order uuid column is not 16 bytes",
                )
                .with_source(error)
            })?;
            string_to_ordered_bytes(&uuid.hyphenated().to_string(), buffer);
        }
        ZColumnKind::Bytes => {
            let bytes = match array.data_type() {
                arrow_schema::DataType::Binary => array.as_binary::<i32>().value(row),
                arrow_schema::DataType::LargeBinary => array.as_binary::<i64>().value(row),
                arrow_schema::DataType::BinaryView => array.as_binary_view().value(row),
                arrow_schema::DataType::FixedSizeBinary(_) => fixed_bytes(array, row)?,
                other => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!("Z-order binary column has arrow type {other}"),
                    ));
                }
            };
            byte_truncate_or_fill(bytes, buffer);
        }
    }
    Ok(())
}

fn fixed_bytes(array: &ArrayRef, row: usize) -> Result<&[u8]> {
    let values = array
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Z-order fixed-width column is not a FixedSizeBinaryArray",
            )
        })?;
    Ok(values.value(row))
}

pub(super) fn whole_number_ordered_bytes(value: i64) -> [u8; PRIMITIVE_BUFFER_SIZE] {
    (value ^ i64::MIN).to_be_bytes()
}

pub(super) fn floating_point_ordered_bytes(value: f64) -> [u8; PRIMITIVE_BUFFER_SIZE] {
    let bits = value.to_bits() as i64;
    let mask = (bits >> 31) | i64::MIN;
    (bits ^ mask).to_be_bytes()
}

pub(super) fn string_to_ordered_bytes(value: &str, buffer: &mut [u8]) {
    buffer.fill(0);
    let mut end = 0usize;
    for (index, character) in value.char_indices() {
        let next = index + character.len_utf8();
        if next > buffer.len() {
            break;
        }
        end = next;
    }
    buffer[..end].copy_from_slice(&value.as_bytes()[..end]);
}

pub(super) fn byte_truncate_or_fill(value: &[u8], buffer: &mut [u8]) {
    buffer.fill(0);
    let width = buffer.len().min(value.len());
    buffer[..width].copy_from_slice(&value[..width]);
}

pub(super) fn interleave_bits(columns: &[Vec<u8>], output_size: usize, out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + output_size, 0);
    if columns.is_empty() || output_size == 0 {
        return;
    }
    let mut source_column = 0usize;
    let mut source_byte = 0usize;
    let mut source_bit = 7i32;
    let mut output_byte = 0usize;
    let mut output_bit = 7i32;
    while output_byte < output_size {
        let bits = columns[source_column][source_byte];
        let bit = (bits >> source_bit) & 1;
        out[start + output_byte] |= bit << output_bit;
        output_bit -= 1;
        if output_bit == -1 {
            output_byte += 1;
            output_bit = 7;
        }
        if output_byte == output_size {
            break;
        }
        let mut spins = 0usize;
        loop {
            source_column += 1;
            if source_column == columns.len() {
                source_column = 0;
                source_bit -= 1;
                if source_bit == -1 {
                    source_byte += 1;
                    source_bit = 7;
                }
            }
            if columns[source_column].len() > source_byte {
                break;
            }
            spins += 1;
            if spins > columns.len() {
                return;
            }
        }
    }
}
