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

use super::encode::{
    OrcCompression, boolean_rle, compress_chunks, put_unbounded_varint_i128, rle_v1_signed,
    rle_v1_unsigned,
};
use super::footer_write::{EncodingKind, StreamKind, StreamRecord};
use super::orc_type::{ColumnShape, OrcSchema};
use crate::spec::{Literal, PrimitiveLiteral, PrimitiveType, Type};
use crate::{Error, ErrorKind, Result};

pub(crate) const ORC_EPOCH_UTC_SECONDS: i64 = 1_420_070_400;

#[derive(Debug, Default)]
struct ColumnBuffers {
    present: Vec<bool>,
    has_null: bool,
    bools: Vec<bool>,
    ints: Vec<i64>,
    secondary: Vec<i64>,
    lengths: Vec<i64>,
    bytes: Vec<u8>,
}

impl ColumnBuffers {
    fn estimated_size(&self) -> usize {
        self.present.len() / 8
            + self.bools.len() / 8
            + self.ints.len() * 2
            + self.secondary.len() * 2
            + self.lengths.len() * 2
            + self.bytes.len()
    }

    fn clear(&mut self) {
        self.present.clear();
        self.has_null = false;
        self.bools.clear();
        self.ints.clear();
        self.secondary.clear();
        self.lengths.clear();
        self.bytes.clear();
    }
}

#[derive(Debug)]
pub(crate) struct StripeEncoder {
    buffers: Vec<ColumnBuffers>,
    rows: u64,
}

impl StripeEncoder {
    pub(crate) fn new(schema: &OrcSchema) -> Self {
        StripeEncoder {
            buffers: (0..schema.columns.len())
                .map(|_| ColumnBuffers::default())
                .collect(),
            rows: 0,
        }
    }

    pub(crate) fn rows(&self) -> u64 {
        self.rows
    }

    pub(crate) fn estimated_size(&self) -> usize {
        self.buffers.iter().map(ColumnBuffers::estimated_size).sum()
    }

    pub(crate) fn append_row(&mut self, schema: &OrcSchema, row: Option<&Literal>) -> Result<()> {
        append(&mut self.buffers, schema, 0, row)?;
        self.rows += 1;
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        schema: &OrcSchema,
        codec: OrcCompression,
        block_size: usize,
    ) -> Result<FinishedStripe> {
        let mut data = Vec::new();
        let mut streams = Vec::new();
        for (index, buffers) in self.buffers.iter().enumerate() {
            for (kind, payload) in column_streams(schema.columns[index].shape, buffers) {
                let compressed = compress_chunks(&payload, codec, block_size)?;
                streams.push(StreamRecord {
                    kind,
                    column: u32::try_from(index).map_err(|_| too_many_columns())?,
                    length: compressed.len() as u64,
                });
                data.extend_from_slice(&compressed);
            }
        }
        let encodings = vec![EncodingKind::Direct; self.buffers.len()];
        let rows = self.rows;
        for buffers in &mut self.buffers {
            buffers.clear();
        }
        self.rows = 0;
        Ok(FinishedStripe {
            data,
            streams,
            encodings,
            rows,
        })
    }
}

pub(crate) struct FinishedStripe {
    pub(crate) data: Vec<u8>,
    pub(crate) streams: Vec<StreamRecord>,
    pub(crate) encodings: Vec<EncodingKind>,
    pub(crate) rows: u64,
}

fn too_many_columns() -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "The ORC schema has more columns than the stripe footer can address",
    )
}

fn column_streams(shape: ColumnShape, buffers: &ColumnBuffers) -> Vec<(StreamKind, Vec<u8>)> {
    let mut streams = Vec::with_capacity(3);
    if buffers.has_null {
        streams.push((StreamKind::Present, boolean_rle(&buffers.present)));
    }
    match shape {
        ColumnShape::Struct => {}
        ColumnShape::Boolean => streams.push((StreamKind::Data, boolean_rle(&buffers.bools))),
        ColumnShape::Int | ColumnShape::Long | ColumnShape::Date => {
            streams.push((StreamKind::Data, rle_v1_signed(&buffers.ints)));
        }
        ColumnShape::Float | ColumnShape::Double => {
            streams.push((StreamKind::Data, buffers.bytes.clone()));
        }
        ColumnShape::Timestamp => {
            streams.push((StreamKind::Data, rle_v1_signed(&buffers.ints)));
            streams.push((StreamKind::Secondary, rle_v1_unsigned(&buffers.secondary)));
        }
        ColumnShape::Decimal => {
            streams.push((StreamKind::Data, buffers.bytes.clone()));
            streams.push((StreamKind::Secondary, rle_v1_signed(&buffers.secondary)));
        }
        ColumnShape::Utf8 | ColumnShape::Bytes => {
            streams.push((StreamKind::Data, buffers.bytes.clone()));
            streams.push((StreamKind::Length, rle_v1_unsigned(&buffers.lengths)));
        }
        ColumnShape::List | ColumnShape::Map => {
            streams.push((StreamKind::Length, rle_v1_unsigned(&buffers.lengths)));
        }
    }
    streams
}

fn append(
    buffers: &mut [ColumnBuffers],
    schema: &OrcSchema,
    index: usize,
    value: Option<&Literal>,
) -> Result<()> {
    let Some(value) = value else {
        buffers[index].present.push(false);
        buffers[index].has_null = true;
        return Ok(());
    };
    buffers[index].present.push(true);

    let column = &schema.columns[index];
    match (column.shape, value) {
        (ColumnShape::Struct, Literal::Struct(fields)) => {
            let children = column.children.clone();
            let values: Vec<Option<Literal>> = fields.iter().map(|f| f.cloned()).collect();
            if values.len() != children.len() {
                return Err(shape_mismatch("struct", children.len(), values.len()));
            }
            for (child, child_value) in children.iter().zip(values.iter()) {
                append(buffers, schema, *child, child_value.as_ref())?;
            }
            Ok(())
        }
        (ColumnShape::List, Literal::List(elements)) => {
            buffers[index].lengths.push(elements.len() as i64);
            let child = column.children[0];
            for element in elements {
                append(buffers, schema, child, element.as_ref())?;
            }
            Ok(())
        }
        (ColumnShape::Map, Literal::Map(entries)) => {
            let pairs = entries.pairs();
            buffers[index].lengths.push(pairs.len() as i64);
            let key_column = column.children[0];
            let value_column = column.children[1];
            for (key, mapped) in pairs {
                append(buffers, schema, key_column, Some(key))?;
                append(buffers, schema, value_column, mapped.as_ref())?;
            }
            Ok(())
        }
        (_, Literal::Primitive(primitive)) => {
            append_primitive(&mut buffers[index], column.shape, column, primitive)
        }
        (shape, other) => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("An ORC {shape:?} column received a {other:?} value"),
        )),
    }
}

fn shape_mismatch(what: &str, expected: usize, actual: usize) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("An ORC {what} column expected {expected} fields but the row carried {actual}"),
    )
}

fn append_primitive(
    buffers: &mut ColumnBuffers,
    shape: ColumnShape,
    column: &super::orc_type::OrcColumn,
    primitive: &PrimitiveLiteral,
) -> Result<()> {
    match (shape, primitive) {
        (ColumnShape::Boolean, PrimitiveLiteral::Boolean(value)) => {
            buffers.bools.push(*value);
            Ok(())
        }
        (ColumnShape::Int | ColumnShape::Date, PrimitiveLiteral::Int(value)) => {
            buffers.ints.push(i64::from(*value));
            Ok(())
        }
        (ColumnShape::Long, PrimitiveLiteral::Long(value)) => {
            buffers.ints.push(*value);
            Ok(())
        }
        (ColumnShape::Float, PrimitiveLiteral::Float(value)) => {
            buffers.bytes.extend_from_slice(&value.0.to_le_bytes());
            Ok(())
        }
        (ColumnShape::Double, PrimitiveLiteral::Double(value)) => {
            buffers.bytes.extend_from_slice(&value.0.to_le_bytes());
            Ok(())
        }
        (ColumnShape::Timestamp, PrimitiveLiteral::Long(value)) => {
            let (seconds, nanos) = split_timestamp(*value, timestamp_nanos_per_unit(column))?;
            buffers.ints.push(seconds);
            buffers.secondary.push(nanos);
            Ok(())
        }
        (ColumnShape::Decimal, PrimitiveLiteral::Int128(value)) => {
            put_unbounded_varint_i128(&mut buffers.bytes, *value);
            buffers.secondary.push(i64::from(decimal_scale(column)?));
            Ok(())
        }
        (ColumnShape::Utf8, PrimitiveLiteral::String(value)) => {
            buffers.bytes.extend_from_slice(value.as_bytes());
            buffers.lengths.push(value.len() as i64);
            Ok(())
        }
        (ColumnShape::Bytes, PrimitiveLiteral::Binary(value)) => {
            buffers.bytes.extend_from_slice(value);
            buffers.lengths.push(value.len() as i64);
            Ok(())
        }
        (ColumnShape::Bytes, PrimitiveLiteral::UInt128(value)) => {
            buffers.bytes.extend_from_slice(&value.to_be_bytes());
            buffers.lengths.push(16);
            Ok(())
        }
        (shape, other) => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("An ORC {shape:?} column received the primitive literal {other:?}"),
        )),
    }
}

fn timestamp_nanos_per_unit(column: &super::orc_type::OrcColumn) -> i64 {
    match column.iceberg_type.as_ref() {
        Some(Type::Primitive(PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs)) => 1,
        _ => 1_000,
    }
}

fn decimal_scale(column: &super::orc_type::OrcColumn) -> Result<u32> {
    match column.iceberg_type.as_ref() {
        Some(Type::Primitive(PrimitiveType::Decimal { scale, .. })) => Ok(*scale),
        _ => Err(Error::new(
            ErrorKind::Unexpected,
            "An ORC decimal column carries no Iceberg decimal type",
        )),
    }
}

pub(crate) fn split_timestamp(value: i64, nanos_per_unit: i64) -> Result<(i64, i64)> {
    let total_nanos = i128::from(value) * i128::from(nanos_per_unit);
    let floor_seconds = total_nanos.div_euclid(1_000_000_000);
    let nanos = total_nanos.rem_euclid(1_000_000_000) as u64;
    let seconds = if floor_seconds < 0 && nanos > 999_999 {
        floor_seconds + 1
    } else {
        floor_seconds
    };
    let seconds = i64::try_from(seconds - i128::from(ORC_EPOCH_UTC_SECONDS)).map_err(|_| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Timestamp {value} is outside the range an ORC timestamp column can carry"),
        )
    })?;
    Ok((seconds, encode_nanos(nanos) as i64))
}

pub(crate) fn encode_nanos(nanos: u64) -> u64 {
    if nanos == 0 {
        return 0;
    }
    if !nanos.is_multiple_of(100) {
        return nanos << 3;
    }
    let mut trimmed = nanos / 100;
    let mut zeros = 1u64;
    while trimmed.is_multiple_of(10) && zeros < 7 {
        trimmed /= 10;
        zeros += 1;
    }
    (trimmed << 3) | zeros
}

#[cfg(test)]
mod tests {
    include!("column_tests.rs");
}
