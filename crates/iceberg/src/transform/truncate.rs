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

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::DataType;

use super::TransformFunction;
use crate::Error;
use crate::spec::decimal_utils::decimal_from_i128_with_scale;
use crate::spec::{Datum, PrimitiveLiteral, PrimitiveType};

#[derive(Debug)]
pub struct Truncate {
    /// Truncation width, proven at construction to lie in `1..=i32::MAX` (the Java `int`
    /// contract — `Truncate.get(int)`, Truncate.java:42 in 1.10.0), so the modulo in
    /// `truncate_i32`/`truncate_i64`/`truncate_decimal_i128` can never divide by zero.
    width: u32,
}

impl Truncate {
    /// Creates a truncate transform function with the given width.
    ///
    /// Rejects `width` outside `1..=i32::MAX` with [`crate::ErrorKind::DataInvalid`] —
    /// a defense-in-depth guard independent of the parse-time bound in
    /// `Transform::validate` (Java parity: `Preconditions.checkArgument(width > 0,
    /// "Invalid truncate width: %s (must be > 0)")`, Truncate.java:42; widths above
    /// `i32::MAX` are unrepresentable in Java's `int`).
    pub fn new(width: u32) -> crate::Result<Self> {
        if width == 0 {
            return Err(Error::new(
                crate::ErrorKind::DataInvalid,
                "Invalid truncate width: 0 (must be > 0)",
            ));
        }
        if i32::try_from(width).is_err() {
            return Err(Error::new(
                crate::ErrorKind::DataInvalid,
                format!(
                    "Invalid truncate width: {width} (must be <= {}, the Java int maximum)",
                    i32::MAX
                ),
            ));
        }
        Ok(Self { width })
    }

    #[inline]
    fn truncate_str(s: &str, width: usize) -> &str {
        match s.char_indices().nth(width) {
            None => s,
            Some((idx, _)) => &s[..idx],
        }
    }

    #[inline]
    fn truncate_binary(s: &[u8], width: usize) -> &[u8] {
        if s.len() > width { &s[0..width] } else { s }
    }

    #[inline]
    fn truncate_i32(v: i32, width: i32) -> i32 {
        v - v.rem_euclid(width)
    }

    #[inline]
    fn truncate_i64(v: i64, width: i64) -> i64 {
        v - (((v % width) + width) % width)
    }

    #[inline]
    fn truncate_decimal_i128(v: i128, width: i128) -> i128 {
        v - (((v % width) + width) % width)
    }
}

/// Downcast a transform input to the concrete Arrow array its [`DataType`] implies, or return a
/// typed error naming both.
///
/// Every caller sits in an arm of a `match input.data_type()`, and for arrow's own arrays the
/// pairing is exact: `arrow_array::make_array` maps each `DataType` to exactly one array struct,
/// so no arrow-native value can take an arm whose concrete type it is not. That is a property of
/// the VALUES arrow builds, though, not of this function's signature: `transform` accepts
/// `Arc<dyn Array>`, and `Array` is a public trait, so the pairing is not something this crate
/// can enforce. Answering an unexpected implementation with a typed error rather than a panic
/// costs nothing and keeps a library call from aborting its caller's process.
fn downcast_input<T: 'static>(input: &ArrayRef) -> crate::Result<&T> {
    input.as_any().downcast_ref::<T>().ok_or_else(|| {
        Error::new(
            crate::ErrorKind::DataInvalid,
            format!(
                "Array with data type {:?} is not a {} and cannot be truncate-transformed",
                input.data_type(),
                std::any::type_name::<T>()
            ),
        )
    })
}

impl TransformFunction for Truncate {
    fn transform(&self, input: ArrayRef) -> crate::Result<ArrayRef> {
        match input.data_type() {
            DataType::Int32 => {
                let width: i32 = self.width.try_into().map_err(|_| {
                    Error::new(
                        crate::ErrorKind::DataInvalid,
                        "width is failed to convert to i32 when truncate Int32Array",
                    )
                })?;
                let res: arrow_array::Int32Array =
                    downcast_input::<arrow_array::Int32Array>(&input)?
                        .unary(|v| Self::truncate_i32(v, width));
                Ok(Arc::new(res))
            }
            DataType::Int64 => {
                let width = self.width as i64;
                let res: arrow_array::Int64Array =
                    downcast_input::<arrow_array::Int64Array>(&input)?
                        .unary(|v| Self::truncate_i64(v, width));
                Ok(Arc::new(res))
            }
            DataType::Decimal128(precision, scale) => {
                let width = self.width as i128;
                let decimals = downcast_input::<arrow_array::Decimal128Array>(&input)?;
                let res: arrow_array::Decimal128Array = decimals
                    .unary(|v| Self::truncate_decimal_i128(v, width))
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|err| Error::new(crate::ErrorKind::Unexpected, format!("{err}")))?;
                Ok(Arc::new(res))
            }
            DataType::Utf8 => {
                let len = self.width as usize;
                let res: arrow_array::StringArray = arrow_array::StringArray::from_iter(
                    downcast_input::<arrow_array::StringArray>(&input)?
                        .iter()
                        .map(|v| v.map(|v| Self::truncate_str(v, len))),
                );
                Ok(Arc::new(res))
            }
            DataType::LargeUtf8 => {
                let len = self.width as usize;
                let res: arrow_array::LargeStringArray = arrow_array::LargeStringArray::from_iter(
                    downcast_input::<arrow_array::LargeStringArray>(&input)?
                        .iter()
                        .map(|v| v.map(|v| Self::truncate_str(v, len))),
                );
                Ok(Arc::new(res))
            }
            DataType::Binary => {
                let len = self.width as usize;
                let res: arrow_array::BinaryArray = arrow_array::BinaryArray::from_iter(
                    downcast_input::<arrow_array::BinaryArray>(&input)?
                        .iter()
                        .map(|v| v.map(|v| Self::truncate_binary(v, len))),
                );
                Ok(Arc::new(res))
            }
            DataType::LargeBinary => {
                let len = self.width as usize;
                let res: arrow_array::LargeBinaryArray = arrow_array::LargeBinaryArray::from_iter(
                    downcast_input::<arrow_array::LargeBinaryArray>(&input)?
                        .iter()
                        .map(|v| v.map(|v| Self::truncate_binary(v, len))),
                );
                Ok(Arc::new(res))
            }
            DataType::BinaryView => {
                let len = self.width as usize;
                let res: arrow_array::BinaryViewArray = arrow_array::BinaryViewArray::from_iter(
                    downcast_input::<arrow_array::BinaryViewArray>(&input)?
                        .iter()
                        .map(|v| v.map(|v| Self::truncate_binary(v, len))),
                );
                Ok(Arc::new(res))
            }
            DataType::Utf8View => {
                let len = self.width as usize;
                let res: arrow_array::StringViewArray = arrow_array::StringViewArray::from_iter(
                    downcast_input::<arrow_array::StringViewArray>(&input)?
                        .iter()
                        .map(|v| v.map(|v| Self::truncate_str(v, len))),
                );
                Ok(Arc::new(res))
            }
            _ => Err(crate::Error::new(
                crate::ErrorKind::FeatureUnsupported,
                format!(
                    "Unsupported data type for truncate transform: {:?}",
                    input.data_type()
                ),
            )),
        }
    }

    fn transform_literal(&self, input: &Datum) -> crate::Result<Option<Datum>> {
        match input.literal() {
            PrimitiveLiteral::Int(v) => Ok(Some({
                let width: i32 = self.width.try_into().map_err(|_| {
                    Error::new(
                        crate::ErrorKind::DataInvalid,
                        "width is failed to convert to i32 when truncate Int32Array",
                    )
                })?;
                Datum::int(Self::truncate_i32(*v, width))
            })),
            PrimitiveLiteral::Long(v) => Ok(Some({
                let width = self.width as i64;
                Datum::long(Self::truncate_i64(*v, width))
            })),
            PrimitiveLiteral::Int128(v) => Ok(Some({
                let width = self.width as i128;
                Datum::decimal(decimal_from_i128_with_scale(
                    Self::truncate_decimal_i128(*v, width),
                    0,
                ))?
            })),
            PrimitiveLiteral::String(v) => Ok(Some({
                let len = self.width as usize;
                Datum::string(Self::truncate_str(v, len).to_string())
            })),
            PrimitiveLiteral::Binary(v) if matches!(input.data_type(), PrimitiveType::Binary) => {
                let len = self.width as usize;
                Ok(Some(Datum::binary(Self::truncate_binary(v, len).to_vec())))
            }
            _ => Err(crate::Error::new(
                crate::ErrorKind::FeatureUnsupported,
                format!(
                    "Unsupported data type for truncate transform: {:?}",
                    input.data_type()
                ),
            )),
        }
    }
}

#[cfg(test)]
#[path = "truncate_tests.rs"]
mod truncate_tests;
