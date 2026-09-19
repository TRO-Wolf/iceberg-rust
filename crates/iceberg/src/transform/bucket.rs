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
use arrow_schema::{DataType, TimeUnit};

use super::TransformFunction;
use crate::spec::{Datum, PrimitiveLiteral, PrimitiveType};

#[derive(Debug)]
pub struct Bucket {
    /// Number of buckets, proven at construction to lie in `1..=i32::MAX` (the Java `int`
    /// contract — `Bucket.get(int)`, Bucket.java:41-42 in 1.10.0), so the modulo in
    /// [`Bucket::bucket_n`] can never divide by zero or by a wrapped-negative count.
    mod_n: i32,
}

impl Bucket {
    /// Creates a bucket transform function with `mod_n` buckets.
    ///
    /// Rejects `mod_n` outside `1..=i32::MAX` with [`crate::ErrorKind::DataInvalid`] —
    /// a defense-in-depth guard independent of the parse-time bound in
    /// `Transform::validate` (Java parity: `Preconditions.checkArgument(numBuckets > 0,
    /// "Invalid number of buckets: %s (must be > 0)")`, Bucket.java:41-42; counts above
    /// `i32::MAX` are unrepresentable in Java's `int` and pre-fix wrapped negative here,
    /// producing silently WRONG bucket values).
    pub fn new(mod_n: u32) -> crate::Result<Self> {
        if mod_n == 0 {
            return Err(crate::Error::new(
                crate::ErrorKind::DataInvalid,
                "Invalid number of buckets: 0 (must be > 0)",
            ));
        }
        let mod_n = i32::try_from(mod_n).map_err(|_| {
            crate::Error::new(
                crate::ErrorKind::DataInvalid,
                format!(
                    "Invalid number of buckets: {mod_n} (must be <= {}, the Java int maximum)",
                    i32::MAX
                ),
            )
        })?;
        Ok(Self { mod_n })
    }
}

impl Bucket {
    /// When switch the hash function, we only need to change this function.
    ///
    /// The `unwrap` cannot fire: `murmur3_32` reads through `std::io::Read`, and the only
    /// error it can propagate is one from the reader — here `&[u8]`, whose `Read` impl copies
    /// out of a slice and is total (it returns `Ok` for every call, `Ok(0)` at the end).
    /// Making this fallible would mean threading a `Result` through all eight `bucket_*`
    /// helpers for a branch nothing can reach; the invariant is documented instead.
    ///
    /// The `as i32` is a deliberate two's-complement reinterpretation, not a lossy conversion:
    /// the spec's `bucket_N(x) = (murmur3_x86_32_hash(x) & Integer.MAX_VALUE) % N` is defined
    /// over Java's SIGNED 32-bit hash, so the sign bit has to survive.
    #[inline]
    fn hash_bytes(mut v: &[u8]) -> i32 {
        murmur3::murmur3_32(&mut v, 0).unwrap() as i32
    }

    #[inline]
    fn hash_int(v: i32) -> i32 {
        Self::hash_long(v as i64)
    }

    #[inline]
    fn hash_long(v: i64) -> i32 {
        Self::hash_bytes(v.to_le_bytes().as_slice())
    }

    /// v is days from unix epoch
    #[inline]
    fn hash_date(v: i32) -> i32 {
        Self::hash_int(v)
    }

    /// v is microseconds from midnight
    #[inline]
    fn hash_time(v: i64) -> i32 {
        Self::hash_long(v)
    }

    /// v is microseconds from unix epoch
    #[inline]
    fn hash_timestamp(v: i64) -> i32 {
        Self::hash_long(v)
    }

    #[inline]
    fn hash_str(s: &str) -> i32 {
        Self::hash_bytes(s.as_bytes())
    }

    /// Decimal values are hashed using the minimum number of bytes required to hold the unscaled value as a two’s complement big-endian
    /// ref: https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements
    #[inline]
    fn hash_decimal(v: i128) -> i32 {
        if v == 0 {
            return Self::hash_bytes(&[0]);
        }

        let bytes = v.to_be_bytes();
        let start = if v > 0 {
            // Positive: skip 0x00 unless next byte would appear negative
            bytes
                .windows(2)
                .position(|w| w[0] != 0x00 || w[1] & 0x80 != 0)
                .unwrap_or(15)
        } else {
            // Negative: skip 0xFF only if next byte stays negative
            bytes
                .windows(2)
                .position(|w| w[0] != 0xFF || w[1] & 0x80 == 0)
                .unwrap_or(15)
        };

        Self::hash_bytes(&bytes[start..])
    }

    /// def bucket_N(x) = (murmur3_x86_32_hash(x) & Integer.MAX_VALUE) % N
    /// ref: https://iceberg.apache.org/spec/#partitioning
    ///
    /// `self.mod_n` is positive by construction ([`Bucket::new`]), so the modulo can
    /// neither panic (÷0) nor wrap negative.
    #[inline]
    fn bucket_n(&self, v: i32) -> i32 {
        (v & i32::MAX) % self.mod_n
    }

    #[inline]
    fn bucket_int(&self, v: i32) -> i32 {
        self.bucket_n(Self::hash_int(v))
    }

    #[inline]
    fn bucket_long(&self, v: i64) -> i32 {
        self.bucket_n(Self::hash_long(v))
    }

    #[inline]
    fn bucket_decimal(&self, v: i128) -> i32 {
        self.bucket_n(Self::hash_decimal(v))
    }

    #[inline]
    fn bucket_date(&self, v: i32) -> i32 {
        self.bucket_n(Self::hash_date(v))
    }

    #[inline]
    fn bucket_time(&self, v: i64) -> i32 {
        self.bucket_n(Self::hash_time(v))
    }

    #[inline]
    fn bucket_timestamp(&self, v: i64) -> i32 {
        self.bucket_n(Self::hash_timestamp(v))
    }

    #[inline]
    fn bucket_str(&self, v: &str) -> i32 {
        self.bucket_n(Self::hash_str(v))
    }

    #[inline]
    fn bucket_bytes(&self, v: &[u8]) -> i32 {
        self.bucket_n(Self::hash_bytes(v))
    }

    fn bucket_with<'a, V: ?Sized + 'a, A: arrow_array::Array>(
        &self,
        array: &'a A,
        value: impl Fn(usize) -> &'a V,
        bucket: impl Fn(&V) -> i32,
    ) -> arrow_array::Int32Array {
        let mut values = Vec::with_capacity(array.len());
        for i in 0..array.len() {
            values.push(if array.is_valid(i) {
                bucket(value(i))
            } else {
                0
            });
        }
        arrow_array::Int32Array::new(values.into(), array.nulls().cloned())
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
        crate::Error::new(
            crate::ErrorKind::DataInvalid,
            format!(
                "Array with data type {:?} is not a {} and cannot be bucket-transformed",
                input.data_type(),
                std::any::type_name::<T>()
            ),
        )
    })
}

impl TransformFunction for Bucket {
    fn transform(&self, input: ArrayRef) -> crate::Result<ArrayRef> {
        let res: arrow_array::Int32Array = match input.data_type() {
            DataType::Int32 => {
                downcast_input::<arrow_array::Int32Array>(&input)?.unary(|v| self.bucket_int(v))
            }
            DataType::Int64 => {
                downcast_input::<arrow_array::Int64Array>(&input)?.unary(|v| self.bucket_long(v))
            }
            DataType::Decimal128(_, _) => downcast_input::<arrow_array::Decimal128Array>(&input)?
                .unary(|v| self.bucket_decimal(v)),
            DataType::Date32 => {
                downcast_input::<arrow_array::Date32Array>(&input)?.unary(|v| self.bucket_date(v))
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                downcast_input::<arrow_array::Time64MicrosecondArray>(&input)?
                    .unary(|v| self.bucket_time(v))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                downcast_input::<arrow_array::TimestampMicrosecondArray>(&input)?
                    .unary(|v| self.bucket_timestamp(v))
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                downcast_input::<arrow_array::Time64NanosecondArray>(&input)?
                    .unary(|v| self.bucket_time(v / 1000))
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                downcast_input::<arrow_array::TimestampNanosecondArray>(&input)?
                    .unary(|v| self.bucket_timestamp(v / 1000))
            }
            DataType::Utf8 => {
                let array = downcast_input::<arrow_array::StringArray>(&input)?;
                self.bucket_with(array, |i| array.value(i), |v| self.bucket_str(v))
            }
            DataType::LargeUtf8 => {
                let array = downcast_input::<arrow_array::LargeStringArray>(&input)?;
                self.bucket_with(array, |i| array.value(i), |v| self.bucket_str(v))
            }
            DataType::Binary => {
                let array = downcast_input::<arrow_array::BinaryArray>(&input)?;
                self.bucket_with(array, |i| array.value(i), |v| self.bucket_bytes(v))
            }
            DataType::LargeBinary => {
                let array = downcast_input::<arrow_array::LargeBinaryArray>(&input)?;
                self.bucket_with(array, |i| array.value(i), |v| self.bucket_bytes(v))
            }
            DataType::FixedSizeBinary(_) => {
                let array = downcast_input::<arrow_array::FixedSizeBinaryArray>(&input)?;
                self.bucket_with(array, |i| array.value(i), |v| self.bucket_bytes(v))
            }
            DataType::BinaryView => {
                let array = downcast_input::<arrow_array::BinaryViewArray>(&input)?;
                self.bucket_with(array, |i| array.value(i), |v| self.bucket_bytes(v))
            }
            DataType::Utf8View => {
                let array = downcast_input::<arrow_array::StringViewArray>(&input)?;
                self.bucket_with(array, |i| array.value(i), |v| self.bucket_str(v))
            }
            _ => {
                return Err(crate::Error::new(
                    crate::ErrorKind::FeatureUnsupported,
                    format!(
                        "Unsupported data type for bucket transform: {:?}",
                        input.data_type()
                    ),
                ));
            }
        };
        Ok(Arc::new(res))
    }

    fn transform_literal(&self, input: &Datum) -> crate::Result<Option<Datum>> {
        let val = match (input.data_type(), input.literal()) {
            (PrimitiveType::Int, PrimitiveLiteral::Int(v)) => self.bucket_int(*v),
            (PrimitiveType::Long, PrimitiveLiteral::Long(v)) => self.bucket_long(*v),
            (PrimitiveType::Decimal { .. }, PrimitiveLiteral::Int128(v)) => self.bucket_decimal(*v),
            (PrimitiveType::Date, PrimitiveLiteral::Int(v)) => self.bucket_date(*v),
            (PrimitiveType::Time, PrimitiveLiteral::Long(v)) => self.bucket_time(*v),
            (PrimitiveType::Timestamp, PrimitiveLiteral::Long(v)) => self.bucket_timestamp(*v),
            (PrimitiveType::Timestamptz, PrimitiveLiteral::Long(v)) => self.bucket_timestamp(*v),
            (PrimitiveType::TimestampNs, PrimitiveLiteral::Long(v)) => {
                self.bucket_timestamp(*v / 1000)
            }
            (PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(v)) => {
                self.bucket_timestamp(*v / 1000)
            }
            (PrimitiveType::String, PrimitiveLiteral::String(v)) => self.bucket_str(v.as_str()),
            (PrimitiveType::Uuid, PrimitiveLiteral::UInt128(v)) => {
                self.bucket_bytes(uuid::Uuid::from_u128(*v).as_ref())
            }
            (PrimitiveType::Binary, PrimitiveLiteral::Binary(v)) => self.bucket_bytes(v.as_ref()),
            (PrimitiveType::Fixed(_), PrimitiveLiteral::Binary(v)) => self.bucket_bytes(v.as_ref()),
            _ => {
                return Err(crate::Error::new(
                    crate::ErrorKind::FeatureUnsupported,
                    format!(
                        "Unsupported data type for bucket transform: {:?}",
                        input.data_type()
                    ),
                ));
            }
        };
        Ok(Some(Datum::int(val)))
    }
}

#[cfg(test)]
#[path = "bucket_tests.rs"]
mod bucket_tests;
