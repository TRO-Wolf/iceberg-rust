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

//! Variant value and metadata serialization — the WRITE side. A port of the Java 1.10.0
//! construction and serialization surface. Each item below carries the rule it pins.
//!
//! | Java | Rust |
//! |---|---|
//! | `Variants.metadata(Collection)` | [`VariantMetadata::from_field_names`] |
//! | `PrimitiveWrapper.sizeInBytes`/`writeTo` | [`VariantValue::size_in_bytes`] / [`VariantValue::write_to`] |
//! | `ValueArray` | [`VariantArray::push`](crate::variant::VariantArray::push) |
//! | `ShreddedObject`, plain object core | [`VariantObjectBuilder`] |
//! | `Variants.of(...)` | the `VariantValue::of_*` constructors |
//! | `VariantMetadata.writeTo` / `Variant` emission | [`VariantMetadata::to_bytes`] / [`Variant::to_bytes`] |
//!
//! # Differences from Java on the write side
//!
//! | Difference | Effect |
//! |---|---|
//! | Re-serializing a PARSED value canonicalizes | Java copies the backing buffer verbatim. The eager form has no buffer, so widths become `sizeOf`-minimal and object field ids are re-resolved by name. Output stays byte-identical for anything Java's own writer produced. |
//! | Width overflow errors instead of masking | Java masks the value to the width. For more than 255 names whose UTF-8 data still fits one byte, it truncates the count and LOSES every name. This port returns [`DataInvalid`](crate::ErrorKind::DataInvalid). |
//! | Java's `int` domain is enforced | Every size, length and count is doored at `i32::MAX`, so a value Java could never serialize is rejected, not silently emitted. |
//! | Write recursion is depth-guarded | [`MAX_NESTING_DEPTH`] bounds it and Java has none. Parsing cannot build a value that deep, but manual `push` and `put` can, and unbounded recursion would overflow the stack. |
//! | Canonicalization stops at the shredding overlay | `shredded.rs` keeps an untouched field's original bytes VERBATIM over a serialized backing, where Java's verbatim copy is load-bearing. [`VariantObjectBuilder`] stays the plain writer. |
//! | ISO-string factories are not ported | They are `DateTimeUtil` parsing helpers, not format surface. The numeric `of_date` and `of_timestamptz` constructors carry the format. |

use std::collections::HashMap;

use crate::variant::metadata::{
    HEADER_SIZE as METADATA_HEADER_SIZE, OFFSET_SIZE_SHIFT as METADATA_OFFSET_SIZE_SHIFT,
    SORTED_STRINGS, SUPPORTED_VERSION, VariantMetadata,
};
use crate::variant::types::PhysicalType;
use crate::variant::util;
use crate::variant::value::{
    ARRAY_IS_LARGE, FIELD_ID_SIZE_SHIFT, HEADER_SIZE, MAX_NESTING_DEPTH, OBJECT_IS_LARGE,
    OFFSET_SIZE_SHIFT, PRIMITIVE_TYPE_SHIFT, Variant, VariantArray, VariantObject,
    VariantObjectField, VariantPrimitive, VariantValue,
};
use crate::{Error, ErrorKind, Result};

/// The longest string written in the SHORT-STRING form, in UTF-8 bytes (Java
/// `PrimitiveWrapper.MAX_SHORT_STRING_LENGTH` = 63). The spill decision uses the UTF-8 byte length,
/// not the char count, and is made at write time.
const MAX_SHORT_STRING_LENGTH: usize = 63;

/// Java's `int` ceiling: every serialized size, length, count and offset must fit a signed 32-bit
/// integer, because Java buffers and `sizeInBytes()` are `int`-addressed. A larger value is
/// unrepresentable in Java and is rejected here by name.
pub(super) const JAVA_INT_MAX: usize = i32::MAX as usize;

/// The basic-type tag of an object header (Java `VariantUtil.BASIC_TYPE_OBJECT`, the low two
/// bits of `objectHeader`).
const BASIC_TYPE_OBJECT: u8 = 0b10;
/// The basic-type tag of an array header (Java `VariantUtil.BASIC_TYPE_ARRAY`).
const BASIC_TYPE_ARRAY: u8 = 0b11;
/// The basic-type tag of a short-string header (Java `VariantUtil.BASIC_TYPE_SHORT_STRING`).
const BASIC_TYPE_SHORT_STRING: u8 = 0b01;

/// Bytes needed to store `max_value` unsigned: the Java `VariantUtil.sizeOf` thresholds (1 for
/// <= 0xFF, 2 for <= 0xFFFF, 3 for <= 0xFFFFFF, else 4). Callers door inputs at [`JAVA_INT_MAX`].
pub(super) fn size_of_unsigned(max_value: usize) -> usize {
    if max_value <= 0xFF {
        1
    } else if max_value <= 0xFFFF {
        2
    } else if max_value <= 0xFF_FFFF {
        3
    } else {
        4
    }
}

/// Writes the byte at `offset`, bounds-checked (Java `VariantUtil.writeByte` relies on
/// `ByteBuffer`'s unchecked exception).
pub(super) fn write_u8(buffer: &mut [u8], offset: usize, value: u8) -> Result<()> {
    let buffer_len = buffer.len();
    let slot = buffer.get_mut(offset).ok_or_else(|| {
        util::invalid(format!(
            "Invalid variant write: offset {offset} is out of bounds for {buffer_len} bytes"
        ))
    })?;
    *slot = value;
    Ok(())
}

/// Copies `data` to `offset`, bounds-checked (Java `VariantUtil.writeBufferAbsolute`).
pub(super) fn write_bytes(buffer: &mut [u8], offset: usize, data: &[u8]) -> Result<()> {
    let end = offset.checked_add(data.len()).ok_or_else(|| {
        util::invalid(format!(
            "Invalid variant write: byte range {offset}+{} overflows",
            data.len()
        ))
    })?;
    let buffer_len = buffer.len();
    let slot = buffer.get_mut(offset..end).ok_or_else(|| {
        util::invalid(format!(
            "Invalid variant write: byte range {offset}..{end} is out of bounds for \
             {buffer_len} bytes"
        ))
    })?;
    slot.copy_from_slice(data);
    Ok(())
}

/// Writes a `size`-byte (1..=4) little-endian unsigned integer (Java
/// `VariantUtil.writeLittleEndianUnsigned`). Java MASKS an oversized value to the width and
/// silently corrupts it; here it is a named error. Every internal caller picks the width with
/// [`size_of_unsigned`], so this door fires only on the count-wider-than-data pathology.
pub(super) fn write_le_unsigned(
    buffer: &mut [u8],
    value: usize,
    offset: usize,
    size: usize,
) -> Result<()> {
    debug_assert!(
        (1..=4).contains(&size),
        "width selection yields sizes 1..=4"
    );
    let value = value as u64;
    if size < 8 && (value >> (8 * size as u32)) != 0 {
        return Err(util::invalid(format!(
            "Invalid variant write: value {value} does not fit {size} byte(s) \
             (Java would silently truncate it)"
        )));
    }
    write_bytes(buffer, offset, &value.to_le_bytes()[..size])
}

/// Doors a container write up front: `offset + total_size` must fit the buffer, with checked math
/// so a hostile offset cannot wrap. Afterwards interior offset arithmetic cannot overflow.
pub(super) fn door_value_span(
    buffer: &[u8],
    offset: usize,
    total_size: usize,
    what: &str,
) -> Result<()> {
    let fits = offset
        .checked_add(total_size)
        .is_some_and(|end| end <= buffer.len());
    if !fits {
        return Err(util::invalid(format!(
            "Invalid variant write: {what} needs {total_size} bytes at offset {offset}, \
             but the buffer has {} bytes",
            buffer.len()
        )));
    }
    Ok(())
}

/// Converts a width (1..=4) to its header bit field value, `width - 1`.
fn width_bits(width: usize) -> u8 {
    debug_assert!(
        (1..=4).contains(&width),
        "width selection yields sizes 1..=4"
    );
    u8::try_from(width.saturating_sub(1) & 0b11).expect("a 2-bit value fits a byte")
}

/// Builds a metadata header byte (Java `VariantUtil.metadataHeader`:
/// `((offsetSize - 1) << 6) | (isSorted ? 0b10000 : 0) | 0b0001`).
fn metadata_header(is_sorted: bool, offset_size: usize) -> u8 {
    (width_bits(offset_size) << METADATA_OFFSET_SIZE_SHIFT)
        | (if is_sorted { SORTED_STRINGS } else { 0 })
        | SUPPORTED_VERSION
}

/// Builds an object header byte (Java `VariantUtil.objectHeader`:
/// `(isLarge ? 0b1000000 : 0) | ((fieldIdSize - 1) << 4) | ((offsetSize - 1) << 2) | 0b10`).
pub(super) fn object_header(is_large: bool, field_id_size: usize, offset_size: usize) -> u8 {
    (if is_large { OBJECT_IS_LARGE } else { 0 })
        | (width_bits(field_id_size) << FIELD_ID_SIZE_SHIFT)
        | (width_bits(offset_size) << OFFSET_SIZE_SHIFT)
        | BASIC_TYPE_OBJECT
}

/// Builds an array header byte (Java `VariantUtil.arrayHeader`:
/// `(isLarge ? 0b10000 : 0) | (offsetSize - 1) << 2 | 0b11`).
fn array_header(is_large: bool, offset_size: usize) -> u8 {
    (if is_large { ARRAY_IS_LARGE } else { 0 })
        | (width_bits(offset_size) << OFFSET_SIZE_SHIFT)
        | BASIC_TYPE_ARRAY
}

/// Builds a primitive header byte (Java `VariantUtil.primitiveHeader`:
/// `primitiveType << 2`; the low basic-type bits are 0b00 = PRIMITIVE).
fn primitive_header(physical_type: PhysicalType) -> Result<u8> {
    let type_info = physical_type.to_type_info().ok_or_else(|| {
        util::invalid(format!(
            "Invalid variant write: {physical_type:?} has no primitive type id"
        ))
    })?;
    Ok(type_info << PRIMITIVE_TYPE_SHIFT)
}

/// Builds a short-string header byte (Java `VariantUtil.shortStringHeader`:
/// `(length << 2) | BASIC_TYPE_SHORT_STRING`); the caller guarantees `length <= 63`.
fn short_string_header(length: usize) -> u8 {
    (u8::try_from(length).expect("short-string lengths are at most 63") << PRIMITIVE_TYPE_SHIFT)
        | BASIC_TYPE_SHORT_STRING
}

/// The computed serialized layout of a metadata dictionary (`Variants.metadata`'s size math).
struct MetadataLayout {
    data_size: usize,
    offset_size: usize,
    total_size: usize,
}

/// Computes the serialized layout for a dictionary (`Variants.metadata`): `dataSize` = total UTF-8
/// bytes of all names, `offsetSize = sizeOf(dataSize)`, and
/// `totalSize = 1 + offsetSize + (1 + numElements) * offsetSize + dataSize`.
///
/// # Errors
///
/// [`crate::ErrorKind::DataInvalid`] when a size escapes Java's `int` domain, or when the name
/// COUNT does not fit `offsetSize`, the pathology Java silently truncates.
fn metadata_layout(names: &[String]) -> Result<MetadataLayout> {
    let mut data_size = 0usize;
    for name in names {
        data_size = data_size
            .checked_add(name.len())
            .filter(|size| *size <= JAVA_INT_MAX)
            .ok_or_else(|| {
                util::invalid(format!(
                    "Invalid variant metadata: total dictionary string data exceeds {JAVA_INT_MAX} bytes"
                ))
            })?;
    }
    let offset_size = size_of_unsigned(data_size);
    // Java writes the dictionary size with offsetSize bytes and MASKS it; reachable only
    // with empty names (any non-empty names make dataSize >= numElements).
    if (names.len() as u64) >> (8 * offset_size as u32) != 0 {
        return Err(util::invalid(format!(
            "Invalid variant metadata: {} dictionary entries do not fit the {offset_size}-byte \
             offset size selected for {data_size} data byte(s) (Java 1.10.0 silently truncates \
             this dictionary; refusing to write corrupt metadata)",
            names.len()
        )));
    }
    let offsets_len = names
        .len()
        .checked_add(1)
        .and_then(|count| count.checked_mul(offset_size))
        .ok_or_else(|| util::invalid("Invalid variant metadata: offset list size overflows"))?;
    let total_size = METADATA_HEADER_SIZE
        .checked_add(offset_size)
        .and_then(|size| size.checked_add(offsets_len))
        .and_then(|size| size.checked_add(data_size))
        .filter(|size| *size <= JAVA_INT_MAX)
        .ok_or_else(|| {
            util::invalid(format!(
                "Invalid variant metadata: serialized size exceeds {JAVA_INT_MAX} bytes"
            ))
        })?;
    Ok(MetadataLayout {
        data_size,
        offset_size,
        total_size,
    })
}

// ===== metadata building and serialization ==================================================
// The write-side surface of `VariantMetadata` (`Variants.metadata`, `VariantMetadata.writeTo`).

impl VariantMetadata {
    /// Builds metadata from field names, exactly as Java `Variants.metadata(Collection)`. The
    /// dictionary keeps the INSERTION order, with no dedup and no re-sort. The sorted flag is set
    /// only when the input is already STRICTLY ascending in Java `String.compareTo` (UTF-16 code
    /// unit) order, so a duplicate name clears it. An empty input gives the empty-v1 metadata
    /// `01 00 00`, with the sorted flag NOT set.
    ///
    /// # Errors
    ///
    /// [`crate::ErrorKind::DataInvalid`] when the dictionary escapes Java's `int` domain, or on the
    /// count-truncation pathology (see the module doc).
    pub fn from_field_names<I, S>(field_names: I) -> Result<VariantMetadata>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let dictionary: Vec<String> = field_names.into_iter().map(Into::into).collect();
        if dictionary.is_empty() {
            // Variants.metadata returns EMPTY_V1_METADATA (buffer `01 00 00`): the sorted
            // flag is NOT set on the empty dictionary.
            return Ok(VariantMetadata::from_parts(false, dictionary, 3));
        }
        let is_sorted = dictionary
            .windows(2)
            .all(|pair| util::java_string_compare(&pair[0], &pair[1]) == std::cmp::Ordering::Less);
        let layout = metadata_layout(&dictionary)?;
        Ok(VariantMetadata::from_parts(
            is_sorted,
            dictionary,
            layout.total_size,
        ))
    }

    /// Serializes this metadata to the on-disk layout: header byte, dictionary size,
    /// `dictionarySize + 1` offsets, then the concatenated UTF-8 strings.
    ///
    /// Output is byte-identical to Java's for metadata built by [`Self::from_field_names`] and for
    /// anything Java's own writer produced. A PARSED metadata re-encodes with the canonical writer
    /// widths, because Java's `SerializedMetadata.writeTo` copies its original buffer verbatim. The
    /// sorted flag is preserved as parsed, and the output length can then differ from
    /// [`Self::size_in_bytes`], which reports the PARSED size.
    ///
    /// # Errors
    ///
    /// [`crate::ErrorKind::DataInvalid`] per [`Self::from_field_names`]'s doors.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let names = self.dictionary();
        let layout = metadata_layout(names)?;
        let offset_size = layout.offset_size;
        let offset_list_offset = METADATA_HEADER_SIZE + offset_size;
        let data_offset = offset_list_offset + (1 + names.len()) * offset_size;

        let mut buffer = vec![0u8; layout.total_size];
        write_u8(
            &mut buffer,
            0,
            metadata_header(self.is_sorted(), offset_size),
        )?;
        write_le_unsigned(&mut buffer, names.len(), METADATA_HEADER_SIZE, offset_size)?;
        let mut next_offset = 0usize;
        for (index, name) in names.iter().enumerate() {
            write_le_unsigned(
                &mut buffer,
                next_offset,
                offset_list_offset + index * offset_size,
                offset_size,
            )?;
            write_bytes(&mut buffer, data_offset + next_offset, name.as_bytes())?;
            next_offset += name.len();
        }
        // The final offset entry is the total string-data length.
        write_le_unsigned(
            &mut buffer,
            next_offset,
            offset_list_offset + names.len() * offset_size,
            offset_size,
        )?;
        debug_assert_eq!(next_offset, layout.data_size);
        Ok(buffer)
    }
}

// ===== `Variants.of(...)` factory constructors ==============================================
// The write-side construction surface of `VariantValue`.

impl VariantValue {
    /// A JSON-style null (Java `Variants.ofNull()`).
    pub fn of_null() -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Null)
    }

    /// A boolean (Java `Variants.of(boolean)`). The physical type, `BOOLEAN_TRUE` or
    /// `BOOLEAN_FALSE`, is derived from the value.
    pub fn of_boolean(value: bool) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Boolean(value))
    }

    /// An 8-bit integer (Java `Variants.of(byte)`).
    pub fn of_int8(value: i8) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Int8(value))
    }

    /// A 16-bit integer (Java `Variants.of(short)`).
    pub fn of_int16(value: i16) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Int16(value))
    }

    /// A 32-bit integer (Java `Variants.of(int)`).
    pub fn of_int32(value: i32) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Int32(value))
    }

    /// A 64-bit integer (Java `Variants.of(long)`).
    pub fn of_int64(value: i64) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Int64(value))
    }

    /// A single-precision float (Java `Variants.of(float)`).
    pub fn of_float(value: f32) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Float(value))
    }

    /// A double-precision float (Java `Variants.of(double)`).
    pub fn of_double(value: f64) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Double(value))
    }

    /// A date as days from the unix epoch (Java `Variants.ofDate(int)`).
    pub fn of_date(days_from_epoch: i32) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Date(days_from_epoch))
    }

    /// A UTC-adjusted timestamp in microseconds from the unix epoch (Java
    /// `Variants.ofTimestamptz(long)`).
    pub fn of_timestamptz(micros_from_epoch: i64) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Timestamptz(micros_from_epoch))
    }

    /// A local (zone-less) timestamp in microseconds from the unix epoch (Java
    /// `Variants.ofTimestampntz(long)`).
    pub fn of_timestampntz(micros_from_epoch: i64) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Timestampntz(micros_from_epoch))
    }

    /// A time of day in microseconds from midnight (Java `Variants.ofTime(long)`).
    pub fn of_time(micros_from_midnight: i64) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Time(micros_from_midnight))
    }

    /// A UTC-adjusted timestamp in nanoseconds from the unix epoch (Java
    /// `Variants.ofTimestamptzNanos(long)`).
    pub fn of_timestamptz_nanos(nanos_from_epoch: i64) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::TimestamptzNanos(nanos_from_epoch))
    }

    /// A local timestamp in nanoseconds from the unix epoch (Java
    /// `Variants.ofTimestampntzNanos(long)`).
    pub fn of_timestampntz_nanos(nanos_from_epoch: i64) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::TimestampntzNanos(nanos_from_epoch))
    }

    /// An opaque byte sequence (Java `Variants.of(ByteBuffer)`).
    pub fn of_binary(data: Vec<u8>) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Binary(data))
    }

    /// A UTF-8 string (Java `Variants.of(String)`). The SHORT form or the long `STRING` form is
    /// decided at write time by the UTF-8 byte length, exactly like Java.
    pub fn of_string(value: impl Into<String>) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::String(value.into()))
    }

    /// A UUID from its 16 big-endian (RFC 4122) bytes (Java `Variants.ofUUID`, which writes
    /// `UUIDUtil.convertToByteBuffer(uuid)` — the same big-endian byte order).
    pub fn of_uuid(big_endian_bytes: [u8; 16]) -> VariantValue {
        VariantValue::Primitive(VariantPrimitive::Uuid(big_endian_bytes))
    }

    /// A decimal from its two's-complement unscaled value and scale. Picks the smallest physical
    /// decimal type by PRECISION, like Java `Variants.of(BigDecimal)`: 1..=9 gives decimal4, 10..=18
    /// gives decimal8, and up to 38 gives decimal16. Precision is the digit count of `|unscaled|`,
    /// and zero has precision 1.
    ///
    /// # Errors
    ///
    /// [`crate::ErrorKind::FeatureUnsupported`] above precision 38, like Java's
    /// `UnsupportedOperationException`. A 39-digit `i128` such as `i128::MIN` is therefore not
    /// constructible here, exactly as it is not through `Variants.of(BigDecimal)`.
    /// [`VariantPrimitive::Decimal16`] can still represent it directly.
    pub fn of_decimal(unscaled: i128, scale: u8) -> Result<VariantValue> {
        let precision = decimal_precision(unscaled);
        let primitive = if precision <= 9 {
            VariantPrimitive::Decimal4 {
                scale,
                unscaled: i32::try_from(unscaled)
                    .expect("a value of at most 9 decimal digits fits i32"),
            }
        } else if precision <= 18 {
            VariantPrimitive::Decimal8 {
                scale,
                unscaled: i64::try_from(unscaled)
                    .expect("a value of at most 18 decimal digits fits i64"),
            }
        } else if precision <= 38 {
            VariantPrimitive::Decimal16 { scale, unscaled }
        } else {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("Unsupported decimal precision: {precision}"),
            ));
        };
        Ok(VariantValue::Primitive(primitive))
    }
}

/// Returns the decimal digit count of `|unscaled|` — Java `BigDecimal.precision()` for an
/// integer unscaled value (zero reports 1).
fn decimal_precision(unscaled: i128) -> u32 {
    let mut magnitude = unscaled.unsigned_abs();
    let mut digits = 1u32;
    while magnitude >= 10 {
        magnitude /= 10;
        digits += 1;
    }
    digits
}

// ===== value serialization ==================================================================
// The write-side surface of `VariantValue` (`sizeInBytes()` and `writeTo`).

impl VariantValue {
    /// Returns the serialized size in bytes of this value (Java `sizeInBytes()`). An object needs
    /// `metadata`, because its field-id width is `sizeOf(metadata.dictionarySize())`.
    ///
    /// # Errors
    ///
    /// [`crate::ErrorKind::DataInvalid`] when the value escapes Java's `int` domain or exceeds
    /// [`MAX_NESTING_DEPTH`].
    pub fn size_in_bytes(&self, metadata: &VariantMetadata) -> Result<usize> {
        value_size(self, metadata, 0)
    }

    /// Writes this value into `buffer` at `offset` and returns the bytes written. The port of Java
    /// `VariantValue.writeTo(ByteBuffer, int)`, with absolute offsets.
    ///
    /// # Errors
    ///
    /// [`crate::ErrorKind::DataInvalid`] when the buffer is too small, a field name is missing from
    /// `metadata` (Java `checkState`: "Invalid metadata, missing: %s"), the value escapes Java's
    /// `int` domain, or nesting exceeds [`MAX_NESTING_DEPTH`].
    pub fn write_to(
        &self,
        metadata: &VariantMetadata,
        buffer: &mut [u8],
        offset: usize,
    ) -> Result<usize> {
        // Door the caller-supplied offset before any offset arithmetic. A slice is bounded by
        // isize::MAX, so once offset <= buffer.len() the interior offset math cannot overflow.
        if offset > buffer.len() {
            return Err(util::invalid(format!(
                "Invalid variant write: offset {offset} is out of bounds for {} bytes",
                buffer.len()
            )));
        }
        write_value(self, metadata, buffer, offset, 0)
    }

    /// Serializes this value to its on-disk bytes: an exact-size buffer filled by
    /// [`Self::write_to`], the pattern every Java caller uses.
    ///
    /// # Errors
    ///
    /// Any [`Self::size_in_bytes`] or [`Self::write_to`] error.
    pub fn to_bytes(&self, metadata: &VariantMetadata) -> Result<Vec<u8>> {
        let size = self.size_in_bytes(metadata)?;
        let mut buffer = vec![0u8; size];
        let written = self.write_to(metadata, &mut buffer, 0)?;
        debug_assert_eq!(written, size, "write_to must fill exactly size_in_bytes");
        Ok(buffer)
    }
}

// ===== whole-variant emission ===============================================================

impl Variant {
    /// Serializes the variant as metadata bytes immediately followed by value bytes, the
    /// single-buffer layout [`Variant::from_bytes`] parses.
    ///
    /// # Errors
    ///
    /// Any [`VariantMetadata::to_bytes`] or [`VariantValue::to_bytes`] error.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.metadata().to_bytes()?;
        let value_bytes = self.value().to_bytes(self.metadata())?;
        bytes.extend_from_slice(&value_bytes);
        Ok(bytes)
    }
}

/// Recursive size computation. Depth-bounded by [`MAX_NESTING_DEPTH`]: the format is genuinely
/// recursive, and the bound is checked before any child walk, so stack usage stays capped.
pub(super) fn value_size(
    value: &VariantValue,
    metadata: &VariantMetadata,
    depth: usize,
) -> Result<usize> {
    if depth > MAX_NESTING_DEPTH {
        return Err(util::invalid(format!(
            "Invalid variant: nesting depth exceeds the supported maximum {MAX_NESTING_DEPTH}"
        )));
    }
    match value {
        VariantValue::Primitive(primitive) => primitive_size(primitive),
        VariantValue::Object(object) => {
            let layout = object_layout(object, metadata, depth)?;
            Ok(layout.total_size)
        }
        VariantValue::Array(array) => {
            let layout = array_layout(array, metadata, depth)?;
            Ok(layout.total_size)
        }
    }
}

/// Primitive serialized sizes — the exact `PrimitiveWrapper.sizeInBytes()` table (1.10.0).
fn primitive_size(primitive: &VariantPrimitive) -> Result<usize> {
    let size = match primitive {
        VariantPrimitive::Null | VariantPrimitive::Boolean(_) => 1,
        VariantPrimitive::Int8(_) => 2,
        VariantPrimitive::Int16(_) => 3,
        VariantPrimitive::Int32(_) | VariantPrimitive::Date(_) | VariantPrimitive::Float(_) => 5,
        VariantPrimitive::Int64(_)
        | VariantPrimitive::Double(_)
        | VariantPrimitive::Timestamptz(_)
        | VariantPrimitive::Timestampntz(_)
        | VariantPrimitive::TimestamptzNanos(_)
        | VariantPrimitive::TimestampntzNanos(_)
        | VariantPrimitive::Time(_) => 9,
        VariantPrimitive::Decimal4 { .. } => 6,
        VariantPrimitive::Decimal8 { .. } => 10,
        VariantPrimitive::Decimal16 { .. } => 18,
        VariantPrimitive::Uuid(_) => 17,
        VariantPrimitive::Binary(data) => length_prefixed_size(data.len(), "binary")?,
        VariantPrimitive::String(value) => {
            let utf8_length = value.len();
            if utf8_length <= MAX_SHORT_STRING_LENGTH {
                // 1 header byte, the length packed into it (the SHORT-STRING form).
                HEADER_SIZE + utf8_length
            } else {
                length_prefixed_size(utf8_length, "string")?
            }
        }
    };
    Ok(size)
}

/// Size of a length-prefixed payload (binary / long string): 1 header + 4 length + payload,
/// doored at Java's `int` domain.
fn length_prefixed_size(payload_length: usize, what: &str) -> Result<usize> {
    (HEADER_SIZE + 4)
        .checked_add(payload_length)
        .filter(|size| *size <= JAVA_INT_MAX)
        .ok_or_else(|| {
            util::invalid(format!(
                "Invalid variant {what}: serialized size exceeds {JAVA_INT_MAX} bytes"
            ))
        })
}

/// The computed serialized layout of a container value.
struct ContainerLayout {
    is_large: bool,
    offset_size: usize,
    /// Field-id width for objects; 0 for arrays (no field-id list).
    field_id_size: usize,
    data_size: usize,
    total_size: usize,
}

/// Object layout per `ShreddedObject.SerializationState` (1.10.0): `fieldIdSize =
/// sizeOf(metadata.dictionarySize())` (the dictionary SIZE, not the largest id),
/// `isLarge = numElements > 0xFF`, `offsetSize = sizeOf(dataSize)`,
/// `size = 1 + (4|1) + n*fieldIdSize + (1+n)*offsetSize + dataSize`.
fn object_layout(
    object: &VariantObject,
    metadata: &VariantMetadata,
    depth: usize,
) -> Result<ContainerLayout> {
    let num_elements = object.num_fields();
    let field_id_size = size_of_unsigned(metadata.dictionary_size());
    let mut data_size = 0usize;
    for field in object.fields() {
        data_size = checked_data_size(data_size, value_size(&field.value, metadata, depth + 1)?)?;
    }
    let is_large = num_elements > 0xFF;
    let offset_size = size_of_unsigned(data_size);
    let count_size = if is_large { 4 } else { 1 };
    let total_size = HEADER_SIZE
        .checked_add(count_size)
        .and_then(|size| size.checked_add(num_elements.checked_mul(field_id_size)?))
        .and_then(|size| size.checked_add(num_elements.checked_add(1)?.checked_mul(offset_size)?))
        .and_then(|size| size.checked_add(data_size))
        .filter(|size| *size <= JAVA_INT_MAX)
        .ok_or_else(|| {
            util::invalid(format!(
                "Invalid variant object: serialized size exceeds {JAVA_INT_MAX} bytes"
            ))
        })?;
    Ok(ContainerLayout {
        is_large,
        offset_size,
        field_id_size,
        data_size,
        total_size,
    })
}

/// Array layout per `ValueArray.SerializationState` (1.10.0): `isLarge = numElements > 0xFF`,
/// `offsetSize = sizeOf(dataSize)`, `size = 1 + (4|1) + (1+n)*offsetSize + dataSize`.
fn array_layout(
    array: &VariantArray,
    metadata: &VariantMetadata,
    depth: usize,
) -> Result<ContainerLayout> {
    let num_elements = array.num_elements();
    let mut data_size = 0usize;
    for element in array.elements() {
        data_size = checked_data_size(data_size, value_size(element, metadata, depth + 1)?)?;
    }
    let is_large = num_elements > 0xFF;
    let offset_size = size_of_unsigned(data_size);
    let count_size = if is_large { 4 } else { 1 };
    let total_size = HEADER_SIZE
        .checked_add(count_size)
        .and_then(|size| size.checked_add(num_elements.checked_add(1)?.checked_mul(offset_size)?))
        .and_then(|size| size.checked_add(data_size))
        .filter(|size| *size <= JAVA_INT_MAX)
        .ok_or_else(|| {
            util::invalid(format!(
                "Invalid variant array: serialized size exceeds {JAVA_INT_MAX} bytes"
            ))
        })?;
    Ok(ContainerLayout {
        is_large,
        offset_size,
        field_id_size: 0,
        data_size,
        total_size,
    })
}

/// Accumulates a child size into a container data size, doored at Java's `int` domain.
pub(super) fn checked_data_size(accumulated: usize, child_size: usize) -> Result<usize> {
    accumulated
        .checked_add(child_size)
        .filter(|size| *size <= JAVA_INT_MAX)
        .ok_or_else(|| {
            util::invalid(format!(
                "Invalid variant: container data exceeds {JAVA_INT_MAX} bytes"
            ))
        })
}

/// Recursive write dispatch (the `writeTo` side of the three serialization states). The depth
/// bound mirrors [`value_size`].
pub(super) fn write_value(
    value: &VariantValue,
    metadata: &VariantMetadata,
    buffer: &mut [u8],
    offset: usize,
    depth: usize,
) -> Result<usize> {
    if depth > MAX_NESTING_DEPTH {
        return Err(util::invalid(format!(
            "Invalid variant: nesting depth exceeds the supported maximum {MAX_NESTING_DEPTH}"
        )));
    }
    match value {
        VariantValue::Primitive(primitive) => write_primitive(primitive, buffer, offset),
        VariantValue::Object(object) => write_object(object, metadata, buffer, offset, depth),
        VariantValue::Array(array) => write_array(array, metadata, buffer, offset, depth),
    }
}

/// Writes a primitive (the `PrimitiveWrapper.writeTo` payload layouts): header byte, then the
/// little-endian payload. A decimal writes the raw scale byte then the LE unscaled value, which is
/// `i128::to_le_bytes` for decimal16. A UUID writes its 16 big-endian bytes as stored. A string
/// uses the short form when its UTF-8 length is at most 63, else the long form.
fn write_primitive(
    primitive: &VariantPrimitive,
    buffer: &mut [u8],
    offset: usize,
) -> Result<usize> {
    let header = match primitive {
        // The string header depends on the spill decision below.
        VariantPrimitive::String(_) => 0,
        other => primitive_header(other.physical_type())?,
    };
    let payload_offset = offset + HEADER_SIZE;
    match primitive {
        VariantPrimitive::Null | VariantPrimitive::Boolean(_) => {
            write_u8(buffer, offset, header)?;
            Ok(1)
        }
        VariantPrimitive::Int8(value) => {
            write_u8(buffer, offset, header)?;
            write_bytes(buffer, payload_offset, &value.to_le_bytes())?;
            Ok(2)
        }
        VariantPrimitive::Int16(value) => {
            write_u8(buffer, offset, header)?;
            write_bytes(buffer, payload_offset, &value.to_le_bytes())?;
            Ok(3)
        }
        VariantPrimitive::Int32(value) | VariantPrimitive::Date(value) => {
            write_u8(buffer, offset, header)?;
            write_bytes(buffer, payload_offset, &value.to_le_bytes())?;
            Ok(5)
        }
        VariantPrimitive::Float(value) => {
            write_u8(buffer, offset, header)?;
            write_bytes(buffer, payload_offset, &value.to_le_bytes())?;
            Ok(5)
        }
        VariantPrimitive::Int64(value)
        | VariantPrimitive::Timestamptz(value)
        | VariantPrimitive::Timestampntz(value)
        | VariantPrimitive::Time(value)
        | VariantPrimitive::TimestamptzNanos(value)
        | VariantPrimitive::TimestampntzNanos(value) => {
            write_u8(buffer, offset, header)?;
            write_bytes(buffer, payload_offset, &value.to_le_bytes())?;
            Ok(9)
        }
        VariantPrimitive::Double(value) => {
            write_u8(buffer, offset, header)?;
            write_bytes(buffer, payload_offset, &value.to_le_bytes())?;
            Ok(9)
        }
        VariantPrimitive::Decimal4 { scale, unscaled } => {
            write_u8(buffer, offset, header)?;
            write_u8(buffer, payload_offset, *scale)?;
            write_bytes(buffer, payload_offset + 1, &unscaled.to_le_bytes())?;
            Ok(6)
        }
        VariantPrimitive::Decimal8 { scale, unscaled } => {
            write_u8(buffer, offset, header)?;
            write_u8(buffer, payload_offset, *scale)?;
            write_bytes(buffer, payload_offset + 1, &unscaled.to_le_bytes())?;
            Ok(10)
        }
        VariantPrimitive::Decimal16 { scale, unscaled } => {
            write_u8(buffer, offset, header)?;
            write_u8(buffer, payload_offset, *scale)?;
            write_bytes(buffer, payload_offset + 1, &unscaled.to_le_bytes())?;
            Ok(18)
        }
        VariantPrimitive::Binary(data) => {
            let size = length_prefixed_size(data.len(), "binary")?;
            write_u8(buffer, offset, header)?;
            write_le_unsigned(buffer, data.len(), payload_offset, 4)?;
            write_bytes(buffer, payload_offset + 4, data)?;
            Ok(size)
        }
        VariantPrimitive::String(value) => {
            let utf8 = value.as_bytes();
            if utf8.len() <= MAX_SHORT_STRING_LENGTH {
                write_u8(buffer, offset, short_string_header(utf8.len()))?;
                write_bytes(buffer, payload_offset, utf8)?;
                Ok(HEADER_SIZE + utf8.len())
            } else {
                let size = length_prefixed_size(utf8.len(), "string")?;
                write_u8(buffer, offset, primitive_header(PhysicalType::String)?)?;
                write_le_unsigned(buffer, utf8.len(), payload_offset, 4)?;
                write_bytes(buffer, payload_offset + 4, utf8)?;
                Ok(size)
            }
        }
        VariantPrimitive::Uuid(big_endian_bytes) => {
            write_u8(buffer, offset, header)?;
            write_bytes(buffer, payload_offset, big_endian_bytes)?;
            Ok(17)
        }
    }
}

/// Writes an object (`ShreddedObject.SerializationState.writeTo`): header, count, field-id list,
/// offset list with one extra entry holding the data length, then the field values. Fields are
/// emitted in STORED order, which [`VariantObjectBuilder::build`] and a parsed spec-conforming
/// object both keep name-sorted (UTF-16). Each field id is re-resolved from the metadata BY NAME.
fn write_object(
    object: &VariantObject,
    metadata: &VariantMetadata,
    buffer: &mut [u8],
    offset: usize,
    depth: usize,
) -> Result<usize> {
    let layout = object_layout(object, metadata, depth)?;
    // Door the whole span up front, so the interior offset arithmetic below cannot overflow.
    door_value_span(buffer, offset, layout.total_size, "object")?;
    let num_elements = object.num_fields();
    let count_size = if layout.is_large { 4 } else { 1 };
    let field_id_list_offset = offset + HEADER_SIZE + count_size;
    let offset_list_offset = field_id_list_offset + num_elements * layout.field_id_size;
    let data_offset = offset_list_offset + (1 + num_elements) * layout.offset_size;

    write_u8(
        buffer,
        offset,
        object_header(layout.is_large, layout.field_id_size, layout.offset_size),
    )?;
    write_le_unsigned(buffer, num_elements, offset + HEADER_SIZE, count_size)?;

    let mut next_value_offset = 0usize;
    for (index, field) in object.fields().iter().enumerate() {
        // Java: checkState(metadata.id(field) >= 0, "Invalid metadata, missing: %s").
        let id = metadata
            .id(&field.name)
            .ok_or_else(|| util::invalid(format!("Invalid metadata, missing: {}", field.name)))?;
        write_le_unsigned(
            buffer,
            id,
            field_id_list_offset + index * layout.field_id_size,
            layout.field_id_size,
        )?;
        write_le_unsigned(
            buffer,
            next_value_offset,
            offset_list_offset + index * layout.offset_size,
            layout.offset_size,
        )?;
        let value_size = write_value(
            &field.value,
            metadata,
            buffer,
            data_offset + next_value_offset,
            depth + 1,
        )?;
        next_value_offset += value_size;
    }
    // The final offset entry is the total size of the data section.
    write_le_unsigned(
        buffer,
        next_value_offset,
        offset_list_offset + num_elements * layout.offset_size,
        layout.offset_size,
    )?;
    debug_assert_eq!(next_value_offset, layout.data_size);
    Ok((data_offset - offset) + layout.data_size)
}

/// Writes an array (`ValueArray.SerializationState.writeTo`, 1.10.0): header, count, offset
/// list (one extra entry holding the data length), then the elements in insertion order.
fn write_array(
    array: &VariantArray,
    metadata: &VariantMetadata,
    buffer: &mut [u8],
    offset: usize,
    depth: usize,
) -> Result<usize> {
    let layout = array_layout(array, metadata, depth)?;
    // Door the whole span up front (checked) — see `write_object`.
    door_value_span(buffer, offset, layout.total_size, "array")?;
    let num_elements = array.num_elements();
    let count_size = if layout.is_large { 4 } else { 1 };
    let offset_list_offset = offset + HEADER_SIZE + count_size;
    let data_offset = offset_list_offset + (1 + num_elements) * layout.offset_size;

    write_u8(
        buffer,
        offset,
        array_header(layout.is_large, layout.offset_size),
    )?;
    write_le_unsigned(buffer, num_elements, offset + HEADER_SIZE, count_size)?;

    let mut next_value_offset = 0usize;
    for (index, element) in array.elements().iter().enumerate() {
        write_le_unsigned(
            buffer,
            next_value_offset,
            offset_list_offset + index * layout.offset_size,
            layout.offset_size,
        )?;
        let value_size = write_value(
            element,
            metadata,
            buffer,
            data_offset + next_value_offset,
            depth + 1,
        )?;
        next_value_offset += value_size;
    }
    write_le_unsigned(
        buffer,
        next_value_offset,
        offset_list_offset + num_elements * layout.offset_size,
        layout.offset_size,
    )?;
    debug_assert_eq!(next_value_offset, layout.data_size);
    Ok((data_offset - offset) + layout.data_size)
}

/// A plain variant-object writer, the object-writing core of Java's `ShreddedObject`
/// (`Variants.object(metadata)` + `put`), WITHOUT the shredding overlay.
///
/// `put` validates the field name against the metadata dictionary up front and replaces any
/// previous value for the same name. [`Self::build`] sorts fields by name in Java
/// `String.compareTo` (UTF-16 code unit) order, the on-disk field order Java's writer emits.
#[derive(Debug)]
pub struct VariantObjectBuilder<'a> {
    metadata: &'a VariantMetadata,
    fields: HashMap<String, VariantValue>,
}

impl<'a> VariantObjectBuilder<'a> {
    /// Starts an object against the given metadata dictionary (Java `Variants.object(metadata)`).
    pub fn new(metadata: &'a VariantMetadata) -> VariantObjectBuilder<'a> {
        VariantObjectBuilder {
            metadata,
            fields: HashMap::new(),
        }
    }

    /// Sets a field, replacing any previous value for the same name (Java `ShreddedObject.put`).
    ///
    /// # Errors
    ///
    /// [`crate::ErrorKind::DataInvalid`] when the name is not in the metadata dictionary (Java
    /// throws `IllegalArgumentException("Cannot find field name in metadata: %s")`).
    pub fn put(&mut self, name: impl Into<String>, value: VariantValue) -> Result<()> {
        let name = name.into();
        if self.metadata.id(&name).is_none() {
            return Err(util::invalid(format!(
                "Cannot find field name in metadata: {name}"
            )));
        }
        self.fields.insert(name, value);
        Ok(())
    }

    /// Finishes the object: fields are sorted by name in Java `String.compareTo` (UTF-16) order
    /// with their dictionary ids resolved, the exact on-disk order `ShreddedObject.writeTo` emits.
    pub fn build(self) -> VariantObject {
        let VariantObjectBuilder { metadata, fields } = self;
        let mut named: Vec<(String, VariantValue)> = fields.into_iter().collect();
        named.sort_by(|left, right| util::java_string_compare(&left.0, &right.0));
        let fields = named
            .into_iter()
            .map(|(name, value)| {
                let id = metadata
                    .id(&name)
                    .expect("field names are validated against the metadata at put time");
                VariantObjectField {
                    field_id: u32::try_from(id)
                        .expect("dictionary ids are bounded by Java's signed 32-bit domain"),
                    name,
                    value,
                }
            })
            .collect();
        VariantObject::from_fields(fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Risk pinned: the width thresholds are Java's UNSIGNED `sizeOf` boundaries. An off-by-one
    /// flips every offset and count width at the boundary and corrupts the container layout.
    #[test]
    fn test_size_of_unsigned_matches_java_thresholds() {
        assert_eq!(size_of_unsigned(0), 1);
        assert_eq!(size_of_unsigned(0xFF), 1);
        assert_eq!(size_of_unsigned(0x100), 2);
        assert_eq!(size_of_unsigned(0xFFFF), 2);
        assert_eq!(size_of_unsigned(0x1_0000), 3);
        assert_eq!(size_of_unsigned(0xFF_FFFF), 3);
        assert_eq!(size_of_unsigned(0x100_0000), 4);
        assert_eq!(size_of_unsigned(JAVA_INT_MAX), 4);
    }

    /// Risk pinned: Java's `writeLittleEndianUnsigned` MASKS an oversized value into the
    /// requested width (silent corruption); the Rust door must reject it by name instead.
    #[test]
    fn test_write_le_unsigned_rejects_oversized_values_java_would_truncate() {
        let mut buffer = [0u8; 4];
        let error =
            write_le_unsigned(&mut buffer, 0x100, 0, 1).expect_err("256 does not fit one byte");
        assert!(
            error.to_string().contains("does not fit"),
            "error must name the truncation, got: {error}"
        );
        write_le_unsigned(&mut buffer, 0xFF, 0, 1).expect("255 fits one byte");
        assert_eq!(buffer[0], 0xFF);
        write_le_unsigned(&mut buffer, 0x030201, 0, 3).expect("3-byte value");
        assert_eq!(&buffer[..3], &[0x01, 0x02, 0x03]);
    }

    /// Risk pinned: `BigDecimal.precision()` semantics. Zero reports 1 (decimal4), the 9/10 and
    /// 18/19 boundaries flip the physical width, and 39 digits must error with Java's message.
    #[test]
    fn test_decimal_precision_boundaries_match_java_bigdecimal() {
        assert_eq!(decimal_precision(0), 1);
        assert_eq!(decimal_precision(-9), 1);
        assert_eq!(decimal_precision(999_999_999), 9);
        assert_eq!(decimal_precision(1_000_000_000), 10);
        assert_eq!(decimal_precision(999_999_999_999_999_999), 18);
        assert_eq!(decimal_precision(1_000_000_000_000_000_000), 19);
        assert_eq!(decimal_precision(i128::MIN), 39);

        match VariantValue::of_decimal(999_999_999, 2).expect("precision 9") {
            VariantValue::Primitive(VariantPrimitive::Decimal4 { scale: 2, .. }) => {}
            other => panic!("precision 9 must be decimal4, got {other:?}"),
        }
        match VariantValue::of_decimal(1_000_000_000, 2).expect("precision 10") {
            VariantValue::Primitive(VariantPrimitive::Decimal8 { scale: 2, .. }) => {}
            other => panic!("precision 10 must be decimal8, got {other:?}"),
        }
        match VariantValue::of_decimal(-1_000_000_000_000_000_000, 38).expect("precision 19") {
            VariantValue::Primitive(VariantPrimitive::Decimal16 { scale: 38, .. }) => {}
            other => panic!("precision 19 must be decimal16, got {other:?}"),
        }
        let error = VariantValue::of_decimal(i128::MIN, 38).expect_err("precision 39");
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        assert!(
            error
                .to_string()
                .contains("Unsupported decimal precision: 39"),
            "error must carry Java's message, got: {error}"
        );
    }
}
