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

//! In-memory deletion vectors and their on-disk `deletion-vector-v1` (Puffin DV blob) encoding.
//!
//! [`DeleteVector`] is the Rust analogue of Java's `PositionDeleteIndex`. It holds the deleted row
//! positions of one data file in a roaring treemap. The blob encoding is byte-compatible with Java
//! `BitmapPositionDeleteIndex` and `RoaringPositionBitmap`. The merge-on-read scan applies these
//! vectors, and [`crate::writer::base_writer::deletion_vector_writer`] writes them.

use std::io::Read;
use std::ops::BitOrAssign;

use roaring::bitmap::Iter;
use roaring::treemap::BitmapIter;
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::io::FileIO;
use crate::spec::{DataContentType, DataFile, DataFileFormat};
use crate::{Error, ErrorKind, Result};

/// The three manifest fields a Puffin `deletion-vector-v1` blob must carry before it can be read,
/// with the untrusted `i64` ranges checked into `u64`.
pub(crate) struct DeleteVectorCoordinates {
    pub(crate) referenced_data_file: String,
    pub(crate) content_offset: u64,
    pub(crate) content_size_in_bytes: u64,
}

/// Validates the DV metadata on a delete file and range-checks the coordinates into `u64`.
///
/// Java `BaseDeleteLoader.validateDV`, plus a keying prerequisite. `referenced_data_file` must be
/// present. The Puffin spec makes it mandatory, and the loaded vector is keyed by it.
///
/// # Errors
///
/// `DataInvalid` when a field is absent, the offset is negative, or the size is outside 0..=2GB.
///
/// # Notes
///
/// The scan path and the public loader share this, so their messages cannot drift.
pub(crate) fn validate_delete_vector_coordinates(
    file_path: &str,
    referenced_data_file: Option<String>,
    content_offset: Option<i64>,
    content_size_in_bytes: Option<i64>,
) -> Result<DeleteVectorCoordinates> {
    let referenced_data_file = referenced_data_file.ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Invalid deletion vector '{file_path}': missing referenced_data_file"),
        )
    })?;

    let checked_offset = content_offset
        .and_then(|offset| u64::try_from(offset).ok())
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid deletion vector '{file_path}': content_offset must be a non-negative integer, got {content_offset:?}"
                ),
            )
        })?;

    // Java caps contentSizeInBytes at Integer.MAX_VALUE. A negative size is equally invalid.
    let checked_size = content_size_in_bytes
        .filter(|size| (0..=i64::from(i32::MAX)).contains(size))
        .and_then(|size| u64::try_from(size).ok())
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid deletion vector '{file_path}': content_size_in_bytes must be between 0 and {} (2GB), got {content_size_in_bytes:?}",
                    i32::MAX
                ),
            )
        })?;

    Ok(DeleteVectorCoordinates {
        referenced_data_file,
        content_offset: checked_offset,
        content_size_in_bytes: checked_size,
    })
}

/// Read one committed deletion vector back off storage.
///
/// Java `BaseDeleteLoader.readDV`. One ranged read at the manifest's blob coordinates, then the
/// `deletion-vector-v1` decode. To merge into an existing DV, pass the result to
/// [`crate::writer::base_writer::deletion_vector_writer::DVFileWriter::with_previous_deletes`].
///
/// # Errors
///
/// `DataInvalid` when `delete_file` is not a Puffin position-delete file, when its blob coordinates
/// are missing or out of range, or when the decoded cardinality disagrees with `record_count`.
///
/// # Notes
///
/// This does not cache. The scan path has its own caching loader. A writer reads each DV once.
pub async fn load_delete_vector(file_io: &FileIO, delete_file: &DataFile) -> Result<DeleteVector> {
    let file_path = delete_file.file_path();
    if delete_file.content_type() != DataContentType::PositionDeletes
        || delete_file.file_format() != DataFileFormat::Puffin
    {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Invalid deletion vector '{file_path}': expected a Puffin position-delete file, got {:?} / {:?}",
                delete_file.content_type(),
                delete_file.file_format()
            ),
        ));
    }

    let coordinates = validate_delete_vector_coordinates(
        file_path,
        delete_file.referenced_data_file(),
        delete_file.content_offset(),
        delete_file.content_size_in_bytes(),
    )?;
    let end = coordinates
        .content_offset
        .checked_add(coordinates.content_size_in_bytes)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid deletion vector '{file_path}': offset {} + length {} overflows",
                    coordinates.content_offset, coordinates.content_size_in_bytes
                ),
            )
        })?;

    let blob = file_io
        .new_input(file_path)?
        .reader()
        .await?
        .read(coordinates.content_offset..end)
        .await?;
    let delete_vector = DeleteVector::deserialize_deletion_vector_v1(&blob)?;

    // A mismatch means the manifest and the blob disagree about how many rows are deleted.
    if delete_vector.len() != delete_file.record_count() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Invalid deletion vector cardinality for '{file_path}': decoded {} positions, manifest record_count expects {}",
                delete_vector.len(),
                delete_file.record_count()
            ),
        ));
    }
    Ok(delete_vector)
}

/// An in-memory set of deleted row positions for one data file. The Rust analogue of Java
/// `BitmapPositionDeleteIndex`. A 64-bit roaring treemap keeps large or sparse sets compact.
#[derive(Debug, Default, Clone)]
pub struct DeleteVector {
    inner: RoaringTreemap,
}

/// Size in bytes of the big-endian `u32` length prefix of a `deletion-vector-v1` blob
/// (Java `BitmapPositionDeleteIndex.LENGTH_SIZE_BYTES`).
const DV_LENGTH_PREFIX_SIZE: usize = 4;
/// Size in bytes of the magic sequence (Java `BitmapPositionDeleteIndex.MAGIC_NUMBER_SIZE_BYTES`).
const DV_MAGIC_SIZE: usize = 4;
/// Size in bytes of the big-endian CRC-32 trailer (Java `BitmapPositionDeleteIndex.CRC_SIZE_BYTES`).
const DV_CRC_SIZE: usize = 4;
/// The `deletion-vector-v1` magic sequence as it appears on disk. The Puffin spec fixes it at
/// `D1 D3 39 64`, the little-endian form of Java `BitmapPositionDeleteIndex.MAGIC_NUMBER`.
const DV_MAGIC_BYTES: [u8; DV_MAGIC_SIZE] = [0xD1, 0xD3, 0x39, 0x64];
/// Size in bytes of the little-endian `u64` bitmap count that starts the portable 64-bit roaring
/// serialization (Java `RoaringPositionBitmap.BITMAP_COUNT_SIZE_BYTES`).
const DV_BITMAP_COUNT_SIZE: usize = 8;
/// Size in bytes of one little-endian `u32` bitmap key (Java
/// `RoaringPositionBitmap.BITMAP_KEY_SIZE_BYTES`).
const DV_BITMAP_KEY_SIZE: usize = 4;
/// The minimum serialized size of one (key, 32-bit bitmap) pair. It rejects a hostile bitmap
/// count that cannot fit in the payload, before the decoder loops over it.
const DV_MIN_BITMAP_ENTRY_SIZE: u64 = (DV_BITMAP_KEY_SIZE + 8) as u64;
/// The largest key Java `RoaringPositionBitmap.readKey` accepts. A key with the sign bit set
/// reads as negative in Java, so Java rejects it too.
const DV_MAX_BITMAP_KEY: u32 = i32::MAX as u32 - 1;

fn dv_blob_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataInvalid, message)
}

impl DeleteVector {
    /// Wraps an existing 64-bit roaring treemap of deleted positions as a `DeleteVector`.
    pub fn new(roaring_treemap: RoaringTreemap) -> DeleteVector {
        DeleteVector {
            inner: roaring_treemap,
        }
    }

    /// Iterates the deleted positions in ascending order.
    pub fn iter(&self) -> DeleteVectorIterator<'_> {
        let outer = self.inner.bitmaps();
        DeleteVectorIterator { outer, inner: None }
    }

    /// Marks `pos` as deleted; returns `true` if it was newly added (`false` if already present).
    pub fn insert(&mut self, pos: u64) -> bool {
        self.inner.insert(pos)
    }

    /// Unions every position of `other` into this vector. Java `PositionDeleteIndex.merge`.
    ///
    /// [`crate::writer::base_writer::deletion_vector_writer::DVFileWriter`] folds a data file's
    /// previous deletes into the new vector with this. It takes `&other`, unlike the
    /// [`BitOrAssign`] impl, so the caller can still read the previous vector's source files.
    pub fn merge(&mut self, other: &DeleteVector) {
        self.inner |= &other.inner;
    }

    /// Marks the given `positions` as deleted and returns the number of elements appended.
    ///
    /// The input slice must ascend strictly, and every value must exceed all existing values.
    ///
    /// # Errors
    ///
    /// Returns an error if the precondition is not met.
    #[allow(dead_code)]
    pub fn insert_positions(&mut self, positions: &[u64]) -> Result<usize> {
        if let Err(err) = self.inner.append(positions.iter().copied()) {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "failed to marks rows as deleted".to_string(),
            )
            .with_source(err));
        }
        Ok(positions.len())
    }

    /// Returns the number of deleted positions (the cardinality of the set).
    pub fn len(&self) -> u64 {
        self.inner.len()
    }

    /// Returns `true` if no positions are deleted (the set is empty).
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns `true` if `pos` is deleted. The Avro scan path applies positional deletes row by
    /// row with this. The Parquet path builds a `RowSelection` from the same bitmap.
    pub fn contains(&self, pos: u64) -> bool {
        self.inner.contains(pos)
    }

    /// Deserializes a Puffin `deletion-vector-v1` blob payload into a [`DeleteVector`].
    ///
    /// `blob` must be exactly the bytes the `DeleteFile` coordinates describe.
    ///
    /// | Bytes | Content |
    /// |---|---|
    /// | 4 | big-endian `u32` length of (magic + bitmap) |
    /// | 4 | the magic sequence `D1 D3 39 64` |
    /// | n | the portable 64-bit roaring bitmap |
    /// | 4 | big-endian `u32` CRC-32 (zlib) of (magic + bitmap) |
    ///
    /// # Errors
    ///
    /// These bytes are untrusted. Every framing violation returns [`ErrorKind::DataInvalid`] and
    /// names what failed. The parser never panics. Java `BitmapPositionDeleteIndex.deserialize`.
    ///
    /// # Notes
    ///
    /// The CRC is verified before the bitmap parses. Java parses first and checks the CRC last.
    /// Both sides require every check to pass, so the accepted-input set is identical.
    pub fn deserialize_deletion_vector_v1(blob: &[u8]) -> Result<DeleteVector> {
        // 1. The big-endian length prefix covering magic + bitmap.
        let length_prefix: [u8; DV_LENGTH_PREFIX_SIZE] = blob
            .get(..DV_LENGTH_PREFIX_SIZE)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or_else(|| {
                dv_blob_error(format!(
                    "Invalid deletion vector blob: {} bytes is too short to hold the 4-byte length prefix",
                    blob.len()
                ))
            })?;
        let bitmap_data_length = u64::from(u32::from_be_bytes(length_prefix));

        // The slice IS content_size_in_bytes bytes, so Java `readBitmapDataLength`'s check
        // becomes: the declared length plus prefix and CRC equals the slice length exactly.
        let expected_total = (DV_LENGTH_PREFIX_SIZE + DV_CRC_SIZE) as u64 + bitmap_data_length;
        if blob.len() as u64 != expected_total {
            return Err(dv_blob_error(format!(
                "Invalid deletion vector blob: length prefix declares {bitmap_data_length} bytes \
                 of magic + bitmap (total {expected_total}), but the blob is {} bytes",
                blob.len()
            )));
        }
        if bitmap_data_length < (DV_MAGIC_SIZE + DV_BITMAP_COUNT_SIZE) as u64 {
            return Err(dv_blob_error(format!(
                "Invalid deletion vector blob: declared magic + bitmap length \
                 {bitmap_data_length} is shorter than the minimum {} (magic + bitmap count)",
                DV_MAGIC_SIZE + DV_BITMAP_COUNT_SIZE
            )));
        }
        // The equality check above proved `bitmap_data_length == blob.len() - 8`.
        let bitmap_data_end = blob.len() - DV_CRC_SIZE;
        let bitmap_data = &blob[DV_LENGTH_PREFIX_SIZE..bitmap_data_end];

        // 2. The magic sequence (Java `deserializeBitmap`: "Invalid magic number").
        let magic = &bitmap_data[..DV_MAGIC_SIZE];
        if magic != DV_MAGIC_BYTES {
            return Err(dv_blob_error(format!(
                "Invalid deletion vector magic: {magic:02X?}, expected {DV_MAGIC_BYTES:02X?}"
            )));
        }

        // 3. The CRC-32 trailer over magic + bitmap. Checked before the bitmap parses, so
        //    corrupt bytes never reach the bitmap parser.
        let mut crc = flate2::Crc::new();
        crc.update(bitmap_data);
        let actual_crc = crc.sum();
        let expected_crc_bytes: [u8; DV_CRC_SIZE] = blob[bitmap_data_end..]
            .try_into()
            .expect("length validated above: exactly DV_CRC_SIZE bytes remain");
        let expected_crc = u32::from_be_bytes(expected_crc_bytes);
        if actual_crc != expected_crc {
            return Err(dv_blob_error(format!(
                "Invalid deletion vector CRC: computed {actual_crc:#010X}, stored {expected_crc:#010X}"
            )));
        }

        // 4. The portable 64-bit roaring bitmap (after the magic).
        Self::deserialize_portable_bitmap(&bitmap_data[DV_MAGIC_SIZE..])
    }

    /// Parses the portable 64-bit roaring serialization: a little-endian `u64` bitmap count, then
    /// per bitmap an ascending little-endian `u32` key and a standard-format 32-bit roaring
    /// bitmap. Java `RoaringPositionBitmap.deserialize`.
    ///
    /// The region must be consumed exactly. Trailing bytes are corruption, and are rejected.
    fn deserialize_portable_bitmap(region: &[u8]) -> Result<DeleteVector> {
        let mut cursor = std::io::Cursor::new(region);

        let mut count_bytes = [0u8; DV_BITMAP_COUNT_SIZE];
        cursor.read_exact(&mut count_bytes).map_err(|source| {
            dv_blob_error("Invalid deletion vector bitmap: truncated before the bitmap count")
                .with_source(source)
        })?;
        let bitmap_count = u64::from_le_bytes(count_bytes);

        // Java `readBitmapCount` rejects counts above Integer.MAX_VALUE. We also reject a count
        // that cannot fit in the remaining bytes, so a hostile count fails fast and by name.
        if bitmap_count > i32::MAX as u64 {
            return Err(dv_blob_error(format!(
                "Invalid deletion vector bitmap count: {bitmap_count} exceeds the maximum {}",
                i32::MAX
            )));
        }
        let remaining = (region.len() - DV_BITMAP_COUNT_SIZE) as u64;
        if bitmap_count > remaining / DV_MIN_BITMAP_ENTRY_SIZE {
            return Err(dv_blob_error(format!(
                "Invalid deletion vector bitmap count: {bitmap_count} bitmaps cannot fit in the \
                 {remaining} remaining payload bytes"
            )));
        }

        let mut treemap = RoaringTreemap::new();
        let mut last_key: Option<u32> = None;
        for _ in 0..bitmap_count {
            let mut key_bytes = [0u8; DV_BITMAP_KEY_SIZE];
            cursor.read_exact(&mut key_bytes).map_err(|source| {
                dv_blob_error("Invalid deletion vector bitmap: truncated before a bitmap key")
                    .with_source(source)
            })?;
            let key = u32::from_le_bytes(key_bytes);

            // Java `readKey`: keys are non-negative, at most `i32::MAX - 1`, and strictly ascend.
            if key > DV_MAX_BITMAP_KEY {
                return Err(dv_blob_error(format!(
                    "Invalid deletion vector bitmap key: {key} exceeds the maximum {DV_MAX_BITMAP_KEY}"
                )));
            }
            if let Some(last) = last_key
                && key <= last
            {
                return Err(dv_blob_error(format!(
                    "Invalid deletion vector bitmap key order: key {key} follows {last}, keys \
                     must be strictly ascending"
                )));
            }

            // The checked deserializer consumes exactly the bitmap's serialized bytes.
            let bitmap = RoaringBitmap::deserialize_from(&mut cursor).map_err(|source| {
                dv_blob_error(format!(
                    "Invalid deletion vector: malformed 32-bit roaring bitmap for key {key}"
                ))
                .with_source(source)
            })?;

            // The key holds the high 32 bits and the bitmap the low 32. Keys ascend strictly and
            // each bitmap iterates ascending, so the appended sequence is strictly ascending.
            let high_bits = u64::from(key) << 32;
            treemap
                .append(bitmap.iter().map(|low| high_bits | u64::from(low)))
                .map_err(|source| {
                    dv_blob_error(format!(
                        "Invalid deletion vector: positions for key {key} are not strictly ascending"
                    ))
                    .with_source(source)
                })?;

            last_key = Some(key);
        }

        // Leftover bytes mean the length prefix and the bitmap disagree. That is corruption.
        let consumed = cursor.position();
        if consumed != region.len() as u64 {
            return Err(dv_blob_error(format!(
                "Invalid deletion vector bitmap: {} trailing bytes after the declared {bitmap_count} bitmaps",
                region.len() as u64 - consumed
            )));
        }

        Ok(DeleteVector { inner: treemap })
    }

    /// Serializes this vector as a Puffin `deletion-vector-v1` blob payload. The bytes match what
    /// Java `BitmapPositionDeleteIndex.serialize` writes for the same position set.
    ///
    /// | Bytes | Content |
    /// |---|---|
    /// | 4 | big-endian `u32` length of (magic + bitmap) |
    /// | 4 | the magic sequence `D1 D3 39 64` |
    /// | n | the dense portable 64-bit roaring bitmap |
    /// | 4 | big-endian `u32` CRC-32 (zlib) of (magic + bitmap) |
    ///
    /// # Notes
    ///
    /// The bitmap count is DENSE. Java writes `max key + 1` entries, and includes an empty bitmap
    /// for every gap key. `roaring-rs` writes a sparse count, so the outer framing is hand-rolled
    /// here. Readers accept both forms, but byte parity with Java needs the dense one.
    ///
    /// Each sub-bitmap is run-length encoded on a clone first, like Java `runLengthEncode()`, so
    /// `&self` stays unmutated. For array and bitmap stores [`RoaringBitmap::optimize`] uses
    /// Java's run-iff-strictly-smaller criterion, so container choice matches on exact ties too.
    /// One byte-parity divergence remains, and only a re-serialized DESERIALIZED vector reaches
    /// it: for a store already a run container, `roaring-rs` omits Java's 2-byte cardinality
    /// overhead, so at `cardinality == 2 * runs` Java keeps the run and we emit the array.
    ///
    /// # Errors
    ///
    /// - the vector is EMPTY. `BaseDVFileWriter` never serializes an empty index, and a
    ///   cardinality-0 DV `DeleteFile` is meaningless. Fail loud instead of writing one.
    /// - a position's high 32 bits exceed `i32::MAX - 1`. Java's dense bitmap array cannot hold
    ///   that key, but our `RoaringTreemap` can, so the door check lives here.
    /// - the blob would exceed 2 GB. Java `computeBitmapDataLength` rejects the same. Checked
    ///   before the buffer is allocated.
    pub fn serialize_deletion_vector_v1(&self) -> Result<Vec<u8>> {
        if self.inner.is_empty() {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "Cannot serialize an empty deletion vector: a deletion-vector-v1 blob must \
                 delete at least one position (BaseDVFileWriter never writes an empty DV)",
            ));
        }

        // Run-length encode each present sub-bitmap (on a clone) and index them by key.
        let mut optimized_by_key: std::collections::BTreeMap<u32, RoaringBitmap> =
            std::collections::BTreeMap::new();
        for (key, bitmap) in self.inner.bitmaps() {
            if key > DV_MAX_BITMAP_KEY {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot serialize deletion vector: bitmap key {key} exceeds the maximum \
                         {DV_MAX_BITMAP_KEY} Java's dense bitmap array can represent"
                    ),
                ));
            }
            let mut optimized = bitmap.clone();
            optimized.optimize();
            optimized_by_key.insert(key, optimized);
        }
        let max_key = *optimized_by_key
            .keys()
            .next_back()
            .expect("non-empty vector has at least one sub-bitmap");
        let dense_bitmap_count = u64::from(max_key) + 1;

        // Enforce the 2 GB bound BEFORE allocating. Every dense slot carries a key, and an absent
        // key adds an empty bitmap. The count is O(present keys), so a hostile sparse high key is
        // rejected without a walk of the dense range.
        let empty_bitmap = RoaringBitmap::new();
        let empty_bitmap_size = empty_bitmap.serialized_size() as u64;
        let present_bitmaps_size: u64 = optimized_by_key
            .values()
            .map(|bitmap| bitmap.serialized_size() as u64)
            .sum();
        let absent_bitmap_count = dense_bitmap_count - optimized_by_key.len() as u64;
        let portable_bitmap_size = DV_BITMAP_COUNT_SIZE as u64
            + dense_bitmap_count * DV_BITMAP_KEY_SIZE as u64
            + present_bitmaps_size
            + absent_bitmap_count * empty_bitmap_size;
        let bitmap_data_length = DV_MAGIC_SIZE as u64 + portable_bitmap_size;
        let total_blob_size = (DV_LENGTH_PREFIX_SIZE + DV_CRC_SIZE) as u64 + bitmap_data_length;
        if total_blob_size > i32::MAX as u64 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot serialize deletion vector: blob would be {total_blob_size} bytes, \
                     which exceeds the 2GB limit (Java BitmapPositionDeleteIndex rejects \
                     indexes > Integer.MAX_VALUE bytes)"
                ),
            ));
        }
        // `as` would silently truncate here. `try_from` keeps the range proof local.
        let bitmap_data_length_u32 = u32::try_from(bitmap_data_length)
            .expect("bitmap data length bounded by the 2GB check above");

        // 1. + 2. The big-endian length prefix and the magic.
        let mut blob = Vec::with_capacity(total_blob_size as usize);
        blob.extend_from_slice(&bitmap_data_length_u32.to_be_bytes());
        blob.extend_from_slice(&DV_MAGIC_BYTES);

        // 3. The DENSE portable 64-bit roaring bitmap.
        blob.extend_from_slice(&dense_bitmap_count.to_le_bytes());
        for key in 0..=max_key {
            blob.extend_from_slice(&key.to_le_bytes());
            let bitmap = optimized_by_key.get(&key).unwrap_or(&empty_bitmap);
            bitmap.serialize_into(&mut blob).map_err(|source| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to serialize the 32-bit roaring bitmap for key {key}"),
                )
                .with_source(source)
            })?;
        }

        // 4. The big-endian CRC-32 (zlib) of magic + bitmap (everything after the length prefix).
        let mut crc = flate2::Crc::new();
        crc.update(&blob[DV_LENGTH_PREFIX_SIZE..]);
        blob.extend_from_slice(&crc.sum().to_be_bytes());

        debug_assert_eq!(blob.len() as u64, total_blob_size);
        Ok(blob)
    }
}

// `roaring::treemap::Iter` has no `advance_to`, which ArrowReader::build_deletes_row_selection
// needs, so this type walks the bitmaps itself. Upstream PR:
// https://github.com/RoaringBitmap/roaring-rs/pull/314.
/// Ascending iterator over a [`DeleteVector`]'s positions, with an [`advance_to`](Self::advance_to)
/// fast-forward used by the scan-side row-selection builder.
pub struct DeleteVectorIterator<'a> {
    // `BitmapIter` is public only in an unreleased roaring version, so Cargo.toml pins a git ref.
    outer: BitmapIter<'a>,
    inner: Option<DeleteVectorIteratorInner<'a>>,
}

struct DeleteVectorIteratorInner<'a> {
    high_bits: u32,
    bitmap_iter: Iter<'a>,
}

impl Iterator for DeleteVectorIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(inner) = &mut self.inner
            && let Some(inner_next) = inner.bitmap_iter.next()
        {
            return Some(u64::from(inner.high_bits) << 32 | u64::from(inner_next));
        }

        if let Some((high_bits, next_bitmap)) = self.outer.next() {
            self.inner = Some(DeleteVectorIteratorInner {
                high_bits,
                bitmap_iter: next_bitmap.iter(),
            })
        } else {
            return None;
        }

        self.next()
    }
}

impl DeleteVectorIterator<'_> {
    /// Fast-forwards the iterator so the next yielded position is the smallest delete `>= pos`.
    ///
    /// `pos` splits into the high-bits group `hi = pos >> 32` and the low value `lo = pos as u32`.
    /// The outer walk steps groups until the current group is `>= hi`. It skips within the bitmap
    /// only when it lands exactly on `hi`.
    ///
    /// # Notes
    ///
    /// When `hi`'s group is ABSENT, the outer walk overshoots into the next present group. Every
    /// position there is already `> pos`, so the inner skip must not run. It would consume an
    /// in-range position. The iterator therefore stops at that group's start.
    ///
    /// The call does nothing until one `next()` primes the iterator. It only moves forward.
    pub fn advance_to(&mut self, pos: u64) {
        let hi = (pos >> 32) as u32;
        let lo = pos as u32;

        let Some(ref mut inner) = self.inner else {
            return;
        };

        while inner.high_bits < hi {
            let Some((next_hi, next_bitmap)) = self.outer.next() else {
                return;
            };

            *inner = DeleteVectorIteratorInner {
                high_bits: next_hi,
                bitmap_iter: next_bitmap.iter(),
            }
        }

        // On overshoot into a higher group, every position there is already `> pos`. Leave
        // `bitmap_iter` at that group's start.
        if inner.high_bits == hi {
            inner.bitmap_iter.advance_to(lo);
        }
    }
}

impl BitOrAssign for DeleteVector {
    fn bitor_assign(&mut self, other: Self) {
        self.inner.bitor_assign(&other.inner);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Frames `bitmap_bytes` as a `deletion-vector-v1` blob, to synthesize fixtures. The inverse
    /// of [`DeleteVector::deserialize_deletion_vector_v1`]'s framing.
    pub(crate) fn frame_deletion_vector_v1(bitmap_bytes: &[u8]) -> Vec<u8> {
        let mut bitmap_data = Vec::with_capacity(DV_MAGIC_SIZE + bitmap_bytes.len());
        bitmap_data.extend_from_slice(&DV_MAGIC_BYTES);
        bitmap_data.extend_from_slice(bitmap_bytes);

        let mut crc = flate2::Crc::new();
        crc.update(&bitmap_data);

        let mut blob = Vec::with_capacity(DV_LENGTH_PREFIX_SIZE + bitmap_data.len() + DV_CRC_SIZE);
        blob.extend_from_slice(
            &u32::try_from(bitmap_data.len())
                .expect("test blob < 4GB")
                .to_be_bytes(),
        );
        blob.extend_from_slice(&bitmap_data);
        blob.extend_from_slice(&crc.sum().to_be_bytes());
        blob
    }

    /// Encodes a NON-EMPTY position set through the production serializer, so the fixtures carry
    /// Java's dense layout.
    pub(crate) fn encode_deletion_vector_v1(positions: &[u64]) -> Vec<u8> {
        let treemap: RoaringTreemap = positions.iter().copied().collect();
        DeleteVector::new(treemap)
            .serialize_deletion_vector_v1()
            .expect("serialize test positions")
    }

    /// Encodes explicit (key, 32-bit bitmap) pairs, for run-container and malformed-order
    /// fixtures that need full control of the layout.
    fn encode_deletion_vector_v1_from_pairs(pairs: &[(u32, RoaringBitmap)]) -> Vec<u8> {
        let mut bitmap_bytes = Vec::new();
        bitmap_bytes.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
        for (key, bitmap) in pairs {
            bitmap_bytes.extend_from_slice(&key.to_le_bytes());
            bitmap
                .serialize_into(&mut bitmap_bytes)
                .expect("serialize test bitmap");
        }
        frame_deletion_vector_v1(&bitmap_bytes)
    }

    /// Recomputes and rewrites the CRC trailer after a deliberate payload mutation, so a test can
    /// reach the validations BEHIND the CRC check (magic, count, keys, bitmap structure).
    fn rewrite_valid_crc(blob: &mut [u8]) {
        let end = blob.len() - DV_CRC_SIZE;
        let mut crc = flate2::Crc::new();
        crc.update(&blob[DV_LENGTH_PREFIX_SIZE..end]);
        blob[end..].copy_from_slice(&crc.sum().to_be_bytes());
    }

    /// Asserts decode rejects `blob` with a `DataInvalid` error naming `expected_fragment`. A
    /// panic inside fails the test, which pins the no-panic contract for malformed input.
    fn assert_rejects(blob: &[u8], expected_fragment: &str) {
        let error = DeleteVector::deserialize_deletion_vector_v1(blob)
            .expect_err("malformed deletion vector blob must be rejected");
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            error.to_string().contains(expected_fragment),
            "error {error} does not name the failure (expected fragment {expected_fragment:?})"
        );
    }

    /// Risk pinned: a framing or decode regression that silently changes the position set. The
    /// exact set, including values across the 32-bit key boundary, must survive a round-trip.
    #[test]
    fn test_dv_blob_round_trip_preserves_positions_across_key_boundary() {
        let positions = [0u64, 5, 1022, (1u64 << 32) + 5, (1u64 << 33) + 1];
        let blob = encode_deletion_vector_v1(&positions);

        let decoded =
            DeleteVector::deserialize_deletion_vector_v1(&blob).expect("valid blob must decode");

        let decoded_positions: Vec<u64> = decoded.iter().collect();
        assert_eq!(decoded_positions, positions);
        assert_eq!(decoded.len(), positions.len() as u64);
    }

    /// Risk pinned: the high-bits reassembly `(key << 32) | low`. Drop the shift and positions
    /// above 2^32 collapse onto their low words.
    #[test]
    fn test_dv_blob_positions_above_u32_range_keep_high_bits() {
        let positions = [7u64, (1u64 << 32) + 7, (5u64 << 32) + 7];
        let blob = encode_deletion_vector_v1(&positions);

        let decoded =
            DeleteVector::deserialize_deletion_vector_v1(&blob).expect("valid blob must decode");

        let decoded_positions: Vec<u64> = decoded.iter().collect();
        assert_eq!(
            decoded_positions, positions,
            "positions above 2^32 must keep their high 32 bits"
        );
    }

    /// Risk pinned: a DV of 0 bitmaps must decode to an empty vector, not error. Java's portable
    /// format legally encodes zero bitmaps. The frame is raw because the production serializer
    /// refuses empty vectors.
    #[test]
    fn test_dv_blob_empty_vector_decodes_to_zero_positions() {
        let blob = frame_deletion_vector_v1(&0u64.to_le_bytes());
        let decoded =
            DeleteVector::deserialize_deletion_vector_v1(&blob).expect("empty blob must decode");
        assert_eq!(decoded.len(), 0);
    }

    /// Risk pinned: a serializer regression that changes ANY byte of the on-disk encoding, which
    /// corrupts the table for every other reader. The expected bytes for positions
    /// {0, 5, 2^32+1} are hand-computed from the Puffin and Roaring format specs, independent of
    /// this code. The array below names each field.
    #[test]
    fn test_dv_serialize_golden_bytes_hand_computed() {
        let treemap: RoaringTreemap = [0u64, 5, (1u64 << 32) + 1].into_iter().collect();
        let blob = DeleteVector::new(treemap)
            .serialize_deletion_vector_v1()
            .expect("serialize golden fixture");

        #[rustfmt::skip]
        let expected: [u8; 66] = [
            // BE u32 length of magic + bitmap = 58
            0x00, 0x00, 0x00, 0x3A,
            // LE magic
            0xD1, 0xD3, 0x39, 0x64,
            // LE u64 dense bitmap count = 2
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // key 0 (LE u32) + standard bitmap {0, 5}
            0x00, 0x00, 0x00, 0x00,
            0x3A, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00,
            // key 1 (LE u32) + standard bitmap {1}
            0x01, 0x00, 0x00, 0x00,
            0x3A, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x10, 0x00, 0x00, 0x00, 0x01, 0x00,
            // BE u32 CRC-32 of magic + bitmap
            0x9A, 0xCC, 0x8C, 0xA4,
        ];
        assert_eq!(
            blob, expected,
            "serialized DV blob must match the hand-computed bytes"
        );
    }

    /// Risk pinned: the DENSE-layout contract. Java writes `max key + 1` entries and includes
    /// empty gap bitmaps. A sparse encoding is readable but not byte-identical. Positions
    /// {0, 2^33} take keys 0 and 2, so the blob must declare count 3 and carry an empty key-1
    /// entry.
    #[test]
    fn test_dv_serialize_dense_gap_writes_empty_middle_bitmap_like_java() {
        let treemap: RoaringTreemap = [0u64, 1u64 << 33].into_iter().collect();
        let blob = DeleteVector::new(treemap)
            .serialize_deletion_vector_v1()
            .expect("serialize dense-gap fixture");

        #[rustfmt::skip]
        let expected: [u8; 76] = [
            // BE u32 length of magic + bitmap = 68
            0x00, 0x00, 0x00, 0x44,
            // LE magic
            0xD1, 0xD3, 0x39, 0x64,
            // LE u64 dense bitmap count = 3 (NOT 2: the gap key 1 is included)
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // key 0 + standard bitmap {0}
            0x00, 0x00, 0x00, 0x00,
            0x3A, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
            // key 1 + EMPTY standard bitmap (cookie 12346, 0 containers) — the dense gap entry
            0x01, 0x00, 0x00, 0x00,
            0x3A, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // key 2 + standard bitmap {0}
            0x02, 0x00, 0x00, 0x00,
            0x3A, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
            // BE u32 CRC-32 of magic + bitmap
            0xBC, 0x98, 0x85, 0x1A,
        ];
        assert_eq!(
            blob, expected,
            "dense layout must include the empty key-1 gap bitmap, like Java writes it"
        );
    }

    /// Risk pinned: the serializer must round-trip through the decoder for every shape the writer
    /// produces. Real Java bytes proved the decoder, so this round-trip is the in-house
    /// byte-compatibility floor.
    #[test]
    fn test_dv_serialize_round_trips_through_decoder() {
        let shapes: Vec<Vec<u64>> = vec![
            vec![0],
            vec![0, 5, (1u64 << 32) + 1],
            vec![7, (1u64 << 32) + 7, (5u64 << 32) + 7], // gap keys 2..=4
            (1000..6000).collect(),                      // run-shaped
        ];
        // A key near DV_MAX_BITMAP_KEY cannot round-trip: the dense layout makes the blob > 2GB,
        // and Java's serialize refuses it too. `test_dv_blob_max_valid_key_boundary_accepted`
        // pins the decode-side accept boundary.
        for positions in shapes {
            let treemap: RoaringTreemap = positions.iter().copied().collect();
            let blob = DeleteVector::new(treemap)
                .serialize_deletion_vector_v1()
                .expect("serialize round-trip fixture");
            let decoded = DeleteVector::deserialize_deletion_vector_v1(&blob)
                .expect("production-serialized blob must decode");
            let decoded_positions: Vec<u64> = decoded.iter().collect();
            assert_eq!(decoded_positions, positions);
        }
    }

    /// Risk pinned: `runLengthEncode` parity. Java run-length encodes before it serializes, so a
    /// 5000-long run becomes a RUN container (cookie 12347). Skip our `optimize()` and the bytes
    /// become a larger array container. That is readable, but not byte-identical to Java.
    #[test]
    fn test_dv_serialize_run_shaped_input_emits_run_container_like_java() {
        let treemap: RoaringTreemap = (1000u64..6000).collect();
        let blob = DeleteVector::new(treemap)
            .serialize_deletion_vector_v1()
            .expect("serialize run fixture");

        // The first sub-bitmap starts after prefix(4) + magic(4) + count(8) + key(4). Its first
        // two bytes are the cookie: 12347 means it contains run containers.
        let cookie_at =
            DV_LENGTH_PREFIX_SIZE + DV_MAGIC_SIZE + DV_BITMAP_COUNT_SIZE + DV_BITMAP_KEY_SIZE;
        let cookie = u16::from_le_bytes([blob[cookie_at], blob[cookie_at + 1]]);
        assert_eq!(
            cookie, 12347,
            "a 5000-run must serialize as a run container"
        );

        let decoded = DeleteVector::deserialize_deletion_vector_v1(&blob)
            .expect("run-container blob must decode");
        assert_eq!(decoded.len(), 5000);
    }

    /// Risk pinned: the run-versus-array TIE. Positions {0,1,2} cost 6 bytes either way. Java
    /// converts only when the array is strictly larger, so the tie keeps the ARRAY container, and
    /// `roaring-rs` agrees. A drift to `>=` or `<` on either side flips the container cookie and
    /// breaks byte parity. `tests/interop_dv_write.rs` settles the same tie against Java.
    #[test]
    fn test_dv_serialize_array_run_size_tie_keeps_array_container_like_java() {
        let treemap: RoaringTreemap = [0u64, 1, 2].into_iter().collect();
        let blob = DeleteVector::new(treemap)
            .serialize_deletion_vector_v1()
            .expect("serialize tie fixture");

        let cookie_at =
            DV_LENGTH_PREFIX_SIZE + DV_MAGIC_SIZE + DV_BITMAP_COUNT_SIZE + DV_BITMAP_KEY_SIZE;
        let cookie = u16::from_le_bytes([blob[cookie_at], blob[cookie_at + 1]]);
        assert_eq!(
            cookie, 12346,
            "the exact array/run size tie must KEEP the array container (cookie 12346), like \
             Java's strictly-smaller criterion"
        );

        let decoded =
            DeleteVector::deserialize_deletion_vector_v1(&blob).expect("tie-case blob must decode");
        assert_eq!(decoded.iter().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    /// Risk pinned: serializing an EMPTY vector must fail loud. `BaseDVFileWriter` never writes
    /// one, and a cardinality-0 DV `DeleteFile` is a meaningless table entry.
    #[test]
    fn test_dv_serialize_empty_vector_rejected() {
        let error = DeleteVector::default()
            .serialize_deletion_vector_v1()
            .expect_err("empty vector must not serialize");
        assert!(
            error.to_string().contains("empty deletion vector"),
            "error must name the empty-vector rejection, got: {error}"
        );
    }

    /// Risk pinned: a key above `i32::MAX - 1` does not fit Java's dense bitmap array, so writing
    /// it makes a blob Java's `readKey` rejects. Our treemap can hold such a position, so the
    /// serializer is the door.
    #[test]
    fn test_dv_serialize_key_above_java_max_rejected() {
        let mut treemap = RoaringTreemap::new();
        treemap.insert(u64::from(i32::MAX as u32) << 32); // key == i32::MAX > MAX-1
        let error = DeleteVector::new(treemap)
            .serialize_deletion_vector_v1()
            .expect_err("key i32::MAX must not serialize");
        assert!(
            error.to_string().contains("exceeds the maximum"),
            "error must name the key bound, got: {error}"
        );
    }

    /// Risk pinned: the 2GB bound must fire from the size pre-computation, before any allocation.
    /// One position with a huge key forces a dense count of about 179M entries, near 2.15 GB.
    #[test]
    fn test_dv_serialize_over_2gb_rejected_before_allocating() {
        let mut treemap = RoaringTreemap::new();
        treemap.insert(179_000_000u64 << 32);
        let error = DeleteVector::new(treemap)
            .serialize_deletion_vector_v1()
            .expect_err("a >2GB blob must not serialize");
        assert!(
            error.to_string().contains("2GB"),
            "error must name the 2GB bound, got: {error}"
        );
    }

    /// Risk pinned: real DV blobs carry RUN containers, because Java run-length encodes before it
    /// serializes. The decoder must take that container path, not only the array path.
    #[test]
    fn test_dv_blob_run_length_container_decodes() {
        let mut dense = RoaringBitmap::new();
        dense.insert_range(1000..200_000);
        dense.optimize(); // run-length encode wherever smaller, like Java's runLengthEncode()

        // Prove the fixture really carries run containers. Cookie 12347 means it does, 12346
        // means it does not.
        let mut serialized = Vec::new();
        dense
            .serialize_into(&mut serialized)
            .expect("serialize dense fixture bitmap");
        let cookie = u16::from_le_bytes([serialized[0], serialized[1]]);
        assert_eq!(
            cookie, 12347,
            "fixture must actually contain run containers"
        );

        let blob = encode_deletion_vector_v1_from_pairs(&[(0, dense)]);
        let decoded = DeleteVector::deserialize_deletion_vector_v1(&blob)
            .expect("run-container blob must decode");

        assert_eq!(decoded.len(), 199_000);
        let decoded_positions: Vec<u64> = decoded.iter().collect();
        assert_eq!(decoded_positions[0], 1000);
        assert_eq!(decoded_positions[198_999], 199_999);
    }

    /// Risk pinned: Java pads the bitmap array densely from key 0 to the max key, so a blob with
    /// only high keys carries EMPTY gap bitmaps. They must decode as no positions, not error.
    #[test]
    fn test_dv_blob_with_empty_gap_bitmap_decodes_like_java_writes_it() {
        let mut key0 = RoaringBitmap::new();
        key0.insert(3);
        let key1_empty = RoaringBitmap::new();
        let mut key2 = RoaringBitmap::new();
        key2.insert(9);

        let blob = encode_deletion_vector_v1_from_pairs(&[(0, key0), (1, key1_empty), (2, key2)]);
        let decoded = DeleteVector::deserialize_deletion_vector_v1(&blob)
            .expect("blob with an empty gap bitmap must decode");

        let decoded_positions: Vec<u64> = decoded.iter().collect();
        assert_eq!(decoded_positions, vec![3, (2u64 << 32) + 9]);
    }

    /// Risk pinned: truncated input at every framing boundary must give a clean error, never a
    /// panic. A truncated valid blob always trips the total-length equality first. The
    /// bitmap-garbage tests below cover the inner truncation paths.
    #[test]
    fn test_dv_blob_truncation_at_each_boundary_rejects_cleanly() {
        let blob = encode_deletion_vector_v1(&[1, 2, 3]);

        // Shorter than the 4-byte length prefix.
        assert_rejects(&[], "too short to hold the 4-byte length prefix");
        assert_rejects(&blob[..3], "too short to hold the 4-byte length prefix");
        // Cut inside the magic, the bitmap, and the CRC trailer.
        assert_rejects(&blob[..6], "length prefix declares");
        assert_rejects(&blob[..blob.len() / 2], "length prefix declares");
        assert_rejects(&blob[..blob.len() - 2], "length prefix declares");
    }

    /// Risk pinned: a wrong magic must be rejected by NAME even when the CRC is valid for the
    /// corrupted bytes. The magic check does not depend on the checksum.
    #[test]
    fn test_dv_blob_wrong_magic_rejects() {
        let mut blob = encode_deletion_vector_v1(&[1, 2, 3]);
        blob[DV_LENGTH_PREFIX_SIZE] ^= 0xFF;
        rewrite_valid_crc(&mut blob);
        assert_rejects(&blob, "Invalid deletion vector magic");
    }

    /// Risk pinned: the CRC check fires. One corrupted bitmap byte, with the stored CRC
    /// untouched, must be rejected as a CRC mismatch before the bitmap parser sees it.
    #[test]
    fn test_dv_blob_crc_mismatch_rejects() {
        let mut blob = encode_deletion_vector_v1(&[1, 2, 3]);
        let corrupt_at = blob.len() - DV_CRC_SIZE - 1;
        blob[corrupt_at] ^= 0x01;
        assert_rejects(&blob, "Invalid deletion vector CRC");

        // The stored CRC itself corrupted must also reject.
        let mut blob = encode_deletion_vector_v1(&[1, 2, 3]);
        let crc_at = blob.len() - 1;
        blob[crc_at] ^= 0x01;
        assert_rejects(&blob, "Invalid deletion vector CRC");
    }

    /// Risk pinned: a length prefix that disagrees with the payload must reject cleanly. Both
    /// directions, and the hostile u32::MAX that overflows naive arithmetic.
    #[test]
    fn test_dv_blob_length_prefix_mismatch_rejects() {
        let blob = encode_deletion_vector_v1(&[1, 2, 3]);

        let mut longer = blob.clone();
        let declared = u32::from_be_bytes(longer[..4].try_into().expect("4 bytes")) + 1;
        longer[..4].copy_from_slice(&declared.to_be_bytes());
        assert_rejects(&longer, "length prefix declares");

        let mut shorter = blob.clone();
        let declared = u32::from_be_bytes(shorter[..4].try_into().expect("4 bytes")) - 1;
        shorter[..4].copy_from_slice(&declared.to_be_bytes());
        assert_rejects(&shorter, "length prefix declares");

        let mut hostile = blob.clone();
        hostile[..4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert_rejects(&hostile, "length prefix declares");

        // An empty bitmap region declares length 4, below the 12-byte magic + count minimum.
        let tiny = frame_deletion_vector_v1(&[]);
        assert_rejects(&tiny, "shorter than the minimum");
    }

    /// Risk pinned: garbage where the 32-bit bitmap belongs, with the CRC recomputed so it
    /// passes. The checked bitmap parser must reject it, not panic and not decode it.
    #[test]
    fn test_dv_blob_garbage_bitmap_bytes_reject() {
        // count = 1, key = 0, then garbage instead of a serialized bitmap.
        let mut bitmap_bytes = Vec::new();
        bitmap_bytes.extend_from_slice(&1u64.to_le_bytes());
        bitmap_bytes.extend_from_slice(&0u32.to_le_bytes());
        bitmap_bytes.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF]);
        let blob = frame_deletion_vector_v1(&bitmap_bytes);
        assert_rejects(&blob, "malformed 32-bit roaring bitmap");
    }

    /// Risk pinned: a bitmap count larger than the payload holds must fail fast with a named
    /// error. u64::MAX included, which loops near forever or overflows naive size math.
    #[test]
    fn test_dv_blob_bitmap_count_overflow_rejects() {
        // count = u64::MAX over an empty remainder.
        let blob = frame_deletion_vector_v1(&u64::MAX.to_le_bytes());
        assert_rejects(&blob, "exceeds the maximum");

        // A count far larger than the remaining bytes could hold fails the fit bound by name.
        let single = encode_deletion_vector_v1(&[1, 2, 3]);
        let mut bitmap_bytes =
            single[DV_LENGTH_PREFIX_SIZE + DV_MAGIC_SIZE..single.len() - DV_CRC_SIZE].to_vec();
        bitmap_bytes[..8].copy_from_slice(&1000u64.to_le_bytes());
        let blob = frame_deletion_vector_v1(&bitmap_bytes);
        assert_rejects(&blob, "cannot fit");

        // count = 2 with one entry passes the minimum-size bound, then runs out of bytes.
        let mut bitmap_bytes =
            single[DV_LENGTH_PREFIX_SIZE + DV_MAGIC_SIZE..single.len() - DV_CRC_SIZE].to_vec();
        bitmap_bytes[..8].copy_from_slice(&2u64.to_le_bytes());
        let blob = frame_deletion_vector_v1(&bitmap_bytes);
        assert_rejects(&blob, "truncated before a bitmap key");
    }

    /// Risk pinned: Java `readKey` rejects out-of-range and non-ascending keys. Accepting them
    /// silently reorders or aliases position ranges.
    #[test]
    fn test_dv_blob_invalid_keys_reject() {
        let mut one = RoaringBitmap::new();
        one.insert(1);

        // Key with the sign bit set (reads negative in Java).
        let blob = encode_deletion_vector_v1_from_pairs(&[(u32::MAX, one.clone())]);
        assert_rejects(&blob, "exceeds the maximum");

        // Key == Integer.MAX_VALUE (Java allows at most MAX_VALUE - 1).
        let blob = encode_deletion_vector_v1_from_pairs(&[(i32::MAX as u32, one.clone())]);
        assert_rejects(&blob, "exceeds the maximum");

        // Non-ascending keys.
        let blob = encode_deletion_vector_v1_from_pairs(&[(5, one.clone()), (3, one.clone())]);
        assert_rejects(&blob, "strictly ascending");

        // Duplicate keys.
        let blob = encode_deletion_vector_v1_from_pairs(&[(5, one.clone()), (5, one)]);
        assert_rejects(&blob, "strictly ascending");
    }

    /// Risk pinned: denial of service by allocation through the INNER container count. The outer
    /// bitmap-count bound cannot constrain what a per-key payload claims. `roaring` must reject a
    /// hostile cookie fast and within bounded memory, never loop or panic.
    #[test]
    fn test_dv_blob_hostile_inner_container_count_rejects_fast() {
        // count = 1, key = 0, then a no-run cookie (12346) claiming u32::MAX containers.
        let mut bitmap_bytes = Vec::new();
        bitmap_bytes.extend_from_slice(&1u64.to_le_bytes());
        bitmap_bytes.extend_from_slice(&0u32.to_le_bytes());
        bitmap_bytes.extend_from_slice(&12346u32.to_le_bytes()); // SERIAL_COOKIE_NO_RUNCONTAINER
        bitmap_bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // hostile container count
        let blob = frame_deletion_vector_v1(&bitmap_bytes);
        let start = std::time::Instant::now();
        assert_rejects(&blob, "malformed 32-bit roaring bitmap");
        assert!(
            start.elapsed().as_secs() < 2,
            "must fail fast, no huge alloc/loop"
        );

        // Run cookie (12347) with max upper-16 size (65536 containers) over an empty payload.
        let mut bitmap_bytes = Vec::new();
        bitmap_bytes.extend_from_slice(&1u64.to_le_bytes());
        bitmap_bytes.extend_from_slice(&0u32.to_le_bytes());
        let run_cookie: u32 = 12347 | (0xFFFFu32 << 16);
        bitmap_bytes.extend_from_slice(&run_cookie.to_le_bytes());
        let blob = frame_deletion_vector_v1(&bitmap_bytes);
        // The outer fit bound trips first: the entry is under the 12-byte minimum.
        assert_rejects(&blob, "cannot fit");

        // Same run cookie but padded past the outer fit bound so the inner parser sees it.
        let mut bitmap_bytes = Vec::new();
        bitmap_bytes.extend_from_slice(&1u64.to_le_bytes());
        bitmap_bytes.extend_from_slice(&0u32.to_le_bytes());
        bitmap_bytes.extend_from_slice(&run_cookie.to_le_bytes());
        bitmap_bytes.extend_from_slice(&[0u8; 16]);
        let blob = frame_deletion_vector_v1(&bitmap_bytes);
        let start = std::time::Instant::now();
        assert_rejects(&blob, "malformed 32-bit roaring bitmap");
        assert!(start.elapsed().as_secs() < 2, "must fail fast");

        // No-run cookie with a count just under the inner cap (65536) but no payload.
        let mut bitmap_bytes = Vec::new();
        bitmap_bytes.extend_from_slice(&1u64.to_le_bytes());
        bitmap_bytes.extend_from_slice(&0u32.to_le_bytes());
        bitmap_bytes.extend_from_slice(&12346u32.to_le_bytes());
        bitmap_bytes.extend_from_slice(&65536u32.to_le_bytes());
        let blob = frame_deletion_vector_v1(&bitmap_bytes);
        assert_rejects(&blob, "malformed 32-bit roaring bitmap");
    }

    /// Risk pinned: the ACCEPT side of the key boundary. Java `readKey` accepts keys up to
    /// `i32::MAX - 1`. A tighter bound here rejects valid Java-written DVs with high keys, and
    /// shrinks the accepted-input set below Java's.
    #[test]
    fn test_dv_blob_max_valid_key_boundary_accepted() {
        let mut one = RoaringBitmap::new();
        one.insert(1);
        let key = i32::MAX as u32 - 1;
        let blob = encode_deletion_vector_v1_from_pairs(&[(key, one)]);
        let decoded = DeleteVector::deserialize_deletion_vector_v1(&blob)
            .expect("key i32::MAX-1 is the largest key Java accepts");
        let positions: Vec<u64> = decoded.iter().collect();
        assert_eq!(positions, vec![(u64::from(key) << 32) | 1]);
    }

    /// Risk pinned: trailing bytes inside the declared bitmap region mean the length prefix and
    /// the bitmap disagree. Silent acceptance masks corruption.
    #[test]
    fn test_dv_blob_trailing_bytes_after_bitmaps_reject() {
        let valid = encode_deletion_vector_v1(&[1, 2, 3]);
        let mut bitmap_bytes =
            valid[DV_LENGTH_PREFIX_SIZE + DV_MAGIC_SIZE..valid.len() - DV_CRC_SIZE].to_vec();
        bitmap_bytes.extend_from_slice(&[0u8; 3]);
        let blob = frame_deletion_vector_v1(&bitmap_bytes);
        assert_rejects(&blob, "trailing bytes");
    }

    /// Empirical Java byte-compatibility pin, env-gated behind
    /// `dev/java-interop/run-interop-dv.sh`. The Java oracle writes `dv_blob.bin` and
    /// `dv_blob_expected.json`. This test decodes the real Java bytes and asserts the exact
    /// position set. It settles whether `roaring-rs`'s portable format matches Java's.
    #[test]
    fn test_dv_blob_decodes_java_written_blob_when_env_set() {
        let Some(dir) = std::env::var_os("ICEBERG_INTEROP_DV_DIR")
            .filter(|value| !value.is_empty())
            .map(std::path::PathBuf::from)
        else {
            println!(
                "skipping java DV blob decode pin — set ICEBERG_INTEROP_DV_DIR \
                 (run dev/java-interop/run-interop-dv.sh)"
            );
            return;
        };

        let blob = std::fs::read(dir.join("dv_blob.bin")).expect("read dv_blob.bin");
        let expected_json =
            std::fs::read_to_string(dir.join("dv_blob_expected.json")).expect("read expected json");
        let expected: Vec<u64> =
            serde_json::from_str(&expected_json).expect("parse expected positions");

        let decoded = DeleteVector::deserialize_deletion_vector_v1(&blob)
            .expect("Java-written deletion-vector-v1 blob must decode");

        let decoded_positions: Vec<u64> = decoded.iter().collect();
        assert_eq!(
            decoded_positions, expected,
            "Rust-decoded positions must equal the positions Java serialized"
        );
    }

    #[test]
    fn test_insertion_and_iteration() {
        let mut dv = DeleteVector::default();
        assert!(dv.insert(42));
        assert!(dv.insert(100));
        assert!(!dv.insert(42));

        let mut items: Vec<u64> = dv.iter().collect();
        items.sort();
        assert_eq!(items, vec![42, 100]);
        assert_eq!(dv.len(), 2);
    }

    #[test]
    fn test_successful_insert_positions() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 2, 3, 1000, 1 << 33];
        assert_eq!(dv.insert_positions(&positions).unwrap(), 5);

        let mut collected: Vec<u64> = dv.iter().collect();
        collected.sort();
        assert_eq!(collected, positions);
    }

    /// Bulk insertion fails: the input positions do not strictly increase.
    #[test]
    fn test_failed_insertion_unsorted_elements() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 3, 5, 4];
        let res = dv.insert_positions(&positions);
        assert!(res.is_err());
    }

    /// Bulk insertion fails: the input positions intersect the existing ones.
    #[test]
    fn test_failed_insertion_with_intersection() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 3, 5];
        assert_eq!(dv.insert_positions(&positions).unwrap(), 3);

        let res = dv.insert_positions(&[2, 4]);
        assert!(res.is_err());
    }

    /// Bulk insertion fails: the input positions carry duplicates.
    #[test]
    fn test_failed_insertion_duplicate_elements() {
        let mut dv = DeleteVector::default();
        let positions = vec![1, 3, 5, 5];
        let res = dv.insert_positions(&positions);
        assert!(res.is_err());
    }

    const KEY_BOUNDARY: u64 = 1 << 32;

    /// Builds a [`DeleteVector`] from explicit positions (deterministic, no RNG).
    fn dv_with(positions: &[u64]) -> DeleteVector {
        let mut dv = DeleteVector::new(RoaringTreemap::new());
        for &p in positions {
            dv.insert(p);
        }
        dv
    }

    /// Risk pinned: the `advance_to` overshoot contract across a GAP GROUP. Group 1 is absent, so
    /// the outer walk overshoots into group 2. The iterator must stop at group 2's start. An
    /// unconditional `bitmap_iter.advance_to(lo)` with `lo = 0xFFFFFFFE` would consume group 2's
    /// only element and yield `None`.
    #[test]
    fn test_advance_to_across_gap_group_yields_next_higher_delete() {
        let dv = dv_with(&[KEY_BOUNDARY - 2, 2 * KEY_BOUNDARY]);

        // Target sits in the absent group 1, with a low value larger than group 2's only element.
        let mut iter = dv.iter();
        assert_eq!(iter.next(), Some(KEY_BOUNDARY - 2), "prime yields group 0");
        iter.advance_to(2 * KEY_BOUNDARY - 2);
        assert_eq!(
            iter.next(),
            Some(2 * KEY_BOUNDARY),
            "across a gap group, advance_to must leave the higher group's first delete unconsumed"
        );
        assert_eq!(iter.next(), None, "no deletes beyond 2*KB");
    }

    /// Risk pinned: two consecutive gap groups. The outer walk skips both and still yields the
    /// delete in group 3. The target's low bits `0xFFFFFFFE` exceed group 3's element low bits
    /// `7`, so an unconditional `bitmap_iter.advance_to(lo)` would consume that element.
    #[test]
    fn test_advance_to_across_multiple_gap_groups() {
        let dv = dv_with(&[KEY_BOUNDARY - 1, 3 * KEY_BOUNDARY + 7]);

        let mut iter = dv.iter();
        assert_eq!(iter.next(), Some(KEY_BOUNDARY - 1));
        // Target in absent group 2 with a high low-bits value (KB-2 = 0xFFFFFFFE).
        iter.advance_to(2 * KEY_BOUNDARY + (KEY_BOUNDARY - 2));
        assert_eq!(
            iter.next(),
            Some(3 * KEY_BOUNDARY + 7),
            "advance_to must skip multiple absent groups without consuming the next delete"
        );
        assert_eq!(iter.next(), None);
    }

    /// The target's group is PRESENT. `advance_to` must still skip within the group to the first
    /// low value `>= lo`, which keeps the fast-skip.
    #[test]
    fn test_advance_to_within_present_group_skips_lower_positions() {
        // Single group 0 with several positions.
        let dv = dv_with(&[3, 7, 11, 19]);

        let mut iter = dv.iter();
        assert_eq!(iter.next(), Some(3), "prime");
        iter.advance_to(10); // present group, skip past 3 and 7
        assert_eq!(
            iter.next(),
            Some(11),
            "within a present group, advance_to skips to the first position >= target"
        );
        assert_eq!(iter.next(), Some(19));
        assert_eq!(iter.next(), None);

        // Present higher group: target lands exactly on an existing position.
        let dv2 = dv_with(&[5, KEY_BOUNDARY + 4, KEY_BOUNDARY + 8]);
        let mut iter2 = dv2.iter();
        assert_eq!(iter2.next(), Some(5));
        iter2.advance_to(KEY_BOUNDARY + 4);
        assert_eq!(
            iter2.next(),
            Some(KEY_BOUNDARY + 4),
            "advance_to lands exactly on an existing position in a present higher group"
        );
        assert_eq!(iter2.next(), Some(KEY_BOUNDARY + 8));
        assert_eq!(iter2.next(), None);
    }
}

#[cfg(test)]
mod loader_tests {
    use tempfile::TempDir;

    use super::*;
    use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

    /// Writes one DV over `s3://b/d.parquet` and returns the FileIO to read it back through.
    async fn written_dv(dir: &TempDir, positions: &[u64]) -> (FileIO, DataFile) {
        let file_io = FileIO::new_with_fs();
        let path = dir.path().join("deletes.puffin");
        let output = file_io
            .new_output(path.to_str().expect("utf-8 temp path"))
            .expect("create output file");
        let mut writer = DVFileWriter::new(output);
        for position in positions {
            writer
                .delete("s3://b/d.parquet", *position, None)
                .expect("delete");
        }
        let mut files = writer.close().await.expect("close");
        assert_eq!(files.len(), 1, "one DeleteFile per referenced data file");
        (file_io, files.remove(0))
    }

    #[tokio::test]
    async fn round_trips_a_written_dv() {
        let dir = TempDir::new().expect("temp dir");
        let (file_io, delete_file) = written_dv(&dir, &[0, 3, (1u64 << 32) + 1]).await;

        let loaded = load_delete_vector(&file_io, &delete_file)
            .await
            .expect("a DV this writer just wrote must load back");
        assert_eq!(loaded.iter().collect::<Vec<_>>(), vec![
            0,
            3,
            (1u64 << 32) + 1
        ]);
    }

    #[tokio::test]
    async fn rejects_a_file_that_is_not_a_dv() {
        let dir = TempDir::new().expect("temp dir");
        let (file_io, mut delete_file) = written_dv(&dir, &[1]).await;
        delete_file.file_format = DataFileFormat::Parquet;

        let error = load_delete_vector(&file_io, &delete_file)
            .await
            .expect_err("a Parquet position-delete file is not a DV");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains("expected a Puffin position-delete"),
            "got: {}",
            error.message()
        );
    }

    #[tokio::test]
    async fn rejects_a_missing_referenced_data_file() {
        let dir = TempDir::new().expect("temp dir");
        let (file_io, mut delete_file) = written_dv(&dir, &[1]).await;
        // Only a decoded manifest entry can carry this shape; `DataFileBuilder` refuses it.
        delete_file.referenced_data_file = None;

        let error = load_delete_vector(&file_io, &delete_file)
            .await
            .expect_err("a DV with no referenced data file cannot be keyed");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("missing referenced_data_file"),
            "got: {}",
            error.message()
        );
    }

    #[tokio::test]
    async fn rejects_a_negative_content_offset() {
        let dir = TempDir::new().expect("temp dir");
        let (file_io, mut delete_file) = written_dv(&dir, &[1]).await;
        delete_file.content_offset = Some(-1);

        let error = load_delete_vector(&file_io, &delete_file)
            .await
            .expect_err("a negative offset is not a readable coordinate");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .ends_with("content_offset must be a non-negative integer, got Some(-1)"),
            "the exact text is pinned: these literals were once hand-wrapped with no `\\` \
             continuation, so the message carried 22 literal spaces and no test noticed. got: {}",
            error.message()
        );
    }

    #[tokio::test]
    async fn rejects_a_size_over_two_gigabytes() {
        let dir = TempDir::new().expect("temp dir");
        let (file_io, mut delete_file) = written_dv(&dir, &[1]).await;
        delete_file.content_size_in_bytes = Some(i64::from(i32::MAX) + 1);

        let error = load_delete_vector(&file_io, &delete_file)
            .await
            .expect_err("Java refuses to read a DV larger than 2GB");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("content_size_in_bytes must be"),
            "got: {}",
            error.message()
        );
    }

    /// Risk pinned: the manifest and the blob disagree about how many rows are deleted. A silent
    /// accept resurrects or over-deletes rows.
    #[tokio::test]
    async fn rejects_a_cardinality_mismatch() {
        let dir = TempDir::new().expect("temp dir");
        let (file_io, mut delete_file) = written_dv(&dir, &[0, 3]).await;
        delete_file.record_count = 99;

        let error = load_delete_vector(&file_io, &delete_file)
            .await
            .expect_err("a record_count that disagrees with the blob must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .contains("Invalid deletion vector cardinality"),
            "got: {}",
            error.message()
        );
    }
}
