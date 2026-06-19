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

//! A minimal, dependency-free ORC **Footer** parser, just enough to recover each type's
//! `iceberg.id` (and `iceberg.required`) attribute and build a `field-id → ORC column index` map.
//!
//! `orc-rust` drops ORC type attributes, so we re-parse the footer ourselves. The wire format:
//!
//! ```text
//! … [Footer (codec-compressed)] [PostScript (uncompressed protobuf)] [postscriptLength: 1 byte]
//! ```
//!
//! The last byte is the PostScript length. The PostScript protobuf gives `footerLength` (field 1)
//! and `compression` (field 2, enum NONE=0 / ZLIB=1 / SNAPPY=2 / LZO=3 / LZ4=4 / ZSTD=5). The Footer
//! is the `footerLength` bytes immediately before the PostScript; when compressed it is framed in
//! ORC **compression chunks** (a 3-byte little-endian header `(chunkLength << 1) | isOriginal`).
//! ORC ZLIB is **raw DEFLATE** (no zlib wrapper). We support NONE + ZLIB footers here; other codecs
//! yield a clear `FeatureUnsupported` (the *data* decode handles every codec via `orc-rust`).
//!
//! The Footer's `types` (field 4) is a **pre-order flat list**: `types[0]` is the root struct, and
//! each `Type`'s `subtypes` (field 2, packed `uint32`) are the column indices of its children. Each
//! `Type` may carry `attributes` (field 7, repeated `StringPair{ key=1, value=2 }`); the one keyed
//! `iceberg.id` gives the column's Iceberg field id.
//!
//! **Retirement tracking (why we own this).** This hand-rolled parser exists ONLY because `orc-rust`
//! drops ORC type attributes (its proto module is private and `RootDataType::from_proto` discards
//! them), so it is the only way to resolve by field id today. If a future `orc-rust` exposes type
//! attributes upstream, this file can be retired in favour of the upstream decode — revisit on the
//! next `orc-rust` bump. v1 scope is deliberately narrow: NONE/ZLIB footer codecs and top-level
//! primitive structs (nested struct/list/map by-id evolution is not yet covered — see GAP_MATRIX
//! row "Read: ORC data files").

use std::collections::HashMap;
use std::io::Read;

use flate2::read::DeflateDecoder;

use crate::{Error, ErrorKind, Result};

/// The ORC physical category of a type, as decoded from the footer `Type.kind`. Only the subset the
/// reader maps is named; everything else is `Other` (e.g. nested compound types we defer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OrcCategory {
    Boolean,
    Byte,
    Short,
    Int,
    Long,
    Float,
    Double,
    String,
    Binary,
    Timestamp,
    List,
    Map,
    Struct,
    Union,
    Decimal,
    Date,
    Varchar,
    Char,
    TimestampInstant,
    Other(i32),
}

impl OrcCategory {
    /// Map a raw ORC `Type.kind` enum value (a protobuf varint) to a category. Takes `u64` (the
    /// varint's native type) so there is no truncating cast on the parse path; an out-of-range kind
    /// (corrupt file) becomes `Other`, which the reader rejects downstream. The clamped `i32` carried
    /// by `Other` is for diagnostics only.
    fn from_kind(kind: u64) -> Self {
        match kind {
            0 => OrcCategory::Boolean,
            1 => OrcCategory::Byte,
            2 => OrcCategory::Short,
            3 => OrcCategory::Int,
            4 => OrcCategory::Long,
            5 => OrcCategory::Float,
            6 => OrcCategory::Double,
            7 => OrcCategory::String,
            8 => OrcCategory::Binary,
            9 => OrcCategory::Timestamp,
            10 => OrcCategory::List,
            11 => OrcCategory::Map,
            12 => OrcCategory::Struct,
            13 => OrcCategory::Union,
            14 => OrcCategory::Decimal,
            15 => OrcCategory::Date,
            16 => OrcCategory::Varchar,
            17 => OrcCategory::Char,
            18 => OrcCategory::TimestampInstant,
            // Clamp to i32::MAX for the diagnostic tag rather than a truncating `as` cast.
            other => OrcCategory::Other(i32::try_from(other).unwrap_or(i32::MAX)),
        }
    }
}

/// One file column resolved by its Iceberg `field-id`: its ORC column index and physical category
/// (+ decimal precision/scale) — enough for the projection's read-compatibility check.
#[derive(Debug, Clone)]
pub(crate) struct OrcFileType {
    /// The ORC column index (position in the pre-order `types[]` list), used to project `orc-rust`.
    pub(crate) orc_column_index: usize,
    /// The ORC physical category.
    pub(crate) category: OrcCategory,
    /// Decimal precision (0 if not a decimal).
    pub(crate) precision: u32,
    /// Decimal scale (0 if not a decimal).
    pub(crate) scale: u32,
    /// The `iceberg.required` attribute, if present (informational; the projection drives off the
    /// *expected* schema's required flag, matching Java). Parsed and asserted by `footer_tests`,
    /// but never read by the production projection — hence the targeted `dead_code` allow.
    #[allow(dead_code)]
    pub(crate) required: Option<bool>,
}

/// One raw ORC `Type` from the footer (pre-order entry). `kind`/`precision`/`scale` are held as the
/// protobuf varint's native `u64` so the parse path carries no truncating `as` cast; precision/scale
/// are narrowed to `u32` via a checked conversion at decimal-build time.
#[derive(Debug, Default)]
struct RawType {
    kind: u64,
    precision: u64,
    scale: u64,
    /// `iceberg.id` attribute value, if present.
    iceberg_id: Option<i32>,
    /// `iceberg.required` attribute value, if present.
    iceberg_required: Option<bool>,
}

/// The ORC compression codec from the PostScript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compression {
    None,
    Zlib,
    Snappy,
    Lzo,
    Lz4,
    Zstd,
    Other(u64),
}

impl Compression {
    /// Map the PostScript `compression` enum varint to a codec. Takes `u64` (the varint's native
    /// type) so there is no truncating cast; an out-of-range value becomes `Other`, which the footer
    /// decompressor rejects with a clear `FeatureUnsupported`.
    fn from_enum(v: u64) -> Self {
        match v {
            0 => Compression::None,
            1 => Compression::Zlib,
            2 => Compression::Snappy,
            3 => Compression::Lzo,
            4 => Compression::Lz4,
            5 => Compression::Zstd,
            other => Compression::Other(other),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Compression::None => "NONE",
            Compression::Zlib => "ZLIB",
            Compression::Snappy => "SNAPPY",
            Compression::Lzo => "LZO",
            Compression::Lz4 => "LZ4",
            Compression::Zstd => "ZSTD",
            Compression::Other(_) => "UNKNOWN",
        }
    }
}

// =================================================================================================
// Public entry: parse the footer into a field-id → OrcFileType map
// =================================================================================================

/// Parse the ORC footer of `bytes`, returning a `field-id → OrcFileType` map keyed by `iceberg.id`.
///
/// Errors loudly if the file carries no `iceberg.id` attributes (name-mapping fallback is not
/// supported), if the footer codec is unsupported (only NONE + ZLIB), or if the bytes are truncated.
pub(crate) fn parse_footer(bytes: &[u8]) -> Result<HashMap<i32, OrcFileType>> {
    // (1) PostScript: the last byte is its length; the PS protobuf precedes it.
    let ps_len = *bytes.last().ok_or_else(|| truncated("empty ORC file"))? as usize;
    if ps_len == 0 || ps_len + 1 > bytes.len() {
        return Err(truncated("ORC PostScript length is out of range"));
    }
    let ps_end = bytes.len() - 1;
    let ps_start = ps_end - ps_len;
    let postscript = &bytes[ps_start..ps_end];

    let (footer_length, compression) = parse_postscript(postscript)?;

    // (2) The Footer is `footer_length` bytes immediately before the PostScript.
    if footer_length > ps_start {
        return Err(truncated("ORC footer length exceeds the available bytes"));
    }
    let footer_start = ps_start - footer_length;
    let raw_footer = &bytes[footer_start..ps_start];

    // (3) Decompress the footer (NONE = raw; ZLIB = ORC chunk framing + raw DEFLATE).
    let footer = decompress_footer(raw_footer, compression)?;

    // (4) Parse `types` (field 4) into the pre-order list of RawType.
    let types = parse_footer_types(&footer)?;

    // (5) Build the field-id → OrcFileType map from the `iceberg.id` attributes.
    build_id_map(&types)
}

// =================================================================================================
// PostScript
// =================================================================================================

/// Parse the PostScript protobuf, returning `(footer_length, compression)`.
fn parse_postscript(ps: &[u8]) -> Result<(usize, Compression)> {
    let mut reader = ProtoReader::new(ps);
    let mut footer_length: Option<u64> = None;
    let mut compression = Compression::None;

    while let Some((field, wire)) = reader.read_tag()? {
        match (field, wire) {
            // field 1: footerLength (varint uint64)
            (1, WireType::Varint) => footer_length = Some(reader.read_varint()?),
            // field 2: compression (varint enum)
            (2, WireType::Varint) => {
                compression = Compression::from_enum(reader.read_varint()?);
            }
            // field 3: compressionBlockSize (varint uint64) — unused here.
            _ => reader.skip_field(wire)?,
        }
    }

    let footer_length = footer_length.ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            "ORC PostScript is missing footerLength",
        )
    })?;
    let footer_length =
        usize::try_from(footer_length).map_err(|_| truncated("ORC footerLength exceeds usize"))?;
    Ok((footer_length, compression))
}

// =================================================================================================
// Footer decompression (NONE + ZLIB only)
// =================================================================================================

/// Decompress the raw footer bytes per the PostScript codec. NONE returns the bytes as-is; ZLIB
/// walks the ORC compression-chunk framing and raw-DEFLATE-decodes each non-original chunk. Any
/// other codec is a clear `FeatureUnsupported`.
fn decompress_footer(raw: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(raw.to_vec()),
        Compression::Zlib => inflate_orc_chunks(raw),
        other => Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "ORC footer compression {} is not supported by the field-id footer parser yet \
                 (only NONE and ZLIB); the data decode itself supports every codec",
                other.label()
            ),
        )),
    }
}

/// Walk ORC compression chunks and concatenate their decompressed contents. Each chunk has a 3-byte
/// little-endian header: `(chunkLength << 1) | isOriginal`. If `isOriginal`, the next `chunkLength`
/// bytes are raw; otherwise they are raw-DEFLATE-compressed (ORC ZLIB has no zlib wrapper).
fn inflate_orc_chunks(mut raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len() * 2);
    while !raw.is_empty() {
        if raw.len() < 3 {
            return Err(truncated("ORC compression chunk header is truncated"));
        }
        let header = (raw[0] as usize) | ((raw[1] as usize) << 8) | ((raw[2] as usize) << 16);
        let is_original = (header & 1) == 1;
        let chunk_len = header >> 1;
        raw = &raw[3..];
        if chunk_len > raw.len() {
            return Err(truncated(
                "ORC compression chunk length exceeds the footer bytes",
            ));
        }
        let (chunk, rest) = raw.split_at(chunk_len);
        if is_original {
            out.extend_from_slice(chunk);
        } else {
            let mut decoder = DeflateDecoder::new(chunk);
            decoder.read_to_end(&mut out).map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Failed to inflate an ORC ZLIB (raw DEFLATE) footer chunk",
                )
                .with_source(e)
            })?;
        }
        raw = rest;
    }
    Ok(out)
}

// =================================================================================================
// Footer.types parsing
// =================================================================================================

/// Parse the Footer protobuf, extracting just `types` (field 4) into a pre-order `RawType` list.
fn parse_footer_types(footer: &[u8]) -> Result<Vec<RawType>> {
    let mut reader = ProtoReader::new(footer);
    let mut types = Vec::new();
    while let Some((field, wire)) = reader.read_tag()? {
        match (field, wire) {
            // field 4: types (repeated message Type)
            (4, WireType::Len) => {
                let msg = reader.read_len_delimited()?;
                types.push(parse_type(msg)?);
            }
            _ => reader.skip_field(wire)?,
        }
    }
    if types.is_empty() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "ORC footer has no types",
        ));
    }
    Ok(types)
}

/// Parse one ORC `Type` message, capturing `kind`, decimal `precision`/`scale`, and the
/// `iceberg.id` / `iceberg.required` attributes. `subtypes`/`fieldNames` are skipped: the pre-order
/// position in `types[]` is itself the ORC column index, which is what `orc-rust` projects by.
fn parse_type(msg: &[u8]) -> Result<RawType> {
    let mut reader = ProtoReader::new(msg);
    let mut ty = RawType::default();
    while let Some((field, wire)) = reader.read_tag()? {
        match (field, wire) {
            // field 1: kind (varint enum)
            (1, WireType::Varint) => ty.kind = reader.read_varint()?,
            // field 5: precision (varint uint32)
            (5, WireType::Varint) => ty.precision = reader.read_varint()?,
            // field 6: scale (varint uint32)
            (6, WireType::Varint) => ty.scale = reader.read_varint()?,
            // field 7: attributes (repeated message StringPair{ key=1, value=2 })
            (7, WireType::Len) => {
                let pair = reader.read_len_delimited()?;
                if let Some((key, value)) = parse_string_pair(pair)? {
                    match key.as_str() {
                        "iceberg.id" => {
                            ty.iceberg_id = value.parse::<i32>().ok();
                        }
                        "iceberg.required" => {
                            ty.iceberg_required = Some(value.eq_ignore_ascii_case("true"));
                        }
                        _ => {}
                    }
                }
            }
            // field 2: subtypes (packed uint32) and field 3: fieldNames (repeated string) — skipped.
            _ => reader.skip_field(wire)?,
        }
    }
    Ok(ty)
}

/// Parse a `StringPair{ key=1 string, value=2 string }`, returning `(key, value)`.
fn parse_string_pair(msg: &[u8]) -> Result<Option<(String, String)>> {
    let mut reader = ProtoReader::new(msg);
    let mut key: Option<String> = None;
    let mut value: Option<String> = None;
    while let Some((field, wire)) = reader.read_tag()? {
        match (field, wire) {
            (1, WireType::Len) => key = Some(read_utf8(reader.read_len_delimited()?)?),
            (2, WireType::Len) => value = Some(read_utf8(reader.read_len_delimited()?)?),
            _ => reader.skip_field(wire)?,
        }
    }
    match (key, value) {
        (Some(k), Some(v)) => Ok(Some((k, v))),
        // A StringPair missing a side is ignored (no iceberg attribute can come from it).
        _ => Ok(None),
    }
}

/// Build the `field-id → OrcFileType` map from the pre-order types, keyed by `iceberg.id`.
fn build_id_map(types: &[RawType]) -> Result<HashMap<i32, OrcFileType>> {
    let mut map = HashMap::new();
    let mut saw_any_id = false;
    for (index, ty) in types.iter().enumerate() {
        if let Some(id) = ty.iceberg_id {
            saw_any_id = true;
            // Narrow the varint precision/scale to `u32` with a checked conversion. A decimal's
            // precision/scale never exceeds 38, so an out-of-range value means a corrupt footer;
            // reject it rather than truncating with `as`.
            let precision = u32::try_from(ty.precision)
                .map_err(|_| corrupt("ORC decimal precision exceeds u32"))?;
            let scale =
                u32::try_from(ty.scale).map_err(|_| corrupt("ORC decimal scale exceeds u32"))?;
            map.insert(id, OrcFileType {
                orc_column_index: index,
                category: OrcCategory::from_kind(ty.kind),
                precision,
                scale,
                required: ty.iceberg_required,
            });
        }
    }
    if !saw_any_id {
        return Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "ORC data file carries no iceberg.id type attributes; name-mapping fallback is not \
             supported yet",
        ));
    }
    Ok(map)
}

fn read_utf8(bytes: &[u8]) -> Result<String> {
    String::from_utf8(bytes.to_vec()).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            "ORC type attribute string is not valid UTF-8",
        )
        .with_source(e)
    })
}

fn truncated(msg: &str) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("Corrupt or truncated ORC footer: {msg}"),
    )
}

// =================================================================================================
// Minimal protobuf reader (varint + length-delimited; no prost/build.rs)
// =================================================================================================

/// Protobuf wire types (only the two we need + length-delimited; fixed64/32 handled in `skip`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireType {
    Varint,
    Fixed64,
    Len,
    Fixed32,
    Other(u8),
}

impl WireType {
    fn from_tag(t: u8) -> Self {
        match t {
            0 => WireType::Varint,
            1 => WireType::Fixed64,
            2 => WireType::Len,
            5 => WireType::Fixed32,
            other => WireType::Other(other),
        }
    }
}

/// A bounds-checked, panic-free reader over a protobuf message slice.
struct ProtoReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        ProtoReader { buf, pos: 0 }
    }

    /// Read the next field tag, returning `(field_number, wire_type)` or `None` at end of buffer.
    fn read_tag(&mut self) -> Result<Option<(u32, WireType)>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.read_varint()?;
        let field =
            u32::try_from(key >> 3).map_err(|_| corrupt("protobuf field number exceeds u32"))?;
        let wire = WireType::from_tag((key & 0x7) as u8);
        Ok(Some((field, wire)))
    }

    /// Read a base-128 varint (max 10 bytes / 64 bits), guarding against truncation and overflow.
    fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if shift >= 64 {
                return Err(corrupt("protobuf varint is too long (>10 bytes)"));
            }
            let byte = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| corrupt("protobuf varint is truncated"))?;
            self.pos += 1;
            result |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Ok(result)
    }

    /// Read a length-delimited field's bytes (a varint length followed by that many bytes).
    fn read_len_delimited(&mut self) -> Result<&'a [u8]> {
        let len = usize::try_from(self.read_varint()?)
            .map_err(|_| corrupt("protobuf length exceeds usize"))?;
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| corrupt("protobuf length overflow"))?;
        if end > self.buf.len() {
            return Err(corrupt("protobuf length-delimited field is truncated"));
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// Skip a field of the given wire type (used for fields we don't care about).
    fn skip_field(&mut self, wire: WireType) -> Result<()> {
        match wire {
            WireType::Varint => {
                self.read_varint()?;
            }
            WireType::Fixed64 => self.advance(8)?,
            WireType::Len => {
                let len = usize::try_from(self.read_varint()?)
                    .map_err(|_| corrupt("protobuf length exceeds usize"))?;
                self.advance(len)?;
            }
            WireType::Fixed32 => self.advance(4)?,
            WireType::Other(t) => {
                return Err(corrupt(&format!("unknown protobuf wire type {t}")));
            }
        }
        Ok(())
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| corrupt("protobuf advance overflow"))?;
        if end > self.buf.len() {
            return Err(corrupt("protobuf field is truncated"));
        }
        self.pos = end;
        Ok(())
    }
}

fn corrupt(msg: &str) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("Corrupt ORC protobuf: {msg}"),
    )
}

#[cfg(test)]
mod tests {
    include!("footer_tests.rs");
}
