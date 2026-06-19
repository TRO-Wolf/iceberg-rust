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

// Hand-built-protobuf fixtures for the ORC footer parser. These build the ORC file *suffix*
// (Footer + PostScript + length byte) entirely in-test so the parser is exercised without needing
// the binary golden file. The real Java-written interop oracle lives in `orc_reader_tests.rs`.

use std::io::Write;

use flate2::Compression as CompressionLevel;
use flate2::write::DeflateEncoder;

use super::*;

// -------------------------------------------------------------------------------------------------
// Minimal protobuf encoder (the inverse of the parser under test)
// -------------------------------------------------------------------------------------------------

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
}

fn tag(field: u32, wire: u8, out: &mut Vec<u8>) {
    varint(((field as u64) << 3) | wire as u64, out);
}

fn field_varint(field: u32, value: u64, out: &mut Vec<u8>) {
    tag(field, 0, out);
    varint(value, out);
}

fn field_len(field: u32, bytes: &[u8], out: &mut Vec<u8>) {
    tag(field, 2, out);
    varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

/// Encode a `StringPair{ key, value }`.
fn string_pair(key: &str, value: &str) -> Vec<u8> {
    let mut out = Vec::new();
    field_len(1, key.as_bytes(), &mut out);
    field_len(2, value.as_bytes(), &mut out);
    out
}

/// Encode an ORC `Type` with kind, optional decimal precision/scale, and attributes.
fn orc_type(
    kind: i32,
    precision: Option<u32>,
    scale: Option<u32>,
    attrs: &[(&str, &str)],
) -> Vec<u8> {
    let mut out = Vec::new();
    field_varint(1, kind as u64, &mut out); // kind
    if let Some(p) = precision {
        field_varint(5, p as u64, &mut out);
    }
    if let Some(s) = scale {
        field_varint(6, s as u64, &mut out);
    }
    for (k, v) in attrs {
        let pair = string_pair(k, v);
        field_len(7, &pair, &mut out); // attributes
    }
    out
}

/// Encode a `Type` with subtypes (field 2 packed uint32) + field names (field 3 repeated string),
/// to prove the parser correctly skips those fields.
fn struct_type(subtypes: &[u32], names: &[&str], attrs: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::new();
    field_varint(1, 12, &mut out); // kind = Struct
    // subtypes: packed uint32 (field 2, len-delimited)
    let mut packed = Vec::new();
    for &s in subtypes {
        varint(s as u64, &mut packed);
    }
    field_len(2, &packed, &mut out);
    for n in names {
        field_len(3, n.as_bytes(), &mut out); // field_names
    }
    for (k, v) in attrs {
        let pair = string_pair(k, v);
        field_len(7, &pair, &mut out);
    }
    out
}

/// Encode a Footer carrying just `types` (field 4, repeated Type).
fn footer(types: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    // header_length (field 1) and content_length (field 2) — plausible noise the parser skips.
    field_varint(1, 3, &mut out);
    field_varint(2, 100, &mut out);
    for t in types {
        field_len(4, t, &mut out);
    }
    field_varint(6, 7, &mut out); // number_of_rows — skipped
    out
}

/// Encode a PostScript with footer_length + compression.
fn postscript(footer_length: usize, compression: u64) -> Vec<u8> {
    let mut out = Vec::new();
    field_varint(1, footer_length as u64, &mut out); // footerLength
    field_varint(2, compression, &mut out); // compression
    field_varint(3, 262_144, &mut out); // compressionBlockSize — skipped
    out
}

/// Assemble a full ORC file *suffix* the parser reads: `[footer][postscript][ps_len: 1 byte]`.
fn assemble(footer_bytes: &[u8], compression: u64) -> Vec<u8> {
    let ps = postscript(footer_bytes.len(), compression);
    let mut file = Vec::new();
    // A little leading filler so footer_start > 0 and the slicing is non-trivial.
    file.extend_from_slice(b"ORC\x00\x00\x00FILLER");
    file.extend_from_slice(footer_bytes);
    file.extend_from_slice(&ps);
    file.push(ps.len() as u8);
    file
}

/// Wrap raw footer bytes into a single ORC compression chunk: header `(len<<1)|isOriginal`, then
/// either raw (isOriginal=1) or raw-DEFLATE-compressed (isOriginal=0).
fn orc_chunk(raw: &[u8], original: bool) -> Vec<u8> {
    let payload = if original {
        raw.to_vec()
    } else {
        let mut enc = DeflateEncoder::new(Vec::new(), CompressionLevel::default());
        enc.write_all(raw).expect("deflate footer");
        enc.finish().expect("finish deflate")
    };
    let header = (payload.len() << 1) | usize::from(original);
    let mut out = vec![
        (header & 0xFF) as u8,
        ((header >> 8) & 0xFF) as u8,
        ((header >> 16) & 0xFF) as u8,
    ];
    out.extend_from_slice(&payload);
    out
}

// -------------------------------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------------------------------

/// The canonical fixture: a root struct + a few primitives, NONE compression, with iceberg.id /
/// iceberg.required attributes — the parser must recover the full id → (orc_index, category) map.
#[test]
fn test_uncompressed_footer_id_map() {
    let types = vec![
        struct_type(&[1, 2, 3], &["id", "amount", "tag"], &[]),
        orc_type(4, None, None, &[("iceberg.id", "1"), ("iceberg.required", "true")]), // LONG
        orc_type(14, Some(10), Some(2), &[("iceberg.id", "2")]), // DECIMAL(10,2)
        orc_type(7, None, None, &[("iceberg.id", "3")]),         // STRING
    ];
    let file = assemble(&footer(&types), 0);

    let map = parse_footer(&file).expect("parse uncompressed footer");
    assert_eq!(map.len(), 3, "three id-bearing columns");

    let id1 = &map[&1];
    assert_eq!(id1.orc_column_index, 1);
    assert_eq!(id1.category, OrcCategory::Long);
    assert_eq!(id1.required, Some(true));

    let id2 = &map[&2];
    assert_eq!(id2.orc_column_index, 2);
    assert_eq!(id2.category, OrcCategory::Decimal);
    assert_eq!(id2.precision, 10);
    assert_eq!(id2.scale, 2);

    let id3 = &map[&3];
    assert_eq!(id3.orc_column_index, 3);
    assert_eq!(id3.category, OrcCategory::String);
    assert_eq!(id3.required, None, "iceberg.required absent → None");
}

/// The same logical footer but ZLIB-compressed (single raw-DEFLATE chunk) must parse identically —
/// proves the ORC chunk framing + raw-DEFLATE (NOT zlib-wrapped) path.
#[test]
fn test_zlib_footer_matches_uncompressed() {
    let types = vec![
        struct_type(&[1, 2], &["a", "b"], &[]),
        orc_type(3, None, None, &[("iceberg.id", "10")]), // INT
        orc_type(8, None, None, &[("iceberg.id", "20")]), // BINARY
    ];
    let footer_plain = footer(&types);
    let footer_zlib = orc_chunk(&footer_plain, false);
    let file = assemble(&footer_zlib, 1);

    let map = parse_footer(&file).expect("parse zlib footer");
    assert_eq!(map.len(), 2);
    assert_eq!(map[&10].category, OrcCategory::Int);
    assert_eq!(map[&10].orc_column_index, 1);
    assert_eq!(map[&20].category, OrcCategory::Binary);
    assert_eq!(map[&20].orc_column_index, 2);
}

/// A ZLIB footer wrapped in an "original" (uncompressed) chunk must also parse — exercises the
/// `isOriginal` branch of the chunk framing.
#[test]
fn test_zlib_original_chunk() {
    let types = vec![
        struct_type(&[1], &["x"], &[]),
        orc_type(6, None, None, &[("iceberg.id", "5")]), // DOUBLE
    ];
    let footer_plain = footer(&types);
    let footer_framed = orc_chunk(&footer_plain, true);
    let file = assemble(&footer_framed, 1);

    let map = parse_footer(&file).expect("parse original-chunk zlib footer");
    assert_eq!(map[&5].category, OrcCategory::Double);
}

/// A file carrying no iceberg.id attributes must error loudly (name-mapping fallback unsupported),
/// never silently resolve by name.
#[test]
fn test_no_iceberg_ids_errors() {
    let types = vec![
        struct_type(&[1], &["x"], &[]),
        orc_type(4, None, None, &[]), // LONG with no iceberg.id
    ];
    let file = assemble(&footer(&types), 0);

    let err = parse_footer(&file).expect_err("must reject id-less file");
    assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    assert!(
        err.message().contains("name-mapping"),
        "error names the name-mapping gap: {err}"
    );
}

/// An unsupported *footer* codec (e.g. SNAPPY=2) is a clear FeatureUnsupported.
#[test]
fn test_unsupported_footer_codec_errors() {
    let types = vec![
        struct_type(&[1], &["x"], &[]),
        orc_type(4, None, None, &[("iceberg.id", "1")]),
    ];
    // We assemble with compression=SNAPPY but leave the footer bytes raw; the parser must reject the
    // codec before attempting to decode (so the raw bytes never matter).
    let file = assemble(&footer(&types), 2);

    let err = parse_footer(&file).expect_err("must reject snappy footer");
    assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    assert!(
        err.message().contains("SNAPPY"),
        "error names the codec: {err}"
    );
}

/// Truncation guards: an empty file, a bad PostScript length, and a too-large footer length all
/// produce a DataInvalid error rather than a panic.
#[test]
fn test_truncation_guards() {
    assert_eq!(
        parse_footer(&[]).expect_err("empty").kind(),
        ErrorKind::DataInvalid
    );

    // PostScript length byte points past the buffer.
    let mut bad = vec![0u8; 4];
    *bad.last_mut().expect("non-empty") = 200;
    assert_eq!(
        parse_footer(&bad).expect_err("ps len oob").kind(),
        ErrorKind::DataInvalid
    );

    // Valid PS but footer_length larger than the bytes before it.
    let ps = postscript(9999, 0);
    let mut file = vec![1, 2, 3];
    file.extend_from_slice(&ps);
    file.push(ps.len() as u8);
    assert_eq!(
        parse_footer(&file).expect_err("footer len oob").kind(),
        ErrorKind::DataInvalid
    );
}

/// The minimal protobuf reader rejects a malformed varint (overlong) without panicking.
#[test]
fn test_protobuf_varint_overflow_guard() {
    // 11 continuation bytes (shift would exceed 64).
    let bad = [0x80u8; 11];
    let mut reader = ProtoReader::new(&bad);
    assert!(reader.read_varint().is_err(), "overlong varint must error");
}

/// A varint that runs off the end of the buffer is a clean error, not a panic.
#[test]
fn test_protobuf_truncated_varint() {
    let bad = [0x80u8, 0x80u8]; // continuation bits set but no terminator
    let mut reader = ProtoReader::new(&bad);
    assert!(reader.read_varint().is_err(), "truncated varint must error");
}
