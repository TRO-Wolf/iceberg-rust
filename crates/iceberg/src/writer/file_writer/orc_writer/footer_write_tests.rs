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

use super::*;

fn read_varint(bytes: &[u8], at: &mut usize) -> u64 {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = bytes[*at];
        *at += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return value;
        }
        shift += 7;
    }
}

fn fields(bytes: &[u8]) -> Vec<(u32, u32, Vec<u8>)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < bytes.len() {
        let key = read_varint(bytes, &mut at);
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        match wire {
            0 => {
                let value = read_varint(bytes, &mut at);
                out.push((field, wire, value.to_le_bytes().to_vec()));
            }
            2 => {
                let len = read_varint(bytes, &mut at) as usize;
                out.push((field, wire, bytes[at..at + len].to_vec()));
                at += len;
            }
            other => panic!("unexpected wire type {other}"),
        }
    }
    out
}

fn varint_of(entries: &[(u32, u32, Vec<u8>)], field: u32) -> Option<u64> {
    entries
        .iter()
        .find(|(f, w, _)| *f == field && *w == 0)
        .map(|(_, _, v)| u64::from_le_bytes(v.as_slice().try_into().expect("8 bytes")))
}

fn bytes_of(entries: &[(u32, u32, Vec<u8>)], field: u32) -> Vec<Vec<u8>> {
    entries
        .iter()
        .filter(|(f, w, _)| *f == field && *w == 2)
        .map(|(_, _, v)| v.clone())
        .collect()
}

#[test]
fn test_type_encoding_carries_kind_subtypes_names_and_attributes() {
    let orc_type = OrcType {
        kind: Some(super::super::orc_type::OrcKind::Struct),
        subtypes: vec![1, 2],
        field_names: vec!["id".to_string(), "data".to_string()],
        maximum_length: None,
        precision: None,
        scale: None,
        attributes: vec![
            ("iceberg.id".to_string(), "5".to_string()),
            ("iceberg.required".to_string(), "true".to_string()),
        ],
    };
    let entries = fields(&encode_type(&orc_type));

    assert_eq!(varint_of(&entries, 1), Some(12));

    let packed = bytes_of(&entries, 2);
    assert_eq!(packed, vec![vec![1u8, 2u8]], "subtypes must be packed");

    let names = bytes_of(&entries, 3);
    assert_eq!(names, vec![b"id".to_vec(), b"data".to_vec()]);

    let attributes = bytes_of(&entries, 7);
    assert_eq!(attributes.len(), 2);
    let first = fields(&attributes[0]);
    assert_eq!(bytes_of(&first, 1), vec![b"iceberg.id".to_vec()]);
    assert_eq!(bytes_of(&first, 2), vec![b"5".to_vec()]);
    let second = fields(&attributes[1]);
    assert_eq!(bytes_of(&second, 1), vec![b"iceberg.required".to_vec()]);
    assert_eq!(bytes_of(&second, 2), vec![b"true".to_vec()]);
}

#[test]
fn test_decimal_precision_and_scale_reach_the_wire() {
    let orc_type = OrcType {
        kind: Some(super::super::orc_type::OrcKind::Decimal),
        precision: Some(10),
        scale: Some(2),
        ..OrcType::default()
    };
    let entries = fields(&encode_type(&orc_type));
    assert_eq!(varint_of(&entries, 1), Some(14));
    assert_eq!(varint_of(&entries, 5), Some(10));
    assert_eq!(varint_of(&entries, 6), Some(2));
}

#[test]
fn test_stripe_footer_carries_streams_encodings_and_the_utc_writer_timezone() {
    let streams = vec![
        StreamRecord {
            kind: StreamKind::Present,
            column: 1,
            length: 5,
        },
        StreamRecord {
            kind: StreamKind::Data,
            column: 1,
            length: 9,
        },
    ];
    let entries = fields(&encode_stripe_footer(
        &streams,
        &[EncodingKind::Direct, EncodingKind::Direct],
        Some(ORC_WRITER_TIMEZONE),
    ));

    let encoded_streams = bytes_of(&entries, 1);
    assert_eq!(encoded_streams.len(), 2);
    let present = fields(&encoded_streams[0]);
    assert_eq!(varint_of(&present, 1), Some(0));
    assert_eq!(varint_of(&present, 2), Some(1));
    assert_eq!(varint_of(&present, 3), Some(5));
    let data = fields(&encoded_streams[1]);
    assert_eq!(varint_of(&data, 1), Some(1));
    assert_eq!(varint_of(&data, 3), Some(9));

    let encodings = bytes_of(&entries, 2);
    assert_eq!(encodings.len(), 2);
    assert_eq!(varint_of(&fields(&encodings[0]), 1), Some(0));

    assert_eq!(bytes_of(&entries, 3), vec![b"UTC".to_vec()]);
}

#[test]
fn test_footer_declares_the_header_length_the_stripes_and_no_row_index() {
    let stripes = vec![StripeRecord {
        offset: 3,
        index_length: 0,
        data_length: 100,
        footer_length: 20,
        number_of_rows: 7,
    }];
    let types = vec![OrcType {
        kind: Some(super::super::orc_type::OrcKind::Struct),
        ..OrcType::default()
    }];
    let entries = fields(&encode_footer(3, 123, &stripes, &types, 7));

    assert_eq!(varint_of(&entries, 1), Some(3));
    assert_eq!(varint_of(&entries, 2), Some(123));
    assert_eq!(varint_of(&entries, 6), Some(7));
    assert_eq!(
        varint_of(&entries, 8),
        Some(0),
        "no ROW_INDEX streams are written, so the stride must be 0"
    );

    let encoded = bytes_of(&entries, 3);
    assert_eq!(encoded.len(), 1);
    let stripe = fields(&encoded[0]);
    assert_eq!(varint_of(&stripe, 1), Some(3));
    assert_eq!(varint_of(&stripe, 2), Some(0));
    assert_eq!(varint_of(&stripe, 3), Some(100));
    assert_eq!(varint_of(&stripe, 4), Some(20));
    assert_eq!(varint_of(&stripe, 5), Some(7));

    assert_eq!(bytes_of(&entries, 4).len(), 1);
}

#[test]
fn test_postscript_declares_zlib_the_block_size_version_0_12_and_the_magic() {
    let entries = fields(&encode_postscript(500, 0, OrcCompression::Zlib, 262_144));
    assert_eq!(varint_of(&entries, 1), Some(500));
    assert_eq!(varint_of(&entries, 2), Some(1));
    assert_eq!(varint_of(&entries, 3), Some(262_144));
    assert_eq!(bytes_of(&entries, 4), vec![vec![0u8, 12u8]]);
    assert_eq!(varint_of(&entries, 5), Some(0));
    assert_eq!(bytes_of(&entries, 8000), vec![b"ORC".to_vec()]);
}

#[test]
fn test_postscript_omits_the_block_size_when_uncompressed() {
    let entries = fields(&encode_postscript(10, 0, OrcCompression::None, 262_144));
    assert_eq!(varint_of(&entries, 2), Some(0));
    assert_eq!(
        varint_of(&entries, 3),
        None,
        "an uncompressed file declares no compression block size"
    );
}

#[test]
fn test_the_postscript_fits_the_single_trailing_length_byte() {
    let encoded = encode_postscript(u64::MAX, u64::MAX, OrcCompression::Zlib, u64::MAX);
    assert!(
        encoded.len() < 256,
        "the PostScript length is one byte: {} bytes",
        encoded.len()
    );
}
