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

fn decode_rle_v1(mut bytes: &[u8], signed: bool) -> Vec<i64> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        let header = bytes[0] as i8;
        bytes = &bytes[1..];
        if header < 0 {
            let count = header.unsigned_abs() as usize;
            for _ in 0..count {
                let (value, rest) = read_varint(bytes, signed);
                out.push(value);
                bytes = rest;
            }
        } else {
            let count = header as usize + 3;
            let delta = bytes[0] as i8 as i64;
            bytes = &bytes[1..];
            let (base, rest) = read_varint(bytes, signed);
            bytes = rest;
            for step in 0..count {
                out.push(base + delta * step as i64);
            }
        }
    }
    out
}

fn read_varint(bytes: &[u8], signed: bool) -> (i64, &[u8]) {
    let mut raw: u64 = 0;
    let mut shift = 0;
    let mut index = 0;
    loop {
        let byte = bytes[index];
        index += 1;
        raw |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    let value = if signed {
        ((raw >> 1) as i64) ^ -((raw & 1) as i64)
    } else {
        raw as i64
    };
    (value, &bytes[index..])
}

fn decode_byte_rle(mut bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        let header = bytes[0] as i8;
        bytes = &bytes[1..];
        if header < 0 {
            let count = header.unsigned_abs() as usize;
            out.extend_from_slice(&bytes[..count]);
            bytes = &bytes[count..];
        } else {
            let count = header as usize + 3;
            out.extend(std::iter::repeat_n(bytes[0], count));
            bytes = &bytes[1..];
        }
    }
    out
}

#[test]
fn test_uvarint_matches_protobuf_base128() {
    let mut out = Vec::new();
    put_uvarint(&mut out, 0);
    assert_eq!(out, vec![0x00]);

    out.clear();
    put_uvarint(&mut out, 127);
    assert_eq!(out, vec![0x7F]);

    out.clear();
    put_uvarint(&mut out, 128);
    assert_eq!(out, vec![0x80, 0x01]);

    out.clear();
    put_uvarint(&mut out, u64::MAX);
    assert_eq!(out, vec![
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01
    ]);
}

#[test]
fn test_svarint_is_zigzag() {
    let mut out = Vec::new();
    put_svarint(&mut out, 0);
    assert_eq!(out, vec![0x00]);

    out.clear();
    put_svarint(&mut out, -1);
    assert_eq!(out, vec![0x01]);

    out.clear();
    put_svarint(&mut out, 1);
    assert_eq!(out, vec![0x02]);

    out.clear();
    put_svarint(&mut out, i64::MIN);
    let (decoded, rest) = read_varint(&out, true);
    assert_eq!(decoded, i64::MIN);
    assert!(rest.is_empty());
}

#[test]
fn test_unbounded_varint_round_trips_i128_decimal_range() {
    for value in [
        0i128,
        1,
        -1,
        1234,
        -9_999_999_999,
        i128::from(i64::MAX) * 1_000,
        -(i128::from(i64::MAX) * 1_000),
        i128::MIN,
        i128::MAX,
        99_999_999_999_999_999_999_999_999_999_999_999_999i128,
        -99_999_999_999_999_999_999_999_999_999_999_999_999i128,
    ] {
        let mut out = Vec::new();
        put_unbounded_varint_i128(&mut out, value);
        let mut raw: u128 = 0;
        let mut shift = 0;
        for byte in &out {
            raw |= u128::from(byte & 0x7F) << shift;
            shift += 7;
        }
        let decoded = ((raw >> 1) as i128) ^ -((raw & 1) as i128);
        assert_eq!(decoded, value, "unbounded varint round trip for {value}");
    }
}

#[test]
fn test_rle_v1_repeat_run_matches_the_orc_spec_example() {
    let values = vec![7i64; 100];
    assert_eq!(rle_v1_unsigned(&values), vec![0x61, 0x00, 0x07]);
}

#[test]
fn test_rle_v1_literal_run_matches_the_orc_spec_example() {
    let values = vec![2i64, 3, 6, 7, 11];
    assert_eq!(rle_v1_unsigned(&values), vec![
        0xFB, 0x02, 0x03, 0x06, 0x07, 0x0B
    ]);
}

#[test]
fn test_rle_v1_round_trips_signed_and_unsigned_shapes() {
    let cases: Vec<Vec<i64>> = vec![
        vec![],
        vec![0],
        vec![1, 2],
        vec![5; 3],
        vec![5; 130],
        vec![5; 131],
        (0..300).collect(),
        (0..300).rev().collect(),
        vec![i64::MIN, i64::MAX, 0, -1, 1],
        (0..200).map(|i| if i % 7 == 0 { i * 31 } else { -i }).collect(),
    ];
    for values in cases {
        let encoded = rle_v1_signed(&values);
        assert_eq!(decode_rle_v1(&encoded, true), values, "signed {values:?}");

        let unsigned: Vec<i64> = values.iter().map(|v| v.rem_euclid(1 << 20)).collect();
        let encoded = rle_v1_unsigned(&unsigned);
        assert_eq!(
            decode_rle_v1(&encoded, false),
            unsigned,
            "unsigned {unsigned:?}"
        );
    }
}

#[test]
fn test_rle_v1_never_emits_a_delta_run_outside_the_i8_range() {
    let values = vec![0i64, 1000, 2000, 3000, 4000];
    let encoded = rle_v1_signed(&values);
    assert_eq!(decode_rle_v1(&encoded, true), values);
    assert_eq!(encoded[0] as i8, -5, "a 1000-step delta must stay literal");
}

#[test]
fn test_byte_rle_matches_the_orc_spec_examples() {
    assert_eq!(byte_rle(&[0x01; 100]), vec![0x61, 0x01]);
    assert_eq!(byte_rle(&[0x44, 0x45]), vec![0xFE, 0x44, 0x45]);
}

#[test]
fn test_byte_rle_round_trips() {
    let cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![9],
        vec![1, 2, 3],
        vec![7; 130],
        vec![7; 131],
        (0..=255u8).collect(),
        {
            let mut v = vec![0u8; 50];
            v.extend([1, 2, 3, 4]);
            v.extend(vec![9u8; 200]);
            v
        },
    ];
    for values in cases {
        let encoded = byte_rle(&values);
        assert_eq!(decode_byte_rle(&encoded), values, "byte rle {values:?}");
    }
}

#[test]
fn test_boolean_rle_packs_bits_most_significant_first() {
    let bits = [false, true, true, true, true, true, true, true];
    assert_eq!(decode_byte_rle(&boolean_rle(&bits)), vec![0x7F]);

    let bits = [true, false, false, false, false, false, false, false];
    assert_eq!(decode_byte_rle(&boolean_rle(&bits)), vec![0x80]);
}

#[test]
fn test_boolean_rle_pads_the_final_byte_with_zero_bits() {
    let bits = [true, true, true];
    assert_eq!(decode_byte_rle(&boolean_rle(&bits)), vec![0xE0]);
}

#[test]
fn test_compress_chunks_none_is_the_identity() {
    let payload = b"the quick brown fox".to_vec();
    let out = compress_chunks(&payload, OrcCompression::None, 1024).expect("no-op compression");
    assert_eq!(out, payload);
}

#[test]
fn test_compress_chunks_zlib_round_trips_through_raw_inflate() {
    let payload: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
    let out = compress_chunks(&payload, OrcCompression::Zlib, 1024).expect("zlib chunks");
    assert_eq!(inflate_chunks(&out), payload);
    assert!(out.len() < payload.len(), "a repetitive payload must shrink");
}

#[test]
fn test_compress_chunks_marks_incompressible_blocks_original() {
    let payload: Vec<u8> = (0..600u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let out = compress_chunks(&payload, OrcCompression::Zlib, 64).expect("zlib chunks");
    assert_eq!(inflate_chunks(&out), payload);
    let header = (out[0] as usize) | ((out[1] as usize) << 8) | ((out[2] as usize) << 16);
    assert_eq!(header & 1, 1, "a random 64-byte block must stay original");
}

fn inflate_chunks(mut raw: &[u8]) -> Vec<u8> {
    use std::io::Read;

    let mut out = Vec::new();
    while !raw.is_empty() {
        let header = (raw[0] as usize) | ((raw[1] as usize) << 8) | ((raw[2] as usize) << 16);
        let is_original = header & 1 == 1;
        let len = header >> 1;
        raw = &raw[3..];
        let (chunk, rest) = raw.split_at(len);
        if is_original {
            out.extend_from_slice(chunk);
        } else {
            flate2::read::DeflateDecoder::new(chunk)
                .read_to_end(&mut out)
                .expect("inflate an ORC chunk");
        }
        raw = rest;
    }
    out
}

#[test]
fn test_compression_codec_names_follow_java_and_refuse_the_rest() {
    assert_eq!(
        OrcCompression::from_codec_name("zlib").expect("zlib"),
        OrcCompression::Zlib
    );
    assert_eq!(
        OrcCompression::from_codec_name("NONE").expect("none"),
        OrcCompression::None
    );
    let error = OrcCompression::from_codec_name("snappy").expect_err("snappy is not supported yet");
    assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
    assert!(
        error.message().contains("snappy"),
        "the refusal must name the codec: {error}"
    );
}
