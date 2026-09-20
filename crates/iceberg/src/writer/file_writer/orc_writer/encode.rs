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

use std::io::Write;

use flate2::Compression as FlateLevel;
use flate2::write::DeflateEncoder;

use crate::{Error, ErrorKind, Result};

pub(crate) const MIN_REPEAT: usize = 3;
pub(crate) const MAX_LITERAL_RUN: usize = 128;
pub(crate) const MAX_REPEAT_RUN: usize = 127 + MIN_REPEAT;
pub(crate) const DEFAULT_COMPRESSION_BLOCK_SIZE: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OrcCompression {
    None,
    Zlib,
}

impl OrcCompression {
    pub(crate) fn from_codec_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "none" | "uncompressed" => Ok(OrcCompression::None),
            "zlib" => Ok(OrcCompression::Zlib),
            other => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "ORC compression codec '{other}' is not supported by the Rust ORC writer \
                     (supported: none, zlib)"
                ),
            )),
        }
    }

    pub(crate) fn postscript_kind(self) -> u64 {
        match self {
            OrcCompression::None => 0,
            OrcCompression::Zlib => 1,
        }
    }
}

pub(crate) fn put_uvarint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        if value < 0x80 {
            out.push(value as u8);
            return;
        }
        out.push(((value & 0x7F) | 0x80) as u8);
        value >>= 7;
    }
}

pub(crate) fn put_svarint(out: &mut Vec<u8>, value: i64) {
    put_uvarint(out, ((value << 1) ^ (value >> 63)) as u64);
}

pub(crate) fn put_unbounded_varint_i128(out: &mut Vec<u8>, value: i128) {
    let mut zigzag = ((value << 1) ^ (value >> 127)) as u128;
    loop {
        if zigzag < 0x80 {
            out.push(zigzag as u8);
            return;
        }
        out.push(((zigzag & 0x7F) | 0x80) as u8);
        zigzag >>= 7;
    }
}

pub(crate) fn byte_rle(values: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() + values.len() / 8 + 1);
    let mut index = 0usize;
    while index < values.len() {
        if repeat_len_at(values, index) >= MIN_REPEAT {
            let run = repeat_len_at(values, index).min(MAX_REPEAT_RUN);
            out.push((run - MIN_REPEAT) as u8);
            out.push(values[index]);
            index += run;
            continue;
        }
        let start = index;
        let mut end = index;
        while end < values.len() && end - start < MAX_LITERAL_RUN {
            if end > start && repeat_len_at(values, end) >= MIN_REPEAT {
                break;
            }
            end += 1;
        }
        out.push(literal_control(end - start));
        out.extend_from_slice(&values[start..end]);
        index = end;
    }
    out
}

fn repeat_len_at(values: &[u8], index: usize) -> usize {
    let first = values[index];
    let mut len = 1usize;
    while index + len < values.len() && values[index + len] == first && len < MAX_REPEAT_RUN {
        len += 1;
    }
    len
}

fn literal_control(len: usize) -> u8 {
    (-(len as i64)) as i8 as u8
}

pub(crate) fn boolean_rle(bits: &[bool]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(bits.len().div_ceil(8));
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (offset, bit) in chunk.iter().enumerate() {
            if *bit {
                byte |= 1 << (7 - offset);
            }
        }
        packed.push(byte);
    }
    byte_rle(&packed)
}

pub(crate) fn rle_v1_signed(values: &[i64]) -> Vec<u8> {
    rle_v1(values, true)
}

pub(crate) fn rle_v1_unsigned(values: &[i64]) -> Vec<u8> {
    rle_v1(values, false)
}

fn rle_v1(values: &[i64], signed: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() + 8);
    let mut index = 0usize;
    while index < values.len() {
        if let Some((delta, run)) = delta_run_at(values, index) {
            out.push((run - MIN_REPEAT) as u8);
            out.push(delta as u8);
            put_base(&mut out, values[index], signed);
            index += run;
            continue;
        }
        let start = index;
        let mut end = index;
        while end < values.len() && end - start < MAX_LITERAL_RUN {
            if end > start && delta_run_at(values, end).is_some() {
                break;
            }
            end += 1;
        }
        out.push(literal_control(end - start));
        for value in &values[start..end] {
            put_base(&mut out, *value, signed);
        }
        index = end;
    }
    out
}

fn put_base(out: &mut Vec<u8>, value: i64, signed: bool) {
    if signed {
        put_svarint(out, value);
    } else {
        put_uvarint(out, value as u64);
    }
}

fn delta_run_at(values: &[i64], index: usize) -> Option<(i8, usize)> {
    if index + MIN_REPEAT > values.len() {
        return None;
    }
    let delta = i128::from(values[index + 1]) - i128::from(values[index]);
    let delta = i8::try_from(delta).ok()?;
    let step = i128::from(delta);
    let mut run = 1usize;
    while index + run < values.len()
        && run < MAX_REPEAT_RUN
        && i128::from(values[index + run]) - i128::from(values[index + run - 1]) == step
    {
        run += 1;
    }
    if run >= MIN_REPEAT {
        Some((delta, run))
    } else {
        None
    }
}

pub(crate) fn compress_chunks(
    payload: &[u8],
    codec: OrcCompression,
    block_size: usize,
) -> Result<Vec<u8>> {
    if codec == OrcCompression::None {
        return Ok(payload.to_vec());
    }
    let block_size = block_size.max(1);
    let mut out = Vec::with_capacity(payload.len());
    for block in payload.chunks(block_size) {
        let deflated = deflate_raw(block)?;
        let (bytes, is_original) = if deflated.len() < block.len() {
            (deflated.as_slice(), 0u32)
        } else {
            (block, 1u32)
        };
        let header =
            (u32::try_from(bytes.len()).map_err(|_| chunk_too_large())? << 1) | is_original;
        out.push((header & 0xFF) as u8);
        out.push(((header >> 8) & 0xFF) as u8);
        out.push(((header >> 16) & 0xFF) as u8);
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

fn chunk_too_large() -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "An ORC compression chunk exceeded the 3-byte chunk-header range",
    )
}

fn deflate_raw(block: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = DeflateEncoder::new(Vec::new(), FlateLevel::default());
    encoder.write_all(block).map_err(deflate_failed)?;
    encoder.finish().map_err(deflate_failed)
}

fn deflate_failed(error: std::io::Error) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Failed to raw-DEFLATE an ORC compression chunk",
    )
    .with_source(error)
}

#[cfg(test)]
mod tests {
    include!("encode_tests.rs");
}
