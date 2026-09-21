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

use super::encode::{OrcCompression, put_uvarint};
use super::orc_type::OrcType;

pub(crate) const ORC_MAGIC: &[u8] = b"ORC";
pub(crate) const ORC_FILE_VERSION: [u32; 2] = [0, 12];
pub(crate) const ORC_WRITER_ID_UNKNOWN: u32 = u32::MAX;
pub(crate) const ORC_WRITER_VERSION_FUTURE: u32 = u32::MAX;
pub(crate) const ORC_WRITER_TIMEZONE: &str = "UTC";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamKind {
    Present = 0,
    Data = 1,
    Length = 2,
    Secondary = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EncodingKind {
    Direct = 0,
}

#[derive(Debug, Clone)]
pub(crate) struct StreamRecord {
    pub(crate) kind: StreamKind,
    pub(crate) column: u32,
    pub(crate) length: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StripeRecord {
    pub(crate) offset: u64,
    pub(crate) index_length: u64,
    pub(crate) data_length: u64,
    pub(crate) footer_length: u64,
    pub(crate) number_of_rows: u64,
}

fn tag(out: &mut Vec<u8>, field: u32, wire: u32) {
    put_uvarint(out, u64::from((field << 3) | wire));
}

fn put_varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
    tag(out, field, 0);
    put_uvarint(out, value);
}

fn put_bytes_field(out: &mut Vec<u8>, field: u32, value: &[u8]) {
    tag(out, field, 2);
    put_uvarint(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn put_message_field(out: &mut Vec<u8>, field: u32, body: &[u8]) {
    put_bytes_field(out, field, body);
}

fn put_packed_uint32_field(out: &mut Vec<u8>, field: u32, values: &[u32]) {
    if values.is_empty() {
        return;
    }
    let mut body = Vec::with_capacity(values.len() * 2);
    for value in values {
        put_uvarint(&mut body, u64::from(*value));
    }
    put_bytes_field(out, field, &body);
}

fn encode_string_pair(key: &str, value: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(key.len() + value.len() + 4);
    put_bytes_field(&mut out, 1, key.as_bytes());
    put_bytes_field(&mut out, 2, value.as_bytes());
    out
}

pub(crate) fn encode_type(orc_type: &OrcType) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(kind) = orc_type.kind {
        put_varint_field(&mut out, 1, kind as u64);
    }
    put_packed_uint32_field(&mut out, 2, &orc_type.subtypes);
    for name in &orc_type.field_names {
        put_bytes_field(&mut out, 3, name.as_bytes());
    }
    if let Some(maximum_length) = orc_type.maximum_length {
        put_varint_field(&mut out, 4, u64::from(maximum_length));
    }
    if let Some(precision) = orc_type.precision {
        put_varint_field(&mut out, 5, u64::from(precision));
    }
    if let Some(scale) = orc_type.scale {
        put_varint_field(&mut out, 6, u64::from(scale));
    }
    for (key, value) in &orc_type.attributes {
        put_message_field(&mut out, 7, &encode_string_pair(key, value));
    }
    out
}

pub(crate) fn encode_stripe_footer(
    streams: &[StreamRecord],
    encodings: &[EncodingKind],
    writer_timezone: Option<&str>,
) -> Vec<u8> {
    let mut out = Vec::new();
    for stream in streams {
        let mut body = Vec::with_capacity(8);
        put_varint_field(&mut body, 1, stream.kind as u64);
        put_varint_field(&mut body, 2, u64::from(stream.column));
        put_varint_field(&mut body, 3, stream.length);
        put_message_field(&mut out, 1, &body);
    }
    for encoding in encodings {
        let mut body = Vec::with_capacity(2);
        put_varint_field(&mut body, 1, *encoding as u64);
        put_message_field(&mut out, 2, &body);
    }
    if let Some(timezone) = writer_timezone {
        put_bytes_field(&mut out, 3, timezone.as_bytes());
    }
    out
}

pub(crate) fn encode_footer(
    header_length: u64,
    content_length: u64,
    stripes: &[StripeRecord],
    types: &[OrcType],
    number_of_rows: u64,
) -> Vec<u8> {
    let mut out = Vec::new();
    put_varint_field(&mut out, 1, header_length);
    put_varint_field(&mut out, 2, content_length);
    for stripe in stripes {
        let mut body = Vec::with_capacity(16);
        put_varint_field(&mut body, 1, stripe.offset);
        put_varint_field(&mut body, 2, stripe.index_length);
        put_varint_field(&mut body, 3, stripe.data_length);
        put_varint_field(&mut body, 4, stripe.footer_length);
        put_varint_field(&mut body, 5, stripe.number_of_rows);
        put_message_field(&mut out, 3, &body);
    }
    for orc_type in types {
        put_message_field(&mut out, 4, &encode_type(orc_type));
    }
    put_varint_field(&mut out, 6, number_of_rows);
    put_varint_field(&mut out, 8, 0);
    put_varint_field(&mut out, 9, u64::from(ORC_WRITER_ID_UNKNOWN));
    out
}

pub(crate) fn encode_postscript(
    footer_length: u64,
    metadata_length: u64,
    codec: OrcCompression,
    compression_block_size: u64,
) -> Vec<u8> {
    let mut out = Vec::new();
    put_varint_field(&mut out, 1, footer_length);
    put_varint_field(&mut out, 2, codec.postscript_kind());
    if codec != OrcCompression::None {
        put_varint_field(&mut out, 3, compression_block_size);
    }
    put_packed_uint32_field(&mut out, 4, &ORC_FILE_VERSION);
    put_varint_field(&mut out, 5, metadata_length);
    put_varint_field(&mut out, 6, u64::from(ORC_WRITER_VERSION_FUTURE));
    put_bytes_field(&mut out, 8000, ORC_MAGIC);
    out
}

#[cfg(test)]
mod tests {
    include!("footer_write_tests.rs");
}
