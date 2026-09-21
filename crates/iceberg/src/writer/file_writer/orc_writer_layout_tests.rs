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

fn test_read_varint(bytes: &[u8], at: &mut usize) -> u64 {
    let mut value = 0u64;
    let mut shift = 0u32;
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

fn test_skip_field(bytes: &[u8], at: &mut usize, wire: u64) {
    match wire {
        0 => {
            test_read_varint(bytes, at);
        }
        2 => {
            let len = test_read_varint(bytes, at) as usize;
            *at += len;
        }
        other => panic!("unexpected wire type {other}"),
    }
}

fn test_inflate_chunks(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while !raw.is_empty() {
        let header =
            usize::from(raw[0]) | (usize::from(raw[1]) << 8) | (usize::from(raw[2]) << 16);
        let is_original = header & 1 == 1;
        let chunk_len = header >> 1;
        raw = &raw[3..];
        let (chunk, rest) = raw.split_at(chunk_len);
        if is_original {
            out.extend_from_slice(chunk);
        } else {
            let mut decoder = flate2::read::DeflateDecoder::new(chunk);
            decoder
                .read_to_end(&mut out)
                .expect("inflate a zlib footer chunk");
        }
        raw = rest;
    }
    out
}

struct TestFileLayout {
    footer_rows: u64,
    stripe_rows: Vec<u64>,
    stripe_offsets: Vec<u64>,
    header_length: u64,
    content_length: u64,
    row_index_stride: u64,
    footer_start: u64,
    postscript_length: u64,
    postscript_footer_length: u64,
    postscript_compression: u64,
    postscript_block_size: Option<u64>,
    postscript_version: Vec<u64>,
    postscript_magic: Vec<u8>,
    stripe_timezones: Vec<Vec<u8>>,
    type_attributes: Vec<std::collections::HashMap<String, String>>,
}

fn test_read_stripe_record(body: &[u8]) -> (u64, u64, u64, u64) {
    let mut inner = 0;
    let mut offset = 0;
    let mut data_length = 0;
    let mut footer_length = 0;
    let mut rows = 0;
    while inner < body.len() {
        let sub = test_read_varint(body, &mut inner);
        match (sub >> 3, sub & 7) {
            (1, 0) => offset = test_read_varint(body, &mut inner),
            (3, 0) => data_length = test_read_varint(body, &mut inner),
            (4, 0) => footer_length = test_read_varint(body, &mut inner),
            (5, 0) => rows = test_read_varint(body, &mut inner),
            (_, wire) => test_skip_field(body, &mut inner, wire),
        }
    }
    (offset, data_length, footer_length, rows)
}

fn test_read_timezone(stripe_footer: &[u8]) -> Vec<u8> {
    let mut at = 0;
    while at < stripe_footer.len() {
        let key = test_read_varint(stripe_footer, &mut at);
        match (key >> 3, key & 7) {
            (3, 2) => {
                let len = test_read_varint(stripe_footer, &mut at) as usize;
                return stripe_footer[at..at + len].to_vec();
            }
            (_, wire) => test_skip_field(stripe_footer, &mut at, wire),
        }
    }
    Vec::new()
}

fn test_read_string_pair(body: &[u8]) -> (String, String) {
    let mut at = 0;
    let mut key = String::new();
    let mut value = String::new();
    while at < body.len() {
        let sub = test_read_varint(body, &mut at);
        match (sub >> 3, sub & 7) {
            (1, 2) => {
                let len = test_read_varint(body, &mut at) as usize;
                key = String::from_utf8(body[at..at + len].to_vec())
                    .expect("type attribute keys are utf-8");
                at += len;
            }
            (2, 2) => {
                let len = test_read_varint(body, &mut at) as usize;
                value = String::from_utf8(body[at..at + len].to_vec())
                    .expect("type attribute values are utf-8");
                at += len;
            }
            (_, wire) => test_skip_field(body, &mut at, wire),
        }
    }
    (key, value)
}

fn test_read_type_attributes(body: &[u8]) -> std::collections::HashMap<String, String> {
    let mut at = 0;
    let mut attributes = std::collections::HashMap::new();
    while at < body.len() {
        let sub = test_read_varint(body, &mut at);
        match (sub >> 3, sub & 7) {
            (7, 2) => {
                let len = test_read_varint(body, &mut at) as usize;
                let (key, value) = test_read_string_pair(&body[at..at + len]);
                attributes.insert(key, value);
                at += len;
            }
            (_, wire) => test_skip_field(body, &mut at, wire),
        }
    }
    attributes
}

fn test_file_layout(file: &[u8]) -> TestFileLayout {
    let ps_len = usize::from(*file.last().expect("a non-empty ORC file"));
    let ps_end = file.len() - 1;
    let ps_start = ps_end - ps_len;
    let ps = &file[ps_start..ps_end];
    let mut at = 0;
    let mut footer_length = 0usize;
    let mut compression = 0u64;
    let mut block_size = None;
    let mut version = Vec::new();
    let mut magic = Vec::new();
    while at < ps.len() {
        let key = test_read_varint(ps, &mut at);
        match (key >> 3, key & 7) {
            (1, 0) => footer_length = test_read_varint(ps, &mut at) as usize,
            (2, 0) => compression = test_read_varint(ps, &mut at),
            (3, 0) => block_size = Some(test_read_varint(ps, &mut at)),
            (4, 2) => {
                let len = test_read_varint(ps, &mut at) as usize;
                let packed = &ps[at..at + len];
                at += len;
                let mut inner = 0;
                while inner < packed.len() {
                    version.push(test_read_varint(packed, &mut inner));
                }
            }
            (8000, 2) => {
                let len = test_read_varint(ps, &mut at) as usize;
                magic = ps[at..at + len].to_vec();
                at += len;
            }
            (_, wire) => test_skip_field(ps, &mut at, wire),
        }
    }
    let footer_start = ps_start - footer_length;
    let raw_footer = &file[footer_start..ps_start];
    let footer = match compression {
        0 => raw_footer.to_vec(),
        1 => test_inflate_chunks(raw_footer),
        other => panic!("unexpected test footer compression {other}"),
    };
    let mut at = 0;
    let mut number_of_rows = None;
    let mut header_length = 0;
    let mut content_length = 0;
    let mut row_index_stride = 0;
    let mut stripes = Vec::new();
    let mut type_attributes = Vec::new();
    while at < footer.len() {
        let key = test_read_varint(&footer, &mut at);
        match (key >> 3, key & 7) {
            (1, 0) => header_length = test_read_varint(&footer, &mut at),
            (2, 0) => content_length = test_read_varint(&footer, &mut at),
            (6, 0) => number_of_rows = Some(test_read_varint(&footer, &mut at)),
            (8, 0) => row_index_stride = test_read_varint(&footer, &mut at),
            (3, 2) => {
                let len = test_read_varint(&footer, &mut at) as usize;
                stripes.push(test_read_stripe_record(&footer[at..at + len]));
                at += len;
            }
            (4, 2) => {
                let len = test_read_varint(&footer, &mut at) as usize;
                type_attributes.push(test_read_type_attributes(&footer[at..at + len]));
                at += len;
            }
            (_, wire) => test_skip_field(&footer, &mut at, wire),
        }
    }
    let mut stripe_rows = Vec::with_capacity(stripes.len());
    let mut stripe_offsets = Vec::with_capacity(stripes.len());
    let mut stripe_timezones = Vec::with_capacity(stripes.len());
    for (offset, data_length, stripe_footer_length, rows) in &stripes {
        stripe_rows.push(*rows);
        stripe_offsets.push(*offset);
        let start = (*offset + *data_length) as usize;
        let end = start + *stripe_footer_length as usize;
        let raw_stripe_footer = &file[start..end];
        let stripe_footer = match compression {
            0 => raw_stripe_footer.to_vec(),
            1 => test_inflate_chunks(raw_stripe_footer),
            other => panic!("unexpected test stripe compression {other}"),
        };
        stripe_timezones.push(test_read_timezone(&stripe_footer));
    }
    TestFileLayout {
        footer_rows: number_of_rows.expect("Footer.numberOfRows must be present"),
        stripe_rows,
        stripe_offsets,
        header_length,
        content_length,
        row_index_stride,
        footer_start: u64::try_from(footer_start).expect("the test file fits in a u64"),
        postscript_length: u64::try_from(ps_len).expect("the test file fits in a u64"),
        postscript_footer_length: u64::try_from(footer_length)
            .expect("the test footer fits in a u64"),
        postscript_compression: compression,
        postscript_block_size: block_size,
        postscript_version: version,
        postscript_magic: magic,
        stripe_timezones,
        type_attributes,
    }
}
