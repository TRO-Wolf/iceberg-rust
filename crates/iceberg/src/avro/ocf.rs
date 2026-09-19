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

use std::borrow::Cow;
use std::io::{BufReader, Chain, Cursor, Read};

use serde_json::Value as JsonValue;

use crate::avro::name::{ICEBERG_FIELD_NAME_PROP, repair_target_name};
use crate::{Error, ErrorKind, Result};

const OCF_MAGIC: &[u8; 4] = b"Obj\x01";
const OCF_MAX_HEADER_LEN: usize = 64 * 1024 * 1024;
const AVRO_SCHEMA_META_KEY: &[u8] = b"avro.schema";

#[cfg(test)]
static OCF_JSON_PARSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn ocf_json_parse_count() -> usize {
    OCF_JSON_PARSES.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) struct OcfHeader {
    pub(crate) bytes: Vec<u8>,
    pub(crate) complete: bool,
}

pub(crate) type OcfStream<'a, R> = Chain<Cursor<Vec<u8>>, BufReader<&'a mut R>>;

fn read_avro_long(cursor: &mut &[u8]) -> Option<i64> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    loop {
        let (&byte, rest) = cursor.split_first()?;
        *cursor = rest;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(((value >> 1) as i64) ^ -((value & 1) as i64));
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

fn read_avro_bytes<'a>(cursor: &mut &'a [u8]) -> Option<&'a [u8]> {
    let len = usize::try_from(read_avro_long(cursor)?).ok()?;
    if cursor.len() < len {
        return None;
    }
    let (bytes, rest) = cursor.split_at(len);
    *cursor = rest;
    Some(bytes)
}

fn read_avro_metadata<'a>(cursor: &mut &'a [u8]) -> Option<Vec<(&'a [u8], &'a [u8])>> {
    let mut entries = Vec::new();
    loop {
        let mut count = read_avro_long(cursor)?;
        if count == 0 {
            return Some(entries);
        }
        if count < 0 {
            count = count.checked_neg()?;
            read_avro_long(cursor)?;
        }
        if count > i64::try_from(cursor.len()).unwrap_or(i64::MAX) {
            return None;
        }
        for _ in 0..count {
            let key = read_avro_bytes(cursor)?;
            let value = read_avro_bytes(cursor)?;
            entries.push((key, value));
        }
    }
}

fn write_avro_long(out: &mut Vec<u8>, value: i64) {
    let mut n = ((value << 1) ^ (value >> 63)) as u64;
    loop {
        if n & !0x7f == 0 {
            out.push(n as u8);
            return;
        }
        out.push((n as u8 & 0x7f) | 0x80);
        n >>= 7;
    }
}

fn fill<R: Read>(reader: &mut R, out: &mut Vec<u8>, n: usize) -> std::io::Result<usize> {
    let mut got = 0;
    let mut buf = [0u8; 512];
    while got < n {
        let want = (n - got).min(buf.len());
        match reader.read(&mut buf[..want]) {
            Ok(0) => break,
            Ok(m) => {
                out.extend_from_slice(&buf[..m]);
                got += m;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

fn read_stream_long<R: Read>(reader: &mut R, out: &mut Vec<u8>) -> std::io::Result<Option<i64>> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(None),
            Ok(_) => {
                out.push(byte[0]);
                value |= u64::from(byte[0] & 0x7f) << shift;
                if byte[0] & 0x80 == 0 {
                    return Ok(Some(((value >> 1) as i64) ^ -((value & 1) as i64)));
                }
                shift += 7;
                if shift > 63 {
                    return Ok(None);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}

fn read_stream_bytes<R: Read>(reader: &mut R, out: &mut Vec<u8>) -> std::io::Result<Option<()>> {
    let Some(len) = read_stream_long(reader, out)? else {
        return Ok(None);
    };
    let Ok(len) = usize::try_from(len) else {
        return Ok(None);
    };
    if len > OCF_MAX_HEADER_LEN {
        return Ok(None);
    }
    Ok((fill(reader, out, len)? == len).then_some(()))
}

fn header_io_err(e: std::io::Error) -> Error {
    Error::new(ErrorKind::Unexpected, "Failed to read Avro data file bytes").with_source(e)
}

pub(crate) fn read_ocf_header<R: Read>(reader: &mut R) -> Result<OcfHeader> {
    fn incomplete(bytes: Vec<u8>) -> Result<OcfHeader> {
        Ok(OcfHeader {
            bytes,
            complete: false,
        })
    }
    let mut out = Vec::new();
    if fill(reader, &mut out, 4).map_err(header_io_err)? < 4 || out[..] != OCF_MAGIC[..] {
        return incomplete(out);
    }
    loop {
        let Some(count) = read_stream_long(reader, &mut out).map_err(header_io_err)? else {
            return incomplete(out);
        };
        if count == 0 {
            break;
        }
        let n = if count < 0 {
            let Some(_byte_size) = read_stream_long(reader, &mut out).map_err(header_io_err)?
            else {
                return incomplete(out);
            };
            count.checked_neg().unwrap_or(i64::MAX)
        } else {
            count
        };
        for _ in 0..n {
            if read_stream_bytes(reader, &mut out)
                .map_err(header_io_err)?
                .is_none()
            {
                return incomplete(out);
            }
            if read_stream_bytes(reader, &mut out)
                .map_err(header_io_err)?
                .is_none()
            {
                return incomplete(out);
            }
        }
        if out.len() > OCF_MAX_HEADER_LEN {
            return incomplete(out);
        }
    }
    if fill(reader, &mut out, 16).map_err(header_io_err)? < 16 {
        return incomplete(out);
    }
    Ok(OcfHeader {
        bytes: out,
        complete: true,
    })
}

fn ocf_header_end(bs: &[u8]) -> Option<usize> {
    if !bs.starts_with(OCF_MAGIC) {
        return None;
    }
    let mut cursor = &bs[4..];
    read_avro_metadata(&mut cursor)?;
    let end = bs.len() - cursor.len() + 16;
    (end <= bs.len()).then_some(end)
}

struct JsonScan<'a> {
    bs: &'a [u8],
    pos: usize,
}

impl<'a> JsonScan<'a> {
    fn ws(&mut self) {
        while self.pos < self.bs.len() && matches!(self.bs[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn next(&mut self) -> Option<u8> {
        self.ws();
        self.bs.get(self.pos).copied()
    }

    fn string(&mut self) -> Option<Cow<'a, str>> {
        self.pos += 1;
        let mut start = self.pos;
        let mut owned = String::new();
        loop {
            match *self.bs.get(self.pos)? {
                b'"' => {
                    let tail = std::str::from_utf8(&self.bs[start..self.pos]).ok()?;
                    self.pos += 1;
                    return if owned.is_empty() {
                        Some(Cow::Borrowed(tail))
                    } else {
                        owned.push_str(tail);
                        Some(Cow::Owned(owned))
                    };
                }
                b'\\' => {
                    owned.push_str(std::str::from_utf8(&self.bs[start..self.pos]).ok()?);
                    self.pos += 1;
                    match *self.bs.get(self.pos)? {
                        b'u' => {
                            let hex = self.bs.get(self.pos + 1..self.pos + 5)?;
                            let cp =
                                u16::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok()?;
                            owned.push(char::from_u32(u32::from(cp))?);
                            self.pos += 5;
                        }
                        esc @ (b'"' | b'\\' | b'/') => {
                            owned.push(esc as char);
                            self.pos += 1;
                        }
                        b'b' => {
                            owned.push('\u{8}');
                            self.pos += 1;
                        }
                        b'f' => {
                            owned.push('\u{c}');
                            self.pos += 1;
                        }
                        b'n' => {
                            owned.push('\n');
                            self.pos += 1;
                        }
                        b'r' => {
                            owned.push('\r');
                            self.pos += 1;
                        }
                        b't' => {
                            owned.push('\t');
                            self.pos += 1;
                        }
                        _ => return None,
                    }
                    start = self.pos;
                }
                _ => self.pos += 1,
            }
        }
    }
}

fn schema_names_need_repair(schema: &[u8]) -> bool {
    #[derive(Default)]
    struct Frame {
        name: Option<String>,
        prop: Option<String>,
    }
    let mut scan = JsonScan { bs: schema, pos: 0 };
    let mut stack: Vec<Frame> = Vec::new();
    loop {
        let Some(byte) = scan.next() else {
            return !stack.is_empty();
        };
        match byte {
            b'{' => {
                scan.pos += 1;
                stack.push(Frame::default());
            }
            b'}' => {
                scan.pos += 1;
                let Some(frame) = stack.pop() else {
                    return true;
                };
                if let Some(name) = frame.name {
                    let original = frame.prop.as_deref().unwrap_or(name.as_str());
                    if repair_target_name(original) != name {
                        return true;
                    }
                }
            }
            b'"' => {
                let Some(key) = scan.string() else {
                    return true;
                };
                if scan.next() != Some(b':') {
                    continue;
                }
                scan.pos += 1;
                if scan.next() != Some(b'"') {
                    continue;
                }
                let Some(value) = scan.string() else {
                    return true;
                };
                if let Some(frame) = stack.last_mut() {
                    match key.as_ref() {
                        "name" => frame.name = Some(value.into_owned()),
                        ICEBERG_FIELD_NAME_PROP => frame.prop = Some(value.into_owned()),
                        _ => {}
                    }
                }
            }
            _ => scan.pos += 1,
        }
    }
}

fn patch_record_field(field: &mut serde_json::Map<String, JsonValue>) -> bool {
    let Some(name) = field
        .get("name")
        .and_then(JsonValue::as_str)
        .map(str::to_owned)
    else {
        return false;
    };
    let original = field
        .get(ICEBERG_FIELD_NAME_PROP)
        .and_then(JsonValue::as_str)
        .unwrap_or(name.as_str())
        .to_owned();
    let target = repair_target_name(&original);
    if name == target.as_ref() {
        return false;
    }
    field.insert("name".to_string(), JsonValue::String(target.into_owned()));
    field
        .entry(ICEBERG_FIELD_NAME_PROP.to_string())
        .or_insert(JsonValue::String(original));
    true
}

fn patch_schema_node(node: &mut JsonValue) -> bool {
    match node {
        JsonValue::Object(map) => {
            let mut changed = false;
            if let Some(JsonValue::Array(fields)) = map.get_mut("fields") {
                for field in fields.iter_mut() {
                    if let JsonValue::Object(field) = field {
                        changed |= patch_record_field(field);
                    }
                }
            }
            for value in map.values_mut() {
                changed |= patch_schema_node(value);
            }
            changed
        }
        JsonValue::Array(items) => items
            .iter_mut()
            .fold(false, |c, v| c | patch_schema_node(v)),
        _ => false,
    }
}

pub(crate) fn repair_avro_container(bs: &[u8]) -> Result<Cow<'_, [u8]>> {
    if !bs.starts_with(OCF_MAGIC) {
        return Ok(Cow::Borrowed(bs));
    }
    let mut cursor = &bs[4..];
    let Some(entries) = read_avro_metadata(&mut cursor) else {
        return Ok(Cow::Borrowed(bs));
    };
    let Some(schema_bytes) = entries
        .iter()
        .find(|(key, _)| *key == AVRO_SCHEMA_META_KEY)
        .map(|(_, value)| *value)
    else {
        return Ok(Cow::Borrowed(bs));
    };
    if !schema_names_need_repair(schema_bytes) {
        return Ok(Cow::Borrowed(bs));
    }
    #[cfg(test)]
    OCF_JSON_PARSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let Ok(mut json) = serde_json::from_slice::<JsonValue>(schema_bytes) else {
        return Ok(Cow::Borrowed(bs));
    };
    if !patch_schema_node(&mut json) {
        return Ok(Cow::Borrowed(bs));
    }
    let patched_schema = serde_json::to_vec(&json).map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            "Failed to encode patched Avro schema",
        )
        .with_source(e)
    })?;

    let mut out = Vec::with_capacity(bs.len() + patched_schema.len());
    out.extend_from_slice(OCF_MAGIC);
    write_avro_long(&mut out, i64::try_from(entries.len()).unwrap_or(i64::MAX));
    for (key, value) in entries {
        write_avro_long(&mut out, i64::try_from(key.len()).unwrap_or(i64::MAX));
        out.extend_from_slice(key);
        if key == AVRO_SCHEMA_META_KEY {
            write_avro_long(
                &mut out,
                i64::try_from(patched_schema.len()).unwrap_or(i64::MAX),
            );
            out.extend_from_slice(&patched_schema);
        } else {
            write_avro_long(&mut out, i64::try_from(value.len()).unwrap_or(i64::MAX));
            out.extend_from_slice(value);
        }
    }
    write_avro_long(&mut out, 0);
    out.extend_from_slice(cursor);
    Ok(Cow::Owned(out))
}

pub(crate) fn repair_ocf_header(header: &[u8]) -> Result<Option<Vec<u8>>> {
    Ok(match repair_avro_container(header)? {
        Cow::Borrowed(_) => None,
        Cow::Owned(h) => Some(h),
    })
}

pub(crate) fn repair_ocf_container_header(bs: &[u8]) -> Result<Option<(Vec<u8>, usize)>> {
    let Some(end) = ocf_header_end(bs) else {
        return Ok(None);
    };
    Ok(repair_ocf_header(&bs[..end])?.map(|h| (h, end)))
}

pub(crate) fn repaired_ocf_parts<'a>(bs: &'a [u8]) -> Result<(Cow<'a, [u8]>, &'a [u8])> {
    Ok(match repair_ocf_container_header(bs)? {
        Some((h, off)) => (Cow::Owned(h), &bs[off..]),
        None => (Cow::Borrowed(&[][..]), bs),
    })
}

pub(crate) fn ocf_repaired_stream<'a, R: Read>(reader: &'a mut R) -> Result<OcfStream<'a, R>> {
    let mut buffered = BufReader::new(reader);
    let header = read_ocf_header(&mut buffered)?;
    let bytes = if header.complete {
        repair_ocf_header(&header.bytes)?.unwrap_or(header.bytes)
    } else {
        header.bytes
    };
    Ok(Cursor::new(bytes).chain(buffered))
}
