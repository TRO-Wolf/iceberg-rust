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
use std::collections::BTreeMap;
use std::fmt::Write as _;

use apache_avro::Schema as AvroSchema;
use apache_avro::schema::{Name, RecordField as AvroRecordField, UnionSchema};
use apache_avro::types::Value as AvroValue;
use serde_json::Value as JsonValue;

use crate::{Error, ErrorKind, Result};

pub(crate) const ICEBERG_FIELD_NAME_PROP: &str = "iceberg-field-name";

#[cfg(test)]
static OCF_JSON_PARSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn ocf_json_parse_count() -> usize {
    OCF_JSON_PARSES.load(std::sync::atomic::Ordering::Relaxed)
}

static JAVA_LETTER_RANGES: &[(u16, u16)] = &[
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00AA, 0x00AA),
    (0x00B5, 0x00B5),
    (0x00BA, 0x00BA),
    (0x00C0, 0x00D6),
    (0x00D8, 0x00F6),
    (0x00F8, 0x02C1),
    (0x02C6, 0x02D1),
    (0x02E0, 0x02E4),
    (0x02EC, 0x02EC),
    (0x02EE, 0x02EE),
    (0x0370, 0x0374),
    (0x0376, 0x0377),
    (0x037A, 0x037D),
    (0x037F, 0x037F),
    (0x0386, 0x0386),
    (0x0388, 0x038A),
    (0x038C, 0x038C),
    (0x038E, 0x03A1),
    (0x03A3, 0x03F5),
    (0x03F7, 0x0481),
    (0x048A, 0x052F),
    (0x0531, 0x0556),
    (0x0559, 0x0559),
    (0x0560, 0x0588),
    (0x05D0, 0x05EA),
    (0x05EF, 0x05F2),
    (0x0620, 0x064A),
    (0x066E, 0x066F),
    (0x0671, 0x06D3),
    (0x06D5, 0x06D5),
    (0x06E5, 0x06E6),
    (0x06EE, 0x06EF),
    (0x06FA, 0x06FC),
    (0x06FF, 0x06FF),
    (0x0710, 0x0710),
    (0x0712, 0x072F),
    (0x074D, 0x07A5),
    (0x07B1, 0x07B1),
    (0x07CA, 0x07EA),
    (0x07F4, 0x07F5),
    (0x07FA, 0x07FA),
    (0x0800, 0x0815),
    (0x081A, 0x081A),
    (0x0824, 0x0824),
    (0x0828, 0x0828),
    (0x0840, 0x0858),
    (0x0860, 0x086A),
    (0x0870, 0x0887),
    (0x0889, 0x088E),
    (0x08A0, 0x08C9),
    (0x0904, 0x0939),
    (0x093D, 0x093D),
    (0x0950, 0x0950),
    (0x0958, 0x0961),
    (0x0971, 0x0980),
    (0x0985, 0x098C),
    (0x098F, 0x0990),
    (0x0993, 0x09A8),
    (0x09AA, 0x09B0),
    (0x09B2, 0x09B2),
    (0x09B6, 0x09B9),
    (0x09BD, 0x09BD),
    (0x09CE, 0x09CE),
    (0x09DC, 0x09DD),
    (0x09DF, 0x09E1),
    (0x09F0, 0x09F1),
    (0x09FC, 0x09FC),
    (0x0A05, 0x0A0A),
    (0x0A0F, 0x0A10),
    (0x0A13, 0x0A28),
    (0x0A2A, 0x0A30),
    (0x0A32, 0x0A33),
    (0x0A35, 0x0A36),
    (0x0A38, 0x0A39),
    (0x0A59, 0x0A5C),
    (0x0A5E, 0x0A5E),
    (0x0A72, 0x0A74),
    (0x0A85, 0x0A8D),
    (0x0A8F, 0x0A91),
    (0x0A93, 0x0AA8),
    (0x0AAA, 0x0AB0),
    (0x0AB2, 0x0AB3),
    (0x0AB5, 0x0AB9),
    (0x0ABD, 0x0ABD),
    (0x0AD0, 0x0AD0),
    (0x0AE0, 0x0AE1),
    (0x0AF9, 0x0AF9),
    (0x0B05, 0x0B0C),
    (0x0B0F, 0x0B10),
    (0x0B13, 0x0B28),
    (0x0B2A, 0x0B30),
    (0x0B32, 0x0B33),
    (0x0B35, 0x0B39),
    (0x0B3D, 0x0B3D),
    (0x0B5C, 0x0B5D),
    (0x0B5F, 0x0B61),
    (0x0B71, 0x0B71),
    (0x0B83, 0x0B83),
    (0x0B85, 0x0B8A),
    (0x0B8E, 0x0B90),
    (0x0B92, 0x0B95),
    (0x0B99, 0x0B9A),
    (0x0B9C, 0x0B9C),
    (0x0B9E, 0x0B9F),
    (0x0BA3, 0x0BA4),
    (0x0BA8, 0x0BAA),
    (0x0BAE, 0x0BB9),
    (0x0BD0, 0x0BD0),
    (0x0C05, 0x0C0C),
    (0x0C0E, 0x0C10),
    (0x0C12, 0x0C28),
    (0x0C2A, 0x0C39),
    (0x0C3D, 0x0C3D),
    (0x0C58, 0x0C5A),
    (0x0C5D, 0x0C5D),
    (0x0C60, 0x0C61),
    (0x0C80, 0x0C80),
    (0x0C85, 0x0C8C),
    (0x0C8E, 0x0C90),
    (0x0C92, 0x0CA8),
    (0x0CAA, 0x0CB3),
    (0x0CB5, 0x0CB9),
    (0x0CBD, 0x0CBD),
    (0x0CDD, 0x0CDE),
    (0x0CE0, 0x0CE1),
    (0x0CF1, 0x0CF2),
    (0x0D04, 0x0D0C),
    (0x0D0E, 0x0D10),
    (0x0D12, 0x0D3A),
    (0x0D3D, 0x0D3D),
    (0x0D4E, 0x0D4E),
    (0x0D54, 0x0D56),
    (0x0D5F, 0x0D61),
    (0x0D7A, 0x0D7F),
    (0x0D85, 0x0D96),
    (0x0D9A, 0x0DB1),
    (0x0DB3, 0x0DBB),
    (0x0DBD, 0x0DBD),
    (0x0DC0, 0x0DC6),
    (0x0E01, 0x0E30),
    (0x0E32, 0x0E33),
    (0x0E40, 0x0E46),
    (0x0E81, 0x0E82),
    (0x0E84, 0x0E84),
    (0x0E86, 0x0E8A),
    (0x0E8C, 0x0EA3),
    (0x0EA5, 0x0EA5),
    (0x0EA7, 0x0EB0),
    (0x0EB2, 0x0EB3),
    (0x0EBD, 0x0EBD),
    (0x0EC0, 0x0EC4),
    (0x0EC6, 0x0EC6),
    (0x0EDC, 0x0EDF),
    (0x0F00, 0x0F00),
    (0x0F40, 0x0F47),
    (0x0F49, 0x0F6C),
    (0x0F88, 0x0F8C),
    (0x1000, 0x102A),
    (0x103F, 0x103F),
    (0x1050, 0x1055),
    (0x105A, 0x105D),
    (0x1061, 0x1061),
    (0x1065, 0x1066),
    (0x106E, 0x1070),
    (0x1075, 0x1081),
    (0x108E, 0x108E),
    (0x10A0, 0x10C5),
    (0x10C7, 0x10C7),
    (0x10CD, 0x10CD),
    (0x10D0, 0x10FA),
    (0x10FC, 0x1248),
    (0x124A, 0x124D),
    (0x1250, 0x1256),
    (0x1258, 0x1258),
    (0x125A, 0x125D),
    (0x1260, 0x1288),
    (0x128A, 0x128D),
    (0x1290, 0x12B0),
    (0x12B2, 0x12B5),
    (0x12B8, 0x12BE),
    (0x12C0, 0x12C0),
    (0x12C2, 0x12C5),
    (0x12C8, 0x12D6),
    (0x12D8, 0x1310),
    (0x1312, 0x1315),
    (0x1318, 0x135A),
    (0x1380, 0x138F),
    (0x13A0, 0x13F5),
    (0x13F8, 0x13FD),
    (0x1401, 0x166C),
    (0x166F, 0x167F),
    (0x1681, 0x169A),
    (0x16A0, 0x16EA),
    (0x16F1, 0x16F8),
    (0x1700, 0x1711),
    (0x171F, 0x1731),
    (0x1740, 0x1751),
    (0x1760, 0x176C),
    (0x176E, 0x1770),
    (0x1780, 0x17B3),
    (0x17D7, 0x17D7),
    (0x17DC, 0x17DC),
    (0x1820, 0x1878),
    (0x1880, 0x1884),
    (0x1887, 0x18A8),
    (0x18AA, 0x18AA),
    (0x18B0, 0x18F5),
    (0x1900, 0x191E),
    (0x1950, 0x196D),
    (0x1970, 0x1974),
    (0x1980, 0x19AB),
    (0x19B0, 0x19C9),
    (0x1A00, 0x1A16),
    (0x1A20, 0x1A54),
    (0x1AA7, 0x1AA7),
    (0x1B05, 0x1B33),
    (0x1B45, 0x1B4C),
    (0x1B83, 0x1BA0),
    (0x1BAE, 0x1BAF),
    (0x1BBA, 0x1BE5),
    (0x1C00, 0x1C23),
    (0x1C4D, 0x1C4F),
    (0x1C5A, 0x1C7D),
    (0x1C80, 0x1C88),
    (0x1C90, 0x1CBA),
    (0x1CBD, 0x1CBF),
    (0x1CE9, 0x1CEC),
    (0x1CEE, 0x1CF3),
    (0x1CF5, 0x1CF6),
    (0x1CFA, 0x1CFA),
    (0x1D00, 0x1DBF),
    (0x1E00, 0x1F15),
    (0x1F18, 0x1F1D),
    (0x1F20, 0x1F45),
    (0x1F48, 0x1F4D),
    (0x1F50, 0x1F57),
    (0x1F59, 0x1F59),
    (0x1F5B, 0x1F5B),
    (0x1F5D, 0x1F5D),
    (0x1F5F, 0x1F7D),
    (0x1F80, 0x1FB4),
    (0x1FB6, 0x1FBC),
    (0x1FBE, 0x1FBE),
    (0x1FC2, 0x1FC4),
    (0x1FC6, 0x1FCC),
    (0x1FD0, 0x1FD3),
    (0x1FD6, 0x1FDB),
    (0x1FE0, 0x1FEC),
    (0x1FF2, 0x1FF4),
    (0x1FF6, 0x1FFC),
    (0x2071, 0x2071),
    (0x207F, 0x207F),
    (0x2090, 0x209C),
    (0x2102, 0x2102),
    (0x2107, 0x2107),
    (0x210A, 0x2113),
    (0x2115, 0x2115),
    (0x2119, 0x211D),
    (0x2124, 0x2124),
    (0x2126, 0x2126),
    (0x2128, 0x2128),
    (0x212A, 0x212D),
    (0x212F, 0x2139),
    (0x213C, 0x213F),
    (0x2145, 0x2149),
    (0x214E, 0x214E),
    (0x2183, 0x2184),
    (0x2C00, 0x2CE4),
    (0x2CEB, 0x2CEE),
    (0x2CF2, 0x2CF3),
    (0x2D00, 0x2D25),
    (0x2D27, 0x2D27),
    (0x2D2D, 0x2D2D),
    (0x2D30, 0x2D67),
    (0x2D6F, 0x2D6F),
    (0x2D80, 0x2D96),
    (0x2DA0, 0x2DA6),
    (0x2DA8, 0x2DAE),
    (0x2DB0, 0x2DB6),
    (0x2DB8, 0x2DBE),
    (0x2DC0, 0x2DC6),
    (0x2DC8, 0x2DCE),
    (0x2DD0, 0x2DD6),
    (0x2DD8, 0x2DDE),
    (0x2E2F, 0x2E2F),
    (0x3005, 0x3006),
    (0x3031, 0x3035),
    (0x303B, 0x303C),
    (0x3041, 0x3096),
    (0x309D, 0x309F),
    (0x30A1, 0x30FA),
    (0x30FC, 0x30FF),
    (0x3105, 0x312F),
    (0x3131, 0x318E),
    (0x31A0, 0x31BF),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0xA48C),
    (0xA4D0, 0xA4FD),
    (0xA500, 0xA60C),
    (0xA610, 0xA61F),
    (0xA62A, 0xA62B),
    (0xA640, 0xA66E),
    (0xA67F, 0xA69D),
    (0xA6A0, 0xA6E5),
    (0xA717, 0xA71F),
    (0xA722, 0xA788),
    (0xA78B, 0xA7CA),
    (0xA7D0, 0xA7D1),
    (0xA7D3, 0xA7D3),
    (0xA7D5, 0xA7D9),
    (0xA7F2, 0xA801),
    (0xA803, 0xA805),
    (0xA807, 0xA80A),
    (0xA80C, 0xA822),
    (0xA840, 0xA873),
    (0xA882, 0xA8B3),
    (0xA8F2, 0xA8F7),
    (0xA8FB, 0xA8FB),
    (0xA8FD, 0xA8FE),
    (0xA90A, 0xA925),
    (0xA930, 0xA946),
    (0xA960, 0xA97C),
    (0xA984, 0xA9B2),
    (0xA9CF, 0xA9CF),
    (0xA9E0, 0xA9E4),
    (0xA9E6, 0xA9EF),
    (0xA9FA, 0xA9FE),
    (0xAA00, 0xAA28),
    (0xAA40, 0xAA42),
    (0xAA44, 0xAA4B),
    (0xAA60, 0xAA76),
    (0xAA7A, 0xAA7A),
    (0xAA7E, 0xAAAF),
    (0xAAB1, 0xAAB1),
    (0xAAB5, 0xAAB6),
    (0xAAB9, 0xAABD),
    (0xAAC0, 0xAAC0),
    (0xAAC2, 0xAAC2),
    (0xAADB, 0xAADD),
    (0xAAE0, 0xAAEA),
    (0xAAF2, 0xAAF4),
    (0xAB01, 0xAB06),
    (0xAB09, 0xAB0E),
    (0xAB11, 0xAB16),
    (0xAB20, 0xAB26),
    (0xAB28, 0xAB2E),
    (0xAB30, 0xAB5A),
    (0xAB5C, 0xAB69),
    (0xAB70, 0xABE2),
    (0xAC00, 0xD7A3),
    (0xD7B0, 0xD7C6),
    (0xD7CB, 0xD7FB),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0xFB00, 0xFB06),
    (0xFB13, 0xFB17),
    (0xFB1D, 0xFB1D),
    (0xFB1F, 0xFB28),
    (0xFB2A, 0xFB36),
    (0xFB38, 0xFB3C),
    (0xFB3E, 0xFB3E),
    (0xFB40, 0xFB41),
    (0xFB43, 0xFB44),
    (0xFB46, 0xFBB1),
    (0xFBD3, 0xFD3D),
    (0xFD50, 0xFD8F),
    (0xFD92, 0xFDC7),
    (0xFDF0, 0xFDFB),
    (0xFE70, 0xFE74),
    (0xFE76, 0xFEFC),
    (0xFF21, 0xFF3A),
    (0xFF41, 0xFF5A),
    (0xFF66, 0xFFBE),
    (0xFFC2, 0xFFC7),
    (0xFFCA, 0xFFCF),
    (0xFFD2, 0xFFD7),
    (0xFFDA, 0xFFDC),
];

static JAVA_DIGIT_RANGES: &[(u16, u16)] = &[
    (0x0030, 0x0039),
    (0x0660, 0x0669),
    (0x06F0, 0x06F9),
    (0x07C0, 0x07C9),
    (0x0966, 0x096F),
    (0x09E6, 0x09EF),
    (0x0A66, 0x0A6F),
    (0x0AE6, 0x0AEF),
    (0x0B66, 0x0B6F),
    (0x0BE6, 0x0BEF),
    (0x0C66, 0x0C6F),
    (0x0CE6, 0x0CEF),
    (0x0D66, 0x0D6F),
    (0x0DE6, 0x0DEF),
    (0x0E50, 0x0E59),
    (0x0ED0, 0x0ED9),
    (0x0F20, 0x0F29),
    (0x1040, 0x1049),
    (0x1090, 0x1099),
    (0x17E0, 0x17E9),
    (0x1810, 0x1819),
    (0x1946, 0x194F),
    (0x19D0, 0x19D9),
    (0x1A80, 0x1A89),
    (0x1A90, 0x1A99),
    (0x1B50, 0x1B59),
    (0x1BB0, 0x1BB9),
    (0x1C40, 0x1C49),
    (0x1C50, 0x1C59),
    (0xA620, 0xA629),
    (0xA8D0, 0xA8D9),
    (0xA900, 0xA909),
    (0xA9D0, 0xA9D9),
    (0xA9F0, 0xA9F9),
    (0xAA50, 0xAA59),
    (0xABF0, 0xABF9),
    (0xFF10, 0xFF19),
];

fn in_ranges(ranges: &[(u16, u16)], unit: u16) -> bool {
    ranges
        .binary_search_by(|&(lo, hi)| {
            if unit < lo {
                std::cmp::Ordering::Greater
            } else if unit > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

fn is_java_letter(unit: u16) -> bool {
    in_ranges(JAVA_LETTER_RANGES, unit)
}

fn is_java_digit(unit: u16) -> bool {
    in_ranges(JAVA_DIGIT_RANGES, unit)
}

fn is_java_letter_or_digit(unit: u16) -> bool {
    is_java_letter(unit) || is_java_digit(unit)
}

fn is_ascii_letter(unit: u16) -> bool {
    matches!(unit, 0x41..=0x5A | 0x61..=0x7A)
}

fn is_ascii_digit(unit: u16) -> bool {
    matches!(unit, 0x30..=0x39)
}

fn is_ascii_letter_or_digit(unit: u16) -> bool {
    is_ascii_letter(unit) || is_ascii_digit(unit)
}

fn sanitize_name(
    name: &str,
    is_letter: fn(u16) -> bool,
    is_letter_or_digit: fn(u16) -> bool,
    is_digit: fn(u16) -> bool,
) -> Cow<'_, str> {
    let mut units = name.encode_utf16();
    let Some(first) = units.next() else {
        return Cow::Borrowed(name);
    };
    if (is_letter(first) || first == u16::from(b'_'))
        && units.all(|u| is_letter_or_digit(u) || u == u16::from(b'_'))
    {
        return Cow::Borrowed(name);
    }

    let mut out = String::with_capacity(name.len() + 8);
    let mut units = name.encode_utf16();
    let first = units.next().unwrap_or_default();
    if is_letter(first) || first == u16::from(b'_') {
        if let Some(c) = char::from_u32(u32::from(first)) {
            out.push(c);
        }
    } else if is_digit(first) {
        out.push('_');
        if let Some(c) = char::from_u32(u32::from(first)) {
            out.push(c);
        }
    } else {
        let _ = write!(out, "_x{first:X}");
    }
    for unit in units {
        if is_letter_or_digit(unit) || unit == u16::from(b'_') {
            if let Some(c) = char::from_u32(u32::from(unit)) {
                out.push(c);
            }
        } else if is_digit(unit) {
            out.push('_');
            if let Some(c) = char::from_u32(u32::from(unit)) {
                out.push(c);
            }
        } else {
            let _ = write!(out, "_x{unit:X}");
        }
    }
    Cow::Owned(out)
}

pub(crate) fn java_avro_name(name: &str) -> Cow<'_, str> {
    sanitize_name(name, is_java_letter, is_java_letter_or_digit, is_java_digit)
}

pub(crate) fn strict_avro_name(name: &str) -> Cow<'_, str> {
    sanitize_name(
        name,
        is_ascii_letter,
        is_ascii_letter_or_digit,
        is_ascii_digit,
    )
}

fn is_apache_avro_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub(crate) fn sanitize_avro_value_names(mut value: AvroValue) -> AvroValue {
    sanitize_avro_value_names_in_place(&mut value);
    value
}

fn sanitize_avro_value_names_in_place(value: &mut AvroValue) {
    match value {
        AvroValue::Record(fields) => {
            for (name, field) in fields.iter_mut() {
                if let Cow::Owned(renamed) = java_avro_name(name) {
                    *name = renamed;
                }
                sanitize_avro_value_names_in_place(field);
            }
        }
        AvroValue::Array(items) => items
            .iter_mut()
            .for_each(sanitize_avro_value_names_in_place),
        AvroValue::Map(map) => map
            .values_mut()
            .for_each(sanitize_avro_value_names_in_place),
        AvroValue::Union(_, inner) => sanitize_avro_value_names_in_place(inner),
        _ => {}
    }
}

pub(crate) fn strictify_avro_field_names(schema: &mut AvroSchema) {
    match schema {
        AvroSchema::Record(record) => {
            let mut renamed = false;
            for field in &mut record.fields {
                if let Cow::Owned(name) = strict_avro_name(&field.name) {
                    field.name = name;
                    renamed = true;
                }
                strictify_avro_field_names(&mut field.schema);
            }
            if renamed {
                record.lookup = record
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| (f.name.clone(), i))
                    .collect();
            }
        }
        AvroSchema::Union(union) => {
            let mut variants = union.variants().to_vec();
            for variant in &mut variants {
                strictify_avro_field_names(variant);
            }
            if let Ok(union) = UnionSchema::new(variants) {
                *schema = AvroSchema::Union(union);
            }
        }
        AvroSchema::Array(array) => strictify_avro_field_names(&mut array.items),
        AvroSchema::Map(map) => strictify_avro_field_names(&mut map.types),
        _ => {}
    }
}

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

fn read_avro_metadata(cursor: &mut &[u8]) -> Option<Vec<(String, Vec<u8>)>> {
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
            let key = String::from_utf8(read_avro_bytes(cursor)?.to_vec()).ok()?;
            let value = read_avro_bytes(cursor)?.to_vec();
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
    let canonical = java_avro_name(&original);
    let target = if is_apache_avro_name(&canonical) {
        canonical
    } else {
        strict_avro_name(&original)
    };
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
    const MAGIC: &[u8; 4] = b"Obj\x01";
    if !bs.starts_with(MAGIC) {
        return Ok(Cow::Borrowed(bs));
    }
    let mut cursor = &bs[4..];
    let Some(entries) = read_avro_metadata(&mut cursor) else {
        return Ok(Cow::Borrowed(bs));
    };
    let Some(schema_bytes) = entries
        .iter()
        .find(|(key, _)| key == "avro.schema")
        .map(|(_, value)| value.clone())
    else {
        return Ok(Cow::Borrowed(bs));
    };
    #[cfg(test)]
    OCF_JSON_PARSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let Ok(mut json) = serde_json::from_slice::<JsonValue>(&schema_bytes) else {
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
    out.extend_from_slice(MAGIC);
    write_avro_long(&mut out, i64::try_from(entries.len()).unwrap_or(i64::MAX));
    for (key, value) in entries {
        write_avro_long(&mut out, i64::try_from(key.len()).unwrap_or(i64::MAX));
        out.extend_from_slice(key.as_bytes());
        if key == "avro.schema" {
            write_avro_long(
                &mut out,
                i64::try_from(patched_schema.len()).unwrap_or(i64::MAX),
            );
            out.extend_from_slice(&patched_schema);
        } else {
            write_avro_long(&mut out, i64::try_from(value.len()).unwrap_or(i64::MAX));
            out.extend_from_slice(&value);
        }
    }
    write_avro_long(&mut out, 0);
    out.extend_from_slice(cursor);
    Ok(Cow::Owned(out))
}
pub(crate) fn avro_field_name(
    field_name: &str,
    custom_attributes: &mut BTreeMap<String, JsonValue>,
) -> String {
    let avro_name = java_avro_name(field_name);
    if matches!(avro_name, Cow::Owned(_)) {
        custom_attributes.insert(
            ICEBERG_FIELD_NAME_PROP.to_string(),
            JsonValue::String(field_name.to_string()),
        );
    }
    avro_name.into_owned()
}

pub(crate) fn set_record_name(schema: &mut AvroSchema, name: String) {
    if let AvroSchema::Record(record) = schema {
        record.name = Name::from(name.as_str());
    }
}

pub(crate) fn iceberg_field_name(field: &AvroRecordField) -> &str {
    field
        .custom_attributes
        .get(ICEBERG_FIELD_NAME_PROP)
        .and_then(|value| value.as_str())
        .unwrap_or(field.name.as_str())
}
