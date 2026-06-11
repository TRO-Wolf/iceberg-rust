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

//! Variant value decode tests: hand-built vectors per primitive layout, object/array
//! semantics, the malformed-input (security boundary) suite, and the Java-1.10.0-pinned
//! fixture bytes.
//!
//! # Provenance of the `JAVA_*` fixture constants
//!
//! Generated on 2026-06-11 by `/tmp/variant-fixture-gen/VariantFixtureGen.java` (quoted per
//! constant below) compiled and run against the PINNED 1.10.0 jars:
//!
//! ```text
//! CP=~/.m2/repository/org/apache/iceberg/iceberg-api/1.10.0/iceberg-api-1.10.0.jar:\
//!    ~/.m2/repository/org/apache/iceberg/iceberg-core/1.10.0/iceberg-core-1.10.0.jar:\
//!    ~/.m2/repository/org/apache/iceberg/iceberg-bundled-guava/1.10.0/iceberg-bundled-guava-1.10.0.jar:\
//!    ~/.m2/repository/org/slf4j/slf4j-api/2.0.17/slf4j-api-2.0.17.jar
//! javac -cp "$CP" VariantFixtureGen.java && java -cp "$CP:." VariantFixtureGen
//! ```
//!
//! Each value was produced by `Variants.<factory>(...)` (iceberg-core 1.10.0), serialized via
//! the public `VariantValue.writeTo` / `VariantMetadata.writeTo` into a little-endian buffer
//! of `sizeInBytes()` bytes, hex-dumped, and round-trip re-read by Java 1.10.0 itself
//! (`Variants.value(metadata, bytes)`, asserted equal) before being pinned here.

use super::*;

/// Decodes a fixture hex string into bytes (test helper).
fn hex(hex_string: &str) -> Vec<u8> {
    assert!(
        hex_string.len().is_multiple_of(2),
        "hex fixtures have even length"
    );
    (0..hex_string.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&hex_string[index..index + 2], 16).expect("valid fixture hex")
        })
        .collect()
}

/// The empty metadata (Java `SerializedMetadata.EMPTY_V1_BUFFER`), for values that never
/// touch the dictionary.
fn empty_metadata() -> VariantMetadata {
    VariantMetadata::parse(&[0x01, 0x00, 0x00]).expect("the empty metadata must parse")
}

/// Parses a value against the empty metadata, panicking the test on error.
fn parse_ok(bytes: &[u8]) -> VariantValue {
    VariantValue::parse(&empty_metadata(), bytes).expect("test vector must parse")
}

/// Asserts the bytes decode to the given primitive.
fn assert_primitive(bytes: &[u8], expected: VariantPrimitive) {
    match parse_ok(bytes) {
        VariantValue::Primitive(primitive) => assert_eq!(primitive, expected),
        other => panic!("expected a primitive, got {other:?}"),
    }
}

/// Asserts the bytes are rejected with an error (and, by running, that they never panic).
fn assert_rejects(bytes: &[u8]) {
    assert!(
        VariantValue::parse(&empty_metadata(), bytes).is_err(),
        "malformed value {bytes:02x?} must be rejected"
    );
}

// ===== hand-built per-primitive vectors =====================================================
// Layouts per Java `SerializedPrimitive.read()` (1.10.0 bytecode-verified): header byte
// `type_info << 2`, then the payload at offset 1.

/// Risk pinned: null/true/false carry their value in the TYPE ID (ids 0/1/2, no payload) — a
/// transposed id silently flips booleans.
#[test]
fn test_primitive_null_true_false_decode_from_type_id_alone() {
    assert_primitive(&[0x00], VariantPrimitive::Null);
    assert_primitive(&[0x04], VariantPrimitive::Boolean(true));
    assert_primitive(&[0x08], VariantPrimitive::Boolean(false));
}

/// Risk pinned: integer payloads are little-endian two's complement at offset 1 — the most
/// negative values are the sign-extension/byte-order sentinels (i64::MIN's only set bit is in
/// the LAST byte).
#[test]
fn test_primitive_integers_decode_boundary_values() {
    // int8 (id 3): header 0x0C.
    assert_primitive(&[0x0C, 0x80], VariantPrimitive::Int8(i8::MIN));
    assert_primitive(&[0x0C, 0x7F], VariantPrimitive::Int8(i8::MAX));
    // int16 (id 4): header 0x10.
    assert_primitive(&[0x10, 0x00, 0x80], VariantPrimitive::Int16(i16::MIN));
    // int32 (id 5): header 0x14.
    assert_primitive(
        &[0x14, 0x00, 0x00, 0x00, 0x80],
        VariantPrimitive::Int32(i32::MIN),
    );
    // int64 (id 6): header 0x18.
    let mut int64_min = vec![0x18];
    int64_min.extend_from_slice(&i64::MIN.to_le_bytes());
    assert_primitive(&int64_min, VariantPrimitive::Int64(i64::MIN));
}

/// Risk pinned: float/double are IEEE-754 little-endian — compared at exact bit precision so
/// an encoding drift cannot hide behind float tolerance.
#[test]
fn test_primitive_float_double_decode_exact_bits() {
    let mut float_bytes = vec![0x38]; // id 14
    float_bytes.extend_from_slice(&(-1.25f32).to_le_bytes());
    match parse_ok(&float_bytes) {
        VariantValue::Primitive(VariantPrimitive::Float(value)) => {
            assert_eq!(value.to_bits(), (-1.25f32).to_bits());
        }
        other => panic!("expected a float, got {other:?}"),
    }

    let mut double_bytes = vec![0x1C]; // id 7
    double_bytes.extend_from_slice(&2.5f64.to_le_bytes());
    match parse_ok(&double_bytes) {
        VariantValue::Primitive(VariantPrimitive::Double(value)) => {
            assert_eq!(value.to_bits(), 2.5f64.to_bits());
        }
        other => panic!("expected a double, got {other:?}"),
    }
}

/// Risk pinned: decimals are a raw scale byte + a little-endian unscaled value — negative
/// unscaled values, the max scale byte (255 — Java accepts any byte, no validation), and
/// i128::MIN (decimal16's byte-order sentinel) must all survive.
#[test]
fn test_primitive_decimals_decode_negative_and_max_scale_and_i128_min() {
    // decimal4 (id 8): header 0x20; scale 255 (Java reads the raw byte, never validates).
    let mut decimal4 = vec![0x20, 0xFF];
    decimal4.extend_from_slice(&(-7i32).to_le_bytes());
    assert_primitive(&decimal4, VariantPrimitive::Decimal4 {
        scale: 255,
        unscaled: -7,
    });

    // decimal8 (id 9): header 0x24.
    let mut decimal8 = vec![0x24, 0x09];
    decimal8.extend_from_slice(&i64::MIN.to_le_bytes());
    assert_primitive(&decimal8, VariantPrimitive::Decimal8 {
        scale: 9,
        unscaled: i64::MIN,
    });

    // decimal16 (id 10): header 0x28; 16 little-endian payload bytes — Java reverses them
    // into a big-endian BigInteger (`SerializedPrimitive.read`, offsets 17 down to 2), which
    // is exactly i128::from_le_bytes.
    let mut decimal16 = vec![0x28, 0x26];
    decimal16.extend_from_slice(&i128::MIN.to_le_bytes());
    assert_primitive(&decimal16, VariantPrimitive::Decimal16 {
        scale: 38,
        unscaled: i128::MIN,
    });
}

/// Risk pinned: date is an i32 day ordinal, the temporal types are i64 — all signed (pre-epoch
/// values are negative), each with its own type id (a transposed id mislabels micros as
/// nanos, a silent 1000x error).
#[test]
fn test_primitive_temporal_types_decode_signed_values() {
    let mut date = vec![0x2C]; // id 11
    date.extend_from_slice(&(-3000i32).to_le_bytes());
    assert_primitive(&date, VariantPrimitive::Date(-3000));

    let mut timestamptz = vec![0x30]; // id 12
    timestamptz.extend_from_slice(&(-1_000_000i64).to_le_bytes());
    assert_primitive(&timestamptz, VariantPrimitive::Timestamptz(-1_000_000));

    let mut timestampntz = vec![0x34]; // id 13
    timestampntz.extend_from_slice(&7i64.to_le_bytes());
    assert_primitive(&timestampntz, VariantPrimitive::Timestampntz(7));

    let mut time = vec![0x44]; // id 17
    time.extend_from_slice(&86_399_999_999i64.to_le_bytes());
    assert_primitive(&time, VariantPrimitive::Time(86_399_999_999));

    let mut timestamptz_nanos = vec![0x48]; // id 18
    timestamptz_nanos.extend_from_slice(&(-5i64).to_le_bytes());
    assert_primitive(&timestamptz_nanos, VariantPrimitive::TimestamptzNanos(-5));

    let mut timestampntz_nanos = vec![0x4C]; // id 19
    timestampntz_nanos.extend_from_slice(&9i64.to_le_bytes());
    assert_primitive(&timestampntz_nanos, VariantPrimitive::TimestampntzNanos(9));
}

/// Risk pinned: the UUID payload is stored big-endian (RFC 4122) and must come back byte-for-
/// byte — any reordering scrambles every UUID read.
#[test]
fn test_primitive_uuid_preserves_stored_byte_order() {
    let uuid_bytes: [u8; 16] = [
        0xF2, 0x4F, 0x9B, 0x64, 0x81, 0xFA, 0x49, 0xD1, 0xB7, 0x4E, 0x8C, 0x09, 0xA6, 0xE3, 0x1C,
        0x56,
    ];
    let mut value = vec![0x50]; // id 20
    value.extend_from_slice(&uuid_bytes);
    assert_primitive(&value, VariantPrimitive::Uuid(uuid_bytes));
}

/// Risk pinned: binary and long-form string carry an i32 length at offset 1 and the payload at
/// offset 5 — including the empty cases (length 0).
#[test]
fn test_primitive_binary_and_long_string_decode_including_empty() {
    // binary (id 15): header 0x3C.
    let mut binary = vec![0x3C];
    binary.extend_from_slice(&4u32.to_le_bytes());
    binary.extend_from_slice(&[0x0A, 0x0B, 0x0C, 0x0D]);
    assert_primitive(
        &binary,
        VariantPrimitive::Binary(vec![0x0A, 0x0B, 0x0C, 0x0D]),
    );

    let mut empty_binary = vec![0x3C];
    empty_binary.extend_from_slice(&0u32.to_le_bytes());
    assert_primitive(&empty_binary, VariantPrimitive::Binary(vec![]));

    // string (id 16): header 0x40.
    let mut string = vec![0x40];
    string.extend_from_slice(&7u32.to_le_bytes());
    string.extend_from_slice(b"iceberg");
    assert_primitive(&string, VariantPrimitive::String("iceberg".to_string()));

    let mut empty_string = vec![0x40];
    empty_string.extend_from_slice(&0u32.to_le_bytes());
    assert_primitive(&empty_string, VariantPrimitive::String(String::new()));
}

/// Risk pinned: the short-string length is the high 6 header bits — empty (0), 1, and the
/// 63-byte maximum are the mask/shift boundary cases, and multi-byte UTF-8 must survive.
#[test]
fn test_short_string_lengths_empty_one_and_max_63() {
    // length 0: header 0b000001.
    assert_primitive(&[0x01], VariantPrimitive::String(String::new()));
    // length 1: header (1 << 2) | 1 = 0x05.
    assert_primitive(&[0x05, b'x'], VariantPrimitive::String("x".to_string()));
    // length 63 (the 6-bit max): header (63 << 2) | 1 = 0xFD.
    let body = "y".repeat(63);
    let mut value = vec![0xFD];
    value.extend_from_slice(body.as_bytes());
    assert_primitive(&value, VariantPrimitive::String(body));
    // multi-byte UTF-8 (6 bytes, 2 chars).
    let mut utf8 = vec![(6 << 2) | 1];
    utf8.extend_from_slice("日本".as_bytes());
    assert_primitive(&utf8, VariantPrimitive::String("日本".to_string()));
}

// ===== malformed-input suite (the security boundary) ========================================
// Every case must return Err — and by RUNNING these, a panic in any of them fails the test.

/// Risk pinned: the degenerate empty input must error cleanly at the header read.
#[test]
fn test_empty_value_rejects() {
    assert_rejects(&[]);
}

/// Risk pinned: a truncated payload for EVERY fixed-size primitive must be a clean error —
/// these are the exact reads that would be out-of-bounds panics if unchecked.
#[test]
fn test_primitive_truncated_payloads_reject() {
    // header-only for every payload-carrying type id.
    for type_id in [3u8, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20] {
        assert_rejects(&[type_id << 2]);
    }
    // one byte short of each fixed payload.
    assert_rejects(&[0x10, 0x01]); // int16 with 1 of 2 bytes
    assert_rejects(&[0x14, 0x01, 0x02, 0x03]); // int32 with 3 of 4
    assert_rejects(&[0x18, 0, 0, 0, 0, 0, 0, 0]); // int64 with 7 of 8
    assert_rejects(&[0x28, 0x05, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // decimal16 with 9 of 16
    assert_rejects(&[0x50, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); // uuid with 10 of 16
    // binary/string: a length prefix but a short payload.
    let mut binary = vec![0x3C];
    binary.extend_from_slice(&10u32.to_le_bytes());
    binary.extend_from_slice(&[1, 2, 3]);
    assert_rejects(&binary);
}

/// Risk pinned: type ids above Java's 0..=20 set (a future spec or garbage) must error, never
/// decode as something else.
#[test]
fn test_unknown_primitive_type_id_rejects() {
    assert_rejects(&[21 << 2]);
    assert_rejects(&[63 << 2, 0x00]);
}

/// Risk pinned: a hostile binary/string length — negative in Java's signed domain
/// (0x8000_0000) or absurdly larger than the buffer — must fail fast by name, with no
/// allocation sized from the untrusted value.
#[test]
fn test_string_and_binary_hostile_lengths_reject() {
    for header in [0x3Cu8, 0x40] {
        let mut negative = vec![header];
        negative.extend_from_slice(&0x8000_0000u32.to_le_bytes());
        negative.extend_from_slice(&[0u8; 8]);
        assert_rejects(&negative);

        let mut absurd = vec![header];
        absurd.extend_from_slice(&(i32::MAX as u32).to_le_bytes());
        absurd.extend_from_slice(&[0u8; 8]);
        assert_rejects(&absurd);
    }
}

/// Risk pinned: a short string whose 6-bit length exceeds the actual bytes must error.
#[test]
fn test_short_string_truncated_rejects() {
    assert_rejects(&[(5 << 2) | 1, b'a', b'b']);
}

/// Risk pinned (documented divergence): invalid UTF-8 in string payloads errors loudly here
/// (Java silently substitutes U+FFFD).
#[test]
fn test_invalid_utf8_in_strings_rejects() {
    // short string, 2 bytes of invalid UTF-8.
    assert_rejects(&[(2 << 2) | 1, 0xC3, 0x28]);
    // long string, same payload.
    let mut long = vec![0x40];
    long.extend_from_slice(&2u32.to_le_bytes());
    long.extend_from_slice(&[0xC3, 0x28]);
    assert_rejects(&long);
}

// ===== objects ===============================================================================
// Layout per Java `SerializedObject` (1.10.0): header bits — offset size (bits 2..4), field-id
// size (bits 4..6), is-large (bit 6) — then the field count, the field-id list, the offset
// list (one extra entry = data length), then the concatenated field values.

/// Builds metadata with the given dictionary, sorted-flagged, offset size 1 (test helper).
fn metadata_with(names: &[&str]) -> VariantMetadata {
    let bytes = crate::variant::metadata::tests::encode_metadata(names, 1, true);
    VariantMetadata::parse(&bytes).expect("test dictionary must parse")
}

/// Risk pinned: the empty object (0 fields, only the data-length offset entry) must decode —
/// it is the `{}` of every sparse row.
#[test]
fn test_object_empty_decodes() {
    let object_bytes = [0x02u8, 0x00, 0x00];
    let value = VariantValue::parse(&metadata_with(&[]), &object_bytes).expect("empty object");
    let object = value.as_object().expect("must be an object");
    assert_eq!(object.num_fields(), 0);
    assert_eq!(object.get("a"), None);
    assert_eq!(value.physical_type(), PhysicalType::Object);
}

/// Risk pinned: the single-field happy path plus the `get` hit/miss contract (Java returns
/// null on a miss → `None` here).
#[test]
fn test_object_single_field_get_hit_and_miss() {
    // {a: int8 34}: header 0x02, count 1, field ids [0], offsets [0, 2], data 0x0C 0x22.
    let object_bytes = [0x02u8, 0x01, 0x00, 0x00, 0x02, 0x0C, 0x22];
    let metadata = metadata_with(&["a"]);
    let value = VariantValue::parse(&metadata, &object_bytes).expect("must parse");
    let object = value.as_object().expect("must be an object");
    assert_eq!(object.num_fields(), 1);
    assert_eq!(
        object.get("a"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(34)))
    );
    assert_eq!(object.get("missing"), None);
    assert_eq!(object.fields()[0].field_id, 0);
    assert_eq!(object.fields()[0].name, "a");
}

/// Risk pinned: the non-trivial header combination — is-large (4-byte count), 2-byte field
/// ids, 2-byte offsets — exercises every size field's mask/shift at once; a wrong shift
/// misreads all of them.
#[test]
fn test_object_large_with_two_byte_ids_and_offsets_decodes() {
    // header: is_large | (field_id_size 2 - 1) << 4 | (offset_size 2 - 1) << 2 | 0b10 = 0x56.
    let mut object_bytes = vec![0x56u8];
    object_bytes.extend_from_slice(&2u32.to_le_bytes()); // count (4 bytes, is_large)
    object_bytes.extend_from_slice(&0u16.to_le_bytes()); // field id "a"
    object_bytes.extend_from_slice(&1u16.to_le_bytes()); // field id "b"
    object_bytes.extend_from_slice(&0u16.to_le_bytes()); // offset of a
    object_bytes.extend_from_slice(&2u16.to_le_bytes()); // offset of b
    object_bytes.extend_from_slice(&10u16.to_le_bytes()); // data length
    object_bytes.extend_from_slice(&[0x0C, 0xDE]); // a: int8 -34
    object_bytes.extend_from_slice(&[(7 << 2) | 1]); // b: short string "iceberg"
    object_bytes.extend_from_slice(b"iceberg");

    let metadata = metadata_with(&["a", "b"]);
    let value = VariantValue::parse(&metadata, &object_bytes).expect("must parse");
    let object = value.as_object().expect("must be an object");
    assert_eq!(object.num_fields(), 2);
    assert_eq!(
        object.get("a"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(-34)))
    );
    assert_eq!(
        object.get("b"),
        Some(&VariantValue::Primitive(VariantPrimitive::String(
            "iceberg".to_string()
        )))
    );
    assert_eq!(object.field_names().collect::<Vec<_>>(), vec!["a", "b"]);
}

/// Risk pinned: object field values are located by SORTED-DISTINCT offset spans (Java
/// `initOffsetsAndLengths`), not consecutive entries — fields whose data order differs from
/// name order must still decode correctly.
#[test]
fn test_object_field_data_order_differs_from_field_order() {
    // Fields in name order [a, b], but a's bytes AFTER b's: offsets a=8, b=0.
    // data: b = "iceberg" (8 bytes), a = int8 34 (2 bytes); data length 10.
    let object_bytes = [
        0x02u8,
        0x02,
        0x00,
        0x01,
        0x08,
        0x00,
        0x0A, // header, count, ids, offsets [8, 0, 10]
        (7 << 2) | 1,
        b'i',
        b'c',
        b'e',
        b'b',
        b'e',
        b'r',
        b'g', // b at 0
        0x0C,
        0x22, // a at 8
    ];
    let metadata = metadata_with(&["a", "b"]);
    let value = VariantValue::parse(&metadata, &object_bytes).expect("must parse");
    let object = value.as_object().expect("must be an object");
    assert_eq!(
        object.get("a"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(34)))
    );
    assert_eq!(
        object.get("b"),
        Some(&VariantValue::Primitive(VariantPrimitive::String(
            "iceberg".to_string()
        )))
    );
}

/// Risk pinned (Java-miss parity): `get(name)` binary-searches assuming name-sorted fields —
/// on a NON-conforming object Java 1.10.0 misses some present fields, and a "helpful" linear
/// fallback here would diverge by finding them.
#[test]
fn test_object_get_on_unsorted_fields_misses_exactly_like_java() {
    // Fields stored [b, a] (NOT name-sorted): ids [1, 0], values int8 1 / int8 2.
    let object_bytes = [
        0x02u8, 0x02, 0x01, 0x00, 0x00, 0x02, 0x04, 0x0C, 0x01, 0x0C, 0x02,
    ];
    let metadata = metadata_with(&["a", "b"]);
    let value = VariantValue::parse(&metadata, &object_bytes).expect("must parse");
    let object = value.as_object().expect("must be an object");
    // Java's probe for "a" over names [b, a]: mid 0 -> "b", "a" < "b" -> miss.
    assert_eq!(object.get("a"), None, "Java's binary search misses here");
    // Java's probe for "b": mid 0 -> "b" -> hit.
    assert!(object.get("b").is_some());
    // The field IS present in the decoded structure (only the lookup mirrors Java's miss).
    assert_eq!(object.fields()[1].name, "a");
}

/// Risk pinned: nesting object → array → object must decode through the recursion path with
/// names resolved at every level.
#[test]
fn test_object_array_object_nesting_decodes() {
    let metadata = metadata_with(&["a", "b", "c"]);
    // inner object {c: int8 9}: 02 01 02 00 02 0c 09 (7 bytes)
    let inner_object = [0x02u8, 0x01, 0x02, 0x00, 0x02, 0x0C, 0x09];
    // array [inner_object, short "x"]: 03 02 00 07 09 <inner> 05 78 (14 bytes)
    let mut array = vec![0x03u8, 0x02, 0x00, 0x07, 0x09];
    array.extend_from_slice(&inner_object);
    array.extend_from_slice(&[0x05, b'x']);
    // outer object {a: array, b: double 4.5}
    let mut outer = vec![0x02u8, 0x02, 0x00, 0x01, 0x00, 0x0E, 0x17];
    outer.extend_from_slice(&array);
    outer.push(0x1C);
    outer.extend_from_slice(&4.5f64.to_le_bytes());

    let value = VariantValue::parse(&metadata, &outer).expect("nested value must parse");
    let object = value.as_object().expect("outer object");
    let array_value = object.get("a").expect("field a").as_array().expect("array");
    assert_eq!(array_value.num_elements(), 2);
    let inner = array_value
        .get(0)
        .expect("element 0")
        .as_object()
        .expect("inner object");
    assert_eq!(
        inner.get("c"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(9)))
    );
    assert_eq!(
        array_value.get(1),
        Some(&VariantValue::Primitive(VariantPrimitive::String(
            "x".to_string()
        )))
    );
    assert_eq!(
        object.get("b"),
        Some(&VariantValue::Primitive(VariantPrimitive::Double(4.5)))
    );
}

/// Risk pinned (malformed): a field id outside the dictionary must error at parse (Java
/// throws on access) — a wrong-but-in-range id would silently rename a field, so the id is
/// validated against the dictionary, not clamped.
#[test]
fn test_object_field_id_past_dictionary_rejects() {
    // {<id 5>: int8 1} against a 1-entry dictionary.
    let object_bytes = [0x02u8, 0x01, 0x05, 0x00, 0x02, 0x0C, 0x01];
    let metadata = metadata_with(&["a"]);
    assert!(VariantValue::parse(&metadata, &object_bytes).is_err());
}

/// Risk pinned (malformed): duplicate field offsets break Java's sorted-distinct length
/// scheme (its `sortedOffsets.get(index + 1)` throws) — rejected by name here.
#[test]
fn test_object_duplicate_field_offsets_reject() {
    // 2 fields, both at offset 0, data length 2.
    let object_bytes = [0x02u8, 0x02, 0x00, 0x01, 0x00, 0x00, 0x02, 0x0C, 0x01];
    let metadata = metadata_with(&["a", "b"]);
    let error = VariantValue::parse(&metadata, &object_bytes).expect_err("duplicate offsets");
    assert!(
        error.to_string().contains("duplicate field offsets"),
        "error must name the duplicates, got: {error}"
    );
}

/// Risk pinned (malformed): a declared data length running past the end of the value must
/// error at the field slice, not read out of bounds.
#[test]
fn test_object_field_range_past_end_rejects() {
    // 1 field at offset 0, data length 9, but only 2 data bytes follow.
    let object_bytes = [0x02u8, 0x01, 0x00, 0x00, 0x09, 0x0C, 0x01];
    let metadata = metadata_with(&["a"]);
    assert!(VariantValue::parse(&metadata, &object_bytes).is_err());
}

/// Risk pinned (malformed/DoS): an is-large object declaring i32::MAX fields over a tiny
/// buffer must fail FAST on the header-region bound — before any allocation sized from the
/// untrusted count.
#[test]
fn test_object_absurd_field_count_rejects_fast() {
    let mut object_bytes = vec![0x42u8]; // is_large, 1-byte ids, 1-byte offsets
    object_bytes.extend_from_slice(&(i32::MAX as u32).to_le_bytes());
    object_bytes.extend_from_slice(&[0u8; 16]);
    let metadata = metadata_with(&["a"]);
    assert!(VariantValue::parse(&metadata, &object_bytes).is_err());
}

// ===== arrays ================================================================================
// Layout per Java `SerializedArray` (1.10.0): header bits — offset size (bits 2..4), is-large
// (bit 4) — then the element count and `count + 1` offsets delimiting consecutive elements.

/// Risk pinned: the empty array must decode (count 0, single offset entry).
#[test]
fn test_array_empty_decodes() {
    let value = parse_ok(&[0x03, 0x00, 0x00]);
    let array = value.as_array().expect("must be an array");
    assert_eq!(array.num_elements(), 0);
    assert_eq!(array.get(0), None);
    assert_eq!(value.physical_type(), PhysicalType::Array);
}

/// Risk pinned: mixed-type elements with consecutive offsets, plus the out-of-range `get`
/// contract (Java throws unchecked; `None` here).
#[test]
fn test_array_mixed_types_and_out_of_range_get() {
    // [int8 -34, "iceberg", null, true]: offsets [0, 2, 10, 11, 12].
    let mut array_bytes = vec![0x03u8, 0x04, 0x00, 0x02, 0x0A, 0x0B, 0x0C];
    array_bytes.extend_from_slice(&[0x0C, 0xDE]);
    array_bytes.push((7 << 2) | 1);
    array_bytes.extend_from_slice(b"iceberg");
    array_bytes.extend_from_slice(&[0x00, 0x04]);

    let value = parse_ok(&array_bytes);
    let array = value.as_array().expect("must be an array");
    assert_eq!(array.num_elements(), 4);
    assert_eq!(
        array.get(0),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(-34)))
    );
    assert_eq!(
        array.get(1),
        Some(&VariantValue::Primitive(VariantPrimitive::String(
            "iceberg".to_string()
        )))
    );
    assert_eq!(
        array.get(2),
        Some(&VariantValue::Primitive(VariantPrimitive::Null))
    );
    assert_eq!(
        array.get(3),
        Some(&VariantValue::Primitive(VariantPrimitive::Boolean(true)))
    );
    assert_eq!(
        array.get(4),
        None,
        "out of range is None, like Java's throw"
    );
}

/// Risk pinned (malformed): descending array offsets (Java's `next - offset` slice would be
/// negative) must reject by name, never wrap.
#[test]
fn test_array_descending_offsets_reject() {
    // 2 elements, offsets [2, 0, 4].
    let array_bytes = [0x03u8, 0x02, 0x02, 0x00, 0x04, 0x0C, 0x01, 0x0C, 0x02];
    assert_rejects(&array_bytes);
}

/// Risk pinned (malformed): an element span past the end of the buffer must error at the
/// slice bound.
#[test]
fn test_array_element_past_end_rejects() {
    // 1 element, offsets [0, 9], but only 2 data bytes.
    let array_bytes = [0x03u8, 0x01, 0x00, 0x09, 0x0C, 0x01];
    assert_rejects(&array_bytes);
}

/// Risk pinned (malformed/DoS): an is-large array declaring i32::MAX elements over a tiny
/// buffer fails fast on the header-region bound, before any allocation.
#[test]
fn test_array_absurd_element_count_rejects_fast() {
    let mut array_bytes = vec![0x13u8]; // is_large array, 1-byte offsets
    array_bytes.extend_from_slice(&(i32::MAX as u32).to_le_bytes());
    array_bytes.extend_from_slice(&[0u8; 16]);
    assert_rejects(&array_bytes);
}

/// Risk pinned (DoS — the explicit recursion guard): nesting at exactly
/// [`MAX_NESTING_DEPTH`] parses; one deeper is rejected, NOT a stack overflow. Each wrapper
/// is a 1-element array with 2-byte offsets.
#[test]
fn test_nesting_depth_guard_boundary() {
    // header 0b0111: array, offset_size 2, not large.
    let wrap = |inner: &[u8]| -> Vec<u8> {
        let mut wrapped = vec![0x07u8, 0x01];
        wrapped.extend_from_slice(&0u16.to_le_bytes());
        wrapped.extend_from_slice(&(inner.len() as u16).to_le_bytes());
        wrapped.extend_from_slice(inner);
        wrapped
    };

    // MAX_NESTING_DEPTH wrappers put the innermost null exactly AT the depth limit.
    let mut at_limit = vec![0x00u8];
    for _ in 0..MAX_NESTING_DEPTH {
        at_limit = wrap(&at_limit);
    }
    assert!(
        VariantValue::parse(&empty_metadata(), &at_limit).is_ok(),
        "nesting at the limit must parse"
    );

    let beyond = wrap(&at_limit);
    let error = VariantValue::parse(&empty_metadata(), &beyond)
        .expect_err("nesting beyond the limit must be rejected");
    assert!(
        error.to_string().contains("nesting depth"),
        "error must name the depth guard, got: {error}"
    );
}

/// Risk pinned (Java parity): trailing bytes after a top-level value are IGNORED — Java's
/// lazy reads never touch them, and rejecting them would refuse buffers Java accepts.
#[test]
fn test_top_level_trailing_bytes_tolerated_like_java() {
    let value = parse_ok(&[0x0C, 0x22, 0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(value, VariantValue::Primitive(VariantPrimitive::Int8(34)));
}

/// Risk pinned: the `as_*` accessors mirror Java's `asPrimitive`/`asObject`/`asArray` throws
/// — the wrong kind is an error, not a panic or a silent None.
#[test]
fn test_as_accessors_reject_wrong_kind() {
    let primitive = parse_ok(&[0x00]);
    assert!(primitive.as_object().is_err());
    assert!(primitive.as_array().is_err());
    assert!(primitive.as_primitive().is_ok());

    let array = parse_ok(&[0x03, 0x00, 0x00]);
    assert!(array.as_primitive().is_err());
    assert!(array.as_object().is_err());
    assert!(array.as_array().is_ok());
}

// ===== Variant (metadata + value) ============================================================

/// Risk pinned: `Variant::from_bytes` must slice the value at the metadata's TRUE end (Java
/// `Variant.from` slices at `metadata.sizeInBytes()`) — an off-by-one reads the value header
/// out of the dictionary bytes.
#[test]
fn test_variant_from_bytes_concatenated_metadata_then_value() {
    let metadata_bytes = crate::variant::metadata::tests::encode_metadata(&["a"], 1, true);
    let object_bytes = [0x02u8, 0x01, 0x00, 0x00, 0x02, 0x0C, 0x22];
    let mut buffer = metadata_bytes.clone();
    buffer.extend_from_slice(&object_bytes);

    let variant = Variant::from_bytes(&buffer).expect("concatenated variant must parse");
    assert_eq!(variant.metadata().dictionary_size(), 1);
    let object = variant.value().as_object().expect("object value");
    assert_eq!(
        object.get("a"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(34)))
    );

    // The truncated buffer (metadata only — no value) must error, not panic.
    assert!(Variant::from_bytes(&metadata_bytes).is_err());
}

// ===== Java 1.10.0 pinned fixtures ===========================================================
// See the module doc for generation provenance. Every constant is the EXACT byte output of
// iceberg 1.10.0, so these pin byte-level decode compatibility with the Java implementation.

/// `Variants.emptyMetadata()` → `writeTo` (1.10.0).
const JAVA_METADATA_EMPTY: &str = "010000";
/// `Variants.metadata("a", "b", "c")` → `writeTo` (1.10.0) — sorted, offset size 1.
const JAVA_METADATA_ABC: &str = "110300010203616263";

/// Risk pinned: metadata bytes as Java 1.10.0 actually writes them (header flags included)
/// must decode with the same dictionary and lookup behavior.
#[test]
fn test_java_fixture_metadata_decodes() {
    let empty = VariantMetadata::parse(&hex(JAVA_METADATA_EMPTY)).expect("empty metadata");
    assert_eq!(empty.dictionary_size(), 0);
    assert_eq!(empty.size_in_bytes(), 3);

    let abc = VariantMetadata::parse(&hex(JAVA_METADATA_ABC)).expect("abc metadata");
    assert!(abc.is_sorted(), "1.10.0 writes the sorted flag");
    assert_eq!(abc.dictionary_size(), 3);
    assert_eq!(abc.get(0).expect("id 0"), "a");
    assert_eq!(abc.get(2).expect("id 2"), "c");
    assert_eq!(abc.id("b"), Some(1));
    assert_eq!(abc.size_in_bytes(), hex(JAVA_METADATA_ABC).len());
}

/// Risk pinned: every 1.10.0 primitive physical type decodes from Java's EXACT bytes to the
/// expected value — the cross-implementation decode contract, one (fixture, expected) pair
/// per type id. Provenance: each hex is the `writeTo` output of the quoted `Variants` call.
#[test]
fn test_java_fixture_primitives_decode() {
    let cases: Vec<(&str, &str, VariantPrimitive)> = vec![
        // Variants.ofNull()
        ("primitive_null", "00", VariantPrimitive::Null),
        // Variants.of(true)
        ("primitive_true", "04", VariantPrimitive::Boolean(true)),
        // Variants.of(false)
        ("primitive_false", "08", VariantPrimitive::Boolean(false)),
        // Variants.of((byte) -34)
        ("primitive_int8", "0cde", VariantPrimitive::Int8(-34)),
        // Variants.of((short) -1234)
        ("primitive_int16", "102efb", VariantPrimitive::Int16(-1234)),
        // Variants.of(-12345678)
        (
            "primitive_int32",
            "14b29e43ff",
            VariantPrimitive::Int32(-12345678),
        ),
        // Variants.of(Long.MIN_VALUE)
        (
            "primitive_int64_min",
            "180000000000000080",
            VariantPrimitive::Int64(i64::MIN),
        ),
        // Variants.of(-1.25f)
        (
            "primitive_float",
            "380000a0bf",
            VariantPrimitive::Float(-1.25),
        ),
        // Variants.of(2.5d)
        (
            "primitive_double",
            "1c0000000000000440",
            VariantPrimitive::Double(2.5),
        ),
        // Variants.of(new BigDecimal("-123.4567"))
        (
            "primitive_decimal4",
            "20047929edff",
            VariantPrimitive::Decimal4 {
                scale: 4,
                unscaled: -1234567,
            },
        ),
        // Variants.of(new BigDecimal("-12345678.901234567"))
        (
            "primitive_decimal8",
            "240979b494a2ab23d4ff",
            VariantPrimitive::Decimal8 {
                scale: 9,
                unscaled: -12345678901234567,
            },
        ),
        // Variants.of(new BigDecimal("-9876543210.123456789123456789012345678"))
        (
            "primitive_decimal16",
            "281bb20c9b7ac45ac2fef7a1c7ecd2d891f8",
            VariantPrimitive::Decimal16 {
                scale: 27,
                unscaled: -9876543210123456789123456789012345678,
            },
        ),
        // Variants.ofIsoDate("2024-11-07")
        (
            "primitive_date",
            "2c424e0000",
            VariantPrimitive::Date(20034),
        ),
        // Variants.ofIsoTimestamptz("2024-11-07T12:33:54.123456+00:00")
        (
            "primitive_timestamptz",
            "30c0b2f0d851260600",
            VariantPrimitive::Timestamptz(1730982834123456),
        ),
        // Variants.ofIsoTimestampntz("2024-11-07T12:33:54.123456")
        (
            "primitive_timestampntz",
            "34c0b2f0d851260600",
            VariantPrimitive::Timestampntz(1730982834123456),
        ),
        // Variants.ofIsoTime("12:33:54.123456")
        (
            "primitive_time",
            "44c0f229880a000000",
            VariantPrimitive::Time(45234123456),
        ),
        // Variants.ofIsoTimestamptzNanos("2024-11-07T12:33:54.123456789+00:00")
        (
            "primitive_timestamptz_nanos",
            "4815413a6cb7af0518",
            VariantPrimitive::TimestamptzNanos(1730982834123456789),
        ),
        // Variants.ofIsoTimestampntzNanos("2024-11-07T12:33:54.123456789")
        (
            "primitive_timestampntz_nanos",
            "4c15413a6cb7af0518",
            VariantPrimitive::TimestampntzNanos(1730982834123456789),
        ),
        // Variants.ofUUID("f24f9b64-81fa-49d1-b74e-8c09a6e31c56")
        (
            "primitive_uuid",
            "50f24f9b6481fa49d1b74e8c09a6e31c56",
            VariantPrimitive::Uuid([
                0xF2, 0x4F, 0x9B, 0x64, 0x81, 0xFA, 0x49, 0xD1, 0xB7, 0x4E, 0x8C, 0x09, 0xA6, 0xE3,
                0x1C, 0x56,
            ]),
        ),
        // Variants.of(ByteBuffer.wrap(new byte[] {0x0a, 0x0b, 0x0c, 0x0d}))
        (
            "primitive_binary",
            "3c040000000a0b0c0d",
            VariantPrimitive::Binary(vec![0x0A, 0x0B, 0x0C, 0x0D]),
        ),
        // Variants.of("iceberg") — 7 chars, written as a SHORT string by 1.10.0
        (
            "primitive_short_string",
            "1d69636562657267",
            VariantPrimitive::String("iceberg".to_string()),
        ),
        // Variants.of("x".repeat(70)) — 70 chars, written as a LONG string
        (
            "primitive_long_string",
            "404600000078787878787878787878787878787878787878787878787878787878787878787878\
             787878787878787878787878787878787878787878787878787878787878787878787878787878\
             7878",
            VariantPrimitive::String("x".repeat(70)),
        ),
    ];
    for (name, fixture_hex, expected) in cases {
        let bytes = hex(&fixture_hex.replace(char::is_whitespace, ""));
        match VariantValue::parse(&empty_metadata(), &bytes) {
            Ok(VariantValue::Primitive(primitive)) => {
                assert_eq!(primitive, expected, "fixture {name} decoded wrong");
            }
            other => panic!("fixture {name} must decode to a primitive, got {other:?}"),
        }
    }
}

/// Risk pinned: an OBJECT as 1.10.0 writes it (`Variants.object(metadata)` with
/// `put("a", Variants.of((byte) -34))`, `put("b", Variants.of("iceberg"))`, against
/// `Variants.metadata("a", "b", "c")`) decodes with the right ids, names, and values.
#[test]
fn test_java_fixture_object_decodes() {
    let metadata = VariantMetadata::parse(&hex(JAVA_METADATA_ABC)).expect("abc metadata");
    let value = VariantValue::parse(&metadata, &hex("0202000100020a0cde1d69636562657267"))
        .expect("object fixture must parse");
    let object = value.as_object().expect("must be an object");
    assert_eq!(object.num_fields(), 2);
    assert_eq!(
        object.get("a"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(-34)))
    );
    assert_eq!(
        object.get("b"),
        Some(&VariantValue::Primitive(VariantPrimitive::String(
            "iceberg".to_string()
        )))
    );
    assert_eq!(
        object.get("c"),
        None,
        "in the dictionary but not the object"
    );

    // The Java `Variant.from(ByteBuffer)` layout: metadata immediately followed by the value.
    let mut concatenated = hex(JAVA_METADATA_ABC);
    concatenated.extend_from_slice(&hex("0202000100020a0cde1d69636562657267"));
    let variant = Variant::from_bytes(&concatenated).expect("concatenated Java bytes");
    assert_eq!(variant.value().physical_type(), PhysicalType::Object);
}

/// Risk pinned: an ARRAY as 1.10.0 writes it (`Variants.array()` with int8 -34, "iceberg",
/// null, true) decodes element-for-element in order.
#[test]
fn test_java_fixture_array_decodes() {
    let metadata = VariantMetadata::parse(&hex(JAVA_METADATA_ABC)).expect("abc metadata");
    let value = VariantValue::parse(&metadata, &hex("030400020a0b0c0cde1d696365626572670004"))
        .expect("array fixture must parse");
    let array = value.as_array().expect("must be an array");
    assert_eq!(array.num_elements(), 4);
    assert_eq!(
        array.get(0),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(-34)))
    );
    assert_eq!(
        array.get(1),
        Some(&VariantValue::Primitive(VariantPrimitive::String(
            "iceberg".to_string()
        )))
    );
    assert_eq!(
        array.get(2),
        Some(&VariantValue::Primitive(VariantPrimitive::Null))
    );
    assert_eq!(
        array.get(3),
        Some(&VariantValue::Primitive(VariantPrimitive::Boolean(true)))
    );
}

// ===== Java 1.10.0 reviewer probes =========================================================
// Bytes below were generated by / fed to Java 1.10.0 via /tmp/variant-probe/VariantProbe.java
// (same classpath as the fixture generator, 2026-06-11); each test quotes the observed Java
// behavior it pins.

/// Risk pinned (THE UTF-16 comparator trap, Java-generated bytes): Java's writer sorts object
/// fields by `String.compareTo` — UTF-16 code units — so the supplementary 😀 (U+1F600,
/// surrogate D83D) sorts BEFORE the BMP U+FFFF. `Variants.object` on 1.10.0 wrote the fields
/// in that order, and its `get` found both (probe p1: get_bmp=1, get_supp=2). A byte-order
/// comparator would probe left at "😀" and silently MISS U+FFFF — the corruption class this
/// module exists to avoid.
#[test]
fn test_java_fixture_object_utf16_field_order_lookup_finds_both_names() {
    // Variants.metadata("\u{FFFF}", "\u{1F600}") — 1.10.0 wrote it UNSORTED (insertion order:
    // the input is not compareTo-sorted), dictionary [U+FFFF, U+1F600].
    let metadata = VariantMetadata::parse(&hex("0102000307efbfbff09f9880"))
        .expect("Java-written metadata must parse");
    assert!(!metadata.is_sorted());
    assert_eq!(metadata.id("\u{FFFF}"), Some(0));
    assert_eq!(metadata.id("\u{1F600}"), Some(1));

    // Object {U+FFFF: int8 1, U+1F600: int8 2} — field order on disk is [😀, ￿] (UTF-16).
    let value = VariantValue::parse(&metadata, &hex("020201000002040c020c01"))
        .expect("Java-written object must parse");
    let object = value.as_object().expect("must be an object");
    assert_eq!(
        object.field_names().collect::<Vec<_>>(),
        vec!["\u{1F600}", "\u{FFFF}"],
        "Java sorts fields in UTF-16 order: supplementary below U+FFFF"
    );
    assert_eq!(
        object.get("\u{FFFF}"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(1))),
        "a byte-order comparator would walk left past 😀 and miss this BMP name"
    );
    assert_eq!(
        object.get("\u{1F600}"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(2)))
    );
}

/// Risk pinned: the array is-large bit is bit 4 (`SerializedArray.IS_LARGE = 16`) — a
/// transposed bit test (e.g. the object's bit 6) reads a garbage count for every large array.
/// Java 1.10.0 decoded these bytes as a 1-element array of int8 5 (probe p13).
#[test]
fn test_java_probe_large_array_four_byte_count_decodes() {
    let value = parse_ok(&hex("130100000000020c05"));
    let array = value.as_array().expect("must be an array");
    assert_eq!(array.num_elements(), 1);
    assert_eq!(
        array.get(0),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(5)))
    );
}

/// Risk pinned: the object is-large bit is bit 6 (`SerializedObject.IS_LARGE = 64`), NOT bit 4
/// — bit 4 belongs to the field-id-size field, so a NON-large object with 2-byte field ids
/// (header 0x12) is the input that exposes a transposed bit (it would be misread as large,
/// consuming a 4-byte count). Java 1.10.0 decoded these bytes as {a: int8 34} (probe p14).
#[test]
fn test_java_probe_object_two_byte_field_ids_not_large_decodes() {
    let metadata = VariantMetadata::parse(&hex(JAVA_METADATA_ABC)).expect("abc metadata");
    let value = VariantValue::parse(&metadata, &hex("1201000000020c22"))
        .expect("non-large object with 2-byte field ids must parse");
    let object = value.as_object().expect("must be an object");
    assert_eq!(object.num_fields(), 1);
    assert_eq!(
        object.get("a"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(34)))
    );
}

/// Risk pinned (accepted-set parity): an array whose FIRST offset is nonzero (gap bytes before
/// the element data) is legal — Java 1.10.0 reads element 0 from `dataOffset + offset[0]` and
/// accepted these bytes (probe p8: elem0=7). Rejecting the gap would refuse Java-readable data.
#[test]
fn test_array_gap_before_first_offset_tolerated_like_java() {
    let value = parse_ok(&hex("03010204eeee0c07"));
    let array = value.as_array().expect("must be an array");
    assert_eq!(
        array.get(0),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(7)))
    );
}

/// Risk pinned (accepted-set parity): an object field whose offset SPAN is larger than the
/// value inside it (short string "x" + 2 slack bytes) is legal — Java's lazy reads never touch
/// the slack and 1.10.0 accepted these bytes (probe p9: a="x"); the eager parse must ignore
/// trailing bytes inside a field span exactly as it does at top level.
#[test]
fn test_object_field_span_slack_tolerated_like_java() {
    let metadata = VariantMetadata::parse(&hex(JAVA_METADATA_ABC)).expect("abc metadata");
    let value = VariantValue::parse(&metadata, &hex("02010000040578eeee"))
        .expect("field span slack must parse");
    let object = value.as_object().expect("must be an object");
    assert_eq!(
        object.get("a"),
        Some(&VariantValue::Primitive(VariantPrimitive::String(
            "x".to_string()
        )))
    );
}

/// Risk pinned (DOCUMENTED DIVERGENCE — see the module doc): Java 1.10.0 ACCEPTS a zero-count
/// object/array whose mandatory final offset entry is truncated away (`[0x02, 0x00]` /
/// `[0x03, 0x00]`) and an empty-dictionary metadata whose declared string-data end overruns
/// the buffer (`[0x01, 0x00, 0x05]`), because its lazy reader never reads those regions when
/// the count is zero (probes p2/p3/p4: numFields=0 / numElements=0 / dictionarySize=0, no
/// throw). These are spec-violating shapes no Java writer emits; this port deliberately
/// rejects all three at the door.
#[test]
fn test_truncated_empty_containers_reject_documented_divergence() {
    assert_rejects(&[0x02, 0x00]);
    assert_rejects(&[0x03, 0x00]);
    assert!(VariantMetadata::parse(&[0x01, 0x00, 0x05]).is_err());
}

/// Risk pinned: the metadata's DECLARED data length (the final offset entry) — not the buffer
/// length — locates the value region. Java 1.10.0 truncated `[01 00 05]` + 10-byte buffer to
/// metadata size 8 and read the value at offset 8 (probe p15: metadataSize=8, value=34);
/// a `size_in_bytes` that reported the un-truncated buffer length would misread every
/// concatenated variant carrying trailing dictionary slack.
#[test]
fn test_variant_from_bytes_declared_metadata_end_shifts_value_start_like_java() {
    let variant = Variant::from_bytes(&hex("010005eeeeeeeeee0c22"))
        .expect("over-declared (in-bounds) metadata data length must parse");
    assert_eq!(variant.metadata().size_in_bytes(), 8);
    assert_eq!(variant.metadata().dictionary_size(), 0);
    assert_eq!(
        variant.value(),
        &VariantValue::Primitive(VariantPrimitive::Int8(34))
    );
}

/// Risk pinned (DoS, the wide axis): the depth guard bounds DEEP inputs; a WIDE input
/// (70,000 sibling nulls, ~280 KB of offsets) must decode in linear time with allocation
/// clamped by the buffer length — no quadratic blowup, no count-driven pre-allocation.
#[test]
fn test_wide_array_70000_elements_decodes_cheaply() {
    const COUNT: usize = 70_000;
    // header 0x1B: array, is-large (bit 4), offset size 3.
    let mut bytes = vec![0x1Bu8];
    bytes.extend_from_slice(&(COUNT as u32).to_le_bytes());
    for offset in 0..=COUNT {
        bytes.extend_from_slice(&offset.to_le_bytes()[..3]);
    }
    bytes.extend_from_slice(&vec![0x00u8; COUNT]); // one null per element
    let value = parse_ok(&bytes);
    let array = value.as_array().expect("must be an array");
    assert_eq!(array.num_elements(), COUNT);
    assert_eq!(
        array.get(COUNT - 1),
        Some(&VariantValue::Primitive(VariantPrimitive::Null))
    );
}

/// Risk pinned: 1.10.0's NESTED bytes — outer object {a: [{c: 9}, "x"], b: 4.5} — decode
/// through all three container levels (`Variants.object` / `Variants.array` / inner
/// `Variants.object`, names from `Variants.metadata("a", "b", "c")`).
#[test]
fn test_java_fixture_nested_object_decodes() {
    let metadata = VariantMetadata::parse(&hex(JAVA_METADATA_ABC)).expect("abc metadata");
    let value = VariantValue::parse(
        &metadata,
        &hex("02020001000e17030200070902010200020c0905781c0000000000001240"),
    )
    .expect("nested fixture must parse");
    let outer = value.as_object().expect("outer object");
    assert_eq!(outer.num_fields(), 2);
    assert_eq!(
        outer.get("b"),
        Some(&VariantValue::Primitive(VariantPrimitive::Double(4.5)))
    );
    let array = outer.get("a").expect("field a").as_array().expect("array");
    assert_eq!(array.num_elements(), 2);
    let inner = array
        .get(0)
        .expect("element 0")
        .as_object()
        .expect("inner object");
    assert_eq!(
        inner.get("c"),
        Some(&VariantValue::Primitive(VariantPrimitive::Int8(9)))
    );
    assert_eq!(
        array.get(1),
        Some(&VariantValue::Primitive(VariantPrimitive::String(
            "x".to_string()
        )))
    );
}
