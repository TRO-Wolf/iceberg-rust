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

//! Tests for Iceberg value types

use apache_avro::to_value;
use apache_avro::types::Value;
use ordered_float::OrderedFloat;
use serde_bytes::ByteBuf;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use super::decimal_utils::{decimal_from_i128_with_scale, decimal_new};
use crate::ErrorKind;
use crate::avro::schema_to_avro_schema;
use crate::spec::Schema;
use crate::spec::Type::Primitive;
use crate::spec::datatypes::{ListType, MapType, NestedField, PrimitiveType, StructType, Type};
use crate::spec::values::datum::{INT_MAX, INT_MIN};
use crate::spec::values::serde::_serde;
use crate::spec::values::{Datum, Literal, Map, PrimitiveLiteral, RawLiteral, Struct};

fn check_json_serde(json: &str, expected_literal: Literal, expected_type: &Type) {
    let raw_json_value = serde_json::from_str::<JsonValue>(json).unwrap();
    let desered_literal = Literal::try_from_json(raw_json_value.clone(), expected_type).unwrap();
    assert_eq!(desered_literal, Some(expected_literal.clone()));

    let expected_json_value: JsonValue = expected_literal.try_into_json(expected_type).unwrap();
    let sered_json = serde_json::to_string(&expected_json_value).unwrap();
    let parsed_json_value = serde_json::from_str::<JsonValue>(&sered_json).unwrap();

    assert_eq!(parsed_json_value, raw_json_value);
}

fn check_avro_bytes_serde(input: Vec<u8>, expected_datum: Datum, expected_type: &PrimitiveType) {
    let raw_schema = r#""bytes""#;
    let schema = apache_avro::Schema::parse_str(raw_schema).unwrap();

    let bytes = ByteBuf::from(input);
    let datum = Datum::try_from_bytes(&bytes, expected_type.clone()).unwrap();
    assert_eq!(datum, expected_datum);

    let mut writer = apache_avro::Writer::new(&schema, Vec::new());
    writer.append_ser(datum.to_bytes().unwrap()).unwrap();
    let encoded = writer.into_inner().unwrap();
    let reader = apache_avro::Reader::with_schema(&schema, &*encoded).unwrap();

    for record in reader {
        let result = apache_avro::from_value::<ByteBuf>(&record.unwrap()).unwrap();
        let desered_datum = Datum::try_from_bytes(&result, expected_type.clone()).unwrap();
        assert_eq!(desered_datum, expected_datum);
    }
}

fn check_convert_with_avro(expected_literal: Literal, expected_type: &Type) {
    let fields = vec![NestedField::required(1, "col", expected_type.clone()).into()];
    let schema = Schema::builder()
        .with_fields(fields.clone())
        .build()
        .unwrap();
    let avro_schema = schema_to_avro_schema("test", &schema).unwrap();
    let struct_type = Type::Struct(StructType::new(fields));
    let struct_literal = Literal::Struct(Struct::from_iter(vec![Some(expected_literal.clone())]));

    let mut writer = apache_avro::Writer::new(&avro_schema, Vec::new());
    let raw_literal = RawLiteral::try_from(struct_literal.clone(), &struct_type).unwrap();
    writer.append_ser(raw_literal).unwrap();
    let encoded = writer.into_inner().unwrap();

    let reader = apache_avro::Reader::new(&*encoded).unwrap();
    for record in reader {
        let result = apache_avro::from_value::<RawLiteral>(&record.unwrap()).unwrap();
        let desered_literal = result.try_into(&struct_type).unwrap().unwrap();
        assert_eq!(desered_literal, struct_literal);
    }
}

fn check_serialize_avro(literal: Literal, ty: &Type, expect_value: Value) {
    let expect_value = Value::Record(vec![("col".to_string(), expect_value)]);

    let fields = vec![NestedField::required(1, "col", ty.clone()).into()];
    let schema = Schema::builder()
        .with_fields(fields.clone())
        .build()
        .unwrap();
    let avro_schema = schema_to_avro_schema("test", &schema).unwrap();
    let struct_type = Type::Struct(StructType::new(fields));
    let struct_literal = Literal::Struct(Struct::from_iter(vec![Some(literal.clone())]));
    let mut writer = apache_avro::Writer::new(&avro_schema, Vec::new());
    let raw_literal = RawLiteral::try_from(struct_literal.clone(), &struct_type).unwrap();
    let value = to_value(raw_literal)
        .unwrap()
        .resolve(&avro_schema)
        .unwrap();
    writer.append_value_ref(&value).unwrap();
    let encoded = writer.into_inner().unwrap();

    let reader = apache_avro::Reader::new(&*encoded).unwrap();
    for record in reader {
        assert_eq!(record.unwrap(), expect_value);
    }
}

#[test]
fn json_boolean() {
    let record = r#"true"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Boolean(true)),
        &Type::Primitive(PrimitiveType::Boolean),
    );
}

#[test]
fn json_int() {
    let record = r#"32"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Int(32)),
        &Type::Primitive(PrimitiveType::Int),
    );
}

#[test]
fn json_long() {
    let record = r#"32"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Long(32)),
        &Type::Primitive(PrimitiveType::Long),
    );
}

#[test]
fn json_float() {
    let record = r#"1.0"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Float(OrderedFloat(1.0))),
        &Type::Primitive(PrimitiveType::Float),
    );
}

#[test]
fn json_double() {
    let record = r#"1.0"#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Double(OrderedFloat(1.0))),
        &Type::Primitive(PrimitiveType::Double),
    );
}

#[test]
fn json_date() {
    let record = r#""2017-11-16""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Int(17486)),
        &Type::Primitive(PrimitiveType::Date),
    );
}

#[test]
fn json_time() {
    let record = r#""22:31:08.123456""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Long(81068123456)),
        &Type::Primitive(PrimitiveType::Time),
    );
}

#[test]
fn json_timestamp() {
    let record = r#""2017-11-16T22:31:08.123456""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Long(1510871468123456)),
        &Type::Primitive(PrimitiveType::Timestamp),
    );
}

#[test]
fn json_timestamptz() {
    let record = r#""2017-11-16T22:31:08.123456+00:00""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Long(1510871468123456)),
        &Type::Primitive(PrimitiveType::Timestamptz),
    );
}

#[test]
fn json_string() {
    let record = r#""iceberg""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::String("iceberg".to_string())),
        &Type::Primitive(PrimitiveType::String),
    );
}

#[test]
fn json_uuid() {
    let record = r#""f79c3e09-677c-4bbd-a479-3f349cb785e7""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::UInt128(
            Uuid::parse_str("f79c3e09-677c-4bbd-a479-3f349cb785e7")
                .unwrap()
                .as_u128(),
        )),
        &Type::Primitive(PrimitiveType::Uuid),
    );
}

#[test]
fn json_decimal() {
    let record = r#""14.20""#;

    check_json_serde(
        record,
        Literal::Primitive(PrimitiveLiteral::Int128(1420)),
        &Type::decimal(28, 2).unwrap(),
    );
}

#[test]
fn json_struct() {
    let record = r#"{"1": 1, "2": "bar", "3": null}"#;

    check_json_serde(
        record,
        Literal::Struct(Struct::from_iter(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::String(
                "bar".to_string(),
            ))),
            None,
        ])),
        &Type::Struct(StructType::new(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "address", Type::Primitive(PrimitiveType::String)).into(),
        ])),
    );
}

#[test]
fn json_list() {
    let record = r#"[1, 2, 3, null]"#;

    check_json_serde(
        record,
        Literal::List(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(3))),
            None,
        ]),
        &Type::List(ListType {
            element_field: NestedField::list_element(0, Type::Primitive(PrimitiveType::Int), true)
                .into(),
        }),
    );
}

#[test]
fn json_map() {
    let record = r#"{ "keys": ["a", "b", "c"], "values": [1, 2, null] }"#;

    check_json_serde(
        record,
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::String("a".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("b".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("c".to_string())),
                None,
            ),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(0, Type::Primitive(PrimitiveType::String))
                .into(),
            value_field: NestedField::map_value_element(
                1,
                Type::Primitive(PrimitiveType::Int),
                true,
            )
            .into(),
        }),
    );
}

/// Risk: `try_from_json` for fixed and binary was `todo!()` and panicked on Java-written
/// metadata. The fixtures are Java's own, and their lowercase form pins case-insensitive accept.
/// The mutation this discriminates: restore an unimplemented or error arm.
#[test]
fn json_fixed_binary_from_json_java_fixtures() {
    let fixed = Literal::try_from_json(
        JsonValue::String("111f".to_string()),
        &Type::Primitive(PrimitiveType::Fixed(2)),
    )
    .expect("Java fixed[2] fixture \"111f\" must parse")
    .expect("non-null");
    assert_eq!(fixed, Literal::fixed(vec![0x11u8, 0x1f]));

    let binary = Literal::try_from_json(
        JsonValue::String("0000ff".to_string()),
        &Type::Primitive(PrimitiveType::Binary),
    )
    .expect("Java binary fixture \"0000ff\" must parse")
    .expect("non-null");
    assert_eq!(binary, Literal::binary(vec![0x00u8, 0x00, 0xff]));

    // All three spellings decode to the same bytes.
    for spelling in ["a1b2", "A1B2", "a1B2"] {
        let lit = Literal::try_from_json(
            JsonValue::String(spelling.to_string()),
            &Type::Primitive(PrimitiveType::Binary),
        )
        .expect("mixed-case hex must parse (Java uppercases before decoding)")
        .expect("non-null");
        assert_eq!(lit, Literal::binary(vec![0xa1u8, 0xb2]), "{spelling}");
    }
}

/// The realistic Java-written-metadata entry path: a schema JSON whose fields carry fixed and
/// binary defaults, exactly as Java's `SchemaParser` and `SingleValueParser.toJson` emit them in
/// uppercase base16. Deserializing this document used to panic through the `todo!()` arms.
#[test]
fn json_schema_with_fixed_and_binary_defaults_from_java_metadata() {
    let java_schema_json = r#"
    {
        "type": "struct",
        "schema-id": 0,
        "fields": [
            {
                "id": 1,
                "name": "bin_col",
                "required": true,
                "type": "binary",
                "initial-default": "000102FF",
                "write-default": "0A"
            },
            {
                "id": 2,
                "name": "fixed_col",
                "required": true,
                "type": "fixed[2]",
                "initial-default": "111F",
                "write-default": "0BAD"
            }
        ]
    }"#;
    let schema: Schema = serde_json::from_str(java_schema_json)
        .expect("a Java-written schema with fixed/binary defaults must deserialize");

    let bin_field = schema.field_by_id(1).expect("bin_col present");
    assert_eq!(
        bin_field.initial_default,
        Some(Literal::binary(vec![0x00u8, 0x01, 0x02, 0xff]))
    );
    assert_eq!(bin_field.write_default, Some(Literal::binary(vec![0x0au8])));

    let fixed_field = schema.field_by_id(2).expect("fixed_col present");
    assert_eq!(
        fixed_field.initial_default,
        Some(Literal::fixed(vec![0x11u8, 0x1f]))
    );
    assert_eq!(
        fixed_field.write_default,
        Some(Literal::fixed(vec![0x0bu8, 0xad]))
    );
}

/// Risk: `try_into_json` emitted lowercase unpadded hex, so byte 0x0A became `"a"`, which Java's
/// strict decoder cannot parse. The strings below are byte-for-byte Java's uppercase two-digit
/// output. The mutations this discriminates: flip the emit case, or drop the zero padding.
#[test]
fn json_binary_fixed_emit_uppercase_padded_java_compatible() {
    let json = Literal::binary(vec![0x00u8, 0x0a, 0x1b, 0xff])
        .try_into_json(&Type::Primitive(PrimitiveType::Binary))
        .expect("binary emit");
    assert_eq!(json, JsonValue::String("000A1BFF".to_string()));

    let json = Literal::fixed(vec![0x0bu8, 0xad])
        .try_into_json(&Type::Primitive(PrimitiveType::Fixed(2)))
        .expect("fixed emit");
    assert_eq!(json, JsonValue::String("0BAD".to_string()));

    // Empty binary is legal and emits the empty string, as Java does.
    let json = Literal::binary(vec![])
        .try_into_json(&Type::Primitive(PrimitiveType::Binary))
        .expect("empty binary emit");
    assert_eq!(json, JsonValue::String(String::new()));
}

/// Risk: `parse(emit(x))` must be byte-exact for binary and fixed. It covers bytes below 0x10,
/// which the old unpadded emitter corrupted, and the 0x00 and 0xFF extremes.
#[test]
fn json_binary_fixed_round_trip_byte_exact() {
    let bytes = vec![0x00u8, 0x01, 0x0a, 0x10, 0x7f, 0x80, 0xf0, 0xff];

    let binary_type = Type::Primitive(PrimitiveType::Binary);
    let json = Literal::binary(bytes.clone())
        .try_into_json(&binary_type)
        .expect("binary emit");
    let back = Literal::try_from_json(json, &binary_type)
        .expect("binary re-parse")
        .expect("non-null");
    assert_eq!(back, Literal::binary(bytes.clone()));

    let fixed_type = Type::Primitive(PrimitiveType::Fixed(8));
    let json = Literal::fixed(bytes.clone())
        .try_into_json(&fixed_type)
        .expect("fixed emit");
    let back = Literal::try_from_json(json, &fixed_type)
        .expect("fixed re-parse")
        .expect("non-null");
    assert_eq!(back, Literal::fixed(bytes));
}

/// Risk: fixed length must be enforced both ways. Java checks `2 * L` on parse and `L` on emit.
/// The mutation this discriminates: drop either check. Over-broadening also reds, because the
/// round-trip tests pin the legal boundary case.
#[test]
fn json_fixed_length_mismatch_is_data_invalid() {
    // Java's own invalid fixture: 5 hex chars on fixed[2], both odd and the wrong length.
    let err = Literal::try_from_json(
        JsonValue::String("111ff".to_string()),
        &Type::Primitive(PrimitiveType::Fixed(2)),
    )
    .expect_err("fixed[2] must reject a 5-char hex string");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);

    // Even-length valid hex, but 3 bytes against fixed[2]. Only the length check rejects it.
    let err = Literal::try_from_json(
        JsonValue::String("1122FF".to_string()),
        &Type::Primitive(PrimitiveType::Fixed(2)),
    )
    .expect_err("fixed[2] must reject a 3-byte value");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);

    // The emit side: a 3-byte literal against fixed[2].
    let err = Literal::fixed(vec![0x11u8, 0x22, 0xff])
        .try_into_json(&Type::Primitive(PrimitiveType::Fixed(2)))
        .expect_err("fixed[2] must refuse to emit a 3-byte value");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
}

/// Risk: odd-length hex, non-hex ASCII, and non-ASCII input must all be `DataInvalid`, never a
/// panic and never accepted. Java's strict `base16().decode` throws on each.
#[test]
fn json_binary_fixed_malformed_hex_is_data_invalid() {
    let binary_type = Type::Primitive(PrimitiveType::Binary);
    for bad in ["abc", "zz", "0g", "€€", "0x0A", " 0A"] {
        let err = Literal::try_from_json(JsonValue::String(bad.to_string()), &binary_type)
            .expect_err("malformed hex must fail closed");
        assert_eq!(err.kind(), ErrorKind::DataInvalid, "input: {bad:?}");
    }

    // The fixed door too: the right string length, but non-hex content.
    let err = Literal::try_from_json(
        JsonValue::String("zz".to_string()),
        &Type::Primitive(PrimitiveType::Fixed(1)),
    )
    .expect_err("fixed[1] must reject non-hex content of the right length");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);

    // A non-string JSON value for binary is a type mismatch, not a panic.
    let err = Literal::try_from_json(JsonValue::Number(7.into()), &binary_type)
        .expect_err("a JSON number is not a binary single value");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
}

#[test]
fn avro_bytes_boolean() {
    let bytes = vec![1u8];

    check_avro_bytes_serde(bytes, Datum::bool(true), &PrimitiveType::Boolean);
}

#[test]
fn avro_bytes_int() {
    let bytes = vec![32u8, 0u8, 0u8, 0u8];

    check_avro_bytes_serde(bytes, Datum::int(32), &PrimitiveType::Int);
}

#[test]
fn avro_bytes_long() {
    let bytes = vec![32u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];

    check_avro_bytes_serde(bytes, Datum::long(32), &PrimitiveType::Long);
}

#[test]
fn avro_bytes_long_from_int() {
    let bytes = vec![32u8, 0u8, 0u8, 0u8];

    check_avro_bytes_serde(bytes, Datum::long(32), &PrimitiveType::Long);
}

#[test]
fn avro_bytes_float() {
    let bytes = vec![0u8, 0u8, 128u8, 63u8];

    check_avro_bytes_serde(bytes, Datum::float(1.0), &PrimitiveType::Float);
}

#[test]
fn avro_bytes_double() {
    let bytes = vec![0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 240u8, 63u8];

    check_avro_bytes_serde(bytes, Datum::double(1.0), &PrimitiveType::Double);
}

#[test]
fn avro_bytes_double_from_float() {
    let bytes = vec![0u8, 0u8, 128u8, 63u8];

    check_avro_bytes_serde(bytes, Datum::double(1.0), &PrimitiveType::Double);
}

#[test]
fn avro_bytes_string() {
    let bytes = vec![105u8, 99u8, 101u8, 98u8, 101u8, 114u8, 103u8];

    check_avro_bytes_serde(bytes, Datum::string("iceberg"), &PrimitiveType::String);
}

#[test]
fn avro_bytes_decimal() {
    // (input_bytes, decimal_num, expect_scale, expect_precision)
    let cases = vec![
        (vec![4u8, 210u8], 1234, 2, 38),
        (vec![251u8, 46u8], -1234, 2, 38),
        (vec![4u8, 210u8], 1234, 3, 38),
        (vec![251u8, 46u8], -1234, 3, 38),
        (vec![42u8], 42, 2, 2),
        (vec![214u8], -42, 2, 2),
    ];

    for (input_bytes, decimal_num, expect_scale, expect_precision) in cases {
        check_avro_bytes_serde(
            input_bytes,
            Datum::decimal_with_precision(decimal_new(decimal_num, expect_scale), expect_precision)
                .unwrap(),
            &PrimitiveType::Decimal {
                precision: expect_precision,
                scale: expect_scale,
            },
        );
    }
}

#[test]
fn avro_bytes_decimal_expect_error() {
    // (decimal_num, expect_scale, expect_precision)
    let cases = vec![(1234, 0, 1), (42, 2, 1)];

    for (decimal_num, expect_scale, expect_precision) in cases {
        let result =
            Datum::decimal_with_precision(decimal_new(decimal_num, expect_scale), expect_precision);
        assert!(result.is_err(), "expect error but got {result:?}");
        assert_eq!(
            result.unwrap_err().kind(),
            ErrorKind::DataInvalid,
            "expect error DataInvalid",
        );
    }
}

#[test]
fn datum_decimal_precision_counts_digits_not_bytes() {
    for (value, encoded) in [(99, vec![0x63]), (-99, vec![0x9d])] {
        let datum = Datum::decimal_with_precision(decimal_new(value, 0), 2)
            .expect("two-digit decimal must fit precision two");
        assert_eq!(datum.to_bytes().expect("valid decimal bytes"), encoded);
    }

    for value in [100, -100] {
        let result = Datum::decimal_with_precision(decimal_new(value, 0), 2);
        assert_eq!(
            result
                .expect_err("three-digit decimal must exceed precision two")
                .kind(),
            ErrorKind::DataInvalid
        );
    }
}

#[test]
fn datum_decimal_boundaries_preserve_negative_binary_encoding() {
    for (value, scale, encoded) in [(99, 2, vec![0x63]), (-99, 2, vec![0x9d])] {
        let datum = Datum::decimal_with_precision(decimal_new(value, scale), 2)
            .expect("decimal precision and scale boundary must be valid");
        assert_eq!(datum.to_bytes().expect("valid decimal bytes"), encoded);
    }
}

/// [`Datum::try_from_bytes`] is a metadata READ door. It must accept every decimal type Java can
/// build and refuse only `of(39,0)`, which Java itself refuses.
///
/// The mutations this discriminates: add `Type::decimal` or `validate_decimal_literal` to the
/// decimal arm, which made `decimal(1,2)` unreadable; or delete `ensure_java_decimal_precision`.
#[test]
fn datum_decimal_bytes_accept_java_legal_metadata_and_reject_precision_over_38() {
    for (precision, scale) in [(1, 2), (0, 0), (10, 11), (38, 38), (2, 0)] {
        let data_type = PrimitiveType::Decimal { precision, scale };
        let datum = Datum::try_from_bytes(&[0], data_type.clone()).unwrap_or_else(|error| {
            panic!("Java builds decimal({precision},{scale}), so we must read it: {error}")
        });
        assert_eq!(datum, Datum::new(data_type, PrimitiveLiteral::Int128(0)));
    }

    let error = Datum::try_from_bytes(&[0], PrimitiveType::Decimal {
        precision: 39,
        scale: 0,
    })
    .expect_err("Java's DecimalType constructor refuses precision 39, so we must too");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
}

/// A zero-padded or sign-padded decimal bound decodes, as it does in Java. Java's DECIMAL arm is
/// a bare `new BigInteger`, and it rejects only an empty buffer. The LONG and DOUBLE arms in the
/// same switch DO branch on `remaining()`, which proves the omission is deliberate.
///
/// It matters beyond one value: `parse_bytes_entry` propagates with `?`, so one padded bound
/// makes the whole manifest unparsable and aborts every scan.
///
/// The mutations this discriminates: restore the `i128_to_be_bytes_min` canonical check in
/// `Datum::try_from_bytes`, or drop the empty-buffer guard beside it.
/// Live-probe transcript: [`task/spec-decode-evidence.md`].
#[test]
fn datum_decimal_byte_decode_accepts_non_minimal_encodings_like_java() {
    let decimal_9_2 = PrimitiveType::Decimal {
        precision: 9,
        scale: 2,
    };
    let decimal_2_0 = PrimitiveType::Decimal {
        precision: 2,
        scale: 0,
    };

    let mut padded_20 = vec![0x00; 20];
    padded_20[18] = 0x04;
    padded_20[19] = 0xd2;
    let mut padded_20_negative = vec![0xffu8; 20];
    padded_20_negative[18] = 0xfb;
    padded_20_negative[19] = 0x2e;

    for (data_type, bytes, expected) in [
        // Java decodes this to 12.34.
        (decimal_9_2.clone(), vec![0x00, 0x00, 0x04, 0xd2], 1234),
        (decimal_9_2.clone(), vec![0xff, 0xff, 0xfb, 0x2e], -1234),
        (decimal_9_2.clone(), padded_20, 1234),
        (decimal_9_2.clone(), padded_20_negative, -1234),
        // Minimal encodings keep working.
        (decimal_9_2, vec![0x04, 0xd2], 1234),
        (decimal_2_0.clone(), vec![0x00], 0),
        (decimal_2_0.clone(), vec![0x63], 99),
        (decimal_2_0.clone(), vec![0x9d], -99),
        // Redundant sign extension at every width.
        (decimal_2_0.clone(), vec![0x00, 0x00], 0),
        (decimal_2_0.clone(), vec![0x00, 0x63], 99),
        (decimal_2_0.clone(), vec![0xff, 0x9d], -99),
        (decimal_2_0.clone(), vec![0xff, 0xff], -1),
    ] {
        let datum = Datum::try_from_bytes(&bytes, data_type.clone()).unwrap_or_else(|error| {
            panic!("Java decodes {bytes:?} as {data_type}, so we must too: {error}")
        });
        assert_eq!(
            datum,
            Datum::new(data_type, PrimitiveLiteral::Int128(expected)),
            "bytes={bytes:?}"
        );
    }

    // The one byte-level input Java rejects: an empty buffer throws in `new BigInteger`.
    let error = Datum::try_from_bytes(&[], decimal_2_0.clone())
        .expect_err("Java throws on a zero-length decimal buffer, so we must reject it");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.message().contains("Zero length BigInteger"),
        "the empty-buffer diagnostic should name Java's own failure: {error}"
    );

    // A magnitude that genuinely exceeds i128, not mere sign padding, is still rejected.
    let mut too_large = vec![0x00; 17];
    too_large[1] = 0x80;
    let error = Datum::try_from_bytes(&too_large, decimal_2_0)
        .expect_err("2^127 is outside i128 and must not wrap");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
}

/// Java does not compare a decimal's magnitude against its precision on READ. It decodes
/// `999999` at `decimal(2,0)`. The fork gates only where the encoder would truncate. The mutation
/// this discriminates: add `validate_decimal_literal` after the match in `try_from_bytes`.
#[test]
fn datum_decimal_byte_decode_accepts_values_wider_than_declared_precision() {
    let decimal_2_0 = PrimitiveType::Decimal {
        precision: 2,
        scale: 0,
    };
    for (bytes, expected) in [
        (vec![0x0f, 0x42, 0x3f], 999_999_i128),
        (vec![0x64], 100),
        (vec![0x9c], -100),
    ] {
        let datum = Datum::try_from_bytes(&bytes, decimal_2_0.clone()).unwrap_or_else(|error| {
            panic!("Java decodes {bytes:?} as decimal(2,0) without a precision check: {error}")
        });
        assert_eq!(
            datum,
            Datum::new(decimal_2_0.clone(), PrimitiveLiteral::Int128(expected))
        );
    }

    let datum = Datum::try_from_bytes(&i128::MIN.to_be_bytes(), PrimitiveType::Decimal {
        precision: 38,
        scale: 0,
    })
    .expect("Java decodes a 39-digit BigInteger for a decimal(38,0) bound");
    assert_eq!(
        datum,
        Datum::new(
            PrimitiveType::Decimal {
                precision: 38,
                scale: 0,
            },
            PrimitiveLiteral::Int128(i128::MIN),
        )
    );
}

/// A `Datum` from the Java-permissive read path must survive its own serde round trip. Scan
/// tasks carry `Datum` bounds across process boundaries, so a precision gate in `Serialize` would
/// let a Java-written table plan but fail to distribute.
///
/// Both `DatumVisitor` arms run here: `visit_map` for self-describing formats, `visit_seq` for
/// the compact formats that carry a scan task. The mutation this discriminates: add
/// `validate_decimal_literal` to `impl Serialize for Datum` or to either arm.
#[test]
fn datum_decimal_serde_round_trip_preserves_java_readable_values() {
    let decimal_type = PrimitiveType::Decimal {
        precision: 2,
        scale: 0,
    };

    for value in [99_i128, -99, 100, -100, 999_999] {
        let datum = Datum::new(decimal_type.clone(), PrimitiveLiteral::Int128(value));
        let json = serde_json::to_value(&datum)
            .unwrap_or_else(|error| panic!("decimal(2,0) datum {value} must serialize: {error}"));
        assert_eq!(
            json,
            serde_json::json!({
                "type": "decimal(2,0)",
                "literal": value.to_be_bytes(),
            })
        );
        assert_eq!(
            serde_json::from_value::<Datum>(json).unwrap_or_else(|error| panic!(
                "decimal(2,0) datum {value} must deserialize: {error}"
            )),
            datum
        );
    }

    // The sequence route must be as permissive as the map route. A JSON array drives
    // `DatumVisitor::visit_seq` directly.
    for value in [99_i128, -99, 100, -100, 999_999] {
        let seq = serde_json::json!(["decimal(2,0)", value.to_be_bytes()]);
        assert_eq!(
            serde_json::from_value::<Datum>(seq).unwrap_or_else(|error| panic!(
                "Java-readable decimal(2,0) bound {value} must survive the seq route: {error}"
            )),
            Datum::new(decimal_type.clone(), PrimitiveLiteral::Int128(value))
        );
    }

    // Java-legal metadata the strict constructor refuses still round-trips as data.
    let odd_type = PrimitiveType::Decimal {
        precision: 10,
        scale: 11,
    };
    let datum = Datum::new(odd_type.clone(), PrimitiveLiteral::Int128(7));
    let json = serde_json::to_value(&datum).expect("decimal(10,11) is Java-legal metadata");
    assert_eq!(json["type"], serde_json::json!("decimal(10,11)"));
    assert_eq!(
        serde_json::from_value::<Datum>(json).expect("decimal(10,11) must deserialize"),
        datum
    );
    assert_eq!(
        serde_json::from_value::<Datum>(serde_json::json!([
            "decimal(10,11)",
            7_i128.to_be_bytes()
        ]))
        .expect("decimal(10,11) must deserialize through the seq route"),
        datum
    );
}

/// The write path keeps its gate, because `Datum::to_bytes` truncates the two's-complement
/// buffer and silently corrupts a wider value. Java has no such check, but Java also never
/// truncates. The mutation this discriminates: delete `validate_decimal_literal` from
/// `Datum::to_bytes`.
#[test]
fn datum_decimal_write_path_still_rejects_values_wider_than_precision() {
    let decimal_2_0 = PrimitiveType::Decimal {
        precision: 2,
        scale: 0,
    };
    for (value, encoded) in [(99_i128, vec![0x63]), (-99, vec![0x9d])] {
        let datum = Datum::new(decimal_2_0.clone(), PrimitiveLiteral::Int128(value));
        assert_eq!(
            datum.to_bytes().expect("in-precision value encodes"),
            encoded
        );
    }
    for value in [100_i128, -100, 999_999] {
        let datum = Datum::new(decimal_2_0.clone(), PrimitiveLiteral::Int128(value));
        assert_eq!(
            datum
                .to_bytes()
                .expect_err("a value the encoder would truncate must be refused")
                .kind(),
            ErrorKind::DataInvalid
        );
    }

    // `unsigned_abs` keeps the 39-digit `i128::MIN` check overflow-free.
    let over_max = Datum::new(
        PrimitiveType::Decimal {
            precision: 38,
            scale: 0,
        },
        PrimitiveLiteral::Int128(i128::MIN),
    );
    assert_eq!(
        over_max
            .to_bytes()
            .expect_err("i128::MIN has 39 digits and cannot encode at precision 38")
            .kind(),
        ErrorKind::DataInvalid
    );
}

#[test]
fn datum_decimal_serialization_rejects_invalid_metadata() {
    // `precision > 38` is the one decimal-metadata rule Java enforces, so both encoders refuse.
    let over_max = Datum::new(
        PrimitiveType::Decimal {
            precision: 39,
            scale: 0,
        },
        PrimitiveLiteral::Int128(-1),
    );
    let error = over_max
        .to_bytes()
        .expect_err("precision 39 is not encodable");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "PrimitiveType Decimal must has valid precision but got 39",
        "Datum boundary must retain its compatibility diagnostic: {error}"
    );
    let error = serde_json::to_string(&over_max).expect_err("precision 39 must not serialize");
    assert!(error.to_string().contains("Decimals with precision larger"));

    // `precision = 0` has no byte width, so the encoder refuses even though Java's
    // `DecimalType.of(0,0)` constructs. The type string itself stays readable and writable.
    let zero_precision = Datum::new(
        PrimitiveType::Decimal {
            precision: 0,
            scale: 0,
        },
        PrimitiveLiteral::Int128(0),
    );
    assert_eq!(
        zero_precision
            .to_bytes()
            .expect_err("precision 0 has no decimal byte width")
            .kind(),
        ErrorKind::DataInvalid
    );

    // `scale > precision` is not a Java invariant, so a value that fits the precision encodes.
    let odd_shape = Datum::new(
        PrimitiveType::Decimal {
            precision: 1,
            scale: 2,
        },
        PrimitiveLiteral::Int128(-1),
    );
    assert_eq!(
        odd_shape
            .to_bytes()
            .expect("decimal(1,2) is Java-legal and -1 fits precision 1"),
        vec![0xffu8]
    );
    assert!(
        serde_json::to_string(&odd_shape).is_ok(),
        "decimal(1,2) is Java-legal metadata and must serialize"
    );
}

#[test]
fn datum_json_byte_lists_reject_out_of_range_elements_in_map_and_sequence_forms() {
    for bad_byte in [-1_i64, 256, 257] {
        let type_and_bytes = [
            (serde_json::json!("binary"), vec![bad_byte]),
            (serde_json::json!("fixed[1]"), vec![bad_byte]),
            (
                serde_json::json!("uuid"),
                std::iter::once(bad_byte)
                    .chain(std::iter::repeat_n(0, 15))
                    .collect(),
            ),
            (
                serde_json::json!("decimal(38,0)"),
                std::iter::repeat_n(0, 15)
                    .chain(std::iter::once(bad_byte))
                    .collect(),
            ),
        ];

        for (data_type, bytes) in type_and_bytes {
            for datum_json in [
                serde_json::json!({"type": data_type.clone(), "literal": bytes.clone()}),
                serde_json::json!([data_type, bytes]),
            ] {
                let error = serde_json::from_value::<Datum>(datum_json)
                    .expect_err("JSON byte values outside u8 must not truncate");
                let message = error.to_string();
                assert!(
                    message.contains("DataInvalid") && message.contains(&bad_byte.to_string()),
                    "bad byte {bad_byte} must produce a typed, value-bearing error: {message}"
                );
            }
        }
    }
}

fn check_raw_literal_bytes_serde_via_avro(
    input_bytes: Vec<u8>,
    expected_literal: Literal,
    expected_type: &Type,
) {
    use apache_avro::types::Value;

    let avro_value = Value::Bytes(input_bytes);
    let raw_literal: _serde::RawLiteral = apache_avro::from_value(&avro_value).unwrap();
    let result = raw_literal.try_into(expected_type).unwrap();
    assert_eq!(result, Some(expected_literal));
}

fn check_raw_literal_bytes_error_via_avro(input_bytes: Vec<u8>, expected_type: &Type) {
    use apache_avro::types::Value;

    let avro_value = Value::Bytes(input_bytes);
    let raw_literal: _serde::RawLiteral = apache_avro::from_value(&avro_value).unwrap();
    let result = raw_literal.try_into(expected_type);
    assert!(result.is_err(), "Expected error but got: {result:?}");
}

#[test]
fn test_raw_literal_bytes_binary() {
    let bytes = vec![1u8, 2u8, 3u8, 4u8, 5u8];
    check_raw_literal_bytes_serde_via_avro(
        bytes.clone(),
        Literal::binary(bytes),
        &Type::Primitive(PrimitiveType::Binary),
    );
}

#[test]
fn test_raw_literal_bytes_binary_empty() {
    let bytes = vec![];
    check_raw_literal_bytes_serde_via_avro(
        bytes.clone(),
        Literal::binary(bytes),
        &Type::Primitive(PrimitiveType::Binary),
    );
}

#[test]
fn test_raw_literal_bytes_fixed_correct_length() {
    let bytes = vec![1u8, 2u8, 3u8, 4u8];
    check_raw_literal_bytes_serde_via_avro(
        bytes.clone(),
        Literal::fixed(bytes),
        &Type::Primitive(PrimitiveType::Fixed(4)),
    );
}

#[test]
fn test_raw_literal_bytes_fixed_wrong_length() {
    let bytes = vec![1u8, 2u8, 3u8]; // 3 bytes, but expecting 4
    check_raw_literal_bytes_error_via_avro(bytes, &Type::Primitive(PrimitiveType::Fixed(4)));
}

#[test]
fn test_raw_literal_bytes_fixed_empty_correct_length() {
    let bytes = vec![];
    check_raw_literal_bytes_serde_via_avro(
        bytes.clone(),
        Literal::fixed(bytes),
        &Type::Primitive(PrimitiveType::Fixed(0)),
    );
}

#[test]
fn test_raw_literal_bytes_uuid_correct_length() {
    let uuid_bytes = vec![
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
        0xef,
    ];
    let expected_uuid = u128::from_be_bytes([
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
        0xef,
    ]);
    check_raw_literal_bytes_serde_via_avro(
        uuid_bytes,
        Literal::Primitive(PrimitiveLiteral::UInt128(expected_uuid)),
        &Type::Primitive(PrimitiveType::Uuid),
    );
}

#[test]
fn test_raw_literal_bytes_uuid_wrong_length() {
    let bytes = vec![1u8, 2u8, 3u8]; // 3 bytes, but UUID needs 16
    check_raw_literal_bytes_error_via_avro(bytes, &Type::Primitive(PrimitiveType::Uuid));
}

#[test]
fn test_raw_literal_bytes_decimal_precision_4_scale_2() {
    let decimal_bytes = vec![0x04, 0xd2]; // 1234 in 2 bytes
    let expected_decimal = 1234i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 4,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_4_negative() {
    let decimal_bytes = vec![0xfb, 0x2e]; // -1234 in 2 bytes
    let expected_decimal = -1234i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 4,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_9_scale_2() {
    let decimal_bytes = vec![0x00, 0x12, 0xd6, 0x87]; // 1234567 in 4 bytes
    let expected_decimal = 1234567i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 9,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_18_scale_2() {
    let decimal_bytes = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xd2]; // 1234 in 8 bytes
    let expected_decimal = 1234i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 18,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_38_scale_2() {
    let decimal_bytes = vec![
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,
        0xd2, // 1234 in 16 bytes
    ];
    let expected_decimal = 1234i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 38,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_1_scale_0() {
    let decimal_bytes = vec![0x07]; // 7 in 1 byte
    let expected_decimal = 7i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 1,
            scale: 0,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_precision_1_negative() {
    let decimal_bytes = vec![0xf9]; // -7 in 1 byte (two's complement)
    let expected_decimal = -7i128;
    check_raw_literal_bytes_serde_via_avro(
        decimal_bytes,
        Literal::Primitive(PrimitiveLiteral::Int128(expected_decimal)),
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 1,
            scale: 0,
        }),
    );
}

/// The Avro `bytes` decimal decode is a READ door with no precision gate, as Java has none. The
/// mutation this discriminates: add `validate_decimal_value` to the `RawLiteralEnum::Bytes` arm.
#[test]
fn raw_literal_decimal_decode_accepts_values_wider_than_declared_precision() {
    let decimal_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 2,
        scale: 0,
    });

    for (bytes, expected) in [
        (vec![0x63], 99),
        (vec![0x9d], -99),
        // Java's decode of the same buffers: 100 and -100 at decimal(2,0), unchecked.
        (vec![0x64], 100),
        (vec![0x9c], -100),
    ] {
        check_raw_literal_bytes_serde_via_avro(bytes, Literal::decimal(expected), &decimal_type);
    }
}

/// The same absence of a precision gate on the recursive struct and partition route. The
/// mutation this discriminates: add `validate_decimal_value` to the `PrimitiveLiteral::Int128`
/// arm of `RawLiteral::try_from`, which makes a Java-written partition value unwritable.
#[test]
fn raw_literal_recursive_struct_route_accepts_java_written_decimals() {
    let decimal_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 2,
        scale: 0,
    });
    let struct_type = Type::Struct(StructType::new(vec![
        NestedField::required(1, "partition_decimal", decimal_type).into(),
    ]));

    for value in [99, -99, 100, -100, 999_999] {
        let literal = Literal::Struct(Struct::from_iter([Some(Literal::decimal(value))]));
        assert!(
            RawLiteral::try_from(literal, &struct_type).is_ok(),
            "manifest-routed decimal {value} is Java-legal and must be accepted"
        );
    }

    let literal = Literal::Struct(Struct::from_iter([Some(Literal::decimal(1))]));
    let wrong_type = Type::Struct(StructType::new(vec![
        NestedField::required(
            1,
            "partition_decimal",
            Type::Struct(StructType::new(vec![])),
        )
        .into(),
    ]));
    assert_eq!(
        RawLiteral::try_from(literal, &wrong_type)
            .expect_err("a decimal literal under a struct type is still invalid")
            .kind(),
        ErrorKind::DataInvalid
    );
}

#[test]
fn literal_decimal_json_boundaries_validate_direct_and_nested_values() {
    let decimal_type = Type::Primitive(PrimitiveType::Decimal {
        precision: 2,
        scale: 0,
    });

    for value in [99, -99] {
        let json = JsonValue::String(value.to_string());
        assert_eq!(
            Literal::try_from_json(json.clone(), &decimal_type).expect("valid direct decimal JSON"),
            Some(Literal::decimal(value))
        );
        assert_eq!(
            Literal::decimal(value)
                .try_into_json(&decimal_type)
                .expect("valid direct decimal literal"),
            json
        );
    }
    for value in [100, -100, 999_999] {
        let json = JsonValue::String(value.to_string());
        assert_eq!(
            Literal::try_from_json(json.clone(), &decimal_type)
                .expect("Java's SingleValueParser applies no precision check on read"),
            Some(Literal::decimal(value))
        );
        assert_eq!(
            Literal::decimal(value)
                .try_into_json(&decimal_type)
                .expect("Java's SingleValueParser applies no precision check on write"),
            json
        );
    }

    let struct_type = Type::Struct(StructType::new(vec![
        NestedField::optional(1, "decimal", decimal_type.clone()).into(),
    ]));
    let list_type = Type::List(ListType {
        element_field: NestedField::list_element(2, decimal_type.clone(), false).into(),
    });
    let map_type = Type::Map(MapType {
        key_field: NestedField::map_key_element(3, Type::Primitive(PrimitiveType::String)).into(),
        value_field: NestedField::map_value_element(4, decimal_type.clone(), false).into(),
    });

    let valid_cases = [
        (
            serde_json::json!({"1": "99"}),
            struct_type.clone(),
            Literal::Struct(Struct::from_iter([Some(Literal::decimal(99))])),
        ),
        (
            serde_json::json!(["-99"]),
            list_type.clone(),
            Literal::List(vec![Some(Literal::decimal(-99))]),
        ),
        (
            serde_json::json!({"keys": ["k"], "values": ["99"]}),
            map_type.clone(),
            Literal::Map(Map::from_iter([(
                Literal::string("k"),
                Some(Literal::decimal(99)),
            )])),
        ),
    ];
    for (json, data_type, literal) in valid_cases {
        assert_eq!(
            Literal::try_from_json(json.clone(), &data_type).expect("nested valid decimal JSON"),
            Some(literal.clone())
        );
        assert_eq!(
            literal
                .try_into_json(&data_type)
                .expect("nested valid decimal literal"),
            json
        );
    }

    // Values wider than the declared precision are Java-legal on this path too, nested or not.
    let wide_cases = [
        (
            serde_json::json!({"1": "100"}),
            struct_type.clone(),
            Literal::Struct(Struct::from_iter([Some(Literal::decimal(100))])),
        ),
        (
            serde_json::json!(["-100"]),
            list_type,
            Literal::List(vec![Some(Literal::decimal(-100))]),
        ),
        (
            serde_json::json!({"keys": ["k"], "values": ["100"]}),
            map_type,
            Literal::Map(Map::from_iter([(
                Literal::string("k"),
                Some(Literal::decimal(100)),
            )])),
        ),
    ];
    for (json, data_type, literal) in wide_cases {
        assert_eq!(
            Literal::try_from_json(json.clone(), &data_type)
                .expect("Java parses this default without a precision check"),
            Some(literal.clone())
        );
        assert_eq!(
            literal
                .try_into_json(&data_type)
                .expect("Java writes this default back without a precision check"),
            json
        );
    }

    // A malformed nested value must surface as a typed error, not vanish into a missing field.
    assert_eq!(
        Literal::try_from_json(serde_json::json!({"1": 123}), &struct_type)
            .expect_err("a non-string decimal default is not parseable")
            .kind(),
        ErrorKind::DataInvalid
    );
}

/// The `decimal(P,S)` JSON value path applies Java's rules only: `isTextual`, `new BigDecimal`,
/// and a scale match. `toJson` has no gate, so `decimal(0,0)`, `decimal(39,0)`, and
/// `decimal(2,3)` must not be refused. The mutation this discriminates: add `Type::decimal` or
/// `validate_decimal_value` to `Literal::try_from_json` or `try_into_json`.
#[test]
fn literal_decimal_json_applies_no_precision_gate_like_java() {
    for data_type in [
        Type::Primitive(PrimitiveType::Decimal {
            precision: 0,
            scale: 0,
        }),
        Type::Primitive(PrimitiveType::Decimal {
            precision: 39,
            scale: 0,
        }),
        Type::Primitive(PrimitiveType::Decimal {
            precision: 2,
            scale: 3,
        }),
    ] {
        assert_eq!(
            Literal::try_from_json(JsonValue::String("0".to_string()), &data_type)
                .expect("Java's SingleValueParser accepts this decimal default"),
            Some(Literal::decimal(0))
        );
        assert!(
            Literal::decimal(0).try_into_json(&data_type).is_ok(),
            "Java's SingleValueParser writes this decimal default back unchecked"
        );
    }

    assert_eq!(
        Literal::decimal(0)
            .try_into_json(&Type::Primitive(PrimitiveType::Int))
            .expect_err("an Int128 literal is only valid under a decimal type")
            .kind(),
        ErrorKind::DataInvalid
    );
}

#[test]
fn test_raw_literal_bytes_decimal_wrong_length() {
    let bytes = vec![1u8, 2u8, 3u8];
    check_raw_literal_bytes_error_via_avro(
        bytes,
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 4,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_decimal_wrong_length_too_few() {
    let bytes = vec![0x42];
    check_raw_literal_bytes_error_via_avro(
        bytes,
        &Type::Primitive(PrimitiveType::Decimal {
            precision: 9,
            scale: 2,
        }),
    );
}

#[test]
fn test_raw_literal_bytes_unsupported_type() {
    let bytes = vec![1u8, 2u8, 3u8, 4u8];
    check_raw_literal_bytes_error_via_avro(bytes, &Type::Primitive(PrimitiveType::Int));
}

#[test]
fn avro_convert_test_int() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Int(32)),
        &Type::Primitive(PrimitiveType::Int),
    );
}

#[test]
fn avro_convert_test_long() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(32)),
        &Type::Primitive(PrimitiveType::Long),
    );
}

#[test]
fn avro_convert_test_float() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Float(OrderedFloat(1.0))),
        &Type::Primitive(PrimitiveType::Float),
    );
}

#[test]
fn avro_convert_test_double() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Double(OrderedFloat(1.0))),
        &Type::Primitive(PrimitiveType::Double),
    );
}

#[test]
fn avro_convert_test_string() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::String("iceberg".to_string())),
        &Type::Primitive(PrimitiveType::String),
    );
}

#[test]
fn avro_convert_test_date() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Int(17486)),
        &Type::Primitive(PrimitiveType::Date),
    );
}

#[test]
fn avro_convert_test_time() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(81068123456)),
        &Type::Primitive(PrimitiveType::Time),
    );
}

#[test]
fn avro_convert_test_timestamp() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(1510871468123456)),
        &Type::Primitive(PrimitiveType::Timestamp),
    );
}

#[test]
fn avro_convert_test_timestamptz() {
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(1510871468123456)),
        &Type::Primitive(PrimitiveType::Timestamptz),
    );
}

#[test]
fn avro_convert_test_timestamp_ns() {
    // Risk: `Long -> timestamp_ns` was missing, so a V3 identity partition written as an Avro
    // long could not be reloaded.
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(1_704_083_400_000_000_000)),
        &Type::Primitive(PrimitiveType::TimestampNs),
    );
}

#[test]
fn avro_convert_test_timestamptz_ns() {
    // Risk: the twin of the arm above. `Long -> timestamptz_ns` must decode for inspect.
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::Long(1_704_083_400_000_000_000)),
        &Type::Primitive(PrimitiveType::TimestamptzNs),
    );
}

#[test]
fn avro_convert_test_list() {
    check_convert_with_avro(
        Literal::List(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(3))),
            None,
        ]),
        &Type::List(ListType {
            element_field: NestedField::list_element(0, Type::Primitive(PrimitiveType::Int), false)
                .into(),
        }),
    );

    check_convert_with_avro(
        Literal::List(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            Some(Literal::Primitive(PrimitiveLiteral::Int(3))),
        ]),
        &Type::List(ListType {
            element_field: NestedField::list_element(0, Type::Primitive(PrimitiveType::Int), true)
                .into(),
        }),
    );
}

#[test]
fn avro_convert_test_uuid() {
    // A Uuid-typed `UInt128` literal must round-trip through a real Avro schema, which emits
    // `AvroSchema::Uuid`. The deserialize side takes both the String form used here and the
    // legacy 16-byte Bytes form, so Java-written bytes still decode.
    let uuid = Uuid::parse_str("f79c3e09-677c-4bbd-a479-3f349cb785e7").unwrap();
    check_convert_with_avro(
        Literal::Primitive(PrimitiveLiteral::UInt128(uuid.as_u128())),
        &Type::Primitive(PrimitiveType::Uuid),
    );
}

#[test]
fn avro_serialize_uuid_resolves_against_schema_uuid() {
    // The bug was on the Avro schema-RESOLUTION path, which `check_serialize_avro` drives.
    // `resolve_uuid` takes a `Value::String` and rejects a `Value::Bytes`. The mutation this
    // discriminates: revert the serialize arm to `Bytes`. The plain round trip above stays green,
    // because the 16-byte deserialize arm still decodes.
    let uuid = Uuid::parse_str("f79c3e09-677c-4bbd-a479-3f349cb785e7").unwrap();
    check_serialize_avro(
        Literal::Primitive(PrimitiveLiteral::UInt128(uuid.as_u128())),
        &Type::Primitive(PrimitiveType::Uuid),
        Value::Uuid(uuid),
    );
}

fn check_convert_with_avro_map(expected_literal: Literal, expected_type: &Type) {
    let fields = vec![NestedField::required(1, "col", expected_type.clone()).into()];
    let schema = Schema::builder()
        .with_fields(fields.clone())
        .build()
        .unwrap();
    let avro_schema = schema_to_avro_schema("test", &schema).unwrap();
    let struct_type = Type::Struct(StructType::new(fields));
    let struct_literal = Literal::Struct(Struct::from_iter(vec![Some(expected_literal.clone())]));

    let mut writer = apache_avro::Writer::new(&avro_schema, Vec::new());
    let raw_literal = RawLiteral::try_from(struct_literal.clone(), &struct_type).unwrap();
    writer.append_ser(raw_literal).unwrap();
    let encoded = writer.into_inner().unwrap();

    let reader = apache_avro::Reader::new(&*encoded).unwrap();
    for record in reader {
        let result = apache_avro::from_value::<RawLiteral>(&record.unwrap()).unwrap();
        let desered_literal = result.try_into(&struct_type).unwrap().unwrap();
        match (&desered_literal, &struct_literal) {
            (Literal::Struct(desered), Literal::Struct(expected)) => {
                match (&desered.fields()[0], &expected.fields()[0]) {
                    (Some(Literal::Map(desered)), Some(Literal::Map(expected))) => {
                        assert!(desered.has_same_content(expected))
                    }
                    _ => {
                        unreachable!()
                    }
                }
            }
            _ => {
                panic!("unexpected literal type");
            }
        }
    }
}

#[test]
fn avro_convert_test_map() {
    check_convert_with_avro_map(
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::Int(1)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::Int(2)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(2))),
            ),
            (Literal::Primitive(PrimitiveLiteral::Int(3)), None),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(2, Type::Primitive(PrimitiveType::Int)).into(),
            value_field: NestedField::map_value_element(
                3,
                Type::Primitive(PrimitiveType::Long),
                false,
            )
            .into(),
        }),
    );

    check_convert_with_avro_map(
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::Int(1)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::Int(2)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(2))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::Int(3)),
                Some(Literal::Primitive(PrimitiveLiteral::Long(3))),
            ),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(2, Type::Primitive(PrimitiveType::Int)).into(),
            value_field: NestedField::map_value_element(
                3,
                Type::Primitive(PrimitiveType::Long),
                true,
            )
            .into(),
        }),
    );
}

#[test]
fn avro_convert_test_string_map() {
    check_convert_with_avro_map(
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::String("a".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("b".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("c".to_string())),
                None,
            ),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(2, Type::Primitive(PrimitiveType::String))
                .into(),
            value_field: NestedField::map_value_element(
                3,
                Type::Primitive(PrimitiveType::Int),
                false,
            )
            .into(),
        }),
    );

    check_convert_with_avro_map(
        Literal::Map(Map::from([
            (
                Literal::Primitive(PrimitiveLiteral::String("a".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("b".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(2))),
            ),
            (
                Literal::Primitive(PrimitiveLiteral::String("c".to_string())),
                Some(Literal::Primitive(PrimitiveLiteral::Int(3))),
            ),
        ])),
        &Type::Map(MapType {
            key_field: NestedField::map_key_element(2, Type::Primitive(PrimitiveType::String))
                .into(),
            value_field: NestedField::map_value_element(
                3,
                Type::Primitive(PrimitiveType::Int),
                true,
            )
            .into(),
        }),
    );
}

#[test]
fn avro_convert_test_record() {
    check_convert_with_avro(
        Literal::Struct(Struct::from_iter(vec![
            Some(Literal::Primitive(PrimitiveLiteral::Int(1))),
            Some(Literal::Primitive(PrimitiveLiteral::String(
                "bar".to_string(),
            ))),
            None,
        ])),
        &Type::Struct(StructType::new(vec![
            NestedField::required(2, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(4, "address", Type::Primitive(PrimitiveType::String)).into(),
        ])),
    );
}

// TODO(#86): rust avro cannot deserialize a bytes representation, so binary and decimal are
// not covered here.
#[test]
fn avro_convert_test_binary_ser() {
    let literal = Literal::Primitive(PrimitiveLiteral::Binary(vec![1, 2, 3, 4, 5]));
    let ty = Type::Primitive(PrimitiveType::Binary);
    let expect_value = Value::Bytes(vec![1, 2, 3, 4, 5]);
    check_serialize_avro(literal, &ty, expect_value);
}

#[test]
fn avro_convert_test_decimal_ser() {
    let literal = Literal::decimal(12345);
    let ty = Type::Primitive(PrimitiveType::Decimal {
        precision: 9,
        scale: 8,
    });
    let expect_value = Value::Decimal(apache_avro::Decimal::from(12345_i128.to_be_bytes()));
    check_serialize_avro(literal, &ty, expect_value);
}

// TODO(#86): rust avro cannot convert a byte-like type to avro fixed, so uuid and fixed
// ser/de are not covered here.

#[test]
fn test_parse_timestamp() {
    let value = Datum::timestamp_from_str("2021-08-01T01:09:00.0899").unwrap();
    assert_eq!(&format!("{value}"), "2021-08-01 01:09:00.089900");

    let value = Datum::timestamp_from_str("2023-01-06T00:00:00").unwrap();
    assert_eq!(&format!("{value}"), "2023-01-06 00:00:00");

    let value = Datum::timestamp_from_str("2021-08-01T01:09:00.0899+0800");
    assert!(value.is_err(), "Parse timestamp with timezone should fail!");

    let value = Datum::timestamp_from_str("dfa");
    assert!(
        value.is_err(),
        "Parse timestamp with invalid input should fail!"
    );
}

#[test]
fn test_parse_timestamptz() {
    let value = Datum::timestamptz_from_str("2021-08-01T09:09:00.0899+0800").unwrap();
    assert_eq!(&format!("{value}"), "2021-08-01 01:09:00.089900 UTC");

    let value = Datum::timestamptz_from_str("2021-08-01T01:09:00.0899");
    assert!(
        value.is_err(),
        "Parse timestamptz without timezone should fail!"
    );

    let value = Datum::timestamptz_from_str("dfa");
    assert!(
        value.is_err(),
        "Parse timestamptz with invalid input should fail!"
    );
}

#[test]
fn test_datum_ser_deser() {
    let test_fn = |datum: Datum| {
        let json = serde_json::to_value(&datum).unwrap();
        let desered_datum: Datum = serde_json::from_value(json).unwrap();
        assert_eq!(datum, desered_datum);
    };
    let datum = Datum::int(1);
    test_fn(datum);
    let datum = Datum::long(1);
    test_fn(datum);

    let datum = Datum::float(1.0);
    test_fn(datum);
    let datum = Datum::float(0_f32);
    test_fn(datum);
    let datum = Datum::float(-0_f32);
    test_fn(datum);
    let datum = Datum::float(f32::MAX);
    test_fn(datum);
    let datum = Datum::float(f32::MIN);
    test_fn(datum);

    // serde_json can't serialize f32::INFINITY, f32::NEG_INFINITY, f32::NAN
    let datum = Datum::float(f32::INFINITY);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());
    let datum = Datum::float(f32::NEG_INFINITY);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());
    let datum = Datum::float(f32::NAN);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());

    let datum = Datum::double(1.0);
    test_fn(datum);
    let datum = Datum::double(f64::MAX);
    test_fn(datum);
    let datum = Datum::double(f64::MIN);
    test_fn(datum);

    // serde_json can't serialize f32::INFINITY, f32::NEG_INFINITY, f32::NAN
    let datum = Datum::double(f64::INFINITY);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());
    let datum = Datum::double(f64::NEG_INFINITY);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());
    let datum = Datum::double(f64::NAN);
    let json = serde_json::to_string(&datum).unwrap();
    assert!(serde_json::from_str::<Datum>(&json).is_err());

    let datum = Datum::string("iceberg");
    test_fn(datum);
    let datum = Datum::bool(true);
    test_fn(datum);
    let datum = Datum::date(17486);
    test_fn(datum);
    let datum = Datum::time_from_hms_micro(22, 15, 33, 111).unwrap();
    test_fn(datum);
    let datum = Datum::timestamp_micros(1510871468123456);
    test_fn(datum);
    let datum = Datum::timestamptz_micros(1510871468123456);
    test_fn(datum);
    let datum = Datum::uuid(Uuid::parse_str("f79c3e09-677c-4bbd-a479-3f349cb785e7").unwrap());
    test_fn(datum);
    let datum = Datum::decimal(decimal_new(1420, 0)).unwrap();
    test_fn(datum);
    let datum = Datum::binary(vec![1, 2, 3, 4, 5]);
    test_fn(datum);
    let datum = Datum::fixed(vec![1, 2, 3, 4, 5]);
    test_fn(datum);
}

#[test]
fn test_datum_date_convert_to_int() {
    let datum_date = Datum::date(12345);

    let result = datum_date.to(&Primitive(PrimitiveType::Int)).unwrap();

    let expected = Datum::int(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_int_convert_to_date() {
    let datum_int = Datum::int(12345);

    let result = datum_int.to(&Primitive(PrimitiveType::Date)).unwrap();

    let expected = Datum::date(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_int() {
    let datum = Datum::long(12345);

    let result = datum.to(&Primitive(PrimitiveType::Int)).unwrap();

    let expected = Datum::int(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_int_above_max() {
    let datum = Datum::long(INT_MAX as i64 + 1);

    let result = datum.to(&Primitive(PrimitiveType::Int)).unwrap();

    let expected = Datum::new(PrimitiveType::Int, PrimitiveLiteral::AboveMax);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_int_below_min() {
    let datum = Datum::long(INT_MIN as i64 - 1);

    let result = datum.to(&Primitive(PrimitiveType::Int)).unwrap();

    let expected = Datum::new(PrimitiveType::Int, PrimitiveLiteral::BelowMin);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_timestamp() {
    let datum = Datum::long(12345);

    let result = datum.to(&Primitive(PrimitiveType::Timestamp)).unwrap();

    let expected = Datum::timestamp_micros(12345);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_long_convert_to_timestamptz() {
    let datum = Datum::long(12345);

    let result = datum.to(&Primitive(PrimitiveType::Timestamptz)).unwrap();

    let expected = Datum::timestamptz_micros(12345);

    assert_eq!(result, expected);
}

// Java `DecimalLiteral.to` accepts DECIMAL only, so Decimal-to-Long is a reject. These pin it
// across the value range. The mutation this discriminates: restore the `Int128 -> Long` arm.
#[test]
fn test_datum_decimal_convert_to_long_rejected() {
    let datum = Datum::decimal(decimal_new(12345, 0)).unwrap();
    let result = datum.to(&Primitive(PrimitiveType::Long));
    assert!(result.is_err());
}

#[test]
fn test_datum_decimal_convert_to_long_above_i64_max_rejected() {
    let datum = Datum::decimal(decimal_from_i128_with_scale(i64::MAX as i128 + 1, 0)).unwrap();
    let result = datum.to(&Primitive(PrimitiveType::Long));
    assert!(result.is_err());
}

#[test]
fn test_datum_decimal_convert_to_long_below_i64_min_rejected() {
    let datum = Datum::decimal(decimal_from_i128_with_scale(i64::MIN as i128 - 1, 0)).unwrap();
    let result = datum.to(&Primitive(PrimitiveType::Long));
    assert!(result.is_err());
}

// Java `StringLiteral.to` rejects BOOLEAN, INTEGER, and LONG. The inputs below are well-formed,
// so these pin the type contract, not a parse failure. The mutation this discriminates: restore
// any of the over-permissive arms.
#[test]
fn test_datum_string_convert_to_boolean_rejected() {
    let result = Datum::string("true").to(&Primitive(PrimitiveType::Boolean));
    assert!(result.is_err());
}

#[test]
fn test_datum_string_convert_to_int_rejected() {
    let result = Datum::string("12345").to(&Primitive(PrimitiveType::Int));
    assert!(result.is_err());
}

#[test]
fn test_datum_string_convert_to_long_rejected() {
    let result = Datum::string("12345").to(&Primitive(PrimitiveType::Long));
    assert!(result.is_err());
}

#[test]
fn test_datum_string_convert_to_timestamp() {
    let datum = Datum::string("1925-05-20T19:25:00.000");

    let result = datum.to(&Primitive(PrimitiveType::Timestamp)).unwrap();

    let expected = Datum::timestamp_micros(-1407990900000000);

    assert_eq!(result, expected);
}

#[test]
fn test_datum_string_convert_to_timestamptz() {
    let datum = Datum::string("1925-05-20T19:25:00.000 UTC");

    let result = datum.to(&Primitive(PrimitiveType::Timestamptz)).unwrap();

    let expected = Datum::timestamptz_micros(-1407990900000000);

    assert_eq!(result, expected);
}

// `Datum::to` promotions, one assertion per arm: accept, boundary sentinel, and reject.

#[test]
fn test_datum_int_convert_to_float_and_double() {
    assert_eq!(
        Datum::int(7).to(&Primitive(PrimitiveType::Float)).unwrap(),
        Datum::float(7.0)
    );
    assert_eq!(
        Datum::int(7).to(&Primitive(PrimitiveType::Double)).unwrap(),
        Datum::double(7.0)
    );
}

#[test]
fn test_datum_long_convert_to_float_and_double() {
    assert_eq!(
        Datum::long(7).to(&Primitive(PrimitiveType::Float)).unwrap(),
        Datum::float(7.0)
    );
    assert_eq!(
        Datum::long(7)
            .to(&Primitive(PrimitiveType::Double))
            .unwrap(),
        Datum::double(7.0)
    );
}

#[test]
fn test_datum_long_convert_to_date() {
    let result = Datum::long(12345)
        .to(&Primitive(PrimitiveType::Date))
        .unwrap();
    assert_eq!(result, Datum::date(12345));
}

#[test]
fn test_datum_long_convert_to_date_above_max() {
    // One past Integer.MAX_VALUE gives the `aboveMax` sentinel, typed as Date.
    let result = Datum::long(INT_MAX as i64 + 1)
        .to(&Primitive(PrimitiveType::Date))
        .unwrap();
    assert_eq!(
        result,
        Datum::new(PrimitiveType::Date, PrimitiveLiteral::AboveMax)
    );
}

#[test]
fn test_datum_long_convert_to_date_below_min() {
    let result = Datum::long(INT_MIN as i64 - 1)
        .to(&Primitive(PrimitiveType::Date))
        .unwrap();
    assert_eq!(
        result,
        Datum::new(PrimitiveType::Date, PrimitiveLiteral::BelowMin)
    );
}

#[test]
fn test_datum_float_convert_to_double() {
    let result = Datum::float(1.5)
        .to(&Primitive(PrimitiveType::Double))
        .unwrap();
    assert_eq!(result, Datum::double(1.5));
}

#[test]
fn test_datum_double_convert_to_float() {
    let result = Datum::double(1.5)
        .to(&Primitive(PrimitiveType::Float))
        .unwrap();
    assert_eq!(result, Datum::float(1.5));
}

#[test]
fn test_datum_double_convert_to_float_above_max() {
    // Beyond Float.MAX_VALUE the result is the `aboveMax` sentinel, typed as Float.
    let result = Datum::double(1.0e40)
        .to(&Primitive(PrimitiveType::Float))
        .unwrap();
    assert_eq!(
        result,
        Datum::new(PrimitiveType::Float, PrimitiveLiteral::AboveMax)
    );
}

#[test]
fn test_datum_double_convert_to_float_below_min() {
    let result = Datum::double(-1.0e40)
        .to(&Primitive(PrimitiveType::Float))
        .unwrap();
    assert_eq!(
        result,
        Datum::new(PrimitiveType::Float, PrimitiveLiteral::BelowMin)
    );
}

// A value EQUAL to the sentinel threshold must keep the value. The mutation this discriminates:
// `>` to `>=`, or `<` to `<=`, which the `+1` and `-1` tests above do not catch.
#[test]
fn test_datum_long_convert_to_date_at_int_max() {
    let result = Datum::long(INT_MAX as i64)
        .to(&Primitive(PrimitiveType::Date))
        .unwrap();
    assert_eq!(result, Datum::date(INT_MAX));
}

#[test]
fn test_datum_long_convert_to_date_at_int_min() {
    let result = Datum::long(INT_MIN as i64)
        .to(&Primitive(PrimitiveType::Date))
        .unwrap();
    assert_eq!(result, Datum::date(INT_MIN));
}

#[test]
fn test_datum_double_convert_to_float_at_max() {
    let result = Datum::double(f32::MAX as f64)
        .to(&Primitive(PrimitiveType::Float))
        .unwrap();
    assert_eq!(result, Datum::float(f32::MAX));
}

#[test]
fn test_datum_decimal_convert_to_decimal_preserves_scale() {
    // Java returns `this`, so the source value and scale survive a different target scale.
    let datum = Datum::decimal(decimal_new(12345, 2)).unwrap();
    let expected = datum.clone();
    let result = datum
        .to(&Primitive(PrimitiveType::Decimal {
            precision: 10,
            scale: 4,
        }))
        .unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_datum_fixed_convert_to_binary() {
    let result = Datum::fixed(vec![1u8, 2, 3])
        .to(&Primitive(PrimitiveType::Binary))
        .unwrap();
    assert_eq!(result, Datum::binary(vec![1u8, 2, 3]));
}

#[test]
fn test_datum_binary_convert_to_fixed_matching_length() {
    let result = Datum::binary(vec![1u8, 2, 3])
        .to(&Primitive(PrimitiveType::Fixed(3)))
        .unwrap();
    assert_eq!(result, Datum::fixed(vec![1u8, 2, 3]));
}

#[test]
fn test_datum_binary_convert_to_fixed_length_mismatch_rejected() {
    // Java BinaryLiteral.to(FIXED) returns null on a length mismatch — a reject here.
    let result = Datum::binary(vec![1u8, 2, 3]).to(&Primitive(PrimitiveType::Fixed(4)));
    assert!(result.is_err());
}

#[test]
fn test_datum_string_convert_to_uuid() {
    let datum = Datum::string("550e8400-e29b-41d4-a716-446655440000");
    let result = datum.to(&Primitive(PrimitiveType::Uuid)).unwrap();
    let expected = Datum::uuid_from_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_datum_string_convert_to_uuid_invalid_rejected() {
    let result = Datum::string("not-a-uuid").to(&Primitive(PrimitiveType::Uuid));
    assert!(result.is_err());
}

#[test]
fn test_datum_string_convert_to_date() {
    let datum = Datum::string("2017-11-16");
    let result = datum.to(&Primitive(PrimitiveType::Date)).unwrap();
    assert_eq!(result, Datum::date_from_str("2017-11-16").unwrap());
}

#[test]
fn test_datum_string_convert_to_time() {
    let datum = Datum::string("22:31:08");
    let result = datum.to(&Primitive(PrimitiveType::Time)).unwrap();
    assert_eq!(result, Datum::time_from_str("22:31:08").unwrap());
}

#[test]
fn test_datum_boolean_convert_to_int_rejected() {
    // Java `BooleanLiteral.to` accepts BOOLEAN only. This guards an over-permissive arm.
    let result = Datum::bool(true).to(&Primitive(PrimitiveType::Int));
    assert!(result.is_err());
}

#[test]
fn test_iceberg_float_order() {
    let float_values = vec![
        Datum::float(f32::NAN),
        Datum::float(-f32::NAN),
        Datum::float(f32::MAX),
        Datum::float(f32::MIN),
        Datum::float(f32::INFINITY),
        Datum::float(-f32::INFINITY),
        Datum::float(1.0),
        Datum::float(-1.0),
        Datum::float(0.0),
        Datum::float(-0.0),
    ];

    let mut float_sorted = float_values.clone();
    float_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let float_expected = vec![
        Datum::float(-f32::NAN),
        Datum::float(-f32::INFINITY),
        Datum::float(f32::MIN),
        Datum::float(-1.0),
        Datum::float(-0.0),
        Datum::float(0.0),
        Datum::float(1.0),
        Datum::float(f32::MAX),
        Datum::float(f32::INFINITY),
        Datum::float(f32::NAN),
    ];

    assert_eq!(float_sorted, float_expected);

    let double_values = vec![
        Datum::double(f64::NAN),
        Datum::double(-f64::NAN),
        Datum::double(f64::INFINITY),
        Datum::double(-f64::INFINITY),
        Datum::double(f64::MAX),
        Datum::double(f64::MIN),
        Datum::double(1.0),
        Datum::double(-1.0),
        Datum::double(0.0),
        Datum::double(-0.0),
    ];

    let mut double_sorted = double_values.clone();
    double_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let double_expected = vec![
        Datum::double(-f64::NAN),
        Datum::double(-f64::INFINITY),
        Datum::double(f64::MIN),
        Datum::double(-1.0),
        Datum::double(-0.0),
        Datum::double(0.0),
        Datum::double(1.0),
        Datum::double(f64::MAX),
        Datum::double(f64::INFINITY),
        Datum::double(f64::NAN),
    ];

    assert_eq!(double_sorted, double_expected);
}

#[test]
fn test_negative_zero_less_than_positive_zero() {
    {
        let neg_zero = Datum::float(-0.0);
        let pos_zero = Datum::float(0.0);

        assert_eq!(
            neg_zero.partial_cmp(&pos_zero),
            Some(std::cmp::Ordering::Less),
            "IEEE 754 totalOrder requires -0.0 < +0.0 on F32"
        );
    }

    {
        let neg_zero = Datum::double(-0.0);
        let pos_zero = Datum::double(0.0);

        assert_eq!(
            neg_zero.partial_cmp(&pos_zero),
            Some(std::cmp::Ordering::Less),
            "IEEE 754 totalOrder requires -0.0 < +0.0 on F64"
        );
    }
}

/// A Date must deserialize from JSON as a number of days since epoch, not only from a string.
/// Java's `TestAddFilesProcedure` writes a date partition default as `18628`, and parsing strings
/// only lost every such `initial_default`. Both forms must work.
#[test]
fn test_date_from_json_as_number() {
    use serde_json::json;

    // The number form, used in an `add_files` initial default.
    let date_number = json!(18628); // 2021-01-01 is 18628 days since 1970-01-01
    let result =
        Literal::try_from_json(date_number, &Type::Primitive(PrimitiveType::Date)).unwrap();
    assert_eq!(
        result,
        Some(Literal::Primitive(PrimitiveLiteral::Int(18628)))
    );

    // The string form.
    let date_string = json!("2021-01-01");
    let result =
        Literal::try_from_json(date_string, &Type::Primitive(PrimitiveType::Date)).unwrap();
    assert_eq!(
        result,
        Some(Literal::Primitive(PrimitiveLiteral::Int(18628)))
    );

    // Both forms give the same literal.
}

// Risk: a variant column can carry no default. Java's parser has no VARIANT case and throws,
// while a JSON null means no default. Accepting one writes metadata Java cannot parse back.
#[test]
fn test_variant_default_value_json_is_rejected_but_null_is_none() {
    use serde_json::json;

    // Null parses to None, as it does for every other type.
    let null_result = Literal::try_from_json(JsonValue::Null, &Type::Variant)
        .expect("a JSON null is 'no default' for variant too");
    assert_eq!(null_result, None);

    // A non-null default is rejected with Java's message.
    let error = Literal::try_from_json(json!({"a": 1}), &Type::Variant)
        .expect_err("a non-null variant default must be rejected");
    assert_eq!(error.kind(), crate::ErrorKind::FeatureUnsupported);
    assert_eq!(error.message(), "Type: variant is not supported");
}

// Risk: the write direction must also fail loudly. Neither language has a variant literal, and
// Java's `toJson` throws. The catch-all must not render another literal under a variant type.
#[test]
fn test_variant_literal_to_json_is_rejected() {
    let error = Literal::Primitive(PrimitiveLiteral::Long(7))
        .try_into_json(&Type::Variant)
        .expect_err("no literal fits the variant type");
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        error.message().contains("variant"),
        "the rejection must name the variant type, got: {}",
        error.message()
    );
}

// Risk: `unknown` has no single-value byte encoding, because its values are always null.
// `Datum::try_from_bytes` must reject it. The mutation: add a value-producing arm.
#[test]
fn test_datum_try_from_bytes_rejects_unknown() {
    let error = Datum::try_from_bytes(&[0u8, 1u8, 2u8], PrimitiveType::Unknown)
        .expect_err("unknown has no single-value byte encoding");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.message().contains("unknown"),
        "the rejection must name the unknown type, got: {}",
        error.message()
    );
}

// Risk: a null single-value JSON for an `unknown` column must give `None`, as Java gives null
// for a null node of any type. A non-null value for such a column must not parse.
#[test]
fn test_unknown_single_value_json_null_is_none() {
    let none = Literal::try_from_json(JsonValue::Null, &Primitive(PrimitiveType::Unknown))
        .expect("a null unknown value parses");
    assert_eq!(none, None, "a null unknown single-value must be None");

    let error = Literal::try_from_json(JsonValue::from(7), &Primitive(PrimitiveType::Unknown))
        .expect_err("a non-null unknown value must not parse");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
}
