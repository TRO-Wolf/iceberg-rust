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

use std::sync::Arc;

use super::super::orc_type::build_orc_schema;
use super::*;
use crate::spec::{NestedField, Schema};

fn decode_orc_nanos(encoded: u64) -> u64 {
    let zeros = encoded & 0x7;
    let mut value = encoded >> 3;
    if zeros != 0 {
        value *= 10u64.pow(zeros as u32 + 1);
    }
    value
}

fn decode_orc_timestamp(seconds_delta: i64, encoded_nanos: i64) -> i128 {
    let nanos = decode_orc_nanos(encoded_nanos as u64) as i128;
    let seconds_since_epoch = i128::from(seconds_delta) + i128::from(ORC_EPOCH_UTC_SECONDS);
    let seconds = if seconds_since_epoch < 0 && nanos > 999_999 {
        seconds_since_epoch - 1
    } else {
        seconds_since_epoch
    };
    seconds * 1_000_000_000 + nanos
}

#[test]
fn test_encode_nanos_round_trips_every_trailing_zero_shape() {
    for nanos in [
        0u64,
        1,
        7,
        99,
        100,
        1_000,
        123_456,
        1_000_000,
        123_456_789,
        500_000_000,
        999_999_999,
        900_000_000,
    ] {
        assert_eq!(
            decode_orc_nanos(encode_nanos(nanos)),
            nanos,
            "nanos round trip for {nanos}"
        );
    }
}

#[test]
fn test_encode_nanos_zero_is_the_empty_encoding() {
    assert_eq!(encode_nanos(0), 0);
}

#[test]
fn test_micros_timestamps_round_trip_through_the_orc_reader_rule() {
    for micros in [
        0i64,
        1,
        999_999,
        1_000_000,
        -1_000_000,
        -1_999_999,
        -2_000_000,
        -999_999_999,
        1_704_103_200_000_000,
        253_402_300_799_999_999,
        -62_135_596_800_000_000,
    ] {
        let (seconds, nanos) = split_timestamp(micros, 1_000).expect("split a micros timestamp");
        assert_eq!(
            decode_orc_timestamp(seconds, nanos),
            i128::from(micros) * 1_000,
            "timestamp round trip for {micros} micros"
        );
    }
}

#[test]
fn test_nanos_timestamps_round_trip_through_the_orc_reader_rule() {
    for nanos_value in [
        0i64,
        1,
        999_999_999,
        -999_999_999,
        -1_000_000_000,
        -2_000_000_001,
        1_704_103_200_123_456_789,
    ] {
        let (seconds, nanos) = split_timestamp(nanos_value, 1).expect("split a nanos timestamp");
        assert_eq!(
            decode_orc_timestamp(seconds, nanos),
            i128::from(nanos_value),
            "timestamp round trip for {nanos_value} nanos"
        );
    }
}

#[test]
fn test_the_orc_763_window_below_the_epoch_matches_java_s_encoding() {
    let (seconds, nanos) = split_timestamp(-500_000, 1_000).expect("split -0.5s");
    assert_eq!(
        seconds,
        -ORC_EPOCH_UTC_SECONDS,
        "Java truncates -0.5s toward zero, so the stored second is the epoch second"
    );
    assert_eq!(decode_orc_nanos(nanos as u64), 500_000_000);
    assert_eq!(
        decode_orc_timestamp(seconds, nanos),
        500_000_000,
        "ORC-763: the final second before the epoch decodes to its positive mirror, in Java too"
    );
}

#[test]
fn test_the_orc_763_dead_window_is_exactly_the_microsecond_before_the_epoch() {
    for micros in [-1i64, -999_000, -500_000] {
        let (seconds, nanos) = split_timestamp(micros, 1_000).expect("split a dead-window value");
        assert_ne!(
            decode_orc_timestamp(seconds, nanos),
            i128::from(micros) * 1_000,
            "{micros} micros sits in the ORC-763 window Java cannot round trip either"
        );
    }
    for micros in [-999_999i64, -1_000_000, -1_000_001, -999_999_999] {
        let (seconds, nanos) = split_timestamp(micros, 1_000).expect("split a live value");
        assert_eq!(
            decode_orc_timestamp(seconds, nanos),
            i128::from(micros) * 1_000,
            "{micros} micros is outside the ORC-763 window and must round trip"
        );
    }
}

fn flat_schema() -> Schema {
    Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Int),
            )),
            Arc::new(NestedField::optional(
                2,
                "name",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build the flat schema")
}

fn row(id: i32, name: Option<&str>) -> Literal {
    Literal::Struct(crate::spec::Struct::from_iter(vec![
        Some(Literal::Primitive(PrimitiveLiteral::Int(id))),
        name.map(|value| Literal::Primitive(PrimitiveLiteral::String(value.to_string()))),
    ]))
}

#[test]
fn test_a_column_without_nulls_writes_no_present_stream() {
    let schema = flat_schema();
    let orc_schema = build_orc_schema(&schema).expect("map the schema");
    let mut encoder = StripeEncoder::new(&orc_schema);
    encoder
        .append_row(&orc_schema, Some(&row(1, Some("a"))))
        .expect("append row 1");
    encoder
        .append_row(&orc_schema, Some(&row(2, Some("b"))))
        .expect("append row 2");

    let finished = encoder
        .finish(&orc_schema, OrcCompression::None, 1024)
        .expect("finish the stripe");
    let kinds: Vec<(u32, StreamKind)> = finished
        .streams
        .iter()
        .map(|s| (s.column, s.kind))
        .collect();
    assert_eq!(kinds, vec![
        (1, StreamKind::Data),
        (2, StreamKind::Data),
        (2, StreamKind::Length),
    ]);
    assert_eq!(finished.rows, 2);
}

#[test]
fn test_a_column_with_a_null_writes_a_present_stream_first() {
    let schema = flat_schema();
    let orc_schema = build_orc_schema(&schema).expect("map the schema");
    let mut encoder = StripeEncoder::new(&orc_schema);
    encoder
        .append_row(&orc_schema, Some(&row(1, Some("a"))))
        .expect("append row 1");
    encoder
        .append_row(&orc_schema, Some(&row(2, None)))
        .expect("append row 2");

    let finished = encoder
        .finish(&orc_schema, OrcCompression::None, 1024)
        .expect("finish the stripe");
    let kinds: Vec<(u32, StreamKind)> = finished
        .streams
        .iter()
        .map(|s| (s.column, s.kind))
        .collect();
    assert_eq!(kinds, vec![
        (1, StreamKind::Data),
        (2, StreamKind::Present),
        (2, StreamKind::Data),
        (2, StreamKind::Length),
    ]);
}

#[test]
fn test_finish_resets_the_encoder_for_the_next_stripe() {
    let schema = flat_schema();
    let orc_schema = build_orc_schema(&schema).expect("map the schema");
    let mut encoder = StripeEncoder::new(&orc_schema);
    encoder
        .append_row(&orc_schema, Some(&row(1, Some("a"))))
        .expect("append row 1");
    let first = encoder
        .finish(&orc_schema, OrcCompression::None, 1024)
        .expect("finish stripe 1");
    assert_eq!(first.rows, 1);
    assert_eq!(encoder.rows(), 0);
    assert_eq!(encoder.estimated_size(), 0);

    encoder
        .append_row(&orc_schema, Some(&row(2, Some("bb"))))
        .expect("append row 2");
    let second = encoder
        .finish(&orc_schema, OrcCompression::None, 1024)
        .expect("finish stripe 2");
    assert_eq!(second.rows, 1);
    assert_ne!(
        first.data, second.data,
        "the second stripe must carry only its own row"
    );
}

#[test]
fn test_a_value_of_the_wrong_shape_is_a_typed_error_not_a_panic() {
    let schema = flat_schema();
    let orc_schema = build_orc_schema(&schema).expect("map the schema");
    let mut encoder = StripeEncoder::new(&orc_schema);
    let bad = Literal::Struct(crate::spec::Struct::from_iter(vec![
        Some(Literal::Primitive(PrimitiveLiteral::String("x".to_string()))),
        None,
    ]));
    let error = encoder
        .append_row(&orc_schema, Some(&bad))
        .expect_err("an Int column cannot take a String literal");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
}
