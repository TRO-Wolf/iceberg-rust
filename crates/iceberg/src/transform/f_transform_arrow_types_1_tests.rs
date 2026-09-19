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

use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, FixedSizeBinaryArray, Int32Array,
    LargeBinaryArray, LargeStringArray, StringArray, StringViewArray,
};
use arrow_schema::DataType;

use crate::expr::{BoundPredicate, Predicate, PredicateOperator};
use crate::spec::{Datum, NestedField, PrimitiveLiteral, PrimitiveType, Transform, Type};
use crate::transform::create_transform_function;
use crate::transform::test::TestProjectionFixture;
use crate::{ErrorKind, Result};

const BINARY_ROWS: [Option<&[u8]>; 7] = [
    Some(&[]),
    Some(&[0x01]),
    Some(&[0x01, 0x02]),
    Some(&[0x01, 0x02, 0x03]),
    Some(&[0xff, 0x00, 0xff]),
    None,
    Some(&[0xe4, 0xb8, 0xad]),
];

const STRING_ROWS: [Option<&str>; 7] = [
    Some(""),
    Some("iceberg"),
    Some("中文字"),
    Some("a中b"),
    Some("🚀"),
    None,
    Some("abcdefg"),
];

fn binary_arrays() -> Vec<ArrayRef> {
    vec![
        Arc::new(BinaryArray::from(BINARY_ROWS.to_vec())),
        Arc::new(LargeBinaryArray::from(BINARY_ROWS.to_vec())),
        Arc::new(BinaryViewArray::from(BINARY_ROWS.to_vec())),
    ]
}

fn string_arrays() -> Vec<ArrayRef> {
    vec![
        Arc::new(StringArray::from(STRING_ROWS.to_vec())),
        Arc::new(LargeStringArray::from(STRING_ROWS.to_vec())),
        Arc::new(StringViewArray::from(STRING_ROWS.to_vec())),
    ]
}

fn binary_value(array: &ArrayRef, row: usize) -> Vec<u8> {
    match array.data_type() {
        DataType::Binary => array
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("Binary array")
            .value(row)
            .to_vec(),
        DataType::LargeBinary => array
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .expect("LargeBinary array")
            .value(row)
            .to_vec(),
        DataType::BinaryView => array
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .expect("BinaryView array")
            .value(row)
            .to_vec(),
        other => panic!("expected a binary layout, got {other:?}"),
    }
}

fn string_value(array: &ArrayRef, row: usize) -> String {
    match array.data_type() {
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Utf8 array")
            .value(row)
            .to_string(),
        DataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("LargeUtf8 array")
            .value(row)
            .to_string(),
        DataType::Utf8View => array
            .as_any()
            .downcast_ref::<StringViewArray>()
            .expect("Utf8View array")
            .value(row)
            .to_string(),
        other => panic!("expected a string layout, got {other:?}"),
    }
}

fn transform_of(transform: &Transform) -> crate::transform::BoxedTransformFunction {
    create_transform_function(transform).expect("create transform function")
}

fn assert_binary_rows(out: &ArrayRef, expected: &[Option<Vec<u8>>]) {
    assert_eq!(out.len(), expected.len());
    for (row, expected) in expected.iter().enumerate() {
        match expected {
            None => assert!(out.is_null(row), "row {row} must be null"),
            Some(bytes) => {
                assert!(!out.is_null(row), "row {row} must not be null");
                assert_eq!(&binary_value(out, row), bytes, "row {row}");
            }
        }
    }
}

fn assert_string_rows(out: &ArrayRef, expected: &[Option<&str>]) {
    assert_eq!(out.len(), expected.len());
    for (row, expected) in expected.iter().enumerate() {
        match expected {
            None => assert!(out.is_null(row), "row {row} must be null"),
            Some(s) => {
                assert!(!out.is_null(row), "row {row} must not be null");
                assert_eq!(string_value(out, row), *s, "row {row}");
            }
        }
    }
}

fn assert_int_rows(out: &ArrayRef, expected: &[Option<i32>]) {
    let out = out
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("bucket output must be Int32");
    assert_eq!(out.len(), expected.len());
    for (row, expected) in expected.iter().enumerate() {
        match expected {
            None => assert!(out.is_null(row), "row {row} must be null"),
            Some(v) => {
                assert!(!out.is_null(row), "row {row} must not be null");
                assert_eq!(out.value(row), *v, "row {row}");
            }
        }
    }
}

#[test]
fn test_truncate_binary_oracle_every_layout() {
    let cases: [(u32, [Option<Vec<u8>>; 7]); 3] = [
        (1, [
            Some(vec![]),
            Some(vec![0x01]),
            Some(vec![0x01]),
            Some(vec![0x01]),
            Some(vec![0xff]),
            None,
            Some(vec![0xe4]),
        ]),
        (2, [
            Some(vec![]),
            Some(vec![0x01]),
            Some(vec![0x01, 0x02]),
            Some(vec![0x01, 0x02]),
            Some(vec![0xff, 0x00]),
            None,
            Some(vec![0xe4, 0xb8]),
        ]),
        (3, [
            Some(vec![]),
            Some(vec![0x01]),
            Some(vec![0x01, 0x02]),
            Some(vec![0x01, 0x02, 0x03]),
            Some(vec![0xff, 0x00, 0xff]),
            None,
            Some(vec![0xe4, 0xb8, 0xad]),
        ]),
    ];
    for input in binary_arrays() {
        let layout = input.data_type().clone();
        for (width, expected) in &cases {
            let out = transform_of(&Transform::Truncate(*width))
                .transform(input.clone())
                .unwrap_or_else(|e| panic!("truncate[{width}] must accept {layout:?}: {e}"));
            assert_eq!(
                out.data_type(),
                &layout,
                "truncate[{width}] output must keep the input layout {layout:?}"
            );
            assert_binary_rows(&out, expected);
        }
    }
}

#[test]
fn test_bucket_binary_oracle_every_layout() {
    let cases: [(u32, [Option<i32>; 7]); 2] = [
        (4, [
            Some(0),
            Some(3),
            Some(2),
            Some(0),
            Some(2),
            None,
            Some(2),
        ]),
        (16, [
            Some(0),
            Some(11),
            Some(14),
            Some(4),
            Some(6),
            None,
            Some(2),
        ]),
    ];
    for input in binary_arrays() {
        let layout = input.data_type().clone();
        for (num_buckets, expected) in &cases {
            let out = transform_of(&Transform::Bucket(*num_buckets))
                .transform(input.clone())
                .unwrap_or_else(|e| panic!("bucket[{num_buckets}] must accept {layout:?}: {e}"));
            assert_eq!(out.data_type(), &DataType::Int32);
            assert_int_rows(&out, expected);
        }
    }
}

#[test]
fn test_identity_and_void_binary_every_layout() {
    let oracle: [Option<Vec<u8>>; 7] = BINARY_ROWS.map(|v| v.map(<[u8]>::to_vec));
    for input in binary_arrays() {
        let layout = input.data_type().clone();
        let identity = transform_of(&Transform::Identity)
            .transform(input.clone())
            .expect("identity must accept every binary layout");
        assert_eq!(identity.data_type(), &layout);
        assert_binary_rows(&identity, &oracle);
        let void = transform_of(&Transform::Void)
            .transform(input.clone())
            .expect("void must accept every binary layout");
        assert_eq!(void.data_type(), &layout);
        assert_binary_rows(&void, &vec![None::<Vec<u8>>; 7]);
    }
}

#[test]
fn test_truncate_string_oracle_every_layout() {
    let cases: [(u32, [Option<&str>; 7]); 2] = [
        (2, [
            Some(""),
            Some("ic"),
            Some("中文"),
            Some("a中"),
            Some("🚀"),
            None,
            Some("ab"),
        ]),
        (4, [
            Some(""),
            Some("iceb"),
            Some("中文字"),
            Some("a中b"),
            Some("🚀"),
            None,
            Some("abcd"),
        ]),
    ];
    for input in string_arrays() {
        let layout = input.data_type().clone();
        for (width, expected) in &cases {
            let out = transform_of(&Transform::Truncate(*width))
                .transform(input.clone())
                .unwrap_or_else(|e| panic!("truncate[{width}] must accept {layout:?}: {e}"));
            assert_eq!(out.data_type(), &layout);
            assert_string_rows(&out, expected);
        }
    }
}

#[test]
fn test_bucket_string_oracle_every_layout() {
    let cases: [(u32, [Option<i32>; 7]); 2] = [
        (4, [
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(1),
            None,
            Some(2),
        ]),
        (16, [
            Some(0),
            Some(9),
            Some(10),
            Some(11),
            Some(5),
            None,
            Some(6),
        ]),
    ];
    for input in string_arrays() {
        let layout = input.data_type().clone();
        for (num_buckets, expected) in &cases {
            let out = transform_of(&Transform::Bucket(*num_buckets))
                .transform(input.clone())
                .unwrap_or_else(|e| panic!("bucket[{num_buckets}] must accept {layout:?}: {e}"));
            assert_eq!(out.data_type(), &DataType::Int32);
            assert_int_rows(&out, expected);
        }
    }
}

#[test]
fn test_identity_and_void_string_every_layout() {
    for input in string_arrays() {
        let layout = input.data_type().clone();
        let identity = transform_of(&Transform::Identity)
            .transform(input.clone())
            .expect("identity must accept every string layout");
        assert_eq!(identity.data_type(), &layout);
        assert_string_rows(&identity, &STRING_ROWS);
        let void = transform_of(&Transform::Void)
            .transform(input.clone())
            .expect("void must accept every string layout");
        assert_eq!(void.data_type(), &layout);
        assert_string_rows(&void, &[None; 7]);
    }
}

#[test]
fn test_truncate_fixed_size_binary_rejected_java_parity() {
    let input: ArrayRef = Arc::new(FixedSizeBinaryArray::from(vec![
        &b"\x01\x02"[..],
        &b"\x03\x04"[..],
    ]));
    let err = transform_of(&Transform::Truncate(1))
        .transform(input)
        .expect_err("Java Truncate.canTransform does not admit FIXED");
    assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
}

#[test]
fn test_bucket_fixed_size_binary_oracle() {
    let input: ArrayRef = Arc::new(FixedSizeBinaryArray::from(vec![
        &b"\x01\x02"[..],
        &b"\x01\x02"[..],
        &b"\x01\x02"[..],
    ]));
    let out = transform_of(&Transform::Bucket(16))
        .transform(input)
        .expect("Java Bucket.canTransform admits FIXED");
    assert_int_rows(&out, &[Some(14), Some(14), Some(14)]);
}

#[test]
fn test_truncate_literal_binary() -> Result<()> {
    let f = transform_of(&Transform::Truncate(1));
    let out = f
        .transform_literal(&Datum::binary(vec![0x01, 0x02]))?
        .expect("truncate literal on binary must return a value");
    assert_eq!(out, Datum::binary(vec![0x01]));
    let f = transform_of(&Transform::Truncate(2));
    let out = f
        .transform_literal(&Datum::binary(vec![0xe4, 0xb8, 0xad]))?
        .expect("truncate literal on binary must return a value");
    assert_eq!(out, Datum::binary(vec![0xe4, 0xb8]));
    let f = transform_of(&Transform::Truncate(4));
    let out = f
        .transform_literal(&Datum::binary(vec![0x01, 0x02]))?
        .expect("truncate literal on binary must return a value");
    assert_eq!(out, Datum::binary(vec![0x01, 0x02]));
    Ok(())
}

#[test]
fn test_bucket_literal_binary_and_fixed() -> Result<()> {
    let f = transform_of(&Transform::Bucket(4));
    let out = f
        .transform_literal(&Datum::binary(vec![0x01, 0x02]))?
        .expect("bucket literal on binary must return a value");
    assert_eq!(out, Datum::int(2));
    let out = f
        .transform_literal(&Datum::new(
            PrimitiveType::Fixed(2),
            PrimitiveLiteral::Binary(vec![0x01, 0x02]),
        ))?
        .expect("bucket literal on fixed must return a value");
    assert_eq!(out, Datum::int(2));
    let f = transform_of(&Transform::Bucket(16));
    let out = f
        .transform_literal(&Datum::binary(vec![]))?
        .expect("bucket literal on empty binary must return a value");
    assert_eq!(out, Datum::int(0));
    Ok(())
}

fn binary_fixture(transform: Transform) -> TestProjectionFixture {
    TestProjectionFixture::new(
        transform,
        "b_trunc",
        NestedField::required(1, "b", Type::Primitive(PrimitiveType::Binary)),
    )
}

fn projected_binary(predicate: BoundPredicate) -> Option<(PredicateOperator, Datum)> {
    match Transform::Truncate(1)
        .project("b_trunc", &predicate)
        .expect("project must not error on a binary literal")
    {
        Some(Predicate::Binary(expr)) => Some((expr.op(), expr.literal().clone())),
        other => panic!("expected a binary partition predicate, got {other:?}"),
    }
}

#[test]
fn test_project_truncate_binary_eq() {
    let fixture = binary_fixture(Transform::Truncate(1));
    let (op, datum) = projected_binary(
        fixture.binary_predicate(PredicateOperator::Eq, Datum::binary(vec![0x01, 0x02])),
    )
    .expect("truncate(1) on b = X'0102' must project to a partition predicate");
    assert_eq!(op, PredicateOperator::Eq);
    assert_eq!(datum, Datum::binary(vec![0x01]));
}

#[test]
fn test_project_truncate_binary_in_set() -> Result<()> {
    let fixture = binary_fixture(Transform::Truncate(1));
    let predicate = fixture.set_predicate(PredicateOperator::In, vec![
        Datum::binary(vec![0x01, 0x02]),
        Datum::binary(vec![0xe4, 0xb8, 0xad]),
    ]);
    let projected = Transform::Truncate(1)
        .project("b_trunc", &predicate)?
        .expect("truncate(1) on an IN list must project to a partition predicate");
    match projected {
        Predicate::Set(expr) => {
            let literals: Vec<Datum> = expr.literals().iter().cloned().collect();
            assert_eq!(literals.len(), 2);
            assert!(literals.contains(&Datum::binary(vec![0x01])));
            assert!(literals.contains(&Datum::binary(vec![0xe4])));
        }
        other => panic!("expected a set partition predicate, got {other:?}"),
    }
    Ok(())
}

#[test]
fn test_project_truncate_binary_range() {
    let fixture = binary_fixture(Transform::Truncate(1));
    let (op, datum) = projected_binary(fixture.binary_predicate(
        PredicateOperator::LessThanOrEq,
        Datum::binary(vec![0x01, 0x02]),
    ))
    .expect("truncate(1) on b <= X'0102' must project to a partition predicate");
    assert_eq!(op, PredicateOperator::LessThanOrEq);
    assert_eq!(datum, Datum::binary(vec![0x01]));
    let (op, datum) = projected_binary(fixture.binary_predicate(
        PredicateOperator::GreaterThan,
        Datum::binary(vec![0x01, 0x02]),
    ))
    .expect("truncate(1) on b > X'0102' must project to a partition predicate");
    assert_eq!(op, PredicateOperator::GreaterThanOrEq);
    assert_eq!(datum, Datum::binary(vec![0x01]));
}

#[test]
fn test_strict_project_truncate_binary_not_starts_with() -> Result<()> {
    let fixture = binary_fixture(Transform::Truncate(1));
    let projected = Transform::Truncate(1).strict_project(
        "b_trunc",
        &fixture.binary_predicate(
            PredicateOperator::NotStartsWith,
            Datum::binary(vec![0x01, 0x02]),
        ),
    )?;
    assert!(
        projected.is_none(),
        "Java strict projection returns null for a NotStartsWith literal longer than the width, got {projected:?}"
    );
    Ok(())
}

#[test]
fn test_project_truncate_binary_not_starts_with_longer_than_width_is_none() {
    let fixture = binary_fixture(Transform::Truncate(1));
    let projected = Transform::Truncate(1)
        .project(
            "b_trunc",
            &fixture.binary_predicate(
                PredicateOperator::NotStartsWith,
                Datum::binary(vec![0x01, 0x02]),
            ),
        )
        .expect("project must not error on a binary literal");
    assert!(
        projected.is_none(),
        "a binary NOT STARTS WITH literal longer than the width cannot inclusive-project, got {projected:?}"
    );
}

#[test]
fn test_project_truncate_binary_not_starts_with_width_boundaries() {
    let fixture = binary_fixture(Transform::Truncate(1));
    let (op, datum) = projected_binary(
        fixture.binary_predicate(PredicateOperator::NotStartsWith, Datum::binary(vec![0x01])),
    )
    .expect("len == width must project to NotEq");
    assert_eq!(op, PredicateOperator::NotEq);
    assert_eq!(datum, Datum::binary(vec![0x01]));
    let (op, datum) = projected_binary(
        fixture.binary_predicate(PredicateOperator::NotStartsWith, Datum::binary(vec![])),
    )
    .expect("len < width must project to NotStartsWith on the literal");
    assert_eq!(op, PredicateOperator::NotStartsWith);
    assert_eq!(datum, Datum::binary(vec![]));
}

#[test]
fn test_project_truncate_binary_starts_with_width_boundaries() {
    let fixture = binary_fixture(Transform::Truncate(1));
    let (op, datum) = projected_binary(
        fixture.binary_predicate(PredicateOperator::StartsWith, Datum::binary(vec![0x01])),
    )
    .expect("len == width must project to Eq");
    assert_eq!(op, PredicateOperator::Eq);
    assert_eq!(datum, Datum::binary(vec![0x01]));
    let (op, datum) = projected_binary(
        fixture.binary_predicate(PredicateOperator::StartsWith, Datum::binary(vec![])),
    )
    .expect("len < width must project to StartsWith on the literal");
    assert_eq!(op, PredicateOperator::StartsWith);
    assert_eq!(datum, Datum::binary(vec![]));
    let (op, datum) = projected_binary(fixture.binary_predicate(
        PredicateOperator::StartsWith,
        Datum::binary(vec![0x01, 0x02]),
    ))
    .expect("len > width must project to StartsWith on the truncated literal");
    assert_eq!(op, PredicateOperator::StartsWith);
    assert_eq!(datum, Datum::binary(vec![0x01]));
}

#[test]
fn test_truncate_transform_to_type_writes_expected_layout() {
    let truncate = transform_of(&Transform::Truncate(1));
    let binary_view: ArrayRef = Arc::new(BinaryViewArray::from(BINARY_ROWS.to_vec()));
    let out = truncate
        .transform_to_type(&binary_view, &DataType::LargeBinary)
        .expect("binary truncate into LargeBinary must be supported")
        .expect("truncate into LargeBinary must succeed");
    assert_eq!(out.data_type(), &DataType::LargeBinary);
    assert_binary_rows(&out, &[
        Some(vec![]),
        Some(vec![0x01]),
        Some(vec![0x01]),
        Some(vec![0x01]),
        Some(vec![0xff]),
        None,
        Some(vec![0xe4]),
    ]);
    let utf8_view: ArrayRef = Arc::new(StringViewArray::from(STRING_ROWS.to_vec()));
    let out = truncate
        .transform_to_type(&utf8_view, &DataType::Utf8)
        .expect("string truncate into Utf8 must be supported")
        .expect("truncate into Utf8 must succeed");
    assert_eq!(out.data_type(), &DataType::Utf8);
    assert_string_rows(&out, &[
        Some(""),
        Some("i"),
        Some("中"),
        Some("a"),
        Some("🚀"),
        None,
        Some("a"),
    ]);
    assert!(
        truncate
            .transform_to_type(&binary_view, &DataType::Int32)
            .is_none(),
        "unrelated expected layouts must decline the fast path"
    );
}
