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

#[test]
fn test_variant_rejected_by_value_producing_transforms() {
    for transform in [
        Transform::Identity,
        Transform::Bucket(16),
        Transform::Truncate(4),
        Transform::Year,
        Transform::Month,
        Transform::Day,
        Transform::Hour,
    ] {
        let error = transform
            .result_type(&Type::Variant)
            .expect_err("variant must not be a transform input");
        assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
        assert!(
            error.message().contains("variant"),
            "{transform} rejection must name the variant input, got: {}",
            error.message()
        );
    }
}

#[test]
fn test_from_str_rejects_zero_bucket_and_zero_truncate() {
    let error = "bucket[0]"
        .parse::<Transform>()
        .expect_err("bucket[0] must be rejected at parse");
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("Invalid number of buckets: 0 (must be > 0)"),
        "message must match the Java precondition text, got: {}",
        error.message()
    );

    let error = "truncate[0]"
        .parse::<Transform>()
        .expect_err("truncate[0] must be rejected at parse");
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("Invalid truncate width: 0 (must be > 0)"),
        "message must match the Java precondition text, got: {}",
        error.message()
    );
}

#[test]
fn test_from_str_rejects_parameters_above_java_int_max() {
    for input in [
        "bucket[2147483648]",
        "truncate[2147483648]",
        "bucket[4294967296]",
        "truncate[4294967296]",
    ] {
        let error = input
            .parse::<Transform>()
            .expect_err("parameters above the Java int maximum must be rejected at parse");
        assert_eq!(
            error.kind(),
            crate::ErrorKind::DataInvalid,
            "input: {input}"
        );
    }
}

#[test]
fn test_from_str_accepts_boundary_legal_parameters() {
    assert_eq!(
        "bucket[1]"
            .parse::<Transform>()
            .expect("bucket[1] is legal"),
        Transform::Bucket(1)
    );
    assert_eq!(
        "bucket[2147483647]"
            .parse::<Transform>()
            .expect("bucket[i32::MAX] is legal"),
        Transform::Bucket(2147483647)
    );
    assert_eq!(
        "truncate[1]"
            .parse::<Transform>()
            .expect("truncate[1] is legal"),
        Transform::Truncate(1)
    );
    assert_eq!(
        "truncate[2147483647]"
            .parse::<Transform>()
            .expect("truncate[i32::MAX] is legal"),
        Transform::Truncate(2147483647)
    );
}

#[test]
fn test_serde_rejects_zero_parameter_transforms() {
    for json in [
        r#""bucket[0]""#,
        r#""truncate[0]""#,
        r#""bucket[2147483648]""#,
    ] {
        assert!(
            serde_json::from_str::<Transform>(json).is_err(),
            "{json} must fail deserialization"
        );
    }
}

#[test]
fn test_variant_accepted_by_void_and_unknown_transforms() {
    assert_eq!(
        Transform::Void
            .result_type(&Type::Variant)
            .expect("void accepts any source type"),
        Type::Variant,
    );
    assert_eq!(
        Transform::Unknown
            .result_type(&Type::Variant)
            .expect("unknown accepts any source type"),
        Type::Primitive(PrimitiveType::String),
    );
}
