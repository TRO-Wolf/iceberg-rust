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

//! WG3-L2 pins: [`PartitionSpec::partition_to_path`] is TOTAL. An inconsistent
//! `(spec, schema, tuple)` triple renders `name=null` for the offending field and warns, and it
//! never aborts. Four abort vectors were reachable before this change:
//!
//! | # | input | pre-change abort |
//! |---|---|---|
//! | V1 | tuple shorter than the spec | `data[i]` index out of bounds |
//! | V2 | source column absent from the schema | `partition_type(..).unwrap()` |
//! | V3 | non-primitive partition-field type + a primitive value | `as_primitive_type().unwrap()` |
//! | V4 | value literal kind incompatible with the field type | `Display for Datum`'s `unreachable!()` |
//!
//! Java is LENIENT for V1: `PartitionData.get(pos)` returns `null` past the end of the tuple and
//! `Transform.toHumanString(type, null)` renders `"null"`. Java throws for the other three.
//! [`PartitionSpec::try_partition_to_path`] surfaces all four as typed errors.

use std::sync::Arc;

use super::*;
use crate::spec::{Datum, Literal, PrimitiveLiteral, PrimitiveType, Type};

/// `identity(x: long)` + `identity(y: long)` over a two-column schema.
fn two_field_spec() -> (SchemaRef, PartitionSpec) {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("two-column schema must build"),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("x", "x", Transform::Identity)
        .expect("identity(x) is a legal partition field")
        .add_partition_field("y", "y", Transform::Identity)
        .expect("identity(y) is a legal partition field")
        .build()
        .expect("the two-field spec must build");
    (schema, spec)
}

// NULL partition values stay legal: a NULL tuple slot is a first-class Iceberg value.

/// A `PartitionKey` carrying a NULL value renders `name=null` on both paths. Java renders a
/// null partition value as the literal `"null"` (`Transform.toHumanString(type, null)`).
#[test]
fn partition_key_new_accepts_null_value() {
    let (schema, spec) = two_field_spec();
    let data = Struct::from_iter([Some(Literal::long(5)), None]);
    let key = PartitionKey::new(spec.clone(), schema.clone(), data.clone())
        .expect("PartitionKey::new: valid partition tuple");

    assert_eq!(key.to_path(), "x=5/y=null");
    assert_eq!(
        spec.try_partition_to_path(&data, schema)
            .expect("a NULL partition value is legal, not an anomaly"),
        "x=5/y=null"
    );
}

/// `PartitionKey::new` rejects an invalid triple (a short non-void tuple) with a typed error.
#[test]
fn partition_key_new_rejects_short_non_void_tuple() {
    let (schema, spec) = two_field_spec();
    let data = Struct::from_iter([Some(Literal::long(5))]); // missing y
    let err = PartitionKey::new(spec, schema, data)
        .expect_err("a short non-void tuple must not construct a PartitionKey");
    assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
}

/// Unit 3: `PartitionKey::new` rejects an incompatible literal kind.
#[test]
fn partition_key_new_rejects_incompatible_literal() {
    let (schema, spec) = two_field_spec();
    let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::string("7"))]);
    let err = PartitionKey::new(spec, schema, data)
        .expect_err("a String in a Long partition slot must not construct a PartitionKey");
    assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
}

/// Unit 3: all-void + empty tuple remains constructible (the void trap).
#[test]
fn partition_key_new_accepts_all_void_empty_tuple() {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("one-column schema must build"),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("x", "x_void", Transform::Void)
        .expect("void(x) is a legal partition field")
        .build()
        .expect("the all-void spec must build");
    let key = PartitionKey::new(spec, schema, Struct::empty())
        .expect("all-void + empty tuple is a legitimate PartitionKey");
    assert_eq!(key.to_path(), "x_void=null");
}

// V1 — tuple shorter than the spec.

/// V1: a tuple shorter than the spec renders the missing fields as `null` (Java's past-end
/// `PartitionData.get` leniency) instead of indexing out of bounds.
#[test]
fn test_partition_to_path_short_tuple_renders_null_instead_of_aborting() {
    let (schema, spec) = two_field_spec();
    let data = Struct::from_iter([Some(Literal::long(5))]);

    assert_eq!(spec.partition_to_path(&data, schema), "x=5/y=null");
}

/// The same short tuple is a typed `DataInvalid` on the fallible path.
#[test]
fn test_try_partition_to_path_short_tuple_errors() {
    let (schema, spec) = two_field_spec();
    let data = Struct::from_iter([Some(Literal::long(5))]);

    let err = spec
        .try_partition_to_path(&data, schema)
        .expect_err("a tuple shorter than the spec must be a typed error");
    assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        err.message().contains("has 1 value(s)"),
        "unexpected message: {}",
        err.message()
    );
}

// V2 — source column absent from the schema (the spec-evolved commit-path shape).

/// Rendering a spec against a schema that dropped one of its source columns renders THAT field
/// as `null` and still renders the others. Java's `partitionType()` substitutes `UnknownType`.
#[test]
fn test_partition_to_path_missing_source_column_renders_null_per_field() {
    let (_schema, spec) = two_field_spec();
    // The evolved schema dropped `x` (source id 1) and kept `y` (source id 2).
    let evolved: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("evolved schema must build"),
    );
    let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::long(7))]);

    assert_eq!(
        spec.partition_to_path(&data, evolved.clone()),
        "x=null/y=7",
        "the field whose source survived must still render its value"
    );
    let err = spec
        .try_partition_to_path(&data, evolved)
        .expect_err("a dropped source column must be a typed error on the fallible path");
    assert_eq!(err.kind(), crate::ErrorKind::Unexpected);
}

// V3 — non-primitive partition-field type (a legal `void` over a non-primitive source).

/// `void` over a STRUCT source is a legal partition field, so the partition type can be
/// non-primitive. A primitive value in that slot renders `null` instead of aborting.
#[test]
fn test_partition_to_path_non_primitive_field_type_renders_null() {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(
                    1,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(2, "inner", Type::Primitive(PrimitiveType::Long))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .expect("struct-column schema must build"),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("s", "s_void", Transform::Void)
        .expect("void over a non-primitive source is legal")
        .build()
        .expect("the void spec must build");
    let data = Struct::from_iter([Some(Literal::long(5))]);

    assert_eq!(spec.partition_to_path(&data, schema.clone()), "s_void=null");
    let err = spec
        .try_partition_to_path(&data, schema)
        .expect_err("a primitive value under a non-primitive field type must be a typed error");
    assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
}

// V4 — value literal kind incompatible with the partition-field type.

/// An `Int` literal in a `Long`-typed partition slot renders `null`. `PrimitiveType::compatible`
/// decides, the same predicate the commit-path `validate_partition_value` uses.
#[test]
fn test_partition_to_path_incompatible_literal_renders_null() {
    let (schema, spec) = two_field_spec();
    let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::int(7))]);

    assert_eq!(spec.partition_to_path(&data, schema.clone()), "x=5/y=null");
    let err = spec
        .try_partition_to_path(&data, schema)
        .expect_err("an incompatible literal kind must be a typed error");
    assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        err.message().contains("not compatible"),
        "unexpected message: {}",
        err.message()
    );
}

/// A NON-primitive literal in a primitive slot already rendered `null` (never aborted); the
/// fallible path surfaces it, matching `SnapshotProducer::validate_partition_value`'s posture.
#[test]
fn test_partition_to_path_non_primitive_literal_renders_null() {
    let (schema, spec) = two_field_spec();
    let nested = Struct::from_iter([Some(Literal::long(1))]);
    let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::Struct(nested))]);

    assert_eq!(spec.partition_to_path(&data, schema.clone()), "x=5/y=null");
    let err = spec
        .try_partition_to_path(&data, schema)
        .expect_err("a non-primitive partition literal must be a typed error");
    assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        err.message().contains("primitive literal"),
        "unexpected message: {}",
        err.message()
    );
}

// The void trap: an all-`void` spec is `is_unpartitioned() == true`, so `(void_spec,
// Struct::empty())` is a LEGITIMATE pair that a naive arity rule would reject.

/// TRAP: an all-`void` spec reports `is_unpartitioned() == true`, and callers legitimately hand
/// it an EMPTY tuple. A missing value for a `void` field carries no information, so neither path
/// treats it as an anomaly.
///
/// MUTATION (drop the `void` carve-out): this test and the mixed-void test go RED while
/// `test_try_partition_to_path_short_tuple_errors` stays GREEN.
#[test]
fn test_all_void_spec_with_empty_tuple_is_not_an_anomaly() {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("one-column schema must build"),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("x", "x_void", Transform::Void)
        .expect("void(x) is a legal partition field")
        .build()
        .expect("the all-void spec must build");
    assert!(
        spec.is_unpartitioned(),
        "fixture sanity: an all-void spec reports unpartitioned"
    );

    let data = Struct::empty();
    assert_eq!(spec.partition_to_path(&data, schema.clone()), "x_void=null");
    assert_eq!(
        spec.try_partition_to_path(&data, schema)
            .expect("an all-void spec paired with an empty tuple is legitimate"),
        "x_void=null"
    );
}

/// The MIXED shape (`identity(x)` + `void(y)`) with a tuple covering only `x`: the identity
/// field renders its value, the past-the-end `void` field renders `null`, and neither path errors.
#[test]
fn test_partition_to_path_mixed_void_short_tuple_is_not_an_anomaly() {
    let schema: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("two-column schema must build"),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .add_partition_field("x", "x", Transform::Identity)
        .expect("identity(x) is a legal partition field")
        .add_partition_field("y", "y_void", Transform::Void)
        .expect("void(y) is a legal partition field")
        .build()
        .expect("the mixed spec must build");
    assert!(
        !spec.is_unpartitioned(),
        "fixture sanity: a spec with a non-void field is partitioned"
    );

    let data = Struct::from_iter([Some(Literal::long(5))]);
    assert_eq!(
        spec.partition_to_path(&data, schema.clone()),
        "x=5/y_void=null"
    );
    assert_eq!(
        spec.try_partition_to_path(&data, schema)
            .expect("a missing value for a void field is not an anomaly"),
        "x=5/y_void=null"
    );
}

// The two paths agree on well-formed input.

/// On a self-consistent triple the fallible path returns EXACTLY the string the total path
/// renders — the total path's leniency is confined to the anomaly branches.
#[test]
fn test_try_partition_to_path_matches_partition_to_path_when_well_formed() {
    let (schema, spec) = two_field_spec();
    let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::long(7))]);

    let total = spec.partition_to_path(&data, schema.clone());
    let fallible = spec
        .try_partition_to_path(&data, schema)
        .expect("a well-formed triple must not error");
    assert_eq!(total, fallible);
    assert_eq!(total, "x=5/y=7");
}

// Drift alarm: every pair `PrimitiveType::compatible` accepts must RENDER.

/// The anomaly guard admits exactly the pairs `PrimitiveType::compatible` accepts, and every
/// admitted pair must survive `Datum`'s `Display`, whose `(_, _)` arm is `unreachable!()`. This
/// test runs the whole accepted matrix, so a dropped `Display` arm PANICS here. The converse is
/// safe: `compatible` is narrower, so a rejected pair renders `null`.
#[test]
fn test_every_compatible_type_literal_pair_renders() {
    let types = [
        PrimitiveType::Boolean,
        PrimitiveType::Int,
        PrimitiveType::Long,
        PrimitiveType::Float,
        PrimitiveType::Double,
        PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        },
        PrimitiveType::Date,
        PrimitiveType::Time,
        PrimitiveType::Timestamp,
        PrimitiveType::Timestamptz,
        PrimitiveType::TimestampNs,
        PrimitiveType::TimestamptzNs,
        PrimitiveType::String,
        PrimitiveType::Uuid,
        PrimitiveType::Fixed(4),
        PrimitiveType::Binary,
    ];
    let literals = [
        PrimitiveLiteral::Boolean(true),
        PrimitiveLiteral::Int(1),
        PrimitiveLiteral::Long(1),
        PrimitiveLiteral::Float(1.0.into()),
        PrimitiveLiteral::Double(1.0.into()),
        PrimitiveLiteral::String("s".to_string()),
        PrimitiveLiteral::Binary(vec![1, 2, 3, 4]),
        PrimitiveLiteral::Int128(1),
        PrimitiveLiteral::UInt128(1),
        PrimitiveLiteral::AboveMax,
        PrimitiveLiteral::BelowMin,
    ];

    let mut rendered = 0usize;
    for ty in &types {
        for literal in &literals {
            if ty.compatible(literal) {
                // Panics loudly (and names the pair) if `Display for Datum` cannot render it.
                let _ = Datum::new(ty.clone(), literal.clone()).to_human_string();
                rendered += 1;
            }
        }
    }
    assert_eq!(
        rendered, 16,
        "the accepted matrix changed shape — re-check the guard against `Display for Datum`"
    );
}
