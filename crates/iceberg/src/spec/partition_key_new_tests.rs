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

//! Constructor pins that must not grow `partition.rs` (legacy ceiling).

use std::sync::Arc;

use super::{
    Literal, NestedField, PartitionKey, PartitionSpec, PrimitiveType, Schema, SchemaRef, Struct,
    Transform, Type,
};

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
        .with_spec_id(1)
        .add_partition_field("x", "x", Transform::Identity)
        .expect("identity(x)")
        .add_partition_field("y", "y", Transform::Identity)
        .expect("identity(y)")
        .build()
        .expect("two-field spec must build");
    (schema, spec)
}

/// Spec bound to S, constructor called with a schema that dropped a source column.
/// Live kind is `Unexpected`, same as `try_partition_to_path`. Not `DataInvalid`.
#[test]
fn partition_key_new_dropped_source_column_is_unexpected() {
    let (_schema, spec) = two_field_spec();
    let evolved: SchemaRef = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("evolved schema must build"),
    );
    let data = Struct::from_iter([Some(Literal::long(5)), Some(Literal::long(7))]);
    let err = PartitionKey::new(spec, evolved, data)
        .expect_err("dropped source column must not construct a PartitionKey");
    assert_eq!(err.kind(), crate::ErrorKind::Unexpected);
}
