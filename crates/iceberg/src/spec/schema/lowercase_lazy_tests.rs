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

use super::Schema;
use crate::spec::{NestedField, PrimitiveType, Type};

fn twin_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "A", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .expect("case-twin columns must build")
}

#[test]
fn test_twin_schema_builds_and_case_sensitive_lookup_finds_each_twin() {
    let schema = twin_schema();
    let lower = schema.field_by_name("a").expect("column a must resolve");
    let upper = schema.field_by_name("A").expect("column A must resolve");
    assert_ne!(lower.id, upper.id);
    assert_eq!(lower.id, 1);
    assert_eq!(upper.id, 2);
}

#[test]
fn test_twin_schema_round_trips_through_serde_json() {
    let schema = twin_schema();
    let json = serde_json::to_string(&schema).expect("twin schema must serialize");
    let parsed: Schema = serde_json::from_str(&json).expect("twin schema must deserialize");
    assert_eq!(parsed, schema);
    assert_eq!(
        parsed.field_by_name("a").expect("column a").id,
        1,
        "deserialized twin schema must keep column a at id 1"
    );
    assert_eq!(
        parsed.field_by_name("A").expect("column A").id,
        2,
        "deserialized twin schema must keep column A at id 2"
    );
}

#[test]
fn test_twin_schema_case_insensitive_lookup_refuses_with_collision_text() {
    let schema = twin_schema();
    let error = schema
        .try_field_by_name_case_insensitive("a")
        .expect_err("case-insensitive lookup on twins must refuse");
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot build lower case index: a and A collide"
    );
    assert!(schema.field_by_name_case_insensitive("A").is_none());
}

#[test]
fn test_twin_schema_clone_and_equality_ignore_cached_index() {
    let schema = twin_schema();
    assert_eq!(schema.clone(), schema);
    assert_eq!(twin_schema(), schema);
}
