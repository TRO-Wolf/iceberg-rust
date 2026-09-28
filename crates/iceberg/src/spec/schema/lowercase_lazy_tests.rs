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
use crate::spec::{
    MappedField, NameMapping, NestedField, NullOrder, PartitionSpec, PrimitiveType, SortDirection,
    SortField, SortOrder, StructType, Transform, Type,
};

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

#[test]
fn test_twin_schema_partition_spec_builder_binds_exact_source_id() {
    let spec = PartitionSpec::builder(twin_schema())
        .add_partition_field("A", "a_part", Transform::Identity)
        .expect("partition field on twin A must resolve")
        .build()
        .expect("partition spec over twins must build");
    assert_eq!(spec.fields().len(), 1);
    assert_eq!(spec.fields()[0].source_id, 2);
    assert_eq!(spec.fields()[0].name, "a_part");
}

#[test]
fn test_twin_schema_sort_order_builder_builds_for_source_id() {
    let schema = twin_schema();
    let order = SortOrder::builder()
        .with_sort_field(
            SortField::builder()
                .source_id(2)
                .direction(SortDirection::Ascending)
                .null_order(NullOrder::First)
                .transform(Transform::Identity)
                .build(),
        )
        .build(&schema)
        .expect("sort order over twins must build");
    assert_eq!(order.fields.len(), 1);
    assert_eq!(order.fields[0].source_id, 2);
}

#[test]
fn test_twin_schema_name_mapping_round_trip_keeps_both_twins() {
    let mapping = NameMapping::new(vec![
        MappedField::new(Some(1), vec!["a".to_string()], vec![]),
        MappedField::new(Some(2), vec!["A".to_string()], vec![]),
    ]);
    let json = serde_json::to_string(&mapping).expect("mapping must serialize");
    let parsed: NameMapping = serde_json::from_str(&json).expect("mapping must deserialize");
    assert_eq!(parsed, mapping);
    assert_eq!(parsed.fields().len(), 2);
    assert_eq!(parsed.fields()[0].names(), &["a".to_string()]);
    assert_eq!(parsed.fields()[0].field_id(), Some(1));
    assert_eq!(parsed.fields()[1].names(), &["A".to_string()]);
    assert_eq!(parsed.fields()[1].field_id(), Some(2));
}

#[test]
fn test_nested_twin_pair_builds_and_insensitive_lookup_refuses_dotted() {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(
                1,
                "s",
                Type::Struct(StructType::new(vec![
                    NestedField::required(2, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(3, "A", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
        ])
        .build()
        .expect("nested twin columns must build");
    assert_eq!(schema.field_by_name("s.a").expect("column s.a").id, 2);
    assert_eq!(schema.field_by_name("s.A").expect("column s.A").id, 3);
    let error = schema
        .try_field_by_name_case_insensitive("s.a")
        .expect_err("insensitive lookup on nested twins must refuse");
    assert_eq!(error.kind(), crate::ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "Cannot build lower case index: s.a and s.A collide"
    );
}

#[test]
fn test_triple_collision_reports_smallest_id_pair_first_deterministically() {
    for _ in 0..20 {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "ab", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "Ab", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(3, "AB", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .expect("triple-collision columns must build");
        let error = schema
            .try_field_by_name_case_insensitive("AB")
            .expect_err("insensitive lookup on triple collision must refuse");
        assert_eq!(
            error.message(),
            "Cannot build lower case index: ab and Ab collide"
        );
    }
}
