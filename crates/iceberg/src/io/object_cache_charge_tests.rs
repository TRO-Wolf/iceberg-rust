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

use super::*;
use crate::expr::accessor::StructAccessor;
use crate::spec::{ListType, Map, MapType, NestedField, PrimitiveType, Schema, StructType, Type};

#[derive(Clone, Copy)]
enum NestedWrapper {
    Struct,
    List,
    Map,
}

fn required_field(id: i32, name: &str, field_type: Type) -> Arc<NestedField> {
    Arc::new(NestedField::required(id, name, field_type))
}

fn schema_with_field(field: Arc<NestedField>) -> Schema {
    Schema::builder().with_fields([field]).build().unwrap()
}

fn nested_payload_schema(wrapper: NestedWrapper, payload: &str) -> Schema {
    let payload_field = Arc::new(
        NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::String))
            .with_doc(payload),
    );
    let field_type = match wrapper {
        NestedWrapper::Struct => Type::Struct(StructType::new(vec![payload_field])),
        NestedWrapper::List => Type::List(ListType::new(payload_field)),
        NestedWrapper::Map => Type::Map(MapType::new(
            required_field(3, "key", Type::Primitive(PrimitiveType::String)),
            payload_field,
        )),
    };
    schema_with_field(required_field(1, "root", field_type))
}

#[test]
fn test_schema_accessor_charge_counts_actual_arc_and_box_nodes() {
    let flat = schema_with_field(required_field(
        1,
        "value",
        Type::Primitive(PrimitiveType::String),
    ));
    let nested = schema_with_field(required_field(
        1,
        "outer",
        Type::Struct(StructType::new(vec![required_field(
            2,
            "value",
            Type::Primitive(PrimitiveType::String),
        )])),
    ));
    let doubly_nested = schema_with_field(required_field(
        1,
        "outer",
        Type::Struct(StructType::new(vec![required_field(
            2,
            "inner",
            Type::Struct(StructType::new(vec![required_field(
                3,
                "value",
                Type::Primitive(PrimitiveType::String),
            )])),
        )])),
    ));
    let list = nested_payload_schema(NestedWrapper::List, "doc");
    let map = nested_payload_schema(NestedWrapper::Map, "doc");
    let variant = schema_with_field(required_field(1, "value", Type::Variant));
    let arc_charge = arc_allocation_charge::<StructAccessor>();
    let box_charge = shallow_charge::<StructAccessor>();
    let struct_payload = sequence_charge::<Arc<NestedField>>(1);

    assert_eq!(schema_accessor_charge(&flat), arc_charge);
    assert_eq!(
        schema_accessor_charge(&nested),
        2 * arc_charge + box_charge + struct_payload
    );
    assert_eq!(
        schema_accessor_charge(&doubly_nested),
        3 * arc_charge + 3 * box_charge + 3 * struct_payload
    );
    assert_eq!(schema_accessor_charge(&list), arc_charge);
    assert_eq!(schema_accessor_charge(&map), arc_charge);
    assert_eq!(schema_accessor_charge(&variant), arc_charge);
}

#[test]
fn test_schema_accessor_charge_matches_the_accessor_map_schema_builds() {
    for schema in [
        schema_with_field(required_field(
            1,
            "value",
            Type::Primitive(PrimitiveType::String),
        )),
        nested_payload_schema(NestedWrapper::Struct, "doc"),
        nested_payload_schema(NestedWrapper::List, "doc"),
        nested_payload_schema(NestedWrapper::Map, "doc"),
        Schema::builder()
            .with_fields([
                required_field(1, "id", Type::Primitive(PrimitiveType::Long)),
                Arc::new(NestedField::optional(
                    2,
                    "st",
                    Type::Struct(StructType::new(vec![
                        Arc::new(NestedField::optional(
                            3,
                            "a",
                            Type::Primitive(PrimitiveType::String),
                        )),
                        Arc::new(NestedField::optional(
                            4,
                            "inner",
                            Type::Struct(StructType::new(vec![Arc::new(NestedField::optional(
                                5,
                                "ys",
                                Type::List(ListType::new(required_field(
                                    6,
                                    "element",
                                    Type::Primitive(PrimitiveType::Int),
                                ))),
                            ))])),
                        )),
                    ])),
                )),
                Arc::new(NestedField::optional(
                    7,
                    "mp",
                    Type::Map(MapType::new(
                        required_field(8, "key", Type::Primitive(PrimitiveType::String)),
                        Arc::new(NestedField::optional(
                            9,
                            "value",
                            Type::Primitive(PrimitiveType::Int),
                        )),
                    )),
                )),
            ])
            .build()
            .unwrap(),
    ] {
        let mut expected = 0u64;
        for accessor in schema.accessor_entries() {
            expected += arc_allocation_charge::<StructAccessor>()
                + accessor_type_payload_charge(accessor.r#type());
            let mut wrapped = accessor.inner();
            while let Some(node) = wrapped {
                expected += shallow_charge::<StructAccessor>()
                    + accessor_type_payload_charge(node.r#type());
                wrapped = node.inner();
            }
        }
        assert!(expected > 0);
        assert_eq!(schema_accessor_charge(&schema), expected);
    }
}

#[test]
fn test_schema_accessor_charge_grows_with_every_accessor_map_entry() {
    let one_field = schema_with_field(required_field(
        1,
        "outer",
        Type::Struct(StructType::new(vec![required_field(
            2,
            "a",
            Type::Primitive(PrimitiveType::String),
        )])),
    ));
    let two_fields = schema_with_field(required_field(
        1,
        "outer",
        Type::Struct(StructType::new(vec![
            required_field(2, "a", Type::Primitive(PrimitiveType::String)),
            required_field(3, "b", Type::Primitive(PrimitiveType::String)),
        ])),
    ));
    let with_list = Schema::builder()
        .with_fields([
            required_field(1, "value", Type::Primitive(PrimitiveType::String)),
            required_field(
                2,
                "xs",
                Type::List(ListType::new(required_field(
                    3,
                    "element",
                    Type::Primitive(PrimitiveType::Int),
                ))),
            ),
        ])
        .build()
        .unwrap();
    let without_list = schema_with_field(required_field(
        1,
        "value",
        Type::Primitive(PrimitiveType::String),
    ));

    assert!(schema_accessor_charge(&two_fields) > schema_accessor_charge(&one_field));
    assert!(schema_accessor_charge(&with_list) > schema_accessor_charge(&without_list));
    assert_eq!(with_list.accessor_entries().count(), 2);
}

#[test]
fn test_schema_type_graph_charge_tracks_struct_list_and_map_payloads() {
    for wrapper in [
        NestedWrapper::Struct,
        NestedWrapper::List,
        NestedWrapper::Map,
    ] {
        let short = nested_payload_schema(wrapper, "d");
        let long = nested_payload_schema(wrapper, &"d".repeat(4096));
        assert!(
            schema_type_graph_charge(long.as_struct())
                > schema_type_graph_charge(short.as_struct())
        );
    }
}

#[test]
fn test_schema_charge_tracks_identifier_storage() {
    let field = required_field(1, "value", Type::Primitive(PrimitiveType::Long));
    let plain = schema_with_field(field.clone());
    let identified = Schema::builder()
        .with_fields([field])
        .with_identifier_field_ids([1])
        .build()
        .unwrap();

    assert!(schema_charge(&identified) > schema_charge(&plain));
}

#[test]
fn test_map_literal_charge_tracks_duplicate_keys_and_single_values() {
    let short_key = Literal::Map(Map::from([(Literal::string("k"), None)]));
    let long_key = Literal::Map(Map::from([(Literal::string("k".repeat(4096)), None)]));
    assert_eq!(
        literal_payload_charge(&long_key) - literal_payload_charge(&short_key),
        2 * 4095
    );

    let short_value = Literal::Map(Map::from([(
        Literal::string("k"),
        Some(Literal::string("v")),
    )]));
    let long_value = Literal::Map(Map::from([(
        Literal::string("k"),
        Some(Literal::string("v".repeat(4096))),
    )]));
    assert_eq!(
        literal_payload_charge(&long_value) - literal_payload_charge(&short_value),
        4095
    );
}
