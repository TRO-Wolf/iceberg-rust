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

    assert_eq!(schema_accessor_charge(&flat), arc_charge);
    assert_eq!(schema_accessor_charge(&nested), arc_charge + box_charge);
    assert_eq!(
        schema_accessor_charge(&doubly_nested),
        arc_charge + 2 * box_charge
    );
    assert_eq!(schema_accessor_charge(&list), 0);
    assert_eq!(schema_accessor_charge(&map), 0);
    assert_eq!(schema_accessor_charge(&variant), 0);
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
