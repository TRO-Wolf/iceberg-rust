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

use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::field_id_stamping::{
    add_fallback_field_ids_to_arrow_schema, apply_name_mapping_to_arrow_schema,
    stamped_field_id_map,
};
use crate::spec::{
    ListType, MappedField, NameMapping, NestedField, PrimitiveType, Schema, StructType, Type,
};

fn int_field(id: i32, name: &str) -> NestedField {
    NestedField::optional(id, name, Type::Primitive(PrimitiveType::Int))
}

fn struct_of(fields: Vec<NestedField>) -> Type {
    Type::Struct(StructType::new(fields.into_iter().map(Arc::new).collect()))
}

fn table(fields: Vec<NestedField>) -> Schema {
    Schema::builder()
        .with_fields(fields.into_iter().map(Arc::new).collect::<Vec<_>>())
        .build()
        .expect("build the table schema")
}

fn file_int(name: &str) -> Field {
    Field::new(name, DataType::Int32, true)
}

fn file_struct(name: &str, children: Vec<Field>) -> Field {
    Field::new(name, DataType::Struct(Fields::from(children)), true)
}

fn file_schema(fields: Vec<Field>) -> ArrowSchemaRef {
    Arc::new(ArrowSchema::new(fields))
}

fn id_of(field: &Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .and_then(|id| id.parse().ok())
}

fn field_at<'a>(schema: &'a ArrowSchema, path: &[&str]) -> &'a Field {
    let (first, rest) = path.split_first().expect("a non-empty path");
    let mut current = schema
        .field_with_name(first)
        .expect("top-level field present");
    for name in rest {
        current = match current.data_type() {
            DataType::Struct(children) => children
                .iter()
                .find(|child| child.name() == name)
                .expect("struct child present")
                .as_ref(),
            DataType::List(element) => element.as_ref(),
            other => panic!("cannot descend into {other}"),
        };
    }
    current
}

fn mapped(id: i32, name: &str, fields: Vec<MappedField>) -> MappedField {
    MappedField::new(Some(id), vec![name.to_string()], fields)
}

#[test]
fn nested_name_mapping_stamps_struct_children_and_list_elements() {
    let file = file_schema(vec![
        Field::new("id", DataType::Int32, false),
        file_struct("s", vec![
            file_int("d"),
            file_struct("t", vec![file_int("x")]),
        ]),
        Field::new(
            "l",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(Fields::from(vec![file_int("y")])),
                true,
            ))),
            true,
        ),
    ]);
    let mapping = NameMapping::new(vec![
        mapped(1, "id", vec![]),
        mapped(2, "s", vec![
            mapped(3, "d", vec![]),
            mapped(4, "t", vec![mapped(5, "x", vec![])]),
        ]),
        mapped(6, "l", vec![mapped(7, "element", vec![mapped(
            8,
            "y",
            vec![],
        )])]),
    ]);
    let stamped = apply_name_mapping_to_arrow_schema(file, &mapping).expect("apply the mapping");
    assert_eq!(id_of(field_at(&stamped, &["s"])), Some(2));
    assert_eq!(id_of(field_at(&stamped, &["s", "d"])), Some(3));
    assert_eq!(id_of(field_at(&stamped, &["s", "t"])), Some(4));
    assert_eq!(id_of(field_at(&stamped, &["s", "t", "x"])), Some(5));
    assert_eq!(id_of(field_at(&stamped, &["l", "item"])), Some(7));
    assert_eq!(id_of(field_at(&stamped, &["l", "item", "y"])), Some(8));
}

#[test]
fn nested_name_mapping_leaves_an_unmapped_child_without_an_id() {
    let file = file_schema(vec![
        Field::new("id", DataType::Int32, false),
        file_struct("s", vec![file_int("d"), file_int("unmapped")]),
    ]);
    let mapping = NameMapping::new(vec![
        mapped(1, "id", vec![]),
        mapped(2, "s", vec![mapped(3, "d", vec![])]),
    ]);
    let stamped = apply_name_mapping_to_arrow_schema(file, &mapping).expect("apply the mapping");
    assert_eq!(id_of(field_at(&stamped, &["s", "d"])), Some(3));
    assert_eq!(id_of(field_at(&stamped, &["s", "unmapped"])), None);
}

#[test]
fn fallback_stamps_a_nested_child_by_name_and_position() {
    let file = file_schema(vec![
        Field::new("id", DataType::Int32, false),
        file_struct("s", vec![file_int("d"), file_int("e")]),
    ]);
    let schema = table(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(
            2,
            "s",
            struct_of(vec![int_field(7, "d"), int_field(9, "e")]),
        ),
    ]);
    let stamped = add_fallback_field_ids_to_arrow_schema(&file, &schema).expect("stamp ids");
    assert_eq!(id_of(field_at(&stamped, &["id"])), Some(1));
    assert_eq!(id_of(field_at(&stamped, &["s"])), Some(2));
    assert_eq!(id_of(field_at(&stamped, &["s", "d"])), Some(7));
    assert_eq!(id_of(field_at(&stamped, &["s", "e"])), Some(9));
}

#[test]
fn fallback_gives_an_unmatched_nested_child_its_preorder_counter() {
    let file = file_schema(vec![
        Field::new("id", DataType::Int32, false),
        file_struct("s", vec![file_struct("t", vec![file_int("d")])]),
    ]);
    let schema = table(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(
            2,
            "s",
            struct_of(vec![NestedField::optional(
                3,
                "t",
                struct_of(vec![int_field(5, "k")]),
            )]),
        ),
    ]);
    let stamped = add_fallback_field_ids_to_arrow_schema(&file, &schema).expect("stamp ids");
    assert_eq!(id_of(field_at(&stamped, &["s", "t"])), Some(3));
    assert_eq!(id_of(field_at(&stamped, &["s", "t", "d"])), Some(4));
}

#[test]
fn fallback_matches_a_list_element_structurally() {
    let file = file_schema(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "l",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(Fields::from(vec![file_int("d")])),
                false,
            ))),
            true,
        ),
    ]);
    let element = NestedField::list_element(6, struct_of(vec![int_field(8, "d")]), true);
    let schema = table(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(
            2,
            "l",
            Type::List(ListType {
                element_field: element.into(),
            }),
        ),
    ]);
    let stamped = add_fallback_field_ids_to_arrow_schema(&file, &schema).expect("stamp ids");
    assert_eq!(id_of(field_at(&stamped, &["l", "item"])), Some(6));
    assert_eq!(id_of(field_at(&stamped, &["l", "item", "d"])), Some(8));
}

#[test]
fn fallback_counter_that_collides_with_a_claimed_id_yields_no_id() {
    let file = file_schema(vec![
        Field::new("id", DataType::Int32, false),
        file_struct("s", vec![file_int("x"), file_int("y")]),
        file_struct("t", vec![file_int("b")]),
    ]);
    let schema = table(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "s", struct_of(vec![int_field(4, "a")])),
        NestedField::optional(3, "t", struct_of(vec![int_field(5, "b")])),
    ]);
    let stamped = add_fallback_field_ids_to_arrow_schema(&file, &schema).expect("stamp ids");
    assert_eq!(id_of(field_at(&stamped, &["s", "x"])), Some(4));
    assert_eq!(id_of(field_at(&stamped, &["s", "y"])), None);
    assert_eq!(id_of(field_at(&stamped, &["t", "b"])), Some(5));
}

#[test]
fn fallback_keeps_the_top_level_counter_and_name_match() {
    let file = file_schema(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("renamed", DataType::Int32, true),
    ]);
    let schema = table(vec![int_field(1, "a"), int_field(2, "b")]);
    let stamped = add_fallback_field_ids_to_arrow_schema(&file, &schema).expect("stamp ids");
    assert_eq!(id_of(field_at(&stamped, &["a"])), Some(1));
    assert_eq!(id_of(field_at(&stamped, &["renamed"])), Some(2));
}

#[test]
fn stamped_field_id_map_indexes_stamped_leaves_at_every_depth() {
    let file = file_schema(vec![
        Field::new("b", DataType::Int32, true),
        file_struct("s", vec![file_int("x"), file_int("d")]),
        Field::new("a", DataType::Int32, true),
    ]);
    let mapping = NameMapping::new(vec![
        mapped(1, "a", vec![]),
        mapped(2, "b", vec![]),
        mapped(3, "s", vec![mapped(4, "d", vec![])]),
    ]);
    let stamped = apply_name_mapping_to_arrow_schema(file, &mapping).expect("apply the mapping");
    let map = stamped_field_id_map(&stamped);
    assert_eq!(map.get(&2), Some(&0));
    assert_eq!(map.get(&4), Some(&2));
    assert_eq!(map.get(&1), Some(&3));
    assert_eq!(map.len(), 3);
    let fallback = add_fallback_field_ids_to_arrow_schema(
        &file_schema(vec![file_int("a"), file_int("b")]),
        &table(vec![int_field(2, "a"), int_field(3, "b")]),
    )
    .expect("stamp ids");
    let map = stamped_field_id_map(&fallback);
    assert_eq!(map.get(&2), Some(&0));
    assert_eq!(map.get(&3), Some(&1));
}
