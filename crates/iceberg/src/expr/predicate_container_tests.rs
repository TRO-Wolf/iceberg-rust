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

use super::predicate::tests::test_bound_predicate_serialize_diserialize;
use crate::ErrorKind;
use crate::expr::{Bind, Reference};
use crate::spec::{
    Datum, ListType, MapType, NestedField, PrimitiveType, Schema, SchemaRef, StructType, Type,
};

fn table_schema_with_containers() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(
                    2,
                    "xs",
                    Type::List(ListType::new(
                        NestedField::optional(3, "element", Type::Primitive(PrimitiveType::Int))
                            .into(),
                    )),
                )
                .into(),
                NestedField::optional(
                    4,
                    "mp",
                    Type::Map(MapType::new(
                        NestedField::required(5, "key", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::optional(6, "value", Type::Primitive(PrimitiveType::Int))
                            .into(),
                    )),
                )
                .into(),
                NestedField::optional(
                    7,
                    "person",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(8, "name", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::required(9, "age", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    )
}

#[test]
fn test_bind_is_null_on_container_columns() {
    let schema = table_schema_with_containers();

    for (column, expected) in [
        ("xs", "xs IS NULL"),
        ("mp", "mp IS NULL"),
        ("person", "person IS NULL"),
    ] {
        let bound = Reference::new(column)
            .is_null()
            .bind(schema.clone(), true)
            .unwrap_or_else(|e| panic!("`{column} IS NULL` must bind: {e}"));
        assert_eq!(&format!("{bound}"), expected);
        test_bound_predicate_serialize_diserialize(bound);

        let bound_not = Reference::new(column)
            .is_not_null()
            .bind(schema.clone(), true)
            .unwrap_or_else(|e| panic!("`{column} IS NOT NULL` must bind: {e}"));
        assert_eq!(format!("{bound_not}"), format!("{column} IS NOT NULL"));
        test_bound_predicate_serialize_diserialize(bound_not);
    }
}

#[test]
fn test_bind_comparison_on_container_column_fails_at_bind() {
    let schema = table_schema_with_containers();

    for (column, datum) in [
        ("xs", Datum::int(1)),
        ("mp", Datum::int(1)),
        ("person", Datum::int(1)),
    ] {
        let error = Reference::new(column)
            .equal_to(datum)
            .bind(schema.clone(), true)
            .expect_err(&format!("`{column} = <literal>` must fail to bind"));
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("Can't convert"),
            "`{column} = <literal>` must fail in literal conversion, got: {}",
            error.message()
        );
    }
}

#[test]
fn test_bind_is_null_required_leaf_under_optional_parent_does_not_fold() {
    let schema = table_schema_with_containers();
    let bound = Reference::new("person.age")
        .is_null()
        .bind(schema, true)
        .expect("`person.age IS NULL` must bind");
    assert_eq!(&format!("{bound}"), "person.age IS NULL");
}

#[test]
fn test_bind_is_not_null_required_leaf_under_optional_parent_does_not_fold() {
    let schema = table_schema_with_containers();
    let bound = Reference::new("person.age")
        .is_not_null()
        .bind(schema, true)
        .expect("`person.age IS NOT NULL` must bind");
    assert_eq!(&format!("{bound}"), "person.age IS NOT NULL");
}
