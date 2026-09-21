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

use super::tests::table_schema_nested;
use crate::spec::values::Map as MapValue;
use crate::spec::{Datum, Literal};

#[test]
fn test_build_accessors_includes_container_and_struct_fields() {
    let schema = table_schema_nested();

    for field_id in [4, 6, 11, 15] {
        assert!(
            schema.accessor_by_field_id(field_id).is_some(),
            "accessor for container field id {field_id} must exist"
        );
    }
}

#[test]
fn test_build_accessors_omits_element_key_value_and_container_nested_ids() {
    let schema = table_schema_nested();

    for field_id in [5, 7, 8, 9, 10, 12, 13, 14] {
        assert!(
            schema.accessor_by_field_id(field_id).is_none(),
            "field id {field_id} inside a list or map must have no accessor"
        );
    }
}

#[test]
fn test_build_accessors() {
    let schema = table_schema_nested();

    let test_struct = crate::spec::Struct::from_iter(vec![
        Some(Literal::string("foo value")),
        Some(Literal::int(1002)),
        Some(Literal::bool(true)),
        Some(Literal::List(vec![
            Some(Literal::string("qux item 1")),
            Some(Literal::string("qux item 2")),
        ])),
        Some(Literal::Map(MapValue::from([(
            Literal::string("quux key 1"),
            Some(Literal::Map(MapValue::from([(
                Literal::string("quux nested key 1"),
                Some(Literal::int(1000)),
            )]))),
        )]))),
        Some(Literal::List(vec![Some(Literal::Struct(
            crate::spec::Struct::from_iter(vec![
                Some(Literal::float(52.509_09)),
                Some(Literal::float(-1.885_249)),
            ]),
        ))])),
        Some(Literal::Struct(crate::spec::Struct::from_iter(vec![
            Some(Literal::string("Testy McTest")),
            Some(Literal::int(33)),
        ]))),
    ]);

    assert_eq!(
        schema
            .accessor_by_field_id(1)
            .unwrap()
            .get(&test_struct)
            .unwrap(),
        Some(Datum::string("foo value"))
    );
    assert_eq!(
        schema
            .accessor_by_field_id(2)
            .unwrap()
            .get(&test_struct)
            .unwrap(),
        Some(Datum::int(1002))
    );
    assert_eq!(
        schema
            .accessor_by_field_id(3)
            .unwrap()
            .get(&test_struct)
            .unwrap(),
        Some(Datum::bool(true))
    );
    assert_eq!(
        schema
            .accessor_by_field_id(16)
            .unwrap()
            .get(&test_struct)
            .unwrap(),
        Some(Datum::string("Testy McTest"))
    );
    assert_eq!(
        schema
            .accessor_by_field_id(17)
            .unwrap()
            .get(&test_struct)
            .unwrap(),
        Some(Datum::int(33))
    );
}
