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

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::record_batch_predicate::evaluate_predicate_to_mask;
use super::record_batch_predicate::tests::two_valued;
use crate::expr::{Bind, Reference};
use crate::spec::{Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};

fn container_test_schema() -> SchemaRef {
    let int_list = |id: i32| {
        Type::List(crate::spec::ListType::new(
            NestedField::optional(id, "element", Type::Primitive(PrimitiveType::Int)).into(),
        ))
    };
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(
                    10,
                    "st",
                    Type::Struct(crate::spec::StructType::new(vec![
                        NestedField::optional(11, "a", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::optional(12, "b", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
                NestedField::optional(13, "xs", int_list(14)).into(),
                NestedField::optional(
                    15,
                    "mp",
                    Type::Map(crate::spec::MapType::new(
                        NestedField::required(16, "key", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::optional(17, "value", Type::Primitive(PrimitiveType::Int))
                            .into(),
                    )),
                )
                .into(),
                NestedField::optional(
                    18,
                    "deep",
                    Type::Struct(crate::spec::StructType::new(vec![
                        NestedField::optional(
                            19,
                            "inner",
                            Type::Struct(crate::spec::StructType::new(vec![
                                NestedField::optional(
                                    20,
                                    "x",
                                    Type::Primitive(PrimitiveType::String),
                                )
                                .into(),
                                NestedField::optional(21, "ys", int_list(22)).into(),
                            ])),
                        )
                        .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .expect("build the container pin schema"),
    )
}

fn container_truth_batch() -> RecordBatch {
    let fid = |id: i32| HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]);

    let a_field = Arc::new(Field::new("a", DataType::Utf8, true).with_metadata(fid(11)));
    let b_field = Arc::new(Field::new("b", DataType::Int32, true).with_metadata(fid(12)));
    let st_field = Field::new(
        "st",
        DataType::Struct(Fields::from([a_field.clone(), b_field.clone()])),
        true,
    )
    .with_metadata(fid(10));
    let st_col = Arc::new(arrow_array::StructArray::new(
        Fields::from([a_field, b_field]),
        vec![
            Arc::new(arrow_array::StringArray::from(vec![
                Some("a1"),
                None,
                None,
                Some("a4"),
            ])) as ArrayRef,
            Arc::new(arrow_array::Int32Array::from(vec![
                Some(1),
                None,
                None,
                Some(4),
            ])) as ArrayRef,
        ],
        Some(NullBuffer::from(vec![true, false, true, true])),
    )) as ArrayRef;

    let xs_element = Arc::new(Field::new("element", DataType::Int32, true).with_metadata(fid(14)));
    let xs_field =
        Field::new("xs", DataType::List(xs_element.clone()), true).with_metadata(fid(13));
    let xs_col = Arc::new(arrow_array::ListArray::new(
        xs_element,
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0i32, 2, 2, 2, 4])),
        Arc::new(arrow_array::Int32Array::from(vec![1, 2, 4, 5])),
        Some(NullBuffer::from(vec![true, false, true, true])),
    )) as ArrayRef;

    let key_field = Arc::new(Field::new("key", DataType::Utf8, false).with_metadata(fid(16)));
    let value_field = Arc::new(Field::new("value", DataType::Int32, true).with_metadata(fid(17)));
    let entries_field = Arc::new(Field::new(
        "entries",
        DataType::Struct(Fields::from([key_field.clone(), value_field.clone()])),
        false,
    ));
    let mp_field =
        Field::new("mp", DataType::Map(entries_field.clone(), false), true).with_metadata(fid(15));
    let mp_entries = arrow_array::StructArray::new(
        Fields::from([key_field, value_field]),
        vec![
            Arc::new(arrow_array::StringArray::from(vec!["k", "k2"])) as ArrayRef,
            Arc::new(arrow_array::Int32Array::from(vec![1, 4])) as ArrayRef,
        ],
        None,
    );
    let mp_col = Arc::new(arrow_array::MapArray::new(
        entries_field,
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0i32, 1, 1, 1, 2])),
        mp_entries,
        Some(NullBuffer::from(vec![true, false, true, true])),
        false,
    )) as ArrayRef;

    let x_field = Arc::new(Field::new("x", DataType::Utf8, true).with_metadata(fid(20)));
    let ys_element = Arc::new(Field::new("element", DataType::Int32, true).with_metadata(fid(22)));
    let ys_field =
        Arc::new(Field::new("ys", DataType::List(ys_element.clone()), true).with_metadata(fid(21)));
    let inner_field = Arc::new(
        Field::new(
            "inner",
            DataType::Struct(Fields::from([x_field.clone(), ys_field.clone()])),
            true,
        )
        .with_metadata(fid(19)),
    );
    let deep_field = Field::new(
        "deep",
        DataType::Struct(Fields::from([inner_field.clone()])),
        true,
    )
    .with_metadata(fid(18));
    let ys_col = Arc::new(arrow_array::ListArray::new(
        ys_element,
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0i32, 1, 1, 1, 1])),
        Arc::new(arrow_array::Int32Array::from(vec![1])),
        Some(NullBuffer::from(vec![true, true, false, true])),
    )) as ArrayRef;
    let inner_col = Arc::new(arrow_array::StructArray::new(
        Fields::from([x_field, ys_field]),
        vec![
            Arc::new(arrow_array::StringArray::from(vec![
                Some("x1"),
                None,
                None,
                None,
            ])) as ArrayRef,
            ys_col,
        ],
        Some(NullBuffer::from(vec![true, false, true, false])),
    )) as ArrayRef;
    let deep_col = Arc::new(arrow_array::StructArray::new(
        Fields::from([inner_field]),
        vec![inner_col],
        Some(NullBuffer::from(vec![true, false, true, true])),
    )) as ArrayRef;

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false).with_metadata(fid(1)),
        st_field,
        xs_field,
        mp_field,
        deep_field,
    ]));
    let id_col = Arc::new(arrow_array::Int64Array::from_iter_values([1, 2, 3, 4])) as ArrayRef;
    RecordBatch::try_new(arrow_schema, vec![id_col, st_col, xs_col, mp_col, deep_col])
        .expect("build the container truth batch")
}

#[test]
fn container_null_predicates_match_spark_oracle_on_materialized_batch() {
    let schema = container_test_schema();
    let batch = container_truth_batch();

    let cases: Vec<(&str, crate::expr::Predicate, Vec<bool>)> = vec![
        ("st IS NULL", Reference::new("st").is_null(), vec![
            false, true, false, false,
        ]),
        ("st IS NOT NULL", Reference::new("st").is_not_null(), vec![
            true, false, true, true,
        ]),
        ("xs IS NULL", Reference::new("xs").is_null(), vec![
            false, true, false, false,
        ]),
        ("mp IS NULL", Reference::new("mp").is_null(), vec![
            false, true, false, false,
        ]),
        ("st.a IS NULL", Reference::new("st.a").is_null(), vec![
            false, true, true, false,
        ]),
        (
            "deep.inner IS NULL",
            Reference::new("deep.inner").is_null(),
            vec![false, true, false, true],
        ),
        (
            "deep.inner.x IS NULL",
            Reference::new("deep.inner.x").is_null(),
            vec![false, true, true, true],
        ),
        (
            "deep.inner.ys IS NULL",
            Reference::new("deep.inner.ys").is_null(),
            vec![false, true, true, true],
        ),
        (
            "st IS NULL OR id = 1",
            Reference::new("st")
                .is_null()
                .or(Reference::new("id").equal_to(Datum::long(1))),
            vec![true, true, false, false],
        ),
        (
            "st IS NOT NULL AND id > 2",
            Reference::new("st")
                .is_not_null()
                .and(Reference::new("id").greater_than(Datum::long(2))),
            vec![false, false, true, true],
        ),
    ];

    for (display, predicate, expected) in cases {
        let bound = predicate
            .bind(schema.clone(), true)
            .unwrap_or_else(|e| panic!("bind `{display}`: {e}"));
        let mask = evaluate_predicate_to_mask(&bound, &batch)
            .unwrap_or_else(|e| panic!("evaluate `{display}`: {e}"));
        assert_eq!(two_valued(&mask), expected, "`{display}`");
    }
}
