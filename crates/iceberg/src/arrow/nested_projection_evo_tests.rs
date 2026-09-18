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

use arrow_array::{Decimal128Array, FixedSizeListArray, Int64Array, LargeListArray};
use arrow_buffer::NullBuffer;

use super::NestedProjectionPlan;
use crate::spec::Literal;

#[test]
fn nested_child_with_initial_default_reads_default() {
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(4, "b", Type::Primitive(PrimitiveType::String))
                            .with_initial_default(Literal::string("x"))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let result = transformer
        .process_record_batch(file_batch_with_struct_a_only())
        .unwrap();
    let s = result
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let b = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(b.value(0), "x");
    assert_eq!(b.value(1), "x");
}

#[test]
fn mixed_field_id_struct_matches_idless_child_by_name() {
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(4, "b", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::optional(5, "c", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let s_fields = Fields::from(vec![
        Arc::new(id_field("a", DataType::Int32, true, 3)) as Arc<Field>,
        Arc::new(Field::new("b", DataType::Utf8, true)),
        Arc::new(id_field("c", DataType::Int32, true, 5)),
    ]);
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("s", DataType::Struct(s_fields.clone()), true, 2),
    ]));
    let s = StructArray::new(
        s_fields,
        vec![
            Arc::new(Int32Array::from(vec![7, 8])) as ArrayRef,
            Arc::new(StringArray::from(vec!["keep", "kept"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![11, 12])) as ArrayRef,
        ],
        None,
    );
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(s) as ArrayRef,
    ])
    .unwrap();
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let result = transformer.process_record_batch(file_batch).unwrap();
    let s = result
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let a = s.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(a.values(), &[7, 8]);
    let b = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(b.value(0), "keep");
    assert_eq!(b.value(1), "kept");
    let c = s.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(c.values(), &[11, 12]);
}

#[test]
fn zero_child_file_struct_fills_target_children() {
    let snapshot_schema = table_schema_with_struct_a_b();
    let s_fields = Fields::default();
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("s", DataType::Struct(s_fields.clone()), true, 2),
    ]));
    let s = StructArray::new_empty_fields(2, Some(NullBuffer::from(vec![true, true])));
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(s) as ArrayRef,
    ])
    .unwrap();
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let result = transformer.process_record_batch(file_batch).unwrap();
    assert_eq!(result.num_rows(), 2);
    let s = result
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert_eq!(s.num_columns(), 2);
    let b = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(b.is_null(0));
    assert!(b.is_null(1));
}

#[test]
fn map_key_struct_child_added_after_file_written_reads_null() {
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    9,
                    "m",
                    Type::Map(MapType::new(
                        NestedField::required(
                            10,
                            "key",
                            Type::Struct(StructType::new(vec![
                                NestedField::required(
                                    11,
                                    "k1",
                                    Type::Primitive(PrimitiveType::Int),
                                )
                                .into(),
                                NestedField::optional(
                                    12,
                                    "k2",
                                    Type::Primitive(PrimitiveType::String),
                                )
                                .into(),
                            ])),
                        )
                        .into(),
                        NestedField::optional(13, "value", Type::Primitive(PrimitiveType::Int))
                            .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let key_field = Arc::new(id_field(
        "key",
        DataType::Struct(Fields::from(vec![id_field(
            "k1",
            DataType::Int32,
            false,
            11,
        )])),
        false,
        10,
    ));
    let value_field = Arc::new(id_field("value", DataType::Int32, true, 13));
    let entries_fields = Fields::from(vec![
        key_field.clone() as Arc<Field>,
        value_field.clone() as Arc<Field>,
    ]);
    let entries_field = Arc::new(Field::new(
        "key_value",
        DataType::Struct(entries_fields.clone()),
        false,
    ));
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("m", DataType::Map(entries_field.clone(), false), true, 9),
    ]));
    let keys = StructArray::new(
        Fields::from(vec![id_field("k1", DataType::Int32, false, 11)]),
        vec![Arc::new(Int32Array::from(vec![5, 6])) as ArrayRef],
        None,
    );
    let entries = StructArray::new(
        entries_fields,
        vec![
            Arc::new(keys) as ArrayRef,
            Arc::new(Int32Array::from(vec![50, 60])) as ArrayRef,
        ],
        None,
    );
    let m = MapArray::new(
        entries_field,
        OffsetBuffer::new(vec![0, 1, 2].into()),
        entries,
        None,
        false,
    );
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(m) as ArrayRef,
    ])
    .unwrap();
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 9]).build();
    let result = transformer.process_record_batch(file_batch).unwrap();
    let m = result
        .column(1)
        .as_any()
        .downcast_ref::<MapArray>()
        .unwrap();
    let keys = m.keys().as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(keys.num_columns(), 2);
    let k1 = keys.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(k1.values(), &[5, 6]);
    let k2 = keys.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(k2.is_null(0));
    assert!(k2.is_null(1));
    let values = m.values().as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(values.values(), &[50, 60]);
}

fn deep_table_struct_type(levels: usize) -> Type {
    let mut ty = Type::Struct(StructType::new(vec![
        NestedField::optional(1000, "a", Type::Primitive(PrimitiveType::Int)).into(),
        NestedField::optional(1001, "b", Type::Primitive(PrimitiveType::String)).into(),
    ]));
    for i in 0..levels {
        ty = Type::Struct(StructType::new(vec![
            NestedField::optional(10 + i as i32, "f", ty).into(),
        ]));
    }
    ty
}

fn deep_file_struct_fields(levels: usize) -> (DataType, Fields) {
    let mut ty = DataType::Struct(Fields::from(vec![id_field(
        "a",
        DataType::Int32,
        true,
        1000,
    )]));
    let mut fields = Fields::from(vec![id_field("a", DataType::Int32, true, 1000)]);
    for i in 0..levels {
        ty = DataType::Struct(Fields::from(vec![id_field(
            "f",
            ty.clone(),
            true,
            10 + i as i32,
        )]));
        fields = Fields::from(vec![id_field("f", ty.clone(), true, 10 + i as i32)]);
    }
    (ty, fields)
}

fn deep_file_struct_array(levels: usize, a_values: Int32Array) -> StructArray {
    let mut array: ArrayRef = Arc::new(StructArray::new(
        Fields::from(vec![id_field("a", DataType::Int32, true, 1000)]),
        vec![Arc::new(a_values) as ArrayRef],
        None,
    ));
    for i in 0..levels {
        array = Arc::new(StructArray::new(
            Fields::from(vec![id_field("f", array.data_type().clone(), true, 10 + i as i32)]),
            vec![array],
            None,
        ));
    }
    array.as_any().downcast_ref::<StructArray>().unwrap().clone()
}

#[test]
fn deeply_nested_struct_child_add_projects_within_bound() {
    let levels = 40;
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "s", deep_table_struct_type(levels)).into(),
            ])
            .build()
            .unwrap(),
    );
    let (s_type, s_fields) = deep_file_struct_fields(levels);
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("s", s_type, true, 2),
    ]));
    let s = deep_file_struct_array(levels, Int32Array::from(vec![1, 2]));
    let _ = s_fields;
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(s) as ArrayRef,
    ])
    .unwrap();
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let result = transformer.process_record_batch(file_batch).unwrap();
    let mut inner = result.column(1).clone();
    for _ in 0..levels {
        inner = inner
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .column(0)
            .clone();
    }
    let leaf = inner.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(leaf.num_columns(), 2);
    let b = leaf
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(b.is_null(0));
    assert!(b.is_null(1));
}

#[test]
fn required_nested_child_missing_without_default_errors() {
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::required(4, "b", Type::Primitive(PrimitiveType::String)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let err = match transformer.process_record_batch(file_batch_with_struct_a_only()) {
        Ok(_) => panic!("expected a missing-required-field error"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("Missing required field"),
        "{err}"
    );
}

#[test]
fn required_nested_child_missing_with_default_reads_default() {
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::required(4, "b", Type::Primitive(PrimitiveType::String))
                            .with_initial_default(Literal::string("x"))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let result = transformer
        .process_record_batch(file_batch_with_struct_a_only())
        .unwrap();
    let s = result
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let b = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(b.value(0), "x");
    assert_eq!(b.value(1), "x");
    assert_eq!(b.null_count(), 0);
}

#[test]
fn required_list_element_with_null_projected_values_errors() {
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    5,
                    "arrs",
                    Type::List(ListType::new(
                        NestedField::required(
                            6,
                            "element",
                            Type::Struct(StructType::new(vec![
                                NestedField::optional(
                                    7,
                                    "x",
                                    Type::Primitive(PrimitiveType::Int),
                                )
                                .into(),
                                NestedField::optional(
                                    8,
                                    "y",
                                    Type::Primitive(PrimitiveType::String),
                                )
                                .into(),
                            ])),
                        )
                        .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let element_fields = Fields::from(vec![id_field("x", DataType::Int32, true, 7)]);
    let element_field = Arc::new(id_field(
        "element",
        DataType::Struct(element_fields.clone()),
        true,
        6,
    ));
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("arrs", DataType::List(element_field.clone()), true, 5),
    ]));
    let values = StructArray::new(
        element_fields,
        vec![Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef],
        Some(NullBuffer::from(vec![true, false])),
    );
    let arrs = ListArray::new(
        element_field,
        OffsetBuffer::new(vec![0, 1, 2].into()),
        Arc::new(values) as ArrayRef,
        None,
    );
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(arrs) as ArrayRef,
    ])
    .unwrap();
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 5]).build();
    let result = transformer.process_record_batch(file_batch);
    assert!(result.is_err());
}

fn list_projection_schema() -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    5,
                    "arrs",
                    Type::List(ListType::new(
                        NestedField::optional(
                            6,
                            "element",
                            Type::Struct(StructType::new(vec![
                                NestedField::optional(
                                    7,
                                    "x",
                                    Type::Primitive(PrimitiveType::Int),
                                )
                                .into(),
                                NestedField::optional(
                                    8,
                                    "y",
                                    Type::Primitive(PrimitiveType::String),
                                )
                                .into(),
                            ])),
                        )
                        .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    )
}

#[test]
fn list_source_projects_into_large_list_target() {
    let schema = list_projection_schema();
    let element_fields = Fields::from(vec![id_field("x", DataType::Int32, true, 7)]);
    let element_field = Arc::new(id_field(
        "element",
        DataType::Struct(element_fields.clone()),
        true,
        6,
    ));
    let values = StructArray::new(
        element_fields,
        vec![Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef],
        None,
    );
    let arrs = ListArray::new(
        element_field,
        OffsetBuffer::new(vec![0, 2].into()),
        Arc::new(values) as ArrayRef,
        None,
    );
    let target_type = DataType::LargeList(Arc::new(id_field(
        "element",
        DataType::Struct(Fields::from(vec![
            id_field("x", DataType::Int32, true, 7),
            id_field("y", DataType::Utf8, true, 8),
        ])),
        true,
        6,
    )));
    let mut plan = NestedProjectionPlan::build(arrs.data_type(), &target_type, &schema).unwrap();
    let result = plan.apply(Arc::new(arrs) as ArrayRef).unwrap();
    let arrs = result.as_any().downcast_ref::<LargeListArray>().unwrap();
    let elements = arrs
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert_eq!(elements.num_columns(), 2);
    let x = elements
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(x.values(), &[10, 20]);
    let y = elements
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(y.is_null(0));
    assert!(y.is_null(1));
}

#[test]
fn large_list_source_projects_into_list_target() {
    let schema = list_projection_schema();
    let element_fields = Fields::from(vec![id_field("x", DataType::Int32, true, 7)]);
    let element_field = Arc::new(id_field(
        "element",
        DataType::Struct(element_fields.clone()),
        true,
        6,
    ));
    let values = StructArray::new(
        element_fields,
        vec![Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef],
        None,
    );
    let arrs = LargeListArray::new(
        element_field,
        OffsetBuffer::new(vec![0i64, 2].into()),
        Arc::new(values) as ArrayRef,
        None,
    );
    let target_type = DataType::List(Arc::new(id_field(
        "element",
        DataType::Struct(Fields::from(vec![
            id_field("x", DataType::Int32, true, 7),
            id_field("y", DataType::Utf8, true, 8),
        ])),
        true,
        6,
    )));
    let mut plan = NestedProjectionPlan::build(arrs.data_type(), &target_type, &schema).unwrap();
    let result = plan.apply(Arc::new(arrs) as ArrayRef).unwrap();
    let arrs = result.as_any().downcast_ref::<ListArray>().unwrap();
    let elements = arrs
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert_eq!(elements.num_columns(), 2);
    let x = elements
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(x.values(), &[10, 20]);
    let y = elements
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(y.is_null(0));
    assert!(y.is_null(1));
}

#[test]
fn fixed_size_list_rebuild_uses_target_size() {
    let schema = list_projection_schema();
    let element_fields = Fields::from(vec![id_field("x", DataType::Int32, true, 7)]);
    let source = FixedSizeListArray::new(
        Arc::new(id_field(
            "element",
            DataType::Struct(element_fields.clone()),
            true,
            6,
        )),
        2,
        Arc::new(StructArray::new(
            element_fields,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef],
            None,
        )) as ArrayRef,
        None,
    );
    let target_type = DataType::FixedSizeList(
        Arc::new(id_field(
            "element",
            DataType::Struct(Fields::from(vec![
                id_field("x", DataType::Int32, true, 7),
                id_field("z", DataType::Int32, true, 9),
            ])),
            true,
            6,
        )),
        3,
    );
    let mut plan = NestedProjectionPlan::build(source.data_type(), &target_type, &schema).unwrap();
    let result = plan.apply(Arc::new(source) as ArrayRef);
    assert!(result.is_err());
}

#[test]
fn nested_projection_plan_errors_past_max_depth() {
    let mut source_type = DataType::Struct(Fields::from(vec![id_field(
        "a",
        DataType::Int32,
        true,
        1000,
    )]));
    let mut target_type = DataType::Struct(Fields::from(vec![
        id_field("a", DataType::Int32, true, 1000),
        id_field("b", DataType::Utf8, true, 1001),
    ]));
    for i in 0..130i32 {
        source_type = DataType::Struct(Fields::from(vec![id_field(
            "f",
            source_type,
            true,
            10 + i,
        )]));
        target_type = DataType::Struct(Fields::from(vec![id_field(
            "f",
            target_type,
            true,
            10 + i,
        )]));
    }
    let err = NestedProjectionPlan::build(&source_type, &target_type, &list_projection_schema())
        .unwrap_err();
    assert!(err.to_string().contains("exceeds depth"), "{err}");
}

#[test]
fn nested_add_with_sibling_promotion_and_decimal_widen() {
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Long)).into(),
                        NestedField::optional(4, "b", Type::Primitive(PrimitiveType::String)).into(),
                        NestedField::optional(
                            5,
                            "d",
                            Type::Primitive(PrimitiveType::Decimal {
                                precision: 12,
                                scale: 2,
                            }),
                        )
                        .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let d = Decimal128Array::from(vec![1234i128, 5678])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let s_fields = Fields::from(vec![
        id_field("a", DataType::Int32, true, 3),
        id_field("d", DataType::Decimal128(10, 2), true, 5),
    ]);
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("s", DataType::Struct(s_fields.clone()), true, 2),
    ]));
    let s = StructArray::new(
        s_fields,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(d) as ArrayRef,
        ],
        None,
    );
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(s) as ArrayRef,
    ])
    .unwrap();
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let result = transformer.process_record_batch(file_batch).unwrap();
    let s = result
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert_eq!(s.num_columns(), 3);
    let a = s.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(a.values(), &[1, 2]);
    let b = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(b.is_null(0));
    assert!(b.is_null(1));
    let d = s
        .column(2)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(d.data_type(), &DataType::Decimal128(12, 2));
    assert_eq!(d.value(0), 1234);
    assert_eq!(d.value(1), 5678);
}
