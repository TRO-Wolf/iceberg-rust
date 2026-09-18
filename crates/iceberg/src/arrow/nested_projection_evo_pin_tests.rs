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

use arrow_array::{FixedSizeBinaryArray, LargeBinaryArray, Time64MicrosecondArray};
use uuid::Uuid;

use crate::arrow::record_batch_transformer::{
    BatchTransform, ColumnSource, RecordBatchTransformer,
};

fn snapshot_schema_with_typed_struct_b(b_type: PrimitiveType, default: Literal) -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(4, "b", Type::Primitive(b_type))
                            .with_initial_default(default)
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    )
}

#[test]
fn identical_nested_column_on_a_modify_batch_uses_pass_through() {
    let snapshot_schema = table_schema_with_struct_a_b();
    let s_fields = Fields::from(vec![
        id_field("a", DataType::Int32, true, 3),
        id_field("b", DataType::Utf8, true, 4),
    ]);
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("s", DataType::Struct(s_fields), true, 2),
    ]));
    let transform = RecordBatchTransformer::generate_batch_transform(
        &file_schema,
        snapshot_schema.as_ref(),
        &[2, 1],
        &HashMap::new(),
        None,
        None,
    )
    .unwrap();
    let BatchTransform::Modify { operations, .. } = transform else {
        panic!("a reordered projection must take the Modify transform")
    };
    assert!(matches!(
        operations[0],
        ColumnSource::PassThrough { .. }
    ));
    assert!(matches!(
        operations[1],
        ColumnSource::PassThrough { .. }
    ));
}

#[test]
fn idless_source_child_named_like_a_readded_field_errors_loud() {
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
                        NestedField::optional(5, "b", Type::Primitive(PrimitiveType::String))
                            .into(),
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
    ]);
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("s", DataType::Struct(s_fields.clone()), true, 2),
    ]));
    let s = StructArray::new(
        s_fields,
        vec![
            Arc::new(Int32Array::from(vec![7, 8])) as ArrayRef,
            Arc::new(StringArray::from(vec!["old", "vals"])) as ArrayRef,
        ],
        None,
    );
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(s) as ArrayRef,
    ])
    .unwrap();
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let err = transformer.process_record_batch(file_batch).unwrap_err();
    assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        err.to_string().contains("'b'") && err.to_string().contains("stamped and unstamped"),
        "{err}"
    );
}

#[test]
fn idless_child_when_sibling_struct_consumed_higher_ids_errors_loud() {
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "other",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "x", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(4, "y", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(5, "z", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
                NestedField::optional(
                    6,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(7, "a", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(8, "b", Type::Primitive(PrimitiveType::String))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let other_fields = Fields::from(vec![
        id_field("x", DataType::Int32, true, 3),
        id_field("y", DataType::Int32, true, 4),
        id_field("z", DataType::Int32, true, 5),
    ]);
    let s_fields = Fields::from(vec![
        Arc::new(id_field("a", DataType::Int32, true, 7)) as Arc<Field>,
        Arc::new(Field::new("b", DataType::Utf8, true)),
    ]);
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("other", DataType::Struct(other_fields.clone()), true, 2),
        id_field("s", DataType::Struct(s_fields.clone()), true, 6),
    ]));
    let other = StructArray::new(
        other_fields,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![3, 4])) as ArrayRef,
            Arc::new(Int32Array::from(vec![5, 6])) as ArrayRef,
        ],
        None,
    );
    let s = StructArray::new(
        s_fields,
        vec![
            Arc::new(Int32Array::from(vec![7, 8])) as ArrayRef,
            Arc::new(StringArray::from(vec!["keep", "kept"])) as ArrayRef,
        ],
        None,
    );
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(other) as ArrayRef,
        Arc::new(s) as ArrayRef,
    ])
    .unwrap();
    let mut transformer =
        RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2, 6]).build();
    let err = transformer.process_record_batch(file_batch).unwrap_err();
    assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
    assert!(
        err.to_string().contains("'b'") && err.to_string().contains("stamped and unstamped"),
        "{err}"
    );
}

#[test]
fn idless_source_child_matching_no_target_child_is_ignored() {
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
    ]);
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("s", DataType::Struct(s_fields.clone()), true, 2),
    ]));
    let s = StructArray::new(
        s_fields,
        vec![
            Arc::new(Int32Array::from(vec![7, 8])) as ArrayRef,
            Arc::new(StringArray::from(vec!["old", "vals"])) as ArrayRef,
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
    assert_eq!(s.num_columns(), 1);
    let a = s.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(a.values(), &[7, 8]);
}

#[test]
fn null_parent_struct_propagates_null_rows() {
    let snapshot_schema = table_schema_with_struct_a_b();
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field(
            "s",
            DataType::Struct(Fields::from(vec![id_field(
                "a",
                DataType::Int32,
                true,
                3,
            )])),
            true,
            2,
        ),
    ]));
    let s = StructArray::new(
        Fields::from(vec![id_field("a", DataType::Int32, true, 3)]),
        vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
        Some(NullBuffer::from(vec![true, false])),
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
    assert!(!s.is_null(0));
    assert!(s.is_null(1));
}

#[test]
fn nested_field_dropped_then_readded_same_name_new_id_reads_null() {
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
                        NestedField::optional(5, "x", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let s_fields = Fields::from(vec![
        id_field("a", DataType::Int32, true, 3),
        id_field("x", DataType::Int32, true, 4),
    ]);
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("s", DataType::Struct(s_fields.clone()), true, 2),
    ]));
    let s = StructArray::new(
        s_fields,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![9, 10])) as ArrayRef,
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
    assert_eq!(a.values(), &[1, 2]);
    let x = s.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
    assert!(x.is_null(0));
    assert!(x.is_null(1));
}

#[test]
fn list_element_field_named_differently_still_projects() {
    let schema = list_projection_schema();
    let element_fields = Fields::from(vec![id_field("x", DataType::Int32, true, 7)]);
    let element_field = Arc::new(id_field(
        "item",
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
    let target_type = DataType::List(Arc::new(id_field(
        "element",
        DataType::Struct(Fields::from(vec![
            id_field("x", DataType::Int32, true, 7),
            id_field("y", DataType::Utf8, true, 8),
        ])),
        true,
        6,
    )));
    let mut plan = NestedProjectionPlan::build(arrs.data_type(), &target_type, &schema, "list").unwrap();
    let result = plan.apply(Arc::new(arrs) as ArrayRef).unwrap();
    assert_eq!(result.data_type(), &target_type);
    let arrs = result.as_any().downcast_ref::<ListArray>().unwrap();
    let elements = arrs
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert_eq!(elements.num_columns(), 2);
    let y = elements
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(y.is_null(0));
    assert!(y.is_null(1));
}

#[test]
fn list_of_list_element_struct_child_added_reads_null() {
    let schema = Arc::new(
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
                            Type::List(ListType::new(
                                NestedField::optional(
                                    7,
                                    "element",
                                    Type::Struct(StructType::new(vec![
                                        NestedField::optional(
                                            8,
                                            "x",
                                            Type::Primitive(PrimitiveType::Int),
                                        )
                                        .into(),
                                        NestedField::optional(
                                            9,
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
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap(),
    );
    let inner_element = Arc::new(id_field(
        "element",
        DataType::Struct(Fields::from(vec![id_field(
            "x",
            DataType::Int32,
            true,
            8,
        )])),
        true,
        7,
    ));
    let outer_element = Arc::new(id_field(
        "element",
        DataType::List(inner_element.clone()),
        true,
        6,
    ));
    let inner_values = StructArray::new(
        Fields::from(vec![id_field("x", DataType::Int32, true, 8)]),
        vec![Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef],
        None,
    );
    let inner = ListArray::new(
        inner_element,
        OffsetBuffer::new(vec![0, 1, 2].into()),
        Arc::new(inner_values) as ArrayRef,
        None,
    );
    let outer = ListArray::new(
        outer_element.clone(),
        OffsetBuffer::new(vec![0, 2].into()),
        Arc::new(inner) as ArrayRef,
        None,
    );
    let file_schema = Arc::new(ArrowSchema::new(vec![
        id_field("id", DataType::Int32, false, 1),
        id_field("arrs", DataType::List(outer_element.clone()), true, 5),
    ]));
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1])) as ArrayRef,
        Arc::new(outer) as ArrayRef,
    ])
    .unwrap();
    let mut transformer = RecordBatchTransformerBuilder::new(schema, &[1, 5]).build();
    let result = transformer.process_record_batch(file_batch).unwrap();
    let outer = result
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let inner = outer
        .values()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let elements = inner
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert_eq!(elements.num_columns(), 2);
    let y = elements
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(y.is_null(0));
    assert!(y.is_null(1));
}

#[test]
fn nested_time_initial_default_reads_time64() {
    let snapshot_schema =
        snapshot_schema_with_typed_struct_b(PrimitiveType::Time, Literal::time(123456789));
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let result = transformer
        .process_record_batch(file_batch_with_struct_a_only())
        .unwrap();
    let s = result
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let b = s
        .column(1)
        .as_any()
        .downcast_ref::<Time64MicrosecondArray>()
        .unwrap();
    assert_eq!(b.value(0), 123456789);
    assert_eq!(b.value(1), 123456789);
}

#[test]
fn nested_uuid_initial_default_reads_fixed_size_binary() {
    let uuid = Uuid::from_u128(0xa1a2a3a4b1b2c1c2d1d2d3d4d5d6d7d8);
    let snapshot_schema =
        snapshot_schema_with_typed_struct_b(PrimitiveType::Uuid, Literal::uuid(uuid));
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let result = transformer
        .process_record_batch(file_batch_with_struct_a_only())
        .unwrap();
    let s = result
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let b = s
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap();
    assert_eq!(b.value(0), uuid.as_bytes().as_slice());
    assert_eq!(b.value(1), uuid.as_bytes().as_slice());
}

#[test]
fn nested_binary_initial_default_reads_large_binary() {
    let snapshot_schema = snapshot_schema_with_typed_struct_b(
        PrimitiveType::Binary,
        Literal::binary([1u8, 2, 3]),
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
    let b = s
        .column(1)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap();
    assert_eq!(b.value(0), &[1u8, 2, 3]);
    assert_eq!(b.value(1), &[1u8, 2, 3]);
}

#[test]
fn nested_fixed_initial_default_reads_fixed_size_binary() {
    let snapshot_schema = snapshot_schema_with_typed_struct_b(
        PrimitiveType::Fixed(4),
        Literal::fixed([9u8, 8, 7, 6]),
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
    let b = s
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap();
    assert_eq!(b.value(0), &[9u8, 8, 7, 6]);
    assert_eq!(b.value(1), &[9u8, 8, 7, 6]);
}

#[test]
fn top_level_time_initial_default_reads_time64() {
    let snapshot_schema = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "t", Type::Primitive(PrimitiveType::Time))
                    .with_initial_default(Literal::time(999))
                    .into(),
            ])
            .build()
            .unwrap(),
    );
    let file_schema = Arc::new(ArrowSchema::new(vec![id_field(
        "id",
        DataType::Int32,
        false,
        1,
    )]));
    let file_batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
    ])
    .unwrap();
    let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
    let result = transformer.process_record_batch(file_batch).unwrap();
    let t = result
        .column(1)
        .as_any()
        .downcast_ref::<Time64MicrosecondArray>()
        .unwrap();
    assert_eq!(t.value(0), 999);
    assert_eq!(t.value(1), 999);
}

#[test]
fn later_batch_with_a_richer_nested_layout_rebuilds_the_plan() {
    let snapshot_schema = table_schema_with_struct_a_b();
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
    assert!(b.is_null(0));

    let s_fields = Fields::from(vec![
        id_field("a", DataType::Int32, true, 3),
        id_field("b", DataType::Utf8, true, 4),
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
        ],
        None,
    );
    let batch = RecordBatch::try_new(file_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        Arc::new(s) as ArrayRef,
    ])
    .unwrap();
    let result = transformer.process_record_batch(batch).unwrap();
    let s = result
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let b = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(b.value(0), "keep");
    assert_eq!(b.value(1), "kept");
}
