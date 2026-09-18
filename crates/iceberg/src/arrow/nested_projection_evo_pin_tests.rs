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

use crate::arrow::record_batch_transformer::{
    BatchTransform, ColumnSource, RecordBatchTransformer,
};

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
fn idless_source_child_named_like_a_readded_field_reads_null() {
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
    let result = transformer.process_record_batch(file_batch).unwrap();
    let s = result
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let b = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(b.is_null(0));
    assert!(b.is_null(1));
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
    let mut plan = NestedProjectionPlan::build(arrs.data_type(), &target_type, &schema).unwrap();
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
