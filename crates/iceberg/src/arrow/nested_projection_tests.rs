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

    use arrow_array::{
        Array, ArrayRef, Int32Array, ListArray, MapArray, RecordBatch, StringArray, StructArray,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use crate::arrow::record_batch_transformer::RecordBatchTransformerBuilder;
    use crate::spec::{ListType, MapType, NestedField, PrimitiveType, Schema, StructType, Type};

    fn id_field(name: &str, ty: DataType, nullable: bool, id: i32) -> Field {
        Field::new(name, ty, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    }

    fn table_schema_with_struct_a_b() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        2,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int))
                                .into(),
                            NestedField::optional(4, "b", Type::Primitive(PrimitiveType::String))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn file_batch_with_struct_a_only() -> RecordBatch {
        let file_schema = Arc::new(ArrowSchema::new(vec![
            id_field("id", DataType::Int32, false, 1),
            id_field(
                "s",
                DataType::Struct(Fields::from(vec![id_field("a", DataType::Int32, true, 3)])),
                true,
                2,
            ),
        ]));
        let s = StructArray::new(
            Fields::from(vec![id_field("a", DataType::Int32, true, 3)]),
            vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
            None,
        );
        RecordBatch::try_new(file_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(s) as ArrayRef,
        ])
        .unwrap()
    }

    #[test]
    fn struct_child_added_after_file_written_reads_null() {
        let mut transformer =
            RecordBatchTransformerBuilder::new(table_schema_with_struct_a_b(), &[1, 2]).build();
        let result = transformer
            .process_record_batch(file_batch_with_struct_a_only())
            .unwrap();
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 2);
        let s = result
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(s.num_columns(), 2);
        let a = s.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(a.value(0), 1);
        assert_eq!(a.value(1), 2);
        let b = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert!(b.is_null(0));
        assert!(b.is_null(1));
    }

    #[test]
    fn list_element_struct_child_added_after_file_written_reads_null() {
        let snapshot_schema = Arc::new(
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
        );
        let element_field = Arc::new(id_field(
            "element",
            DataType::Struct(Fields::from(vec![id_field("x", DataType::Int32, true, 7)])),
            true,
            6,
        ));
        let file_schema = Arc::new(ArrowSchema::new(vec![
            id_field("id", DataType::Int32, false, 1),
            id_field("arrs", DataType::List(element_field.clone()), true, 5),
        ]));
        let values = StructArray::new(
            Fields::from(vec![id_field("x", DataType::Int32, true, 7)]),
            vec![Arc::new(Int32Array::from(vec![10, 11, 20])) as ArrayRef],
            None,
        );
        let arrs = ListArray::new(
            element_field,
            OffsetBuffer::new(vec![0, 2, 3].into()),
            Arc::new(values) as ArrayRef,
            None,
        );
        let file_batch = RecordBatch::try_new(file_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(arrs) as ArrayRef,
        ])
        .unwrap();
        let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 5]).build();
        let result = transformer.process_record_batch(file_batch).unwrap();
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 2);
        let arrs = result
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
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
        assert_eq!(x.values(), &[10, 11, 20]);
        let y = elements
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(y.len(), 3);
        assert!(y.is_null(0));
        assert!(y.is_null(1));
        assert!(y.is_null(2));
    }

    #[test]
    fn map_value_struct_child_added_after_file_written_reads_null() {
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
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            NestedField::optional(
                                13,
                                "value",
                                Type::Struct(StructType::new(vec![
                                    NestedField::optional(
                                        11,
                                        "v",
                                        Type::Primitive(PrimitiveType::Int),
                                    )
                                    .into(),
                                    NestedField::optional(
                                        12,
                                        "w",
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
        let key_field = Arc::new(id_field("key", DataType::Utf8, false, 10));
        let file_value_field = Arc::new(id_field(
            "value",
            DataType::Struct(Fields::from(vec![id_field("v", DataType::Int32, true, 11)])),
            true,
            13,
        ));
        let entries_field = Arc::new(Field::new(
            "key_value",
            DataType::Struct(Fields::from(vec![
                key_field.as_ref().clone(),
                file_value_field.as_ref().clone(),
            ])),
            false,
        ));
        let file_schema = Arc::new(ArrowSchema::new(vec![
            id_field("id", DataType::Int32, false, 1),
            id_field("m", DataType::Map(entries_field.clone(), false), true, 9),
        ]));
        let values_struct = StructArray::new(
            Fields::from(vec![
                key_field.as_ref().clone(),
                file_value_field.as_ref().clone(),
            ]),
            vec![
                Arc::new(StringArray::from(vec!["k1", "k2", "k3"])) as ArrayRef,
                Arc::new(StructArray::new(
                    Fields::from(vec![id_field("v", DataType::Int32, true, 11)]),
                    vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef],
                    None,
                )) as ArrayRef,
            ],
            None,
        );
        let m = MapArray::new(
            entries_field,
            OffsetBuffer::new(vec![0, 1, 3].into()),
            values_struct,
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
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 2);
        let m = result
            .column(1)
            .as_any()
            .downcast_ref::<MapArray>()
            .unwrap();
        let keys = m.keys().as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(keys.value(0), "k1");
        assert_eq!(keys.value(1), "k2");
        assert_eq!(keys.value(2), "k3");
        let value_struct = m.values().as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(value_struct.num_columns(), 2);
        let v = value_struct
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(v.values(), &[1, 2, 3]);
        let w = value_struct
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(w.len(), 3);
        assert!(w.is_null(0));
        assert!(w.is_null(1));
        assert!(w.is_null(2));
    }

    #[test]
    fn struct_child_added_two_levels_deep_reads_null() {
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        2,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(
                                3,
                                "mid",
                                Type::Struct(StructType::new(vec![
                                    NestedField::optional(
                                        4,
                                        "a",
                                        Type::Primitive(PrimitiveType::Int),
                                    )
                                    .into(),
                                    NestedField::optional(
                                        5,
                                        "b",
                                        Type::Primitive(PrimitiveType::String),
                                    )
                                    .into(),
                                ])),
                            )
                            .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let mid_field = id_field(
            "mid",
            DataType::Struct(Fields::from(vec![id_field("a", DataType::Int32, true, 4)])),
            true,
            3,
        );
        let file_schema = Arc::new(ArrowSchema::new(vec![
            id_field("id", DataType::Int32, false, 1),
            id_field(
                "s",
                DataType::Struct(Fields::from(vec![mid_field.clone()])),
                true,
                2,
            ),
        ]));
        let mid = StructArray::new(
            Fields::from(vec![id_field("a", DataType::Int32, true, 4)]),
            vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
            None,
        );
        let s = StructArray::new(
            Fields::from(vec![mid_field]),
            vec![Arc::new(mid) as ArrayRef],
            None,
        );
        let file_batch = RecordBatch::try_new(file_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(s) as ArrayRef,
        ])
        .unwrap();
        let mut transformer = RecordBatchTransformerBuilder::new(snapshot_schema, &[1, 2]).build();
        let result = transformer.process_record_batch(file_batch).unwrap();
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 2);
        let s = result
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let mid = s.column(0).as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(mid.num_columns(), 2);
        let a = mid.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(a.values(), &[1, 2]);
        let b = mid
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(b.is_null(0));
        assert!(b.is_null(1));
    }

    #[test]
    fn renamed_struct_child_reads_by_field_id() {
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        2,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int))
                                .into(),
                            NestedField::optional(4, "c", Type::Primitive(PrimitiveType::String))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let file_schema = Arc::new(ArrowSchema::new(vec![
            id_field("id", DataType::Int32, false, 1),
            id_field(
                "s",
                DataType::Struct(Fields::from(vec![
                    id_field("a", DataType::Int32, true, 3),
                    id_field("b", DataType::Utf8, true, 4),
                ])),
                true,
                2,
            ),
        ]));
        let s = StructArray::new(
            Fields::from(vec![
                id_field("a", DataType::Int32, true, 3),
                id_field("b", DataType::Utf8, true, 4),
            ]),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef,
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
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 2);
        let s = result
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(s.num_columns(), 2);
        let renamed = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(renamed.value(0), "x");
        assert_eq!(renamed.value(1), "y");
        assert_eq!(
            result.schema().field(1).data_type(),
            result.column(1).data_type()
        );
        if let DataType::Struct(children) = result.schema().field(1).data_type() {
            assert_eq!(children[1].name(), "c");
        } else {
            panic!("expected struct");
        }
    }

    #[tokio::test]
    async fn parquet_file_missing_struct_child_reads_null_through_reader() {
        use futures::{TryStreamExt, stream};
        use parquet::arrow::ArrowWriter;
        use parquet::basic::Compression;
        use parquet::file::properties::WriterProperties;

        use crate::arrow::ArrowReaderBuilder;
        use crate::io::FileIO;
        use crate::scan::{FileScanTask, FileScanTaskStream};
        use crate::spec::DataFileFormat;

        let file_schema = Arc::new(ArrowSchema::new(vec![
            id_field("id", DataType::Int32, false, 1),
            id_field(
                "s",
                DataType::Struct(Fields::from(vec![id_field("a", DataType::Int32, true, 3)])),
                true,
                2,
            ),
        ]));
        let s = StructArray::new(
            Fields::from(vec![id_field("a", DataType::Int32, true, 3)]),
            vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
            None,
        );
        let to_write = RecordBatch::try_new(file_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(s) as ArrayRef,
        ])
        .unwrap();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let path = format!("{}/1.parquet", tmp_dir.path().to_str().unwrap());
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).unwrap();
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
        let tasks = Box::pin(stream::iter(vec![Ok(FileScanTask {
            file_size_in_bytes: std::fs::metadata(&path).unwrap().len(),
            start: 0,
            length: 0,
            record_count: None,
            file_record_count: None,
            data_file_path: Arc::from(path),
            data_file_format: DataFileFormat::Parquet,
            schema: table_schema_with_struct_a_b(),
            project_field_ids: Arc::from(vec![1, 2]),
            predicate: None,
            deletes: Arc::from(vec![]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        })])) as FileScanTaskStream;
        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
        let s = batch
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(s.num_columns(), 2);
        let a = s.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(a.values(), &[1, 2]);
        let b = s.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert!(b.is_null(0));
        assert!(b.is_null(1));
    }

    #[test]
    fn reordered_struct_children_read_by_field_id() {
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        2,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(4, "b", Type::Primitive(PrimitiveType::String))
                                .into(),
                            NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let file_schema = Arc::new(ArrowSchema::new(vec![
            id_field("id", DataType::Int32, false, 1),
            id_field(
                "s",
                DataType::Struct(Fields::from(vec![
                    id_field("a", DataType::Int32, true, 3),
                    id_field("b", DataType::Utf8, true, 4),
                ])),
                true,
                2,
            ),
        ]));
        let s = StructArray::new(
            Fields::from(vec![
                id_field("a", DataType::Int32, true, 3),
                id_field("b", DataType::Utf8, true, 4),
            ]),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef,
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
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 2);
        let s = result
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(s.num_columns(), 2);
        let b = s.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(b.value(0), "x");
        assert_eq!(b.value(1), "y");
        let a = s.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(a.value(0), 1);
        assert_eq!(a.value(1), 2);
    }
