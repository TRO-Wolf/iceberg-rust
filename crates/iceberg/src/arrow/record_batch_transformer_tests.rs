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

    /// Reads a string from a `StringArray` or a run-end-encoded one. A null gives `""`.
    fn get_string_value(array: &dyn Array, index: usize) -> String {
        if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
            if string_array.is_null(index) {
                String::new()
            } else {
                string_array.value(index).to_string()
            }
        } else if let Some(run_array) = array
            .as_any()
            .downcast_ref::<arrow_array::RunArray<arrow_array::types::Int32Type>>()
        {
            let values = run_array.values();
            let string_values = values
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("REE values should be StringArray");
            // Every row of an REE constant column shares the value at index 0.
            if string_values.is_null(0) {
                String::new()
            } else {
                string_values.value(0).to_string()
            }
        } else {
            panic!("Expected StringArray or RunEndEncoded<StringArray>");
        }
    }

    /// Reads an int from an `Int32Array` or a run-end-encoded one.
    fn get_int_value(array: &dyn Array, index: usize) -> i32 {
        if let Some(int_array) = array.as_any().downcast_ref::<Int32Array>() {
            int_array.value(index)
        } else if let Some(run_array) = array
            .as_any()
            .downcast_ref::<arrow_array::RunArray<arrow_array::types::Int32Type>>()
        {
            let values = run_array.values();
            let int_values = values
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("REE values should be Int32Array");
            int_values.value(0)
        } else {
            panic!("Expected Int32Array or RunEndEncoded<Int32Array>");
        }
    }

    #[test]
    fn build_field_id_to_source_schema_map_works() {
        let arrow_schema = arrow_schema_already_same_as_target();

        let result =
            RecordBatchTransformer::build_field_id_to_arrow_schema_map(&arrow_schema).unwrap();

        let expected = HashMap::from_iter([
            (10, (arrow_schema.fields()[0].clone(), 0)),
            (11, (arrow_schema.fields()[1].clone(), 1)),
            (12, (arrow_schema.fields()[2].clone(), 2)),
            (14, (arrow_schema.fields()[3].clone(), 3)),
            (15, (arrow_schema.fields()[4].clone(), 4)),
        ]);

        assert!(result.eq(&expected));
    }

    #[test]
    fn processor_returns_properly_shaped_record_batch_when_no_schema_migration_required() {
        let snapshot_schema = Arc::new(iceberg_table_schema());
        let projected_iceberg_field_ids = [13, 14];

        let mut inst =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let result = inst
            .process_record_batch(source_record_batch_no_migration_required())
            .unwrap();

        let expected = source_record_batch_no_migration_required();

        assert_eq!(result, expected);
    }

    #[test]
    fn processor_returns_properly_shaped_record_batch_when_schema_migration_required() {
        let snapshot_schema = Arc::new(iceberg_table_schema());
        let projected_iceberg_field_ids = [10, 11, 12, 14, 15]; // a, b, c, e, f

        let mut inst =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let result = inst.process_record_batch(source_record_batch()).unwrap();

        let expected = expected_record_batch_migration_required();

        assert_eq!(result, expected);
    }

    #[test]
    fn schema_evolution_adds_date_column_with_nulls() {
        // A DATE column added after the file was written must materialize as NULLs.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "date_col", Type::Primitive(PrimitiveType::Date))
                        .into(),
                ])
                .build()
                .unwrap(),
        );
        let projected_iceberg_field_ids = [1, 2, 3];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let file_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("name", DataType::Utf8, true, "2"),
        ]));

        let file_batch = RecordBatch::try_new(file_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                Some("Alice"),
                Some("Bob"),
                Some("Charlie"),
            ])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(file_batch).unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.num_rows(), 3);

        let id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_column.values(), &[1, 2, 3]);

        let name_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_column.value(0), "Alice");
        assert_eq!(name_column.value(1), "Bob");
        assert_eq!(name_column.value(2), "Charlie");

        let date_column = result
            .column(2)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        assert!(date_column.is_null(0));
        assert!(date_column.is_null(1));
        assert!(date_column.is_null(2));
    }

    #[test]
    fn row_position_metadata_column_counts_physical_ordinal_across_batches() {
        // Risk pinned: the `_pos` counter must CONTINUE across batches. A downstream engine
        // writes position deletes against it.
        use arrow_array::Int64Array;

        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );
        // Project the data column plus the reserved `_pos` metadata column.
        let projected_iceberg_field_ids = [1, crate::metadata_columns::RESERVED_FIELD_ID_POS];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let file_schema = Arc::new(ArrowSchema::new(vec![simple_field(
            "id",
            DataType::Int32,
            false,
            "1",
        )]));

        // First batch of 3 rows gives _pos 0, 1, 2.
        let batch1 =
            RecordBatch::try_new(file_schema.clone(), vec![Arc::new(Int32Array::from(vec![
                10, 20, 30,
            ]))])
            .unwrap();
        let out1 = transformer.process_record_batch(batch1).unwrap();
        assert_eq!(out1.num_columns(), 2);
        assert_eq!(
            out1.column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[0, 1, 2]
        );

        // Second batch of 2 rows gives _pos 3, 4. The counter does NOT restart.
        let batch2 =
            RecordBatch::try_new(file_schema, vec![Arc::new(Int32Array::from(vec![40, 50]))])
                .unwrap();
        let out2 = transformer.process_record_batch(batch2).unwrap();
        assert_eq!(
            out2.column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[3, 4]
        );
    }

    #[test]
    fn schema_evolution_adds_struct_column_with_nulls() {
        // A struct column added after the data files were written must materialize as nulls.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(
                        3,
                        "struct_col",
                        Type::Struct(crate::spec::StructType::new(vec![
                            NestedField::optional(
                                100,
                                "inner_field",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let projected_iceberg_field_ids = [1, 2, 3];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_iceberg_field_ids)
                .build();

        let file_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("data", DataType::Utf8, false, "2"),
        ]));

        let file_batch = RecordBatch::try_new(file_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(file_batch).unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.num_rows(), 3);

        let id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_column.values(), &[1, 2, 3]);

        let data_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(data_column.value(0), "a");
        assert_eq!(data_column.value(1), "b");
        assert_eq!(data_column.value(2), "c");

        let struct_column = result
            .column(2)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap();
        assert!(struct_column.is_null(0));
        assert!(struct_column.is_null(1));
        assert!(struct_column.is_null(2));
    }

    pub fn source_record_batch() -> RecordBatch {
        RecordBatch::try_new(
            arrow_schema_promotion_addition_and_renaming_required(),
            vec![
                Arc::new(Int32Array::from(vec![Some(1001), Some(1002), Some(1003)])), // b
                Arc::new(Float32Array::from(vec![
                    Some(12.125),
                    Some(23.375),
                    Some(34.875),
                ])), // c
                Arc::new(Int32Array::from(vec![Some(2001), Some(2002), Some(2003)])), // d
                Arc::new(StringArray::from(vec![
                    Some("Apache"),
                    Some("Iceberg"),
                    Some("Rocks"),
                ])), // e
            ],
        )
        .unwrap()
    }

    pub fn source_record_batch_no_migration_required() -> RecordBatch {
        RecordBatch::try_new(
            arrow_schema_no_promotion_addition_or_renaming_required(),
            vec![
                Arc::new(Int32Array::from(vec![Some(2001), Some(2002), Some(2003)])), // d
                Arc::new(StringArray::from(vec![
                    Some("Apache"),
                    Some("Iceberg"),
                    Some("Rocks"),
                ])), // e
            ],
        )
        .unwrap()
    }

    pub fn expected_record_batch_migration_required() -> RecordBatch {
        RecordBatch::try_new(arrow_schema_already_same_as_target(), vec![
            Arc::new(StringArray::from(Vec::<Option<String>>::from([
                None, None, None,
            ]))), // a
            Arc::new(Int64Array::from(vec![Some(1001), Some(1002), Some(1003)])), // b
            Arc::new(Float64Array::from(vec![
                Some(12.125),
                Some(23.375),
                Some(34.875),
            ])), // c
            Arc::new(StringArray::from(vec![
                Some("Apache"),
                Some("Iceberg"),
                Some("Rocks"),
            ])), // e (d skipped by projection)
            Arc::new(StringArray::from(vec![
                Some("(╯°□°）╯"),
                Some("(╯°□°）╯"),
                Some("(╯°□°）╯"),
            ])), // f
        ])
        .unwrap()
    }

    pub fn iceberg_table_schema() -> Schema {
        Schema::builder()
            .with_schema_id(2)
            .with_fields(vec![
                NestedField::optional(10, "a", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(11, "b", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(12, "c", Type::Primitive(PrimitiveType::Double)).into(),
                NestedField::required(13, "d", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(14, "e", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(15, "f", Type::Primitive(PrimitiveType::String))
                    .with_initial_default(Literal::string("(╯°□°）╯"))
                    .into(),
            ])
            .build()
            .unwrap()
    }

    fn arrow_schema_already_same_as_target() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            simple_field("a", DataType::Utf8, true, "10"),
            simple_field("b", DataType::Int64, false, "11"),
            simple_field("c", DataType::Float64, false, "12"),
            simple_field("e", DataType::Utf8, true, "14"),
            simple_field("f", DataType::Utf8, false, "15"),
        ]))
    }

    fn arrow_schema_promotion_addition_and_renaming_required() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            simple_field("b", DataType::Int32, false, "11"),
            simple_field("c", DataType::Float32, false, "12"),
            simple_field("d", DataType::Int32, false, "13"),
            simple_field("e_old", DataType::Utf8, true, "14"),
        ]))
    }

    fn arrow_schema_no_promotion_addition_or_renaming_required() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            simple_field("d", DataType::Int32, false, "13"),
            simple_field("e", DataType::Utf8, true, "14"),
        ]))
    }

    /// Create a simple arrow field with metadata.
    fn simple_field(name: &str, ty: DataType, nullable: bool, value: &str) -> Field {
        Field::new(name, ty, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            value.to_string(),
        )]))
    }

    /// `add_files` over a Hive-style Parquet file with no field ids. `ArrowReader` has already
    /// applied the name mapping, so `name` and `subdept` arrive with ids 2 and 4.
    ///
    /// Risk pinned: the partition columns `id` and `dept` are absent from the file and must come
    /// from `initial_default`, spec rule 3. The two mapped columns must come from the file.
    #[test]
    fn add_files_with_name_mapping_applied_in_reader() {
        // Schema after add_files: id (partition), name, dept (partition), subdept.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int))
                        .with_initial_default(Literal::int(1))
                        .into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "dept", Type::Primitive(PrimitiveType::String))
                        .with_initial_default(Literal::string("hr"))
                        .into(),
                    NestedField::optional(4, "subdept", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        // The file held name and subdept with no ids. `reader.rs` mapped them to 2 and 4. The
        // partition columns id and dept live in the directory path, not the file.
        use std::collections::HashMap;
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, true).with_metadata(HashMap::from([(
                "PARQUET:field_id".to_string(),
                "2".to_string(),
            )])),
            Field::new("subdept", DataType::Utf8, true).with_metadata(HashMap::from([(
                "PARQUET:field_id".to_string(),
                "4".to_string(),
            )])),
        ]));

        let projected_field_ids = [1, 2, 3, 4]; // id, name, dept, subdept

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids).build();

        // The file's two columns.
        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(StringArray::from(vec!["John Doe"])),
            Arc::new(StringArray::from(vec!["communications"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // id and dept come from initial_default, name and subdept from the file.
        assert_eq!(result.num_columns(), 4);
        assert_eq!(result.num_rows(), 1);

        let id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_column.value(0), 1);

        let name_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_column.value(0), "John Doe");

        let dept_column = result
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(dept_column.value(0), "hr");

        let subdept_column = result
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(subdept_column.value(0), "communications");
    }

    /// Risk pinned: treating a bucket-partitioned field as a constant. Partition metadata holds
    /// `id_bucket = 2`, while the real `id` values 100, 200, 300 live only in the data file.
    /// Replacing the column with the bucket number breaks runtime filtering, so `WHERE id = 100`
    /// would match no rows. Java `PartitionUtil.constantsMap` filters on `isIdentity()`, which is
    /// false for a bucket transform.
    #[test]
    fn bucket_partitioning_reads_source_column_from_file() {
        use crate::spec::{Struct, Transform};

        // Schema: id and name are data columns, id_bucket is the partition column.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Partition spec: bucket(4, id).
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("id", "id_bucket", Transform::Bucket(4))
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition tuple: bucket value 2.
        let partition_data = Struct::from_iter(vec![Some(Literal::int(2))]);

        // The file carries both id and name.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("name", DataType::Utf8, true, "2"),
        ]));

        let projected_field_ids = [1, 2]; // id, name

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("Failed to add partition constants")
                .build();

        // The id column MUST be read from here, not treated as a constant.
        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200, 300])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // id must come from the file, not from partition metadata.
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);

        let id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        // These values MUST come from the file, not from a constant.
        assert_eq!(id_column.value(0), 100);
        assert_eq!(id_column.value(1), 200);
        assert_eq!(id_column.value(2), 300);

        let name_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_column.value(0), "Alice");
        assert_eq!(name_column.value(1), "Bob");
        assert_eq!(name_column.value(2), "Charlie");
    }

    /// The complement to `bucket_partitioning_reads_source_column_from_file`. An identity
    /// transform stores the real value, so `dept = "engineering"` comes from partition metadata
    /// and the file is never consulted. This is what makes a metadata-only Hive migration work.
    #[test]
    fn identity_partition_uses_constant_from_metadata() {
        use crate::spec::{Struct, Transform};

        // Schema: id and name are data columns, dept is the partition column.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Partition spec: identity(dept).
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("dept", "dept", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition tuple: dept="engineering".
        let partition_data = Struct::from_iter(vec![Some(Literal::string("engineering"))]);

        // The file carries only id and name. dept lives in the partition path.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("name", DataType::Utf8, true, "3"),
        ]));

        let projected_field_ids = [1, 2, 3]; // id, dept, name

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("Failed to add partition constants")
                .build();

        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // dept must carry the partition-metadata constant.
        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.num_rows(), 2);

        assert_eq!(get_int_value(result.column(0).as_ref(), 0), 100);
        assert_eq!(get_int_value(result.column(0).as_ref(), 1), 200);

        // dept comes from partition metadata, so it is REE.
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 0),
            "engineering"
        );
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 1),
            "engineering"
        );

        // name comes from the file.
        assert_eq!(get_string_value(result.column(2).as_ref(), 0), "Alice");
        assert_eq!(get_string_value(result.column(2).as_ref(), 1), "Bob");
    }

    /// Risk pinned: a partition tuple SHORTER than its spec used to index past the end of
    /// `Struct` and panic, which killed the scan task. Java `PartitionData.get` returns null past
    /// the end, so the field must instead stay out of the constants map and resolve as null.
    #[test]
    fn constants_map_tolerates_a_partition_tuple_shorter_than_the_spec() {
        use crate::spec::{Datum, PartitionSpec, Struct, Transform};

        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(3, "region", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .expect("build snapshot schema"),
        );

        // Two identity partition fields.
        let partition_spec = PartitionSpec::builder(snapshot_schema.clone())
            .with_spec_id(0)
            .add_partition_field("dept", "dept", Transform::Identity)
            .expect("add dept partition field")
            .add_partition_field("region", "region", Transform::Identity)
            .expect("add region partition field")
            .build()
            .expect("build partition spec");

        // A tuple carrying only the first value.
        let partition_data = Struct::from_iter(vec![Some(Literal::string("engineering"))]);

        let constants = constants_map(&partition_spec, &partition_data, &snapshot_schema)
            .expect("a short partition tuple must not fail the read");

        assert_eq!(
            constants.get(&2),
            Some(&Datum::string("engineering")),
            "the value the tuple DOES carry must still be used as a constant"
        );
        assert!(
            !constants.contains_key(&3),
            "the missing position must resolve as null (absent from the constants map), \
             not as some other field's value: {constants:?}"
        );
    }

    /// The bucket case after `RENAME COLUMN id TO row_id`. The file still names the column `id`,
    /// and both sides keep field id 1.
    ///
    /// Risk pinned: the lookup must match on field id, not on name, and must still read 100, 200,
    /// 300 from the file rather than the bucket constant 2.
    #[test]
    fn test_bucket_partitioning_with_renamed_source_column() {
        use crate::spec::{Struct, Transform};

        // Schema after the rename: row_id, name.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "row_id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Partition spec: bucket(4, row_id), with source_id still 1.
        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("row_id", "row_id_bucket", Transform::Bucket(4))
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition tuple: bucket value 2.
        let partition_data = Struct::from_iter(vec![Some(Literal::int(2))]);

        // The file keeps the OLD name "id" and the SAME field id 1.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("name", DataType::Utf8, true, "2"),
        ]));

        let projected_field_ids = [1, 2]; // row_id (field_id=1), name (field_id=2)

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("Failed to add partition constants")
                .build();

        // The data must be read through field id 1.
        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200, 300])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // The name mismatch must not change the result.
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);

        let row_id_column = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        // These values MUST come from the file through field id 1, not from the bucket constant.
        assert_eq!(row_id_column.value(0), 100);
        assert_eq!(row_id_column.value(1), 200);
        assert_eq!(row_id_column.value(2), 300);

        let name_column = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name_column.value(0), "Alice");
        assert_eq!(name_column.value(1), "Bob");
        assert_eq!(name_column.value(2), "Charlie");
    }
