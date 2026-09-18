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

    /// All four Column Projection rules in one batch.
    ///
    /// | Column | Rule | Source |
    /// |---|---|---|
    /// | dept | 1 | the identity-partition constant |
    /// | data | 2 | the file, through the name mapping |
    /// | category | 3 | `initial_default` |
    /// | notes | 4 | null |
    #[test]
    fn test_all_four_spec_rules() {
        use crate::spec::Transform;

        // One column per spec rule.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    // The normal case: found in the file by field id.
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    // Rule 1: identity-partitioned.
                    NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
                    // Rule 2: resolved by the name mapping.
                    NestedField::required(3, "data", Type::Primitive(PrimitiveType::String)).into(),
                    // Rule 3: has an initial_default.
                    NestedField::optional(4, "category", Type::Primitive(PrimitiveType::String))
                        .with_initial_default(Literal::string("default_category"))
                        .into(),
                    // Rule 4: no default, so null.
                    NestedField::optional(5, "notes", Type::Primitive(PrimitiveType::String))
                        .into(),
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

        // The post-ArrowReader file schema: id (1) and data (3). dept, category, and notes are
        // absent.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("data", DataType::Utf8, false, "3"),
        ]));

        let projected_field_ids = [1, 2, 3, 4, 5]; // id, dept, data, category, notes

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("Failed to add partition constants")
                .build();

        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![100, 200])),
            Arc::new(StringArray::from(vec!["value1", "value2"])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        assert_eq!(result.num_columns(), 5);
        assert_eq!(result.num_rows(), 2);

        // Each column below demonstrates one spec rule.

        // The normal case: id from the file by field id.
        assert_eq!(get_int_value(result.column(0).as_ref(), 0), 100);
        assert_eq!(get_int_value(result.column(0).as_ref(), 1), 200);

        // Rule 1: dept from partition metadata, so REE.
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 0),
            "engineering"
        );
        assert_eq!(
            get_string_value(result.column(1).as_ref(), 1),
            "engineering"
        );

        // Rule 2: data from the file, so a plain array.
        assert_eq!(get_string_value(result.column(2).as_ref(), 0), "value1");
        assert_eq!(get_string_value(result.column(2).as_ref(), 1), "value2");

        // Rule 3: category from initial_default, so REE.
        assert_eq!(
            get_string_value(result.column(3).as_ref(), 0),
            "default_category"
        );
        assert_eq!(
            get_string_value(result.column(3).as_ref(), 1),
            "default_category"
        );

        // Rule 4: notes is a null REE column.
        assert_eq!(get_string_value(result.column(4).as_ref(), 0), "");
        assert_eq!(get_string_value(result.column(4).as_ref(), 1), "");
    }

    /// Risk pinned: a null value in an identity-partitioned column used to error. It must
    /// materialize as a null column.
    #[test]
    fn null_identity_partition_value() {
        use crate::spec::{Struct, Transform};

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(schema.clone())
                .with_spec_id(0)
                .add_partition_field("data", "data", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // The partition tuple holds a null.
        let partition_data = Struct::from_iter(vec![None]);

        let file_schema = Arc::new(ArrowSchema::new(vec![simple_field(
            "id",
            DataType::Int32,
            true,
            "1",
        )]));

        let projected_field_ids = [1, 2];

        let mut transformer = RecordBatchTransformerBuilder::new(schema, &projected_field_ids)
            .with_partition(partition_spec, partition_data)
            .expect("Should handle null partition values")
            .build();

        let file_batch =
            RecordBatch::try_new(file_schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
                .unwrap();

        let result = transformer.process_record_batch(file_batch).unwrap();

        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);

        let id_col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id_col.values(), &[1, 2, 3]);

        // The partition column must produce nulls.
        let data_col = result.column(1);
        assert!(data_col.is_null(0));
        assert!(data_col.is_null(1));
        assert!(data_col.is_null(2));
    }

    /// Risk pinned: the REE leak. An identity-partition column exists in the table schema, so the
    /// output batch must declare its plain physical type, `Utf8` here, never `RunEndEncoded`.
    /// Materializing the constant as REE made the output schema disagree with the scan schema.
    ///
    /// The test asserts the EXACT physical type, not just the value. A `get_string_value` helper
    /// would pass under REE too.
    #[test]
    fn identity_partition_constant_is_plain_array_not_run_end_encoded() {
        use arrow_schema::DataType;

        use crate::spec::Transform;

        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "category", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("category", "category", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        let partition_data = Struct::from_iter(vec![Some(Literal::string("electronics"))]);

        // The file lacks the partition column, as in a Hive migration. It carries only `id`.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![simple_field(
            "id",
            DataType::Int32,
            false,
            "1",
        )]));

        let projected_field_ids = [1, 2];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("partition constants")
                .build();

        let parquet_batch =
            RecordBatch::try_new(parquet_schema, vec![Arc::new(Int32Array::from(vec![
                1, 2, 3,
            ]))])
            .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // The declared field for `category` must be plain Utf8, never RunEndEncoded.
        assert_eq!(
            result.schema().field(1).data_type(),
            &DataType::Utf8,
            "identity-partition constant must declare its plain scan-schema type, not REE"
        );

        // The materialized column must be a plain StringArray, not a RunArray.
        let category = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("category column must be a plain StringArray, not RunEndEncoded");
        assert_eq!(category.value(0), "electronics");
        assert_eq!(category.value(1), "electronics");
        assert_eq!(category.value(2), "electronics");
    }

    /// Risk pinned: the int-to-long widening bug. A partition tuple can carry a literal narrower
    /// than a type-promoted column, such as `Int(19)` for a `Long` column. `Datum::to` must
    /// coerce the value to the FIELD's type, like Java `IdentityPartitionConverters
    /// .convertConstant`. Without it the array builder sees `(Int64, Int(19))` and errors.
    #[test]
    fn identity_partition_widens_int_literal_to_long_column() {
        use arrow_schema::DataType;

        use crate::spec::Transform;

        // Column `p` is Long, but the tuple still stores the narrower Int(19) variant.
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "p", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("p", "p", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // The partition value in its NARROW Int(19) form.
        let partition_data = Struct::from_iter(vec![Some(Literal::int(19))]);

        let parquet_schema = Arc::new(ArrowSchema::new(vec![simple_field(
            "id",
            DataType::Int32,
            false,
            "1",
        )]));

        let projected_field_ids = [1, 2];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("partition constants must widen Int(i32) to a Long column")
                .build();

        let parquet_batch =
            RecordBatch::try_new(parquet_schema, vec![Arc::new(Int32Array::from(vec![7, 8]))])
                .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // `p` must materialize as a plain Int64 column with the widened value.
        assert_eq!(result.schema().field(1).data_type(), &DataType::Int64);
        let p_col = result
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("p column must be a plain Int64Array");
        assert_eq!(p_col.value(0), 19);
        assert_eq!(p_col.value(1), 19);
    }

    /// Risk pinned: a REORDERED and SUBSET projection must give an output schema exactly equal to
    /// the declared projection, in names, plain physical types, nullability, AND order. The
    /// constant column must be a plain array carrying the PARTITION value. The reordered shape
    /// already forces the `Modify` path, so this does not isolate the
    /// `constant_overrides_file_column` flag.
    #[test]
    fn identity_partition_reordered_subset_projection_matches_declared_schema() {
        use arrow_schema::DataType;

        use crate::spec::Transform;

        // Schema order: id(1, Int), category(2, String, partitioned), extra(3, Long).
        let snapshot_schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "category", Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::optional(3, "extra", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = Arc::new(
            crate::spec::PartitionSpec::builder(snapshot_schema.clone())
                .with_spec_id(0)
                .add_partition_field("category", "category", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        // The partition value DIFFERS from the file's, so the override is observable.
        let partition_data = Struct::from_iter(vec![Some(Literal::string("books"))]);

        // The file carries ALL THREE columns, add_files style, with a different `category`.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![
            simple_field("id", DataType::Int32, false, "1"),
            simple_field("category", DataType::Utf8, false, "2"),
            simple_field("extra", DataType::Int64, true, "3"),
        ]));

        // Project category(2) first, id(1) second, and drop extra(3).
        let projected_field_ids = [2, 1];

        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema, &projected_field_ids)
                .with_partition(partition_spec, partition_data)
                .expect("partition constants")
                .build();

        let parquet_batch = RecordBatch::try_new(parquet_schema, vec![
            Arc::new(Int32Array::from(vec![10, 11])),
            Arc::new(StringArray::from(vec![
                "ignored_file_value",
                "ignored_file_value",
            ])),
            Arc::new(Int64Array::from(vec![100, 200])),
        ])
        .unwrap();

        let result = transformer.process_record_batch(parquet_batch).unwrap();

        // The output schema must be exactly [category: Utf8, id: Int32], both non-null.
        assert_eq!(result.num_columns(), 2);
        let sch = result.schema();
        assert_eq!(sch.field(0).name(), "category");
        assert_eq!(sch.field(0).data_type(), &DataType::Utf8);
        assert!(!sch.field(0).is_nullable());
        assert_eq!(sch.field(1).name(), "id");
        assert_eq!(sch.field(1).data_type(), &DataType::Int32);
        assert!(!sch.field(1).is_nullable());

        // category is the constant plain StringArray, and it OVERRIDES the file value.
        let category = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("category must be a plain StringArray, not REE");
        assert_eq!(category.value(0), "books");
        assert_eq!(category.value(1), "books");

        // id comes from the file unchanged.
        let id = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id.value(0), 10);
        assert_eq!(id.value(1), 11);
    }
