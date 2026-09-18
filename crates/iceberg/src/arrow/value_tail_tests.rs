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

    #[test]
    fn test_create_decimal_array_respects_precision() {
        // Decimal128Array::from() uses Arrow's default precision (38) instead of the
        // target precision, causing RecordBatch construction to fail when schemas don't match.
        let target_precision = 18u8;
        let target_scale = 10i8;
        let target_type = DataType::Decimal128(target_precision, target_scale);
        let value = PrimitiveLiteral::Int128(10000000000);

        let array = create_primitive_array_single_element(&target_type, &Some(value))
            .expect("Failed to create decimal array");

        match array.data_type() {
            DataType::Decimal128(precision, scale) => {
                assert_eq!(*precision, target_precision);
                assert_eq!(*scale, target_scale);
            }
            other => panic!("Expected Decimal128, got {other:?}"),
        }
    }

    #[test]
    fn test_create_decimal_array_repeated_respects_precision() {
        // Ensure repeated arrays also respect target precision, not Arrow's default.
        let target_precision = 18u8;
        let target_scale = 10i8;
        let target_type = DataType::Decimal128(target_precision, target_scale);
        let value = PrimitiveLiteral::Int128(10000000000);
        let num_rows = 5;

        let array = create_primitive_array_repeated(&target_type, &Some(value), num_rows)
            .expect("Failed to create repeated decimal array");

        match array.data_type() {
            DataType::Decimal128(precision, scale) => {
                assert_eq!(*precision, target_precision);
                assert_eq!(*scale, target_scale);
            }
            other => panic!("Expected Decimal128, got {other:?}"),
        }

        assert_eq!(array.len(), num_rows);
    }

    #[test]
    fn test_create_timestamp_microsecond_array_repeated() {
        let target_type = DataType::Timestamp(TimeUnit::Microsecond, None);
        let value = PrimitiveLiteral::Long(1_740_600_000_000_000);
        let num_rows = 3;

        let array = create_primitive_array_repeated(&target_type, &Some(value), num_rows)
            .expect("Failed to create repeated timestamp microsecond array");

        assert_eq!(array.data_type(), &target_type);
        assert_eq!(array.len(), num_rows);
    }

    #[test]
    fn test_create_timestamp_microsecond_with_timezone_array_repeated() {
        let target_type = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let value = PrimitiveLiteral::Long(1_740_600_000_000_000);
        let num_rows = 2;

        let array = create_primitive_array_repeated(&target_type, &Some(value), num_rows)
            .expect("Failed to create repeated timestamp microsecond array with timezone");

        assert_eq!(array.data_type(), &target_type);
        assert_eq!(array.len(), num_rows);
    }

    // =============================================================================================
    // WG5 (a) — the required-field check must be evaluated relative to the ENCLOSING struct's
    // validity. Java never inspects a null struct's children on write:
    // `ValueWriters$OptionWriter.write` (iceberg-core 1.10.0 bytecode) is
    // `if (value == null) { encoder.writeIndex(nullIndex); }` — the value writer is NOT invoked,
    // so a `required` child under a NULL optional parent is unreachable and therefore legal.
    // The three tests below are a minimal pair + a control: they differ ONLY in the outer struct's
    // null bit / the child's value.
    // =============================================================================================

    /// Build `struct<1: outer optional struct<2: inner required int>>` as (arrow root struct,
    /// iceberg struct type), exactly the way the Avro writer does
    /// (`avro_writer::encode_batch_to_values` → `StructArray::from(batch)` →
    /// `arrow_struct_to_literal`). `outer_valid` is the outer struct's per-row validity and
    /// `inner` the child values (arrow-nullable, as a file-derived batch is).
    fn optional_struct_with_required_child(
        outer_valid: Vec<bool>,
        inner: Vec<Option<i32>>,
    ) -> (ArrayRef, StructType) {
        let inner_field = Arc::new(Field::new("inner", DataType::Int32, true).with_metadata(
            HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]),
        ));
        let inner_array = Arc::new(Int32Array::from(inner)) as ArrayRef;
        let outer_array = StructArray::try_new(
            Fields::from(vec![inner_field]),
            vec![inner_array],
            Some(NullBuffer::from(outer_valid)),
        )
        .expect("outer struct array");
        let outer_field = Arc::new(
            Field::new("outer", outer_array.data_type().clone(), true).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
            ),
        );
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![outer_field])),
            vec![Arc::new(outer_array) as ArrayRef],
        )
        .expect("record batch");
        let root = Arc::new(StructArray::from(batch)) as ArrayRef;

        let iceberg_type = StructType::new(vec![
            NestedField::optional(
                1,
                "outer",
                Type::Struct(StructType::new(vec![
                    NestedField::required(2, "inner", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
        ]);
        (root, iceberg_type)
    }

    #[test]
    fn test_required_child_null_under_null_parent_is_accepted() {
        // Row 0: outer struct is NULL and the child carries a null in that slot (the shape a
        // Parquet/Avro reader produces for a null struct). Java accepts it — the child is never
        // written. Row 1: outer valid, child 42.
        let (root, iceberg_type) =
            optional_struct_with_required_child(vec![false, true], vec![None, Some(42)]);

        let result = arrow_struct_to_literal(&root, &iceberg_type)
            .expect("a NULL optional struct must not make its required child a violation");

        assert_eq!(result, vec![
            Some(Literal::Struct(Struct::from_iter(vec![None]))),
            Some(Literal::Struct(Struct::from_iter(vec![Some(
                Literal::Struct(Struct::from_iter(vec![Some(Literal::int(42))]))
            )]))),
        ]);
    }

    #[test]
    fn test_required_child_null_under_valid_parent_still_rejected() {
        // The negative pin, and the minimal pair of the test above: the ONLY difference is the
        // outer struct's null bit at row 0. A NULL required child under a LIVE parent is a real
        // violation and must keep failing loudly.
        let (root, iceberg_type) =
            optional_struct_with_required_child(vec![true, true], vec![None, Some(42)]);

        let err = arrow_struct_to_literal(&root, &iceberg_type)
            .expect_err("a NULL required child under a LIVE parent must still be rejected");

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string()
                .contains("The field is required but has null value"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_null_parent_discards_stale_child_value() {
        // Control: a NULL parent whose child slot holds a live-but-meaningless value (arrow does
        // not require a null struct to mask its children). The row must decode to NULL — the
        // stale 7 must never surface.
        let (root, iceberg_type) =
            optional_struct_with_required_child(vec![false, true], vec![Some(7), Some(42)]);

        let result = arrow_struct_to_literal(&root, &iceberg_type)
            .expect("a NULL parent with a live child slot must decode");

        assert_eq!(result, vec![
            Some(Literal::Struct(Struct::from_iter(vec![None]))),
            Some(Literal::Struct(Struct::from_iter(vec![Some(
                Literal::Struct(Struct::from_iter(vec![Some(Literal::int(42))]))
            )]))),
        ]);
    }

    // FIX 4: the list/map offset slicing must turn degenerate buffers into typed errors, never
    // panic. `slice_list_by_offsets` is the shared core of the List / LargeList / Map paths.

    #[test]
    fn test_slice_list_by_offsets_empty_buffer_does_not_underflow() {
        // An empty offset buffer: `len() - 1` would underflow to usize::MAX (→ with_capacity
        // abort) without the `saturating_sub` guard. Must return an empty result, no panic.
        let offsets: Vec<i32> = Vec::new();
        let elements: Vec<Option<Literal>> = Vec::new();
        let result =
            slice_list_by_offsets(&offsets, &elements).expect("empty offsets must not panic");
        assert!(result.is_empty(), "empty offsets must yield no rows");
    }

    #[test]
    fn test_slice_list_by_offsets_non_monotonic_errors() {
        // end < start must error rather than slice-panic.
        let offsets: Vec<i32> = vec![5, 2];
        let elements: Vec<Option<Literal>> = vec![
            Some(Literal::int(0)),
            Some(Literal::int(1)),
            Some(Literal::int(2)),
        ];
        let err = slice_list_by_offsets(&offsets, &elements)
            .expect_err("non-monotonic offsets must error");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn test_slice_list_by_offsets_out_of_bounds_errors() {
        // end > elements.len() must error rather than slice-panic.
        let offsets: Vec<i32> = vec![0, 10];
        let elements: Vec<Option<Literal>> = vec![Some(Literal::int(0))];
        let err = slice_list_by_offsets(&offsets, &elements)
            .expect_err("out-of-bounds offsets must error");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn test_slice_list_by_offsets_valid() {
        // A well-formed buffer still reassembles correctly (no regression).
        let offsets: Vec<i32> = vec![0, 2, 3];
        let elements: Vec<Option<Literal>> = vec![
            Some(Literal::int(10)),
            Some(Literal::int(20)),
            Some(Literal::int(30)),
        ];
        let result =
            slice_list_by_offsets(&offsets, &elements).expect("valid offsets must convert");
        assert_eq!(result.len(), 2, "two rows expected from 3 offsets");
    }

    #[test]
    fn test_fixed_size_list_zero_width_errors() {
        use arrow_array::FixedSizeListArray;
        // A zero-width FixedSizeList would divide-by-zero in the conversion; build one via
        // `new_null` (which does not validate width > 0) and assert a typed error.
        let element_field = Arc::new(Field::new("item", DataType::Int32, true));
        let array: ArrayRef = Arc::new(FixedSizeListArray::new_null(element_field, 0, 0));
        let list_type = ListType {
            element_field: NestedField::list_element(1, Type::Primitive(PrimitiveType::Int), false)
                .into(),
        };
        let mut converter = ArrowArrayToIcebergStructConverter;
        let err = SchemaWithPartnerVisitor::list(&mut converter, &list_type, &array, Vec::new())
            .expect_err("zero-width FixedSizeList must error, not divide-by-zero");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }
