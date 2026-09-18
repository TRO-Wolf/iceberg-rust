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

    fn row_lineage_schema() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    /// A batch of `id` values, optionally carrying a reserved row-lineage column.
    fn row_lineage_batch(
        ids: Vec<i64>,
        extra: Option<(i32, &str, Vec<Option<i64>>)>,
    ) -> RecordBatch {
        let mut fields = vec![simple_field("id", DataType::Int64, false, "1")];
        let mut columns: Vec<arrow_array::ArrayRef> = vec![Arc::new(Int64Array::from(ids))];
        if let Some((field_id, name, values)) = extra {
            fields.push(simple_field(
                name,
                DataType::Int64,
                true,
                &field_id.to_string(),
            ));
            columns.push(Arc::new(Int64Array::from(values)));
        }
        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).unwrap()
    }

    fn int64_col(batch: &RecordBatch, index: usize) -> Vec<Option<i64>> {
        let array = batch
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column");
        (0..array.len())
            .map(|row| {
                if array.is_null(row) {
                    None
                } else {
                    Some(array.value(row))
                }
            })
            .collect()
    }

    #[test]
    fn row_id_is_computed_from_first_row_id_and_position_when_absent_from_the_file() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let first = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2, 3], None))
            .unwrap();
        assert_eq!(int64_col(&first, 1), vec![Some(100), Some(101), Some(102)]);

        // The counter CONTINUES across batches. A restart repeats the same row ids.
        let second = transformer
            .process_record_batch(row_lineage_batch(vec![4, 5], None))
            .unwrap();
        assert_eq!(int64_col(&second, 1), vec![Some(103), Some(104)]);
    }

    #[test]
    fn row_id_stored_in_the_file_wins_over_the_computed_value() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2, 3],
                Some((RESERVED_FIELD_ID_ROW_ID, "_row_id", vec![
                    Some(900),
                    Some(901),
                    Some(902),
                ])),
            ))
            .unwrap();
        assert_eq!(
            int64_col(&batch, 1),
            vec![Some(900), Some(901), Some(902)],
            "a file that carries row ids keeps them — they are the rows' durable identity, and \
             recomputing would renumber rows that were carried through a rewrite"
        );
    }

    /// The discriminating cell: stored and computed values INTERLEAVE within one batch.
    #[test]
    fn a_null_row_id_in_the_file_falls_back_to_first_row_id_plus_position() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2, 3, 4],
                Some((RESERVED_FIELD_ID_ROW_ID, "_row_id", vec![
                    Some(900),
                    None,
                    Some(902),
                    None,
                ])),
            ))
            .unwrap();
        assert_eq!(
            int64_col(&batch, 1),
            vec![Some(900), Some(101), Some(902), Some(103)],
            "each NULL takes `first_row_id + ITS OWN position` (101 at row 1, 103 at row 3) — not \
             a running count of the nulls, and not the whole column recomputed"
        );
    }

    /// Java returns an all-NULL column here, not an error. An error would make `SELECT _row_id`
    /// unusable on a mixed-version table.
    #[test]
    fn projecting_row_id_without_an_assigned_range_yields_nulls() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(None, Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect("no assigned range is not an error");
        assert_eq!(int64_col(&batch, 1), vec![None, None]);
    }

    #[test]
    fn last_updated_sequence_number_is_the_files_own_when_absent_from_the_file() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .unwrap();
        assert_eq!(
            int64_col(&batch, 1),
            vec![Some(7), Some(7)],
            "a constant per file — the file's own sequence number, NOT the row position"
        );
    }

    /// The discriminating cell for the sequence column.
    #[test]
    fn a_null_last_updated_sequence_number_falls_back_to_the_files_own() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2, 3],
                Some((
                    RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
                    "_last_updated_sequence_number",
                    vec![Some(3), None, Some(5)],
                )),
            ))
            .unwrap();
        assert_eq!(
            int64_col(&batch, 1),
            vec![Some(3), Some(7), Some(5)],
            "the stored per-row value wins; only the NULL takes the file's sequence number"
        );
    }

    /// The discriminating cell for the absent-range arm. The stored column is IGNORED, because
    /// the arm is chosen before the file is consulted.
    #[test]
    fn a_stored_row_id_is_discarded_when_there_is_no_assigned_range() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(None, Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2, 3],
                Some((RESERVED_FIELD_ID_ROW_ID, "_row_id", vec![
                    Some(900),
                    Some(901),
                    Some(902),
                ])),
            ))
            .expect("no assigned range is not an error");
        assert_eq!(
            int64_col(&batch, 1),
            vec![None, None, None],
            "no assigned range means NO row identity — the stored column is discarded, not \
             preferred. Java reaches `constant(null)` without consulting the file at all."
        );
    }

    /// The same cell for the sequence column: a stored value is discarded when the gate fails.
    #[test]
    fn a_stored_last_updated_sequence_number_is_discarded_without_a_first_row_id() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(None, Some(5))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(
                vec![1, 2],
                Some((
                    RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
                    "_last_updated_sequence_number",
                    vec![Some(31), Some(33)],
                )),
            ))
            .expect("a missing first_row_id is not an error");
        assert_eq!(
            int64_col(&batch, 1),
            vec![None, None],
            "Java gates on BOTH inputs BEFORE reading the column, so a stored value is discarded"
        );
    }

    /// Java gates `_last_updated_sequence_number` on BOTH inputs, so a V1 or V2 file reports
    /// NULL. The sequence number alone fabricates a value for every pre-V3 row.
    #[test]
    fn last_updated_sequence_number_is_null_without_a_first_row_id() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(None, Some(5))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect("a missing first_row_id is not an error");
        assert_eq!(
            int64_col(&batch, 1),
            vec![None, None],
            "NULL, not the file's sequence number — Java gates on BOTH inputs"
        );
    }

    /// The other half of the same gate: no file sequence number is also NULL.
    #[test]
    fn last_updated_sequence_number_is_null_without_a_file_sequence_number() {
        let projected = [1, RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), None)
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1], None))
            .expect("a missing file sequence number is not an error");
        assert_eq!(int64_col(&batch, 1), vec![None]);
    }

    /// The `num_rows == 0` guard in `row_ids_from_positions` is load-bearing. Without it
    /// `num_rows - 1` underflows on an ordinary empty batch.
    #[test]
    fn an_empty_batch_yields_an_empty_row_id_column() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(100), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(Vec::new(), None))
            .expect("an empty batch is not an error");
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(int64_col(&batch, 1), Vec::<Option<i64>>::new());

        // The counter is unmoved, so the NEXT batch starts at the range's first id.
        let next = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect("second batch");
        assert_eq!(int64_col(&next, 1), vec![Some(100), Some(101)]);
    }

    /// The boundary the overflow check must NOT reject: a batch whose last id is exactly
    /// `i64::MAX`. Only here does `start + num_rows` differ from `start + num_rows - 1`.
    #[test]
    fn a_row_id_of_exactly_i64_max_is_allowed() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(i64::MAX - 1), Some(7))
            .build();

        let batch = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect("the last id is exactly i64::MAX, which is representable");
        assert_eq!(int64_col(&batch, 1), vec![
            Some(i64::MAX - 1),
            Some(i64::MAX)
        ]);
    }

    /// Fail closed instead of wrapping into a negative row id (Java's `long` addition wraps).
    #[test]
    fn a_row_id_computation_that_overflows_i64_is_refused() {
        let projected = [1, RESERVED_FIELD_ID_ROW_ID];
        let mut transformer = RecordBatchTransformerBuilder::new(row_lineage_schema(), &projected)
            .with_row_lineage(Some(i64::MAX), Some(7))
            .build();

        let error = transformer
            .process_record_batch(row_lineage_batch(vec![1, 2], None))
            .expect_err("i64::MAX + 2 has no representable row id");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("overflowed i64"),
            "got: {}",
            error.message()
        );
    }
