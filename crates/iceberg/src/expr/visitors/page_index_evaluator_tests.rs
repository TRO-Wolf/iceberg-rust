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
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::{
        ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelector,
    };
    use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData};
    use parquet::file::properties::WriterProperties;
    use rand::Rng;
    use tempfile::NamedTempFile;

    use crate::expr::visitors::page_index_evaluator::PageIndexEvaluator;
    use crate::expr::{Bind, Reference};
    use crate::spec::{Datum, NestedField, PrimitiveType, Schema, Type};
    use crate::{ErrorKind, Result};

    fn create_test_parquet_file() -> Result<(Arc<ParquetMetaData>, NamedTempFile)> {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("col_float", DataType::Float32, true),
            Field::new("col_string", DataType::Utf8, true),
        ]));

        let temp_file = NamedTempFile::new().unwrap();
        let file = temp_file.reopen().unwrap();

        let props = WriterProperties::builder()
            .set_data_page_row_count_limit(1024)
            .set_write_batch_size(512)
            .build();

        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();

        let mut batches = vec![];

        let float_vals: Vec<Option<f32>> = vec![None; 1024];
        let mut string_vals = vec![];
        string_vals.push(Some("AARDVARK".to_string()));
        for _ in 1..1023 {
            string_vals.push(Some("BEAR".to_string()));
        }
        string_vals.push(Some("BISON".to_string()));

        batches.push(
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Float32Array::from(float_vals)),
                Arc::new(StringArray::from(string_vals)),
            ])
            .unwrap(),
        );

        let float_vals: Vec<Option<f32>> = vec![None; 1024];
        let string_vals = vec![Some("DEER".to_string()); 1024];

        batches.push(
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Float32Array::from(float_vals)),
                Arc::new(StringArray::from(string_vals)),
            ])
            .unwrap(),
        );

        let mut float_vals = vec![];
        for i in 0..1024 {
            float_vals.push(Some(i as f32 * 10.0 / 1024.0));
        }
        let mut string_vals = vec![];
        string_vals.push(Some("GIRAFFE".to_string()));
        string_vals.push(None);
        for _ in 2..1024 {
            string_vals.push(Some("HIPPO".to_string()));
        }

        batches.push(
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Float32Array::from(float_vals)),
                Arc::new(StringArray::from(string_vals)),
            ])
            .unwrap(),
        );

        let mut float_vals = vec![None];
        for i in 1..1024 {
            float_vals.push(Some(10.0 + i as f32 * 10.0 / 1024.0));
        }
        let string_vals = vec![Some("HIPPO".to_string()); 1024];

        batches.push(
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Float32Array::from(float_vals)),
                Arc::new(StringArray::from(string_vals)),
            ])
            .unwrap(),
        );

        for batch in &batches {
            for i in 0..batch.num_rows() {
                writer.write(&batch.slice(i, 1)).unwrap();
            }
        }

        writer.close().unwrap();

        let file = temp_file.reopen().unwrap();
        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options).unwrap();
        let metadata = reader.metadata().clone();

        Ok((metadata, temp_file))
    }

    fn get_test_metadata(
        metadata: &ParquetMetaData,
    ) -> (
        Vec<parquet::file::page_index::column_index::ColumnIndexMetaData>,
        Vec<parquet::file::page_index::offset_index::OffsetIndexMetaData>,
        &parquet::file::metadata::RowGroupMetaData,
    ) {
        let row_group_metadata = metadata.row_group(0);
        let column_index = metadata.column_index().unwrap()[0].to_vec();
        let offset_index = metadata.offset_index().unwrap()[0].to_vec();
        (column_index, offset_index, row_group_metadata)
    }

    #[test]
    fn eval_matches_no_rows_for_empty_row_group() -> Result<()> {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("col_float", DataType::Float32, true),
            Field::new("col_string", DataType::Utf8, true),
        ]));

        let empty_float: ArrayRef = Arc::new(Float32Array::from(Vec::<Option<f32>>::new()));
        let empty_string: ArrayRef = Arc::new(StringArray::from(Vec::<Option<String>>::new()));
        let empty_batch =
            RecordBatch::try_new(arrow_schema.clone(), vec![empty_float, empty_string]).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        let file = temp_file.reopen().unwrap();

        let mut writer = ArrowWriter::try_new(file, arrow_schema, None).unwrap();
        writer.write(&empty_batch).unwrap();
        writer.close().unwrap();

        let file = temp_file.reopen().unwrap();
        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(file, options).unwrap();
        let metadata = reader.metadata();

        if metadata.num_row_groups() == 0 || metadata.row_group(0).num_rows() == 0 {
            return Ok(());
        }

        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .greater_than(Datum::float(1.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let row_group_metadata = metadata.row_group(0);
        let column_index = metadata.column_index().unwrap()[0].to_vec();
        let offset_index = metadata.offset_index().unwrap()[0].to_vec();

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        assert_eq!(result.len(), 0);

        Ok(())
    }

    #[test]
    fn eval_is_null_select_only_pages_with_nulls() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_null()
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![
            RowSelector::select(2048),
            RowSelector::skip(1024),
            RowSelector::select(1024),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_is_not_null_dont_select_pages_with_all_nulls() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_not_null()
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::skip(2048), RowSelector::select(2048)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_is_nan_select_all() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_nan()
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::select(4096)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_not_nan_select_all() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_not_nan()
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::select(4096)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_inequality_nan_datum_all_rows_except_all_null_pages() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .less_than(Datum::float(f32::NAN))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::skip(2048), RowSelector::select(2048)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_inequality_pages_containing_value_except_all_null_pages() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .less_than(Datum::float(5.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![
            RowSelector::skip(2048),
            RowSelector::select(1024),
            RowSelector::skip(1024),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_eq_pages_containing_value_except_all_null_pages() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .equal_to(Datum::float(5.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![
            RowSelector::skip(2048),
            RowSelector::select(1024),
            RowSelector::skip(1024),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_not_eq_all_rows() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .not_equal_to(Datum::float(5.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::select(4096)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_starts_with_error_float_col() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .starts_with(Datum::float(5.0))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        );

        assert_eq!(result.unwrap_err().kind(), ErrorKind::Unexpected);

        Ok(())
    }

    #[test]
    fn eval_starts_with_pages_containing_value_except_all_null_pages() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_string")
            .starts_with(Datum::string("B"))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::select(1024), RowSelector::skip(3072)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_not_starts_with_pages_containing_value_except_pages_with_min_and_max_equal_to_prefix_and_all_null_pages()
    -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_string")
            .not_starts_with(Datum::string("DE"))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![
            RowSelector::select(1024),
            RowSelector::skip(1024),
            RowSelector::select(2048),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_in_length_of_set_above_limit_all_rows() -> Result<()> {
        let mut rng = rand::rng();
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_float")
            .is_in(std::iter::repeat_with(|| Datum::float(rng.random_range(0.0..10.0))).take(1000))
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![RowSelector::select(4096)];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn eval_in_valid_set_size_some_rows() -> Result<()> {
        let (metadata, _temp_file) = create_test_parquet_file()?;
        let (column_index, offset_index, row_group_metadata) = get_test_metadata(&metadata);
        let (iceberg_schema_ref, field_id_map) = build_iceberg_schema_and_field_map()?;

        let filter = Reference::new("col_string")
            .is_in([Datum::string("AARDVARK"), Datum::string("GIRAFFE")])
            .bind(iceberg_schema_ref.clone(), false)?;

        let result = PageIndexEvaluator::eval(
            &filter,
            &column_index,
            &offset_index,
            row_group_metadata,
            &field_id_map,
            iceberg_schema_ref.as_ref(),
        )?;

        let expected = vec![
            RowSelector::select(1024),
            RowSelector::skip(1024),
            RowSelector::select(1024),
            RowSelector::skip(1024),
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    fn build_iceberg_schema_and_field_map() -> Result<(Arc<Schema>, HashMap<i32, usize>)> {
        let iceberg_schema = Schema::builder()
            .with_fields([
                Arc::new(NestedField::new(
                    1,
                    "col_float",
                    Type::Primitive(PrimitiveType::Float),
                    false,
                )),
                Arc::new(NestedField::new(
                    2,
                    "col_string",
                    Type::Primitive(PrimitiveType::String),
                    false,
                )),
            ])
            .build()?;
        let iceberg_schema_ref = Arc::new(iceberg_schema);

        let field_id_map = HashMap::from_iter([(1, 0), (2, 1)]);

        Ok((iceberg_schema_ref, field_id_map))
    }
}
