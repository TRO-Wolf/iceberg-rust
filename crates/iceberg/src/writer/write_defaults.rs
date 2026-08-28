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

//! Fill missing Iceberg columns from `write-default` before a data file is written.
//!
//! Spec: a writer must emit every known field. A missing field takes `write-default`.
//! A required field with no `write-default` fails. An optional field with none writes null.

use std::borrow::Cow;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeBinaryArray, LargeBinaryArray, RecordBatch, Time64MicrosecondArray,
    new_null_array,
};
use arrow_schema::{DataType, Field, TimeUnit};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use uuid::Uuid;

use crate::arrow::{create_primitive_array_repeated, schema_to_arrow_schema};
use crate::spec::{Literal, NestedField, PrimitiveLiteral, Schema};
use crate::{Error, ErrorKind, Result};

/// Project `batch` onto `schema`, filling any missing field from `write-default`.
///
/// A complete batch in Iceberg field order is returned borrowed. Extra batch columns
/// are dropped. Nested `write-default` fill is refused (row R92 residue).
///
/// # Errors
///
/// [`ErrorKind::DataInvalid`] when a required field is missing and has no `write-default`.
/// [`ErrorKind::FeatureUnsupported`] when a missing field's `write-default` is not primitive.
pub(crate) fn apply_write_defaults<'a>(
    schema: &Schema,
    batch: &'a RecordBatch,
) -> Result<Cow<'a, RecordBatch>> {
    let iceberg_fields = schema.as_struct().fields();
    if batch.num_rows() == 0 {
        let target_schema = schema_to_arrow_schema(schema)?;
        return Ok(Cow::Owned(RecordBatch::new_empty(Arc::new(target_schema))));
    }
    if batch_matches_schema_order(iceberg_fields, batch) {
        return Ok(Cow::Borrowed(batch));
    }

    let target_schema = schema_to_arrow_schema(schema)?;
    let num_rows = batch.num_rows();
    let mut columns = Vec::with_capacity(iceberg_fields.len());
    for (iceberg_field, arrow_field) in iceberg_fields.iter().zip(target_schema.fields()) {
        if let Some(idx) = batch_column_index(batch, iceberg_field) {
            columns.push(batch.column(idx).clone());
        } else {
            columns.push(fill_missing_column(iceberg_field, arrow_field, num_rows)?);
        }
    }

    Ok(Cow::Owned(RecordBatch::try_new(
        Arc::new(target_schema),
        columns,
    )?))
}

fn batch_matches_schema_order(fields: &[crate::spec::NestedFieldRef], batch: &RecordBatch) -> bool {
    if batch.num_columns() != fields.len() {
        return false;
    }
    fields
        .iter()
        .enumerate()
        .all(|(i, field)| batch_field_id(batch.schema().field(i)) == Some(field.id))
}

fn batch_field_id(field: &Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .and_then(|value| value.parse().ok())
}

fn batch_column_index(batch: &RecordBatch, field: &NestedField) -> Option<usize> {
    let schema = batch.schema();
    for (idx, arrow_field) in schema.fields().iter().enumerate() {
        if batch_field_id(arrow_field) == Some(field.id) {
            return Some(idx);
        }
    }
    schema.fields().iter().position(|arrow_field| {
        arrow_field.name() == field.name.as_str() && batch_field_id(arrow_field).is_none()
    })
}

fn fill_missing_column(
    field: &NestedField,
    arrow_field: &Field,
    num_rows: usize,
) -> Result<ArrayRef> {
    match field.write_default.as_ref() {
        Some(Literal::Primitive(prim)) => {
            repeat_write_default(arrow_field.data_type(), prim, num_rows).map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot apply write-default for field '{}' (id {}): {}",
                        field.name,
                        field.id,
                        err.message()
                    ),
                )
            })
        }
        Some(_) => Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "write-default fill for non-primitive field '{}' is not supported",
                field.name
            ),
        )),
        None if field.required => Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Required field '{}' (id {}) is missing from the write batch and has no write-default",
                field.name, field.id
            ),
        )),
        None => Ok(new_null_array(arrow_field.data_type(), num_rows)),
    }
}

fn repeat_write_default(
    data_type: &DataType,
    prim: &PrimitiveLiteral,
    num_rows: usize,
) -> Result<ArrayRef> {
    match (data_type, prim) {
        (DataType::Time64(TimeUnit::Microsecond), PrimitiveLiteral::Long(value)) => {
            Ok(Arc::new(Time64MicrosecondArray::from(vec![
                *value;
                num_rows
            ])))
        }
        (DataType::FixedSizeBinary(16), PrimitiveLiteral::UInt128(value)) => {
            let bytes = Uuid::from_u128(*value).into_bytes();
            let values: Vec<&[u8]> = vec![bytes.as_slice(); num_rows];
            Ok(Arc::new(FixedSizeBinaryArray::try_from_iter(
                values.into_iter(),
            )?))
        }
        (DataType::FixedSizeBinary(width), PrimitiveLiteral::Binary(bytes))
            if bytes.len() == *width as usize =>
        {
            let values: Vec<&[u8]> = vec![bytes.as_slice(); num_rows];
            Ok(Arc::new(FixedSizeBinaryArray::try_from_iter(
                values.into_iter(),
            )?))
        }
        (DataType::LargeBinary, PrimitiveLiteral::Binary(bytes)) => Ok(Arc::new(
            LargeBinaryArray::from_iter_values(std::iter::repeat_n(bytes.as_slice(), num_rows)),
        )),
        _ => create_primitive_array_repeated(data_type, &Some(prim.clone()), num_rows),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Array, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::io::FileIO;
    use crate::spec::{DataFileFormat, NestedField, PrimitiveLiteral, PrimitiveType, Schema, Type};
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};

    fn id_meta(id: i32) -> HashMap<String, String> {
        HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())])
    }

    fn schema_id_and_name() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                    .with_write_default(Literal::string("anon"))
                    .into(),
            ])
            .build()
            .expect("schema")
    }

    fn id_only_batch() -> RecordBatch {
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
        ]);
        RecordBatch::try_new(Arc::new(arrow_schema), vec![Arc::new(Int32Array::from(
            vec![1, 2, 3],
        ))])
        .expect("id-only batch")
    }

    fn complete_batch() -> RecordBatch {
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
            Field::new("name", DataType::Utf8, true).with_metadata(id_meta(2)),
        ]);
        RecordBatch::try_new(Arc::new(arrow_schema), vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
        ])
        .expect("complete batch")
    }

    #[test]
    fn missing_optional_write_default_is_filled() {
        let schema = schema_id_and_name();
        let batch = id_only_batch();
        let filled = apply_write_defaults(&schema, &batch).expect("fill");
        let names = filled
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("name col");
        assert_eq!(names.value(0), "anon");
        assert_eq!(names.value(1), "anon");
        assert_eq!(names.value(2), "anon");
    }

    #[test]
    fn supplied_column_is_not_replaced() {
        let schema = schema_id_and_name();
        let batch = complete_batch();
        let filled = apply_write_defaults(&schema, &batch).expect("fill");
        let names = filled
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("name col");
        assert_eq!(names.value(0), "a");
        assert_eq!(names.value(1), "b");
        assert!(matches!(filled, Cow::Borrowed(_)));
    }

    #[test]
    fn missing_required_without_write_default_fails() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("schema");
        let err = apply_write_defaults(&schema, &id_only_batch()).expect_err("must fail");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("write-default"),
            "got {}",
            err.message()
        );
    }

    #[test]
    fn missing_optional_without_write_default_is_null() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("schema");
        let batch = id_only_batch();
        let filled = apply_write_defaults(&schema, &batch).expect("null fill");
        let names = filled
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("name col");
        assert!(names.is_null(0));
        assert!(names.is_null(1));
        assert!(names.is_null(2));
    }

    #[test]
    fn missing_required_write_default_is_filled() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String))
                    .with_write_default(Literal::string("x"))
                    .into(),
            ])
            .build()
            .expect("schema");
        let batch = id_only_batch();
        let filled = apply_write_defaults(&schema, &batch).expect("fill");
        let names = filled
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("name col");
        assert_eq!(names.value(0), "x");
    }

    #[test]
    fn name_fallback_matches_when_field_id_is_absent() {
        let schema = schema_id_and_name();
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]);
        let batch = RecordBatch::try_new(Arc::new(arrow_schema), vec![
            Arc::new(Int32Array::from(vec![9])),
            Arc::new(StringArray::from(vec![Some("kept")])),
        ])
        .expect("name-only batch");
        let filled = apply_write_defaults(&schema, &batch).expect("name match");
        let names = filled
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("name col");
        assert_eq!(names.value(0), "kept");
    }

    #[test]
    fn non_primitive_write_default_on_missing_field_fails() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(crate::spec::StructType::new(vec![
                        NestedField::optional(3, "n", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .with_write_default(Literal::Struct(crate::spec::Struct::from_iter([Some(
                    Literal::int(1),
                )])))
                .into(),
            ])
            .build()
            .expect("schema");
        let err = apply_write_defaults(&schema, &id_only_batch()).expect_err("nested");
        assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
        assert!(err.message().contains("non-primitive"));
    }

    #[tokio::test]
    async fn data_file_writer_writes_write_default_into_parquet() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = Arc::new(schema_id_and_name());
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().expect("utf8 path").to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("wd".to_string(), None, DataFileFormat::Parquet);
        let parquet = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema);
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );
        let mut writer = DataFileWriterBuilder::new(rolling)
            .unpartitioned()
            .build(None)
            .await
            .expect("build writer");
        writer
            .write(id_only_batch())
            .await
            .expect("write missing name");
        let data_files = writer.close().await.expect("close");
        assert_eq!(data_files.len(), 1, "one data file");

        let input = file_io
            .new_input(data_files[0].file_path.clone())
            .expect("input")
            .read()
            .await
            .expect("read parquet");
        let reader = ParquetRecordBatchReaderBuilder::try_new(input)
            .expect("parquet reader")
            .build()
            .expect("build reader");
        let batches: Vec<_> = reader.map(|b| b.expect("batch")).collect();
        assert_eq!(batches.len(), 1);
        let names = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("name col");
        assert_eq!(names.value(0), "anon");
        assert_eq!(names.value(1), "anon");
        assert_eq!(names.value(2), "anon");
        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id col");
        assert_eq!(ids.value(0), 1);
        assert_eq!(ids.value(1), 2);
        assert_eq!(ids.value(2), 3);
    }

    #[test]
    fn missing_binary_write_default_is_filled() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::Binary))
                    .with_write_default(Literal::Primitive(PrimitiveLiteral::Binary(vec![
                        0x01, 0x02,
                    ])))
                    .into(),
            ])
            .build()
            .expect("schema");
        let batch = id_only_batch();
        let filled = apply_write_defaults(&schema, &batch).expect("binary fill");
        let col = filled
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .expect("large binary");
        assert_eq!(col.value(0), &[0x01, 0x02]);
        assert_eq!(col.value(1), &[0x01, 0x02]);
    }

    #[test]
    fn missing_time_write_default_is_filled() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "t", Type::Primitive(PrimitiveType::Time))
                    .with_write_default(Literal::Primitive(PrimitiveLiteral::Long(1_000)))
                    .into(),
            ])
            .build()
            .expect("schema");
        let batch = id_only_batch();
        let filled = apply_write_defaults(&schema, &batch).expect("time fill");
        let col = filled
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::Time64MicrosecondArray>()
            .expect("time");
        assert_eq!(col.value(0), 1_000);
    }

    #[test]
    fn missing_uuid_write_default_is_filled() {
        let uuid = uuid::Uuid::parse_str("ec5911be-b0a7-458c-8438-c9a3e53cffae").expect("uuid");
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Uuid))
                    .with_write_default(Literal::Primitive(PrimitiveLiteral::UInt128(
                        uuid.as_u128(),
                    )))
                    .into(),
            ])
            .build()
            .expect("schema");
        let batch = id_only_batch();
        let filled = apply_write_defaults(&schema, &batch).expect("uuid fill");
        let col = filled
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
            .expect("uuid bytes");
        assert_eq!(col.value(0), uuid.as_bytes());
    }

    #[test]
    fn missing_fixed_write_default_is_filled() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "f", Type::Primitive(PrimitiveType::Fixed(4)))
                    .with_write_default(Literal::Primitive(PrimitiveLiteral::Binary(vec![
                        9, 8, 7, 6,
                    ])))
                    .into(),
            ])
            .build()
            .expect("schema");
        let batch = id_only_batch();
        let filled = apply_write_defaults(&schema, &batch).expect("fixed fill");
        let col = filled
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
            .expect("fixed");
        assert_eq!(col.value(0), &[9, 8, 7, 6]);
    }

    #[test]
    fn zero_row_omitted_uuid_write_default_is_empty_not_error() {
        let uuid = uuid::Uuid::parse_str("ec5911be-b0a7-458c-8438-c9a3e53cffae").expect("uuid");
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Uuid))
                    .with_write_default(Literal::Primitive(PrimitiveLiteral::UInt128(
                        uuid.as_u128(),
                    )))
                    .into(),
            ])
            .build()
            .expect("schema");
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
        ]);
        let batch = RecordBatch::new_empty(Arc::new(arrow_schema));
        let filled = apply_write_defaults(&schema, &batch).expect("0-row uuid fill");
        assert_eq!(filled.num_rows(), 0);
        assert_eq!(filled.num_columns(), 2);
        assert_eq!(
            filled.schema().field(1).data_type(),
            &DataType::FixedSizeBinary(16)
        );
    }

    #[test]
    fn zero_row_omitted_fixed_write_default_is_empty_not_error() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "f", Type::Primitive(PrimitiveType::Fixed(4)))
                    .with_write_default(Literal::Primitive(PrimitiveLiteral::Binary(vec![
                        9, 8, 7, 6,
                    ])))
                    .into(),
            ])
            .build()
            .expect("schema");
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(id_meta(1)),
        ]);
        let batch = RecordBatch::new_empty(Arc::new(arrow_schema));
        let filled = apply_write_defaults(&schema, &batch).expect("0-row fixed fill");
        assert_eq!(filled.num_rows(), 0);
        assert_eq!(filled.num_columns(), 2);
        assert_eq!(
            filled.schema().field(1).data_type(),
            &DataType::FixedSizeBinary(4)
        );
    }

    #[test]
    fn type_mismatched_write_default_is_data_invalid() {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))
                    .with_write_default(Literal::int(7))
                    .into(),
            ])
            .build()
            .expect("schema");
        let batch = id_only_batch();
        let err = apply_write_defaults(&schema, &batch).expect_err("mismatch");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("Cannot apply write-default"),
            "got {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn data_file_writer_writes_binary_write_default_into_parquet() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "payload", Type::Primitive(PrimitiveType::Binary))
                        .with_write_default(Literal::Primitive(PrimitiveLiteral::Binary(vec![
                            0xaa, 0xbb,
                        ])))
                        .into(),
                ])
                .build()
                .expect("schema"),
        );
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().expect("utf8 path").to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("bin".to_string(), None, DataFileFormat::Parquet);
        let parquet = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema);
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );
        let mut writer = DataFileWriterBuilder::new(rolling)
            .unpartitioned()
            .build(None)
            .await
            .expect("build writer");
        writer.write(id_only_batch()).await.expect("write");
        let data_files = writer.close().await.expect("close");
        let input = file_io
            .new_input(data_files[0].file_path.clone())
            .expect("input")
            .read()
            .await
            .expect("read");
        let reader = ParquetRecordBatchReaderBuilder::try_new(input)
            .expect("parquet reader")
            .build()
            .expect("build reader");
        let batches: Vec<_> = reader.map(|b| b.expect("batch")).collect();
        let payload = batches[0].column(1);
        let bytes = if let Some(array) = payload.as_any().downcast_ref::<arrow_array::BinaryArray>()
        {
            array.value(0).to_vec()
        } else if let Some(array) = payload
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
        {
            array.value(0).to_vec()
        } else {
            panic!("payload not binary, type {:?}", payload.data_type());
        };
        assert_eq!(bytes, vec![0xaa, 0xbb]);
    }

    #[tokio::test]
    async fn equality_delete_writer_with_projected_schema_writes_only_equality_ids() {
        use crate::arrow::arrow_schema_to_schema;
        use crate::writer::base_writer::equality_delete_writer::{
            EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
        };

        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = Arc::new(schema_id_and_name());
        let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone()).expect("eq config");
        let projected = Arc::new(
            arrow_schema_to_schema(config.projected_arrow_schema_ref())
                .expect("projected iceberg schema"),
        );
        let parquet = ParquetWriterBuilder::new(WriterProperties::builder().build(), projected);
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet,
            file_io.clone(),
            DefaultLocationGenerator::with_data_location(
                temp_dir.path().to_str().expect("utf8 path").to_string(),
            ),
            DefaultFileNameGenerator::new("eq".to_string(), None, DataFileFormat::Parquet),
        );
        let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
            .unpartitioned()
            .build(None)
            .await
            .expect("build eq writer");
        writer
            .write(id_only_batch())
            .await
            .expect("id-only equality delete");
        let files = writer.close().await.expect("close");
        assert_eq!(files.len(), 1);
        let input = file_io
            .new_input(files[0].file_path.clone())
            .expect("input")
            .read()
            .await
            .expect("read");
        let reader = ParquetRecordBatchReaderBuilder::try_new(input)
            .expect("parquet reader")
            .build()
            .expect("build reader");
        let batches: Vec<_> = reader.map(|b| b.expect("batch")).collect();
        assert_eq!(
            batches[0].num_columns(),
            1,
            "must not add write-default name"
        );
        assert_eq!(batches[0].schema().field(0).name(), "id");
    }

    #[tokio::test]
    async fn equality_delete_writer_does_not_fill_omitted_equality_key() {
        use crate::arrow::arrow_schema_to_schema;
        use crate::writer::base_writer::equality_delete_writer::{
            EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
        };

        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = Arc::new(schema_id_and_name());
        let config =
            EqualityDeleteWriterConfig::new(vec![1, 2], schema.clone()).expect("eq config");
        let projected = Arc::new(
            arrow_schema_to_schema(config.projected_arrow_schema_ref())
                .expect("projected iceberg schema"),
        );
        let parquet = ParquetWriterBuilder::new(WriterProperties::builder().build(), projected);
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet,
            file_io,
            DefaultLocationGenerator::with_data_location(
                temp_dir.path().to_str().expect("utf8 path").to_string(),
            ),
            DefaultFileNameGenerator::new("eq2".to_string(), None, DataFileFormat::Parquet),
        );
        let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
            .unpartitioned()
            .build(None)
            .await
            .expect("build eq writer");
        let err = writer
            .write(id_only_batch())
            .await
            .expect_err("omitted equality key must not be filled from write-default");
        assert_ne!(err.kind(), ErrorKind::FeatureUnsupported);
        let rendered = format!("{err:?}");
        assert!(
            !rendered.contains("anon"),
            "must not have filled name=anon into equality keys, got {rendered}"
        );
    }
}
