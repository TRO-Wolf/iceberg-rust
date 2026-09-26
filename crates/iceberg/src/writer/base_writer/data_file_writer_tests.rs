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

use arrow_array::cast::AsArray;
use arrow_array::{Array, Int32Array, StringArray};
use arrow_schema::{DataType, Field};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use crate::io::FileIO;
use crate::spec::{
    DataContentType, DataFileFormat, Literal, NestedField, PartitionKey, PartitionSpec,
    PrimitiveType, Schema, Struct, Transform, Type, UnboundPartitionField,
};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::write_defaults_tests::{
    assert_nested_field_ids, nested_batch, nested_ids_schema,
};
use crate::writer::{IcebergWriter, IcebergWriterBuilder, RecordBatch};
use crate::{ErrorKind, Result};

#[tokio::test]
async fn test_parquet_writer() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let file_io = FileIO::new_with_fs();
    let location_gen =
        DefaultLocationGenerator::with_data_location(temp_dir.path().to_str().unwrap().to_string());
    let file_name_gen =
        DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

    let schema = Schema::builder()
        .with_schema_id(3)
        .with_fields(vec![
            NestedField::required(3, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(4, "bar", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;

    let pw = ParquetWriterBuilder::new(WriterProperties::builder().build(), Arc::new(schema));

    let rolling_file_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
        pw,
        file_io.clone(),
        location_gen,
        file_name_gen,
    );

    let mut data_file_writer = DataFileWriterBuilder::new(rolling_file_writer_builder)
        .unpartitioned()
        .build(None)
        .await
        .unwrap();

    use crate::writer::CurrentFileStatus;
    assert_eq!(data_file_writer.current_file_path(), "");
    assert_eq!(data_file_writer.current_row_num(), 0);
    assert_eq!(data_file_writer.current_written_size(), 0);

    let arrow_schema = arrow_schema::Schema::new(vec![
        Field::new("foo", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            3.to_string(),
        )])),
        Field::new("bar", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            4.to_string(),
        )])),
    ]);
    let batch = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
    ])?;
    data_file_writer.write(batch).await?;

    let data_files = data_file_writer.close().await.unwrap();
    assert_eq!(data_files.len(), 1);

    let data_file = &data_files[0];
    assert_eq!(data_file.file_format, DataFileFormat::Parquet);
    assert_eq!(data_file.content, DataContentType::Data);
    assert_eq!(data_file.partition, Struct::empty());

    assert_eq!(
        data_file_writer.current_file_path(),
        "",
        "closed DataFileWriter reports empty path"
    );
    assert_eq!(
        data_file_writer.current_row_num(),
        0,
        "closed DataFileWriter reports zero rows"
    );
    assert_eq!(
        data_file_writer.current_written_size(),
        0,
        "closed DataFileWriter reports zero size"
    );

    let input_file = file_io.new_input(data_file.file_path.clone())?;
    let input_content = input_file.read().await?;

    let parquet_reader = ArrowReaderMetadata::load(&input_content, ArrowReaderOptions::default())
        .expect("Failed to load Parquet metadata");

    let field_ids: Vec<i32> = parquet_reader
        .parquet_schema()
        .columns()
        .iter()
        .map(|col| col.self_type().get_basic_info().id())
        .collect();

    assert_eq!(field_ids, vec![3, 4]);
    Ok(())
}

#[tokio::test]
async fn test_parquet_writer_with_partition() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let file_io = FileIO::new_with_fs();
    let location_gen =
        DefaultLocationGenerator::with_data_location(temp_dir.path().to_str().unwrap().to_string());
    let file_name_gen = DefaultFileNameGenerator::new(
        "test_partitioned".to_string(),
        None,
        DataFileFormat::Parquet,
    );

    let schema = Schema::builder()
        .with_schema_id(5)
        .with_fields(vec![
            NestedField::required(5, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(6, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let schema_ref = Arc::new(schema);

    let partition_value = Struct::from_iter([Some(Literal::int(1))]);
    let partition_key = PartitionKey::new(
        PartitionSpec::builder(schema_ref.clone()).build()?,
        schema_ref.clone(),
        partition_value.clone(),
    )
    .expect("PartitionKey::new: valid partition tuple");

    let parquet_writer_builder =
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema_ref.clone());

    let rolling_file_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_writer_builder,
        file_io.clone(),
        location_gen,
        file_name_gen,
    );

    let mut data_file_writer = DataFileWriterBuilder::new(rolling_file_writer_builder)
        .build(Some(partition_key))
        .await?;

    let arrow_schema = arrow_schema::Schema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            5.to_string(),
        )])),
        Field::new("name", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            6.to_string(),
        )])),
    ]);
    let batch = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
    ])?;
    data_file_writer.write(batch).await?;

    let data_files = data_file_writer.close().await.unwrap();
    assert_eq!(data_files.len(), 1);

    let data_file = &data_files[0];
    assert_eq!(data_file.file_format, DataFileFormat::Parquet);
    assert_eq!(data_file.content, DataContentType::Data);
    assert_eq!(data_file.partition, partition_value);

    let input_file = file_io.new_input(data_file.file_path.clone())?;
    let input_content = input_file.read().await?;

    let parquet_reader = ArrowReaderMetadata::load(&input_content, ArrowReaderOptions::default())?;

    let field_ids: Vec<i32> = parquet_reader
        .parquet_schema()
        .columns()
        .iter()
        .map(|col| col.self_type().get_basic_info().id())
        .collect();
    assert_eq!(field_ids, vec![5, 6]);

    let field_names: Vec<&str> = parquet_reader
        .parquet_schema()
        .columns()
        .iter()
        .map(|col| col.name())
        .collect();
    assert_eq!(field_names, vec!["id", "name"]);

    Ok(())
}

fn stamp_test_schema() -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("build stamp test schema"),
    )
}

fn dept_spec(schema: &Arc<Schema>, spec_id: i32, transform: Transform) -> PartitionSpec {
    PartitionSpec::builder(schema.as_ref().clone())
        .with_spec_id(spec_id)
        .add_unbound_field(
            UnboundPartitionField::builder()
                .source_id(2)
                .name("dept_part".to_string())
                .transform(transform)
                .build(),
        )
        .expect("add partition field")
        .build()
        .expect("build spec")
}

fn stamp_writer_builder(
    file_io: &FileIO,
    temp_dir: &TempDir,
    schema: &Arc<Schema>,
) -> DataFileWriterBuilder<ParquetWriterBuilder, DefaultLocationGenerator, DefaultFileNameGenerator>
{
    let location_gen = DefaultLocationGenerator::with_data_location(
        temp_dir
            .path()
            .to_str()
            .expect("temp dir path is utf-8")
            .to_string(),
    );
    let file_name_gen =
        DefaultFileNameGenerator::new("stamp".to_string(), None, DataFileFormat::Parquet);
    let parquet_writer_builder =
        ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone());
    DataFileWriterBuilder::new(RollingFileWriterBuilder::new_with_default_file_size(
        parquet_writer_builder,
        file_io.clone(),
        location_gen,
        file_name_gen,
    ))
}

fn stamp_test_batch() -> RecordBatch {
    let arrow_schema = arrow_schema::Schema::new(vec![
        Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            1.to_string(),
        )])),
        Field::new("dept", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            2.to_string(),
        )])),
    ]);
    RecordBatch::try_new(Arc::new(arrow_schema), vec![
        Arc::new(arrow_array::Int64Array::from(vec![1i64])),
        Arc::new(StringArray::from(vec!["eng"])),
    ])
    .expect("build stamp test batch")
}

#[tokio::test]
async fn test_data_file_writer_stamps_configured_unpartitioned_spec_id() -> Result<()> {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = stamp_test_schema();
    let spec = PartitionSpec::builder(schema.as_ref().clone())
        .with_spec_id(7)
        .build()
        .expect("unpartitioned spec 7");
    assert!(spec.is_unpartitioned(), "fixture: spec 7 is unpartitioned");

    let mut writer = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .with_partition_spec(spec)
        .build(None)
        .await?;
    writer.write(stamp_test_batch()).await?;
    let data_files = writer.close().await?;

    assert_eq!(data_files.len(), 1);
    assert_eq!(
        data_files[0].partition_spec_id(),
        7,
        "the file must claim the CONFIGURED spec, not the fabricated default 0"
    );
    assert_eq!(
        data_files[0].partition,
        Struct::empty(),
        "an unpartitioned spec still carries an empty tuple"
    );
    Ok(())
}

#[tokio::test]
async fn test_data_file_writer_rejects_partitioned_spec_without_partition_key() {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = stamp_test_schema();
    let spec = dept_spec(&schema, 3, Transform::Identity);

    let err = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .with_partition_spec(spec)
        .build(None)
        .await
        .expect_err("a partitioned spec with no PartitionKey must be rejected");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.to_string().contains("must carry its partition tuple"),
        "unexpected error: {err}"
    );
    assert!(
        err.to_string().contains("Partition spec 3"),
        "the error must name the offending spec: {err}"
    );
}

#[tokio::test]
async fn test_data_file_writer_rejects_all_void_spec_without_partition_key() {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = stamp_test_schema();
    let void_spec = dept_spec(&schema, 5, Transform::Void);

    assert!(
        void_spec.is_unpartitioned(),
        "fixture: an all-void spec reports is_unpartitioned() == true"
    );
    assert_eq!(
        void_spec.fields().len(),
        1,
        "fixture: the all-void spec still has one partition field"
    );
    assert_eq!(
        void_spec
            .partition_type(schema.as_ref())
            .expect("void partition type")
            .fields()
            .len(),
        1,
        "fixture: its partition TYPE has one field, so a tuple is required"
    );

    let err = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .with_partition_spec(void_spec)
        .build(None)
        .await
        .expect_err("an all-void spec with no PartitionKey must be rejected");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.to_string().contains("Partition spec 5"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_data_file_writer_accepts_all_void_spec_with_null_tuple_key() -> Result<()> {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = stamp_test_schema();
    let void_spec = dept_spec(&schema, 5, Transform::Void);
    let null_tuple = Struct::from_iter([None]);
    let partition_key = PartitionKey::new(void_spec, schema.clone(), null_tuple.clone())
        .expect("PartitionKey::new: valid partition tuple");

    let mut writer = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .build(Some(partition_key))
        .await?;
    writer.write(stamp_test_batch()).await?;
    let data_files = writer.close().await?;

    assert_eq!(data_files[0].partition_spec_id(), 5);
    assert_eq!(
        data_files[0].partition, null_tuple,
        "a NULL partition value stays legal"
    );
    Ok(())
}

#[tokio::test]
async fn test_data_file_writer_partition_key_spec_wins_over_configured_spec() -> Result<()> {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = stamp_test_schema();
    let configured = PartitionSpec::builder(schema.as_ref().clone())
        .with_spec_id(7)
        .build()
        .expect("unpartitioned spec 7");
    let key_partition = Struct::from_iter([Some(Literal::string("eng"))]);
    let partition_key = PartitionKey::new(
        dept_spec(&schema, 3, Transform::Identity),
        schema.clone(),
        key_partition.clone(),
    )
    .expect("PartitionKey::new: valid partition tuple");

    let mut writer = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .with_partition_spec(configured)
        .build(Some(partition_key))
        .await?;
    writer.write(stamp_test_batch()).await?;
    let data_files = writer.close().await?;

    assert_eq!(
        data_files[0].partition_spec_id(),
        3,
        "the PartitionKey's spec must win over the configured spec"
    );
    assert_eq!(data_files[0].partition, key_partition);
    Ok(())
}

#[tokio::test]
async fn test_data_file_writer_without_spec_or_key_errors() {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = stamp_test_schema();

    let err = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .build(None)
        .await
        .expect_err("build(None) with no spec must error");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.to_string().contains("unpartitioned()"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_data_file_writer_unpartitioned_stamps_spec_zero() -> Result<()> {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = stamp_test_schema();

    let mut writer = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .unpartitioned()
        .build(None)
        .await?;
    writer.write(stamp_test_batch()).await?;
    let data_files = writer.close().await?;

    assert_eq!(data_files[0].partition_spec_id(), 0);
    assert_eq!(data_files[0].partition, Struct::empty());
    Ok(())
}

fn unknown_column_schema() -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "u", Type::Primitive(PrimitiveType::Unknown)).into(),
            ])
            .build()
            .expect("schema with unknown column"),
    )
}

fn unknown_column_batch() -> RecordBatch {
    let arrow_schema = arrow_schema::Schema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            1.to_string(),
        )])),
        Field::new("u", DataType::Null, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            2.to_string(),
        )])),
    ]);
    RecordBatch::try_new(Arc::new(arrow_schema), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Arc::new(arrow_array::NullArray::new(3)),
    ])
    .expect("batch with Null unknown column")
}

#[tokio::test]
async fn data_file_writer_writes_unknown_null_column_without_parquet_column() -> Result<()> {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = unknown_column_schema();
    let mut writer = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .unpartitioned()
        .build(None)
        .await
        .expect("build data file writer");

    writer
        .write(unknown_column_batch())
        .await
        .expect("a Null unknown column writes without refusal");
    let data_files = writer.close().await.expect("close writer");
    assert_eq!(data_files.len(), 1);
    assert_eq!(data_files[0].record_count(), 3);
    assert!(
        !data_files[0].value_counts().contains_key(&2),
        "value_counts must carry no entry for the unknown field id"
    );
    assert!(
        !data_files[0].null_value_counts().contains_key(&2),
        "null_value_counts must carry no entry for the unknown field id"
    );
    assert!(
        !data_files[0].column_sizes().contains_key(&2),
        "column_sizes must carry no entry for the unknown field id"
    );

    assert_file_holds_only_id(&file_io, &data_files[0]).await
}

async fn assert_file_holds_only_id(
    file_io: &FileIO,
    data_file: &crate::spec::DataFile,
) -> Result<()> {
    let bytes = file_io
        .new_input(data_file.file_path.clone())?
        .read()
        .await?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .expect("open parquet")
        .build()
        .expect("build reader");
    let batches = reader
        .map(|batch| batch.expect("file batch"))
        .collect::<Vec<_>>();
    let first = batches.first().expect("one file batch");
    let first_schema = first.schema();
    let names: Vec<&str> = first_schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect();
    assert_eq!(
        names,
        vec!["id"],
        "the parquet file holds no column for the unknown field"
    );
    assert!(
        !data_file.column_sizes().contains_key(&2),
        "column_sizes must carry no entry for the unknown field id"
    );
    Ok(())
}

#[tokio::test]
async fn data_file_writer_fills_omitted_optional_unknown_column_with_null() -> Result<()> {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = unknown_column_schema();
    let mut writer = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .unpartitioned()
        .build(None)
        .await
        .expect("build data file writer");

    let arrow_schema = arrow_schema::Schema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            1.to_string(),
        )])),
    ]);
    let batch = RecordBatch::try_new(Arc::new(arrow_schema), vec![Arc::new(Int32Array::from(
        vec![1, 2, 3],
    ))])
    .expect("batch omitting unknown");

    writer
        .write(batch)
        .await
        .expect("an omitted optional unknown fills with null");
    let data_files = writer.close().await.expect("close writer");
    assert_eq!(data_files.len(), 1);
    assert_eq!(data_files[0].record_count(), 3);
    assert!(
        !data_files[0].value_counts().contains_key(&2),
        "value_counts must carry no entry for the omitted unknown field id"
    );
    assert_file_holds_only_id(&file_io, &data_files[0]).await
}

#[tokio::test]
async fn data_file_writer_writes_and_reads_back_int_string_batch() -> Result<()> {
    let temp_dir = TempDir::new().expect("temp dir");
    let file_io = FileIO::new_with_fs();
    let schema = Schema::builder()
        .with_schema_id(3)
        .with_fields(vec![
            NestedField::required(3, "foo", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(4, "bar", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;
    let schema = Arc::new(schema);

    let mut writer = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .unpartitioned()
        .build(None)
        .await
        .expect("build data file writer");

    let arrow_schema = arrow_schema::Schema::new(vec![
        Field::new("foo", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            3.to_string(),
        )])),
        Field::new("bar", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            4.to_string(),
        )])),
    ]);
    let batch = RecordBatch::try_new(Arc::new(arrow_schema), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
    ])?;
    writer.write(batch).await?;
    let data_files = writer.close().await.expect("close");
    assert_eq!(data_files.len(), 1);

    let bytes = file_io
        .new_input(data_files[0].file_path.clone())?
        .read()
        .await?;
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes)
        .expect("open parquet")
        .build()
        .expect("build reader");
    let read = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .expect("read batches");
    assert_eq!(read.len(), 1);
    assert_eq!(
        read[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int column")
            .values(),
        &[1, 2, 3]
    );
    let names = read[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("string column");
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert_eq!(names.value(2), "Charlie");
    Ok(())
}

#[tokio::test]
async fn data_file_writer_stamps_nested_field_ids_in_parquet_footer() {
    let file_io = FileIO::new_with_fs();
    let temp_dir = TempDir::new().expect("temp dir");
    let schema = Arc::new(nested_ids_schema());
    let mut writer = stamp_writer_builder(&file_io, &temp_dir, &schema)
        .unpartitioned()
        .build(None)
        .await
        .expect("build writer");
    writer
        .write(nested_batch("item", false))
        .await
        .expect("write");
    let bytes = file_io
        .new_input(writer.close().await.expect("close")[0].file_path.clone())
        .expect("input")
        .read()
        .await
        .expect("read");
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).expect("open parquet");
    assert_nested_field_ids(builder.schema());
    let batch = builder
        .build()
        .expect("build reader")
        .next()
        .expect("one batch")
        .expect("batch");
    let nums = batch.column(1).as_list::<i32>();
    assert!(nums.is_null(1) && nums.values().is_null(1) && nums.value_length(2) == 0);
    assert!(batch.column(3).as_map().is_null(2));
}
