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

//! Position-delete writer. Field ids in [`crate::metadata_columns`]. Write-as-given.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef as ArrowSchemaRef;
use parquet::file::properties::WriterProperties;

use crate::arrow::schema_to_arrow_schema;
use crate::metadata_columns::{delete_file_path_field, delete_file_pos_field};
use crate::spec::{DataContentType, DataFile, PartitionKey, PartitionSpec, Schema, SchemaRef};
use crate::writer::base_writer::data_file_writer::resolve_partition_spec_id;
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::{RollingFileWriter, RollingFileWriterBuilder};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Builds the Iceberg position-delete schema: required `file_path: string` then `pos: long`.
/// The reserved ids come from [`crate::metadata_columns`], so they match Java.
pub fn pos_delete_schema() -> Result<Schema> {
    Schema::builder()
        .with_fields(vec![
            delete_file_path_field().clone(),
            delete_file_pos_field().clone(),
        ])
        .build()
}

/// Parquet [`WriterProperties`] for position-delete files.
///
/// Turns off the default 64-byte statistics truncation, so the `file_path` bounds stay exact.
/// Path identity rides those bounds alone: v2 parquet deletes carry no `referenced_data_file`, and
/// [`crate::delete_file_index::referenced_data_file_location`] routes on equal lower and upper
/// bounds. A truncated S3 URI drops the bounds, and the routing silently misses.
pub fn position_delete_writer_properties() -> WriterProperties {
    WriterProperties::builder()
        .set_statistics_truncate_length(None)
        .build()
}

/// Config for [`PositionDeleteFileWriter`].
///
/// Holds the position-delete [`Schema`] and its Arrow projection. The Arrow schema is used to
/// validate every incoming [`RecordBatch`] so a mismatched batch is rejected up front rather than
/// silently producing a delete file Java cannot read.
#[derive(Debug, Clone)]
pub struct PositionDeleteWriterConfig {
    schema: SchemaRef,
    arrow_schema: ArrowSchemaRef,
}

impl PositionDeleteWriterConfig {
    /// Create a new `PositionDeleteWriterConfig`.
    pub fn new() -> Result<Self> {
        let schema = Arc::new(pos_delete_schema()?);
        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema)?);
        Ok(Self {
            schema,
            arrow_schema,
        })
    }

    /// Return the position-delete [`Schema`].
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Return the position-delete Arrow schema (the schema every input batch must match).
    pub fn arrow_schema(&self) -> &ArrowSchemaRef {
        &self.arrow_schema
    }
}

/// Builder for [`PositionDeleteFileWriter`].
#[derive(Debug, Clone)]
pub struct PositionDeleteFileWriterBuilder<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner: RollingFileWriterBuilder<B, L, F>,
    config: PositionDeleteWriterConfig,
    partition_spec: Option<PartitionSpec>,
}

impl<B, L, F> PositionDeleteFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Inner rolling writer must carry the position-delete schema.
    pub fn new(
        inner: RollingFileWriterBuilder<B, L, F>,
        config: PositionDeleteWriterConfig,
    ) -> Self {
        Self {
            inner,
            config,
            partition_spec: None,
        }
    }

    /// Stamp [`PartitionSpec::unpartition_spec`] (spec id 0, no fields).
    pub fn unpartitioned(self) -> Self {
        self.with_partition_spec(PartitionSpec::unpartition_spec())
    }

    /// Spec of the DATA FILES these deletes reference. Java `PositionDeleteWriter(spec)`.
    pub fn with_partition_spec(mut self, partition_spec: PartitionSpec) -> Self {
        self.partition_spec = Some(partition_spec);
        self
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriterBuilder for PositionDeleteFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    type R = PositionDeleteFileWriter<B, L, F>;

    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
        let partition_spec_id =
            resolve_partition_spec_id(self.partition_spec.as_ref(), partition_key.as_ref())?;
        Ok(PositionDeleteFileWriter {
            inner: Some(self.inner.build()),
            arrow_schema: self.config.arrow_schema.clone(),
            partition_key,
            partition_spec_id,
        })
    }
}

/// Writer that writes position-delete files within one spec/partition.
///
/// Each input [`RecordBatch`] must match the position-delete Arrow schema exactly
/// (`file_path: string`, `pos: long`, carrying field ids `2147483546` / `2147483545`). Records are
/// written in the order given — see the [module docs](self) for the sorting contract.
#[derive(Debug)]
pub struct PositionDeleteFileWriter<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner: Option<RollingFileWriter<B, L, F>>,
    arrow_schema: ArrowSchemaRef,
    partition_key: Option<PartitionKey>,
    /// The spec id stamped on every produced delete file, resolved once at build time by
    /// `resolve_partition_spec_id`.
    partition_spec_id: i32,
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriter for PositionDeleteFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        // Reject a mismatched batch here. A delete file with the wrong columns, ids, or types
        // silently deletes nothing, or Java cannot read it.
        if batch.schema().as_ref() != self.arrow_schema.as_ref() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Position delete batch schema does not match the position-delete schema. \
                     Expected {:?}, got {:?}",
                    self.arrow_schema,
                    batch.schema()
                ),
            ));
        }

        if let Some(writer) = self.inner.as_mut() {
            // The caller sorts by (file_path, pos). See the module docs.
            writer.write(&self.partition_key, &batch).await
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Position delete inner writer has been closed.",
            ))
        }
    }

    async fn close(&mut self) -> Result<Vec<DataFile>> {
        if let Some(writer) = self.inner.take() {
            writer
                .close()
                .await?
                .into_iter()
                .map(|mut res| {
                    res.content(DataContentType::PositionDeletes);
                    // Stamp the spec id always, not only when a partition key is present. A delete
                    // file with the wrong spec commits and then silently never applies.
                    res.partition_spec_id(self.partition_spec_id);
                    if let Some(pk) = self.partition_key.as_ref() {
                        res.partition(pk.data().clone());
                    }
                    res.build().map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Failed to build position delete file: {e}"),
                        )
                    })
                })
                .collect()
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Position delete inner writer has been closed.",
            ))
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::DataType;
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::TempDir;

    use super::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig, pos_delete_schema,
        position_delete_writer_properties,
    };
    use crate::ErrorKind;
    use crate::io::FileIO;
    use crate::metadata_columns::{
        RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
    };
    use crate::spec::{
        DataContentType, DataFileFormat, MetricsConfig, PrimitiveLiteral, PrimitiveType, Type,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};

    /// Build a `(file_path, pos)` RecordBatch in the canonical position-delete Arrow schema.
    fn pos_delete_batch(pairs: &[(&str, i64)]) -> RecordBatch {
        let config = PositionDeleteWriterConfig::new().unwrap();
        let file_paths: Vec<&str> = pairs.iter().map(|(p, _)| *p).collect();
        let positions: Vec<i64> = pairs.iter().map(|(_, p)| *p).collect();
        let file_path_col = Arc::new(StringArray::from(file_paths)) as ArrayRef;
        let pos_col = Arc::new(Int64Array::from(positions)) as ArrayRef;
        RecordBatch::try_new(config.arrow_schema().clone(), vec![file_path_col, pos_col]).unwrap()
    }

    fn make_writer_builder(
        file_io: &FileIO,
        temp_dir: &TempDir,
    ) -> PositionDeleteFileWriterBuilder<
        ParquetWriterBuilder,
        DefaultLocationGenerator,
        DefaultFileNameGenerator,
    > {
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen = DefaultFileNameGenerator::new(
            "test-pos-del".to_string(),
            None,
            DataFileFormat::Parquet,
        );

        let config = PositionDeleteWriterConfig::new().unwrap();
        let parquet_writer_builder =
            ParquetWriterBuilder::new(position_delete_writer_properties(), config.schema().clone())
                .with_metrics_config(MetricsConfig::for_position_delete());
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_writer_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );
        PositionDeleteFileWriterBuilder::new(rolling_writer_builder, config).unpartitioned()
    }

    /// A realistic S3 path over 64 bytes must survive as equal lower and upper bounds, so
    /// equal-bounds routing recovers `referenced_data_file_location` without the DV field.
    #[tokio::test]
    async fn test_position_delete_long_file_path_bounds_are_full_and_equal()
    -> Result<(), anyhow::Error> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let mut writer = make_writer_builder(&file_io, &temp_dir)
            .unpartitioned()
            .build(None)
            .await?;

        // 120-char path (well past parquet's default 64-byte stats truncate).
        let long_path = format!(
            "s3://bucket-name/warehouse/ns/table/data/{}",
            "a".repeat(80)
        );
        assert!(
            long_path.len() > 64,
            "fixture must exceed default statistics_truncate_length"
        );
        let batch = pos_delete_batch(&[(&long_path, 0), (&long_path, 1)]);
        writer.write(batch).await?;
        let data_files = writer.close().await?;
        assert_eq!(data_files.len(), 1);
        let df = &data_files[0];

        let lower = df
            .lower_bounds()
            .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
            .expect("file_path lower bound must be present (Full + exact stats)");
        let upper = df
            .upper_bounds()
            .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
            .expect("file_path upper bound must be present (Full + exact stats)");
        assert_eq!(lower, upper, "single-path delete must have equal bounds");
        match lower.literal() {
            PrimitiveLiteral::String(s) => {
                assert_eq!(
                    s.as_str(),
                    long_path.as_str(),
                    "bound must be the FULL path, not a 64-byte truncated prefix"
                );
            }
            other => panic!("expected string bound, got {other:?}"),
        }
        Ok(())
    }

    /// The schema this writer advertises must be exactly the Iceberg position-delete schema, with
    /// the reserved field ids that Java uses. A wrong id here means Java cannot read the file.
    #[test]
    fn test_pos_delete_schema_has_reserved_field_ids() {
        let schema = pos_delete_schema().unwrap();
        let fields = schema.as_struct().fields();
        assert_eq!(fields.len(), 2);

        assert_eq!(fields[0].name, "file_path");
        assert_eq!(fields[0].id, RESERVED_FIELD_ID_DELETE_FILE_PATH);
        assert_eq!(fields[0].id, 2147483546);
        assert!(fields[0].required);
        assert_eq!(
            fields[0].field_type.as_ref(),
            &Type::Primitive(PrimitiveType::String)
        );

        assert_eq!(fields[1].name, "pos");
        assert_eq!(fields[1].id, RESERVED_FIELD_ID_DELETE_FILE_POS);
        assert_eq!(fields[1].id, 2147483545);
        assert!(fields[1].required);
        assert_eq!(
            fields[1].field_type.as_ref(),
            &Type::Primitive(PrimitiveType::Long)
        );
    }

    /// Round-trip risk: dropped or mangled positions = data silently NOT deleted. Write a set of
    /// (file_path, pos) pairs, read the parquet back, and assert the exact pairs survive in order.
    #[tokio::test]
    async fn test_position_delete_round_trips_exact_positions() -> Result<(), anyhow::Error> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let mut writer = make_writer_builder(&file_io, &temp_dir)
            .unpartitioned()
            .build(None)
            .await?;

        let pairs = [
            ("s3://bucket/data/1.parquet", 0i64),
            ("s3://bucket/data/1.parquet", 5),
            ("s3://bucket/data/1.parquet", 1023),
        ];
        writer.write(pos_delete_batch(&pairs)).await?;
        let data_files = writer.close().await?;

        assert_eq!(data_files.len(), 1);
        let data_file = &data_files[0];
        assert_eq!(data_file.content, DataContentType::PositionDeletes);
        assert_eq!(data_file.record_count, pairs.len() as u64);

        // Read the written parquet back and assert the exact (file_path, pos) pairs round-trip.
        let input = file_io.new_input(data_file.file_path.clone())?;
        let bytes = input.read().await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;
        let mut read_pairs: Vec<(String, i64)> = Vec::new();
        for batch in reader {
            let batch = batch?;
            let paths = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let positions = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                read_pairs.push((paths.value(i).to_string(), positions.value(i)));
            }
        }

        let expected: Vec<(String, i64)> = pairs.iter().map(|(p, n)| (p.to_string(), *n)).collect();
        assert_eq!(read_pairs, expected);
        Ok(())
    }

    /// Interop risk: the written parquet must carry field ids 2147483546 (file_path) / 2147483545
    /// (pos). Wrong ids => Java cannot read the delete file.
    #[tokio::test]
    async fn test_position_delete_written_field_ids_match_reserved() -> Result<(), anyhow::Error> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let mut writer = make_writer_builder(&file_io, &temp_dir)
            .unpartitioned()
            .build(None)
            .await?;

        writer
            .write(pos_delete_batch(&[("s3://b/d/1.parquet", 7)]))
            .await?;
        let data_files = writer.close().await?;
        let data_file = &data_files[0];

        let input = file_io.new_input(data_file.file_path.clone())?;
        let bytes = input.read().await?;
        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let arrow_schema = reader_builder.schema();

        let path_field = arrow_schema.field(0);
        assert_eq!(path_field.name(), "file_path");
        assert_eq!(path_field.data_type(), &DataType::Utf8);
        assert_eq!(
            path_field.metadata().get(PARQUET_FIELD_ID_META_KEY),
            Some(&"2147483546".to_string())
        );

        let pos_field = arrow_schema.field(1);
        assert_eq!(pos_field.name(), "pos");
        assert_eq!(pos_field.data_type(), &DataType::Int64);
        assert_eq!(
            pos_field.metadata().get(PARQUET_FIELD_ID_META_KEY),
            Some(&"2147483545".to_string())
        );

        Ok(())
    }

    /// One delete file may carry positions for MULTIPLE data files. Assert every pair round-trips,
    /// preserving the order given (the writer does not reorder — sorting is the caller's job).
    #[tokio::test]
    async fn test_position_delete_multiple_data_files_one_delete_file() -> Result<(), anyhow::Error>
    {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let mut writer = make_writer_builder(&file_io, &temp_dir)
            .unpartitioned()
            .build(None)
            .await?;

        // Intentionally interleaved / unsorted across two data files: written AS GIVEN.
        let pairs = [
            ("s3://b/d/2.parquet", 3i64),
            ("s3://b/d/1.parquet", 0),
            ("s3://b/d/2.parquet", 1),
            ("s3://b/d/1.parquet", 9),
        ];
        writer.write(pos_delete_batch(&pairs)).await?;
        let data_files = writer.close().await?;

        assert_eq!(data_files.len(), 1);
        assert_eq!(data_files[0].record_count, pairs.len() as u64);

        let input = file_io.new_input(data_files[0].file_path.clone())?;
        let bytes = input.read().await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;
        let mut read_pairs: Vec<(String, i64)> = Vec::new();
        for batch in reader {
            let batch = batch?;
            let paths = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let positions = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                read_pairs.push((paths.value(i).to_string(), positions.value(i)));
            }
        }

        let expected: Vec<(String, i64)> = pairs.iter().map(|(p, n)| (p.to_string(), *n)).collect();
        assert_eq!(read_pairs, expected);
        Ok(())
    }

    /// A batch whose schema is NOT the position-delete schema must be rejected — writing it would
    /// produce a delete file that fails to delete (or that Java cannot read).
    #[tokio::test]
    async fn test_position_delete_rejects_mismatched_schema() -> Result<(), anyhow::Error> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let mut writer = make_writer_builder(&file_io, &temp_dir)
            .unpartitioned()
            .build(None)
            .await?;

        // Wrong: a plain (path, pos) batch with no field-id metadata and a different field order.
        let bad_schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("pos", DataType::Int64, false),
            arrow_schema::Field::new("file_path", DataType::Utf8, false),
        ]));
        let bad_batch = RecordBatch::try_new(bad_schema, vec![
            Arc::new(Int64Array::from(vec![0i64])) as ArrayRef,
            Arc::new(StringArray::from(vec!["s3://b/d/1.parquet"])) as ArrayRef,
        ])
        .unwrap();

        let err = writer.write(bad_batch).await.expect_err("must reject");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string().contains("position-delete schema"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    /// Empty input convention (mirrors the equality-delete writer): a writer that received no rows
    /// still closes cleanly. Closing with zero deletes yields a 0-row delete file.
    #[tokio::test]
    async fn test_position_delete_empty_input_closes() -> Result<(), anyhow::Error> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let mut writer = make_writer_builder(&file_io, &temp_dir)
            .unpartitioned()
            .build(None)
            .await?;

        // No write() calls — close immediately.
        let data_files = writer.close().await?;
        // The rolling writer produces no file when nothing was written.
        assert!(
            data_files.is_empty(),
            "expected no delete files for empty input, got {}",
            data_files.len()
        );
        Ok(())
    }

    /// CONFIGURED SPEC, NO KEY. An unpartitioned spec whose id is NOT 0 must be stamped as itself —
    /// the position-delete leg of [`super::resolve_partition_spec_id`].
    #[tokio::test]
    async fn test_position_delete_stamps_configured_unpartitioned_spec_id()
    -> Result<(), anyhow::Error> {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = crate::spec::Schema::builder()
            .with_fields(vec![
                crate::spec::NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long))
                    .into(),
            ])
            .build()?;
        let spec = crate::spec::PartitionSpec::builder(schema)
            .with_spec_id(7)
            .build()?;

        let mut writer = make_writer_builder(&file_io, &temp_dir)
            .with_partition_spec(spec)
            .build(None)
            .await?;
        writer
            .write(pos_delete_batch(&[("s3://b/d/1.parquet", 0)]))
            .await?;
        let delete_files = writer.close().await?;

        assert_eq!(
            delete_files[0].partition_spec_id(),
            7,
            "the delete file must claim the CONFIGURED spec, not the fabricated default 0"
        );
        Ok(())
    }

    /// CONFIGURED PARTITIONED SPEC, NO KEY. Rejected at build time — the delete would claim a
    /// partitioned spec while carrying an empty tuple.
    #[tokio::test]
    async fn test_position_delete_rejects_partitioned_spec_without_partition_key()
    -> Result<(), anyhow::Error> {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = crate::spec::Schema::builder()
            .with_fields(vec![
                crate::spec::NestedField::required(
                    1,
                    "dept",
                    Type::Primitive(PrimitiveType::String),
                )
                .into(),
            ])
            .build()?;
        let spec = crate::spec::PartitionSpec::builder(schema)
            .with_spec_id(3)
            .add_unbound_field(
                crate::spec::UnboundPartitionField::builder()
                    .source_id(1)
                    .name("dept_part".to_string())
                    .transform(crate::spec::Transform::Identity)
                    .build(),
            )?
            .build()?;

        let err = make_writer_builder(&file_io, &temp_dir)
            .with_partition_spec(spec)
            .build(None)
            .await
            .expect_err("a partitioned spec with no PartitionKey must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string().contains("must carry its partition tuple"),
            "unexpected error: {err}"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "position_delete_writer_spec_stamp.rs"]
mod spec_stamp_e2e_test;
