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

//! This module provides the [`PositionDeleteFileWriter`].
//!
//! A position-delete file marks rows as deleted by `(file_path, pos)` — the full URI of the target
//! data file and the ordinal position (starting at 0) of the deleted row within it. This is the
//! merge-on-read counterpart to the [`EqualityDeleteFileWriter`](super::equality_delete_writer); the
//! read side that consumes these files lives in [`crate::arrow::delete_filter`].
//!
//! # Schema
//!
//! The file's schema is exactly the Iceberg position-delete schema (spec
//! [§position-delete-files](https://iceberg.apache.org/spec/#position-delete-files)):
//!
//! | field id     | name        | type     |
//! |--------------|-------------|----------|
//! | `2147483546` | `file_path` | `string` |
//! | `2147483545` | `pos`       | `long`   |
//!
//! These reserved field ids must match Java (`MetadataColumns.DELETE_FILE_PATH` /
//! `DELETE_FILE_POS`) for interop — a delete file with the wrong ids cannot be read by Java. They are
//! defined once in [`crate::metadata_columns`] and reused here.
//!
//! The optional `row` column (field id `2147483544`, "position deletes with row data") is **out of
//! scope** for this writer.
//!
//! # Sorting
//!
//! The Iceberg spec recommends that rows in a position-delete file be sorted by `file_path` then
//! `pos` so that readers can binary-search. Mirroring Java's basic
//! [`PositionDeleteWriter`](https://github.com/apache/iceberg/blob/main/core/src/main/java/org/apache/iceberg/deletes/PositionDeleteWriter.java),
//! **this writer writes records in the order given and never reorders them** — producing the sorted
//! ordering is the caller's responsibility (Java delegates it to `SortingPositionOnlyDeleteWriter`).
//! Feeding unsorted positions yields a valid, readable delete file that is merely sub-optimal for
//! scan-time filtering.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef as ArrowSchemaRef;

use crate::arrow::schema_to_arrow_schema;
use crate::metadata_columns::{delete_file_path_field, delete_file_pos_field};
use crate::spec::{DataContentType, DataFile, PartitionKey, PartitionSpec, Schema, SchemaRef};
use crate::writer::base_writer::data_file_writer::resolve_partition_spec_id;
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::{RollingFileWriter, RollingFileWriterBuilder};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Build the canonical Iceberg position-delete schema: `file_path: string` (field id `2147483546`)
/// followed by `pos: long` (field id `2147483545`), both required.
///
/// The fields (and their reserved ids) come from [`crate::metadata_columns`], so they match the Java
/// `MetadataColumns.DELETE_FILE_PATH` / `DELETE_FILE_POS` definitions for interop.
pub fn pos_delete_schema() -> Result<Schema> {
    Schema::builder()
        .with_fields(vec![
            delete_file_path_field().clone(),
            delete_file_pos_field().clone(),
        ])
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
    /// Create a new `PositionDeleteFileWriterBuilder` using a `RollingFileWriterBuilder`.
    ///
    /// The inner [`RollingFileWriterBuilder`] must be configured with a parquet (or other) file
    /// writer whose schema is the position-delete schema (see [`pos_delete_schema`] /
    /// [`PositionDeleteWriterConfig::schema`]).
    ///
    /// Prefer chaining [`with_partition_spec`](Self::with_partition_spec): without it, a writer built
    /// with no [`PartitionKey`] falls back to stamping `DEFAULT_PARTITION_SPEC_ID` (0) — and a
    /// POSITION delete is paired to data on `(spec_id, partition)`
    /// (`DeleteFileIndex::get_deletes_for_data_file`), so one that claims spec 0 is never applied to
    /// data files under any other spec: the rows it was written to delete come back. See
    /// `resolve_partition_spec_id` and `docs/ENGINE_CONTRACT.md` §7a.
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

    /// Set the [`PartitionSpec`] the produced delete files are written under.
    ///
    /// This is the Rust counterpart of Java's REQUIRED `PositionDeleteWriter(…, PartitionSpec spec,
    /// …)` argument (`core/.../deletes/PositionDeleteWriter.java`, which feeds
    /// `FileMetadata.deleteFileBuilder(spec)`). The spec MUST be the spec of the DATA FILES the
    /// deletes reference, not the table's current spec — a delete file only ever applies to data
    /// files carrying the same `(spec_id, partition)`. It is used only when the writer is built
    /// WITHOUT a [`PartitionKey`]; a key always wins. See `resolve_partition_spec_id`.
    ///
    /// **This writer OWNS `partition_spec_id` on every [`DataFile`] it emits** — `close()` sets the
    /// field unconditionally, overriding anything a custom
    /// [`FileWriter`](crate::writer::file_writer::FileWriter) put on the returned `DataFileBuilder`.
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
        // Validate the incoming batch against the position-delete schema. A delete file whose
        // columns/ids/types do not match the reserved (file_path, pos) schema would silently fail to
        // delete rows (or be unreadable by Java), so reject it here rather than write it.
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
            // Write records in the order given — sorting by (file_path, pos) is the caller's
            // responsibility (see the module docs).
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
                    // ALWAYS stamp the spec id (Java `FileMetadata.Builder(spec)` does), never only
                    // when a partition key happens to be present — a delete file stamped with the
                    // wrong spec commits and then silently never applies. See
                    // `resolve_partition_spec_id`.
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
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::{PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig, pos_delete_schema};
    use crate::ErrorKind;
    use crate::io::FileIO;
    use crate::metadata_columns::{
        RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
    };
    use crate::spec::{DataContentType, DataFileFormat, PrimitiveType, Type};
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
            ParquetWriterBuilder::new(WriterProperties::builder().build(), config.schema().clone());
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_writer_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );
        PositionDeleteFileWriterBuilder::new(rolling_writer_builder, config)
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
        let mut writer = make_writer_builder(&file_io, &temp_dir).build(None).await?;

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
        let mut writer = make_writer_builder(&file_io, &temp_dir).build(None).await?;

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
        let mut writer = make_writer_builder(&file_io, &temp_dir).build(None).await?;

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
        let mut writer = make_writer_builder(&file_io, &temp_dir).build(None).await?;

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
        let mut writer = make_writer_builder(&file_io, &temp_dir).build(None).await?;

        // No write() calls — close immediately.
        let data_files = writer.close().await?;
        // The rolling writer produces no file when nothing was written (matches the data /
        // equality-delete writers' behavior for an empty writer).
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

/// End-to-end partition-spec-id stamping
/// ======================================
///
/// Writer → commit → scan, over a table whose partition spec EVOLVED. These are the observable
/// consequences of the fabricated `DEFAULT_PARTITION_SPEC_ID` stamp; they are stated normatively in
/// `docs/ENGINE_CONTRACT.md` §7a (WG4c). The equality-delete legs live here too (rather than in
/// `equality_delete_writer.rs`) because the catalog/commit/scan machinery below is shared: they pin
/// the read-side ASYMMETRY §7a has to state — a keyless equality delete carries an EMPTY tuple and
/// is therefore GLOBAL, not inert.
#[cfg(test)]
mod spec_stamp_e2e_test {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use futures::TryStreamExt;
    use parquet::file::properties::WriterProperties;

    use super::{PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig};
    use crate::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataFile, DataFileFormat, FormatVersion, Literal, NestedField, PartitionKey, PartitionSpec,
        PrimitiveType, Schema, SchemaRef, Struct, Transform, Type, UnboundPartitionField,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::base_writer::equality_delete_writer::{
        EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    use crate::{Catalog, ErrorKind, TableCreation, TableIdent};

    // -------------------------------------------------------------------------------------------
    // Fixtures.
    // -------------------------------------------------------------------------------------------

    /// `1: id long`, `2: dept string`, both required.
    fn test_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "dept", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("build test schema")
    }

    /// An `identity(dept)` spec under `spec_id`.
    fn identity_dept_spec(spec_id: i32) -> PartitionSpec {
        PartitionSpec::builder(test_schema())
            .with_spec_id(spec_id)
            .add_unbound_field(
                UnboundPartitionField::builder()
                    .source_id(2)
                    .name("dept".to_string())
                    .transform(Transform::Identity)
                    .build(),
            )
            .expect("add identity(dept)")
            .build()
            .expect("build identity(dept) spec")
    }

    /// A `truncate[5](dept)` spec under `spec_id`, partition field named `dept_trunc`.
    ///
    /// Its partition TYPE is the same shape as [`identity_dept_spec`]'s — one required-source string
    /// field — and for any `dept` value of five characters or fewer the two transforms produce the
    /// SAME tuple. That is what lets a fixture vary the spec id while holding the tuple constant.
    fn truncate5_dept_spec(spec_id: i32) -> PartitionSpec {
        PartitionSpec::builder(test_schema())
            .with_spec_id(spec_id)
            .add_unbound_field(
                UnboundPartitionField::builder()
                    .source_id(2)
                    .name("dept_trunc".to_string())
                    .transform(Transform::Truncate(5))
                    .build(),
            )
            .expect("add truncate[5](dept)")
            .build()
            .expect("build truncate[5](dept) spec")
    }

    /// A fresh V2 table in `catalog` under `spec`.
    async fn make_table(catalog: &impl Catalog, spec: PartitionSpec) -> Table {
        let table_ident =
            TableIdent::from_strs([format!("ns-{}", uuid::Uuid::new_v4()), "t".to_string()])
                .expect("table ident");
        catalog
            .create_namespace(table_ident.namespace(), HashMap::new())
            .await
            .expect("create namespace");
        let creation = TableCreation::builder()
            .schema(test_schema())
            .partition_spec(spec)
            .name(table_ident.name().to_string())
            .format_version(FormatVersion::V2)
            .build();
        catalog
            .create_table(table_ident.namespace(), creation)
            .await
            .expect("create table")
    }

    fn rows_batch(schema: &SchemaRef, rows: &[(i64, &str)]) -> RecordBatch {
        let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
        let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
        let depts: Vec<&str> = rows.iter().map(|(_, dept)| *dept).collect();
        RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(depts)) as ArrayRef,
        ])
        .expect("rows batch")
    }

    fn rolling_builder(
        table: &Table,
        prefix: &str,
        schema: SchemaRef,
    ) -> RollingFileWriterBuilder<
        ParquetWriterBuilder,
        DefaultLocationGenerator,
        DefaultFileNameGenerator,
    > {
        let location_gen =
            DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
        let file_name_gen =
            DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet);
        RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema),
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        )
    }

    /// Write one data file, optionally under a configured spec and/or a partition key.
    async fn write_data_file(
        table: &Table,
        configured_spec: Option<PartitionSpec>,
        partition_key: Option<PartitionKey>,
        rows: &[(i64, &str)],
    ) -> DataFile {
        let schema = table.metadata().current_schema();
        let mut builder =
            DataFileWriterBuilder::new(rolling_builder(table, "data", schema.clone()));
        if let Some(spec) = configured_spec {
            builder = builder.with_partition_spec(spec);
        }
        let mut writer = builder
            .build(partition_key)
            .await
            .expect("build data writer");
        writer
            .write(rows_batch(schema, rows))
            .await
            .expect("write rows");
        writer
            .close()
            .await
            .expect("close data writer")
            .into_iter()
            .next()
            .expect("one data file")
    }

    /// Write one position-delete file, optionally under a configured spec and/or a partition key.
    async fn write_pos_delete(
        table: &Table,
        configured_spec: Option<PartitionSpec>,
        partition_key: Option<PartitionKey>,
        pairs: &[(&str, i64)],
    ) -> DataFile {
        let config = PositionDeleteWriterConfig::new().expect("pos-delete config");
        let mut builder = PositionDeleteFileWriterBuilder::new(
            rolling_builder(table, "pos-del", config.schema().clone()),
            config.clone(),
        );
        if let Some(spec) = configured_spec {
            builder = builder.with_partition_spec(spec);
        }
        let mut writer = builder
            .build(partition_key)
            .await
            .expect("build pos-delete writer");
        let paths: Vec<&str> = pairs.iter().map(|(path, _)| *path).collect();
        let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
        let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions)) as ArrayRef,
        ])
        .expect("pos-delete batch");
        writer.write(batch).await.expect("write pos deletes");
        writer
            .close()
            .await
            .expect("close pos-delete writer")
            .into_iter()
            .next()
            .expect("one pos-delete file")
    }

    /// Write one equality-delete file on `id`, optionally under a configured spec and/or a key.
    ///
    /// `rows` are full table rows; the writer projects them down to the equality columns.
    async fn write_eq_delete_on_id(
        table: &Table,
        configured_spec: Option<PartitionSpec>,
        partition_key: Option<PartitionKey>,
        rows: &[(i64, &str)],
    ) -> DataFile {
        let schema = table.metadata().current_schema();
        let config = EqualityDeleteWriterConfig::new(vec![1], schema.clone())
            .expect("equality-delete config on id");
        let projected = Arc::new(
            arrow_schema_to_schema(config.projected_arrow_schema_ref())
                .expect("projected iceberg schema"),
        );
        let mut builder = EqualityDeleteFileWriterBuilder::new(
            rolling_builder(table, "eq-del", projected),
            config,
        );
        if let Some(spec) = configured_spec {
            builder = builder.with_partition_spec(spec);
        }
        let mut writer = builder
            .build(partition_key)
            .await
            .expect("build eq-delete writer");
        writer
            .write(rows_batch(schema, rows))
            .await
            .expect("write eq deletes");
        writer
            .close()
            .await
            .expect("close eq-delete writer")
            .into_iter()
            .next()
            .expect("one eq-delete file")
    }

    async fn fast_append(
        catalog: &impl Catalog,
        table: &Table,
        files: Vec<DataFile>,
    ) -> crate::Result<Table> {
        let tx = Transaction::new(table);
        let tx = tx
            .fast_append()
            .add_data_files(files)
            .apply(tx)
            .expect("apply fast_append");
        tx.commit(catalog).await
    }

    async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let tx = tx
            .row_delta()
            .add_deletes(deletes)
            .apply(tx)
            .expect("apply row_delta");
        tx.commit(catalog).await.expect("commit row_delta")
    }

    /// The merge-on-read live `id` set, ascending.
    async fn scan_ids(table: &Table) -> Vec<i64> {
        let stream = table
            .scan()
            .select(["id"])
            .build()
            .expect("scan build")
            .to_arrow()
            .await
            .expect("to_arrow");
        let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
        let mut ids = Vec::new();
        for batch in batches {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id column is int64");
            for i in 0..col.len() {
                ids.push(col.value(i));
            }
        }
        ids.sort_unstable();
        ids
    }

    /// Evolve the spec by ADDING `identity(field)`; returns the table and its new default spec id.
    async fn evolve_add_field(catalog: &impl Catalog, table: &Table, field: &str) -> (Table, i32) {
        let tx = Transaction::new(table);
        let tx = tx
            .update_partition_spec()
            .add_field(field)
            .apply(tx)
            .expect("apply update_partition_spec");
        let table = tx.commit(catalog).await.expect("commit spec evolution");
        let spec_id = table.metadata().default_partition_spec_id();
        (table, spec_id)
    }

    /// Evolve the spec by REMOVING `field`; returns the table and its new default spec id.
    async fn evolve_remove_field(
        catalog: &impl Catalog,
        table: &Table,
        field: &str,
    ) -> (Table, i32) {
        let tx = Transaction::new(table);
        let tx = tx
            .update_partition_spec()
            .remove_field(field)
            .apply(tx)
            .expect("apply update_partition_spec");
        let table = tx.commit(catalog).await.expect("commit spec evolution");
        let spec_id = table.metadata().default_partition_spec_id();
        (table, spec_id)
    }

    // -------------------------------------------------------------------------------------------
    // The two consequences.
    // -------------------------------------------------------------------------------------------

    /// SILENT UNDER-DELETE, ENGINE-REACHABLE SHAPE (the probe that sized this unit; 2026-07-25).
    ///
    /// Table: spec 0 UNPARTITIONED, evolved to a partitioned spec 1. Data lands under spec 1. A
    /// position delete built with neither a `PartitionKey` nor a configured spec claims spec 0 — and
    /// spec 0's partition type is EMPTY, exactly the tuple the file carries, so
    /// `SnapshotProducer::validate_partition_value` ACCEPTS it. The read side then never pairs it
    /// with the data, so every "deleted" row survives. Nothing anywhere fails.
    ///
    /// ATTRIBUTION (be precise — 2026-07-25 Critic): the unstamped delete here differs from the data
    /// on BOTH halves of the read-side `(spec_id, partition)` key — its tuple is `Struct::empty()`
    /// while the data's is `{"eng"}` — so the miss happens at the partition-bucket lookup
    /// (`pos_deletes_by_partition.get(data_file.partition())`) and never reaches the
    /// `partition_spec_id` comparison. This test pins the SHAPE an engine actually produces (a
    /// writer given no key at all); the spec id is isolated as the sole discriminator by its twin,
    /// [`test_e2e_same_tuple_wrong_spec_id_alone_silently_under_deletes`].
    ///
    /// The second half is the fix: the same call with the spec configured is REJECTED at build time
    /// instead of producing the silent artifact.
    #[tokio::test]
    async fn test_e2e_unstamped_delete_under_evolved_spec_commits_and_never_applies() {
        let catalog = new_memory_catalog().await;
        let table = make_table(
            &catalog,
            PartitionSpec::builder(test_schema())
                .with_spec_id(0)
                .build()
                .expect("unpartitioned spec 0"),
        )
        .await;
        assert!(
            table.metadata().default_partition_spec().is_unpartitioned(),
            "fixture: spec 0 is unpartitioned"
        );

        let (table, cur_spec_id) = evolve_add_field(&catalog, &table, "dept").await;
        assert_ne!(cur_spec_id, 0, "fixture: the spec evolved away from 0");

        // Data under the CURRENT (partitioned) spec, via its PartitionKey — the correct path.
        let partition = Struct::from_iter([Some(Literal::string("eng"))]);
        let partition_key = PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            partition,
        )
        .expect("PartitionKey::new: valid partition tuple");
        let data = write_data_file(&table, None, Some(partition_key), &[
            (1, "eng"),
            (2, "eng"),
            (3, "eng"),
        ])
        .await;
        assert_eq!(data.partition_spec_id(), cur_spec_id);
        let data_path = data.file_path().to_string();
        let table = fast_append(&catalog, &table, vec![data])
            .await
            .expect("commit data");
        assert_eq!(scan_ids(&table).await, vec![1, 2, 3]);

        // A delete built with NEITHER a key NOR a configured spec: the legacy fallback stamps 0.
        let delete = write_pos_delete(&table, None, None, &[
            (data_path.as_str(), 0),
            (data_path.as_str(), 1),
            (data_path.as_str(), 2),
        ])
        .await;
        assert_eq!(
            delete.partition_spec_id(),
            0,
            "the unconfigured delete claims spec 0"
        );

        // The commit ACCEPTS it — spec 0's partition type is empty, matching the empty tuple.
        let table = add_deletes(&catalog, &table, vec![delete]).await;

        // ... and every row survives it. This is the silent under-delete.
        assert_eq!(
            scan_ids(&table).await,
            vec![1, 2, 3],
            "the wrong-spec delete silently never applies — rows resurrect"
        );

        // POSITIVE CONTROL, same table / same data file / same positions: a delete carrying the
        // data file's OWN PartitionKey removes exactly those rows. Without this leg the survival
        // above would be attributable to "deletes never work in this fixture"; with it, the
        // difference is exactly "the delete was given its data files' key" vs "it was given
        // nothing". (Which HALF of the `(spec_id, partition)` key does the excluding is isolated by
        // the twin test, not here — see this test's doc comment.)
        let correct_key = PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::string("eng"))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let correct_delete = write_pos_delete(&table, None, Some(correct_key), &[
            (data_path.as_str(), 0),
            (data_path.as_str(), 1),
            (data_path.as_str(), 2),
        ])
        .await;
        assert_eq!(correct_delete.partition_spec_id(), cur_spec_id);
        let table = add_deletes(&catalog, &table, vec![correct_delete]).await;
        assert!(
            scan_ids(&table).await.is_empty(),
            "the correctly-stamped delete DOES apply — the stamp is the only difference"
        );

        // THE FIX: configure the spec and the same unstamped call cannot produce that artifact.
        let config = PositionDeleteWriterConfig::new().expect("pos-delete config");
        let err = PositionDeleteFileWriterBuilder::new(
            rolling_builder(&table, "pos-del", config.schema().clone()),
            config,
        )
        .with_partition_spec(table.metadata().default_partition_spec().as_ref().clone())
        .build(None)
        .await
        .expect_err("a partitioned spec with no PartitionKey must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string().contains("must carry its partition tuple"),
            "unexpected error: {err}"
        );
    }

    /// UNWRITABLE TABLE (the other half of the same defect).
    ///
    /// Table: spec 0 = `identity(dept)`, evolved by removing its only field — on V2 the field is
    /// OMITTED, so the current spec is UNPARTITIONED with a NON-ZERO id. Without a configured spec
    /// the writers stamp 0, and spec 0 is partitioned, so the commit rejects the empty tuple: the
    /// table cannot be written at all (control leg below, the exact pre-fix failure).
    ///
    /// With the spec configured, the whole round-trip works: data commits, the delete carries the
    /// same spec id, and the read side applies it.
    #[tokio::test]
    async fn test_e2e_unpartitioned_nonzero_spec_round_trips_with_configured_spec() {
        let catalog = new_memory_catalog().await;
        let table = make_table(&catalog, identity_dept_spec(0)).await;
        let (table, cur_spec_id) = evolve_remove_field(&catalog, &table, "dept").await;
        let cur_spec = table.metadata().default_partition_spec().as_ref().clone();
        assert!(
            cur_spec.is_unpartitioned(),
            "fixture: the current spec is unpartitioned"
        );
        assert_ne!(cur_spec_id, 0, "fixture: its id is NOT 0");

        // CONTROL — the legacy path: stamped 0, and spec 0 is partitioned ⇒ the commit rejects it.
        let unstamped = write_data_file(&table, None, None, &[(1, "eng")]).await;
        assert_eq!(unstamped.partition_spec_id(), 0);
        let err = fast_append(&catalog, &table, vec![unstamped])
            .await
            .expect_err("a file claiming partitioned spec 0 with an empty tuple must be rejected");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string()
                .contains("Partition value is not compatible with partition type"),
            "unexpected error: {err}"
        );

        // FIXED — configure the current spec; the file claims it and commits.
        let data = write_data_file(&table, Some(cur_spec.clone()), None, &[
            (1, "eng"),
            (2, "eng"),
            (3, "eng"),
        ])
        .await;
        assert_eq!(
            data.partition_spec_id(),
            cur_spec_id,
            "the data file must claim the CURRENT unpartitioned spec"
        );
        let data_path = data.file_path().to_string();
        let table = fast_append(&catalog, &table, vec![data])
            .await
            .expect("commit data under the configured spec");
        assert_eq!(scan_ids(&table).await, vec![1, 2, 3]);

        // The delete carries the same spec id — and is APPLIED. The ROW-LEVEL outcome is asserted
        // first: it is what discriminates this test from the wrong-spec twin above (identical
        // machinery, the ONLY difference being whether the two spec ids agree), and asserting the
        // stamp first would mask it — every wrong-stamp mutation would red on the stamp instead.
        let delete = write_pos_delete(&table, Some(cur_spec), None, &[
            (data_path.as_str(), 0),
            (data_path.as_str(), 2),
        ])
        .await;
        let delete_spec_id = delete.partition_spec_id();
        let table = add_deletes(&catalog, &table, vec![delete]).await;
        assert_eq!(
            scan_ids(&table).await,
            vec![2],
            "positions 0 and 2 must be deleted — the delete and the data agree on the spec id"
        );
        // Corroborating guard (not the discriminating assertion): the delete really did claim the
        // current spec rather than reaching the data by some other route.
        assert_eq!(delete_spec_id, cur_spec_id);
    }

    /// SILENT UNDER-DELETE, ISOLATED ON THE SPEC ID ALONE (2026-07-25, Critic-supplied fixture).
    ///
    /// The engine-reachable twin above cannot attribute the miss to the spec id, because its
    /// unkeyed delete also differs in the partition TUPLE. Here the tuple is held CONSTANT and only
    /// the spec id varies: spec 0 is `truncate[5](dept)` and the current spec is `identity(dept)`,
    /// and for `"eng"` both transforms yield the byte-identical tuple `{"eng"}`. The delete is built
    /// from a `PartitionKey` on the OLD spec, so it carries the data file's exact partition value
    /// while claiming a different `partition_spec_id`.
    ///
    /// The commit ACCEPTS it (`validate_partition_value` checks the tuple against the spec the file
    /// CLAIMS — arity 1, type string — never *which* spec the file belongs to) and the read side
    /// then drops it at the `data_file.partition_spec_id == delete.partition_spec_id` condition in
    /// `DeleteFileIndex::get_deletes_for_data_file`. This is the fixture that makes that condition
    /// load-bearing: deleting it turns this test red.
    #[tokio::test]
    async fn test_e2e_same_tuple_wrong_spec_id_alone_silently_under_deletes() {
        let catalog = new_memory_catalog().await;
        let old_spec = truncate5_dept_spec(0);
        let table = make_table(&catalog, old_spec.clone()).await;

        // spec 0 `truncate[5](dept)` → (remove) unpartitioned → (add) `identity(dept)`.
        let (table, _) = evolve_remove_field(&catalog, &table, "dept_trunc").await;
        let (table, cur_spec_id) = evolve_add_field(&catalog, &table, "dept").await;
        let cur_spec = table.metadata().default_partition_spec().as_ref().clone();
        assert_ne!(cur_spec_id, 0, "fixture: the current spec is not spec 0");
        assert_eq!(
            cur_spec.fields().len(),
            old_spec.fields().len(),
            "fixture: both specs have the same partition arity"
        );

        // The one tuple both specs produce for dept = "eng".
        let tuple = Struct::from_iter([Some(Literal::string("eng"))]);

        let data = write_data_file(
            &table,
            None,
            Some(
                PartitionKey::new(
                    cur_spec.clone(),
                    table.metadata().current_schema().clone(),
                    tuple.clone(),
                )
                .expect("PartitionKey::new: valid partition tuple"),
            ),
            &[(1, "eng"), (2, "eng")],
        )
        .await;
        assert_eq!(data.partition_spec_id(), cur_spec_id);
        assert_eq!(
            data.partition, tuple,
            "fixture: the data carries {{\"eng\"}}"
        );
        let data_path = data.file_path().to_string();
        let table = fast_append(&catalog, &table, vec![data])
            .await
            .expect("commit data");
        assert_eq!(scan_ids(&table).await, vec![1, 2]);

        // The delete: SAME tuple, OLD spec id.
        let wrong_key = PartitionKey::new(
            old_spec,
            table.metadata().current_schema().clone(),
            tuple.clone(),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let delete = write_pos_delete(&table, None, Some(wrong_key), &[
            (data_path.as_str(), 0),
            (data_path.as_str(), 1),
        ])
        .await;
        assert_eq!(
            delete.partition, tuple,
            "ISOLATION: the delete's tuple equals the data's, byte for byte"
        );
        assert_eq!(
            delete.partition_spec_id(),
            0,
            "ISOLATION: the spec id is the ONLY difference"
        );

        let table = add_deletes(&catalog, &table, vec![delete]).await;
        assert_eq!(
            scan_ids(&table).await,
            vec![1, 2],
            "a delete differing ONLY in spec id commits and silently never applies"
        );

        // POSITIVE CONTROL: the same positions, the same tuple, the CURRENT spec id ⇒ applied.
        let correct_key =
            PartitionKey::new(cur_spec, table.metadata().current_schema().clone(), tuple)
                .expect("PartitionKey::new: valid partition tuple");
        let correct_delete = write_pos_delete(&table, None, Some(correct_key), &[
            (data_path.as_str(), 0),
            (data_path.as_str(), 1),
        ])
        .await;
        let correct_spec_id = correct_delete.partition_spec_id();
        let table = add_deletes(&catalog, &table, vec![correct_delete]).await;
        assert!(
            scan_ids(&table).await.is_empty(),
            "the same delete under the matching spec id DOES apply"
        );
        assert_eq!(correct_spec_id, cur_spec_id);
    }

    /// EQUALITY DELETES ARE THE OTHER DIRECTION: a keyless one is GLOBAL, not inert.
    ///
    /// The Iceberg spec says "equality delete files stored with an unpartitioned spec are applied as
    /// global deletes", and both engines implement it — Rust routes on the file's EMPTY TUPLE
    /// (`PopulatedDeleteFileIndex::new` → `global_equality_deletes`), Java on the SPEC being
    /// unpartitioned (`DeleteFileIndex.java` `add(...)`, 1.10.0). The global bucket is consulted with
    /// NO spec-id and NO partition condition, only the sequence-number filter.
    ///
    /// So the hazard of a missing `PartitionKey` INVERTS between the two delete kinds: for a
    /// position delete it under-deletes (rows resurrect); for an equality delete it OVER-deletes,
    /// table-wide. `docs/ENGINE_CONTRACT.md` §7a must say so, and this test is its pin.
    #[tokio::test]
    async fn test_e2e_keyless_equality_delete_is_global_not_inert() {
        let catalog = new_memory_catalog().await;
        let table = make_table(
            &catalog,
            PartitionSpec::builder(test_schema())
                .with_spec_id(0)
                .build()
                .expect("unpartitioned spec 0"),
        )
        .await;
        let (table, cur_spec_id) = evolve_add_field(&catalog, &table, "dept").await;
        let cur_spec = table.metadata().default_partition_spec().as_ref().clone();
        let schema = table.metadata().current_schema().clone();

        let eng = write_data_file(
            &table,
            None,
            Some(
                PartitionKey::new(
                    cur_spec.clone(),
                    schema.clone(),
                    Struct::from_iter([Some(Literal::string("eng"))]),
                )
                .expect("PartitionKey::new: valid partition tuple"),
            ),
            &[(1, "eng"), (2, "eng")],
        )
        .await;
        let ops = write_data_file(
            &table,
            None,
            Some(
                PartitionKey::new(
                    cur_spec,
                    schema,
                    Struct::from_iter([Some(Literal::string("ops"))]),
                )
                .expect("PartitionKey::new: valid partition tuple"),
            ),
            &[(1, "ops"), (3, "ops")],
        )
        .await;
        assert_eq!(eng.partition_spec_id(), cur_spec_id);
        let table = fast_append(&catalog, &table, vec![eng, ops])
            .await
            .expect("commit data");
        assert_eq!(scan_ids(&table).await, vec![1, 1, 2, 3]);

        // No key, no configured spec: spec 0 AND an EMPTY tuple.
        let delete = write_eq_delete_on_id(&table, None, None, &[(1, "eng")]).await;
        assert_eq!(delete.partition_spec_id(), 0);
        assert!(
            delete.partition.fields().is_empty(),
            "the keyless equality delete carries an empty tuple"
        );

        let table = add_deletes(&catalog, &table, vec![delete]).await;
        assert_eq!(
            scan_ids(&table).await,
            vec![2, 3],
            "id = 1 is gone from BOTH partitions: the keyless equality delete is GLOBAL, and it \
             ignored the spec id it claimed"
        );
    }

    /// The contrast leg: an equality delete with a NON-EMPTY tuple IS partition-scoped, and is the
    /// case §7a's `(spec_id, partition)` pairing rule actually covers.
    ///
    /// Same fixture as the global twin; the only change is that the delete is built from the `eng`
    /// `PartitionKey`. It must remove `id = 1` from `eng` ONLY — leaving `ops`'s `id = 1` alive,
    /// which is exactly what distinguishes `[1, 2, 3]` here from `[2, 3]` there.
    #[tokio::test]
    async fn test_e2e_keyed_equality_delete_is_partition_scoped() {
        let catalog = new_memory_catalog().await;
        let table = make_table(
            &catalog,
            PartitionSpec::builder(test_schema())
                .with_spec_id(0)
                .build()
                .expect("unpartitioned spec 0"),
        )
        .await;
        let (table, cur_spec_id) = evolve_add_field(&catalog, &table, "dept").await;
        let cur_spec = table.metadata().default_partition_spec().as_ref().clone();
        let schema = table.metadata().current_schema().clone();
        let eng_tuple = Struct::from_iter([Some(Literal::string("eng"))]);

        let eng = write_data_file(
            &table,
            None,
            Some(
                PartitionKey::new(cur_spec.clone(), schema.clone(), eng_tuple.clone())
                    .expect("PartitionKey::new: valid partition tuple"),
            ),
            &[(1, "eng"), (2, "eng")],
        )
        .await;
        let ops = write_data_file(
            &table,
            None,
            Some(
                PartitionKey::new(
                    cur_spec.clone(),
                    schema.clone(),
                    Struct::from_iter([Some(Literal::string("ops"))]),
                )
                .expect("PartitionKey::new: valid partition tuple"),
            ),
            &[(1, "ops"), (3, "ops")],
        )
        .await;
        let table = fast_append(&catalog, &table, vec![eng, ops])
            .await
            .expect("commit data");
        assert_eq!(scan_ids(&table).await, vec![1, 1, 2, 3]);

        let delete = write_eq_delete_on_id(
            &table,
            None,
            Some(
                PartitionKey::new(cur_spec, schema, eng_tuple.clone())
                    .expect("PartitionKey::new: valid partition tuple"),
            ),
            &[(1, "eng")],
        )
        .await;
        let delete_spec_id = delete.partition_spec_id();
        let delete_partition = delete.partition.clone();

        let table = add_deletes(&catalog, &table, vec![delete]).await;
        assert_eq!(
            scan_ids(&table).await,
            vec![1, 2, 3],
            "only eng's id = 1 is deleted — ops's survives, so this delete is partition-scoped"
        );
        // Corroborating guards, after the row-level outcome.
        assert_eq!(delete_spec_id, cur_spec_id);
        assert_eq!(delete_partition, eng_tuple);
    }
}
