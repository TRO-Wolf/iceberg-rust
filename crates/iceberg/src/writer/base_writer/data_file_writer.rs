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

//! This module provide `DataFileWriter`.
//!
//! It also hosts `resolve_partition_spec_id` — the partition-spec-id stamping rule shared by all
//! three base writers (data, position-delete, equality-delete).

use std::borrow::Cow;

use arrow_array::RecordBatch;

use crate::spec::{
    DEFAULT_PARTITION_SPEC_ID, DataContentType, DataFile, PartitionKey, PartitionSpec, SchemaRef,
};
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::{RollingFileWriter, RollingFileWriterBuilder};
use crate::writer::write_defaults::apply_write_defaults;
use crate::writer::{CurrentFileStatus, IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Resolve the `partition_spec_id` a base writer stamps on every file it produces, validating the
/// (spec, partition key) pair up front.
///
/// # Why this exists
///
/// Java takes the [`PartitionSpec`] as a REQUIRED constructor argument on every file builder —
/// `FileMetadata.Builder(spec)` (`core/.../FileMetadata.java`: `this.specId = spec.specId()`) and
/// `DataFiles.Builder(spec)` — and stamps `spec.specId()` unconditionally, so a Java-written file
/// always claims a spec that exists in the table. Rust's `DataFileBuilder` instead *defaults*
/// `partition_spec_id` to [`DEFAULT_PARTITION_SPEC_ID`] (0), a fabricated value with no table behind
/// it. A file stamped 0 by that default is silently wrong whenever the spec it was actually written
/// under is not spec 0 — see `docs/ENGINE_CONTRACT.md` §7a for the observable outcomes (a same-arity
/// wrong-spec POSITION delete commits and then never applies; a keyless EQUALITY delete becomes a
/// GLOBAL delete instead; an unpartitioned current spec with a non-zero id cannot be written at all).
///
/// # Precedence
///
/// 1. **The [`PartitionKey`]'s own spec**, when a key is given. The key carries the partition tuple
///    *and* the spec that tuple was produced from, so it is authoritative — and a delete file must
///    claim the spec of the DATA FILES it deletes from, which is not necessarily the table's current
///    spec. A key whose spec differs from `configured_spec` is therefore legal, not an error.
/// 2. **The spec configured on the builder** (`with_partition_spec`), when there is no key.
/// 3. **[`DEFAULT_PARTITION_SPEC_ID`] (0)** when neither is given — the legacy path, kept so the
///    pre-existing builder API stays source-compatible. It is correct only for a table whose current
///    spec really is spec 0.
///
/// # The partitioned-without-a-key rejection
///
/// Case 2 with a spec that has partition fields is rejected with an [`ErrorKind::DataInvalid`]: the
/// file would carry the builder's default EMPTY partition tuple while claiming a spec whose
/// partition type has fields, which the commit path rejects anyway
/// (`SnapshotProducer::validate_partition_value`, Java `PartitionData` accessors) — failing here
/// names the actual mistake instead of surfacing an arity error one layer later.
///
/// The test is the spec's partition-field ARITY (`fields().is_empty()`), deliberately NOT
/// [`PartitionSpec::is_unpartitioned`], which is also `true` for an ALL-VOID spec (a V1 spec whose
/// fields were void-replaced). An all-void spec still has partition fields, so its partition type
/// still has that arity and a file under it still needs a tuple (of nulls) — `is_unpartitioned`
/// would wave it through into the same commit-time arity failure this check exists to prevent.
pub(crate) fn resolve_partition_spec_id(
    configured_spec: Option<&PartitionSpec>,
    partition_key: Option<&PartitionKey>,
) -> Result<i32> {
    match (partition_key, configured_spec) {
        // The key is authoritative — it carries both the tuple and the spec it came from.
        (Some(partition_key), _) => Ok(partition_key.spec().spec_id()),
        (None, Some(spec)) if !spec.fields().is_empty() => Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Partition spec {} has {} partition field(s) but the writer was built without a \
                 PartitionKey: a file written under a partitioned spec must carry its partition tuple",
                spec.spec_id(),
                spec.fields().len()
            ),
        )),
        (None, Some(spec)) => Ok(spec.spec_id()),
        (None, None) => Ok(DEFAULT_PARTITION_SPEC_ID),
    }
}

/// Builder for `DataFileWriter`.
#[derive(Debug)]
pub struct DataFileWriterBuilder<B: FileWriterBuilder, L: LocationGenerator, F: FileNameGenerator> {
    inner: RollingFileWriterBuilder<B, L, F>,
    partition_spec: Option<PartitionSpec>,
}

impl<B, L, F> DataFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Create a new `DataFileWriterBuilder` using a `RollingFileWriterBuilder`.
    ///
    /// Prefer chaining [`with_partition_spec`](Self::with_partition_spec): without it, a writer built
    /// with no [`PartitionKey`] falls back to stamping `DEFAULT_PARTITION_SPEC_ID` (0) — see
    /// `resolve_partition_spec_id`.
    pub fn new(inner: RollingFileWriterBuilder<B, L, F>) -> Self {
        Self {
            inner,
            partition_spec: None,
        }
    }

    /// Set the [`PartitionSpec`] the produced files are written under.
    ///
    /// This is the Rust counterpart of Java's REQUIRED `DataFiles.Builder(spec)` argument. It is used
    /// only when the writer is built WITHOUT a [`PartitionKey`]; a key always wins, because it
    /// carries the spec its tuple was produced from. See `resolve_partition_spec_id` for the full
    /// precedence and for why a partitioned spec with no key is rejected.
    ///
    /// **This writer OWNS `partition_spec_id` on every [`DataFile`] it emits.** `close()` sets the
    /// field unconditionally, so a custom [`FileWriter`](crate::writer::file_writer::FileWriter)
    /// that stamps it on the `DataFileBuilder` it returns will be overridden; give the spec to this
    /// builder instead. (No in-tree `FileWriter` stamps it — `ParquetWriter` leaves the field at its
    /// derive default.)
    pub fn with_partition_spec(mut self, partition_spec: PartitionSpec) -> Self {
        self.partition_spec = Some(partition_spec);
        self
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriterBuilder for DataFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    type R = DataFileWriter<B, L, F>;

    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
        let partition_spec_id =
            resolve_partition_spec_id(self.partition_spec.as_ref(), partition_key.as_ref())?;
        Ok(DataFileWriter {
            inner: Some(self.inner.build()),
            partition_key,
            partition_spec_id,
            schema: self.inner.iceberg_schema().cloned(),
        })
    }
}

/// A writer write data is within one spec/partition.
#[derive(Debug)]
pub struct DataFileWriter<B: FileWriterBuilder, L: LocationGenerator, F: FileNameGenerator> {
    inner: Option<RollingFileWriter<B, L, F>>,
    partition_key: Option<PartitionKey>,
    /// The spec id stamped on every produced file, resolved once at build time by
    /// `resolve_partition_spec_id`.
    partition_spec_id: i32,
    schema: Option<SchemaRef>,
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriter for DataFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        let filled = match &self.schema {
            Some(schema) => apply_write_defaults(schema, &batch)?,
            None => Cow::Borrowed(&batch),
        };
        if let Some(writer) = self.inner.as_mut() {
            writer.write(&self.partition_key, filled.as_ref()).await
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Writer is not initialized!",
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
                    res.content(DataContentType::Data);
                    // ALWAYS stamp the spec id (Java `DataFiles.Builder(spec)` does), never only when
                    // a partition key happens to be present — see `resolve_partition_spec_id`.
                    res.partition_spec_id(self.partition_spec_id);
                    if let Some(pk) = self.partition_key.as_ref() {
                        res.partition(pk.data().clone());
                    }
                    res.build().map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Failed to build data file: {e}"),
                        )
                    })
                })
                .collect()
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Data file writer has been closed.",
            ))
        }
    }
}

impl<B, L, F> CurrentFileStatus for DataFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    fn current_file_path(&self) -> String {
        // Post-`close()` the inner writer is taken; report empty rather than panicking on a
        // status query against a closed writer (same posture as `RollingFileWriter`).
        self.inner
            .as_ref()
            .map(|inner| inner.current_file_path())
            .unwrap_or_default()
    }

    fn current_row_num(&self) -> usize {
        self.inner
            .as_ref()
            .map(|inner| inner.current_row_num())
            .unwrap_or(0)
    }

    fn current_written_size(&self) -> usize {
        self.inner
            .as_ref()
            .map(|inner| inner.current_written_size())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
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
    use crate::writer::{IcebergWriter, IcebergWriterBuilder, RecordBatch};
    use crate::{ErrorKind, Result};

    #[tokio::test]
    async fn test_parquet_writer() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
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
            .build(None)
            .await
            .unwrap();

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

        // Post-close CurrentFileStatus must not panic (inner writer is taken on close).
        use crate::writer::CurrentFileStatus;
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

        let parquet_reader =
            ArrowReaderMetadata::load(&input_content, ArrowReaderOptions::default())
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
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
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

        let parquet_reader =
            ArrowReaderMetadata::load(&input_content, ArrowReaderOptions::default())?;

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

    // ============================================================================================
    // Partition-spec-id stamping (`resolve_partition_spec_id`).
    //
    // Java takes the spec as a REQUIRED builder argument and stamps `spec.specId()` unconditionally
    // (`DataFiles.Builder(spec)` / `FileMetadata.Builder(spec)`); Rust's `DataFileBuilder` defaults
    // the field to `DEFAULT_PARTITION_SPEC_ID` (0). These pin the precedence and the rejections.
    // ============================================================================================

    /// `1: id long`, `2: dept string`, both required — the fixture schema for the stamping tests.
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

    /// A one-field spec over `dept` under `spec_id`, with the given transform.
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

    /// A `DataFileWriterBuilder` writing one-row files under `temp_dir`.
    fn stamp_writer_builder(
        file_io: &FileIO,
        temp_dir: &TempDir,
        schema: &Arc<Schema>,
    ) -> DataFileWriterBuilder<
        ParquetWriterBuilder,
        DefaultLocationGenerator,
        DefaultFileNameGenerator,
    > {
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

    /// One row `(1, "eng")` in the fixture schema.
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

    /// CONFIGURED SPEC, NO KEY. An unpartitioned spec whose id is NOT 0 must be stamped as itself.
    /// Before the stamp fix this produced a file claiming spec 0 — which the commit path then
    /// validates against spec 0's (possibly partitioned) type, and which no reader ever pairs with
    /// data files under the real spec.
    #[tokio::test]
    async fn test_data_file_writer_stamps_configured_unpartitioned_spec_id() -> Result<()> {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = stamp_test_schema();
        // An UNPARTITIONED spec with a NON-ZERO id — reachable by evolving a partitioned spec's only
        // field away on V2.
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

    /// CONFIGURED PARTITIONED SPEC, NO KEY. Rejected at build time: the file would claim a spec whose
    /// partition type has fields while carrying the builder's empty tuple.
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

    /// ALL-VOID SPEC, NO KEY. `is_unpartitioned()` is TRUE for an all-void spec, but its partition
    /// TYPE still has one field — so a file under it still needs a (null) tuple. The rejection is
    /// keyed on partition-field ARITY, not on `is_unpartitioned()`; keying it on the latter would
    /// wave this case through into the commit-time arity failure the check exists to prevent.
    #[tokio::test]
    async fn test_data_file_writer_rejects_all_void_spec_without_partition_key() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = stamp_test_schema();
        let void_spec = dept_spec(&schema, 5, Transform::Void);

        // Fixture sanity: this is the trap shape — unpartitioned by the predicate, 1-field by arity.
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

    /// ALL-VOID SPEC WITH A KEY. The legal counterpart of the leg above: a one-field null tuple is
    /// accepted and stamped under the void spec's own id.
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

    /// PRECEDENCE. The `PartitionKey`'s own spec wins over the configured spec — the key carries the
    /// spec its tuple was produced from, and a file may legitimately be written under an OLDER spec
    /// than the one the builder was configured with.
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

    /// LEGACY PATH PIN. With neither a configured spec nor a key the writer still stamps
    /// `DEFAULT_PARTITION_SPEC_ID` (0) — source-compatible with every pre-existing caller, and
    /// correct only when the table's current spec really is spec 0. Pinned so that changing it is a
    /// deliberate (breaking) act. See `docs/ENGINE_CONTRACT.md` §7a.
    #[tokio::test]
    async fn test_data_file_writer_without_spec_or_key_stamps_default_zero() -> Result<()> {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let schema = stamp_test_schema();

        let mut writer = stamp_writer_builder(&file_io, &temp_dir, &schema)
            .build(None)
            .await?;
        writer.write(stamp_test_batch()).await?;
        let data_files = writer.close().await?;

        assert_eq!(data_files[0].partition_spec_id(), 0);
        assert_eq!(data_files[0].partition, Struct::empty());
        Ok(())
    }
}
