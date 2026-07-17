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

//! The module contains the file writer for parquet file format.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, RecordBatch, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef, TimeUnit};
use bytes::Bytes;
use futures::future::BoxFuture;
use itertools::Itertools;
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::async_writer::AsyncFileWriter as ArrowAsyncFileWriter;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::properties::WriterProperties;
use parquet::file::statistics::Statistics;

use super::{FileWriter, FileWriterBuilder};
use crate::arrow::{
    ArrowFileReader, DEFAULT_MAP_FIELD_NAME, FieldMatchMode, NanValueCountVisitor, UTC_TIME_ZONE,
    get_parquet_stat_max_as_datum, get_parquet_stat_min_as_datum,
};
use crate::io::{FileIO, FileWrite, OutputFile};
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Datum, ListType, Literal, MapType,
    MetricsConfig, MetricsMode, NestedFieldRef, PartitionSpec, PrimitiveType, Schema, SchemaRef,
    SchemaVisitor, Struct, StructType, TableMetadata, Type, visit_schema,
};
use crate::transform::create_transform_function;
use crate::writer::{CurrentFileStatus, DataFile};
use crate::{Error, ErrorKind, Result};

/// ParquetWriterBuilder is used to builder a [`ParquetWriter`]
#[derive(Clone, Debug)]
pub struct ParquetWriterBuilder {
    props: WriterProperties,
    schema: SchemaRef,
    match_mode: FieldMatchMode,
    metrics_config: MetricsConfig,
}

impl ParquetWriterBuilder {
    /// Create a new `ParquetWriterBuilder`
    /// To construct the write result, the schema should contain the `PARQUET_FIELD_ID_META_KEY` metadata for each field.
    ///
    /// The column metrics use the default [`MetricsConfig`] (`truncate(16)`), matching Java
    /// `MetricsConfig.getDefault()` — so string/binary lower/upper bounds are truncated to 16
    /// code points / bytes. Use [`ParquetWriterBuilder::with_metrics_config`] to override
    /// (e.g. [`MetricsConfig::for_position_delete`] for delete files, which keeps `file_path`
    /// bounds full).
    pub fn new(props: WriterProperties, schema: SchemaRef) -> Self {
        Self::new_with_match_mode(props, schema, FieldMatchMode::Id)
    }

    /// Create a new `ParquetWriterBuilder` with custom match mode
    pub fn new_with_match_mode(
        props: WriterProperties,
        schema: SchemaRef,
        match_mode: FieldMatchMode,
    ) -> Self {
        Self {
            props,
            schema,
            match_mode,
            metrics_config: MetricsConfig::default(),
        }
    }

    /// Set the [`MetricsConfig`] governing which column statistics (counts, truncated/full bounds)
    /// are persisted in the resulting data file, overriding the default `truncate(16)`.
    ///
    /// Chains onto either constructor, e.g. for a position-delete writer that must keep `file_path`
    /// bounds full: `ParquetWriterBuilder::new(props, schema).with_metrics_config(MetricsConfig::for_position_delete())`.
    pub fn with_metrics_config(mut self, metrics_config: MetricsConfig) -> Self {
        self.metrics_config = metrics_config;
        self
    }
}

impl FileWriterBuilder for ParquetWriterBuilder {
    type R = ParquetWriter;

    async fn build(&self, output_file: OutputFile) -> Result<Self::R> {
        Ok(ParquetWriter {
            schema: self.schema.clone(),
            writer_arrow_schema: Arc::new(self.schema.as_ref().try_into()?),
            inner_writer: None,
            writer_properties: self.props.clone(),
            current_row_num: 0,
            output_file,
            nan_value_count_visitor: NanValueCountVisitor::new_with_match_mode(self.match_mode),
            metrics_config: self.metrics_config.clone(),
        })
    }
}

/// A mapping from Parquet column path names to internal field id
struct IndexByParquetPathName {
    name_to_id: HashMap<String, i32>,

    field_names: Vec<String>,

    field_id: i32,
}

impl IndexByParquetPathName {
    /// Creates a new, empty `IndexByParquetPathName`
    pub fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            field_names: Vec::new(),
            field_id: 0,
        }
    }

    /// Retrieves the internal field ID
    pub fn get(&self, name: &str) -> Option<&i32> {
        self.name_to_id.get(name)
    }
}

impl Default for IndexByParquetPathName {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaVisitor for IndexByParquetPathName {
    type T = ();

    fn before_struct_field(&mut self, field: &NestedFieldRef) -> Result<()> {
        self.field_names.push(field.name.to_string());
        self.field_id = field.id;
        Ok(())
    }

    fn after_struct_field(&mut self, _field: &NestedFieldRef) -> Result<()> {
        self.field_names.pop();
        Ok(())
    }

    fn before_list_element(&mut self, field: &NestedFieldRef) -> Result<()> {
        self.field_names.push(format!("list.{}", field.name));
        self.field_id = field.id;
        Ok(())
    }

    fn after_list_element(&mut self, _field: &NestedFieldRef) -> Result<()> {
        self.field_names.pop();
        Ok(())
    }

    fn before_map_key(&mut self, field: &NestedFieldRef) -> Result<()> {
        self.field_names
            .push(format!("{DEFAULT_MAP_FIELD_NAME}.key"));
        self.field_id = field.id;
        Ok(())
    }

    fn after_map_key(&mut self, _field: &NestedFieldRef) -> Result<()> {
        self.field_names.pop();
        Ok(())
    }

    fn before_map_value(&mut self, field: &NestedFieldRef) -> Result<()> {
        self.field_names
            .push(format!("{DEFAULT_MAP_FIELD_NAME}.value"));
        self.field_id = field.id;
        Ok(())
    }

    fn after_map_value(&mut self, _field: &NestedFieldRef) -> Result<()> {
        self.field_names.pop();
        Ok(())
    }

    fn schema(&mut self, _schema: &Schema, _value: Self::T) -> Result<Self::T> {
        Ok(())
    }

    fn field(&mut self, _field: &NestedFieldRef, _value: Self::T) -> Result<Self::T> {
        Ok(())
    }

    fn r#struct(&mut self, _struct: &StructType, _results: Vec<Self::T>) -> Result<Self::T> {
        Ok(())
    }

    fn list(&mut self, _list: &ListType, _value: Self::T) -> Result<Self::T> {
        Ok(())
    }

    fn map(&mut self, _map: &MapType, _key_value: Self::T, _value: Self::T) -> Result<Self::T> {
        Ok(())
    }

    fn primitive(&mut self, _p: &PrimitiveType) -> Result<Self::T> {
        let full_name = self.field_names.iter().map(String::as_str).join(".");
        let field_id = self.field_id;
        if let Some(existing_field_id) = self.name_to_id.get(full_name.as_str()) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid schema: multiple fields for name {full_name}: {field_id} and {existing_field_id}"
                ),
            ));
        } else {
            self.name_to_id.insert(full_name, field_id);
        }

        Ok(())
    }
}

/// `ParquetWriter`` is used to write arrow data into parquet file on storage.
pub struct ParquetWriter {
    schema: SchemaRef,
    /// The Arrow schema the underlying Parquet `ArrowWriter` is built with — derived once from
    /// `schema` (Iceberg->Arrow, so `timestamptz` carries `UTC_TIME_ZONE`). Cached here so the
    /// hot write path reuses it for both writer construction and UTC-alias batch normalization.
    writer_arrow_schema: ArrowSchemaRef,
    output_file: OutputFile,
    inner_writer: Option<AsyncArrowWriter<AsyncFileWriter>>,
    writer_properties: WriterProperties,
    current_row_num: usize,
    nan_value_count_visitor: NanValueCountVisitor,
    metrics_config: MetricsConfig,
}

/// Used to aggregate min and max value of each column.
struct MinMaxColAggregator {
    lower_bounds: HashMap<i32, Datum>,
    upper_bounds: HashMap<i32, Datum>,
    schema: SchemaRef,
    metrics_config: MetricsConfig,
}

impl MinMaxColAggregator {
    /// Creates new and empty `MinMaxColAggregator`
    fn new(schema: SchemaRef, metrics_config: MetricsConfig) -> Self {
        Self {
            lower_bounds: HashMap::new(),
            upper_bounds: HashMap::new(),
            schema,
            metrics_config,
        }
    }

    /// The resolved [`MetricsMode`] for a column, by its (dotted) schema name; columns absent from
    /// the schema fall back to the table default.
    fn mode_for(&self, field_id: i32) -> MetricsMode {
        match self.schema.name_by_field_id(field_id) {
            Some(name) => self.metrics_config.column_mode(name),
            None => self.metrics_config.default_mode_of(),
        }
    }

    fn update_state_min(&mut self, field_id: i32, datum: Datum) {
        self.lower_bounds
            .entry(field_id)
            .and_modify(|e| {
                if *e > datum {
                    *e = datum.clone()
                }
            })
            .or_insert(datum);
    }

    fn update_state_max(&mut self, field_id: i32, datum: Datum) {
        self.upper_bounds
            .entry(field_id)
            .and_modify(|e| {
                if *e < datum {
                    *e = datum.clone()
                }
            })
            .or_insert(datum);
    }

    /// Update statistics
    fn update(&mut self, field_id: i32, value: Statistics) -> Result<()> {
        let Some(ty) = self
            .schema
            .field_by_id(field_id)
            .map(|f| f.field_type.as_ref())
        else {
            // Following java implementation: https://github.com/apache/iceberg/blob/29a2c456353a6120b8c882ed2ab544975b168d7b/parquet/src/main/java/org/apache/iceberg/parquet/ParquetUtil.java#L163
            // Ignore the field if it is not in schema.
            return Ok(());
        };
        let Type::Primitive(ty) = ty.clone() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("Composed type {ty} is not supported for min max aggregation."),
            ));
        };

        if value.min_is_exact() {
            let Some(min_datum) = get_parquet_stat_min_as_datum(&ty, &value)? else {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("Statistics {value} is not match with field type {ty}."),
                ));
            };

            self.update_state_min(field_id, min_datum);
        }

        if value.max_is_exact() {
            let Some(max_datum) = get_parquet_stat_max_as_datum(&ty, &value)? else {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("Statistics {value} is not match with field type {ty}."),
                ));
            };

            self.update_state_max(field_id, max_datum);
        }

        Ok(())
    }

    /// Returns lower and upper bounds, each truncated per the column's [`MetricsMode`] (Java
    /// `ParquetMetrics.truncateLowerBound` / `truncateUpperBound`). A bound is dropped when the
    /// mode persists none (`None`/`Counts`) or when an upper bound cannot be truncated up.
    fn produce(self) -> (HashMap<i32, Datum>, HashMap<i32, Datum>) {
        let mut lower_bounds = HashMap::with_capacity(self.lower_bounds.len());
        for (field_id, datum) in &self.lower_bounds {
            if let Some(bound) = self.mode_for(*field_id).truncate_lower_bound(datum) {
                lower_bounds.insert(*field_id, bound);
            }
        }
        let mut upper_bounds = HashMap::with_capacity(self.upper_bounds.len());
        for (field_id, datum) in &self.upper_bounds {
            if let Some(bound) = self.mode_for(*field_id).truncate_upper_bound(datum) {
                upper_bounds.insert(*field_id, bound);
            }
        }
        (lower_bounds, upper_bounds)
    }
}

impl ParquetWriter {
    /// Converts parquet files to data files
    #[allow(dead_code)]
    pub(crate) async fn parquet_files_to_data_files(
        file_io: &FileIO,
        file_paths: Vec<String>,
        table_metadata: &TableMetadata,
    ) -> Result<Vec<DataFile>> {
        // TODO: support adding to partitioned table
        let mut data_files: Vec<DataFile> = Vec::new();

        for file_path in file_paths {
            let input_file = file_io.new_input(&file_path)?;
            let file_metadata = input_file.metadata().await?;
            let file_size_in_bytes = file_metadata.size as usize;
            let reader = input_file.reader().await?;

            let mut parquet_reader = ArrowFileReader::new(file_metadata, reader);
            let parquet_metadata = parquet_reader.get_metadata(None).await.map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Error reading Parquet metadata: {err}"),
                )
            })?;
            let mut builder = ParquetWriter::parquet_to_data_file_builder(
                table_metadata.current_schema().clone(),
                parquet_metadata,
                file_size_in_bytes,
                file_path,
                // TODO: Implement nan_value_counts here
                HashMap::new(),
                &MetricsConfig::from_properties(table_metadata.properties()),
            )?;
            builder.partition_spec_id(table_metadata.default_partition_spec_id());
            let data_file = builder.build().unwrap();
            data_files.push(data_file);
        }

        Ok(data_files)
    }

    /// `ParquetMetadata` to data file builder
    ///
    /// `metrics_config` governs, per column, which statistics are persisted (Java
    /// `ParquetMetrics`): [`MetricsMode::None`] columns are skipped entirely (no sizes, counts, or
    /// bounds), [`MetricsMode::Counts`] keeps counts but drops bounds, and `Truncate`/`Full` keep
    /// counts plus (truncated/full) bounds.
    pub(crate) fn parquet_to_data_file_builder(
        schema: SchemaRef,
        metadata: Arc<ParquetMetaData>,
        written_size: usize,
        file_path: String,
        nan_value_counts: HashMap<i32, u64>,
        metrics_config: &MetricsConfig,
    ) -> Result<DataFileBuilder> {
        let index_by_parquet_path = {
            let mut visitor = IndexByParquetPathName::new();
            visit_schema(&schema, &mut visitor)?;
            visitor
        };

        let mut per_col_size: HashMap<i32, u64> = HashMap::new();
        let mut per_col_val_num: HashMap<i32, u64> = HashMap::new();
        let mut per_col_null_val_num: HashMap<i32, u64> = HashMap::new();
        let mut min_max_agg = MinMaxColAggregator::new(schema, metrics_config.clone());

        for row_group in metadata.row_groups() {
            for column_chunk_metadata in row_group.columns() {
                let parquet_path = column_chunk_metadata.column_descr().path().string();

                let Some(&field_id) = index_by_parquet_path.get(&parquet_path) else {
                    continue;
                };

                let mode = min_max_agg.mode_for(field_id);
                // MetricsMode::None — persist nothing for this column (Java skips it entirely).
                if !mode.collects_counts() {
                    continue;
                }

                *per_col_size.entry(field_id).or_insert(0) +=
                    column_chunk_metadata.compressed_size() as u64;
                *per_col_val_num.entry(field_id).or_insert(0) +=
                    column_chunk_metadata.num_values() as u64;

                if let Some(statistics) = column_chunk_metadata.statistics() {
                    if let Some(null_count) = statistics.null_count_opt() {
                        *per_col_null_val_num.entry(field_id).or_insert(0) += null_count;
                    }

                    // Bounds are only collected for Truncate/Full; produce() truncates them.
                    if mode.collects_bounds() {
                        min_max_agg.update(field_id, statistics.clone())?;
                    }
                }
            }
        }

        // Drop nan counts for columns whose mode persists nothing (`None`), mirroring the
        // skip-the-column behavior above.
        let mut nan_value_counts = nan_value_counts;
        nan_value_counts.retain(|field_id, _| min_max_agg.mode_for(*field_id).collects_counts());

        let column_sizes = per_col_size;
        let value_counts = per_col_val_num;
        let null_value_counts = per_col_null_val_num;
        let (lower_bounds, upper_bounds) = min_max_agg.produce();

        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(file_path)
            .file_format(DataFileFormat::Parquet)
            .partition(Struct::empty())
            .record_count(metadata.file_metadata().num_rows() as u64)
            .file_size_in_bytes(written_size as u64)
            .column_sizes(column_sizes)
            .value_counts(value_counts)
            .null_value_counts(null_value_counts)
            .nan_value_counts(nan_value_counts)
            // # NOTE:
            // - We can ignore implementing distinct_counts due to this: https://lists.apache.org/thread/j52tsojv0x4bopxyzsp7m7bqt23n5fnd
            .lower_bounds(lower_bounds)
            .upper_bounds(upper_bounds)
            .split_offsets(Some(
                metadata
                    .row_groups()
                    .iter()
                    .filter_map(|group| group.file_offset())
                    .collect(),
            ));

        Ok(builder)
    }

    #[allow(dead_code)]
    fn partition_value_from_bounds(
        table_spec: Arc<PartitionSpec>,
        lower_bounds: &HashMap<i32, Datum>,
        upper_bounds: &HashMap<i32, Datum>,
    ) -> Result<Struct> {
        let mut partition_literals: Vec<Option<Literal>> = Vec::new();

        for field in table_spec.fields() {
            if let (Some(lower), Some(upper)) = (
                lower_bounds.get(&field.source_id),
                upper_bounds.get(&field.source_id),
            ) {
                if !field.transform.preserves_order() {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "cannot infer partition value for non linear partition field (needs to preserve order): {} with transform {}",
                            field.name, field.transform
                        ),
                    ));
                }

                if lower != upper {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "multiple partition values for field {}: lower: {:?}, upper: {:?}",
                            field.name, lower, upper
                        ),
                    ));
                }

                let transform_fn = create_transform_function(&field.transform)?;
                let transform_literal =
                    Literal::from(transform_fn.transform_literal_result(lower)?);

                partition_literals.push(Some(transform_literal));
            } else {
                partition_literals.push(None);
            }
        }

        let partition_struct = Struct::from_iter(partition_literals);

        Ok(partition_struct)
    }
}

impl FileWriter for ParquetWriter {
    async fn write(&mut self, batch: &arrow_array::RecordBatch) -> Result<()> {
        // Skip empty batch
        if batch.num_rows() == 0 {
            return Ok(());
        }

        self.current_row_num += batch.num_rows();

        let batch_c = batch.clone();
        self.nan_value_count_visitor
            .compute(self.schema.clone(), batch_c)?;

        // Normalize UTC-alias timestamp timezones to the writer schema (metadata-only). Spark
        // tags Iceberg `timestamptz` batches `Timestamp(_, "UTC")`; the writer schema tags them
        // `Timestamp(_, "+00:00")` (UTC_TIME_ZONE), and the Parquet writer's strict schema check
        // is timezone-sensitive. The relabel reuses the values buffer (bit-identical instants);
        // genuinely different timezones and nested mismatches are left to fail loudly below.
        let normalized = normalize_utc_alias_timestamps(batch, &self.writer_arrow_schema)?;
        let batch = normalized.as_ref().unwrap_or(batch);

        // Lazy initialize the writer
        let writer = if let Some(writer) = &mut self.inner_writer {
            writer
        } else {
            let inner_writer = self.output_file.writer().await?;
            let async_writer = AsyncFileWriter::new(inner_writer);
            let writer = AsyncArrowWriter::try_new(
                async_writer,
                self.writer_arrow_schema.clone(),
                Some(self.writer_properties.clone()),
            )
            .map_err(|err| {
                Error::new(ErrorKind::Unexpected, "Failed to build parquet writer.")
                    .with_source(err)
            })?;
            self.inner_writer = Some(writer);
            self.inner_writer.as_mut().unwrap()
        };

        writer.write(batch).await.map_err(|err| {
            Error::new(
                ErrorKind::Unexpected,
                "Failed to write using parquet writer.",
            )
            .with_source(err)
        })?;

        Ok(())
    }

    async fn close(mut self) -> Result<Vec<DataFileBuilder>> {
        let mut writer = match self.inner_writer.take() {
            Some(writer) => writer,
            None => return Ok(vec![]),
        };

        let metadata = writer.finish().await.map_err(|err| {
            Error::new(ErrorKind::Unexpected, "Failed to finish parquet writer.").with_source(err)
        })?;

        let written_size = writer.bytes_written();

        if self.current_row_num == 0 {
            self.output_file.delete().await.map_err(|err| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to delete empty parquet file.",
                )
                .with_source(err)
            })?;
            Ok(vec![])
        } else {
            let parquet_metadata = Arc::new(metadata);

            Ok(vec![Self::parquet_to_data_file_builder(
                self.schema,
                parquet_metadata,
                written_size,
                self.output_file.location().to_string(),
                self.nan_value_count_visitor.nan_value_counts,
                &self.metrics_config,
            )?])
        }
    }
}

// ==============================================================================================
// UTC-alias timestamp normalization at the write funnel (F-A2-3)
//
// Spark-written parquet tags Iceberg `timestamptz` batches `Timestamp(_, "UTC")`. This crate's
// Arrow->Iceberg conversion accepts both `"UTC"` and `"+00:00"` as `timestamptz`, but its
// Iceberg->Arrow conversion canonicalizes to `"+00:00"` (UTC_TIME_ZONE) — so the writer schema
// tags timestamptz `"+00:00"` while a Spark batch tags it `"UTC"`. The Parquet `ArrowWriter`'s
// schema check (`types_compatible`) is timezone-sensitive for `Timestamp` and rejects the batch
// even though both aliases denote UTC and carry identical instants. Java Iceberg coerces write
// batches to the file schema; this normalizes the (closed) UTC-alias case metadata-only at the
// single write funnel, leaving every genuine mismatch to fail loud.
// ==============================================================================================

/// The timezone strings this crate treats as interchangeable UTC aliases — exactly the set the
/// Arrow->Iceberg conversion accepts as `timestamptz` (`arrow::schema`, the
/// `zone == "UTC" || zone == "+00:00"` arms). The set is CLOSED: any other timezone (e.g.
/// `"+05:00"`) is a genuinely different type and is never silently reinterpreted.
fn is_utc_alias(tz: &str) -> bool {
    tz == "UTC" || tz == UTC_TIME_ZONE
}

/// Relabel a timestamp array's timezone metadata to `tz`, reusing the values buffer.
///
/// `PrimitiveArray::with_timezone` rewrites only the `DataType` (`with_timezone_opt` rebuilds the
/// array `..self`, keeping the same values buffer), so the microsecond / nanosecond integers are
/// bit-identical. The caller guarantees the source and target units match, so no precision changes.
fn relabel_timestamp_timezone(col: &ArrayRef, unit: TimeUnit, tz: &str) -> Result<ArrayRef> {
    fn downcast<T: 'static>(col: &ArrayRef) -> Result<&T> {
        col.as_any().downcast_ref::<T>().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "timestamp column downcast failed during UTC-alias normalization",
            )
        })
    }
    Ok(match unit {
        TimeUnit::Second => Arc::new(
            downcast::<TimestampSecondArray>(col)?
                .clone()
                .with_timezone(tz),
        ),
        TimeUnit::Millisecond => Arc::new(
            downcast::<TimestampMillisecondArray>(col)?
                .clone()
                .with_timezone(tz),
        ),
        TimeUnit::Microsecond => Arc::new(
            downcast::<TimestampMicrosecondArray>(col)?
                .clone()
                .with_timezone(tz),
        ),
        TimeUnit::Nanosecond => Arc::new(
            downcast::<TimestampNanosecondArray>(col)?
                .clone()
                .with_timezone(tz),
        ),
    })
}

/// If `batch_dt` and `writer_dt` are both `Timestamp` of the SAME unit but differ by a UTC-alias
/// timezone string, return `col` relabeled to the writer's timezone (metadata-only). Otherwise
/// `None` — the column is passed through unchanged so any genuine mismatch (different timezone,
/// naive-vs-tz, nested timestamp, differing type) is rejected loudly by the Parquet writer.
fn utc_alias_relabel(
    col: &ArrayRef,
    batch_dt: &DataType,
    writer_dt: &DataType,
) -> Result<Option<ArrayRef>> {
    if let (
        DataType::Timestamp(batch_unit, Some(batch_tz)),
        DataType::Timestamp(writer_unit, Some(writer_tz)),
    ) = (batch_dt, writer_dt)
        && batch_unit == writer_unit
        && batch_tz.as_ref() != writer_tz.as_ref()
        && is_utc_alias(batch_tz.as_ref())
        && is_utc_alias(writer_tz.as_ref())
    {
        return Ok(Some(relabel_timestamp_timezone(
            col,
            *batch_unit,
            writer_tz.as_ref(),
        )?));
    }
    Ok(None)
}

/// Normalize a record batch whose schema differs from `writer_schema` ONLY by UTC-alias timezone
/// strings on TOP-LEVEL timestamp columns, via a metadata-only timezone relabel to the writer's
/// timezone. Returns `Some(batch)` when at least one column was relabeled, else `None` (the caller
/// writes the original batch unchanged — the common, zero-copy path taken by already-canonical
/// `"+00:00"` batches, naive timestamps, and non-timestamp schemas).
///
/// TOP-LEVEL only, by design: the Parquet writer applies its strict `types_compatible` check
/// positionally per (writer field, column) pair and recurses into nested types itself, so a
/// nested UTC-alias timestamp inside a struct/list is left unchanged here and still fails loud
/// (no silent partial normalization). Widening this to nested is a deliberate fork follow-up.
fn normalize_utc_alias_timestamps(
    batch: &RecordBatch,
    writer_schema: &ArrowSchema,
) -> Result<Option<RecordBatch>> {
    let batch_schema = batch.schema();
    // A differing column count is a genuine mismatch — let the writer reject it loudly.
    if batch_schema.fields().len() != writer_schema.fields().len() {
        return Ok(None);
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    let mut fields: Vec<Field> = Vec::with_capacity(batch.num_columns());
    let mut relabeled = false;

    for (idx, (batch_field, writer_field)) in batch_schema
        .fields()
        .iter()
        .zip(writer_schema.fields())
        .enumerate()
    {
        let column = batch.column(idx);
        match utc_alias_relabel(column, batch_field.data_type(), writer_field.data_type())? {
            Some(new_col) => {
                // Preserve the batch field's name / nullability / metadata; change only the type.
                let field = batch_field
                    .as_ref()
                    .clone()
                    .with_data_type(new_col.data_type().clone());
                columns.push(new_col);
                fields.push(field);
                relabeled = true;
            }
            None => {
                columns.push(column.clone());
                fields.push(batch_field.as_ref().clone());
            }
        }
    }

    if !relabeled {
        return Ok(None);
    }

    let normalized_schema = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        batch_schema.metadata().clone(),
    ));
    let normalized = RecordBatch::try_new(normalized_schema, columns).map_err(|err| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to build UTC-alias-normalized record batch.",
        )
        .with_source(err)
    })?;
    Ok(Some(normalized))
}

impl CurrentFileStatus for ParquetWriter {
    fn current_file_path(&self) -> String {
        self.output_file.location().to_string()
    }

    fn current_row_num(&self) -> usize {
        self.current_row_num
    }

    fn current_written_size(&self) -> usize {
        if let Some(inner) = self.inner_writer.as_ref() {
            // inner/AsyncArrowWriter contains sync and async writers
            // written size = bytes flushed to inner's async writer + bytes buffered in the inner's sync writer
            inner.bytes_written() + inner.in_progress_size()
        } else {
            // inner writer is not initialized yet
            0
        }
    }
}

/// AsyncFileWriter is a wrapper of FileWrite to make it compatible with tokio::io::AsyncWrite.
///
/// # NOTES
///
/// We keep this wrapper been used inside only.
struct AsyncFileWriter(Box<dyn FileWrite>);

impl AsyncFileWriter {
    /// Create a new `AsyncFileWriter` with the given writer.
    pub fn new(writer: Box<dyn FileWrite>) -> Self {
        Self(writer)
    }
}

impl ArrowAsyncFileWriter for AsyncFileWriter {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, parquet::errors::Result<()>> {
        Box::pin(async {
            self.0
                .write(bs)
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))
        })
    }

    fn complete(&mut self) -> BoxFuture<'_, parquet::errors::Result<()>> {
        Box::pin(async {
            self.0
                .close()
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use anyhow::Result;
    use arrow_array::builder::{Float32Builder, Int32Builder, MapBuilder};
    use arrow_array::types::{Float32Type, Int64Type};
    use arrow_array::{
        Array, ArrayRef, BooleanArray, Decimal128Array, Float32Array, Float64Array, Int32Array,
        Int64Array, ListArray, MapArray, RecordBatch, StructArray,
    };
    use arrow_schema::{DataType, Field, Fields, SchemaRef as ArrowSchemaRef};
    use arrow_select::concat::concat_batches;
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::file::statistics::ValueStatistics;
    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;
    use crate::arrow::schema_to_arrow_schema;
    use crate::io::FileIO;
    use crate::spec::decimal_utils::{decimal_mantissa, decimal_new, decimal_scale};
    use crate::spec::{PrimitiveLiteral, Struct, *};
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
    };
    use crate::writer::tests::check_parquet_data_file;

    fn schema_for_all_type() -> Schema {
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(0, "boolean", Type::Primitive(PrimitiveType::Boolean)).into(),
                NestedField::optional(1, "int", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "long", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(3, "float", Type::Primitive(PrimitiveType::Float)).into(),
                NestedField::optional(4, "double", Type::Primitive(PrimitiveType::Double)).into(),
                NestedField::optional(5, "string", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(6, "binary", Type::Primitive(PrimitiveType::Binary)).into(),
                NestedField::optional(7, "date", Type::Primitive(PrimitiveType::Date)).into(),
                NestedField::optional(8, "time", Type::Primitive(PrimitiveType::Time)).into(),
                NestedField::optional(9, "timestamp", Type::Primitive(PrimitiveType::Timestamp))
                    .into(),
                NestedField::optional(
                    10,
                    "timestamptz",
                    Type::Primitive(PrimitiveType::Timestamptz),
                )
                .into(),
                NestedField::optional(
                    11,
                    "timestamp_ns",
                    Type::Primitive(PrimitiveType::TimestampNs),
                )
                .into(),
                NestedField::optional(
                    12,
                    "timestamptz_ns",
                    Type::Primitive(PrimitiveType::TimestamptzNs),
                )
                .into(),
                NestedField::optional(
                    13,
                    "decimal",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 10,
                        scale: 5,
                    }),
                )
                .into(),
                NestedField::optional(14, "uuid", Type::Primitive(PrimitiveType::Uuid)).into(),
                NestedField::optional(15, "fixed", Type::Primitive(PrimitiveType::Fixed(10)))
                    .into(),
                // Parquet Statistics will use different representation for Decimal with precision 38 and scale 5,
                // so we need to add a new field for it.
                NestedField::optional(
                    16,
                    "decimal_38",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 38,
                        scale: 5,
                    }),
                )
                .into(),
            ])
            .build()
            .unwrap()
    }

    fn nested_schema_for_test() -> Schema {
        // Int, Struct(Int,Int), String, List(Int), Struct(Struct(Int)), Map(String, List(Int))
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(0, "col0", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(
                    1,
                    "col1",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(5, "col_1_5", Type::Primitive(PrimitiveType::Long))
                            .into(),
                        NestedField::required(6, "col_1_6", Type::Primitive(PrimitiveType::Long))
                            .into(),
                    ])),
                )
                .into(),
                NestedField::required(2, "col2", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(
                    3,
                    "col3",
                    Type::List(ListType::new(
                        NestedField::required(7, "element", Type::Primitive(PrimitiveType::Long))
                            .into(),
                    )),
                )
                .into(),
                NestedField::required(
                    4,
                    "col4",
                    Type::Struct(StructType::new(vec![
                        NestedField::required(
                            8,
                            "col_4_8",
                            Type::Struct(StructType::new(vec![
                                NestedField::required(
                                    9,
                                    "col_4_8_9",
                                    Type::Primitive(PrimitiveType::Long),
                                )
                                .into(),
                            ])),
                        )
                        .into(),
                    ])),
                )
                .into(),
                NestedField::required(
                    10,
                    "col5",
                    Type::Map(MapType::new(
                        NestedField::required(11, "key", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::required(
                            12,
                            "value",
                            Type::List(ListType::new(
                                NestedField::required(
                                    13,
                                    "item",
                                    Type::Primitive(PrimitiveType::Long),
                                )
                                .into(),
                            )),
                        )
                        .into(),
                    )),
                )
                .into(),
            ])
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_index_by_parquet_path() {
        let expect = HashMap::from([
            ("col0".to_string(), 0),
            ("col1.col_1_5".to_string(), 5),
            ("col1.col_1_6".to_string(), 6),
            ("col2".to_string(), 2),
            ("col3.list.element".to_string(), 7),
            ("col4.col_4_8.col_4_8_9".to_string(), 9),
            ("col5.key_value.key".to_string(), 11),
            ("col5.key_value.value.list.item".to_string(), 13),
        ]);
        let mut visitor = IndexByParquetPathName::new();
        visit_schema(&nested_schema_for_test(), &mut visitor).unwrap();
        assert_eq!(visitor.name_to_id, expect);
    }

    #[tokio::test]
    async fn test_parquet_writer() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // prepare data
        let schema = {
            let fields =
                vec![
                    Field::new("col", DataType::Int64, true).with_metadata(HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_string(),
                        "0".to_string(),
                    )])),
                ];
            Arc::new(arrow_schema::Schema::new(fields))
        };
        let col = Arc::new(Int64Array::from_iter_values(0..1024)) as ArrayRef;
        let null_col = Arc::new(Int64Array::new_null(1024)) as ArrayRef;
        let to_write = RecordBatch::try_new(schema.clone(), vec![col]).unwrap();
        let to_write_null = RecordBatch::try_new(schema.clone(), vec![null_col]).unwrap();

        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;

        // write data
        let mut pw = ParquetWriterBuilder::new(
            WriterProperties::builder()
                .set_max_row_group_size(128)
                .build(),
            Arc::new(to_write.schema().as_ref().try_into().unwrap()),
        )
        .build(output_file)
        .await?;
        pw.write(&to_write).await?;
        pw.write(&to_write_null).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            // Put dummy field for build successfully.
            .content(DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();

        // check data file
        assert_eq!(data_file.record_count(), 2048);
        assert_eq!(*data_file.value_counts(), HashMap::from([(0, 2048)]));
        assert_eq!(
            *data_file.lower_bounds(),
            HashMap::from([(0, Datum::long(0))])
        );
        assert_eq!(
            *data_file.upper_bounds(),
            HashMap::from([(0, Datum::long(1023))])
        );
        assert_eq!(*data_file.null_value_counts(), HashMap::from([(0, 1024)]));

        // check the written file
        let expect_batch = concat_batches(&schema, vec![&to_write, &to_write_null]).unwrap();
        check_parquet_data_file(&file_io, &data_file, &expect_batch).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_parquet_writer_with_complex_schema() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // prepare data
        let schema = nested_schema_for_test();
        let arrow_schema: ArrowSchemaRef = Arc::new((&schema).try_into().unwrap());
        let col0 = Arc::new(Int64Array::from_iter_values(0..1024)) as ArrayRef;
        let col1 = Arc::new(StructArray::new(
            {
                if let DataType::Struct(fields) = arrow_schema.field(1).data_type() {
                    fields.clone()
                } else {
                    unreachable!()
                }
            },
            vec![
                Arc::new(Int64Array::from_iter_values(0..1024)),
                Arc::new(Int64Array::from_iter_values(0..1024)),
            ],
            None,
        ));
        let col2 = Arc::new(arrow_array::StringArray::from_iter_values(
            (0..1024).map(|n| n.to_string()),
        )) as ArrayRef;
        let col3 = Arc::new({
            let list_parts = arrow_array::ListArray::from_iter_primitive::<Int64Type, _, _>(
                (0..1024).map(|n| Some(vec![Some(n)])),
            )
            .into_parts();
            arrow_array::ListArray::new(
                {
                    if let DataType::List(field) = arrow_schema.field(3).data_type() {
                        field.clone()
                    } else {
                        unreachable!()
                    }
                },
                list_parts.1,
                list_parts.2,
                list_parts.3,
            )
        }) as ArrayRef;
        let col4 = Arc::new(StructArray::new(
            {
                if let DataType::Struct(fields) = arrow_schema.field(4).data_type() {
                    fields.clone()
                } else {
                    unreachable!()
                }
            },
            vec![Arc::new(StructArray::new(
                {
                    if let DataType::Struct(fields) = arrow_schema.field(4).data_type() {
                        if let DataType::Struct(fields) = fields[0].data_type() {
                            fields.clone()
                        } else {
                            unreachable!()
                        }
                    } else {
                        unreachable!()
                    }
                },
                vec![Arc::new(Int64Array::from_iter_values(0..1024))],
                None,
            ))],
            None,
        ));
        let col5 = Arc::new({
            let mut map_array_builder = MapBuilder::new(
                None,
                arrow_array::builder::StringBuilder::new(),
                arrow_array::builder::ListBuilder::new(arrow_array::builder::PrimitiveBuilder::<
                    Int64Type,
                >::new()),
            );
            for i in 0..1024 {
                map_array_builder.keys().append_value(i.to_string());
                map_array_builder
                    .values()
                    .append_value(vec![Some(i as i64); i + 1]);
                map_array_builder.append(true)?;
            }
            let (_, offset_buffer, struct_array, null_buffer, ordered) =
                map_array_builder.finish().into_parts();
            let struct_array = {
                let (_, mut arrays, nulls) = struct_array.into_parts();
                let list_array = {
                    let list_array = arrays[1]
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .clone();
                    let (_, offsets, array, nulls) = list_array.into_parts();
                    let list_field = {
                        if let DataType::Map(map_field, _) = arrow_schema.field(5).data_type() {
                            if let DataType::Struct(fields) = map_field.data_type() {
                                if let DataType::List(list_field) = fields[1].data_type() {
                                    list_field.clone()
                                } else {
                                    unreachable!()
                                }
                            } else {
                                unreachable!()
                            }
                        } else {
                            unreachable!()
                        }
                    };
                    ListArray::new(list_field, offsets, array, nulls)
                };
                arrays[1] = Arc::new(list_array) as ArrayRef;
                StructArray::new(
                    {
                        if let DataType::Map(map_field, _) = arrow_schema.field(5).data_type() {
                            if let DataType::Struct(fields) = map_field.data_type() {
                                fields.clone()
                            } else {
                                unreachable!()
                            }
                        } else {
                            unreachable!()
                        }
                    },
                    arrays,
                    nulls,
                )
            };
            arrow_array::MapArray::new(
                {
                    if let DataType::Map(map_field, _) = arrow_schema.field(5).data_type() {
                        map_field.clone()
                    } else {
                        unreachable!()
                    }
                },
                offset_buffer,
                struct_array,
                null_buffer,
                ordered,
            )
        }) as ArrayRef;
        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![
            col0, col1, col2, col3, col4, col5,
        ])
        .unwrap();
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;

        // write data
        let mut pw =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), Arc::new(schema))
                .build(output_file)
                .await?;
        pw.write(&to_write).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            // Put dummy field for build successfully.
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();

        // check data file
        assert_eq!(data_file.record_count(), 1024);
        assert_eq!(
            *data_file.value_counts(),
            HashMap::from([
                (0, 1024),
                (5, 1024),
                (6, 1024),
                (2, 1024),
                (7, 1024),
                (9, 1024),
                (11, 1024),
                (13, (1..1025).sum()),
            ])
        );
        assert_eq!(
            *data_file.lower_bounds(),
            HashMap::from([
                (0, Datum::long(0)),
                (5, Datum::long(0)),
                (6, Datum::long(0)),
                (2, Datum::string("0")),
                (7, Datum::long(0)),
                (9, Datum::long(0)),
                (11, Datum::string("0")),
                (13, Datum::long(0))
            ])
        );
        assert_eq!(
            *data_file.upper_bounds(),
            HashMap::from([
                (0, Datum::long(1023)),
                (5, Datum::long(1023)),
                (6, Datum::long(1023)),
                (2, Datum::string("999")),
                (7, Datum::long(1023)),
                (9, Datum::long(1023)),
                (11, Datum::string("999")),
                (13, Datum::long(1023))
            ])
        );

        // check the written file
        check_parquet_data_file(&file_io, &data_file, &to_write).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_all_type_for_write() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // prepare data
        // generate iceberg schema for all type
        let schema = schema_for_all_type();
        let arrow_schema: ArrowSchemaRef = Arc::new((&schema).try_into().unwrap());
        let col0 = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
        ])) as ArrayRef;
        let col1 = Arc::new(Int32Array::from(vec![Some(1), Some(2), None, Some(4)])) as ArrayRef;
        let col2 = Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(4)])) as ArrayRef;
        let col3 = Arc::new(arrow_array::Float32Array::from(vec![
            Some(0.5),
            Some(2.0),
            None,
            Some(3.5),
        ])) as ArrayRef;
        let col4 = Arc::new(arrow_array::Float64Array::from(vec![
            Some(0.5),
            Some(2.0),
            None,
            Some(3.5),
        ])) as ArrayRef;
        let col5 = Arc::new(arrow_array::StringArray::from(vec![
            Some("a"),
            Some("b"),
            None,
            Some("d"),
        ])) as ArrayRef;
        let col6 = Arc::new(arrow_array::LargeBinaryArray::from_opt_vec(vec![
            Some(b"one"),
            None,
            Some(b""),
            Some(b"zzzz"),
        ])) as ArrayRef;
        let col7 = Arc::new(arrow_array::Date32Array::from(vec![
            Some(0),
            Some(1),
            None,
            Some(3),
        ])) as ArrayRef;
        let col8 = Arc::new(arrow_array::Time64MicrosecondArray::from(vec![
            Some(0),
            Some(1),
            None,
            Some(3),
        ])) as ArrayRef;
        let col9 = Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![
            Some(0),
            Some(1),
            None,
            Some(3),
        ])) as ArrayRef;
        let col10 = Arc::new(
            arrow_array::TimestampMicrosecondArray::from(vec![Some(0), Some(1), None, Some(3)])
                .with_timezone_utc(),
        ) as ArrayRef;
        let col11 = Arc::new(arrow_array::TimestampNanosecondArray::from(vec![
            Some(0),
            Some(1),
            None,
            Some(3),
        ])) as ArrayRef;
        let col12 = Arc::new(
            arrow_array::TimestampNanosecondArray::from(vec![Some(0), Some(1), None, Some(3)])
                .with_timezone_utc(),
        ) as ArrayRef;
        let col13 = Arc::new(
            arrow_array::Decimal128Array::from(vec![Some(1), Some(2), None, Some(100)])
                .with_precision_and_scale(10, 5)
                .unwrap(),
        ) as ArrayRef;
        let col14 = Arc::new(
            arrow_array::FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![
                    Some(Uuid::from_u128(0).as_bytes().to_vec()),
                    Some(Uuid::from_u128(1).as_bytes().to_vec()),
                    None,
                    Some(Uuid::from_u128(3).as_bytes().to_vec()),
                ]
                .into_iter(),
                16,
            )
            .unwrap(),
        ) as ArrayRef;
        let col15 = Arc::new(
            arrow_array::FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![
                    Some(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    Some(vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
                    None,
                    Some(vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
                ]
                .into_iter(),
                10,
            )
            .unwrap(),
        ) as ArrayRef;
        let col16 = Arc::new(
            arrow_array::Decimal128Array::from(vec![Some(1), Some(2), None, Some(100)])
                .with_precision_and_scale(38, 5)
                .unwrap(),
        ) as ArrayRef;
        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![
            col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13,
            col14, col15, col16,
        ])
        .unwrap();
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;

        // write data
        let mut pw =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), Arc::new(schema))
                .build(output_file)
                .await?;
        pw.write(&to_write).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            // Put dummy field for build successfully.
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();

        // check data file
        assert_eq!(data_file.record_count(), 4);
        assert!(data_file.value_counts().iter().all(|(_, &v)| { v == 4 }));
        assert!(
            data_file
                .null_value_counts()
                .iter()
                .all(|(_, &v)| { v == 1 })
        );
        assert_eq!(
            *data_file.lower_bounds(),
            HashMap::from([
                (0, Datum::bool(false)),
                (1, Datum::int(1)),
                (2, Datum::long(1)),
                (3, Datum::float(0.5)),
                (4, Datum::double(0.5)),
                (5, Datum::string("a")),
                (6, Datum::binary(vec![])),
                (7, Datum::date(0)),
                (8, Datum::time_micros(0).unwrap()),
                (9, Datum::timestamp_micros(0)),
                (10, Datum::timestamptz_micros(0)),
                (11, Datum::timestamp_nanos(0)),
                (12, Datum::timestamptz_nanos(0)),
                (
                    13,
                    Datum::new(
                        PrimitiveType::Decimal {
                            precision: 10,
                            scale: 5
                        },
                        PrimitiveLiteral::Int128(1)
                    )
                ),
                (14, Datum::uuid(Uuid::from_u128(0))),
                (15, Datum::fixed(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
                (
                    16,
                    Datum::new(
                        PrimitiveType::Decimal {
                            precision: 38,
                            scale: 5
                        },
                        PrimitiveLiteral::Int128(1)
                    )
                ),
            ])
        );
        assert_eq!(
            *data_file.upper_bounds(),
            HashMap::from([
                (0, Datum::bool(true)),
                (1, Datum::int(4)),
                (2, Datum::long(4)),
                (3, Datum::float(3.5)),
                (4, Datum::double(3.5)),
                (5, Datum::string("d")),
                (6, Datum::binary(vec![122, 122, 122, 122])),
                (7, Datum::date(3)),
                (8, Datum::time_micros(3).unwrap()),
                (9, Datum::timestamp_micros(3)),
                (10, Datum::timestamptz_micros(3)),
                (11, Datum::timestamp_nanos(3)),
                (12, Datum::timestamptz_nanos(3)),
                (
                    13,
                    Datum::new(
                        PrimitiveType::Decimal {
                            precision: 10,
                            scale: 5
                        },
                        PrimitiveLiteral::Int128(100)
                    )
                ),
                (14, Datum::uuid(Uuid::from_u128(3))),
                (
                    15,
                    Datum::fixed(vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
                ),
                (
                    16,
                    Datum::new(
                        PrimitiveType::Decimal {
                            precision: 38,
                            scale: 5
                        },
                        PrimitiveLiteral::Int128(100)
                    )
                ),
            ])
        );

        // check the written file
        check_parquet_data_file(&file_io, &data_file, &to_write).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_decimal_bound() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // test 1.1 and 2.2
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(
                        0,
                        "decimal",
                        Type::Primitive(PrimitiveType::Decimal {
                            precision: 28,
                            scale: 10,
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;
        let mut pw = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone())
            .build(output_file)
            .await?;
        let col0 = Arc::new(
            Decimal128Array::from(vec![Some(22000000000), Some(11000000000)])
                .with_data_type(DataType::Decimal128(28, 10)),
        ) as ArrayRef;
        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![col0]).unwrap();
        pw.write(&to_write).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();
        assert_eq!(
            data_file.upper_bounds().get(&0),
            Some(Datum::decimal_with_precision(decimal_new(22000000000_i64, 10), 28).unwrap())
                .as_ref()
        );
        assert_eq!(
            data_file.lower_bounds().get(&0),
            Some(Datum::decimal_with_precision(decimal_new(11000000000_i64, 10), 28).unwrap())
                .as_ref()
        );

        // test -1.1 and -2.2
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(
                        0,
                        "decimal",
                        Type::Primitive(PrimitiveType::Decimal {
                            precision: 28,
                            scale: 10,
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;
        let mut pw = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone())
            .build(output_file)
            .await?;
        let col0 = Arc::new(
            Decimal128Array::from(vec![Some(-22000000000), Some(-11000000000)])
                .with_data_type(DataType::Decimal128(28, 10)),
        ) as ArrayRef;
        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![col0]).unwrap();
        pw.write(&to_write).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();
        assert_eq!(
            data_file.upper_bounds().get(&0),
            Some(Datum::decimal_with_precision(decimal_new(-11000000000_i64, 10), 28).unwrap())
                .as_ref()
        );
        assert_eq!(
            data_file.lower_bounds().get(&0),
            Some(Datum::decimal_with_precision(decimal_new(-22000000000_i64, 10), 28).unwrap())
                .as_ref()
        );

        // test 38-digit precision decimal values (Iceberg spec max)
        // Note: fastnum D128::MAX/MIN have impractical exponents, so we use meaningful values
        use crate::spec::decimal_utils::decimal_from_str_exact;
        let decimal_max = decimal_from_str_exact("99999999999999999999999999999999999999").unwrap();
        let decimal_min =
            decimal_from_str_exact("-99999999999999999999999999999999999999").unwrap();
        assert_eq!(decimal_scale(&decimal_max), decimal_scale(&decimal_min));
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(
                        0,
                        "decimal",
                        Type::Primitive(PrimitiveType::Decimal {
                            precision: 38,
                            scale: decimal_scale(&decimal_max),
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;
        let mut pw = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema)
            .build(output_file)
            .await?;
        let col0 = Arc::new(
            Decimal128Array::from(vec![
                Some(decimal_mantissa(&decimal_max)),
                Some(decimal_mantissa(&decimal_min)),
            ])
            .with_data_type(DataType::Decimal128(38, 0)),
        ) as ArrayRef;
        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![col0]).unwrap();
        pw.write(&to_write).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();
        assert_eq!(
            data_file.upper_bounds().get(&0),
            Some(Datum::decimal(decimal_max).unwrap()).as_ref()
        );
        assert_eq!(
            data_file.lower_bounds().get(&0),
            Some(Datum::decimal(decimal_min).unwrap()).as_ref()
        );

        // test max and min for scale 38
        // # TODO
        // Readd this case after resolve https://github.com/apache/iceberg-rust/issues/669
        // let schema = Arc::new(
        //     Schema::builder()
        //         .with_fields(vec![NestedField::optional(
        //             0,
        //             "decimal",
        //             Type::Primitive(PrimitiveType::Decimal {
        //                 precision: 38,
        //                 scale: 0,
        //             }),
        //         )
        //         .into()])
        //         .build()
        //         .unwrap(),
        // );
        // let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        // let mut pw = ParquetWriterBuilder::new(
        //     WriterProperties::builder().build(),
        //     schema,
        //     file_io.clone(),
        //     loccation_gen,
        //     file_name_gen,
        // )
        // .build()
        // .await?;
        // let col0 = Arc::new(
        //     Decimal128Array::from(vec![
        //         Some(99999999999999999999999999999999999999_i128),
        //         Some(-99999999999999999999999999999999999999_i128),
        //     ])
        //     .with_data_type(DataType::Decimal128(38, 0)),
        // ) as ArrayRef;
        // let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![col0]).unwrap();
        // pw.write(&to_write).await?;
        // let res = pw.close().await?;
        // assert_eq!(res.len(), 1);
        // let data_file = res
        //     .into_iter()
        //     .next()
        //     .unwrap()
        //     .content(crate::spec::DataContentType::Data)
        //     .partition(Struct::empty())
        //     .build()
        //     .unwrap();
        // assert_eq!(
        //     data_file.upper_bounds().get(&0),
        //     Some(Datum::new(
        //         PrimitiveType::Decimal {
        //             precision: 38,
        //             scale: 0
        //         },
        //         PrimitiveLiteral::Int128(99999999999999999999999999999999999999_i128)
        //     ))
        //     .as_ref()
        // );
        // assert_eq!(
        //     data_file.lower_bounds().get(&0),
        //     Some(Datum::new(
        //         PrimitiveType::Decimal {
        //             precision: 38,
        //             scale: 0
        //         },
        //         PrimitiveLiteral::Int128(-99999999999999999999999999999999999999_i128)
        //     ))
        //     .as_ref()
        // );

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_write() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // Test that file will create if data to write
        let schema = {
            let fields = vec![
                arrow_schema::Field::new("col", arrow_schema::DataType::Int64, true).with_metadata(
                    HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "0".to_string())]),
                ),
            ];
            Arc::new(arrow_schema::Schema::new(fields))
        };
        let col = Arc::new(Int64Array::from_iter_values(0..1024)) as ArrayRef;
        let to_write = RecordBatch::try_new(schema.clone(), vec![col]).unwrap();
        let file_path = location_gen.generate_location(None, &file_name_gen.generate_file_name());
        let output_file = file_io.new_output(&file_path)?;
        let mut pw = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(to_write.schema().as_ref().try_into().unwrap()),
        )
        .build(output_file)
        .await?;
        pw.write(&to_write).await?;
        pw.close().await.unwrap();
        assert!(file_io.exists(&file_path).await.unwrap());

        // Test that file will not create if no data to write
        let file_name_gen =
            DefaultFileNameGenerator::new("test_empty".to_string(), None, DataFileFormat::Parquet);
        let file_path = location_gen.generate_location(None, &file_name_gen.generate_file_name());
        let output_file = file_io.new_output(&file_path)?;
        let pw = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(to_write.schema().as_ref().try_into().unwrap()),
        )
        .build(output_file)
        .await?;
        pw.close().await.unwrap();
        assert!(!file_io.exists(&file_path).await.unwrap());

        Ok(())
    }

    #[tokio::test]
    async fn test_nan_val_cnts_primitive_type() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // prepare data
        let arrow_schema = {
            let fields = vec![
                Field::new("col", arrow_schema::DataType::Float32, false).with_metadata(
                    HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "0".to_string())]),
                ),
                Field::new("col2", arrow_schema::DataType::Float64, false).with_metadata(
                    HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
                ),
            ];
            Arc::new(arrow_schema::Schema::new(fields))
        };

        let float_32_col = Arc::new(Float32Array::from_iter_values_with_nulls(
            [1.0_f32, f32::NAN, 2.0, 2.0].into_iter(),
            None,
        )) as ArrayRef;

        let float_64_col = Arc::new(Float64Array::from_iter_values_with_nulls(
            [1.0_f64, f64::NAN, 2.0, 2.0].into_iter(),
            None,
        )) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![float_32_col, float_64_col]).unwrap();
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;

        // write data
        let mut pw = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(to_write.schema().as_ref().try_into().unwrap()),
        )
        .build(output_file)
        .await?;

        pw.write(&to_write).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            // Put dummy field for build successfully.
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();

        // check data file
        assert_eq!(data_file.record_count(), 4);
        assert_eq!(*data_file.value_counts(), HashMap::from([(0, 4), (1, 4)]));
        assert_eq!(
            *data_file.lower_bounds(),
            HashMap::from([(0, Datum::float(1.0)), (1, Datum::double(1.0)),])
        );
        assert_eq!(
            *data_file.upper_bounds(),
            HashMap::from([(0, Datum::float(2.0)), (1, Datum::double(2.0)),])
        );
        assert_eq!(
            *data_file.null_value_counts(),
            HashMap::from([(0, 0), (1, 0)])
        );
        assert_eq!(
            *data_file.nan_value_counts(),
            HashMap::from([(0, 1), (1, 1)])
        );

        // check the written file
        let expect_batch = concat_batches(&arrow_schema, vec![&to_write]).unwrap();
        check_parquet_data_file(&file_io, &data_file, &expect_batch).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_nan_val_cnts_struct_type() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let schema_struct_float_fields = Fields::from(vec![
            Field::new("col4", DataType::Float32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "4".to_string(),
            )])),
        ]);

        let schema_struct_nested_float_fields = Fields::from(vec![
            Field::new("col7", DataType::Float32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "7".to_string(),
            )])),
        ]);

        let schema_struct_nested_fields = Fields::from(vec![
            Field::new(
                "col6",
                arrow_schema::DataType::Struct(schema_struct_nested_float_fields.clone()),
                false,
            )
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "6".to_string(),
            )])),
        ]);

        // prepare data
        let arrow_schema = {
            let fields = vec![
                Field::new(
                    "col3",
                    arrow_schema::DataType::Struct(schema_struct_float_fields.clone()),
                    false,
                )
                .with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "3".to_string(),
                )])),
                Field::new(
                    "col5",
                    arrow_schema::DataType::Struct(schema_struct_nested_fields.clone()),
                    false,
                )
                .with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "5".to_string(),
                )])),
            ];
            Arc::new(arrow_schema::Schema::new(fields))
        };

        let float_32_col = Arc::new(Float32Array::from_iter_values_with_nulls(
            [1.0_f32, f32::NAN, 2.0, 2.0].into_iter(),
            None,
        )) as ArrayRef;

        let struct_float_field_col = Arc::new(StructArray::new(
            schema_struct_float_fields,
            vec![float_32_col.clone()],
            None,
        )) as ArrayRef;

        let struct_nested_float_field_col = Arc::new(StructArray::new(
            schema_struct_nested_fields,
            vec![Arc::new(StructArray::new(
                schema_struct_nested_float_fields,
                vec![float_32_col.clone()],
                None,
            )) as ArrayRef],
            None,
        )) as ArrayRef;

        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![
            struct_float_field_col,
            struct_nested_float_field_col,
        ])
        .unwrap();
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;

        // write data
        let mut pw = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(to_write.schema().as_ref().try_into().unwrap()),
        )
        .build(output_file)
        .await?;

        pw.write(&to_write).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            // Put dummy field for build successfully.
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();

        // check data file
        assert_eq!(data_file.record_count(), 4);
        assert_eq!(*data_file.value_counts(), HashMap::from([(4, 4), (7, 4)]));
        assert_eq!(
            *data_file.lower_bounds(),
            HashMap::from([(4, Datum::float(1.0)), (7, Datum::float(1.0)),])
        );
        assert_eq!(
            *data_file.upper_bounds(),
            HashMap::from([(4, Datum::float(2.0)), (7, Datum::float(2.0)),])
        );
        assert_eq!(
            *data_file.null_value_counts(),
            HashMap::from([(4, 0), (7, 0)])
        );
        assert_eq!(
            *data_file.nan_value_counts(),
            HashMap::from([(4, 1), (7, 1)])
        );

        // check the written file
        let expect_batch = concat_batches(&arrow_schema, vec![&to_write]).unwrap();
        check_parquet_data_file(&file_io, &data_file, &expect_batch).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_nan_val_cnts_list_type() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let schema_list_float_field = Field::new("element", DataType::Float32, true).with_metadata(
            HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
        );

        let schema_struct_list_float_field = Field::new("element", DataType::Float32, true)
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "4".to_string(),
            )]));

        let schema_struct_list_field = Fields::from(vec![
            Field::new_list("col2", schema_struct_list_float_field.clone(), true).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "3".to_string())]),
            ),
        ]);

        let arrow_schema = {
            let fields = vec![
                Field::new_list("col0", schema_list_float_field.clone(), true).with_metadata(
                    HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "0".to_string())]),
                ),
                Field::new_struct("col1", schema_struct_list_field.clone(), true)
                    .with_metadata(HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_string(),
                        "2".to_string(),
                    )]))
                    .clone(),
                // Field::new_large_list("col3", schema_large_list_float_field.clone(), true).with_metadata(
                //     HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "5".to_string())]),
                // ).clone(),
            ];
            Arc::new(arrow_schema::Schema::new(fields))
        };

        let list_parts = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(vec![
            Some(1.0_f32),
            Some(f32::NAN),
            Some(2.0),
            Some(2.0),
        ])])
        .into_parts();

        let list_float_field_col = Arc::new({
            let list_parts = list_parts.clone();
            ListArray::new(
                {
                    if let DataType::List(field) = arrow_schema.field(0).data_type() {
                        field.clone()
                    } else {
                        unreachable!()
                    }
                },
                list_parts.1,
                list_parts.2,
                list_parts.3,
            )
        }) as ArrayRef;

        let struct_list_fields_schema =
            if let DataType::Struct(fields) = arrow_schema.field(1).data_type() {
                fields.clone()
            } else {
                unreachable!()
            };

        let struct_list_float_field_col = Arc::new({
            ListArray::new(
                {
                    if let DataType::List(field) = struct_list_fields_schema
                        .first()
                        .expect("could not find first list field")
                        .data_type()
                    {
                        field.clone()
                    } else {
                        unreachable!()
                    }
                },
                list_parts.1,
                list_parts.2,
                list_parts.3,
            )
        }) as ArrayRef;

        let struct_list_float_field_col = Arc::new(StructArray::new(
            struct_list_fields_schema,
            vec![struct_list_float_field_col.clone()],
            None,
        )) as ArrayRef;

        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![
            list_float_field_col,
            struct_list_float_field_col,
            // large_list_float_field_col,
        ])
        .expect("Could not form record batch");
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;

        // write data
        let mut pw = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(
                to_write
                    .schema()
                    .as_ref()
                    .try_into()
                    .expect("Could not convert iceberg schema"),
            ),
        )
        .build(output_file)
        .await?;

        pw.write(&to_write).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();

        // check data file
        assert_eq!(data_file.record_count(), 1);
        assert_eq!(*data_file.value_counts(), HashMap::from([(1, 4), (4, 4)]));
        assert_eq!(
            *data_file.lower_bounds(),
            HashMap::from([(1, Datum::float(1.0)), (4, Datum::float(1.0))])
        );
        assert_eq!(
            *data_file.upper_bounds(),
            HashMap::from([(1, Datum::float(2.0)), (4, Datum::float(2.0))])
        );
        assert_eq!(
            *data_file.null_value_counts(),
            HashMap::from([(1, 0), (4, 0)])
        );
        assert_eq!(
            *data_file.nan_value_counts(),
            HashMap::from([(1, 1), (4, 1)])
        );

        // check the written file
        let expect_batch = concat_batches(&arrow_schema, vec![&to_write]).unwrap();
        check_parquet_data_file(&file_io, &data_file, &expect_batch).await;

        Ok(())
    }

    macro_rules! construct_map_arr {
        ($map_key_field_schema:ident, $map_value_field_schema:ident) => {{
            let int_builder = Int32Builder::new();
            let float_builder = Float32Builder::with_capacity(4);
            let mut builder = MapBuilder::new(None, int_builder, float_builder);
            builder.keys().append_value(1);
            builder.values().append_value(1.0_f32);
            builder.append(true).unwrap();
            builder.keys().append_value(2);
            builder.values().append_value(f32::NAN);
            builder.append(true).unwrap();
            builder.keys().append_value(3);
            builder.values().append_value(2.0);
            builder.append(true).unwrap();
            builder.keys().append_value(4);
            builder.values().append_value(2.0);
            builder.append(true).unwrap();
            let array = builder.finish();

            let (_field, offsets, entries, nulls, ordered) = array.into_parts();
            let new_struct_fields_schema =
                Fields::from(vec![$map_key_field_schema, $map_value_field_schema]);

            let entries = {
                let (_, arrays, nulls) = entries.into_parts();
                StructArray::new(new_struct_fields_schema.clone(), arrays, nulls)
            };

            let field = Arc::new(Field::new(
                DEFAULT_MAP_FIELD_NAME,
                DataType::Struct(new_struct_fields_schema),
                false,
            ));

            Arc::new(MapArray::new(field, offsets, entries, nulls, ordered))
        }};
    }

    #[tokio::test]
    async fn test_nan_val_cnts_map_type() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let map_key_field_schema =
            Field::new(MAP_KEY_FIELD_NAME, DataType::Int32, false).with_metadata(HashMap::from([
                (PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string()),
            ]));

        let map_value_field_schema =
            Field::new(MAP_VALUE_FIELD_NAME, DataType::Float32, true).with_metadata(HashMap::from(
                [(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())],
            ));

        let struct_map_key_field_schema =
            Field::new(MAP_KEY_FIELD_NAME, DataType::Int32, false).with_metadata(HashMap::from([
                (PARQUET_FIELD_ID_META_KEY.to_string(), "6".to_string()),
            ]));

        let struct_map_value_field_schema =
            Field::new(MAP_VALUE_FIELD_NAME, DataType::Float32, true).with_metadata(HashMap::from(
                [(PARQUET_FIELD_ID_META_KEY.to_string(), "7".to_string())],
            ));

        let schema_struct_map_field = Fields::from(vec![
            Field::new_map(
                "col3",
                DEFAULT_MAP_FIELD_NAME,
                struct_map_key_field_schema.clone(),
                struct_map_value_field_schema.clone(),
                false,
                false,
            )
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "5".to_string(),
            )])),
        ]);

        let arrow_schema = {
            let fields = vec![
                Field::new_map(
                    "col0",
                    DEFAULT_MAP_FIELD_NAME,
                    map_key_field_schema.clone(),
                    map_value_field_schema.clone(),
                    false,
                    false,
                )
                .with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "0".to_string(),
                )])),
                Field::new_struct("col1", schema_struct_map_field.clone(), true)
                    .with_metadata(HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_string(),
                        "3".to_string(),
                    )]))
                    .clone(),
            ];
            Arc::new(arrow_schema::Schema::new(fields))
        };

        let map_array = construct_map_arr!(map_key_field_schema, map_value_field_schema);

        let struct_map_arr =
            construct_map_arr!(struct_map_key_field_schema, struct_map_value_field_schema);

        let struct_list_float_field_col = Arc::new(StructArray::new(
            schema_struct_map_field,
            vec![struct_map_arr],
            None,
        )) as ArrayRef;

        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![
            map_array,
            struct_list_float_field_col,
        ])
        .expect("Could not form record batch");
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;

        // write data
        let mut pw = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(
                to_write
                    .schema()
                    .as_ref()
                    .try_into()
                    .expect("Could not convert iceberg schema"),
            ),
        )
        .build(output_file)
        .await?;

        pw.write(&to_write).await?;
        let res = pw.close().await?;
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();

        // check data file
        assert_eq!(data_file.record_count(), 4);
        assert_eq!(
            *data_file.value_counts(),
            HashMap::from([(1, 4), (2, 4), (6, 4), (7, 4)])
        );
        assert_eq!(
            *data_file.lower_bounds(),
            HashMap::from([
                (1, Datum::int(1)),
                (2, Datum::float(1.0)),
                (6, Datum::int(1)),
                (7, Datum::float(1.0))
            ])
        );
        assert_eq!(
            *data_file.upper_bounds(),
            HashMap::from([
                (1, Datum::int(4)),
                (2, Datum::float(2.0)),
                (6, Datum::int(4)),
                (7, Datum::float(2.0))
            ])
        );
        assert_eq!(
            *data_file.null_value_counts(),
            HashMap::from([(1, 0), (2, 0), (6, 0), (7, 0)])
        );
        assert_eq!(
            *data_file.nan_value_counts(),
            HashMap::from([(2, 1), (7, 1)])
        );

        // check the written file
        let expect_batch = concat_batches(&arrow_schema, vec![&to_write]).unwrap();
        check_parquet_data_file(&file_io, &data_file, &expect_batch).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_write_empty_parquet_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);
        let output_file = file_io
            .new_output(location_gen.generate_location(None, &file_name_gen.generate_file_name()))
            .unwrap();

        // write data
        let pw = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(
                Schema::builder()
                    .with_schema_id(1)
                    .with_fields(vec![
                        NestedField::required(0, "col", Type::Primitive(PrimitiveType::Long))
                            .with_id(0)
                            .into(),
                    ])
                    .build()
                    .expect("Failed to create schema"),
            ),
        )
        .build(output_file)
        .await
        .unwrap();

        let res = pw.close().await.unwrap();
        assert_eq!(res.len(), 0);

        // Check that file should have been deleted.
        assert_eq!(std::fs::read_dir(temp_dir.path()).unwrap().count(), 0);
    }

    /// Write a single `string` column (field id 0) under `metrics_config` and return the resulting
    /// `DataFile`, so a test can inspect which metrics were persisted.
    async fn write_single_string_column(
        values: Vec<&str>,
        metrics_config: MetricsConfig,
    ) -> DataFile {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::optional(0, "s", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema: ArrowSchemaRef = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let output_file = file_io
            .new_output(location_gen.generate_location(None, &file_name_gen.generate_file_name()))
            .unwrap();
        let mut pw = ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone())
            .with_metrics_config(metrics_config)
            .build(output_file)
            .await
            .unwrap();
        let col0 = Arc::new(arrow_array::StringArray::from(values)) as ArrayRef;
        let to_write = RecordBatch::try_new(arrow_schema, vec![col0]).unwrap();
        pw.write(&to_write).await.unwrap();
        let res = pw.close().await.unwrap();
        assert_eq!(res.len(), 1);
        res.into_iter()
            .next()
            .unwrap()
            .content(crate::spec::DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_string_bounds_truncated_by_default() {
        // Two 20-code-point strings; the default config (`truncate(16)`, == Java
        // `MetricsConfig.getDefault()`) truncates the string bounds to 16.
        let min = "a".repeat(20);
        let max = "b".repeat(20);
        let data_file =
            write_single_string_column(vec![min.as_str(), max.as_str()], MetricsConfig::default())
                .await;
        // Lower bound = first 16 'a' (truncateStringMin); upper bound = 15 'b' + 'c'
        // (truncateStringMax increments the last retained code point so it stays an upper bound).
        assert_eq!(
            data_file.lower_bounds().get(&0),
            Some(Datum::string("a".repeat(16))).as_ref()
        );
        assert_eq!(
            data_file.upper_bounds().get(&0),
            Some(Datum::string(format!("{}c", "b".repeat(15)))).as_ref()
        );
    }

    #[tokio::test]
    async fn test_metrics_modes_full_counts_none() {
        let min = "a".repeat(20);
        let max = "b".repeat(20);
        let mode_config = |mode: &str| {
            MetricsConfig::from_properties(&HashMap::from([(
                "write.metadata.metrics.default".to_string(),
                mode.to_string(),
            )]))
        };

        // Full → untruncated bounds, counts present.
        let df =
            write_single_string_column(vec![min.as_str(), max.as_str()], mode_config("full")).await;
        assert_eq!(
            df.lower_bounds().get(&0),
            Some(Datum::string(&min)).as_ref()
        );
        assert_eq!(
            df.upper_bounds().get(&0),
            Some(Datum::string(&max)).as_ref()
        );
        assert_eq!(df.value_counts().get(&0), Some(&2));

        // Counts → counts kept, NO bounds.
        let df =
            write_single_string_column(vec![min.as_str(), max.as_str()], mode_config("counts"))
                .await;
        assert!(df.lower_bounds().get(&0).is_none());
        assert!(df.upper_bounds().get(&0).is_none());
        assert_eq!(df.value_counts().get(&0), Some(&2));

        // None → nothing persisted for the column (no bounds, counts, or column sizes).
        let df =
            write_single_string_column(vec![min.as_str(), max.as_str()], mode_config("none")).await;
        assert!(df.lower_bounds().get(&0).is_none());
        assert!(df.upper_bounds().get(&0).is_none());
        assert!(df.value_counts().get(&0).is_none());
        assert!(df.column_sizes().get(&0).is_none());
    }

    #[test]
    fn test_min_max_aggregator() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(0, "col", Type::Primitive(PrimitiveType::Int))
                        .with_id(0)
                        .into(),
                ])
                .build()
                .expect("Failed to create schema"),
        );
        // Int columns are never truncated (Java truncates only string/binary), so the default
        // config does not affect this aggregator's bounds.
        let mut min_max_agg = MinMaxColAggregator::new(schema, MetricsConfig::default());
        let create_statistics =
            |min, max| Statistics::Int32(ValueStatistics::new(min, max, None, None, false));
        min_max_agg
            .update(0, create_statistics(None, Some(42)))
            .unwrap();
        min_max_agg
            .update(0, create_statistics(Some(0), Some(i32::MAX)))
            .unwrap();
        min_max_agg
            .update(0, create_statistics(Some(i32::MIN), None))
            .unwrap();
        min_max_agg
            .update(0, create_statistics(None, None))
            .unwrap();

        let (lower_bounds, upper_bounds) = min_max_agg.produce();

        assert_eq!(lower_bounds, HashMap::from([(0, Datum::int(i32::MIN))]));
        assert_eq!(upper_bounds, HashMap::from([(0, Datum::int(i32::MAX))]));
    }

    // ==========================================================================================
    // F-A2-3: timestamptz UTC-alias normalization at the parquet writer.
    //
    // Spark-written parquet tags Iceberg `timestamptz` batches `Timestamp(_, "UTC")`; this crate
    // canonicalizes `timestamptz` to `Timestamp(_, "+00:00")` (UTC_TIME_ZONE) on the
    // Iceberg->Arrow path, and the Parquet `ArrowWriter`'s strict `types_compatible` check is
    // timezone-sensitive for `Timestamp` — so a `"UTC"`-tagged batch is rejected against a
    // `"+00:00"` writer schema even though both are the SAME Iceberg type and the SAME instants.
    // The write funnel `ParquetWriter::write` normalizes UTC-alias mismatches metadata-only.
    // ==========================================================================================

    /// Read the single parquet data file back into one concatenated `RecordBatch`.
    async fn read_back_single_file(file_io: &FileIO, data_file: &DataFile) -> RecordBatch {
        let input_file = file_io.new_input(data_file.file_path.clone()).unwrap();
        let bytes = input_file.read().await.unwrap();
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .build()
            .unwrap();
        let batches = reader.map(|b| b.unwrap()).collect::<Vec<_>>();
        let schema = batches[0].schema();
        concat_batches(&schema, &batches).unwrap()
    }

    /// Build a `ParquetWriter` from `iceberg_schema`, write `batch`, and return the write result
    /// (surfacing a write-time error verbatim so the loud-rejection pins can assert on it).
    async fn write_batch_via_parquet(
        temp_dir: &TempDir,
        file_io: &FileIO,
        iceberg_schema: Schema,
        batch: &RecordBatch,
    ) -> Result<DataFile> {
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);
        let output_file = file_io.new_output(
            location_gen.generate_location(None, &file_name_gen.generate_file_name()),
        )?;
        let mut pw = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(iceberg_schema),
        )
        .build(output_file)
        .await?;
        pw.write(batch).await?;
        let res = pw.close().await?;
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .content(DataContentType::Data)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();
        Ok(data_file)
    }

    fn single_primitive_schema(field_id: i32, name: &str, ty: PrimitiveType) -> Schema {
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(field_id, name, Type::Primitive(ty)).into(),
            ])
            .build()
            .unwrap()
    }

    /// Entry pin (F-A2-3 repro): a `Timestamp(µs, "UTC")` batch written against a
    /// `timestamptz`-derived (`"+00:00"`) writer schema normalizes metadata-only and round-trips
    /// with BIT-IDENTICAL microsecond integers (compared as `i64`, value AND null mask — never
    /// display). Doubles as the load-bearing mutation proof.
    #[tokio::test]
    async fn test_write_utc_alias_timestamptz_normalizes_bit_identical() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();

        let schema = single_primitive_schema(3, "ingestion_timestamp", PrimitiveType::Timestamptz);

        // Same instants Spark would write, but tagged "UTC" (as Spark-written parquet reads back).
        let values = vec![
            Some(0_i64),
            Some(1_i64),
            None,
            Some(1_700_000_000_000_000_i64),
        ];
        let batch_schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new(
                "ingestion_timestamp",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
                true,
            )
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "3".to_string(),
            )])),
        ]));
        let col = Arc::new(
            arrow_array::TimestampMicrosecondArray::from(values.clone()).with_timezone("UTC"),
        ) as ArrayRef;
        let batch = RecordBatch::try_new(batch_schema, vec![col]).unwrap();

        let data_file = write_batch_via_parquet(&temp_dir, &file_io, schema, &batch).await?;

        let read = read_back_single_file(&file_io, &data_file).await;
        let read_col = read
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
            .expect("read-back column is TimestampMicrosecondArray");
        // Bit-identical microsecond integers + null positions.
        assert_eq!(read_col.iter().collect::<Vec<_>>(), values);
        // Normalized to the UTC alias the writer emits (canonical "+00:00").
        assert_eq!(
            read_col.data_type(),
            &DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("+00:00".into()))
        );
        Ok(())
    }

    /// Pin: the relabel is unit-generic — a `Timestamp(ns, "UTC")` batch against a
    /// `timestamptz_ns`-derived (`"+00:00"`) writer schema normalizes and round-trips
    /// bit-identical (nanosecond integers).
    #[tokio::test]
    async fn test_write_utc_alias_timestamptz_ns_normalizes_bit_identical() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let schema = single_primitive_schema(7, "ts_ns", PrimitiveType::TimestamptzNs);

        let values = vec![
            Some(0_i64),
            None,
            Some(-42_i64),
            Some(9_223_372_036_854_775_i64),
        ];
        let batch_schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new(
                "ts_ns",
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, Some("UTC".into())),
                true,
            )
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "7".to_string(),
            )])),
        ]));
        let col = Arc::new(
            arrow_array::TimestampNanosecondArray::from(values.clone()).with_timezone("UTC"),
        ) as ArrayRef;
        let batch = RecordBatch::try_new(batch_schema, vec![col]).unwrap();

        let data_file = write_batch_via_parquet(&temp_dir, &file_io, schema, &batch).await?;
        let read = read_back_single_file(&file_io, &data_file).await;
        let read_col = read
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::TimestampNanosecondArray>()
            .expect("read-back column is TimestampNanosecondArray");
        assert_eq!(read_col.iter().collect::<Vec<_>>(), values);
        assert_eq!(
            read_col.data_type(),
            &DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, Some("+00:00".into()))
        );
        Ok(())
    }

    /// Regression pin: an already-canonical `Timestamp(µs, "+00:00")` batch is written on the
    /// zero-copy pass-through path (no relabel) and round-trips bit-identical.
    #[tokio::test]
    async fn test_write_canonical_utc_timestamptz_unchanged() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let schema = single_primitive_schema(3, "ts", PrimitiveType::Timestamptz);

        let values = vec![Some(10_i64), Some(20_i64), None, Some(30_i64)];
        let batch_schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("+00:00".into())),
                true,
            )
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "3".to_string(),
            )])),
        ]));
        let col = Arc::new(
            arrow_array::TimestampMicrosecondArray::from(values.clone()).with_timezone("+00:00"),
        ) as ArrayRef;
        let batch = RecordBatch::try_new(batch_schema, vec![col]).unwrap();

        let data_file = write_batch_via_parquet(&temp_dir, &file_io, schema, &batch).await?;
        let read = read_back_single_file(&file_io, &data_file).await;
        let read_col = read
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
            .expect("read-back column is TimestampMicrosecondArray");
        assert_eq!(read_col.iter().collect::<Vec<_>>(), values);
        assert_eq!(
            read_col.data_type(),
            &DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("+00:00".into()))
        );
        Ok(())
    }

    /// Pin: a naive (no-timezone) `timestamp` column is untouched by normalization and still
    /// writes + round-trips — it stays naive (never handed a timezone).
    #[tokio::test]
    async fn test_write_naive_timestamp_untouched() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let schema = single_primitive_schema(3, "ts", PrimitiveType::Timestamp);

        let values = vec![Some(1_i64), Some(2_i64), None, Some(4_i64)];
        let batch_schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
                true,
            )
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "3".to_string(),
            )])),
        ]));
        let col =
            Arc::new(arrow_array::TimestampMicrosecondArray::from(values.clone())) as ArrayRef;
        let batch = RecordBatch::try_new(batch_schema, vec![col]).unwrap();

        let data_file = write_batch_via_parquet(&temp_dir, &file_io, schema, &batch).await?;
        let read = read_back_single_file(&file_io, &data_file).await;
        let read_col = read
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
            .expect("read-back column is TimestampMicrosecondArray");
        assert_eq!(read_col.iter().collect::<Vec<_>>(), values);
        assert_eq!(
            read_col.data_type(),
            &DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None)
        );
        Ok(())
    }

    /// Pin: the UTC-alias set is CLOSED — a genuinely different timezone (`"+05:00"`) is NOT
    /// coerced; the write is rejected LOUD (no generic timezone reinterpretation).
    #[tokio::test]
    async fn test_write_non_utc_timezone_rejected_loud() {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let schema = single_primitive_schema(3, "event_time", PrimitiveType::Timestamptz);

        let batch_schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("+05:00".into())),
                true,
            )
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "3".to_string(),
            )])),
        ]));
        let col = Arc::new(
            arrow_array::TimestampMicrosecondArray::from(vec![Some(0_i64), Some(1_i64)])
                .with_timezone("+05:00"),
        ) as ArrayRef;
        let batch = RecordBatch::try_new(batch_schema, vec![col]).unwrap();

        let err = write_batch_via_parquet(&temp_dir, &file_io, schema, &batch)
            .await
            .expect_err("a non-UTC timezone must be rejected, not silently reinterpreted");
        let rendered = format!("{err:?}");
        assert!(
            rendered.contains("Incompatible type") && rendered.contains("+05:00"),
            "expected a loud incompatible-type rejection naming the +05:00 timezone, got: {rendered}"
        );
    }

    /// Nested pin (top-level-only seam): a `"UTC"`-tagged `timestamptz` nested inside a struct is
    /// NOT normalized (normalization is top-level only) and the write is rejected LOUD at the
    /// nested leaf — no silent partial normalization.
    #[tokio::test]
    async fn test_write_nested_utc_timestamptz_rejected_loud() {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();

        // Iceberg: struct { ts: timestamptz }. The writer's Arrow child tz is "+00:00".
        let schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(
                    1,
                    "wrapper",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(2, "ts", Type::Primitive(PrimitiveType::Timestamptz))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();

        // Batch: struct { ts: Timestamp(µs, "UTC") } — the NESTED child mismatches.
        let child_field = Field::new(
            "ts",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            true,
        )
        .with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )]));
        let struct_fields = Fields::from(vec![child_field]);
        let wrapper_field =
            Field::new("wrapper", DataType::Struct(struct_fields.clone()), true).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())]),
            );
        let batch_schema = Arc::new(arrow_schema::Schema::new(vec![wrapper_field]));
        let ts_col = Arc::new(
            arrow_array::TimestampMicrosecondArray::from(vec![Some(0_i64), Some(1_i64)])
                .with_timezone("UTC"),
        ) as ArrayRef;
        let struct_col = Arc::new(StructArray::new(struct_fields, vec![ts_col], None)) as ArrayRef;
        let batch = RecordBatch::try_new(batch_schema, vec![struct_col]).unwrap();

        let err = write_batch_via_parquet(&temp_dir, &file_io, schema, &batch)
            .await
            .expect_err("a nested UTC-alias mismatch must fail loud (top-level-only seam)");
        // The writer rejects at the top-level struct field; crucially the rejected batch STILL
        // carries the nested "UTC" against the writer's "+00:00" (both timezone strings appear) —
        // proving normalization never touched the nested child (no silent partial normalization).
        let rendered = format!("{err:?}");
        assert!(
            rendered.contains("Incompatible type")
                && rendered.contains("UTC")
                && rendered.contains("+00:00"),
            "expected a loud incompatible-type rejection exposing the un-normalized nested \
             \"UTC\" vs \"+00:00\" timestamp mismatch, got: {rendered}"
        );
    }
}
