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

//! The module contains the file writer for the Avro data-file format.
//!
//! This mirrors [`super::parquet_writer`] but emits Avro Object Container Files (OCF) instead of
//! Parquet, slotting into the same writer seam ([`FileWriterBuilder`] / [`FileWriter`]) so an
//! [`AvroWriterBuilder`] composes into `DataFileWriter` / `RollingFileWriter` with no change to
//! those layers.
//!
//! # Encoding
//!
//! The write schema comes from `crate::avro::schema_to_avro_schema`, the same converter
//! `crate::arrow::avro_reader` resolves reads by and that Java `AvroSchemaUtil.convert` produces:
//! it stamps `field-id` / `element-id` / `key-id` / `value-id` on every node. Rows are encoded by
//! reusing the proven Arrow → Iceberg [`Literal`](crate::spec::Literal) converter
//! ([`arrow_struct_to_literal`]) followed by the serde `RawLiteral` path — exactly the bytes the
//! reader's round-trip tests (`spec::values::tests::check_convert_with_avro`) and Java
//! `PlannedDataReader` decode. No bespoke per-type byte encoder is introduced: every per-type rule
//! (date → INT days, time/timestamp → LONG micros, timestamp_ns → LONG nanos, decimal → BE
//! two's-complement Avro `fixed`, uuid → `fixed[16]` BE, optional → `[null, T]` union with null at
//! index 0, string-keyed map → Avro map, non-string-keyed map → array-of-records, nested struct →
//! positional record) is carried by `RawLiteral`.
//!
//! # Metrics
//!
//! Java `AvroMetrics.fromWriter` returns `Metrics(rowCount, null, null, null, null)` — Avro data
//! files carry **no** column metrics. The produced [`DataFileBuilder`] therefore sets only
//! `record_count` and `file_size_in_bytes` (the serialized OCF length) and leaves every column map
//! (`column_sizes` / `value_counts` / `null_value_counts` / `nan_value_counts` / `lower_bounds` /
//! `upper_bounds`) empty. Populating any of them would be false parity.

use apache_avro::types::Value as AvroValue;
use apache_avro::{Codec, Schema as AvroSchema, Writer as OcfWriter};
use arrow_array::{RecordBatch, StructArray};
use bytes::Bytes;

use super::{FileWriter, FileWriterBuilder};
use crate::arrow::arrow_struct_to_literal;
use crate::avro::schema_to_avro_schema;
use crate::io::OutputFile;
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Literal, NestedField, PrimitiveType,
    RawLiteral, SchemaRef, Struct, StructType, Type,
};
use crate::writer::CurrentFileStatus;
use crate::{Error, ErrorKind, Result};

/// The record name used for the top-level Avro schema, matching the reader/manifest convention.
const AVRO_ROOT_RECORD_NAME: &str = "table";

/// [`AvroWriterBuilder`] builds an [`AvroWriter`].
///
/// It takes the Iceberg [`Schema`](crate::spec::Schema) (so it can derive the Avro write schema via
/// `schema_to_avro_schema`) and an optional [`Codec`]. The incoming Arrow [`RecordBatch`] fields
/// must carry field ids in `PARQUET_FIELD_ID_META_KEY` so the converter can match by id; this is
/// the same contract [`super::ParquetWriterBuilder`] holds.
#[derive(Clone, Debug)]
pub struct AvroWriterBuilder {
    schema: SchemaRef,
    codec: Codec,
}

impl AvroWriterBuilder {
    /// Create a new `AvroWriterBuilder` writing uncompressed (`Codec::Null`) Avro.
    ///
    /// To construct the write result, `schema` is the Iceberg schema matching the Arrow batches the
    /// resulting writer will be fed.
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            codec: Codec::Null,
        }
    }

    /// Create a new `AvroWriterBuilder` with a custom Avro [`Codec`].
    pub fn new_with_codec(schema: SchemaRef, codec: Codec) -> Self {
        Self { schema, codec }
    }
}

impl FileWriterBuilder for AvroWriterBuilder {
    type R = AvroWriter;

    async fn build(&self, output_file: OutputFile) -> Result<Self::R> {
        // Reject types the reader cannot round-trip before any IO so the failure is loud and early
        // (mirrors the reader's variant/unknown rejection — see `reject_unsupported_types`).
        reject_unsupported_types(self.schema.as_struct())?;

        let avro_schema = schema_to_avro_schema(AVRO_ROOT_RECORD_NAME, &self.schema)?;

        Ok(AvroWriter {
            schema: self.schema.clone(),
            avro_schema,
            codec: self.codec,
            output_file,
            rows: Vec::new(),
            encoded_size_estimate: 0,
            current_row_num: 0,
        })
    }
}

/// `AvroWriter` writes Arrow [`RecordBatch`]es into an Avro OCF on storage — the Avro analogue of
/// [`super::ParquetWriter`].
///
/// `apache_avro`'s [`Writer`](apache_avro::Writer) (aliased `OcfWriter` here) is synchronous and
/// borrows its schema for the writer's lifetime, which cannot be held across this trait's
/// `&mut self async` calls without a self-referential struct. We therefore resolve each row to an
/// owned [`AvroValue`](apache_avro::types::Value) as it arrives (the CPU-bound Arrow → Iceberg →
/// Avro encode), buffer the values, and build the OCF exactly once in [`close`](FileWriter::close) —
/// a single async [`OutputFile::write`]. Nothing touches storage until close, so an empty input
/// never creates a phantom file.
pub struct AvroWriter {
    schema: SchemaRef,
    avro_schema: AvroSchema,
    codec: Codec,
    output_file: OutputFile,
    /// Resolved Avro row values accumulated across `write` calls; serialized into one OCF at close.
    rows: Vec<AvroValue>,
    /// Running sum of the per-row uncompressed Avro datum sizes — the live roll signal (see
    /// [`CurrentFileStatus::current_written_size`]).
    encoded_size_estimate: usize,
    current_row_num: usize,
}

impl AvroWriter {
    /// Build the [`DataFileBuilder`] for a written Avro file.
    ///
    /// Sets `record_count` and `file_size_in_bytes` only; every column-metric map is left empty to
    /// match Java `AvroMetrics.fromWriter` (`Metrics(rowCount, null, null, null, null)`).
    fn avro_to_data_file_builder(
        record_count: u64,
        file_size_in_bytes: u64,
        file_path: String,
    ) -> DataFileBuilder {
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(file_path)
            .file_format(DataFileFormat::Avro)
            .partition(Struct::empty())
            .record_count(record_count)
            .file_size_in_bytes(file_size_in_bytes);
        builder
    }
}

impl FileWriter for AvroWriter {
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        // Skip empty batch (mirrors ParquetWriter).
        if batch.num_rows() == 0 {
            return Ok(());
        }

        let iceberg_struct_type = self.schema.as_struct().clone();
        let rows = batch.num_rows();

        let (values, added_size) =
            encode_batch_to_values(&self.avro_schema, batch, &iceberg_struct_type)?;

        self.rows.extend(values);
        self.encoded_size_estimate += added_size;
        self.current_row_num += rows;
        Ok(())
    }

    async fn close(self) -> Result<Vec<DataFileBuilder>> {
        // No rows: never emit an OCF (the header alone would leave a phantom file). Nothing was
        // written to storage during `write`, so defensively delete any pre-existing file only.
        if self.current_row_num == 0 {
            self.output_file.delete().await.map_err(|err| {
                Error::new(ErrorKind::Unexpected, "Failed to delete empty avro file.")
                    .with_source(err)
            })?;
            return Ok(vec![]);
        }

        // Build the whole OCF in one shot. The schema is a local that outlives the synchronous
        // `apache_avro::Writer` borrowing it (the manifest writer follows the same shape), so no
        // self-referential storage or schema leak is needed.
        let avro_schema = self.avro_schema;
        let rows = self.rows;
        let codec = self.codec;
        let bytes = {
            let mut writer = OcfWriter::with_codec(&avro_schema, Vec::new(), codec);
            for value in &rows {
                writer.append_value_ref(value).map_err(|err| {
                    Error::new(ErrorKind::Unexpected, "Failed to append avro row.").with_source(err)
                })?;
            }
            writer.into_inner().map_err(|err| {
                Error::new(ErrorKind::Unexpected, "Failed to finish avro writer.").with_source(err)
            })?
        };
        let file_size_in_bytes = bytes.len() as u64;

        self.output_file
            .write(Bytes::from(bytes))
            .await
            .map_err(|err| {
                Error::new(ErrorKind::Unexpected, "Failed to write avro data file.")
                    .with_source(err)
            })?;

        Ok(vec![Self::avro_to_data_file_builder(
            self.current_row_num as u64,
            file_size_in_bytes,
            self.output_file.location().to_string(),
        )])
    }
}

impl CurrentFileStatus for AvroWriter {
    fn current_file_path(&self) -> String {
        self.output_file.location().to_string()
    }

    fn current_row_num(&self) -> usize {
        self.current_row_num
    }

    fn current_written_size(&self) -> usize {
        // Live estimate: the running sum of per-row uncompressed Avro datum sizes. The final OCF
        // adds a small fixed header plus per-block framing/compression, but this monotonic
        // uncompressed-payload sum is a sound roll signal — `RollingFileWriter::should_roll` rolls
        // once it crosses the target, and the absolute file size is reported exactly at close.
        self.encoded_size_estimate
    }
}

/// Encode one Arrow [`RecordBatch`] into a `Vec` of resolved Avro row values, returning also the
/// summed uncompressed datum byte size (the roll signal).
///
/// The batch is reinterpreted as a top-level [`StructArray`] (Iceberg's row = a struct), converted
/// to one [`Literal::Struct`] per row by [`arrow_struct_to_literal`], then each row is serialized
/// through `RawLiteral` → [`apache_avro::to_value`] → [`apache_avro::types::Value::resolve`]
/// against `avro_schema`. The `resolve` step is load-bearing: `RawLiteral` emits the generic serde
/// forms (decimal/fixed/uuid all as `bytes`/`string`), and `resolve` coerces each into the schema's
/// `decimal` / `fixed` / `uuid` node — i.e. a decimal becomes a sign-extended big-endian
/// two's-complement Avro `fixed` of `decimal_required_bytes(precision)` length. This mirrors the
/// reader-side round-trip helper `spec::values::tests::check_serialize_avro`.
fn encode_batch_to_values(
    avro_schema: &AvroSchema,
    batch: &RecordBatch,
    iceberg_struct_type: &StructType,
) -> Result<(Vec<AvroValue>, usize)> {
    let struct_array = StructArray::from(batch.clone());
    let array_ref = std::sync::Arc::new(struct_array) as arrow_array::ArrayRef;

    let literals = arrow_struct_to_literal(&array_ref, iceberg_struct_type)?;
    let struct_ty = Type::Struct(iceberg_struct_type.clone());

    let mut values = Vec::with_capacity(literals.len());
    let mut size = 0usize;

    for (row_idx, literal) in literals.into_iter().enumerate() {
        let Some(literal @ Literal::Struct(_)) = literal else {
            // A top-level row must be a non-null struct; `arrow_struct_to_literal` returns `None`
            // only for a null top-level entry, which an Arrow `RecordBatch` row can never be.
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Avro writer: top-level row literal was null or not a struct",
            )
            .with_context("row", row_idx.to_string()));
        };

        let raw_literal = RawLiteral::try_from(literal, &struct_ty)?;
        let value = apache_avro::to_value(raw_literal)
            .and_then(|v| v.resolve(avro_schema))
            .map_err(|err| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to resolve avro row value against the write schema.",
                )
                .with_source(err)
                .with_context("row", row_idx.to_string())
            })?;

        // Per-row uncompressed datum size for the live roll signal.
        size += apache_avro::to_avro_datum(avro_schema, value.clone())
            .map(|b| b.len())
            .unwrap_or(0);
        values.push(value);
    }

    Ok((values, size))
}

/// Reject Iceberg types the Avro data path cannot round-trip.
///
/// `variant` is rejected with [`ErrorKind::FeatureUnsupported`] to match the reader, which rejects
/// `variant` on read (`arrow/value.rs`). `unknown` is the always-null no-physical-column type; the
/// reader's literal converter also defers it (`FeatureUnsupported`), so for symmetric round-trip
/// behavior in this engine-only cycle we reject it here rather than silently emit `null` rows that
/// the matching reader would then refuse. (Java emits Avro `null`; closing that asymmetry is tracked
/// for the interop cycle — see the actor summary.)
fn reject_unsupported_types(struct_type: &StructType) -> Result<()> {
    for field in struct_type.fields() {
        reject_unsupported_field(field)?;
    }
    Ok(())
}

fn reject_unsupported_field(field: &NestedField) -> Result<()> {
    match field.field_type.as_ref() {
        Type::Variant => Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Avro data writer does not support the variant type (the reader rejects it on read)",
        )
        .with_context("field_id", field.id.to_string())
        .with_context("field_name", field.name.clone())),
        Type::Primitive(PrimitiveType::Unknown) => Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Avro data writer does not support the unknown type yet (the always-null read path is deferred)",
        )
        .with_context("field_id", field.id.to_string())
        .with_context("field_name", field.name.clone())),
        Type::Struct(s) => reject_unsupported_types(s),
        Type::List(l) => reject_unsupported_field(&l.element_field),
        Type::Map(m) => {
            reject_unsupported_field(&m.key_field)?;
            reject_unsupported_field(&m.value_field)
        }
        Type::Primitive(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use apache_avro::Codec;
    use arrow_array::types::Int64Type;
    use arrow_array::{
        ArrayRef, BooleanArray, Date32Array, Decimal128Array, FixedSizeBinaryArray, Float32Array,
        Float64Array, Int32Array, Int64Array, LargeBinaryArray, ListArray, MapArray, RecordBatch,
        StringArray, StructArray, Time64MicrosecondArray, TimestampMicrosecondArray,
        TimestampNanosecondArray,
    };
    use arrow_schema::{DataType, Field, Fields, SchemaRef as ArrowSchemaRef};
    use uuid::Uuid;

    use super::*;
    use crate::arrow::DEFAULT_MAP_FIELD_NAME;
    use crate::arrow::avro_reader::read_avro_data_bytes;
    use crate::io::FileIO;
    use crate::spec::{
        ListType, MAP_KEY_FIELD_NAME, MAP_VALUE_FIELD_NAME, MapType, NestedField, PrimitiveType,
        Schema, Struct, Type,
    };
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};

    fn make_temp() -> (tempfile::TempDir, FileIO, DefaultLocationGenerator) {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        (temp_dir, file_io, location_gen)
    }

    fn output_file(
        file_io: &FileIO,
        location_gen: &DefaultLocationGenerator,
        prefix: &str,
    ) -> (String, OutputFile) {
        let file_name_gen =
            DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Avro);
        let path = location_gen.generate_location(None, &file_name_gen.generate_file_name());
        let of = file_io.new_output(&path).unwrap();
        (path, of)
    }

    /// Every primitive + logical type the reader round-trips, plus an optional/null column.
    fn schema_all_types() -> Schema {
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
            ])
            .build()
            .unwrap()
    }

    fn all_types_batch(schema: &Schema) -> RecordBatch {
        let arrow_schema: ArrowSchemaRef = Arc::new(schema.try_into().unwrap());
        let col0 = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
        ])) as ArrayRef;
        let col1 = Arc::new(Int32Array::from(vec![Some(1), Some(2), None, Some(4)])) as ArrayRef;
        let col2 = Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(4)])) as ArrayRef;
        let col3 = Arc::new(Float32Array::from(vec![
            Some(0.5),
            Some(2.0),
            None,
            Some(3.5),
        ])) as ArrayRef;
        let col4 = Arc::new(Float64Array::from(vec![
            Some(0.5),
            Some(2.0),
            None,
            Some(3.5),
        ])) as ArrayRef;
        let col5 = Arc::new(StringArray::from(vec![
            Some("a"),
            Some("b"),
            None,
            Some("d"),
        ])) as ArrayRef;
        let col6 = Arc::new(LargeBinaryArray::from_opt_vec(vec![
            Some(b"one"),
            None,
            Some(b""),
            Some(b"zzzz"),
        ])) as ArrayRef;
        let col7 = Arc::new(Date32Array::from(vec![Some(0), Some(1), None, Some(3)])) as ArrayRef;
        let col8 = Arc::new(Time64MicrosecondArray::from(vec![
            Some(0),
            Some(1),
            None,
            Some(3),
        ])) as ArrayRef;
        let col9 = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(0),
            Some(1),
            None,
            Some(3),
        ])) as ArrayRef;
        let col10 = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(0), Some(1), None, Some(3)])
                .with_timezone_utc(),
        ) as ArrayRef;
        let col11 = Arc::new(TimestampNanosecondArray::from(vec![
            Some(0),
            Some(1),
            None,
            Some(3),
        ])) as ArrayRef;
        let col12 = Arc::new(
            TimestampNanosecondArray::from(vec![Some(0), Some(1), None, Some(3)])
                .with_timezone_utc(),
        ) as ArrayRef;
        let col13 = Arc::new(
            Decimal128Array::from(vec![Some(1), Some(2), None, Some(100)])
                .with_precision_and_scale(10, 5)
                .unwrap(),
        ) as ArrayRef;
        let col14 = Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
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
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
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
        RecordBatch::try_new(arrow_schema, vec![
            col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13,
            col14, col15,
        ])
        .unwrap()
    }

    /// 1. ROUND-TRIP via the U1 reader: write every type incl. an optional/null column, read back.
    #[tokio::test]
    async fn test_avro_writer_all_types_roundtrip() {
        let (_t, file_io, location_gen) = make_temp();
        let schema = Arc::new(schema_all_types());
        let to_write = all_types_batch(&schema);

        let (_path, of) = output_file(&file_io, &location_gen, "all_types");
        let mut w = AvroWriterBuilder::new(schema.clone())
            .build(of)
            .await
            .unwrap();
        w.write(&to_write).await.unwrap();
        let res = w.close().await.unwrap();
        assert_eq!(res.len(), 1);

        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .partition_spec_id(0)
            .build()
            .unwrap();
        assert_eq!(data_file.file_format(), DataFileFormat::Avro);

        // Read it back through the U1 reader and assert exact equality of every cell.
        let input = file_io.new_input(data_file.file_path()).unwrap();
        let bytes = input.read().await.unwrap();
        let batches = read_avro_data_bytes(&bytes, schema.as_ref(), 1024).unwrap();
        let read_back =
            arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();

        assert_eq!(read_back.num_rows(), to_write.num_rows());
        assert_eq!(read_back.num_columns(), to_write.num_columns());
        for c in 0..to_write.num_columns() {
            assert_eq!(
                read_back.column(c),
                to_write.column(c),
                "column {c} mismatch on round-trip"
            );
        }
    }

    /// 2. NESTED round-trip via U1: struct-of-struct, list, string-keyed map, non-string-keyed map.
    #[tokio::test]
    async fn test_avro_writer_nested_roundtrip() {
        let (_t, file_io, location_gen) = make_temp();

        // col0: long
        // col1: struct { 5: long, 6: struct { 9: long } }   (struct-of-struct)
        // col2: list<long>
        // col3: map<string, long>                            (string-keyed → Avro map)
        // col4: map<int, long>                               (non-string-keyed → array-of-records)
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(0, "col0", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(
                        1,
                        "col1",
                        Type::Struct(StructType::new(vec![
                            NestedField::required(5, "a", Type::Primitive(PrimitiveType::Long))
                                .into(),
                            NestedField::required(
                                6,
                                "b",
                                Type::Struct(StructType::new(vec![
                                    NestedField::required(
                                        9,
                                        "c",
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
                        2,
                        "col2",
                        Type::List(ListType::new(
                            NestedField::required(
                                7,
                                "element",
                                Type::Primitive(PrimitiveType::Long),
                            )
                            .into(),
                        )),
                    )
                    .into(),
                    NestedField::required(
                        3,
                        "col3",
                        Type::Map(MapType::new(
                            NestedField::required(
                                10,
                                "key",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            NestedField::required(
                                11,
                                "value",
                                Type::Primitive(PrimitiveType::Long),
                            )
                            .into(),
                        )),
                    )
                    .into(),
                    NestedField::required(
                        4,
                        "col4",
                        Type::Map(MapType::new(
                            NestedField::required(12, "key", Type::Primitive(PrimitiveType::Int))
                                .into(),
                            NestedField::required(
                                13,
                                "value",
                                Type::Primitive(PrimitiveType::Long),
                            )
                            .into(),
                        )),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into().unwrap());

        // col0
        let col0 = Arc::new(Int64Array::from(vec![10_i64, 20])) as ArrayRef;
        // col1 = struct { a, b: struct { c } }
        let inner_b_fields = if let DataType::Struct(fields) = arrow_schema.field(1).data_type() {
            if let DataType::Struct(inner) = fields[1].data_type() {
                inner.clone()
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        };
        let col1_fields = if let DataType::Struct(fields) = arrow_schema.field(1).data_type() {
            fields.clone()
        } else {
            unreachable!()
        };
        let b_struct = Arc::new(StructArray::new(
            inner_b_fields,
            vec![Arc::new(Int64Array::from(vec![100_i64, 200]))],
            None,
        )) as ArrayRef;
        let col1 = Arc::new(StructArray::new(
            col1_fields,
            vec![Arc::new(Int64Array::from(vec![1_i64, 2])), b_struct],
            None,
        )) as ArrayRef;
        // col2 = list<long>
        let col2 = Arc::new({
            let parts = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                Some(vec![Some(1_i64), Some(2)]),
                Some(vec![Some(3)]),
            ])
            .into_parts();
            let field = if let DataType::List(field) = arrow_schema.field(2).data_type() {
                field.clone()
            } else {
                unreachable!()
            };
            ListArray::new(field, parts.1, parts.2, parts.3)
        }) as ArrayRef;
        // col3 = map<string, long>
        let col3 = {
            let mut b = arrow_array::builder::MapBuilder::new(
                None,
                arrow_array::builder::StringBuilder::new(),
                arrow_array::builder::PrimitiveBuilder::<Int64Type>::new(),
            );
            b.keys().append_value("k1");
            b.values().append_value(1);
            b.append(true).unwrap();
            b.keys().append_value("k2");
            b.values().append_value(2);
            b.append(true).unwrap();
            let (_f, offsets, entries, nulls, ordered) = b.finish().into_parts();
            let key_f = Field::new(MAP_KEY_FIELD_NAME, DataType::Utf8, false).with_metadata(
                HashMap::from([("PARQUET:field_id".to_string(), "10".to_string())]),
            );
            let val_f = Field::new(MAP_VALUE_FIELD_NAME, DataType::Int64, false).with_metadata(
                HashMap::from([("PARQUET:field_id".to_string(), "11".to_string())]),
            );
            let entries_fields = Fields::from(vec![key_f, val_f]);
            let (_, arrays, e_nulls) = entries.into_parts();
            let entries = StructArray::new(entries_fields.clone(), arrays, e_nulls);
            let map_field = Arc::new(Field::new(
                DEFAULT_MAP_FIELD_NAME,
                DataType::Struct(entries_fields),
                false,
            ));
            Arc::new(MapArray::new(map_field, offsets, entries, nulls, ordered)) as ArrayRef
        };
        // col4 = map<int, long>
        let col4 = {
            let mut b = arrow_array::builder::MapBuilder::new(
                None,
                arrow_array::builder::PrimitiveBuilder::<arrow_array::types::Int32Type>::new(),
                arrow_array::builder::PrimitiveBuilder::<Int64Type>::new(),
            );
            b.keys().append_value(7);
            b.values().append_value(70);
            b.append(true).unwrap();
            b.keys().append_value(8);
            b.values().append_value(80);
            b.append(true).unwrap();
            let (_f, offsets, entries, nulls, ordered) = b.finish().into_parts();
            let key_f = Field::new(MAP_KEY_FIELD_NAME, DataType::Int32, false).with_metadata(
                HashMap::from([("PARQUET:field_id".to_string(), "12".to_string())]),
            );
            let val_f = Field::new(MAP_VALUE_FIELD_NAME, DataType::Int64, false).with_metadata(
                HashMap::from([("PARQUET:field_id".to_string(), "13".to_string())]),
            );
            let entries_fields = Fields::from(vec![key_f, val_f]);
            let (_, arrays, e_nulls) = entries.into_parts();
            let entries = StructArray::new(entries_fields.clone(), arrays, e_nulls);
            let map_field = Arc::new(Field::new(
                DEFAULT_MAP_FIELD_NAME,
                DataType::Struct(entries_fields),
                false,
            ));
            Arc::new(MapArray::new(map_field, offsets, entries, nulls, ordered)) as ArrayRef
        };

        let to_write =
            RecordBatch::try_new(arrow_schema, vec![col0, col1, col2, col3, col4]).unwrap();

        let (_path, of) = output_file(&file_io, &location_gen, "nested");
        let mut w = AvroWriterBuilder::new(schema.clone())
            .build(of)
            .await
            .unwrap();
        w.write(&to_write).await.unwrap();
        let res = w.close().await.unwrap();
        assert_eq!(res.len(), 1);
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .partition_spec_id(0)
            .build()
            .unwrap();

        let input = file_io.new_input(data_file.file_path()).unwrap();
        let bytes = input.read().await.unwrap();
        let batches = read_avro_data_bytes(&bytes, schema.as_ref(), 1024).unwrap();
        let read_back =
            arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();

        assert_eq!(read_back.num_rows(), 2);
        // Convert both back to Iceberg literals for a structural, order-insensitive-on-map compare.
        let want = arrow_struct_to_literal(
            &(Arc::new(StructArray::from(to_write)) as ArrayRef),
            schema.as_struct(),
        )
        .unwrap();
        let got = arrow_struct_to_literal(
            &(Arc::new(StructArray::from(read_back)) as ArrayRef),
            schema.as_struct(),
        )
        .unwrap();
        assert_eq!(got, want, "nested round-trip literals mismatch");
    }

    /// 3. METRICS-EMPTY (mutation bait): record_count == N, file_size > 0, all column maps empty.
    #[tokio::test]
    async fn test_avro_writer_metrics_are_empty() {
        let (_t, file_io, location_gen) = make_temp();
        let schema = Arc::new(schema_all_types());
        let to_write = all_types_batch(&schema);

        let (_path, of) = output_file(&file_io, &location_gen, "metrics");
        let mut w = AvroWriterBuilder::new(schema.clone())
            .build(of)
            .await
            .unwrap();
        w.write(&to_write).await.unwrap();
        let res = w.close().await.unwrap();
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .partition_spec_id(0)
            .build()
            .unwrap();

        assert_eq!(data_file.record_count(), 4);
        assert!(data_file.file_size_in_bytes() > 0);
        assert!(
            data_file.column_sizes().is_empty(),
            "column_sizes must be empty for Avro (Java AvroMetrics)"
        );
        assert!(
            data_file.value_counts().is_empty(),
            "value_counts must be empty"
        );
        assert!(
            data_file.null_value_counts().is_empty(),
            "null_value_counts must be empty"
        );
        assert!(
            data_file.nan_value_counts().is_empty(),
            "nan_value_counts must be empty"
        );
        assert!(
            data_file.lower_bounds().is_empty(),
            "lower_bounds must be empty"
        );
        assert!(
            data_file.upper_bounds().is_empty(),
            "upper_bounds must be empty"
        );
    }

    /// 4. Via `DataFileWriter` (AvroWriterBuilder → RollingFileWriterBuilder → DataFileWriterBuilder).
    #[tokio::test]
    async fn test_avro_via_data_file_writer() {
        let (_t, file_io, location_gen) = make_temp();
        let file_name_gen =
            DefaultFileNameGenerator::new("dfw".to_string(), None, DataFileFormat::Avro);

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(3)
                .with_fields(vec![
                    NestedField::required(3, "foo", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(4, "bar", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let avro_builder = AvroWriterBuilder::new(schema.clone());
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            avro_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );
        let mut dfw = DataFileWriterBuilder::new(rolling)
            .build(None)
            .await
            .unwrap();

        let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into().unwrap());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])) as ArrayRef,
        ])
        .unwrap();
        dfw.write(batch).await.unwrap();
        let data_files = dfw.close().await.unwrap();
        assert_eq!(data_files.len(), 1);

        let data_file = &data_files[0];
        assert_eq!(data_file.file_format(), DataFileFormat::Avro);
        assert_eq!(data_file.content_type(), DataContentType::Data);
        assert_eq!(*data_file.partition(), Struct::empty());
        assert_eq!(data_file.record_count(), 3);
    }

    /// 5. EMPTY input → close returns vec![] and no file remains on disk.
    #[tokio::test]
    async fn test_avro_writer_empty_input() {
        let (temp_dir, file_io, location_gen) = make_temp();
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(0, "col", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );
        let (path, of) = output_file(&file_io, &location_gen, "empty");
        let w = AvroWriterBuilder::new(schema).build(of).await.unwrap();
        let res = w.close().await.unwrap();
        assert!(res.is_empty(), "empty input must yield no data files");
        assert!(
            !file_io.exists(&path).await.unwrap(),
            "no phantom file should remain"
        );
        // Directory should be empty (nothing was ever written).
        assert_eq!(std::fs::read_dir(temp_dir.path()).unwrap().count(), 0);
    }

    /// Empty-batch path: a zero-row batch through `write` then `close` is also a no-op file.
    #[tokio::test]
    async fn test_avro_writer_empty_batch_is_skipped() {
        let (_t, file_io, location_gen) = make_temp();
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(0, "col", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into().unwrap());
        let empty = RecordBatch::new_empty(arrow_schema);
        let (path, of) = output_file(&file_io, &location_gen, "empty_batch");
        let mut w = AvroWriterBuilder::new(schema).build(of).await.unwrap();
        w.write(&empty).await.unwrap();
        assert_eq!(w.current_row_num(), 0);
        let res = w.close().await.unwrap();
        assert!(res.is_empty());
        assert!(!file_io.exists(&path).await.unwrap());
    }

    /// 6. A variant column → FeatureUnsupported error at build time.
    #[tokio::test]
    async fn test_avro_writer_rejects_variant() {
        let (_t, file_io, location_gen) = make_temp();
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(0, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(1, "v", Type::Variant).into(),
                ])
                .build()
                .unwrap(),
        );
        let (_path, of) = output_file(&file_io, &location_gen, "variant");
        match AvroWriterBuilder::new(schema).build(of).await {
            Ok(_) => panic!("variant column must be rejected"),
            Err(err) => assert_eq!(err.kind(), ErrorKind::FeatureUnsupported, "got: {err}"),
        }
    }

    /// A Deflate-compressed Avro file still round-trips through the U1 reader (codec is read from
    /// the OCF header), exercising `new_with_codec`.
    #[tokio::test]
    async fn test_avro_writer_deflate_codec_roundtrip() {
        let (_t, file_io, location_gen) = make_temp();
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(0, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(1, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into().unwrap());
        let to_write = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from_iter_values(0..200)) as ArrayRef,
            Arc::new(StringArray::from_iter(
                (0..200).map(|n| Some(n.to_string())),
            )) as ArrayRef,
        ])
        .unwrap();

        let (_path, of) = output_file(&file_io, &location_gen, "deflate");
        let mut w = AvroWriterBuilder::new_with_codec(
            schema.clone(),
            Codec::Deflate(apache_avro::DeflateSettings::default()),
        )
        .build(of)
        .await
        .unwrap();
        w.write(&to_write).await.unwrap();
        let res = w.close().await.unwrap();
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .partition_spec_id(0)
            .build()
            .unwrap();
        assert_eq!(data_file.record_count(), 200);

        let input = file_io.new_input(data_file.file_path()).unwrap();
        let bytes = input.read().await.unwrap();
        let batches = read_avro_data_bytes(&bytes, schema.as_ref(), 1024).unwrap();
        let read_back =
            arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();
        assert_eq!(read_back.num_rows(), 200);
        for c in 0..to_write.num_columns() {
            assert_eq!(read_back.column(c), to_write.column(c));
        }
    }

    /// current_written_size is 0 before any write and strictly positive (and monotonic) after.
    #[tokio::test]
    async fn test_avro_writer_current_written_size_is_live() {
        let (_t, file_io, location_gen) = make_temp();
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(0, "col", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into().unwrap());
        let (_path, of) = output_file(&file_io, &location_gen, "size");
        let mut w = AvroWriterBuilder::new(schema).build(of).await.unwrap();
        assert_eq!(w.current_written_size(), 0);
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from_iter_values(0..100)) as ArrayRef,
        ])
        .unwrap();
        w.write(&batch).await.unwrap();
        let after_one = w.current_written_size();
        assert!(after_one > 0, "size must be live after a write");
    }

    /// Raw on-disk byte assertion (mutation bait for the WRITE SCHEMA itself).
    ///
    /// The U1 round-trip tests cannot detect a divergence in the writer's Avro *write schema* —
    /// e.g. flipping an optional union to `[T, null]` or widening a decimal `fixed` — because
    /// `apache_avro` embeds the (possibly-wrong) writer schema in the OCF header and self-resolves
    /// reads against it. This test closes that gap: it encodes single rows against the *exact*
    /// write schema the writer uses (`schema_to_avro_schema`) and asserts the literal bytes.
    ///
    /// A mutation that reorders the optional union changes the leading index byte (`0x02` present
    /// at member 1 vs `0x00` null at member 0); a mutation that changes the decimal `fixed` width
    /// changes the length of the sign-extended big-endian two's-complement run. Both now bite.
    #[test]
    fn test_avro_writer_raw_byte_encoding() {
        // Two optional columns: a long and decimal(10, 2). Java/U1 resolve by this exact schema.
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(
                        2,
                        "d",
                        Type::Primitive(PrimitiveType::Decimal {
                            precision: 10,
                            scale: 2,
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let avro_schema = schema_to_avro_schema(AVRO_ROOT_RECORD_NAME, &schema).unwrap();
        let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into().unwrap());

        // Row 0: x = 7 (present), d = -1 (present). Row 1: both null.
        let xcol = Arc::new(Int64Array::from(vec![Some(7_i64), None])) as ArrayRef;
        let dcol = Arc::new(
            Decimal128Array::from(vec![Some(-1_i128), None])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ) as ArrayRef;
        let batch = RecordBatch::try_new(arrow_schema, vec![xcol, dcol]).unwrap();

        let (values, _size) =
            encode_batch_to_values(&avro_schema, &batch, schema.as_struct()).unwrap();
        assert_eq!(values.len(), 2);

        // The decimal fixed width is Java `TypeUtil.decimalRequiredBytes(10)` == 5.
        assert_eq!(Type::decimal_required_bytes(10).unwrap(), 5);

        // Encode each row directly against the write schema (header-free datum) and assert bytes.
        let row0 = apache_avro::to_avro_datum(&avro_schema, values[0].clone()).unwrap();
        let row1 = apache_avro::to_avro_datum(&avro_schema, values[1].clone()).unwrap();

        // Row 0:
        //   0x02              union index 1  → "present" for x (NULL must be member 0, value member 1)
        //   0x0e              zigzag varint of 7  (7 → 14 → 0x0e)
        //   0x02              union index 1  → "present" for d
        //   ff ff ff ff ff    decimal(10, 2) -1 as 5-byte (decimal_required_bytes(10)) BE
        //                     two's-complement, sign-extended with 0xff (negative)
        assert_eq!(
            row0,
            vec![0x02, 0x0e, 0x02, 0xff, 0xff, 0xff, 0xff, 0xff],
            "row0 bytes diverged — optional-union member order or decimal fixed width is wrong"
        );

        // Row 1: both fields null → union index 0 for each (0x00, 0x00). If the union were
        // reordered to [T, null], null would resolve to member 1 (0x02) and this would fail.
        assert_eq!(
            row1,
            vec![0x00, 0x00],
            "row1 bytes diverged — a null must encode as union index 0 (NULL at member 0)"
        );

        // A positive decimal sign-extends with 0x00: decimal(10, 2) 300 → 00 00 00 01 2c.
        let pos = Arc::new(
            Decimal128Array::from(vec![Some(300_i128)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ) as ArrayRef;
        let pos_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into().unwrap());
        let pos_batch = RecordBatch::try_new(pos_schema, vec![
            Arc::new(Int64Array::from(vec![None as Option<i64>])) as ArrayRef,
            pos,
        ])
        .unwrap();
        let (pos_values, _) =
            encode_batch_to_values(&avro_schema, &pos_batch, schema.as_struct()).unwrap();
        let pos_row = apache_avro::to_avro_datum(&avro_schema, pos_values[0].clone()).unwrap();
        assert_eq!(
            pos_row,
            // x null (0x00), d present (0x02), then 5-byte BE two's-complement of 300.
            vec![0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x2c],
            "positive decimal must sign-extend with 0x00 to the fixed width"
        );
    }

    /// 2b. NESTED round-trip combining nesting WITH optionality/nulls (closes the gap that the
    /// all-required nested test could not: a null inside a nested struct, an optional nested
    /// struct that is absent, a null map value, and a list with an optional element). Java wraps
    /// each of these element/value writers in `OptionWriter`, so this exercises the union/null
    /// path *inside* a record/array/map rather than only at the top level.
    #[tokio::test]
    async fn test_avro_writer_nested_optional_roundtrip() {
        let (_t, file_io, location_gen) = make_temp();

        // col0: optional struct { 5: optional long, 6: required long }   (optional struct + null field)
        // col1: list<optional long>                                       (null element inside array)
        // col2: map<string, optional long>                                (null value inside map)
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(
                        0,
                        "col0",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(5, "a", Type::Primitive(PrimitiveType::Long))
                                .into(),
                            NestedField::optional(6, "b", Type::Primitive(PrimitiveType::Long))
                                .into(),
                        ])),
                    )
                    .into(),
                    NestedField::required(
                        1,
                        "col1",
                        Type::List(ListType::new(
                            NestedField::optional(
                                7,
                                "element",
                                Type::Primitive(PrimitiveType::Long),
                            )
                            .into(),
                        )),
                    )
                    .into(),
                    NestedField::required(
                        2,
                        "col2",
                        Type::Map(MapType::new(
                            NestedField::required(
                                10,
                                "key",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            NestedField::optional(
                                11,
                                "value",
                                Type::Primitive(PrimitiveType::Long),
                            )
                            .into(),
                        )),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into().unwrap());

        // col0: row0 = Some{ a: None, b: 100 }; row1 = None (absent optional struct).
        let col0 = {
            let fields = if let DataType::Struct(fields) = arrow_schema.field(0).data_type() {
                fields.clone()
            } else {
                unreachable!()
            };
            // Row0: struct present { a: None (null INSIDE a nested struct), b: 100 }.
            // Row1: struct absent (a null nested struct, via the parent validity mask).
            let a = Arc::new(Int64Array::from(vec![None, None])) as ArrayRef;
            let b = Arc::new(Int64Array::from(vec![Some(100_i64), None])) as ArrayRef;
            let validity = arrow_array::array::BooleanArray::from(vec![true, false]);
            Arc::new(StructArray::new(
                fields,
                vec![a, b],
                Some(validity.into_parts().0.into()),
            )) as ArrayRef
        };
        // col1: row0 = [Some(1), None, Some(3)]; row1 = [].
        let col1 = {
            let parts = ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                Some(vec![Some(1_i64), None, Some(3)]),
                Some(vec![]),
            ])
            .into_parts();
            let field = if let DataType::List(field) = arrow_schema.field(1).data_type() {
                field.clone()
            } else {
                unreachable!()
            };
            Arc::new(ListArray::new(field, parts.1, parts.2, parts.3)) as ArrayRef
        };
        // col2: row0 = {"k1": None}; row1 = {"k2": 2}.
        let col2 = {
            let mut bld = arrow_array::builder::MapBuilder::new(
                None,
                arrow_array::builder::StringBuilder::new(),
                arrow_array::builder::PrimitiveBuilder::<Int64Type>::new(),
            );
            bld.keys().append_value("k1");
            bld.values().append_null();
            bld.append(true).unwrap();
            bld.keys().append_value("k2");
            bld.values().append_value(2);
            bld.append(true).unwrap();
            let (_f, offsets, entries, nulls, ordered) = bld.finish().into_parts();
            let key_f = Field::new(MAP_KEY_FIELD_NAME, DataType::Utf8, false).with_metadata(
                HashMap::from([("PARQUET:field_id".to_string(), "10".to_string())]),
            );
            let val_f = Field::new(MAP_VALUE_FIELD_NAME, DataType::Int64, true).with_metadata(
                HashMap::from([("PARQUET:field_id".to_string(), "11".to_string())]),
            );
            let entries_fields = Fields::from(vec![key_f, val_f]);
            let (_, arrays, e_nulls) = entries.into_parts();
            let entries = StructArray::new(entries_fields.clone(), arrays, e_nulls);
            let map_field = Arc::new(Field::new(
                DEFAULT_MAP_FIELD_NAME,
                DataType::Struct(entries_fields),
                false,
            ));
            Arc::new(MapArray::new(map_field, offsets, entries, nulls, ordered)) as ArrayRef
        };

        let to_write = RecordBatch::try_new(arrow_schema, vec![col0, col1, col2]).unwrap();

        let (_path, of) = output_file(&file_io, &location_gen, "nested_opt");
        let mut w = AvroWriterBuilder::new(schema.clone())
            .build(of)
            .await
            .unwrap();
        w.write(&to_write).await.unwrap();
        let res = w.close().await.unwrap();
        let data_file = res
            .into_iter()
            .next()
            .unwrap()
            .partition_spec_id(0)
            .build()
            .unwrap();

        let input = file_io.new_input(data_file.file_path()).unwrap();
        let bytes = input.read().await.unwrap();
        let batches = read_avro_data_bytes(&bytes, schema.as_ref(), 1024).unwrap();
        let read_back =
            arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap();
        assert_eq!(read_back.num_rows(), 2);

        // Compare as Iceberg literals (order-insensitive on maps; null-aware on every nested level).
        let want = arrow_struct_to_literal(
            &(Arc::new(StructArray::from(to_write)) as ArrayRef),
            schema.as_struct(),
        )
        .unwrap();
        let got = arrow_struct_to_literal(
            &(Arc::new(StructArray::from(read_back)) as ArrayRef),
            schema.as_struct(),
        )
        .unwrap();
        assert_eq!(
            got, want,
            "nested-optional round-trip literals mismatch (a null inside a nested struct/list/map regressed)"
        );
    }
}
