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

use apache_avro::{Codec, DeflateSettings, ZstandardSettings};
use arrow_array::RecordBatch;
use parquet::file::properties::WriterProperties;

use super::{
    AvroWriter, AvroWriterBuilder, FileWriter, FileWriterBuilder, OrcWriter, OrcWriterBuilder,
    ParquetWriter, ParquetWriterBuilder, parquet_compression_from_properties,
};
use crate::arrow::FieldMatchMode;
use crate::io::OutputFile;
use crate::spec::{DataFileBuilder, DataFileFormat, MetricsConfig, SchemaRef};
use crate::writer::CurrentFileStatus;
use crate::{Error, ErrorKind, Result};

#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub enum AnyFileWriterBuilder {
    Parquet(Box<ParquetWriterBuilder>),
    Avro(AvroWriterBuilder),
    Orc(OrcWriterBuilder),
}

#[allow(missing_docs)]
pub enum AnyFileWriter {
    Parquet(Box<ParquetWriter>),
    Avro(Box<AvroWriter>),
    Orc(Box<OrcWriter>),
}

impl AnyFileWriterBuilder {
    #[allow(missing_docs)]
    pub fn for_format(
        format: DataFileFormat,
        schema: SchemaRef,
        table_properties: &HashMap<String, String>,
        metrics: MetricsConfig,
        match_mode: FieldMatchMode,
    ) -> Result<Self> {
        match format {
            DataFileFormat::Parquet => {
                let compression = parquet_compression_from_properties(table_properties)?;
                let dictionary_enabled = table_properties
                    .get("parquet.enable.dictionary")
                    .is_some_and(|value| value.eq_ignore_ascii_case("true"));
                let props = WriterProperties::builder()
                    .set_compression(compression)
                    .set_dictionary_enabled(dictionary_enabled)
                    .build();
                Ok(Self::Parquet(Box::new(
                    ParquetWriterBuilder::new_with_match_mode(props, schema, match_mode)
                        .with_metrics_config(metrics),
                )))
            }
            DataFileFormat::Avro => match table_properties.get("write.avro.compression-codec") {
                None => Ok(Self::Avro(AvroWriterBuilder::new(schema))),
                Some(raw) => {
                    let codec = parse_avro_codec(raw)?;
                    Ok(Self::Avro(AvroWriterBuilder::new_with_codec(schema, codec)))
                }
            },
            DataFileFormat::Orc => Ok(Self::Orc(
                OrcWriterBuilder::new_from_properties(schema, table_properties)?
                    .with_metrics_config(metrics),
            )),
            DataFileFormat::Puffin => Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot build a data-file writer for format {format}: a sidecar is never a data file"
                ),
            )),
        }
    }
}

fn parse_avro_codec(raw: &str) -> Result<Codec> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "null" => Ok(Codec::Null),
        "deflate" => Ok(Codec::Deflate(DeflateSettings::default())),
        "zstandard" => Ok(Codec::Zstandard(ZstandardSettings::default())),
        "uncompressed" => Ok(Codec::Null),
        "gzip" => Ok(Codec::Deflate(DeflateSettings::default())),
        "zstd" => Ok(Codec::Zstandard(ZstandardSettings::default())),
        _ => Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Invalid value for write.avro.compression-codec: {raw}"),
        )),
    }
}

impl FileWriterBuilder for AnyFileWriterBuilder {
    type R = AnyFileWriter;

    fn iceberg_schema(&self) -> Option<&SchemaRef> {
        match self {
            Self::Parquet(builder) => builder.iceberg_schema(),
            Self::Avro(builder) => builder.iceberg_schema(),
            Self::Orc(builder) => builder.iceberg_schema(),
        }
    }

    async fn build(&self, output_file: OutputFile) -> Result<Self::R> {
        match self {
            Self::Parquet(builder) => Ok(Self::R::Parquet(Box::new(
                builder.build(output_file).await?,
            ))),
            Self::Avro(builder) => Ok(Self::R::Avro(Box::new(builder.build(output_file).await?))),
            Self::Orc(builder) => Ok(Self::R::Orc(Box::new(builder.build(output_file).await?))),
        }
    }
}

impl CurrentFileStatus for AnyFileWriter {
    fn current_file_path(&self) -> String {
        match self {
            Self::Parquet(writer) => writer.current_file_path(),
            Self::Avro(writer) => writer.current_file_path(),
            Self::Orc(writer) => writer.current_file_path(),
        }
    }

    fn current_row_num(&self) -> usize {
        match self {
            Self::Parquet(writer) => writer.current_row_num(),
            Self::Avro(writer) => writer.current_row_num(),
            Self::Orc(writer) => writer.current_row_num(),
        }
    }

    fn current_written_size(&self) -> usize {
        match self {
            Self::Parquet(writer) => writer.current_written_size(),
            Self::Avro(writer) => writer.current_written_size(),
            Self::Orc(writer) => writer.current_written_size(),
        }
    }
}

impl FileWriter for AnyFileWriter {
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        match self {
            Self::Parquet(writer) => writer.write(batch).await,
            Self::Avro(writer) => writer.write(batch).await,
            Self::Orc(writer) => writer.write(batch).await,
        }
    }

    async fn close(self) -> Result<Vec<DataFileBuilder>> {
        match self {
            Self::Parquet(writer) => writer.close().await,
            Self::Avro(writer) => writer.close().await,
            Self::Orc(writer) => writer.close().await,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::sync::Arc;

    use apache_avro::Reader as AvroReader;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
    use arrow_schema::SchemaRef as ArrowSchemaRef;
    use parquet::basic::Encoding;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::*;
    use crate::arrow::schema_to_arrow_schema;
    use crate::io::FileIO;
    use crate::spec::{NestedField, PrimitiveType, Schema, Type};
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
    };

    fn make_temp() -> (tempfile::TempDir, FileIO, DefaultLocationGenerator) {
        let temp_dir = tempfile::TempDir::new().expect("create a temp dir");
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir
                .path()
                .to_str()
                .expect("temp dir path is utf-8")
                .to_string(),
        );
        (temp_dir, file_io, location_gen)
    }

    fn output_file(
        file_io: &FileIO,
        location_gen: &DefaultLocationGenerator,
        prefix: &str,
        format: DataFileFormat,
    ) -> (String, OutputFile) {
        let file_name_gen = DefaultFileNameGenerator::new(prefix.to_string(), None, format);
        let path = location_gen.generate_location(None, &file_name_gen.generate_file_name());
        let of = file_io.new_output(&path).expect("create the output file");
        (path, of)
    }

    fn schema_simple() -> Schema {
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(1, "c_int", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "c_str", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("build the simple schema")
    }

    fn simple_batch(schema: &Schema) -> RecordBatch {
        let arrow_schema: ArrowSchemaRef =
            Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow schema"));
        RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(2), None])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("a"), Some("bb"), None])) as ArrayRef,
        ])
        .expect("build the simple batch")
    }

    async fn write_single_batch(
        builder: &AnyFileWriterBuilder,
        file_io: &FileIO,
        location_gen: &DefaultLocationGenerator,
        prefix: &str,
        format: DataFileFormat,
        batch: &RecordBatch,
    ) -> (String, Vec<DataFileBuilder>) {
        let (path, output) = output_file(file_io, location_gen, prefix, format);
        let mut writer = builder.build(output).await.expect("build the file writer");
        writer.write(batch).await.expect("write the batch");
        let builders = writer.close().await.expect("close the writer");
        (path, builders)
    }

    async fn read_back_bytes(file_io: &FileIO, path: &str) -> bytes::Bytes {
        file_io
            .new_input(path)
            .expect("open the written file")
            .read()
            .await
            .expect("read the written file")
    }

    fn ocf_read_long(rest: &mut &[u8]) -> i64 {
        let mut shift = 0u32;
        let mut raw = 0u64;
        loop {
            assert!(
                shift < 70,
                "the OCF header holds a malformed avro long past ten bytes"
            );
            let byte = *rest
                .first()
                .expect("the OCF header ends inside an avro long");
            *rest = &rest[1..];
            raw |= u64::from(byte & 0x7f) << shift;
            shift += 7;
            if (byte & 0x80) == 0 {
                break;
            }
        }
        let magnitude =
            i64::try_from(raw >> 1).expect("the OCF zigzag magnitude must fit in i64");
        if raw & 1 == 0 {
            magnitude
        } else {
            !magnitude
        }
    }

    fn ocf_read_bytes<'a>(rest: &mut &'a [u8]) -> &'a [u8] {
        let len = ocf_read_long(rest);
        let len = usize::try_from(len).expect("the OCF header holds a negative field length");
        assert!(
            rest.len() >= len,
            "the OCF header ends inside a length-prefixed field"
        );
        let (head, tail) = rest.split_at(len);
        *rest = tail;
        head
    }

    fn ocf_codec_name(bytes: &[u8]) -> String {
        assert!(
            bytes.starts_with(b"Obj\x01"),
            "the bytes must open with the OCF magic"
        );
        let mut rest = &bytes[4..];
        loop {
            let count = ocf_read_long(&mut rest);
            if count == 0 {
                break;
            }
            let count = if count < 0 {
                let block_len = ocf_read_long(&mut rest);
                assert!(
                    block_len >= 0,
                    "the OCF metadata block size must not be negative"
                );
                count
                    .checked_neg()
                    .expect("the OCF metadata block count must fit in i64")
            } else {
                count
            };
            for _ in 0..count {
                let key = ocf_read_bytes(&mut rest);
                let value = ocf_read_bytes(&mut rest);
                if key == b"avro.codec" {
                    return String::from_utf8(value.to_vec())
                        .expect("the avro.codec value must be utf-8");
                }
            }
        }
        panic!("the OCF header must carry an avro.codec key");
    }

    async fn avro_codec_name_and_rows_for_property(value: &str) -> (String, usize) {
        let (_temp, file_io, location_gen) = make_temp();
        let schema = Arc::new(schema_simple());
        let properties = HashMap::from([(
            "write.avro.compression-codec".to_string(),
            value.to_string(),
        )]);
        let builder = AnyFileWriterBuilder::for_format(
            DataFileFormat::Avro,
            schema.clone(),
            &properties,
            MetricsConfig::default(),
            FieldMatchMode::Id,
        )
        .expect("route the avro arm");
        let batch = simple_batch(&schema);
        let (path, builders) = write_single_batch(
            &builder,
            &file_io,
            &location_gen,
            "any-avro-alias",
            DataFileFormat::Avro,
            &batch,
        )
        .await;
        assert_eq!(builders.len(), 1);
        let bytes = read_back_bytes(&file_io, &path).await;
        let codec = ocf_codec_name(&bytes);
        let rows = AvroReader::new(&bytes[..])
            .expect("open the OCF")
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("decode the OCF rows");
        (codec, rows.len())
    }

    fn repeated_batch(schema: &Schema) -> RecordBatch {
        let arrow_schema: ArrowSchemaRef =
            Arc::new(schema_to_arrow_schema(schema).expect("iceberg schema to arrow schema"));
        let words = ["alpha", "beta", "gamma", "delta"];
        RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(
                (0..512).map(|i| Some(i % 8)).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                (0..512).map(|i| Some(words[i % 4])).collect::<Vec<_>>(),
            )) as ArrayRef,
        ])
        .expect("build the repeated batch")
    }

    fn parquet_columns_use_dictionary(bytes: bytes::Bytes) -> Vec<bool> {
        let reader = SerializedFileReader::new(bytes).expect("read the parquet footer");
        let metadata = reader.metadata();
        assert!(
            metadata.num_row_groups() > 0,
            "the file must hold a row group"
        );
        let mut flags = Vec::new();
        for row_group in metadata.row_groups() {
            for column in row_group.columns() {
                let dictionary_encoded = column.encodings().any(|encoding| {
                    matches!(
                        encoding,
                        Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY
                    )
                });
                assert_eq!(
                    column.dictionary_page_offset().is_some(),
                    dictionary_encoded,
                    "the dictionary page and the dictionary encoding must agree"
                );
                flags.push(dictionary_encoded);
            }
        }
        flags
    }

    fn for_format_default(format: DataFileFormat, schema: SchemaRef) -> AnyFileWriterBuilder {
        AnyFileWriterBuilder::for_format(
            format,
            schema,
            &HashMap::new(),
            MetricsConfig::default(),
            FieldMatchMode::Id,
        )
        .expect("route the format")
    }

    #[tokio::test]
    async fn parquet_arm_writes_one_batch() {
        let (_temp, file_io, location_gen) = make_temp();
        let schema = Arc::new(schema_simple());
        let builder = for_format_default(DataFileFormat::Parquet, schema.clone());
        assert!(matches!(builder, AnyFileWriterBuilder::Parquet(_)));
        let batch = simple_batch(&schema);
        let (_path, builders) = write_single_batch(
            &builder,
            &file_io,
            &location_gen,
            "any-parquet",
            DataFileFormat::Parquet,
            &batch,
        )
        .await;
        assert_eq!(builders.len(), 1);
        let data_file = builders
            .into_iter()
            .next()
            .expect("one builder")
            .build()
            .expect("build the data file");
        assert_eq!(data_file.file_format(), DataFileFormat::Parquet);
        assert_eq!(data_file.record_count(), 3);
        assert!(data_file.file_size_in_bytes() > 0);
    }

    #[tokio::test]
    async fn parquet_dictionary_follows_table_property() {
        let (_temp, file_io, location_gen) = make_temp();
        let schema = Arc::new(schema_simple());
        let batch = repeated_batch(&schema);
        let unset_builder = for_format_default(DataFileFormat::Parquet, schema.clone());
        let (unset_path, _) = write_single_batch(
            &unset_builder,
            &file_io,
            &location_gen,
            "any-parquet-dict-unset",
            DataFileFormat::Parquet,
            &batch,
        )
        .await;
        let enabled_properties =
            HashMap::from([("parquet.enable.dictionary".to_string(), "true".to_string())]);
        let enabled_builder = AnyFileWriterBuilder::for_format(
            DataFileFormat::Parquet,
            schema.clone(),
            &enabled_properties,
            MetricsConfig::default(),
            FieldMatchMode::Id,
        )
        .expect("route the parquet arm");
        let (enabled_path, _) = write_single_batch(
            &enabled_builder,
            &file_io,
            &location_gen,
            "any-parquet-dict-on",
            DataFileFormat::Parquet,
            &batch,
        )
        .await;
        let unset_flags =
            parquet_columns_use_dictionary(read_back_bytes(&file_io, &unset_path).await);
        let enabled_flags =
            parquet_columns_use_dictionary(read_back_bytes(&file_io, &enabled_path).await);
        assert!(!unset_flags.is_empty(), "the file must hold columns");
        assert!(
            unset_flags.iter().all(|flag| !flag),
            "an unset property must leave dictionary encoding off"
        );
        assert!(!enabled_flags.is_empty(), "the file must hold columns");
        assert!(
            enabled_flags.iter().all(|flag| *flag),
            "parquet.enable.dictionary=true must turn dictionary encoding on"
        );
    }

    #[tokio::test]
    async fn avro_arm_writes_one_batch_with_null_codec_by_default() {
        let (_temp, file_io, location_gen) = make_temp();
        let schema = Arc::new(schema_simple());
        let builder = for_format_default(DataFileFormat::Avro, schema.clone());
        assert!(matches!(builder, AnyFileWriterBuilder::Avro(_)));
        let batch = simple_batch(&schema);
        let (path, builders) = write_single_batch(
            &builder,
            &file_io,
            &location_gen,
            "any-avro",
            DataFileFormat::Avro,
            &batch,
        )
        .await;
        assert_eq!(builders.len(), 1);
        let data_file = builders
            .into_iter()
            .next()
            .expect("one builder")
            .build()
            .expect("build the data file");
        assert_eq!(data_file.file_format(), DataFileFormat::Avro);
        assert_eq!(data_file.record_count(), 3);
        assert!(data_file.file_size_in_bytes() > 0);
        let bytes = read_back_bytes(&file_io, &path).await;
        assert_eq!(ocf_codec_name(&bytes), "null");
        let dump = String::from_utf8_lossy(&bytes);
        assert!(
            !dump.contains("deflate"),
            "the default arm must not emit a deflate header"
        );
        assert!(
            !dump.contains("zstandard"),
            "the default arm must not emit a zstandard header"
        );
        let rows = AvroReader::new(&bytes[..])
            .expect("open the OCF")
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("decode the OCF rows");
        assert_eq!(rows.len(), 3);
    }

    #[tokio::test]
    async fn orc_arm_writes_one_batch() {
        let (_temp, file_io, location_gen) = make_temp();
        let schema = Arc::new(schema_simple());
        let builder = for_format_default(DataFileFormat::Orc, schema.clone());
        assert!(matches!(builder, AnyFileWriterBuilder::Orc(_)));
        let batch = simple_batch(&schema);
        let (_path, builders) = write_single_batch(
            &builder,
            &file_io,
            &location_gen,
            "any-orc",
            DataFileFormat::Orc,
            &batch,
        )
        .await;
        assert_eq!(builders.len(), 1);
        let data_file = builders
            .into_iter()
            .next()
            .expect("one builder")
            .build()
            .expect("build the data file");
        assert_eq!(data_file.file_format(), DataFileFormat::Orc);
        assert_eq!(data_file.record_count(), 3);
        assert!(data_file.file_size_in_bytes() > 0);
    }

    #[test]
    fn puffin_arm_errors_naming_format() {
        let schema = Arc::new(schema_simple());
        let err = AnyFileWriterBuilder::for_format(
            DataFileFormat::Puffin,
            schema,
            &HashMap::new(),
            MetricsConfig::default(),
            FieldMatchMode::Id,
        )
        .expect_err("puffin must not route to a data-file writer");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("puffin"),
            "error must name the format, got {}",
            err.message()
        );
    }

    #[test]
    fn puffin_parsed_case_insensitively_still_errors() {
        let format = DataFileFormat::from_str("PUFFIN").expect("parse PUFFIN");
        assert_eq!(format, DataFileFormat::Puffin);
        let schema = Arc::new(schema_simple());
        let err = AnyFileWriterBuilder::for_format(
            format,
            schema,
            &HashMap::new(),
            MetricsConfig::default(),
            FieldMatchMode::Id,
        )
        .expect_err("parsed puffin must still hit the error arm");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("puffin"),
            "error must name the format, got {}",
            err.message()
        );
    }

    #[test]
    fn unknown_format_string_fails_at_from_str() {
        let err = DataFileFormat::from_str("csv").expect_err("csv is not a data file format");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.message().contains("csv"),
            "error must name the bad value, got {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn avro_codec_deflate_honored() {
        let (_temp, file_io, location_gen) = make_temp();
        let schema = Arc::new(schema_simple());
        let properties = HashMap::from([(
            "write.avro.compression-codec".to_string(),
            "deflate".to_string(),
        )]);
        let builder = AnyFileWriterBuilder::for_format(
            DataFileFormat::Avro,
            schema.clone(),
            &properties,
            MetricsConfig::default(),
            FieldMatchMode::Id,
        )
        .expect("route the avro arm");
        assert!(matches!(builder, AnyFileWriterBuilder::Avro(_)));
        let batch = simple_batch(&schema);
        let (path, builders) = write_single_batch(
            &builder,
            &file_io,
            &location_gen,
            "any-avro-deflate",
            DataFileFormat::Avro,
            &batch,
        )
        .await;
        assert_eq!(builders.len(), 1);
        let bytes = read_back_bytes(&file_io, &path).await;
        assert!(
            String::from_utf8_lossy(&bytes).contains("deflate"),
            "the OCF header must name the deflate codec"
        );
        let rows = AvroReader::new(&bytes[..])
            .expect("open the OCF")
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("decode the OCF rows");
        assert_eq!(rows.len(), 3);
    }

    #[tokio::test]
    async fn avro_java_alias_gzip_writes_deflate() {
        let (codec, rows) = avro_codec_name_and_rows_for_property("gzip").await;
        assert_eq!(codec, "deflate");
        assert_eq!(rows, 3);
    }

    #[tokio::test]
    async fn avro_java_alias_zstd_writes_zstandard() {
        let (codec, rows) = avro_codec_name_and_rows_for_property("zstd").await;
        assert_eq!(codec, "zstandard");
        assert_eq!(rows, 3);
    }

    #[test]
    fn ocf_codec_name_ignores_key_order() {
        fn push_long(out: &mut Vec<u8>, value: i64) {
            let mut raw =
                u64::try_from(value).expect("the fixture long must be positive") << 1;
            loop {
                let chunk = u8::try_from(raw & 0x7f).expect("seven bits must fit in a byte");
                raw >>= 7;
                if raw == 0 {
                    out.push(chunk);
                    break;
                }
                out.push(chunk | 0x80);
            }
        }
        fn push_bytes(out: &mut Vec<u8>, field: &[u8]) {
            let len = i64::try_from(field.len()).expect("the fixture field must fit in i64");
            push_long(out, len);
            out.extend_from_slice(field);
        }
        let mut header = b"Obj\x01".to_vec();
        push_long(&mut header, 2);
        push_bytes(&mut header, b"avro.codec.compression_level");
        push_bytes(&mut header, &[6]);
        push_bytes(&mut header, b"avro.codec");
        push_bytes(&mut header, b"zstandard");
        push_long(&mut header, 0);
        header.extend_from_slice(&[0u8; 16]);
        assert_eq!(ocf_codec_name(&header), "zstandard");
    }

    #[tokio::test]
    async fn avro_java_alias_zstd_stable_across_writers() {
        for _ in 0..25 {
            let (codec, rows) = avro_codec_name_and_rows_for_property("zstd").await;
            assert_eq!(codec, "zstandard");
            assert_eq!(rows, 3);
        }
    }

    #[tokio::test]
    async fn avro_java_alias_uncompressed_writes_null() {
        let (codec, rows) = avro_codec_name_and_rows_for_property("uncompressed").await;
        assert_eq!(codec, "null");
        assert_eq!(rows, 3);
    }

    #[tokio::test]
    async fn avro_codec_name_folds_case_and_whitespace() {
        let (codec, rows) = avro_codec_name_and_rows_for_property("  GZip  ").await;
        assert_eq!(codec, "deflate");
        assert_eq!(rows, 3);
    }

    #[test]
    fn avro_bogus_codec_errors_naming_value() {
        for value in ["broccoli", "snappy", "bzip2", "xz"] {
            let schema = Arc::new(schema_simple());
            let properties = HashMap::from([(
                "write.avro.compression-codec".to_string(),
                value.to_string(),
            )]);
            let err = AnyFileWriterBuilder::for_format(
                DataFileFormat::Avro,
                schema,
                &properties,
                MetricsConfig::default(),
                FieldMatchMode::Id,
            )
            .expect_err("an unsupported codec must fail");
            assert_eq!(err.kind(), ErrorKind::DataInvalid);
            assert_eq!(
                err.message(),
                format!("Invalid value for write.avro.compression-codec: {value}")
            );
        }
    }

    #[tokio::test]
    async fn orc_metrics_restrictive_config_drops_bounds() {
        let (_temp, file_io, location_gen) = make_temp();
        let schema = Arc::new(schema_simple());
        let batch = simple_batch(&schema);
        let restrictive = MetricsConfig::from_properties(&HashMap::from([(
            "write.metadata.metrics.default".to_string(),
            "none".to_string(),
        )]))
        .expect("parse the restrictive config");
        let default_builder = AnyFileWriterBuilder::for_format(
            DataFileFormat::Orc,
            schema.clone(),
            &HashMap::new(),
            MetricsConfig::default(),
            FieldMatchMode::Id,
        )
        .expect("route the default ORC arm");
        let restrictive_builder = AnyFileWriterBuilder::for_format(
            DataFileFormat::Orc,
            schema.clone(),
            &HashMap::new(),
            restrictive,
            FieldMatchMode::Id,
        )
        .expect("route the restrictive ORC arm");
        let (_path, default_builders) = write_single_batch(
            &default_builder,
            &file_io,
            &location_gen,
            "any-orc-default",
            DataFileFormat::Orc,
            &batch,
        )
        .await;
        let (_path, restrictive_builders) = write_single_batch(
            &restrictive_builder,
            &file_io,
            &location_gen,
            "any-orc-none",
            DataFileFormat::Orc,
            &batch,
        )
        .await;
        let default_file = default_builders
            .into_iter()
            .next()
            .expect("one builder")
            .build()
            .expect("build the default data file");
        let restrictive_file = restrictive_builders
            .into_iter()
            .next()
            .expect("one builder")
            .build()
            .expect("build the restrictive data file");
        assert!(
            default_file.lower_bounds().contains_key(&1),
            "the default config must keep the int lower bound"
        );
        assert!(
            default_file.upper_bounds().contains_key(&1),
            "the default config must keep the int upper bound"
        );
        assert!(
            restrictive_file.lower_bounds().is_empty(),
            "the none config must drop every lower bound"
        );
        assert!(
            restrictive_file.upper_bounds().is_empty(),
            "the none config must drop every upper bound"
        );
    }
}
