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

use arrow_array::{RecordBatch, StructArray};
use bytes::Bytes;

use super::{FileWriter, FileWriterBuilder};
use crate::arrow::arrow_struct_to_literal;
use crate::io::OutputFile;
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, MetricsConfig, SchemaRef, Struct,
    TableProperties,
};
use crate::writer::CurrentFileStatus;
use crate::{Error, ErrorKind, Result};

mod column;
mod encode;
mod footer_write;
mod metrics;
mod null_repair;
mod orc_type;

use column::StripeEncoder;
use encode::{DEFAULT_COMPRESSION_BLOCK_SIZE, OrcCompression, compress_chunks};
use footer_write::{
    ORC_MAGIC, ORC_WRITER_TIMEZONE, StripeRecord, encode_footer, encode_postscript,
    encode_stripe_footer,
};
use metrics::OrcMetricsCollector;
use null_repair::{repair_container_nulls, schema_has_container};
use orc_type::{OrcSchema, build_orc_schema};

#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub struct OrcWriterBuilder {
    schema: SchemaRef,
    metrics_config: Arc<MetricsConfig>,
    compression: OrcCompression,
    stripe_size: usize,
    compression_block_size: usize,
}

impl OrcWriterBuilder {
    #[allow(missing_docs)]
    pub fn new(schema: SchemaRef) -> Self {
        OrcWriterBuilder {
            schema,
            metrics_config: Arc::new(MetricsConfig::default()),
            compression: OrcCompression::Zlib,
            stripe_size: TableProperties::PROPERTY_ORC_STRIPE_SIZE_BYTES_DEFAULT as usize,
            compression_block_size: DEFAULT_COMPRESSION_BLOCK_SIZE,
        }
    }

    #[allow(missing_docs)]
    #[allow(clippy::missing_errors_doc)]
    pub fn new_from_properties(
        schema: SchemaRef,
        properties: &HashMap<String, String>,
    ) -> Result<Self> {
        let codec_raw = properties
            .get(TableProperties::PROPERTY_ORC_COMPRESSION_CODEC)
            .map(String::as_str)
            .unwrap_or(TableProperties::PROPERTY_ORC_COMPRESSION_CODEC_DEFAULT);
        let stripe_size = match properties.get(TableProperties::PROPERTY_ORC_STRIPE_SIZE_BYTES) {
            None => TableProperties::PROPERTY_ORC_STRIPE_SIZE_BYTES_DEFAULT,
            Some(raw) => raw.trim().parse::<u64>().map_err(|error| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Invalid value for {}: {raw}",
                        TableProperties::PROPERTY_ORC_STRIPE_SIZE_BYTES
                    ),
                )
                .with_source(error)
            })?,
        };
        Ok(OrcWriterBuilder {
            schema,
            metrics_config: Arc::new(MetricsConfig::default()),
            compression: OrcCompression::from_codec_name(codec_raw.trim())?,
            stripe_size: stripe_size.max(1) as usize,
            compression_block_size: DEFAULT_COMPRESSION_BLOCK_SIZE,
        })
    }

    #[allow(missing_docs)]
    pub fn with_metrics_config(mut self, metrics_config: impl Into<Arc<MetricsConfig>>) -> Self {
        self.metrics_config = metrics_config.into();
        self
    }
}

impl FileWriterBuilder for OrcWriterBuilder {
    type R = OrcWriter;

    fn iceberg_schema(&self) -> Option<&SchemaRef> {
        Some(&self.schema)
    }

    async fn build(&self, output_file: OutputFile) -> Result<Self::R> {
        let orc_schema = build_orc_schema(&self.schema)?;
        Ok(OrcWriter {
            stripe: StripeEncoder::new(&orc_schema),
            metrics: OrcMetricsCollector::new(&self.schema, &self.metrics_config),
            schema: self.schema.clone(),
            orc_schema,
            output_file,
            compression: self.compression,
            stripe_size: self.stripe_size,
            compression_block_size: self.compression_block_size,
            body: ORC_MAGIC.to_vec(),
            stripes: Vec::new(),
            total_rows: 0,
            repair_container_nulls: schema_has_container(self.schema.as_struct()),
        })
    }
}

#[allow(missing_docs)]
pub struct OrcWriter {
    schema: SchemaRef,
    orc_schema: OrcSchema,
    output_file: OutputFile,
    compression: OrcCompression,
    stripe_size: usize,
    compression_block_size: usize,
    stripe: StripeEncoder,
    metrics: OrcMetricsCollector,
    body: Vec<u8>,
    stripes: Vec<StripeRecord>,
    total_rows: u64,
    repair_container_nulls: bool,
}

impl OrcWriter {
    fn flush_stripe(&mut self) -> Result<()> {
        if self.stripe.rows() == 0 {
            return Ok(());
        }
        let finished = self.stripe.finish(
            &self.orc_schema,
            self.compression,
            self.compression_block_size,
        )?;

        let mut sizes_by_orc_index: HashMap<usize, u64> = HashMap::new();
        for stream in &finished.streams {
            *sizes_by_orc_index
                .entry(stream.column as usize)
                .or_insert(0) += stream.length;
        }
        self.metrics
            .observe_stripe_column_sizes(&self.orc_schema, &sizes_by_orc_index);

        let stripe_footer = encode_stripe_footer(
            &finished.streams,
            &finished.encodings,
            Some(ORC_WRITER_TIMEZONE),
        );
        let stripe_footer = compress_chunks(
            &stripe_footer,
            self.compression,
            self.compression_block_size,
        )?;

        let offset = self.body.len() as u64;
        let data_length = finished.data.len() as u64;
        let footer_length = stripe_footer.len() as u64;
        self.body.extend_from_slice(&finished.data);
        self.body.extend_from_slice(&stripe_footer);
        self.stripes.push(StripeRecord {
            offset,
            index_length: 0,
            data_length,
            footer_length,
            number_of_rows: finished.rows,
        });
        Ok(())
    }

    fn finish_file(&mut self) -> Result<(Vec<u8>, Vec<i64>)> {
        self.flush_stripe()?;
        let content_length = self.body.len() as u64;
        let footer = encode_footer(
            ORC_MAGIC.len() as u64,
            content_length,
            &self.stripes,
            &self.orc_schema.types,
            self.total_rows,
        );
        let footer = compress_chunks(&footer, self.compression, self.compression_block_size)?;
        let postscript = encode_postscript(
            footer.len() as u64,
            0,
            self.compression,
            self.compression_block_size as u64,
        );
        let postscript_length = u8::try_from(postscript.len()).map_err(|_| {
            Error::new(
                ErrorKind::Unexpected,
                "The ORC PostScript exceeded the 255-byte length its trailing byte can carry",
            )
        })?;

        let split_offsets = self
            .stripes
            .iter()
            .map(|stripe| i64::try_from(stripe.offset).unwrap_or(i64::MAX))
            .collect();

        let mut bytes = std::mem::take(&mut self.body);
        bytes.extend_from_slice(&footer);
        bytes.extend_from_slice(&postscript);
        bytes.push(postscript_length);
        Ok((bytes, split_offsets))
    }
}

impl FileWriter for OrcWriter {
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let struct_array = StructArray::from(batch.clone());
        let array_ref = Arc::new(struct_array.clone()) as arrow_array::ArrayRef;
        let mut rows = arrow_struct_to_literal(&array_ref, self.schema.as_struct())?;
        if self.repair_container_nulls {
            for (row_index, row) in rows.iter_mut().enumerate() {
                *row = repair_container_nulls(
                    self.schema.as_struct(),
                    &struct_array,
                    row_index,
                    row.take(),
                );
            }
        }
        for row in &rows {
            self.stripe.append_row(&self.orc_schema, row.as_ref())?;
            self.metrics.observe_row(&self.schema, row.as_ref());
        }
        self.total_rows += rows.len() as u64;
        if self.stripe.estimated_size() >= self.stripe_size {
            self.flush_stripe()?;
        }
        Ok(())
    }

    async fn close(mut self) -> Result<Vec<DataFileBuilder>> {
        if self.total_rows == 0 {
            self.output_file.delete().await.map_err(|error| {
                Error::new(ErrorKind::Unexpected, "Failed to delete an empty ORC file.")
                    .with_source(error)
            })?;
            return Ok(vec![]);
        }

        let (bytes, split_offsets) = self.finish_file()?;
        let file_size_in_bytes = bytes.len() as u64;
        let record_count = self.total_rows;

        let OrcWriter {
            output_file,
            metrics,
            ..
        } = self;

        output_file
            .write(Bytes::from(bytes))
            .await
            .map_err(|error| {
                Error::new(ErrorKind::Unexpected, "Failed to write an ORC data file.")
                    .with_source(error)
            })?;

        let column_metrics = metrics.build();
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(output_file.location().to_string())
            .file_format(DataFileFormat::Orc)
            .partition(Struct::empty())
            .record_count(record_count)
            .file_size_in_bytes(file_size_in_bytes)
            .column_sizes(column_metrics.column_sizes)
            .value_counts(column_metrics.value_counts)
            .null_value_counts(column_metrics.null_value_counts)
            .nan_value_counts(column_metrics.nan_value_counts)
            .lower_bounds(column_metrics.lower_bounds)
            .upper_bounds(column_metrics.upper_bounds)
            .split_offsets(Some(split_offsets));
        Ok(vec![builder])
    }
}

impl CurrentFileStatus for OrcWriter {
    fn current_file_path(&self) -> String {
        self.output_file.location().to_string()
    }

    fn current_row_num(&self) -> usize {
        self.total_rows as usize
    }

    fn current_written_size(&self) -> usize {
        self.body.len() + self.stripe.estimated_size()
    }
}

#[cfg(test)]
mod tests {
    include!("orc_writer_tests.rs");
    include!("orc_writer_layout_tests.rs");
    include!("orc_writer_metrics_tests.rs");
}
