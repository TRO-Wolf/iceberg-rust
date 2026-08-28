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

//! Iceberg writer module.
//!
//! Two writer layers compose. A `FileWriter` writes a physical file, such as Parquet or ORC. An
//! `IcebergWriter` writes a logical Iceberg file, such as a data file or a delete file, and uses a
//! `FileWriter` inside. Partitioning writers wrap an `IcebergWriter`.
//!
//! Each layer has a builder trait beside it: `FileWriterBuilder` and `IcebergWriterBuilder`.
//! Build a writer by nesting the builders. Implement `IcebergWriter` to add your own layer.
//!
//! # Data file writer over Parquet
//! ```rust, no_run
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! use arrow_array::{ArrayRef, BooleanArray, Int32Array, RecordBatch, StringArray};
//! use async_trait::async_trait;
//! use iceberg::io::{FileIO, FileIOBuilder};
//! use iceberg::spec::DataFile;
//! use iceberg::transaction::Transaction;
//! use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
//! use iceberg::writer::file_writer::ParquetWriterBuilder;
//! use iceberg::writer::file_writer::location_generator::{
//!     DefaultFileNameGenerator, DefaultLocationGenerator,
//! };
//! use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
//! use iceberg::{Catalog, CatalogBuilder, MemoryCatalog, Result, TableIdent};
//! use parquet::file::properties::WriterProperties;
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() -> Result<()> {
//!     use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
//!     use iceberg::writer::file_writer::rolling_writer::{
//!         RollingFileWriter, RollingFileWriterBuilder,
//!     };
//!     let catalog = MemoryCatalogBuilder::default()
//!         .load(
//!             "memory",
//!             HashMap::from([(
//!                 MEMORY_CATALOG_WAREHOUSE.to_string(),
//!                 "file:///path/to/warehouse".to_string(),
//!             )]),
//!         )
//!         .await?;
//!
//!     let table = catalog
//!         .load_table(&TableIdent::from_strs(["hello", "world"])?)
//!         .await?;
//!     let location_generator = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
//!     let file_name_generator = DefaultFileNameGenerator::new(
//!         "test".to_string(),
//!         None,
//!         iceberg::spec::DataFileFormat::Parquet,
//!     );
//!
//!     let parquet_writer_builder = ParquetWriterBuilder::new(
//!         WriterProperties::default(),
//!         table.metadata().current_schema().clone(),
//!     );
//!
//!     let rolling_file_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
//!         parquet_writer_builder,
//!         table.file_io().clone(),
//!         location_generator.clone(),
//!         file_name_generator.clone(),
//!     );
//!
//!     let data_file_writer_builder = DataFileWriterBuilder::new(rolling_file_writer_builder);
//!     let mut data_file_writer = data_file_writer_builder.build(None).await?;
//!
//!     // Write the data using data_file_writer...
//!
//!     // `close` returns the data files to commit.
//!     let data_files = data_file_writer.close().await.unwrap();
//!
//!     Ok(())
//! }
//! ```
//!
//! # Custom writer layer
//!
//! Implement both traits to wrap any writer. This one records latency.
//!
//! ```rust, no_run
//! use std::time::Instant;
//!
//! use arrow_array::RecordBatch;
//! use iceberg::Result;
//! use iceberg::spec::{DataFile, PartitionKey};
//! use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
//!
//! #[derive(Clone)]
//! struct LatencyRecordWriterBuilder<B> {
//!     inner_writer_builder: B,
//! }
//!
//! #[async_trait::async_trait]
//! impl<B: IcebergWriterBuilder> IcebergWriterBuilder for LatencyRecordWriterBuilder<B> {
//!     type R = LatencyRecordWriter<B::R>;
//!
//!     async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
//!         Ok(LatencyRecordWriter {
//!             inner_writer: self.inner_writer_builder.build(partition_key).await?,
//!         })
//!     }
//! }
//!
//! struct LatencyRecordWriter<W> {
//!     inner_writer: W,
//! }
//!
//! #[async_trait::async_trait]
//! impl<W: IcebergWriter> IcebergWriter for LatencyRecordWriter<W> {
//!     async fn write(&mut self, input: RecordBatch) -> Result<()> {
//!         let start = Instant::now();
//!         self.inner_writer.write(input).await?;
//!         let _latency = start.elapsed();
//!         Ok(())
//!     }
//!
//!     async fn close(&mut self) -> Result<Vec<DataFile>> {
//!         let start = Instant::now();
//!         let res = self.inner_writer.close().await?;
//!         let _latency = start.elapsed();
//!         Ok(res)
//!     }
//! }
//! ```
//!
//! # Partitioning writers
//!
//! A partitioning writer wraps a `DataFileWriter` to write a partitioned table.
//!
//! ## FanoutWriter - For Unsorted Data
//!
//! `FanoutWriter` keeps one writer open per partition, so writes may arrive in any order.
//!
//! ```rust, no_run
//! # // Same setup as the simple example above...
//! # use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
//! # use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
//! # use iceberg::{Catalog, CatalogBuilder, Result, TableIdent};
//! # use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
//! # use iceberg::writer::file_writer::ParquetWriterBuilder;
//! # use iceberg::writer::file_writer::location_generator::{
//! #     DefaultFileNameGenerator, DefaultLocationGenerator,
//! # };
//! # use parquet::file::properties::WriterProperties;
//! # use std::collections::HashMap;
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<()> {
//! # let catalog = MemoryCatalogBuilder::default()
//! #     .load("memory", HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), "file:///path/to/warehouse".to_string())]))
//! #     .await?;
//! # let table = catalog.load_table(&TableIdent::from_strs(["hello", "world"])?).await?;
//! # let location_generator = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
//! # let file_name_generator = DefaultFileNameGenerator::new("test".to_string(), None, iceberg::spec::DataFileFormat::Parquet);
//! # let parquet_writer_builder = ParquetWriterBuilder::new(WriterProperties::default(), table.metadata().current_schema().clone());
//! # let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
//! #     parquet_writer_builder, table.file_io().clone(), location_generator, file_name_generator);
//! # let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder);
//!
//! // Wrap the data file writer with FanoutWriter for partitioning
//! use iceberg::writer::partitioning::fanout_writer::FanoutWriter;
//! use iceberg::writer::partitioning::PartitioningWriter;
//! use iceberg::spec::{Literal, PartitionKey, Struct};
//!
//! let mut fanout_writer = FanoutWriter::new(data_file_writer_builder);
//!
//! // Create partition keys for different regions
//! let schema = table.metadata().current_schema().clone();
//! let partition_spec = table.metadata().default_partition_spec().as_ref().clone();
//!
//! let partition_key_us = PartitionKey::new(
//!     partition_spec.clone(),
//!     schema.clone(),
//!     Struct::from_iter([Some(Literal::string("US"))]),
//! )?;
//!
//! let partition_key_eu = PartitionKey::new(
//!     partition_spec.clone(),
//!     schema.clone(),
//!     Struct::from_iter([Some(Literal::string("EU"))]),
//! )?;
//!
//! // Write to different partitions in any order - can interleave partition writes
//! // fanout_writer.write(partition_key_us.clone(), batch_us1).await?;
//! // fanout_writer.write(partition_key_eu.clone(), batch_eu1).await?;
//! // fanout_writer.write(partition_key_us.clone(), batch_us2).await?; // Back to US - OK!
//! // fanout_writer.write(partition_key_eu.clone(), batch_eu2).await?; // Back to EU - OK!
//!
//! let data_files = fanout_writer.close().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## ClusteredWriter - For Sorted Data
//!
//! `ClusteredWriter` keeps one writer open in total, so it uses less memory. It needs input
//! already sorted by partition key. A write that returns to a closed partition fails.
//!
//! ```rust, no_run
//! # // Same setup as the simple example above...
//! # use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
//! # use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
//! # use iceberg::{Catalog, CatalogBuilder, Result, TableIdent};
//! # use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
//! # use iceberg::writer::file_writer::ParquetWriterBuilder;
//! # use iceberg::writer::file_writer::location_generator::{
//! #     DefaultFileNameGenerator, DefaultLocationGenerator,
//! # };
//! # use parquet::file::properties::WriterProperties;
//! # use std::collections::HashMap;
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<()> {
//! # let catalog = MemoryCatalogBuilder::default()
//! #     .load("memory", HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), "file:///path/to/warehouse".to_string())]))
//! #     .await?;
//! # let table = catalog.load_table(&TableIdent::from_strs(["hello", "world"])?).await?;
//! # let location_generator = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
//! # let file_name_generator = DefaultFileNameGenerator::new("test".to_string(), None, iceberg::spec::DataFileFormat::Parquet);
//! # let parquet_writer_builder = ParquetWriterBuilder::new(WriterProperties::default(), table.metadata().current_schema().clone());
//! # let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
//! #     parquet_writer_builder, table.file_io().clone(), location_generator, file_name_generator);
//! # let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder);
//!
//! // Wrap the data file writer with ClusteredWriter for sorted partitioning
//! use iceberg::writer::partitioning::clustered_writer::ClusteredWriter;
//! use iceberg::writer::partitioning::PartitioningWriter;
//! use iceberg::spec::{Literal, PartitionKey, Struct};
//!
//! let mut clustered_writer = ClusteredWriter::new(data_file_writer_builder);
//!
//! // Create partition keys (must write in sorted order)
//! let schema = table.metadata().current_schema().clone();
//! let partition_spec = table.metadata().default_partition_spec().as_ref().clone();
//!
//! let partition_key_asia = PartitionKey::new(
//!     partition_spec.clone(),
//!     schema.clone(),
//!     Struct::from_iter([Some(Literal::string("ASIA"))]),
//! )?;
//!
//! let partition_key_eu = PartitionKey::new(
//!     partition_spec.clone(),
//!     schema.clone(),
//!     Struct::from_iter([Some(Literal::string("EU"))]),
//! )?;
//!
//! let partition_key_us = PartitionKey::new(
//!     partition_spec.clone(),
//!     schema.clone(),
//!     Struct::from_iter([Some(Literal::string("US"))]),
//! )?;
//!
//! // Write to partitions in sorted order (ASIA -> EU -> US)
//! // clustered_writer.write(partition_key_asia, batch_asia).await?;
//! // clustered_writer.write(partition_key_eu, batch_eu).await?;
//! // clustered_writer.write(partition_key_us, batch_us).await?;
//! // Writing back to ASIA would fail since data must be sorted!
//!
//! let data_files = clustered_writer.close().await?;
//!
//!     Ok(())
//! }
//! ```

pub mod base_writer;
pub mod file_writer;
pub mod partitioning;
pub(crate) mod write_defaults;

use arrow_array::RecordBatch;

use crate::Result;
use crate::spec::{DataFile, PartitionKey};

type DefaultInput = RecordBatch;
type DefaultOutput = Vec<DataFile>;

/// The builder for iceberg writer.
#[async_trait::async_trait]
pub trait IcebergWriterBuilder<I = DefaultInput, O = DefaultOutput>: Send + Sync + 'static {
    /// The associated writer type.
    type R: IcebergWriter<I, O>;
    /// Build the iceberg writer with an optional partition key.
    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R>;
}

/// The iceberg writer used to write data to iceberg table.
#[async_trait::async_trait]
pub trait IcebergWriter<I = DefaultInput, O = DefaultOutput>: Send + 'static {
    /// Write data to iceberg table.
    async fn write(&mut self, input: I) -> Result<()>;
    /// Close the writer and return the written data files. A failed close may lose the
    /// data already written. Build a new writer and write it again.
    ///
    /// # Panics
    ///
    /// Any call after `close` panics, whether the close succeeded or failed.
    async fn close(&mut self) -> Result<O>;
}

/// The current file status. Only a writer that holds one file at a time implements this.
pub trait CurrentFileStatus {
    /// Get the current file path.
    fn current_file_path(&self) -> String;
    /// Get the current file row number.
    fn current_row_num(&self) -> usize;
    /// Get the current file written size.
    fn current_written_size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use arrow_array::RecordBatch;
    use arrow_schema::Schema;
    use arrow_select::concat::concat_batches;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use super::IcebergWriter;
    use crate::io::FileIO;
    use crate::spec::{DataFile, DataFileFormat};

    // Compile-time proof that the trait stays object safe. Never called.
    async fn _guarantee_object_safe(mut w: Box<dyn IcebergWriter>) {
        let _ = w
            .write(RecordBatch::new_empty(Schema::empty().into()))
            .await;
        let _ = w.close().await;
    }

    // Assert the Parquet file reads back as the batch written, and that the DataFile
    // metadata matches it.
    pub(crate) async fn check_parquet_data_file(
        file_io: &FileIO,
        data_file: &DataFile,
        batch: &RecordBatch,
    ) {
        assert_eq!(data_file.file_format, DataFileFormat::Parquet);

        let input_file = file_io.new_input(data_file.file_path.clone()).unwrap();
        let input_content = input_file.read().await.unwrap();
        let reader_builder =
            ParquetRecordBatchReaderBuilder::try_new(input_content.clone()).unwrap();

        let reader = reader_builder.build().unwrap();
        let batches = reader.map(|batch| batch.unwrap()).collect::<Vec<_>>();
        let res = concat_batches(&batch.schema(), &batches).unwrap();
        assert_eq!(*batch, res);
    }
}
