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

//! This module provides the `FanoutWriter` implementation.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::marker::PhantomData;

use async_trait::async_trait;

use crate::spec::{PartitionKey, Struct};
use crate::writer::partitioning::PartitioningWriter;
use crate::writer::{DefaultInput, DefaultOutput, IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// A writer that can write data to multiple partitions simultaneously.
///
/// Unlike `ClusteredWriter` which expects sorted input and maintains only one active writer,
/// `FanoutWriter` can handle unsorted data by maintaining multiple active writers in a map.
/// This allows writing to any partition at any time, but uses more memory as all writers
/// remain active until the writer is closed.
///
/// # Type Parameters
///
/// * `B` - The inner writer builder type
/// * `I` - Input type (defaults to `RecordBatch`)
/// * `O` - Output collection type (defaults to `Vec<DataFile>`)
pub struct FanoutWriter<B, I = DefaultInput, O = DefaultOutput>
where
    B: IcebergWriterBuilder<I, O>,
    O: IntoIterator + FromIterator<<O as IntoIterator>::Item>,
    <O as IntoIterator>::Item: Clone,
{
    inner_builder: B,
    partition_writers: HashMap<Struct, B::R>,
    output: Vec<<O as IntoIterator>::Item>,
    _phantom: PhantomData<I>,
}

impl<B, I, O> FanoutWriter<B, I, O>
where
    B: IcebergWriterBuilder<I, O>,
    I: Send + 'static,
    O: IntoIterator + FromIterator<<O as IntoIterator>::Item>,
    <O as IntoIterator>::Item: Send + Clone,
{
    /// Create a new `FanoutWriter`.
    pub fn new(inner_builder: B) -> Self {
        Self {
            inner_builder,
            partition_writers: HashMap::new(),
            output: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Get or create a writer for the specified partition.
    async fn get_or_create_writer(&mut self, partition_key: &PartitionKey) -> Result<&mut B::R> {
        if !self.partition_writers.contains_key(partition_key.data()) {
            let writer = self
                .inner_builder
                .build(Some(partition_key.clone()))
                .await?;
            self.partition_writers
                .insert(partition_key.data().clone(), writer);
        }

        self.partition_writers
            .get_mut(partition_key.data())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to get partition writer after creation",
                )
            })
    }
}

#[async_trait]
impl<B, I, O> PartitioningWriter<I, O> for FanoutWriter<B, I, O>
where
    B: IcebergWriterBuilder<I, O>,
    I: Send + 'static,
    O: IntoIterator + FromIterator<<O as IntoIterator>::Item> + Send + 'static,
    <O as IntoIterator>::Item: Send + Clone,
{
    async fn write(&mut self, partition_key: PartitionKey, input: I) -> Result<()> {
        let writer = self.get_or_create_writer(&partition_key).await?;
        writer.write(input).await
    }

    async fn close(mut self) -> Result<O> {
        // Close all partition writers
        let mut keys: Vec<Struct> = self.partition_writers.keys().cloned().collect();
        keys.sort_by(ascending_partition_order);
        for key in keys {
            let mut writer = self.partition_writers.remove(&key).ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to get partition writer after creation",
                )
            })?;
            self.output.extend(writer.close().await?);
        }

        // Collect all output items into the output collection type
        Ok(O::from_iter(self.output))
    }
}

fn ascending_partition_order(left: &Struct, right: &Struct) -> Ordering {
    use crate::spec::Literal;
    for (left_field, right_field) in left.iter().zip(right.iter()) {
        match (left_field, right_field) {
            (None, None) => {}
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(left_value), Some(right_value)) => {
                let order = match (left_value, right_value) {
                    (Literal::Primitive(left_lit), Literal::Primitive(right_lit)) => {
                        left_lit.partial_cmp(right_lit)
                    }
                    _ => None,
                };
                match order {
                    Some(Ordering::Equal) | None => {}
                    Some(other) => return other,
                }
            }
        }
    }
    left.fields().len().cmp(&right.fields().len())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::io::FileIO;
    use crate::spec::{
        DataFileFormat, Literal, NestedField, PartitionKey, PartitionSpec, PrimitiveType, Struct,
        Type,
    };
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;

    #[tokio::test]
    async fn test_fanout_writer_single_partition() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // Create schema with partition field
        let schema = Arc::new(
            crate::spec::Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(3, "region", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()?,
        );

        // Create partition spec - using the same pattern as data_file_writer tests
        let partition_spec = PartitionSpec::builder(schema.clone()).build()?;
        let partition_value = Struct::from_iter([Some(Literal::string("US"))]);
        let partition_key =
            PartitionKey::new(partition_spec, schema.clone(), partition_value.clone())
                .expect("PartitionKey::new: valid partition tuple");

        // Create writer builder
        let parquet_writer_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone());

        // Create rolling file writer builder
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_writer_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );

        // Create data file writer builder
        let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder);

        // Create fanout writer
        let mut writer = FanoutWriter::new(data_file_writer_builder);

        // Create test data with proper field ID metadata
        let arrow_schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                1.to_string(),
            )])),
            Field::new("name", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                2.to_string(),
            )])),
            Field::new("region", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                3.to_string(),
            )])),
        ]);

        let batch1 = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(StringArray::from(vec!["US", "US"])),
        ])?;

        let batch2 = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["Charlie", "Dave"])),
            Arc::new(StringArray::from(vec!["US", "US"])),
        ])?;

        // Write data to the same partition
        writer.write(partition_key.clone(), batch1).await?;
        writer.write(partition_key.clone(), batch2).await?;

        // Close writer and get data files
        let data_files = writer.close().await?;

        // Verify at least one file was created
        assert!(
            !data_files.is_empty(),
            "Expected at least one data file to be created"
        );

        // Verify that all data files have the correct partition value
        for data_file in &data_files {
            assert_eq!(data_file.partition, partition_value);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_fanout_writer_multiple_partitions() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // Create schema with partition field
        let schema = Arc::new(
            crate::spec::Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(3, "region", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()?,
        );

        // Create partition spec
        let partition_spec = PartitionSpec::builder(schema.clone()).build()?;

        // Create partition keys for different regions
        let partition_value_us = Struct::from_iter([Some(Literal::string("US"))]);
        let partition_key_us = PartitionKey::new(
            partition_spec.clone(),
            schema.clone(),
            partition_value_us.clone(),
        )
        .expect("PartitionKey::new: valid partition tuple");

        let partition_value_eu = Struct::from_iter([Some(Literal::string("EU"))]);
        let partition_key_eu = PartitionKey::new(
            partition_spec.clone(),
            schema.clone(),
            partition_value_eu.clone(),
        )
        .expect("PartitionKey::new: valid partition tuple");

        let partition_value_asia = Struct::from_iter([Some(Literal::string("ASIA"))]);
        let partition_key_asia = PartitionKey::new(
            partition_spec.clone(),
            schema.clone(),
            partition_value_asia.clone(),
        )
        .expect("PartitionKey::new: valid partition tuple");

        // Create writer builder
        let parquet_writer_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone());

        // Create rolling file writer builder
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_writer_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );

        // Create data file writer builder
        let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder);

        // Create fanout writer
        let mut writer = FanoutWriter::new(data_file_writer_builder);

        // Create test data with proper field ID metadata
        let arrow_schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                1.to_string(),
            )])),
            Field::new("name", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                2.to_string(),
            )])),
            Field::new("region", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                3.to_string(),
            )])),
        ]);

        // Create batches for different partitions
        let batch_us1 = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(StringArray::from(vec!["US", "US"])),
        ])?;

        let batch_eu1 = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec!["Charlie", "Dave"])),
            Arc::new(StringArray::from(vec!["EU", "EU"])),
        ])?;

        let batch_us2 = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
            Arc::new(Int32Array::from(vec![5])),
            Arc::new(StringArray::from(vec!["Eve"])),
            Arc::new(StringArray::from(vec!["US"])),
        ])?;

        let batch_asia1 = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
            Arc::new(Int32Array::from(vec![6, 7])),
            Arc::new(StringArray::from(vec!["Frank", "Grace"])),
            Arc::new(StringArray::from(vec!["ASIA", "ASIA"])),
        ])?;

        // Write data in mixed partition order to demonstrate fanout capability
        // This is the key difference from ClusteredWriter - we can write to any partition at any time
        writer.write(partition_key_us.clone(), batch_us1).await?;
        writer.write(partition_key_eu.clone(), batch_eu1).await?;
        writer.write(partition_key_us.clone(), batch_us2).await?; // Back to US partition
        writer
            .write(partition_key_asia.clone(), batch_asia1)
            .await?;

        // Close writer and get data files
        let data_files = writer.close().await?;

        // Verify files were created for all partitions
        assert!(
            data_files.len() >= 3,
            "Expected at least 3 data files (one per partition), got {}",
            data_files.len()
        );

        // Verify that we have files for each partition
        let mut partitions_found = std::collections::HashSet::new();
        for data_file in &data_files {
            partitions_found.insert(data_file.partition.clone());
        }

        assert!(
            partitions_found.contains(&partition_value_us),
            "Missing US partition"
        );
        assert!(
            partitions_found.contains(&partition_value_eu),
            "Missing EU partition"
        );
        assert!(
            partitions_found.contains(&partition_value_asia),
            "Missing ASIA partition"
        );

        Ok(())
    }

    fn partition_int(data_file: &crate::spec::DataFile) -> Option<i32> {
        match data_file.partition.fields() {
            [None] => None,
            [Some(Literal::Primitive(crate::spec::PrimitiveLiteral::Int(value)))] => Some(*value),
            other => panic!("expected single int partition, got {other:?}"),
        }
    }

    async fn close_int_partitions(values: &[Option<i32>]) -> Result<Vec<Option<i32>>> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);
        let schema = Arc::new(
            crate::spec::Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "part", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()?,
        );
        let partition_spec = PartitionSpec::builder(schema.clone()).build()?;
        let parquet_writer_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone());
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_writer_builder,
            file_io,
            location_gen,
            file_name_gen,
        );
        let mut writer = FanoutWriter::new(DataFileWriterBuilder::new(rolling_writer_builder));
        let arrow_schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                1.to_string(),
            )])),
            Field::new("part", DataType::Int32, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                2.to_string(),
            )])),
        ]);
        for (index, value) in values.iter().enumerate() {
            let key = PartitionKey::new(
                partition_spec.clone(),
                schema.clone(),
                Struct::from_iter([value.map(Literal::int)]),
            )
            .expect("PartitionKey::new: valid partition tuple");
            let batch = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
                Arc::new(Int32Array::from(vec![index as i32])),
                Arc::new(Int32Array::from(vec![*value])),
            ])?;
            writer.write(key, batch).await?;
        }
        let data_files = writer.close().await?;
        Ok(data_files.iter().map(partition_int).collect())
    }

    #[tokio::test]
    async fn test_fanout_close_drains_identity_int_partitions_ascending() -> Result<()> {
        let order = close_int_partitions(&[Some(3), Some(1), Some(4), Some(0), Some(2)]).await?;
        assert_eq!(
            order,
            vec![Some(0), Some(1), Some(2), Some(3), Some(4)],
            "FanoutWriter::close must drain in ascending partition-value order"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_fanout_close_drains_null_partition_first() -> Result<()> {
        let order = close_int_partitions(&[Some(0), None]).await?;
        assert_eq!(
            order,
            vec![None, Some(0)],
            "null partition values must close before non-null"
        );
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn measure_fanout_one_million_rows_eight_partitions() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("bench".to_string(), None, DataFileFormat::Parquet);
        let schema = Arc::new(
            crate::spec::Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "part", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()?,
        );
        let partition_spec = PartitionSpec::builder(schema.clone()).build()?;
        let parquet_writer_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone());
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_writer_builder,
            file_io,
            location_gen,
            file_name_gen,
        );
        let mut writer = FanoutWriter::new(DataFileWriterBuilder::new(rolling_writer_builder));
        let arrow_schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                1.to_string(),
            )])),
            Field::new("part", DataType::Int32, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                2.to_string(),
            )])),
        ]);
        let rows_per_partition = 1_000_000 / 8;
        let write_start = std::time::Instant::now();
        for part in 0..8 {
            let ids: Vec<i32> = (0..rows_per_partition)
                .map(|row| row + part * rows_per_partition)
                .collect();
            let parts: Vec<Option<i32>> = (0..rows_per_partition).map(|_| Some(part)).collect();
            let key = PartitionKey::new(
                partition_spec.clone(),
                schema.clone(),
                Struct::from_iter([Some(Literal::int(part))]),
            )
            .expect("PartitionKey::new: valid partition tuple");
            let batch = RecordBatch::try_new(Arc::new(arrow_schema.clone()), vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(parts)),
            ])?;
            writer.write(key, batch).await?;
        }
        let write_ms = write_start.elapsed().as_millis();
        let close_start = std::time::Instant::now();
        let data_files = writer.close().await?;
        let close_ms = close_start.elapsed().as_millis();
        assert_eq!(data_files.len(), 8);
        println!(
            "fanout 1e6/8 write_ms={write_ms} close_ms={close_ms} files={}",
            data_files.len()
        );
        Ok(())
    }
}
