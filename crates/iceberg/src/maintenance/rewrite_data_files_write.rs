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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::basic::{Encoding, Type as PhysicalType};
use parquet::file::FOOTER_SIZE;
use parquet::file::metadata::{ColumnChunkMetaData, ParquetMetaData};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

use crate::arrow::{
    ArrowFileReader, ArrowReaderBuilder, ParquetReadOptions, RecordBatchPartitionSplitter,
};
use crate::error::{Error, ErrorKind, Result};
use crate::io::{FileIO, FileMetadata};
use crate::maintenance::rewrite_data_files_router::BoundedPartitionRouter;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_FIELD_ID_ROW_ID,
    format_supports_row_lineage, schema_with_row_lineage,
};
use crate::scan::FileScanTask;
use crate::spec::{DataFile, DataFileFormat, SchemaRef};
use crate::table::Table;
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::file_writer::{ParquetWriterBuilder, parquet_compression_from_properties};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

pub(crate) struct CompactedWrite {
    pub files: Vec<DataFile>,
    #[allow(dead_code)]
    pub peak_open_partition_writers: usize,
}

pub(crate) async fn write_compacted_files(
    table: &Table,
    group: &[FileScanTask],
    target_file_size_bytes: u64,
    max_open_partition_writers: usize,
) -> Result<CompactedWrite> {
    if max_open_partition_writers == 0 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "'max-open-partition-writers' is set to 0 but must be > 0",
        ));
    }

    let schema = rewrite_write_schema(table)?;
    let spec = table.metadata().default_partition_spec().as_ref().clone();

    let location_generator = DefaultLocationGenerator::new(table.metadata().clone())?;
    let file_name_generator = DefaultFileNameGenerator::new(
        "compacted".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let compression = parquet_compression_from_properties(table.metadata().properties())?;
    let (fallback_columns, input_footers) =
        dictionary_fallback_columns(table.file_io(), group).await?;
    let mut writer_properties = WriterProperties::builder().set_compression(compression);
    for path in fallback_columns {
        writer_properties = writer_properties.set_column_dictionary_enabled(path, false);
    }
    let parquet_builder = ParquetWriterBuilder::new(writer_properties.build(), schema.clone());
    let rolling_builder = RollingFileWriterBuilder::new(
        parquet_builder,
        usize::try_from(target_file_size_bytes).unwrap_or(usize::MAX),
        table.file_io().clone(),
        location_generator,
        file_name_generator,
    );
    let writer_builder =
        DataFileWriterBuilder::new(rolling_builder).with_partition_spec(spec.clone());

    let carry_lineage = format_supports_row_lineage(table.metadata().format_version());
    let current_schema = table.metadata().current_schema().clone();
    let current_field_ids: Vec<i32> = current_schema
        .as_struct()
        .fields()
        .iter()
        .map(|field| field.id)
        .collect();
    let tasks: Vec<Result<FileScanTask>> = group
        .iter()
        .cloned()
        .map(|mut task| {
            task.predicate = None;
            task.schema = Arc::clone(&current_schema);
            task.project_field_ids = Arc::from(current_field_ids.as_slice());
            if carry_lineage {
                project_row_lineage(&mut task);
            }
            Ok(task)
        })
        .collect();
    let task_stream = Box::pin(futures::stream::iter(tasks)) as crate::scan::FileScanTaskStream;
    let mut batch_stream = ArrowReaderBuilder::new(table.file_io().clone())
        .with_prefetched_parquet_metadata(input_footers)
        .build()
        .read(task_stream)?;

    if spec.fields().is_empty() {
        let mut writer = writer_builder.build(None).await?;
        while let Some(batch) = batch_stream.try_next().await? {
            writer.write(batch).await?;
        }
        let files = writer.close().await?;
        return Ok(CompactedWrite {
            files,
            peak_open_partition_writers: 1,
        });
    }

    let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
        schema.clone(),
        table.metadata().default_partition_spec().clone(),
    )?;
    let mut router = BoundedPartitionRouter::new(writer_builder, max_open_partition_writers)?;
    while let Some(batch) = batch_stream.try_next().await? {
        for (partition_key, partition_batch) in splitter.split(&batch)? {
            router.write(partition_key, partition_batch).await?;
        }
    }
    let peak_open_partition_writers = router.peak_open_partition_writers();
    let files = router.close().await?;
    Ok(CompactedWrite {
        files,
        peak_open_partition_writers,
    })
}

fn rewrite_write_schema(table: &Table) -> Result<SchemaRef> {
    let schema = table.metadata().current_schema();
    if format_supports_row_lineage(table.metadata().format_version()) {
        Ok(Arc::new(schema_with_row_lineage(schema)?))
    } else {
        Ok(schema.clone())
    }
}

#[derive(Default)]
struct ColumnDictionaryStats {
    missing_dictionary_page: bool,
    plain_data_pages: bool,
    uncompressed_bytes: u64,
    num_values: u64,
    value_bytes: u64,
}

fn expected_value_bytes(column: &ColumnChunkMetaData) -> u64 {
    let descr = column.column_descr();
    match descr.physical_type() {
        PhysicalType::BOOLEAN | PhysicalType::INT32 | PhysicalType::FLOAT => 4,
        PhysicalType::INT64 | PhysicalType::DOUBLE => 8,
        PhysicalType::INT96 => 12,
        PhysicalType::FIXED_LEN_BYTE_ARRAY => {
            u64::try_from(descr.type_length()).unwrap_or(8).max(1)
        }
        PhysicalType::BYTE_ARRAY => column
            .statistics()
            .and_then(|stats| {
                let min = stats.min_bytes_opt()?.len();
                let max = stats.max_bytes_opt()?.len();
                Some((min + max) as u64 / 2)
            })
            .map(|average| 4 + average.max(4))
            .unwrap_or(12),
    }
}

pub(super) async fn input_parquet_metadata(
    file_io: &FileIO,
    task: &FileScanTask,
) -> Result<(Arc<str>, Arc<ParquetMetaData>)> {
    let input = file_io.new_input(task.data_file_path.as_ref())?;
    let size = if task.file_size_in_bytes > 0 {
        task.file_size_in_bytes
    } else {
        input.metadata().await?.size
    };
    let mut reader = ArrowFileReader::new(FileMetadata { size }, input.reader().await?)
        .with_parquet_read_options(
            ParquetReadOptions::builder()
                .with_metadata_size_hint(Some(FOOTER_SIZE))
                .with_preload_page_index(false)
                .with_preload_column_index(false)
                .with_preload_offset_index(false)
                .build(),
        );
    let metadata = reader.get_metadata(None).await?;
    Ok((Arc::clone(&task.data_file_path), metadata))
}

pub(super) async fn dictionary_fallback_columns(
    file_io: &FileIO,
    group: &[FileScanTask],
) -> Result<(Vec<ColumnPath>, HashMap<Arc<str>, Arc<ParquetMetaData>>)> {
    let mut seen: HashSet<Arc<str>> = HashSet::new();
    let mut columns: HashMap<ColumnPath, ColumnDictionaryStats> = HashMap::new();
    let mut footers: HashMap<Arc<str>, Arc<ParquetMetaData>> = HashMap::new();
    let mut metadata_stream = futures::stream::iter(
        group
            .iter()
            .filter(|task| {
                task.data_file_format == DataFileFormat::Parquet
                    && seen.insert(Arc::clone(&task.data_file_path))
            })
            .map(|task| input_parquet_metadata(file_io, task)),
    )
    .buffered(8);
    while let Some((path, metadata)) = metadata_stream.try_next().await? {
        fold_column_dictionary_stats(&mut columns, &metadata);
        footers.insert(path, metadata);
    }

    let mut paths: Vec<ColumnPath> = columns
        .into_iter()
        .filter(|(_, stats)| {
            stats.missing_dictionary_page
                || stats.plain_data_pages
                || stats.uncompressed_bytes.saturating_mul(2)
                    >= stats.num_values.saturating_mul(stats.value_bytes)
        })
        .map(|(path, _)| path)
        .collect();
    paths.sort_by_key(|path| path.string());
    Ok((paths, footers))
}

fn fold_column_dictionary_stats(
    columns: &mut HashMap<ColumnPath, ColumnDictionaryStats>,
    metadata: &ParquetMetaData,
) {
    for row_group in metadata.row_groups() {
        for column in row_group.columns() {
            let stats = columns
                .entry(column.column_descr().path().clone())
                .or_default();
            stats.num_values += column.num_values().max(0) as u64;
            stats.uncompressed_bytes += column.uncompressed_size().max(0) as u64;
            stats.value_bytes = stats.value_bytes.max(expected_value_bytes(column));
            match column.dictionary_page_offset() {
                None => stats.missing_dictionary_page = true,
                Some(_) => {
                    if column
                        .page_encoding_stats_mask()
                        .is_some_and(|mask| mask.is_set(Encoding::PLAIN))
                    {
                        stats.plain_data_pages = true;
                    }
                }
            }
        }
    }
}

fn project_row_lineage(task: &mut FileScanTask) {
    let mut ids = task.project_field_ids.to_vec();
    if !ids.contains(&RESERVED_FIELD_ID_ROW_ID) {
        ids.push(RESERVED_FIELD_ID_ROW_ID);
    }
    if !ids.contains(&RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER) {
        ids.push(RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER);
    }
    task.project_field_ids = Arc::from(ids);
}
