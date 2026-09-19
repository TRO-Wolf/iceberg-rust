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

use arrow_arith::boolean::is_null;
use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, RecordBatch, StructArray};
use arrow_ord::sort::{SortColumn, SortOptions, lexsort_to_indices};
use arrow_schema::{ArrowError, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use arrow_select::nullif::nullif;
use arrow_select::take::take_record_batch;
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::basic::{Encoding, Type as PhysicalType};
use parquet::file::FOOTER_SIZE;
use parquet::file::metadata::{ColumnChunkMetaData, ParquetMetaData};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

use crate::arrow::{
    ArrowFileReader, ArrowReaderBuilder, ParquetReadOptions, RecordBatchPartitionSplitter,
    schema_to_arrow_schema,
};
use crate::error::{Error, ErrorKind, Result};
use crate::io::{FileIO, FileMetadata};
use crate::maintenance::rewrite_data_files_plan::{
    ResolvedConfig, input_split_size, plan_read_tasks, write_max_file_size,
};
use crate::maintenance::rewrite_data_files_router::BoundedPartitionRouter;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER, RESERVED_FIELD_ID_ROW_ID,
    format_supports_row_lineage, schema_with_row_lineage,
};
use crate::scan::FileScanTask;
use crate::spec::{
    DataFile, DataFileFormat, NestedFieldRef, NullOrder, PartitionSpec, PartitionSpecRef,
    PrimitiveType, Schema as IcebergSchema, SchemaRef, SortDirection, Transform, Type,
};
use crate::table::Table;
use crate::transform::{BoxedTransformFunction, create_transform_function};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, TableLocationGenerator,
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
    config: &ResolvedConfig,
    output_spec: &PartitionSpecRef,
) -> Result<CompactedWrite> {
    if config.max_open_partition_writers == 0 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "'max-open-partition-writers' is set to 0 but must be > 0",
        ));
    }

    let schema = rewrite_write_schema(table)?;
    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema)?);
    let sort = rewrite_sort_plan(table, &arrow_schema);
    let spec = output_spec.as_ref().clone();

    let location_generator = TableLocationGenerator::new(table.metadata())?;
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
    let write_max = write_max_file_size(config.target_file_size_bytes, config.max_file_size_bytes);
    let rolling_builder = RollingFileWriterBuilder::new(
        parquet_builder,
        usize::try_from(write_max).unwrap_or(usize::MAX),
        table.file_io().clone(),
        location_generator,
        file_name_generator,
    );

    let carry_lineage = format_supports_row_lineage(table.metadata().format_version());
    let current_schema = table.metadata().current_schema().clone();
    let current_field_ids: Vec<i32> = current_schema
        .as_struct()
        .fields()
        .iter()
        .map(|field| field.id)
        .collect();
    let tasks: Vec<FileScanTask> = group
        .iter()
        .cloned()
        .map(|mut task| {
            task.predicate = None;
            task.schema = Arc::clone(&current_schema);
            task.project_field_ids = Arc::from(current_field_ids.as_slice());
            if carry_lineage {
                project_row_lineage(&mut task);
            }
            task
        })
        .collect();

    let input_size: u64 = tasks
        .iter()
        .fold(0u64, |sum, task| sum.saturating_add(task.length));
    let split_size = input_split_size(input_size, config);
    let read_tasks = plan_read_tasks(tasks, split_size)?;

    let splitter = (!spec.fields().is_empty())
        .then(|| {
            RecordBatchPartitionSplitter::try_new_with_computed_values(
                schema.clone(),
                output_spec.clone(),
            )
        })
        .transpose()?;
    let run_target = usize::try_from(write_max).unwrap_or(usize::MAX);

    let reader = ArrowReaderBuilder::new(table.file_io().clone())
        .with_prefetched_parquet_metadata(input_footers)
        .build();

    let mut files = Vec::new();
    let mut peak = 0usize;
    for read_task in read_tasks {
        let task_stream = Box::pin(futures::stream::iter(read_task.into_iter().map(Ok)))
            as crate::scan::FileScanTaskStream;
        let mut batch_stream = reader.clone().read(task_stream)?;

        if let Some(keys) = &sort.keys {
            let mut run: Vec<RecordBatch> = Vec::new();
            let mut run_bytes = 0usize;
            while let Some(batch) = batch_stream.try_next().await? {
                run_bytes += batch.get_array_memory_size();
                run.push(batch);
                if run_bytes >= run_target {
                    write_sorted_run(
                        &arrow_schema,
                        std::mem::take(&mut run),
                        keys,
                        splitter.as_ref(),
                        &rolling_builder,
                        &spec,
                        sort.stamp,
                        config.max_open_partition_writers,
                        &mut files,
                        &mut peak,
                    )
                    .await?;
                    run_bytes = 0;
                }
            }
            write_sorted_run(
                &arrow_schema,
                run,
                keys,
                splitter.as_ref(),
                &rolling_builder,
                &spec,
                sort.stamp,
                config.max_open_partition_writers,
                &mut files,
                &mut peak,
            )
            .await?;
            continue;
        }

        let writer_builder = DataFileWriterBuilder::new(rolling_builder.clone())
            .with_partition_spec(spec.clone())
            .with_sort_order_id(sort.stamp);
        if let Some(splitter) = &splitter {
            let mut router =
                BoundedPartitionRouter::new(writer_builder, config.max_open_partition_writers)?;
            while let Some(batch) = batch_stream.try_next().await? {
                for (partition_key, partition_batch) in splitter.split(&batch)? {
                    router.write(partition_key, partition_batch).await?;
                }
            }
            peak = peak.max(router.peak_open_partition_writers());
            files.extend(router.close().await?);
        } else {
            let mut writer = writer_builder.build(None).await?;
            while let Some(batch) = batch_stream.try_next().await? {
                writer.write(batch).await?;
            }
            files.extend(writer.close().await?);
            peak = peak.max(1);
        }
    }
    Ok(CompactedWrite {
        files,
        peak_open_partition_writers: peak,
    })
}

#[allow(clippy::too_many_arguments)]
async fn write_sorted_run(
    arrow_schema: &arrow_schema::SchemaRef,
    run: Vec<RecordBatch>,
    keys: &[RewriteSortKey],
    splitter: Option<&RecordBatchPartitionSplitter>,
    rolling_builder: &RollingFileWriterBuilder<
        ParquetWriterBuilder,
        TableLocationGenerator,
        DefaultFileNameGenerator,
    >,
    spec: &PartitionSpec,
    stamp: i32,
    max_open_partition_writers: usize,
    files: &mut Vec<DataFile>,
    peak_open_partition_writers: &mut usize,
) -> Result<()> {
    if run.is_empty() {
        return Ok(());
    }
    let sorted = {
        let run_batch = concat_batches(arrow_schema, &run).map_err(arrow_sort_err)?;
        sort_group_batch(&run_batch, keys)?
    };
    let writer_builder = DataFileWriterBuilder::new(rolling_builder.clone())
        .with_partition_spec(spec.clone())
        .with_sort_order_id(stamp);
    if let Some(splitter) = splitter {
        let mut router = BoundedPartitionRouter::new(writer_builder, max_open_partition_writers)?;
        for (partition_key, partition_batch) in splitter.split(&sorted)? {
            router.write(partition_key, partition_batch).await?;
        }
        *peak_open_partition_writers =
            (*peak_open_partition_writers).max(router.peak_open_partition_writers());
        files.extend(router.close().await?);
    } else {
        let mut writer = writer_builder.build(None).await?;
        writer.write(sorted).await?;
        *peak_open_partition_writers = (*peak_open_partition_writers).max(1);
        files.extend(writer.close().await?);
    }
    Ok(())
}

fn rewrite_write_schema(table: &Table) -> Result<SchemaRef> {
    let schema = table.metadata().current_schema();
    if format_supports_row_lineage(table.metadata().format_version()) {
        Ok(Arc::new(schema_with_row_lineage(schema)?))
    } else {
        Ok(schema.clone())
    }
}

struct RewriteSortKey {
    column: usize,
    nested_path: Vec<String>,
    transform: Option<BoxedTransformFunction>,
    canonical_nan: bool,
    options: SortOptions,
}

struct RewriteSort {
    keys: Option<Vec<RewriteSortKey>>,
    stamp: i32,
}

fn rewrite_sort_plan(table: &Table, input_schema: &ArrowSchema) -> RewriteSort {
    let order = table.metadata().default_sort_order();
    let unsorted = || RewriteSort {
        keys: None,
        stamp: 0,
    };
    if order.is_unsorted() {
        return unsorted();
    }
    let Ok(order_id) = i32::try_from(order.order_id) else {
        return unsorted();
    };
    let iceberg_schema = table.metadata().current_schema();
    let mut keys = Vec::with_capacity(order.fields.len());
    for field in &order.fields {
        if field.transform == Transform::Void {
            continue;
        }
        let Some(source) = iceberg_schema.field_by_id(field.source_id) else {
            return unsorted();
        };
        let Some((top_name, nested_path)) = sort_source_path(iceberg_schema, field.source_id)
        else {
            return unsorted();
        };
        let Ok(column) = input_schema.index_of(&top_name) else {
            return unsorted();
        };
        let (transform, key_type) = if field.transform == Transform::Identity {
            (None, source.field_type.as_ref().clone())
        } else {
            let (Ok(result_type), Ok(function)) = (
                field.transform.result_type(source.field_type.as_ref()),
                create_transform_function(&field.transform),
            ) else {
                return unsorted();
            };
            (Some(function), result_type)
        };
        keys.push(RewriteSortKey {
            column,
            nested_path,
            transform,
            canonical_nan: matches!(
                key_type,
                Type::Primitive(PrimitiveType::Float | PrimitiveType::Double)
            ),
            options: SortOptions {
                descending: field.direction == SortDirection::Descending,
                nulls_first: field.null_order == NullOrder::First,
            },
        });
    }
    RewriteSort {
        keys: (!keys.is_empty()).then_some(keys),
        stamp: order_id,
    }
}

fn sort_source_path(schema: &IcebergSchema, source_id: i32) -> Option<(String, Vec<String>)> {
    let mut stack: Vec<(Vec<String>, &[NestedFieldRef])> =
        vec![(Vec::new(), schema.as_struct().fields())];
    while let Some((prefix, fields)) = stack.pop() {
        for field in fields {
            if field.id == source_id {
                let mut full = prefix.clone();
                full.push(field.name.clone());
                let mut names = full.into_iter();
                return Some((names.next()?, names.collect()));
            }
            if let Type::Struct(inner) = field.field_type.as_ref() {
                let mut child_prefix = prefix.clone();
                child_prefix.push(field.name.clone());
                stack.push((child_prefix, inner.fields()));
            }
        }
    }
    None
}

fn sort_group_batch(batch: &RecordBatch, keys: &[RewriteSortKey]) -> Result<RecordBatch> {
    let mut sort_columns = Vec::with_capacity(keys.len());
    for key in keys {
        let mut array = batch.column(key.column).clone();
        for segment in &key.nested_path {
            let Some(parent) = array.as_any().downcast_ref::<StructArray>() else {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("sort key column '{segment}' is not a struct"),
                ));
            };
            let Some(child) = parent.column_by_name(segment) else {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("sort key struct field '{segment}' not found"),
                ));
            };
            array = if parent.null_count() > 0 {
                nullif(child.as_ref(), &is_null(parent).map_err(arrow_sort_err)?)
                    .map_err(arrow_sort_err)?
            } else {
                child.clone()
            };
        }
        if let Some(function) = &key.transform {
            array = function.transform(array)?;
        }
        if key.canonical_nan {
            array = canonicalize_nan(array);
        }
        sort_columns.push(SortColumn {
            values: array,
            options: Some(key.options),
        });
    }
    let indices = lexsort_to_indices(&sort_columns, None).map_err(arrow_sort_err)?;
    take_record_batch(batch, &indices).map_err(arrow_sort_err)
}

fn canonicalize_nan(array: ArrayRef) -> ArrayRef {
    if let Some(floats) = array.as_any().downcast_ref::<Float32Array>() {
        return Arc::new(Float32Array::from_iter(floats.iter().map(|value| {
            value.map(|float| if float.is_nan() { f32::NAN } else { float })
        })));
    }
    if let Some(floats) = array.as_any().downcast_ref::<Float64Array>() {
        return Arc::new(Float64Array::from_iter(floats.iter().map(|value| {
            value.map(|float| if float.is_nan() { f64::NAN } else { float })
        })));
    }
    array
}

fn arrow_sort_err(error: ArrowError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Failed to sort compacted rows by the table's default sort order",
    )
    .with_source(error)
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
