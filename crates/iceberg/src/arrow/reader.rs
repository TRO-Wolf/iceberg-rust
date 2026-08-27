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

//! Parquet file data reader

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::str::FromStr;
use std::sync::Arc;

use arrow_arith::boolean::{and, and_kleene, is_not_null, is_null, not, or, or_kleene};
use arrow_array::{Array, ArrayRef, BooleanArray, Datum as ArrowDatum, RecordBatch, Scalar};
use arrow_cast::cast::cast;
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow_schema::{
    ArrowError, DataType, FieldRef, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use arrow_select::filter::filter_record_batch;
use arrow_string::like::starts_with;
use bytes::Bytes;
use fnv::FnvHashSet;
use futures::future::BoxFuture;
use futures::{FutureExt, Stream, StreamExt, TryFutureExt, TryStreamExt};
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ArrowReaderMetadata, ArrowReaderOptions, RowFilter, RowSelection, RowSelector,
};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::file::metadata::{
    ColumnChunkMetaData, PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader, RowGroupMetaData,
};
use parquet::schema::types::{SchemaDescriptor, Type as ParquetType};
use typed_builder::TypedBuilder;

use crate::arrow::avro_reader::read_avro_data_file;
use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
use crate::arrow::delete_filter::positional_delete_keep_mask;
use crate::arrow::equality_delete_set::EqDeleteKeySet;
use crate::arrow::int96::coerce_int96_timestamps;
use crate::arrow::orc_reader::read_orc_data_file;
use crate::arrow::record_batch_predicate::{
    evaluate_predicate_to_mask, is_nan_row_mask, not_nan_row_mask, null_filled,
};
use crate::arrow::record_batch_transformer::{
    RecordBatchTransformer, RecordBatchTransformerBuilder,
};
use crate::arrow::{arrow_schema_to_schema, get_arrow_datum};
use crate::delete_vector::DeleteVector;
use crate::error::Result;
use crate::expr::visitors::bound_predicate_visitor::{BoundPredicateVisitor, visit};
use crate::expr::visitors::page_index_evaluator::PageIndexEvaluator;
use crate::expr::visitors::row_group_metrics_evaluator::RowGroupMetricsEvaluator;
use crate::expr::{BoundPredicate, BoundReference};
use crate::io::{FileIO, FileMetadata, FileRead};
use crate::metadata_columns::{
    RESERVED_FIELD_ID_FILE, RESERVED_FIELD_ID_POS, RESERVED_FIELD_ID_ROW_ID, get_metadata_field,
    is_metadata_field, is_row_lineage_field,
};
use crate::scan::{ArrowRecordBatchStream, FileScanTask, FileScanTaskStream};
use crate::spec::{DataFileFormat, Datum, NameMapping, NestedField, PrimitiveType, Schema, Type};
use crate::utils::available_parallelism;
use crate::{Error, ErrorKind};

/// Default gap between byte ranges below which they are coalesced into a
/// single request. Matches object_store's `OBJECT_STORE_COALESCE_DEFAULT`.
const DEFAULT_RANGE_COALESCE_BYTES: u64 = 1024 * 1024;

/// Default maximum number of coalesced byte ranges fetched concurrently.
/// Matches object_store's `OBJECT_STORE_COALESCE_PARALLEL`.
const DEFAULT_RANGE_FETCH_CONCURRENCY: usize = 10;

/// Default number of bytes to prefetch when parsing Parquet footer metadata.
/// Matches DataFusion's default `ParquetOptions::metadata_size_hint`.
const DEFAULT_METADATA_SIZE_HINT: usize = 512 * 1024;

/// Options for tuning Parquet file I/O.
#[derive(Clone, Copy, Debug, TypedBuilder)]
#[builder(field_defaults(setter(prefix = "with_")))]
pub(crate) struct ParquetReadOptions {
    /// Number of bytes to prefetch for parsing the Parquet metadata.
    ///
    /// This hint can help reduce the number of fetch requests. For more details see the
    /// [ParquetMetaDataReader documentation](https://docs.rs/parquet/latest/parquet/file/metadata/struct.ParquetMetaDataReader.html#method.with_prefetch_hint).
    ///
    /// Defaults to 512 KiB, matching DataFusion's default `ParquetOptions::metadata_size_hint`.
    #[builder(default = Some(DEFAULT_METADATA_SIZE_HINT))]
    pub(crate) metadata_size_hint: Option<usize>,
    /// Gap threshold for merging nearby byte ranges into a single request.
    /// Ranges with gaps smaller than this value will be coalesced.
    ///
    /// Defaults to 1 MiB, matching object_store's `OBJECT_STORE_COALESCE_DEFAULT`.
    #[builder(default = DEFAULT_RANGE_COALESCE_BYTES)]
    pub(crate) range_coalesce_bytes: u64,
    /// Maximum number of merged byte ranges to fetch concurrently.
    ///
    /// Defaults to 10, matching object_store's `OBJECT_STORE_COALESCE_PARALLEL`.
    #[builder(default = DEFAULT_RANGE_FETCH_CONCURRENCY)]
    pub(crate) range_fetch_concurrency: usize,
    /// Whether to preload the column index when reading Parquet metadata.
    #[builder(default = true)]
    pub(crate) preload_column_index: bool,
    /// Whether to preload the offset index when reading Parquet metadata.
    #[builder(default = true)]
    pub(crate) preload_offset_index: bool,
    /// Whether to preload the page index when reading Parquet metadata.
    #[builder(default = false)]
    pub(crate) preload_page_index: bool,
}

impl ParquetReadOptions {
    pub(crate) fn metadata_size_hint(&self) -> Option<usize> {
        self.metadata_size_hint
    }

    pub(crate) fn range_coalesce_bytes(&self) -> u64 {
        self.range_coalesce_bytes
    }

    pub(crate) fn range_fetch_concurrency(&self) -> usize {
        self.range_fetch_concurrency
    }

    pub(crate) fn preload_column_index(&self) -> bool {
        self.preload_column_index
    }

    pub(crate) fn preload_offset_index(&self) -> bool {
        self.preload_offset_index
    }

    pub(crate) fn preload_page_index(&self) -> bool {
        self.preload_page_index
    }
}

/// Builder to create ArrowReader
pub struct ArrowReaderBuilder {
    batch_size: Option<usize>,
    file_io: FileIO,
    concurrency_limit_data_files: usize,
    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
    parquet_read_options: ParquetReadOptions,
}

impl ArrowReaderBuilder {
    /// Create a new ArrowReaderBuilder
    pub fn new(file_io: FileIO) -> Self {
        let num_cpus = available_parallelism().get();

        ArrowReaderBuilder {
            batch_size: None,
            file_io,
            concurrency_limit_data_files: num_cpus,
            row_group_filtering_enabled: true,
            row_selection_enabled: false,
            parquet_read_options: ParquetReadOptions::builder().build(),
        }
    }

    /// Sets the max number of in flight data files that are being fetched
    pub fn with_data_file_concurrency_limit(mut self, val: usize) -> Self {
        self.concurrency_limit_data_files = val;
        self
    }

    /// Sets the desired size of batches in the response
    /// to something other than the default
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Determines whether to enable row group filtering.
    pub fn with_row_group_filtering_enabled(mut self, row_group_filtering_enabled: bool) -> Self {
        self.row_group_filtering_enabled = row_group_filtering_enabled;
        self
    }

    /// Determines whether to enable row selection.
    pub fn with_row_selection_enabled(mut self, row_selection_enabled: bool) -> Self {
        self.row_selection_enabled = row_selection_enabled;
        self
    }

    /// Provide a hint as to the number of bytes to prefetch for parsing the Parquet metadata
    ///
    /// This hint can help reduce the number of fetch requests. For more details see the
    /// [ParquetMetaDataReader documentation](https://docs.rs/parquet/latest/parquet/file/metadata/struct.ParquetMetaDataReader.html#method.with_prefetch_hint).
    pub fn with_metadata_size_hint(mut self, metadata_size_hint: usize) -> Self {
        self.parquet_read_options.metadata_size_hint = Some(metadata_size_hint);
        self
    }

    /// Sets the gap threshold for merging nearby byte ranges into a single request.
    /// Ranges with gaps smaller than this value will be coalesced.
    ///
    /// Defaults to 1 MiB, matching object_store's OBJECT_STORE_COALESCE_DEFAULT.
    pub fn with_range_coalesce_bytes(mut self, range_coalesce_bytes: u64) -> Self {
        self.parquet_read_options.range_coalesce_bytes = range_coalesce_bytes;
        self
    }

    /// Sets the maximum number of merged byte ranges to fetch concurrently.
    ///
    /// Defaults to 10, matching object_store's OBJECT_STORE_COALESCE_PARALLEL.
    pub fn with_range_fetch_concurrency(mut self, range_fetch_concurrency: usize) -> Self {
        self.parquet_read_options.range_fetch_concurrency = range_fetch_concurrency;
        self
    }

    /// Build the ArrowReader.
    pub fn build(self) -> ArrowReader {
        ArrowReader {
            batch_size: self.batch_size,
            file_io: self.file_io.clone(),
            delete_file_loader: CachingDeleteFileLoader::new(
                self.file_io.clone(),
                self.concurrency_limit_data_files,
            ),
            concurrency_limit_data_files: self.concurrency_limit_data_files,
            row_group_filtering_enabled: self.row_group_filtering_enabled,
            row_selection_enabled: self.row_selection_enabled,
            parquet_read_options: self.parquet_read_options,
        }
    }
}

/// Reads data from Parquet files
#[derive(Clone)]
pub struct ArrowReader {
    batch_size: Option<usize>,
    file_io: FileIO,
    delete_file_loader: CachingDeleteFileLoader,

    /// the maximum number of data files that can be fetched at the same time
    concurrency_limit_data_files: usize,

    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
    parquet_read_options: ParquetReadOptions,
}

impl ArrowReader {
    /// Take a stream of FileScanTasks and reads all the files.
    /// Returns a stream of Arrow RecordBatches containing the data from the files
    pub fn read(self, tasks: FileScanTaskStream) -> Result<ArrowRecordBatchStream> {
        let file_io = self.file_io.clone();
        let batch_size = self.batch_size;
        let concurrency_limit_data_files = self.concurrency_limit_data_files;
        let row_group_filtering_enabled = self.row_group_filtering_enabled;
        let row_selection_enabled = self.row_selection_enabled;
        let parquet_read_options = self.parquet_read_options;

        // Fast-path for single concurrency to avoid overhead of try_flatten_unordered
        let stream: ArrowRecordBatchStream = if concurrency_limit_data_files == 1 {
            Box::pin(
                tasks
                    .and_then(move |task| {
                        let file_io = file_io.clone();

                        Self::process_file_scan_task(
                            task,
                            batch_size,
                            file_io,
                            self.delete_file_loader.clone(),
                            row_group_filtering_enabled,
                            row_selection_enabled,
                            parquet_read_options,
                        )
                    })
                    .map_err(|err| {
                        Error::new(ErrorKind::Unexpected, "file scan task generate failed")
                            .with_source(err)
                    })
                    .try_flatten(),
            )
        } else {
            Box::pin(
                tasks
                    .map_ok(move |task| {
                        let file_io = file_io.clone();

                        Self::process_file_scan_task(
                            task,
                            batch_size,
                            file_io,
                            self.delete_file_loader.clone(),
                            row_group_filtering_enabled,
                            row_selection_enabled,
                            parquet_read_options,
                        )
                    })
                    .map_err(|err| {
                        Error::new(ErrorKind::Unexpected, "file scan task generate failed")
                            .with_source(err)
                    })
                    .try_buffer_unordered(concurrency_limit_data_files)
                    .try_flatten_unordered(concurrency_limit_data_files),
            )
        };

        Ok(stream)
    }

    /// Dispatches one [`FileScanTask`] on its data-file format, as Java `GenericReader` does.
    ///
    /// | Format | Reader | Filtering |
    /// |---|---|---|
    /// | `Parquet` | [`Self::process_parquet_file_scan_task`] | pushdown: row-group skip, `RowFilter`, `RowSelection`, byte-range split |
    /// | `Avro` | [`Self::process_avro_file_scan_task`] | whole file materialized, then filtered post-decode |
    /// | `Orc` | [`Self::process_orc_file_scan_task`] | whole file materialized, then filtered post-decode |
    /// | `Puffin` | none | `FeatureUnsupported`: a sidecar is never a data file |
    async fn process_file_scan_task(
        task: FileScanTask,
        batch_size: Option<usize>,
        file_io: FileIO,
        delete_file_loader: CachingDeleteFileLoader,
        row_group_filtering_enabled: bool,
        row_selection_enabled: bool,
        parquet_read_options: ParquetReadOptions,
    ) -> Result<ArrowRecordBatchStream> {
        Self::reject_variant_projection(&task)?;
        match task.data_file_format {
            DataFileFormat::Parquet => {
                Self::process_parquet_file_scan_task(
                    task,
                    batch_size,
                    file_io,
                    delete_file_loader,
                    row_group_filtering_enabled,
                    row_selection_enabled,
                    parquet_read_options,
                )
                .await
            }
            DataFileFormat::Avro => {
                Self::process_avro_file_scan_task(task, batch_size, file_io, delete_file_loader)
                    .await
            }
            DataFileFormat::Orc => {
                Self::process_orc_file_scan_task(task, batch_size, file_io, delete_file_loader)
                    .await
            }
            DataFileFormat::Puffin => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "A Puffin file ('{}') is a statistics / deletion-vector sidecar, not a \
                     scannable data file",
                    task.data_file_path
                ),
            )),
        }
    }

    /// Refuse a scan that projects a variant column, in any data-file format.
    ///
    /// File-level variant I/O is unimplemented. The guard sits ahead of the format dispatch,
    /// because the Iceberg-to-Arrow conversion no longer throws on `variant`.
    fn reject_variant_projection(task: &FileScanTask) -> Result<()> {
        for &field_id in task.project_field_ids() {
            let Some(field) = task.schema.field_by_id(field_id) else {
                continue;
            };
            if let Some(path) =
                crate::arrow::variant_path_within(&field.name, field.field_type.as_ref())
            {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!(
                        "Scanning the variant column '{path}' (under projected field id \
                         {field_id}) is not supported yet: file-level variant I/O is \
                         unimplemented (GAP_MATRIX row R88). The Iceberg→Arrow conversion of \
                         `variant` now succeeds, so this is a deliberate reader-side bound, not a \
                         schema failure."
                    ),
                ));
            }
        }
        Ok(())
    }

    async fn process_parquet_file_scan_task(
        task: FileScanTask,
        batch_size: Option<usize>,
        file_io: FileIO,
        delete_file_loader: CachingDeleteFileLoader,
        row_group_filtering_enabled: bool,
        row_selection_enabled: bool,
        parquet_read_options: ParquetReadOptions,
    ) -> Result<ArrowRecordBatchStream> {
        let should_load_page_index =
            (row_selection_enabled && task.predicate.is_some()) || !task.deletes.is_empty();
        let mut parquet_read_options = parquet_read_options;
        parquet_read_options.preload_page_index = should_load_page_index;

        let delete_filter_rx =
            delete_file_loader.load_deletes(&task.deletes, Arc::clone(&task.schema));

        let (parquet_file_reader, arrow_metadata) = Self::open_parquet_file(
            &task.data_file_path,
            &file_io,
            task.file_size_in_bytes,
            parquet_read_options,
        )
        .await?;

        // Java `ParquetSchemaUtil.hasIds()`.
        let missing_field_ids = arrow_metadata
            .schema()
            .fields()
            .iter()
            .next()
            .is_some_and(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none());

        // Projection reads columns by field id (spec, Column Projection). A file written by a
        // Hive or Spark migration carries no field ids, so this must assign them before the read.
        // Java `ReadConf` picks one of three branches:
        //
        // | Branch | Condition | Java |
        // |---|---|---|
        // | 1 | the file has field ids | `pruneColumns()` |
        // | 2 | a name mapping exists | `applyNameMapping()` then `pruneColumns()` |
        // | 3 | neither | `addFallbackIds()` then `pruneColumnsFallback()` |
        let arrow_metadata = if missing_field_ids {
            // The file has no field ids, so assign them before the read.
            let arrow_schema = if let Some(name_mapping) = &task.name_mapping {
                // Branch 2: Java `ParquetSchemaUtil.applyNameMapping()`.
                apply_name_mapping_to_arrow_schema(
                    Arc::clone(arrow_metadata.schema()),
                    name_mapping,
                )?
            } else {
                // Branch 3: Java `ParquetSchemaUtil.addFallbackIds()`, position-based.
                add_fallback_field_ids_to_arrow_schema(arrow_metadata.schema())
            };

            let options = ArrowReaderOptions::new().with_schema(arrow_schema);
            ArrowReaderMetadata::try_new(Arc::clone(arrow_metadata.metadata()), options).map_err(
                |e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Failed to create ArrowReaderMetadata with field ID schema",
                    )
                    .with_source(e)
                },
            )?
        } else {
            // Branch 1: the file carries field ids.
            arrow_metadata
        };

        // Position-based projection applies to Branch 3 only. Branch 2 stamps real field ids, so
        // it must project by field id. A positional projection there ignores the mapping and reads
        // the wrong columns.
        let use_position_fallback = missing_field_ids && task.name_mapping.is_none();

        // Coerce INT96 timestamps before the stream reader is built, or arrow-rs overflows i64.
        let arrow_metadata = if let Some(coerced_schema) =
            coerce_int96_timestamps(arrow_metadata.schema(), &task.schema)
        {
            let options = ArrowReaderOptions::new().with_schema(Arc::clone(&coerced_schema));
            ArrowReaderMetadata::try_new(Arc::clone(arrow_metadata.metadata()), options).map_err(
                |e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "Failed to create ArrowReaderMetadata with INT96-coerced schema: {coerced_schema}"
                        ),
                    )
                    .with_source(e)
                },
            )?
        } else {
            arrow_metadata
        };

        let mut record_batch_stream_builder =
            ParquetRecordBatchStreamBuilder::new_with_metadata(parquet_file_reader, arrow_metadata);

        // Metadata fields are not in the file. The V3 row-lineage pair is the exception: it can be
        // stored, and Java prefers the stored value.
        let project_field_ids_without_metadata: Vec<i32> = task
            .project_field_ids
            .iter()
            .filter(|&&id| !is_metadata_field(id) || is_row_lineage_field(id))
            .copied()
            .collect();

        // Only fallback ids project by position. Both other branches project by field id.
        let projection_mask = Self::get_arrow_projection_mask(
            &project_field_ids_without_metadata,
            &task.schema,
            record_batch_stream_builder.parquet_schema(),
            record_batch_stream_builder.schema(),
            use_position_fallback, // position-based (true) only for id-less files with NO name mapping
        )?;

        record_batch_stream_builder =
            record_batch_stream_builder.with_projection(projection_mask.clone());

        // A `_pos` projection needs each row's true physical ordinal, to write position deletes.
        // `RowSelection` skips rows at the decode layer and loses those ordinals, so this path
        // decodes in order with no RowFilter, RowSelection, or row-group prune. Batches still
        // stream, so memory stays O(batch). `_row_id` needs the same, because its fallback is
        // `first_row_id + pos`. A scan that does not project either keeps full pushdown.
        let needs_physical_ordinals = task.project_field_ids().contains(&RESERVED_FIELD_ID_POS)
            || task.project_field_ids().contains(&RESERVED_FIELD_ID_ROW_ID);
        if needs_physical_ordinals {
            // This path decodes the whole file with ordinals from 0. A ranged split task would
            // re-emit every row per split, with wrong `_pos`, which corrupts written position
            // deletes. A whole-file task carries `start == 0` and either `length == 0` or
            // `length == file_size_in_bytes`. Reject anything else loudly. The guard covers the
            // public `PartitionWork` and direct-reader seams.
            let whole_file =
                task.start == 0 && (task.length == 0 || task.length == task.file_size_in_bytes);
            if !whole_file {
                // Name the column that forced this path. A `_row_id` scan must not be told to
                // drop `_pos`, which it never projected.
                let projected_columns = if task.project_field_ids().contains(&RESERVED_FIELD_ID_POS)
                {
                    if task.project_field_ids().contains(&RESERVED_FIELD_ID_ROW_ID) {
                        "`_pos` and `_row_id`"
                    } else {
                        "`_pos`"
                    }
                } else {
                    "`_row_id`"
                };
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!(
                        "{projected_columns} projection over a ranged split task is unsupported: \
                         task covers {}..{} of {} bytes of '{}', but this path decodes whole \
                         files with ordinals from 0 (each split would duplicate every row). Plan \
                         without splitting, or drop {projected_columns} from the projection.",
                        task.start,
                        task.start.saturating_add(task.length),
                        task.file_size_in_bytes,
                        task.data_file_path
                    ),
                ));
            }
            if let Some(batch_size) = batch_size {
                record_batch_stream_builder =
                    record_batch_stream_builder.with_batch_size(batch_size);
            }
            let parquet_stream = record_batch_stream_builder.build()?;
            return Self::stream_pos_projection_scan_task(task, parquet_stream, delete_filter_rx)
                .await;
        }

        // RecordBatchTransformer applies type promotion, defaults, reordering, partition
        // constants, and virtual fields such as `_file`.
        let mut record_batch_transformer_builder =
            RecordBatchTransformerBuilder::new(task.schema_ref(), task.project_field_ids())
                .with_row_lineage(task.first_row_id, task.file_sequence_number);

        if task.project_field_ids().contains(&RESERVED_FIELD_ID_FILE) {
            let file_datum = Datum::string(task.data_file_path.clone());
            record_batch_transformer_builder =
                record_batch_transformer_builder.with_constant(RESERVED_FIELD_ID_FILE, file_datum);
        }

        if let (Some(partition_spec), Some(partition_data)) =
            (task.partition_spec.clone(), task.partition.clone())
        {
            record_batch_transformer_builder =
                record_batch_transformer_builder.with_partition(partition_spec, partition_data)?;
        }

        let mut record_batch_transformer = record_batch_transformer_builder.build();

        if let Some(batch_size) = batch_size {
            record_batch_stream_builder = record_batch_stream_builder.with_batch_size(batch_size);
        }

        let delete_filter = delete_filter_rx.await.map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "delete-filter task was dropped before sending the filter",
            )
            .with_source(e)
        })??;

        // Equality-delete routing. Take the O(R) keyset fast path only when every eq-delete file
        // is type-eligible under one key schema, and every key field id is projected. The
        // RowFilter residual is then the scan predicate alone, and eq-deletes apply after the
        // transformer. Otherwise AND the eq-delete predicate into the RowFilter, which can still
        // read key columns the data projection omits.
        let eq_delete_sets = delete_filter.collect_equality_delete_keysets(&task).await;
        let delete_predicate = delete_filter.build_equality_delete_predicate(&task).await?;
        let keyset_post_decode = eq_delete_sets.as_ref().is_some_and(|sets| {
            !sets.is_empty()
                && eq_delete_key_fields_projected(sets, &project_field_ids_without_metadata)
        });

        // The residual pushed into RowFilter, row-group skip, and page selection. It carries the
        // eq-delete predicate only when the keyset path is not taken.
        let (final_predicate, post_decode_eq_sets, post_decode_eq_predicate) = if keyset_post_decode
        {
            (
                task.predicate.as_deref().cloned(),
                eq_delete_sets,
                delete_predicate,
            )
        } else {
            let final_predicate = match (task.predicate.as_deref(), delete_predicate) {
                (None, None) => None,
                (Some(predicate), None) => Some(predicate.clone()),
                (None, Some(predicate)) => Some(predicate),
                (Some(filter_predicate), Some(delete_predicate)) => {
                    Some(filter_predicate.clone().and(delete_predicate))
                }
            };
            (final_predicate, None, None)
        };

        // Row-group selection has three sources: a byte range from a split task, applicable
        // equality deletes, and a scan predicate when row-group filtering is on. `RowSelection`
        // has two: applicable positional deletes, and a scan predicate when row selection is on.
        // Positional deletes only apply through a `RowSelection`, so that path runs whenever
        // deletes exist, even with predicate filtering off.
        let mut selected_row_group_indices = None;
        let mut row_selection = None;

        // Filter row groups based on byte range from task.start and task.length.
        // If both start and length are 0, read the entire file (backwards compatibility).
        if task.start != 0 || task.length != 0 {
            let byte_range_filtered_row_groups = Self::filter_row_groups_by_byte_range(
                record_batch_stream_builder.metadata(),
                task.start,
                task.length,
            )?;
            selected_row_group_indices = Some(byte_range_filtered_row_groups);
        }

        if let Some(predicate) = final_predicate {
            let (iceberg_field_ids, field_id_map) = Self::build_field_id_set_and_map(
                record_batch_stream_builder.parquet_schema(),
                &predicate,
            )?;

            let row_filter = Self::get_row_filter(
                &predicate,
                record_batch_stream_builder.parquet_schema(),
                &iceberg_field_ids,
                &field_id_map,
            )?;
            record_batch_stream_builder = record_batch_stream_builder.with_row_filter(row_filter);

            if row_group_filtering_enabled {
                let predicate_filtered_row_groups = Self::get_selected_row_group_indices(
                    &predicate,
                    record_batch_stream_builder.metadata(),
                    &field_id_map,
                    &task.schema,
                )?;

                selected_row_group_indices = match selected_row_group_indices {
                    Some(byte_range_filtered) => {
                        let intersection: Vec<usize> = byte_range_filtered
                            .into_iter()
                            .filter(|idx| predicate_filtered_row_groups.contains(idx))
                            .collect();
                        Some(intersection)
                    }
                    None => Some(predicate_filtered_row_groups),
                };
            }

            if row_selection_enabled {
                row_selection = Some(Self::get_row_selection_for_filter_predicate(
                    &predicate,
                    record_batch_stream_builder.metadata(),
                    &selected_row_group_indices,
                    &field_id_map,
                    &task.schema,
                )?);
            }
        }

        let positional_delete_indexes = delete_filter.get_delete_vector(&task);

        if let Some(positional_delete_indexes) = positional_delete_indexes {
            // The frozen `Arc<DeleteVector>` needs no mutex: row selection reads the bitmap.
            let delete_row_selection = Self::build_deletes_row_selection(
                record_batch_stream_builder.metadata().row_groups(),
                &selected_row_group_indices,
                positional_delete_indexes.as_ref(),
            )?;

            row_selection = match row_selection {
                None => Some(delete_row_selection),
                Some(filter_row_selection) => {
                    Some(filter_row_selection.intersection(&delete_row_selection))
                }
            };
        }

        if let Some(row_selection) = row_selection {
            record_batch_stream_builder =
                record_batch_stream_builder.with_row_selection(row_selection);
        }

        if let Some(selected_row_group_indices) = selected_row_group_indices {
            record_batch_stream_builder =
                record_batch_stream_builder.with_row_groups(selected_row_group_indices);
        }

        // With `keyset_post_decode` set, eq-deletes apply here, after the transform, rather than
        // through the RowFilter residual above.
        let record_batch_stream =
            record_batch_stream_builder
                .build()?
                .map(move |batch| match batch {
                    Ok(batch) => {
                        let transformed = record_batch_transformer.process_record_batch(batch)?;
                        if post_decode_eq_sets.is_none() && post_decode_eq_predicate.is_none() {
                            return Ok(transformed);
                        }
                        // Same routing as `survival_mask`: keysets first, the bound predicate on
                        // a NULL-key batch.
                        match Self::eq_delete_keep_mask(
                            &transformed,
                            transformed.num_rows(),
                            post_decode_eq_predicate.as_ref(),
                            post_decode_eq_sets.as_deref(),
                        )? {
                            None => Ok(transformed),
                            Some(mask) => filter_record_batch(&transformed, &mask).map_err(|e| {
                                Error::new(
                                    ErrorKind::Unexpected,
                                    "Failed to apply equality-delete keyset keep-mask to a Parquet data batch",
                                )
                                .with_source(e)
                            }),
                        }
                    }
                    Err(err) => Err(err.into()),
                });

        Ok(Box::pin(record_batch_stream) as ArrowRecordBatchStream)
    }

    /// Fail closed on a task carrying a real byte sub-window for a format whose reader materializes
    /// WHOLE files. The Avro and ORC readers ignore `task.start` and `task.length` and decode the
    /// whole file. A ranged sub-task therefore re-emits every row, and an N-way split returns N
    /// copies with no error.
    fn reject_ranged_whole_file_task(task: &FileScanTask, format: &str) -> Result<()> {
        let whole_file =
            task.start == 0 && (task.length == 0 || task.length == task.file_size_in_bytes);
        if whole_file {
            return Ok(());
        }
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            format!(
                "a ranged split task over a {format} data file is unsupported: task covers {}..{} \
                 of {} bytes of '{}', but the {format} reader decodes whole files (each split \
                 would re-emit every row). Plan without splitting this file.",
                task.start,
                task.start.saturating_add(task.length),
                task.file_size_in_bytes,
                task.data_file_path
            ),
        ))
    }

    /// Read one **Avro** data-file scan task into an [`ArrowRecordBatchStream`]. Avro has no
    /// footer, statistics, or row groups, so there is no pushdown. The reader materializes the file
    /// and applies every filter after the decode.
    async fn process_avro_file_scan_task(
        task: FileScanTask,
        batch_size: Option<usize>,
        file_io: FileIO,
        delete_file_loader: CachingDeleteFileLoader,
    ) -> Result<ArrowRecordBatchStream> {
        Self::reject_ranged_whole_file_task(&task, "AVRO")?;

        let delete_filter_rx =
            delete_file_loader.load_deletes(&task.deletes, Arc::clone(&task.schema));

        // The projected field ids present in the file, plus the V3 row-lineage pair, whose stored
        // value wins. The transformer re-adds the rest.
        let expected = Self::build_expected_schema(&task)?;

        // The reader requires a positive batch size, so fall back to the arrow-rs default 1024.
        let avro_batch_size = batch_size.unwrap_or(1024).max(1);
        let input_file = file_io.new_input(&task.data_file_path)?;
        let batches = read_avro_data_file(&input_file, expected, avro_batch_size).await?;

        // The tail after the decode is shared with the ORC path.
        Self::finish_whole_file_scan_task(task, batches, delete_filter_rx).await
    }

    /// Read one **ORC** data-file scan task into an [`ArrowRecordBatchStream`].
    ///
    /// Same shape as [`Self::process_avro_file_scan_task`]. Only the decode differs: it calls
    /// [`read_orc_data_file`] (Java `GenericOrcReader`). The rest is the format-agnostic
    /// [`Self::finish_whole_file_scan_task`] tail. This reader pushes no predicate into the ORC
    /// stripe metadata, so every filter applies after the decode.
    async fn process_orc_file_scan_task(
        task: FileScanTask,
        batch_size: Option<usize>,
        file_io: FileIO,
        delete_file_loader: CachingDeleteFileLoader,
    ) -> Result<ArrowRecordBatchStream> {
        Self::reject_ranged_whole_file_task(&task, "ORC")?;

        let delete_filter_rx =
            delete_file_loader.load_deletes(&task.deletes, Arc::clone(&task.schema));

        // Shared with the Avro path through `build_expected_schema`.
        let expected = Self::build_expected_schema(&task)?;

        // The reader requires a positive batch size, so fall back to the arrow-rs default 1024.
        let orc_batch_size = batch_size.unwrap_or(1024).max(1);
        let input_file = file_io.new_input(&task.data_file_path)?;
        let batches = read_orc_data_file(&input_file, expected, orc_batch_size).await?;

        // The shared tail. See [`Self::finish_whole_file_scan_task`].
        Self::finish_whole_file_scan_task(task, batches, delete_filter_rx).await
    }

    /// The Parquet path when `_pos` is projected. It decodes in physical order with no
    /// `RowFilter`, `RowSelection`, or row-group prune, and streams batches through
    /// [`Self::apply_pos_aware_batch`], so memory stays O(batch).
    async fn stream_pos_projection_scan_task<S>(
        task: FileScanTask,
        parquet_stream: S,
        delete_filter_rx: tokio::sync::oneshot::Receiver<
            Result<crate::arrow::delete_filter::DeleteFilter>,
        >,
    ) -> Result<ArrowRecordBatchStream>
    where
        S: Stream<Item = parquet::errors::Result<RecordBatch>> + Send + 'static,
    {
        let mut record_batch_transformer = Self::build_scan_task_transformer(&task)?;
        let (positional_deletes, residual_predicate, eq_delete_predicate, eq_delete_sets) =
            Self::resolve_whole_file_delete_context(&task, delete_filter_rx).await?;

        let mut absolute_pos: u64 = 0;
        let record_batch_stream = parquet_stream.map(move |batch_result| {
            let batch = batch_result.map_err(Error::from)?;
            Self::apply_pos_aware_batch(
                batch,
                &mut record_batch_transformer,
                &mut absolute_pos,
                positional_deletes.as_ref(),
                residual_predicate.as_deref(),
                eq_delete_predicate.as_ref(),
                eq_delete_sets.as_deref(),
            )
        });

        Ok(Box::pin(record_batch_stream) as ArrowRecordBatchStream)
    }

    /// The format-agnostic tail shared by the Avro and ORC paths, run once a whole-file reader has
    /// materialized `batches`. It builds the same [`RecordBatchTransformer`] the Parquet path
    /// feeds, ANDs the equality-delete predicate with the scan residual into one survival
    /// predicate, and applies merge-on-read deletes after materialization. The per-batch apply is
    /// [`Self::apply_pos_aware_batch`], shared with the Parquet `_pos` streaming path.
    async fn finish_whole_file_scan_task(
        task: FileScanTask,
        batches: Vec<RecordBatch>,
        delete_filter_rx: tokio::sync::oneshot::Receiver<
            Result<crate::arrow::delete_filter::DeleteFilter>,
        >,
    ) -> Result<ArrowRecordBatchStream> {
        let mut record_batch_transformer = Self::build_scan_task_transformer(&task)?;
        let (positional_deletes, residual_predicate, eq_delete_predicate, eq_delete_sets) =
            Self::resolve_whole_file_delete_context(&task, delete_filter_rx).await?;

        // The file is already decoded, so an eager loop costs nothing here.
        let mut output: Vec<Result<RecordBatch>> = Vec::with_capacity(batches.len());
        let mut absolute_pos: u64 = 0;
        for batch in batches {
            match Self::apply_pos_aware_batch(
                batch,
                &mut record_batch_transformer,
                &mut absolute_pos,
                positional_deletes.as_ref(),
                residual_predicate.as_deref(),
                eq_delete_predicate.as_ref(),
                eq_delete_sets.as_deref(),
            ) {
                Ok(b) => output.push(Ok(b)),
                Err(e) => {
                    output.push(Err(e));
                    break;
                }
            }
        }

        Ok(Box::pin(futures::stream::iter(output)) as ArrowRecordBatchStream)
    }

    /// Build the [`RecordBatchTransformer`] shared by Parquet `_pos` streaming and Avro/ORC
    /// whole-file tails (schema evolution, reorder, `_file` / identity-partition constants).
    fn build_scan_task_transformer(task: &FileScanTask) -> Result<RecordBatchTransformer> {
        let mut record_batch_transformer_builder =
            RecordBatchTransformerBuilder::new(task.schema_ref(), task.project_field_ids())
                .with_row_lineage(task.first_row_id, task.file_sequence_number);
        if task.project_field_ids().contains(&RESERVED_FIELD_ID_FILE) {
            let file_datum = Datum::string(task.data_file_path.clone());
            record_batch_transformer_builder =
                record_batch_transformer_builder.with_constant(RESERVED_FIELD_ID_FILE, file_datum);
        }
        if let (Some(partition_spec), Some(partition_data)) =
            (task.partition_spec.clone(), task.partition.clone())
        {
            record_batch_transformer_builder =
                record_batch_transformer_builder.with_partition(partition_spec, partition_data)?;
        }
        Ok(record_batch_transformer_builder.build())
    }

    /// Resolve positional delete vector + residual + eq-delete set/predicate for whole-file /
    /// `_pos` post-decode apply. Delete load was started concurrently with the data read.
    async fn resolve_whole_file_delete_context(
        task: &FileScanTask,
        delete_filter_rx: tokio::sync::oneshot::Receiver<
            Result<crate::arrow::delete_filter::DeleteFilter>,
        >,
    ) -> Result<(
        Option<Arc<DeleteVector>>,
        Option<Arc<BoundPredicate>>,
        Option<BoundPredicate>,
        Option<Vec<EqDeleteKeySet>>,
    )> {
        let delete_filter = delete_filter_rx.await.map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "delete-file loader dropped before delivering the delete filter",
            )
            .with_source(e)
        })??;
        // When every eq-delete file is type-eligible under one key schema, hashed set membership
        // applies the deletes in O(R) instead of O(E·R). A batch with a NULL key column falls back
        // to the eq-delete predicate, so both stay available.
        let eq_delete_sets = delete_filter.collect_equality_delete_keysets(task).await;
        let eq_delete_predicate = delete_filter.build_equality_delete_predicate(task).await?;
        let residual_predicate = task.predicate.clone();
        let positional_deletes = delete_filter.get_delete_vector(task);
        Ok((
            positional_deletes,
            residual_predicate,
            eq_delete_predicate,
            eq_delete_sets,
        ))
    }

    /// Transform one decoded batch, assign `_pos` from the running physical counter (via the
    /// transformer), apply MoR survival (positional / residual / eq), advance `absolute_pos` by the
    /// **full pre-filter** row count. Shared by Parquet `_pos` streaming and Avro/ORC whole-file. #
    /// Notes `absolute_pos` and the transformer's `next_row_position` must track the same physical
    /// ordinal base.
    fn apply_pos_aware_batch(
        batch: RecordBatch,
        transformer: &mut RecordBatchTransformer,
        absolute_pos: &mut u64,
        positional_deletes: Option<&Arc<DeleteVector>>,
        residual_predicate: Option<&BoundPredicate>,
        eq_delete_predicate: Option<&BoundPredicate>,
        eq_delete_sets: Option<&[EqDeleteKeySet]>,
    ) -> Result<RecordBatch> {
        let row_count = batch.num_rows();
        let batch_base = *absolute_pos;
        let transformed = transformer.process_record_batch(batch)?;
        // `absolute_pos` and the transformer's `next_row_position` must stay aligned. Under a
        // `_pos` projection the first ordinal in the batch equals `batch_base`. A desync corrupts
        // written position deletes.
        debug_assert!(
            {
                use arrow_array::Int64Array;

                use crate::metadata_columns::RESERVED_COL_NAME_POS;
                match transformed.column_by_name(RESERVED_COL_NAME_POS) {
                    Some(col) if row_count > 0 => col
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .is_some_and(|a| a.value(0) as u64 == batch_base),
                    _ => true,
                }
            },
            "absolute_pos desynced from transformer _pos (batch_base={batch_base}, rows={row_count})"
        );
        let mask = Self::survival_mask(
            &transformed,
            row_count,
            batch_base,
            positional_deletes,
            residual_predicate,
            eq_delete_predicate,
            eq_delete_sets,
        )?;
        // Advance by the full batch before any mask filter, so the next batch's ordinals follow.
        *absolute_pos = absolute_pos.saturating_add(row_count as u64);
        match mask {
            None => Ok(transformed),
            Some(mask) => filter_record_batch(&transformed, &mask).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to apply merge-on-read deletes to a data batch under _pos / whole-file scan",
                )
                .with_source(e)
            }),
        }
    }

    /// The projected Iceberg [`Schema`] a whole-file reader resolves against: the projected field
    /// ids present in the file, plus the V3 row-lineage pair, whose stored value wins. Every other
    /// reserved column is synthesized. Field order follows the projection.
    fn build_expected_schema(task: &FileScanTask) -> Result<Arc<Schema>> {
        let mut fields = Vec::new();
        for &field_id in task.project_field_ids() {
            // The row-lineage pair is the one stored metadata pair, and the stored value wins.
            let field = if is_row_lineage_field(field_id) {
                get_metadata_field(field_id)?.clone()
            } else if is_metadata_field(field_id) {
                continue;
            } else {
                task.schema
                    .field_by_id(field_id)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Projected field id {field_id} is not present in the scan schema \
                                 for data file '{}'",
                                task.data_file_path
                            ),
                        )
                    })?
                    .clone()
            };
            fields.push(field);
        }
        let schema = Schema::builder()
            .with_schema_id(task.schema.schema_id())
            .with_fields(fields)
            .build()?;
        Ok(Arc::new(schema))
    }

    /// Builds the per-row survival mask for a transformed batch, from the positional deletes over
    /// `[batch_base, batch_base + num_rows)`, the scan residual, and the equality deletes. Returns
    /// `None` when nothing applies, else a mask where `true` keeps the row. `eq_delete_sets`, when
    /// `Some`, holds the hashed key sets for the task's eq-delete files.
    fn survival_mask(
        batch: &RecordBatch,
        num_rows: usize,
        batch_base: u64,
        positional_deletes: Option<&Arc<DeleteVector>>,
        residual_predicate: Option<&BoundPredicate>,
        eq_delete_predicate: Option<&BoundPredicate>,
        eq_delete_sets: Option<&[EqDeleteKeySet]>,
    ) -> Result<Option<BooleanArray>> {
        // Positional deletes give a keep-mask of `!deleted` over this batch's position window.
        // The memoized vector is frozen, so the apply path takes no lock.
        let positional_mask: Option<BooleanArray> = match positional_deletes {
            Some(deletes) => {
                if deletes.is_empty() {
                    None
                } else {
                    // The range walk equals the per-row `!contains` probe, in O(D_window).
                    Some(positional_delete_keep_mask(
                        deletes.as_ref(),
                        batch_base,
                        num_rows,
                    ))
                }
            }
            None => None,
        };

        // The mask is already two-valued under Java nulls-first semantics. The coercion is
        // defense in depth.
        let predicate_keep = |predicate: &BoundPredicate| -> Result<BooleanArray> {
            Ok(coerce_nulls_to_false(&evaluate_predicate_to_mask(
                predicate, batch,
            )?))
        };

        // Scan residual (`task.predicate`) → always via the predicate path.
        let residual_mask: Option<BooleanArray> = match residual_predicate {
            Some(predicate) => Some(predicate_keep(predicate)?),
            None => None,
        };

        // Equality-delete keep-mask (shared with the Parquet pushdown post-decode keyset path).
        let eq_delete_mask =
            Self::eq_delete_keep_mask(batch, num_rows, eq_delete_predicate, eq_delete_sets)?;

        // AND the present keep-masks together.
        let combine =
            |a: Option<BooleanArray>, b: Option<BooleanArray>| -> Result<Option<BooleanArray>> {
                match (a, b) {
                    (None, None) => Ok(None),
                    (Some(m), None) | (None, Some(m)) => Ok(Some(m)),
                    (Some(x), Some(y)) => Ok(Some(and(&x, &y).map_err(|e| {
                        Error::new(
                            ErrorKind::Unexpected,
                            "Failed to combine merge-on-read delete masks for a data batch",
                        )
                        .with_source(e)
                    })?)),
                }
            };
        let combined = combine(positional_mask, residual_mask)?;
        combine(combined, eq_delete_mask)
    }

    /// The equality-delete keep-mask for one transformed batch. It uses the O(R)
    /// [`EqDeleteKeySet`] path, and falls back to the bound predicate for the whole batch when any
    /// set reports a NULL key column. Returns `None` when no eq-deletes apply.
    fn eq_delete_keep_mask(
        batch: &RecordBatch,
        num_rows: usize,
        eq_delete_predicate: Option<&BoundPredicate>,
        eq_delete_sets: Option<&[EqDeleteKeySet]>,
    ) -> Result<Option<BooleanArray>> {
        // A NULL key column in any set sends the whole batch to the predicate.
        let mut from_sets: Option<BooleanArray> = None;
        if let Some(sets) = eq_delete_sets.filter(|s| !s.is_empty()) {
            let mut keep = vec![true; num_rows];
            let mut all_sets_safe = true;
            for set in sets {
                // Call `delete_mask` even on an empty set. The I64 store drops null delete cells,
                // so a null-only eq-delete file reports empty but must still bail to the
                // predicate, or `col IS NULL` deletes never apply.
                match set.delete_mask(batch)? {
                    Some(deleted) => {
                        for (k, d) in keep.iter_mut().zip(deleted.iter()) {
                            *k &= !*d;
                        }
                    }
                    None => {
                        all_sets_safe = false;
                        break;
                    }
                }
            }
            if all_sets_safe {
                from_sets = Some(BooleanArray::from(keep));
            }
        }
        match from_sets {
            Some(mask) => Ok(Some(mask)),
            // No usable set, so use the predicate.
            None => match eq_delete_predicate {
                Some(predicate) => Ok(Some(coerce_nulls_to_false(&evaluate_predicate_to_mask(
                    predicate, batch,
                )?))),
                None => Ok(None),
            },
        }
    }

    /// Opens a Parquet file and loads its metadata, returning both the reader and metadata.
    /// The reader can be reused to build a `ParquetRecordBatchStreamBuilder` without
    /// reopening the file.
    pub(crate) async fn open_parquet_file(
        data_file_path: &str,
        file_io: &FileIO,
        file_size_in_bytes: u64,
        parquet_read_options: ParquetReadOptions,
    ) -> Result<(ArrowFileReader, ArrowReaderMetadata)> {
        let parquet_file = file_io.new_input(data_file_path)?;
        let parquet_reader = parquet_file.reader().await?;
        let mut reader = ArrowFileReader::new(
            FileMetadata {
                size: file_size_in_bytes,
            },
            parquet_reader,
        )
        .with_parquet_read_options(parquet_read_options);

        let arrow_metadata = ArrowReaderMetadata::load_async(&mut reader, Default::default())
            .await
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "Failed to load Parquet metadata").with_source(e)
            })?;

        Ok((reader, arrow_metadata))
    }

    /// computes a `RowSelection` from positional delete indices.
    ///
    /// Using the Parquet page index, we build a `RowSelection` that rejects rows that are indicated
    /// as having been deleted by a positional delete, taking into account any row groups that have
    /// been skipped entirely by the filter predicate
    fn build_deletes_row_selection(
        row_group_metadata_list: &[RowGroupMetaData],
        selected_row_groups: &Option<Vec<usize>>,
        positional_deletes: &DeleteVector,
    ) -> Result<RowSelection> {
        let mut results: Vec<RowSelector> = Vec::new();
        let mut selected_row_groups_idx = 0;
        let mut current_row_group_base_idx: u64 = 0;
        let mut delete_vector_iter = positional_deletes.iter();
        let mut next_deleted_row_idx_opt = delete_vector_iter.next();

        for (idx, row_group_metadata) in row_group_metadata_list.iter().enumerate() {
            let row_group_num_rows = row_group_metadata.num_rows() as u64;
            let next_row_group_base_idx = current_row_group_base_idx + row_group_num_rows;

            if let Some(selected_row_groups) = selected_row_groups {
                if selected_row_groups_idx == selected_row_groups.len() {
                    break;
                }

                if idx == selected_row_groups[selected_row_groups_idx] {
                    selected_row_groups_idx += 1;
                } else {
                    // Advance iterator past all deletes in the skipped row group.
                    // advance_to() positions the iterator to the first delete >= next_row_group_base_idx.
                    // However, if our cached next_deleted_row_idx_opt is in the skipped range,
                    // we need to call next() to update the cache with the newly positioned value.
                    delete_vector_iter.advance_to(next_row_group_base_idx);
                    // Only update the cache if the cached value is stale (in the skipped range)
                    if let Some(cached_idx) = next_deleted_row_idx_opt
                        && cached_idx < next_row_group_base_idx
                    {
                        next_deleted_row_idx_opt = delete_vector_iter.next();
                    }

                    current_row_group_base_idx += row_group_num_rows;
                    continue;
                }
            }

            let mut next_deleted_row_idx = match next_deleted_row_idx_opt {
                Some(next_deleted_row_idx) => {
                    if next_deleted_row_idx >= next_row_group_base_idx {
                        results.push(RowSelector::select(row_group_num_rows as usize));
                        current_row_group_base_idx += row_group_num_rows;
                        continue;
                    }

                    next_deleted_row_idx
                }

                _ => {
                    results.push(RowSelector::select(row_group_num_rows as usize));
                    current_row_group_base_idx += row_group_num_rows;
                    continue;
                }
            };

            let mut current_idx = current_row_group_base_idx;
            'chunks: while next_deleted_row_idx < next_row_group_base_idx {
                if current_idx < next_deleted_row_idx {
                    let run_length = next_deleted_row_idx - current_idx;
                    results.push(RowSelector::select(run_length as usize));
                    current_idx += run_length;
                }

                let mut run_length = 0;
                while next_deleted_row_idx == current_idx
                    && next_deleted_row_idx < next_row_group_base_idx
                {
                    run_length += 1;
                    current_idx += 1;

                    next_deleted_row_idx_opt = delete_vector_iter.next();
                    next_deleted_row_idx = match next_deleted_row_idx_opt {
                        Some(next_deleted_row_idx) => next_deleted_row_idx,
                        _ => {
                            results.push(RowSelector::skip(run_length));
                            break 'chunks;
                        }
                    };
                }
                if run_length > 0 {
                    results.push(RowSelector::skip(run_length));
                }
            }

            if current_idx < next_row_group_base_idx {
                results.push(RowSelector::select(
                    (next_row_group_base_idx - current_idx) as usize,
                ));
            }

            current_row_group_base_idx += row_group_num_rows;
        }

        Ok(results.into())
    }

    fn build_field_id_set_and_map(
        parquet_schema: &SchemaDescriptor,
        predicate: &BoundPredicate,
    ) -> Result<(HashSet<i32>, HashMap<i32, usize>)> {
        let mut collector = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut collector, predicate)?;

        let iceberg_field_ids = collector.field_ids();

        // Without embedded field IDs, we fall back to position-based mapping for compatibility
        let field_id_map = match build_field_id_map(parquet_schema)? {
            Some(map) => map,
            None => build_fallback_field_id_map(parquet_schema),
        };

        Ok((iceberg_field_ids, field_id_map))
    }

    /// Recursively extract leaf field IDs because Parquet projection works at the leaf column level.
    /// Nested types (struct/list/map) are flattened in Parquet's columnar format.
    fn include_leaf_field_id(field: &NestedField, field_ids: &mut Vec<i32>) {
        match field.field_type.as_ref() {
            Type::Primitive(_) => {
                field_ids.push(field.id);
            }
            // A variant column is a leaf. Reading variant data is deferred, and the door is
            // `reject_variant_projection`, not this conversion.
            Type::Variant => {
                field_ids.push(field.id);
            }
            Type::Struct(struct_type) => {
                for nested_field in struct_type.fields() {
                    Self::include_leaf_field_id(nested_field, field_ids);
                }
            }
            Type::List(list_type) => {
                Self::include_leaf_field_id(&list_type.element_field, field_ids);
            }
            Type::Map(map_type) => {
                Self::include_leaf_field_id(&map_type.key_field, field_ids);
                Self::include_leaf_field_id(&map_type.value_field, field_ids);
            }
        }
    }

    fn get_arrow_projection_mask(
        field_ids: &[i32],
        iceberg_schema_of_task: &Schema,
        parquet_schema: &SchemaDescriptor,
        arrow_schema: &ArrowSchemaRef,
        use_fallback: bool, // Whether file lacks embedded field IDs (e.g., migrated from Hive/Spark)
    ) -> Result<ProjectionMask> {
        fn type_promotion_is_valid(
            file_type: Option<&PrimitiveType>,
            projected_type: Option<&PrimitiveType>,
        ) -> bool {
            match (file_type, projected_type) {
                (Some(lhs), Some(rhs)) if lhs == rhs => true,
                (Some(PrimitiveType::Int), Some(PrimitiveType::Long)) => true,
                (Some(PrimitiveType::Float), Some(PrimitiveType::Double)) => true,
                (
                    Some(PrimitiveType::Decimal {
                        precision: file_precision,
                        scale: file_scale,
                    }),
                    Some(PrimitiveType::Decimal {
                        precision: requested_precision,
                        scale: requested_scale,
                    }),
                ) if requested_precision >= file_precision && file_scale == requested_scale => true,
                // Uuid will be store as Fixed(16) in parquet file, so the read back type will be Fixed(16).
                (Some(PrimitiveType::Fixed(16)), Some(PrimitiveType::Uuid)) => true,
                _ => false,
            }
        }

        if field_ids.is_empty() {
            return Ok(ProjectionMask::all());
        }

        if use_fallback {
            // Position-based projection necessary because file lacks embedded field IDs
            Self::get_arrow_projection_mask_fallback(field_ids, parquet_schema)
        } else {
            // Parquet's columnar format requires leaf-level (not top-level struct/list/map) projection
            let mut leaf_field_ids = vec![];
            for field_id in field_ids {
                // The row-lineage ids are not in the table schema but can be in the file. They
                // are scalars, so the leaf id is the id.
                let field = iceberg_schema_of_task
                    .field_by_id(*field_id)
                    .cloned()
                    .or_else(|| get_metadata_field(*field_id).ok().cloned());
                if let Some(field) = field {
                    Self::include_leaf_field_id(&field, &mut leaf_field_ids);
                }
            }

            Self::get_arrow_projection_mask_with_field_ids(
                &leaf_field_ids,
                iceberg_schema_of_task,
                parquet_schema,
                arrow_schema,
                type_promotion_is_valid,
            )
        }
    }

    /// Standard projection using embedded field IDs from Parquet metadata.
    /// For iceberg-java compatibility with ParquetSchemaUtil.pruneColumns().
    fn get_arrow_projection_mask_with_field_ids(
        leaf_field_ids: &[i32],
        iceberg_schema_of_task: &Schema,
        parquet_schema: &SchemaDescriptor,
        arrow_schema: &ArrowSchemaRef,
        type_promotion_is_valid: fn(Option<&PrimitiveType>, Option<&PrimitiveType>) -> bool,
    ) -> Result<ProjectionMask> {
        let mut column_map = HashMap::new();
        let fields = arrow_schema.fields();

        let mut projected_fields: HashMap<FieldRef, i32> = HashMap::new();
        let projected_arrow_schema = ArrowSchema::new_with_metadata(
            fields.filter_leaves(|_, f| {
                f.metadata()
                    .get(PARQUET_FIELD_ID_META_KEY)
                    .and_then(|field_id| i32::from_str(field_id).ok())
                    .is_some_and(|field_id| {
                        projected_fields.insert((*f).clone(), field_id);
                        leaf_field_ids.contains(&field_id)
                    })
            }),
            arrow_schema.metadata().clone(),
        );
        let iceberg_schema = arrow_schema_to_schema(&projected_arrow_schema)?;

        fields.filter_leaves(|idx, field| {
            let Some(field_id) = projected_fields.get(field).cloned() else {
                return false;
            };

            // The row-lineage columns are not in the table schema but can be in the file, so take
            // their type from the reserved-column registry.
            let iceberg_field = iceberg_schema_of_task
                .field_by_id(field_id)
                .cloned()
                .or_else(|| get_metadata_field(field_id).ok().cloned());
            let parquet_iceberg_field = iceberg_schema.field_by_id(field_id);

            let (Some(iceberg_field), Some(parquet_iceberg_field)) =
                (iceberg_field, parquet_iceberg_field)
            else {
                return false;
            };

            if !type_promotion_is_valid(
                parquet_iceberg_field.field_type.as_primitive_type(),
                iceberg_field.field_type.as_primitive_type(),
            ) {
                return false;
            }

            column_map.insert(field_id, idx);
            true
        });

        // Schema evolution: New columns may not exist in old Parquet files.
        // We only project existing columns; RecordBatchTransformer adds default/NULL values.
        let mut indices = vec![];
        for field_id in leaf_field_ids {
            if let Some(col_idx) = column_map.get(field_id) {
                indices.push(*col_idx);
            }
        }

        if indices.is_empty() {
            // Edge case: All requested columns are new (don't exist in file).
            // Project all columns so RecordBatchTransformer has a batch to transform.
            Ok(ProjectionMask::all())
        } else {
            Ok(ProjectionMask::leaves(parquet_schema, indices))
        }
    }

    /// Fallback projection for Parquet files without field IDs.
    /// Uses position-based matching: field ID N → column position N-1.
    /// Projects entire top-level columns (including nested content) for iceberg-java compatibility.
    fn get_arrow_projection_mask_fallback(
        field_ids: &[i32],
        parquet_schema: &SchemaDescriptor,
    ) -> Result<ProjectionMask> {
        let parquet_root_fields = parquet_schema.root_schema().get_fields();
        let mut root_indices = vec![];

        for field_id in field_ids.iter() {
            let parquet_pos = (*field_id - 1) as usize;

            if parquet_pos < parquet_root_fields.len() {
                root_indices.push(parquet_pos);
            }
            // RecordBatchTransformer adds missing columns with NULL values
        }

        if root_indices.is_empty() {
            Ok(ProjectionMask::all())
        } else {
            Ok(ProjectionMask::roots(parquet_schema, root_indices))
        }
    }

    fn get_row_filter(
        predicates: &BoundPredicate,
        parquet_schema: &SchemaDescriptor,
        iceberg_field_ids: &HashSet<i32>,
        field_id_map: &HashMap<i32, usize>,
    ) -> Result<RowFilter> {
        // If the field id is not found in Parquet schema, it will be ignored due to schema evolution.
        let mut column_indices = iceberg_field_ids
            .iter()
            .filter_map(|field_id| field_id_map.get(field_id).cloned())
            .collect::<Vec<_>>();
        column_indices.sort();

        let mut converter = PredicateConverter {
            parquet_schema,
            column_map: field_id_map,
            column_indices: &column_indices,
        };

        let projection_mask = ProjectionMask::leaves(parquet_schema, column_indices.clone());
        let predicate_func = visit(&mut converter, predicates)?;
        let arrow_predicate = ArrowPredicateFn::new(projection_mask, predicate_func);
        Ok(RowFilter::new(vec![Box::new(arrow_predicate)]))
    }

    fn get_selected_row_group_indices(
        predicate: &BoundPredicate,
        parquet_metadata: &Arc<ParquetMetaData>,
        field_id_map: &HashMap<i32, usize>,
        snapshot_schema: &Schema,
    ) -> Result<Vec<usize>> {
        let row_groups_metadata = parquet_metadata.row_groups();
        let mut results = Vec::with_capacity(row_groups_metadata.len());

        for (idx, row_group_metadata) in row_groups_metadata.iter().enumerate() {
            if RowGroupMetricsEvaluator::eval(
                predicate,
                row_group_metadata,
                field_id_map,
                snapshot_schema,
            )? {
                results.push(idx);
            }
        }

        Ok(results)
    }

    fn get_row_selection_for_filter_predicate(
        predicate: &BoundPredicate,
        parquet_metadata: &Arc<ParquetMetaData>,
        selected_row_groups: &Option<Vec<usize>>,
        field_id_map: &HashMap<i32, usize>,
        snapshot_schema: &Schema,
    ) -> Result<RowSelection> {
        let Some(column_index) = parquet_metadata.column_index() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Parquet file metadata does not contain a column index",
            ));
        };

        let Some(offset_index) = parquet_metadata.offset_index() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Parquet file metadata does not contain an offset index",
            ));
        };

        if let Some(selected_row_groups) = selected_row_groups
            && selected_row_groups.is_empty()
        {
            return Ok(RowSelection::from(Vec::new()));
        }

        let mut selected_row_groups_idx = 0;

        let page_index = column_index
            .iter()
            .enumerate()
            .zip(offset_index)
            .zip(parquet_metadata.row_groups());

        let mut results = Vec::new();
        for (((idx, column_index), offset_index), row_group_metadata) in page_index {
            if let Some(selected_row_groups) = selected_row_groups {
                if idx == selected_row_groups[selected_row_groups_idx] {
                    selected_row_groups_idx += 1;
                } else {
                    continue;
                }
            }

            let selections_for_page = PageIndexEvaluator::eval(
                predicate,
                column_index,
                offset_index,
                row_group_metadata,
                field_id_map,
                snapshot_schema,
            )?;

            results.push(selections_for_page);

            if let Some(selected_row_groups) = selected_row_groups
                && selected_row_groups_idx == selected_row_groups.len()
            {
                break;
            }
        }

        Ok(results.into_iter().flatten().collect::<Vec<_>>().into())
    }

    /// Java's `ParquetMetadataConverter.getOffset(ColumnChunk)`: the byte offset at which a column
    /// chunk's data begins, which for the first column of a row group is that row group's real
    /// start position in the file. The rule is `MIN(data_page_offset, dictionary_page_offset)`. The
    /// dictionary offset wins only when it is set AND strictly smaller.
    fn parquet_column_chunk_offset(column: &ColumnChunkMetaData) -> i64 {
        let data_page_offset = column.data_page_offset();
        match column.dictionary_page_offset() {
            Some(dictionary_page_offset) if data_page_offset > dictionary_page_offset => {
                dictionary_page_offset
            }
            _ => data_page_offset,
        }
    }

    /// Filters row groups by byte range to support Iceberg's file splitting.
    ///
    /// A row group is kept when its MIDPOINT falls in the half-open window
    /// `[start, start + length)`. This is parquet-mr's `filterFileMetaDataByMidpoint` rule. Every
    /// row group belongs to exactly one window, so a tiling of the file reads every row once. An
    /// overlap rule hands a straddling row group to BOTH adjacent tasks and duplicates rows.
    ///
    /// | Detail | Rule |
    /// |---|---|
    /// | Midpoint | `start + compressed_size / 2`, truncating division, not the endpoint average |
    /// | Window | inclusive low, exclusive high, so a boundary midpoint belongs to the higher split |
    /// | Row-group start | read from the footer, never modelled as `4 + Σ compressed_size`, which drifts on padding or inline bloom filters |
    ///
    /// # Errors
    ///
    /// Three fail-closed divergences from Java, all [`ErrorKind::DataInvalid`]. A negative offset
    /// or size, where Java computes a negative midpoint and silently drops the row group. A row
    /// group with no column chunks, where Java throws `IndexOutOfBoundsException`. A size sum that
    /// overflows, where `RowGroupMetaData::compressed_size()` panics or wraps.
    ///
    /// # Notes
    ///
    /// Midpoint selection means a window that misses a row group's midpoint reads none of its
    /// rows. Callers must TILE `[0, file_size)`. Java behaves the same way.
    fn filter_row_groups_by_byte_range(
        parquet_metadata: &Arc<ParquetMetaData>,
        start: u64,
        length: u64,
    ) -> Result<Vec<usize>> {
        let row_groups = parquet_metadata.row_groups();
        let mut selected = Vec::new();
        // `start + length` can overflow `u64` on a hostile/corrupt split descriptor.
        let end = start.checked_add(length).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Row-group byte range start + length overflows u64",
            )
        })?;

        for (idx, row_group) in row_groups.iter().enumerate() {
            // Java throws IndexOutOfBounds on an empty row group. Fail with a typed error.
            let first_column = row_group.columns().first().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Row group {idx} has no column chunks; its byte offset cannot be determined"
                    ),
                )
            })?;
            let row_group_start = u64::try_from(Self::parquet_column_chunk_offset(first_column))
                .map_err(|_| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Row group {idx} has a negative byte offset"),
                    )
                })?;

            // Java's else-branch sum, because parquet-rs does not decode the thrift
            // `RowGroup.total_compressed_size`. Sum with `checked_add`: parquet-rs range-validates
            // nothing, so `RowGroupMetaData::compressed_size()` panics or wraps on a corrupt
            // footer, before any guard below runs.
            let mut row_group_size_i64: i64 = 0;
            for column in row_group.columns() {
                row_group_size_i64 = row_group_size_i64
                    .checked_add(column.compressed_size())
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Row group {idx} compressed size overflows i64"),
                        )
                    })?;
            }
            // A corrupt negative size must not wrap when converted.
            let row_group_size = u64::try_from(row_group_size_i64).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Row-group compressed size is negative",
                )
            })?;

            let midpoint = row_group_start
                .checked_add(row_group_size / 2)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Row group {idx} midpoint overflows u64"),
                    )
                })?;

            if midpoint >= start && midpoint < end {
                selected.push(idx);
            }
        }

        Ok(selected)
    }
}

/// Build the map of parquet field id to Parquet column index in the schema.
/// Returns None if the Parquet file doesn't have field IDs embedded (e.g., migrated tables).
fn build_field_id_map(parquet_schema: &SchemaDescriptor) -> Result<Option<HashMap<i32, usize>>> {
    let mut column_map = HashMap::new();

    for (idx, field) in parquet_schema.columns().iter().enumerate() {
        let field_type = field.self_type();
        match field_type {
            ParquetType::PrimitiveType { basic_info, .. } => {
                if !basic_info.has_id() {
                    return Ok(None);
                }
                column_map.insert(basic_info.id(), idx);
            }
            ParquetType::GroupType { .. } => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Leaf column in schema should be primitive type but got {field_type:?}"
                    ),
                ));
            }
        };
    }

    Ok(Some(column_map))
}

/// Build a fallback field ID map for Parquet files without embedded field IDs.
///
/// Returns the number of primitive (leaf) columns in a Parquet type, recursing into groups.
fn leaf_count(ty: &parquet::schema::types::Type) -> usize {
    if ty.is_primitive() {
        1
    } else {
        ty.get_fields().iter().map(|f| leaf_count(f)).sum()
    }
}

/// Maps fallback field ids to leaf column indices, for primitive top-level fields only. Java
/// `ParquetSchemaUtil.addFallbackIds()`. # Notes Use top-level field positions, not leaf positions,
/// to match `add_fallback_field_ids_to_arrow_schema`.
fn build_fallback_field_id_map(parquet_schema: &SchemaDescriptor) -> HashMap<i32, usize> {
    let mut column_map = HashMap::new();
    let mut leaf_idx = 0;

    for (top_pos, field) in parquet_schema.root_schema().get_fields().iter().enumerate() {
        let field_id = (top_pos + 1) as i32;
        if field.is_primitive() {
            column_map.insert(field_id, leaf_idx);
        }
        leaf_idx += leaf_count(field);
    }

    column_map
}

/// Assigns Iceberg field ids by column name, so a migrated file projects correctly. Java
/// `ParquetSchemaUtil.applyNameMapping()`, on an Arrow schema instead of a Parquet `MessageType`. #
/// Arguments * `arrow_schema` - the Arrow schema read from the file, without field ids *
/// `name_mapping` - the table's `schema.name-mapping.default` # Returns The Arrow schema with field
/// ids assigned.
fn apply_name_mapping_to_arrow_schema(
    arrow_schema: ArrowSchemaRef,
    name_mapping: &NameMapping,
) -> Result<Arc<ArrowSchema>> {
    debug_assert!(
        arrow_schema
            .fields()
            .iter()
            .next()
            .is_none_or(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none()),
        "Schema already has field IDs - name mapping should not be applied"
    );

    use arrow_schema::Field;

    let fields_with_mapped_ids: Vec<_> = arrow_schema
        .fields()
        .iter()
        .map(|field| {
            // Java `ApplyNameMapping` calls `nameMapping.find(currentPath())`.
            //
            // A field absent from the mapping keeps no id, as in Java, and projection then
            // filters it out.
            let mapped_field_opt = name_mapping
                .fields()
                .iter()
                .find(|f| f.names().contains(&field.name().to_string()));

            let mut metadata = field.metadata().clone();

            if let Some(mapped_field) = mapped_field_opt
                && let Some(field_id) = mapped_field.field_id()
            {
                // The mapping names the field, so assign its id.
                metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string());
            }
            // No id, so projection filters the field out.

            Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                .with_metadata(metadata)
        })
        .collect();

    Ok(Arc::new(ArrowSchema::new_with_metadata(
        fields_with_mapped_ids,
        arrow_schema.metadata().clone(),
    )))
}

/// Adds position-based fallback field ids to an Arrow schema, so a migrated file projects.
///
/// # Notes
///
/// Ids are 1-indexed, to match Java `ParquetSchemaUtil.addFallbackIds()`. Only top-level fields
/// get one, because nested projection uses leaf column indices.
fn add_fallback_field_ids_to_arrow_schema(arrow_schema: &ArrowSchemaRef) -> Arc<ArrowSchema> {
    debug_assert!(
        arrow_schema
            .fields()
            .iter()
            .next()
            .is_none_or(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none()),
        "Schema already has field IDs"
    );

    use arrow_schema::Field;

    let fields_with_fallback_ids: Vec<_> = arrow_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(pos, field)| {
            let mut metadata = field.metadata().clone();
            let field_id = (pos + 1) as i32; // 1-indexed for Java compatibility
            metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string());

            Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                .with_metadata(metadata)
        })
        .collect();

    Arc::new(ArrowSchema::new_with_metadata(
        fields_with_fallback_ids,
        arrow_schema.metadata().clone(),
    ))
}

/// A visitor to collect field ids from bound predicates.
struct CollectFieldIdVisitor {
    field_ids: HashSet<i32>,
}

impl CollectFieldIdVisitor {
    fn field_ids(self) -> HashSet<i32> {
        self.field_ids
    }
}

impl BoundPredicateVisitor for CollectFieldIdVisitor {
    type T = ();

    fn always_true(&mut self) -> Result<()> {
        Ok(())
    }

    fn always_false(&mut self) -> Result<()> {
        Ok(())
    }

    fn and(&mut self, _lhs: (), _rhs: ()) -> Result<()> {
        Ok(())
    }

    fn or(&mut self, _lhs: (), _rhs: ()) -> Result<()> {
        Ok(())
    }

    fn not(&mut self, _inner: ()) -> Result<()> {
        Ok(())
    }

    fn is_null(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_null(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn is_nan(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_nan(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }
}

/// A visitor to convert Iceberg bound predicates to Arrow predicates.
struct PredicateConverter<'a> {
    /// The Parquet schema descriptor.
    pub parquet_schema: &'a SchemaDescriptor,
    /// The map between field id and leaf column index in Parquet schema.
    pub column_map: &'a HashMap<i32, usize>,
    /// The required column indices in Parquet schema for the predicates.
    pub column_indices: &'a Vec<usize>,
}

impl PredicateConverter<'_> {
    /// When visiting a bound reference, we return index of the leaf column in the
    /// required column indices which is used to project the column in the record batch.
    /// Return None if the field id is not found in the column map, which is possible
    /// due to schema evolution.
    fn bound_reference(&mut self, reference: &BoundReference) -> Result<Option<usize>> {
        if let Some(column_idx) = self.column_map.get(&reference.field().id) {
            if self.parquet_schema.get_column_root(*column_idx).is_group() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Leaf column `{}` in predicates isn't a root column in Parquet schema.",
                        reference.field().name
                    ),
                ));
            }

            let index = self
                .column_indices
                .iter()
                .position(|&idx| idx == *column_idx)
                .ok_or(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                "Leaf column `{}` in predicates cannot be found in the required column indices.",
                reference.field().name
            ),
                ))?;

            Ok(Some(index))
        } else {
            Ok(None)
        }
    }

    /// Build an Arrow predicate that always returns true.
    fn build_always_true(&self) -> Result<Box<PredicateResult>> {
        Ok(Box::new(|batch| {
            Ok(BooleanArray::from(vec![true; batch.num_rows()]))
        }))
    }

    /// Build an Arrow predicate that always returns false.
    fn build_always_false(&self) -> Result<Box<PredicateResult>> {
        Ok(Box::new(|batch| {
            Ok(BooleanArray::from(vec![false; batch.num_rows()]))
        }))
    }
}

/// Coerces a three-valued keep-mask to two values, turning NULL into `false`, as the Parquet
/// `RowFilter` does. [`evaluate_predicate_to_mask`] already returns a two-valued mask, so this is
/// defense in depth against a future three-valued-logic leak.
fn coerce_nulls_to_false(mask: &BooleanArray) -> BooleanArray {
    if mask.null_count() == 0 {
        return mask.clone();
    }
    BooleanArray::from_iter((0..mask.len()).map(|i| Some(mask.is_valid(i) && mask.value(i))))
}

/// `true` when every key field id of `sets` is projected. The keyset path needs it to resolve
/// keys on a transformed batch. Otherwise the reader uses the eq-delete RowFilter path.
pub(crate) fn eq_delete_key_fields_projected(
    sets: &[EqDeleteKeySet],
    projected_non_metadata_field_ids: &[i32],
) -> bool {
    if sets.is_empty() {
        return false;
    }
    // Every set shares one key schema, so the first set's key ids stand for the task.
    let projected: HashSet<i32> = projected_non_metadata_field_ids.iter().copied().collect();
    sets[0]
        .key_field_ids()
        .iter()
        .all(|id| projected.contains(id))
}

/// Gets the leaf column from the record batch for the required column index. Only
/// supports top-level columns for now.
fn project_column(
    batch: &RecordBatch,
    column_idx: usize,
) -> std::result::Result<ArrayRef, ArrowError> {
    let column = batch.column(column_idx);

    match column.data_type() {
        DataType::Struct(_) => Err(ArrowError::SchemaError(
            "Does not support struct column yet.".to_string(),
        )),
        _ => Ok(column.clone()),
    }
}

type PredicateResult =
    dyn FnMut(RecordBatch) -> std::result::Result<BooleanArray, ArrowError> + Send + 'static;

impl BoundPredicateVisitor for PredicateConverter<'_> {
    type T = Box<PredicateResult>;

    fn always_true(&mut self) -> Result<Box<PredicateResult>> {
        self.build_always_true()
    }

    fn always_false(&mut self) -> Result<Box<PredicateResult>> {
        self.build_always_false()
    }

    fn and(
        &mut self,
        mut lhs: Box<PredicateResult>,
        mut rhs: Box<PredicateResult>,
    ) -> Result<Box<PredicateResult>> {
        Ok(Box::new(move |batch| {
            let left = lhs(batch.clone())?;
            let right = rhs(batch)?;
            and_kleene(&left, &right)
        }))
    }

    fn or(
        &mut self,
        mut lhs: Box<PredicateResult>,
        mut rhs: Box<PredicateResult>,
    ) -> Result<Box<PredicateResult>> {
        Ok(Box::new(move |batch| {
            let left = lhs(batch.clone())?;
            let right = rhs(batch)?;
            or_kleene(&left, &right)
        }))
    }

    fn not(&mut self, mut inner: Box<PredicateResult>) -> Result<Box<PredicateResult>> {
        Ok(Box::new(move |batch| {
            let pred_ret = inner(batch)?;
            not(&pred_ret)
        }))
    }

    fn is_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                is_null(&column)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn not_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                is_not_null(&column)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn is_nan(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                // Java `NaNUtil.isNaN`. A NULL cell gives false, and the mask stays two-valued.
                Ok(is_nan_row_mask(&column))
            }))
        } else {
            // A missing column, treating it as null: Java `NaNUtil.isNaN(null)` == false.
            self.build_always_false()
        }
    }

    fn not_nan(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                // A NULL cell is not NaN, so the row stays. Java `EvalVisitor.notNaN`.
                Ok(not_nan_row_mask(&column))
            }))
        } else {
            // A missing column, treating it as null: `!NaNUtil.isNaN(null)` == true.
            self.build_always_true()
        }
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                // A NULL cell keeps the row. Java's nulls-first comparator gives
                // `compare(null, lit) == -1`, and `EvalVisitor.lt` tests `< 0`. A
                // three-valued-logic NULL slot would make the `RowFilter` drop the row.
                Ok(null_filled(lt(&left, literal.as_ref())?, true))
            }))
        } else {
            // A missing column is a NULL column ⇒ TRUE (nulls-first: null < lit).
            self.build_always_true()
        }
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                // A NULL cell keeps the row. Java `ltEq` tests `<= 0` over -1.
                Ok(null_filled(lt_eq(&left, literal.as_ref())?, true))
            }))
        } else {
            // A missing column is a NULL column ⇒ TRUE (nulls-first: null <= lit).
            self.build_always_true()
        }
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                // A NULL cell drops the row. Java `gt` tests `> 0`. Stating it keeps `not`,
                // `and`, and `or` composition plain boolean.
                Ok(null_filled(gt(&left, literal.as_ref())?, false))
            }))
        } else {
            // A missing column is a NULL column ⇒ FALSE (nulls-first: null > lit is false).
            self.build_always_false()
        }
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                // A NULL cell drops the row. Java `gtEq` tests `>= 0`.
                Ok(null_filled(gt_eq(&left, literal.as_ref())?, false))
            }))
        } else {
            // A missing column is a NULL column ⇒ FALSE (nulls-first: null >= lit is false).
            self.build_always_false()
        }
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                // A NULL cell drops the row. Java `eq` tests `== 0` over -1.
                Ok(null_filled(eq(&left, literal.as_ref())?, false))
            }))
        } else {
            // A missing column is a NULL column ⇒ FALSE (null == lit is false under
            // nulls-first).
            self.build_always_false()
        }
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                // A NULL cell keeps the row. Java `notEq` is `!eq`. The kernel's three-valued
                // NULL made the `RowFilter` drop every NULL cell under `!=`.
                Ok(null_filled(neq(&left, literal.as_ref())?, true))
            }))
        } else {
            // A missing column is NULL, and Java `notEq(null, lit)` is true. An always-false
            // build made a schema-evolved file return zero rows under `!=`.
            self.build_always_true()
        }
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                // A NULL cell drops the row. Java `startsWith` null-guards to false.
                Ok(null_filled(starts_with(&left, literal.as_ref())?, false))
            }))
        } else {
            // A missing column is a NULL column ⇒ FALSE (Java's explicit null guard).
            self.build_always_false()
        }
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                // Update this if arrow adds a native not_starts_with.
                // A NULL cell keeps the row. Java `notStartsWith` negates the null guard.
                Ok(null_filled(
                    not(&starts_with(&left, literal.as_ref())?)?,
                    true,
                ))
            }))
        } else {
            // A missing column is a NULL column ⇒ TRUE (`notStartsWith` negates the null
            // guard's false).
            self.build_always_true()
        }
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            // `get_arrow_datum` fails on a decimal literal past Arrow's Decimal128 precision, and
            // on an unsupported type. Propagate a typed error, never panic the predicate build.
            let literals = literals
                .iter()
                .map(get_arrow_datum)
                .collect::<Result<Vec<_>>>()?;

            Ok(Box::new(move |batch| {
                // update this if arrow ever adds a native is_in kernel
                let left = project_column(&batch, idx)?;

                let mut acc = BooleanArray::from(vec![false; batch.num_rows()]);
                for literal in &literals {
                    let literal = try_cast_literal(literal, left.data_type())?;
                    acc = or(&acc, &eq(&left, literal.as_ref())?)?
                }

                // A NULL cell drops the row. Java `in` is `literalSet.contains(null)`, false in
                // both set implementations.
                Ok(null_filled(acc, false))
            }))
        } else {
            // A missing column is a NULL column ⇒ FALSE (`contains(null)` is false).
            self.build_always_false()
        }
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            // Fallible like `r#in` above, so propagate a typed error.
            let literals = literals
                .iter()
                .map(get_arrow_datum)
                .collect::<Result<Vec<_>>>()?;

            Ok(Box::new(move |batch| {
                // update this if arrow ever adds a native not_in kernel
                let left = project_column(&batch, idx)?;
                let mut acc = BooleanArray::from(vec![true; batch.num_rows()]);
                for literal in &literals {
                    let literal = try_cast_literal(literal, left.data_type())?;
                    acc = and(&acc, &neq(&left, literal.as_ref())?)?
                }

                // A NULL cell keeps the row. Java `notIn` negates `contains(null)`. An
                // accumulated three-valued NULL made the `RowFilter` drop NULL cells.
                Ok(null_filled(acc, true))
            }))
        } else {
            // A missing column is a NULL column ⇒ TRUE (`notIn` negates contains(null)).
            self.build_always_true()
        }
    }
}

/// ArrowFileReader is a wrapper around a FileRead that impls parquets AsyncFileReader.
pub struct ArrowFileReader {
    meta: FileMetadata,
    parquet_read_options: ParquetReadOptions,
    r: Box<dyn FileRead>,
}

impl ArrowFileReader {
    /// Create a new ArrowFileReader
    pub fn new(meta: FileMetadata, r: Box<dyn FileRead>) -> Self {
        Self {
            meta,
            parquet_read_options: ParquetReadOptions::builder().build(),
            r,
        }
    }

    /// Configure all Parquet read options.
    pub(crate) fn with_parquet_read_options(mut self, options: ParquetReadOptions) -> Self {
        self.parquet_read_options = options;
        self
    }
}

impl AsyncFileReader for ArrowFileReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        Box::pin(
            self.r
                .read(range.start..range.end)
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err))),
        )
    }

    /// Overrides the default `get_byte_ranges`, which calls `get_bytes` in series. Without this,
    /// every column chunk of a row group is a serial round trip to object storage. Adapted from
    /// object_store's `coalesce_ranges`.
    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        let coalesce_bytes = self.parquet_read_options.range_coalesce_bytes();
        let concurrency = self.parquet_read_options.range_fetch_concurrency().max(1);

        async move {
            // Merge nearby ranges to reduce the number of object store requests.
            let fetch_ranges = merge_ranges(&ranges, coalesce_bytes);
            let r = &self.r;

            // Fetch merged ranges concurrently.
            let fetched: Vec<Bytes> = futures::stream::iter(fetch_ranges.iter().cloned())
                .map(|range| async move {
                    r.read(range)
                        .await
                        .map_err(|e| parquet::errors::ParquetError::External(Box::new(e)))
                })
                .buffered(concurrency)
                .try_collect()
                .await?;

            // Slice the fetched data back into the originally requested ranges.
            Ok(ranges
                .iter()
                .map(|range| {
                    let idx = fetch_ranges.partition_point(|v| v.start <= range.start) - 1;
                    let fetch_range = &fetch_ranges[idx];
                    let fetch_bytes = &fetched[idx];
                    let start = (range.start - fetch_range.start) as usize;
                    let end = (range.end - fetch_range.start) as usize;
                    fetch_bytes.slice(start..end.min(fetch_bytes.len()))
                })
                .collect())
        }
        .boxed()
    }

    // TODO: currently we don't respect `ArrowReaderOptions` cause it don't expose any method to access the option field
    // we will fix it after `v55.1.0` is released in https://github.com/apache/arrow-rs/issues/7393
    fn get_metadata(
        &mut self,
        _options: Option<&'_ ArrowReaderOptions>,
    ) -> BoxFuture<'_, parquet::errors::Result<Arc<ParquetMetaData>>> {
        async move {
            let reader = ParquetMetaDataReader::new()
                .with_prefetch_hint(self.parquet_read_options.metadata_size_hint())
                // Set the page policy first because it updates both column and offset policies.
                .with_page_index_policy(PageIndexPolicy::from(
                    self.parquet_read_options.preload_page_index(),
                ))
                .with_column_index_policy(PageIndexPolicy::from(
                    self.parquet_read_options.preload_column_index(),
                ))
                .with_offset_index_policy(PageIndexPolicy::from(
                    self.parquet_read_options.preload_offset_index(),
                ));
            let size = self.meta.size;
            let meta = reader.load_and_finish(self, size).await?;

            Ok(Arc::new(meta))
        }
        .boxed()
    }
}

/// Merge overlapping or nearby byte ranges, combining ranges with gaps <= `coalesce` bytes.
/// Adapted from object_store's `merge_ranges` in `util.rs`.
fn merge_ranges(ranges: &[Range<u64>], coalesce: u64) -> Vec<Range<u64>> {
    if ranges.is_empty() {
        return vec![];
    }

    let mut ranges = ranges.to_vec();
    ranges.sort_unstable_by_key(|r| r.start);

    let mut merged = Vec::with_capacity(ranges.len());
    let mut start_idx = 0;
    let mut end_idx = 1;

    while start_idx != ranges.len() {
        let mut range_end = ranges[start_idx].end;

        while end_idx != ranges.len()
            && ranges[end_idx]
                .start
                .checked_sub(range_end)
                .map(|delta| delta <= coalesce)
                .unwrap_or(true)
        {
            range_end = range_end.max(ranges[end_idx].end);
            end_idx += 1;
        }

        merged.push(ranges[start_idx].start..range_end);
        start_idx = end_idx;
        end_idx += 1;
    }

    merged
}

/// Casts a literal to the column's Arrow type. The reader may return `LargeUtf8` or `Utf8View`
/// where Iceberg's literal is `Utf8`, and the compute kernels need an exact type match.
///
/// It is `pub(crate)` so the `ConvertEqualityDeleteFiles` action aligns literals with the same
/// logic the read side uses.
pub(crate) fn try_cast_literal(
    literal: &Arc<dyn ArrowDatum + Send + Sync>,
    column_type: &DataType,
) -> std::result::Result<Arc<dyn ArrowDatum + Send + Sync>, ArrowError> {
    let literal_array = literal.get().0;

    // No cast required
    if literal_array.data_type() == column_type {
        return Ok(Arc::clone(literal));
    }

    let literal_array = cast(literal_array, column_type)?;
    Ok(Arc::new(Scalar::new(literal_array)))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::ops::Range;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, LargeStringArray, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
    use fnv::FnvHashSet;
    use futures::TryStreamExt;
    use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
    use parquet::arrow::{ArrowWriter, ProjectionMask};
    use parquet::basic::Compression;
    use parquet::file::metadata::{
        ColumnChunkMetaData, FileMetaData, ParquetMetaData, RowGroupMetaData,
    };
    use parquet::file::properties::WriterProperties;
    use parquet::schema::parser::parse_message_type;
    use parquet::schema::types::{SchemaDescPtr, SchemaDescriptor};
    use roaring::RoaringTreemap;
    use tempfile::TempDir;

    use crate::ErrorKind;
    use crate::arrow::reader::{
        CollectFieldIdVisitor, PARQUET_FIELD_ID_META_KEY, PredicateConverter,
    };
    use crate::arrow::{ArrowReader, ArrowReaderBuilder};
    use crate::delete_vector::DeleteVector;
    use crate::expr::accessor::StructAccessor;
    use crate::expr::visitors::bound_predicate_visitor::visit;
    use crate::expr::{
        Bind, BoundPredicate, BoundReference, Predicate, PredicateOperator, Reference,
        SetExpression,
    };
    use crate::io::FileIO;
    use crate::metadata_columns::{RESERVED_FIELD_ID_POS, RESERVED_FIELD_ID_ROW_ID};
    use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
    use crate::spec::{
        DataContentType, DataFileFormat, Datum, NestedField, PrimitiveLiteral, PrimitiveType,
        Schema, SchemaRef, Type,
    };

    fn table_schema_simple() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_identifier_field_ids(vec![2])
                .with_fields(vec![
                    NestedField::optional(1, "foo", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean)).into(),
                    NestedField::optional(4, "qux", Type::Primitive(PrimitiveType::Float)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn test_collect_field_id() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux").is_null();
        let bound_expr = expr.bind(schema, true).unwrap();

        let mut visitor = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut visitor, &bound_expr).unwrap();

        let mut expected = HashSet::default();
        expected.insert(4_i32);

        assert_eq!(visitor.field_ids, expected);
    }

    /// Drives the `IN` and `NOT IN` arms with a decimal literal past Arrow's precision limit of
    /// 38. The visitor must propagate the typed error, never panic while building the row filter.
    /// Precision above 38 is reachable: neither the `decimal(P,S)` deserializer nor
    /// `Datum::try_from_bytes` bound-checks it, so corrupt catalog metadata can supply one.
    fn assert_set_predicate_over_max_decimal_is_typed_error(op: PredicateOperator) {
        // One leaf decimal column carrying field id 1.
        let message_type = "
message schema {
  optional fixed_len_byte_array(16) d (DECIMAL(38,0)) = 1;
}
        ";
        let parquet_type = parse_message_type(message_type).expect("should parse schema");
        let parquet_schema = SchemaDescriptor::new(Arc::new(parquet_type));
        let column_map = HashMap::from([(1_i32, 0_usize)]);
        let column_indices = vec![0_usize];

        let field = NestedField::optional(
            1,
            "d",
            Type::Primitive(PrimitiveType::Decimal {
                precision: 38,
                scale: 0,
            }),
        )
        .into();
        let accessor = Arc::new(StructAccessor::new(0, PrimitiveType::Decimal {
            precision: 38,
            scale: 0,
        }));
        let bound_ref = BoundReference::new("d", field, accessor);

        // precision 50 > 38: get_arrow_datum returns Err for this literal.
        let bad = Datum::new(
            PrimitiveType::Decimal {
                precision: 50,
                scale: 0,
            },
            PrimitiveLiteral::Int128(1234),
        );
        let mut literals = FnvHashSet::default();
        literals.insert(bad);

        let predicate = BoundPredicate::Set(SetExpression::new(op, bound_ref, literals));

        let mut converter = PredicateConverter {
            parquet_schema: &parquet_schema,
            column_map: &column_map,
            column_indices: &column_indices,
        };

        match visit(&mut converter, &predicate) {
            Ok(_) => panic!(
                "{op:?} with a decimal literal of precision 50 must return a typed error, not panic"
            ),
            Err(err) => assert_eq!(err.kind(), ErrorKind::DataInvalid),
        }
    }

    #[test]
    fn in_predicate_over_max_decimal_precision_is_typed_error_not_panic() {
        assert_set_predicate_over_max_decimal_is_typed_error(PredicateOperator::In);
    }

    #[test]
    fn not_in_predicate_over_max_decimal_precision_is_typed_error_not_panic() {
        assert_set_predicate_over_max_decimal_is_typed_error(PredicateOperator::NotIn);
    }

    #[test]
    fn test_collect_field_id_with_and() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux")
            .is_null()
            .and(Reference::new("baz").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();

        let mut visitor = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut visitor, &bound_expr).unwrap();

        let mut expected = HashSet::default();
        expected.insert(4_i32);
        expected.insert(3);

        assert_eq!(visitor.field_ids, expected);
    }

    #[test]
    fn test_collect_field_id_with_or() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux")
            .is_null()
            .or(Reference::new("baz").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();

        let mut visitor = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut visitor, &bound_expr).unwrap();

        let mut expected = HashSet::default();
        expected.insert(4_i32);
        expected.insert(3);

        assert_eq!(visitor.field_ids, expected);
    }

    #[test]
    fn test_arrow_projection_mask() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_identifier_field_ids(vec![1])
                .with_fields(vec![
                    NestedField::required(1, "c1", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(2, "c2", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        3,
                        "c3",
                        Type::Primitive(PrimitiveType::Decimal {
                            precision: 38,
                            scale: 3,
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("c1", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("c2", DataType::Duration(TimeUnit::Microsecond), true).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]),
            ),
            Field::new("c3", DataType::Decimal128(39, 3), true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "3".to_string(),
            )])),
        ]));

        let message_type = "
message schema {
  required binary c1 (STRING) = 1;
  optional int32 c2 (INTEGER(8,true)) = 2;
  optional fixed_len_byte_array(17) c3 (DECIMAL(39,3)) = 3;
}
    ";
        let parquet_type = parse_message_type(message_type).expect("should parse schema");
        let parquet_schema = SchemaDescriptor::new(Arc::new(parquet_type));

        let err = ArrowReader::get_arrow_projection_mask(
            &[1, 2, 3],
            &schema,
            &parquet_schema,
            &arrow_schema,
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            err.to_string(),
            "DataInvalid => Unsupported Arrow data type: Duration(µs)".to_string()
        );

        let err = ArrowReader::get_arrow_projection_mask(
            &[1, 3],
            &schema,
            &parquet_schema,
            &arrow_schema,
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            err.to_string(),
            "DataInvalid => Failed to create decimal type, source: DataInvalid => Decimals with precision larger than 38 are not supported: 39".to_string()
        );

        let mask = ArrowReader::get_arrow_projection_mask(
            &[1],
            &schema,
            &parquet_schema,
            &arrow_schema,
            false,
        )
        .expect("Some ProjectionMask");
        assert_eq!(mask, ProjectionMask::leaves(&parquet_schema, vec![0]));
    }

    #[tokio::test]
    async fn test_kleene_logic_or_behaviour() {
        // a IS NULL OR a = 'foo'
        let predicate = Reference::new("a")
            .is_null()
            .or(Reference::new("a").equal_to(Datum::string("foo")));

        // Table data: [NULL, "foo", "bar"]
        let data_for_col_a = vec![None, Some("foo".to_string()), Some("bar".to_string())];

        // Expected: [NULL, "foo"].
        let expected = vec![None, Some("foo".to_string())];

        let (file_io, schema, table_location, _temp_dir) =
            setup_kleene_logic(data_for_col_a, DataType::Utf8);
        let reader = ArrowReaderBuilder::new(file_io).build();

        let result_data = test_perform_read(predicate, schema, table_location, reader).await;

        assert_eq!(result_data, expected);
    }

    #[tokio::test]
    async fn test_kleene_logic_and_behaviour() {
        // a IS NOT NULL AND a != 'foo'
        let predicate = Reference::new("a")
            .is_not_null()
            .and(Reference::new("a").not_equal_to(Datum::string("foo")));

        // Table data: [NULL, "foo", "bar"]
        let data_for_col_a = vec![None, Some("foo".to_string()), Some("bar".to_string())];

        // Expected: ["bar"].
        let expected = vec![Some("bar".to_string())];

        let (file_io, schema, table_location, _temp_dir) =
            setup_kleene_logic(data_for_col_a, DataType::Utf8);
        let reader = ArrowReaderBuilder::new(file_io).build();

        let result_data = test_perform_read(predicate, schema, table_location, reader).await;

        assert_eq!(result_data, expected);
    }

    #[tokio::test]
    async fn test_predicate_cast_literal() {
        let predicates = vec![
            (Reference::new("a").equal_to(Datum::string("foo")), vec![
                Some("foo".to_string()),
            ]),
            (
                Reference::new("a").not_equal_to(Datum::string("foo")),
                vec![Some("bar".to_string())],
            ),
            (Reference::new("a").starts_with(Datum::string("f")), vec![
                Some("foo".to_string()),
            ]),
            (
                Reference::new("a").not_starts_with(Datum::string("f")),
                vec![Some("bar".to_string())],
            ),
            (Reference::new("a").less_than(Datum::string("foo")), vec![
                Some("bar".to_string()),
            ]),
            (
                Reference::new("a").less_than_or_equal_to(Datum::string("foo")),
                vec![Some("foo".to_string()), Some("bar".to_string())],
            ),
            (
                Reference::new("a").greater_than(Datum::string("bar")),
                vec![Some("foo".to_string())],
            ),
            (
                Reference::new("a").greater_than_or_equal_to(Datum::string("foo")),
                vec![Some("foo".to_string())],
            ),
            (
                Reference::new("a").is_in([Datum::string("foo"), Datum::string("baz")]),
                vec![Some("foo".to_string())],
            ),
            (
                Reference::new("a").is_not_in([Datum::string("foo"), Datum::string("baz")]),
                vec![Some("bar".to_string())],
            ),
        ];

        // Table data: ["foo", "bar"]
        let data_for_col_a = vec![Some("foo".to_string()), Some("bar".to_string())];

        let (file_io, schema, table_location, _temp_dir) =
            setup_kleene_logic(data_for_col_a, DataType::LargeUtf8);
        let reader = ArrowReaderBuilder::new(file_io).build();

        for (predicate, expected) in predicates {
            println!("testing predicate {predicate}");
            let result_data = test_perform_read(
                predicate.clone(),
                schema.clone(),
                table_location.clone(),
                reader.clone(),
            )
            .await;

            assert_eq!(result_data, expected, "predicate={predicate}");
        }
    }

    async fn test_perform_read(
        predicate: Predicate,
        schema: SchemaRef,
        table_location: String,
        reader: ArrowReader,
    ) -> Vec<Option<String>> {
        let tasks = Box::pin(futures::stream::iter(vec![Ok(FileScanTask {
            file_size_in_bytes: std::fs::metadata(format!("{table_location}/1.parquet"))
                .unwrap()
                .len(),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(format!("{table_location}/1.parquet")),
            data_file_format: DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: Arc::from(vec![1]),
            predicate: Some(Arc::new(predicate.bind(schema, true).unwrap())),
            deletes: Arc::from(vec![]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        })])) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        result[0].columns()[0]
            .as_string_opt::<i32>()
            .unwrap()
            .iter()
            .map(|v| v.map(ToOwned::to_owned))
            .collect::<Vec<_>>()
    }

    fn setup_kleene_logic(
        data_for_col_a: Vec<Option<String>>,
        col_a_type: DataType,
    ) -> (FileIO, SchemaRef, String, TempDir) {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "a", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", col_a_type.clone(), true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();

        let file_io = FileIO::new_with_fs();

        let col = match col_a_type {
            DataType::Utf8 => Arc::new(StringArray::from(data_for_col_a)) as ArrayRef,
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(data_for_col_a)) as ArrayRef,
            _ => panic!("unexpected col_a_type"),
        };

        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![col]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer =
            ArrowWriter::try_new(file, to_write.schema(), Some(props.clone())).unwrap();

        writer.write(&to_write).expect("Writing batch");

        // writer must be closed to write footer
        writer.close().unwrap();

        (file_io, schema, table_location, tmp_dir)
    }

    #[test]
    fn test_build_deletes_row_selection() {
        let schema_descr = get_test_schema_descr();

        let mut columns = vec![];
        for ptr in schema_descr.columns() {
            let column = ColumnChunkMetaData::builder(ptr.clone()).build().unwrap();
            columns.push(column);
        }

        let row_groups_metadata = vec![
            build_test_row_group_meta(schema_descr.clone(), columns.clone(), 1000, 0),
            build_test_row_group_meta(schema_descr.clone(), columns.clone(), 500, 1),
            build_test_row_group_meta(schema_descr.clone(), columns.clone(), 500, 2),
            build_test_row_group_meta(schema_descr.clone(), columns.clone(), 1000, 3),
            build_test_row_group_meta(schema_descr.clone(), columns.clone(), 500, 4),
        ];

        let selected_row_groups = Some(vec![1, 3]);

        /* cases to cover:
           * {skip|select} {first|intermediate|last} {one row|multiple rows} in
             {first|intermediate|last} {skipped|selected} row group
           * row group selection disabled
        */

        let positional_deletes = RoaringTreemap::from_iter(&[
            1, // in skipped rg 0, should be ignored
            3, // run of three consecutive items in skipped rg0
            4, 5, 998, // two consecutive items at end of skipped rg0
            999, 1000, // solitary row at start of selected rg1 (1, 9)
            1010, // run of 3 rows in selected rg1
            1011, 1012, // (3, 485)
            1498, // run of two items at end of selected rg1
            1499, 1500, // run of two items at start of skipped rg2
            1501, 1600, // should ignore, in skipped rg2
            1999, // single row at end of skipped rg2
            2000, // run of two items at start of selected rg3
            2001, // (4, 98)
            2100, // single row in selected row group 3 (1, 99)
            2200, // run of 3 consecutive rows in selected row group 3
            2201, 2202, // (3, 796)
            2999, // single item at end of selected rg3 (1)
            3000, // single item at start of skipped rg4
        ]);

        let positional_deletes = DeleteVector::new(positional_deletes);

        let result = ArrowReader::build_deletes_row_selection(
            &row_groups_metadata,
            &selected_row_groups,
            &positional_deletes,
        )
        .unwrap();

        let expected = RowSelection::from(vec![
            RowSelector::skip(1),
            RowSelector::select(9),
            RowSelector::skip(3),
            RowSelector::select(485),
            RowSelector::skip(4),
            RowSelector::select(98),
            RowSelector::skip(1),
            RowSelector::select(99),
            RowSelector::skip(3),
            RowSelector::select(796),
            RowSelector::skip(1),
        ]);

        assert_eq!(result, expected);

        let result = ArrowReader::build_deletes_row_selection(
            &row_groups_metadata,
            &None,
            &positional_deletes,
        )
        .unwrap();

        let expected = RowSelection::from(vec![
            RowSelector::select(1),
            RowSelector::skip(1),
            RowSelector::select(1),
            RowSelector::skip(3),
            RowSelector::select(992),
            RowSelector::skip(3),
            RowSelector::select(9),
            RowSelector::skip(3),
            RowSelector::select(485),
            RowSelector::skip(4),
            RowSelector::select(98),
            RowSelector::skip(1),
            RowSelector::select(398),
            RowSelector::skip(3),
            RowSelector::select(98),
            RowSelector::skip(1),
            RowSelector::select(99),
            RowSelector::skip(3),
            RowSelector::select(796),
            RowSelector::skip(2),
            RowSelector::select(499),
        ]);

        assert_eq!(result, expected);
    }

    fn build_test_row_group_meta(
        schema_descr: SchemaDescPtr,
        columns: Vec<ColumnChunkMetaData>,
        num_rows: i64,
        ordinal: i16,
    ) -> RowGroupMetaData {
        RowGroupMetaData::builder(schema_descr.clone())
            .set_num_rows(num_rows)
            .set_total_byte_size(2000)
            .set_column_metadata(columns)
            .set_ordinal(ordinal)
            .build()
            .unwrap()
    }

    fn get_test_schema_descr() -> SchemaDescPtr {
        use parquet::schema::types::Type as SchemaType;

        let schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![
                Arc::new(
                    SchemaType::primitive_type_builder("a", parquet::basic::Type::INT32)
                        .build()
                        .unwrap(),
                ),
                Arc::new(
                    SchemaType::primitive_type_builder("b", parquet::basic::Type::INT32)
                        .build()
                        .unwrap(),
                ),
            ])
            .build()
            .unwrap();

        Arc::new(SchemaDescriptor::new(Arc::new(schema)))
    }

    /// `filter_row_groups_by_byte_range` guards `start + length` with `checked_add`. A descriptor
    /// of `start = u64::MAX, length = 1` must return a typed `DataInvalid` error. Cases `(h)` and
    /// `(i)` of `test_midpoint_selection_offset_and_boundary_semantics` pin the size branches.
    #[test]
    fn test_filter_row_groups_by_byte_range_start_plus_length_overflow() {
        use parquet::file::metadata::{FileMetaData, ParquetMetaData};

        let schema_descr = get_test_schema_descr();
        // Empty metadata is enough: the guard fires before any row group is examined.
        let file_metadata = FileMetaData::new(1, 0, None, None, schema_descr, None);
        let parquet_metadata = Arc::new(ParquetMetaData::new(file_metadata, Vec::new()));

        let err = ArrowReader::filter_row_groups_by_byte_range(&parquet_metadata, u64::MAX, 1)
            .expect_err("start + length overflowing u64 must error, not overflow");
        assert_eq!(
            err.kind(),
            ErrorKind::DataInvalid,
            "an overflowing byte range must be a typed DataInvalid error, got: {err}"
        );
    }

    // Midpoint row-group selection (parquet-mr `filterFileMetaDataByMidpoint`).
    //
    // Every helper here derives row-group positions from the REAL footer. A helper that uses the
    // `4 + Σ compressed_size` model cannot catch offset drift.

    /// Writes `num_row_groups` row groups of `rows_per_group` sequential `id` values (ids start at
    /// 0 and run across row-group boundaries). With `bloom_filters = true`, parquet-rs writes each
    /// bloom filter after its row group, so the row groups are not contiguous and their real
    /// offsets diverge from the naive model. [`DEFAULT_BLOOM_FILTER_POSITION`]:
    /// parquet::file::properties::DEFAULT_BLOOM_FILTER_POSITION.
    fn write_midpoint_fixture(path: &str, num_row_groups: usize, rows_per_group: i32, bloom: bool) {
        use arrow_array::Int32Array;

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(
                usize::try_from(rows_per_group).expect("positive rows_per_group"),
            ))
            .set_bloom_filter_enabled(bloom)
            .build();
        let file = File::create(path).expect("create midpoint fixture");
        let mut writer =
            ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).expect("arrow writer");
        for group in 0..num_row_groups {
            let base = i32::try_from(group).expect("group index fits i32") * rows_per_group;
            let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
                Int32Array::from_iter_values(base..base + rows_per_group),
            )])
            .expect("id batch");
            writer.write(&batch).expect("write row group");
        }
        writer.close().expect("close midpoint fixture");
    }

    fn footer_metadata(path: &str) -> Arc<ParquetMetaData> {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        let file = File::open(path).expect("open fixture");
        let reader = SerializedFileReader::new(file).expect("read footer");
        Arc::new(reader.metadata().clone())
    }

    /// Java `getOffset(rg.getColumns().get(0))` — `min(data_page_offset, dictionary_page_offset)`
    /// — read from the real footer, i.e. the true start position of each row group.
    fn footer_row_group_starts(metadata: &ParquetMetaData) -> Vec<u64> {
        metadata
            .row_groups()
            .iter()
            .map(|rg| {
                let col = rg.columns().first().expect("row group has a column chunk");
                let data = col.data_page_offset();
                let start = match col.dictionary_page_offset() {
                    Some(dict) if data > dict => dict,
                    _ => data,
                };
                u64::try_from(start).expect("non-negative offset")
            })
            .collect()
    }

    /// Java `startIndex + totalSize / 2` (truncating division on the SIZE).
    fn footer_row_group_midpoints(metadata: &ParquetMetaData) -> Vec<u64> {
        footer_row_group_starts(metadata)
            .into_iter()
            .zip(metadata.row_groups())
            .map(|(start, rg)| {
                start + u64::try_from(rg.compressed_size()).expect("non-negative size") / 2
            })
            .collect()
    }

    /// The DEFECTIVE model the production code used to synthesize (`4 + Σ compressed_size`). Only
    /// ever used here to *assert that it drifts* on a padded file — never to build an expectation.
    fn synthetic_row_group_starts(metadata: &ParquetMetaData) -> Vec<u64> {
        let mut offset = 4u64;
        metadata
            .row_groups()
            .iter()
            .map(|rg| {
                let start = offset;
                offset += u64::try_from(rg.compressed_size()).expect("non-negative size");
                start
            })
            .collect()
    }

    /// Tiles `[0, file_size)` into half-open windows of `stride` bytes; the last window is short.
    fn tile_windows(file_size: u64, stride: u64) -> Vec<(u64, u64)> {
        let mut windows = Vec::new();
        let mut start = 0u64;
        while start < file_size {
            let length = stride.min(file_size - start);
            windows.push((start, length));
            start += length;
        }
        windows
    }

    fn midpoint_scan_task(path: &str, schema: SchemaRef, start: u64, length: u64) -> FileScanTask {
        FileScanTask {
            file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
            start,
            length,
            record_count: None,
            data_file_path: Arc::from(path.to_string()),
            data_file_format: DataFileFormat::Parquet,
            schema,
            project_field_ids: Arc::from(vec![1]),
            predicate: None,
            deletes: Arc::from(vec![]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        }
    }

    async fn read_ids_for_window(
        path: &str,
        schema: SchemaRef,
        start: u64,
        length: u64,
    ) -> Vec<i32> {
        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
        let task = midpoint_scan_task(path, schema, start, length);
        let batches = reader
            .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
            .expect("stream construction")
            .try_collect::<Vec<RecordBatch>>()
            .await
            .expect("ranged read");
        batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    /// Expected id multiset for a window under the JAVA rule, derived from real footer midpoints.
    fn expected_ids_for_window(
        metadata: &ParquetMetaData,
        start: u64,
        length: u64,
        rows_per_group: i32,
    ) -> Vec<i32> {
        let end = start + length;
        let mut ids = Vec::new();
        for (idx, mid) in footer_row_group_midpoints(metadata).into_iter().enumerate() {
            if mid >= start && mid < end {
                let base = i32::try_from(idx).expect("index fits i32") * rows_per_group;
                ids.extend(base..base + rows_per_group);
            }
        }
        ids
    }

    /// A fixed-size split tiling that straddles row groups must read every row exactly once.
    ///
    /// An overlap rule makes each window claim every row group it touches, so this fixture
    /// returned 500 rows from a 300-row file, with no error. The midpoint rule gives each row
    /// group one window. Every expectation comes from real footer offsets.
    #[tokio::test]
    async fn test_midpoint_selection_straddling_splits_read_each_row_exactly_once() {
        let tmp = TempDir::new().unwrap();
        let path = tmp
            .path()
            .join("straddle.parquet")
            .to_string_lossy()
            .to_string();
        write_midpoint_fixture(&path, 3, 100, false);

        let metadata = footer_metadata(&path);
        assert_eq!(
            metadata.num_row_groups(),
            3,
            "fixture must have 3 row groups"
        );
        let file_size = std::fs::metadata(&path).unwrap().len();
        let schema = id_schema_for_pos();

        let windows = tile_windows(file_size, 800);
        assert!(
            windows.len() >= 3,
            "the 800-byte tiling must produce several windows over a {file_size}-byte file"
        );
        // Non-vacuity: a row group must straddle a boundary, or the two rules agree here.
        let starts = footer_row_group_starts(&metadata);
        let straddles = starts.iter().zip(metadata.row_groups()).any(|(s, rg)| {
            let e = s + u64::try_from(rg.compressed_size()).unwrap();
            windows
                .iter()
                .any(|(ws, wl)| *s < *ws && e > *ws && *wl > 0)
        });
        assert!(
            straddles,
            "fixture is non-discriminating: no row group straddles a window boundary \
             (starts={starts:?}, windows={windows:?})"
        );

        let mut union: Vec<i32> = Vec::new();
        for (start, length) in &windows {
            let got = read_ids_for_window(&path, schema.clone(), *start, *length).await;
            let want = expected_ids_for_window(&metadata, *start, *length, 100);
            assert_eq!(
                got,
                want,
                "window [{start}, {}) must contain exactly the row groups whose midpoint lands in it",
                start + length
            );
            union.extend(got);
        }

        // The exactly-once property. The overlap rule violates it, and fixture drift cannot
        // weaken the assertion.
        union.sort_unstable();
        assert_eq!(
            union,
            (0..300).collect::<Vec<i32>>(),
            "the union over a full tiling must be every row EXACTLY once (duplicates here mean a \
             straddling row group was handed to two adjacent splits)"
        );
    }

    /// A task with the legacy whole-file sentinel `start == 0, length == 0` must still read every
    /// row after [`FileScanTask::split`], the way `TableScan::plan_tasks` calls it. Without the
    /// sentinel guard, the fixed-size branch returned zero sub-tasks: the file left `plan_tasks`
    /// while `plan_files` still listed it, and the scan read 0 rows with no error. The reader
    /// accepts the same spelling as whole-file, so the two halves must agree.
    #[tokio::test]
    async fn test_whole_file_length_sentinel_survives_split_and_reads_every_row() {
        let tmp = TempDir::new().unwrap();
        let path = tmp
            .path()
            .join("sentinel.parquet")
            .to_string_lossy()
            .to_string();
        write_midpoint_fixture(&path, 3, 20, false);
        let schema = id_schema_for_pos();
        let file_size = std::fs::metadata(&path).unwrap().len();

        // The sentinel spelling: start 0, length 0, with the REAL file size alongside it.
        let sentinel = midpoint_scan_task(&path, schema.clone(), 0, 0);
        assert_eq!(
            (sentinel.start, sentinel.length),
            (0, 0),
            "fixture guard: this task must carry the legacy whole-file sentinel"
        );

        // Non-vacuity: the same file with an explicit length does split at this target.
        let target = file_size / 3 + 1;
        let sized = midpoint_scan_task(&path, schema.clone(), 0, file_size);
        assert!(
            sized.split(target).expect("sized split").len() > 1,
            "fixture is non-discriminating: target {target} must split a {file_size}-byte file"
        );

        let sub_tasks = sentinel.split(target).expect("sentinel split");
        assert_eq!(
            sub_tasks.len(),
            1,
            "the whole-file sentinel must survive split as ONE task, not evaporate"
        );

        let mut union: Vec<i32> = Vec::new();
        for sub_task in sub_tasks {
            let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
            let batches = reader
                .read(Box::pin(futures::stream::iter(vec![Ok(sub_task)])) as FileScanTaskStream)
                .expect("stream construction")
                .try_collect::<Vec<RecordBatch>>()
                .await
                .expect("sentinel read");
            for batch in &batches {
                union.extend(
                    batch
                        .column(0)
                        .as_primitive::<arrow_array::types::Int32Type>()
                        .values()
                        .iter()
                        .copied(),
                );
            }
        }
        union.sort_unstable();
        assert_eq!(
            union,
            (0..60).collect::<Vec<i32>>(),
            "splitting a whole-file sentinel task must still read every row exactly once (an \
             empty sub-task set reads ZERO rows and reports no error at all)"
        );
    }

    /// Splitting an already-ranged task must not relocate its byte window. Both split branches
    /// treat the byte space as absolute from zero. Without the `start != 0` passthrough, a parent
    /// covering `[starts[1], file_size)` came back anchored at 0: the products re-read a prefix the
    /// parent never owned and dropped the tail it did.
    #[tokio::test]
    async fn test_split_of_a_ranged_task_reads_the_parents_rows_not_the_whole_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp
            .path()
            .join("ranged_parent.parquet")
            .to_string_lossy()
            .to_string();
        write_midpoint_fixture(&path, 3, 20, false);
        let schema = id_schema_for_pos();
        let metadata = footer_metadata(&path);
        let file_size = std::fs::metadata(&path).unwrap().len();
        let starts = footer_row_group_starts(&metadata);
        assert_eq!(starts.len(), 3, "fixture must have 3 row groups");

        // The parent: a genuine sub-window covering the LAST TWO row groups only.
        let start = starts[1];
        let length = file_size - start;
        let parent_ids = read_ids_for_window(&path, schema.clone(), start, length).await;
        assert_eq!(
            parent_ids,
            (20..60).collect::<Vec<i32>>(),
            "fixture guard: the parent window must own the last two row groups only — a window \
             that already owned every row could not detect a relocation"
        );

        // Non-vacuity: the same target splits the whole-file parent into many windows.
        let target = file_size / 3 + 1;
        let whole = midpoint_scan_task(&path, schema.clone(), 0, file_size);
        assert!(
            whole.split(target).expect("whole split").len() > 1,
            "fixture is non-discriminating: target {target} must split the whole-file parent"
        );

        let ranged = midpoint_scan_task(&path, schema.clone(), start, length);
        let sub_tasks = ranged.split(target).expect("ranged split");
        assert_eq!(
            sub_tasks.len(),
            1,
            "an already-ranged parent must pass through split unchanged"
        );
        assert_eq!(
            (sub_tasks[0].start, sub_tasks[0].length),
            (start, length),
            "the sub-task window must stay the parent's; a window anchored at 0 covers bytes the \
             parent never owned"
        );

        let mut union: Vec<i32> = Vec::new();
        for sub_task in sub_tasks {
            let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
            let batches = reader
                .read(Box::pin(futures::stream::iter(vec![Ok(sub_task)])) as FileScanTaskStream)
                .expect("stream construction")
                .try_collect::<Vec<RecordBatch>>()
                .await
                .expect("ranged sub-task read");
            for batch in &batches {
                union.extend(
                    batch
                        .column(0)
                        .as_primitive::<arrow_array::types::Int32Type>()
                        .values()
                        .iter()
                        .copied(),
                );
            }
        }
        union.sort_unstable();
        assert_eq!(
            union, parent_ids,
            "the split of a ranged parent must read EXACTLY the parent's rows; reading the whole \
             file here means the window was relocated to offset 0"
        );

        // ---- the `start != 0` disjunct, ON ITS OWN ----
        //
        // The fixture above trips both disjuncts, so dropping `self.start != 0` leaves it green.
        // This parent moves only the left edge, which `start != 0` alone can see.
        let relocated = midpoint_scan_task(&path, schema.clone(), start, file_size);
        let sub_tasks = relocated.split(target).expect("relocated split");
        assert_eq!(
            sub_tasks.len(),
            1,
            "a parent whose left edge moved must pass through split as ONE task even when its \
             length still equals the file size"
        );
        assert_eq!(
            (sub_tasks[0].start, sub_tasks[0].length),
            (start, file_size),
            "the passthrough must keep the parent's own window; re-splitting relocates it to 0"
        );

        let mut relocated_union: Vec<i32> = Vec::new();
        for sub_task in sub_tasks {
            let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
            let batches = reader
                .read(Box::pin(futures::stream::iter(vec![Ok(sub_task)])) as FileScanTaskStream)
                .expect("stream construction")
                .try_collect::<Vec<RecordBatch>>()
                .await
                .expect("relocated sub-task read");
            for batch in &batches {
                relocated_union.extend(
                    batch
                        .column(0)
                        .as_primitive::<arrow_array::types::Int32Type>()
                        .values()
                        .iter()
                        .copied(),
                );
            }
        }
        relocated_union.sort_unstable();
        assert_eq!(
            relocated_union, parent_ids,
            "a window anchored at `starts[1]` covers the last two row groups' midpoints however \
             far past EOF it runs; splitting it would re-read the prefix the parent never owned"
        );
    }

    /// The byte-range entry gate `task.start != 0 || task.length != 0` must fire on
    /// `start > 0, length == 0`.
    ///
    /// The disjunction is what makes only `start == 0 && length == 0` mean "whole file". A
    /// non-zero start with a zero length is the empty window `[start, start)`, which Java spells
    /// `withRange(start, start)` and never selects. A gate of `task.length != 0` alone turns that
    /// into a full-file read, and no other test in the suite uses the shape.
    #[tokio::test]
    async fn test_byte_range_gate_fires_on_a_zero_length_window_at_a_nonzero_start() {
        let tmp = TempDir::new().unwrap();
        let path = tmp
            .path()
            .join("empty_window.parquet")
            .to_string_lossy()
            .to_string();
        write_midpoint_fixture(&path, 3, 20, false);
        let schema = id_schema_for_pos();
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert!(file_size > 1, "fixture guard: the file must be non-trivial");

        // Non-vacuity control: the SENTINEL pair (0, 0) bypasses the gate and reads the file.
        let whole = read_ids_for_window(&path, schema.clone(), 0, 0).await;
        assert_eq!(
            whole,
            (0..60).collect::<Vec<i32>>(),
            "the (0, 0) sentinel must bypass the gate and read the whole file"
        );

        // A non-zero start with a zero length is an empty window, so nothing is selected.
        for start in [1u64, file_size / 2, file_size - 1] {
            let ids = read_ids_for_window(&path, schema.clone(), start, 0).await;
            assert!(
                ids.is_empty(),
                "window [{start}, {start}) is empty and must select no row groups, got {} rows \
                 (a full-file read here means the gate stopped distinguishing start == 0)",
                ids.len()
            );
        }
    }

    /// The offset-source pin: a file whose row groups are not contiguous.
    ///
    /// parquet-rs writes each bloom filter after its row group, so the real starts run ahead of
    /// `4 + Σ compressed_size`. The windows here are the file's own row-group boundaries, which
    /// is what the offsets-aware split branch produces. Offsets-aligned splits over a padded file
    /// duplicated too, not only offsets-less external manifests.
    #[tokio::test]
    async fn test_midpoint_selection_reads_real_offsets_on_padded_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp
            .path()
            .join("padded.parquet")
            .to_string_lossy()
            .to_string();
        write_midpoint_fixture(&path, 3, 100, true);

        let metadata = footer_metadata(&path);
        assert_eq!(
            metadata.num_row_groups(),
            3,
            "fixture must have 3 row groups"
        );
        let real = footer_row_group_starts(&metadata);
        let synthetic = synthetic_row_group_starts(&metadata);
        // Assert the drift this test detects. Without padding it duplicates the tiling test.
        assert_ne!(
            real, synthetic,
            "fixture is non-discriminating: real row-group starts must differ from the \
             `4 + Σ compressed_size` model (real={real:?})"
        );

        let file_size = std::fs::metadata(&path).unwrap().len();
        let schema = id_schema_for_pos();
        // Tile at the file's own row-group boundaries, last window running to EOF.
        let windows: Vec<(u64, u64)> = real
            .iter()
            .enumerate()
            .map(|(i, start)| {
                let end = real.get(i + 1).copied().unwrap_or(file_size);
                (*start, end - start)
            })
            .collect();

        let mut union: Vec<i32> = Vec::new();
        for (i, (start, length)) in windows.iter().enumerate() {
            let got = read_ids_for_window(&path, schema.clone(), *start, *length).await;
            let base = i32::try_from(i).unwrap() * 100;
            assert_eq!(
                got,
                (base..base + 100).collect::<Vec<i32>>(),
                "offsets-aligned window {i} = [{start}, {}) must read exactly its own row group",
                start + length
            );
            union.extend(got);
        }
        union.sort_unstable();
        assert_eq!(
            union,
            (0..300).collect::<Vec<i32>>(),
            "offsets-aligned tiling of a padded file must read every row exactly once"
        );
    }

    /// The exactly-once property, over a sweep of strides and both fixture shapes.
    ///
    /// For any tiling of `[0, file_size)` the selected index sets must partition
    /// `0..num_row_groups`. A missing index under-reads. A repeated index duplicates rows.
    #[test]
    fn test_midpoint_selection_partitions_row_groups_over_stride_sweep() {
        let tmp = TempDir::new().unwrap();
        for (name, bloom) in [("contig.parquet", false), ("padded.parquet", true)] {
            let path = tmp.path().join(name).to_string_lossy().to_string();
            write_midpoint_fixture(&path, 4, 100, bloom);
            let metadata = footer_metadata(&path);
            let num_row_groups = metadata.num_row_groups();
            assert_eq!(num_row_groups, 4, "{name}: fixture must have 4 row groups");
            let file_size = std::fs::metadata(&path).unwrap().len();

            for stride in [256u64, 512, 800, 1024, 4096] {
                let mut seen: Vec<usize> = Vec::new();
                for (start, length) in tile_windows(file_size, stride) {
                    seen.extend(
                        ArrowReader::filter_row_groups_by_byte_range(&metadata, start, length)
                            .expect("byte-range filter"),
                    );
                }
                seen.sort_unstable();
                assert_eq!(
                    seen,
                    (0..num_row_groups).collect::<Vec<usize>>(),
                    "{name} @ stride {stride}: the tiling must select every row group exactly once"
                );
            }

            // Every window boundary sits on a row-group midpoint. Only the half-open convention
            // partitions here: a strict low bound drops each row group, an inclusive high bound
            // selects each twice.
            let mut boundaries = footer_row_group_midpoints(&metadata);
            boundaries.retain(|b| *b > 0 && *b < file_size);
            boundaries.insert(0, 0);
            boundaries.push(file_size);
            boundaries.dedup();
            let mut seen: Vec<usize> = Vec::new();
            for pair in boundaries.windows(2) {
                seen.extend(
                    ArrowReader::filter_row_groups_by_byte_range(
                        &metadata,
                        pair[0],
                        pair[1] - pair[0],
                    )
                    .expect("byte-range filter"),
                );
            }
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..num_row_groups).collect::<Vec<usize>>(),
                "{name}: a tiling whose boundaries sit exactly on row-group midpoints must still \
                 select every row group exactly once (half-open [start, end))"
            );
        }
    }

    /// A schema descriptor with `n` `INT32` leaf columns (`get_test_schema_descr` is fixed at two,
    /// and `RowGroupMetaData::build` validates the column count against the descriptor).
    fn schema_descr_with_columns(n: usize) -> SchemaDescPtr {
        use parquet::schema::types::Type as SchemaType;

        let fields = (0..n)
            .map(|i| {
                Arc::new(
                    SchemaType::primitive_type_builder(
                        &format!("c{i}"),
                        parquet::basic::Type::INT32,
                    )
                    .build()
                    .expect("primitive field"),
                )
            })
            .collect::<Vec<_>>();
        let schema = SchemaType::group_type_builder("schema")
            .with_fields(fields)
            .build()
            .expect("group type");
        Arc::new(SchemaDescriptor::new(Arc::new(schema)))
    }

    /// The gap from the first fabricated column chunk to each trailing one. Reading any other
    /// column moves the midpoint out of every window the semantics test declares.
    const FABRICATED_COLUMN_STRIDE: i64 = 1_000_000;

    /// Builds row groups from `(data_page_offset, compressed_size, dict_offset)` triples — the
    /// triple always describes `columns()[0]` — so the selection rule can be probed at exact byte
    /// positions. Two properties keep the fixture discriminating. Each row group carries three
    /// column chunks, the trailing two [`FABRICATED_COLUMN_STRIDE`] bytes apart, so reading any
    /// column but `columns()[0]` changes the answer.
    fn midpoint_test_metadata(groups: &[(i64, i64, Option<i64>)]) -> Arc<ParquetMetaData> {
        const TRAILING_COLUMNS: usize = 2;

        let schema_descr = schema_descr_with_columns(1 + TRAILING_COLUMNS);
        let row_groups: Vec<RowGroupMetaData> = groups
            .iter()
            .enumerate()
            .map(|(idx, (data_page_offset, size, dict))| {
                let mut columns = vec![
                    ColumnChunkMetaData::builder(schema_descr.column(0))
                        .set_data_page_offset(*data_page_offset)
                        .set_dictionary_page_offset(*dict)
                        .set_total_compressed_size(*size)
                        .build()
                        .expect("column chunk metadata"),
                ];
                for col in 1..=TRAILING_COLUMNS {
                    let stride = FABRICATED_COLUMN_STRIDE
                        * i64::try_from(col).expect("column index fits i64");
                    columns.push(
                        ColumnChunkMetaData::builder(schema_descr.column(col))
                            .set_data_page_offset(data_page_offset.saturating_add(stride))
                            .set_total_compressed_size(0)
                            .build()
                            .expect("trailing column chunk metadata"),
                    );
                }
                RowGroupMetaData::builder(schema_descr.clone())
                    .set_num_rows(10)
                    .set_total_byte_size(size.saturating_mul(4).saturating_add(7))
                    .set_column_metadata(columns)
                    .set_ordinal(i16::try_from(idx).expect("ordinal fits i16"))
                    .build()
                    .expect("row group metadata")
            })
            .collect();
        let file_metadata = FileMetaData::new(1, 0, None, None, schema_descr, None);
        Arc::new(ParquetMetaData::new(file_metadata, row_groups))
    }

    /// The row-group start comes from the first column chunk, proved on a real multi-column file.
    ///
    /// Java is `getOffset(rowGroup.getColumns().get(0))`. On a wide file the last column chunk
    /// starts thousands of bytes downstream, so indexing the wrong column pushes every midpoint
    /// into the following window. The first window then reads nothing and a later one claims two
    /// row groups. A single-column fixture cannot see it.
    #[test]
    fn test_midpoint_selection_uses_first_column_chunk_on_real_file() {
        use arrow_array::{Int32Array, StringArray};

        let tmp = TempDir::new().unwrap();
        let path = tmp
            .path()
            .join("multi_column.parquet")
            .to_string_lossy()
            .to_string();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("payload", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();
        let file = File::create(&path).expect("create multi-column fixture");
        let mut writer =
            ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).expect("arrow writer");
        for group in 0..3i32 {
            let base = group * 100;
            let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Int32Array::from_iter_values(base..base + 100)),
                // Distinct, poorly compressible values so the payload column chunk is large and
                // its start is far from the `id` chunk's.
                Arc::new(StringArray::from_iter_values((base..base + 100).map(|i| {
                    format!("payload-{i:09}-{:x}", i.wrapping_mul(2_654_435_i32))
                }))),
            ])
            .expect("multi-column batch");
            writer.write(&batch).expect("write row group");
        }
        writer.close().expect("close multi-column fixture");

        let metadata = footer_metadata(&path);
        assert_eq!(
            metadata.num_row_groups(),
            3,
            "fixture must have 3 row groups"
        );

        let starts = footer_row_group_starts(&metadata);
        let midpoints = footer_row_group_midpoints(&metadata);
        for (idx, rg) in metadata.row_groups().iter().enumerate() {
            let columns = rg.columns();
            assert!(columns.len() > 1, "fixture must be multi-column");
            let last_start = u64::try_from(
                columns[columns.len() - 1]
                    .dictionary_page_offset()
                    .filter(|dict| *dict < columns[columns.len() - 1].data_page_offset())
                    .unwrap_or_else(|| columns[columns.len() - 1].data_page_offset()),
            )
            .expect("non-negative offset");
            // Non-vacuity guard: if the columns ever coincide this test degenerates.
            assert!(
                last_start > starts[idx],
                "fixture is non-discriminating: row group {idx}'s last column chunk must start \
                 well after its first ({last_start} vs {})",
                starts[idx]
            );
        }

        // A one-byte window at the true midpoint selects exactly that row group. Any other column
        // chunk moves the midpoint and empties the window.
        for (idx, mid) in midpoints.iter().enumerate() {
            assert_eq!(
                ArrowReader::filter_row_groups_by_byte_range(&metadata, *mid, 1)
                    .expect("byte-range filter"),
                vec![idx],
                "the window [{mid}, {}) must select exactly row group {idx}; its start is the \
                 FIRST column chunk's offset",
                mid + 1
            );
        }
    }

    /// U3 / T4 — `getOffset` semantics, window bounds, and the typed-error paths, probed directly
    /// on fabricated footer metadata.
    #[test]
    fn test_midpoint_selection_offset_and_boundary_semantics() {
        // (a) dictionary offset SMALLER than the data page offset wins: start = 100,
        //     size = 20 → midpoint 110.
        let md = midpoint_test_metadata(&[(140, 20, Some(100))]);
        assert_eq!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 105, 10).expect("filter"),
            vec![0],
            "dictionary offset below the data page offset is the row-group start (midpoint 110)"
        );
        assert!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 141, 100)
                .expect("filter")
                .is_empty(),
            "a window past the dictionary-based midpoint must select nothing"
        );

        // (b) dictionary offset NOT smaller (Java takes the MIN, it is not "dict wins when set"):
        //     start = 100, size = 20 → midpoint 110. A naive "dict wins" port would say 310.
        let md = midpoint_test_metadata(&[(100, 20, Some(300))]);
        assert_eq!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 105, 10).expect("filter"),
            vec![0],
            "when the dictionary offset is NOT smaller, the data page offset is the start"
        );
        assert!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 300, 100)
                .expect("filter")
                .is_empty(),
            "a larger dictionary offset must never become the row-group start"
        );

        // (c) no dictionary page: the data page offset is the start.
        let md = midpoint_test_metadata(&[(100, 20, None)]);
        assert_eq!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 110, 1).expect("filter"),
            vec![0],
            "without a dictionary page the data page offset is the start (midpoint 110)"
        );

        // (d) a midpoint landing EXACTLY on a split boundary belongs to the HIGHER window:
        //     start 100, size 20 → midpoint exactly 110.
        let md = midpoint_test_metadata(&[(100, 20, None)]);
        assert!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 0, 110)
                .expect("filter")
                .is_empty(),
            "the window ENDING at the midpoint must not claim it (high bound is exclusive)"
        );
        assert_eq!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 110, 10).expect("filter"),
            vec![0],
            "the window STARTING at the midpoint claims it (low bound is inclusive)"
        );

        // (d2) the division on the SIZE TRUNCATES (Java `ldiv`). With an ODD size the truncating
        //      and rounding-up forms differ by one byte: start 100, size 21 → midpoint 110, NOT
        //      111. Every other case here uses an even size, so this is the only arm that can see
        //      the difference.
        let md = midpoint_test_metadata(&[(100, 21, None)]);
        assert_eq!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 0, 111).expect("filter"),
            vec![0],
            "an odd compressed size must truncate: midpoint 110 lies inside [0, 111)"
        );
        assert!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 111, 10)
                .expect("filter")
                .is_empty(),
            "an odd compressed size must not round the midpoint up into [111, 121)"
        );

        // (d3) `totalSize` is the COMPRESSED size. The fabricated row groups declare a distinctly
        //      larger `total_byte_size`, so reading the uncompressed size instead moves the
        //      midpoint out of the window.
        let md = midpoint_test_metadata(&[(100, 20, None)]);
        assert_eq!(
            md.row_group(0).compressed_size(),
            20,
            "fixture guard: the row-group compressed size must be the requested value"
        );
        assert_ne!(
            md.row_group(0).total_byte_size(),
            md.row_group(0).compressed_size(),
            "fixture is non-discriminating: the uncompressed size must differ from the compressed \
             size, otherwise reading the wrong one is invisible"
        );

        // (d4) the row-group start comes from `columns()[0]` (Java `getColumns().get(0)`). The
        //      fabricated trailing chunks sit far downstream, so reading any other column moves
        //      the midpoint out of the window.
        let md = midpoint_test_metadata(&[(100, 20, None)]);
        let columns = md.row_group(0).columns();
        assert!(
            columns.len() > 1
                && columns[columns.len() - 1].data_page_offset() > columns[0].data_page_offset(),
            "fixture is non-discriminating: row groups must carry several column chunks at \
             different offsets, otherwise the column index is invisible"
        );
        assert_eq!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 105, 10).expect("filter"),
            vec![0],
            "the first column chunk determines the row-group start (midpoint 110)"
        );

        // (d5) a dictionary page offset of exactly zero is still set, and still wins. parquet-mr
        //      has two offset helpers that differ here. The read path uses
        //      `ParquetMetadataConverter.getOffset`, which has no `> 0` test. The split-offset
        //      writer uses `ColumnChunkMetaData.getStartingPos`, which does. A `> 0` test here
        //      moves this row group's midpoint from 50 to 1050, into a different split.
        let md = midpoint_test_metadata(&[(1000, 100, Some(0))]);
        assert_eq!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 0, 51).expect("filter"),
            vec![0],
            "a dictionary page offset of 0 is SET and below the data page offset, so the row group \
             starts at 0 (midpoint 50) — the `getStartingPos` variant with `> 0` would start it at \
             1000"
        );
        assert!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 1000, 100)
                .expect("filter")
                .is_empty(),
            "the window around the DATA page offset must select nothing once the zero dictionary \
             offset is honoured"
        );

        // (e) a row group with no column chunks is a typed error, not a panic. (Java indexes
        //     `getColumns().get(0)` unguarded and throws IndexOutOfBounds.)
        let schema_descr = schema_descr_with_columns(0);
        let empty_group = RowGroupMetaData::builder(schema_descr.clone())
            .set_num_rows(10)
            .set_total_byte_size(0)
            .set_column_metadata(vec![])
            .set_ordinal(0)
            .build()
            .expect("row group with no columns");
        let md = Arc::new(ParquetMetaData::new(
            FileMetaData::new(1, 0, None, None, schema_descr, None),
            vec![empty_group],
        ));
        let err = ArrowReader::filter_row_groups_by_byte_range(&md, 0, 1000)
            .expect_err("a column-less row group must error");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string().contains("no column chunks"),
            "the typed error must name the missing column chunks, got: {err}"
        );

        // (f) a negative data page offset is a typed error, not a panic. (Routing through
        //     `ColumnChunkMetaData::byte_range()` would abort here on its internal `assert!`.)
        let md = midpoint_test_metadata(&[(-1, 20, None)]);
        let err = ArrowReader::filter_row_groups_by_byte_range(&md, 0, 1000)
            .expect_err("a negative row-group offset must error");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string().contains("negative byte offset"),
            "the typed error must name the negative offset, got: {err}"
        );

        // (g) the largest footer values that can be decoded at all: both inputs are `i64`, so
        //     `offset + size / 2` is bounded by `2^63 + 2^62 < u64::MAX` and the `checked_add`
        //     midpoint guard is unreachable by construction — it stays as a defensive assertion.
        //     What must hold here is that the extreme values still produce an ANSWER, not a panic.
        let md = midpoint_test_metadata(&[(i64::MAX, i64::MAX, None)]);
        assert_eq!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 0, u64::MAX)
                .expect("extreme but decodable offsets must not panic or error"),
            vec![0],
            "a midpoint inside [0, u64::MAX) must still be selected"
        );

        // (h) a negative compressed size is a typed error, not a silent selection by start.
        //     Without the guard the size collapses to 0 and the midpoint equals the row-group
        //     start: wrong rows, no error. The public builder accepts a negative
        //     `total_compressed_size`, so the branch is constructible.
        let md = midpoint_test_metadata(&[(100, -20, None)]);
        assert!(
            md.row_group(0).compressed_size() < 0,
            "fixture guard: the fabricated row group must really declare a negative size, got {}",
            md.row_group(0).compressed_size()
        );
        let err = ArrowReader::filter_row_groups_by_byte_range(&md, 0, 1000)
            .expect_err("a negative row-group compressed size must error");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string().contains("compressed size is negative"),
            "the typed error must name the negative size, got: {err}"
        );
        // Non-vacuity: a window starting AT the row-group start would select it if the size
        // collapsed to 0, so the error above is the only thing preventing selection-by-start.
        assert!(
            ArrowReader::filter_row_groups_by_byte_range(&md, 100, 1).is_err(),
            "the negative size must fail closed for every window, including one at the start"
        );

        // (i) a footer whose chunks sum past `i64::MAX` is a typed error, not a panic.
        //     `RowGroupMetaData::compressed_size()` sums with an unchecked `i64` `sum()`, and
        //     parquet-rs range-validates nothing when it decodes the thrift field.
        let schema_descr = schema_descr_with_columns(3);
        let huge_columns: Vec<ColumnChunkMetaData> = (0..3)
            .map(|col| {
                ColumnChunkMetaData::builder(schema_descr.column(col))
                    .set_data_page_offset(100)
                    .set_total_compressed_size(i64::MAX)
                    .build()
                    .expect("huge column chunk metadata")
            })
            .collect();
        let huge_group = RowGroupMetaData::builder(schema_descr.clone())
            .set_num_rows(10)
            .set_total_byte_size(0)
            .set_column_metadata(huge_columns)
            .set_ordinal(0)
            .build()
            .expect("row group with an overflowing column-size sum");
        let md = Arc::new(ParquetMetaData::new(
            FileMetaData::new(1, 0, None, None, schema_descr, None),
            vec![huge_group],
        ));
        let err = ArrowReader::filter_row_groups_by_byte_range(&md, 0, u64::MAX)
            .expect_err("an overflowing column-size sum must error, not panic");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            err.to_string().contains("overflows i64"),
            "the typed error must name the overflow, got: {err}"
        );
    }

    /// Verifies that file splits respect byte ranges and only read specific row groups.
    #[tokio::test]
    async fn test_file_splits_respect_byte_ranges() {
        use arrow_array::Int32Array;
        use parquet::file::reader::{FileReader, SerializedFileReader};

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/multi_row_group.parquet");

        let batch1 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Int32Array::from(
            (0..100).collect::<Vec<i32>>(),
        ))])
        .unwrap();
        let batch2 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Int32Array::from(
            (100..200).collect::<Vec<i32>>(),
        ))])
        .unwrap();
        let batch3 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Int32Array::from(
            (200..300).collect::<Vec<i32>>(),
        ))])
        .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();

        let file = File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        writer.write(&batch1).expect("Writing batch 1");
        writer.write(&batch2).expect("Writing batch 2");
        writer.write(&batch3).expect("Writing batch 3");
        writer.close().unwrap();

        let file = File::open(&file_path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let metadata = reader.metadata();

        println!("File has {} row groups", metadata.num_row_groups());
        assert_eq!(metadata.num_row_groups(), 3, "Expected 3 row groups");

        let row_group_0 = metadata.row_group(0);
        let row_group_1 = metadata.row_group(1);
        let row_group_2 = metadata.row_group(2);

        // Window boundaries come from the real footer, not from `4 + Σ compressed_size`. That
        // model agrees on a contiguous fixture, so it could never catch offset drift.
        let real_starts = footer_row_group_starts(metadata);
        let rg0_start = real_starts[0];
        let rg1_start = real_starts[1];
        let rg2_start = real_starts[2];
        let file_end = rg2_start + row_group_2.compressed_size() as u64;

        println!(
            "Row group 0: {} rows, starts at byte {}, {} bytes compressed",
            row_group_0.num_rows(),
            rg0_start,
            row_group_0.compressed_size()
        );
        println!(
            "Row group 1: {} rows, starts at byte {}, {} bytes compressed",
            row_group_1.num_rows(),
            rg1_start,
            row_group_1.compressed_size()
        );
        println!(
            "Row group 2: {} rows, starts at byte {}, {} bytes compressed",
            row_group_2.num_rows(),
            rg2_start,
            row_group_2.compressed_size()
        );

        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io).build();

        let task1 = FileScanTask {
            file_size_in_bytes: std::fs::metadata(&file_path).unwrap().len(),
            start: rg0_start,
            length: row_group_0.compressed_size() as u64,
            record_count: Some(100),
            data_file_path: Arc::from(file_path.clone()),
            data_file_format: DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: Arc::from(vec![1]),
            predicate: None,
            deletes: Arc::from(vec![]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };

        let task2 = FileScanTask {
            file_size_in_bytes: std::fs::metadata(&file_path).unwrap().len(),
            start: rg1_start,
            length: file_end - rg1_start,
            record_count: Some(200),
            data_file_path: Arc::from(file_path.clone()),
            data_file_format: DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: Arc::from(vec![1]),
            predicate: None,
            deletes: Arc::from(vec![]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };

        let tasks1 = Box::pin(futures::stream::iter(vec![Ok(task1)])) as FileScanTaskStream;
        let result1 = reader
            .clone()
            .read(tasks1)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let total_rows_task1: usize = result1.iter().map(|b| b.num_rows()).sum();
        println!(
            "Task 1 (bytes {}-{}) returned {} rows",
            rg0_start,
            rg0_start + row_group_0.compressed_size() as u64,
            total_rows_task1
        );

        let tasks2 = Box::pin(futures::stream::iter(vec![Ok(task2)])) as FileScanTaskStream;
        let result2 = reader
            .read(tasks2)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let total_rows_task2: usize = result2.iter().map(|b| b.num_rows()).sum();
        println!("Task 2 (bytes {rg1_start}-{file_end}) returned {total_rows_task2} rows");

        assert_eq!(
            total_rows_task1, 100,
            "Task 1 should read only the first row group (100 rows), but got {total_rows_task1} rows"
        );

        assert_eq!(
            total_rows_task2, 200,
            "Task 2 should read only the second+third row groups (200 rows), but got {total_rows_task2} rows"
        );

        if total_rows_task1 > 0 {
            let first_batch = &result1[0];
            let id_col = first_batch
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            let first_val = id_col.value(0);
            let last_val = id_col.value(id_col.len() - 1);
            println!("Task 1 data range: {first_val} to {last_val}");

            assert_eq!(first_val, 0, "Task 1 should start with id=0");
            assert_eq!(last_val, 99, "Task 1 should end with id=99");
        }

        if total_rows_task2 > 0 {
            let first_batch = &result2[0];
            let id_col = first_batch
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            let first_val = id_col.value(0);
            println!("Task 2 first value: {first_val}");

            assert_eq!(first_val, 100, "Task 2 should start with id=100, not id=0");
        }
    }

    /// Reads an old file holding only column `a` under a schema with `a` and `b`.
    /// `get_arrow_projection_mask` must allow the missing column, and RecordBatchTransformer must
    /// add `b` as NULL.
    #[tokio::test]
    async fn test_schema_evolution_add_column() {
        use arrow_array::{Array, Int32Array};

        let new_schema = Arc::new(
            Schema::builder()
                .with_schema_id(2)
                .with_fields(vec![
                    NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "b", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema_old = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let data_a = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let to_write = RecordBatch::try_new(arrow_schema_old.clone(), vec![data_a]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(format!("{table_location}/old_file.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io).build();
        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/old_file.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/old_file.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: new_schema.clone(),
                project_field_ids: Arc::from(vec![1, 2]), // Request both columns 'a' and 'b'
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);

        let col_a = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(col_a.values(), &[1, 2, 3]);

        // Column 'b' should be all NULLs (it didn't exist in the old file)
        let col_b = batch
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(col_b.null_count(), 3);
        assert!(col_b.is_null(0));
        assert!(col_b.is_null(1));
        assert!(col_b.is_null(2));
    }

    #[tokio::test]
    async fn test_scan_projects_pos_metadata_column() {
        use arrow_array::{Array, Int32Array, Int64Array};

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Int32Array::from(vec![
                10, 20, 30, 40, 50,
            ])) as ArrayRef])
            .unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(format!("{table_location}/data.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io).build();
        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/data.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/data.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![
                    1,
                    crate::metadata_columns::RESERVED_FIELD_ID_POS,
                ]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 5);

        let id_col = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(id_col.values(), &[10, 20, 30, 40, 50]);

        // `_pos` is the 0-based physical ordinal of each row in the data file.
        let pos_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(pos_col.values(), &[0, 1, 2, 3, 4]);
    }

    // The `_pos` projection streaming half.
    //
    // Covered: dense and sparse pos-deletes with a residual, row sets and `_pos` values against an
    // unpruned oracle, multi-batch continuity, a MERGE-shaped write pin, and ordinal-advance bait.

    /// Collect `(id, _pos)` pairs from a scan projecting field 1 + `_pos`, sorted by id.
    fn collect_id_pos_pairs(batches: &[RecordBatch]) -> Vec<(i32, i64)> {
        use arrow_array::{Int32Array, Int64Array};
        let mut out = Vec::new();
        for batch in batches {
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id Int32");
            let pos = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("_pos Int64");
            assert_eq!(id.len(), pos.len());
            for i in 0..id.len() {
                out.push((id.value(i), pos.value(i)));
            }
        }
        out.sort_by_key(|(id, _)| *id);
        out
    }

    /// Write a single-column Int32 parquet file (field id 1 = `id`) with optional multi-RG layout.
    fn write_id_parquet_for_pos(path: &str, ids: &[i32], max_row_group_size: usize) {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));
        let batch =
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(arrow_array::Int32Array::from(ids.to_vec())) as ArrayRef,
            ])
            .expect("id batch");
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(max_row_group_size))
            .build();
        let file = File::create(path).expect("create data");
        let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
    }

    fn id_schema_for_pos() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .expect("schema"),
        )
    }

    fn pos_scan_task(
        data_path: &str,
        schema: SchemaRef,
        deletes: Vec<FileScanTaskDeleteFile>,
        predicate: Option<BoundPredicate>,
    ) -> FileScanTask {
        FileScanTask {
            file_size_in_bytes: std::fs::metadata(data_path).expect("stat").len(),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(data_path.to_string()),
            data_file_format: DataFileFormat::Parquet,
            schema,
            project_field_ids: Arc::from(vec![1, RESERVED_FIELD_ID_POS]),
            predicate: predicate.map(Arc::new),
            deletes: Arc::from(deletes),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        }
    }

    /// Local pos-delete writer for FK5 pins (nested avro test helper is not visible here).
    fn fk5_write_pos_delete_file(
        path: &str,
        referenced_data_path: &str,
        positions: &[i64],
    ) -> FileScanTaskDeleteFile {
        use arrow_array::Int64Array;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                (i32::MAX - 101).to_string(),
            )])),
            Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                (i32::MAX - 102).to_string(),
            )])),
        ]));
        let paths: Vec<&str> = positions.iter().map(|_| referenced_data_path).collect();
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions.to_vec())) as ArrayRef,
        ])
        .expect("build pos-delete batch");
        let file = File::create(path).expect("create pos-delete file");
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                .expect("pos-delete writer");
        writer.write(&batch).expect("write pos-delete batch");
        writer.close().expect("close pos-delete writer");
        FileScanTaskDeleteFile {
            file_path: path.to_string(),
            file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }
    }

    async fn run_pos_scan(task: FileScanTask, batch_size: Option<usize>) -> Vec<RecordBatch> {
        let file_io = FileIO::new_with_fs();
        let mut builder = ArrowReaderBuilder::new(file_io);
        if let Some(bs) = batch_size {
            builder = builder.with_batch_size(bs);
        }
        // Enable row selection and the row-group filter, so a regression that pushes them onto
        // the `_pos` path fails the ordinal oracle.
        let reader = builder
            .with_row_group_filtering_enabled(true)
            .with_row_selection_enabled(true)
            .build();
        reader
            .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
            .expect("read")
            .try_collect::<Vec<RecordBatch>>()
            .await
            .expect("collect")
    }

    /// FK5: multi-batch streaming still assigns continuous physical `_pos` 0..N-1.
    #[tokio::test]
    async fn fk5_pos_projection_multi_batch_continuity() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids: Vec<i32> = (0..20).map(|i| (i + 1) * 10).collect(); // 10,20,...,200
        write_id_parquet_for_pos(&data_path, &ids, 1000);
        let schema = id_schema_for_pos();
        let task = pos_scan_task(&data_path, schema, vec![], None);
        let batches = run_pos_scan(task, Some(3)).await;
        assert!(
            batches.len() > 1,
            "expected multi-batch stream, got {} batch(es)",
            batches.len()
        );
        let pairs = collect_id_pos_pairs(&batches);
        let expected: Vec<(i32, i64)> = ids
            .iter()
            .enumerate()
            .map(|(pos, id)| (*id, pos as i64))
            .collect();
        assert_eq!(
            pairs, expected,
            "physical _pos must be 0..N-1 across batches"
        );
    }

    /// Write a Parquet file carrying `id` PLUS a physically-stored `_row_id` column, the shape a
    /// lineage-preserving rewrite produces.
    fn write_parquet_with_stored_row_id(path: &str, ids: &[i32], row_ids: &[Option<i64>]) {
        write_parquet_with_stored_reserved_column(
            path,
            ids,
            row_ids,
            "_row_id",
            RESERVED_FIELD_ID_ROW_ID,
        )
    }

    /// Write a Parquet file carrying `id` plus a physically-stored RESERVED column, the shape a
    /// lineage-preserving rewrite produces. Parameterised over which column, so both halves of the
    /// projection fix are pinned.
    fn write_parquet_with_stored_reserved_column(
        path: &str,
        ids: &[i32],
        values: &[Option<i64>],
        column_name: &str,
        column_field_id: i32,
    ) {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new(column_name, DataType::Int64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                column_field_id.to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(arrow_array::Int32Array::from(ids.to_vec())) as ArrayRef,
            Arc::new(arrow_array::Int64Array::from(values.to_vec())) as ArrayRef,
        ])
        .expect("batch");
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(path).expect("create data");
        let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
    }

    /// Stripping every `is_metadata_field` id from the projection left a stored `_row_id`
    /// undecoded, so every row got a computed `first_row_id + pos`. Only a read through the real
    /// `ArrowReader` sees it.
    #[tokio::test]
    async fn stored_row_id_in_the_data_file_survives_the_real_reader() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("lineage.parquet")
            .to_string_lossy()
            .to_string();
        write_parquet_with_stored_row_id(&data_path, &[1, 2, 3], &[Some(777), None, Some(999)]);

        let mut task = pos_scan_task(&data_path, id_schema_for_pos(), vec![], None);
        task.project_field_ids = Arc::from(vec![1, RESERVED_FIELD_ID_ROW_ID]);
        task.first_row_id = Some(1_000);

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
        let batches: Vec<RecordBatch> = reader
            .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
            .expect("stream")
            .try_collect()
            .await
            .expect("read");

        let row_ids: Vec<Option<i64>> = batches
            .iter()
            .flat_map(|batch| {
                let column = batch
                    .column_by_name("_row_id")
                    .expect("_row_id projected")
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .expect("Int64")
                    .clone();
                use arrow_array::Array as _;
                (0..column.len())
                    .map(|row| {
                        if arrow_array::Array::is_null(&column, row) {
                            None
                        } else {
                            Some(column.value(row))
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        assert_eq!(
            row_ids,
            vec![Some(777), Some(1_001), Some(999)],
            "the STORED ids must win (777, 999); only the NULL row falls back to \
             first_row_id + its own position (1000 + 1). Getting [1000, 1001, 1002] here means \
             the stored column was never decoded."
        );
    }

    /// The other half of the projection fix: a stored `_last_updated_sequence_number` must win over
    /// the file's own sequence number. Narrowing the exemption to `_row_id` alone passed the suite.
    #[tokio::test]
    async fn stored_last_updated_sequence_number_survives_the_real_reader() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("stored_seq.parquet")
            .to_string_lossy()
            .to_string();
        write_parquet_with_stored_reserved_column(
            &data_path,
            &[1, 2, 3],
            &[Some(31), None, Some(33)],
            "_last_updated_sequence_number",
            crate::metadata_columns::RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
        );

        let mut task = pos_scan_task(&data_path, id_schema_for_pos(), vec![], None);
        task.project_field_ids = Arc::from(vec![
            1,
            crate::metadata_columns::RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
        ]);
        task.first_row_id = Some(1_000);
        task.file_sequence_number = Some(42);

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
        let batches: Vec<RecordBatch> = reader
            .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
            .expect("stream")
            .try_collect()
            .await
            .expect("read");

        let values: Vec<Option<i64>> = batches
            .iter()
            .flat_map(|batch| {
                let column = batch
                    .column_by_name("_last_updated_sequence_number")
                    .expect("projected")
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .expect("Int64")
                    .clone();
                (0..column.len())
                    .map(|row| {
                        if arrow_array::Array::is_null(&column, row) {
                            None
                        } else {
                            Some(column.value(row))
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        assert_eq!(
            values,
            vec![Some(31), Some(42), Some(33)],
            "STORED values win (31, 33); only the NULL row falls back to the file's own sequence \
             number (42). All-42 means the stored column was never decoded."
        );
    }

    /// `_last_updated_sequence_number` needs no physical ordinals, so it takes a different
    /// transformer construction site than the `_row_id` test above. Deleting either
    /// `with_row_lineage` call passed the whole suite.
    #[tokio::test]
    async fn last_updated_sequence_number_is_materialized_on_the_ordinary_parquet_path() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp.path().join("seq.parquet").to_string_lossy().to_string();
        write_id_parquet_for_pos(&data_path, &[1, 2, 3], 1000);

        let mut task = pos_scan_task(&data_path, id_schema_for_pos(), vec![], None);
        task.project_field_ids = Arc::from(vec![
            1,
            crate::metadata_columns::RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
        ]);
        task.first_row_id = Some(1_000);
        task.file_sequence_number = Some(42);

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
        let batches: Vec<RecordBatch> = reader
            .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
            .expect("stream")
            .try_collect()
            .await
            .expect("read");

        let values: Vec<i64> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("_last_updated_sequence_number")
                    .expect("projected")
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .expect("Int64")
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(
            values,
            vec![42, 42, 42],
            "the FILE's sequence number, threaded from the task — not 0, and not the row position"
        );
    }

    /// The canonical variant Arrow type removed the door that stopped a scan reading variant
    /// data. Without the reader-side refusal, that path opens silently.
    #[tokio::test]
    async fn scanning_a_variant_column_is_refused() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp.path().join("v.parquet").to_string_lossy().to_string();
        write_id_parquet_for_pos(&data_path, &[1, 2], 1000);

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "v", Type::Variant).into(),
                ])
                .build()
                .expect("schema"),
        );
        // Every data-file format. A Parquet-only pin lets the guard be gated on Parquet, and Avro
        // and ORC would then decode a variant column silently.
        for format in [
            DataFileFormat::Parquet,
            DataFileFormat::Avro,
            DataFileFormat::Orc,
        ] {
            let mut task = pos_scan_task(&data_path, schema.clone(), vec![], None);
            task.project_field_ids = Arc::from(vec![1, 2]);
            task.data_file_format = format;

            let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
            let err = reader
                .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
                .expect("stream construction")
                .try_collect::<Vec<RecordBatch>>()
                .await
                .expect_err("a variant projection must be refused on EVERY format");
            assert_eq!(
                err.kind(),
                ErrorKind::FeatureUnsupported,
                "format {format:?} must refuse, not decode a variant column"
            );
            assert!(
                err.to_string().contains("variant column 'v'"),
                "the error must name the column on {format:?}, got: {err}"
            );
        }
    }

    /// A variant nested in a struct, list element or map value must be refused too: a guard
    /// checking only the projected field's own type lets `struct<v: variant>` through.
    #[tokio::test]
    async fn scanning_a_variant_nested_in_any_container_is_refused() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("nested.parquet")
            .to_string_lossy()
            .to_string();
        write_id_parquet_for_pos(&data_path, &[1, 2], 1000);

        let cases: Vec<(&str, Type, &str)> = vec![
            (
                "struct",
                Type::Struct(crate::spec::StructType::new(vec![
                    NestedField::optional(3, "v", Type::Variant).into(),
                ])),
                "s.v",
            ),
            (
                "list",
                Type::List(crate::spec::ListType {
                    element_field: NestedField::list_element(3, Type::Variant, true).into(),
                }),
                "s.element",
            ),
            (
                "map value",
                Type::Map(crate::spec::MapType {
                    key_field: NestedField::map_key_element(
                        3,
                        Type::Primitive(PrimitiveType::String),
                    )
                    .into(),
                    value_field: NestedField::map_value_element(4, Type::Variant, true).into(),
                }),
                "s.value",
            ),
            // The fourth container position. Java's `Types$MapType` factories constrain only the
            // VALUE type, so `map<variant, _>` is constructible.
            (
                "map key",
                Type::Map(crate::spec::MapType {
                    key_field: NestedField::map_key_element(3, Type::Variant).into(),
                    value_field: NestedField::map_value_element(
                        4,
                        Type::Primitive(PrimitiveType::String),
                        true,
                    )
                    .into(),
                }),
                "s.key",
            ),
            // Nested TWO deep, so a one-level descent is not enough.
            (
                "struct in a list",
                Type::List(crate::spec::ListType {
                    element_field: NestedField::list_element(
                        3,
                        Type::Struct(crate::spec::StructType::new(vec![
                            NestedField::optional(4, "v", Type::Variant).into(),
                        ])),
                        true,
                    )
                    .into(),
                }),
                "s.element.v",
            ),
        ];

        for (label, container, expected_path) in cases {
            let schema = Arc::new(
                Schema::builder()
                    .with_schema_id(1)
                    .with_fields(vec![
                        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                        NestedField::optional(2, "s", container).into(),
                    ])
                    .build()
                    .expect("schema"),
            );
            let mut task = pos_scan_task(&data_path, schema, vec![], None);
            task.project_field_ids = Arc::from(vec![1, 2]);
            // Off Parquet, so the nested walk is exercised on another format too.
            task.data_file_format = DataFileFormat::Avro;

            let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
            let err = reader
                .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
                .expect("stream construction")
                .try_collect::<Vec<RecordBatch>>()
                .await
                .unwrap_err();
            assert_eq!(
                err.kind(),
                ErrorKind::FeatureUnsupported,
                "a variant nested in a {label} must be REFUSED, and with the typed refusal — not \
                 an opaque decode failure later on"
            );
            assert!(
                err.to_string().contains(expected_path),
                "the error must name the PATH to the variant ({expected_path}) so the caller can \
                 find it in a wide schema, got: {err}"
            );
        }
    }

    /// A table that HAS a variant column still scans when that column is not projected. The refusal
    /// is scoped to the projection, not the table.
    #[tokio::test]
    async fn a_variant_column_that_is_not_projected_does_not_block_the_scan() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp.path().join("v2.parquet").to_string_lossy().to_string();
        write_id_parquet_for_pos(&data_path, &[1, 2], 1000);

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "v", Type::Variant).into(),
                ])
                .build()
                .expect("schema"),
        );
        let mut task = pos_scan_task(&data_path, schema, vec![], None);
        task.project_field_ids = Arc::from(vec![1]);

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
        let batches: Vec<RecordBatch> = reader
            .read(Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream)
            .expect("stream")
            .try_collect()
            .await
            .expect("a non-projected variant column must not block the scan");
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    }

    /// `_row_id` reaches the same whole-file guard as `_pos`: Java's `RowIdReader` falls back to
    /// `firstRowId + pos`, so an id computed after row-skipping belongs to a different row.
    #[tokio::test]
    async fn row_id_ranged_split_task_is_rejected_fail_loud() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids: Vec<i32> = (0..20).collect();
        write_id_parquet_for_pos(&data_path, &ids, 1000);

        // All three message arms: `_row_id` alone, `_pos` alone, and both. Pinning one arm leaves
        // the combined arm free to name only `_pos`.
        for (projected, expected) in [
            (vec![1, RESERVED_FIELD_ID_ROW_ID], "`_row_id` projection"),
            (vec![1, RESERVED_FIELD_ID_POS], "`_pos` projection"),
            (
                vec![1, RESERVED_FIELD_ID_POS, RESERVED_FIELD_ID_ROW_ID],
                "`_pos` and `_row_id` projection",
            ),
        ] {
            let mut ranged = pos_scan_task(&data_path, id_schema_for_pos(), vec![], None);
            ranged.project_field_ids = Arc::from(projected);
            ranged.first_row_id = Some(1_000);
            ranged.start = 1;
            ranged.length = ranged.file_size_in_bytes;

            let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
            let err = reader
                .read(Box::pin(futures::stream::iter(vec![Ok(ranged)])) as FileScanTaskStream)
                .expect("stream construction")
                .try_collect::<Vec<RecordBatch>>()
                .await
                .expect_err("a ranged task must fail loud, not mint wrong ids");
            assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
            assert!(
                err.to_string().contains(expected),
                "the error must name exactly the columns that forced the whole-file path \
                 ({expected}), got: {err}"
            );
        }
    }

    /// A ranged split task must not take the `_pos` streaming path. That path decodes the whole
    /// file with ordinals from 0, so every split re-emits every row with a wrong `_pos`. Fail
    /// loud. The public `PartitionWork` seam and direct reader use reach it.
    #[tokio::test]
    async fn fk5_pos_ranged_split_task_is_rejected_fail_loud() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids: Vec<i32> = (0..20).collect();
        write_id_parquet_for_pos(&data_path, &ids, 1000);

        // Ranged shapes on both axes of the guard. Varying only the length leaves `start == 0`
        // unpinned: a start-blind guard accepts `(1, file_size)` and decodes the whole file.
        let file_size =
            pos_scan_task(&data_path, id_schema_for_pos(), vec![], None).file_size_in_bytes;
        assert!(file_size > 1, "fixture file must be non-trivial");
        for (start, length) in [(0u64, file_size / 2), (1u64, file_size), (1u64, 0u64)] {
            let mut ranged = pos_scan_task(&data_path, id_schema_for_pos(), vec![], None);
            ranged.start = start;
            ranged.length = length;

            let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
            let err = reader
                .read(Box::pin(futures::stream::iter(vec![Ok(ranged)])) as FileScanTaskStream)
                .expect("stream construction")
                .try_collect::<Vec<RecordBatch>>()
                .await
                .expect_err("a ranged task projecting `_pos` must fail loud, not duplicate rows");
            assert_eq!(
                err.kind(),
                ErrorKind::FeatureUnsupported,
                "ranged `_pos` task ({start}, {length}) must fail loud, not duplicate rows"
            );
            assert!(
                err.to_string().contains("ranged split task is unsupported"),
                "typed error must name the ranged-split rejection ({start}, {length}), got: {err}"
            );
        }

        // Control: the same file as a whole-file task still streams. The guard rejects only
        // ranged windows.
        let whole = pos_scan_task(&data_path, id_schema_for_pos(), vec![], None);
        let pairs = collect_id_pos_pairs(&run_pos_scan(whole, Some(7)).await);
        assert_eq!(pairs.len(), 20, "whole-file control still reads all rows");
        // Control 2: explicit full-length window (plan_files shape) is also whole-file.
        let mut full = pos_scan_task(&data_path, id_schema_for_pos(), vec![], None);
        full.length = full.file_size_in_bytes;
        let pairs = collect_id_pos_pairs(&run_pos_scan(full, Some(7)).await);
        assert_eq!(pairs.len(), 20, "explicit full-length window is whole-file");
    }

    /// FK5 oracle: dense pos-deletes (every other row) — survivors keep TRUE physical `_pos`.
    #[tokio::test]
    async fn fk5_pos_oracle_dense_pos_deletes() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids: Vec<i32> = (0..10).map(|i| i * 10).collect(); // 0,10,...,90 at pos 0..9
        write_id_parquet_for_pos(&data_path, &ids, 1000);
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let del = fk5_write_pos_delete_file(&del_path, &data_path, &[0, 2, 4, 6, 8]);
        let schema = id_schema_for_pos();
        let task = pos_scan_task(&data_path, schema, vec![del], None);
        let pairs = collect_id_pos_pairs(&run_pos_scan(task, Some(2)).await);
        assert_eq!(
            pairs,
            vec![(10, 1), (30, 3), (50, 5), (70, 7), (90, 9)],
            "dense pos-deletes must leave true physical _pos on survivors"
        );
    }

    /// FK5 oracle: sparse pos-deletes across a multi-row-group file.
    ///
    /// This test passed unchanged with one row group, so it was a false green for its own name.
    /// It now asserts the row-group count from the real footer, and adds a second leg that prunes
    /// the first row group, so the survivors' `_pos` must be offset by its row count. A reader
    /// that restarts ordinals per row group passes leg 1 and fails leg 2.
    #[tokio::test]
    async fn fk5_pos_oracle_sparse_pos_deletes_multi_rg() {
        use parquet::file::reader::{FileReader, SerializedFileReader};

        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids: Vec<i32> = (1..=200).collect();
        write_id_parquet_for_pos(&data_path, &ids, 100);
        // The fixture must be multi-row-group. Passing `None` for the row count turns this red.
        let footer = SerializedFileReader::new(File::open(&data_path).expect("open fixture"))
            .expect("read footer");
        assert_eq!(
            footer.metadata().num_row_groups(),
            2,
            "fixture must span 2 row groups for this oracle to mean anything"
        );
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let del = fk5_write_pos_delete_file(&del_path, &data_path, &[0, 50, 100, 199]);
        let schema = id_schema_for_pos();
        let task = pos_scan_task(&data_path, schema, vec![del], None);
        let pairs = collect_id_pos_pairs(&run_pos_scan(task, Some(17)).await);
        assert_eq!(pairs.len(), 196);
        assert!(pairs.contains(&(2, 1)), "pos 1 (id=2) survives");
        assert!(!pairs.contains(&(51, 50)), "pos 50 deleted");
        assert!(pairs.contains(&(50, 49)), "pos 49 survives with _pos=49");
        assert!(!pairs.contains(&(101, 100)), "pos 100 deleted");
        assert!(
            pairs.contains(&(102, 101)),
            "pos 101 survives with _pos=101"
        );
        assert!(!pairs.contains(&(200, 199)), "pos 199 deleted");
        assert!(
            pairs.contains(&(199, 198)),
            "pos 198 survives with _pos=198"
        );
        let deleted: HashSet<i64> = [0i64, 50, 100, 199].into_iter().collect();
        let expected: Vec<(i32, i64)> = (0i64..200)
            .filter(|p| !deleted.contains(p))
            .map(|p| ((p + 1) as i32, p))
            .collect();
        assert_eq!(pairs, expected);

        // Leg 2 is the behavioural discriminator. The `_pos` path decodes with no row skipping,
        // so the multi-row-group shape reaches the reader only through decode batch boundaries.
        // parquet-rs never spans a batch across a row group, so a batch size that does not divide
        // the row-group row count gives a short batch at every seam. That is where the two
        // counters can desync.
        let no_delete_task = pos_scan_task(&data_path, id_schema_for_pos(), vec![], None);
        let batches = run_pos_scan(no_delete_task, Some(17)).await;
        let batch_lengths: Vec<usize> = batches.iter().map(|b| b.num_rows()).collect();
        assert_eq!(
            batch_lengths,
            vec![17, 17, 17, 17, 17, 15, 17, 17, 17, 17, 17, 15],
            "decode batches must break at the row-group seam (a short batch at pos 100)"
        );
        // …and the `_pos` values must run 0..199 unbroken ACROSS that seam.
        let pairs_no_delete = collect_id_pos_pairs(&batches);
        assert_eq!(
            pairs_no_delete,
            (0i64..200)
                .map(|p| ((p + 1) as i32, p))
                .collect::<Vec<(i32, i64)>>(),
            "_pos must be the absolute file ordinal across the row-group seam"
        );
    }

    /// FK5 oracle: residual filter AND sparse pos-deletes — row set + `_pos` vs physical baseline.
    #[tokio::test]
    async fn fk5_pos_oracle_residual_and_pos_deletes() {
        use crate::expr::{Bind, Reference};

        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids: Vec<i32> = (0..20).collect(); // id == physical pos
        write_id_parquet_for_pos(&data_path, &ids, 1000);
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let del = fk5_write_pos_delete_file(&del_path, &data_path, &[3, 7, 15]);
        let schema = id_schema_for_pos();
        let residual = Reference::new("id")
            .greater_than_or_equal_to(Datum::int(5))
            .and(Reference::new("id").less_than(Datum::int(18)));
        let bound = residual.bind(schema.clone(), true).expect("bind");
        let task = pos_scan_task(&data_path, schema, vec![del], Some(bound));
        let pairs = collect_id_pos_pairs(&run_pos_scan(task, Some(4)).await);
        let expected: Vec<(i32, i64)> = (5i64..18)
            .filter(|p| *p != 7 && *p != 15)
            .map(|p| (p as i32, p))
            .collect();
        assert_eq!(
            pairs, expected,
            "residual∩¬pos-delete must keep true physical _pos (unpruned baseline)"
        );
    }

    /// `absolute_pos` must advance by the full pre-filter batch size. Advancing by the survivor
    /// count in `apply_pos_aware_batch` shifts the second batch's `_pos` and turns this red.
    #[tokio::test]
    async fn fk5_pos_mutation_absolute_pos_advances_by_full_batch() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids: Vec<i32> = vec![0, 1, 2, 3, 4, 5];
        write_id_parquet_for_pos(&data_path, &ids, 1000);
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let del = fk5_write_pos_delete_file(&del_path, &data_path, &[0, 1]);
        let schema = id_schema_for_pos();
        let task = pos_scan_task(&data_path, schema, vec![del], None);
        let pairs = collect_id_pos_pairs(&run_pos_scan(task, Some(3)).await);
        assert_eq!(
            pairs,
            vec![(2, 2), (3, 3), (4, 4), (5, 5)],
            "absolute_pos must advance by full pre-filter batch size (mutation bait)"
        );
    }

    /// Residual drops the entire first batch; later batches must still carry
    /// physical `_pos` (not renumbered from 0).
    #[tokio::test]
    async fn fk5_pos_residual_empties_first_batch_preserves_physical_pos() {
        use crate::expr::{Bind, Reference};

        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        // 6 rows; batch_size=3 → batch0 ids 0,1,2 and batch1 ids 3,4,5. Residual keeps id >= 3.
        let ids: Vec<i32> = vec![0, 1, 2, 3, 4, 5];
        write_id_parquet_for_pos(&data_path, &ids, 1000);
        let schema = id_schema_for_pos();
        let residual = Reference::new("id").greater_than_or_equal_to(Datum::int(3));
        let bound = residual.bind(schema.clone(), true).expect("bind");
        let task = pos_scan_task(&data_path, schema, vec![], Some(bound));
        let pairs = collect_id_pos_pairs(&run_pos_scan(task, Some(3)).await);
        assert_eq!(
            pairs,
            vec![(3, 3), (4, 4), (5, 5)],
            "after residual empties batch0, batch1 _pos must remain physical 3..5 not 0..2"
        );
    }

    /// Every row position-deleted → empty result (no panic / bogus _pos).
    #[tokio::test]
    async fn fk5_pos_all_rows_position_deleted() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids: Vec<i32> = vec![10, 20, 30];
        write_id_parquet_for_pos(&data_path, &ids, 1000);
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let del = fk5_write_pos_delete_file(&del_path, &data_path, &[0, 1, 2]);
        let schema = id_schema_for_pos();
        let task = pos_scan_task(&data_path, schema, vec![del], None);
        let pairs = collect_id_pos_pairs(&run_pos_scan(task, Some(1)).await);
        assert!(
            pairs.is_empty(),
            "all-deleted file must yield no (id,_pos) pairs, got {pairs:?}"
        );
    }

    /// Single-batch vs multi-batch streaming must yield identical (id,_pos) sets
    /// under dense pos-deletes + residual (unpruned baseline identity).
    #[tokio::test]
    async fn fk5_pos_single_vs_multi_batch_identity() {
        use crate::expr::{Bind, Reference};

        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids: Vec<i32> = (0..30).collect();
        write_id_parquet_for_pos(&data_path, &ids, 1000);
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let del = fk5_write_pos_delete_file(&del_path, &data_path, &[1, 5, 11, 22, 29]);
        let schema = id_schema_for_pos();
        let residual = Reference::new("id")
            .greater_than_or_equal_to(Datum::int(2))
            .and(Reference::new("id").less_than(Datum::int(28)));
        let bound = residual.bind(schema.clone(), true).expect("bind");

        let task_multi = pos_scan_task(
            &data_path,
            schema.clone(),
            vec![fk5_write_pos_delete_file(
                &tmp.path().join("d1.parquet").to_string_lossy(),
                &data_path,
                &[1, 5, 11, 22, 29],
            )],
            Some(bound.clone()),
        );
        let task_single = pos_scan_task(&data_path, schema, vec![del], Some(bound));
        let multi = collect_id_pos_pairs(&run_pos_scan(task_multi, Some(4)).await);
        let single = collect_id_pos_pairs(&run_pos_scan(task_single, Some(10_000)).await);
        assert_eq!(
            multi, single,
            "streamed multi-batch must match single-batch (id,_pos) under residual∩pos-deletes"
        );
        let deleted: HashSet<i64> = [1i64, 5, 11, 22, 29].into_iter().collect();
        let expected: Vec<(i32, i64)> = (2i64..28)
            .filter(|p| !deleted.contains(p))
            .map(|p| (p as i32, p))
            .collect();
        assert_eq!(single, expected);
    }

    /// `_file` + `_pos` together under multi-batch streaming (constants + ordinals).
    #[tokio::test]
    async fn fk5_pos_with_file_metadata_column() {
        use arrow_array::{Int32Array, Int64Array};

        use crate::metadata_columns::RESERVED_FIELD_ID_FILE;

        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids = vec![10, 20, 30, 40];
        write_id_parquet_for_pos(&data_path, &ids, 1000);
        let schema = id_schema_for_pos();
        let del = fk5_write_pos_delete_file(
            &tmp.path().join("d.parquet").to_string_lossy(),
            &data_path,
            &[1],
        );
        let task = FileScanTask {
            file_size_in_bytes: std::fs::metadata(&data_path).unwrap().len(),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(data_path.clone()),
            data_file_format: DataFileFormat::Parquet,
            schema,
            project_field_ids: Arc::from(vec![1, RESERVED_FIELD_ID_FILE, RESERVED_FIELD_ID_POS]),
            predicate: None,
            deletes: Arc::from(vec![del]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };
        let batches = run_pos_scan(task, Some(2)).await;
        let mut pairs = Vec::new();
        for batch in &batches {
            assert_eq!(batch.num_columns(), 3);
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let pos = batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                // _file is constant path for the task
                pairs.push((id.value(i), pos.value(i)));
            }
        }
        pairs.sort_by_key(|(id, _)| *id);
        assert_eq!(
            pairs,
            vec![(10, 0), (30, 2), (40, 3)],
            "_file+_pos stream must drop pos 1 and keep physical ordinals"
        );
    }

    /// Equality-delete + `_pos` streaming path (survival_mask eq branch).
    #[tokio::test]
    async fn fk5_pos_with_equality_deletes() {
        // Two-column schema so eq-delete on `id` is well-formed; project id + _pos.
        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("data", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(arrow_array::Int32Array::from(vec![10, 20, 30, 40, 50])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])) as ArrayRef,
        ])
        .expect("batch");
        let file = File::create(&data_path).unwrap();
        let mut writer = ArrowWriter::try_new(
            file,
            arrow_schema,
            Some(WriterProperties::builder().build()),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Eq-delete file: delete id=20 and id=40 (field id 1)
        let eq_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));
        let eq_batch =
            RecordBatch::try_new(eq_schema.clone(), vec![
                Arc::new(arrow_array::Int32Array::from(vec![20, 40])) as ArrayRef,
            ])
            .unwrap();
        let eq_file = File::create(&eq_path).unwrap();
        let mut eq_writer = ArrowWriter::try_new(
            eq_file,
            eq_schema,
            Some(WriterProperties::builder().build()),
        )
        .unwrap();
        eq_writer.write(&eq_batch).unwrap();
        eq_writer.close().unwrap();

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );
        let eq_del = FileScanTaskDeleteFile {
            file_path: eq_path.clone(),
            file_size_in_bytes: std::fs::metadata(&eq_path).unwrap().len(),
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: Some(vec![1]),
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };
        let task = FileScanTask {
            file_size_in_bytes: std::fs::metadata(&data_path).unwrap().len(),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(data_path),
            data_file_format: DataFileFormat::Parquet,
            schema,
            project_field_ids: Arc::from(vec![1, RESERVED_FIELD_ID_POS]),
            predicate: None,
            deletes: Arc::from(vec![eq_del]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };
        let pairs = collect_id_pos_pairs(&run_pos_scan(task, Some(2)).await);
        // Survivors keep physical positions: 10@0, 30@2, 50@4
        assert_eq!(
            pairs,
            vec![(10, 0), (30, 2), (50, 4)],
            "eq-deletes under _pos stream must drop keys and keep physical ordinals"
        );
    }

    /// Residual ∩ pos-delete ∩ eq-delete under streaming `_pos`.
    #[tokio::test]
    async fn fk5_pos_residual_and_pos_and_eq_deletes() {
        use crate::expr::{Bind, Reference};

        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));
        let batch =
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(arrow_array::Int32Array::from((0..10).collect::<Vec<i32>>())) as ArrayRef,
            ])
            .unwrap();
        let file = File::create(&data_path).unwrap();
        let mut writer = ArrowWriter::try_new(
            file,
            arrow_schema,
            Some(WriterProperties::builder().build()),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // pos-delete 1, 8; eq-delete id=4; residual id >= 2 && id < 9
        let pos_del = fk5_write_pos_delete_file(
            &tmp.path().join("pos.parquet").to_string_lossy(),
            &data_path,
            &[1, 8],
        );
        let eq_path = tmp.path().join("eq.parquet").to_string_lossy().to_string();
        let eq_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));
        let eq_batch =
            RecordBatch::try_new(eq_schema.clone(), vec![
                Arc::new(arrow_array::Int32Array::from(vec![4])) as ArrayRef,
            ])
            .unwrap();
        let eq_file = File::create(&eq_path).unwrap();
        let mut eq_writer = ArrowWriter::try_new(
            eq_file,
            eq_schema,
            Some(WriterProperties::builder().build()),
        )
        .unwrap();
        eq_writer.write(&eq_batch).unwrap();
        eq_writer.close().unwrap();
        let eq_del = FileScanTaskDeleteFile {
            file_path: eq_path.clone(),
            file_size_in_bytes: std::fs::metadata(&eq_path).unwrap().len(),
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: Some(vec![1]),
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let schema = id_schema_for_pos();
        let residual = Reference::new("id")
            .greater_than_or_equal_to(Datum::int(2))
            .and(Reference::new("id").less_than(Datum::int(9)));
        let bound = residual.bind(schema.clone(), true).unwrap();
        let task = FileScanTask {
            file_size_in_bytes: std::fs::metadata(&data_path).unwrap().len(),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(data_path),
            data_file_format: DataFileFormat::Parquet,
            schema,
            project_field_ids: Arc::from(vec![1, RESERVED_FIELD_ID_POS]),
            predicate: Some(Arc::new(bound)),
            deletes: Arc::from(vec![pos_del, eq_del]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };
        let pairs = collect_id_pos_pairs(&run_pos_scan(task, Some(3)).await);
        // residual [2,9) minus pos{1,8} minus eq{4} → 2,3,5,6,7 (pos 1,8 outside residual for 1)
        // pos 8 is in residual range and deleted; pos 1 is outside residual.
        assert_eq!(
            pairs,
            vec![(2, 2), (3, 3), (5, 5), (6, 6), (7, 7)],
            "residual∩¬pos∩¬eq must keep true physical _pos"
        );
    }

    /// FK5 MERGE-shaped pin: streamed `(_file,_pos)` scan → write pos deletes → MoR omits rows.
    #[tokio::test]
    async fn fk5_merge_shaped_pos_delete_from_streamed_identity_scan() {
        use arrow_array::{Int32Array, Int64Array};

        let tmp = TempDir::new().unwrap();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        let ids = vec![10, 20, 30, 40, 50];
        write_id_parquet_for_pos(&data_path, &ids, 1000);
        let schema = id_schema_for_pos();

        let discover_task = pos_scan_task(&data_path, schema.clone(), vec![], None);
        let discover_batches = run_pos_scan(discover_task, Some(2)).await;
        let all_pairs = collect_id_pos_pairs(&discover_batches);
        assert_eq!(
            all_pairs,
            vec![(10, 0), (20, 1), (30, 2), (40, 3), (50, 4)],
            "identity scan must report true physical positions"
        );
        let mutate_pos: Vec<i64> = all_pairs
            .iter()
            .filter(|(id, _)| *id == 20 || *id == 40)
            .map(|(_, p)| *p)
            .collect();
        assert_eq!(mutate_pos, vec![1, 3]);

        let del_path = tmp
            .path()
            .join("merge-pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let del = fk5_write_pos_delete_file(&del_path, &data_path, &mutate_pos);

        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io)
            .with_row_selection_enabled(true)
            .build();
        let mor_task = FileScanTask {
            file_size_in_bytes: std::fs::metadata(&data_path).unwrap().len(),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(data_path.clone()),
            data_file_format: DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: Arc::from(vec![1]),
            predicate: None,
            deletes: Arc::from(vec![del]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };
        let batches = reader
            .read(Box::pin(futures::stream::iter(vec![Ok(mor_task)])) as FileScanTaskStream)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();
        let mut live: Vec<i32> = Vec::new();
        for b in &batches {
            let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            live.extend(col.values().iter().copied());
        }
        live.sort();
        assert_eq!(
            live,
            vec![10, 30, 50],
            "position deletes written from streamed _pos scan must remove exactly ids 20 and 40"
        );

        let with_pos = pos_scan_task(
            &data_path,
            schema,
            vec![fk5_write_pos_delete_file(
                &tmp.path().join("del2.parquet").to_string_lossy(),
                &data_path,
                &mutate_pos,
            )],
            None,
        );
        let pairs = collect_id_pos_pairs(&run_pos_scan(with_pos, Some(2)).await);
        assert_eq!(pairs, vec![(10, 0), (30, 2), (50, 4)]);
        let _ = std::mem::size_of::<Int64Array>();
    }

    /// A position delete in a later row group must still apply. `build_deletes_row_selection`
    /// failed to increment `current_row_group_base_idx` while it skipped row groups.
    ///
    /// The fixture deletes row 199 of a 200-row, two-row-group file. The read must return 199
    /// rows. The defect returned 200.
    #[tokio::test]
    async fn test_position_delete_across_multiple_row_groups() {
        use arrow_array::{Int32Array, Int64Array};
        use parquet::file::reader::{FileReader, SerializedFileReader};

        const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: u64 = 2147483546;
        const FIELD_ID_POSITIONAL_DELETE_POS: u64 = 2147483545;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();

        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let data_file_path = format!("{table_location}/data.parquet");

        let batch1 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(1..=100),
        )])
        .unwrap();

        let batch2 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(101..=200),
        )])
        .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();

        let file = File::create(&data_file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        writer.write(&batch1).expect("Writing batch 1");
        writer.write(&batch2).expect("Writing batch 2");
        writer.close().unwrap();

        let verify_file = File::open(&data_file_path).unwrap();
        let verify_reader = SerializedFileReader::new(verify_file).unwrap();
        assert_eq!(
            verify_reader.metadata().num_row_groups(),
            2,
            "Should have 2 row groups"
        );

        let delete_file_path = format!("{table_location}/deletes.parquet");

        let delete_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                FIELD_ID_POSITIONAL_DELETE_FILE_PATH.to_string(),
            )])),
            Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                FIELD_ID_POSITIONAL_DELETE_POS.to_string(),
            )])),
        ]));

        let delete_batch = RecordBatch::try_new(delete_schema.clone(), vec![
            Arc::new(StringArray::from_iter_values(vec![data_file_path.clone()])),
            Arc::new(Int64Array::from_iter_values(vec![199i64])),
        ])
        .unwrap();

        let delete_props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let delete_file = File::create(&delete_file_path).unwrap();
        let mut delete_writer =
            ArrowWriter::try_new(delete_file, delete_schema, Some(delete_props)).unwrap();
        delete_writer.write(&delete_batch).unwrap();
        delete_writer.close().unwrap();

        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io).build();

        let task = FileScanTask {
            file_size_in_bytes: std::fs::metadata(&data_file_path).unwrap().len(),
            start: 0,
            length: 0,
            record_count: Some(200),
            data_file_path: Arc::from(data_file_path.clone()),
            data_file_format: DataFileFormat::Parquet,
            schema: table_schema.clone(),
            project_field_ids: Arc::from(vec![1]),
            predicate: None,
            deletes: Arc::from(vec![FileScanTaskDeleteFile {
                file_size_in_bytes: std::fs::metadata(&delete_file_path).unwrap().len(),
                file_path: delete_file_path,
                file_type: DataContentType::PositionDeletes,
                partition_spec_id: 0,
                equality_ids: None,
                file_format: DataFileFormat::Parquet,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
                record_count: None,
            }]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };

        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

        println!("Total rows read: {total_rows}");
        println!("Expected: 199 rows (deleted row 199 which had id=200)");

        assert_eq!(
            total_rows, 199,
            "Expected 199 rows after deleting row 199, but got {total_rows} rows. \
             The bug causes position deletes in later row groups to be ignored."
        );

        let all_ids: Vec<i32> = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();

        assert!(
            !all_ids.contains(&200),
            "Row with id=200 should be deleted but was found in results"
        );

        let expected_ids: Vec<i32> = (1..=199).collect();
        assert_eq!(
            all_ids, expected_ids,
            "Should have ids 1-199 but got different values"
        );
    }

    /// A position delete must survive the skip over an unselected row group. This is the
    /// row-group-selection variant of `test_position_delete_across_multiple_row_groups`.
    ///
    /// The fixture deletes row 199 and selects row group 1 only. The read must return 99 rows.
    /// The defect returned 100:
    ///
    /// ```rust
    /// delete_vector_iter.advance_to(next_row_group_base_idx); // Position at first delete >= 100
    /// next_deleted_row_idx_opt = delete_vector_iter.next(); // BUG: Consumes delete at 199!
    /// ```
    ///
    /// `advance_to()` already positions the iterator, so the following `next()` consumes the
    /// delete at 199.
    #[tokio::test]
    async fn test_position_delete_with_row_group_selection() {
        use arrow_array::{Int32Array, Int64Array};
        use parquet::file::reader::{FileReader, SerializedFileReader};

        const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: u64 = 2147483546;
        const FIELD_ID_POSITIONAL_DELETE_POS: u64 = 2147483545;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();

        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let data_file_path = format!("{table_location}/data.parquet");

        let batch1 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(1..=100),
        )])
        .unwrap();

        let batch2 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(101..=200),
        )])
        .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();

        let file = File::create(&data_file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        writer.write(&batch1).expect("Writing batch 1");
        writer.write(&batch2).expect("Writing batch 2");
        writer.close().unwrap();

        let verify_file = File::open(&data_file_path).unwrap();
        let verify_reader = SerializedFileReader::new(verify_file).unwrap();
        assert_eq!(
            verify_reader.metadata().num_row_groups(),
            2,
            "Should have 2 row groups"
        );

        let delete_file_path = format!("{table_location}/deletes.parquet");

        let delete_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                FIELD_ID_POSITIONAL_DELETE_FILE_PATH.to_string(),
            )])),
            Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                FIELD_ID_POSITIONAL_DELETE_POS.to_string(),
            )])),
        ]));

        let delete_batch = RecordBatch::try_new(delete_schema.clone(), vec![
            Arc::new(StringArray::from_iter_values(vec![data_file_path.clone()])),
            Arc::new(Int64Array::from_iter_values(vec![199i64])),
        ])
        .unwrap();

        let delete_props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let delete_file = File::create(&delete_file_path).unwrap();
        let mut delete_writer =
            ArrowWriter::try_new(delete_file, delete_schema, Some(delete_props)).unwrap();
        delete_writer.write(&delete_batch).unwrap();
        delete_writer.close().unwrap();

        let metadata_file = File::open(&data_file_path).unwrap();
        let metadata_reader = SerializedFileReader::new(metadata_file).unwrap();
        let metadata = metadata_reader.metadata();

        let row_group_0 = metadata.row_group(0);
        let row_group_1 = metadata.row_group(1);

        // The window comes from the real footer offsets, never from the `4 + Σ compressed_size`
        // model, which made this test blind to offset drift. The assertion below records that this
        // fixture is contiguous, so both agree here.
        let real_starts = footer_row_group_starts(metadata);
        let rg0_start = real_starts[0];
        let rg1_start = real_starts[1];
        let rg1_length = u64::try_from(row_group_1.compressed_size()).expect("non-negative size");
        assert_eq!(
            rg1_start,
            rg0_start + u64::try_from(row_group_0.compressed_size()).expect("non-negative size"),
            "this fixture is expected to have contiguous row groups"
        );

        println!(
            "Row group 0: starts at byte {}, {} bytes compressed",
            rg0_start,
            row_group_0.compressed_size()
        );
        println!(
            "Row group 1: starts at byte {}, {} bytes compressed",
            rg1_start,
            row_group_1.compressed_size()
        );

        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io).build();

        let task = FileScanTask {
            file_size_in_bytes: std::fs::metadata(&data_file_path).unwrap().len(),
            start: rg1_start,
            length: rg1_length,
            record_count: Some(100), // Row group 1 has 100 rows
            data_file_path: Arc::from(data_file_path.clone()),
            data_file_format: DataFileFormat::Parquet,
            schema: table_schema.clone(),
            project_field_ids: Arc::from(vec![1]),
            predicate: None,
            deletes: Arc::from(vec![FileScanTaskDeleteFile {
                file_size_in_bytes: std::fs::metadata(&delete_file_path).unwrap().len(),
                file_path: delete_file_path,
                file_type: DataContentType::PositionDeletes,
                partition_spec_id: 0,
                equality_ids: None,
                file_format: DataFileFormat::Parquet,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
                record_count: None,
            }]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };

        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

        println!("Total rows read from row group 1: {total_rows}");
        println!("Expected: 99 rows (row group 1 has 100 rows, 1 delete at position 199)");

        assert_eq!(
            total_rows, 99,
            "Expected 99 rows from row group 1 after deleting position 199, but got {total_rows} rows. \
             The bug causes position deletes to be lost when advance_to() is followed by next() \
             when skipping unselected row groups."
        );

        let all_ids: Vec<i32> = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();

        assert!(
            !all_ids.contains(&200),
            "Row with id=200 should be deleted but was found in results"
        );

        let expected_ids: Vec<i32> = (101..=199).collect();
        assert_eq!(
            all_ids, expected_ids,
            "Should have ids 101-199 but got different values"
        );
    }
    /// A stale cached delete must not hang the row-group skip. This is the inverse of
    /// `test_position_delete_with_row_group_selection`: the delete sits in the SKIPPED row group.
    ///
    /// ```rust
    /// let mut next_deleted_row_idx_opt = delete_vector_iter.next(); // Some(0)
    /// // ... skip to row group 1 ...
    /// delete_vector_iter.advance_to(100); // Iterator advances past delete at 0
    /// // BUG: next_deleted_row_idx_opt is still Some(0) - STALE!
    /// // When processing row group 1:
    /// //   current_idx = 100, next_deleted_row_idx = 0, next_row_group_base_idx = 200
    /// //   Loop condition: 0 < 200 (true)
    /// //   But: current_idx (100) > next_deleted_row_idx (0)
    /// //   And: current_idx (100) != next_deleted_row_idx (0)
    /// //   Neither branch executes -> INFINITE LOOP!
    /// ```
    ///
    /// `advance_to()` moves the iterator past the delete, but the cached index stays at 0, so
    /// neither branch can run and `build_deletes_row_selection` spins. The read must return all
    /// 100 rows.
    #[tokio::test]
    async fn test_position_delete_in_skipped_row_group() {
        use arrow_array::{Int32Array, Int64Array};
        use parquet::file::reader::{FileReader, SerializedFileReader};

        const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: u64 = 2147483546;
        const FIELD_ID_POSITIONAL_DELETE_POS: u64 = 2147483545;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();

        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let data_file_path = format!("{table_location}/data.parquet");

        let batch1 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(1..=100),
        )])
        .unwrap();

        let batch2 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(101..=200),
        )])
        .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();

        let file = File::create(&data_file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        writer.write(&batch1).expect("Writing batch 1");
        writer.write(&batch2).expect("Writing batch 2");
        writer.close().unwrap();

        let verify_file = File::open(&data_file_path).unwrap();
        let verify_reader = SerializedFileReader::new(verify_file).unwrap();
        assert_eq!(
            verify_reader.metadata().num_row_groups(),
            2,
            "Should have 2 row groups"
        );

        let delete_file_path = format!("{table_location}/deletes.parquet");

        let delete_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                FIELD_ID_POSITIONAL_DELETE_FILE_PATH.to_string(),
            )])),
            Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                FIELD_ID_POSITIONAL_DELETE_POS.to_string(),
            )])),
        ]));

        let delete_batch = RecordBatch::try_new(delete_schema.clone(), vec![
            Arc::new(StringArray::from_iter_values(vec![data_file_path.clone()])),
            Arc::new(Int64Array::from_iter_values(vec![0i64])),
        ])
        .unwrap();

        let delete_props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let delete_file = File::create(&delete_file_path).unwrap();
        let mut delete_writer =
            ArrowWriter::try_new(delete_file, delete_schema, Some(delete_props)).unwrap();
        delete_writer.write(&delete_batch).unwrap();
        delete_writer.close().unwrap();

        let metadata_file = File::open(&data_file_path).unwrap();
        let metadata_reader = SerializedFileReader::new(metadata_file).unwrap();
        let metadata = metadata_reader.metadata();

        let row_group_0 = metadata.row_group(0);
        let row_group_1 = metadata.row_group(1);

        // The window comes from the real footer offsets, never from the `4 + Σ compressed_size`
        // model, which made this test blind to offset drift. The assertion below records that this
        // fixture is contiguous, so both agree here.
        let real_starts = footer_row_group_starts(metadata);
        let rg0_start = real_starts[0];
        let rg1_start = real_starts[1];
        let rg1_length = u64::try_from(row_group_1.compressed_size()).expect("non-negative size");
        assert_eq!(
            rg1_start,
            rg0_start + u64::try_from(row_group_0.compressed_size()).expect("non-negative size"),
            "this fixture is expected to have contiguous row groups"
        );

        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io).build();

        let task = FileScanTask {
            file_size_in_bytes: std::fs::metadata(&data_file_path).unwrap().len(),
            start: rg1_start,
            length: rg1_length,
            record_count: Some(100), // Row group 1 has 100 rows
            data_file_path: Arc::from(data_file_path.clone()),
            data_file_format: DataFileFormat::Parquet,
            schema: table_schema.clone(),
            project_field_ids: Arc::from(vec![1]),
            predicate: None,
            deletes: Arc::from(vec![FileScanTaskDeleteFile {
                file_size_in_bytes: std::fs::metadata(&delete_file_path).unwrap().len(),
                file_path: delete_file_path,
                file_type: DataContentType::PositionDeletes,
                partition_spec_id: 0,
                equality_ids: None,
                file_format: DataFileFormat::Parquet,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
                record_count: None,
            }]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };

        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        // The delete at position 0 is in row group 0, which is skipped, so it doesn't affect us
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

        assert_eq!(
            total_rows, 100,
            "Expected 100 rows from row group 1 (delete at position 0 is in skipped row group 0). \
             If this hangs or fails, it indicates the cached delete index was not updated after advance_to()."
        );

        let all_ids: Vec<i32> = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();

        let expected_ids: Vec<i32> = (101..=200).collect();
        assert_eq!(
            all_ids, expected_ids,
            "Should have ids 101-200 (all of row group 1)"
        );
    }

    /// Reads a file with no field id metadata, through the position-based fallback path.
    /// Java `ParquetSchemaUtil.addFallbackIds()` and `pruneColumnsFallback()`.
    #[tokio::test]
    async fn test_read_parquet_file_without_field_ids() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "age", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let name_data = vec!["Alice", "Bob", "Charlie"];
        let age_data = vec![30, 25, 35];

        use arrow_array::Int32Array;
        let name_col = Arc::new(StringArray::from(name_data.clone())) as ArrayRef;
        let age_col = Arc::new(Int32Array::from(age_data.clone())) as ArrayRef;

        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![name_col, age_col]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();

        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/1.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/1.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 2]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);

        let name_array = batch.column(0).as_string::<i32>();
        assert_eq!(name_array.value(0), "Alice");
        assert_eq!(name_array.value(1), "Bob");
        assert_eq!(name_array.value(2), "Charlie");

        let age_array = batch
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(age_array.value(0), 30);
        assert_eq!(age_array.value(1), 25);
        assert_eq!(age_array.value(2), 35);
    }

    /// Test reading Parquet files without field IDs with partial projection.
    /// Only a subset of columns are requested, verifying position-based fallback
    /// handles column selection correctly.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_partial_projection() {
        use arrow_array::Int32Array;

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "col1", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "col2", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(3, "col3", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(4, "col4", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("col1", DataType::Utf8, false),
            Field::new("col2", DataType::Int32, false),
            Field::new("col3", DataType::Utf8, false),
            Field::new("col4", DataType::Int32, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let col1_data = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let col2_data = Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef;
        let col3_data = Arc::new(StringArray::from(vec!["c", "d"])) as ArrayRef;
        let col4_data = Arc::new(Int32Array::from(vec![30, 40])) as ArrayRef;

        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![
            col1_data, col2_data, col3_data, col4_data,
        ])
        .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();

        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/1.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/1.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 3]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);

        let col1_array = batch.column(0).as_string::<i32>();
        assert_eq!(col1_array.value(0), "a");
        assert_eq!(col1_array.value(1), "b");

        let col3_array = batch.column(1).as_string::<i32>();
        assert_eq!(col3_array.value(0), "c");
        assert_eq!(col3_array.value(1), "d");
    }

    /// Test reading Parquet files without field IDs with schema evolution.
    /// The Iceberg schema has more fields than the Parquet file, testing that
    /// missing columns are filled with NULLs.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_schema_evolution() {
        use arrow_array::{Array, Int32Array};

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "age", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(3, "city", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let name_data = Arc::new(StringArray::from(vec!["Alice", "Bob"])) as ArrayRef;
        let age_data = Arc::new(Int32Array::from(vec![30, 25])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![name_data, age_data]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();

        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/1.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/1.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 2, 3]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);

        let name_array = batch.column(0).as_string::<i32>();
        assert_eq!(name_array.value(0), "Alice");
        assert_eq!(name_array.value(1), "Bob");

        let age_array = batch
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(age_array.value(0), 30);
        assert_eq!(age_array.value(1), 25);

        let city_array = batch.column(2).as_string::<i32>();
        assert_eq!(city_array.null_count(), 2);
        assert!(city_array.is_null(0));
        assert!(city_array.is_null(1));
    }

    /// Test reading Parquet files without field IDs that have multiple row groups.
    /// This ensures the position-based fallback works correctly across row group boundaries.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_multiple_row_groups() {
        use arrow_array::Int32Array;

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "value", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_write_batch_size(2)
            .set_max_row_group_row_count(Some(2))
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();

        for batch_num in 0..3 {
            let name_data = Arc::new(StringArray::from(vec![
                format!("name_{}", batch_num * 2),
                format!("name_{}", batch_num * 2 + 1),
            ])) as ArrayRef;
            let value_data =
                Arc::new(Int32Array::from(vec![batch_num * 2, batch_num * 2 + 1])) as ArrayRef;

            let batch =
                RecordBatch::try_new(arrow_schema.clone(), vec![name_data, value_data]).unwrap();
            writer.write(&batch).expect("Writing batch");
        }
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/1.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/1.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 2]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert!(!result.is_empty());

        let mut all_names = Vec::new();
        let mut all_values = Vec::new();

        for batch in &result {
            let name_array = batch.column(0).as_string::<i32>();
            let value_array = batch
                .column(1)
                .as_primitive::<arrow_array::types::Int32Type>();

            for i in 0..batch.num_rows() {
                all_names.push(name_array.value(i).to_string());
                all_values.push(value_array.value(i));
            }
        }

        assert_eq!(all_names.len(), 6);
        assert_eq!(all_values.len(), 6);

        for i in 0..6 {
            assert_eq!(all_names[i], format!("name_{i}"));
            assert_eq!(all_values[i], i as i32);
        }
    }

    /// Test reading Parquet files without field IDs with nested types (struct).
    /// Java's pruneColumnsFallback() projects entire top-level columns including nested content.
    /// This test verifies that a top-level struct field is projected correctly with all its nested fields.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_with_struct() {
        use arrow_array::{Int32Array, StructArray};
        use arrow_schema::Fields;

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(
                        2,
                        "person",
                        Type::Struct(crate::spec::StructType::new(vec![
                            NestedField::required(
                                3,
                                "name",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            NestedField::required(4, "age", Type::Primitive(PrimitiveType::Int))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "person",
                DataType::Struct(Fields::from(vec![
                    Field::new("name", DataType::Utf8, false),
                    Field::new("age", DataType::Int32, false),
                ])),
                false,
            ),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let id_data = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let name_data = Arc::new(StringArray::from(vec!["Alice", "Bob"])) as ArrayRef;
        let age_data = Arc::new(Int32Array::from(vec![30, 25])) as ArrayRef;
        let person_data = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("name", DataType::Utf8, false)),
                name_data,
            ),
            (
                Arc::new(Field::new("age", DataType::Int32, false)),
                age_data,
            ),
        ])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![id_data, person_data]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();

        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/1.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/1.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 2]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);

        let id_array = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(id_array.value(0), 1);
        assert_eq!(id_array.value(1), 2);

        let person_array = batch.column(1).as_struct();
        assert_eq!(person_array.num_columns(), 2);

        let name_array = person_array.column(0).as_string::<i32>();
        assert_eq!(name_array.value(0), "Alice");
        assert_eq!(name_array.value(1), "Bob");

        let age_array = person_array
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(age_array.value(0), 30);
        assert_eq!(age_array.value(1), 25);
    }

    /// Test reading Parquet files without field IDs with schema evolution - column added in the middle.
    /// When a new column is inserted between existing columns in the schema order,
    /// the fallback projection must correctly map field IDs to output positions.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_schema_evolution_add_column_in_middle() {
        use arrow_array::{Array, Int32Array};

        let arrow_schema_old = Arc::new(ArrowSchema::new(vec![
            Field::new("col0", DataType::Int32, true),
            Field::new("col1", DataType::Int32, true),
        ]));

        // New column added between existing columns: col0 (id=1), newCol (id=5), col1 (id=2)
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "col0", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(5, "newCol", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "col1", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let col0_data = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let col1_data = Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema_old.clone(), vec![col0_data, col1_data]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/1.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/1.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 5, 2]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);

        let result_col0 = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(result_col0.value(0), 1);
        assert_eq!(result_col0.value(1), 2);

        // New column should be NULL (doesn't exist in old file)
        let result_newcol = batch
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(result_newcol.null_count(), 2);
        assert!(result_newcol.is_null(0));
        assert!(result_newcol.is_null(1));

        let result_col1 = batch
            .column(2)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(result_col1.value(0), 10);
        assert_eq!(result_col1.value(1), 20);
    }

    /// Reads a file with no field ids under a filter that removes every row group. That case
    /// panicked while row selection was on.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_filter_eliminates_all_rows() {
        use arrow_array::{Float64Array, Int32Array};

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(3, "value", Type::Primitive(PrimitiveType::Double))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let id_data = Arc::new(Int32Array::from(vec![10, 11, 12])) as ArrayRef;
        let name_data = Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef;
        let value_data = Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![id_data, name_data, value_data])
                .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        // Filter that eliminates all row groups: id < 5
        let predicate = Reference::new("id").less_than(Datum::int(5));

        // Enable both row_group_filtering and row_selection - triggered the panic
        let reader = ArrowReaderBuilder::new(file_io)
            .with_row_group_filtering_enabled(true)
            .with_row_selection_enabled(true)
            .build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/1.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/1.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 2, 3]),
                predicate: Some(Arc::new(predicate.bind(schema, true).unwrap())),
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert!(result.is_empty() || result.iter().all(|batch| batch.num_rows() == 0));
    }

    /// Test that concurrency=1 reads all files correctly and in deterministic order.
    /// This verifies the fast-path optimization for single concurrency.
    #[tokio::test]
    async fn test_read_with_concurrency_one() {
        use arrow_array::Int32Array;

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "file_num", Type::Primitive(PrimitiveType::Int))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("file_num", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        for file_num in 0..3 {
            let id_data = Arc::new(Int32Array::from_iter_values(
                file_num * 10..(file_num + 1) * 10,
            )) as ArrayRef;
            let file_num_data = Arc::new(Int32Array::from(vec![file_num; 10])) as ArrayRef;

            let to_write =
                RecordBatch::try_new(arrow_schema.clone(), vec![id_data, file_num_data]).unwrap();

            let file = File::create(format!("{table_location}/file_{file_num}.parquet")).unwrap();
            let mut writer =
                ArrowWriter::try_new(file, to_write.schema(), Some(props.clone())).unwrap();
            writer.write(&to_write).expect("Writing batch");
            writer.close().unwrap();
        }

        let reader = ArrowReaderBuilder::new(file_io)
            .with_data_file_concurrency_limit(1)
            .build();

        let tasks = vec![
            Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/file_0.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/file_0.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 2]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            }),
            Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/file_1.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/file_1.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 2]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            }),
            Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/file_2.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/file_2.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 2]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            }),
        ];

        let tasks_stream = Box::pin(futures::stream::iter(tasks)) as FileScanTaskStream;

        let result = reader
            .read(tasks_stream)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 30, "Should have 30 total rows");

        let mut all_ids = Vec::new();
        let mut all_file_nums = Vec::new();

        for batch in &result {
            let id_col = batch
                .column(0)
                .as_primitive::<arrow_array::types::Int32Type>();
            let file_num_col = batch
                .column(1)
                .as_primitive::<arrow_array::types::Int32Type>();

            for i in 0..batch.num_rows() {
                all_ids.push(id_col.value(i));
                all_file_nums.push(file_num_col.value(i));
            }
        }

        assert_eq!(all_ids.len(), 30);
        assert_eq!(all_file_nums.len(), 30);

        // With concurrency=1 and sequential processing, files should be processed in order
        // file_0: ids 0-9, file_num=0
        // file_1: ids 10-19, file_num=1
        // file_2: ids 20-29, file_num=2
        for i in 0..10 {
            assert_eq!(all_file_nums[i], 0, "First 10 rows should be from file_0");
            assert_eq!(all_ids[i], i as i32, "IDs should be 0-9");
        }
        for i in 10..20 {
            assert_eq!(all_file_nums[i], 1, "Next 10 rows should be from file_1");
            assert_eq!(all_ids[i], i as i32, "IDs should be 10-19");
        }
        for i in 20..30 {
            assert_eq!(all_file_nums[i], 2, "Last 10 rows should be from file_2");
            assert_eq!(all_ids[i], i as i32, "IDs should be 20-29");
        }
    }

    /// Test bucket partitioning reads source column from data file (not partition metadata). The
    /// spec takes a value from partition metadata only for an identity transform. A `bucket(4, id)`
    /// partition stores the bucket number, not the source value, so the reader must read `id` from
    /// the data file.
    #[tokio::test]
    async fn test_bucket_partitioning_reads_source_column_from_file() {
        use arrow_array::Int32Array;

        use crate::spec::{Literal, PartitionSpec, Struct, Transform};

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(0)
                .add_partition_field("id", "id_bucket", Transform::Bucket(4))
                .unwrap()
                .build()
                .unwrap(),
        );

        let partition_data = Struct::from_iter(vec![Some(Literal::int(1))]);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("name", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let id_data = Arc::new(Int32Array::from(vec![1, 5, 9, 13])) as ArrayRef;
        let name_data =
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "Dave"])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![id_data, name_data]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(format!("{}/data.parquet", &table_location)).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io).build();
        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(format!("{table_location}/data.parquet"))
                    .unwrap()
                    .len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(format!("{table_location}/data.parquet")),
                data_file_format: DataFileFormat::Parquet,
                schema: schema.clone(),
                project_field_ids: Arc::from(vec![1, 2]),
                predicate: None,
                deletes: Arc::from(vec![]),
                partition: Some(partition_data),
                partition_spec: Some(partition_spec),
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 4);

        // `id` must hold the file's values, not the constant partition value 1.
        let id_col = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(id_col.value(0), 1);
        assert_eq!(id_col.value(1), 5);
        assert_eq!(id_col.value(2), 9);
        assert_eq!(id_col.value(3), 13);

        let name_col = batch.column(1).as_string::<i32>();
        assert_eq!(name_col.value(0), "Alice");
        assert_eq!(name_col.value(1), "Bob");
        assert_eq!(name_col.value(2), "Charlie");
        assert_eq!(name_col.value(3), "Dave");
    }

    #[test]
    fn test_merge_ranges_empty() {
        assert_eq!(super::merge_ranges(&[], 1024), Vec::<Range<u64>>::new());
    }

    #[test]
    fn test_merge_ranges_no_coalesce() {
        // Ranges far apart should not be merged
        let ranges = vec![0..100, 1_000_000..1_000_100];
        let merged = super::merge_ranges(&ranges, 1024);
        assert_eq!(merged, vec![0..100, 1_000_000..1_000_100]);
    }

    #[test]
    fn test_merge_ranges_coalesce() {
        // Ranges within the gap threshold should be merged
        let ranges = vec![0..100, 200..300, 500..600];
        let merged = super::merge_ranges(&ranges, 1024);
        assert_eq!(merged, vec![0..600]);
    }

    #[test]
    fn test_merge_ranges_overlapping() {
        let ranges = vec![0..200, 100..300];
        let merged = super::merge_ranges(&ranges, 0);
        assert_eq!(merged, vec![0..300]);
    }

    #[test]
    fn test_merge_ranges_unsorted() {
        let ranges = vec![500..600, 0..100, 200..300];
        let merged = super::merge_ranges(&ranges, 1024);
        assert_eq!(merged, vec![0..600]);
    }

    /// Mock FileRead backed by a flat byte buffer.
    struct MockFileRead {
        data: bytes::Bytes,
    }

    impl MockFileRead {
        fn new(size: usize) -> Self {
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            Self {
                data: bytes::Bytes::from(data),
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::io::FileRead for MockFileRead {
        async fn read(&self, range: Range<u64>) -> crate::Result<bytes::Bytes> {
            Ok(self.data.slice(range.start as usize..range.end as usize))
        }
    }

    #[tokio::test]
    async fn test_get_byte_ranges_no_coalesce() {
        use parquet::arrow::async_reader::AsyncFileReader;

        let mock = MockFileRead::new(2048);
        let expected_0 = mock.data.slice(0..100);
        let expected_1 = mock.data.slice(1500..1600);

        let mut reader =
            super::ArrowFileReader::new(crate::io::FileMetadata { size: 2048 }, Box::new(mock))
                .with_parquet_read_options(
                    super::ParquetReadOptions::builder()
                        .with_range_coalesce_bytes(0)
                        .build(),
                );

        let result = reader
            .get_byte_ranges(vec![0..100, 1500..1600])
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], expected_0);
        assert_eq!(result[1], expected_1);
    }

    #[tokio::test]
    async fn test_get_byte_ranges_with_coalesce() {
        use parquet::arrow::async_reader::AsyncFileReader;

        let mock = MockFileRead::new(1024);
        let expected_0 = mock.data.slice(0..100);
        let expected_1 = mock.data.slice(200..300);
        let expected_2 = mock.data.slice(500..600);

        let mut reader =
            super::ArrowFileReader::new(crate::io::FileMetadata { size: 1024 }, Box::new(mock))
                .with_parquet_read_options(
                    super::ParquetReadOptions::builder()
                        .with_range_coalesce_bytes(1024)
                        .build(),
                );

        // All ranges within coalesce threshold — should merge into one fetch.
        let result = reader
            .get_byte_ranges(vec![0..100, 200..300, 500..600])
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], expected_0);
        assert_eq!(result[1], expected_1);
        assert_eq!(result[2], expected_2);
    }

    #[tokio::test]
    async fn test_get_byte_ranges_empty() {
        use parquet::arrow::async_reader::AsyncFileReader;

        let mock = MockFileRead::new(1024);
        let mut reader =
            super::ArrowFileReader::new(crate::io::FileMetadata { size: 1024 }, Box::new(mock));

        let result = reader.get_byte_ranges(vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_get_byte_ranges_coalesce_max() {
        use parquet::arrow::async_reader::AsyncFileReader;

        let mock = MockFileRead::new(2048);
        let expected_0 = mock.data.slice(0..100);
        let expected_1 = mock.data.slice(1500..1600);

        let mut reader =
            super::ArrowFileReader::new(crate::io::FileMetadata { size: 2048 }, Box::new(mock))
                .with_parquet_read_options(
                    super::ParquetReadOptions::builder()
                        .with_range_coalesce_bytes(u64::MAX)
                        .build(),
                );

        // u64::MAX coalesce — all ranges merge into a single fetch.
        let result = reader
            .get_byte_ranges(vec![0..100, 1500..1600])
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], expected_0);
        assert_eq!(result[1], expected_1);
    }

    #[tokio::test]
    async fn test_get_byte_ranges_concurrency_zero() {
        use parquet::arrow::async_reader::AsyncFileReader;

        // concurrency=0 is clamped to 1, so this should not hang.
        let mock = MockFileRead::new(1024);
        let expected = mock.data.slice(0..100);

        let mut reader =
            super::ArrowFileReader::new(crate::io::FileMetadata { size: 1024 }, Box::new(mock))
                .with_parquet_read_options(
                    super::ParquetReadOptions::builder()
                        .with_range_fetch_concurrency(0)
                        .build(),
                );

        let result = reader
            .get_byte_ranges(vec![0..100, 200..300])
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], expected);
    }

    #[tokio::test]
    async fn test_get_byte_ranges_concurrency_one() {
        use parquet::arrow::async_reader::AsyncFileReader;

        let mock = MockFileRead::new(2048);
        let expected_0 = mock.data.slice(0..100);
        let expected_1 = mock.data.slice(500..600);
        let expected_2 = mock.data.slice(1500..1600);

        let mut reader =
            super::ArrowFileReader::new(crate::io::FileMetadata { size: 2048 }, Box::new(mock))
                .with_parquet_read_options(
                    super::ParquetReadOptions::builder()
                        .with_range_coalesce_bytes(0)
                        .with_range_fetch_concurrency(1)
                        .build(),
                );

        // concurrency=1 with no coalescing — sequential fetches.
        let result = reader
            .get_byte_ranges(vec![0..100, 500..600, 1500..1600])
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], expected_0);
        assert_eq!(result[1], expected_1);
        assert_eq!(result[2], expected_2);
    }

    /// Regression for <https://github.com/apache/iceberg-rust/issues/2306>:
    /// predicate on a column after nested types in a migrated file (no field IDs).
    /// Schema has struct, list, and map columns before the predicate target (`id`),
    /// exercising the fallback field ID mapping across all nested type variants.
    #[tokio::test]
    async fn test_predicate_on_migrated_file_with_nested_types() {
        use serde::{Deserialize, Serialize};
        use serde_arrow::schema::{SchemaLike, TracingOptions};

        #[derive(Serialize, Deserialize)]
        struct Person {
            name: String,
            age: i32,
        }

        #[derive(Serialize, Deserialize)]
        struct Row {
            person: Person,
            people: Vec<Person>,
            props: std::collections::BTreeMap<String, String>,
            id: i32,
        }

        let rows = vec![
            Row {
                person: Person {
                    name: "Alice".into(),
                    age: 30,
                },
                people: vec![Person {
                    name: "Alice".into(),
                    age: 30,
                }],
                props: [("k1".into(), "v1".into())].into(),
                id: 1,
            },
            Row {
                person: Person {
                    name: "Bob".into(),
                    age: 25,
                },
                people: vec![Person {
                    name: "Bob".into(),
                    age: 25,
                }],
                props: [("k2".into(), "v2".into())].into(),
                id: 2,
            },
            Row {
                person: Person {
                    name: "Carol".into(),
                    age: 40,
                },
                people: vec![Person {
                    name: "Carol".into(),
                    age: 40,
                }],
                props: [("k3".into(), "v3".into())].into(),
                id: 3,
            },
        ];

        let tracing_options = TracingOptions::default()
            .map_as_struct(false)
            .strings_as_large_utf8(false)
            .sequence_as_large_list(false);
        let fields = Vec::<arrow_schema::FieldRef>::from_type::<Row>(tracing_options).unwrap();
        let arrow_schema = Arc::new(ArrowSchema::new(fields.clone()));
        let batch = serde_arrow::to_record_batch(&fields, &rows).unwrap();

        // Fallback field IDs: person=1, people=2, props=3, id=4
        let iceberg_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(
                        1,
                        "person",
                        Type::Struct(crate::spec::StructType::new(vec![
                            NestedField::required(
                                5,
                                "name",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            NestedField::required(6, "age", Type::Primitive(PrimitiveType::Int))
                                .into(),
                        ])),
                    )
                    .into(),
                    NestedField::required(
                        2,
                        "people",
                        Type::List(crate::spec::ListType {
                            element_field: NestedField::required(
                                7,
                                "element",
                                Type::Struct(crate::spec::StructType::new(vec![
                                    NestedField::required(
                                        8,
                                        "name",
                                        Type::Primitive(PrimitiveType::String),
                                    )
                                    .into(),
                                    NestedField::required(
                                        9,
                                        "age",
                                        Type::Primitive(PrimitiveType::Int),
                                    )
                                    .into(),
                                ])),
                            )
                            .into(),
                        }),
                    )
                    .into(),
                    NestedField::required(
                        3,
                        "props",
                        Type::Map(crate::spec::MapType {
                            key_field: NestedField::required(
                                10,
                                "key",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            value_field: NestedField::required(
                                11,
                                "value",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                        }),
                    )
                    .into(),
                    NestedField::required(4, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/1.parquet");

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).unwrap();
        writer.write(&batch).expect("Writing batch");
        writer.close().unwrap();

        let predicate = Reference::new("id").greater_than(Datum::int(1));

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs())
            .with_row_group_filtering_enabled(true)
            .with_row_selection_enabled(true)
            .build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask {
                file_size_in_bytes: std::fs::metadata(&file_path).unwrap().len(),
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: Arc::from(file_path),
                data_file_format: DataFileFormat::Parquet,
                schema: iceberg_schema.clone(),
                project_field_ids: Arc::from(vec![4]),
                predicate: Some(Arc::new(predicate.bind(iceberg_schema, true).unwrap())),
                deletes: Arc::from(vec![]),
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
                split_offsets: None,
                first_row_id: None,
                file_sequence_number: None,
            })]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let ids: Vec<i32> = result
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(ids, vec![2, 3]);
    }

    // INT96 encoding: [nanos_low_u32, nanos_high_u32, julian_day_u32]
    // Julian day 2_440_588 = Unix epoch (1970-01-01)
    const UNIX_EPOCH_JULIAN: i64 = 2_440_588;
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    // Noon on 3333-01-01 (Julian day 2_953_529) — outside the i64 nanosecond range (~1677-2262).
    const INT96_TEST_NANOS_WITHIN_DAY: u64 = 43_200_000_000_000;
    const INT96_TEST_JULIAN_DAY: u32 = 2_953_529;

    fn make_int96_test_value() -> (parquet::data_type::Int96, i64) {
        let mut val = parquet::data_type::Int96::new();
        val.set_data(
            (INT96_TEST_NANOS_WITHIN_DAY & 0xFFFFFFFF) as u32,
            (INT96_TEST_NANOS_WITHIN_DAY >> 32) as u32,
            INT96_TEST_JULIAN_DAY,
        );
        let expected_micros = (INT96_TEST_JULIAN_DAY as i64 - UNIX_EPOCH_JULIAN) * MICROS_PER_DAY
            + (INT96_TEST_NANOS_WITHIN_DAY / 1_000) as i64;
        (val, expected_micros)
    }

    async fn read_int96_batches(
        file_path: &str,
        schema: SchemaRef,
        project_field_ids: Vec<i32>,
    ) -> Vec<RecordBatch> {
        let file_io = FileIO::new_with_fs();
        let reader = ArrowReaderBuilder::new(file_io).build();

        let file_size = std::fs::metadata(file_path).unwrap().len();
        let task = FileScanTask {
            file_size_in_bytes: file_size,
            start: 0,
            length: file_size,
            record_count: None,
            data_file_path: Arc::from(file_path),
            data_file_format: DataFileFormat::Parquet,
            schema,
            project_field_ids: Arc::from(project_field_ids),

            predicate: None,
            deletes: Arc::from(vec![]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };

        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        reader.read(tasks).unwrap().try_collect().await.unwrap()
    }

    // ArrowWriter cannot write INT96, so we use SerializedFileWriter directly.
    fn write_int96_parquet_file(
        table_location: &str,
        filename: &str,
        with_field_ids: bool,
    ) -> (String, Vec<i64>) {
        use parquet::basic::{Repetition, Type as PhysicalType};
        use parquet::data_type::{Int32Type, Int96, Int96Type};
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::types::Type as SchemaType;

        let file_path = format!("{table_location}/{filename}");

        let mut ts_builder = SchemaType::primitive_type_builder("ts", PhysicalType::INT96)
            .with_repetition(Repetition::OPTIONAL);
        let mut id_builder = SchemaType::primitive_type_builder("id", PhysicalType::INT32)
            .with_repetition(Repetition::REQUIRED);

        if with_field_ids {
            ts_builder = ts_builder.with_id(Some(1));
            id_builder = id_builder.with_id(Some(2));
        }

        let schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![
                Arc::new(ts_builder.build().unwrap()),
                Arc::new(id_builder.build().unwrap()),
            ])
            .build()
            .unwrap();

        // Dates outside the i64 nanosecond range (~1677-2262) overflow without coercion.
        const NOON_NANOS: u64 = INT96_TEST_NANOS_WITHIN_DAY;
        const JULIAN_3333: u32 = INT96_TEST_JULIAN_DAY;
        const JULIAN_2100: u32 = 2_488_070;

        let test_data: Vec<(u32, u32, u32, i64)> = vec![
            // 3333-01-01 00:00:00
            (
                0,
                0,
                JULIAN_3333,
                (JULIAN_3333 as i64 - UNIX_EPOCH_JULIAN) * MICROS_PER_DAY,
            ),
            // 3333-01-01 12:00:00
            (
                (NOON_NANOS & 0xFFFFFFFF) as u32,
                (NOON_NANOS >> 32) as u32,
                JULIAN_3333,
                (JULIAN_3333 as i64 - UNIX_EPOCH_JULIAN) * MICROS_PER_DAY
                    + (NOON_NANOS / 1_000) as i64,
            ),
            // 2100-01-01 00:00:00
            (
                0,
                0,
                JULIAN_2100,
                (JULIAN_2100 as i64 - UNIX_EPOCH_JULIAN) * MICROS_PER_DAY,
            ),
        ];

        let int96_values: Vec<Int96> = test_data
            .iter()
            .map(|(lo, hi, day, _)| {
                let mut v = Int96::new();
                v.set_data(*lo, *hi, *day);
                v
            })
            .collect();

        let id_values: Vec<i32> = (0..test_data.len() as i32).collect();
        let expected_micros: Vec<i64> = test_data.iter().map(|(_, _, _, m)| *m).collect();

        let file = File::create(&file_path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, Arc::new(schema), Default::default()).unwrap();

        let mut row_group = writer.next_row_group().unwrap();
        {
            // def=1: ts is OPTIONAL and present. No repetition levels (top-level columns).
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int96Type>()
                .write_batch(&int96_values, Some(&vec![1; test_data.len()]), None)
                .unwrap();
            col.close().unwrap();
        }
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int32Type>()
                .write_batch(&id_values, None, None)
                .unwrap();
            col.close().unwrap();
        }
        row_group.close().unwrap();
        writer.close().unwrap();

        (file_path, expected_micros)
    }

    async fn assert_int96_read_matches(
        file_path: &str,
        schema: SchemaRef,
        project_field_ids: Vec<i32>,
        expected_micros: &[i64],
    ) {
        use arrow_array::TimestampMicrosecondArray;

        let batches = read_int96_batches(file_path, schema, project_field_ids).await;

        assert_eq!(batches.len(), 1);
        let ts_array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("Expected TimestampMicrosecondArray");

        for (i, expected) in expected_micros.iter().enumerate() {
            assert_eq!(
                ts_array.value(i),
                *expected,
                "Row {i}: got {}, expected {expected}",
                ts_array.value(i)
            );
        }
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_with_field_ids() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp))
                        .into(),
                    NestedField::required(2, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let (file_path, expected_micros) =
            write_int96_parquet_file(&table_location, "with_ids.parquet", true);

        assert_int96_read_matches(&file_path, schema, vec![1, 2], &expected_micros).await;
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_without_field_ids() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "ts", Type::Primitive(PrimitiveType::Timestamp))
                        .into(),
                    NestedField::required(2, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let (file_path, expected_micros) =
            write_int96_parquet_file(&table_location, "no_ids.parquet", false);

        assert_int96_read_matches(&file_path, schema, vec![1, 2], &expected_micros).await;
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_in_struct() {
        use arrow_array::{StructArray, TimestampMicrosecondArray};
        use parquet::basic::{Repetition, Type as PhysicalType};
        use parquet::data_type::Int96Type;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::types::Type as SchemaType;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/struct_int96.parquet");

        let ts_type = SchemaType::primitive_type_builder("ts", PhysicalType::INT96)
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(2))
            .build()
            .unwrap();

        let struct_type = SchemaType::group_type_builder("data")
            .with_repetition(Repetition::REQUIRED)
            .with_id(Some(1))
            .with_fields(vec![Arc::new(ts_type)])
            .build()
            .unwrap();

        let parquet_schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(struct_type)])
            .build()
            .unwrap();

        let (int96_val, expected_micros) = make_int96_test_value();

        let file = File::create(&file_path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, Arc::new(parquet_schema), Default::default()).unwrap();

        // def=1: struct is REQUIRED so no level, ts is OPTIONAL and present (1).
        // No repetition levels needed (no repeated groups).
        let mut row_group = writer.next_row_group().unwrap();
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int96Type>()
                .write_batch(&[int96_val], Some(&[1]), None)
                .unwrap();
            col.close().unwrap();
        }
        row_group.close().unwrap();
        writer.close().unwrap();

        let iceberg_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(
                        1,
                        "data",
                        Type::Struct(crate::spec::StructType::new(vec![
                            NestedField::optional(
                                2,
                                "ts",
                                Type::Primitive(PrimitiveType::Timestamp),
                            )
                            .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let batches = read_int96_batches(&file_path, iceberg_schema, vec![1]).await;

        assert_eq!(batches.len(), 1);
        let struct_array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("Expected StructArray");
        let ts_array = struct_array
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("Expected TimestampMicrosecondArray inside struct");

        assert_eq!(
            ts_array.value(0),
            expected_micros,
            "INT96 in struct: got {}, expected {expected_micros}",
            ts_array.value(0)
        );
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_in_list() {
        use arrow_array::{ListArray, TimestampMicrosecondArray};
        use parquet::basic::{Repetition, Type as PhysicalType};
        use parquet::data_type::Int96Type;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::types::Type as SchemaType;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/list_int96.parquet");

        // 3-level LIST encoding:
        //   optional group timestamps (LIST) {
        //     repeated group list {
        //       optional int96 element;
        //     }
        //   }
        let element_type = SchemaType::primitive_type_builder("element", PhysicalType::INT96)
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(2))
            .build()
            .unwrap();

        let list_group = SchemaType::group_type_builder("list")
            .with_repetition(Repetition::REPEATED)
            .with_fields(vec![Arc::new(element_type)])
            .build()
            .unwrap();

        let list_type = SchemaType::group_type_builder("timestamps")
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(1))
            .with_logical_type(Some(parquet::basic::LogicalType::List))
            .with_fields(vec![Arc::new(list_group)])
            .build()
            .unwrap();

        let parquet_schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(list_type)])
            .build()
            .unwrap();

        let (int96_val, expected_micros) = make_int96_test_value();

        let file = File::create(&file_path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, Arc::new(parquet_schema), Default::default()).unwrap();

        // Write a single row with a list containing one INT96 element.
        // def=3: list present (1) + repeated group (2) + element present (3)
        // rep=0: start of a new list
        let mut row_group = writer.next_row_group().unwrap();
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int96Type>()
                .write_batch(&[int96_val], Some(&[3]), Some(&[0]))
                .unwrap();
            col.close().unwrap();
        }
        row_group.close().unwrap();
        writer.close().unwrap();

        let iceberg_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(
                        1,
                        "timestamps",
                        Type::List(crate::spec::ListType {
                            element_field: NestedField::optional(
                                2,
                                "element",
                                Type::Primitive(PrimitiveType::Timestamp),
                            )
                            .into(),
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let batches = read_int96_batches(&file_path, iceberg_schema, vec![1]).await;

        assert_eq!(batches.len(), 1);
        let list_array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray");
        let ts_array = list_array
            .values()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("Expected TimestampMicrosecondArray inside list");

        assert_eq!(
            ts_array.value(0),
            expected_micros,
            "INT96 in list: got {}, expected {expected_micros}",
            ts_array.value(0)
        );
    }

    #[tokio::test]
    async fn test_read_int96_timestamps_in_map() {
        use arrow_array::{MapArray, TimestampMicrosecondArray};
        use parquet::basic::{Repetition, Type as PhysicalType};
        use parquet::data_type::{ByteArrayType, Int96Type};
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::types::Type as SchemaType;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/map_int96.parquet");

        // MAP encoding: optional group ts_map (MAP) { repeated group key_value { required binary
        // key (UTF8); optional int96 value; } }.
        let key_type = SchemaType::primitive_type_builder("key", PhysicalType::BYTE_ARRAY)
            .with_repetition(Repetition::REQUIRED)
            .with_logical_type(Some(parquet::basic::LogicalType::String))
            .with_id(Some(2))
            .build()
            .unwrap();

        let value_type = SchemaType::primitive_type_builder("value", PhysicalType::INT96)
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(3))
            .build()
            .unwrap();

        let key_value_group = SchemaType::group_type_builder("key_value")
            .with_repetition(Repetition::REPEATED)
            .with_fields(vec![Arc::new(key_type), Arc::new(value_type)])
            .build()
            .unwrap();

        let map_type = SchemaType::group_type_builder("ts_map")
            .with_repetition(Repetition::OPTIONAL)
            .with_id(Some(1))
            .with_logical_type(Some(parquet::basic::LogicalType::Map))
            .with_fields(vec![Arc::new(key_value_group)])
            .build()
            .unwrap();

        let parquet_schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(map_type)])
            .build()
            .unwrap();

        let (int96_val, expected_micros) = make_int96_test_value();

        let file = File::create(&file_path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, Arc::new(parquet_schema), Default::default()).unwrap();

        // Write a single row with a map containing one key-value pair.
        // rep=0 for both columns: start of a new map.
        // key def=2: map present (1) + key_value entry present (2), key is REQUIRED.
        // value def=3: map present (1) + key_value entry present (2) + value present (3).
        let mut row_group = writer.next_row_group().unwrap();
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<ByteArrayType>()
                .write_batch(
                    &[parquet::data_type::ByteArray::from("event_time")],
                    Some(&[2]),
                    Some(&[0]),
                )
                .unwrap();
            col.close().unwrap();
        }
        {
            let mut col = row_group.next_column().unwrap().unwrap();
            col.typed::<Int96Type>()
                .write_batch(&[int96_val], Some(&[3]), Some(&[0]))
                .unwrap();
            col.close().unwrap();
        }
        row_group.close().unwrap();
        writer.close().unwrap();

        let iceberg_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(
                        1,
                        "ts_map",
                        Type::Map(crate::spec::MapType {
                            key_field: NestedField::required(
                                2,
                                "key",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            value_field: NestedField::optional(
                                3,
                                "value",
                                Type::Primitive(PrimitiveType::Timestamp),
                            )
                            .into(),
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let batches = read_int96_batches(&file_path, iceberg_schema, vec![1]).await;

        assert_eq!(batches.len(), 1);
        let map_array = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<MapArray>()
            .expect("Expected MapArray");
        let ts_array = map_array
            .values()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("Expected TimestampMicrosecondArray as map values");

        assert_eq!(
            ts_array.value(0),
            expected_micros,
            "INT96 in map: got {}, expected {expected_micros}",
            ts_array.value(0)
        );
    }

    // A variant column contributes its own field id to the projection-mask leaf set, like a
    // primitive. Java `TypeUtil.select` keeps the column whole. No real scan reaches this arm
    // today, so only this unit test stops a dropped `field_ids.push` from projecting variant
    // columns out once the arrow door opens.
    #[test]
    fn test_include_leaf_field_id_treats_variant_as_leaf() {
        let variant_field: NestedField = NestedField::optional(7, "v", Type::Variant);
        let mut field_ids = vec![];
        ArrowReader::include_leaf_field_id(&variant_field, &mut field_ids);
        assert_eq!(
            field_ids,
            vec![7],
            "a variant column is one projectable leaf (its own field id)"
        );
    }
}

#[cfg(test)]
mod avro_scan_tests {
    //! Scan-level tests for the Avro and ORC read paths, over a real Avro OCF and the committed
    //! golden Java ORC fixture. They cover projection, merge-on-read deletes applied after
    //! materialization, and resolution by field id under a rename. `avro_reader_tests.rs` and
    //! `orc_reader_tests.rs` cover the decode cores.

    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use apache_avro::Writer as AvroWriter;
    use apache_avro::types::Value as AvroWriteValue;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use crate::arrow::ArrowReaderBuilder;
    use crate::avro::schema_to_avro_schema;
    use crate::io::FileIO;
    use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
    use crate::spec::{
        DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema, SchemaRef, Type,
    };

    /// The 2-field test table schema `{1 id long required, 2 data string optional}` (the same shape
    /// the scan-exec interop oracle uses).
    fn test_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("build test schema"),
        )
    }

    /// Write a real Avro OCF data file of `(id, data)` rows to `path`, using the canonical
    /// iceberg→avro schema conversion (which stamps the `field-id` props the reader resolves by).
    fn write_avro_data_file(path: &str, schema: &Schema, rows: &[(i64, Option<&str>)]) {
        let avro_schema = schema_to_avro_schema("data", schema).expect("iceberg→avro schema");
        let mut writer = AvroWriter::new(&avro_schema, Vec::new());
        for (id, data) in rows {
            let data_value = match data {
                Some(s) => {
                    AvroWriteValue::Union(1, Box::new(AvroWriteValue::String(s.to_string())))
                }
                None => AvroWriteValue::Union(0, Box::new(AvroWriteValue::Null)),
            };
            let record = AvroWriteValue::Record(vec![
                ("id".to_string(), AvroWriteValue::Long(*id)),
                ("data".to_string(), data_value),
            ]);
            writer.append_value_ref(&record).expect("append avro row");
        }
        let bytes = writer.into_inner().expect("finalize avro OCF");
        std::fs::write(path, bytes).expect("write avro file");
    }

    /// The Avro/ORC half of the same defect: `build_expected_schema` is the only schema those
    /// readers see, and it excluded the pair. Parameterised over which column.
    #[tokio::test]
    async fn stored_row_lineage_in_an_avro_file_survives_the_real_reader() {
        for (column_name, field_id, stored, expected) in [
            (
                "_row_id",
                crate::metadata_columns::RESERVED_FIELD_ID_ROW_ID,
                vec![Some(777i64), None, Some(999)],
                // The NULL row falls back to `first_row_id + its own position` (1000 + 1).
                vec![Some(777i64), Some(1_001), Some(999)],
            ),
            (
                "_last_updated_sequence_number",
                crate::metadata_columns::RESERVED_FIELD_ID_LAST_UPDATED_SEQUENCE_NUMBER,
                vec![Some(31i64), None, Some(33)],
                // The NULL row falls back to the FILE's sequence number.
                vec![Some(31i64), Some(42), Some(33)],
            ),
        ] {
            let tmp = TempDir::new().expect("temp dir");
            let path = tmp
                .path()
                .join(format!("lineage{field_id}.avro"))
                .to_string_lossy()
                .to_string();

            let reserved = crate::metadata_columns::get_metadata_field(field_id)
                .expect("reserved field")
                .clone();
            let file_schema = Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    reserved,
                ])
                .build()
                .expect("file schema");

            let avro_schema = schema_to_avro_schema("data", &file_schema).expect("iceberg→avro");
            let mut writer = AvroWriter::new(&avro_schema, Vec::new());
            for (row, value) in stored.iter().enumerate() {
                let stored_value = match value {
                    Some(v) => AvroWriteValue::Union(1, Box::new(AvroWriteValue::Long(*v))),
                    None => AvroWriteValue::Union(0, Box::new(AvroWriteValue::Null)),
                };
                writer
                    .append_value_ref(&AvroWriteValue::Record(vec![
                        ("id".to_string(), AvroWriteValue::Long(row as i64 + 1)),
                        (column_name.to_string(), stored_value),
                    ]))
                    .expect("append row");
            }
            std::fs::write(&path, writer.into_inner().expect("finalize")).expect("write avro");

            // The TABLE schema carries only `id`; both are reserved metadata columns.
            let table_schema = Arc::new(
                Schema::builder()
                    .with_schema_id(1)
                    .with_fields(vec![
                        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    ])
                    .build()
                    .expect("table schema"),
            );
            let mut task = avro_task(&path, table_schema, vec![1, field_id], vec![]);
            task.first_row_id = Some(1_000);
            task.file_sequence_number = Some(42);

            let batches = run_scan(FileIO::new_with_fs(), task).await;
            let values: Vec<Option<i64>> = batches
                .iter()
                .flat_map(|batch| {
                    let column = batch
                        .column_by_name(column_name)
                        .expect("projected")
                        .as_any()
                        .downcast_ref::<arrow_array::Int64Array>()
                        .expect("Int64")
                        .clone();
                    (0..column.len())
                        .map(|row| {
                            if arrow_array::Array::is_null(&column, row) {
                                None
                            } else {
                                Some(column.value(row))
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            assert_eq!(
                values, expected,
                "the STORED {column_name} values must win on the AVRO path; a fully-computed \
                 column means the stored one never reached the transformer"
            );
        }
    }

    /// Build a whole-file [`FileScanTask`] for an Avro data file at `path` projecting
    /// `project_field_ids`, with the given delete attachments.
    fn avro_task(
        path: &str,
        schema: SchemaRef,
        project_field_ids: Vec<i32>,
        deletes: Vec<FileScanTaskDeleteFile>,
    ) -> FileScanTask {
        FileScanTask {
            file_size_in_bytes: std::fs::metadata(path).expect("stat avro file").len(),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(path),
            data_file_format: DataFileFormat::Avro,
            schema,
            project_field_ids: Arc::from(project_field_ids),

            predicate: None,
            deletes: Arc::from(deletes),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        }
    }

    /// Drive a one-task scan and collect the resulting batches.
    async fn run_scan(file_io: FileIO, task: FileScanTask) -> Vec<RecordBatch> {
        let reader = ArrowReaderBuilder::new(file_io).build();
        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        reader
            .read(tasks)
            .expect("build scan stream")
            .try_collect::<Vec<RecordBatch>>()
            .await
            .expect("collect scan batches")
    }

    /// Extract the sorted `(id, data)` rows from the scan batches.
    fn rows_of(batches: &[RecordBatch]) -> Vec<(i64, Option<String>)> {
        let mut out = Vec::new();
        for batch in batches {
            let id = batch
                .column_by_name("id")
                .expect("id column")
                .as_primitive::<Int64Type>();
            let data = batch.column_by_name("data").expect("data column");
            for i in 0..batch.num_rows() {
                let d = if data.is_null(i) {
                    None
                } else {
                    match data.data_type() {
                        DataType::Utf8 => Some(data.as_string::<i32>().value(i).to_string()),
                        DataType::LargeUtf8 => Some(data.as_string::<i64>().value(i).to_string()),
                        other => panic!("unexpected data arrow type {other:?}"),
                    }
                };
                out.push((id.value(i), d));
            }
        }
        out.sort_by_key(|(id, _)| *id);
        out
    }

    /// Write a real parquet POSITION-delete file (`file_path` + `pos` columns) deleting the given
    /// positions of `referenced_data_path`, returning a [`FileScanTaskDeleteFile`] pointing at it.
    fn write_pos_delete_file(
        path: &str,
        referenced_data_path: &str,
        positions: &[i64],
    ) -> FileScanTaskDeleteFile {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                (i32::MAX - 101).to_string(),
            )])),
            Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                (i32::MAX - 102).to_string(),
            )])),
        ]));
        let paths: Vec<&str> = positions.iter().map(|_| referenced_data_path).collect();
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions.to_vec())) as ArrayRef,
        ])
        .expect("build pos-delete batch");
        let file = File::create(path).expect("create pos-delete file");
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                .expect("pos-delete writer");
        writer.write(&batch).expect("write pos-delete batch");
        writer.close().expect("close pos-delete writer");
        FileScanTaskDeleteFile {
            file_path: path.to_string(),
            file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }
    }

    /// Write a real parquet EQUALITY-delete file on field id 1 (`id`) deleting the given id values,
    /// returning a [`FileScanTaskDeleteFile`] pointing at it.
    fn write_eq_delete_file(path: &str, delete_ids: &[i64]) -> FileScanTaskDeleteFile {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(
            delete_ids.to_vec(),
        )) as ArrayRef])
        .expect("build eq-delete batch");
        let file = File::create(path).expect("create eq-delete file");
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                .expect("eq-delete writer");
        writer.write(&batch).expect("write eq-delete batch");
        writer.close().expect("close eq-delete writer");
        FileScanTaskDeleteFile {
            file_path: path.to_string(),
            file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: Some(vec![1]),
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }
    }

    // -- ORC scan helpers: the committed Java-Iceberg golden fixture (the ONLY ORC source with the
    //    `iceberg.id` footer attributes the by-field-id reader requires — orc-rust's writer does not
    //    stamp them, so a generated ORC file would be rejected loudly). The fixture is 3 rows / 14
    //    columns; the ORC scan tests project field ids 1 (`id` long [1,2,3]) and 6 (`string_col`). ---

    /// The committed Java-Iceberg 1.10.0 golden ORC file (ZLIB, 14 columns, 3 rows).
    const ORC_FIXTURE: &[u8] = include_bytes!("../../testdata/orc/iceberg_primitives.orc");

    /// Write the golden ORC fixture into `tmp` and return its path (the scan reads via `FileIO`).
    fn orc_fixture_on_disk(tmp: &TempDir) -> String {
        let path = tmp.path().join("fixture.orc").to_string_lossy().to_string();
        std::fs::write(&path, ORC_FIXTURE).expect("write orc fixture");
        path
    }

    /// The two-field projection schema over the fixture: `{1 id long required, 6 string_col string
    /// optional}` — the fixture's field ids 1 and 6, used by the read-all / projection tests.
    fn orc_fixture_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(6, "string_col", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .expect("build orc fixture schema"),
        )
    }

    /// Build a whole-file [`FileScanTask`] for an ORC data file (`DataFileFormat::Orc`) at `path`.
    fn orc_task(
        path: &str,
        schema: SchemaRef,
        project_field_ids: Vec<i32>,
        deletes: Vec<FileScanTaskDeleteFile>,
    ) -> FileScanTask {
        FileScanTask {
            file_size_in_bytes: std::fs::metadata(path).expect("stat orc file").len(),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(path),
            data_file_format: DataFileFormat::Orc,
            schema,
            project_field_ids: Arc::from(project_field_ids),

            predicate: None,
            deletes: Arc::from(deletes),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        }
    }

    /// Extract `(id, string_col)` rows from the ORC fixture scan, sorted by id.
    fn orc_id_string_rows(batches: &[RecordBatch]) -> Vec<(i64, Option<String>)> {
        let mut out = Vec::new();
        for batch in batches {
            let id = batch
                .column_by_name("id")
                .expect("id column")
                .as_primitive::<Int64Type>();
            let s = batch.column_by_name("string_col").expect("string_col");
            for i in 0..batch.num_rows() {
                let v = if s.is_null(i) {
                    None
                } else {
                    match s.data_type() {
                        DataType::Utf8 => Some(s.as_string::<i32>().value(i).to_string()),
                        DataType::LargeUtf8 => Some(s.as_string::<i64>().value(i).to_string()),
                        other => panic!("unexpected string_col arrow type {other:?}"),
                    }
                };
                out.push((id.value(i), v));
            }
        }
        out.sort_by_key(|(id, _)| *id);
        out
    }

    // -- End-to-end Avro scan (no deletes). -----------------------------------------------------------

    #[tokio::test]
    async fn avro_scan_reads_all_rows() {
        let tmp = TempDir::new().unwrap();
        let schema = test_schema();
        let data_path = tmp.path().join("data.avro").to_string_lossy().to_string();
        write_avro_data_file(&data_path, &schema, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, None),
        ]);

        let task = avro_task(&data_path, schema, vec![1, 2], vec![]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;

        assert_eq!(rows_of(&batches), vec![
            (10, Some("a".to_string())),
            (20, Some("b".to_string())),
            (30, None),
        ]);
    }

    // -- U3 cycle 3 / hazard-1 sibling: AVRO must never be read through a byte sub-window. ------------

    /// The AVRO reader decodes WHOLE files — it never reads `task.start` / `task.length`. So if the
    /// planner split an Avro file into byte windows, every sub-task would re-emit every row and an
    /// N-way split would silently return N copies of the file: the exact silent-duplication class
    /// the Parquet midpoint row-group selection was written to eliminate, with no error at any
    /// layer.
    #[tokio::test]
    async fn avro_split_reads_every_row_exactly_once() {
        let tmp = TempDir::new().unwrap();
        let schema = test_schema();
        let data_path = tmp.path().join("split.avro").to_string_lossy().to_string();
        let rows: Vec<(i64, Option<&str>)> = (0..200i64)
            .map(|i| (i, if i % 3 == 0 { None } else { Some("payload") }))
            .collect();
        write_avro_data_file(&data_path, &schema, &rows);

        let whole = avro_task(&data_path, schema, vec![1, 2], vec![]);
        let file_len = whole.file_size_in_bytes;
        // `plan_tasks` hands `split` the file length as the task length; mirror that exactly.
        let whole = FileScanTask {
            length: file_len,
            ..whole
        };

        // Non-vacuity: the target must be small enough that a split WOULD produce several windows.
        // A PARQUET task with the same geometry does split, which is what makes this test able to
        // fail when the AVRO gate is removed.
        let target = file_len / 4 + 1;
        assert!(
            target < file_len,
            "fixture is non-discriminating: the split target ({target}) must be well under the \
             file length ({file_len}), otherwise a single window is the trivially correct answer"
        );
        let parquet_shaped = FileScanTask {
            data_file_format: DataFileFormat::Parquet,
            ..whole.clone()
        };
        assert!(
            parquet_shaped.split(target).expect("parquet split").len() > 1,
            "fixture is non-discriminating: this geometry must split into several windows for a \
             format whose reader honours byte ranges"
        );

        let sub_tasks = whole.split(target).expect("avro split");
        assert_eq!(
            sub_tasks.len(),
            1,
            "an AVRO file must not be split into byte windows its reader cannot honour"
        );

        let mut all_ids = Vec::new();
        for sub_task in sub_tasks {
            let batches = run_scan(FileIO::new_with_fs(), sub_task).await;
            all_ids.extend(rows_of(&batches).into_iter().map(|(id, _)| id));
        }
        all_ids.sort_unstable();
        assert_eq!(
            all_ids,
            (0..200i64).collect::<Vec<_>>(),
            "the union over every sub-task must be each row EXACTLY once, never a duplicate"
        );
    }

    /// Defence in depth for the same hazard: even if a ranged AVRO/ORC task reaches the reader by
    /// some other route (the public `PartitionWork` seam, a hand-built task), it must fail with a
    /// typed error rather than silently re-emitting the whole file.
    #[tokio::test]
    async fn avro_ranged_task_is_rejected_with_a_typed_error() {
        let tmp = TempDir::new().unwrap();
        let schema = test_schema();
        let data_path = tmp.path().join("ranged.avro").to_string_lossy().to_string();
        write_avro_data_file(&data_path, &schema, &[(1, Some("a")), (2, Some("b"))]);

        let whole = avro_task(&data_path, schema, vec![1, 2], vec![]);
        let file_len = whole.file_size_in_bytes;
        assert!(
            file_len > 2,
            "fixture guard: the Avro file must be non-trivial"
        );

        // Genuine sub-windows on BOTH axes of the guard. Varying only the LENGTH would leave the
        // guard's `task.start == 0 &&` half unpinned: `(1, file_len)` is a real window that a
        // start-blind guard would ACCEPT, and the Avro reader would then re-emit the whole file —
        // the silent-duplication class this guard exists to stop. (U3 cycle 4 / F-3.)
        for (start, length) in [(0u64, file_len / 2), (1u64, file_len)] {
            let ranged = FileScanTask {
                start,
                length,
                ..whole.clone()
            };
            let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
            let tasks = Box::pin(futures::stream::iter(vec![Ok(ranged)])) as FileScanTaskStream;
            let result = reader
                .read(tasks)
                .expect("build scan stream")
                .try_collect::<Vec<RecordBatch>>()
                .await;
            let err = match result {
                Ok(batches) => panic!(
                    "a ranged AVRO task ({start}, {length}) must fail closed, not re-emit the \
                     whole file — got {} row(s)",
                    rows_of(&batches).len()
                ),
                Err(err) => err,
            };
            assert_eq!(err.kind(), crate::ErrorKind::FeatureUnsupported);
            assert!(
                err.to_string().contains("ranged split task over a AVRO"),
                "the typed error must name the ranged AVRO task ({start}, {length}), got: {err}"
            );
        }

        // Both whole-file spellings must still be accepted: the legacy `length == 0` sentinel and
        // an explicit `length == file_size_in_bytes`.
        for length in [0, file_len] {
            let batches = run_scan(FileIO::new_with_fs(), FileScanTask {
                start: 0,
                length,
                ..whole.clone()
            })
            .await;
            assert_eq!(
                rows_of(&batches).len(),
                2,
                "a whole-file AVRO task (length {length}) must still read normally"
            );
        }
    }

    // -- Projection: only the projected field id materializes. ----------------------------------------

    #[tokio::test]
    async fn avro_scan_projects_single_column() {
        let tmp = TempDir::new().unwrap();
        let schema = test_schema();
        let data_path = tmp.path().join("data.avro").to_string_lossy().to_string();
        write_avro_data_file(&data_path, &schema, &[(10, Some("a")), (20, Some("b"))]);

        // Project only `id` (field 1). The output batch must have exactly one column = id.
        let task = avro_task(&data_path, schema, vec![1], vec![]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;

        assert_eq!(batches[0].num_columns(), 1, "only `id` is projected");
        let ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .expect("id col")
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(ids, vec![10, 20]);
    }

    // -- MoR positional deletes applied post-materialization. -----------------------------------------

    #[tokio::test]
    async fn avro_scan_applies_positional_deletes() {
        let tmp = TempDir::new().unwrap();
        let schema = test_schema();
        let data_path = tmp.path().join("data.avro").to_string_lossy().to_string();
        // 5 rows at positions 0..4.
        write_avro_data_file(&data_path, &schema, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
            (40, Some("d")),
            (50, Some("e")),
        ]);
        // Delete positions {1, 3} (rows id=20, id=40).
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let pos_del = write_pos_delete_file(&del_path, &data_path, &[1, 3]);

        let task = avro_task(&data_path, schema, vec![1, 2], vec![pos_del]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;

        // Survivors are exactly {10, 30, 50} — matching the Parquet position-delete semantics.
        assert_eq!(rows_of(&batches), vec![
            (10, Some("a".to_string())),
            (30, Some("c".to_string())),
            (50, Some("e".to_string())),
        ]);
    }

    // -- MoR equality deletes applied by VALUE. -------------------------------------------------------

    #[tokio::test]
    async fn avro_scan_applies_equality_deletes() {
        let tmp = TempDir::new().unwrap();
        let schema = test_schema();
        let data_path = tmp.path().join("data.avro").to_string_lossy().to_string();
        write_avro_data_file(&data_path, &schema, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
            (40, Some("d")),
            (50, Some("e")),
        ]);
        // Equality-delete ids {20, 40} by VALUE on field id 1.
        let del_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_del = write_eq_delete_file(&del_path, &[20, 40]);

        let task = avro_task(&data_path, schema, vec![1, 2], vec![eq_del]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;

        assert_eq!(rows_of(&batches), vec![
            (10, Some("a".to_string())),
            (30, Some("c".to_string())),
            (50, Some("e".to_string())),
        ]);
    }

    // -- ORC dispatch now SCANS the file (U2): the golden Java-Iceberg fixture, by field-id. ---------
    //
    // The pre-U2 `orc_data_file_errors_cleanly` (which asserted the OLD FeatureUnsupported behavior)
    // is RETARGETED here to a REAL ORC scan: dispatching `DataFileFormat::Orc` now routes through
    // `process_orc_file_scan_task` → the U1 ORC reader → the SAME transformer + delete machinery the
    // Avro path uses. A file failing to materialize (or a wrong format dispatch) RED-s this test.

    #[tokio::test]
    async fn orc_scan_reads_fixture_rows() {
        let tmp = TempDir::new().unwrap();
        let path = orc_fixture_on_disk(&tmp);
        // The golden fixture's field 1 (`id` long) carries [1, 2, 3]; field 6 (`string_col`) carries
        // ["hello", null, ""]. Project both and scan through the full reader.
        let task = orc_task(&path, orc_fixture_schema(), vec![1, 6], vec![]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;

        assert_eq!(orc_id_string_rows(&batches), vec![
            (1, Some("hello".to_string())),
            (2, None),
            (3, Some(String::new())),
        ]);
    }

    /// The ORC half of the fail-closed guard, mirroring
    /// [`avro_ranged_task_is_rejected_with_a_typed_error`] one-for-one.
    /// `reject_ranged_whole_file_task` is invoked from BOTH `process_avro_file_scan_task` and
    /// `process_orc_file_scan_task`, but only the AVRO call site was pinned: deleting the ORC line
    /// left the whole suite green. Since `process_orc_file_scan_task` never reads `task.start` /
    /// `task.length`, an unguarded ranged ORC task re-emits every row of the file — N copies per.
    #[tokio::test]
    async fn orc_ranged_task_is_rejected_with_a_typed_error() {
        let tmp = TempDir::new().unwrap();
        let path = orc_fixture_on_disk(&tmp);
        let whole = orc_task(&path, orc_fixture_schema(), vec![1, 6], vec![]);
        let file_len = whole.file_size_in_bytes;
        assert!(
            file_len > 2,
            "fixture guard: the ORC file must be non-trivial"
        );

        // BOTH axes of the guard: `(0, file_len / 2)` varies only the length, `(1, file_len)` is a
        // genuine window that a start-blind guard would ACCEPT — after which the whole-file ORC
        // reader would re-emit every row.
        for (start, length) in [(0u64, file_len / 2), (1u64, file_len)] {
            let ranged = FileScanTask {
                start,
                length,
                ..whole.clone()
            };
            let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
            let tasks = Box::pin(futures::stream::iter(vec![Ok(ranged)])) as FileScanTaskStream;
            let result = reader
                .read(tasks)
                .expect("build scan stream")
                .try_collect::<Vec<RecordBatch>>()
                .await;
            let err = match result {
                Ok(batches) => panic!(
                    "a ranged ORC task ({start}, {length}) must fail closed, not re-emit the whole \
                     file — got {} row(s)",
                    orc_id_string_rows(&batches).len()
                ),
                Err(err) => err,
            };
            assert_eq!(err.kind(), crate::ErrorKind::FeatureUnsupported);
            assert!(
                err.to_string().contains("ranged split task over a ORC"),
                "the typed error must name the ranged ORC task ({start}, {length}), got: {err}"
            );
        }

        // Non-vacuity: both whole-file spellings must still stream, so this cannot pass on a reader
        // that errors on everything.
        for length in [0, file_len] {
            let batches = run_scan(FileIO::new_with_fs(), FileScanTask {
                start: 0,
                length,
                ..whole.clone()
            })
            .await;
            assert_eq!(
                orc_id_string_rows(&batches).len(),
                3,
                "a whole-file ORC task (length {length}) must still read normally"
            );
        }
    }

    // -- ORC projection: only the projected field id materializes (by field-id, not position). -------

    #[tokio::test]
    async fn orc_scan_projects_single_column() {
        let tmp = TempDir::new().unwrap();
        let path = orc_fixture_on_disk(&tmp);
        // Project only `id` (field 1). The output batch must have exactly one column.
        let task = orc_task(&path, orc_fixture_schema(), vec![1], vec![]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;

        assert_eq!(batches[0].num_columns(), 1, "only `id` is projected");
        let ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .expect("id col")
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    // -- ORC MoR positional deletes applied post-materialization (the SAME DeleteVector path). --------

    #[tokio::test]
    async fn orc_scan_applies_positional_deletes() {
        let tmp = TempDir::new().unwrap();
        let path = orc_fixture_on_disk(&tmp);
        // Delete position 1 (the id=2 row). The delete file references the ORC data path.
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let pos_del = write_pos_delete_file(&del_path, &path, &[1]);

        let task = orc_task(&path, orc_fixture_schema(), vec![1], vec![pos_del]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;

        let ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .expect("id col")
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        // Survivors are exactly {1, 3} — position 1 (id=2) deleted via the absolute-position vector.
        assert_eq!(ids, vec![1, 3]);
    }

    // -- ORC MoR equality deletes applied by VALUE (the SAME shared survival evaluator). --------------

    #[tokio::test]
    async fn orc_scan_applies_equality_deletes() {
        let tmp = TempDir::new().unwrap();
        let path = orc_fixture_on_disk(&tmp);
        // Equality-delete id value {2} by VALUE on field id 1.
        let del_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_del = write_eq_delete_file(&del_path, &[2]);

        let task = orc_task(&path, orc_fixture_schema(), vec![1], vec![eq_del]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;

        let ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .expect("id col")
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        // id=2 is removed by value; {1, 3} survive.
        assert_eq!(ids, vec![1, 3]);
    }

    // -- ORC FIELD-ID PROOF (load-bearing): read a Java-written ORC file with a RENAMED expected
    //    field (SAME field id 1, DIFFERENT name "renamed_id"). The value MUST land in the renamed
    //    column. A NAME-BASED reader would look up "id" in the file and either miss or wrongly
    //    resolve — this proves resolution is by `iceberg.id`, never by ORC column name.

    #[tokio::test]
    async fn orc_scan_resolves_by_field_id_not_name() {
        let tmp = TempDir::new().unwrap();
        let path = orc_fixture_on_disk(&tmp);
        // The file's field 1 is named "id"; the EXPECTED schema renames field-id 1 to "renamed_id".
        let renamed_schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "renamed_id", Type::Primitive(PrimitiveType::Long))
                        .into(),
                ])
                .build()
                .expect("build renamed schema"),
        );
        let task = orc_task(&path, renamed_schema, vec![1], vec![]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;

        // The output column is named "renamed_id" (the EXPECTED name, not the file name "id") and
        // carries the file's field-1 values — proving by-field-id resolution.
        assert!(
            batches[0].column_by_name("id").is_none(),
            "the file's ORC name 'id' must NOT appear; resolution is by field-id"
        );
        let renamed = batches[0]
            .column_by_name("renamed_id")
            .expect("renamed_id column (by field-id 1)");
        assert_eq!(renamed.as_primitive::<Int64Type>().values(), &[1, 2, 3]);
    }

    // -- MUTATION BAIT: a positional-delete keep-mask that did NOT drop the deleted rows would let
    //    id=20/id=40 survive. This pins the absolute-position membership test.
    #[tokio::test]
    async fn avro_positional_delete_mutation_bait() {
        let tmp = TempDir::new().unwrap();
        let schema = test_schema();
        let data_path = tmp.path().join("data.avro").to_string_lossy().to_string();
        write_avro_data_file(&data_path, &schema, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
        ]);
        // Delete only position 0 (id=10).
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let pos_del = write_pos_delete_file(&del_path, &data_path, &[0]);
        let task = avro_task(&data_path, schema, vec![1, 2], vec![pos_del]);
        let rows = rows_of(&run_scan(FileIO::new_with_fs(), task).await);
        // id=10 MUST be gone; 20 and 30 survive. A reader that ignored the delete vector would keep 10.
        assert!(
            !rows.iter().any(|(id, _)| *id == 10),
            "position-0 row must be deleted"
        );
        assert_eq!(rows, vec![
            (20, Some("b".to_string())),
            (30, Some("c".to_string())),
        ]);
    }

    // -- MUTATION BAIT (ORC): a positional-delete keep-mask that did NOT drop the deleted row would
    //    let id=1 (position 0) survive. This pins the absolute-position membership test on the ORC
    //    path (the SAME `DeleteVector`/survival-mask machinery the Avro/Parquet paths use).
    #[tokio::test]
    async fn orc_positional_delete_mutation_bait() {
        let tmp = TempDir::new().unwrap();
        let path = orc_fixture_on_disk(&tmp);
        // Delete only position 0 (id=1).
        let del_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        let pos_del = write_pos_delete_file(&del_path, &path, &[0]);
        let task = orc_task(&path, orc_fixture_schema(), vec![1], vec![pos_del]);
        let batches = run_scan(FileIO::new_with_fs(), task).await;
        let ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .expect("id col")
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        // id=1 MUST be gone; 2 and 3 survive. A reader ignoring the delete vector would keep 1.
        assert!(!ids.contains(&1), "position-0 row (id=1) must be deleted");
        assert_eq!(ids, vec![2, 3]);
    }
}

#[cfg(test)]
mod parquet_eq_keyset_mor_tests {
    //! Wave B: Parquet MoR path wires [`EqDeleteKeySet`] when key columns are projected. Routing
    //! (see `process_parquet_file_scan_task`): * keys ⊆ projected non-metadata field ids →
    //! post-decode keyset keep-mask (RowFilter residual is scan-predicate only); * otherwise →
    //! today's AND of eq-delete predicate into the Parquet RowFilter. Both routes must produce the
    //! same survivors (predicate oracle).

    use std::collections::HashMap;
    use std::fs::File;
    use std::ops::Not;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::eq_delete_key_fields_projected;
    use crate::arrow::equality_delete_set::EqDeleteKeySet;
    use crate::arrow::{ArrowReader, ArrowReaderBuilder};
    use crate::io::FileIO;
    use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
    use crate::spec::{
        DataContentType, DataFileFormat, Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type,
    };

    fn test_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("build test schema"),
        )
    }

    fn write_parquet_data_file(path: &str, rows: &[(i64, Option<&str>)]) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("data", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
        let data: Vec<Option<&str>> = rows.iter().map(|(_, d)| *d).collect();
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(data)) as ArrayRef,
        ])
        .expect("build data batch");
        let file = File::create(path).expect("create data file");
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                .expect("data writer");
        writer.write(&batch).expect("write data");
        writer.close().expect("close data");
    }

    fn write_eq_delete_file(path: &str, delete_ids: &[i64]) -> FileScanTaskDeleteFile {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(
            delete_ids.to_vec(),
        )) as ArrayRef])
        .expect("build eq-delete batch");
        let file = File::create(path).expect("create eq-delete file");
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                .expect("eq-delete writer");
        writer.write(&batch).expect("write eq-delete");
        writer.close().expect("close eq-delete");
        FileScanTaskDeleteFile {
            file_path: path.to_string(),
            file_size_in_bytes: std::fs::metadata(path).expect("stat").len(),
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: Some(vec![1]),
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }
    }

    fn parquet_task(
        path: &str,
        schema: SchemaRef,
        project_field_ids: Vec<i32>,
        deletes: Vec<FileScanTaskDeleteFile>,
    ) -> FileScanTask {
        FileScanTask {
            file_size_in_bytes: std::fs::metadata(path).expect("stat data").len(),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(path),
            data_file_format: DataFileFormat::Parquet,
            schema,
            project_field_ids: Arc::from(project_field_ids),

            predicate: None,
            deletes: Arc::from(deletes),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        }
    }

    async fn run_scan(task: FileScanTask) -> Vec<RecordBatch> {
        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        reader
            .read(tasks)
            .expect("build scan stream")
            .try_collect::<Vec<RecordBatch>>()
            .await
            .expect("collect scan batches")
    }

    fn surviving_ids(batches: &[RecordBatch]) -> Vec<i64> {
        let mut ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .expect("id column")
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Unit pin: keyset routing requires every key field id to be projected.
    #[test]
    fn test_eq_delete_key_fields_projected_gate() {
        let set =
            EqDeleteKeySet::try_build(vec![(1, "id".to_string(), PrimitiveType::Long)], vec![
                vec![Some(Datum::long(20))],
            ])
            .expect("long key is eligible");
        assert!(
            eq_delete_key_fields_projected(std::slice::from_ref(&set), &[1, 2]),
            "keys ⊆ projection → eligible"
        );
        assert!(
            eq_delete_key_fields_projected(std::slice::from_ref(&set), &[1]),
            "key alone is enough"
        );
        assert!(
            !eq_delete_key_fields_projected(std::slice::from_ref(&set), &[2]),
            "key missing from projection → not eligible (RowFilter fallback)"
        );
        assert!(
            !eq_delete_key_fields_projected(&[], &[1]),
            "empty sets are never eligible"
        );
    }

    /// When the eq-delete key is projected, the Parquet MoR path applies the keyset keep-mask
    /// post-decode. Survivors must match the predicate oracle (delete ids 20 and 40).
    #[tokio::test]
    async fn parquet_eq_keyset_path_when_keys_projected() {
        let tmp = TempDir::new().expect("tempdir");
        let schema = test_schema();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        write_parquet_data_file(&data_path, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
            (40, Some("d")),
            (50, Some("e")),
        ]);
        let del_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_del = write_eq_delete_file(&del_path, &[20, 40]);

        // Project both id (key) and data → keyset path eligible.
        let task = parquet_task(&data_path, schema, vec![1, 2], vec![eq_del]);
        let batches = run_scan(task).await;
        assert_eq!(
            surviving_ids(&batches),
            vec![10, 30, 50],
            "keyset MoR path must drop deleted ids 20 and 40"
        );
    }

    /// Mutation bait: a keyset path that forgot to apply deletes would keep 20/40.
    #[tokio::test]
    async fn parquet_eq_keyset_mutation_bait_drops_deleted_ids() {
        let tmp = TempDir::new().expect("tempdir");
        let schema = test_schema();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        write_parquet_data_file(&data_path, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
        ]);
        let del_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_del = write_eq_delete_file(&del_path, &[20]);
        let task = parquet_task(&data_path, schema, vec![1, 2], vec![eq_del]);
        let ids = surviving_ids(&run_scan(task).await);
        assert!(
            !ids.contains(&20),
            "id=20 MUST be deleted by the keyset/post-decode path"
        );
        assert_eq!(ids, vec![10, 30]);
    }

    /// When the eq-delete key is NOT projected, the reader falls back to the RowFilter path
    /// without error. Survivors still match the oracle (deletes applied via predicate pushdown).
    ///
    /// Note: projecting only `data` (field 2) means the output has no `id` column — we assert
    /// surviving *data* values instead.
    #[tokio::test]
    async fn parquet_eq_keyset_falls_back_when_keys_not_projected() {
        let tmp = TempDir::new().expect("tempdir");
        let schema = test_schema();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        write_parquet_data_file(&data_path, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
            (40, Some("d")),
            (50, Some("e")),
        ]);
        let del_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_del = write_eq_delete_file(&del_path, &[20, 40]);

        // Project ONLY data (field 2) — key field 1 is absent → RowFilter fallback.
        let task = parquet_task(&data_path, schema, vec![2], vec![eq_del]);
        let batches = run_scan(task).await;

        let mut data_vals: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                let col = b.column_by_name("data").expect("data column");
                (0..b.num_rows()).map(move |i| {
                    if col.is_null(i) {
                        String::new()
                    } else {
                        col.as_string::<i32>().value(i).to_string()
                    }
                })
            })
            .collect();
        data_vals.sort();
        assert_eq!(
            data_vals,
            vec!["a".to_string(), "c".to_string(), "e".to_string()],
            "fallback RowFilter path must still drop rows whose id was equality-deleted"
        );
        // No `id` column in the projection — proves we did not require the key in the output.
        assert!(
            batches.iter().all(|b| b.column_by_name("id").is_none()),
            "key column must remain unprojected"
        );
    }

    /// Direct unit pin of the shared keep-mask helper: set path matches predicate oracle on a
    /// non-null long-key batch (same contract as delete_filter H6 harness, routed through
    /// `ArrowReader::eq_delete_keep_mask`).
    #[test]
    fn test_eq_delete_keep_mask_set_matches_oracle() {
        use crate::arrow::record_batch_predicate::evaluate_predicate_to_mask;
        use crate::expr::{Bind, Predicate, Reference};

        let schema = test_schema();
        // Data batch with field-id metadata so both paths resolve column 1.
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int64, true).with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    "1".to_string(),
                )])),
            ])),
            vec![Arc::new(Int64Array::from(vec![Some(10i64), Some(20), Some(30)])) as ArrayRef],
        )
        .expect("batch");

        let delete_rows = vec![vec![Some(Datum::long(20))]];
        let set = EqDeleteKeySet::try_build(
            vec![(1, "id".to_string(), PrimitiveType::Long)],
            delete_rows.clone(),
        )
        .expect("set builds");

        // Predicate oracle: NOT(id = 20) survival → deleted mask [false, true, false].
        let survival = Reference::new("id")
            .equal_to(Datum::long(20))
            .not()
            .rewrite_not();
        // Fold into a single-file survival predicate shape (AlwaysTrue.and leaves).
        let survival: Predicate = survival;
        let bound = survival.bind(schema, false).expect("bind");
        let survives = evaluate_predicate_to_mask(&bound, &batch).expect("eval");
        let oracle_deleted: Vec<bool> = (0..survives.len())
            .map(|i| !(survives.is_valid(i) && survives.value(i)))
            .collect();

        let keep = ArrowReader::eq_delete_keep_mask(
            &batch,
            batch.num_rows(),
            Some(&bound),
            Some(std::slice::from_ref(&set)),
        )
        .expect("keep mask")
        .expect("mask present");
        let set_deleted: Vec<bool> = (0..keep.len()).map(|i| !keep.value(i)).collect();
        assert_eq!(
            set_deleted, oracle_deleted,
            "eq_delete_keep_mask set path must match predicate oracle"
        );
        assert_eq!(set_deleted, vec![false, true, false]);
        // Sanity: BooleanArray keep mask length matches the batch.
        assert_eq!(keep.len(), 3);
    }

    /// When keys are projected (keyset post-decode path), the scan residual
    /// must still be applied via RowFilter. Survivors = residual ∩ ¬eq-deleted.
    #[tokio::test]
    async fn parquet_eq_keyset_with_scan_residual() {
        use crate::expr::{Bind, Predicate, Reference};

        let tmp = TempDir::new().expect("tempdir");
        let schema = test_schema();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        // ids 10..50; residual will keep only data ∈ {a,b,c}; eq-delete removes 20.
        write_parquet_data_file(&data_path, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
            (40, Some("d")),
            (50, Some("e")),
        ]);
        let del_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_del = write_eq_delete_file(&del_path, &[20]);

        let residual = Reference::new("data")
            .equal_to(Datum::string("a"))
            .or(Reference::new("data").equal_to(Datum::string("b")))
            .or(Reference::new("data").equal_to(Datum::string("c")));
        let residual: Predicate = residual;
        let bound = residual.bind(schema.clone(), true).expect("bind residual");

        let mut task = parquet_task(&data_path, schema, vec![1, 2], vec![eq_del]);
        task.predicate = Some(Arc::new(bound));

        let batches = run_scan(task).await;
        // Residual keeps a,b,c (ids 10,20,30); eq-delete drops 20 → 10,30.
        assert_eq!(
            surviving_ids(&batches),
            vec![10, 30],
            "keyset MoR path must AND scan residual with eq-deletes (C1-Q-001)"
        );
    }

    /// Nullable key column with a NULL cell forces predicate fallback
    /// under the keyset-eligible path. A value-delete must NOT drop the NULL-key row
    /// (Java nulls-first / unit A2).
    #[tokio::test]
    async fn parquet_eq_keyset_null_key_falls_back_to_predicate() {
        let tmp = TempDir::new().expect("tempdir");
        // Optional id so NULL keys are legal in the data file.
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("schema"),
        );
        let data_path = tmp
            .path()
            .join("data-null-key.parquet")
            .to_string_lossy()
            .to_string();
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("data", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int64Array::from(vec![Some(10i64), None, Some(20)])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("null-key"),
                Some("b"),
            ])) as ArrayRef,
        ])
        .expect("batch");
        {
            let file = File::create(&data_path).expect("create");
            let mut writer = ArrowWriter::try_new(
                file,
                arrow_schema,
                Some(WriterProperties::builder().build()),
            )
            .expect("writer");
            writer.write(&batch).expect("write");
            writer.close().expect("close");
        }
        let del_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        // Value-delete id=20 only — must not delete the NULL-key row.
        let eq_del = write_eq_delete_file(&del_path, &[20]);
        let task = parquet_task(&data_path, schema, vec![1, 2], vec![eq_del]);
        let batches = run_scan(task).await;

        let mut data_vals: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                let col = b.column_by_name("data").expect("data");
                (0..b.num_rows()).map(move |i| {
                    if col.is_null(i) {
                        String::new()
                    } else {
                        col.as_string::<i32>().value(i).to_string()
                    }
                })
            })
            .collect();
        data_vals.sort();
        assert_eq!(
            data_vals,
            vec!["a".to_string(), "null-key".to_string()],
            "NULL-key row must survive a value eq-delete; id=20 must be dropped (C1-Q-002)"
        );
    }

    /// Composite equality key (id + data) on the Parquet keyset path.
    /// Pins multi-column tuple membership — a sabotage that only matches the first key column
    /// would over-delete.
    #[tokio::test]
    async fn parquet_eq_keyset_composite_key() {
        let tmp = TempDir::new().expect("tempdir");
        let schema = test_schema();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        write_parquet_data_file(&data_path, &[
            (10, Some("a")),
            (20, Some("b")),
            (20, Some("c")), // same id, different data — must SURVIVE if delete is (20,b) only
            (30, Some("d")),
        ]);
        let del_path = tmp
            .path()
            .join("eq-del-composite.parquet")
            .to_string_lossy()
            .to_string();
        // Composite eq-delete: only (id=20, data=b).
        let del_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("data", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let del_batch = RecordBatch::try_new(del_schema.clone(), vec![
            Arc::new(Int64Array::from(vec![20i64])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("b")])) as ArrayRef,
        ])
        .expect("del batch");
        {
            let file = File::create(&del_path).expect("create");
            let mut writer =
                ArrowWriter::try_new(file, del_schema, Some(WriterProperties::builder().build()))
                    .expect("writer");
            writer.write(&del_batch).expect("write");
            writer.close().expect("close");
        }
        let eq_del = FileScanTaskDeleteFile {
            file_path: del_path.clone(),
            file_size_in_bytes: std::fs::metadata(&del_path).expect("stat").len(),
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: Some(vec![1, 2]),
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };
        let task = parquet_task(&data_path, schema, vec![1, 2], vec![eq_del]);
        let batches = run_scan(task).await;
        // Expect (10,a), (20,c), (30,d) — not (20,b).
        let mut pairs: Vec<(i64, String)> = batches
            .iter()
            .flat_map(|b| {
                let ids = b
                    .column_by_name("id")
                    .expect("id")
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec();
                let data = b.column_by_name("data").expect("data");
                ids.into_iter().enumerate().map(move |(i, id)| {
                    let s = if data.is_null(i) {
                        String::new()
                    } else {
                        data.as_string::<i32>().value(i).to_string()
                    };
                    (id, s)
                })
            })
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                (10, "a".to_string()),
                (20, "c".to_string()),
                (30, "d".to_string())
            ],
            "composite keyset must delete only the matching (id,data) tuple (C2-Q-001)"
        );
    }

    /// Positional RowSelection + eq keyset post-decode on one Parquet task.
    #[tokio::test]
    async fn parquet_eq_keyset_with_positional_deletes() {
        let tmp = TempDir::new().expect("tempdir");
        let schema = test_schema();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        // positions: 0→10, 1→20, 2→30, 3→40
        write_parquet_data_file(&data_path, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
            (40, Some("d")),
        ]);
        let eq_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_del = write_eq_delete_file(&eq_path, &[20]); // drops id 20 (pos 1)

        let pos_path = tmp
            .path()
            .join("pos-del.parquet")
            .to_string_lossy()
            .to_string();
        // Pos-delete physical position 3 (id 40).
        let pos_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2147483546".to_string(),
            )])),
            Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2147483545".to_string(),
            )])),
        ]));
        let pos_batch = RecordBatch::try_new(pos_schema.clone(), vec![
            Arc::new(StringArray::from(vec![data_path.as_str()])) as ArrayRef,
            Arc::new(Int64Array::from(vec![3i64])) as ArrayRef,
        ])
        .expect("pos batch");
        {
            let file = File::create(&pos_path).expect("create");
            let mut writer =
                ArrowWriter::try_new(file, pos_schema, Some(WriterProperties::builder().build()))
                    .expect("writer");
            writer.write(&pos_batch).expect("write");
            writer.close().expect("close");
        }
        let pos_del = FileScanTaskDeleteFile {
            file_path: pos_path.clone(),
            file_size_in_bytes: std::fs::metadata(&pos_path).expect("stat").len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let task = parquet_task(&data_path, schema, vec![1, 2], vec![eq_del, pos_del]);
        let ids = surviving_ids(&run_scan(task).await);
        assert_eq!(
            ids,
            vec![10, 30],
            "pos must drop id=40 and eq-keyset must drop id=20 (C2-Q-002)"
        );
    }

    /// Keyset path when the projection is *only* the key column (no
    /// non-key data columns). Gate is keys ⊆ projection, not projection == full schema.
    #[tokio::test]
    async fn parquet_eq_keyset_project_key_only() {
        let tmp = TempDir::new().expect("tempdir");
        let schema = test_schema();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        write_parquet_data_file(&data_path, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
        ]);
        let del_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_del = write_eq_delete_file(&del_path, &[20]);
        // Project only field 1 (the key) — keyset eligible, output has no `data` column.
        let task = parquet_task(&data_path, schema, vec![1], vec![eq_del]);
        let ids = surviving_ids(&run_scan(task).await);
        assert_eq!(
            ids,
            vec![10, 30],
            "key-only projection keyset path (C3-Q-001)"
        );
    }

    /// Two eq-delete files OR-combined under the keyset path
    /// (a row matching EITHER file is deleted).
    #[tokio::test]
    async fn parquet_eq_keyset_two_delete_files_or() {
        let tmp = TempDir::new().expect("tempdir");
        let schema = test_schema();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        write_parquet_data_file(&data_path, &[
            (10, Some("a")),
            (20, Some("b")),
            (30, Some("c")),
            (40, Some("d")),
        ]);
        let d1 = tmp.path().join("eq1.parquet").to_string_lossy().to_string();
        let d2 = tmp.path().join("eq2.parquet").to_string_lossy().to_string();
        let eq1 = write_eq_delete_file(&d1, &[20]);
        let eq2 = write_eq_delete_file(&d2, &[40]);
        let task = parquet_task(&data_path, schema, vec![1, 2], vec![eq1, eq2]);
        assert_eq!(
            surviving_ids(&run_scan(task).await),
            vec![10, 30],
            "two eq-delete keysets must OR (drop 20 and 40) (C3-Q-002)"
        );
    }

    /// Keyset path that deletes every row yields empty (or no-row) output —
    /// must not error and must not resurrect rows.
    #[tokio::test]
    async fn parquet_eq_keyset_deletes_all_rows() {
        let tmp = TempDir::new().expect("tempdir");
        let schema = test_schema();
        let data_path = tmp
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .to_string();
        write_parquet_data_file(&data_path, &[(10, Some("a")), (20, Some("b"))]);
        let del_path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();
        let eq_del = write_eq_delete_file(&del_path, &[10, 20]);
        let task = parquet_task(&data_path, schema, vec![1, 2], vec![eq_del]);
        let batches = run_scan(task).await;
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total, 0,
            "all-deleted keyset path must leave zero rows (C4-Q-002)"
        );
    }
}
