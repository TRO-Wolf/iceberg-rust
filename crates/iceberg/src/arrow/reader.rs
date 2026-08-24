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

    /// Dispatch one [`FileScanTask`] on its [`data_file_format`](FileScanTask::data_file_format),
    /// porting Java's `FileFormat`-keyed reader selection (`GenericReader`/`BaseReader` pick the
    /// `parquet`/`avro`/`orc` data reader from `task.file().format()`):
    ///
    /// * [`Parquet`](DataFileFormat::Parquet) → [`Self::process_parquet_file_scan_task`] (the
    ///   pushdown read path — row-group skip, `RowFilter`/`RowSelection` deletes, byte-range split),
    ///   byte-for-byte unchanged.
    /// * [`Avro`](DataFileFormat::Avro) → [`Self::process_avro_file_scan_task`]: the Avro data file
    ///   is MATERIALIZED via the [`crate::arrow::avro_reader`] core (Java `PlannedDataReader`), then
    ///   schema-evolved + delete-filtered post-hoc (Avro cannot push down).
    /// * [`Orc`](DataFileFormat::Orc) → [`Self::process_orc_file_scan_task`]: the ORC data file is
    ///   MATERIALIZED via the [`crate::arrow::orc_reader`] core (Java `GenericOrcReader`), then
    ///   schema-evolved + delete-filtered post-hoc exactly as the Avro path is (ORC has footer
    ///   structure but this reader materializes whole-file; deletes are applied post-decode — GAP
    ///   row R118, READ-only).
    /// * [`Puffin`](DataFileFormat::Puffin) → a clean `FeatureUnsupported` error: a Puffin file is a
    ///   stats/deletion-vector sidecar, never a scannable DATA file, so it must never reach here.
    async fn process_file_scan_task(
        task: FileScanTask,
        batch_size: Option<usize>,
        file_io: FileIO,
        delete_file_loader: CachingDeleteFileLoader,
        row_group_filtering_enabled: bool,
        row_selection_enabled: bool,
        parquet_read_options: ParquetReadOptions,
    ) -> Result<ArrowRecordBatchStream> {
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

        // Open the Parquet file once, loading its metadata
        let (parquet_file_reader, arrow_metadata) = Self::open_parquet_file(
            &task.data_file_path,
            &file_io,
            task.file_size_in_bytes,
            parquet_read_options,
        )
        .await?;

        // Check if Parquet file has embedded field IDs
        // Corresponds to Java's ParquetSchemaUtil.hasIds()
        // Reference: parquet/src/main/java/org/apache/iceberg/parquet/ParquetSchemaUtil.java:118
        let missing_field_ids = arrow_metadata
            .schema()
            .fields()
            .iter()
            .next()
            .is_some_and(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none());

        // Three-branch schema resolution strategy matching Java's ReadConf constructor
        //
        // Per Iceberg spec Column Projection rules:
        // "Columns in Iceberg data files are selected by field id. The table schema's column
        //  names and order may change after a data file is written, and projection must be done
        //  using field ids."
        // https://iceberg.apache.org/spec/#column-projection
        //
        // When Parquet files lack field IDs (e.g., Hive/Spark migrations via add_files),
        // we must assign field IDs BEFORE reading data to enable correct projection.
        //
        // Java's ReadConf determines field ID strategy:
        // - Branch 1: hasIds(fileSchema) → trust embedded field IDs, use pruneColumns()
        // - Branch 2: nameMapping present → applyNameMapping(), then pruneColumns()
        // - Branch 3: fallback → addFallbackIds(), then pruneColumnsFallback()
        let arrow_metadata = if missing_field_ids {
            // Parquet file lacks field IDs - must assign them before reading
            let arrow_schema = if let Some(name_mapping) = &task.name_mapping {
                // Branch 2: Apply name mapping to assign correct Iceberg field IDs
                // Per spec rule #2: "Use schema.name-mapping.default metadata to map field id
                // to columns without field id"
                // Corresponds to Java's ParquetSchemaUtil.applyNameMapping()
                apply_name_mapping_to_arrow_schema(
                    Arc::clone(arrow_metadata.schema()),
                    name_mapping,
                )?
            } else {
                // Branch 3: No name mapping - use position-based fallback IDs
                // Corresponds to Java's ParquetSchemaUtil.addFallbackIds()
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
            // Branch 1: File has embedded field IDs - trust them
            arrow_metadata
        };

        // Whether the final projection must fall back to POSITION-based matching. This is true
        // ONLY when the file lacked embedded field ids AND no name mapping supplied them (Branch 3
        // above, `ParquetSchemaUtil.addFallbackIds` → `pruneColumnsFallback`). When a name mapping
        // WAS applied (Branch 2), the schema now carries the correct Iceberg field ids stamped by
        // `apply_name_mapping_to_arrow_schema`, so projection must be FIELD-ID-based (Java
        // `applyNameMapping` → `pruneColumns`) — a positional projection here would ignore the
        // mapping and read columns by physical position, the wrong-column class this whole path
        // exists to prevent.
        let use_position_fallback = missing_field_ids && task.name_mapping.is_none();

        // Coerce INT96 timestamp columns to the resolution specified by the Iceberg schema.
        // This must happen before building the stream reader to avoid i64 overflow in arrow-rs.
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

        // Build the stream reader, reusing the already-opened file reader
        let mut record_batch_stream_builder =
            ParquetRecordBatchStreamBuilder::new_with_metadata(parquet_file_reader, arrow_metadata);

        // Filter out metadata fields for Parquet projection (they don't exist in files).
        //
        // The V3 ROW-LINEAGE pair is the exception and must NOT be filtered: unlike `_file` or
        // `_pos`, `_row_id` / `_last_updated_sequence_number` CAN be physically present in a data
        // file — a lineage-preserving rewrite carries the original ids forward — and Java reads
        // the stored value in preference to the computed one (`ValueReaders$RowIdReader.read`
        // offsets 34-39). Filtering them here would leave the stored column undecoded, so the
        // transformer would silently fall back to `first_row_id + pos` and report plausible but
        // WRONG identities for exactly the rows whose identity was being preserved.
        // Requesting an id the file does not carry is harmless: the mask builder drops unmatched
        // ids and the transformer then takes its computed/constant arm.
        let project_field_ids_without_metadata: Vec<i32> = task
            .project_field_ids
            .iter()
            .filter(|&&id| !is_metadata_field(id) || is_row_lineage_field(id))
            .copied()
            .collect();

        // Create projection mask based on field IDs
        // - If file has embedded IDs: field-ID-based projection (missing_field_ids=false)
        // - If name mapping applied: field-ID-based projection (missing_field_ids=true but IDs now match)
        // - If fallback IDs: position-based projection (missing_field_ids=true)
        let projection_mask = Self::get_arrow_projection_mask(
            &project_field_ids_without_metadata,
            &task.schema,
            record_batch_stream_builder.parquet_schema(),
            record_batch_stream_builder.schema(),
            use_position_fallback, // position-based (true) only for id-less files with NO name mapping
        )?;

        record_batch_stream_builder =
            record_batch_stream_builder.with_projection(projection_mask.clone());

        // `_pos` (row position) requested: a row-identity scan needs each row's TRUE physical
        // ordinal in the data file (to build position deletes). Parquet applies positional deletes
        // and the scan predicate via `RowSelection`, which SKIPS rows at the decode layer, making
        // the surviving rows' physical positions unrecoverable. So when `_pos` is projected we
        // decode the file in order with NO row-skipping (no RowFilter / RowSelection / row-group
        // pruning). FK5 streaming half: stream batches through transform + post-decode survival
        // (pos-deletes / residual / eq-deletes) instead of whole-file `try_collect` — memory is
        // O(batch), not O(file). The transformer assigns `_pos` = 0-based file ordinal. Selection-
        // aware ordinal pushdown is STOP-gated (see `task/fk5-pos-projection-ledger.md`): a
        // `RowFilter` does not expose physical ordinals of undelivered rows. Pushdown is unaffected
        // for scans that do not request `_pos`.
        // `_row_id` shares `_pos`'s requirement: Java's `RowIdReader` falls back to
        // `firstRowId + pos`, so a row id computed for a row whose physical ordinal was lost to
        // row-skipping is WRONG — and wrong silently, as a plausible id belonging to another row.
        // The guard is unconditional on projection rather than conditional on the file carrying a
        // stored `_row_id` column, because whether the fallback arm is needed is only knowable
        // per NULL row AFTER decoding, by which point the ordinals are already gone.
        let needs_physical_ordinals = task.project_field_ids().contains(&RESERVED_FIELD_ID_POS)
            || task.project_field_ids().contains(&RESERVED_FIELD_ID_ROW_ID);
        if needs_physical_ordinals {
            // Review rider (2026-08-03): this path decodes the WHOLE file in physical order with
            // ordinals from 0 (see `stream_pos_projection_scan_task`); a RANGED split task here
            // would re-emit the full file per split — duplicate rows with wrong `_pos`, which
            // corrupts written position deletes. Whole-file tasks carry `start == 0` with
            // `length == 0` (legacy sentinel) or `length == file_size_in_bytes`; anything else
            // is a `plan_tasks`/split product and is rejected loud. (The `to_arrow` within-file
            // expand already suppresses itself under `_pos`; this guards the public
            // `PartitionWork` / direct-reader seam.)
            let whole_file =
                task.start == 0 && (task.length == 0 || task.length == task.file_size_in_bytes);
            if !whole_file {
                // Name the column that actually forced this path, so a `_row_id` scan does not
                // get told to "drop `_pos`" — a projection it never asked for.
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

        // RecordBatchTransformer performs any transformations required on the RecordBatches
        // that come back from the file, such as type promotion, default column insertion,
        // column re-ordering, partition constants, and virtual field addition (like _file)
        let mut record_batch_transformer_builder =
            RecordBatchTransformerBuilder::new(task.schema_ref(), task.project_field_ids())
                .with_row_lineage(task.first_row_id, task.file_sequence_number);

        // Add the _file metadata column if it's in the projected fields
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

        // Equality-delete routing for the Parquet pushdown path (Wave B):
        //
        // * Prefer the O(R) [`EqDeleteKeySet`] post-decode fast path when
        //   (a) every eq-delete file for this task is type-eligible and shares one key schema
        //       (`collect_equality_delete_keysets` → `Some(sets)`), AND
        //   (b) every key field id is present in the projected non-metadata columns (so the
        //       transformed batch can resolve keys by `PARQUET_FIELD_ID_META_KEY`).
        //   In that case the Parquet `RowFilter` residual is **scan-predicate only** — eq-deletes
        //   are applied AFTER `RecordBatchTransformer` via the same keep-mask logic as the
        //   whole-file (`survival_mask`) eq branch (set first; NULL-key batches fall back to the
        //   bound eq-delete predicate).
        // * Otherwise (no keysets / keys missing from projection / ineligible types): keep today's
        //   correctness-first path and AND the eq-delete predicate into the RowFilter residual so
        //   Parquet still filters deleted rows at decode time (the RowFilter can still read key
        //   columns that the data projection omitted).
        let eq_delete_sets = delete_filter.collect_equality_delete_keysets(&task).await;
        let delete_predicate = delete_filter.build_equality_delete_predicate(&task).await?;
        let keyset_post_decode = eq_delete_sets.as_ref().is_some_and(|sets| {
            !sets.is_empty()
                && eq_delete_key_fields_projected(sets, &project_field_ids_without_metadata)
        });

        // Residual pushed into RowFilter / RG skip / page selection: scan predicate always; AND
        // eq-delete predicate only when the post-decode keyset path is NOT taken.
        // Normalize to owned BoundPredicate: task residual is Arc-shared, delete predicate is not.
        //
        // Owned state for the post-decode keyset apply (only when the routing above selected it).
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

        // There are three possible sources for potential lists of selected RowGroup indices,
        // and two for `RowSelection`s.
        // Selected RowGroup index lists can come from three sources:
        //   * When task.start and task.length specify a byte range (file splitting);
        //   * When there are equality delete files that are applicable;
        //   * When there is a scan predicate and row_group_filtering_enabled = true.
        // `RowSelection`s can be created in either or both of the following cases:
        //   * When there are positional delete files that are applicable;
        //   * When there is a scan predicate and row_selection_enabled = true
        // Note that row group filtering from predicates only happens when
        // there is a scan predicate AND row_group_filtering_enabled = true,
        // but we perform row selection filtering if there are applicable
        // equality delete files OR (there is a scan predicate AND row_selection_enabled),
        // since the only implemented method of applying positional deletes is
        // by using a `RowSelection`.
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

                // Merge predicate-based filtering with byte range filtering (if present)
                // by taking the intersection of both filters
                selected_row_group_indices = match selected_row_group_indices {
                    Some(byte_range_filtered) => {
                        // Keep only row groups that are in both filters
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
            // Frozen `Arc<DeleteVector>` — no mutex; apply/row-selection is lock-free on the bitmap.
            let delete_row_selection = Self::build_deletes_row_selection(
                record_batch_stream_builder.metadata().row_groups(),
                &selected_row_group_indices,
                positional_delete_indexes.as_ref(),
            )?;

            // merge the row selection from the delete files with the row selection
            // from the filter predicate, if there is one from the filter predicate
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

        // Build the batch stream and send all the RecordBatches that it generates
        // to the requester. When `keyset_post_decode` is set, eq-deletes are applied
        // here (post-transform) rather than via the Parquet RowFilter residual above.
        let record_batch_stream =
            record_batch_stream_builder
                .build()?
                .map(move |batch| match batch {
                    Ok(batch) => {
                        // Process the record batch (type promotion, column reordering, virtual fields, etc.)
                        let transformed = record_batch_transformer.process_record_batch(batch)?;
                        if post_decode_eq_sets.is_none() && post_decode_eq_predicate.is_none() {
                            return Ok(transformed);
                        }
                        // Same keep-mask routing as `survival_mask`'s eq branch: prefer keysets;
                        // fall back to the bound eq-delete predicate on NULL-key batches.
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

    /// Fail closed on a task carrying a real byte sub-window for a format whose reader
    /// materializes WHOLE files.
    ///
    /// [`Self::process_avro_file_scan_task`] and [`Self::process_orc_file_scan_task`] never read
    /// `task.start` / `task.length` — the Avro OCF and ORC readers decode the entire file. So a
    /// ranged sub-task would re-emit every row of the file, and an N-way split would return N
    /// copies with no error at all (measured: a 500-row Avro OCF split four ways returned 2,000
    /// rows). That is the same silent-duplication class the Parquet midpoint row-group selection
    /// exists to eliminate, and it must not be reachable by any route.
    ///
    /// [`FileScanTask::split`] already declines to split these formats
    /// (`scan::task::reader_honors_byte_range`), so the planner never produces such a task; this
    /// guard covers the public `PartitionWork` / direct-reader seams, exactly as the `_pos` guard
    /// in [`Self::process_parquet_file_scan_task`] does.
    ///
    /// Whole-file tasks carry `start == 0` with either `length == 0` (the legacy sentinel) or
    /// `length == file_size_in_bytes`; anything else is a split product and is rejected loud.
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

    /// Read one **Avro** data-file scan task into an [`ArrowRecordBatchStream`].
    ///
    /// Avro has no footer metadata, statistics, or row-group structure, so — unlike the Parquet
    /// path — there is NO pushdown: the file is MATERIALIZED in full (Java's `PlannedDataReader`
    /// is whole-block) and every filter is applied POST-materialization. The pipeline mirrors the
    /// Parquet path's semantics rung for rung, only with the mechanism moved after decode:
    ///
    /// 1. **Read** via the [`crate::arrow::avro_reader`] core ([`read_avro_data_file`]) against the
    ///    *projected* Iceberg schema (`task.schema` restricted to the file-present projected field
    ///    ids — metadata columns like `_file` are excluded, exactly as the Parquet path strips them
    ///    from its projection mask). Resolution is by field id, with U1's full schema-evolution
    ///    (projection / skip / missing-default) machinery.
    /// 2. **Transform** each batch through the SAME [`RecordBatchTransformer`] the Parquet path
    ///    feeds (type promotion, column reorder, `_file` + identity-partition constants, virtual
    ///    fields) — built identically (`with_constant(_file)` + `with_partition`).
    /// 3. **Apply merge-on-read deletes** on the materialized batch, matching the Parquet delete
    ///    semantics exactly: a POSITIONAL delete drops rows whose absolute file position (tracked
    ///    across batches) is in the [`DeleteVector`]; an EQUALITY delete keeps rows where the bound
    ///    delete predicate is TRUE — both via [`evaluate_predicate_to_mask`] /
    ///    [`filter_record_batch`], the same kernels the Parquet `RowFilter`/`RowSelection` use. The
    ///    scan `task.predicate` residual is ALSO applied here (the Parquet path pushes it into the
    ///    `RowFilter`); on Avro it is AND-ed into the per-batch mask.
    async fn process_avro_file_scan_task(
        task: FileScanTask,
        batch_size: Option<usize>,
        file_io: FileIO,
        delete_file_loader: CachingDeleteFileLoader,
    ) -> Result<ArrowRecordBatchStream> {
        Self::reject_ranged_whole_file_task(&task, "AVRO")?;

        // Kick off delete loading concurrently with the file read (as the Parquet path does).
        let delete_filter_rx =
            delete_file_loader.load_deletes(&task.deletes, Arc::clone(&task.schema));

        // The projected Iceberg schema the Avro reader resolves against: the scan schema restricted
        // to the projected field ids that are NOT reserved metadata columns (those don't exist in
        // the data file — the Parquet path strips them from its projection mask the same way; the
        // transformer re-adds `_file` / partition constants afterwards).
        let expected = Self::build_expected_schema(&task)?;

        // Avro decode is whole-block; the U1 reader requires a positive batch size. Fall back to the
        // arrow-rs default (1024) when the scan left it unset, matching the Parquet reader's default.
        let avro_batch_size = batch_size.unwrap_or(1024).max(1);
        let input_file = file_io.new_input(&task.data_file_path)?;
        let batches = read_avro_data_file(&input_file, expected, avro_batch_size).await?;

        // Everything after the format-specific decode is identical to the ORC path: build the SAME
        // RecordBatchTransformer the Parquet path feeds, resolve the deletes, and apply merge-on-read
        // deletes + the scan residual post-materialization.
        Self::finish_whole_file_scan_task(task, batches, delete_filter_rx).await
    }

    /// Read one **ORC** data-file scan task into an [`ArrowRecordBatchStream`].
    ///
    /// Structurally identical to [`Self::process_avro_file_scan_task`]: the ONLY difference is the
    /// step-(1) materialization, which calls the U1 ORC reader ([`read_orc_data_file`], Java
    /// `GenericOrcReader`) instead of the Avro reader. Everything after the decode — the projected
    /// expected schema, the SAME [`RecordBatchTransformer`] the Parquet / Avro paths feed (`_file` +
    /// identity-partition constants, schema evolution, reorder), and the merge-on-read delete
    /// machinery (positional via [`DeleteVector`] membership over the absolute row position, plus the
    /// equality and scan-residual masks) — is FORMAT-AGNOSTIC and lives in the shared
    /// [`Self::finish_whole_file_scan_task`] tail. ORC is materialized whole-file (this reader does
    /// not push predicates into the ORC stripe metadata), so, like Avro, every filter is applied
    /// POST-materialization.
    async fn process_orc_file_scan_task(
        task: FileScanTask,
        batch_size: Option<usize>,
        file_io: FileIO,
        delete_file_loader: CachingDeleteFileLoader,
    ) -> Result<ArrowRecordBatchStream> {
        Self::reject_ranged_whole_file_task(&task, "ORC")?;

        // Kick off delete loading concurrently with the file read (as the Parquet path does).
        let delete_filter_rx =
            delete_file_loader.load_deletes(&task.deletes, Arc::clone(&task.schema));

        // The projected Iceberg schema the ORC reader resolves against: the scan schema restricted to
        // the projected file-present (non-reserved-metadata) field ids — identical to the Avro path.
        let expected = Self::build_expected_schema(&task)?;

        // ORC decode is whole-file; the U1 reader requires a positive batch size. Fall back to the
        // arrow-rs default (1024) when the scan left it unset, matching the Parquet/Avro readers.
        let orc_batch_size = batch_size.unwrap_or(1024).max(1);
        let input_file = file_io.new_input(&task.data_file_path)?;
        let batches = read_orc_data_file(&input_file, expected, orc_batch_size).await?;

        // Identical format-agnostic tail to the Avro path (see [`Self::finish_whole_file_scan_task`]).
        Self::finish_whole_file_scan_task(task, batches, delete_filter_rx).await
    }

    /// Parquet path when `_pos` is projected (FK5 streaming half): decode in physical order with
    /// **no** `RowFilter` / `RowSelection` / RG prune, but stream batches through
    /// [`Self::apply_pos_aware_batch`] instead of whole-file `try_collect`. Memory is O(batch).
    /// Selection-aware ordinal pushdown remains STOP-gated — see ledger.
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

    /// The format-agnostic tail shared by [`Self::process_avro_file_scan_task`] and
    /// [`Self::process_orc_file_scan_task`]: everything after a whole-file reader has MATERIALIZED
    /// `batches` (the decoded data columns, by field id, in projection order). Given the in-flight
    /// `delete_filter_rx` (delete loading kicked off concurrently with the read), it:
    ///
    /// 1. builds the SAME [`RecordBatchTransformer`] the Parquet path feeds (type promotion, column
    ///    reorder, `_file` + identity-partition constants, virtual fields);
    /// 2. resolves the deletes and AND-s the equality-delete predicate with the scan `task.predicate`
    ///    residual into one per-batch survival predicate (the single predicate the Parquet path forms
    ///    before pushing it into the `RowFilter`);
    /// 3. applies merge-on-read deletes post-materialization — a POSITIONAL delete drops rows whose
    ///    absolute file position (tracked across batches) is in the [`DeleteVector`]; an EQUALITY
    ///    delete + residual keeps rows the predicate proves TRUE — via the same
    ///    [`evaluate_predicate_to_mask`] / [`filter_record_batch`] kernels the Parquet
    ///    `RowFilter`/`RowSelection` use.
    ///
    /// Per-batch apply is shared with the Parquet `_pos` streaming path
    /// ([`Self::apply_pos_aware_batch`]). Avro/ORC still materialize decode first (format limit);
    /// the eager output `Vec` here is fine because `batches` is already fully in memory.
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

        // File already fully decoded — eager loop is fine; counters stay simplest on owned batches.
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
        // Equality-delete fast path: if EVERY eq-delete file for this task is type-eligible and they
        // share one key schema, the deletes can be applied via hashed set membership (O(R)) instead
        // of the E-leaf predicate tree (O(E·R)). The set is used per-batch only when it is safe for
        // that batch (no NULL in a key column — see `EqDeleteKeySet::delete_mask`); otherwise the
        // eq-delete predicate is the fallback. The scan residual (`task.predicate`) and the eq-delete
        // predicate are kept available for the predicate path; the set only accelerates the common
        // case. `eq_delete_predicate` is `None` when the task has no eq-deletes.
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
    /// **full pre-filter** row count. Shared by Parquet `_pos` streaming and Avro/ORC whole-file.
    ///
    /// Correctness: `absolute_pos` / transformer `next_row_position` must track the same physical
    /// file ordinal base. Skipping rows at decode (RowSelection/RowFilter) would desync them —
    /// callers that project `_pos` must not enable decode-layer row skips.
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
        // Dual-counter invariant: `absolute_pos` (pos-delete base) and the transformer's
        // `next_row_position` (feeds `_pos` values) must stay aligned. When `_pos` is projected,
        // the first surviving ordinal in the transformed batch must equal `batch_base`. A
        // desync here corrupts WRITTEN position deletes (MERGE identity).
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
        // Advance by FULL batch before any mask filter so the next batch's physical ordinals
        // continue correctly (matches RecordBatchTransformer::next_row_position advance).
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

    /// The projected Iceberg [`Schema`] a whole-file (Avro / ORC) data reader resolves the file
    /// against: `task.schema` restricted to the projected field ids that exist in the data file
    /// (reserved metadata columns like `_file` / `_pos` are excluded — they are supplied as constants
    /// by the transformer / the positional-delete path, never read from the file). Field order
    /// follows the projection order. Format-agnostic: both the Avro and ORC paths share it.
    fn build_expected_schema(task: &FileScanTask) -> Result<Arc<Schema>> {
        let mut fields = Vec::new();
        for &field_id in task.project_field_ids() {
            if is_metadata_field(field_id) {
                continue;
            }
            let field = task.schema.field_by_id(field_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Projected field id {field_id} is not present in the scan schema for data \
                         file '{}'",
                        task.data_file_path
                    ),
                )
            })?;
            fields.push(field.clone());
        }
        let schema = Schema::builder()
            .with_schema_id(task.schema.schema_id())
            .with_fields(fields)
            .build()?;
        Ok(Arc::new(schema))
    }

    /// Build the per-row survival mask for a transformed whole-file (Avro / ORC) batch, combining
    /// positional deletes (rows whose absolute file position `[batch_base, batch_base + num_rows)`
    /// falls in `positional_deletes`), the scan `residual_predicate` (`task.predicate`), and the
    /// equality deletes — applied via the O(R) `eq_delete_sets` fast path when present and safe for
    /// the batch, else via `eq_delete_predicate`. Returns `None` when nothing applies (the caller
    /// emits the batch unchanged), else a [`BooleanArray`] where `true` ⇒ keep the row.
    /// Format-agnostic: shared by both readers.
    ///
    /// Equality-delete routing: `eq_delete_sets`, when `Some`, is the hashed key sets for the task's
    /// eq-delete files (all type-eligible, shared key schema). A row is dropped iff it matches ANY
    /// set's delete tuple — byte-identical to the eq-delete predicate for batches with no NULL in a
    /// key column (proven in `delete_filter.rs`'s harness). If ANY set reports a key-column NULL in
    /// this batch (`delete_mask` → `None`), the WHOLE batch's eq-deletes fall back to
    /// `eq_delete_predicate` (the proven 3VL-correct path). When `eq_delete_sets` is `None`, the
    /// eq-deletes are always applied via `eq_delete_predicate`.
    fn survival_mask(
        batch: &RecordBatch,
        num_rows: usize,
        batch_base: u64,
        positional_deletes: Option<&Arc<DeleteVector>>,
        residual_predicate: Option<&BoundPredicate>,
        eq_delete_predicate: Option<&BoundPredicate>,
        eq_delete_sets: Option<&[EqDeleteKeySet]>,
    ) -> Result<Option<BooleanArray>> {
        // Positional deletes → a keep-mask of `!deleted` over this batch's absolute position window.
        // The memoized vector is frozen (`Arc<DeleteVector>`); no mutex on the apply path.
        let positional_mask: Option<BooleanArray> = match positional_deletes {
            Some(deletes) => {
                if deletes.is_empty() {
                    None
                } else {
                    // Range-walk the delete window — byte-identical to the per-row `!contains` probe,
                    // O(D_window) instead of O(num_rows). See `positional_delete_keep_mask`.
                    Some(positional_delete_keep_mask(
                        deletes.as_ref(),
                        batch_base,
                        num_rows,
                    ))
                }
            }
            None => None,
        };

        // Helper: a keep-mask from a bound predicate (true ⇒ row survives). The mask is already
        // two-valued under Java nulls-first semantics; the coercion is defense in depth.
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

    /// Equality-delete keep-mask for one transformed batch: prefer the O(R) [`EqDeleteKeySet`]
    /// fast path; if any set reports a key-column NULL in this batch, fall back to the bound
    /// eq-delete predicate for the whole batch. Returns `None` when no eq-deletes apply.
    ///
    /// Shared by the whole-file path ([`Self::survival_mask`]) and the Parquet pushdown path when
    /// keysets are eligible and keys are projected (see `process_parquet_file_scan_task` routing).
    fn eq_delete_keep_mask(
        batch: &RecordBatch,
        num_rows: usize,
        eq_delete_predicate: Option<&BoundPredicate>,
        eq_delete_sets: Option<&[EqDeleteKeySet]>,
    ) -> Result<Option<BooleanArray>> {
        // Prefer the O(R) set fast path; if any set reports a key-column NULL in this batch,
        // fall back to the eq-delete predicate for the whole batch.
        let mut from_sets: Option<BooleanArray> = None;
        if let Some(sets) = eq_delete_sets.filter(|s| !s.is_empty()) {
            let mut keep = vec![true; num_rows];
            let mut all_sets_safe = true;
            for set in sets {
                // Always call `delete_mask` — do not skip on `is_empty()`. The I64 store drops
                // null delete cells; a null-only Long eq-delete file reports store-empty but still
                // must null-bail (or hit the predicate) so `col IS NULL` deletes apply. Skipping
                // empty sets produced a keep-all mask and under-deleted null data (FK1 critic-octo
                // cycle 2). Truly empty sets return `Some(all-false)` after the null check.
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
            // No set path (absent, empty, or a key-column NULL forced fallback) → predicate.
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

            // if row group selection is enabled,
            if let Some(selected_row_groups) = selected_row_groups {
                // if we've consumed all the selected row groups, we're done
                if selected_row_groups_idx == selected_row_groups.len() {
                    break;
                }

                if idx == selected_row_groups[selected_row_groups_idx] {
                    // we're in a selected row group. Increment selected_row_groups_idx
                    // so that next time around the for loop we're looking for the next
                    // selected row group
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

                    // still increment the current page base index but then skip to the next row group
                    // in the file
                    current_row_group_base_idx += row_group_num_rows;
                    continue;
                }
            }

            let mut next_deleted_row_idx = match next_deleted_row_idx_opt {
                Some(next_deleted_row_idx) => {
                    // if the index of the next deleted row is beyond this row group, add a selection for
                    // the remainder of this row group and skip to the next row group
                    if next_deleted_row_idx >= next_row_group_base_idx {
                        results.push(RowSelector::select(row_group_num_rows as usize));
                        current_row_group_base_idx += row_group_num_rows;
                        continue;
                    }

                    next_deleted_row_idx
                }

                // If there are no more pos deletes, add a selector for the entirety of this row group.
                _ => {
                    results.push(RowSelector::select(row_group_num_rows as usize));
                    current_row_group_base_idx += row_group_num_rows;
                    continue;
                }
            };

            let mut current_idx = current_row_group_base_idx;
            'chunks: while next_deleted_row_idx < next_row_group_base_idx {
                // `select` all rows that precede the next delete index
                if current_idx < next_deleted_row_idx {
                    let run_length = next_deleted_row_idx - current_idx;
                    results.push(RowSelector::select(run_length as usize));
                    current_idx += run_length;
                }

                // `skip` all consecutive deleted rows in the current row group
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
                            // We've processed the final positional delete.
                            // Conclude the skip and then break so that we select the remaining
                            // rows in the row group and move on to the next row group
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
        // Collects all Iceberg field IDs referenced in the filter predicate
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
            // A variant column is a leaf in the schema tree (its nested ids, if any, belong to
            // the F2+ shredding overlay). Reading variant DATA is deferred: a scan projecting a
            // variant column fails loudly earlier, in the Iceberg→Arrow schema conversion
            // (`ToArrowSchemaConverter::variant`), before this projection mask is consulted.
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
            // Field-ID-based projection using embedded field IDs from Parquet metadata

            // Parquet's columnar format requires leaf-level (not top-level struct/list/map) projection
            let mut leaf_field_ids = vec![];
            for field_id in field_ids {
                // Reserved ROW-LINEAGE ids are not in the table schema (they are reserved
                // metadata columns) but CAN be physically present in the file, so resolve them
                // from the reserved-column registry. Without this they are silently dropped
                // here — the leaf never reaches the projection, the stored column is never
                // decoded, and the transformer falls back to a COMPUTED row id: plausible,
                // wrong, and silent. They are scalars, so `include_leaf_field_id` on the
                // reserved field yields exactly the id itself.
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

        // Pre-project only the fields that have been selected, possibly avoiding converting
        // some Arrow types that are not yet supported.
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

            // Reserved row-lineage columns are not in the TABLE schema — they are reserved
            // metadata fields — but they can be present in the FILE, so resolve their declared
            // type from the reserved-column registry instead of failing the lookup.
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
        // Position-based: field_id N → column N-1 (field IDs are 1-indexed)
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
        // Collect Parquet column indices from field ids.
        // If the field id is not found in Parquet schema, it will be ignored due to schema evolution.
        let mut column_indices = iceberg_field_ids
            .iter()
            .filter_map(|field_id| field_id_map.get(field_id).cloned())
            .collect::<Vec<_>>();
        column_indices.sort();

        // The converter that converts `BoundPredicates` to `ArrowPredicates`
        let mut converter = PredicateConverter {
            parquet_schema,
            column_map: field_id_map,
            column_indices: &column_indices,
        };

        // After collecting required leaf column indices used in the predicate,
        // creates the projection mask for the Arrow predicates.
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

        // If all row groups were filtered out, return an empty RowSelection (select no rows)
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
                // skip row groups that aren't present in selected_row_groups
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
    /// start position in the file.
    ///
    /// The rule is `MIN(data_page_offset, dictionary_page_offset)` — the dictionary offset wins
    /// only when it is *set* **and strictly smaller**. Writers that emit a dictionary offset which
    /// is not smaller (or a garbage value with the `isSet` bit on) must still yield
    /// `data_page_offset`.
    ///
    /// Deliberately hand-rolled rather than using `ColumnChunkMetaData::byte_range()`, which is
    /// `dictionary_page_offset().unwrap_or(data_page_offset())` (no `min`, so it diverges from
    /// Java) and which `assert!`s on negative offsets (a panic on corrupt metadata).
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
    /// A row group is kept iff its **midpoint** falls in the half-open window
    /// `[start, start + length)`. This is parquet-mr's rule
    /// (`ParquetMetadataConverter.filterFileMetaDataByMidpoint` + `RangeMetadataFilter.contains`),
    /// which Iceberg drives through `Parquet.ReadBuilder.split(start, length)` →
    /// `ParquetReadOptions.Builder.withRange(start, start + length)`. Because every row group
    /// belongs to exactly one window, a full tiling of the file reads every row exactly once —
    /// an *overlap* rule would instead hand a row group that straddles a split boundary to **both**
    /// adjacent tasks, silently duplicating rows.
    ///
    /// Two details are load-bearing and must not be paraphrased:
    ///
    /// * The midpoint is `start_of_row_group + compressed_size / 2` with **truncating** integer
    ///   division on the size (Java `ldiv`), not the average of the two endpoints.
    /// * The window is inclusive at the low end and exclusive at the high end, so a midpoint
    ///   landing exactly on a split boundary belongs to the **higher** split.
    ///
    /// Row-group start positions are read from the real footer metadata
    /// ([`Self::parquet_column_chunk_offset`] over `columns()[0]`), never modelled as
    /// `4 + Σ compressed_size` — that model drifts on any file whose row groups are not perfectly
    /// contiguous (padding, inline bloom filters, a non-4 first offset). It is also exactly Java's
    /// *degenerate error-recovery* path (`invalidFileOffset`), reachable only in the omitted-inline-
    /// `ColumnMetaData` regime, which parquet-rs refuses to decode without the `encryption` feature.
    ///
    /// Two deliberate, fail-closed divergences from Java on corrupt metadata:
    ///
    /// * A negative offset or size is a typed [`ErrorKind::DataInvalid`] here. Java's `getOffset`
    ///   has no non-negativity guard, so it computes a negative midpoint, fails `>= startOffset`
    ///   and silently **drops** the row group. Rust is stricter and never silently under-reads.
    /// * A row group with no column chunks is a typed error; Java indexes `getColumns().get(0)`
    ///   unguarded and throws `IndexOutOfBoundsException`.
    /// * The row-group size is summed with `checked_add` instead of through
    ///   `RowGroupMetaData::compressed_size()`, whose unchecked `i64` `sum()` panics (debug) or
    ///   wraps (release) on a footer declaring several column chunks near `i64::MAX`. Java sums
    ///   into a `long` and wraps silently.
    ///
    /// Named residue (Java-identical, not a defect): because selection is midpoint-based, a window
    /// that does not cover a row group's midpoint reads none of its rows. A caller whose window set
    /// under-covers the file — a window narrower than the file with no sibling covering the rest,
    /// or a `split_offsets[0]` above the first midpoint — therefore loses rows silently, where the
    /// old overlap rule would have over-read. Java behaves the same way; the invariant callers
    /// must preserve is that their windows TILE `[0, file_size)`. (An understated manifest
    /// `file_size_in_bytes` is *not* one of those routes: `ArrowFileReader` anchors the footer read
    /// at that value, so it fails LOUDLY at metadata decode — measured — long before selection.)
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
            // Java reads `columns().get(0)` unguarded and throws IndexOutOfBounds on an empty row
            // group; we fail with a typed error instead of panicking.
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

            // `Σ columns.total_compressed_size`, i.e. Java's else-branch (parquet-rs does not
            // decode the thrift `RowGroup.total_compressed_size` field Java prefers). Summed HERE
            // with `checked_add` rather than via `RowGroupMetaData::compressed_size()`, whose
            // `i64` `sum()` is unchecked: parquet-rs applies no range validation to that thrift
            // field, so a corrupt footer declaring several chunks near `i64::MAX` PANICS there
            // (debug) or wraps to a bogus/negative size (release) before any guard below runs.
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

/// Builds a mapping from fallback field IDs to leaf column indices for Parquet files
/// without embedded field IDs. Returns entries only for primitive top-level fields.
///
/// Must use top-level field positions (not leaf column positions) to stay consistent
/// with `add_fallback_field_ids_to_arrow_schema`, which assigns ordinal IDs to
/// top-level Arrow fields. Using leaf positions instead would produce wrong indices
/// when nested types (struct/list/map) expand into multiple leaf columns.
///
/// Mirrors iceberg-java's ParquetSchemaUtil.addFallbackIds() which iterates
/// fileSchema.getFields() assigning ordinal IDs to top-level fields.
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

/// Apply name mapping to Arrow schema for Parquet files lacking field IDs.
///
/// Assigns Iceberg field IDs based on column names using the name mapping,
/// enabling correct projection on migrated files (e.g., from Hive/Spark via add_files).
///
/// Per Iceberg spec Column Projection rule #2:
/// "Use schema.name-mapping.default metadata to map field id to columns without field id"
/// https://iceberg.apache.org/spec/#column-projection
///
/// Corresponds to Java's ParquetSchemaUtil.applyNameMapping() and ApplyNameMapping visitor.
/// The key difference is Java operates on Parquet MessageType, while we operate on Arrow Schema.
///
/// # Arguments
/// * `arrow_schema` - Arrow schema from Parquet file (without field IDs)
/// * `name_mapping` - Name mapping from table metadata (TableProperties.DEFAULT_NAME_MAPPING)
///
/// # Returns
/// Arrow schema with field IDs assigned based on name mapping
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
            // Look up this column name in name mapping to get the Iceberg field ID.
            // Corresponds to Java's ApplyNameMapping visitor which calls
            // nameMapping.find(currentPath()) and returns field.withId() if found.
            //
            // If the field isn't in the mapping, leave it WITHOUT assigning an ID
            // (matching Java's behavior of returning the field unchanged).
            // Later, during projection, fields without IDs are filtered out.
            let mapped_field_opt = name_mapping
                .fields()
                .iter()
                .find(|f| f.names().contains(&field.name().to_string()));

            let mut metadata = field.metadata().clone();

            if let Some(mapped_field) = mapped_field_opt
                && let Some(field_id) = mapped_field.field_id()
            {
                // Field found in mapping with a field_id → assign it
                metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string());
            }
            // If field_id is None, leave the field without an ID (will be filtered by projection)

            Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                .with_metadata(metadata)
        })
        .collect();

    Ok(Arc::new(ArrowSchema::new_with_metadata(
        fields_with_mapped_ids,
        arrow_schema.metadata().clone(),
    )))
}

/// Add position-based fallback field IDs to Arrow schema for Parquet files lacking them.
/// Enables projection on migrated files (e.g., from Hive/Spark).
///
/// Why at schema level (not per-batch): Efficiency - avoids repeated schema modification.
/// Why only top-level: Nested projection uses leaf column indices, not parent struct IDs.
/// Why 1-indexed: Compatibility with iceberg-java's ParquetSchemaUtil.addFallbackIds().
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
        // The leaf column's index in Parquet schema.
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

            // The leaf column's index in the required column indices.
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

/// Coerce a three-valued [`BooleanArray`] keep-mask to a two-valued one where every NULL becomes
/// `false` (drop the row), matching the Parquet `RowFilter` (which never keeps a null result).
/// Defense in depth: [`evaluate_predicate_to_mask`] now resolves NULL cells to their Java
/// nulls-first verdicts and returns a TWO-valued mask (see `record_batch_predicate`'s module
/// docs), so this is a no-op on its output — it only guards future 3VL leaks.
fn coerce_nulls_to_false(mask: &BooleanArray) -> BooleanArray {
    if mask.null_count() == 0 {
        return mask.clone();
    }
    BooleanArray::from_iter((0..mask.len()).map(|i| Some(mask.is_valid(i) && mask.value(i))))
}

/// `true` iff every key field id of `sets` appears in the projected non-metadata field ids.
/// The Parquet MoR keyset path requires this so post-transform batches can resolve keys by
/// `PARQUET_FIELD_ID_META_KEY`; otherwise the reader falls back to the eq-delete RowFilter path.
pub(crate) fn eq_delete_key_fields_projected(
    sets: &[EqDeleteKeySet],
    projected_non_metadata_field_ids: &[i32],
) -> bool {
    if sets.is_empty() {
        return false;
    }
    // `collect_equality_delete_keysets` only returns sets that share one key schema, so the first
    // set's key ids stand for the whole task.
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
                // Per-row NaN test mirroring Java `NaNUtil.isNaN` (NULL cell ⇒ false; the mask
                // is two-valued, so the `RowFilter` never drops a row for being NULL here).
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
                // NULL cell ⇒ NOT NaN ⇒ `true` (row KEPT), matching Java
                // `Evaluator$EvalVisitor.notNaN` = `!NaNUtil.isNaN(value)`.
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
                // NULL cell ⇒ TRUE (row KEPT): Java's nulls-first comparator yields
                // compare(null, lit) == -1, and `Evaluator$EvalVisitor.lt` is `< 0` (`ifge 36`
                // at offset 29). A 3VL-null mask slot would make the `RowFilter` DROP the row.
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
                // NULL cell ⇒ TRUE: Java `ltEq` is `<= 0` (`ifgt 36`) over compare(null, lit)
                // == -1.
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
                // NULL cell ⇒ FALSE (Java `gt` is `> 0`, branch at offset 29) — same verdict
                // the RowFilter's null-drop produced, made explicit so `not`/`and`/`or`
                // composition stays plain boolean (Java's).
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
                // NULL cell ⇒ FALSE (Java `gtEq` is `>= 0`, `iflt 36`).
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
                // NULL cell ⇒ FALSE (Java `eq` is `== 0`, `ifne 36`; compare(null, lit) == -1).
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
                // NULL cell ⇒ TRUE (row KEPT): Java `notEq` = !eq. Pre-fix the kernel's 3VL
                // null made the `RowFilter` DROP every NULL cell under `!=` (audit BUG-003).
                Ok(null_filled(neq(&left, literal.as_ref())?, true))
            }))
        } else {
            // A missing column is a NULL column ⇒ TRUE: Java `notEq(null, lit)` is true. The
            // pre-fix `build_always_false()` made a schema-evolved file return ZERO rows under
            // `!=` (audit BUG-002).
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
                // NULL cell ⇒ FALSE: Java `startsWith` null-guards to false (`ifnull 38`,
                // offsets 11-12).
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
                // update here if arrow ever adds a native not_starts_with
                // NULL cell ⇒ TRUE (row KEPT): Java `notStartsWith` = !startsWith(null) =
                // !false.
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
            // `get_arrow_datum` is fallible (e.g. a decimal literal whose precision exceeds Arrow's
            // Decimal128 max, or a type it does not yet support). Propagate that as a typed error,
            // exactly like the scalar comparison arms above (`less_than`, `eq`, …) — never
            // `.unwrap()`-panic the predicate build.
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

                // NULL cell ⇒ FALSE: Java `in` = `literalSet.contains(null)` = false for
                // both set impls (see `record_batch_predicate`'s module docs).
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
            // Fallible for the same reasons as `r#in` above — propagate as a typed error rather
            // than `.unwrap()`-panicking the predicate build.
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

                // NULL cell ⇒ TRUE (row KEPT): Java `notIn` = !in = !contains(null) = true.
                // Pre-fix the accumulated 3VL null made the `RowFilter` DROP NULL cells.
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

    /// Override the default `get_byte_ranges` which calls `get_bytes` sequentially.
    /// The parquet reader calls this to fetch column chunks for a row group, so
    /// without this override each column chunk is a serial round-trip to object storage.
    /// Adapted from object_store's `coalesce_ranges` in `util.rs`.
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

/// The Arrow type of an array that the Parquet reader reads may not match the exact Arrow type
/// that Iceberg uses for literals - but they are effectively the same logical type,
/// i.e. LargeUtf8 and Utf8 or Utf8View and Utf8 or Utf8View and LargeUtf8.
///
/// The Arrow compute kernels that we use must match the type exactly, so first cast the literal
/// into the type of the batch we read from Parquet before sending it to the compute kernel.
///
/// `pub(crate)` so the `ConvertEqualityDeleteFiles` maintenance action's standalone predicate
/// evaluator can align literals to column types with the SAME logic the read-side `PredicateConverter`
/// uses.
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

    /// Drives the `IN` / `NOT IN` predicate-conversion arms with a decimal literal whose precision
    /// exceeds Arrow's Decimal128 max (38). `get_arrow_datum` returns a typed error for such a
    /// literal; the visitor must PROPAGATE it (like every scalar-comparison arm does), never
    /// `.unwrap()`-panic while building the row filter. A precision above 38 is reachable because
    /// the `decimal(P,S)` type-string deserializer and `Datum::try_from_bytes` do not bound-check
    /// precision, so hostile/corrupt catalog metadata can push such a datum into a bound predicate.
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
            // Type not supported
            Field::new("c2", DataType::Duration(TimeUnit::Microsecond), true).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]),
            ),
            // Precision is beyond the supported range
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

        // Try projecting the fields c2 and c3 with the unsupported data types
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

        // Omitting field c2, we still get an error due to c3 being selected
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

        // Finally avoid selecting fields with unsupported data types
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
            // a == 'foo'
            (Reference::new("a").equal_to(Datum::string("foo")), vec![
                Some("foo".to_string()),
            ]),
            // a != 'foo'
            (
                Reference::new("a").not_equal_to(Datum::string("foo")),
                vec![Some("bar".to_string())],
            ),
            // STARTS_WITH(a, 'foo')
            (Reference::new("a").starts_with(Datum::string("f")), vec![
                Some("foo".to_string()),
            ]),
            // NOT STARTS_WITH(a, 'foo')
            (
                Reference::new("a").not_starts_with(Datum::string("f")),
                vec![Some("bar".to_string())],
            ),
            // a < 'foo'
            (Reference::new("a").less_than(Datum::string("foo")), vec![
                Some("bar".to_string()),
            ]),
            // a <= 'foo'
            (
                Reference::new("a").less_than_or_equal_to(Datum::string("foo")),
                vec![Some("foo".to_string()), Some("bar".to_string())],
            ),
            // a > 'foo'
            (
                Reference::new("a").greater_than(Datum::string("bar")),
                vec![Some("foo".to_string())],
            ),
            // a >= 'foo'
            (
                Reference::new("a").greater_than_or_equal_to(Datum::string("foo")),
                vec![Some("foo".to_string())],
            ),
            // a IN ('foo', 'bar')
            (
                Reference::new("a").is_in([Datum::string("foo"), Datum::string("baz")]),
                vec![Some("foo".to_string())],
            ),
            // a NOT IN ('foo', 'bar')
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

        // Write the Parquet files
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

        // using selected row groups 1 and 3
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

        // selecting all row groups
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

    /// FIX (reader): `filter_row_groups_by_byte_range` guards `start + length` with
    /// `checked_add`; a split descriptor with `start = u64::MAX, length = 1` must return a typed
    /// `DataInvalid` error rather than overflowing `u64`. (The negative- and overflowing-
    /// `compressed_size` branches are pinned by cases `(h)` / `(i)` of
    /// `test_midpoint_selection_offset_and_boundary_semantics`.)
    #[test]
    fn test_filter_row_groups_by_byte_range_start_plus_length_overflow() {
        use parquet::file::metadata::{FileMetaData, ParquetMetaData};

        let schema_descr = get_test_schema_descr();
        // An empty file metadata suffices: the overflow guard fires before any row group is
        // examined, and a corrupt split must not even begin iterating.
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

    // ============================================================================================
    // U3 / hazard-1 — midpoint row-group selection (parquet-mr
    // `ParquetMetadataConverter.filterFileMetaDataByMidpoint`).
    //
    // Every helper below derives row-group positions from the REAL footer. NOTHING here may use
    // the `4 + Σ compressed_size` model: that model is what the production code used to do, so a
    // test that recomputes it is structurally incapable of catching offset drift.
    // ============================================================================================

    /// Writes `num_row_groups` row groups of `rows_per_group` sequential `id` values (ids start at
    /// 0 and run across row-group boundaries).
    ///
    /// With `bloom_filters = true`, parquet-rs writes each row group's bloom filter immediately
    /// after that row group ([`DEFAULT_BLOOM_FILTER_POSITION`] is `AfterRowGroup`), so the row
    /// groups are **not** contiguous and their real start offsets diverge from
    /// `4 + Σ compressed_size`.
    ///
    /// [`DEFAULT_BLOOM_FILTER_POSITION`]: parquet::file::properties::DEFAULT_BLOOM_FILTER_POSITION
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

    /// U3 / T1 — a fixed-size split tiling that STRADDLES row groups must read every row EXACTLY
    /// ONCE.
    ///
    /// Under the old OVERLAP rule the three 800-byte windows over this ~2.4 KiB file each claimed
    /// every row group they touched, so ids 100..299 were returned TWICE (500 rows read from a
    /// 300-row file) — a silent duplication, never an error. The midpoint rule assigns each row
    /// group to exactly one window.
    ///
    /// Every expectation is derived from real footer offsets; no compressed size is hardcoded
    /// (compression output is not contractually stable).
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
        // Non-vacuity: at least one row group must straddle a window boundary, otherwise this
        // fixture cannot distinguish the OVERLAP rule from the midpoint rule.
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

        // The exactly-once tiling property — the invariant the OVERLAP rule violates. This
        // assertion survives fixture drift (row-group sizes, stride, compression).
        union.sort_unstable();
        assert_eq!(
            union,
            (0..300).collect::<Vec<i32>>(),
            "the union over a full tiling must be every row EXACTLY once (duplicates here mean a \
             straddling row group was handed to two adjacent splits)"
        );
    }

    /// U3 cycle 4 / F-1 — a PARQUET task carrying the legacy whole-file sentinel
    /// (`start == 0, length == 0`) must still read every row after going through
    /// [`FileScanTask::split`], the way `TableScan::plan_tasks` does
    /// (`scan/mod.rs`: `split_tasks.extend(task.split(split_size)?)`).
    ///
    /// Before the sentinel guard in `split`, the fixed-size branch (`remaining = self.length`,
    /// `while remaining > 0`) returned ZERO sub-tasks for such a task: the file vanished from
    /// `plan_tasks` while `plan_files` still returned it, and the scan read 0 rows with NO error.
    /// The reader accepts the same spelling as whole-file (the byte-range gate below, the `_pos`
    /// guard, `reject_ranged_whole_file_task`), so the two halves must agree.
    ///
    /// **Which branch this now reaches: (1a), not (1b).** Cycle 6 widened (1a) to
    /// `start != 0 || length != file_size_in_bytes`, and this fixture (`0, 0` over a REAL file
    /// size) trips the second disjunct, so it returns at (1a) and the `length == 0` sentinel is no
    /// longer what answers it — measured: both sentinel mutants leave this test green. Branch
    /// (1b)'s only remaining reachable shape is `file_size_in_bytes == 0`. That shape IS reachable —
    /// `split` does return exactly one task `(start 0, length 0)` for it — but it cannot be exercised
    /// END-TO-END at the reader level: `ArrowFileReader` anchors the footer read at
    /// `file_size_in_bytes`, so the subsequent READ fails before any row is decoded, measured as
    /// `ErrorKind::Unexpected` "Failed to load Parquet metadata" with source
    /// "EOF: file size of 0 is less than footer". (1b) is therefore pinned at the unit level only, by
    /// `scan::task::tests::split_whole_file_sentinel_on_an_empty_file_is_one_task_not_zero`. What
    /// this test still pins end-to-end is the observable that matters: a sentinel-spelled task
    /// survives `split` as one task and reads every row. (Independent review of the reviewer rider,
    /// 2026-08-08 / Falsifier 5b.)
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

        // Non-vacuity: the same file spelled with an explicit length DOES split into several
        // windows at this target, so a `split` that returned an empty Vec here would be a real
        // asymmetry and not "this target never splits".
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

    /// U3 cycle 5 / F5 — splitting an ALREADY-RANGED task must not RELOCATE its byte window.
    ///
    /// `split`'s two real branches both treat the byte space as absolute from zero, so before the
    /// `start != 0` passthrough a parent covering `[starts[1], file_size)` came back as windows
    /// anchored at 0: the products re-read the prefix the parent never owned (ids 0..19 here) and
    /// dropped the tail it did. That is silent CORRUPTION — strictly worse than the `length == 0`
    /// row loss F-1 closed — and it is reachable through the `pub` struct / derived `Deserialize`.
    ///
    /// **Which branch each half reaches.** The first parent (`start = starts[1]`,
    /// `length = file_size - start`) trips BOTH of (1a)'s disjuncts, so it cannot tell them apart;
    /// the second half below keeps `length == file_size_in_bytes` and is answered by
    /// `start != 0` ALONE. Both halves are needed — the first is the honest end-to-end geometry,
    /// the second is the discriminating one.
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

        // Non-vacuity: the SAME target really does split the whole-file parent into many windows.
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
        // The fixture above trips BOTH disjuncts of (1a) (`start != 0` AND `length != file_size`),
        // so dropping `self.start != 0` leaves it green — measured. This second parent keeps
        // `length == file_size_in_bytes` and moves only the left edge, the one shape the
        // `length != file_size` disjunct cannot see, so it is `start != 0` alone that must answer.
        // (Independent review of the reviewer rider, 2026-08-08 / Falsifier 5a.)
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

    /// U3 cycle 4 / F-2 — the byte-range ENTRY gate
    /// (`if task.start != 0 || task.length != 0`) must fire on `start > 0, length == 0`.
    ///
    /// That disjunction is what makes `start == 0 && length == 0` — and ONLY that pair — mean
    /// "whole file". A task with a non-zero start and a zero length is an EMPTY window
    /// (`[start, start)`), which Java spells `withRange(start, start)`: `RangeMetadataFilter`'s
    /// `contains` is never true, so nothing is selected. Weakening the gate to `task.length != 0`
    /// silently turns that empty window into a full-file read — the whole point of this unit
    /// inverted — and every other test in the suite stays green, because none of them uses that
    /// shape.
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

        // The pinned shape: a non-zero start with a zero length is an EMPTY window, so no row
        // group's midpoint can lie in it and the read returns nothing.
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

    /// U3 / T2 — the OFFSET-SOURCE pin: a file whose row groups are NOT contiguous.
    ///
    /// parquet-rs writes bloom filters after each row group by default, so real row-group starts
    /// run ahead of `4 + Σ compressed_size`. The windows here are the file's OWN row-group
    /// boundaries — exactly what `FileScanTask::split`'s offsets-aware branch produces from the
    /// fork writer's `split_offsets` (`parquet_writer.rs` emits `RowGroupMetaData::file_offset()`,
    /// i.e. the real starts). This refutes the work order's exposure note: offsets-ALIGNED splits
    /// over a padded file were duplicating too, not only offsets-less external manifests.
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
        // Non-vacuity guard: if bloom filters ever stop padding the file this test silently
        // degrades into a duplicate of T1, so assert the drift it exists to detect.
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

    /// U3 / T3 — the exactly-once property over a sweep of strides and both fixture shapes.
    ///
    /// For any tiling of `[0, file_size)`, the selected row-group index sets must PARTITION
    /// `0..num_row_groups`: no index missing (silent under-read), no index selected twice (silent
    /// duplication). Stride- and fixture-independent, so it survives any change to compression or
    /// row-group sizing.
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

            // Adversarial tiling: put every window boundary EXACTLY on a row-group midpoint. Only
            // the half-open `[start, end)` convention partitions here — a strict low bound drops
            // every row group (silent under-read), an inclusive high bound selects each twice.
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

    /// Distance between the fabricated first column chunk and each trailing one. Large enough that
    /// selecting any column other than `columns()[0]` moves the midpoint out of every window the
    /// semantics test declares.
    const FABRICATED_COLUMN_STRIDE: i64 = 1_000_000;

    /// Builds row groups from `(data_page_offset, compressed_size, dict_offset)` triples — the
    /// triple always describes `columns()[0]` — so the selection rule can be probed at exact byte
    /// positions.
    ///
    /// Two properties keep this fixture DISCRIMINATING; both were mutation-survivable gaps in the
    /// first cut of this unit and are load-bearing, not incidental:
    ///
    /// * Each row group carries THREE column chunks, the trailing two placed
    ///   [`FABRICATED_COLUMN_STRIDE`] bytes apart, so reading any column other than `columns()[0]`
    ///   (Java's `getColumns().get(0)`) changes the answer. The trailing chunks declare a **zero**
    ///   compressed size, so `RowGroupMetaData::compressed_size()` — the sum over columns —
    ///   remains exactly the requested `compressed_size`.
    /// * `total_byte_size` (the UNCOMPRESSED size) is set to a value clearly different from the
    ///   compressed size, so reading it in place of `compressed_size()` (Java's `totalSize` is
    ///   `total_compressed_size`) changes the answer.
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

    /// U3 / T2b — the row-group start comes from the FIRST column chunk, proved on a REAL
    /// multi-column file rather than fabricated metadata.
    ///
    /// Java is `getOffset(rowGroup.getColumns().get(0))`. On a many-column file the last column
    /// chunk starts thousands of bytes downstream of the first, so a reader that indexes the wrong
    /// column pushes every midpoint into the following window: the first window then reads
    /// NOTHING and a later window claims two row groups — silent row loss plus duplication, with
    /// no error. A single-column fixture cannot see that at all.
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

        // A one-byte window at each row group's TRUE midpoint must select exactly that row group.
        // Indexing any other column chunk moves the midpoint elsewhere and empties the window.
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

        // (d5) a dictionary page offset of EXACTLY ZERO is still "set" and still wins. parquet-mr
        //      has TWO offset helpers that differ at precisely this predicate, and this call site
        //      must be the first one:
        //        * `ParquetMetadataConverter.getOffset(ColumnChunk)` — `isSetDictionary_page_offset()`
        //          with NO `> 0` test — is what drives `filterFileMetaDataByMidpoint`, i.e. the
        //          rule Iceberg's `withRange` READ path uses (this function).
        //        * `ColumnChunkMetaData.getStartingPos()` — `dictionaryPageOffset > 0 &&` — is the
        //          rule Iceberg's split-offset WRITER uses.
        //      Adding `> 0` here would push this row group's start from 0 to 1000 and its midpoint
        //      from 50 to 1050, quietly moving it into a different split. (U3 cycle 4 / F-4.)
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

        // (h) a NEGATIVE row-group compressed size is a typed error, not a silent selection by
        //     START. Without the guard the `u64` conversion would be replaced by a 0-valued size,
        //     making the midpoint equal the row-group start — wrong rows, no error. The public
        //     `ColumnChunkMetaData` builder accepts a negative `total_compressed_size`, so this
        //     branch IS constructible (an earlier revision of this file wrongly claimed otherwise).
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

        // (i) a footer whose column chunks sum past `i64::MAX` is a typed error, not a panic.
        //     `RowGroupMetaData::compressed_size()` sums the chunks with an UNCHECKED `i64`
        //     `sum()` and parquet-rs applies no range validation when decoding the thrift field,
        //     so calling it here would abort (debug) or wrap to a bogus size (release).
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

        // Force each batch into its own row group for testing byte range filtering.
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

        // Read the file metadata to get row group byte positions
        let file = File::open(&file_path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let metadata = reader.metadata();

        println!("File has {} row groups", metadata.num_row_groups());
        assert_eq!(metadata.num_row_groups(), 3, "Expected 3 row groups");

        // Get byte positions for each row group
        let row_group_0 = metadata.row_group(0);
        let row_group_1 = metadata.row_group(1);
        let row_group_2 = metadata.row_group(2);

        // U3 repair: the window boundaries are read from the REAL footer (Java `getOffset` =
        // `min(data_page_offset, dictionary_page_offset)` of the first column chunk), not modelled
        // as `4 + Σ compressed_size`. The old model happened to agree on this contiguous fixture,
        // which is precisely why this test could never catch offset drift; derived offsets make it
        // fail on a padded/bloom-filtered file.
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

        // Task 1: read only the first row group
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

        // Task 2: read the second and third row groups
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

        // Verify the actual data values are correct (not just the row count)
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

    /// Test schema evolution: reading old Parquet file (with only column 'a')
    /// using a newer table schema (with columns 'a' and 'b').
    /// This tests that:
    /// 1. get_arrow_projection_mask allows missing columns
    /// 2. RecordBatchTransformer adds missing column 'b' with NULL values
    #[tokio::test]
    async fn test_schema_evolution_add_column() {
        use arrow_array::{Array, Int32Array};

        // New table schema: columns 'a' and 'b' (b was added later, file only has 'a')
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

        // Create Arrow schema for old Parquet file (only has column 'a')
        let arrow_schema_old = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        // Write old Parquet file with only column 'a'
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

        // Read the old Parquet file using the NEW schema (with column 'b')
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

        // Verify we got the correct data
        assert_eq!(result.len(), 1);
        let batch = &result[0];

        // Should have 2 columns now
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);

        // Column 'a' should have the original data
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

        // Table schema: a single `id` column (field id 1).
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

        // Write a single Parquet file with 5 rows.
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

        // Scan projecting the data column AND the reserved `_pos` metadata column.
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

    // ===========================================================================================
    // FK5 — `_pos` projection streaming half (scout #16)
    //
    // Bar: dense + sparse pos-deletes + residual; row sets AND `_pos` values vs unpruned physical
    // oracle; multi-batch continuity; MERGE-shaped write pin; mutation bait on ordinal advance.
    // Selection-aware pushdown is STOP-gated (ledger).
    // ===========================================================================================

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
        // Enable row selection / RG filter so a REGRESSION that accidentally pushes them on the
        // `_pos` path would be exercised (and fail the ordinal oracle).
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
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("_row_id", DataType::Int64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                RESERVED_FIELD_ID_ROW_ID.to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(arrow_array::Int32Array::from(ids.to_vec())) as ArrayRef,
            Arc::new(arrow_array::Int64Array::from(row_ids.to_vec())) as ArrayRef,
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

    /// THE S1 REGRESSION PIN (bundle Critic F-A). `_row_id` and
    /// `_last_updated_sequence_number` are the only reserved metadata columns that can be
    /// PHYSICALLY PRESENT in a data file. The reader used to strip every `is_metadata_field` id
    /// from the Parquet projection, so the stored column was never decoded, the transformer's
    /// `RowIdFromFile` arm could not execute in production, and every row got a computed
    /// `first_row_id + pos` instead — plausible, wrong, and silent, for exactly the rows whose
    /// identity a rewrite had preserved.
    ///
    /// This test reads through the REAL `ArrowReader`, not a hand-built transformer fixture; the
    /// transformer-level tests cannot see this bug because they are handed a batch that already
    /// contains the column.
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

    /// `_last_updated_sequence_number` does NOT need physical ordinals, so it takes the ordinary
    /// Parquet path — a different transformer construction site than the `_row_id` test above.
    /// Both sites must be fed the task's row lineage; deleting either `with_row_lineage` call
    /// passed the entire suite before this test existed (bundle-Critic F-E/M3+M4).
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

    /// `_row_id` reaches the SAME whole-file guard as `_pos`, and for the same reason: Java's
    /// `RowIdReader` falls back to `firstRowId + pos`, so a row id computed after row-skipping is
    /// silently wrong — a plausible id belonging to a different row. Pinning this on `_pos` alone
    /// leaves a `_row_id`-only scan able to take the split path.
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

        let mut ranged = pos_scan_task(&data_path, id_schema_for_pos(), vec![], None);
        // `_row_id` ALONE — no `_pos` in the projection, so only the new half of the guard can
        // reject this.
        ranged.project_field_ids = Arc::from(vec![1, RESERVED_FIELD_ID_ROW_ID]);
        ranged.first_row_id = Some(1_000);
        ranged.start = 1;
        ranged.length = ranged.file_size_in_bytes;

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs()).build();
        let err = reader
            .read(Box::pin(futures::stream::iter(vec![Ok(ranged)])) as FileScanTaskStream)
            .expect("stream construction")
            .try_collect::<Vec<RecordBatch>>()
            .await
            .expect_err("a ranged task projecting `_row_id` must fail loud, not mint wrong ids");
        assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
        assert!(
            err.to_string()
                .contains("`_row_id` projection over a ranged split task"),
            "the error must name `_row_id`, not tell the caller to drop a `_pos` it never \
             projected, got: {err}"
        );
    }

    /// Review rider (2026-08-03): a RANGED split task must NOT take the `_pos` streaming path —
    /// it decodes the WHOLE file with ordinals from 0, so each split of one file would re-emit
    /// every row (duplicates) with wrong `_pos`. Fail loud instead. (Hazard-2 of the 2026-08-01
    /// plan_tasks review: reachable via the public `PartitionWork` seam / direct reader use, not
    /// via the DF provider, whose schema does not expose metadata columns.)
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

        // Ranged shapes on BOTH axes of the guard (`start == 0 && (length == 0 || length ==
        // file_size)`). Varying only the LENGTH leaves the `start == 0 &&` half unpinned:
        // `(1, file_size)` is a genuine window that a start-blind guard would ACCEPT, and the
        // `_pos` path would then decode the whole file with ordinals from 0. (U3 cycle 4 / F-3 —
        // the sibling of the same gap in `reject_ranged_whole_file_task`.)
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

        // Control: the SAME file as a whole-file task (0,0 legacy sentinel) still streams fine —
        // the guard must reject ONLY ranged windows.
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
    /// U3 rider (2026-08-07): this test used to be a FALSE GREEN for its own name — it passed
    /// unchanged with `max_row_group_row_count = None` (one row group), so nothing in it depended
    /// on the file being multi-RG. It now (a) asserts the fixture's row-group count from the real
    /// footer, and (b) carries a second leg that PRUNES the first row group with a residual
    /// predicate, so the surviving row group's `_pos` values must be offset by the pruned group's
    /// row count. A reader that restarted ordinals per surviving row group passes leg 1 and fails
    /// leg 2.
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
        // Structural pin: the fixture must actually be multi-row-group. (Mutation: pass `None` for
        // the row-group row count — this assertion goes RED, where before nothing did.)
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

        // Leg 2 — the BEHAVIOURAL discriminator. The `_pos` path deliberately decodes with no
        // row-skipping (no RowSelection / RowFilter / row-group pruning), so the only way the
        // multi-row-group shape reaches the reader is through the DECODE BATCH boundaries:
        // parquet-rs never spans a batch across a row group, so a batch size that does not divide
        // the row-group row count produces a SHORT batch at every row-group seam. That is exactly
        // where the `absolute_pos` / transformer dual counter can desync, and it is the property
        // this test's name claims to cover. With one row group the sequence would be
        // `[17; 11] + [13]`; with two it is `[17; 5] + [15]`, twice.
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

    /// FK5 mutation bait: absolute_pos must advance by full pre-filter batch size.
    ///
    /// MUTATION: in `apply_pos_aware_batch`, advance by filtered survivor count instead of
    /// pre-filter `row_count` → this test RED (second batch `_pos` shifts).
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

    /// Critic-octo C1: residual drops the entire first batch; later batches must still carry
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

    /// Critic-octo C1: every row position-deleted → empty result (no panic / bogus _pos).
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

    /// Critic-octo C1: single-batch vs multi-batch streaming must yield identical (id,_pos) sets
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
        // Unpruned oracle
        let deleted: HashSet<i64> = [1i64, 5, 11, 22, 29].into_iter().collect();
        let expected: Vec<(i32, i64)> = (2i64..28)
            .filter(|p| !deleted.contains(p))
            .map(|p| (p as i32, p))
            .collect();
        assert_eq!(single, expected);
    }

    /// Critic-octo C2: `_file` + `_pos` together under multi-batch streaming (constants + ordinals).
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

    /// Critic-octo C2: equality-delete + `_pos` streaming path (survival_mask eq branch).
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

    /// Critic-octo C5: residual ∩ pos-delete ∩ eq-delete under streaming `_pos`.
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

    /// Test for bug where position deletes in later row groups are not applied correctly.
    ///
    /// When a file has multiple row groups and a position delete targets a row in a later
    /// row group, the `build_deletes_row_selection` function had a bug where it would
    /// fail to increment `current_row_group_base_idx` when skipping row groups.
    ///
    /// This test creates:
    /// - A data file with 200 rows split into 2 row groups (0-99, 100-199)
    /// - A position delete file that deletes row 199 (last row in second row group)
    ///
    /// Expected behavior: Should return 199 rows (with id=200 deleted)
    /// Bug behavior: Returns 200 rows (delete is not applied)
    ///
    /// This bug was discovered while running Apache Spark + Apache Iceberg integration tests
    /// through DataFusion Comet. The following Iceberg Java tests failed due to this bug:
    /// - `org.apache.iceberg.spark.extensions.TestMergeOnReadDelete::testDeleteWithMultipleRowGroupsParquet`
    /// - `org.apache.iceberg.spark.extensions.TestMergeOnReadUpdate::testUpdateWithMultipleRowGroupsParquet`
    #[tokio::test]
    async fn test_position_delete_across_multiple_row_groups() {
        use arrow_array::{Int32Array, Int64Array};
        use parquet::file::reader::{FileReader, SerializedFileReader};

        // Field IDs for positional delete schema
        const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: u64 = 2147483546;
        const FIELD_ID_POSITIONAL_DELETE_POS: u64 = 2147483545;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();

        // Create table schema with a single 'id' column
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

        // Step 1: Create data file with 200 rows in 2 row groups
        // Row group 0: rows 0-99 (ids 1-100)
        // Row group 1: rows 100-199 (ids 101-200)
        let data_file_path = format!("{table_location}/data.parquet");

        let batch1 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(1..=100),
        )])
        .unwrap();

        let batch2 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(101..=200),
        )])
        .unwrap();

        // Force each batch into its own row group
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();

        let file = File::create(&data_file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        writer.write(&batch1).expect("Writing batch 1");
        writer.write(&batch2).expect("Writing batch 2");
        writer.close().unwrap();

        // Verify we created 2 row groups
        let verify_file = File::open(&data_file_path).unwrap();
        let verify_reader = SerializedFileReader::new(verify_file).unwrap();
        assert_eq!(
            verify_reader.metadata().num_row_groups(),
            2,
            "Should have 2 row groups"
        );

        // Step 2: Create position delete file that deletes row 199 (id=200, last row in row group 1)
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

        // Delete row at position 199 (0-indexed, so it's the last row: id=200)
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

        // Step 3: Read the data file with the delete applied
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

        // Step 4: Verify we got 199 rows (not 200)
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

        println!("Total rows read: {total_rows}");
        println!("Expected: 199 rows (deleted row 199 which had id=200)");

        // This assertion will FAIL before the fix and PASS after the fix
        assert_eq!(
            total_rows, 199,
            "Expected 199 rows after deleting row 199, but got {total_rows} rows. \
             The bug causes position deletes in later row groups to be ignored."
        );

        // Verify the deleted row (id=200) is not present
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

        // Verify we have all other ids (1-199)
        let expected_ids: Vec<i32> = (1..=199).collect();
        assert_eq!(
            all_ids, expected_ids,
            "Should have ids 1-199 but got different values"
        );
    }

    /// Test for bug where position deletes are lost when skipping unselected row groups.
    ///
    /// This is a variant of `test_position_delete_across_multiple_row_groups` that exercises
    /// the row group selection code path (`selected_row_groups: Some([...])`).
    ///
    /// When a file has multiple row groups and only some are selected for reading,
    /// the `build_deletes_row_selection` function must correctly skip over deletes in
    /// unselected row groups WITHOUT consuming deletes that belong to selected row groups.
    ///
    /// This test creates:
    /// - A data file with 200 rows split into 2 row groups (0-99, 100-199)
    /// - A position delete file that deletes row 199 (last row in second row group)
    /// - Row group selection that reads ONLY row group 1 (rows 100-199)
    ///
    /// Expected behavior: Should return 99 rows (with row 199 deleted)
    /// Bug behavior: Returns 100 rows (delete is lost when skipping row group 0)
    ///
    /// The bug occurs when processing row group 0 (unselected):
    /// ```rust
    /// delete_vector_iter.advance_to(next_row_group_base_idx); // Position at first delete >= 100
    /// next_deleted_row_idx_opt = delete_vector_iter.next(); // BUG: Consumes delete at 199!
    /// ```
    ///
    /// The fix is to NOT call `next()` after `advance_to()` when skipping unselected row groups,
    /// because `advance_to()` already positions the iterator correctly without consuming elements.
    #[tokio::test]
    async fn test_position_delete_with_row_group_selection() {
        use arrow_array::{Int32Array, Int64Array};
        use parquet::file::reader::{FileReader, SerializedFileReader};

        // Field IDs for positional delete schema
        const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: u64 = 2147483546;
        const FIELD_ID_POSITIONAL_DELETE_POS: u64 = 2147483545;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();

        // Create table schema with a single 'id' column
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

        // Step 1: Create data file with 200 rows in 2 row groups
        // Row group 0: rows 0-99 (ids 1-100)
        // Row group 1: rows 100-199 (ids 101-200)
        let data_file_path = format!("{table_location}/data.parquet");

        let batch1 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(1..=100),
        )])
        .unwrap();

        let batch2 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(101..=200),
        )])
        .unwrap();

        // Force each batch into its own row group
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();

        let file = File::create(&data_file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        writer.write(&batch1).expect("Writing batch 1");
        writer.write(&batch2).expect("Writing batch 2");
        writer.close().unwrap();

        // Verify we created 2 row groups
        let verify_file = File::open(&data_file_path).unwrap();
        let verify_reader = SerializedFileReader::new(verify_file).unwrap();
        assert_eq!(
            verify_reader.metadata().num_row_groups(),
            2,
            "Should have 2 row groups"
        );

        // Step 2: Create position delete file that deletes row 199 (id=200, last row in row group 1)
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

        // Delete row at position 199 (0-indexed, so it's the last row: id=200)
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

        // Step 3: Get byte ranges to read ONLY row group 1 (rows 100-199)
        // This exercises the row group selection code path where row group 0 is skipped
        let metadata_file = File::open(&data_file_path).unwrap();
        let metadata_reader = SerializedFileReader::new(metadata_file).unwrap();
        let metadata = metadata_reader.metadata();

        let row_group_0 = metadata.row_group(0);
        let row_group_1 = metadata.row_group(1);

        // U3 repair: the window is derived from the REAL footer offsets (Java `getOffset` =
        // `min(data_page_offset, dictionary_page_offset)` of the first column chunk), never from
        // the `4 + Σ compressed_size` model the production code used to synthesize — that model is
        // what made this test blind to offset drift. The assertion below records that this
        // particular fixture is contiguous (so both agree here); a padded/bloom-filtered file
        // would now be caught.
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

        // Create FileScanTask that reads ONLY row group 1 via byte range filtering
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

        // Step 4: Verify we got 99 rows (not 100)
        // Row group 1 has 100 rows (ids 101-200), minus 1 delete (id=200) = 99 rows
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

        println!("Total rows read from row group 1: {total_rows}");
        println!("Expected: 99 rows (row group 1 has 100 rows, 1 delete at position 199)");

        // This assertion will FAIL before the fix and PASS after the fix
        assert_eq!(
            total_rows, 99,
            "Expected 99 rows from row group 1 after deleting position 199, but got {total_rows} rows. \
             The bug causes position deletes to be lost when advance_to() is followed by next() \
             when skipping unselected row groups."
        );

        // Verify the deleted row (id=200) is not present
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

        // Verify we have ids 101-199 (not 101-200)
        let expected_ids: Vec<i32> = (101..=199).collect();
        assert_eq!(
            all_ids, expected_ids,
            "Should have ids 101-199 but got different values"
        );
    }
    /// Test for bug where stale cached delete causes infinite loop when skipping row groups.
    ///
    /// This test exposes the inverse scenario of `test_position_delete_with_row_group_selection`:
    /// - Position delete targets a row in the SKIPPED row group (not the selected one)
    /// - After calling advance_to(), the cached delete index is stale
    /// - Without updating the cache, the code enters an infinite loop
    ///
    /// This test creates:
    /// - A data file with 200 rows split into 2 row groups (0-99, 100-199)
    /// - A position delete file that deletes row 0 (first row in SKIPPED row group 0)
    /// - Row group selection that reads ONLY row group 1 (rows 100-199)
    ///
    /// The bug occurs when skipping row group 0:
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
    /// Expected behavior: Should return 100 rows (delete at 0 doesn't affect row group 1)
    /// Bug behavior: Infinite loop in build_deletes_row_selection
    #[tokio::test]
    async fn test_position_delete_in_skipped_row_group() {
        use arrow_array::{Int32Array, Int64Array};
        use parquet::file::reader::{FileReader, SerializedFileReader};

        // Field IDs for positional delete schema
        const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: u64 = 2147483546;
        const FIELD_ID_POSITIONAL_DELETE_POS: u64 = 2147483545;

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();

        // Create table schema with a single 'id' column
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

        // Step 1: Create data file with 200 rows in 2 row groups
        // Row group 0: rows 0-99 (ids 1-100)
        // Row group 1: rows 100-199 (ids 101-200)
        let data_file_path = format!("{table_location}/data.parquet");

        let batch1 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(1..=100),
        )])
        .unwrap();

        let batch2 = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(101..=200),
        )])
        .unwrap();

        // Force each batch into its own row group
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(100))
            .build();

        let file = File::create(&data_file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();
        writer.write(&batch1).expect("Writing batch 1");
        writer.write(&batch2).expect("Writing batch 2");
        writer.close().unwrap();

        // Verify we created 2 row groups
        let verify_file = File::open(&data_file_path).unwrap();
        let verify_reader = SerializedFileReader::new(verify_file).unwrap();
        assert_eq!(
            verify_reader.metadata().num_row_groups(),
            2,
            "Should have 2 row groups"
        );

        // Step 2: Create position delete file that deletes row 0 (id=1, first row in row group 0)
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

        // Delete row at position 0 (0-indexed, so it's the first row: id=1)
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

        // Step 3: Get byte ranges to read ONLY row group 1 (rows 100-199)
        // This exercises the row group selection code path where row group 0 is skipped
        let metadata_file = File::open(&data_file_path).unwrap();
        let metadata_reader = SerializedFileReader::new(metadata_file).unwrap();
        let metadata = metadata_reader.metadata();

        let row_group_0 = metadata.row_group(0);
        let row_group_1 = metadata.row_group(1);

        // U3 repair: the window is derived from the REAL footer offsets (Java `getOffset` =
        // `min(data_page_offset, dictionary_page_offset)` of the first column chunk), never from
        // the `4 + Σ compressed_size` model the production code used to synthesize — that model is
        // what made this test blind to offset drift. The assertion below records that this
        // particular fixture is contiguous (so both agree here); a padded/bloom-filtered file
        // would now be caught.
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

        // Create FileScanTask that reads ONLY row group 1 via byte range filtering
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

        // Step 4: Verify we got 100 rows (all of row group 1)
        // The delete at position 0 is in row group 0, which is skipped, so it doesn't affect us
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

        assert_eq!(
            total_rows, 100,
            "Expected 100 rows from row group 1 (delete at position 0 is in skipped row group 0). \
             If this hangs or fails, it indicates the cached delete index was not updated after advance_to()."
        );

        // Verify we have all ids from row group 1 (101-200)
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

    /// Test reading Parquet files without field ID metadata (e.g., migrated tables).
    /// This exercises the position-based fallback path.
    ///
    /// Corresponds to Java's ParquetSchemaUtil.addFallbackIds() + pruneColumnsFallback()
    /// in /parquet/src/main/java/org/apache/iceberg/parquet/ParquetSchemaUtil.java
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

        // Parquet file from a migrated table - no field ID metadata
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

        // Verify position-based mapping: field_id 1 → position 0, field_id 2 → position 1
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

        // Schema with field 3 added after the file was written
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

        // Verify missing column filled with NULLs
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

        // Small row group size to create multiple row groups
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_write_batch_size(2)
            .set_max_row_group_row_count(Some(2))
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();

        // Write 6 rows in 3 batches (will create 3 row groups)
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

    /// Test reading Parquet files without field IDs with a filter that eliminates all row groups.
    /// During development of field ID mapping, we saw a panic when row_selection_enabled=true and
    /// all row groups are filtered out.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_filter_eliminates_all_rows() {
        use arrow_array::{Float64Array, Int32Array};

        // Schema with fields that will use fallback IDs 1, 2, 3
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

        // Write data where all ids are >= 10
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

        // Should no longer panic
        let result = reader
            .read(tasks)
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        // Should return empty results
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

        // Create 3 parquet files with different data
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

        // Read with concurrency=1 (fast-path)
        let reader = ArrowReaderBuilder::new(file_io)
            .with_data_file_concurrency_limit(1)
            .build();

        // Create tasks in a specific order: file_0, file_1, file_2
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

        // Verify we got all 30 rows (10 from each file)
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 30, "Should have 30 total rows");

        // Collect all ids and file_nums to verify data
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

    /// Test bucket partitioning reads source column from data file (not partition metadata).
    ///
    /// This is an integration test verifying the complete ArrowReader pipeline with bucket partitioning.
    /// It corresponds to TestRuntimeFiltering tests in Iceberg Java (e.g., testRenamedSourceColumnTable).
    ///
    /// # Iceberg Spec Requirements
    ///
    /// Per the Iceberg spec "Column Projection" section:
    /// > "Return the value from partition metadata if an **Identity Transform** exists for the field"
    ///
    /// This means:
    /// - Identity transforms (e.g., `identity(dept)`) use constants from partition metadata
    /// - Non-identity transforms (e.g., `bucket(4, id)`) must read source columns from data files
    /// - Partition metadata for bucket transforms stores bucket numbers (0-3), NOT source values
    ///
    /// Java's PartitionUtil.constantsMap() implements this via:
    /// ```java
    /// if (field.transform().isIdentity()) {
    ///     idToConstant.put(field.sourceId(), converted);
    /// }
    /// ```
    ///
    /// # What This Test Verifies
    ///
    /// This test ensures the full ArrowReader → RecordBatchTransformer pipeline correctly handles
    /// bucket partitioning when FileScanTask provides partition_spec and partition_data:
    ///
    /// - Parquet file has field_id=1 named "id" with actual data [1, 5, 9, 13]
    /// - FileScanTask specifies partition_spec with bucket(4, id) and partition_data with bucket=1
    /// - RecordBatchTransformer.constants_map() excludes bucket-partitioned field from constants
    /// - ArrowReader correctly reads [1, 5, 9, 13] from the data file
    /// - Values are NOT replaced with constant 1 from partition metadata
    ///
    /// # Why This Matters
    ///
    /// Without correct handling:
    /// - Runtime filtering would break (e.g., `WHERE id = 5` would fail)
    /// - Query results would be incorrect (all rows would have id=1)
    /// - Bucket partitioning would be unusable for query optimization
    ///
    /// # References
    /// - Iceberg spec: format/spec.md "Column Projection" + "Partition Transforms"
    /// - Java test: spark/src/test/java/.../TestRuntimeFiltering.java
    /// - Java impl: core/src/main/java/org/apache/iceberg/util/PartitionUtil.java
    #[tokio::test]
    async fn test_bucket_partitioning_reads_source_column_from_file() {
        use arrow_array::Int32Array;

        use crate::spec::{Literal, PartitionSpec, Struct, Transform};

        // Iceberg schema with id and name columns
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

        // Partition spec: bucket(4, id)
        let partition_spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(0)
                .add_partition_field("id", "id_bucket", Transform::Bucket(4))
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition data: bucket value is 1
        let partition_data = Struct::from_iter(vec![Some(Literal::int(1))]);

        // Create Arrow schema with field IDs for Parquet file
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

        // Write Parquet file with data
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

        // Read the Parquet file with partition spec and data
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

        // Verify we got the correct data
        assert_eq!(result.len(), 1);
        let batch = &result[0];

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 4);

        // The id column MUST contain actual values from the Parquet file [1, 5, 9, 13],
        // NOT the constant partition value 1
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
            // Fill with sequential byte values so slices are verifiable.
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

        // MAP encoding:
        //   optional group ts_map (MAP) {
        //     repeated group key_value {
        //       required binary key (UTF8);
        //       optional int96 value;
        //     }
        //   }
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

    // RISK (the compile-forced reader leaf arm): a variant column contributes its OWN field id to
    // the parquet projection-mask leaf set, exactly like a primitive — its nested parquet column
    // ids (if any) belong to the F2+ shredding overlay, and Java's pruning treats the variant
    // column as one selectable unit (1.10.0 `TypeUtil.select` keeps it whole; live-Java-probed).
    // This arm is unreachable through a real scan today (the Iceberg→Arrow conversion errors
    // first), so this direct unit test is the only thing pinning its semantics: dropping the
    // `field_ids.push` would silently project variant columns out once the arrow door opens.
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
    //! Scan-level tests for the Avro AND ORC data-file READ paths
    //! (`process_avro_file_scan_task` / `process_orc_file_scan_task`): a real Avro OCF / the committed
    //! golden Java-Iceberg ORC fixture on disk, scanned through `ArrowReader::read`, with projection +
    //! merge-on-read positional/equality deletes applied post-materialization, the by-field-id
    //! (rename) proof, and mutation baits. These exercise the U2 wiring end-to-end; the U1
    //! `avro_reader_tests.rs` / `orc_reader_tests.rs` cover the decode cores in isolation.

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

    /// The AVRO reader decodes WHOLE files — it never reads `task.start` / `task.length`. So if
    /// the planner split an Avro file into byte windows, every sub-task would re-emit every row
    /// and an N-way split would silently return N copies of the file: the exact silent-duplication
    /// class the Parquet midpoint row-group selection was written to eliminate, with no error at
    /// any layer.
    ///
    /// This drives the REAL `FileScanTask::split` → `ArrowReader::read` path (what
    /// `TableScan::plan_tasks` does at `scan/mod.rs`) and asserts the exactly-once property over
    /// the whole split set. Before `scan::task::reader_honors_byte_range` this returned 4× the
    /// file's rows.
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
    ///
    /// `reject_ranged_whole_file_task` is invoked from BOTH `process_avro_file_scan_task` and
    /// `process_orc_file_scan_task`, but only the AVRO call site was pinned: deleting the ORC line
    /// left the whole suite green (U3 cycle 5, the Critic's S2 / the Falsifier's F9). Since
    /// `process_orc_file_scan_task` never reads `task.start` / `task.length`, an unguarded ranged
    /// ORC task re-emits every row of the file — N copies per N-way split, with no error.
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
    //! Wave B: Parquet MoR path wires [`EqDeleteKeySet`] when key columns are projected.
    //!
    //! Routing (see `process_parquet_file_scan_task`):
    //! * keys ⊆ projected non-metadata field ids → post-decode keyset keep-mask (RowFilter residual
    //!   is scan-predicate only);
    //! * otherwise → today's AND of eq-delete predicate into the Parquet RowFilter.
    //!
    //! Both routes must produce the same survivors (predicate oracle).

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

    /// Critic-octo C1-Q-001: when keys are projected (keyset post-decode path), the scan residual
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

    /// Critic-octo C1-Q-002: nullable key column with a NULL cell forces predicate fallback
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

    /// Critic-octo C2-Q-001: composite equality key (id + data) on the Parquet keyset path.
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

    /// Critic-octo C2-Q-002: positional RowSelection + eq keyset post-decode on one Parquet task.
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

    /// Critic-octo C3-Q-001: keyset path when the projection is *only* the key column (no
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

    /// Critic-octo C3-Q-002: two eq-delete files OR-combined under the keyset path
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

    /// Critic-octo C4-Q-002: keyset path that deletes every row yields empty (or no-row) output —
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
