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

use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, RowSelection};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};

use crate::arrow::footer_cache::TableFooterCache;
use crate::arrow::reader::{
    ArrowFileReader, ArrowReader, CollectFieldIdVisitor, ParquetReadOptions,
};
use crate::error::Result;
use crate::expr::BoundPredicate;
use crate::expr::visitors::bound_predicate_visitor::visit;
use crate::expr::visitors::page_index_evaluator::PageIndexEvaluator;
use crate::io::{FileIO, FileMetadata};
use crate::metadata_columns::{
    RESERVED_FIELD_ID_POS, RESERVED_FIELD_ID_ROW_ID, get_metadata_field, is_metadata_field,
    is_row_lineage_field,
};
use crate::scan::FileScanTask;
use crate::spec::Schema;
use crate::{Error, ErrorKind};

pub(crate) fn page_index_policy(needed: bool) -> PageIndexPolicy {
    if needed {
        PageIndexPolicy::Optional
    } else {
        PageIndexPolicy::Skip
    }
}

#[cfg(test)]
thread_local! {
    pub(crate) static ROW_SELECTIONS_APPLIED: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    pub(crate) static PAGE_INDEX_STRIPS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_applied_selection(selection: &Option<RowSelection>) {
    if selection.is_some() {
        ROW_SELECTIONS_APPLIED.with(|count| count.set(count.get() + 1));
    }
}

#[cfg(not(test))]
fn record_applied_selection(_: &Option<RowSelection>) {}

#[cfg(test)]
fn record_index_strip() {
    PAGE_INDEX_STRIPS.with(|count| count.set(count.get() + 1));
}

#[cfg(not(test))]
fn record_index_strip() {}

pub(crate) fn effective_row_selection(selection: Option<RowSelection>) -> Option<RowSelection> {
    let effective = selection.filter(|s| s.skipped_row_count() > 0 || !s.selects_any());
    record_applied_selection(&effective);
    effective
}

#[derive(Debug)]
pub(crate) enum OpenParquetError {
    Footer(Error),
    Other(Error),
}

impl OpenParquetError {
    fn into_error(self) -> Error {
        match self {
            Self::Footer(e) | Self::Other(e) => e,
        }
    }
}

impl ArrowReader {
    pub(crate) async fn open_parquet_file(
        data_file_path: &str,
        file_io: &FileIO,
        file_size_in_bytes: u64,
        parquet_read_options: ParquetReadOptions,
        prefetched_metadata: Option<Arc<ParquetMetaData>>,
    ) -> Result<(ArrowFileReader, ArrowReaderMetadata)> {
        Self::open_parquet_file_cached(
            &Arc::from(data_file_path),
            file_io,
            file_size_in_bytes,
            parquet_read_options,
            prefetched_metadata,
            None,
        )
        .await
    }

    pub(crate) async fn open_parquet_file_cached(
        data_file_path: &Arc<str>,
        file_io: &FileIO,
        file_size_in_bytes: u64,
        parquet_read_options: ParquetReadOptions,
        prefetched_metadata: Option<Arc<ParquetMetaData>>,
        footer_cache: Option<TableFooterCache>,
    ) -> Result<(ArrowFileReader, ArrowReaderMetadata)> {
        let first_error = match Self::open_parquet_file_sized(
            data_file_path,
            file_io,
            file_size_in_bytes,
            parquet_read_options,
            prefetched_metadata.clone(),
            footer_cache.as_ref(),
        )
        .await
        {
            Ok(opened) => return Ok(opened),
            Err(OpenParquetError::Other(e)) => return Err(e),
            Err(OpenParquetError::Footer(e)) => e,
        };
        let actual_size = match file_io.new_input(data_file_path) {
            Ok(input) => input
                .metadata()
                .await
                .map(|meta| meta.size)
                .unwrap_or(file_size_in_bytes),
            Err(_) => file_size_in_bytes,
        };
        if actual_size == file_size_in_bytes {
            return Err(first_error);
        }
        match Self::open_parquet_file_sized(
            data_file_path,
            file_io,
            actual_size,
            parquet_read_options,
            prefetched_metadata,
            footer_cache.as_ref(),
        )
        .await
        {
            Ok(opened) => Ok(opened),
            Err(retry_error) => {
                let retry_error = retry_error.into_error();
                Err(
                    Error::new(retry_error.kind(), retry_error.message().to_string())
                        .with_source(first_error),
                )
            }
        }
    }

    pub(crate) fn build_expected_schema(task: &FileScanTask) -> Result<Arc<Schema>> {
        let mut field_ids: Vec<i32> = task.project_field_ids().to_vec();
        if let Some(predicate) = task.predicate.as_deref() {
            let mut collector = CollectFieldIdVisitor::default();
            visit(&mut collector, predicate)?;
            field_ids.extend(collector.field_ids());
        }
        for delete in task.deletes.iter() {
            if let Some(equality_ids) = &delete.equality_ids {
                field_ids.extend(equality_ids.iter().copied());
            }
        }
        field_ids.sort_unstable();
        field_ids.dedup();
        let mut fields = Vec::new();
        for &field_id in &field_ids {
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

    async fn open_parquet_file_sized(
        data_file_path: &Arc<str>,
        file_io: &FileIO,
        file_size_in_bytes: u64,
        parquet_read_options: ParquetReadOptions,
        prefetched_metadata: Option<Arc<ParquetMetaData>>,
        footer_cache: Option<&TableFooterCache>,
    ) -> std::result::Result<(ArrowFileReader, ArrowReaderMetadata), OpenParquetError> {
        let parquet_file = file_io
            .new_input(data_file_path)
            .map_err(OpenParquetError::Other)?;
        let parquet_reader = parquet_file
            .reader()
            .await
            .map_err(OpenParquetError::Other)?;
        let mut reader = ArrowFileReader::new(
            FileMetadata {
                size: file_size_in_bytes,
            },
            parquet_reader,
        )
        .with_parquet_read_options(parquet_read_options);

        let arrow_metadata = if let Some(footer_cache) = footer_cache {
            if let Some(prefetched) = prefetched_metadata {
                footer_cache
                    .seed(data_file_path, file_size_in_bytes, prefetched)
                    .await;
            }
            footer_cache
                .footer_or_fetch(
                    data_file_path,
                    file_size_in_bytes,
                    parquet_read_options,
                    &mut reader,
                )
                .await?
                .as_ref()
                .clone()
        } else {
            match prefetched_metadata {
                Some(metadata) => {
                    let mut metadata_reader = ParquetMetaDataReader::new_with_metadata(
                        ParquetMetaData::clone(metadata.as_ref()),
                    )
                    .with_page_index_policy(page_index_policy(
                        parquet_read_options.preload_page_index(),
                    ))
                    .with_column_index_policy(page_index_policy(
                        parquet_read_options.preload_column_index(),
                    ))
                    .with_offset_index_policy(page_index_policy(
                        parquet_read_options.preload_offset_index(),
                    ));
                    metadata_reader
                        .load_page_index(&mut reader)
                        .await
                        .map_err(|e| {
                            OpenParquetError::Other(
                                Error::new(
                                    ErrorKind::Unexpected,
                                    "Failed to load Parquet page index",
                                )
                                .with_source(e),
                            )
                        })?;
                    let metadata = metadata_reader.finish().map_err(|e| {
                        OpenParquetError::Other(
                            Error::new(ErrorKind::Unexpected, "Failed to load Parquet metadata")
                                .with_source(e),
                        )
                    })?;
                    ArrowReaderMetadata::try_new(Arc::new(metadata), Default::default()).map_err(
                        |e| {
                            OpenParquetError::Other(
                                Error::new(
                                    ErrorKind::Unexpected,
                                    "Failed to load Parquet metadata",
                                )
                                .with_source(e),
                            )
                        },
                    )?
                }
                None => {
                    let options = ArrowReaderOptions::default();
                    let metadata = reader.get_metadata(Some(&options)).await.map_err(|e| {
                        OpenParquetError::Footer(
                            Error::new(ErrorKind::Unexpected, "Failed to load Parquet metadata")
                                .with_source(e),
                        )
                    })?;
                    ArrowReaderMetadata::try_new(metadata, options).map_err(|e| {
                        OpenParquetError::Other(
                            Error::new(ErrorKind::Unexpected, "Failed to load Parquet metadata")
                                .with_source(e),
                        )
                    })?
                }
            }
        };

        Ok((reader, arrow_metadata))
    }

    pub(crate) fn prune_indexed_metadata_for_scan(
        task: &FileScanTask,
        arrow_metadata: ArrowReaderMetadata,
        row_group_filtering_enabled: bool,
        row_selection_enabled: bool,
        predicate_can_prune: bool,
    ) -> Result<(
        ArrowReaderMetadata,
        Option<Vec<usize>>,
        Option<RowSelection>,
    )> {
        let needs_physical_ordinals = task.project_field_ids().contains(&RESERVED_FIELD_ID_POS)
            || task.project_field_ids().contains(&RESERVED_FIELD_ID_ROW_ID);
        let decide_early = task.deletes.is_empty()
            && row_selection_enabled
            && predicate_can_prune
            && !needs_physical_ordinals;
        let Some(predicate) = task.predicate.as_deref().filter(|_| decide_early) else {
            return Ok((arrow_metadata, None, None));
        };
        let (_, field_id_map) =
            Self::build_field_id_set_and_map(arrow_metadata.parquet_schema(), predicate)?;
        let mut selected_row_group_indices = (task.start != 0 || task.length != 0)
            .then(|| {
                Self::filter_row_groups_by_byte_range(
                    arrow_metadata.metadata(),
                    task.start,
                    task.length,
                )
            })
            .transpose()?;
        if row_group_filtering_enabled {
            let pruned = Self::get_selected_row_group_indices(
                predicate,
                arrow_metadata.metadata(),
                &field_id_map,
                &task.schema,
            )?;
            selected_row_group_indices = Some(match selected_row_group_indices {
                Some(byte_range) => byte_range
                    .into_iter()
                    .filter(|idx| pruned.contains(idx))
                    .collect(),
                None => pruned,
            });
        }
        let row_selection = Self::get_row_selection_for_filter_predicate(
            predicate,
            arrow_metadata.metadata(),
            &selected_row_group_indices,
            &field_id_map,
            &task.schema,
        )?;
        let redundant = row_selection
            .as_ref()
            .is_some_and(|s| s.selects_any() && s.skipped_row_count() == 0);
        if !redundant {
            return Ok((arrow_metadata, selected_row_group_indices, row_selection));
        }
        record_index_strip();
        let schema = Arc::clone(arrow_metadata.schema());
        let metadata = Arc::clone(arrow_metadata.metadata());
        drop(arrow_metadata);
        let stripped = match Arc::try_unwrap(metadata) {
            Ok(owned) => owned
                .into_builder()
                .set_column_index(None)
                .set_offset_index(None)
                .build(),
            Err(shared) => {
                ParquetMetaData::new(shared.file_metadata().clone(), shared.row_groups().to_vec())
            }
        };
        let options = ArrowReaderOptions::new().with_schema(schema);
        let arrow_metadata =
            ArrowReaderMetadata::try_new(Arc::new(stripped), options).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to create ArrowReaderMetadata without page index",
                )
                .with_source(e)
            })?;
        Ok((arrow_metadata, selected_row_group_indices, row_selection))
    }
}

impl ArrowReader {
    pub(crate) fn get_row_selection_for_filter_predicate(
        predicate: &BoundPredicate,
        parquet_metadata: &Arc<ParquetMetaData>,
        selected_row_groups: &Option<Vec<usize>>,
        field_id_map: &HashMap<i32, usize>,
        snapshot_schema: &Schema,
    ) -> Result<Option<RowSelection>> {
        let (Some(column_index), Some(offset_index)) = (
            parquet_metadata.column_index(),
            parquet_metadata.offset_index(),
        ) else {
            return Ok(None);
        };

        if let Some(selected_row_groups) = selected_row_groups
            && selected_row_groups.is_empty()
        {
            return Ok(Some(RowSelection::from(Vec::new())));
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

        Ok(Some(
            results.into_iter().flatten().collect::<Vec<_>>().into(),
        ))
    }
}
