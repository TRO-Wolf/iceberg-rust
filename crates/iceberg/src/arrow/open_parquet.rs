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

use parquet::arrow::arrow_reader::{ArrowReaderMetadata, RowSelection};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};

use crate::arrow::reader::{
    ArrowFileReader, ArrowReader, CollectFieldIdVisitor, ParquetReadOptions,
};
use crate::error::Result;
use crate::expr::BoundPredicate;
use crate::expr::visitors::bound_predicate_visitor::visit;
use crate::expr::visitors::page_index_evaluator::PageIndexEvaluator;
use crate::io::{FileIO, FileMetadata};
use crate::metadata_columns::{get_metadata_field, is_metadata_field, is_row_lineage_field};
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

impl ArrowReader {
    pub(crate) async fn open_parquet_file(
        data_file_path: &str,
        file_io: &FileIO,
        file_size_in_bytes: u64,
        parquet_read_options: ParquetReadOptions,
        prefetched_metadata: Option<Arc<ParquetMetaData>>,
    ) -> Result<(ArrowFileReader, ArrowReaderMetadata)> {
        let opened = Self::open_parquet_file_sized(
            data_file_path,
            file_io,
            file_size_in_bytes,
            parquet_read_options,
            prefetched_metadata.clone(),
        )
        .await;
        let Err(first_error) = opened else {
            return opened;
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
        Self::open_parquet_file_sized(
            data_file_path,
            file_io,
            actual_size,
            parquet_read_options,
            prefetched_metadata,
        )
        .await
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
        data_file_path: &str,
        file_io: &FileIO,
        file_size_in_bytes: u64,
        parquet_read_options: ParquetReadOptions,
        prefetched_metadata: Option<Arc<ParquetMetaData>>,
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

        let arrow_metadata = match prefetched_metadata {
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
                        Error::new(ErrorKind::Unexpected, "Failed to load Parquet page index")
                            .with_source(e)
                    })?;
                let metadata = metadata_reader.finish().map_err(|e| {
                    Error::new(ErrorKind::Unexpected, "Failed to load Parquet metadata")
                        .with_source(e)
                })?;
                ArrowReaderMetadata::try_new(Arc::new(metadata), Default::default()).map_err(
                    |e| {
                        Error::new(ErrorKind::Unexpected, "Failed to load Parquet metadata")
                            .with_source(e)
                    },
                )?
            }
            None => ArrowReaderMetadata::load_async(&mut reader, Default::default())
                .await
                .map_err(|e| {
                    Error::new(ErrorKind::Unexpected, "Failed to load Parquet metadata")
                        .with_source(e)
                })?,
        };

        Ok((reader, arrow_metadata))
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
