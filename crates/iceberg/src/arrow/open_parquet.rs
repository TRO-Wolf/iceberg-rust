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

use std::sync::Arc;

use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};

use crate::arrow::reader::{ArrowFileReader, ArrowReader, ParquetReadOptions};
use crate::error::Result;
use crate::io::{FileIO, FileMetadata};
use crate::{Error, ErrorKind};

impl ArrowReader {
    pub(crate) async fn open_parquet_file(
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
                .with_page_index_policy(PageIndexPolicy::from(
                    parquet_read_options.preload_page_index(),
                ))
                .with_column_index_policy(PageIndexPolicy::from(
                    parquet_read_options.preload_column_index(),
                ))
                .with_offset_index_policy(PageIndexPolicy::from(
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
