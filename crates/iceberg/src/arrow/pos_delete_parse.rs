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

use arrow_array::{Array, Int64Array, StringArray};
use futures::StreamExt;

use crate::delete_vector::DeleteVector;
use crate::scan::ArrowRecordBatchStream;
use crate::{Error, ErrorKind, Result};

pub(crate) fn checked_delete_position(
    delete_file_path: &str,
    data_file_path: &str,
    pos: i64,
) -> Result<u64> {
    u64::try_from(pos).map_err(|_| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Invalid position delete file '{delete_file_path}': negative position \
                 {pos} for data file '{data_file_path}'"
            ),
        )
    })
}

pub(crate) async fn parse_positional_deletes_record_batch_stream(
    delete_file_path: &str,
    mut stream: ArrowRecordBatchStream,
) -> Result<HashMap<String, DeleteVector>> {
    let mut result: HashMap<String, DeleteVector> = HashMap::default();

    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let columns = batch.columns();

        let [file_paths, positions, ..] = columns else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid position delete file '{delete_file_path}': expected at least 2 \
                     columns (file_path, pos), found {}",
                    columns.len()
                ),
            ));
        };

        let Some(file_paths) = file_paths.as_any().downcast_ref::<StringArray>() else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Could not downcast file paths array to StringArray",
            ));
        };
        let Some(positions) = positions.as_any().downcast_ref::<Int64Array>() else {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Could not downcast positions array to Int64Array",
            ));
        };

        let mut current: Option<(&str, DeleteVector)> = None;
        for (file_path, pos) in file_paths.iter().zip(positions.iter()) {
            let Some(file_path) = file_path else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Invalid position delete file '{delete_file_path}': null file_path \
                         value (the file_path column is required)"
                    ),
                ));
            };
            let Some(pos) = pos else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Invalid position delete file '{delete_file_path}': null position \
                         value for data file '{file_path}' (the pos column is required)"
                    ),
                ));
            };

            match &mut current {
                Some((path, vector)) if *path == file_path => {
                    vector.insert(checked_delete_position(delete_file_path, file_path, pos)?);
                }
                _ => {
                    if let Some((path, vector)) = current.take() {
                        *result.entry(path.to_string()).or_default() |= vector;
                    }
                    let mut vector =
                        std::mem::take(result.entry(file_path.to_string()).or_default());
                    vector.insert(checked_delete_position(delete_file_path, file_path, pos)?);
                    current = Some((file_path, vector));
                }
            }
        }
        if let Some((path, vector)) = current.take() {
            *result.entry(path.to_string()).or_default() |= vector;
        }
    }

    Ok(result)
}
