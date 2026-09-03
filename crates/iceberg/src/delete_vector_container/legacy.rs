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

use arrow_array::Array;
use futures::StreamExt;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::delete_file_index::referenced_data_file_location;
use crate::io::FileIO;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::spec::{DataFile, PrimitiveLiteral, Struct};
use crate::{Error, ErrorKind, Result};

/// Live parquet position-delete that names a touched data file.
#[derive(Debug, Clone)]
pub struct LegacyPositionDelete {
    /// The delete file.
    pub file: DataFile,
    /// Touched data-file paths this delete applies to.
    pub touched: Vec<String>,
    /// Whether the delete is file-scoped.
    pub file_scoped: bool,
    /// Sequence number of the delete file.
    pub data_sequence_number: i64,
}

/// Positions of `touched_path` in a live parquet position-delete, projected to `pos` (and `file_path` when not file-scoped).
pub async fn load_legacy_positions(
    file_io: &FileIO,
    delete: &LegacyPositionDelete,
    touched_path: &str,
) -> Result<Vec<u64>> {
    if !delete
        .touched
        .iter()
        .any(|path| path.as_str() == touched_path)
    {
        return Ok(Vec::new());
    }
    let mut by_path = load_legacy_index(file_io, delete).await?;
    Ok(by_path.remove(touched_path).unwrap_or_default())
}

pub(super) async fn load_legacy_index(
    file_io: &FileIO,
    delete: &LegacyPositionDelete,
) -> Result<HashMap<String, Vec<u64>>> {
    let loader = BasicDeleteFileLoader::new(file_io.clone());
    let field_ids: &[i32] = if delete.file_scoped {
        &[RESERVED_FIELD_ID_DELETE_FILE_POS]
    } else {
        &[
            RESERVED_FIELD_ID_DELETE_FILE_PATH,
            RESERVED_FIELD_ID_DELETE_FILE_POS,
        ]
    };
    let mut stream = loader
        .parquet_to_batch_stream_with_projection(
            delete.file.file_path(),
            delete.file.file_size_in_bytes,
            Some(field_ids),
        )
        .await?;
    let cap: usize = delete.file.record_count().try_into().unwrap_or(0);
    let mut out: HashMap<String, Vec<u64>> = HashMap::new();
    let file_scoped_key = if delete.file_scoped {
        delete
            .touched
            .first()
            .cloned()
            .or_else(|| referenced_data_file_location(&delete.file))
    } else {
        None
    };
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let mut path_idx: Option<usize> = None;
        let mut pos_idx: Option<usize> = None;
        for (idx, field) in batch.schema().fields().iter().enumerate() {
            if let Some(id_str) = field.metadata().get(PARQUET_FIELD_ID_META_KEY)
                && let Ok(id) = id_str.parse::<i32>()
            {
                if id == RESERVED_FIELD_ID_DELETE_FILE_PATH {
                    path_idx = Some(idx);
                } else if id == RESERVED_FIELD_ID_DELETE_FILE_POS {
                    pos_idx = Some(idx);
                }
            }
        }
        let pos_idx = pos_idx.ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Position delete '{}' is missing the reserved pos column (field id {})",
                    delete.file.file_path(),
                    RESERVED_FIELD_ID_DELETE_FILE_POS
                ),
            )
        })?;
        let pos_col = batch
            .column(pos_idx)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Position delete '{}' pos column is not an int64 array",
                        delete.file.file_path()
                    ),
                )
            })?;
        if delete.file_scoped {
            let key = file_scoped_key.as_deref().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Position delete '{}' is file-scoped but names no data file",
                        delete.file.file_path()
                    ),
                )
            })?;
            for row in 0..batch.num_rows() {
                if pos_col.is_null(row) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Position delete '{}' has a null pos at row {row}",
                            delete.file.file_path()
                        ),
                    ));
                }
                let pos = pos_col.value(row);
                let pos_u64 = u64::try_from(pos).map_err(|_| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Position delete '{}' has a negative pos {pos}",
                            delete.file.file_path()
                        ),
                    )
                })?;
                out.entry(key.to_string())
                    .or_insert_with(|| Vec::with_capacity(cap))
                    .push(pos_u64);
            }
            continue;
        }
        let path_idx = path_idx.ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Position delete '{}' is missing the reserved file_path column (field id {})",
                    delete.file.file_path(),
                    RESERVED_FIELD_ID_DELETE_FILE_PATH
                ),
            )
        })?;
        let path_col = batch
            .column(path_idx)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Position delete '{}' file_path column is not a string array",
                        delete.file.file_path()
                    ),
                )
            })?;
        for row in 0..batch.num_rows() {
            if path_col.is_null(row) || pos_col.is_null(row) {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Position delete '{}' has a null file_path/pos at row {row}",
                        delete.file.file_path()
                    ),
                ));
            }
            let path = path_col.value(row);
            if !delete.touched.iter().any(|touched| touched == path) {
                continue;
            }
            let pos = pos_col.value(row);
            let pos_u64 = u64::try_from(pos).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Position delete '{}' has a negative pos {pos}",
                        delete.file.file_path()
                    ),
                )
            })?;
            out.entry(path.to_string())
                .or_insert_with(|| Vec::with_capacity(cap))
                .push(pos_u64);
        }
    }
    Ok(out)
}

pub(super) fn file_path_bounds_admit(delete_file: &DataFile, path: &str) -> bool {
    let Some(lower) = delete_file
        .lower_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
    else {
        return true;
    };
    let Some(upper) = delete_file
        .upper_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
    else {
        return true;
    };
    match (lower.literal(), upper.literal()) {
        (PrimitiveLiteral::String(lower), PrimitiveLiteral::String(upper)) => {
            lower.as_str() <= path && path <= upper.as_str()
        }
        _ => true,
    }
}

pub(super) fn partition_matches(delete_file: &DataFile, spec_id: i32, partition: &Struct) -> bool {
    delete_file.partition_spec_id() == spec_id && delete_file.partition() == partition
}
