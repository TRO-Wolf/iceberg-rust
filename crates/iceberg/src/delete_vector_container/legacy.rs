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

use arrow_array::Array;
use futures::StreamExt;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::io::FileIO;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::spec::{DataContentType, DataFile, PrimitiveLiteral, Struct};
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
    pub data_sequence_number: Option<i64>,
}

/// Positions grouped by data-file path, one projected parquet read of `delete`.
pub async fn load_legacy_positions_by_path(
    file_io: &FileIO,
    delete: &LegacyPositionDelete,
) -> Result<HashMap<String, Vec<u64>>> {
    if delete.file_scoped && delete.touched.len() != 1 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Position delete '{}' is file-scoped but names {} data files",
                delete.file.file_path(),
                delete.touched.len()
            ),
        ));
    }
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
    let mut out: HashMap<String, Vec<u64>> = HashMap::new();
    let file_scoped_key = if delete.file_scoped {
        referenced_location_ref(&delete.file)
            .map(str::to_string)
            .or_else(|| delete.touched.first().cloned())
    } else {
        None
    };
    let touched: HashSet<&str> = delete.touched.iter().map(String::as_str).collect();
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
            push_file_scoped_positions(&mut out, key, pos_col, delete.file.file_path())?;
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
        push_partition_scoped_positions(
            &mut out,
            &touched,
            path_col,
            pos_col,
            delete.file.file_path(),
        )?;
    }
    Ok(out)
}

/// Positions of `touched_path` from a loaded [`load_legacy_positions_by_path`] index.
pub async fn load_legacy_positions(
    file_io: &FileIO,
    delete: &LegacyPositionDelete,
    touched_path: &str,
) -> Result<Vec<u64>> {
    let mut by_path = load_legacy_positions_by_path(file_io, delete).await?;
    Ok(by_path.remove(touched_path).unwrap_or_default())
}

fn push_file_scoped_positions(
    out: &mut HashMap<String, Vec<u64>>,
    key: &str,
    pos_col: &arrow_array::Int64Array,
    delete_path: &str,
) -> Result<()> {
    if pos_col.null_count() == 0 {
        let values = pos_col.values();
        if let Some(positions) = out.get_mut(key) {
            positions.reserve(values.len());
            for &pos in values {
                positions.push(u64::try_from(pos).map_err(|_| negative_pos(delete_path, pos))?);
            }
        } else {
            let mut positions = Vec::with_capacity(values.len());
            for &pos in values {
                positions.push(u64::try_from(pos).map_err(|_| negative_pos(delete_path, pos))?);
            }
            out.insert(key.to_string(), positions);
        }
        return Ok(());
    }
    for row in 0..pos_col.len() {
        if pos_col.is_null(row) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Position delete '{delete_path}' has a null pos at row {row}"),
            ));
        }
        let pos = pos_col.value(row);
        let pos_u64 = u64::try_from(pos).map_err(|_| negative_pos(delete_path, pos))?;
        if let Some(positions) = out.get_mut(key) {
            positions.push(pos_u64);
        } else {
            out.insert(key.to_string(), vec![pos_u64]);
        }
    }
    Ok(())
}

fn push_partition_scoped_positions(
    out: &mut HashMap<String, Vec<u64>>,
    touched: &HashSet<&str>,
    path_col: &arrow_array::StringArray,
    pos_col: &arrow_array::Int64Array,
    delete_path: &str,
) -> Result<()> {
    for row in 0..pos_col.len() {
        if path_col.is_null(row) || pos_col.is_null(row) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Position delete '{delete_path}' has a null file_path/pos at row {row}"),
            ));
        }
        let path = path_col.value(row);
        if !touched.contains(path) {
            continue;
        }
        let pos = pos_col.value(row);
        let pos_u64 = u64::try_from(pos).map_err(|_| negative_pos(delete_path, pos))?;
        if let Some(positions) = out.get_mut(path) {
            positions.push(pos_u64);
        } else {
            out.insert(path.to_string(), vec![pos_u64]);
        }
    }
    Ok(())
}

pub(super) fn referenced_location_ref(delete_file: &DataFile) -> Option<&str> {
    if delete_file.content_type() == DataContentType::EqualityDeletes {
        return None;
    }
    if let Some(referenced) = delete_file.referenced_data_file_ref() {
        return Some(referenced);
    }
    let lower = delete_file
        .lower_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)?;
    let upper = delete_file
        .upper_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)?;
    match (lower.literal(), upper.literal()) {
        (PrimitiveLiteral::String(lower), PrimitiveLiteral::String(upper)) if lower == upper => {
            Some(lower.as_str())
        }
        _ => None,
    }
}

fn negative_pos(delete_path: &str, pos: i64) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("Position delete '{delete_path}' has a negative pos {pos}"),
    )
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

#[cfg(test)]
pub(super) fn partition_matches(delete_file: &DataFile, spec_id: i32, partition: &Struct) -> bool {
    delete_file.partition_spec_id() == spec_id && delete_file.partition() == partition
}

pub(super) struct PendingLegacy {
    pub(super) file: DataFile,
    pub(super) seq: Option<i64>,
    pub(super) referenced: Option<String>,
}

pub(super) fn finalize_legacy(
    pending: Vec<PendingLegacy>,
    touched_paths: &HashSet<&str>,
    known_partitions: &HashMap<String, (i32, Struct)>,
    extra_partitions: &HashMap<String, (i32, Struct)>,
) -> Vec<Arc<LegacyPositionDelete>> {
    let mut by_partition: HashMap<(i32, Struct), Vec<&str>> = HashMap::new();
    for path in touched_paths {
        let Some((spec_id, partition)) = extra_partitions
            .get(*path)
            .or_else(|| known_partitions.get(*path))
        else {
            continue;
        };
        by_partition
            .entry((*spec_id, partition.clone()))
            .or_default()
            .push(*path);
    }
    let mut out = Vec::new();
    for item in pending {
        if let Some(referenced) = item.referenced {
            if touched_paths.contains(referenced.as_str()) {
                out.push(Arc::new(LegacyPositionDelete {
                    file: item.file,
                    touched: vec![referenced],
                    file_scoped: true,
                    data_sequence_number: item.seq,
                }));
            }
            continue;
        }
        let key = (item.file.partition_spec_id(), item.file.partition().clone());
        let mut touched = Vec::new();
        if let Some(paths) = by_partition.get(&key) {
            for path in paths {
                if file_path_bounds_admit(&item.file, path) {
                    touched.push((*path).to_string());
                }
            }
        }
        if !touched.is_empty() {
            out.push(Arc::new(LegacyPositionDelete {
                file: item.file,
                touched,
                file_scoped: false,
                data_sequence_number: item.seq,
            }));
        }
    }
    out
}

#[cfg(test)]
mod scope_tests {
    use std::collections::{HashMap, HashSet};

    use super::{PendingLegacy, file_path_bounds_admit, finalize_legacy, partition_matches};
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, Struct,
    };

    fn parquet_delete(spec_id: i32, part: i64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("s3://b/d.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(spec_id)
            .partition(Struct::from_iter([Some(Literal::long(part))]))
            .build()
            .expect("delete file")
    }

    fn parquet_delete_bounds(spec_id: i32, part: i64, lower: &str, upper: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("s3://b/d.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(spec_id)
            .partition(Struct::from_iter([Some(Literal::long(part))]))
            .lower_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string(lower),
            )]))
            .upper_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string(upper),
            )]))
            .build()
            .expect("delete file")
    }

    #[test]
    fn partition_scoped_delete_is_not_collected_for_another_partition() {
        let file = parquet_delete(0, 1);
        assert!(
            !partition_matches(&file, 0, &Struct::from_iter([Some(Literal::long(0))])),
            "partition 1 does not match 0"
        );
        let pending = vec![PendingLegacy {
            file,
            seq: Some(1),
            referenced: None,
        }];
        let touched = HashSet::from(["s3://b/a.parquet"]);
        let known = HashMap::from([(
            "s3://b/a.parquet".to_string(),
            (0i32, Struct::from_iter([Some(Literal::long(0))])),
        )]);
        let out = finalize_legacy(pending, &touched, &known, &HashMap::new());
        assert!(
            out.is_empty(),
            "a partition-1 delete must not apply to a file in partition 0"
        );
    }

    #[test]
    fn partition_scoped_delete_is_not_collected_under_another_spec() {
        let file = parquet_delete(1, 0);
        assert!(
            !partition_matches(&file, 0, &Struct::from_iter([Some(Literal::long(0))])),
            "spec 1 does not match spec 0"
        );
        let pending = vec![PendingLegacy {
            file,
            seq: Some(1),
            referenced: None,
        }];
        let touched = HashSet::from(["s3://b/a.parquet"]);
        let known = HashMap::from([(
            "s3://b/a.parquet".to_string(),
            (0i32, Struct::from_iter([Some(Literal::long(0))])),
        )]);
        let out = finalize_legacy(pending, &touched, &known, &HashMap::new());
        assert!(
            out.is_empty(),
            "the same partition value under another spec_id must not match"
        );
    }

    #[test]
    fn bounds_range_excluding_the_path_is_not_admitted() {
        let file = parquet_delete_bounds(0, 0, "s3://b/a.parquet", "s3://b/m.parquet");
        assert!(
            !file_path_bounds_admit(&file, "s3://b/z.parquet"),
            "z is outside a..=m"
        );
        assert!(
            file_path_bounds_admit(&file, "s3://b/c.parquet"),
            "c is inside a..=m"
        );
        let pending = vec![PendingLegacy {
            file,
            seq: Some(1),
            referenced: None,
        }];
        let touched = HashSet::from(["s3://b/z.parquet"]);
        let known = HashMap::from([(
            "s3://b/z.parquet".to_string(),
            (0i32, Struct::from_iter([Some(Literal::long(0))])),
        )]);
        let out = finalize_legacy(pending, &touched, &known, &HashMap::new());
        assert!(
            out.is_empty(),
            "a bounds range that excludes the path must not admit it"
        );
    }

    #[test]
    fn bounds_range_including_the_path_is_admitted() {
        let file = parquet_delete_bounds(0, 0, "s3://b/a.parquet", "s3://b/m.parquet");
        let pending = vec![PendingLegacy {
            file,
            seq: Some(1),
            referenced: None,
        }];
        let touched = HashSet::from(["s3://b/c.parquet"]);
        let known = HashMap::from([(
            "s3://b/c.parquet".to_string(),
            (0i32, Struct::from_iter([Some(Literal::long(0))])),
        )]);
        let out = finalize_legacy(pending, &touched, &known, &HashMap::new());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].touched, vec!["s3://b/c.parquet".to_string()]);
        assert!(!out[0].file_scoped);
    }
}
