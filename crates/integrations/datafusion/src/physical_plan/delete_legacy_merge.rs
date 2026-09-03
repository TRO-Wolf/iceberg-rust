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

use datafusion::common::{DataFusionError, Result as DFResult};
use iceberg::arrow::delete_file_loader::BasicDeleteFileLoader;
use iceberg::delete_vector_container::{
    DvContainerClose, close_touched_dv_containers_with_partitions,
};
use iceberg::spec::{DataFile, Struct, is_deletion_vector, referenced_data_file_location};
use iceberg::table::Table;

use super::cow_affected::{live_data_file_partitions, snapshot_for_scan};
use crate::to_datafusion_error;

type LegacyPositionDeletes = Vec<(Option<String>, i32, Struct, Option<i64>, DataFile)>;

async fn live_legacy_position_deletes(
    table: &Table,
    snapshot_id: Option<i64>,
) -> DFResult<LegacyPositionDeletes> {
    let metadata = table.metadata();
    let mut live = LegacyPositionDeletes::new();
    let Some(snapshot) = snapshot_for_scan(table, snapshot_id) else {
        return Ok(live);
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await
        .map_err(to_datafusion_error)?;
    for manifest_entry in manifest_list.entries() {
        if manifest_entry.content != iceberg::spec::ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_entry
            .load_manifest(table.file_io())
            .await
            .map_err(to_datafusion_error)?;
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            if let Some(LiveDeleteKind::LegacyPositionDelete) = classify_live_delete(df) {
                live.push(legacy_position_delete_entry(df, entry.sequence_number()));
            }
        }
    }
    Ok(live)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveDeleteKind {
    DeletionVector,
    LegacyPositionDelete,
}

fn classify_live_delete(delete_file: &DataFile) -> Option<LiveDeleteKind> {
    if delete_file.content_type() != iceberg::spec::DataContentType::PositionDeletes {
        return None;
    }
    if is_deletion_vector(delete_file) {
        Some(LiveDeleteKind::DeletionVector)
    } else {
        Some(LiveDeleteKind::LegacyPositionDelete)
    }
}

fn legacy_position_delete_entry(
    delete_file: &DataFile,
    sequence_number: Option<i64>,
) -> (Option<String>, i32, Struct, Option<i64>, DataFile) {
    (
        referenced_data_file_location(delete_file),
        delete_file.partition_spec_id(),
        delete_file.partition().clone(),
        sequence_number,
        delete_file.clone(),
    )
}

fn legacy_position_delete_applies(
    delete: &(Option<String>, i32, Struct, Option<i64>),
    data_file_path: &str,
    data_spec_id: i32,
    data_partition: &Struct,
    data_seq: Option<i64>,
) -> bool {
    let (referenced, delete_spec_id, delete_partition, delete_seq) = delete;
    let scope_matches = match referenced {
        Some(referenced) => referenced == data_file_path,
        None => *delete_spec_id == data_spec_id && delete_partition == data_partition,
    };
    scope_matches
        && match (delete_seq, data_seq) {
            (Some(delete_seq), Some(data_seq)) => *delete_seq >= data_seq,
            _ => true,
        }
}

pub(crate) async fn write_deletion_vectors(
    table: &Table,
    pairs: &[(String, i64)],
    scan_snapshot_id: Option<i64>,
) -> DFResult<DvContainerClose> {
    let path_to_partition = live_data_file_partitions(table, scan_snapshot_id).await?;
    let live = live_legacy_position_deletes(table, scan_snapshot_id).await?;
    let mut resolved: Vec<(&str, i32, Struct, Option<i64>)> = Vec::new();
    let mut seen = HashSet::new();
    for (path, _) in pairs {
        if !seen.insert(path.as_str()) {
            continue;
        }
        let (spec_id, partition, data_seq) = path_to_partition.get(path).cloned().ok_or_else(|| DataFusionError::Internal(format!("deletion-vector: data file `{path}` is not a live file of the current snapshot, so its partition cannot be resolved")))?;
        resolved.push((path.as_str(), spec_id, partition, data_seq));
    }
    let mut legacy_by_path: HashMap<String, Vec<u64>> = HashMap::new();
    let mut file_scoped_to_remove: Vec<DataFile> = Vec::new();
    let mut seen_remove: HashSet<String> = HashSet::new();
    for (path, spec_id, partition, data_seq) in &resolved {
        for legacy in &live {
            let key = (legacy.0.clone(), legacy.1, legacy.2.clone(), legacy.3);
            if !legacy_position_delete_applies(&key, path, *spec_id, partition, *data_seq) {
                continue;
            }
            let delete_file = &legacy.4;
            let pairs =
                BasicDeleteFileLoader::load_position_delete_pairs(table.file_io(), delete_file)
                    .await
                    .map_err(to_datafusion_error)?;
            for (fp, pos) in pairs {
                if fp == *path {
                    let pos_u64 = u64::try_from(pos).map_err(|_| {
                        DataFusionError::Internal(format!(
                            "deletion-vector: negative row position {pos} for data file `{path}`"
                        ))
                    })?;
                    legacy_by_path
                        .entry(path.to_string())
                        .or_default()
                        .push(pos_u64);
                }
            }
            if referenced_data_file_location(delete_file).is_some()
                && seen_remove.insert(delete_file.file_path().to_string())
            {
                file_scoped_to_remove.push(delete_file.clone());
            }
        }
    }
    let mut new_positions: HashMap<String, Vec<u64>> = HashMap::new();
    for (path, position) in pairs {
        let position = u64::try_from(*position).map_err(|_| {
            DataFusionError::Internal(format!(
                "deletion-vector: negative row position {position} for data file `{path}`"
            ))
        })?;
        new_positions
            .entry(path.clone())
            .or_default()
            .push(position);
    }
    for (path, mut vec) in legacy_by_path {
        new_positions.entry(path).or_default().append(&mut vec);
    }
    let known_partitions: HashMap<String, (i32, Struct)> = resolved
        .into_iter()
        .map(|(path, spec_id, partition, _)| (path.to_string(), (spec_id, partition)))
        .collect();
    let mut close = close_touched_dv_containers_with_partitions(
        table,
        &new_positions,
        scan_snapshot_id,
        &known_partitions,
    )
    .await
    .map_err(to_datafusion_error)?;
    close.removed.extend(file_scoped_to_remove);
    Ok(close)
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Datum, Literal, Struct as IcebergStruct,
    };

    use super::*;

    fn delete_file_of(content: DataContentType, file_format: DataFileFormat) -> DataFile {
        let mut builder = DataFileBuilder::default();
        builder
            .content(content)
            .file_path("s3://b/d".to_string())
            .file_format(file_format)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0);
        if content == DataContentType::EqualityDeletes {
            builder.equality_ids(Some(vec![1]));
        }
        if file_format == DataFileFormat::Puffin {
            builder
                .content_offset(Some(4))
                .content_size_in_bytes(Some(40))
                .referenced_data_file(Some("s3://b/a.parquet".to_string()));
        }
        builder.build().expect("build the delete file")
    }

    #[test]
    fn test_classify_live_delete_ignores_equality_deletes() {
        assert_eq!(
            classify_live_delete(&delete_file_of(
                DataContentType::EqualityDeletes,
                DataFileFormat::Parquet
            )),
            None
        );
        assert_eq!(
            classify_live_delete(&delete_file_of(
                DataContentType::PositionDeletes,
                DataFileFormat::Puffin
            )),
            Some(LiveDeleteKind::DeletionVector)
        );
        assert_eq!(
            classify_live_delete(&delete_file_of(
                DataContentType::PositionDeletes,
                DataFileFormat::Parquet
            )),
            Some(LiveDeleteKind::LegacyPositionDelete)
        );
        assert_eq!(
            classify_live_delete(&delete_file_of(
                DataContentType::Data,
                DataFileFormat::Parquet
            )),
            None
        );
    }

    #[test]
    fn test_legacy_delete_entry_derives_the_name_from_equal_file_path_bounds() {
        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("s3://b/pos-del.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(7)
            .partition(IcebergStruct::from_iter([Some(Literal::long(999))]))
            .lower_bounds(std::collections::HashMap::from([(
                iceberg::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("s3://b/a.parquet"),
            )]))
            .upper_bounds(std::collections::HashMap::from([(
                iceberg::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("s3://b/a.parquet"),
            )]))
            .build()
            .expect("build a bounds-scoped position delete");
        let entry = legacy_position_delete_entry(&delete_file, Some(3));
        assert_eq!(entry.0.as_deref(), Some("s3://b/a.parquet"));
        assert_eq!(entry.1, 7);
        assert_eq!(
            entry.2,
            IcebergStruct::from_iter([Some(Literal::long(999))])
        );
        assert_eq!(entry.3, Some(3));
    }

    #[test]
    fn test_legacy_delete_named_by_path_applies_across_partitions() {
        let delete = (
            Some("s3://b/a.parquet".to_string()),
            7,
            IcebergStruct::from_iter([Some(Literal::long(999))]),
            Some(1),
        );
        let data_partition = IcebergStruct::from_iter([Some(Literal::long(0))]);
        assert!(legacy_position_delete_applies(
            &delete,
            "s3://b/a.parquet",
            0,
            &data_partition,
            Some(1)
        ));
        assert!(!legacy_position_delete_applies(
            &delete,
            "s3://b/other.parquet",
            7,
            &delete.2,
            Some(1)
        ));
    }

    #[test]
    fn test_legacy_delete_without_a_name_applies_by_partition() {
        let partition = IcebergStruct::from_iter([Some(Literal::long(0))]);
        let delete = (None, 0, partition.clone(), Some(1));
        assert!(legacy_position_delete_applies(
            &delete,
            "s3://b/a.parquet",
            0,
            &partition,
            Some(1)
        ));
        let other = IcebergStruct::from_iter([Some(Literal::long(1))]);
        assert!(!legacy_position_delete_applies(
            &delete,
            "s3://b/a.parquet",
            0,
            &other,
            Some(1)
        ));
        assert!(!legacy_position_delete_applies(
            &delete,
            "s3://b/a.parquet",
            1,
            &partition,
            Some(1)
        ));
    }

    #[test]
    fn test_legacy_delete_older_than_the_data_file_does_not_apply() {
        let partition = IcebergStruct::from_iter([Some(Literal::long(0))]);
        let delete = (None, 0, partition.clone(), Some(1));
        assert!(!legacy_position_delete_applies(
            &delete,
            "s3://b/new.parquet",
            0,
            &partition,
            Some(2)
        ));
        assert!(legacy_position_delete_applies(
            &delete,
            "s3://b/old.parquet",
            0,
            &partition,
            Some(1)
        ));
        assert!(legacy_position_delete_applies(
            &delete,
            "s3://b/x.parquet",
            0,
            &partition,
            None
        ));
        let named = (
            Some("s3://b/new.parquet".to_string()),
            0,
            partition.clone(),
            Some(1),
        );
        assert!(!legacy_position_delete_applies(
            &named,
            "s3://b/new.parquet",
            0,
            &partition,
            Some(2)
        ));
        assert!(legacy_position_delete_applies(
            &named,
            "s3://b/new.parquet",
            0,
            &partition,
            Some(1)
        ));
    }
}
