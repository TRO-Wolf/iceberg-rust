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

use datafusion::common::{DataFusionError, Result as DFResult};
use futures::{StreamExt, TryStreamExt, stream};
use iceberg::arrow::delete_file_loader::load_position_deletes_by_path;
use iceberg::delete_vector_container::{
    DvContainerClose, close_touched_dv_containers_with_partitions,
};
use iceberg::spec::{
    DataFile, Manifest, ManifestContentType, Struct, is_deletion_vector,
    referenced_data_file_location,
};
use iceberg::table::Table;

use super::cow_affected::{live_data_file_partitions, snapshot_for_scan};
use crate::to_datafusion_error;

const DV_IO_CONCURRENCY: usize = 8;

struct LiveLegacy<'a> {
    file: &'a DataFile,
    seq: Option<i64>,
    referenced: Option<String>,
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

fn seq_applies(delete_seq: Option<i64>, data_seq: Option<i64>) -> bool {
    match (delete_seq, data_seq) {
        (Some(delete_seq), Some(data_seq)) => delete_seq >= data_seq,
        _ => true,
    }
}

fn legacy_position_delete_applies(
    item: &LiveLegacy<'_>,
    data_file_path: &str,
    data_spec_id: i32,
    data_partition: &Struct,
    data_seq: Option<i64>,
) -> bool {
    let scope_matches = match item.referenced.as_deref() {
        Some(referenced) => referenced == data_file_path,
        None => {
            item.file.partition_spec_id() == data_spec_id && item.file.partition() == data_partition
        }
    };
    scope_matches && seq_applies(item.seq, data_seq)
}

async fn load_delete_manifests(table: &Table, snapshot_id: Option<i64>) -> DFResult<Vec<Manifest>> {
    let Some(snapshot) = snapshot_for_scan(table, snapshot_id) else {
        return Ok(Vec::new());
    };
    let file_io = table.file_io().clone();
    let manifest_list = snapshot
        .load_manifest_list(&file_io, table.metadata())
        .await
        .map_err(to_datafusion_error)?;
    let files: Vec<_> = manifest_list
        .entries()
        .iter()
        .filter(|manifest_file| manifest_file.content == ManifestContentType::Deletes)
        .cloned()
        .collect();
    stream::iter(files)
        .map(move |manifest_file| {
            let file_io = file_io.clone();
            async move {
                manifest_file
                    .load_manifest(&file_io)
                    .await
                    .map_err(to_datafusion_error)
            }
        })
        .buffer_unordered(DV_IO_CONCURRENCY)
        .try_collect()
        .await
}

pub(crate) async fn write_deletion_vectors(
    table: &Table,
    pairs: &[(String, i64)],
    scan_snapshot_id: Option<i64>,
) -> DFResult<DvContainerClose> {
    let mut seen_paths = HashSet::new();
    let mut unique_paths: Vec<&str> = Vec::new();
    for (path, _) in pairs {
        if seen_paths.insert(path.as_str()) {
            unique_paths.push(path.as_str());
        }
    }
    let path_to_partition =
        live_data_file_partitions(table, scan_snapshot_id, Some(&seen_paths)).await?;
    let mut resolved: Vec<(&str, i32, Struct, Option<i64>)> = Vec::new();
    for path in &unique_paths {
        let (spec_id, partition, data_seq) = path_to_partition.get(*path).cloned().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "deletion-vector: data file `{path}` is not a live file of the current snapshot, so its partition cannot be resolved"
            ))
        })?;
        resolved.push((*path, spec_id, partition, data_seq));
    }

    let manifests = load_delete_manifests(table, scan_snapshot_id).await?;
    let mut live: Vec<LiveLegacy<'_>> = Vec::new();
    let mut file_scoped: HashMap<String, Vec<usize>> = HashMap::new();
    let mut partition_scoped: Vec<usize> = Vec::new();
    for manifest in &manifests {
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let df = entry.data_file();
            if classify_live_delete(df) != Some(LiveDeleteKind::LegacyPositionDelete) {
                continue;
            }
            let idx = live.len();
            let referenced = referenced_data_file_location(df);
            if let Some(path) = referenced.as_deref() {
                file_scoped.entry(path.to_string()).or_default().push(idx);
            } else {
                partition_scoped.push(idx);
            }
            live.push(LiveLegacy {
                file: df,
                seq: entry.sequence_number(),
                referenced,
            });
        }
    }

    let mut to_load: Vec<&DataFile> = Vec::new();
    let mut seen_load: HashSet<&str> = HashSet::new();
    for (path, spec_id, partition, data_seq) in &resolved {
        if let Some(idxs) = file_scoped.get(*path) {
            for &idx in idxs {
                let item = &live[idx];
                if !legacy_position_delete_applies(item, path, *spec_id, partition, *data_seq) {
                    continue;
                }
                if seen_load.insert(item.file.file_path()) {
                    to_load.push(item.file);
                }
            }
        }
        for &idx in &partition_scoped {
            let item = &live[idx];
            if !legacy_position_delete_applies(item, path, *spec_id, partition, *data_seq) {
                continue;
            }
            if seen_load.insert(item.file.file_path()) {
                to_load.push(item.file);
            }
        }
    }

    let file_io = table.file_io().clone();
    let to_load_owned: Vec<DataFile> = to_load.into_iter().cloned().collect();
    let loaded: HashMap<String, Arc<HashMap<String, Vec<u64>>>> =
        stream::iter(to_load_owned.into_iter().map(|delete_file| {
            let file_io = file_io.clone();
            async move {
                let path = delete_file.file_path().to_string();
                let index = load_position_deletes_by_path(&file_io, &delete_file)
                    .await
                    .map_err(to_datafusion_error)?;
                Ok::<_, DataFusionError>((path, Arc::new(index)))
            }
        }))
        .buffer_unordered(DV_IO_CONCURRENCY)
        .try_collect()
        .await?;

    let mut legacy_by_path: HashMap<String, Vec<u64>> = HashMap::new();
    let mut file_scoped_to_remove: Vec<DataFile> = Vec::new();
    let mut seen_remove: HashSet<String> = HashSet::new();
    for (path, spec_id, partition, data_seq) in &resolved {
        let mut extra: Vec<u64> = Vec::new();
        {
            let mut apply_item = |item: &LiveLegacy<'_>| {
                if !legacy_position_delete_applies(item, path, *spec_id, partition, *data_seq) {
                    return;
                }
                if let Some(index) = loaded.get(item.file.file_path())
                    && let Some(positions) = index.get(*path)
                {
                    extra.extend(positions.iter().copied());
                }
                if item.referenced.is_some() {
                    let delete_path = item.file.file_path();
                    if !seen_remove.contains(delete_path) {
                        seen_remove.insert(delete_path.to_string());
                        file_scoped_to_remove.push(item.file.clone());
                    }
                }
            };
            if let Some(idxs) = file_scoped.get(*path) {
                for &idx in idxs {
                    apply_item(&live[idx]);
                }
            }
            for &idx in &partition_scoped {
                let item = &live[idx];
                if item.file.partition_spec_id() == *spec_id && item.file.partition() == partition {
                    apply_item(item);
                }
            }
        }
        if !extra.is_empty() {
            legacy_by_path
                .entry(path.to_string())
                .or_default()
                .append(&mut extra);
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

    fn legacy_item<'a>(
        file: &'a DataFile,
        seq: Option<i64>,
        referenced: Option<&str>,
    ) -> LiveLegacy<'a> {
        LiveLegacy {
            file,
            seq,
            referenced: referenced.map(str::to_string),
        }
    }

    #[test]
    fn test_classify_live_delete_ignores_equality_deletes() {
        assert_eq!(
            classify_live_delete(&delete_file_of(
                DataContentType::EqualityDeletes,
                DataFileFormat::Parquet
            )),
            None,
            "equality deletes are not legacy position deletes"
        );
        assert_eq!(
            classify_live_delete(&delete_file_of(
                DataContentType::PositionDeletes,
                DataFileFormat::Puffin
            )),
            Some(LiveDeleteKind::DeletionVector),
            "a puffin position delete is a deletion vector"
        );
        assert_eq!(
            classify_live_delete(&delete_file_of(
                DataContentType::PositionDeletes,
                DataFileFormat::Parquet
            )),
            Some(LiveDeleteKind::LegacyPositionDelete),
            "a parquet position delete is a live legacy delete"
        );
        assert_eq!(
            classify_live_delete(&delete_file_of(
                DataContentType::Data,
                DataFileFormat::Parquet
            )),
            None,
            "data files are not delete files"
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
        assert_eq!(
            referenced_data_file_location(&delete_file).as_deref(),
            Some("s3://b/a.parquet"),
            "equal file_path bounds name the data file"
        );
        assert_eq!(
            delete_file.partition_spec_id(),
            7,
            "bounds-scoped delete keeps its spec id"
        );
        assert_eq!(
            delete_file.partition(),
            &IcebergStruct::from_iter([Some(Literal::long(999))]),
            "bounds-scoped delete keeps its partition"
        );
    }

    #[test]
    fn test_legacy_delete_named_by_path_applies_across_partitions() {
        let file = delete_file_of(DataContentType::PositionDeletes, DataFileFormat::Parquet);
        let named = legacy_item(&file, Some(1), Some("s3://b/a.parquet"));
        let data_partition = IcebergStruct::from_iter([Some(Literal::long(0))]);
        let delete_partition = IcebergStruct::from_iter([Some(Literal::long(999))]);
        assert!(
            legacy_position_delete_applies(&named, "s3://b/a.parquet", 0, &data_partition, Some(1)),
            "a named delete matches on path alone"
        );
        assert!(
            !legacy_position_delete_applies(
                &named,
                "s3://b/other.parquet",
                7,
                &delete_partition,
                Some(1)
            ),
            "a named delete does not apply to a different path"
        );
    }

    #[test]
    fn test_legacy_delete_without_a_name_applies_by_partition() {
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::PositionDeletes)
            .file_path("s3://b/d".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(IcebergStruct::from_iter([Some(Literal::long(0))]));
        let file = builder.build().expect("build the delete file");
        let item = legacy_item(&file, Some(1), None);
        let partition = IcebergStruct::from_iter([Some(Literal::long(0))]);
        assert!(
            legacy_position_delete_applies(&item, "s3://b/a.parquet", 0, &partition, Some(1)),
            "a partition-scoped delete applies in its partition"
        );
        let other = IcebergStruct::from_iter([Some(Literal::long(1))]);
        assert!(
            !legacy_position_delete_applies(&item, "s3://b/a.parquet", 0, &other, Some(1)),
            "a partition-scoped delete does not apply in another partition"
        );
        assert!(
            !legacy_position_delete_applies(&item, "s3://b/a.parquet", 1, &partition, Some(1)),
            "a partition-scoped delete does not apply under another spec"
        );
    }

    #[test]
    fn test_legacy_delete_older_than_the_data_file_does_not_apply() {
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::PositionDeletes)
            .file_path("s3://b/d".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(IcebergStruct::from_iter([Some(Literal::long(0))]));
        let file = builder.build().expect("build the delete file");
        let item = legacy_item(&file, Some(1), None);
        let partition = IcebergStruct::from_iter([Some(Literal::long(0))]);
        assert!(
            !legacy_position_delete_applies(&item, "s3://b/new.parquet", 0, &partition, Some(2)),
            "delete_seq < data_seq does not apply"
        );
        assert!(
            legacy_position_delete_applies(&item, "s3://b/old.parquet", 0, &partition, Some(1)),
            "delete_seq == data_seq applies"
        );
        assert!(
            legacy_position_delete_applies(&item, "s3://b/x.parquet", 0, &partition, None),
            "a missing data_seq applies"
        );
        let named = legacy_item(&file, Some(1), Some("s3://b/new.parquet"));
        assert!(
            !legacy_position_delete_applies(&named, "s3://b/new.parquet", 0, &partition, Some(2)),
            "a named delete still honors delete_seq < data_seq"
        );
        assert!(
            legacy_position_delete_applies(&named, "s3://b/new.parquet", 0, &partition, Some(1)),
            "a named delete applies when delete_seq >= data_seq"
        );
    }
}
