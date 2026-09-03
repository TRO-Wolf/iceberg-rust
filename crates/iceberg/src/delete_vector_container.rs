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

//! Deletion-vector container closure for DataFusion DML.

use std::collections::{HashMap, HashSet};

use futures::{StreamExt, TryStreamExt, stream};
use uuid::Uuid;

use crate::delete_file_index::{is_deletion_vector, referenced_data_file_location};
use crate::delete_vector::{DeleteVector, load_delete_vector};
use crate::io::FileIO;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Manifest, ManifestContentType, ManifestList,
    PartitionKey, PartitionSpec, Struct,
};
use crate::table::Table;
use crate::writer::base_writer::deletion_vector_writer::{DVFileWriter, PreviousDeletes};
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
};
use crate::{Error, ErrorKind, Result};

mod legacy;
pub use legacy::{LegacyPositionDelete, load_legacy_positions};

use self::legacy::{file_path_bounds_admit, load_legacy_index, partition_matches};

/// Concurrent IO bound for DV container close and legacy-position loads.
pub const DV_IO_CONCURRENCY: usize = 8;

/// Result of closing the deletion vectors of one commit.
#[derive(Debug, Default)]
pub struct DvContainerClose {
    /// Replacement DV metadata.
    pub added: Vec<DataFile>,
    /// The superseded blobs, one per touched referenced file. Untouched siblings are not here.
    pub removed: Vec<DataFile>,
    /// Live non-Puffin position deletes that name a touched data file.
    pub legacy_deletes: Vec<LegacyPositionDelete>,
    /// Data sequence numbers of the touched data files.
    pub data_sequence_numbers: HashMap<String, i64>,
}

impl DvContainerClose {
    /// Referenced data-file paths of the replacement blobs this commit adds.
    pub fn referenced_data_files(&self) -> HashSet<String> {
        self.added
            .iter()
            .filter_map(|file| file.referenced_data_file_ref().map(str::to_string))
            .collect()
    }
}

struct BlobWrite {
    referenced: String,
    positions: Vec<u64>,
    previous: Option<DeleteVector>,
    partition_key: PartitionKey,
}

struct PendingLegacy {
    file: DataFile,
    seq: i64,
    referenced: Option<String>,
}

/// Close the deletion vectors touched by `new_positions` against the current snapshot.
pub async fn close_touched_dv_containers(
    table: &Table,
    new_positions: &HashMap<String, Vec<u64>>,
) -> Result<DvContainerClose> {
    close_touched_dv_containers_at(table, new_positions, None).await
}

/// Close touched DV containers against `snapshot_id`, or the current snapshot when `None`.
pub async fn close_touched_dv_containers_at(
    table: &Table,
    new_positions: &HashMap<String, Vec<u64>>,
    snapshot_id: Option<i64>,
) -> Result<DvContainerClose> {
    close_touched_dv_containers_with_partitions(
        table,
        new_positions,
        snapshot_id,
        &HashMap::new(),
        None,
    )
    .await
}

/// Close touched DV containers, taking partitions from `known_partitions` and an optional pre-loaded manifest list.
pub async fn close_touched_dv_containers_with_partitions(
    table: &Table,
    new_positions: &HashMap<String, Vec<u64>>,
    snapshot_id: Option<i64>,
    known_partitions: &HashMap<String, (i32, Struct)>,
    manifest_list: Option<&ManifestList>,
) -> Result<DvContainerClose> {
    if new_positions.is_empty() {
        return Ok(DvContainerClose::default());
    }

    // SQL DML scans first, then closes here. Tests set this to fail after that scan
    // and before write_dv_blobs, which is otherwise unreachable from one execute().
    if std::env::var_os("ICEBERG_FAIL_DV_CONTAINER_BEFORE_WRITE").is_some() {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "injected failure before shared-Puffin replacement write",
        ));
    }

    let owned_list = match manifest_list {
        Some(_) => None,
        None => live_manifest_list(table, snapshot_id).await?,
    };
    let list_ref = manifest_list.or(owned_list.as_ref());
    let touched_paths: HashSet<&str> = new_positions.keys().map(String::as_str).collect();
    let (touched_dvs, pending_legacy) =
        collect_delete_index(table, list_ref, &touched_paths).await?;

    let mut partitions = known_partitions.clone();
    for data_file in &touched_dvs {
        let referenced = referenced_path(data_file)?;
        partitions
            .entry(referenced.to_string())
            .or_insert_with(|| (data_file.partition_spec_id(), data_file.partition().clone()));
    }

    let mut data_sequence_numbers: HashMap<String, i64> = HashMap::new();
    let unresolved: HashSet<&str> = touched_paths
        .iter()
        .copied()
        .filter(|path| !partitions.contains_key(*path))
        .collect();
    let discovered = if unresolved.is_empty() {
        HashMap::new()
    } else {
        collect_live_data_files(table, list_ref, &unresolved).await?
    };
    for (path, (data_file, seq)) in &discovered {
        partitions
            .entry(path.clone())
            .or_insert_with(|| (data_file.partition_spec_id(), data_file.partition().clone()));
        data_sequence_numbers.insert(path.clone(), *seq);
    }

    if !pending_legacy.is_empty() {
        let seq_wanted: HashSet<&str> = touched_paths
            .iter()
            .copied()
            .filter(|path| !data_sequence_numbers.contains_key(*path))
            .collect();
        if !seq_wanted.is_empty() {
            let sequenced = collect_live_data_files(table, list_ref, &seq_wanted).await?;
            for (path, (_data_file, seq)) in sequenced {
                data_sequence_numbers.entry(path).or_insert(seq);
            }
        }
    }

    let legacy_deletes = finalize_legacy(pending_legacy, &touched_paths, &partitions);
    let mut positions_to_write = new_positions.clone();
    let mut file_scoped_to_remove: Vec<DataFile> = Vec::new();
    let mut seen_remove: HashSet<String> = HashSet::new();
    if !legacy_deletes.is_empty() {
        let file_io = table.file_io().clone();
        let loaded: Vec<(LegacyPositionDelete, HashMap<String, Vec<u64>>)> =
            stream::iter(legacy_deletes.iter().cloned().map(|item| {
                let file_io = file_io.clone();
                async move {
                    let index = load_legacy_index(&file_io, &item).await?;
                    Ok::<_, Error>((item, index))
                }
            }))
            .buffer_unordered(DV_IO_CONCURRENCY)
            .try_collect()
            .await?;
        for (item, index) in loaded {
            let mut applied = false;
            for path in &item.touched {
                let data_seq = data_sequence_numbers.get(path).copied();
                if !seq_applies(item.data_sequence_number, data_seq) {
                    continue;
                }
                applied = true;
                if let Some(extra) = index.get(path)
                    && !extra.is_empty()
                {
                    positions_to_write
                        .entry(path.clone())
                        .or_default()
                        .extend(extra.iter().copied());
                }
            }
            if item.file_scoped && applied {
                let delete_path = item.file.file_path();
                if seen_remove.insert(delete_path.to_string()) {
                    file_scoped_to_remove.push(item.file);
                }
            }
        }
    }

    let mut close = DvContainerClose {
        legacy_deletes,
        data_sequence_numbers,
        ..DvContainerClose::default()
    };
    let mut specs: Vec<BlobWrite> = Vec::with_capacity(positions_to_write.len());

    let file_io = table.file_io().clone();
    let loaded: Vec<(DataFile, DeleteVector)> =
        stream::iter(touched_dvs.into_iter().map(|data_file| {
            let file_io = file_io.clone();
            async move {
                let previous = load_delete_vector(&file_io, &data_file).await?;
                Ok::<_, Error>((data_file, previous))
            }
        }))
        .buffer_unordered(DV_IO_CONCURRENCY)
        .try_collect()
        .await?;

    for (data_file, previous) in loaded {
        let referenced = referenced_path(&data_file)?.to_string();
        let added = positions_to_write.get(&referenced).ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!("deletion-vector: untouched blob `{referenced}` entered the rewrite set"),
            )
        })?;
        if added.is_empty() {
            continue;
        }
        specs.push(BlobWrite {
            partition_key: partition_key_for(table, &data_file)?,
            referenced,
            positions: added.clone(),
            previous: Some(previous),
        });
        close.removed.push(data_file);
    }

    let fresh: Vec<&String> = {
        let rewritten: HashSet<&str> = specs.iter().map(|spec| spec.referenced.as_str()).collect();
        positions_to_write
            .iter()
            .filter(|(path, positions)| !positions.is_empty() && !rewritten.contains(path.as_str()))
            .map(|(path, _)| path)
            .collect()
    };

    if !fresh.is_empty() {
        for path in fresh {
            let partition_key = match partitions.get(path) {
                Some((spec_id, partition)) => partition_key_of(table, *spec_id, partition, path)?,
                None => {
                    let data_file = discovered.get(path.as_str()).map(|(file, _)| file).ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "deletion-vector: data file `{path}` is not a live file of the scanned snapshot"
                            ),
                        )
                    })?;
                    partition_key_for(table, data_file)?
                }
            };
            specs.push(BlobWrite {
                partition_key,
                referenced: path.clone(),
                positions: positions_to_write[path].clone(),
                previous: None,
            });
        }
    }

    close.removed.sort_unstable_by(|left, right| {
        (left.file_path(), left.content_offset()).cmp(&(right.file_path(), right.content_offset()))
    });
    close.added = write_dv_blobs(table, specs).await?;
    close.removed.extend(file_scoped_to_remove);

    Ok(close)
}

fn seq_applies(delete_seq: i64, data_seq: Option<i64>) -> bool {
    match data_seq {
        Some(data_seq) => delete_seq >= data_seq,
        None => true,
    }
}

fn snapshot_for_live(
    table: &Table,
    snapshot_id: Option<i64>,
) -> Result<Option<crate::spec::SnapshotRef>> {
    match snapshot_id {
        Some(id) => table
            .metadata()
            .snapshot_by_id(id)
            .cloned()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("deletion-vector: snapshot {id} not found"),
                )
            })
            .map(Some),
        None => Ok(table.metadata().current_snapshot().cloned()),
    }
}

async fn live_manifest_list(
    table: &Table,
    snapshot_id: Option<i64>,
) -> Result<Option<ManifestList>> {
    let Some(snapshot) = snapshot_for_live(table, snapshot_id)? else {
        return Ok(None);
    };
    snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .map(Some)
}

fn manifest_stream<'a>(
    file_io: &FileIO,
    manifest_list: &'a ManifestList,
    content: ManifestContentType,
) -> impl futures::Stream<Item = Result<Manifest>> + 'a {
    let file_io = file_io.clone();
    stream::iter(
        manifest_list
            .entries()
            .iter()
            .filter(move |manifest_file| manifest_file.content == content),
    )
    .map(move |manifest_file| {
        let file_io = file_io.clone();
        async move { manifest_file.load_manifest(&file_io).await }
    })
    .buffer_unordered(DV_IO_CONCURRENCY)
}

fn is_live_dv(entry: &crate::spec::ManifestEntry) -> bool {
    entry.is_alive()
        && entry.data_file().content_type() == DataContentType::PositionDeletes
        && is_deletion_vector(entry.data_file())
}

fn is_live_legacy_position_delete(entry: &crate::spec::ManifestEntry) -> bool {
    entry.is_alive()
        && entry.data_file().content_type() == DataContentType::PositionDeletes
        && !is_deletion_vector(entry.data_file())
}

async fn collect_delete_index(
    table: &Table,
    manifest_list: Option<&ManifestList>,
    touched_paths: &HashSet<&str>,
) -> Result<(Vec<DataFile>, Vec<PendingLegacy>)> {
    let mut touched_dvs = Vec::new();
    let mut pending_legacy = Vec::new();
    let Some(manifest_list) = manifest_list else {
        return Ok((touched_dvs, pending_legacy));
    };
    let mut manifests =
        manifest_stream(table.file_io(), manifest_list, ManifestContentType::Deletes);
    while let Some(manifest) = manifests.try_next().await? {
        for entry in manifest.entries() {
            if is_live_dv(entry) {
                let data_file = entry.data_file();
                let referenced = referenced_path(data_file)?;
                if touched_paths.contains(referenced) {
                    touched_dvs.push(data_file.clone());
                }
                continue;
            }
            if !is_live_legacy_position_delete(entry) {
                continue;
            }
            let data_file = entry.data_file();
            let referenced = referenced_data_file_location(data_file);
            if let Some(path) = referenced.as_deref()
                && !touched_paths.contains(path)
            {
                continue;
            }
            pending_legacy.push(PendingLegacy {
                file: data_file.clone(),
                seq: entry.sequence_number().unwrap_or(0),
                referenced,
            });
        }
    }
    Ok((touched_dvs, pending_legacy))
}

fn finalize_legacy(
    pending: Vec<PendingLegacy>,
    touched_paths: &HashSet<&str>,
    partitions: &HashMap<String, (i32, Struct)>,
) -> Vec<LegacyPositionDelete> {
    let mut out = Vec::new();
    for item in pending {
        if let Some(referenced) = item.referenced {
            if touched_paths.contains(referenced.as_str()) {
                out.push(LegacyPositionDelete {
                    file: item.file,
                    touched: vec![referenced],
                    file_scoped: true,
                    data_sequence_number: item.seq,
                });
            }
            continue;
        }
        let mut touched = Vec::new();
        for path in touched_paths {
            let Some((spec_id, partition)) = partitions.get(*path) else {
                continue;
            };
            if partition_matches(&item.file, *spec_id, partition)
                && file_path_bounds_admit(&item.file, path)
            {
                touched.push((*path).to_string());
            }
        }
        if !touched.is_empty() {
            out.push(LegacyPositionDelete {
                file: item.file,
                touched,
                file_scoped: false,
                data_sequence_number: item.seq,
            });
        }
    }
    out
}

async fn collect_live_data_files(
    table: &Table,
    manifest_list: Option<&ManifestList>,
    wanted: &HashSet<&str>,
) -> Result<HashMap<String, (DataFile, i64)>> {
    let mut files = HashMap::with_capacity(wanted.len());
    let Some(manifest_list) = manifest_list else {
        return Ok(files);
    };
    if wanted.is_empty() {
        return Ok(files);
    }
    let mut manifests = manifest_stream(table.file_io(), manifest_list, ManifestContentType::Data);
    while let Some(manifest) = manifests.try_next().await? {
        for entry in manifest.entries() {
            if !entry.is_alive() || entry.data_file().content_type() != DataContentType::Data {
                continue;
            }
            if !wanted.contains(entry.file_path()) {
                continue;
            }
            let data_file = entry.data_file().clone();
            let seq = entry.sequence_number().unwrap_or(0);
            files
                .entry(data_file.file_path().to_string())
                .or_insert((data_file, seq));
        }
    }
    Ok(files)
}

fn referenced_path(data_file: &DataFile) -> Result<&str> {
    data_file.referenced_data_file_ref().ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Deletion vector '{}' has no referenced_data_file",
                data_file.file_path()
            ),
        )
    })
}

fn partition_key_for(table: &Table, data_file: &DataFile) -> Result<PartitionKey> {
    partition_key_of(
        table,
        data_file.partition_spec_id(),
        data_file.partition(),
        data_file.file_path(),
    )
}

fn partition_key_of(
    table: &Table,
    spec_id: i32,
    partition: &Struct,
    described: &str,
) -> Result<PartitionKey> {
    let spec: PartitionSpec = table
        .metadata()
        .partition_spec_by_id(spec_id)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Deletion vector '{described}' references unknown partition spec {spec_id}"
                ),
            )
        })?
        .as_ref()
        .clone();
    PartitionKey::new(
        spec,
        table.metadata().current_schema().clone(),
        partition.clone(),
    )
}

async fn write_dv_blobs(table: &Table, blobs: Vec<BlobWrite>) -> Result<Vec<DataFile>> {
    if blobs.is_empty() {
        return Ok(Vec::new());
    }
    let metadata = table.metadata();
    let location_generator = DefaultLocationGenerator::new(metadata.clone())?;
    let file_name_generator = DefaultFileNameGenerator::new(
        "dv".to_string(),
        Some(Uuid::now_v7().to_string()),
        DataFileFormat::Puffin,
    );
    let location =
        location_generator.generate_location(None, &file_name_generator.generate_file_name());
    let mut previous_by_path: HashMap<String, PreviousDeletes> = HashMap::new();
    let mut writes: Vec<(String, Vec<u64>, PartitionKey)> = Vec::with_capacity(blobs.len());
    for blob in blobs {
        if let Some(previous) = blob.previous {
            previous_by_path.insert(
                blob.referenced.clone(),
                PreviousDeletes::new(previous, Vec::new()),
            );
        }
        writes.push((blob.referenced, blob.positions, blob.partition_key));
    }
    let mut writer = DVFileWriter::new(table.file_io().new_output(location)?)
        .unpartitioned()
        .with_previous_deletes(previous_by_path);
    for (referenced, positions, partition_key) in &writes {
        for &position in positions {
            writer.delete(referenced, position, Some(partition_key))?;
        }
    }
    writer.close().await
}

#[cfg(test)]
mod tests;
