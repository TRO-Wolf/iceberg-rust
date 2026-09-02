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

//! Deletion-vector container closure for DataFusion DML and maintenance.

use std::collections::{HashMap, HashSet};

use futures::{StreamExt, TryStreamExt, stream};
use uuid::Uuid;

use crate::delete_file_index::is_deletion_vector;
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

const DV_IO_CONCURRENCY: usize = 8;

/// One rewritten DV blob plus the data sequence to stamp, or `None` to inherit the new snapshot.
pub type StampedDeleteFile = (DataFile, Option<i64>);

/// Result of closing the deletion vectors of one commit.
#[derive(Debug, Default)]
pub struct DvContainerClose {
    /// Replacement DV metadata. `Some(seq)` keeps a sibling's original data sequence.
    pub added: Vec<StampedDeleteFile>,
    /// The superseded blobs, one per touched referenced file. Untouched siblings are not here.
    pub removed: Vec<DataFile>,
    /// Untouched sibling references in a rewritten blob's container; the commit still checks them.
    pub retained_references: HashSet<String>,
}

impl DvContainerClose {
    /// Referenced data-file paths the commit depends on: replacements plus retained siblings.
    pub fn referenced_data_files(&self) -> HashSet<String> {
        let mut references: HashSet<String> = self
            .added
            .iter()
            .filter_map(|(file, _)| file.referenced_data_file())
            .collect();
        references.extend(self.retained_references.iter().cloned());
        references
    }
}

/// Maintenance plan: drop DVs whose referenced data file is gone, rewrite live siblings.
#[derive(Debug, Default)]
pub struct DvDropPlan {
    /// All blobs in each affected Puffin (dropped + rewritten siblings).
    pub removed: Vec<DataFile>,
    /// Sibling replacements stamped with the original data sequence.
    pub rewritten_siblings: Vec<(DataFile, i64)>,
    /// Count of dropped blobs, not including rewritten siblings.
    pub dropped_count: usize,
}

struct LiveDv {
    data_file: DataFile,
    sequence_number: i64,
}

struct DvIndex {
    touched: Vec<DataFile>,
    siblings: Vec<(String, String)>,
}

struct BlobWrite {
    referenced: String,
    positions: Vec<u64>,
    previous: Option<DeleteVector>,
    partition_key: PartitionKey,
    data_sequence: Option<i64>,
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
    close_touched_dv_containers_with_partitions(table, new_positions, snapshot_id, &HashMap::new())
        .await
}

/// Close touched DV containers, taking each touched path's `(partition_spec_id, partition)` from `known_partitions`.
pub async fn close_touched_dv_containers_with_partitions(
    table: &Table,
    new_positions: &HashMap<String, Vec<u64>>,
    snapshot_id: Option<i64>,
    known_partitions: &HashMap<String, (i32, Struct)>,
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

    let manifest_list = live_manifest_list(table, snapshot_id).await?;
    let DvIndex { touched, siblings } =
        collect_dv_index(table, manifest_list.as_ref(), new_positions).await?;

    let mut close = DvContainerClose::default();
    let mut specs: Vec<BlobWrite> = Vec::with_capacity(new_positions.len());

    let file_io = table.file_io().clone();
    let loaded: Vec<(DataFile, DeleteVector)> =
        stream::iter(touched.into_iter().map(|data_file| {
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
        let referenced = referenced_path(&data_file)?;
        let added = new_positions.get(&referenced).ok_or_else(|| {
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
            data_sequence: None,
        });
        close.removed.push(data_file);
    }

    let fresh: Vec<&String> = {
        let rewritten: HashSet<&str> = specs.iter().map(|spec| spec.referenced.as_str()).collect();
        new_positions
            .iter()
            .filter(|(path, positions)| !positions.is_empty() && !rewritten.contains(path.as_str()))
            .map(|(path, _)| path)
            .collect()
    };

    if !fresh.is_empty() {
        let unresolved: HashSet<&str> = fresh
            .iter()
            .map(|path| path.as_str())
            .filter(|path| !known_partitions.contains_key(*path))
            .collect();
        let discovered = if unresolved.is_empty() {
            HashMap::new()
        } else {
            collect_live_data_files(table, manifest_list.as_ref(), &unresolved).await?
        };
        for path in fresh {
            let partition_key = match known_partitions.get(path) {
                Some((spec_id, partition)) => partition_key_of(table, *spec_id, partition, path)?,
                None => {
                    let data_file = discovered.get(path.as_str()).ok_or_else(|| {
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
                positions: new_positions[path].clone(),
                previous: None,
                data_sequence: None,
            });
        }
    }

    close.removed.sort_unstable_by(|left, right| {
        (left.file_path(), left.content_offset()).cmp(&(right.file_path(), right.content_offset()))
    });
    close.added = write_dv_blobs(table, specs).await?;

    let rewritten_containers: HashSet<&str> = close
        .removed
        .iter()
        .map(|data_file| data_file.file_path())
        .collect();
    for (container, referenced) in &siblings {
        if rewritten_containers.contains(container.as_str()) {
            close.retained_references.insert(referenced.clone());
        }
    }

    Ok(close)
}

/// Drop DVs that reference `dropped_data_paths` and rewrite live siblings in those Puffins.
pub async fn rewrite_siblings_for_dropped_references(
    table: &Table,
    dropped_data_paths: &HashSet<String>,
) -> Result<DvDropPlan> {
    if dropped_data_paths.is_empty() {
        return Ok(DvDropPlan::default());
    }

    let manifest_list = live_manifest_list(table, None).await?;
    let live_dvs = collect_live_dvs(table, manifest_list.as_ref()).await?;
    let mut by_puffin: HashMap<String, Vec<LiveDv>> = HashMap::new();
    for dv in live_dvs {
        by_puffin
            .entry(dv.data_file.file_path().to_string())
            .or_default()
            .push(dv);
    }

    let mut plan = DvDropPlan::default();
    for (_puffin, blobs) in by_puffin {
        let mut dropping = Vec::new();
        let mut siblings = Vec::new();
        for blob in blobs {
            match blob.data_file.referenced_data_file() {
                Some(referenced) if dropped_data_paths.contains(&referenced) => {
                    dropping.push(blob);
                }
                _ => siblings.push(blob),
            }
        }
        if dropping.is_empty() {
            continue;
        }
        plan.dropped_count += dropping.len();
        plan.removed
            .extend(dropping.into_iter().map(|blob| blob.data_file));
        if !siblings.is_empty() {
            let mut specs = Vec::with_capacity(siblings.len());
            for sibling in &siblings {
                let referenced = referenced_path(&sibling.data_file)?;
                let mut positions: Vec<u64> =
                    load_delete_vector(table.file_io(), &sibling.data_file)
                        .await?
                        .iter()
                        .collect();
                positions.sort_unstable();
                specs.push(BlobWrite {
                    partition_key: partition_key_for(table, &sibling.data_file)?,
                    referenced,
                    positions,
                    previous: None,
                    data_sequence: Some(sibling.sequence_number),
                });
            }
            for (file, seq) in write_dv_blobs(table, specs).await? {
                let seq = seq.ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!(
                            "rewritten sibling deletion vector '{}' lost its data sequence",
                            file.file_path()
                        ),
                    )
                })?;
                plan.rewritten_siblings.push((file, seq));
            }
            plan.removed
                .extend(siblings.into_iter().map(|blob| blob.data_file));
        }
    }
    Ok(plan)
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

async fn collect_dv_index(
    table: &Table,
    manifest_list: Option<&ManifestList>,
    new_positions: &HashMap<String, Vec<u64>>,
) -> Result<DvIndex> {
    let mut index = DvIndex {
        touched: Vec::new(),
        siblings: Vec::new(),
    };
    let Some(manifest_list) = manifest_list else {
        return Ok(index);
    };
    let mut manifests =
        manifest_stream(table.file_io(), manifest_list, ManifestContentType::Deletes);
    while let Some(manifest) = manifests.try_next().await? {
        for entry in manifest.entries() {
            if !is_live_dv(entry) {
                continue;
            }
            let data_file = entry.data_file();
            let referenced = referenced_path(data_file)?;
            if new_positions.contains_key(&referenced) {
                index.touched.push(data_file.clone());
            } else {
                index
                    .siblings
                    .push((data_file.file_path().to_string(), referenced));
            }
        }
    }
    Ok(index)
}

async fn collect_live_dvs(
    table: &Table,
    manifest_list: Option<&ManifestList>,
) -> Result<Vec<LiveDv>> {
    let mut live = Vec::new();
    let Some(manifest_list) = manifest_list else {
        return Ok(live);
    };
    let mut manifests =
        manifest_stream(table.file_io(), manifest_list, ManifestContentType::Deletes);
    while let Some(manifest) = manifests.try_next().await? {
        for entry in manifest.entries() {
            if !is_live_dv(entry) {
                continue;
            }
            live.push(LiveDv {
                data_file: entry.data_file().clone(),
                sequence_number: entry.sequence_number().unwrap_or(0),
            });
        }
    }
    Ok(live)
}

async fn collect_live_data_files(
    table: &Table,
    manifest_list: Option<&ManifestList>,
    wanted: &HashSet<&str>,
) -> Result<HashMap<String, DataFile>> {
    let mut files = HashMap::with_capacity(wanted.len());
    let Some(manifest_list) = manifest_list else {
        return Ok(files);
    };
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
            files
                .entry(data_file.file_path().to_string())
                .or_insert(data_file);
        }
    }
    Ok(files)
}

fn referenced_path(data_file: &DataFile) -> Result<String> {
    data_file.referenced_data_file().ok_or_else(|| {
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

async fn write_dv_blobs(table: &Table, blobs: Vec<BlobWrite>) -> Result<Vec<StampedDeleteFile>> {
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
    let mut seq_by_ref: HashMap<String, Option<i64>> = HashMap::with_capacity(blobs.len());
    let mut previous_by_path: HashMap<String, PreviousDeletes> = HashMap::new();
    let mut writes: Vec<(String, Vec<u64>, PartitionKey)> = Vec::with_capacity(blobs.len());
    for blob in blobs {
        seq_by_ref.insert(blob.referenced.clone(), blob.data_sequence);
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
    let files = writer.close().await?;
    let mut out = Vec::with_capacity(files.len());
    for file in files {
        let referenced = referenced_path(&file)?;
        let sequence = seq_by_ref.get(&referenced).copied().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "Rewritten deletion vector '{}' referenced unknown data file '{referenced}'",
                    file.file_path()
                ),
            )
        })?;
        out.push((file, sequence));
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
