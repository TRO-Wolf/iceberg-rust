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

//! Shared-Puffin deletion-vector container closure.
//!
//! Path-keyed `RowDelta` removal tombstones every blob in a Puffin. Callers that rewrite one
//! referenced file must rewrite every live sibling in the same file. Maintenance and DataFusion
//! DML both use this module so the grouping and sibling copy live in one place.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::delete_file_index::is_deletion_vector;
use crate::delete_vector::load_delete_vector;
use crate::spec::{DataContentType, DataFile, DataFileFormat, ManifestContentType, PartitionKey};
use crate::table::Table;
use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
};
use crate::{Error, ErrorKind, Result};

/// One rewritten DV blob plus the data sequence to stamp, or `None` to inherit the new snapshot.
pub type StampedDeleteFile = (DataFile, Option<i64>);

/// Result of closing one or more physical Puffin containers.
#[derive(Debug, Default)]
pub struct DvContainerClose {
    /// Replacement DV metadata. `Some(seq)` keeps a sibling's original data sequence.
    pub added: Vec<StampedDeleteFile>,
    /// Every live blob in each replaced Puffin. Path-keyed removal needs the full set.
    pub removed: Vec<DataFile>,
}

impl DvContainerClose {
    /// Referenced data-file paths carried by the replacement blobs.
    pub fn referenced_data_files(&self) -> HashSet<String> {
        self.added
            .iter()
            .filter_map(|(file, _)| file.referenced_data_file())
            .collect()
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

struct BlobWrite {
    referenced: String,
    positions: Vec<u64>,
    partition_key: PartitionKey,
    data_sequence: Option<i64>,
}

/// Rewrite every Puffin that holds a blob for `new_positions`. Touched blobs union the new
/// positions and inherit the commit sequence. Untouched siblings keep their positions and data
/// sequence. Touched files with no previous DV go into a new Puffin.
pub async fn close_touched_dv_containers(
    table: &Table,
    new_positions: &HashMap<String, Vec<u64>>,
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

    let live_dvs = collect_live_dvs(table).await?;
    let data_files = collect_live_data_files(table).await?;
    let mut by_puffin: HashMap<String, Vec<LiveDv>> = HashMap::new();
    for dv in live_dvs {
        by_puffin
            .entry(dv.data_file.file_path().to_string())
            .or_default()
            .push(dv);
    }

    let mut close = DvContainerClose::default();
    let mut covered: HashSet<String> = HashSet::new();

    for (_puffin, blobs) in by_puffin {
        let affected = blobs.iter().any(|blob| {
            blob.data_file
                .referenced_data_file()
                .map(|referenced| new_positions.contains_key(&referenced))
                .unwrap_or(false)
        });
        if !affected {
            continue;
        }

        let mut specs = Vec::with_capacity(blobs.len());
        for blob in &blobs {
            let referenced = referenced_path(&blob.data_file)?;
            let mut positions: Vec<u64> = load_delete_vector(table.file_io(), &blob.data_file)
                .await?
                .iter()
                .collect();
            let data_sequence = if let Some(added) = new_positions.get(&referenced) {
                covered.insert(referenced.clone());
                positions.extend(added.iter().copied());
                None
            } else {
                Some(blob.sequence_number)
            };
            positions.sort_unstable();
            positions.dedup();
            specs.push(BlobWrite {
                partition_key: partition_key_for(table, &blob.data_file)?,
                referenced,
                positions,
                data_sequence,
            });
        }
        close.added.extend(write_dv_blobs(table, &specs).await?);
        close
            .removed
            .extend(blobs.into_iter().map(|blob| blob.data_file));
    }

    let mut remaining = Vec::new();
    for (path, positions) in new_positions {
        if covered.contains(path) {
            continue;
        }
        let data_file = data_files.get(path).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "deletion-vector: data file `{path}` is not a live file of the current snapshot"
                ),
            )
        })?;
        let mut positions = positions.clone();
        positions.sort_unstable();
        positions.dedup();
        remaining.push(BlobWrite {
            partition_key: partition_key_for(table, data_file)?,
            referenced: path.clone(),
            positions,
            data_sequence: None,
        });
    }
    if !remaining.is_empty() {
        close.added.extend(write_dv_blobs(table, &remaining).await?);
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

    let live_dvs = collect_live_dvs(table).await?;
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
                    data_sequence: Some(sibling.sequence_number),
                });
            }
            for (file, seq) in write_dv_blobs(table, &specs).await? {
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

async fn collect_live_dvs(table: &Table) -> Result<Vec<LiveDv>> {
    let mut live = Vec::new();
    let metadata = table.metadata();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(live);
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await?;
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let data_file = entry.data_file();
            if data_file.content_type() != DataContentType::PositionDeletes
                || !is_deletion_vector(data_file)
            {
                continue;
            }
            live.push(LiveDv {
                data_file: data_file.clone(),
                sequence_number: entry.sequence_number().unwrap_or(0),
            });
        }
    }
    Ok(live)
}

async fn collect_live_data_files(table: &Table) -> Result<HashMap<String, DataFile>> {
    let mut files = HashMap::new();
    let metadata = table.metadata();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(files);
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await?;
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() && entry.data_file().content_type() == DataContentType::Data {
                let data_file = entry.data_file().clone();
                files
                    .entry(data_file.file_path().to_string())
                    .or_insert(data_file);
            }
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
    let spec = table
        .metadata()
        .partition_spec_by_id(data_file.partition_spec_id())
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Deletion vector '{}' references unknown partition spec {}",
                    data_file.file_path(),
                    data_file.partition_spec_id()
                ),
            )
        })?
        .as_ref()
        .clone();
    PartitionKey::new(
        spec,
        table.metadata().current_schema().clone(),
        data_file.partition().clone(),
    )
}

async fn write_dv_blobs(table: &Table, blobs: &[BlobWrite]) -> Result<Vec<StampedDeleteFile>> {
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
    let mut writer = DVFileWriter::new(table.file_io().new_output(location)?).unpartitioned();
    let mut seq_by_ref: HashMap<String, Option<i64>> = HashMap::new();
    for blob in blobs {
        for &position in &blob.positions {
            writer.delete(&blob.referenced, position, Some(&blob.partition_key))?;
        }
        seq_by_ref.insert(blob.referenced.clone(), blob.data_sequence);
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
