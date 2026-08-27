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

//! Drop deletion vectors that reference data files this rewrite removes.
//!
//! Java `MergingSnapshotProducer.apply` calls `deleteFilterManager.removeDanglingDeletesFor`
//! with the rewritten data files. A DV whose `referenced_data_file` is in that set is
//! dropped in the same commit.
//!
//! Delete-file removal here is path-keyed. One Puffin can hold several blobs, so a drop
//! of one blob would take its siblings. Those siblings are rewritten into a new Puffin.

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

/// DVs to drop, and sibling blobs that must be rewritten because path-keyed removal
/// would otherwise delete them.
pub(super) struct DvRewritePlan {
    pub(super) removed: Vec<DataFile>,
    pub(super) rewritten_siblings: Vec<(DataFile, i64)>,
    pub(super) removed_count: usize,
}

struct LiveDv {
    data_file: DataFile,
    sequence_number: i64,
}

/// Plan DV drops for `rewritten_data_paths`. Empty input yields an empty plan.
pub(super) async fn plan_dv_removal(
    table: &Table,
    rewritten_data_paths: &HashSet<String>,
) -> Result<DvRewritePlan> {
    if rewritten_data_paths.is_empty() {
        return Ok(DvRewritePlan {
            removed: Vec::new(),
            rewritten_siblings: Vec::new(),
            removed_count: 0,
        });
    }

    let live = collect_live_dvs(table).await?;
    let mut by_puffin: HashMap<String, Vec<LiveDv>> = HashMap::new();
    for dv in live {
        by_puffin
            .entry(dv.data_file.file_path().to_string())
            .or_default()
            .push(dv);
    }

    let mut removed = Vec::new();
    let mut rewritten_siblings = Vec::new();
    let mut removed_count = 0usize;

    for (_puffin_path, blobs) in by_puffin {
        let mut dropping = Vec::new();
        let mut siblings = Vec::new();
        for blob in blobs {
            match blob.data_file.referenced_data_file() {
                Some(referenced) if rewritten_data_paths.contains(&referenced) => {
                    dropping.push(blob);
                }
                _ => siblings.push(blob),
            }
        }
        if dropping.is_empty() {
            continue;
        }
        removed_count += dropping.len();
        // Path-keyed removal tombstones every blob in this Puffin.
        removed.extend(dropping.into_iter().map(|blob| blob.data_file));
        if !siblings.is_empty() {
            rewritten_siblings.extend(rewrite_sibling_dvs(table, &siblings).await?);
            removed.extend(siblings.into_iter().map(|blob| blob.data_file));
        }
    }

    Ok(DvRewritePlan {
        removed,
        rewritten_siblings,
        removed_count,
    })
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

async fn rewrite_sibling_dvs(table: &Table, siblings: &[LiveDv]) -> Result<Vec<(DataFile, i64)>> {
    let metadata = table.metadata();
    let schema = metadata.current_schema().clone();
    let location_generator = DefaultLocationGenerator::new(metadata.clone())?;
    let file_name_generator = DefaultFileNameGenerator::new(
        "compacted-dv".to_string(),
        Some(Uuid::now_v7().to_string()),
        DataFileFormat::Puffin,
    );
    let location =
        location_generator.generate_location(None, &file_name_generator.generate_file_name());
    let mut writer = DVFileWriter::new(table.file_io().new_output(location)?);

    let mut seq_by_ref: HashMap<String, i64> = HashMap::new();
    for sibling in siblings {
        let referenced = sibling.data_file.referenced_data_file().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Deletion vector '{}' has no referenced_data_file",
                    sibling.data_file.file_path()
                ),
            )
        })?;
        let spec = metadata
            .partition_spec_by_id(sibling.data_file.partition_spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Deletion vector '{}' references unknown partition spec {}",
                        sibling.data_file.file_path(),
                        sibling.data_file.partition_spec_id
                    ),
                )
            })?
            .as_ref()
            .clone();
        let partition_key =
            PartitionKey::new(spec, schema.clone(), sibling.data_file.partition().clone())?;
        let vector = load_delete_vector(table.file_io(), &sibling.data_file).await?;
        for position in vector.iter() {
            writer.delete(&referenced, position, Some(&partition_key))?;
        }
        seq_by_ref.insert(referenced, sibling.sequence_number);
    }

    let new_files = writer.close().await?;
    let mut out = Vec::with_capacity(new_files.len());
    for file in new_files {
        let referenced = file.referenced_data_file().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Rewritten deletion vector '{}' has no referenced_data_file",
                    file.file_path()
                ),
            )
        })?;
        let sequence_number = seq_by_ref.get(&referenced).copied().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Rewritten deletion vector '{}' referenced unknown data file '{referenced}'",
                    file.file_path()
                ),
            )
        })?;
        out.push((file, sequence_number));
    }
    Ok(out)
}
