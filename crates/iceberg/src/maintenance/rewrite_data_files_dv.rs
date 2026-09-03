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

//! Drop file-scoped deletes that reference data files this rewrite removes.
//!
//! Java 1.10.0 `ManifestFilterManager.isDanglingDV` is `ContentFileUtil.isDV` and
//! `removedDataFilePaths.contains(referencedDataFile())`. The apply path drops DVs
//! only. File-scoped parquet position deletes are a fork extension of that predicate.
//!
//! Delete-file removal is keyed by the Java `DeleteFileSet` triple, so a drop leaves
//! a sibling blob at the same Puffin path in place.

use std::collections::HashSet;

use futures::{StreamExt, TryStreamExt, stream};

use crate::Result;
use crate::delete_file_index::referenced_data_file_location;
use crate::io::FileIO;
use crate::spec::{DataContentType, DataFile, Manifest, ManifestContentType, ManifestList};
use crate::table::Table;

const DELETE_MANIFEST_IO_CONCURRENCY: usize = 8;

pub(super) struct DvRewritePlan {
    pub(super) removed: Vec<DataFile>,
    pub(super) removed_count: usize,
}

pub(super) fn plan_dv_removal(
    live: &[(DataFile, String)],
    rewritten_data_paths: &HashSet<String>,
) -> DvRewritePlan {
    let mut removed = Vec::new();
    let mut removed_count: usize = 0;
    for (delete_file, referenced) in live {
        if rewritten_data_paths.contains(referenced) {
            removed_count = removed_count.saturating_add(1);
            removed.push(delete_file.clone());
        }
    }
    DvRewritePlan {
        removed,
        removed_count,
    }
}

pub(super) fn file_scoped_delete_paths_from(live: &[(DataFile, String)]) -> HashSet<String> {
    live.iter()
        .map(|(delete_file, _)| delete_file.file_path().to_string())
        .collect()
}

#[cfg(test)]
pub(super) async fn file_scoped_delete_paths(table: &Table) -> Result<HashSet<String>> {
    let mut paths = HashSet::new();
    let Some(manifests) = load_delete_manifests(table).await? else {
        return Ok(paths);
    };
    for manifest in manifests {
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let data_file = entry.data_file();
            if data_file.content_type() != DataContentType::PositionDeletes {
                continue;
            }
            if referenced_data_file_location(data_file).is_some() {
                paths.insert(data_file.file_path().to_string());
            }
        }
    }
    Ok(paths)
}

pub(super) async fn live_file_scoped_position_deletes(
    table: &Table,
) -> Result<Vec<(DataFile, String)>> {
    let mut out = Vec::new();
    let Some(manifests) = load_delete_manifests(table).await? else {
        return Ok(out);
    };
    for manifest in manifests {
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let data_file = entry.data_file();
            if data_file.content_type() != DataContentType::PositionDeletes {
                continue;
            }
            if let Some(referenced) = referenced_data_file_location(data_file) {
                out.push((data_file.clone(), referenced));
            }
        }
    }
    Ok(out)
}

async fn load_delete_manifests(table: &Table) -> Result<Option<Vec<Manifest>>> {
    let metadata = table.metadata();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(None);
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await?;
    let manifests = delete_manifest_stream(table.file_io(), &manifest_list)
        .try_collect()
        .await?;
    Ok(Some(manifests))
}

fn delete_manifest_stream<'a>(
    file_io: &FileIO,
    manifest_list: &'a ManifestList,
) -> impl futures::Stream<Item = Result<Manifest>> + 'a {
    let file_io = file_io.clone();
    stream::iter(
        manifest_list
            .entries()
            .iter()
            .filter(|manifest_file| manifest_file.content == ManifestContentType::Deletes),
    )
    .map(move |manifest_file| {
        let file_io = file_io.clone();
        async move { manifest_file.load_manifest(&file_io).await }
    })
    .buffer_unordered(DELETE_MANIFEST_IO_CONCURRENCY)
}
