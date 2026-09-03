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

use crate::Result;
use crate::delete_file_index::referenced_data_file_location;
use crate::spec::{DataContentType, DataFile, ManifestContentType};
use crate::table::Table;

pub(super) struct DvRewritePlan {
    pub(super) removed: Vec<DataFile>,
    pub(super) removed_count: usize,
}

pub(super) async fn plan_dv_removal(
    table: &Table,
    rewritten_data_paths: &HashSet<String>,
) -> Result<DvRewritePlan> {
    let mut removed = Vec::new();
    let mut removed_count: usize = 0;
    for (delete_file, referenced) in live_file_scoped_position_deletes(table).await? {
        if rewritten_data_paths.contains(&referenced) {
            removed_count = removed_count.saturating_add(1);
            removed.push(delete_file);
        }
    }
    Ok(DvRewritePlan {
        removed,
        removed_count,
    })
}

pub(super) async fn file_scoped_delete_paths(table: &Table) -> Result<HashSet<String>> {
    let mut paths = HashSet::new();
    for (delete_file, _) in live_file_scoped_position_deletes(table).await? {
        paths.insert(delete_file.file_path().to_string());
    }
    Ok(paths)
}

async fn live_file_scoped_position_deletes(table: &Table) -> Result<Vec<(DataFile, String)>> {
    let mut out = Vec::new();
    let metadata = table.metadata();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(out);
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
