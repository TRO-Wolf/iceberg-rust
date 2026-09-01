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
//! Delete-file removal here is path-keyed. One Puffin can hold several blobs, so a drop
//! of one blob would take its siblings. Those siblings are rewritten into a new Puffin.

use std::collections::HashSet;

use crate::Result;
use crate::delete_file_index::{is_deletion_vector, referenced_data_file_location};
use crate::delete_vector_container::{DvDropPlan, rewrite_siblings_for_dropped_references};
use crate::spec::{DataContentType, DataFile, ManifestContentType};
use crate::table::Table;

/// DVs to drop, and sibling blobs that must be rewritten because path-keyed removal
/// would otherwise delete them.
pub(super) struct DvRewritePlan {
    pub(super) removed: Vec<DataFile>,
    pub(super) rewritten_siblings: Vec<(DataFile, i64)>,
    pub(super) removed_count: usize,
}

pub(super) async fn plan_dv_removal(
    table: &Table,
    rewritten_data_paths: &HashSet<String>,
) -> Result<DvRewritePlan> {
    let mut plan: DvDropPlan =
        rewrite_siblings_for_dropped_references(table, rewritten_data_paths).await?;
    for (delete_file, referenced) in live_file_scoped_position_deletes(table).await? {
        if is_deletion_vector(&delete_file) {
            continue;
        }
        if rewritten_data_paths.contains(&referenced) {
            plan.dropped_count = plan.dropped_count.saturating_add(1);
            plan.removed.push(delete_file);
        }
    }
    Ok(DvRewritePlan {
        removed: plan.removed,
        rewritten_siblings: plan.rewritten_siblings,
        removed_count: plan.dropped_count,
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
