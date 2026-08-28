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

use std::collections::HashSet;

use crate::Result;
use crate::delete_vector_container::{DvDropPlan, rewrite_siblings_for_dropped_references};
use crate::spec::DataFile;
use crate::table::Table;

/// DVs to drop, and sibling blobs that must be rewritten because path-keyed removal
/// would otherwise delete them.
pub(super) struct DvRewritePlan {
    pub(super) removed: Vec<DataFile>,
    pub(super) rewritten_siblings: Vec<(DataFile, i64)>,
    pub(super) removed_count: usize,
}

/// Plan DV drops for `rewritten_data_paths`. Empty input yields an empty plan.
pub(super) async fn plan_dv_removal(
    table: &Table,
    rewritten_data_paths: &HashSet<String>,
) -> Result<DvRewritePlan> {
    let plan: DvDropPlan =
        rewrite_siblings_for_dropped_references(table, rewritten_data_paths).await?;
    Ok(DvRewritePlan {
        removed: plan.removed,
        rewritten_siblings: plan.rewritten_siblings,
        removed_count: plan.dropped_count,
    })
}
