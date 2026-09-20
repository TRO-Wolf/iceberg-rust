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

use crate::error::Result;
use crate::spec::{SnapshotRef, TableMetadata};
use crate::transaction::cherry_pick::is_wap_id_published;
use crate::transaction::{
    MergeAppendAction, OverwriteFilesAction, ReplacePartitionsAction, RowDeltaAction,
};
use crate::{Error, ErrorKind};

const STAGED_WAP_ID_PROP: &str = "wap.id";

fn data_invalid(message: String) -> Error {
    Error::new(ErrorKind::DataInvalid, message)
}

#[allow(missing_docs)]
pub fn staged_snapshot_for_wap_id(metadata: &TableMetadata, wap_id: &str) -> Result<SnapshotRef> {
    let mut staged = None;
    for snapshot in metadata.snapshots() {
        if snapshot
            .summary()
            .additional_properties
            .get(STAGED_WAP_ID_PROP)
            .is_some_and(|value| value == wap_id)
        {
            if staged.is_some() {
                return Err(data_invalid(format!(
                    "Cannot apply non-unique WAP ID. Found multiple snapshots with WAP ID '{wap_id}'"
                )));
            }
            staged = Some(snapshot);
        }
    }
    let Some(staged) = staged else {
        return Err(data_invalid(format!(
            "Cannot apply unknown WAP ID '{wap_id}'"
        )));
    };
    if is_wap_id_published(metadata, wap_id) {
        return Err(data_invalid(format!(
            "Duplicate request to cherry pick wap id that was published already: {wap_id}"
        )));
    }
    Ok(staged.clone())
}

impl MergeAppendAction {
    #[allow(missing_docs)]
    pub fn stage_only(mut self) -> Self {
        self.stage_only = true;
        self
    }
}

impl OverwriteFilesAction {
    #[allow(missing_docs)]
    pub fn stage_only(mut self) -> Self {
        self.stage_only = true;
        self
    }
}

impl ReplacePartitionsAction {
    #[allow(missing_docs)]
    pub fn stage_only(mut self) -> Self {
        self.stage_only = true;
        self
    }
}

impl RowDeltaAction {
    #[allow(missing_docs)]
    pub fn stage_only(mut self) -> Self {
        self.stage_only = true;
        self
    }
}
