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

use crate::spec::{Operation, SnapshotRef, TableMetadata};
use crate::{Error, ErrorKind, Result};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NonAppendPolicy {
    pub(crate) fail_loud: bool,
    pub(crate) skip_overwrite: bool,
    pub(crate) skip_delete: bool,
}

impl NonAppendPolicy {
    fn refuses(&self, snapshot: &SnapshotRef) -> bool {
        match snapshot.summary().operation {
            Operation::Append | Operation::Replace => false,
            Operation::Overwrite => !self.skip_overwrite,
            Operation::Delete => !self.skip_delete,
        }
    }
}

pub(crate) fn appends_between(
    metadata: &TableMetadata,
    from_snapshot_id_exclusive: Option<i64>,
    to_snapshot_id: i64,
    policy: NonAppendPolicy,
) -> Result<Vec<SnapshotRef>> {
    if from_snapshot_id_exclusive == Some(to_snapshot_id) {
        return Ok(vec![]);
    }

    let mut window = Vec::new();
    let mut current = metadata.snapshot_by_id(to_snapshot_id).cloned();
    while let Some(snapshot) = current {
        if Some(snapshot.snapshot_id()) == from_snapshot_id_exclusive {
            break;
        }
        current = snapshot
            .parent_snapshot_id()
            .and_then(|parent_id| metadata.snapshot_by_id(parent_id).cloned());
        window.push(snapshot);
    }

    if policy.fail_loud
        && let Some(refused) = window
            .iter()
            .rev()
            .find(|snapshot| policy.refuses(snapshot))
    {
        let operation = refused.summary().operation.as_str();
        let from =
            from_snapshot_id_exclusive.map_or_else(|| "root".to_string(), |id| id.to_string());
        return Err(Error::new(
            ErrorKind::PreconditionFailed,
            format!(
                "Cannot process {operation} snapshot {} in the incremental append window ({from}, {to_snapshot_id}]; to skip {operation} snapshots, set with_skip_{operation}_snapshots(true)",
                refused.snapshot_id()
            ),
        ));
    }

    Ok(window
        .into_iter()
        .filter(|snapshot| snapshot.summary().operation == Operation::Append)
        .collect())
}
