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

use std::collections::HashMap;

use crate::error::{Error, ErrorKind, Result};
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::scan::FileScanTask;
use crate::spec::{DataContentType, DataFile, SnapshotRef};

impl RewriteDataFiles {
    #[allow(missing_docs)]
    pub fn branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = Some(branch.into());
        self
    }

    pub(super) fn branch_starting_snapshot(&self) -> Result<Option<SnapshotRef>> {
        match self.branch.as_deref() {
            None => Ok(self.table.metadata().current_snapshot().cloned()),
            Some(name) => {
                let snapshot = self
                    .table
                    .metadata()
                    .snapshot_for_ref(name)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("snapshot ref '{name}' not found"),
                        )
                    })?;
                Ok(Some(snapshot.clone()))
            }
        }
    }

    pub(super) async fn plan_scan_tasks(&self) -> Result<Vec<FileScanTask>> {
        use futures::TryStreamExt;

        let starting_id = self
            .branch_starting_snapshot()?
            .map(|snapshot| snapshot.snapshot_id());
        let mut scan = self.table.scan().with_file_prune_only(self.filter.clone());
        if let Some(snapshot_id) = starting_id {
            scan = scan.snapshot_id(snapshot_id);
        }
        let stream = scan.build()?.plan_files().await?;
        stream.try_collect().await
    }

    pub(super) async fn collect_live_data_files(&self) -> Result<HashMap<String, DataFile>> {
        let mut by_path: HashMap<String, DataFile> = HashMap::new();
        let Some(snapshot) = self.branch_starting_snapshot()? else {
            return Ok(by_path);
        };
        let metadata = self.table.metadata();
        let manifest_list = snapshot
            .load_manifest_list(self.table.file_io(), metadata)
            .await?;
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
            for entry in manifest.entries() {
                if entry.is_alive() && entry.content_type() == DataContentType::Data {
                    by_path.insert(entry.file_path().to_string(), entry.data_file().clone());
                }
            }
        }
        Ok(by_path)
    }
}
