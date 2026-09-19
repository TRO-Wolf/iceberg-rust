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

use std::collections::HashSet;

use crate::error::Result;
use crate::expr::Predicate;
use crate::spec::{DataFile, ManifestEntry, ManifestFile, Operation};
use crate::transaction::snapshot::{SnapshotProduceOperation, SnapshotProducer};

/// The [`SnapshotProduceOperation`] for [`OverwriteFilesAction`].
///
/// It classifies the operation (Java `BaseOverwriteFiles.operation()`), exposes the current manifests, and
/// resolves the delete paths against the live data entries. The added files reach the producer separately,
/// so one snapshot carries both the added manifest and the rewritten manifests.
pub(crate) struct OverwriteFilesOperation {
    pub(crate) delete_paths: HashSet<String>,
    /// The delete-by-row-filter predicate (Java `deleteExpression`). `Some` unions every strictly matched
    /// live data file with the path-resolved deletes. `None` means Java `alwaysFalse`.
    pub(crate) row_filter: Option<Predicate>,
    /// Whether this overwrite requested any added data files. With the requested delete state it classifies
    /// the operation like Java `BaseOverwriteFiles.operation()`.
    pub(crate) adds_data_files: bool,
    /// Case sensitivity for binding `row_filter` (Java default `true`).
    pub(crate) case_sensitive: bool,
    pub(crate) allow_empty_commit: bool,
}

impl SnapshotProduceOperation for OverwriteFilesOperation {
    /// Classify the operation on the REQUESTED sets, like Java `BaseOverwriteFiles.operation()`. Delete-only
    /// gives [`Operation::Delete`], add-only gives [`Operation::Append`], both give [`Operation::Overwrite`].
    /// An empty overwrite is rejected earlier, so the both-empty arm never commits.
    fn operation(&self) -> Operation {
        // Java `containsDeletes()`: a set row filter counts as a delete before any file resolves.
        let deletes_data_files = !self.delete_paths.is_empty() || self.row_filter.is_some();
        match (self.adds_data_files, deletes_data_files) {
            (false, true) => Operation::Delete,
            (true, false) => Operation::Append,
            _ => Operation::Overwrite,
        }
    }

    fn allows_empty_commit(&self) -> bool {
        self.allow_empty_commit
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn delete_files(&self, snapshot_produce: &SnapshotProducer<'_>) -> Result<Vec<DataFile>> {
        // Every requested path must match a live entry (Java `failMissingDeletePaths`).
        let mut resolved = snapshot_produce
            .resolve_delete_paths(&self.delete_paths)
            .await?;

        // Union the row-filter matches (Java `deleteByRowFilter`). De-dupe by path so a file removed by
        // both a path and the filter counts once. `process_deletes` matches by path and tolerates a
        // duplicate, but the summary counts must stay accurate, and Java's `DataFileSet` dedupes too.
        if let Some(row_filter) = &self.row_filter {
            let filter_deletes = snapshot_produce
                .resolve_filter_deletes(row_filter, self.case_sensitive)
                .await?;
            let mut seen: HashSet<String> = resolved
                .iter()
                .map(|df| df.file_path().to_string())
                .collect();
            for data_file in filter_deletes {
                if seen.insert(data_file.file_path().to_string()) {
                    resolved.push(data_file);
                }
            }
        }

        Ok(resolved)
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        snapshot_produce.current_manifests().await
    }
}
