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

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{DataFile, ManifestFile, TableMetadata};
use crate::transaction::rewrite_manifests::RewriteManifestsAction;

impl RewriteManifestsAction {
    #[allow(missing_docs)]
    pub fn cluster_by_columns(mut self, columns: Vec<String>) -> Result<Self> {
        if columns.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "sort_by must not be empty when provided",
            ));
        }
        self.cluster_by_columns = Some(columns);
        self.cluster_by = None;
        Ok(self)
    }

    pub(super) fn validate_deleted_manifests(
        &self,
        current_manifests: &[ManifestFile],
        current_snapshot_id: i64,
    ) -> Result<()> {
        let current_paths: HashSet<&str> = current_manifests
            .iter()
            .map(|manifest| manifest.manifest_path.as_str())
            .collect();

        for deleted in &self.deleted_manifests {
            if !current_paths.contains(deleted.manifest_path.as_str()) {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Deleted manifest {} could not be found in the latest snapshot {}",
                        deleted.manifest_path, current_snapshot_id
                    ),
                ));
            }
        }

        Ok(())
    }
}

pub(super) fn cluster_key_for_columns(
    columns: &[String],
    file: &DataFile,
    metadata: &TableMetadata,
) -> Result<String> {
    let spec_id = file.partition_spec_id();
    let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot cluster by columns: partition spec {spec_id} not found"),
        )
    })?;
    let fields = spec.fields();
    let partition = file.partition();
    let mut selected = Vec::with_capacity(columns.len());
    for column in columns {
        let index = fields
            .iter()
            .position(|field| field.name == *column)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot cluster by column '{column}': not a partition field of spec {}",
                        spec.spec_id()
                    ),
                )
            })?;
        selected.push(partition.fields().get(index).cloned().flatten());
    }
    Ok(format!("{selected:?}"))
}
