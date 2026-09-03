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

use std::collections::{HashMap, HashSet};

use datafusion::common::{DataFusionError, Result as DFResult};
use iceberg::spec::{DataFile, Struct};
use iceberg::table::Table;

use crate::to_datafusion_error;

pub(crate) fn position_delete_unpartitioned_fast_path(
    spec_count: usize,
    default_field_count: usize,
) -> bool {
    spec_count == 1 && default_field_count == 0
}

pub(crate) fn snapshot_for_scan(
    table: &Table,
    snapshot_id: Option<i64>,
) -> Option<&iceberg::spec::SnapshotRef> {
    let metadata = table.metadata();
    match snapshot_id {
        Some(id) => metadata.snapshot_by_id(id),
        None => metadata.current_snapshot(),
    }
}

pub(crate) async fn live_data_file_partitions(
    table: &Table,
    snapshot_id: Option<i64>,
    wanted: Option<&HashSet<&str>>,
) -> DFResult<HashMap<String, (i32, Struct, Option<i64>)>> {
    let metadata = table.metadata();
    let mut path_to_partition: HashMap<String, (i32, Struct, Option<i64>)> = HashMap::new();
    let Some(snapshot) = snapshot_for_scan(table, snapshot_id) else {
        return Ok(path_to_partition);
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await
        .map_err(to_datafusion_error)?;

    for manifest_entry in manifest_list.entries() {
        if manifest_entry.content != iceberg::spec::ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_entry
            .load_manifest(table.file_io())
            .await
            .map_err(to_datafusion_error)?;
        for entry in manifest.entries() {
            if entry.is_alive()
                && entry.data_file().content_type() == iceberg::spec::DataContentType::Data
            {
                let data_file = entry.data_file();
                if let Some(wanted) = wanted
                    && !wanted.contains(entry.file_path())
                {
                    continue;
                }
                path_to_partition
                    .entry(data_file.file_path().to_string())
                    .or_insert_with(|| {
                        (
                            data_file.partition_spec_id(),
                            data_file.partition().clone(),
                            entry.sequence_number(),
                        )
                    });
            }
        }
    }
    Ok(path_to_partition)
}

pub(crate) async fn resolve_affected_data_files(
    table: &Table,
    affected: &HashSet<String>,
    snapshot_id: Option<i64>,
) -> DFResult<Vec<DataFile>> {
    let metadata = table.metadata();
    let mut resolved: Vec<DataFile> = Vec::with_capacity(affected.len());
    let mut found: HashSet<String> = HashSet::with_capacity(affected.len());

    if let Some(snapshot) = snapshot_for_scan(table, snapshot_id) {
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), metadata)
            .await
            .map_err(to_datafusion_error)?;
        for manifest_entry in manifest_list.entries() {
            if manifest_entry.content != iceberg::spec::ManifestContentType::Data {
                continue;
            }
            let manifest = manifest_entry
                .load_manifest(table.file_io())
                .await
                .map_err(to_datafusion_error)?;
            for entry in manifest.entries() {
                if entry.is_alive()
                    && entry.data_file().content_type() == iceberg::spec::DataContentType::Data
                    && affected.contains(entry.file_path())
                    && !found.contains(entry.file_path())
                {
                    found.insert(entry.file_path().to_string());
                    resolved.push(entry.data_file().clone());
                }
            }
        }
    }

    if found.len() != affected.len() {
        let missing: Vec<&str> = affected
            .iter()
            .map(String::as_str)
            .filter(|path| !found.contains(*path))
            .collect();
        return Err(DataFusionError::Internal(format!(
            "copy-on-write: scanned data file(s) not live in the scanned snapshot: {}",
            missing.join(", ")
        )));
    }

    Ok(resolved)
}
