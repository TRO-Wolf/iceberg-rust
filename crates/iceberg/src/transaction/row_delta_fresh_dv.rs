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

//! Fresh-DV commit door extracted so `row_delta.rs` can stay under its legacy ceiling.

use std::collections::{HashMap, HashSet};

use crate::delete_file_index::{is_deletion_vector, referenced_data_file_location};
use crate::spec::{DataContentType, DataFile, ManifestContentType, Struct};
use crate::table::Table;
use crate::transaction::snapshot::dv_desc;
use crate::{Error, ErrorKind, Result};

/// Reject a DV that would silently supersede a live position-scoped delete, unless this commit
/// removes that delete.
pub(crate) async fn validate_fresh_dvs_only(
    table: &Table,
    added_dvs: &HashMap<String, &DataFile>,
    removed_delete_files: &[DataFile],
) -> Result<()> {
    if added_dvs.is_empty() {
        return Ok(());
    }

    let removed_delete_paths: HashSet<&str> = removed_delete_files
        .iter()
        .map(|file| file.file_path())
        .collect();

    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(());
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), &table.metadata_ref())
        .await?;

    let mut live_data_entry_by_path: HashMap<String, (i32, Struct, Option<i64>)> = HashMap::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Data {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let file = entry.data_file();
            if added_dvs.contains_key(file.file_path()) {
                live_data_entry_by_path.insert(
                    file.file_path().to_string(),
                    (
                        file.partition_spec_id(),
                        file.partition().clone(),
                        entry.sequence_number(),
                    ),
                );
            }
        }
    }

    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let existing = entry.data_file();
            if existing.content_type() != DataContentType::PositionDeletes {
                continue;
            }
            if removed_delete_paths.contains(existing.file_path()) {
                continue;
            }

            if is_deletion_vector(existing) {
                if let Some(referenced) = existing.referenced_data_file()
                    && added_dvs.contains_key(&referenced)
                {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot commit deletion vector for {}: the current snapshot already \
                             carries a live deletion vector for that data file ({}). Read it \
                             back with delete_vector::load_delete_vector, merge it through \
                             DVFileWriter::with_previous_deletes, and pass the superseded file \
                             to RowDelta::remove_deletes_many in THIS commit (Java \
                             BaseDVFileWriter.loadPreviousDeletes + RowDelta.removeDeletes). \
                             Committing as-is would leave two DVs for one data file, which the \
                             scan rejects",
                            referenced,
                            dv_desc(existing)
                        ),
                    ));
                }
            } else {
                for referenced in added_dvs.keys() {
                    let Some((data_spec_id, data_partition, data_seq)) =
                        live_data_entry_by_path.get(referenced)
                    else {
                        continue;
                    };
                    let scope_matches = match referenced_data_file_location(existing) {
                        Some(path) => &path == referenced,
                        None => {
                            existing.partition_spec_id() == *data_spec_id
                                && existing.partition() == data_partition
                        }
                    };
                    let applies = scope_matches
                        && match (entry.sequence_number(), *data_seq) {
                            (Some(delete_seq), Some(data_seq)) => delete_seq >= data_seq,
                            _ => true,
                        };
                    if applies {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Cannot commit deletion vector for {}: live position delete file \
                                 {} still applies to that data file and would be silently \
                                 superseded by the DV at read time. Merging previous deletes into \
                                 the new DV (Java BaseDVFileWriter.loadPreviousDeletes) is deferred \
                                 in this port",
                                referenced,
                                existing.file_path()
                            ),
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}
