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

//! Planner predicates for [`super::RewriteDataFiles`]. Java `BinPackRewriteFilePlanner`.

use std::collections::{HashMap, HashSet};

use crate::error::{Error, ErrorKind, Result};
use crate::scan::{FileScanTask, FileScanTaskDeleteFile};
use crate::spec::{DataContentType, Struct, TableProperties};

/// Java `SizeBasedFileRewritePlanner.MIN_FILE_SIZE_DEFAULT_RATIO`.
pub(super) const MIN_FILE_SIZE_DEFAULT_RATIO: f64 = 0.75;

/// Java `SizeBasedFileRewritePlanner.MAX_FILE_SIZE_DEFAULT_RATIO`.
pub(super) const MAX_FILE_SIZE_DEFAULT_RATIO: f64 = 1.80;

/// Java `SizeBasedFileRewritePlanner.MIN_INPUT_FILES_DEFAULT`.
pub(super) const MIN_INPUT_FILES_DEFAULT: usize = 5;

/// Java `SizeBasedFileRewritePlanner.MAX_FILE_GROUP_SIZE_BYTES_DEFAULT`.
pub(super) const MAX_FILE_GROUP_SIZE_BYTES_DEFAULT: u64 = 100 * 1024 * 1024 * 1024;

/// Java `BinPackRewriteFilePlanner.DELETE_FILE_THRESHOLD_DEFAULT`. Disables the delete-count
/// clause.
pub(super) const DELETE_FILE_THRESHOLD_DEFAULT: usize = usize::MAX;

/// Java `BinPackRewriteFilePlanner.DELETE_RATIO_THRESHOLD_DEFAULT`.
pub(super) const DELETE_RATIO_THRESHOLD_DEFAULT: f64 = 0.3;

/// Thresholds after defaults and preconditions.
pub(super) struct ResolvedConfig {
    pub(super) target_file_size_bytes: u64,
    pub(super) min_file_size_bytes: u64,
    pub(super) max_file_size_bytes: u64,
    pub(super) min_input_files: usize,
    pub(super) delete_file_threshold: usize,
    pub(super) delete_ratio_threshold: f64,
    pub(super) max_file_group_size_bytes: u64,
    pub(super) file_scoped_delete_paths: HashSet<String>,
}

/// Groups scan tasks by partition, filters candidates, bin-packs, and filters groups. Java
/// `BinPackRewriteFilePlanner.planFileGroups`.
pub(super) fn plan_file_groups(
    tasks: Vec<FileScanTask>,
    config: &ResolvedConfig,
    default_spec: &crate::spec::PartitionSpecRef,
) -> Vec<Vec<FileScanTask>> {
    let default_spec_id = default_spec.spec_id();

    // Java `groupByPartition` keys on the file's partition only when its spec id is the table's
    // current default. Anything else goes in the unpartitioned bucket.
    let mut by_partition: HashMap<Struct, Vec<FileScanTask>> = HashMap::new();
    for task in tasks {
        let key = match (&task.partition, task_spec_id(&task)) {
            (Some(partition), Some(spec_id)) if spec_id == default_spec_id => partition.clone(),
            _ => Struct::empty(),
        };
        by_partition.entry(key).or_default().push(task);
    }

    let mut groups: Vec<Vec<FileScanTask>> = Vec::new();
    for (_partition, partition_tasks) in by_partition {
        let candidates: Vec<FileScanTask> = partition_tasks
            .into_iter()
            .filter(|task| is_candidate(task, config))
            .collect();
        if candidates.is_empty() {
            continue;
        }

        let bins = pack_bins(
            candidates,
            |task| task.file_size_in_bytes,
            config.max_file_group_size_bytes,
        );

        for bin in bins {
            if group_qualifies(&bin, config) {
                groups.push(bin);
            }
        }
    }
    groups
}

fn task_spec_id(task: &FileScanTask) -> Option<i32> {
    task.partition_spec.as_ref().map(|spec| spec.spec_id())
}

/// Java `BinPackRewriteFilePlanner.filterFiles`.
pub(super) fn is_candidate(task: &FileScanTask, config: &ResolvedConfig) -> bool {
    let length = task.file_size_in_bytes;
    let outside_desired_size =
        length < config.min_file_size_bytes || length > config.max_file_size_bytes;
    outside_desired_size || too_many_deletes(task, config) || too_high_delete_ratio(task, config)
}

/// Java `SizeBasedFileRewritePlanner.filterFileGroups` plus the two delete clauses.
pub(super) fn group_qualifies(group: &[FileScanTask], config: &ResolvedConfig) -> bool {
    let size = group.len();
    let input_size: u64 = group.iter().fold(0u64, |sum, task| {
        sum.saturating_add(task.file_size_in_bytes)
    });

    let enough_input_files = size > 1 && size >= config.min_input_files;
    let enough_content = size > 1 && input_size > config.target_file_size_bytes;
    let too_much_content = input_size > config.max_file_size_bytes;
    let any_too_many_deletes = group.iter().any(|task| too_many_deletes(task, config));
    let any_too_high_delete_ratio = group.iter().any(|task| too_high_delete_ratio(task, config));

    enough_input_files
        || enough_content
        || too_much_content
        || any_too_many_deletes
        || any_too_high_delete_ratio
}

/// Java `BinPackRewriteFilePlanner.tooManyDeletes`.
fn too_many_deletes(task: &FileScanTask, config: &ResolvedConfig) -> bool {
    task.deletes.len() >= config.delete_file_threshold
}

/// Java `BinPackRewriteFilePlanner.tooHighDeleteRatio`.
pub(super) fn too_high_delete_ratio(task: &FileScanTask, config: &ResolvedConfig) -> bool {
    if task.deletes.is_empty() {
        return false;
    }
    let Some(data_record_count) = task.record_count else {
        return false;
    };
    if data_record_count == 0 {
        return false;
    }
    let known_deleted: u64 = task
        .deletes
        .iter()
        .filter(|delete| is_file_scoped_scan_delete(delete, config))
        .map(|delete| delete.record_count.unwrap_or(0))
        .sum();
    let deleted = known_deleted.min(data_record_count);
    (deleted as f64) / (data_record_count as f64) >= config.delete_ratio_threshold
}

fn is_file_scoped_scan_delete(delete: &FileScanTaskDeleteFile, config: &ResolvedConfig) -> bool {
    if delete.file_type == DataContentType::EqualityDeletes {
        return false;
    }
    delete.referenced_data_file.is_some()
        || config.file_scoped_delete_paths.contains(&delete.file_path)
}

pub(super) fn parse_target_file_size(properties: &HashMap<String, String>) -> Result<u64> {
    match properties.get(TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES) {
        None => Ok(TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES_DEFAULT as u64),
        Some(value) => value.parse::<u64>().map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid value '{value}' for table property \
                     '{}'",
                    TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES
                ),
            )
            .with_source(error)
        }),
    }
}

/// Forward greedy first-fit bin-packing. Java `BinPacking.ListPacker` with lookback 1.
pub(super) fn pack_bins<T>(
    items: Vec<T>,
    weight: impl Fn(&T) -> u64,
    target_weight: u64,
) -> Vec<Vec<T>> {
    let mut bins: Vec<Vec<T>> = Vec::new();
    let mut open_bin: Vec<T> = Vec::new();
    let mut open_weight: u64 = 0;

    for item in items {
        let w = weight(&item);
        if !open_bin.is_empty() && open_weight.saturating_add(w) <= target_weight {
            open_weight = open_weight.saturating_add(w);
            open_bin.push(item);
        } else {
            if !open_bin.is_empty() {
                bins.push(std::mem::take(&mut open_bin));
            }
            open_weight = w;
            open_bin.push(item);
        }
    }
    if !open_bin.is_empty() {
        bins.push(open_bin);
    }
    bins
}
