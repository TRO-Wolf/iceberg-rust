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

//! Bin-pack compaction. Java `RewriteDataFiles`. A wrong replaced set is silent corruption.
//!
//! # Algorithm (Java `BinPackRewriteFilePlanner.plan`)
//!
//! | step | rule |
//! |---|---|
//! | enumerate | scan the current snapshot with [`with_file_prune_only`](crate::scan::TableScanBuilder::with_file_prune_only) |
//! | group | file partition, or empty struct when the spec is not the table default |
//! | filter files | outside `[min_file_size, max_file_size]`, at `delete_file_threshold`, or at `delete_ratio_threshold` |
//! | bin-pack | forward greedy first-fit, lookback 1 |
//! | filter groups | enough files, enough content, too much content, or any delete-laden file |
//! | rewrite | live rows only; output files use the current spec and a tuple computed from each row. Format v3 writes stored `_row_id` / `_last_updated_sequence_number` (Java `SparkRewriteTable.rewriteSchema`) |
//!
//! A non-default-spec file can hold several current partitions, so Java groups it as unpartitioned.
//!
//! # Defaults
//!
//! | option | default |
//! |---|---|
//! | `target_file_size_bytes` | `write.target-file-size-bytes`, 512 MiB |
//! | `min_file_size_bytes` | `0.75 * target` |
//! | `max_file_size_bytes` | `1.8 * target` |
//! | `min_input_files` | 5 |
//! | `delete_file_threshold` | disabled (`usize::MAX`) |
//! | `delete_ratio_threshold` | 0.3 |
//! | `max_file_group_size_bytes` | 100 GiB |
//! | `use_starting_sequence_number` | true |
//! | `remove_dangling_deletes` | false |
//! | `max_open_partition_writers` | 64 |
//! | `filter` | always true |
//!
//! `remove-dangling-deletes` runs after a non-empty plan only. Failure fails the action.
//!
//! # Deferred
//!
//! | not ported | consequence |
//! |---|---|
//! | sort and Z-order | only bin-pack is ported |
//! | oversized-file splitting | an input over `max_file_size` is rewritten whole |

use std::collections::{HashMap, HashSet};

use crate::Catalog;
use crate::error::{Error, ErrorKind, Result};
use crate::expr::Predicate;
pub use crate::maintenance::rewrite_data_files_plan::RewriteJobOrder;
use crate::maintenance::rewrite_data_files_plan::{
    DELETE_FILE_THRESHOLD_DEFAULT, DELETE_RATIO_THRESHOLD_DEFAULT,
    PARTIAL_PROGRESS_MAX_COMMITS_DEFAULT, ResolvedConfig, format_java_double, order_groups,
    plan_commit_batches, plan_file_groups,
};
pub(super) use crate::maintenance::rewrite_data_files_plan::{
    MAX_FILE_GROUP_SIZE_BYTES_DEFAULT, MAX_FILE_SIZE_DEFAULT_RATIO,
    MAX_OPEN_PARTITION_WRITERS_DEFAULT, MIN_FILE_SIZE_DEFAULT_RATIO, MIN_INPUT_FILES_DEFAULT,
    pack_bins, parse_target_file_size,
};
#[cfg(test)]
use crate::maintenance::rewrite_data_files_plan::{group_qualifies, is_candidate};
use crate::maintenance::{RemoveDanglingDeleteFiles, rewrite_data_files_dv as rewrite_dv};
use crate::scan::FileScanTask;
use crate::spec::{DataFile, PartitionSpecRef};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};

/// The outcome of a [`RewriteDataFiles::execute`] run (Java `RewriteDataFiles.Result`).
///
/// A no-op plan returns zero counts and no groups, and commits no snapshot.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RewriteDataFilesResult {
    /// Data files added across all groups (Java `Result.addedDataFilesCount()`).
    pub added_data_files_count: usize,
    /// Data files replaced across all groups (Java `Result.rewrittenDataFilesCount()`).
    pub rewritten_data_files_count: usize,
    /// Bytes of the replaced input files (Java `Result.rewrittenBytesCount()`).
    pub rewritten_bytes_count: u64,
    /// Delete files removed: apply-path DVs (Java Result), apply-path file-scoped parquet (fork), and the composed dangling-delete sub-action.
    pub removed_delete_files_count: usize,
    /// Per-group results, in commit order (Java `Result.rewriteResults()`).
    pub file_groups: Vec<FileGroupRewriteResult>,
}

/// The result of rewriting a single file group (Java `RewriteDataFiles.FileGroupRewriteResult`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileGroupRewriteResult {
    /// Data files added for this group (Java `addedDataFilesCount()`).
    pub added_data_files_count: usize,
    /// Data files replaced in this group (Java `rewrittenDataFilesCount()`).
    pub rewritten_data_files_count: usize,
    /// Bytes of the replaced input files in this group (Java `rewrittenBytesCount()`).
    pub rewritten_bytes_count: u64,
}

/// The bin-pack compaction action. Build it with [`RewriteDataFiles::new`], configure the
/// thresholds, then run [`Self::execute`]. The module docs carry the algorithm, the
/// sequence-number rule, and the defaults.
pub struct RewriteDataFiles {
    table: Table,
    /// `None` resolves from the table property at execute (Java `defaultTargetFileSize`).
    target_file_size_bytes: Option<u64>,
    /// `None` resolves to `0.75 * target` at execute.
    min_file_size_bytes: Option<u64>,
    /// `None` resolves to `1.8 * target` at execute.
    max_file_size_bytes: Option<u64>,
    min_input_files: usize,
    delete_file_threshold: usize,
    delete_ratio_threshold: f64,
    max_file_group_size_bytes: u64,
    use_starting_sequence_number: bool,
    /// Java `REMOVE_DANGLING_DELETES`, default `false`.
    remove_dangling_deletes: bool,
    max_open_partition_writers: Option<usize>,
    filter: Predicate,
    rewrite_all: bool,
    partial_progress: bool,
    partial_progress_max_commits: usize,
    output_spec_id: Option<i32>,
    rewrite_job_order: RewriteJobOrder,
    max_concurrent_file_group_rewrites: usize,
}

impl RewriteDataFiles {
    /// Creates the action for `table` with Java's defaults. [`Self::execute`] resolves the size
    /// thresholds from `write.target-file-size-bytes` when they are not overridden.
    pub fn new(table: Table) -> Self {
        RewriteDataFiles {
            table,
            target_file_size_bytes: None,
            min_file_size_bytes: None,
            max_file_size_bytes: None,
            min_input_files: MIN_INPUT_FILES_DEFAULT,
            delete_file_threshold: DELETE_FILE_THRESHOLD_DEFAULT,
            delete_ratio_threshold: DELETE_RATIO_THRESHOLD_DEFAULT,
            max_file_group_size_bytes: MAX_FILE_GROUP_SIZE_BYTES_DEFAULT,
            use_starting_sequence_number: true,
            remove_dangling_deletes: false,
            max_open_partition_writers: None,
            filter: Predicate::AlwaysTrue,
            rewrite_all: false,
            partial_progress: false,
            partial_progress_max_commits: PARTIAL_PROGRESS_MAX_COMMITS_DEFAULT,
            output_spec_id: None,
            rewrite_job_order: RewriteJobOrder::None,
            max_concurrent_file_group_rewrites: 1,
        }
    }

    /// Sets the target output file size (Java `TARGET_FILE_SIZE_BYTES`). It also shifts the
    /// default `min` and `max` thresholds, unless those are overridden too.
    pub fn target_file_size_bytes(mut self, target_file_size_bytes: u64) -> Self {
        self.target_file_size_bytes = Some(target_file_size_bytes);
        self
    }

    /// A file smaller than this is always a candidate (Java `MIN_FILE_SIZE_BYTES`).
    pub fn min_file_size_bytes(mut self, min_file_size_bytes: u64) -> Self {
        self.min_file_size_bytes = Some(min_file_size_bytes);
        self
    }

    /// A file larger than this is always a candidate (Java `MAX_FILE_SIZE_BYTES`).
    pub fn max_file_size_bytes(mut self, max_file_size_bytes: u64) -> Self {
        self.max_file_size_bytes = Some(max_file_size_bytes);
        self
    }

    /// A group with at least this many files is rewritten whatever its size (Java
    /// `MIN_INPUT_FILES`). [`Self::execute`] rejects zero.
    pub fn min_input_files(mut self, min_input_files: usize) -> Self {
        self.min_input_files = min_input_files;
        self
    }

    /// A file with this many delete files is a candidate whatever its size, and its group is
    /// rewritten whatever its file count (Java `DELETE_FILE_THRESHOLD`).
    pub fn delete_file_threshold(mut self, delete_file_threshold: usize) -> Self {
        self.delete_file_threshold = delete_file_threshold;
        self
    }

    /// File-scoped delete ratio that always admits a file (Java `DELETE_RATIO_THRESHOLD`). Default 0.3. Must be in `(0, 1]`.
    pub fn delete_ratio_threshold(mut self, delete_ratio_threshold: f64) -> Self {
        self.delete_ratio_threshold = delete_ratio_threshold;
        self
    }

    /// The largest total input size of one group (Java `MAX_FILE_GROUP_SIZE_BYTES`). Must be
    /// greater than zero.
    pub fn max_file_group_size_bytes(mut self, max_file_group_size_bytes: u64) -> Self {
        self.max_file_group_size_bytes = max_file_group_size_bytes;
        self
    }

    /// Whether to stamp rewritten files with the starting snapshot's sequence number (Java
    /// `USE_STARTING_SEQUENCE_NUMBER`). Keep it true whenever the table carries outstanding
    /// merge-on-read deletes. See [the sequence-number rule](self#the-sequence-number-rule).
    pub fn use_starting_sequence_number(mut self, use_starting_sequence_number: bool) -> Self {
        self.use_starting_sequence_number = use_starting_sequence_number;
        self
    }

    /// Whether to run [`RemoveDanglingDeleteFiles`] after the group loop (Java
    /// `REMOVE_DANGLING_DELETES`). It runs only on a non-empty plan, and its total folds into
    /// [`RewriteDataFilesResult::removed_delete_files_count`]. See
    /// [the sub-action section](self#the-composed-remove-dangling-deletes-sub-action).
    pub fn remove_dangling_deletes(mut self, remove_dangling_deletes: bool) -> Self {
        self.remove_dangling_deletes = remove_dangling_deletes;
        self
    }

    /// Caps how many partition writers stay open at once. Default 64. Zero is rejected.
    pub fn max_open_partition_writers(mut self, max_open_partition_writers: usize) -> Self {
        self.max_open_partition_writers = Some(max_open_partition_writers);
        self
    }

    /// Restricts the rewrite to files matching `filter` (Java `RewriteDataFiles.filter`). The
    /// predicate selects files only; no residual applies. Every live row of a selected file is
    /// rewritten, so a co-located non-matching row survives.
    pub fn filter(mut self, filter: Predicate) -> Self {
        self.filter = filter;
        self
    }

    /// Rewrites every file and qualifies every group, bypassing all size filters (Java `REWRITE_ALL`).
    pub fn rewrite_all(mut self, rewrite_all: bool) -> Self {
        self.rewrite_all = rewrite_all;
        self
    }

    /// Commits rewritten groups in batches instead of one atomic commit (Java `PARTIAL_PROGRESS_ENABLED`).
    pub fn partial_progress(mut self, partial_progress: bool) -> Self {
        self.partial_progress = partial_progress;
        self
    }

    /// Caps the commit count under partial progress; groups per commit round up (Java `PARTIAL_PROGRESS_MAX_COMMITS`).
    pub fn partial_progress_max_commits(mut self, partial_progress_max_commits: usize) -> Self {
        self.partial_progress_max_commits = partial_progress_max_commits;
        self
    }

    /// Writes and groups output files under this partition spec id (Java `OUTPUT_SPEC_ID`).
    pub fn output_spec_id(mut self, output_spec_id: i32) -> Self {
        self.output_spec_id = Some(output_spec_id);
        self
    }

    /// Rewrites and commits groups in this order (Java `REWRITE_JOB_ORDER`).
    pub fn rewrite_job_order(mut self, rewrite_job_order: RewriteJobOrder) -> Self {
        self.rewrite_job_order = rewrite_job_order;
        self
    }

    /// Accepted group-rewrite parallelism; the fork rewrites sequentially (Java `MAX_CONCURRENT_FILE_GROUP_REWRITES`).
    pub fn max_concurrent_file_group_rewrites(
        mut self,
        max_concurrent_file_group_rewrites: usize,
    ) -> Self {
        self.max_concurrent_file_group_rewrites = max_concurrent_file_group_rewrites;
        self
    }

    /// Plans the compaction, rewrites each group into target-sized files, and commits through
    /// [`RewriteFilesAction`](crate::transaction::rewrite_files): one commit for all groups by
    /// default, batched commits under partial progress. Each group is read with merge-on-read
    /// deletes applied, so the output carries only live rows. When no file qualifies it returns
    /// zero counts and commits nothing.
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<RewriteDataFilesResult> {
        let mut config = self.resolve_config()?;
        let output_spec = self.resolve_output_spec_in(&self.table)?;

        let Some(starting_snapshot) = self.table.metadata().current_snapshot().cloned() else {
            return Ok(RewriteDataFilesResult::default());
        };
        let starting_snapshot_id = starting_snapshot.snapshot_id();
        let starting_sequence_number = starting_snapshot.sequence_number();

        let tasks = self.plan_scan_tasks().await?;
        let data_files_by_path = self.collect_live_data_files().await?;
        let live_deletes = rewrite_dv::live_file_scoped_position_deletes(&self.table).await?;
        config.file_scoped_delete_paths = rewrite_dv::file_scoped_delete_paths_from(&live_deletes);
        let mut groups = plan_file_groups(tasks, &config, &output_spec);

        if groups.is_empty() {
            return Ok(RewriteDataFilesResult::default());
        }
        order_groups(&mut groups, self.rewrite_job_order);

        let batches = plan_commit_batches(
            groups.len(),
            self.partial_progress,
            self.partial_progress_max_commits,
        );
        let mut result = RewriteDataFilesResult::default();
        let mut table = self.table.clone();
        let mut cursor = 0;
        for batch_size in batches {
            let mut batch_deletes: Vec<DataFile> = Vec::new();
            let mut batch_adds: Vec<DataFile> = Vec::new();
            let mut batch_dv_removed: Vec<DataFile> = Vec::new();
            let mut batch_dv_count = 0;
            for _ in 0..batch_size {
                let written = self
                    .write_group(
                        &table,
                        &groups[cursor],
                        &data_files_by_path,
                        &live_deletes,
                        &config,
                        &output_spec,
                    )
                    .await?;
                cursor += 1;
                result.added_data_files_count += written.result.added_data_files_count;
                result.rewritten_data_files_count += written.result.rewritten_data_files_count;
                result.rewritten_bytes_count += written.result.rewritten_bytes_count;
                result.file_groups.push(written.result);
                batch_dv_count += written.dv_removed_count;
                batch_deletes.extend(written.files_to_delete);
                batch_adds.extend(written.added_files);
                batch_dv_removed.extend(written.dv_removed);
            }
            let transaction = Transaction::new(&table);
            let mut action = transaction
                .rewrite_files(batch_deletes, batch_adds)
                .validate_from_snapshot(starting_snapshot_id);
            if self.use_starting_sequence_number {
                action = action.data_sequence_number(starting_sequence_number);
            }
            if !batch_dv_removed.is_empty() {
                action = action.delete_delete_files(batch_dv_removed);
            }
            let transaction = action.apply(transaction)?;
            // The committed table is the base for the next batch's commit.
            table = transaction.commit(catalog).await?;
            result.removed_delete_files_count += batch_dv_count;
        }

        // Java's two empty-result early returns precede this step, so it runs only on a non-empty
        // plan. `table` is the last batch's committed table, which is what Java's own handle
        // observes. A failure propagates and fails the whole action, as it does in Java.
        if self.remove_dangling_deletes {
            let removed = RemoveDanglingDeleteFiles::new(table)
                .execute(catalog)
                .await?;
            result.removed_delete_files_count += removed.removed_delete_files.len();
        }

        Ok(result)
    }

    /// Resolves the thresholds with Java's defaults, then applies Java's `sizeThresholds`
    /// preconditions.
    fn resolve_config(&self) -> Result<ResolvedConfig> {
        let target = match self.target_file_size_bytes {
            Some(target) => target,
            None => parse_target_file_size(self.table.metadata().properties())?,
        };

        if target == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("'target-file-size-bytes' is set to {target} but must be > 0"),
            ));
        }

        let default_min = (target as f64 * MIN_FILE_SIZE_DEFAULT_RATIO) as u64;
        let default_max = (target as f64 * MAX_FILE_SIZE_DEFAULT_RATIO) as u64;
        let min_file_size_bytes = self.min_file_size_bytes.unwrap_or(default_min);
        let max_file_size_bytes = self.max_file_size_bytes.unwrap_or(default_max);

        if target <= min_file_size_bytes {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "'target-file-size-bytes' ({target}) must be > 'min-file-size-bytes' \
                     ({min_file_size_bytes}), all new files will be smaller than the min threshold"
                ),
            ));
        }
        if target >= max_file_size_bytes {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "'target-file-size-bytes' ({target}) must be < 'max-file-size-bytes' \
                     ({max_file_size_bytes}), all new files will be larger than the max threshold"
                ),
            ));
        }
        if self.min_input_files == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "'min-input-files' is set to 0 but must be > 0",
            ));
        }
        if self.max_file_group_size_bytes == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "'max-file-group-size-bytes' is set to 0 but must be > 0",
            ));
        }
        if self.delete_ratio_threshold.is_nan() || self.delete_ratio_threshold <= 0.0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "'delete-ratio-threshold' is set to {} but must be > 0",
                    format_java_double(self.delete_ratio_threshold)
                ),
            ));
        }
        if self.delete_ratio_threshold > 1.0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "'delete-ratio-threshold' is set to {} but must be <= 1",
                    format_java_double(self.delete_ratio_threshold)
                ),
            ));
        }

        let max_open_partition_writers = self
            .max_open_partition_writers
            .unwrap_or(MAX_OPEN_PARTITION_WRITERS_DEFAULT);
        if max_open_partition_writers == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "'max-open-partition-writers' is set to 0 but must be > 0",
            ));
        }
        if self.max_concurrent_file_group_rewrites == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot set max-concurrent-file-group-rewrites to {}, the value must be positive.",
                    self.max_concurrent_file_group_rewrites
                ),
            ));
        }
        if self.partial_progress && self.partial_progress_max_commits == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot set partial-progress.max-commits to {}, the value must be positive when partial-progress.enabled is true",
                    self.partial_progress_max_commits
                ),
            ));
        }

        Ok(ResolvedConfig {
            target_file_size_bytes: target,
            min_file_size_bytes,
            max_file_size_bytes,
            min_input_files: self.min_input_files,
            delete_file_threshold: self.delete_file_threshold,
            delete_ratio_threshold: self.delete_ratio_threshold,
            max_file_group_size_bytes: self.max_file_group_size_bytes,
            max_open_partition_writers,
            rewrite_all: self.rewrite_all,
            file_scoped_delete_paths: HashSet::new(),
        })
    }

    fn resolve_output_spec_in(&self, table: &Table) -> Result<PartitionSpecRef> {
        match self.output_spec_id {
            None => Ok(table.metadata().default_partition_spec().clone()),
            Some(spec_id) => table
                .metadata()
                .partition_specs_iter()
                .find(|spec| spec.spec_id() == spec_id)
                .cloned()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot use output spec id {spec_id} because the table does not contain a reference to this spec-id."
                        ),
                    )
                }),
        }
    }

    /// Plan live data-file scan tasks. The filter selects files only; no residual applies.
    async fn plan_scan_tasks(&self) -> Result<Vec<FileScanTask>> {
        use futures::TryStreamExt;

        let stream = self
            .table
            .scan()
            .with_file_prune_only(self.filter.clone())
            .build()?
            .plan_files()
            .await?;
        stream.try_collect().await
    }

    /// Maps each live data file path to its [`DataFile`]. The rewrite removal set needs the whole
    /// file, and a scan task carries only the path.
    async fn collect_live_data_files(&self) -> Result<HashMap<String, DataFile>> {
        use crate::spec::DataContentType;

        let mut by_path: HashMap<String, DataFile> = HashMap::new();
        let metadata = self.table.metadata();
        let Some(snapshot) = metadata.current_snapshot() else {
            return Ok(by_path);
        };
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

struct WrittenGroup {
    result: FileGroupRewriteResult,
    files_to_delete: Vec<DataFile>,
    added_files: Vec<DataFile>,
    dv_removed: Vec<DataFile>,
    dv_removed_count: usize,
}

impl RewriteDataFiles {
    async fn write_group(
        &self,
        table: &Table,
        group: &[FileScanTask],
        data_files_by_path: &HashMap<String, DataFile>,
        live_file_scoped_deletes: &[(DataFile, String)],
        config: &ResolvedConfig,
        output_spec: &PartitionSpecRef,
    ) -> Result<WrittenGroup> {
        // A path that vanished since planning means a concurrent commit removed it. Fail here,
        // naming the file, rather than in the writer's own missing-path check.
        let mut files_to_delete: Vec<DataFile> = Vec::with_capacity(group.len());
        let mut rewritten_bytes_count: u64 = 0;
        for task in group {
            let data_file = data_files_by_path
                .get(task.data_file_path())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot rewrite data file {}: it is no longer a live data file in the \
                         current snapshot (a concurrent commit removed it)",
                            task.data_file_path()
                        ),
                    )
                })?;
            rewritten_bytes_count = rewritten_bytes_count.saturating_add(task.file_size_in_bytes);
            files_to_delete.push(data_file.clone());
        }

        let added_files = crate::maintenance::rewrite_data_files_write::write_compacted_files(
            table,
            group,
            config.target_file_size_bytes,
            config.max_open_partition_writers,
            output_spec,
        )
        .await?
        .files;

        let result = FileGroupRewriteResult {
            added_data_files_count: added_files.len(),
            rewritten_data_files_count: files_to_delete.len(),
            rewritten_bytes_count,
        };

        let rewritten_paths: HashSet<String> = files_to_delete
            .iter()
            .map(|file| file.file_path().to_string())
            .collect();
        let dv_plan = rewrite_dv::plan_dv_removal(live_file_scoped_deletes, &rewritten_paths);

        Ok(WrittenGroup {
            result,
            files_to_delete,
            added_files,
            dv_removed: dv_plan.removed,
            dv_removed_count: dv_plan.removed_count,
        })
    }
}

#[cfg(test)]
impl RewriteDataFiles {
    pub(crate) async fn write_compacted_files(
        &self,
        table: &Table,
        group: &[FileScanTask],
        target_file_size_bytes: u64,
    ) -> Result<Vec<DataFile>> {
        let output_spec = self.resolve_output_spec_in(table)?;
        crate::maintenance::rewrite_data_files_write::write_compacted_files(
            table,
            group,
            target_file_size_bytes,
            self.max_open_partition_writers
                .unwrap_or(MAX_OPEN_PARTITION_WRITERS_DEFAULT),
            &output_spec,
        )
        .await
        .map(|compacted| compacted.files)
    }

    pub(crate) fn resolved_max_open_partition_writers(&self) -> Result<usize> {
        Ok(self.resolve_config()?.max_open_partition_writers)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use futures::TryStreamExt;
    use tempfile::TempDir;

    use super::*;
    use crate::io::LocalFsStorageFactory;
    use crate::memory::MemoryCatalogBuilder;
    use crate::spec::{
        DataContentType, DataFile, DataFileFormat, Literal, NestedField, PartitionSpec,
        PrimitiveType, Schema, Struct, Transform, Type,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::equality_delete_writer::{
        EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
    };
    use crate::writer::base_writer::position_delete_writer::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
    };
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

    // Test harness: a local-fs memory catalog with real parquet on disk, and a table partitioned by
    // identity(x) over three long columns.

    /// A memory catalog over the local filesystem. Returns the catalog and the temp-dir guard.
    pub(crate) async fn local_fs_catalog() -> (impl Catalog, TempDir) {
        let temp_dir = TempDir::new().expect("temp dir");
        let warehouse = temp_dir
            .path()
            .to_str()
            .expect("utf8 temp path")
            .to_string();
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "memory",
                HashMap::from([("warehouse".to_string(), warehouse)]),
            )
            .await
            .expect("load local-fs memory catalog");
        (catalog, temp_dir)
    }

    /// A schema of three required long columns `x`, `y`, `z`.
    fn three_long_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "x",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::required(
                    2,
                    "y",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::required(
                    3,
                    "z",
                    Type::Primitive(PrimitiveType::Long),
                )),
            ])
            .build()
            .expect("build schema")
    }

    /// A table partitioned by identity(x), format version `format_version`, under a fresh namespace.
    pub(crate) async fn create_partitioned_table(
        catalog: &impl Catalog,
        format_version: crate::spec::FormatVersion,
    ) -> Table {
        let schema = three_long_schema();
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_partition_field("x", "x", Transform::Identity)
            .expect("add partition field")
            .build()
            .expect("build spec");
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace");
        let table_ident = TableIdent::new(namespace.clone(), "t".to_string());
        let creation = TableCreation::builder()
            .name(table_ident.name().to_string())
            .schema(schema)
            .partition_spec(spec)
            .format_version(format_version)
            .build();
        catalog
            .create_table(&namespace, creation)
            .await
            .expect("create table")
    }

    /// Writes a real parquet data file into partition `x = part_value`.
    pub(crate) async fn write_data_file(
        table: &Table,
        file_name: &str,
        part_value: i64,
        rows: &[(i64, i64, i64)],
    ) -> DataFile {
        use crate::arrow::schema_to_arrow_schema;

        let schema = table.metadata().current_schema();
        let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());

        let xs: Vec<i64> = rows.iter().map(|(x, _, _)| *x).collect();
        let ys: Vec<i64> = rows.iter().map(|(_, y, _)| *y).collect();
        let zs: Vec<i64> = rows.iter().map(|(_, _, z)| *z).collect();
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from(xs)) as ArrayRef,
            Arc::new(Int64Array::from(ys)) as ArrayRef,
            Arc::new(Int64Array::from(zs)) as ArrayRef,
        ])
        .unwrap();

        let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
        let output = table.file_io().new_output(file_path).unwrap();
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            schema.clone(),
        );
        let mut writer = parquet_builder.build(output).await.unwrap();
        writer.write(&batch).await.unwrap();
        let data_file_builders = writer.close().await.unwrap();

        let mut builder = data_file_builders.into_iter().next().unwrap();
        builder
            .content(DataContentType::Data)
            .partition_spec_id(0)
            .partition(Struct::from_iter([Some(Literal::long(part_value))]))
            .build()
            .unwrap()
    }

    /// Writes a real equality-delete file on `y`, in partition `x = part_value`.
    pub(crate) async fn write_equality_delete_file(
        table: &Table,
        part_value: i64,
        delete_ys: &[i64],
    ) -> DataFile {
        use crate::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};

        let schema = table.metadata().current_schema().clone();
        let config = EqualityDeleteWriterConfig::new(vec![2], schema.clone()).unwrap();
        let delete_schema =
            Arc::new(arrow_schema_to_schema(config.projected_arrow_schema_ref()).unwrap());

        let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
        let file_name_gen = DefaultFileNameGenerator::new(
            "eq-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            delete_schema,
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );

        let partition_key = crate::spec::PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            schema.clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
            .build(Some(partition_key))
            .await
            .unwrap();

        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let xs: Vec<i64> = delete_ys.iter().map(|_| part_value).collect();
        let ys: Vec<i64> = delete_ys.to_vec();
        let zs: Vec<i64> = delete_ys.iter().map(|_| 0).collect();
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from(xs)) as ArrayRef,
            Arc::new(Int64Array::from(ys)) as ArrayRef,
            Arc::new(Int64Array::from(zs)) as ArrayRef,
        ])
        .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap().into_iter().next().unwrap()
    }

    /// Writes a real parquet position-delete file in partition `x = part_value`.
    pub(crate) async fn write_position_delete_file(
        table: &Table,
        part_value: i64,
        deletes: &[(String, i64)],
    ) -> DataFile {
        use arrow_array::StringArray;

        let config = PositionDeleteWriterConfig::new().unwrap();
        let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
        let file_name_gen = DefaultFileNameGenerator::new(
            "pos-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            config.schema().clone(),
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );
        let partition_key = crate::spec::PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            table.metadata().current_schema().clone(),
            Struct::from_iter([Some(Literal::long(part_value))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
            .build(Some(partition_key))
            .await
            .unwrap();

        let paths: Vec<&str> = deletes.iter().map(|(path, _)| path.as_str()).collect();
        let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
        let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions)) as ArrayRef,
        ])
        .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap().into_iter().next().unwrap()
    }

    /// Append `files` in one fast-append commit, returning the updated table.
    pub(crate) async fn append_files(
        catalog: &impl Catalog,
        table: &Table,
        files: Vec<DataFile>,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx.fast_append().add_data_files(files);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Add `deletes` (a row_delta) in one commit, returning the updated table.
    pub(crate) async fn add_deletes(
        catalog: &impl Catalog,
        table: &Table,
        deletes: Vec<DataFile>,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx.row_delta().add_deletes(deletes);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Collects every row the scan returns, with merge-on-read deletes applied.
    pub(crate) async fn scan_rows(table: &Table) -> Vec<(i64, i64, i64)> {
        let stream = table
            .scan()
            .select(["x", "y", "z"])
            .build()
            .unwrap()
            .to_arrow()
            .await
            .unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let mut rows: Vec<(i64, i64, i64)> = Vec::new();
        for batch in batches {
            let xs = column_i64(&batch, "x");
            let ys = column_i64(&batch, "y");
            let zs = column_i64(&batch, "z");
            for index in 0..xs.len() {
                rows.push((xs.value(index), ys.value(index), zs.value(index)));
            }
        }
        rows.sort_unstable();
        rows
    }

    /// Downcast a named column of `batch` to `Int64Array`.
    fn column_i64<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int64Array {
        let index = batch.schema().index_of(name).unwrap();
        batch
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
    }

    /// The set of live (Added/Existing) data-file paths in the table's current snapshot.
    pub(crate) async fn live_data_file_paths(table: &Table) -> HashSet<String> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut paths = HashSet::new();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.is_alive() && entry.content_type() == DataContentType::Data {
                    paths.insert(entry.file_path().to_string());
                }
            }
        }
        paths
    }

    /// The current snapshot id (or `None` for a fresh table).
    pub(crate) fn current_snapshot_id(table: &Table) -> Option<i64> {
        table.metadata().current_snapshot_id()
    }

    /// The explicit on-disk data sequence number of every live data file, read from the raw avro
    /// with no inheritance. A file that would re-inherit the snapshot's number maps to `None`.
    async fn on_disk_data_seqs(table: &Table) -> HashMap<String, Option<i64>> {
        use crate::spec::Manifest;

        let mut seqs = HashMap::new();
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        for manifest_file in manifest_list.entries() {
            let bytes = table
                .file_io()
                .new_input(&manifest_file.manifest_path)
                .unwrap()
                .read()
                .await
                .unwrap();
            let (_, raw_entries) = Manifest::try_from_avro_bytes(&bytes).unwrap();
            for entry in raw_entries {
                if entry.is_alive() && entry.content_type() == DataContentType::Data {
                    seqs.insert(entry.file_path().to_string(), entry.sequence_number());
                }
            }
        }
        seqs
    }

    // E2E tests on the local-fs MemoryCatalog + real parquet.

    /// Row conservation. A compaction that drops or duplicates a row is silent corruption, so the
    /// post-compaction row set must equal the pre-compaction one exactly, sorted, and the file
    /// count must drop.
    #[tokio::test]
    async fn test_bin_pack_compaction_conserves_every_row_exactly() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // Distinct y per file, so a drop or a duplicate is detectable.
        let mut files = Vec::new();
        for index in 0..6i64 {
            files.push(
                write_data_file(&table, &format!("small-{index}.parquet"), 0, &[(
                    0,
                    100 + index,
                    1000 + index,
                )])
                .await,
            );
        }
        let table = append_files(&catalog, &table, files).await;

        let rows_before = scan_rows(&table).await;
        let files_before = live_data_file_paths(&table).await.len();
        assert_eq!(files_before, 6, "fixture: 6 small files before compaction");

        // Target above the sum packs all six into one group, and a huge min makes each undersized.
        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1_000_000)
            .execute(&catalog)
            .await
            .expect("compaction must succeed");

        let table = catalog.load_table(table.identifier()).await.unwrap();
        let rows_after = scan_rows(&table).await;
        let files_after = live_data_file_paths(&table).await.len();

        assert_eq!(
            rows_after, rows_before,
            "the post-compaction scan must return EXACTLY the pre-compaction live rows (no drop, no dup)"
        );
        assert!(
            files_after < files_before,
            "compaction must reduce the file count ({files_before} -> {files_after})"
        );
        assert_eq!(
            result.rewritten_data_files_count, 6,
            "all 6 input files were rewritten"
        );
        assert_eq!(
            result.added_data_files_count, files_after,
            "the result's added count matches the new live file count"
        );
    }

    /// Write-path fidelity on a richer schema. A type can drift through the arrow round trip: a
    /// decimal can lose precision, a timestamp can lose its unit, and a missing field id makes the
    /// rewritten parquet unreadable. Every row must survive byte-exactly, the parquet must carry
    /// the iceberg field ids, and the committed record count must be right.
    #[tokio::test]
    async fn test_richer_schema_compaction_conserves_rows_field_ids_and_stats() {
        use arrow_array::{Decimal128Array, TimestampMicrosecondArray};
        use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

        use crate::arrow::{UTC_TIME_ZONE, schema_to_arrow_schema};

        let (catalog, _temp) = local_fs_catalog().await;

        // Schema: id long, amount decimal(9,2), ts timestamptz. Unpartitioned.
        let schema = Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::required(
                    2,
                    "amount",
                    Type::Primitive(PrimitiveType::Decimal {
                        precision: 9,
                        scale: 2,
                    }),
                )),
                Arc::new(NestedField::required(
                    3,
                    "ts",
                    Type::Primitive(PrimitiveType::Timestamptz),
                )),
            ])
            .build()
            .expect("build richer schema");
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("create namespace");
        let table_ident = TableIdent::new(namespace.clone(), "rich".to_string());
        let creation = TableCreation::builder()
            .name(table_ident.name().to_string())
            .schema(schema.clone())
            .format_version(crate::spec::FormatVersion::V2)
            .build();
        let table = catalog
            .create_table(&namespace, creation)
            .await
            .expect("create richer table");

        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());

        // Amount keeps two decimals, and each ts is a distinct micro value.
        let mut files = Vec::new();
        for index in 0..5i64 {
            // Unscaled 10_000 at scale 2 is 100.00.
            let amount = Decimal128Array::from(vec![10_000 + index as i128])
                .with_precision_and_scale(9, 2)
                .unwrap();
            let ts = TimestampMicrosecondArray::from(vec![1_700_000_000_000_000 + index])
                .with_timezone(UTC_TIME_ZONE);
            let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Int64Array::from(vec![index])) as ArrayRef,
                Arc::new(amount) as ArrayRef,
                Arc::new(ts) as ArrayRef,
            ])
            .unwrap();

            let file_path = format!("{}/data/rich-{index}.parquet", table.metadata().location());
            let output = table.file_io().new_output(file_path).unwrap();
            let parquet_builder = ParquetWriterBuilder::new(
                parquet::file::properties::WriterProperties::builder().build(),
                Arc::new(schema.clone()),
            );
            let mut writer = parquet_builder.build(output).await.unwrap();
            writer.write(&batch).await.unwrap();
            let mut builder = writer.close().await.unwrap().into_iter().next().unwrap();
            files.push(
                builder
                    .content(DataContentType::Data)
                    .partition(Struct::empty())
                    .build()
                    .unwrap(),
            );
        }
        let table = append_files(&catalog, &table, files).await;

        // A sortable signal for the before state.
        let read_rich = |table: Table| async move {
            let stream = table.scan().build().unwrap().to_arrow().await.unwrap();
            let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
            let mut rows: Vec<(i64, i128, i64)> = Vec::new();
            for batch in batches {
                let ids = column_i64(&batch, "id");
                let amounts = batch
                    .column(batch.schema().index_of("amount").unwrap())
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .unwrap();
                let timestamps = batch
                    .column(batch.schema().index_of("ts").unwrap())
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap();
                for index in 0..ids.len() {
                    rows.push((
                        ids.value(index),
                        amounts.value(index),
                        timestamps.value(index),
                    ));
                }
            }
            rows.sort_unstable();
            rows
        };
        let rows_before = read_rich(table.clone()).await;
        assert_eq!(rows_before.len(), 5, "fixture: 5 rows");

        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1_000_000)
            .execute(&catalog)
            .await
            .expect("richer-schema compaction must succeed");
        assert_eq!(result.rewritten_data_files_count, 5);

        let table = catalog.load_table(table.identifier()).await.unwrap();
        let rows_after = read_rich(table.clone()).await;
        assert_eq!(
            rows_after, rows_before,
            "every row survives byte-exactly through the decimal/timestamp arrow round-trip"
        );

        // The parquet must carry the field ids, and the record count must match the conserved rows.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut total_records: u64 = 0;
        let mut checked_field_ids = false;
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if !(entry.is_alive() && entry.content_type() == DataContentType::Data) {
                    continue;
                }
                total_records += entry.data_file().record_count();
                // Read the rewritten parquet's field-id metadata.
                let input = table.file_io().new_input(entry.file_path()).unwrap();
                let bytes = input.read().await.unwrap();
                let arrow_reader_builder =
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes)
                        .unwrap();
                let field_ids: HashSet<i32> = arrow_reader_builder
                    .schema()
                    .fields()
                    .iter()
                    .filter_map(|field| {
                        field
                            .metadata()
                            .get(PARQUET_FIELD_ID_META_KEY)
                            .and_then(|value| value.parse::<i32>().ok())
                    })
                    .collect();
                assert_eq!(
                    field_ids,
                    HashSet::from([1, 2, 3]),
                    "the rewritten parquet must carry iceberg field IDs 1,2,3 in its arrow schema"
                );
                checked_field_ids = true;
            }
        }
        assert!(
            checked_field_ids,
            "at least one rewritten file was inspected"
        );
        assert_eq!(
            total_records, 5,
            "the committed DataFile record counts sum to the 5 conserved rows"
        );
    }

    /// Filter-leak guard, a data-loss class. The filter must select files only. If it leaks into
    /// the group read, compacting a partially matching file silently discards its other live rows.
    #[tokio::test]
    async fn test_filtered_compaction_keeps_non_matching_live_rows() {
        use crate::expr::Reference;
        use crate::spec::Datum;

        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // Each file holds one row below y = 100 and one at or above it.
        let mut files = Vec::new();
        for index in 0..5i64 {
            files.push(
                write_data_file(&table, &format!("split-{index}.parquet"), 0, &[
                    (0, 10 + index, 1000 + index),
                    (0, 100 + index, 2000 + index),
                ])
                .await,
            );
        }
        let table = append_files(&catalog, &table, files).await;

        let rows_before = scan_rows(&table).await;
        assert_eq!(rows_before.len(), 10, "fixture: 10 live rows (2 per file)");
        assert!(
            rows_before.iter().any(|(_, y, _)| *y < 100)
                && rows_before.iter().any(|(_, y, _)| *y >= 100),
            "fixture: rows straddle the y=100 filter boundary"
        );

        // The filter selects the files, but must not drop their y < 100 rows.
        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1_000_000)
            .filter(Reference::new("y").greater_than_or_equal_to(Datum::long(100)))
            .execute(&catalog)
            .await
            .expect("filtered compaction must succeed");
        assert_eq!(
            result.rewritten_data_files_count, 5,
            "all 5 selected files were rewritten"
        );

        let table = catalog.load_table(table.identifier()).await.unwrap();
        let rows_after = scan_rows(&table).await;
        assert_eq!(
            rows_after, rows_before,
            "EVERY live row survives a filtered compaction — the y<100 rows of the rewritten files \
             must NOT be dropped (the filter is a file-selection device, not a row filter on the read)"
        );
    }

    /// A partition filter must leave a non-matching partition's files untouched while still
    /// rewriting the matching undersized ones. Ignoring `self.filter` would rewrite every
    /// partition and still pass the row-conservation tests.
    #[tokio::test]
    async fn test_filtered_compaction_excludes_non_matching_partition_files() {
        use crate::expr::Reference;
        use crate::spec::Datum;

        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // 5 undersized files in partition x=0 + 1 undersized file in partition x=1.
        let mut files = Vec::new();
        for index in 0..5i64 {
            files.push(
                write_data_file(&table, &format!("p0-{index}.parquet"), 0, &[(
                    0, index, index,
                )])
                .await,
            );
        }
        let p1 = write_data_file(&table, "p1-keep.parquet", 1, &[(1, 99, 99)]).await;
        let p1_path = p1.file_path().to_string();
        files.push(p1);
        let table = append_files(&catalog, &table, files).await;

        let rows_before = scan_rows(&table).await;
        assert_eq!(rows_before.len(), 6);

        // Filter x == 0: only partition-0 files are candidates. p1 must stay.
        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1_000_000)
            .filter(Reference::new("x").equal_to(Datum::long(0)))
            .execute(&catalog)
            .await
            .expect("filtered compaction");
        assert_eq!(
            result.rewritten_data_files_count, 5,
            "only the 5 x==0 files are rewritten"
        );

        let table = catalog.load_table(table.identifier()).await.unwrap();
        let paths_after: std::collections::HashSet<String> = {
            use futures::TryStreamExt;
            table
                .scan()
                .build()
                .unwrap()
                .plan_files()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .into_iter()
                .map(|t| t.data_file_path.to_string())
                .collect()
        };
        assert!(
            paths_after.contains(&p1_path),
            "partition x==1 file must remain (filter excluded it from rewrite): {paths_after:?}"
        );

        let rows_after = scan_rows(&table).await;
        assert_eq!(
            rows_after, rows_before,
            "all live rows conserved across filtered rewrite"
        );
    }

    /// Candidate selection both ways: a target-sized file keeps its path, and an undersized file is
    /// rewritten.
    #[tokio::test]
    async fn test_target_sized_file_untouched_undersized_rewritten() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // Five small files form a qualifying group; the large one is well-sized.
        let mut small_files = Vec::new();
        for index in 0..5i64 {
            small_files.push(
                write_data_file(&table, &format!("s-{index}.parquet"), 0, &[(
                    0, index, index,
                )])
                .await,
            );
        }
        // A file with ~20 rows in partition x=1: bigger than the small ones.
        let big_rows: Vec<(i64, i64, i64)> = (0..200).map(|n| (1, n, n)).collect();
        let big = write_data_file(&table, "big.parquet", 1, &big_rows).await;
        let big_path = big.file_path().to_string();
        let big_size = big.file_size_in_bytes();

        let mut all = small_files;
        all.push(big);
        let table = append_files(&catalog, &table, all).await;

        let rows_before = scan_rows(&table).await;

        // Place the big file inside [min, max] and the small ones well below min.
        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(big_size)
            .min_file_size_bytes(big_size / 2)
            .max_file_size_bytes(big_size * 2)
            .execute(&catalog)
            .await
            .expect("compaction must succeed");

        let table = catalog.load_table(table.identifier()).await.unwrap();
        let live = live_data_file_paths(&table).await;

        assert!(
            live.contains(&big_path),
            "the well-sized file must be untouched (same path in the new snapshot)"
        );
        assert_eq!(
            result.rewritten_data_files_count, 5,
            "only the 5 undersized files were rewritten, not the well-sized one"
        );
        assert_eq!(
            scan_rows(&table).await,
            rows_before,
            "row conservation across the selective compaction"
        );
    }

    /// The resurrection guard. A compaction that lifts the rewritten data above an outstanding
    /// equality delete resurrects deleted rows. With the starting sequence number preserved, the
    /// delete still applies and the scan still drops y=20.
    ///
    /// Mutation: pass `use_starting_sequence_number(false)` and y=20 resurrects.
    #[tokio::test]
    async fn test_compaction_preserves_outstanding_equality_delete_no_resurrection() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // One row per file, y = 10..50, including the y=20 the delete targets.
        let mut files = Vec::new();
        for index in 0..5i64 {
            let y = 10 + index * 10; // 10, 20, 30, 40, 50
            files.push(
                write_data_file(&table, &format!("d-{index}.parquet"), 0, &[(0, y, y * 10)]).await,
            );
        }
        let table = append_files(&catalog, &table, files).await;

        // An equality delete (equality_ids=[y]) removing y=20, at a higher seq.
        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

        // Before compaction the scan drops y=20.
        let rows_before = scan_rows(&table).await;
        assert!(
            !rows_before.iter().any(|(_, y, _)| *y == 20),
            "the equality delete drops y=20 before compaction"
        );

        // Compact with use_starting_sequence_number = TRUE (default).
        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1_000_000)
            .execute(&catalog)
            .await
            .expect("compaction must succeed on a table with outstanding deletes");
        assert_eq!(result.rewritten_data_files_count, 5);

        let table = catalog.load_table(table.identifier()).await.unwrap();
        let rows_after = scan_rows(&table).await;
        assert!(
            !rows_after.iter().any(|(_, y, _)| *y == 20),
            "with the starting seq preserved, the equality delete STILL drops y=20 — no resurrection"
        );
        assert_eq!(
            rows_after, rows_before,
            "the live row set is unchanged by the compaction (deletes still applied)"
        );
    }

    /// The sequence-number mechanism, pinned on the raw on-disk value in both directions. A
    /// compaction reads deletes-applied, so an existing delete's rows go regardless. The stamped
    /// number is what keeps a *concurrent* equality delete applying.
    ///
    /// Mutation: drop the `data_sequence_number` call in `rewrite_group` and the true branch sees a
    /// fresh, higher number.
    #[tokio::test]
    async fn test_rewritten_file_carries_starting_seq_with_flag_else_fresh() {
        let (catalog, _temp) = local_fs_catalog().await;

        // With the flag on, the on-disk number equals the starting snapshot's.
        {
            let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
            let mut files = Vec::new();
            for index in 0..5i64 {
                files.push(
                    write_data_file(&table, &format!("d-{index}.parquet"), 0, &[(
                        0, index, index,
                    )])
                    .await,
                );
            }
            let table = append_files(&catalog, &table, files).await;
            // The data must stay below this delete's sequence number.
            let eq_delete = write_equality_delete_file(&table, 0, &[2]).await;
            let table = add_deletes(&catalog, &table, vec![eq_delete]).await;
            let starting_seq = table
                .metadata()
                .current_snapshot()
                .unwrap()
                .sequence_number();

            let old_paths = live_data_file_paths(&table).await;
            RewriteDataFiles::new(table.clone())
                .target_file_size_bytes(1_000_000)
                .execute(&catalog)
                .await
                .expect("compaction must succeed");

            let table = catalog.load_table(table.identifier()).await.unwrap();
            let seqs = on_disk_data_seqs(&table).await;
            let new_files: Vec<_> = seqs.keys().filter(|p| !old_paths.contains(*p)).collect();
            assert!(!new_files.is_empty(), "compaction produced new data files");
            for path in new_files {
                assert_eq!(
                    seqs[path],
                    Some(starting_seq),
                    "the rewritten file must carry the STARTING snapshot's seq EXPLICITLY on disk \
                     (so a concurrent equality delete at a higher seq still applies)"
                );
            }
        }

        // With the flag off, the rewritten file takes a fresh, higher number.
        {
            let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
            let mut files = Vec::new();
            for index in 0..5i64 {
                files.push(
                    write_data_file(&table, &format!("e-{index}.parquet"), 0, &[(
                        0, index, index,
                    )])
                    .await,
                );
            }
            let table = append_files(&catalog, &table, files).await;
            let starting_seq = table
                .metadata()
                .current_snapshot()
                .unwrap()
                .sequence_number();
            let old_paths = live_data_file_paths(&table).await;

            RewriteDataFiles::new(table.clone())
                .target_file_size_bytes(1_000_000)
                .use_starting_sequence_number(false)
                .execute(&catalog)
                .await
                .expect("compaction must succeed");

            let table = catalog.load_table(table.identifier()).await.unwrap();
            let new_seq = table
                .metadata()
                .current_snapshot()
                .unwrap()
                .sequence_number();
            assert!(
                new_seq > starting_seq,
                "the rewrite minted a new (higher) snapshot seq"
            );
            let seqs = on_disk_data_seqs(&table).await;
            for (path, seq) in &seqs {
                if !old_paths.contains(path) {
                    // No explicit stamp means it re-inherits the new, higher snapshot number.
                    assert_eq!(
                        *seq, None,
                        "without the flag the rewritten file has NO explicit seq (re-inherits fresh)"
                    );
                }
            }
        }
    }

    /// A compaction must physically drop a position-deleted row, after which the old position
    /// delete dangles harmlessly. The scan must not resurrect the row.
    #[tokio::test]
    async fn test_compaction_applies_position_delete_then_old_delete_dangles_harmlessly() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // 5 files in partition x=0; one has 2 rows so a position delete can remove its row 0.
        let mut files = Vec::new();
        let two_row =
            write_data_file(&table, "two-row.parquet", 0, &[(0, 11, 110), (0, 22, 220)]).await;
        let two_row_path = two_row.file_path().to_string();
        files.push(two_row);
        for index in 0..4i64 {
            files.push(
                write_data_file(&table, &format!("one-{index}.parquet"), 0, &[(
                    0,
                    30 + index,
                    300,
                )])
                .await,
            );
        }
        let table = append_files(&catalog, &table, files).await;

        // A position delete removing row 0 (y=11) of two-row.parquet.
        let pos_delete = write_position_delete_file(&table, 0, &[(two_row_path.clone(), 0)]).await;
        let table = add_deletes(&catalog, &table, vec![pos_delete]).await;

        let rows_before = scan_rows(&table).await;
        assert!(
            !rows_before.iter().any(|(_, y, _)| *y == 11),
            "the position delete drops y=11 (row 0) before compaction"
        );

        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1_000_000)
            .execute(&catalog)
            .await
            .expect("compaction must succeed");
        assert!(result.rewritten_data_files_count >= 5);

        let table = catalog.load_table(table.identifier()).await.unwrap();
        let rows_after = scan_rows(&table).await;
        assert!(
            !rows_after.iter().any(|(_, y, _)| *y == 11),
            "the rewritten file contains only live rows; the dangling position delete is harmless"
        );
        assert_eq!(
            rows_after, rows_before,
            "row conservation: only the position-deleted row stays gone"
        );
        // The compacted file no longer exists, so the position delete dangles (Java keeps it).
        assert!(
            !live_data_file_paths(&table).await.contains(&two_row_path),
            "the position-deleted file was rewritten away (its delete now dangles)"
        );
    }

    /// A delete-laden but well-sized file must still be rewritten, or its deletes are never applied
    /// physically. At threshold 1 the lone well-sized file is a candidate and its group qualifies.
    #[tokio::test]
    async fn test_delete_threshold_triggers_rewrite_of_well_sized_delete_laden_file() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // A single, reasonably-sized file in partition x=0 with several rows.
        let rows: Vec<(i64, i64, i64)> = (0..50).map(|n| (0, n, n * 10)).collect();
        let data = write_data_file(&table, "laden.parquet", 0, &rows).await;
        let data_path = data.file_path().to_string();
        let data_size = data.file_size_in_bytes();
        let table = append_files(&catalog, &table, vec![data]).await;

        // A position delete removing y=0 (row 0).
        let pos_delete = write_position_delete_file(&table, 0, &[(data_path.clone(), 0)]).await;
        let table = add_deletes(&catalog, &table, vec![pos_delete]).await;

        let rows_before = scan_rows(&table).await;
        assert!(!rows_before.iter().any(|(_, y, _)| *y == 0));

        // The file is well-sized, so only the delete threshold can select it.
        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(data_size)
            .min_file_size_bytes(data_size / 2)
            .max_file_size_bytes(data_size * 2)
            .delete_file_threshold(1)
            .execute(&catalog)
            .await
            .expect("compaction must succeed");

        assert_eq!(
            result.rewritten_data_files_count, 1,
            "the delete-laden well-sized file IS rewritten via the delete threshold"
        );
        let table = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            !live_data_file_paths(&table).await.contains(&data_path),
            "the delete-laden file was rewritten (its delete physically applied)"
        );
        assert_eq!(
            scan_rows(&table).await,
            rows_before,
            "the rewrite physically applied the delete: y=0 stays gone, all else conserved"
        );
    }

    /// The delete threshold must not over-fire. A well-sized file under the threshold is left
    /// alone.
    #[tokio::test]
    async fn test_delete_threshold_under_count_leaves_well_sized_file_alone() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        let rows: Vec<(i64, i64, i64)> = (0..50).map(|n| (0, n, n * 10)).collect();
        let data = write_data_file(&table, "laden.parquet", 0, &rows).await;
        let data_path = data.file_path().to_string();
        let data_size = data.file_size_in_bytes();
        let table = append_files(&catalog, &table, vec![data]).await;

        let pos_delete = write_position_delete_file(&table, 0, &[(data_path.clone(), 0)]).await;
        let table = add_deletes(&catalog, &table, vec![pos_delete]).await;

        // One delete against a threshold of 2, and well-sized, so the plan is empty.
        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(data_size)
            .min_file_size_bytes(data_size / 2)
            .max_file_size_bytes(data_size * 2)
            .delete_file_threshold(2)
            .execute(&catalog)
            .await
            .expect("execute must succeed (no-op)");

        assert_eq!(
            result,
            RewriteDataFilesResult::default(),
            "an under-threshold well-sized file is left alone (no-op)"
        );
        let table = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            live_data_file_paths(&table).await.contains(&data_path),
            "the file is untouched"
        );
    }

    /// Partition isolation. Packing two partitions into one group would give a file carrying rows
    /// of two partition values. Every output file must hold one partition's rows.
    #[tokio::test]
    async fn test_partitions_never_pack_into_one_group() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        let mut files = Vec::new();
        for index in 0..3i64 {
            files.push(
                write_data_file(&table, &format!("p0-{index}.parquet"), 0, &[(
                    0, index, index,
                )])
                .await,
            );
            files.push(
                write_data_file(&table, &format!("p1-{index}.parquet"), 1, &[(
                    1, index, index,
                )])
                .await,
            );
        }
        let table = append_files(&catalog, &table, files).await;
        let rows_before = scan_rows(&table).await;

        // min_input_files = 3 so each 3-file partition group qualifies on its own.
        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1_000_000)
            .min_input_files(3)
            .execute(&catalog)
            .await
            .expect("compaction must succeed");
        assert_eq!(
            result.rewritten_data_files_count, 6,
            "all 6 files rewritten"
        );

        let table = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            scan_rows(&table).await,
            rows_before,
            "row conservation across partitions"
        );

        // A cross-partition group would give a file whose tuple contradicts its rows. Each of x=0
        // and x=1 must have at least one output file.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut output_partition_values: HashSet<i64> = HashSet::new();
        let mut output_file_count = 0usize;
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.is_alive() && entry.content_type() == DataContentType::Data {
                    output_file_count += 1;
                    // The partition struct's single field is the x value.
                    match entry.data_file().partition().iter().next() {
                        Some(Some(Literal::Primitive(prim))) => {
                            let value: i64 = format!("{prim:?}")
                                .trim_start_matches("Long(")
                                .trim_end_matches(')')
                                .parse()
                                .expect("partition x value parses");
                            output_partition_values.insert(value);
                        }
                        other => panic!("unexpected partition tuple shape: {other:?}"),
                    }
                }
            }
        }
        assert_eq!(
            output_partition_values,
            HashSet::from([0, 1]),
            "each partition (x=0, x=1) produced its own output file(s), never a mixed group"
        );
        // 6 small files → 2 partitions → 2 output files (one compacted file per partition).
        assert_eq!(
            output_file_count, 2,
            "one compacted output file per partition"
        );
    }

    /// A run with no qualifying file must commit no snapshot and return zero counts.
    #[tokio::test]
    async fn test_empty_plan_is_a_no_op_with_no_commit() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // One well-sized file in partition x=0.
        let rows: Vec<(i64, i64, i64)> = (0..100).map(|n| (0, n, n)).collect();
        let data = write_data_file(&table, "ok.parquet", 0, &rows).await;
        let data_size = data.file_size_in_bytes();
        let table = append_files(&catalog, &table, vec![data]).await;

        let snapshots_before = table.metadata().snapshots().count();
        let snapshot_id_before = current_snapshot_id(&table);

        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(data_size)
            .min_file_size_bytes(data_size / 2)
            .max_file_size_bytes(data_size * 2)
            .execute(&catalog)
            .await
            .expect("execute must succeed (no-op)");

        assert_eq!(
            result,
            RewriteDataFilesResult::default(),
            "an empty plan returns a zero-count result"
        );
        let table = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            table.metadata().snapshots().count(),
            snapshots_before,
            "no snapshot was committed for an empty plan"
        );
        assert_eq!(
            current_snapshot_id(&table),
            snapshot_id_before,
            "the current snapshot is unchanged"
        );
    }

    /// A table with no current snapshot is a clean no-op, not a crash.
    #[tokio::test]
    async fn test_fresh_table_with_no_snapshot_is_a_no_op() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
        assert!(
            current_snapshot_id(&table).is_none(),
            "fixture: no snapshot yet"
        );

        let result = RewriteDataFiles::new(table.clone())
            .execute(&catalog)
            .await
            .expect("a fresh table is a clean no-op");
        assert_eq!(result, RewriteDataFilesResult::default());
    }

    /// A lone undersized file is left alone. Every group clause needs more than one file, or an
    /// oversized input, so a one-file group is churn with no benefit.
    #[tokio::test]
    async fn test_lone_small_file_below_group_minimum_is_left_alone() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // A single tiny file in partition x=0 — undersized, but a group of ONE.
        let data = write_data_file(&table, "lone.parquet", 0, &[(0, 1, 1)]).await;
        let data_path = data.file_path().to_string();
        let table = append_files(&catalog, &table, vec![data]).await;

        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1_000_000)
            .execute(&catalog)
            .await
            .expect("execute must succeed (no-op)");

        assert_eq!(
            result,
            RewriteDataFilesResult::default(),
            "a lone undersized file (group of 1) is left alone (size > 1 required to qualify)"
        );
        let table = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            live_data_file_paths(&table).await.contains(&data_path),
            "the lone file is untouched"
        );
    }

    /// The `min_input_files` boundary. Three files at a minimum of 3 qualify, and two do not.
    #[tokio::test]
    async fn test_min_input_files_boundary_two_below_three_at() {
        let (catalog, _temp) = local_fs_catalog().await;

        // Case A: 2 files, min_input_files=3 ⇒ NOT enough ⇒ no-op.
        {
            let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
            let files = vec![
                write_data_file(&table, "a.parquet", 0, &[(0, 1, 1)]).await,
                write_data_file(&table, "b.parquet", 0, &[(0, 2, 2)]).await,
            ];
            let table = append_files(&catalog, &table, files).await;
            let result = RewriteDataFiles::new(table.clone())
                .target_file_size_bytes(1_000_000)
                .min_input_files(3)
                .execute(&catalog)
                .await
                .unwrap();
            assert_eq!(
                result,
                RewriteDataFilesResult::default(),
                "2 files < min_input_files 3 and inputSize < target ⇒ no-op"
            );
        }

        // Case B: 3 files, min_input_files=3 ⇒ enough ⇒ rewritten.
        {
            let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
            let files = vec![
                write_data_file(&table, "a.parquet", 0, &[(0, 1, 1)]).await,
                write_data_file(&table, "b.parquet", 0, &[(0, 2, 2)]).await,
                write_data_file(&table, "c.parquet", 0, &[(0, 3, 3)]).await,
            ];
            let table = append_files(&catalog, &table, files).await;
            let result = RewriteDataFiles::new(table.clone())
                .target_file_size_bytes(1_000_000)
                .min_input_files(3)
                .execute(&catalog)
                .await
                .unwrap();
            assert_eq!(
                result.rewritten_data_files_count, 3,
                "3 files == min_input_files 3 ⇒ the group qualifies and is rewritten"
            );
        }
    }

    /// The aggregate counts must equal the work done: six files rewritten, one added, and the
    /// input bytes summed.
    #[tokio::test]
    async fn test_result_counts_match_the_actual_rewrite() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        let mut files = Vec::new();
        let mut expected_bytes: u64 = 0;
        for index in 0..6i64 {
            let file = write_data_file(&table, &format!("s-{index}.parquet"), 0, &[(
                0, index, index,
            )])
            .await;
            expected_bytes += file.file_size_in_bytes();
            files.push(file);
        }
        let table = append_files(&catalog, &table, files).await;

        let result = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(10_000_000)
            .execute(&catalog)
            .await
            .expect("compaction must succeed");

        assert_eq!(
            result.rewritten_data_files_count, 6,
            "6 input files rewritten"
        );
        assert_eq!(
            result.rewritten_bytes_count, expected_bytes,
            "rewritten bytes = sum of the input file sizes"
        );
        let table = catalog.load_table(table.identifier()).await.unwrap();
        let added = live_data_file_paths(&table).await.len();
        assert_eq!(
            result.added_data_files_count, added,
            "added count matches the new live file count"
        );
        assert_eq!(result.file_groups.len(), 1, "one group was rewritten");
        assert_eq!(result.file_groups[0].rewritten_data_files_count, 6);
        assert_eq!(result.file_groups[0].rewritten_bytes_count, expected_bytes);
    }

    /// A compaction must not commit over a concurrent conflicting delete. The group commit runs
    /// `RewriteFiles`' validate, which rejects a new position delete on a replaced file. This
    /// proves the action does not bypass that validation.
    #[tokio::test]
    async fn test_concurrent_conflicting_delete_fails_the_compaction_commit() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // 5 small files in partition x=0 (a qualifying group). One file has 2 rows so a concurrent
        // position delete can target it.
        let target =
            write_data_file(&table, "target.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
        let target_path = target.file_path().to_string();
        let mut files = vec![target];
        for index in 0..4i64 {
            files.push(
                write_data_file(&table, &format!("o-{index}.parquet"), 0, &[(
                    0,
                    30 + index,
                    300,
                )])
                .await,
            );
        }
        let table = append_files(&catalog, &table, files).await;

        // Reproduce the commit path by hand so a concurrent commit can interleave. The action
        // commits each group in its own transaction, which hits this same validate.
        let action = RewriteDataFiles::new(table.clone()).target_file_size_bytes(1_000_000);
        let config = action.resolve_config().unwrap();
        let starting = table.metadata().current_snapshot().unwrap().clone();
        let tasks = action.plan_scan_tasks().await.unwrap();
        let data_files_by_path = action.collect_live_data_files().await.unwrap();
        let groups = plan_file_groups(tasks, &config, table.metadata().default_partition_spec());
        assert_eq!(groups.len(), 1, "fixture: one qualifying group");

        // Build the rewrite tx for the group (read + write new files), but do NOT commit yet.
        let group = &groups[0];
        let mut files_to_delete: Vec<DataFile> = Vec::new();
        for task in group {
            files_to_delete.push(
                data_files_by_path
                    .get(task.data_file_path())
                    .unwrap()
                    .clone(),
            );
        }
        let added = action
            .write_compacted_files(&table, group, config.target_file_size_bytes)
            .await
            .unwrap();
        let transaction = Transaction::new(&table);
        let rewrite = transaction
            .rewrite_files(files_to_delete, added)
            .validate_from_snapshot(starting.snapshot_id())
            .data_sequence_number(starting.sequence_number());
        let transaction = rewrite.apply(transaction).unwrap();

        // Concurrent: a NEW position delete targeting the replaced file lands.
        let pos_delete = write_position_delete_file(&table, 0, &[(target_path.clone(), 1)]).await;
        let _concurrent = add_deletes(&catalog, &table, vec![pos_delete]).await;

        let error = transaction
            .commit(&catalog)
            .await
            .expect_err("a concurrent position delete on a replaced file must fail the commit");
        assert!(
            error
                .message()
                .contains("found new position delete for replaced data file"),
            "unexpected error: {}",
            error.message()
        );
    }

    /// The load-bearing role of `use_starting_sequence_number`. An equality delete that lands after
    /// the starting snapshot is captured but before the commit must still apply. Keeping the lower
    /// number does that, and the commit still succeeds because preserving it sets
    /// `ignore_equality_deletes` in the validate.
    #[tokio::test]
    async fn test_concurrent_equality_delete_still_applies_after_compaction() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // 5 small files in partition x=0 (a qualifying group), rows y = 10, 20, 30, 40, 50.
        let mut files = Vec::new();
        for index in 0..5i64 {
            let y = 10 + index * 10;
            files.push(
                write_data_file(&table, &format!("c-{index}.parquet"), 0, &[(0, y, y * 10)]).await,
            );
        }
        let table = append_files(&catalog, &table, files).await;

        // Pre-compaction the table has NO deletes — y=20 is live.
        let rows_before = scan_rows(&table).await;
        assert!(
            rows_before.iter().any(|(_, y, _)| *y == 20),
            "fixture: y=20 is live before the concurrent delete"
        );

        // --- Drive the action's internals to interleave a concurrent commit (the action commits each
        // group in its own tx, so a concurrent delete landing between plan and commit hits exactly
        // this path). Capture the STARTING snapshot S now.
        let action = RewriteDataFiles::new(table.clone()).target_file_size_bytes(1_000_000);
        let config = action.resolve_config().unwrap();
        let starting = table.metadata().current_snapshot().unwrap().clone();
        let tasks = action.plan_scan_tasks().await.unwrap();
        let data_files_by_path = action.collect_live_data_files().await.unwrap();
        let groups = plan_file_groups(tasks, &config, table.metadata().default_partition_spec());
        assert_eq!(groups.len(), 1, "fixture: one qualifying group");

        let group = &groups[0];
        let mut files_to_delete: Vec<DataFile> = Vec::new();
        for task in group {
            files_to_delete.push(
                data_files_by_path
                    .get(task.data_file_path())
                    .unwrap()
                    .clone(),
            );
        }
        // Read from the starting snapshot, which has no deletes yet, so y=20 is carried.
        let added = action
            .write_compacted_files(&table, group, config.target_file_size_bytes)
            .await
            .unwrap();

        // The concurrent delete lands after the starting snapshot and before the commit.
        let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
        let concurrent = add_deletes(&catalog, &table, vec![eq_delete]).await;
        assert!(
            concurrent
                .metadata()
                .current_snapshot()
                .unwrap()
                .sequence_number()
                > starting.sequence_number(),
            "the concurrent equality delete is at a strictly higher seq than the starting snapshot"
        );

        // Stamping the starting number commits over the concurrent delete without conflict.
        let transaction = Transaction::new(&table);
        let rewrite = transaction
            .rewrite_files(files_to_delete, added)
            .validate_from_snapshot(starting.snapshot_id())
            .data_sequence_number(starting.sequence_number());
        let transaction = rewrite.apply(transaction).unwrap();
        transaction
            .commit(&catalog)
            .await
            .expect("the seq-preserving compaction commits over a concurrent equality delete");

        // y=20 is gone: the concurrent delete still applies to the lower-numbered data.
        let table = catalog.load_table(table.identifier()).await.unwrap();
        let rows_after = scan_rows(&table).await;
        assert!(
            !rows_after.iter().any(|(_, y, _)| *y == 20),
            "the concurrently-added equality delete STILL drops y=20 after compaction — no \
             resurrection (the rewritten data kept the starting seq, below the delete's seq)"
        );
        // Every other row is conserved.
        let expected: Vec<(i64, i64, i64)> = rows_before
            .into_iter()
            .filter(|(_, y, _)| *y != 20)
            .collect();
        assert_eq!(
            rows_after, expected,
            "exactly y=20 is removed; all other live rows survive the compaction"
        );
    }

    /// `validate_from_snapshot` through the production `execute()` path. Without it, a concurrent
    /// position delete on a replaced file slips through the commit and is lost.
    ///
    /// Mutation: drop the `validate_from_snapshot` call in `rewrite_group` and `execute()` succeeds.
    #[tokio::test]
    async fn test_execute_rejects_concurrent_position_delete_on_replaced_file() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;

        // 5 small files in partition x=0 (a qualifying group); the first has 2 rows so a concurrent
        // position delete can target it.
        let target = write_data_file(&table, "vfs-target.parquet", 0, &[
            (0, 10, 100),
            (0, 20, 200),
        ])
        .await;
        let target_path = target.file_path().to_string();
        let mut files = vec![target];
        for index in 0..4i64 {
            files.push(
                write_data_file(&table, &format!("vfs-o-{index}.parquet"), 0, &[(
                    0,
                    30 + index,
                    300,
                )])
                .await,
            );
        }
        let table = append_files(&catalog, &table, files).await;

        // The commit refreshes against the catalog head, where the concurrent delete now lives.
        let action = RewriteDataFiles::new(table.clone()).target_file_size_bytes(1_000_000);

        let pos_delete = write_position_delete_file(&table, 0, &[(target_path.clone(), 1)]).await;
        let _concurrent = add_deletes(&catalog, &table, vec![pos_delete]).await;

        let error = action
            .execute(&catalog)
            .await
            .expect_err("a concurrent position delete on a replaced file must fail .execute()");
        assert!(
            error
                .message()
                .contains("found new position delete for replaced data file"),
            "unexpected error: {}",
            error.message()
        );
    }

    /// A misconfigured threshold must not silently do the wrong thing. `target >= max` is rejected
    /// with Java's message.
    #[tokio::test]
    async fn test_invalid_size_thresholds_rejected() {
        let (catalog, _temp) = local_fs_catalog().await;
        let table = create_partitioned_table(&catalog, crate::spec::FormatVersion::V2).await;
        let data = write_data_file(&table, "a.parquet", 0, &[(0, 1, 1)]).await;
        let table = append_files(&catalog, &table, vec![data]).await;

        // target == max ⇒ rejected (`target < max` required).
        let error = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1000)
            .max_file_size_bytes(1000)
            .execute(&catalog)
            .await
            .expect_err("target >= max must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("must be < 'max-file-size-bytes'"),
            "unexpected message: {}",
            error.message()
        );

        // min_input_files == 0 ⇒ rejected.
        let error = RewriteDataFiles::new(table.clone())
            .target_file_size_bytes(1_000_000)
            .min_input_files(0)
            .execute(&catalog)
            .await
            .expect_err("min_input_files 0 must be rejected");
        assert!(error.message().contains("'min-input-files' is set to 0"));
    }

    // ----- pure planning-function unit tests (no table needed) -----

    /// A minimal [`FileScanTask`] with a size, an `x` partition value, and a delete count. It
    /// carries an identity spec so partition grouping works.
    pub(crate) fn synthetic_task(
        path: &str,
        size: u64,
        part_value: i64,
        delete_count: usize,
        spec: &Arc<crate::spec::PartitionSpec>,
        schema: &crate::spec::SchemaRef,
    ) -> FileScanTask {
        use crate::scan::FileScanTaskDeleteFile;

        let deletes: Vec<FileScanTaskDeleteFile> = (0..delete_count)
            .map(|index| FileScanTaskDeleteFile {
                file_path: format!("{path}.delete-{index}"),
                file_size_in_bytes: 1,
                file_type: DataContentType::PositionDeletes,
                partition_spec_id: 0,
                equality_ids: None,
                file_format: DataFileFormat::Parquet,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
                record_count: Some(0),
            })
            .collect();
        FileScanTask {
            file_size_in_bytes: size,
            start: 0,
            length: size,
            record_count: Some(1),
            data_file_path: Arc::from(path),
            data_file_format: DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: Arc::from(vec![1, 2, 3]),
            predicate: None,
            deletes: Arc::from(deletes),
            partition: Some(Struct::from_iter([Some(Literal::long(part_value))])),
            partition_spec: Some(spec.clone()),
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        }
    }

    /// The identity(x) spec + schema for the synthetic tasks (spec id 0).
    pub(crate) fn synthetic_spec_and_schema()
    -> (Arc<crate::spec::PartitionSpec>, crate::spec::SchemaRef) {
        let schema: crate::spec::SchemaRef = Arc::new(three_long_schema());
        let spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(0)
                .add_partition_field("x", "x", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );
        (spec, schema)
    }

    pub(crate) fn config_for(
        target: u64,
        min: u64,
        max: u64,
        min_input_files: usize,
    ) -> ResolvedConfig {
        ResolvedConfig {
            target_file_size_bytes: target,
            min_file_size_bytes: min,
            max_file_size_bytes: max,
            min_input_files,
            delete_file_threshold: DELETE_FILE_THRESHOLD_DEFAULT,
            delete_ratio_threshold: DELETE_RATIO_THRESHOLD_DEFAULT,
            max_file_group_size_bytes: 1_000_000,
            max_open_partition_writers: MAX_OPEN_PARTITION_WRITERS_DEFAULT,
            rewrite_all: false,
            file_scoped_delete_paths: HashSet::new(),
        }
    }

    /// Bin-packing parity with Java `ListPacker.pack`: `[3,3,3,3]` at target 6 gives `[[3,3],
    /// [3,3]]`, and `[4,3,3]` gives `[[4],[3,3]]`. This pins the forward order, which `packEnd`
    /// would reverse.
    #[test]
    fn test_pack_bins_forward_first_fit() {
        let (spec, schema) = synthetic_spec_and_schema();
        let sizes_of = |bins: &[Vec<FileScanTask>]| -> Vec<Vec<u64>> {
            bins.iter()
                .map(|bin| bin.iter().map(|task| task.file_size_in_bytes).collect())
                .collect()
        };

        let tasks: Vec<FileScanTask> = [3u64, 3, 3, 3]
            .iter()
            .enumerate()
            .map(|(index, &size)| synthetic_task(&format!("f{index}"), size, 0, 0, &spec, &schema))
            .collect();
        assert_eq!(
            sizes_of(&pack_bins(tasks, |task| task.file_size_in_bytes, 6)),
            vec![vec![3, 3], vec![3, 3]]
        );

        let tasks: Vec<FileScanTask> = [4u64, 3, 3]
            .iter()
            .enumerate()
            .map(|(index, &size)| synthetic_task(&format!("g{index}"), size, 0, 0, &spec, &schema))
            .collect();
        assert_eq!(
            sizes_of(&pack_bins(tasks, |task| task.file_size_in_bytes, 6)),
            vec![vec![4], vec![3, 3]]
        );

        // A single item over target gets its own bin.
        let tasks: Vec<FileScanTask> = [7u64, 2, 2]
            .iter()
            .enumerate()
            .map(|(index, &size)| synthetic_task(&format!("h{index}"), size, 0, 0, &spec, &schema))
            .collect();
        assert_eq!(
            sizes_of(&pack_bins(tasks, |task| task.file_size_in_bytes, 6)),
            vec![vec![7], vec![2, 2]]
        );
    }

    /// The candidate predicate. Undersized, oversized, or delete-laden qualifies; well-sized and
    /// delete-free does not.
    #[test]
    fn test_is_candidate_predicate() {
        let (spec, schema) = synthetic_spec_and_schema();
        // target 100, min 75, max 180.
        let mut config = config_for(100, 75, 180, 5);
        config.delete_file_threshold = 2;

        // Well-sized (100) + no deletes ⇒ NOT a candidate.
        let well = synthetic_task("w", 100, 0, 0, &spec, &schema);
        assert!(!is_candidate(&well, &config));
        // Undersized (50 < 75) ⇒ candidate.
        let small = synthetic_task("s", 50, 0, 0, &spec, &schema);
        assert!(is_candidate(&small, &config));
        // Oversized (200 > 180) ⇒ candidate.
        let big = synthetic_task("b", 200, 0, 0, &spec, &schema);
        assert!(is_candidate(&big, &config));
        // Well-sized but 2 deletes (>= threshold 2) ⇒ candidate via tooManyDeletes.
        let laden = synthetic_task("l", 100, 0, 2, &spec, &schema);
        assert!(is_candidate(&laden, &config));
        // Well-sized + 1 delete (< threshold 2) ⇒ NOT a candidate.
        let one_delete = synthetic_task("o", 100, 0, 1, &spec, &schema);
        assert!(!is_candidate(&one_delete, &config));
    }

    /// The group filter. A lone undersized file does not qualify, a five-file group does, and a
    /// lone oversized file does.
    #[test]
    fn test_group_filter() {
        let (spec, schema) = synthetic_spec_and_schema();
        let config = config_for(100, 75, 180, 5);

        let lone_small = vec![synthetic_task("s", 50, 0, 0, &spec, &schema)];
        assert!(
            !group_qualifies(&lone_small, &config),
            "a lone small file does not qualify"
        );

        let five_small: Vec<FileScanTask> = (0..5)
            .map(|index| synthetic_task(&format!("s{index}"), 50, 0, 0, &spec, &schema))
            .collect();
        assert!(
            group_qualifies(&five_small, &config),
            "5 files qualify via enoughInputFiles"
        );

        let lone_big = vec![synthetic_task("b", 200, 0, 0, &spec, &schema)];
        assert!(
            group_qualifies(&lone_big, &config),
            "an oversized file qualifies via tooMuchContent"
        );

        // Two small files whose sum exceeds target qualify via enoughContent.
        let two_over_target = vec![
            synthetic_task("a", 60, 0, 0, &spec, &schema),
            synthetic_task("b", 60, 0, 0, &spec, &schema),
        ];
        assert!(
            group_qualifies(&two_over_target, &config),
            "2 small files summing > target qualify via enoughContent"
        );
    }

    /// Partition grouping. Different partition values never share a group, and a task of a
    /// non-default spec buckets as unpartitioned.
    #[test]
    fn test_plan_file_groups_partition_isolation_and_incompatible_spec() {
        let (spec, schema) = synthetic_spec_and_schema();
        let config = config_for(100, 75, 180, 2);

        // 2 undersized files in x=0, 2 in x=1 ⇒ two groups, one per partition.
        let tasks = vec![
            synthetic_task("p0a", 10, 0, 0, &spec, &schema),
            synthetic_task("p0b", 10, 0, 0, &spec, &schema),
            synthetic_task("p1a", 10, 1, 0, &spec, &schema),
            synthetic_task("p1b", 10, 1, 0, &spec, &schema),
        ];
        let groups = plan_file_groups(tasks, &config, &spec);
        assert_eq!(groups.len(), 2, "two partitions ⇒ two groups");
        for group in &groups {
            let partitions: HashSet<String> = group
                .iter()
                .map(|task| format!("{:?}", task.partition))
                .collect();
            assert_eq!(
                partitions.len(),
                1,
                "each group holds ONE partition value only"
            );
        }

        // A task of an incompatible spec buckets under the empty struct.
        let old_spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(1)
                .add_partition_field("y", "y", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );
        // Both tasks carry the byte-identical partition struct `[0]`, so a naive "always key by
        // partition" co-groups them. Correct bucketing keeps them apart.
        let mut incompatible = synthetic_task("old", 10, 0, 0, &old_spec, &schema);
        incompatible.partition = Some(Struct::from_iter([Some(Literal::long(0))]));
        let current_file = synthetic_task("cur", 10, 0, 0, &spec, &schema);
        // A co-grouped 2-file bucket would qualify at min_input_files 2. Correct bucketing gives
        // two single-file buckets and zero groups, so dropping the spec check reddens this.
        let groups = plan_file_groups(vec![incompatible, current_file], &config, &spec);
        assert!(
            groups.is_empty(),
            "an incompatible-spec file and a current-spec file with the SAME partition struct are \
             bucketed SEPARATELY (incompatible ⇒ empty struct), never merged into a qualifying group"
        );
    }

    // The composed `remove-dangling-deletes` sub-action.

    /// The live delete-file paths of the current snapshot, the signal for whether a delete file was
    /// really removed.
    pub(crate) async fn live_delete_file_paths(table: &Table) -> HashSet<String> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let manifest_list = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let mut paths = HashSet::new();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
            for entry in manifest.entries() {
                if entry.is_alive() && entry.content_type() != DataContentType::Data {
                    paths.insert(entry.file_path().to_string());
                }
            }
        }
        paths
    }
}
