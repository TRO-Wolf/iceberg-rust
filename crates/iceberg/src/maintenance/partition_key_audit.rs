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

//! [`AuditPartitionKeys`] + [`RepairPartitionKeys`] — detect data files whose RECORDED manifest
//! partition tuple disagrees with the tuple RECOMPUTED from the file's own rows, and repair them by
//! rewriting the rows under the correct partition keys.
//!
//! **FORK-ORIGINAL.** There is no Java `org.apache.iceberg.actions` counterpart, so these are NOT
//! [`ActionsProvider`](crate::maintenance::ActionsProvider) methods (that surface mirrors Java's
//! javap-confirmed 12-method set and stays closed). What IS Java-pinned is everything they lean on:
//! the partition transforms ([`crate::transform`]), the source-column projection + transform chain
//! ([`PartitionValueCalculator`](crate::arrow::PartitionValueCalculator) /
//! [`RecordBatchPartitionSplitter`]), and the commit vehicle
//! ([`RewriteFilesAction`](crate::transaction::rewrite_files) = Java `BaseRewriteFiles`).
//!
//! # Why this exists — the exposure window
//!
//! Verbatim from the 2026-07-25 back-to-goal brief (Unit 2, G1):
//!
//! > commits via the DataFusion provider's `insert_into` into a partitioned table where the SELECT
//! > list contained a computed or reordered partition-source column; `VALUES` and plain passthrough
//! > are clean.
//!
//! The mechanism (fixed in `PartitionExpr`, Unit 1): the expression that computed the partition
//! tuple declared no children, so DataFusion's `ProjectionPushdown` re-parented it onto a different
//! input and the tuple was computed from the WRONG batch. The result is a **real-but-wrong**
//! partition tuple in the manifest — nothing in the commit path rejects it (Java and Rust both
//! validate the tuple's ARITY and TYPES against the spec, never its VALUES), and nothing in the read
//! path notices, so the damage is silent:
//!
//! - a partition-pruned read DROPS the file (its rows vanish from query results, in Rust AND in
//!   Java/Spark — the tuple is on-disk metadata, not an engine artifact);
//! - an unpruned read hands back the RECORDED value for every identity-partitioned column, because
//!   partition metadata is authoritative over the file's own column (Iceberg "Column Projection"
//!   rule 1 / Java `PartitionUtil.constantsMap`) — the rows come back with the wrong values.
//!
//! Both actions here are offline and mechanical: they need only the table and its storage.
//!
//! # Detection
//!
//! For every LIVE data file in the current snapshot: read the file's rows, recompute the partition
//! tuple per row through the file's own partition spec (identity / bucket / truncate / temporal /
//! void — every transform is a pure function of the source columns, so all are recomputable), and
//! compare against the tuple recorded in the manifest entry. A file whose rows do not ALL recompute
//! to the recorded tuple is reported as a [`PartitionKeyFinding`] carrying the path, the recorded
//! tuple, every distinct recomputed tuple, and the mismatching row count.
//!
//! Two implementation decisions carry the whole audit:
//!
//! 1. **The read must not be told the partition tuple.** A [`FileScanTask`] carries `partition` +
//!    `partition_spec` so the reader can materialize identity-partitioned columns as CONSTANTS from
//!    the manifest tuple — and those constants OVERRIDE the column physically present in the file
//!    (`record_batch_transformer.rs`, `constant_overrides_file_column`). Reading a file that way and
//!    then recomputing `identity(col)` returns the RECORDED value by construction: the audit would
//!    be vacuous for exactly the transform the exposure window hits hardest. Both actions therefore
//!    clear `partition`/`partition_spec` on the task ([`prepare_read_task`]) so every source column
//!    comes from the FILE. (Mutation-pinned: restoring them drops the identity-only finding.)
//! 2. **Detection reads ALL physical rows; repair reads LIVE rows.** The spec invariant is about
//!    every row stored in the file, so detection strips the task's delete files. The repair must
//!    conserve exactly the LIVE row set, so it keeps them (merge-on-read deletes applied), exactly
//!    like [`RewriteDataFiles`](crate::maintenance::RewriteDataFiles).
//!
//! # Repair
//!
//! For each flagged file: re-read its LIVE rows (deletes applied), split them by their RECOMPUTED
//! partition value, write one data file per distinct correct key through the standard rolling
//! data-file writer behind a [`FanoutWriter`] (a miskeyed file's rows can legitimately span several
//! true partitions — the reordered-column symptom produces exactly that), and commit ONE
//! [`RewriteFilesAction`](crate::transaction::rewrite_files) that REPLACES the miskeyed file with
//! the new ones. There is no in-place metadata surgery: the swap is an ordinary atomic `Replace`
//! snapshot through the catalog, and a concurrent commit that added a row-level delete for the
//! replaced file makes it fail loudly rather than silently resurrect rows.
//!
//! The added files keep the STARTING snapshot's data sequence number by default (Java
//! `RewriteDataFiles.USE_STARTING_SEQUENCE_NUMBER_DEFAULT = true`), so outstanding equality deletes
//! still apply to the repaired rows — see
//! [`RewriteDataFiles`](crate::maintenance::RewriteDataFiles)' "sequence-number rule". Position
//! deletes that referenced the replaced file dangle harmlessly afterwards (their rows were already
//! excluded from what was rewritten); that is the same posture `RewriteDataFiles` takes.
//!
//! Each file is rewritten under **its own** partition spec (the spec its manifest entry claims), not
//! the table's current default: the repair fixes the tuple and changes nothing else.
//!
//! # Scope limits (read before trusting a finding)
//!
//! - **The partition source columns must be stored in the data file.** Everything this fork writes
//!   (including the DataFusion provider) stores every table column. A file registered by a
//!   Hive-style migration that OMITS its identity partition columns from the data would recompute to
//!   NULL and be reported as miskeyed — a false finding. Such tables are out of scope.
//! - Files under an UNPARTITIONED spec are skipped (counted in
//!   [`AuditPartitionKeysResult::data_files_skipped`]): the manifest decodes a tuple against its
//!   spec's partition type, so an unpartitioned spec's tuple is empty and unfalsifiable.
//! - The audit READS EVERY LIVE ROW of the table. It is a diagnostic sweep, not a hot path.
//!
//! # Recipe
//!
//! Operator-facing walkthrough (including how to reproduce a corrupted table for drills):
//! [`docs/partition-key-audit.md`](https://github.com/apache/iceberg-rust/blob/main/docs/partition-key-audit.md).

use std::collections::HashMap;
use std::sync::Arc;

use futures::TryStreamExt;

use super::rewrite_data_files::parse_target_file_size;
use crate::arrow::{ArrowReaderBuilder, RecordBatchPartitionSplitter};
use crate::scan::{ArrowRecordBatchStream, FileScanTask, FileScanTaskStream};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, PartitionSpecRef, SchemaRef, Struct, TableMetadata,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::partitioning::PartitioningWriter;
use crate::writer::partitioning::fanout_writer::FanoutWriter;
use crate::{Catalog, Error, ErrorKind, Result};

/// ============================================================================================
/// Results
/// ============================================================================================
/// One data file whose recorded partition tuple disagrees with its rows.
///
/// `computed_partitions` lists EVERY distinct tuple the file's rows recompute to, in first-seen
/// order — a miskeyed file may legitimately hold rows of several true partitions (that is what a
/// reordered partition-source column produces), and one of them may even equal `recorded_partition`
/// (a partially-wrong file). `mismatched_rows` counts only the rows that do NOT recompute to
/// `recorded_partition`; a finding is emitted iff that count is non-zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionKeyFinding {
    /// Full path of the miskeyed data file.
    pub data_file_path: String,
    /// The partition spec id the file's manifest entry claims (the spec the tuple is read under).
    pub partition_spec_id: i32,
    /// The tuple recorded in the manifest entry — what every reader prunes and projects with.
    pub recorded_partition: Struct,
    /// Every distinct tuple the file's rows recompute to, in first-seen order.
    pub computed_partitions: Vec<Struct>,
    /// Rows read from the file (all physical rows; delete files are NOT applied for detection).
    pub rows_examined: u64,
    /// Rows whose recomputed tuple differs from `recorded_partition`.
    pub mismatched_rows: u64,
}

/// The outcome of an [`AuditPartitionKeys`] sweep.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuditPartitionKeysResult {
    /// Live data files whose tuple was recomputed and compared.
    pub data_files_examined: usize,
    /// Live data files skipped because their spec is unpartitioned (nothing to falsify).
    pub data_files_skipped: usize,
    /// Rows read across all examined files.
    pub rows_examined: u64,
    /// One entry per miskeyed file, in scan-plan order.
    pub findings: Vec<PartitionKeyFinding>,
}

impl AuditPartitionKeysResult {
    /// `true` when no examined file disagreed with its manifest tuple.
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    /// The paths of the miskeyed files, in finding order.
    pub fn corrupt_file_paths(&self) -> Vec<&str> {
        self.findings
            .iter()
            .map(|finding| finding.data_file_path.as_str())
            .collect()
    }
}

/// The outcome of a [`RepairPartitionKeys`] run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepairPartitionKeysResult {
    /// Miskeyed data files replaced (one `RewriteFiles` commit each).
    pub repaired_data_files_count: usize,
    /// Correctly-keyed data files written in their place.
    pub added_data_files_count: usize,
    /// Live rows rewritten.
    pub repaired_rows_count: u64,
}

/// ============================================================================================
/// `AuditPartitionKeys` — the detector
/// ============================================================================================
/// Re-derive every live data file's partition tuple from its own rows and report the files whose
/// manifest entry disagrees. Read-only: it commits nothing and writes nothing.
///
/// ```no_run
/// # use iceberg::maintenance::AuditPartitionKeys;
/// # async fn example(table: iceberg::table::Table) -> iceberg::Result<()> {
/// let report = AuditPartitionKeys::new(table).execute().await?;
/// for finding in &report.findings {
///     println!(
///         "{}: recorded {:?}, rows say {:?}",
///         finding.data_file_path, finding.recorded_partition, finding.computed_partitions
///     );
/// }
/// # Ok(())
/// # }
/// ```
pub struct AuditPartitionKeys {
    table: Table,
}

impl AuditPartitionKeys {
    /// Audit `table`'s current snapshot.
    pub fn new(table: Table) -> Self {
        Self { table }
    }

    /// Run the sweep. A table with no current snapshot yields an empty (clean) result.
    pub async fn execute(self) -> Result<AuditPartitionKeysResult> {
        let metadata = self.table.metadata();
        let Some(snapshot) = metadata.current_snapshot() else {
            return Ok(AuditPartitionKeysResult::default());
        };
        let schema = snapshot.schema(metadata)?;

        let recorded_files = live_data_files_by_path(&self.table).await?;
        let tasks = plan_whole_file_tasks(&self.table).await?;

        let mut splitters: HashMap<i32, Option<Arc<RecordBatchPartitionSplitter>>> = HashMap::new();
        let mut result = AuditPartitionKeysResult::default();

        for task in tasks {
            let data_file = lookup_data_file(&recorded_files, task.data_file_path())?;
            let spec_id = data_file.partition_spec_id();
            let Some(splitter) = splitter_for(
                &mut splitters,
                metadata,
                &schema,
                spec_id,
                task.data_file_path(),
            )?
            else {
                tracing::debug!(
                    data_file = task.data_file_path(),
                    partition_spec_id = spec_id,
                    "partition-key audit: skipping a file under an unpartitioned spec"
                );
                result.data_files_skipped += 1;
                continue;
            };

            // Detection reads ALL physical rows (delete files stripped): the spec invariant covers
            // every row stored in the file, not just the live ones.
            let (rows_examined, computed) =
                recompute_file_partitions(&self.table, &task, &splitter, false).await?;

            let recorded_partition = data_file.partition().clone();
            let mismatched_rows: u64 = computed
                .iter()
                .filter(|(partition, _)| partition != &recorded_partition)
                .map(|(_, rows)| *rows)
                .sum();

            result.data_files_examined += 1;
            result.rows_examined = result.rows_examined.saturating_add(rows_examined);
            if mismatched_rows > 0 {
                result.findings.push(PartitionKeyFinding {
                    data_file_path: task.data_file_path().to_string(),
                    partition_spec_id: spec_id,
                    recorded_partition,
                    computed_partitions: computed
                        .into_iter()
                        .map(|(partition, _)| partition)
                        .collect(),
                    rows_examined,
                    mismatched_rows,
                });
            }
        }

        Ok(result)
    }
}

/// ============================================================================================
/// `RepairPartitionKeys` — the remediation
/// ============================================================================================
/// Rewrite every miskeyed data file's LIVE rows under their correct partition keys, replacing the
/// miskeyed file through one atomic [`RewriteFilesAction`](crate::transaction::rewrite_files)
/// commit per file.
///
/// **This action rewrites data.** It is a no-op on a clean table (nothing is committed).
pub struct RepairPartitionKeys {
    table: Table,
    use_starting_sequence_number: bool,
}

impl RepairPartitionKeys {
    /// Repair `table`'s current snapshot.
    pub fn new(table: Table) -> Self {
        Self {
            table,
            use_starting_sequence_number: true,
        }
    }

    /// Stamp the rewritten files with the starting snapshot's data sequence number (default
    /// `true`, Java `RewriteDataFiles.USE_STARTING_SEQUENCE_NUMBER_DEFAULT`). With `false` the
    /// rewritten files take a fresh, higher sequence number and outstanding EQUALITY deletes stop
    /// applying to them — the Java-identical hazard, exposed for parity, not recommended.
    pub fn use_starting_sequence_number(mut self, use_starting_sequence_number: bool) -> Self {
        self.use_starting_sequence_number = use_starting_sequence_number;
        self
    }

    /// Audit, then repair. Returns zero counts (and commits nothing) when the audit is clean.
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<RepairPartitionKeysResult> {
        let audit = AuditPartitionKeys::new(self.table.clone())
            .execute()
            .await?;
        let mut result = RepairPartitionKeysResult::default();
        if audit.is_clean() {
            return Ok(result);
        }

        let metadata = self.table.metadata();
        let snapshot = metadata.current_snapshot().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Cannot repair partition keys: the table has no current snapshot",
            )
        })?;
        let starting_snapshot_id = snapshot.snapshot_id();
        let starting_sequence_number = snapshot.sequence_number();
        let schema = snapshot.schema(metadata)?;
        let target_file_size_bytes = parse_target_file_size(metadata.properties())?;

        let recorded_files = live_data_files_by_path(&self.table).await?;
        let tasks: HashMap<String, FileScanTask> = plan_whole_file_tasks(&self.table)
            .await?
            .into_iter()
            .map(|task| (task.data_file_path().to_string(), task))
            .collect();

        // One `RewriteFiles` commit per miskeyed file, sequentially, each validated against the
        // SAME starting snapshot (Java `RewriteDataFilesCommitManager` per-group commit shape).
        // Untouched entries are carried forward unchanged, so the remaining tasks stay valid.
        let mut table = self.table.clone();
        for finding in &audit.findings {
            let path = finding.data_file_path.as_str();
            let data_file = lookup_data_file(&recorded_files, path)?;
            let task = tasks.get(path).ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Cannot repair partition keys: no scan task for a file the audit flagged",
                )
                .with_context("data_file", path.to_string())
            })?;
            let spec = resolve_spec(metadata, data_file.partition_spec_id(), path)?;
            let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
                schema.clone(),
                spec.clone(),
            )?;

            let (rows_written, added_files) = rewrite_under_computed_keys(
                &table,
                task,
                &splitter,
                &schema,
                target_file_size_bytes,
            )
            .await?;

            // A file with no delete files that reads back EMPTY while its manifest entry claims
            // rows is an anomaly, not a repair: replacing it would silently drop those rows.
            if rows_written == 0 && task.deletes.is_empty() && data_file.record_count() > 0 {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot repair partition keys: the data file read back empty although its \
                     manifest entry claims rows and no delete file applies to it",
                )
                .with_context("data_file", path.to_string())
                .with_context("record_count", data_file.record_count().to_string()));
            }

            let transaction = Transaction::new(&table);
            let mut action = transaction
                .rewrite_files(vec![data_file.clone()], added_files.clone())
                .validate_from_snapshot(starting_snapshot_id);
            if self.use_starting_sequence_number {
                action = action.data_sequence_number(starting_sequence_number);
            }
            let transaction = action.apply(transaction)?;
            table = transaction.commit(catalog).await?;

            result.repaired_data_files_count += 1;
            result.added_data_files_count += added_files.len();
            result.repaired_rows_count = result.repaired_rows_count.saturating_add(rows_written);
        }

        Ok(result)
    }
}

/// ============================================================================================
/// Shared internals
/// ============================================================================================
/// Strip the manifest-derived state that would make a recompute VACUOUS or a rewrite CORRUPT.
///
/// - `partition` / `partition_spec`: the reader turns an identity-partitioned field into a CONSTANT
///   taken from the manifest tuple, and that constant OVERRIDES the column stored in the file. Left
///   in place, the recompute would compare the recorded tuple against itself, and the repair would
///   write the WRONG value into the rewritten rows. Clearing them makes every source column come
///   from the file, which is the whole point.
/// - `predicate`: the planning scan attaches a per-file residual; it is a file-SELECTION device, and
///   as a row filter it would silently drop rows (Java `.ignoreResiduals()` on the rewrite scan).
/// - `deletes`: kept only when `apply_deletes` — see the module docs.
fn prepare_read_task(task: &FileScanTask, apply_deletes: bool) -> FileScanTask {
    let mut task = task.clone();
    task.partition = None;
    task.partition_spec = None;
    task.predicate = None;
    if !apply_deletes {
        task.deletes = Vec::new();
    }
    task
}

/// Read one data file and hand back its batches (see [`prepare_read_task`] for what is stripped).
fn read_data_file(
    table: &Table,
    task: &FileScanTask,
    apply_deletes: bool,
) -> Result<ArrowRecordBatchStream> {
    let prepared = prepare_read_task(task, apply_deletes);
    let task_stream = Box::pin(futures::stream::iter(vec![Ok(prepared)])) as FileScanTaskStream;
    ArrowReaderBuilder::new(table.file_io().clone())
        .build()
        .read(task_stream)
}

/// Stream a data file and tally the DISTINCT partition tuples its rows recompute to, in first-seen
/// order. Returns `(rows_read, [(tuple, rows)])`. Memory is O(distinct tuples), not O(rows).
async fn recompute_file_partitions(
    table: &Table,
    task: &FileScanTask,
    splitter: &RecordBatchPartitionSplitter,
    apply_deletes: bool,
) -> Result<(u64, Vec<(Struct, u64)>)> {
    let mut stream = read_data_file(table, task, apply_deletes)?;
    let mut order: Vec<(Struct, u64)> = Vec::new();
    let mut index: HashMap<Struct, usize> = HashMap::new();
    let mut rows_read: u64 = 0;

    while let Some(batch) = stream
        .try_next()
        .await
        .map_err(|error| error.with_context("data_file", task.data_file_path().to_string()))?
    {
        for (partition_key, partition_batch) in splitter.split(&batch)? {
            let rows = row_count(partition_batch.num_rows());
            rows_read = rows_read.saturating_add(rows);
            // `index` and `order` are only ever grown together (below), so a hit in `index` always
            // addresses an existing `order` slot; the `None` arm of `get_mut` is unreachable and is
            // written as a no-op rather than a panic.
            match index.get(partition_key.data()) {
                Some(position) => {
                    if let Some(entry) = order.get_mut(*position) {
                        entry.1 = entry.1.saturating_add(rows);
                    }
                }
                None => {
                    index.insert(partition_key.data().clone(), order.len());
                    order.push((partition_key.data().clone(), rows));
                }
            }
        }
    }

    Ok((rows_read, order))
}

/// Read one miskeyed file's LIVE rows, split them by their RECOMPUTED partition value, and write
/// one data file per distinct correct key. Returns `(rows_written, added_files)`.
///
/// The [`FanoutWriter`] is what makes a miskeyed file whose rows span SEVERAL true partitions
/// repairable in one pass; each partition key carries the file's own spec, so the rewritten entries
/// are stamped with that spec id.
async fn rewrite_under_computed_keys(
    table: &Table,
    task: &FileScanTask,
    splitter: &RecordBatchPartitionSplitter,
    schema: &SchemaRef,
    target_file_size_bytes: u64,
) -> Result<(u64, Vec<DataFile>)> {
    let location_generator = DefaultLocationGenerator::new(table.metadata().clone())?;
    let file_name_generator = DefaultFileNameGenerator::new(
        "repaired".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling_builder = RollingFileWriterBuilder::new(
        parquet_builder,
        usize::try_from(target_file_size_bytes).unwrap_or(usize::MAX),
        table.file_io().clone(),
        location_generator,
        file_name_generator,
    );
    let mut writer = FanoutWriter::new(DataFileWriterBuilder::new(rolling_builder));

    let mut stream = read_data_file(table, task, true)?;
    let mut rows_written: u64 = 0;
    while let Some(batch) = stream
        .try_next()
        .await
        .map_err(|error| error.with_context("data_file", task.data_file_path().to_string()))?
    {
        for (partition_key, partition_batch) in splitter.split(&batch)? {
            rows_written = rows_written.saturating_add(row_count(partition_batch.num_rows()));
            writer.write(partition_key, partition_batch).await?;
        }
    }

    Ok((rows_written, writer.close().await?))
}

/// Build a `path -> DataFile` map over the current snapshot's LIVE data-file manifest entries. The
/// full [`DataFile`] is what the rewrite removal set needs; a scan task only carries the path.
async fn live_data_files_by_path(table: &Table) -> Result<HashMap<String, DataFile>> {
    let mut by_path: HashMap<String, DataFile> = HashMap::new();
    let metadata = table.metadata();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(by_path);
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await?;
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == DataContentType::Data {
                by_path.insert(entry.file_path().to_string(), entry.data_file().clone());
            }
        }
    }
    Ok(by_path)
}

/// Plan one whole-file scan task per live data file of the current snapshot (no row filter, so no
/// file is pruned away). Each task carries the delete files that apply to it.
async fn plan_whole_file_tasks(table: &Table) -> Result<Vec<FileScanTask>> {
    let stream = table.scan().build()?.plan_files().await?;
    stream.try_collect().await
}

fn lookup_data_file<'a>(files: &'a HashMap<String, DataFile>, path: &str) -> Result<&'a DataFile> {
    files.get(path).ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            "Cannot audit partition keys: a planned data file is not a live manifest entry of the \
             current snapshot (a concurrent commit removed it)",
        )
        .with_context("data_file", path.to_string())
    })
}

fn resolve_spec(metadata: &TableMetadata, spec_id: i32, path: &str) -> Result<PartitionSpecRef> {
    metadata
        .partition_spec_by_id(spec_id)
        .cloned()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Cannot audit partition keys: the data file claims a partition spec the table does \
                 not have",
            )
            .with_context("data_file", path.to_string())
            .with_context("partition_spec_id", spec_id.to_string())
        })
}

/// The cached per-spec splitter, or `None` when that spec is unpartitioned (nothing to falsify).
fn splitter_for(
    cache: &mut HashMap<i32, Option<Arc<RecordBatchPartitionSplitter>>>,
    metadata: &TableMetadata,
    schema: &SchemaRef,
    spec_id: i32,
    path: &str,
) -> Result<Option<Arc<RecordBatchPartitionSplitter>>> {
    if let Some(cached) = cache.get(&spec_id) {
        return Ok(cached.clone());
    }
    let spec = resolve_spec(metadata, spec_id, path)?;
    let splitter = if spec.is_unpartitioned() {
        None
    } else {
        Some(Arc::new(
            RecordBatchPartitionSplitter::try_new_with_computed_values(schema.clone(), spec)?,
        ))
    };
    cache.insert(spec_id, splitter.clone());
    Ok(splitter)
}

/// Row counts are `usize` on the Arrow side and `u64` in every Iceberg count; saturate rather than
/// truncate (no `as`), which on every supported target is exact.
fn row_count(rows: usize) -> u64 {
    u64::try_from(rows).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[path = "partition_key_audit_tests.rs"]
mod tests;
