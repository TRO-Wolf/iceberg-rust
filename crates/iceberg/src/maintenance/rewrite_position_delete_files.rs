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

//! `RewritePositionDeleteFiles` — the engine-agnostic maintenance action that COMPACTS the live PARQUET
//! position-delete files in the current snapshot, per `(spec, partition)` group, into FEWER position-delete
//! files and commits the swap in one `Replace` snapshot per group. The Rust port of Java's
//! `org.apache.iceberg.actions.RewritePositionDeleteFiles` (api 1.10.0).
//!
//! # The Java contract this mirrors
//!
//! `RewritePositionDeleteFiles extends SnapshotUpdate<RewritePositionDeleteFiles, Result>` with one own
//! method `filter(Expression)` plus the inherited `execute() -> Result` (javap-verified against
//! `iceberg-api` 1.10.0). `RewritePositionDeleteFiles$Result` exposes four counts —
//! `rewrittenDeleteFilesCount()` / `addedDeleteFilesCount()` (`int`) and `rewrittenBytesCount()` /
//! `addedBytesCount()` (`long`) — mirrored 1:1 by [`RewritePositionDeleteFilesResult`]. The PLANNING +
//! COMMIT machinery is engine-agnostic iceberg-core (`BinPackRewritePositionDeletePlanner` groups by
//! partition; `RewritePositionDeletesCommitManager` runs `newRewrite().validateFromSnapshot(J)
//! .deleteFile(rewritten).addFile(added, J).commit()` where `J` = the group MAX rewritten data-seq); the
//! read/sort/write MATERIALIZATION is a Spark-surface action (no Spark bytecode is available locally), so
//! the pipeline below is built engine-agnostically, exactly as [`ConvertEqualityDeleteFiles`] was. The
//! Java contract pins the INTERFACE shape, the `Result` counts, and the commit recipe (seq stamp +
//! validate-from-snapshot).
//!
//! Unlike [`ConvertEqualityDeleteFiles`] (a free-standing action), `RewritePositionDeleteFiles` IS one of
//! the twelve Java `ActionsProvider` methods (`rewritePositionDeletes(Table)`), so it is wired into the
//! [`ActionsProvider`](crate::maintenance::ActionsProvider) factory.
//!
//! # The compaction (many parquet pos-deletes → fewer)
//!
//! A position-delete file deletes rows by `(file_path, pos)`. Compaction reads the `(file_path, pos)`
//! pairs out of EVERY live parquet position-delete file in a `(spec, partition)` group, concatenates and
//! sorts them, and writes them into FEWER position-delete files that mask EXACTLY the same rows. A
//! merge-on-read scan therefore returns an identical live row set before and after.
//!
//! This action is a STRICT SUBSET of `ConvertEqualityDeleteFiles`: there is no data-file row matching, no
//! survival-predicate inversion, and no equality-tuple parsing — the positions are read directly off the
//! delete files. For each `(spec_id, partition)` group of live parquet position-delete files (optionally
//! restricted by [`RewritePositionDeleteFiles::filter`], applied to each delete file's partition):
//!
//! 1. Read each live parquet position-delete file's two RESERVED columns — `file_path` (reserved field id
//!    [`RESERVED_FIELD_ID_DELETE_FILE_PATH`] = `2147483546`) and `pos`
//!    ([`RESERVED_FIELD_ID_DELETE_FILE_POS`] = `2147483545`) — by FIELD ID, not by name, into
//!    `Vec<(String, i64)>` (the read path lives in
//!    [`crate::arrow::delete_file_loader`]); Puffin deletion vectors (`format == Puffin`) are SKIPPED.
//! 2. Concatenate every group member's pairs and sort by `(file_path, pos)` (the spec-recommended
//!    position-delete ordering). Java does NOT dedup within a group — the reader bitmap dedups — so
//!    duplicates are harmless; we sort and keep them.
//! 3. Write the sorted pairs into FEWER position-delete files (one per group) under the group's spec +
//!    partition key, via the [`PositionDeleteFileWriter`](crate::writer::base_writer::position_delete_writer).
//! 4. Compute the group MAX rewritten data sequence number (staller — see below).
//! 5. Commit ONE [`RewriteFilesAction`](crate::transaction::rewrite_files) per group that REPLACES the
//!    rewritten position-delete files with the new ones, STAMPING the new file with the group MAX
//!    rewritten data-seq (via `add_delete_file_with_sequence_number`, NOT the default-inherit
//!    `add_delete_file`) and validating from the starting snapshot.
//!
//! # The silent-corruption staller (handled EXPLICITLY): SEQ STAMPING
//!
//! The added compacted file MUST be stamped with the group MAX rewritten data sequence number — NOT the
//! inherited (higher) seq, NOT the min. A position delete applies to data with `data_seq < delete_seq`;
//! stamping the MAX of the rewritten group's data-seqs preserves exactly which data generation the
//! compacted delete masks. A wrong (higher / inherited) seq makes the compacted pos-delete stop applying
//! to its older data and RESURRECTS deleted rows; a wrong (lower) seq over-applies. Java's
//! `RewritePositionDeletesCommitManager` adds the rewritten file with `Long.valueOf(maxRewrittenSeq)`
//! exactly for this reason.
//!
//! # Divergence: V2 PARQUET only (V3 deletion vectors are OUT of scope)
//!
//! This action compacts V2 PARQUET position-delete files only. V3 Puffin DELETION VECTORS are
//! file-scoped (one DV per data file, never bin-packed across files) and are SKIPPED here — a DV is never
//! "compacted" by this action. (Java's V3 DV maintenance is a separate concern.) This divergence is
//! documented on `docs/parity/GAP_MATRIX.md` row 134.
//!
//! # No-op
//!
//! With no current snapshot, no live parquet position-delete files, a [`filter`](RewritePositionDeleteFiles::filter)
//! that matches none, or a group of only ONE position-delete file (nothing to compact — Java's planner
//! drops single-file groups), the action commits NOTHING for that group and the result counts stay zero
//! (Java commits only when there is real compaction work).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::StreamExt;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::expr::visitors::expression_evaluator::ExpressionEvaluator;
use crate::expr::visitors::inclusive_projection::InclusiveProjection;
use crate::expr::{Bind, Predicate};
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, MetricsConfig, PartitionKey, Schema, Snapshot,
    Struct,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, Error, ErrorKind, Result};

/// The `(spec_id, partition)` group a position-delete file belongs to (Java's
/// `BinPackRewritePositionDeletePlanner` groups by partition + spec).
type GroupKey = (i32, Struct);

/// The outcome of a [`RewritePositionDeleteFiles::execute`] run, mirroring Java
/// `RewritePositionDeleteFiles$Result`'s four counts.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RewritePositionDeleteFilesResult {
    /// Number of position-delete files rewritten away (Java `rewrittenDeleteFilesCount()`).
    pub rewritten_delete_files_count: usize,
    /// Number of position-delete files added (Java `addedDeleteFilesCount()`).
    pub added_delete_files_count: usize,
    /// Total size in bytes of the rewritten position-delete files (Java `rewrittenBytesCount()`).
    pub rewritten_bytes_count: u64,
    /// Total size in bytes of the added position-delete files (Java `addedBytesCount()`).
    pub added_bytes_count: u64,
}

impl RewritePositionDeleteFilesResult {
    /// Number of position-delete files rewritten away (Java `rewrittenDeleteFilesCount()`).
    pub fn rewritten_delete_files_count(&self) -> usize {
        self.rewritten_delete_files_count
    }

    /// Number of position-delete files added (Java `addedDeleteFilesCount()`).
    pub fn added_delete_files_count(&self) -> usize {
        self.added_delete_files_count
    }

    /// Total size in bytes of the rewritten position-delete files (Java `rewrittenBytesCount()`).
    pub fn rewritten_bytes_count(&self) -> u64 {
        self.rewritten_bytes_count
    }

    /// Total size in bytes of the added position-delete files (Java `addedBytesCount()`).
    pub fn added_bytes_count(&self) -> u64 {
        self.added_bytes_count
    }
}

/// The `RewritePositionDeleteFiles` maintenance action. Build it with [`Self::new`], optionally restrict
/// the compacted partitions with [`Self::filter`], and run it with [`Self::execute`].
pub struct RewritePositionDeleteFiles {
    table: Table,
    filter: Predicate,
}

impl RewritePositionDeleteFiles {
    /// Create a `RewritePositionDeleteFiles` action for `table`. With no [`filter`](Self::filter), every
    /// `(spec, partition)` group of live parquet position-delete files in the current snapshot is
    /// compacted.
    pub fn new(table: Table) -> Self {
        Self {
            table,
            filter: Predicate::AlwaysTrue,
        }
    }

    /// Restrict the compaction to position-delete files whose partition matches `filter` (Java
    /// `RewritePositionDeleteFiles.filter(Expression)`). The predicate is bound to the table schema,
    /// inclusively projected onto each delete file's partition spec, and evaluated against the delete
    /// file's PARTITION values — the SAME partition-pruning path the table scan uses. The default is
    /// [`Predicate::AlwaysTrue`] (compact all).
    pub fn filter(mut self, filter: Predicate) -> Self {
        self.filter = filter;
        self
    }

    /// Run the compaction: for every `(spec, partition)` group of live (filter-matching) parquet
    /// position-delete files in the current snapshot, read their `(file_path, pos)` pairs, concat + sort
    /// them, write FEWER position-delete files, and commit the swap in one `Replace` snapshot per group.
    /// Returns the [`RewritePositionDeleteFilesResult`] counts.
    ///
    /// Commits NOTHING and returns zero counts when there is no current snapshot, no live parquet
    /// position-delete files, none match the filter, or no group has more than one file (nothing to
    /// compact).
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<RewritePositionDeleteFilesResult> {
        let metadata = self.table.metadata();
        let Some(snapshot) = metadata.current_snapshot().cloned() else {
            return Ok(RewritePositionDeleteFilesResult::default());
        };
        let starting_snapshot_id = snapshot.snapshot_id();

        // (1) Enumerate the live PARQUET position-delete entries, grouped by (spec_id, partition).
        // Puffin DVs are SKIPPED (file-scoped, never bin-packed) — the documented V2-parquet-only scope.
        let groups = self.collect_position_delete_groups(&snapshot).await?;

        let mut result = RewritePositionDeleteFilesResult::default();
        for (key, entries) in groups {
            // Java's planner drops single-file groups (nothing to compact). A group must have at least
            // TWO position-delete files for compaction to do real work.
            if entries.len() < 2 {
                continue;
            }
            // Filter on the group's partition (every entry in a group shares the partition + spec, so the
            // first entry represents the group).
            if !self.group_matches_filter(&entries[0])? {
                continue;
            }

            self.compact_group(catalog, &key, &entries, starting_snapshot_id, &mut result)
                .await?;
        }

        Ok(result)
    }

    /// Walk the current snapshot's manifests once and collect the live PARQUET position-delete entries
    /// grouped by `(spec_id, partition)`. Puffin deletion vectors (`format == Puffin`) and
    /// equality/data entries are EXCLUDED. One pass over the manifest list.
    async fn collect_position_delete_groups(
        &self,
        snapshot: &Snapshot,
    ) -> Result<HashMap<GroupKey, Vec<LiveDeleteEntry>>> {
        let metadata = self.table.metadata();
        let manifest_list = snapshot
            .load_manifest_list(self.table.file_io(), metadata)
            .await?;

        let mut groups: HashMap<GroupKey, Vec<LiveDeleteEntry>> = HashMap::new();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let data_file = entry.data_file();
                // Only PARQUET position deletes. Skip data, equality deletes, and Puffin DVs.
                if data_file.content_type() != DataContentType::PositionDeletes {
                    continue;
                }
                if data_file.file_format() != DataFileFormat::Parquet {
                    // A Puffin DELETION VECTOR — file-scoped, never bin-packed by this action.
                    continue;
                }
                let key = (data_file.partition_spec_id, data_file.partition().clone());
                groups.entry(key).or_default().push(LiveDeleteEntry {
                    data_file: data_file.clone(),
                    // A live pos-delete always carries a concrete post-inheritance seq; the unwrap-or
                    // default never fires for a real on-disk entry.
                    sequence_number: entry.sequence_number().unwrap_or(0),
                });
            }
        }

        Ok(groups)
    }

    /// Whether the position-delete group's partition matches [`Self::filter`]. `AlwaysTrue` (the default)
    /// matches everything without binding. Otherwise the row-level filter is bound to the table schema,
    /// inclusively projected onto the delete file's partition spec, and evaluated against the delete
    /// file's partition struct — the SAME partition-pruning path the table scan uses.
    fn group_matches_filter(&self, entry: &LiveDeleteEntry) -> Result<bool> {
        if matches!(self.filter, Predicate::AlwaysTrue) {
            return Ok(true);
        }
        let metadata = self.table.metadata();
        let schema = metadata.current_schema().clone();
        let bound_row_filter = self
            .filter
            .clone()
            .bind(schema.clone(), true)
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "RewritePositionDeleteFiles filter could not be bound to the table schema",
                )
                .with_source(e)
            })?;

        let spec = metadata
            .partition_spec_by_id(entry.data_file.partition_spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Position delete '{}' references unknown partition spec {}",
                        entry.data_file.file_path(),
                        entry.data_file.partition_spec_id
                    ),
                )
            })?;

        let partition_type = spec.partition_type(&schema)?;
        let partition_schema = Arc::new(
            Schema::builder()
                .with_schema_id(spec.spec_id())
                .with_fields(partition_type.fields().to_owned())
                .build()?,
        );
        let mut inclusive_projection = InclusiveProjection::new(spec.clone());
        let partition_filter = inclusive_projection
            .project(&bound_row_filter)?
            .rewrite_not()
            .bind(partition_schema, true)?;

        ExpressionEvaluator::new(partition_filter).eval(&entry.data_file)
    }

    /// Compact ONE `(spec, partition)` group: read every member file's `(file_path, pos)` pairs, concat +
    /// sort, write FEWER position-delete files, and commit ONE `RewriteFiles` that replaces the rewritten
    /// files with the new one, stamped with the group MAX rewritten data-seq and validated from the
    /// starting snapshot. Accumulates the four `Result` counts.
    async fn compact_group(
        &self,
        catalog: &dyn Catalog,
        key: &GroupKey,
        entries: &[LiveDeleteEntry],
        starting_snapshot_id: i64,
        result: &mut RewritePositionDeleteFilesResult,
    ) -> Result<()> {
        // (2) Read + concat the (file_path, pos) pairs across the group.
        let mut pairs: Vec<(String, i64)> = Vec::new();
        for entry in entries {
            self.read_position_pairs(&entry.data_file, &mut pairs)
                .await?;
        }

        // A group of live pos-delete files always carries rows (a position delete with no rows is
        // degenerate); if somehow empty, there is nothing to compact — leave the group untouched.
        if pairs.is_empty() {
            return Ok(());
        }

        // Spec-recommended position-delete ordering: sort by (file_path, pos). Java does NOT dedup within
        // a group (the reader bitmap dedups); we keep duplicates and only sort.
        pairs.sort();

        // (3) Write FEWER position-delete files (one per group) under the group spec + partition key.
        let new_file = self.write_compacted_file(key, &pairs).await?;

        // (4) STALLER — the group MAX rewritten data sequence number. A position delete applies to data
        // with `data_seq < delete_seq`; stamping the MAX of the rewritten group preserves exactly which
        // data generation the compacted delete masks. A higher (inherited) seq resurrects deleted rows;
        // a lower seq over-applies.
        let max_seq = entries
            .iter()
            .map(|e| e.sequence_number)
            .max()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "compact_group called with an empty group (no sequence numbers)",
                )
            })?;

        // Accumulate the byte counts (Java `rewrittenBytesCount` / `addedBytesCount`).
        let rewritten_bytes: u64 = entries.iter().map(|e| e.data_file.file_size_in_bytes).sum();
        let rewritten_count = entries.len();
        let added_bytes = new_file.file_size_in_bytes;
        let rewritten_files: Vec<DataFile> = entries.iter().map(|e| e.data_file.clone()).collect();

        // (5) Commit ONE RewriteFiles per group: REPLACE the rewritten pos-deletes with the new one,
        // stamped with the group MAX rewritten data-seq via `add_delete_file_with_sequence_number` (NOT
        // the default-inherit add), validating from the starting snapshot (Java
        // `newRewrite().validateFromSnapshot(J).deleteFile(rewritten).addFile(added, J).commit()`).
        let transaction = Transaction::new(&self.table);
        let action = transaction
            .rewrite_files(Vec::new(), Vec::new())
            .delete_delete_files(rewritten_files)
            .add_delete_file_with_sequence_number(new_file, max_seq)
            .validate_from_snapshot(starting_snapshot_id);
        let transaction = action.apply(transaction)?;
        transaction.commit(catalog).await?;

        result.rewritten_delete_files_count += rewritten_count;
        result.added_delete_files_count += 1;
        result.rewritten_bytes_count = result
            .rewritten_bytes_count
            .checked_add(rewritten_bytes)
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "rewritten bytes count overflow"))?;
        result.added_bytes_count = result
            .added_bytes_count
            .checked_add(added_bytes)
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "added bytes count overflow"))?;

        Ok(())
    }

    /// Read one parquet position-delete file's two RESERVED columns — `file_path` (field id
    /// [`RESERVED_FIELD_ID_DELETE_FILE_PATH`]) and `pos` (field id [`RESERVED_FIELD_ID_DELETE_FILE_POS`])
    /// — by FIELD ID, not by name, appending every `(file_path, pos)` pair into `pairs`. The columns are
    /// located by their `PARQUET_FIELD_ID_META_KEY` metadata so a renamed-but-correctly-id'd file still
    /// reads (interop-faithful).
    async fn read_position_pairs(
        &self,
        delete_file: &DataFile,
        pairs: &mut Vec<(String, i64)>,
    ) -> Result<()> {
        let loader = BasicDeleteFileLoader::new(self.table.file_io().clone());
        let mut stream = loader
            .parquet_to_batch_stream(delete_file.file_path(), delete_file.file_size_in_bytes)
            .await?;

        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let (path_col, pos_col) = locate_reserved_columns(&batch, delete_file.file_path())?;
            for row in 0..batch.num_rows() {
                if path_col.is_null(row) || pos_col.is_null(row) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Position delete '{}' has a null file_path/pos at row {row}",
                            delete_file.file_path()
                        ),
                    ));
                }
                pairs.push((path_col.value(row).to_string(), pos_col.value(row)));
            }
        }

        Ok(())
    }

    /// Write the sorted `(file_path, pos)` pairs into ONE compacted position-delete file under the
    /// group's spec + partition key, returning the resulting [`DataFile`].
    async fn write_compacted_file(
        &self,
        key: &GroupKey,
        pairs: &[(String, i64)],
    ) -> Result<DataFile> {
        let metadata = self.table.metadata();
        let schema = metadata.current_schema().clone();
        let (spec_id, partition) = key;
        let spec = metadata
            .partition_spec_by_id(*spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Position delete group references unknown partition spec {spec_id}"),
                )
            })?
            .as_ref()
            .clone();

        let config = PositionDeleteWriterConfig::new()?;
        let location_gen = DefaultLocationGenerator::new(metadata.clone())?;
        let file_name_gen = DefaultFileNameGenerator::new(
            "compacted-pos-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        // Position-delete files keep `file_path`/`pos` bounds FULL (Java `MetricsConfig.forPositionDelete`)
        // so delete-file path pruning stays precise — the default `truncate(16)` would widen the path range.
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            config.schema().clone(),
        )
        .with_metrics_config(MetricsConfig::for_position_delete());
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            self.table.file_io().clone(),
            location_gen,
            file_name_gen,
        );

        // The new pos-delete must live in the SAME partition + spec as the files it replaces (so it lands
        // in the same bucket and applies to the same data files). An unpartitioned spec takes no key.
        let partition_key = if spec.is_unpartitioned() {
            None
        } else {
            Some(PartitionKey::new(spec, schema.clone(), partition.clone()))
        };
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
            .build(partition_key)
            .await?;

        let paths: Vec<&str> = pairs.iter().map(|(path, _)| path.as_str()).collect();
        let positions: Vec<i64> = pairs.iter().map(|(_, pos)| *pos).collect();
        let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions)) as ArrayRef,
        ])
        .map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "Failed to build compacted position-delete record batch",
            )
            .with_source(e)
        })?;
        writer.write(batch).await?;
        let files = writer.close().await?;
        files.into_iter().next().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Position-delete writer produced no file for a non-empty input",
            )
        })
    }
}

/// Locate the `file_path` (string) and `pos` (int64) columns of a position-delete record batch by their
/// RESERVED FIELD IDs (`PARQUET_FIELD_ID_META_KEY` metadata), NOT by name or column order. A delete file
/// written with the reserved ids but a renamed column still reads. Errors if either reserved column is
/// absent or has the wrong arrow type.
fn locate_reserved_columns<'a>(
    batch: &'a RecordBatch,
    file_path: &str,
) -> Result<(&'a StringArray, &'a Int64Array)> {
    let mut path_idx: Option<usize> = None;
    let mut pos_idx: Option<usize> = None;
    for (idx, field) in batch.schema().fields().iter().enumerate() {
        if let Some(id_str) = field.metadata().get(PARQUET_FIELD_ID_META_KEY)
            && let Ok(id) = id_str.parse::<i32>()
        {
            if id == RESERVED_FIELD_ID_DELETE_FILE_PATH {
                path_idx = Some(idx);
            } else if id == RESERVED_FIELD_ID_DELETE_FILE_POS {
                pos_idx = Some(idx);
            }
        }
    }

    let path_idx = path_idx.ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Position delete '{file_path}' is missing the reserved file_path column \
                 (field id {RESERVED_FIELD_ID_DELETE_FILE_PATH})"
            ),
        )
    })?;
    let pos_idx = pos_idx.ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Position delete '{file_path}' is missing the reserved pos column \
                 (field id {RESERVED_FIELD_ID_DELETE_FILE_POS})"
            ),
        )
    })?;

    let path_col = batch
        .column(path_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Position delete '{file_path}' file_path column is not a string array"),
            )
        })?;
    let pos_col = batch
        .column(pos_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Position delete '{file_path}' pos column is not an int64 array"),
            )
        })?;

    Ok((path_col, pos_col))
}

/// A live PARQUET position-delete entry: its [`DataFile`] and its post-inheritance data sequence number.
struct LiveDeleteEntry {
    data_file: DataFile,
    sequence_number: i64,
}

#[cfg(test)]
#[path = "rewrite_position_delete_files_tests.rs"]
mod tests;
