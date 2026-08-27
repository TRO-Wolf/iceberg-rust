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

//! Compacts live PARQUET position-delete files of the current snapshot.
//! Java `RewritePositionDeleteFiles`. One `Replace` snapshot per admitted bin.
//!
//! Each added file carries that bin's max rewritten data sequence number.
//! An over-high stamp deletes rows the bin never masked. An under-low stamp
//! resurrects rows the bin must still delete.
//!
//! | Named non-port | Why it stays out |
//! |---|---|
//! | Java `inputSplitSize` / `expectedOutputFiles` | Read-side scan options. This action reads pairs directly. A port would be dead code. Grep for the snake_case spellings must return zero hits. |
//!
//! V1/V2 compact parquet position deletes. V3 converts them to one Puffin DV
//! per referenced data file. A V3 table cannot commit a fresh parquet position
//! delete. See [`RewritePositionDeleteFiles::rewrite_to_deletion_vectors`].
//!
//! On V3, `Ok` with zeros means the arm looked and found nothing to convert.
//! An input it cannot express is `Err` when the filter admits it.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::StreamExt;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::rewrite_data_files::{
    MAX_FILE_GROUP_SIZE_BYTES_DEFAULT, MAX_FILE_SIZE_DEFAULT_RATIO, MIN_FILE_SIZE_DEFAULT_RATIO,
    MIN_INPUT_FILES_DEFAULT, pack_bins,
};
use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::delete_file_index::referenced_data_file_location;
use crate::delete_vector::load_delete_vector;
use crate::expr::visitors::expression_evaluator::ExpressionEvaluator;
use crate::expr::visitors::inclusive_projection::InclusiveProjection;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, MetricsConfig, PartitionKey, Schema,
    Snapshot, Struct, TableMetadata, TableProperties,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig, position_delete_writer_properties,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator, FileNameGenerator, LocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, Error, ErrorKind, Result};

/// The `(spec_id, partition)` group a position-delete file belongs to (Java's
/// `BinPackRewritePositionDeletePlanner` groups by partition + spec).
type GroupKey = (i32, Struct);

/// One admitted bin: group key plus the entries the seq stamp is defined over.
type AdmittedBin = (GroupKey, Vec<LiveDeleteEntry>);

// Shared with `rewrite_data_files`. Both Java planners inherit the same four constants.

/// Default target in the `i64` parse domain. The const assert proves the `as` cannot truncate.
const DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT_I64: i64 = {
    let default = TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT;
    assert!(default <= i64::MAX as u64);
    default as i64
};

/// Java `writeMaxFileSize()` band ratio. Not fork-authored.
const WRITE_MAX_FILE_SIZE_RATIO: f64 = 0.5;

/// Max pairs per rolling-writer `write`. Java has no analogue: it never chunks a writer feed.
/// `should_roll` runs once per `write`, so a whole-bin batch can never roll.
const CHUNK_PAIRS: usize = 256;

/// Max measured serialized size of one chunk. A rolled file overshoots by at most one chunk.
/// That overshoot must fit in `max_file_size_bytes - write_max_file_size`.
const CHUNK_MAX_SERIALIZED_BYTES: u64 = 16384;

/// Reserve half the candidate-filter headroom for the Parquet footer. `should_roll` excludes it.
const CHUNK_HEADROOM_FOOTER_SHARE: u64 = 2;

/// Java `d2l`. The JVM saturates at `Long.MAX_VALUE`; Rust `as u64` saturates at `u64::MAX`.
/// The `.min(i64::MAX as u64)` is the parity act. Residue: `rewrite_data_files` stays unclamped.
fn d2l(x: f64) -> u64 {
    (x as u64).min(i64::MAX as u64)
}

/// Parse `write.delete.target-file-size-bytes` as `i64`, matching `Long.parseLong`.
/// A `u64` parse would reject `"-1"` with a fork-only message and admit values above `i64::MAX`.
fn parse_delete_target_file_size(properties: &HashMap<String, String>) -> Result<i64> {
    match properties.get(TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES) {
        None => Ok(DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT_I64),
        Some(value) => value.parse::<i64>().map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid value '{value}' for table property '{}'",
                    TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES
                ),
            )
            .with_source(error)
        }),
    }
}

/// Refuse a builder override Java cannot express as `long`. Admitting it makes `too_much_content` unreachable.
fn reject_size_override_above_i64_max(option: &str, value: u64) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Invalid value '{value}' for '{option}': it must be <= {} — Java's option is a `long`, \
             so a larger threshold has no Java analogue",
            i64::MAX
        ),
    )
}

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

/// Thresholds for one `execute` run. [`RewritePositionDeleteFiles::resolve_config`] is the only home.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedConfig {
    target_file_size_bytes: u64,
    min_file_size_bytes: u64,
    max_file_size_bytes: u64,
    min_input_files: usize,
    max_file_group_size_bytes: u64,
    /// Java `writeMaxFileSize()` — the rolling-writer bound, not the resolved target.
    write_max_file_size: u64,
    /// Per-chunk measured-byte cap, derived from candidate-filter headroom.
    chunk_budget: u64,
}

/// Compacts live parquet position deletes. Java `RewritePositionDeleteFiles`.
///
/// | Deferred option | Why it is not emulable here |
/// |---|---|
/// | `rewrite-all` | `min_file_size_bytes(0)` plus a huge max empties the candidate set. That is the inverse of Java's bypass. |
/// | `rewrite-job-order` | Bins commit in plan order. |
/// | `partial-progress.*` / `max-concurrent-file-group-rewrites` | Sequential, one commit per bin. Failure is not atomic. |
/// | `output-spec-id` | Each bin writes under its group spec. Java never consults this option. |
/// | Per-group `Result` list | [`RewritePositionDeleteFilesResult`] carries four aggregates only. |
pub struct RewritePositionDeleteFiles {
    table: Table,
    filter: Predicate,
    /// `Some(t)` once the caller pins a target size; `None` ⇒ resolve from the table property at
    /// execute (Java `BinPackRewritePositionDeletePlanner.defaultTargetFileSize`).
    target_file_size_bytes: Option<u64>,
    /// `Some(min)` once the caller pins it; `None` ⇒ `0.75 * target` at execute.
    min_file_size_bytes: Option<u64>,
    /// `Some(max)` once the caller pins it; `None` ⇒ `1.8 * target` at execute.
    max_file_size_bytes: Option<u64>,
    min_input_files: usize,
    max_file_group_size_bytes: u64,
}

impl RewritePositionDeleteFiles {
    /// Create the action with Java's defaults. Size thresholds resolve at [`Self::execute`].
    pub fn new(table: Table) -> Self {
        Self {
            table,
            filter: Predicate::AlwaysTrue,
            target_file_size_bytes: None,
            min_file_size_bytes: None,
            max_file_size_bytes: None,
            min_input_files: MIN_INPUT_FILES_DEFAULT,
            max_file_group_size_bytes: MAX_FILE_GROUP_SIZE_BYTES_DEFAULT,
        }
    }

    /// Target output size in bytes (Java `TARGET_FILE_SIZE_BYTES`). Values above `i64::MAX` fail at execute.
    pub fn target_file_size_bytes(mut self, target_file_size_bytes: u64) -> Self {
        self.target_file_size_bytes = Some(target_file_size_bytes);
        self
    }

    /// Files smaller than this are always candidates (Java `MIN_FILE_SIZE_BYTES`). Default 75% of target.
    pub fn min_file_size_bytes(mut self, min_file_size_bytes: u64) -> Self {
        self.min_file_size_bytes = Some(min_file_size_bytes);
        self
    }

    /// Files larger than this are always candidates (Java `MAX_FILE_SIZE_BYTES`). Default 180% of target.
    pub fn max_file_size_bytes(mut self, max_file_size_bytes: u64) -> Self {
        self.max_file_size_bytes = Some(max_file_size_bytes);
        self
    }

    /// Compact a group with at least this many files regardless of size (Java `MIN_INPUT_FILES`, default 5).
    pub fn min_input_files(mut self, min_input_files: usize) -> Self {
        self.min_input_files = min_input_files;
        self
    }

    /// Largest total input size compacted in one group (Java `MAX_FILE_GROUP_SIZE_BYTES`, default 100 GiB).
    pub fn max_file_group_size_bytes(mut self, max_file_group_size_bytes: u64) -> Self {
        self.max_file_group_size_bytes = max_file_group_size_bytes;
        self
    }

    /// Compact only partitions matching `filter`. Java `RewritePositionDeleteFiles.filter`.
    pub fn filter(mut self, filter: Predicate) -> Self {
        self.filter = filter;
        self
    }

    /// Compact admitted bins, one `Replace` snapshot each. Sequential, so a mid-loop failure
    /// leaves earlier bins committed. Re-run to continue. Java's non-partial path is one commit.
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<RewritePositionDeleteFilesResult> {
        // Validate thresholds before any IO, as Java's `sizeThresholds` does at planner `init`.
        let config = self.resolve_config()?;

        let metadata = self.table.metadata();
        let Some(snapshot) = metadata.current_snapshot().cloned() else {
            return Ok(RewritePositionDeleteFilesResult::default());
        };
        let starting_snapshot_id = snapshot.snapshot_id();

        // Bind after the no-snapshot return, before the walk. An unbindable filter must fail loud.
        let mut partition_filter = self.bind_filter()?;

        // V3 cannot commit a fresh parquet position delete, so compacting into another is unreachable.
        if metadata.format_version() >= FormatVersion::V3 {
            return self
                .rewrite_to_deletion_vectors(
                    catalog,
                    &snapshot,
                    &mut partition_filter,
                    starting_snapshot_id,
                )
                .await;
        }

        let groups = self
            .collect_position_delete_groups(&snapshot, &mut partition_filter)
            .await?;

        let bins = plan_bins(groups, &config);

        // Advance the base after each commit so later bins skip a full stale-base re-apply.
        let mut table = self.table.clone();
        let mut result = RewritePositionDeleteFilesResult::default();
        for bin in bins {
            table = self
                .compact_group(
                    catalog,
                    &table,
                    &bin,
                    &config,
                    starting_snapshot_id,
                    &mut result,
                )
                .await?;
        }

        Ok(result)
    }

    /// Resolve thresholds and enforce Java's `sizeThresholds` order.
    /// Check (7) at read time so each knob reports its own rejection and the `as i64` cannot wrap.
    /// Hoisting (3) above (1) reports the wrong message at a negative target.
    fn resolve_config(&self) -> Result<ResolvedConfig> {
        // (7) at read time, one leg per knob, before defaults and before (1).
        if let Some(target) = self.target_file_size_bytes
            && target > i64::MAX as u64
        {
            return Err(reject_size_override_above_i64_max(
                "target-file-size-bytes",
                target,
            ));
        }
        if let Some(min) = self.min_file_size_bytes
            && min > i64::MAX as u64
        {
            return Err(reject_size_override_above_i64_max(
                "min-file-size-bytes",
                min,
            ));
        }
        if let Some(max) = self.max_file_size_bytes
            && max > i64::MAX as u64
        {
            return Err(reject_size_override_above_i64_max(
                "max-file-size-bytes",
                max,
            ));
        }

        // Stay in i64 until (1) so a negative property reaches Java's `checkArgument`.
        let target = match self.target_file_size_bytes {
            Some(target) => target as i64,
            None => parse_delete_target_file_size(self.table.metadata().properties())?,
        };

        // Java: `defaultMin = d2l(target * 0.75)`, `defaultMax = d2l(target * 1.8)`, resolved BEFORE
        // the checks. `target as f64` is Java's `l2d` — the same f64 rounding on a large `long`.
        let default_min = d2l(target as f64 * MIN_FILE_SIZE_DEFAULT_RATIO);
        let default_max = d2l(target as f64 * MAX_FILE_SIZE_DEFAULT_RATIO);
        let min_file_size_bytes = self.min_file_size_bytes.unwrap_or(default_min);
        let max_file_size_bytes = self.max_file_size_bytes.unwrap_or(default_max);

        // (1) target > 0.
        if target <= 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("'target-file-size-bytes' is set to {target} but must be > 0"),
            ));
        }
        // (2) min >= 0 — structurally unreachable (see the doc comment); not coded.

        // Proven positive by (1), so the narrowing is total; carried as a checked conversion rather
        // than an `as` so a future edit to (1) cannot silently wrap.
        let target: u64 = u64::try_from(target).map_err(|error| {
            Error::new(
                ErrorKind::Unexpected,
                "'target-file-size-bytes' passed the > 0 precondition but is not a u64",
            )
            .with_source(error)
        })?;

        // (3) target > min.
        if target <= min_file_size_bytes {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "'target-file-size-bytes' ({target}) must be > 'min-file-size-bytes' \
                     ({min_file_size_bytes}), all new files will be smaller than the min threshold"
                ),
            ));
        }
        // (4) target < max.
        if target >= max_file_size_bytes {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "'target-file-size-bytes' ({target}) must be < 'max-file-size-bytes' \
                     ({max_file_size_bytes}), all new files will be larger than the max threshold"
                ),
            ));
        }
        // (5) min_input_files > 0.
        if self.min_input_files == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "'min-input-files' is set to 0 but must be > 0",
            ));
        }
        // (6) max_file_group_size_bytes > 0.
        if self.max_file_group_size_bytes == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "'max-file-group-size-bytes' is set to 0 but must be > 0",
            ));
        }

        // Derived after every precondition: both need `target < max`. Rolls at writeMax, not target.
        // `RewriteDataFiles` still rolls at its resolved target.
        let write_max_file_size =
            d2l(target as f64 + (max_file_size_bytes - target) as f64 * WRITE_MAX_FILE_SIZE_RATIO);

        let headroom = max_file_size_bytes.saturating_sub(write_max_file_size);
        let chunk_budget = CHUNK_MAX_SERIALIZED_BYTES.min(headroom / CHUNK_HEADROOM_FOOTER_SHARE);

        // Trivially true by construction. Loud red if an editor stops respecting the headroom.
        assert!(
            chunk_budget <= headroom,
            "the write-feed chunk budget ({chunk_budget}) must fit inside the candidate-filter \
             headroom ({headroom} = max {max_file_size_bytes} - write_max {write_max_file_size})"
        );

        Ok(ResolvedConfig {
            target_file_size_bytes: target,
            min_file_size_bytes,
            max_file_size_bytes,
            min_input_files: self.min_input_files,
            max_file_group_size_bytes: self.max_file_group_size_bytes,
            write_max_file_size,
            chunk_budget,
        })
    }

    /// Collect live parquet position deletes the filter admits, grouped by `(spec_id, partition)`.
    /// Filter per entry, as Java does at the scan. That changes when an unbindable filter errors.
    async fn collect_position_delete_groups(
        &self,
        snapshot: &Snapshot,
        partition_filter: &mut PartitionFilter,
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
                if data_file.content_type() != DataContentType::PositionDeletes {
                    continue;
                }
                // Fork divergence: skip Puffin DVs and V2 ORC/Avro. Java's planner is format-blind.
                if data_file.file_format() != DataFileFormat::Parquet {
                    continue;
                }
                if !partition_filter.matches(metadata, data_file)? {
                    continue;
                }
                let key = (data_file.partition_spec_id, data_file.partition().clone());
                groups.entry(key).or_default().push(LiveDeleteEntry {
                    data_file: data_file.clone(),
                    sequence_number: entry.sequence_number().unwrap_or(0),
                });
            }
        }

        Ok(groups)
    }

    /// Bind [`Self::filter`] once before the walk. `AlwaysTrue` never binds.
    fn bind_filter(&self) -> Result<PartitionFilter> {
        if matches!(self.filter, Predicate::AlwaysTrue) {
            return Ok(PartitionFilter::always_true());
        }
        let schema = self.table.metadata().current_schema().clone();
        let bound_row_filter = self.filter.clone().bind(schema, true).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                "RewritePositionDeleteFiles filter could not be bound to the table schema",
            )
            .with_source(e)
        })?;
        Ok(PartitionFilter::bound(bound_row_filter))
    }

    /// Compact one admitted bin. Stamp every output with THIS bin's max rewritten data-seq.
    /// Ranging over the partition or reusing another bin's max is a stamping error.
    async fn compact_group(
        &self,
        catalog: &dyn Catalog,
        table: &Table,
        bin: &AdmittedBin,
        config: &ResolvedConfig,
        starting_snapshot_id: i64,
        result: &mut RewritePositionDeleteFilesResult,
    ) -> Result<Table> {
        let (key, entries) = bin;

        let mut pairs: Vec<(String, i64)> = Vec::new();
        for entry in entries {
            self.read_position_pairs(table, &entry.data_file, &mut pairs)
                .await?;
        }

        // Per-bin skip. An early return in `execute` would drop every later bin.
        if pairs.is_empty() {
            return Ok(table.clone());
        }

        // Sort once, before any split. Per-chunk sort still writes every pair but breaks range pruning.
        pairs.sort();

        let new_files = self
            .write_compacted_file(table, key, &pairs, config)
            .await?;

        // THIS bin's max. Over-high over-applies; under-low resurrects.
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

        let rewritten_bytes: u64 = entries.iter().map(|e| e.data_file.file_size_in_bytes).sum();
        let rewritten_count = entries.len();
        let added_count = new_files.len();
        let mut added_bytes: u64 = 0;
        for file in &new_files {
            added_bytes = added_bytes
                .checked_add(file.file_size_in_bytes)
                .ok_or_else(|| Error::new(ErrorKind::Unexpected, "added bytes count overflow"))?;
        }
        let rewritten_files: Vec<DataFile> = entries.iter().map(|e| e.data_file.clone()).collect();

        // Stamp through the explicit-seq add, not the inherit add.
        let transaction = Transaction::new(table);
        let mut action = transaction
            .rewrite_files(Vec::new(), Vec::new())
            .delete_delete_files(rewritten_files);
        for file in new_files {
            action = action.add_delete_file_with_sequence_number(file, max_seq);
        }
        let action = action.validate_from_snapshot(starting_snapshot_id);
        let transaction = action.apply(transaction)?;
        let committed = transaction.commit(catalog).await?;

        result.rewritten_delete_files_count += rewritten_count;
        result.added_delete_files_count += added_count;
        result.rewritten_bytes_count = result
            .rewritten_bytes_count
            .checked_add(rewritten_bytes)
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "rewritten bytes count overflow"))?;
        result.added_bytes_count = result
            .added_bytes_count
            .checked_add(added_bytes)
            .ok_or_else(|| Error::new(ErrorKind::Unexpected, "added bytes count overflow"))?;

        Ok(committed)
    }

    /// Read reserved `file_path` and `pos` by field id, so a renamed column still reads.
    async fn read_position_pairs(
        &self,
        table: &Table,
        delete_file: &DataFile,
        pairs: &mut Vec<(String, i64)>,
    ) -> Result<()> {
        let loader = BasicDeleteFileLoader::new(table.file_io().clone());
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

    /// Write globally sorted pairs under the group spec. One rolling writer, bounded chunks.
    /// Do not use `new_with_default_file_size`: that hard-wires the 512 MiB data default.
    async fn write_compacted_file(
        &self,
        table: &Table,
        key: &GroupKey,
        pairs: &[(String, i64)],
        config: &ResolvedConfig,
    ) -> Result<Vec<DataFile>> {
        let metadata = table.metadata();
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

        let writer_config = PositionDeleteWriterConfig::new()?;
        let location_gen = DefaultLocationGenerator::new(metadata.clone())?;
        let file_name_gen = DefaultFileNameGenerator::new(
            "compacted-pos-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        // Keep path bounds full. The default `truncate(16)` would widen the path range.
        let parquet_builder = ParquetWriterBuilder::new(
            position_delete_writer_properties(),
            writer_config.schema().clone(),
        )
        .with_metrics_config(MetricsConfig::for_position_delete());
        // writeMax, not the resolved target. On 32-bit a larger bound saturates to "never roll".
        let rolling = RollingFileWriterBuilder::new(
            parquet_builder,
            usize::try_from(config.write_max_file_size).unwrap_or(usize::MAX),
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );

        // Always pass a PartitionKey, empty tuples included, so we never fabricate spec_id 0.
        let partition_key = PartitionKey::new(spec, schema.clone(), partition.clone())?;
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, writer_config.clone())
            .build(Some(partition_key))
            .await?;

        let mut start = 0usize;
        while start < pairs.len() {
            let end = chunk_end(pairs, start, config.chunk_budget);
            let chunk = &pairs[start..end];
            let paths: Vec<&str> = chunk.iter().map(|(path, _)| path.as_str()).collect();
            let positions: Vec<i64> = chunk.iter().map(|(_, pos)| *pos).collect();
            let batch = RecordBatch::try_new(writer_config.arrow_schema().clone(), vec![
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
            start = end;
        }

        require_non_empty(writer.close().await?)
    }

    /// Convert live filter-matching parquet position deletes into one Puffin DV per data file.
    ///
    /// # Errors
    ///
    /// Unreadable format, a live excluded delete a written vector would shadow, two live DVs
    /// for one data file, or a DV seq below its data file. `Ok` with zeros means nothing to convert.
    async fn rewrite_to_deletion_vectors(
        &self,
        catalog: &dyn Catalog,
        snapshot: &Snapshot,
        partition_filter: &mut PartitionFilter,
        starting_snapshot_id: i64,
    ) -> Result<RewritePositionDeleteFilesResult> {
        let inventory = self
            .collect_v3_delete_inventory(snapshot, partition_filter)
            .await?;
        if inventory.legacy_position_deletes.is_empty() {
            return Ok(RewritePositionDeleteFilesResult::default());
        }

        let (plans, superseded_puffin_paths) = self.plan_deletion_vectors(&inventory).await?;
        refuse_shadowed_deletes(&inventory, &plans)?;
        let new_deletion_vectors = self.write_deletion_vectors(&plans, &inventory).await?;

        // Path-keyed removal: superseding one DV removes every sibling in the same Puffin.
        let mut rewritten_files: Vec<DataFile> = inventory
            .legacy_position_deletes
            .iter()
            .map(|entry| entry.data_file.clone())
            .collect();
        rewritten_files.extend(
            inventory
                .deletion_vectors
                .values()
                .filter(|entry| superseded_puffin_paths.contains(entry.data_file.file_path()))
                .map(|entry| entry.data_file.clone()),
        );

        let result = summarize_v3_rewrite(&rewritten_files, &new_deletion_vectors)?;

        let transaction = Transaction::new(&self.table);
        let mut action = transaction
            .rewrite_files(Vec::new(), Vec::new())
            .delete_delete_files(rewritten_files);
        for delete_file in new_deletion_vectors {
            let sequence_number = deletion_vector_sequence_number(&delete_file, &plans)?;
            action = action.add_delete_file_with_sequence_number(delete_file, sequence_number);
        }
        let action = action.validate_from_snapshot(starting_snapshot_id);
        action.apply(transaction)?.commit(catalog).await?;

        Ok(result)
    }

    /// One walk: live data files, admitted parquet position deletes, and Puffin DVs by data file.
    ///
    /// # Errors
    ///
    /// A position delete that is neither Parquet nor Puffin, a DV with no referenced data file,
    /// or two live DVs for one data file.
    async fn collect_v3_delete_inventory(
        &self,
        snapshot: &Snapshot,
        partition_filter: &mut PartitionFilter,
    ) -> Result<V3DeleteInventory> {
        let metadata = self.table.metadata();
        let manifest_list = snapshot
            .load_manifest_list(self.table.file_io(), metadata)
            .await?;

        let mut inventory = V3DeleteInventory::default();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let data_file = entry.data_file();
                let sequence_number = entry.sequence_number().unwrap_or(0);
                match data_file.content_type() {
                    DataContentType::Data => {
                        inventory.data_files.insert(
                            data_file.file_path().to_string(),
                            LiveDataFile {
                                partition_spec_id: data_file.partition_spec_id,
                                partition: data_file.partition().clone(),
                                sequence_number,
                            },
                        );
                    }
                    DataContentType::PositionDeletes => {
                        inventory.admit_position_delete(
                            metadata,
                            partition_filter,
                            data_file,
                            sequence_number,
                        )?;
                    }
                    DataContentType::EqualityDeletes => {}
                }
            }
        }

        Ok(inventory)
    }

    /// Plan one merged DV per data file. Rewrite every sibling blob in a superseded Puffin.
    /// Merging a non-superset DV would make a shadowed position effective and delete live rows.
    async fn plan_deletion_vectors(
        &self,
        inventory: &V3DeleteInventory,
    ) -> Result<(HashMap<String, DeletionVectorPlan>, HashSet<String>)> {
        let mut plans: HashMap<String, DeletionVectorPlan> = HashMap::new();
        for entry in &inventory.legacy_position_deletes {
            let mut pairs: Vec<(String, i64)> = Vec::new();
            self.read_position_pairs(&self.table, &entry.data_file, &mut pairs)
                .await?;
            for (data_file_path, position) in pairs {
                let position = u64::try_from(position).map_err(|error| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Position delete '{}' has a negative position {position} for data file \
                             '{data_file_path}'",
                            entry.data_file.file_path()
                        ),
                    )
                    .with_source(error)
                })?;
                // Drop positions for data files the snapshot no longer holds. Refusing dead-ends the table.
                if !inventory.data_files.contains_key(&data_file_path) {
                    continue;
                }
                let plan = plans.entry(data_file_path).or_default();
                plan.positions.push(position);
                plan.sequence_number = plan.sequence_number.max(entry.sequence_number);
            }
        }

        let mut superseded_puffin_paths: HashSet<String> = plans
            .keys()
            .filter_map(|path| inventory.deletion_vectors.get(path))
            .map(|entry| entry.data_file.file_path().to_string())
            .collect();
        for (data_file_path, entry) in &inventory.deletion_vectors {
            if superseded_puffin_paths.contains(entry.data_file.file_path())
                && inventory.data_files.contains_key(data_file_path)
            {
                plans.entry(data_file_path.clone()).or_default();
            }
        }

        for (data_file_path, plan) in &mut plans {
            let data_file = inventory.live_data_file(data_file_path)?;
            if let Some(entry) = inventory.deletion_vectors.get(data_file_path) {
                let previous = load_delete_vector(self.table.file_io(), &entry.data_file).await?;
                // This DV already suppresses the legacy delete. Folding in a position it does not
                // hold would silently delete a live row. Refusal writes nothing.
                if let Some(unshadowed) = plan
                    .positions
                    .iter()
                    .find(|position| !previous.contains(**position))
                {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Data file '{data_file_path}' holds a deletion vector that does not \
                             cover position {unshadowed} of a legacy position delete it already \
                             suppresses. Converting would DELETE rows the table returns today, so \
                             this run refuses. THIS ACTION CANNOT CLEAR THAT STATE at any filter \
                             width. RewriteDataFiles with remove_dangling_deletes(true) clears it \
                             when the planner admits the file. The default delete-ratio-threshold \
                             0.3 admits a file whose file-scoped deletes cover at least 30% of its \
                             rows."
                        ),
                    ));
                }
                plan.positions.extend(previous.iter());
                plan.sequence_number = plan.sequence_number.max(entry.sequence_number);
                superseded_puffin_paths.insert(entry.data_file.file_path().to_string());
            }
            if plan.sequence_number < data_file.sequence_number {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Deletion vector for data file '{data_file_path}' would carry data sequence \
                         number {} but the data file is at {}",
                        plan.sequence_number, data_file.sequence_number
                    ),
                ));
            }
        }

        Ok((plans, superseded_puffin_paths))
    }

    /// Write every planned DV into one Puffin. Each `delete` carries that data file's PartitionKey.
    /// Do not use `with_partition_spec`: one Puffin spans every partition this arm touches.
    async fn write_deletion_vectors(
        &self,
        plans: &HashMap<String, DeletionVectorPlan>,
        inventory: &V3DeleteInventory,
    ) -> Result<Vec<DataFile>> {
        let metadata = self.table.metadata();
        let schema = metadata.current_schema().clone();
        let location_generator = DefaultLocationGenerator::new(metadata.clone())?;
        let file_name_generator = DefaultFileNameGenerator::new(
            "rewritten-dv".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Puffin,
        );
        let location =
            location_generator.generate_location(None, &file_name_generator.generate_file_name());
        let mut writer = DVFileWriter::new(self.table.file_io().new_output(location)?);

        for (data_file_path, plan) in plans {
            let data_file = inventory.live_data_file(data_file_path)?;
            let spec = metadata
                .partition_spec_by_id(data_file.partition_spec_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Data file '{data_file_path}' references unknown partition spec {}",
                            data_file.partition_spec_id
                        ),
                    )
                })?
                .as_ref()
                .clone();
            let partition_key =
                PartitionKey::new(spec, schema.clone(), data_file.partition.clone())?;
            for position in &plan.positions {
                writer.delete(data_file_path, *position, Some(&partition_key))?;
            }
        }

        writer.close().await
    }
}

/// Measured size of one pair: UTF-8 path length plus 8 for the `int64` position.
fn pair_serialized_bytes(pair: &(String, i64)) -> u64 {
    u64::try_from(pair.0.len())
        .unwrap_or(u64::MAX)
        .saturating_add(8)
}

/// Exclusive end of the chunk at `start`. Always at least one pair, or a zero budget would spin.
fn chunk_end(pairs: &[(String, i64)], start: usize, chunk_budget: u64) -> usize {
    let limit = pairs.len().min(start.saturating_add(CHUNK_PAIRS));
    let mut end = start;
    let mut bytes: u64 = 0;
    while end < limit {
        let next = bytes.saturating_add(pair_serialized_bytes(&pairs[end]));
        if end > start && next > chunk_budget {
            break;
        }
        bytes = next;
        end += 1;
    }
    end
}

/// Fail closed if a non-empty bin produces no file. The parquet writer treats that as normal.
/// Without this, `execute` would drop live position deletes and add none.
fn require_non_empty(files: Vec<DataFile>) -> Result<Vec<DataFile>> {
    if files.is_empty() {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "Position-delete writer produced no file for a non-empty input",
        ));
    }
    Ok(files)
}

/// User filter bound once, with the partition evaluator cached per `spec_id`.
/// `bound_row_filter == None` is `AlwaysTrue` and never binds.
struct PartitionFilter {
    bound_row_filter: Option<BoundPredicate>,
    by_spec_id: HashMap<i32, ExpressionEvaluator>,
}

impl PartitionFilter {
    /// Unfiltered default: every entry matches.
    fn always_true() -> Self {
        Self {
            bound_row_filter: None,
            by_spec_id: HashMap::new(),
        }
    }

    /// Filter already bound to the table schema by [`RewritePositionDeleteFiles::bind_filter`].
    fn bound(bound_row_filter: BoundPredicate) -> Self {
        Self {
            bound_row_filter: Some(bound_row_filter),
            by_spec_id: HashMap::new(),
        }
    }

    /// Whether `data_file`'s partition matches the filter. Same path as the table scan.
    fn matches(&mut self, metadata: &TableMetadata, data_file: &DataFile) -> Result<bool> {
        let Self {
            bound_row_filter,
            by_spec_id,
        } = self;
        let Some(bound_row_filter) = bound_row_filter.as_ref() else {
            return Ok(true);
        };

        let spec_id = data_file.partition_spec_id;
        let evaluator = match by_spec_id.entry(spec_id) {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => vacant.insert(build_partition_evaluator(
                metadata,
                bound_row_filter,
                spec_id,
                data_file.file_path(),
            )?),
        };
        evaluator.eval(data_file)
    }
}

/// Project the bound row filter onto spec `spec_id` and bind it to that spec's partition schema.
fn build_partition_evaluator(
    metadata: &TableMetadata,
    bound_row_filter: &BoundPredicate,
    spec_id: i32,
    file_path: &str,
) -> Result<ExpressionEvaluator> {
    let schema = metadata.current_schema().clone();
    let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Position delete '{file_path}' references unknown partition spec {spec_id}"),
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
        .project(bound_row_filter)?
        .rewrite_not()
        .bind(partition_schema, true)?;

    Ok(ExpressionEvaluator::new(partition_filter))
}

/// Candidate-filter, pack, then group-filter. Packing first would split candidates that belong together.
fn plan_bins(
    groups: HashMap<GroupKey, Vec<LiveDeleteEntry>>,
    config: &ResolvedConfig,
) -> Vec<AdmittedBin> {
    let mut admitted: Vec<AdmittedBin> = Vec::new();

    for (key, entries) in groups {
        let candidates: Vec<LiveDeleteEntry> = entries
            .into_iter()
            .filter(|entry| is_candidate(entry, config))
            .collect();
        if candidates.is_empty() {
            continue;
        }

        let bins = pack_bins(
            candidates,
            |entry| entry.data_file.file_size_in_bytes,
            config.max_file_group_size_bytes,
        );

        for bin in bins {
            if group_qualifies(&bin, config) {
                admitted.push((key.clone(), bin));
            }
        }
    }

    admitted
}

/// Candidate iff undersized or oversized, both strict. No delete-count clause: that is the data planner.
fn is_candidate(entry: &LiveDeleteEntry, config: &ResolvedConfig) -> bool {
    let length = entry.data_file.file_size_in_bytes;
    length < config.min_file_size_bytes || length > config.max_file_size_bytes
}

/// Java `filterFileGroups`. `too_much_content` has no `size > 1` guard, so a lone oversized file
/// is admitted. Do not delete the unreachable leaves: white-box tests kill those mutants.
/// Saturate the input-size sum. Java wraps; a wrapped negative would decline a bin we must admit.
fn group_qualifies(bin: &[LiveDeleteEntry], config: &ResolvedConfig) -> bool {
    let size = bin.len();
    let input_size: u64 = bin.iter().fold(0u64, |sum, entry| {
        sum.saturating_add(entry.data_file.file_size_in_bytes)
    });

    let enough_input_files = size > 1 && size >= config.min_input_files;
    let enough_content = size > 1 && input_size > config.target_file_size_bytes;
    let too_much_content = input_size > config.max_file_size_bytes;

    enough_input_files || enough_content || too_much_content
}

/// Locate `file_path` and `pos` by reserved field id, never by name or column order.
///
/// # Errors
///
/// A reserved column is absent or has the wrong Arrow type.
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

/// A live parquet position-delete entry and its post-inheritance data sequence number.
struct LiveDeleteEntry {
    data_file: DataFile,
    sequence_number: i64,
}

/// Live data file: the partition a covering DV must carry, and the seq it must not fall below.
struct LiveDataFile {
    partition_spec_id: i32,
    partition: Struct,
    sequence_number: i64,
}

/// Live delete inventory of a V3 table, taken in one manifest walk.
#[derive(Default)]
struct V3DeleteInventory {
    data_files: HashMap<String, LiveDataFile>,
    /// Parquet position deletes the filter admits.
    legacy_position_deletes: Vec<LiveDeleteEntry>,
    /// Puffin DVs keyed by the data file each one references.
    deletion_vectors: HashMap<String, LiveDeleteEntry>,
    /// Position deletes the filter rejected. A new DV covering their data file would shadow them.
    unconverted_position_deletes: Vec<LiveDeleteEntry>,
}

impl V3DeleteInventory {
    /// Route one live position-delete entry into the inventory.
    ///
    /// # Errors
    ///
    /// Neither Parquet nor Puffin, a DV with no referenced data file, or a second live DV for one file.
    fn admit_position_delete(
        &mut self,
        metadata: &TableMetadata,
        partition_filter: &mut PartitionFilter,
        data_file: &DataFile,
        sequence_number: i64,
    ) -> Result<()> {
        let entry = LiveDeleteEntry {
            data_file: data_file.clone(),
            sequence_number,
        };
        match data_file.file_format() {
            DataFileFormat::Puffin => {
                let referenced = referenced_data_file_location(data_file).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Deletion vector '{}' names no referenced data file",
                            data_file.file_path()
                        ),
                    )
                })?;
                if self
                    .deletion_vectors
                    .insert(referenced.clone(), entry)
                    .is_some()
                {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Data file '{referenced}' has more than one live deletion vector"),
                    ));
                }
                Ok(())
            }
            DataFileFormat::Parquet => {
                if partition_filter.matches(metadata, data_file)? {
                    self.legacy_position_deletes.push(entry);
                } else {
                    self.unconverted_position_deletes.push(entry);
                }
                Ok(())
            }
            // Refuse, do not skip. Skipping would make zero counts mean "did not look".
            format => {
                if partition_filter.matches(metadata, data_file)? {
                    return Err(Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!(
                            "Position delete '{}' is {format}: only Parquet position deletes and \
                             Puffin deletion vectors are supported on format version 3",
                            data_file.file_path()
                        ),
                    ));
                }
                self.unconverted_position_deletes.push(entry);
                Ok(())
            }
        }
    }

    /// The live data file at `data_file_path`. A miss is a planner bug, not a table state.
    fn live_data_file(&self, data_file_path: &str) -> Result<&LiveDataFile> {
        self.data_files.get(data_file_path).ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "Planned deletion vector names data file '{data_file_path}', which is not live"
                ),
            )
        })
    }
}

/// Merged positions and the data sequence number to stamp on one written DV.
#[derive(Default)]
struct DeletionVectorPlan {
    positions: Vec<u64>,
    sequence_number: i64,
}

/// Refuse a run that would leave a live position delete shadowed by a DV this run writes.
/// Re-derives reader routing; it does not share `PopulatedDeleteFileIndex` keying.
///
/// # Errors
///
/// `DataInvalid`. Widen the filter, unless the delete is ORC or Avro, which no width converts.
fn refuse_shadowed_deletes(
    inventory: &V3DeleteInventory,
    plans: &HashMap<String, DeletionVectorPlan>,
) -> Result<()> {
    if plans.is_empty() {
        return Ok(());
    }
    let mut planned_partitions: HashSet<(i32, &Struct)> = HashSet::new();
    for data_file_path in plans.keys() {
        let data_file = inventory.live_data_file(data_file_path)?;
        planned_partitions.insert((data_file.partition_spec_id, &data_file.partition));
    }

    for entry in &inventory.unconverted_position_deletes {
        let delete_file = &entry.data_file;
        let shadowed_data_file = match referenced_data_file_location(delete_file) {
            Some(referenced) => plans.contains_key(&referenced).then_some(referenced),
            // Partition-scoped: do not apply the seq rule. A false alarm beats losing rows.
            None => planned_partitions
                .contains(&(delete_file.partition_spec_id, delete_file.partition()))
                .then(|| "a data file in the same partition".to_string()),
        };
        let Some(shadowed_data_file) = shadowed_data_file else {
            continue;
        };
        let remedy = if delete_file.file_format() == DataFileFormat::Parquet {
            "Widen the filter so the same run converts it."
        } else {
            "This arm cannot read that format, so NO filter setting converts this table."
        };
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Position delete '{}' still applies to {shadowed_data_file} but the filter excluded \
                 it: the deletion vector this run would write there SHADOWS it and its deleted rows \
                 would come back. {remedy}",
                delete_file.file_path()
            ),
        ));
    }
    Ok(())
}

/// Stamp for one written DV, read back from the plan it came from.
fn deletion_vector_sequence_number(
    delete_file: &DataFile,
    plans: &HashMap<String, DeletionVectorPlan>,
) -> Result<i64> {
    delete_file
        .referenced_data_file()
        .and_then(|path| plans.get(&path))
        .map(|plan| plan.sequence_number)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "Written deletion vector '{}' has no planned sequence number",
                    delete_file.file_path()
                ),
            )
        })
}

/// Result counts of one V3 rewrite. Sum bytes over distinct paths: each DV `DataFile` carries the
/// whole Puffin size, so a per-entry sum would count the same bytes once per blob.
fn summarize_v3_rewrite(
    rewritten_files: &[DataFile],
    added_files: &[DataFile],
) -> Result<RewritePositionDeleteFilesResult> {
    let distinct_bytes = |files: &[DataFile]| -> Result<u64> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut total: u64 = 0;
        for file in files {
            if seen.insert(file.file_path()) {
                total = total.checked_add(file.file_size_in_bytes).ok_or_else(|| {
                    Error::new(ErrorKind::Unexpected, "rewrite bytes count overflow")
                })?;
            }
        }
        Ok(total)
    };

    Ok(RewritePositionDeleteFilesResult {
        rewritten_delete_files_count: rewritten_files.len(),
        added_delete_files_count: added_files.len(),
        rewritten_bytes_count: distinct_bytes(rewritten_files)?,
        added_bytes_count: distinct_bytes(added_files)?,
    })
}

#[cfg(test)]
#[path = "rewrite_position_delete_files_tests.rs"]
mod tests;
