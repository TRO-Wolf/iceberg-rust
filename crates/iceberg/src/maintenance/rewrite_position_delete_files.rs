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
//! documented on `docs/parity/GAP_MATRIX.md` row R136.
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
    Struct, TableProperties,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig, position_delete_writer_properties,
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

/// Java `SizeBasedFileRewritePlanner.MIN_FILE_SIZE_DEFAULT_RATIO` — an unset `min_file_size_bytes`
/// defaults to 75% of the resolved target (bytecode-verified vs `iceberg-core-1.10.0.jar`:
/// `sizeThresholds` loads `ldc2_w // double 0.75d` and multiplies).
const MIN_FILE_SIZE_DEFAULT_RATIO: f64 = 0.75;

/// Java `SizeBasedFileRewritePlanner.MAX_FILE_SIZE_DEFAULT_RATIO` — an unset `max_file_size_bytes`
/// defaults to 180% of the resolved target (bytecode-verified vs `iceberg-core-1.10.0.jar`:
/// `sizeThresholds` loads `ldc2_w // double 1.8d` and multiplies).
const MAX_FILE_SIZE_DEFAULT_RATIO: f64 = 1.80;

/// Java `SizeBasedFileRewritePlanner.MIN_INPUT_FILES_DEFAULT` = `5` (bytecode-verified vs
/// `iceberg-core-1.10.0.jar`). THE parity number this action was missing: Java admits a group of
/// small position-delete files only from five files up, where this port used to admit two.
const MIN_INPUT_FILES_DEFAULT: usize = 5;

/// Java `SizeBasedFileRewritePlanner.MAX_FILE_GROUP_SIZE_BYTES_DEFAULT` = `107374182400` (100 GiB,
/// bytecode-verified vs `iceberg-core-1.10.0.jar`).
const MAX_FILE_GROUP_SIZE_BYTES_DEFAULT: u64 = 107374182400;

/// [`TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT`] in the `i64` domain the
/// parse works in (see [`parse_delete_target_file_size`]). The `as` is proven lossless by the
/// const-evaluated assertion, so the cast can never truncate.
const DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT_I64: i64 = {
    let default = TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT;
    assert!(default <= i64::MAX as u64);
    default as i64
};

/// Java's `d2l` on the ratio products in `SizeBasedFileRewritePlanner.sizeThresholds` — the JVM's
/// `(long) doubleValue`, which SATURATES at `Long.MAX_VALUE` (and at `Long.MIN_VALUE` / `0` for NaN).
///
/// Rust's `as u64` on an `f64` also saturates, but at `u64::MAX` — roughly twice Java's ceiling — so
/// the `.min(i64::MAX as u64)` IS the parity act, not defensive padding: without it a target above
/// `2^63 / 1.8` resolves a `max_file_size_bytes` Java can never produce. Negative and NaN inputs map
/// to `0` here where Java gives a negative; that divergence is unreachable in this action (a negative
/// target is rejected by precondition (1) of [`RewritePositionDeleteFiles::resolve_config`]) and is
/// stated rather than assumed.
///
/// NAMED RESIDUE (RES-9): the sibling [`super::rewrite_data_files`] resolver at its own ratio-default
/// site is UNCLAMPED and deliberately stays so; this action does not change the template.
fn d2l(x: f64) -> u64 {
    (x as u64).min(i64::MAX as u64)
}

/// Parse the `write.delete.target-file-size-bytes` table property — Java
/// `BinPackRewritePositionDeletePlanner.defaultTargetFileSize()` via `PropertyUtil.propertyAsLong`,
/// which is `map.get(key)`, `null -> default`, else `Long.parseLong`.
///
/// Parses into **`i64`, not `u64`, on purpose**: `i64::from_str`'s accept/reject domain COINCIDES with
/// `Long.parseLong`'s (both take an optional leading `+`/`-` then decimal digits only — no
/// underscores, whitespace or radix prefix — and both reject a magnitude outside the 64-bit signed
/// range). ONE stated exception, fail-closed: `Long.parseLong` digits through `Character.digit`, so
/// it also accepts non-ASCII Unicode decimal digits (Arabic-Indic `U+0660`–`U+0669` and the other
/// decimal-digit ranges) where Rust's parser is ASCII-only. The fork REJECTS what Java would accept
/// there, so the divergence errors loudly rather than resolving a different threshold, and no real
/// table carries such a value.
///
/// So `"0"` and `"-1"` PARSE here exactly as they parse in Java, and are then rejected
/// downstream by [`RewritePositionDeleteFiles::resolve_config`]'s `target > 0` precondition carrying
/// Java's verbatim message. A `u64` parse would reject `"-1"` at the parse with a fork-only message
/// and would ADMIT values above `i64::MAX` that `Long.parseLong` throws on.
///
/// NAMED RESIDUE (RES-10): the sibling `rewrite_data_files::parse_target_file_size` parses `u64` and
/// stays as-is; that asymmetry is a follow-up, not a change this action makes.
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

/// The rejection for a builder override outside Java's `long` domain — precondition (7) of
/// [`RewritePositionDeleteFiles::resolve_config`], reusing [`parse_delete_target_file_size`]'s
/// out-of-range shape. Java's thresholds are `long` fields fed by `Long.parseLong`, so a `u64`
/// above `i64::MAX` is a config Java cannot express; the fork refuses it rather than silently
/// accepting a state in which `too_much_content` is unreachable.
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

/// The resolved size/count thresholds for one [`RewritePositionDeleteFiles::execute`] run — Java
/// `SizeBasedFileRewritePlanner`'s post-`init` field set, restricted to the five options this action
/// ports. Produced by [`RewritePositionDeleteFiles::resolve_config`], which is the ONLY place the
/// defaults and Java's preconditions live.
///
/// The fields are consumed by the planner increment (candidate filter, bin packer, group gate).
/// Keep the `Clone`/`PartialEq`/`Eq` derives — they are what keeps `dead_code` disarmed here.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedConfig {
    target_file_size_bytes: u64,
    min_file_size_bytes: u64,
    max_file_size_bytes: u64,
    min_input_files: usize,
    max_file_group_size_bytes: u64,
}

/// The `RewritePositionDeleteFiles` maintenance action. Build it with [`Self::new`], optionally restrict
/// the compacted partitions with [`Self::filter`], configure the size / count thresholds with the
/// builder methods, and run it with [`Self::execute`].
///
/// # Ported options (five)
///
/// [`Self::target_file_size_bytes`], [`Self::min_file_size_bytes`], [`Self::max_file_size_bytes`],
/// [`Self::min_input_files`] and [`Self::max_file_group_size_bytes`] — Java's
/// `SizeBasedFileRewritePlanner.validOptions()` minus the deferrals below. Defaults are Java's:
///
/// - `target_file_size_bytes` = `write.delete.target-file-size-bytes` (default 64 MiB — the
///   DELETE-specific property, NOT the 512 MiB `write.target-file-size-bytes` the data-file planner
///   reads).
/// - `min_file_size_bytes` = `0.75 * target` (resolved lazily when unset).
/// - `max_file_size_bytes` = `1.8 * target` (resolved lazily when unset).
/// - `min_input_files` = 5.
/// - `max_file_group_size_bytes` = 100 GiB.
// TODO(doc pass): the `# Deferred (loudly)` block enumerating the six unported options
// (`rewrite-all` with its inverted-emulation warning, `rewrite-job-order`,
// `partial-progress.enabled`, `partial-progress.max-commits`, `max-concurrent-file-group-rewrites`
// and `output-spec-id` with its carry-forward citation) is written in this PR's documentation
// increment. It is deliberately NOT drafted here — its text is owned by that increment.
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
    /// Create a `RewritePositionDeleteFiles` action for `table` with Java's defaults (see the struct
    /// docs). With no [`filter`](Self::filter), every `(spec, partition)` group of live parquet
    /// position-delete files in the current snapshot is considered. The size thresholds are resolved
    /// lazily at [`Self::execute`] from the table's `write.delete.target-file-size-bytes` property when
    /// not overridden.
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

    /// Set the target output position-delete file size in bytes (Java `TARGET_FILE_SIZE_BYTES`). When
    /// unset, the table's `write.delete.target-file-size-bytes` property is used (default 64 MiB).
    /// Setting this also shifts the default `min`/`max` thresholds (0.75× / 1.8× of the target) unless
    /// those are independently overridden.
    ///
    /// Values above `i64::MAX` are rejected at [`Self::execute`] — Java's option is a `long`.
    pub fn target_file_size_bytes(mut self, target_file_size_bytes: u64) -> Self {
        self.target_file_size_bytes = Some(target_file_size_bytes);
        self
    }

    /// Position-delete files smaller than this are always candidates for compaction (Java
    /// `MIN_FILE_SIZE_BYTES`). Defaults to 75% of the target file size.
    ///
    /// Values above `i64::MAX` are rejected at [`Self::execute`] — Java's option is a `long`.
    pub fn min_file_size_bytes(mut self, min_file_size_bytes: u64) -> Self {
        self.min_file_size_bytes = Some(min_file_size_bytes);
        self
    }

    /// Position-delete files larger than this are always candidates for compaction (Java
    /// `MAX_FILE_SIZE_BYTES`). Defaults to 180% of the target file size.
    ///
    /// Values above `i64::MAX` are rejected at [`Self::execute`] — Java's option is a `long`.
    pub fn max_file_size_bytes(mut self, max_file_size_bytes: u64) -> Self {
        self.max_file_size_bytes = Some(max_file_size_bytes);
        self
    }

    /// A group with at least this many position-delete files is compacted regardless of total size
    /// (Java `MIN_INPUT_FILES`, default 5). Must be `> 0` — a zero is rejected at [`Self::execute`].
    pub fn min_input_files(mut self, min_input_files: usize) -> Self {
        self.min_input_files = min_input_files;
        self
    }

    /// The largest total size of input position-delete files compacted in a single group (Java
    /// `MAX_FILE_GROUP_SIZE_BYTES`, default 100 GiB). Must be `> 0`.
    pub fn max_file_group_size_bytes(mut self, max_file_group_size_bytes: u64) -> Self {
        self.max_file_group_size_bytes = max_file_group_size_bytes;
        self
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
        // Resolve + VALIDATE the size / count thresholds BEFORE any manifest is read, mirroring Java,
        // where `sizeThresholds` runs at planner `init` and therefore before planning.
        //
        // BEHAVIOUR FLIP (4th, for the PR body): `write.delete.target-file-size-bytes` is a standard
        // Iceberg property that Java writers and users already set, so a PRE-EXISTING table carrying
        // it with an unparsable value, a value above `i64::MAX`, or a value <= 1 now makes `execute`
        // return `Err` where it previously returned counts. (At 1 the defaulted max is `d2l(1.8)`
        // = 1, so the STRICT `target < max` fires; from 2 up every precondition passes.) This is
        // parity-correct — Java throws on exactly the same inputs — but it IS a flip, and R-9's
        // PR-body list currently names only three.
        //
        // TODO(size gate): bound as `_config` because the admission gate that CONSUMES it lands in
        // the planner increment of this PR; the `entries.len() < 2` guard below is still the
        // pre-parity admission rule until then.
        let _config = self.resolve_config()?;

        let metadata = self.table.metadata();
        let Some(snapshot) = metadata.current_snapshot().cloned() else {
            return Ok(RewritePositionDeleteFilesResult::default());
        };
        let starting_snapshot_id = snapshot.snapshot_id();

        // (1) Enumerate the live PARQUET position-delete entries, grouped by (spec_id, partition).
        // Puffin DVs are SKIPPED (file-scoped, never bin-packed) — the documented V2-parquet-only scope.
        let groups = self.collect_position_delete_groups(&snapshot).await?;

        // Advance the base table after each group commit so the next group's `Transaction` is
        // built on the committed tip (mirrors RewriteDataFiles). Without this, groups 2..N still
        // succeed via `Transaction::do_commit`'s stale-base refresh + re-apply, but each group
        // pays a full rewrite re-apply against the refreshed tip. Advancing avoids that redundant
        // re-apply work; it is not required for CAS correctness under the retry/refresh loop.
        let mut table = self.table.clone();
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

            table = self
                .compact_group(
                    catalog,
                    &table,
                    &key,
                    &entries,
                    starting_snapshot_id,
                    &mut result,
                )
                .await?;
        }

        Ok(result)
    }

    /// Resolve the size / count thresholds and enforce Java's `SizeBasedFileRewritePlanner`
    /// preconditions. This is the single home for the defaults, the property lookup and the
    /// rejections — every default pin calls it directly.
    ///
    /// # Order (Java's, plus this port's three additions)
    ///
    /// Java's `sizeThresholds(Map)` resolves the min/max DEFAULTS first and only then checks, in this
    /// order: (1) `target > 0`, (2) `min >= 0`, (3) `target > min`, (4) `target < max` — all STRICT,
    /// each carrying the verbatim message reproduced below. The order is load-bearing: at a negative
    /// target this port's [`d2l`] saturates BOTH ratio products to `0`, so (3) `target > min` is
    /// independently false there; hoisting (3) above (1) would report the wrong message.
    ///
    /// **(2) `min >= 0` is STRUCTURALLY UNREACHABLE here and is therefore not coded**: this port's
    /// `min_file_size_bytes` is a `u64` with no table property behind it, only the builder, so no
    /// caller can express a negative. It is recorded rather than written as a comparison that is
    /// always true.
    ///
    /// Three preconditions live OUTSIDE Java's `sizeThresholds`. Only (7) is fork-authored — (5) and
    /// (6) ARE Java's, just raised elsewhere, so their message shape is Java's and is not ours to
    /// reword:
    ///
    /// - (5) `min_input_files > 0` and (6) `max_file_group_size_bytes > 0` — Java's own
    ///   `checkArgument`s on the same two options, raised in `SizeBasedFileRewritePlanner.init(Map)`
    ///   rather than in `sizeThresholds`. Reproduced here in Java's message shape, checked after (4).
    /// - (7) every EXPLICIT builder override of the three size knobs is `<= i64::MAX`. THIS one has
    ///   no Java analogue: Java's thresholds are `long`s fed by `Long.parseLong`, so a larger value
    ///   is a config Java cannot express, and admitting it would open a state in which
    ///   `too_much_content` (`input_size > max`) is unreachable.
    ///
    ///   **(7) is checked when each override is READ — i.e. before the defaults resolve and before
    ///   (1)** — and this position is a deliberate, recorded deviation from the ledger's numbering.
    ///   TWO reasons, both verified by removing the checks rather than argued:
    ///
    ///   1. **(7) is what makes the `as i64` cast below lossless.** The target override is narrowed
    ///      to `i64` before the defaults resolve; with (7)'s target leg removed, a `u64` above
    ///      `i64::MAX` WRAPS to a negative and (1) then reports
    ///      `'target-file-size-bytes' is set to -9223372036854775808 but must be > 0` — a value the
    ///      caller never wrote, leaked into user-facing error text. The cast's soundness rests on
    ///      (7) running first.
    ///   2. **Checked last, only ONE of its three legs would be observable.** A `min` override above
    ///      `i64::MAX` is caught first by (3) (`target > min` fails, wrong message); a `target`
    ///      override above `i64::MAX` is caught first by (1) via the wrap in point 1. Only the `max`
    ///      leg would survive to report its own rejection.
    ///
    ///   Checking the overrides at read time is therefore the only placement under which each knob
    ///   reports its own rejection AND the narrowing cast is total, and it leaves the relative order
    ///   of Java's own (1) → (3) → (4) untouched.
    fn resolve_config(&self) -> Result<ResolvedConfig> {
        // (7) at READ time, one leg per knob, in builder-declaration order.
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

        // The target stays in the `i64` domain until precondition (1) has run, so a negative property
        // value reaches (1) exactly as it reaches Java's `checkArgument`.
        let target = match self.target_file_size_bytes {
            // In range by (7) above, so the widening cast cannot truncate.
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

        Ok(ResolvedConfig {
            target_file_size_bytes: target,
            min_file_size_bytes,
            max_file_size_bytes,
            min_input_files: self.min_input_files,
            max_file_group_size_bytes: self.max_file_group_size_bytes,
        })
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
    ///
    /// Returns the committed [`Table`] so the caller can advance the base for the next group
    /// (mirrors [`crate::maintenance::rewrite_data_files::RewriteDataFiles`]).
    async fn compact_group(
        &self,
        catalog: &dyn Catalog,
        table: &Table,
        key: &GroupKey,
        entries: &[LiveDeleteEntry],
        starting_snapshot_id: i64,
        result: &mut RewritePositionDeleteFilesResult,
    ) -> Result<Table> {
        // (2) Read + concat the (file_path, pos) pairs across the group.
        let mut pairs: Vec<(String, i64)> = Vec::new();
        for entry in entries {
            self.read_position_pairs(table, &entry.data_file, &mut pairs)
                .await?;
        }

        // A group of live pos-delete files always carries rows (a position delete with no rows is
        // degenerate); if somehow empty, there is nothing to compact — leave the group untouched.
        if pairs.is_empty() {
            return Ok(table.clone());
        }

        // Spec-recommended position-delete ordering: sort by (file_path, pos). Java does NOT dedup within
        // a group (the reader bitmap dedups); we keep duplicates and only sort.
        pairs.sort();

        // (3) Write FEWER position-delete files (one per group) under the group spec + partition key.
        let new_file = self.write_compacted_file(table, key, &pairs).await?;

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
        let transaction = Transaction::new(table);
        let action = transaction
            .rewrite_files(Vec::new(), Vec::new())
            .delete_delete_files(rewritten_files)
            .add_delete_file_with_sequence_number(new_file, max_seq)
            .validate_from_snapshot(starting_snapshot_id);
        let transaction = action.apply(transaction)?;
        let committed = transaction.commit(catalog).await?;

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

        Ok(committed)
    }

    /// Read one parquet position-delete file's two RESERVED columns — `file_path` (field id
    /// [`RESERVED_FIELD_ID_DELETE_FILE_PATH`]) and `pos` (field id [`RESERVED_FIELD_ID_DELETE_FILE_POS`])
    /// — by FIELD ID, not by name, appending every `(file_path, pos)` pair into `pairs`. The columns are
    /// located by their `PARQUET_FIELD_ID_META_KEY` metadata so a renamed-but-correctly-id'd file still
    /// reads (interop-faithful).
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

    /// Write the sorted `(file_path, pos)` pairs into ONE compacted position-delete file under the
    /// group's spec + partition key, returning the resulting [`DataFile`].
    async fn write_compacted_file(
        &self,
        table: &Table,
        key: &GroupKey,
        pairs: &[(String, i64)],
    ) -> Result<DataFile> {
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

        let config = PositionDeleteWriterConfig::new()?;
        let location_gen = DefaultLocationGenerator::new(metadata.clone())?;
        let file_name_gen = DefaultFileNameGenerator::new(
            "compacted-pos-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        // Position-delete files keep `file_path`/`pos` bounds FULL (Java `MetricsConfig.forPositionDelete`)
        // so delete-file path pruning stays precise — the default `truncate(16)` would widen the path range.
        let parquet_builder =
            ParquetWriterBuilder::new(position_delete_writer_properties(), config.schema().clone())
                .with_metrics_config(MetricsConfig::for_position_delete());
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );

        // The new pos-delete must live in the SAME partition + spec as the files it replaces (so it
        // lands in the same bucket and applies to the same data files). Always pass a PartitionKey —
        // including empty/unpartitioned and all-Void null tuples — so we never fabricate spec_id 0
        // via `build(None)` without `with_partition_spec` (C2-L-001 / C1-L-001 class).
        let partition_key = PartitionKey::new(spec, schema.clone(), partition.clone())?;
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
            .build(Some(partition_key))
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
