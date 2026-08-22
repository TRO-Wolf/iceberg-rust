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
//! # Planning — Java's six-stage pipeline
//!
//! Between the manifest walk and the first commit, the live entries pass through exactly six stages,
//! in Java's order (`BinPackRewritePositionDeletePlanner.planFileGroups` +
//! `SizeBasedFileRewritePlanner.filterFileGroups`, with the user filter applied at Java's
//! `planFiles` scan):
//!
//! 1. **Collect** the live `PositionDeletes` entries, skipping every non-PARQUET delete file (the
//!    fork-only V2-parquet scope — see the divergence section below).
//! 2. **User filter** — [`RewritePositionDeleteFiles::filter`] is applied PER ENTRY *at collection*,
//!    because Java applies it at the `PositionDeletes` scan, STRICTLY BEFORE grouping. The predicate
//!    is bound to the table schema ONCE (right after the no-snapshot early return) and the projected
//!    partition evaluator is cached per `spec_id`, so binding is O(specs), not O(entries).
//! 3. **Group** by `(spec_id, partition)`.
//! 4. **Candidate filter** — Java `filterFiles` → `outsideDesiredFileSizeRange`: a file is a
//!    candidate iff `length < min_file_size_bytes || length > max_file_size_bytes`, both STRICT.
//!    There is NO delete-count clause here: `tooManyDeletes` is `BinPackRewriteFilePlanner`-only and
//!    the position-delete planner does not inherit it.
//! 5. **Bin-pack** each partition's candidates through the shared
//!    [`pack_bins`](super::rewrite_data_files::pack_bins) — Java
//!    `BinPacking$ListPacker(maxGroupSize, lookback = 1, largestBinFirst = false)`, inherited
//!    unchanged by this planner.
//! 6. **Group filter** — Java `filterFileGroups`, a plain three-way disjunction with no fourth
//!    clause: `enough_input_files || enough_content || too_much_content`.
//!
//! [`RewritePositionDeleteFiles::execute`] then iterates BINS (not partitions), committing one
//! `Replace` snapshot per admitted bin.
//!
//! # Named NON-PORT: Java's `inputSplitSize` / `expectedOutputFiles`
//!
//! `SizeBasedFileRewritePlanner.inputSplitSize(long)` and its helper
//! `SizeBasedFileRewritePlanner.expectedOutputFiles(long)` are DELIBERATELY NOT PORTED. The bytecode
//! reason: `SparkRewritePositionDeleteRunner.doRewrite` consumes `inputSplitSize` on the **READ**
//! side, as the scan option `split-size` passed to `DataFrameReader.option`; the **WRITE** bound is a
//! SEPARATE option (`target-delete-file-size-bytes`), fed from `group.maxOutputFileSize()` =
//! `writeMaxFileSize()`. Java never chunks a writer feed with it. This action reads the
//! `(file_path, pos)` pairs directly and has NO split-size-driven scan for either function to
//! parameterise, so a port would be dead code.
//!
//! GUARD ON THE PIN — do not "helpfully" broaden it. The tripwire for this non-port is a repo-wide
//! grep for the SNAKE_CASE Rust spellings of those two Java names, which must return ZERO hits. It is
//! snake-case-ONLY **by design**, because the camelCase Java names MUST appear: this very paragraph
//! names them, and [`super::rewrite_data_files`]'s "Deferred (loudly)" rustdoc cites `inputSplitSize`
//! correctly and load-bearingly. Re-broadening the pattern to the camelCase spellings would classify
//! both of those required, correct sentences as violations.
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
//! that matches none, no CANDIDATE in a partition, or a bin the three-clause admission gate declines,
//! the action commits NOTHING for that bin and the result counts stay zero (Java commits only when
//! there is real compaction work). A bin of ONE file is declined by the two `size > 1` guards —
//! UNLESS that file is larger than `max_file_size_bytes`, which `too_much_content` admits with no
//! such guard.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::StreamExt;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::rewrite_data_files::{
    MAX_FILE_GROUP_SIZE_BYTES_DEFAULT, MAX_FILE_SIZE_DEFAULT_RATIO, MIN_FILE_SIZE_DEFAULT_RATIO,
    MIN_INPUT_FILES_DEFAULT, pack_bins,
};
use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::expr::visitors::expression_evaluator::ExpressionEvaluator;
use crate::expr::visitors::inclusive_projection::InclusiveProjection;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, MetricsConfig, PartitionKey, Schema, Snapshot,
    Struct, TableMetadata, TableProperties,
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

/// One ADMITTED BIN as [`plan_bins`] emits it: the `(spec_id, partition)` key it was grouped under,
/// and the member entries the packer put in this bin. Kept as one value so `compact_group` takes the
/// bin as a UNIT — the two halves are never meaningful apart, and the seq stamp is defined over
/// exactly this entry set (C-010).
type AdmittedBin = (GroupKey, Vec<LiveDeleteEntry>);

// ONE HOME for the four Java planner constants, imported above from the sibling template rather
// than duplicated here: `MIN_FILE_SIZE_DEFAULT_RATIO` (0.75), `MAX_FILE_SIZE_DEFAULT_RATIO` (1.8),
// `MIN_INPUT_FILES_DEFAULT` (5) and `MAX_FILE_GROUP_SIZE_BYTES_DEFAULT` (107374182400 = 100 GiB).
// All four are `SizeBasedFileRewritePlanner`'s and are inherited UNCHANGED by
// `BinPackRewritePositionDeletePlanner` (bytecode-verified vs `iceberg-core-1.10.0.jar`:
// `sizeThresholds` loads `ldc2_w // double 0.75d` and `ldc2_w // double 1.8d`), so a single home
// is what keeps the two ports from drifting apart. `MIN_INPUT_FILES_DEFAULT` = 5 is THE parity
// number this action was missing: Java admits a group of small position-delete files only from
// FIVE files up, where this port used to admit two.

/// [`TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT`] in the `i64` domain the
/// parse works in (see [`parse_delete_target_file_size`]). The `as` is proven lossless by the
/// const-evaluated assertion, so the cast can never truncate.
const DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT_I64: i64 = {
    let default = TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT;
    assert!(default <= i64::MAX as u64);
    default as i64
};

/// Java's `SizeBasedFileRewritePlanner.writeMaxFileSize()` band ratio — the `ldc2_w // double 0.5d`
/// the bytecode multiplies the `(max_file_size - target)` band by before adding it back to the
/// target (offsets 0-21: `getfield targetFileSize; l2d; getfield maxFileSize; getfield
/// targetFileSize; lsub; l2d; ldc2_w 0.5d; dmul; dadd; d2l`). PARITY-CITED, not fork-authored.
const WRITE_MAX_FILE_SIZE_RATIO: f64 = 0.5;

/// FORK-AUTHORED — **no Java analogue** (see the module rustdoc's named non-port: Java applies
/// `inputSplitSize` on the READ side and never chunks a writer feed). The maximum number of
/// `(file_path, pos)` pairs handed to the rolling writer in ONE `write` call.
///
/// WHY 256: [`RollingFileWriter::should_roll`](crate::writer::file_writer::rolling_writer) is
/// evaluated once per `write`, so the feed granularity IS the roll granularity. A single
/// whole-bin batch (what this action used to write) can never roll; 256 pairs bounds the batch so
/// the check runs many times per bin while keeping the per-batch Arrow allocation trivial. It is a
/// COUNT cap layered on top of the BYTE cap below — whichever binds first wins.
const CHUNK_PAIRS: usize = 256;

/// FORK-AUTHORED — **no Java analogue** (as [`CHUNK_PAIRS`]). The absolute ceiling on one chunk's
/// MEASURED serialized size, `sum(file_path.len() + 8)` over the chunk.
///
/// WHY 16384: [`RollingFileWriter::should_roll`] is a PRE-check, so a rolled file has already
/// exceeded the bound by at most one chunk's contribution. That overshoot must fit inside the
/// candidate-filter headroom `max_file_size_bytes - write_max_file_size` (26843546 at the delete
/// defaults) or a run-1 output would be re-admitted by `too_much_content` forever. 16 KiB sits
/// three orders of magnitude below that headroom, which is the margin the convergence argument
/// wants; it is not a Java number and must never be cited as one.
const CHUNK_MAX_SERIALIZED_BYTES: u64 = 16384;

/// FORK-AUTHORED — **no Java analogue** (as [`CHUNK_PAIRS`]). The divisor that reserves HALF the
/// candidate-filter headroom for the Parquet FOOTER.
///
/// WHY 2: `should_roll` reads `current_written_size()` = `bytes_written() + in_progress_size()`
/// (`parquet_writer.rs`), which EXCLUDES the footer — and this action inflates the footer by
/// writing FULL untruncated `file_path` bounds (`MetricsConfig::for_position_delete`). The final
/// `file_size_in_bytes` is therefore the roll-time size PLUS a footer, so the headroom must cover
/// chunk AND footer. Reserving half for each gives `write_max + 2 * chunk_budget <= max_file_size`,
/// which is why the runtime pin's overshoot ceiling is `2 * chunk_budget`.
const CHUNK_HEADROOM_FOOTER_SHARE: u64 = 2;

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
    /// Java `SizeBasedFileRewritePlanner.writeMaxFileSize()` — the bound the ROLLING WRITER rolls
    /// at, which is NOT the resolved target. Derived, never user-settable (Java has no option for
    /// it either).
    write_max_file_size: u64,
    /// FORK-AUTHORED (see [`CHUNK_MAX_SERIALIZED_BYTES`]) — the per-chunk MEASURED-serialized-byte
    /// cap for the writer feed. Derived from the candidate-filter headroom; never user-settable.
    chunk_budget: u64,
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

    /// Run the compaction: plan the live (filter-matching) parquet position-delete files of the
    /// current snapshot through Java's six-stage pipeline (see the module docs), then for every
    /// ADMITTED BIN read its `(file_path, pos)` pairs, concat + sort them, write the compacted
    /// position-delete file(s), and commit the swap in one `Replace` snapshot per bin. Returns the
    /// [`RewritePositionDeleteFilesResult`] counts.
    ///
    /// Commits NOTHING and returns zero counts when there is no current snapshot, no live parquet
    /// position-delete files, none match the filter, no partition yields a CANDIDATE, or every bin
    /// is declined by the three-clause admission gate.
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
        let config = self.resolve_config()?;

        let metadata = self.table.metadata();
        let Some(snapshot) = metadata.current_snapshot().cloned() else {
            return Ok(RewritePositionDeleteFilesResult::default());
        };
        let starting_snapshot_id = snapshot.snapshot_id();

        // S2, BOUND ONCE — after the no-snapshot early return and BEFORE the manifest walk, mirroring
        // Java, which binds the filter at the `PositionDeletes` scan.
        //
        // BEHAVIOUR FLIP (2nd, for the PR body): an UNBINDABLE filter previously errored only when
        // some group happened to hold >= 2 files; hoisted to the scan it errors on ANY table with a
        // current snapshot. That is Java's shape (a filter Java cannot bind fails the scan), and it
        // fails loudly instead of silently compacting nothing.
        //
        // MICRO-RESIDUE, stated not claimed: whether Java also binds on a snapshot-less table is
        // unverified, so this port keeps its pre-binding early return.
        let mut partition_filter = self.bind_filter()?;

        // S1 + S2 + S3 — enumerate the live PARQUET position-delete entries, drop the ones the filter
        // rejects, and group by (spec_id, partition). Puffin DVs are SKIPPED (file-scoped, never
        // bin-packed) — the documented V2-parquet-only scope.
        let groups = self
            .collect_position_delete_groups(&snapshot, &mut partition_filter)
            .await?;

        // S4 + S5 + S6 — candidate filter, bin-pack, group filter.
        let bins = plan_bins(groups, &config);

        // Advance the base table after each bin commit so the next bin's `Transaction` is built on
        // the committed tip (mirrors RewriteDataFiles). Without this, bins 2..N still succeed via
        // `Transaction::do_commit`'s stale-base refresh + re-apply, but each bin pays a full rewrite
        // re-apply against the refreshed tip. Advancing avoids that redundant re-apply work; it is
        // not required for CAS correctness under the retry/refresh loop.
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
    ///
    /// # Derived (not options, on either engine)
    ///
    /// Two more fields are computed AFTER the preconditions, because both need `target < max`:
    /// [`ResolvedConfig::write_max_file_size`] (Java's `writeMaxFileSize()`, the ROLLING WRITER's
    /// bound) and [`ResolvedConfig::chunk_budget`] (fork-authored, the writer feed's per-chunk byte
    /// cap). Neither is user-settable and neither rejects a config.
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

        // ---- Derived, AFTER every precondition (both derivations below need `target < max`) ----

        // Java `writeMaxFileSize()`:
        //   `target + (max_file_size - target) * 0.5`
        // with the subtraction a LONG `lsub` performed BEFORE the `l2d` — hence the `u64`
        // subtraction here, then the widening, then the same `d2l` clamp `sizeThresholds` uses. The
        // subtraction cannot underflow: precondition (4) has just proven `target < max`.
        //
        // This is the bound the ROLLING WRITER rolls at, deliberately NOT the resolved target. It is
        // what makes the fixed point STRUCTURAL rather than argued: wherever the doubles are exact,
        // `write_max < max_file_size`, so a run-1 output is never re-admitted by `too_much_content`.
        // (NOT universal — at `target = 2^62 + 513, max = target + 1` the `l2d` rounds the target up
        // and `d2l` returns a value ABOVE max. Java behaves identically, so that bounds the claim
        // rather than naming a divergence, and the `saturating_sub` below keeps the headroom sane.)
        //
        // SIBLING DIVERGENCE, recorded not hidden: `RewriteDataFiles` rolls at its resolved TARGET
        // (its own named deviation), so until that template follows, the two ports roll at different
        // bounds.
        let write_max_file_size =
            d2l(target as f64 + (max_file_size_bytes - target) as f64 * WRITE_MAX_FILE_SIZE_RATIO);

        // The candidate-filter HEADROOM: how far a run-1 output may exceed `write_max` and still
        // land inside `[min, max]`, where `outsideDesiredFileSizeRange` declines it forever.
        let headroom = max_file_size_bytes.saturating_sub(write_max_file_size);
        let chunk_budget = CHUNK_MAX_SERIALIZED_BYTES.min(headroom / CHUNK_HEADROOM_FOOTER_SHARE);

        // INTENT DOCUMENTATION WITH A TRIPWIRE — and nothing more. This assert is TRIVIALLY TRUE by
        // construction (`chunk_budget = min(_, headroom / 2) <= headroom`), so it does NOT establish
        // the runtime clearance and must not be described as doing so: `chunk_serialized_bytes` is a
        // per-chunk RUNTIME quantity this function cannot see, and the footer is not visible here at
        // all. The RUNTIME clearance is established by a MEASURED-OUTPUT pin
        // (`test_no_split_output_exceeds_max_file_size`), never by this line. What the assert buys is
        // a loud red if a future editor changes `CHUNK_HEADROOM_FOOTER_SHARE` or rewrites the
        // derivation in a way that stops respecting the headroom.
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

    /// Stages S1 - S3: walk the current snapshot's manifests ONCE, collect the live PARQUET
    /// position-delete entries that pass the user `filter`, and group them by `(spec_id, partition)`.
    /// Puffin deletion vectors (`format == Puffin`) and equality/data entries are EXCLUDED.
    ///
    /// The filter is applied PER ENTRY here (S2), not per group, because Java applies it at the
    /// `PositionDeletes` scan — strictly before `groupByPartition`. The selection is identical either
    /// way (every entry in a group shares the partition and the projected predicate is
    /// partition-only); what changes is WHEN an unbindable filter errors (see [`Self::execute`]).
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
                // Only PARQUET position deletes. Skip data, equality deletes, and Puffin DVs.
                if data_file.content_type() != DataContentType::PositionDeletes {
                    continue;
                }
                if data_file.file_format() != DataFileFormat::Parquet {
                    // A Puffin DELETION VECTOR — file-scoped, never bin-packed by this action.
                    continue;
                }
                // S2 — the user filter, at collection (Java's scan-level `filter`).
                if !partition_filter.matches(metadata, data_file)? {
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

    /// Stage S2's ONE bind: bind [`Self::filter`] to the table schema, once, before the manifest
    /// walk. `AlwaysTrue` (the default) matches everything and is never bound.
    ///
    /// This is where an unbindable filter now fails — see [`Self::execute`]'s behaviour-flip note.
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

    /// Compact ONE ADMITTED BIN of a `(spec, partition)` group: read every member file's
    /// `(file_path, pos)` pairs, concat + sort, write the compacted position-delete file(s), and
    /// commit ONE `RewriteFiles` that replaces the rewritten files with ALL of them, each stamped
    /// with the bin MAX rewritten data-seq and validated from the starting snapshot. Accumulates the
    /// four `Result` counts.
    ///
    /// The bin may produce MORE THAN ONE output file: the rolling writer rolls at
    /// [`ResolvedConfig::write_max_file_size`], so a bin larger than that bound splits. Every output
    /// of THIS bin carries THIS bin's own max — Java's
    /// `RewritePositionDeletesGroup.maxRewrittenDataSequenceNumber` ranges the max over that GROUP's
    /// task list, and one Java group IS one bin, so ranging over the whole partition (or reusing a
    /// previous bin's max) is a stamping error, not a rounding one.
    ///
    /// Returns the committed [`Table`] so the caller can advance the base for the next group
    /// (mirrors [`crate::maintenance::rewrite_data_files::RewriteDataFiles`]).
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
        //
        // GLOBAL SORT, BEFORE ANY SPLIT — load-bearing, and the one line an editor can silently
        // break. `write_compacted_file` chunks THIS already-sorted `Vec` in order, so the bin's N
        // outputs carry DISJOINT ASCENDING `(file_path, pos)` ranges whose union is exactly this
        // multiset. Sorting per chunk instead, or chunking before this call, still writes every pair
        // — so the masked row set survives — but destroys the global ordering the spec recommends
        // and the delete-file range pruning depends on. Do not move this below the split.
        pairs.sort();

        // (3) Write the compacted position-delete file(s) under the group spec + partition key. The
        // rolling writer rolls at `write_max_file_size`, so this is ONE file for a bin below that
        // bound and N files for a bin above it.
        let new_files = self
            .write_compacted_file(table, key, &pairs, config)
            .await?;

        // (4) STALLER — THIS BIN's MAX rewritten data sequence number, ranged over `entries`, which
        // IS the bin (Java `RewritePositionDeletesGroup.<init>` maxes over that group's task list,
        // and one group is one bin). Ranging over the whole partition, or carrying a previous bin's
        // value, is a stamping error. Stamping the MAX of the rewritten bin preserves exactly which
        // data generation the compacted delete masks.
        //
        // DIRECTION OF DANGER, against the fork's OWN rule — `delete_file_index.rs`'s
        // `applicable_pos_deletes` keeps a delete whose `delete_seq >= data_seq`. So an OVER-HIGH
        // (e.g. inherited) stamp reaches data it never masked and OVER-APPLIES; an UNDER-LOW stamp
        // stops applying and RESURRECTS deleted rows. Carrying a previous bin's max is the UNDER
        // direction whenever that bin's max is the lower one, which is why C-010's ranging pin
        // asserts the two bins' maxima DIFFER before it asserts either stamp.
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

        // Accumulate the byte counts (Java `rewrittenBytesCount` / `addedBytesCount`). The added
        // side is a CHECKED sum across the N split outputs — the guard the single-output form
        // carried, retained at the new arity rather than replaced by a plain `sum()`.
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

        // (5) Commit ONE RewriteFiles per BIN: REPLACE the rewritten pos-deletes with ALL of this
        // bin's outputs, EACH stamped with THIS bin's max rewritten data-seq via
        // `add_delete_file_with_sequence_number` (NOT the default-inherit add), validating from the
        // starting snapshot (Java `newRewrite().validateFromSnapshot(J).deleteFile(rewritten)
        // .addFile(added, J).commit()`).
        //
        // `add_delete_file_with_sequence_number` PUSHES onto the action's added list, so N chained
        // calls stamp N files with the one seq — no new transaction API is needed for the fan-out.
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

    /// Write the GLOBALLY SORTED `(file_path, pos)` pairs into the bin's compacted position-delete
    /// file(s) under the group's spec + partition key, returning every resulting [`DataFile`].
    ///
    /// # One writer, many chunks
    ///
    /// `pairs` is fed to ONE [`RollingFileWriter`](crate::writer::file_writer::rolling_writer) in
    /// bounded chunks — at most [`CHUNK_PAIRS`] pairs AND at most `config.chunk_budget` MEASURED
    /// serialized bytes, with a floor of ONE pair so an absurdly tight budget still terminates. The
    /// chunking exists because `should_roll` is evaluated once per `write`: the single whole-bin
    /// batch this action used to write could never roll, whatever bound it was given.
    ///
    /// The writer's bound is `config.write_max_file_size` (Java `writeMaxFileSize()`), passed
    /// EXPLICITLY — `RollingFileWriterBuilder::new_with_default_file_size` hard-wires the 512 MiB
    /// **data** default, which is eight times the delete target and unrelated to this action.
    ///
    /// # The split preserves the global order
    ///
    /// The chunks are contiguous slices of the already-sorted `pairs`, taken front to back, so
    /// output *k*'s `(file_path, pos)` range is entirely below output *k+1*'s and the union of the
    /// outputs is exactly the input multiset — the masked row set is unchanged by the split.
    ///
    /// # Stated assumption (measured by the pins, not assumed away)
    ///
    /// `chunk_budget` counts RAW INPUT bytes (`file_path.len() + 8` per pair) while `should_roll`
    /// measures OUTPUT parquet bytes. The overshoot bound therefore rests on "one chunk's parquet
    /// contribution <= its raw bytes", which holds for a sorted dictionary/RLE-encoded path column
    /// plus an int64 `pos` column. It is MEASURED by `test_no_split_output_exceeds_max_file_size`
    /// rather than argued here.
    ///
    /// # RESIDUE (RES-8), both halves
    ///
    /// **Correctness half:** on an absurdly tight `[write_max, max]` band the one-pair floor, or a
    /// footer larger than its reserved half, can still overshoot — the same best-effort class of
    /// bound Java carries.
    ///
    /// **Throughput half:** `chunk_budget` resolves to `0` for any config with
    /// `max_file_size_bytes - target_file_size_bytes <= 2`, and to `1` at `3` — in every such band
    /// it is below one pair's serialized size, so the one-pair floor governs and the feed degrades
    /// to ONE Arrow batch (and one `should_roll` evaluation) PER PAIR. Correct, and arbitrarily
    /// slow. Both bands are legal under `resolve_config`'s preconditions, so this is reachable
    /// configuration rather than a theoretical edge; it is named rather than defended against,
    /// because clamping the budget upward would trade a throughput cliff for a correctness one.
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
        // Position-delete files keep `file_path`/`pos` bounds FULL (Java `MetricsConfig.forPositionDelete`)
        // so delete-file path pruning stays precise — the default `truncate(16)` would widen the path range.
        let parquet_builder = ParquetWriterBuilder::new(
            position_delete_writer_properties(),
            writer_config.schema().clone(),
        )
        .with_metrics_config(MetricsConfig::for_position_delete());
        // Java `writeMaxFileSize()`, NOT the resolved target — see `ResolvedConfig`. The builder
        // takes a `usize`, so a `write_max` above `usize::MAX` (only reachable on a 32-bit target)
        // saturates to "never roll", which is exactly what the default constructor would have done.
        let rolling = RollingFileWriterBuilder::new(
            parquet_builder,
            usize::try_from(config.write_max_file_size).unwrap_or(usize::MAX),
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );

        // The new pos-delete must live in the SAME partition + spec as the files it replaces (so it
        // lands in the same bucket and applies to the same data files). Always pass a PartitionKey —
        // including empty/unpartitioned and all-Void null tuples — so we never fabricate spec_id 0
        // via `build(None)` without `with_partition_spec` (C2-L-001 / C1-L-001 class).
        let partition_key = PartitionKey::new(spec, schema.clone(), partition.clone())?;
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, writer_config.clone())
            .build(Some(partition_key))
            .await?;

        // THE BOUNDED CHUNK FEED. `pairs` is already globally sorted (see `compact_group`); each
        // chunk is the NEXT contiguous slice, so the split preserves that order.
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
}

/// One `(file_path, pos)` pair's MEASURED serialized size: the UTF-8 length of the path plus 8 for
/// the `int64` position. This is the RAW INPUT measure `chunk_budget` is denominated in — see
/// [`RewritePositionDeleteFiles::write_compacted_file`]'s stated assumption.
fn pair_serialized_bytes(pair: &(String, i64)) -> u64 {
    u64::try_from(pair.0.len())
        .unwrap_or(u64::MAX)
        .saturating_add(8)
}

/// The exclusive end of the chunk starting at `start`: at most [`CHUNK_PAIRS`] pairs, at most
/// `chunk_budget` [`pair_serialized_bytes`], and ALWAYS at least one pair.
///
/// The one-pair floor is what keeps the feed loop total — `chunk_budget` can legitimately resolve to
/// `0` (a `[write_max, max]` band narrower than two bytes), and a chunk of zero pairs would spin
/// forever. It is also the residue RES-8 names: a single pair whose path is longer than the whole
/// budget overshoots the bound by construction, on any config where such a band is configured.
fn chunk_end(pairs: &[(String, i64)], start: usize, chunk_budget: u64) -> usize {
    let limit = pairs.len().min(start.saturating_add(CHUNK_PAIRS));
    let mut end = start;
    let mut bytes: u64 = 0;
    while end < limit {
        let next = bytes.saturating_add(pair_serialized_bytes(&pairs[end]));
        // `end > start` IS the one-pair floor: the first pair of a chunk is taken unconditionally.
        if end > start && next > chunk_budget {
            break;
        }
        bytes = next;
        end += 1;
    }
    end
}

/// The FAIL-CLOSED guard on the writer's output, extracted so it survives — and stays pinnable — at
/// the `Vec<DataFile>` arity.
///
/// A bin reaching here has NON-EMPTY pairs, so it must produce at least one file. "No file" is a
/// NORMAL return from the parquet writer, not an error: `ParquetWriter::close` returns `Ok(vec![])`
/// and DELETES the output whenever `current_row_num == 0`. Without this check `execute` would go on
/// to commit a `Replace` snapshot that removes live position-delete files and adds none — silent
/// UNDER-masking — and nothing downstream would reject it, because `RewriteFilesAction::validate`
/// early-returns when `deleted_data_files` is empty, which is always true for this action.
fn require_non_empty(files: Vec<DataFile>) -> Result<Vec<DataFile>> {
    if files.is_empty() {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "Position-delete writer produced no file for a non-empty input",
        ));
    }
    Ok(files)
}

/// Stage S2's evaluator: the user `filter`, bound ONCE to the table schema, plus the projected +
/// bound partition evaluator CACHED PER `spec_id`. Binding is therefore O(specs), not O(entries).
///
/// `bound_row_filter == None` is the [`Predicate::AlwaysTrue`] default — it matches everything and
/// never binds, so a table whose filter is unset does no projection work at all.
struct PartitionFilter {
    bound_row_filter: Option<BoundPredicate>,
    by_spec_id: HashMap<i32, ExpressionEvaluator>,
}

impl PartitionFilter {
    /// The unfiltered default: every entry matches, nothing is bound.
    fn always_true() -> Self {
        Self {
            bound_row_filter: None,
            by_spec_id: HashMap::new(),
        }
    }

    /// A real filter, already bound to the table schema by
    /// [`RewritePositionDeleteFiles::bind_filter`].
    fn bound(bound_row_filter: BoundPredicate) -> Self {
        Self {
            bound_row_filter: Some(bound_row_filter),
            by_spec_id: HashMap::new(),
        }
    }

    /// Whether `data_file`'s PARTITION matches the filter — the SAME partition-pruning path the
    /// table scan uses (inclusive projection onto the file's spec, then evaluation against the
    /// file's partition struct). The per-spec evaluator is built at most once.
    fn matches(&mut self, metadata: &TableMetadata, data_file: &DataFile) -> Result<bool> {
        // Split the borrow so the cache can be filled while the bound predicate is read.
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

/// Project the bound row filter onto partition spec `spec_id` and bind it to that spec's partition
/// schema, yielding the evaluator [`PartitionFilter`] caches.
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

/// Stages S4 - S6 of the planner (Java `BinPackRewritePositionDeletePlanner.planFileGroups`, reached
/// per partition): candidate-filter each group, bin-pack the survivors, and keep the bins the
/// three-clause group filter admits. Returns each admitted bin alongside its group key.
///
/// Java's order is `filterFiles` → `ListPacker.pack` → `filterFileGroups`, and it is load-bearing:
/// packing first would let an in-range file consume bin headroom and split the candidates that
/// belong together.
fn plan_bins(
    groups: HashMap<GroupKey, Vec<LiveDeleteEntry>>,
    config: &ResolvedConfig,
) -> Vec<AdmittedBin> {
    let mut admitted: Vec<AdmittedBin> = Vec::new();

    for (key, entries) in groups {
        // S4 — the candidate filter.
        let candidates: Vec<LiveDeleteEntry> = entries
            .into_iter()
            .filter(|entry| is_candidate(entry, config))
            .collect();
        // A partition with no candidate contributes no bin. The packer already returns an empty
        // `Vec` for an empty input, so this is a cheap short-circuit rather than a behaviour: the
        // branch is deliberately UNKILLABLE by mutation and is not carried as covered anywhere.
        if candidates.is_empty() {
            continue;
        }

        // S5 — bin-pack through the shared packer (Java's inherited `ListPacker`).
        let bins = pack_bins(
            candidates,
            |entry| entry.data_file.file_size_in_bytes,
            config.max_file_group_size_bytes,
        );

        // S6 — the three-clause group filter.
        for bin in bins {
            if group_qualifies(&bin, config) {
                admitted.push((key.clone(), bin));
            }
        }
    }

    admitted
}

/// Java `SizeBasedFileRewritePlanner.filterFiles` → `outsideDesiredFileSizeRange`: a position-delete
/// file is a CANDIDATE iff it is undersized or oversized, both comparisons STRICT.
///
/// There is deliberately NO delete-count clause: `tooManyDeletes` / `tooHighDeleteRatio` live on
/// `BinPackRewriteFilePlanner` (the DATA-file planner), and `BinPackRewritePositionDeletePlanner`
/// does not inherit them — its `filterFiles` is `Iterables.filter(tasks, outsideDesiredFileSizeRange)`
/// and nothing else.
fn is_candidate(entry: &LiveDeleteEntry, config: &ResolvedConfig) -> bool {
    let length = entry.data_file.file_size_in_bytes;
    length < config.min_file_size_bytes || length > config.max_file_size_bytes
}

/// Java `SizeBasedFileRewritePlanner.filterFileGroups` — a plain three-way disjunction over one bin,
/// with NO fourth clause (the template's `any_too_many_deletes` disjunct is
/// `BinPackRewriteFilePlanner`-only and must not appear here):
///
/// - `enough_input_files` = `size > 1 && size >= min_input_files`
/// - `enough_content`     = `size > 1 && input_size > target_file_size_bytes`
/// - `too_much_content`   = `input_size > max_file_size_bytes` — **no `size > 1` guard**, which is
///   what admits a LONE oversized position-delete file.
///
/// Every size comparison is STRICT; `>=` appears only on `min_input_files`.
///
/// TWO leaf sub-expressions here are unreachable END TO END, so no end-to-end fixture can kill their
/// mutants. (a) `enough_content`'s `size > 1`: a bin of one that reaches this gate is a candidate, so
/// its length is either below `min` — and `min < target`, so `enough_content` is false either way —
/// or above `max`, in which case `too_much_content` is already true. (b) `too_much_content`'s
/// boundary STRICTNESS: a bin at `input_size == max` is either of size 1, whose file then has
/// `length == max` and is not a candidate at all, or of size >= 2, and then `enough_content` is
/// already true because `max > target`. Both proofs use only `min < target < max`, which
/// `resolve_config`'s preconditions (3) and (4) guarantee.
///
/// UNREACHABLE IS NOT UNPINNED. Both rest on candidate-filter REACHABILITY, not on the config space,
/// so both states are constructible through the WHITE-BOX seam the test module already opens on this
/// function, and both mutants ARE killed there — see
/// `test_gate_enough_content_size_guard_declines_lone_over_target_file_white_box` and
/// `test_gate_input_size_exactly_max_is_declined_white_box`. Do not delete either leaf as dead code.
///
/// The input-size sum SATURATES (template form) where Java's `mapToLong(..).sum()` WRAPS on `long`
/// overflow. A wrapped negative sum makes both size clauses false, so Java would DECLINE where this
/// port ADMITS. Unreachable in practice — it needs more than 8 EiB of live delete files in ONE bin —
/// and the input is manifest-trusted, so saturating is the safer of the two; the divergence is
/// recorded rather than assumed.
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
