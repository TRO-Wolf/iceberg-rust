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

//! `RewritePositionDeleteFiles`: the maintenance action that compacts the live PARQUET
//! position-delete files of the current snapshot. It bin-packs per `(spec, partition)` group and
//! commits one `Replace` snapshot per admitted bin. Java
//! `org.apache.iceberg.actions.RewritePositionDeleteFiles`.
//!
//! # The Java contract this mirrors
//!
//! Java's `Result` derives its four counts from one abstract `rewriteResults()` list.
//! [`RewritePositionDeleteFilesResult`] carries the four counts; the per-group list is deferred.
//! The planning and commit machinery is engine-agnostic iceberg-core. The read, sort and write
//! materialization is a Spark-surface action, so the pipeline below is built engine-agnostically,
//! as [`ConvertEqualityDeleteFiles`] was. `RewritePositionDeleteFiles` is one of the twelve Java
//! `ActionsProvider` methods, so it is wired into the
//! [`ActionsProvider`](crate::maintenance::ActionsProvider) factory.
//!
//! # The compaction: fuse in one direction, split in the other
//!
//! Compaction reads the `(file_path, pos)` pairs out of every admitted delete file in a group and
//! writes files that mask EXACTLY the same rows. A merge-on-read scan returns an identical live
//! row set before and after. The file count moves in both directions:
//!
//! - **FUSE**: a bin of many small files is rewritten into FEWER files.
//! - **SPLIT**: a bin whose single file exceeds `max_file_size_bytes` is admitted ALONE, because
//!   `too_much_content` carries no `size > 1` guard, and splits as the rolling writer rolls.
//!
//! Each admitted bin reads its members' reserved `file_path` and `pos` columns by FIELD ID, sorts
//! the pairs, writes them under the group's spec and partition key, and commits one
//! [`RewriteFilesAction`](crate::transaction::rewrite_files) that validates from the starting
//! snapshot. Java does not dedup within a group, because the reader bitmap dedups.
//!
//! # The silent-corruption staller: SEQ STAMPING
//!
//! Every file a bin adds MUST carry THAT BIN's max rewritten data sequence number. Not the
//! inherited seq, not the min, and not another bin's max. `applicable_pos_deletes` keeps a
//! position delete whose `delete_seq >= data_seq`, so the max of the rewritten bin preserves
//! exactly which data generation the compacted delete masks.
//!
//! Direction of danger, read off that same rule. An OVER-HIGH stamp reaches data the rewritten
//! deletes never masked and deletes rows it must not. An UNDER-LOW stamp stops applying to the
//! bin's own higher-seq data and resurrects rows it must delete.
//!
//! # Planning: Java's six-stage pipeline
//!
//! 1. **Collect** the live `PositionDeletes` entries, skipping every non-Parquet delete file.
//! 2. **User filter**, applied PER ENTRY at collection, because Java applies it at the scan,
//!    strictly before grouping.
//! 3. **Group** by `(spec_id, partition)`.
//! 4. **Candidate filter**: undersized or oversized, both strict. No delete-count clause.
//! 5. **Bin-pack** through the shared [`pack_bins`](super::rewrite_data_files::pack_bins).
//! 6. **Group filter**: `enough_input_files || enough_content || too_much_content`.
//!
//! [`RewritePositionDeleteFiles::execute`] then iterates BINS, not partitions.
//!
//! # Named non-port: Java's `inputSplitSize` / `expectedOutputFiles`
//!
//! Java consumes `inputSplitSize` on the READ side, as a scan option. The write bound is a
//! separate option. Java never chunks a writer feed with it. This action reads the pairs directly
//! and has no split-size-driven scan, so a port would be dead code.
//!
//! GUARD ON THE PIN. The tripwire for this non-port is a repo-wide grep for the SNAKE_CASE Rust
//! spellings of those two Java names, which must return ZERO hits. It is snake-case only BY
//! DESIGN, because the camelCase Java names must appear: this paragraph names them, and
//! [`super::rewrite_data_files`] cites `inputSplitSize` load-bearingly. Broadening the pattern to
//! camelCase would classify both correct sentences as violations.
//!
//! # Format-version dispatch: V1/V2 compact, V3 converts to deletion vectors
//!
//! A V3 table cannot hold a FRESH parquet position delete, so compacting one into another is
//! unreachable. The V3 arm converts every legacy parquet position delete into one Puffin DV per
//! referenced data file, merged with that file's existing DV. See
//! [`RewritePositionDeleteFiles::rewrite_to_deletion_vectors`]. V2 ORC and Avro position deletes
//! stay skipped on V1/V2 and are refused on V3.
//!
//! # No-op: zeros mean "looked, found nothing to do"
//!
//! On the V3 arm that reading is total WITHIN THE FILTER'S SCOPE. An input the arm cannot express
//! returns `Err` if the filter admits it, or if a vector this run writes would shadow it. An
//! unreadable delete the filter rejects, touching nothing this run converts, is still skipped
//! silently. The V1/V2 arm keeps its weaker contract: it returns zeros for a V2 ORC or Avro
//! position delete it skipped.
//!
//! With no current snapshot, no live parquet position deletes, a
//! [`filter`](RewritePositionDeleteFiles::filter) that matches none, no candidate in a partition,
//! or a bin the three-clause gate declines, the action commits nothing for that bin and the counts
//! stay zero. A bin of ONE file is declined by the two `size > 1` guards, unless that file is
//! larger than `max_file_size_bytes`.

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

/// One ADMITTED BIN as [`plan_bins`] emits it: its group key and its member entries. Kept as one
/// value so `compact_group` takes the bin as a UNIT. The seq stamp is defined over exactly this
/// entry set.
type AdmittedBin = (GroupKey, Vec<LiveDeleteEntry>);

// ONE HOME for the four Java planner constants, imported from the sibling template rather than
// duplicated here. `BinPackRewritePositionDeletePlanner` inherits all four unchanged from
// `SizeBasedFileRewritePlanner`, so a single home keeps the two ports from drifting apart.

/// [`TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT`] in the `i64` domain the
/// parse works in (see [`parse_delete_target_file_size`]). The `as` is proven lossless by the
/// const-evaluated assertion, so the cast can never truncate.
const DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT_I64: i64 = {
    let default = TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES_DEFAULT;
    assert!(default <= i64::MAX as u64);
    default as i64
};

/// Java's `SizeBasedFileRewritePlanner.writeMaxFileSize()` band ratio: the `(max_file_size -
/// target)` band is scaled by this and added back to the target. Parity-cited, not fork-authored.
const WRITE_MAX_FILE_SIZE_RATIO: f64 = 0.5;

/// FORK-AUTHORED — **no Java analogue** (see the module rustdoc's named non-port: Java applies
/// `inputSplitSize` on the READ side and never chunks a writer feed). The maximum number of
/// `(file_path, pos)` pairs handed to the rolling writer in ONE `write` call.
///
/// WHY 256: [`RollingFileWriter::should_roll`](crate::writer::file_writer::rolling_writer) runs
/// once per `write`, so the feed granularity IS the roll granularity. A single whole-bin batch can
/// never roll. 256 pairs makes the check run many times per bin. It is a COUNT cap layered on the
/// BYTE cap below; whichever binds first wins.
const CHUNK_PAIRS: usize = 256;

/// FORK-AUTHORED — **no Java analogue** (as [`CHUNK_PAIRS`]). The absolute ceiling on one chunk's
/// MEASURED serialized size, `sum(file_path.len() + 8)` over the chunk.
///
/// WHY 16384: [`RollingFileWriter::should_roll`] is a PRE-check, so a rolled file overshoots the
/// bound by at most one chunk. That overshoot must fit inside the candidate-filter headroom
/// `max_file_size_bytes - write_max_file_size`, or `too_much_content` re-admits a run-1 output
/// forever. 16 KiB sits far below that headroom. It is not a Java number.
const CHUNK_MAX_SERIALIZED_BYTES: u64 = 16384;

/// FORK-AUTHORED — **no Java analogue** (as [`CHUNK_PAIRS`]). The divisor that reserves HALF the
/// candidate-filter headroom for the Parquet FOOTER.
///
/// WHY 2: `should_roll` reads `current_written_size()`, which EXCLUDES the footer, and this action
/// inflates the footer by writing FULL untruncated `file_path` bounds. The final
/// `file_size_in_bytes` is the roll-time size PLUS a footer, so the headroom must cover chunk AND
/// footer. Half each gives `write_max + 2 * chunk_budget <= max_file_size`.
const CHUNK_HEADROOM_FOOTER_SHARE: u64 = 2;

/// Java's `d2l` on the ratio products in `SizeBasedFileRewritePlanner.sizeThresholds`. The JVM
/// saturates at `Long.MAX_VALUE`.
///
/// Rust's `as u64` saturates at `u64::MAX` instead, so the `.min(i64::MAX as u64)` IS the parity
/// act. Without it a target above `2^63 / 1.8` resolves a `max_file_size_bytes` Java can never
/// produce. Negative and NaN inputs map to `0` here where Java gives a negative. Precondition (1)
/// of [`RewritePositionDeleteFiles::resolve_config`] makes that divergence unreachable.
///
/// NAMED RESIDUE (RES-9): the sibling [`super::rewrite_data_files`] resolver stays UNCLAMPED at its
/// own ratio-default site. This action does not change the template.
fn d2l(x: f64) -> u64 {
    (x as u64).min(i64::MAX as u64)
}

/// Parse the `write.delete.target-file-size-bytes` table property — Java
/// `BinPackRewritePositionDeletePlanner.defaultTargetFileSize()` via `PropertyUtil.propertyAsLong`,
/// which is `map.get(key)`, `null -> default`, else `Long.parseLong`.
///
/// Parses into **`i64`, not `u64`, on purpose**. `i64::from_str`'s accept and reject domain
/// coincides with `Long.parseLong`'s. ONE stated exception, fail-closed: `Long.parseLong` also
/// accepts non-ASCII Unicode decimal digits, where Rust is ASCII-only. The fork rejects those
/// loudly rather than resolving a different threshold.
///
/// So `"0"` and `"-1"` parse here exactly as in Java, and
/// [`RewritePositionDeleteFiles::resolve_config`]'s `target > 0` precondition then rejects them
/// with Java's verbatim message. A `u64` parse would reject `"-1"` with a fork-only message, and
/// would admit values above `i64::MAX` that `Long.parseLong` throws on.
///
/// NAMED RESIDUE (RES-10): the sibling `rewrite_data_files::parse_target_file_size` parses `u64`
/// and stays as-is.
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

/// The rejection for a builder override outside Java's `long` domain, precondition (7) of
/// [`RewritePositionDeleteFiles::resolve_config`]. A `u64` above `i64::MAX` is a config Java cannot
/// express. The fork refuses it rather than accepting a state where `too_much_content` is
/// unreachable.
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

/// The resolved size and count thresholds for one [`RewritePositionDeleteFiles::execute`] run.
/// [`RewritePositionDeleteFiles::resolve_config`] is the ONLY home for the defaults and Java's
/// preconditions.
///
/// Keep the `Clone`/`PartialEq`/`Eq` derives. They are what keeps `dead_code` disarmed here.
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
/// [`Self::min_input_files`] and [`Self::max_file_group_size_bytes`]. Defaults are Java's: the
/// target reads `write.delete.target-file-size-bytes` (64 MiB), NOT the 512 MiB data property; min
/// and max resolve lazily to `0.75 *` and `1.8 *` the target; `min_input_files` is 5; the group
/// size is 100 GiB.
///
/// # Deferred (loudly)
///
/// Java's option domain here is ELEVEN keys. Five are ported above. The other six are deferred by
/// name:
///
/// - **`rewrite-all`**: not exposed, and NOT emulable through the ported knobs. **The emulation is
///   INVERTED, so do not attempt it.** `rewriteAll` bypasses both filters while keeping the
///   packing. Reaching for `min_file_size_bytes(0)` plus a huge `max_file_size_bytes` instead makes
///   `is_candidate` false for EVERY file, which empties the candidate set and admits NOTHING.
/// - **`rewrite-job-order`**: bins are planned and committed in plan order. There is no cost or
///   size ordering of the rewrite jobs.
/// - **`partial-progress.enabled`**: each admitted bin commits in its OWN `RewriteFiles`
///   transaction, sequentially. See [`Self::execute`]'s abort contract.
/// - **`partial-progress.max-commits`**: meaningless without the flag above.
/// - **`max-concurrent-file-group-rewrites`**: the bin sweep is SEQUENTIAL and there is no
///   executor seam.
/// - **`output-spec-id`**: not exposed. Each bin writes under the spec of its own group
///   (`GroupKey.0`). Java's pos-delete write path resolves the output spec from the FIRST rewritten
///   delete file's `specId()` and never consults `outputSpecId()`. Cited from a firsthand Spark
///   read, not re-derived here. Were that false, this action would diverge from Java on
///   spec-evolved tables.
///
/// Deferred alongside them, and NOT an option key:
///
/// - **The per-group `Result` list.** [`RewritePositionDeleteFilesResult`] carries only the four
///   aggregate counts. A partition can yield SEVERAL bins, so the per-group list is no longer
///   inferable from the aggregates.
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
    /// Create a `RewritePositionDeleteFiles` action for `table` with Java's defaults. With no
    /// [`filter`](Self::filter), every group of live parquet position deletes in the current
    /// snapshot is considered. The size thresholds resolve lazily at [`Self::execute`].
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

    /// Restrict the compaction to position-delete files whose partition matches `filter`. Java
    /// `RewritePositionDeleteFiles.filter`. The predicate binds to the table schema, projects onto
    /// each delete file's spec, and evaluates against its PARTITION values. This is the same
    /// partition-pruning path the table scan uses. The default compacts all.
    pub fn filter(mut self, filter: Predicate) -> Self {
        self.filter = filter;
        self
    }

    /// Run the compaction through Java's six-stage pipeline, then commit one `Replace` snapshot
    /// per admitted bin. See the module docs.
    ///
    /// Commits NOTHING and returns zero counts when there is no current snapshot, no live parquet
    /// position deletes, none match the filter, no partition yields a candidate, or the gate
    /// declines every bin.
    ///
    /// # ONE commit per admitted BIN
    ///
    /// Every admitted bin gets its OWN [`RewriteFiles`](crate::transaction::rewrite_files), so `B`
    /// bins produce `B` snapshots. Bins replace pairwise DISJOINT sets of position-delete files, so
    /// committing them in sequence is correct. A FIXED `starting_snapshot_id` across all bins
    /// cannot trip conflict validation, because
    /// [`RewriteFilesAction::validate`](crate::transaction::rewrite_files) early-returns on an
    /// empty `deleted_data_files`, and this action never populates a DATA-file set.
    ///
    /// NOT part of that argument: the loop advances the base table to each committed tip. That is a
    /// cost optimisation only, because `do_commit` refreshes a stale base and re-applies.
    ///
    /// # The abort contract
    ///
    /// The bin loop is SEQUENTIAL and each iteration commits, so failure is NOT atomic across bins.
    /// When any bin fails, every bin committed before it STANDS. **No rollback is attempted**, and
    /// none is possible without a compensating snapshot this action deliberately does not write.
    /// The caller receives no partial [`RewritePositionDeleteFilesResult`].
    ///
    /// A table left mid-loop is CONSISTENT, only less compacted. Each committed bin replaced its
    /// own delete files with an equivalent compacted set, so the masked row set is unchanged.
    /// Re-running the action resumes from there.
    ///
    /// DIVERGENCE: Java's non-partial-progress path commits the whole plan through ONE commit
    /// manager call, so the fork's snapshot COUNT diverges from Java's.
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<RewritePositionDeleteFilesResult> {
        // Resolve and VALIDATE the thresholds before any manifest is read, mirroring Java, where
        // `sizeThresholds` runs at planner `init`.
        //
        // A pre-existing table whose `write.delete.target-file-size-bytes` is unparsable, above
        // `i64::MAX`, `<= 1`, or `== i64::MAX` now makes `execute` return `Err`. Java throws on the
        // same inputs, so this is parity-correct, but it IS a behaviour flip.
        let config = self.resolve_config()?;

        let metadata = self.table.metadata();
        let Some(snapshot) = metadata.current_snapshot().cloned() else {
            return Ok(RewritePositionDeleteFilesResult::default());
        };
        let starting_snapshot_id = snapshot.snapshot_id();

        // S2, BOUND ONCE, after the no-snapshot early return and before the manifest walk. Java
        // binds the filter at the `PositionDeletes` scan. An UNBINDABLE filter therefore errors on
        // ANY table with a current snapshot, which is loud rather than silently compacting nothing.
        //
        // MICRO-RESIDUE: whether Java also binds on a snapshot-less table is unverified, so this
        // port keeps its pre-binding early return.
        let mut partition_filter = self.bind_filter()?;

        // FORMAT-VERSION DISPATCH. On V3 a fresh parquet position delete cannot be committed at all
        // (`validate_delete_file_for_version`), so compacting one into another is unreachable. The V3
        // arm converts instead. V1/V2 fall through to the bin-pack pipeline below, unchanged.
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

        // S1 + S2 + S3 — enumerate the live PARQUET position-delete entries, drop the ones the filter
        // rejects, and group by (spec_id, partition). Puffin DVs are SKIPPED (file-scoped, never
        // bin-packed) — the documented V2-parquet-only scope.
        let groups = self
            .collect_position_delete_groups(&snapshot, &mut partition_filter)
            .await?;

        // S4 + S5 + S6 — candidate filter, bin-pack, group filter.
        let bins = plan_bins(groups, &config);

        // Advance the base table after each bin commit, so the next bin builds on the committed
        // tip. Without this the later bins still succeed through `do_commit`'s stale-base refresh,
        // but each pays a full re-apply. This is a cost optimisation, not a CAS requirement.
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
    /// Java's `sizeThresholds` resolves the min and max DEFAULTS first, then checks (1) `target >
    /// 0`, (2) `min >= 0`, (3) `target > min`, (4) `target < max`, all STRICT. The order is
    /// load-bearing. At a negative target [`d2l`] saturates both ratio products to `0`, so (3) is
    /// independently false, and hoisting (3) above (1) would report the wrong message.
    ///
    /// **(2) is STRUCTURALLY UNREACHABLE here and is not coded.** `min_file_size_bytes` is a `u64`
    /// reachable only through the builder, so no caller can express a negative.
    ///
    /// Three preconditions live OUTSIDE Java's `sizeThresholds`. Only (7) is fork-authored:
    ///
    /// - (5) `min_input_files > 0` and (6) `max_file_group_size_bytes > 0` are Java's own
    ///   `checkArgument`s, raised from `init(Map)` rather than `sizeThresholds`. Their message shape
    ///   is Java's and is not ours to reword. Both are checked after (4).
    /// - (7) every EXPLICIT builder override of the three size knobs is `<= i64::MAX`. A larger
    ///   value is a config Java cannot express, and admitting it opens a state where
    ///   `too_much_content` is unreachable.
    ///
    ///   **(7) is checked when each override is READ, before the defaults resolve and before (1).**
    ///   Two reasons, both verified by removing the checks:
    ///
    ///   1. (7) is what makes the `as i64` cast below lossless. Without its target leg a `u64`
    ///      above `i64::MAX` wraps negative, and (1) then reports a value the caller never wrote.
    ///   2. Checked last, only the `max` leg would be observable. A large `min` is caught first by
    ///      (3) and a large `target` by (1), each with the wrong message.
    ///
    ///   Read-time checking is therefore the only placement under which each knob reports its own
    ///   rejection and the narrowing cast is total. Java's (1), (3), (4) order is untouched.
    ///
    /// # Derived (not options, on either engine)
    ///
    /// [`ResolvedConfig::write_max_file_size`] and [`ResolvedConfig::chunk_budget`] are computed
    /// after the preconditions, because both need `target < max`. Neither is user-settable and
    /// neither rejects a config.
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
            // In range by (7) above, so the narrowing cast cannot truncate.
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

        // Java `writeMaxFileSize()` = `target + (max_file_size - target) * 0.5`, with the
        // subtraction done on longs BEFORE the widening. Precondition (4) has just proven
        // `target < max`, so it cannot underflow.
        //
        // This is the bound the ROLLING WRITER rolls at, deliberately NOT the resolved target.
        // Wherever the doubles are exact `write_max < max_file_size`, so `too_much_content` never
        // re-admits a run-1 output. NOT universal: near `2^62` the rounding can put it above max.
        // Java behaves identically there.
        //
        // SIBLING DIVERGENCE: `RewriteDataFiles` rolls at its resolved TARGET, so the two ports
        // roll at different bounds until that template follows.
        let write_max_file_size =
            d2l(target as f64 + (max_file_size_bytes - target) as f64 * WRITE_MAX_FILE_SIZE_RATIO);

        // The candidate-filter HEADROOM: how far a run-1 output may exceed `write_max` and still
        // land inside `[min, max]`, where `outsideDesiredFileSizeRange` declines it forever.
        let headroom = max_file_size_bytes.saturating_sub(write_max_file_size);
        let chunk_budget = CHUNK_MAX_SERIALIZED_BYTES.min(headroom / CHUNK_HEADROOM_FOOTER_SHARE);

        // INTENT DOCUMENTATION WITH A TRIPWIRE, and nothing more. The assert is TRIVIALLY TRUE by
        // construction, so it does NOT establish the runtime clearance and must not be described as
        // doing so. `test_no_split_output_exceeds_max_file_size` establishes that by measurement.
        // The assert buys a loud red if an editor stops respecting the headroom.
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

    /// Stages S1 to S3: walk the current snapshot's manifests ONCE, collect the live PARQUET
    /// position-delete entries the user `filter` admits, and group them by `(spec_id, partition)`.
    /// Equality and data entries are excluded, and so is every non-Parquet position delete.
    ///
    /// The filter is applied PER ENTRY, not per group, because Java applies it at the scan. The
    /// selection is identical either way. What changes is WHEN an unbindable filter errors.
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
                // Only PARQUET position deletes. Skip data and equality deletes here; the FORMAT
                // skip below drops every non-Parquet position delete.
                if data_file.content_type() != DataContentType::PositionDeletes {
                    continue;
                }
                // THE FORMAT SKIP, a FORK DIVERGENCE rather than a port. It drops two classes of
                // live position delete: Puffin deletion vectors, which are file-scoped and never
                // bin-packed here, and V2 ORC or Avro files, a legal encoding this action cannot
                // read. Java's planner is FORMAT-BLIND and its filter is size-only, so Java
                // compacts an ORC or Avro table that this action leaves alone.
                if data_file.file_format() != DataFileFormat::Parquet {
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

    /// Stage S2's ONE bind: bind [`Self::filter`] to the table schema before the manifest walk.
    /// The `AlwaysTrue` default matches everything and never binds. An unbindable filter fails
    /// here.
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

    /// Compact ONE ADMITTED BIN: read every member's pairs, sort them, write the compacted
    /// file(s), and commit ONE `RewriteFiles` that stamps each output with the bin MAX rewritten
    /// data-seq and validates from the starting snapshot.
    ///
    /// The output COUNT moves in both directions and the mask-identity promise holds either way.
    /// The rolling writer rolls at [`ResolvedConfig::write_max_file_size`], so a bin above that
    /// bound produces several files. Every output of THIS bin carries THIS bin's own max. Java
    /// ranges the max over one group's task list, and one Java group IS one bin, so ranging over
    /// the whole partition or reusing another bin's max is a stamping error.
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

        // THE PER-BIN ZERO-PAIRS SKIP, defensive and deliberately NOT a parity claim.
        //
        // A bin can pass the gate and yield no pairs, because the gate reads `file_size_in_bytes`
        // and never a row count. Only an externally written zero-row position delete reaches that
        // state; Java cannot reach it at all.
        //
        // The skip is PER BIN and must STAY per bin. Returning the table unchanged lets `execute`
        // continue with the remaining bins. An early return in `execute` would silently drop every
        // later bin, because the counts would merely come back smaller.
        if pairs.is_empty() {
            return Ok(table.clone());
        }

        // Spec-recommended ordering: sort by `(file_path, pos)`. Java does not dedup within a
        // group, because the reader bitmap dedups, so duplicates are kept.
        //
        // GLOBAL SORT, BEFORE ANY SPLIT. `write_compacted_file` chunks this sorted `Vec` in order,
        // so the bin's outputs carry DISJOINT ASCENDING ranges whose union is this multiset.
        // Sorting per chunk still writes every pair, but destroys the ordering that delete-file
        // range pruning depends on. Do not move this below the split.
        pairs.sort();

        // (3) Write the compacted position-delete file(s) under the group spec + partition key. The
        // rolling writer rolls at `write_max_file_size`, so this is ONE file for a bin below that
        // bound and N files for a bin above it.
        let new_files = self
            .write_compacted_file(table, key, &pairs, config)
            .await?;

        // (4) STALLER: THIS BIN's max rewritten data sequence number, ranged over `entries`, which
        // IS the bin. Ranging over the whole partition, or carrying another bin's value, is a
        // stamping error. The max of the rewritten bin preserves which data generation the
        // compacted delete masks.
        //
        // DIRECTION OF DANGER: `applicable_pos_deletes` keeps a delete whose
        // `delete_seq >= data_seq`. An OVER-HIGH stamp reaches data it never masked and
        // over-applies. An UNDER-LOW stamp stops applying and resurrects deleted rows.
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

        // Java `rewrittenBytesCount` / `addedBytesCount`. The added side stays a CHECKED sum
        // across the split outputs rather than a plain `sum()`.
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

        // (5) Commit ONE RewriteFiles per BIN. Each output is stamped through
        // `add_delete_file_with_sequence_number`, NOT the default-inherit add, and the commit
        // validates from the starting snapshot. Java `newRewrite().validateFromSnapshot(J)
        // .deleteFile(rewritten).addFile(added, J).commit()`.
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

    /// Read one parquet position-delete file's reserved `file_path` and `pos` columns by FIELD ID,
    /// appending every pair into `pairs`. The columns are located by their
    /// `PARQUET_FIELD_ID_META_KEY` metadata, so a renamed but correctly identified file still
    /// reads.
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
    /// `pairs` feeds ONE [`RollingFileWriter`](crate::writer::file_writer::rolling_writer) in
    /// bounded chunks: at most [`CHUNK_PAIRS`] pairs and at most `config.chunk_budget` measured
    /// serialized bytes, with a floor of ONE pair so a tight budget still terminates. `should_roll`
    /// runs once per `write`, so a single whole-bin batch could never roll.
    ///
    /// The writer's bound is `config.write_max_file_size`, passed EXPLICITLY.
    /// `RollingFileWriterBuilder::new_with_default_file_size` hard-wires the 512 MiB **data**
    /// default, which is unrelated to this action.
    ///
    /// # The split preserves the global order
    ///
    /// The chunks are contiguous slices of the sorted `pairs`, taken front to back, so output *k*'s
    /// range is entirely below output *k+1*'s and their union is the input multiset. The masked row
    /// set is unchanged by the split.
    ///
    /// # Stated assumption (measured by the pins, not assumed away)
    ///
    /// `chunk_budget` counts RAW INPUT bytes while `should_roll` measures OUTPUT parquet bytes.
    /// The overshoot bound therefore rests on one chunk's parquet contribution staying below its
    /// raw bytes. `test_no_split_output_exceeds_max_file_size` measures that rather than assuming
    /// it.
    ///
    /// # RESIDUE (RES-8), both halves
    ///
    /// **Correctness half:** on a very tight `[write_max, max]` band the one-pair floor, or a
    /// footer larger than its reserved half, can still overshoot. Java's bound is best-effort too.
    ///
    /// **Throughput half:** `chunk_budget` resolves to `0` when
    /// `max_file_size_bytes - target_file_size_bytes <= 2`, and to `1` at `3`. The one-pair floor
    /// then governs and the feed degrades to one Arrow batch PER PAIR. Correct, and arbitrarily
    /// slow. Both bands are legal, so this is reachable configuration. Clamping the budget upward
    /// would trade a throughput cliff for a correctness one.
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
        // Position-delete files keep `file_path` and `pos` bounds FULL, so path pruning stays
        // precise. The default `truncate(16)` would widen the path range.
        let parquet_builder = ParquetWriterBuilder::new(
            position_delete_writer_properties(),
            writer_config.schema().clone(),
        )
        .with_metrics_config(MetricsConfig::for_position_delete());
        // Java `writeMaxFileSize()`, NOT the resolved target. The builder takes a `usize`, so on a
        // 32-bit target a larger bound saturates to "never roll", as the default would.
        let rolling = RollingFileWriterBuilder::new(
            parquet_builder,
            usize::try_from(config.write_max_file_size).unwrap_or(usize::MAX),
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );

        // The new pos-delete must live in the SAME partition and spec as the files it replaces, so
        // it lands in the same bucket and applies to the same data files. Always pass a
        // `PartitionKey`, empty and all-Void tuples included, so we never fabricate spec_id 0.
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

    /// THE V3 ARM. Convert every live filter-matching PARQUET position delete into one Puffin
    /// DELETION VECTOR per referenced data file, merged with that file's existing DV, in ONE
    /// `RewriteFiles`.
    ///
    /// # Errors
    ///
    /// A format the arm cannot read, a live excluded delete a written vector would SHADOW, two live
    /// DVs for one data file, or a DV whose sequence would fall below its data file's. So `Ok` with
    /// zero counts means "looked, found nothing to convert".
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

        // Every DV in a superseded Puffin is removed, INCLUDING the siblings the closure rewrote.
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

    /// Take the V3 delete inventory in ONE manifest walk: every live data file, every live PARQUET
    /// position delete the user filter admits, and every live Puffin DV keyed by the data file it
    /// references.
    ///
    /// # Errors
    ///
    /// A live position delete that is neither Parquet nor Puffin, a Puffin DV with no derivable
    /// referenced data file, or two live DVs for one data file.
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

    /// Plan one merged deletion vector per data file, and name the Puffin files the plan supersedes.
    ///
    /// # Notes
    ///
    /// THE PUFFIN CLOSURE. A delete file is removed BY PATH, and one Puffin holds a blob per data
    /// file, so superseding one DV removes every sibling blob with it. Each sibling is rewritten
    /// too, or its deleted rows come back. The merge also makes a SHADOWED position effective; Java's
    /// `BaseDVFileWriter` already folds those into the DV it writes, so a real writer's table holds
    /// a superset and the live rows do not move.
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
                // A position naming a data file the snapshot no longer holds deletes nothing, so
                // drop it rather than carry it into a DV. Refusing here dead-ends the table:
                // `RemoveDanglingDeleteFiles` cannot clear a delete file that still names one live
                // data file, so nothing could, and the arm would refuse for ever.
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
        // The closure: pull in every sibling blob of a superseded Puffin (see the Notes above).
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
                // THE OTHER DIRECTION of the shadow. This DV already suppresses the legacy delete,
                // so a position it does NOT hold is a row the table returns TODAY, and folding it
                // in would silently delete it. `positions` holds only legacy-derived positions, so
                // a Puffin-closure sibling passes. A refusal is safe because
                // `write_deletion_vectors` opens no Puffin and commits nothing until later.
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
                             width. RewriteDataFiles can, given remove_dangling_deletes(true) AND a \
                             delete_file_threshold low enough to make this data file a rewrite \
                             candidate — its default disables that clause, so the rewrite is a \
                             no-op without it unless the planner admits the file for another \
                             reason. The legacy delete is shadowed, so rewriting keeps today's live \
                             rows and both delete files then fall dangling."
                        ),
                    ));
                }
                plan.positions.extend(previous.iter());
                plan.sequence_number = plan.sequence_number.max(entry.sequence_number);
                superseded_puffin_paths.insert(entry.data_file.file_path().to_string());
            }
            // A DV below its data file's sequence number fails the scan, so refuse to write one
            // rather than commit a table the reader rejects.
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

    /// Write every planned deletion vector into ONE Puffin file and return the DV `DataFile`s.
    ///
    /// # Notes
    ///
    /// Every [`DVFileWriter::delete`] call carries the referenced data file's OWN [`PartitionKey`].
    /// That keeps the writer off `resolve_partition_spec_id`'s keyless arm, which would stamp spec
    /// 0 with an empty partition tuple. [`DVFileWriter::with_partition_spec`] is deliberately NOT
    /// used: one Puffin spans every partition this arm touches, so no single spec describes it.
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
/// The one-pair floor keeps the feed loop total. `chunk_budget` can legitimately resolve to `0`,
/// and a chunk of zero pairs would spin forever. It is also what RES-8 names: a pair whose path
/// exceeds the whole budget overshoots the bound by construction.
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
/// NORMAL return from the parquet writer, not an error. Without this check `execute` commits a
/// `Replace` snapshot that removes live position-delete files and adds none, which under-masks
/// silently. Nothing downstream rejects it, because `RewriteFilesAction::validate` early-returns on
/// the empty `deleted_data_files` this action always passes.
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

/// Stages S4 to S6 of the planner: candidate-filter each group, bin-pack the survivors, and keep
/// the bins the three-clause group filter admits. Java
/// `BinPackRewritePositionDeletePlanner.planFileGroups`.
///
/// Java's `filterFiles` then `pack` then `filterFileGroups` order is load-bearing. Packing first
/// would let an in-range file consume bin headroom and split candidates that belong together.
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
        // `Vec`, so this is a short-circuit rather than a behaviour.
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
/// There is deliberately NO delete-count clause. `tooManyDeletes` and `tooHighDeleteRatio` live on
/// the DATA-file planner, which this planner does not inherit from.
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
/// Two leaf sub-expressions are unreachable end to end, so no end-to-end fixture kills their
/// mutants. (a) `enough_content`'s `size > 1`: a lone candidate is either below `min`, where
/// `enough_content` is false because `min < target`, or above `max`, where `too_much_content` is
/// already true. (b) `too_much_content`'s boundary strictness: a bin at `input_size == max` is
/// either size 1, whose file is then not a candidate, or size >= 2, where `enough_content` is
/// already true. Both proofs need only `min < target < max`.
///
/// UNREACHABLE IS NOT UNPINNED. Both states are constructible through the white-box seam the test
/// module opens on this function, and both mutants are killed by
/// `test_gate_enough_content_size_guard_declines_lone_over_target_file_white_box` and
/// `test_gate_input_size_exactly_max_is_declined_white_box`. Do not delete either leaf as dead code.
///
/// The input-size sum SATURATES where Java's `sum()` wraps on overflow. A wrapped negative sum
/// makes both size clauses false, so Java would decline where this port admits. It needs more than
/// 8 EiB in one bin, and the input is manifest-trusted, so saturating is the safer choice.
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

/// Locate the `file_path` and `pos` columns of a position-delete record batch by their RESERVED
/// FIELD IDs, never by name or column order. A file written with the reserved ids but a renamed
/// column still reads.
///
/// # Errors
///
/// Either reserved column is absent, or has the wrong arrow type.
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

/// One live DATA file, as the V3 arm needs it: where a deletion vector covering it must be stamped,
/// and the sequence number such a vector must not fall below.
struct LiveDataFile {
    partition_spec_id: i32,
    partition: Struct,
    sequence_number: i64,
}

/// The live delete inventory of a V3 table, taken in ONE manifest walk by
/// [`RewritePositionDeleteFiles::collect_v3_delete_inventory`].
#[derive(Default)]
struct V3DeleteInventory {
    data_files: HashMap<String, LiveDataFile>,
    /// The live PARQUET position deletes the user filter admits — the files the V3 arm consumes.
    legacy_position_deletes: Vec<LiveDeleteEntry>,
    /// The live Puffin deletion vectors, keyed by the data file each one references.
    deletion_vectors: HashMap<String, LiveDeleteEntry>,
    /// The live non-Puffin position deletes the filter REJECTED. They stay live, so a new deletion
    /// vector for a data file they cover would SHADOW them. See [`refuse_shadowed_deletes`].
    unconverted_position_deletes: Vec<LiveDeleteEntry>,
}

impl V3DeleteInventory {
    /// Route ONE live position-delete entry into the inventory.
    ///
    /// # Errors
    ///
    /// A format that is neither Parquet nor Puffin, a Puffin DV with no derivable referenced data
    /// file, or a second live DV for a data file that already has one.
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
            // Refused, not skipped. Dropping these silently would make zero counts mean "did not
            // look".
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
                // Outside the filter's scope, so unreadable is not yet fatal. It still cannot be
                // shadowed by a new DV, which `refuse_shadowed_deletes` decides.
                self.unconverted_position_deletes.push(entry);
                Ok(())
            }
        }
    }

    /// The live data file at `data_file_path`.
    ///
    /// # Errors
    ///
    /// `Unexpected` when the current snapshot does not hold it. Planned paths are filtered to live
    /// data files first, so a miss is a bug in this module, not a table state.
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

/// The deletion vector the V3 arm will write for ONE data file: the merged positions and the data
/// sequence number to stamp on it.
#[derive(Default)]
struct DeletionVectorPlan {
    positions: Vec<u64>,
    sequence_number: i64,
}

/// REFUSE a run that would leave a live position delete SHADOWED by a deletion vector it wrote.
///
/// # Notes
///
/// A DV wins over every position delete for the same data file, so an excluded delete goes INERT
/// and its rows come back. This RE-DERIVES that routing rather than sharing it.
/// `PopulatedDeleteFileIndex::new` owns the real keying. If that changes, this diverges silently.
///
/// # Errors
///
/// `DataInvalid`. Widen the filter — unless the delete is ORC or Avro, which no width converts.
fn refuse_shadowed_deletes(
    inventory: &V3DeleteInventory,
    plans: &HashMap<String, DeletionVectorPlan>,
) -> Result<()> {
    if plans.is_empty() {
        return Ok(());
    }
    // The `(spec_id, partition)` of every data file about to gain a DV — the key the reader routes a
    // PARTITION-scoped position delete on.
    let mut planned_partitions: HashSet<(i32, &Struct)> = HashSet::new();
    for data_file_path in plans.keys() {
        let data_file = inventory.live_data_file(data_file_path)?;
        planned_partitions.insert((data_file.partition_spec_id, &data_file.partition));
    }

    for entry in &inventory.unconverted_position_deletes {
        let delete_file = &entry.data_file;
        let shadowed_data_file = match referenced_data_file_location(delete_file) {
            // File-scoped: routed by path alone.
            Some(referenced) => plans.contains_key(&referenced).then_some(referenced),
            // Partition-scoped: routed by the DATA file's spec and partition. The sequence rule is
            // deliberately NOT applied. A false alarm the caller can clear beats losing rows.
            None => planned_partitions
                .contains(&(delete_file.partition_spec_id, delete_file.partition()))
                .then(|| "a data file in the same partition".to_string()),
        };
        let Some(shadowed_data_file) = shadowed_data_file else {
            continue;
        };
        // Widening the filter only helps if the arm could then READ it. An ORC or Avro delete is
        // routed to `FeatureUnsupported` at any filter width, so such a table cannot be converted.
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

/// The stamp for one written deletion vector, read back out of the plan it came from.
///
/// # Errors
///
/// `Unexpected` when the written DV carries no referenced data file, or names one the plan does not
/// hold — both are writer bugs, not table states.
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

/// The four `Result` counts of one V3 rewrite. DVs count as delete files on both sides.
///
/// # Notes
///
/// Bytes are summed over DISTINCT file paths, counts are not. One Puffin holds a blob per data
/// file, and each of those `DataFile`s carries the WHOLE Puffin's size, so a per-entry sum would
/// report the same bytes once per blob.
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
