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

//! `ConvertEqualityDeleteFiles` — the engine-agnostic maintenance action that materializes every live
//! EQUALITY-delete file in the current snapshot into an equivalent POSITION-delete file and commits the
//! swap in one `Replace` snapshot. The Rust port of Java's
//! `org.apache.iceberg.actions.ConvertEqualityDeleteFiles` (api 1.10.0).
//!
//! # The Java contract this mirrors
//!
//! `ConvertEqualityDeleteFiles extends SnapshotUpdate<ConvertEqualityDeleteFiles, Result>` with one own
//! method `filter(Expression)` plus the inherited `execute() -> Result` (javap-verified against
//! `iceberg-api` 1.10.0). `ConvertEqualityDeleteFiles$Result` exposes exactly two `int` counts —
//! `convertedEqualityDeleteFilesCount()` and `addedPositionDeleteFilesCount()` — mirrored 1:1 by
//! [`ConvertEqualityDeleteFilesResult`]. The RUNNABLE impl is a Spark-surface action (no Spark bytecode
//! is available locally), so the materialization pipeline below is built engine-agnostically; the Java
//! contract pins the INTERFACE shape + the two `Result` counts only.
//!
//! Like the partition-statistics action, `ConvertEqualityDeleteFiles` is NOT an `ActionsProvider`
//! method (the 12-method provider surface is javap-confirmed and does not include it) — it is a
//! FREE-STANDING action built `ConvertEqualityDeleteFiles::new(table)` and run `.execute(catalog)`.
//!
//! # The conversion (eq → pos)
//!
//! An equality-delete file deletes every row that matches one of its tuples, across all data files in
//! its applicability scope. A position-delete file deletes by `(file_path, pos)` — the exact rows. The
//! conversion reads each data file the equality delete applies to, finds the matching rows' absolute
//! positions, and writes them into a position-delete file. The eq-delete and the new pos-delete then
//! mask the SAME live rows, so a merge-on-read scan returns an identical row set before and after.
//!
//! For each live equality-delete file in the current snapshot (optionally restricted by
//! [`ConvertEqualityDeleteFiles::filter`], applied to the eq-delete's partition):
//!
//! 1. Build the eq-delete's [`Predicate`] from its on-disk tuples (reusing the read-side
//!    `parse_equality_deletes_record_batch_stream` machinery). The parser yields the SURVIVAL predicate
//!    (a row that does NOT match any tuple), so a row is DELETED iff that predicate is FALSE for it.
//! 2. Determine the applicable LIVE data files — same partition + spec, STRICTLY-LOWER data sequence
//!    number (the equality-delete applicability rule; global across partitions for an unpartitioned
//!    equality delete).
//! 3. Read each applicable data file in physical (file) order with NO row-group skipping, tracking an
//!    ABSOLUTE `_pos` across row groups, and evaluate the survival predicate to collect the MATCHING
//!    `(file_path, pos)` rows (the rows the eq-delete deletes — the INVERSE of the read-side survival
//!    filter).
//! 4. Sort the collected pairs by `(file_path, pos)` (the spec-recommended pos-delete ordering) and
//!    write them into one position-delete file per source equality-delete file.
//! 5. Commit one [`RewriteFilesAction`](crate::transaction::rewrite_files) that REPLACES the converted
//!    equality-delete files with the new position-delete files, STAMPING each new position-delete with
//!    its source equality-delete's DATA sequence number (via
//!    `add_delete_file_with_sequence_number`), so the converted deletes keep applying to exactly the
//!    same data generation.
//!
//! # The four silent-corruption stallers (each handled EXPLICITLY)
//!
//! 1. **Absolute `_pos`.** Positions are FILE-absolute across row groups, not batch-relative. The read
//!    path reads every row group in physical order with NO row-group skip / row filter / byte-range
//!    split, and the position counter is a monotonically increasing `i64` advanced by `batch.num_rows()`
//!    per emitted batch — so a multi-row-group file produces positions `0..record_count` exactly.
//! 2. **Data-seq stamping.** Each new position delete is added with
//!    `add_delete_file_with_sequence_number(pos_file, eq_delete_data_seq)`, NOT the default-inherit
//!    `add_delete_file`. A position delete applies to data with `data_seq < delete_seq`; stamping the
//!    eq-delete's own data seq preserves exactly which data generation the converted delete masks. An
//!    inherited (higher) seq would resurrect rows; a lower seq would over-mask.
//! 3. **Applicability scope.** An equality delete applies ONLY to data files in the SAME partition+spec
//!    with a STRICTLY-LOWER data sequence number (`data_seq < eq_seq`); an unpartitioned equality delete
//!    applies GLOBALLY (to every lower-seq data file regardless of partition). Including an equal- or
//!    higher-seq data file over-masks; missing an applicable one under-masks.
//! 4. **Matching-vs-surviving inversion.** The pipeline collects the rows the eq-delete DELETES (the
//!    survival predicate is FALSE), the inverse of the read-side `RowFilter` path which keeps survivors
//!    and discards the matched positions.
//!
//! # No-op
//!
//! With no current snapshot, no live equality-delete files, or a [`filter`](ConvertEqualityDeleteFiles::filter)
//! that matches none, the action commits NOTHING and returns a zero-count result (Java commits only when
//! there is work).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_arith::boolean::{and, and_kleene, is_not_null, is_null, not, or, or_kleene};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Datum as ArrowDatum, Int64Array, RecordBatch, StringArray,
};
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow_schema::ArrowError;
use arrow_string::like::starts_with;
use fnv::FnvHashSet;
use futures::StreamExt;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::arrow::{ArrowReader, ParquetReadOptions, get_arrow_datum, try_cast_literal};
use crate::expr::visitors::bound_predicate_visitor::{BoundPredicateVisitor, visit};
use crate::expr::visitors::expression_evaluator::ExpressionEvaluator;
use crate::expr::visitors::inclusive_projection::InclusiveProjection;
use crate::expr::{Bind, BoundPredicate, BoundReference, Predicate};
use crate::io::FileIO;
use crate::scan::ArrowRecordBatchStream;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, PartitionKey, Schema, Snapshot,
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

/// The outcome of a [`ConvertEqualityDeleteFiles::execute`] run, mirroring Java
/// `ConvertEqualityDeleteFiles$Result`'s two `int` counts.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ConvertEqualityDeleteFilesResult {
    /// Number of equality-delete files converted (Java `convertedEqualityDeleteFilesCount()`).
    pub converted_equality_delete_files_count: usize,
    /// Number of position-delete files added (Java `addedPositionDeleteFilesCount()`).
    pub added_position_delete_files_count: usize,
}

impl ConvertEqualityDeleteFilesResult {
    /// Number of equality-delete files converted (Java `convertedEqualityDeleteFilesCount()`).
    pub fn converted_equality_delete_files_count(&self) -> usize {
        self.converted_equality_delete_files_count
    }

    /// Number of position-delete files added (Java `addedPositionDeleteFilesCount()`).
    pub fn added_position_delete_files_count(&self) -> usize {
        self.added_position_delete_files_count
    }
}

/// The `ConvertEqualityDeleteFiles` maintenance action. Build it with [`Self::new`], optionally restrict
/// the converted partitions with [`Self::filter`], and run it with [`Self::execute`].
pub struct ConvertEqualityDeleteFiles {
    table: Table,
    filter: Predicate,
}

impl ConvertEqualityDeleteFiles {
    /// Create a `ConvertEqualityDeleteFiles` action for `table`. With no [`filter`](Self::filter), every
    /// live equality-delete file in the current snapshot is converted.
    pub fn new(table: Table) -> Self {
        Self {
            table,
            filter: Predicate::AlwaysTrue,
        }
    }

    /// Restrict the conversion to equality-delete files whose partition matches `filter` (Java
    /// `ConvertEqualityDeleteFiles.filter(Expression)`). The predicate is projected onto each
    /// equality-delete file's partition spec (inclusive projection) and evaluated against the
    /// eq-delete's PARTITION values, so a partition pruning expression selects which equality deletes
    /// are converted. The default is [`Predicate::AlwaysTrue`] (convert all).
    pub fn filter(mut self, filter: Predicate) -> Self {
        self.filter = filter;
        self
    }

    /// Run the conversion: for every live (filter-matching) equality-delete file in the current
    /// snapshot, materialize it into a position-delete file and commit the swap in one `Replace`
    /// snapshot. Returns the [`ConvertEqualityDeleteFilesResult`] counts.
    ///
    /// Commits NOTHING and returns a zero-count result when there is no current snapshot, no live
    /// equality-delete files, or none match the filter / mask any live row.
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<ConvertEqualityDeleteFilesResult> {
        let metadata = self.table.metadata();
        let Some(snapshot) = metadata.current_snapshot().cloned() else {
            return Ok(ConvertEqualityDeleteFilesResult::default());
        };

        let live = self.collect_live_entries(&snapshot).await?;
        if live.equality_deletes.is_empty() {
            return Ok(ConvertEqualityDeleteFilesResult::default());
        }

        // Materialize each (filter-matching) equality-delete file into a position-delete file, pairing
        // the new pos-delete with the seq to stamp it and the eq-delete to replace.
        let mut converted: Vec<DataFile> = Vec::new();
        let mut added: Vec<(DataFile, i64)> = Vec::new();
        for eq in &live.equality_deletes {
            if !self.eq_delete_matches_filter(eq)? {
                continue;
            }
            match self.materialize_one(eq, &live).await? {
                Some(pos_delete) => {
                    converted.push(eq.data_file.clone());
                    added.push((pos_delete, eq.sequence_number));
                }
                None => {
                    // The equality delete matches no live row in any applicable data file (e.g. its
                    // only applicable data was already compacted away). Java emits no pos-delete; we
                    // still REPLACE the eq-delete (it carries no live information) without adding a
                    // 0-row pos-delete file.
                    converted.push(eq.data_file.clone());
                }
            }
        }

        if converted.is_empty() {
            return Ok(ConvertEqualityDeleteFilesResult::default());
        }

        // Commit ONE RewriteFiles replacing the equality deletes with the new position deletes. The
        // position deletes are added with their source eq-delete's DATA sequence number (staller 2),
        // NOT the inherited seq — so they apply to exactly the same data generation. The 4-set
        // `rewrite_files_with_deletes` would inherit the seq, so the explicit-seq add surface is
        // driven directly here.
        let added_count = added.len();
        let converted_count = converted.len();
        let transaction = Transaction::new(&self.table);
        let mut action = transaction
            .rewrite_files(Vec::new(), Vec::new())
            .delete_delete_files(converted);
        for (pos_delete, seq) in added {
            action = action.add_delete_file_with_sequence_number(pos_delete, seq);
        }
        let transaction = action.apply(transaction)?;
        transaction.commit(catalog).await?;

        Ok(ConvertEqualityDeleteFilesResult {
            converted_equality_delete_files_count: converted_count,
            added_position_delete_files_count: added_count,
        })
    }

    /// Whether the equality-delete file's partition matches [`Self::filter`]. `AlwaysTrue` (the default)
    /// matches everything without binding. Otherwise the row-level filter is bound to the table schema,
    /// inclusively projected onto the eq-delete's partition spec, and evaluated against the eq-delete's
    /// partition struct — the SAME partition-pruning path the table scan uses.
    fn eq_delete_matches_filter(&self, eq: &LiveDeleteEntry) -> Result<bool> {
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
                    "ConvertEqualityDeleteFiles filter could not be bound to the table schema",
                )
                .with_source(e)
            })?;

        let spec = metadata
            .partition_spec_by_id(eq.data_file.partition_spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Equality delete '{}' references unknown partition spec {}",
                        eq.data_file.file_path(),
                        eq.data_file.partition_spec_id
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

        ExpressionEvaluator::new(partition_filter).eval(&eq.data_file)
    }

    /// Walk the current snapshot's manifests once and collect: the live equality-delete entries, and the
    /// live DATA files (path + partition + spec + data sequence number) so applicability can be decided
    /// per equality delete. One pass over the manifest list.
    async fn collect_live_entries(&self, snapshot: &Snapshot) -> Result<LiveEntries> {
        let metadata = self.table.metadata();
        let manifest_list = snapshot
            .load_manifest_list(self.table.file_io(), metadata)
            .await?;

        let mut live = LiveEntries::default();
        for manifest_file in manifest_list.entries() {
            let manifest = manifest_file.load_manifest(self.table.file_io()).await?;
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let data_file = entry.data_file();
                match entry.content_type() {
                    DataContentType::Data => {
                        live.data_files.push(LiveDataFile {
                            data_file: data_file.clone(),
                            sequence_number: entry.sequence_number(),
                        });
                    }
                    DataContentType::EqualityDeletes => {
                        live.equality_deletes.push(LiveDeleteEntry {
                            data_file: data_file.clone(),
                            // A live eq-delete always carries a concrete post-inheritance seq; the
                            // unwrap-or default never fires for a real on-disk entry.
                            sequence_number: entry.sequence_number().unwrap_or(0),
                        });
                    }
                    DataContentType::PositionDeletes => {
                        // Position deletes (parquet or DV) are not converted; leave them untouched.
                    }
                }
            }
        }

        Ok(live)
    }

    /// Materialize one equality-delete file into a position-delete file. Returns `Ok(None)` when the
    /// equality delete matches no live row in any applicable data file (no positions to write).
    async fn materialize_one(
        &self,
        eq: &LiveDeleteEntry,
        live: &LiveEntries,
    ) -> Result<Option<DataFile>> {
        let metadata = self.table.metadata();
        let schema = metadata.current_schema().clone();

        // The predicate + STALLER (4): parse the eq-delete's tuples into the SURVIVAL predicate (a row
        // that does NOT match any tuple), then bind it. A row is DELETED iff this predicate is FALSE.
        let equality_ids = eq.data_file.equality_ids().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Equality delete '{}' has no equality_ids",
                    eq.data_file.file_path()
                ),
            )
        })?;
        let survival_predicate = self
            .build_survival_predicate(&eq.data_file, &equality_ids)
            .await?;
        let bound_survival = survival_predicate.bind(schema.clone(), true).map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Could not bind the equality-delete predicate for '{}' to the table schema",
                    eq.data_file.file_path()
                ),
            )
            .with_source(e)
        })?;

        // STALLER (3): the applicable data files — strictly-lower data seq, same partition+spec (global
        // across partitions when the eq-delete is unpartitioned).
        let eq_unpartitioned = eq.data_file.partition().fields().is_empty();
        let mut pairs: Vec<(String, i64)> = Vec::new();
        for data in &live.data_files {
            if !is_applicable(eq, data, eq_unpartitioned) {
                continue;
            }
            // STALLER (1): read the data file in physical order with NO row-group skipping, tracking the
            // absolute position, and collect the rows the eq-delete deletes (survival predicate FALSE).
            self.collect_matching_positions(
                &data.data_file,
                &equality_ids,
                &schema,
                &bound_survival,
                &mut pairs,
            )
            .await?;
        }

        if pairs.is_empty() {
            return Ok(None);
        }

        // Spec-recommended position-delete ordering: sort by (file_path, pos) before writing.
        pairs.sort();

        let pos_delete = self.write_position_delete_file(eq, &pairs).await?;
        Ok(Some(pos_delete))
    }

    /// Build the eq-delete's SURVIVAL predicate by reading its on-disk tuples and reusing the read-side
    /// equality-delete parser (the same machinery the scan uses). The returned predicate is true for a
    /// row that does NOT match any tuple (so a DELETED row makes it false).
    async fn build_survival_predicate(
        &self,
        eq_file: &DataFile,
        equality_ids: &[i32],
    ) -> Result<Predicate> {
        let stream = read_data_file_stream(
            self.table.file_io(),
            eq_file.file_path(),
            eq_file.file_size_in_bytes,
        )
        .await?;
        let equality_id_set: HashSet<i32> = equality_ids.iter().copied().collect();
        CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(stream, equality_id_set)
            .await
    }

    /// Read `data_file` in physical (file) order with NO row-group skipping, evaluate `bound_survival`
    /// per batch, and push the absolute `(data_file_path, pos)` of every DELETED row (survival predicate
    /// FALSE) into `pairs`.
    async fn collect_matching_positions(
        &self,
        data_file: &DataFile,
        equality_ids: &[i32],
        schema: &Schema,
        bound_survival: &BoundPredicate,
        pairs: &mut Vec<(String, i64)>,
    ) -> Result<()> {
        // Read the equality-id columns evolved to the table schema. The evolution stamps each column
        // with its field id (PARQUET_FIELD_ID_META_KEY), which the evaluator maps references against.
        let raw_stream = read_data_file_stream(
            self.table.file_io(),
            data_file.file_path(),
            data_file.file_size_in_bytes,
        )
        .await?;
        let target_schema = Arc::new(schema.clone());
        let mut stream =
            BasicDeleteFileLoader::evolve_schema(raw_stream, target_schema, equality_ids).await?;

        let mut absolute_pos: i64 = 0;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let num_rows = batch.num_rows();

            // Evaluate the SURVIVAL predicate to a BooleanArray. A DELETED row is one where survival is
            // false. A NULL survival (the row matched a tuple and the negation produced null under
            // three-valued logic) is treated conservatively as NOT deleted so the conversion never
            // over-masks beyond what the eq-delete's read path drops.
            let mut evaluator = MatchingPositionEvaluator::new(&batch)?;
            let survives = visit(&mut evaluator, bound_survival)?;
            for row in 0..num_rows {
                let deleted = !survives.is_null(row) && !survives.value(row);
                if deleted {
                    // STALLER (1): absolute_pos is FILE-absolute (advanced by full batch row counts
                    // across every row group), never batch-relative.
                    let row_offset = i64::try_from(row).map_err(|e| {
                        Error::new(ErrorKind::Unexpected, "row index exceeds i64").with_source(e)
                    })?;
                    pairs.push((data_file.file_path().to_string(), absolute_pos + row_offset));
                }
            }
            let batch_rows = i64::try_from(num_rows).map_err(|e| {
                Error::new(ErrorKind::Unexpected, "batch row count exceeds i64").with_source(e)
            })?;
            absolute_pos = absolute_pos
                .checked_add(batch_rows)
                .ok_or_else(|| Error::new(ErrorKind::Unexpected, "absolute position overflow"))?;
        }

        Ok(())
    }

    /// Write the sorted `(file_path, pos)` pairs into one position-delete file under the eq-delete's
    /// partition, returning the resulting [`DataFile`].
    async fn write_position_delete_file(
        &self,
        eq: &LiveDeleteEntry,
        pairs: &[(String, i64)],
    ) -> Result<DataFile> {
        let metadata = self.table.metadata();
        let schema = metadata.current_schema().clone();
        let spec = metadata
            .partition_spec_by_id(eq.data_file.partition_spec_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Equality delete '{}' references unknown partition spec {}",
                        eq.data_file.file_path(),
                        eq.data_file.partition_spec_id
                    ),
                )
            })?
            .as_ref()
            .clone();

        let config = PositionDeleteWriterConfig::new()?;
        let location_gen = DefaultLocationGenerator::new(metadata.clone())?;
        let file_name_gen = DefaultFileNameGenerator::new(
            "converted-pos-del".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        let parquet_builder = ParquetWriterBuilder::new(
            parquet::file::properties::WriterProperties::builder().build(),
            config.schema().clone(),
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            self.table.file_io().clone(),
            location_gen,
            file_name_gen,
        );

        // The new pos-delete must live in the SAME partition as the eq-delete it replaces (so it lands
        // in the same partition+spec bucket and applies to the same data files). An unpartitioned spec
        // takes no partition key.
        let partition_key = if spec.is_unpartitioned() {
            None
        } else {
            Some(PartitionKey::new(
                spec,
                schema.clone(),
                eq.data_file.partition().clone(),
            ))
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
                "Failed to build position-delete record batch",
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

/// Whether equality-delete `eq` applies to live data file `data` (STALLER 3): strictly-lower data
/// sequence number AND (same partition+spec, OR the eq-delete is unpartitioned → global).
fn is_applicable(eq: &LiveDeleteEntry, data: &LiveDataFile, eq_unpartitioned: bool) -> bool {
    // Equality deletes apply STRICTLY to lower data seq (`data_seq < eq_seq`).
    let Some(data_seq) = data.sequence_number else {
        // A data file with no seq (degenerate fixture only) is treated as not-yet-applicable.
        return false;
    };
    if data_seq >= eq.sequence_number {
        return false;
    }
    if eq_unpartitioned {
        // An unpartitioned equality delete is a GLOBAL delete (applies across every partition).
        return true;
    }
    // Otherwise it applies only within its own partition + spec.
    eq.data_file.partition_spec_id == data.data_file.partition_spec_id
        && eq.data_file.partition() == data.data_file.partition()
}

/// A live equality-delete entry: its [`DataFile`] and its post-inheritance data sequence number.
struct LiveDeleteEntry {
    data_file: DataFile,
    sequence_number: i64,
}

/// A live data file: its [`DataFile`] and its post-inheritance data sequence number.
struct LiveDataFile {
    data_file: DataFile,
    sequence_number: Option<i64>,
}

/// The live-entry view the conversion runs over.
#[derive(Default)]
struct LiveEntries {
    equality_deletes: Vec<LiveDeleteEntry>,
    data_files: Vec<LiveDataFile>,
}

/// Open a data/delete parquet file and stream its record batches in physical (file) order with NO
/// row-group skipping, row filter, or byte-range split — so an absolute position counter advanced by
/// each batch's row count is the true file position (STALLER 1).
async fn read_data_file_stream(
    file_io: &FileIO,
    file_path: &str,
    file_size_in_bytes: u64,
) -> Result<ArrowRecordBatchStream> {
    let parquet_read_options = ParquetReadOptions::builder().build();
    let (parquet_file_reader, arrow_metadata) = ArrowReader::open_parquet_file(
        file_path,
        file_io,
        file_size_in_bytes,
        parquet_read_options,
    )
    .await?;
    let record_batch_stream = parquet::arrow::ParquetRecordBatchStreamBuilder::new_with_metadata(
        parquet_file_reader,
        arrow_metadata,
    )
    .build()
    .map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to build parquet record batch stream",
        )
        .with_source(e)
    })?
    .map(|res| res.map_err(|e| Error::new(ErrorKind::Unexpected, format!("{e}"))));

    Ok(Box::pin(record_batch_stream) as ArrowRecordBatchStream)
}

// =================================================================================================
// The matching-position evaluator: a BoundPredicateVisitor that evaluates the eq-delete SURVIVAL
// predicate against ONE RecordBatch to a BooleanArray, mapping each BoundReference to a column by its
// field-id (PARQUET_FIELD_ID_META_KEY) metadata. Mirrors the read path's `PredicateConverter` arrow
// kernels, but resolves columns by field id (the batch is already schema-evolved) rather than by a
// parquet projection-mask leaf index.
// =================================================================================================

struct MatchingPositionEvaluator<'a> {
    /// field id -> column index in the batch.
    field_id_to_col: HashMap<i32, usize>,
    batch: &'a RecordBatch,
}

impl<'a> MatchingPositionEvaluator<'a> {
    fn new(batch: &'a RecordBatch) -> Result<Self> {
        let mut field_id_to_col = HashMap::new();
        for (idx, field) in batch.schema().fields().iter().enumerate() {
            if let Some(id_str) = field.metadata().get(PARQUET_FIELD_ID_META_KEY)
                && let Ok(id) = id_str.parse::<i32>()
            {
                field_id_to_col.insert(id, idx);
            }
        }
        Ok(Self {
            field_id_to_col,
            batch,
        })
    }

    /// The batch column for a reference, by field id (None when the column is absent — schema
    /// evolution).
    fn column_for(&self, reference: &BoundReference) -> Option<ArrayRef> {
        self.field_id_to_col
            .get(&reference.field().id)
            .map(|idx| self.batch.column(*idx).clone())
    }

    fn all_true(&self) -> Result<BooleanArray> {
        Ok(BooleanArray::from(vec![true; self.batch.num_rows()]))
    }

    fn all_false(&self) -> Result<BooleanArray> {
        Ok(BooleanArray::from(vec![false; self.batch.num_rows()]))
    }

    /// Cast the literal to the column's arrow type (the read-side `try_cast_literal`) and run `kernel`.
    fn binary_cmp(
        &self,
        reference: &BoundReference,
        literal: &Datum,
        on_missing_true: bool,
        kernel: impl Fn(&ArrayRef, &dyn ArrowDatum) -> std::result::Result<BooleanArray, ArrowError>,
    ) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => {
                let lit = get_arrow_datum(literal)?;
                let cast = try_cast_literal(&lit, col.data_type()).map_err(arrow_err)?;
                kernel(&col, cast.as_ref()).map_err(arrow_err)
            }
            None if on_missing_true => self.all_true(),
            None => self.all_false(),
        }
    }
}

fn arrow_err(e: ArrowError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Failed to evaluate equality-delete predicate over a data-file batch",
    )
    .with_source(e)
}

impl BoundPredicateVisitor for MatchingPositionEvaluator<'_> {
    type T = BooleanArray;

    fn always_true(&mut self) -> Result<BooleanArray> {
        self.all_true()
    }

    fn always_false(&mut self) -> Result<BooleanArray> {
        self.all_false()
    }

    fn and(&mut self, lhs: BooleanArray, rhs: BooleanArray) -> Result<BooleanArray> {
        and_kleene(&lhs, &rhs).map_err(arrow_err)
    }

    fn or(&mut self, lhs: BooleanArray, rhs: BooleanArray) -> Result<BooleanArray> {
        or_kleene(&lhs, &rhs).map_err(arrow_err)
    }

    fn not(&mut self, inner: BooleanArray) -> Result<BooleanArray> {
        not(&inner).map_err(arrow_err)
    }

    fn is_null(&mut self, reference: &BoundReference, _p: &BoundPredicate) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => is_null(&col).map_err(arrow_err),
            None => self.all_true(),
        }
    }

    fn not_null(
        &mut self,
        reference: &BoundReference,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => is_not_null(&col).map_err(arrow_err),
            None => self.all_false(),
        }
    }

    fn is_nan(&mut self, reference: &BoundReference, _p: &BoundPredicate) -> Result<BooleanArray> {
        if self.column_for(reference).is_some() {
            self.all_true()
        } else {
            self.all_false()
        }
    }

    fn not_nan(&mut self, reference: &BoundReference, _p: &BoundPredicate) -> Result<BooleanArray> {
        if self.column_for(reference).is_some() {
            self.all_false()
        } else {
            self.all_true()
        }
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, true, |c, l| lt(c, l))
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, true, |c, l| lt_eq(c, l))
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, false, |c, l| gt(c, l))
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, false, |c, l| gt_eq(c, l))
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, false, |c, l| eq(c, l))
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, false, |c, l| neq(c, l))
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, false, |c, l| starts_with(c, l))
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        self.binary_cmp(reference, literal, true, |c, l| not(&starts_with(c, l)?))
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => {
                let mut acc = BooleanArray::from(vec![false; col.len()]);
                for literal in literals {
                    let lit = get_arrow_datum(literal)?;
                    let cast = try_cast_literal(&lit, col.data_type()).map_err(arrow_err)?;
                    let matches = eq(&col, cast.as_ref()).map_err(arrow_err)?;
                    acc = or(&acc, &matches).map_err(arrow_err)?;
                }
                Ok(acc)
            }
            None => self.all_false(),
        }
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _p: &BoundPredicate,
    ) -> Result<BooleanArray> {
        match self.column_for(reference) {
            Some(col) => {
                let mut acc = BooleanArray::from(vec![true; col.len()]);
                for literal in literals {
                    let lit = get_arrow_datum(literal)?;
                    let cast = try_cast_literal(&lit, col.data_type()).map_err(arrow_err)?;
                    let nonmatch = neq(&col, cast.as_ref()).map_err(arrow_err)?;
                    acc = and(&acc, &nonmatch).map_err(arrow_err)?;
                }
                Ok(acc)
            }
            None => self.all_true(),
        }
    }
}

#[cfg(test)]
#[path = "convert_equality_delete_files_tests.rs"]
mod tests;
