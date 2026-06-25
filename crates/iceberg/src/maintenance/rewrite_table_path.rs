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

//! `RewriteTablePath` (FULL-rewrite mode) — the engine-agnostic maintenance action that REWRITES every
//! absolute path prefix in a table's metadata graph from `source` to `target`, STAGING the rewritten
//! metadata (metadata.json + manifest-lists + manifests + position-delete CONTENT) at a caller-chosen
//! staging location and emitting a `(source, target)` COPY-PLAN for the caller to physically copy. The
//! Rust port of Java 1.10.0's engine-agnostic core `org.apache.iceberg.RewriteTablePathUtil`
//! (`replacePaths` / `rewriteManifestList` / `rewriteDataManifest` / `rewriteDeleteManifest` /
//! `rewritePositionDeleteFile`), the bytecode-verified iceberg-CORE surface behind the Spark
//! `RewriteTablePath` action.
//!
//! **This action does NOT physically copy data files.** It STAGES the rewritten metadata graph and
//! returns the copy-plan; the caller (or an external copier) is responsible for performing the copies
//! the plan names. The only payloads physically rewritten in place are position-delete CONTENT files
//! (their `file_path` column is path-rewritten, so they cannot be a verbatim copy) — these are written
//! into the staging location.
//!
//! # The Java contract this mirrors (javap-verified against `iceberg-core` 1.10.0)
//!
//! `RewriteTablePathUtil` is a static utility, NOT a Spark class — so this is a faithful CORE port, not
//! a Spark-surface approximation. The load-bearing methods:
//!
//! - `replacePaths(TableMetadata, source, target) -> TableMetadata` — rewrites the metadata.json fields:
//!   1. `location` via `String.replaceFirst(source, target)` — REGEX semantics, the ONLY field NOT
//!      using `newPath` (this asymmetry is mirrored EXACTLY: see [`replace_first_prefix`]).
//!   2. each snapshot: ONLY `manifest_list` via `newPath`; every other snapshot field verbatim.
//!   3. metadata-log entries `.file` via `newPath` (timestamp preserved).
//!   4. EXACTLY four properties IF present — `write.object-storage.path` / `write.folder-storage.path`
//!      / `write.data.path` / `write.metadata.path` — via `newPath`; all other properties untouched.
//!   5. `statisticsFiles` (Puffin) `.path` via `newPath`.
//!   6. DIVERGENCES MIRRORED EXACTLY: `partitionStatisticsFiles` is PASSED THROUGH UN-REWRITTEN in
//!      1.10.0 (verified at bytecode offset 142 — `partitionStatisticsFiles()` flows to the ctor
//!      unmodified); `encryptionKeys` / `refs` / `schemas` / `specs` / `sortOrders` verbatim; the
//!      rewritten metadata's `metadataFileLocation` is left null (the caller names the new file);
//!      `currentSnapshotId` carried.
//! - `rewriteManifestList(snapshot, io, metadata, manifestsToRewrite, source, target, staging, out)` —
//!   writes a NEW manifest-list in staging; each `ManifestFile` is copied then its `manifest_path` is
//!   set to `newPath` (pointing at TARGET). For each rewritten manifest, the copy-plan entry is
//!   `(stagingPath(origPath, source, staging), newPath(origPath, source, target))` — STAGED files copy
//!   FROM the staging location.
//! - DATA manifest (`writeDataFileEntry`): a [`DataFile`] whose location does NOT start with `source`
//!   is a precondition violation ("Encountered data file %s not under the source prefix %s"); otherwise
//!   `copy(df).withPath(newPath(loc))` preserves all other metadata. The copy-plan entry (live + in the
//!   snapshot set) is `(originalSourceLocation, newPath target)` — VERBATIM data copies FROM the source.
//! - DELETE manifest (`writeDeleteFileEntry`): POSITION_DELETES → path via `newPath`, bounds via
//!   `replacePathBounds` (the file_path-column lower/upper bound metrics are rewritten), the content is
//!   ADDED to `toRewrite` (physically rewritten), and the copy-plan entry is `(stagingPath(origLoc),
//!   newLoc)` — STAGED. EQUALITY_DELETES → path via `newPath`, content VERBATIM (NO content rewrite),
//!   copy-plan `(originalSourceLocation, newLoc)` — VERBATIM. Any other content → unsupported.
//! - POSITION-DELETE CONTENT (`rewritePositionDeleteFile`): physically read each pos-delete record,
//!   rewrite column 0 (`file_path`) via `newPath`, preserve column 1 (`pos`), write a new file into
//!   staging. The ONLY content-rewritten payload.
//! - `referenced_data_file` (the pos-delete / DV back-reference) is ALSO a path and is rewritten when
//!   present (a position-delete's `DataFile.referenced_data_file`).
//!
//! # The copy-plan direction, by class (the load-bearing asymmetry)
//!
//! The plan is a set of `(sourceToCopyFrom, targetToCopyTo)` pairs. The `sourceToCopyFrom` differs by
//! whether the payload was content-rewritten (STAGED) or carried verbatim:
//!
//! | class | content rewritten? | copy FROM | copy TO |
//! |---|---|---|---|
//! | manifest-list / manifest / position-delete | YES (staged) | `stagingPath(orig, source, staging)` | `newPath(orig, source, target)` |
//! | data file | no (verbatim) | `originalSourceLocation` | `newPath(orig, source, target)` |
//! | equality-delete | no (verbatim) | `originalSourceLocation` | `newPath(orig, source, target)` |
//!
//! So STAGED entries copy FROM the staging location (where this action wrote the rewritten bytes);
//! VERBATIM entries copy FROM the original source. Get this backwards and the copier reads the wrong
//! bytes — hence it is asserted directly in the offline tests and the interop oracle.
//!
//! # On-disk format stability
//!
//! Only path STRINGS change. The metadata.json is re-serialized through the SAME
//! [`TableMetadata::write_to`] codec; manifests/lists are re-emitted through the SAME
//! [`ManifestWriter`](crate::spec::ManifestWriter) / [`ManifestListWriter`](crate::spec::ManifestListWriter)
//! preserving every entry's status / sequence number / snapshot id (via [`reemit_entry`], which
//! dispatches on the original entry status exactly like Java's `appendEntryWithFile`); the
//! `format_version` is threaded from the source metadata. No encoding/format drift.
//!
//! # Deferred (loudly — additive later)
//!
//! - **Incremental mode** (Java `startVersion`/`endVersion` + the version-diff walk + the version-hint
//!   write): this port is FULL rewrite only (endVersion = current, all live files, no snapshot-id Set
//!   filter). The incremental version-diff is a Spark-shell concern.
//! - **The CSV file-list output**: Java's Spark layer writes a CSV; the CORE plan is `Set<(from, to)>`,
//!   returned here as [`RewriteTablePathResult::copy_plan`] (a `Vec<(String, String)>`).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::StreamExt;

use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::io::FileIO;
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, ManifestContentType,
    ManifestEntry, ManifestFile, ManifestListWriter, ManifestStatus, ManifestWriterBuilder,
    MetricsConfig, PrimitiveLiteral, Snapshot, TableMetadata,
};
use crate::table::Table;
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::DefaultFileNameGenerator;
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// The platform file separator Java's `RewriteTablePathUtil.FILE_SEPARATOR` uses — always `/` for
/// object-store / table-format paths (NOT the OS separator).
const FILE_SEPARATOR: &str = "/";

/// The four object-storage / folder-storage property keys Java's `updateProperties` path-rewrites IF
/// PRESENT (javap-verified literal order: object-storage, folder-storage, data, metadata). Every other
/// property is left untouched.
const PATH_PROPERTY_KEYS: [&str; 4] = [
    "write.object-storage.path",
    "write.folder-storage.path",
    "write.data.path",
    "write.metadata.path",
];

/// The outcome of a [`RewriteTablePath::execute`] run — the Rust analog of Java's
/// `RewriteTablePathUtil$RewriteResult` (the copy-plan) plus the staging location and the rewritten
/// metadata's logical version.
///
/// The action STAGES the rewritten metadata graph at [`Self::staging_location`] and returns the
/// [`Self::copy_plan`]; it does NOT physically copy data files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteTablePathResult {
    /// The staging directory the rewritten metadata graph was written under (the
    /// content-rewritten manifests / manifest-lists / position-deletes live here; the new
    /// metadata.json is at [`Self::staged_metadata_location`]).
    pub staging_location: String,
    /// The location of the newly-staged rewritten metadata.json (under the staging location). Java
    /// leaves `metadataFileLocation` null and the caller names the file; this port names it for the
    /// caller and reports the chosen location.
    pub staged_metadata_location: String,
    /// The `(sourceToCopyFrom, targetToCopyTo)` copy-plan — the Rust analog of Java
    /// `RewriteResult.copyPlan()`. Deterministically sorted. The caller (or an external copier) copies
    /// each `from` → `to`; this action does NOT perform the copies. See the module docs for the
    /// per-class direction.
    pub copy_plan: Vec<(String, String)>,
    /// The logical version of the rewritten metadata (the current snapshot id, or -1 for an empty
    /// table) — Java's FULL rewrite endVersion = current.
    pub latest_version: i64,
}

/// The `RewriteTablePath` maintenance action (FULL-rewrite mode). Build it with [`Self::new`], set the
/// prefixes with [`Self::rewrite_location_prefix`] and the staging directory with
/// [`Self::staging_location`], then run it with [`Self::execute`].
///
/// See the module docs for the full Java contract, the divergences mirrored, and the copy-plan
/// direction by class.
pub struct RewriteTablePath {
    table: Table,
    source_prefix: Option<String>,
    target_prefix: Option<String>,
    staging_location: Option<String>,
}

impl RewriteTablePath {
    /// Create a `RewriteTablePath` action for `table`. The source/target prefixes
    /// ([`Self::rewrite_location_prefix`]) and the staging location ([`Self::staging_location`]) MUST be
    /// set before [`Self::execute`].
    pub fn new(table: Table) -> Self {
        Self {
            table,
            source_prefix: None,
            target_prefix: None,
            staging_location: None,
        }
    }

    /// Set the absolute path prefixes to rewrite: every path starting with `source` is rewritten to
    /// start with `target` (Java's `sourcePrefix` / `targetPrefix`). REQUIRED.
    pub fn rewrite_location_prefix(
        mut self,
        source: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        self.source_prefix = Some(source.into());
        self.target_prefix = Some(target.into());
        self
    }

    /// Set the staging directory the rewritten metadata graph is written under (Java's
    /// `stagingLocation`). The content-rewritten manifests / manifest-lists / position-deletes and the
    /// new metadata.json land here. REQUIRED.
    pub fn staging_location(mut self, dir: impl Into<String>) -> Self {
        self.staging_location = Some(dir.into());
        self
    }

    /// Run the FULL rewrite: rewrite the metadata graph into the staging location with every absolute
    /// path prefix swapped `source` → `target`, and return the [`RewriteTablePathResult`] (the staging
    /// location + the `(source, target)` copy-plan). Does NOT physically copy data files.
    ///
    /// Returns `Err` (without staging anything) when the prefixes / staging location are unset, or when
    /// a referenced data file is not under the source prefix (the Java precondition).
    pub async fn execute(self, file_io: &FileIO) -> Result<RewriteTablePathResult> {
        let source = self.source_prefix.as_deref().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "RewriteTablePath: source/target prefixes must be set via rewrite_location_prefix()",
            )
        })?;
        let target = self.target_prefix.as_deref().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "RewriteTablePath: source/target prefixes must be set via rewrite_location_prefix()",
            )
        })?;
        let staging = self.staging_location.as_deref().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "RewriteTablePath: a staging_location() must be set",
            )
        })?;

        let metadata = self.table.metadata();

        let mut copy_plan: Vec<(String, String)> = Vec::new();

        // (1) Rewrite each snapshot's manifest list + every manifest it references into staging,
        //     accumulating the copy-plan for manifest-lists / manifests / data / delete files +
        //     physically rewriting position-delete content.
        for snapshot in metadata.snapshots() {
            self.rewrite_snapshot(snapshot, file_io, source, target, staging, &mut copy_plan)
                .await?;
        }

        // (2) Rewrite the metadata.json fields (replace_paths) and stage the new metadata.json.
        let rewritten_metadata = replace_paths(metadata, source, target)?;
        let staged_metadata_location = combine_paths(
            staging,
            &format!("{}-rewritten.metadata.json", metadata.uuid()),
        );
        rewritten_metadata
            .write_to(file_io, &staged_metadata_location)
            .await?;

        copy_plan.sort();
        copy_plan.dedup();

        Ok(RewriteTablePathResult {
            staging_location: staging.to_string(),
            staged_metadata_location,
            copy_plan,
            latest_version: metadata.current_snapshot_id().unwrap_or(-1),
        })
    }

    /// Rewrite ONE snapshot: stage a new manifest-list (every manifest path rewritten), and for each
    /// manifest stage a rewritten manifest (data or delete) — accumulating the copy-plan and physically
    /// rewriting position-delete content. Mirrors Java's per-snapshot `rewriteManifestList` +
    /// `rewriteDataManifest` / `rewriteDeleteManifest` composition for the FULL case.
    async fn rewrite_snapshot(
        &self,
        snapshot: &Snapshot,
        file_io: &FileIO,
        source: &str,
        target: &str,
        staging: &str,
        copy_plan: &mut Vec<(String, String)>,
    ) -> Result<()> {
        let metadata = self.table.metadata();
        let format_version = metadata.format_version();
        let manifest_list = snapshot.load_manifest_list(file_io, metadata).await?;

        // The rewritten manifest-list is staged at stagingPath(origManifestListPath). The new
        // manifest-list ENTRIES point manifest_path at the TARGET location (newPath).
        let orig_manifest_list = snapshot.manifest_list();
        let staged_manifest_list = staging_path(orig_manifest_list, source, staging)?;
        let manifest_list_output = file_io.new_output(&staged_manifest_list)?;

        let mut list_writer = build_manifest_list_writer(
            format_version,
            manifest_list_output,
            snapshot.snapshot_id(),
            snapshot.parent_snapshot_id(),
            snapshot.sequence_number(),
            snapshot.first_row_id(),
        );

        let mut rewritten_manifest_files: Vec<ManifestFile> = Vec::new();
        for manifest_file in manifest_list.entries() {
            let orig_manifest_path = manifest_file.manifest_path.clone();

            // Stage the rewritten manifest at stagingPath(origManifestPath); its entries' file paths
            // point at TARGET. Returns the rewritten ManifestFile (with manifest_path = TARGET) for the
            // manifest-list, and the per-content-file copy-plan entries it produced.
            let rewritten = self
                .rewrite_manifest(
                    manifest_file,
                    file_io,
                    source,
                    target,
                    staging,
                    snapshot.snapshot_id(),
                    copy_plan,
                )
                .await?;
            rewritten_manifest_files.push(rewritten);

            // The manifest itself is a STAGED (content-rewritten) file: copy FROM staging TO target.
            copy_plan.push((
                staging_path(&orig_manifest_path, source, staging)?,
                new_path(&orig_manifest_path, source, target)?,
            ));
        }

        list_writer.add_manifests(rewritten_manifest_files.into_iter())?;
        list_writer.close().await?;

        // The manifest-list itself is a STAGED file: copy FROM staging TO target.
        copy_plan.push((
            staged_manifest_list,
            new_path(orig_manifest_list, source, target)?,
        ));

        Ok(())
    }

    /// Rewrite ONE manifest into staging (data or delete) and return the rewritten [`ManifestFile`]
    /// (with `manifest_path` = TARGET) for inclusion in the staged manifest-list. Accumulates the
    /// per-content-file copy-plan entries into `copy_plan` and physically rewrites position-delete
    /// content. Mirrors Java's `rewriteDataManifest` / `rewriteDeleteManifest`.
    #[allow(clippy::too_many_arguments)]
    async fn rewrite_manifest(
        &self,
        manifest_file: &ManifestFile,
        file_io: &FileIO,
        source: &str,
        target: &str,
        staging: &str,
        snapshot_id: i64,
        copy_plan: &mut Vec<(String, String)>,
    ) -> Result<ManifestFile> {
        let metadata = self.table.metadata();
        let manifest = manifest_file.load_manifest(file_io).await?;
        let manifest_metadata = manifest.metadata();
        let format_version = *manifest_metadata.format_version();
        let schema = manifest_metadata.schema().clone();
        let partition_spec = manifest_metadata.partition_spec().clone();
        let content = *manifest_metadata.content();

        // Stage the rewritten manifest at stagingPath(origManifestPath); its added entries reference
        // TARGET paths. We use the source manifest's snapshot id as the writer's snapshot id (it is
        // only the fallback for entries with no explicit snapshot id; re-emitted entries carry theirs).
        let staged_manifest_path = staging_path(&manifest_file.manifest_path, source, staging)?;
        let output = file_io.new_output(&staged_manifest_path)?;
        let mut writer = build_manifest_writer(
            format_version,
            content,
            output,
            Some(snapshot_id),
            schema,
            partition_spec,
        );

        for entry in manifest.entries() {
            match content {
                ManifestContentType::Data => {
                    let rewritten_file = rewrite_data_file_path(entry.data_file(), source, target)?;
                    reemit_entry(&mut writer, entry, rewritten_file)?;
                    // Data files are VERBATIM: copy FROM the ORIGINAL SOURCE location TO target. Java
                    // adds the plan entry only for LIVE entries in the snapshot set (FULL ⇒ all).
                    if entry.is_alive() {
                        copy_plan.push((
                            entry.data_file().file_path().to_string(),
                            new_path(entry.data_file().file_path(), source, target)?,
                        ));
                    }
                }
                ManifestContentType::Deletes => {
                    self.rewrite_delete_entry(
                        &mut writer,
                        entry,
                        file_io,
                        source,
                        target,
                        staging,
                        copy_plan,
                    )
                    .await?;
                }
            }
        }

        let mut rewritten_manifest_file = writer.write_manifest_file().await?;
        // Java copies the ManifestFile and sets manifest_path = newPath(origPath) for the manifest-list
        // entry; the manifest-list entry must point at the TARGET, not the staging path.
        rewritten_manifest_file.manifest_path =
            new_path(&manifest_file.manifest_path, source, target)?;
        let _ = metadata; // metadata is the action's; kept for symmetry with the data-manifest path.

        Ok(rewritten_manifest_file)
    }

    /// Rewrite ONE delete-manifest entry: POSITION_DELETES → path + bounds rewritten, content physically
    /// rewritten into staging, copy-plan STAGED; EQUALITY_DELETES → path rewritten, content verbatim,
    /// copy-plan from SOURCE. Mirrors Java's `writeDeleteFileEntry`.
    #[allow(clippy::too_many_arguments)]
    async fn rewrite_delete_entry(
        &self,
        writer: &mut crate::spec::ManifestWriter,
        entry: &ManifestEntry,
        file_io: &FileIO,
        source: &str,
        target: &str,
        staging: &str,
        copy_plan: &mut Vec<(String, String)>,
    ) -> Result<()> {
        let delete_file = entry.data_file();
        match delete_file.content_type() {
            DataContentType::PositionDeletes => {
                let orig_location = delete_file.file_path().to_string();
                // Path via newPath + bounds via replacePathBounds + referenced_data_file via newPath.
                let rewritten_file =
                    rewrite_position_delete_file_metadata(delete_file, source, target)?;
                let new_location = rewritten_file.file_path().to_string();

                // Physically rewrite the position-delete CONTENT (column 0 file_path) into staging.
                // The content is the ONLY rewritten payload. A Puffin deletion vector cannot be
                // record-rewritten by the parquet pos-delete writer; defer it loudly. Returns the EXACT
                // staged location the content landed at (the copy-plan's "from" for this STAGED file).
                let staged_content_path = if delete_file.file_format() == DataFileFormat::Parquet {
                    self.rewrite_position_delete_content(
                        delete_file,
                        file_io,
                        source,
                        target,
                        staging,
                    )
                    .await?
                } else {
                    return Err(Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!(
                            "RewriteTablePath: position-delete content rewrite for non-parquet \
                             format {:?} (e.g. a Puffin deletion vector) is not yet supported \
                             (file {orig_location})",
                            delete_file.file_format()
                        ),
                    ));
                };

                reemit_entry(writer, entry, rewritten_file)?;

                // POSITION_DELETES are STAGED (content-rewritten): copy FROM the staged content
                // location TO newLoc. Java adds the plan entry only for LIVE entries in the snapshot
                // set (FULL ⇒ all).
                if entry.is_alive() {
                    copy_plan.push((staged_content_path, new_location));
                }
            }
            DataContentType::EqualityDeletes => {
                let orig_location = delete_file.file_path().to_string();
                // Path via newPath; CONTENT VERBATIM (no rewrite).
                let rewritten_file = rewrite_data_file_path(delete_file, source, target)?;
                let new_location = rewritten_file.file_path().to_string();
                reemit_entry(writer, entry, rewritten_file)?;
                // EQUALITY_DELETES are VERBATIM: copy FROM the ORIGINAL SOURCE location TO newLoc.
                if entry.is_alive() {
                    copy_plan.push((orig_location, new_location));
                }
            }
            DataContentType::Data => {
                // Java's writeDeleteFileEntry default arm throws UnsupportedOperationException — a Data
                // entry in a DELETE manifest is malformed.
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "RewriteTablePath: a Data-content file {} appeared in a DELETE manifest \
                         (Java's writeDeleteFileEntry rejects this)",
                        delete_file.file_path()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Physically rewrite a parquet position-delete file's CONTENT into `staged_content_path`: read each
    /// `(file_path, pos)` record, rewrite `file_path` via `newPath`, preserve `pos`, and write a new
    /// parquet position-delete file. Mirrors Java's `rewritePositionDeleteFile` + `newPositionDeleteRecord`
    /// (the ONLY content-rewritten payload). The row column (Java col 2) is not carried (this fork's
    /// pos-delete writer is `(file_path, pos)`).
    async fn rewrite_position_delete_content(
        &self,
        delete_file: &DataFile,
        file_io: &FileIO,
        source: &str,
        target: &str,
        staging: &str,
    ) -> Result<String> {
        // Read the original (file_path, pos) pairs and rewrite each file_path.
        let loader = BasicDeleteFileLoader::new(file_io.clone());
        let mut stream = loader
            .parquet_to_batch_stream(delete_file.file_path(), delete_file.file_size_in_bytes)
            .await?;

        let mut rewritten_pairs: Vec<(String, i64)> = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let (path_col, pos_col) = locate_reserved_columns(&batch, delete_file.file_path())?;
            for row in 0..batch.num_rows() {
                if path_col.is_null(row) || pos_col.is_null(row) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "RewriteTablePath: position delete '{}' has a null file_path/pos at \
                             row {row}",
                            delete_file.file_path()
                        ),
                    ));
                }
                let old_path = path_col.value(row);
                let new_referenced = new_path(old_path, source, target)?;
                rewritten_pairs.push((new_referenced, pos_col.value(row)));
            }
        }

        // Write the rewritten records into the staging location and return the EXACT location the
        // content landed at (the copy-plan's "from" for this STAGED file).
        self.write_position_delete_content(delete_file, &rewritten_pairs, source, staging)
            .await
    }

    /// Write the rewritten `(file_path, pos)` pairs into a parquet position-delete file UNDER the
    /// staging location (mirroring the source file's source-relative directory), returning the EXACT
    /// location the content landed at. The returned path is the copy-plan's "from" for this STAGED file
    /// and is the location the rewritten content is read from by the copier.
    async fn write_position_delete_content(
        &self,
        delete_file: &DataFile,
        pairs: &[(String, i64)],
        source: &str,
        staging: &str,
    ) -> Result<String> {
        let config = PositionDeleteWriterConfig::new()?;

        // The staged content lands at EXACTLY stagingPath(origLoc) — the SAME source-relative path under
        // the staging location that Java's `RewriteTablePathUtil` uses (so the copy-plan's "from" tag is
        // deterministic + cross-engine-comparable). The location generator returns that exact path and
        // ignores the writer's generated file name; a dummy name generator satisfies the writer API.
        let staged_content_path = staging_path(delete_file.file_path(), source, staging)?;
        let location_gen = StagedLocationGenerator {
            exact_path: staged_content_path,
        };
        let file_name_gen = DefaultFileNameGenerator::new(
            "rewritten-pos-del".to_string(),
            None,
            DataFileFormat::Parquet,
        );
        // The rewritten position-delete content keeps `file_path`/`pos` bounds FULL (Java
        // `MetricsConfig.forPositionDelete`) so delete-file path pruning stays precise — the default
        // `truncate(16)` would widen the path range.
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
        // Position deletes carry their partition in the manifest entry (not the parquet rows), so the
        // content file needs no partition key.
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
            .build(None)
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
                "RewriteTablePath: failed to build rewritten position-delete record batch",
            )
            .with_source(e)
        })?;
        writer.write(batch).await?;
        let written = writer.close().await?;
        let staged_file = written.into_iter().next().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "RewriteTablePath: position-delete content writer produced no file",
            )
        })?;
        Ok(staged_file.file_path().to_string())
    }
}

/// ============================================================================================
/// `replace_paths` — the metadata.json field rewrite (Java `RewriteTablePathUtil.replacePaths`).
/// ============================================================================================
///
/// Rewrites `location` (regex `replaceFirst`), each snapshot's `manifest_list` (newPath), metadata-log
/// `.file` (newPath), the four path PROPERTIES (newPath), and `statisticsFiles.path` (newPath). Passes
/// `partition_statistics` through UN-REWRITTEN (the 1.10.0 divergence), and carries everything else
/// (`refs`, `schemas`, `specs`, `sortOrders`, `encryption_keys`, ids, sequence numbers) verbatim.
pub(crate) fn replace_paths(
    metadata: &TableMetadata,
    source: &str,
    target: &str,
) -> Result<TableMetadata> {
    // Clone the whole metadata, then mutate ONLY the path-bearing fields. Cloning preserves every
    // verbatim field exactly (refs/schemas/specs/sortOrders/encryptionKeys/ids/seqs) — the conservative
    // mirror of Java's reconstruct-with-most-fields-carried.
    let mut rewritten = metadata.clone();

    // (1) location via String.replaceFirst (REGEX) — the ONLY field NOT using newPath. Mirror exactly.
    rewritten.location = replace_first_prefix(&metadata.location, source, target);

    // (2) snapshots: ONLY manifest_list via newPath; every other snapshot field verbatim.
    let mut new_snapshots = HashMap::with_capacity(metadata.snapshots.len());
    for (id, snapshot) in &metadata.snapshots {
        let mut s = snapshot.as_ref().clone();
        s.manifest_list = new_path(&snapshot.manifest_list, source, target)?;
        new_snapshots.insert(*id, Arc::new(s));
    }
    rewritten.snapshots = new_snapshots;

    // (3) metadata-log entries .file via newPath (timestamp preserved).
    for entry in rewritten.metadata_log.iter_mut() {
        entry.metadata_file = new_path(&entry.metadata_file, source, target)?;
    }

    // (4) EXACTLY four properties IF present — via newPath; all other properties untouched.
    for key in PATH_PROPERTY_KEYS {
        if let Some(value) = rewritten.properties.get(key) {
            let rewritten_value = new_path(value, source, target)?;
            rewritten
                .properties
                .insert(key.to_string(), rewritten_value);
        }
    }

    // (5) statisticsFiles (Puffin) .path via newPath.
    for stats in rewritten.statistics.values_mut() {
        stats.statistics_path = new_path(&stats.statistics_path, source, target)?;
    }

    // DIVERGENCE: partition_statistics PASSED THROUGH UN-REWRITTEN (the 1.10.0 behavior). The clone
    // already carried them verbatim — do NOT touch them.

    Ok(rewritten)
}

// ============================================================================================
// Path helpers — faithful ports of Java's `RewriteTablePathUtil` path math.
// ============================================================================================

/// `newPath(path, sourcePrefix, targetPrefix)` = `combinePaths(target, relativize(path, source))`.
/// Errors (Java throws `IllegalArgumentException`) if `path` does not start with `source`.
fn new_path(path: &str, source: &str, target: &str) -> Result<String> {
    let rel = relativize(path, source)?;
    Ok(combine_paths(target, &rel))
}

/// `relativize(path, prefix)` = `path.substring(maybeAppendFileSeparator(prefix).length())`. Errors if
/// `path` does not start with the separator-appended prefix (Java's "Path %s does not start with %s").
fn relativize(path: &str, prefix: &str) -> Result<String> {
    let with_sep = maybe_append_file_separator(prefix);
    path.strip_prefix(&with_sep)
        .map(|rest| rest.to_string())
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("RewriteTablePath: path {path} does not start with {with_sep}"),
            )
        })
}

/// `combinePaths(prefix, suffix)` = `maybeAppendFileSeparator(prefix) + suffix`.
fn combine_paths(prefix: &str, suffix: &str) -> String {
    format!("{}{}", maybe_append_file_separator(prefix), suffix)
}

/// `maybeAppendFileSeparator(prefix)` — appends `/` only if `prefix` lacks a trailing `/`.
fn maybe_append_file_separator(prefix: &str) -> String {
    if prefix.ends_with(FILE_SEPARATOR) {
        prefix.to_string()
    } else {
        format!("{prefix}{FILE_SEPARATOR}")
    }
}

/// `stagingPath(origPath, sourcePrefix, stagingDir)` (the 3-arg form) =
/// `combinePaths(stagingDir, relativize(origPath, sourcePrefix))`. The staged location of a
/// content-rewritten file mirrors its source-relative path under the staging dir.
fn staging_path(orig_path: &str, source: &str, staging_dir: &str) -> Result<String> {
    let rel = relativize(orig_path, source)?;
    Ok(combine_paths(staging_dir, &rel))
}

/// `String.replaceFirst(sourcePrefix, targetPrefix)` on `location` — Java REGEX semantics, the ONLY
/// field of `replacePaths` that does NOT use `newPath`. This asymmetry (regex-replace-first vs the
/// path-aware `newPath` everywhere else) is mirrored EXACTLY: the FIRST occurrence of `source` in
/// `location` is replaced once by `target`.
///
/// Java's `source` is a REGEX. The action's `source` is always an absolute PATH PREFIX, which contains
/// no regex metacharacters in practice, so for these inputs `String.replaceFirst` is exactly a literal
/// first-occurrence replace — implemented directly here (the `regex` crate is a dev-only dependency and
/// MUST NOT be pulled into the library surface). Regex metacharacters in a path prefix are not a
/// supported input (Java would interpret them as a pattern; the precondition that data files start with
/// the literal `source` is enforced by `newPath`/`relativize` everywhere else, which would reject them).
fn replace_first_prefix(location: &str, source: &str, target: &str) -> String {
    match location.find(source) {
        Some(idx) => {
            let mut out = String::with_capacity(location.len() - source.len() + target.len());
            out.push_str(&location[..idx]);
            out.push_str(target);
            out.push_str(&location[idx + source.len()..]);
            out
        }
        None => location.to_string(),
    }
}

// ============================================================================================
// DataFile / DeleteFile path rewrite helpers.
// ============================================================================================

/// Rebuild a [`DataFile`] with `file_path` rewritten via `newPath` and ALL other metadata preserved
/// (Java's `copy(df).withPath(newPath(loc))`). Used for DATA files and (path-only) EQUALITY deletes.
/// Errors if the file location is not under the source prefix (Java's precondition).
fn rewrite_data_file_path(data_file: &DataFile, source: &str, target: &str) -> Result<DataFile> {
    if !data_file.file_path().starts_with(source) {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "RewriteTablePath: encountered data file {} not under the source prefix {source}",
                data_file.file_path()
            ),
        ));
    }
    let mut rewritten = data_file.clone();
    rewritten.file_path = new_path(data_file.file_path(), source, target)?;
    // referenced_data_file is ALSO a path — rewrite it when present (the pos-delete / DV back-reference).
    if let Some(referenced) = &data_file.referenced_data_file {
        rewritten.referenced_data_file = Some(new_path(referenced, source, target)?);
    }
    Ok(rewritten)
}

/// Rebuild a POSITION-DELETE [`DataFile`] with: `file_path` rewritten via `newPath`; the
/// `referenced_data_file` back-reference rewritten via `newPath` when present; and the file_path-column
/// lower/upper BOUNDS rewritten via `replacePathBounds`. Java's POSITION_DELETES branch of
/// `writeDeleteFileEntry`. Errors if the location is not under the source prefix.
fn rewrite_position_delete_file_metadata(
    delete_file: &DataFile,
    source: &str,
    target: &str,
) -> Result<DataFile> {
    // path + referenced_data_file via newPath (also asserts the source-prefix precondition).
    let mut rewritten = rewrite_data_file_path(delete_file, source, target)?;
    replace_path_bounds(&mut rewritten, source, target)?;
    Ok(rewritten)
}

/// `ContentFileUtil.replacePathBounds(deleteFile, source, target)` — rewrites the file_path-column
/// (reserved field id [`RESERVED_FIELD_ID_DELETE_FILE_PATH`]) lower/upper bound metrics. If either
/// bound is absent, or the lower != upper bound (the delete references MORE than one data file), the
/// bounds are CLEARED (Java's `metricsWithoutPathBounds`). Only when lower == upper (a single
/// referenced data file) are both rewritten to `newPath(decoded)`.
fn replace_path_bounds(delete_file: &mut DataFile, source: &str, target: &str) -> Result<()> {
    let lower = delete_file
        .lower_bounds
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH);
    let upper = delete_file
        .upper_bounds
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH);

    let (Some(lower), Some(upper)) = (lower, upper) else {
        // Java: if either bound is null, return metricsWithoutPathBounds — drop the path bounds.
        delete_file
            .lower_bounds
            .remove(&RESERVED_FIELD_ID_DELETE_FILE_PATH);
        delete_file
            .upper_bounds
            .remove(&RESERVED_FIELD_ID_DELETE_FILE_PATH);
        return Ok(());
    };

    let lower_str = datum_as_string(lower);
    let upper_str = datum_as_string(upper);

    match (lower_str, upper_str) {
        (Some(l), Some(u)) if l == u => {
            // Single referenced data file — rewrite both bounds to newPath(decoded).
            let rewritten = new_path(&l, source, target)?;
            delete_file.lower_bounds.insert(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string(&rewritten),
            );
            delete_file.upper_bounds.insert(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string(&rewritten),
            );
        }
        _ => {
            // lower != upper (spans multiple files) or non-string bounds — drop the path bounds
            // (Java's metricsWithoutPathBounds).
            delete_file
                .lower_bounds
                .remove(&RESERVED_FIELD_ID_DELETE_FILE_PATH);
            delete_file
                .upper_bounds
                .remove(&RESERVED_FIELD_ID_DELETE_FILE_PATH);
        }
    }
    Ok(())
}

/// Decode a [`Datum`]'s string value (the file_path bound is a string). Returns `None` for a non-string.
fn datum_as_string(datum: &Datum) -> Option<String> {
    match datum.literal() {
        PrimitiveLiteral::String(value) => Some(value.clone()),
        _ => None,
    }
}

// ============================================================================================
// Identity-preserving manifest entry re-emission (Java `appendEntryWithFile`).
// ============================================================================================

/// Re-emit `entry` (with its `data_file` swapped for `new_file`) into `writer`, DISPATCHING on the
/// original entry status exactly like Java's `appendEntryWithFile`: ADDED → `add_file`, EXISTING →
/// `add_existing_file` (preserving snapshot id + seq), DELETED → `add_delete_file` (preserving seq). The
/// status / sequence numbers / snapshot ids survive the rewrite — only the path changed.
fn reemit_entry(
    writer: &mut crate::spec::ManifestWriter,
    entry: &ManifestEntry,
    new_file: DataFile,
) -> Result<()> {
    match entry.status() {
        ManifestStatus::Added => {
            // Java's writer.add() — the data sequence number is assigned at commit; a live ADDED entry
            // carries its post-inheritance seq, which we preserve via add_file.
            let seq = entry.sequence_number().unwrap_or(0);
            writer.add_file(new_file, seq)?;
        }
        ManifestStatus::Existing => {
            let snapshot_id = entry.snapshot_id().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "RewriteTablePath: an EXISTING manifest entry must carry a snapshot id",
                )
            })?;
            let seq = entry.sequence_number().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "RewriteTablePath: an EXISTING manifest entry must carry a sequence number",
                )
            })?;
            writer.add_existing_file(new_file, snapshot_id, seq, entry.file_sequence_number)?;
        }
        ManifestStatus::Deleted => {
            let seq = entry.sequence_number().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "RewriteTablePath: a DELETED manifest entry must carry a sequence number",
                )
            })?;
            writer.add_delete_file(new_file, seq, entry.file_sequence_number)?;
        }
    }
    Ok(())
}

// ============================================================================================
// Manifest / manifest-list writer construction (format-version-threaded).
// ============================================================================================

/// Build a [`ManifestListWriter`] for `format_version`, threading the snapshot identity.
fn build_manifest_list_writer(
    format_version: FormatVersion,
    output: crate::io::OutputFile,
    snapshot_id: i64,
    parent_snapshot_id: Option<i64>,
    sequence_number: i64,
    first_row_id: Option<u64>,
) -> ManifestListWriter {
    match format_version {
        FormatVersion::V1 => ManifestListWriter::v1(output, snapshot_id, parent_snapshot_id),
        FormatVersion::V2 => {
            ManifestListWriter::v2(output, snapshot_id, parent_snapshot_id, sequence_number)
        }
        FormatVersion::V3 => ManifestListWriter::v3(
            output,
            snapshot_id,
            parent_snapshot_id,
            sequence_number,
            first_row_id,
        ),
    }
}

/// Build a [`ManifestWriter`](crate::spec::ManifestWriter) for `format_version` + `content`, threading
/// the schema + partition spec.
fn build_manifest_writer(
    format_version: FormatVersion,
    content: ManifestContentType,
    output: crate::io::OutputFile,
    snapshot_id: Option<i64>,
    schema: crate::spec::SchemaRef,
    partition_spec: crate::spec::PartitionSpec,
) -> crate::spec::ManifestWriter {
    let builder = ManifestWriterBuilder::new(output, snapshot_id, None, schema, partition_spec);
    match (format_version, content) {
        (FormatVersion::V1, _) => builder.build_v1(),
        (FormatVersion::V2, ManifestContentType::Data) => builder.build_v2_data(),
        (FormatVersion::V2, ManifestContentType::Deletes) => builder.build_v2_deletes(),
        (FormatVersion::V3, ManifestContentType::Data) => builder.build_v3_data(),
        (FormatVersion::V3, ManifestContentType::Deletes) => builder.build_v3_deletes(),
    }
}

/// Locate the `file_path` (string) and `pos` (int64) columns of a position-delete record batch by their
/// RESERVED FIELD IDs (`PARQUET_FIELD_ID_META_KEY` metadata), not by name. Mirrors the same helper in
/// [`crate::maintenance::RewritePositionDeleteFiles`].
fn locate_reserved_columns<'a>(
    batch: &'a RecordBatch,
    file_path: &str,
) -> Result<(&'a StringArray, &'a Int64Array)> {
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

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
                "RewriteTablePath: position delete '{file_path}' is missing the reserved file_path \
                 column (field id {RESERVED_FIELD_ID_DELETE_FILE_PATH})"
            ),
        )
    })?;
    let pos_idx = pos_idx.ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!(
                "RewriteTablePath: position delete '{file_path}' is missing the reserved pos column \
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
                format!(
                    "RewriteTablePath: position delete '{file_path}' file_path column is not a \
                     string array"
                ),
            )
        })?;
    let pos_col = batch
        .column(pos_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "RewriteTablePath: position delete '{file_path}' pos column is not an int64 \
                     array"
                ),
            )
        })?;

    Ok((path_col, pos_col))
}

/// A [`LocationGenerator`](crate::writer::file_writer::location_generator::LocationGenerator) that emits
/// a FIXED, EXACT path (`stagingPath(origLoc)`), so the rewritten position-delete content lands at the
/// precise staged location the copy-plan + the staged manifest reference — IGNORING the writer's
/// generated file name. This keeps the staged path deterministic and cross-engine-comparable (Java uses
/// the same `stagingPath` layout).
#[derive(Clone)]
struct StagedLocationGenerator {
    exact_path: String,
}

impl crate::writer::file_writer::location_generator::LocationGenerator for StagedLocationGenerator {
    fn generate_location(
        &self,
        _partition_key: Option<&crate::spec::PartitionKey>,
        _file_name: &str,
    ) -> String {
        self.exact_path.clone()
    }
}

#[cfg(test)]
#[path = "rewrite_table_path_tests.rs"]
mod tests;
