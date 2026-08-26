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

//! `RewriteTablePath` in FULL-rewrite mode. It rewrites every absolute path prefix in a table's
//! metadata graph from `source` to `target`, stages the rewritten metadata, and returns a
//! `(from, to)` copy plan. Rust port of Java 1.10.0 `org.apache.iceberg.RewriteTablePathUtil`.
//!
//! The action never copies a data file. The caller performs every copy the plan names.
//! Position-delete content is the only payload rewritten in place, because its `file_path` column
//! holds a path.
//!
//! # What `replace_paths` rewrites in metadata.json
//!
//! | field | rewrite |
//! |---|---|
//! | `location` | `String.replaceFirst` regex semantics, not `newPath` (see [`replace_first_prefix`]) |
//! | snapshot `manifest_list` | `newPath`; every other snapshot field verbatim |
//! | metadata-log `.file` | `newPath`; timestamp preserved |
//! | `write.{object-storage,folder-storage,data,metadata}.path` | `newPath` if present; other properties untouched |
//! | `statisticsFiles.path` | `newPath` |
//! | `partitionStatisticsFiles` | not rewritten in 1.10.0; the fork mirrors that |
//! | `encryptionKeys`, `refs`, `schemas`, `specs`, `sortOrders` | verbatim |
//!
//! The rewritten metadata leaves `metadataFileLocation` null. Java lets the caller name the new file.
//!
//! # The copy-plan direction, by class
//!
//! A staged entry copies FROM the staging location, where this action wrote the rewritten bytes. A
//! verbatim entry copies FROM the original source. Reverse the two and the copier reads the wrong
//! bytes. The offline tests and the interop oracle assert the direction directly.
//!
//! | class | content rewritten? | copy FROM | copy TO |
//! |---|---|---|---|
//! | manifest-list / manifest / position-delete | yes, staged | `stagingPath(orig, source, staging)` | `newPath(orig, source, target)` |
//! | data file | no | `originalSourceLocation` | `newPath(orig, source, target)` |
//! | equality-delete | no | `originalSourceLocation` | `newPath(orig, source, target)` |
//!
//! A data file not under `source` is a precondition violation. A delete manifest also rewrites the
//! position-delete `file_path` bounds and the `referenced_data_file` back-reference. Any other
//! content type in a delete manifest is unsupported.
//!
//! # On-disk format stability
//!
//! Only path strings change. The metadata.json, manifest lists, and manifests re-serialize through
//! the same codecs, with the format version threaded from the source metadata. [`reemit_entry`]
//! preserves each entry's status, sequence number, and snapshot id.
//!
//! # Deferred
//!
//! Incremental mode (Java `startVersion`/`endVersion` and the version-diff walk) is a Spark-shell
//! concern; this port is full rewrite only. Java's Spark layer writes a CSV file list; the core plan
//! is [`RewriteTablePathResult::copy_plan`].

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
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig, position_delete_writer_properties,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::DefaultFileNameGenerator;
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Java `RewriteTablePathUtil.FILE_SEPARATOR`. Table paths always use `/`, never the OS separator.
const FILE_SEPARATOR: &str = "/";

/// The only property keys Java's `updateProperties` path-rewrites, in Java's order. Every other
/// property is left untouched.
const PATH_PROPERTY_KEYS: [&str; 4] = [
    "write.object-storage.path",
    "write.folder-storage.path",
    "write.data.path",
    "write.metadata.path",
];

/// The outcome of a [`RewriteTablePath::execute`] run. Java's
/// `RewriteTablePathUtil$RewriteResult` plus the staging location and the logical version.
///
/// The action stages the rewritten metadata graph and returns the plan. It copies no data file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteTablePathResult {
    /// The directory the rewritten manifests, manifest lists, and position deletes were written
    /// under.
    pub staging_location: String,
    /// The staged rewritten metadata.json. Java leaves `metadataFileLocation` null for the caller
    /// to name. This port names the file and reports the location it chose.
    pub staged_metadata_location: String,
    /// The sorted `(from, to)` copy plan (Java `RewriteResult.copyPlan()`). The caller performs
    /// each copy. The module docs give the direction per class.
    pub copy_plan: Vec<(String, String)>,
    /// The current snapshot id, or -1 for an empty table. Java's full-rewrite `endVersion`.
    pub latest_version: i64,
}

/// The `RewriteTablePath` maintenance action, in full-rewrite mode. Build it with [`Self::new`],
/// set [`Self::rewrite_location_prefix`] and [`Self::staging_location`], then call
/// [`Self::execute`]. The module docs carry the Java contract and the copy-plan direction.
pub struct RewriteTablePath {
    table: Table,
    source_prefix: Option<String>,
    target_prefix: Option<String>,
    staging_location: Option<String>,
}

impl RewriteTablePath {
    /// Creates the action for `table`. [`Self::rewrite_location_prefix`] and
    /// [`Self::staging_location`] must be set before [`Self::execute`].
    pub fn new(table: Table) -> Self {
        Self {
            table,
            source_prefix: None,
            target_prefix: None,
            staging_location: None,
        }
    }

    /// Sets the absolute path prefixes to rewrite (Java `sourcePrefix` / `targetPrefix`). Required.
    pub fn rewrite_location_prefix(
        mut self,
        source: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        self.source_prefix = Some(source.into());
        self.target_prefix = Some(target.into());
        self
    }

    /// Sets the directory the rewritten metadata graph is written under (Java `stagingLocation`).
    /// Required.
    pub fn staging_location(mut self, dir: impl Into<String>) -> Self {
        self.staging_location = Some(dir.into());
        self
    }

    /// Runs the full rewrite and returns the staging location and the copy plan. It copies no data
    /// file.
    ///
    /// # Errors
    ///
    /// Fails when the prefixes or the staging location are unset, or when a referenced data file is
    /// not under the source prefix.
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

        for snapshot in metadata.snapshots() {
            self.rewrite_snapshot(snapshot, file_io, source, target, staging, &mut copy_plan)
                .await?;
        }

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

    /// Rewrites one snapshot: a staged manifest list, plus a staged manifest per entry. Mirrors
    /// Java `rewriteManifestList` over `rewriteDataManifest` / `rewriteDeleteManifest`.
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

        // The list is staged at stagingPath, but its entries point at the target.
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

            // The manifest is content-rewritten, so it copies from staging.
            copy_plan.push((
                staging_path(&orig_manifest_path, source, staging)?,
                new_path(&orig_manifest_path, source, target)?,
            ));
        }

        list_writer.add_manifests(rewritten_manifest_files.into_iter())?;
        list_writer.close().await?;

        // The manifest list is content-rewritten, so it copies from staging.
        copy_plan.push((
            staged_manifest_list,
            new_path(orig_manifest_list, source, target)?,
        ));

        Ok(())
    }

    /// Rewrites one manifest into staging and returns it with `manifest_path` set to the target,
    /// for the staged manifest list. Mirrors Java `rewriteDataManifest` / `rewriteDeleteManifest`.
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

        // The writer's snapshot id is only the fallback for an entry that carries none. Every
        // re-emitted entry carries its own.
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
                    // A data file is verbatim, so it copies from the source. Java plans only live
                    // entries.
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
        // The manifest-list entry must point at the target, not at the staging path.
        rewritten_manifest_file.manifest_path =
            new_path(&manifest_file.manifest_path, source, target)?;
        let _ = metadata; // metadata is the action's; kept for symmetry with the data-manifest path.

        Ok(rewritten_manifest_file)
    }

    /// Rewrites one delete-manifest entry. Java `writeDeleteFileEntry`. A position delete has its
    /// content rewritten into staging; an equality delete is verbatim.
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
                let rewritten_file =
                    rewrite_position_delete_file_metadata(delete_file, source, target)?;
                let new_location = rewritten_file.file_path().to_string();

                // The parquet pos-delete writer cannot rewrite a Puffin deletion vector record by
                // record, so a non-parquet delete fails loudly here. The returned path is the copy
                // plan's `from` for this staged file.
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

                // A position delete is content-rewritten, so it copies from staging. Java plans
                // only live entries.
                if entry.is_alive() {
                    copy_plan.push((staged_content_path, new_location));
                }
            }
            DataContentType::EqualityDeletes => {
                let orig_location = delete_file.file_path().to_string();
                let rewritten_file = rewrite_data_file_path(delete_file, source, target)?;
                let new_location = rewritten_file.file_path().to_string();
                reemit_entry(writer, entry, rewritten_file)?;
                // An equality delete is verbatim, so it copies from the source.
                if entry.is_alive() {
                    copy_plan.push((orig_location, new_location));
                }
            }
            DataContentType::Data => {
                // A data entry in a delete manifest is malformed. Java throws here too.
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

    /// Rewrites a parquet position-delete file's content into staging. Java
    /// `rewritePositionDeleteFile`. This fork's writer emits `(file_path, pos)` only, so Java's
    /// third `row` column is not carried.
    async fn rewrite_position_delete_content(
        &self,
        delete_file: &DataFile,
        file_io: &FileIO,
        source: &str,
        target: &str,
        staging: &str,
    ) -> Result<String> {
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

        self.write_position_delete_content(delete_file, &rewritten_pairs, source, staging)
            .await
    }

    /// Writes the rewritten pairs into a parquet position-delete file under the staging location,
    /// at the source-relative path. The returned location is the copy plan's `from`, which the
    /// copier reads.
    async fn write_position_delete_content(
        &self,
        delete_file: &DataFile,
        pairs: &[(String, i64)],
        source: &str,
        staging: &str,
    ) -> Result<String> {
        let config = PositionDeleteWriterConfig::new()?;

        // The content must land at exactly stagingPath(origLoc), the layout Java uses, so the copy
        // plan is comparable across engines. The location generator forces that path and ignores
        // the generated file name.
        let staged_content_path = staging_path(delete_file.file_path(), source, staging)?;
        let location_gen = StagedLocationGenerator {
            exact_path: staged_content_path,
        };
        let file_name_gen = DefaultFileNameGenerator::new(
            "rewritten-pos-del".to_string(),
            None,
            DataFileFormat::Parquet,
        );
        // Full bounds keep delete-file path pruning precise (Java
        // `MetricsConfig.forPositionDelete`). The default `truncate(16)` widens the path range.
        let parquet_builder =
            ParquetWriterBuilder::new(position_delete_writer_properties(), config.schema().clone())
                .with_metrics_config(MetricsConfig::for_position_delete());
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            self.table.file_io().clone(),
            location_gen,
            file_name_gen,
        );
        // A position delete carries its partition in the manifest entry, not in the rows.
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

/// The metadata.json field rewrite (Java `RewriteTablePathUtil.replacePaths`). The module docs
/// table names every field this touches and every field it carries verbatim.
pub(crate) fn replace_paths(
    metadata: &TableMetadata,
    source: &str,
    target: &str,
) -> Result<TableMetadata> {
    // Clone, then mutate only the path-bearing fields. A clone carries every verbatim field
    // exactly, where Java's reconstruct-and-carry could drop a new one.
    let mut rewritten = metadata.clone();

    // `location` is the only field Java rewrites by regex, not by newPath.
    rewritten.location = replace_first_prefix(&metadata.location, source, target);

    let mut new_snapshots = HashMap::with_capacity(metadata.snapshots.len());
    for (id, snapshot) in &metadata.snapshots {
        let mut s = snapshot.as_ref().clone();
        s.manifest_list = new_path(&snapshot.manifest_list, source, target)?;
        new_snapshots.insert(*id, Arc::new(s));
    }
    rewritten.snapshots = new_snapshots;

    for entry in rewritten.metadata_log.iter_mut() {
        entry.metadata_file = new_path(&entry.metadata_file, source, target)?;
    }

    for key in PATH_PROPERTY_KEYS {
        if let Some(value) = rewritten.properties.get(key) {
            let rewritten_value = new_path(value, source, target)?;
            rewritten
                .properties
                .insert(key.to_string(), rewritten_value);
        }
    }

    for stats in rewritten.statistics.values_mut() {
        stats.statistics_path = new_path(&stats.statistics_path, source, target)?;
    }

    // Java 1.10.0 does not rewrite partition_statistics. The clone carried them. Leave them.

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

/// `relativize(path, prefix)`. Errors if `path` does not start with the separator-appended prefix.
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

/// `stagingPath(origPath, sourcePrefix, stagingDir)`. A content-rewritten file stages at its
/// source-relative path under the staging directory.
fn staging_path(orig_path: &str, source: &str, staging_dir: &str) -> Result<String> {
    let rel = relativize(orig_path, source)?;
    Ok(combine_paths(staging_dir, &rel))
}

/// `String.replaceFirst(sourcePrefix, targetPrefix)` on `location`. This is the only field of
/// `replacePaths` that does not use `newPath`, and the asymmetry is deliberate.
///
/// Java treats `source` as a regex. An absolute path prefix carries no metacharacter in practice, so
/// a literal first-occurrence replace matches Java for every supported input. That avoids pulling
/// `regex`, a dev-only dependency, into the library. A prefix with metacharacters is unsupported.
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

/// Rebuilds a [`DataFile`] with `file_path` rewritten and all other metadata preserved (Java
/// `copy(df).withPath(newPath(loc))`). Errors if the location is not under the source prefix.
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
    // referenced_data_file is a path too.
    if let Some(referenced) = &data_file.referenced_data_file {
        rewritten.referenced_data_file = Some(new_path(referenced, source, target)?);
    }
    Ok(rewritten)
}

/// Rebuilds a position-delete [`DataFile`] with its path, its `referenced_data_file`, and its
/// file_path-column bounds rewritten. Java's POSITION_DELETES branch of `writeDeleteFileEntry`.
fn rewrite_position_delete_file_metadata(
    delete_file: &DataFile,
    source: &str,
    target: &str,
) -> Result<DataFile> {
    let mut rewritten = rewrite_data_file_path(delete_file, source, target)?;
    replace_path_bounds(&mut rewritten, source, target)?;
    Ok(rewritten)
}

/// Java `ContentFileUtil.replacePathBounds`. It rewrites the file_path-column bound metrics only
/// when lower equals upper, which means one referenced data file. Otherwise it clears both bounds,
/// because a rewritten range would no longer bound the paths it covers.
fn replace_path_bounds(delete_file: &mut DataFile, source: &str, target: &str) -> Result<()> {
    let lower = delete_file
        .lower_bounds
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH);
    let upper = delete_file
        .upper_bounds
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH);

    let (Some(lower), Some(upper)) = (lower, upper) else {
        // Java returns metricsWithoutPathBounds when either bound is null.
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
            // The range spans several files, or the bounds are not strings.
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

/// Decodes a [`Datum`]'s string value. Returns `None` for a non-string.
fn datum_as_string(datum: &Datum) -> Option<String> {
    match datum.literal() {
        PrimitiveLiteral::String(value) => Some(value.clone()),
        _ => None,
    }
}

// ============================================================================================
// Identity-preserving manifest entry re-emission (Java `appendEntryWithFile`).
// ============================================================================================

/// Re-emits `entry` with `new_file`, dispatching on the original status like Java
/// `appendEntryWithFile`. Status, sequence numbers, and snapshot ids must survive the rewrite,
/// because only the path changed.
fn reemit_entry(
    writer: &mut crate::spec::ManifestWriter,
    entry: &ManifestEntry,
    new_file: DataFile,
) -> Result<()> {
    match entry.status() {
        ManifestStatus::Added => {
            // A live ADDED entry already carries its inherited sequence number. Preserve it.
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

/// Locates the `file_path` and `pos` columns of a position-delete batch by reserved field id, not by
/// name. [`crate::maintenance::RewritePositionDeleteFiles`] carries the same helper.
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

/// A [`LocationGenerator`](crate::writer::file_writer::location_generator::LocationGenerator) that
/// emits one fixed path and ignores the generated file name. The rewritten content must land at the
/// exact location the copy plan and the staged manifest name.
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
